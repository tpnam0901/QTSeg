import warnings

# Ignore all warnings
warnings.simplefilter("ignore")

import os
import random
import numpy as np
import torch

SEED = 1996
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

import argparse
import logging
import mlflow
import datetime

from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from data.dataloader import build_dataloader
from configs.base import Config, import_config
from networks import models, losses, metrics, optimizers, schedulers


def main(cfg: Config, val_prefetch: bool):
    # ----------------- Setup -----------------
    cfg.checkpoint_dir = os.path.join(cfg.checkpoint_dir, cfg.dataloader)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow_dir = os.path.join(cfg.checkpoint_dir, "mlruns")
    os.makedirs(mlflow_dir, exist_ok=True)
    mlflow_run_name = current_time + "-" + cfg.name
    mlflow_exp_name = cfg.name

    cfg.checkpoint_dir = os.path.join(cfg.checkpoint_dir, cfg.name, current_time)

    # Log, weight, mlflow folder
    log_dir = os.path.join(cfg.checkpoint_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    ## Add logger to log folder
    logging.getLogger().setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger = logging.getLogger(cfg.name)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())

    ## Add mlflow to log folder
    mlflow.set_tracking_uri(uri=f"file://{os.path.abspath(mlflow_dir)}")
    ## Set mlflow name
    mlflow.set_experiment(mlflow_exp_name)

    # Preparing checkpoint output
    weight_last_path = os.path.join(cfg.checkpoint_dir, "weight_last.pt")

    # Save configs
    logger.info("Saving config to {}".format(cfg.checkpoint_dir))
    cfg.save(cfg.checkpoint_dir)
    cfg.show()

    # ----------------- Prepare training -----------------
    # Build dataloader
    logger.info("Building dataset...")
    train_dataloader = build_dataloader(
        cfg,
        mode="train",
        logger=logger,
    )

    test_dataloader = build_dataloader(
        cfg,
        mode=cfg.valid_type,
        logger=logger,
        batch_size=1,
    )

    logger.info("Building model, loss and optimizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    model = getattr(models, cfg.model_type)(cfg)
    model.to(torch.device("cpu"))
    if cfg.model_pretrained:
        model.load_state_dict(
            torch.load(
                cfg.model_pretrained,
                map_location="cpu",
            ),
        )
    model.to(device)

    logger.info(
        "Number of total parameters: {}".format(
            sum(p.numel() for p in model.parameters())
        )
    )

    criterion = losses.Losses(cfg)
    loss_params = criterion.get_params()
    criterion.to(device)

    metric = metrics.Metrics(cfg)
    metric.to(device)

    train_params = []
    for key, value in loss_params.items():
        num_params = sum(p.numel() for p in value)
        logger.info("Number of parameters in loss {}: {}".format(key, num_params))
        if num_params > 0:
            train_params.append({"params": value})
    train_params.append({"params": model.parameters()})
    if len(train_params) == 1:
        train_params = model.parameters()

    optimizer = getattr(optimizers, cfg.optimizer)(train_params, cfg)

    lr_scheduler = getattr(schedulers, cfg.scheduler)(optimizer, cfg)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    best_values = {}

    global_train_step = 0
    global_val_step = 0
    num_steps = len(train_dataloader)
    classes = [str(i) for i in range(cfg.num_masks)]

    if val_prefetch:
        logger.info("Prefetching validation data...")
        test_dataloader = list(test_dataloader)  # prefetch validation data
        logger.info("Prefetching {} data".format(len(test_dataloader)))

    # ----------------- Training -----------------
    logger.info("Start training...")
    with mlflow.start_run(run_name=mlflow_run_name):
        # Log all configs to mlflow
        for key, value in cfg.get_params().items():
            mlflow.log_param(key, value)
        epoch = 0
        while epoch < cfg.epochs:
            if epoch < cfg.transfer_epochs:
                if cfg.encoder_pretrained:
                    model.freeze_encoder()
                if cfg.decoder_pretrained:
                    model.freeze_decoder()
                if cfg.model_pretrained:
                    model.freeze_model()
                logger.info(
                    "Number of trainable parameters: {}".format(
                        sum(p.numel() for p in model.parameters() if p.requires_grad)
                    )
                )
            else:
                model.unfreeze_encoder()
                model.unfreeze_decoder()
                model.unfreeze_model()
                logger.info(
                    "Number of trainable parameters: {}".format(
                        sum(p.numel() for p in model.parameters() if p.requires_grad)
                    )
                )

            total_loss_train = []
            logger.info("Train epoch {}/{}".format(epoch + 1, cfg.epochs))

            metric.reset()
            model.train()

            with tqdm(total=num_steps, ascii=True) as pbar:
                for step, (inputs, targets, _) in enumerate(train_dataloader):
                    global_train_step += 1

                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    with torch.autocast(
                        device_type=device_str, dtype=torch.float16, enabled=cfg.use_amp
                    ):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        metric_dict = metric(outputs, targets)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    if cfg.scheduler == "IdentityScheduler":
                        lr_ = (
                            cfg.learning_rate
                            * (1.0 - global_train_step / (num_steps * cfg.epochs))
                            ** 0.9
                        )
                        optimizer.param_groups[0]["lr"] = lr_

                    inputs = inputs.detach().cpu().numpy()
                    targets = targets.detach().cpu().numpy()
                    loss = loss.detach().cpu().numpy()

                    total_loss_train.append(loss.item())

                    if step % cfg.log_freq == 0:
                        mlflow.log_metric("loss", loss.item(), step=global_train_step)

                    postfix = "Epoch {}/{} - loss: {:.4f}".format(
                        epoch + 1, cfg.epochs, loss.item()
                    )

                    for key, value in metric_dict.items():
                        cls = key.split("_")[-1]
                        if cls not in classes:
                            if step % cfg.log_freq == 0:
                                mlflow.log_metric(key, value, step=global_train_step)
                            postfix += " - {}: {:.4f}".format(key, value)

                    pbar.set_description(postfix)
                    pbar.update(1)

                    if global_train_step % cfg.ckpt_save_fred == 0:
                        checkpoint = {
                            "epoch": epoch,
                            "epoch_current_step": step,
                            "global_train_step": global_train_step,
                            "global_val_step": global_val_step,
                            "state_dict_model": model.state_dict(),
                            "state_dict_optim_model": optimizer.state_dict(),
                            "state_dict_scheduler_model": lr_scheduler.state_dict(),
                        }
                        checkpoint.update(best_values)
                        torch.save(checkpoint, weight_last_path)

            log_info = "Epoch {}/{} - epoch_loss: {:.4f}".format(
                epoch + 1,
                cfg.epochs,
                np.mean(total_loss_train).item(),
            )
            for key, value in metric.compute().items():
                cls = key.split("_")[-1]
                if cls not in classes:
                    log_info += " - {}: {:.4f}".format(key, value)
            logger.info(log_info)

            # Limit the learning rate
            if (
                optimizer.param_groups[0]["lr"] > cfg.learning_rate_min
                and cfg.scheduler != "IdentityScheduler"
            ):
                lr_scheduler.step()
                if optimizer.param_groups[0]["lr"] < cfg.learning_rate_min:
                    optimizer.param_groups[0]["lr"] = cfg.learning_rate_min
            mlflow.log_metric(
                "learning_rate",
                optimizer.param_groups[0]["lr"],
                step=epoch + 1,
            )

            checkpoint = {
                "epoch": epoch,
                "epoch_current_step": 0,
                "global_train_step": global_train_step,
                "global_val_step": global_val_step,
                "state_dict_model": model.state_dict(),
                "state_dict_optim_model": optimizer.state_dict(),
                "state_dict_scheduler_model": lr_scheduler.state_dict(),
            }
            checkpoint.update(best_values)
            torch.save(checkpoint, weight_last_path)

            # ----------------- Validation -----------------
            if (epoch + 1) % cfg.val_epoch_freq == 0:
                logger.info("Validation epoch {}/{}".format(epoch + 1, cfg.epochs))
                total_loss_val = []
                model.eval()
                metric.reset()
                global_val_step += 1
                random_viz = random.randint(0, len(test_dataloader) - 1)
                for index, (inputs, targets, inputs_raw) in enumerate(
                    tqdm(test_dataloader)
                ):
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    with torch.no_grad():
                        with torch.autocast(
                            device_type=device_str,
                            dtype=torch.float16,
                            enabled=cfg.use_amp,
                        ):
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            metric_dict = metric(outputs, targets)

                        inputs = inputs.detach().cpu().numpy()
                        targets = targets.detach().cpu().numpy()
                        loss = loss.detach().cpu().numpy()
                    inputs_raw = inputs_raw.detach().cpu().numpy()
                    total_loss_val.append(loss.item())

                    if index == random_viz:
                        # Log image to mlflow
                        ax = plt.subplot(1, 2 + cfg.num_masks, 1)

                        image = inputs_raw[0].astype(np.uint8)
                        target = targets[0].astype(np.float32)
                        prediction = outputs[0][0].detach().cpu().numpy()
                        ax.imshow(image)
                        ax.axis("off")
                        ax.set_title("Input")
                        labels = np.unique(target)
                        target *= 255 / np.max(labels)
                        target = target.astype(np.uint8)
                        ax = plt.subplot(1, 2 + cfg.num_masks, 2)
                        ax.imshow(target, cmap="gray")
                        ax.axis("off")
                        ax.set_title("Target")
                        if prediction.shape[0] == 1:
                            # sigmoid
                            prediction = np.squeeze(prediction)
                            prediction = 1 / (1 + np.exp(-prediction))
                            prediction = (prediction > 0.5).astype(np.float32)
                            prediction *= 255.0 / np.max(labels)
                            prediction = prediction.astype(np.uint8)
                            ax = plt.subplot(1, 2 + cfg.num_masks, 3)
                            ax.imshow(prediction, cmap="gray")
                            ax.axis("off")
                            ax.set_title("Prediction")
                        else:
                            prediction = np.argmax(prediction, axis=0).astype(np.uint8)
                            for i in range(cfg.num_masks):
                                ax = plt.subplot(1, 2 + cfg.num_masks, 3 + i)
                                mask = (prediction == i) * 255
                                ax.imshow(mask, cmap="gray")
                                ax.axis("off")
                                ax.set_title("Predict class {}".format(i))

                        plt.tight_layout()
                        mlflow.log_figure(
                            plt.gcf(),
                            "validation_{}.png".format(str(global_val_step).zfill(4)),
                        )

                total_loss_val = np.mean(total_loss_val).item()
                mlflow.log_metric(
                    "val_loss", float(total_loss_val), step=global_val_step
                )
                metric_dict = metric.compute()
                log_metric_dict = {}

                for key, value in metric_dict.items():
                    mlflow.log_metric("val_" + key, value, step=global_val_step)
                    cls = key.split("_")[-1]
                    if cls not in classes:
                        log_metric_dict[key] = value

                log_info = "Epoch {}/{} - val_loss: {:.4f}".format(
                    epoch + 1, cfg.epochs, total_loss_val
                )
                for key, value in metric_dict.items():
                    log_info += " - {}: {:.4f}".format(key, value)

                logger.info(log_info)
                if total_loss_val < best_values.get("loss", float(np.inf)):
                    logger.info(
                        "Loss improved from {:.4f} to {:.4f}".format(
                            best_values.get("loss", float(np.inf)), total_loss_val
                        )
                    )
                    best_values["loss"] = total_loss_val
                    torch.save(
                        model.state_dict(),
                        os.path.join(cfg.checkpoint_dir, "weight_best_loss.pt"),
                    )

                for key, value in log_metric_dict.items():
                    if key.startswith("mae"):
                        best_value = best_values.get(key, float(np.inf))
                    else:
                        best_value = best_values.get(key, -float(np.inf))

                    found_new_best = (
                        value < best_value
                        if key.startswith("mae")
                        else value > best_value
                    )

                    if found_new_best:
                        logger.info(
                            "Metric {} improved from {:.4f} to {:.4f}".format(
                                key, best_value, value
                            )
                        )
                        best_values[key] = value
                        torch.save(
                            model.state_dict(),
                            os.path.join(
                                cfg.checkpoint_dir, "weight_best_{}.pt".format(key)
                            ),
                        )
            # Update epoch
            epoch += cfg.val_epoch_freq

    log_info = "Best validation |"
    for key, value in best_values.items():
        log_info += " - {}: {:.4f}".format(key, value)
    logger.info(log_info)
    end_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info("Training finished at {}".format(end_time))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg",
        "--config",
        type=str,
        default="../src/configs/base.py",
        help="Path to config.py file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to resume cfg.log file if want to resume training",
    )
    parser.add_argument(
        "--val_prefetch",
        action="store_true",
        help="Prefetch validation data",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    cfg: Config = import_config(args.config)

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(message)s",
    )
    main(cfg, args.val_prefetch)
