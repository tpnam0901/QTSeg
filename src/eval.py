import torch
import os
import glob
import argparse
import logging

from time import time_ns
from tqdm.auto import tqdm
from data.dataloader import build_test_dataloader, build_dataloader
from configs.base import Config
from networks import models, metrics
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)


def main(cfg: Config, ckpt: str = "", ori_res: bool = False):
    # Change the batch size to 1
    cfg.num_workers = 0
    cfg.batch_size = 1

    weight_paths = glob.glob(os.path.join(cfg.checkpoint_dir, "*.pt"))
    if ckpt:
        weight_paths = [ckpt]
    # Build dataloader
    logging.info("Building dataset...")

    dataloader_fn = build_test_dataloader if ori_res else build_dataloader
    test_dataloader = dataloader_fn(
        cfg,
        mode=cfg.valid_type,
        logger=logging.getLogger("Eval"),
    )

    logging.info("Building model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = getattr(models, cfg.model_type)(cfg)
    model.to(device)

    metric = metrics.Metrics(cfg)
    metric.to(device)

    model.eval()
    metric.reset()

    inputs = torch.randn(1, cfg.image_channel, cfg.img_size, cfg.img_size).to(device)
    flops = FlopCountAnalysis(model, (inputs,)).total()
    acts = ActivationCountAnalysis(model, (inputs,)).total()
    total_flops = (flops + acts) / 1e9
    params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info("Model FLOPs: {}G".format(total_flops))
    logging.info("Model Params: {}M".format(params))

    inference_time = []
    model = getattr(models, cfg.model_type)(cfg)
    model.to(device)
    model.eval()

    for weight_path in weight_paths:
        logging.info("Inferencing with {}".format(weight_path))
        model.to(torch.device("cpu"))
        weight = torch.load(weight_path, map_location="cpu")
        if "best" not in weight_path:
            weight = weight["state_dict_model"]
        model.load_state_dict(weight)
        model.to(device)
        model.eval()
        metric.reset()

        process_start = time_ns()
        for inputs, targets, _ in tqdm(iter(test_dataloader)):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                outputs = model(inputs)

                metric(outputs, targets)

                inputs = inputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()
        process_end = time_ns()
        # Convert to ms
        cost_time = (process_end - process_start) / len(test_dataloader) / 1e6
        inference_time.append(cost_time)
        metric_dict = metric.compute()

        log_info = "Evaluated"
        for key, value in metric_dict.items():
            log_info += " - {}: {:.4f}".format(key, value)
        logging.info(log_info)

    logging.info(
        "Average inference time: {:.4f} ms".format(
            sum(inference_time) / len(inference_time)
        )
    )


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg",
        "--config",
        type=str,
        default="working/checkpoints/cfg.log",
        help="Path to cfg.log file",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="",
        help="Whether to change the metric",
    )
    parser.add_argument(
        "--valid_type",
        type=str,
        default="",
        help="Whether to change the metric",
    )
    parser.add_argument(
        "--ori_res",
        action="store_true",
        help="Whether to evaluate the original resolution",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Device to run inference",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    cfg = Config()
    cfg.load(args.config)
    if args.metric:
        cfg.metric = args.metric

    if args.valid_type:
        cfg.valid_type = args.valid_type
    main(cfg, args.ckpt, args.ori_res)
