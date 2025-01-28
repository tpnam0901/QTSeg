import torch
import os
import cv2
import glob
import argparse
import logging
import numpy as np
from tqdm.auto import tqdm
from configs.base import Config
from networks import models
from data.utils import load_img, preprocess, resize_longest_side, pad_to_square


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)


def main(cfg: Config, input_dir: str, output_dir: str, ckpt: str = ""):

    os.makedirs(output_dir, exist_ok=True)
    logging.info("Building model...")
    weight_paths = glob.glob(os.path.join(cfg.checkpoint_dir, "*.pt"))
    if ckpt:
        weight_paths = [ckpt]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = getattr(models, cfg.model_type)(cfg)
    model.eval()

    input_paths = glob.glob(os.path.join(input_dir, "*"))
    logging.info("Found {} images".format(len(input_paths)))
    logging.info(
        "Running inference with weights trained from {} dataset".format(cfg.data_root)
    )
    for weight_path in weight_paths:
        logging.info("Inferencing with {}".format(weight_path))
        model.to(torch.device("cpu"))
        weight = torch.load(weight_path, map_location="cpu")
        if "loss" in weight:
            weight = weight["state_dict_model"]
        model.load_state_dict(weight)
        model.to(device)
        model.eval()

        for path in tqdm(input_paths):
            x = load_img(path)
            x = resize_longest_side(x, cfg.img_size, interpolation=cv2.INTER_AREA)
            if len(x.shape) == 4:
                x = cv2.cvtColor(x, cv2.COLOR_RGBA2RGB)
            h, w = x.shape[:2]
            x = pad_to_square(x)
            if cfg.cvtColor is not None:
                x = cv2.cvtColor(x, cfg.cvtColor)
            x = preprocess(x)
            x = torch.from_numpy(x).to(device).unsqueeze(0)

            with torch.no_grad():
                outputs = model(x)

            prediction = outputs[0][0].detach().cpu().numpy()
            if prediction.shape[0] == 1:
                # sigmoid
                prediction = np.squeeze(prediction)
                if np.max(prediction) > 1 and np.min(prediction) < 0:
                    prediction = 1 / (1 + np.exp(-prediction))
                prediction = (prediction > 0.5).astype(np.float32)
                prediction *= 255.0
                prediction = prediction.astype(np.uint8)
            else:
                prediction = np.argmax(prediction, axis=0).astype(np.float32)
                prediction *= 255.0 / (cfg.num_classes - 1)
                prediction = prediction.astype(np.uint8)
            name = os.path.basename(path).split(".")[0]
            weight_name = os.path.basename(weight_path).split(".")[0]
            # Crop back to original size
            prediction = prediction[:h, :w]
            cv2.imwrite(
                os.path.join(output_dir, name + weight_name + ".png"), prediction
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
        "--input_dir",
        type=str,
        default="working/inputs",
        help="Whether to change the metric",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="working/outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Device to run inference",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    cfg = Config()
    cfg.load(args.config)
    main(cfg, args.input_dir, args.output_dir, args.ckpt)
