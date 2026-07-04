from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def preconfigure_cuda_visible_devices() -> None:
    for index, arg in enumerate(sys.argv):
        if arg == "--cuda-devices" and index + 1 < len(sys.argv):
            value = sys.argv[index + 1]
            break
        if arg.startswith("--cuda-devices="):
            value = arg.split("=", 1)[1]
            break
    else:
        value = os.environ.get("HIDING_CUDA_DEVICES")

    if value:
        os.environ["CUDA_VISIBLE_DEVICES"] = value


preconfigure_cuda_visible_devices()

import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

import config as c
from models.network import HidingNetwork, print_activate_params
from utils.dataset import load_test_dataset
from utils.image import calculate_mae, calculate_psnr, calculate_rmse, calculate_ssim

IMAGE_GROUPS = (
    "cover",
    "secret",
    "stego",
    "secret_rev",
    "cover_resi",
    "secret_resi",
    "noisy",
    "denoised",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test-only image hiding evaluation.")
    parser.add_argument("--data-dir", default=c.test_data_dir, help="Directory of test images.")
    parser.add_argument("--checkpoint", default=c.checkpoint_path, help="Model checkpoint path.")
    parser.add_argument("--output-dir", default=c.output_dir, help="Directory for logs, metrics, and images.")
    parser.add_argument("--device", default=c.device, help="Device, for example auto, cpu, or cuda:0.")
    parser.add_argument("--cuda-devices", default=c.cuda_device_ids, help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--batch-size", type=int, default=c.batch_size_test, help="Even test batch size.")
    parser.add_argument("--resize", type=int, default=c.resize_size_test, help="Square resize used for testing.")
    parser.add_argument("--sigma", type=float, default=c.noise_sigma, help="Gaussian noise sigma in pixel scale.")
    parser.add_argument("--num-workers", type=int, default=c.num_workers_test, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=c.seed, help="Random seed for deterministic noise.")
    parser.add_argument("--resi-magnification", type=float, default=c.resi_magnification, help="Residual image scale.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional smoke-test limit.")

    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument("--save-images", dest="save_images", action="store_true", help="Save processed images.")
    image_group.add_argument("--no-save-images", dest="save_images", action="store_false", help="Skip image saving.")
    parser.set_defaults(save_images=c.save_processed_img)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return device


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "test.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d : %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")],
    )
    return logging.getLogger("hiding_eval")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")

    normalized = {}
    for key, value in state_dict.items():
        normalized[key[7:] if key.startswith("module.") else key] = value

    model.load_state_dict(normalized, strict=True)


def tensor_to_numpy_255(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().clamp(0.0, 1.0).cpu().numpy().astype(np.float32) * 255.0


def metric_lists() -> Dict[str, List[float]]:
    return {
        "stego_psnr": [],
        "secret_psnr": [],
        "noise_psnr": [],
        "denoise_psnr": [],
        "stego_ssim": [],
        "secret_ssim": [],
        "noise_ssim": [],
        "denoise_ssim": [],
        "stego_mae": [],
        "secret_mae": [],
        "noise_mae": [],
        "denoise_mae": [],
        "stego_rmse": [],
        "secret_rmse": [],
        "noise_rmse": [],
        "denoise_rmse": [],
    }


def update_metrics(
    metrics: Dict[str, List[float]],
    cover: torch.Tensor,
    secret: torch.Tensor,
    stego: torch.Tensor,
    secret_rev: torch.Tensor,
    noisy: torch.Tensor,
    denoised: torch.Tensor,
) -> None:
    arrays = {
        "cover": tensor_to_numpy_255(cover),
        "secret": tensor_to_numpy_255(secret),
        "stego": tensor_to_numpy_255(stego),
        "secret_rev": tensor_to_numpy_255(secret_rev),
        "noisy": tensor_to_numpy_255(noisy),
        "denoised": tensor_to_numpy_255(denoised),
    }

    for idx in range(cover.shape[0]):
        cover_i = arrays["cover"][idx]
        secret_i = arrays["secret"][idx]
        stego_i = arrays["stego"][idx]
        secret_rev_i = arrays["secret_rev"][idx]
        noisy_i = arrays["noisy"][idx]
        denoised_i = arrays["denoised"][idx]

        metrics["stego_psnr"].append(calculate_psnr(cover_i, stego_i))
        metrics["secret_psnr"].append(calculate_psnr(secret_i, secret_rev_i))
        metrics["noise_psnr"].append(calculate_psnr(secret_i, noisy_i))
        metrics["denoise_psnr"].append(calculate_psnr(secret_i, denoised_i))

        metrics["stego_ssim"].append(calculate_ssim(cover_i, stego_i))
        metrics["secret_ssim"].append(calculate_ssim(secret_i, secret_rev_i))
        metrics["noise_ssim"].append(calculate_ssim(secret_i, noisy_i))
        metrics["denoise_ssim"].append(calculate_ssim(secret_i, denoised_i))

        metrics["stego_mae"].append(calculate_mae(cover_i, stego_i))
        metrics["secret_mae"].append(calculate_mae(secret_i, secret_rev_i))
        metrics["noise_mae"].append(calculate_mae(secret_i, noisy_i))
        metrics["denoise_mae"].append(calculate_mae(secret_i, denoised_i))

        metrics["stego_rmse"].append(calculate_rmse(cover_i, stego_i))
        metrics["secret_rmse"].append(calculate_rmse(secret_i, secret_rev_i))
        metrics["noise_rmse"].append(calculate_rmse(secret_i, noisy_i))
        metrics["denoise_rmse"].append(calculate_rmse(secret_i, denoised_i))


def make_image_dirs(output_dir: Path, data_name: str) -> Dict[str, Path]:
    image_root = output_dir / data_name
    dirs = {name: image_root / name for name in IMAGE_GROUPS}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_batch_images(
    dirs: Dict[str, Path],
    start_index: int,
    cover: torch.Tensor,
    secret: torch.Tensor,
    stego: torch.Tensor,
    secret_rev: torch.Tensor,
    noisy: torch.Tensor,
    denoised: torch.Tensor,
    resi_magnification: float,
) -> None:
    cover_resi = (cover - stego).abs() * resi_magnification
    secret_resi = (secret - secret_rev).abs() * resi_magnification
    tensors = {
        "cover": cover,
        "secret": secret,
        "stego": stego,
        "secret_rev": secret_rev,
        "cover_resi": cover_resi,
        "secret_resi": secret_resi,
        "noisy": noisy,
        "denoised": denoised,
    }

    for local_idx in range(cover.shape[0]):
        image_name = f"{start_index + local_idx:06d}.png"
        for group, tensor in tensors.items():
            save_image(tensor[local_idx].detach().clamp(0.0, 1.0), dirs[group] / image_name)


def summarize(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    return {name: float(np.mean(values)) if values else float("nan") for name, values in metrics.items()}


def write_metrics(output_dir: Path, payload: Dict[str, object]) -> None:
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    metrics = payload["metrics"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for name, value in metrics.items():
            writer.writerow([name, value])


def log_summary(logger: logging.Logger, summary: Dict[str, float]) -> None:
    logger.info(
        "testing, stego_avg_psnr: %.2f, secret_avg_psnr: %.2f, noise_avg_psnr: %.2f, denoise_avg_psnr: %.2f",
        summary["stego_psnr"],
        summary["secret_psnr"],
        summary["noise_psnr"],
        summary["denoise_psnr"],
    )
    logger.info(
        "testing, stego_avg_ssim: %.4f, secret_avg_ssim: %.4f, noise_avg_ssim: %.4f, denoise_avg_ssim: %.4f",
        summary["stego_ssim"],
        summary["secret_ssim"],
        summary["noise_ssim"],
        summary["denoise_ssim"],
    )
    logger.info(
        "testing, stego_avg_mae: %.2f, secret_avg_mae: %.2f, noise_avg_mae: %.2f, denoise_avg_mae: %.2f",
        summary["stego_mae"],
        summary["secret_mae"],
        summary["noise_mae"],
        summary["denoise_mae"],
    )
    logger.info(
        "testing, stego_avg_rmse: %.2f, secret_avg_rmse: %.2f, noise_avg_rmse: %.2f, denoise_avg_rmse: %.2f",
        summary["stego_rmse"],
        summary["secret_rmse"],
        summary["noise_rmse"],
        summary["denoise_rmse"],
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 2 or args.batch_size % 2 != 0:
        raise ValueError("--batch-size must be an even number greater than or equal to 2.")
    if args.resize <= 0:
        raise ValueError("--resize must be positive.")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive when provided.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)
    set_seed(args.seed)
    device = select_device(args.device)

    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    logger.info("#" * 50)
    logger.info("model: hiding_network")
    logger.info("checkpoint: %s", checkpoint_path)
    logger.info("test data dir: %s", data_dir)
    logger.info("device: %s", device)
    logger.info("noise sigma: %s", args.sigma)

    model = HidingNetwork().to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()
    logger.info("activate params bytes: %s", print_activate_params(model))

    test_loader = load_test_dataset(
        data_dir=data_dir,
        batch_size=args.batch_size,
        resize_size=args.resize,
        sigma=args.sigma,
        num_workers=args.num_workers,
    )

    data_name = data_dir.name or "test_data"
    image_dirs = make_image_dirs(output_dir, data_name) if args.save_images else None
    metrics = metric_lists()
    pair_count = 0

    with torch.no_grad():
        stream: Iterable = tqdm(test_loader, desc="Testing")
        for batch_index, (data, noised_data) in enumerate(stream):
            data = data.to(device)
            noised_data = noised_data.to(device)

            half = data.shape[0] // 2
            cover = data[:half]
            secret = data[half:]
            noisy = noised_data[half:]

            denoised = model(noisy, None, "denoising")
            stego = model(cover, secret, "hiding")
            secret_rev = model(stego, None, "recover")

            update_metrics(metrics, cover, secret, stego, secret_rev, noisy, denoised)
            if image_dirs is not None:
                save_batch_images(
                    image_dirs,
                    pair_count,
                    cover,
                    secret,
                    stego,
                    secret_rev,
                    noisy,
                    denoised,
                    args.resi_magnification,
                )
            pair_count += half
            if args.max_batches is not None and batch_index + 1 >= args.max_batches:
                break

    summary = summarize(metrics)
    payload = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "resize": args.resize,
        "sigma": args.sigma,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "num_pairs": pair_count,
        "metrics": summary,
    }
    write_metrics(output_dir, payload)
    log_summary(logger, summary)
    logger.info("metrics saved to: %s", output_dir)


if __name__ == "__main__":
    main()
