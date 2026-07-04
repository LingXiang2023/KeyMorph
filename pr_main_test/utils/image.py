import math

import cv2
import numpy as np
import torch


def quantization(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(torch.clamp(tensor * 255, min=0.0, max=255.0)) / 255


def calculate_rmse(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return float(np.sqrt(mse))


def calculate_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mae = np.mean(np.abs(img1 - img2))
    if mae == 0:
        return float("inf")
    return float(mae)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * math.log10(255.0 / math.sqrt(mse)))


def _ssim_single_channel(img1: np.ndarray, img2: np.ndarray) -> float:
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean())


def _as_hwc(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[0] in (1, 3):
        return img.transpose((1, 2, 0))
    return img


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = _as_hwc(img1)
    img2 = _as_hwc(img2)
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return _ssim_single_channel(img1, img2)
    if img1.ndim == 3 and img1.shape[2] in (1, 3):
        return float(np.mean([_ssim_single_channel(img1[:, :, i], img2[:, :, i]) for i in range(img1.shape[2])]))
    raise ValueError("Expected image shape HxW, HxWxC, or CxHxW.")
