from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ImagePath = Union[str, Path]
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


class TestImageDataset(Dataset):
    def __init__(self, img_dir: ImagePath, resize_size: int, sigma: Optional[float]) -> None:
        self.img_dir = Path(img_dir)
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Test image directory does not exist: {self.img_dir}")

        self.img_paths: Sequence[Path] = sorted(
            path for path in self.img_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.img_paths:
            raise FileNotFoundError(f"No test images found in: {self.img_dir}")

        self.transform = T.Compose([T.Resize([resize_size, resize_size]), T.ToTensor()])
        self.sigma = sigma

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int):
        img = Image.open(self.img_paths[index]).convert("RGB")
        img = self.transform(img)
        if self.sigma is None:
            return img, img.clone()
        noised_img = img + torch.randn_like(img).mul_(float(self.sigma) / 255.0)
        return img, noised_img


def load_test_dataset(
    data_dir: ImagePath,
    batch_size: int,
    resize_size: int,
    sigma: Optional[float],
    num_workers: int,
) -> DataLoader:
    dataset = TestImageDataset(data_dir, resize_size, sigma)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        drop_last=True,
    )
