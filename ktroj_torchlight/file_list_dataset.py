from typing import Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset


class FileListDataset(Dataset):
    """Dataset that loads images from a list of file paths."""

    def __init__(self, image_paths: Sequence[str], transform: Optional[torch.nn.Module] = None) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path
