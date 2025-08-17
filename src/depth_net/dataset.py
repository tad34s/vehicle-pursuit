from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class MaskDataset(Dataset):
    def __init__(self, input_images_path: str, masks_path: str) -> None:
        self.input_images = {int(x.name[-5]): str(x) for x in Path(input_images_path).glob("*.png")}
        self.masks = {int(x.name[-5]): str(x) for x in Path(masks_path).glob("*.png")}
        if len(self.input_images) != len(self.masks):
            raise ValueError

    def __len__(self) -> int:
        return len(self.input_images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, mask = read_image(self.input_images[index]), read_image(self.masks[index])
        return img.type(torch.float32), mask.type(torch.float32)
