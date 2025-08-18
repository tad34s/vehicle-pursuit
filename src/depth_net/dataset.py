from pathlib import Path

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image


class MaskDataset(Dataset):
    def __init__(
        self, input_images_path: str, masks_path: str, device, resized_image_size=None
    ) -> None:
        self.input_images = {
            int(x.name[:-4]): str(x) for x in Path(input_images_path).glob("*.png")
        }
        self.masks = {int(x.name[:-4]): str(x) for x in Path(masks_path).glob("*.png")}
        self.ids = sorted(list(self.input_images.keys()))
        if len(self.input_images) != len(self.masks):
            raise ValueError

        if resized_image_size is not None:
            self.transform = torchvision.transforms.Resize(resized_image_size)
        else:
            self.transform = None
        self.device = device

    def __len__(self) -> int:
        return len(self.input_images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        id = self.ids[index]
        img, mask = (
            read_image(self.input_images[id]),
            read_image(self.masks[id]),
        )
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        return img.type(torch.float32), mask.type(torch.float32)
