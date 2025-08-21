from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image


class MaskDataset(Dataset):
    def __init__(
        self,
        input_images_path: str,
        masks_path: str,
        ids: list[int],
        device,
        resized_image_size=None,
        flip=False,
    ) -> None:
        self.input_images = {
            int(x.name[:-4]): str(x) for x in Path(input_images_path).glob("*.png")
        }
        self.masks = {int(x.name[:-4]): str(x) for x in Path(masks_path).glob("*.png")}
        self.ids = sorted(ids)
        if len(self.input_images) != len(self.masks):
            raise ValueError

        if resized_image_size is not None:
            self.transform = torchvision.transforms.Resize(resized_image_size, antialias=True)
        else:
            self.transform = None
        self.device = device

        self.flip = flip
        self.flip_prob = 0.5

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        id = self.ids[index]
        img, mask = (
            read_image(self.input_images[id]),
            read_image(self.masks[id]),
        )
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        if self.flip:
            if torch.rand(1) < self.flip_prob:
                img = torch.flip(img, [-1])
                mask = torch.flip(mask, [-1])

                self._save_flipped_image(img, f"debug_flip/flipped_image_{id}.png")
                self._flip_saved = True

        return img.type(torch.float32), mask.type(torch.float32)

        return img.to(self.device), mask.to(self.device)

    def _save_flipped_image(self, image_tensor: torch.Tensor, filepath: str):
        """Save a flipped image for verification"""
        # Convert tensor to PIL Image and save
        pil_image = F.to_pil_image(image_tensor.cpu())
        pil_image.save(filepath)
        print(f"Flipped image saved to: {filepath}")


class TestDataset(Dataset):
    def __init__(
        self,
        input_images_path: str,
        t_ref_path: str,
        ids: list[int],
        device,
        resized_image_size=None,
    ) -> None:
        self.input_images = {
            int(x.name[:-4]): str(x) for x in Path(input_images_path).glob("*.png")
        }
        self.t_refs = {int(x.name[:-4]): str(x) for x in Path(t_ref_path).glob("*.npy")}
        self.ids = sorted(ids)
        if len(self.input_images) != len(self.t_refs):
            raise ValueError

        if resized_image_size is not None:
            self.transform = torchvision.transforms.Resize(resized_image_size, antialias=True)
        else:
            self.transform = None
        self.device = device

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        id = self.ids[index]
        img = read_image(self.input_images[id])
        t_ref = np.load(self.t_refs[id])
        if self.transform:
            img = self.transform(img)
        t_ref = torch.tensor(t_ref, dtype=torch.float32)
        return img.type(torch.float32), t_ref
