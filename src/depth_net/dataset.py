from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, Sampler
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
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

        return (
            img.type(torch.float32),
            mask.type(torch.float32),
            id,
        )


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

    def __getitem__(self, id: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        img = read_image(self.input_images[id])
        t_ref = np.load(self.t_refs[id])
        if self.transform:
            img = self.transform(img)
        t_ref = torch.tensor(t_ref, dtype=torch.float32)
        return img.type(torch.float32), t_ref, id


class OverSampler(Sampler):
    def __init__(
        self, dataset: MaskDataset, losses: dict[int, float] | None = None, batch_size=64, nbins=16
    ):
        self.batch_size = batch_size
        self.data = dataset
        if losses is None:
            sorted_indices = np.random.shuffle(dataset.ids)
        else:
            sorted_indices = sorted(losses.keys(), key=lambda x: losses[x])
        self.bins = [np.random.shuffle(x) for x in np.array_split(sorted_indices, nbins)]
        self.nbins = nbins

    @staticmethod
    def bin_iter(bin):
        for i in bin:
            yield i

    def __iter__(self):
        bin_iters = [self.bin_iter(x) for x in self.bins]

        batch = []
        for i in range(0, len(self.data), self.nbins):
            new_data = [next(x) for x in bin_iters]
            batch += new_data

            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def __len__(self):
        return len(self.data)
