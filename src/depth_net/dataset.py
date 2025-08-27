import copy
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch._dynamo.utils import istype
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

    def __getitem__(self, id: int):
        if istype(id, torch.Tensor):
            id = id.item()
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        id = self.ids[index]
        img = read_image(self.input_images[id])
        t_ref = np.load(self.t_refs[id])
        if self.transform:
            img = self.transform(img)
        t_ref = torch.tensor(t_ref, dtype=torch.float32)
        return img.type(torch.float32), t_ref


class ActiveLearningDataset(Dataset):
    def __init__(self, train_dataset: MaskDataset, dataset_dict: dict[int, torch.Tensor]) -> None:
        self.input_images = train_dataset.input_images
        self.transform = train_dataset.transform
        self.dataset_dict = dataset_dict
        self.ids = list(self.dataset_dict.keys())
        print(dataset_dict)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        id = self.ids[index]
        y = self.dataset_dict[id]
        img = read_image(self.input_images[id])
        if self.transform:
            img = self.transform(img)
        return img, y


class OverSampler(Sampler):
    def __init__(
        self,
        dataset: MaskDataset,
        losses: dict[int, float] | None = None,
        batch_size=64,
        nbins=16,
        from_each=4,
        drop_last=False,
    ):
        self.batch_size = batch_size
        self.data = dataset
        if losses is not None:
            self.bins = self.create_histogram(losses, nbins)
            for x in self.bins:
                np.random.shuffle(x)
        else:
            self.bins = None

        self.nbins = nbins
        self.drop_last = drop_last
        self.from_each = from_each
        self.num_batches = len(self.data) // self.batch_size
        if not drop_last and len(self.data) % self.batch_size != 0:
            self.num_batches += 1

    @staticmethod
    def create_histogram(losses: dict[int, float], nbins):
        values = list(losses.values())
        min_val = np.min(values)
        max_val = np.max(values)

        bin_step = (max_val - min_val) / nbins

        histogram = [[] for _ in range(nbins)]

        bin_edges = [min_val + i * bin_step for i in range(nbins + 1)]

        for key, value in losses.items():
            if value == max_val:  # Handle the edge case of max value
                bin_index = nbins - 1
            else:
                bin_index = int((value - min_val) / bin_step)

            histogram[bin_index].append(key)

        return histogram

    @staticmethod
    def bin_iter(bin):
        i = 0
        while True:
            yield bin[i]
            i = (i + 1) % len(bin)

    def __iter__(self):
        random_ids = copy.copy(self.data.ids)
        np.random.shuffle(random_ids)
        all_iter = iter(x for x in random_ids)

        if self.bins is None:
            batch = []
            for id in random_ids:
                batch.append(id)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        else:
            bin_iters = [self.bin_iter(x) for x in self.bins if x]

            for _ in range(self.num_batches):
                batch = []
                for _ in range(self.from_each):
                    new_data = [next(x) for x in bin_iters]
                    batch.extend(new_data)

                to_add = self.batch_size - len(batch)
                new_data = [next(all_iter) for _ in range(to_add)]
                batch += new_data

                yield batch[: self.batch_size]

    def __len__(self):
        return self.num_batches
