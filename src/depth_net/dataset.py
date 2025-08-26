import copy
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
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

    def get_by_id(self, id: int):
        img, mask = (
            read_image(self.input_images[id]),
            read_image(self.masks[id]),
        )
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return (
            img.type(torch.float32),
            mask.type(torch.float32),
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
        self.train_datset = train_dataset
        self.dataset_dict = dataset_dict
        self.ids = list(self.dataset_dict.keys())

    def __len__(self) -> int:
        return len(self.dataset_dict.keys())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        id = self.ids[index]
        gt = self.dataset_dict[id]
        x, _ = self.train_datset.get_by_id(id)
        print(x, gt)
        return x, gt


class OverSampler(Sampler):
    def __init__(
        self,
        dataset: MaskDataset,
        losses: dict[int, float] | None = None,
        batch_size=64,
        nbins=16,
        drop_last=False,
    ):
        self.batch_size = batch_size
        self.data = dataset
        if losses is None:
            sorted_indices = copy.copy(dataset.ids)
            np.random.shuffle(sorted_indices)
            self.bins = np.array_split(sorted_indices, nbins)
        else:
            self.bins = self.create_histogram(losses, nbins)

        for x in self.bins:
            np.random.shuffle(x)
        self.nbins = nbins
        self.drop_last = drop_last
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
        bin_iters = [self.bin_iter(x) for x in self.bins]
        for _ in range(self.num_batches):
            batch = []
            for _ in range(self.batch_size // self.nbins):
                new_data = [next(x) for x in bin_iters]
                batch.extend(new_data)
            if len(batch) < self.batch_size and not self.drop_last:
                while len(batch) < self.batch_size:
                    for it in bin_iters:
                        if len(batch) >= self.batch_size:
                            break
                        batch.append(next(it))
            yield batch[: self.batch_size]  # Ensure exact batch size

    def __len__(self):
        return self.num_batches


class ImprovedOverSampler:
    def __init__(
        self,
        dataset,
        losses: dict[int, float] | None = None,
        batch_size=64,
        focus_factor=2.0,
        min_weight=0.1,
    ):
        self.batch_size = batch_size
        self.dataset = dataset
        self.focus_factor = focus_factor  # Controls how much to focus on high-loss samples
        self.min_weight = min_weight  # Minimum weight to ensure all samples have some chance

        if losses is None:
            # Uniform sampling if no losses provided
            self.weights = torch.ones(len(dataset))
            self.sampler = WeightedRandomSampler(self.weights, len(dataset), replacement=True)
        else:
            # Create weights based on losses with specialized scaling for [0,1] range
            self.weights = self._create_weights_from_losses(losses)
            self.sampler = WeightedRandomSampler(self.weights, len(self.weights), replacement=True)

    def _create_weights_from_losses(self, losses):
        """Create sampling weights from losses with specialized scaling for [0,1] range"""
        # Get losses for all samples in the dataset
        loss_values = []
        for i, id in enumerate(self.dataset.ids):
            if id in losses:
                loss_values.append(losses[id])
            else:
                # Use median loss if not found (shouldn't happen in practice)
                median_loss = np.median(list(losses.values()))
                loss_values.append(median_loss)

        # Convert to tensor
        loss_tensor = torch.tensor(loss_values, dtype=torch.float32)

        # Specialized scaling for [0,1] range
        # We use a power function to emphasize high-loss samples
        # The focus_factor controls how much we focus on high-loss samples
        weights = (loss_tensor + self.min_weight) ** self.focus_factor

        # Normalize to avoid extreme values
        weights = weights / weights.mean()

        return weights

    def get_sampler(self):
        return self.sampler

    def update_parameters(self, focus_factor=None, min_weight=None):
        """Update sampling parameters dynamically"""
        if focus_factor is not None:
            self.focus_factor = focus_factor
        if min_weight is not None:
            self.min_weight = min_weight

    def update_weights(self, losses):
        self.weights = self._create_weights_from_losses(losses)
        self.sampler = WeightedRandomSampler(self.weights, len(self.weights), replacement=True)
