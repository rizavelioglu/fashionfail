from typing import Callable

import torch
from torchvision import datasets

from fashionfail.features._dataset_wrapper import wrap_dataset_for_transforms_v2


def prepare_dataloader(
    img_dir: str,
    ann_path: str,
    transform: Callable,
    batch_size: int = 4,
    shuffle: bool = False,
    n_workers: int = 24,
):
    # Adapted from: https://pytorch.org/vision/0.15/auto_examples/plot_transforms_v2_e2e.html
    dataset = datasets.CocoDetection(
        root=img_dir,
        annFile=ann_path,
        transforms=transform,
    )
    # Convert target into required types
    dataset = wrap_dataset_for_transforms_v2(
        dataset, target_keys=("boxes", "labels", "masks")
    )

    # Construct dataloader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_workers,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )
    return data_loader
