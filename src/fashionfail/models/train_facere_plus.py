import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms.v2 as transforms
from aim.pytorch_lightning import AimLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import tv_tensors

from fashionfail.features.dataloader import prepare_dataloader
from fashionfail.models.facere import facere_plus


def get_cli_args_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for pseudo-random generators. To ensure full reproducibility, set `deterministic` flag in `pl.Trainer`.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Name of the experiment to be shown on Aim.",
    )
    parser.add_argument(
        "--overfit-batches",
        type=int,
        default=0,
        help="Number of batches to overfit on.",
    )
    parser.add_argument(
        "--train_img_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train_ann",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--val_img_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--val_ann",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        required=True,
        help="Path to pretrained model to be initialized for fine-tuning.",
    )

    return parser


# Define the excluded indices
excluded_indices = {2, 12, 16, 19, 20}
# Create a DataFrame for the remaining categories
remaining_categories = list(set(range(27)) - excluded_indices)
df_mapping = pd.DataFrame(
    {"old_cat_id": remaining_categories, "new_cat_id": range(len(remaining_categories))}
)
# Create a dictionary that maps old IDs to new IDs
category_mapping = dict(zip(df_mapping["old_cat_id"], df_mapping["new_cat_id"]))

# Define data augmentations
initial_transform_train = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.RandomPhotometricDistort(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomZoomOut(
            fill={tv_tensors.Image: 255, tv_tensors.Mask: 0}, side_range=(1, 8), p=0.5
        ),
        transforms.ScaleJitter(
            scale_range=(0.2, 2.0), target_size=(2400, 2400), antialias=True
        ),
        transforms.SanitizeBoundingBoxes(min_size=10),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

initial_transform_val = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)


def common_transform(image, target):
    # Map the original id to new id
    old_id = target["labels"].item()
    new_id = category_mapping.get(old_id) + 1  # labels should start from 1, not 0.
    target["labels"] = torch.tensor([new_id], dtype=torch.int64)
    return image, target


def transform_train(image, target):
    image, target = initial_transform_train(image, target)
    image, target = common_transform(image, target)
    return image, target


def transform_val(image, target):
    image, target = initial_transform_val(image, target)
    image, target = common_transform(image, target)
    return image, target


def main(parser):
    # Add model-specific arguments
    parser = facere_plus.add_model_specific_args(parser)

    # Add trainer-related arguments
    parser = pl.Trainer.add_argparse_args(parser)

    # Parse arguments
    args = parser.parse_args()

    # Set seeds for numpy, torch and python.random
    pl.seed_everything(args.seed, workers=True)

    # Prepare data loaders
    train_loader = prepare_dataloader(
        img_dir=os.path.expanduser(args.train_img_dir),
        ann_path=os.path.expanduser(args.train_ann),
        transform=transform_train,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        n_workers=args.n_workers,
    )
    val_loader = prepare_dataloader(
        img_dir=os.path.expanduser(args.val_img_dir),
        ann_path=os.path.expanduser(args.val_ann),
        transform=transform_val,
    )

    # Prepare model.
    simple_model = facere_plus(
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Prepare logger.
    aim_logger = AimLogger(
        experiment=args.experiment_name,
        train_metric_prefix="train_",
        val_metric_prefix="val_",
    )

    # Prepare Callbacks
    cb_ckpt = ModelCheckpoint(
        dirpath="./saved_models/",
        save_top_k=1,
        monitor="val_loss_sum",
        filename=args.experiment_name + "-{epoch:02d}-{val_loss_sum:.2f}",
    )

    # Prepare trainer.
    trainer = pl.Trainer.from_argparse_args(
        args, logger=aim_logger, callbacks=[cb_ckpt]
    )

    # Fit model to data.
    trainer.fit(
        model=simple_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )


if __name__ == "__main__":
    cli_args = get_cli_args_parser()
    main(cli_args)
