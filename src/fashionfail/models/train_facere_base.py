import os
import random

import pytorch_lightning as pl
import torch
import torchvision.transforms.v2 as transforms
from aim.pytorch_lightning import AimLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from fashionfail.features.dataloader import prepare_dataloader
from fashionfail.features.transforms import crop_largest_bbox
from fashionfail.models.facere import facere_base


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

    return parser


# Define data augmentations
initial_transform_train = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.RandomPhotometricDistort(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.SanitizeBoundingBoxes(min_size=10),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)
final_transform_train = transforms.Compose(
    [
        transforms.ScaleJitter(
            scale_range=(0.1, 2), target_size=(2400, 2400), antialias=True
        ),
    ]
)

initial_transform_val = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.SanitizeBoundingBoxes(
            min_size=10
        ),  # required as val has invalid boxes
        transforms.ToDtype(torch.float32, scale=True),
    ]
)


def common_transform(image, target):
    # Labels should start from 1 (0 is preserved for background)
    target["labels"] += 1
    return image, target


def transform_train(image, target):
    image, target = initial_transform_train(image, target)
    if random.random() > 0.5:
        image, target = crop_largest_bbox(image, target)
    image, target = final_transform_train(image, target)
    image, target = common_transform(image, target)
    return image, target


def transform_val(image, target):
    image, target = initial_transform_val(image, target)
    image, target = common_transform(image, target)
    return image, target


def main(parser):
    # Add model-specific arguments
    parser = facere_base.add_model_specific_args(parser)

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
    simple_model = facere_base(
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
        args,
        logger=aim_logger,
        callbacks=[cb_ckpt],
    )

    # Fit model to data.
    trainer.fit(
        model=simple_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )


if __name__ == "__main__":
    cli_args = get_cli_args_parser()
    main(cli_args)
