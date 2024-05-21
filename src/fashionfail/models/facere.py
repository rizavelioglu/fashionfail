import os.path
from typing import Any

import pytorch_lightning as pl
import torch
from loguru import logger
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

TRAIN_METRIC_PREFIX = "train_"
VAL_METRIC_PREFIX = "val_"


def _model_selector(pretrained_model: str):
    if pretrained_model == "maskrcnn_resnet50_fpn_v2":
        return maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
    elif pretrained_model == "facere_base_pretrained":
        expected_path = "fashionfail/saved_models/facere_base.ckpt"

        if os.path.exists(expected_path):
            return facere_base.load_from_checkpoint(expected_path).model

        logger.error(f"Error: `facere_base_pretrained` expected at :{expected_path}")
        return None


def construct_pretrained_model(num_classes, model_name: str):
    num_classes += 1  # add 1 for background class
    # Load an instance segmentation model pre-trained on COCO.
    model = _model_selector(pretrained_model=model_name)

    # Get number of input features for the classifier.
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get the number of input features for the mask classifier.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256  # TODO Replace magic number!

    # Replace the mask predictor with a new one.
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


class facere_base(pl.LightningModule):
    def __init__(
        self,
        *,
        optimizer_name: str,
        learning_rate: float,
        beta1: float,
        beta2: float,
        momentum: float,
        weight_decay: float,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.model = construct_pretrained_model(
            num_classes=46, model_name="maskrcnn_resnet50_fpn_v2"
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        child_parser = parent_parser.add_argument_group("Optimization")
        child_parser.add_argument(
            "--optimizer", type=str, choices=["sgd", "adam", "adamW"], default="sgd"
        )
        child_parser.add_argument("--learning-rate", type=float)
        child_parser.add_argument("--beta1", type=float)
        child_parser.add_argument("--beta2", type=float)
        child_parser.add_argument("--momentum", type=float)
        child_parser.add_argument("--weight-decay", type=float)
        return parent_parser

    def optimizer_parameters(self):
        if self.optimizer_name == "sgd":
            parameters = {
                "lr": self.learning_rate if self.learning_rate else 0.005,
                "momentum": self.momentum if self.momentum else 0.9,
                "weight_decay": self.weight_decay if self.weight_decay else 0.0005,
            }
        elif self.optimizer_name == "adam":
            parameters = {
                "lr": self.learning_rate if self.learning_rate else 1e-3,
                "betas": (
                    self.beta1 if self.beta1 else 0.9,
                    self.beta2 if self.beta2 else 0.95,
                ),
                "weight_decay": self.weight_decay if self.weight_decay else 0,
                "fused": True,
            }
        elif self.optimizer_name == "adamW":
            parameters = {
                "lr": self.learning_rate if self.learning_rate else 1e-3,
                "betas": (
                    self.beta1 if self.beta1 else 0.9,
                    self.beta2 if self.beta2 else 0.999,
                ),
                "weight_decay": self.weight_decay if self.weight_decay else 1e-2,
                "fused": True,
            }
        else:
            raise ValueError(f"Optimizer `{self.optimizer_name}` unknown")
        return parameters

    def configure_optimizers(self):
        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), **self.optimizer_parameters()
            )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), **self.optimizer_parameters()
            )
        elif self.optimizer_name == "adamW":
            optimizer = torch.optim.AdamW(
                self.parameters(), **self.optimizer_parameters()
            )
        else:
            raise ValueError(f"Cannot configure optimizer `{self.optimizer_name}`")
        return optimizer

    def forward(self, inputs: list[torch.Tensor]):
        # Only used for inference.
        preds = self.model(inputs)

        return preds

    def training_step(self, batch, batch_idx):
        # Prepare data.
        images = list(image for image in batch[0])
        targets = [{k: v for k, v in t.items()} for t in batch[1]]

        # Perform forward pass.
        loss_dict: dict[str, Any] = self.model(images, targets)

        # Log individual losses.
        for loss_key, loss_value in loss_dict.items():
            self.log(f"train_{loss_key}", loss_value, batch_size=len(batch))

        # Sum losses.
        loss_sum = sum(loss for loss in loss_dict.values())

        # Log summed loss.
        self.log(f"train_loss_sum", loss_sum.item(), batch_size=len(batch))

        return loss_sum

    def validation_step(self, batch, batch_idx):
        # Set the model to training mode. This is strange, of course, but necessary.
        # In evaluation mode, torchvision's current implementation does not return
        # losses, but predictions. And then, it is not straight-forward to calculate
        # the losses.
        self.model.train()

        # Prepare data.
        images = list(image for image in batch[0])
        targets = [{k: v for k, v in t.items()} for t in batch[1]]

        # Perform forward pass.
        loss_dict: dict[str, Any] = self.model(images, targets)

        # Log individual losses.
        for loss_key, loss_value in loss_dict.items():
            self.log(f"val_{loss_key}", loss_value, batch_size=len(batch))

        # Sum losses.
        loss_sum = sum(loss for loss in loss_dict.values())

        # Log summed loss.
        self.log(f"val_loss_sum", loss_sum.item(), batch_size=len(batch))

        return loss_sum


class facere_plus(pl.LightningModule):
    def __init__(
        self,
        *,
        optimizer_name: str,
        learning_rate: float,
        beta1: float,
        beta2: float,
        momentum: float,
        weight_decay: float,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.model = construct_pretrained_model(
            num_classes=22, model_name="facere_base_pretrained"
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        child_parser = parent_parser.add_argument_group("Optimization")
        child_parser.add_argument(
            "--optimizer", type=str, choices=["sgd", "adam"], default="sgd"
        )
        child_parser.add_argument("--learning-rate", type=float)
        child_parser.add_argument("--beta1", type=float)
        child_parser.add_argument("--beta2", type=float)
        child_parser.add_argument("--momentum", type=float)
        child_parser.add_argument("--weight-decay", type=float)
        return parent_parser

    def optimizer_parameters(self):
        if self.optimizer_name == "sgd":
            parameters = {
                "lr": self.learning_rate if self.learning_rate else 0.005,
                "momentum": self.momentum if self.momentum else 0.9,
                "weight_decay": self.weight_decay if self.weight_decay else 0.0005,
            }
        elif self.optimizer_name == "adam":
            parameters = {
                "lr": self.learning_rate if self.learning_rate else 1e-3,
                "betas": (
                    self.beta1 if self.beta1 else 0.9,
                    self.beta2 if self.beta2 else 0.95,
                ),
                "weight_decay": self.weight_decay if self.weight_decay else 0,
            }
        else:
            raise ValueError(f"Optimizer `{self.optimizer_name}` unknown")
        return parameters

    def configure_optimizers(self):
        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), **self.optimizer_parameters()
            )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), **self.optimizer_parameters()
            )
        else:
            raise ValueError(f"Cannot configure optimizer `{self.optimizer_name}`")
        return optimizer

    def forward(self, inputs: list[torch.Tensor]):
        # Only used for inference.
        preds = self.model(inputs)

        return preds

    def training_step(self, batch, batch_idx):
        # Prepare data.
        images = list(image for image in batch[0])
        targets = [{k: v for k, v in t.items()} for t in batch[1]]

        # Perform forward pass.
        loss_dict: dict[str, Any] = self.model(images, targets)

        # Log individual losses.
        for loss_key, loss_value in loss_dict.items():
            self.log(f"train_{loss_key}", loss_value, batch_size=len(batch))

        # Sum losses.
        loss_sum = sum(loss for loss in loss_dict.values())

        # Log summed loss.
        self.log(f"train_loss_sum", loss_sum.item(), batch_size=len(batch))

        return loss_sum

    def validation_step(self, batch, batch_idx):
        # Set the model to training mode. This is strange, of course, but necessary.
        # In evaluation mode, torchvision's current implementation does not return
        # losses, but predictions. And then, it is not straight-forward to calculate
        # the losses.
        self.model.train()

        # Prepare data.
        images = list(image for image in batch[0])
        targets = [{k: v for k, v in t.items()} for t in batch[1]]

        # Perform forward pass.
        loss_dict: dict[str, Any] = self.model(images, targets)

        # Log individual losses.
        for loss_key, loss_value in loss_dict.items():
            self.log(f"val_{loss_key}", loss_value, batch_size=len(batch))

        # Sum losses.
        loss_sum = sum(loss for loss in loss_dict.values())

        # Log summed loss.
        self.log(f"val_loss_sum", loss_sum.item(), batch_size=len(batch))

        return loss_sum


model_classes = {
    "facere_base": facere_base,
    "facere_plus": facere_plus,
}
