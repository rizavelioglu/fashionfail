import os
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from pycocotools import mask as mask_api
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from fashionfail.utils import extended_box_convert, load_categories


# Helper Functions
def show(imgs, return_fig=False):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    # Return figure when requested
    if return_fig:
        return fig


def visualize_mask_predictions(
    predictions,
    img_folder="~/.cache/fashionfail/",
    top_k_masks=5,
    score_threshold=0.2,
):
    """
    Draw segmentation predictions (masks) individually alongside the raw image.

    Args:
        predictions (np.array): An array of dictionaries representing the predictions.
        img_folder (str): Input directory containing the raw images.
        top_k_masks (int): Top 'k' masks (based on confidence score) to show image.
        score_threshold (float): Filter out boxes with prediction scores below this threshold.

    Returns:
        fig (matplotlib.figure.Figure): The figure showing the raw image as well as the top-k most confident mask predictions.
    """

    fig, axs = plt.subplots(
        nrows=len(predictions), ncols=top_k_masks + 1, figsize=(18, 10)
    )
    fig.tight_layout()
    category_id_to_name = load_categories()

    for row, pred in enumerate(predictions):
        # Parse predictions
        labels_id = pred["classes"][pred["scores"] > score_threshold].tolist()
        labels = [category_id_to_name[int(i) - 1] for i in labels_id]
        scores = pred["scores"][pred["scores"] > score_threshold].tolist()

        # Masks require individual process
        encoded_masks = pred["masks"]
        decoded_masks = mask_api.decode(encoded_masks)
        masks = decoded_masks.transpose(2, 0, 1)  # from HWN to NHW
        masks = torch.tensor(masks.astype(bool))
        masks = masks[pred["scores"] > score_threshold]
        masks = masks.type(torch.uint8)

        # Plot raw image on the first column (index 0)
        axs[row, 0].imshow(mpimg.imread(os.path.join(img_folder, pred["image_file"])))
        axs[row, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[row, 0].set_title(f"{pred['image_file']}", fontsize=9)

        # Plot each mask individually
        for col, (mask, label, score) in enumerate(
            zip(masks[:top_k_masks], labels[:top_k_masks], scores[:top_k_masks]), 1
        ):
            mask = F.to_pil_image(mask)

            axs[row, col].imshow(np.asarray(mask))
            axs[row, col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs[row, col].set_title(f"{label}: {score:.2f}", fontsize=9)

        # clear empty places
        if masks.shape[0] < top_k_masks:
            for c in range(masks.shape[0] + 1, top_k_masks + 1):
                axs[row, c].remove()

    return fig


def visualize_bbox_predictions(
    predictions,
    model_type="amrcnn",
    img_folder="~/.cache/fashionfail/",
    score_threshold=0.2,
    bbox_font_size=25,
    bbox_width=3,
    n_row=10,
    n_col=10,
    dpi=600,
    figsize=(9, 9),
    out_path=None,
):
    """
    Draw object detection predictions(bounding boxes) on raw images.

    Args:
        predictions (np.array): An array of dictionaries representing the predictions.
        model_type (str): Either 'amrcnn' or 'fformer' which are models whose predictions require individual processing.
        img_folder (str): Input directory containing the raw images.
        score_threshold (float): Filter out boxes with prediction scores below this threshold.
        bbox_font_size (int): Font size for the bounding box labels.
        bbox_width (int): Width of the bounding box lines.
        n_row (int): Number of rows in the visualization grid.
        n_col (int): Number of columns in the visualization grid.
        dpi (int): DPI (dots per inch) for the output figure.
        out_path (str): Full path of the output figure file.
        figsize (tuple(int)): Figure size in inches.

    Returns:
        fig (matplotlib.figure.Figure): The generated matplotlib figure.
    """

    fig = plt.figure(dpi=dpi, figsize=figsize)
    category_id_to_name = load_categories()

    for i, pred in enumerate(predictions):
        image = read_image(os.path.join(img_folder, pred["image_file"]))
        boxes = pred["boxes"][pred["scores"] > score_threshold]
        scores = pred["scores"][pred["scores"] > score_threshold]
        labels = [
            (
                category_id_to_name[cat_id - 1]
                if model_type == "amrcnn"
                else category_id_to_name[cat_id]
            )
            for cat_id in pred["classes"][pred["scores"] > score_threshold].tolist()
        ]
        # Convert boxes to "xyxy" format
        in_fmt = "yxyx" if model_type == "amrcnn" else "xyxy"
        boxes = extended_box_convert(torch.tensor(boxes), in_fmt=in_fmt, out_fmt="xyxy")

        # Sort the labels and scores by scores in descending order
        sorted_idx = np.argsort(scores)[::-1]
        sorted_labels = [labels[i] for i in sorted_idx]
        sorted_scores = scores[sorted_idx]

        # Draw bounding boxes on the image
        img_with_boxes = draw_bounding_boxes(
            image=image,
            boxes=boxes,
            labels=labels,
            width=bbox_width,
            font="Ubuntu-B.ttf",
            font_size=bbox_font_size,
            colors=["red"] * boxes.shape[0],
        )

        # Create the subplot
        ax = fig.add_subplot(n_row, n_col, i + 1, xticks=[], yticks=[])
        img = F.to_pil_image(img_with_boxes)
        ax.imshow(np.asarray(img))
        ax.set_title(
            f"{Path(pred['image_file']).name}", fontdict={"fontsize": 2}, pad=1
        )

        # Create the legend
        legend_str = [
            f"{label[:10]}: {score:.2f}"
            for label, score in zip(sorted_labels, sorted_scores)
        ]
        ax.text(
            x=0.05,
            y=0.95,
            s="\n".join(legend_str),
            fontsize=1,
            horizontalalignment="left",
            verticalalignment="top",
            color="black",
            bbox={"facecolor": "red", "alpha": 0.7, "pad": 0.7, "edgecolor": "none"},
            transform=ax.transAxes,
        )

    if out_path:
        plt.savefig(f"{out_path}", dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return fig


def visualize_predictions(
    predictions,
    model_type="amrcnn",
    img_folder="~/.cache/fashionfail/",
    score_threshold=0.2,
    bbox_width=3,
    n_row=10,
    n_col=10,
    dpi=600,
    figsize=(9, 9),
    legend_fontsize=1.5,
    out_path=None,
):
    """
    Draw both segmentation masks and bounding boxes on raw images.

    Args:
        predictions (np.array): An array of dictionaries representing the predictions.
        model_type (str): Either 'amrcnn' or 'fformer' which are models whose predictions require individual processing.
        img_folder (str): Input directory containing the raw images.
        score_threshold (float): Filter out boxes with prediction scores below this threshold.
        bbox_width (int): Width of the bounding box lines.
        n_row (int): Number of rows in the visualization grid.
        n_col (int): Number of columns in the visualization grid.
        dpi (int): DPI (dots per inch) for the output figure.
        figsize (tuple(int)): Figure size in inches.
        legend_fontsize (float/int): Font size of text inside the legend.
        out_path (str): Full path of the output figure file.

    Returns:
        fig (matplotlib.figure.Figure): The generated matplotlib figure.
    """

    fig = plt.figure(dpi=dpi, figsize=figsize)
    category_id_to_name = load_categories()

    for i, pred in enumerate(predictions):
        image = read_image(os.path.join(img_folder, pred["image_file"]))
        boxes = pred["boxes"][pred["scores"] > score_threshold]
        scores = pred["scores"][pred["scores"] > score_threshold]
        labels = [
            (
                category_id_to_name[cat_id - 1]
                if model_type == "amrcnn"
                else category_id_to_name[cat_id]
            )
            for cat_id in pred["classes"][pred["scores"] > score_threshold].tolist()
        ]
        # Convert boxes to "xyxy" format
        in_fmt = "yxyx" if model_type == "amrcnn" else "xyxy"
        boxes = extended_box_convert(torch.tensor(boxes), in_fmt=in_fmt, out_fmt="xyxy")

        # Sort the labels and scores by scores in descending order
        sorted_idx = np.argsort(scores)[::-1]
        sorted_labels = [labels[i] for i in sorted_idx]
        sorted_scores = scores[sorted_idx]

        # Draw masks on the image (skip if there is no mask)
        masks = np.array(pred["masks"])
        masks = masks[pred["scores"] > score_threshold]
        if len(masks) == 0:
            img_with_masks = image
        else:
            decoded_masks = mask_api.decode(masks.tolist())
            decoded_masks = decoded_masks.transpose(2, 0, 1)  # from HWN to NHW
            decoded_masks = torch.tensor(decoded_masks.astype(bool))
            img_with_masks = draw_segmentation_masks(image, decoded_masks, alpha=0.8)

        # Draw bounding boxes on the image
        img_with_boxes = draw_bounding_boxes(
            image=img_with_masks,
            boxes=boxes,
            width=bbox_width,
            font="Ubuntu-B.ttf",
            # colors=["red"] * boxes.shape[0],
        )

        # Create the subplot
        ax = fig.add_subplot(n_row, n_col, i + 1, xticks=[], yticks=[])
        img = F.to_pil_image(img_with_boxes)
        ax.imshow(np.asarray(img))
        plt.subplots_adjust(
            hspace=0.05
        )  # the amount of height reserved for white space between subplots

        # Create the legend
        legend_str = [
            f"{label[:10]}: {score:.2f}"
            for label, score in zip(sorted_labels, sorted_scores)
        ]
        ax.text(
            x=0.05,
            y=0.95,
            s="\n".join(legend_str),
            fontsize=legend_fontsize,
            horizontalalignment="left",
            verticalalignment="top",
            color="black",
            bbox={"facecolor": "red", "alpha": 0.7, "pad": 0.7, "edgecolor": "none"},
            transform=ax.transAxes,
        )

    if out_path:
        plt.savefig(f"{out_path}", dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return fig
