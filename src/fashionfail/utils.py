import json
from functools import lru_cache
from pathlib import Path

import torch
from torchvision.ops import box_convert


@lru_cache
def load_categories() -> dict:
    """
    Load Fashionpedia categories from a JSON file and return a dictionary mapping
    category IDs to their corresponding names.

    Returns:
        dict: A dictionary where keys are category IDs (integers) and values are
        corresponding category names (strings).

    Raises:
        FileNotFoundError: If the expected JSON file 'categories.json' is not found
        at the specified path.

    Example:
        >>> category_id_to_name = load_categories()
        >>> print(category_id_to_name[23])
        'Shoe'
    """
    expected_path = Path(
        "/home/rizavelioglu/work/repos/segmentation/segmentation/visualization/categories.json"
    )

    if expected_path.exists():
        # Parse Fashionpedia categories.
        with open(expected_path) as fp:
            categories = json.load(fp)
        category_id_to_name = {d["id"]: d["name"] for d in categories}
        return category_id_to_name

    raise FileNotFoundError(f"`categories.json` expected at `{expected_path}`")


def _box_yxyx_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (y1, x1, y2, x2) format to (x1, y1, x2, y2) format.

    Args:
        boxes (torch.Tensor): A tensor of bounding boxes in the (y1, x1, y2, x2) format.

    Returns:
        torch.Tensor: A tensor of bounding boxes in the (x1, y1, x2, y2) format.
    """
    y1, x1, y2, x2 = boxes.unbind(-1)
    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return boxes


# Adapted from: https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py#L168
def extended_box_convert(
    boxes: torch.Tensor, in_fmt: str, out_fmt: str
) -> torch.Tensor:
    """
    Converts boxes from given in_fmt to out_fmt.
    Supported in_fmt and out_fmt are:

    'xyxy': boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right.
    This is the format that torchvision utilities expect.

    'xywh' : boxes are represented via corner, width and height, x1, y2 being top left, w, h being width and height.

    'cxcywh' : boxes are represented via centre, width and height, cx, cy being center of box, w, h
    being width and height.

    'yxyx': boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right.
    This is the format that `amrcnn` model outputs.

    Args:
        boxes (Tensor[N, 4]): boxes which will be converted.
        in_fmt (str): Input format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh', 'yxyx'].
        out_fmt (str): Output format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh']

    Returns:
        Tensor[N, 4]: Boxes into converted format.
    """

    if in_fmt == "yxyx":
        # convert to xyxy and change in_fmt xyxy
        boxes = _box_yxyx_to_xyxy(boxes)
        in_fmt = "xyxy"
        if out_fmt == "xyxy":
            boxes = boxes
        elif out_fmt == "xywh":
            boxes = box_convert(boxes, in_fmt=in_fmt, out_fmt="xywh")
        elif out_fmt == "cxcywh":
            boxes = box_convert(boxes, in_fmt=in_fmt, out_fmt="cxcywh")
    else:
        boxes = box_convert(boxes, in_fmt=in_fmt, out_fmt=out_fmt)

    return boxes


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)

        if x.shape[0] == 0:
            continue

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes
