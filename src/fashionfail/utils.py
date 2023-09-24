import json
from functools import lru_cache
from pathlib import Path

import torch
from loguru import logger
from pycocotools.coco import COCO
from torchvision.ops import box_convert

from fashionfail.models.cocoeval2 import COCOeval2


@lru_cache
def load_categories(
    return_raw_categories: bool = False,
):
    """
    Load Fashionpedia categories from a JSON file and return a dictionary mapping
    category IDs to their corresponding names.

    Args:
        return_raw_categories (bool, optional): If True, return the raw category
            dictionary. If False (default), return a dictionary where keys are
            category IDs (integers) and values are corresponding category names
            (strings).

    Returns:
        Union[Dict[int, str], List[Dict[str, Any]]]: A dictionary where keys are
        category IDs (integers) and values are corresponding category names (strings).
        If `return_raw_categories` is True, returns the raw category dictionary.

    Raises:
        FileNotFoundError: If the expected JSON file 'categories.json' is not found
        at the specified path.

    Example:
        >>> category_id_to_name = load_categories()
        >>> print(category_id_to_name[23])
        'Shoe'
    """
    # TODO: replace the absolute path. Download it when it's not found.
    expected_path = Path(
        "/home/rizavelioglu/work/repos/segmentation/segmentation/visualization/categories.json"
    )

    if not expected_path.exists():
        raise FileNotFoundError(f"`categories.json` expected at `{expected_path}`")

    # Load official Fashionpedia categories
    with open(expected_path) as fp:
        categories = json.load(fp)

    if return_raw_categories:
        return categories
    else:
        category_id_to_name = {d["id"]: d["name"] for d in categories}
        return category_id_to_name


def print_category_counts_from_coco(coco_annotation_file: str) -> None:
    """
    Print category counts from a COCO annotation file.

    Args:
        coco_annotation_file (str): Path to the COCO annotation file.

    Example:
        >>> print_category_counts_from_coco("path/to/coco/annotation.json")
    """
    # Load COCO annotation from file
    coco_ann = COCO(coco_annotation_file)
    # Load categories
    categories = list(load_categories().values())

    print("_" * 44)
    print(f"| {'id':<2} | {'cat':<24} | #samples |")  # header
    print(f"|{'-'*4}|{'-'*26}|{'-'*10}|")  # separator

    total = 0

    for i in range(0, len(categories)):
        anns = coco_ann.getAnnIds(catIds=[i])
        if anns:
            print(f"| {i:<2} | {categories[i][:24]:<24} | {len(anns):<8} |")
            total += len(anns)

    print(f"{'-'*44}")
    print(f"{'Total':<31} | {total:<8} |")

    # remove annotations from memory
    del coco_ann


def print_tp_fp_fn_counts(coco_eval, iou_idx=0, area_idx=0, max_dets_idx=2):
    """
    Print a summary of metrics; TP, FP, FN counts, based on COCO evaluation results.

    Args:
        coco_eval (COCOeval2): An instance of the custom `COCOeval2` class, which is used as an alternative
            implementation to calculate and evaluate metrics that are not provided by the official COCOeval class.
        iou_idx (int, optional): Index for IoU threshold in [0.50, 0.05, 0.95]. Default is 0.
        area_idx (int, optional): Index for area range in ['all', 'small', 'medium', 'large']. Default is 0.
        max_dets_idx (int, optional): Index for maximum detections in [1, 10, 100]. Default is 2.

    Example:
        >>> print_tp_fp_fn_counts(coco_eval)
    """

    if not isinstance(coco_eval, COCOeval2):
        logger.error(f"`coco_eval` object must be an object of {COCOeval2}!")
        return

    print(
        f"Metrics @[",
        f"IoU={coco_eval.params.iouThrs[iou_idx]} |",
        f"area={coco_eval.params.areaRngLbl[area_idx]} |",
        f"maxDets={coco_eval.params.maxDets[max_dets_idx]} ]",
    )

    print("_" * 30)
    print(f"| {'cat':<2} | {'TP':<5} | {'FP':<5} | {'FN':<5} |")  # header
    print(f"|{'-' * 5}|{'-' * 7}|{'-' * 7}|{'-' * 7}|")  # separator

    total_tp, total_fp, total_fn = 0, 0, 0

    for catId in range(0, 27):
        num_tp = int(coco_eval.eval["num_tp"][iou_idx, catId, area_idx, max_dets_idx])
        num_fp = int(coco_eval.eval["num_fp"][iou_idx, catId, area_idx, max_dets_idx])
        num_fn = int(coco_eval.eval["num_fn"][iou_idx, catId, area_idx, max_dets_idx])

        print(f"| {catId:<3} | {num_tp:<5} | {num_fp:<5} | {num_fn:<5} |")

        total_tp += num_tp
        total_fp += num_fp
        total_fn += num_fn

    print(f"{'-' * 30}")
    print(f"{'Total':<5} | {total_tp:<5} | {total_fp:<5} | {total_fn:<5} |")


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
