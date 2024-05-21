import json
from functools import lru_cache
from pathlib import Path

import torch
from pycocotools.coco import COCO
from torchvision.ops import box_convert


@lru_cache
def load_categories(
    return_raw_categories: bool = False,
):
    r"""
    Load Fashionpedia categories from a JSON file.

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
        FileNotFoundError: If the expected JSON file 'instances_attributes_val2020.json' is not found
        at the specified path.

    Notes:
        Fashionpedia officially consists of 46 categories: :math:`catId \in [0,45]`, but we will
        limit ourselves to the top 27 categories in FashionFail: :math:`catId \in [0,26]`. Therefore,
        this function will only return those categories that we consider to be primary fashion apparel.
        It means that any category belonging to the "garment parts," "closures," and "decorations"
        super-categories are filtered out, such as "sleeve," "neckline," "pocket," and so on.

    Example:
        >>> category_id_to_name = load_categories()
        >>> print(category_id_to_name.get(23))
        shoe

    Example:
        >>> raw_categories = load_categories(return_raw_categories=True)
        >>> print(raw_categories[:2])
        [{'id': 0, 'name': 'shirt, blouse', 'supercategory': 'upperbody'},
        {'id': 1, 'name': 'top, t-shirt, sweatshirt', 'supercategory': 'upperbody'}]
    """
    expected_path = Path(
        "~/.cache/fashionpedia/instances_attributes_val2020.json"
    ).expanduser()

    if not expected_path.exists():
        raise FileNotFoundError(
            f"Fashionpedia annotations are expected to be at: `{expected_path}`. Please execute"
            f"`/fashionfail/data/make_fashionpedia.py` to download the dataset to the expected "
            f"location."
        )

    # Load top-27 Fashionpedia categories
    with open(expected_path) as fp:
        categories = json.load(fp)["categories"]
        categories = categories[:27]

    # Remove unnecessary keys from each dictionary
    keys_to_remove = ["level", "taxonomy_id"]
    for c in categories:
        for key in keys_to_remove:
            c.pop(key, None)

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
    coco_ann = COCO(coco_annotation_file)
    categories = load_categories()
    num_samples = 0

    print("_" * 44)
    print(f"| {'id':<2} | {'cat':<24} | #samples |")  # header
    print(f"|{'-'*4}|{'-'*26}|{'-'*10}|")  # separator

    for id, cat in categories.items():
        anns = coco_ann.getAnnIds(catIds=[id])
        if anns:
            print(f"| {id:<2} | {cat[:24]:<24} | {len(anns):<8} |")
            num_samples += len(anns)

    print(f"{'-'*44}")
    print(f"{'Total':<31} | {num_samples:<8} |")

    # remove annotations from memory
    del coco_ann


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


def _box_xyxy_to_yxyx(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (x1, y1, x2, y2) format to (y1, x1, y2, x2) format.

    Args:
        boxes (torch.Tensor): A tensor of bounding boxes in the (x1, y1, x2, y2) format.

    Returns:
        torch.Tensor: A tensor of bounding boxes in the (y1, x1, y2, x2) format.
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    boxes = torch.stack((y1, x1, y2, x2), dim=-1)
    return boxes


# Adapted from: https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py#L168
def extended_box_convert(
    boxes: torch.Tensor, in_fmt: str, out_fmt: str
) -> torch.Tensor:
    """
    Converts boxes from given in_fmt to out_fmt.

    Supported in_fmt and out_fmt are:
        - 'xyxy': boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right. This is the format that torchvision utilities expect.
        - 'xywh' : boxes are represented via corner, width and height, x1, y2 being top left, w, h being width and height.
        - 'cxcywh' : boxes are represented via centre, width and height, cx, cy being center of box, w, h being width and height.
        - 'yxyx': boxes are represented via corners, y1, x1 being top left and y2, x2 being bottom right. This is the format that `amrcnn` model outputs.

    Args:
        boxes (Tensor[N, 4]): boxes which will be converted.
        in_fmt (str): Input format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh', 'yxyx'].
        out_fmt (str): Output format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh', 'yxyx'].

    Returns:
        Tensor[N, 4]: Boxes into converted format.
    """

    if in_fmt == "yxyx":
        # Convert to xyxy and assign in_fmt accordingly
        boxes = _box_yxyx_to_xyxy(boxes)
        in_fmt = "xyxy"

    if out_fmt == "yxyx":
        # Convert to xyxy if not already in that format
        if in_fmt != "xyxy":
            boxes = box_convert(boxes, in_fmt=in_fmt, out_fmt="xyxy")
        # Convert to yxyx
        boxes = _box_xyxy_to_yxyx(boxes)
    else:
        # Use torchvision's box_convert for other conversions
        boxes = box_convert(boxes, in_fmt=in_fmt, out_fmt=out_fmt)

    return boxes
