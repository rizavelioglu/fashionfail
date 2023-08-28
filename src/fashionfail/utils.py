import json
from functools import lru_cache
from pathlib import Path

import torch


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


def yxyx_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (y1, x1, y2, x2) format to (x1, y1, x2, y2) format.

    Args:
        boxes (torch.Tensor): A tensor of bounding boxes in the (y1, x1, y2, x2) format.

    Returns:
        torch.Tensor: A tensor of bounding boxes in the (x1, y1, x2, y2) format.
    """
    y1, x1, y2, x2 = boxes.unbind(-1)
    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return boxes
