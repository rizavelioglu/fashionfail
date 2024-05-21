import torch
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


def _crop_image_given_bbox(
    image: tv_tensors.Image, target: dict, x_min, y_min, x_max, y_max, offset=50
):
    # Calculate crop and offset dimensions
    crop_left = max(0, x_min - offset)
    crop_top = max(0, y_min - offset)
    crop_right = min(image.shape[-1], x_max + offset)
    crop_bottom = min(image.shape[-2], y_max + offset)

    # Crop the image and convert back to `tv_tensors`
    image = transforms.functional.crop(
        image,
        top=crop_top,
        left=crop_left,
        height=crop_bottom - crop_top,
        width=crop_right - crop_left,
    )
    image = tv_tensors.Image(image)

    # Adjust the bounding box coordinates
    target["boxes"][:, 0] -= crop_left
    target["boxes"][:, 1] -= crop_top
    target["boxes"][:, 2] -= crop_left
    target["boxes"][:, 3] -= crop_top
    target["boxes"].canvas_size = tuple(F.get_size(image))

    # Adjust the masks and convert it back to its original dtype: tv_tensors
    masks = target["masks"][:, crop_top:crop_bottom, crop_left:crop_right]
    target["masks"] = tv_tensors.wrap(masks, like=target["masks"])

    return image, target


def crop_single_bbox_from_ff(image, target, offset=50):
    """
    Randomly pick a single annotation from FashionFail categories and crop the image using its bbox.

    Note:
        Be very careful about category id's. When training MaskRCNN, the 0th class is preserved for background class.
        Hence, the original class labels are incremented by one, e.g.`target["labels"] += 1`. Make sure that statement
        is applied after this transformation function is executed.
    """
    # Define FashionFail categories
    excluded_indices = {2, 12, 16, 19, 20}
    ff_categories = list(set(range(27)) - excluded_indices)

    inds = sum(target["labels"] == i for i in ff_categories).bool()

    if inds.any():
        # Randomly select a single box
        selected_index = torch.randint(0, torch.sum(inds).item(), (1,))
        selected_box = target["boxes"][inds][selected_index][0]

        # Extract box coordinates
        x1, y1, x2, y2 = map(int, selected_box)

        # Crop `image` and update `target`
        image, target = _crop_image_given_bbox(
            image, target, x_min=x1, y_min=y1, x_max=x2, y_max=y2, offset=offset
        )
        # Update target to include only the selected object's annotations
        target["masks"] = tv_tensors.wrap(
            target["masks"][inds][selected_index], like=target["masks"]
        )
        target["boxes"] = tv_tensors.wrap(
            target["boxes"][inds][selected_index], like=target["boxes"]
        )
        target["labels"] = target["labels"][inds][selected_index]

    return image, target


def crop_largest_bbox(image, target, offset=50):
    # Find the bounding box enclosing all other bounding boxes
    x_min = int(torch.min(target["boxes"][:, 0]).item())
    y_min = int(torch.min(target["boxes"][:, 1]).item())
    x_max = int(torch.max(target["boxes"][:, 2]).item())
    y_max = int(torch.max(target["boxes"][:, 3]).item())

    # Crop `image` and adjust `target` accordingly
    image, target = _crop_image_given_bbox(
        image, target, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, offset=offset
    )

    return image, target


def erase_background(image, target):
    masks = target["masks"]
    combined_mask = torch.any(masks, dim=0)
    image[:, ~combined_mask.bool()] = (
        1.0  # Setting the background to 1.0 (white) for all channels
    )

    return image, target
