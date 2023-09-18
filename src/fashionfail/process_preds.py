import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from pycocotools.coco import COCO

from fashionfail.utils import extended_box_convert

bbox_conversion_formats = {"amrcnn": ("yxyx", "xywh"), "fformer": ("xyxy", "xywh")}
"""
A dictionary that maps model names to bounding box (bbox) conversion formats.

Each key represents a model name, and the corresponding value is a tuple of two strings.
The first string in the tuple represents the input format (in_fmt), and the second string
represents the output format (out_fmt) for bounding box conversions.

Supported Models and Formats:
- "amrcnn": Input format is "yxyx" (top-left and bottom-right corners),
            and output format is "xywh" (x, y, width, height).
- "fformer": Input format is "xyxy" (top-left and bottom-right corners),
            and output format is "xywh" (x, y, width, height).

Usage:
You can use this dictionary to determine the appropriate input and output formats
for bounding box conversions based on the model name.

Example:
model_name = "amrcnn"
in_fmt, out_fmt = bbox_formats.get(model_name, (None, None))
if in_fmt and out_fmt:
    # Use in_fmt and out_fmt for bbox conversions.
else:
    # Handle the case where the model name is not found in the dictionary.
"""


def load_tpu_preds(path_to_preds: str, preprocess: bool = True) -> pd.DataFrame:
    """
    Load predictions from a file and optionally apply preprocessing.

    Predictions must have the following attributes:
    {
        "image_file": full name of the image,
        "boxes": list of bounding box coordinates,
        "classes": list of class id's,
        "scores": list of floats,
        "masks": list of binary values,
    }

    Args:
        path_to_preds (str): Path to the predictions file.
        preprocess (bool, optional): Whether to apply basic preprocessing. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing the loaded predictions.
    """
    # Load predictions to a DataFrame
    preds = np.load(path_to_preds, allow_pickle=True)
    if Path(path_to_preds).suffix == ".npz":
        df_preds = pd.DataFrame.from_records(preds["data"])
    else:
        df_preds = pd.DataFrame.from_records(preds)
    logger.info(f"Predictions loaded from: {path_to_preds}")

    # Apply basic preprocessing on request
    if preprocess:
        df_preds = clean_df_preds(df_preds)

    return df_preds


def _filter_preds_for_classes(row):
    """
    Filter prediction attributes in the dataframe based on class IDs.

    This function filters classes, scores, boxes, and masks attributes in the predictions
    dataframe based on the 'class ID'. Class IDs that are larger or equal to 28, such as
    "sleeve", "neckline", etc., belonging to super-categories "garment parts", "closures",
    and "decorations", are filtered out.

    Args:
        row (pd.Series): A Pandas Series representing a row of predictions with attributes
            'classes', 'scores', 'boxes', and 'masks'.

    Returns:
        tuple: A tuple containing filtered arrays for 'classes', 'scores', 'boxes', and 'masks'.
    """

    filtered_classes = np.array(
        [class_id for class_id in row["classes"] if 1 <= class_id <= 28]
    )
    filtered_scores = np.array(
        [
            score
            for i, score in enumerate(row["scores"])
            if row["classes"][i] in filtered_classes
        ]
    )
    filtered_boxes = np.array(
        [
            box
            for i, box in enumerate(row["boxes"])
            if row["classes"][i] in filtered_classes
        ]
    )
    filtered_masks = [
        mask
        for i, mask in enumerate(row["masks"])
        if row["classes"][i] in filtered_classes
    ]
    return filtered_classes, filtered_scores, filtered_boxes, filtered_masks


def clean_df_preds(df_preds: pd.DataFrame) -> pd.DataFrame:
    logger.info("Processing predictions dataframe...")
    # Preprocess dataframe
    df_preds = df_preds[["image_file", "classes", "scores", "boxes", "masks"]]
    df_preds = df_preds.reset_index(drop=True)
    # Check if image_file attributes contain a "/"
    has_full_path = df_preds["image_file"].str.contains("/")
    # Apply the split operation only to rows with full path
    df_preds.loc[has_full_path, "image_file"] = df_preds.loc[
        has_full_path, "image_file"
    ].apply(lambda x: x.split("/")[1])

    # Logging
    nb_of_samples = df_preds.shape[0]
    nb_of_samples_w_no_preds = df_preds[
        df_preds["classes"].apply(lambda x: x.size == 0)
    ].shape[0]
    logger.debug(
        f"Number of samples: {nb_of_samples}, of which {nb_of_samples_w_no_preds} "
        f"(%{nb_of_samples_w_no_preds / nb_of_samples * 100:.1f}) have no predictions!"
    )
    logger.info("Filtering out predictions made for categoryID >= 28...")

    # Apply the filtering function to the dataframe and update the predictions
    df_preds["classes"], df_preds["scores"], df_preds["boxes"], df_preds["masks"] = zip(
        *df_preds.apply(_filter_preds_for_classes, axis=1)
    )

    # Logging
    nb_of_samples_w_no_preds_after_filter = (
        df_preds[df_preds["classes"].apply(lambda x: x.size == 0)].shape[0]
        - nb_of_samples_w_no_preds
    )
    logger.debug(
        f"Number of samples with no predictions after filtering categories: {nb_of_samples_w_no_preds_after_filter}."
    )

    return df_preds


def convert_preds_to_coco(preds_path: str, anns_path: str, model_name: str) -> str:
    logger.info("Converting raw predictions to COCO format...")
    # The path to the resulting .json file
    output_json_file = preds_path.replace(Path(preds_path).suffix, "-coco.json")

    if Path(output_json_file).exists():
        logger.warning(
            f"Predictions are already converted to COCO format! To re-convert, remove the file at: {output_json_file}."
        )
        return output_json_file

    # Load predictions
    df_preds = load_tpu_preds(preds_path, preprocess=False)
    # Remove samples with no predictions, otherwise following processing fails
    df_preds = df_preds[df_preds["boxes"].apply(lambda box: box.size != 0)]

    # Explode predictions for each sample
    df_exploded = df_preds.explode(["classes", "scores", "boxes", "masks"])

    # Rename and reorder columns
    df_exploded = df_exploded.rename(
        columns={"classes": "class", "boxes": "bbox", "scores": "score"}
    )

    # Apply the extended_box_convert function to each box in the "bbox" column
    in_fmt, out_fmt = bbox_conversion_formats.get(model_name, (None, None))
    if in_fmt is None or out_fmt is None:
        raise ValueError(f"Unsupported model_name: {model_name}")
    df_exploded["bbox"] = df_exploded["bbox"].apply(
        lambda bbox: extended_box_convert(
            torch.tensor(bbox), in_fmt=in_fmt, out_fmt=out_fmt
        ).numpy()
    )

    # Sort predictions by score
    df_exploded = df_exploded.sort_values(["image_file", "score"], ascending=False)

    # Load GT annotations (required for setting `image_id`)
    coco = COCO(anns_path)
    # Parse GT-images
    coco_imgs = pd.DataFrame(coco.imgs.values())

    # Convert the DataFrame to a list of dictionaries in the desired format
    json_data = []
    for _, row in df_exploded.iterrows():
        try:
            image_id = coco_imgs[
                coco_imgs.file_name == row["image_file"].split("/")[1]
            ].id.values[0]
        except IndexError:
            try:
                image_id = coco_imgs[
                    coco_imgs.file_name == row["image_file"]
                ].id.values[0]
            except Exception as e:
                # Handle the exception here, e.g., log an error message or take appropriate action
                print(f"An error occurred: {str(e)}")
                break

        json_data.append(
            {
                "image_id": int(image_id),
                # Subtract 1 to match official annotation in `amrcnn`
                "category_id": int(row["class"]) - 1
                if model_name == "amrcnn"
                else int(row["class"]),
                # Round coordinates to the nearest tenth of a pixel
                "bbox": [round(float(coord), 1) for coord in row["bbox"]],
                "score": float(row["score"]),
            }
        )

    # Save the list of dictionaries as a JSON file
    with open(output_json_file, "w") as f:
        json.dump(json_data, f, indent=4)

    logger.info(f"The resulting file is saved at: {output_json_file}.")
    return output_json_file
