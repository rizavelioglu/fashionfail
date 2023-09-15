import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from pycocotools.coco import COCO
from torchvision.ops import box_convert

from fashionfail.utils import yxyx_to_xyxy


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


def convert_tpu_preds_to_coco(preds_path: str, anns_path: str) -> str:
    logger.info("Converting AMRCNN predictions to COCO format...")
    # The path to the resulting .json file
    output_json_file = preds_path.replace(".npy", "-coco.json")

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

    # Apply the box_convert function to each box in the "bbox" column
    df_exploded["bbox"] = df_exploded["bbox"].apply(
        lambda bbox: box_convert(
            yxyx_to_xyxy(torch.tensor(bbox)), in_fmt="xyxy", out_fmt="xywh"
        ).numpy()
    )

    # Round coordinates to the nearest tenth of a pixel
    df_exploded["bbox"] = df_exploded["bbox"].apply(
        lambda bbox: [round(coord, 1) for coord in bbox]
    )

    # Load GT annotations (required for setting `image_id`)
    coco = COCO(anns_path)
    # Parse GT-images
    coco_imgs = pd.DataFrame(coco.imgs.values())

    # Convert the DataFrame to a list of dictionaries in the desired format
    json_data = []
    for _, row in df_exploded.iterrows():
        image_id = coco_imgs[
            coco_imgs.file_name == row["image_file"].split("/")[1]
        ].id.values[0]

        json_data.append(
            {
                "image_id": int(image_id),
                # subtract 1 to match official annotation
                "category_id": int(row["class"]) - 1,
                # convert bbox coord.s to rounded floats
                "bbox": [round(float(coord), 1) for coord in row["bbox"]],
                "score": float(row["score"]),
            }
        )

    # Save the list of dictionaries as a JSON file
    with open(output_json_file, "w") as f:
        json.dump(json_data, f, indent=4)

    logger.info(f"The resulting file is saved at: {output_json_file}.")
    return output_json_file
