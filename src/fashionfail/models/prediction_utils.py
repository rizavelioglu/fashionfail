import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from pycocotools.coco import COCO

from fashionfail.utils import extended_box_convert

# A dictionary that maps model names to bbox conversion formats. First string is the bbox
# format the model outputs, and the second string is the format we want the bboxes to be in.
# e.g. >>> in_fmt, out_fmt = bbox_conversion_formats.get(model_name, (None, None))
bbox_conversion_formats = {
    "amrcnn": ("yxyx", "xywh"),
    "fformer": ("xyxy", "xywh"),
    "facere": ("yxyx", "xywh"),
    "facere_plus": ("yxyx", "xywh"),
}


def load_tpu_preds(
    path_to_preds: str, filter_cat_ids: Optional[list[int]] = None
) -> pd.DataFrame:
    """
    Load predictions from a file, apply preprocessing, and optionally filter predictions based on category IDs.

    Args:
        path_to_preds (str): Path to the predictions file.
        filter_cat_ids (list[int], optional): If specified, only keeps predictions for these category IDs and filters
        out others. Defaults to None.

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

    # Apply basic preprocessing & filter for catIds if specified
    df_preds = clean_df_preds(df_preds, filter_cat_ids=filter_cat_ids)

    return df_preds


def _filter_preds_for_classes(
    row: pd.Series, class_ids: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Filter prediction attributes based on class IDs.

    This function filters classes, scores, boxes, and masks attributes in the predictions
    dataframe based on the `class_ids`.

    Args:
        row (pd.Series): A Pandas Series representing a row of predictions with attributes
            'classes', 'scores', 'boxes', and 'masks'.
        class_ids (List[int]): A list of class IDs to filter the predictions.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]: A tuple containing filtered
        arrays for 'classes', 'scores', 'boxes', and a list of filtered 'masks'.
    """
    class_mask = np.isin(row["classes"], class_ids)
    filtered_classes = row["classes"][class_mask].astype(np.int32)
    filtered_scores = row["scores"][class_mask].astype(np.float32)
    filtered_boxes = row["boxes"][class_mask].astype(np.float32)

    # Ensure that filtered_boxes has the desired shape when empty
    filtered_boxes = (
        filtered_boxes.reshape((0, 4)) if filtered_boxes.size == 0 else filtered_boxes
    )

    # Filter masks based on filtered_classes
    filtered_masks = [row["masks"][i] for i in np.where(class_mask)[0]]

    return filtered_classes, filtered_scores, filtered_boxes, filtered_masks


def clean_df_preds(
    df_preds: pd.DataFrame, filter_cat_ids: Optional[list[int]] = None
) -> pd.DataFrame:
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

    # Apply the filtering function to the dataframe and update the predictions
    if filter_cat_ids:
        (
            df_preds["classes"],
            df_preds["scores"],
            df_preds["boxes"],
            df_preds["masks"],
        ) = zip(
            *df_preds.apply(_filter_preds_for_classes, axis=1, class_ids=filter_cat_ids)
        )
        # Logging
        logger.info(
            f"Filtered out predictions made for categoryID's = {filter_cat_ids}..."
        )
        nb_of_samples_w_no_preds_after_filter = (
            df_preds[df_preds["classes"].apply(lambda x: x.size == 0)].shape[0]
            - nb_of_samples_w_no_preds
        )
        logger.debug(
            f"Number of samples with no predictions after filtering categories: {nb_of_samples_w_no_preds_after_filter}."
        )

    # Remove samples with no predictions, otherwise calculations fail
    df_preds = df_preds[df_preds["boxes"].apply(lambda box: box.size != 0)]

    return df_preds


def _fix_class_ids(series):
    # Define the excluded indices
    excluded_indices = {2, 12, 16, 19, 20}
    # Create a DataFrame for the remaining categories
    remaining_categories = list(set(range(27)) - excluded_indices)
    df_mapping = pd.DataFrame(
        {
            "old_cat_id": remaining_categories,
            "new_cat_id": range(len(remaining_categories)),
        }
    )
    # Create a dictionary that maps new IDs to old(original) IDs
    new_id_to_org_id = dict(zip(df_mapping["new_cat_id"], df_mapping["old_cat_id"]))

    return series.apply(lambda i: new_id_to_org_id.get(i - 1))


def convert_preds_to_coco(
    preds_path: str,
    anns_path: str,
    model_name: str,
    filter_cat_ids: Optional[list[int]] = None,
) -> str:
    logger.info("Converting raw predictions to COCO format...")
    # The path to the resulting .json file
    output_json_file = preds_path.replace(Path(preds_path).suffix, "-coco.json")

    if Path(output_json_file).exists():
        logger.warning(
            f"Predictions are already converted to COCO format! To re-convert, remove the file at: {output_json_file}."
        )
        return output_json_file

    # Load predictions belonging to classes of interest
    df_preds = load_tpu_preds(preds_path, filter_cat_ids=filter_cat_ids)

    # Explode predictions for each sample
    df_exploded = df_preds.explode(["classes", "scores", "boxes", "masks"])

    # Rename and reorder columns
    df_exploded = df_exploded.rename(
        columns={
            "classes": "class",
            "boxes": "bbox",
            "scores": "score",
            "masks": "mask",
        }
    )

    # re-map class id's to original ones
    if model_name == "facere_plus":
        df_exploded["class"] = _fix_class_ids(df_exploded["class"])

    # Apply the extended_box_convert function to each box in the "bbox" column
    in_fmt, out_fmt = bbox_conversion_formats.get(model_name)
    df_exploded["bbox"] = df_exploded["bbox"].apply(
        lambda bbox: extended_box_convert(
            torch.tensor(bbox), in_fmt=in_fmt, out_fmt=out_fmt
        ).numpy()
    )

    # Process masks: convert 'counts' from binary to utf-8
    df_exploded["mask"] = df_exploded["mask"].apply(
        lambda x: {"size": x["size"], "counts": x["counts"].decode("utf-8")}
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
        # retrieve COCO imgId
        image_id = coco_imgs[coco_imgs.file_name == row["image_file"]].id.values[0]

        json_data.append(
            {
                "image_id": int(image_id),
                # Subtract 1 to match official annotation in `amrcnn`
                "category_id": (
                    int(row["class"]) - 1
                    if model_name == "amrcnn" or model_name == "facere"
                    else int(row["class"])
                ),
                # Round coordinates to the nearest tenth of a pixel
                "bbox": [round(float(coord), 1) for coord in row["bbox"]],
                "score": float(row["score"]),
                "segmentation": row["mask"],
            }
        )

    # Save the list of dictionaries as a JSON file
    with open(output_json_file, "w") as f:
        json.dump(json_data, f, indent=4)

    logger.info(f"The resulting file is saved at: {output_json_file}.")
    return output_json_file
