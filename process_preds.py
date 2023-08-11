import numpy as np
import pandas as pd
from loguru import logger


def load_tpu_preds(path_to_preds):
    # Load tpu predictions to dataframe
    preds = np.load(path_to_preds, allow_pickle=True)
    df_preds = pd.DataFrame.from_dict(list(preds))

    logger.info(f"Predictions loaded from: {path_to_preds}")

    return df_preds


def _filter_preds_for_classes(row):
    """Filter classes, scores, boxes, and masks attribute in the predictions dataframe based on the 'class id'.

    ClassID's larger or equal to 27 (e.g. "sleeve", "neckline", etc.) belong to super-categories; "garment parts",
    "closures", "decorations", which we want to filter out.

    """
    filtered_classes = np.array(
        [class_id for class_id in row["classes"] if 1 <= class_id <= 27]
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


def clean_df_preds(df_preds):
    logger.info("Processing predictions dataframe...")
    # Preprocess dataframe
    df_preds = df_preds[["image_file", "classes", "scores", "boxes", "masks"]]
    df_preds = df_preds.reset_index(drop=True)
    df_preds["image_file"] = df_preds["image_file"].apply(lambda x: x.split("/")[1])

    # Logging
    nb_of_samples = df_preds.shape[0]
    nb_of_samples_w_no_preds = df_preds[
        df_preds["classes"].apply(lambda x: x.size == 0)
    ].shape[0]
    logger.debug(
        f"Number of samples: {nb_of_samples}, of which {nb_of_samples_w_no_preds} "
        f"(%{nb_of_samples_w_no_preds/nb_of_samples*100:.1f}) have no predictions!"
    )

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
        f"Number of samples with no predictions after filtering: {nb_of_samples_w_no_preds_after_filter}."
    )
    logger.info(
        f"Filtering samples that have no predictions, in total: {nb_of_samples_w_no_preds + nb_of_samples_w_no_preds_after_filter}."
    )

    # Filter out samples where no predictions made
    df_preds = df_preds[df_preds["classes"].apply(lambda x: x.size != 0)]

    return df_preds
