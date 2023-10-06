import argparse
import json
import os
import pickle

import cv2
import numpy as np
import torch
from groundingdino.util.inference import Model
from loguru import logger
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Full path to the images directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="The folder where the model outputs will be saved.",
    )
    parser.add_argument(
        "--image_names",
        type=str,
        required=True,
        help="Full path to the .json file where the image names of interest are stored.",
    )

    return parser.parse_args()


def save_annotations(save_path, annotations):
    # Write the annotations to a pickle file
    with open(f"{save_path}.pkl", "wb") as f:
        pickle.dump(annotations, f)


def has_anomaly_bbox(bboxes, img_height, img_width, img_name) -> bool:
    """
    Check for anomalies in bounding boxes.

    Args:
        bboxes (np.ndarray): Array of bounding boxes in format (x_min, y_min, x_max, y_max).
        img_height (int): Height of the image.
        img_width (int): Width of the image.
        img_name (str): Name or identifier of the image.

    Returns:
        bool: True if anomalies are detected, False otherwise.
    """
    if bboxes.shape[0] == 0:
        logger.warning(f"No box detected for image: {img_name}!")
        return True

    if bboxes.shape[0] > 1:
        logger.warning(f"Detected {bboxes.shape[0]} boxes for image: {img_name}!")
        return True

    if np.any(bboxes[:, :2] < 0) or np.any(
        bboxes[:, 2:] > np.array([img_width, img_height]), axis=1
    ):
        logger.warning(
            f"Invalid bounding box (out of bounds) detected for image: {img_name}!"
        )
        return True

    return False


def main():
    # Load GroundingDINO
    grounding_dino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
    )

    # Load SAM
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
        device=DEVICE
    )
    sam_predictor = SamPredictor(sam)

    for image_name in tqdm(IMAGE_NAMES):
        image = cv2.imread(IMAGES_DIR + image_name)

        # Perform object detection with GroundingDINO
        detections = grounding_dino_model.predict_with_caption(
            image=image,
            caption="an object",
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )
        # Skip image if there is an anomaly in detected boxes
        boxes = detections[0].xyxy
        has_anomaly = has_anomaly_bbox(
            bboxes=boxes,
            img_height=image.shape[0],
            img_width=image.shape[1],
            img_name=image_name,
        )
        if has_anomaly:
            continue

        # Run SAM with detected box
        sam_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result_masks = []
        for box in boxes:
            masks, scores, logits = sam_predictor.predict(
                box=box, multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])

        detections[0].mask = np.array(result_masks)

        # Save annotations to disk
        annotation_save_path = os.path.join(
            GT_SAVE_DIR, f'{image_name.replace(".jpg", "")}'
        )
        save_annotations(annotation_save_path, detections[0])


if __name__ == "__main__":
    # Parse cli arguments
    args = get_cli_args()

    # TODO: replace HOME (place dino_sam folder inside ff)
    HOME = "/home/rizavelioglu/work/repos/dino_sam/"
    GROUNDING_DINO_CONFIG_PATH = os.path.join(
        HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(
        HOME, "weights", "groundingdino_swint_ogc.pth"
    )
    SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
    SAM_ENCODER_VERSION = "vit_h"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BOX_THRESHOLD = 0.45
    TEXT_THRESHOLD = 0.25

    IMAGES_DIR = args.images_dir
    GT_SAVE_DIR = args.out_dir
    with open(args.image_names) as f:
        IMAGE_NAMES = json.load(f)["images_to_keep"]

    logger.info(
        f"Running GroundingDINO+SAM for images inside: {IMAGES_DIR}. Total number of images: {len(IMAGE_NAMES)}"
        f"\nAnnotations will be saved at: {GT_SAVE_DIR}"
    )
    main()
