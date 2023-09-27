import argparse
import json
import pickle

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from pycocotools import mask as mask_api
from torchvision.ops import box_convert
from tqdm import tqdm

from fashionfail.utils import load_categories


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Full path to the images directory.",
    )
    parser.add_argument(
        "--anns_dir",
        type=str,
        required=True,
        help="Full path to the bbox and masks annotations directory.",
    )
    parser.add_argument(
        "--label_anns_file",
        type=str,
        required=True,
        help="Full path to the label annotations file.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Full path to the output file.",
    )

    return parser.parse_args()


def get_coco_json_format():
    coco_base = {
        "info": {
            "description": "FashionFail Dataset",
            "url": "https://github.com/rizavelioglu/fashionfail",
            "version": "1.0",
            "year": 2023,
            "contributor": "Riza Velioglu",
            "date_created": "2023/10/10",
        },
        "licenses": [
            {
                "url": "https://www.adidas.de/",
                "id": 1,
                "name": "Copyright © 2017, adidas AG",
            },
        ],
        "categories": [],
        "images": [],
        "annotations": [],
    }

    return coco_base


def main():
    # Adapted from: https://www.kaggle.com/code/coldfir3/efficient-coco-dataset-generator

    # Load categories
    categories = load_categories(return_raw_categories=True)

    coco = get_coco_json_format()
    coco["categories"] = categories

    # Load label-GT
    df_cat = pd.read_csv(LABEL_ANNS_FILE, usecols=["image_name", "class_id"])
    if df_cat.duplicated("image_name").any():
        logger.error(
            "The dataframe for the ground-truth labels have duplicated image names!"
        )
        return

    for image_id, image_name in tqdm(
        enumerate(df_cat.image_name.values.tolist(), start=1), total=df_cat.shape[0]
    ):
        # Accumulate image data
        width, height = Image.open(f"{IMAGES_DIR}/{image_name}").size
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_name,
                "height": height,
                "width": width,
                "license": 1,
            }
        )

        # Load box_mask-GT
        ann_name = image_name.replace(".jpg", ".pkl")
        with open(f"{ANNS_DIR}/{ann_name}", "rb") as f:
            gt = pickle.load(f)

        if len(gt) != 1:
            logger.error(
                f"Detections must have exactly 1 detection, but got: {len(gt)}, in {ann_name}."
            )
            break

        # Load class-GT
        gt_class = df_cat[df_cat.image_name == image_name].class_id.values[0]
        gt.class_id = int(gt_class)

        # Process the mask
        mask = np.asfortranarray(gt.mask.squeeze().astype(np.uint8))  # Binary mask
        segmentation = mask_api.encode(mask)  # Encoding it back to rle (coco format)
        segmentation["counts"] = segmentation["counts"].decode(
            "utf-8"
        )  # converting from binary to utf-8
        area = mask_api.area(segmentation).item()  # calculating the area

        # Accumulate annotation data
        coco["annotations"].append(
            {
                "id": image_id,  # we have exactly 1 annotation per image, hence, image_id = annotation_id
                "image_id": image_id,
                "category_id": gt.class_id,
                "area": float(area),
                "iscrowd": 0,
                "bbox": box_convert(
                    torch.tensor(gt.xyxy), in_fmt="xyxy", out_fmt="xywh"
                ).tolist()[0],
                "segmentation": segmentation,
            }
        )

    # export as json
    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as outfile:
        json.dump(coco, outfile, separators=(",", ":"), indent=4)


if __name__ == "__main__":
    # Parse cli arguments
    args = get_cli_args()

    # Define constants
    IMAGES_DIR = args.images_dir
    ANNS_DIR = args.anns_dir
    LABEL_ANNS_FILE = args.label_anns_file
    OUTPUT_JSON_FILE = args.out_path

    logger.info("Converting the dataset to COCO format...")
    main()
    logger.info(f"Resulting COCO dataset is saved at: {OUTPUT_JSON_FILE}")
