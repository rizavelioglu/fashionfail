import argparse
import datetime
import json
import pickle

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from pycocotools import mask as mask_api
from sklearn.model_selection import train_test_split
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
        "--cat_anns",
        type=str,
        required=True,
        help="Full path to the category annotations file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Full path to the directory where the resulting datasets will be saved.",
    )

    return parser.parse_args()


def get_coco_json_format(info_split_name: str):
    t = datetime.datetime.now()

    coco_base = {
        "info": {
            "year": int(t.year),
            "version": "1.0",
            "description": f"FashionFail-{info_split_name} Dataset",  # Update dataset name based on split name
            "contributor": "Riza Velioglu",
            "url": "https://rizavelioglu.github.io/fashionfail/",
            "date_created": t.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),  # Convert datetime to string format
        },
        "licenses": [
            {
                "id": 1,
                "name": "Copyright © 2017, adidas AG",
                "url": "https://www.adidas-group.com/en/service/legal-notice/",
            },
        ],
        "categories": [],
        "images": [],
        "annotations": [],
    }

    return coco_base


def construct_dataset(images_dir, bbox_mask_anns_dir, df_cat, out_dir, split):
    # Load categories
    categories = load_categories(return_raw_categories=True)

    coco = get_coco_json_format(info_split_name=split)
    coco["categories"] = categories

    # Check category annotations
    if df_cat.duplicated("image_name").any():
        logger.error(
            "The dataframe for the ground-truth labels have duplicated image names!"
        )
        return

    for image_id, image_name in tqdm(
        enumerate(df_cat.image_name.values.tolist(), start=1), total=df_cat.shape[0]
    ):
        # Accumulate image data
        width, height = Image.open(f"{images_dir}/{image_name}").size
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_name,
                "height": height,
                "width": width,
                "license": 1,
                "original_url": df_cat[df_cat.image_name == image_name].images.values[
                    0
                ],
            }
        )

        # Load box_mask-GT
        ann_name = image_name.replace(".jpg", ".pkl")
        with open(f"{bbox_mask_anns_dir}/{ann_name}", "rb") as f:
            gt = pickle.load(f)

        if len(gt) != 1:
            logger.error(
                f"Detections must have exactly 1 detection, but got: {len(gt)}, in {ann_name}."
            )
            break

        # Load class-GT
        gt_class = df_cat[df_cat.image_name == image_name].class_id.values[0]
        gt.class_id = int(gt_class)

        # Process the box
        bbox = box_convert(
            torch.tensor(gt.xyxy), in_fmt="xyxy", out_fmt="xywh"
        ).tolist()[
            0
        ]  # convert to coco format
        bbox = [
            round(coord, 1) for coord in bbox
        ]  # Round coordinates to the nearest tenth of a pixel

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
                "area": int(area),
                "iscrowd": 0,
                "bbox": bbox,
                "segmentation": segmentation,
            }
        )

    # export as json
    with open(f"{out_dir}/ff_{split}.json", "w", encoding="utf-8") as outfile:
        json.dump(coco, outfile, separators=(",", ":"), indent=2)


def split_data(category_anns):
    # Set the random seed for reproducibility
    random_seed = 42

    # Load category/label annotations
    df_cat = pd.read_csv(category_anns)

    # Split into train-test (specific test_size ensures 1k+ samples for test set)
    df_train, df_test = train_test_split(
        df_cat, test_size=0.401, stratify=df_cat["class_id"], random_state=random_seed
    )

    # Some classes have only 1 sample in train, do split without them
    rare_class_id = 3
    df_rare_class = df_train[df_train.class_id.isin([rare_class_id])]

    df_train = df_train[~df_train.class_id.isin([rare_class_id])]
    df_train, df_val = train_test_split(
        df_train, test_size=0.1, stratify=df_train["class_id"], random_state=random_seed
    )
    # Add rare sample to train set only
    df_train = pd.concat([df_train, df_rare_class])

    return df_train, df_test, df_val


if __name__ == "__main__":
    # Parse cli arguments
    args = get_cli_args()

    logger.info("Splitting the data into `train`, `val`, and `test`...")
    anns_train, anns_test, anns_val = split_data(category_anns=args.cat_anns)

    logger.info("Constructing datasets in COCO format...")
    for ann, split_name in zip(
        [anns_train, anns_test, anns_val], ["train", "test", "val"]
    ):
        construct_dataset(
            images_dir=args.images_dir,
            bbox_mask_anns_dir=args.anns_dir,
            df_cat=ann,
            out_dir=args.out_dir,
            split=split_name,
        )
    logger.info(f"Datasets are saved at: {args.out_dir}")
