import argparse
import os
import pickle

import pandas as pd
import torch
from PIL import Image
from sahi.utils.coco import Coco, CocoAnnotation, CocoImage
from sahi.utils.file import save_json
from supervision.dataset.utils import approximate_mask_with_polygons
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


def mask_to_polygon(mask):
    # Taken from: https://github.com/roboflow/supervision/blob/b47bc808f768fe1287647b7d391ebef254e172e3/supervision
    # /dataset/formats/coco.py#L111-L120
    polygon = []
    if mask is not None:
        polygon = list(
            approximate_mask_with_polygons(
                mask=mask,
                min_image_area_percentage=0,
                max_image_area_percentage=1.0,
                approximation_percentage=0,
            )[0].flatten()
        )
    return [polygon] if polygon else []


def load_and_process_categories():
    # Return the mapping
    category_id_to_name = load_categories()
    # Also return the raw categories (sahi.COCO requires id's to be strings)
    raw_categories = load_categories(return_raw_categories=True)
    categories = [
        {"id": str(d["id"]), "name": d["name"], "supercategory": d["supercategory"]}
        for d in raw_categories
    ]
    return categories, category_id_to_name


def main():
    # Load categories
    categories, category_id_to_name = load_and_process_categories()

    # Init sahi.Coco instance
    coco = Coco()
    # add categories
    coco.add_categories_from_coco_category_list(categories)

    # Load label-GT
    df_cat = pd.read_csv(LABEL_ANNS_FILE, usecols=["image_name", "class_id"])

    for image_name in tqdm(os.listdir(IMAGES_DIR)):
        # Load box_mask-GT
        ann_name = image_name.replace(".jpg", ".pkl")
        with open(f"{ANNS_DIR}/{ann_name}", "rb") as f:
            gt = pickle.load(f)

        # Load class-GT
        gt_class = df_cat[df_cat.image_name == image_name].class_id.values[0]
        gt.class_id = int(gt_class)

        # Load image
        im = Image.open(f"{IMAGES_DIR}/{image_name}")
        width, height = im.size
        coco_image = CocoImage(file_name=image_name, height=height, width=width)

        # Add annotations
        coco_image.add_annotation(
            CocoAnnotation(
                bbox=box_convert(
                    torch.tensor(gt.xyxy), in_fmt="xyxy", out_fmt="xywh"
                ).tolist()[0],
                segmentation=mask_to_polygon(gt.mask.squeeze()),
                category_id=gt.class_id,
                category_name=category_id_to_name[gt.class_id],
            )
        )
        coco.add_image(coco_image)

    save_json(coco.json, OUTPUT_JSON_FILE)


if __name__ == "__main__":
    # Parse cli arguments
    args = get_cli_args()

    # Define constants
    IMAGES_DIR = args.images_dir
    ANNS_DIR = args.anns_dir
    LABEL_ANNS_FILE = args.label_anns_file
    OUTPUT_JSON_FILE = args.out_path

    main()
