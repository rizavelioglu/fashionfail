"""
Convert the raw annotations (both label(from `textdavinci`) and bbox-masks(from GroundingDINO+SAM)) of FashionFail
dataset to COCO format, yielding a `.json` file.
"""
import json
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sahi.utils.coco import Coco, CocoAnnotation, CocoImage
from sahi.utils.file import save_json
from supervision.dataset.utils import approximate_mask_with_polygons
from torchvision.ops import box_convert
from tqdm import tqdm
from utils import load_categories


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


ANNS_DIR = "/home/rizavelioglu/work/data/fashionfail/annotations"
GT_DIR = "/home/rizavelioglu/work/data/fashionfail/labeled_images_GT.json"
RAW_CATEGORIES = "/home/rizavelioglu/work/repos/segmentation/segmentation/visualization/categories.json"

# Load categories
category_id_to_name = load_categories()
name_to_category_id = {v: k for k, v in category_id_to_name.items()}

with open(RAW_CATEGORIES) as fp:
    cats = json.load(fp)
categories = [
    {"id": str(d["id"]), "name": d["name"], "supercategory": d["supercategory"]}
    for d in cats
]

# Init sahi.Coco instance
coco = Coco()
# add categories
coco.add_categories_from_coco_category_list(categories)

# Load label-GT
df_cat = pd.read_csv(
    "/home/rizavelioglu/work/data/fashionfail/df_cat-processed.csv",
    usecols=["image_name", "label"],
)
# Convert label to int id and create a new column "label_id"
df_cat["label_id"] = df_cat["label"].map(lambda label: name_to_category_id.get(label))

# Get GT image names
with open(GT_DIR, "r+") as f:
    gt_image_names = json.load(f)
gt_image_names = gt_image_names["images_to_keep"]


for image_name in tqdm(gt_image_names):
    ann_name = image_name.replace(".jpg", ".pkl")
    # Load class-GT
    gt_class = df_cat[df_cat.image_name == image_name].label_id.values[0]

    if np.isnan(gt_class):
        continue

    # Load box-GT
    with open(f"{ANNS_DIR}/{ann_name}", "rb") as f:
        gt = pickle.load(f)

    gt.class_id = int(gt_class)

    im = Image.open(f"/home/rizavelioglu/work/data/fashionfail/images/{image_name}")
    width, height = im.size

    coco_image = CocoImage(file_name=image_name, height=height, width=width)
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

save_json(coco.json, "/home/rizavelioglu/work/data/fashionfail/ff-sample_coco.json")
