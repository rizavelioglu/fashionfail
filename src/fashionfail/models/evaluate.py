import pickle
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_convert
from tqdm import tqdm

from fashionfail.process_preds import (
    clean_df_preds,
    convert_tpu_preds_to_coco,
    load_tpu_preds,
)
from fashionfail.utils import load_categories, yxyx_to_xyxy


def get_cli_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preds_path",
        type=str,
        required=True,
        help="Full path to the predictions file.",
    )
    parser.add_argument(
        "--anns_path",
        type=str,
        required=True,
        help="Full path to the annotations file.",
    )
    parser.add_argument(
        "--iou_type",
        type=str,
        choices=["bbox", "segm"],
        default="bbox",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "fashionpedia",
            "fashionpedia-coco",
            "fashionfail",
            "fashionfail-coco",
        ],
        help="Name of the dataset for which evaluation should be executed.",
    )

    return parser.parse_args()


def eval_on_fashionpedia(args):
    # Define constants
    catIds = list(range(0, 28))

    # Load GT annotations
    coco = COCO(args.anns_path)

    # Load predictions from disk
    df_preds = load_tpu_preds(cli_args.preds_path)
    df_preds = clean_df_preds(df_preds)

    # Initialize metric
    metric = MeanAveragePrecision(
        class_metrics=True, box_format="xywh", iou_type=args.iou_type
    )

    for i in tqdm(
        coco.imgs.values(),
        desc="Accumulating predictions and targets",
    ):
        img_id, img_name = i["id"], i["file_name"]

        # Get Ground-Truth for the image
        annIds = coco.getAnnIds(imgIds=img_id, catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        gt = pd.DataFrame(anns)
        # Sanity check whether annotation exists for each image
        assert (
            not gt.empty
        ), f"No annotation found in the annotation file for image: {img_name} with image ID: {img_id}"
        gt = gt.groupby("image_id").agg(
            {"category_id": list, "segmentation": list, "bbox": list}
        )
        # Parse GT
        gt_box = np.array(gt.bbox.values[0])
        gt_class = np.array(gt.category_id.values[0]) + 1

        # Get detections for the image
        dt = df_preds[df_preds.image_file == img_name]
        # Sanity check whether annotation exists for each image
        assert (
            not dt.empty
        ), f"No prediction found in predictions file for image: {img_name} with image ID: {img_id}"
        # Parse DT
        dt_boxes = box_convert(
            yxyx_to_xyxy(torch.tensor(dt["boxes"].values[0])),
            in_fmt="xyxy",
            out_fmt="xywh",
        )
        dt_scores = dt["scores"].values[0]
        dt_classes = dt["classes"].values[0]

        preds = [
            dict(
                boxes=dt_boxes,
                scores=torch.tensor(dt_scores),
                labels=torch.tensor(dt_classes),
            )
        ]

        target = [
            dict(
                boxes=torch.tensor(gt_box),
                labels=torch.tensor(gt_class),
            )
        ]

        # Update metric with predictions and respective ground truth
        metric.update(preds, target)

    logger.info("Computing evaluation metrics...")
    # Compute the results
    result = metric.compute()
    pprint(result)


def eval_on_fashionpedia_coco(args):
    # Convert AMRCNN predictions to COCO format
    annotation_file = convert_tpu_preds_to_coco(
        preds_path=args.preds_path, anns_path=args.anns_path
    )

    # Load GT annotations
    coco = COCO(args.anns_path)

    # Load predictions (dt)
    coco_dt = coco.loadRes(annotation_file)

    # running evaluation
    coco_eval = COCOeval(coco, coco_dt, args.iou_type)
    coco_eval.params.catIds = list(range(0, 28))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def eval_on_fashionfail(args):
    # Load predictions from disk
    df_preds = load_tpu_preds(cli_args.preds_path)
    df_preds = clean_df_preds(df_preds)

    # Parse Fashionpedia categories.
    category_id_to_name = load_categories()
    name_to_category_id = {v: k for k, v in category_id_to_name.items()}

    GT_SAVE_DIR = "/home/rizavelioglu/work/data/fashionfail/annotations"

    # Load label-GT
    df_cat = pd.read_csv(
        "/home/rizavelioglu/work/data/fashionfail/df_cat-processed.csv",
        usecols=["image_name", "label"],
    )
    # Convert label to int id and create a new column "label_id"
    df_cat["label_id"] = df_cat["label"].map(
        lambda label: name_to_category_id.get(label)
    )

    # Initialize metric
    metric = MeanAveragePrecision(
        class_metrics=True, box_format="xywh", iou_type=args.iou_type
    )

    for i, row in tqdm(
        df_preds.iterrows(),
        total=df_preds.shape[0],
        desc="Accumulating predictions and targets",
    ):
        image_name = row["image_file"].replace(".jpg", "")

        # Load box-GT
        with open(f"{GT_SAVE_DIR}/{image_name}.pkl", "rb") as f:
            gt = pickle.load(f)
        gt_box = torch.tensor(gt.xyxy)

        # Load class-GT
        gt_class = (
            df_cat[df_cat.image_name == f"{image_name}.jpg"].label_id.values[0] + 1
        )

        # Get prediction for the image
        dt_boxes = yxyx_to_xyxy(torch.tensor(row["boxes"]))
        dt_classes = row["classes"]
        dt_scores = row["scores"]

        preds = [
            dict(
                boxes=dt_boxes,
                scores=torch.tensor(dt_scores),
                labels=torch.tensor(dt_classes),
            )
        ]

        target = [
            dict(
                boxes=gt_box,
                labels=torch.tensor([gt_class]),
            )
        ]

        # Update metric with predictions and respective ground truth
        metric.update(preds, target)

    logger.info("Computing evaluation metrics...")
    # Compute the results
    result = metric.compute()
    pprint(result)

    # Display per class metrics.
    # cats = pd.read_json("/home/rizavelioglu/recommendy/repos/segmentation/segmentation/visualization/categories.json")[["name"]]
    # cats["AP"] = result["map_per_class"].tolist()
    # display(cats.sort_values('name').transpose())


def eval_on_fashionfail_coco(args):
    pass


if __name__ == "__main__":
    cli_args = get_cli_args()

    match cli_args.dataset:
        case "fashionpedia":
            eval_on_fashionpedia(cli_args)
        case "fashionpedia-coco":
            eval_on_fashionpedia_coco(cli_args)
        case "fashionfail":
            eval_on_fashionfail(cli_args)
        case "fashionfail-coco":
            eval_on_fashionfail_coco(cli_args)
