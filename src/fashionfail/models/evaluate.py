from pprint import pprint

import numpy as np
import pandas as pd
import torch
from IPython.core.display_functions import display
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from fashionfail.process_preds import (
    bbox_conversion_formats,
    convert_preds_to_coco,
    load_tpu_preds,
)
from fashionfail.utils import extended_box_convert, load_categories


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
        "--eval_method",
        type=str,
        required=True,
        choices=["COCO", "TorchMetrics", "all"],
        help="The name of the evaluation framework to be used, or `all` to run all eval methods.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["amrcnn", "fformer"],
        help="The name of the model, either 'amrcnn' or 'fformer'.",
    )

    return parser.parse_args()


def _print_per_class_metrics(coco: COCO, coco_eval: COCOeval) -> None:
    # Display per class metrics
    cat_ids = coco_eval.params.catIds
    cat_names = [cat["name"] for cat in coco.loadCats(ids=cat_ids)]

    m_aps = []
    for c in cat_ids:
        pr = coco_eval.eval["precision"][:, :, c, 0, 2]
        if not np.isnan(pr).all():
            m_ap = np.nanmean(pr)  # Use np.nanmean to handle NaN values
        else:
            m_ap = 0.0  # Set a default value if there are no positive detections
        m_aps.append(m_ap)

    cats = pd.DataFrame({"name": cat_names, "AP": m_aps})
    cats["name"] = cats["name"].str.slice(
        0, 15
    )  # Limit the number of characters to 15 for better readability
    display(cats)


def eval_with_coco(args) -> None:
    # Convert predictions to COCO format
    annotation_file = convert_preds_to_coco(
        preds_path=args.preds_path, anns_path=args.anns_path, model_name=args.model_name
    )

    # Load GT annotations
    coco = COCO(args.anns_path)

    # Load predictions (dt)
    coco_dt = coco.loadRes(annotation_file)

    # running evaluation
    coco_eval = COCOeval(coco, coco_dt, args.iou_type)
    coco_eval.params.catIds = list(load_categories().keys())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    _print_per_class_metrics(coco, coco_eval)


def eval_with_torchmetrics(args) -> None:
    # Load GT annotations
    coco = COCO(args.anns_path)

    # Load predictions
    df_preds = load_tpu_preds(args.preds_path, preprocess=True)

    # Initialize metric
    metric = MeanAveragePrecision(
        class_metrics=True, box_format="xywh", iou_type=args.iou_type
    )

    catIds = list(load_categories().keys())

    for img_info in tqdm(
        coco.imgs.values(), desc="Accumulating predictions and targets"
    ):
        img_id, img_name = img_info["id"], img_info["file_name"]

        # Get Ground-Truth for the image
        annIds = coco.getAnnIds(imgIds=img_id, catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        gt_df = pd.DataFrame(anns)

        if gt_df.empty:
            raise AssertionError(
                f"No annotation found for image: {img_name} with image ID: {img_id}"
            )
        # Check if there are multiple ground truths, if so aggregate them
        if gt_df.shape[0] > 1:
            gt_df = gt_df.groupby("image_id").agg(
                {"category_id": list, "segmentation": list, "bbox": list}
            )

        gt_box = np.array(gt_df["bbox"].values[0])
        gt_class = np.array(gt_df["category_id"].values[0])
        # Increment GT class ([0,45]) to match with `amrcnn` predictions ([1,46])
        gt_class = gt_class + 1 if args.model_name == "amrcnn" else gt_class

        # Get detections for the image
        dt_df = df_preds[df_preds.image_file == img_name]
        if dt_df.empty:
            raise AssertionError(
                f"No prediction found for image: {img_name} with image ID: {img_id}"
            )
        # Handle the case where no detection is predicted
        if dt_df["boxes"].values[0].size == 0:
            dt_boxes = torch.tensor([])
        else:
            # Convert bboxes according to their format
            in_fmt, out_fmt = bbox_conversion_formats.get(args.model_name, (None, None))
            if in_fmt is None or out_fmt is None:
                raise ValueError(f"Unsupported model_name: {args.model_name}")
            dt_boxes = extended_box_convert(
                torch.tensor(dt_df["boxes"].values[0]),
                in_fmt=in_fmt,
                out_fmt=out_fmt,
            )
        dt_scores = dt_df["scores"].values[0]
        dt_classes = dt_df["classes"].values[0]

        preds = [
            dict(
                boxes=dt_boxes,
                scores=torch.tensor(dt_scores),
                labels=torch.tensor(dt_classes),
            )
        ]

        target = [
            dict(
                boxes=torch.unsqueeze(torch.tensor(gt_box), dim=0)
                if gt_box.ndim == 1
                else torch.tensor(gt_box),
                labels=torch.unsqueeze(torch.tensor(gt_class), dim=0)
                if gt_class.size == 1
                else torch.tensor(gt_class),
            )
        ]

        # Update metric with predictions and respective ground truth
        metric.update(preds, target)

    logger.info("Computing evaluation metrics...")
    # Compute the results
    result = metric.compute()
    pprint(result)


if __name__ == "__main__":
    cli_args = get_cli_args()

    if cli_args.eval_method == "COCO":
        eval_with_coco(cli_args)
    elif cli_args.eval_method == "TorchMetrics":
        eval_with_torchmetrics(cli_args)
    else:
        logger.info("=" * 10 + "Evaluating with COCOeval" + "=" * 10)
        eval_with_coco(cli_args)
        logger.info("=" * 10 + "Evaluating with TorchMetrics" + "=" * 10)
        eval_with_torchmetrics(cli_args)
