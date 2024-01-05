from pprint import pprint
from typing import Literal

import numpy as np
import pandas as pd
import torch
from IPython.core.display_functions import display
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from fashionfail.models.cocoeval2 import COCOeval2
from fashionfail.models.prediction_utils import (
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
        choices=["COCO", "COCO-extended", "TorchMetrics", "all"],
        help="The name of the evaluation framework to be used, or `all` to run all eval methods.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["amrcnn", "fformer", "facere_plus"],
        help="The name of the model, either 'amrcnn' or 'fformer'.",
    )

    return parser.parse_args()


def print_per_class_metrics(coco_eval: COCOeval, return_results: bool = False):
    # Display per class metrics
    categories = load_categories()
    cat_ids = coco_eval.params.catIds
    cat_names = [categories.get(cat_id) for cat_id in cat_ids]

    m_aps = []
    for c in cat_ids:
        # [TxRxKxAxM]: A=0: area="all" & M=2: maxDets=100
        pr = coco_eval.eval["precision"][:, :, c, 0, 2]
        if len(pr[pr > -1]) == 0:
            m_ap = -1
        else:
            m_ap = np.mean(pr[pr > -1])
        m_aps.append(m_ap)

    cats = pd.DataFrame({"name": cat_names, "AP": m_aps})
    # Limit the number of characters to 15 for better readability
    cats["name"] = cats["name"].str.slice(0, 15)
    display(cats)
    if return_results:
        return cats


def print_tp_fp_fn_counts(coco_eval, iou_idx=0, area_idx=0, max_dets_idx=2):
    """
    Print a summary of metrics; TP, FP, FN counts, based on COCO evaluation results.

    Args:
        coco_eval (COCOeval2): An instance of the custom `COCOeval2` class, which is used as an alternative
            implementation to calculate and evaluate metrics that are not provided by the official COCOeval class.
        iou_idx (int, optional): Index for IoU threshold in [0.50, 0.05, 0.95]. Default is 0.
        area_idx (int, optional): Index for area range in ['all', 'small', 'medium', 'large']. Default is 0.
        max_dets_idx (int, optional): Index for maximum detections in [1, 10, 100]. Default is 2.

    Example:
        >>> print_tp_fp_fn_counts(coco_eval)
    """

    # Can't use `isinstance` because `coco_eval` is modified
    if coco_eval.__module__ != COCOeval2.__module__:
        logger.error(f"`coco_eval` object must be an object of {COCOeval2}!")
        return

    print(
        f"Metrics @[",
        f"IoU={coco_eval.params.iouThrs[iou_idx]} |",
        f"area={coco_eval.params.areaRngLbl[area_idx]} |",
        f"maxDets={coco_eval.params.maxDets[max_dets_idx]} ]",
    )

    print("_" * 30)
    print(f"| {'cat':<2} | {'TP':<5} | {'FP':<5} | {'FN':<5} |")  # header
    print(f"|{'-' * 5}|{'-' * 7}|{'-' * 7}|{'-' * 7}|")  # separator

    total_tp, total_fp, total_fn = 0, 0, 0

    for catId in list(load_categories().keys()):
        num_tp = int(coco_eval.eval["num_tp"][iou_idx, catId, area_idx, max_dets_idx])
        num_fp = int(coco_eval.eval["num_fp"][iou_idx, catId, area_idx, max_dets_idx])
        num_fn = int(coco_eval.eval["num_fn"][iou_idx, catId, area_idx, max_dets_idx])

        print(f"| {catId:<3} | {num_tp:<5} | {num_fp:<5} | {num_fn:<5} |")

        total_tp += num_tp
        total_fp += num_fp
        total_fn += num_fn

    print(f"{'-' * 30}")
    print(f"{'Total':<5} | {total_tp:<5} | {total_fp:<5} | {total_fn:<5} |")


def get_cocoeval(
    annotations_path: str,
    predictions_path: str,
    iou_type: Literal["bbox", "segm"] = "bbox",
    use_coco_eval2: bool = False,
):
    """
    Calculate COCO evaluation metrics for object detection or instance segmentation.

    Args:
        annotations_path (str): The file path to the ground truth annotations in COCO format.
        predictions_path (str): The file path to the prediction results in COCO format.
        iou_type (str): The type of intersection over union (IoU) to use for evaluation.
            Can be either "bbox" for bounding box IoU or "segm" for segmentation IoU. Default is "bbox".
        use_coco_eval2 (bool): If True, use a custom implementation (COCOeval2) to compute evaluation metrics,
            including TP (True Positives), FP (False Positives), and FN (False Negatives) counts.
            If False, use the standard COCOeval. Default is False.

    Returns:
        coco_eval: A COCO evaluation object containing computed metrics and results.

    Examples:
        Run official evalution and get access to 'eval' dict including metrics; 'precision',' recall', 'scores'.

        >>> coco_eval = get_cocoeval(annotations_path, predictions_path, iou_type="bbox", use_coco_eval2=False)

        Run customized evalution and get access to 'eval' dict including metrics; "num_tp", "num_fp", "num_fn",
        "scores_tp", "scores_fp" alongside 'precision',' recall', 'scores'.

        >>> coco_eval = get_cocoeval(annotations_path, predictions_path, iou_type="bbox", use_coco_eval2=False)

    """
    # Load GT annotations
    coco = COCO(annotations_path)

    # Load predictions (dt)
    coco_dt = coco.loadRes(predictions_path)

    # Use own implementation if specified which returns TP,FP,FN counts
    if use_coco_eval2:
        coco_eval = COCOeval2(coco, coco_dt, iouType=iou_type)
    else:
        coco_eval = COCOeval(coco, coco_dt, iouType=iou_type)

    # Specify the category IDs for evaluation
    coco_eval.params.catIds = list(load_categories().keys())

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()

    return coco_eval


def eval_with_coco(args, use_extended_coco: bool = False) -> None:
    # Convert predictions to COCO format
    preds_path = convert_preds_to_coco(
        preds_path=args.preds_path, anns_path=args.anns_path, model_name=args.model_name
    )

    # Run evaluation and print results
    coco_eval = get_cocoeval(
        annotations_path=args.anns_path,
        predictions_path=preds_path,
        iou_type=args.iou_type,
        use_coco_eval2=use_extended_coco,
    )
    coco_eval.summarize()
    print_per_class_metrics(coco_eval)
    if use_extended_coco:
        print_tp_fp_fn_counts(coco_eval)


def eval_with_torchmetrics(args) -> None:
    # Load GT annotations
    coco = COCO(args.anns_path)

    # Load predictions
    df_preds = load_tpu_preds(args.preds_path)

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
    elif cli_args.eval_method == "COCO-extended":
        eval_with_coco(cli_args, use_extended_coco=True)
    elif cli_args.eval_method == "TorchMetrics":
        eval_with_torchmetrics(cli_args)
    else:
        logger.info("=" * 10 + "Evaluating with official COCOeval" + "=" * 10)
        eval_with_coco(cli_args)
        logger.info("=" * 10 + "Evaluating with extended COCOeval" + "=" * 10)
        eval_with_coco(cli_args, use_extended_coco=True)
        logger.info("=" * 10 + "Evaluating with TorchMetrics" + "=" * 10)
        eval_with_torchmetrics(cli_args)
