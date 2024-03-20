"""
This will yield a mongodb instance at: "~/.fiftyone/"

Notes:
    Check out for more info about fiftyone:
    https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb#scrollTo=Jt2d_dlhW3Fv
    More on COCO evalution:
    - https://docs.voxel51.com/integrations/coco.html#coco-style-evaluation
    - https://docs.voxel51.com/tutorials/evaluate_detections.html#Evaluate-detections
"""
import argparse

import fiftyone as fo
import fiftyone.utils.coco as fouc
from loguru import logger

from fashionfail.utils import load_categories


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
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
        "--dataset_name",
        type=str,
        required=True,
        default="FashionFail-test",
        help="Name of the dataset within fiftyone.",
    )

    return parser.parse_args()


def init_fo_dataset(image_dir, ann_path, name):
    # Import dataset by explicitly providing paths to the source media and labels
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=image_dir,
        labels_path=ann_path,
        name=name,
        label_field="ground_truth",
    )
    return dataset


def add_predictions(dataset, preds_path, preds_name: str):
    # And add model predictions
    fouc.add_coco_labels(
        dataset,
        label_field=preds_name,
        labels_or_path=preds_path,
        # classes=dataset.distinct("ground_truth_detections.detections.label"),
        classes=list(load_categories().values()),
        label_type="detections",
    )


if __name__ == "__main__":
    # Prediction paths of models
    # `amrcnn`
    PREDS_AMRCNN_FF_TEST = "/home/rizavelioglu/work/repos/tpu/models/official/detection/outputs/spinenet143-ff_test(filtered).json"
    PREDS_AMRCNN_FP_VAL = (
        "/home/rizavelioglu/work/repos/tpu/models/official/detection/outputs/"
    )
    # `amrcnn-R50`
    PREDS_AMRCNNR50_FF_TEST = "/home/rizavelioglu/work/repos/tpu/models/official/detection/outputs/r50fpn-ff_test(filtered).json"
    PREDS_AMRCNNR50_FP_VAL = (
        "/home/rizavelioglu/work/repos/tpu/models/official/detection/outputs/"
    )
    # `fformer`
    PREDS_FFORMER_FF_TEST = "/home/rizavelioglu/work/repos/FashionFormer/outputs/fashionformer_swin_b_3x-ff-test(filtered).json"
    PREDS_FFORMER_FP_VAL = "/home/rizavelioglu/work/repos/FashionFormer/outputs/"
    # `fformer-R50`
    PREDS_FFORMERR50_FF_TEST = "/home/rizavelioglu/work/repos/FashionFormer/outputs/fashionformer_r50_3x-ff-test(filtered).json"
    PREDS_FFORMERR50_FP_VAL = "/home/rizavelioglu/work/repos/FashionFormer/outputs/"
    # `facere`
    PREDS_FACERE_FF_TEST = "/home/rizavelioglu/work/repos/segmentation/segmentation/saved_models/facere_base/facere_2_ff-test(filtered).json"
    PREDS_FACERE_FP_VAL = "/home/rizavelioglu/work/repos/segmentation/segmentation/saved_models/facere_base/"
    # `facere_plus`
    PREDS_FACEREP_FF_TEST = "/home/rizavelioglu/work/repos/segmentation/segmentation/saved_models/facere_plus_ff-test-coco.json"
    PREDS_FACEREP_FP_VAL = "/home/rizavelioglu/work/repos/segmentation/segmentation/saved_models/facere_plus_fp-val-coco.json"

    # Parse cli arguments
    args = get_cli_args()

    try:
        fo_dataset = fo.load_dataset(args.dataset_name)
    except ValueError:
        logger.info(f"{args.dataset_name} does not exist! Initializing one now...")
        fo_dataset = init_fo_dataset(
            image_dir=args.image_dir, ann_path=args.anns_dir, name=args.dataset_name
        )
        logger.info(f"Adding predictions...")
        add_predictions(fo_dataset, PREDS_AMRCNN_FF_TEST, "preds-amrcnn")
        add_predictions(fo_dataset, PREDS_AMRCNNR50_FF_TEST, "preds-amrcnnR50")
        add_predictions(fo_dataset, PREDS_FFORMER_FF_TEST, "preds-fformer")
        add_predictions(fo_dataset, PREDS_FFORMERR50_FF_TEST, "preds-fformerR50")
        add_predictions(fo_dataset, PREDS_FACERE_FF_TEST, "preds-facere")

        # Save the dataset to disk
        fo_dataset.persistent = True
        fo_dataset.save()

    session = fo.launch_app(fo_dataset, port=5151)
    session.wait()
