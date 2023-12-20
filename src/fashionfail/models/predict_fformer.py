import os
from glob import glob
from pathlib import Path

import mmcv
import numpy as np
import torch
from loguru import logger
from mmdet.apis import inference_detector, init_detector
from pycocotools import mask as mask_api
from tqdm import tqdm

from fashionfail.utils import masks_to_boxes


def get_cli_args_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The path to the checkpoint/trained model.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="The image directory for prediction.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the dataset, which will be included in the output filename.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="The output directory where the predictions file will be saved.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="The image directory for prediction.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.0,
        help="Minimum score of bboxes to be shown.",
    )
    return parser


def predict(
    model_path: str,
    config_path: str,
    dataset_name: str,
    out_dir: str,
    image_dir: str,
    score_threshold: float,
    device: torch.device,
) -> None:
    # build the model from a config file and a checkpoint file
    model = init_detector(config_path, model_path, device=device)

    # Get image paths from `image_dir`
    img_list = glob(os.path.join(image_dir, "*.jpg"))

    # Accumulate results in a list, save as `.npz` file.
    preds = []
    logger.debug("Running inference now...")
    for image in tqdm(img_list):
        # Run inference for a single image
        result = inference_detector(model, [image])

        # Parse result --> list[tuple(bbox, masks, attributes)]
        bbox_result, segm_result, _ = result[0]

        # Process labels
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        # Process segmentation masks
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)

        # Process scores
        scores = np.vstack(bbox_result)[:, -1]

        # Filter results based on threshold
        if score_threshold:
            inds = scores > score_threshold
            scores = scores[inds]
            labels = labels[inds]
            if segms is not None:
                segms = segms[inds, ...]

        # Get boxes from masks
        boxes = masks_to_boxes(torch.from_numpy(segms))

        # Process masks
        encoded_masks = [
            mask_api.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in segms
        ]

        # Accumulate results.
        preds.append(
            {
                "image_file": Path(image).name,
                "boxes": boxes.numpy(),
                "classes": labels,
                "scores": scores,
                "masks": encoded_masks,
            }
        )

    # Save results in a compressed `.npz` file: 'model_name-dataset_name.npz'
    out_file_name = f"{Path(model_path).stem}-{dataset_name}.npz"
    out_file_path = os.path.join(out_dir, out_file_name)
    np.savez_compressed(out_file_path, data=preds)
    logger.debug(f"Results are saved at: {out_file_path}")


if __name__ == "__main__":
    # Parse args
    args = get_cli_args_parser().parse_args()

    # setting device on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run inference and store results
    predict(
        args.model_path,
        args.config_path,
        args.dataset_name,
        args.out_dir,
        args.image_dir,
        args.score_threshold,
        device,
    )
