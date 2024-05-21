import os
from glob import glob
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from pycocotools import mask as mask_api
from torchvision.io import read_image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from tqdm import tqdm

from fashionfail.utils import extended_box_convert


def get_cli_args_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["facere_base", "facere_plus"],
        help="Name of the model to run inference, either `facere_base` or `facere_plus`.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        default=None,
        help="The image directory for prediction.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="The directory where predictions will be saved.",
    )

    return parser


def predict_with_onnx(model_name, image_dir, out_dir):
    # Load pre-trained model transformations.
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()

    # Load model
    path_to_onnx = hf_hub_download(
        repo_id="rizavelioglu/fashionfail",
        filename=f"{model_name}.onnx",
        repo_type="model",
    )

    # Create an inference session.
    ort_session = onnxruntime.InferenceSession(
        path_to_onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    # Run inference on images, accumulate results in a list, save as `.npz` file.
    preds = []
    proba_threshold = 0.5

    logger.debug("Running inference now...")
    for image in tqdm(glob(os.path.join(image_dir, "*.jpg"))):
        # Preprocess image
        img = read_image(image)
        img_transformed = transforms(img)
        # Get predictions.
        ort_inputs = {
            ort_session.get_inputs()[0].name: img_transformed.unsqueeze(dim=0).numpy()
        }
        ort_outs = ort_session.run(None, ort_inputs)

        # Parse the outputs
        boxes, labels, scores, masks = ort_outs

        # Process masks
        masks = masks.squeeze(1)
        filtered_masks = masks > proba_threshold
        encoded_masks = [
            mask_api.encode(np.asfortranarray(mask.astype(np.uint8)))
            for mask in filtered_masks
        ]

        # Convert boxes to same format as `amrcnn` model
        boxes = extended_box_convert(torch.tensor(boxes), in_fmt="xyxy", out_fmt="yxyx")

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

    # Save results in a compressed `.npz` file
    out_file_name = model_name + ".npz"
    np.savez_compressed(os.path.join(out_dir, out_file_name), data=preds)
    logger.debug(f"Results are saved at: {args.out_dir + out_file_name}")


if __name__ == "__main__":
    # Parse args
    parser = get_cli_args_parser()
    args = parser.parse_args()

    # call the respective function
    predict_with_onnx(args.model_name, args.image_dir, args.out_dir)
