import os
from glob import glob
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from loguru import logger
from models import SimpleModel
from pycocotools import mask as mask_api
from torchvision.io import read_image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from tqdm import tqdm

from fashionfail.utils import extended_box_convert


def get_cli_args_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        default=None,
        help="The path to the trained model.",
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


def predict_with_pytorch(model_path, image_dir, out_dir):
    # Load model
    model = SimpleModel.load_from_checkpoint(model_path)
    model.eval()

    # Load pre-trained model transformations.
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()

    # Load images & apply transformations.
    img_list = [read_image(path) for path in glob(os.path.join(image_dir, "*.jpg"))]
    images = [transforms(im) for im in img_list]

    # Run inference
    logger.debug("Running inference now...")
    with torch.no_grad():
        preds = model(images)

    # TODO: encode masks, then save as in `predict_with_onnx` function.
    # Save results in a compressed `.npz` file
    out_file_name = model_path.split("/")[-1] + ".npz"
    np.savez_compressed(os.path.join(out_dir, out_file_name), data=preds)
    logger.debug(f"Results are saved at: {args.out_dir + out_file_name}")


def predict_with_onnx(model_path, image_dir, out_dir):
    # Load pre-trained model transformations.
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()

    # Create an inference session.
    ort_session = onnxruntime.InferenceSession(
        str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
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
    out_file_name = model_path.split("/")[-1] + ".npz"
    np.savez_compressed(os.path.join(out_dir, out_file_name), data=preds)
    logger.debug(f"Results are saved at: {args.out_dir + out_file_name}")


if __name__ == "__main__":
    SUPPORTED_MODEL_EXTENSIONS = {
        "ckpt": predict_with_pytorch,
        "onnx": predict_with_onnx,
    }
    # Parse args
    parser = get_cli_args_parser()
    args = parser.parse_args()

    # retrieve model extension
    model_extension = args.model_path.split(".")[-1]
    # call the respective function
    if model_extension in SUPPORTED_MODEL_EXTENSIONS:
        logger.debug(f"Loading the .{model_extension} model...")
        SUPPORTED_MODEL_EXTENSIONS[model_extension](
            args.model_path, args.image_dir, args.out_dir
        )
    else:
        logger.error(
            f"Supported model extensions: {list(SUPPORTED_MODEL_EXTENSIONS.keys())}, but passed: {model_extension}"
        )
