import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from pycocotools import mask as mask_api
from sklearn.model_selection import train_test_split


def create_dirs(out_dir) -> None:
    # Create directories if not exist
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = out_dir / "images/"
    img_dir.mkdir(parents=True, exist_ok=True)


def download_annotation(url, out_dir) -> None:
    """Download an annotation file."""
    try:
        response = requests.get(url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Extract the filename from the URL
            filename = os.path.basename(url)
            # Write the content to a file
            with open(os.path.join(out_dir, filename), "wb") as f:
                f.write(response.content)
        else:
            print(f"Failed to download JSON file. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")


def download_images_and_unzip(url, extract_to):
    try:
        # Make a GET request to download the zip file
        response = requests.get(url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Extract the content of the zip file
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            print(
                f"Failed to download the zip file. Status code: {response.status_code}"
            )
    except Exception as e:
        print(f"An error occurred: {e}")


def split_train_val_and_export(ann_file, target_dir, test_size=0.25, random_seed=42):
    """
    Split data into train and validation sets, filter annotations and images, and export filtered data into separate JSON files.

    Args:
    - ann_file (str): Path to the annotation file.
    - target_dir (str): Directory to save the filtered data.
    - test_size (float): Proportion of the dataset to include in the validation split.
    - random_seed (int): Random seed for reproducibility.
    """

    # Load original training dataset
    with open(ann_file) as file:
        data = json.load(file)

    # Split data into train-val
    img_ids = pd.DataFrame(data["images"]).id.values
    imgId_train, imgId_val = train_test_split(
        img_ids, test_size=test_size, random_state=random_seed
    )

    def filter_and_export_data(data, image_ids, target_file):
        # Filter annotations and images
        anns = pd.DataFrame(data["annotations"])
        imgs = pd.DataFrame(data["images"])

        anns_filtered = anns[anns.image_id.isin(image_ids)]
        imgs_filtered = imgs[imgs.id.isin(image_ids)]

        # Replace old dictionary with filtered data
        data["annotations"] = anns_filtered.to_dict("records")
        data["images"] = imgs_filtered.to_dict("records")

        # Export as json
        with open(target_file, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, separators=(",", ":"), indent=4)

    # Process train data
    filter_and_export_data(data.copy(), imgId_train, target_dir / "fp_train.json")

    # Process val data
    filter_and_export_data(data.copy(), imgId_val, target_dir / "fp_val.json")


def convert_mixed_masks_to_rle(in_path, out_path):
    with open(in_path) as file:
        data = json.load(file)

    imgs = pd.DataFrame(data["images"])
    anns = pd.DataFrame(data["annotations"])

    to_convert = anns[
        ~anns.segmentation.apply(lambda x: True if isinstance(x, dict) else False)
    ]
    already_converted = anns[
        anns.segmentation.apply(lambda x: True if isinstance(x, dict) else False)
    ]

    # convert polygons to encoded RLE
    for idx, row in to_convert.iterrows():
        imgId = row["image_id"]
        h = imgs[imgs.id == imgId].height.values[0]
        w = imgs[imgs.id == imgId].width.values[0]

        segmentation = row["segmentation"]
        segmentation = mask_api.merge(mask_api.frPyObjects(segmentation, h, w))
        segmentation["counts"] = segmentation["counts"].decode("utf-8")

        to_convert.at[idx, "segmentation"] = segmentation

    df = pd.concat([already_converted, to_convert])
    data["annotations"] = df.to_dict("records")

    # export as json
    with open(out_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, separators=(",", ":"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="~/.cache/fashionpedia/",
        help="The directory where dataset will be saved.",
    )

    # Parse cli arguments.
    args = parser.parse_args()

    # Handle if tilde (~) is present in the `save_dir`
    save_dir = (
        Path(args.save_dir).expanduser() if "~" in args.save_dir else args.save_dir
    )

    # Create directories where dataset will be saved
    create_dirs(out_dir=save_dir)

    # 1. Download dataset
    logger.info(f"Downloading Fashionpedia dataset. This may take a few minutes...")
    ann_train = "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json"
    ann_val = "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json"
    img_train = "https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip"
    img_val_test = (
        "https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip"
    )

    # Download annotation file for each split
    download_annotation(url=ann_train, out_dir=save_dir)
    download_annotation(url=ann_val, out_dir=save_dir)

    # Download images.zip and unzip
    download_images_and_unzip(url=img_train, extract_to=save_dir / "images/")
    download_images_and_unzip(url=img_val_test, extract_to=save_dir / "images")

    # 2. Split train data into train & val
    logger.info(f"Splitting Fashionpedia-train into train and val sets...")
    split_train_val_and_export(
        ann_file=save_dir / "instances_attributes_train2020.json",
        target_dir=save_dir,
        test_size=0.25,
        random_seed=42,
    )

    # 3. Convert the mask annotations to encoded RLE format (required for torchvision.transforms_v2)
    logger.info(f"Converting mask annotations to encoded RLE format...")
    convert_mixed_masks_to_rle(
        in_path=save_dir / "fp_val.json", out_path=save_dir / "fp_val_rle.json"
    )

    convert_mixed_masks_to_rle(
        in_path=save_dir / "fp_train.json", out_path=save_dir / "fp_train_rle.json"
    )

    # Display a completion message
    logger.info(f"Done! Images and annotations are saved at: {args.save_dir}")
