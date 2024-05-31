import argparse
import concurrent.futures
import json
from pathlib import Path

import pandas as pd
import requests  # type: ignore
from huggingface_hub import hf_hub_download
from loguru import logger
from tqdm import tqdm

# Add transformations to apply to images
# See: https://cloudinary.com/documentation/transformation_reference
IMAGE_PARAMS = (
    "images/",
    "images/f_jpg,q_auto,fl_lossy,c_fill,g_auto/",
)


def download_image(session, url, img_params, save_dir, filename):
    img_url = url.replace(img_params[0], img_params[1])
    save_path = f"{save_dir}/{filename}"
    try:
        response = session.get(img_url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    except (requests.exceptions.RequestException, OSError):
        print(f"Could not download: {url}")


def download_image_wrapper(params):
    session, url, img_params, save_dir, filename, pbar = params
    download_image(session, url, img_params, save_dir, filename)
    pbar.update(1)


def download_images(out_dir) -> None:
    for split in ["train", "val", "test"]:
        img_dir = f"{out_dir}/images/{split}"

        # Retrieve image URLs & image filenames from .json
        with open(out_dir / f"ff_{split}.json") as fp:
            data = json.load(fp)
        df_imgs = pd.DataFrame(data["images"])
        img_urls = df_imgs["original_url"].tolist()
        img_names = df_imgs["file_name"].tolist()

        # Initialize tqdm progress bar with the total number of URLs
        with tqdm(total=len(img_urls)) as pbar, requests.Session() as session:
            # Use ThreadPoolExecutor for concurrent image downloads
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # List to store futures for monitoring completion
                futures = []

                # Iterate over URLs and image names
                for url, filename in zip(img_urls, img_names):
                    # Pack parameters for download_image_wrapper
                    params = (session, url, IMAGE_PARAMS, img_dir, filename, pbar)

                    # Submit the download task to the executor
                    futures.append(executor.submit(download_image_wrapper, params))

                # Wait for all futures to complete
                concurrent.futures.wait(futures)


def create_dirs(out_dir) -> None:
    # Create directories if not exist
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        img_dir = out_dir / "images/" / split
        img_dir.mkdir(parents=True, exist_ok=True)


def download_datasets(out_dir) -> None:
    """Download train/val/test splits of FashionFail from HuggingFace."""
    repo_id = "rizavelioglu/fashionfail"

    for filename in ["ff_train.json", "ff_val.json", "ff_test.json"]:
        _ = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=out_dir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="~/.cache/fashionfail/",
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

    # Download .json file for each split (image URLs and annotations)
    download_datasets(out_dir=save_dir)

    # Download images for each split
    download_images(out_dir=save_dir)

    # Display a completion message
    logger.info(
        f"Download finished! Images and annotations are saved at: {args.save_dir}"
    )
