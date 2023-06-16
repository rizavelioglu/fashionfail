import argparse
import concurrent.futures
import json
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

# Add transformations to apply to images
# See: https://cloudinary.com/documentation/transformation_reference
IMAGE_SOURCES_TO_PARAMS = {
    "assetmanagerpim-res.cloudinary.com": (
        "images/",
        "images/f_jpg,q_auto,fl_lossy,c_fill,g_auto/",
    ),  # adidas
    "images.puma.com": (
        "upload/",
        "upload/f_jpg,q_auto,fl_lossy,c_fill,g_auto/",
    ),  # puma
    "nb.scene7.com": (
        "$dw_detail_gallery$",
        "&bgcolor=f1f1f1&wid=1600&hei=1600.jpg",
    ),  # new balance
    "birkenstock.com": ("", ""),  # birkenstock
}


def download_image(session, url, img_params, save_dir):
    img_url = url.replace(img_params[0], img_params[1])
    filename = Path(url).name
    save_path = save_dir / filename
    try:
        response = session.get(img_url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    except (requests.exceptions.RequestException, OSError):
        print(f"Could not download: {url}")


def determine_source(url):
    # Split the URL by the forward slash (/)
    components = url.split("/")
    if len(components) >= 3:
        source = components[2]
        return source
    else:
        return None


def get_img_params_for_source(source):
    try:
        img_params = IMAGE_SOURCES_TO_PARAMS[source]
        return img_params
    except KeyError:
        logger.exception(f"Image URL source: `{source}` is unknown!")


def download_images(cli_args):
    save_dir = Path(cli_args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(cli_args.json_path) as f:
        img_info = json.load(f)

    img_urls = img_info["images"]

    # Filter out already downloaded image URLs
    already_downloaded_urls = [f.name for f in save_dir.iterdir()]
    remaining_urls = [
        url for url in img_urls if Path(url).name not in already_downloaded_urls
    ]

    with tqdm(total=len(remaining_urls)) as pbar, requests.Session() as session:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=cli_args.max_workers
        ) as executor:
            for url in remaining_urls:
                source = determine_source(url)  # Determine the appropriate image source
                img_params = get_img_params_for_source(
                    source
                )  # Get the corresponding img_params
                executor.submit(download_image, session, url, img_params, save_dir)
                pbar.update(1)

    print("Download completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="The path to .json file where image URLs are stored.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="The directory where images will be saved.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        required=False,
        default=None,
        help="Number of workers in Multithread.",
    )
    # Parse cli arguments.
    args = parser.parse_args()
    # Execute the main function with args.
    download_images(args)
