import argparse
import concurrent.futures
from pathlib import Path

import pandas as pd
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
    "www.birkenstock.com": ("", ""),  # birkenstock
}


def download_image(session, url, img_params, save_dir, filename):
    img_url = url.replace(img_params[0], img_params[1])
    save_path = save_dir / filename
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
    # Create the directory if it doesn't exist
    save_dir = Path(cli_args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Read the DataFrame from a CSV file, extracting relevant columns
    df = pd.read_csv(cli_args.csv_path, usecols=["images", "image_name"])
    img_urls = df["images"].tolist()  # Extract the "images" column as a list
    image_names = df["image_name"].tolist()  # Extract the "image_name" column as a list

    # Filter out already downloaded image URLs based on the "image_name" column
    already_downloaded_names = [f.name for f in save_dir.iterdir()]
    remaining_urls = [
        url
        for url, name in zip(img_urls, image_names)
        if name not in already_downloaded_names
    ]

    # Initialize tqdm progress bar with the total number of remaining URLs
    with tqdm(total=len(remaining_urls)) as pbar, requests.Session() as session:
        # Use ThreadPoolExecutor for concurrent image downloads
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=cli_args.max_workers
        ) as executor:
            # List to store futures for monitoring completion
            futures = []

            # Iterate over remaining URLs and image names
            for url, filename in zip(remaining_urls, image_names):
                # Determine the appropriate image source
                source = determine_source(url)

                # Get the corresponding img_params
                img_params = get_img_params_for_source(source)

                # Pack parameters for download_image_wrapper
                params = (session, url, img_params, save_dir, filename, pbar)

                # Submit the download task to the executor
                futures.append(executor.submit(download_image_wrapper, params))

            # Wait for all futures to complete
            concurrent.futures.wait(futures)

    # Display a completion message
    print("Download completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="The path to .csv file where image URLs are stored.",
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
