import argparse
import concurrent.futures
import json
import os
import urllib.request
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm


def download_images(cli_args) -> None:
    """Download images from their URLs in and save them to `cli_args.save_dir`.

    Multithreading is used to speed up the process.

    Notes
    -----
    See the following for more info on `concurrent.futures`, where the code is taken from:
        https://stackoverflow.com/questions/51601756/use-tqdm-with-concurrent-futures
    See the following for more info on Multithreading and Multiprocessing:
        https://www.toptal.com/python/beginners-guide-to-concurrency-and-parallelism-in-python
    """
    global map_fn

    def map_fn(img_url):
        # Add transformations to apply to images
        # See: https://cloudinary.com/documentation/transformation_reference
        img_params = "images/f_jpg,q_auto,fl_lossy,c_fill,g_auto/"
        img_url = img_url.replace("images/", img_params)
        try:
            urllib.request.urlretrieve(
                img_url, f"{cli_args.save_dir}/{Path(img_url).name}"
            )
        except FileNotFoundError:
            logger.exception(f"Could not download: {img_url}")

    with open(cli_args.json_path) as f:
        img_info = json.load(f)

    img_urls = img_info["images"]
    img_info["source_name"]

    # Filter out already downloaded image URLs
    already_downloaded_urls = os.listdir(cli_args.save_dir)
    df_all_img_urls = pd.DataFrame(img_urls, columns=["url"])
    df_all_img_urls["filename"] = df_all_img_urls["url"].apply(lambda x: Path(x).name)
    remaining_urls = df_all_img_urls[
        ~df_all_img_urls.filename.isin(already_downloaded_urls)
    ].url.tolist()

    with tqdm(total=len(remaining_urls)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=cli_args.max_workers
        ) as executor:
            futures = {executor.submit(map_fn, url): url for url in remaining_urls}
            results = {}
            for future in concurrent.futures.as_completed(futures):
                arg = futures[future]
                results[arg] = future.result()
                pbar.update(1)


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
