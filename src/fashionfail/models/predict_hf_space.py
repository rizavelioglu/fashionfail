import argparse
import base64
from pathlib import Path

import requests


def encode_image(image_path):
    """Encode an image file as a base64 string.

    Args:
        image_path (str): The path to the image file to encode.

    Returns:
        str: The base64-encoded string representation of the image data.
    """
    with open(image_path, "rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
    encoded_string = encoded_bytes.decode("utf-8")
    return encoded_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_img",
        type=Path,
        required=True,
        help="The path to the image file to run inference for.",
    )

    # Parse cli arguments.
    args = parser.parse_args()

    # Encode the image as a base64 string
    img_base64 = encode_image(args.path_to_img)

    # Make a POST request to the API
    response = requests.post(
        "https://rizavelioglu-fashion-segmentation.hf.space/run/predict",
        json={"data": ["data:image/jpeg;base64," + img_base64]},
    ).json()

    print(response["data"])
