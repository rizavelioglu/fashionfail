import argparse
import json
import os
import pickle
import time
import tkinter as tk
import tkinter.messagebox
from pathlib import Path

import cv2
import pandas as pd
import PIL
import supervision as sv
from loguru import logger
from PIL import Image, ImageTk

from fashionfail.utils import load_categories


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
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
        "--label_anns_file",
        type=str,
        required=True,
        help="Full path to the label annotations file.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Full path to the file which stores labeled data.",
    )

    return parser.parse_args()


class ImageFilterGUI:
    def __init__(self, image_names, images_dir: Path, out_path: Path):
        self.images_dir = images_dir
        self.image_paths = image_names
        self.labeled_images_file = out_path
        self.nb_of_images = len(self.image_paths)
        self.start_time = time.time()

        self.root = tk.Tk(className="Labeling Tool")
        self.canvas = tk.Canvas(self.root, width=1500, height=600)
        self.canvas.pack()
        self.canvas.create_text(
            620,
            10,
            text="Are the label, mask, and bbox annotations accurate?",
            font="Verdana 10 bold",
            anchor=tk.CENTER,
        )
        self.keep_button = tk.Button(
            self.root, text="Yes (y)", command=self.keep_image, bg="green"
        )
        self.keep_button.pack(side=tk.LEFT, padx=10)
        self.discard_button = tk.Button(
            self.root, text="No (n)", command=self.discard_image, bg="red"
        )
        self.discard_button.pack(side=tk.LEFT)
        self.back_button = tk.Button(
            self.root, text="Back (b)", command=self.back_to_previous, bg="yellow"
        )
        self.back_button.pack(side=tk.LEFT, padx=10)
        self.save_button = tk.Button(
            self.root,
            text="Save & Quit (s)",
            command=self.save_and_quit,
            bg="lightblue",
        )
        self.save_button.pack(side=tk.RIGHT, padx=10)
        self.current_image_index = 0
        self.images_to_keep: list[str] = []
        self.images_to_discard: list[str] = []
        self.previous_image_selection = None
        self.root.bind("<Key>", self.handle_key_press)

    def display_image(self):
        image_path = self.image_paths[self.current_image_index]
        # Show image name on the canvas
        self.canvas.delete("image_name")
        self.canvas.create_text(
            620,
            30,
            text=f"{image_path}",
            font="Verdana 9",
            anchor=tk.CENTER,
            tags="image_name",
        )

        # Show original image with its label
        image = Image.open(self.images_dir / image_path)
        image = image.resize((500, 500))
        img_org = ImageTk.PhotoImage(image)

        class_id = DF_LABELS[DF_LABELS.image_name == image_path].class_id.values[0]
        class_name = CATEGORY_ID_TO_NAME[class_id]
        label1 = tk.Label(image=img_org, text=f"Label: {class_name}", compound="bottom")
        label1.image = img_org
        label1.place(x=0, y=60)

        # Show annotated images (mask)
        img_ann_bbox, img_ann_mask = get_annotated_images(
            images_dir=self.images_dir, image_path=image_path
        )
        img_ann_mask = img_ann_mask.resize((500, 500))
        img_ann_mask = ImageTk.PhotoImage(img_ann_mask)

        label3 = tk.Label(
            image=img_ann_mask, text="Annotated Image (mask):", compound="bottom"
        )
        label3.image = img_ann_mask
        label3.place(x=500, y=60)

        # Show annotated images (bbox)
        img_ann_bbox = img_ann_bbox.resize((500, 500))
        img_ann_bbox = ImageTk.PhotoImage(img_ann_bbox)

        label2 = tk.Label(
            image=img_ann_bbox, text="Annotated Image (bbox):", compound="bottom"
        )
        label2.image = img_ann_bbox
        label2.place(x=1000, y=60)

        # Show labeling statistics in the frame
        self.display_stats()

    def display_stats(self):
        # Show number of images labeled
        stat1 = tk.Label(
            text=f"{self.current_image_index + 1} / {self.nb_of_images}",
            compound="bottom",
            fg="red",
        )
        stat1.place(x=1000, y=25)

        # Show the speed of labeling: image per second
        label_speed = self.current_image_index / (time.time() - self.start_time)
        stat2 = tk.Label(text=f"speed (im/s): {label_speed:.2f}", fg="red")
        stat2.place(x=30, y=25)

        # Show expected time left
        if self.current_image_index == 0:
            label_speed = 0.0001
        seconds = (self.nb_of_images - self.current_image_index) / label_speed
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        stat3 = tk.Label(text=f"time left: {hours:.0f}h:{minutes:.0f}m", fg="red")
        stat3.place(x=30, y=40)

    def keep_image(self):
        self.images_to_keep.append(self.image_paths[self.current_image_index])
        self.previous_image_selection = "keep"
        self.next_image()

    def discard_image(self):
        self.images_to_discard.append(self.image_paths[self.current_image_index])
        self.previous_image_selection = "discard"
        self.next_image()

    def back_to_previous(self):
        if self.previous_image_selection == "keep":
            self.images_to_keep.pop()
        elif self.previous_image_selection == "discard":
            self.images_to_discard.pop()
        self.current_image_index -= 1
        if self.current_image_index >= 0:
            self.display_image()

    def next_image(self):
        self.current_image_index += 1
        if self.current_image_index < len(self.image_paths):
            self.display_image()
        else:
            self.save_and_quit()

    def save_and_quit(self):
        nb_of_kept_images = len(self.images_to_keep)
        nb_of_discarded_images = len(self.images_to_discard)
        # Save the responses
        data = {
            "images_to_keep": self.images_to_keep,
            "images_to_discard": self.images_to_discard,
        }

        if os.path.exists(self.labeled_images_file):
            with open(self.labeled_images_file, "r+") as f:
                labeled_images_data = json.load(f)
                labeled_images_data["images_to_keep"].extend(data["images_to_keep"])
                labeled_images_data["images_to_discard"].extend(
                    data["images_to_discard"]
                )
                f.seek(0)  # Move the file pointer to the beginning
                json.dump(labeled_images_data, f)
        else:
            with open(self.labeled_images_file, "w") as f:
                json.dump(data, f)

        # Show stats
        nb_of_labeled_images = nb_of_kept_images + nb_of_discarded_images
        tk.messagebox.showinfo(
            "Message",
            f"Filtered a total of {nb_of_labeled_images} images: "
            f"{nb_of_kept_images} single object + {nb_of_discarded_images} not-single object!"
            f"\nLabel speed: {nb_of_labeled_images / (time.time() - self.start_time):.2f} image per second"
            f"\nResponses saved to: {self.labeled_images_file}",
        )

        self.root.quit()

    def handle_key_press(self, event):
        if event.char.lower() == "y":
            self.keep_image()
        elif event.char.lower() == "n":
            self.discard_image()
        elif event.char.lower() == "b":
            self.back_to_previous()
        elif event.char.lower() == "s":
            self.save_and_quit()

    def run(self):
        self.display_image()
        self.root.mainloop()


if __name__ == "__main__":
    # Parse cli arguments
    args = get_cli_args()
    IMAGES_DIR = Path(args.images_dir)
    OUT_FILE = Path(args.out_path)
    GT_SAVE_DIR = Path(args.anns_dir)

    # Load label annotations and category names
    DF_LABELS = pd.read_csv(args.label_anns_file)
    CATEGORY_ID_TO_NAME = load_categories()

    def get_image_names():
        # Get only those picture names that have ground truth
        image_names = [f.stem + ".jpg" for f in GT_SAVE_DIR.iterdir()]

        # Filter out already labeled images, if any
        if os.path.exists(OUT_FILE):
            logger.info("Filtering already labeled images...")
            image_names = filter_image_names(image_names, OUT_FILE)

        image_names.sort()

        return image_names

    def filter_image_names(all_image_names, labeled_images_file):
        # Read labeled image paths from labeled_images.json
        with open(labeled_images_file) as f:
            labeled_images_data = json.load(f)

        already_labeled_images = (
            labeled_images_data["images_to_keep"]
            + labeled_images_data["images_to_discard"]
        )

        # Filter image paths based on labeled images
        filtered_paths = [
            path for path in all_image_names if path not in already_labeled_images
        ]

        return filtered_paths

    def get_annotated_images(images_dir, image_path):
        image_name = image_path.replace(".jpg", "")
        # Load annotations
        with open(f"{GT_SAVE_DIR}/{image_name}.pkl", "rb") as f:
            detections = pickle.load(f)

        box_annotator = sv.BoxAnnotator(text_scale=1, thickness=4)
        # mask_annotator = sv.MaskAnnotator()

        image = cv2.imread(str(images_dir) + f"/{image_path}")

        labels = [
            f"an object: {confidence:0.2f}"
            for _, _, confidence, class_id, _ in detections
        ]
        annotated_image_mask = PIL.Image.fromarray(detections.mask.squeeze())
        annotated_image_bbox = box_annotator.annotate(
            scene=image.copy(), detections=detections, labels=labels
        )

        return (
            PIL.Image.fromarray(annotated_image_bbox[:, :, ::-1]),
            annotated_image_mask,
        )

    image_filenames = get_image_names()
    # Check if there are images to be labeled
    if image_filenames:
        gui = ImageFilterGUI(
            image_filenames,
            images_dir=IMAGES_DIR,
            out_path=OUT_FILE,
        )
        gui.run()
    else:
        logger.info("All images are labeled! Quitting...")
