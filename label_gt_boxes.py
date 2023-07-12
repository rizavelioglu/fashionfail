import json
import os
import time
import tkinter as tk
import tkinter.messagebox
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from PIL import ImageTk
from torchvision.io import read_image
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes


def yxyx_to_xywh(boxes):
    """Converts boxes from ymin, xmin, ymax, xmax to xmin, ymin, width, height.
    Args:
    boxes: a numpy array whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    Returns:
    boxes: a numpy array whose shape is the same as `boxes` in new format.
    Raises:
    ValueError: If the last dimension of boxes is not 4.
    """
    if boxes.shape[-1] != 4:
        raise ValueError(f"boxes.shape[-1] is {boxes.shape[-1]:d}, but must be 4.")

    boxes_ymin = boxes[..., 0]
    boxes_xmin = boxes[..., 1]
    boxes_width = boxes[..., 3] - boxes[..., 1]
    boxes_height = boxes[..., 2] - boxes[..., 0]
    new_boxes = np.stack([boxes_xmin, boxes_ymin, boxes_width, boxes_height], axis=-1)

    return new_boxes


def process_boxes(boxes):
    # Convert boxes to "xywh" format,
    # TODO: convert boxes from 'yxyx' directly to 'xyxy'. no need to use box_convert.
    boxes = yxyx_to_xywh(boxes)
    boxes = box_convert(torch.tensor(boxes), in_fmt="xywh", out_fmt="xyxy")
    return boxes


def get_fashionpedia_cat_mapping():
    # Parse Fashionpedia categories.
    with open(FASHIONPEDIA_CAT_DIR) as fp:
        categories = json.load(fp)

    category_id_to_name = {d["id"]: d["name"] for d in categories}

    return category_id_to_name


class ImageFilterGUI:
    def __init__(self, df_preds, images_dir: Path, labels_dir: Path):
        self.images_dir = images_dir
        self.df_preds = df_preds
        self.labeled_images_file = labels_dir
        self.nb_of_images = self.df_preds.shape[0]
        self.category_id_to_name = get_fashionpedia_cat_mapping()
        self.start_time = time.time()

        self.root = tk.Tk(className="Labeling Tool")
        self.canvas = tk.Canvas(self.root, width=600, height=600)
        self.canvas.pack()
        self.canvas.create_text(
            300,
            10,
            text="Is the box correctly annotated?",
            font="Verdana 10 bold",
            anchor=tk.CENTER,
        )
        self.keep_button = tk.Button(
            self.root, text="Yes (y)", command=self.keep_box, bg="green"
        )
        self.keep_button.pack(side=tk.LEFT, padx=10)
        self.discard_button = tk.Button(
            self.root, text="No (n)", command=self.discard_box, bg="red"
        )
        self.discard_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(
            self.root,
            text="Save & Quit (s)",
            command=self.save_and_quit,
            bg="lightblue",
        )
        self.save_button.pack(side=tk.RIGHT, padx=10)
        self.current_image_index = 0
        self.current_box_index = 0
        self.root.bind("<Key>", self.handle_key_press)

    def display_image(self):
        pred = self.df_preds.iloc[self.current_image_index]
        image = read_image(str(self.images_dir / pred["image_file"]))

        bboxes = process_boxes(pred["boxes"][self.current_box_index])
        image_with_bbox = draw_bounding_boxes(
            image, torch.tensor([bboxes.tolist()]), colors="red", width=12, fill=False
        )
        img = F.to_pil_image(image_with_bbox)
        img = img.resize((500, 500))
        img = ImageTk.PhotoImage(img)

        # Show image and its filename
        label1 = tk.Label(
            image=img, text=f"File: {pred['image_file']}", compound="bottom"
        )
        label1.image = img
        label1.place(x=0, y=60)

        # Show the predicted class/category name & its confidence score
        pred_label = self.category_id_to_name[
            pred["classes"][self.current_box_index] - 1
        ]
        pred_conf = pred["scores"][self.current_box_index]
        label2 = tk.Label(
            text=f"Class: {pred_label} ({pred_conf:.2f})", compound="bottom", fg="blue"
        )
        label2.place(x=200, y=80)

        # Show labeling statistics in the frame
        self.display_stats()

    def display_stats(self):
        # Show number of images labeled
        stat1 = tk.Label(
            text=f"{self.current_image_index + 1} / {self.nb_of_images}",
            compound="bottom",
            fg="red",
        )
        stat1.place(x=400, y=25)

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

    def keep_box(self):
        self.df_preds.at[self.current_image_index, "matched"].append(1)
        self.next_box()

    def discard_box(self):
        self.df_preds.at[self.current_image_index, "matched"].append(0)
        self.next_box()

    def next_image(self):
        self.current_image_index += 1
        self.current_box_index = 0
        if self.current_image_index < self.df_preds.shape[0]:
            self.display_image()
        else:
            self.save_and_quit()

    def next_box(self):
        self.current_box_index += 1
        if (
            self.current_box_index
            < self.df_preds.iloc[self.current_image_index]["boxes"].shape[0]
        ):
            self.display_image()
        else:
            self.next_image()

    def save_and_quit(self):
        # Merge with already labeled samples, if any
        if os.path.exists(self.labeled_images_file):
            with open(self.labeled_images_file, "r+") as f:
                labeled_images_data = json.load(f)
            df_old = pd.DataFrame(labeled_images_data)
            df_old = df_old[df_old["matched"].apply(lambda x: x != [])]
            self.df_preds = pd.concat([df_old, self.df_preds]).reset_index(drop=True)

        # Save the DataFrame
        self.df_preds.to_json(self.labeled_images_file)

        self.root.quit()

    def handle_key_press(self, event):
        if event.char.lower() == "y":
            self.keep_box()
        elif event.char.lower() == "n":
            self.discard_box()
        elif event.char.lower() == "s":
            self.save_and_quit()

    def run(self):
        self.display_image()
        self.root.mainloop()


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    BASE_DIR = Path("/home/rizavelioglu/work/data/fashionfail/")
    IMAGES_DIR = BASE_DIR / "images-sample"
    LABELS_DIR = BASE_DIR / "labeled_images_GT_boxes.json"
    PREDS_DIR = "/home/rizavelioglu/work/repos/tpu/models/official/detection/outputs/output-ff_sample.npy"
    FASHIONPEDIA_CAT_DIR = "/home/rizavelioglu/work/repos/segmentation/segmentation/visualization/categories.json"

    def load_model_preds():
        # Load model predictions
        preds = np.load(PREDS_DIR, allow_pickle=True)
        # Preprocessing
        df = pd.DataFrame.from_dict(list(preds))[
            ["image_file", "classes", "scores", "boxes"]
        ]
        df = df.reset_index(drop=True)
        df["image_file"] = df["image_file"].apply(lambda x: x.split("/")[1])
        # Add a new column where labeling results will be stored
        df["matched"] = np.empty((len(df), 0)).tolist()

        # Filter out samples where no predictions made
        df = df[df["classes"].apply(lambda x: x.size != 0)]

        # Filter out already labeled images, if any
        if os.path.exists(LABELS_DIR):
            with open(LABELS_DIR, "r+") as f:
                labeled_images_data = json.load(f)
            # Filter not labeled image filenames
            df_labeled = pd.DataFrame(labeled_images_data)
            df_labeled = df_labeled[df_labeled["matched"].apply(lambda x: x != [])]
            # Select image filenames to be labeled
            df = df[~df["image_file"].isin(df_labeled["image_file"])]

        return df.reset_index(drop=True)

    df_predictions = load_model_preds()

    # Launch app
    gui = ImageFilterGUI(
        df_predictions,
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
    )
    gui.run()
