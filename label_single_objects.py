import json
import time
import tkinter as tk
import tkinter.messagebox
from pathlib import Path

from PIL import Image, ImageTk


class ImageFilterGUI:
    def __init__(self, image_names, images_dir: Path, labels_dir: Path):
        self.images_dir = images_dir
        self.image_paths = image_names
        self.labeled_images_file = labels_dir
        self.nb_of_images = len(self.image_paths)
        self.start_time = time.time()

        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=600, height=600)
        self.canvas.pack()
        self.canvas.create_text(
            300,
            20,
            text="Does this image show a single item? Yes(y) or No(n)",
            font="Helvetica 15 bold",
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
        image = Image.open(self.images_dir / image_path)
        image.thumbnail((600, 600))
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 40, anchor=tk.NW, image=photo)
        self.canvas.image = photo
        # Show number of images labeled
        self.canvas.create_text(
            500,
            55,
            text=f"{self.current_image_index + 1} / {self.nb_of_images}",
            anchor=tk.SW,
            fill="red",
            width=400,
        )
        # Show the speed of labeling: image per second
        label_speed = self.current_image_index / (time.time() - self.start_time)
        self.canvas.create_text(
            30,
            55,
            anchor=tk.SW,
            text=f"speed: {label_speed:.2f}",
            fill="red",
        )
        # Show expected time left
        if self.current_image_index == 0:
            label_speed = 0.0001
        seconds = (self.nb_of_images - self.current_image_index) / label_speed
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        self.canvas.create_text(
            30,
            70,
            anchor=tk.SW,
            text=f"time left: {hours:.0f}h:{minutes:.0f}m",
            fill="red",
        )

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
    import logging
    import os

    BASE_DIR = Path("/home/rizavelioglu/work/data/fashionfail/")
    IMAGES_DIR = BASE_DIR / "images"
    LABELS_DIR = BASE_DIR / "labeled_images.json"

    def get_image_names():
        image_names = [f.name for f in IMAGES_DIR.iterdir()]

        # Filter out already labeled images, if any
        if os.path.exists(LABELS_DIR):
            logging.warning("Filtering already labeled images...")
            image_names = filter_image_names(image_names, LABELS_DIR)

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

    image_filenames = get_image_names()
    # Check if there are images to be labeled
    if image_filenames:
        gui = ImageFilterGUI(
            image_filenames,
            images_dir=IMAGES_DIR,
            labels_dir=LABELS_DIR,
        )
        gui.run()
    else:
        logging.warning("All images are labeled! Quitting...")
