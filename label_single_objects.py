import json
import tkinter as tk
import tkinter.messagebox

from PIL import Image, ImageTk


class ImageFilterGUI:
    def __init__(self,
                 image_paths,
                 save_dir: str = "./labeled_images.json"
                 ):
        self.image_paths = image_paths
        self.labeled_images_file = save_dir

        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=600, height=600)
        self.canvas.pack()
        self.canvas.create_text(300, 20, text="Does this image show a single item? Yes(y) or No(n)",
                                font="Helvetica 15 bold", anchor=tk.CENTER)
        self.keep_button = tk.Button(self.root, text="Yes (y)", command=self.keep_image, bg="green")
        self.keep_button.pack(side=tk.LEFT, padx=10)
        self.discard_button = tk.Button(self.root, text="No (n)", command=self.discard_image, bg="red")
        self.discard_button.pack(side=tk.LEFT)
        self.back_button = tk.Button(self.root, text="Back (b)", command=self.back_to_previous, bg="yellow")
        self.back_button.pack(side=tk.LEFT, padx=10)
        self.save_button = tk.Button(self.root, text="Save & Quit (s)", command=self.save_and_quit, bg="lightblue")
        self.save_button.pack(side=tk.RIGHT, padx=10)
        self.current_image_index = 0
        self.images_to_keep = []
        self.images_to_discard = []
        self.previous_image_selection = None
        self.root.bind("<Key>", self.handle_key_press)

    def display_image(self):
        image_path = self.image_paths[self.current_image_index]
        image = Image.open(image_path)
        image.thumbnail((600, 600))
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 40, anchor=tk.NW, image=photo)
        self.canvas.image = photo
        # Show number of images labeled
        self.canvas.create_text(550, 50,
                                text=f"{self.current_image_index + 1} / {len(self.image_paths)}",
                                font="Helvetica 15 bold", anchor=tk.CENTER, fill="red", width=400)

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
        data = {"images_to_keep": self.images_to_keep,
                "images_to_discard": self.images_to_discard}

        if os.path.exists(self.labeled_images_file):
            with open(self.labeled_images_file, 'r+') as f:
                labeled_images_data = json.load(f)
                labeled_images_data['images_to_keep'].extend(data['images_to_keep'])
                labeled_images_data['images_to_discard'].extend(data['images_to_discard'])
                f.seek(0)  # Move the file pointer to the beginning
                json.dump(labeled_images_data, f)
        else:
            with open(self.labeled_images_file, 'w') as f:
                json.dump(data, f)

        # Show stats
        tk.messagebox.showinfo("Message",
                               f"Filtered a total of {nb_of_kept_images + nb_of_discarded_images} images: "
                               f"{nb_of_kept_images} single object + {nb_of_discarded_images} not-single object!"
                               f"\nResponses saved to: ")

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
    import os, glob, logging

    def get_image_paths():
        image_dir = "/home/rizavelioglu/work/data/adidas/images/"
        image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))[:5]

        # Filter out already labeled images, if any
        labeled_images_file = "./labeled_images.json"
        if os.path.exists(labeled_images_file):
            logging.info("Filtering already labeled images...")
            image_paths = filter_image_paths(image_paths, labeled_images_file)

        return image_paths

    def filter_image_paths(image_paths, labeled_images_file):
        # Read labeled image paths from labeled_images.json
        with open(labeled_images_file, 'r') as f:
            labeled_images_data = json.load(f)

        already_labeled_images = labeled_images_data["images_to_keep"] + labeled_images_data["images_to_discard"]

        # Filter image paths based on labeled images
        filtered_paths = [path for path in image_paths if path not in already_labeled_images]

        return filtered_paths

    image_paths = get_image_paths()
    # Check if there are images to be labeled
    if image_paths:
        gui = ImageFilterGUI(image_paths)
        gui.run()
    else:
        print("All images are labeled! Quitting...")
