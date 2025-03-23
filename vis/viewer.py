import tkinter as tk
import os
import random
from PIL import Image, ImageTk
import glob
import multiprocessing


class ImageViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Graph Viewer")
        self.root.geometry("1920x1080")
        self.root.configure(bg="#a0a0a0")  # Gray background

        # Initialize variables
        self.current_graph_index = 0
        self.graph_paths = []
        self.temp_dir = "visualized_graphs"

        # Path display at the top
        self.path_frame = tk.Frame(root, bg="white", height=30)
        self.path_frame.pack(fill=tk.X, padx=20, pady=(20, 10))

        self.path_label = tk.Label(
            self.path_frame,
            text="Original Graph Path: None",
            font=("Arial", 10),
            bg="white",
        )
        self.path_label.pack(pady=5)

        # Main content area
        self.content_frame = tk.Frame(root, bg="#a0a0a0")
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left frame for main image
        self.left_frame = tk.Frame(
            self.content_frame, bg="white", width=300, height=300
        )
        self.left_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.BOTH, expand=True)
        self.left_frame.pack_propagate(False)

        self.main_image_label = tk.Label(
            self.left_frame,
            text="Placeholder for PIL Image",
            bg="white",
            font=("Arial", 12),
        )
        self.main_image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Right frame for smaller images
        self.right_frame = tk.Frame(self.content_frame, bg="#a0a0a0", width=200)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Create three placeholder frames on the right
        self.small_image_labels = []
        for i in range(3):
            frame = tk.Frame(self.right_frame, bg="white", width=200, height=90)
            frame.pack(pady=(0 if i > 0 else 0, 10), fill=tk.X)
            frame.pack_propagate(False)

            label = tk.Label(
                frame, text="Placeholder for PIL Image", bg="white", font=("Arial", 10)
            )
            label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

            self.small_image_labels.append(label)

        # Button at the bottom
        self.button_frame = tk.Frame(root, bg="#a0a0a0")
        self.button_frame.pack(pady=(0, 20))

        self.explorer_button = tk.Button(
            self.button_frame,
            text="Open in Explorer",
            bg="#4a75e6",
            fg="white",
            width=20,
            command=self.open_in_explorer,
        )
        self.explorer_button.pack()

        # Bind arrow keys for navigation
        self.root.bind("<Left>", lambda event: self.navigate_graphs(-1))
        self.root.bind("<Right>", lambda event: self.navigate_graphs(1))

        # Initialize with a random graph
        self.load_random_graphs()

    def load_random_graphs(self):
        """
        Load a list of random graph paths.
        In a real application, this would fetch actual graph data.
        For this example, we'll simulate with dummy paths.
        """
        # This is a placeholder. In your actual implementation,
        # you would load real graph paths from your data source
        base_path = "/home/graphs"
        self.graph_paths = [f"{base_path}/graph_{i}" for i in range(1, 11)]
        random.shuffle(self.graph_paths)
        self.current_graph_index = 0
        self.load_current_graph()

    def load_current_graph(self):
        """Load the current graph and its associated images"""
        if not self.graph_paths:
            return

        current_path = self.graph_paths[self.current_graph_index]
        self.path_label.config(text=f"Original Graph Path: {current_path}")

        # Simulate loading images for the current graph
        self.load_images_for_current_graph()

    def load_images_for_current_graph(self):
        """
        Load images for the current graph.
        This is a placeholder method that you'll replace with your actual implementation.
        """
        # Clear any existing images in temp directory
        for file in glob.glob(os.path.join(self.temp_dir, "*.png")):
            try:
                os.remove(file)
            except:
                pass

        # For demonstration, we'll create placeholder colored rectangles
        # In your actual implementation, you would load real images

        # Create a main image (larger)
        main_img = Image.new(
            "RGB",
            (300, 300),
            color=(
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            ),
        )
        main_img_path = os.path.join(self.temp_dir, "main.png")
        main_img.save(main_img_path)

        # Create three smaller images
        small_img_paths = []
        for i in range(3):
            img = Image.new(
                "RGB",
                (200, 90),
                color=(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ),
            )
            path = os.path.join(self.temp_dir, f"small_{i}.png")
            img.save(path)
            small_img_paths.append(path)

        # Load the images into the GUI
        self.load_image(main_img_path, self.main_image_label, 300, 300)

        for i, path in enumerate(small_img_paths):
            self.load_image(path, self.small_image_labels[i], 200, 90)

    def navigate_graphs(self, direction):
        """Navigate to the next or previous graph"""
        if not self.graph_paths:
            return

        new_index = self.current_graph_index + direction

        # Wrap around if we reach the end or beginning
        if new_index >= len(self.graph_paths):
            new_index = 0
        elif new_index < 0:
            new_index = len(self.graph_paths) - 1

        self.current_graph_index = new_index
        self.load_current_graph()

    def load_image(self, path, target_label, width, height):
        """Load an image from path and display it in the target label"""
        try:
            img = Image.open(path)
            img = img.resize((width, height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            target_label.config(image=photo, text="")
            target_label.image = photo  # Keep a reference
        except Exception as e:
            target_label.config(text=f"Error loading image")
            print(f"Error loading image: {e}")

    def __del__(self):
        """Clean up temporary directory when the app is closed"""
        try:
            import shutil

            shutil.rmtree(self.temp_dir)
        except:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewerApp(root)
    root.mainloop()
