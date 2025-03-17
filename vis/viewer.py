import tkinter as tk
from tkinter import messagebox
import os
import random
from PIL import Image, ImageTk
from glob import glob
import arguebuf as ab
from render import render, RenderMethod
from pathlib import Path
import shutil
import multiprocessing
import queue
import threading
import time
import signal
import sys


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
        self.favourites_dir = "favourites"
        self.pregenerated_graphs = set()  # Track which graphs have been pre-generated
        self.pregeneration_queue = queue.Queue()  # Queue for graphs to pre-generate
        self.pregeneration_pool = None  # Will hold our process pool
        self.pregeneration_active = False
        self.buffer_size = 100  # Number of graphs to keep pre-generated

        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.favourites_dir, exist_ok=True)

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

        # Configure grid to make columns equal width
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.columnconfigure(1, weight=1)

        # Left frame for main image
        self.left_frame = tk.Frame(
            self.content_frame, bg="white", width=400, height=300
        )
        self.left_frame.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
        self.left_frame.pack_propagate(False)

        self.main_image_label = tk.Label(
            self.left_frame,
            text="Placeholder for PIL Image",
            bg="white",
            font=("Arial", 12),
        )
        self.main_image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Right frame for smaller images
        self.right_frame = tk.Frame(self.content_frame, bg="#a0a0a0", width=400)
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        # Create three placeholder frames on the right
        self.small_image_labels = []
        for i in range(3):
            frame = tk.Frame(self.right_frame, bg="white", width=400, height=90)
            frame.pack(pady=(0 if i > 0 else 0, 10), fill=tk.X)
            frame.pack_propagate(False)

            label = tk.Label(
                frame, text="Placeholder for PIL Image", bg="white", font=("Arial", 10)
            )
            label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

            self.small_image_labels.append(label)

        # Navigation buttons at the bottom
        self.button_frame = tk.Frame(root, bg="#a0a0a0")
        self.button_frame.pack(pady=(0, 20))

        # Previous button
        self.prev_button = tk.Button(
            self.button_frame,
            text="Previous",
            bg="#4a75e6",
            fg="white",
            width=15,
            command=lambda: self.navigate_graphs(-1),
        )
        self.prev_button.pack(side=tk.LEFT, padx=10)

        # Next button
        self.next_button = tk.Button(
            self.button_frame,
            text="Next",
            bg="#4a75e6",
            fg="white",
            width=15,
            command=lambda: self.navigate_graphs(1),
        )
        self.next_button.pack(side=tk.LEFT, padx=10)

        # Favourite button
        self.fav_button = tk.Button(
            self.button_frame,
            text="Favourite",
            bg="#4a75e6",
            fg="white",
            width=15,
            command=self.mark_as_favourite,
        )
        self.fav_button.pack(side=tk.LEFT, padx=10)

        # Bind arrow keys for navigation
        self.root.bind("<Left>", self.navigate_left)
        self.root.bind("<Right>", self.navigate_right)

        # Set up proper window closing behavior
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start the pregeneration process
        self.start_pregeneration()

        # Initialize with a random graph
        self.load_random_graphs()

        # set focus
        self.root.focus_force()

        # Set up periodic check for pregeneration status
        self.root.after(1000, self.check_pregeneration_status)

    def navigate_left(self, event):
        """Navigate to the previous graph"""
        self.navigate_graphs(-1)

    def navigate_right(self, event):
        """Navigate to the next graph"""
        self.navigate_graphs(1)

    def load_random_graphs(self):
        """
        Load a list of random graph paths.
        In a real application, this would fetch actual graph data.
        For this example, we'll simulate with dummy paths.
        """
        # This is a placeholder. In your actual implementation,
        # you would load real graph paths from your data source
        base_path = "../../arguebase-public"
        files = glob(os.path.join(base_path, "**/format=*/*.json"))
        random.shuffle(files)
        self.graph_paths.append(files[0])
        self.current_graph_index = len(self.graph_paths) - 1
        self.load_current_graph()

        # Queue up more graphs for pregeneration
        self.queue_more_graphs(files[1 : self.buffer_size + 1])

    def queue_more_graphs(self, graph_paths):
        """Add more graphs to the pregeneration queue"""
        for path in graph_paths:
            if path not in self.pregenerated_graphs:
                self.pregeneration_queue.put(path)
                self.pregenerated_graphs.add(path)

    def load_current_graph(self):
        """Load the current graph and its associated images"""
        if not self.graph_paths:
            return

        current_path = self.graph_paths[self.current_graph_index]
        graph_name = os.path.basename(current_path)

        if not os.path.exists(
            os.path.join(self.temp_dir, f"graphviz_{graph_name}.png")
        ):
            self.generate_graph_images(current_path)

        self.load_images_for_current_graph()

        self.path_label.config(text=f"Original Graph Path: {current_path}")

    def generate_graph_images(self, graph_path):
        """Generate images for a specific graph path"""
        print(f"Generating images for {graph_path}")
        graph = ab.load.file(graph_path)
        graph_name = os.path.basename(graph_path)
        ab.render.graphviz(
            ab.dump.graphviz(graph),
            os.path.join(self.temp_dir, f"graphviz_{graph_name}.png"),
        )
        render(
            graph_path,
            Path(os.path.join(self.temp_dir, f"srip2_{graph_name}.png")),
            RenderMethod.SRIP2,
        )
        render(
            graph_path,
            Path(os.path.join(self.temp_dir, f"logical_{graph_name}.png")),
            RenderMethod.LOGICAL,
        )
        render(
            graph_path,
            Path(os.path.join(self.temp_dir, f"treemap_{graph_name}.png")),
            RenderMethod.TREEMAP,
        )

    def load_images_for_current_graph(self):
        """
        Load images for the current graph.
        This is a placeholder method that you'll replace with your actual implementation.
        """
        # Clear any existing images in temp directory
        current_path = self.graph_paths[self.current_graph_index]
        print(current_path)
        graph_name = os.path.basename(current_path)

        self.load_image(
            os.path.join(self.temp_dir, f"graphviz_{graph_name}.png"),
            self.main_image_label,
            400,
            300,
        )

        small_img_paths = [
            os.path.join(self.temp_dir, f"srip2_{graph_name}.png"),
            os.path.join(self.temp_dir, f"logical_{graph_name}.png"),
            os.path.join(self.temp_dir, f"treemap_{graph_name}.png"),
        ]

        for i, path in enumerate(small_img_paths):
            self.load_image(path, self.small_image_labels[i], 400, 90)

    def navigate_graphs(self, direction):
        """Navigate to the next or previous graph"""
        if not self.graph_paths:
            return

        new_index = self.current_graph_index + direction

        # Wrap around if we reach the end or beginning
        if new_index >= len(self.graph_paths):
            self.load_random_graphs()
        if new_index < 0:
            return
        else:
            print(new_index)
            self.current_graph_index = new_index
            self.load_current_graph()

    def load_image(self, path, target_label, width, height):
        """Load an image from path and display it in the target label"""
        try:
            img = Image.open(path)
            img = img.resize((width, height))
            photo = ImageTk.PhotoImage(img)
            target_label.config(image=photo, text="")
            target_label.image = photo  # Keep a reference
        except Exception as e:
            target_label.config(text=f"Error loading image")
            print(f"Error loading image: {e}")

    def mark_as_favourite(self):
        """Mark the current graph as a favourite by copying it to the favourites folder"""
        if not self.graph_paths:
            return

        current_path = self.graph_paths[self.current_graph_index]
        graph_name = os.path.basename(current_path)

        # Copy the original graph file
        shutil.copy(current_path, os.path.join(self.favourites_dir, graph_name))

        # Copy the associated images
        img_paths = [
            os.path.join(self.temp_dir, f"graphviz_{graph_name}.png"),
            os.path.join(self.temp_dir, f"srip2_{graph_name}.png"),
            os.path.join(self.temp_dir, f"logical_{graph_name}.png"),
            os.path.join(self.temp_dir, f"treemap_{graph_name}.png"),
        ]

        for img_path in img_paths:
            if os.path.exists(img_path):
                shutil.copy(
                    img_path,
                    os.path.join(self.favourites_dir, os.path.basename(img_path)),
                )

        messagebox.showinfo("Success", f"Graph '{graph_name}' marked as favourite!")

    def start_pregeneration(self):
        """Start the background pregeneration process"""
        if self.pregeneration_active:
            return

        self.pregeneration_active = True

        # Initialize multiprocessing with 'spawn' method for better process termination
        if sys.platform != "win32":  # Not needed on Windows as it defaults to 'spawn'
            multiprocessing.set_start_method("spawn", force=True)

        # Create a process pool
        self.pregeneration_pool = multiprocessing.Pool(
            processes=max(1, multiprocessing.cpu_count() - 1),
            initializer=self.worker_init,
        )

        # Start a thread to manage the pregeneration queue
        self.pregeneration_thread = threading.Thread(
            target=self.pregeneration_manager, daemon=True
        )
        self.pregeneration_thread.start()

    @staticmethod
    def worker_init():
        """Initialize worker processes to handle signals properly"""
        # Set up signal handler for worker processes
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def pregeneration_manager(self):
        """Manages the pregeneration of graphs in the background"""
        while self.pregeneration_active:
            try:
                # Get a graph path from the queue (with timeout to check active flag)
                try:
                    graph_path = self.pregeneration_queue.get(timeout=0.5)
                except queue.Empty:
                    # If queue is empty, continue the loop to check active flag
                    continue

                # Check if the graph is already generated
                graph_name = os.path.basename(graph_path)
                if os.path.exists(
                    os.path.join(self.temp_dir, f"graphviz_{graph_name}.png")
                ):
                    continue

                # Skip if we're no longer active
                if not self.pregeneration_active:
                    break

                # Submit the graph generation task to the process pool
                self.pregeneration_pool.apply_async(
                    self.pregenerate_graph_images, args=(graph_path, self.temp_dir)
                )

                # Add the graph to our list of available graphs
                if graph_path not in self.graph_paths:
                    self.graph_paths.append(graph_path)

            except Exception as e:
                print(f"Error in pregeneration manager: {e}")
                if not self.pregeneration_active:
                    break
                time.sleep(1)  # Avoid tight loop in case of repeated errors

    @staticmethod
    def pregenerate_graph_images(graph_path, temp_dir):
        """Static method to generate graph images in a separate process"""
        try:
            graph = ab.load.file(graph_path)
            graph_name = os.path.basename(graph_path)

            # Generate all the required images
            ab.render.graphviz(
                ab.dump.graphviz(graph),
                os.path.join(temp_dir, f"graphviz_{graph_name}.png"),
            )
            render(
                graph_path,
                Path(os.path.join(temp_dir, f"srip2_{graph_name}.png")),
                RenderMethod.SRIP2,
            )
            render(
                graph_path,
                Path(os.path.join(temp_dir, f"logical_{graph_name}.png")),
                RenderMethod.LOGICAL,
            )
            render(
                graph_path,
                Path(os.path.join(temp_dir, f"treemap_{graph_name}.png")),
                RenderMethod.TREEMAP,
            )

            print(f"Pre-generated images for {graph_name}")
            return True
        except Exception as e:
            print(f"Error pre-generating images for {graph_path}: {e}")
            return False

    def check_pregeneration_status(self):
        """Periodically check if we need to queue more graphs for pregeneration"""
        if not self.pregeneration_active:
            return

        try:
            # Check if we need to queue more graphs
            base_path = "../../arguebase-public"
            if self.pregeneration_queue.qsize() < 20:  # If queue is getting low
                files = glob(os.path.join(base_path, "**/format=*/*.json"))
                random.shuffle(files)

                # Filter out graphs we've already processed
                new_files = [f for f in files if f not in self.pregenerated_graphs]

                # Queue up to buffer_size new graphs
                self.queue_more_graphs(new_files[: self.buffer_size])
        except Exception as e:
            print(f"Error checking pregeneration status: {e}")

        # Schedule the next check only if we're still active
        if self.pregeneration_active:
            self.root.after(5000, self.check_pregeneration_status)

    def on_closing(self):
        """Clean up resources when the application is closing"""
        print("Application closing, terminating background processes...")

        # Set flag to stop the pregeneration thread
        self.pregeneration_active = False

        # Terminate the process pool immediately
        if self.pregeneration_pool:
            try:
                # Terminate all worker processes immediately
                self.pregeneration_pool.terminate()

                # Small timeout for cleanup (non-blocking)
                self.pregeneration_pool.join()

                print("Process pool terminated")
            except Exception as e:
                print(f"Error terminating process pool: {e}")

        # Destroy the window immediately
        self.root.destroy()

        # Force exit if needed (as a last resort)
        # This ensures we don't wait for any lingering processes
        print("Application closed")


if __name__ == "__main__":
    # Set up proper signal handling for the main process
    if sys.platform != "win32":  # Not needed on Windows
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    root = tk.Tk()
    app = ImageViewerApp(root)
    root.mainloop()
