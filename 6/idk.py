import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from sklearn.cluster import DBSCAN
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation Tool")
        self.root.geometry("1200x800")
        
        # Variables
        self.original_image = None
        self.segmented_image = None
        self.image_path = ""
        self.method_var = tk.StringVar(value="kmeans")
        
        # Create UI
        self.create_widgets()
        
    def create_widgets(self):
        # Left panel - controls
        control_frame = tk.Frame(self.root, width=300, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Method selection
        tk.Label(control_frame, text="Select Segmentation Method:").pack(pady=(10, 5))
        methods = [
            ("K-Means", "kmeans"),
            ("Mean Shift", "meanshift"),
            ("DBSCAN", "dbscan"),
            ("Active Contour", "activecontour")
        ]
        for text, method in methods:
            tk.Radiobutton(control_frame, text=text, variable=self.method_var, 
                          value=method, command=self.update_parameters).pack(anchor=tk.W)
        
        # Parameters frame
        self.param_frame = tk.Frame(control_frame)
        self.param_frame.pack(pady=10, fill=tk.X)
        
        # Default parameters (will be updated based on method)
        self.create_kmeans_params()
        
        # Load image button
        tk.Button(control_frame, text="Load Image", command=self.load_image).pack(pady=10)
        
        # Segment button
        tk.Button(control_frame, text="Segment Image", command=self.segment_image).pack(pady=10)
        
        # Save button
        tk.Button(control_frame, text="Save Result", command=self.save_image).pack(pady=10)
        
        # Right panel - images
        image_frame = tk.Frame(self.root, padx=10, pady=10)
        image_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        # Original image
        tk.Label(image_frame, text="Original Image").pack()
        self.original_label = tk.Label(image_frame)
        self.original_label.pack()
        
        # Segmented image
        tk.Label(image_frame, text="Segmented Image").pack(pady=(10, 0))
        self.segmented_label = tk.Label(image_frame)
        self.segmented_label.pack()
        
        # Active contour plot (hidden by default)
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=image_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack_forget()
    
    def create_kmeans_params(self):
        # Clear existing parameters
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        tk.Label(self.param_frame, text="K-Means Parameters").pack()
        
        tk.Label(self.param_frame, text="Number of clusters (k):").pack()
        self.k_entry = tk.Entry(self.param_frame)
        self.k_entry.insert(0, "3")
        self.k_entry.pack()
        
        tk.Label(self.param_frame, text="Max iterations:").pack()
        self.max_iter_entry = tk.Entry(self.param_frame)
        self.max_iter_entry.insert(0, "100")
        self.max_iter_entry.pack()
        
        tk.Label(self.param_frame, text="Epsilon:").pack()
        self.epsilon_entry = tk.Entry(self.param_frame)
        self.epsilon_entry.insert(0, "0.2")
        self.epsilon_entry.pack()
    
    def create_meanshift_params(self):
        # Clear existing parameters
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        tk.Label(self.param_frame, text="Mean Shift Parameters").pack()
        
        tk.Label(self.param_frame, text="Spatial radius:").pack()
        self.spatial_radius_entry = tk.Entry(self.param_frame)
        self.spatial_radius_entry.insert(0, "50")
        self.spatial_radius_entry.pack()
        
        tk.Label(self.param_frame, text="Color radius:").pack()
        self.color_radius_entry = tk.Entry(self.param_frame)
        self.color_radius_entry.insert(0, "20")
        self.color_radius_entry.pack()
        
        tk.Label(self.param_frame, text="Max pyramid level:").pack()
        self.max_level_entry = tk.Entry(self.param_frame)
        self.max_level_entry.insert(0, "2")
        self.max_level_entry.pack()
        
        tk.Label(self.param_frame, text="Color space:").pack()
        self.color_space_var = tk.StringVar(value="LAB")
        color_spaces = ["BGR", "LAB", "HSV", "YUV"]
        for space in color_spaces:
            tk.Radiobutton(self.param_frame, text=space, variable=self.color_space_var, 
                          value=space).pack(anchor=tk.W)
    
    def create_dbscan_params(self):
        # Clear existing parameters
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        tk.Label(self.param_frame, text="DBSCAN Parameters").pack()
        
        tk.Label(self.param_frame, text="Eps:").pack()
        self.eps_entry = tk.Entry(self.param_frame)
        self.eps_entry.insert(0, "3")
        self.eps_entry.pack()
        
        tk.Label(self.param_frame, text="Min samples:").pack()
        self.min_samples_entry = tk.Entry(self.param_frame)
        self.min_samples_entry.insert(0, "10")
        self.min_samples_entry.pack()
        
        tk.Label(self.param_frame, text="Color space:").pack()
        self.dbscan_color_space_var = tk.StringVar(value="RGB")
        color_spaces = ["RGB", "LAB", "HSV", "YUV"]
        for space in color_spaces:
            tk.Radiobutton(self.param_frame, text=space, variable=self.dbscan_color_space_var, 
                          value=space).pack(anchor=tk.W)
    
    def create_active_contour_params(self):
        # Clear existing parameters
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        tk.Label(self.param_frame, text="Active Contour Parameters").pack()
        
        tk.Label(self.param_frame, text="Alpha (smoothness):").pack()
        self.alpha_entry = tk.Spinbox(self.param_frame, from_=0.01, to=1.0, increment=0.01, format="%.2f")
        self.alpha_entry.delete(0, tk.END)
        self.alpha_entry.insert(0, "0.2")
        self.alpha_entry.pack()
        
        tk.Label(self.param_frame, text="Beta (edge attraction):").pack()
        self.beta_entry = tk.Spinbox(self.param_frame, from_=0.01, to=1.0, increment=0.01, format="%.2f")
        self.beta_entry.delete(0, tk.END)
        self.beta_entry.insert(0, "0.5")
        self.beta_entry.pack()
        
        tk.Label(self.param_frame, text="Initial contour:").pack()
        
        contour_frame = tk.Frame(self.param_frame)
        contour_frame.pack()
        
        tk.Label(contour_frame, text="Center X:").grid(row=0, column=0)
        self.center_x_entry = tk.Entry(contour_frame, width=5)
        self.center_x_entry.insert(0, "100")
        self.center_x_entry.grid(row=0, column=1)
        
        tk.Label(contour_frame, text="Center Y:").grid(row=1, column=0)
        self.center_y_entry = tk.Entry(contour_frame, width=5)
        self.center_y_entry.insert(0, "100")
        self.center_y_entry.grid(row=1, column=1)
        
        tk.Label(contour_frame, text="Radius:").grid(row=2, column=0)
        self.radius_entry = tk.Entry(contour_frame, width=5)
        self.radius_entry.insert(0, "50")
        self.radius_entry.grid(row=2, column=1)
        
        tk.Label(contour_frame, text="Resolution:").grid(row=3, column=0)
        self.resolution_entry = tk.Entry(contour_frame, width=5)
        self.resolution_entry.insert(0, "300")
        self.resolution_entry.grid(row=3, column=1)
    
    def update_parameters(self):
        method = self.method_var.get()
        if method == "kmeans":
            self.create_kmeans_params()
        elif method == "meanshift":
            self.create_meanshift_params()
        elif method == "dbscan":
            self.create_dbscan_params()
        elif method == "activecontour":
            self.create_active_contour_params()
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.display_image(self.original_image, self.original_label)
                # Hide active contour plot if it's visible
                self.canvas_widget.pack_forget()
                self.segmented_label.pack()
    
    def display_image(self, image, label_widget):
        if image is not None:
            # Convert BGR to RGB for display
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Resize for display while maintaining aspect ratio
            h, w = image.shape[:2]
            max_size = 500
            if h > w:
                new_h = max_size
                new_w = int(w * (max_size / h))
            else:
                new_w = max_size
                new_h = int(h * (max_size / w))
            
            resized = cv2.resize(image, (new_w, new_h))
            
            # Convert to PhotoImage
            img_pil = Image.fromarray(resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            # Update label
            label_widget.config(image=img_tk)
            label_widget.image = img_tk
    
    def segment_image(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please load an image first.")
            return
        
        method = self.method_var.get()
        
        try:
            if method == "kmeans":
                self.kmeans_segmentation()
            elif method == "meanshift":
                self.meanshift_segmentation()
            elif method == "dbscan":
                self.dbscan_segmentation()
            elif method == "activecontour":
                self.active_contour_segmentation()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during segmentation:\n{str(e)}")
    
    def kmeans_segmentation(self):
        k = int(self.k_entry.get())
        max_iter = int(self.max_iter_entry.get())
        epsilon = float(self.epsilon_entry.get())
        
        # Reshape the image to a 2D array of pixels
        pixels = self.original_image.reshape(-1, 3).astype(np.float32)
        
        # Define criteria and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Convert back to 8-bit values
        centers = np.uint8(centers)
        
        # Map the labels to centers
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(self.original_image.shape)
        
        self.segmented_image = segmented_image
        self.display_image(segmented_image, self.segmented_label)
    
    def meanshift_segmentation(self):
        spatial_radius = int(self.spatial_radius_entry.get())
        color_radius = int(self.color_radius_entry.get())
        max_level = int(self.max_level_entry.get())
        color_space = self.color_space_var.get()
        
        # Convert to selected color space
        if color_space == "LAB":
            converted = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2Lab)
        elif color_space == "HSV":
            converted = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        elif color_space == "YUV":
            converted = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2YUV)
        else:  # BGR
            converted = self.original_image.copy()
        
        # Apply Mean Shift
        segmented = cv2.pyrMeanShiftFiltering(
            converted, spatial_radius, color_radius, maxLevel=max_level
        )
        
        # Convert back to BGR if needed
        if color_space == "LAB":
            segmented = cv2.cvtColor(segmented, cv2.COLOR_Lab2BGR)
        elif color_space == "HSV":
            segmented = cv2.cvtColor(segmented, cv2.COLOR_HSV2BGR)
        elif color_space == "YUV":
            segmented = cv2.cvtColor(segmented, cv2.COLOR_YUV2BGR)
        
        self.segmented_image = segmented
        self.display_image(segmented, self.segmented_label)
    
    def dbscan_segmentation(self):
        eps = float(self.eps_entry.get())
        min_samples = int(self.min_samples_entry.get())
        color_space = self.dbscan_color_space_var.get()
        
        # Convert to selected color space
        if color_space == "LAB":
            converted = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2Lab)
        elif color_space == "HSV":
            converted = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        elif color_space == "YUV":
            converted = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2YUV)
        else:  # RGB
            converted = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Reshape the image to a 2D array of pixels
        pixels = converted.reshape(-1, 3)
        
        # Normalize pixel values to 0-1 range
        pixels = pixels.astype(np.float32) / 255.0
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(pixels)
        
        # Reshape labels to image dimensions
        labels = labels.reshape(self.original_image.shape[:2])
        
        # Create segmented image
        segmented_image = np.zeros_like(self.original_image)
        
        # Assign colors to clusters (excluding noise which is -1)
        unique_labels = set(labels)
        colors = [np.random.randint(0, 255, 3) for _ in unique_labels if _ != -1]
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                # Noise points - color them black
                segmented_image[labels == label] = [0, 0, 0]
            else:
                segmented_image[labels == label] = colors[i]
        
        self.segmented_image = segmented_image
        self.display_image(segmented_image, self.segmented_label)
    
    def active_contour_segmentation(self):
        alpha = float(self.alpha_entry.get())
        beta = float(self.beta_entry.get())
        center_x = int(self.center_x_entry.get())
        center_y = int(self.center_y_entry.get())
        radius = int(self.radius_entry.get())
        resolution = int(self.resolution_entry.get())
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Generate initial contour (circle)
        def circle_points(resolution, center, radius):
            radians = np.linspace(0, 2*np.pi, resolution)
            c = center[0] + radius * np.cos(radians)
            r = center[1] + radius * np.sin(radians)
            return np.array([c, r]).T
        
        points = circle_points(resolution=resolution, center=[center_x, center_y], 
                             radius=radius)[:-1]
        
        # Perform active contour segmentation
        snake = active_contour(gray, points, alpha=alpha, beta=beta)
        
        # Create a copy of the original image for display
        display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Clear previous plot
        self.ax.clear()
        
        # Plot the results
        self.ax.imshow(display_image)
        self.ax.plot(points[:, 0], points[:, 1], '--r', lw=2, label="Initial contour")
        self.ax.plot(snake[:, 0], snake[:, 1], '-g', lw=2, label="Segmented contour")
        self.ax.axis('off')
        self.ax.legend()
        
        # Hide the segmented label and show the plot
        self.segmented_label.pack_forget()
        self.canvas_widget.pack()
        self.canvas.draw()
        
        # Store the result (we'll store the plotted image)
        self.segmented_image = None  # For active contour, we don't have a segmented image matrix
    
    def save_image(self):
        if self.segmented_image is None and self.method_var.get() != "activecontour":
            messagebox.showerror("Error", "No segmented image to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if self.method_var.get() == "activecontour":
                    # For active contour, save the figure
                    self.fig.savefig(file_path, bbox_inches='tight', pad_inches=0)
                else:
                    # For other methods, save the segmented image
                    cv2.imwrite(file_path, cv2.cvtColor(self.segmented_image, cv2.COLOR_RGB2BGR))
                messagebox.showinfo("Success", "Image saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()