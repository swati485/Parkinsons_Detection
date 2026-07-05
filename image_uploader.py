# image_uploader.py
# A companion script for the Parkinson's Disease Detection System
# Provides a simple interface for uploading images and getting predictions

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import subprocess
import argparse

class ParkinsonsImageUploader:
    def __init__(self, root):
        """Initialize the image uploader interface"""
        self.root = root
        self.root.title("Parkinson's Disease Detection - Image Uploader")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Set icon if available
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        # Variables
        self.image_path = None
        self.current_image = None
        self.selected_dataset = tk.StringVar(value="spiral")
        self.selected_model = tk.StringVar(value="RF")
        
        # Create the main frame
        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create frames
        self.create_control_panel()
        self.create_image_panel()
        self.create_result_panel()
        
        # Check if models exist
        self.check_models()

    def check_models(self):
        """Check if trained models exist"""
        model_dirs = ['./models/spiral', './models/wave']
        models_exist = True
        
        for directory in model_dirs:
            if not os.path.exists(directory):
                models_exist = False
                break
            
            # Check for model files
            model_files = os.listdir(directory)
            if not any(file.endswith('_model.pkl') for file in model_files):
                models_exist = False
                break
        
        if not models_exist:
            messagebox.showwarning(
                "Models Not Found", 
                "Trained models were not found. Please train models first using:\n\n"
                "python parkinsons_enhanced.py --action train"
            )

    def create_control_panel(self):
        """Create the control panel with buttons and options"""
        control_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Create a grid layout
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)
        
        # Upload button
        upload_btn = ttk.Button(control_frame, text="Upload Image", command=self.upload_image)
        upload_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # Dataset selection
        dataset_frame = ttk.Frame(control_frame)
        dataset_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(dataset_frame, text="Dataset:").pack(side=tk.LEFT, padx=5)
        dataset_combo = ttk.Combobox(dataset_frame, textvariable=self.selected_dataset, state="readonly")
        dataset_combo['values'] = ('spiral', 'wave')
        dataset_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Model selection
        model_frame = ttk.Frame(control_frame)
        model_frame.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        model_combo = ttk.Combobox(model_frame, textvariable=self.selected_model, state="readonly")
        model_combo['values'] = ('RF', 'SVM', 'KNN')
        model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Analysis button
        analyze_btn = ttk.Button(control_frame, text="Analyze Image", command=self.analyze_image)
        analyze_btn.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

    def create_image_panel(self):
        """Create the panel to display the uploaded image"""
        self.image_frame = ttk.LabelFrame(self.main_frame, text="Uploaded Image", padding=10)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a canvas for the image
        self.image_canvas = tk.Canvas(self.image_frame, bg="white")
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Default text
        self.image_canvas.create_text(
            400, 200, 
            text="No image uploaded. Click 'Upload Image' to select an image.", 
            fill="gray", 
            font=("Arial", 14)
        )

    def create_result_panel(self):
        """Create the panel to display analysis results"""
        result_frame = ttk.LabelFrame(self.main_frame, text="Analysis Results", padding=10)
        result_frame.pack(fill=tk.X, pady=5)
        
        # Text widget for displaying results
        self.result_text = tk.Text(result_frame, height=6, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.config(state=tk.DISABLED)
        
        # Add default text
        self.update_result_text("Analysis results will appear here after processing an image.")

    def upload_image(self):
        """Handle image upload"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(title="Select Image", filetypes=file_types)
        
        if filepath:
            try:
                # Store the image path
                self.image_path = filepath
                
                # Load and display the image
                self.load_display_image(filepath)
                
                # Update result text
                filename = os.path.basename(filepath)
                self.update_result_text(f"Image '{filename}' loaded successfully.\nClick 'Analyze Image' to process it.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def load_display_image(self, filepath):
        """Load and display the image on the canvas"""
        # Clear the canvas
        self.image_canvas.delete("all")
        
        # Load the image
        image = Image.open(filepath)
        
        # Resize the image to fit the canvas (maintaining aspect ratio)
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        # Use default sizes if the canvas hasn't been drawn yet
        if canvas_width <= 1:
            canvas_width = 700
        if canvas_height <= 1:
            canvas_height = 400
            
        # Resize image maintaining aspect ratio
        image_ratio = image.width / image.height
        canvas_ratio = canvas_width / canvas_height
        
        if image_ratio > canvas_ratio:
            # Width constrained
            new_width = canvas_width
            new_height = int(canvas_width / image_ratio)
        else:
            # Height constrained
            new_height = canvas_height
            new_width = int(canvas_height * image_ratio)
            
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to Tkinter PhotoImage
        self.current_image = ImageTk.PhotoImage(image)
        
        # Calculate position to center the image
        x_pos = (canvas_width - new_width) // 2
        y_pos = (canvas_height - new_height) // 2
        
        # Display the image
        self.image_canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.current_image)

    def analyze_image(self):
        """Analyze the uploaded image using the Parkinson's detection system"""
        if not self.image_path:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return
            
        try:
            # Update status
            self.update_result_text("Analyzing image... Please wait.")
            self.root.update()
            
            # Check if the main script exists
            if not os.path.exists("parkinsons_enhanced.py"):
                messagebox.showerror(
                    "Error", 
                    "The main script 'parkinsons_enhanced.py' was not found in the current directory."
                )
                return
                
            # Call the main script
            dataset = self.selected_dataset.get()
            model = self.selected_model.get()
            
            # Run the prediction using subprocess
            command = [
                sys.executable, "parkinsons_enhanced.py", 
                "--action", "predict",
                "--image", self.image_path,
                "--dataset", dataset,
                "--model", model
            ]
            
            # Run the command and capture output
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            stdout, stderr = process.communicate()
            
            # Check for errors
            if process.returncode != 0:
                messagebox.showerror("Error", f"Analysis failed:\n{stderr}")
                self.update_result_text(f"Analysis failed. Error:\n{stderr}")
                return
                
            # Look for prediction results in the output
            prediction_line = None
            for line in stdout.split('\n'):
                if line.startswith("Prediction:"):
                    prediction_line = line
                    break
                    
            if not prediction_line:
                self.update_result_text(
                    "Analysis completed but no clear prediction was found.\n"
                    f"Raw output:\n{stdout}"
                )
            else:
                # Display the result
                result_filename = f"./results/prediction_{os.path.basename(self.image_path)}"
                
                self.update_result_text(
                    f"Analysis completed:\n{prediction_line}\n\n"
                    f"Result image saved to: {result_filename}\n\n"
                    f"Model used: {model} on {dataset} dataset"
                )
                
                # Load and display the result image if it exists
                if os.path.exists(result_filename):
                    self.load_display_image(result_filename)
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis: {str(e)}")
            self.update_result_text(f"Analysis error: {str(e)}")

    def update_result_text(self, text):
        """Update the text in the result panel"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)

def main():
    """Main function to run the application"""
    parser = argparse.ArgumentParser(description="Parkinson's Disease Detection - Image Uploader")
    parser.add_argument('--image', type=str, help='Path to image to load on startup')
    args = parser.parse_args()
    
    # Create the root window
    root = tk.Tk()
    app = ParkinsonsImageUploader(root)
    
    # If an image was specified, load it
    if args.image and os.path.exists(args.image):
        app.image_path = args.image
        app.load_display_image(args.image)
        filename = os.path.basename(args.image)
        app.update_result_text(f"Image '{filename}' loaded successfully.\nClick 'Analyze Image' to process it.")
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()