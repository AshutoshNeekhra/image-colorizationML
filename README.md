# image-colorizationML                                                                                               Made by-ASHUTOSH NEEKHRA
                                                                                                                     Enrollment no -0176AL221038


🌄 Landscape Image Colorization using PyTorch
Project Overview:
Uses a Convolutional Neural Network (CNN) to colorize grayscale landscape images.
Built with PyTorch, trained on paired grayscale and color images.


Dataset Structure:
landscape_images/
├── color/
│   ├── image1.jpg
└── gray/
    ├── image1.jpg

    
    
Key Features:
Custom PyTorch Dataset class (LandscapeDataset)
Paired grayscale and color image loading
Preprocessing with torchvision.transforms
Train/test DataLoaders
Visualization of grayscale vs. predicted color images
Optional Google Colab and Gradio support
Setup & Installation:

Install dependencies:
pip install torch torchvision matplotlib tqdm pillow gradio
In Google Colab:

python
from google.colab import drive
drive.mount('/content/gdrive')
Extract dataset (example):
!unzip "/content/gdrive/MyDrive/archive (3).zip" -d "/content/"


Workflow:
Preprocess images
Load data using custom Dataset
Train a CNN model to convert grayscale → color
Evaluate model and visualize output
Optionally, deploy with Gradio


Model Architecture:
Multiple convolutional layers
ReLU activations
Upsampling layers for reconstructing color


Training:
Run:
python train.py
Parameters:
Epochs: 8
Learning Rate: 0.001
Batch Size: 32
Device: CUDA if available, else CPU

Visualization:
Run:
python visualize.py
Displays side-by-side grayscale and colorized images

Gradio Web Demo (Optional):
import gradio as gr

def predict(image):
    # Model inference code here
    return colorized_image

gr.Interface(fn=predict, inputs="image", outputs="image").launch()
Example Output:

(Insert sample grayscale and colorized image comparisons)

Author:

Ashutosh Neekhra

Enrollment No.: 0176AL221038

Contact:

For questions or feedback, open an issue on this GitHub repository.
