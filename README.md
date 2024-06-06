# Bild1_Segmentation

## Problem Definition

Given an image with recognizable shapes, divide the image into segments so each segment represents one shape.

This work aims to solve simple image segmentation task with traditional computer vision algorithms instead of machine learning models.

We will break down this problems into smaller tasks:
1. Read the Image: Load the image using OpenCV.
2. Convert the Image to Grayscale: Convert the image from color (BGR) to grayscale.
3. Apply Gaussian Blur: Apply Gaussian blur to smooth the image and reduce noise.
4. Threshold the Image: Apply a binary threshold to create a binary image where the foreground (woman) is white, and the background is black.
5. Find Contours: Use the findContours function to detect the contours in the thresholded image.
6. Draw Contours: Draw the detected contours on the original image or on a blank image to visualize the segmentation.

## Setup/Preconfiguration

Only needs to be done once

Requires: ``Python 3.4+`` and ``pip`` (included in `Python 3.4+` by default)

Follow these steps in your terminal:

1. Change directory to code folder (if necessary)
    ```
    # For Windows
    cd Bild1_Segmentation
    ```
2. Create virtual environment:
    ```
    # Use .venv as name of virtual environment
    python -m venv .venv 
    ```
3. Activate virtual environment:
    ```
    # For Windows
    .venv/Scripts/activate 
    ```
4. Install the necessary modules/libraries:
    ```
    pip install -r requirements.txt    
    ```