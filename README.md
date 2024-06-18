<br>
<br>

<div style="text-align: center;">
  <img src="https://upload.wikimedia.org/wikipedia/de/9/9f/Frankfurt_University_of_Applied_Sciences_logo.svg" width="300" alt="Logo">
</div>

<br>
<br>
<br>

<div style="text-align: center;">
  <font size="4"><strong>Project Report</strong></font>
</div>

<br>
<br>
<br>

<div style="text-align: center;">
  <font size="6.9">Exploring Image Segmentation</font>
</div>
<br>
<div style="text-align: center;">
  <font size="6.9">With OpenCV</font>
</div>

<br>
<br>
<br>

<div style="text-align: center;">
  <font size="3"><em>by</em></font>
</div>
<br>
<div style="text-align: center;">
  <font size="4">Minh Kien Nguyen</font>
  <br>
</div>

<br>
<br>

<div style="text-align: center;">
  <font size="3"><strong>Supervisor</strong></font>
  <br>
  <font size="4">Prof. Dr. Peter Nauth</font>
</div>

<br>
<br>

<div style="text-align: center;">
  <font size="3"><strong>Submission Date</strong></font>
  <br>
  <font size="4">July 22nd, 2024</font>
</div>



<div style="page-break-after: always"></div>


# Overview

**Introduction**: The project *Exploring Image Segmentation With OpenCV* is part of the Bilderkennung SS24 module of Prof. Dr. Peter Nauth at the Frankfurt University of Applied Sciences.

**Objective**: This project aims to present possible ways of segmenting images with OpenCV and to show the influence of each image processing step on the final segmentation result.

**Duration**: 06.05.2024 - 22.07.2024

**Source Code**: [Link](https://github.com/krieger1512/Bild1_Segmentation)

**Step-by-Step Architecture**

```mermaid
flowchart TB
  import_resize[Import & Resize]

  convert_grayscale[Convert to Grayscale]

  gaussian_blur[Apply\nGaussian Blur]

  otsu[Apply\nOtsu Binarization]

  morph[Apply\nMorphological Transformation]

  

  draw[Draw Segments]

  subgraph watershed_way[Watershed]
    direction TB
    find_bg[Determine\nSure Background]

    find_fg[Determine\nSure Foreground]

    find_unknown[Determine\nUnknown Region]

    marker[Label With Markers]

    watershed[Apply\nWatershed Algorithm]

    find_bg & find_fg --> find_unknown --> marker --> watershed
  end

  
  import_resize --> convert_grayscale --> gaussian_blur --> otsu --> morph
  morph --> watershed_way --> draw
  
```

## Problem Definition

Given an image with objects of recognizable shapes, divide the image into segments so each segment represents one shape.

This work aims to solve simple image segmentation task with traditional computer vision algorithms instead of machine learning models.
  
Area for Nilsblack "window": twice the size of object

We will break down this problems into smaller tasks:
1. Read the Image: Load the image using OpenCV.
2. Convert the Image to Grayscale: Convert the image from color (BGR) to grayscale.
3. Apply Gaussian Blur: Apply Gaussian blur to smooth the image and reduce noise.
4. Watershed algorithm
5. GrabCut algorithm
6. Find Contours: Use the findContours function to detect the contours in the thresholded image.
7. Draw Contours: Draw the detected contours on the original image or on a blank image to visualize the segmentation.
8. Color the segments

TODO:
1. Suzuki algorithm in OpenCV
2. Write Docu

# Setup/Preconfiguration

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

# Niblack

# Canny

# Watershed

# References