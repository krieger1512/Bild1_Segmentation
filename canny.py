import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def plot_gray(image):
    plt.figure(figsize=(16, 10))
    return plt.imshow(image, cmap="Greys_r")


def plot_rgb(image):
    plt.figure(figsize=(16, 10))
    return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":

    # Step 1: Import the image
    IMAGE_PATH = os.path.join(os.getcwd(), "input", "art.jpg")
    image = cv2.imread(IMAGE_PATH)
    original = image.copy()
    image = opencv_resize(
        image, ratio=500 / image.shape[0]
    )  # Downscale image as finding contour is more efficient on a small image

    # Step 2: Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plot_gray(gray)

    ######################## CANNY ########################
    # Step 3: Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # plot_gray(blurred)

    # Morphological Transformations: Opening
    # Our objects are black, so we want to remove white noise
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, rectKernel)
    # plot_gray(opened)

    # Apply Canny edge detection
    # Link: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    edged = cv2.Canny(opened, 50, 180, apertureSize=3)
    # plot_gray(edged)

    # Detect all contours in Canny-edged image
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 1)
    plot_rgb(image_with_contours)
    ######################## CANNY ########################

    plt.show()
