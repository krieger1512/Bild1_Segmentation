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

    ######################## WATERSHED ########################
    # Apply thresholding
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    # plot_gray(thresh)

    # black noise (small holes) removal
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    # plot_gray(closing)

    # sure background area
    sure_bg = cv2.dilate(closing, kernel, iterations=2)
    # plot_gray(sure_bg)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    dist_output = cv2.normalize(
        dist_transform, None, 0, 1.0, cv2.NORM_MINMAX
    )  # Make the distance transform normal.
    # plot_gray(dist_output)
    ret, sure_fg = cv2.threshold(dist_transform, 0.01 * dist_transform.max(), 255, 0)
    # plot_gray(sure_fg)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # plot_gray(unknown)

    # Marker labelling
    # Connected Components determines the connectivity of blob-like regions in a binary image.
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)
    image[markers == -1] = [0, 255, 0]
    plot_rgb(image)
    ######################## WATERSHED ########################

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
