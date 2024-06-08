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


# approximate the contour by a more primitive polygon shape
def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.032 * peri, True)


if __name__ == "__main__":

    # Step 1: Import the image
    IMAGE_PATH = os.path.join(os.getcwd(), "input", "art.jpg")
    image = cv2.imread(IMAGE_PATH)
    original = image.copy()
    image = opencv_resize(
        image, ratio=500 / image.shape[0]
    )  # Downscale image as finding contour is more efficient on a small image

    # In order to find object contour, standard edge detection preprocessing is applied:
    # Step 2: Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plot_gray(gray)

    # Step 3: Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # plot_gray(blurred)

    # Morphological Transformations: Opening
    # Our objects are black, so we want to remove white noise
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, rectKernel)
    plot_gray(opened)

    # Apply Canny edge detection
    # Link: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    edged = cv2.Canny(opened, 100, 200, apertureSize=3)
    plot_gray(edged)

    # # Apply thresholding
    # _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plot_gray(thresh)

    # # Combine edges and thresholded image
    # combined = cv2.bitwise_or(edged, thresh)
    # plot_gray(combined)

    # # For better accuracy, use binary images. So before finding contours, apply threshold or canny edge detection.
    # # Detect all contours in Canny-edged image
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contour = cv2.convexHull(contours)
    # print([cv2.contourArea(cnt) for cnt in contours])
    # contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]

    image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
    plot_rgb(image_with_contours)

    # # Get 10 largest contours
    # largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    # image_with_largest_contours = cv2.drawContours(
    #     image.copy(), largest_contours, -1, (0, 255, 0), 2
    # )
    # plot_rgb(image_with_largest_contours)
    plt.show()
