import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def niblack_threshold(image, k, window_size):
    # Convert the image to float32 for precision
    image = image.astype(np.float32)

    # Calculate the local mean (mean filter)
    mean = cv2.boxFilter(image, cv2.CV_32F, window_size)

    # Calculate the local variance and then standard deviation
    sqr_mean = cv2.sqrBoxFilter(image, cv2.CV_32F, window_size)
    variance = sqr_mean - mean**2
    stddev = np.sqrt(variance)

    # Calculate the Niblack threshold
    threshold = mean + k * stddev

    # Apply the threshold to binarize the image
    binary_image = (image > threshold).astype(np.uint8) * 255

    return binary_image


def import_image(image_name):
    image_path = os.path.join(os.getcwd(), "input", image_name)
    image = cv2.imread(image_path)
    return image


def resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image


def plot_gray(image, title):
    plt.figure(figsize=(16, 10))
    plt.imshow(image, cmap="Greys_r")
    plt.title(title)
    plt.axis("off")


def plot_rgb(image, title):
    plt.figure(figsize=(16, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")


def niblack_segment(
    image_name,
    resize_ratio,
    blurring_kernel_size,
    niblack_k,
    niblack_window_size,
    morph_type,
    morph_kernel_size,
    morph_iterations,
):

    # Import image
    image = import_image(image_name)

    # Resize image
    image = resize(image, resize_ratio)
    # plot_rgb(image, "Original Image")

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plot_gray(gray, "Grayscale Image")

    # Apply Gaussian blur for better binarization
    blurred = cv2.GaussianBlur(gray, blurring_kernel_size, 0)
    # plot_gray(blurred, "Blurred Image")

    # Apply Niblack binarization
    thresh = niblack_threshold(
        image=blurred, k=niblack_k, window_size=niblack_window_size
    )
    # plot_gray(thresh, "Binarized Image")

    # Reduce noise with morphological transformations
    morphed = cv2.morphologyEx(
        thresh,
        morph_type,
        kernel=np.ones(morph_kernel_size, np.uint8),
        iterations=morph_iterations,
    )
    # plot_gray(morphed, "Morphed Image")

    # Detect all contours in Canny-edged image
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 1)
    # plot_rgb(image_with_contours, "Segmented Image")

    colored_segments = color_segments(image, contours)

    # plt.show()
    return thresh, morphed, colored_segments, image_with_contours


def color_segments(image, contours):
    colored_segments = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    colors = plt.get_cmap("hsv", len(contours))

    for i, contour in enumerate(contours):
        color = tuple(
            int(c * 255) for c in colors(i)[:3]
        )  # Convert colormap to BGR tuple
        cv2.drawContours(colored_segments, [contour], -1, color, cv2.FILLED)
    return colored_segments


def update_image(x):
    window_name = "Niblack Workstation"

    thresh, morphed, colored_segments, image_with_contours = niblack_segment(
        image_name=str(get_value("Image", window_name)) + ".jpg",
        resize_ratio=get_value("Resize(%)", window_name),
        blurring_kernel_size=get_kernel_size("Blur KS", window_name),
        niblack_k=get_value("NB k (%)", window_name),
        niblack_window_size=get_kernel_size("NB KS", window_name),
        morph_type=get_type("Open/Close", window_name),
        morph_kernel_size=get_kernel_size("Morph KS", window_name),
        morph_iterations=get_value("Morph Ite", window_name),
    )

    # cv2.imshow("Niblack Binarization", thresh)
    # cv2.imshow("Morphological Transformations", morphed)
    cv2.imshow("Segments", colored_segments)
    cv2.imshow("Image Segmentation", image_with_contours)


def get_value(trackbar_name, window_name):
    value = cv2.getTrackbarPos(trackbar_name, window_name)
    if trackbar_name in ["Resize(%)", "NB k (%)"]:
        value = value / 100
    return value


def get_kernel_size(trackbar_name, window_name):
    kernel_size = cv2.getTrackbarPos(trackbar_name, window_name)
    if kernel_size % 2 == 0:  # Kernel size must be odd
        kernel_size += 1
        cv2.setTrackbarPos(trackbar_name, window_name, kernel_size)
    return (kernel_size, kernel_size)


def get_type(trackbar_name, window_name):
    type = cv2.getTrackbarPos(trackbar_name, window_name)
    return cv2.MORPH_OPEN if type == 0 else cv2.MORPH_CLOSE


def create_trackbar_window(number_of_images, kernel_size_limit, iteration_limit):
    window_name = "Niblack Workstation"
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, 1000, 309)

    cv2.createTrackbar("Image", window_name, 1, number_of_images, update_image)
    cv2.setTrackbarMin("Image", window_name, 1)

    cv2.createTrackbar("Resize(%)", window_name, 25, 100, update_image)
    cv2.setTrackbarMin("Resize(%)", window_name, 1)

    cv2.createTrackbar("Blur KS", window_name, 3, kernel_size_limit, update_image)
    cv2.setTrackbarMin("Blur KS", window_name, 3)

    cv2.createTrackbar("NB k (%)", window_name, 20, 100, update_image)
    cv2.setTrackbarMin("NB k (%)", window_name, -100)

    cv2.createTrackbar("NB KS", window_name, 3, kernel_size_limit, update_image)
    cv2.setTrackbarMin("NB KS", window_name, 3)

    cv2.createTrackbar("Open/Close", window_name, 0, 1, update_image)

    cv2.createTrackbar("Morph KS", window_name, 3, kernel_size_limit, update_image)
    cv2.setTrackbarMin("Morph KS", window_name, 3)

    cv2.createTrackbar("Morph Ite", window_name, 1, iteration_limit, update_image)
    cv2.setTrackbarMin("Morph Ite", window_name, 1)


if __name__ == "__main__":

    # If you want to check the interim results during segmentation,
    # uncomment the plot_gray()/plot_rgb() and the final plt.show() in niblack_segment()

    # niblack_segment(
    #     image_name="2.jpg",
    #     resize_ratio=1,
    #     blurring_kernel_size=(5, 5),
    #     niblack_k=-0.2,
    #     niblack_window_size=(3, 3),
    #     morph_type=cv2.MORPH_OPEN,
    #     morph_kernel_size=(3, 3),
    #     morph_iterations=1,
    # )

    # If you want to check the influence of different parameters on the segmentation,
    # comment out all the plot_gray()/plot_rgb() and the final plt.show() in niblack_segment()

    input_images = os.listdir(os.path.join(os.getcwd(), "input"))
    jpg_files = [file for file in input_images if file.lower().endswith((".jpg"))]
    create_trackbar_window(
        number_of_images=len(jpg_files), kernel_size_limit=17, iteration_limit=40
    )
    while True:
        update_image(0)
        cv2.waitKey(1)
