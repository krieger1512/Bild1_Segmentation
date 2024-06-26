import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def import_image(image_name):
    image_path = os.path.join(os.getcwd(), "input", image_name)
    image = cv2.imread(image_path)
    return image


def resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image


def find_sure_fg(foreground_thresh, image):
    dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, cv2.DIST_MASK_5)
    # plot_gray(
    #     cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX),
    #     "Distance-transformed Normalized Image",
    # )
    _, sure_fg = cv2.threshold(
        dist_transform, foreground_thresh * dist_transform.max(), 255, 0
    )

    return sure_fg


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


def watershed_segment(
    image_name,
    resize_ratio,
    blurring_kernel_size,
    morph_type,
    morph_kernel_size,
    morph_iterations,
    dilate_kernel_size,
    dilate_iterations,
    foreground_thresh,
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

    # Apply Otsu binarization (global adaptive thresholding)
    _, thresh = cv2.threshold(
        blurred,
        0,  # This value can be selected arbitrarily due to Otsu binarization
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
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

    # Find sure background area
    sure_bg = cv2.dilate(
        morphed,
        kernel=np.ones(dilate_kernel_size, np.uint8),
        iterations=dilate_iterations,
    )
    # plot_gray(sure_bg, "Sure Background")

    # Find sure foreground area
    sure_fg = find_sure_fg(foreground_thresh, image=morphed)
    # plot_gray(sure_fg, "Sure Foreground")

    # Find unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # plot_gray(unknown, "Unknown Area")

    # Marker labelling
    # Connected Components determines the connectivity of blob-like regions in a binary image.
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(image, markers)
    # plot_gray(markers, "Segments")
    image[markers == -1] = [0, 255, 0]
    # plot_rgb(image, "Segmented Image")

    markers_colormap = color_markers(markers)

    # plt.show()
    return thresh, morphed, sure_bg, sure_fg, markers_colormap, image


def color_markers(markers):
    markers_as_array = np.array(markers, dtype=np.int32)
    markers_normalized = cv2.normalize(
        markers_as_array,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    markers_colormap = cv2.applyColorMap(markers_normalized, cv2.COLORMAP_JET)
    return markers_colormap


def update_image(x):
    window_name = "Watershed Controller"

    thresh, morphed, sure_bg, sure_fg, markers, image = watershed_segment(
        image_name=str(get_value("Image", window_name)) + ".jpg",
        resize_ratio=get_value("Resize(%)", window_name),
        blurring_kernel_size=get_kernel_size("Blur KS", window_name),
        morph_type=get_type("Open/Close", window_name),
        morph_kernel_size=get_kernel_size("Morph KS", window_name),
        morph_iterations=get_value("Morph Ite", window_name),
        dilate_kernel_size=get_kernel_size("Dilate KS", window_name),
        dilate_iterations=get_value("Dilate Ite", window_name),
        foreground_thresh=get_value("FG-Thre(%)", window_name),
    )

    # cv2.imshow("Binarization", thresh)
    # cv2.imshow("Morphological Transformations", morphed)
    # cv2.imshow("Sure Background Area", sure_bg)
    # cv2.imshow("Sure Foreground Area", sure_fg)
    cv2.imshow("Segments", markers)
    cv2.imshow("Image Segmentation with Watershed", image)


def get_value(trackbar_name, window_name):
    value = cv2.getTrackbarPos(trackbar_name, window_name)
    if trackbar_name in ["Resize(%)", "FG-Thre(%)"]:
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
    window_name = "Watershed Controller"
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, 1000, 340)

    cv2.createTrackbar("Image", window_name, 1, number_of_images, update_image)
    cv2.setTrackbarMin("Image", window_name, 1)

    cv2.createTrackbar("Resize(%)", window_name, 25, 100, update_image)
    cv2.setTrackbarMin("Resize(%)", window_name, 1)

    cv2.createTrackbar("Blur KS", window_name, 3, kernel_size_limit, update_image)
    cv2.setTrackbarMin("Blur KS", window_name, 3)

    cv2.createTrackbar("Open/Close", window_name, 0, 1, update_image)

    cv2.createTrackbar("Morph KS", window_name, 3, kernel_size_limit, update_image)
    cv2.setTrackbarMin("Morph KS", window_name, 3)

    cv2.createTrackbar("Morph Ite", window_name, 1, iteration_limit, update_image)
    cv2.setTrackbarMin("Morph Ite", window_name, 1)

    cv2.createTrackbar("Dilate KS", window_name, 3, kernel_size_limit, update_image)
    cv2.setTrackbarMin("Dilate KS", window_name, 3)

    cv2.createTrackbar("Dilate Ite", window_name, 1, iteration_limit, update_image)
    cv2.setTrackbarMin("Dilate Ite", window_name, 1)

    cv2.createTrackbar("FG-Thre(%)", window_name, 20, 100, update_image)
    cv2.setTrackbarMin("FG-Thre(%)", window_name, 1)


if __name__ == "__main__":

    # If you want to check the interim results during segmentation,
    # uncomment the plot_gray()/plot_rgb() and the final plt.show() in watershed_segment()

    # watershed_segment(
    #     image_name="1.jpg",
    #     resize_ratio=0.25,
    #     blurring_kernel_size=(7, 7),
    #     morph_type=cv2.MORPH_CLOSE,  # Image with white background: Close
    #     morph_kernel_size=(5, 5),
    #     morph_iterations=6,  # 4; 6
    #     dilate_kernel_size=(5, 5),
    #     dilate_iterations=10,
    #     foreground_thresh=0.26,  # 0.239; 0.26
    # )

    # watershed_segment(
    #     image_name="3.jpg",
    #     resize_ratio=1,
    #     blurring_kernel_size=(5, 5),
    #     morph_type=cv2.MORPH_OPEN, # Image with black background: Open
    #     morph_kernel_size=(3, 3),
    #     morph_iterations=3,
    #     dilate_kernel_size=(3, 3),
    #     dilate_iterations=1,
    #     foreground_thresh=0.78,
    # )

    # If you want to check the influence of different parameters on the segmentation,
    # comment out all the plot_gray()/plot_rgb() and the final plt.show() in watershed_segment()

    input_images = os.listdir(os.path.join(os.getcwd(), "input"))
    jpg_files = [file for file in input_images if file.lower().endswith((".jpg"))]
    create_trackbar_window(
        number_of_images=len(jpg_files), kernel_size_limit=13, iteration_limit=40
    )
    while True:
        update_image(0)
        cv2.waitKey(1)
