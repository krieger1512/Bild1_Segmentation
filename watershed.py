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


def find_sure_fg(foreground_factor, image):
    dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, cv2.DIST_MASK_5)
    # plot_gray(cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX))
    _, sure_fg = cv2.threshold(
        dist_transform, foreground_factor * dist_transform.max(), 255, 0
    )

    return sure_fg


def plot_gray(image):
    plt.figure(figsize=(16, 10))
    plt.imshow(image, cmap="Greys_r")


def plot_rgb(image):
    plt.figure(figsize=(16, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def watershed_segment(
    image_name,
    resize_ratio,
    blurring_kernel_size,
    closing_kernel_size,
    closing_iterations,
    dilate_kernel_size,
    dilate_iterations,
    foreground_factor,
):

    # Import image
    image = import_image(image_name)

    # Resize image
    image = resize(image, resize_ratio)
    # plot_rgb(image)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plot_gray(gray)

    # Apply Gaussian blur for better binarization
    blurred = cv2.GaussianBlur(gray, blurring_kernel_size, 0)
    # plot_gray(blurred)

    # Apply Otsu binarization (global adaptive thresholding); local adaptive thresholding is not possible due to no separation between foreground and background objects
    _, thresh = cv2.threshold(
        blurred,
        0,  # This value can be selected arbitrarily due to Otsu binarization
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    # plot_gray(thresh)

    # Close small holes (black points) inside foreground objects
    closed = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel=np.ones(closing_kernel_size, np.uint8),
        iterations=closing_iterations,
    )
    # plot_gray(closed)

    # Find sure background area
    sure_bg = cv2.dilate(
        closed,
        kernel=np.ones(dilate_kernel_size, np.uint8),
        iterations=dilate_iterations,
    )
    # plot_gray(sure_bg)

    # Find sure foreground area
    sure_fg = find_sure_fg(foreground_factor, image=closed)
    # plot_gray(sure_fg)

    # Find unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # plot_gray(unknown)

    # Marker labelling
    # Connected Components determines the connectivity of blob-like regions in a binary image.
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(image, markers)
    # plot_gray(markers)
    image[markers == -1] = [0, 255, 0]
    plot_rgb(image)

    plt.show()


if __name__ == "__main__":

    watershed_segment(
        image_name="art.jpg",
        resize_ratio=0.25,
        blurring_kernel_size=(7, 7),
        closing_kernel_size=(5, 5),
        closing_iterations=6,  # 4; 6
        dilate_kernel_size=(5, 5),
        dilate_iterations=10,
        foreground_factor=0.26,  # 0.239; 0.26
    )
