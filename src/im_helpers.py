import imutils
import numpy as np
import cv2
from   typing import Iterator, Tuple, cast
import flow_vis
import os

import utils

# Based on: https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/

def pyramid(image: np.ndarray, scale: float = 1.5, minSize: Tuple[int, int] = (30, 30)) -> Iterator[np.ndarray]:
    """Yields pyramid layers of a larger image

    Args:
        image (np.ndarray): Input image
        scale (float, optional): Factor between different layer sizes. Defaults to 1.5.
        minSize (Tuple[int, int], optional): start size of the largest layer. Defaults to (30, 30).

    Yields:
        Iterator[np.ndarray]: layers of the pyramid
    """
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image


def sliding_window(image: np.ndarray, stepSize: int, windowSize: Tuple[int, int]) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Slides a window across the image

    Args:
        image (np.ndarray): Input image
        stepSize (int): translation between two windows
        windowSize (Tuple[int, int]): size of the resulting window

    Yields:
        Iterator[np.ndarray]: [description]
    """
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def get_simple_bounding_box(img: np.ndarray) -> utils.Rectangle:
    """Fits a bounding box around all pixels with intensity higher than threshold.

    Args:
        image (np.ndarray): the input image

    Returns:
        utils.Rectangle: the resulting bounding box
    """
    height, width = img.shape[:2]
    start_x, start_y = -1, -1
    end_x, end_y = -1, -1
    threshold = 0.1 * np.max(img)
    mask = img > threshold

    for y in range(height):
        if np.sum(mask[y, ...]) > 0:
            end_y = y

            if start_y == -1:
                start_y = y

    for x in range(width):
        if np.sum(mask[:, x, ...]) > 0:
            end_x = x

            if start_x == -1:
                start_x = x

    return utils.Rectangle.from_points((start_x, start_y), (end_x, end_y))


def get_flow_radial(frame: np.ndarray) -> np.ndarray:
    """Gives the radial flow normalized to one

    Args:
        frame (np.ndarray): [description]

    Returns:
        np.ndarray: radial flow normalized to one
    """
    flow_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    flow_hsv[..., 0] = flow_hsv[..., 0]
    flow_hsv[..., 1] = 255
    flow_hsv[..., 2] = 255
    return cast(np.ndarray, cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR))


def get_flow_vis(frame: np.ndarray, magnitude_factor: float = 1.0) -> np.ndarray:
    """Visualize a flow field array

    Args:
        frame (np.ndarray): the raw flow field (w, h, 2)

    Returns:
        np.ndarray: BGR flow field visualized in HSV space
    """
    return cast(np.ndarray, flow_vis.flow_to_color(frame, convert_to_bgr=True))


def apply_colormap(img: np.ndarray, max_value: float = None, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Applies the jet colormap to the input image.

    Args:
        img (np.ndarray): the image to apply the colormap to
        max_value (float, optional): scaling variable which sets the input image's range to [0, max_value]. Defaults to None.
        colormap (int): the colormap type to apply. Defaults to COLORMAP_JET.
    Returns:
        np.ndarray: [description]
    """
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = to_int(img, normalize=True, max_value=max_value)

    if max_value is None:
        return cast(np.ndarray, cv2.applyColorMap(img, colormap))

    old_value = img[0, 0, ...]
    img[0, 0, ...] = max_value
    result = cv2.applyColorMap(img, colormap)
    result[0, 0, ...] = cv2.applyColorMap(np.ones((1, 1, 3), dtype=np.uint8) * old_value, colormap)
    return cast(np.ndarray, result)


def get_rho(img: np.ndarray) -> np.ndarray:
    """Calculates the angle of the vectors in the last dimension of img.

    Args:
        img (np.ndarray): the input vector field

    Returns:
        np.ndarray: (h, w) array of angles (radians)
    """
    return cast(np.ndarray, np.arctan2(img[:, :, 1], img[:, :, 0]))


def get_magnitude(img: np.ndarray) -> np.ndarray:
    """Calculates the magnitude of the vectors in the last dimension of img.

    Args:
        img (np.ndarray): the input image

    Returns:
        np.ndarray: (h, w) array of magnitudes
    """
    return cast(np.ndarray, np.linalg.norm(img, axis=-1))


def to_rgb(img: np.ndarray, max_value: float = None)-> np.ndarray:
    """Converts grayscale to RGB.

    Args:
        img (np.ndarray): grayscale input image

    Returns:
        np.ndarray: output RGB image
    """
    return cast(np.ndarray,
        cv2.cvtColor(to_int(img, np.uint8, True, max_value=max_value), cv2.COLOR_GRAY2RGB)
    )


def to_int(img: np.ndarray, type: type=np.uint8, normalize: bool=False, max_value: float = None) -> np.ndarray:
    """Transform img of floating-point type to integers.

    Args:
        img (np.ndarray): input image
        type (type, optional): output image type. Defaults to np.uint8.
        normalize (bool, optional): whether to normalize the input image to [0, 255]. Defaults to False.
        max_value (float, optional): scaling variable used for the normalization. Defaults to None.

    Returns:
        np.ndarray: [description]
    """
    img_normalized = img

    if normalize:
        if max_value is None:
            max_value = np.max(img)
        elif max_value <= 0.0:
            max_value = 1.0

        img_normalized = np.abs(img_normalized) * 255 / max_value

    return cast(np.ndarray,
        np.around(img_normalized).astype(type)
    )


def get_fft(frame: np.ndarray) -> np.ndarray:
    fft = np.fft.fft2(frame[..., 0])
    fshift = np.fft.fftshift(fft)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    magnitude_rgb = np.zeros_like(frame)
    magnitude_rgb[..., 0] = magnitude_spectrum
    return magnitude_rgb


def plot_colorbar(path: str = 'media/colorbar.png') -> np.ndarray:
    if os.path.exists(path):
        return cast(np.ndarray, cv2.imread(path))

    img = np.zeros((200, 30, 3), dtype=np.uint8)
    for y in range(img.shape[0]):
        img[y, ...] = y

    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite(path, img)
    return img


def get_colorwheel(path: str = 'media/colorwheel.png') -> np.ndarray:
    if os.path.exists(path):
        return cast(np.ndarray, cv2.imread(path))

    diameter = 250
    radius = diameter / 2
    img = np.zeros((diameter, diameter, 2))
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img[y, x, 0] = x - radius
            img[y, x, 1] = y - radius

            if np.sqrt((x - radius) ** 2 + (y - radius) ** 2) > radius:
                img[y, x, :] = 0

    img = get_flow_vis(img)
    cv2.imwrite(path, img)
    return img

def calculate_tpr_fpr(gt_img: np.ndarray, img: np.ndarray) -> Tuple[float, float]:
    positives = np.sum(gt_img > 127)
    negatives = np.sum((255 - gt_img) > 127)
    true_positives = np.sum((gt_img * img) > 127)
    false_positives = np.sum(((255 - gt_img) * img) > 127)

    tpr = true_positives / positives
    fpr = false_positives / negatives
    return (cast(float, tpr), cast(float, fpr))

def resize_percent(img: np.ndarray, scale_percent: float) -> np.ndarray:
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cast(np.ndarray,
        cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    )
