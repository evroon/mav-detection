import imutils
import numpy as np
import cv2
from   typing import Iterator, Tuple
import flow_vis

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

def sliding_window(image: np.ndarray, stepSize: int, windowSize: Tuple[int, int]) -> Iterator[np.ndarray]:
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
	return cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR)

def get_flow_vis(frame: np.ndarray) -> np.ndarray:
	"""Visualize a flow field array

	Args:
		frame (np.ndarray): the raw flow field (w, h, 2)

	Returns:
		np.ndarray: BGR flow field visualized in HSV space
	"""
	return flow_vis.flow_to_color(frame, convert_to_bgr=True)