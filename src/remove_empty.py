import cv2
import glob
import numpy as np
import os

masks = glob.glob('/home/erik/tno/datasets/data/mountains-orbit/*/segmentations/*.png')
masks.sort()

for p in masks:
    img = cv2.imread(p)
    if np.sum(img) < 1:
        os.remove(p)
