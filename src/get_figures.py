import numpy as np
import cv2

pixel_counts = []

for i in range(5, 18):
    img = cv2.imread(f'data/mountains-moving/lake-orbit-0-north-low-40.0-15-default/segmentations/image_{i:05d}.png', 0)
    pixel_count = np.sum(img) / 255
    pixel_counts.append(pixel_count)

pixel_counts = np.array(pixel_counts)
print(np.average(pixel_counts), np.std(pixel_counts))
print(pixel_counts)
