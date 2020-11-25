import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
from   lucas_kanade import LucasKanade
from   farneback import Farneback

# sequence = '06'
# capture, N = utils.get_kitti_capture(sequence)

sequence = 'uav0000244_01440_v'
capture, N = utils.get_vis_drone_capture(sequence)

# capture, N = utils.get_train_capture()

output = utils.get_output('detection', capture)
feature_detector = LucasKanade(capture, output)

i = 0

try:
    while i < N - 1:
        if i % int(N / 10) == 0:
            print('{:.2f}'.format(i / N * 100) + '%', i, '/', N)

        frame = feature_detector.process()
        cv2.imshow('frame', frame)
        output.write(frame)
        i += 1

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
finally:
    capture.release()
    output.release()
    cv2.destroyAllWindows()
