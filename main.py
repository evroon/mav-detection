import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
from   lucas_kanade import LucasKanade
from   farneback import Farneback

# sequence = '06'
# capture, N = utils.get_kitti_capture(sequence)

# sequence = 'uav0000244_01440_v'
# capture, N = utils.get_vis_drone_capture(sequence)

# sequence = 'indoor-modern/warehouse-transition'

sequence = 'dataset2'
orig_capture, N = utils.get_cenek_capture(sequence, 'cam1')
flow_capture = cv2.VideoCapture('media/flownet2.mp4')
capture_size = utils.get_capture_size(flow_capture)

output = utils.get_output('detection', capture_size=(capture_size[0] * 2, capture_size[1]))
# feature_detector = LucasKanade(capture, output)
color = (0, 255, 0)
i = 0
enable_preview = False
prev_estimate = None
ious = np.zeros(N)
ious[:] = np.nan
failed_detections = 0


# print(utils.get_frame_count(orig_capture))
# print(utils.get_frame_count(flow_capture))
# print(utils.get_fps(orig_capture))
# print(utils.get_fps(flow_capture))

try:
    ann_path = utils.get_cenek_annotation(sequence, 'cam1')
    f = open(ann_path, 'r')

    while i < N - 1:
        if i % int(N / 10) == 0:
            print('{:.2f}'.format(i / N * 100) + '%', i, '/', N)

        # frame = feature_detector.process()
        s1, orig_frame = orig_capture.read()
        s2, flow_frame = flow_capture.read()

        if not s1 or not s2:
            break

        # if i < 1500:
        #     i += 1
        #     line = f.readline()
        #     continue

        orig_frame = orig_frame[::-1, ::-1, :]

        # ann_path = utils.get_midgard_annotation(sequence, i)

        line = f.readline()
        values = line.replace('  ', ' ').replace('  ', ' ').split(' ')
        size = 60
        ground_truth = utils.Rectangle.from_center((int(float(values[2])), int(float(values[1]))), (size, size))
        ground_truth.to_int()

        # with open(ann_path, 'r') as f:
        #     for line in f.readlines():
        #         values = line.split(',')
        #         values = [round(float(x)) for x in values]
        #         topleft = (values[1], values[2])
        #         bottomright = (values[1] + values[3], values[2] + values[4])
        #         orig_frame = cv2.rectangle(orig_frame, topleft, bottomright, color, 2)
        #         flow_frame = cv2.rectangle(flow_frame, topleft, bottomright, color, 2)

        flow_hsv = cv2.cvtColor(flow_frame, cv2.COLOR_BGR2HSV)
        # flow_hsv[..., 0] = flow_hsv[..., 1]
        # flow_hsv[..., 1] = 255
        # flow_hsv[..., 2] = 255
        # flow_frame = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR)

        v = flow_hsv[..., 1]
        v_max = v.max()
        max_x, max_y = [], []
        window_size = 60
        center = np.unravel_index(v.argmax(), v.shape)
        threshold = 50

        for x in range(max(0, center[0] - window_size), min(v.shape[0], center[0] + window_size)):
            for y in range(max(0, center[1] - window_size), min(v.shape[1], center[1] + window_size)):
                if v[x, y] > v_max - threshold:
                    max_x.append(x)
                    max_y.append(y)

        max_x = np.array(max_x)
        max_y = np.array(max_y)
        estimate = utils.Rectangle.from_center((int(np.average(max_x)) + size // 2, int(np.average(max_y))), (size, size))
        estimate_int = utils.Rectangle(estimate.topleft, estimate.size)
        estimate_int.to_int()

        background = flow_hsv.copy()
        background[estimate_int.get_bottom():estimate_int.get_top(), estimate_int.get_left():estimate_int.get_right()] = np.nan
        SNR = v_max ** 2.0 / np.var(background)

        iou = utils.Rectangle.calculate_iou(ground_truth, estimate) / estimate.get_area()
        ious[i] = iou
        if iou < 0.0:
            failed_detections += 1

        if prev_estimate is not None:
            diff = np.array(estimate.get_center()) - np.array(prev_estimate.get_center())
            diff_mag = np.sqrt(diff[0] ** 2.0 + diff[1])

            if diff_mag < 100.0:
                orig_frame = cv2.rectangle(np.array(orig_frame), estimate_int.get_topleft(), estimate_int.get_bottomright(), (0, 0, 0), 5)

                cv2.putText(orig_frame,
                    f'SNR = {SNR:.02f}, IoU={iou:.02f}',
                    (estimate_int.get_top(), estimate_int.get_left() - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2
                )
            elif ground_truth.get_center() != (0.0, 0.0):
                failed_detections += 1


        if ground_truth.get_center() != (0.0, 0.0):
            orig_frame = cv2.rectangle(np.array(orig_frame), ground_truth.get_topleft(), ground_truth.get_bottomright(), (0, 255, 0), 3)

        prev_estimate = estimate
        orig_frame = cv2.resize(orig_frame, capture_size, interpolation = cv2.INTER_AREA)
        output.write(np.hstack((orig_frame, flow_frame)))
        i += 1

        if enable_preview:
            cv2.imshow('original', orig_frame)
            cv2.imshow('flow', flow_frame)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

finally:
    orig_capture.release()
    flow_capture.release()
    output.release()
    cv2.destroyAllWindows()
    f.close()

    plt.hist(ious, np.linspace(0.0, 1.0, 20))
    plt.grid()
    plt.xlabel('IoU')
    plt.ylabel('Frequency (frames)')
    plt.savefig('media/output/ious.png', bbox_inches = 'tight')

    print(f'failed detections: {failed_detections} / {N}, {failed_detections / N * 100:.03f}%')
