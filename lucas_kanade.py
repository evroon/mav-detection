import numpy as np
import cv2
import argparse
import utils
import matplotlib.pyplot as plt

plt.rcParams['axes.axisbelow'] = True


class LucasKanade:
    def __init__(self, capture, output):
        # Script based on: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

        self.num_corners = 2000
        self.minimum_num_corners = self.num_corners // 3
        self.total_num_corners = self.num_corners + self.minimum_num_corners
        self.capture = capture
        self.output = output

        # Parameters for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=self.num_corners,
                                   qualityLevel=0.2,
                                   minDistance=7,
                                   blockSize=7)

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(21, 21),
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        # Create some random colors
        self.color = np.random.randint(0, 255, (self.total_num_corners, 3))

        _, self.old_frame = self.capture.read()
        self.capture_size = self.old_frame.shape

        self.time = 0
        self.trace = np.zeros((self.total_num_corners, 1000), dtype=np.int)
        self.roll_back = 20
        self.num_features = 0
        self.random_lines = np.random.randint(0, self.total_num_corners, self.total_num_corners)
        self.treshold = np.cos(15 * np.pi / 180.0)
        self.flow_per_distance = np.zeros((self.total_num_corners, 4)) # Store distance, flow magnitude pairs
        self.enable_plots = False
        self.p0 = None

    def process(self):
        ret, frame = self.capture.read()

        if not ret:
            return

        self.mask = np.zeros_like(self.old_frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.num_features < self.minimum_num_corners:
            # Take first frame and find corners in it
            self.old_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)

            current_time = np.max(self.trace[:, 0])
            mask = self.trace[:, 0] == current_time
            recycled_features = self.trace[mask]

            if len(recycled_features) <= self.minimum_num_corners:
                print('Finding new features, recycling {} features'.format(self.num_features))
                self.trace[self.num_corners:self.num_corners+recycled_features.shape[0]] = recycled_features

            self.trace[:self.num_corners, :] = np.zeros(1000)

            new_features = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
            if self.p0 is None:
                self.p0 = new_features
            else:
                recycled_p0 = self.p0[mask[:len(self.p0)]]
                self.p0 = np.zeros((len(new_features) + len(recycled_features), new_features.shape[1], new_features.shape[2]))
                self.p0[:len(new_features), ...] = new_features
                self.p0[len(new_features):, ...] = recycled_p0

            print('Found {} new features'.format(self.p0.shape))

        # Calculate optical flow
        good_new, status, _ = cv2.calcOpticalFlowPyrLK(
            self.old_gray, frame_gray, self.p0, None, **self.lk_params)

        # Select good points
        self.num_features = 0
        lines = []
        intersections = np.zeros((len(good_new), 2))

        # Draw the new tracks
        for i, (new, old) in enumerate(zip(good_new, self.p0)):
            if status[i] != 1:
                continue

            a, b, c, d = *new.ravel(), *old.ravel()
            a, b, c, d = [int(x) for x in [a, b, c, d]]
            color = self.color[i].tolist()

            frame = cv2.circle(frame, (a, b), 4, color, -1)

            l = self.trace[i, 0] + 1
            self.trace[i, l] = c
            self.trace[i, l+1] = d
            self.trace[i, 0] += 2
            color = self.color[i].tolist()
            # m, n = self.trace[i, 1:3]
            self.num_features += 1

            # for j in range(3, l+2, 2):
                # self.mask = cv2.line(self.mask, (m, n), (self.trace[i, j], self.trace[i, j+1]), color, 2)
                # m, n = self.trace[i, j], self.trace[i, j+1]

            if l >= 3:
                k = 1 if l < 1 + self.roll_back * 2 else l - self.roll_back * 2
                a, b = self.trace[i, l:l+2]
                c, d = self.trace[i, k:k+2]

                diff_x = float(c) - float(a)
                diff_y = float(d) - float(b)

                diff = np.array([diff_x, diff_y])
                length = np.linalg.norm(diff)

                # if length < 5: # or b < self.capture_size[1] // 4:
                    # continue

                new_xy = np.array([a, b]) + diff
                new_xy = new_xy.astype(np.uint16)

                while new_xy[1] < 0.0 or new_xy[1] > self.capture_size[1]:
                    diff /= 2.0
                    new_xy = np.array([a, b]) + diff
                    new_xy = new_xy.astype(np.uint16)

                result = ((a, b), (new_xy[0], new_xy[1]))
                lines.append(result)

        # Sample intersections between flow vectors.
        for i, line_a in enumerate(lines):
            line_b = lines[np.random.randint(0, len(lines))]
            int_x, int_y = utils.line_intersection(line_a, line_b)
            intersections[i, :] = int_x, int_y

        intersections = intersections[intersections[:, 0] != 0.0, :]
        FoE = np.median(intersections, axis=0).astype(np.uint16)
        FoE = (FoE[0], FoE[1])
        frame = cv2.circle(frame, FoE, 20, [0, 42, 255], -1)

        for i, line in enumerate(lines):
            # Calculate angle between line from FoE and feature with the flow vector of the feature.
            diff1 = np.asarray(line[0]) - np.asarray(line[1])
            diff2 = np.asarray(line[0]) - np.asarray(FoE)

            flow_magnitude = np.linalg.norm(diff1)
            img_distance = np.linalg.norm(diff2)
            flow_angle = np.arctan2(diff1[1], diff1[0]) * 180.0 / np.pi
            img_angle = np.arctan2(diff2[1], diff2[0]) * 180.0 / np.pi

            angle_diff = np.dot(diff1, diff2) / (flow_magnitude * img_distance)
            color = [0, 255, 0]

            if angle_diff < self.treshold:
                color = [0, 0, 255]

            self.mask = cv2.line(self.mask, line[0], line[1], color, 2)
            self.flow_per_distance[i, 0] = img_distance
            self.flow_per_distance[i, 1] = flow_magnitude
            self.flow_per_distance[i, 2] = img_angle
            self.flow_per_distance[i, 3] = flow_angle

        if self.enable_plots:
            plt.figure(1)
            plt.clf()
            plt.xlabel('Distance (pixels)')
            plt.ylabel('Flow magnitude (pixels)')
            plt.grid()
            plt.scatter(self.flow_per_distance[:, 0], self.flow_per_distance[:, 1])

            plt.figure(2)
            plt.clf()
            plt.xlabel('Angle (pixels)')
            plt.ylabel('Flow angle (pixels)')
            plt.grid()
            plt.scatter(self.flow_per_distance[:, 2], self.flow_per_distance[:, 3])
            plt.pause(0.01)

        # Now update the previous frame and previous points
        self.old_gray = frame_gray.copy()
        self.p0 = good_new.reshape(-1, 1, 2)

        self.time += 1
        self.old_frame = frame
        result = cv2.add(frame, self.mask)

        return result
