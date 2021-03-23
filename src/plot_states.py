import os
import glob
import json
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# for d in os.listdir('data/citypark'):
#     img_count = len(glob.glob(f'data/citypark/{d}/images/*'))
#     # if img_count < 25:
#     print(d, len(glob.glob(f'data/citypark/{d}/images/*')))

def plot_states(observable: str) -> np.ndarray:
    dir = 'data/mountains-stationary/lake-north-low-2.5-10-default/states'
    states = glob.glob(f'{dir}/*.json')
    states = [x for x in states if 'timestamp' not in x]
    states.sort()
    start = 0
    clockspeed = 1

    N = len(states)
    t = np.zeros(N)
    x = np.zeros((N, 4))

    y_unit = {
        'position': '$m$',
        'linear_velocity': '$m/s$',
        'linear_acceleration': '$m/s^2$',
        'orientation': '$\deg$',
        'angular_velocity': '$\deg/s$',
        'angular_acceleration': '$\deg/s^2$',
    }[observable]
    observable_pretty = observable.replace('_', ' ').capitalize()

    for i, state in enumerate(states):
        with open(f'{state}', 'r') as f:
            state_dict = json.load(f)

            data = state_dict['imu'][observable]

            if i == 0:
                start = state_dict['imu']['time_stamp']

            t[i] = (state_dict['timestamp'] - start) / 1e9 * clockspeed
            x[i, 0] = data['x_val']
            x[i, 1] = data['y_val']
            x[i, 2] = data['z_val']
            if 'w_val' in data:
                x[i, 3] = data['w_val']

    plt.figure()
    plt.xlabel('Time [s]')
    plt.ylabel(f'{observable_pretty} [{y_unit}]')
    plt.grid()

    # Convert quaternions to euler angles.
    if observable == 'orientation':
        for k in range(x.shape[0]):
            rot = Rotation.from_quat(x[k, :])
            x[k, :3] = rot.as_euler('xyz', degrees=True)

    for i, label in enumerate(['x', 'y', 'z']):
        # if observable in ['position', 'orientation']:
        #     x[:, i] -= x[0, i]

        if observable in ['angular_velocity', 'angular_acceleration']:
            x[:, i] *= 180 / np.pi

        plt.plot(t, x[:, i], label=label)

    plt.legend()
    plt.savefig(f'media/states/{observable}.png', bbox_inches='tight')
    return t


plot_states('linear_acceleration')
plot_states('orientation')
t = plot_states('angular_velocity')

# plot_states('position')
# plot_states('linear_velocity')
# plot_states('angular_acceleration')

deltas = np.diff(t)
print(f'average delta time: {np.average(deltas):.03f}s ({1/np.average(deltas):.02f}Hz) std: {np.std(deltas):.03f}s')
