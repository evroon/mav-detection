import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

# for d in os.listdir('data/citypark'):
#     img_count = len(glob.glob(f'data/citypark/{d}/images/*'))
#     # if img_count < 25:
#     print(d, len(glob.glob(f'data/citypark/{d}/images/*')))

def plot_states(observable: str) -> None:
    dir = 'data/citypark-moving/soccerfield-north-medium-5.0-10-default/states'
    states = glob.glob(f'{dir}/*.json')
    states = [x for x in states if 'timestamp' not in x]
    states.sort()
    start = 0

    N = len(states)
    t = np.zeros(N)
    x = np.zeros((N, 3))

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

            lin_acc = state_dict['kinematics_estimated'][observable]

            if i == 0:
                start = state_dict['timestamp']

            t[i] = (state_dict['timestamp'] - start) / 1e9
            x[i, 0] = lin_acc['x_val']
            x[i, 1] = lin_acc['y_val']
            x[i, 2] = lin_acc['z_val']

    plt.figure()
    plt.xlabel('Time [s]')
    plt.ylabel(f'{observable_pretty} [{y_unit}]')
    plt.grid()

    for i, label in zip(range(3), ['x', 'y', 'z']):
        if observable in ['position', 'orientation']:
            x[:, i] -= x[0, i]

        if observable in ['orientation', 'angular_velocity', 'angular_acceleration']:
            x[:, i] *= 180 / np.pi

        plt.plot(t, x[:, i], label=label)

    plt.legend()
    plt.savefig(f'media/states/{observable}.png', bbox_inches='tight')

plot_states('position')
plot_states('linear_velocity')
plot_states('linear_acceleration')

plot_states('orientation')
plot_states('angular_velocity')
plot_states('angular_acceleration')
