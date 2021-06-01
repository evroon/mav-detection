import glob
import numpy as np
import matplotlib.pyplot as plt

validation_data = glob.glob('/home/erik/tno/datasets/data/mountains-line/*/validation.npy')
validation_data.sort()

for d in validation_data:
    validation = np.load(d, allow_pickle=True)
    tpr = validation[:2]
    size = validation[2:4]
    flow_x = validation[4:6]
    flow_y = validation[6:8]

    label = f'{flow_x[0]} (±{flow_x[1]}) px/s'
    plt.errorbar(size[0], size[1], tpr[0], tpr[1],
                marker='o', markersize=6, capsize=3,
                barsabove=False, label=label, zorder=1, color='indigo')
    plt.savefig('test')
