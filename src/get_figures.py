import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json
from matplotlib import cm
from matplotlib.ticker import LinearLocator

validation_data = glob.glob(os.getenv('SIMDATA_PATH') + '/mountains-line-clouds/lake-line-0-north-low-5.0-0.*-default/validation.npy')
validation_data.sort()

# Plot errorbars only for the optimal threshold.
plt.grid()
plt.xlabel(r'$\phi$ [deg]')
plt.ylabel('True Positive Rate')
plt.ylim(0, 1.0)

colors = ['g', 'b', 'indigo', 'purple']
flows = [2.02, 4.2, 8.77]
flows = [2.0, 4.0, 8.0]

data_per_velocity = {}

for i, d in enumerate(validation_data):
    matches = re.findall('^.+lake-line-0-north-low-(.+)-(.+)-default.+$', d)
    distance = float(matches[0][0])
    velocity = float(matches[0][1])

    validation = np.load(d, allow_pickle=True)
    tpr = validation[:2]
    size = validation[2:4]
    flow_x = validation[4:6]
    flow_y = validation[6:8]
    avg_std = validation[8]

    label = f'flow: {flow_x[0]:.01f} px/frame ({velocity:.1f}m/s)' #±{flow_x[1]:.01f}

    if isinstance(avg_std, np.ndarray):
        data_per_velocity[flow_x[0]] = (np.absolute(avg_std[2:, 0]), avg_std[2:, 1])

        plt.errorbar(avg_std[:, 0], avg_std[:, 1],# yerr=avg_std[:, 2],
            marker='o', markersize=6, capsize=3, barsabove=False, label=label, zorder=1, color=colors[i])

plt.savefig('test.png')

y = np.array(list(data_per_velocity.keys()))
x = data_per_velocity[y[0]][0]
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(Y)

for i in range(len(y)):
    Z[i, ...] = data_per_velocity[y[i]][1]

Z = np.nan_to_num(Z, 0)

# Plot the surface.
plt.figure()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, cmap=cm.jet,
                       linewidth=0, antialiased=False, vmax=1)

# Customize the z axis.
ax.set_zlim(0, 1)
ax.zaxis.set_major_locator(LinearLocator(11))

# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.01f}')
ax.set_xlabel(r'$\phi$ [deg]')
ax.set_ylabel(r'flow [px/frame]')
ax.set_zlabel('True Positive Rate')

# ax.set_ylim(10, 0)
ax.set_xlim(180, 0)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.7, aspect=10, ax=ax, pad=0.12)

plt.savefig(f'tpr_vs_phi.png', bbox_inches='tight')
plt.savefig(f'tpr_vs_phi.eps', bbox_inches='tight')
