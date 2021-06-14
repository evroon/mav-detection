import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json
from matplotlib import cm
from matplotlib.ticker import LinearLocator

base_path = os.getenv('SIMDATA_PATH')
validation_data = glob.glob(f'{base_path}/mountains-line-cloudsv2/lake-line-0-north-low-5.0-0.*-default/validation.npy')
validation_data.sort()

# Plot errorbars only for the optimal threshold.
plt.grid()
plt.xlabel(r'$\phi$ [deg]')
plt.ylabel('True Positive Rate')
plt.ylim(0, 1.0)

colors = ['g', 'b', 'indigo', 'purple', 'r', 'orange', 'yellow', 'pink', 'darkgrey']

data_per_velocity = {}
tpr_at_180_list = []

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

    label = f'flow: {flow_x[0]:.01f} px/frame'# ({velocity:.1f}m/s)' #±{flow_x[1]:.01f}

    if isinstance(avg_std, np.ndarray):
        data_per_velocity[flow_x[0]] = (np.absolute(avg_std[2:, 0]), avg_std[2:, 1])
        tpr_sample = np.nan_to_num(avg_std[2:30, 1], 0)
        tpr_at_180_list.append((np.average(tpr_sample), np.std(tpr_sample)))

        plt.errorbar(avg_std[:, 0], avg_std[:, 1],# yerr=avg_std[:, 2],
            marker='o', markersize=6, capsize=3, barsabove=False, label=label, zorder=1, color=colors[i])

if len(data_per_velocity.keys()) < 1:
    raise ValueError('Could not load data.')

plt.legend()
plt.savefig('tpr_vs_phi.png', bbox_inches='tight')

flows = np.array(list(data_per_velocity.keys()))
x = data_per_velocity[flows[0]][0]
X, Y = np.meshgrid(x, flows)
Z = np.zeros_like(Y)

for i in range(len(flows)):
    Z[i, ...] = data_per_velocity[flows[i]][1]

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
ax.set_ylabel(r'OF magnitude [px/frame]')
ax.set_zlabel('True Positive Rate')

ax.set_ylim(bottom=0)
ax.set_xlim(180, 0)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.7, aspect=10, ax=ax, pad=0.12)

plt.savefig('tpr_flow_vs_phi.png', bbox_inches='tight')
plt.savefig('tpr_flow_vs_phi.eps', bbox_inches='tight')

tpr_at_180 = np.array(tpr_at_180_list)
plt.figure()
plt.grid()
plt.xlabel(r'OF magnitude [px/frame]')
plt.ylabel(r'TPR')
plt.plot(flows, tpr_at_180[..., 0])

plt.errorbar(flows, tpr_at_180[..., 0], yerr=tpr_at_180[..., 1],
    marker='o', markersize=6, capsize=3, barsabove=False, zorder=1, color='black')

plt.ylim(0, 1)
plt.xlim(left=0)
plt.savefig('tpr_vs_flow.png', bbox_inches='tight')
