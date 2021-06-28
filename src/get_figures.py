import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from typing import Tuple

base_path = os.getenv('SIMDATA_PATH')
use_foe_data = False

if use_foe_data:
    validation_data = glob.glob(f'{base_path}/mountains-demo/lake-foe_demo_*-0-north-low-5.0-*-default/validation.npy')
    validation_data.sort()
    match_pattern = '^.+lake-.+-0-north-low-(.+)-(.+)-default.+$'
else:
    validation_data = glob.glob(f'{base_path}/mountains-line-cloudsv2/lake-line-0-north-low-5.0-0.*-default/validation.npy')
    validation_data.sort()
    validation_data = validation_data[:-1]
    match_pattern = '^.+lake-line-0-north-low-(.+)-(.+)-default.+$'

data_per_velocity = {}
tpr_at_180_list = []
tpr_at_180_fixed_list = []
fpr_at_180_list = []
fpr_at_180_fixed_list = []
# foe_error_list = []

# Plot errorbars only for the optimal threshold.
plt.grid()
plt.xlabel(r'$\kappa$ [deg]')
plt.ylabel('True Positive Rate')
plt.ylim(0, 1.0)

colors = ['g', 'b', 'indigo', 'purple', 'darkgrey', 'orange', 'yellow', 'pink', 'r']

for i, d in enumerate(validation_data):
    matches = re.findall(match_pattern, d)
    distance = float(matches[0][0])
    velocity = float(matches[0][1])

    validation = np.load(d, allow_pickle=True)
    if len(validation) != 14:
        continue

    tpr = validation[:2]
    size = validation[2:4]
    flow_x = validation[4:6]
    flow_y = validation[6:8]
    tpr = validation[8]
    tpr_fixed = validation[9]
    fpr = validation[10]
    fpr_fixed = validation[11]
    # foe_error = validation[14]

    if isinstance(tpr, np.ndarray):
        data_per_velocity[flow_x[0]] = (
            np.absolute(tpr[2:, 0]), tpr[2:, 1],
            np.absolute(fpr[2:, 0]), fpr[2:, 1],
        )

        tpr_sample = np.nan_to_num(tpr[2:30, 1], 0)
        tpr_sample_fixed = np.nan_to_num(tpr_fixed[2:30, 1], 0)
        fpr_sample = np.nan_to_num(fpr[2:30, 1], 0)
        fpr_sample_fixed = np.nan_to_num(fpr_fixed[2:30, 1], 0)

        tpr_at_180_list.append((np.average(tpr_sample), np.std(tpr_sample)))
        tpr_at_180_fixed_list.append((np.average(tpr_sample_fixed), np.std(tpr_sample_fixed)))
        fpr_at_180_list.append((np.average(fpr_sample), np.std(fpr_sample)))
        fpr_at_180_fixed_list.append((np.average(fpr_sample_fixed), np.std(fpr_sample_fixed)))
        # foe_error_list.append(foe_error)

        label = f'OF: {flow_x[0]:.01f} px/frame'

        if i < 3 or i == len(validation_data) - 2:
            plt.errorbar(np.abs(tpr[:-1, 0]), tpr[:-1, 1],# yerr=avg_std[:, 2],
                marker='o', markersize=6, capsize=3, barsabove=False, label=label, zorder=1, color=colors[i])

def plot_3d() -> None:
    """Plots tpr_flow_vs_phi.png"""
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
    ax.set_xlabel(r'$\kappa$ [deg]')
    ax.set_ylabel(r'OF magnitude [px/frame]')
    ax.set_zlabel('True Positive Rate')

    ax.set_ylim(bottom=0)
    ax.set_xlim(180, 0)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.7, aspect=10, ax=ax, pad=0.12)

    plt.savefig('media/tpr_flow_vs_phi.png', bbox_inches='tight')
    plt.savefig('media/tpr_flow_vs_phi.eps', bbox_inches='tight')


def plot_vs_magnitude(type: str, pr_at_180_list: list, pr_at_180_fixed_list: list, ax: plt.axis) -> Tuple[plt.axis, plt.axis]:
    """Plots tpr_vs_flow.png and fpr_vs_flow.png

    Args:
        type (str): 'TPR' or 'FPR'
        pr_at_180_list (list): list of TPR/FPR values around phi=180
        pr_at_180_fixed_list (list): list of TPR/FPR values around phi=180 for fixed threshold
    """
    pr_at_180 = np.array(pr_at_180_list)
    pr_at_180_fixed = np.array(pr_at_180_fixed_list)
    color = 'tab:blue' if type == 'TPR' else 'tab:green'

    ax.set_ylabel(type, color=color)

    ln1 = ax.errorbar(flows, pr_at_180[..., 0], yerr=pr_at_180[..., 1], label=f'{type}, dynamic',
        marker='o', markersize=6, capsize=3, barsabove=False, zorder=1, color=color)

    ln2 = ax.errorbar(flows, pr_at_180_fixed[..., 0], yerr=pr_at_180_fixed[..., 1], label=f'{type}, fixed',
        marker='o', markersize=6, capsize=3, barsabove=False, zorder=1, color=color, ls='--')

    print(ln1)

    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylim(0, 1 if type == 'TPR' else 0.03)
    return ln1, ln2

def plot_foe_hist(foe_error: np.ndarray) -> None:
    outlier_threshold = 50.0
    plt.figure()
    # plt.subplot(1, 2, 1)

    colors = [
        ['tab:red', 'darkred'],
        ['tab:green', 'darkgreen'],
        ['tab:blue', 'darkblue'],
    ]

    # Order is determined by glob.
    directions = [
        'center',
        'left',
        'right',
    ]


    means = [
        (2.81, -7.18),
        (9.16, -7.44),
        (-8.09, -2.37),
    ]
    stds = [
        (4.9, 6.4),
        (9.6, 5.6),
        (6.5, 5.0),
    ]

    fig, axes = plt.subplots(nrows=2, ncols=1)

    for i in range(foe_error.shape[0]):
        label = f'{directions[i]} ({means[i][0]:.02f}$\pm${stds[i][0]:.01f} px)'
        axes[0].hist(foe_error[i, ..., 0], np.linspace(-outlier_threshold, outlier_threshold, 40), histtype=u'step', label=label, color=colors[i][0])

        label = f'{directions[i]} ({means[i][1]:.02f}$\pm${stds[i][1]:.01f} px)'
        axes[1].hist(foe_error[i, ..., 1], np.linspace(-outlier_threshold, outlier_threshold, 40), histtype=u'step', label=label, color=colors[i][0])

    # Change order of legend items.
    order = [1, 0, 2]

    for i, ax in enumerate(axes):
        ax_label = 'x' if i == 0 else 'y'
        handles, labels = ax.get_legend_handles_labels()

        ax.set_xlabel(f'FoE error ({ax_label}) [pixels]')
        ax.grid()
        ax.set_ylabel('Frequency [frames]')
        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

    fig.tight_layout()
    plt.savefig('media/output/foe-error.eps', bbox_inches='tight')
    plt.savefig('media/output/foe-error.png', bbox_inches='tight')


plt.legend()
plt.xlim(180, 0)
plt.savefig('media/tpr_vs_phi.png', bbox_inches='tight')
plt.savefig('media/tpr_vs_phi.eps', bbox_inches='tight')

if len(data_per_velocity.keys()) < 1:
    raise ValueError('Could not load data.')

flows = np.array(list(data_per_velocity.keys()))

fig, ax1 = plt.subplots()
plt.grid()
ax1.set_xlabel(r'OF magnitude [px/frame]')
ln1, ln2 = plot_vs_magnitude('TPR', tpr_at_180_list, tpr_at_180_fixed_list, ax1)

ax2 = ax1.twinx()
ln3, ln4 = plot_vs_magnitude('FPR', fpr_at_180_list, fpr_at_180_fixed_list, ax2)

lns = [ln1, ln2, ln3, ln4]
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=4, bbox_to_anchor=(1.1, 1.02), ncol=4)
ax1.set_xlim(left=0)

plt.savefig('media/tpr_fpr_vs_flow.png', bbox_inches='tight')
plt.savefig('media/tpr_fpr_vs_flow.eps', bbox_inches='tight')

# plot_foe_hist(np.array(foe_error_list))
