import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from matplotlib import cm
from matplotlib.ticker import LinearLocator

base_path = os.getenv('SIMDATA_PATH')
validation_data = glob.glob(f'{base_path}/mountains-line-cloudsv2/lake-line-0-north-low-5.0-0.*-default/validation.npy')
validation_data.sort()

data_per_velocity = {}
tpr_at_180_list = []
tpr_at_180_fixed_list = []
fpr_at_180_list = []
fpr_at_180_fixed_list = []

for i, d in enumerate(validation_data[:1]):
    matches = re.findall('^.+lake-line-0-north-low-(.+)-(.+)-default.+$', d)
    distance = float(matches[0][0])
    velocity = float(matches[0][1])

    validation = np.load(d, allow_pickle=True)
    tpr = validation[:2]
    size = validation[2:4]
    flow_x = validation[4:6]
    flow_y = validation[6:8]
    tpr = validation[8]
    tpr_fixed = validation[9]
    fpr = validation[10]
    fpr_fixed = validation[11]

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

if len(data_per_velocity.keys()) < 1:
    raise ValueError('Could not load data.')

flows = np.array(list(data_per_velocity.keys()))

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
    ax.set_xlabel(r'$\phi$ [deg]')
    ax.set_ylabel(r'OF magnitude [px/frame]')
    ax.set_zlabel('True Positive Rate')

    ax.set_ylim(bottom=0)
    ax.set_xlim(180, 0)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.7, aspect=10, ax=ax, pad=0.12)

    plt.savefig('media/tpr_flow_vs_phi.png', bbox_inches='tight')
    plt.savefig('media/tpr_flow_vs_phi.eps', bbox_inches='tight')


def plot_vs_magnitude(type: str, pr_at_180_list: list, pr_at_180_fixed_list: list) -> None:
    """Plots tpr_vs_flow.png and fpr_vs_flow.png

    Args:
        type (str): 'TPR' or 'FPR'
        pr_at_180_list (list): list of TPR/FPR values around phi=180
        pr_at_180_fixed_list (list): list of TPR/FPR values around phi=180 for fixed threshold
    """
    pr_at_180 = np.array(pr_at_180_list)
    pr_at_180_fixed = np.array(pr_at_180_fixed_list)

    plt.figure()
    plt.grid()
    plt.xlabel(r'OF magnitude [px/frame]')
    plt.ylabel(type)

    plt.errorbar(flows, pr_at_180[..., 0], yerr=pr_at_180[..., 1], label='dynamic threshold',
        marker='o', markersize=6, capsize=3, barsabove=False, zorder=1, color='black')

    plt.errorbar(flows, pr_at_180_fixed[..., 0], yerr=pr_at_180_fixed[..., 1], label='fixed threshold',
        marker='o', markersize=6, capsize=3, barsabove=False, zorder=1, color='blue')

    plt.ylim(0, 1)
    plt.xlim(left=0)
    plt.legend()
    plt.savefig(f'media/{type.lower()}_vs_flow.png', bbox_inches='tight')


plot_vs_magnitude('TPR', tpr_at_180_list, tpr_at_180_fixed_list)
plot_vs_magnitude('FPR', fpr_at_180_list, fpr_at_180_fixed_list)
