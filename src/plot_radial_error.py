import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import glob
import os

from typing import Tuple

dump_path = 'results/rad_err_combined.npy'
start = 20
end = 120
N = end - start
resolution = 1920 * 1024

def gather_data() -> Tuple[np.ndarray, np.ndarray]:
    files = glob.glob('results/mag_vs_rad/mag_vs_rad_err_*.npy')
    files.sort()
    x, y = np.zeros(resolution * N), np.zeros(resolution * N)
    i = 0

    for fp in files[start:end]:
        flow_mag, flow_error_radial = np.load(fp)
        x[i:i + len(flow_mag)] = flow_mag
        y[i:i + len(flow_mag)] = flow_error_radial
        i += len(flow_mag)

    x = x[:i+1]
    y = y[:i+1]
    np.save(dump_path, [x, y])
    print('finished writing combined results.')
    return x, y

if os.path.exists(dump_path):
    x, y = np.load(dump_path)
else:
    x, y = gather_data()

bins = (np.linspace(0, 10, 160), np.linspace(-20, 20, 160))
counts, _, _ = np.histogram2d(x, y, bins=bins)

fig, ax = plt.subplots()
plt.xlabel('OF magnitude [px/frame]')
plt.ylabel('Radial error in OF [deg]')

pcm = ax.pcolor(bins[0], bins[1], counts.T / N,
                   norm=colors.LogNorm(vmin=1, vmax=counts.max()),
                   cmap='jet')
fig.colorbar(pcm, ax=ax, extend='max')

x_theoretic = np.linspace(0, 10, 1000)
y_theoretic = 0.5 + 8 / x_theoretic
ax.plot(x_theoretic, 0.25 + y_theoretic, color='white', ls='--')
ax.plot(x_theoretic, 0.25 - y_theoretic, color='white', ls='--')
ax.set_ybound(np.min(bins[1]), np.max(bins[1]))
ax.text(5, -17, r'Fit: $0.25 \pm (0.5 + \frac{8}{|OF|})$', fontsize=12)

plt.savefig('results/mag_vs_rad_err.png', bbox_inches='tight')
plt.savefig('results/mag_vs_rad_err.eps', bbox_inches='tight')
