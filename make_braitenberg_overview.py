import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(SCRIPT_DIR, 'output'), exist_ok=True)

import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.collections import LineCollection
import numpy as np

with open(os.path.join(SCRIPT_DIR, 'output', 'sim_log.json')) as f:
    log = json.load(f)

img = mpimg.imread(os.path.join(SCRIPT_DIR, 'micropsi_core', 'world', 'island', 'resources', 'groundmaps', 'psi_1.png'))
SCALE = 8

xs = np.array([d['x'] for d in log])
ys = np.array([d['y'] for d in log])
steps = np.array([d['step'] for d in log])

fig = plt.figure(figsize=(13, 6.2), dpi=150)
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.25)

# --- Left: full trajectory, colored by time ---
axm = fig.add_subplot(gs[0])
axm.imshow(img, extent=[0, 256*SCALE, 256*SCALE, 0])

points = np.array([xs, ys]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap='plasma', linewidth=2.4)
lc.set_array(steps)
axm.add_collection(lc)

axm.scatter([xs[0]], [ys[0]], s=90, color='white', edgecolor='black', zorder=5, label='start (step 0)')
axm.scatter([xs[-1]], [ys[-1]], s=110, color='#ffdd00', edgecolor='#7a5a00', zorder=5, label='settled (step 399)')
axm.legend(loc='lower right', fontsize=8, framealpha=0.9)
axm.set_xlim(0, 256*SCALE)
axm.set_ylim(256*SCALE, 0)
axm.set_xticks([]); axm.set_yticks([])
axm.set_title("Full 400-step trajectory \u2014 colored by time", fontsize=11.5, weight='bold')
cbar = fig.colorbar(lc, ax=axm, fraction=0.046, pad=0.03)
cbar.set_label('mind-cycle step', fontsize=8.5)

# --- Right: modulator time series, full run ---
axe = fig.add_subplot(gs[1])
emo_keys = ['emo_activation', 'emo_pleasure', 'emo_resolution', 'emo_competence', 'emo_valence']
colors = {'emo_activation': '#c0392b', 'emo_pleasure': '#27ae60', 'emo_resolution': '#2980b9',
          'emo_competence': '#8e44ad', 'emo_valence': '#d68910'}
labels = {'emo_activation': 'Activation', 'emo_pleasure': 'Pleasure', 'emo_resolution': 'Resolution',
          'emo_competence': 'Competence', 'emo_valence': 'Valence'}
for k in emo_keys:
    series = [d.get(k, 0) for d in log]
    axe.plot(steps, series, color=colors[k], lw=1.9, label=labels[k])

# mark the point where movement stops for good (search from the end backwards)
settle_step = None
for i in range(len(log)-1, 0, -1):
    if abs(xs[i]-xs[i-1]) > 0.001 or abs(ys[i]-ys[i-1]) > 0.001:
        settle_step = steps[i]
        break
if settle_step is not None:
    axe.axvline(settle_step, color='#555555', ls='--', lw=1.2)
    axe.text(settle_step-8, -1.9, f'agent settles\n(step {settle_step})', fontsize=8.3, color='#555555', ha='right')

axe.axhline(0, color='#aaaaaa', lw=0.6)
axe.set_xlim(0, 400)
axe.set_ylim(-2.2, 2.2)
axe.set_xlabel('mind-cycle step', fontsize=10)
axe.set_title('Doernerian emotional modulators \u2014 full run', fontsize=11.5, weight='bold')
axe.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
axe.grid(alpha=0.25)

fig.suptitle("Real micropsi2 (joschabach/micropsi2) simulation: Braitenberg agent on D\u00f6rner's Island",
             fontsize=11, weight='bold', y=1.01)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'output', 'micropsi2_island_overview.png'), dpi=150, bbox_inches='tight', facecolor='white')
print('saved overview')
