import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(SCRIPT_DIR, 'output'), exist_ok=True)

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.image as mpimg
import imageio.v2 as imageio
import io

with open(os.path.join(SCRIPT_DIR, 'output', 'sim_log.json')) as f:
    log = json.load(f)

img = mpimg.imread(os.path.join(SCRIPT_DIR, 'micropsi_core', 'world', 'island', 'resources', 'groundmaps', 'psi_1.png'))
SCALE = 8  # world-units per groundmap pixel

xs = [d['x'] for d in log]
ys = [d['y'] for d in log]

emo_keys = ['emo_activation', 'emo_pleasure', 'emo_resolution', 'emo_competence', 'emo_valence']
colors = {'emo_activation': '#c0392b', 'emo_pleasure': '#27ae60', 'emo_resolution': '#2980b9',
          'emo_competence': '#8e44ad', 'emo_valence': '#d68910'}
labels = {'emo_activation': 'Activation', 'emo_pleasure': 'Pleasure', 'emo_resolution': 'Resolution',
          'emo_competence': 'Competence', 'emo_valence': 'Valence'}

series = {k: [d.get(k, 0) for d in log] for k in emo_keys}
N = len(log)

FRAME_STEP = 4  # render every Nth simulated step to keep the gif a reasonable size
frames = []

for i in range(0, N, FRAME_STEP):
    fig = plt.figure(figsize=(10, 6), dpi=110)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.05], wspace=0.28)

    # --- left: island map + trajectory ---
    axm = fig.add_subplot(gs[0])
    axm.imshow(img, extent=[0, 256*SCALE, 256*SCALE, 0])
    axm.plot(xs[:i+1], ys[:i+1], color='white', lw=1.6, alpha=0.85)
    axm.plot(xs[:i+1], ys[:i+1], color='#ff5533', lw=0.9, alpha=0.9)
    axm.add_patch(Circle((xs[i], ys[i]), 28, color='#ffdd00', ec='#7a5a00', lw=1.5, zorder=5))
    axm.set_xlim(0, 256*SCALE)
    axm.set_ylim(256*SCALE, 0)
    axm.set_xticks([]); axm.set_yticks([])
    axm.set_title(f"Braitenberg agent on D\u00f6rner's Island \u2014 step {log[i]['step']}", fontsize=10.5, weight='bold')

    # --- right: emotion telemetry ---
    axe = fig.add_subplot(gs[1])
    t = list(range(i+1))
    for k in emo_keys:
        axe.plot(t, series[k][:i+1], color=colors[k], lw=1.8, label=labels[k])
    axe.axhline(0, color='#999999', lw=0.6)
    axe.set_xlim(0, N)
    axe.set_ylim(-2.2, 2.2)
    axe.set_xlabel('mind-cycle step', fontsize=9)
    axe.set_title('Doernerian emotional modulators (live)', fontsize=10.5, weight='bold')
    axe.legend(loc='upper right', fontsize=7.5, ncol=1, frameon=True)
    axe.grid(alpha=0.25)

    fig.suptitle("Real micropsi2 simulation \u2014 joschabach/micropsi2, Island world", fontsize=9, color='#666666', y=0.99)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    frame = imageio.imread(buf)
    frames.append(frame[:, :, :3])  # drop alpha channel for consistency

out_path = os.path.join(SCRIPT_DIR, 'output', 'micropsi2_island_simulation.gif')
imageio.mimsave(out_path, frames, duration=0.09, loop=0)
print('saved', out_path, 'frames:', len(frames))
