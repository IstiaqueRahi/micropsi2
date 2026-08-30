import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(SCRIPT_DIR, 'output'), exist_ok=True)

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle
import imageio.v2 as imageio
import io

with open(os.path.join(SCRIPT_DIR, 'output', 'survivor_depletion_log.json')) as f:
    log = json.load(f)

img = mpimg.imread(os.path.join(SCRIPT_DIR, 'micropsi_core', 'world', 'island', 'resources', 'groundmaps', 'psi_1.png'))
SCALE = 8
N = len(log)
xs = [d['x'] for d in log]
ys = [d['y'] for d in log]

obj_markers = {
    'Waterhole': ('#2980ff', 'o'), 'Champignon': ('#27ae60', '^'), 'Wirselkraut': ('#16a085', 's'),
    'Juniper': ('#8e44ad', 'D'), 'FlyAgaric': ('#c0392b', 'X'),
}
resource_specs = [
    ('Waterhole', (300, 500)), ('Waterhole', (1350, 650)), ('Champignon', (500, 350)),
    ('Champignon', (1150, 550)), ('Wirselkraut', (750, 650)), ('Juniper', (350, 900)),
    ('FlyAgaric', (620, 420)),
]
# stable per-type labels, same scheme as run_survivor.py, to look up each frame's logged supply
depletable = {'Waterhole', 'Champignon', 'Wirselkraut', 'Juniper'}
res_labels, _tc = [], {}
for objtype, _pos in resource_specs:
    if objtype in depletable:
        res_labels.append('%s%d' % (objtype, _tc.get(objtype, 0)))
        _tc[objtype] = _tc.get(objtype, 0) + 1
    else:
        res_labels.append(None)
supply_styles = {
    'Waterhole0': ('#2980ff', '-'), 'Waterhole1': ('#2980ff', '--'),
    'Champignon0': ('#27ae60', '-'), 'Champignon1': ('#27ae60', '--'),
    'Wirselkraut0': ('#16a085', '-'), 'Juniper0': ('#8e44ad', '-'),
}
tracked_labels = [l for l in supply_styles if l in log[0].get('supply', {})]
supply_ts = {l: [d['supply'][l] for d in log] for l in tracked_labels}

emo_keys = ['emo_activation', 'emo_pleasure', 'emo_resolution', 'emo_competence', 'emo_valence']
colors = {'emo_activation': '#c0392b', 'emo_pleasure': '#27ae60', 'emo_resolution': '#2980b9',
          'emo_competence': '#8e44ad', 'emo_valence': '#d68910'}
labels = {'emo_activation': 'Activation', 'emo_pleasure': 'Pleasure', 'emo_resolution': 'Resolution',
          'emo_competence': 'Competence', 'emo_valence': 'Valence'}
motive_colors = {'energy': '#c0392b', 'water': '#2980b9', 'integrity': '#8e44ad'}
series = {k: [d.get(k, 0) for d in log] for k in emo_keys}
demand_series = {k: [d[k] for d in log] for k in ['energy', 'water', 'integrity']}

FRAME_STEP = 10
TRAIL = 60  # how many past steps of trajectory to show, fading
frames = []

for i in range(0, N, FRAME_STEP):
    fig = plt.figure(figsize=(11, 7.6), dpi=110)
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.25], height_ratios=[1, 1, 1], wspace=0.27, hspace=0.4)

    axm = fig.add_subplot(gs[:, 0])
    axm.imshow(img, extent=[0, 256*SCALE, 256*SCALE, 0])
    for (objtype, pos), label in zip(resource_specs, res_labels):
        c, m = obj_markers[objtype]
        empty = label is not None and log[i]['supply'].get(label, 1.0) < 1.0   # depleted right now
        axm.scatter([pos[0]], [pos[1]], s=70, color=('none' if empty else c), marker=m,
                    edgecolor=('#888' if empty else 'black'), linewidth=0.7, zorder=4)
    lo = max(0, i-TRAIL)
    axm.plot(xs[lo:i+1], ys[lo:i+1], color='white', lw=2.0, alpha=0.9, zorder=5)
    axm.plot(xs[lo:i+1], ys[lo:i+1], color='#ff5533', lw=1.1, alpha=0.9, zorder=6)
    motive = log[i]['ruling_motive']
    axm.add_patch(Circle((xs[i], ys[i]), 26, color=motive_colors[motive], ec='black', lw=1.3, zorder=7))
    axm.set_xlim(0, 256*SCALE); axm.set_ylim(256*SCALE, 0)
    axm.set_xticks([]); axm.set_yticks([])
    axm.set_title(f"step {log[i]['step']}  \u2014  ruling motive: {motive}", fontsize=10.5, weight='bold')

    axd = fig.add_subplot(gs[0, 1])
    t = list(range(i+1))
    axd.plot(t, demand_series['energy'][:i+1], color='#c0392b', lw=1.3, label='Energy')
    axd.plot(t, demand_series['water'][:i+1], color='#2980b9', lw=1.3, label='Water')
    axd.plot(t, demand_series['integrity'][:i+1], color='#8e44ad', lw=1.3, label='Integrity')
    axd.set_xlim(0, N); axd.set_ylim(-0.05, 1.05)
    axd.set_title('Body demands', fontsize=10, weight='bold')
    axd.legend(loc='lower left', fontsize=7, ncol=3)
    axd.grid(alpha=0.2)

    axe = fig.add_subplot(gs[1, 1])
    for k in emo_keys:
        axe.plot(t, series[k][:i+1], color=colors[k], lw=1.2, label=labels[k])
    axe.axhline(0, color='#999', lw=0.5)
    axe.set_xlim(0, N); axe.set_ylim(-1.6, 1.6)
    axe.set_title('Doernerian modulators (live)', fontsize=10, weight='bold')
    axe.legend(loc='upper right', fontsize=6.5, ncol=3)
    axe.grid(alpha=0.2)

    axs = fig.add_subplot(gs[2, 1])
    for label in tracked_labels:
        col, ls = supply_styles[label]
        axs.plot(t, supply_ts[label][:i+1], color=col, ls=ls, lw=1.2, label=label)
    axs.axhline(1.0, color='#c0392b', lw=0.8, ls=':', alpha=0.7)
    axs.set_xlim(0, N); axs.set_ylim(-0.3, 7.6)
    axs.set_title('Resource supply (1 serving per eat/drink; <1.0 = depleted)', fontsize=10, weight='bold')
    axs.legend(loc='upper right', fontsize=5.8, ncol=3)
    axs.set_xlabel('mind-cycle step', fontsize=8.5)
    axs.grid(alpha=0.2)

    fig.suptitle("Real micropsi2 Survivor agent \u2014 competing demands on D\u00f6rner's Island", fontsize=10.5,
                 weight='bold', y=1.0)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    frames.append(imageio.imread(buf)[:, :, :3])

out_path = os.path.join(SCRIPT_DIR, 'output', 'survivor_depletion.gif')
imageio.mimsave(out_path, frames, duration=0.08, loop=0)
print('saved', out_path, 'frames:', len(frames))
