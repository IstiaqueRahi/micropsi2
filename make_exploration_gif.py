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

with open(os.path.join(SCRIPT_DIR, 'output', 'survivor_exploration_log.json')) as f:
    log = json.load(f)

img = mpimg.imread(os.path.join(SCRIPT_DIR, 'micropsi_core', 'world', 'island', 'resources', 'groundmaps', 'psi_1.png'))
SCALE = 8
WORLD = 256 * SCALE
GRID = 8
CELL = WORLD // GRID           # same grid as run_survivor_exploration.py
N = len(log)
xs = [d['x'] for d in log]
ys = [d['y'] for d in log]

def cell_of(x, y):
    return (min(GRID - 1, max(0, int(x) // CELL)), min(GRID - 1, max(0, int(y) // CELL)))

CERT_COLOR = '#e67e22'
obj_markers = {
    'Waterhole': ('#2980ff', 'o'), 'Champignon': ('#27ae60', '^'), 'Wirselkraut': ('#16a085', 's'),
    'Juniper': ('#8e44ad', 'D'), 'FlyAgaric': ('#c0392b', 'X'),
}
resource_specs = [
    ('Waterhole', (300, 500)), ('Waterhole', (900, 1400)), ('Champignon', (500, 350)),
    ('Champignon', (1150, 550)), ('Wirselkraut', (750, 650)), ('Juniper', (350, 900)),
    ('FlyAgaric', (620, 420)),
]
motive_colors = {'energy': '#c0392b', 'water': '#2980b9', 'integrity': '#8e44ad', 'certainty': CERT_COLOR}

emo_keys = ['emo_activation', 'emo_pleasure', 'emo_resolution', 'emo_competence', 'emo_valence']
emo_colors = {'emo_activation': '#c0392b', 'emo_pleasure': '#27ae60', 'emo_resolution': '#2980b9',
              'emo_competence': '#8e44ad', 'emo_valence': '#d68910'}
emo_labels = {'emo_activation': 'Activation', 'emo_pleasure': 'Pleasure', 'emo_resolution': 'Resolution',
              'emo_competence': 'Competence', 'emo_valence': 'Valence'}
series = {k: [d.get(k, 0) for d in log] for k in emo_keys}
demand_series = {k: [d[k] for d in log] for k in ['energy', 'water', 'integrity']}
cert_series = [d['certainty_urge'] for d in log]

STALE_CAP = 750.0             # matches STALENESS_CEIL in the driver (for the colour ramp only)
FRAME_STEP = 10
TRAIL = 60
frames = []

for i in range(0, N, FRAME_STEP):
    # coverage as of this frame, straight from the trajectory
    last_visited = np.full((GRID, GRID), -1e9)
    for d in log[:i + 1]:
        cx, cy = cell_of(d['x'], d['y'])
        last_visited[cy, cx] = d['step']
    staleness = np.clip(i - last_visited, 0, STALE_CAP) / STALE_CAP   # 0 = fresh, 1 = long unseen / never

    fig = plt.figure(figsize=(11, 7.6), dpi=110)
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.25], height_ratios=[1, 1, 1], wspace=0.27, hspace=0.4)

    # ---- map with per-frame familiarity overlay ----
    axm = fig.add_subplot(gs[:, 0])
    axm.imshow(img, extent=[0, WORLD, WORLD, 0])
    # green where freshly seen, fading to red where the agent hasn't been in a while
    overlay = np.zeros((GRID, GRID, 4))
    overlay[..., 0] = staleness            # R rises with staleness
    overlay[..., 1] = 1.0 - staleness      # G falls with staleness
    overlay[..., 3] = 0.32                 # constant translucency
    axm.imshow(overlay, extent=[0, WORLD, WORLD, 0], origin='upper', interpolation='nearest', zorder=2)
    for g in range(GRID + 1):
        axm.axvline(g * CELL, color='white', lw=0.4, alpha=0.3, zorder=2)
        axm.axhline(g * CELL, color='white', lw=0.4, alpha=0.3, zorder=2)
    for objtype, pos in resource_specs:
        c, m = obj_markers[objtype]
        axm.scatter([pos[0]], [pos[1]], s=70, color=c, marker=m, edgecolor='black', linewidth=0.7, zorder=4)
    lo = max(0, i - TRAIL)
    axm.plot(xs[lo:i + 1], ys[lo:i + 1], color='white', lw=2.0, alpha=0.9, zorder=5)
    axm.plot(xs[lo:i + 1], ys[lo:i + 1], color='#ff5533', lw=1.1, alpha=0.9, zorder=6)
    motive = log[i]['ruling_motive']
    axm.add_patch(Circle((xs[i], ys[i]), 26, color=motive_colors[motive], ec='black', lw=1.3, zorder=7))
    et = log[i].get('explore_target_xy')
    if motive == 'certainty' and et:
        axm.add_patch(Circle((et[0], et[1]), 40, fill=False, ec=CERT_COLOR, lw=2.2, ls='--', zorder=7))
    axm.set_xlim(0, WORLD); axm.set_ylim(WORLD, 0)
    axm.set_xticks([]); axm.set_yticks([])
    n_seen = int((last_visited > -1e8).sum())
    axm.set_title(f"step {log[i]['step']}  —  ruling: {motive}   |   cells seen: {n_seen}/64\n"
                  f"green = freshly familiar, red = gone stale", fontsize=10, weight='bold')

    # ---- body demands + certainty urge ----
    axd = fig.add_subplot(gs[0, 1])
    t = list(range(i + 1))
    axd.plot(t, demand_series['energy'][:i + 1], color='#c0392b', lw=1.3, label='Energy')
    axd.plot(t, demand_series['water'][:i + 1], color='#2980b9', lw=1.3, label='Water')
    axd.plot(t, demand_series['integrity'][:i + 1], color='#8e44ad', lw=1.3, label='Integrity')
    axd.plot(t, cert_series[:i + 1], color=CERT_COLOR, lw=1.5, ls='--', label='Certainty urge (derived)')
    axd.set_xlim(0, N); axd.set_ylim(-0.05, 1.05)
    axd.set_title('Body demand levels + derived certainty urge', fontsize=10, weight='bold')
    axd.legend(loc='lower left', fontsize=6.5, ncol=2)
    axd.grid(alpha=0.2)

    # ---- ruling motive so far ----
    axr = fig.add_subplot(gs[1, 1])
    mv = [d['ruling_motive'] for d in log[:i + 1]]
    s = 0
    for j in range(1, len(mv) + 1):
        if j == len(mv) or mv[j] != mv[s]:
            axr.axvspan(s, j, color=motive_colors[mv[s]], alpha=0.6)
            s = j
    axr.set_xlim(0, N); axr.set_yticks([])
    axr.set_title('Ruling motive (energy / water / integrity / certainty)', fontsize=10, weight='bold')
    axr.set_xlabel('mind-cycle step', fontsize=8.5)

    # ---- modulators ----
    axe = fig.add_subplot(gs[2, 1])
    for k in emo_keys:
        axe.plot(t, series[k][:i + 1], color=emo_colors[k], lw=1.1, label=emo_labels[k])
    axe.axhline(0, color='#999', lw=0.5)
    axe.set_xlim(0, N); axe.set_ylim(-1.6, 1.6)
    axe.set_title('Doernerian modulators (live)', fontsize=10, weight='bold')
    axe.legend(loc='upper right', fontsize=6, ncol=3)
    axe.set_xlabel('mind-cycle step', fontsize=8.5)
    axe.grid(alpha=0.2)

    fig.suptitle("micropsi2 Survivor + synthetic CERTAINTY demand — periodic exploration of stale map regions",
                 fontsize=10.5, weight='bold', y=1.0)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    frames.append(imageio.imread(buf)[:, :, :3])

out_path = os.path.join(SCRIPT_DIR, 'output', 'survivor_exploration.gif')
imageio.mimsave(out_path, frames, duration=0.08, loop=0)
print('saved', out_path, 'frames:', len(frames))
