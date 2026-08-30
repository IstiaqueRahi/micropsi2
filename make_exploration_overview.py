import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(SCRIPT_DIR, 'output'), exist_ok=True)

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.collections import LineCollection

with open(os.path.join(SCRIPT_DIR, 'output', 'survivor_exploration_log.json')) as f:
    log = json.load(f)

img = mpimg.imread(os.path.join(SCRIPT_DIR, 'micropsi_core', 'world', 'island', 'resources', 'groundmaps', 'psi_1.png'))
SCALE = 8
WORLD = 256 * SCALE          # 2048
GRID = 8
CELL = WORLD // GRID         # 256 -- same grid as run_survivor_exploration.py

xs = np.array([d['x'] for d in log])
ys = np.array([d['y'] for d in log])
steps = np.array([d['step'] for d in log])
N = len(log)

# certainty gets its own colour, distinct from the energy/water/integrity palette
CERT_COLOR = '#e67e22'
motive_colors = {'energy': '#c0392b', 'water': '#2980b9', 'integrity': '#8e44ad', 'certainty': CERT_COLOR}

# ---- reconstruct per-cell "last visited step" from the trajectory alone ----
last_visited = np.full((GRID, GRID), -1, dtype=float)   # -1 == never
for d in log:
    cx = min(GRID - 1, max(0, int(d['x']) // CELL))
    cy = min(GRID - 1, max(0, int(d['y']) // CELL))
    last_visited[cy, cx] = d['step']
visited_mask = last_visited >= 0

fig = plt.figure(figsize=(15, 15), dpi=140)
gs = fig.add_gridspec(4, 2, width_ratios=[1, 1.3], height_ratios=[1.25, 1, 1, 1], wspace=0.22, hspace=0.42)

# ---- top-left: trajectory on the island ----
axm = fig.add_subplot(gs[0, 0])
axm.imshow(img, extent=[0, WORLD, WORLD, 0])
points = np.array([xs, ys]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap='plasma', linewidth=1.1, alpha=0.85)
lc.set_array(steps)
axm.add_collection(lc)

obj_markers = {
    'Waterhole': ('#2980ff', 'o'), 'Champignon': ('#27ae60', '^'), 'Wirselkraut': ('#16a085', 's'),
    'Juniper': ('#8e44ad', 'D'), 'FlyAgaric': ('#c0392b', 'X'),
}
resource_specs = [
    ('Waterhole', (300, 500)), ('Waterhole', (900, 1400)), ('Champignon', (500, 350)),
    ('Champignon', (1150, 550)), ('Wirselkraut', (750, 650)), ('Juniper', (350, 900)),
    ('FlyAgaric', (620, 420)),
]
for objtype, pos in resource_specs:
    color, marker = obj_markers[objtype]
    axm.scatter([pos[0]], [pos[1]], s=90, color=color, marker=marker, edgecolor='black', linewidth=0.8, zorder=5)
axm.scatter([xs[0]], [ys[0]], s=70, color='white', edgecolor='black', zorder=6)
axm.set_xlim(0, WORLD); axm.set_ylim(WORLD, 0)
axm.set_xticks([]); axm.set_yticks([])
axm.set_title("Full trajectory (1400 steps) — Survivor + certainty demand", fontsize=11, weight='bold')
legend_handles = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor=c, markersize=9,
                  markeredgecolor='black', label=t) for t, (c, m) in obj_markers.items()]
axm.legend(handles=legend_handles, loc='lower right', fontsize=7, framealpha=0.9)

# ---- top-right: map-familiarity / coverage heatmap (the headline chart) ----
axc = fig.add_subplot(gs[0, 1])
axc.imshow(img, extent=[0, WORLD, WORLD, 0], alpha=0.30)
recency = np.where(visited_mask, last_visited / max(1, N), np.nan)   # 0..1, NaN = never
cmap = plt.cm.viridis.copy(); cmap.set_bad('#3b0a0a')               # never-visited = dark red
im = axc.imshow(recency, extent=[0, WORLD, WORLD, 0], origin='upper', cmap=cmap,
                vmin=0, vmax=1, alpha=0.72, interpolation='nearest')
for gx in range(GRID + 1):
    axc.axvline(gx * CELL, color='white', lw=0.5, alpha=0.4)
    axc.axhline(gx * CELL, color='white', lw=0.5, alpha=0.4)
for cy in range(GRID):
    for cx in range(GRID):
        lv = last_visited[cy, cx]
        axc.text(cx * CELL + CELL / 2, cy * CELL + CELL / 2,
                 '—' if lv < 0 else str(int(lv)), ha='center', va='center',
                 fontsize=7, color='white',
                 bbox=dict(boxstyle='round,pad=0.1', fc='black', ec='none', alpha=0.35))
n_vis = int(visited_mask.sum())
axc.set_xlim(0, WORLD); axc.set_ylim(WORLD, 0)
axc.set_xticks([]); axc.set_yticks([])
axc.set_title('Map familiarity — 8×8 cells, last step the agent was there\n'
              '(%d / 64 cells visited; baseline covers ~5)' % n_vis, fontsize=10.5, weight='bold')
cb = fig.colorbar(im, ax=axc, fraction=0.046, pad=0.02)
cb.set_label('last visit (fraction of run)', fontsize=8)

# ---- row 1: body demands + derived certainty urge ----
axd = fig.add_subplot(gs[1, :])
axd.plot(steps, [d['energy'] for d in log], color='#c0392b', lw=1.3, label='Energy (body)')
axd.plot(steps, [d['water'] for d in log], color='#2980b9', lw=1.3, label='Water (body)')
axd.plot(steps, [d['integrity'] for d in log], color='#8e44ad', lw=1.3, label='Integrity (body)')
axd.plot(steps, [d['certainty_urge'] for d in log], color=CERT_COLOR, lw=1.6, ls='--',
         label='Certainty URGE (derived, 0 = familiar)')
axd.axhline(0, color='#999', lw=0.5)
for d in log:
    if d['unexpected']:
        axd.axvline(d['step'], color='#e74c3c', lw=1, ls=':', alpha=0.7)
axd.set_ylim(-0.05, 1.05)
axd.set_xlim(0, N)
axd.set_title('Body demand levels (1 = full) vs. the derived certainty urge (dashed — a different kind of quantity: '
              'staleness of the map, not a body datasource)', fontsize=10.5, weight='bold')
axd.legend(loc='lower left', fontsize=8, ncol=4)
axd.set_xlabel('mind-cycle step', fontsize=9)
axd.grid(alpha=0.2)

# ---- row 2: ruling motive over time (4 colours now) ----
axr = fig.add_subplot(gs[2, :])
motives = [d['ruling_motive'] for d in log]
start = 0
for i in range(1, N + 1):
    if i == N or motives[i] != motives[start]:
        axr.axvspan(steps[start], steps[i - 1] + 1, color=motive_colors[motives[start]], alpha=0.55)
        start = i
axr.set_xlim(0, N); axr.set_yticks([])
switches = sum(1 for i in range(1, N) if motives[i] != motives[i - 1])
axr.set_title('Ruling motive over time — energy / water / integrity / certainty compete through the SAME '
              'hysteresis (%d switches; baseline ~55)' % switches, fontsize=10.5, weight='bold')
handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.55) for c in motive_colors.values()]
axr.legend(handles, motive_colors.keys(), loc='upper right', ncol=4, fontsize=8.5)
axr.set_xlabel('mind-cycle step', fontsize=9)

# ---- row 3: the real Doernerian emotional modulators ----
axe = fig.add_subplot(gs[3, :])
emo_keys = ['emo_activation', 'emo_pleasure', 'emo_resolution', 'emo_competence', 'emo_valence']
colors = {'emo_activation': '#c0392b', 'emo_pleasure': '#27ae60', 'emo_resolution': '#2980b9',
          'emo_competence': '#8e44ad', 'emo_valence': '#d68910'}
labels = {'emo_activation': 'Activation', 'emo_pleasure': 'Pleasure', 'emo_resolution': 'Resolution',
          'emo_competence': 'Competence', 'emo_valence': 'Valence'}
for k in emo_keys:
    axe.plot(steps, [d.get(k, 0) for d in log], color=colors[k], lw=1.3, label=labels[k])
axe.axhline(0, color='#999', lw=0.5)
axe.set_xlim(0, N)
axe.set_title('Doernerian emotional modulators — live, now driven by four competing demands', fontsize=11, weight='bold')
axe.legend(loc='upper right', fontsize=8, ncol=5)
axe.set_xlabel('mind-cycle step', fontsize=9)
axe.grid(alpha=0.2)

from collections import Counter
mc = Counter(motives)
fig.suptitle("micropsi2 Survivor + a synthetic CERTAINTY demand — the agent leaves its resource loop to "
             "re-familiarise stale regions\n"
             "certainty ruled %.0f%% of steps (target 10–25%%); %d/64 map cells visited vs the baseline's tight ~5"
             % (100.0 * mc['certainty'] / N, n_vis), fontsize=12, weight='bold', y=1.0)

plt.savefig(os.path.join(SCRIPT_DIR, 'output', 'survivor_exploration_overview.png'), dpi=140,
            bbox_inches='tight', facecolor='white')
print('saved')
