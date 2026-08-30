import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(SCRIPT_DIR, 'output'), exist_ok=True)

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

with open(os.path.join(SCRIPT_DIR, 'output', 'survivor_log.json')) as f:
    log = json.load(f)

img = mpimg.imread(os.path.join(SCRIPT_DIR, 'micropsi_core', 'world', 'island', 'resources', 'groundmaps', 'psi_1.png'))
SCALE = 8

xs = np.array([d['x'] for d in log])
ys = np.array([d['y'] for d in log])
steps = np.array([d['step'] for d in log])
N = len(log)

fig = plt.figure(figsize=(15, 11), dpi=140)
gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.3], height_ratios=[1.3, 1, 1], wspace=0.22, hspace=0.38)

# ---- top-left: trajectory on the island ----
axm = fig.add_subplot(gs[0, 0])
axm.imshow(img, extent=[0, 256*SCALE, 256*SCALE, 0])
points = np.array([xs, ys]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap='plasma', linewidth=1.1, alpha=0.85)
lc.set_array(steps)
axm.add_collection(lc)

# mark resource objects
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
    axm.scatter([pos[0]], [pos[1]], s=90, color=color, marker=marker, edgecolor='black',
                linewidth=0.8, zorder=5)
axm.scatter([xs[0]], [ys[0]], s=70, color='white', edgecolor='black', zorder=6)
axm.set_xlim(0, 256*SCALE); axm.set_ylim(256*SCALE, 0)
axm.set_xticks([]); axm.set_yticks([])
axm.set_title("Full trajectory (1400 steps) — real Survivor agent, real terrain", fontsize=11, weight='bold')

legend_handles = [plt.Line2D([0],[0], marker=m, color='w', markerfacecolor=c, markersize=9,
                  markeredgecolor='black', label=t) for t,(c,m) in obj_markers.items()]
axm.legend(handles=legend_handles, loc='lower right', fontsize=7, framealpha=0.9)

# ---- top-right: demand levels over time ----
axd = fig.add_subplot(gs[0, 1])
axd.plot(steps, [d['energy'] for d in log], color='#c0392b', lw=1.3, label='Energy')
axd.plot(steps, [d['water'] for d in log], color='#2980b9', lw=1.3, label='Water')
axd.plot(steps, [d['integrity'] for d in log], color='#8e44ad', lw=1.3, label='Integrity')
axd.axhline(0, color='#999', lw=0.5)
for i, d in enumerate(log):
    if d['unexpected']:
        axd.axvline(d['step'], color='#e74c3c', lw=1, ls=':', alpha=0.7)
        axd.annotate('FlyAgaric trap', (d['step'], 0.15), fontsize=7.5, color='#c0392b',
                     ha='left', rotation=90, va='bottom')
axd.set_ylim(-0.05, 1.05)
axd.set_title('Body demands (Survivor datasources)', fontsize=11, weight='bold')
axd.legend(loc='lower left', fontsize=8, ncol=3)
axd.set_xlabel('mind-cycle step', fontsize=9)
axd.grid(alpha=0.2)

# ---- middle: ruling motive over time (as colored bands) ----
axr = fig.add_subplot(gs[1, :])
motive_colors = {'energy': '#c0392b', 'water': '#2980b9', 'integrity': '#8e44ad'}
motives = [d['ruling_motive'] for d in log]
start = 0
for i in range(1, N+1):
    if i == N or motives[i] != motives[start]:
        axr.axvspan(steps[start], steps[i-1]+1, color=motive_colors[motives[start]], alpha=0.5)
        start = i
axr.set_xlim(0, N)
axr.set_yticks([])
axr.set_title('Ruling motive over time (which demand is dominant — note the hysteresis: no rapid flip-flopping)',
              fontsize=11, weight='bold')
handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.5) for c in motive_colors.values()]
axr.legend(handles, motive_colors.keys(), loc='upper right', ncol=3, fontsize=8.5)
axr.set_xlabel('mind-cycle step', fontsize=9)

# ---- bottom: the real Doernerian emotional modulators ----
axe = fig.add_subplot(gs[2, :])
emo_keys = ['emo_activation', 'emo_pleasure', 'emo_resolution', 'emo_competence', 'emo_valence']
colors = {'emo_activation': '#c0392b', 'emo_pleasure': '#27ae60', 'emo_resolution': '#2980b9',
          'emo_competence': '#8e44ad', 'emo_valence': '#d68910'}
labels = {'emo_activation': 'Activation', 'emo_pleasure': 'Pleasure', 'emo_resolution': 'Resolution',
          'emo_competence': 'Competence', 'emo_valence': 'Valence'}
for k in emo_keys:
    series = [d.get(k, 0) for d in log]
    axe.plot(steps, series, color=colors[k], lw=1.3, label=labels[k])
axe.axhline(0, color='#999', lw=0.5)
axe.set_xlim(0, N)
axe.set_title('Doernerian emotional modulators — live, driven by real competing demands', fontsize=11, weight='bold')
axe.legend(loc='upper right', fontsize=8, ncol=5)
axe.set_xlabel('mind-cycle step', fontsize=9)
axe.grid(alpha=0.2)

fig.suptitle("Real micropsi2 Survivor agent — competing demands, real terrain, real Doernerian modulators\n"
             "(3 bugs found & fixed in the process: N/S movement inversion, action_drink dispatch, and target-lock oscillation)",
             fontsize=12, weight='bold', y=1.0)

plt.savefig(os.path.join(SCRIPT_DIR, 'output', 'survivor_simulation_overview.png'), dpi=140, bbox_inches='tight', facecolor='white')
print('saved')
