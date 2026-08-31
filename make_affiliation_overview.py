import json
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "survivor_affiliation_log.json")) as handle:
    log = json.load(handle)

groundmap = mpimg.imread(
    os.path.join(
        SCRIPT_DIR,
        "micropsi_core",
        "world",
        "island",
        "resources",
        "groundmaps",
        "psi_1.png",
    )
)

SCALE = 8
N = len(log)
steps = np.array([entry["step"] for entry in log])
agent_keys = ("agent_a", "agent_b")
agent_names = {"agent_a": "Agent A", "agent_b": "Agent B"}
agent_colors = {"agent_a": "#ff5b3a", "agent_b": "#19a7ce"}
agent_markers = {"agent_a": "o", "agent_b": "s"}

motive_colors = {
    "energy": "#c0392b",
    "water": "#2980b9",
    "integrity": "#8e44ad",
    "affiliation": "#f39c12",
}
emo_keys = [
    "emo_activation",
    "emo_pleasure",
    "emo_resolution",
    "emo_competence",
    "emo_valence",
]
emo_colors = {
    "emo_activation": "#c0392b",
    "emo_pleasure": "#27ae60",
    "emo_resolution": "#2980b9",
    "emo_competence": "#8e44ad",
    "emo_valence": "#d68910",
}
emo_labels = {
    "emo_activation": "Activation",
    "emo_pleasure": "Pleasure",
    "emo_resolution": "Resolution",
    "emo_competence": "Competence",
    "emo_valence": "Valence",
}

resource_markers = {
    "Waterhole": ("#2980ff", "o"),
    "Champignon": ("#27ae60", "^"),
    "Wirselkraut": ("#16a085", "s"),
    "Juniper": ("#8e44ad", "D"),
    "FlyAgaric": ("#c0392b", "X"),
}
resource_specs = [
    ("Waterhole", (300, 500)),
    ("Waterhole", (900, 1400)),
    ("Champignon", (500, 350)),
    ("Champignon", (1150, 550)),
    ("Wirselkraut", (750, 650)),
    ("Juniper", (350, 900)),
    ("FlyAgaric", (620, 420)),
]


def agent_series(agent_key, field):
    return np.array([entry[agent_key].get(field, 0) for entry in log])


def draw_motive_band(axis, agent_key, y_min, y_max):
    motives = [entry[agent_key]["ruling_motive"] for entry in log]
    start = 0
    for index in range(1, N + 1):
        if index == N or motives[index] != motives[start]:
            axis.axvspan(
                steps[start],
                steps[index - 1] + 1,
                ymin=y_min,
                ymax=y_max,
                color=motive_colors[motives[start]],
                alpha=0.72,
                linewidth=0,
            )
            start = index


fig = plt.figure(figsize=(16, 14), dpi=140)
grid = fig.add_gridspec(
    4,
    2,
    height_ratios=[1.35, 1.0, 0.55, 1.0],
    width_ratios=[1.0, 1.35],
    hspace=0.38,
    wspace=0.24,
)

# Trajectories on the real island terrain.
map_axis = fig.add_subplot(grid[0:2, 0])
map_axis.imshow(groundmap, extent=[0, 256 * SCALE, 256 * SCALE, 0])
for object_type, position in resource_specs:
    color, marker = resource_markers[object_type]
    map_axis.scatter(
        [position[0]],
        [position[1]],
        s=82,
        color=color,
        marker=marker,
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
    )

for agent_key in agent_keys:
    xs = agent_series(agent_key, "x")
    ys = agent_series(agent_key, "y")
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    collection = LineCollection(
        segments,
        colors=agent_colors[agent_key],
        linewidth=1.25,
        alpha=0.72,
        label=agent_names[agent_key],
    )
    map_axis.add_collection(collection)
    map_axis.scatter(
        xs[0],
        ys[0],
        s=100,
        marker=agent_markers[agent_key],
        facecolor="white",
        edgecolor=agent_colors[agent_key],
        linewidth=2.0,
        zorder=7,
    )
    map_axis.scatter(
        xs[-1],
        ys[-1],
        s=150,
        marker="*",
        color=agent_colors[agent_key],
        edgecolor="black",
        linewidth=0.8,
        zorder=8,
    )

map_axis.set_xlim(0, 256 * SCALE)
map_axis.set_ylim(256 * SCALE, 0)
map_axis.set_xticks([])
map_axis.set_yticks([])
map_axis.set_title(
    "Two independent trajectories\noutlined markers = starts, stars = final positions",
    fontsize=11,
    weight="bold",
)
map_axis.legend(
    handles=[
        plt.Line2D([0], [0], color=agent_colors[key], lw=2, label=agent_names[key])
        for key in agent_keys
    ],
    loc="lower right",
    fontsize=8,
    framealpha=0.9,
)

# Inter-agent distance, the variation's central observable.
distance_axis = fig.add_subplot(grid[0, 1])
distances = np.array([entry["distance"] for entry in log])
distance_axis.plot(steps, distances, color="#34495e", lw=1.35)
distance_axis.axhline(
    180,
    color=motive_colors["affiliation"],
    ls="--",
    lw=1,
    label="close enough (180)",
)
for agent_key, alpha in (("agent_a", 0.10), ("agent_b", 0.07)):
    affiliation_steps = agent_series(agent_key, "ruling_motive") == "affiliation"
    distance_axis.fill_between(
        steps,
        0,
        distances,
        where=affiliation_steps,
        color=agent_colors[agent_key],
        alpha=alpha,
        step="mid",
    )
distance_axis.set_xlim(0, N)
distance_axis.set_ylim(0, max(1500, distances.max() * 1.05))
distance_axis.set_title(
    "Inter-agent distance — separation and affiliation-driven reunion",
    fontsize=11,
    weight="bold",
)
distance_axis.set_ylabel("map units")
distance_axis.set_xlabel("mind-cycle step")
distance_axis.legend(loc="upper right", fontsize=8)
distance_axis.grid(alpha=0.2)

# Body demand state for both agents.
demand_axes = [fig.add_subplot(grid[1, 1])]
demand_axis = demand_axes[0]
for agent_key, line_style in (("agent_a", "-"), ("agent_b", "--")):
    for field, color in (
        ("energy", "#c0392b"),
        ("water", "#2980b9"),
        ("integrity", "#8e44ad"),
    ):
        demand_axis.plot(
            steps,
            agent_series(agent_key, field),
            color=color,
            ls=line_style,
            lw=1.0,
            alpha=0.9,
            label=f"{agent_names[agent_key]} {field}",
        )
demand_axis.set_xlim(0, N)
demand_axis.set_ylim(-0.05, 1.05)
demand_axis.set_title("Body state (solid A, dashed B)", fontsize=11, weight="bold")
demand_axis.set_xlabel("mind-cycle step")
demand_axis.legend(loc="lower right", fontsize=7, ncol=3)
demand_axis.grid(alpha=0.2)

# Two stacked ruling-motive bands.
motive_axis = fig.add_subplot(grid[2, :])
draw_motive_band(motive_axis, "agent_a", 0.53, 0.98)
draw_motive_band(motive_axis, "agent_b", 0.02, 0.47)
motive_axis.set_xlim(0, N)
motive_axis.set_ylim(0, 1)
motive_axis.set_yticks([0.25, 0.75], ["Agent B", "Agent A"])
motive_axis.set_xlabel("mind-cycle step")
motive_axis.set_title(
    "Ruling motives — shared 0.12 hysteresis, affiliation in orange",
    fontsize=11,
    weight="bold",
)
motive_axis.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.72)
        for color in motive_colors.values()
    ],
    labels=list(motive_colors),
    loc="upper right",
    ncol=4,
    fontsize=8,
)

# Separate panels keep all five real modulators legible for both nodenets.
for column, agent_key in enumerate(agent_keys):
    modulator_axis = fig.add_subplot(grid[3, column])
    for modulator in emo_keys:
        modulator_axis.plot(
            steps,
            agent_series(agent_key, modulator),
            color=emo_colors[modulator],
            lw=1.15,
            label=emo_labels[modulator],
        )
    modulator_axis.axhline(0, color="#999", lw=0.5)
    modulator_axis.set_xlim(0, N)
    modulator_axis.set_title(
        f"{agent_names[agent_key]} — Doernerian emotional modulators",
        fontsize=10.5,
        weight="bold",
    )
    modulator_axis.set_xlabel("mind-cycle step")
    modulator_axis.grid(alpha=0.2)
    modulator_axis.legend(loc="upper right", fontsize=7, ncol=2)

fig.suptitle(
    "micropsi2 Survivor affiliation variation — two bodies, two nodenets, one social demand",
    fontsize=13,
    weight="bold",
    y=0.995,
)

output_path = os.path.join(OUTPUT_DIR, "affiliation_simulation_overview.png")
plt.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
print("saved", output_path)
