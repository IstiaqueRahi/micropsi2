import io
import json
import os

import imageio.v2 as imageio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


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

positions = {
    agent_key: {
        "x": [entry[agent_key]["x"] for entry in log],
        "y": [entry[agent_key]["y"] for entry in log],
    }
    for agent_key in agent_keys
}
distances = [entry["distance"] for entry in log]
modulator_series = {
    agent_key: {
        key: [entry[agent_key].get(key, 0) for entry in log] for key in emo_keys
    }
    for agent_key in agent_keys
}


def draw_motive_history(axis, agent_key, frame_index, y_min, y_max):
    motives = [entry[agent_key]["ruling_motive"] for entry in log[: frame_index + 1]]
    start = 0
    for index in range(1, len(motives) + 1):
        if index == len(motives) or motives[index] != motives[start]:
            axis.axvspan(
                start,
                index,
                ymin=y_min,
                ymax=y_max,
                color=motive_colors[motives[start]],
                alpha=0.78,
                linewidth=0,
            )
            start = index


FRAME_STEP = 10
TRAIL_LENGTH = 100
frames = []

for frame_index in range(0, N, FRAME_STEP):
    fig = plt.figure(figsize=(12.5, 7.8), dpi=105)
    grid = fig.add_gridspec(
        4,
        2,
        width_ratios=[1.0, 1.35],
        height_ratios=[1.0, 0.5, 0.9, 0.9],
        wspace=0.27,
        hspace=0.44,
    )

    map_axis = fig.add_subplot(grid[:, 0])
    map_axis.imshow(groundmap, extent=[0, 256 * SCALE, 256 * SCALE, 0])
    for object_type, position in resource_specs:
        color, marker = resource_markers[object_type]
        map_axis.scatter(
            position[0],
            position[1],
            s=62,
            color=color,
            marker=marker,
            edgecolor="black",
            linewidth=0.6,
            zorder=4,
        )

    trail_start = max(0, frame_index - TRAIL_LENGTH)
    for agent_key in agent_keys:
        xs = positions[agent_key]["x"]
        ys = positions[agent_key]["y"]
        map_axis.plot(
            xs[trail_start : frame_index + 1],
            ys[trail_start : frame_index + 1],
            color="white",
            lw=2.8,
            alpha=0.75,
            zorder=5,
        )
        map_axis.plot(
            xs[trail_start : frame_index + 1],
            ys[trail_start : frame_index + 1],
            color=agent_colors[agent_key],
            lw=1.5,
            alpha=0.95,
            zorder=6,
        )
        map_axis.scatter(
            xs[0],
            ys[0],
            s=75,
            marker=agent_markers[agent_key],
            facecolor="white",
            edgecolor=agent_colors[agent_key],
            linewidth=1.8,
            zorder=7,
        )
        current_motive = log[frame_index][agent_key]["ruling_motive"]
        map_axis.add_patch(
            Circle(
                (xs[frame_index], ys[frame_index]),
                28,
                facecolor=agent_colors[agent_key],
                edgecolor=motive_colors[current_motive],
                linewidth=3.0,
                zorder=8,
            )
        )

    motives_now = {
        key: log[frame_index][key]["ruling_motive"] for key in agent_keys
    }
    if "affiliation" in motives_now.values():
        map_axis.plot(
            [positions[key]["x"][frame_index] for key in agent_keys],
            [positions[key]["y"][frame_index] for key in agent_keys],
            color=motive_colors["affiliation"],
            ls="--",
            lw=1.3,
            alpha=0.85,
            zorder=7,
        )
    map_axis.set_xlim(0, 256 * SCALE)
    map_axis.set_ylim(256 * SCALE, 0)
    map_axis.set_xticks([])
    map_axis.set_yticks([])
    map_axis.set_title(
        f"step {log[frame_index]['step']}  |  distance {distances[frame_index]:.0f}\n"
        f"A: {motives_now['agent_a']}   B: {motives_now['agent_b']}",
        fontsize=10.5,
        weight="bold",
    )

    distance_axis = fig.add_subplot(grid[0, 1])
    timeline = list(range(frame_index + 1))
    distance_axis.plot(timeline, distances[: frame_index + 1], color="#34495e", lw=1.4)
    distance_axis.axhline(180, color="#f39c12", ls="--", lw=0.9)
    distance_axis.set_xlim(0, N)
    distance_axis.set_ylim(0, 1500)
    distance_axis.set_title("Inter-agent distance", fontsize=10, weight="bold")
    distance_axis.set_ylabel("map units", fontsize=8)
    distance_axis.grid(alpha=0.2)

    motive_axis = fig.add_subplot(grid[1, 1])
    draw_motive_history(motive_axis, "agent_a", frame_index, 0.53, 0.98)
    draw_motive_history(motive_axis, "agent_b", frame_index, 0.02, 0.47)
    motive_axis.set_xlim(0, N)
    motive_axis.set_ylim(0, 1)
    motive_axis.set_yticks([0.25, 0.75], ["B", "A"])
    motive_axis.set_title("Ruling motives (affiliation = orange)", fontsize=9.5, weight="bold")

    for row, agent_key in ((2, "agent_a"), (3, "agent_b")):
        modulator_axis = fig.add_subplot(grid[row, 1])
        for modulator in emo_keys:
            modulator_axis.plot(
                timeline,
                modulator_series[agent_key][modulator][: frame_index + 1],
                color=emo_colors[modulator],
                lw=1.0,
                label=emo_labels[modulator],
            )
        modulator_axis.axhline(0, color="#999", lw=0.5)
        modulator_axis.set_xlim(0, N)
        modulator_axis.set_ylim(-1.6, 1.6)
        modulator_axis.set_title(
            f"{agent_names[agent_key]} modulators", fontsize=9.5, weight="bold"
        )
        modulator_axis.grid(alpha=0.2)
        if row == 2:
            modulator_axis.legend(loc="upper right", fontsize=6.2, ncol=3)
        else:
            modulator_axis.set_xlabel("mind-cycle step", fontsize=8)

    fig.suptitle(
        "micropsi2 Survivor affiliation — independent survival and social reunion",
        fontsize=11.5,
        weight="bold",
        y=0.995,
    )
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    frames.append(imageio.imread(buffer)[:, :, :3])

output_path = os.path.join(OUTPUT_DIR, "affiliation_simulation.gif")
imageio.mimsave(output_path, frames, duration=0.08, loop=0)
print("saved", output_path, "frames:", len(frames))
