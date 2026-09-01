"""Animate all nine ensemble trajectories and emit both GIF and MP4 files."""

import io
import json
import os
import struct
import subprocess


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(OUTPUT_DIR, ".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import imageio.v2 as imageio
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


ABUNDANCE_ORDER = ("sparse", "medium", "plentiful")
DISTANCE_ORDER = ("close", "medium", "far")
MOTIVE_COLORS = {"energy": "#d64b3c", "water": "#2878b5", "integrity": "#8b52a1"}
RESOURCE_MARKERS = {
    "Waterhole": ("#2582d8", "o"),
    "Champignon": ("#2a9d65", "^"),
    "Wirselkraut": ("#1b8a82", "s"),
    "Juniper": ("#8b52a1", "D"),
    "FlyAgaric": ("#c83d35", "X"),
}
VIDEO_SECONDS = 15
FPS = 5
FRAME_COUNT = VIDEO_SECONDS * FPS


def validate_mp4(mp4_path, frame_count):
    """Require a complete FFmpeg/libx264 stream with the expected frame count."""
    with open(mp4_path, "rb") as handle:
        payload = handle.read()
    sample_table = payload.find(b"stsz")
    encoded_samples = (
        struct.unpack_from(">I", payload, sample_table + 12)[0] if sample_table >= 0 else 0
    )
    if b"avc1" not in payload or b"avcC" not in payload or encoded_samples != frame_count:
        raise RuntimeError(
            f"Invalid MP4: expected {frame_count} H.264 frames, found {encoded_samples}"
        )

    completed = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            mp4_path,
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"FFmpeg could not decode the generated MP4: {completed.stderr.strip()}")
    print(f"Validated {encoded_samples} FFmpeg/libx264 frames")


def load_runs():
    runs = []
    for config_id in range(1, 10):
        path = os.path.join(OUTPUT_DIR, f"ensemble_config_{config_id}.json")
        with open(path, encoding="utf-8") as handle:
            runs.append(json.load(handle))
    return {(run["abundance"], run["starting_distance"]): run for run in runs}


def render_frame(background, runs, requested_step):
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), dpi=100)
    for row, abundance in enumerate(ABUNDANCE_ORDER):
        for column, distance in enumerate(DISTANCE_ORDER):
            ax = axes[row, column]
            run = runs[(abundance, distance)]
            log = run["steps"]
            index = min(requested_step, len(log) - 1)
            current = log[index]
            ax.imshow(background, extent=[0, 2048, 2048, 0])
            for resource in run["resources"]:
                color, marker = RESOURCE_MARKERS[resource["type"]]
                ax.scatter(
                    [resource["position"][0]],
                    [resource["position"][1]],
                    s=30,
                    color=color,
                    marker=marker,
                    edgecolor="black",
                    linewidth=0.35,
                    zorder=4,
                )
            xs = [point["x"] for point in log[: index + 1]]
            ys = [point["y"] for point in log[: index + 1]]
            ax.plot(xs, ys, color="white", linewidth=2.0, alpha=0.85, zorder=5)
            ax.plot(xs, ys, color="#f05a3f", linewidth=1.05, alpha=0.95, zorder=6)
            ax.scatter(
                [run["start_position"][0]],
                [run["start_position"][1]],
                s=24,
                facecolor="white",
                edgecolor="black",
                linewidth=0.7,
                zorder=7,
            )
            ax.add_patch(
                Circle(
                    (current["x"], current["y"]),
                    27,
                    facecolor=MOTIVE_COLORS[current["ruling_motive"]],
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=8,
                )
            )
            ax.set_xlim(150, 1850)
            ax.set_ylim(1650, 200)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{distance.title()} start", fontsize=10, weight="bold")
            if column == 0:
                ax.set_ylabel(abundance.title(), fontsize=10, weight="bold")
            status = "alive" if requested_step < len(log) or run["survived"] else "dead"
            ax.text(
                0.02,
                0.02,
                f"C{run['config_id']}  step {current['step']}  {status}\n"
                f"P {current.get('emo_pleasure', 0):+.2f}  V {current.get('emo_valence', 0):+.2f}",
                transform=ax.transAxes,
                fontsize=6.8,
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
                zorder=9,
            )

    fig.suptitle(
        "Variation D — nine Survivor runs under varied environmental conditions",
        fontsize=13,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "Rows: safe-resource abundance (6 / 10 / 14). Columns: starting distance. "
        "Agent color shows ruling motive: red energy, blue water, purple integrity.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.055, wspace=0.08, hspace=0.13)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return imageio.imread(buffer)[:, :, :3]


def main():
    runs = load_runs()
    if len(runs) != 9:
        raise ValueError(f"Expected nine factorial cells, found {len(runs)}")
    background = mpimg.imread(
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
    max_steps = max(len(run["steps"]) for run in runs.values())
    # Sample the complete 800-step experiment uniformly into an exact 15-second
    # animation. At 75 frames / 5 fps, the simulation progresses roughly three
    # times more slowly than the previous 41-frame / 8-fps rendering.
    frame_steps = [
        round(index * (max_steps - 1) / (FRAME_COUNT - 1))
        for index in range(FRAME_COUNT)
    ]

    gif_path = os.path.join(OUTPUT_DIR, "ensemble_simulation.gif")
    mp4_path = os.path.join(OUTPUT_DIR, "ensemble_simulation.mp4")
    with imageio.get_writer(gif_path, mode="I", duration=round(1000 / FPS), loop=0) as gif_writer:
        with imageio.get_writer(
            mp4_path,
            format="FFMPEG",
            mode="I",
            fps=FPS,
            codec="libx264",
            pixelformat="yuv420p",
            quality=None,
            macro_block_size=1,
            output_params=["-crf", "20", "-preset", "medium", "-movflags", "+faststart"],
            ffmpeg_log_level="warning",
        ) as mp4_writer:
            for frame_number, step in enumerate(frame_steps, start=1):
                frame = render_frame(background, runs, step)
                gif_writer.append_data(frame)
                mp4_writer.append_data(frame)
                print(f"Rendered frame {frame_number}/{len(frame_steps)} (step {step})")
    validate_mp4(mp4_path, len(frame_steps))
    print(f"Saved {gif_path}")
    print(f"Saved {mp4_path}")


if __name__ == "__main__":
    main()
