"""Create a 30-second MP4 of the nine representative experiment runs."""

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
from matplotlib.patches import Circle, Rectangle


ABUNDANCE = ("sparse", "medium", "plentiful")
DISTANCE = ("close", "medium", "far")
MOTIVE_COLORS = {"energy": "#d64b3c", "water": "#2878b5", "integrity": "#8b52a1"}
RESOURCE_MARKERS = {
    "Waterhole": ("#2582d8", "o"),
    "Champignon": ("#2a9d65", "^"),
    "Wirselkraut": ("#1b8a82", "s"),
    "Juniper": ("#8b52a1", "D"),
    "FlyAgaric": ("#c83d35", "X"),
}
VIDEO_SECONDS = 30
FPS = 5
FRAME_COUNT = VIDEO_SECONDS * FPS


def validate_video(path):
    with open(path, "rb") as handle:
        payload = handle.read()
    table = payload.find(b"stsz")
    samples = struct.unpack_from(">I", payload, table + 12)[0] if table >= 0 else 0
    if samples != FRAME_COUNT or b"avcC" not in payload:
        raise RuntimeError(f"Expected {FRAME_COUNT} H.264 samples, found {samples}")
    completed = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", path, "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True, text=True, timeout=90,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Video decode validation failed: {completed.stderr.strip()}")
    print(f"Validated {samples} H.264 frames ({VIDEO_SECONDS} seconds at {FPS} fps)")


def main():
    with open(os.path.join(OUTPUT_DIR, "survivor_experiment_results.json"), encoding="utf-8") as handle:
        document = json.load(handle)
    traces = {
        (trace["abundance"], trace["starting_distance"]): trace
        for trace in document["representative_traces"]
    }
    if len(traces) != 9:
        raise ValueError(f"Expected nine representative traces, found {len(traces)}")
    background = mpimg.imread(
        os.path.join(SCRIPT_DIR, "micropsi_core", "world", "island", "resources", "groundmaps", "psi_1.png")
    )
    max_steps = max(len(trace["steps"]) for trace in traces.values())
    frame_steps = [round(index * (max_steps - 1) / (FRAME_COUNT - 1)) for index in range(FRAME_COUNT)]

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), dpi=100)
    artists = {}
    for row, abundance in enumerate(ABUNDANCE):
        for column, distance in enumerate(DISTANCE):
            key = (abundance, distance)
            trace = traces[key]
            ax = axes[row, column]
            ax.imshow(background, extent=[0, 2048, 2048, 0])
            for resource in trace["resources"]:
                color, marker = RESOURCE_MARKERS[resource["type"]]
                ax.scatter(
                    [resource["position"][0]], [resource["position"][1]], s=28,
                    color=color, marker=marker, edgecolor="black", linewidth=0.35, zorder=4,
                )
            start = trace["start_position"]
            ax.scatter([start[0]], [start[1]], s=24, facecolor="white", edgecolor="black", linewidth=0.7, zorder=7)
            white_line, = ax.plot([], [], color="white", linewidth=2.2, alpha=0.85, zorder=5)
            trail_line, = ax.plot([], [], color="#f05a3f", linewidth=1.05, alpha=0.95, zorder=6)
            agent = Circle((start[0], start[1]), 27, facecolor=MOTIVE_COLORS["energy"], edgecolor="black", linewidth=0.8, zorder=8)
            ax.add_patch(agent)
            status = ax.text(
                0.02, 0.02, "", transform=ax.transAxes, fontsize=6.6, color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.4}, zorder=9,
            )
            ax.set_xlim(150, 1850); ax.set_ylim(1650, 200); ax.set_xticks([]); ax.set_yticks([])
            if row == 0: ax.set_title(f"{distance.title()} start", fontsize=10, weight="bold")
            if column == 0: ax.set_ylabel(abundance.title(), fontsize=10, weight="bold")
            artists[key] = {"white": white_line, "trail": trail_line, "agent": agent, "status": status, "ax": ax}

    title = fig.suptitle("", fontsize=13, weight="bold", y=0.985)
    fig.text(
        0.5, 0.014,
        "Representative matched seed 1001 · corrected valence input · red energy / blue water / purple integrity",
        ha="center", fontsize=8,
    )
    progress_background = Rectangle((0.08, 0.035), 0.84, 0.008, transform=fig.transFigure, facecolor="#dfe6eb", edgecolor="none")
    progress = Rectangle((0.08, 0.035), 0, 0.008, transform=fig.transFigure, facecolor="#2a9d65", edgecolor="none")
    fig.add_artist(progress_background); fig.add_artist(progress)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.06, wspace=0.08, hspace=0.13)

    output_path = os.path.join(OUTPUT_DIR, "survivor_experiment_simulation_30s.mp4")
    with imageio.get_writer(
        output_path,
        format="FFMPEG", mode="I", fps=FPS, codec="libx264", pixelformat="yuv420p",
        quality=None, macro_block_size=1,
        output_params=["-crf", "20", "-preset", "medium", "-movflags", "+faststart"],
        ffmpeg_log_level="warning",
    ) as writer:
        for frame_number, step in enumerate(frame_steps, start=1):
            title.set_text(f"Replicated Survivor experiment — step {step:03d} / 799")
            progress.set_width(0.84 * (frame_number / FRAME_COUNT))
            for key, trace in traces.items():
                index = min(step, len(trace["steps"]) - 1)
                current = trace["steps"][index]
                rows = trace["steps"][:index + 1]
                xs = [row["x"] for row in rows]; ys = [row["y"] for row in rows]
                item = artists[key]
                item["white"].set_data(xs, ys); item["trail"].set_data(xs, ys)
                item["agent"].center = (current["x"], current["y"])
                item["agent"].set_facecolor(MOTIVE_COLORS[current["ruling_motive"]])
                trap_step = next((row["step"] for row in trace["steps"] if row["unexpected"]), None)
                trap_label = "trap avoided" if trap_step is None else ("TRAP!" if abs(step - trap_step) <= 6 else f"trap @ {trap_step}")
                item["status"].set_text(
                    f"C{trace['config_id']} · seed {trace['seed']} · {trap_label}\n"
                    f"P {current['emo_pleasure']:+.2f}  V {current['emo_valence']:+.2f}  A {current['emo_activation']:.2f}"
                )
                border = "#d64b3c" if trap_step is not None and abs(step - trap_step) <= 6 else "#222222"
                width = 2.2 if border == "#d64b3c" else 0.8
                for spine in item["ax"].spines.values():
                    spine.set_edgecolor(border); spine.set_linewidth(width)

            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=100, facecolor="white")
            buffer.seek(0)
            writer.append_data(imageio.imread(buffer)[:, :, :3])
            if frame_number == 1 or frame_number % 15 == 0 or frame_number == FRAME_COUNT:
                print(f"Rendered frame {frame_number}/{FRAME_COUNT} (step {step})")
    plt.close(fig)
    validate_video(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
