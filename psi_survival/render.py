"""Saved-result-only plots and exact 180-second playback rendering."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import math
import os
from pathlib import Path
import struct
import subprocess
import time
from typing import Any

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .storage import load_result, write_json


TERRAIN_COLORS = {
    "road": "#d8c89b",
    "rocky": "#8f9295",
    "puddle": "#6aaed6",
}
RESOURCE_COLORS = {
    "food": "#2a9d65",
    "well": "#2582d8",
    "healing": "#8b52a1",
    "mixed_harmful": "#c83d35",
    "corpse": "#1d1d1d",
}
AGENT_COLORS = (
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
    "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324",
    "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075",
    "#808080", "#000000", "#a9a9ff", "#ff7f7f", "#7fbf7f", "#ffbf7f",
)
EMOTION_COLORS = {
    "Angry": "#d73027",
    "Fear": "#7b3294",
    "Happy": "#1a9850",
    "Sad": "#4575b4",
    "Unclassified": "#bdbdbd",
    "Mixed": "#fdae61",
    "Dead": "#222222",
}


def _font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _terrain_image(result: dict[str, Any], map_left: int, map_top: int, map_size: int) -> Image.Image:
    rows = result["initial_state"]["terrain"]
    height, width = len(rows), len(rows[0])
    image = Image.new("RGB", (map_size, map_size), "white")
    draw = ImageDraw.Draw(image)
    cell_x = map_size / width
    cell_y = map_size / height
    for y, row in enumerate(rows):
        for x, terrain in enumerate(row):
            draw.rectangle(
                (int(x * cell_x), int(y * cell_y), int((x + 1) * cell_x + 1), int((y + 1) * cell_y + 1)),
                fill=TERRAIN_COLORS[terrain],
            )
    return image


def _map_point(position, map_left: int, map_top: int, map_size: int, width: int = 64, height: int = 64):
    return (
        map_left + int((position[0] + 0.5) / width * map_size),
        map_top + int((position[1] + 0.5) / height * map_size),
    )


def render_video(result_path: str | Path, output_path: str | Path) -> Path:
    result = load_result(result_path)
    frames = result["world_frames"]
    config = result["config"]
    expected_frames = int(config["playback"]["ticks"])
    fps = int(config["playback"]["fps"])
    if len(frames) != expected_frames:
        raise ValueError(f"Expected {expected_frames} post-tick frames, found {len(frames)}")
    width = int(config["playback"].get("width_px", 1280))
    height = int(config["playback"].get("height_px", 720))
    if expected_frames == 3600 and fps != 20:
        raise ValueError("The preregistered demonstration must use 20 fps")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())

    map_left, map_top, map_size = 20, 50, min(640, height - 70)
    terrain = _terrain_image(result, map_left, map_top, map_size)
    base = Image.new("RGB", (width, height), "#f7f7f5")
    base.paste(terrain, (map_left, map_top))
    base_draw = ImageDraw.Draw(base)
    base_draw.rectangle((map_left, map_top, map_left + map_size, map_top + map_size), outline="#222222", width=2)
    title_font = _font(22, bold=True)
    text_font = _font(14)
    small_font = _font(11)
    base_draw.text((20, 14), "Hybrid PSI survival experiment — observer truth", fill="#111111", font=title_font)
    base_draw.text((700, 18), "Observer view: hidden types/stocks are not controller inputs", fill="#7a1f1f", font=text_font)
    base_draw.text(
        (700, height - 58),
        "Events: discovery ◆  consume ○  failed consume ○  attack/death ×",
        fill="#333333",
        font=small_font,
    )
    initial_agents = sorted(result["initial_state"]["agents"], key=lambda item: item["id"])
    agent_colors = {item["id"]: AGENT_COLORS[index % len(AGENT_COLORS)] for index, item in enumerate(initial_agents)}
    events_by_tick: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in result["events"]:
        events_by_tick[int(event["tick"])].append(event)

    with imageio.get_writer(
        str(output),
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        quality=None,
        output_params=["-crf", "20", "-preset", "medium", "-movflags", "+faststart"],
        ffmpeg_log_level="warning",
    ) as writer:
        for index, frame in enumerate(frames):
            image = base.copy()
            draw = ImageDraw.Draw(image)
            tick = int(frame["tick"])
            resources = frame["resources"]
            agents = frame["agents"]
            for resource_id, resource in resources.items():
                x, y = _map_point(resource["position"], map_left, map_top, map_size)
                stock_fraction = resource["stock"] / resource["capacity"] if resource["capacity"] else 0.0
                radius = 3 + int(4 * stock_fraction)
                color = RESOURCE_COLORS[resource["type"]]
                if resource["is_corpse"]:
                    draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=3)
                    draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=3)
                else:
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="#111111")
            for agent_id, agent in agents.items():
                if not agent["alive"]:
                    continue
                x, y = _map_point(agent["position"], map_left, map_top, map_size)
                target_id = agent.get("target_id")
                target = resources.get(target_id) or agents.get(target_id)
                if target is not None:
                    tx, ty = _map_point(target["position"], map_left, map_top, map_size)
                    draw.line((x, y, tx, ty), fill=agent_colors[agent_id], width=1)
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=agent_colors[agent_id], outline="#000000", width=1)
                draw.text((x + 7, y - 7), agent_id.split("-")[-1], fill="#111111", font=small_font)
            for event in events_by_tick.get(tick, ()):
                if event["event_type"] == "attack" and event["position"] is not None:
                    x, y = _map_point(event["position"], map_left, map_top, map_size)
                    draw.ellipse((x - 13, y - 13, x + 13, y + 13), outline="#ff0000", width=3)
                elif event["event_type"] == "death" and event["position"] is not None:
                    x, y = _map_point(event["position"], map_left, map_top, map_size)
                    draw.line((x - 10, y - 10, x + 10, y + 10), fill="#000000", width=3)
                    draw.line((x - 10, y + 10, x + 10, y - 10), fill="#000000", width=3)
                elif event["event_type"] == "inspection" and event["position"] is not None:
                    x, y = _map_point(event["position"], map_left, map_top, map_size)
                    draw.polygon(((x, y - 11), (x + 11, y), (x, y + 11), (x - 11, y)), outline="#ffd000")
                elif event["event_type"] == "consumption" and event["position"] is not None:
                    x, y = _map_point(event["position"], map_left, map_top, map_size)
                    draw.ellipse((x - 11, y - 11, x + 11, y + 11), outline="#00a651", width=3)
                elif event["event_type"] == "failed_consumption" and event["position"] is not None:
                    x, y = _map_point(event["position"], map_left, map_top, map_size)
                    draw.ellipse((x - 11, y - 11, x + 11, y + 11), outline="#ff8c00", width=3)
            draw.text((700, 52), f"Tick {tick:04d} / {expected_frames}   simulated time {frame['time']:.0f}s", fill="#111111", font=text_font)
            alive_count = sum(agent["alive"] for agent in agents.values())
            draw.text((700, 78), f"Alive: {alive_count}/{len(agents)}", fill="#111111", font=text_font)
            y_cursor = 112
            for agent_id in sorted(agents):
                agent = agents[agent_id]
                body = agent["body"]
                status = (
                    f"E {body[0]:.2f} W {body[1]:.2f} I {body[2]:.2f}  "
                    f"{agent['strategy'] or '-'} · {agent['emotion']}"
                    if agent["alive"] else "DEAD"
                )
                draw.rectangle((700, y_cursor + 2, 712, y_cursor + 14), fill=agent_colors[agent_id], outline="#111111")
                draw.text((718, y_cursor), f"{agent_id}: {status}", fill="#222222", font=small_font)
                y_cursor += 24
            progress_left, progress_right = 700, width - 35
            draw.rectangle((progress_left, height - 35, progress_right, height - 23), fill="#d8dde0")
            draw.rectangle(
                (progress_left, height - 35, progress_left + int((progress_right - progress_left) * (index + 1) / expected_frames), height - 23),
                fill="#2a9d65",
            )
            writer.append_data(np.asarray(image))
    validate_video(output, expected_frames, fps)
    return output


def validate_video(path: str | Path, expected_frames: int, fps: int) -> None:
    payload = Path(path).read_bytes()
    table = payload.find(b"stsz")
    samples = struct.unpack_from(">I", payload, table + 12)[0] if table >= 0 else 0
    if samples != expected_frames or b"avcC" not in payload:
        raise RuntimeError(f"Expected {expected_frames} H.264 samples, found {samples}")
    completed = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True,
        text=True,
        timeout=max(90, int(expected_frames / fps * 2)),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Video decode validation failed: {completed.stderr.strip()}")


def _bucket_indices(ticks: int, bins: int) -> list[tuple[int, int]]:
    return [
        (round(index * ticks / bins), round((index + 1) * ticks / bins))
        for index in range(bins)
    ]


def _modal(values: list[str]) -> str:
    if not values:
        return "Dead"
    counts = Counter(values)
    maximum = max(counts.values())
    winners = [name for name, count in counts.items() if count == maximum]
    return winners[0] if len(winners) == 1 else "Mixed"


def render_static_plots(
    result_path: str | Path,
    output_dir: str | Path,
    *,
    display_ema_lambda: float | None = None,
) -> list[Path]:
    result = load_result(result_path)
    if not result["traces"]:
        raise ValueError("Static plots require a full-trace run")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    agents = sorted(result["initial_state"]["agents"], key=lambda item: item["id"])
    agent_ids = [item["id"] for item in agents]
    ticks = int(result["config"]["playback"]["ticks"])
    bins = int(result["config"]["playback"]["heatmap_bins"])
    bucket_ranges = _bucket_indices(ticks, bins)
    paths: list[Path] = []

    fig, axes = plt.subplots(4, math.ceil(len(agent_ids) / 4), figsize=(16, 11), sharex=True, sharey=True)
    flat = np.asarray(axes).reshape(-1)
    for axis, agent_id in zip(flat, agent_ids):
        rows = result["traces"][agent_id]
        times = [row["time"] for row in rows]
        for need, color in (("energy", "#d64b3c"), ("water", "#2878b5"), ("integrity", "#8b52a1")):
            axis.plot(times, [row["urges"][need] for row in rows], color=color, linewidth=0.8, label=need)
        hunt = [row["strategy"] == "hunt" for row in rows]
        axis.fill_between(times, 0, 1, where=hunt, color="#f4a261", alpha=0.25, transform=axis.get_xaxis_transform())
        axis.set_title(agent_id, fontsize=9)
        axis.set_ylim(0, 1)
    for axis in flat[len(agent_ids):]:
        axis.axis("off")
    flat[0].legend(loc="upper right", fontsize=7)
    fig.suptitle("Physiological urges by agent; orange bands indicate hunting")
    fig.supxlabel("Simulated seconds"); fig.supylabel("Urge")
    fig.tight_layout()
    path = output / "motives-small-multiples.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)

    label_order = list(EMOTION_COLORS)
    label_code = {label: index for index, label in enumerate(label_order)}
    heat = np.zeros((len(agent_ids), bins), dtype=int)
    for row_index, agent_id in enumerate(agent_ids):
        by_tick = {int(row["tick"]): row for row in result["traces"][agent_id]}
        labels = [
            (by_tick[tick]["dominant_emotion"] if tick in by_tick and by_tick[tick]["alive"] else "Dead")
            for tick in range(1, ticks + 1)
        ]
        for column, (start, end) in enumerate(bucket_ranges):
            heat[row_index, column] = label_code[_modal(labels[start:end])]
    fig, axis = plt.subplots(figsize=(16, 5))
    axis.imshow(heat, aspect="auto", interpolation="nearest", cmap=ListedColormap([EMOTION_COLORS[label] for label in label_order]), vmin=0, vmax=len(label_order) - 1)
    axis.set_yticks(range(len(agent_ids)), agent_ids)
    axis.set_xlabel("120 equal time bins across 3,600 seconds")
    axis.set_title("Displayed emotion labels; categorical ties are Mixed")
    axis.legend(handles=[Patch(color=EMOTION_COLORS[label], label=label) for label in label_order], ncol=len(label_order), loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8)
    fig.tight_layout()
    path = output / "emotion-categorical-heatmap.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)

    panels = ("activation", "Angry", "Fear", "Happy", "Sad")
    arrays = {panel: np.full((len(agent_ids), bins), np.nan) for panel in panels}
    for row_index, agent_id in enumerate(agent_ids):
        by_tick = {int(row["tick"]): row for row in result["traces"][agent_id]}
        for column, (start, end) in enumerate(bucket_ranges):
            rows = [by_tick[tick] for tick in range(start + 1, end + 1) if tick in by_tick]
            if rows:
                arrays["activation"][row_index, column] = np.mean([row["raw_modulators"]["emo_activation"] for row in rows])
                for emotion in panels[1:]:
                    arrays[emotion][row_index, column] = np.mean([row["emotion_scores"][emotion] for row in rows])
    fig, axes = plt.subplots(len(panels), 1, figsize=(16, 11), sharex=True)
    for axis, panel in zip(axes, panels):
        image = axis.imshow(arrays[panel], aspect="auto", interpolation="nearest", cmap="viridis")
        axis.set_yticks(range(len(agent_ids)), agent_ids, fontsize=7)
        axis.set_title("Raw activation" if panel == "activation" else f"{panel} membership intensity")
        fig.colorbar(image, ax=axis, fraction=0.015, pad=0.01)
    axes[-1].set_xlabel("120 equal time bins")
    fig.tight_layout()
    path = output / "emotion-intensity-panels.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(4, math.ceil(len(agent_ids) / 4), figsize=(16, 11), sharex=True)
    flat = np.asarray(axes).reshape(-1)
    smoothing = (
        float(result["config"]["emotion"]["display_ema_lambda"])
        if display_ema_lambda is None
        else float(display_ema_lambda)
    )
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("display_ema_lambda must lie in (0, 1]")
    for axis, agent_id in zip(flat, agent_ids):
        rows = result["traces"][agent_id]
        raw = [row["raw_modulators"]["emo_activation"] for row in rows]
        ema = []
        for value in raw:
            ema.append(value if not ema else smoothing * value + (1.0 - smoothing) * ema[-1])
        axis.plot([row["time"] for row in rows], raw, color="#aaaaaa", linewidth=0.55, label="raw")
        axis.plot([row["time"] for row in rows], ema, color="#c0392b", linewidth=1.0, label="display EMA")
        axis.set_title(agent_id, fontsize=9)
    for axis in flat[len(agent_ids):]: axis.axis("off")
    flat[0].legend(fontsize=7)
    fig.suptitle(f"Raw activation and causal display-only EMA (λ = {smoothing:g})")
    fig.supxlabel("Simulated seconds")
    fig.tight_layout()
    path = output / "activation-raw-and-display-ema.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)
    return paths


def render_all(
    result_path: str | Path,
    output_dir: str | Path,
    *,
    display_ema_lambda: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    source = Path(result_path)
    source_hash_before = hashlib.sha256(source.read_bytes()).hexdigest()
    result = load_result(source)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plots = render_static_plots(source, output, display_ema_lambda=display_ema_lambda)
    video = render_video(source, output / "demonstration-180s.mp4")
    source_hash_after = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash_before != source_hash_after:
        raise RuntimeError("Rendering mutated the recorded simulation artifact")
    ticks = int(result["config"]["playback"]["ticks"])
    fps = int(result["config"]["playback"]["fps"])
    smoothing = (
        float(result["config"]["emotion"]["display_ema_lambda"])
        if display_ema_lambda is None
        else float(display_ema_lambda)
    )
    metadata = {
        "source_result": str(source),
        "source_result_sha256": source_hash_before,
        "video": str(video),
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "plots": [str(path) for path in plots],
        "post_tick_frames": ticks,
        "fps": fps,
        "playback_seconds": ticks / fps,
        "display_ema_lambda": smoothing,
        "elapsed_render_seconds": time.perf_counter() - started,
    }
    write_json(output / "render-metadata.json", metadata, compressed=False)
    return metadata
