"""Literal Cai et al. Table 2 fuzzy emotion classifier."""

from __future__ import annotations

from dataclasses import dataclass

from .config import EmotionParameters


PROFILE_ORDER = ("Angry", "Fear", "Happy", "Sad")
DIMENSION_ORDER = ("activation", "resolution", "securing_threshold", "selection_threshold", "pleasure")
PROFILES: dict[str, tuple[str, ...]] = {
    "Angry": ("H", "L", "H", "L", "L"),
    "Fear": ("H", "L", "L", "H", "L"),
    "Happy": ("H", "L", "H", "H", "H"),
    "Sad": ("L", "H", "L", "L", "L"),
}


@dataclass(frozen=True)
class Classification:
    scores: dict[str, float]
    literal_argmax: str
    displayed_label: str
    maximum: float
    runner_up: float
    gap: float


def fuzzy_equal(value: float, target: float, sharpness: float = 150.0) -> float:
    return 1.0 / (1.0 + sharpness * (value - target) ** 2)


def fuzzy_low(value: float, target: float, sharpness: float = 150.0) -> float:
    return 1.0 if value <= target else fuzzy_equal(value, target, sharpness)


def fuzzy_high(value: float, target: float, sharpness: float = 150.0) -> float:
    return 1.0 if value >= target else fuzzy_equal(value, target, sharpness)


def membership(value: float, indicator: str, sharpness: float = 150.0) -> float:
    if indicator == "EL":
        return fuzzy_low(value, 0.15, sharpness)
    if indicator == "L":
        return fuzzy_low(value, 0.30, sharpness)
    if indicator == "M":
        return fuzzy_equal(value, 0.50, sharpness)
    if indicator == "H":
        return fuzzy_high(value, 0.70, sharpness)
    if indicator == "EH":
        return fuzzy_high(value, 0.85, sharpness)
    if indicator == "U":
        return 0.0
    raise ValueError(f"Unknown fuzzy indicator: {indicator}")


def classify(inputs: dict[str, float], parameters: EmotionParameters | None = None) -> Classification:
    parameters = parameters or EmotionParameters()
    missing = set(DIMENSION_ORDER) - set(inputs)
    if missing:
        raise ValueError(f"Missing classifier inputs: {sorted(missing)}")
    scores = {
        emotion: sum(
            0.2 * membership(inputs[dimension], indicator, parameters.fuzzy_sharpness)
            for dimension, indicator in zip(DIMENSION_ORDER, profile)
        )
        for emotion, profile in PROFILES.items()
    }
    ranked = sorted(PROFILE_ORDER, key=lambda name: (-scores[name], PROFILE_ORDER.index(name)))
    maximum = scores[ranked[0]]
    runner_up = scores[ranked[1]]
    gap = maximum - runner_up
    displayed = (
        ranked[0]
        if maximum >= parameters.display_score_threshold and gap >= parameters.display_gap_threshold
        else "Unclassified"
    )
    return Classification(scores, ranked[0], displayed, maximum, runner_up, gap)

