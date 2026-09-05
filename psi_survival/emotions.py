"""Pinned MicroPsi2 operator adapter and behavior-facing bounded controls."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from micropsi_core.nodenet.stepoperators import DoernerianEmotionalModulators

from .classifier import classify
from .config import ExperimentConfig
from .state import AgentState, ControlValues, Need


PER_TICK_INPUTS = (
    "base_sum_importance_of_intentions",
    "base_sum_urgency_of_intentions",
    "base_competence_for_intention",
    "base_importance_of_intention",
    "base_urgency_of_intention",
    "base_number_of_active_motives",
    "base_number_of_expected_events",
    "base_number_of_unexpected_events",
    "base_urge_change",
    "base_sum_of_urges",
)

PERSISTENT_INPUTS = (
    "base_age",
    "base_unexpectedness",
    "base_age_influence_on_competence",
    "base_porret_decay_factor",
    "emo_competence",
    "emo_sustaining_joy",
)

RAW_OUTPUTS = (
    "emo_activation",
    "emo_resolution",
    "emo_selection_threshold",
    "emo_securing_rate",
    "emo_pleasure",
    "emo_competence",
    "emo_sustaining_joy",
    "emo_valence",
    "base_unexpectedness",
    "base_age",
)


class ExplicitModulatorStore:
    """Minimal nodenet/netapi surface that rejects undocumented missing values."""

    def __init__(self):
        self.values: dict[str, float] = {
            **{name: 0.0 for name in PER_TICK_INPUTS},
            "base_age": 0.0,
            "base_unexpectedness": 0.0,
            "base_age_influence_on_competence": 0.0,
            "base_porret_decay_factor": 1.0,
            "emo_competence": 0.5,
            "emo_sustaining_joy": 0.0,
            "emo_activation": 0.5,
            "emo_resolution": 0.5,
            "emo_selection_threshold": 0.5,
            "emo_securing_rate": 0.5,
            "emo_pleasure": 0.0,
            "emo_valence": 0.0,
        }

    def get_modulator(self, name: str) -> float:
        if name not in self.values:
            raise KeyError(f"The pinned operator requested an uninitialized modulator: {name}")
        return self.values[name]

    def set_modulator(self, name: str, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError(f"Non-finite modulator {name}: {value}")
        self.values[name] = float(value)


class ExplicitNetAPI:
    def __init__(self, store: ExplicitModulatorStore):
        self.store = store

    def get_modulator(self, name: str) -> float:
        return self.store.get_modulator(name)


@dataclass(frozen=True)
class DriveState:
    urges: dict[Need, float]
    importance: dict[Need, float]
    urgency: dict[Need, float]
    strength: dict[Need, float]

    @property
    def sum_urges(self) -> float:
        return sum(self.urges.values())


@dataclass(frozen=True)
class OperatorResult:
    raw: dict[str, float]
    behavior_controls: ControlValues
    classifier_inputs: dict[str, float]
    clipping: dict[str, bool]
    literal_emotion: str
    displayed_emotion: str
    emotion_scores: dict[str, float]
    score_maximum: float
    score_gap: float


def compute_drives(agent: AgentState, config: ExperimentConfig) -> DriveState:
    values = {
        Need.ENERGY: agent.energy,
        Need.WATER: agent.water,
        Need.INTEGRITY: agent.integrity,
    }
    rates = {
        Need.ENERGY: config.physiology.energy_decay,
        Need.WATER: config.physiology.water_decay,
        Need.INTEGRITY: config.physiology.integrity_decay,
    }
    urges = {need: max(0.0, 1.0 - value) for need, value in values.items()}
    urgency = {}
    for need, value in values.items():
        tau = value / rates[need] if rates[need] > 0 else math.inf
        urgency[need] = min(1.0, max(0.0, 1.0 - tau / config.planner.urgency_horizon))
    importance = dict(urges)
    strength = {need: 0.5 * importance[need] + 0.5 * urgency[need] for need in Need}
    return DriveState(urges, importance, urgency, strength)


def _clip(value: float, minimum: float = 0.0, maximum: float = 1.0) -> tuple[float, bool]:
    clipped = min(maximum, max(minimum, value))
    return clipped, clipped != value


def _adapt(raw: Mapping[str, float], config: ExperimentConfig) -> tuple[ControlValues, dict[str, float], dict[str, bool]]:
    activation, clip_a = _clip(raw["emo_activation"])
    resolution, clip_r = _clip(raw["emo_resolution"])
    selection, clip_s = _clip(raw["emo_selection_threshold"])
    q_unclipped = (raw["emo_securing_rate"] - config.emotion.securing_raw_min) / (
        config.emotion.securing_raw_max - config.emotion.securing_raw_min
    )
    securing_rate, clip_q = _clip(q_unclipped)
    pleasure_unclipped = (raw["emo_pleasure"] - config.emotion.pleasure_raw_min) / (
        config.emotion.pleasure_raw_max - config.emotion.pleasure_raw_min
    )
    pleasure, clip_p = _clip(pleasure_unclipped)
    securing_threshold = 1.0 - securing_rate
    classifier_inputs = {
        "activation": activation,
        "resolution": resolution,
        "securing_threshold": securing_threshold,
        "selection_threshold": selection,
        "pleasure": pleasure,
    }
    bounded = ControlValues(activation, resolution, selection, securing_rate, securing_threshold, pleasure)
    clipping = {
        "activation": clip_a,
        "resolution": clip_r,
        "selection_threshold": clip_s,
        "securing_rate": clip_q,
        "pleasure": clip_p,
    }
    return bounded, classifier_inputs, clipping


def _behavior_controls(bounded: ControlValues, condition: str) -> ControlValues:
    resolution = bounded.resolution
    selection = bounded.selection_threshold
    securing_rate = bounded.securing_rate
    if condition == "fixed":
        resolution = selection = securing_rate = 0.5
    elif condition == "ablate_resolution":
        resolution = 0.5
    elif condition == "ablate_selection":
        selection = 0.5
    elif condition == "ablate_securing":
        securing_rate = 0.5
    securing_threshold = 1.0 - securing_rate
    return ControlValues(
        bounded.activation,
        resolution,
        selection,
        securing_rate,
        securing_threshold,
        bounded.pleasure,
    )


def execute_operator(
    agent: AgentState,
    drives: DriveState,
    expected_events: int,
    unexpected_events: int,
    config: ExperimentConfig,
) -> OperatorResult:
    if agent.operator_store is None:
        agent.operator_store = ExplicitModulatorStore()
    store: ExplicitModulatorStore = agent.operator_store
    previous_sum = drives.sum_urges if agent.sum_urges_previous is None else agent.sum_urges_previous
    strategy_competence = 0.5 if agent.strategy is None else agent.competence_by_strategy[agent.strategy]
    ruling = agent.ruling_need
    values = {
        "base_sum_importance_of_intentions": sum(drives.importance.values()),
        "base_sum_urgency_of_intentions": sum(drives.urgency.values()),
        "base_competence_for_intention": strategy_competence,
        "base_importance_of_intention": drives.importance[ruling],
        "base_urgency_of_intention": drives.urgency[ruling],
        "base_number_of_active_motives": 3.0,
        "base_number_of_expected_events": float(expected_events),
        "base_number_of_unexpected_events": float(unexpected_events),
        "base_urge_change": drives.sum_urges - previous_sum,
        "base_sum_of_urges": drives.sum_urges,
    }
    for name, value in values.items():
        store.set_modulator(name, value)
    DoernerianEmotionalModulators().execute(store, (), ExplicitNetAPI(store))
    raw = {name: store.get_modulator(name) for name in RAW_OUTPUTS}
    if not all(math.isfinite(value) for value in raw.values()):
        raise ValueError(f"Non-finite operator output for {agent.id}: {raw}")
    bounded, classifier_inputs, clipping = _adapt(raw, config)
    behavior = _behavior_controls(bounded, config.controller_condition)
    classification = classify(classifier_inputs, config.emotion)
    agent.sum_urges_previous = drives.sum_urges
    agent.raw_modulators = raw
    agent.controls = behavior
    agent.classifier_inputs = classifier_inputs
    agent.emotion_scores = classification.scores
    agent.literal_emotion = classification.literal_argmax
    agent.dominant_emotion = classification.displayed_label
    return OperatorResult(
        raw,
        behavior,
        classifier_inputs,
        clipping,
        classification.literal_argmax,
        classification.displayed_label,
        classification.scores,
        classification.maximum,
        classification.gap,
    )

