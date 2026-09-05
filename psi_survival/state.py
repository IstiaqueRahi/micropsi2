"""Authoritative and agent-local state structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any


Position = tuple[float, float]
BodyVector = tuple[float, float, float]


class Terrain(str, Enum):
    ROAD = "road"
    ROCKY = "rocky"
    PUDDLE = "puddle"


class ResourceType(str, Enum):
    FOOD = "food"
    WELL = "well"
    HEALING = "healing"
    MIXED_HARMFUL = "mixed_harmful"
    CORPSE = "corpse"


class Need(str, Enum):
    ENERGY = "energy"
    WATER = "water"
    INTEGRITY = "integrity"


class Strategy(str, Enum):
    FORAGE = "forage"
    INSPECT = "inspect"
    WAIT = "wait"
    EXPLORE = "explore"
    FLEE = "flee"
    HUNT = "hunt"


RESOURCE_EFFECTS: dict[ResourceType, BodyVector] = {
    ResourceType.FOOD: (0.80, 0.0, 0.0),
    ResourceType.WELL: (0.0, 0.80, 0.0),
    ResourceType.HEALING: (0.0, 0.0, 0.60),
    ResourceType.MIXED_HARMFUL: (0.30, 0.0, -0.40),
}


@dataclass
class ResourceState:
    id: str
    position: Position
    type: ResourceType
    effect_per_unit: BodyVector
    capacity: float
    stock: float
    regeneration_rate: float
    is_corpse: bool = False
    public_type: bool = False
    spawned_at: float = 0.0
    source_agent_id: str | None = None

    def clamp_stock(self) -> None:
        self.stock = min(self.capacity, max(0.0, self.stock))


@dataclass
class ResourceBelief:
    resource_id: str
    position: Position
    type_distribution: dict[ResourceType, float]
    last_observed_stock: float | None = None
    last_observed_time: float | None = None
    estimated_stock: float = 0.5
    confidence: float = 0.0
    observed_effect: BodyVector | None = None
    is_corpse: bool = False
    last_presence_time: float | None = None

    @property
    def identified(self) -> bool:
        return len(self.type_distribution) == 1

    @property
    def most_likely_types(self) -> list[ResourceType]:
        if not self.type_distribution:
            return []
        maximum = max(self.type_distribution.values())
        return sorted((key for key, value in self.type_distribution.items() if value == maximum), key=lambda x: x.value)


@dataclass
class AgentBelief:
    agent_id: str
    last_position: Position
    last_seen_time: float
    confidence: float = 1.0
    alive: bool = True
    estimated_body: BodyVector = (0.5, 0.5, 0.5)
    last_attack_on_self: float | None = None


@dataclass(frozen=True)
class ControlValues:
    activation: float
    resolution: float
    selection_threshold: float
    securing_rate: float
    securing_threshold: float
    pleasure: float

    @classmethod
    def initial(cls) -> "ControlValues":
        return cls(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)


@dataclass
class PredictionRecord:
    id: str
    category: str
    episode_id: str
    created_tick: int
    predicted: Any
    resolution_condition: str
    status: str = "pending"
    actual: Any = None
    resolved_tick: int | None = None
    counted: bool = False


@dataclass
class StrategyEpisode:
    id: str
    strategy: Strategy
    objective: str
    started_tick: int
    target_id: str | None
    status: str = "active"
    success: bool | None = None
    ended_tick: int | None = None
    counted: bool = False


@dataclass
class Commitment:
    strategy: Strategy
    target_id: str | None
    target_position: Position
    value: float
    need: Need
    route_kind: str
    wait_seconds: int = 0
    wait_until_time: float | None = None
    episode_id: str | None = None
    created_tick: int = 0


@dataclass
class AgentState:
    id: str
    position: Position
    energy: float
    water: float
    integrity: float
    ruling_need: Need
    alive: bool = True
    strategy: Strategy | None = None
    target_id: str | None = None
    commitment: Commitment | None = None
    resource_beliefs: dict[str, ResourceBelief] = field(default_factory=dict)
    agent_beliefs: dict[str, AgentBelief] = field(default_factory=dict)
    competence_by_strategy: dict[Strategy, float] = field(
        default_factory=lambda: {strategy: 0.5 for strategy in Strategy}
    )
    raw_modulators: dict[str, float] = field(default_factory=dict)
    controls: ControlValues = field(default_factory=ControlValues.initial)
    previous_controls: ControlValues = field(default_factory=ControlValues.initial)
    classifier_inputs: dict[str, float] = field(default_factory=dict)
    pending_predictions: dict[str, PredictionRecord] = field(default_factory=dict)
    episodes: dict[str, StrategyEpisode] = field(default_factory=dict)
    observed_event_ledger: list[dict[str, Any]] = field(default_factory=list)
    emotion_scores: dict[str, float] = field(default_factory=dict)
    certainty: float = 0.0
    literal_emotion: str = "Unclassified"
    dominant_emotion: str = "Unclassified"
    kills: int = 0
    death_time: float | None = None
    death_cause: str | None = None
    sum_urges_previous: float | None = None
    operator_store: Any = None
    plan_trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def body(self) -> BodyVector:
        return self.energy, self.water, self.integrity

    def set_body(self, values: BodyVector) -> None:
        self.energy, self.water, self.integrity = values

    def clamp_upper(self) -> BodyVector:
        before = self.body
        self.energy = min(1.0, self.energy)
        self.water = min(1.0, self.water)
        self.integrity = min(1.0, self.integrity)
        return tuple(max(0.0, before[i] - self.body[i]) for i in range(3))  # type: ignore[return-value]


@dataclass
class WorldEvent:
    tick: int
    time: float
    phase: str
    event_type: str
    position: Position | None = None
    agent_id: str | None = None
    target_id: str | None = None
    resource_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldState:
    tick: int
    time: float
    agents: dict[str, AgentState]
    resources: dict[str, ResourceState]
    events: list[WorldEvent] = field(default_factory=list)
    events_since_snapshot: list[WorldEvent] = field(default_factory=list)
    extinct_at: float | None = None

    @property
    def living_agents(self) -> list[AgentState]:
        return [agent for agent in self.agents.values() if agent.alive]


def distance(first: Position, second: Position) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def body_deficit(values: BodyVector) -> float:
    return sum((1.0 - value) ** 2 for value in values) / 3.0
