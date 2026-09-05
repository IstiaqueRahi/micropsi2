"""Resource and agent beliefs maintained without authoritative-state leakage."""

from __future__ import annotations

import math

from .config import ExperimentConfig
from .state import AgentBelief, AgentState, RESOURCE_EFFECTS, ResourceBelief, ResourceState, ResourceType


def initialize_resource_beliefs(agent: AgentState, resources: dict[str, ResourceState], config: ExperimentConfig) -> None:
    prior = {ResourceType(name): probability for name, probability in config.resources.unidentified_prior}
    for resource in resources.values():
        distribution = {resource.type: 1.0} if resource.public_type else dict(prior)
        agent.resource_beliefs[resource.id] = ResourceBelief(
            resource_id=resource.id,
            position=resource.position,
            type_distribution=distribution,
            estimated_stock=0.5 * resource.capacity,
            confidence=0.0,
            observed_effect=RESOURCE_EFFECTS.get(resource.type) if resource.public_type else None,
            last_presence_time=0.0,
        )


def refresh_resource_estimates(agent: AgentState, now: float, config: ExperimentConfig) -> None:
    for belief in agent.resource_beliefs.values():
        capacity = 1.0 if belief.is_corpse else config.resources.capacity
        regeneration_rate = 0.0 if belief.is_corpse else config.resources.regeneration_rate * config.regeneration_multiplier
        if belief.last_observed_stock is None or belief.last_observed_time is None:
            belief.estimated_stock = 0.5 * capacity
            belief.confidence = 0.0
            continue
        elapsed = max(0.0, now - belief.last_observed_time)
        belief.estimated_stock = min(capacity, belief.last_observed_stock + regeneration_rate * elapsed)
        belief.confidence = math.exp(-elapsed / config.perception.resource_confidence_tau)


def observe_resource_presence(
    agent: AgentState,
    resource_id: str,
    position: tuple[float, float],
    is_corpse: bool,
    now: float,
    config: ExperimentConfig,
) -> None:
    if resource_id not in agent.resource_beliefs:
        if is_corpse:
            distribution = {ResourceType.CORPSE: 1.0}
            observed_effect = None
        else:
            distribution = {ResourceType(name): probability for name, probability in config.resources.unidentified_prior}
            observed_effect = None
        agent.resource_beliefs[resource_id] = ResourceBelief(
            resource_id=resource_id,
            position=position,
            type_distribution=distribution,
            estimated_stock=0.5 * (1.0 if is_corpse else config.resources.capacity),
            confidence=0.0,
            observed_effect=observed_effect,
            is_corpse=is_corpse,
            last_presence_time=now,
        )
    else:
        belief = agent.resource_beliefs[resource_id]
        belief.position = position
        belief.last_presence_time = now


def observe_resource_exact(agent: AgentState, resource: ResourceState, now: float) -> None:
    belief = agent.resource_beliefs[resource.id]
    belief.type_distribution = {resource.type: 1.0}
    belief.last_observed_stock = resource.stock
    belief.last_observed_time = now
    belief.estimated_stock = resource.stock
    belief.confidence = 1.0
    belief.observed_effect = resource.effect_per_unit
    belief.is_corpse = resource.is_corpse
    belief.last_presence_time = now


def observe_agent(
    agent: AgentState,
    other_id: str,
    position: tuple[float, float],
    alive: bool,
    now: float,
) -> None:
    existing = agent.agent_beliefs.get(other_id)
    last_attack = existing.last_attack_on_self if existing else None
    agent.agent_beliefs[other_id] = AgentBelief(
        agent_id=other_id,
        last_position=position,
        last_seen_time=now,
        confidence=1.0,
        alive=alive,
        estimated_body=(0.5, 0.5, 0.5),
        last_attack_on_self=last_attack,
    )


def remember_attack(agent: AgentState, attacker_id: str, attacker_position: tuple[float, float], now: float) -> None:
    belief = agent.agent_beliefs.get(attacker_id)
    if belief is None:
        belief = AgentBelief(attacker_id, attacker_position, now)
        agent.agent_beliefs[attacker_id] = belief
    belief.last_position = attacker_position
    belief.last_seen_time = now
    belief.confidence = 1.0
    belief.alive = True
    belief.last_attack_on_self = now


def age_agent_beliefs(agent: AgentState, now: float, config: ExperimentConfig) -> None:
    expired = []
    for other_id, belief in agent.agent_beliefs.items():
        age = max(0.0, now - belief.last_seen_time)
        if age > config.perception.agent_memory_seconds:
            expired.append(other_id)
        else:
            belief.confidence = math.exp(-age / config.perception.agent_memory_tau)
    for other_id in expired:
        del agent.agent_beliefs[other_id]


def active_threats(agent: AgentState, now: float, config: ExperimentConfig) -> list[AgentBelief]:
    return sorted(
        (
            belief
            for belief in agent.agent_beliefs.values()
            if belief.alive
            and belief.last_attack_on_self is not None
            and now - belief.last_attack_on_self <= config.perception.threat_memory_seconds
        ),
        key=lambda belief: belief.agent_id,
    )


def expected_effect(belief: ResourceBelief) -> tuple[float, float, float]:
    if belief.observed_effect is not None:
        return belief.observed_effect
    totals = [0.0, 0.0, 0.0]
    for resource_type, probability in belief.type_distribution.items():
        effect = RESOURCE_EFFECTS.get(resource_type, (0.4, 0.4, 0.0) if resource_type is ResourceType.CORPSE else (0.0, 0.0, 0.0))
        for index in range(3):
            totals[index] += probability * effect[index]
    return tuple(totals)  # type: ignore[return-value]


def type_entropy(belief: ResourceBelief) -> float:
    probabilities = [value for value in belief.type_distribution.values() if value > 0]
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(value * math.log(value) for value in probabilities)
    return entropy / math.log(len(ResourceType) - 1)


def mean_consulted_confidence(resource_beliefs: list[ResourceBelief], agent_beliefs: list[AgentBelief]) -> float:
    values = [belief.confidence for belief in resource_beliefs] + [belief.confidence for belief in agent_beliefs]
    return sum(values) / len(values) if values else 0.0
