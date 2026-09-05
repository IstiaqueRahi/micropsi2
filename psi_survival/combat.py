"""Probabilistic one-duel combat and order-independent matching."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .config import ExperimentConfig
from .rng import priority, uniform
from .state import AgentState, BodyVector


@dataclass(frozen=True)
class AttackRequest:
    attacker_id: str
    defender_id: str


@dataclass(frozen=True)
class DuelPair:
    first_id: str
    second_id: str
    requesters: tuple[str, ...]
    priority: int


@dataclass(frozen=True)
class DuelResult:
    winner_id: str
    loser_id: str
    first_win_probability: float
    draw: float


def strength(body: BodyVector, config: ExperimentConfig) -> float:
    return (
        config.combat.strength_energy_weight * body[0]
        + config.combat.strength_water_weight * body[1]
        + config.combat.strength_integrity_weight * body[2]
    )


def win_probability(first_body: BodyVector, second_body: BodyVector, config: ExperimentConfig) -> float:
    delta = (strength(first_body, config) - strength(second_body, config)) / config.combat.sigmoid_scale
    if delta >= 0:
        return 1.0 / (1.0 + math.exp(-delta))
    exponential = math.exp(delta)
    return exponential / (1.0 + exponential)


def match_attacks(requests: list[AttackRequest], tick: int, config: ExperimentConfig) -> tuple[list[DuelPair], list[AttackRequest]]:
    unique: dict[tuple[str, str], set[str]] = {}
    for request in requests:
        if request.attacker_id == request.defender_id:
            continue
        pair = tuple(sorted((request.attacker_id, request.defender_id)))
        unique.setdefault(pair, set()).add(request.attacker_id)
    pairs = [
        DuelPair(
            first_id=pair[0],
            second_id=pair[1],
            requesters=tuple(sorted(requesters)),
            priority=priority(config.world_seed, "combat-match", tick, pair[0], pair[1]),
        )
        for pair, requesters in unique.items()
    ]
    pairs.sort(key=lambda pair: (pair.priority, pair.first_id, pair.second_id))
    accepted: list[DuelPair] = []
    used: set[str] = set()
    accepted_keys: set[tuple[str, str]] = set()
    for pair in pairs:
        if pair.first_id in used or pair.second_id in used:
            continue
        accepted.append(pair)
        used.update((pair.first_id, pair.second_id))
        accepted_keys.add((pair.first_id, pair.second_id))
    deferred = [
        request
        for request in requests
        if tuple(sorted((request.attacker_id, request.defender_id))) not in accepted_keys
    ]
    return accepted, deferred


def resolve_duel(pair: DuelPair, agents: dict[str, AgentState], tick: int, config: ExperimentConfig) -> DuelResult:
    first = agents[pair.first_id]
    second = agents[pair.second_id]
    probability = win_probability(first.body, second.body, config)
    draw = uniform(config.world_seed, "combat-outcome", tick, pair.first_id, pair.second_id)
    if draw < probability:
        winner, loser = first, second
    else:
        winner, loser = second, first
    return DuelResult(winner.id, loser.id, probability, draw)

