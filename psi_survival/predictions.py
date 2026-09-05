"""Exactly-once predictions and strategy-episode competence updates."""

from __future__ import annotations

from dataclasses import asdict

from .state import AgentState, PredictionRecord, Strategy, StrategyEpisode


def start_episode(agent: AgentState, strategy: Strategy, objective: str, target_id: str | None, tick: int) -> StrategyEpisode:
    episode_id = f"{agent.id}:episode:{tick}:{strategy.value}:{target_id or 'none'}"
    existing = agent.episodes.get(episode_id)
    if existing is not None:
        return existing
    episode = StrategyEpisode(episode_id, strategy, objective, tick, target_id)
    agent.episodes[episode_id] = episode
    return episode


def abandon_active_episode(agent: AgentState, tick: int, involuntary_death: bool = False) -> None:
    for episode in agent.episodes.values():
        if episode.status == "active":
            episode.status = "terminal_failure" if involuntary_death else "abandoned"
            episode.success = False
            episode.ended_tick = tick


def complete_episode(agent: AgentState, episode_id: str | None, tick: int, success: bool) -> None:
    if episode_id is None or episode_id not in agent.episodes:
        return
    episode = agent.episodes[episode_id]
    if episode.status != "active":
        return
    episode.status = "completed" if success else "failed"
    episode.success = success
    episode.ended_tick = tick


def create_prediction(
    agent: AgentState,
    episode_id: str,
    category: str,
    predicted,
    condition: str,
    tick: int,
) -> PredictionRecord:
    prediction_id = f"{episode_id}:prediction:{category}"
    if prediction_id in agent.pending_predictions:
        return agent.pending_predictions[prediction_id]
    prediction = PredictionRecord(prediction_id, category, episode_id, tick, predicted, condition)
    agent.pending_predictions[prediction_id] = prediction
    return prediction


def resolve_prediction(agent: AgentState, prediction_id: str, actual, tick: int) -> None:
    prediction = agent.pending_predictions.get(prediction_id)
    if prediction is None or prediction.status != "pending":
        return
    prediction.actual = actual
    prediction.resolved_tick = tick
    prediction.status = "resolved"


def resolve_episode_predictions(agent: AgentState, episode_id: str | None, actual_by_category: dict[str, object], tick: int) -> None:
    if episode_id is None:
        return
    for prediction in agent.pending_predictions.values():
        if prediction.episode_id == episode_id and prediction.category in actual_by_category:
            resolve_prediction(agent, prediction.id, actual_by_category[prediction.category], tick)


def consume_cognitive_outcomes(agent: AgentState) -> tuple[int, int, list[dict], list[dict]]:
    expected = 0
    unexpected = 0
    prediction_records: list[dict] = []
    episode_records: list[dict] = []
    for prediction in agent.pending_predictions.values():
        if prediction.status == "resolved" and not prediction.counted:
            agrees = prediction.predicted == prediction.actual
            expected += int(agrees)
            unexpected += int(not agrees)
            prediction.counted = True
            record = {**asdict(prediction), "agreement": agrees}
            prediction_records.append(record)
            agent.observed_event_ledger.append({"kind": "prediction", **record})
    for episode in agent.episodes.values():
        if episode.status != "active" and not episode.counted:
            success = bool(episode.success)
            previous = agent.competence_by_strategy[episode.strategy]
            updated = 0.95 * previous + 0.05 * float(success)
            agent.competence_by_strategy[episode.strategy] = updated
            episode.counted = True
            record = {**asdict(episode), "competence_before": previous, "competence_after": updated}
            episode_records.append(record)
            agent.observed_event_ledger.append({"kind": "strategy_episode", **record})
    return expected, unexpected, prediction_records, episode_records
