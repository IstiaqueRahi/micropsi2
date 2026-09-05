"""Branch-stable keyed random streams.

Each draw is determined by semantic keys rather than by the number of earlier
draws.  Different controller branches therefore cannot shift combat, map, or
perception randomness elsewhere in a paired run.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Sequence, TypeVar


T = TypeVar("T")


def keyed_seed(master_seed: int, *parts: Any) -> int:
    payload = json.dumps([int(master_seed), *parts], sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16, person=b"psi-survival-v2").digest()
    return int.from_bytes(digest, "big")


def stream(master_seed: int, *parts: Any) -> random.Random:
    return random.Random(keyed_seed(master_seed, *parts))


def uniform(master_seed: int, *parts: Any) -> float:
    return stream(master_seed, *parts).random()


def bernoulli(probability: float, master_seed: int, *parts: Any) -> bool:
    probability = min(1.0, max(0.0, probability))
    return uniform(master_seed, *parts) < probability


def choice(values: Sequence[T], master_seed: int, *parts: Any) -> T:
    if not values:
        raise ValueError("Cannot choose from an empty sequence")
    return values[stream(master_seed, *parts).randrange(len(values))]


def priority(master_seed: int, *parts: Any) -> int:
    return keyed_seed(master_seed, "priority", *parts)

