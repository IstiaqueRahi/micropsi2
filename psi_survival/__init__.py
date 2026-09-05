"""Reproducible hybrid MicroPsi2 survival experiment.

The package deliberately separates the authoritative physical world from the
policy views given to agents.  MicroPsi2 supplies emotional-modulator updates;
terrain, beliefs, planning, combat, and experimental analysis are project
extensions described by the version 2.0 protocol.
"""

from .config import ExperimentConfig

__all__ = ["ExperimentConfig"]
__version__ = "2.0.0"
