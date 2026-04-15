from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp

from ..io.input import Para
from .system import System


@dataclass
class RepulsivePot:
    local_min: System
    nearby_ts: list[System] = field(default_factory=list)
    para: Para | None = None
    output_path: str = ""


@dataclass
class AttractivePot:
    initial_state: System
    final_state: System
    para: Para | None = None
    output_path: str = ""


@dataclass
class SynthesisPot:
    initial_state: System
    mol_index: list[list[int]] = field(default_factory=list)
    para: Para | None = None
    output_path: str = ""


@dataclass
class EvolutionPot:
    initial_state: System
    initial_velocity: jnp.ndarray | None = None
    para: Para | None = None
    output_path: str = ""


@dataclass
class DistributedPot:
    visited_states: list[System] = field(default_factory=list)
    para: Para | None = None
    output_path: str = ""
