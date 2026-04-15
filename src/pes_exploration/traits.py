from __future__ import annotations

import typing as t
from typing import Protocol

import jax.numpy as jnp

if t.TYPE_CHECKING:
    from .system import System


class PES(Protocol):
    def get_energy(self, s: "System") -> float:
        ...

    def get_energy_force(self, s: "System") -> tuple[float, jnp.ndarray]:
        ...


class RtipPathSampling(Protocol):
    def rtip_path_sampling(self, comm: t.Any, real_pes: PES) -> None:
        ...


class IdwmPathSampling(Protocol):
    def idwm_path_sampling(self, comm: t.Any, real_pes: PES) -> None:
        ...


class RtipNVTMD(Protocol):
    def rtip_nvt_md(self, comm: t.Any, real_pes: PES) -> None:
        ...
