"""DeepMD potential energy surface, replacing the CP2K B97-3c PES of the Rust code.

DeepMD works in Angstrom / eV; RTIP-MD works in Bohr / Hartree. The conversion happens here only.
"""

from __future__ import annotations

import typing as t

import numpy as np

from .constants import BOHR_TO_ANGSTROM, EV_TO_HARTREE, atomic_number
from .system import System

# eV/Angstrom -> Hartree/Bohr
EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR = EV_TO_HARTREE * BOHR_TO_ANGSTROM


def lowest_multiplicity(atom_type: list[str], charge: int) -> int:
    """Singlet for an even electron count, doublet for an odd one."""
    n_electron = sum(atomic_number(e) for e in atom_type) - charge
    return 1 if n_electron % 2 == 0 else 2


class DeepmdPES:
    """Energy and forces from a DeepMD model.

    ``pbc=False`` (default) evaluates an isolated cluster, matching the Rust/CP2K setup
    (``B97-3C.inp``: ``PERIODIC NONE``). Coordinates are never wrapped, so the RTI distance
    and the PES always see the same geometry.

    Models with two frame parameters (DPA-3.x) take ``fparam = [charge, spin multiplicity]``.
    The multiplicity defaults to the lowest one consistent with the electron count.
    """

    def __init__(self, dp: t.Any, pbc: bool = False, charge: int = 0, spin: int | None = None) -> None:
        self.dp = dp
        self.type_map: list[str] = list(dp.get_type_map())
        self.pbc = pbc
        self.charge = charge
        self.spin = spin
        self.dim_fparam = int(dp.get_dim_fparam())
        if self.dim_fparam not in (0, 2):
            raise ValueError(f"Unsupported DeepMD model with dim_fparam={self.dim_fparam}")
        if self.dim_fparam == 0 and (charge != 0 or spin is not None):
            raise ValueError("This DeepMD model has no charge/spin frame parameters")
        self._cache: tuple[tuple[str, ...], np.ndarray, np.ndarray | None] | None = None

    @classmethod
    def load(cls, model_file: str, head: str | None = None, **kwargs: t.Any) -> "DeepmdPES":
        from deepmd.infer import DeepPot

        return cls(DeepPot(model_file, head=head) if head else DeepPot(model_file), **kwargs)

    def frame_inputs(self, atom_type: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
        """DeepMD atom types and frame parameters for a given element list."""
        key = tuple(atom_type)
        if self._cache is None or self._cache[0] != key:
            missing = sorted(set(atom_type) - set(self.type_map))
            if missing:
                raise ValueError(f"Elements {missing} are not in the DeepMD model type_map")
            index = {e: i for i, e in enumerate(self.type_map)}
            atype = np.array([index[e] for e in atom_type], dtype=np.int32)
            fparam = None
            if self.dim_fparam == 2:
                spin = self.spin if self.spin is not None else lowest_multiplicity(atom_type, self.charge)
                fparam = np.array([float(self.charge), float(spin)])
            self._cache = (key, atype, fparam)
        return self._cache[1], self._cache[2]

    def get_energy_force(self, s: System) -> tuple[float, np.ndarray]:
        coord = (s.coord * BOHR_TO_ANGSTROM).reshape(1, -1)
        cell = None
        if self.pbc:
            if s.cell is None:
                raise ValueError("pbc=True requires System.cell")
            cell = (s.cell * BOHR_TO_ANGSTROM).reshape(1, 9)
        atype, fparam = self.frame_inputs(s.atom_type)
        if fparam is None:
            energy, force, _virial = self.dp.eval(coord, cell, atype)
        else:
            energy, force, _virial = self.dp.eval(coord, cell, atype, fparam=fparam)
        energy_h = float(np.asarray(energy).reshape(-1)[0]) * EV_TO_HARTREE
        force_h = np.asarray(force, dtype=np.float64).reshape(s.natom, 3) * EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR
        return energy_h, force_h
