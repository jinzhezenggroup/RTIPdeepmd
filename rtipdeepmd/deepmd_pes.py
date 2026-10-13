"""DeepMD potential energy surface, replacing the CP2K B97-3c PES of the Rust code.

DeepMD works in Angstrom / eV; RTIP-MD works in Bohr / Hartree. The conversion happens here only.
"""

from __future__ import annotations

import numpy as np

from .constants import BOHR_TO_ANGSTROM, EV_TO_HARTREE
from .system import System

# eV/Angstrom -> Hartree/Bohr
EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR = EV_TO_HARTREE * BOHR_TO_ANGSTROM


class DeepmdPES:
    """Energy and forces from a DeepMD model.

    ``pbc=False`` (default) evaluates an isolated cluster, matching the Rust/CP2K setup
    (``B97-3C.inp``: ``PERIODIC NONE``). Coordinates are never wrapped, so the RTI distance
    and the PES always see the same geometry.
    """

    def __init__(self, model_file: str, head: str | None = None, pbc: bool = False) -> None:
        from deepmd.infer import DeepPot

        self.dp = DeepPot(model_file, head=head) if head else DeepPot(model_file)
        self.type_map: list[str] = list(self.dp.get_type_map())
        self.pbc = pbc
        self._atype_cache: tuple[tuple[str, ...], np.ndarray] | None = None

    def _atype(self, atom_type: list[str]) -> np.ndarray:
        key = tuple(atom_type)
        if self._atype_cache is None or self._atype_cache[0] != key:
            missing = sorted(set(atom_type) - set(self.type_map))
            if missing:
                raise ValueError(f"Elements {missing} are not in the DeepMD model type_map")
            index = {e: i for i, e in enumerate(self.type_map)}
            self._atype_cache = (key, np.array([index[e] for e in atom_type], dtype=np.int32))
        return self._atype_cache[1]

    def get_energy_force(self, s: System) -> tuple[float, np.ndarray]:
        coord = (s.coord * BOHR_TO_ANGSTROM).reshape(1, -1)
        cell = None
        if self.pbc:
            if s.cell is None:
                raise ValueError("pbc=True requires System.cell")
            cell = (s.cell * BOHR_TO_ANGSTROM).reshape(1, 9)
        energy, force, _virial = self.dp.eval(coord, cell, self._atype(s.atom_type))
        energy_h = float(np.asarray(energy).reshape(-1)[0]) * EV_TO_HARTREE
        force_h = np.asarray(force, dtype=np.float64).reshape(s.natom, 3) * EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR
        return energy_h, force_h
