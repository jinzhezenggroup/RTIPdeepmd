"""ASE calculator around ``DeepmdPES`` (for geometry optimisation, NEB, ...).

Uses the same head and charge/spin frame parameters as the MD. Requires ``ase``.
"""

from __future__ import annotations

import typing as t

from ase.calculators.calculator import Calculator, all_changes

from .constants import ANGSTROM_TO_BOHR, EV_TO_HARTREE
from .deepmd_pes import EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR, DeepmdPES
from .system import System


class DeepmdCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, pes: DeepmdPES, **kwargs: t.Any) -> None:
        super().__init__(**kwargs)
        self.pes = pes

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        s = System(coord=self.atoms.positions * ANGSTROM_TO_BOHR, atom_type=self.atoms.get_chemical_symbols())
        energy, force = self.pes.get_energy_force(s)
        self.results = {
            "energy": energy / EV_TO_HARTREE,
            "forces": force / EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR,
        }
