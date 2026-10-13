import numpy as np
import pytest

from rtipdeepmd.constants import ANGSTROM_TO_BOHR, EV_TO_HARTREE
from rtipdeepmd.deepmd_pes import DeepmdPES, lowest_multiplicity
from rtipdeepmd.system import System


class StubDeepPot:
    """Records the inputs of ``eval``; energy 1 eV and force 1 eV/Angstrom on every component."""

    def __init__(self, dim_fparam=2):
        self.dim_fparam = dim_fparam
        self.calls = []

    def get_type_map(self):
        return ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca"]

    def get_dim_fparam(self):
        return self.dim_fparam

    def eval(self, coord, cell, atype, fparam=None):
        self.calls.append({"coord": coord, "cell": cell, "atype": atype, "fparam": fparam})
        natom = len(atype)
        return np.ones((1, 1)), np.ones((1, natom, 3)), np.zeros((1, 9))


def ca_hydroxide():
    coord = np.array([[0.0, 0.0, 0.0], [2.1, 0.0, 0.0], [3.06, 0.0, 0.0]]) * ANGSTROM_TO_BOHR
    return System(coord=coord, atom_type=["Ca", "O", "H"], cell=np.eye(3) * 20.0)


def test_lowest_multiplicity():
    assert lowest_multiplicity(["O", "H", "H"], 0) == 1
    assert lowest_multiplicity(["O", "H"], 0) == 2
    assert lowest_multiplicity(["O", "H"], -1) == 1


def test_units_types_and_default_fparam():
    dp = StubDeepPot()
    pes = DeepmdPES(dp)
    s = ca_hydroxide()
    energy, force = pes.get_energy_force(s)
    call = dp.calls[-1]
    np.testing.assert_allclose(call["coord"].reshape(-1, 3), s.coord / ANGSTROM_TO_BOHR)
    assert call["cell"] is None
    assert list(call["atype"]) == [19, 7, 0]
    # Ca + OH: 20 + 8 + 1 = 29 electrons -> doublet
    np.testing.assert_allclose(call["fparam"], [0.0, 2.0])
    assert energy == pytest.approx(EV_TO_HARTREE)
    np.testing.assert_allclose(force, EV_TO_HARTREE / ANGSTROM_TO_BOHR)


def test_explicit_charge_spin_and_pbc():
    dp = StubDeepPot()
    pes = DeepmdPES(dp, pbc=True, charge=-1, spin=3)
    pes.get_energy_force(ca_hydroxide())
    np.testing.assert_allclose(dp.calls[-1]["fparam"], [-1.0, 3.0])
    np.testing.assert_allclose(dp.calls[-1]["cell"].reshape(3, 3), np.eye(3) * 20.0 / ANGSTROM_TO_BOHR)


def test_model_without_fparam():
    dp = StubDeepPot(dim_fparam=0)
    DeepmdPES(dp).get_energy_force(ca_hydroxide())
    assert dp.calls[-1]["fparam"] is None
    with pytest.raises(ValueError):
        DeepmdPES(StubDeepPot(dim_fparam=0), charge=1)


def test_unknown_element():
    s = System(coord=np.zeros((1, 3)), atom_type=["Fe"])
    with pytest.raises(ValueError, match="Fe"):
        DeepmdPES(StubDeepPot()).get_energy_force(s)
