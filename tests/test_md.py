import numpy as np
import pytest

from rtipdeepmd.constants import ANGSTROM_TO_BOHR, BOLTZMANN, FEMTOSECOND_TO_AU, HARTREE_TO_JOULE
from rtipdeepmd.md import EvolutionPot, evolution_nvt_md, evolution_target, kinetic_temperature
from rtipdeepmd.params import Para
from rtipdeepmd.system import System, split_into_mol

A = ANGSTROM_TO_BOHR


class ZeroPES:
    def get_energy_force(self, s):
        return 0.0, np.zeros_like(s.coord)


class BondSpringPES:
    """Harmonic springs on the initially bonded pairs; no intermolecular interaction."""

    def __init__(self, s, k=0.3):
        d = s.get_dist_mat()
        r = s.get_atomic_radii()
        self.pairs = np.argwhere(np.triu(d < 1.25 * (r[:, None] + r[None, :]), 1))
        self.r0 = d[self.pairs[:, 0], self.pairs[:, 1]]
        self.k = k

    def get_energy_force(self, s):
        i, j = self.pairs.T
        v = s.coord[i] - s.coord[j]
        r = np.linalg.norm(v, axis=1)
        e = 0.5 * self.k * np.sum((r - self.r0) ** 2)
        g = (self.k * (r - self.r0) / r)[:, None] * v
        force = np.zeros_like(s.coord)
        np.add.at(force, i, -g)
        np.add.at(force, j, g)
        return e, force


def read_log(path):
    rows = [line.split() for line in (path / "rtip.out").read_text().splitlines()[1:] if not line.startswith("#")]
    return {
        "step": np.array([int(r[0]) for r in rows]),
        "temp": np.array([float(r[3]) for r in rows]),
        "status": [r[9] for r in rows],
        "a_min": np.array([float(r[10]) for r in rows]),
    }


def two_waters(gap):
    o_h = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
    coord = np.array(o_h + [[x + gap, y, z] for x, y, z in o_h]) * A
    return System(coord=coord, atom_type=["O", "H", "H"] * 2)


def test_kinetic_temperature_uses_electron_masses():
    s = two_waters(5.0)
    masses = s.get_atomic_masses()[:, None]
    rng = np.random.default_rng(0)
    vel = rng.normal(size=s.coord.shape) * np.sqrt(BOLTZMANN * 300.0 / HARTREE_TO_JOULE / masses)
    kin, temp = kinetic_temperature(masses, vel)
    assert kin == pytest.approx(0.5 * np.sum(masses * vel**2))
    assert temp == pytest.approx(np.sum(masses * vel**2) * HARTREE_TO_JOULE / (BOLTZMANN * 3 * (s.natom - 1)))
    assert 100.0 < temp < 900.0


def test_evolution_target_moves_molecules_one_percent_to_centre():
    s = two_waters(5.0)
    mols = split_into_mol(s.get_atomic_radii(), s.get_dist_mat(), 1.25)
    target = evolution_target(s.coord, mols)
    c0, c1 = s.coord[:3].mean(0), s.coord[3:].mean(0)
    t0, t1 = target[:3].mean(0), target[3:].mean(0)
    assert np.linalg.norm(t1 - t0) == pytest.approx(0.99 * np.linalg.norm(c1 - c0))
    np.testing.assert_allclose(target[:3] - s.coord[:3], np.broadcast_to(t0 - c0, (3, 3)))


def test_attractive_rtip_pulls_molecules_together(tmp_path):
    s = two_waters(6.0)
    para = Para(max_step=40, temp_bath=300.0, a0=0.0005)
    final = evolution_nvt_md(EvolutionPot(initial_state=s, para=para, output_path=tmp_path), BondSpringPES(s))
    log = read_log(tmp_path)

    # a_min = -a0 * step while Increasing: the RTIP is attractive and deepens linearly
    assert log["status"] == ["Increasing"] * para.max_step
    np.testing.assert_allclose(log["a_min"], -para.a0 * log["step"])

    d0 = np.linalg.norm(s.coord[3:].mean(0) - s.coord[:3].mean(0))
    d1 = np.linalg.norm(final.coord[3:].mean(0) - final.coord[:3].mean(0))
    assert d1 < d0 - 0.1 * A
    assert np.all(np.isfinite(log["temp"]))
    assert (tmp_path / "rtip.pdb").read_text().count("END\n") == para.max_step + 1


def test_contact_switches_to_falling(tmp_path):
    s = two_waters(1.15)   # O...O at 1.15 A < 1.2 (r_O + r_O): molecules already in contact
    para = Para(max_step=3, temp_bath=300.0)
    evolution_nvt_md(EvolutionPot(initial_state=s, para=para, output_path=tmp_path), BondSpringPES(s))
    log = read_log(tmp_path)
    assert log["status"] == ["Falling"] * 3
    # step 1 is Increasing (initial status), steps 2-3 are Falling: a_min = -a0, 0, +a0
    np.testing.assert_allclose(log["a_min"], [-0.0005, 0.0, 0.0005])


def test_bond_breaking_triggers_decreasing_then_resets(tmp_path):
    # Two H2 molecules far apart; the first one is given velocities that break its bond.
    coord = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0], [6.0, 0.0, 0.0], [6.74, 0.0, 0.0]]) * A
    s = System(coord=coord, atom_type=["H"] * 4)
    v = 0.02 / FEMTOSECOND_TO_AU * A   # 0.02 Angstrom/fs
    vel = np.array([[0.0, -v, 0.0], [0.0, v, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    para = Para(max_step=200, temp_bath=300.0, tau=1.0e9, split_step=None)
    evolution_nvt_md(EvolutionPot(initial_state=s, para=para, output_path=tmp_path, initial_velocity=vel), ZeroPES())
    log = read_log(tmp_path)

    first_dec = log["status"].index("Decreasing")
    back = log["status"].index("Increasing", first_dec)
    a_change = log["a_min"][first_dec - 1]
    # While Decreasing, a_min rises by 2 a0 per step until it passes 0.5 * a_min(at change)
    assert np.allclose(np.diff(log["a_min"][first_dec - 1 : back]), 2 * para.a0)
    assert log["a_min"][back - 1] > 0.5 * a_change >= log["a_min"][back - 2]

    dec_lines = (tmp_path / "rtip_decreasing_steps").read_text().splitlines()
    begin, end = (int(x) for x in dec_lines[1].split())
    assert begin == first_dec + 1 and end == back
