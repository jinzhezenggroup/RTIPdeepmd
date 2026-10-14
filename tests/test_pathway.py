import numpy as np
import pytest

from rtipdeepmd.pathway import (
    LineSearchError,
    PathPara,
    SynthesisPot,
    min_1d,
    place_molecules,
    synthesis_path_sampling,
)
from rtipdeepmd.system import System, dist_mat


def test_min_1d_finds_quadratic_minimum():
    x, f = min_1d(lambda x: (x - 0.05) ** 2, 0.05**2, 1e-12)
    assert x == pytest.approx(0.05, abs=1e-4)
    # the step is capped once the bracket passes x = 0.1
    x, _ = min_1d(lambda x: -x, 0.0, 1e-12)
    assert 0.1 <= x < 0.3
    with pytest.raises(LineSearchError):
        min_1d(lambda x: x, 0.0, 1e-12)


def test_place_molecules_two_and_three():
    rng = np.random.default_rng(0)
    a = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.5, 1.7, 0.0]])
    s = System(coord=np.concatenate([a, a[:2] + 3.0, a + 7.0]), atom_type=["O", "H", "H", "H", "H", "O", "H", "H"])
    mols = [[0, 1, 2], [3, 4], [5, 6, 7]]
    two = System(coord=s.coord[:5], atom_type=s.atom_type[:5])
    out = place_molecules(two, mols[:2], 5.0, rng)
    np.testing.assert_allclose(out.coord[[0, 1, 2]].mean(0), [-5.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(out.coord[[3, 4]].mean(0), [5.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(dist_mat(out.coord[:3]), dist_mat(two.coord[:3]), atol=1e-12)
    out3 = place_molecules(s, mols, 5.0, rng)
    centres = [out3.coord[m].mean(0) for m in mols]
    assert all(np.linalg.norm(c) == pytest.approx(5.0) for c in centres)


class DoubleWell:
    """E(r) of two atoms: barrier of height h at r = 3 Bohr, product well of -0.1 Ha at r = 1.4 Bohr."""

    def __init__(self, h):
        self.h = h

    def _e_de(self, r):
        e = self.h * np.exp(-((r - 3.0) ** 2) / 0.5) - 0.1 * np.exp(-((r - 1.4) ** 2) / 0.3)
        de = self.h * np.exp(-((r - 3.0) ** 2) / 0.5) * (-2 * (r - 3.0) / 0.5) - 0.1 * np.exp(-((r - 1.4) ** 2) / 0.3) * (
            -2 * (r - 1.4) / 0.3
        )
        return e, de

    def get_energy(self, s):
        return self.get_energy_force(s)[0]

    def get_energy_force(self, s):
        v = s.coord[1] - s.coord[0]
        r = np.linalg.norm(v)
        e, de = self._e_de(r)
        f = np.zeros_like(s.coord)
        f[1] = -de * v / r
        f[0] = de * v / r
        return float(e), f


def two_atoms(rng):
    s = System(coord=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), atom_type=["H", "H"])
    return place_molecules(s, [[0], [1]], 5.0, rng)


def test_synthesis_crosses_barrier_and_relaxes_to_product(tmp_path):
    s = two_atoms(np.random.default_rng(0))
    res = synthesis_path_sampling(SynthesisPot(s, [[0], [1]], PathPara(), tmp_path), DoubleWell(0.03))
    assert res.status == "product"
    assert res.bias_removed_at is not None and res.bias_removed_at < res.steps
    assert np.linalg.norm(res.final_state.coord[1] - res.final_state.coord[0]) == pytest.approx(1.4, abs=0.02)
    assert res.pot_max == pytest.approx(0.03, abs=0.005)
    log = (tmp_path / "rtip.out").read_text().splitlines()
    assert log[-1].startswith("# END") and "status=product" in log[-1]
    a_min = [float(l.split()[6]) for l in log[1:-1]]
    np.testing.assert_allclose(a_min[:5], [-0.0005 * k for k in range(1, 6)])
    assert all(a == 0.0 for a in a_min[res.bias_removed_at:])


def test_synthesis_gives_up_above_pot_climb(tmp_path):
    s = two_atoms(np.random.default_rng(0))
    res = synthesis_path_sampling(SynthesisPot(s, [[0], [1]], PathPara(), tmp_path), DoubleWell(0.3))
    assert res.status == "climb"
    assert res.pot_max > res.pot_min + PathPara().pot_climb


def test_cli_synthesis_dry_run(tmp_path):
    import json

    from rtipdeepmd.cli import DEFAULT_XYZ_DIR, main

    out = tmp_path / "syn"
    main(["--method", "synthesis", "--dry-run", "--seed", "3", "--output-dir", str(out),
          "--molecule", str(DEFAULT_XYZ_DIR / "CH2O.xyz"), "--molecule", str(DEFAULT_XYZ_DIR / "H2O.xyz")])
    s = System.read_xyz(out / "initial.xyz")
    assert s.atom_type == ["O", "C", "H", "H", "O", "H", "H"] or s.natom == 7
    info = json.loads((out / "run.json").read_text())
    assert info["method"] == "synthesis" and info["para"]["max_step"] == 2000
    c1, c2 = s.coord[:4].mean(0), s.coord[4:].mean(0)
    assert np.linalg.norm(c2 - c1) == pytest.approx(10.0, abs=1e-6)   # two centres at -/+ 5 Bohr


def test_fragment_rtip_only_acts_on_fragment_atoms(tmp_path):
    from rtipdeepmd.cli import parse_fragments

    assert parse_fragments(["1:2,4-5"], [[0, 1, 2, 3, 4], [5, 6]]) == [[1, 3, 4], [5, 6]]
    # the third atom is an environment spectator far away: no RTIP force on it
    s = System(coord=np.array([[-5.0, 0, 0], [5.0, 0, 0], [0.0, 30.0, 0.0]]), atom_type=["H", "H", "H"])

    class Env(DoubleWell):
        def get_energy_force(self, s):
            e, f = DoubleWell.get_energy_force(self, System(coord=s.coord[:2], atom_type=["H", "H"]))
            return e, np.vstack([f, np.zeros((1, 3))])

    res = synthesis_path_sampling(SynthesisPot(s, [[0], [1]], PathPara(), tmp_path), Env(0.03))
    assert res.status == "product"
    np.testing.assert_allclose(res.final_state.coord[2], [0.0, 30.0, 0.0])


def test_place_molecules_min_contact():
    rng = np.random.default_rng(5)
    a = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    s = System(coord=np.concatenate([a, a]), atom_type=["C"] * 6)
    out = place_molecules(s, [[0, 1, 2], [3, 4, 5]], 4.0, rng, min_contact=3.0)
    assert dist_mat(out.coord)[:3, 3:].min() >= 3.0
    with pytest.raises(ValueError):
        place_molecules(s, [[0, 1, 2], [3, 4, 5]], 0.5, rng, min_contact=50.0, max_tries=20)


def test_attractive_pulls_to_final_state_and_relaxes(tmp_path):
    from rtipdeepmd.pathway import AttractivePot, attractive_path_sampling

    start = System(coord=np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), atom_type=["H", "H"])
    final = System(coord=np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]), atom_type=["H", "H"])
    res = attractive_path_sampling(AttractivePot(start, final, PathPara(), tmp_path), DoubleWell(0.03))
    assert res.status == "product"
    assert res.bias_removed_at is not None
    assert np.linalg.norm(res.final_state.coord[1] - res.final_state.coord[0]) == pytest.approx(1.4, abs=0.02)
    assert res.pot_max == pytest.approx(0.03, abs=0.005)
    with pytest.raises(ValueError):
        attractive_path_sampling(AttractivePot(start, System(coord=np.zeros((3, 3)), atom_type=["H"] * 3), PathPara(), tmp_path), DoubleWell(0.03))
