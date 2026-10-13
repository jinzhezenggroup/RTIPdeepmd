import numpy as np
import pytest

from rtipdeepmd.cli import DEFAULT_XYZ_DIR, build_initial_state
from rtipdeepmd.constants import ANGSTROM_TO_BOHR, MASSUNIT, atomic_mass, atomic_radius
from rtipdeepmd.params import Para
from rtipdeepmd.system import (
    System,
    get_adj_mat,
    judge_adj_of_mol,
    judge_variation_of_bonding,
    rand_rot,
    split_into_mol,
)

A = ANGSTROM_TO_BOHR


def water(shift=(0.0, 0.0, 0.0)):
    coord = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]) + np.asarray(shift)
    return System(coord=coord * A, atom_type=["O", "H", "H"])


def test_masses_and_radii_match_rust_tables():
    assert atomic_mass("H") == pytest.approx(1.00794 * MASSUNIT)
    assert atomic_mass("H") == pytest.approx(1837.36, rel=1e-4)
    assert atomic_mass("Ca") == pytest.approx(40.078 * MASSUNIT)
    assert atomic_radius("C") == pytest.approx(0.76 * A)
    assert atomic_radius("O") == pytest.approx(0.66 * A)
    assert atomic_radius("Ca") == pytest.approx(1.76 * A)


def test_read_xyz_converts_to_bohr():
    s = System.read_xyz(DEFAULT_XYZ_DIR / "H2O.xyz")
    assert s.atom_type == ["O", "H", "H"]
    d = np.linalg.norm(s.coord[0] - s.coord[1]) / A
    assert 0.9 < d < 1.0


def test_rand_rot_is_proper_rotation():
    rot = rand_rot(np.random.default_rng(1))
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rot) == pytest.approx(1.0)


def test_split_and_adjacency():
    s = System(
        coord=np.concatenate([water().coord, water((4.0, 0.0, 0.0)).coord]),
        atom_type=["O", "H", "H", "O", "H", "H"],
    )
    radii = s.get_atomic_radii()
    dist = s.get_dist_mat()
    assert split_into_mol(radii, dist, 1.25) == [[0, 1, 2], [3, 4, 5]]

    adj = get_adj_mat(s.atom_type, radii, dist, 1.25, ignored_pair=[])
    assert adj[0, 1] == 1 and adj[0, 3] == -1 and adj[1, 2] == -1 and adj[0, 0] == 0
    adj_ignored = get_adj_mat(s.atom_type, radii, dist, 1.25, ignored_pair=Para().ignored_pair)
    assert adj_ignored[0, 1] == 0 and adj_ignored[3, 1] == 0  # H-O pairs are ignored
    assert adj_ignored[1, 4] == -1                              # H-H pairs are not


def test_judge_variation_of_bonding():
    s = water()
    radii = s.get_atomic_radii()
    adj = get_adj_mat(s.atom_type, radii, s.get_dist_mat(), 1.25, ignored_pair=[])
    assert not judge_variation_of_bonding(radii, s.get_dist_mat(), adj, 1.0, 1.6)

    stretched = s.clone()
    stretched.coord[1, 0] = 1.6 * (0.31 + 0.66) * A + 0.01   # O-H beyond 1.6 (r_O + r_H)
    assert judge_variation_of_bonding(radii, stretched.get_dist_mat(), adj, 1.0, 1.6)

    slightly = s.clone()
    slightly.coord[1, 0] = 1.5 * (0.31 + 0.66) * A            # stretched but not broken
    assert not judge_variation_of_bonding(radii, slightly.get_dist_mat(), adj, 1.0, 1.6)

    formed = s.clone()
    formed.coord[2] = formed.coord[1] + np.array([0.6, 0.0, 0.0]) * A   # H-H at 0.6 A < 1.0 (r_H + r_H)
    assert judge_variation_of_bonding(radii, formed.get_dist_mat(), adj, 1.0, 1.6)


def test_judge_adj_of_mol():
    far = System(coord=np.concatenate([water().coord, water((5.0, 0.0, 0.0)).coord]), atom_type=["O", "H", "H"] * 2)
    radii = far.get_atomic_radii()
    adj = get_adj_mat(far.atom_type, radii, far.get_dist_mat(), 1.25, ignored_pair=[])
    mol_index = split_into_mol(radii, far.get_dist_mat(), 1.25)
    assert not judge_adj_of_mol(mol_index, radii, adj, far.get_dist_mat(), 1.2)

    near = far.clone()
    near.coord[3:] -= np.array([5.0 - 1.1, 0.0, 0.0]) * A   # O...O at 1.1 A < 1.2 (r_O + r_O)
    assert judge_adj_of_mol(mol_index, radii, adj, near.get_dist_mat(), 1.2)

    assert judge_adj_of_mol([[0, 1, 2]], radii[:3], adj[:3, :3], far.get_dist_mat()[:3, :3], 1.2)


def test_recipe_box_matches_rust_composition():
    s = build_initial_state(DEFAULT_XYZ_DIR, 3.5, np.random.default_rng(3))
    assert s.natom == 54
    assert sorted(set(s.atom_type)) == ["C", "Ca", "H", "O"]
    edge = s.cell[0, 0]
    assert np.all(s.coord >= 0.0) and np.all(s.coord <= edge)
    mols = split_into_mol(s.get_atomic_radii(), s.get_dist_mat(), 1.25)
    assert len(mols) == 13   # Ca, 2 OH, pentose, 3 CH2O, 6 H2O


def test_cli_molecule_and_ignored_pairs(tmp_path):
    from rtipdeepmd.cli import main, parse_ignored_pairs, parse_molecules
    import json

    assert parse_molecules(["a.xyz:3", "b.xyz", "c:d.xyz:2"]) == [("a.xyz", 3), ("b.xyz", 1), ("c:d.xyz", 2)]
    assert parse_ignored_pairs("none") == []
    assert parse_ignored_pairs("H-O, C-Ca") == [("H", "O"), ("C", "Ca")]
    assert parse_ignored_pairs(None) is None
    with pytest.raises(ValueError):
        parse_ignored_pairs("HO")

    out = tmp_path / "run"
    h2o = str(DEFAULT_XYZ_DIR / "H2O.xyz")
    ch2o = str(DEFAULT_XYZ_DIR / "CH2O.xyz")
    main(["--dry-run", "--seed", "2", "--output-dir", str(out), "--molecule", f"{h2o}:2", "--molecule", ch2o,
          "--ignored-pairs", "none"])
    s = System.read_xyz(out / "supercell.xyz")
    assert s.natom == 10
    info = json.loads((out / "run.json").read_text())
    assert info["para"]["ignored_pair"] == []
