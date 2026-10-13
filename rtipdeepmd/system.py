"""Molecular system container, structure I/O, supercell construction and bonding analysis.

Ported from the upstream Rust RTIP-MD ``src/pes_exploration/system.rs`` and
``src/matrix/mod.rs``. Coordinates are stored in Bohr.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import typing as t

import numpy as np

from .constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, atomic_mass, atomic_radius, check_element


@dataclass
class System:
    coord: np.ndarray                       # (natom, 3), Bohr
    atom_type: list[str]                    # element symbols
    cell: np.ndarray | None = None          # (3, 3), Bohr
    atom_add_pot: list[int] | None = None   # atoms that feel the RTIP; None means all atoms
    pot: float = 0.0                        # real potential energy, Hartree

    def __post_init__(self) -> None:
        self.coord = np.array(self.coord, dtype=np.float64)
        if self.coord.ndim != 2 or self.coord.shape[1] != 3:
            raise ValueError("coord must have shape (natom, 3)")
        self.atom_type = [check_element(e) for e in self.atom_type]
        if len(self.atom_type) != self.coord.shape[0]:
            raise ValueError("atom_type length must match the number of atoms")
        if self.cell is not None:
            self.cell = np.array(self.cell, dtype=np.float64).reshape(3, 3)

    @property
    def natom(self) -> int:
        return self.coord.shape[0]

    def clone(self) -> "System":
        return copy.deepcopy(self)

    def get_atomic_radii(self) -> np.ndarray:
        """Covalent radii in Bohr."""
        return np.array([atomic_radius(e) for e in self.atom_type])

    def get_atomic_masses(self) -> np.ndarray:
        """Atomic masses in electron masses."""
        return np.array([atomic_mass(e) for e in self.atom_type])

    def get_dist_mat(self) -> np.ndarray:
        """Interatomic distance matrix in Bohr (no periodic images, as in Rust)."""
        return dist_mat(self.coord)

    @classmethod
    def read_xyz(cls, filename: str | Path) -> "System":
        lines = Path(filename).read_text(encoding="utf-8").splitlines()
        try:
            natom = int(lines[0].strip())
            body = lines[2 : 2 + natom]
            atom_type = [line.split()[0] for line in body]
            coord = np.array([[float(x) for x in line.split()[1:4]] for line in body])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid XYZ file: {filename}") from exc
        if coord.shape != (natom, 3):
            raise ValueError(f"Invalid XYZ file: {filename}")
        return cls(coord=coord * ANGSTROM_TO_BOHR, atom_type=atom_type)

    def write_xyz(self, filename: str | Path, create_new_file: bool, step: int) -> None:
        with open(filename, "w" if create_new_file else "a", encoding="utf-8") as f:
            f.write(f"{self.natom:8d}\n")
            f.write(f"Step = {step:8d}, E = {self.pot:15.8f}\n")
            for e, (x, y, z) in zip(self.atom_type, self.coord * BOHR_TO_ANGSTROM):
                f.write(f"{e:>4} {x:20.10f} {y:20.10f} {z:20.10f}\n")

    def pdb_frame(self, step: int) -> str:
        lines = [f"REMARK    , Step = {step:8d}, E = {self.pot:15.8f}\n"]
        for i, (e, (x, y, z)) in enumerate(zip(self.atom_type, self.coord * BOHR_TO_ANGSTROM)):
            lines.append(f"ATOM  {(i + 1) % 100000:>5} {e:>4}              {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {e:>2}\n")
        lines.append("END\n")
        return "".join(lines)

    @staticmethod
    def super_cell(
        systems: list["System"],
        mul: list[int],
        size: float,
        min_dist: float,
        rng: np.random.Generator,
    ) -> "System":
        """Randomly place ``mul[i]`` copies of ``systems[i]`` into a cubic cell of edge ``size`` (Angstrom).

        Raises ``ValueError`` if a molecule cannot be placed within 1000 trials.
        """
        if len(systems) != len(mul):
            raise ValueError("systems and mul must have the same length")
        min_dist *= ANGSTROM_TO_BOHR
        edge = size * ANGSTROM_TO_BOHR
        coords: list[np.ndarray] = []
        atom_type: list[str] = []
        for s, n in zip(systems, mul):
            for _ in range(n):
                placed = np.concatenate(coords) if coords else np.zeros((0, 3))
                for _trial in range(1000):
                    coord_mol = (s.coord - s.coord.mean(axis=0)) @ rand_rot(rng)
                    coord_mol = coord_mol + rng.uniform(min_dist, edge - min_dist, size=3)
                    if np.any(coord_mol < 0.0) or np.any(coord_mol > edge):
                        continue
                    if len(placed) and np.min(dist_mat_between(coord_mol, placed)) < min_dist:
                        continue
                    break
                else:
                    raise ValueError("The constructed simulation cell is too small. Please increase the size of the cell.")
                coords.append(coord_mol)
                atom_type.extend(s.atom_type)
        if not coords:
            raise ValueError("No molecule was requested for the supercell")
        return System(coord=np.concatenate(coords), atom_type=atom_type, cell=np.eye(3) * edge)

    @staticmethod
    def auto_super_cell(
        systems: list["System"],
        mul: list[int],
        min_dist: float,
        rng: np.random.Generator,
    ) -> "System":
        """Try cubic cells of increasing integer edge (Angstrom) until all molecules fit."""
        lo = int(np.ceil(min_dist))
        error = ""
        for size in range(2 * lo + 1, 2 * lo + 100):
            try:
                return System.super_cell(systems, mul, float(size), min_dist, rng)
            except ValueError as exc:
                error = str(exc)
        raise ValueError(error)


def rand_rot(rng: np.random.Generator) -> np.ndarray:
    """Random rotation matrix from three uniform Euler angles, as Rust `matrix::rand_rot`."""
    alpha, beta, gamma = rng.uniform(0.0, 2.0 * np.pi, size=3)
    ca, sa, cb, sb, cg, sg = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta), np.cos(gamma), np.sin(gamma)
    return np.array(
        [
            [ca * cg - cb * sa * sg, -cb * cg * sa - ca * sg, sa * sb],
            [cg * sa + ca * cb * sg, ca * cb * cg - sa * sg, -ca * sb],
            [sb * sg, cg * sb, cb],
        ]
    )


def dist_mat(coord: np.ndarray) -> np.ndarray:
    return dist_mat_between(coord, coord)


def dist_mat_between(coord1: np.ndarray, coord2: np.ndarray) -> np.ndarray:
    diff = coord1[:, None, :] - coord2[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def split_into_mol(atomic_radii: np.ndarray, dist: np.ndarray, transition_multiple: float) -> list[list[int]]:
    """Split atoms into molecules: atoms closer than ``transition_multiple * (r_i + r_j)`` are connected."""
    connected = dist < (atomic_radii[:, None] + atomic_radii[None, :]) * transition_multiple
    untreated = list(range(len(atomic_radii)))
    mol_index: list[list[int]] = []
    while untreated:
        index = [untreated.pop(0)]
        n = 0
        while n < len(index):
            row = connected[index[n]]
            newly = [m for m in untreated if row[m]]
            index.extend(newly)
            untreated = [m for m in untreated if not row[m]]
            n += 1
        mol_index.append(index)
    return mol_index


def get_adj_mat(
    atom_type: list[str],
    atomic_radii: np.ndarray,
    dist: np.ndarray,
    transition_multiple: float,
    ignored_pair: t.Iterable[tuple[str, str]],
) -> np.ndarray:
    """Adjacency matrix: 1 bonded, -1 unbonded, 0 for the diagonal and ignored element pairs."""
    ignored = {frozenset(p) for p in ignored_pair}
    natom = len(atom_type)
    adj = np.where(dist < (atomic_radii[:, None] + atomic_radii[None, :]) * transition_multiple, 1, -1).astype(np.int8)
    for i in range(natom):
        adj[i, i] = 0
        for j in range(i + 1, natom):
            if frozenset((atom_type[i], atom_type[j])) in ignored:
                adj[i, j] = adj[j, i] = 0
    return adj


def judge_adj_of_mol(
    mol_index: list[list[int]],
    atomic_radii: np.ndarray,
    adj_mat: np.ndarray,
    dist: np.ndarray,
    multiple: float,
) -> bool:
    """True if two molecules have an originally unbonded atom pair closer than ``multiple * (r_i + r_j)``.

    A system consisting of a single molecule is always considered adjacent.
    """
    if len(mol_index) == 1:
        return True
    label = np.empty(len(atomic_radii), dtype=np.int64)
    for k, mol in enumerate(mol_index):
        label[mol] = k
    close = dist < (atomic_radii[:, None] + atomic_radii[None, :]) * multiple
    return bool(np.any((label[:, None] != label[None, :]) & (adj_mat == -1) & close))


def judge_variation_of_bonding(
    atomic_radii: np.ndarray,
    dist: np.ndarray,
    p: np.ndarray,
    bonded_multiple: float,
    unbonded_multiple: float,
) -> bool:
    """True if a bond in ``p`` broke (> ``unbonded_multiple``) or a non-bond formed (< ``bonded_multiple``)."""
    r = atomic_radii[:, None] + atomic_radii[None, :]
    broken = (p == 1) & (dist > r * unbonded_multiple)
    formed = (p == -1) & (dist < r * bonded_multiple)
    return bool(np.any(broken | formed))
