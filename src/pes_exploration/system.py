from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from ..common.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, Element
from ..matrix import rand_rot


@dataclass
class System:
    natom: int
    coord: jnp.ndarray
    cell: jnp.ndarray | None = None
    atom_type: list[Element] | None = None
    atom_add_pot: list[int] | None = None
    mutable: jnp.ndarray | None = None
    pot: float = 0.0

    def __post_init__(self) -> None:
        self.coord = jnp.asarray(self.coord, dtype=jnp.float64)
        if self.coord.ndim != 2 or self.coord.shape[1] != 3:
            raise ValueError("coord must have shape (natom, 3)")
        self.natom = int(self.coord.shape[0])

        if self.cell is not None:
            self.cell = jnp.asarray(self.cell, dtype=jnp.float64)

        if self.mutable is not None:
            self.mutable = jnp.asarray(self.mutable)

    def clone(self) -> "System":
        return copy.deepcopy(self)

    @classmethod
    def from_deepmd(cls, force_env: object) -> "System":
        from ..external import deepmd

        nparticle = int(deepmd.deepmd_get_nparticle(force_env))
        natom = int(nparticle)
        pos = deepmd.deepmd_get_positions(force_env, natom * 3)
        coord = np.asarray(pos, dtype=float).reshape(natom, 3)
        pot = float(deepmd.deepmd_get_potential_energy(force_env))

        return cls(
            natom=natom,
            coord=jnp.asarray(coord, dtype=jnp.float64),
            cell=None,
            atom_type=None,
            atom_add_pot=None,
            mutable=None,
            pot=pot,
        )

    def to_deepmd(self, force_env: object) -> None:
        from ..external import deepmd

        n_el = int(self.natom * 3)
        pos = np.asarray(self.coord, dtype=float).reshape(-1).tolist()
        deepmd.deepmd_set_positions(force_env, pos, n_el)

    # 兼容旧接口名（历史上绑定 CP2K）
    @classmethod
    def from_cp2k(cls, force_env: object) -> "System":
        return cls.from_deepmd(force_env)

    def to_cp2k(self, force_env: object) -> None:
        self.to_deepmd(force_env)

    def write_pdb(self, filename: str, create_new_file: bool, step: int) -> None:
        mode = "w" if create_new_file else "a"
        with open(filename, mode, encoding="utf-8") as pdb:
            if create_new_file:
                pdb.write("TITLE     PDB file created by RTIP-JAX\n")

            pdb.write(f"REMARK    , Step = {step:8d}, E = {self.pot:15.8f}\n")

            coord_ang = np.asarray(self.coord) * BOHR_TO_ANGSTROM
            if self.atom_type is None:
                atom_labels = ["XX"] * self.natom
            else:
                atom_labels = [atom.value for atom in self.atom_type]

            for i in range(self.natom):
                pdb.write(
                    "ATOM  {:>5} {:>4}              {:8.3f}{:8.3f}{:8.3f}  1.00  0.00          {:>2}\n".format(
                        (i + 1) % 100000,
                        atom_labels[i],
                        coord_ang[i, 0],
                        coord_ang[i, 1],
                        coord_ang[i, 2],
                        atom_labels[i],
                    )
                )
            pdb.write("END\n")

    @classmethod
    def read_xyz(cls, filename: str) -> "System":
        with open(filename, "r", encoding="utf-8") as f:
            raw_lines = [line.rstrip("\n") for line in f]

        if len(raw_lines) < 2:
            raise ValueError(f"Invalid XYZ file: {filename}")

        natom = int(raw_lines[0].strip())

        # Keep the second line as the comment line even if it is empty.
        body: list[str] = []
        for line in raw_lines[2:]:
            if not line.strip():
                continue
            body.append(line)
            if len(body) == natom:
                break

        if len(body) != natom:
            raise ValueError(f"Invalid XYZ body: {filename}")

        atom_type: list[Element] = []
        coord = np.zeros((natom, 3), dtype=float)
        for i, line in enumerate(body):
            cols = line.split()
            if len(cols) < 4:
                raise ValueError(f"Invalid XYZ line: {line}")
            atom_type.append(Element.from_str(cols[0]))
            coord[i, 0] = float(cols[1])
            coord[i, 1] = float(cols[2])
            coord[i, 2] = float(cols[3])

        coord *= ANGSTROM_TO_BOHR
        return cls(natom=natom, coord=jnp.asarray(coord), atom_type=atom_type)

    def write_xyz(self, filename: str, create_new_file: bool, step: int) -> None:
        mode = "w" if create_new_file else "a"
        with open(filename, mode, encoding="utf-8") as xyz:
            xyz.write(f"{self.natom}\n")
            xyz.write(f"step={step} energy={self.pot:.8f}\n")

            coord_ang = np.asarray(self.coord) * BOHR_TO_ANGSTROM
            if self.atom_type is None:
                atom_labels = ["XX"] * self.natom
            else:
                atom_labels = [atom.value for atom in self.atom_type]

            for i in range(self.natom):
                xyz.write(
                    f"{atom_labels[i]:>2} {coord_ang[i, 0]:15.8f} {coord_ang[i, 1]:15.8f} {coord_ang[i, 2]:15.8f}\n"
                )

    @staticmethod
    def super_cell(systems: list["System"], mul: list[int], size: list[float] | tuple[float, float, float], min_dist: float) -> "System":
        if len(systems) != len(mul):
            raise ValueError("systems and mul must have the same length")
        if len(size) != 3:
            raise ValueError("size must be a length-3 sequence in Angstrom")

        min_dist_bohr = float(min_dist) * ANGSTROM_TO_BOHR
        size_bohr = np.asarray(size, dtype=float) * ANGSTROM_TO_BOHR
        cell = np.diag(size_bohr)

        total_natom = 0
        for i in range(len(systems)):
            total_natom += int(systems[i].natom) * int(mul[i])
        if total_natom <= 0:
            raise ValueError("No atoms requested in super_cell construction")

        coord = np.zeros((total_natom, 3), dtype=float)
        cursor = 0

        has_all_atom_types = True
        for i in range(len(systems)):
            if mul[i] > 0 and systems[i].atom_type is None:
                has_all_atom_types = False
                break

        atom_type: list[Element] = []

        for i in range(len(systems)):
            for _ in range(int(mul[i])):
                base_coord = np.asarray(systems[i].coord, dtype=float)
                suitable = False
                candidate = base_coord.copy()

                for _trial in range(1000):
                    coord_mol = base_coord.copy()
                    coord_mol -= coord_mol.mean(axis=0, keepdims=True)

                    rot = np.asarray(rand_rot(), dtype=float)
                    coord_mol = coord_mol @ rot

                    tran = np.array(
                        [
                            np.random.uniform(min_dist_bohr, cell[0, 0] - min_dist_bohr),
                            np.random.uniform(min_dist_bohr, cell[1, 1] - min_dist_bohr),
                            np.random.uniform(min_dist_bohr, cell[2, 2] - min_dist_bohr),
                        ],
                        dtype=float,
                    )
                    coord_mol += tran

                    out_of_cell = (
                        np.any(coord_mol[:, 0] < 0.0)
                        or np.any(coord_mol[:, 0] > cell[0, 0])
                        or np.any(coord_mol[:, 1] < 0.0)
                        or np.any(coord_mol[:, 1] > cell[1, 1])
                        or np.any(coord_mol[:, 2] < 0.0)
                        or np.any(coord_mol[:, 2] > cell[2, 2])
                    )
                    if out_of_cell:
                        continue

                    too_close = False
                    if cursor > 0:
                        for k in range(coord_mol.shape[0]):
                            dr = coord[:cursor, :] - coord_mol[k, :]
                            if np.any(np.einsum("ij,ij->i", dr, dr) < (min_dist_bohr * min_dist_bohr)):
                                too_close = True
                                break
                    if too_close:
                        continue

                    suitable = True
                    candidate = coord_mol
                    break

                if not suitable:
                    raise ValueError("The constructed simulation cell is too small. Please increase the size of the cell.")

                nrow = candidate.shape[0]
                coord[cursor : cursor + nrow, :] = candidate

                if has_all_atom_types and systems[i].atom_type is not None:
                    atom_type.extend(systems[i].atom_type)

                cursor += nrow

        return System(
            natom=cursor,
            coord=jnp.asarray(coord[:cursor, :], dtype=jnp.float64),
            cell=jnp.asarray(cell, dtype=jnp.float64),
            atom_type=atom_type if has_all_atom_types else None,
            atom_add_pot=None,
            mutable=None,
            pot=0.0,
        )

    @staticmethod
    def auto_super_cell(systems: list["System"], mul: list[int], min_dist: float) -> "System":
        last_error = ""
        min_size = int(np.ceil(float(min_dist)))

        for i in range(2 * min_size + 1, 2 * min_size + 100):
            size = [float(i), float(i), float(i)]
            try:
                return System.super_cell(systems, mul, size, min_dist)
            except ValueError as exc:
                last_error = str(exc)

        raise ValueError(last_error or "Failed to build super cell automatically.")

    def get_atomic_radii(self) -> list[float]:
        if self.atom_type is None:
            return [1.5 * ANGSTROM_TO_BOHR for _ in range(self.natom)]
        return [atom.get_atomic_radius() for atom in self.atom_type]

    def get_dist_mat(self) -> jnp.ndarray:
        diff = self.coord[:, None, :] - self.coord[None, :, :]
        return jnp.linalg.norm(diff, axis=2)

    @staticmethod
    def split_into_mol(atomic_radii: list[float], dist_mat: jnp.ndarray, transition_multiple: float) -> list[list[int]]:
        d = np.asarray(dist_mat)
        natom = d.shape[0]
        visited = [False] * natom
        mol_index: list[list[int]] = []

        for start in range(natom):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            comp: list[int] = []

            while stack:
                i = stack.pop()
                comp.append(i)
                for j in range(natom):
                    if visited[j] or i == j:
                        continue
                    cutoff = transition_multiple * (atomic_radii[i] + atomic_radii[j])
                    if d[i, j] <= cutoff:
                        visited[j] = True
                        stack.append(j)

            mol_index.append(sorted(comp))

        return mol_index

    @staticmethod
    def judge_adj_of_mol(
        mol_index: list[list[int]],
        atomic_radii: list[float],
        adj_mat: jnp.ndarray,
        dist_mat: jnp.ndarray,
        multiple: float,
    ) -> bool:
        adj = np.asarray(adj_mat)
        dist = np.asarray(dist_mat)

        for a in range(len(mol_index)):
            for b in range(a + 1, len(mol_index)):
                connected = False
                for i in mol_index[a]:
                    for j in mol_index[b]:
                        if adj[i, j] == 1:
                            connected = True
                            break
                        if dist[i, j] <= multiple * (atomic_radii[i] + atomic_radii[j]):
                            connected = True
                            break
                    if connected:
                        break
                if connected:
                    return True
        return False

    @staticmethod
    def get_adj_mat(
        atomic_type: list[Element],
        atomic_radii: list[float],
        dist_mat: jnp.ndarray,
        transition_multiple: float,
        ignored_pair: list[tuple[Element, Element]],
    ) -> jnp.ndarray:
        natom = len(atomic_type)
        dist = np.asarray(dist_mat)
        adj = np.zeros((natom, natom), dtype=np.int8)

        ignored = set(ignored_pair)
        ignored.update((b, a) for (a, b) in ignored_pair)

        for i in range(natom - 1):
            for j in range(i + 1, natom):
                if (atomic_type[i], atomic_type[j]) in ignored:
                    continue
                cutoff = transition_multiple * (atomic_radii[i] + atomic_radii[j])
                if dist[i, j] <= cutoff:
                    adj[i, j] = 1
                    adj[j, i] = 1

        return jnp.asarray(adj)

    @staticmethod
    def judge_variation_of_bonding(
        atomic_radii: list[float],
        dist_mat: jnp.ndarray,
        p: jnp.ndarray,
        bonded_multiple: float,
        unbonded_multiple: float,
    ) -> bool:
        dist = np.asarray(dist_mat)
        bonded = np.asarray(p)
        natom = dist.shape[0]

        for i in range(natom - 1):
            for j in range(i + 1, natom):
                ref = atomic_radii[i] + atomic_radii[j]
                if bonded[i, j] == 1 and dist[i, j] > bonded_multiple * ref:
                    return True
                if bonded[i, j] == 0 and dist[i, j] < unbonded_multiple * ref:
                    return True

        return False


def from_deepmd(force_env: object) -> System:
    return System.from_deepmd(force_env)


def from_cp2k(force_env: object) -> System:
    return System.from_deepmd(force_env)


def read_xyz(filename: str) -> System:
    return System.read_xyz(filename)
