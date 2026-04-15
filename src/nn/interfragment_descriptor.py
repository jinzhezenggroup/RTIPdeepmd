from __future__ import annotations

import math
import numpy as np

from ..common.constants import ANGSTROM_TO_BOHR
from .protein import ProteinSystem

RCUT = 5.0 * ANGSTROM_TO_BOHR
NATOM_WITHIN_2RCUT = 2000
NATOM_WITHIN_RCUT = 400

MU_Z = [1.0, 6.0, 7.0, 8.0, 16.0, 11.0, 20.0, 26.0, 30.0, 34.0, 42.0, 50.0]
N_DENSE = 5

SIGMA_Z_DENSE = 1.0
MU_D_DENSE = [0.0, RCUT / 3.0, RCUT * 2.0 / 3.0, RCUT]
SIGMA_D_DENSE = RCUT / 3.0
MU_THETA_DENSE = [-1.0, -0.5, 0.0, 0.5, 1.0]
SIGMA_THETA_DENSE = 0.5

SIGMA_Z_SPARSE = 2.0
MU_D_SPARSE = [RCUT / 4.0, RCUT * 3.0 / 4.0]
SIGMA_D_SPARSE = RCUT / 2.0
MU_THETA_SPARSE = [-1.0, 0.0, 1.0]
SIGMA_THETA_SPARSE = 1.0

N_DES = (
    (len(MU_Z) * len(MU_D_SPARSE)) * (len(MU_Z) * len(MU_D_SPARSE) + 1) * len(MU_THETA_SPARSE)
    + (N_DENSE * len(MU_D_DENSE)) * (N_DENSE * len(MU_D_DENSE) + 1) * len(MU_THETA_DENSE)
    - (N_DENSE * len(MU_D_SPARSE)) * (N_DENSE * len(MU_D_SPARSE) + 1) * len(MU_THETA_SPARSE)
) // 2

_FD_EPS = 1.0e-6
_DENSE_Z = [1.0, 6.0, 7.0, 8.0, 16.0]
_DENSE_Z_SET = set(_DENSE_Z)
_NON_DENSE_Z_INDICES = [i for i, z in enumerate(MU_Z) if z not in _DENSE_Z_SET]
_NON_DENSE_CHANNEL_INDICES = [
    z_idx * len(MU_D_SPARSE) + d_idx
    for z_idx in _NON_DENSE_Z_INDICES
    for d_idx in range(len(MU_D_SPARSE))
]



def fcut(r: float, rcut: float = RCUT) -> float:
    if r >= rcut:
        return 0.0
    if r <= 0.0:
        return 1.0
    return 0.5 * (math.cos(math.pi * r / rcut) + 1.0)



def dfcut(r: float, rcut: float = RCUT) -> float:
    if r >= rcut or r <= 0.0:
        return 0.0
    return -0.5 * math.pi / rcut * math.sin(math.pi * r / rcut)



def gaussian(x: float, mu: float, sigma: float) -> float:
    sigma = max(sigma, 1.0e-12)
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z)



def dgaussian(x: float, mu: float, sigma: float) -> float:
    sigma = max(sigma, 1.0e-12)
    return -(x - mu) / (sigma * sigma) * gaussian(x, mu, sigma)



def _cell_shifts(cell: np.ndarray | None) -> np.ndarray:
    if cell is None:
        return np.zeros((1, 3), dtype=float)

    shifts: list[np.ndarray] = []
    for i in (0, -1, 1):
        for j in (0, -1, 1):
            for k in (0, -1, 1):
                shifts.append(i * cell[0] + j * cell[1] + k * cell[2])
    return np.asarray(shifts, dtype=float)



def update_neighbor_list(s: ProteinSystem, index: int) -> None:
    frag = s.fragment[index]
    if frag.atom_list is None:
        frag.neighbor_list = []
        return

    in_frag = set(frag.atom_list)
    coord = np.asarray(s.coord, dtype=float)
    cell = None if s.cell is None else np.asarray(s.cell, dtype=float)
    shifts = _cell_shifts(cell)

    frag_coord = coord[frag.atom_list]
    frag_images = (frag_coord[None, :, :] + shifts[:, None, :]).reshape(-1, 3)

    rnei2 = (2.0 * RCUT) ** 2
    neigh: list[int] = []

    for i in range(s.natom):
        if i in in_frag:
            continue

        pi = coord[i]
        dr = frag_images - pi
        if float(np.min(np.einsum("ij,ij->i", dr, dr))) < rnei2:
            neigh.append(i)

    frag.neighbor_list = neigh



def truncate_cluster(s: ProteinSystem, index: int) -> tuple[np.ndarray, list[int], list[int]]:
    frag = s.fragment[index]
    if frag.atom_list is None:
        raise ValueError("fragment.atom_list is None")

    frag_indices = list(frag.atom_list)
    coord = np.asarray(s.coord, dtype=float)
    cell = None if s.cell is None else np.asarray(s.cell, dtype=float)
    shifts = _cell_shifts(cell)

    coord_within: list[np.ndarray] = [coord[i] for i in frag_indices]
    list_within: list[int] = list(frag_indices)

    candidates = frag.neighbor_list if frag.neighbor_list is not None else list(range(s.natom))
    in_frag = set(frag_indices)
    rcut2 = RCUT * RCUT

    frag_coord = coord[frag_indices]
    frag_images = (frag_coord[None, :, :] + shifts[:, None, :]).reshape(-1, 3)

    for idx_atom in candidates:
        if idx_atom in in_frag:
            continue

        pi = coord[idx_atom]
        dr = frag_images - pi
        if float(np.min(np.einsum("ij,ij->i", dr, dr))) < rcut2:
            coord_within.append(pi)
            list_within.append(idx_atom)

    atomic_numbers = [int(s.atom_type[i].get_atomic_number()) for i in list_within]
    return np.asarray(coord_within, dtype=float), list_within, atomic_numbers



def _distance_features(
    coord_within_rcut: np.ndarray,
    atomic_number_within_rcut: list[int],
    natom_fragment: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ncut = max(coord_within_rcut.shape[0] - 1, 0)
    sparse_radial = np.zeros((ncut, len(MU_D_SPARSE)), dtype=float)
    dense_radial = np.zeros((ncut, len(MU_D_DENSE)), dtype=float)
    z_full = np.zeros((ncut, len(MU_Z)), dtype=float)
    z_dense = np.zeros((ncut, len(_DENSE_Z)), dtype=float)
    vec_center = np.zeros((ncut, 3), dtype=float)

    if ncut == 0:
        return sparse_radial, dense_radial, z_full, z_dense, vec_center

    center = coord_within_rcut[0]
    for i in range(ncut):
        atom_idx = i + 1
        zi = float(atomic_number_within_rcut[atom_idx])

        z_full[i] = [gaussian(zi, mu, SIGMA_Z_SPARSE) for mu in MU_Z]
        z_dense[i] = [gaussian(zi, mu, SIGMA_Z_DENSE) for mu in _DENSE_Z]

        vec = coord_within_rcut[atom_idx] - center
        vec_center[i] = vec

        for k in range(min(natom_fragment, coord_within_rcut.shape[0])):
            if atom_idx == k:
                continue

            dr = coord_within_rcut[atom_idx] - coord_within_rcut[k]
            d = float(np.linalg.norm(dr))
            if d <= 1.0e-12:
                continue

            w = fcut(d)
            for m, mu in enumerate(MU_D_SPARSE):
                sparse_radial[i, m] += w * gaussian(d, mu, SIGMA_D_SPARSE)
            for m, mu in enumerate(MU_D_DENSE):
                dense_radial[i, m] += w * gaussian(d, mu, SIGMA_D_DENSE)

    return sparse_radial, dense_radial, z_full, z_dense, vec_center



def _pair_weight_matrix(vec_center: np.ndarray, mus: list[float], sigma: float) -> list[np.ndarray]:
    ncut = vec_center.shape[0]
    out: list[np.ndarray] = []
    if ncut == 0:
        for _ in mus:
            out.append(np.zeros((0, 0), dtype=float))
        return out

    norms = np.linalg.norm(vec_center, axis=1)
    norm_safe = np.where(norms > 1.0e-12, norms, 1.0)
    unit = vec_center / norm_safe[:, None]
    cos_mat = np.clip(unit @ unit.T, -1.0, 1.0)

    for mu in mus:
        g = np.vectorize(lambda x: gaussian(float(x), mu, sigma))(cos_mat)
        out.append(np.triu(g))
    return out



def _upper_tri_values(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return np.zeros((0,), dtype=float)
    i, j = np.triu_indices(mat.shape[0])
    return mat[i, j]



def _descriptor_from_cluster(
    coord_within_rcut: np.ndarray,
    atomic_number_within_rcut: list[int],
    natom_fragment: int,
) -> np.ndarray:
    ncut = max(coord_within_rcut.shape[0] - 1, 0)
    if ncut == 0:
        return np.zeros((N_DES,), dtype=float)

    sparse_radial, dense_radial, z_full, z_dense, vec_center = _distance_features(
        coord_within_rcut,
        atomic_number_within_rcut,
        natom_fragment,
    )

    sparse_all = np.einsum("iz,id->izd", z_full, sparse_radial).reshape(ncut, len(MU_Z) * len(MU_D_SPARSE))
    sparse_nd = sparse_all[:, _NON_DENSE_CHANNEL_INDICES]
    sparse_dense = np.einsum("iz,id->izd", z_dense, sparse_radial).reshape(ncut, len(_DENSE_Z) * len(MU_D_SPARSE))
    dense_dense = np.einsum("iz,id->izd", z_dense, dense_radial).reshape(ncut, len(_DENSE_Z) * len(MU_D_DENSE))

    theta_sparse_weights = _pair_weight_matrix(vec_center, MU_THETA_SPARSE, SIGMA_THETA_SPARSE)
    theta_dense_weights = _pair_weight_matrix(vec_center, MU_THETA_DENSE, SIGMA_THETA_DENSE)

    descriptor_parts: list[np.ndarray] = []

    for w in theta_sparse_weights:
        mm_nd = sparse_nd.T @ w @ sparse_nd
        mm_cross = sparse_nd.T @ w @ sparse_dense
        descriptor_parts.append(_upper_tri_values(mm_nd))
        descriptor_parts.append(mm_cross.reshape(-1))

    for w in theta_dense_weights:
        mm_dense = dense_dense.T @ w @ dense_dense
        descriptor_parts.append(_upper_tri_values(mm_dense))

    desc = np.concatenate(descriptor_parts, axis=0) if descriptor_parts else np.zeros((0,), dtype=float)
    if desc.shape[0] != N_DES:
        raise ValueError(f"Descriptor length mismatch: expected {N_DES}, got {desc.shape[0]}")
    return desc



def get_interfragment_descriptor(
    coord_within_rcut: np.ndarray,
    atomic_number_within_rcut: list[int],
    natom_fragment: int,
) -> tuple[list[float], np.ndarray]:
    coord = np.asarray(coord_within_rcut, dtype=float)
    base = _descriptor_from_cluster(coord, atomic_number_within_rcut, natom_fragment)

    n_atom = coord.shape[0]
    grad = np.zeros((n_atom * 3, N_DES), dtype=float)

    # Keep the same finite-difference contract as Rust-side training_data: descriptor +/- EPS * gradient.
    for i in range(n_atom):
        for axis in range(3):
            row = i * 3 + axis
            cp = coord.copy()
            cm = coord.copy()
            cp[i, axis] += _FD_EPS
            cm[i, axis] -= _FD_EPS
            dp = _descriptor_from_cluster(cp, atomic_number_within_rcut, natom_fragment)
            dm = _descriptor_from_cluster(cm, atomic_number_within_rcut, natom_fragment)
            grad[row] = (dp - dm) / (2.0 * _FD_EPS)

    return base.tolist(), grad
