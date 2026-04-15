from __future__ import annotations

import numpy as np

from ..common.constants import AminoAcid
from .protein import Fragment

_FD_EPS = 1.0e-6

AMINO_ACID_TO_EQUI_INDICES: dict[AminoAcid, list[list[int]]] = {
    AminoAcid.GLY: [],
    AminoAcid.ALA: [[4, 5, 6]],
    AminoAcid.VAL: [[5, 9], [6, 7, 8, 10, 11, 12]],
    AminoAcid.LEU: [[4, 5], [8, 12], [9, 10, 11, 13, 14, 15]],
    AminoAcid.ILE: [[6, 7, 8], [10, 11], [13, 14, 15]],
    AminoAcid.SER: [[4, 5]],
    AminoAcid.THR: [[6, 7, 8]],
    AminoAcid.ASP: [[4, 5], [7, 8]],
    AminoAcid.ASH: [[4, 5], [7, 8]],
    AminoAcid.ASN: [[4, 5], [9, 10]],
    AminoAcid.GLU: [[4, 5], [7, 8], [10, 11]],
    AminoAcid.GLH: [[4, 5], [7, 8], [10, 11]],
    AminoAcid.GLN: [[4, 5], [7, 8], [12, 13]],
    AminoAcid.LYS: [[4, 5], [7, 8], [10, 11], [13, 14], [16, 17, 18]],
    AminoAcid.LYN: [[4, 5], [7, 8], [10, 11], [13, 14], [16, 17]],
    AminoAcid.ARG: [[4, 5], [7, 8], [10, 11], [15, 18], [16, 17, 19, 20]],
    AminoAcid.ARN: [[4, 5], [7, 8], [10, 11], [15, 18], [16, 17, 19]],
    AminoAcid.CYS: [[4, 5]],
    AminoAcid.CYX: [[4, 5]],
    AminoAcid.MET: [[4, 5], [7, 8], [11, 12, 13]],
    AminoAcid.HID: [[4, 5]],
    AminoAcid.HIE: [[4, 5]],
    AminoAcid.HIP: [[4, 5]],
    AminoAcid.PHE: [[4, 5], [7, 15], [8, 16], [9, 13], [10, 14]],
    AminoAcid.TYR: [[4, 5], [7, 16], [8, 17], [9, 14], [10, 15]],
    AminoAcid.TRP: [[4, 5]],
    AminoAcid.PRO: [[2, 3], [5, 6], [8, 9]],
}



def get_intrafragment_descriptor(fragment: Fragment) -> tuple[list[float], np.ndarray] | None:
    if fragment.fragment_type.kind != "Residue":
        return None

    coord = np.asarray(fragment.coord, dtype=float)
    natom = coord.shape[0]
    if natom < 3:
        return None

    amino_acid = fragment.fragment_type.value
    if not isinstance(amino_acid, AminoAcid):
        return None

    if amino_acid == AminoAcid.PRO:
        index = 10
        index2 = 7
    else:
        index = 2
        index2 = 3

    if index >= natom:
        index = min(2, natom - 1)
    if index2 >= max(natom - 1, 1):
        index2 = min(3, max(natom - 2, 0))

    equi_indices = AMINO_ACID_TO_EQUI_INDICES.get(amino_acid, [])
    return get_intraresidue_descriptor(coord, index, index2, equi_indices)



def _descriptor_only(coord: np.ndarray, index: int, index2: int, equi_indices: list[list[int]]) -> np.ndarray:
    coord_tran, _grad = center_at_c_alpha(coord, index)

    index1 = max(coord.shape[0] - 3, 0)
    index1 = min(index1, coord_tran.shape[0] - 1)
    index2 = min(max(index2, 0), coord_tran.shape[0] - 1)

    s = np.array(
        [
            [coord_tran[0, 0], coord_tran[index1, 0], coord_tran[index2, 0]],
            [coord_tran[0, 1], coord_tran[index1, 1], coord_tran[index2, 1]],
            [coord_tran[0, 2], coord_tran[index1, 2], coord_tran[index2, 2]],
        ],
        dtype=float,
    )

    c, d_c_s = schmidt_orthonormalization(s)
    coord_rot, _grad2 = rotate_around_c_alpha(coord_tran, index2, c, d_c_s)
    _grad3 = average_equivalent_atoms(coord_rot, equi_indices)
    descriptor, _grad4 = convert_into_descriptor(coord_rot)
    return np.asarray(descriptor, dtype=float)



def get_intraresidue_descriptor(
    coord: np.ndarray,
    index: int,
    index2: int,
    equi_indices: list[list[int]],
) -> tuple[list[float], np.ndarray]:
    coord = np.asarray(coord, dtype=float)
    descriptor = _descriptor_only(coord, index, index2, equi_indices)

    n_in = coord.shape[0] * 3
    n_out = descriptor.shape[0]
    gradient = np.zeros((n_in, n_out), dtype=float)

    for atom in range(coord.shape[0]):
        for axis in range(3):
            row = atom * 3 + axis
            cp = coord.copy()
            cm = coord.copy()
            cp[atom, axis] += _FD_EPS
            cm[atom, axis] -= _FD_EPS
            dp = _descriptor_only(cp, index, index2, equi_indices)
            dm = _descriptor_only(cm, index, index2, equi_indices)
            gradient[row] = (dp - dm) / (2.0 * _FD_EPS)

    return descriptor.tolist(), gradient



def center_at_c_alpha(coord: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    natom = coord.shape[0]
    if not (0 <= index < natom):
        raise ValueError("index out of range")

    coord_tran = np.zeros((natom - 1, 3), dtype=float)
    for i in range(natom):
        if i < index:
            coord_tran[i] = coord[i] - coord[index]
        elif i > index:
            coord_tran[i - 1] = coord[i] - coord[index]

    gradient = center_at_c_alpha_gradient(natom, index)
    return coord_tran, gradient



def center_at_c_alpha_gradient(natom: int, index: int) -> np.ndarray:
    grad = np.zeros((3 * natom, 3 * natom - 3), dtype=float)
    for i in range(natom - 1):
        col = 3 * i
        if i < index:
            grad[col, col] = 1.0
            grad[col + 1, col + 1] = 1.0
            grad[col + 2, col + 2] = 1.0
        else:
            grad[col + 3, col] = 1.0
            grad[col + 4, col + 1] = 1.0
            grad[col + 5, col + 2] = 1.0

        grad[3 * index, col] = -1.0
        grad[3 * index + 1, col + 1] = -1.0
        grad[3 * index + 2, col + 2] = -1.0

    return grad



def schmidt_orthonormalization(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q, _r = np.linalg.qr(s)
    gradient = np.zeros((3, 3, 3, 3), dtype=float)
    return q, gradient



def rotate_around_c_alpha(
    coord_tran: np.ndarray,
    index2: int,
    c: np.ndarray,
    d_c_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    del index2
    del d_c_s

    coord_rot = coord_tran @ c

    n_in = coord_tran.shape[0] * 3
    n_out = coord_rot.shape[0] * 3
    grad = np.zeros((n_in, n_out), dtype=float)

    for atom in range(coord_tran.shape[0]):
        for i in range(3):
            for j in range(3):
                grad[atom * 3 + i, atom * 3 + j] = c[i, j]

    return coord_rot, grad



def average_equivalent_atoms(coord_rot: np.ndarray, equi_indices: list[list[int]]) -> np.ndarray:
    n = coord_rot.shape[0]
    grad = np.eye(3 * n, dtype=float)

    for group in equi_indices:
        if not group:
            continue

        group = [idx for idx in group if 0 <= idx < n]
        if not group:
            continue

        mean_xyz = coord_rot[group].mean(axis=0)
        for idx in group:
            coord_rot[idx] = mean_xyz

        w = 1.0 / len(group)
        for idx in group:
            row0 = idx * 3
            for axis in range(3):
                grad[row0 + axis, :] = 0.0
                for src in group:
                    grad[row0 + axis, src * 3 + axis] = w

    return grad



def convert_into_descriptor(coord_rot: np.ndarray) -> tuple[list[float], np.ndarray]:
    descriptor = coord_rot.reshape(-1).tolist()
    gradient = convert_into_descriptor_gradient(coord_rot)
    return descriptor, gradient



def convert_into_descriptor_gradient(coord_rot: np.ndarray) -> np.ndarray:
    n = coord_rot.size
    return np.eye(n, dtype=float)
