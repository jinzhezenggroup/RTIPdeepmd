"""Roto-translationally invariant (RTI) distance and the RTI potential (RTIP).

Ported from the upstream Rust RTIP-MD ``src/pes_exploration/rtip.rs``.

The RTI distances d_0 <= d_1 <= d_2 <= d_3 between two structures are the square roots of
the eigenvalues of the 4x4 quaternion matrix S. The RTIP is a weighted sum of Gaussians

    E = sum_k (w_k / W) * a * exp(-d_k^2 / (2 sigma^2)),   w_k = d_k^-7,   W = sum_k w_k,

so it is dominated by the minimum distance d_0 but stays smooth when eigenvalues cross.
"""

from __future__ import annotations

import numpy as np


def _centered(coord1: np.ndarray, coord2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c1 = np.asarray(coord1, dtype=np.float64)
    c2 = np.asarray(coord2, dtype=np.float64)
    if c1.shape != c2.shape or c1.ndim != 2 or c1.shape[1] != 3:
        raise ValueError("coord1 and coord2 must both have shape (natom, 3)")
    return c1 - c1.mean(axis=0), c2 - c2.mean(axis=0)


def _quat_matrix(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    sub = c1 - c2
    add = c1 + c2
    zero = np.zeros(len(c1))
    a = np.stack(
        [
            np.stack([zero, sub[:, 0], sub[:, 1], sub[:, 2]], axis=-1),
            np.stack([-sub[:, 0], zero, -add[:, 2], add[:, 1]], axis=-1),
            np.stack([-sub[:, 1], add[:, 2], zero, -add[:, 0]], axis=-1),
            np.stack([-sub[:, 2], -add[:, 1], add[:, 0], zero], axis=-1),
        ],
        axis=1,
    )
    return np.einsum("nki,nkj->ij", a, a)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = q
    return np.array(
        [
            [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2.0 * (q1 * q2 + q0 * q3), 2.0 * (q1 * q3 - q0 * q2)],
            [2.0 * (q1 * q2 - q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2.0 * (q2 * q3 + q0 * q1)],
            [2.0 * (q1 * q3 + q0 * q2), 2.0 * (q2 * q3 - q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
        ]
    )


def rti_dist(coord1: np.ndarray, coord2: np.ndarray) -> float:
    """Minimum RTI distance between two structures (Bohr)."""
    c1, c2 = _centered(coord1, coord2)
    return float(np.sqrt(max(np.linalg.eigvalsh(_quat_matrix(c1, c2))[0], 0.0)))


def rti_dists_vecs(coord1: np.ndarray, coord2: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """All four RTI distances (ascending) and the displacement vectors ``coord2 - coord1 @ rot_k``."""
    c1, c2 = _centered(coord1, coord2)
    eigvals, eigvecs = np.linalg.eigh(_quat_matrix(c1, c2))
    dists = np.sqrt(np.maximum(eigvals, 0.0))
    vecs = [c2 - c1 @ _quat_to_rot(eigvecs[:, k]) for k in range(4)]
    return dists, vecs


def rti_pot(coord1: np.ndarray, coord2: np.ndarray, a: float, sigma: float) -> float:
    """RTIP energy of ``coord2`` for a Gaussian of height ``a`` and width ``sigma`` centred at ``coord1``."""
    return rti_pot_force(coord1, coord2, a, sigma)[0]


def rti_pot_force(coord1: np.ndarray, coord2: np.ndarray, a: float, sigma: float) -> tuple[float, np.ndarray]:
    """RTIP energy and the atomic forces on ``coord2``.

    When the two structures coincide (d_0 -> 0) the weights diverge; the limit is
    E = a with zero force, which is returned instead of the NaN the Rust code would produce.
    """
    dists, vecs = rti_dists_vecs(coord1, coord2)
    if not dists[0] > 1.0e-8:
        return float(a), np.zeros_like(vecs[0])

    w = dists**-7
    dw = -7.0 * dists**-8
    sum_w = w.sum()
    u = a * np.exp(-(dists * dists) / (2.0 * sigma * sigma))
    pot_k = w / sum_w * u

    force = np.zeros_like(vecs[0])
    for k in range(4):
        coupling = np.sum(w * (u - u[k]))
        force += vecs[k] * (pot_k[k] / (sigma * sigma) + dw[k] / (sum_w * sum_w * dists[k]) * coupling)
    return float(pot_k.sum()), force
