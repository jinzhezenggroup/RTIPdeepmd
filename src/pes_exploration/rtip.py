from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from .system import System


def _as_np(coord: jnp.ndarray) -> np.ndarray:
    arr = np.asarray(coord, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("coord must have shape (natom, 3)")
    return arr


def _quat_matrix(coord1: np.ndarray, coord2: np.ndarray) -> np.ndarray:
    o1 = coord1.mean(axis=0)
    o2 = coord2.mean(axis=0)
    c1 = coord1 - o1
    c2 = coord2 - o2

    add = c1 + c2
    sub = c1 - c2
    s = np.zeros((4, 4), dtype=float)
    for i in range(c1.shape[0]):
        a = np.array(
            [
                [0.0, sub[i, 0], sub[i, 1], sub[i, 2]],
                [-sub[i, 0], 0.0, -add[i, 2], add[i, 1]],
                [-sub[i, 1], add[i, 2], 0.0, -add[i, 0]],
                [-sub[i, 2], -add[i, 1], add[i, 0], 0.0],
            ],
            dtype=float,
        )
        s += a.T @ a
    return s


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = q
    return np.array(
        [
            [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2.0 * (q1 * q2 + q0 * q3), 2.0 * (q1 * q3 - q0 * q2)],
            [2.0 * (q1 * q2 - q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2.0 * (q2 * q3 + q0 * q1)],
            [2.0 * (q1 * q3 + q0 * q2), 2.0 * (q2 * q3 - q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
        ],
        dtype=float,
    )

@dataclass
class Rtip0PES:
    local_min: System
    nearby_ts: list[System] = field(default_factory=list)
    a_min: float = 0.0
    a_ts: float = 0.0
    sigma_min: float = 1.0
    sigma_ts: list[float] = field(default_factory=list)

    def _coord_ref(self, s: System, indices: list[int] | None) -> jnp.ndarray:
        if indices is None:
            return s.coord
        return s.coord[indices, :]

    def get_energy(self, s: System) -> float:
        idx_ref = self.local_min.atom_add_pot
        idx_s = s.atom_add_pot

        local_min_coord = self._coord_ref(self.local_min, idx_ref)
        s_coord = self._coord_ref(s, idx_s)

        pot_rtip = rti_pot(local_min_coord, s_coord, self.a_min, self.sigma_min)
        for i, ts in enumerate(self.nearby_ts):
            sigma_ts_i = self.sigma_ts[i] if i < len(self.sigma_ts) else self.sigma_min
            ts_coord = self._coord_ref(ts, idx_ref)
            pot_rtip += rti_pot(ts_coord, s_coord, self.a_ts, sigma_ts_i)

        return float(pot_rtip)

    def get_energy_force(self, s: System) -> tuple[float, jnp.ndarray]:
        idx_ref = self.local_min.atom_add_pot
        idx_s = s.atom_add_pot

        local_min_coord = self._coord_ref(self.local_min, idx_ref)
        s_coord = self._coord_ref(s, idx_s)

        pot_rtip, force_rtip_eval = rti_pot_force(local_min_coord, s_coord, self.a_min, self.sigma_min)
        for i, ts in enumerate(self.nearby_ts):
            sigma_ts_i = self.sigma_ts[i] if i < len(self.sigma_ts) else self.sigma_min
            ts_coord = self._coord_ref(ts, idx_ref)
            pot_ts, force_ts = rti_pot_force(ts_coord, s_coord, self.a_ts, sigma_ts_i)
            pot_rtip += pot_ts
            force_rtip_eval = force_rtip_eval + force_ts

        if idx_s is None:
            return float(pot_rtip), force_rtip_eval

        force_total = np.zeros_like(np.asarray(s.coord), dtype=float)
        force_eval_np = np.asarray(force_rtip_eval)
        for i, atom_index in enumerate(idx_s):
            force_total[atom_index] = force_eval_np[i]
        return float(pot_rtip), jnp.asarray(force_total)

@dataclass
class RtipPES:
    visited_states: list[System] = field(default_factory=list)
    a: float = 0.0
    sigma: float = 1.0
    comm: object | None = None

    def _coord_ref(self, s: System, indices: list[int] | None) -> jnp.ndarray:
        if indices is None:
            return s.coord
        return s.coord[indices, :]

    def get_energy(self, s: System) -> float:
        if not self.visited_states:
            return 0.0

        idx_states = self.visited_states[0].atom_add_pot
        idx_s = s.atom_add_pot
        s_coord = self._coord_ref(s, idx_s)

        pot_rtip = 0.0
        for state in self.visited_states:
            state_coord = self._coord_ref(state, idx_states)
            pot_rtip += rti_pot(state_coord, s_coord, self.a, self.sigma)
        return float(pot_rtip)

    def get_energy_force(self, s: System) -> tuple[float, jnp.ndarray]:
        if not self.visited_states:
            return 0.0, jnp.zeros_like(s.coord)

        idx_states = self.visited_states[0].atom_add_pot
        idx_s = s.atom_add_pot
        s_coord = self._coord_ref(s, idx_s)

        if idx_s is None:
            force_acc = np.zeros_like(np.asarray(s.coord), dtype=float)
        else:
            force_acc = np.zeros((len(idx_s), 3), dtype=float)

        pot_rtip = 0.0
        for state in self.visited_states:
            state_coord = self._coord_ref(state, idx_states)
            pot_i, force_i = rti_pot_force(state_coord, s_coord, self.a, self.sigma)
            pot_rtip += pot_i
            force_acc += np.asarray(force_i)

        if idx_s is None:
            return float(pot_rtip), jnp.asarray(force_acc)

        force_total = np.zeros_like(np.asarray(s.coord), dtype=float)
        for i, atom_index in enumerate(idx_s):
            force_total[atom_index] = force_acc[i]
        return float(pot_rtip), jnp.asarray(force_total)


def rti_dist(coord1: jnp.ndarray, coord2: jnp.ndarray) -> float:
    c1 = _as_np(coord1)
    c2 = _as_np(coord2)
    if c1.shape != c2.shape:
        raise ValueError("coord1 and coord2 must have same shape")

    s = _quat_matrix(c1, c2)
    eigvals = np.linalg.eigvalsh(s)
    return float(np.sqrt(max(eigvals[0], 0.0)))


def rti_dists(coord1: jnp.ndarray, coord2: jnp.ndarray) -> list[float]:
    c1 = _as_np(coord1)
    c2 = _as_np(coord2)
    if c1.shape != c2.shape:
        raise ValueError("coord1 and coord2 must have same shape")

    s = _quat_matrix(c1, c2)
    eigvals = np.linalg.eigvalsh(s)
    return [float(np.sqrt(max(v, 0.0))) for v in eigvals]


def rti_dist_vec(coord1: jnp.ndarray, coord2: jnp.ndarray) -> tuple[float, jnp.ndarray]:
    c1 = _as_np(coord1)
    c2 = _as_np(coord2)
    if c1.shape != c2.shape:
        raise ValueError("coord1 and coord2 must have same shape")

    o1 = c1.mean(axis=0)
    o2 = c2.mean(axis=0)
    c1c = c1 - o1
    c2c = c2 - o2

    s = _quat_matrix(c1, c2)
    eigvals, eigvecs = np.linalg.eigh(s)
    rot = _quat_to_rot(eigvecs[:, 0])
    vec = c2c - c1c @ rot
    return float(np.sqrt(max(eigvals[0], 0.0))), jnp.asarray(vec)


def rti_dists_vecs(coord1: jnp.ndarray, coord2: jnp.ndarray) -> tuple[list[float], list[jnp.ndarray]]:
    c1 = _as_np(coord1)
    c2 = _as_np(coord2)
    if c1.shape != c2.shape:
        raise ValueError("coord1 and coord2 must have same shape")

    o1 = c1.mean(axis=0)
    o2 = c2.mean(axis=0)
    c1c = c1 - o1
    c2c = c2 - o2

    s = _quat_matrix(c1, c2)
    eigvals, eigvecs = np.linalg.eigh(s)

    dists: list[float] = []
    vecs: list[jnp.ndarray] = []
    for i in range(4):
        rot = _quat_to_rot(eigvecs[:, i])
        vec = c2c - c1c @ rot
        dists.append(float(np.sqrt(max(eigvals[i], 0.0))))
        vecs.append(jnp.asarray(vec))

    return dists, vecs


def rti_rot_tran(coord1: jnp.ndarray, coord2: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    c1 = _as_np(coord1)
    c2 = _as_np(coord2)
    if c1.shape != c2.shape:
        raise ValueError("coord1 and coord2 must have same shape")

    o1 = c1.mean(axis=0)
    o2 = c2.mean(axis=0)
    s = _quat_matrix(c1, c2)
    _eigvals, eigvecs = np.linalg.eigh(s)
    rot = _quat_to_rot(eigvecs[:, 0])
    tran = o2 - o1 @ rot
    return jnp.asarray(rot), jnp.asarray(tran)


def f(x: float) -> float:
    return 1.0 / (x**7)


def df(x: float) -> float:
    return -7.0 / (x**8)


def rti_pot(coord1: jnp.ndarray, coord2: jnp.ndarray, a: float, sigma: float) -> float:
    dists = rti_dists(coord1, coord2)
    if np.isfinite(dists[0]) and abs(dists[0]) > 1.0e-3:
        weights = [f(d) for d in dists]
        sum_w = float(sum(weights))
        pot = 0.0
        for wi, di in zip(weights, dists):
            pot += a * (wi / sum_w) * np.exp(-(di * di) / (2.0 * sigma * sigma))
        return float(pot)
    return float(a)


def rti_pot_force(coord1: jnp.ndarray, coord2: jnp.ndarray, a: float, sigma: float) -> tuple[float, jnp.ndarray]:
    dists, vecs = rti_dists_vecs(coord1, coord2)
    if not (np.isfinite(dists[0]) and abs(dists[0]) > 1.0e-3):
        return float(a), jnp.zeros_like(coord2)

    weights = [f(d) for d in dists]
    dweights = [df(d) for d in dists]
    sum_w = float(sum(weights))
    u = [a * np.exp(-(d * d) / (2.0 * sigma * sigma)) for d in dists]
    pot = 0.0
    force = np.zeros_like(np.asarray(coord2), dtype=float)

    for i in range(4):
        pot_i = (weights[i] / sum_w) * u[i]
        pot += pot_i

        coupling = 0.0
        for j in range(4):
            if j == i:
                continue
            coupling += weights[j] * (u[j] - u[i])

        denom = max(dists[i], 1.0e-12)
        scale = pot_i / (sigma * sigma) + dweights[i] / (sum_w * sum_w * denom) * coupling
        force += np.asarray(vecs[i]) * scale

    return float(pot), jnp.asarray(force)


def get_energy(pes: Rtip0PES | RtipPES, s: System) -> float:
    return pes.get_energy(s)


def get_energy_force(pes: Rtip0PES | RtipPES, s: System) -> tuple[float, jnp.ndarray]:
    return pes.get_energy_force(s)
