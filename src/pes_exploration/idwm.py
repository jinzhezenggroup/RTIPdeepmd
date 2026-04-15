from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp

from .system import System


def w(x: float) -> float:
    n = 5
    sigma = 3.0
    return float(np.exp(-((x / sigma) ** n)) + 1.0)


def dw(x: float) -> float:
    n = 5
    sigma = 3.0
    return float(np.exp(-((x / sigma) ** n)) * (-(n * (x ** (n - 1))) / (sigma**n)))


def wei_dist_mat(coord: jnp.ndarray) -> jnp.ndarray:
    coord_np = np.asarray(coord, dtype=float)
    natom = coord_np.shape[0]
    w_mat = np.zeros((natom, natom), dtype=float)

    for i in range(natom - 1):
        for j in range(i + 1, natom):
            rij = np.linalg.norm(coord_np[i] - coord_np[j])
            w_mat[i, j] = w(float(rij))

    return jnp.asarray(w_mat)


def idw_dist(w_mat0: jnp.ndarray, coord: jnp.ndarray) -> float:
    w0 = np.asarray(w_mat0, dtype=float)
    coord_np = np.asarray(coord, dtype=float)
    natom = coord_np.shape[0]

    dist = 0.0
    for i in range(natom - 1):
        for j in range(i + 1, natom):
            rij = np.linalg.norm(coord_np[i] - coord_np[j])
            dist += (w(float(rij)) - w0[i, j]) ** 2

    return float(np.sqrt(dist))


def idw_dist1(w_mat0: jnp.ndarray, w_mat1: jnp.ndarray) -> float:
    w0 = np.asarray(w_mat0, dtype=float)
    w1 = np.asarray(w_mat1, dtype=float)
    natom = w0.shape[0]

    dist = 0.0
    for i in range(natom - 1):
        for j in range(i + 1, natom):
            dist += (w1[i, j] - w0[i, j]) ** 2

    return float(np.sqrt(dist))


def idw_dist_vec(w_mat0: jnp.ndarray, coord: jnp.ndarray) -> tuple[float, jnp.ndarray]:
    w0 = np.asarray(w_mat0, dtype=float)
    coord_np = np.asarray(coord, dtype=float)
    natom = coord_np.shape[0]

    vec_mat = np.zeros((natom, natom, 3), dtype=float)
    dist_mat = np.zeros((natom, natom), dtype=float)
    w_mat = np.zeros((natom, natom), dtype=float)
    dw_mat = np.zeros((natom, natom), dtype=float)

    dist = 0.0
    for i in range(natom - 1):
        for j in range(i + 1, natom):
            vec_mat[i, j] = coord_np[i] - coord_np[j]
            dist_mat[i, j] = np.linalg.norm(vec_mat[i, j])
            w_mat[i, j] = w(float(dist_mat[i, j]))
            dw_mat[i, j] = dw(float(dist_mat[i, j]))
            dist += (w_mat[i, j] - w0[i, j]) ** 2

    dist = float(np.sqrt(dist))

    vec = np.zeros((natom, 3), dtype=float)
    if dist < 1.0e-16:
        return 0.0, jnp.asarray(vec)

    for i in range(natom):
        for j in range(natom):
            if i < j and dist_mat[i, j] > 1.0e-16:
                factor = (w_mat[i, j] - w0[i, j]) * dw_mat[i, j] / dist_mat[i, j]
                vec[i] += factor * vec_mat[i, j]
            if j < i and dist_mat[j, i] > 1.0e-16:
                factor = (w_mat[j, i] - w0[j, i]) * dw_mat[j, i] / dist_mat[j, i]
                vec[i] -= factor * vec_mat[j, i]

    vec /= dist
    return dist, jnp.asarray(vec)


def idw_pot(w_mat0: jnp.ndarray, coord: jnp.ndarray, a: float, sigma: float) -> float:
    idw_distance = idw_dist(w_mat0, coord)
    return float(a * np.exp(-(idw_distance * idw_distance) / (2.0 * sigma * sigma)))


def idw_pot_force(w_mat0: jnp.ndarray, coord: jnp.ndarray, a: float, sigma: float) -> tuple[float, jnp.ndarray]:
    idw_distance, idw_vec = idw_dist_vec(w_mat0, coord)
    pot_idw = float(a * np.exp(-(idw_distance * idw_distance) / (2.0 * sigma * sigma)))
    force_idw = idw_vec * (pot_idw * idw_distance / (sigma * sigma))
    return pot_idw, force_idw


@dataclass
class Idwm0PES:
    local_min: jnp.ndarray
    nearby_ts: list[jnp.ndarray] = field(default_factory=list)
    a_min: float = 0.0
    a_ts: float = 0.0
    sigma_min: float = 1.0
    sigma_ts: list[float] = field(default_factory=list)

    def _fragment_coord(self, s: System) -> tuple[jnp.ndarray, list[int] | None]:
        if s.atom_add_pot is None:
            return s.coord, None
        indices = list(s.atom_add_pot)
        return s.coord[indices, :], indices

    def get_energy(self, s: System) -> float:
        coord_eval, _ = self._fragment_coord(s)

        pot_idwm = idw_pot(self.local_min, coord_eval, self.a_min, self.sigma_min)
        for i, ts in enumerate(self.nearby_ts):
            sigma_ts = self.sigma_ts[i] if i < len(self.sigma_ts) else self.sigma_min
            pot_idwm += idw_pot(ts, coord_eval, self.a_ts, sigma_ts)
        return float(pot_idwm)

    def get_energy_force(self, s: System) -> tuple[float, jnp.ndarray]:
        coord_eval, indices = self._fragment_coord(s)

        pot_idwm, force_fragment = idw_pot_force(self.local_min, coord_eval, self.a_min, self.sigma_min)
        for i, ts in enumerate(self.nearby_ts):
            sigma_ts = self.sigma_ts[i] if i < len(self.sigma_ts) else self.sigma_min
            pot_ts, force_ts = idw_pot_force(ts, coord_eval, self.a_ts, sigma_ts)
            pot_idwm += pot_ts
            force_fragment = force_fragment + force_ts

        if indices is None:
            return float(pot_idwm), force_fragment

        force_total = np.zeros_like(np.asarray(s.coord), dtype=float)
        force_fragment_np = np.asarray(force_fragment)
        for i, atom_index in enumerate(indices):
            force_total[atom_index] = force_fragment_np[i]

        return float(pot_idwm), jnp.asarray(force_total)


def get_energy(idwm_pes: Idwm0PES, s: System) -> float:
    return idwm_pes.get_energy(s)


def get_energy_force(idwm_pes: Idwm0PES, s: System) -> tuple[float, jnp.ndarray]:
    return idwm_pes.get_energy_force(s)
