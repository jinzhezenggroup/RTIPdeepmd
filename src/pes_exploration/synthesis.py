from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..matrix import rand_rot
from .system import System


def _center_and_rotate(coord: np.ndarray) -> np.ndarray:
	centered = coord - coord.mean(axis=0, keepdims=True)
	rot = np.asarray(rand_rot(), dtype=float)
	return centered @ rot


def synthesis(comm: object | None, s: System, mol_index: list[list[int]], dist: float) -> None:
	del comm
	if len(mol_index) not in (2, 3, 4):
		raise ValueError("Synthesis currently supports 2, 3, or 4 molecules only.")

	coord = np.asarray(s.coord, dtype=float).copy()

	mol_coords: list[np.ndarray] = []
	for atom_ids in mol_index:
		if not atom_ids:
			raise ValueError("Each molecule in mol_index must contain at least one atom index.")
		mol_coords.append(_center_and_rotate(coord[atom_ids, :]))

	if len(mol_index) == 2:
		shifts = [
			np.array([-dist, 0.0, 0.0], dtype=float),
			np.array([dist, 0.0, 0.0], dtype=float),
		]
	elif len(mol_index) == 3:
		shifts = [
			np.array([0.0, dist, 0.0], dtype=float),
			np.array([dist * 0.866025403784439, -dist * 0.5, 0.0], dtype=float),
			np.array([-dist * 0.866025403784439, -dist * 0.5, 0.0], dtype=float),
		]
	else:
		shifts = [
			np.array([0.0, 0.0, dist], dtype=float),
			np.array([0.0, dist * 0.942809041582063, -dist * 0.333333333333333], dtype=float),
			np.array([dist * 0.816496580927726, -dist * 0.471404520791032, -dist * 0.333333333333333], dtype=float),
			np.array([-dist * 0.816496580927726, -dist * 0.471404520791032, -dist * 0.333333333333333], dtype=float),
		]

	for mol_i, atom_ids in enumerate(mol_index):
		moved = mol_coords[mol_i] + shifts[mol_i]
		for local_i, atom_idx in enumerate(atom_ids):
			coord[atom_idx, :] = moved[local_i, :]

	s.coord = jnp.asarray(coord)


__all__ = ["synthesis"]
