from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import typing as t

import jax.numpy as jnp
import numpy as np

from ..common.constants import BOLTZMANN, FEMTOSECOND_TO_AU, HARTREE_TO_JOULE
from .potential import DistributedPot, EvolutionPot, RepulsivePot
from .rtip import Rtip0PES, RtipPES, rti_dist
from .system import System
from .traits import PES


@dataclass
class MdRunResult:
	final_system: System
	trajectory_file: str
	log_file: str


def _coord_with_atom_add_pot(s: System) -> jnp.ndarray:
	if s.atom_add_pot is None:
		return s.coord
	return s.coord[s.atom_add_pot, :]


def _random_perturbation(s: System, rng: np.random.Generator) -> jnp.ndarray:
	if s.atom_add_pot is None:
		dcoord = rng.uniform(0.0, 1.0, size=np.asarray(s.coord).shape)
		dcoord -= dcoord.mean(axis=0, keepdims=True)
		norm = max(float(np.linalg.norm(dcoord)), 1.0e-12)
		return jnp.asarray(np.asarray(s.coord) + dcoord * (0.1 / norm))

	idx = s.atom_add_pot
	dfrag = rng.uniform(0.0, 1.0, size=(len(idx), 3))
	dfrag -= dfrag.mean(axis=0, keepdims=True)
	norm = max(float(np.linalg.norm(dfrag)), 1.0e-12)

	coord = np.asarray(s.coord).copy()
	for i, atom_idx in enumerate(idx):
		coord[atom_idx] += dfrag[i] * (0.1 / norm)
	return jnp.asarray(coord)


def _atom_mass_array(s: System) -> np.ndarray:
	if s.atom_type is None:
		return np.ones((s.natom,), dtype=float)
	return np.asarray([atom.get_atomic_mass() for atom in s.atom_type], dtype=float)


def _write_header(log_file: str, header: str) -> None:
	with open(log_file, "w", encoding="utf-8") as f:
		f.write(header)


def _append_line(log_file: str, line: str) -> None:
	with open(log_file, "a", encoding="utf-8") as f:
		f.write(line)


def _make_repulsive_bias_pes(pot: RepulsivePot, s: System, step: int) -> Rtip0PES:
	sigma_min = rti_dist(_coord_with_atom_add_pot(pot.local_min), _coord_with_atom_add_pot(s))
	sigma_ts: list[float] = []
	if pot.para is None:
		scale_ts_sigma = None
	else:
		scale_ts_sigma = pot.para.rtip_para.scale_ts_sigma

	if scale_ts_sigma is None:
		for ts in pot.nearby_ts:
			sigma_ts.append(rti_dist(_coord_with_atom_add_pot(ts), _coord_with_atom_add_pot(s)))
	else:
		for ts in pot.nearby_ts:
			sigma_ts.append(0.5 * scale_ts_sigma * rti_dist(_coord_with_atom_add_pot(ts), _coord_with_atom_add_pot(pot.local_min)))

	a0 = pot.para.rtip_para.a0 if pot.para is not None else 0.0
	scale_ts_a0 = pot.para.rtip_para.scale_ts_a0 if pot.para is not None else 1.0
	a_min = a0 * float(step)
	return Rtip0PES(
		local_min=pot.local_min,
		nearby_ts=pot.nearby_ts,
		a_min=a_min,
		a_ts=a_min * scale_ts_a0,
		sigma_min=sigma_min,
		sigma_ts=sigma_ts,
	)


def _build_evolution_target(s: System, mol_index: list[list[int]]) -> System:
	coord = np.asarray(s.coord).copy()
	if not mol_index:
		return s.clone()

	o_all = coord.mean(axis=0)
	target = coord.copy()
	for atom_ids in mol_index:
		if not atom_ids:
			continue
		o_i = coord[atom_ids].mean(axis=0)
		shift = 0.01 * (o_all - o_i)
		target[atom_ids] = coord[atom_ids] + shift

	return System(
		natom=s.natom,
		coord=jnp.asarray(target),
		cell=s.cell,
		atom_type=s.atom_type,
		atom_add_pot=s.atom_add_pot,
		mutable=s.mutable,
		pot=s.pot,
	)


def _nvt_loop(
	initial_state: System,
	para,
	output_path: str,
	real_pes: PES,
	bias_builder: t.Callable[[System, int], tuple[float, jnp.ndarray, float]],
	initial_velocity: jnp.ndarray | None = None,
	header: str = "  step            time       bias_dist            temp             kin        pot_real        pot_bias          f_real          f_bias\n",
) -> MdRunResult:
	out_dir = Path(output_path) if output_path else Path(".")
	out_dir.mkdir(parents=True, exist_ok=True)
	str_output_file = str(out_dir / "rtip.pdb")
	rtip_output_file = str(out_dir / "rtip.out")

	s = initial_state.clone()
	s.write_pdb(str_output_file, True, 0)
	_write_header(rtip_output_file, header)

	dt = para.md_para.dt
	max_step = para.md_para.max_step
	print_step = para.md_para.print_step

	masses = _atom_mass_array(s)
	vel = np.zeros_like(np.asarray(s.coord), dtype=float) if initial_velocity is None else np.asarray(initial_velocity, dtype=float).copy()
	acc = np.zeros_like(np.asarray(s.coord), dtype=float)

	t_fs = 0.0
	for i in range(1, max_step + 1):
		t_fs += dt

		vel += 0.5 * dt * FEMTOSECOND_TO_AU * acc
		coord = np.asarray(s.coord) + dt * FEMTOSECOND_TO_AU * vel
		s.coord = jnp.asarray(coord)

		kin_pre = 0.0
		for j in range(s.natom):
			kin_pre += masses[j] * float(np.dot(vel[j], vel[j]))
		temp = kin_pre * HARTREE_TO_JOULE / (BOLTZMANN * 3.0 * max(s.natom - 1, 1))
		if temp < 1.0:
			temp = 1.0
		lam = np.sqrt(1.0 + (dt / para.md_para.tau) * (para.md_para.temp_bath / temp - 1.0))

		pot_real, force_real = real_pes.get_energy_force(s)
		s.pot = float(pot_real)
		force_real_np = np.asarray(force_real, dtype=float)
		f_real = float(np.sqrt(np.sum(force_real_np * force_real_np)))

		pot_bias, force_bias, bias_dist = bias_builder(s, i)
		force_bias_np = np.asarray(force_bias, dtype=float)
		f_bias = float(np.sqrt(np.sum(force_bias_np * force_bias_np)))

		force_total = force_real_np + force_bias_np
		for j in range(s.natom):
			acc[j, 0] = force_total[j, 0] / masses[j]
			acc[j, 1] = force_total[j, 1] / masses[j]
			acc[j, 2] = force_total[j, 2] / masses[j]

		vel += 0.5 * dt * FEMTOSECOND_TO_AU * acc
		vel *= lam

		kin = 0.0
		for j in range(s.natom):
			kin += masses[j] * float(np.dot(vel[j], vel[j]))
		temp = kin * HARTREE_TO_JOULE / (BOLTZMANN * 3.0 * max(s.natom - 1, 1))
		kin *= 0.5

		if (i % print_step) == 0:
			s.write_pdb(str_output_file, False, i)

		_append_line(
			rtip_output_file,
			f"{i:6d} {t_fs:15.8f} {bias_dist:15.8f} {temp:15.8f} {kin:15.8f} {pot_real:15.8f} {pot_bias:15.8f} {f_real:15.8f} {f_bias:15.8f}\n",
		)

	return MdRunResult(final_system=s, trajectory_file=str_output_file, log_file=rtip_output_file)


def rtip_nvt_md(repulsive_pot: RepulsivePot, real_pes: PES, comm: object | None = None) -> MdRunResult:
	if repulsive_pot.para is None:
		raise ValueError("repulsive_pot.para is required")

	rng = np.random.default_rng(1314)
	initial_state = repulsive_pot.local_min.clone()
	initial_state.coord = _random_perturbation(initial_state, rng)

	def bias_builder(s: System, step: int) -> tuple[float, jnp.ndarray, float]:
		pes = _make_repulsive_bias_pes(repulsive_pot, s, step)
		pot_bias, force_bias = pes.get_energy_force(s)
		return float(pot_bias), force_bias, float(pes.sigma_min)

	return _nvt_loop(
		initial_state=initial_state,
		para=repulsive_pot.para,
		output_path=repulsive_pot.output_path,
		real_pes=real_pes,
		bias_builder=bias_builder,
	)


def evolution_nvt_md(evolution_pot: EvolutionPot, real_pes: PES, comm: object | None = None) -> MdRunResult:
	if evolution_pot.para is None:
		raise ValueError("evolution_pot.para is required")

	s0 = evolution_pot.initial_state.clone()
	if s0.atom_type is None:
		raise ValueError("initial_state.atom_type is required for evolution_nvt_md")

	atom_type_ref = s0.atom_type
	atomic_radii = s0.get_atomic_radii()
	if s0.atom_add_pot is None:
		dist_mat = s0.get_dist_mat()
		adj_mat = System.get_adj_mat(atom_type_ref, atomic_radii, dist_mat, 1.25, evolution_pot.para.md_para.ignored_pair)
		mol_index = System.split_into_mol(atomic_radii, dist_mat, 1.25)
	else:
		idx = s0.atom_add_pot
		frag = System(
			natom=len(idx),
			coord=s0.coord[idx, :],
			cell=None,
			atom_type=[atom_type_ref[i] for i in idx],
			atom_add_pot=None,
			mutable=None,
			pot=s0.pot,
		)
		atomic_radii = frag.get_atomic_radii()
		dist_mat = frag.get_dist_mat()
		adj_mat = System.get_adj_mat(frag.atom_type or [], atomic_radii, dist_mat, 1.25, evolution_pot.para.md_para.ignored_pair)
		mol_index = System.split_into_mol(atomic_radii, dist_mat, 1.25)

	a_min = 0.0

	def bias_builder(s: System, step: int) -> tuple[float, jnp.ndarray, float]:
		nonlocal dist_mat, adj_mat, mol_index, a_min

		eval_state = s
		if s.atom_add_pot is not None:
			idx = s.atom_add_pot
			eval_state = System(
				natom=len(idx),
				coord=s.coord[idx, :],
				cell=None,
				atom_type=[s.atom_type[i] for i in idx] if s.atom_type is not None else None,
				atom_add_pot=None,
				mutable=None,
				pot=s.pot,
			)

		dist_mat = eval_state.get_dist_mat()
		if (step % max(1, evolution_pot.para.md_para.split_step or 1)) == 0:
			mol_index = System.split_into_mol(atomic_radii, dist_mat, 1.25)

		final_state = _build_evolution_target(eval_state, mol_index)
		a_min += evolution_pot.para.rtip_para.a0

		sigma_min = rti_dist(final_state.coord, eval_state.coord)
		pes = Rtip0PES(
			local_min=final_state,
			nearby_ts=[],
			a_min=a_min,
			a_ts=0.0,
			sigma_min=sigma_min,
			sigma_ts=[],
		)

		if s.atom_add_pot is None:
			pot_bias, force_bias = pes.get_energy_force(s)
			return float(pot_bias), force_bias, float(sigma_min)

		pot_bias, force_frag = pes.get_energy_force(eval_state)
		force_total = np.zeros_like(np.asarray(s.coord), dtype=float)
		idx = s.atom_add_pot
		force_frag_np = np.asarray(force_frag)
		for i, atom_idx in enumerate(idx):
			force_total[atom_idx] = force_frag_np[i]
		return float(pot_bias), jnp.asarray(force_total), float(sigma_min)

	return _nvt_loop(
		initial_state=evolution_pot.initial_state.clone(),
		para=evolution_pot.para,
		output_path=evolution_pot.output_path,
		real_pes=real_pes,
		bias_builder=bias_builder,
		initial_velocity=evolution_pot.initial_velocity,
	)


def distributed_nvt_md(distributed_pot: DistributedPot, real_pes: PES, comm: object | None = None) -> MdRunResult:
	if distributed_pot.para is None:
		raise ValueError("distributed_pot.para is required")
	if not distributed_pot.visited_states:
		raise ValueError("distributed_pot.visited_states is empty")

	initial_state = distributed_pot.visited_states[-1].clone()
	visited_states = list(distributed_pot.visited_states)

	def bias_builder(s: System, step: int) -> tuple[float, jnp.ndarray, float]:
		del step
		pes = RtipPES(
			visited_states=visited_states,
			a=distributed_pot.para.rtip_para.a0,
			sigma=distributed_pot.para.rtip_para.sigma,
			comm=comm,
		)
		pot_bias, force_bias = pes.get_energy_force(s)
		return float(pot_bias), force_bias, 0.0

	return _nvt_loop(
		initial_state=initial_state,
		para=distributed_pot.para,
		output_path=distributed_pot.output_path,
		real_pes=real_pes,
		bias_builder=bias_builder,
	)
