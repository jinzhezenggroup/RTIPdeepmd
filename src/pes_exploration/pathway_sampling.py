from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import typing as t

import jax.numpy as jnp
import numpy as np

from . import optimization
from .idwm import Idwm0PES, idw_dist, idw_dist1, wei_dist_mat
from .potential import AttractivePot, RepulsivePot, SynthesisPot
from .rtip import Rtip0PES, rti_dist
from .system import System
from .traits import PES


@dataclass
class PathSamplingResult:
	final_system: System
	trajectory_file: str
	log_file: str


def _force_norm(force: jnp.ndarray) -> float:
	force_np = np.asarray(force, dtype=float)
	return float(np.sqrt(np.sum(force_np * force_np)))


def _coord_with_atom_add_pot(s: System) -> jnp.ndarray:
	if s.atom_add_pot is None:
		return s.coord
	return s.coord[s.atom_add_pot, :]


def _random_perturbation(s: System, rng: np.random.Generator) -> jnp.ndarray:
	coord = np.asarray(s.coord, dtype=float).copy()
	if s.atom_add_pot is None:
		dcoord = rng.uniform(0.0, 1.0, size=coord.shape)
		dcoord -= dcoord.mean(axis=0, keepdims=True)
		norm = max(float(np.linalg.norm(dcoord)), 1.0e-12)
		coord += dcoord * (0.1 / norm)
		return jnp.asarray(coord)

	idx = list(s.atom_add_pot)
	dfrag = rng.uniform(0.0, 1.0, size=(len(idx), 3))
	dfrag -= dfrag.mean(axis=0, keepdims=True)
	norm = max(float(np.linalg.norm(dfrag)), 1.0e-12)
	for i, atom_idx in enumerate(idx):
		coord[atom_idx] += dfrag[i] * (0.1 / norm)
	return jnp.asarray(coord)


def _init_outputs(output_path: str, s0: System, header: str) -> tuple[str, str]:
	out_dir = Path(output_path) if output_path else Path(".")
	out_dir.mkdir(parents=True, exist_ok=True)
	str_output_file = str(out_dir / "rtip.pdb")
	rtip_output_file = str(out_dir / "rtip.out")
	s0.write_pdb(str_output_file, True, 0)
	with open(rtip_output_file, "w", encoding="utf-8") as f:
		f.write(header)
	return str_output_file, rtip_output_file


def _append_path_sampling_line(
	log_file: str,
	step: int,
	sigma: float,
	pot_real: float,
	pot_bias: float,
	f_real: float,
	f_bias: float,
) -> None:
	with open(log_file, "a", encoding="utf-8") as f:
		f.write(f"{step:6d} {sigma:15.8f} {pot_real:15.8f} {pot_bias:15.8f} {f_real:15.8f} {f_bias:15.8f}\n")


def _make_fragment(source: System, indices: list[int]) -> System:
	atom_type = None if source.atom_type is None else [source.atom_type[i] for i in indices]
	return System(
		natom=len(indices),
		coord=source.coord[indices, :],
		cell=None,
		atom_type=atom_type,
		atom_add_pot=None,
		mutable=None,
		pot=source.pot,
	)


def rtip_path_sampling_repulsive(
	repulsive_pot: RepulsivePot,
	real_pes: PES,
	comm: object | None = None,
) -> PathSamplingResult:
	del comm
	if repulsive_pot.para is None:
		raise ValueError("repulsive_pot.para is required")

	para = repulsive_pot.para
	s = repulsive_pot.local_min.clone()
	rng = np.random.default_rng(1314)
	s.coord = _random_perturbation(s, rng)

	str_output_file, rtip_output_file = _init_outputs(
		repulsive_pot.output_path,
		repulsive_pot.local_min,
		"  step        rti_dist        pot_real        pot_rtip          f_real          f_rtip\n",
	)

	atom_add_pot = repulsive_pot.local_min.atom_add_pot
	if atom_add_pot is None:
		local_min_fragment = repulsive_pot.local_min
		nearby_ts_fragment = repulsive_pot.nearby_ts
		s_fragment = None
	else:
		atom_add_pot = list(atom_add_pot)
		local_min_fragment = _make_fragment(repulsive_pot.local_min, atom_add_pot)
		nearby_ts_fragment = [_make_fragment(ts, atom_add_pot) for ts in repulsive_pot.nearby_ts]
		s_fragment = _make_fragment(s, atom_add_pot)

	rtip0_pes = Rtip0PES(
		local_min=local_min_fragment,
		nearby_ts=nearby_ts_fragment,
		a_min=0.0,
		a_ts=0.0,
		sigma_min=10.0,
		sigma_ts=[10.0 for _ in nearby_ts_fragment],
	)

	pot_real_max = -1.0e15
	pot_real_min = 1.0e15
	add_rtip = True

	for i in range(1, para.path_samp_para.max_step + 1):
		pot_real, force_real = real_pes.get_energy_force(s)
		pot_real = float(pot_real)
		s.pot = pot_real
		f_real = _force_norm(force_real)

		force_rtip = jnp.zeros_like(s.coord)
		if atom_add_pot is None:
			rtip0_pes.sigma_min = rti_dist(repulsive_pot.local_min.coord, s.coord)
			if add_rtip:
				if para.rtip_para.scale_ts_sigma is None:
					rtip0_pes.sigma_ts = [rti_dist(ts.coord, s.coord) for ts in repulsive_pot.nearby_ts]
				else:
					rtip0_pes.sigma_ts = [
						(0.5 * para.rtip_para.scale_ts_sigma) * rti_dist(ts.coord, repulsive_pot.local_min.coord)
						for ts in repulsive_pot.nearby_ts
					]
				rtip0_pes.a_min = para.rtip_para.a0 * float(i)
				rtip0_pes.a_ts = rtip0_pes.a_min * para.rtip_para.scale_ts_a0
				pot_rtip, force_rtip = rtip0_pes.get_energy_force(s)
				pot_rtip = float(pot_rtip)
				f_rtip = _force_norm(force_rtip)
			else:
				rtip0_pes.sigma_ts = [10.0 for _ in repulsive_pot.nearby_ts]
				rtip0_pes.a_min = 0.0
				rtip0_pes.a_ts = 0.0
				pot_rtip = 0.0
				f_rtip = 0.0
		else:
			if s_fragment is None:
				raise ValueError("s_fragment must not be None when atom_add_pot is set")
			s_fragment.coord = s.coord[atom_add_pot, :]
			rtip0_pes.sigma_min = rti_dist(local_min_fragment.coord, s_fragment.coord)
			if add_rtip:
				if para.rtip_para.scale_ts_sigma is None:
					rtip0_pes.sigma_ts = [rti_dist(ts.coord, s_fragment.coord) for ts in nearby_ts_fragment]
				else:
					rtip0_pes.sigma_ts = [
						(0.5 * para.rtip_para.scale_ts_sigma) * rti_dist(ts.coord, local_min_fragment.coord)
						for ts in nearby_ts_fragment
					]
				rtip0_pes.a_min = para.rtip_para.a0 * float(i)
				rtip0_pes.a_ts = rtip0_pes.a_min * para.rtip_para.scale_ts_a0
				pot_rtip, force_rtip_fragment = rtip0_pes.get_energy_force(s_fragment)
				pot_rtip = float(pot_rtip)
				f_rtip = _force_norm(force_rtip_fragment)

				force_total_np = np.zeros_like(np.asarray(s.coord), dtype=float)
				frag_force_np = np.asarray(force_rtip_fragment, dtype=float)
				for j, atom_idx in enumerate(atom_add_pot):
					force_total_np[atom_idx] = frag_force_np[j]
				force_rtip = jnp.asarray(force_total_np)
			else:
				rtip0_pes.sigma_ts = [10.0 for _ in nearby_ts_fragment]
				rtip0_pes.a_min = 0.0
				rtip0_pes.a_ts = 0.0
				pot_rtip = 0.0
				f_rtip = 0.0

		if (i % para.path_samp_para.print_step) == 0:
			s.write_pdb(str_output_file, False, i)
		_append_path_sampling_line(rtip_output_file, i, float(rtip0_pes.sigma_min), pot_real, pot_rtip, f_real, f_rtip)

		force_total = force_real + force_rtip
		dcoord = optimization.min_1d_real_bias(
			real_pes,
			rtip0_pes,
			s,
			force_total,
			pot_real + pot_rtip,
			para.path_samp_para.pot_epsilon * float(s.natom),
		)
		s.coord = s.coord + dcoord

		pot_real_max = max(pot_real_max, pot_real)
		pot_real_min = min(pot_real_min, pot_real)
		if pot_real < (pot_real_max - para.path_samp_para.pot_drop):
			add_rtip = False
		if (not add_rtip) and (f_real / np.sqrt(float(s.natom)) < para.path_samp_para.f_epsilon):
			break
		if pot_real > (pot_real_min + para.path_samp_para.pot_climb) or f_rtip > 1000.0:
			break

	return PathSamplingResult(final_system=s, trajectory_file=str_output_file, log_file=rtip_output_file)


def idwm_path_sampling(
	repulsive_pot: RepulsivePot,
	real_pes: PES,
	comm: object | None = None,
) -> PathSamplingResult:
	del comm
	if repulsive_pot.para is None:
		raise ValueError("repulsive_pot.para is required")

	para = repulsive_pot.para
	s = repulsive_pot.local_min.clone()
	rng = np.random.default_rng(1314)
	s.coord = _random_perturbation(s, rng)

	str_output_file, rtip_output_file = _init_outputs(
		repulsive_pot.output_path,
		repulsive_pot.local_min,
		"  step        idw_dist        pot_real        pot_idwm          f_real          f_idwm\n",
	)

	atom_add_pot = repulsive_pot.local_min.atom_add_pot
	if atom_add_pot is None:
		local_min_w = wei_dist_mat(repulsive_pot.local_min.coord)
		nearby_ts_w = [wei_dist_mat(ts.coord) for ts in repulsive_pot.nearby_ts]
		s_fragment = None
	else:
		atom_add_pot = list(atom_add_pot)
		local_min_coord = repulsive_pot.local_min.coord[atom_add_pot, :]
		nearby_ts_coord = [ts.coord[atom_add_pot, :] for ts in repulsive_pot.nearby_ts]
		local_min_w = wei_dist_mat(local_min_coord)
		nearby_ts_w = [wei_dist_mat(ts_coord) for ts_coord in nearby_ts_coord]
		s_fragment = _make_fragment(s, atom_add_pot)

	idwm0_pes = Idwm0PES(
		local_min=local_min_w,
		nearby_ts=nearby_ts_w,
		a_min=0.0,
		a_ts=0.0,
		sigma_min=10.0,
		sigma_ts=[10.0 for _ in nearby_ts_w],
	)

	pot_real_max = -1.0e15
	pot_real_min = 1.0e15
	add_idwm = True

	for i in range(1, para.path_samp_para.max_step + 1):
		pot_real, force_real = real_pes.get_energy_force(s)
		pot_real = float(pot_real)
		s.pot = pot_real
		f_real = _force_norm(force_real)

		force_idwm = jnp.zeros_like(s.coord)
		if atom_add_pot is None:
			idwm0_pes.sigma_min = idw_dist(idwm0_pes.local_min, s.coord)
			if add_idwm:
				if para.rtip_para.scale_ts_sigma is None:
					idwm0_pes.sigma_ts = [idw_dist(ts, s.coord) for ts in idwm0_pes.nearby_ts]
				else:
					idwm0_pes.sigma_ts = [
						(0.5 * para.rtip_para.scale_ts_sigma) * idw_dist1(ts, idwm0_pes.local_min)
						for ts in idwm0_pes.nearby_ts
					]
				idwm0_pes.a_min = para.rtip_para.a0 * float(i)
				idwm0_pes.a_ts = idwm0_pes.a_min * para.rtip_para.scale_ts_a0
				pot_idwm, force_idwm = idwm0_pes.get_energy_force(s)
				pot_idwm = float(pot_idwm)
				f_idwm = _force_norm(force_idwm)
			else:
				idwm0_pes.sigma_ts = [10.0 for _ in repulsive_pot.nearby_ts]
				idwm0_pes.a_min = 0.0
				idwm0_pes.a_ts = 0.0
				pot_idwm = 0.0
				f_idwm = 0.0
		else:
			if s_fragment is None:
				raise ValueError("s_fragment must not be None when atom_add_pot is set")
			s_fragment.coord = s.coord[atom_add_pot, :]
			idwm0_pes.sigma_min = idw_dist(idwm0_pes.local_min, s_fragment.coord)
			if add_idwm:
				if para.rtip_para.scale_ts_sigma is None:
					idwm0_pes.sigma_ts = [idw_dist(ts, s_fragment.coord) for ts in idwm0_pes.nearby_ts]
				else:
					idwm0_pes.sigma_ts = [
						(0.5 * para.rtip_para.scale_ts_sigma) * idw_dist1(ts, idwm0_pes.local_min)
						for ts in idwm0_pes.nearby_ts
					]
				idwm0_pes.a_min = para.rtip_para.a0 * float(i)
				idwm0_pes.a_ts = idwm0_pes.a_min * para.rtip_para.scale_ts_a0
				pot_idwm, force_idwm_fragment = idwm0_pes.get_energy_force(s_fragment)
				pot_idwm = float(pot_idwm)
				f_idwm = _force_norm(force_idwm_fragment)

				force_total_np = np.zeros_like(np.asarray(s.coord), dtype=float)
				frag_force_np = np.asarray(force_idwm_fragment, dtype=float)
				for j, atom_idx in enumerate(atom_add_pot):
					force_total_np[atom_idx] = frag_force_np[j]
				force_idwm = jnp.asarray(force_total_np)
			else:
				idwm0_pes.sigma_ts = [10.0 for _ in repulsive_pot.nearby_ts]
				idwm0_pes.a_min = 0.0
				idwm0_pes.a_ts = 0.0
				pot_idwm = 0.0
				f_idwm = 0.0

		if (i % para.path_samp_para.print_step) == 0:
			s.write_pdb(str_output_file, False, i)
		_append_path_sampling_line(rtip_output_file, i, float(idwm0_pes.sigma_min), pot_real, pot_idwm, f_real, f_idwm)

		force_total = force_real + force_idwm
		dcoord = optimization.min_1d_real_bias(
			real_pes,
			idwm0_pes,
			s,
			force_total,
			pot_real + pot_idwm,
			para.path_samp_para.pot_epsilon * float(s.natom),
		)
		s.coord = s.coord + dcoord

		pot_real_max = max(pot_real_max, pot_real)
		pot_real_min = min(pot_real_min, pot_real)
		if pot_real < (pot_real_max - para.path_samp_para.pot_drop):
			add_idwm = False
		if (not add_idwm) and (f_real / np.sqrt(float(s.natom)) < para.path_samp_para.f_epsilon):
			break
		if pot_real > (pot_real_min + para.path_samp_para.pot_climb) or f_idwm > 1000.0:
			break

	return PathSamplingResult(final_system=s, trajectory_file=str_output_file, log_file=rtip_output_file)


def rtip_path_sampling_attractive(
	attractive_pot: AttractivePot,
	real_pes: PES,
	comm: object | None = None,
) -> PathSamplingResult:
	del comm
	if attractive_pot.para is None:
		raise ValueError("attractive_pot.para is required")

	para = attractive_pot.para
	s = attractive_pot.initial_state.clone()

	str_output_file, rtip_output_file = _init_outputs(
		attractive_pot.output_path,
		attractive_pot.initial_state,
		"  step        rti_dist        pot_real        pot_rtip          f_real          f_rtip\n",
	)

	atom_add_pot = attractive_pot.final_state.atom_add_pot
	if atom_add_pot is None:
		final_state_fragment = attractive_pot.final_state
		s_fragment = None
	else:
		atom_add_pot = list(atom_add_pot)
		final_state_fragment = _make_fragment(attractive_pot.final_state, atom_add_pot)
		s_fragment = _make_fragment(s, atom_add_pot)

	rtip0_pes = Rtip0PES(
		local_min=final_state_fragment,
		nearby_ts=[],
		a_min=0.0,
		a_ts=0.0,
		sigma_min=10.0,
		sigma_ts=[],
	)

	add_rtip = True
	for i in range(1, para.path_samp_para.max_step + 1):
		pot_real, force_real = real_pes.get_energy_force(s)
		pot_real = float(pot_real)
		s.pot = pot_real
		f_real = _force_norm(force_real)

		force_rtip = jnp.zeros_like(s.coord)
		if atom_add_pot is None:
			rtip0_pes.sigma_min = rti_dist(attractive_pot.final_state.coord, s.coord)
			if add_rtip:
				rtip0_pes.a_min = -para.rtip_para.a0 * float(i)
				pot_rtip, force_rtip = rtip0_pes.get_energy_force(s)
				pot_rtip = float(pot_rtip)
				f_rtip = _force_norm(force_rtip)
			else:
				rtip0_pes.a_min = 0.0
				pot_rtip = 0.0
				f_rtip = 0.0
		else:
			if s_fragment is None:
				raise ValueError("s_fragment must not be None when atom_add_pot is set")
			s_fragment.coord = s.coord[atom_add_pot, :]
			rtip0_pes.sigma_min = rti_dist(final_state_fragment.coord, s_fragment.coord)
			if add_rtip:
				rtip0_pes.a_min = -para.rtip_para.a0 * float(i)
				pot_rtip, force_rtip_fragment = rtip0_pes.get_energy_force(s_fragment)
				pot_rtip = float(pot_rtip)
				f_rtip = _force_norm(force_rtip_fragment)

				force_total_np = np.zeros_like(np.asarray(s.coord), dtype=float)
				frag_force_np = np.asarray(force_rtip_fragment, dtype=float)
				for j, atom_idx in enumerate(atom_add_pot):
					force_total_np[atom_idx] = frag_force_np[j]
				force_rtip = jnp.asarray(force_total_np)
			else:
				rtip0_pes.a_min = 0.0
				pot_rtip = 0.0
				f_rtip = 0.0

		if (i % para.path_samp_para.print_step) == 0:
			s.write_pdb(str_output_file, False, i)
		_append_path_sampling_line(rtip_output_file, i, float(rtip0_pes.sigma_min), pot_real, pot_rtip, f_real, f_rtip)

		force_total = force_real + force_rtip
		dcoord = optimization.min_1d_real_bias(
			real_pes,
			rtip0_pes,
			s,
			force_total,
			pot_real + pot_rtip,
			para.path_samp_para.pot_epsilon * float(s.natom),
		)
		s.coord = s.coord + dcoord

		if rtip0_pes.sigma_min < 1.0 or f_rtip > 1000.0:
			add_rtip = False
		if (not add_rtip) and (f_real / np.sqrt(float(s.natom)) < para.path_samp_para.f_epsilon):
			break

	return PathSamplingResult(final_system=s, trajectory_file=str_output_file, log_file=rtip_output_file)


def rtip_path_sampling_synthesis(
	synthesis_pot: SynthesisPot,
	real_pes: PES,
	comm: object | None = None,
) -> PathSamplingResult:
	del comm
	if synthesis_pot.para is None:
		raise ValueError("synthesis_pot.para is required")

	para = synthesis_pot.para
	s = synthesis_pot.initial_state.clone()

	str_output_file, rtip_output_file = _init_outputs(
		synthesis_pot.output_path,
		synthesis_pot.initial_state,
		"  step        rti_dist        pot_real        pot_rtip          f_real          f_rtip\n",
	)

	mol_indices: list[int] = []
	for idx in synthesis_pot.mol_index:
		mol_indices.extend(idx)

	final_state = System(
		natom=len(mol_indices),
		coord=jnp.zeros((len(mol_indices), 3), dtype=jnp.float64),
		cell=None,
		atom_type=None,
		atom_add_pot=None,
		mutable=None,
		pot=0.0,
	)

	if len(mol_indices) == s.natom:
		s.atom_add_pot = None
		s_fragment = None
	else:
		s.atom_add_pot = list(mol_indices)
		s_fragment = System(
			natom=len(mol_indices),
			coord=jnp.zeros((len(mol_indices), 3), dtype=jnp.float64),
			cell=None,
			atom_type=None,
			atom_add_pot=None,
			mutable=None,
			pot=0.0,
		)

	pot_real_max = -1.0e15
	pot_real_min = 1.0e15
	pot_real_initial = 0.0
	add_rtip = True

	for i in range(1, para.path_samp_para.max_step + 1):
		pot_real, force_real = real_pes.get_energy_force(s)
		pot_real = float(pot_real)
		s.pot = pot_real
		f_real = _force_norm(force_real)

		force_rtip = jnp.zeros_like(s.coord)

		if len(mol_indices) == s.natom:
			s_coord_np = np.asarray(s.coord, dtype=float)
			final_np = np.asarray(final_state.coord, dtype=float)
			o_all = s_coord_np.mean(axis=0)
			for mol in synthesis_pot.mol_index:
				if not mol:
					continue
				o_i = s_coord_np[mol, :].mean(axis=0)
				shift = 0.01 * (o_all - o_i)
				for atom_idx in mol:
					final_np[atom_idx, :] = s_coord_np[atom_idx, :] + shift
			final_state.coord = jnp.asarray(final_np)

			if add_rtip:
				rtip0_pes = Rtip0PES(
					local_min=final_state,
					nearby_ts=[],
					a_min=-para.rtip_para.a0 * float(i),
					a_ts=0.0,
					sigma_min=rti_dist(final_state.coord, s.coord),
					sigma_ts=[],
				)
				pot_rtip, force_rtip = rtip0_pes.get_energy_force(s)
				pot_rtip = float(pot_rtip)
				f_rtip = _force_norm(force_rtip)
			else:
				rtip0_pes = Rtip0PES(
					local_min=final_state,
					nearby_ts=[],
					a_min=0.0,
					a_ts=0.0,
					sigma_min=rti_dist(final_state.coord, s.coord),
					sigma_ts=[],
				)
				pot_rtip = 0.0
				f_rtip = 0.0
		else:
			if s_fragment is None:
				raise ValueError("s_fragment must not be None for synthesis with environment")
			s_fragment.coord = s.coord[mol_indices, :]

			s_frag_np = np.asarray(s_fragment.coord, dtype=float)
			full_np = np.asarray(s.coord, dtype=float)
			final_np = np.asarray(final_state.coord, dtype=float)
			o_all = s_frag_np.mean(axis=0)
			index = 0
			for mol in synthesis_pot.mol_index:
				if not mol:
					continue
				o_i = full_np[mol, :].mean(axis=0)
				shift = 0.01 * (o_all - o_i)
				for atom_idx in mol:
					final_np[index, :] = full_np[atom_idx, :] + shift
					index += 1
			final_state.coord = jnp.asarray(final_np)

			if add_rtip:
				rtip0_pes = Rtip0PES(
					local_min=final_state,
					nearby_ts=[],
					a_min=-para.rtip_para.a0 * float(i),
					a_ts=0.0,
					sigma_min=rti_dist(final_state.coord, s_fragment.coord),
					sigma_ts=[],
				)
				pot_rtip, force_rtip_fragment = rtip0_pes.get_energy_force(s_fragment)
				pot_rtip = float(pot_rtip)
				f_rtip = _force_norm(force_rtip_fragment)

				force_total_np = np.zeros_like(np.asarray(s.coord), dtype=float)
				frag_force_np = np.asarray(force_rtip_fragment, dtype=float)
				for j, atom_idx in enumerate(mol_indices):
					force_total_np[atom_idx] = frag_force_np[j]
				force_rtip = jnp.asarray(force_total_np)
			else:
				rtip0_pes = Rtip0PES(
					local_min=final_state,
					nearby_ts=[],
					a_min=0.0,
					a_ts=0.0,
					sigma_min=rti_dist(final_state.coord, s_fragment.coord),
					sigma_ts=[],
				)
				pot_rtip = 0.0
				f_rtip = 0.0

		if (i % para.path_samp_para.print_step) == 0:
			s.write_pdb(str_output_file, False, i)
		_append_path_sampling_line(rtip_output_file, i, float(rtip0_pes.sigma_min), pot_real, pot_rtip, f_real, f_rtip)

		force_total = force_real + force_rtip
		dcoord = optimization.min_1d_real_bias(
			real_pes,
			rtip0_pes,
			s,
			force_total,
			pot_real + pot_rtip,
			para.path_samp_para.pot_epsilon * float(s.natom),
		)
		s.coord = s.coord + dcoord

		pot_real_max = max(pot_real_max, pot_real)
		pot_real_min = min(pot_real_min, pot_real)
		if i == 1:
			pot_real_initial = pot_real
		if (pot_real < (pot_real_max - para.path_samp_para.pot_drop)) and (pot_real > pot_real_initial):
			add_rtip = False
		if (not add_rtip) and (f_real / np.sqrt(float(s.natom)) < para.path_samp_para.f_epsilon):
			break
		if pot_real > (pot_real_min + para.path_samp_para.pot_climb) or f_rtip > 1000.0:
			break

	return PathSamplingResult(final_system=s, trajectory_file=str_output_file, log_file=rtip_output_file)


def rtip_path_sampling(
	pot: RepulsivePot | AttractivePot | SynthesisPot,
	real_pes: PES,
	comm: object | None = None,
) -> PathSamplingResult:
	if isinstance(pot, RepulsivePot):
		return rtip_path_sampling_repulsive(pot, real_pes, comm)
	if isinstance(pot, AttractivePot):
		return rtip_path_sampling_attractive(pot, real_pes, comm)
	if isinstance(pot, SynthesisPot):
		return rtip_path_sampling_synthesis(pot, real_pes, comm)
	raise TypeError(f"Unsupported potential type for rtip_path_sampling: {type(pot)}")


def _repulsive_rtip_method(self: RepulsivePot, comm: t.Any, real_pes: PES) -> None:
	rtip_path_sampling_repulsive(self, real_pes, comm)


def _repulsive_idwm_method(self: RepulsivePot, comm: t.Any, real_pes: PES) -> None:
	idwm_path_sampling(self, real_pes, comm)


def _attractive_rtip_method(self: AttractivePot, comm: t.Any, real_pes: PES) -> None:
	rtip_path_sampling_attractive(self, real_pes, comm)


def _synthesis_rtip_method(self: SynthesisPot, comm: t.Any, real_pes: PES) -> None:
	rtip_path_sampling_synthesis(self, real_pes, comm)


RepulsivePot.rtip_path_sampling = _repulsive_rtip_method
RepulsivePot.idwm_path_sampling = _repulsive_idwm_method
AttractivePot.rtip_path_sampling = _attractive_rtip_method
SynthesisPot.rtip_path_sampling = _synthesis_rtip_method


__all__ = [
	"PathSamplingResult",
	"rtip_path_sampling",
	"rtip_path_sampling_repulsive",
	"rtip_path_sampling_attractive",
	"rtip_path_sampling_synthesis",
	"idwm_path_sampling",
]
