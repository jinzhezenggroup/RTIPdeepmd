from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import typing as t

import jax.numpy as jnp
import numpy as np

try:
	from .common.constants import ROOT_RANK
	from .external.deepmd import DeepmdPES
	from .io import output
	from .io.input import Para
	from .pes_exploration.md import evolution_nvt_md
	from .pes_exploration.potential import EvolutionPot
	from .pes_exploration.system import System
except ImportError:
	from src.common.constants import ROOT_RANK
	from src.external.deepmd import DeepmdPES
	from src.io import output
	from src.io.input import Para
	from src.pes_exploration.md import evolution_nvt_md
	from src.pes_exploration.potential import EvolutionPot
	from src.pes_exploration.system import System


LIBRS_RECIPE: list[tuple[str, int]] = [
	("Ca.xyz", 1),
	("OH.xyz", 2),
	("20-linear-pentose.xyz", 1),
	("19-tetrose.xyz", 0),
	("Ca-OH-OH.xyz", 0),
	("10-1-HCOCHOH.xyz", 0),
	("10-Ca-OH-HCOCHOH.xyz", 0),
	("16-Ca-OH-HCOCHOHCHOH.xyz", 0),
	("Glycolaldehyde.xyz", 0),
	("18-glyceraldehyde.xyz", 0),
	("Ethylene_Glycol.xyz", 0),
	("Ca-OH-HCOCH2O.xyz", 0),
	("Ca-OH-HOCH2CH2O.xyz", 0),
	("Ca-OH-CH2OH.xyz", 0),
	("Ca-OH-OH-HOCH2CH2OH.xyz", 0),
	("12-HOCH2CH2OH.xyz", 0),
	("13-Ca-OH-COCH2OH.xyz", 0),
	("14-Ca-OH-HOCH2CHOH.xyz", 0),
	("15-HOCHCHOH.xyz", 0),
	("CH2O.xyz", 3),
	("H2O.xyz", 6),
	("HCO.xyz", 0),
]


class _SingleProcessComm:
	rank: int = ROOT_RANK

	def bcast(self, obj: t.Any, root: int = ROOT_RANK) -> t.Any:
		del root
		return obj


def _comm_rank(comm: t.Any) -> int:
	if hasattr(comm, "rank"):
		rank_attr = getattr(comm, "rank")
		return int(rank_attr() if callable(rank_attr) else rank_attr)
	if hasattr(comm, "Get_rank") and callable(getattr(comm, "Get_rank")):
		return int(comm.Get_rank())
	return ROOT_RANK


def _init_comm(enable_mpi: bool) -> t.Any:
	if not enable_mpi:
		return _SingleProcessComm()

	try:
		from mpi4py import MPI  # type: ignore

		world = MPI.COMM_WORLD
		# Mirrors Rust side split_by_color(world.rank() % 1), which always yields color 0.
		return world.Split(color=world.Get_rank() % 1, key=world.Get_rank())
	except Exception:
		return _SingleProcessComm()


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Python entrypoint aligned with lib.rs workflow, using DeepMD as real PES backend."
	)
	parser.add_argument("--xyz-dir", default=".", help="Directory containing XYZ files listed in the lib.rs recipe.")
	parser.add_argument(
		"--model",
		default=os.environ.get("MODEL_PATH") or os.environ.get("DEEPMD_MODEL", "deepmd.pb"),
		help="DeepMD model file path (.pb/.pth) or pretrained model name.",
	)
	parser.add_argument(
		"--type-map",
		nargs="*",
		metavar="ELEM:ID",
		help="Element-to-type mapping, e.g. H:0 O:1 C:2 Ca:3.",
	)
	parser.add_argument("--type-map-json", default=None, help="JSON file containing type map, e.g. {'H':0,'O':1}.")
	parser.add_argument("--min-dist", type=float, default=3.5, help="Minimum inter-molecule distance in Angstrom for auto_super_cell.")
	parser.add_argument("--output-index", type=int, default=0, help="Output directory index passed to io.output.create_output_path.")
	parser.add_argument("--supercell-out", default="supercell.xyz", help="Output XYZ file for the generated supercell.")
	parser.add_argument("--strict-xyz", action="store_true", help="Fail if any recipe XYZ file is missing.")
	parser.add_argument("--dry-run", action="store_true", help="Build supercell and outputs only; skip DeepMD evaluation and MD.")
	parser.add_argument("--no-mpi", action="store_true", help="Force single-process mode even if mpi4py is available.")
	return parser


def _parse_type_map_items(items: list[str]) -> dict[str, int]:
	mapping: dict[str, int] = {}
	for item in items:
		if ":" not in item:
			raise ValueError(f"Invalid --type-map item: {item}. Expected ELEM:ID.")
		elem, idx = item.split(":", 1)
		elem = elem.strip()
		if not elem:
			raise ValueError(f"Invalid --type-map item: {item}. Empty element symbol.")
		mapping[elem] = int(idx.strip())
	return mapping


def _load_type_map(type_map_json: str | None, type_map_items: list[str] | None) -> dict[str, int]:
	mapping: dict[str, int] = {}
	if type_map_json is not None:
		json_path = Path(type_map_json)
		raw = json.loads(json_path.read_text(encoding="utf-8"))
		if not isinstance(raw, dict):
			raise ValueError("--type-map-json must contain a JSON object, e.g. {'H':0,'O':1}.")
		for key, value in raw.items():
			mapping[str(key)] = int(value)

	if type_map_items:
		mapping.update(_parse_type_map_items(type_map_items))

	return mapping


def _resolve_type_map(user_type_map: dict[str, int], system: System, rank: int) -> dict[str, int]:
	if system.atom_type is None:
		return user_type_map

	needed = sorted({atom.value for atom in system.atom_type})
	if user_type_map:
		missing = [elem for elem in needed if elem not in user_type_map]
		if missing:
			raise ValueError(f"type_map is missing elements required by the system: {missing}")
		return user_type_map

	# No user map: defer to DeepMD model's own type_map ordering in DeepmdPES.
	if rank == ROOT_RANK:
		print("No type_map specified; using DeepMD model type_map ordering.")
	return {}


def _load_initial_system(recipe_dir: Path, strict_xyz: bool, min_dist: float) -> tuple[System, list[str], bool]:
	systems: list[System] = []
	mul: list[int] = []
	missing: list[str] = []

	for filename, multiplicity in LIBRS_RECIPE:
		xyz_path = recipe_dir / filename
		if not xyz_path.exists():
			missing.append(filename)
			continue
		systems.append(System.read_xyz(str(xyz_path)))
		mul.append(int(multiplicity))

	if strict_xyz and missing:
		raise FileNotFoundError(f"Missing XYZ files from recipe: {', '.join(missing)}")

	if not systems:
		raise FileNotFoundError(f"No recipe XYZ files were found under: {recipe_dir}")

	auto_adjusted_mul = False
	if sum(mul) <= 0:
		mul[0] = 1
		auto_adjusted_mul = True

	state = System.auto_super_cell(systems, mul, float(min_dist))
	return state, missing, auto_adjusted_mul


def _broadcast_state(s: System, comm: t.Any) -> System:
	if not hasattr(comm, "bcast") or not callable(getattr(comm, "bcast")):
		return s

	rank = _comm_rank(comm)
	coord = comm.bcast(np.asarray(s.coord, dtype=np.float64) if rank == ROOT_RANK else None, root=ROOT_RANK)
	cell = comm.bcast(np.asarray(s.cell, dtype=np.float64) if (rank == ROOT_RANK and s.cell is not None) else None, root=ROOT_RANK)

	s.coord = jnp.asarray(np.asarray(coord, dtype=np.float64))
	s.cell = None if cell is None else jnp.asarray(np.asarray(cell, dtype=np.float64))
	return s


def _run(argv: list[str] | None = None) -> int:
	args = _build_parser().parse_args(argv)

	print("Begin Initialization")
	comm = _init_comm(enable_mpi=not args.no_mpi)
	rank = _comm_rank(comm)

	para = Para.new()
	output_path = output.create_output_path(comm, args.output_index)

	s, missing, auto_adjusted_mul = _load_initial_system(
		Path(args.xyz_dir),
		strict_xyz=args.strict_xyz,
		min_dist=args.min_dist,
	)
	if rank == ROOT_RANK and missing:
		print(f"Skipped missing XYZ files: {', '.join(missing)}")
	if rank == ROOT_RANK and auto_adjusted_mul:
		print("Recipe multiplicities sum to zero for available XYZ files; adjusted first multiplicity to 1.")

	s = _broadcast_state(s, comm)
	if rank == ROOT_RANK:
		s.write_xyz(args.supercell_out, True, 0)

	evolution_pot = EvolutionPot(
		initial_state=s,
		initial_velocity=None,
		para=para,
		output_path=output_path,
	)

	if args.dry_run:
		if rank == ROOT_RANK:
			print("Dry-run complete: supercell and output path prepared, MD was skipped.")
		print("Finalization done")
		return 0

	model_name = str(args.model).strip()
	if not model_name:
		raise ValueError("DeepMD model is empty. Provide --model as a file path or pretrained model name.")

	model_path = Path(model_name)
	looks_like_file = (os.sep in model_name) or (model_path.suffix.lower() in {".pb", ".pth", ".pt"})
	if model_path.exists():
		if not model_path.is_file():
			raise FileNotFoundError(f"DeepMD model path is not a file: {model_path}")
	else:
		if looks_like_file:
			raise FileNotFoundError(f"DeepMD model file not found: {model_path}")
		if rank == ROOT_RANK:
			print(f"DeepMD model '{model_name}' not found locally; trying it as a pretrained model name.")

	user_type_map = _load_type_map(args.type_map_json, args.type_map)
	type_map = _resolve_type_map(user_type_map, s, rank)

	try:
		with DeepmdPES.new(model_name, type_map=type_map) as deepmd_pes:
			evolution_nvt_md(evolution_pot, deepmd_pes, comm=comm)
	except ValueError as exc:
		if (not model_path.exists()) and (not looks_like_file) and ("Cannot detect the backend of the model file" in str(exc)):
			raise RuntimeError(
				f"DeepMD model '{model_name}' could not be resolved as a pretrained model name in the current environment. "
				"Please provide an existing .pb/.pth/.pt file via --model or MODEL_PATH."
			) from exc
		raise

	print("Finalization done")
	return 0


def main() -> int:
	return _run()


if __name__ == "__main__":
	raise SystemExit(_run())
