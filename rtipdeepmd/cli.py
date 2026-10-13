"""Command-line entry point: build the initial box from the recipe and run RTIP evolution MD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets

import numpy as np

from .constants import BOHR_TO_ANGSTROM
from .md import EvolutionPot, evolution_nvt_md
from .params import Para
from .system import System

# Same molecules and multiplicities as the Rust lib.rs (formose reaction, Ca(OH)2 + CH2O + H2O).
RECIPE: list[tuple[str, int]] = [
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

DEFAULT_XYZ_DIR = Path(__file__).resolve().parent.parent / "molecules"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m rtipdeepmd", description="RTIP evolution MD with a DeepMD PES.")
    p.add_argument("--model", help="DeepMD model file (.pt/.pth/.pb). Required unless --dry-run.")
    p.add_argument("--head", default=None, help="Model head for multi-task DeepMD models.")
    p.add_argument("--output-dir", default="run", help="Directory for all outputs.")
    p.add_argument("--xyz-dir", type=Path, default=DEFAULT_XYZ_DIR, help="Directory with the recipe XYZ files.")
    p.add_argument("--init", type=Path, default=None, help="Start from this XYZ instead of building a box.")
    p.add_argument("--min-dist", type=float, default=3.5, help="Minimum intermolecular distance when building the box (Angstrom).")
    p.add_argument("--seed", type=int, default=None, help="Random seed for box construction (default: random, recorded).")
    p.add_argument("--pbc", action="store_true", help="Evaluate DeepMD with the box as periodic cell (Rust/CP2K: non-periodic).")
    p.add_argument("--stop-temp", type=float, default=None, help="Abort when the temperature exceeds this value (K).")
    p.add_argument("--dry-run", action="store_true", help="Only build and write the initial structure.")
    defaults = Para()
    for name in ("a0", "dt", "tau", "temp_bath", "decreasing_multiple", "decreasing_bound"):
        p.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    for name in ("max_step", "print_step", "split_step"):
        p.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    return p


def build_initial_state(xyz_dir: Path, min_dist: float, rng: np.random.Generator) -> System:
    systems, mul = [], []
    for name, n in RECIPE:
        if n > 0:
            systems.append(System.read_xyz(xyz_dir / name))
            mul.append(n)
    return System.auto_super_cell(systems, mul, min_dist, rng)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    para = Para(
        a0=args.a0,
        dt=args.dt,
        tau=args.tau,
        temp_bath=args.temp_bath,
        max_step=args.max_step,
        print_step=args.print_step,
        decreasing_multiple=args.decreasing_multiple,
        decreasing_bound=args.decreasing_bound,
        split_step=args.split_step if args.split_step and args.split_step > 0 else None,
    )
    seed = args.seed if args.seed is not None else secrets.randbits(32)

    if args.init is not None:
        s = System.read_xyz(args.init)
    else:
        s = build_initial_state(args.xyz_dir, args.min_dist, np.random.default_rng(seed))
    s.write_xyz(out / "supercell.xyz", True, 0)
    box = None if s.cell is None else float(s.cell[0, 0] * BOHR_TO_ANGSTROM)
    print(f"Initial structure: {s.natom} atoms, box {box} Angstrom, seed {seed} -> {out / 'supercell.xyz'}", flush=True)

    run_info = {
        "model": args.model,
        "head": args.head,
        "pbc": args.pbc,
        "seed": seed,
        "init": None if args.init is None else str(args.init),
        "min_dist": args.min_dist,
        "stop_temp": args.stop_temp,
        "para": para.to_dict(),
    }
    (out / "run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        return 0
    if not args.model:
        raise SystemExit("--model is required")

    from .deepmd_pes import DeepmdPES

    pes = DeepmdPES(args.model, head=args.head, pbc=args.pbc)
    print(f"DeepMD model {args.model} (type_map: {' '.join(pes.type_map[:10])} ...)", flush=True)
    evolution_nvt_md(EvolutionPot(initial_state=s, para=para, output_path=out), pes, stop_temp=args.stop_temp)
    print("Finished", flush=True)
    return 0
