"""Command-line entry point: build the initial box from the recipe and run RTIP evolution MD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets

import numpy as np

from .constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, check_element
from .md import EvolutionPot, evolution_nvt_md
from .params import Para
from .pathway import (
    AttractivePot,
    PathPara,
    SynthesisPot,
    attractive_path_sampling,
    place_molecules,
    synthesis_path_sampling,
)
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
    p = argparse.ArgumentParser(prog="python -m rtipdeepmd", description="RTIP evolution MD / pathway sampling with a DeepMD PES.")
    p.add_argument(
        "--method",
        choices=["evolution", "synthesis", "attractive"],
        default="evolution",
        help="evolution: RTIP-driven NVT MD of a box of molecules (Rust EvolutionPot); "
        "synthesis: RTIP pathway sampling of 2-4 molecules (Rust SynthesisPot); "
        "attractive: RTIP pathway sampling toward --final (Rust AttractivePot).",
    )
    p.add_argument("--model", help="DeepMD model file (.pt/.pth/.pb). Required unless --dry-run.")
    p.add_argument("--head", default="OMol25", help="Head of a multi-task DeepMD model (default: OMol25, molecular DFT).")
    p.add_argument("--charge", type=int, default=0, help="Total charge (frame parameter of DPA-3.x models).")
    p.add_argument("--spin", type=int, default=None, help="Spin multiplicity 2S+1 (default: lowest consistent with the electron count).")
    p.add_argument("--output-dir", default="run", help="Directory for all outputs.")
    p.add_argument("--xyz-dir", type=Path, default=DEFAULT_XYZ_DIR, help="Directory with the recipe XYZ files.")
    p.add_argument(
        "--molecule",
        action="append",
        metavar="FILE[:COUNT]",
        help="Build the box from these molecules instead of the built-in formose recipe (repeatable).",
    )
    p.add_argument(
        "--ignored-pairs",
        default=None,
        help='Element pairs ignored in bond-change detection, e.g. "H-O,C-O"; "none" for no pair '
        "(default: the Rust formose list H-O,C-O,H-Ca,C-Ca,O-Ca,Ca-Ca).",
    )
    p.add_argument("--init", type=Path, default=None, help="Start from this XYZ instead of building a box.")
    p.add_argument("--final", type=Path, default=None, help="attractive: final-state XYZ, same atom order as the initial state.")
    p.add_argument("--min-dist", type=float, default=3.5, help="Minimum intermolecular distance when building the box (Angstrom).")
    p.add_argument("--seed", type=int, default=None, help="Random seed for box construction (default: random, recorded).")
    p.add_argument("--pbc", action="store_true", help="Evaluate DeepMD with the box as periodic cell (Rust/CP2K: non-periodic).")
    p.add_argument("--stop-temp", type=float, default=None, help="Abort when the temperature exceeds this value (K).")
    p.add_argument("--dry-run", action="store_true", help="Only build and write the initial structure.")
    p.add_argument(
        "--fragment",
        action="append",
        metavar="MOL:ATOMS",
        help="synthesis: only these atoms of molecule MOL (1-based, e.g. 1:6,17-18) feel the RTIP; "
        "the rest of that molecule is environment. Default: whole molecules.",
    )
    p.add_argument(
        "--min-contact",
        type=float,
        default=0.0,
        help="synthesis: redraw placements until intermolecular distances >= this (Angstrom; Rust: no check).",
    )
    p.add_argument(
        "--synthesis-dist",
        type=float,
        default=5.0,
        help="synthesis: distance of each molecule centre from the origin, Bohr (Rust: 5.0).",
    )
    defaults, path_defaults = Para(), PathPara()
    for name in ("a0", "dt", "tau", "temp_bath", "decreasing_multiple", "decreasing_bound"):
        p.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    for name in ("pot_climb", "pot_drop", "pot_epsilon", "f_epsilon"):
        p.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(path_defaults, name), help="synthesis only")
    p.add_argument("--split-step", type=int, default=defaults.split_step)
    p.add_argument("--max-step", type=int, default=None, help=f"default: {defaults.max_step} (evolution), {path_defaults.max_step} (synthesis)")
    p.add_argument("--print-step", type=int, default=None, help="default: 1")
    return p


def build_initial_state(
    xyz_dir: Path,
    min_dist: float,
    rng: np.random.Generator,
    recipe: list[tuple[str, int]] = RECIPE,
) -> System:
    systems, mul = [], []
    for name, n in recipe:
        if n > 0:
            systems.append(System.read_xyz(xyz_dir / name))
            mul.append(n)
    return System.auto_super_cell(systems, mul, min_dist, rng)


def parse_molecules(items: list[str]) -> list[tuple[str, int]]:
    recipe = []
    for item in items:
        name, sep, count = item.rpartition(":")
        if not sep or not count.isdigit():
            name, count = item, "1"
        recipe.append((name, int(count)))
    return recipe


def parse_ignored_pairs(text: str | None) -> list[tuple[str, str]] | None:
    if text is None:
        return None
    if text.strip().lower() == "none":
        return []
    pairs = []
    for item in text.split(","):
        a, sep, b = item.strip().partition("-")
        if not sep:
            raise ValueError(f"Invalid element pair {item!r}; expected e.g. H-O")
        pairs.append((check_element(a), check_element(b)))
    return pairs


def parse_fragments(items: list[str] | None, mol_index: list[list[int]]) -> list[list[int]]:
    """Reacting atoms per molecule from "MOL:ATOMS" items (1-based, ranges allowed)."""
    frags = [list(m) for m in mol_index]
    for item in items or []:
        mol, _, atoms = item.partition(":")
        k = int(mol) - 1
        picked = []
        for part in atoms.split(","):
            lo, _, hi = part.partition("-")
            picked.extend(range(int(lo), int(hi or lo) + 1))
        if not 0 <= k < len(mol_index) or not all(1 <= a <= len(mol_index[k]) for a in picked):
            raise SystemExit(f"invalid --fragment {item!r}")
        frags[k] = [mol_index[k][a - 1] for a in picked]
    return frags


def build_synthesis_state(
    recipe: list[tuple[str, int]], dist: float, rng: np.random.Generator, min_contact: float = 0.0
) -> tuple[System, list[list[int]]]:
    if not 2 <= len(recipe) <= 4 or any(n != 1 for _, n in recipe):
        raise SystemExit("--method synthesis needs 2 to 4 --molecule files, each with count 1")
    mols = [System.read_xyz(Path(name)) for name, _ in recipe]
    s = System(coord=np.concatenate([m.coord for m in mols]), atom_type=[e for m in mols for e in m.atom_type])
    bounds = np.cumsum([0] + [m.natom for m in mols])
    mol_index = [list(range(int(bounds[k]), int(bounds[k + 1]))) for k in range(len(mols))]
    return place_molecules(s, mol_index, dist, rng, min_contact), mol_index


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.method in ("synthesis", "attractive"):
        return main_synthesis(args, out)

    para = Para(
        a0=args.a0,
        dt=args.dt,
        tau=args.tau,
        temp_bath=args.temp_bath,
        max_step=args.max_step if args.max_step is not None else Para().max_step,
        print_step=args.print_step if args.print_step is not None else 1,
        decreasing_multiple=args.decreasing_multiple,
        decreasing_bound=args.decreasing_bound,
        split_step=args.split_step if args.split_step and args.split_step > 0 else None,
    )
    ignored = parse_ignored_pairs(args.ignored_pairs)
    if ignored is not None:
        para.ignored_pair = ignored
    seed = args.seed if args.seed is not None else secrets.randbits(32)

    if args.init is not None:
        s = System.read_xyz(args.init)
    elif args.molecule:
        s = build_initial_state(Path("."), args.min_dist, np.random.default_rng(seed), parse_molecules(args.molecule))
    else:
        s = build_initial_state(args.xyz_dir, args.min_dist, np.random.default_rng(seed))
    s.write_xyz(out / "supercell.xyz", True, 0)
    box = None if s.cell is None else float(s.cell[0, 0] * BOHR_TO_ANGSTROM)
    print(f"Initial structure: {s.natom} atoms, box {box} Angstrom, seed {seed} -> {out / 'supercell.xyz'}", flush=True)

    run_info = {
        "model": args.model,
        "head": args.head,
        "charge": args.charge,
        "spin": args.spin,
        "pbc": args.pbc,
        "seed": seed,
        "init": None if args.init is None else str(args.init),
        "molecule": args.molecule,
        "min_dist": args.min_dist,
        "stop_temp": args.stop_temp,
        "para": para.to_dict(),
    }
    (out / "run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        return 0

    pes = load_pes(args)
    report_model(args, pes, s)
    evolution_nvt_md(EvolutionPot(initial_state=s, para=para, output_path=out), pes, stop_temp=args.stop_temp)
    print("Finished", flush=True)
    return 0


def load_pes(args: argparse.Namespace):
    if not args.model:
        raise SystemExit("--model is required")
    from .deepmd_pes import DeepmdPES

    return DeepmdPES.load(args.model, head=args.head, pbc=args.pbc, charge=args.charge, spin=args.spin)


def report_model(args: argparse.Namespace, pes, s: System) -> None:
    _, fparam = pes.frame_inputs(s.atom_type)
    fparam_text = "none" if fparam is None else f"charge {fparam[0]:g}, multiplicity {fparam[1]:g}"
    print(f"DeepMD model {args.model}, head {args.head}, frame parameters: {fparam_text}", flush=True)


def main_synthesis(args: argparse.Namespace, out: Path) -> int:
    if args.method == "attractive" and args.final is None:
        raise SystemExit("--method attractive needs --final")
    if not args.molecule and args.init is None:
        raise SystemExit(f"--method {args.method} needs --molecule files or --init")
    path_para = PathPara(
        a0=args.a0,
        pot_climb=args.pot_climb,
        pot_drop=args.pot_drop,
        pot_epsilon=args.pot_epsilon,
        f_epsilon=args.f_epsilon,
        max_step=args.max_step if args.max_step is not None else PathPara().max_step,
        print_step=args.print_step if args.print_step is not None else 1,
    )
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    if args.init is not None:
        s = System.read_xyz(args.init)
        mol_index = [list(range(s.natom))]
    else:
        s, mol_index = build_synthesis_state(
            parse_molecules(args.molecule), args.synthesis_dist, np.random.default_rng(seed), args.min_contact * ANGSTROM_TO_BOHR
        )
    mol_index = parse_fragments(args.fragment, mol_index)
    s.write_xyz(out / "initial.xyz", True, 0)
    print(f"Initial structure: {s.natom} atoms in {len(mol_index)} molecules, seed {seed} -> {out / 'initial.xyz'}", flush=True)
    run_info = {
        "method": args.method,
        "init": None if args.init is None else str(args.init),
        "final": None if args.final is None else str(args.final),
        "model": args.model,
        "head": args.head,
        "charge": args.charge,
        "spin": args.spin,
        "seed": seed,
        "molecule": args.molecule,
        "synthesis_dist_bohr": args.synthesis_dist,
        "fragment": args.fragment,
        "min_contact_angstrom": args.min_contact,
        "para": path_para.to_dict(),
    }
    (out / "run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        return 0

    pes = load_pes(args)
    report_model(args, pes, s)
    if args.method == "attractive":
        res = attractive_path_sampling(AttractivePot(s, System.read_xyz(args.final), path_para, out), pes)
    else:
        res = synthesis_path_sampling(SynthesisPot(s, mol_index, path_para, out), pes)
    run_info["result"] = {
        "status": res.status,
        "steps": res.steps,
        "bias_removed_at": res.bias_removed_at,
        "pot_initial": res.pot_initial,
        "pot_max": res.pot_max,
        "pot_min": res.pot_min,
        "pot_final": res.final_state.pot,
    }
    (out / "run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    print(f"Finished: {res.status} after {res.steps} steps", flush=True)
    return 0
