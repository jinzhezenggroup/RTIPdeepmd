"""Analyse RTIP evolution runs of 1-R + small molecule (N2 coupling, ic5c02384).

Atom roles are detected from the first molecule (the methyleneborane-N2 complex 1-R):
B, N_alpha (the N2 atom bonded to B) and N_beta (terminal N). The second molecule is the small
molecule (CO2, CS2, H2CO, MeCN, MeCH=NMe). A frame counts as coupled when N_beta binds the
small molecule's carbon and B binds one of its heteroatoms (both shorter than 1.2 times the sum
of the covalent radii, the Rust contact criterion), i.e. the B-N=N-C-X-B ring of the paper's
products.

With --relax, the last frame of each coupled run is relaxed with DeepMD and its bond lengths
(the columns of the paper's Table 1) and the reaction energy relative to the DeepMD-relaxed
separated reactants are reported.

Usage: python analyze_rtip.py --model DPA-3.2-5M.pt [--relax] RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rtipdeepmd.cli import parse_molecules
from rtipdeepmd.constants import BOHR_TO_ANGSTROM, atomic_radius

HARTREE_TO_KCAL = 627.509474
COUPLED = 1.2  # bonded when d < COUPLED * (r_i + r_j)


def bond_cutoff(a: str, b: str) -> float:
    return COUPLED * (atomic_radius(a) + atomic_radius(b)) * BOHR_TO_ANGSTROM


def read_pdb_frames(path: Path) -> tuple[list[str], list[int], np.ndarray]:
    steps, frames, types, cur = [], [], [], []
    for line in path.read_text().splitlines():
        if line.startswith("REMARK"):
            steps.append(int(line.split("Step =")[1].split(",")[0]))
            types, cur = [], []
        elif line.startswith("ATOM"):
            types.append(line[12:16].strip())
            cur.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif line.startswith("END"):
            frames.append(np.array(cur))
    return types, steps, np.array(frames)


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    n = int(lines[0])
    body = [l.split() for l in lines[2 : 2 + n]]
    return [b[0] for b in body], np.array([[float(x) for x in b[1:4]] for b in body])


def roles(types: list[str], coord: np.ndarray, n1: int) -> dict:
    """Indices of B, N_alpha, N_beta (molecule 1) and the small molecule's C and heteroatoms."""
    b = [i for i in range(n1) if types[i] == "B"]
    if len(b) != 1:
        raise ValueError("expected exactly one B atom in the first molecule")
    b = b[0]
    n_idx = [i for i in range(n1) if types[i] == "N"]
    pairs = [(i, j) for i in n_idx for j in n_idx if i < j and np.linalg.norm(coord[i] - coord[j]) < 1.30]
    # the N2 unit: the N-N pair one of whose atoms is closest to B
    i, j = min(pairs, key=lambda p: min(np.linalg.norm(coord[p[0]] - coord[b]), np.linalg.norm(coord[p[1]] - coord[b])))
    n_alpha, n_beta = (i, j) if np.linalg.norm(coord[i] - coord[b]) < np.linalg.norm(coord[j] - coord[b]) else (j, i)
    mol2 = list(range(n1, len(types)))
    carbons = [k for k in mol2 if types[k] == "C"]
    hetero = [k for k in mol2 if types[k] in ("O", "S", "N")]
    return {"B": b, "N_alpha": n_alpha, "N_beta": n_beta, "C": carbons, "X": hetero}


def frame_metrics(coord: np.ndarray, r: dict) -> dict:
    d = lambda a, b: float(np.linalg.norm(coord[a] - coord[b]))
    nc = min((d(r["N_beta"], c), c) for c in r["C"])
    bx = min((d(r["B"], x), x) for x in r["X"])
    return {"N_beta-C": nc[0], "C": nc[1], "B-X": bx[0], "X": bx[1], "B-N_alpha": d(r["B"], r["N_alpha"]), "N-N": d(r["N_alpha"], r["N_beta"])}


def table1_bonds(coord: np.ndarray, r: dict) -> dict:
    m = frame_metrics(coord, r)
    d = lambda a, b: float(np.linalg.norm(coord[a] - coord[b]))
    return {"B-N": m["B-N_alpha"], "N-N": m["N-N"], "B-O": m["B-X"], "C-O": d(m["C"], m["X"]), "C-N2": m["N_beta-C"]}


def relax(pes, types, coord, fmax=0.02, steps=2000):
    from ase import Atoms
    from ase.optimize import BFGS

    from rtipdeepmd.ase_calculator import DeepmdCalculator
    from rtipdeepmd.constants import EV_TO_HARTREE

    atoms = Atoms(types, positions=coord)
    atoms.calc = DeepmdCalculator(pes)
    ok = bool(BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps))
    return atoms.get_potential_energy() * EV_TO_HARTREE, atoms.positions.copy(), ok


_SEPARATED: dict[tuple[str, str], float] = {}


def separated_energy(pes, mol1: Path, mol2: Path) -> float:
    """DeepMD energy of the two relaxed, separated reactants (Hartree)."""
    key = (str(mol1), str(mol2))
    if key not in _SEPARATED:
        _SEPARATED[key] = sum(relax(pes, *read_xyz(path))[0] for path in (mol1, mol2))
    return _SEPARATED[key]


def analyze_run(run: Path, pes=None) -> dict:
    info = json.loads((run / "run.json").read_text())
    method = info.get("method", "evolution")
    (mol1, _), (mol2, _) = parse_molecules(info["molecule"])[:2]
    mol1, mol2 = Path(mol1), Path(mol2)
    n1 = len(read_xyz(mol1)[0])
    types, steps, frames = read_pdb_frames(run / "rtip.pdb")
    r = roles(types, frames[0], n1)
    metrics = [frame_metrics(c, r) for c in frames]
    coupled = [m["N_beta-C"] < bond_cutoff("N", "C") and m["B-X"] < bond_cutoff("B", types[m["X"]]) for m in metrics]
    n2_lost = [m["B-N_alpha"] > 2.2 or m["N-N"] > 1.6 for m in metrics]
    log = [l.split() for l in (run / "rtip.out").read_text().splitlines()[1:] if not l.startswith("#")]
    out = {"run": str(run), "method": method, "seed": info["seed"], "steps": int(steps[-1])}
    if method in ("synthesis", "attractive"):
        pot_real = np.array([float(x[2]) for x in log])
        out["status"] = info.get("result", {}).get("status")
        out["bias_removed_at"] = info.get("result", {}).get("bias_removed_at")
    else:
        pot_real = np.array([float(x[5]) for x in log])
        out["temp_bath"] = info["para"]["temp_bath"]
        out["max_temp"] = float(max(float(x[3]) for x in log))
    out.update(
        {
            "first_coupled_step": next((s for s, c in zip(steps, coupled) if c), None),
            "coupled_at_end": bool(coupled[-1]),
            "n2_lost_at_end": bool(n2_lost[-1]),
            "final_metrics": metrics[-1],
            "min_N_beta-C": float(min(m["N_beta-C"] for m in metrics)),
            "min_B-X": float(min(m["B-X"] for m in metrics)),
        }
    )
    if pes is not None:
        e_sep = separated_energy(pes, mol1, mol2)
        out["path_max_rel_separated_kcal"] = float((pot_real.max() - e_sep) * HARTREE_TO_KCAL)
        out["path_initial_rel_separated_kcal"] = float((pot_real[0] - e_sep) * HARTREE_TO_KCAL)
        if out["coupled_at_end"]:
            e_p, xyz, ok = relax(pes, types, frames[-1])
            out["relaxed_product"] = {
                "energy_hartree": e_p,
                "converged": ok,
                "bonds": table1_bonds(xyz, r),
                "dE_kcal": (e_p - e_sep) * HARTREE_TO_KCAL,
            }
            (run / "relaxed_product.xyz").write_text(
                f"{len(types)}\nDeepMD-relaxed, E = {e_p:.8f} Ha\n"
                + "".join(f"{t:>2} {x:14.8f} {y:14.8f} {z:14.8f}\n" for t, (x, y, z) in zip(types, xyz))
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--model", default=None)
    ap.add_argument("--head", default="OMol25")
    ap.add_argument("--relax", action="store_true")
    ap.add_argument("--out", default="rtip_analysis.json")
    args = ap.parse_args()

    pes = None
    if args.relax:
        from rtipdeepmd.deepmd_pes import DeepmdPES

        pes = DeepmdPES.load(args.model, head=args.head)
    results = []
    for run in args.runs:
        try:
            res = analyze_run(run, pes)
        except Exception as exc:  # keep going for the other runs
            res = {"run": str(run), "error": repr(exc)}
        results.append(res)
        print(json.dumps(res))
    Path(args.out).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
