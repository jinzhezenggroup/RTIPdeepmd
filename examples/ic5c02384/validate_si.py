"""DeepMD vs the paper's DFT on the ic5c02384 SI structures.

Reference: SI electronic energies (omegaB97X-D/cc-pVDZ geometries and energies, Gaussian 16).
For every reaction in manifests/reaction_manifest.csv this compares the reaction energy
dE = E(P) - E(R1) - E(R2) and, where a TS is given, the barrier E(TS) - E(R1) - E(R2),
both at the SI geometries ("vertical"). With --relax, minima are also relaxed with DeepMD and
the adiabatic reaction energies and the RMSD to the SI geometries are reported.

Usage: python validate_si.py --model DPA-3.2-5M.pt [--head OMol25 --head Default] [--relax] --out si.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rtipdeepmd.deepmd_pes import DeepmdPES
from rtipdeepmd.system import System

HARTREE_TO_KCAL = 627.509474
HERE = Path(__file__).resolve().parent


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(np.mean(np.sum((a @ rot - b) ** 2, axis=1))))


def load_structures() -> tuple[dict[str, Path], dict[str, float]]:
    paths = {p.stem.split("_", 1)[1]: p for p in (HERE / "structures").rglob("*.xyz")}
    with open(HERE / "manifests" / "manifest.csv", encoding="utf-8") as f:
        e_si = {r["name"]: float(r["energy_hartree"]) for r in csv.DictReader(f)}
    return paths, e_si


def relax(pes: DeepmdPES, s: System, fmax: float, steps: int) -> tuple[float, np.ndarray, bool]:
    from ase import Atoms
    from ase.optimize import BFGS

    from rtipdeepmd.ase_calculator import DeepmdCalculator
    from rtipdeepmd.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, EV_TO_HARTREE

    atoms = Atoms(s.atom_type, positions=s.coord * BOHR_TO_ANGSTROM)
    atoms.calc = DeepmdCalculator(pes)
    converged = bool(BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps))
    return atoms.get_potential_energy() * EV_TO_HARTREE, atoms.positions * ANGSTROM_TO_BOHR, converged


def stats(err: list[float]) -> dict:
    e = np.asarray(err)
    return {"n": int(e.size), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e**2))), "max_abs": float(np.max(np.abs(e)))}


def spearman(x: list[float], y: list[float]) -> float:
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluate(pes: DeepmdPES, paths: dict[str, Path], e_si: dict[str, float], do_relax: bool, fmax: float) -> dict:
    from rtipdeepmd.deepmd_pes import EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR

    with open(HERE / "manifests" / "reaction_manifest.csv", encoding="utf-8") as f:
        reactions = list(csv.DictReader(f))
    needed = sorted({r[k] for r in reactions for k in ("reactant_1", "reactant_2", "product", "transition_state") if r[k]})

    e_dp, fmax_si, relaxed = {}, {}, {}
    for name in needed:
        s = System.read_xyz(paths[name])
        e, force = pes.get_energy_force(s)
        e_dp[name] = e
        fmax_si[name] = float(np.max(np.linalg.norm(force, axis=1)) / EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR)
        if do_relax and not name.endswith(".TS"):
            e_rel, coord, ok = relax(pes, s, fmax, 1000)
            relaxed[name] = {"energy": e_rel, "rmsd_angstrom": kabsch_rmsd(coord, s.coord) * 0.52917720859, "converged": ok}

    rows = []
    for r in reactions:
        r1, r2, p, ts = r["reactant_1"], r["reactant_2"], r["product"], r["transition_state"]
        row = {"reaction": r["reaction"], "class": r["class"]}
        ref = e_si[r1] + e_si[r2]
        mod = e_dp[r1] + e_dp[r2]
        row["dE_si"] = (e_si[p] - ref) * HARTREE_TO_KCAL
        row["dE_dp"] = (e_dp[p] - mod) * HARTREE_TO_KCAL
        if ts:
            row["barrier_si"] = (e_si[ts] - ref) * HARTREE_TO_KCAL
            row["barrier_dp"] = (e_dp[ts] - mod) * HARTREE_TO_KCAL
        if relaxed:
            row["dE_dp_relaxed"] = (relaxed[p]["energy"] - relaxed[r1]["energy"] - relaxed[r2]["energy"]) * HARTREE_TO_KCAL
        rows.append(row)

    co2 = [x for x in rows if x["class"] == "CO2_product"]
    tsr = [x for x in rows if "barrier_si" in x]
    summary = {
        "reaction_energy_vertical": stats([x["dE_dp"] - x["dE_si"] for x in co2 + [x for x in tsr if x["class"] != "CO2_product"]]),
        "barrier_vertical": stats([x["barrier_dp"] - x["barrier_si"] for x in tsr]),
        "spearman_dE_CO2_products": spearman([x["dE_si"] for x in co2], [x["dE_dp"] for x in co2]),
        "spearman_barrier": spearman([x["barrier_si"] for x in tsr], [x["barrier_dp"] for x in tsr]),
        "max_force_at_SI_minima_eV_per_A": stats([v for k, v in fmax_si.items() if not k.endswith(".TS")]),
    }
    if relaxed:
        summary["reaction_energy_relaxed"] = stats([x["dE_dp_relaxed"] - x["dE_si"] for x in rows if x["class"] == "CO2_product"])
        summary["rmsd_relaxed_vs_SI_angstrom"] = stats([v["rmsd_angstrom"] for v in relaxed.values()])
        summary["relax_not_converged"] = [k for k, v in relaxed.items() if not v["converged"]]
    return {"summary": summary, "reactions": rows, "max_force_at_SI": fmax_si, "relaxed": relaxed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", action="append", default=None, help="Model head(s) to evaluate (default: OMol25).")
    ap.add_argument("--relax", action="store_true")
    ap.add_argument("--fmax", type=float, default=0.01, help="eV/Angstrom")
    ap.add_argument("--out", default="si_validation.json")
    args = ap.parse_args()

    paths, e_si = load_structures()
    result = {}
    for head in args.head or ["OMol25"]:
        pes = DeepmdPES.load(args.model, head=None if head == "Default" else head)
        result[head] = evaluate(pes, paths, e_si, args.relax, args.fmax)
        print(f"== head {head}")
        print(json.dumps(result[head]["summary"], indent=1))
        print(f"{'reaction':38s} {'dE_SI':>7s} {'dE_DP':>7s} {'TS_SI':>7s} {'TS_DP':>7s}")
        for x in result[head]["reactions"]:
            if x["class"] == "CO2_product" and not x["reaction"].startswith("1-"):
                continue
            print(f"{x['reaction']:38s} {x['dE_si']:7.1f} {x['dE_dp']:7.1f} {x.get('barrier_si', float('nan')):7.1f} {x.get('barrier_dp', float('nan')):7.1f}")
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
