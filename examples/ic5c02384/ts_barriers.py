"""Transition states and barriers of the ic5c02384 reactions on the DeepMD PES.

For every reaction with an SI transition state: start from the SI (omegaB97X-D/cc-pVDZ) TS,
take the lowest Hessian eigenvector as the initial mode, refine the saddle point with the ASE
dimer method, and verify it has exactly one imaginary frequency. The reactants and product are
relaxed with DeepMD. Reports DeepMD barriers / reaction energies next to the SI values.

Usage: python ts_barriers.py --model DPA-3.2-5M.pt --out ts.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ase import Atoms
try:  # ASE >= 3.23
    from ase.mep.dimer import DimerControl, MinModeAtoms, MinModeTranslate
except ImportError:
    from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
from ase.optimize import BFGS

from rtipdeepmd.ase_calculator import DeepmdCalculator
from rtipdeepmd.deepmd_pes import DeepmdPES

HERE = Path(__file__).resolve().parent
EV_TO_KCAL = 23.0605478
AMU_EV_A2_TO_CM1 = 521.47090  # sqrt(eV / (A^2 amu)) in cm^-1


def read_xyz(path: Path) -> Atoms:
    lines = path.read_text().splitlines()
    n = int(lines[0])
    body = [l.split() for l in lines[2 : 2 + n]]
    return Atoms([b[0] for b in body], positions=[[float(x) for x in b[1:4]] for b in body])


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    return float(np.sqrt(np.mean(np.sum((a @ (u @ np.diag([1.0, 1.0, d]) @ vt) - b) ** 2, axis=1))))


def hessian(atoms: Atoms, h: float = 0.005) -> np.ndarray:
    x0 = atoms.positions.copy()
    n = 3 * len(atoms)
    hess = np.zeros((n, n))
    for k in range(n):
        for sign in (1, -1):
            x = x0.copy()
            x.flat[k] += sign * h
            atoms.positions = x
            hess[k] -= sign * atoms.get_forces().ravel() / (2 * h)
    atoms.positions = x0
    return 0.5 * (hess + hess.T)


def frequencies(atoms: Atoms, hess: np.ndarray) -> np.ndarray:
    m = np.repeat(atoms.get_masses(), 3)
    w = np.linalg.eigvalsh(hess / np.sqrt(np.outer(m, m)))
    return np.sign(w) * np.sqrt(np.abs(w)) * AMU_EV_A2_TO_CM1


def refine_ts(atoms: Atoms, fmax: float) -> tuple[bool, float, np.ndarray]:
    """Dimer search from ``atoms`` along the lowest Hessian mode; returns (converged, E, frequencies)."""
    _, v = np.linalg.eigh(hessian(atoms))
    control = DimerControl(initial_eigenmode_method="displacement", displacement_method="vector", logfile=None)
    d_atoms = MinModeAtoms(atoms, control)
    d_atoms.displace(displacement_vector=0.01 * v[:, 0].reshape(-1, 3))
    converged = bool(MinModeTranslate(d_atoms, trajectory=None, logfile=None).run(fmax=fmax, steps=1000))
    atoms.positions = d_atoms.get_positions()
    return converged, atoms.get_potential_energy(), frequencies(atoms, hessian(atoms))


def read_path(run: Path) -> tuple[list[Atoms], np.ndarray]:
    frames, cur, types = [], [], []
    for line in (run / "rtip.pdb").read_text().splitlines():
        if line.startswith("REMARK"):
            cur, types = [], []
        elif line.startswith("ATOM"):
            types.append(line[12:16].strip())
            cur.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif line.startswith("END"):
            frames.append(Atoms(types, positions=cur))
    log = [l.split() for l in (run / "rtip.out").read_text().splitlines()[1:] if not l.startswith("#")]
    return frames, np.array([float(x[2]) for x in log])


def ts_from_paths(runs: list[Path], calc, fmax: float) -> list[dict]:
    """Refine the highest-energy point of each RTIP path into a DeepMD transition state."""
    results, e_sep = [], {}
    for run in runs:
        rdir = HERE / "reactions" / run.parent.name
        if rdir.name not in e_sep:
            e = 0.0
            for name in ("1.xyz", "2.xyz"):
                atoms = read_xyz(rdir / name)
                atoms.calc = calc
                e += relax(atoms, fmax)
            e_sep[rdir.name] = e
        frames, pot = read_path(run)
        k = int(np.argmax(pot)) + 1          # frame 0 is the initial structure (step 0)
        atoms = frames[min(k, len(frames) - 1)]
        atoms.calc = calc
        converged, e_ts, freq = refine_ts(atoms, fmax)
        row = {
            "run": str(run),
            "path_max_kcal": (pot.max() * 27.211386 - e_sep[rdir.name]) * EV_TO_KCAL,
            "barrier_dp_kcal": (e_ts - e_sep[rdir.name]) * EV_TO_KCAL,
            "ts_converged": converged,
            "n_imaginary": int(np.sum(freq < -50.0)),
            "imag_freq_cm1": float(freq[0]),
        }
        ts_ref = rdir / "ts_ordered.xyz"
        if ts_ref.exists():
            row["ts_rmsd_to_si_A"] = kabsch_rmsd(atoms.positions, read_xyz(ts_ref).positions)
        results.append(row)
        print(json.dumps(row), flush=True)
        atoms.write(run / "ts_dp.xyz")
    return results


def relax(atoms: Atoms, fmax: float) -> float:
    BFGS(atoms, logfile=None).run(fmax=fmax, steps=2000)
    return atoms.get_potential_energy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", default="OMol25")
    ap.add_argument("--fmax", type=float, default=0.01)
    ap.add_argument("--only", default=None, help="Only reactions whose name contains this text.")
    ap.add_argument("--from-path", nargs="+", type=Path, default=None,
                    help="RTIP run directories (runs/<method>/<reaction>/s*): refine the path maximum instead of the SI TS.")
    ap.add_argument("--out", default="ts_barriers.json")
    args = ap.parse_args()

    calc = DeepmdCalculator(DeepmdPES.load(args.model, head=args.head))
    if args.from_path:
        Path(args.out).write_text(json.dumps(ts_from_paths(args.from_path, calc, args.fmax), indent=1) + "\n", encoding="utf-8")
        return
    paths = {p.stem.split("_", 1)[1]: p for p in (HERE / "structures").rglob("*.xyz")}
    with open(HERE / "manifests" / "manifest.csv", encoding="utf-8") as f:
        e_si = {r["name"]: float(r["energy_hartree"]) for r in csv.DictReader(f)}
    with open(HERE / "manifests" / "reaction_manifest.csv", encoding="utf-8") as f:
        reactions = [r for r in csv.DictReader(f) if r["transition_state"]]
    if args.only:
        reactions = [r for r in reactions if args.only in r["reaction"]]

    e_min: dict[str, float] = {}

    def minimum(name: str) -> float:
        if name not in e_min:
            atoms = read_xyz(paths[name])
            atoms.calc = calc
            e_min[name] = relax(atoms, args.fmax)
        return e_min[name]

    results = []
    for r in reactions:
        r1, r2, p, ts = r["reactant_1"], r["reactant_2"], r["product"], r["transition_state"]
        atoms = read_xyz(paths[ts])
        si_pos = atoms.positions.copy()
        atoms.calc = calc
        converged, e_ts, freq = refine_ts(atoms, args.fmax)
        e_ref = minimum(r1) + minimum(r2)
        row = {
            "reaction": r["reaction"],
            "barrier_si_kcal": (e_si[ts] - e_si[r1] - e_si[r2]) * 627.509474,
            "barrier_dp_kcal": (e_ts - e_ref) * EV_TO_KCAL,
            "dE_si_kcal": (e_si[p] - e_si[r1] - e_si[r2]) * 627.509474,
            "dE_dp_kcal": (minimum(p) - e_ref) * EV_TO_KCAL,
            "ts_converged": converged,
            "n_imaginary": int(np.sum(freq < -50.0)),
            "imag_freq_cm1": float(freq[0]),
            "ts_rmsd_to_si_A": kabsch_rmsd(atoms.positions, si_pos),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        (HERE / "ts_geometries").mkdir(exist_ok=True)
        atoms.write(HERE / "ts_geometries" / f"{ts}.dp.xyz")
    Path(args.out).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
