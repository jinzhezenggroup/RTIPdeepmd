"""DeepMD reaction and activation Gibbs energies of the 1-R reactions, next to the paper's values.

Minima are relaxed with DeepMD from the SI geometries; transition states are the DeepMD saddle
points written by ts_barriers.py (ts_geometries/<TS>.dp.xyz). Harmonic frequencies come from a
finite-difference DeepMD Hessian with translations and rotations projected out, and G(298.15 K,
1 atm) from the ideal-gas / rigid-rotor / harmonic-oscillator model, as in the paper's Gaussian
frequency calculations. The paper's Gibbs energies (Table 1 and the barriers quoted in the text)
are read from manifests/paper_values.csv.

Usage: python thermo.py --model DPA-3.2-5M.pt --out thermo.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ase.thermochemistry import IdealGasThermo

from rtipdeepmd.ase_calculator import DeepmdCalculator
from rtipdeepmd.deepmd_pes import DeepmdPES
from ts_barriers import AMU_EV_A2_TO_CM1, EV_TO_KCAL, HERE, hessian, read_xyz, relax

CM1_TO_EV = 1.239841984e-4
SYMMETRY_NUMBER = {"N2": 2, "CO2": 2, "CS2": 2, "H2CO": 2, "MeCN": 3}


def projected_frequencies(atoms, hess: np.ndarray) -> tuple[np.ndarray, bool]:
    """Vibrational frequencies (cm^-1, imaginary < 0) without translations/rotations; and linearity."""
    m3 = np.repeat(atoms.get_masses(), 3)
    x = atoms.positions - atoms.get_center_of_mass()
    vecs = []
    for k in range(3):
        t = np.zeros_like(x)
        t[:, k] = 1.0
        vecs.append(t.ravel())
        vecs.append(np.cross(np.eye(3)[k], x).ravel())
    u, s, _ = np.linalg.svd(np.array(vecs).T * np.sqrt(m3)[:, None], full_matrices=False)
    basis = u[:, s > 1e-6 * s.max()]
    proj = np.eye(len(m3)) - basis @ basis.T
    w, _ = np.linalg.eigh(proj @ (hess / np.sqrt(np.outer(m3, m3))) @ proj)
    w = np.delete(w, np.argsort(np.abs(w))[: basis.shape[1]])
    return np.sign(w) * np.sqrt(np.abs(w)) * AMU_EV_A2_TO_CM1, basis.shape[1] == 5


def gibbs(atoms, name: str, ts: bool, temperature: float) -> dict:
    """Electronic energy and G = E + G_corr (eV) of a stationary point."""
    freq, linear = projected_frequencies(atoms, hessian(atoms))
    n_imag, lowest = int(np.sum(freq < -50.0)), float(freq[0])
    if ts:
        freq = freq[1:]  # the reaction mode (lowest eigenvalue)
    vib = np.abs(freq) * CM1_TO_EV  # residual small imaginary modes are treated as real, as Gaussian prints them
    thermo = IdealGasThermo(
        vib_energies=vib,
        geometry="linear" if linear else "nonlinear",
        potentialenergy=0.0,
        atoms=atoms,
        symmetrynumber=SYMMETRY_NUMBER.get(name, 1),
        spin=0,
        vib_selection="all",  # translations/rotations (and the TS reaction mode) are already removed
    )
    e = atoms.get_potential_energy()
    g_corr = thermo.get_gibbs_energy(temperature=temperature, pressure=101325.0, verbose=False)
    return {"E_eV": e, "G_eV": e + g_corr, "n_imaginary": n_imag, "lowest_freq_cm1": lowest}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", default="OMol25")
    ap.add_argument("--fmax", type=float, default=0.001, help="Relaxation threshold for minima (eV/A).")
    ap.add_argument("--temperature", type=float, default=298.15)
    ap.add_argument("--out", default="thermo.json")
    args = ap.parse_args()

    calc = DeepmdCalculator(DeepmdPES.load(args.model, head=args.head))
    paths = {p.stem.split("_", 1)[1]: p for p in (HERE / "structures").rglob("*.xyz")}
    with open(HERE / "manifests" / "paper_values.csv", encoding="utf-8") as f:
        paper = {r["reaction"]: r for r in csv.DictReader(f)}
    with open(HERE / "manifests" / "reaction_manifest.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["reactant_1"].startswith("1-")]
    reactions: dict[str, dict] = {}
    for r in rows:  # merge the product row and the transition-state row of each reaction
        reactions.setdefault(r["reaction"], dict(r)).update({k: v for k, v in r.items() if v})

    species: dict[str, dict] = {}

    def point(name: str, ts: bool = False) -> dict:
        if name not in species:
            if ts:
                atoms = read_xyz(HERE / "ts_geometries" / f"{name}.dp.xyz")
                atoms.calc = calc
            else:
                atoms = read_xyz(paths[name])
                atoms.calc = calc
                relax(atoms, args.fmax)
            species[name] = gibbs(atoms, name, ts, args.temperature)
            print(name, json.dumps(species[name]), flush=True)
        return species[name]

    results = []
    for name, r in reactions.items():
        ref = [point(r["reactant_1"]), point(r["reactant_2"])]
        e0, g0 = sum(p["E_eV"] for p in ref), sum(p["G_eV"] for p in ref)
        p = point(r["product"])
        row = {"reaction": name, "dE_dp": (p["E_eV"] - e0) * EV_TO_KCAL, "dG_dp": (p["G_eV"] - g0) * EV_TO_KCAL}
        if r.get("transition_state"):
            t = point(r["transition_state"], ts=True)
            row.update({"dE_ts_dp": (t["E_eV"] - e0) * EV_TO_KCAL, "dG_ts_dp": (t["G_eV"] - g0) * EV_TO_KCAL})
        for key in ("dG_paper", "dG_ts_paper", "dE_ts_paper_tz"):
            if paper.get(name, {}).get(key):
                row[key] = float(paper[name][key])
        results.append(row)
        print(json.dumps(row), flush=True)
    Path(args.out).write_text(
        json.dumps({"temperature_K": args.temperature, "species": species, "reactions": results}, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
