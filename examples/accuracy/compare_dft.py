"""Compare DeepMD energies and forces with the PySCF references written by dft_reference.py.

The DPA-3 OMol25 head predicts absolute omegaB97M-V/def2-TZVPD energies, so the absolute energy
error per atom is reported, along with force errors and (with --reaction) reaction energies.

Usage: python compare_dft.py --model DPA-3.2-5M.pt ref.json [--charge 0]
           [--reaction "013_1-tBu+002_CO2>055_1-tBu-CO2.P" ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rtipdeepmd.deepmd_pes import EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR, DeepmdPES
from rtipdeepmd.system import System

HARTREE_TO_MEV = 27211.386
HARTREE_TO_KCAL = 627.509474


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", default="OMol25")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--reaction", action="append", default=[], help='"A+B>C" with structure basenames')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pes = DeepmdPES.load(args.model, head=args.head, charge=args.charge)
    rows, energy = [], {}
    for ref_file in args.refs:
        for path, ref in json.loads(Path(ref_file).read_text()).items():
            s = System.read_xyz(path)
            e_dp, f_dp = pes.get_energy_force(s)
            name = Path(path).stem
            energy[name] = (ref["energy_hartree"], e_dp)
            row = {"name": name, "natom": s.natom, "e_dft": ref["energy_hartree"], "e_dp": e_dp,
                   "de_meV_per_atom": (e_dp - ref["energy_hartree"]) * HARTREE_TO_MEV / s.natom}
            if ref.get("forces_hartree_per_bohr") is not None:
                f_ref = np.asarray(ref["forces_hartree_per_bohr"]) / EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR
                f_mod = f_dp / EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR
                diff = f_mod - f_ref
                row.update({
                    "force_rmse_eV_A": float(np.sqrt(np.mean(diff**2))),
                    "force_max_err_eV_A": float(np.max(np.linalg.norm(diff, axis=1))),
                    "force_ref_rms_eV_A": float(np.sqrt(np.mean(f_ref**2))),
                    "force_cosine": float(np.sum(f_mod * f_ref) / np.linalg.norm(f_mod) / np.linalg.norm(f_ref)),
                })
            rows.append(row)
            print(json.dumps(row), flush=True)

    reactions = []
    for spec in args.reaction:
        lhs, rhs = spec.split(">")
        terms = [(1, n) for n in rhs.split("+")] + [(-1, n) for n in lhs.split("+")]
        if not all(n in energy for _, n in terms):
            continue
        d_dft = sum(c * energy[n][0] for c, n in terms) * HARTREE_TO_KCAL
        d_dp = sum(c * energy[n][1] for c, n in terms) * HARTREE_TO_KCAL
        reactions.append({"reaction": spec, "dft_kcal": d_dft, "dp_kcal": d_dp, "err_kcal": d_dp - d_dft})
        print(json.dumps(reactions[-1]), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"structures": rows, "reactions": reactions}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
