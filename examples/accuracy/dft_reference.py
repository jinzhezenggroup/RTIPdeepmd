"""PySCF single points (energy + forces) for XYZ structures, as a reference for DeepMD.

Default level: omegaB97M-V/def2-TZVPD, the level of the OMol25 data the DPA-3 OMol25 head was
trained on. Unrestricted for open shells; density fitting for speed.

Usage: python dft_reference.py --charge 0 --spin 2 --out ref.json frame1.xyz frame2.xyz ...
       (--spin is the multiplicity; default: lowest consistent with the electron count)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", nargs="+")
    ap.add_argument("--xc", default="wb97m-v")
    ap.add_argument("--basis", default="def2-tzvpd")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--spin", type=int, default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--memory-mb", type=int, default=64000)
    ap.add_argument("--no-grad", action="store_true")
    ap.add_argument("--out", default="dft_reference.json")
    args = ap.parse_args()

    from pyscf import dft, gto, lib

    from rtipdeepmd.constants import atomic_number

    lib.num_threads(args.threads)
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for path in args.xyz:
        if path in results:
            continue
        lines = Path(path).read_text().splitlines()
        natom = int(lines[0])
        atoms = "\n".join(lines[2 : 2 + natom])
        nelec = sum(atomic_number(line.split()[0]) for line in lines[2 : 2 + natom]) - args.charge
        mult = args.spin if args.spin is not None else (1 if nelec % 2 == 0 else 2)
        mol = gto.M(atom=atoms, basis=args.basis, charge=args.charge, spin=mult - 1, verbose=3, max_memory=args.memory_mb)
        mf = (dft.UKS(mol) if mol.spin else dft.RKS(mol)).density_fit()
        mf.xc = args.xc
        mf.grids.level = 3
        mf.conv_tol = 1e-9
        mf.max_cycle = 200
        t0 = time.time()
        energy = mf.kernel()
        if not mf.converged:
            mf = mf.newton()
            energy = mf.kernel()
        t1 = time.time()
        forces = None
        if not args.no_grad:
            forces = (-mf.nuc_grad_method().kernel()).tolist()   # Hartree/Bohr
        results[path] = {
            "energy_hartree": float(energy),
            "forces_hartree_per_bohr": forces,
            "converged": bool(mf.converged),
            "multiplicity": mult,
            "nao": int(mol.nao),
            "seconds_scf": t1 - t0,
            "seconds_total": time.time() - t0,
            "level": f"{args.xc}/{args.basis}",
        }
        print(path, results[path]["energy_hartree"], f"{results[path]['seconds_total']:.0f}s", flush=True)
        out_path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
