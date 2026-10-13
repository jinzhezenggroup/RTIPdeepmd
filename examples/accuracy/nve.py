"""NVE energy conservation of the DeepMD PES with the RTIP-MD leapfrog integrator.

The RTIP bias (a0 = 0) and the thermostat (tau -> infinity) are switched off; the formose box
starts from Maxwell-Boltzmann velocities. Reports the drift and fluctuation of E_pot + E_kin.

Usage: python nve.py --model DPA-3.2-5M.pt --dt 0.5 --steps 2000 --out nve_dt0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rtipdeepmd.cli import DEFAULT_XYZ_DIR, build_initial_state
from rtipdeepmd.constants import BOLTZMANN, HARTREE_TO_JOULE
from rtipdeepmd.deepmd_pes import DeepmdPES
from rtipdeepmd.md import EvolutionPot, evolution_nvt_md
from rtipdeepmd.params import Para

HARTREE_TO_MEV = 27211.386


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", default="OMol25")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--temp", type=float, default=1500.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="nve")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    s = build_initial_state(DEFAULT_XYZ_DIR, 3.5, rng)
    masses = s.get_atomic_masses()[:, None]
    vel = rng.normal(size=s.coord.shape) * np.sqrt(BOLTZMANN * args.temp / HARTREE_TO_JOULE / masses)
    vel -= np.sum(masses * vel, axis=0) / np.sum(masses)

    para = Para(a0=0.0, tau=1.0e15, dt=args.dt, max_step=args.steps, print_step=args.steps)
    pes = DeepmdPES.load(args.model, head=args.head)
    evolution_nvt_md(EvolutionPot(initial_state=s, para=para, output_path=args.out, initial_velocity=vel), pes)

    rows = [l.split() for l in (Path(args.out) / "rtip.out").read_text().splitlines()[1:] if not l.startswith("#")]
    t = np.array([float(r[1]) for r in rows])
    e_tot = np.array([float(r[5]) + float(r[4]) for r in rows])
    temp = np.array([float(r[3]) for r in rows])
    slope = np.polyfit(t, e_tot, 1)[0]
    result = {
        "dt_fs": args.dt,
        "steps": args.steps,
        "mean_temp_K": float(temp.mean()),
        "drift_meV_per_atom_per_ps": float(slope * 1000.0 * HARTREE_TO_MEV / s.natom),
        "fluct_std_meV_per_atom": float(np.std(e_tot - np.polyval(np.polyfit(t, e_tot, 1), t)) * HARTREE_TO_MEV / s.natom),
        "max_dev_meV_per_atom": float(np.max(np.abs(e_tot - e_tot[0])) * HARTREE_TO_MEV / s.natom),
    }
    print(json.dumps(result, indent=1))
    (Path(args.out) / "nve.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
