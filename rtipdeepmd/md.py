"""RTIP-driven NVT MD for molecular evolution.

Line-by-line port of ``impl RtipNVTMD for EvolutionPot`` in the upstream Rust RTIP-MD
``src/pes_exploration/md.rs``.

Each step, every molecule is shifted by 1% of the vector from its geometric centre to the
centre of all molecules; this contracted structure is the centre of an RTIP Gaussian whose
width is the RTI distance to it. The Gaussian height ``a_min`` is negative (attractive), so the
bias pulls the molecules together. ``a_min`` is driven by a three-state machine:

* Increasing: a_min -= a0                          (deepen while searching)
* Decreasing: a_min += decreasing_multiple * a0    (after a bond formed or broke, release the bias
                                                    until a_min > decreasing_bound * a_min_at_change,
                                                    then re-detect bonds/molecules and search again)
* Falling:    a_min += a0                          (two molecules are already in contact)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import typing as t

import numpy as np

from .constants import BOLTZMANN, FEMTOSECOND_TO_AU, HARTREE_TO_JOULE, atomic_radius
from .params import Para
from .rtip import rti_dist, rti_pot_force
from .system import System, dist_mat, get_adj_mat, judge_adj_of_mol, judge_variation_of_bonding, split_into_mol

# Hard-coded multiples of (r_i + r_j) used by the Rust implementation.
BOND_MULTIPLE = 1.25        # bonded / same molecule
FORMED_MULTIPLE = 1.0       # an originally unbonded pair closer than this has formed a bond
BROKEN_MULTIPLE = 1.6       # an originally bonded pair farther than this has broken
CONTACT_MULTIPLE = 1.2      # two molecules closer than this are in contact
EVOLUTION_SHIFT = 0.01      # fraction of the way each molecule is moved toward the common centre


class PES(t.Protocol):
    def get_energy_force(self, s: System) -> tuple[float, np.ndarray]:
        """Return the energy (Hartree) and atomic forces (Hartree/Bohr)."""


class RtipStatus(Enum):
    Increasing = "Increasing"
    Decreasing = "Decreasing"
    Falling = "Falling"


@dataclass
class EvolutionPot:
    initial_state: System
    para: Para
    output_path: str | Path
    initial_velocity: np.ndarray | None = None  # Bohr per atomic time unit


def evolution_target(coord: np.ndarray, mol_index: list[list[int]]) -> np.ndarray:
    """Shift each molecule by ``EVOLUTION_SHIFT`` of the vector to the common geometric centre."""
    o_all = coord.mean(axis=0)
    target = coord.copy()
    for mol in mol_index:
        target[mol] += EVOLUTION_SHIFT * (o_all - coord[mol].mean(axis=0))
    return target


def kinetic_temperature(masses: np.ndarray, vel: np.ndarray) -> tuple[float, float]:
    """Return (kinetic energy in Hartree, temperature in K) with 3(N-1) degrees of freedom."""
    mv2 = float(np.sum(masses * vel * vel))
    temp = mv2 * HARTREE_TO_JOULE / (BOLTZMANN * 3.0 * (len(masses) - 1))
    return 0.5 * mv2, temp


HEADER = (
    "  step            time        rti_dist            temp             kin        pot_real"
    "        pot_rtip          f_real          f_rtip     rtip_status           a_min\n"
)


def evolution_nvt_md(pot: EvolutionPot, real_pes: PES, stop_temp: float | None = None) -> System:
    """Run RTIP evolution MD and write ``rtip.pdb``, ``rtip.out`` and ``rtip_decreasing_steps``.

    ``stop_temp`` (K) optionally aborts a run whose temperature has run away; it is not part of
    the Rust algorithm. Returns the final structure.
    """
    para = pot.para
    out = Path(pot.output_path)
    out.mkdir(parents=True, exist_ok=True)

    s = pot.initial_state.clone()
    idx = np.arange(s.natom) if s.atom_add_pot is None else np.asarray(s.atom_add_pot)
    frag_type = [s.atom_type[k] for k in idx]
    radii = np.array([atomic_radius(e) for e in frag_type])
    masses = s.get_atomic_masses()[:, None]

    dist = dist_mat(s.coord[idx])
    adj_mat = get_adj_mat(frag_type, radii, dist, BOND_MULTIPLE, para.ignored_pair)
    mol_index = split_into_mol(radii, dist, BOND_MULTIPLE)

    a_min = 0.0
    a_min_threshold = 0.0
    status = RtipStatus.Increasing
    vel = np.zeros_like(s.coord) if pot.initial_velocity is None else np.array(pot.initial_velocity, dtype=np.float64)
    acc = np.zeros_like(s.coord)
    force_rtip = np.zeros_like(s.coord)
    dt = para.dt
    time = 0.0

    with (
        open(out / "rtip.pdb", "w", encoding="utf-8") as pdb,
        open(out / "rtip.out", "w", encoding="utf-8") as log,
        open(out / "rtip_decreasing_steps", "w", encoding="utf-8") as dec,
    ):
        pdb.write("TITLE     PDB file created by RTIP\n")
        pdb.write(s.pdb_frame(0))
        log.write(HEADER)
        dec.write("      begin_step        end_step\n")

        for i in range(1, para.max_step + 1):
            # Control the variation of the RTIP
            if status != RtipStatus.Decreasing and judge_variation_of_bonding(
                radii, dist, adj_mat, FORMED_MULTIPLE, BROKEN_MULTIPLE
            ):
                a_min_threshold = a_min * para.decreasing_bound
                status = RtipStatus.Decreasing
                dec.write(f"{i:16d}")
            if status == RtipStatus.Decreasing and a_min > a_min_threshold:
                adj_mat = get_adj_mat(frag_type, radii, dist, BOND_MULTIPLE, para.ignored_pair)
                mol_index = split_into_mol(radii, dist, BOND_MULTIPLE)
                status = RtipStatus.Increasing
                dec.write(f"{i - 1:16d}\n")
            if para.split_step and i % para.split_step == 0:
                mol_index = split_into_mol(radii, dist, BOND_MULTIPLE)

            if status == RtipStatus.Increasing:
                a_min -= para.a0
            elif status == RtipStatus.Decreasing:
                a_min += para.a0 * para.decreasing_multiple
            else:
                a_min += para.a0

            # First half of leapfrog
            time += dt
            vel += 0.5 * dt * FEMTOSECOND_TO_AU * acc
            s.coord += dt * FEMTOSECOND_TO_AU * vel

            # Berendsen thermostat scaling factor from the half-step velocities
            _, temp = kinetic_temperature(masses, vel)
            temp = max(temp, 1.0)
            lam = np.sqrt(1.0 + (dt / para.tau) * (para.temp_bath / temp - 1.0))

            # Real PES
            pot_real, force_real = real_pes.get_energy_force(s)
            s.pot = pot_real
            f_real = float(np.sqrt(np.sum(force_real * force_real)))

            # RTIP
            frag = s.coord[idx]
            dist = dist_mat(frag)
            if status != RtipStatus.Decreasing:
                in_contact = judge_adj_of_mol(mol_index, radii, adj_mat, dist, CONTACT_MULTIPLE)
                status = RtipStatus.Falling if in_contact else RtipStatus.Increasing
            final_state = evolution_target(frag, mol_index)
            rtip_dist = rti_dist(final_state, frag)
            pot_rtip, force_frag = rti_pot_force(final_state, frag, a_min, rtip_dist)
            force_rtip[idx] = force_frag
            f_rtip = float(np.sqrt(np.sum(force_frag * force_frag)))

            # Second half of leapfrog, then apply the thermostat
            acc = (force_real + force_rtip) / masses
            vel += 0.5 * dt * FEMTOSECOND_TO_AU * acc
            vel *= lam
            kin, temp = kinetic_temperature(masses, vel)

            if i % para.print_step == 0:
                pdb.write(s.pdb_frame(i))
            log.write(
                f"{i:6d} {time:15.8f} {rtip_dist:15.8f} {temp:15.8f} {kin:15.8f} {pot_real:15.8f} "
                f"{pot_rtip:15.8f} {f_real:15.8f} {f_rtip:15.8f} {status.value:>15} {a_min:15.8f}\n"
            )
            log.flush()
            dec.flush()

            if not (np.isfinite(pot_real) and np.isfinite(temp)):
                log.write(f"# STOP step={i}: non-finite energy or temperature\n")
                break
            if stop_temp is not None and temp > stop_temp:
                log.write(f"# STOP step={i}: temperature {temp:.1f} K exceeds --stop-temp {stop_temp:.1f} K\n")
                break

    return s
