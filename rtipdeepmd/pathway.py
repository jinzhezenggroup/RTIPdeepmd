"""RTIP pathway sampling: synthesis of several molecules, and attraction toward a known final state.

Port of the upstream Rust RTIP-MD ``impl RtipPathSampling for SynthesisPot`` and
``impl RtipPathSampling for AttractivePot``
(``src/pes_exploration/pathway_sampling.rs``), ``min_1d`` / ``min_1d_real_bias``
(``optimization.rs``) and ``synthesis`` (``synthesis.rs``).

Each step is a line search along the total force of real PES + attractive RTIP. As in the
evolution MD, the RTIP centre is the current structure with every molecule moved 1% toward the
common centre and its width is the RTI distance to it; its height is a_min = -a0 * step. When the
real energy has dropped by ``pot_drop`` below its maximum (while still above the initial energy)
the barrier is taken as crossed: the RTIP is removed and the structure is relaxed on the real
PES until f_real / sqrt(N) < ``f_epsilon``. The sampling fails when the real energy climbs more
than ``pot_climb`` above its minimum or the RTIP force exceeds 1000.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import typing as t

import numpy as np

from .constants import GOLDEN_RATIO1, GOLDEN_RATIO2
from .md import evolution_target
from .rtip import rti_dist, rti_pot, rti_pot_force
from .system import System, dist_mat, rand_rot


class PES(t.Protocol):
    def get_energy(self, s: System) -> float: ...

    def get_energy_force(self, s: System) -> tuple[float, np.ndarray]: ...


@dataclass
class PathPara:
    """Defaults equal Rust RTIP-MD ``Para::new()`` (``rtip_para.a0`` and ``path_samp_para``)."""

    a0: float = 0.0005            # Hartree; RTIP height is -a0 * step
    pot_climb: float = 0.185      # Hartree; fail if E_real > E_min + pot_climb
    pot_drop: float = 0.02        # Hartree; barrier crossed if E_real < E_max - pot_drop (and > E_initial)
    pot_epsilon: float = 0.00005  # Hartree per atom; line-search tolerance
    f_epsilon: float = 0.001      # Hartree/Bohr; converged if f_real / sqrt(N) < f_epsilon
    max_step: int = 2000
    print_step: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


class LineSearchError(RuntimeError):
    """The energy does not decrease along the force (Rust ``error_min_1d`` panic)."""


def min_1d(fun: t.Callable[[float], float], f0: float, epsilon: float) -> tuple[float, float]:
    """Minimise ``fun`` along x >= 0 given f(0) = f0: bracket by extrapolation, then golden section."""
    delta_x = 0.01
    x0 = 0.0
    x1 = delta_x
    f1 = fun(x1)
    while f1 > f0 and delta_x > 1.0e-15:
        delta_x /= 10.0
        x1 = delta_x
        f1 = fun(x1)
    if f1 > f0:
        raise LineSearchError("the function does not decrease along the search direction")

    delta_x *= GOLDEN_RATIO2
    x2 = x1 + delta_x
    f2 = fun(x2)
    while f2 < f1 and x1 < 0.1:
        delta_x *= GOLDEN_RATIO2
        x0, x1, f1 = x1, x2, f2
        x2 = x1 + delta_x
        f2 = fun(x2)
    if f2 < f1 and x1 >= 0.1:
        return x1, f1

    a, b, u = x0, x2, x1
    v = a + GOLDEN_RATIO1 * (b - a)
    fu, fv = f1, fun(v)
    while True:
        if fu > fv:
            if fu - fv < epsilon:
                return v, fv
            a, u = u, v
            v = a + GOLDEN_RATIO1 * (b - a)
            fu, fv = fv, fun(v)
        else:
            if fv - fu < epsilon:
                return u, fu
            b, v = v, u
            u = b - GOLDEN_RATIO1 * (b - a)
            fv, fu = fu, fun(u)


def min_1d_real_bias(
    real_pes: PES,
    bias_energy: t.Callable[[np.ndarray], float],
    s: System,
    force_total: np.ndarray,
    pot_total: float,
    epsilon: float,
) -> np.ndarray:
    """Displacement minimising real + bias energy along ``force_total``."""
    f_norm = float(np.sqrt(np.sum(force_total * force_total)))
    trial = s.clone()

    def fun(x: float) -> float:
        trial.coord = s.coord + force_total * (x / f_norm)
        return real_pes.get_energy(trial) + bias_energy(trial.coord)

    x_min, _ = min_1d(fun, pot_total, epsilon)
    return force_total * (x_min / f_norm)


# Unit vectors of the Rust placements for 2, 3 and 4 molecules (scaled by ``dist``, in Bohr).
_PLACEMENT = {
    2: [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    3: [(0.0, 1.0, 0.0), (0.866025403784439, -0.5, 0.0), (-0.866025403784439, -0.5, 0.0)],
    4: [
        (0.0, 0.0, 1.0),
        (0.0, 0.942809041582063, -0.333333333333333),
        (0.816496580927726, -0.471404520791032, -0.333333333333333),
        (-0.816496580927726, -0.471404520791032, -0.333333333333333),
    ],
}


def place_molecules(
    s: System,
    mol_index: list[list[int]],
    dist: float,
    rng: np.random.Generator,
    min_contact: float = 0.0,
    max_tries: int = 1000,
) -> System:
    """Randomly rotate each molecule and put its centre at ``dist`` (Bohr) from the origin (Rust ``synthesis``).

    ``min_contact`` (Bohr, default 0 as in Rust) redraws the rotations until no two atoms of
    different molecules are closer than this.
    """
    if len(mol_index) not in _PLACEMENT:
        raise ValueError("synthesis placement is implemented for 2, 3 or 4 molecules")
    label = np.full(s.natom, -1)
    for k, idx in enumerate(mol_index):
        label[idx] = k
    inter = label[:, None] != label[None, :]
    for _ in range(max_tries):
        out = s.clone()
        rots = [rand_rot(rng) for _ in mol_index]
        for idx, rot, direction in zip(mol_index, rots, _PLACEMENT[len(mol_index)]):
            c = s.coord[idx]
            out.coord[idx] = (c - c.mean(axis=0)) @ rot + dist * np.asarray(direction)
        if min_contact <= 0.0 or np.min(np.where(inter, dist_mat(out.coord), np.inf)) >= min_contact:
            return out
    raise ValueError(f"no placement with all intermolecular distances >= {min_contact} Bohr in {max_tries} tries")


@dataclass
class SynthesisPot:
    """``mol_index`` lists the atoms of each reacting molecule. As in Rust, these may be fragments
    (e.g. the reactive site of a large molecule); all other atoms are environment that feels the
    real PES only."""

    initial_state: System
    mol_index: list[list[int]]
    para: PathPara
    output_path: str | Path


@dataclass
class SamplingResult:
    status: str                 # "product", "climb", "bias_force", "line_search" or "max_step"
    steps: int
    final_state: System
    pot_initial: float
    pot_max: float
    pot_min: float
    bias_removed_at: int | None


HEADER = "  step        rti_dist        pot_real        pot_rtip          f_real          f_rtip           a_min\n"


def synthesis_path_sampling(pot: SynthesisPot, real_pes: PES) -> SamplingResult:
    """Run RTIP pathway sampling and write ``rtip.pdb`` and ``rtip.out`` to ``pot.output_path``."""
    para = pot.para
    out = Path(pot.output_path)
    out.mkdir(parents=True, exist_ok=True)
    s = pot.initial_state.clone()
    frag = np.array([i for mol in pot.mol_index for i in mol])
    if len(set(frag.tolist())) != len(frag) or frag.min() < 0 or frag.max() >= s.natom:
        raise ValueError("mol_index must list distinct atoms of the system")
    bounds = np.cumsum([0] + [len(mol) for mol in pot.mol_index])
    local_index = [list(range(int(bounds[k]), int(bounds[k + 1]))) for k in range(len(pot.mol_index))]

    pot_max, pot_min, pot_initial = -np.inf, np.inf, 0.0
    add_rtip = True
    bias_removed_at = None
    status = "max_step"
    step = 0
    with open(out / "rtip.pdb", "w", encoding="utf-8") as pdb, open(out / "rtip.out", "w", encoding="utf-8") as log:
        pdb.write("TITLE     PDB file created by RTIP\n")
        pdb.write(s.pdb_frame(0))
        log.write(HEADER)
        for step in range(1, para.max_step + 1):
            pot_real, force_real = real_pes.get_energy_force(s)
            s.pot = pot_real
            f_real = float(np.sqrt(np.sum(force_real * force_real)))

            final_coord = evolution_target(s.coord[frag], local_index)
            sigma = rti_dist(final_coord, s.coord[frag])
            a_min = -para.a0 * step if add_rtip else 0.0
            force_rtip = np.zeros_like(s.coord)
            if add_rtip:
                pot_rtip, force_rtip[frag] = rti_pot_force(final_coord, s.coord[frag], a_min, sigma)
            else:
                pot_rtip = 0.0
            f_rtip = float(np.sqrt(np.sum(force_rtip * force_rtip)))

            if step % para.print_step == 0:
                pdb.write(s.pdb_frame(step))
            log.write(f"{step:6d} {sigma:15.8f} {pot_real:15.8f} {pot_rtip:15.8f} {f_real:15.8f} {f_rtip:15.8f} {a_min:15.8f}\n")
            log.flush()

            def bias_energy(coord: np.ndarray) -> float:
                return rti_pot(final_coord, coord[frag], a_min, sigma) if add_rtip else 0.0

            try:
                s.coord = s.coord + min_1d_real_bias(
                    real_pes, bias_energy, s, force_real + force_rtip, pot_real + pot_rtip, para.pot_epsilon * s.natom
                )
            except LineSearchError:
                status = "line_search"
                break

            pot_max = max(pot_max, pot_real)
            pot_min = min(pot_min, pot_real)
            if step == 1:
                pot_initial = pot_real
            if add_rtip and pot_real < pot_max - para.pot_drop and pot_real > pot_initial:
                add_rtip = False
                bias_removed_at = step
            if not add_rtip and f_real / np.sqrt(s.natom) < para.f_epsilon:
                status = "product"
                break
            if pot_real > pot_min + para.pot_climb:
                status = "climb"
                break
            if f_rtip > 1000.0:
                status = "bias_force"
                break
        log.write(f"# END step={step} status={status} bias_removed_at={bias_removed_at}\n")
    s.write_xyz(out / "final.xyz", True, step)
    return SamplingResult(status, step, s, pot_initial, pot_max, pot_min, bias_removed_at)


@dataclass
class AttractivePot:
    """Pull ``initial_state`` toward ``final_state`` (same atom order) with an attractive RTIP."""

    initial_state: System
    final_state: System
    para: PathPara
    output_path: str | Path


def attractive_path_sampling(pot: AttractivePot, real_pes: PES) -> SamplingResult:
    """Rust ``AttractivePot::rtip_path_sampling``.

    The RTIP is centred at the final state with height -a0 * step and width equal to the current
    RTI distance to it. It is removed once that distance is below 1 Bohr (or the RTIP force exceeds
    1000); the structure is then relaxed on the real PES until f_real / sqrt(N) < ``f_epsilon``.
    """
    para = pot.para
    out = Path(pot.output_path)
    out.mkdir(parents=True, exist_ok=True)
    s = pot.initial_state.clone()
    final_coord = pot.final_state.coord
    if final_coord.shape != s.coord.shape or pot.final_state.atom_type != s.atom_type:
        raise ValueError("initial and final states must contain the same atoms in the same order")

    pot_max, pot_min, pot_initial = -np.inf, np.inf, 0.0
    add_rtip = True
    bias_removed_at = None
    status = "max_step"
    step = 0
    with open(out / "rtip.pdb", "w", encoding="utf-8") as pdb, open(out / "rtip.out", "w", encoding="utf-8") as log:
        pdb.write("TITLE     PDB file created by RTIP\n")
        pdb.write(s.pdb_frame(0))
        log.write(HEADER)
        for step in range(1, para.max_step + 1):
            pot_real, force_real = real_pes.get_energy_force(s)
            s.pot = pot_real
            f_real = float(np.sqrt(np.sum(force_real * force_real)))
            if step == 1:
                pot_initial = pot_real
            pot_max = max(pot_max, pot_real)
            pot_min = min(pot_min, pot_real)

            sigma = rti_dist(final_coord, s.coord)
            a_min = -para.a0 * step if add_rtip else 0.0
            if add_rtip:
                pot_rtip, force_rtip = rti_pot_force(final_coord, s.coord, a_min, sigma)
            else:
                pot_rtip, force_rtip = 0.0, np.zeros_like(s.coord)
            f_rtip = float(np.sqrt(np.sum(force_rtip * force_rtip)))

            if step % para.print_step == 0:
                pdb.write(s.pdb_frame(step))
            log.write(f"{step:6d} {sigma:15.8f} {pot_real:15.8f} {pot_rtip:15.8f} {f_real:15.8f} {f_rtip:15.8f} {a_min:15.8f}\n")
            log.flush()

            def bias_energy(coord: np.ndarray) -> float:
                return rti_pot(final_coord, coord, a_min, sigma) if add_rtip else 0.0

            try:
                s.coord = s.coord + min_1d_real_bias(
                    real_pes, bias_energy, s, force_real + force_rtip, pot_real + pot_rtip, para.pot_epsilon * s.natom
                )
            except LineSearchError:
                status = "line_search"
                break

            if add_rtip and (sigma < 1.0 or f_rtip > 1000.0):
                add_rtip = False
                bias_removed_at = step
            if not add_rtip and f_real / np.sqrt(s.natom) < para.f_epsilon:
                status = "product"
                break
        log.write(f"# END step={step} status={status} bias_removed_at={bias_removed_at}\n")
    s.write_xyz(out / "final.xyz", True, step)
    return SamplingResult(status, step, s, pot_initial, pot_max, pot_min, bias_removed_at)
