from __future__ import annotations

from dataclasses import dataclass, field
import typing as t

import numpy as np
import jax.numpy as jnp

from ..common.constants import BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR
from ..pes_exploration.system import System


# DeepMD 常见输出单位：energy=eV, force=eV/Å
EV_TO_HARTREE = 0.03674932217565499
EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR = EV_TO_HARTREE / ANGSTROM_TO_BOHR


def _resolve_deep_pot() -> t.Any:
    try:
        from deepmd.infer import DeepPot  # type: ignore

        return DeepPot
    except Exception as exc:  # pragma: no cover - 依赖可选
        raise NotImplementedError(
            "DeepMD backend is not available. Please install deepmd-kit and provide a valid model file."
        ) from exc


@dataclass
class DeepmdState:
    """可选的轻量状态容器，用于与 System.from_deepmd/to_deepmd 互操作。"""

    positions: list[float]
    potential_energy: float = 0.0
    cell: list[float] | None = None


def deepmd_get_nparticle(state: t.Any) -> int:
    if hasattr(state, "positions"):
        return int(len(getattr(state, "positions")) // 3)
    if isinstance(state, dict) and "positions" in state:
        return int(len(state["positions"]) // 3)
    raise ValueError("deepmd state must contain flat 'positions' with shape natom*3")


def deepmd_get_positions(state: t.Any, n_el: int) -> list[float]:
    pos = getattr(state, "positions", None)
    if pos is None and isinstance(state, dict):
        pos = state.get("positions")
    if pos is None:
        raise ValueError("deepmd state does not provide positions")
    out = list(pos)
    if len(out) < n_el:
        raise ValueError(f"deepmd positions length={len(out)} is smaller than requested n_el={n_el}")
    return out[:n_el]


def deepmd_set_positions(state: t.Any, new_pos: list[float], n_el: int) -> None:
    out = list(new_pos[:n_el])
    if hasattr(state, "positions"):
        setattr(state, "positions", out)
        return
    if isinstance(state, dict):
        state["positions"] = out
        return
    raise ValueError("deepmd state is not mutable")


def deepmd_get_potential_energy(state: t.Any) -> float:
    pot = getattr(state, "potential_energy", None)
    if pot is None and isinstance(state, dict):
        pot = state.get("potential_energy", 0.0)
    return float(0.0 if pot is None else pot)


@dataclass
class DeepmdPES:
    model_file: str
    type_map: dict[str, int] = field(default_factory=dict)
    _dp: t.Any | None = field(default=None, init=False, repr=False)

    @classmethod
    def new(cls, model_file: str, type_map: dict[str, int] | None = None) -> "DeepmdPES":
        return cls(model_file=model_file, type_map={} if type_map is None else dict(type_map))

    def _get_dp(self) -> t.Any:
        """Lazily create and cache DeepPot to avoid repeated model I/O/initialization."""
        if self._dp is None:
            DeepPot = _resolve_deep_pot()
            self._dp = DeepPot(self.model_file)
        return self._dp

    def _infer_type_map_from_model(self) -> dict[str, int]:
        dp = self._get_dp()
        get_type_map = getattr(dp, "get_type_map", None)
        if not callable(get_type_map):
            return {}

        raw_type_map = get_type_map()
        if raw_type_map is None:
            return {}

        model_type_map: list[str] = []
        for item in raw_type_map:
            if isinstance(item, bytes):
                elem = item.decode("utf-8", errors="strict").strip()
            else:
                elem = str(item).strip()
            if elem:
                model_type_map.append(elem)

        return {elem: i for i, elem in enumerate(model_type_map)}

    def _build_atype(self, s: System) -> np.ndarray:
        if s.atom_type is None:
            return np.zeros((s.natom,), dtype=np.int32)

        if not self.type_map:
            self.type_map = self._infer_type_map_from_model()
            if not self.type_map:
                raise ValueError(
                    "DeepmdPES requires 'type_map' when System.atom_type is set. "
                    "Provide one explicitly, or use a DeepMD model that exposes get_type_map()."
                )

        out = np.zeros((s.natom,), dtype=np.int32)
        for i, elem in enumerate(s.atom_type):
            key = elem.value
            if key not in self.type_map:
                raise ValueError(
                    f"Element {key} not found in DeepMD type_map. "
                    f"Available keys: {sorted(self.type_map.keys())}"
                )
            out[i] = int(self.type_map[key])
        return out

    def eval_arrays(
        self,
        coord_ang: np.ndarray,
        atype: np.ndarray,
        cell_ang: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """
        Memory-level DeepMD call:
        - coord_ang: shape (natom, 3), unit Å
        - atype: shape (natom,), int type ids
        - cell_ang: shape (3, 3) or (9,), unit Å; optional for non-PBC
        Returns:
        - energy (eV)
        - force (eV/Å), shape (natom, 3)
        """
        coord_np = np.asarray(coord_ang, dtype=np.float64)
        if coord_np.ndim != 2 or coord_np.shape[1] != 3:
            raise ValueError("coord_ang must have shape (natom, 3)")

        atype_np = np.asarray(atype, dtype=np.int32).reshape(-1)
        if atype_np.shape[0] != coord_np.shape[0]:
            raise ValueError("atype length must match number of atoms")

        box_flat: np.ndarray | None = None
        if cell_ang is not None:
            cell_np = np.asarray(cell_ang, dtype=np.float64)
            if cell_np.shape == (3, 3):
                box_flat = cell_np.reshape(1, 9)
            elif cell_np.shape == (9,):
                box_flat = cell_np.reshape(1, 9)
            else:
                raise ValueError("cell_ang must have shape (3, 3) or (9,)")

        coord_flat = coord_np.reshape(1, -1)
        dp = self._get_dp()
        try:
            result = dp.eval(coord_flat, box_flat, atype_np)
        except TypeError:
            box_3x3 = None if box_flat is None else box_flat.reshape(1, 3, 3)
            result = dp.eval(coord_np.reshape(1, coord_np.shape[0], 3), box_3x3, atype_np)

        if isinstance(result, tuple) and len(result) >= 2:
            energy_ev = float(np.asarray(result[0]).reshape(-1)[0])
            force_ev_per_ang = np.asarray(result[1], dtype=np.float64).reshape(coord_np.shape[0], 3)
            return energy_ev, force_ev_per_ang

        if isinstance(result, dict) and "energy" in result and "force" in result:
            energy_ev = float(np.asarray(result["energy"]).reshape(-1)[0])
            force_ev_per_ang = np.asarray(result["force"], dtype=np.float64).reshape(coord_np.shape[0], 3)
            return energy_ev, force_ev_per_ang

        raise RuntimeError("Unsupported DeepMD eval output format")

    def _eval(self, s: System) -> tuple[float, np.ndarray]:
        coord_ang = np.asarray(s.coord, dtype=np.float64) * BOHR_TO_ANGSTROM
        atype = self._build_atype(s)

        cell_ang = None if s.cell is None else (np.asarray(s.cell, dtype=np.float64) * BOHR_TO_ANGSTROM)
        return self.eval_arrays(coord_ang=coord_ang, atype=atype, cell_ang=cell_ang)

    def get_energy(self, s: System) -> float:
        energy_ev, _ = self._eval(s)
        return float(energy_ev * EV_TO_HARTREE)

    def get_energy_force(self, s: System) -> tuple[float, jnp.ndarray]:
        energy_ev, force_ev_per_ang = self._eval(s)
        energy_h = float(energy_ev * EV_TO_HARTREE)
        force_h_per_bohr = force_ev_per_ang * EV_PER_ANGSTROM_TO_HARTREE_PER_BOHR
        return energy_h, jnp.asarray(force_h_per_bohr, dtype=jnp.float64)

    def close(self) -> None:
        # DeepMD 推理对象通常由 Python GC 管理；显式清空引用便于及时释放。
        self._dp = None

    def __enter__(self) -> "DeepmdPES":
        return self

    def __exit__(self, exc_type: t.Any, exc: t.Any, tb: t.Any) -> bool:
        self.close()
        return False
