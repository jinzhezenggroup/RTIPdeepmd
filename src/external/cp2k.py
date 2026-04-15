from __future__ import annotations

from .deepmd import DeepmdPES


def _removed(name: str) -> None:
    raise NotImplementedError(f"{name} has been removed. Use external.deepmd instead.")


# 兼容旧类名：直接映射到 DeepMD 实现
Cp2kPES = DeepmdPES


def cp2k_init() -> None:
    _removed("cp2k_init")


def cp2k_init_without_mpi() -> None:
    _removed("cp2k_init_without_mpi")


def cp2k_finalize() -> None:
    _removed("cp2k_finalize")


def cp2k_finalize_without_mpi() -> None:
    _removed("cp2k_finalize_without_mpi")
