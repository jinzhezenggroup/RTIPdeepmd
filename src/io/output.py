from __future__ import annotations

import os
import typing as t

from ..common.constants import ROOT_RANK


def _comm_rank(comm: t.Any) -> int:
	if hasattr(comm, "rank"):
		rank_attr = getattr(comm, "rank")
		return int(rank_attr() if callable(rank_attr) else rank_attr)
	if hasattr(comm, "Get_rank") and callable(getattr(comm, "Get_rank")):
		return int(comm.Get_rank())
	return ROOT_RANK


def create_output_path(comm: t.Any, index: int | None) -> str:
	if index is None:
		return ""

	if _comm_rank(comm) == ROOT_RANK:
		os.makedirs(str(index), exist_ok=True)

	return f"{index}/"
