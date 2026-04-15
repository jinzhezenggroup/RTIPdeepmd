from __future__ import annotations

import copy
import typing as t

import jax.numpy as jnp

from ..common.constants import GOLDEN_RATIO1, GOLDEN_RATIO2
from ..common.error import error_min_1d
from .system import System
from .traits import PES


def min_1d(fun: t.Callable[[float], float], f0: float, epsilon: float) -> tuple[float, float]:
	delta_x = 0.01
	x0 = 0.0
	x1 = delta_x
	f1 = float(fun(x1))

	while (f1 > f0) and (delta_x > 1.0e-15):
		delta_x /= 10.0
		x1 = delta_x
		f1 = float(fun(x1))

	if f1 > f0:
		raise ValueError(error_min_1d())

	delta_x *= GOLDEN_RATIO2
	x2 = x1 + delta_x
	f2 = float(fun(x2))

	while (f2 < f1) and (x1 < 0.1):
		delta_x *= GOLDEN_RATIO2
		x0 = x1
		x1 = x2
		x2 = x1 + delta_x
		f1 = f2
		f2 = float(fun(x2))

	if (f2 < f1) and (x1 >= 0.1):
		return x1, f1

	a = x0
	b = x2
	u = x1
	v = a + GOLDEN_RATIO1 * (b - a)
	fu = f1
	fv = float(fun(v))

	while True:
		if fu > fv:
			if (fu - fv) < epsilon:
				return v, fv
			a = u
			u = v
			v = a + GOLDEN_RATIO1 * (b - a)
			fu = fv
			fv = float(fun(v))
		else:
			if (fv - fu) < epsilon:
				return u, fu
			b = v
			v = u
			u = b - GOLDEN_RATIO1 * (b - a)
			fv = fu
			fu = float(fun(u))


def min_1d_real_bias(
	real_pes: PES,
	bias_pes: PES,
	s: System,
	force_total: jnp.ndarray,
	pot_total: float,
	epsilon: float,
) -> jnp.ndarray:
	s1 = copy.deepcopy(s)
	f_total = float(jnp.sqrt(jnp.sum(force_total * force_total)))
	if f_total < 1.0e-16:
		return jnp.zeros_like(force_total)

	def fun(x: float) -> float:
		s1.coord = s.coord + force_total * (x / f_total)
		return float(real_pes.get_energy(s1) + bias_pes.get_energy(s1))

	x_min, _ = min_1d(fun, pot_total, epsilon)
	return force_total * (x_min / f_total)
