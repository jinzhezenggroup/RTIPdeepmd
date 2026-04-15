from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from ..common.constants import PI


def rand_rot(key: jax.Array | None = None) -> jnp.ndarray:
    if key is None:
        alpha, beta, gamma = np.random.uniform(0.0, 2.0 * PI, 3)
    else:
        k1, k2, k3 = jax.random.split(key, 3)
        alpha = float(jax.random.uniform(k1, shape=(), minval=0.0, maxval=2.0 * PI))
        beta = float(jax.random.uniform(k2, shape=(), minval=0.0, maxval=2.0 * PI))
        gamma = float(jax.random.uniform(k3, shape=(), minval=0.0, maxval=2.0 * PI))

    return jnp.array(
        [
            [
                np.cos(alpha) * np.cos(gamma) - np.cos(beta) * np.sin(alpha) * np.sin(gamma),
                -np.cos(beta) * np.cos(gamma) * np.sin(alpha) - np.cos(alpha) * np.sin(gamma),
                np.sin(alpha) * np.sin(beta),
            ],
            [
                np.cos(gamma) * np.sin(alpha) + np.cos(alpha) * np.cos(beta) * np.sin(gamma),
                np.cos(alpha) * np.cos(beta) * np.cos(gamma) - np.sin(alpha) * np.sin(gamma),
                -np.cos(alpha) * np.sin(beta),
            ],
            [
                np.sin(beta) * np.sin(gamma),
                np.cos(gamma) * np.sin(beta),
                np.cos(beta),
            ],
        ],
        dtype=jnp.float64,
    )
