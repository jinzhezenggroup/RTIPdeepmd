from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
import typing as t

import jax
import jax.numpy as jnp
import numpy as np
import optax

ModelParams = list[dict[str, jnp.ndarray]]


def _init_mlp(input_dim: int, hidden_dims: tuple[int, ...], key: jax.Array) -> ModelParams:
    dims = (input_dim, *hidden_dims)
    keys = jax.random.split(key, len(dims) - 1)
    layers: ModelParams = []
    for in_dim, out_dim, layer_key in zip(dims[:-1], dims[1:], keys, strict=True):
        w_key, _ = jax.random.split(layer_key)
        limit = jnp.sqrt(6.0 / (in_dim + out_dim))
        layers.append(
            {
                "w": jax.random.uniform(w_key, (in_dim, out_dim), minval=-limit, maxval=limit, dtype=jnp.float64),
                "b": jnp.zeros((out_dim,), dtype=jnp.float64),
            }
        )
    return layers


def _mlp_forward(params: ModelParams, x: jnp.ndarray) -> jnp.ndarray:
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["w"] + layer["b"])
    out = h @ params[-1]["w"] + params[-1]["b"]
    return out


def _tree_zeros(tree: t.Any) -> t.Any:
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _to_numpy_tree(tree: t.Any) -> t.Any:
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def _to_jax_tree(tree: t.Any) -> t.Any:
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), tree)


def _fragment_key(fragment_type: t.Any) -> str:
    if fragment_type is None:
        return "unknown"

    kind = getattr(fragment_type, "kind", type(fragment_type).__name__)
    value = getattr(fragment_type, "value", None)
    if hasattr(value, "value"):
        value = value.value
    return f"{kind}:{value}"


@dataclass
class GlobalNN:
    params: dict[str, ModelParams] = field(default_factory=dict)
    input_dims: dict[str, int] = field(default_factory=dict)
    hidden_dims: tuple[int, ...] = (128, 64, 16, 1)

    def ensure_model(self, key: str, input_dim: int, seed: int = 1314) -> None:
        if key in self.params:
            return
        rng = jax.random.PRNGKey((abs(hash((key, seed))) % (2**31 - 1)) + 1)
        self.params[key] = _init_mlp(input_dim, self.hidden_dims, rng)
        self.input_dims[key] = input_dim

    def forward(self, key: str, x: jnp.ndarray, params: dict[str, ModelParams] | None = None) -> jnp.ndarray:
        tree = self.params if params is None else params
        if key not in tree:
            raise KeyError(f"No NN submodel for key: {key}")
        return _mlp_forward(tree[key], x)

    @classmethod
    def new(cls, input_dims: dict[str, int], hidden_dims: tuple[int, ...] = (128, 64, 16, 1), seed: int = 1314) -> "GlobalNN":
        nn = cls(params={}, input_dims={}, hidden_dims=hidden_dims)
        for key, dim in input_dims.items():
            nn.ensure_model(key, dim, seed=seed)
        return nn

    def save(self, output_sub_dir: str) -> None:
        out_dir = Path("nn") / output_sub_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": _to_numpy_tree(self.params),
            "input_dims": self.input_dims,
            "hidden_dims": self.hidden_dims,
        }
        with (out_dir / "global_nn.pkl").open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, input_sub_dir: str) -> "GlobalNN":
        in_file = Path("nn") / input_sub_dir / "global_nn.pkl"
        with in_file.open("rb") as f:
            payload = pickle.load(f)
        return cls(
            params=_to_jax_tree(payload["params"]),
            input_dims=dict(payload["input_dims"]),
            hidden_dims=tuple(payload["hidden_dims"]),
        )

    def alloc_grads(self) -> dict[str, ModelParams]:
        return _tree_zeros(self.params)

    def zero_grads(self, global_grads: dict[str, ModelParams]) -> dict[str, ModelParams]:
        return _tree_zeros(global_grads)


@dataclass
class GlobalAdam:
    tx: optax.GradientTransformation
    state: optax.OptState

    @classmethod
    def new(cls, global_nn: GlobalNN, learning_rate: float = 1e-3) -> "GlobalAdam":
        tx = optax.adam(learning_rate)
        state = tx.init(global_nn.params)
        return cls(tx=tx, state=state)

    def update(self, global_nn: GlobalNN, grads: dict[str, ModelParams]) -> None:
        updates, self.state = self.tx.update(grads, self.state, global_nn.params)
        global_nn.params = optax.apply_updates(global_nn.params, updates)


def new(input_dims: dict[str, int], hidden_dims: tuple[int, ...] = (128, 64, 16, 1), seed: int = 1314) -> GlobalNN:
    return GlobalNN.new(input_dims, hidden_dims, seed)


def save(global_nn: GlobalNN, output_sub_dir: str) -> None:
    global_nn.save(output_sub_dir)


def load(input_sub_dir: str) -> GlobalNN:
    return GlobalNN.load(input_sub_dir)


def alloc_grads(global_nn: GlobalNN) -> dict[str, ModelParams]:
    return global_nn.alloc_grads()


def zero_grads(global_nn: GlobalNN, global_grads: dict[str, ModelParams]) -> dict[str, ModelParams]:
    return global_nn.zero_grads(global_grads)


def update(global_adam: GlobalAdam, global_nn: GlobalNN, grads: dict[str, ModelParams]) -> None:
    global_adam.update(global_nn, grads)


def fragment_key(fragment_type: t.Any) -> str:
    return _fragment_key(fragment_type)
