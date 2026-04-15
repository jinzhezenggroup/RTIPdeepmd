from __future__ import annotations

from pathlib import Path
import random

import jax
import jax.numpy as jnp

from ..io.input import Para
from .global_nn import GlobalAdam, GlobalNN, fragment_key
from .training_data import EPS, ProteinSystemDescriptor, from_data

MULTIPLE: float = 1.0 / EPS


def _concat_features(inter: jnp.ndarray, intra: jnp.ndarray | None) -> jnp.ndarray:
    if intra is None:
        return jnp.asarray(inter, dtype=jnp.float64)
    return jnp.concatenate([jnp.asarray(inter, dtype=jnp.float64), jnp.asarray(intra, dtype=jnp.float64)], axis=0)


def _concat_features_diff(inter: jnp.ndarray, intra: jnp.ndarray | None) -> jnp.ndarray:
    if intra is None:
        return jnp.asarray(inter, dtype=jnp.float64)
    return jnp.concatenate([jnp.asarray(inter, dtype=jnp.float64), jnp.asarray(intra, dtype=jnp.float64)], axis=1)


def _predict_one(global_nn: GlobalNN, params: dict, one_point_data: ProteinSystemDescriptor) -> tuple[jnp.ndarray, jnp.ndarray]:
    predicted_pot = jnp.asarray(0.0, dtype=jnp.float64)
    predicted_pot_1 = jnp.asarray(0.0, dtype=jnp.float64)
    predicted_pot_diff = jnp.zeros((one_point_data.n_diff,), dtype=jnp.float64)

    for frag in one_point_data.fragment_descriptor:
        key = fragment_key(frag.fragment_type)

        x = _concat_features(frag.interfragment_descriptor, frag.intrafragment_descriptor)
        y = global_nn.forward(key, x, params).reshape(())

        x1 = _concat_features(frag.interfragment_descriptor, frag.intrafragment_descriptor)
        y1 = global_nn.forward(key, x1, params).reshape(())

        x_diff = _concat_features_diff(frag.interfragment_descriptor_diff, frag.intrafragment_descriptor_diff)
        y_diff = jax.vmap(lambda xrow: global_nn.forward(key, xrow, params).reshape(()))(x_diff)

        predicted_pot = predicted_pot + y
        predicted_pot_1 = predicted_pot_1 + y1
        predicted_pot_diff = predicted_pot_diff + y_diff

    predicted_pot_diff = predicted_pot_diff - predicted_pot_1
    return predicted_pot, predicted_pot_diff


def _rmse(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.mean((pred - target) ** 2))


def _one_point_losses(global_nn: GlobalNN, params: dict, one_point_data: ProteinSystemDescriptor) -> tuple[jnp.ndarray, jnp.ndarray]:
    predicted_pot, predicted_pot_diff = _predict_one(global_nn, params, one_point_data)

    target_pot = jnp.asarray(one_point_data.pot, dtype=jnp.float64).reshape(())
    target_pot_diff = jnp.asarray(one_point_data.pot_diff, dtype=jnp.float64).reshape((one_point_data.n_diff,))

    loss_pot = _rmse(predicted_pot, target_pot)
    loss_pot_diff = _rmse(predicted_pot_diff, target_pot_diff) * MULTIPLE
    return loss_pot, loss_pot_diff


def one_point_rmse_mut(global_nn: GlobalNN, one_point_training_data: ProteinSystemDescriptor) -> tuple[float, float, dict]:
    def total_loss(params: dict) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        loss_pot, loss_pot_diff = _one_point_losses(global_nn, params, one_point_training_data)
        return loss_pot + loss_pot_diff, (loss_pot, loss_pot_diff)

    (_loss, (loss_pot, loss_pot_diff)), grads = jax.value_and_grad(total_loss, has_aux=True)(global_nn.params)
    return float(loss_pot), float(loss_pot_diff), grads


def one_point_rmse(global_nn: GlobalNN, one_point_training_data: ProteinSystemDescriptor) -> tuple[float, float]:
    loss_pot, loss_pot_diff = _one_point_losses(global_nn, global_nn.params, one_point_training_data)
    return float(loss_pot), float(loss_pot_diff)


def _infer_input_dims(dataset: list[ProteinSystemDescriptor]) -> dict[str, int]:
    dims: dict[str, int] = {}

    for sample in dataset:
        for frag in sample.fragment_descriptor:
            key = fragment_key(frag.fragment_type)
            inter_dim = int(frag.interfragment_descriptor.shape[0])
            intra_dim = 0 if frag.intrafragment_descriptor is None else int(frag.intrafragment_descriptor.shape[0])
            cur_dim = inter_dim + intra_dim

            if key in dims and dims[key] != cur_dim:
                raise ValueError(f"Inconsistent feature dimension for fragment {key}: {dims[key]} vs {cur_dim}")
            dims[key] = cur_dim

    return dims


def _tree_add(a: dict, b: dict) -> dict:
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _tree_scale(a: dict, factor: float) -> dict:
    return jax.tree_util.tree_map(lambda x: x * factor, a)


def _load_or_create_nn(input_sub_dir: str, input_dims: dict[str, int]) -> GlobalNN:
    if input_sub_dir:
        nn_file = Path("nn") / input_sub_dir / "global_nn.pkl"
        if nn_file.exists():
            nn = GlobalNN.load(input_sub_dir)
            for key, dim in input_dims.items():
                nn.ensure_model(key, dim)
            return nn

    return GlobalNN.new(input_dims)


def train(para: Para) -> None:
    data_set = from_data(para.nn_train_para.input_data_sub_dir)
    data_size = len(data_set)
    if data_size < 2:
        raise ValueError("Training requires at least two structures (train/validation split)")

    input_dims = _infer_input_dims(data_set)
    global_nn = _load_or_create_nn(para.nn_train_para.input_nn_sub_dir, input_dims)
    global_adam = GlobalAdam.new(global_nn, learning_rate=1.0e-3)

    output_nn_dir = Path("nn") / para.nn_train_para.output_nn_sub_dir
    output_nn_dir.mkdir(parents=True, exist_ok=True)

    nn_training_output_file = output_nn_dir / "nn_training.out"
    with nn_training_output_file.open("w", encoding="utf-8") as f:
        f.write("  step       training_pot_loss         training_f_loss     validation_pot_loss       validation_f_loss\n")

        all_idx = list(range(data_size))
        rng = random.Random(1314)

        while True:
            training_idx = [i for i in all_idx if rng.random() < 0.9]
            validation_idx = [i for i in all_idx if i not in training_idx]
            if training_idx and validation_idx:
                break

        train_batch = max(1, para.nn_train_para.training_batch_size)
        valid_batch = max(1, para.nn_train_para.validation_batch_size)
        max_step = max(1, para.nn_train_para.max_step)
        print_step = max(1, para.nn_train_para.print_step)

        for step in range(1, max_step + 1):
            training_loss_pot = 0.0
            training_loss_pot_diff = 0.0
            grad_acc = global_nn.alloc_grads()

            for _ in range(train_batch):
                idx = rng.choice(training_idx)
                loss_pot, loss_pot_diff, grads = one_point_rmse_mut(global_nn, data_set[idx])
                training_loss_pot += loss_pot
                training_loss_pot_diff += loss_pot_diff
                grad_acc = _tree_add(grad_acc, grads)

            training_loss_pot /= float(train_batch)
            training_loss_pot_diff /= float(train_batch)
            grad_acc = _tree_scale(grad_acc, 1.0 / float(train_batch))

            global_adam.update(global_nn, grad_acc)

            validation_loss_pot = 0.0
            validation_loss_pot_diff = 0.0
            for _ in range(valid_batch):
                idx = rng.choice(validation_idx)
                loss_pot, loss_pot_diff = one_point_rmse(global_nn, data_set[idx])
                validation_loss_pot += loss_pot
                validation_loss_pot_diff += loss_pot_diff

            validation_loss_pot /= float(valid_batch)
            validation_loss_pot_diff /= float(valid_batch)

            f.write(
                f"{step:6d} {training_loss_pot:23.8f} {training_loss_pot_diff:23.8f} "
                f"{validation_loss_pot:23.8f} {validation_loss_pot_diff:23.8f}\n"
            )
            f.flush()

            if (step % print_step) == 0:
                global_nn.save(f"{para.nn_train_para.output_nn_sub_dir}/{step}")
