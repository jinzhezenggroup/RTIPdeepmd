from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import typing as t

import jax.numpy as jnp
import numpy as np

from ..common.constants import Element
from .interfragment_descriptor import N_DES, get_interfragment_descriptor, truncate_cluster, update_neighbor_list
from .intrafragment_descriptor import get_intrafragment_descriptor
from .protein import ProteinSystem

EPS: float = 1.0e-6


@dataclass
class DataSaved:
    nstruct: int
    cell: np.ndarray | None
    natom: int
    atom_type: list[Element]
    coord: np.ndarray
    pot: np.ndarray
    force: np.ndarray

    @classmethod
    def from_npz(cls, filename: str | Path) -> "DataSaved":
        data = np.load(filename, allow_pickle=True)

        nstruct = int(np.asarray(data["nstruct"]).item())
        natom = int(np.asarray(data["natom"]).item())

        cell_raw = data.get("cell")
        cell = None
        if cell_raw is not None:
            cell_arr = np.asarray(cell_raw)
            if cell_arr.dtype == object and cell_arr.size == 1 and cell_arr.item() is None:
                cell = None
            elif cell_arr.size > 0:
                cell = np.asarray(cell_arr, dtype=float).reshape(3, 3)

        atom_type = _decode_atom_types(np.asarray(data["atom_type"]))
        coord = np.asarray(data["coord"], dtype=float).reshape(nstruct, natom, 3)
        pot = np.asarray(data["pot"], dtype=float).reshape(nstruct)
        force = np.asarray(data["force"], dtype=float).reshape(nstruct, natom, 3)

        return cls(
            nstruct=nstruct,
            cell=cell,
            natom=natom,
            atom_type=atom_type,
            coord=coord,
            pot=pot,
            force=force,
        )

    @classmethod
    def from_pickle(cls, filename: str | Path) -> "DataSaved":
        with Path(filename).open("rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, DataSaved):
            return payload
        if not isinstance(payload, dict):
            raise TypeError("Unsupported pickle payload for DataSaved; expected dict or DataSaved")

        atom_type = _decode_atom_types(np.asarray(payload["atom_type"]))
        nstruct = int(payload["nstruct"])
        natom = int(payload["natom"])

        cell_raw = payload.get("cell")
        cell = None if cell_raw is None else np.asarray(cell_raw, dtype=float).reshape(3, 3)

        return cls(
            nstruct=nstruct,
            cell=cell,
            natom=natom,
            atom_type=atom_type,
            coord=np.asarray(payload["coord"], dtype=float).reshape(nstruct, natom, 3),
            pot=np.asarray(payload["pot"], dtype=float).reshape(nstruct),
            force=np.asarray(payload["force"], dtype=float).reshape(nstruct, natom, 3),
        )


@dataclass
class FragmentDescriptor:
    fragment_type: t.Any
    interfragment_descriptor: jnp.ndarray
    intrafragment_descriptor: jnp.ndarray | None
    interfragment_descriptor_diff: jnp.ndarray
    intrafragment_descriptor_diff: jnp.ndarray | None


@dataclass
class ProteinSystemDescriptor:
    n_diff: int
    fragment_descriptor: list[FragmentDescriptor]
    pot: jnp.ndarray
    pot_diff: jnp.ndarray



def _decode_atom_types(atom_type_raw: np.ndarray) -> list[Element]:
    values = np.asarray(atom_type_raw).reshape(-1)
    out: list[Element] = []
    for value in values:
        if isinstance(value, Element):
            out.append(value)
            continue

        if isinstance(value, (bytes, np.bytes_)):
            token = value.decode("utf-8")
        else:
            token = str(value)

        # Defensive compatibility for legacy stringified bytes, e.g. "b'H'".
        if token.startswith("b'") and token.endswith("'") and len(token) >= 3:
            token = token[2:-1]

        out.append(Element.from_str(token.strip()))
    return out


def _load_data_saved(input_sub_dir: str) -> DataSaved:
    data_dir = Path("data") / input_sub_dir

    pkl_file = data_dir / "data.pkl"
    if pkl_file.exists():
        return DataSaved.from_pickle(pkl_file)

    npz_file = data_dir / "data.npz"
    if npz_file.exists():
        return DataSaved.from_npz(npz_file)

    bin_file = data_dir / "data.bin"
    if bin_file.exists():
        raise NotImplementedError(
            "Rust savefile binary data.bin is not directly readable in Python. "
            "Please export dataset as data.npz or data.pkl first."
        )

    raise FileNotFoundError(f"No training data file found in {data_dir}; expected data.npz or data.pkl")


def from_struct(s: ProteinSystem) -> ProteinSystemDescriptor:
    if s.force is None:
        raise ValueError("ProteinSystem.force is required to build finite-difference training targets")

    n_diff = s.natom * 6
    force = np.asarray(s.force, dtype=float)
    fragment_descriptor: list[FragmentDescriptor] = []

    for i in range(len(s.fragment)):
        update_neighbor_list(s, i)

        coord_within_rcut, list_within_rcut, atomic_number_within_rcut = truncate_cluster(s, i)
        inter_desc, inter_grad = get_interfragment_descriptor(
            coord_within_rcut,
            atomic_number_within_rcut,
            s.fragment[i].natom,
        )

        inter_desc_arr = np.asarray(inter_desc, dtype=float)
        inter_grad_arr = np.asarray(inter_grad, dtype=float)

        index_map = {atom_idx: local_idx for local_idx, atom_idx in enumerate(list_within_rcut)}
        inter_diff = np.zeros((n_diff, N_DES), dtype=float)

        for atom_idx in range(s.natom):
            local_idx = index_map.get(atom_idx)
            base_row = atom_idx * 6
            if local_idx is None:
                inter_diff[base_row : base_row + 6] = inter_desc_arr
                continue

            for axis in range(3):
                row = local_idx * 3 + axis
                inter_diff[base_row + 2 * axis] = inter_desc_arr + EPS * inter_grad_arr[row]
                inter_diff[base_row + 2 * axis + 1] = inter_desc_arr - EPS * inter_grad_arr[row]

        intra = get_intrafragment_descriptor(s.fragment[i])
        if intra is None:
            intra_desc_jnp = None
            intra_diff_jnp = None
        else:
            intra_desc, intra_grad = intra
            intra_desc_arr = np.asarray(intra_desc, dtype=float)
            intra_grad_arr = np.asarray(intra_grad, dtype=float)
            n_intra = intra_desc_arr.shape[0]

            atom_list = s.fragment[i].atom_list
            if atom_list is None:
                raise ValueError("fragment.atom_list is None for intrafragment descriptor")
            intra_index_map = {atom_idx: local_idx for local_idx, atom_idx in enumerate(atom_list)}

            intra_diff = np.zeros((n_diff, n_intra), dtype=float)
            for atom_idx in range(s.natom):
                local_idx = intra_index_map.get(atom_idx)
                base_row = atom_idx * 6
                if local_idx is None:
                    intra_diff[base_row : base_row + 6] = intra_desc_arr
                    continue

                for axis in range(3):
                    row = local_idx * 3 + axis
                    intra_diff[base_row + 2 * axis] = intra_desc_arr + EPS * intra_grad_arr[row]
                    intra_diff[base_row + 2 * axis + 1] = intra_desc_arr - EPS * intra_grad_arr[row]

            intra_desc_jnp = jnp.asarray(intra_desc_arr, dtype=jnp.float64)
            intra_diff_jnp = jnp.asarray(intra_diff, dtype=jnp.float64)

        fragment_descriptor.append(
            FragmentDescriptor(
                fragment_type=s.fragment[i].fragment_type,
                interfragment_descriptor=jnp.asarray(inter_desc_arr, dtype=jnp.float64),
                intrafragment_descriptor=intra_desc_jnp,
                interfragment_descriptor_diff=jnp.asarray(inter_diff, dtype=jnp.float64),
                intrafragment_descriptor_diff=intra_diff_jnp,
            )
        )

    pot_diff = np.zeros((n_diff, 1), dtype=float)
    for i in range(s.natom):
        for axis in range(3):
            pot_diff[i * 6 + 2 * axis, 0] = -EPS * force[i, axis]
            pot_diff[i * 6 + 2 * axis + 1, 0] = EPS * force[i, axis]

    return ProteinSystemDescriptor(
        n_diff=n_diff,
        fragment_descriptor=fragment_descriptor,
        pot=jnp.asarray([s.pot], dtype=jnp.float64),
        pot_diff=jnp.asarray(pot_diff, dtype=jnp.float64),
    )


def from_data(input_sub_dir: str) -> list[ProteinSystemDescriptor]:
    str_input_file = Path("data") / input_sub_dir / "str.pdb"
    if not str_input_file.exists():
        raise FileNotFoundError(f"Missing structure file: {str_input_file}")

    s = ProteinSystem.read_pdb(str(str_input_file))
    data = _load_data_saved(input_sub_dir)

    if s.natom != data.natom:
        raise ValueError(f"natom mismatch: structure {s.natom}, data {data.natom}")
    if s.atom_type != data.atom_type:
        raise ValueError("atom_type mismatch between structure and training data")

    out: list[ProteinSystemDescriptor] = []
    for i in range(data.nstruct):
        s.coord = jnp.asarray(data.coord[i], dtype=jnp.float64)

        for frag in s.fragment:
            if frag.atom_list is None:
                continue
            frag.coord = jnp.asarray(np.asarray(s.coord)[frag.atom_list], dtype=jnp.float64)

        s.pot = float(data.pot[i])
        s.force = jnp.asarray(data.force[i], dtype=jnp.float64)

        out.append(from_struct(s))

    return out
