from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from ..common.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, Element, FragmentType
from ..pes_exploration.system import System


_TWO_LETTER_ELEMENTS = {e.value.upper() for e in Element if len(e.value) == 2}


def _infer_pdb_element(atom_name: str, res_name: str, elem_col: str) -> Element:
    token = elem_col.strip()
    if token:
        try:
            return Element.from_str(token.title())
        except Exception:
            pass

    letters = "".join(ch for ch in atom_name if ch.isalpha())
    if not letters:
        raise ValueError(f"Cannot infer element from atom name: {atom_name!r}")

    letters_up = letters.upper()
    res_up = res_name.strip().upper()

    # Ion-like records often encode the element as atom/residue name (e.g. NA, CL, CA).
    if len(letters_up) >= 2:
        two = letters_up[:2]
        if two in _TWO_LETTER_ELEMENTS and (res_up == two or letters_up == two):
            try:
                return Element.from_str(two.title())
            except Exception:
                pass

    # Protein-safe fallback: alpha-carbon 'CA' should map to carbon, not calcium.
    return Element.from_str(letters_up[0].title())


@dataclass
class Fragment:
    fragment_type: FragmentType
    natom: int
    coord: jnp.ndarray
    atom_type: list[Element]
    pot: float = 0.0
    force: jnp.ndarray | None = None
    atom_list: list[int] | None = None
    neighbor_list: list[int] | None = None


@dataclass
class ProteinSystem:
    natom: int
    coord: jnp.ndarray
    cell: jnp.ndarray | None
    atom_type: list[Element]
    pot: float = 0.0
    force: jnp.ndarray | None = None
    fragment: list[Fragment] = field(default_factory=list)

    @classmethod
    def from_system(cls, s: System) -> "ProteinSystem":
        if s.atom_type is None:
            raise ValueError("System.atom_type is required for ProteinSystem.from_system")

        fragments: list[Fragment] = []
        for i, atom in enumerate(s.atom_type):
            fragments.append(
                Fragment(
                    fragment_type=FragmentType.atom(atom),
                    natom=1,
                    coord=jnp.asarray(s.coord[i : i + 1]),
                    atom_type=[atom],
                    pot=0.0,
                    force=None,
                    atom_list=[i],
                    neighbor_list=None,
                )
            )

        return cls(
            natom=s.natom,
            coord=jnp.asarray(s.coord),
            cell=None if s.cell is None else jnp.asarray(s.cell),
            atom_type=list(s.atom_type),
            pot=s.pot,
            force=None,
            fragment=fragments,
        )

    @classmethod
    def read_pdb(cls, filename: str) -> "ProteinSystem":
        lines = Path(filename).read_text(encoding="utf-8").splitlines()

        coord: list[list[float]] = []
        atom_type: list[Element] = []

        chain_order: list[tuple[int, str]] = []
        chain_to_res: dict[tuple[int, str], list[tuple[int, str, str, str]]] = {}
        res_to_atoms: dict[tuple[int, str, str, str], list[tuple[Element, list[float], int]]] = {}

        chain_block = 0
        for line in lines:
            rec = line[0:6].strip() if len(line) >= 6 else ""
            if rec == "TER":
                chain_block += 1
                continue
            if rec not in {"ATOM", "HETATM"}:
                continue

            atom_name = line[12:16].strip() if len(line) >= 16 else ""
            res_name = line[17:20].strip() if len(line) >= 20 else "UNK"
            chain_id = line[21:22].strip() if len(line) >= 22 else ""
            res_seq = line[22:26].strip() if len(line) >= 26 else "0"

            x = float(line[30:38]) if len(line) >= 38 else 0.0
            y = float(line[38:46]) if len(line) >= 46 else 0.0
            z = float(line[46:54]) if len(line) >= 54 else 0.0

            elem_col = line[76:78] if len(line) >= 78 else ""
            element = _infer_pdb_element(atom_name, res_name, elem_col)

            idx = len(coord)
            xyz = [x * ANGSTROM_TO_BOHR, y * ANGSTROM_TO_BOHR, z * ANGSTROM_TO_BOHR]
            coord.append(xyz)
            atom_type.append(element)

            chain_key = (chain_block, chain_id)
            res_key = (chain_block, chain_id, res_name, res_seq)

            if chain_key not in chain_to_res:
                chain_order.append(chain_key)
                chain_to_res[chain_key] = []

            if res_key not in res_to_atoms:
                chain_to_res[chain_key].append(res_key)
                res_to_atoms[res_key] = []

            res_to_atoms[res_key].append((element, xyz, idx))

        coord_arr = jnp.asarray(coord, dtype=jnp.float64)

        fragments: list[Fragment] = []
        for chain_key in chain_order:
            res_list = chain_to_res[chain_key]
            for ridx, res_key in enumerate(res_list):
                _chain_block, _chain_id, res_name, _res_seq = res_key
                atoms = list(res_to_atoms[res_key])
                frag_type = recognize_fragment(res_name)

                if frag_type.kind != "Residue":
                    fragments.append(_fragment_from_atoms(frag_type, atoms))
                    continue

                try:
                    expected_natom = FragmentType.get_natom(res_name)
                except Exception:
                    expected_natom = len(atoms)

                is_chain_head = ridx == 0
                is_chain_tail = ridx == (len(res_list) - 1)

                head_atoms: list[tuple[Element, list[float], int]] = []
                tail_atoms: list[tuple[Element, list[float], int]] = []

                work = atoms
                if is_chain_head:
                    while len(work) > expected_natom and work and work[0][0] == Element.H:
                        head_atoms.append(work.pop(0))

                if is_chain_tail:
                    while len(work) > expected_natom and work and work[-1][0] in {Element.O, Element.H}:
                        tail_atoms.insert(0, work.pop())

                for atom in head_atoms:
                    fragments.append(_fragment_from_atoms(FragmentType.head(atom[0]), [atom]))

                if work:
                    fragments.append(_fragment_from_atoms(frag_type, work))

                for atom in tail_atoms:
                    fragments.append(_fragment_from_atoms(FragmentType.tail(atom[0]), [atom]))

        return cls(
            natom=len(coord),
            coord=coord_arr,
            cell=None,
            atom_type=atom_type,
            pot=0.0,
            force=None,
            fragment=fragments,
        )

    def write_pdb(self, filename: str, create_new_file: bool, step: int) -> None:
        mode = "w" if create_new_file else "a"
        with open(filename, mode, encoding="utf-8") as f:
            if create_new_file:
                f.write("TITLE     PDB file created by RTIP-JAX\n")
            f.write(f"REMARK    , Step = {step:8d}, E = {self.pot:15.8f}\n")

            coord_ang = np.asarray(self.coord) * BOHR_TO_ANGSTROM
            for i in range(self.natom):
                elem = self.atom_type[i].value
                f.write(
                    "ATOM  {:>5} {:>4}              {:8.3f}{:8.3f}{:8.3f}  1.00  0.00          {:>2}\n".format(
                        (i + 1) % 100000,
                        elem,
                        coord_ang[i, 0],
                        coord_ang[i, 1],
                        coord_ang[i, 2],
                        elem,
                    )
                )
            f.write("END\n")

    def to_system(self) -> System:
        return System(
            natom=self.natom,
            coord=jnp.asarray(self.coord),
            cell=None if self.cell is None else jnp.asarray(self.cell),
            atom_type=list(self.atom_type),
            atom_add_pot=None,
            mutable=None,
            pot=self.pot,
        )



def _fragment_from_atoms(fragment_type: FragmentType, atoms: list[tuple[Element, list[float], int]]) -> Fragment:
    frag_coord = jnp.asarray([a[1] for a in atoms], dtype=jnp.float64)
    frag_atoms = [a[0] for a in atoms]
    frag_indices = [a[2] for a in atoms]
    return Fragment(
        fragment_type=fragment_type,
        natom=len(atoms),
        coord=frag_coord,
        atom_type=frag_atoms,
        pot=0.0,
        force=None,
        atom_list=frag_indices,
        neighbor_list=None,
    )



def recognize_fragment(name: str) -> FragmentType:
    key = name.strip()
    try:
        return FragmentType.from_str(key)
    except Exception:
        if key:
            try:
                return FragmentType.atom(Element.from_str(key))
            except Exception:
                pass
        return FragmentType.atom(Element.C)



def read_pdb(filename: str) -> ProteinSystem:
    return ProteinSystem.read_pdb(filename)



def write_pdb(s: ProteinSystem, filename: str, create_new_file: bool, step: int) -> None:
    s.write_pdb(filename, create_new_file, step)



def to_system(s: ProteinSystem) -> System:
    return s.to_system()
