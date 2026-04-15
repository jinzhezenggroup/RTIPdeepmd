from __future__ import annotations

import math
import typing as t
from dataclasses import dataclass
from enum import Enum


# Global constants
ROOT_RANK = 0

PI = 3.141592653589793
GOLDEN_RATIO1 = 0.618033988749895
GOLDEN_RATIO2 = 1.618033988749895

C_LIGHT = 299792458.0
H_PLANCK = 6.62606896e-34
BOLTZMANN = 1.3806504e-23
RYBDERG = 10973731.568527

BOHR_TO_ANGSTROM = 0.52917720859
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

HARTREE_TO_JOULE = 2.0 * RYBDERG * H_PLANCK * C_LIGHT
JOULE_TO_HARTREE = 1.0 / HARTREE_TO_JOULE

AU_TO_FEMTOSECOND = 1.0e15 / (4.0 * PI * RYBDERG * C_LIGHT)
FEMTOSECOND_TO_AU = 1.0 / AU_TO_FEMTOSECOND


Device = t.Any


class Element(Enum):
    H = "H"
    He = "He"
    Li = "Li"
    Be = "Be"
    B = "B"
    C = "C"
    N = "N"
    O = "O"
    F = "F"
    Ne = "Ne"
    Na = "Na"
    Mg = "Mg"
    Al = "Al"
    Si = "Si"
    P = "P"
    S = "S"
    Cl = "Cl"
    Ar = "Ar"
    K = "K"
    Ca = "Ca"
    Sc = "Sc"
    Ti = "Ti"
    V = "V"
    Cr = "Cr"
    Mn = "Mn"
    Fe = "Fe"
    Co = "Co"
    Ni = "Ni"
    Cu = "Cu"
    Zn = "Zn"
    Ga = "Ga"
    Ge = "Ge"
    As = "As"
    Se = "Se"
    Br = "Br"
    Kr = "Kr"
    Rb = "Rb"
    Sr = "Sr"
    Y = "Y"
    Zr = "Zr"
    Nb = "Nb"
    Mo = "Mo"
    Tc = "Tc"
    Ru = "Ru"
    Rh = "Rh"
    Pd = "Pd"
    Ag = "Ag"
    Cd = "Cd"
    In = "In"
    Sn = "Sn"
    Sb = "Sb"
    Te = "Te"
    I = "I"
    Xe = "Xe"
    Cs = "Cs"
    Ba = "Ba"
    Hf = "Hf"
    Ta = "Ta"
    W = "W"
    Re = "Re"
    Os = "Os"
    Ir = "Ir"
    Pt = "Pt"
    Au = "Au"
    Hg = "Hg"
    Tl = "Tl"
    Pb = "Pb"
    Bi = "Bi"
    Po = "Po"
    At = "At"
    Rn = "Rn"
    Fr = "Fr"
    Ra = "Ra"
    Rf = "Rf"
    Db = "Db"
    Sg = "Sg"
    Bh = "Bh"
    Hs = "Hs"
    Mt = "Mt"
    Ds = "Ds"
    Rg = "Rg"
    La = "La"
    Ce = "Ce"
    Pr = "Pr"
    Nd = "Nd"
    Pm = "Pm"
    Sm = "Sm"
    Eu = "Eu"
    Gd = "Gd"
    Tb = "Tb"
    Dy = "Dy"
    Ho = "Ho"
    Er = "Er"
    Tm = "Tm"
    Yb = "Yb"
    Lu = "Lu"
    Ac = "Ac"
    Th = "Th"
    Pa = "Pa"
    U = "U"
    Np = "Np"
    Pu = "Pu"
    Am = "Am"
    Cm = "Cm"
    Bk = "Bk"
    Cf = "Cf"
    Es = "Es"
    Fm = "Fm"
    Md = "Md"
    No = "No"
    Lr = "Lr"

    @classmethod
    def from_str(cls, element: str) -> "Element":
        key = element.strip()
        if key in _STR_TO_ELEMENT:
            return _STR_TO_ELEMENT[key]
        raise ValueError(f"Illegal element type: {element}")

    def get_atomic_number(self) -> int:
        return _ATOMIC_NUMBER[self]

    def get_atomic_radius(self) -> float:
        # Bohr
        return _ATOMIC_RADIUS.get(self, 1.5 * ANGSTROM_TO_BOHR)

    def get_atomic_mass(self) -> float:
        # Atomic mass unit (approximate fallback)
        return _ATOMIC_MASS.get(self, float(self.get_atomic_number()))

    @staticmethod
    def get_element_number() -> int:
        return len(_ATOMIC_NUMBER)


class AminoAcid(Enum):
    GLY = "GLY"
    ALA = "ALA"
    VAL = "VAL"
    LEU = "LEU"
    ILE = "ILE"
    SER = "SER"
    THR = "THR"
    ASP = "ASP"
    ASH = "ASH"
    ASN = "ASN"
    GLU = "GLU"
    GLH = "GLH"
    GLN = "GLN"
    LYS = "LYS"
    LYN = "LYN"
    ARG = "ARG"
    ARN = "ARN"
    CYS = "CYS"
    CYX = "CYX"
    MET = "MET"
    HID = "HID"
    HIE = "HIE"
    HIP = "HIP"
    PHE = "PHE"
    TYR = "TYR"
    TRP = "TRP"
    PRO = "PRO"


class Molecule(Enum):
    WAT = "WAT"


H_ATOM_TYPE = (Element.H,)
B_ATOM_TYPE = (Element.B,)
C_ATOM_TYPE = (Element.C,)
N_ATOM_TYPE = (Element.N,)
O_ATOM_TYPE = (Element.O,)
F_ATOM_TYPE = (Element.F,)
NA_ATOM_TYPE = (Element.Na,)
MG_ATOM_TYPE = (Element.Mg,)
SI_ATOM_TYPE = (Element.Si,)
P_ATOM_TYPE = (Element.P,)
S_ATOM_TYPE = (Element.S,)
CL_ATOM_TYPE = (Element.Cl,)
K_ATOM_TYPE = (Element.K,)
CA_ATOM_TYPE = (Element.Ca,)
V_ATOM_TYPE = (Element.V,)
CR_ATOM_TYPE = (Element.Cr,)
MN_ATOM_TYPE = (Element.Mn,)
FE_ATOM_TYPE = (Element.Fe,)
CO_ATOM_TYPE = (Element.Co,)
NI_ATOM_TYPE = (Element.Ni,)
CU_ATOM_TYPE = (Element.Cu,)
ZN_ATOM_TYPE = (Element.Zn,)
AS_ATOM_TYPE = (Element.As,)
SE_ATOM_TYPE = (Element.Se,)
BR_ATOM_TYPE = (Element.Br,)
MO_ATOM_TYPE = (Element.Mo,)
CD_ATOM_TYPE = (Element.Cd,)
SN_ATOM_TYPE = (Element.Sn,)
I_ATOM_TYPE = (Element.I,)

GLY_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.H, Element.C, Element.O)
ALA_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.H, Element.C, Element.O)
VAL_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.H, Element.C, Element.H, Element.H, Element.H, Element.C, Element.O)
LEU_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.H, Element.C, Element.H, Element.H, Element.H, Element.C, Element.O)
ILE_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.H, Element.H, Element.C, Element.O)
SER_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.O, Element.H, Element.C, Element.O)
THR_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.H, Element.O, Element.H, Element.C, Element.O)
ASP_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.O, Element.O, Element.C, Element.O)
ASH_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.O, Element.O, Element.H, Element.C, Element.O)
ASN_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.O, Element.N, Element.H, Element.H, Element.C, Element.O)
GLU_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.H, Element.C, Element.O, Element.O, Element.C, Element.O)
GLH_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.H, Element.C, Element.O, Element.O, Element.H, Element.C, Element.O)
GLN_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.H, Element.C, Element.O, Element.N, Element.H, Element.H, Element.C, Element.O)
LYS_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.H, Element.N, Element.H, Element.H, Element.H, Element.C, Element.O)
LYN_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.H, Element.N, Element.H, Element.H, Element.C, Element.O)
ARG_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.H, Element.N, Element.H, Element.C, Element.N, Element.H, Element.H, Element.N, Element.H, Element.H, Element.C, Element.O)
ARN_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.H, Element.N, Element.H, Element.C, Element.N, Element.H, Element.H, Element.N, Element.H, Element.C, Element.O)
CYS_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.S, Element.H, Element.C, Element.O)
CYX_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.S, Element.C, Element.O)
MET_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.H, Element.S, Element.C, Element.H, Element.H, Element.H, Element.C, Element.O)
HID_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.N, Element.H, Element.C, Element.H, Element.N, Element.C, Element.H, Element.C, Element.O)
HIE_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.N, Element.C, Element.H, Element.N, Element.H, Element.C, Element.H, Element.C, Element.O)
HIP_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.N, Element.H, Element.C, Element.H, Element.N, Element.H, Element.C, Element.H, Element.C, Element.O)
PHE_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.C, Element.H, Element.C, Element.H, Element.C, Element.H, Element.C, Element.H, Element.C, Element.H, Element.C, Element.O)
TYR_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.C, Element.H, Element.C, Element.H, Element.C, Element.O, Element.H, Element.C, Element.H, Element.C, Element.H, Element.C, Element.O)
TRP_ATOM_TYPE = (Element.N, Element.H, Element.C, Element.H, Element.C, Element.H, Element.H, Element.C, Element.C, Element.H, Element.N, Element.H, Element.C, Element.C, Element.H, Element.C, Element.H, Element.C, Element.H, Element.C, Element.H, Element.C, Element.C, Element.O)
PRO_ATOM_TYPE = (Element.N, Element.C, Element.H, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.H, Element.C, Element.H, Element.C, Element.O)

WAT_ATOM_TYPE = (Element.O, Element.H, Element.H)


@dataclass(frozen=True)
class FragmentType:
    kind: str
    value: t.Any

    @staticmethod
    def atom(element: Element) -> "FragmentType":
        return FragmentType("Atom", element)

    @staticmethod
    def residue(amino_acid: AminoAcid) -> "FragmentType":
        return FragmentType("Residue", amino_acid)

    @staticmethod
    def molecule(molecule: Molecule) -> "FragmentType":
        return FragmentType("Molecule", molecule)

    @staticmethod
    def head(element: Element) -> "FragmentType":
        return FragmentType("Head", element)

    @staticmethod
    def tail(element: Element) -> "FragmentType":
        return FragmentType("Tail", element)

    @staticmethod
    def from_str(fragment_type: str) -> "FragmentType":
        key = fragment_type.strip()
        if key in _STR_TO_FRAGMENT_TYPE:
            return _STR_TO_FRAGMENT_TYPE[key]
        raise ValueError(f"Illegal fragment type: {fragment_type}")

    @staticmethod
    def get_natom(fragment_type: str) -> int:
        key = fragment_type.strip()
        if key in _STR_TO_NATOM:
            return _STR_TO_NATOM[key]
        raise ValueError(f"Illegal fragment type: {fragment_type}")

    @staticmethod
    def get_atom_type(fragment_type: str) -> tuple[Element, ...]:
        key = fragment_type.strip()
        if key in _STR_TO_ATOM_TYPE:
            return _STR_TO_ATOM_TYPE[key]
        raise ValueError(f"Illegal fragment type: {fragment_type}")


_PERIODIC_ORDER: tuple[Element, ...] = (
    Element.H, Element.He, Element.Li, Element.Be, Element.B, Element.C, Element.N, Element.O, Element.F, Element.Ne,
    Element.Na, Element.Mg, Element.Al, Element.Si, Element.P, Element.S, Element.Cl, Element.Ar,
    Element.K, Element.Ca, Element.Sc, Element.Ti, Element.V, Element.Cr, Element.Mn, Element.Fe, Element.Co, Element.Ni, Element.Cu, Element.Zn,
    Element.Ga, Element.Ge, Element.As, Element.Se, Element.Br, Element.Kr,
    Element.Rb, Element.Sr, Element.Y, Element.Zr, Element.Nb, Element.Mo, Element.Tc, Element.Ru, Element.Rh, Element.Pd, Element.Ag, Element.Cd,
    Element.In, Element.Sn, Element.Sb, Element.Te, Element.I, Element.Xe,
    Element.Cs, Element.Ba, Element.La, Element.Ce, Element.Pr, Element.Nd, Element.Pm, Element.Sm, Element.Eu, Element.Gd, Element.Tb, Element.Dy, Element.Ho, Element.Er, Element.Tm, Element.Yb, Element.Lu,
    Element.Hf, Element.Ta, Element.W, Element.Re, Element.Os, Element.Ir, Element.Pt, Element.Au, Element.Hg, Element.Tl, Element.Pb, Element.Bi, Element.Po, Element.At, Element.Rn,
    Element.Fr, Element.Ra, Element.Ac, Element.Th, Element.Pa, Element.U, Element.Np, Element.Pu, Element.Am, Element.Cm, Element.Bk, Element.Cf, Element.Es, Element.Fm, Element.Md, Element.No, Element.Lr,
    Element.Rf, Element.Db, Element.Sg, Element.Bh, Element.Hs, Element.Mt, Element.Ds, Element.Rg,
)

_ATOMIC_NUMBER = {element: i + 1 for i, element in enumerate(_PERIODIC_ORDER)}

_ATOMIC_RADIUS = {
    Element.H: 0.53 * ANGSTROM_TO_BOHR,
    Element.C: 0.67 * ANGSTROM_TO_BOHR,
    Element.N: 0.56 * ANGSTROM_TO_BOHR,
    Element.O: 0.48 * ANGSTROM_TO_BOHR,
    Element.F: 0.42 * ANGSTROM_TO_BOHR,
    Element.Na: 1.90 * ANGSTROM_TO_BOHR,
    Element.Mg: 1.45 * ANGSTROM_TO_BOHR,
    Element.Al: 1.18 * ANGSTROM_TO_BOHR,
    Element.Si: 1.11 * ANGSTROM_TO_BOHR,
    Element.P: 0.98 * ANGSTROM_TO_BOHR,
    Element.S: 0.88 * ANGSTROM_TO_BOHR,
    Element.Cl: 0.79 * ANGSTROM_TO_BOHR,
    Element.K: 2.43 * ANGSTROM_TO_BOHR,
    Element.Ca: 1.94 * ANGSTROM_TO_BOHR,
    Element.Fe: 1.56 * ANGSTROM_TO_BOHR,
    Element.Cu: 1.45 * ANGSTROM_TO_BOHR,
    Element.Zn: 1.42 * ANGSTROM_TO_BOHR,
    Element.Br: 0.94 * ANGSTROM_TO_BOHR,
    Element.I: 1.15 * ANGSTROM_TO_BOHR,
    Element.Mo: 1.54 * ANGSTROM_TO_BOHR,
}

_ATOMIC_MASS = {
    Element.H: 1.00794,
    Element.C: 12.0107,
    Element.N: 14.0067,
    Element.O: 15.9994,
    Element.F: 18.9984,
    Element.Na: 22.9898,
    Element.Mg: 24.3050,
    Element.Al: 26.9815,
    Element.Si: 28.0855,
    Element.P: 30.9738,
    Element.S: 32.065,
    Element.Cl: 35.453,
    Element.K: 39.0983,
    Element.Ca: 40.078,
    Element.Fe: 55.845,
    Element.Cu: 63.546,
    Element.Zn: 65.38,
    Element.Br: 79.904,
    Element.I: 126.904,
    Element.Mo: 95.95,
}

_STR_TO_ELEMENT = {e.value: e for e in Element}
_STR_TO_ELEMENT.update(
    {
        "H+": Element.H,
        "F-": Element.F,
        "Na+": Element.Na,
        "Mg2+": Element.Mg,
        "Cl-": Element.Cl,
        "K+": Element.K,
        "Ca2+": Element.Ca,
        "Cr3+": Element.Cr,
        "Mn2+": Element.Mn,
        "Fe2+": Element.Fe,
        "Fe3+": Element.Fe,
        "Co2+": Element.Co,
        "Ni2+": Element.Ni,
        "Cu2+": Element.Cu,
        "Zn2+": Element.Zn,
        "Se2-": Element.Se,
        "Se4+": Element.Se,
        "Br-": Element.Br,
        "Sn2+": Element.Sn,
        "I-": Element.I,
    }
)

_AA_ATOM_TYPES: dict[str, tuple[Element, ...]] = {
    "GLY": GLY_ATOM_TYPE,
    "ALA": ALA_ATOM_TYPE,
    "VAL": VAL_ATOM_TYPE,
    "LEU": LEU_ATOM_TYPE,
    "ILE": ILE_ATOM_TYPE,
    "SER": SER_ATOM_TYPE,
    "THR": THR_ATOM_TYPE,
    "ASP": ASP_ATOM_TYPE,
    "ASH": ASH_ATOM_TYPE,
    "ASN": ASN_ATOM_TYPE,
    "GLU": GLU_ATOM_TYPE,
    "GLH": GLH_ATOM_TYPE,
    "GLN": GLN_ATOM_TYPE,
    "LYS": LYS_ATOM_TYPE,
    "LYN": LYN_ATOM_TYPE,
    "ARG": ARG_ATOM_TYPE,
    "ARN": ARN_ATOM_TYPE,
    "CYS": CYS_ATOM_TYPE,
    "CYX": CYX_ATOM_TYPE,
    "MET": MET_ATOM_TYPE,
    "HID": HID_ATOM_TYPE,
    "HIE": HIE_ATOM_TYPE,
    "HIP": HIP_ATOM_TYPE,
    "PHE": PHE_ATOM_TYPE,
    "TYR": TYR_ATOM_TYPE,
    "TRP": TRP_ATOM_TYPE,
    "PRO": PRO_ATOM_TYPE,
}

_STR_TO_ATOM_TYPE: dict[str, tuple[Element, ...]] = {
    e.value: (e,) for e in Element
}
_STR_TO_ATOM_TYPE.update(_AA_ATOM_TYPES)
_STR_TO_ATOM_TYPE["WAT"] = WAT_ATOM_TYPE

_STR_TO_FRAGMENT_TYPE: dict[str, FragmentType] = {
    key: FragmentType.atom(value) for key, value in _STR_TO_ELEMENT.items()
}
for aa in AminoAcid:
    _STR_TO_FRAGMENT_TYPE[aa.value] = FragmentType.residue(aa)
_STR_TO_FRAGMENT_TYPE["WAT"] = FragmentType.molecule(Molecule.WAT)

_STR_TO_NATOM: dict[str, int] = {key: len(value) for key, value in _STR_TO_ATOM_TYPE.items()}


def from_str(element: str) -> Element:
    return Element.from_str(element)


def get_atomic_number(element: Element) -> int:
    return element.get_atomic_number()


def get_atomic_radius(element: Element) -> float:
    return element.get_atomic_radius()


def get_atomic_mass(element: Element) -> float:
    return element.get_atomic_mass()


def get_element_number() -> int:
    return Element.get_element_number()


def get_natom(fragment_type: str) -> int:
    return FragmentType.get_natom(fragment_type)


def get_atom_type(fragment_type: str) -> tuple[Element, ...]:
    return FragmentType.get_atom_type(fragment_type)
