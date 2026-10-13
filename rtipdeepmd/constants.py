"""Physical constants and element data.

Ported from the upstream Rust RTIP-MD ``src/common/constants.rs``
(https://github.com/MillenniumDream/RTIP-MD). Internal units are atomic units:
Bohr, Hartree, electron mass, and atomic time.
"""

from __future__ import annotations

import math

PI = math.pi

C_LIGHT = 299792458.0
H_PLANCK = 6.62606896e-34
BOLTZMANN = 1.3806504e-23
RYDBERG = 10973731.568527

BOHR_TO_ANGSTROM = 0.52917720859
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

HARTREE_TO_JOULE = 2.0 * RYDBERG * H_PLANCK * C_LIGHT
AU_TO_FEMTOSECOND = 1.0e15 / (4.0 * PI * RYDBERG * C_LIGHT)
FEMTOSECOND_TO_AU = 1.0 / AU_TO_FEMTOSECOND

EV_TO_HARTREE = 0.03674932217565499

# Rust `MASSUNIT`: atomic mass unit expressed in electron masses. MD forces are in
# Hartree/Bohr and time in atomic units, so masses must be in electron masses.
MASSUNIT = 1822.88484264550

# Atomic numbers, identical to Rust `STR_TO_ATOMIC_NUMBER`.
_ATOMIC_NUMBER = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19,
    "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28,
    "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37,
    "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46,
    "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55,
    "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73,
    "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
    "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91,
    "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108,
    "Mt": 109, "Ds": 110, "Rg": 111,
}

# Covalent radii from "Covalent radii revisited" (Dalton Trans., 2008, 2832-2838),
# identical to Rust `STR_TO_ATOMIC_RADIUS` (before conversion to Bohr).
_COVALENT_RADIUS_ANGSTROM = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.61, "Fe": 1.52, "Co": 1.50, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22,
    "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16, "Rb": 2.20, "Sr": 1.95,
    "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42,
    "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38,
    "I": 1.39, "Xe": 1.40, "Cs": 2.44, "Ba": 2.15, "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51,
    "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46,
    "Bi": 1.48, "Po": 1.40, "At": 1.50, "Rn": 1.50, "Fr": 2.60, "Ra": 2.21, "La": 2.07,
    "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96,
    "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87,
    "Ac": 2.15, "Th": 2.06, "Pa": 2.00, "U": 1.96, "Np": 1.90, "Pu": 1.87, "Am": 1.80,
    "Cm": 1.69,
}

# Standard atomic weights in amu, identical to Rust `STR_TO_ATOMIC_MASS` / MASSUNIT.
_ATOMIC_MASS_AMU = {
    "H": 1.00794, "He": 4.002602, "Li": 6.941, "Be": 9.012182, "B": 10.811, "C": 12.0107,
    "N": 14.0067, "O": 15.9994, "F": 18.9984032, "Ne": 20.1797, "Na": 22.98976928, "Mg": 24.305,
    "Al": 26.9815386, "Si": 28.0855, "P": 30.973762, "S": 32.065, "Cl": 35.453, "Ar": 39.948,
    "K": 39.0983, "Ca": 40.078, "Sc": 44.955912, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961,
    "Mn": 54.938045, "Fe": 55.845, "Co": 58.933195, "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.64, "As": 74.9216, "Se": 78.96, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.4678, "Sr": 87.62, "Y": 88.90585, "Zr": 91.224, "Nb": 92.90638, "Mo": 95.96,
    "Tc": 97.9072, "Ru": 101.07, "Rh": 102.9055, "Pd": 106.42, "Ag": 107.8682, "Cd": 112.411,
    "In": 114.818, "Sn": 118.71, "Sb": 121.76, "Te": 127.6, "I": 126.90447, "Xe": 131.293,
    "Cs": 132.9054519, "Ba": 137.327, "Hf": 178.49, "Ta": 180.94788, "W": 183.84, "Re": 186.207,
    "Os": 190.23, "Ir": 192.217, "Pt": 195.084, "Au": 196.966569, "Hg": 200.59, "Tl": 204.3833,
    "Pb": 207.2, "Bi": 208.9804, "Po": 208.9824, "At": 209.9871, "Rn": 222.0176, "Fr": 223.0197,
    "Ra": 226.0254, "La": 138.90547, "Ce": 140.116, "Pr": 140.90765, "Nd": 144.242,
    "Pm": 144.9127, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.92535, "Dy": 162.5,
    "Ho": 164.93032, "Er": 167.259, "Tm": 168.93421, "Yb": 173.054, "Lu": 174.9668, "Ac": 227.0,
    "Th": 232.0377, "Pa": 231.03588, "U": 238.02891,
}


def check_element(symbol: str) -> str:
    """Return ``symbol`` if it is a supported element, otherwise raise ``ValueError``."""
    if symbol not in _ATOMIC_MASS_AMU or symbol not in _COVALENT_RADIUS_ANGSTROM:
        raise ValueError(f"Unsupported element: {symbol!r}")
    return symbol


def atomic_number(symbol: str) -> int:
    return _ATOMIC_NUMBER[symbol]


def atomic_radius(symbol: str) -> float:
    """Covalent radius in Bohr."""
    return _COVALENT_RADIUS_ANGSTROM[symbol] * ANGSTROM_TO_BOHR


def atomic_mass(symbol: str) -> float:
    """Atomic mass in electron masses (atomic units), as Rust `get_atomic_mass`."""
    return _ATOMIC_MASS_AMU[symbol] * MASSUNIT
