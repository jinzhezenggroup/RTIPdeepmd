"""Build a reactant complex for an AttractivePot run from the mapped SI product.

Each relaxed reactant (1.xyz, 2.xyz) is rotated onto its own atoms in product_ordered.xyz
(written by map_atoms.py), so the small molecule faces the reactive site of 1-R with the
orientation it has in the product. The small molecule is then moved away along the line between
the two centroids until the shortest intermolecular distance is at least GAP Angstrom. The
AttractivePot run from this complex towards the product then follows the [3+2] approach instead of
dragging the small molecule around 1-R from a random orientation.

Usage: python reactant_complex.py --gap 3.0 reactions/1-tBu__CO2 [...]   (writes reactant_complex.xyz)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from map_atoms import read_xyz


def align(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``x`` rotated and translated onto ``ref`` (Kabsch, no reflection)."""
    a, b = x - x.mean(0), ref - ref.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    return a @ (u @ np.diag([1.0, 1.0, d]) @ vt) + ref.mean(0)


def min_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(x1[:, None, :] - x2[None, :, :], axis=-1)))


def build(rdir: Path, gap: float) -> float:
    t1, x1 = read_xyz(rdir / "1.xyz")
    t2, x2 = read_xyz(rdir / "2.xyz")
    tp, xp = read_xyz(rdir / "product_ordered.xyz")
    n1 = len(t1)
    if tp != t1 + t2:
        raise ValueError(f"{rdir}: product_ordered.xyz is not in the atom order of 1.xyz + 2.xyz")
    y1, y2 = align(x1, xp[:n1]), align(x2, xp[n1:])
    v = y2.mean(0) - y1.mean(0)
    v /= np.linalg.norm(v)
    while min_distance(y1, y2) < gap:
        y2 = y2 + 0.02 * v
    x = np.concatenate([y1, y2])
    (rdir / "reactant_complex.xyz").write_text(
        f"{len(tp)}\nreactant complex of {rdir.name}, shortest intermolecular distance {gap} A\n"
        + "".join(f"{t:>2} {c[0]:14.8f} {c[1]:14.8f} {c[2]:14.8f}\n" for t, c in zip(tp, x))
    )
    return float(np.linalg.norm(y2.mean(0) - y1.mean(0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reactions", nargs="+", type=Path)
    ap.add_argument("--gap", type=float, default=3.0, help="Shortest intermolecular distance (Angstrom).")
    args = ap.parse_args()
    for rdir in args.reactions:
        print(f"{rdir.name}: centroid distance {build(rdir, args.gap):.2f} A")


if __name__ == "__main__":
    main()
