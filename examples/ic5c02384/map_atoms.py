"""Reorder the SI product / TS of a reaction into the atom order of reactant 1 + reactant 2.

The SI lists product and TS atoms in a different order than the separate reactants. The mapping
is found by graph matching: the bond graph of each reactant (bonds < 1.25 (r_i + r_j)) must be a
subgraph of the product bond graph (bonds < 1.6 (r_i + r_j), allowing for stretched bonds), with
elements preserved. The leading atoms of reactant 1 that have the same elements in the same order
as the product (the common core of the 1-R complexes) are fixed to their own index. Among the
equivalent matches (e.g. permuted methyl hydrogens) the one with the lowest RMSD is kept.

Usage: python map_atoms.py reactions/1-tBu__CO2 [...]   (writes product_ordered.xyz, ts_ordered.xyz)
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rtipdeepmd.constants import BOHR_TO_ANGSTROM, atomic_radius  # noqa: E402

MAX_MATCHES = 5000


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    n = int(lines[0])
    body = [l.split() for l in lines[2 : 2 + n]]
    return [b[0] for b in body], np.array([[float(x) for x in b[1:4]] for b in body])


def graph(types: list[str], x: np.ndarray, mult: float, nodes: list[int] | None = None, fixed: int = 0) -> nx.Graph:
    nodes = list(range(len(types))) if nodes is None else nodes
    r = np.array([atomic_radius(t) * BOHR_TO_ANGSTROM for t in types])
    g = nx.Graph()
    for i in nodes:
        g.add_node(i, label=f"{types[i]}#{i}" if i < fixed else types[i])
    for i, j in itertools.combinations(nodes, 2):
        if np.linalg.norm(x[i] - x[j]) < mult * (r[i] + r[j]):
            g.add_edge(i, j)
    return g


def rmsd_after_fit(xa: np.ndarray, xb: np.ndarray) -> float:
    a, b = xa - xa.mean(0), xb - xb.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(np.mean(np.sum((a @ rot - b) ** 2, axis=1))))


def best_monomorphism(g_small: nx.Graph, g_big: nx.Graph, x_small: np.ndarray, x_big: np.ndarray) -> tuple[dict, float]:
    """Map every node of g_small onto g_big keeping labels and edges; lowest-RMSD match."""
    matcher = isomorphism.GraphMatcher(g_big, g_small, node_match=lambda a, b: a["label"] == b["label"])
    best, best_rmsd = None, np.inf
    for k, m in enumerate(matcher.subgraph_monomorphisms_iter()):
        inv = {small: big for big, small in m.items()}
        idx = sorted(inv)
        r = rmsd_after_fit(x_small[idx], x_big[[inv[i] for i in idx]])
        if r < best_rmsd:
            best, best_rmsd = inv, r
        if k >= MAX_MATCHES:
            break
    if best is None:
        raise ValueError("no element- and bond-preserving mapping found")
    return best, best_rmsd


def map_one(t1, x1, t2, x2, tp, xp) -> tuple[list[int], float, float]:
    longest = 0
    while longest < len(t1) and t1[longest] == tp[longest]:
        longest += 1
    # The common element prefix can run past the core by coincidence; shorten it until a mapping exists.
    for prefix in range(longest, -1, -1):
        try:
            return map_with_core(t1, x1, t2, x2, tp, xp, prefix)
        except ValueError:
            continue
    raise ValueError("no element- and bond-preserving mapping found")


def map_with_core(t1, x1, t2, x2, tp, xp, prefix: int) -> tuple[list[int], float, float]:
    # core atoms keep their index: label them uniquely in both graphs
    g1 = graph(t1, x1, 1.25, fixed=prefix)
    gp = graph(tp, xp, 1.6, fixed=prefix)
    m1, r1 = best_monomorphism(g1, gp, x1, xp)
    used = set(m1.values())
    rest = [j for j in range(len(tp)) if j not in used]
    g2 = graph(t2, x2, 1.25)
    gp_rest = graph(tp, xp, 1.6, nodes=rest)
    m2, r2 = best_monomorphism(g2, gp_rest, x2, xp)
    order = [m1[i] for i in range(len(t1))] + [m2[i] for i in range(len(t2))]
    return order, r1, r2


def map_reaction(rdir: Path) -> dict[str, tuple[float, float]]:
    t1, x1 = read_xyz(rdir / "1.xyz")
    t2, x2 = read_xyz(rdir / "2.xyz")
    out = {}
    for name in ("product", "ts"):
        if not (rdir / f"{name}.xyz").exists():
            continue
        tp, xp = read_xyz(rdir / f"{name}.xyz")
        order, r1, r2 = map_one(t1, x1, t2, x2, tp, xp)
        assert sorted(order) == list(range(len(tp))) and [tp[j] for j in order] == t1 + t2
        (rdir / f"{name}_ordered.xyz").write_text(
            f"{len(order)}\n{name} of {rdir.name} in the atom order of 1.xyz + 2.xyz\n"
            + "".join(f"{tp[j]:>2} {xp[j, 0]:14.8f} {xp[j, 1]:14.8f} {xp[j, 2]:14.8f}\n" for j in order)
        )
        out[name] = (round(r1, 3), round(r2, 3))
    return out


def main() -> None:
    for arg in sys.argv[1:]:
        print(Path(arg).name, "RMSD (reactant 1, reactant 2) after mapping:", map_reaction(Path(arg)))


if __name__ == "__main__":
    main()
