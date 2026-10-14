"""Print the --fragment option selecting the B-N=N reactive site of a methyleneborane-N2 complex.

Usage: python reactive_site.py reactions/1-tBu__CO2/1.xyz   ->   1:6,17,18
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_rtip import read_xyz, roles  # noqa: E402


def main() -> None:
    types, coord = read_xyz(Path(sys.argv[1]))
    r = roles(types, coord, len(types))
    print("1:" + ",".join(str(i + 1) for i in sorted((r["B"], r["N_alpha"], r["N_beta"]))))


if __name__ == "__main__":
    main()
