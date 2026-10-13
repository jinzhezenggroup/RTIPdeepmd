"""RTIP-MD parameters. Defaults equal upstream Rust RTIP-MD ``Para::new()`` (``src/io/input.rs``)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Para:
    # RTIP
    a0: float = 0.0005                  # Hartree; |a_min| changes by a0 per MD step
    # MD (leapfrog + Berendsen)
    dt: float = 0.5                     # fs
    tau: float = 10.0                   # fs, Berendsen coupling time
    temp_bath: float = 1500.0           # K
    max_step: int = 100_000_000
    print_step: int = 1                 # trajectory frame interval
    # Evolution control
    decreasing_multiple: float = 2.0    # a_min += decreasing_multiple * a0 per step while Decreasing
    decreasing_bound: float = 0.5       # leave Decreasing once a_min > decreasing_bound * a_min(at bond change)
    split_step: int | None = 100        # re-split molecules every split_step steps (None: never)
    ignored_pair: list[tuple[str, str]] = field(
        default_factory=lambda: [("H", "O"), ("C", "O"), ("H", "Ca"), ("C", "Ca"), ("O", "Ca"), ("Ca", "Ca")]
    )

    def to_dict(self) -> dict:
        return asdict(self)
