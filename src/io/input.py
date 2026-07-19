from __future__ import annotations

from dataclasses import dataclass, field

from ..common.constants import Element


@dataclass
class RtipPara:
    a0: float = 0.0005
    sigma: float = 0.75
    scale_ts_a0: float = 1.0
    scale_ts_sigma: float | None = 0.25


@dataclass
class PathSampPara:
    pot_climb: float = 0.185
    pot_drop: float = 0.02
    pot_epsilon: float = 0.00005
    f_epsilon: float = 0.001
    max_step: int = 2000
    print_step: int = 1


@dataclass
class MdPara:
    dt: float = 0.5
    tau: float = 10.0
    temp_bath: float = 1500.0
    temp_start_scale: float = 1.0
    temp_ramp_steps: int = 0
    thermostat_lam_min: float = 0.0
    thermostat_lam_max: float = 0.0
    decreasing_multiple: float = 2.0
    decreasing_bound: float = 0.5
    ignored_pair: list[tuple[Element, Element]] = field(
        default_factory=lambda: [
            (Element.H, Element.O),
            (Element.C, Element.O),
            (Element.H, Element.Ca),
            (Element.C, Element.Ca),
            (Element.O, Element.Ca),
            (Element.Ca, Element.Ca),
        ]
    )
    split_step: int | None = 100
    max_step: int = 100000000
    print_step: int = 1


@dataclass
class NnTrainPara:
    training_batch_size: int = 10
    validation_batch_size: int = 5
    max_step: int = 100
    print_step: int = 10
    input_nn_sub_dir: str = ""
    output_nn_sub_dir: str = "new"
    input_data_sub_dir: str = "monomer/Gly"


@dataclass
class Para:
    rtip_para: RtipPara = field(default_factory=RtipPara)
    path_samp_para: PathSampPara = field(default_factory=PathSampPara)
    md_para: MdPara = field(default_factory=MdPara)
    nn_train_para: NnTrainPara = field(default_factory=NnTrainPara)

    @classmethod
    def new(cls) -> "Para":
        return cls()


def new() -> Para:
    return Para.new()
