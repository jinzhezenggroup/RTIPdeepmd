from __future__ import annotations

import typing as t
from dataclasses import dataclass
from enum import Enum, auto

import jax.numpy as jnp

from . import protein
from . import intrafragment_descriptor
from . import interfragment_descriptor
from . import global_nn
from . import training_data
from . import training

__all__ = ["protein", "intrafragment_descriptor", "interfragment_descriptor", "global_nn", "training_data", "training"]
