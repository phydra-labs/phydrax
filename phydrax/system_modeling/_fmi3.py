#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class FMI3Contract:
    model_identifier: str
    mode: Literal["model-exchange", "co-simulation", "scheduled-execution"]
    clock_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.model_identifier:
            raise ValueError("FMI model identifier is required.")


__all__ = ["FMI3Contract"]
