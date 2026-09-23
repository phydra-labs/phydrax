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
        if not isinstance(self.model_identifier, str) or not self.model_identifier:
            raise ValueError("FMI model identifier is required.")
        if self.mode not in ("model-exchange", "co-simulation", "scheduled-execution"):
            raise ValueError("FMI mode is unsupported.")
        if any(
            not isinstance(clock, str) or not clock for clock in self.clock_ids
        ) or len(set(self.clock_ids)) != len(self.clock_ids):
            raise ValueError("FMI clock identifiers must be unique and nonempty.")
        if self.mode == "scheduled-execution" and not self.clock_ids:
            raise ValueError("Scheduled-execution FMI contracts require a clock.")


__all__ = ["FMI3Contract"]
