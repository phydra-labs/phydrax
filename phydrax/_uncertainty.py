#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

from .typing import parse


UncertaintySource: TypeAlias = Literal[
    "epistemic",
    "input",
    "observation",
    "process",
    "numerical",
]


def validate_uncertainty_source(
    source: str,
    /,
    *,
    owner: str = "uncertainty source",
) -> UncertaintySource:
    """Validate and narrow one uncertainty-source label."""
    return parse(source, UncertaintySource, owner)


__all__ = [
    "UncertaintySource",
    "validate_uncertainty_source",
]
