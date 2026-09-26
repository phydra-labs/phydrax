#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import get_args, Literal, TypeAlias

from .typing import parse


UncertaintySource: TypeAlias = Literal[
    "epistemic",
    "input",
    "observation",
    "process",
    "numerical",
]
UNCERTAINTY_SOURCES: tuple[UncertaintySource, ...] = get_args(UncertaintySource)


def validate_uncertainty_source(
    source: str,
    /,
    *,
    owner: str = "uncertainty source",
) -> UncertaintySource:
    """Validate and narrow one uncertainty-source label."""
    return parse(source, UncertaintySource, owner)


__all__ = [
    "UNCERTAINTY_SOURCES",
    "UncertaintySource",
    "validate_uncertainty_source",
]
