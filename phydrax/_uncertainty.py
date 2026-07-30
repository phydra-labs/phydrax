#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias


UncertaintySource: TypeAlias = Literal[
    "epistemic",
    "input",
    "observation",
    "process",
    "numerical",
]
UNCERTAINTY_SOURCES: tuple[UncertaintySource, ...] = (
    "epistemic",
    "input",
    "observation",
    "process",
    "numerical",
)


def validate_uncertainty_source(
    source: str,
    /,
    *,
    owner: str = "uncertainty source",
) -> UncertaintySource:
    """Validate and narrow one uncertainty-source label."""
    if source not in UNCERTAINTY_SOURCES:
        choices = ", ".join(repr(value) for value in UNCERTAINTY_SOURCES)
        raise ValueError(f"{owner} must be one of {choices}; got {source!r}.")
    return source  # type: ignore[return-value]


__all__ = [
    "UNCERTAINTY_SOURCES",
    "UncertaintySource",
    "validate_uncertainty_source",
]
