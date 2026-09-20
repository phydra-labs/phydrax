#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jax import core as jax_core

from .._mass import (
    EstimatedMass,
    ExactMass,
    known_mass_value,
    Mass,
    product_mass,
    require_exact_mass,
    scale_mass,
    sum_mass,
    UnknownMass,
)
from .._strict import StrictModule


MeasureKind = Literal[
    "lebesgue",
    "hausdorff",
    "probability",
    "counting",
    "dirac",
    "trajectory",
    "riemannian",
    "external",
]


class BaseMeasure(StrictModule):
    """Measure family and its currently known total mass."""

    kind: MeasureKind = eqx.field(static=True)
    mass: Mass
    normalized: bool = eqx.field(static=True)

    def __init__(
        self,
        kind: MeasureKind,
        mass: Mass,
        /,
        *,
        normalized: bool = False,
    ):
        if kind not in (
            "lebesgue",
            "hausdorff",
            "probability",
            "counting",
            "dirac",
            "trajectory",
            "riemannian",
            "external",
        ):
            raise ValueError(f"Unknown measure kind {kind!r}.")
        if not isinstance(mass, (ExactMass, EstimatedMass, UnknownMass)):
            raise TypeError("BaseMeasure.mass must be an explicit Mass descriptor.")
        if normalized and isinstance(mass, ExactMass):
            invalid = ~jnp.isclose(mass.value, 1.0)
            if isinstance(invalid, jax_core.Tracer):
                mass = eqx.tree_at(
                    lambda value: value.value,
                    mass,
                    eqx.error_if(
                        mass.value,
                        invalid,
                        "A normalized exact measure must have unit mass.",
                    ),
                )
            elif bool(invalid):
                raise ValueError("A normalized exact measure must have unit mass.")
        self.kind = kind
        self.mass = mass
        self.normalized = bool(normalized)


__all__ = [
    "BaseMeasure",
    "ExactMass",
    "EstimatedMass",
    "Mass",
    "MeasureKind",
    "UnknownMass",
    "known_mass_value",
    "product_mass",
    "require_exact_mass",
    "scale_mass",
    "sum_mass",
]
