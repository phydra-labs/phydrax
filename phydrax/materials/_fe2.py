#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..ein import contract


@dataclass(frozen=True, slots=True)
class FE2Result:
    average_stress: Array
    consistent_tangent: Array
    micro_residual_norm: Array


@dataclass(frozen=True, slots=True)
class FE2Plan:
    micro_solver: Callable[[Array], FE2Result]

    def evaluate(self, macro_strain: ArrayLike, /) -> FE2Result:
        result = self.micro_solver(jnp.asarray(macro_strain))
        if not isinstance(result, FE2Result):
            raise TypeError("FE2 micro solver returned invalid result.")
        return result


@dataclass(frozen=True, slots=True)
class LinearFE2Plan:
    """Linear RVE localization with Hill–Mandel energy diagnostics."""

    strain_localization: Array
    stiffness: Array
    measure_weights: Array

    @classmethod
    def create(
        cls,
        strain_localization: ArrayLike,
        stiffness: ArrayLike,
        measure_weights: ArrayLike,
        /,
    ):
        localization = jnp.asarray(strain_localization)
        moduli = jnp.asarray(stiffness)
        weights = jnp.asarray(measure_weights)
        if localization.ndim != 3 or localization.shape[-1] != localization.shape[-2]:
            raise ValueError("FE2 localization requires shape (point, strain, strain).")
        if moduli.shape != localization.shape:
            raise ValueError("FE2 stiffness and localization arrays must align.")
        if weights.shape != (localization.shape[0],) or not bool(jnp.all(weights > 0)):
            raise ValueError("FE2 measure weights must align and be positive.")
        return cls(localization, moduli, weights)

    def evaluate(self, macro_strain: ArrayLike, /) -> FE2Result:
        strain = jnp.asarray(macro_strain)
        if strain.shape != (self.strain_localization.shape[-1],):
            raise ValueError("Macro strain does not match the FE2 strain basis.")
        weights = self.measure_weights / jnp.sum(self.measure_weights)
        micro_strain = contract("qij,j->qi", self.strain_localization, strain)
        micro_stress = contract("qij,qj->qi", self.stiffness, micro_strain)
        average_stress = contract("q,qi->i", weights, micro_stress)
        consistent_tangent = contract(
            "q,qij,qjk->ik", weights, self.stiffness, self.strain_localization
        )
        micro_work = contract("q,qi,qi->", weights, micro_stress, micro_strain)
        macro_work = contract("i,i->", average_stress, strain)
        residual = jnp.abs(micro_work - macro_work)
        return FE2Result(average_stress, consistent_tangent, residual)


__all__ = ["FE2Plan", "FE2Result", "LinearFE2Plan"]
