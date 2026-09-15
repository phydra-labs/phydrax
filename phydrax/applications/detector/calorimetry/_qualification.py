#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._observables import CalorimeterObservables


class CalorimeterQualificationPlan(StrictModule, NonTrainableState):
    maximum_response_error: float = eqx.field(static=True)
    maximum_layer_fraction_error: float = eqx.field(static=True)
    maximum_occupancy_error: float = eqx.field(static=True)
    maximum_tail_quantile_error: float = eqx.field(static=True)
    upper_quantile: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_response_error: float,
        maximum_layer_fraction_error: float,
        maximum_occupancy_error: float,
        maximum_tail_quantile_error: float,
        upper_quantile: float = 0.99,
    ):
        tolerances = tuple(
            map(
                float,
                (
                    maximum_response_error,
                    maximum_layer_fraction_error,
                    maximum_occupancy_error,
                    maximum_tail_quantile_error,
                ),
            )
        )
        quantile = float(upper_quantile)
        if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError(
                "Calorimeter qualification tolerances must be finite and nonnegative."
            )
        if not 0.5 < quantile < 1.0:
            raise ValueError("upper_quantile must lie in (0.5, 1).")
        (
            self.maximum_response_error,
            self.maximum_layer_fraction_error,
            self.maximum_occupancy_error,
            self.maximum_tail_quantile_error,
        ) = tolerances
        self.upper_quantile = quantile
        self.plan_id = canonical_fingerprint(
            {
                "kind": "calorimeter-qualification-plan",
                "tolerances": list(tolerances),
                "upper_quantile": quantile,
            }
        )


class CalorimeterQualification(StrictModule, NonTrainableState):
    response_error: Array
    layer_fraction_error: Array
    occupancy_error: Array
    tail_quantile_error: Array
    finite: Array
    passed: Array
    plan_id: str = eqx.field(static=True)


def qualify_calorimeter_fast_simulation(
    plan: CalorimeterQualificationPlan,
    reference: CalorimeterObservables,
    candidate: CalorimeterObservables,
    /,
) -> CalorimeterQualification:
    if (
        not isinstance(plan, CalorimeterQualificationPlan)
        or not isinstance(reference, CalorimeterObservables)
        or not isinstance(candidate, CalorimeterObservables)
    ):
        raise TypeError("Calorimeter qualification requires plan and observable values.")
    if reference.geometry_id != candidate.geometry_id:
        raise ValueError("Reference and candidate calorimeter geometries differ.")
    if (
        reference.total_energy.shape != candidate.total_energy.shape
        or reference.layer_fractions.shape != candidate.layer_fractions.shape
    ):
        raise ValueError("Reference and candidate observable supports differ.")
    scale = jnp.maximum(
        jnp.mean(reference.total_energy), jnp.finfo(reference.total_energy.dtype).tiny
    )
    response_error = (
        jnp.abs(jnp.mean(candidate.total_energy) - jnp.mean(reference.total_energy))
        / scale
    )
    layer_fraction_error = jnp.max(
        jnp.abs(
            jnp.mean(candidate.layer_fractions, axis=0)
            - jnp.mean(reference.layer_fractions, axis=0)
        )
    )
    occupancy_scale = jnp.maximum(jnp.mean(reference.occupancy), 1.0)
    occupancy_error = (
        jnp.abs(jnp.mean(candidate.occupancy) - jnp.mean(reference.occupancy))
        / occupancy_scale
    )
    reference_tail = jnp.quantile(reference.total_energy, plan.upper_quantile)
    candidate_tail = jnp.quantile(candidate.total_energy, plan.upper_quantile)
    tail_error = jnp.abs(candidate_tail - reference_tail) / jnp.maximum(
        jnp.abs(reference_tail), jnp.finfo(reference.total_energy.dtype).tiny
    )
    finite = (
        jnp.all(
            jnp.isfinite(
                jnp.asarray(
                    [response_error, layer_fraction_error, occupancy_error, tail_error]
                )
            )
        )
        & jnp.all(reference.valid)
        & jnp.all(candidate.valid)
    )
    passed = (
        finite
        & (response_error <= plan.maximum_response_error)
        & (layer_fraction_error <= plan.maximum_layer_fraction_error)
        & (occupancy_error <= plan.maximum_occupancy_error)
        & (tail_error <= plan.maximum_tail_quantile_error)
    )
    return CalorimeterQualification(
        response_error,
        layer_fraction_error,
        occupancy_error,
        tail_error,
        finite,
        passed,
        plan.plan_id,
    )


__all__ = [
    "CalorimeterQualification",
    "CalorimeterQualificationPlan",
    "qualify_calorimeter_fast_simulation",
]
