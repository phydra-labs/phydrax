#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import PreparedSurfelDiscretization
from ._geometry import SurfelGeometryState


class SurfelQuadratureResult(StrictModule):
    integral: Array
    average: Array
    total_measure: Array
    finite: Array
    successful: Array


class SurfelQuadraturePlan(NonTrainableState, StrictModule):
    """Integrate arbitrary surfel fields using physical surface measure."""

    discretization: PreparedSurfelDiscretization
    deterministic: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: PreparedSurfelDiscretization,
        /,
        *,
        deterministic: bool = False,
    ) -> None:
        if not isinstance(discretization, PreparedSurfelDiscretization):
            raise TypeError("discretization must be PreparedSurfelDiscretization.")
        self.discretization = discretization
        self.deterministic = bool(deterministic)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "surfel-quadrature-plan",
                "discretization": discretization.prepared_id,
                "deterministic": bool(deterministic),
            }
        )

    def evaluate(
        self,
        geometry: SurfelGeometryState,
        values: ArrayLike,
        /,
    ) -> SurfelQuadratureResult:
        if not isinstance(geometry, SurfelGeometryState):
            raise TypeError("geometry must be SurfelGeometryState.")
        if geometry.discretization.prepared_id != self.discretization.prepared_id:
            raise ValueError("Surfel geometry uses a different discretization.")
        value = jnp.asarray(values, dtype=geometry.position.dtype)
        if value.shape[0] != self.discretization.capacity:
            raise ValueError("values must have one leading value per surfel.")
        trailing = (1,) * (value.ndim - 1)
        active = geometry.active_mask
        finite_values = jnp.all(jnp.isfinite(value).reshape((value.shape[0], -1)), axis=1)
        valid = active & finite_values
        weighted = value * geometry.physical_surface_weight.reshape(
            geometry.physical_surface_weight.shape + trailing
        )
        weighted = jnp.where(valid.reshape(valid.shape + trailing), weighted, 0.0)
        if self.deterministic:
            initial = jnp.zeros(value.shape[1:], dtype=value.dtype)

            def add_one(index, current):
                return current + weighted[index]

            integral = jax.lax.fori_loop(0, value.shape[0], add_one, initial)
        else:
            integral = jnp.sum(weighted, axis=0)
        total_measure = jnp.sum(jnp.where(valid, geometry.physical_surface_weight, 0.0))
        average = integral / jnp.where(total_measure > 0.0, total_measure, 1.0)
        finite = (
            jnp.all(~active | finite_values)
            & jnp.all(jnp.isfinite(integral))
            & jnp.all(jnp.isfinite(average))
            & jnp.isfinite(total_measure)
        )
        successful = geometry.evidence.successful & finite & (total_measure > 0.0)
        return SurfelQuadratureResult(
            integral=integral,
            average=average,
            total_measure=total_measure,
            finite=finite,
            successful=successful,
        )


__all__ = ["SurfelQuadraturePlan", "SurfelQuadratureResult"]
