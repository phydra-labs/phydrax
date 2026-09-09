#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class HybridTurbulenceEvaluation(StrictModule):
    grid_scale: Array
    shielding: Array
    elevation: Array
    effective_distance: Array
    intermittency: Array
    rans_fraction: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class HybridRANSLESGridScalePlan(StrictModule, NonTrainableState):
    kind: Literal["maximum", "volume", "vorticity-aligned"] = eqx.field(static=True)
    coefficient: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: Literal["maximum", "volume", "vorticity-aligned"] = "vorticity-aligned",
        /,
        *,
        coefficient: float = 0.65,
    ):
        coefficient_ = float(coefficient)
        if (
            kind not in ("maximum", "volume", "vorticity-aligned")
            or not np.isfinite(coefficient_)
            or coefficient_ <= 0.0
        ):
            raise ValueError("Hybrid turbulence grid scale is invalid.")
        self.kind = kind
        self.coefficient = coefficient_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hybrid-rans-les-grid-scale",
                "scale": kind,
                "coefficient": coefficient_,
            }
        )

    def evaluate(
        self, cell_lengths: ArrayLike, vorticity_direction: ArrayLike | None = None, /
    ) -> Array:
        lengths = jnp.asarray(cell_lengths)
        if lengths.ndim < 1 or lengths.shape[-1] not in (1, 2, 3):
            raise ValueError("Cell lengths must end in the physical dimension.")
        if self.kind == "maximum":
            scale = jnp.max(lengths, axis=-1)
        elif self.kind == "volume":
            scale = jnp.prod(lengths, axis=-1) ** (1.0 / lengths.shape[-1])
        else:
            if vorticity_direction is None:
                raise ValueError("Vorticity-aligned scale requires vorticity direction.")
            direction = jnp.asarray(vorticity_direction, dtype=lengths.dtype)
            if direction.shape != lengths.shape:
                raise ValueError("Vorticity direction must match cell lengths.")
            direction = direction / jnp.maximum(
                jnp.sqrt(jnp.sum(direction * direction, axis=-1))[..., None],
                jnp.finfo(lengths.dtype).tiny,
            )
            scale = jnp.sqrt(jnp.sum((lengths * direction) ** 2, axis=-1))
        return self.coefficient * scale


class DelayedDetachedEddyPlan(StrictModule, NonTrainableState):
    kind: Literal["sa-ddes", "sa-iddes", "sst-ddes"] = eqx.field(static=True)
    grid_scale: HybridRANSLESGridScalePlan
    shielding_constant: float = eqx.field(static=True)
    elevation_constant: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: Literal["sa-ddes", "sa-iddes", "sst-ddes"],
        grid_scale: HybridRANSLESGridScalePlan,
        /,
        *,
        shielding_constant: float = 8.0,
        elevation_constant: float = 2.0,
    ):
        shield = float(shielding_constant)
        elevation = float(elevation_constant)
        if (
            kind not in ("sa-ddes", "sa-iddes", "sst-ddes")
            or not isinstance(grid_scale, HybridRANSLESGridScalePlan)
            or not np.isfinite(shield)
            or shield <= 0.0
            or not np.isfinite(elevation)
            or elevation < 0.0
        ):
            raise ValueError("DDES/IDDES kind, scale, or constants are invalid.")
        self.kind = kind
        self.grid_scale = grid_scale
        self.shielding_constant = shield
        self.elevation_constant = elevation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "delayed-detached-eddy",
                "variant": kind,
                "grid_scale": grid_scale.plan_id,
                "shielding_constant": shield,
                "elevation_constant": elevation,
            }
        )

    def evaluate(
        self,
        wall_distance: ArrayLike,
        cell_lengths: ArrayLike,
        eddy_viscosity: ArrayLike,
        molecular_viscosity: ArrayLike,
        strain_rate: ArrayLike,
        /,
        *,
        vorticity_direction: ArrayLike | None = None,
        intermittency: ArrayLike = 1.0,
    ) -> HybridTurbulenceEvaluation:
        distance = jnp.asarray(wall_distance)
        eddy = jnp.asarray(eddy_viscosity, dtype=distance.dtype)
        molecular = jnp.asarray(molecular_viscosity, dtype=distance.dtype)
        strain = jnp.asarray(strain_rate, dtype=distance.dtype)
        gamma = jnp.asarray(intermittency, dtype=distance.dtype)
        scale = self.grid_scale.evaluate(cell_lengths, vorticity_direction)
        if (
            eddy.shape != distance.shape
            or molecular.shape != distance.shape
            or strain.shape != distance.shape
            or gamma.shape not in ((), distance.shape)
        ):
            raise ValueError("Hybrid turbulence fields are incompatible.")
        gamma = jnp.broadcast_to(gamma, distance.shape)
        ratio = (eddy + molecular) / jnp.maximum(
            strain * distance * distance, jnp.finfo(distance.dtype).tiny
        )
        shielding = 1.0 - jnp.tanh((self.shielding_constant * ratio) ** 3)
        elevation = jnp.zeros_like(distance)
        if self.kind == "sa-iddes":
            elevation = jnp.exp(
                -self.elevation_constant
                * jnp.maximum(
                    distance / jnp.maximum(scale, jnp.finfo(distance.dtype).tiny) - 1.0,
                    0.0,
                )
            )
        detached_distance = jnp.minimum(distance, scale)
        effective = (
            distance
            - shielding * jnp.maximum(distance - detached_distance, 0.0)
            + elevation * scale
        )
        rans_fraction = jnp.clip(1.0 - shielding, 0.0, 1.0)
        finite = (
            jnp.isfinite(effective) & jnp.isfinite(shielding) & jnp.isfinite(elevation)
        )
        successful = (
            finite
            & (distance > 0.0)
            & (scale > 0.0)
            & (molecular > 0.0)
            & (strain >= 0.0)
            & (gamma >= 0.0)
            & (gamma <= 1.0)
        )
        return HybridTurbulenceEvaluation(
            scale,
            shielding,
            elevation,
            effective,
            gamma,
            rans_fraction,
            finite,
            successful,
            self.plan_id,
        )


class PrescribedTransitionPlan(StrictModule, NonTrainableState):
    onset_coordinate: float = eqx.field(static=True)
    transition_width: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, onset_coordinate: float, /, *, transition_width: float = 0.0):
        onset = float(onset_coordinate)
        width = float(transition_width)
        if not np.isfinite(onset) or not np.isfinite(width) or width < 0.0:
            raise ValueError("Prescribed transition onset or width is invalid.")
        self.onset_coordinate = onset
        self.transition_width = width
        self.plan_id = canonical_fingerprint(
            {
                "kind": "prescribed-transition",
                "onset_coordinate": onset,
                "transition_width": width,
            }
        )

    def intermittency(self, coordinate: ArrayLike, /) -> Array:
        value = jnp.asarray(coordinate)
        if self.transition_width == 0.0:
            return (value >= self.onset_coordinate).astype(value.dtype)
        return 0.5 * (
            1.0 + jnp.tanh((value - self.onset_coordinate) / self.transition_width)
        )


__all__ = [
    "DelayedDetachedEddyPlan",
    "HybridRANSLESGridScalePlan",
    "HybridTurbulenceEvaluation",
    "PrescribedTransitionPlan",
]
