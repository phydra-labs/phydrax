#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._spectral._spherical import SphericalHarmonicPlan
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._spherical import SphericalSpectralDiscretization
from ._spherical_layout import SphericalModeLayout


SphericalSpinOperatorKind: TypeAlias = Literal["raise", "lower"]
SphericalCoordinate: TypeAlias = Literal["theta", "phi"]
SphericalDerivativeRepresentation: TypeAlias = Literal["modal", "physical"]


class SphericalSpinOperatorPlan(StrictModule, NonTrainableState):
    """Exact coefficient-space eth or ethbar ladder at one fixed bandlimit."""

    kind: SphericalSpinOperatorKind = eqx.field(static=True)
    physical_units: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: SphericalSpinOperatorKind,
        /,
        *,
        physical_units: bool = True,
    ):
        if kind not in ("raise", "lower"):
            raise ValueError("Spherical spin operator kind must be 'raise' or 'lower'.")
        self.kind = kind
        self.physical_units = bool(physical_units)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spherical-spin-operator-plan",
                "operator": kind,
                "physical_units": self.physical_units,
            }
        )

    def prepare(
        self, discretization: SphericalSpectralDiscretization, /
    ) -> "PreparedSphericalSpinOperator":
        return PreparedSphericalSpinOperator(self, discretization)


class PreparedSphericalSpinOperator(StrictModule, NonTrainableState):
    plan: SphericalSpinOperatorPlan
    discretization: SphericalSpectralDiscretization
    output_layout: SphericalModeLayout
    output_transform: SphericalHarmonicPlan
    multiplier: Array
    source_spin: int = eqx.field(static=True)
    target_spin: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SphericalSpinOperatorPlan,
        discretization: SphericalSpectralDiscretization,
        /,
    ):
        if not isinstance(plan, SphericalSpinOperatorPlan):
            raise TypeError("plan must be a SphericalSpinOperatorPlan.")
        if not isinstance(discretization, SphericalSpectralDiscretization):
            raise TypeError("discretization must be spherical spectral.")
        source = discretization.layout
        target_spin = source.spin + (1 if plan.kind == "raise" else -1)
        if abs(target_spin) >= source.bandlimit:
            raise ValueError("Spin ladder target lies outside the fixed bandlimit.")
        output_layout = SphericalModeLayout(
            source.bandlimit,
            spin=target_spin,
            reality=False,
        )
        output_transform = SphericalHarmonicPlan(
            source.bandlimit,
            sampling=discretization.plan.sampling,
            spin=target_spin,
            reality=False,
            execution=discretization.plan.execution,
            max_precompute_bytes=discretization.plan.max_precompute_bytes,
        )
        degree = source.degrees.astype(discretization.quadrature_weights.dtype)
        spin = float(source.spin)
        if plan.kind == "raise":
            squared = (degree - spin) * (degree + spin + 1.0)
            sign = 1.0
        else:
            squared = (degree + spin) * (degree - spin + 1.0)
            sign = -1.0
        multiplier = sign * jnp.sqrt(jnp.maximum(squared, 0.0))
        multiplier = jnp.where(source.valid_mask, multiplier, 0.0)
        if plan.physical_units:
            multiplier = multiplier / discretization.radius
        self.plan = plan
        self.discretization = discretization
        self.output_layout = output_layout
        self.output_transform = output_transform
        self.multiplier = multiplier
        self.source_spin = source.spin
        self.target_spin = target_spin
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-spherical-spin-operator",
                "plan": plan.plan_id,
                "discretization": discretization.prepared_id,
                "source_layout": source.layout_id,
                "target_layout": output_layout.layout_id,
            }
        )

    def apply(self, coefficients: ArrayLike, /) -> Array:
        source = self.discretization.layout
        modal = source.mask_invalid(coefficients)
        _, _, channel_last = source._coefficient_axes(modal)
        leading = modal.ndim - 2 - int(channel_last)
        multiplier = self.multiplier.reshape(
            (1,) * leading + source.coefficient_shape + ((1,) if channel_last else ())
        )
        return self.output_layout.mask_invalid(modal * multiplier.astype(modal.dtype))

    def reconstruct(self, coefficients: ArrayLike, /) -> Array:
        return self.output_transform.synthesis(self.apply(coefficients))


class SphericalCoordinateDerivativeResult(StrictModule):
    values: Array
    polar_valid_mask: Array
    minimum_absolute_sine: Array
    coordinate: SphericalCoordinate = eqx.field(static=True)
    representation: SphericalDerivativeRepresentation = eqx.field(static=True)
    source_spin: int = eqx.field(static=True)
    chart: str = eqx.field(static=True)
    prepared_ids: tuple[str, str] = eqx.field(static=True)


def spherical_coordinate_derivative(
    discretization: SphericalSpectralDiscretization,
    coefficients: ArrayLike,
    /,
    *,
    coordinate: SphericalCoordinate,
    representation: SphericalDerivativeRepresentation = "physical",
    require_all_valid: bool = True,
    polar_tolerance: float | None = None,
) -> SphericalCoordinateDerivativeResult:
    """Differentiate in the declared colatitude/longitude chart.

    This is chart-valued evidence, not a global tangent-frame gradient.
    """
    if not isinstance(discretization, SphericalSpectralDiscretization):
        raise TypeError("discretization must be spherical spectral.")
    if coordinate not in ("theta", "phi"):
        raise ValueError("coordinate must be 'theta' or 'phi'.")
    if representation not in ("modal", "physical"):
        raise ValueError("representation must be 'modal' or 'physical'.")
    tolerance = (
        64.0 * np.finfo(np.asarray(discretization.points).dtype).eps
        if polar_tolerance is None
        else float(polar_tolerance)
    )
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("polar_tolerance must be finite and positive.")
    points = np.asarray(discretization.points)
    sine = np.sqrt(points[..., 0] ** 2 + points[..., 1] ** 2) / discretization.radius
    valid = sine > tolerance
    if require_all_valid and not bool(np.all(valid)):
        raise ValueError(
            "The spherical coordinate derivative chart is invalid at a sampled pole."
        )
    layout = discretization.layout
    modal = layout.mask_invalid(coefficients)
    raise_operator = SphericalSpinOperatorPlan("raise", physical_units=False).prepare(
        discretization
    )
    lower_operator = SphericalSpinOperatorPlan("lower", physical_units=False).prepare(
        discretization
    )
    if coordinate == "phi" and representation == "modal":
        _, _, channel_last = layout._coefficient_axes(modal)
        leading = modal.ndim - 2 - int(channel_last)
        orders = layout.orders.reshape(
            (1,) * leading + layout.coefficient_shape + ((1,) if channel_last else ())
        )
        values = layout.mask_invalid(1j * orders.astype(modal.dtype) * modal)
    else:
        raised = raise_operator.output_transform.synthesis(raise_operator.apply(modal))
        lowered = lower_operator.output_transform.synthesis(lower_operator.apply(modal))
        physical = discretization.reconstruct(modal)
        if coordinate == "theta":
            physical_derivative = -0.5 * (raised + lowered)
        else:
            sine_array = jnp.asarray(sine)
            cosine_array = jnp.asarray(points[..., 2] / discretization.radius)
            payload_axes = physical.ndim - 2
            sine_array = sine_array.reshape(sine_array.shape + (1,) * payload_axes)
            cosine_array = cosine_array.reshape(cosine_array.shape + (1,) * payload_axes)
            physical_derivative = (
                0.5j * sine_array * (raised - lowered)
                - 1j * layout.spin * cosine_array * physical
            )
        validity = jnp.asarray(valid).reshape(valid.shape + (1,) * (physical.ndim - 2))
        physical_derivative = jnp.where(
            validity,
            physical_derivative,
            jnp.zeros((), dtype=physical_derivative.dtype),
        )
        if representation == "physical":
            values = physical_derivative
        else:
            if layout.reality:
                physical_derivative = jnp.real(physical_derivative)
            values = discretization.project(physical_derivative)
    return SphericalCoordinateDerivativeResult(
        values=values,
        polar_valid_mask=jnp.asarray(valid),
        minimum_absolute_sine=jnp.asarray(np.min(sine)),
        coordinate=coordinate,
        representation=representation,
        source_spin=layout.spin,
        chart="colatitude-longitude",
        prepared_ids=(raise_operator.prepared_id, lower_operator.prepared_id),
    )


__all__ = [
    "PreparedSphericalSpinOperator",
    "SphericalCoordinate",
    "SphericalCoordinateDerivativeResult",
    "SphericalDerivativeRepresentation",
    "SphericalSpinOperatorKind",
    "SphericalSpinOperatorPlan",
    "spherical_coordinate_derivative",
]
