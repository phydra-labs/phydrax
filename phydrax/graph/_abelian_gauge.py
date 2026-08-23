#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..metrix import DifferentialForm, exterior_derivative
from ._cochain import CochainBoundaryKind
from ._cochain_field import CochainField
from ._continuous_bridge import ContinuousCochainBridge, integrate_form_to_cochain


class AbelianGaugeDiagnostics(StrictModule):
    """Gauge, curvature, field-equation, and source-continuity evidence."""

    gauge_curvature_residual: Array
    maxwell_residual_norm: Array
    current_continuity_residual: Array
    action: Array
    valid: Array

    def __init__(
        self,
        *,
        gauge_curvature_residual: ArrayLike,
        maxwell_residual_norm: ArrayLike,
        current_continuity_residual: ArrayLike,
        action: ArrayLike,
    ):
        self.gauge_curvature_residual = jnp.asarray(gauge_curvature_residual)
        self.maxwell_residual_norm = jnp.asarray(maxwell_residual_norm)
        self.current_continuity_residual = jnp.asarray(current_continuity_residual)
        self.action = jnp.asarray(action)
        self.valid = jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        self.gauge_curvature_residual,
                        self.maxwell_residual_norm,
                        self.current_continuity_residual,
                        self.action,
                    )
                )
            )
        )


class AbelianBridgeReport(StrictModule):
    """Continuous/discrete curvature commutation evidence."""

    potential: CochainField
    discrete_curvature: CochainField
    projected_curvature: CochainField
    maximum_residual: Array
    valid: Array

    def __init__(
        self,
        potential: CochainField,
        discrete_curvature: CochainField,
        projected_curvature: CochainField,
        maximum_residual: ArrayLike,
        /,
        *,
        tolerance: float,
    ):
        self.potential = potential
        self.discrete_curvature = discrete_curvature
        self.projected_curvature = projected_curvature
        self.maximum_residual = jnp.asarray(maximum_residual)
        self.valid = jnp.isfinite(self.maximum_residual) & (
            self.maximum_residual <= tolerance
        )


class AbelianMaxwellOperator(StrictModule):
    """Reusable metric-cochain Maxwell residual and action consumer."""

    def __call__(
        self,
        potential: CochainField,
        current: CochainField | None = None,
        /,
    ) -> tuple[CochainField, Array]:
        return (
            abelian_maxwell_residual(potential, current),
            abelian_maxwell_action(potential, current),
        )


def _require_degree(field: CochainField, degree: int, name: str, /) -> None:
    if not isinstance(field, CochainField):
        raise TypeError(f"{name} must be a CochainField.")
    if field.degree != degree:
        raise ValueError(f"{name} must have cochain degree {degree}.")


def abelian_gauge_transform(
    potential: CochainField,
    parameter: CochainField,
    /,
) -> CochainField:
    """Return ``A + d chi``."""
    _require_degree(potential, 1, "potential")
    _require_degree(parameter, 0, "parameter")
    return potential.add(
        parameter.exterior_derivative(),
        field_id=f"gauge({potential.field_id})",
    )


def abelian_curvature(potential: CochainField, /) -> CochainField:
    """Return the degree-two curvature ``F = dA``."""
    _require_degree(potential, 1, "potential")
    return potential.exterior_derivative(field_id=f"F({potential.field_id})")


def abelian_maxwell_residual(
    potential: CochainField,
    current: CochainField | None = None,
    /,
) -> CochainField:
    """Return ``delta dA + J``."""
    curvature = abelian_curvature(potential)
    residual = curvature.codifferential(field_id=f"deltaF({potential.field_id})")
    if current is None:
        return residual
    _require_degree(current, 1, "current")
    return residual.add(current, field_id=f"maxwell({potential.field_id})")


def abelian_current_continuity(current: CochainField, /) -> Array:
    _require_degree(current, 1, "current")
    return current.codifferential(
        field_id=f"continuity({current.field_id})"
    ).norm_squared()


def abelian_maxwell_action(
    potential: CochainField,
    current: CochainField | None = None,
    /,
) -> Array:
    curvature = abelian_curvature(potential)
    action = 0.5 * curvature.norm_squared()
    if current is None:
        return action
    _require_degree(current, 1, "current")
    potential._require_compatible(current)
    return action - potential.inner(current)


def validate_abelian_gauge_system(
    potential: CochainField,
    parameter: CochainField,
    /,
    *,
    current: CochainField | None = None,
) -> AbelianGaugeDiagnostics:
    transformed = abelian_gauge_transform(potential, parameter)
    curvature = abelian_curvature(potential)
    transformed_curvature = abelian_curvature(transformed)
    gauge_residual = jnp.max(jnp.abs(transformed_curvature.values - curvature.values))
    maxwell = abelian_maxwell_residual(potential, current)
    continuity = (
        jnp.asarray(0.0, dtype=maxwell.values.real.dtype)
        if current is None
        else abelian_current_continuity(current)
    )
    return AbelianGaugeDiagnostics(
        gauge_curvature_residual=gauge_residual,
        maxwell_residual_norm=maxwell.norm_squared(),
        current_continuity_residual=continuity,
        action=abelian_maxwell_action(potential, current),
    )


def project_abelian_gauge_field(
    potential: DifferentialForm,
    bridge: ContinuousCochainBridge,
    /,
    *,
    boundary_policy: CochainBoundaryKind = "absolute",
    tolerance: float = 1e-8,
) -> AbelianBridgeReport:
    """Project ``A`` and compare ``d(project A)`` with ``project(dA)``."""
    if not isinstance(potential, DifferentialForm) or potential.degree != 1:
        raise ValueError("potential must be a degree-one DifferentialForm.")
    if not isinstance(bridge, ContinuousCochainBridge):
        raise TypeError("bridge must be a ContinuousCochainBridge.")
    projection = integrate_form_to_cochain(potential, bridge)
    discrete_potential = CochainField(
        bridge.complex,
        projection.values,
        1,
        boundary_policy=boundary_policy,
        field_id="projected-potential",
    )
    discrete_curvature = abelian_curvature(discrete_potential)
    smooth_curvature = exterior_derivative(potential)
    curvature_projection = integrate_form_to_cochain(smooth_curvature, bridge)
    projected_curvature = CochainField(
        bridge.complex,
        curvature_projection.values,
        2,
        boundary_policy=boundary_policy,
        field_id="projected-curvature",
    )
    residual = jnp.max(
        jnp.abs(discrete_curvature.active_values - projected_curvature.active_values)
    )
    return AbelianBridgeReport(
        discrete_potential,
        discrete_curvature,
        projected_curvature,
        residual,
        tolerance=tolerance,
    )


__all__ = [
    "AbelianBridgeReport",
    "AbelianMaxwellOperator",
    "AbelianGaugeDiagnostics",
    "abelian_current_continuity",
    "abelian_curvature",
    "abelian_gauge_transform",
    "abelian_maxwell_action",
    "abelian_maxwell_residual",
    "project_abelian_gauge_field",
    "validate_abelian_gauge_system",
]
