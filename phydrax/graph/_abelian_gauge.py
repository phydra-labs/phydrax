#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from ..exterior._complex import AbstractDeRhamComplex, ComplexBoundary, DiscreteForm
from ..typing import checked, parse


@final
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
    ) -> None:
        gauge = jnp.asarray(gauge_curvature_residual)
        maxwell = jnp.asarray(maxwell_residual_norm)
        continuity = jnp.asarray(current_continuity_residual)
        action_ = jnp.asarray(action)
        self.gauge_curvature_residual = gauge
        self.maxwell_residual_norm = maxwell
        self.current_continuity_residual = continuity
        self.action = action_
        self.valid = jnp.all(
            jnp.isfinite(jnp.stack((gauge, maxwell, continuity, action_)))
        )


@final
class AbelianMaxwellOperator(StrictModule):
    """Prepared realization and boundary semantics for abelian Maxwell fields."""

    complex: AbstractDeRhamComplex
    boundary: ComplexBoundary = eqx.field(static=True)

    @checked
    def __init__(
        self, complex: AbstractDeRhamComplex, /, *, boundary: ComplexBoundary = "absolute"
    ) -> None:
        boundary_ = parse(boundary, ComplexBoundary, "boundary")
        self.complex = complex
        self.boundary = boundary_

    def __call__(
        self, potential: DiscreteForm, current: DiscreteForm | None = None, /
    ) -> tuple[DiscreteForm, Array]:
        return (
            abelian_maxwell_residual(
                self.complex, potential, current, boundary=self.boundary
            ),
            abelian_maxwell_action(
                self.complex, potential, current, boundary=self.boundary
            ),
        )


def _require_form(
    complex: AbstractDeRhamComplex, field: DiscreteForm, degree: int, name: str, /
) -> None:
    if not isinstance(complex, AbstractDeRhamComplex):
        raise TypeError("complex must be an AbstractDeRhamComplex.")
    if not isinstance(field, DiscreteForm):
        raise TypeError(f"{name} must be a DiscreteForm.")
    if field.realization_id != complex.realization_id:
        raise ValueError(f"{name} belongs to a different realization.")
    if field.form_type != complex.form_type(degree):
        raise ValueError(
            f"{name} must have the realization's primal degree-{degree} form type."
        )
    if field.values.shape != (complex.cell_counts[degree],):
        raise ValueError(f"{name} has an incompatible cell extent.")


def _inner(
    complex: AbstractDeRhamComplex,
    left: DiscreteForm,
    right: DiscreteForm,
    /,
    *,
    boundary: ComplexBoundary,
) -> Array:
    from ..discretization._cochain import CochainDiscretization

    degree = left.form_type.degree
    space = complex.hilbert_complex(boundary=boundary).space(degree)
    if isinstance(complex, CochainDiscretization):
        indices = complex.active_indices(degree, boundary=boundary)
        return space.inner(left.values[indices], right.values[indices]).real
    return space.inner(left.values, right.values).real


def abelian_gauge_transform(
    complex: AbstractDeRhamComplex,
    potential: DiscreteForm,
    parameter: DiscreteForm,
    /,
    *,
    boundary: ComplexBoundary = "absolute",
) -> DiscreteForm:
    """Return the primal one-form ``A + d chi``."""
    _require_form(complex, potential, 1, "potential")
    _require_form(complex, parameter, 0, "parameter")
    values = potential.values + complex.exterior_derivative(
        0, parameter.values, boundary=boundary
    )
    return DiscreteForm(complex.realization_id, potential.form_type, values)


def abelian_curvature(
    complex: AbstractDeRhamComplex,
    potential: DiscreteForm,
    /,
    *,
    boundary: ComplexBoundary = "absolute",
) -> DiscreteForm:
    """Return the primal two-form curvature ``F = dA``."""
    _require_form(complex, potential, 1, "potential")
    values = complex.exterior_derivative(1, potential.values, boundary=boundary)
    return DiscreteForm(complex.realization_id, complex.form_type(2), values)


def abelian_maxwell_residual(
    complex: AbstractDeRhamComplex,
    potential: DiscreteForm,
    current: DiscreteForm | None = None,
    /,
    *,
    boundary: ComplexBoundary = "absolute",
) -> DiscreteForm:
    """Return ``delta dA + J`` in the realization's primal coordinates."""
    curvature = abelian_curvature(complex, potential, boundary=boundary)
    values = complex.codifferential(2, curvature.values, boundary=boundary)
    if current is not None:
        _require_form(complex, current, 1, "current")
        values = values + current.values
    return DiscreteForm(complex.realization_id, complex.form_type(1), values)


def abelian_current_continuity(
    complex: AbstractDeRhamComplex,
    current: DiscreteForm,
    /,
    *,
    boundary: ComplexBoundary = "absolute",
) -> Array:
    _require_form(complex, current, 1, "current")
    values = complex.codifferential(1, current.values, boundary=boundary)
    divergence = DiscreteForm(complex.realization_id, complex.form_type(0), values)
    return _inner(complex, divergence, divergence, boundary=boundary)


def abelian_maxwell_action(
    complex: AbstractDeRhamComplex,
    potential: DiscreteForm,
    current: DiscreteForm | None = None,
    /,
    *,
    boundary: ComplexBoundary = "absolute",
) -> Array:
    curvature = abelian_curvature(complex, potential, boundary=boundary)
    action = 0.5 * _inner(complex, curvature, curvature, boundary=boundary)
    if current is None:
        return action
    _require_form(complex, current, 1, "current")
    return action - _inner(complex, potential, current, boundary=boundary)


def validate_abelian_gauge_system(
    complex: AbstractDeRhamComplex,
    potential: DiscreteForm,
    parameter: DiscreteForm,
    /,
    *,
    current: DiscreteForm | None = None,
    boundary: ComplexBoundary = "absolute",
) -> AbelianGaugeDiagnostics:
    transformed = abelian_gauge_transform(
        complex, potential, parameter, boundary=boundary
    )
    curvature = abelian_curvature(complex, potential, boundary=boundary)
    transformed_curvature = abelian_curvature(complex, transformed, boundary=boundary)
    gauge_residual = jnp.max(
        jnp.abs(transformed_curvature.values - curvature.values), initial=0.0
    )
    maxwell = abelian_maxwell_residual(complex, potential, current, boundary=boundary)
    continuity = (
        jnp.asarray(0.0, dtype=maxwell.values.real.dtype)
        if current is None
        else abelian_current_continuity(complex, current, boundary=boundary)
    )
    return AbelianGaugeDiagnostics(
        gauge_curvature_residual=gauge_residual,
        maxwell_residual_norm=_inner(complex, maxwell, maxwell, boundary=boundary),
        current_continuity_residual=continuity,
        action=abelian_maxwell_action(complex, potential, current, boundary=boundary),
    )


__all__ = [
    "AbelianMaxwellOperator",
    "AbelianGaugeDiagnostics",
    "abelian_current_continuity",
    "abelian_curvature",
    "abelian_gauge_transform",
    "abelian_maxwell_action",
    "abelian_maxwell_residual",
    "validate_abelian_gauge_system",
]
