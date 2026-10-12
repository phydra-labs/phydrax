#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import ceil
from typing import final

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._interpolation import GatherStencil
from ..._strict import StrictModule
from ..splatting._assignment import (
    _basis_and_derivative,
    _linear_nonuniform_axis_stencil,
    _tensor_product_state,
    _uniform_axis_stencil,
    MultilinearSplatAssignment,
    SplatAssignmentState,
)
from ..splatting._bspline import TensorBSplineSplatAssignment
from ..splatting._cpdi import AffineCPDISplatAssignment, CPDI2SplatAssignment
from ..splatting._gimp import _gimp_axis_stencil, UniformGIMPSplatAssignment
from ..splatting._structured import ParticleGridSplatState, PreparedParticleGridSplat


type _AxisStencil = tuple[Array, Array, Array, Array, Array, Array]


@final
class AxisymmetricAssignmentResult(StrictModule):
    """Assignment routes with matching odd radial parity and unsigned hoop."""

    assignment: SplatAssignmentState
    radial_parity: Array
    hoop_gradients: Array

    def __init__(
        self,
        assignment: SplatAssignmentState,
        radial_parity: Array,
        hoop_gradients: Array,
    ) -> None:
        if (
            radial_parity.shape != assignment.weights.shape
            or hoop_gradients.shape != assignment.weights.shape
        ):
            raise ValueError("Axisymmetric parity and hoop must match assignment routes.")
        self.assignment = assignment
        self.radial_parity = radial_parity
        self.hoop_gradients = hoop_gradients


def validate_axisymmetric_assignment(splat: PreparedParticleGridSplat) -> None:
    """Validate the prepared meridional assignment, before numerical execution."""
    if len(splat.target_shape) != 2:
        raise ValueError("Axisymmetric routes require a two-dimensional (r, z) grid.")
    assignment = splat.plan.assignment
    if isinstance(assignment, UniformGIMPSplatAssignment):
        if assignment.evolving:
            raise ValueError(
                "Axisymmetric cpGIMP/AABB moving domains are unsupported: their "
                "deformed-domain ring integration and parity are not implemented. "
                "Use fixed UniformGIMPSplatAssignment (evolving=False)."
            )
        assignment.validate_input(None, splat.particles.capacity, 2)
    elif isinstance(assignment, TensorBSplineSplatAssignment):
        if any(degree == 0 for degree in assignment._axis_degrees(2)):
            raise ValueError(
                "Axisymmetric B-spline routes require degree one, two, or three "
                "on each axis; degree-zero routes have no meridional gradient."
            )
    elif isinstance(assignment, (AffineCPDISplatAssignment, CPDI2SplatAssignment)):
        raise ValueError(
            "Axisymmetric CPDI moving domains are unsupported: corner-domain "
            "ring integration and reflected radial parity are not implemented."
        )
    elif not isinstance(assignment, MultilinearSplatAssignment):
        raise ValueError(
            "Axisymmetric routes support multilinear, tensor B-spline degrees "
            "one to three, and fixed uniform GIMP assignments only."
        )
    radial_axis = splat.plan.target.structured_axes[0]
    if radial_axis.periodic:
        raise ValueError("The axisymmetric radial grid cannot be periodic.")
    radial_coordinates = np.asarray(splat.layout.coordinates_by_axis[0])
    lower, _ = splat.axis_bounds[0]
    if lower < 0.0 or np.any(radial_coordinates < 0.0):
        raise ValueError("Axisymmetric grids cannot contain negative physical radii.")
    if lower == 0.0 and radial_coordinates[0] != 0.0:
        raise ValueError(
            "Reflected axisymmetric support requires the radial coordinate origin "
            "to be zero when the radial grid lower bound is zero."
        )


def _reflect_radial_stencil(
    stencil: _AxisStencil,
    coordinates: Array,
    radius: Array,
    *,
    reflect: bool,
) -> tuple[_AxisStencil, Array]:
    indices, weights, derivatives, offsets, valid, source_in_domain = stencil
    if not reflect:
        return stencil, jnp.ones_like(weights)
    spacing = (coordinates[1] - coordinates[0]).astype(radius.dtype)
    # Offsets retain the unreflected integer-node identity, including duplicate
    # physical destinations. Scalar basis values never receive the odd parity.
    raw = jnp.rint((offsets + radius[:, None] - coordinates[0]) / spacing).astype(
        jnp.int32
    )
    physical = jnp.abs(raw)
    reflected = (
        jnp.clip(physical, 0, coordinates.size - 1),
        weights,
        derivatives,
        offsets,
        source_in_domain[:, None] & (physical < coordinates.size),
        source_in_domain,
    )
    parity = jnp.where(raw < 0, -jnp.ones_like(weights), jnp.ones_like(weights))
    return reflected, parity


def _ring_radial_integrals(
    coordinates: Array,
    radius: Array,
    half_width: Array,
    stencil: _AxisStencil,
) -> tuple[_AxisStencil, Array]:
    """Exactly integrate linear nodal bases over a nonnegative radial domain.

    Each node's two polynomial pieces are clipped to [max(0, r-l), r+l].
    Two-point Gauss integration is exact for r*N (quadratic), r*dN/dr
    (linear), and N (linear). No basis formula is duplicated here.
    """
    indices, point_weights, point_gradients, offsets, valid, domain = stencil
    spacing = (coordinates[1] - coordinates[0]).astype(radius.dtype)
    nodes = offsets + radius[:, None]
    lower = jnp.maximum(radius - half_width, 0.0)[:, None, None]
    upper = (radius + half_width)[:, None, None]
    knot_lower = nodes[..., None] + spacing * jnp.asarray((-1.0, 0.0), dtype=radius.dtype)
    knot_upper = nodes[..., None] + spacing * jnp.asarray((0.0, 1.0), dtype=radius.dtype)
    segment_lower = jnp.maximum(knot_lower, lower)
    segment_upper = jnp.maximum(segment_lower, jnp.minimum(knot_upper, upper))
    midpoint = 0.5 * (segment_lower + segment_upper)
    segment_half = 0.5 * (segment_upper - segment_lower)
    gauss = jnp.asarray((-1.0, 1.0), dtype=radius.dtype) / jnp.sqrt(
        jnp.asarray(3.0, dtype=radius.dtype)
    )
    samples = midpoint[..., None] + segment_half[..., None] * gauss
    values, derivatives = _basis_and_derivative(
        1, (samples - nodes[..., None, None]) / spacing
    )
    measure = segment_half[..., None]
    integral = jnp.sum(measure * samples * values, axis=(-2, -1))
    gradient_integral = jnp.sum(measure * samples * derivatives / spacing, axis=(-2, -1))
    hoop_integral = jnp.sum(measure * values, axis=(-2, -1))
    domain_lower = jnp.maximum(radius - half_width, 0.0)
    domain_upper = radius + half_width
    ring_measure = 0.5 * (domain_upper - domain_lower) * (domain_upper + domain_lower)
    point = half_width <= jnp.finfo(radius.dtype).eps * spacing
    safe_measure = jnp.where(point, jnp.ones_like(ring_measure), ring_measure)
    weights = jnp.where(point[:, None], point_weights, integral / safe_measure[:, None])
    gradients = jnp.where(
        point[:, None], point_gradients, gradient_integral / safe_measure[:, None]
    )
    safe_radius = jnp.where(radius > 0.0, radius, jnp.ones_like(radius))
    hoop = jnp.where(
        point[:, None],
        point_weights / safe_radius[:, None],
        hoop_integral / safe_measure[:, None],
    )
    return (indices, weights, gradients, offsets, valid, domain), hoop


def _point_axis_stencils(
    splat: PreparedParticleGridSplat,
    position: Array,
    active: Array,
) -> tuple[_AxisStencil, _AxisStencil]:
    assignment = splat.plan.assignment
    stencils: list[_AxisStencil] = []
    if isinstance(assignment, TensorBSplineSplatAssignment):
        degrees = assignment._axis_degrees(2)
        uniform = True
    elif isinstance(assignment, MultilinearSplatAssignment):
        degrees = (1, 1)
        uniform = any(entity == "interval" for entity in splat.layout.axis_entities)
    else:
        raise ValueError("Point axisymmetric routes require a point assignment family.")
    for axis, (coordinates, bounds, geometry_axis, degree) in enumerate(
        zip(
            splat.layout.coordinates_by_axis,
            splat.axis_bounds,
            splat.plan.target.structured_axes,
            degrees,
            strict=True,
        )
    ):
        if uniform:
            stencil = _uniform_axis_stencil(
                degree,
                coordinates,
                bounds,
                geometry_axis.periodic,
                position[:, axis],
                active,
            )
        else:
            stencil = _linear_nonuniform_axis_stencil(
                coordinates, bounds, geometry_axis.periodic, position[:, axis], active
            )
        stencils.append(stencil)
    return stencils[0], stencils[1]


def build_axisymmetric_assignment(
    splat: PreparedParticleGridSplat,
    position: Array,
    active: Array,
    *,
    half_widths: Array | None = None,
) -> AxisymmetricAssignmentResult:
    """Build assignment, radial parity, and unsigned hoop routes for any batch.

    Quadrature callers may repeat fixed uGIMP widths explicitly with
    ``half_widths``; no particle-capacity restriction is imposed here.
    Preparation owns family/origin validation. Runtime callers own finite
    geometry sanitization and require strictly positive active radii.
    """
    if position.ndim != 2 or position.shape[1] != 2:
        raise ValueError(
            "Axisymmetric assignment positions must have shape (sources, 2)."
        )
    if active.shape != (position.shape[0],):
        raise ValueError("Axisymmetric assignment activity must match the source count.")
    active = active & jnp.all(jnp.isfinite(position), axis=-1) & (position[:, 0] > 0.0)
    assignment = splat.plan.assignment
    if isinstance(assignment, UniformGIMPSplatAssignment):
        if assignment.evolving:
            raise ValueError(
                "Axisymmetric routes require fixed, not moving, GIMP domains."
            )
        widths = assignment.reference_half_widths if half_widths is None else half_widths
        if widths.shape != position.shape:
            raise ValueError(
                "Axisymmetric GIMP widths must match the source batch; quadrature "
                "callers must supply explicitly repeated half_widths."
            )
        widths = widths.astype(position.dtype)
        axis_width = 2 * ceil(1.0 + assignment.maximum_half_width_cells) + 1
        stencils = tuple(
            _gimp_axis_stencil(
                coordinates,
                bounds,
                axis.periodic,
                position[:, index],
                widths[:, index],
                active,
                axis_width,
                assignment.maximum_half_width_cells,
            )
            for index, (coordinates, bounds, axis) in enumerate(
                zip(
                    splat.layout.coordinates_by_axis,
                    splat.axis_bounds,
                    splat.plan.target.structured_axes,
                    strict=True,
                )
            )
        )
        radial, radial_hoop = _ring_radial_integrals(
            splat.layout.coordinates_by_axis[0], position[:, 0], widths[:, 0], stencils[0]
        )
        axial = stencils[1]
    else:
        if half_widths is not None:
            raise ValueError("Only fixed GIMP assignment accepts quadrature half_widths.")
        radial, axial = _point_axis_stencils(splat, position, active)
        safe_radius = jnp.where(position[:, 0] > 0.0, position[:, 0], 1.0)
        radial_hoop = radial[1] / safe_radius[:, None]
    radial, radial_parity = _reflect_radial_stencil(
        radial,
        splat.layout.coordinates_by_axis[0],
        position[:, 0],
        reflect=(
            splat.axis_bounds[0][0] == 0.0
            and not isinstance(assignment, MultilinearSplatAssignment)
        ),
    )
    state = _tensor_product_state((radial, axial), splat.target_shape, active)
    route_shape = state.weights.shape
    parity = jnp.broadcast_to(
        radial_parity[:, :, None],
        (position.shape[0], radial_parity.shape[1], axial[1].shape[1]),
    ).reshape(route_shape)
    hoop = (radial_hoop[:, :, None] * axial[1][:, None, :]).reshape(route_shape)
    return AxisymmetricAssignmentResult(state, parity, hoop)


def build_axisymmetric_routes(
    splat: PreparedParticleGridSplat,
    position: Array,
    assignment_input: object = None,
    active_mask: Array | None = None,
) -> ParticleGridSplatState:
    """Build reflected meridional routes and ring-consistent hoop operators."""
    expected = (splat.particles.capacity, 2)
    if position.shape != expected:
        raise ValueError(f"Axisymmetric particle positions must have shape {expected}.")
    if jnp.issubdtype(position.dtype, jnp.complexfloating):
        raise TypeError("Axisymmetric particle positions must be real.")
    splat.plan.assignment.validate_input(assignment_input, position.shape[0], 2)
    geometry = splat.plan.precision.geometry(position)
    if splat.plan.execution.geometry_ad == "frozen":
        geometry = jax.lax.stop_gradient(geometry)
    active = splat.particles.active_mask
    if active_mask is not None:
        if active_mask.shape != active.shape:
            raise ValueError("active_mask must have particle-capacity shape.")
        active = active & jnp.asarray(active_mask, dtype=jnp.bool_)
    finite = jnp.all(jnp.isfinite(geometry), axis=-1)
    geometry_valid = finite & (geometry[:, 0] > 0.0)
    invalid = active & ~geometry_valid
    fallback = jnp.asarray(
        tuple(bound[0] for bound in splat.axis_bounds), dtype=geometry.dtype
    )
    safe = jnp.where((active & geometry_valid)[:, None], geometry, fallback)
    routes = build_axisymmetric_assignment(splat, safe, active & geometry_valid)
    assignment = routes.assignment
    captured = assignment.captured_fractions.astype(splat.plan.precision.evaluation_dtype)
    tolerance = jnp.finfo(captured.dtype).eps * max(16, splat.route_width)
    supported = active & geometry_valid & (captured > tolerance)
    truncated = active & geometry_valid & (jnp.abs(captured - 1.0) > tolerance)
    outside = active & geometry_valid & ~assignment.source_in_domain
    valid = assignment.valid & supported[:, None]
    stencil = GatherStencil(
        indices=jax.lax.stop_gradient(assignment.indices),
        weights=assignment.weights.astype(splat.plan.precision.evaluation_dtype),
        source_size=splat.target_size,
        valid=jax.lax.stop_gradient(valid),
        support=jax.lax.stop_gradient(supported),
    )
    minimum_weight = jnp.where(
        jnp.any(valid), jnp.min(jnp.where(valid, stencil.weights, jnp.inf)), 0.0
    )
    margins = []
    family = splat.plan.assignment
    for axis, (geometry_axis, (lower, upper)) in enumerate(
        zip(splat.plan.target.structured_axes, splat.axis_bounds, strict=True)
    ):
        domain_lower = safe[:, axis]
        domain_upper = safe[:, axis]
        if isinstance(family, UniformGIMPSplatAssignment):
            half_width = family.reference_half_widths[:, axis].astype(geometry.dtype)
            domain_lower = domain_lower - half_width
            domain_upper = domain_upper + half_width
            if axis == 0:
                # A zero margin records radial-domain clipping at the physical
                # axis without classifying that valid ring domain as lost mass.
                domain_lower = jnp.maximum(domain_lower, 0.0)
        margin = (
            jnp.full((position.shape[0],), jnp.inf, dtype=geometry.dtype)
            if geometry_axis.periodic
            else jnp.minimum(domain_lower - lower, upper - domain_upper)
        )
        margins.append(margin)
    source_margin = jnp.min(jnp.stack(margins, axis=-1), axis=-1)
    minimum_domain_margin = jnp.where(
        jnp.any(supported), jnp.min(jnp.where(supported, source_margin, jnp.inf)), 0.0
    )
    boundary_ok = (splat.plan.boundary == "drop") | ~jnp.any(truncated)
    return ParticleGridSplatState(
        source_active_mask=active,
        stencil=stencil,
        assignment_state=assignment,
        supported_mask=supported,
        truncated_support_mask=truncated,
        out_of_domain_mask=outside,
        invalid_geometry_mask=invalid,
        partition_sums=captured,
        minimum_route_weight=minimum_weight,
        minimum_domain_margin=minimum_domain_margin,
        valid_route_count=jnp.sum(valid, dtype=jnp.int32),
        dropped_source_count=jnp.sum(truncated, dtype=jnp.int32),
        invalid_geometry_count=jnp.sum(invalid, dtype=jnp.int32),
        successful=~jnp.any(invalid) & boundary_ok,
        prepared_id=splat.prepared_id,
        radial_parity=routes.radial_parity,
        hoop_gradients=routes.hoop_gradients,
    )
