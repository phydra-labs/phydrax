#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Axisymmetric geometry, ring ledgers, and independent virtual-work oracles."""

from __future__ import annotations

from math import comb, factorial
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray

import phydrax as phx
from phydrax.discretization.mpm import build_apic_route_payload, gather_apic
from phydrax.discretization.splatting import ParticleGridSplatState


def _prepare(
    *,
    geometry: phx.discretization.MPMGeometryPlan | None = None,
    stabilization: phx.discretization.MPMVolumetricStabilizationPlan | None = None,
    material_dimension: int = 3,
    boundary: phx.discretization.PrescribedGridVelocityPlan | None = None,
    radial_lower: float = 0.0,
    domain_lower: float = 0.0,
    assignment: phx.discretization.AbstractStructuredSplatAssignment | None = None,
    compact: bool = False,
    field_slots: Array | None = None,
    schedule: phx.discretization.AbstractExplicitMPMSchedule | None = None,
) -> tuple[
    phx.equations.CompiledMaterialPointProblem, phx.equations.MaterialPointArguments
]:
    bounds = jnp.asarray(((radial_lower, -0.25), (2.0, 1.25)), dtype=jnp.float64)
    grid_plan = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(17, periodic=False, endpoint=True),
            phx.discretization.UniformAxisSpec(13, periodic=False, endpoint=True),
        ),
        axis_names=("r", "z"),
    )
    grid = grid_plan.prepare(bounds)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4, dtype=jnp.int32),
        jnp.asarray((0.2, 0.4, 0.6, 0.8), dtype=jnp.float64),
        ambient_dimension=2,
    ).prepare()
    assignment = (
        phx.discretization.TensorBSplineSplatAssignment(1)
        if assignment is None
        else assignment
    )
    margin = 0.125 * assignment.capabilities.maximum_support_radius_cells
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=assignment,
        boundary="reject",
    ).prepare(particles)
    index = grid_plan.prepare_index_space(bounds)
    storage = (
        phx.discretization.BlockSparseMPMNodalStoragePlan(
            phx.discretization.SparseBlockTopologyPlan(
                index,
                (1, 1),
                128,
                layout=index.vertices(),
            )
        )
        if compact
        else None
    )
    fields = (
        phx.discretization.MPMNodalFieldPlan(("inner", "outer"), field_slots)
        if field_slots is not None
        else None
    )
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "axisymmetric-contract",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(
                material_dimension
            ),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(
            geometry=geometry,
            volumetric_stabilization=stabilization,
            schedule=schedule,
        ),
        # The grid keeps two axial cells below and above z in [0, 1]; wider
        # quartic and quintic supports shrink the admitted axial material range.
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray(
                (
                    (domain_lower, max(0.0, margin - 0.25)),
                    (2.0 - margin, min(1.0, 1.25 - margin)),
                ),
                dtype=jnp.float64,
            ),
            periodic=(False, False),
            support_margin=(margin, margin),
        ),
        boundary=boundary,
        nodal_storage=storage,
        nodal_fields=fields,
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    return compiled, arguments


def _axisymmetric() -> phx.discretization.MPMGeometryPlan:
    return phx.discretization.MPMGeometryPlan("axisymmetric", spatial_dimension=2)


def _positions() -> Array:
    # The first particle deliberately uses the axis node in its linear stencil.
    return jnp.asarray(
        ((0.01, 0.31), (0.43, 0.38), (0.86, 0.52), (1.34, 0.61)), dtype=jnp.float64
    )


def _volumes() -> Array:
    return jnp.asarray((0.1, 0.2, 0.3, 0.4), dtype=jnp.float64)


def _cardinal_basis(
    coordinate: NDArray[np.float64],
    degree: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Independent centered cardinal polynomial, not a runtime basis helper.

    The even basis is summed on its left half, where every truncated power is
    small, and the odd derivative is mirrored; quintic right-half sums would
    otherwise cancel terms of size several hundred.
    """
    left = -np.abs(coordinate)
    value = np.zeros_like(coordinate)
    derivative = np.zeros_like(coordinate)
    for knot in range(degree + 2):
        positive = np.maximum(left + (degree + 1) / 2.0 - knot, 0.0)
        coefficient = (-1.0) ** knot * comb(degree + 1, knot)
        value += coefficient * positive**degree / factorial(degree)
        derivative += (
            coefficient
            * (
                (positive > 0.0).astype(np.float64)
                if degree == 1
                else positive ** (degree - 1)
            )
            / factorial(degree - 1)
        )
    derivative = np.where(coordinate > 0.0, -derivative, derivative)
    inside = np.abs(coordinate) < (degree + 1) / 2.0
    return np.where(inside, value, 0.0), np.where(inside, derivative, 0.0)


def _axis_integrals(
    nodes: NDArray[np.float64],
    lower: float,
    upper: float,
    degree: int,
    *,
    radial: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Integrate polynomial pieces exactly; radial returns ∫rN, ∫rN', ∫N."""
    spacing = 0.125
    knots = nodes[:, None] + spacing * (
        np.arange(degree + 2, dtype=np.float64) - (degree + 1) / 2.0
    )
    breaks = np.unique(
        np.concatenate(
            (np.asarray((lower, upper)), knots[(knots > lower) & (knots < upper)])
        )
    )
    abscissa, weight = np.polynomial.legendre.leggauss(5)
    basis_integral = np.zeros_like(nodes)
    derivative_integral = np.zeros_like(nodes)
    unweighted_integral = np.zeros_like(nodes)
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        points = (left + right) / 2.0 + (right - left) / 2.0 * abscissa
        quadrature = (right - left) / 2.0 * weight
        basis, derivative = _cardinal_basis(
            (points[:, None] - nodes[None, :]) / spacing, degree
        )
        measure = points if radial else np.ones_like(points)
        basis_integral += (quadrature * measure) @ basis
        derivative_integral += (quadrature * measure) @ derivative / spacing
        unweighted_integral += quadrature @ basis
    return basis_integral, derivative_integral, unweighted_integral


def _ring_average_basis(
    ghost: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    degree: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    radial, radial_derivative, hoop = _axis_integrals(
        ghost[:, 0], float(lower[0]), float(upper[0]), degree, radial=True
    )
    axial, axial_derivative, _ = _axis_integrals(
        ghost[:, 1], float(lower[1]), float(upper[1]), degree, radial=False
    )
    measure = (upper[0] ** 2 - lower[0] ** 2) / 2.0 * (upper[1] - lower[1])
    weights = radial * axial / measure
    gradients = (
        np.stack((radial_derivative * axial, radial * axial_derivative), axis=-1)
        / measure
    )
    return weights, gradients, hoop * axial / measure


def _independent_ring_operators(
    routes: ParticleGridSplatState,
    position: NDArray[np.float64],
    velocity: NDArray[np.float64],
    stress: NDArray[np.float64],
    *,
    degree: int,
    b_bar: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    gradient = np.zeros((4, 3, 3), dtype=np.float64)
    force = np.zeros_like(velocity)
    for particle in range(4):
        ghost = position[particle] + np.asarray(routes.route_offsets)[particle]
        indices = np.asarray(routes.stencil.indices)[particle]
        parity = np.where(ghost[:, 0] < -1e-12, -1.0, 1.0)
        basis, derivative = _cardinal_basis((position[particle] - ghost) / 0.125, degree)
        local_gradient = (
            np.stack(
                (derivative[:, 0] * basis[:, 1], basis[:, 0] * derivative[:, 1]), axis=-1
            )
            / 0.125
        )
        local_hoop = np.prod(basis, axis=-1) / position[particle, 0]
        correction = np.zeros_like(local_gradient)
        if b_bar:
            origin = np.asarray((0.0, -0.25), dtype=np.float64)
            lower = origin + np.floor((position[particle] - origin) / 0.125) * 0.125
            _, mean_gradient, mean_hoop = _ring_average_basis(
                ghost, lower, lower + 0.125, degree
            )
            correction = (mean_gradient - local_gradient) / 3.0
            correction[:, 0] += (mean_hoop - local_hoop) / 3.0
        valid = np.asarray(routes.stencil.valid)[particle]
        local_gradient = np.where(valid[:, None], local_gradient, 0.0)
        local_hoop = np.where(valid, local_hoop, 0.0)
        correction = np.where(valid[:, None], correction, 0.0)
        route_velocity = velocity[indices].copy()
        route_velocity[:, 0] *= parity
        gradient[particle, :2, :2] = route_velocity.T @ local_gradient
        gradient[particle, 2, 2] = route_velocity[:, 0] @ local_hoop
        gradient[particle] += np.eye(3, dtype=np.float64) * np.sum(
            route_velocity * correction
        )
        route_force = local_gradient @ stress[particle, :2, :2].T
        route_force[:, 0] += stress[particle, 2, 2] * local_hoop
        route_force += np.trace(stress[particle]) * correction
        route_force[:, 0] *= parity
        np.add.at(force, indices, -np.asarray(_volumes())[particle] * route_force)
    return gradient, force


_DEGREE_IDS = ("linear", "quadratic", "cubic", "quartic", "quintic")
_KIND_IDS = {"none": "local", "b_bar": "bbar", "f_bar": "fbar"}


# Point ring routes are degree-generic through quintic; the B-bar ring
# quadrature is qualified through cubic bases and refuses higher degrees.
@pytest.mark.parametrize(
    "degree,stabilization_kind",
    [
        pytest.param(degree, kind, id=f"{_DEGREE_IDS[degree - 1]}-{_KIND_IDS[kind]}")
        for degree in range(1, 6)
        for kind in ("none", "b_bar", "f_bar")
        if kind != "b_bar" or degree <= 3
    ],
)
def test_prescribed_uniform_radial_expansion_includes_hoop_stretch(
    stabilization_kind: phx.discretization.MPMVolumetricStabilizationKind,
    degree: int,
) -> None:
    dt, rate = 0.001, 0.2
    radial, axial = jnp.meshgrid(
        jnp.linspace(0.0, 2.0, 17, dtype=jnp.float64),
        jnp.linspace(-0.25, 1.25, 13, dtype=jnp.float64),
        indexing="ij",
    )
    values = jnp.stack((rate * radial, jnp.zeros_like(axial)), axis=-1)
    boundary = phx.discretization.PrescribedGridVelocityPlan(
        jnp.ones(values.shape, dtype=jnp.bool_),
        values,
    )
    compiled, arguments = _prepare(
        geometry=_axisymmetric(),
        boundary=boundary,
        assignment=phx.discretization.TensorBSplineSplatAssignment(degree),
        stabilization=phx.discretization.MPMVolumetricStabilizationPlan(
            stabilization_kind
        ),
    )
    position = _positions()
    state = compiled.initialize_state(
        position, jnp.zeros_like(position), _volumes(), arguments
    )
    detail = jax.jit(lambda value: compiled.dynamics.step_detailed(value, dt, arguments))(
        state
    )
    assert bool(detail.successful)
    assert compiled.dynamics.dimension == 2
    assert compiled.dynamics.constitutive_dimension == 3
    assert compiled.dynamics.geometry.kind == "axisymmetric"
    expected = jnp.broadcast_to(
        jnp.diag(jnp.asarray((1.0 + dt * rate, 1.0, 1.0 + dt * rate))), (4, 3, 3)
    )
    np.testing.assert_allclose(
        detail.accepted_state.particles.deformation_gradient,
        expected,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        detail.accepted_state.particles.position[:, 0],
        position[:, 0] * (1.0 + dt * rate),
        rtol=2e-12,
        atol=2e-12,
    )
    assert detail.accepted_state.particles.affine_velocity.shape == (4, 2, 2)
    np.testing.assert_allclose(
        detail.diagnostics.minimum_jacobian,
        (1.0 + dt * rate) ** 2,
        rtol=2e-12,
        atol=2e-12,
    )
    transfer = detail.diagnostics.transfer
    np.testing.assert_allclose(transfer.particle_mass, 2.0, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(transfer.grid_mass, 2.0, rtol=0.0, atol=2e-14)
    assert float(transfer.relative_mass_defect) < 2e-13
    assert float(transfer.relative_momentum_defect) < 2e-12
    axis = np.asarray(compiled.dynamics.grid_coordinates[:, 0]) == 0.0
    np.testing.assert_array_equal(
        np.asarray(detail.grid.velocity_after).reshape(-1, 2)[axis, 0],
        np.zeros(np.count_nonzero(axis), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        detail.accepted_state.particles.reference_volume, _volumes()
    )


@pytest.mark.parametrize("hoop_only", (False, True), ids=("full-stress", "hoop-only"))
@pytest.mark.parametrize(
    "degree,kind",
    [
        pytest.param(degree, kind, id=f"{_DEGREE_IDS[degree - 1]}-{_KIND_IDS[kind]}")
        for degree in range(1, 6)
        for kind in ("none", "b_bar")
        if kind != "b_bar" or degree <= 3
    ],
)
def test_ring_internal_force_is_exact_virtual_work_adjoint(
    hoop_only: bool,
    kind: phx.discretization.MPMVolumetricStabilizationKind,
    degree: int,
) -> None:
    compiled, _ = _prepare(
        geometry=_axisymmetric(),
        stabilization=phx.discretization.MPMVolumetricStabilizationPlan(kind),
        assignment=phx.discretization.TensorBSplineSplatAssignment(degree),
    )
    dynamics = compiled.dynamics
    position = _positions()
    routes = dynamics._build_routes(position)
    active = jnp.ones((4,), dtype=jnp.bool_)
    rng = np.random.default_rng(863)
    velocity = jnp.asarray(rng.normal(size=(dynamics.grid_count, 2)), dtype=jnp.float64)
    # Virtual displacements obey radial parity at the axis.
    velocity = velocity.at[dynamics.grid_coordinates[:, 0] == 0.0, 0].set(0.0)
    deformation = jnp.broadcast_to(
        jnp.asarray(
            ((1.08, 0.04, 0.0), (-0.03, 0.97, 0.0), (0.0, 0.0, 1.12)), dtype=jnp.float64
        ),
        (4, 3, 3),
    )
    stress = (
        jnp.zeros((4, 3, 3), dtype=jnp.float64)
        .at[:, 2, 2]
        .set(jnp.asarray((0.3, -0.2, 0.4, 0.6)))
        if hoop_only
        else jnp.asarray(rng.normal(size=(4, 3, 3)), dtype=jnp.float64)
    )
    payload = build_apic_route_payload(
        routes,
        dynamics.particles.safe_masses,
        jnp.zeros_like(position),
        jnp.zeros((4, 2, 2), dtype=jnp.float64),
        _volumes(),
        stress,
        deformation,
        jnp.zeros_like(position),
        active,
    )
    force = (
        dynamics.splat.scatter_route_payload(routes, payload)
        .values[..., 2:4]
        .reshape(-1, 2)
    )
    gathered = gather_apic(routes, velocity, active, 1e12)
    assert bool(routes.successful) and bool(gathered.successful)
    assert gathered.velocity_gradient.shape == (4, 3, 3)
    assert gathered.affine_velocity.shape == (4, 2, 2)
    kirchhoff = stress @ jnp.swapaxes(deformation, -1, -2)
    grid_work = jnp.sum(velocity * force)
    particle_work = jnp.sum(
        _volumes()[:, None, None] * kirchhoff * gathered.velocity_gradient
    )
    np.testing.assert_allclose(grid_work, -particle_work, rtol=3e-12, atol=3e-12)
    expected_gradient, expected_force = _independent_ring_operators(
        routes,
        np.asarray(position),
        np.asarray(velocity),
        np.asarray(kirchhoff),
        degree=degree,
        b_bar=kind == "b_bar",
    )
    np.testing.assert_allclose(
        gathered.velocity_gradient, expected_gradient, rtol=3e-11, atol=3e-11
    )
    np.testing.assert_allclose(force, expected_force, rtol=3e-11, atol=3e-11)
    if hoop_only:
        assert abs(float(grid_work)) > 1e-4
        # Suppressing hoop stress is a plausible adjoint-preserving fault when both
        # transfer operators omit it; the independent work oracle must detect it.
        assert abs(float(particle_work)) > 1e-4
        if kind == "none":
            np.testing.assert_array_equal(
                force[:, 1], jnp.zeros((dynamics.grid_count,), dtype=jnp.float64)
            )
        else:
            # A hoop-only stress projects onto meridional mean dilatation too.
            assert float(jnp.linalg.norm(force[:, 1])) > 1e-4


@pytest.mark.parametrize("radius", (0.0, -0.01))
def test_nonpositive_initial_radius_has_clear_refusal(radius: float) -> None:
    compiled, arguments = _prepare(geometry=_axisymmetric())
    position = _positions().at[0, 0].set(radius)
    with pytest.raises(eqx.EquinoxRuntimeError, match=r"Axisymmetric.*r > 0"):
        compiled.initialize_state(
            position, jnp.zeros_like(position), _volumes(), arguments
        )


def test_negative_radial_grid_is_refused_during_preparation() -> None:
    with pytest.raises(
        ValueError, match=r"(?i)(axisymmetric|radial).*(negative|nonnegative|r >= 0)"
    ):
        _prepare(geometry=_axisymmetric(), radial_lower=-0.25, domain_lower=0.125)


@pytest.mark.parametrize("degree", (2, 3, 4, 5), ids=_DEGREE_IDS[1:])
def test_cross_axis_ghost_routes_have_independent_exact_parity(degree: int) -> None:
    compiled, _ = _prepare(
        geometry=_axisymmetric(),
        assignment=phx.discretization.TensorBSplineSplatAssignment(degree),
    )
    position = _positions()
    routes = compiled.dynamics._build_routes(position)
    assert bool(routes.successful)
    parity = routes.radial_parity
    if parity is None:
        raise RuntimeError("Reflected routes require radial parity.")
    ghost = np.asarray(position)[:, None, :] + np.asarray(routes.route_offsets)
    radial_node = np.rint(ghost[..., 0] / 0.125).astype(np.int32)
    axial_node = np.rint((ghost[..., 1] + 0.25) / 0.125).astype(np.int32)
    expected_indices = np.abs(radial_node) * 13 + axial_node
    np.testing.assert_array_equal(routes.stencil.indices, expected_indices)
    np.testing.assert_array_equal(parity, np.where(radial_node < 0, -1.0, 1.0))
    basis, derivative = _cardinal_basis(
        (np.asarray(position)[:, None, :] - ghost) / 0.125, degree
    )
    np.testing.assert_allclose(
        routes.stencil.weights, np.prod(basis, axis=-1), rtol=3e-13, atol=3e-13
    )
    np.testing.assert_allclose(
        routes.weight_gradients,
        np.stack(
            (derivative[..., 0] * basis[..., 1], basis[..., 0] * derivative[..., 1]),
            axis=-1,
        )
        / 0.125,
        rtol=3e-13,
        atol=3e-13,
    )
    assert np.any(radial_node[0] < 0)
    assert np.unique(expected_indices[0]).size < expected_indices.shape[1]
    assert np.all(np.asarray(routes.stencil.weights) >= 0.0)
    np.testing.assert_allclose(jnp.sum(routes.stencil.weights, axis=-1), 1.0, atol=3e-14)
    particle_velocity = jnp.broadcast_to(
        jnp.asarray((0.04, -0.03), dtype=jnp.float64), position.shape
    )
    affine = jnp.broadcast_to(
        jnp.asarray(((0.12, -0.04), (0.03, 0.02)), dtype=jnp.float64), (4, 2, 2)
    )
    payload = build_apic_route_payload(
        routes,
        compiled.dynamics.particles.safe_masses,
        particle_velocity,
        affine,
        _volumes(),
        jnp.zeros((4, 3, 3), dtype=jnp.float64),
        jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (4, 3, 3)),
        jnp.zeros_like(position),
        jnp.ones((4,), dtype=jnp.bool_),
    )
    expected_momentum = np.zeros((compiled.dynamics.grid_count, 2), dtype=np.float64)
    for particle in range(4):
        route_velocity = (
            np.asarray(particle_velocity)[particle]
            + (ghost[particle] - np.asarray(position)[particle])
            @ np.asarray(affine)[particle].T
        )
        route_velocity[:, 0] *= np.where(radial_node[particle] < 0, -1.0, 1.0)
        route_momentum = (
            np.asarray(compiled.dynamics.particles.safe_masses)[particle]
            * np.prod(basis[particle], axis=-1)[:, None]
            * route_velocity
        )
        np.add.at(expected_momentum, expected_indices[particle], route_momentum)
    scattered = compiled.dynamics.splat.scatter_route_payload(routes, payload)
    assert bool(scattered.successful)
    np.testing.assert_allclose(
        scattered.values[..., :2].reshape(-1, 2),
        expected_momentum,
        rtol=3e-13,
        atol=3e-13,
    )


def test_axisymmetric_two_dimensional_material_is_refused() -> None:
    with pytest.raises(ValueError, match=r"(?i)(dimension|three.dimensional|3.d)"):
        _prepare(geometry=_axisymmetric(), material_dimension=2)


@pytest.mark.parametrize("clipped", (False, True), ids=("interior-ring", "axis-clipped"))
def test_fixed_ugimp_ring_average_matches_independent_quadrature(clipped: bool) -> None:
    half_width = jnp.asarray(
        ((0.02, 0.015), (0.018, 0.02), (0.015, 0.019), (0.023, 0.016)),
        dtype=jnp.float64,
    )
    assignment = phx.discretization.UniformGIMPSplatAssignment(
        half_width, maximum_half_width_cells=0.2
    )
    compiled, _ = _prepare(geometry=_axisymmetric(), assignment=assignment)
    position = _positions().at[0, 0].set(0.01 if clipped else 0.07)
    routes = compiled.dynamics._build_routes(position)
    assert bool(routes.successful)
    assert bool(jnp.all(routes.supported_mask))
    assert not bool(jnp.any(routes.truncated_support_mask))
    if clipped:
        np.testing.assert_allclose(routes.minimum_domain_margin, 0.0, atol=2e-14)
    hoop = routes.hoop_gradients
    if hoop is None:
        raise RuntimeError("Ring-averaged uGIMP requires hoop gradients.")
    for particle in range(4):
        ghost = (
            np.asarray(position)[particle] + np.asarray(routes.route_offsets)[particle]
        )
        lower = np.asarray(position)[particle] - np.asarray(half_width)[particle]
        lower[0] = max(0.0, lower[0])
        upper = np.asarray(position)[particle] + np.asarray(half_width)[particle]
        weights, gradients, expected_hoop = _ring_average_basis(ghost, lower, upper, 1)
        np.testing.assert_allclose(routes.stencil.weights[particle], weights, atol=3e-13)
        np.testing.assert_allclose(
            routes.weight_gradients[particle], gradients, atol=3e-12
        )
        np.testing.assert_allclose(hoop[particle], expected_hoop, atol=3e-12)
    np.testing.assert_allclose(jnp.sum(routes.stencil.weights, axis=-1), 1.0, atol=3e-14)
    # Averaging N/r is not evaluating Nbar/rp, even without axis clipping.
    wrong_hoop = routes.stencil.weights / position[:, 0, None]
    assert float(jnp.max(jnp.abs(hoop - wrong_hoop))) > 1e-3
    first_moment = jnp.sum(
        routes.stencil.weights[..., None] * routes.route_offsets, axis=1
    )
    assert float(jnp.max(jnp.abs(first_moment[:, 0]))) > 1e-4


@pytest.mark.parametrize(
    "kind", ("none", "b_bar", "f_bar"), ids=("local", "bbar", "fbar")
)
def test_fixed_ugimp_ring_weighted_uniform_expansion_and_apic_intercept(
    kind: phx.discretization.MPMVolumetricStabilizationKind,
) -> None:
    rate, dt = 0.2, 0.001
    compiled, arguments = _prepare(
        geometry=_axisymmetric(),
        assignment=phx.discretization.UniformGIMPSplatAssignment(
            jnp.full((4, 2), 0.02, dtype=jnp.float64), maximum_half_width_cells=0.2
        ),
        stabilization=phx.discretization.MPMVolumetricStabilizationPlan(kind),
    )
    position = _positions()
    coordinates = compiled.dynamics.grid_coordinates
    velocity = jnp.stack(
        (rate * coordinates[:, 0], jnp.zeros_like(coordinates[:, 1])), axis=-1
    )
    routes = compiled.dynamics._build_routes(position)
    gathered = gather_apic(routes, velocity, jnp.ones((4,), dtype=jnp.bool_), 1e12)
    assert bool(gathered.successful)
    expected_gradient = jnp.broadcast_to(
        jnp.diag(jnp.asarray((rate, 0.0, rate), dtype=jnp.float64)), (4, 3, 3)
    )
    np.testing.assert_allclose(gathered.velocity_gradient, expected_gradient, atol=2e-12)
    np.testing.assert_allclose(gathered.velocity[:, 0], rate * position[:, 0], atol=2e-12)
    np.testing.assert_allclose(
        gathered.affine_velocity,
        jnp.broadcast_to(jnp.diag(jnp.asarray((rate, 0.0))), (4, 2, 2)),
        atol=2e-12,
    )
    boundary = phx.discretization.PrescribedGridVelocityPlan(
        jnp.ones((17, 13, 2), dtype=jnp.bool_), velocity.reshape(17, 13, 2)
    )
    driven, _ = _prepare(
        geometry=_axisymmetric(),
        assignment=compiled.dynamics.splat.plan.assignment,
        stabilization=phx.discretization.MPMVolumetricStabilizationPlan(kind),
        boundary=boundary,
    )
    state = driven.initialize_state(
        position, jnp.zeros_like(position), _volumes(), arguments
    )
    detail = driven.dynamics.step_detailed(state, dt, arguments)
    assert bool(detail.successful)
    np.testing.assert_allclose(
        detail.accepted_state.particles.deformation_gradient,
        jnp.broadcast_to(jnp.eye(3) + dt * expected_gradient[0], (4, 3, 3)),
        atol=2e-12,
    )
    np.testing.assert_allclose(
        detail.accepted_state.particles.position[:, 0],
        position[:, 0] * (1.0 + dt * rate),
        atol=2e-12,
    )
    assert float(detail.diagnostics.transfer.relative_mass_defect) < 2e-13
    assert float(detail.diagnostics.transfer.relative_momentum_defect) < 2e-12


@pytest.mark.parametrize("family", ("cpgimp", "cpdi", "cpdi2"))
def test_axisymmetric_evolving_domains_have_precise_refusal(
    family: Literal["cpgimp", "cpdi", "cpdi2"],
) -> None:
    assignment: phx.discretization.AbstractStructuredSplatAssignment
    if family == "cpgimp":
        assignment = phx.discretization.UniformGIMPSplatAssignment(
            jnp.full((4, 2), 0.01, dtype=jnp.float64),
            maximum_half_width_cells=0.2,
            evolving=True,
        )
    elif family == "cpdi":
        assignment = phx.discretization.AffineCPDISplatAssignment(
            jnp.broadcast_to(0.01 * jnp.eye(2, dtype=jnp.float64), (4, 2, 2)),
            maximum_extent_cells=0.2,
        )
    else:
        offsets = jnp.asarray(
            ((-0.01, -0.01), (-0.01, 0.01), (0.01, -0.01), (0.01, 0.01)),
            dtype=jnp.float64,
        )
        assignment = phx.discretization.CPDI2SplatAssignment(
            jnp.broadcast_to(offsets, (4, 4, 2)), maximum_extent_cells=0.2
        )
    with pytest.raises(ValueError, match=r"(?i)axisymmetric.*(fixed|evolving|cpdi|gimp)"):
        _prepare(geometry=_axisymmetric(), assignment=assignment)


def test_axis_radial_velocity_constraint_and_full_ring_mass_ledger() -> None:
    compiled, arguments = _prepare(geometry=_axisymmetric())
    position = _positions()
    state = compiled.initialize_state(
        position,
        jnp.broadcast_to(jnp.asarray((0.04, 0.03), dtype=jnp.float64), position.shape),
        _volumes(),
        arguments,
    )
    detail = compiled.dynamics.step_detailed(state, 0.001, arguments)
    assert bool(detail.successful)
    axis = np.asarray(compiled.dynamics.grid_coordinates[:, 0]) == 0.0
    velocity = np.asarray(detail.grid.velocity_after).reshape(-1, 2)
    np.testing.assert_array_equal(
        velocity[axis, 0], np.zeros(np.count_nonzero(axis), dtype=np.float64)
    )
    transfer = detail.diagnostics.transfer
    np.testing.assert_allclose(transfer.particle_mass, 2.0, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(transfer.grid_mass, 2.0, rtol=0.0, atol=2e-14)
    assert float(transfer.relative_mass_defect) < 2e-13
    assert float(transfer.relative_momentum_defect) < 2e-12
    assert not bool(transfer.angular_momentum_valid)
    np.testing.assert_array_equal(
        detail.accepted_state.particles.reference_volume, _volumes()
    )
    # Ring volumes are supplied already integrated; neither mass nor energy receives 2*pi*r.
    np.testing.assert_allclose(
        jnp.sum(detail.grid.mass),
        jnp.sum(compiled.dynamics.particles.safe_masses),
        rtol=2e-13,
        atol=2e-13,
    )


def test_cartesian_default_and_explicit_geometry_are_bit_identical() -> None:
    default, arguments = _prepare(material_dimension=2, domain_lower=0.125)
    explicit, _ = _prepare(
        geometry=phx.discretization.MPMGeometryPlan("cartesian", spatial_dimension=2),
        material_dimension=2,
        domain_lower=0.125,
    )
    position = _positions().at[0, 0].set(0.28)
    velocity = jnp.broadcast_to(
        jnp.asarray((0.03, -0.02), dtype=jnp.float64), position.shape
    )
    initial_default = default.initialize_state(position, velocity, _volumes(), arguments)
    initial_explicit = explicit.initialize_state(
        position, velocity, _volumes(), arguments
    )
    result_default = default.dynamics.step_detailed(initial_default, 0.001, arguments)
    result_explicit = explicit.dynamics.step_detailed(initial_explicit, 0.001, arguments)
    assert bool(result_default.successful) and bool(result_explicit.successful)
    leaves_default = jax.tree.leaves(result_default)
    leaves_explicit = jax.tree.leaves(result_explicit)
    assert len(leaves_default) == len(leaves_explicit)
    for left, right in zip(leaves_default, leaves_explicit, strict=True):
        np.testing.assert_array_equal(left, right)


def test_axisymmetric_two_field_step_preserves_ring_ledger() -> None:
    compiled, arguments = _prepare(
        geometry=_axisymmetric(),
        assignment=phx.discretization.TensorBSplineSplatAssignment(3),
        stabilization=phx.discretization.MPMVolumetricStabilizationPlan("b_bar"),
        field_slots=jnp.asarray((0, 1, 0, 1), dtype=jnp.int32),
    )
    position = _positions()
    velocity = jnp.stack(
        (0.02 * position[:, 0], jnp.zeros((4,), dtype=jnp.float64)), axis=-1
    )
    initial = compiled.initialize_state(position, velocity, _volumes(), arguments)
    detail = compiled.dynamics.step_detailed(initial, 0.001, arguments)
    assert bool(detail.successful)
    assert detail.grid.mass.shape[0] == 2
    assert detail.accepted_state.particles.deformation_gradient.shape == (4, 3, 3)
    assert detail.accepted_state.particles.affine_velocity.shape == (4, 2, 2)
    np.testing.assert_allclose(jnp.sum(detail.grid.mass), 2.0, rtol=2e-13, atol=2e-13)
    assert float(detail.diagnostics.transfer.relative_mass_defect) < 2e-13
    assert float(detail.diagnostics.transfer.relative_momentum_defect) < 2e-12
    assert bool(
        jnp.all(detail.accepted_state.particles.deformation_gradient[:, 2, 2] > 1.0)
    )
    assert not bool(detail.diagnostics.transfer.angular_momentum_valid)


@pytest.mark.parametrize("degree", (4, 5), ids=_DEGREE_IDS[3:])
def test_axisymmetric_b_bar_refuses_bspline_degrees_beyond_ring_rules(
    degree: int,
) -> None:
    assignment = phx.discretization.TensorBSplineSplatAssignment(degree)
    # The identical setup is admitted with point routes, so the refusal belongs
    # to the B-bar ring quadrature and not to support halo or geometry checks.
    _prepare(geometry=_axisymmetric(), assignment=assignment)
    with pytest.raises(ValueError, match="B-bar"):
        _prepare(
            geometry=_axisymmetric(),
            assignment=assignment,
            stabilization=phx.discretization.MPMVolumetricStabilizationPlan("b_bar"),
        )


def test_axisymmetric_sparse_and_dense_particle_steps_agree() -> None:
    assignment = phx.discretization.TensorBSplineSplatAssignment(3)
    stabilization = phx.discretization.MPMVolumetricStabilizationPlan("b_bar")
    dense, arguments = _prepare(
        geometry=_axisymmetric(), assignment=assignment, stabilization=stabilization
    )
    sparse, _ = _prepare(
        geometry=_axisymmetric(),
        compact=True,
        assignment=assignment,
        stabilization=stabilization,
    )
    position = _positions()
    velocity = jnp.stack(
        (0.02 * position[:, 0], jnp.zeros((4,), dtype=jnp.float64)), axis=-1
    )
    dense_state = dense.initialize_state(position, velocity, _volumes(), arguments)
    sparse_state = sparse.initialize_state(position, velocity, _volumes(), arguments)
    dense_step = dense.dynamics.step_detailed(dense_state, 0.001, arguments)
    sparse_step = sparse.dynamics.step_detailed(sparse_state, 0.001, arguments)
    assert bool(dense_step.successful) and bool(sparse_step.successful)
    for left, right in zip(
        jax.tree.leaves(dense_step.accepted_state.particles),
        jax.tree.leaves(sparse_step.accepted_state.particles),
        strict=True,
    ):
        np.testing.assert_allclose(left, right, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        sparse_step.diagnostics.transfer.grid_mass,
        dense_step.diagnostics.transfer.grid_mass,
        rtol=2e-13,
        atol=2e-13,
    )
    assert sparse_step.accepted_state.storage_state is not None


def test_axisymmetric_hoop_route_derivative_includes_radius_and_is_adjoint() -> None:
    compiled, _ = _prepare(geometry=_axisymmetric())
    dynamics = compiled.dynamics
    position = _positions()
    deformation = jnp.broadcast_to(jnp.eye(3, dtype=position.dtype), (4, 3, 3))
    plan = phx.solver.MPMRouteSupersetPlan(
        dynamics.splat, dynamics=dynamics, minimum_margin=1e-12
    )
    base = plan.build(position)
    hoop = base.base_routes.hoop_gradients
    if hoop is None:
        raise RuntimeError("Axisymmetric routes require hoop gradients.")
    direction = jnp.asarray(
        ((0.002, -0.001), (0.001, 0.002), (-0.003, 0.001), (0.001, -0.002)),
        dtype=position.dtype,
    )
    cotangents = (
        jnp.zeros_like(base.base_routes.stencil.weights),
        jnp.zeros_like(base.base_routes.weight_gradients),
        jnp.zeros_like(base.base_routes.route_offsets),
        jnp.asarray(np.random.default_rng(94).normal(size=hoop.shape), dtype=hoop.dtype),
    )
    derivative = plan.linearize(
        base,
        position,
        deformation,
        None,
        direction,
        jnp.zeros_like(deformation),
        None,
        cotangents,
    )
    assert bool(derivative.successful)
    assert derivative.primal_hoop_gradients is not None
    assert derivative.hoop_gradient_jvp is not None
    np.testing.assert_array_equal(derivative.primal_hoop_gradients, hoop)
    np.testing.assert_allclose(
        jnp.vdot(derivative.hoop_gradient_jvp, cotangents[3]),
        jnp.vdot(derivative.position_transpose, direction),
        rtol=2e-12,
        atol=2e-12,
    )
    epsilon = 1e-5
    plus = plan.build(position + epsilon * direction).base_routes.hoop_gradients
    minus = plan.build(position - epsilon * direction).base_routes.hoop_gradients
    if plus is None or minus is None:
        raise RuntimeError("Perturbed axisymmetric routes require hoop gradients.")
    np.testing.assert_allclose(
        derivative.hoop_gradient_jvp,
        (plus - minus) / (2.0 * epsilon),
        rtol=2e-8,
        atol=2e-9,
    )


@pytest.mark.parametrize(
    "coupling", (True, False), ids=("hoop-coupling", "negative-hoop-stretch")
)
def test_axisymmetric_initial_deformation_layout_is_admitted_explicitly(
    coupling: bool,
) -> None:
    compiled, arguments = _prepare(geometry=_axisymmetric())
    position = _positions()
    deformation = jnp.broadcast_to(jnp.eye(3, dtype=position.dtype), (4, 3, 3))
    deformation = (
        deformation.at[0, 0, 2].set(0.01)
        if coupling
        else deformation.at[0, 2, 2].set(-1.0)
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="Axisymmetric.*F"):
        compiled.initialize_state(
            position,
            jnp.zeros_like(position),
            _volumes(),
            arguments,
            deformation_gradient=deformation,
        )


@pytest.mark.parametrize(
    "schedule",
    (
        phx.discretization.USFMPMSchedule(),
        phx.discretization.MUSLMPMSchedule(),
        phx.discretization.AffineMUSLMPMSchedule(),
        phx.discretization.PostAdvectionMUSLMPMSchedule(),
    ),
    ids=("usf", "musl", "affine-musl", "post-advection-musl"),
)
def test_reflected_cubic_b_bar_expansion_composes_with_every_stress_schedule(
    schedule: phx.discretization.AbstractExplicitMPMSchedule,
) -> None:
    rate, dt = 0.2, 0.001
    radial, axial = jnp.meshgrid(
        jnp.linspace(0.0, 2.0, 17, dtype=jnp.float64),
        jnp.linspace(-0.25, 1.25, 13, dtype=jnp.float64),
        indexing="ij",
    )
    values = jnp.stack((rate * radial, jnp.zeros_like(axial)), axis=-1)
    compiled, arguments = _prepare(
        geometry=_axisymmetric(),
        assignment=phx.discretization.TensorBSplineSplatAssignment(3),
        stabilization=phx.discretization.MPMVolumetricStabilizationPlan("b_bar"),
        boundary=phx.discretization.PrescribedGridVelocityPlan(
            jnp.ones(values.shape, dtype=jnp.bool_), values
        ),
        schedule=schedule,
    )
    position = _positions()
    velocity = jnp.stack(
        (rate * position[:, 0], jnp.zeros((4,), dtype=position.dtype)), axis=-1
    )
    affine = jnp.broadcast_to(
        jnp.diag(jnp.asarray((rate, 0.0), dtype=position.dtype)), (4, 2, 2)
    )
    state = compiled.initialize_state(
        position, velocity, _volumes(), arguments, affine_velocity=affine
    )
    result = compiled.dynamics.step_detailed(state, dt, arguments)
    assert bool(result.successful)
    assert bool(result.diagnostics.schedule.successful)
    assert float(result.diagnostics.schedule.second_transfer_mass_defect) < 2e-12
    assert float(result.diagnostics.schedule.second_transfer_momentum_defect) < 2e-12
    expected = jnp.broadcast_to(
        jnp.diag(
            jnp.asarray((1.0 + dt * rate, 1.0, 1.0 + dt * rate), dtype=position.dtype)
        ),
        (4, 3, 3),
    )
    np.testing.assert_allclose(
        result.accepted_state.particles.deformation_gradient,
        expected,
        rtol=2e-12,
        atol=2e-12,
    )


def test_frozen_axisymmetric_b_bar_routes_have_zero_position_jvp() -> None:
    compiled, _ = _prepare(
        geometry=_axisymmetric(),
        stabilization=phx.discretization.MPMVolumetricStabilizationPlan("b_bar"),
    )
    dynamics = compiled.dynamics
    execution = phx.discretization.SplatExecutionPolicy(geometry_ad="frozen")
    dynamics = eqx.tree_at(lambda value: value.splat.plan.execution, dynamics, execution)

    def offsets(position: Array) -> Array:
        return dynamics._build_routes(position).route_offsets

    _, tangent = jax.jvp(offsets, (_positions(),), (jnp.ones_like(_positions()),))
    np.testing.assert_array_equal(tangent, 0.0)
