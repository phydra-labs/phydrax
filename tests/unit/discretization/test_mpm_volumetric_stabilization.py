#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Scientific transfer oracles; no constitutive-model implementation is mirrored."""

from __future__ import annotations

from typing import assert_never, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray

import phydrax as phx
import phydrax.ein as ein
from phydrax.discretization.mpm import (
    build_apic_route_payload,
    gather_apic,
    MPMVolumetricStabilizationKind,
    MPMVolumetricStabilizationPlan,
)
from phydrax.linalg import SmallLinearSolvePlan, solve_small_linear
from phydrax.typing import parse


type _DomainAssignmentKind = Literal["gimp", "cpdi", "cpdi2"]


def _prepare(
    position: Array,
    stabilization: MPMVolumetricStabilizationPlan | None,
    *,
    degree: int = 1,
    compact: bool = False,
    slots: Array | None = None,
    points: tuple[int, ...] | None = None,
    bounds: Array | None = None,
    assignment: phx.discretization.AbstractStructuredSplatAssignment | None = None,
) -> tuple[
    phx.equations.CompiledMaterialPointProblem, phx.equations.MaterialPointArguments
]:
    dimension = position.shape[-1]
    points = (9,) * dimension if points is None else points
    bounds = (
        jnp.asarray([[0.0] * dimension, [1.0] * dimension], dtype=jnp.float64)
        if bounds is None
        else bounds
    )
    assignment = (
        phx.discretization.TensorBSplineSplatAssignment(degree)
        if assignment is None
        else assignment
    )
    spacing = (np.asarray(bounds)[1] - np.asarray(bounds)[0]) / (
        np.asarray(points, dtype=np.float64) - 1.0
    )
    margin = assignment.capabilities.maximum_support_radius_cells * spacing
    domain_bounds = jnp.stack(
        (bounds[0] + jnp.asarray(margin), bounds[1] - jnp.asarray(margin))
    )
    grid_plan = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(n, periodic=False, endpoint=True)
            for n in points
        ),
        axis_names=tuple("xyz"[:dimension]),
    )
    grid = grid_plan.prepare(bounds)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(position.shape[0], dtype=jnp.int32),
        jnp.linspace(0.5, 2.0, position.shape[0], dtype=jnp.float64),
        ambient_dimension=dimension,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=assignment,
        boundary="reject",
    ).prepare(particles)
    index = grid_plan.prepare_index_space(bounds)
    storage = (
        phx.discretization.BlockSparseMPMNodalStoragePlan(
            phx.discretization.SparseBlockTopologyPlan(
                index, (3,) * dimension, 3**dimension, layout=index.vertices()
            )
        )
        if compact
        else None
    )
    fields = (
        phx.discretization.MPMNodalFieldPlan(("first", "second"), slots)
        if slots is not None
        else None
    )
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "volumetric-reference",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(dimension),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(volumetric_stabilization=stabilization),
        phx.discretization.MPMParticleDomainPlan(
            domain_bounds,
            periodic=(False,) * dimension,
            support_margin=tuple(float(value) for value in margin),
        ),
        nodal_storage=storage,
        nodal_fields=fields,
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    return compiled, arguments


def _domain_b_bar(
    family: _DomainAssignmentKind,
) -> tuple[phx.discretization.PreparedMPMDynamics, Array, object]:
    family = parse(family, _DomainAssignmentKind, "family")
    position = _positions()
    identity = jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (4, 2, 2))
    assignment: phx.discretization.AbstractStructuredSplatAssignment
    match family:
        case "gimp":
            assignment = phx.discretization.UniformGIMPSplatAssignment(
                jnp.full((4, 2), 0.02, dtype=jnp.float64),
                maximum_half_width_cells=0.4,
            )
        case "cpdi":
            assignment = phx.discretization.AffineCPDISplatAssignment(
                0.02 * identity, maximum_extent_cells=0.4
            )
        case "cpdi2":
            offsets = jnp.asarray(
                ((-0.02, -0.02), (-0.02, 0.02), (0.02, -0.02), (0.02, 0.02)),
                dtype=jnp.float64,
            )
            assignment = phx.discretization.CPDI2SplatAssignment(
                jnp.broadcast_to(offsets, (4, 4, 2)), maximum_extent_cells=0.4
            )
        case _ as unreachable:
            assert_never(unreachable)
    compiled, _ = _prepare(
        position, MPMVolumetricStabilizationPlan("b_bar"), assignment=assignment
    )
    domain = assignment.update_input(position, identity, None)
    return compiled.dynamics, position, domain


@pytest.mark.parametrize("family", ("gimp", "cpdi", "cpdi2"))
def test_b_bar_domain_coalescence_preserves_quadrature_operator(
    family: _DomainAssignmentKind,
) -> None:
    dynamics, position, domain = _domain_b_bar(family)
    routes = dynamics._build_routes(position, domain)
    original = dynamics.splat.build(position, assignment_input=domain)
    active = jnp.ones((4,), dtype=jnp.bool_)
    velocity = jnp.asarray(
        np.random.default_rng(705).normal(size=(dynamics.grid_count, 2)),
        dtype=jnp.float64,
    )
    raw = gather_apic(original, velocity, active, 1e12)
    gathered = gather_apic(routes, velocity, active, 1e12)
    np.testing.assert_allclose(gathered.velocity, raw.velocity, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(
        gathered.affine_velocity, raw.affine_velocity, atol=2e-12, rtol=2e-12
    )
    # Independent quadrature evaluation uses the unmodified assignment owner,
    # not the new cached/coalesced cell-average implementation.
    nodes, _ = np.polynomial.legendre.leggauss(2)
    gauss = np.concatenate(((nodes + 1.0) / 4.0, (nodes + 3.0) / 4.0))
    lower = np.floor(np.asarray(position) / 0.125) * 0.125
    identity = jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (4, 2, 2))
    divergence = np.zeros((4,), dtype=np.float64)
    for x in gauss:
        for y in gauss:
            sample = jnp.asarray(lower + 0.125 * np.asarray((x, y)), dtype=jnp.float64)
            sample_domain = dynamics.splat.plan.assignment.update_input(
                sample, identity, None
            )
            sample_routes = dynamics.splat.build(sample, assignment_input=sample_domain)
            sample_gradient = gather_apic(
                sample_routes, velocity, active, 1e12
            ).velocity_gradient
            divergence += np.trace(np.asarray(sample_gradient), axis1=-2, axis2=-1) / 16.0
    expected = np.asarray(raw.velocity_gradient).copy()
    correction = (divergence - np.trace(expected, axis1=-2, axis2=-1)) / 2.0
    expected += correction[:, None, None] * np.eye(2, dtype=np.float64)[None, :, :]
    np.testing.assert_allclose(
        gathered.velocity_gradient, expected, atol=2e-12, rtol=2e-12
    )
    for indices, valid in zip(
        np.asarray(routes.stencil.indices), np.asarray(routes.stencil.valid), strict=True
    ):
        assert np.unique(indices[valid]).size == np.count_nonzero(valid)
    assert bool(routes.successful) and bool(gathered.successful)


@pytest.mark.parametrize("family", ("gimp", "cpdi", "cpdi2"))
def test_b_bar_coalesced_domain_virtual_work_adjoint(
    family: _DomainAssignmentKind,
) -> None:
    dynamics, position, domain = _domain_b_bar(family)
    routes = dynamics._build_routes(position, domain)
    active = jnp.ones((4,), dtype=jnp.bool_)
    rng = np.random.default_rng(502)
    velocity = jnp.asarray(rng.normal(size=(dynamics.grid_count, 2)), dtype=jnp.float64)
    stress = jnp.asarray(rng.normal(size=(4, 2, 2)), dtype=jnp.float64)
    identity = jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (4, 2, 2))
    volume = jnp.asarray((0.2, 0.3, 0.4, 0.5), dtype=jnp.float64)
    zero = jnp.zeros_like(position)
    payload = build_apic_route_payload(
        routes,
        dynamics.particles.safe_masses,
        zero,
        jnp.zeros_like(identity),
        volume,
        stress,
        identity,
        zero,
        active,
    )
    force = (
        dynamics.splat.scatter_route_payload(routes, payload)
        .values[..., 2:4]
        .reshape(-1, 2)
    )
    gradient = gather_apic(routes, velocity, active, 1e12).velocity_gradient
    grid_work = jnp.sum(velocity * force)
    particle_work = jnp.sum(volume[:, None, None] * stress * gradient)
    np.testing.assert_allclose(grid_work, -particle_work, atol=2e-12, rtol=2e-12)


def test_b_bar_periodic_node_aliases_coalesce() -> None:
    bounds = jnp.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=jnp.float64)
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(3, periodic=True, endpoint=False),) * 2,
        axis_names=("x", "y"),
    ).prepare(bounds)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4, dtype=jnp.int32),
        jnp.ones((4,), dtype=jnp.float64),
        ambient_dimension=2,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
        boundary="reject",
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "periodic-cell-average",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(
            volumetric_stabilization=MPMVolumetricStabilizationPlan("b_bar")
        ),
        phx.discretization.MPMParticleDomainPlan(
            bounds, periodic=(True, True), support_margin=0.0
        ),
    )
    position = jnp.asarray(
        ((0.04, 0.08), (0.22, 0.31), (0.95, 0.98), (0.55, 0.61)), dtype=jnp.float64
    )
    original = splat.build(position)
    routes = compiled.dynamics._build_routes(position)
    velocity = jnp.asarray(
        np.random.default_rng(917).normal(size=(9, 2)), dtype=jnp.float64
    )
    active = jnp.ones((4,), dtype=jnp.bool_)
    raw = gather_apic(original, velocity, active, 1e12)
    gathered = gather_apic(routes, velocity, active, 1e12)
    np.testing.assert_allclose(gathered.velocity, raw.velocity, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(
        gathered.affine_velocity, raw.affine_velocity, atol=2e-12, rtol=2e-12
    )
    for indices, valid in zip(
        np.asarray(routes.stencil.indices), np.asarray(routes.stencil.valid), strict=True
    ):
        assert np.unique(indices[valid]).size == np.count_nonzero(valid)
    assert bool(routes.successful) and bool(gathered.successful)


def _positions(dimension: int = 2) -> Array:
    return jnp.asarray(
        [[0.34, 0.39, 0.43], [0.43, 0.44, 0.49], [0.47, 0.37, 0.46], [0.53, 0.48, 0.51]],
        dtype=jnp.float64,
    )[:, :dimension]


@pytest.mark.parametrize("kind", ["none", "b_bar", "f_bar"])
@pytest.mark.parametrize("dimension", [2, 3], ids=["2d", "3d"])
@pytest.mark.parametrize("degree", [1, 2, 3], ids=["linear", "quadratic", "cubic"])
def test_uniform_affine_patch(
    kind: MPMVolumetricStabilizationKind, dimension: int, degree: int
) -> None:
    position = _positions(dimension)
    compiled, _ = _prepare(position, MPMVolumetricStabilizationPlan(kind), degree=degree)
    dynamics = compiled.dynamics
    routes = dynamics._build_routes(position)
    active = jnp.ones((position.shape[0],), dtype=jnp.bool_)
    matrix = jnp.asarray(
        [[0.03, 0.02, -0.01], [-0.01, 0.04, 0.01], [0.02, 0.01, -0.02]], dtype=jnp.float64
    )[:dimension, :dimension]
    coordinates = dynamics.grid_coordinates
    grid_velocity = coordinates @ matrix.T + 0.07
    gathered = gather_apic(routes, grid_velocity, active, 1e12)
    assert bool(routes.successful)
    assert bool(gathered.successful)
    np.testing.assert_allclose(
        gathered.velocity_gradient,
        jnp.broadcast_to(matrix, (4, dimension, dimension)),
        atol=3e-13,
        rtol=3e-13,
    )
    deformation = jnp.broadcast_to(
        jnp.eye(dimension, dtype=jnp.float64), (4, dimension, dimension)
    )
    step = jnp.asarray(0.01, dtype=jnp.float64)
    updated = dynamics._update_deformation(
        routes,
        deformation,
        gathered.velocity_gradient,
        step,
        dynamics.particles.safe_masses,
        active,
        reference_volume=jnp.ones((4,), dtype=jnp.float64),
        physical_deformation_gradient=deformation,
    )
    np.testing.assert_allclose(
        updated,
        jnp.broadcast_to(
            jnp.eye(dimension, dtype=jnp.float64) + step * matrix, deformation.shape
        ),
        atol=3e-13,
        rtol=3e-13,
    )
    assert bool(jnp.all(jnp.isfinite(updated)))


def test_default_and_explicit_none_are_bitwise_identical() -> None:
    position = _positions()
    default, arguments = _prepare(position, None, degree=2)
    explicit, _ = _prepare(position, MPMVolumetricStabilizationPlan("none"), degree=2)
    velocity = position @ jnp.asarray([[0.02, -0.01], [0.01, 0.03]], dtype=jnp.float64)
    volume = jnp.full((4,), 0.01, dtype=jnp.float64)
    first = default.initialize_state(position, velocity, volume, arguments)
    second = explicit.initialize_state(position, velocity, volume, arguments)
    left = default.dynamics.step_detailed(first, 1e-4, arguments)
    right = explicit.dynamics.step_detailed(second, 1e-4, arguments)
    assert bool(left.successful) and bool(right.successful)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def test_none_preserves_original_transfer_and_deformation_operations() -> None:
    position = _positions()
    compiled, _ = _prepare(position, MPMVolumetricStabilizationPlan("none"), degree=2)
    dynamics = compiled.dynamics
    routes = dynamics._build_routes(position)
    ordinary = dynamics.splat.build(position)
    for left, right in zip(
        jax.tree.leaves(routes), jax.tree.leaves(ordinary), strict=True
    ):
        np.testing.assert_array_equal(left, right)
    rng = np.random.default_rng(445)
    grid_velocity = jnp.asarray(rng.normal(size=(9, 9, 2)), dtype=jnp.float64)
    active = jnp.asarray([True, False, True, True], dtype=jnp.bool_)
    valid = ordinary.stencil.valid
    route_velocity = jnp.where(
        valid[..., None], grid_velocity.reshape(-1, 2)[ordinary.stencil.indices], 0.0
    )
    weights = jnp.where(valid, ordinary.stencil.weights, 0.0)
    original_velocity = ein.contract("pr,pri->pi", weights, route_velocity)
    original_gradient = ein.contract(
        "pri,prj->pij", route_velocity, ordinary.weight_gradients
    )
    original_moment = ein.contract(
        "pr,pri,prj->pij", weights, route_velocity, ordinary.route_offsets
    )
    identity = jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (4, 2, 2))
    solve = solve_small_linear(
        SmallLinearSolvePlan(2),
        jnp.where(active[:, None, None], ordinary.second_moments, identity),
        original_moment.swapaxes(-1, -2),
    )
    original_affine = solve.value.swapaxes(-1, -2)
    gathered = gather_apic(routes, grid_velocity.reshape(-1, 2), active, 1e12)
    # "none" routes are the ordinary routes, so the gather itself is identical.
    for left, right in zip(
        jax.tree.leaves(gathered),
        jax.tree.leaves(
            gather_apic(ordinary, grid_velocity.reshape(-1, 2), active, 1e12)
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    # The fused gather uses bounded elementwise route sums, not batched GEMMs;
    # agreement with the dense contraction oracle is to rounding, not bitwise.
    np.testing.assert_allclose(
        gathered.velocity,
        jnp.where(active[:, None], original_velocity, 0.0),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        gathered.velocity_gradient,
        jnp.where(active[:, None, None], original_gradient, 0.0),
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        gathered.affine_moment, original_moment, rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        gathered.affine_velocity,
        jnp.where(active[:, None, None], original_affine, 0.0),
        rtol=1e-11,
        atol=1e-11,
    )
    step = jnp.asarray(0.01, dtype=jnp.float64)
    expected_deformation = ein.contract(
        "pij,pjk->pik", identity + step * gathered.velocity_gradient, identity
    )
    actual_deformation = dynamics._update_deformation(
        routes,
        identity,
        gathered.velocity_gradient,
        step,
        dynamics.particles.safe_masses,
        active,
    )
    np.testing.assert_array_equal(actual_deformation, expected_deformation)
    stress = jnp.asarray(rng.normal(size=(4, 2, 2)), dtype=jnp.float64)
    volume = jnp.asarray([0.3, 0.4, 0.5, 0.6], dtype=jnp.float64)
    acceleration = jnp.asarray(rng.normal(size=(4, 2)), dtype=jnp.float64)
    affine_route = ein.contract(
        "pij,prj->pri", gathered.affine_velocity, ordinary.route_offsets
    )
    momentum = (
        ordinary.stencil.weights[..., None]
        * dynamics.particles.safe_masses[:, None, None]
        * (gathered.velocity[:, None, :] + affine_route)
    )
    kirchhoff = ein.contract("pij,pkj->pik", stress, identity)
    internal = -volume[:, None, None] * ein.contract(
        "pij,prj->pri", kirchhoff, ordinary.weight_gradients
    )
    external = (
        ordinary.stencil.weights[..., None]
        * dynamics.particles.safe_masses[:, None, None]
        * acceleration[:, None, :]
    )
    expected_payload = jnp.where(
        active[:, None, None] & valid[..., None],
        jnp.concatenate((momentum, internal, external), axis=-1),
        0.0,
    )
    actual_payload = build_apic_route_payload(
        routes,
        dynamics.particles.safe_masses,
        gathered.velocity,
        gathered.affine_velocity,
        volume,
        stress,
        identity,
        acceleration,
        active,
    )
    np.testing.assert_allclose(actual_payload, expected_payload, rtol=1e-14, atol=1e-14)
    assert bool(gathered.successful) and bool(solve.successful.all())


def test_f_bar_physical_volume_force_scaling() -> None:
    position = _positions()
    compiled, _ = _prepare(position, MPMVolumetricStabilizationPlan("f_bar"))
    dynamics = compiled.dynamics
    routes = dynamics._build_routes(position)
    assumed = np.asarray(
        [
            [[1.1, 0.2], [0.0, 0.9]],
            [[0.8, 0.1], [0.0, 1.3]],
            [[1.2, 0.0], [0.1, 0.7]],
            [[0.9, 0.2], [0.0, 1.2]],
        ],
        dtype=np.float64,
    )
    physical = assumed * np.asarray([0.7, 1.2, 0.9, 1.4], dtype=np.float64)[:, None, None]
    piola = np.asarray(
        np.random.default_rng(927).normal(size=(4, 2, 2)), dtype=np.float64
    )
    reference_volume = np.asarray([0.3, 0.4, 0.5, 0.6], dtype=np.float64)
    zero = jnp.zeros_like(position)
    active = jnp.ones((4,), dtype=jnp.bool_)
    payload = build_apic_route_payload(
        routes,
        dynamics.particles.safe_masses,
        zero,
        jnp.zeros((4, 2, 2), dtype=jnp.float64),
        jnp.asarray(reference_volume),
        jnp.asarray(piola),
        jnp.asarray(assumed),
        zero,
        active,
        physical_deformation_gradient=jnp.asarray(physical),
    )
    kirchhoff = piola @ assumed.swapaxes(-1, -2)
    volume_factor = reference_volume * np.linalg.det(physical) / np.linalg.det(assumed)
    expected = (
        -volume_factor[:, None, None]
        * (kirchhoff[:, None, :, :] @ np.asarray(routes.weight_gradients)[..., None])[
            ..., 0
        ]
    )
    np.testing.assert_allclose(payload[..., 2:4], expected, atol=2e-13, rtol=2e-13)
    assert bool(jnp.all(jnp.isfinite(payload)))


@pytest.mark.parametrize("degree", [1, 2, 3], ids=["linear", "quadratic", "cubic"])
def test_b_bar_exact_virtual_work_adjoint(degree: int) -> None:
    position = _positions()
    compiled, _ = _prepare(
        position, MPMVolumetricStabilizationPlan("b_bar"), degree=degree
    )
    dynamics = compiled.dynamics
    routes = dynamics._build_routes(position)
    rng = np.random.default_rng(817)
    grid_velocity = jnp.asarray(
        rng.normal(size=dynamics.splat.target_shape + (2,)), dtype=jnp.float64
    )
    stress = jnp.asarray(rng.normal(size=(4, 2, 2)), dtype=jnp.float64)
    volume = jnp.asarray([0.3, 0.7, 0.9, 0.2], dtype=jnp.float64)
    active = jnp.asarray([True, True, False, True], dtype=jnp.bool_)
    zero = jnp.zeros_like(position)
    identity = jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), stress.shape)
    gathered = gather_apic(routes, grid_velocity.reshape(-1, 2), active, 1e12)
    payload = build_apic_route_payload(
        routes,
        dynamics.particles.safe_masses,
        zero,
        jnp.zeros_like(stress),
        volume,
        stress,
        identity,
        zero,
        active,
    )
    scatter = dynamics.splat.scatter_route_payload(routes, payload)
    assert bool(gathered.successful) and bool(scatter.successful)
    grid_work = jnp.sum(grid_velocity * scatter.values[..., 2:4])
    particle_work = jnp.sum(volume[:, None, None] * stress * gathered.velocity_gradient)
    np.testing.assert_allclose(grid_work, -particle_work, atol=2e-12, rtol=2e-12)


def _q1_weights(
    position: NDArray[np.float64], points: tuple[int, int], bounds: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Independent host shape functions on a Cartesian Q1 mesh (bounded oracle)."""
    nodes = np.stack(
        np.meshgrid(
            np.linspace(bounds[0, 0], bounds[1, 0], points[0]),
            np.linspace(bounds[0, 1], bounds[1, 1], points[1]),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    spacing = (bounds[1] - bounds[0]) / (np.asarray(points, dtype=np.float64) - 1.0)
    tent = np.maximum(
        1.0 - np.abs(position[:, None, :] - nodes[None, :, :]) / spacing[None, None, :],
        0.0,
    )
    return np.prod(tent, axis=-1)


def test_b_bar_q1_cell_integral_independent_reference() -> None:
    position = _positions()
    compiled, _ = _prepare(position, MPMVolumetricStabilizationPlan("b_bar"))
    routes = compiled.dynamics._build_routes(position)
    rng = np.random.default_rng(1181)
    grid_velocity = np.asarray(rng.normal(size=(9, 9, 2)), dtype=np.float64)
    bounds = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    host_position = np.asarray(position)
    cell_center = (np.floor(host_position * 8.0) + 0.5) / 8.0
    raw_gradient = np.zeros((4, 2, 2), dtype=np.float64)
    mean_gradient = np.zeros_like(raw_gradient)
    epsilon = 1e-6
    for axis in range(2):
        shift = np.zeros((1, 2), dtype=np.float64)
        shift[0, axis] = epsilon
        # Q1 is affine in each coordinate within this cell, so centered
        # differences are exact up to roundoff; midpoint integrates its gradient.
        raw_weights = (
            _q1_weights(host_position + shift, (9, 9), bounds)
            - _q1_weights(host_position - shift, (9, 9), bounds)
        ) / (2.0 * epsilon)
        mean_weights = (
            _q1_weights(cell_center + shift, (9, 9), bounds)
            - _q1_weights(cell_center - shift, (9, 9), bounds)
        ) / (2.0 * epsilon)
        raw_gradient[:, :, axis] = raw_weights @ grid_velocity.reshape(-1, 2)
        mean_gradient[:, :, axis] = mean_weights @ grid_velocity.reshape(-1, 2)
    trace_shift = (
        np.trace(mean_gradient, axis1=-2, axis2=-1)
        - np.trace(raw_gradient, axis1=-2, axis2=-1)
    ) / 2.0
    expected = (
        raw_gradient
        + trace_shift[:, None, None] * np.eye(2, dtype=np.float64)[None, :, :]
    )
    gathered = gather_apic(
        routes,
        jnp.asarray(grid_velocity.reshape(-1, 2)),
        jnp.ones((4,), dtype=jnp.bool_),
        1e12,
    )
    assert bool(gathered.successful)
    np.testing.assert_allclose(gathered.velocity_gradient, expected, atol=2e-8, rtol=2e-9)
    assert np.max(np.abs(expected - raw_gradient)) > 0.1


@pytest.mark.parametrize(
    "blend", [0.0, 0.35, 1.0], ids=["local-endpoint", "partial", "projected-endpoint"]
)
@pytest.mark.parametrize(
    "multifield", [False, True], ids=["single-field", "separate-fields"]
)
def test_f_bar_independent_volume_projection(blend: float, multifield: bool) -> None:
    position = _positions()
    slots = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32) if multifield else None
    compiled, _ = _prepare(
        position, MPMVolumetricStabilizationPlan("f_bar", f_bar_blend=blend), slots=slots
    )
    dynamics = compiled.dynamics
    routes = dynamics._build_routes(position)
    assumed = np.asarray(
        [
            [[1.1, 0.03], [0.0, 0.9]],
            [[0.9, 0.0], [0.02, 1.2]],
            [[1.05, 0.04], [0.0, 1.0]],
            [[1.2, 0.0], [0.03, 0.85]],
        ],
        dtype=np.float64,
    )
    physical = (
        assumed * np.asarray([0.8, 1.1, 1.3, 0.95], dtype=np.float64)[:, None, None]
    )
    gradient = np.asarray(
        [
            [[0.1, 0.2], [-0.1, 0.0]],
            [[-0.2, 0.0], [0.1, 0.1]],
            [[0.3, 0.1], [0.0, 0.2]],
            [[0.0, -0.1], [0.2, -0.1]],
        ],
        dtype=np.float64,
    )
    reference_volume = np.asarray([0.4, 0.9, 0.2, 0.7], dtype=np.float64)
    step = 0.07
    increment = np.eye(2, dtype=np.float64)[None, :, :] + step * gradient
    local = np.linalg.det(assumed) * np.linalg.det(increment)
    current_volume = reference_volume * np.linalg.det(physical)
    weights = _q1_weights(
        np.asarray(position),
        (9, 9),
        np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
    )
    fields = np.asarray(slots) if slots is not None else np.zeros((4,), dtype=np.int32)
    projected = np.zeros((4,), dtype=np.float64)
    for field in range(2 if multifield else 1):
        mask = fields == field
        node_volume = weights[mask].T @ current_volume[mask]
        node_numerator = weights[mask].T @ (current_volume[mask] * local[mask])
        node_average = np.divide(
            node_numerator,
            node_volume,
            out=np.zeros_like(node_volume),
            where=node_volume > 0.0,
        )
        projected[mask] = weights[mask] @ node_average
    target = (1.0 - blend) * local + blend * projected
    expected = (target / local)[:, None, None] ** 0.5 * (increment @ assumed)
    actual = dynamics._update_deformation(
        routes,
        jnp.asarray(assumed),
        jnp.asarray(gradient),
        jnp.asarray(step, dtype=jnp.float64),
        dynamics.particles.safe_masses,
        jnp.ones((4,), dtype=jnp.bool_),
        field_slots=slots,
        reference_volume=jnp.asarray(reference_volume),
        physical_deformation_gradient=jnp.asarray(physical),
    )
    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=3e-13)
    assert bool(jnp.all(jnp.isfinite(actual)))
    if blend > 0.0:
        assert np.max(np.abs(expected - increment @ assumed)) > 1e-4


@pytest.mark.parametrize("kind", ["none", "b_bar", "f_bar"])
@pytest.mark.parametrize(
    "multifield", [False, True], ids=["single-field", "separate-fields"]
)
def test_dense_sparse_full_step_equivalence(
    kind: MPMVolumetricStabilizationKind, multifield: bool
) -> None:
    position = _positions()
    slots = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32) if multifield else None
    stabilization = MPMVolumetricStabilizationPlan(kind)
    dense, arguments = _prepare(position, stabilization, degree=2, slots=slots)
    sparse, _ = _prepare(position, stabilization, degree=2, slots=slots, compact=True)
    velocity = position @ jnp.asarray([[0.03, -0.01], [0.01, 0.02]], dtype=jnp.float64)
    volume = jnp.asarray([0.01, 0.02, 0.03, 0.04], dtype=jnp.float64)
    first = dense.initialize_state(
        position, velocity, volume, arguments, velocity_field_slots=slots
    )
    second = sparse.initialize_state(
        position, velocity, volume, arguments, velocity_field_slots=slots
    )
    left = dense.dynamics.step_detailed(first, 1e-4, arguments)
    right = sparse.dynamics.step_detailed(second, 1e-4, arguments)
    assert bool(left.successful) and bool(right.successful)
    for a, b in zip(
        jax.tree.leaves(left.accepted_state.particles),
        jax.tree.leaves(right.accepted_state.particles),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12)
        assert bool(jnp.all(jnp.isfinite(a))) and bool(jnp.all(jnp.isfinite(b)))


@pytest.mark.parametrize(
    "kind,minimum,maximum",
    [("none", 0.08, 0.20), ("b_bar", 0.90, 1.05), ("f_bar", 0.55, 0.80)],
    ids=["unstabilized-locks", "cell-b-bar", "incremental-f-bar"],
)
def test_nearly_incompressible_plane_strain_bending_compliance(
    kind: MPMVolumetricStabilizationKind, minimum: float, maximum: float
) -> None:
    """Modal compliance, not a converged equilibrium or full cantilever solution.

    u=(-k*x*y, k*(x*x+y*y)/2) is divergence-free exact plane-strain
    bending. For nu=.499 its analytic restoring work is mu*k²*L*H³/3.
    Q1 interpolation creates spurious dilation despite exact nodal samples.
    Boundary projection leaves F-bar less accurate than cell B-bar; its honest
    20–45% modal-compliance error is still far below unstabilized locking.
    """
    nx, ny = 16, 4
    length, height, curvature = 4.0, 1.0, 0.01
    bounds = jnp.asarray(
        [[-length / nx, -0.5 - height / ny], [length + length / nx, 0.5 + height / ny]],
        dtype=jnp.float64,
    )
    gauss = (1.0 + np.asarray([-1.0, 1.0], dtype=np.float64) / np.sqrt(3.0)) / 2.0
    x = (
        (np.arange(nx, dtype=np.float64)[:, None] + gauss[None, :]) * length / nx
    ).reshape(-1)
    y = (
        (np.arange(ny, dtype=np.float64)[:, None] + gauss[None, :]) * height / ny
        - height / 2.0
    ).reshape(-1)
    position = jnp.asarray(
        np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2),
        dtype=jnp.float64,
    )
    compiled, _ = _prepare(
        position,
        MPMVolumetricStabilizationPlan(kind),
        points=(nx + 3, ny + 3),
        bounds=bounds,
    )
    dynamics = compiled.dynamics
    routes = dynamics._build_routes(position)
    node_x, node_y = dynamics.grid_coordinates[:, 0], dynamics.grid_coordinates[:, 1]
    displacement = curvature * jnp.stack(
        (-node_x * node_y, (node_x * node_x + node_y * node_y) / 2.0), axis=-1
    )
    count = position.shape[0]
    active = jnp.ones((count,), dtype=jnp.bool_)
    gathered = gather_apic(routes, displacement, active, 1e12)
    assert bool(gathered.successful)
    identity = jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (count, 2, 2))
    volume = jnp.full((count,), length * height / count, dtype=jnp.float64)
    step = 1e-7
    updated = dynamics._update_deformation(
        routes,
        identity,
        gathered.velocity_gradient,
        jnp.asarray(step, dtype=jnp.float64),
        volume,
        active,
        reference_volume=volume,
        physical_deformation_gradient=identity,
    )
    strain = np.asarray((updated - identity) / step)
    strain = (strain + strain.swapaxes(-1, -2)) / 2.0
    poisson = 0.499
    shear = 1.0 / (2.0 * (1.0 + poisson))
    lame = poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    stress = (
        2.0 * shear * strain
        + lame
        * np.trace(strain, axis1=-2, axis2=-1)[:, None, None]
        * np.eye(2, dtype=np.float64)[None, :, :]
    )
    zero = jnp.zeros_like(position)
    payload = build_apic_route_payload(
        routes,
        volume,
        zero,
        jnp.zeros_like(identity),
        volume,
        jnp.asarray(stress),
        identity,
        zero,
        active,
    )
    scatter = dynamics.splat.scatter_route_payload(routes, payload)
    assert bool(scatter.successful)
    restoring_work = -float(jnp.sum(displacement * scatter.values.reshape(-1, 6)[:, 2:4]))
    exact_work = shear * curvature**2 * length * height**3 / 3.0
    relative_compliance = exact_work / restoring_work
    assert np.isfinite(restoring_work) and restoring_work > 0.0
    assert minimum < relative_compliance < maximum, (kind, relative_compliance)


@pytest.mark.parametrize("kind", ["b_bar", "f_bar"])
def test_distributed_refuses_unimplemented_projection_halos(
    kind: MPMVolumetricStabilizationKind,
) -> None:
    with pytest.raises(ValueError, match="projection halo exchange"):
        phx.discretization.MPMDistributedPlan(
            (8, 8),
            (4, 4),
            jnp.asarray([[0, 0], [1, 1]], dtype=jnp.int32),
            device_count=2,
            particle_capacity_per_device=8,
            volumetric_stabilization=MPMVolumetricStabilizationPlan(kind),
        )


@pytest.mark.parametrize("kind", ["b_bar", "f_bar"])
def test_amr_refuses_unimplemented_cross_level_projection(
    kind: MPMVolumetricStabilizationKind,
) -> None:
    with pytest.raises(ValueError, match="cross-level projection restriction"):
        phx.discretization.MPMAMRPlan(
            ((4, 4), (8, 8)),
            (4, 16),
            volumetric_stabilization=MPMVolumetricStabilizationPlan(kind),
        )
