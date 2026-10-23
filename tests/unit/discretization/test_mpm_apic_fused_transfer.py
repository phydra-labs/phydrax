#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native fused APIC transfer against an independent dense-contraction oracle.

The oracle is the pre-fusion algorithm written with dense ``ein.contract``; agreement is
to rounding (the fused path sums bounded elementwise products, not GEMMs), and
first and second derivatives are compared through ordinary JAX AD of both.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
import phydrax.ein as ein
from phydrax.discretization.mpm import (
    APICGatherResult,
    build_apic_route_payload,
    gather_apic,
    MPMVolumetricStabilizationKind,
    MPMVolumetricStabilizationPlan,
)
from phydrax.discretization.mpm._transfer import (
    apic_particle_angular_momentum,
    apic_particle_kinetic_energy,
)
from phydrax.discretization.splatting import ParticleGridSplatState
from phydrax.linalg import SmallLinearSolvePlan, solve_small_linear
from tests.unit.discretization.test_mpm_axisymmetric import (
    _axisymmetric,
    _positions as _ring_positions,
    _prepare as _prepare_ring,
)
from tests.unit.discretization.test_mpm_volumetric_stabilization import (
    _positions as _cartesian_positions,
    _prepare as _prepare_cartesian,
)


_CASES: dict[str, tuple[str, int, int, MPMVolumetricStabilizationKind]] = {
    "cartesian-2d-linear": ("cartesian", 2, 1, "none"),
    "cartesian-2d-quadratic": ("cartesian", 2, 2, "none"),
    "cartesian-3d-quadratic": ("cartesian", 3, 2, "none"),
    "cartesian-2d-bbar": ("cartesian", 2, 2, "b_bar"),
    "cartesian-3d-bbar": ("cartesian", 3, 1, "b_bar"),
    "axisymmetric-local": ("axisymmetric", 2, 2, "none"),
    "axisymmetric-bbar": ("axisymmetric", 2, 2, "b_bar"),
}
_ACTIVE = jnp.asarray((True, False, True, True), dtype=jnp.bool_)


@dataclass(frozen=True)
class _Case:
    state: ParticleGridSplatState
    grid_velocity: Array
    mass: Array
    position: Array


@functools.cache
def _case(name: str) -> _Case:
    geometry, dimension, degree, kind = _CASES[name]
    assignment = phx.discretization.TensorBSplineSplatAssignment(degree)
    stabilization = MPMVolumetricStabilizationPlan(kind)
    if geometry == "axisymmetric":
        compiled, _ = _prepare_ring(
            geometry=_axisymmetric(), stabilization=stabilization, assignment=assignment
        )
        position = _ring_positions()
    else:
        position = _cartesian_positions(dimension)
        compiled, _ = _prepare_cartesian(position, stabilization, degree=degree)
    dynamics = compiled.dynamics
    state = dynamics._build_routes(position)
    assert bool(state.successful)
    # Guard that every transfer branch named by the case is actually exercised.
    assert (state.hoop_gradients is not None) == (geometry == "axisymmetric")
    assert (state.radial_parity is not None) == (geometry == "axisymmetric")
    assert (state.cell_average_gradients is not None) == (kind == "b_bar")
    if geometry == "axisymmetric" and kind == "b_bar":
        assert state.cell_average_hoop_gradients is not None
    rng = np.random.default_rng(sum(map(ord, name)))
    grid_velocity = jnp.asarray(
        rng.normal(size=(dynamics.grid_count, dimension)), dtype=jnp.float64
    )
    return _Case(state, grid_velocity, dynamics.particles.safe_masses, position)


def _masked_routes(state: ParticleGridSplatState) -> ParticleGridSplatState:
    """Invalidate a deterministic subset of routes, including a whole particle row."""
    valid = np.asarray(state.stencil.valid).copy()
    valid[0, : valid.shape[1] // 2] = False
    valid[2, 1::3] = False
    return eqx.tree_at(lambda s: s.stencil.relation.valid, state, jnp.asarray(valid))


def _dense(spec: str, *operands: Array) -> Array:
    return ein.contract(spec, *operands)


def _oracle_gather(
    state: ParticleGridSplatState,
    grid_velocity: Array,
    active: Array,
    maximum_condition: float,
    /,
) -> APICGatherResult:
    dimension = grid_velocity.shape[-1]
    valid = state.stencil.valid
    safe = jnp.where(valid, state.stencil.indices, 0)
    route_velocity = jnp.where(valid[..., None], grid_velocity[safe], 0.0)
    weights = jnp.where(valid, state.stencil.weights, 0.0)
    if state.radial_parity is not None:
        route_velocity = route_velocity.at[..., 0].multiply(state.radial_parity)
    velocity = _dense("pr,pri->pi", weights, route_velocity)
    velocity_gradient = _dense("pri,prj->pij", route_velocity, state.weight_gradients)
    if state.hoop_gradients is None and state.cell_average_gradients is not None:
        divergence = _dense("pri,pri->p", route_velocity, state.cell_average_gradients)
        correction = (
            divergence - jnp.trace(velocity_gradient, axis1=-2, axis2=-1)
        ) / dimension
        velocity_gradient = velocity_gradient + correction[:, None, None] * jnp.eye(
            dimension, dtype=velocity_gradient.dtype
        )
    if state.hoop_gradients is not None:
        hoop_rate = _dense("pr,pr->p", route_velocity[..., 0], state.hoop_gradients)
        velocity_gradient = (
            jnp.zeros((velocity_gradient.shape[0], 3, 3), velocity_gradient.dtype)
            .at[:, :2, :2]
            .set(velocity_gradient)
            .at[:, 2, 2]
            .set(hoop_rate)
        )
        if state.cell_average_gradients is not None:
            assert state.cell_average_hoop_gradients is not None
            divergence = _dense(
                "pri,pri->p", route_velocity, state.cell_average_gradients
            ) + _dense(
                "pr,pr->p", route_velocity[..., 0], state.cell_average_hoop_gradients
            )
            correction = (
                divergence - jnp.trace(velocity_gradient, axis1=-2, axis2=-1)
            ) / 3
            velocity_gradient = velocity_gradient + correction[:, None, None] * jnp.eye(
                3, dtype=velocity_gradient.dtype
            )
    affine_moment = _dense(
        "pr,pri,prj->pij", weights, route_velocity, state.route_offsets
    )
    particle_moment = state.second_moments
    first_moment = state.first_moments
    if state.radial_parity is not None:
        particle_moment = particle_moment - _dense(
            "pi,pj->pij", first_moment, first_moment
        )
        affine_moment = affine_moment - _dense("pi,pj->pij", velocity, first_moment)
    identity = jnp.broadcast_to(
        jnp.eye(dimension, dtype=particle_moment.dtype), particle_moment.shape
    )
    solve = solve_small_linear(
        SmallLinearSolvePlan(dimension),
        jnp.where(active[:, None, None], particle_moment, identity),
        affine_moment.swapaxes(-1, -2),
    )
    affine_velocity = solve.value.swapaxes(-1, -2)
    if state.radial_parity is not None:
        velocity = velocity - _dense("pij,pj->pi", affine_velocity, first_moment)
    successful = jnp.all(
        (~active)
        | (
            solve.successful
            & jnp.isfinite(solve.condition_estimate)
            & (solve.condition_estimate <= maximum_condition)
        )
    )
    return APICGatherResult(
        jnp.where(active[:, None], velocity, 0.0),
        jnp.where(active[:, None, None], velocity_gradient, 0.0),
        affine_moment,
        state.second_moments,
        jnp.where(active[:, None, None], affine_velocity, 0.0),
        solve.condition_estimate,
        successful,
    )


def _oracle_payload(
    state: ParticleGridSplatState,
    mass: Array,
    velocity: Array,
    affine_velocity: Array,
    reference_volume: Array,
    first_piola: Array,
    deformation_gradient: Array,
    external_acceleration: Array,
    active: Array,
    /,
    *,
    physical_deformation_gradient: Array | None = None,
) -> Array:
    weights = state.stencil.weights
    affine_route = _dense("pij,prj->pri", affine_velocity, state.route_offsets)
    momentum = (
        weights[..., None] * mass[:, None, None] * (velocity[:, None, :] + affine_route)
    )
    kirchhoff = _dense("pij,pkj->pik", first_piola, deformation_gradient)
    if physical_deformation_gradient is not None:
        reference_volume = (
            reference_volume
            * jnp.linalg.det(physical_deformation_gradient)
            / jnp.linalg.det(deformation_gradient)
        )
    spatial = state.weight_gradients.shape[-1]
    internal = -reference_volume[:, None, None] * _dense(
        "pij,prj->pri", kirchhoff[:, :spatial, :spatial], state.weight_gradients
    )
    if state.hoop_gradients is not None:
        internal = internal.at[..., 0].add(
            -reference_volume[:, None] * kirchhoff[:, 2, 2, None] * state.hoop_gradients
        )
    if state.cell_average_gradients is not None:
        mean = jnp.trace(kirchhoff, axis1=-2, axis2=-1) / kirchhoff.shape[-1]
        internal = internal - reference_volume[:, None, None] * mean[:, None, None] * (
            state.cell_average_gradients - state.weight_gradients
        )
        if state.hoop_gradients is not None:
            assert state.cell_average_hoop_gradients is not None
            internal = internal.at[..., 0].add(
                -reference_volume[:, None]
                * mean[:, None]
                * (state.cell_average_hoop_gradients - state.hoop_gradients)
            )
    external = (
        weights[..., None] * mass[:, None, None] * external_acceleration[:, None, :]
    )
    if state.radial_parity is not None:
        momentum = momentum.at[..., 0].multiply(state.radial_parity)
        internal = internal.at[..., 0].multiply(state.radial_parity)
        external = external.at[..., 0].multiply(state.radial_parity)
    return jnp.where(
        active[:, None, None] & state.stencil.valid[..., None],
        jnp.concatenate((momentum, internal, external), axis=-1),
        0.0,
    )


def _oracle_kinetic_energy(
    mass: Array,
    velocity: Array,
    affine_velocity: Array,
    particle_moment: Array,
    active: Array,
    /,
    *,
    first_moment: Array | None = None,
) -> Array:
    affine = _dense("pij,pjk,pik->p", affine_velocity, particle_moment, affine_velocity)
    if first_moment is not None:
        affine = affine + 2.0 * _dense(
            "pi,pij,pj->p", velocity, affine_velocity, first_moment
        )
    terms = 0.5 * mass * (jnp.sum(velocity * velocity, axis=-1) + affine)
    return jnp.sum(jnp.where(active, terms, 0.0))


def _oracle_cross(left: Array, right: Array) -> Array:
    if left.shape[-1] == 2:
        return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]
    return jnp.cross(left, right)


def _oracle_angular_momentum(
    position: Array,
    velocity: Array,
    affine_velocity: Array,
    mass: Array,
    state: ParticleGridSplatState,
    active: Array,
    /,
) -> Array:
    three = position.shape[-1] == 3
    orbital = _oracle_cross(position, mass[:, None] * velocity)
    affine_route = _dense("pij,prj->pri", affine_velocity, state.route_offsets)
    momentum = mass[:, None, None] * state.stencil.weights[..., None] * affine_route
    valid = state.stencil.valid[..., None] if three else state.stencil.valid
    affine = jnp.sum(
        jnp.where(valid, _oracle_cross(state.route_offsets, momentum), 0.0), axis=(0, 1)
    )
    mask = active[:, None] if three else active
    return jnp.sum(jnp.where(mask, orbital, 0.0), axis=0) + affine


def _assert_gather_close(
    actual: APICGatherResult, expected: APICGatherResult, *, tolerance: float
) -> None:
    for name in ("velocity", "velocity_gradient", "affine_moment", "affine_velocity"):
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=tolerance,
            atol=tolerance,
            err_msg=name,
        )
    # Conditioning and status depend only on the (unchanged) moment matrices.
    np.testing.assert_array_equal(actual.particle_moment, expected.particle_moment)
    np.testing.assert_allclose(
        actual.condition_estimate, expected.condition_estimate, rtol=1e-12
    )
    np.testing.assert_array_equal(actual.successful, expected.successful)


def _assert_trees_close(actual: object, expected: object, *, tolerance: float) -> None:
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert bool(jnp.all(jnp.isfinite(left)))
        np.testing.assert_allclose(left, right, rtol=tolerance, atol=tolerance)


def _gather_energy(
    gather: Callable[..., APICGatherResult],
    kinetic: Callable[..., Array],
    probe: Array,
    inputs: tuple[ParticleGridSplatState, Array],
    mass: Array,
    /,
) -> Array:
    state, grid_velocity = inputs
    result = gather(state, grid_velocity, _ACTIVE, 1e12)
    first = state.first_moments if state.radial_parity is not None else None
    energy = kinetic(
        mass,
        result.velocity,
        result.affine_velocity,
        result.particle_moment,
        _ACTIVE,
        first_moment=first,
    )
    return (
        energy
        + 0.5 * jnp.sum(result.velocity_gradient**2)
        + jnp.sum(probe * result.affine_moment)
    )


@pytest.mark.parametrize("masked", (False, True), ids=("full-routes", "masked-routes"))
@pytest.mark.parametrize("name", tuple(_CASES))
def test_fused_gather_matches_dense_oracle(name: str, masked: bool) -> None:
    case = _case(name)
    state = _masked_routes(case.state) if masked else case.state
    actual = gather_apic(state, case.grid_velocity, _ACTIVE, 1e12)
    expected = _oracle_gather(state, case.grid_velocity, _ACTIVE, 1e12)
    _assert_gather_close(actual, expected, tolerance=1e-11)
    # Inactive rows are zeroed; the affine moment is reported unmasked.
    np.testing.assert_array_equal(actual.velocity[1], 0.0)
    np.testing.assert_array_equal(actual.velocity_gradient[1], 0.0)
    np.testing.assert_array_equal(actual.affine_velocity[1], 0.0)
    assert float(jnp.max(jnp.abs(actual.affine_moment[1]))) > 0.0
    assert bool(actual.successful)
    # A refused condition limit keeps its meaning (every estimate is >= 1).
    strict = gather_apic(state, case.grid_velocity, _ACTIVE, 0.5)
    assert not bool(strict.successful)
    np.testing.assert_array_equal(
        strict.successful,
        _oracle_gather(state, case.grid_velocity, _ACTIVE, 0.5).successful,
    )


@pytest.mark.parametrize("name", tuple(_CASES))
def test_fused_gather_zero_velocity_is_exactly_inert(name: str) -> None:
    case = _case(name)
    zero = jnp.zeros_like(case.grid_velocity)
    actual = gather_apic(case.state, zero, _ACTIVE, 1e12)
    for value in (
        actual.velocity,
        actual.velocity_gradient,
        actual.affine_moment,
        actual.affine_velocity,
    ):
        np.testing.assert_array_equal(value, jnp.zeros_like(value))
    assert bool(actual.successful)

    def kinetic(grid_velocity: Array) -> Array:
        result = gather_apic(case.state, grid_velocity, _ACTIVE, 1e12)
        return apic_particle_kinetic_energy(
            case.mass,
            result.velocity,
            result.affine_velocity,
            result.particle_moment,
            _ACTIVE,
        )

    gradient = jax.grad(kinetic)(zero)
    np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))


@pytest.mark.parametrize("masked", (False, True), ids=("full-routes", "masked-routes"))
@pytest.mark.parametrize("name", tuple(_CASES))
def test_fused_gather_energy_derivatives_match_oracle(name: str, masked: bool) -> None:
    case = _case(name)
    state = _masked_routes(case.state) if masked else case.state
    dimension = case.grid_velocity.shape[-1]
    rng = np.random.default_rng(31)
    probe = jnp.asarray(rng.normal(size=(4, dimension, dimension)), dtype=jnp.float64)
    fused = functools.partial(
        _gather_energy, gather_apic, apic_particle_kinetic_energy, probe
    )
    oracle = functools.partial(
        _gather_energy, _oracle_gather, _oracle_kinetic_energy, probe
    )
    inputs = (state, case.grid_velocity)
    np.testing.assert_allclose(
        fused(inputs, case.mass), oracle(inputs, case.mass), rtol=1e-11
    )
    # Reverse mode through geometry, weights, moments, and grid velocity.
    _assert_trees_close(
        eqx.filter_grad(fused)(inputs, case.mass),
        eqx.filter_grad(oracle)(inputs, case.mass),
        tolerance=1e-9,
    )
    # Forward-over-reverse Hessian-vector product in the grid velocity.
    direction = jnp.asarray(rng.normal(size=case.grid_velocity.shape))

    def hvp(energy: Callable[..., Array]) -> Array:
        def gradient(grid_velocity: Array) -> Array:
            return jax.grad(lambda v: energy((state, v), case.mass))(grid_velocity)

        return jax.jvp(gradient, (case.grid_velocity,), (direction,))[1]

    np.testing.assert_allclose(hvp(fused), hvp(oracle), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("masked", (False, True), ids=("full-routes", "masked-routes"))
@pytest.mark.parametrize("physical", (False, True), ids=("assumed", "f-bar-volume"))
@pytest.mark.parametrize("name", tuple(_CASES))
def test_fused_payload_and_adjoint_match_oracle(
    name: str, physical: bool, masked: bool
) -> None:
    case = _case(name)
    state = _masked_routes(case.state) if masked else case.state
    dimension = case.position.shape[-1]
    material = 3 if state.hoop_gradients is not None else dimension
    rng = np.random.default_rng(57)
    eye = np.eye(material)
    arguments = (
        jnp.asarray(rng.normal(size=(4, dimension))),
        jnp.asarray(rng.normal(size=(4, dimension, dimension))),
        jnp.asarray(rng.uniform(0.1, 0.4, size=(4,))),
        jnp.asarray(rng.normal(size=(4, material, material))),
        jnp.asarray(eye + 0.1 * rng.normal(size=(4, material, material))),
        jnp.asarray(rng.normal(size=(4, dimension))),
        jnp.asarray(eye + 0.1 * rng.normal(size=(4, material, material))),
    )
    probe = jnp.asarray(rng.normal(size=state.stencil.weights.shape + (3 * dimension,)))

    def payload(builder: Callable[..., Array], inputs: tuple[object, ...]) -> Array:
        route_state, mass, *values, physical_gradient = inputs
        return builder(
            route_state,
            mass,
            *values,
            _ACTIVE,
            physical_deformation_gradient=physical_gradient if physical else None,
        )

    inputs = (state, case.mass, *arguments)
    actual = payload(build_apic_route_payload, inputs)
    expected = payload(_oracle_payload, inputs)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(actual[1], 0.0)
    np.testing.assert_array_equal(
        actual * ~state.stencil.valid[..., None], jnp.zeros_like(actual)
    )

    def work(builder: Callable[..., Array]) -> Callable[..., Array]:
        return lambda inputs: jnp.sum(probe * payload(builder, inputs))

    _assert_trees_close(
        eqx.filter_grad(work(build_apic_route_payload))(inputs),
        eqx.filter_grad(work(_oracle_payload))(inputs),
        tolerance=1e-10,
    )


@pytest.mark.parametrize("first", (False, True), ids=("centered", "first-moment"))
@pytest.mark.parametrize("name", tuple(_CASES))
def test_fused_ledgers_and_energy_gradients_match_oracle(name: str, first: bool) -> None:
    case = _case(name)
    dimension = case.position.shape[-1]
    rng = np.random.default_rng(91)
    velocity = jnp.asarray(rng.normal(size=(4, dimension)))
    affine = jnp.asarray(rng.normal(size=(4, dimension, dimension)))
    moment = case.state.second_moments
    first_moment = case.state.first_moments if first else None

    def kinetic(energy: Callable[..., Array]) -> Callable[..., Array]:
        return lambda inputs: energy(
            case.mass, *inputs, moment, _ACTIVE, first_moment=first_moment
        )

    fused = kinetic(apic_particle_kinetic_energy)
    oracle = kinetic(_oracle_kinetic_energy)
    np.testing.assert_allclose(
        fused((velocity, affine)), oracle((velocity, affine)), rtol=1e-12
    )
    _assert_trees_close(
        jax.grad(fused)((velocity, affine)),
        jax.grad(oracle)((velocity, affine)),
        tolerance=1e-12,
    )
    zero = (jnp.zeros_like(velocity), jnp.zeros_like(affine))
    assert float(fused(zero)) == 0.0
    for gradient in jax.tree.leaves(jax.grad(fused)(zero)):
        np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))

    def angular(ledger: Callable[..., Array]) -> Callable[..., Array]:
        return lambda inputs: ledger(
            case.position, *inputs, case.mass, case.state, _ACTIVE
        )

    np.testing.assert_allclose(
        angular(apic_particle_angular_momentum)((velocity, affine)),
        angular(_oracle_angular_momentum)((velocity, affine)),
        rtol=1e-12,
        atol=1e-13,
    )
    _assert_trees_close(
        jax.jacrev(angular(apic_particle_angular_momentum))((velocity, affine)),
        jax.jacrev(angular(_oracle_angular_momentum))((velocity, affine)),
        tolerance=1e-12,
    )
