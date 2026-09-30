#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Independent conservation, physical work, and Fourier chain contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax.discretization as D
import phydrax.discretization.pic as PIC
from phydrax.discretization import (
    NonuniformCellAxisSpec,
    StructuredCochainBridge,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from phydrax.discretization._cubical_whitney import (
    CubicalSplineWhitneyKernel,
    PICShapeOrder,
)


def _bridge(
    *, nonuniform: bool = False, periodic: bool = False
) -> StructuredCochainBridge:
    axes = tuple(
        NonuniformCellAxisSpec(jnp.asarray((0.0, 0.08, 0.23, 0.47, 0.71, 1.0)))
        if nonuniform
        else UniformCellAxisSpec(8, periodic=periodic)
        for _ in range(3)
    )
    grid = TensorGridPlan(axes, axis_names=("x", "y", "z")).prepare(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), dtype=jnp.float64)
    )
    return StructuredCochainBridge(grid)


def _electric(bridge: StructuredCochainBridge, field: np.ndarray, /) -> Array:
    shapes = bridge.orientation_shapes[1]
    return bridge.pack_edge_circulation(
        (
            jnp.full(shapes[0], field[0], dtype=jnp.float64),
            jnp.full(shapes[1], field[1], dtype=jnp.float64),
            jnp.full(shapes[2], field[2], dtype=jnp.float64),
        )
    )


@pytest.mark.parametrize("order", (1, 2, 3), ids=("linear", "quadratic", "cubic"))
def test_charge_conservation_on_exact_knot_intervals(order: PICShapeOrder) -> None:
    bridge = _bridge()
    kernel = CubicalSplineWhitneyKernel(bridge, order)
    start = jnp.asarray(((0.31, 0.36, 0.41), (0.64, 0.59, 0.53)), dtype=jnp.float64)
    end = jnp.asarray(((0.66, 0.61, 0.57), (0.34, 0.39, 0.43)), dtype=jnp.float64)
    charges = jnp.asarray((1.7, -0.8), dtype=jnp.float64)
    query = jax.jit(lambda a, b: kernel.integrate_segments(a, b, maximum_segments=12))(
        start, end
    )
    difference = kernel.integrate_points(end).deposit(charges) - kernel.integrate_points(
        start
    ).deposit(charges)
    boundary = (
        bridge.cochain.topology.incidences[0]
        .exterior_derivative()
        .transpose_mv(query.deposit(charges))
    )
    assert bool(jnp.all(query.successful))
    np.testing.assert_allclose(boundary, difference, atol=4e-13, rtol=4e-13)


def test_nonuniform_segment_work_uses_actual_cell_widths() -> None:
    bridge = _bridge(nonuniform=True)
    kernel = CubicalSplineWhitneyKernel(bridge, 1)
    start = jnp.asarray(((0.04, 0.20, 0.36), (0.92, 0.77, 0.81)), dtype=jnp.float64)
    end = jnp.asarray(((0.88, 0.73, 0.82), (0.11, 0.18, 0.28)), dtype=jnp.float64)
    field = np.asarray((2.3, -1.1, 0.7))
    electric = _electric(bridge, field)
    query = kernel.integrate_segments(start, end, maximum_segments=20)
    assert bool(jnp.all(query.successful))
    np.testing.assert_allclose(
        query.gather(electric), np.asarray(end - start) @ field, atol=3e-14
    )
    samples = kernel.evaluate(start, 1).gather(electric)
    np.testing.assert_allclose(samples, np.broadcast_to(field, samples.shape), atol=3e-14)


@pytest.mark.parametrize("order", (1, 2, 3), ids=("linear", "quadratic", "cubic"))
@pytest.mark.parametrize(
    "rate", (0.0, 0.03, 7.0, -23.0), ids=("zero", "small", "moderate", "negative-fast")
)
def test_phase_integral_of_constant_physical_field(
    order: PICShapeOrder, rate: float
) -> None:
    bridge = _bridge()
    kernel = CubicalSplineWhitneyKernel(bridge, order)
    start = jnp.asarray(((0.31, 0.34, 0.38),), dtype=jnp.float64)
    end = jnp.asarray(((0.66, 0.63, 0.61),), dtype=jnp.float64)
    field = np.asarray((0.4, -1.2, 2.1))
    electric = _electric(bridge, field)
    query = kernel.integrate_segments(
        start, end, weight="phase", phase_rate=rate, maximum_segments=12
    )
    moment = 1.0 if rate == 0.0 else np.expm1(1j * rate) / (1j * rate)
    expected = np.asarray(end - start) @ field * moment
    assert bool(jnp.all(query.successful))
    np.testing.assert_allclose(query.gather(electric), expected, atol=3e-13, rtol=3e-13)


def test_complex_phase_gather_deposit_are_hermitian_adjoint() -> None:
    kernel = CubicalSplineWhitneyKernel(_bridge(), 3)
    query = kernel.integrate_segments(
        jnp.asarray(((0.33, 0.38, 0.41), (0.56, 0.61, 0.59))),
        jnp.asarray(((0.62, 0.57, 0.51), (0.39, 0.37, 0.42))),
        weight="phase",
        phase_rate=jnp.asarray((3.4, -8.7)),
        maximum_segments=12,
    )
    rng = np.random.default_rng(9407)
    electric = jnp.asarray(
        rng.normal(size=query.dof_count) + 1j * rng.normal(size=query.dof_count)
    )
    weights = jnp.asarray((0.7 + 0.3j, -1.2 + 0.8j))
    np.testing.assert_allclose(
        jnp.vdot(query.gather(electric), weights),
        jnp.vdot(electric, query.deposit(weights)),
        atol=4e-13,
        rtol=4e-13,
    )


def test_periodic_reverse_seam_crossing_preserves_work() -> None:
    bridge = _bridge(periodic=True)
    kernel = CubicalSplineWhitneyKernel(bridge, 1)
    first = jnp.asarray(((0.0, 0.4, 0.6),), dtype=jnp.float64)
    last = jnp.asarray(((-0.37, 0.4, 0.6),), dtype=jnp.float64)
    electric = _electric(bridge, np.asarray((1.0, 1.0, 1.0)))
    query = kernel.integrate_segments(first, last, maximum_segments=8)
    assert bool(jnp.all(query.successful))
    np.testing.assert_allclose(query.gather(electric), (-0.37,), atol=2e-14)


def test_overflow_does_not_claim_a_complete_chain() -> None:
    kernel = CubicalSplineWhitneyKernel(_bridge(), 1)
    start = jnp.asarray(((0.03, 0.04, 0.05),), dtype=jnp.float64)
    end = jnp.asarray(((0.91, 0.87, 0.82),), dtype=jnp.float64)
    query = kernel.integrate_segments(start, end, maximum_segments=2)
    assert bool(query.overflow[0])
    assert not bool(query.successful[0])


def test_zero_length_chain_has_zero_work_without_overflow() -> None:
    kernel = CubicalSplineWhitneyKernel(_bridge(), 3)
    point = jnp.asarray(((0.5, 0.5, 0.5),), dtype=jnp.float64)
    query = kernel.integrate_segments(point, point, maximum_segments=1)
    assert bool(query.successful[0])
    assert not bool(query.overflow[0])
    np.testing.assert_array_equal(
        query.gather(jnp.ones((query.dof_count,), dtype=jnp.float64)), (0.0,)
    )


@pytest.mark.parametrize("order", (1, 2, 3), ids=("linear", "quadratic", "cubic"))
@pytest.mark.parametrize(
    "rate", (0.02, 9.0, -33.0), ids=("small", "oscillatory", "fast-negative")
)
def test_nonconstant_phase_moments_against_independent_quadrature(
    order: PICShapeOrder, rate: float
) -> None:
    bridge = _bridge()
    kernel = CubicalSplineWhitneyKernel(bridge, order)
    axes = bridge.grid.structured_axes
    shapes = bridge.orientation_shapes[1]
    yz = (
        axes[1].point_coordinates[None, :, None]
        * axes[2].point_coordinates[None, None, :]
    )
    electric = bridge.pack_edge_circulation(
        (
            jnp.broadcast_to(yz, shapes[0]),
            jnp.zeros(shapes[1], dtype=jnp.float64),
            jnp.zeros(shapes[2], dtype=jnp.float64),
        )
    )
    first = np.asarray(((0.31, 0.36, 0.41),), dtype=np.float64)
    last = np.asarray(((0.66, 0.61, 0.57),), dtype=np.float64)
    query = kernel.integrate_segments(
        first, last, weight="phase", phase_rate=rate, maximum_segments=12
    )
    nodes, weights = np.polynomial.legendre.leggauss(64)
    parameter = 0.5 * (nodes + 1.0)
    positions = first[0] + parameter[:, None] * (last[0] - first[0])
    reference = (last[0, 0] - first[0, 0]) * np.sum(
        0.5 * weights * positions[:, 1] * positions[:, 2] * np.exp(1j * rate * parameter)
    )
    assert bool(query.successful[0])
    np.testing.assert_allclose(
        query.gather(electric), (reference,), atol=3e-13, rtol=3e-13
    )


def test_conservative_pic_current_is_invariant_to_particle_slot_order() -> None:
    bridge = _bridge()
    particles = D.ParticleSetPlan(
        jnp.arange(3), jnp.ones((3,), dtype=jnp.float64), ambient_dimension=3
    ).prepare()
    species = D.ChargedParticlePlan(
        jnp.ones((3,), dtype=jnp.float64), "chain-slots"
    ).prepare(particles)
    transfer = PIC.PICParticleCochainTransferPlan(bridge, shape_order=3).prepare(species)
    current = PIC.ChargeConservingCurrentPlan(transfer, maximum_segments_per_particle=12)
    first = jnp.asarray(((0.31, 0.36, 0.41), (0.64, 0.59, 0.53), (0.43, 0.47, 0.55)))
    last = jnp.asarray(((0.66, 0.61, 0.57), (0.34, 0.39, 0.43), (0.61, 0.58, 0.42)))
    charges = jnp.asarray((1.7, -0.8, 2.3))
    order = jnp.asarray((2, 0, 1), dtype=jnp.int32)
    original = current.deposit(first, last, 0.2, macrocharge=charges)
    permuted = current.deposit(first[order], last[order], 0.2, macrocharge=charges[order])
    assert bool(original.successful) and bool(permuted.successful)
    np.testing.assert_array_equal(original.current, permuted.current)
    electric = _electric(bridge, np.asarray((0.4, -1.2, 2.1)))
    load = bridge.cochain.hodge_star(1, original.current)
    expected_work = np.sum(
        np.asarray(charges)[:, None]
        * np.asarray(last - first)
        * np.asarray((0.4, -1.2, 2.1))
    )
    np.testing.assert_allclose(jnp.vdot(electric, load) * 0.2, expected_work, atol=4e-13)


@pytest.mark.parametrize("charge_sign", (-1.0, 1.0))
def test_nonuniform_pic_transfer_and_current_preserve_charge_and_work(
    charge_sign: float,
) -> None:
    bridge = _bridge(nonuniform=True)
    masses = jnp.asarray((1.3, 0.7), dtype=jnp.float64)
    charges = charge_sign * jnp.asarray((1.3, 0.7), dtype=jnp.float64)
    particles = D.ParticleSetPlan(jnp.arange(2), masses, ambient_dimension=3).prepare()
    species = D.ChargedParticlePlan(charges, "nonuniform-chain").prepare(particles)
    transfer = PIC.PICParticleCochainTransferPlan(bridge, shape_order=1).prepare(species)
    current = PIC.ChargeConservingCurrentPlan(transfer, maximum_segments_per_particle=20)
    first = jnp.asarray(((0.04, 0.20, 0.36), (0.92, 0.77, 0.81)), dtype=jnp.float64)
    last = jnp.asarray(((0.88, 0.73, 0.82), (0.11, 0.18, 0.28)), dtype=jnp.float64)
    result = current.deposit(first, last, 0.3)
    assert bool(result.successful)
    assert not bool(result.capacity_overflow)
    np.testing.assert_allclose(
        jnp.sum(result.end_charge.content), charge_sign * 2.0, atol=2e-14
    )
    electric = _electric(bridge, np.asarray((2.3, -1.1, 0.7)))
    gathered = transfer.gather_electric(transfer.build(first), electric)
    np.testing.assert_allclose(
        gathered.values, np.broadcast_to((2.3, -1.1, 0.7), (2, 3)), atol=3e-14
    )
    expected = np.sum(
        charge_sign
        * np.asarray((1.3, 0.7))[:, None]
        * np.asarray(last - first)
        * np.asarray((2.3, -1.1, 0.7))
    )
    np.testing.assert_allclose(
        jnp.vdot(electric, bridge.cochain.hodge_star(1, result.current)) * 0.3,
        expected,
        atol=3e-13,
    )
