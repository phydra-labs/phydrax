# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Conservative meshfree graph transport: flux closure, limiting, CFL, inflow, refresh."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import PointCloudPlan, PreparedPointCloudDiscretization
from phydrax.discretization.meshfree import (
    ConservativeTransport,
    LocalStencilPolicy,
    MeshfreeEvolutionPlan,
    MeshfreeEvolutionStatus,
    PreparedMeshfreeEvolution,
    TransportRefreshStatus,
    TransportStatus,
)
from phydrax.discretization.meshfree._exterior import (
    MeshfreeExteriorCalculusPlan,
    PreparedMeshfreeExteriorCalculus,
)
from phydrax.solver import FixedStepProblem, solve_fixed_step


_SIZE = 17

type Lattice = tuple[PreparedMeshfreeExteriorCalculus, PreparedPointCloudDiscretization]


def _lattice_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Unit-square lattice with tensor control volumes and outward facet areas.

    Boundary nodes own facet length times outward normal, halved per facet at
    corners; with them an accepted degree-two metric closes every node.
    """
    axis = np.arange(_SIZE) / (_SIZE - 1)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    spacing = 1.0 / (_SIZE - 1)
    lower, upper = np.isclose(points, 0.0), np.isclose(points, 1.0)
    edge = lower | upper
    volumes = spacing**2 * np.prod(np.where(edge, 0.5, 1.0), axis=1)
    share = np.where(edge.sum(axis=1, keepdims=True) > 1, 0.5, 1.0) * spacing
    areas = np.where(lower, -share, np.where(upper, share, 0.0))
    return points, volumes, edge.any(axis=1), areas, spacing


@pytest.fixture(scope="module")
def lattice() -> Lattice:
    points, volumes, boundary, areas, spacing = _lattice_geometry()
    exterior = MeshfreeExteriorCalculusPlan(
        points,
        1.1 * spacing,
        4 * points.shape[0],
        node_volumes=volumes,
        dirichlet=boundary,
        boundary_area_vectors=areas,
    ).prepare()
    assert bool(exterior.metric_result.accepted)
    cloud = PointCloudPlan(
        points, volumes, stencil=LocalStencilPolicy(polynomial_degree=2)
    ).prepare()
    return exterior, cloud


def _swirl(time: jax.Array, points: jax.Array, args: Any) -> jax.Array:
    """Divergence-free flow tangent to every face of the unit square."""
    del time, args
    x, y = points[:, 0], points[:, 1]
    return jnp.stack(
        (
            jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y),
            -jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y),
        ),
        axis=1,
    )


def _no_inflow(time: jax.Array, points: jax.Array, args: Any) -> jax.Array:
    del time, args
    return jnp.zeros(points.shape[0])


def _block(points: jax.Array) -> jax.Array:
    inside = (jnp.abs(points[:, 0] - 0.35) < 0.15) & (jnp.abs(points[:, 1] - 0.5) < 0.15)
    return inside.astype(jnp.float64)


def _closed_evolution(
    transport: ConservativeTransport, plan_id: str
) -> PreparedMeshfreeEvolution:
    return MeshfreeEvolutionPlan(
        transport,
        velocity=_swirl,
        inflow=_no_inflow,
        positivity=True,
        plan_id=plan_id,
    ).prepare()


def test_translation_and_affine_volume_flux_close_the_graph(lattice: Lattice) -> None:
    exterior, _ = lattice
    transport = ConservativeTransport(exterior, scheme="upwind")
    points = exterior.points
    equations = exterior.equation_mask
    # A uniform translation is divergence free at every node, boundary included:
    # the boundary normal measure is the graph's own first-moment deficit.
    translation = transport.volume_rate(
        jnp.broadcast_to(jnp.asarray([1.0, -0.5]), points.shape)
    )
    np.testing.assert_allclose(translation, 0.0, atol=1e-12)
    # An affine dilation has outgoing volume flux d V at every equation node.
    dilation = transport.volume_rate(points - 0.5)
    np.testing.assert_allclose(
        dilation[equations], 2.0 * exterior.node_volumes[equations], rtol=0.0, atol=1e-12
    )
    spacing = 1.0 / (_SIZE - 1)
    face = (
        jnp.isclose(points[:, 0], 1.0)
        & ~jnp.isclose(points[:, 1], 0.0)
        & ~jnp.isclose(points[:, 1], 1.0)
    )
    normal = transport.boundary_normal_measure[face]
    # The closing measure of a straight face is its outward face length.
    np.testing.assert_allclose(normal[:, 0], spacing, atol=1e-9)
    np.testing.assert_allclose(normal[:, 1], 0.0, atol=1e-9)


def test_limited_euler_candidate_stays_bounded_where_the_unlimited_one_undershoots(
    lattice: Lattice,
) -> None:
    exterior, cloud = lattice
    points = exterior.points
    velocity = _swirl(jnp.asarray(0.0), points, None)
    values = _block(points)
    limited = ConservativeTransport(exterior, scheme="limited", reconstruction=cloud)
    unlimited = ConservativeTransport(
        exterior, scheme="reconstructed", reconstruction=cloud
    )
    zero = jnp.zeros_like(values)
    rate = limited.rate(values, velocity, inflow=zero)
    certificate = limited.cfl(velocity, 1.0)
    assert int(rate.status) == int(TransportStatus.ACCEPTED)
    assert bool(certificate.certified)
    assert int(rate.limiter_switched) > 0
    assert bool(jnp.all((rate.limiter >= 0.0) & (rate.limiter <= 1.0)))
    np.testing.assert_allclose(rate.conservation_residual, 0.0, atol=1e-13)
    step = certificate.step_bound
    candidate = values + step * rate.value_rate
    assert float(jnp.min(candidate)) >= -1e-13
    # Local bounds hold up to the discrete compression of the velocity-derived
    # volume flux (c_t + div(c u) = 0 concentrates c where div_h u < 0).
    compression = jnp.maximum(-limited.volume_rate(velocity) / exterior.node_volumes, 0.0)
    relaxed = rate.upper_bound * (1.0 + step * compression)
    assert bool(jnp.all(candidate <= relaxed + 1e-13))
    # Same flux and step without the limiter: no certificate, and it undershoots.
    raw = values + step * unlimited.rate(values, velocity, inflow=zero).value_rate
    assert not bool(unlimited.cfl(velocity, step).certified)
    assert float(jnp.min(raw)) < -0.01


def test_closed_flow_conserves_mass_and_positivity_under_declared_cfl(
    lattice: Lattice,
) -> None:
    exterior, cloud = lattice
    transport = ConservativeTransport(exterior, scheme="limited", reconstruction=cloud)
    evolution = _closed_evolution(transport, "closed-limited")
    initial = evolution.initial_state(_block(exterior.points))
    bound = float(evolution.transport_cfl(0.0, initial, 1.0).step_bound)
    steps = 12
    solution = solve_fixed_step(
        FixedStepProblem(
            evolution.ssprk_method("ssprk33"),
            initial,
            t0=0.0,
            t1=steps * bound,
            step_size=bound,
        )
    )
    assert solution.successful
    mass = jnp.asarray([evolution.total_content(state) for state in solution.states])
    np.testing.assert_allclose(mass, mass[0], rtol=1e-13, atol=0.0)
    concentration = evolution.fields(solution.states[-1]).concentration
    assert float(jnp.min(concentration)) >= 0.0
    assert float(jnp.max(concentration)) <= 1.0 + 1e-12


def test_refused_candidate_is_kept_raw_and_the_full_step_rolls_back(
    lattice: Lattice,
) -> None:
    exterior, _ = lattice
    evolution = _closed_evolution(
        ConservativeTransport(exterior, scheme="upwind"), "closed-upwind"
    )
    initial = evolution.initial_state(_block(exterior.points))
    step = 4.0 * float(evolution.transport_cfl(0.0, initial, 1.0).step_bound)
    method = evolution.ssprk_method("ssprk33")
    attempt = method.step(
        jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(step), None
    )
    assert not bool(attempt.successful)
    # The raw candidate keeps its negative content; nothing is clipped.
    assert float(jnp.min(attempt.candidate_state)) < 0.0
    np.testing.assert_array_equal(attempt.accepted_state, initial)
    # The refusal publishes its admission and the violated graph CFL.
    evidence = attempt.evidence
    assert evidence is not None
    assert int(evidence.admission.status) == int(MeshfreeEvolutionStatus.NEGATIVE_STATE)
    assert evidence.transport_cfl is not None
    np.testing.assert_allclose(evidence.transport_cfl.cfl, 4.0, rtol=1e-12)
    assert not bool(evidence.transport_cfl.admitted)
    rollout = solve_fixed_step(
        FixedStepProblem(method, initial, t0=0.0, t1=2 * step, step_size=step)
    )
    assert not rollout.successful
    np.testing.assert_array_equal(rollout.states[-1], initial)


def test_inflow_pulse_is_advected_with_reconstruction_accuracy(lattice: Lattice) -> None:
    exterior, cloud = lattice
    speed = jnp.asarray([1.0, 0.5])

    def profile(q: jax.Array) -> jax.Array:
        radial = (q[:, 0] + 0.1) ** 2 + (q[:, 1] - 0.3) ** 2
        return jnp.exp(-30.0 * radial) * (1.0 + 0.5 * jnp.sin(3.0 * q[:, 1]))

    def exact(time: jax.Array | float, points: jax.Array) -> jax.Array:
        return profile(points - time * speed)

    def velocity(time: jax.Array, points: jax.Array, args: Any) -> jax.Array:
        del time, args
        return jnp.broadcast_to(speed, points.shape)

    def inflow(time: jax.Array, points: jax.Array, args: Any) -> jax.Array:
        del args
        return exact(time, points)

    points = exterior.points
    final = 0.4
    errors = {}
    for scheme in ("upwind", "limited"):
        transport = ConservativeTransport(
            exterior,
            scheme=scheme,
            reconstruction=None if scheme == "upwind" else cloud,
        )
        evolution = MeshfreeEvolutionPlan(
            transport,
            velocity=velocity,
            inflow=inflow,
            inflow_treatment="strong",
            plan_id=f"pulse-{scheme}",
        ).prepare()
        initial = evolution.initial_state(exact(0.0, points))
        bound = float(evolution.transport_cfl(0.0, initial, 1.0).step_bound)
        steps = int(np.ceil(final / bound))
        solution = solve_fixed_step(
            FixedStepProblem(
                evolution.ssprk_method("ssprk33"),
                initial,
                t0=0.0,
                t1=final,
                step_size=final / steps,
            ),
            save_every=steps,
        )
        assert solution.successful
        # The pulse enters through the inflow boundary.
        assert float(evolution.total_content(solution.states[-1])) > 2.0 * float(
            evolution.total_content(initial)
        )
        error = jnp.abs(
            evolution.fields(solution.states[-1]).concentration - exact(final, points)
        )
        errors[scheme] = float(jnp.sum(exterior.node_volumes * error))
    assert errors["limited"] < 0.5 * errors["upwind"]


def test_refresh_is_trusted_within_the_topology_margin_and_refused_beyond(
    lattice: Lattice,
) -> None:
    exterior, _ = lattice
    transport = ConservativeTransport(exterior, scheme="upwind")
    margin = exterior.topology_trust_margin
    assert margin > 0.0
    points = exterior.points
    # A rigid translation keeps the lattice moments, so the re-solved metric is
    # admitted; only the topology witness is exercised here.
    moved = points + 0.4 * margin * jnp.asarray([0.6, -0.8])
    near = transport.refresh(moved)
    assert bool(near.accepted)
    assert int(near.status) == int(TransportRefreshStatus.ACCEPTED)
    np.testing.assert_allclose(near.transport.exterior.points, moved)
    assert bool(near.transport.exterior.metric_result.accepted)
    far = transport.refresh(points + 2.0 * margin)
    assert not bool(far.accepted)
    assert int(far.status) == int(TransportRefreshStatus.TOPOLOGY_TRUST_EXCEEDED)
    # Refusal is rollback: the returned owner is the unchanged one.
    np.testing.assert_array_equal(far.transport.exterior.points, points)


def test_strong_inflow_rows_follow_the_inflow_rate_with_an_exact_ledger(
    lattice: Lattice,
) -> None:
    exterior, cloud = lattice
    transport = ConservativeTransport(
        exterior, scheme="reconstructed", reconstruction=cloud
    )
    points = exterior.points
    velocity = jnp.broadcast_to(jnp.asarray([1.0, 0.5]), points.shape)
    values = 2.0 + jnp.sin(2.0 * points[:, 0] + 1.0) * jnp.cos(3.0 * points[:, 1])
    held = jnp.cos(points[:, 1])
    weak = transport.rate(values, velocity, inflow=values)
    strong = transport.rate(values, velocity, inflow=values, inflow_rate=held)
    inflow = ~exterior.equation_mask & (weak.boundary_volume_flux < 0)
    assert bool(jnp.any(inflow))
    # Inflow nodes are prescribed rows; every other node keeps its flux rate.
    np.testing.assert_allclose(strong.value_rate[inflow], held[inflow], atol=1e-12)
    np.testing.assert_allclose(
        strong.content_rate[~inflow], weak.content_rate[~inflow], atol=1e-14
    )
    # The supplied content is boundary exchange, so the ledger stays exact.
    np.testing.assert_allclose(strong.conservation_residual, 0.0, atol=1e-13)
    with pytest.raises(ValueError, match="declared inflow state"):
        transport.rate(values, velocity, inflow_rate=held)


def test_boundary_rows_are_exact_for_affine_data_and_unclosed_exteriors_are_refused(
    lattice: Lattice,
) -> None:
    """Without moment rows, boundary nodes miss the tangential flux (O(1) error).

    With declared area vectors the deficit is the outward facet measure and the
    reconstructed rate of a linear field under a uniform velocity is exact at
    every node, boundary included.
    """
    exterior, cloud = lattice
    points, volumes, boundary, areas, spacing = _lattice_geometry()
    transport = ConservativeTransport(
        exterior, scheme="reconstructed", reconstruction=cloud
    )
    nodes = exterior.points
    velocity = jnp.broadcast_to(jnp.asarray([1.0, 0.5]), nodes.shape)
    np.testing.assert_allclose(transport.boundary_normal_measure, areas, atol=1e-9)
    values = 1.0 + 0.3 * nodes[:, 0] - 0.2 * nodes[:, 1]
    rate = transport.rate(values, velocity, inflow=values)
    np.testing.assert_allclose(rate.value_rate, -(0.3 - 0.5 * 0.2), atol=1e-8)
    np.testing.assert_allclose(rate.conservation_residual, 0.0, atol=1e-13)
    unclosed = MeshfreeExteriorCalculusPlan(
        points,
        1.1 * spacing,
        4 * points.shape[0],
        node_volumes=volumes,
        dirichlet=boundary,
    ).prepare()
    with pytest.raises(ValueError, match="boundary_area_vectors"):
        ConservativeTransport(unclosed, scheme="upwind")


def test_open_graph_requires_a_declared_inflow_state(lattice: Lattice) -> None:
    exterior, _ = lattice
    transport = ConservativeTransport(exterior, scheme="upwind")
    values = jnp.zeros(exterior.points.shape[0])
    with pytest.raises(ValueError, match="declared inflow"):
        transport.rate(values, jnp.ones_like(exterior.points))
    with pytest.raises(ValueError, match="inflow"):
        MeshfreeEvolutionPlan(transport, velocity=_swirl, plan_id="undeclared")
