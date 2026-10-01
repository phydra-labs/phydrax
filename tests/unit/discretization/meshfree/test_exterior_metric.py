# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization.meshfree._exterior import MeshfreeExteriorCalculusPlan
from phydrax.discretization.meshfree._exterior_metric import (
    MeshfreeMetricPolicy,
    MeshfreeMetricStatus,
)


def test_positive_chain_reproduces_quadratic_and_native_one_complex() -> None:
    prepared = MeshfreeExteriorCalculusPlan(
        np.asarray([[-1.0], [0.0], [1.0]]),
        1.1,
        3,
        node_volumes=np.asarray([1.0, 1.0, 1.0]),
        dirichlet=np.asarray([True, False, True]),
    ).prepare()
    np.testing.assert_allclose(prepared.metric_result.weights, [1.0, 1.0], atol=1e-8)
    assert bool(prepared.metric_result.accepted)
    np.testing.assert_allclose(
        prepared.diffusion().mv(jnp.asarray([1.0, 0.0, 1.0])),
        [-1.0, 2.0, -1.0],
        atol=1e-8,
    )
    assert prepared.hilbert_complex().top_degree == 1
    assert bool(prepared.diffusion().evidence.spd)
    assert bool(prepared.diffusion().evidence.maximum_principle)
    assert int(prepared.diffusion().evidence.positive_inertia) == 1


def test_signed_one_sided_metric_is_not_a_hilbert_pairing() -> None:
    prepared = MeshfreeExteriorCalculusPlan(
        np.asarray([[0.0], [1.0], [2.0]]),
        2.1,
        3,
        node_volumes=np.asarray([1.0, 1.0, 1.0]),
        dirichlet=np.asarray([False, True, True]),
    ).prepare()
    np.testing.assert_allclose(prepared.metric_result.weights, [-2.0, 1.0], atol=1e-8)
    assert bool(prepared.metric_result.accepted)
    assert int(prepared.metric_result.negative_count) == 1
    diffusion = prepared.diffusion()
    np.testing.assert_allclose(
        diffusion.mv(jnp.asarray([0.0, 1.0, 4.0]))[0], 2.0, atol=1e-8
    )
    np.testing.assert_allclose(
        diffusion.conservation_residual(np.asarray([0.0, 1.0, 4.0])), 0.0, atol=1e-8
    )
    assert not bool(diffusion.evidence.spd)
    assert not bool(diffusion.evidence.maximum_principle)
    assert not bool(diffusion.evidence.inertia_available)
    assert int(diffusion.evidence.negative_inertia) == -1
    with pytest.raises(ValueError, match="positive cochains"):
        prepared.to_cochain()


def test_dependent_incompatible_equations_are_not_regularized_to_success() -> None:
    prepared = MeshfreeExteriorCalculusPlan(
        np.asarray([[0.0], [1.0]]),
        1.1,
        1,
        node_volumes=np.asarray([1.0, 1.0]),
    ).prepare()
    result = prepared.metric_result
    assert result.rank == 1
    assert result.redundant_constraints == 3
    assert not bool(result.accepted)
    assert int(result.status) == int(MeshfreeMetricStatus.INCOMPATIBLE_MOMENTS)
    assert not bool(jnp.all(result.node_feasible))
    assert float(jnp.max(jnp.abs(result.moment_residual))) > 1.0


def test_relaxed_policy_reports_actual_slack_and_unanchored_nullspace() -> None:
    penalty = 10.0
    prepared = MeshfreeExteriorCalculusPlan(
        np.asarray([[0.0], [1.0]]),
        1.1,
        1,
        node_volumes=np.asarray([1.0, 1.0]),
        metric_policy=MeshfreeMetricPolicy(acceptance="relaxed", slack_penalty=penalty),
    ).prepare()
    result = prepared.metric_result
    phi = float(prepared.metric_system.prior[0])
    expected = 4 * penalty / (1 / phi + 4 * penalty)
    np.testing.assert_allclose(result.weights, [expected], atol=1e-8)
    np.testing.assert_allclose(result.slack, result.moment_residual, atol=1e-12)
    assert bool(result.accepted)
    assert not bool(result.exact)
    assert int(result.status) == int(MeshfreeMetricStatus.RELAXED)
    evidence = prepared.diffusion().evidence
    assert not bool(evidence.anchored_components)
    assert not bool(evidence.spd)
    assert not bool(evidence.maximum_principle)


def test_native_nonnegative_constraints_solve_and_refuse_farkas_infeasibility() -> None:
    feasible = (
        MeshfreeExteriorCalculusPlan(
            np.asarray([[-1.0], [0.0], [1.0]]),
            1.1,
            3,
            node_volumes=np.asarray([1.0, 1.0, 1.0]),
            dirichlet=np.asarray([True, False, True]),
            metric_policy=MeshfreeMetricPolicy("nonnegative", tolerance=1e-7),
        )
        .prepare()
        .metric_result
    )
    assert bool(feasible.accepted)
    assert bool(feasible.nonnegative)
    np.testing.assert_allclose(feasible.weights, [1.0, 1.0], atol=1e-6)
    assert not bool(feasible.derivative_available)
    impossible = MeshfreeExteriorCalculusPlan(
        np.asarray([[0.0], [1.0], [2.0]]),
        2.1,
        3,
        node_volumes=np.asarray([1.0, 1.0, 1.0]),
        dirichlet=np.asarray([False, True, True]),
        metric_policy=MeshfreeMetricPolicy("nonnegative", tolerance=1e-7),
    ).prepare()
    witness = jnp.asarray([2.0, -1.0])
    assert bool(jnp.all(impossible.metric_system.constraint.transpose_mv(witness) >= 0))
    assert float(jnp.dot(impossible.metric_system.rhs, witness)) < 0
    assert not bool(impossible.metric_result.accepted)
    assert not bool(impossible.metric_result.derivative_available)


def test_fixed_rank_coordinate_derivatives_and_topology_refusal() -> None:
    prepared = MeshfreeExteriorCalculusPlan(
        np.asarray([[-1.0], [0.0], [1.0]]),
        1.1,
        3,
        node_volumes=np.asarray([1.0, 1.0, 1.0]),
        dirichlet=np.asarray([True, False, True]),
    ).prepare()

    def weights(scale: Array) -> Array:
        return prepared.refresh(points=prepared.points * scale).metric_result.weights

    value, tangent = jax.jvp(weights, (jnp.asarray(1.0),), (jnp.asarray(1.0),))
    np.testing.assert_allclose(value, [1.0, 1.0], atol=1e-8)
    np.testing.assert_allclose(tangent, [-2.0, -2.0], atol=1e-7)
    np.testing.assert_allclose(
        jax.grad(lambda scale: jnp.sum(weights(scale)))(jnp.asarray(1.0)), -4.0, atol=1e-7
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="fixed-radius topology"):
        prepared.refresh(points=prepared.points * 1.2)


def test_amplification_refusal_uses_absolute_true_moments() -> None:
    prepared = MeshfreeExteriorCalculusPlan(
        np.asarray([[-1.0], [0.0], [1.0]]),
        1.1,
        3,
        node_volumes=np.asarray([1.0, 1.0, 1.0]),
        dirichlet=np.asarray([True, False, True]),
        metric_policy=MeshfreeMetricPolicy(amplification_limit=1.0),
    ).prepare()
    np.testing.assert_allclose(prepared.metric_result.amplification, [2.0], atol=1e-8)
    assert bool(prepared.metric_result.exact)
    assert not bool(prepared.metric_result.accepted)
    assert int(prepared.metric_result.status) == int(
        MeshfreeMetricStatus.AMPLIFICATION_LIMIT
    )


def test_inactive_zero_volume_slots_are_compacted_before_native_admission() -> None:
    prepared = MeshfreeExteriorCalculusPlan(
        np.asarray([[-1.0], [0.0], [1.0], [10.0]]),
        1.1,
        4,
        node_volumes=np.asarray([1.0, 1.0, 1.0, 0.0]),
        active=np.asarray([True, True, True, False]),
        dirichlet=np.asarray([True, False, True, False]),
    ).prepare()
    np.testing.assert_array_equal(prepared.capacity_indices, [0, 1, 2])
    np.testing.assert_allclose(prepared.to_cochain().hodge_diagonal(0), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(
        prepared.diffusion().mv(np.asarray([1.0, 0.0, 1.0])), [-1.0, 2.0, -1.0], atol=1e-8
    )


def test_positive_refresh_retains_raw_incompatible_candidate_and_refuses_conversion() -> (
    None
):
    prepared = MeshfreeExteriorCalculusPlan(
        np.asarray([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]),
        1.1,
        10,
        node_volumes=np.ones(5),
        dirichlet=np.asarray([False, True, True, True, True]),
    ).prepare()
    candidate = prepared.refresh(points=prepared.points.at[4, 0].set(0.02))
    assert not bool(candidate.metric_result.accepted)
    assert int(candidate.metric_result.status) == int(
        MeshfreeMetricStatus.INCOMPATIBLE_MOMENTS
    )
    assert float(jnp.max(jnp.abs(candidate.metric_result.second_moment_residual))) > 0.01
    assert not bool(candidate.metric_result.derivative_available)
    with pytest.raises(eqx.EquinoxRuntimeError, match="rollback only"):
        candidate.to_cochain()


def test_same_shape_foreign_spatial_relation_is_refused_before_metric_fit() -> None:
    from phydrax.discretization.meshfree._neighbors import MeshfreeEdgeRelationPlan

    foreign = MeshfreeEdgeRelationPlan(
        np.asarray([[-1.0], [0.0], [1.0]]), 1.1, 3
    ).prepare()
    plan = MeshfreeExteriorCalculusPlan(
        np.asarray([[-1.0], [0.01], [1.0]]),
        1.1,
        3,
        node_volumes=np.asarray([1.0, 1.0, 1.0]),
        dirichlet=np.asarray([True, False, True]),
    )
    with pytest.raises(ValueError, match="different point source"):
        plan.prepare(edge_relation=foreign)


def test_nonmaximal_compatible_profile_does_not_claim_coordinate_derivatives() -> None:
    # Eight axis-aligned edges admit moments, but the identically zero xy row
    # can become independent under arbitrarily small off-axis perturbations.
    prepared = MeshfreeExteriorCalculusPlan(
        np.asarray(
            [
                [0.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [-0.5, 0.0],
                [0.5, 0.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [0.0, -0.5],
                [0.0, 0.5],
            ]
        ),
        1.1,
        36,
        node_volumes=np.ones(9),
        dirichlet=np.asarray([False, True, True, True, True, True, True, True, True]),
    ).prepare()
    assert bool(prepared.metric_result.accepted)
    assert prepared.metric_result.rank == 4
    assert bool(prepared.metric_result.schur_spd)
    assert not bool(prepared.metric_result.rank_profile_stable)
    assert not bool(prepared.metric_result.derivative_available)


def test_domain_normalization_excludes_inactive_kernel_capacity() -> None:
    prepared = MeshfreeExteriorCalculusPlan(
        np.asarray([[-1.0], [0.0], [1.0], [10.0]]),
        1.1,
        4,
        domain_volume=6.0,
        volume_kernel=np.asarray([1.0, 2.0, 3.0, 1000.0]),
        active=np.asarray([True, True, True, False]),
        dirichlet=np.asarray([True, False, True, False]),
    ).prepare()
    np.testing.assert_allclose(prepared.node_volumes, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(jnp.sum(prepared.node_volumes), 6.0)
    np.testing.assert_allclose(prepared.metric_result.weights, [2.0, 2.0], atol=1e-8)
    np.testing.assert_allclose(
        prepared.diffusion().mv(np.asarray([1.0, 0.0, 1.0]))[1], 2.0, atol=1e-8
    )


@pytest.mark.parametrize("count", (32, 64, 128))
def test_relaxed_intrinsic_sphere_admits_true_nonnegative_provider_with_slack(
    count: int,
) -> None:
    from phydrax.discretization.meshfree._surface import SurfacePointCloudPlan
    from phydrax.discretization.meshfree._surface_geometry import ImplicitSurfaceGeometry
    from phydrax.discretization.meshfree._surface_quadrature import (
        SurfaceQuadraturePolicy,
    )
    from phydrax.metrix import RegularLevelSetManifold

    height = 1 - 2 * (np.arange(count) + 0.5) / count
    angle = np.pi * (3 - np.sqrt(5)) * np.arange(count) + np.random.default_rng(
        0
    ).uniform(0, 2 * np.pi)
    radius = np.sqrt(1 - height**2)
    points = jnp.asarray(
        np.stack((radius * np.cos(angle), radius * np.sin(angle), height), axis=1)
    )

    def equation(value: Array) -> Array:
        return jnp.reshape(jnp.sum(value**2) - 1, (1,))

    manifold = RegularLevelSetManifold(equation, ambient_dimension=3, codimension=1)
    surface = SurfacePointCloudPlan(
        points,
        ImplicitSurfaceGeometry(manifold),
        16,
        quadrature=SurfaceQuadraturePolicy(
            "normalized-density", density=jnp.ones((count,)), total_area=4 * np.pi
        ),
    ).prepare()
    graph = surface.conservative_exterior(
        float(5 / np.sqrt(count)),
        maximum_pairs=count * (count - 1) // 2,
        metric_policy=MeshfreeMetricPolicy(
            "nonnegative", acceptance="relaxed", tolerance=1e-7, slack_penalty=100
        ),
    )
    result = graph.metric_result
    assert bool(result.accepted)
    assert bool(result.nonnegative)
    assert not bool(result.exact)
    assert int(result.status) == int(MeshfreeMetricStatus.RELAXED)
    assert result.conic_result is not None
    assert bool(result.conic_result.successful)
    assert float(result.conic_result.kkt_residual_norm) <= 1e-7
    assert abs(float(result.conic_result.complementarity_gap)) <= 1e-7
    assert float(jnp.max(jnp.abs(result.normalized_residual))) > 0.1
    assert not bool(result.derivative_available)
    assert not bool(result.schur_assessed)
    assert not bool(result.schur_spd)
