# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization.meshfree._exterior import (
    MeshfreeCoercivityPolicy,
    MeshfreeExteriorCalculusPlan,
    MeshfreeExteriorRefreshStatus,
    PreparedMeshfreeExteriorCalculus,
)
from phydrax.discretization.meshfree._exterior_metric import (
    BoundaryClosure,
    MeshfreeMetricPolicy,
    MeshfreeMetricResult,
    MeshfreeMetricStatus,
    MetricSign,
    PreparedMeshfreeMetric,
)
from phydrax.linalg import (
    LinearSolveStatus,
    MaterializationPolicy,
    RankPolicy,
    SparseFactorizationPolicy,
)
from phydrax.linalg.svd import DenseSVD, SVDSolvePolicy
from phydrax.optim import ConicConstraintRole, ConicSensitivityStatus
from phydrax.sparse import EdgeRelation


def _dense_design(system: PreparedMeshfreeMetric) -> np.ndarray:
    relation = system.constraint.relation
    assert isinstance(relation, EdgeRelation)
    design = np.zeros((system.constraint.target.size, system.constraint.source.size))
    np.add.at(
        design,
        (np.asarray(relation.target_indices), np.asarray(relation.source_indices)),
        np.asarray(system.constraint.coefficients),
    )
    return design


_STAR = np.asarray([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
_STAR_DIRICHLET = np.asarray([False, True, True, True, True])


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
    assert int(result.rank) == 1
    assert int(result.redundant_constraints) == 3
    assert not bool(result.accepted)
    assert int(result.status) == int(MeshfreeMetricStatus.INCOMPATIBLE_MOMENTS)
    assert int(result.provider_status) == int(LinearSolveStatus.INCOMPATIBLE_RHS)
    assert not bool(jnp.all(result.node_feasible))
    # Independent Farkas check of the published witness in original rows.
    design = _dense_design(prepared.metric_system)
    witness = np.asarray(result.incompatibility_witness)
    np.testing.assert_allclose(design.T @ witness, 0.0, atol=1e-10)
    assert abs(float(witness @ np.asarray(prepared.metric_system.rhs))) > 0.5


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
    assert feasible.active_set is None
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
    result = prepared.metric_result
    assert bool(result.accepted)
    assert int(result.rank) == 4
    assert result.rank_certificate.route == "dense-svd"
    assert bool(result.rank_certificate.fixed_rank)
    assert not bool(result.rank_maximal)
    assert not bool(result.derivative_available)


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
    assert result.conic_execution is not None
    conic = result.conic_execution.result
    assert bool(conic.successful)
    assert float(conic.kkt_residual_norm) <= 1e-7
    assert abs(float(conic.complementarity_gap)) <= 1e-7
    assert float(jnp.max(jnp.abs(result.normalized_residual))) > 0.1
    assert not bool(result.derivative_available)
    assert result.linear_result is None


def test_exact_signed_metric_matches_dense_weighted_minimum_norm_oracle() -> None:
    # Irregular 2-D cloud with a nonuniform Gaussian prior on unequal edges.
    rng = np.random.default_rng(11)
    interior = 0.25 * rng.uniform(-1.0, 1.0, size=(3, 2))
    ring = np.stack(
        (
            np.cos(np.linspace(0, 2 * np.pi, 9)[:-1]),
            np.sin(np.linspace(0, 2 * np.pi, 9)[:-1]),
        ),
        axis=1,
    )
    points = np.concatenate((interior, ring))
    prepared = MeshfreeExteriorCalculusPlan(
        points,
        1.6,
        55,
        node_volumes=np.full(points.shape[0], 0.3),
        dirichlet=np.arange(points.shape[0]) >= 3,
    ).prepare()
    system, result = prepared.metric_system, prepared.metric_result
    design = _dense_design(system) / np.asarray(system.row_scaling)[:, None]
    root = np.sqrt(np.asarray(system.prior))
    rhs = np.asarray(system.rhs / system.row_scaling)
    expected = root * (np.linalg.pinv(design * root, rcond=1e-12) @ rhs)
    assert bool(result.accepted) and bool(result.exact)
    assert int(result.status) == int(MeshfreeMetricStatus.ACCEPTED)
    np.testing.assert_allclose(result.weights, expected, rtol=1e-7, atol=1e-9)
    assert result.linear_result is not None
    evidence = result.linear_result.minimum_norm
    assert evidence is not None and bool(evidence.stationarity_verified)
    assert not bool(evidence.incompatible)
    # Admitted transform: uniformly rescaling rows changes neither the
    # constraint set nor the weighted objective, hence not the optimum.
    rescaled = PreparedMeshfreeMetric(
        system.constraint,
        system.rhs,
        3.0 * system.row_scaling,
        system.prior,
        system.policy,
        equation_count=system.equation_count,
        intrinsic_dimension=system.intrinsic_dimension,
        geometry=system.geometry,
    ).solve()
    np.testing.assert_allclose(rescaled.weights, expected, rtol=1e-7, atol=1e-9)


def test_relaxed_signed_metric_matches_scaled_penalty_oracle() -> None:
    penalty = 50.0
    prepared = MeshfreeExteriorCalculusPlan(
        _STAR,
        1.1,
        10,
        node_volumes=np.ones(5),
        dirichlet=_STAR_DIRICHLET,
        metric_policy=MeshfreeMetricPolicy(
            acceptance="relaxed", slack_penalty=penalty, moment_degree=4
        ),
    ).prepare()
    system, result = prepared.metric_system, prepared.metric_result
    design = _dense_design(system) / np.asarray(system.row_scaling)[:, None]
    root = np.sqrt(np.asarray(system.prior))
    rhs = np.asarray(system.rhs / system.row_scaling)
    # 0.5|z|^2 + C/2 |D^-1 (A S z - b)|^2 with the original row scaling D.
    transformed = design * root
    z = np.linalg.solve(
        np.eye(root.size) + penalty * transformed.T @ transformed,
        penalty * transformed.T @ rhs,
    )
    np.testing.assert_allclose(result.weights, root * z, rtol=1e-7, atol=1e-10)
    assert bool(result.accepted) and not bool(result.exact)
    assert int(result.status) == int(MeshfreeMetricStatus.RELAXED)
    np.testing.assert_allclose(result.slack, result.moment_residual, atol=1e-14)
    assert float(jnp.max(result.higher_moment_residual)) > 0.5


@pytest.mark.parametrize("sign", ("signed", "nonnegative"), ids=("signed", "nonnegative"))
def test_higher_moment_degree_is_satisfied_or_refused_never_truncated(
    sign: MetricSign,
) -> None:
    def prepare(degree: int) -> MeshfreeExteriorCalculusPlan:
        return MeshfreeExteriorCalculusPlan(
            _STAR,
            1.1,
            10,
            node_volumes=np.ones(5),
            dirichlet=_STAR_DIRICHLET,
            metric_policy=MeshfreeMetricPolicy(
                sign, moment_degree=degree, tolerance=1e-7
            ),
        )

    cubic = prepare(3).prepare()
    assert cubic.metric_system.moment_count == 9
    assert bool(cubic.metric_result.accepted)
    np.testing.assert_allclose(cubic.metric_result.weights, np.ones(4), atol=1e-6)
    quartic = prepare(4).prepare()
    assert quartic.metric_system.moment_count == 14
    result = quartic.metric_result
    # Every x_k^4 moment must vanish while x_k^2 equals 2: no edge metric exists.
    assert not bool(result.accepted)
    assert int(result.status) == int(MeshfreeMetricStatus.INCOMPATIBLE_MOMENTS)
    assert float(jnp.max(jnp.abs(result.normalized_residual))) > 0.1
    assert np.asarray(result.normalized_residual).size == 14
    if sign == "signed":
        design = _dense_design(quartic.metric_system)
        witness = np.asarray(result.incompatibility_witness)
        np.testing.assert_allclose(design.T @ witness, 0.0, atol=1e-10)
        assert abs(float(witness @ np.asarray(quartic.metric_system.rhs))) > 0.5
    refreshed = quartic.refresh(points=quartic.points).metric_result
    assert int(refreshed.status) == int(result.status)


def test_nonnegative_strict_active_derivative_matches_finite_differences() -> None:
    prepared = MeshfreeExteriorCalculusPlan(
        np.asarray([[-1.0], [0.0], [1.0]]),
        1.1,
        3,
        node_volumes=np.ones(3),
        dirichlet=np.asarray([True, False, True]),
        metric_policy=MeshfreeMetricPolicy("nonnegative", tolerance=1e-9),
    ).prepare()
    system = prepared.metric_system
    bound = system.bind_active_set(prepared.metric_result)
    assert bound.active_set is not None
    assert int(bound.active_set.status) == int(
        ConicSensitivityStatus.REGULAR_FIXED_ACTIVE
    )
    np.testing.assert_array_equal(
        np.asarray(bound.active_set.roles),
        [ConicConstraintRole.EQUALITY] * 2 + [ConicConstraintRole.INACTIVE] * 2,
    )
    assert bool(bound.derivative_available)
    direction = jnp.asarray([0.3, 1.0])
    prior_direction = jnp.asarray([0.2, -0.1])
    tangent, evidence = system.strict_active_jvp(
        bound, rhs=direction, prior=prior_direction
    )
    assert bool(evidence.available)
    step = 1e-5

    def weights(scale: float) -> Array:
        return system.solve(
            rhs=system.rhs + scale * direction,
            prior=system.prior + scale * prior_direction,
        ).weights

    central = (weights(step) - weights(-step)) / (2 * step)
    np.testing.assert_allclose(tangent, central, rtol=1e-4, atol=1e-6)
    # The frozen roles refuse a derivative once an edge becomes active.
    zero_edge = system.solve(rhs=jnp.asarray([2.0, 2.0]))
    rebound = system.bind_active_set(
        zero_edge, rhs=jnp.asarray([2.0, 2.0]), fixed_active_set=bound.active_set
    )
    assert rebound.active_set is not None
    assert int(rebound.active_set.status) != int(
        ConicSensitivityStatus.REGULAR_FIXED_ACTIVE
    )
    assert not bool(rebound.derivative_available)


def test_unavailable_rank_certificate_refuses_derivative_but_not_forward_metric() -> None:
    tight = SVDSolvePolicy(
        DenseSVD(),
        rank=RankPolicy(relative_cutoff=1e-12),
        materialization=MaterializationPolicy(max_entries=2),
    )
    prepared = MeshfreeExteriorCalculusPlan(
        np.asarray([[-1.0], [0.0], [1.0]]),
        1.1,
        3,
        node_volumes=np.ones(3),
        dirichlet=np.asarray([True, False, True]),
        metric_policy=MeshfreeMetricPolicy(rank=tight),
    ).prepare()
    result = prepared.metric_result
    assert bool(result.accepted)
    np.testing.assert_allclose(result.weights, [1.0, 1.0], atol=1e-8)
    assert result.rank_certificate.route == "unavailable"
    assert int(result.rank) == -1 and int(result.redundant_constraints) == -1
    assert not bool(result.derivative_available)

    def total(scale: Array) -> Array:
        return jnp.sum(
            prepared.refresh(points=prepared.points * scale).metric_result.weights
        )

    assert bool(jnp.isnan(jax.grad(total)(jnp.asarray(1.0))))


def test_unassessed_coercivity_policy_claims_no_stiffness_property() -> None:
    prepared = MeshfreeExteriorCalculusPlan(
        np.asarray([[-1.0], [0.0], [1.0]]),
        1.1,
        3,
        node_volumes=np.ones(3),
        dirichlet=np.asarray([True, False, True]),
        coercivity_policy=MeshfreeCoercivityPolicy("unassessed"),
    ).prepare()
    diffusion = prepared.diffusion()
    np.testing.assert_allclose(
        diffusion.mv(jnp.asarray([1.0, 0.0, 1.0])), [-1.0, 2.0, -1.0], atol=1e-8
    )
    evidence = diffusion.evidence
    assert not evidence.assessed
    assert not bool(evidence.spd) and not bool(evidence.inertia_available)
    assert not bool(evidence.maximum_principle)
    assert bool(evidence.anchored_components)
    with pytest.raises(ValueError, match="no-shift Cholesky"):
        MeshfreeCoercivityPolicy(
            factorization=SparseFactorizationPolicy("cholesky", diagonal_shift=1e-3)
        )


@pytest.mark.parametrize("count", (12, 24))
def test_signed_exact_metric_on_jittered_irregular_cloud_matches_dense_oracle(
    count: int,
) -> None:
    # Grid with interior nodes jittered by 0.15h, 2.2h radius: dozens of
    # neighbors per node and badly scaled moment rows (graph-transport cloud).
    # B B^T has condition number O(h^-6); the default multilevel Craig solve
    # keeps the iteration count nearly flat, whereas LSMR needs about 1e3 here.
    spacing = 1 / (count - 1)
    grid = (
        np.stack(
            np.meshgrid(np.arange(count), np.arange(count), indexing="ij"), axis=-1
        ).reshape(-1, 2)
        * spacing
    )
    boundary = np.any((grid < 1e-12) | (grid > 1 - 1e-12), axis=1)
    points = np.asarray(grid, dtype=np.float64).copy()
    points[~boundary] += (
        0.15
        * spacing
        * np.random.default_rng(0).uniform(-1, 1, (int((~boundary).sum()), 2))
    )
    prepared = MeshfreeExteriorCalculusPlan(
        points,
        2.2 * spacing,
        40 * count * count,
        node_volumes=np.full(count * count, spacing * spacing),
        dirichlet=boundary,
    ).prepare()
    system, result = prepared.metric_system, prepared.metric_result
    assert result.linear_result is not None
    assert int(result.linear_result.status) == int(LinearSolveStatus.SUCCESS)
    assert int(result.linear_result.diagnostics.iterations) <= 120
    assert bool(result.accepted) and bool(result.exact)
    assert int(result.status) == int(MeshfreeMetricStatus.ACCEPTED)
    design = _dense_design(system) / np.asarray(system.row_scaling)[:, None]
    root = np.sqrt(np.asarray(system.prior))
    rhs = np.asarray(system.rhs / system.row_scaling)
    expected = root * (np.linalg.pinv(design * root, rcond=1e-12) @ rhs)
    np.testing.assert_allclose(result.weights, expected, rtol=1e-6, atol=1e-8)


def test_try_refresh_returns_status_and_rolls_back_unchanged_owner() -> None:
    prepared = MeshfreeExteriorCalculusPlan(
        _STAR,
        1.1,
        10,
        node_volumes=np.ones(5),
        dirichlet=_STAR_DIRICHLET,
    ).prepare()
    margin = prepared.topology_trust_margin
    assert margin > 0
    moved = prepared.points.at[1, 0].add(-0.5 * margin)
    accepted = prepared.try_refresh(points=moved)
    assert bool(accepted.accepted)
    assert int(accepted.status) == int(MeshfreeExteriorRefreshStatus.ACCEPTED)
    np.testing.assert_allclose(
        accepted.exterior.metric_result.weights,
        prepared.refresh(points=moved).metric_result.weights,
    )
    np.testing.assert_allclose(accepted.displacement, 0.5 * margin)
    assert not bool(prepared.within_topology_trust(prepared.points.at[1, 0].add(-margin)))
    far = prepared.try_refresh(points=prepared.points * 1.2)
    assert int(far.status) == int(MeshfreeExteriorRefreshStatus.TOPOLOGY_TRUST_EXCEEDED)
    assert not bool(far.accepted)
    np.testing.assert_array_equal(far.exterior.points, prepared.points)
    # Within trust but y-chart tilt makes the moments incompatible: the raw
    # candidate keeps its refusal evidence and the owner rolls back.
    tilted = prepared.points.at[4, 0].set(0.02)
    refused = prepared.try_refresh(points=tilted)
    assert int(refused.status) == int(MeshfreeExteriorRefreshStatus.METRIC_REFUSED)
    assert int(refused.candidate_metric.status) == int(
        MeshfreeMetricStatus.INCOMPATIBLE_MOMENTS
    )
    np.testing.assert_array_equal(refused.exterior.points, prepared.points)
    np.testing.assert_array_equal(
        refused.exterior.metric_result.weights, prepared.metric_result.weights
    )
    volumes = prepared.try_refresh(node_volumes=np.asarray([1.0, 1.0, 0.0, 1.0, 1.0]))
    assert int(volumes.status) == int(MeshfreeExteriorRefreshStatus.INVALID_VOLUMES)


def test_refused_kkt_certificate_is_not_reported_singular_and_declared_budget_certifies() -> (
    None
):
    # Full 16 x 16 Cartesian lattice: every edge weight is 1 and strictly
    # inactive, but the 1260 x 1260 projection-KKT Jacobian exceeds the default
    # dense materialization limit of the rank policy.
    width = 16
    spacing = 1 / (width - 1)
    grid = np.stack(
        np.meshgrid(np.arange(width), np.arange(width), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    on_edge = (grid == 0) | (grid == width - 1)

    def bound(policy: MeshfreeMetricPolicy) -> MeshfreeMetricResult:
        prepared = MeshfreeExteriorCalculusPlan(
            grid * spacing,
            1.1 * spacing,
            4 * width * width,
            node_volumes=spacing**2 * np.prod(np.where(on_edge, 0.5, 1.0), axis=1),
            dirichlet=np.any(on_edge, axis=1),
            metric_policy=policy,
        ).prepare()
        np.testing.assert_allclose(prepared.metric_result.weights, 1.0, atol=1e-7)
        return prepared.metric_system.bind_active_set(prepared.metric_result)

    refused = bound(MeshfreeMetricPolicy("nonnegative", tolerance=1e-9))
    assert not bool(refused.derivative_available)
    assert refused.kkt_rank_certificate is not None
    assert refused.kkt_rank_certificate.route == "unavailable"
    declared = SVDSolvePolicy(
        DenseSVD(algorithm="qr"),
        rank=RankPolicy(relative_cutoff=1e-12),
        materialization=MaterializationPolicy(max_entries=2_000_000, max_bytes=2**25),
    )
    certified = bound(MeshfreeMetricPolicy("nonnegative", tolerance=1e-9, rank=declared))
    assert certified.active_set is not None
    assert int(certified.active_set.status) == int(
        ConicSensitivityStatus.REGULAR_FIXED_ACTIVE
    )
    assert bool(certified.derivative_available)
    assert certified.kkt_rank_certificate is None


@pytest.mark.parametrize("sign", ("signed", "nonnegative"), ids=("signed", "nonnegative"))
def test_declared_boundary_area_vectors_keep_boundary_edges_with_exact_closure(
    sign: MetricSign,
) -> None:
    side, spacing = 5, 0.25
    grid = np.stack(
        np.meshgrid(np.arange(side), np.arange(side), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    low, high = grid == 0, grid == side - 1
    on_edge = low | high
    boundary = np.any(on_edge, axis=1)
    volumes = spacing**2 * np.prod(np.where(on_edge, 0.5, 1.0), axis=1)
    # Independent finite-volume oracle: outward face length times normal.
    areas = np.zeros((grid.shape[0], 2))
    for axis in range(2):
        length = spacing * np.where(on_edge[:, 1 - axis], 0.5, 1.0)
        areas[:, axis] = np.where(
            low[:, axis], -length, np.where(high[:, axis], length, 0.0)
        )

    def plan(area_vectors: np.ndarray | None) -> MeshfreeExteriorCalculusPlan:
        return MeshfreeExteriorCalculusPlan(
            grid * spacing,
            1.1 * spacing,
            4 * side * side,
            node_volumes=volumes,
            dirichlet=boundary,
            metric_policy=MeshfreeMetricPolicy(
                sign, tolerance=1e-9, boundary_closure="area-vector"
            ),
            boundary_area_vectors=area_vectors,
        )

    closed = plan(areas).prepare()
    result = closed.metric_result
    pairs = np.asarray(closed.pairs)
    tangential = boundary[pairs[:, 0]] & boundary[pairs[:, 1]]
    assert bool(result.accepted) and bool(result.hilbert_admitted)
    # Half faces along the boundary, full faces elsewhere: the FV metric.
    np.testing.assert_allclose(np.asarray(result.weights)[tangential], 0.5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(result.weights)[~tangential], 1.0, atol=1e-7)
    # Deficit identity at every node: sum_e w_e (x_i - x_j) = s_i.
    offsets = (grid * spacing)[pairs[:, 1]] - (grid * spacing)[pairs[:, 0]]
    weights = np.asarray(result.weights)[:, None]
    deficit = np.zeros_like(areas)
    np.add.at(deficit, pairs[:, 0], -weights * offsets)
    np.add.at(deficit, pairs[:, 1], weights * offsets)
    np.testing.assert_allclose(deficit, areas, atol=1e-7)
    np.testing.assert_array_equal(closed.equation_mask, ~boundary)
    open_ = plan(None).prepare()
    assert open_.pairs.shape[0] == pairs.shape[0] - int(tangential.sum())
    with pytest.raises(ValueError, match="Boundary area vectors"):
        plan(areas[:-1])


def test_irregular_boundary_closure_second_moments_admitted_area_vectors_refused() -> (
    None
):
    # Unit-square lattice, interior nodes jittered by 0.15h, radius 2.2h; the
    # nodal volumes are the unjittered tensor control volumes.
    side = 9
    spacing = 1 / (side - 1)
    grid = np.stack(
        np.meshgrid(np.arange(side), np.arange(side), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    low, high = grid == 0, grid == side - 1
    on_edge = low | high
    boundary = np.any(on_edge, axis=1)
    points = grid * spacing
    points[~boundary] += (
        0.15
        * spacing
        * np.random.default_rng(0).uniform(-1, 1, (int((~boundary).sum()), 2))
    )
    volumes = spacing**2 * np.prod(np.where(on_edge, 0.5, 1.0), axis=1)
    areas = np.zeros((grid.shape[0], 2))
    for axis in range(2):
        length = spacing * np.where(on_edge[:, 1 - axis], 0.5, 1.0)
        areas[:, axis] = np.where(
            low[:, axis], -length, np.where(high[:, axis], length, 0.0)
        )

    def prepare(closure: BoundaryClosure) -> PreparedMeshfreeExteriorCalculus:
        return MeshfreeExteriorCalculusPlan(
            points,
            2.2 * spacing,
            12 * side * side,
            node_volumes=volumes,
            dirichlet=boundary,
            boundary_area_vectors=areas,
            metric_policy=MeshfreeMetricPolicy(boundary_closure=closure),
        ).prepare()

    second = prepare("second-moment")
    assert bool(second.metric_result.accepted)
    deficit = np.asarray(second.boundary_normal_measure)
    # Exact discrete Gauss identities of the deficit closure.
    np.testing.assert_allclose(deficit.sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(second.points).T @ deficit, volumes.sum() * np.eye(2), atol=1e-8
    )
    # Both moment families at boundary nodes are overdetermined here: refused
    # with a validated left-null witness, never relaxed.
    both = prepare("area-vector")
    result = both.metric_result
    assert not bool(result.accepted)
    assert int(result.status) == int(MeshfreeMetricStatus.INCOMPATIBLE_MOMENTS)
    design = _dense_design(both.metric_system)
    witness = np.asarray(result.incompatibility_witness)
    margin = abs(float(witness @ np.asarray(both.metric_system.rhs)))
    assert margin > 1e-6
    # Farkas certificate: |<y, A w - b>| >= margin - |A^T y| |w| for every w,
    # so every metric of the accepted closure's scale misses by ~margin.
    leakage = float(np.linalg.norm(design.T @ witness))
    scale = float(np.linalg.norm(np.asarray(second.metric_result.weights)))
    assert leakage * scale < 1e-3 * margin
