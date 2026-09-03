#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.linalg._distributed_line import (
    DistributedLineSolvePlan,
    ExtrudedAxisInvarianceCertificate,
    MultiblockExtrudedReductionPlan,
    PreparedTransverseBatchLineSolve,
    StructuredLineNullspacePolicy,
    StructuredSolveTopologyPlan,
)
from phydrax.linalg._linear_transform import DenseLinearTransform
from phydrax.linalg._transform_line import (
    TransformLineRepresentation,
    TransformLineSolvePlan,
)


jax.config.update("jax_enable_x64", True)


def _poisson_line(size: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    lower = -jnp.ones(size - 1, dtype=jnp.float64)
    diagonal = 2.5 * jnp.ones(size, dtype=jnp.float64)
    upper = -jnp.ones(size - 1, dtype=jnp.float64)
    return lower, diagonal, upper


def _dense_tridiagonal(lower, diagonal, upper):
    return jnp.diag(diagonal) + jnp.diag(lower, -1) + jnp.diag(upper, 1)


def test_line_contiguous_transverse_batch_matches_existing_local_factors():
    transform = DenseLinearTransform(jnp.eye(5), jnp.eye(5))
    lower, diagonal, upper = _poisson_line(6)
    representation = TransformLineRepresentation(
        (transform,),
        1,
        lower,
        diagonal,
        upper,
        jnp.linspace(0.1, 0.5, 5),
    )
    local = TransformLineSolvePlan(representation, tolerance=1.0e-11).prepare()
    topology = StructuredSolveTopologyPlan(
        6,
        3,
        distribution="transverse-batch",
        transverse_line_count=5,
    )
    distributed = PreparedTransverseBatchLineSolve(topology, local)
    rhs = jnp.arange(30, dtype=jnp.float64).reshape(5, 6) / 17.0

    expected = local.solve(rhs)
    actual = distributed.solve(rhs)

    np.testing.assert_allclose(actual.value, expected.value, rtol=1.0e-12, atol=1.0e-12)
    assert topology.partitions.starts == (0, 2, 4)
    assert topology.partitions.sizes == (2, 2, 1)
    assert not distributed.communication.host_gather


def test_uneven_partitioned_thomas_matches_dense_reference():
    lower, diagonal, upper = _poisson_line(11)
    topology = StructuredSolveTopologyPlan(11, 3)
    prepared = DistributedLineSolvePlan(topology, lower, diagonal, upper).prepare()
    rhs = jnp.stack((jnp.linspace(-1.0, 1.0, 11), jnp.cos(jnp.arange(11))))

    result = prepared.solve(rhs)
    expected = jax.vmap(
        lambda value: jnp.linalg.solve(_dense_tridiagonal(lower, diagonal, upper), value)
    )(rhs)

    assert topology.partitions.sizes == (4, 4, 3)
    assert topology.partitions.uneven
    assert bool(result.converged)
    np.testing.assert_allclose(result.value, expected, rtol=1.0e-10, atol=1.0e-11)


def test_spike_matches_reference_and_reports_bounded_interface():
    lower, diagonal, upper = _poisson_line(12)
    topology = StructuredSolveTopologyPlan(
        12,
        4,
        algorithm="spike",
        maximum_reduced_interface_size=8,
    )
    prepared = DistributedLineSolvePlan(topology, lower, diagonal, upper).prepare()
    rhs = jnp.sin(jnp.arange(12, dtype=jnp.float64))

    result = prepared.solve(rhs)
    expected = jnp.linalg.solve(_dense_tridiagonal(lower, diagonal, upper), rhs)

    assert bool(result.converged)
    np.testing.assert_allclose(result.value, expected, rtol=1.0e-10, atol=1.0e-11)
    assert result.preparation.communication.global_rounds == 1
    assert result.preparation.resources.within_budget
    with pytest.raises(ValueError, match="reduced interface"):
        StructuredSolveTopologyPlan(
            12,
            4,
            algorithm="spike",
            maximum_reduced_interface_size=6,
        )


def test_pcr_eligibility_and_balanced_parallel_cyclic_reduction():
    lower, diagonal, upper = _poisson_line(16)
    topology = StructuredSolveTopologyPlan(16, 4, algorithm="pcr")
    prepared = DistributedLineSolvePlan(topology, lower, diagonal, upper).prepare()
    rhs = jnp.linspace(-0.7, 1.3, 16)

    result = prepared.solve(rhs)
    expected = jnp.linalg.solve(_dense_tridiagonal(lower, diagonal, upper), rhs)

    assert bool(result.converged)
    np.testing.assert_allclose(result.value, expected, rtol=1.0e-10, atol=1.0e-11)
    assert result.preparation.communication.neighbor_rounds == 2
    with pytest.raises(ValueError, match="balanced power-of-two"):
        StructuredSolveTopologyPlan(15, 4, algorithm="pcr")
    with pytest.raises(ValueError, match="balanced power-of-two"):
        StructuredSolveTopologyPlan(18, 3, algorithm="pcr")


def test_singular_line_projects_compatibility_and_applies_exact_gauge():
    size = 12
    lower = -jnp.ones(size - 1, dtype=jnp.float64)
    upper = -jnp.ones(size - 1, dtype=jnp.float64)
    diagonal = (2.0 * jnp.ones(size, dtype=jnp.float64)).at[0].set(1.0).at[-1].set(1.0)
    policy = StructuredLineNullspacePolicy(jnp.ones(size), pin_row=3)
    topology = StructuredSolveTopologyPlan(size, 3, tolerance=1.0e-10)
    prepared = DistributedLineSolvePlan(
        topology,
        lower,
        diagonal,
        upper,
        nullspace=policy,
    ).prepare()
    rhs = jnp.arange(size, dtype=jnp.float64) - 2.0

    result = prepared.solve(rhs)

    assert bool(result.converged)
    assert float(result.compatibility_correction_norm) > 0.0
    assert float(result.compatibility_defect) < 1.0e-12
    assert float(result.gauge_defect) < 1.0e-12
    assert float(jnp.mean(result.value)) == pytest.approx(0.0, abs=1.0e-12)
    assert float(result.relative_residual) < 1.0e-10


def test_rhs_jvp_and_vjp_are_the_linear_inverse_actions():
    lower, diagonal, upper = _poisson_line(12)
    prepared = DistributedLineSolvePlan(
        StructuredSolveTopologyPlan(12, 3), lower, diagonal, upper
    ).prepare()
    rhs = jnp.sin(jnp.arange(12, dtype=jnp.float64))
    tangent = jnp.cos(jnp.arange(12, dtype=jnp.float64))

    solve_value = lambda value: prepared.solve(value).candidate
    _, jvp = jax.jvp(solve_value, (rhs,), (tangent,))
    expected_jvp = prepared.solve(tangent).candidate
    _, pullback = jax.vjp(solve_value, rhs)
    cotangent = jnp.linspace(-1.0, 1.0, 12)
    (vjp,) = pullback(cotangent)
    dense = _dense_tridiagonal(lower, diagonal, upper)

    np.testing.assert_allclose(jvp, expected_jvp, rtol=1.0e-10, atol=1.0e-11)
    np.testing.assert_allclose(
        vjp, jnp.linalg.solve(dense.T, cotangent), rtol=1.0e-10, atol=1.0e-11
    )


def test_communication_and_resource_evidence_fail_before_factor_allocation():
    lower, diagonal, upper = _poisson_line(12)
    topology = StructuredSolveTopologyPlan(12, 3, maximum_resource_bytes=64)
    with pytest.raises(ValueError, match="exceed maximum_resource_bytes"):
        DistributedLineSolvePlan(topology, lower, diagonal, upper).prepare()

    admitted = DistributedLineSolvePlan(
        StructuredSolveTopologyPlan(12, 3), lower, diagonal, upper
    ).prepare()
    evidence = admitted.evidence
    assert evidence.resources.total_bytes > evidence.resources.factor_bytes
    assert evidence.communication.deterministic_reduction
    assert not evidence.communication.host_gather
    assert evidence.communication.scalar_values_per_line > 0


def test_multiblock_requires_full_invariance_and_reduces_global_residual_iteratively():
    rejected = ExtrudedAxisInvarianceCertificate(0.0, 0.0, 1.0e-3, 0.0)
    lower, diagonal, upper = _poisson_line(7)
    local_lower = jnp.stack((lower, lower))
    local_diagonal = jnp.stack((diagonal, diagonal + 0.3))
    local_upper = jnp.stack((upper, upper))
    interface = (
        jnp.zeros((1, 2, 7), dtype=jnp.float64)
        .at[0, 0, -1]
        .set(1.0)
        .at[0, 1, 0]
        .set(-1.0)
    )
    mortar = jnp.asarray([[2.0]], dtype=jnp.float64)
    with pytest.raises(
        ValueError, match="certified geometry/metric/coefficient/interface"
    ):
        MultiblockExtrudedReductionPlan(
            local_lower,
            local_diagonal,
            local_upper,
            interface,
            mortar,
            rejected,
        )

    certificate = ExtrudedAxisInvarianceCertificate(0.0, 0.0, 0.0, 0.0)
    prepared = MultiblockExtrudedReductionPlan(
        local_lower,
        local_diagonal,
        local_upper,
        interface,
        mortar,
        certificate,
        tolerance=1.0e-11,
        maximum_iterations=40,
    ).prepare()
    rhs = jnp.stack((jnp.linspace(0.0, 1.0, 7), jnp.linspace(1.0, -0.5, 7)))

    result = prepared.solve(rhs)

    assert bool(result.converged)
    assert int(result.iterations) > 0
    assert float(result.relative_residual) < 1.0e-11
    assert result.local_direct_role == "preconditioner-only"
    assert result.certificate_id == certificate.certificate_id
