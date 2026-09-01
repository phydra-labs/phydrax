import jax.numpy as jnp
import pytest

import phydrax as phx


_VERTICES = jnp.asarray(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
_FACES = jnp.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=jnp.int32)


def _prepared():
    policy = phx.operators.LaplaceSingleLayerDP0GalerkinPolicy3D(
        singular_order=3,
        near_ratio=1.0,
        absolute_tolerance=1.0e-3,
        relative_tolerance=1.0e-3,
        target_block_size=3,
        source_block_size=2,
        dense_oracle=phx.linalg.MaterializationPolicy(
            max_entries=100,
            max_bytes=4096,
        ),
    )
    return phx.operators.prepare_laplace_single_layer_dp0_3d(
        phx.geometry.MeshRegion(_VERTICES, _FACES), policy=policy
    )


def test_pair_coverage_gram_symmetry_and_blocked_dense_parity():
    prepared = _prepared()
    report = prepared.assembly_report
    dense = prepared.dense_oracle.matrix
    weak_dense = prepared.face_areas[:, None] * dense

    assert report.pair_counts == (4, 12, 0, 0, 0)
    assert sum(report.pair_counts) == prepared.face_count**2
    assert bool(report.accuracy_supported)
    assert jnp.allclose(weak_dense, weak_dense.T, rtol=2.0e-3, atol=2.0e-4)

    x = jnp.asarray([0.3, -0.2, 0.7, 0.5])
    y = jnp.asarray([-0.4, 0.1, 0.2, 0.8])
    blocked = prepared.strong_operator.mv(x)
    transposed = prepared.strong_operator.transpose_mv(y)
    assert jnp.allclose(blocked, dense @ x, rtol=1.0e-10, atol=1.0e-10)
    assert jnp.allclose(transposed, dense.T @ y, rtol=1.0e-10, atol=1.0e-10)
    assert jnp.allclose(y @ blocked, x @ transposed, rtol=1.0e-10, atol=1.0e-10)

    diagonal = phx.linalg.assemble_diagonal(prepared.strong_operator)
    assert jnp.allclose(diagonal, jnp.diag(dense))


def test_production_operator_refuses_materialization_and_reports_exact_cost():
    prepared = _prepared()
    with pytest.raises(phx.linalg.LinearCapabilityError, match="does not support"):
        phx.linalg.materialize(
            prepared.strong_operator,
            phx.linalg.MaterializationPolicy(max_entries=100, max_bytes=4096),
        )

    estimate = phx.linalg.estimate_operator_action_cost(prepared.strong_operator)
    assert estimate.exact
    assert estimate.operation_class == "strong-blocked-surface-galerkin-action"
    assert estimate.apply_workspace_bytes_per_rhs > 0
    assert estimate.apply_workspace_bytes_per_rhs == (
        prepared.assembly_report.action_workspace_bytes_per_rhs
    )


def test_dp0_potential_reconstruction_does_not_apply_area_or_permittivity():
    prepared = _prepared()
    coefficients = jnp.asarray([2.0, 0.0, -1.0, 0.5])
    potential = prepared.potential(coefficients)
    q = prepared.panelization.nodes_per_panel

    assert jnp.array_equal(potential.density, jnp.repeat(coefficients, q))
    values, report = phx.operators.evaluate_laplace_layer_3d(
        potential,
        jnp.asarray([[3.0, 3.0, 3.0]]),
        target_side="exterior",
    )
    assert bool(report.pde_membership_valid)
    assert jnp.all(jnp.isfinite(values))


def test_exception_capacity_fails_before_operator_creation():
    policy = phx.operators.LaplaceSingleLayerDP0GalerkinPolicy3D(
        singular_order=3,
        near_ratio=1.0,
        absolute_tolerance=1.0e-3,
        relative_tolerance=1.0e-3,
        max_exception_pairs=3,
    )
    with pytest.raises(ValueError, match="max_exception_pairs"):
        phx.operators.prepare_laplace_single_layer_dp0_3d(
            phx.geometry.MeshRegion(_VERTICES, _FACES), policy=policy
        )
