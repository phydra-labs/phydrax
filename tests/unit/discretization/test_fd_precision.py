#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _grid(size=16):
    return phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(size, endpoint=False, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))


def _precision():
    return phx.discretization.FDExecutionPrecisionPolicy(
        coefficient_dtype="float32",
        field_dtype="float32",
        accumulation_dtype="float64",
        certification_dtype="float64",
    )


def test_fd_precision_controls_prepared_and_lowered_execution():
    precision = _precision()
    fd = phx.discretization.periodic_finite_difference(
        _grid(),
        accuracy_order=4,
        precision=precision,
    )
    operator = fd.operator("d_x_1")
    lowered = phx.discretization.lower_stencil_operator(operator)
    x = jnp.arange(16, dtype=jnp.float32) / 16.0
    values = jnp.sin(2.0 * jnp.pi * x)

    direct = operator.mv(values)
    compact = lowered.mv(values)

    assert operator.source.dtype == jnp.float32
    assert operator.target.dtype == jnp.float32
    assert operator.weights.dtype == jnp.float32
    assert operator.precision.policy_id == precision.policy_id
    assert direct.dtype == jnp.float32
    assert compact.dtype == jnp.float32
    assert jnp.allclose(compact, direct)
    assert operator.consistency_report.maximum_condition_estimate > 0.0


def test_fd_preflight_uses_the_bound_execution_policy():
    precision = _precision()
    grid = _grid(32)
    fd = phx.discretization.periodic_finite_difference(grid, precision=precision)
    lowered = phx.discretization.lower_stencil_operator(fd.operator("d_x_1"))
    preflight = phx.discretization.FDExecutionPreflightPlan(
        grid,
        field_count=2,
        operators=(lowered,),
        precision=precision,
    )
    estimate = preflight.estimate()

    assert estimate.precision_policy_id == precision.policy_id
    assert estimate.state_bytes == grid.size * 2 * 4
    assert (
        estimate.precision_resource_assumptions_id
        == precision.resource_assumptions.assumptions_id
    )

    with pytest.raises(ValueError, match="share one precision"):
        phx.discretization.FDExecutionPreflightPlan(
            grid,
            field_count=2,
            operators=(lowered,),
            precision=phx.discretization.FDExecutionPrecisionPolicy(),
        )


def test_fd_checkpoint_preserves_execution_dtype_and_policy(tmp_path):
    precision = _precision()
    plan = phx.discretization.FDCheckpointPlan(
        ("grid", "operator"),
        "ssprk3",
        precision=precision,
    )
    fields = {"state": jnp.arange(8, dtype=jnp.float32)}
    path = phx.discretization.write_fd_checkpoint(
        tmp_path / "state.phydrax",
        plan,
        jnp.asarray(0.5, dtype=jnp.float64),
        fields,
    )
    restored = phx.discretization.read_fd_checkpoint(path, plan)

    assert restored.field("state").dtype == jnp.float32
    np.testing.assert_array_equal(restored.field("state"), fields["state"])

    with pytest.raises(TypeError, match="expected float32"):
        phx.discretization.write_fd_checkpoint(
            tmp_path / "bad.phydrax",
            plan,
            0.5,
            {"state": jnp.arange(8, dtype=jnp.float64)},
        )


def test_distributed_fd_payload_requires_field_precision():
    precision = _precision()
    fd = phx.discretization.periodic_finite_difference(_grid(), precision=precision)
    partition = phx.discretization.DistributedStencilPartition(
        (16,),
        0,
        fd.halo_plan,
        periodic=True,
        precision=precision,
    )

    sharded = partition.shard(jnp.arange(16, dtype=jnp.float32))
    assert sharded.dtype == jnp.float32
    with pytest.raises(TypeError, match="payload dtype"):
        partition.shard(jnp.arange(16, dtype=jnp.float64))


def test_fd_amr_reflux_accumulates_high_and_returns_field_precision():
    precision = _precision()
    plan = phx.discretization.ConservativeAMRSubcyclingPlan(
        2,
        precision=precision,
    )
    result = plan.advance(
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray([10.0, 20.0], dtype=jnp.float32),
        jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
        jnp.asarray(0.2, dtype=jnp.float32),
        lambda time, state, dt, args: state,
        lambda time, state, dt, args: state,
        lambda state, args: jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        lambda state, args: jnp.asarray([2.0, 0.0], dtype=jnp.float32),
        lambda flux: flux,
        jnp.asarray([True, False]),
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
    )

    assert result.flux_register.coarse_flux.dtype == jnp.float64
    assert result.flux_register.fine_flux.dtype == jnp.float64
    assert result.coarse_state.dtype == jnp.float32
    assert result.precision_evidence.evidence_id == precision.evidence().evidence_id


def test_conservative_multigrid_uses_field_and_certification_precision():
    precision = _precision()
    diffusion_precision = phx.discretization.FiniteVolumePrecisionPolicy(
        "float32",
        reconstruction_dtype="float32",
        flux_dtype="float32",
        reduction_dtype="float64",
        output_dtype="float32",
        checkpoint_dtype="float32",
    )
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(16),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    diffusion = phx.discretization.ConservativeDiffusionPlan(
        grid,
        boundaries={"x": ("dirichlet", "dirichlet")},
        precision=diffusion_precision,
    ).prepare(jnp.asarray(1.0, dtype=jnp.float64))
    multigrid = phx.discretization.StructuredMultigridPlan(
        diffusion,
        minimum_coarse_points=4,
        precision=precision,
    ).prepare()
    rhs = jnp.ones(grid.shape, dtype=jnp.float32)
    result = multigrid.solve(rhs, cycles=2, tolerance=1e-5)

    assert diffusion.source.dtype == jnp.float32
    assert diffusion.coefficient.dtype == jnp.float32
    assert all(
        transfer.restriction_matrices[0].dtype == jnp.float32
        for transfer in multigrid.transfers
    )
    assert result.value.dtype == jnp.float32
    assert result.residual_norms.dtype == jnp.float64
    assert result.precision_evidence.evidence_id == precision.evidence().evidence_id


def test_fd_adjoint_identity_reduces_in_certification_precision():
    precision = _precision()
    action = phx.discretization.FDActionAdjointPlan(
        lambda value: 3.0 * value,
        precision=precision,
    )
    report = action.identity_report(
        (jnp.asarray([1.0, 2.0], dtype=jnp.float32),),
        0,
        jnp.asarray([0.5, -1.0], dtype=jnp.float32),
        jnp.asarray([2.0, 4.0], dtype=jnp.float32),
        tolerance=1e-6,
    )

    assert report.passed
    assert action.precision.policy_id == precision.policy_id
