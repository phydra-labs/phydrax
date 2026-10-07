"""Distributed spectral construction and execution stay typed at the public facade."""

from typing import assert_type

import jax.numpy as jnp
from jax import Array

import phydrax as phx
from phydrax.discretization import (
    DistributedSpectralExecutionPlan,
    DistributedSpectralPreparationReport,
    SpectralExecutionResult,
    SpectralGlobalDiagnostics,
    SpectralMeshTopology,
    SpectralPrecisionPolicy,
    SpectralResourceReport,
)


def distributed_spectral_facade(
    topology: SpectralMeshTopology,
    state: Array,
    vector_state: Array,
) -> None:
    precision = phx.discretization.SpectralPrecisionPolicy(
        jnp.float32,
        coefficient_dtype=jnp.complex64,
        transform_dtype=jnp.complex64,
        reduction_dtype=jnp.float32,
    )
    assert_type(precision, SpectralPrecisionPolicy)

    plan = phx.discretization.DistributedSpectralExecutionPlan(
        topology,
        (8, 8, 8),
        owner_id="typing-distributed-spectral",
        precision=precision,
        admitted_payload_shapes=((), (3,)),
        state_shape=(),
        padded_shape=(12, 12, 12),
        maximum_bytes=64 * 1024**2,
    )
    assert_type(plan, DistributedSpectralExecutionPlan)
    prepared = plan.prepare()
    assert_type(prepared, DistributedSpectralExecutionPlan)
    assert_type(prepared.owner_id, str)
    assert_type(prepared.precision, SpectralPrecisionPolicy)
    assert_type(prepared.admitted_payload_shapes, tuple[tuple[int, ...], ...])
    assert_type(prepared.numerical_id, str)
    assert_type(prepared.execution_id, str)
    assert_type(prepared.plan_id, str)
    assert_type(prepared.report, DistributedSpectralPreparationReport)
    assert_type(prepared.report.resource, SpectralResourceReport)
    assert_type(prepared.report.forward_sequence_id, str)
    assert_type(prepared.report.inverse_sequence_id, str)
    assert_type(prepared.report.padded_forward_sequence_id, str)
    assert_type(prepared.report.padded_inverse_sequence_id, str)
    assert_type(prepared.report.resource.canonical_storage_bytes, int)
    assert_type(prepared.report.resource.padded_storage_bytes, int)
    assert_type(prepared.report.resource.transform_workspace_bytes, int)
    assert_type(prepared.report.resource.collective_payload_bytes, int)
    assert_type(prepared.report.resource.peak_live_bytes, int)

    assert_type(prepared.place(state, representation="physical"), Array)
    assert_type(prepared.place_batched(vector_state, representation="physical"), Array)
    modal = prepared.to_modal(state)
    assert_type(modal, Array)
    assert_type(prepared.to_physical(modal), Array)
    batched_modal = prepared.to_modal_batched(vector_state)
    assert_type(batched_modal, Array)
    assert_type(prepared.to_physical_batched(batched_modal), Array)
    assert_type(prepared.pad_modal(modal), Array)
    assert_type(prepared.unpad_modal(prepared.pad_modal(modal)), Array)
    assert_type(prepared.modal_derivative(modal, 0), Array)

    result = prepared.execute_transform(
        state,
        direction="physical_to_modal",
    )
    assert_type(result, SpectralExecutionResult)
    assert_type(result.value, Array)
    assert_type(result.layout_id, str)
    assert_type(result.plan_id, str)

    diagnostics = prepared.diagnostics(modal)
    assert_type(diagnostics, SpectralGlobalDiagnostics)
    assert_type(diagnostics.total, Array)
    assert_type(diagnostics.maximum_absolute, Array)
    assert_type(diagnostics.l2_norm, Array)
    assert_type(diagnostics.finite, Array)
    assert_type(prepared.global_inner_product(modal, modal), Array)
    assert_type(prepared.global_all(jnp.isfinite(modal)), Array)
