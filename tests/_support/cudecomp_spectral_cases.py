# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Real cuDecomp cases consumed by the canonical supervised MPI worker."""

from __future__ import annotations

from dataclasses import asdict
from functools import partial
from typing import assert_never, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec

from phydrax._execution_collectives import CollectiveLease, prepare_collective_program
from phydrax.discretization.spectral import (
    DistributedSpectralExecutionPlan,
    SpectralMeshTopology,
    SpectralPrecisionPolicy,
)
from phydrax.typing import parse


type Representation = Literal["physical", "modal"]
type CudecompPrecision = Literal["float32", "float64"]
type CudecompFailureCase = Literal[
    "resource-refusal",
    "unsupported-pencil",
    "external-scratch-refusal",
    "inactive-cached-call",
    "asymmetric-activity",
    "asymmetric-zero-cotangent",
]


def _place(
    plan: DistributedSpectralExecutionPlan,
    values: np.ndarray,
    representation: Representation,
    *,
    padded: bool = False,
) -> Array:
    if representation == "physical":
        layout = plan.padded_physical_layout if padded else plan.physical_layout
    else:
        layout = plan.padded_modal_layout if padded else plan.modal_layout
    partition = (*layout.partition[:3], *((None,) * (values.ndim - 3)))
    sharding = NamedSharding(plan.topology.mesh, PartitionSpec(*partition))
    return jax.make_array_from_callback(
        values.shape, sharding, lambda index: values[index]
    )


def _check(
    observed: Array,
    expected: np.ndarray,
    *,
    tolerance: float,
    exact: bool = False,
) -> float:
    assert observed.shape == expected.shape
    assert observed.dtype == expected.dtype
    assert len(observed.addressable_shards) == 1
    error = 0.0
    for shard in observed.addressable_shards:
        values = np.asarray(shard.data)
        reference = expected[shard.index]
        assert np.all(np.isfinite(values))
        if exact:
            np.testing.assert_array_equal(values, reference)
        else:
            np.testing.assert_allclose(values, reference, rtol=tolerance, atol=tolerance)
        error = max(error, float(np.max(np.abs(values - reference), initial=0.0)))
    return error


@partial(jax.jit, static_argnames=("inverse", "padded"))
def _redistribute(
    plan: DistributedSpectralExecutionPlan,
    values: Array,
    *,
    inverse: bool,
    padded: bool,
) -> Array:
    execution = plan._cudecomp_execution
    if execution is None:
        raise ValueError(
            "Exact permutation qualification requires the selected cuDecomp provider."
        )
    physical = plan.padded_physical_layout if padded else plan.physical_layout
    modal = plan.padded_modal_layout if padded else plan.modal_layout
    source, target = (modal, physical) if inverse else (physical, modal)

    def local(value: Array) -> Array:
        return execution.apply_local(value, inverse=inverse, padded=padded)

    return jax.shard_map(
        local,
        mesh=plan.topology.mesh,
        in_specs=source.partition_spec,
        out_specs=target.partition_spec,
        check_vma=True,
    )(values)


@partial(jax.jit, static_argnames=("padded",))
def _forward(
    plan: DistributedSpectralExecutionPlan, values: Array, *, padded: bool
) -> Array:
    return plan.to_modal_batched(values, padded=padded)


@partial(jax.jit, static_argnames=("padded",))
def _inverse(
    plan: DistributedSpectralExecutionPlan, values: Array, *, padded: bool
) -> Array:
    return plan.to_physical_batched(values, padded=padded)


@partial(jax.jit, static_argnames=("padded",))
def _jvp(
    plan: DistributedSpectralExecutionPlan,
    values: Array,
    direction: Array,
    *,
    padded: bool,
) -> tuple[Array, Array]:
    return jax.jvp(
        lambda value: plan.to_modal_batched(value, padded=padded), (values,), (direction,)
    )


@partial(jax.jit, static_argnames=("padded",))
def _hilbert_adjoint(
    plan: DistributedSpectralExecutionPlan,
    values: Array,
    cotangent: Array,
    *,
    padded: bool,
) -> Array:
    _, pullback = jax.vjp(
        lambda value: plan.to_modal_batched(value, padded=padded), values
    )
    return jnp.conj(pullback(jnp.conj(cotangent))[0])


@jax.jit
def _discarded_tangent(
    plan: DistributedSpectralExecutionPlan, values: Array, direction: Array
) -> Array:
    return jax.jvp(plan.to_modal_batched, (values,), (direction,))[0]


@jax.jit
def _discarded_output(plan: DistributedSpectralExecutionPlan, values: Array) -> Array:
    plan.to_modal_batched(values)
    return values + jnp.asarray(1, dtype=values.dtype)


@jax.jit
def _hessian_vector(
    plan: DistributedSpectralExecutionPlan, values: Array, direction: Array
) -> Array:
    def energy(value: Array) -> Array:
        modal = plan.to_modal_batched(value)
        return jnp.real(jnp.vdot(modal, modal))

    return jax.jvp(jax.grad(energy), (values,), (direction,))[1]


@jax.jit
def _field_only(plan: DistributedSpectralExecutionPlan, values: Array) -> Array:
    return jnp.real(plan.to_physical_batched(plan.to_modal_batched(values)[..., :6]))


@jax.jit
def _field_chain(plan: DistributedSpectralExecutionPlan, values: Array) -> Array:
    field = jnp.real(plan.to_physical_batched(plan.to_modal_batched(values)[..., :6]))
    return jnp.real(plan.to_physical_batched(plan.to_modal_batched(field)[..., :2]))


def run_cudecomp_cases(
    lease: CollectiveLease,
    topology: SpectralMeshTopology,
    dtype_name: str,
) -> tuple[dict[str, object], DistributedSpectralExecutionPlan]:
    """Exercise real provider execution; the caller owns bootstrap and lease teardown."""
    dtype_name = parse(dtype_name, CudecompPrecision, "dtype_name")
    dtype = np.dtype(np.complex64 if dtype_name == "float32" else np.complex128)
    precision = SpectralPrecisionPolicy(np.dtype(dtype_name))
    ranks = topology.device_count
    shape = (4 * ranks, 4 * ranks, 6)
    padded_shape = (6 * ranks, 6 * ranks, 10)
    plan = DistributedSpectralExecutionPlan(
        topology,
        shape,
        padded_shape=padded_shape,
        owner_id="cudecomp-spectral-qualification",
        precision=precision,
        state_shape=(),
        admitted_payload_shapes=((), (2,), (1, 11), (1, 6), (1, 2)),
        transpose_provider="cudecomp",
        collective_binding=lease.binding,
        maximum_external_scratch_bytes=64 * 1024**2,
    )
    tolerance = 8e-5 if dtype_name == "float32" else 1e-11
    errors: dict[str, float] = {}
    for padded, grid in ((False, shape), (True, padded_shape)):
        for payload in ((), (2,)):
            full_shape = (*grid, *payload)
            index = np.arange(np.prod(full_shape), dtype=np.float64).reshape(full_shape)
            exact = (index + 1j * (np.prod(full_shape) - 1 - index)).astype(dtype)
            values = _place(plan, exact, "physical", padded=padded)
            redistribute = prepare_collective_program(
                partial(_redistribute, inverse=False, padded=padded),
                plan,
                values,
                lease_key=lease.spec,
            )
            transposed = lease.invoke(redistribute, plan, values)
            errors[f"permutation.{padded}.{payload}"] = _check(
                transposed, exact, tolerance=0.0, exact=True
            )
            reverse = prepare_collective_program(
                partial(_redistribute, inverse=True, padded=padded),
                plan,
                transposed,
                lease_key=lease.spec,
            )
            restored = lease.invoke(reverse, plan, transposed)
            _check(restored, exact, tolerance=0.0, exact=True)
            host = (np.sin(index * 0.13) + 1j * np.cos(index * 0.17)).astype(dtype)
            expected_modal = np.fft.fftn(host, axes=(0, 1, 2), norm="ortho").astype(dtype)
            template = _place(plan, host, "physical", padded=padded)
            modal_template = _place(plan, expected_modal, "modal", padded=padded)
            forward = prepare_collective_program(
                partial(_forward, padded=padded), plan, template, lease_key=lease.spec
            )
            inverse_program = prepare_collective_program(
                partial(_inverse, padded=padded),
                plan,
                modal_template,
                lease_key=lease.spec,
            )
            for generation in (0, 1):
                changed = (host * (1.0 + generation * 0.25)).astype(dtype)
                placed = _place(plan, changed, "physical", padded=padded)
                expected = (expected_modal * (1.0 + generation * 0.25)).astype(dtype)
                modal = lease.invoke(forward, plan, placed)
                key = f"fft.{padded}.{payload}.{generation}"
                errors[key] = _check(modal, expected, tolerance=tolerance)
                inverse = lease.invoke(inverse_program, plan, modal)
                _check(inverse, changed, tolerance=tolerance)
            direction_host = (host * np.asarray(0.3 + 0.2j, dtype=dtype)).astype(dtype)
            direction = _place(plan, direction_host, "physical", padded=padded)
            jvp_program = prepare_collective_program(
                partial(_jvp, padded=padded),
                plan,
                placed,
                direction,
                lease_key=lease.spec,
            )
            primal, tangent = lease.invoke(jvp_program, plan, placed, direction)
            _check(primal, expected, tolerance=tolerance)
            tangent_reference = np.fft.fftn(
                direction_host, axes=(0, 1, 2), norm="ortho"
            ).astype(dtype)
            errors[f"jvp.{padded}.{payload}"] = _check(
                tangent, tangent_reference, tolerance=tolerance
            )
            probe = _place(plan, direction_host, "modal", padded=padded)
            adjoint_program = prepare_collective_program(
                partial(_hilbert_adjoint, padded=padded),
                plan,
                placed,
                probe,
                lease_key=lease.spec,
            )
            adjoint = lease.invoke(adjoint_program, plan, placed, probe)
            adjoint_reference = np.fft.ifftn(
                direction_host, axes=(0, 1, 2), norm="ortho"
            ).astype(dtype)
            errors[f"adjoint.{padded}.{payload}"] = _check(
                adjoint, adjoint_reference, tolerance=tolerance
            )
            zero = _place(plan, np.zeros(full_shape, dtype=dtype), "modal", padded=padded)
            zero_adjoint = lease.invoke(adjoint_program, plan, placed, zero)
            _check(
                zero_adjoint, np.zeros(full_shape, dtype=dtype), tolerance=0.0, exact=True
            )
            if not padded:
                discarded_program = prepare_collective_program(
                    _discarded_tangent, plan, placed, direction, lease_key=lease.spec
                )
                assert len(discarded_program.schedule) == 2
                discarded = lease.invoke(discarded_program, plan, placed, direction)
                _check(discarded, expected, tolerance=tolerance)
                unused_program = prepare_collective_program(
                    _discarded_output, plan, placed, lease_key=lease.spec
                )
                assert len(unused_program.schedule) == 1
                unused = lease.invoke(unused_program, plan, placed)
                _check(unused, (changed + 1).astype(dtype), tolerance=0.0, exact=True)
                hessian_program = prepare_collective_program(
                    _hessian_vector, plan, placed, direction, lease_key=lease.spec
                )
                hessian_vector = lease.invoke(hessian_program, plan, placed, direction)
                errors[f"hvp.{payload}"] = _check(
                    hessian_vector,
                    (2 * np.conj(direction_host)).astype(dtype),
                    tolerance=tolerance,
                )
    host_payload = (
        np.random.default_rng(11)
        .normal(size=(*shape, 1, 11))
        .astype(precision.physical_dtype)
    )
    first = _place(plan, host_payload.astype(dtype), "physical")
    field_program = prepare_collective_program(
        _field_only, plan, first, lease_key=lease.spec
    )
    fields = lease.invoke(field_program, plan, first)
    _check(fields, host_payload[..., :6], tolerance=tolerance)
    chain_program = prepare_collective_program(
        _field_chain, plan, first, lease_key=lease.spec
    )
    for offset in (0.0, 0.25):
        changed = (host_payload + offset).astype(precision.physical_dtype)
        placed_payload = _place(plan, changed.astype(dtype), "physical")
        chained = lease.invoke(chain_program, plan, placed_payload)
        errors[f"payload-history.{offset}"] = _check(
            chained, changed[..., :2], tolerance=tolerance
        )
    evidence = plan.report.transpose_evidence
    if evidence is None:
        raise RuntimeError(
            "Selected cuDecomp execution did not retain preparation evidence."
        )
    assert evidence.external_device_scratch_bytes_per_rank > 0
    assert evidence.external_device_scratch_bytes_per_rank <= 64 * 1024**2
    assert plan.report.resource.transpose_packing_bytes > 0
    assert plan.report.resource.external_host_bytes is None
    return {
        "provider": plan.transpose_provider,
        "dtype": dtype_name,
        "coefficient_dtype": dtype.name,
        "numerical_id": plan.numerical_id,
        "execution_id": plan.execution_id,
        "preparation": asdict(evidence),
        "errors": errors,
        "tolerance": tolerance,
        "exact_permutation": True,
        "zero_cotangent": True,
        "discarded_tangent": True,
        "higher_order_linear_ad": True,
        "composed_payload_history": True,
    }, plan


@jax.jit
def _asymmetric_activity(plan: DistributedSpectralExecutionPlan, values: Array) -> Array:
    return values if jax.process_index() == 0 else plan.to_modal_batched(values)


@jax.jit
def _asymmetric_zero_cotangent(
    plan: DistributedSpectralExecutionPlan, values: Array
) -> Array:
    modal, pullback = jax.vjp(plan.to_modal_batched, values)
    if jax.process_index() == 0:
        return jnp.zeros_like(values)
    return pullback(jnp.ones_like(modal))[0]


def _failure_plan(
    lease: CollectiveLease,
    topology: SpectralMeshTopology,
    dtype_name: str,
    *,
    maximum_bytes: int = 2 * 1024**3,
    maximum_external_scratch_bytes: int = 64 * 1024**2,
) -> DistributedSpectralExecutionPlan:
    return DistributedSpectralExecutionPlan(
        topology,
        (4 * topology.device_count, 4 * topology.device_count, 6),
        owner_id="cudecomp-spectral-failure-qualification",
        precision=SpectralPrecisionPolicy(np.dtype(dtype_name)),
        admitted_payload_shapes=((), (2,)),
        transpose_provider="cudecomp",
        collective_binding=lease.binding,
        maximum_bytes=maximum_bytes,
        maximum_external_scratch_bytes=maximum_external_scratch_bytes,
    )


def run_cudecomp_failure_case(
    lease: CollectiveLease,
    topology: SpectralMeshTopology,
    dtype_name: str,
    case: str,
) -> dict[str, object]:
    """Return only healthy pre-entry refusals; unsafe entry failures terminate the worker."""
    from phydrax.discretization.spectral import SpectralResourceError

    selected = parse(case, CudecompFailureCase, "case")
    match selected:
        case "resource-refusal":
            try:
                _failure_plan(lease, topology, dtype_name, maximum_bytes=1)
            except SpectralResourceError as error:
                return {
                    "case": selected,
                    "refused": True,
                    "peak_live_bytes": error.report.peak_live_bytes,
                    "maximum_bytes": error.report.maximum_bytes,
                }
            raise AssertionError("An insufficient logical resource budget was admitted.")
        case "unsupported-pencil":
            pencil = SpectralMeshTopology(
                (1, topology.device_count),
                devices=tuple(topology.mesh.devices.flat),
                axis_names=("px", "py"),
            )
            try:
                DistributedSpectralExecutionPlan(
                    pencil,
                    (
                        4 * topology.device_count,
                        4 * topology.device_count,
                        topology.device_count,
                    ),
                    owner_id="unsupported-cudecomp-pencil",
                    precision=SpectralPrecisionPolicy(np.dtype(dtype_name)),
                    admitted_payload_shapes=((),),
                    schedule="pencil",
                    transpose_provider="cudecomp",
                    collective_binding=lease.binding,
                )
            except ValueError as error:
                if "direct three-dimensional slabs only" not in str(error):
                    raise
                return {"case": selected, "refused": True, "reason": str(error)}
            raise AssertionError("An unsupported cuDecomp pencil mapping was admitted.")
        case "external-scratch-refusal":
            _failure_plan(lease, topology, dtype_name, maximum_external_scratch_bytes=1)
            raise AssertionError("An insufficient external scratch budget was admitted.")
        case "inactive-cached-call" | "asymmetric-activity" | "asymmetric-zero-cotangent":
            plan = _failure_plan(lease, topology, dtype_name)
            dtype = np.dtype(plan.precision.coefficient_dtype)
            host = (
                np.arange(np.prod(plan.spatial_shape), dtype=np.float64)
                .reshape(plan.spatial_shape)
                .astype(dtype)
            )
            values = _place(plan, host, "physical")
            match selected:
                case "inactive-cached-call":
                    program = prepare_collective_program(
                        partial(_forward, padded=False),
                        plan,
                        values,
                        lease_key=lease.spec,
                    )
                    lease.invoke(program, plan, values)
                    jax.block_until_ready(program.executable(plan, values))
                case "asymmetric-activity":
                    program = prepare_collective_program(
                        _asymmetric_activity, plan, values, lease_key=lease.spec
                    )
                    lease.invoke(program, plan, values)
                case "asymmetric-zero-cotangent":
                    program = prepare_collective_program(
                        _asymmetric_zero_cotangent, plan, values, lease_key=lease.spec
                    )
                    lease.invoke(program, plan, values)
                case _:
                    assert_never(selected)
            raise AssertionError(f"Unsafe cuDecomp entry {selected!r} was admitted.")
        case _:
            assert_never(selected)
