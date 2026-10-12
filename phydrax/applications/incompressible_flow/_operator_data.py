#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import assert_never, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._validation import integer, positive_integer
from ...discretization.spectral import (
    PeriodicLerayProjector,
    RetainedBandRestrictionPlan,
    TensorSpectralDiscretization,
)
from ...execution import compiled_memory_estimate, MemoryBudgetError
from ...nn.operator.data import (
    FunctionSamples,
    OperatorAxis,
    OperatorBatch,
    OperatorCaseProvenance,
    OperatorTargetBatch,
)
from ...nn.operator.training import OperatorDataset
from ...typing import checked, parse
from ._statistical_evidence import PeriodicFlowEnsembleResult


PeriodicFlowDatasetField: TypeAlias = Literal["velocity", "vorticity"]


def _saved_index(value: int, count: int, name: str) -> int:
    position = integer(value, name)
    if position < 0:
        position += count
    if not 0 <= position < count:
        raise ValueError(f"{name} is outside the saved time grid.")
    return position


def _field_values(
    coefficients: Array,
    projector: PeriodicLerayProjector,
    field: PeriodicFlowDatasetField,
) -> Array:
    match field:
        case "velocity":
            modal = coefficients
        case "vorticity":
            modal = projector.vorticity_from_velocity(coefficients)
            if projector.spatial_dimension == 2:
                modal = modal[..., None]
        case _:
            assert_never(field)
    return projector.discretization.reconstruct(modal)


def _dataset_values(
    states: Array,
    valid: Array,
    source: TensorSpectralDiscretization,
    output: TensorSpectralDiscretization,
    selected: tuple[int, ...],
    field_: PeriodicFlowDatasetField,
    chunk: int,
    limit: int,
) -> tuple[Array, Array]:
    """Admit retained and transient buffers, then execute bounded conversion."""
    members, dimension = states.shape[0], len(source.axes)
    targets = selected[1:]
    # Velocity channel count is a conservative admission bound for every field.
    channels = dimension
    cases = members * len(targets)
    itemsize = np.dtype(output.plan.precision.physical_dtype).itemsize
    # Retained host/device dataset buffers plus the largest selected member chunk.
    retained = (
        2 * cases * prod(output.physical_shape) * channels * itemsize
        + cases * (8 + 8 + 1)
        + 16 * sum(output.physical_shape)
        + 8
    )
    selected_saves = len(selected)
    lanes = min(chunk, members)
    transient = (
        8
        * lanes
        * selected_saves
        * (prod(source.modal_shape) * dimension + prod(output.modal_shape) * dimension)
        * states.dtype.itemsize
        + 4 * lanes * selected_saves * prod(output.physical_shape) * channels * itemsize
    )
    required = 2 * retained + transient
    if required > limit:
        raise MemoryBudgetError("maximum_dataset_bytes", required, limit)
    if not np.asarray(valid[:, np.asarray(selected)]).all():
        raise ValueError("Selected dataset saves must all be valid.")
    restriction = (
        None
        if output.prepared_id == source.prepared_id
        else RetainedBandRestrictionPlan(source, output).prepare()
    )
    projector = PeriodicLerayProjector(output)

    @jax.jit
    def evaluate(coefficients: Array) -> Array:
        if restriction is not None:
            coefficients = jax.vmap(jax.vmap(restriction))(coefficients)
        return jax.vmap(jax.vmap(lambda c: _field_values(c, projector, field_)))(
            coefficients
        )

    observation = jax.eval_shape(
        evaluate,
        jax.ShapeDtypeStruct(
            (lanes, selected_saves, *source.modal_shape, dimension), states.dtype
        ),
    )
    channels = observation.shape[-1]
    executable = evaluate.lower(
        jax.ShapeDtypeStruct(
            (lanes, selected_saves, *source.modal_shape, dimension), states.dtype
        )
    ).compile()
    memory = compiled_memory_estimate(executable)
    if memory is None:
        raise RuntimeError(
            "Periodic dataset conversion requires compiler memory analysis."
        )
    compiler_bytes = max(
        memory.argument_bytes
        + memory.output_bytes
        - memory.alias_bytes
        + memory.temporary_bytes
        + memory.generated_code_bytes,
        0 if memory.peak_bytes is None else memory.peak_bytes,
    ) + (
        memory.host_argument_bytes
        + memory.host_output_bytes
        - memory.host_alias_bytes
        + memory.host_temporary_bytes
        + memory.host_generated_code_bytes
    )
    required = 2 * retained + max(transient, compiler_bytes)
    if required > limit:
        raise MemoryBudgetError("maximum_dataset_bytes", required, limit)

    shape = (cases, *output.physical_shape, channels)
    input_values = np.empty(shape, dtype=output.plan.precision.physical_dtype)
    output_values = np.empty_like(input_values)
    for start in range(0, members, chunk):
        end = min(start + chunk, members)
        coefficients = states[start:end, np.asarray(selected)]
        if not np.isfinite(np.asarray(coefficients)).all():
            raise ValueError("Selected dataset coefficients must be finite.")
        if end - start < lanes:
            coefficients = jnp.pad(
                coefficients,
                ((0, lanes - (end - start)), *((0, 0),) * (coefficients.ndim - 1)),
            )
        values = np.asarray(executable(coefficients))[: end - start]
        case_start, case_end = start * len(targets), end * len(targets)
        input_values[case_start:case_end] = np.repeat(values[:, 0], len(targets), axis=0)
        output_values[case_start:case_end] = values[:, 1:].reshape(
            (case_end - case_start, *output.physical_shape, channels)
        )
    inputs, outputs = jnp.asarray(input_values), jnp.asarray(output_values)
    del input_values, output_values, coefficients, values
    return inputs, outputs


@checked
def periodic_flow_operator_dataset(
    result: PeriodicFlowEnsembleResult,
    /,
    *,
    input_time_index: int = 0,
    target_time_indices: Sequence[int],
    field: PeriodicFlowDatasetField = "velocity",
    output_discretization: TensorSpectralDiscretization | None = None,
    member_chunk_size: int = 8,
    maximum_dataset_bytes: int = 2**31,
) -> OperatorDataset:
    """Convert valid modal trajectories to member-major supervised cases.

    Each member and selected target time yields one case. ``initial`` is the
    selected input-time field, ``conditions`` contains one scalar lead time,
    and the named field is the query/target. Both fields use the output grid.
    Vorticity is the spectral curl (one scalar channel in 2D, three in 3D).
    A coarser output grid uses retained-band restriction before reconstruction,
    never pointwise subsampling. The default velocity observable follows the
    velocity statistical solution in Lanthaler, Mishra and Parés-Pulido,
    arXiv:1909.06615, Sections 2 and 4; byte admission is execution policy.

    This is an explicit host dataset boundary. Only selected saves must be
    valid and finite; later failed saves do not invalidate earlier cases.
    Leakage identities contain only root-realization/member identity, shared
    across coupled resolutions. Scientific and execution provenance is in case
    IDs, not leakage groups. The initial-state owner declares modal identity.
    """
    field_ = parse(field, PeriodicFlowDatasetField, "field")
    initial_state = result.initial_state
    native = result.result
    source = initial_state.discretization
    if initial_state.discretization_id != source.prepared_id:
        raise ValueError("Initial state discretization identity is inconsistent.")
    dimension = len(source.axes)
    if dimension not in (2, 3) or any(a.family != "fourier" for a in source.axes):
        raise ValueError("Periodic flow datasets require two or three Fourier axes.")
    states = native.states
    if not isinstance(states, Array):
        raise TypeError("Periodic flow trajectories must be one modal velocity array.")
    members = initial_state.member_indices.shape[0]
    saves = native.times.shape[0]
    if states.shape != (members, saves, *source.modal_shape, dimension):
        raise ValueError(
            "Trajectory shape does not match the declared initial-state owner."
        )
    if not jnp.issubdtype(states.dtype, jnp.complexfloating):
        raise TypeError("Periodic flow trajectories require complex modal coefficients.")
    targets = tuple(
        _saved_index(t, saves, "target_time_indices") for t in target_time_indices
    )
    input_index = _saved_index(input_time_index, saves, "input_time_index")
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Target time indices must be nonempty and unique.")
    times = np.asarray(native.times)
    selected = (input_index, *targets)
    if not np.all(np.isfinite(times[list(selected)])):
        raise ValueError("Selected dataset times must be finite.")
    if any(times[t] <= times[input_index] for t in targets):
        raise ValueError("Target times must be later than the input time.")
    limit = positive_integer(maximum_dataset_bytes, "maximum_dataset_bytes")
    chunk = positive_integer(member_chunk_size, "member_chunk_size")
    output = source if output_discretization is None else output_discretization
    cases = members * len(targets)
    inputs, outputs = _dataset_values(
        states, native.valid, source, output, selected, field_, chunk, limit
    )
    leads = jnp.tile(
        jnp.asarray(times[list(targets)] - times[input_index], dtype=jnp.float64), members
    )
    axes = tuple(
        OperatorAxis(
            f"x{position}",
            axis.nodes,
            quadrature_weights=axis.quadrature_weights,
            basis="fourier",
            periodic=True,
        )
        for position, axis in enumerate(output.axes)
    )
    support_id = output.prepared_id
    batch = OperatorBatch(
        inputs={
            "initial": FunctionSamples(values=inputs, axes=axes, support_id=support_id),
            "conditions": FunctionSamples(
                values=leads[:, None, None],
                coordinates=jnp.zeros((1, 1), dtype=jnp.float64),
                support_id="periodic-flow-lead-time",
            ),
        },
        queries={field_: FunctionSamples(values=None, axes=axes, support_id=support_id)},
        case_axes=("case",),
        case_shape=(cases,),
    )
    provenance = []
    for member in np.asarray(initial_state.member_indices):
        realization = canonical_fingerprint(
            {
                "coupling": initial_state.coupling_id,
                "member": int(member),
            }
        )
        for target in targets:
            case_id = canonical_fingerprint(
                {
                    "kind": "periodic-flow-operator-case",
                    "realization": realization,
                    "measure": initial_state.measure_id,
                    "family": initial_state.family_id,
                    "source": source.prepared_id,
                    "output": output.prepared_id,
                    "prepared": native.prepared_id,
                    "result": native.result_id,
                    "field": field_,
                    "input_time": float(times[input_index]),
                    "target_time": float(times[target]),
                }
            )
            provenance.append(
                OperatorCaseProvenance(
                    case_id,
                    identities={"realization": realization},
                    order={
                        "member": float(member),
                        "input_time": float(times[input_index]),
                        "target_time": float(times[target]),
                        "lead_time": float(times[target] - times[input_index]),
                    },
                )
            )
    return OperatorDataset(
        batch,
        OperatorTargetBatch.from_arrays({field_: outputs}, batch),
        tuple(provenance),
    )


__all__ = ["PeriodicFlowDatasetField", "periodic_flow_operator_dataset"]
