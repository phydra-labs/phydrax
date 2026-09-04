#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jaxtyping import Array, ArrayLike, PyTree

from .._array_tree import ArrayPyTreeSchema
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._identity import ExecutableSignature, NumericRevision, SemanticProvenance
from .._strict import AbstractAttribute, StrictModule
from ._system import DiscreteStepContext, DiscreteSystem


_NONFINITE_STATE_STATUS = -1
_NONFINITE_INPUT_STATUS = -2
_NUMERIC_KINDS = frozenset("biufc")


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} must be a non-empty string.")
    return value.strip()


def _case_shape(value: Sequence[int], ndim: int, /) -> tuple[int, ...]:
    shape: list[int] = []
    for raw_size in value:
        if isinstance(raw_size, bool) or not isinstance(raw_size, (int, np.integer)):
            raise TypeError("case_shape dimensions must be integers.")
        size = int(raw_size)
        if size < 0:
            raise ValueError("case_shape dimensions must be nonnegative.")
        shape.append(size)
    resolved = tuple(shape)
    if len(resolved) != ndim:
        raise ValueError(f"case_shape must contain exactly {ndim} dimensions.")
    return resolved


def _case_array(value: ArrayLike, case_shape: tuple[int, ...], owner: str, /) -> Array:
    array = jnp.asarray(value)
    if np.dtype(array.dtype).kind not in _NUMERIC_KINDS:
        raise TypeError(f"{owner} must be a numeric array.")
    if array.shape == () and case_shape:
        array = jnp.broadcast_to(array, case_shape)
    if array.shape != case_shape:
        raise ValueError(f"{owner} must have shape {case_shape}; got {array.shape}.")
    return array


def _status_array(value: ArrayLike, case_shape: tuple[int, ...], owner: str, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError(f"{owner} must have an integer dtype.")
    if array.shape != case_shape:
        raise ValueError(f"{owner} must have shape {case_shape}; got {array.shape}.")
    return array.astype(jnp.int32)


def _mask_array(value: ArrayLike, case_shape: tuple[int, ...], owner: str, /) -> Array:
    array = jnp.asarray(value)
    if np.dtype(array.dtype) != np.dtype(bool):
        raise TypeError(f"{owner} must have boolean dtype.")
    if array.shape != case_shape:
        raise ValueError(f"{owner} must have shape {case_shape}; got {array.shape}.")
    return array


def _broadcast_mask(value: Array, case_shape: tuple[int, ...], owner: str, /) -> Array:
    if value.shape == case_shape:
        return value
    if value.shape == ():
        return jnp.broadcast_to(value, case_shape)
    raise ValueError(f"{owner} finite mask must be scalar or have shape {case_shape}.")


def _key_parts(
    keys: ArrayLike,
    case_shape: tuple[int, ...],
    owner: str,
    /,
) -> tuple[Array, Array, bool]:
    array = jnp.asarray(keys)
    typed = jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key)
    if typed:
        if array.shape != case_shape:
            raise ValueError(
                f"{owner} typed PRNG keys must have shape {case_shape}; "
                f"got {array.shape}."
            )
        return array, jax.random.key_data(array), True
    if np.dtype(array.dtype) != np.dtype(jnp.uint32):
        raise TypeError(f"{owner} legacy PRNG key data must have uint32 dtype.")
    expected = case_shape + (2,)
    if array.shape != expected:
        raise ValueError(
            f"{owner} legacy PRNG key data must have shape {expected}; got {array.shape}."
        )
    return array, array, False


def _split_keys(keys: Array, case_shape: tuple[int, ...], /) -> tuple[Array, Array]:
    _, data, typed = _key_parts(keys, case_shape, "keys")
    flattened = jnp.reshape(jax.random.wrap_key_data(data), (-1,))
    split = jax.vmap(lambda key: jax.random.split(key, 2))(flattened)
    next_typed = jnp.reshape(split[:, 0], case_shape)
    proposal_typed = jnp.reshape(split[:, 1], case_shape)
    if typed:
        return next_typed, proposal_typed
    return jax.random.key_data(next_typed), jax.random.key_data(proposal_typed)


def _select_keys(
    selector: Array,
    candidate: Array,
    source: Array,
    case_shape: tuple[int, ...],
    /,
) -> Array:
    _, candidate_data, candidate_typed = _key_parts(
        candidate, case_shape, "candidate key"
    )
    _, source_data, source_typed = _key_parts(source, case_shape, "source key")
    if candidate_typed != source_typed:
        raise TypeError(
            "Candidate and source keys must use the same PRNG representation."
        )
    selected_data = jnp.where(selector[..., None], candidate_data, source_data)
    return jax.random.wrap_key_data(selected_data) if source_typed else selected_data


def _template_leaves(
    schema: ArrayPyTreeSchema,
    template: PyTree[Any],
    owner: str,
    /,
) -> tuple[Array, ...]:
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    if treedef != schema.treedef:
        raise ValueError(f"{owner} PyTree structure does not match the state schema.")
    observed_paths = tuple(
        jax.tree_util.keystr(path) or "<root>" for path, _ in path_leaves
    )
    if observed_paths != schema.leaf_paths:
        raise ValueError(f"{owner} leaf paths do not match the state schema.")
    arrays: list[Array] = []
    for (_, value), leaf in zip(path_leaves, schema.leaves, strict=True):
        if not isinstance(value, (jax.Array, jax_core.Tracer, np.ndarray, np.generic)):
            raise TypeError(f"{owner} leaf {leaf.path} must be an array.")
        array = jnp.asarray(value)
        if array.shape != leaf.shape:
            raise ValueError(
                f"{owner} leaf {leaf.path} must have intrinsic shape {leaf.shape}; "
                f"got {array.shape}."
            )
        if np.dtype(array.dtype) != leaf.dtype:
            raise TypeError(
                f"{owner} leaf {leaf.path} dtype {array.dtype} does not match "
                f"schema dtype {leaf.dtype}."
            )
        arrays.append(array)
    return tuple(arrays)


def _broadcast_template(
    schema: ArrayPyTreeSchema,
    template: PyTree[Any],
    case_shape: tuple[int, ...],
    /,
) -> PyTree[Array]:
    leaves = _template_leaves(schema, template, "reset_fallback")
    return schema.treedef.unflatten(
        tuple(
            jnp.broadcast_to(value, case_shape + leaf.shape)
            for value, leaf in zip(leaves, schema.leaves, strict=True)
        )
    )


def _safe_tree(
    schema: ArrayPyTreeSchema,
    tree: PyTree[Any],
    finite: Array,
    /,
) -> PyTree[Array]:
    case_shape = schema.validate(tree)
    selector = _broadcast_mask(finite, case_shape, "Array PyTree")
    return schema.select_cases(selector, tree, schema.zeros(case_shape))


def _assert_tree(tree: PyTree[Any], predicate: Array, message: str, /) -> PyTree[Any]:
    return jax.tree_util.tree_map(
        lambda leaf: eqx.error_if(leaf, jnp.any(predicate), message), tree
    )


class PlantRuntimeState(StrictModule):
    """Thin envelope around one domain-owned complete plant payload."""

    payload: Any
    time: Array
    step_index: Array
    key: Array
    semantic_provenance_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    state_schema_id: str = eqx.field(static=True)
    execution_signature_id: str = eqx.field(static=True)

    def __init__(
        self,
        payload: PyTree[Any],
        time: ArrayLike,
        step_index: ArrayLike,
        key: ArrayLike,
        semantic_provenance_id: str,
        numeric_revision_id: str,
        state_schema_id: str,
        execution_signature_id: str,
        /,
    ):
        self.payload = payload
        self.time = jnp.asarray(time)
        self.step_index = jnp.asarray(step_index, dtype=jnp.int32)
        self.key = jnp.asarray(key)
        self.semantic_provenance_id = _identifier(
            semantic_provenance_id, "semantic_provenance_id"
        )
        self.numeric_revision_id = _identifier(numeric_revision_id, "numeric_revision_id")
        self.state_schema_id = _identifier(state_schema_id, "state_schema_id")
        self.execution_signature_id = _identifier(
            execution_signature_id, "execution_signature_id"
        )


class PlantParameters(StrictModule):
    """Parameter values bound to their exact schema and numeric revision."""

    values: Any
    schema_id: str = eqx.field(static=True)
    numeric_revision: NumericRevision

    def __init__(
        self,
        values: PyTree[Any],
        schema_id: str,
        numeric_revision: NumericRevision,
        /,
    ):
        if not isinstance(numeric_revision, NumericRevision):
            raise TypeError("numeric_revision must be a NumericRevision.")
        self.values = values
        self.schema_id = _identifier(schema_id, "PlantParameters schema_id")
        self.numeric_revision = numeric_revision


class PlantStepContext(StrictModule):
    """Casewise source/target times and source step indices for one transaction."""

    source_time: Array
    target_time: Array
    step_index: Array

    def __init__(
        self,
        source_time: ArrayLike,
        target_time: ArrayLike,
        step_index: ArrayLike,
        /,
    ):
        source = jnp.asarray(source_time)
        target = jnp.asarray(target_time)
        index = jnp.asarray(step_index, dtype=jnp.int32)
        if source.shape != target.shape or source.shape != index.shape:
            raise ValueError(
                "PlantStepContext source_time, target_time, and step_index must "
                "have one case shape."
            )
        if np.dtype(source.dtype).kind not in _NUMERIC_KINDS:
            raise TypeError("PlantStepContext times must be numeric arrays.")
        if np.dtype(target.dtype).kind not in _NUMERIC_KINDS:
            raise TypeError("PlantStepContext times must be numeric arrays.")
        self.source_time = source
        self.target_time = target
        self.step_index = index

    @property
    def duration(self) -> Array:
        return self.target_time - self.source_time


class PlantResetResult(StrictModule):
    """Candidate, atomically accepted reset state, and casewise reset evidence."""

    candidate_state: PlantRuntimeState
    accepted_state: PlantRuntimeState
    attempted: Array
    successful: Array
    status: Array
    backend_status: Array
    evidence: Any


class PlantStepResult(StrictModule):
    """Candidate, atomically accepted step state, and casewise step evidence."""

    candidate_state: PlantRuntimeState
    accepted_state: PlantRuntimeState
    attempted: Array
    successful: Array
    status: Array
    backend_status: Array
    evidence: Any


class PlantProposal(StrictModule):
    """Domain proposal before runtime metadata and transactional selection."""

    candidate_payload: Any
    accepted_payload: Any
    attempted: Array
    successful: Array
    status: Array
    backend_status: Array
    evidence: Any


class PlantCheckpoint(StrictModule):
    """Exact complete-state checkpoint bound to all plant identities."""

    state: PlantRuntimeState
    digest: str = eqx.field(static=True)
    semantic_provenance_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    state_schema_id: str = eqx.field(static=True)
    execution_signature_id: str = eqx.field(static=True)

    def __init__(
        self,
        state: PlantRuntimeState,
        digest: str,
        semantic_provenance_id: str,
        numeric_revision_id: str,
        state_schema_id: str,
        execution_signature_id: str,
        /,
    ):
        if not isinstance(state, PlantRuntimeState):
            raise TypeError("PlantCheckpoint state must be a PlantRuntimeState.")
        self.state = state
        self.digest = _identifier(digest, "PlantCheckpoint digest")
        self.semantic_provenance_id = _identifier(
            semantic_provenance_id, "PlantCheckpoint semantic_provenance_id"
        )
        self.numeric_revision_id = _identifier(
            numeric_revision_id, "PlantCheckpoint numeric_revision_id"
        )
        self.state_schema_id = _identifier(
            state_schema_id, "PlantCheckpoint state_schema_id"
        )
        self.execution_signature_id = _identifier(
            execution_signature_id, "PlantCheckpoint execution_signature_id"
        )


class PlantReplayResult(StrictModule):
    """Replay trajectory with independent transition and exact-digest evidence."""

    final_state: PlantRuntimeState
    accepted_states: tuple[PlantRuntimeState, ...]
    step_results: tuple[PlantStepResult, ...]
    successful: Array
    status: Array
    first_failure_step: Array
    first_failure_status: Array
    matched: bool = eqx.field(static=True)
    first_mismatch_step: int = eqx.field(static=True)
    expected_digest: str | None = eqx.field(static=True)
    actual_digest: str | None = eqx.field(static=True)


def _validate_configuration(plant: AbstractDiscretePlant, /) -> None:
    if not isinstance(plant.state_schema, ArrayPyTreeSchema):
        raise TypeError("state_schema must be an ArrayPyTreeSchema.")
    if plant.control_schema is not None and not isinstance(
        plant.control_schema, ArrayPyTreeSchema
    ):
        raise TypeError("control_schema must be an ArrayPyTreeSchema or None.")
    if not isinstance(plant.parameter_schema, ArrayPyTreeSchema):
        raise TypeError("parameter_schema must be an ArrayPyTreeSchema.")
    if not isinstance(plant.semantic_provenance, SemanticProvenance):
        raise TypeError("semantic_provenance must be a SemanticProvenance.")
    if not isinstance(plant.numeric_revision, NumericRevision):
        raise TypeError("numeric_revision must be a NumericRevision.")
    if not isinstance(plant.execution_signature, ExecutableSignature):
        raise TypeError("execution_signature must be an ExecutableSignature.")
    if plant.numeric_revision.semantic_id != plant.semantic_provenance.semantic_id:
        raise ValueError("Plant numeric revision belongs to different semantics.")
    valid_case_ndims = {0, plant.state_schema.case_ndim}
    if (
        plant.control_schema is not None
        and plant.control_schema.case_ndim not in valid_case_ndims
    ):
        raise ValueError(
            "control_schema case_ndim must be zero or match state_schema case_ndim."
        )
    if plant.parameter_schema.case_ndim not in valid_case_ndims:
        raise ValueError(
            "parameter_schema case_ndim must be zero or match state_schema case_ndim."
        )
    for name, value in (
        ("require_finite_state", plant.require_finite_state),
        ("require_finite_controls", plant.require_finite_controls),
        ("require_finite_parameters", plant.require_finite_parameters),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool.")
    _template_leaves(plant.state_schema, plant.reset_fallback, "reset_fallback")


def _plant_ids(plant: AbstractDiscretePlant, /) -> tuple[str, str, str, str]:
    return (
        plant.semantic_provenance.semantic_id,
        plant.numeric_revision.revision_id,
        plant.state_schema.schema_id,
        plant.execution_signature.signature_id,
    )


def _validate_parameters(
    plant: AbstractDiscretePlant,
    parameters: PlantParameters,
    case_shape: tuple[int, ...],
    /,
) -> Array:
    if not isinstance(parameters, PlantParameters):
        raise TypeError("parameters must be PlantParameters.")
    if parameters.schema_id != plant.parameter_schema.schema_id:
        raise ValueError("PlantParameters schema_id does not match this plant.")
    if parameters.numeric_revision.semantic_id != plant.semantic_provenance.semantic_id:
        raise ValueError("PlantParameters belongs to different semantic provenance.")
    if parameters.numeric_revision.revision_id != plant.numeric_revision.revision_id:
        raise ValueError("PlantParameters numeric revision does not match this plant.")
    parameter_case_shape = plant.parameter_schema.validate(parameters.values)
    if parameter_case_shape not in ((), case_shape):
        raise ValueError(
            "PlantParameters case shape must be scalar/shared or match plant cases."
        )
    return (
        plant.parameter_schema.finite_mask(parameters.values)
        if plant.require_finite_parameters
        else jnp.ones(parameter_case_shape, dtype=bool)
    )


def _validate_commands(
    plant: AbstractDiscretePlant,
    commands: PyTree[Any] | None,
    case_shape: tuple[int, ...],
    /,
) -> Array:
    if plant.control_schema is None:
        if commands is not None:
            raise ValueError("An autonomous plant does not accept commands.")
        return jnp.asarray(True)
    if commands is None:
        raise ValueError("This plant requires explicit commands.")
    command_case_shape = plant.control_schema.validate(commands)
    if command_case_shape not in ((), case_shape):
        raise ValueError("Command case shape must be scalar/shared or match plant cases.")
    return (
        plant.control_schema.finite_mask(commands)
        if plant.require_finite_controls
        else jnp.ones(command_case_shape, dtype=bool)
    )


def _validate_runtime_state(
    plant: AbstractDiscretePlant,
    state: PlantRuntimeState,
    /,
) -> tuple[tuple[int, ...], PlantRuntimeState]:
    if not isinstance(state, PlantRuntimeState):
        raise TypeError("source must be a PlantRuntimeState.")
    expected = _plant_ids(plant)
    observed = (
        state.semantic_provenance_id,
        state.numeric_revision_id,
        state.state_schema_id,
        state.execution_signature_id,
    )
    names = (
        "semantic provenance",
        "numeric revision",
        "state schema",
        "execution signature",
    )
    for name, observed_id, expected_id in zip(names, observed, expected, strict=True):
        if observed_id != expected_id:
            raise ValueError(f"PlantRuntimeState {name} does not match this plant.")
    case_shape = plant.state_schema.validate(state.payload)
    if state.time.shape != case_shape or state.step_index.shape != case_shape:
        raise ValueError(
            "PlantRuntimeState time and step_index must match the payload case shape."
        )
    if np.dtype(state.time.dtype).kind not in _NUMERIC_KINDS:
        raise TypeError("PlantRuntimeState time must have a numeric dtype.")
    _key_parts(state.key, case_shape, "PlantRuntimeState key")
    checked_time = eqx.error_if(
        state.time,
        jnp.any(~jnp.isfinite(state.time)),
        "PlantRuntimeState time must be finite.",
    )
    checked_index = eqx.error_if(
        state.step_index,
        jnp.any(state.step_index < 0),
        "PlantRuntimeState step_index must be nonnegative.",
    )
    checked_payload = (
        _assert_tree(
            state.payload,
            ~plant.state_schema.finite_mask(state.payload),
            "PlantRuntimeState payload must be finite.",
        )
        if plant.require_finite_state
        else state.payload
    )
    checked = PlantRuntimeState(
        checked_payload,
        checked_time,
        checked_index,
        state.key,
        *expected,
    )
    return case_shape, checked


def _validate_context(
    context: PlantStepContext,
    source: PlantRuntimeState,
    case_shape: tuple[int, ...],
    /,
) -> PlantStepContext:
    if not isinstance(context, PlantStepContext):
        raise TypeError("context must be a PlantStepContext.")
    if (
        context.source_time.shape != case_shape
        or context.target_time.shape != case_shape
        or context.step_index.shape != case_shape
    ):
        raise ValueError("PlantStepContext values must match the source case shape.")
    inconsistent = (context.source_time != source.time) | (
        context.step_index != source.step_index
    )
    invalid_interval = (
        ~jnp.isfinite(context.source_time)
        | ~jnp.isfinite(context.target_time)
        | (context.target_time <= context.source_time)
        | (context.step_index < 0)
    )
    checked_source = eqx.error_if(
        context.source_time,
        jnp.any(inconsistent),
        "PlantStepContext source time/index is inconsistent with source state.",
    )
    checked_source = eqx.error_if(
        checked_source,
        jnp.any(invalid_interval),
        "PlantStepContext must define finite, strictly increasing time intervals.",
    )
    return PlantStepContext(checked_source, context.target_time, context.step_index)


def _normalize_proposal(
    proposal: PlantProposal,
    case_shape: tuple[int, ...],
    owner: str,
    /,
) -> PlantProposal:
    if not isinstance(proposal, PlantProposal):
        raise TypeError(f"{owner} must return PlantProposal.")
    attempted = _mask_array(proposal.attempted, case_shape, f"{owner} attempted")
    successful = _mask_array(proposal.successful, case_shape, f"{owner} successful")
    status = _status_array(proposal.status, case_shape, f"{owner} status")
    backend = _status_array(
        proposal.backend_status, case_shape, f"{owner} backend_status"
    )
    successful = eqx.error_if(
        successful,
        jnp.any(successful & ((~attempted) | (status != 0))),
        f"{owner} successful cases must be attempted with status zero.",
    )
    return PlantProposal(
        proposal.candidate_payload,
        proposal.accepted_payload,
        attempted,
        successful,
        status,
        backend,
        proposal.evidence,
    )


def _final_outcome(
    proposal: PlantProposal,
    input_finite: Array,
    candidate_finite: Array,
    /,
) -> tuple[Array, Array]:
    input_failure = proposal.attempted & ~input_finite
    candidate_failure = proposal.attempted & ~candidate_finite
    successful = proposal.successful & input_finite & candidate_finite
    status = jnp.where(
        input_failure & proposal.successful,
        jnp.asarray(_NONFINITE_INPUT_STATUS, dtype=jnp.int32),
        proposal.status,
    )
    status = jnp.where(
        candidate_failure & proposal.successful,
        jnp.asarray(_NONFINITE_STATE_STATUS, dtype=jnp.int32),
        status,
    ).astype(jnp.int32)
    return successful, status


def _select_runtime_state(
    schema: ArrayPyTreeSchema,
    selector: Array,
    candidate: PlantRuntimeState,
    source: PlantRuntimeState,
    case_shape: tuple[int, ...],
    /,
) -> PlantRuntimeState:
    return PlantRuntimeState(
        schema.select_cases(selector, candidate.payload, source.payload),
        jnp.where(selector, candidate.time, source.time),
        jnp.where(selector, candidate.step_index, source.step_index),
        _select_keys(selector, candidate.key, source.key, case_shape),
        source.semantic_provenance_id,
        source.numeric_revision_id,
        source.state_schema_id,
        source.execution_signature_id,
    )


def _runtime_digest(state: PlantRuntimeState, /) -> str:
    arrays = array_tree_fingerprint(
        (state.payload, state.time, state.step_index, jax.random.key_data(state.key))
    )
    return canonical_fingerprint(
        {
            "kind": "plant-runtime-state-v1",
            "semantic_provenance_id": state.semantic_provenance_id,
            "numeric_revision_id": state.numeric_revision_id,
            "state_schema_id": state.state_schema_id,
            "execution_signature_id": state.execution_signature_id,
            "arrays": arrays,
        }
    )


class AbstractDiscretePlant(StrictModule):
    """Domain-neutral final transaction wrappers around protected proposals."""

    state_schema: AbstractAttribute[ArrayPyTreeSchema]
    control_schema: AbstractAttribute[ArrayPyTreeSchema | None]
    parameter_schema: AbstractAttribute[ArrayPyTreeSchema]
    reset_fallback: AbstractAttribute[Any]
    semantic_provenance: AbstractAttribute[SemanticProvenance]
    numeric_revision: AbstractAttribute[NumericRevision]
    execution_signature: AbstractAttribute[ExecutableSignature]
    require_finite_state: AbstractAttribute[bool]
    require_finite_controls: AbstractAttribute[bool]
    require_finite_parameters: AbstractAttribute[bool]

    @abstractmethod
    def propose_reset(
        self,
        keys: Array,
        parameters: PyTree[Any],
        /,
        *,
        case_shape: tuple[int, ...],
        initial_time: Array,
    ) -> PlantProposal:
        """Return a domain payload proposal; the public wrapper owns acceptance."""
        raise NotImplementedError

    @abstractmethod
    def propose_step(
        self,
        context: PlantStepContext,
        source: PyTree[Any],
        commands: PyTree[Any] | None,
        parameters: PyTree[Any],
        keys: Array,
        /,
    ) -> PlantProposal:
        """Return a domain payload proposal; the public wrapper owns acceptance."""
        raise NotImplementedError

    def reset(
        self,
        keys: ArrayLike,
        parameters: PlantParameters,
        /,
        *,
        case_shape: Sequence[int] = (),
        initial_time: ArrayLike = 0,
    ) -> PlantResetResult:
        """Propose reset cases and atomically retain the prepared fallback on failure."""
        _validate_configuration(self)
        resolved_case_shape = _case_shape(case_shape, self.state_schema.case_ndim)
        source_keys, _, _ = _key_parts(keys, resolved_case_shape, "reset keys")
        next_keys, proposal_keys = _split_keys(source_keys, resolved_case_shape)
        initial_time_array = jnp.asarray(initial_time)
        if not jnp.issubdtype(initial_time_array.dtype, jnp.inexact):
            initial_time_array = initial_time_array.astype(jnp.float32)
        time = _case_array(initial_time_array, resolved_case_shape, "initial_time")
        time = eqx.error_if(
            time,
            jnp.any(~jnp.isfinite(time)),
            "Plant reset initial_time must be finite.",
        )
        parameter_finite = _validate_parameters(self, parameters, resolved_case_shape)
        safe_parameters = _safe_tree(
            self.parameter_schema, parameters.values, parameter_finite
        )
        proposal = _normalize_proposal(
            self.propose_reset(
                proposal_keys,
                safe_parameters,
                case_shape=resolved_case_shape,
                initial_time=time,
            ),
            resolved_case_shape,
            "propose_reset",
        )
        if (
            self.state_schema.validate(proposal.candidate_payload) != resolved_case_shape
            or self.state_schema.validate(proposal.accepted_payload)
            != resolved_case_shape
        ):
            raise ValueError("Reset proposal payloads have the wrong case shape.")
        candidate_finite = (
            self.state_schema.finite_mask(proposal.candidate_payload)
            & self.state_schema.finite_mask(proposal.accepted_payload)
            if self.require_finite_state
            else jnp.ones(resolved_case_shape, dtype=bool)
        )
        input_finite = _broadcast_mask(
            parameter_finite, resolved_case_shape, "parameters"
        )
        successful, status = _final_outcome(proposal, input_finite, candidate_finite)
        ids = _plant_ids(self)
        candidate = PlantRuntimeState(
            proposal.candidate_payload,
            time,
            jnp.zeros(resolved_case_shape, dtype=jnp.int32),
            next_keys,
            *ids,
        )
        fallback_payload = _broadcast_template(
            self.state_schema, self.reset_fallback, resolved_case_shape
        )
        if self.require_finite_state:
            fallback_payload = _assert_tree(
                fallback_payload,
                ~self.state_schema.finite_mask(fallback_payload),
                "reset_fallback must be finite for this plant.",
            )
        fallback = PlantRuntimeState(
            fallback_payload,
            time,
            jnp.zeros(resolved_case_shape, dtype=jnp.int32),
            source_keys,
            *ids,
        )
        commit = PlantRuntimeState(
            proposal.accepted_payload,
            time,
            jnp.zeros(resolved_case_shape, dtype=jnp.int32),
            next_keys,
            *ids,
        )
        accepted = _select_runtime_state(
            self.state_schema,
            successful,
            commit,
            fallback,
            resolved_case_shape,
        )
        return PlantResetResult(
            candidate,
            accepted,
            proposal.attempted,
            successful,
            status,
            proposal.backend_status,
            proposal.evidence,
        )

    def step(
        self,
        context: PlantStepContext,
        source: PlantRuntimeState,
        commands: PyTree[Any] | None,
        parameters: PlantParameters,
        /,
    ) -> PlantStepResult:
        """Propose one complete state and atomically commit each successful case."""
        _validate_configuration(self)
        case_shape, source = _validate_runtime_state(self, source)
        checked_context = _validate_context(context, source, case_shape)
        parameter_finite = _validate_parameters(self, parameters, case_shape)
        command_finite = _validate_commands(self, commands, case_shape)
        safe_parameters = _safe_tree(
            self.parameter_schema, parameters.values, parameter_finite
        )
        safe_commands = (
            None
            if self.control_schema is None
            else _safe_tree(self.control_schema, commands, command_finite)
        )
        next_keys, proposal_keys = _split_keys(source.key, case_shape)
        proposal = _normalize_proposal(
            self.propose_step(
                checked_context,
                source.payload,
                safe_commands,
                safe_parameters,
                proposal_keys,
            ),
            case_shape,
            "propose_step",
        )
        if (
            self.state_schema.validate(proposal.candidate_payload) != case_shape
            or self.state_schema.validate(proposal.accepted_payload) != case_shape
        ):
            raise ValueError("Step proposal payloads have the wrong case shape.")
        candidate_finite = (
            self.state_schema.finite_mask(proposal.candidate_payload)
            & self.state_schema.finite_mask(proposal.accepted_payload)
            if self.require_finite_state
            else jnp.ones(case_shape, dtype=bool)
        )
        input_finite = _broadcast_mask(
            parameter_finite, case_shape, "parameters"
        ) & _broadcast_mask(command_finite, case_shape, "commands")
        successful, status = _final_outcome(proposal, input_finite, candidate_finite)
        candidate = PlantRuntimeState(
            proposal.candidate_payload,
            checked_context.target_time,
            source.step_index + jnp.asarray(1, dtype=jnp.int32),
            next_keys,
            *_plant_ids(self),
        )
        commit = PlantRuntimeState(
            proposal.accepted_payload,
            checked_context.target_time,
            source.step_index + jnp.asarray(1, dtype=jnp.int32),
            next_keys,
            *_plant_ids(self),
        )
        accepted = _select_runtime_state(
            self.state_schema, successful, commit, source, case_shape
        )
        return PlantStepResult(
            candidate,
            accepted,
            proposal.attempted,
            successful,
            status,
            proposal.backend_status,
            proposal.evidence,
        )

    def state_digest(self, state: PlantRuntimeState, /) -> str:
        """Return an exact host digest over every payload and runtime atom."""
        _validate_configuration(self)
        _, checked = _validate_runtime_state(self, state)
        return _runtime_digest(checked)

    def checkpoint(self, state: PlantRuntimeState, /) -> PlantCheckpoint:
        """Capture a complete accepted runtime state with an exact digest."""
        _validate_configuration(self)
        _, checked = _validate_runtime_state(self, state)
        return PlantCheckpoint(checked, _runtime_digest(checked), *_plant_ids(self))

    def verify_checkpoint(self, checkpoint: PlantCheckpoint, /) -> bool:
        """Verify identities and every checkpointed array byte exactly."""
        self._validate_checkpoint_identity(checkpoint)
        _, checked = _validate_runtime_state(self, checkpoint.state)
        return _runtime_digest(checked) == checkpoint.digest

    def restore(self, checkpoint: PlantCheckpoint, /) -> PlantRuntimeState:
        """Return an exactly verified checkpoint state or fail closed."""
        self._validate_checkpoint_identity(checkpoint)
        _, checked = _validate_runtime_state(self, checkpoint.state)
        if _runtime_digest(checked) != checkpoint.digest:
            raise ValueError("PlantCheckpoint digest verification failed.")
        return checked

    def _validate_checkpoint_identity(self, checkpoint: PlantCheckpoint, /) -> None:
        if not isinstance(checkpoint, PlantCheckpoint):
            raise TypeError("checkpoint must be a PlantCheckpoint.")
        observed = (
            checkpoint.semantic_provenance_id,
            checkpoint.numeric_revision_id,
            checkpoint.state_schema_id,
            checkpoint.execution_signature_id,
        )
        for name, observed_id, expected_id in zip(
            (
                "semantic provenance",
                "numeric revision",
                "state schema",
                "execution signature",
            ),
            observed,
            _plant_ids(self),
            strict=True,
        ):
            if observed_id != expected_id:
                raise ValueError(f"PlantCheckpoint {name} does not match this plant.")

    def replay(
        self,
        checkpoint: PlantCheckpoint,
        contexts: Sequence[PlantStepContext],
        commands: Sequence[PyTree[Any] | None],
        parameters: PlantParameters,
        /,
        *,
        expected_digests: Sequence[str] = (),
    ) -> PlantReplayResult:
        """Replay transitions and record the first exact accepted-state divergence."""
        contexts_ = tuple(contexts)
        commands_ = tuple(commands)
        expected_ = tuple(expected_digests)
        if len(commands_) != len(contexts_):
            raise ValueError("Replay commands and contexts must have the same length.")
        if expected_ and len(expected_) != len(contexts_):
            raise ValueError(
                "expected_digests must be empty or contain one digest per replay step."
            )
        if any(not isinstance(digest, str) or not digest for digest in expected_):
            raise ValueError("Replay expected digests must be non-empty strings.")

        state = self.restore(checkpoint)
        case_shape, state = _validate_runtime_state(self, state)
        accepted_states: list[PlantRuntimeState] = [state]
        results: list[PlantStepResult] = []
        successful = jnp.ones(case_shape, dtype=bool)
        first_failure_step = jnp.full(case_shape, -1, dtype=jnp.int32)
        first_failure_status = jnp.zeros(case_shape, dtype=jnp.int32)
        matched = True
        first_mismatch_step = -1
        expected_digest: str | None = None
        actual_digest: str | None = None

        for index, (step_context, step_commands) in enumerate(
            zip(contexts_, commands_, strict=True)
        ):
            result = self.step(step_context, state, step_commands, parameters)
            results.append(result)
            state = result.accepted_state
            accepted_states.append(state)
            first = (first_failure_step < 0) & ~result.successful
            first_failure_step = jnp.where(
                first,
                jnp.asarray(index, dtype=jnp.int32),
                first_failure_step,
            )
            first_failure_status = jnp.where(
                first, result.status, first_failure_status
            ).astype(jnp.int32)
            successful = successful & result.successful
            if expected_:
                observed_digest = self.state_digest(state)
                if matched and observed_digest != expected_[index]:
                    matched = False
                    first_mismatch_step = index
                    expected_digest = expected_[index]
                    actual_digest = observed_digest

        return PlantReplayResult(
            state,
            tuple(accepted_states),
            tuple(results),
            successful,
            first_failure_status,
            first_failure_step,
            first_failure_status,
            matched,
            first_mismatch_step,
            expected_digest,
            actual_digest,
        )


def _schema_from_template(template: Array, case_ndim: int, /) -> ArrayPyTreeSchema:
    expanded = jnp.reshape(template, (1,) * case_ndim + template.shape)
    return ArrayPyTreeSchema.from_tree(expanded, case_ndim=case_ndim)


class _ArrayDiscreteSystemEvidence(StrictModule):
    legacy_candidate_state: Array
    legacy_accepted_state: Array


class ArrayDiscreteSystemPlant(AbstractDiscretePlant):
    """Adapter from the legacy single-array ``DiscreteSystem`` contract."""

    system: DiscreteSystem
    initializer: Callable[[Array], ArrayLike]
    state_schema: ArrayPyTreeSchema
    control_schema: ArrayPyTreeSchema | None
    parameter_schema: ArrayPyTreeSchema
    reset_fallback: Array
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    execution_signature: ExecutableSignature
    require_finite_state: bool = eqx.field(static=True)
    require_finite_controls: bool = eqx.field(static=True)
    require_finite_parameters: bool = eqx.field(static=True)
    uses_parameters: bool = eqx.field(static=True)

    def __init__(
        self,
        system: DiscreteSystem,
        initializer: Callable[[Array], ArrayLike],
        /,
        *,
        reset_fallback: ArrayLike,
        semantic_provenance: SemanticProvenance,
        numeric_revision: NumericRevision,
        execution_signature: ExecutableSignature,
        parameter_schema: ArrayPyTreeSchema | None = None,
        case_ndim: int = 0,
        control_dtype: Any | None = None,
        require_finite_state: bool = True,
        require_finite_controls: bool = True,
        require_finite_parameters: bool = True,
    ):
        if not isinstance(system, DiscreteSystem):
            raise TypeError("system must be a DiscreteSystem.")
        if not callable(initializer):
            raise TypeError("initializer must be callable.")
        if isinstance(case_ndim, bool) or not isinstance(case_ndim, (int, np.integer)):
            raise TypeError("case_ndim must be an integer.")
        resolved_case_ndim = int(case_ndim)
        if resolved_case_ndim < 0:
            raise ValueError("case_ndim must be nonnegative.")
        fallback = jnp.asarray(reset_fallback)
        if fallback.shape != system.state_layout.shape:
            raise ValueError(
                "reset_fallback shape must match DiscreteSystem state_layout.shape."
            )
        if np.dtype(fallback.dtype).kind not in _NUMERIC_KINDS:
            raise TypeError("reset_fallback must have numeric or boolean dtype.")
        for name, value in (
            ("require_finite_state", require_finite_state),
            ("require_finite_controls", require_finite_controls),
            ("require_finite_parameters", require_finite_parameters),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool.")
        if require_finite_state and not bool(np.all(np.isfinite(np.asarray(fallback)))):
            raise ValueError("reset_fallback must be finite for this plant.")
        state_schema = _schema_from_template(fallback, resolved_case_ndim)
        if system.input_layout is None:
            control_schema = None
        else:
            resolved_control_dtype = (
                fallback.dtype if control_dtype is None else np.dtype(control_dtype)
            )
            control_template = jnp.zeros(
                system.input_layout.shape, dtype=resolved_control_dtype
            )
            control_schema = _schema_from_template(control_template, resolved_case_ndim)
        uses_parameters = parameter_schema is not None
        resolved_parameter_schema = (
            ArrayPyTreeSchema.from_tree((), case_ndim=0)
            if parameter_schema is None
            else parameter_schema
        )
        if not isinstance(resolved_parameter_schema, ArrayPyTreeSchema):
            raise TypeError("parameter_schema must be an ArrayPyTreeSchema or None.")
        self.system = system
        self.initializer = initializer
        self.state_schema = state_schema
        self.control_schema = control_schema
        self.parameter_schema = resolved_parameter_schema
        self.reset_fallback = fallback
        self.semantic_provenance = semantic_provenance
        self.numeric_revision = numeric_revision
        self.execution_signature = execution_signature
        self.require_finite_state = require_finite_state
        self.require_finite_controls = require_finite_controls
        self.require_finite_parameters = require_finite_parameters
        self.uses_parameters = uses_parameters
        _validate_configuration(self)

    def propose_reset(
        self,
        keys: Array,
        parameters: PyTree[Any],
        /,
        *,
        case_shape: tuple[int, ...],
        initial_time: Array,
    ) -> PlantProposal:
        del parameters, initial_time
        if self.state_schema.case_ndim == 0:
            payload = jnp.asarray(self.initializer(keys))
        else:
            _, key_data, typed = _key_parts(keys, case_shape, "initializer keys")
            key_values = (
                jnp.reshape(keys, (-1,)) if typed else jnp.reshape(key_data, (-1, 2))
            )
            flat_payload = jax.vmap(self.initializer)(key_values)
            payload = jnp.reshape(
                flat_payload, case_shape + self.system.state_layout.shape
            )
        return PlantProposal(
            payload,
            payload,
            jnp.ones(case_shape, dtype=bool),
            jnp.ones(case_shape, dtype=bool),
            jnp.zeros(case_shape, dtype=jnp.int32),
            jnp.zeros(case_shape, dtype=jnp.int32),
            (),
        )

    def propose_step(
        self,
        context: PlantStepContext,
        source: PyTree[Any],
        commands: PyTree[Any] | None,
        parameters: PyTree[Any],
        keys: Array,
        /,
    ) -> PlantProposal:
        del keys
        args = parameters if self.uses_parameters else None
        if self.state_schema.case_ndim == 0:
            result = self.system.evaluate_result(
                DiscreteStepContext(
                    context.source_time, context.target_time, context.step_index
                ),
                source,
                args,
                inputs=commands,
            )
        else:
            case_shape = context.source_time.shape
            case_count = int(np.prod(case_shape))
            flat_source = jnp.reshape(
                source, (case_count,) + self.system.state_layout.shape
            )
            flat_source_time = jnp.reshape(context.source_time, (case_count,))
            flat_target_time = jnp.reshape(context.target_time, (case_count,))
            flat_index = jnp.reshape(context.step_index, (case_count,))
            parameter_axes = 0 if self.parameter_schema.case_ndim else None
            flat_parameters = (
                jax.tree_util.tree_map(
                    lambda leaf: jnp.reshape(
                        leaf,
                        (case_count,) + leaf.shape[self.parameter_schema.case_ndim :],
                    ),
                    parameters,
                )
                if self.parameter_schema.case_ndim
                else parameters
            )

            if self.control_schema is None:

                def evaluate_one(source_time, target_time, index, state, args_):
                    return self.system.evaluate_result(
                        DiscreteStepContext(source_time, target_time, index),
                        state,
                        args_,
                    )

                result = jax.vmap(
                    evaluate_one,
                    in_axes=(0, 0, 0, 0, parameter_axes),
                )(
                    flat_source_time,
                    flat_target_time,
                    flat_index,
                    flat_source,
                    flat_parameters if self.uses_parameters else None,
                )
            else:
                assert commands is not None
                assert self.system.input_layout is not None
                command_axes = 0 if self.control_schema.case_ndim else None
                flat_commands = (
                    jnp.reshape(
                        commands,
                        (case_count,) + self.system.input_layout.shape,
                    )
                    if self.control_schema.case_ndim
                    else commands
                )

                def evaluate_one(source_time, target_time, index, state, command, args_):
                    return self.system.evaluate_result(
                        DiscreteStepContext(source_time, target_time, index),
                        state,
                        args_,
                        inputs=command,
                    )

                result = jax.vmap(
                    evaluate_one,
                    in_axes=(0, 0, 0, 0, command_axes, parameter_axes),
                )(
                    flat_source_time,
                    flat_target_time,
                    flat_index,
                    flat_source,
                    flat_commands,
                    flat_parameters if self.uses_parameters else None,
                )
            result = eqx.tree_at(
                lambda item: (item.candidate_state, item.accepted_state),
                result,
                (
                    jnp.reshape(
                        result.candidate_state,
                        case_shape + self.system.state_layout.shape,
                    ),
                    jnp.reshape(
                        result.accepted_state,
                        case_shape + self.system.state_layout.shape,
                    ),
                ),
            )
            result = eqx.tree_at(
                lambda item: (item.successful, item.status),
                result,
                (
                    jnp.reshape(result.successful, case_shape),
                    jnp.reshape(result.status, case_shape),
                ),
            )
        return PlantProposal(
            result.candidate_state,
            result.accepted_state,
            jnp.ones(result.successful.shape, dtype=bool),
            result.successful,
            result.status,
            result.status,
            _ArrayDiscreteSystemEvidence(result.candidate_state, result.accepted_state),
        )


__all__ = [
    "AbstractDiscretePlant",
    "ArrayDiscreteSystemPlant",
    "PlantCheckpoint",
    "PlantParameters",
    "PlantProposal",
    "PlantReplayResult",
    "PlantResetResult",
    "PlantRuntimeState",
    "PlantStepContext",
    "PlantStepResult",
]
