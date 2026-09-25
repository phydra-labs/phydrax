#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from enum import IntEnum, IntFlag
from math import prod
from operator import index
from typing import Any, ClassVar, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._admissibility import (
    AdmissibilityHeader,
    AdmissibilityReason,
    combine_admissibility,
    DOMAIN_REASON_SHIFT,
    reason_bits_where,
)
from .._differentiation import (
    branch_policy_contract,
    BranchDifferentiationPolicy,
    ComponentAuthority,
    DerivativeContract,
    DerivativeSurface,
)
from .._fingerprint import array_tree_signature, canonical_fingerprint
from .._iteration import (
    bind_iteration_scope,
    finalize_iteration,
    initialize_iteration,
    IterationCapabilities,
    IterationCoordinates,
    IterationEvidence,
    IterationPhase,
    IterationPlan,
    IterationRecord,
    update_iteration,
)
from .._model import (
    AbstractArrayModel,
    AbstractComponentSlot,
    bind_component,
    ComponentContract,
    ModelPorts,
    PortMapping,
)
from .._model._component import bind_positional_component
from .._model._ports import require_port_shapes
from .._numerics._checkpointed_scan import (
    AdaptiveReplayPreparationPolicy,
    checkpointed_scan,
    prepare_replay_schedule,
    PreparedReplaySchedule,
)
from .._numerics._ssp_runge_kutta import (
    AbstractSSPRKStageTransform,
    ssprk33_step_with_evidence,
    ssprk54_step_with_evidence,
    SSPRKStepResult,
    StageTransformResult,
)
from .._strict import StrictModule
from .._trainable import fixed_field, NonTrainableState
from .._tree_math import tree_where
from ..discretization import DiscretizationBundle
from ..lifecycle import commit_candidate, TransactionalCandidate
from ..metrix import AbstractStateGeometry, EuclideanStateGeometry


def _canonical_structured_state(state: Any, /) -> PyTree[Array]:
    leaves, treedef = jax.tree.flatten(state)
    if not leaves:
        raise ValueError("Fixed-step initial_state must contain array leaves.")
    if any(not eqx.is_array(leaf) for leaf in leaves):
        raise TypeError("Every structured fixed-step state leaf must be an array.")
    arrays = tuple(jnp.asarray(leaf) for leaf in leaves)
    if not any(jnp.issubdtype(array.dtype, jnp.inexact) for array in arrays):
        raise TypeError(
            "Structured fixed-step initial_state requires at least one inexact leaf."
        )
    return jax.tree.unflatten(treedef, arrays)


def _state_dtype(state: PyTree[Array], /):
    dtypes = tuple(
        leaf.dtype
        for leaf in jax.tree.leaves(state)
        if jnp.issubdtype(leaf.dtype, jnp.inexact)
    )
    if not dtypes:
        raise TypeError("Fixed-step state requires at least one inexact leaf.")
    return jnp.result_type(*dtypes)


def _validate_result_state(
    role: str, candidate: PyTree[Array], reference: PyTree[Array], /
) -> None:
    if jax.tree.structure(candidate) != jax.tree.structure(reference):
        raise ValueError(f"Fixed-step {role} must preserve the state PyTree structure.")
    for proposed, current in zip(
        jax.tree.leaves(candidate), jax.tree.leaves(reference), strict=True
    ):
        if not eqx.is_array(proposed):
            raise TypeError(f"Every fixed-step {role} leaf must be an array.")
        if proposed.shape != current.shape or proposed.dtype != current.dtype:
            raise ValueError(
                f"Fixed-step {role} must preserve every state leaf shape and dtype."
            )


def _validate_scalar_result(role: str, value: Any, /, *, boolean: bool = False) -> None:
    if not eqx.is_array(value) or value.shape != ():
        raise TypeError(f"Fixed-step {role} must be a scalar array.")
    if boolean and value.dtype != jnp.dtype(jnp.bool_):
        raise TypeError(f"Fixed-step {role} must be Boolean.")


def _prepend_initial_state(
    initial: PyTree[Array], states: PyTree[Array], /
) -> PyTree[Array]:
    return jax.tree.map(
        lambda first, rest: jnp.concatenate((first[None, ...], rest), axis=0),
        initial,
        states,
    )


def _take_saved_states(states: PyTree[Array], indices: Array, /) -> PyTree[Array]:
    return jax.tree.map(lambda leaf: leaf[indices], states)


def _correction_dtype(state: PyTree[Array], /):
    return jnp.finfo(_state_dtype(state)).dtype


def _validate_transform_admissibility(value: Any, /) -> None:
    if value is None:
        return
    if not isinstance(value, AdmissibilityHeader):
        raise TypeError(
            "Fixed-step transform admissibility must be an AdmissibilityHeader or None."
        )
    if value.margin.shape != ():
        raise ValueError("Fixed-step transform admissibility must be scalar.")


class AcceptedStepTransformResult(StrictModule):
    """Transformed fixed-step candidate and its evidence.

    `applied` reports whether the transform acted, `successful` whether it
    succeeded, and `correction_norm` the size of the change it made.
    `admissibility` carries the native-check evidence of a learned proposal the
    transform evaluated (`None` when it evaluates none).
    """

    transformed_state: PyTree[Array]
    applied: Array
    successful: Array
    correction_norm: Array
    admissibility: AdmissibilityHeader | None = None


class AbstractAcceptedStepTransform(AbstractComponentSlot):
    """Slot transforming a fixed-step candidate before the method accepts it.

    Implementations carry `DISCRETIZATION` authority: structural invariants of
    the transformed step certify them, not a residual. The consuming method
    validates every result centrally, exactly as SSP stage results are
    validated: the transformed state preserves the candidate's structure,
    shapes, and dtypes; `applied` and `successful` are Boolean scalars; the
    correction norm is a scalar; admissibility evidence is scalar. A failed
    transform cannot change the candidate.
    """

    component_authority: ClassVar[ComponentAuthority] = ComponentAuthority.DISCRETIZATION
    slot_semantic_id: ClassVar[str] = "solver.accepted-step-transform"
    transform_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def apply(
        self,
        step_index: Array,
        time: Array,
        previous_state: PyTree[Array],
        candidate_state: PyTree[Array],
        args: Any,
        /,
    ) -> AcceptedStepTransformResult:
        raise NotImplementedError


def _apply_accepted_step_transform(
    transform: AbstractAcceptedStepTransform,
    step_index: Array,
    time: Array,
    previous_state: PyTree[Array],
    candidate_state: PyTree[Array],
    args: Any,
    /,
) -> AcceptedStepTransformResult:
    """Apply `transform` and return its centrally validated result.

    The correction norm is cast to the state's real dtype, and an unsuccessful
    result carries `candidate_state` unchanged whatever state it proposed.
    """
    result = transform.apply(step_index, time, previous_state, candidate_state, args)
    if not isinstance(result, AcceptedStepTransformResult):
        raise TypeError(
            "Accepted-step transforms must return AcceptedStepTransformResult."
        )
    _validate_result_state("transformed_state", result.transformed_state, candidate_state)
    _validate_scalar_result("transform applied", result.applied, boolean=True)
    _validate_scalar_result("transform successful", result.successful, boolean=True)
    _validate_scalar_result("transform correction", result.correction_norm)
    _validate_transform_admissibility(result.admissibility)
    return AcceptedStepTransformResult(
        tree_where(result.successful, result.transformed_state, candidate_state),
        result.applied,
        result.successful,
        jnp.asarray(result.correction_norm, dtype=_correction_dtype(candidate_state)),
        result.admissibility,
    )


class IdentityAcceptedStepTransform(AbstractAcceptedStepTransform, NonTrainableState):
    transform_id: str = "accepted-step-transform:identity"

    def apply(
        self,
        step_index: Array,
        time: Array,
        previous_state: PyTree[Array],
        candidate_state: PyTree[Array],
        args: Any,
        /,
    ) -> AcceptedStepTransformResult:
        del step_index, time, previous_state, args
        return AcceptedStepTransformResult(
            candidate_state,
            jnp.asarray(False),
            jnp.asarray(True),
            jnp.zeros((), dtype=_correction_dtype(candidate_state)),
        )


class CompositeAcceptedStepTransform(AbstractAcceptedStepTransform):
    transforms: tuple[AbstractAcceptedStepTransform, ...]
    transform_id: str = eqx.field(static=True)

    def __init__(self, transforms: Sequence[AbstractAcceptedStepTransform], /):
        values = tuple(transforms)
        if any(not isinstance(value, AbstractAcceptedStepTransform) for value in values):
            raise TypeError("Every transform must be an AbstractAcceptedStepTransform.")
        self.transforms = values
        self.transform_id = canonical_fingerprint(
            {
                "kind": "composite-accepted-step-transform",
                "transforms": [value.transform_id for value in values],
            }
        )

    def apply(
        self,
        step_index: Array,
        time: Array,
        previous_state: PyTree[Array],
        candidate_state: PyTree[Array],
        args: Any,
        /,
    ) -> AcceptedStepTransformResult:
        state = candidate_state
        applied = jnp.asarray(False)
        successful = jnp.asarray(True)
        correction = jnp.zeros((), dtype=_correction_dtype(candidate_state))
        headers: list[AdmissibilityHeader] = []
        for transform in self.transforms:
            result = _apply_accepted_step_transform(
                transform, step_index, time, previous_state, state, args
            )
            state = result.transformed_state
            applied = applied | result.applied
            successful = successful & result.successful
            correction = correction + result.correction_norm
            if result.admissibility is not None:
                headers.append(result.admissibility)
        admissibility = (
            None
            if not headers
            else headers[0]
            if len(headers) == 1
            else combine_admissibility(headers, self.transform_id)
        )
        return AcceptedStepTransformResult(
            state, applied, successful, correction, admissibility
        )


class LearnedStepCorrectionReason(IntFlag):
    """Domain reason bits of a learned step correction rejected by native checks.

    A nonfinite proposal or one outside the declared support sets the common
    `AdmissibilityReason` bits instead.
    """

    CONSERVATION = 1 << DOMAIN_REASON_SHIFT
    LOWER_BOUND = 1 << (DOMAIN_REASON_SHIFT + 1)
    STABILITY_BOUND = 1 << (DOMAIN_REASON_SHIFT + 2)


# Admitting a proposal is a runtime decision; derivatives hold it frozen.
_CORRECTION_DERIVATIVE_CONTRACT = branch_policy_contract(
    BranchDifferentiationPolicy.FROZEN_DECISION,
    surfaces=(DerivativeSurface.PRIMAL_STATE, DerivativeSurface.MODEL_PARAMETER),
)


def _model_size(value: Any, /) -> int:
    match value:
        case "scalar":
            return 1
        case tuple():
            return prod(value)
        case _:
            return index(value)


def _conservation_weights(conserved: Any, size: int, /) -> np.ndarray | None:
    if conserved is None:
        return None
    weights = np.asarray(conserved, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] == 0 or weights.shape[1] != size:
        raise ValueError("conserved must have shape (invariant_count, state_size).")
    if not np.all(np.isfinite(weights)):
        raise ValueError("conserved weights must be finite.")
    return weights


def _lower_bounds(bounds: Any, shape: tuple[int, ...], /) -> np.ndarray | None:
    if bounds is None:
        return None
    values = np.broadcast_to(np.asarray(bounds, dtype=np.float64), shape)
    if np.any(np.isnan(values)) or np.any(values == np.inf):
        raise ValueError("lower_bounds must be finite or -inf.")
    return values


def _positive_float(value: Any, name: str, /) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _euclidean_norm(value: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(jnp.square(value)))


@final
class LearnedStepCorrection(AbstractAcceptedStepTransform):
    """Learned correction of a native fixed-step candidate, admitted natively.

    From the accepted state `y` the native method proposes the coarse candidate
    `c`. The model maps the flattened pair `(y, c)` (`in_size == 2 n`) to an
    increment `d` (`out_size == n`), proposing `p = c + d`. Native checks then
    admit or reject `p` as one transaction:

    - `p` is finite and lies in `support` (default: finite Euclidean states);
    - each declared conservation invariant (row `w` of `conserved`) satisfies
      `|w . (p - c)| <= conservation_tolerance * (|w| . |c|)`; the tolerance
      defaults to the square root of the state dtype's machine epsilon;
    - `p >= lower_bounds` componentwise (`0` expresses positivity, `-inf`
      leaves a component unconstrained);
    - stability bound `||p - c|| <= maximum_relative_correction * ||c - y||`:
      the correction stays a bounded perturbation of the native increment.

    An admitted proposal is committed. A rejected one leaves `c` unchanged and
    reports its reason bits (`AdmissibilityReason`,
    `LearnedStepCorrectionReason`) in the result's admissibility header; the
    transform never retries and never fails the step, so `c` remains subject to
    its method's own acceptance. These checks are admissibility conditions, not
    an accuracy certificate: the native coarse residual or error says nothing
    about the accuracy of `p`.

    The model is a dynamic child whose parameters train through discrete
    rollouts (`DISCRETIZATION` authority, rollout objectives, e.g. checkpointed
    `FixedStepRolloutPlan` scans). Rollout derivatives hold every accept/reject
    decision frozen (`derivative_contract`); the checks and the reported
    correction norm are nondifferentiable evidence.

    `ports` declare the scientific identity of the correction's values: the
    accepted state and the native candidate (two distinct ports, each of event
    shape `state_shape`) as inputs, and the increment (event shape
    `state_shape`) as the output. A model declaring ports requires `ports` and
    an explicit `port_mapping` binding its ordered ports to exactly that owner
    order (values are packed, never repacked); a model without ports keeps the
    size checks alone. `component_contract().port_binding` holds the evidence.
    """

    model: AbstractArrayModel
    ports: ModelPorts | None
    port_mapping: PortMapping | None
    support: AbstractStateGeometry = fixed_field()
    conserved: Array | None = fixed_field()
    lower_bounds: Array | None = fixed_field()
    conservation_tolerance: float | None = eqx.field(static=True)
    maximum_relative_correction: float = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: AbstractArrayModel,
        /,
        *,
        state_shape: Sequence[int],
        maximum_relative_correction: float,
        support: AbstractStateGeometry | None = None,
        conserved: Any = None,
        conservation_tolerance: float | None = None,
        lower_bounds: Any = None,
        ports: ModelPorts | None = None,
        port_mapping: PortMapping | None = None,
    ):
        if not isinstance(model, AbstractArrayModel):
            raise TypeError("model must be an AbstractArrayModel.")
        shape = tuple(index(extent) for extent in state_shape)
        if any(extent <= 0 for extent in shape):
            raise ValueError("state_shape extents must be positive.")
        size = prod(shape)
        if _model_size(model.in_size) != 2 * size or _model_size(model.out_size) != size:
            raise ValueError(
                "LearnedStepCorrection models map 2 * state size features "
                "(accepted state, native candidate) to one increment per state entry."
            )
        support_ = EuclideanStateGeometry() if support is None else support
        if not isinstance(support_, AbstractStateGeometry):
            raise TypeError("support must be an AbstractStateGeometry or None.")
        weights = _conservation_weights(conserved, size)
        tolerance = (
            None
            if conservation_tolerance is None
            else _positive_float(conservation_tolerance, "conservation_tolerance")
        )
        if tolerance is not None and weights is None:
            raise ValueError("conservation_tolerance requires conserved invariants.")
        bounds = _lower_bounds(lower_bounds, shape)
        maximum = _positive_float(
            maximum_relative_correction, "maximum_relative_correction"
        )
        site = "LearnedStepCorrection"
        owner_ports = require_port_shapes(
            ports, inputs=(shape, shape), outputs=(shape,), site=site
        )
        component = bind_positional_component(
            model, type(self), owner_ports, port_mapping, site=site
        ).contract()
        self.model = model
        self.ports = owner_ports
        self.port_mapping = port_mapping
        self.support = support_
        self.conserved = None if weights is None else jnp.asarray(weights)
        self.lower_bounds = None if bounds is None else jnp.asarray(bounds)
        self.conservation_tolerance = tolerance
        self.maximum_relative_correction = maximum
        self.state_shape = shape
        self.transform_id = canonical_fingerprint(
            {
                "kind": "learned-step-correction",
                "component": component.bound_semantic_id,
                "ports": None if owner_ports is None else owner_ports.ports_id,
                "state_shape": list(shape),
                "support": support_.geometry_id,
                "conserved": weights,
                "conservation_tolerance": tolerance,
                "lower_bounds": bounds,
                "maximum_relative_correction": maximum,
                "derivative_contract": _CORRECTION_DERIVATIVE_CONTRACT.contract_id,
            }
        )

    @property
    def derivative_contract(self) -> DerivativeContract:
        """Frozen-decision derivatives of the committed state."""
        return _CORRECTION_DERIVATIVE_CONTRACT

    def component_contract(self) -> ComponentContract:
        """Contract of the model bound to this slot."""
        return bind_component(
            self.model,
            type(self),
            owner_ports=self.ports,
            port_mapping=self.port_mapping,
        ).contract()

    def _states(self, previous_state: Any, candidate_state: Any, /):
        if not eqx.is_array(previous_state) or not eqx.is_array(candidate_state):
            raise TypeError("LearnedStepCorrection requires array states.")
        if candidate_state.shape != self.state_shape:
            raise ValueError("Fixed-step candidate does not match state_shape.")
        if (
            previous_state.shape != candidate_state.shape
            or previous_state.dtype != candidate_state.dtype
        ):
            raise ValueError("Accepted state and candidate must share shape and dtype.")
        if not jnp.issubdtype(candidate_state.dtype, jnp.floating):
            raise TypeError("LearnedStepCorrection requires real floating states.")
        return previous_state, candidate_state

    def _checks(self, previous: Array, native: Array, proposed: Array, /):
        correction = proposed - native
        finite = jnp.all(jnp.isfinite(proposed))
        supported = jnp.asarray(self.support.contains(proposed), dtype=jnp.bool_)
        stable = _euclidean_norm(correction) <= (
            self.maximum_relative_correction * _euclidean_norm(native - previous)
        )
        conserved = jnp.asarray(True)
        if self.conserved is not None:
            tolerance = (
                np.sqrt(np.finfo(native.dtype).eps)
                if self.conservation_tolerance is None
                else self.conservation_tolerance
            )
            weights = self.conserved.astype(native.dtype)
            conserved = jnp.all(
                jnp.abs(weights @ correction.reshape(-1))
                <= tolerance * (jnp.abs(weights) @ jnp.abs(native.reshape(-1)))
            )
        bounded = (
            jnp.asarray(True)
            if self.lower_bounds is None
            else jnp.all(proposed >= self.lower_bounds.astype(native.dtype))
        )
        admitted = finite & supported & conserved & bounded & stable
        reasons = (
            reason_bits_where(finite, AdmissibilityReason.NONFINITE)
            | reason_bits_where(supported, AdmissibilityReason.OUTSIDE_SUPPORT)
            | reason_bits_where(conserved, LearnedStepCorrectionReason.CONSERVATION)
            | reason_bits_where(bounded, LearnedStepCorrectionReason.LOWER_BOUND)
            | reason_bits_where(stable, LearnedStepCorrectionReason.STABILITY_BOUND)
        )
        return admitted, reasons

    def propose(
        self,
        step_index: Array,
        previous_state: Array,
        candidate_state: Array,
        /,
    ) -> TransactionalCandidate[Array, AdmissibilityHeader]:
        """Return the natively checked transaction of one learned correction.

        Its source is the native candidate, its proposal the corrected state,
        and its evidence the admissibility header; `commit_candidate` commits
        the proposal only when every check admits it.
        """
        previous, native = self._states(previous_state, candidate_state)
        binding = self.model.input_binding()
        point = binding.pack_point((previous.reshape(-1), native.reshape(-1)))
        increment = jnp.asarray(
            binding.call(self.model, point, key=None, iter_=step_index, kwargs={}),
            dtype=native.dtype,
        )
        if increment.size != native.size:
            raise ValueError(
                "LearnedStepCorrection models must return one increment per state entry."
            )
        proposed = native + increment.reshape(native.shape)
        admitted, reasons = self._checks(
            *jax.lax.stop_gradient((previous, native, proposed))
        )
        header = AdmissibilityHeader(
            jnp.where(admitted, 1.0, -1.0),
            reasons,
            self.model.model_execution_contract().evidence_model_id,
            self.transform_id,
        )
        return TransactionalCandidate(
            native, proposed, header, admitted, self.transform_id
        )

    def apply(
        self,
        step_index: Array,
        time: Array,
        previous_state: PyTree[Array],
        candidate_state: PyTree[Array],
        args: Any,
        /,
    ) -> AcceptedStepTransformResult:
        del time, args
        transaction = self.propose(step_index, previous_state, candidate_state)
        committed = commit_candidate(transaction)
        return AcceptedStepTransformResult(
            committed.state,
            committed.committed,
            jnp.asarray(True),
            _euclidean_norm(jax.lax.stop_gradient(committed.state - transaction.source)),
            committed.evidence,
        )


class IdentitySSPRKStageTransform(AbstractSSPRKStageTransform, NonTrainableState):
    transform_id: str = "ssprk-stage-transform:identity"

    def apply(
        self,
        stage_index: int,
        time: Array,
        candidate_state: Array,
        args: Any,
        /,
    ) -> StageTransformResult:
        del stage_index, time, args
        return StageTransformResult(
            candidate_state,
            jnp.asarray(False),
            jnp.asarray(True),
            jnp.zeros((), dtype=candidate_state.real.dtype),
        )


class CallableSSPRKStageTransform(AbstractSSPRKStageTransform, NonTrainableState):
    transform: Callable[[int, Array, Array, Any], StageTransformResult] = eqx.field(
        static=True
    )
    transform_id: str = eqx.field(static=True)

    def __init__(self, transform, transform_id: str, /):
        if not callable(transform):
            raise TypeError("transform must be callable.")
        identifier = str(transform_id)
        if not identifier:
            raise ValueError("transform_id must be non-empty.")
        self.transform = transform
        self.transform_id = identifier

    def apply(
        self,
        stage_index: int,
        time: Array,
        candidate_state: Array,
        args: Any,
        /,
    ) -> StageTransformResult:
        result = self.transform(stage_index, time, candidate_state, args)
        if not isinstance(result, StageTransformResult):
            raise TypeError("Callable stage transforms must return StageTransformResult.")
        return result


class FixedStepResult(StrictModule):
    """One fixed-step attempt: candidate, accepted state, and step evidence.

    `transform_admissibility` is the scalar native-check evidence of a learned
    proposal evaluated by the method's accepted-step transform (`None` when the
    method evaluates none).
    """

    candidate_state: PyTree[Array]
    accepted_state: PyTree[Array]
    successful: Array
    residual: Array
    iterations: Array
    work: Array
    transform_applied: Array
    transform_correction_norm: Array
    transform_admissibility: AdmissibilityHeader | None = None


class RobustRetryPolicy(StrictModule, NonTrainableState):
    maximum_retries: int = eqx.field(static=True)
    reduction_factor: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_retries: int = 4,
        reduction_factor: float = 0.5,
    ):
        retries = int(maximum_retries)
        factor = float(reduction_factor)
        if retries < 0 or not 0.0 < factor < 1.0:
            raise ValueError("Retry policy requires retries>=0 and reduction in (0,1).")
        self.maximum_retries = retries
        self.reduction_factor = factor
        self.policy_id = canonical_fingerprint(
            {
                "kind": "robust-fixed-step-retry-policy",
                "maximum_retries": retries,
                "reduction_factor": factor,
                "differentiability": "branchwise",
            }
        )


class RetriedFixedStepResult(StrictModule):
    candidate_state: PyTree[Array]
    accepted_state: PyTree[Array]
    successful: Array
    accepted_step_size: Array
    retry_count: Array
    attempted_step_sizes: Array
    decision_id: str = eqx.field(static=True)


class AbstractFixedStepMethod(StrictModule):
    method_id: eqx.AbstractVar[str]

    @property
    def required_step_size(self) -> float | None:
        """Return an exact method step size, or ``None`` for an unrestricted method."""

        return None

    @property
    def allows_step_reduction(self) -> bool:
        """Whether retry and event alignment may reduce the proposed step."""

        return True

    @property
    def schedule_alignment_tolerance(self) -> float | None:
        """Optional absolute-time tolerance owned by an exact-knot method."""

        return None

    @abc.abstractmethod
    def step(
        self,
        step_index: Array,
        time: Array,
        state: PyTree[Array],
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        raise NotImplementedError


class CallableFixedStepMethod(AbstractFixedStepMethod):
    step_function: Callable[[Array, Array, PyTree[Array], Array, Any], FixedStepResult]
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        step_function: Callable[
            [Array, Array, PyTree[Array], Array, Any], FixedStepResult
        ],
        method_id: str,
        /,
    ):
        if not callable(step_function):
            raise TypeError("step_function must be callable.")
        identifier = str(method_id)
        if not identifier:
            raise ValueError("method_id must be non-empty.")
        self.step_function = step_function
        self.method_id = identifier

    def step(
        self,
        step_index: Array,
        time: Array,
        state: PyTree[Array],
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        result = self.step_function(step_index, time, state, step_size, args)
        if not isinstance(result, FixedStepResult):
            raise TypeError("step_function must return FixedStepResult.")
        return result


class AbstractSSPRKFixedStepMethod(AbstractFixedStepMethod):
    vector_field: Callable[[Array, Array, Any], Array]
    transform: AbstractAcceptedStepTransform
    stage_transform: AbstractSSPRKStageTransform
    order: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        vector_field: Callable[[Array, Array, Any], Array],
        /,
        *,
        order: int,
        transform: AbstractAcceptedStepTransform | None = None,
        stage_transform: AbstractSSPRKStageTransform | None = None,
    ):
        if not callable(vector_field):
            raise TypeError("vector_field must be callable.")
        if order not in (3, 4):
            raise ValueError("Fixed-step SSPRK order must be 3 or 4.")
        transform_ = IdentityAcceptedStepTransform() if transform is None else transform
        if not isinstance(transform_, AbstractAcceptedStepTransform):
            raise TypeError("transform must be an AbstractAcceptedStepTransform or None.")
        stage_transform_ = (
            IdentitySSPRKStageTransform() if stage_transform is None else stage_transform
        )
        if not isinstance(stage_transform_, AbstractSSPRKStageTransform):
            raise TypeError(
                "stage_transform must be AbstractSSPRKStageTransform or None."
            )
        self.vector_field = vector_field
        self.transform = transform_
        self.stage_transform = stage_transform_
        self.order = int(order)
        self.method_id = canonical_fingerprint(
            {
                "kind": "fixed-step-ssprk",
                "order": order,
                "transform": transform_.transform_id,
                "stage_transform": stage_transform_.transform_id,
            }
        )

    @abc.abstractmethod
    def _advance(
        self,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> SSPRKStepResult:
        raise NotImplementedError

    def step(
        self,
        step_index: Array,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        advanced = self._advance(time, state, step_size, args)
        candidate = advanced.state
        transformed = _apply_accepted_step_transform(
            self.transform, step_index, time + step_size, state, candidate, args
        )
        successful = (
            advanced.successful
            & transformed.successful
            & jnp.all(jnp.isfinite(transformed.transformed_state))
        )
        accepted = jnp.where(successful, transformed.transformed_state, state)
        return FixedStepResult(
            candidate,
            accepted,
            successful,
            jnp.zeros((), dtype=state.dtype),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(self.order, dtype=jnp.int32),
            advanced.applied | transformed.applied,
            jnp.maximum(advanced.correction_norm, transformed.correction_norm),
            transformed.admissibility,
        )


class SSPRK33FixedStepMethod(AbstractSSPRKFixedStepMethod):
    def __init__(
        self,
        vector_field: Callable[[Array, Array, Any], Array],
        /,
        *,
        transform: AbstractAcceptedStepTransform | None = None,
        stage_transform: AbstractSSPRKStageTransform | None = None,
    ):
        super().__init__(
            vector_field,
            order=3,
            transform=transform,
            stage_transform=stage_transform,
        )

    def _advance(self, time, state, step_size, args, /):
        return ssprk33_step_with_evidence(
            self.vector_field,
            time,
            state,
            step_size,
            args,
            stage_transform=self.stage_transform,
        )


class SSPRK54FixedStepMethod(AbstractSSPRKFixedStepMethod):
    def __init__(
        self,
        vector_field: Callable[[Array, Array, Any], Array],
        /,
        *,
        transform: AbstractAcceptedStepTransform | None = None,
        stage_transform: AbstractSSPRKStageTransform | None = None,
    ):
        super().__init__(
            vector_field,
            order=4,
            transform=transform,
            stage_transform=stage_transform,
        )

    def _advance(self, time, state, step_size, args, /):
        return ssprk54_step_with_evidence(
            self.vector_field,
            time,
            state,
            step_size,
            args,
            stage_transform=self.stage_transform,
        )


def _enforce_required_step_size(
    method: AbstractFixedStepMethod, step_size: Array, /
) -> Array:
    required = method.required_step_size
    if required is None:
        return step_size
    declared_value = float(required)
    if not np.isfinite(declared_value) or declared_value <= 0.0:
        raise ValueError(
            "Fixed-step method required_step_size must be finite and positive."
        )
    declared = jnp.asarray(declared_value, dtype=step_size.dtype)
    incompatible = (
        ~jnp.isfinite(step_size) | ~jnp.isfinite(declared) | (step_size != declared)
    )
    message = "Fixed-step step_size is incompatible with method.required_step_size."
    if isinstance(incompatible, jax.core.Tracer):
        return eqx.error_if(step_size, incompatible, message)
    if bool(incompatible):
        raise ValueError(message)
    return step_size


class FixedStepProblem(StrictModule):
    method: AbstractFixedStepMethod
    initial_state: PyTree[Array] = fixed_field()
    args: Any = fixed_field()
    state_geometry: AbstractStateGeometry = fixed_field()
    discretization_bundle: DiscretizationBundle | None = fixed_field()
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: AbstractFixedStepMethod,
        initial_state: Any,
        /,
        *,
        t0: float,
        t1: float,
        step_size: float,
        args: Any = None,
        state_geometry: AbstractStateGeometry | None = None,
        discretization_bundle: DiscretizationBundle | None = None,
        problem_id: str | None = None,
    ):
        if not isinstance(method, AbstractFixedStepMethod):
            raise TypeError("method must be an AbstractFixedStepMethod.")
        if state_geometry is None:
            initial = jnp.asarray(initial_state)
            if not jnp.issubdtype(initial.dtype, jnp.inexact):
                raise TypeError("Fixed-step initial_state must have an inexact dtype.")
        else:
            initial = _canonical_structured_state(initial_state)
        start = float(t0)
        end = float(t1)
        step = float(step_size)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("Fixed-step times require finite t1 > t0.")
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        raw_steps = (end - start) / step
        count = int(round(raw_steps))
        if count <= 0 or not np.isclose(raw_steps, count, rtol=1e-12, atol=1e-12):
            raise ValueError("Fixed-step interval must contain an integer step count.")
        _enforce_required_step_size(
            method,
            jnp.asarray(step, dtype=_state_dtype(initial)),
        )
        geometry = EuclideanStateGeometry() if state_geometry is None else state_geometry
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("state_geometry must be an AbstractStateGeometry or None.")
        if discretization_bundle is not None and not isinstance(
            discretization_bundle, DiscretizationBundle
        ):
            raise TypeError(
                "discretization_bundle must be a DiscretizationBundle or None."
            )
        state_payload = (
            {
                "state_shape": list(initial.shape),
                "state_dtype": str(initial.dtype),
            }
            if eqx.is_array(initial)
            else {"state_tree": array_tree_signature(initial)}
        )
        generated = canonical_fingerprint(
            {
                "kind": "fixed-step-problem",
                "method": method.method_id,
                **state_payload,
                "t0": start,
                "t1": end,
                "step_size": step,
                "geometry": geometry.geometry_id,
                "bundle": None
                if discretization_bundle is None
                else discretization_bundle.bundle_id,
            }
        )
        identifier = generated if problem_id is None else str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.method = method
        self.initial_state = initial
        self.args = args
        self.state_geometry = geometry
        self.discretization_bundle = discretization_bundle
        self.t0 = start
        self.t1 = end
        self.step_size = step
        self.step_count = count
        self.problem_id = identifier


def retry_fixed_step(
    method: AbstractFixedStepMethod,
    policy: RobustRetryPolicy,
    step_index: Array,
    time: Array,
    state: PyTree[Array],
    step_size: Array,
    args: Any = None,
    /,
) -> RetriedFixedStepResult:
    if not isinstance(method, AbstractFixedStepMethod) or not isinstance(
        policy, RobustRetryPolicy
    ):
        raise TypeError("retry_fixed_step requires method and retry policy.")
    current_step = jnp.asarray(step_size)
    if current_step.shape != () or not jnp.issubdtype(current_step.dtype, jnp.inexact):
        raise TypeError("retry_fixed_step step_size must be an inexact scalar array.")
    current_step = _enforce_required_step_size(method, current_step)
    if policy.maximum_retries and not method.allows_step_reduction:
        raise ValueError("The fixed-step method does not permit retry step reduction.")
    initial = _canonical_structured_state(state)
    successful = jnp.asarray(False)
    selected_state = initial
    selected_candidate = initial
    accepted_step = jnp.zeros_like(current_step)
    retry_count = jnp.asarray(policy.maximum_retries, dtype=jnp.int32)
    attempted = []
    for attempt in range(policy.maximum_retries + 1):
        attempted.append(current_step)
        result = method.step(
            step_index,
            time,
            initial,
            current_step,
            args,
        )
        if not isinstance(result, FixedStepResult):
            raise TypeError("Fixed-step methods must return FixedStepResult.")
        _validate_result_state("candidate_state", result.candidate_state, initial)
        _validate_result_state("accepted_state", result.accepted_state, initial)
        _validate_scalar_result("successful", result.successful, boolean=True)
        take = (~jax.lax.stop_gradient(successful)) & jax.lax.stop_gradient(
            result.successful
        )
        selected_candidate = tree_where(take, result.candidate_state, selected_candidate)
        selected_state = tree_where(take, result.accepted_state, selected_state)
        accepted_step = jnp.where(take, current_step, accepted_step)
        retry_count = jnp.where(take, jnp.asarray(attempt, dtype=jnp.int32), retry_count)
        successful = successful | result.successful
        if attempt < policy.maximum_retries:
            current_step = current_step * policy.reduction_factor
    return RetriedFixedStepResult(
        selected_candidate,
        tree_where(successful, selected_state, initial),
        successful,
        accepted_step,
        retry_count,
        jnp.stack(tuple(attempted)),
        canonical_fingerprint(
            {
                "kind": "retried-fixed-step-decision",
                "method": method.method_id,
                "retry_policy": policy.policy_id,
            }
        ),
    )


class FixedStepSolution(StrictModule, NonTrainableState):
    """Saved fixed-step states with per-step evidence.

    Per-step transform evidence has one entry per step; `transform_admissibility`
    stacks the steps' learned-proposal evidence (`None` when the method
    evaluates no learned proposal).
    """

    times: Array
    states: PyTree[Array]
    valid: Array
    successful: Array
    residuals: Array
    iterations: Array
    work: Array
    transform_applied: Array
    transform_correction_norm: Array
    transform_admissibility: AdmissibilityHeader | None
    iteration_evidence: IterationEvidence | None
    problem_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    state_geometry_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)


FixedStepRetentionPolicy: TypeAlias = Literal["final", "checkpoints", "trajectory"]
FixedStepReplayMode: TypeAlias = Literal["full", "step", "block", "scheduled"]


class FixedStepReplayPolicy(StrictModule, NonTrainableState):
    """Reverse-mode storage and recomputation for one fixed-length step scan."""

    mode: FixedStepReplayMode = eqx.field(static=True)
    block_size: int | None = eqx.field(static=True)
    schedule: PreparedReplaySchedule | None
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: FixedStepReplayMode = "full",
        /,
        *,
        block_size: int | None = None,
        schedule: PreparedReplaySchedule | None = None,
    ):
        if mode not in ("full", "step", "block", "scheduled"):
            raise ValueError("Unknown fixed-step replay mode.")
        size = None if block_size is None else int(block_size)
        if mode == "block":
            if size is None or size <= 0:
                raise ValueError("Block replay requires a positive block_size.")
            if schedule is not None:
                raise ValueError("Block replay does not accept a prepared schedule.")
        elif mode == "scheduled":
            if size is not None:
                raise ValueError("Scheduled replay does not accept block_size.")
            if not isinstance(schedule, PreparedReplaySchedule):
                raise TypeError("Scheduled replay requires PreparedReplaySchedule.")
        elif size is not None or schedule is not None:
            raise ValueError("Replay block/schedule is incompatible with selected mode.")
        self.mode = mode
        self.block_size = size
        self.schedule = schedule
        self.policy_id = canonical_fingerprint(
            {
                "kind": "fixed-step-replay",
                "mode": mode,
                "block_size": size,
                "schedule": None if schedule is None else schedule.schedule_id,
            }
        )


class FixedStepStatus(IntEnum):
    SUCCESS = 0
    STEP_FAILURE = 1
    USER_STOPPED = 2


class FixedStepIterationMetrics(StrictModule):
    time: Array
    residual: Array
    iterations: Array
    work: Array
    transform_applied: Array
    transform_correction_norm: Array

    def __init__(
        self,
        time,
        residual,
        iterations,
        work,
        transform_applied,
        transform_correction_norm,
        /,
    ):
        self.time = jnp.asarray(time)
        self.residual = jnp.asarray(residual)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int64)
        self.work = jnp.asarray(work, dtype=jnp.int64)
        self.transform_applied = jnp.asarray(transform_applied, dtype=jnp.bool_)
        self.transform_correction_norm = jnp.asarray(transform_correction_norm)


def _fixed_step_advance(
    problem: FixedStepProblem,
    state_dtype: Any,
    carry: tuple[PyTree[Array], Array],
    step_index: Array,
    /,
):
    state, previous_success = carry
    step_size = jnp.asarray(problem.step_size, dtype=state_dtype)
    time = jnp.asarray(problem.t0, dtype=state_dtype) + step_index * step_size
    result = problem.method.step(step_index, time, state, step_size, problem.args)
    if not isinstance(result, FixedStepResult):
        raise TypeError("Fixed-step methods must return FixedStepResult.")
    _validate_result_state("candidate_state", result.candidate_state, state)
    _validate_result_state("accepted_state", result.accepted_state, state)
    _validate_scalar_result("successful", result.successful, boolean=True)
    _validate_scalar_result("residual", result.residual)
    _validate_scalar_result("iterations", result.iterations)
    _validate_scalar_result("work", result.work)
    _validate_scalar_result("transform_applied", result.transform_applied, boolean=True)
    _validate_scalar_result("transform_correction_norm", result.transform_correction_norm)
    _validate_transform_admissibility(result.transform_admissibility)
    accepted = tree_where(previous_success, result.accepted_state, state)
    successful = previous_success & result.successful
    payload = (
        successful,
        result.residual,
        result.iterations,
        result.work,
        result.transform_applied,
        result.transform_correction_norm,
        result.transform_admissibility,
    )
    return (accepted, successful), payload


def _fixed_step_iteration_record(
    phase,
    ordinal,
    active,
    successful,
    metrics,
    /,
    *,
    terminal=False,
    status=None,
) -> IterationRecord:
    committed = jnp.asarray(active, dtype=jnp.bool_) & jnp.asarray(
        successful, dtype=jnp.bool_
    )
    accepted = jnp.where(
        committed,
        jnp.asarray(ordinal, dtype=jnp.int32),
        jnp.maximum(jnp.asarray(ordinal, dtype=jnp.int32) - 1, 0),
    )
    status_ = (
        jnp.where(
            successful,
            int(FixedStepStatus.SUCCESS),
            int(FixedStepStatus.STEP_FAILURE),
        )
        if status is None
        else status
    )
    return IterationRecord(
        IterationCoordinates(
            phase,
            ordinal,
            attempt=ordinal,
            accepted=accepted,
            rejected=jnp.asarray(ordinal, dtype=jnp.int32) - accepted,
            active=active,
            committed=committed,
            terminal=terminal,
        ),
        status_,
        metrics,
    )


class FixedStepRolloutResult(StrictModule, NonTrainableState):
    """Retained fixed-step states with per-step evidence.

    Per-step transform evidence has one entry per step; `transform_admissibility`
    stacks the steps' learned-proposal evidence (`None` when the method
    evaluates no learned proposal).
    """

    final_state: PyTree[Array]
    successful: Array
    times: Array
    states: PyTree[Array]
    valid: Array
    residuals: Array
    iterations: Array
    work: Array
    transform_applied: Array
    transform_correction_norm: Array
    transform_admissibility: AdmissibilityHeader | None
    iteration_evidence: IterationEvidence | None
    problem_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    state_geometry_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class FixedStepRolloutPlan(StrictModule):
    """Fixed-step retention, replay, and transform-safe iteration observation."""

    retention: FixedStepRetentionPolicy = eqx.field(static=True)
    checkpoint_stride: int = eqx.field(static=True)
    replay: FixedStepReplayPolicy = fixed_field()
    iteration: IterationPlan | None = fixed_field()
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        retention: FixedStepRetentionPolicy = "final",
        checkpoint_stride: int = 1,
        replay: FixedStepReplayPolicy | None = None,
        iteration: IterationPlan | None = None,
    ):
        if retention not in ("final", "checkpoints", "trajectory"):
            raise ValueError("Unknown fixed-step retention policy.")
        stride = int(checkpoint_stride)
        if stride <= 0:
            raise ValueError("checkpoint_stride must be positive.")
        if retention != "checkpoints" and stride != 1:
            raise ValueError(
                "checkpoint_stride differs from one only for checkpoint retention."
            )
        replay_ = FixedStepReplayPolicy() if replay is None else replay
        if not isinstance(replay_, FixedStepReplayPolicy):
            raise TypeError("replay must be FixedStepReplayPolicy or None.")
        if iteration is not None and not isinstance(iteration, IterationPlan):
            raise TypeError("iteration must be IterationPlan or None.")
        self.retention = retention
        self.checkpoint_stride = stride
        self.replay = replay_
        self.iteration = iteration
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-step-rollout-plan",
                "retention": retention,
                "checkpoint_stride": stride,
                "replay": replay_.policy_id,
                "iteration": None if iteration is None else iteration.plan_id,
            }
        )

    def rollout(
        self,
        problem: FixedStepProblem,
        /,
    ) -> FixedStepRolloutResult:
        if not isinstance(problem, FixedStepProblem):
            raise TypeError("problem must be a FixedStepProblem.")
        state_dtype = _state_dtype(problem.initial_state)
        step_size = jnp.asarray(problem.step_size, dtype=state_dtype)
        numerical_initial = (problem.initial_state, jnp.asarray(True))
        iteration_scope = None
        iteration_capabilities = None

        if self.iteration is None:
            initial_carry = numerical_initial

            def step(carry, step_index):
                return _fixed_step_advance(problem, state_dtype, carry, step_index)

        else:
            iteration = self.iteration
            assert iteration is not None
            iteration_capabilities = IterationCapabilities(
                ("terminal", "step"),
                device_stop=True,
                mapped_records=True,
            )
            iteration_scope = bind_iteration_scope(
                iteration,
                iteration_capabilities,
                problem.method.method_id,
            )
            initial_metrics = FixedStepIterationMetrics(
                jnp.asarray(problem.t0, dtype=state_dtype),
                jnp.zeros((), dtype=state_dtype),
                jnp.asarray(0, dtype=jnp.int64),
                jnp.asarray(0, dtype=jnp.int64),
                jnp.asarray(False),
                jnp.zeros((), dtype=state_dtype),
            )
            initial_record = _fixed_step_iteration_record(
                IterationPhase.START,
                0,
                True,
                True,
                initial_metrics,
            )
            initial_carry = (
                numerical_initial,
                initialize_iteration(iteration, initial_record),
            )

            def step(carry, step_index):
                numerical, iteration_state = carry
                state, previous_success = numerical
                active = previous_success & ~iteration_state.stop_requested
                next_numerical, built_in = _fixed_step_advance(
                    problem,
                    state_dtype,
                    (state, active),
                    step_index,
                )
                (
                    successful,
                    residual,
                    iterations,
                    work,
                    transformed,
                    correction,
                    _,
                ) = built_in
                endpoint = (
                    jnp.asarray(problem.t0, dtype=state_dtype)
                    + (step_index + 1) * step_size
                )
                metrics = FixedStepIterationMetrics(
                    endpoint,
                    residual,
                    iterations,
                    work,
                    transformed,
                    correction,
                )
                phase = jnp.where(
                    successful,
                    int(IterationPhase.COMMIT),
                    int(IterationPhase.ATTEMPT),
                )
                record = _fixed_step_iteration_record(
                    phase,
                    step_index + 1,
                    active,
                    successful,
                    metrics,
                )
                next_iteration = update_iteration(
                    iteration,
                    iteration_state,
                    record,
                    allow_stop=successful,
                )
                return (next_numerical, next_iteration), built_in

        def numerical_carry(carry):
            return carry if self.iteration is None else carry[0]

        indices = jnp.arange(problem.step_count, dtype=jnp.int32)

        if self.retention == "trajectory":

            def trajectory_step(carry, step_index):
                next_carry, payload = step(carry, step_index)
                accepted, _ = numerical_carry(next_carry)
                return next_carry, (accepted, *payload)

            result_carry, payload = checkpointed_scan(
                trajectory_step,
                initial_carry,
                indices,
                length=problem.step_count,
                mode=self.replay.mode,
                block_size=self.replay.block_size,
                schedule=self.replay.schedule,
            )
            (
                states,
                valid,
                residuals,
                iterations,
                work,
                transformed,
                correction,
                admissibility,
            ) = payload
            retained_states = _prepend_initial_state(problem.initial_state, states)
            retained_valid = jnp.concatenate((jnp.asarray([True]), valid), axis=0)
            retained_times = jnp.asarray(
                problem.t0, dtype=step_size.dtype
            ) + step_size * jnp.arange(problem.step_count + 1)
        elif self.retention == "final":
            result_carry, payload = checkpointed_scan(
                step,
                initial_carry,
                indices,
                length=problem.step_count,
                mode=self.replay.mode,
                block_size=self.replay.block_size,
                schedule=self.replay.schedule,
            )
            (
                valid,
                residuals,
                iterations,
                work,
                transformed,
                correction,
                admissibility,
            ) = payload
            final_state, final_success = numerical_carry(result_carry)
            retained_states = jax.tree.map(lambda leaf: leaf[None, ...], final_state)
            retained_valid = final_success[None]
            retained_times = jnp.asarray([problem.t1], dtype=step_size.dtype)
        else:
            saved_indices = tuple(
                range(0, problem.step_count + 1, self.checkpoint_stride)
            )
            if saved_indices[-1] != problem.step_count:
                saved_indices = (*saved_indices, problem.step_count)
            save_after_step = np.zeros((problem.step_count,), dtype=np.bool_)
            for endpoint in saved_indices[1:]:
                save_after_step[endpoint - 1] = True
            save_mask = jnp.asarray(save_after_step)
            retained_states = jax.tree.map(
                lambda leaf: (
                    jnp.zeros((len(saved_indices), *leaf.shape), dtype=leaf.dtype)
                    .at[0]
                    .set(leaf)
                ),
                problem.initial_state,
            )
            retained_valid = (
                jnp.zeros((len(saved_indices),), dtype=jnp.bool_).at[0].set(True)
            )

            def checkpoint_step(carry, step_index):
                state_carry, saved, saved_valid, cursor = carry
                next_carry, payload = step(state_carry, step_index)
                accepted, successful = numerical_carry(next_carry)

                def store(values):
                    states_, valid_, cursor_ = values
                    states_ = jax.tree.map(
                        lambda buffer, value: buffer.at[cursor_].set(value),
                        states_,
                        accepted,
                    )
                    valid_ = valid_.at[cursor_].set(successful)
                    return states_, valid_, cursor_ + 1

                saved, saved_valid, cursor = jax.lax.cond(
                    save_mask[step_index],
                    store,
                    lambda values: values,
                    (saved, saved_valid, cursor),
                )
                return (next_carry, saved, saved_valid, cursor), payload

            checkpoint_carry = (
                initial_carry,
                retained_states,
                retained_valid,
                jnp.asarray(1, dtype=jnp.int32),
            )
            checkpoint_result, payload = checkpointed_scan(
                checkpoint_step,
                checkpoint_carry,
                indices,
                length=problem.step_count,
                mode=self.replay.mode,
                block_size=self.replay.block_size,
                schedule=self.replay.schedule,
            )
            result_carry, retained_states, retained_valid, _ = checkpoint_result
            (
                valid,
                residuals,
                iterations,
                work,
                transformed,
                correction,
                admissibility,
            ) = payload
            retained_times = jnp.asarray(
                problem.t0, dtype=step_size.dtype
            ) + step_size * jnp.asarray(saved_indices, dtype=step_size.dtype)

        final_state, final_success = numerical_carry(result_carry)
        iteration_evidence = None
        result_success = final_success
        if self.iteration is not None:
            assert iteration_scope is not None
            assert iteration_capabilities is not None
            iteration_state = result_carry[1]
            last = iteration_state.last
            completed = last.coordinates.ordinal >= problem.step_count
            terminal_status = jnp.where(
                iteration_state.stop_requested & ~completed,
                int(FixedStepStatus.USER_STOPPED),
                jnp.where(
                    final_success,
                    int(FixedStepStatus.SUCCESS),
                    int(FixedStepStatus.STEP_FAILURE),
                ),
            )
            result_success = terminal_status == int(FixedStepStatus.SUCCESS)
            terminal_record = IterationRecord(
                IterationCoordinates(
                    IterationPhase.TERMINAL,
                    last.coordinates.ordinal,
                    invocation=last.coordinates.invocation,
                    attempt=last.coordinates.attempt,
                    accepted=last.coordinates.accepted,
                    rejected=last.coordinates.rejected,
                    active=True,
                    committed=last.coordinates.committed,
                    terminal=True,
                ),
                terminal_status,
                last.metrics,
            )
            iteration_evidence = finalize_iteration(
                self.iteration,
                iteration_scope,
                iteration_capabilities,
                iteration_state,
                terminal_record,
            )

        bundle_id = (
            None
            if problem.discretization_bundle is None
            else problem.discretization_bundle.bundle_id
        )
        return FixedStepRolloutResult(
            final_state,
            result_success,
            retained_times,
            retained_states,
            retained_valid,
            residuals,
            iterations,
            work,
            transformed,
            correction,
            admissibility,
            iteration_evidence,
            problem.problem_id,
            problem.method.method_id,
            problem.state_geometry.geometry_id,
            bundle_id,
            self.plan_id,
        )


def solve_fixed_step(
    problem: FixedStepProblem,
    /,
    *,
    save_every: int = 1,
    replay: FixedStepReplayPolicy | None = None,
    iteration: IterationPlan | None = None,
) -> FixedStepSolution:
    """Run one pure fixed-step scan with orthogonal saving and observation."""

    if not isinstance(problem, FixedStepProblem):
        raise TypeError("problem must be a FixedStepProblem.")
    stride = int(save_every)
    if stride <= 0:
        raise ValueError("save_every must be positive.")
    retention: FixedStepRetentionPolicy = "trajectory" if stride == 1 else "checkpoints"
    rollout = FixedStepRolloutPlan(
        retention=retention,
        checkpoint_stride=stride,
        replay=replay,
        iteration=iteration,
    ).rollout(problem)
    return FixedStepSolution(
        rollout.times,
        rollout.states,
        rollout.valid,
        rollout.successful,
        rollout.residuals,
        rollout.iterations,
        rollout.work,
        rollout.transform_applied,
        rollout.transform_correction_norm,
        rollout.transform_admissibility,
        rollout.iteration_evidence,
        rollout.problem_id,
        rollout.method_id,
        rollout.state_geometry_id,
        rollout.discretization_bundle_id,
    )


__all__ = [
    "AdaptiveReplayPreparationPolicy",
    "AbstractAcceptedStepTransform",
    "AbstractSSPRKStageTransform",
    "CallableFixedStepMethod",
    "CallableSSPRKStageTransform",
    "AbstractFixedStepMethod",
    "AcceptedStepTransformResult",
    "CompositeAcceptedStepTransform",
    "FixedStepProblem",
    "FixedStepReplayMode",
    "FixedStepReplayPolicy",
    "PreparedReplaySchedule",
    "prepare_replay_schedule",
    "FixedStepRetentionPolicy",
    "FixedStepRolloutPlan",
    "FixedStepRolloutResult",
    "FixedStepIterationMetrics",
    "FixedStepStatus",
    "FixedStepResult",
    "RetriedFixedStepResult",
    "RobustRetryPolicy",
    "retry_fixed_step",
    "FixedStepSolution",
    "IdentityAcceptedStepTransform",
    "IdentitySSPRKStageTransform",
    "LearnedStepCorrection",
    "LearnedStepCorrectionReason",
    "SSPRK33FixedStepMethod",
    "SSPRK54FixedStepMethod",
    "StageTransformResult",
    "solve_fixed_step",
]
