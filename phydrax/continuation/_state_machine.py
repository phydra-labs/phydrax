#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._tree_math import validate_inexact_tree, validate_real_inexact_tree
from ._geometry import ContinuationRepresentationPolicy


def _parameter_paths(tree: PyTree[Any], /) -> tuple[str, ...]:
    return tuple(
        jax.tree_util.keystr(path) or "<root>"
        for path, _ in jax.tree_util.tree_flatten_with_path(tree)[0]
    )


def _tree_content_id(kind: str, tree: PyTree[Any], /) -> str:
    return canonical_fingerprint(
        {
            "kind": kind,
            "tree": array_tree_fingerprint(tree),
        }
    )


class ParameterRealization(StrictModule):
    """One immutable, content-addressed realization of a physical parameter path."""

    coordinate: Array
    parameters: PyTree[Array]
    finite: Array
    problem_id: str = eqx.field(static=True)
    parameter_paths: tuple[str, ...] = eqx.field(static=True)
    path_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameters: PyTree[Any],
        coordinate: Any,
        /,
        *,
        problem_id: str,
    ):
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("Parameter realization problem_id must be non-empty.")
        coordinate_ = jnp.asarray(coordinate)
        if coordinate_.shape != () or not jnp.issubdtype(coordinate_.dtype, jnp.floating):
            raise TypeError("Parameter realization coordinate must be a real scalar.")
        parameters_ = validate_real_inexact_tree(
            parameters,
            name="continuation parameter realization",
        )
        paths = _parameter_paths(parameters_)
        if not paths:
            raise ValueError("A parameter realization must contain at least one leaf.")
        finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(parameters_)
                )
            )
        ) & jnp.isfinite(coordinate_)
        path_id = canonical_fingerprint(
            {
                "kind": "continuation-parameter-path",
                "problem": identifier,
                "paths": paths,
                "signature": array_tree_fingerprint(parameters_)["signature"],
            }
        )
        realization_id = canonical_fingerprint(
            {
                "kind": "continuation-parameter-realization",
                "problem": identifier,
                "path": path_id,
                "coordinate": array_tree_fingerprint(coordinate_),
                "parameters": array_tree_fingerprint(parameters_),
            }
        )
        self.coordinate = coordinate_
        self.parameters = parameters_
        self.finite = jnp.asarray(finite, dtype=bool)
        self.problem_id = identifier
        self.parameter_paths = paths
        self.path_id = path_id
        self.realization_id = realization_id


class ContinuationCandidate(StrictModule):
    """Numerical candidate isolated from application state until final acceptance."""

    state: PyTree[Array]
    coordinate: Array
    tangent_state: PyTree[Array]
    tangent_coordinate: Array
    tangent_parameters: PyTree[Array]
    residual_norm: Array
    step_size: Array
    tangent_residual_norm: Array
    tangent_alignment: Array
    corrector_iterations: Array
    corrector_status: Array
    tangent_status: Array
    realization: ParameterRealization
    numerical_accepted: Array
    point_id: str = eqx.field(static=True)
    parent_point_id: str = eqx.field(static=True)
    attempt_index: int = eqx.field(static=True)
    retry_index: int = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        state: PyTree[Any],
        coordinate: Any,
        tangent_state: PyTree[Any],
        tangent_coordinate: Any,
        tangent_parameters: PyTree[Any],
        residual_norm: Any,
        step_size: Any,
        tangent_residual_norm: Any,
        tangent_alignment: Any,
        corrector_iterations: Any,
        corrector_status: Any,
        tangent_status: Any,
        realization: ParameterRealization,
        numerical_accepted: Any,
        point_id: str,
        parent_point_id: str = "",
        attempt_index: int,
        retry_index: int,
    ):
        if not isinstance(realization, ParameterRealization):
            raise TypeError("realization must be a ParameterRealization.")
        state_ = validate_inexact_tree(state, name="continuation candidate state")
        tangent_ = validate_inexact_tree(
            tangent_state,
            name="continuation candidate tangent",
        )
        if jax.tree.structure(state_) != jax.tree.structure(tangent_):
            raise ValueError("Candidate state and tangent structures must match.")
        tangent_parameters_ = validate_real_inexact_tree(
            tangent_parameters,
            name="candidate parameter tangent",
        )
        if jax.tree.structure(realization.parameters) != jax.tree.structure(
            tangent_parameters_
        ):
            raise ValueError(
                "Candidate parameter realization and tangent structures must match."
            )
        identifier = str(point_id)
        if not identifier:
            raise ValueError("Continuation candidate point_id must be non-empty.")
        attempt = int(attempt_index)
        retry = int(retry_index)
        if attempt < 0 or retry < 0:
            raise ValueError("Candidate attempt and retry indices must be non-negative.")
        coordinate_ = jnp.asarray(coordinate)
        tangent_coordinate_ = jnp.asarray(tangent_coordinate)
        scalar_values = tuple(
            jnp.asarray(value)
            for value in (
                residual_norm,
                step_size,
                tangent_residual_norm,
                tangent_alignment,
                corrector_iterations,
                corrector_status,
                tangent_status,
                numerical_accepted,
            )
        )
        if (
            coordinate_.shape != ()
            or tangent_coordinate_.shape != ()
            or any(value.shape != () for value in scalar_values)
        ):
            raise ValueError(
                "Candidate coordinates, diagnostics, and gates must be scalar."
            )
        if not bool(jnp.array_equal(coordinate_, realization.coordinate)):
            raise ValueError("Candidate coordinate and parameter realization must match.")
        numerical = jnp.asarray(numerical_accepted, dtype=bool)
        candidate_id = canonical_fingerprint(
            {
                "kind": "continuation-candidate",
                "point": identifier,
                "parent": str(parent_point_id),
                "attempt": attempt,
                "retry": retry,
                "state": array_tree_fingerprint(state_),
                "coordinate": array_tree_fingerprint(coordinate_),
                "tangent_state": array_tree_fingerprint(tangent_),
                "tangent_coordinate": array_tree_fingerprint(tangent_coordinate_),
                "realization": realization.realization_id,
                "numerical_accepted": bool(numerical),
            }
        )
        self.state = state_
        self.coordinate = coordinate_
        self.tangent_state = tangent_
        self.tangent_coordinate = tangent_coordinate_
        self.tangent_parameters = tangent_parameters_
        self.residual_norm = scalar_values[0]
        self.step_size = scalar_values[1]
        self.tangent_residual_norm = scalar_values[2]
        self.tangent_alignment = scalar_values[3]
        self.corrector_iterations = jnp.asarray(scalar_values[4], dtype=jnp.int32)
        self.corrector_status = jnp.asarray(scalar_values[5], dtype=jnp.int32)
        self.tangent_status = jnp.asarray(scalar_values[6], dtype=jnp.int32)
        self.realization = realization
        self.numerical_accepted = numerical
        self.point_id = identifier
        self.parent_point_id = str(parent_point_id)
        self.attempt_index = attempt
        self.retry_index = retry
        self.candidate_id = candidate_id

    @property
    def parameters(self) -> PyTree[Array]:
        return self.realization.parameters


class ParameterTransferEvidence(StrictModule):
    """Numerical evidence that one immutable parameter realization was transferred."""

    evaluated: Array
    paths_match: Array
    finite: Array
    application_accepted: Array
    source_realization_id: str = eqx.field(static=True)
    target_realization_id: str = eqx.field(static=True)
    parameter_paths: tuple[str, ...] = eqx.field(static=True)
    message: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        evaluated: Any,
        paths_match: Any,
        finite: Any,
        application_accepted: Any,
        source_realization_id: str,
        target_realization_id: str,
        parameter_paths: tuple[str, ...],
        message: str = "",
    ):
        target = str(target_realization_id)
        paths = tuple(str(path) for path in parameter_paths)
        if not target or not paths or any(not path for path in paths):
            raise ValueError(
                "Transfer evidence requires a target ID and parameter paths."
            )
        flags = tuple(
            jnp.asarray(value, dtype=bool)
            for value in (evaluated, paths_match, finite, application_accepted)
        )
        if any(value.shape != () for value in flags):
            raise ValueError("Parameter transfer evidence flags must be scalar.")
        message_ = str(message)
        evidence_id = canonical_fingerprint(
            {
                "kind": "continuation-parameter-transfer",
                "source": str(source_realization_id),
                "target": target,
                "paths": paths,
                "evaluated": bool(flags[0]),
                "paths_match": bool(flags[1]),
                "finite": bool(flags[2]),
                "application_accepted": bool(flags[3]),
                "message": message_,
            }
        )
        (
            self.evaluated,
            self.paths_match,
            self.finite,
            self.application_accepted,
        ) = flags
        self.source_realization_id = str(source_realization_id)
        self.target_realization_id = target
        self.parameter_paths = paths
        self.message = message_
        self.evidence_id = evidence_id

    @property
    def accepted(self) -> Array:
        return self.evaluated & self.paths_match & self.finite & self.application_accepted

    @classmethod
    def for_candidate(
        cls,
        source: ContinuationAcceptedState | None,
        candidate: ContinuationCandidate,
        /,
        *,
        application_accepted: Any = True,
        message: str = "",
    ) -> ParameterTransferEvidence:
        source_id = "" if source is None else source.realization.realization_id
        source_paths = (
            candidate.realization.parameter_paths
            if source is None
            else source.realization.parameter_paths
        )
        return cls(
            evaluated=True,
            paths_match=source_paths == candidate.realization.parameter_paths,
            finite=candidate.realization.finite,
            application_accepted=application_accepted,
            source_realization_id=source_id,
            target_realization_id=candidate.realization.realization_id,
            parameter_paths=candidate.realization.parameter_paths,
            message=message,
        )

    @classmethod
    def not_evaluated(
        cls,
        source: ContinuationAcceptedState | None,
        candidate: ContinuationCandidate,
        /,
        *,
        message: str,
    ) -> ParameterTransferEvidence:
        source_id = "" if source is None else source.realization.realization_id
        return cls(
            evaluated=False,
            paths_match=False,
            finite=candidate.realization.finite,
            application_accepted=False,
            source_realization_id=source_id,
            target_realization_id=candidate.realization.realization_id,
            parameter_paths=candidate.realization.parameter_paths,
            message=message,
        )


class ContinuationAcceptedState(StrictModule):
    """One committed numerical/application state at an accepted branch point."""

    candidate: ContinuationCandidate
    application_state: Any
    realization: ParameterRealization
    application_state_id: str = eqx.field(static=True)
    decision_id: str = eqx.field(static=True)
    accepted_index: int = eqx.field(static=True)
    accepted_state_id: str = eqx.field(static=True)

    def __init__(
        self,
        candidate: ContinuationCandidate,
        application_state: Any,
        /,
        *,
        application_state_id: str,
        decision_id: str,
        accepted_index: int,
    ):
        if not isinstance(candidate, ContinuationCandidate):
            raise TypeError("candidate must be a ContinuationCandidate.")
        if not bool(candidate.numerical_accepted):
            raise ValueError("Only a numerically accepted candidate can be committed.")
        application_id = str(application_state_id)
        decision = str(decision_id)
        index = int(accepted_index)
        if not application_id or not decision or index < 0:
            raise ValueError("Accepted continuation identities and index are invalid.")
        accepted_state_id = canonical_fingerprint(
            {
                "kind": "continuation-accepted-state",
                "candidate": candidate.candidate_id,
                "application_state": application_id,
                "decision": decision,
                "accepted_index": index,
            }
        )
        self.candidate = candidate
        self.application_state = application_state
        self.realization = candidate.realization
        self.application_state_id = application_id
        self.decision_id = decision
        self.accepted_index = index
        self.accepted_state_id = accepted_state_id


def continuation_step_decision_id(
    candidate: ContinuationCandidate,
    transfer: ParameterTransferEvidence,
    /,
    *,
    transaction_id: str,
    source_application_state_id: str,
    restored_application_state_id: str,
    numerical_accepted: bool,
    accepted: bool,
    committed: bool,
    rolled_back: bool,
    message: str = "",
) -> str:
    """Return the canonical identity of one exclusive commit-or-rollback route."""
    return canonical_fingerprint(
        {
            "kind": "continuation-step-decision",
            "candidate": candidate.candidate_id,
            "transfer": transfer.evidence_id,
            "transaction": str(transaction_id),
            "source_application_state": str(source_application_state_id),
            "restored_application_state": str(restored_application_state_id),
            "numerical_accepted": bool(numerical_accepted),
            "accepted": bool(accepted),
            "committed": bool(committed),
            "rolled_back": bool(rolled_back),
            "message": str(message),
        }
    )


class ContinuationStepResult(StrictModule):
    """One atomic attempt decision, including exactly one commit or rollback route."""

    candidate: ContinuationCandidate
    transfer: ParameterTransferEvidence
    accepted_state: ContinuationAcceptedState | None
    numerical_accepted: Array
    accepted: Array
    committed: Array
    rolled_back: Array
    transaction_id: str = eqx.field(static=True)
    source_application_state_id: str = eqx.field(static=True)
    restored_application_state_id: str = eqx.field(static=True)
    message: str = eqx.field(static=True)
    decision_id: str = eqx.field(static=True)

    def __init__(
        self,
        candidate: ContinuationCandidate,
        transfer: ParameterTransferEvidence,
        /,
        *,
        accepted_state: ContinuationAcceptedState | None,
        numerical_accepted: Any,
        accepted: Any,
        committed: Any,
        rolled_back: Any,
        transaction_id: str,
        source_application_state_id: str,
        restored_application_state_id: str,
        message: str = "",
    ):
        if not isinstance(candidate, ContinuationCandidate):
            raise TypeError("candidate must be a ContinuationCandidate.")
        if not isinstance(transfer, ParameterTransferEvidence):
            raise TypeError("transfer must be ParameterTransferEvidence.")
        if accepted_state is not None and not isinstance(
            accepted_state, ContinuationAcceptedState
        ):
            raise TypeError("accepted_state must be ContinuationAcceptedState or None.")
        if transfer.target_realization_id != candidate.realization.realization_id:
            raise ValueError("Transfer evidence targets another parameter realization.")
        flags = tuple(
            jnp.asarray(value, dtype=bool)
            for value in (numerical_accepted, accepted, committed, rolled_back)
        )
        if any(value.shape != () for value in flags):
            raise ValueError("Continuation step decision flags must be scalar.")
        numerical, accepted_, committed_, rolled_back_ = tuple(
            bool(value) for value in flags
        )
        if numerical != bool(candidate.numerical_accepted):
            raise ValueError("Step numerical acceptance must match its candidate.")
        if accepted_ != (numerical and bool(transfer.accepted)):
            raise ValueError(
                "A continuation attempt accepts only with numerical and transfer evidence."
            )
        if accepted_ != committed_ or accepted_ == rolled_back_:
            raise ValueError(
                "Each attempt must commit once or roll back once, exclusively."
            )
        if accepted_ != (accepted_state is not None):
            raise ValueError("Accepted continuation attempts require an accepted state.")
        transaction = str(transaction_id)
        source = str(source_application_state_id)
        restored = str(restored_application_state_id)
        if not transaction or not source or not restored:
            raise ValueError("Continuation transaction identities must be non-empty.")
        if rolled_back_ and restored != source:
            raise ValueError(
                "A rejected attempt must restore the source application state."
            )
        message_ = str(message)
        decision_id = continuation_step_decision_id(
            candidate,
            transfer,
            transaction_id=transaction,
            source_application_state_id=source,
            restored_application_state_id=restored,
            numerical_accepted=numerical,
            accepted=accepted_,
            committed=committed_,
            rolled_back=rolled_back_,
            message=message_,
        )
        if accepted_state is not None and (
            accepted_state.decision_id != decision_id
            or accepted_state.candidate.candidate_id != candidate.candidate_id
            or accepted_state.application_state_id != restored
        ):
            raise ValueError(
                "Accepted state and continuation step decisions do not match."
            )
        self.candidate = candidate
        self.transfer = transfer
        self.accepted_state = accepted_state
        (
            self.numerical_accepted,
            self.accepted,
            self.committed,
            self.rolled_back,
        ) = flags
        self.transaction_id = transaction
        self.source_application_state_id = source
        self.restored_application_state_id = restored
        self.message = message_
        self.decision_id = decision_id


class AbstractContinuationAdapter(StrictModule):
    """Curve adapter with opaque, application-owned attempt transactions."""

    continuation_problem: AbstractAttribute[Any]
    coordinate_lower: AbstractAttribute[float]
    coordinate_upper: AbstractAttribute[float]
    problem_id: AbstractAttribute[str]
    adapter_id: AbstractAttribute[str]
    __strict_abstract__ = True

    @abc.abstractmethod
    def freeze_application_state(
        self, application_state: Any, args: Any = None, /
    ) -> Any:
        """Freeze the last committed application state for one isolated attempt."""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_candidate(
        self,
        transaction: Any,
        source: ContinuationAcceptedState | None,
        candidate: ContinuationCandidate,
        args: Any = None,
        /,
    ) -> ParameterTransferEvidence:
        """Evaluate an isolated candidate without committing application state."""
        raise NotImplementedError

    @abc.abstractmethod
    def commit_candidate(
        self,
        transaction: Any,
        source: ContinuationAcceptedState | None,
        candidate: ContinuationCandidate,
        evidence: ParameterTransferEvidence,
        args: Any = None,
        /,
    ) -> Any:
        """Commit exactly once after every numerical and application gate passes."""
        raise NotImplementedError

    @abc.abstractmethod
    def rollback_candidate(
        self,
        transaction: Any,
        source: ContinuationAcceptedState | None,
        candidate: ContinuationCandidate,
        evidence: ParameterTransferEvidence,
        args: Any = None,
        /,
    ) -> Any:
        """Restore the exact committed application state after a rejected attempt."""
        raise NotImplementedError

    @abc.abstractmethod
    def application_state_identity(
        self, application_state: Any, args: Any = None, /
    ) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def checkpoint_application_state(
        self, application_state: Any, args: Any = None, /
    ) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def restore_application_state(self, data: Any, args: Any = None, /) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def declared_spaces(self, /):
        raise NotImplementedError

    @abc.abstractmethod
    def representation_policy(self, /) -> ContinuationRepresentationPolicy:
        raise NotImplementedError

    @abc.abstractmethod
    def residual(self, state: PyTree[Any], coordinate: Any, args: Any = None, /):
        raise NotImplementedError

    @abc.abstractmethod
    def parameters(self, coordinate: Any, args: Any = None, /):
        raise NotImplementedError

    @abc.abstractmethod
    def state_jacobian_action(
        self,
        state: PyTree[Any],
        coordinate: Any,
        tangent: PyTree[Any],
        args: Any = None,
        /,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def coordinate_derivative(
        self,
        state: PyTree[Any],
        coordinate: Any,
        args: Any = None,
        /,
    ):
        raise NotImplementedError


class CallableContinuationAdapter(AbstractContinuationAdapter):
    """Continuation adapter backed by explicit pure transaction callbacks."""

    continuation_problem: Any
    freeze_function: Callable[[Any, Any], Any] | None
    evaluate_function: Callable[..., ParameterTransferEvidence] | None
    commit_function: Callable[..., Any] | None
    rollback_function: Callable[..., Any] | None
    state_identity_function: Callable[[Any, Any], str] | None
    checkpoint_function: Callable[[Any, Any], Any] | None
    restore_function: Callable[[Any, Any], Any] | None
    coordinate_lower: float = eqx.field(static=True)
    coordinate_upper: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)
    supports_opaque_state: bool = eqx.field(static=True)

    def __init__(
        self,
        problem: Any,
        /,
        *,
        adapter_id: str | None = None,
        freeze: Callable[[Any, Any], Any] | None = None,
        evaluate: Callable[..., ParameterTransferEvidence] | None = None,
        commit: Callable[..., Any] | None = None,
        rollback: Callable[..., Any] | None = None,
        state_identity: Callable[[Any, Any], str] | None = None,
        checkpoint: Callable[[Any, Any], Any] | None = None,
        restore: Callable[[Any, Any], Any] | None = None,
    ):
        from ._core import ContinuationCurveProblem

        if not isinstance(problem, ContinuationCurveProblem):
            raise TypeError("problem must be a ContinuationCurveProblem.")
        callbacks = (
            freeze,
            evaluate,
            commit,
            rollback,
            state_identity,
            checkpoint,
            restore,
        )
        if any(value is not None and not callable(value) for value in callbacks):
            raise TypeError("Continuation adapter callbacks must be callable or None.")
        state_callbacks = (freeze, commit, rollback, state_identity, checkpoint, restore)
        supplied_state_callbacks = tuple(value is not None for value in state_callbacks)
        if any(supplied_state_callbacks) and not all(supplied_state_callbacks):
            raise ValueError(
                "Opaque application state requires freeze, commit, rollback, identity, "
                "checkpoint, and restore callbacks together."
            )
        identifier = (
            f"{problem.problem_id}/continuation-adapter"
            if adapter_id is None
            else str(adapter_id)
        )
        if not identifier:
            raise ValueError("adapter_id must be non-empty.")
        self.continuation_problem = problem
        self.freeze_function = freeze
        self.evaluate_function = evaluate
        self.commit_function = commit
        self.rollback_function = rollback
        self.state_identity_function = state_identity
        self.checkpoint_function = checkpoint
        self.restore_function = restore
        self.coordinate_lower = problem.coordinate_lower
        self.coordinate_upper = problem.coordinate_upper
        self.problem_id = problem.problem_id
        self.adapter_id = identifier
        self.supports_opaque_state = all(supplied_state_callbacks)

    def freeze_application_state(
        self, application_state: Any, args: Any = None, /
    ) -> Any:
        if self.freeze_function is None:
            if application_state is not None:
                raise ValueError(
                    "A callback-free continuation adapter supports only stateless curves."
                )
            return None
        return self.freeze_function(application_state, args)

    def evaluate_candidate(
        self,
        transaction: Any,
        source: ContinuationAcceptedState | None,
        candidate: ContinuationCandidate,
        args: Any = None,
        /,
    ) -> ParameterTransferEvidence:
        if self.evaluate_function is None:
            return ParameterTransferEvidence.for_candidate(source, candidate)
        evidence = self.evaluate_function(transaction, source, candidate, args)
        if not isinstance(evidence, ParameterTransferEvidence):
            raise TypeError("evaluate callback must return ParameterTransferEvidence.")
        if evidence.target_realization_id != candidate.realization.realization_id:
            raise ValueError("Transfer evidence targets another parameter realization.")
        return evidence

    def commit_candidate(
        self,
        transaction: Any,
        source: ContinuationAcceptedState | None,
        candidate: ContinuationCandidate,
        evidence: ParameterTransferEvidence,
        args: Any = None,
    ) -> Any:
        if self.commit_function is None:
            if transaction is not None:
                raise ValueError(
                    "A callback-free continuation adapter cannot commit application state."
                )
            return None
        return self.commit_function(transaction, source, candidate, evidence, args)

    def rollback_candidate(
        self,
        transaction: Any,
        source: ContinuationAcceptedState | None,
        candidate: ContinuationCandidate,
        evidence: ParameterTransferEvidence,
        args: Any = None,
        /,
    ) -> Any:
        if self.rollback_function is None:
            if transaction is not None:
                raise ValueError(
                    "A callback-free continuation adapter cannot roll back application state."
                )
            return None
        return self.rollback_function(transaction, source, candidate, evidence, args)

    def application_state_identity(
        self, application_state: Any, args: Any = None, /
    ) -> str:
        if self.state_identity_function is not None:
            identifier = str(self.state_identity_function(application_state, args))
        else:
            if application_state is not None:
                raise ValueError(
                    "Opaque application state requires an explicit identity callback."
                )
            identifier = _tree_content_id("stateless-continuation-application", None)
        if not identifier:
            raise ValueError("Application state identity must be non-empty.")
        return identifier

    def checkpoint_application_state(
        self, application_state: Any, args: Any = None, /
    ) -> Any:
        if self.checkpoint_function is None:
            if application_state is not None:
                raise ValueError(
                    "Opaque application state requires an explicit checkpoint callback."
                )
            return None
        return self.checkpoint_function(application_state, args)

    def restore_application_state(self, data: Any, args: Any = None, /) -> Any:
        if self.restore_function is None:
            if data is not None:
                raise ValueError(
                    "Opaque application state requires an explicit restore callback."
                )
            return None
        return self.restore_function(data, args)

    def declared_spaces(self, /):
        return self.continuation_problem.declared_spaces()

    def representation_policy(self, /) -> ContinuationRepresentationPolicy:
        return self.continuation_problem.representation_policy()

    def residual(self, state: PyTree[Any], coordinate: Any, args: Any = None, /):
        return self.continuation_problem.residual(state, coordinate, args)

    def parameters(self, coordinate: Any, args: Any = None, /):
        return self.continuation_problem.parameters(coordinate, args)

    def state_jacobian_action(
        self,
        state: PyTree[Any],
        coordinate: Any,
        tangent: PyTree[Any],
        args: Any = None,
        /,
    ):
        return self.continuation_problem.state_jacobian_action(
            state, coordinate, tangent, args
        )

    def coordinate_derivative(
        self,
        state: PyTree[Any],
        coordinate: Any,
        args: Any = None,
        /,
    ):
        return self.continuation_problem.coordinate_derivative(state, coordinate, args)


class ContinuationAdapterAudit(StrictModule):
    """Compatibility audit proving that an adapter delegates one authoritative curve."""

    problem_identity_matches: Array
    coordinate_interval_matches: Array
    spaces_match: Array
    representation_matches: Array
    adapter_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    audit_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        adapter_id: str,
        problem_id: str,
        problem_identity_matches: Any,
        coordinate_interval_matches: Any,
        spaces_match: Any,
        representation_matches: Any,
    ):
        flags = tuple(
            jnp.asarray(value, dtype=bool)
            for value in (
                problem_identity_matches,
                coordinate_interval_matches,
                spaces_match,
                representation_matches,
            )
        )
        if any(value.shape != () for value in flags):
            raise ValueError("Continuation adapter audit flags must be scalar.")
        identifier = str(adapter_id)
        problem = str(problem_id)
        if not identifier or not problem:
            raise ValueError("Continuation adapter audit identities must be non-empty.")
        audit_id = canonical_fingerprint(
            {
                "kind": "continuation-adapter-audit",
                "adapter": identifier,
                "problem": problem,
                "flags": tuple(bool(value) for value in flags),
            }
        )
        (
            self.problem_identity_matches,
            self.coordinate_interval_matches,
            self.spaces_match,
            self.representation_matches,
        ) = flags
        self.adapter_id = identifier
        self.problem_id = problem
        self.audit_id = audit_id

    @property
    def compatible(self) -> Array:
        return (
            self.problem_identity_matches
            & self.coordinate_interval_matches
            & self.spaces_match
            & self.representation_matches
        )


def audit_continuation_adapter(
    adapter: AbstractContinuationAdapter, /
) -> ContinuationAdapterAudit:
    """Audit adapter/problem identity, interval, spaces, and representation delegation."""
    if not isinstance(adapter, AbstractContinuationAdapter):
        raise TypeError("adapter must be an AbstractContinuationAdapter.")
    problem = adapter.continuation_problem
    adapter_spaces = adapter.declared_spaces()
    problem_spaces = problem.declared_spaces()
    spaces_match = all(
        (left is None and right is None)
        or (left is not None and right is not None and left.space_id == right.space_id)
        for left, right in zip(adapter_spaces, problem_spaces, strict=True)
    )
    return ContinuationAdapterAudit(
        adapter_id=adapter.adapter_id,
        problem_id=adapter.problem_id,
        problem_identity_matches=adapter.problem_id == problem.problem_id,
        coordinate_interval_matches=(
            adapter.coordinate_lower == problem.coordinate_lower
            and adapter.coordinate_upper == problem.coordinate_upper
        ),
        spaces_match=spaces_match,
        representation_matches=(
            adapter.representation_policy().policy_id
            == problem.representation_policy().policy_id
        ),
    )


__all__ = [
    "AbstractContinuationAdapter",
    "CallableContinuationAdapter",
    "ContinuationAcceptedState",
    "ContinuationAdapterAudit",
    "ContinuationCandidate",
    "ContinuationStepResult",
    "ParameterRealization",
    "continuation_step_decision_id",
    "ParameterTransferEvidence",
    "audit_continuation_adapter",
]
