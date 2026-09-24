#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._training import (
    DelayedTargetPolicy,
    ExponentialMovingAverageTargetPolicy,
)
from .._training_checkpoint import (
    load_training_checkpoint,
    read_training_checkpoint_metadata,
    save_training_checkpoint,
)
from .._training_kernel import build_training_checkpoint, PreparedTrainingKernel
from ._functional_training import FunctionalTrainingPlan, FunctionalTrainingState


_FUNCTIONAL_CHECKPOINT_FORMAT = "phydrax-functional-training-checkpoint"


def _target_policy_contract(kernel: PreparedTrainingKernel, /) -> dict[str, Any] | None:
    policy = kernel.target_policy
    if policy is None:
        return None
    if isinstance(policy, DelayedTargetPolicy):
        return {
            "kind": "delayed",
            "delay": policy.delay,
        }
    if isinstance(policy, ExponentialMovingAverageTargetPolicy):
        return {
            "kind": "exponential-moving-average",
            "decay": policy.decay,
            "start_step": policy.start_step,
            "update_every": policy.update_every,
            "source": policy.source,
        }
    raise TypeError("Functional checkpoint target policy has an unsupported type.")


def _sharding_identity(plan: FunctionalTrainingPlan, /) -> str | None:
    return None if plan.sharding is None else plan.sharding.policy_id


def _metadata(
    kernel: PreparedTrainingKernel,
    solver: Any,
    state: FunctionalTrainingState,
    plan: FunctionalTrainingPlan,
    /,
) -> dict[str, Any]:
    enforcement = state.enforcement_state
    return {
        "plan_id": plan.plan_id,
        "run_id": state.run_id,
        "gradient_accumulation": state.gradient_accumulation,
        "target_policy": _target_policy_contract(kernel),
        "enforcement_generation": None if enforcement is None else enforcement.generation,
        "enforcement_accepted_step": (
            None if enforcement is None else enforcement.accepted_step
        ),
        "discretization_bundle_id": solver.discretization_bundle.bundle_id,
        "training_seconds": state.training_seconds,
        "resumed_from_step": state.resumed_from_step,
    }


def _extra(solver: Any, state: FunctionalTrainingState, /) -> tuple[Any, ...]:
    return (
        solver.functions,
        solver.objective,
        state.current_functions,
        state.best_functions,
        state.previous_functions,
        state.enforcement_state,
        state.pseudo_inverse_steps,
        state.term_multipliers,
        state.previous_gradient,
    )


@dataclass(frozen=True, slots=True)
class FunctionalTrainingCheckpoint:
    functions: Any
    objective: Any
    state: FunctionalTrainingState
    path: Path


def save_functional_training_checkpoint(
    path: str | Path,
    kernel: PreparedTrainingKernel,
    solver: Any,
    state: FunctionalTrainingState,
    plan: FunctionalTrainingPlan,
    /,
    *,
    final: bool = False,
) -> Path:
    """Atomically publish one functional training state through the kernel payload.

    Periodic checkpoints are accepted-update boundaries. A `final` checkpoint may
    close a run on a rejected attempt; the kernel manifest records the boundary.
    """
    if not isinstance(state, FunctionalTrainingState):
        raise TypeError("state must be a FunctionalTrainingState.")
    if not isinstance(plan, FunctionalTrainingPlan):
        raise TypeError("plan must be a FunctionalTrainingPlan.")
    if state.kernel_checkpoint_id != kernel.checkpoint_id:
        raise ValueError("Functional checkpoint state was not produced by this kernel.")
    payload = build_training_checkpoint(
        kernel,
        state.kernel_state,
        selection=state.progress,
        sharding_identity=_sharding_identity(plan),
        allow_intermediate=final,
    )
    destination = Path(path)
    save_training_checkpoint(
        destination,
        payload,
        _extra(solver, state),
        format=_FUNCTIONAL_CHECKPOINT_FORMAT,
        metadata=_metadata(kernel, solver, state, plan),
    )
    return destination


def load_functional_training_checkpoint(
    path: str | Path,
    kernel: PreparedTrainingKernel,
    solver_like: Any,
    state_like: FunctionalTrainingState,
    plan: FunctionalTrainingPlan,
    /,
) -> FunctionalTrainingCheckpoint:
    """Verify and restore a checkpoint against the run's kernel and contracts.

    The kernel payload fails closed on any role, objective, rule, authority,
    structure, key, boundary, or parameter-revision mismatch; the functional
    metadata must match the plan, run, target policy, and discretization.
    """
    if not isinstance(state_like, FunctionalTrainingState):
        raise TypeError("state_like must be a FunctionalTrainingState.")
    if not isinstance(plan, FunctionalTrainingPlan):
        raise TypeError("plan must be a FunctionalTrainingPlan.")
    metadata = read_training_checkpoint_metadata(
        path, format=_FUNCTIONAL_CHECKPOINT_FORMAT
    )
    if set(metadata) != set(_metadata(kernel, solver_like, state_like, plan)):
        raise ValueError("Functional checkpoint metadata fields are not canonical.")
    expected = {
        "plan_id": (plan.plan_id, "training-plan"),
        "run_id": (state_like.run_id, "run"),
        "gradient_accumulation": (
            state_like.gradient_accumulation,
            "gradient-accumulation",
        ),
        "target_policy": (_target_policy_contract(kernel), "target-policy"),
        "discretization_bundle_id": (
            solver_like.discretization_bundle.bundle_id,
            "discretization",
        ),
    }
    for name, (value, label) in expected.items():
        if metadata[name] != value:
            raise ValueError(f"Functional checkpoint {label} identity mismatch.")
    loaded = load_training_checkpoint(
        path,
        kernel,
        state_like.kernel_state,
        _extra(solver_like, state_like),
        format=_FUNCTIONAL_CHECKPOINT_FORMAT,
        sharding_identity=_sharding_identity(plan),
    )
    if loaded.metadata != metadata:
        raise ValueError("Functional checkpoint manifest changed while loading.")
    progress = loaded.restored.selection
    if progress is None:
        raise ValueError("Functional checkpoint progress is missing.")
    kernel_state = loaded.restored.state
    if int(kernel_state.accepted_cursor) != progress.update_step:
        raise ValueError(
            "Functional checkpoint progress disagrees with its kernel state."
        )
    (
        functions,
        objective,
        current_functions,
        best_functions,
        previous_functions,
        enforcement_state,
        pseudo_inverse_steps,
        term_multipliers,
        previous_gradient,
    ) = loaded.extra
    if (enforcement_state is None) != (metadata["enforcement_generation"] is None) or (
        enforcement_state is not None
        and (
            enforcement_state.generation != int(metadata["enforcement_generation"])
            or enforcement_state.accepted_step
            != int(metadata["enforcement_accepted_step"])
        )
    ):
        raise ValueError(
            "Functional checkpoint enforcement state disagrees with its manifest."
        )
    restored = FunctionalTrainingState(
        current_functions=current_functions,
        best_functions=best_functions,
        previous_functions=previous_functions,
        kernel_state=kernel_state,
        kernel_checkpoint_id=kernel.checkpoint_id,
        enforcement_state=enforcement_state,
        pseudo_inverse_steps=pseudo_inverse_steps,
        term_multipliers=term_multipliers,
        previous_gradient=previous_gradient,
        progress=progress,
        run_id=str(metadata["run_id"]),
        gradient_accumulation=int(metadata["gradient_accumulation"]),
        training_seconds=float(metadata["training_seconds"]),
        resumed_from_step=int(metadata["resumed_from_step"]),
    )
    return FunctionalTrainingCheckpoint(functions, objective, restored, Path(path))


__all__ = [
    "FunctionalTrainingCheckpoint",
    "load_functional_training_checkpoint",
    "save_functional_training_checkpoint",
]
