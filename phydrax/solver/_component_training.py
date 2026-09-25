#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Public trainer of components bound into fixed prepared solves."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Key, PyTree

from .._sampling._addressing import derive_key, SampleAddress
from .._strict import StrictModule
from .._trainable import partition_parameters
from .._training_checkpoint import load_training_checkpoint, save_training_checkpoint
from .._training_kernel import (
    _parameter_revision,
    AbstractKernelUpdateRule,
    build_training_checkpoint,
    OptaxUpdateRule,
    prepare_training_kernel,
    run_training_attempt,
    training_site_key,
    TrainingAttemptEvidence,
    TrainingKernelSpec,
)
from ..optim._evolution_strategy import (
    AbstractDistributionEvolutionMethod,
    DistributionEvolutionPayload,
    DistributionEvolutionUpdateRule,
)
from ..optim._kernel_rules import (
    MirrorUpdateRule,
    RiemannianLineSearchUpdateRule,
    RiemannianUpdateRule,
)
from ..optim._kfac._config import KFAC
from ..optim._mirror_descent import AbstractMirrorOptimizer
from ..optim._riemannian import (
    AbstractRiemannianLineSearchOptimizer,
    AbstractRiemannianOptimizer,
)
from ._solver_objective import AbstractSolverObjective, kernel_objective


ComponentOptimizer = (
    optax.GradientTransformation
    | optax.GradientTransformationExtraArgs
    | AbstractDistributionEvolutionMethod
    | KFAC
    | AbstractMirrorOptimizer
    | AbstractRiemannianOptimizer
)

_CONTEXT = "train_components"
_CHECKPOINT_FORMAT = "phydrax-component-training"
_EVOLUTION_OBJECTIVE_ID = "train-components-evolution"
_EVOLUTION_INITIALIZATION = SampleAddress(
    "training", _EVOLUTION_OBJECTIVE_ID, target="algorithm", role="initialization"
)
_LINE_SEARCH_STATES = (
    optax.ScaleByBacktrackingLinesearchState,
    optax.ScaleByZoomLinesearchState,
)


@final
class ComponentTrainingResult(StrictModule):
    """Committed components and accepted-update evidence of one training run.

    `tree` is the committed trained tree (only accepted attempts change it).
    Per attempt of this call: `values` (the weighted objective at the committed
    parameters), `objective_values` (`attempts x objectives`), `outcomes`
    (`TrainingAttemptOutcome` codes), `gradient_norms`, and `failed_cases`
    (failed solver cases per objective). Counters are cumulative over the run,
    including attempts restored from a checkpoint. `selection` lists, per
    objective, the PARAMETER paths it trains; `authorities` the slot-derived
    `(path, authority)` of every PARAMETER leaf. `training_id` identifies the
    run (roles, objectives, update rule, authorities) and `parameter_revision`
    the committed parameter values.
    """

    tree: Any
    values: Array
    objective_values: Array
    outcomes: Array
    gradient_norms: Array
    failed_cases: Array
    objective_ids: tuple[str, ...] = eqx.field(static=True)
    selection: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    authorities: tuple[tuple[str, str], ...] = eqx.field(static=True)
    attempts: int = eqx.field(static=True)
    accepted_updates: int = eqx.field(static=True)
    finite_rejections: int = eqx.field(static=True)
    nonfinite_rejections: int = eqx.field(static=True)
    resumed_from_attempt: int = eqx.field(static=True)
    training_id: str = eqx.field(static=True)
    parameter_revision: str = eqx.field(static=True)
    checkpoint_path: str | None = eqx.field(static=True)


def _has_line_search(
    optimizer: optax.GradientTransformationExtraArgs, tree: PyTree[Any], /
) -> bool:
    state = jax.eval_shape(optimizer.init, partition_parameters(tree)[0])
    return any(
        isinstance(value, _LINE_SEARCH_STATES)
        for value in jax.tree_util.tree_leaves(
            state, is_leaf=lambda value: isinstance(value, _LINE_SEARCH_STATES)
        )
    )


def _update_rule(
    optimizer: Any, tree: PyTree[Any], key: Key[Array, ""], /
) -> AbstractKernelUpdateRule:
    """The kernel rule of one member of the FunctionalSolver optimizer union."""
    if isinstance(optimizer, KFAC):
        raise ValueError(
            f"{_CONTEXT}: KFAC forms its curvature from FunctionalSolver residual "
            "terms; solver objectives expose one scalar objective. Use an Optax, "
            "mirror, Riemannian, or distribution-evolution optimizer."
        )
    if isinstance(optimizer, AbstractDistributionEvolutionMethod):
        return DistributionEvolutionUpdateRule(
            optimizer,
            derive_key(key, _EVOLUTION_INITIALIZATION),
            rule_id="train-components-distribution-evolution",
        )
    if isinstance(optimizer, AbstractMirrorOptimizer):
        return MirrorUpdateRule(optimizer, rule_id="train-components")
    if isinstance(optimizer, AbstractRiemannianLineSearchOptimizer):
        return RiemannianLineSearchUpdateRule(optimizer, rule_id="train-components")
    if isinstance(optimizer, AbstractRiemannianOptimizer):
        return RiemannianUpdateRule(optimizer, rule_id="train-components")
    if isinstance(optimizer, optax.GradientTransformationExtraArgs):
        line_search = _has_line_search(optimizer, tree)
        return OptaxUpdateRule(
            optimizer,
            rule_id=(
                "train-components-optax-line-search"
                if line_search
                else "train-components-optax"
            ),
            reevaluates_objective=line_search,
        )
    if isinstance(optimizer, optax.GradientTransformation):
        return OptaxUpdateRule(optimizer, rule_id="train-components-optax")
    raise TypeError(
        "optimizer must be an Optax transformation, a Phydrax mirror or Riemannian "
        "optimizer, a distribution-evolution method, or a KFAC configuration."
    )


def _history(
    records: Sequence[TrainingAttemptEvidence], objectives: int, /
) -> tuple[Array, Array, Array, Array, Array]:
    if not records:
        empty = jnp.zeros((0,))
        return (
            empty,
            jnp.zeros((0, objectives)),
            jnp.zeros((0,), dtype=jnp.int32),
            empty,
            jnp.zeros((0, objectives), dtype=jnp.int32),
        )
    return (
        jnp.stack([record.value for record in records]),
        jnp.stack([jnp.stack(record.objective_values) for record in records]),
        jnp.stack([record.outcome for record in records]),
        jnp.stack([record.gradient_norm for record in records]),
        jnp.stack(
            [
                jnp.stack([failures for _, failures in record.diagnostics])
                for record in records
            ]
        ),
    )


def train_components(
    tree: PyTree[Any],
    objectives: Sequence[AbstractSolverObjective],
    /,
    *,
    optimizer: ComponentOptimizer,
    steps: int,
    key: Key[Array, ""],
    checkpoint: str | Path | None = None,
    rejection_budget: int = 0,
) -> ComponentTrainingResult:
    """Train components bound into fixed prepared solves by solver objectives.

    Every PARAMETER leaf of `tree` takes its authority from its owning component
    slot or `ComponentBinding`; there is no root authority. Every authority group
    needs an objective whose `(route, kind)` it admits, and each objective trains
    only the groups that admit it, so mixed-authority trees need one compatible
    objective per group (for example a `RolloutObjective` for a closure and an
    `AlgorithmicWorkObjective` for a preconditioner). Admission of every
    objective runs before anything is traced.

    `optimizer` is the `FunctionalSolver.solve` optimizer union: Optax
    transformations (extra-argument line searches re-evaluate the objective),
    mirror and Riemannian optimizers, and distribution-evolution methods
    (derivative-free; derivative admission is then not required, but every
    objective must admit every authority group). KFAC is refused: its curvature
    needs FunctionalSolver residual terms.

    `steps` counts attempts of the whole run through the one training kernel
    (accepted, rejected, or skipped). `key` is the typed root key of every
    semantic training key. With `checkpoint`, a directory holding a checkpoint
    of the same run resumes from it (identities are verified and mismatches fail
    closed) and the committed state is published after every attempt. More than
    `rejection_budget` consecutive rejected attempts raise
    `TrainingRejectionBudgetError`.
    """
    objectives_ = tuple(objectives)
    if not objectives_:
        raise ValueError(f"{_CONTEXT}: at least one solver objective is required.")
    if any(
        not isinstance(objective, AbstractSolverObjective) for objective in objectives_
    ):
        raise TypeError("objectives must be solver objectives.")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer.")
    key_ = jnp.asarray(key)
    if not jax.dtypes.issubdtype(key_.dtype, jax.dtypes.prng_key) or key_.shape:
        raise TypeError("key must be one typed JAX PRNG key (jax.random.key).")
    rule = _update_rule(optimizer, tree, key_)
    differentiable = not rule.forms_own_derivatives
    for objective in objectives_:
        objective.admit(tree, differentiable=differentiable)
    kernel = prepare_training_kernel(
        tree,
        tuple(kernel_objective(objective) for objective in objectives_),
        TrainingKernelSpec(rule, context=_CONTEXT, rejection_budget=rejection_budget),
        root_authority=None,
    )
    state = kernel.init(tree, key_)
    path = None if checkpoint is None else Path(checkpoint)
    if path is not None and (path / "manifest.json").is_file():
        state = load_training_checkpoint(
            path, kernel, state, None, format=_CHECKPOINT_FORMAT
        ).restored.state
    resumed = int(jax.device_get(state.attempt_cursor))
    evolution = isinstance(rule, DistributionEvolutionUpdateRule)
    records: list[TrainingAttemptEvidence] = []
    for attempt in range(resumed, steps):
        payload = (
            DistributionEvolutionPayload(
                None,
                training_site_key(
                    state.root_key,
                    objective_id=_EVOLUTION_OBJECTIVE_ID,
                    site="ask",
                    attempt=attempt,
                    microstep=0,
                ),
            )
            if evolution
            else None
        )
        state, evidence = run_training_attempt(kernel, state, payload)
        records.append(evidence)
        if path is not None:
            # Attempts never accumulate, so every post-attempt state is committed.
            save_training_checkpoint(
                path,
                build_training_checkpoint(kernel, state, allow_intermediate=True),
                None,
                format=_CHECKPOINT_FORMAT,
                metadata={},
            )
    values, objective_values, outcomes, gradient_norms, failed = _history(
        records, len(objectives_)
    )
    attempts, accepted, finite, nonfinite = (
        int(value)
        for value in jax.device_get(
            (
                state.attempt_cursor,
                state.accepted_cursor,
                state.finite_rejections,
                state.nonfinite_rejections,
            )
        )
    )
    return ComponentTrainingResult(
        tree=kernel.tree(state),
        values=values,
        objective_values=objective_values,
        outcomes=outcomes,
        gradient_norms=gradient_norms,
        failed_cases=failed,
        objective_ids=tuple(objective.objective_id for objective in objectives_),
        selection=tuple(
            tuple(
                path_
                for path_, trained in zip(kernel.parameter_paths, admitted, strict=True)
                if trained
            )
            for admitted in kernel.admission
        ),
        authorities=tuple(
            (path_, authority.value)
            for path_, authority in zip(
                kernel.parameter_paths, kernel.parameter_authorities, strict=True
            )
        ),
        attempts=attempts,
        accepted_updates=accepted,
        finite_rejections=finite,
        nonfinite_rejections=nonfinite,
        resumed_from_attempt=resumed,
        training_id=kernel.checkpoint_id,
        parameter_revision=_parameter_revision(kernel, state.parameters).revision_id,
        checkpoint_path=None if path is None else str(path),
    )


__all__ = ["ComponentOptimizer", "ComponentTrainingResult", "train_components"]
