#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""FunctionalSolver lowering onto the internal training kernel.

Every FunctionalSolver route (gradient, KFAC, evolution, windows, decomposition)
trains one kernel objective: the solver's ordered raw total, lowered as a single
`_ObjectiveContribution(total, 1, 0)` so the kernel's normalization is the
identity and the floating-point summation order of the authored terms is kept.
Parameters without an owning component slot take the SURROGATE root authority.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Key

from .._differentiation import ComponentAuthority, DerivativeRoute, ObjectiveKind
from .._strict import StrictModule
from .._trainable import combine_parameters, partition_parameters
from .._training_kernel import (
    build_training_checkpoint,
    KernelObjective,
    PreparedTrainingKernel,
    restore_training_checkpoint,
    SubspaceTrainingTree,
    training_site_key,
    TrainingKernelState,
)
from .._training_objective import _ObjectiveContribution


FUNCTIONAL_OBJECTIVE_ID = "functional-objective"
FUNCTIONAL_ROOT_AUTHORITY = ComponentAuthority.SURROGATE
FUNCTIONAL_OBJECTIVE_KIND = ObjectiveKind.PHYSICAL_RESIDUAL
# Consecutive rejected attempts a FunctionalSolver run tolerates before it raises
# `TrainingRejectionBudgetError`. Native least-squares and iterative methods run
# their own damping and globalization, so the budget is a stagnation guard.
FUNCTIONAL_REJECTION_BUDGET = 64

# `loss(parameters, held, payload, keys) -> (ordered total, diagnostics)` over the
# legacy functional lanes: `parameters` is the trained lane and `held` is the
# combined MODEL_STATE and FIXED lanes (or the frozen `ParameterSubspace`).
FunctionalLoss = Callable[[Any, Any, Any, Any], tuple[Array, Any]]


def functional_training_tree(
    functions: Any,
    /,
    *,
    sharding: Any = None,
    subspace: Any = None,
) -> Any:
    """Kernel tree of one FunctionalSolver run.

    Without a subspace the tree is the functions themselves (placed by role lane
    under a sharding policy); with an explicit `ParameterSubspace` it is the
    `SubspaceTrainingTree` that trains exactly the selected leaves.
    """
    if subspace is None:
        if sharding is None:
            return functions
        return combine_parameters(*sharding.place_lanes(*partition_parameters(functions)))
    if sharding is not None:
        subspace = eqx.tree_at(
            lambda value: value.initial,
            sharding.place_tree(subspace),
            sharding.place_parameters(subspace.initial),
        )
    return SubspaceTrainingTree.from_subspace(subspace)


def functional_tree_functions(tree: Any, /) -> Any:
    """Functions represented by a kernel tree from `functional_training_tree`."""
    return tree.model() if isinstance(tree, SubspaceTrainingTree) else tree


def functional_lanes(parameters: Any, model_state: Any, fixed: Any, /) -> tuple[Any, Any]:
    """Kernel role lanes as the legacy `(trained parameters, held)` pair.

    Recombine with `eqx.combine(parameters, held)`, or `held.reconstruct(
    parameters)` when `held` is a `ParameterSubspace`.
    """
    if isinstance(fixed, SubspaceTrainingTree):
        return parameters.selected, fixed.complement.subspace
    return parameters, eqx.combine(model_state, fixed)


def functional_parameter_lane(tree: Any, /) -> Any:
    """Legacy trained lane of a kernel PARAMETER lane (identity without subspace)."""
    return tree.selected if isinstance(tree, SubspaceTrainingTree) else tree


def _site_keys(
    root: Key[Array, ""], attempt: Array, microstep: Array, sites: tuple[str, ...], /
) -> tuple[Key[Array, ""], ...]:
    return tuple(
        training_site_key(
            root,
            objective_id=FUNCTIONAL_OBJECTIVE_ID,
            site=site,
            attempt=attempt,
            microstep=microstep,
        )
        for site in sites
    )


# One compiled dispatch derives every site key an attempt needs.
_compiled_site_keys = eqx.filter_jit(_site_keys)


def functional_site_keys(
    state: TrainingKernelState, /, *sites: str
) -> tuple[Key[Array, ""], ...]:
    """Attempt-addressed keys of host-side FunctionalSolver sites.

    Refresh, sampling, term selection, NTK probes, selection, and reporting
    realizations are fresh on every attempt (including retries after a
    rejection) and distinct across microsteps of one accumulation window.
    """
    return _compiled_site_keys(
        state.root_key, state.attempt_cursor, state.microstep, sites
    )


def functional_site_key(state: TrainingKernelState, site: str, /) -> Key[Array, ""]:
    """Attempt-addressed key of one host-side FunctionalSolver site."""
    return functional_site_keys(state, site)[0]


@final
class FunctionalKernelObjective(StrictModule):
    """Kernel objective function over the legacy functional lanes.

    `loss` must be a module-level function or a callable module whose arrays are
    visible FIXED fields, so the kernel's objective role preflight can see them.
    """

    loss: FunctionalLoss

    def __call__(
        self, parameters: Any, model_state: Any, fixed: Any, payload: Any, keys: Any
    ) -> tuple[_ObjectiveContribution, Any, Any]:
        trained, held = functional_lanes(parameters, model_state, fixed)
        total, diagnostics = self.loss(trained, held, payload, keys)
        total = jnp.asarray(total)
        dtype = jnp.real(total).dtype
        contribution = _ObjectiveContribution(
            total,
            jnp.ones((), dtype=dtype),
            jnp.zeros((), dtype=dtype),
        )
        # No functional objective evolves model state: it is carried unchanged.
        return contribution, model_state, diagnostics


def functional_kernel_objective(loss: FunctionalLoss, /) -> KernelObjective:
    """The single FunctionalSolver objective: the ordered total as `(total, 1, 0)`."""
    return KernelObjective(
        objective_id=FUNCTIONAL_OBJECTIVE_ID,
        kind=FUNCTIONAL_OBJECTIVE_KIND,
        route=DerivativeRoute.DIRECT,
        fn=FunctionalKernelObjective(loss),
    )


def resume_functional_kernel_state(
    kernel: PreparedTrainingKernel,
    state: TrainingKernelState,
    checkpoint_id: str,
    /,
) -> TrainingKernelState:
    """Verify an in-memory kernel state against `kernel` before resuming it.

    The stored checkpoint identity must equal the kernel's (role schema,
    objective, rule, authorities); the state's structures must match the
    kernel's (fails closed with `ValueError`).
    """
    if checkpoint_id != kernel.checkpoint_id:
        raise ValueError(
            "In-memory functional training identity mismatch: the resumed run's "
            "roles, objective, or update rule differ from the stored state's."
        )
    payload = build_training_checkpoint(kernel, state, allow_intermediate=True)
    return restore_training_checkpoint(kernel, payload).state


__all__ = [
    "FUNCTIONAL_OBJECTIVE_ID",
    "FUNCTIONAL_OBJECTIVE_KIND",
    "FUNCTIONAL_REJECTION_BUDGET",
    "FUNCTIONAL_ROOT_AUTHORITY",
    "FunctionalKernelObjective",
    "FunctionalLoss",
    "functional_kernel_objective",
    "functional_lanes",
    "functional_parameter_lane",
    "functional_site_key",
    "functional_site_keys",
    "functional_training_tree",
    "functional_tree_functions",
    "resume_functional_kernel_state",
]
