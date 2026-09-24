#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Solver objectives: training signals formed by fixed prepared solves.

A solver objective holds one FIXED prepared solve and binds the trained
component, held separately in the trained tree, into it on every evaluation
through the owner's refresh or binding function. The objective's kind and route
decide which component authorities it may train. Every stochastic component must
be bound to one frozen realization. Failed cases reduce the support or reject
the attempt; they never contribute a plausible value or derivative.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._differentiation import (
    admit_regularity,
    authority_admits,
    ComponentAuthority,
    DerivativeRoute,
    DerivativeSurface,
    DifferentiationRequest,
    ObjectiveKind,
    RegularityPolicy,
)
from .._fingerprint import canonical_fingerprint
from .._model import AbstractArrayModel, ComponentBinding
from .._model._component import AbstractComponentSlot, admit_randomness
from .._model._realization import FrozenRealization
from .._strict import StrictModule
from .._trainable import (
    ArrayRole,
    combine_parameters,
    ExplicitFreeze,
    partition_parameters,
    require_parameter_roles,
    resolve_array_roles,
)
from .._training_kernel import (
    _component_authorities,
    _structure_signature,
    KernelObjective,
)
from .._training_objective import _ObjectiveContribution
from .._tree_math import tree_inner


AcceptedResultPolicy = Literal["reduce-support", "reject-attempt"]
_ACCEPTED_RESULT_POLICIES = ("reduce-support", "reject-attempt")
_DERIVATIVE_FREE_ALTERNATIVES = (
    "derivative-free consumers remain available: train_components with a "
    "distribution-evolution optimizer, or posterior_problem_from_solver_objective "
    "with fit_eki"
)
_RANDOMNESS_HINTS = {
    "randomness-undeclared": (
        "declare a RandomnessContract in the model's model_execution_contract()"
    ),
    "inference-state-unbound": (
        "evaluate the model in its inference state (phydrax.nn.layers.inference_mode)"
    ),
    "realization-unbound": (
        "bind one realization with phydrax.FrozenRealization(model, key, "
        "realization_id=...)"
    ),
    "realization-unidentified": "declare the realization_id of the fixed realization",
    "resampled-randomness-not-admitted": (
        "bind one realization with phydrax.FrozenRealization(model, key, "
        "realization_id=...)"
    ),
}


def _identifier(value: Any, name: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _scalar(value: Any, name: str, /, *, kind: str) -> Array:
    array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar.")
    match kind:
        case "bool":
            valid = array.dtype == jnp.bool_
        case "integer":
            valid = jnp.issubdtype(array.dtype, jnp.integer)
        case "real":
            valid = jnp.issubdtype(array.dtype, jnp.floating)
        case _:
            raise ValueError(f"Unknown scalar kind {kind!r}.")
    if not valid:
        raise TypeError(f"{name} must be a {kind} scalar array.")
    return array


def _inexact_tree(value: Any, name: str, /) -> PyTree[Array]:
    leaves = jax.tree_util.tree_leaves(value)
    if not leaves or any(not eqx.is_inexact_array(leaf) for leaf in leaves):
        raise TypeError(f"{name} must be a nonempty PyTree of inexact arrays.")
    return value


def _array_tree(value: Any, name: str, /) -> PyTree[Array]:
    if any(not eqx.is_array(leaf) for leaf in jax.tree_util.tree_leaves(value)):
        raise TypeError(f"{name} must be a PyTree of arrays.")
    return value


# Per-case outcomes ------------------------------------------------------------------


@final
class SolverCaseResult(StrictModule):
    """Outcome of one case of a `SolverObjective` or `RolloutObjective`.

    The case loss is `value` (a real scalar) or, for a residual-valued case,
    `0.5 * ||residual||^2` over every array leaf of `residual`; exactly one of
    them is given. Only residual-valued objectives feed
    `posterior_problem_from_solver_objective`. `accepted` is the owner's primal
    status of the case (for example a root result's `successful`); a case with a
    nonfinite loss also counts as failed. `aux` is an array PyTree kept for
    inspection, such as a proposal next to its corrected state; its derivatives
    are stopped.
    """

    value: Array | None
    residual: PyTree[Array] | None
    accepted: Array
    aux: PyTree[Array]

    def __init__(
        self,
        *,
        accepted: Any,
        value: Any = None,
        residual: PyTree[Any] | None = None,
        aux: PyTree[Any] = (),
    ):
        if (value is None) == (residual is None):
            raise ValueError("Give exactly one of value and residual.")
        value_ = None if value is None else _scalar(value, "value", kind="real")
        residual_ = None if residual is None else _inexact_tree(residual, "residual")
        accepted_ = _scalar(accepted, "accepted", kind="bool")
        aux_ = _array_tree(aux, "aux")
        self.value = value_
        self.residual = residual_
        self.accepted = accepted_
        self.aux = aux_

    @property
    def loss(self) -> Array:
        """Case loss: `value`, or half the squared residual norm."""
        if self.value is not None:
            return self.value
        return 0.5 * tree_inner(self.residual, self.residual)


@final
class AlgorithmicWorkResult(StrictModule):
    """Outcome of one fixed-work case of an `AlgorithmicWorkObjective`.

    `initial_residual` and `final_residual` are the owner's original-problem
    residuals (vectors, or their norms) before and after the fixed work;
    `iterations` is the work the owner performed and `accepted` its primal
    validity (finite, no breakdown). `aux` is an array PyTree kept for
    inspection; its derivatives are stopped.
    """

    initial_residual: PyTree[Array]
    final_residual: PyTree[Array]
    iterations: Array
    accepted: Array
    aux: PyTree[Array]

    def __init__(
        self,
        *,
        initial_residual: PyTree[Any],
        final_residual: PyTree[Any],
        iterations: Any,
        accepted: Any,
        aux: PyTree[Any] = (),
    ):
        initial = _inexact_tree(initial_residual, "initial_residual")
        final = _inexact_tree(final_residual, "final_residual")
        iterations_ = _scalar(iterations, "iterations", kind="integer")
        accepted_ = _scalar(accepted, "accepted", kind="bool")
        aux_ = _array_tree(aux, "aux")
        self.initial_residual = initial
        self.final_residual = final
        self.iterations = iterations_
        self.accepted = accepted_
        self.aux = aux_


def _safe_norm(tree: PyTree[Array], /) -> Array:
    """Euclidean norm with an exact zero derivative at an exactly zero residual."""
    squared = tree_inner(tree, tree)
    positive = squared > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1.0)), 0.0)


def algorithmic_work_loss(result: AlgorithmicWorkResult, /) -> Array:
    """`log((||r_k|| + floor) / (||r_0|| + floor))` of one fixed-work case.

    The reference `||r_0||` and the floor `eps * ||r_0|| + tiny` (in the final
    residual's real dtype) are stopped: the loss rewards only the final
    residual after the fixed work, saturates at the working precision, and is
    finite with a zero derivative at an exactly zero residual.
    """
    if not isinstance(result, AlgorithmicWorkResult):
        raise TypeError("result must be an AlgorithmicWorkResult.")
    final = _safe_norm(result.final_residual)
    initial = jax.lax.stop_gradient(
        _safe_norm(result.initial_residual).astype(final.dtype)
    )
    info = jnp.finfo(final.dtype)
    floor = jnp.asarray(info.eps, final.dtype) * initial + jnp.asarray(
        info.tiny, final.dtype
    )
    return jnp.log(final + floor) - jnp.log(initial + floor)


# Evaluation record ------------------------------------------------------------------


@final
class SolverObjectiveEvaluation(StrictModule):
    """Value, support, primal status, and evidence of one objective evaluation.

    `value` is the support-normalized objective (zero at zero support, and not
    a number when a `"reject-attempt"` objective saw a failed case). `accepted`
    is the per-case primal status (scalar without cases), `case_values` the
    per-case losses, `work` the per-case iterations of a fixed-work objective,
    and `aux` the stacked per-case inspection data. `trained` and `stopped` list
    the `(path, authority)` PARAMETER leaves this objective trains and holds
    fixed; `realization_evidence` and `derivative_evidence` are the component
    admission records; `binding_identity` content-addresses the objective and
    its bound component structure.
    """

    value: Array
    support: Array
    accepted: Array
    case_values: Array
    work: Array | None
    aux: PyTree[Array]
    objective_id: str = eqx.field(static=True)
    kind: ObjectiveKind = eqx.field(static=True)
    route: DerivativeRoute = eqx.field(static=True)
    trained: tuple[tuple[str, str], ...] = eqx.field(static=True)
    stopped: tuple[tuple[str, str], ...] = eqx.field(static=True)
    realization_evidence: tuple[str, ...] = eqx.field(static=True)
    derivative_evidence: tuple[str, ...] = eqx.field(static=True)
    binding_identity: str = eqx.field(static=True)

    @property
    def failures(self) -> Array:
        """Number of failed cases."""
        return jnp.sum(~self.accepted).astype(jnp.int32)


# Admission --------------------------------------------------------------------------


_OWNER_TYPES = (AbstractComponentSlot, ComponentBinding, AbstractArrayModel)


@final
class SolverObjectiveAdmission(StrictModule):
    """Static admission of one solver objective against one trained tree.

    `trained` and `stopped` are the `(path, authority)` PARAMETER leaves of the
    selected component this objective trains and holds fixed.
    `realization_evidence` records each component model's randomness decision,
    `derivative_evidence` each trained model's derivative admission, and
    `derivative_failures` the reasons a differentiating consumer is refused.
    """

    trained: tuple[tuple[str, str], ...] = eqx.field(static=True)
    stopped: tuple[tuple[str, str], ...] = eqx.field(static=True)
    realization_evidence: tuple[str, ...] = eqx.field(static=True)
    derivative_evidence: tuple[str, ...] = eqx.field(static=True)
    derivative_failures: tuple[str, ...] = eqx.field(static=True)
    binding_identity: str = eqx.field(static=True)


def _location(keys: tuple[Any, ...], /) -> str:
    return jax.tree_util.keystr(keys) or "<root>"


def _classify(
    node: Any,
    keys: tuple[Any, ...],
    authority: ComponentAuthority | None,
    owners: list[tuple[str, str, str]],
    models: list[tuple[str, AbstractArrayModel, ComponentAuthority | None]],
    /,
) -> None:
    location = _location(keys)
    if isinstance(node, ComponentBinding):
        owners.append((location, node.authority.value, node.contract().bound_semantic_id))
        models.append((location, node.model, node.authority))
    elif isinstance(node, AbstractComponentSlot):
        own = type(node).component_authority
        owners.append((location, own.value, type(node).slot_semantic_id))
        if isinstance(node, AbstractArrayModel):
            models.append((location, node, own))
        else:
            _visit(node, keys, own, owners, models)
    elif isinstance(node, AbstractArrayModel):
        models.append((location, node, authority))
    else:
        _visit(node, keys, authority, owners, models)


def _visit(
    node: Any,
    keys: tuple[Any, ...],
    authority: ComponentAuthority | None,
    owners: list[tuple[str, str, str]],
    models: list[tuple[str, AbstractArrayModel, ComponentAuthority | None]],
    /,
) -> None:
    entries = jax.tree_util.tree_flatten_with_path(
        node, is_leaf=lambda value: value is not node and isinstance(value, _OWNER_TYPES)
    )[0]
    for path, child in entries:
        if child is not node and isinstance(child, _OWNER_TYPES):
            _classify(child, keys + tuple(path), authority, owners, models)


def _inventory(
    tree: PyTree[Any], /
) -> tuple[
    tuple[tuple[str, str, str], ...],
    tuple[tuple[str, AbstractArrayModel, ComponentAuthority | None], ...],
]:
    """Owning slots and bindings, and the outermost models with their authority."""
    owners: list[tuple[str, str, str]] = []
    models: list[tuple[str, AbstractArrayModel, ComponentAuthority | None]] = []
    _classify(tree, (), None, owners, models)
    return tuple(owners), tuple(models)


def _has_parameters(model: AbstractArrayModel, /) -> bool:
    return any(role is ArrayRole.PARAMETER for role in resolve_array_roles(model).roles)


# Objectives -------------------------------------------------------------------------


@final
class _FixedHolder(StrictModule, ExplicitFreeze):
    """FIXED holder of an objective's prepared solve or case data."""

    value: Any


class AbstractSolverObjective(StrictModule):
    """Training objective formed by one fixed prepared solve.

    `solve` is the FIXED prepared solve. The trained component lives in the
    tree handed to `evaluate` or `train_components`, separate from the solve:
    `component(tree)` selects it (`None` selects the whole tree) and
    `bind(solve, component)` binds it into the solve through the owner's
    refresh or binding path on every evaluation (D26). `measure(owner, case)`
    runs the bound owner on one case of `cases` (a PyTree with a leading case
    axis, or `None` for one case evaluated as `measure(owner, None)`).
    Callables must hide no inexact arrays: data belongs in `solve` or `cases`.

    Authority comes only from the owning component slots and
    `ComponentBinding`s of the selected component. Every PARAMETER leaf must
    have one; at least one authority group must admit the objective's
    `(route, kind)`, and the remaining groups are held fixed (stop-gradient) in
    this objective and need a separate compatible objective to train. Every
    component model must be deterministic or bound to one `FrozenRealization`.
    Differentiating consumers also require each trained model's derivative
    contract to admit the objective's route under `regularity_policy`.

    `accepted_results` reduces failed cases: `"reject-attempt"` (default) turns
    any failed case into a nonfinite value, which the training kernel rolls
    back; `"reduce-support"` removes failed cases from the value and support
    and gates their derivatives to exact zeros. `case_batch_size` evaluates the
    cases with a bounded `lax.map` working set instead of one `vmap`.
    """

    solve: _FixedHolder
    cases: _FixedHolder
    bind: Callable[[Any, Any], Any]
    measure: Callable[[Any, Any], Any]
    component: Callable[[Any], Any] | None
    regularity_policy: RegularityPolicy
    objective_id: str = eqx.field(static=True)
    weight: float = eqx.field(static=True)
    accepted_results: AcceptedResultPolicy = eqx.field(static=True)
    case_batch_size: int | None = eqx.field(static=True)
    kind: eqx.AbstractClassVar[ObjectiveKind]
    route: eqx.AbstractClassVar[DerivativeRoute]

    @property
    def context(self) -> str:
        return f"solver objective {self.objective_id!r}"

    def select(self, tree: PyTree[Any], /) -> Any:
        """The trained component of `tree` this objective binds."""
        return tree if self.component is None else self.component(tree)

    # Admission ----------------------------------------------------------------------

    def admit(
        self, tree: PyTree[Any], /, *, differentiable: bool = True
    ) -> SolverObjectiveAdmission:
        """Admit the selected component before anything is traced.

        Raises `ValueError` for an unowned PARAMETER leaf, a component whose
        authority admits no training signal from this objective, and a
        stochastic model without a frozen realization. A trained model whose
        derivative contract does not admit the route is recorded in
        `derivative_failures` and raises only when `differentiable`: derivative-free
        consumers remain admitted (A11), and are never selected silently.
        """
        if not isinstance(differentiable, bool):
            raise TypeError("differentiable must be a bool.")
        component = self.select(tree)
        resolution = require_parameter_roles(component, context=self.context)
        parameters = tuple(
            path
            for path, role in zip(resolution.paths, resolution.roles, strict=True)
            if role is ArrayRole.PARAMETER
        )
        if not parameters:
            raise ValueError(f"{self.context}: the component has no PARAMETER leaves.")
        authorities = _component_authorities(component)[0]
        unowned = tuple(path for path in parameters if authorities[path] is None)
        if unowned:
            raise ValueError(
                f"{self.context}: parameters at {unowned!r} have no owning component "
                "slot or ComponentBinding; solver objectives take authority only from "
                "slots and bindings."
            )
        trained = tuple(
            (path, authorities[path].value)
            for path in parameters
            if authority_admits(authorities[path], self.route, self.kind)
        )
        stopped = tuple(
            (path, authorities[path].value)
            for path in parameters
            if not authority_admits(authorities[path], self.route, self.kind)
        )
        if not trained:
            groups = sorted({authority for _, authority in stopped})
            raise ValueError(
                f"{self.context}: no admissible training signal for {groups!r} "
                f"parameters at {tuple(path for path, _ in stopped)!r} under "
                f"({self.route.value}, {self.kind.value}); train them with an "
                "objective their authority admits."
            )
        owners, models = _inventory(component)
        realization: list[str] = []
        derivative: list[str] = []
        failures: list[str] = []
        for location, model, authority in models:
            realization.append(self._admit_randomness(location, model))
            owner = ComponentAuthority.MODEL if authority is None else authority
            if authority_admits(owner, self.route, self.kind):
                records, failure = self._admit_derivative(location, model, owner)
                derivative.extend(records)
                failures.extend(() if failure is None else (failure,))
        if differentiable and failures:
            raise ValueError(
                f"{self.context}: {failures[0]}; {_DERIVATIVE_FREE_ALTERNATIVES}."
            )
        identity = canonical_fingerprint(
            {
                "kind": "solver-objective-binding",
                "objective_id": self.objective_id,
                "objective_kind": self.kind.value,
                "route": self.route.value,
                "owners": [list(owner) for owner in sorted(owners)],
                "trained": [list(entry) for entry in trained],
                "stopped": [list(entry) for entry in stopped],
                "parameters": _structure_signature(partition_parameters(component)[0]),
            }
        )
        return SolverObjectiveAdmission(
            trained=trained,
            stopped=stopped,
            realization_evidence=tuple(sorted(realization)),
            derivative_evidence=tuple(sorted(derivative)),
            derivative_failures=tuple(failures),
            binding_identity=identity,
        )

    def _admit_randomness(self, location: str, model: AbstractArrayModel, /) -> str:
        contract = model.model_execution_contract()
        admitted, reason = admit_randomness(
            contract.randomness,
            implicit=self.route is DerivativeRoute.IMPLICIT,
            authoritative=True,
            realization_bound=isinstance(model, FrozenRealization),
            inference_state_bound=False,
        )
        if not admitted:
            hint = _RANDOMNESS_HINTS.get(reason)
            raise ValueError(
                f"{self.context}: component {location} ({type(model).__name__}) needs "
                f"one frozen realization: {reason}"
                + ("." if hint is None else f"; {hint}.")
            )
        return f"{location}:{reason}"

    def _admit_derivative(
        self,
        location: str,
        model: AbstractArrayModel,
        authority: ComponentAuthority,
        /,
    ) -> tuple[tuple[str, ...], str | None]:
        """Derivative evidence records and the failure of one trained model."""
        contract = model.model_execution_contract()
        route = contract.derivative.route
        name = f"component {location} ({type(model).__name__})"
        if contract.execution.host_only or route is DerivativeRoute.STOPPED:
            return (f"{location}:derivative-unsupported",), (
                f"{name} declares no JAX derivative (route {route.value!r}, "
                f"host_only={contract.execution.host_only})"
            )
        # The owner differentiates through the model's value argument (its
        # INPUT surface) and, for trained models, its parameters; the implicit
        # route additionally needs classical C^1 regularity of that value map.
        surfaces = (DerivativeSurface.INPUT,) + (
            (DerivativeSurface.MODEL_PARAMETER,) if _has_parameters(model) else ()
        )
        request = DifferentiationRequest(surfaces, authority=authority)
        admissions = [contract.derivative.admit(request, policy=self.regularity_policy)]
        if self.route is DerivativeRoute.IMPLICIT:
            admissions.append(
                admit_regularity(
                    contract.regularity,
                    request,
                    route=DerivativeRoute.IMPLICIT,
                    policy=self.regularity_policy,
                )
            )
        reasons = sorted({reason for entry in admissions for reason in entry.reasons})
        if reasons:
            return (f"{location}:derivative-unsupported",), (
                f"{name} is not admitted for {self.route.value} differentiation: "
                f"{', '.join(reasons)}"
            )
        conditions = sorted(
            {condition for entry in admissions for condition in entry.conditions}
        )
        return (f"{location}:derivative-supported",) + tuple(
            f"{location}:{condition}" for condition in conditions
        ), None

    # Evaluation ---------------------------------------------------------------------

    def _case_loss(self, result: Any, /) -> tuple[Array, Array, Array | None]:
        """`(loss, accepted, iterations)` of one measured case."""
        if not isinstance(result, SolverCaseResult):
            raise TypeError(
                f"{self.context}: measure must return a SolverCaseResult, got "
                f"{type(result).__name__}."
            )
        return result.loss, result.accepted, None

    def _case(self, component: Any, case: Any, /) -> tuple[Array, tuple[Array, Any]]:
        """`(loss, (status, aux))` of one case.

        `status` holds `(accepted, iterations)` in the loss dtype: outputs of the
        failure gate must be inexact, so integer and Boolean evidence crosses it
        as exact small floats and is recovered in `_evaluate_cases`.
        """
        result = self.measure(self.bind(self.solve.value, component), case)
        loss, accepted, iterations = self._case_loss(result)
        loss = jnp.asarray(loss)
        if loss.shape != () or not jnp.issubdtype(loss.dtype, jnp.floating):
            raise ValueError(f"{self.context}: the case loss must be a real scalar.")
        status = jnp.stack(
            (
                (accepted & jnp.isfinite(loss)).astype(loss.dtype),
                jnp.zeros((), loss.dtype)
                if iterations is None
                else iterations.astype(loss.dtype),
            )
        )
        return loss, (jax.lax.stop_gradient(status), jax.lax.stop_gradient(result.aux))

    def _evaluate_cases(
        self, component: Any, /
    ) -> tuple[Array, Array, Array | None, Any]:
        """Per-case losses, acceptance, work, and aux, failure-gated per case."""

        def one(case: Any) -> Any:
            return _gated_case(component, self, case)

        cases = self.cases.value
        if cases is None:
            loss, (status, aux) = one(None)
        elif self.case_batch_size is None:
            loss, (status, aux) = jax.vmap(one)(cases)
        else:
            loss, (status, aux) = jax.lax.map(one, cases, batch_size=self.case_batch_size)
        status = jax.lax.stop_gradient(status)
        accepted = status[..., 0] > 0
        iterations = (
            status[..., 1].astype(jnp.int32)
            if self.kind is ObjectiveKind.ALGORITHMIC_WORK
            else None
        )
        return loss, accepted, iterations, aux

    def _residual_cases(self, component: Any, /) -> tuple[Array, Array]:
        """Stacked real residual vectors and primal status of residual-valued cases.

        Serves likelihood consumers, which need the residual itself rather than
        its reduced loss; a nonfinite residual counts as a failed case.
        """

        def one(case: Any) -> tuple[Array, Array]:
            result = self.measure(self.bind(self.solve.value, component), case)
            if not isinstance(result, SolverCaseResult) or result.residual is None:
                raise ValueError(
                    f"{self.context}: a likelihood needs residual-valued "
                    "SolverCaseResult cases."
                )
            leaves = jax.tree_util.tree_leaves(result.residual)
            if any(jnp.iscomplexobj(leaf) for leaf in leaves):
                raise ValueError(f"{self.context}: likelihood residuals must be real.")
            vector = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])
            return vector, result.accepted & jnp.all(jnp.isfinite(vector))

        cases = self.cases.value
        if cases is None:
            vector, accepted = one(None)
            return vector[None], accepted[None]
        if self.case_batch_size is None:
            return jax.vmap(one)(cases)
        return jax.lax.map(one, cases, batch_size=self.case_batch_size)

    def _reduce(self, loss: Array, accepted: Array, /) -> tuple[Array, Array]:
        """Numerator and stop-gradient support of the case losses."""
        total = jnp.sum(jnp.where(accepted, loss, jnp.zeros_like(loss)))
        match self.accepted_results:
            case "reduce-support":
                return total, jnp.sum(accepted).astype(loss.dtype)
            case "reject-attempt":
                numerator = jnp.where(
                    jnp.all(accepted), total, jnp.asarray(jnp.nan, loss.dtype)
                )
                return numerator, jnp.asarray(accepted.size, loss.dtype)
            case _:
                raise ValueError(
                    f"Unknown accepted-result policy {self.accepted_results!r}."
                )

    def evaluate(
        self, tree: PyTree[Any], /, *, differentiable: bool = True
    ) -> SolverObjectiveEvaluation:
        """Admit, bind, and evaluate the objective on the trained `tree`.

        Admission runs first, so a refused component never reaches `bind` or
        `measure`. Parameters this objective does not train are stop-gradient.
        """
        admission = self.admit(tree, differentiable=differentiable)
        stopped = frozenset(path for path, _ in admission.stopped)
        component = jax.tree_util.tree_map_with_path(
            lambda path, leaf: (
                jax.lax.stop_gradient(leaf)
                if eqx.is_array(leaf) and _location(path) in stopped
                else leaf
            ),
            self.select(tree),
        )
        loss, accepted, iterations, aux = self._evaluate_cases(component)
        numerator, support = self._reduce(loss, accepted)
        return SolverObjectiveEvaluation(
            value=_ObjectiveContribution(numerator, support).value,
            support=support,
            accepted=accepted,
            case_values=loss,
            work=iterations,
            aux=aux,
            objective_id=self.objective_id,
            kind=self.kind,
            route=self.route,
            trained=admission.trained,
            stopped=admission.stopped,
            realization_evidence=admission.realization_evidence,
            derivative_evidence=admission.derivative_evidence,
            binding_identity=admission.binding_identity,
        )


@eqx.filter_custom_vjp
def _gated_case(component: Any, objective: AbstractSolverObjective, case: Any) -> Any:
    return objective._case(component, case)


@_gated_case.def_fwd
def _gated_case_fwd(
    perturbed: Any, component: Any, objective: AbstractSolverObjective, case: Any
) -> Any:
    del perturbed
    arrays, static = eqx.partition(component, eqx.is_inexact_array)
    loss, pullback, rest = jax.vjp(
        lambda values: objective._case(eqx.combine(values, static), case),
        arrays,
        has_aux=True,
    )
    return (loss, rest), (pullback, rest[0][0] > 0)


@_gated_case.def_bwd
def _gated_case_bwd(
    residuals: Any,
    grad_obj: Any,
    perturbed: Any,
    component: Any,
    objective: AbstractSolverObjective,
    case: Any,
) -> Any:
    # A failed case contributes an exact zero derivative, selected rather than
    # multiplied, so a nonfinite pullback of a failed solve cannot leak.
    del component, objective, case
    pullback, accepted = residuals
    cotangent = grad_obj[0]
    if cotangent is None:
        return jax.tree.map(lambda _: None, perturbed)
    (gradient,) = pullback(cotangent)
    gated = jax.tree.map(
        lambda leaf: jnp.where(accepted, leaf, jnp.zeros_like(leaf)), gradient
    )
    return eqx.filter(gated, perturbed)


def _objective_fields(
    objective: AbstractSolverObjective,
    solve: Any,
    bind: Any,
    measure: Any,
    /,
    *,
    objective_id: Any,
    cases: Any,
    component: Any,
    accepted_results: Any,
    weight: Any,
    case_batch_size: Any,
    regularity_policy: Any,
) -> None:
    """Validate and assign the fields shared by every solver objective."""
    if not callable(bind):
        raise TypeError("bind must be callable.")
    if not callable(measure):
        raise TypeError("measure must be callable.")
    if component is not None and not callable(component):
        raise TypeError("component must be a callable selector or None.")
    identifier = _identifier(objective_id, "objective_id")
    if accepted_results not in _ACCEPTED_RESULT_POLICIES:
        raise ValueError(
            f"accepted_results must be one of {_ACCEPTED_RESULT_POLICIES!r}."
        )
    if isinstance(weight, (bool, np.bool_)) or not isinstance(
        weight, (int, float, np.integer, np.floating)
    ):
        raise TypeError("weight must be a real number.")
    weight_ = float(weight)
    if not np.isfinite(weight_) or weight_ <= 0.0:
        raise ValueError("weight must be finite and positive.")
    if case_batch_size is not None and (
        isinstance(case_batch_size, bool)
        or not isinstance(case_batch_size, int)
        or case_batch_size <= 0
    ):
        raise ValueError("case_batch_size must be a positive integer or None.")
    if cases is not None:
        leaves = jax.tree_util.tree_leaves(cases)
        if not leaves or any(not eqx.is_array(leaf) for leaf in leaves):
            raise TypeError("cases must be a nonempty PyTree of arrays or None.")
        counts = {leaf.shape[0] if leaf.ndim else None for leaf in leaves}
        if len(counts) != 1 or None in counts or 0 in counts:
            raise ValueError("Every cases leaf must share one positive leading axis.")
    policy = RegularityPolicy() if regularity_policy is None else regularity_policy
    if not isinstance(policy, RegularityPolicy):
        raise TypeError("regularity_policy must be a RegularityPolicy or None.")
    objective.solve = _FixedHolder(solve)
    objective.cases = _FixedHolder(cases)
    objective.bind = bind
    objective.measure = measure
    objective.component = component
    objective.regularity_policy = policy
    objective.objective_id = identifier
    objective.weight = weight_
    objective.accepted_results = accepted_results
    objective.case_batch_size = case_batch_size


@final
class SolverObjective(AbstractSolverObjective):
    """Solution-map objective differentiated implicitly through a converged solve.

    Kind `SOLUTION_MAP`, route `IMPLICIT`: `measure` scores the accepted
    solution of the bound owner (for example an implicit root result against
    observations). Implicit differentiation needs classical `C^1` component
    regularity in the primal state. Accelerator components change solver work,
    not the solution, and are never admitted.
    """

    kind: ClassVar[ObjectiveKind] = ObjectiveKind.SOLUTION_MAP
    route: ClassVar[DerivativeRoute] = DerivativeRoute.IMPLICIT

    def __init__(
        self,
        solve: Any,
        bind: Callable[[Any, Any], Any],
        measure: Callable[[Any, Any], SolverCaseResult],
        /,
        *,
        objective_id: str,
        cases: PyTree[Any] | None = None,
        component: Callable[[Any], Any] | None = None,
        accepted_results: AcceptedResultPolicy = "reject-attempt",
        weight: float = 1.0,
        case_batch_size: int | None = None,
        regularity_policy: RegularityPolicy | None = None,
    ):
        _objective_fields(
            self,
            solve,
            bind,
            measure,
            objective_id=objective_id,
            cases=cases,
            component=component,
            accepted_results=accepted_results,
            weight=weight,
            case_batch_size=case_batch_size,
            regularity_policy=regularity_policy,
        )


@final
class RolloutObjective(AbstractSolverObjective):
    """Trajectory objective differentiated through an unrolled rollout.

    Kind `ROLLOUT`, route `UNROLLED`: `measure` runs the bound owner's rollout
    (finite-volume rollout plans, finite-difference adjoint rollouts, DEM
    replay, learned incompressible transitions, kinetic rollouts, or a native
    corrector re-evaluating a learned proposal) and scores the trajectory.
    `checkpointed=True` rematerializes each case in reverse mode instead of
    storing its trajectory.
    """

    checkpointed: bool = eqx.field(static=True)
    kind: ClassVar[ObjectiveKind] = ObjectiveKind.ROLLOUT
    route: ClassVar[DerivativeRoute] = DerivativeRoute.UNROLLED

    def __init__(
        self,
        solve: Any,
        bind: Callable[[Any, Any], Any],
        measure: Callable[[Any, Any], SolverCaseResult],
        /,
        *,
        objective_id: str,
        cases: PyTree[Any] | None = None,
        component: Callable[[Any], Any] | None = None,
        accepted_results: AcceptedResultPolicy = "reject-attempt",
        weight: float = 1.0,
        case_batch_size: int | None = None,
        regularity_policy: RegularityPolicy | None = None,
        checkpointed: bool = False,
    ):
        if not isinstance(checkpointed, bool):
            raise TypeError("checkpointed must be a bool.")
        _objective_fields(
            self,
            solve,
            bind,
            measure,
            objective_id=objective_id,
            cases=cases,
            component=component,
            accepted_results=accepted_results,
            weight=weight,
            case_batch_size=case_batch_size,
            regularity_policy=regularity_policy,
        )
        self.checkpointed = checkpointed

    def _case(self, component: Any, case: Any, /) -> tuple[Array, tuple[Array, Any]]:
        if not self.checkpointed:
            return AbstractSolverObjective._case(self, component, case)
        # The objective is an explicit argument, so the rematerialized case
        # closes over no traced array.
        return eqx.filter_checkpoint(AbstractSolverObjective._case)(self, component, case)


@final
class AlgorithmicWorkObjective(AbstractSolverObjective):
    """Fixed-work objective of an accelerator inside a native iteration.

    Kind `ALGORITHMIC_WORK`, route `UNROLLED`: `measure` runs the bound owner
    for exactly `work` iterations without early exit and returns an
    `AlgorithmicWorkResult` of the original problem's residuals. The case loss
    is `algorithmic_work_loss`, `log((||r_k|| + floor) / (||r_0|| + floor))`
    with a stopped precision-aware floor. A case whose reported iterations
    differ from `work` (an early exit or a breakdown) counts as failed, so the
    work evidence reaches the result.
    """

    work: int = eqx.field(static=True)
    kind: ClassVar[ObjectiveKind] = ObjectiveKind.ALGORITHMIC_WORK
    route: ClassVar[DerivativeRoute] = DerivativeRoute.UNROLLED

    def __init__(
        self,
        solve: Any,
        bind: Callable[[Any, Any], Any],
        measure: Callable[[Any, Any], AlgorithmicWorkResult],
        /,
        *,
        work: int,
        objective_id: str,
        cases: PyTree[Any] | None = None,
        component: Callable[[Any], Any] | None = None,
        accepted_results: AcceptedResultPolicy = "reject-attempt",
        weight: float = 1.0,
        case_batch_size: int | None = None,
        regularity_policy: RegularityPolicy | None = None,
    ):
        if isinstance(work, bool) or not isinstance(work, int) or work <= 0:
            raise ValueError("work must be a positive integer.")
        _objective_fields(
            self,
            solve,
            bind,
            measure,
            objective_id=objective_id,
            cases=cases,
            component=component,
            accepted_results=accepted_results,
            weight=weight,
            case_batch_size=case_batch_size,
            regularity_policy=regularity_policy,
        )
        self.work = work

    def _case_loss(self, result: Any, /) -> tuple[Array, Array, Array | None]:
        if not isinstance(result, AlgorithmicWorkResult):
            raise TypeError(
                f"{self.context}: measure must return an AlgorithmicWorkResult, got "
                f"{type(result).__name__}."
            )
        iterations = result.iterations.astype(jnp.int32)
        accepted = result.accepted & (iterations == self.work)
        return algorithmic_work_loss(result), accepted, iterations


# Kernel lowering --------------------------------------------------------------------


@final
class _KernelSolverObjective(StrictModule):
    """Kernel objective function of one admitted solver objective."""

    objective: AbstractSolverObjective

    def __call__(
        self, parameters: Any, model_state: Any, fixed: Any, payload: Any, keys: Any, /
    ) -> tuple[_ObjectiveContribution, Any, tuple[Array, Array]]:
        # Solver objectives carry their cases and bind one frozen realization;
        # neither the attempt payload nor attempt keys enter the evaluation.
        del payload, keys
        tree = combine_parameters(parameters, model_state, fixed)
        loss, accepted, _, _ = self.objective._evaluate_cases(self.objective.select(tree))
        numerator, support = self.objective._reduce(loss, accepted)
        failures = jnp.sum(~accepted).astype(jnp.int32)
        return (
            _ObjectiveContribution(numerator, support),
            model_state,
            (
                support,
                failures,
            ),
        )


def kernel_objective(objective: AbstractSolverObjective, /) -> KernelObjective:
    """Lower one solver objective to the internal training kernel."""
    if not isinstance(objective, AbstractSolverObjective):
        raise TypeError("objective must be an AbstractSolverObjective.")
    return KernelObjective(
        objective_id=objective.objective_id,
        kind=objective.kind,
        route=objective.route,
        fn=_KernelSolverObjective(objective),
        weight=objective.weight,
    )


__all__ = [
    "AbstractSolverObjective",
    "AcceptedResultPolicy",
    "AlgorithmicWorkObjective",
    "AlgorithmicWorkResult",
    "RolloutObjective",
    "SolverCaseResult",
    "SolverObjective",
    "SolverObjectiveAdmission",
    "SolverObjectiveEvaluation",
    "algorithmic_work_loss",
]
