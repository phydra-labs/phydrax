#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Accepted, fixed-realization physical response pullbacks, without optimization."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, final, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from .._differentiation import (
    authority_admits,
    ComponentAuthority,
    DERIVATIVE_UNSUPPORTED,
    DerivativeRoute,
    DerivativeSurface,
    DifferentiationRequest,
    ObjectiveKind,
    RegularityPolicy,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState, partition_parameters
from .._tree_math import (
    tree_allfinite as _tree_allfinite,
    validate_real_inexact_tree as _validate_real_inexact_tree,
)
from ..linalg import (
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    PreparedLinearSolve,
    PyTreeSpace,
    solve as solve_linear,
    transpose,
)
from ._pde_constrained import (
    _default_adjoint_policy,
    AdjointAcceptanceEvidence,
    StateAcceptanceEvidence,
    StateDesignProblem,
    StateEquationResult,
)


if TYPE_CHECKING:
    from .._model import ComponentBinding


@final
class StateDesignComponentAdmission(StrictModule, NonTrainableState):
    """Owner-side admission of one bound model as the design of a state-design response.

    The design is the bound model's PARAMETER lane (``design(binding)``). A
    response cotangent crosses the owner's accepted transpose solve, which the
    response reports separately as primal and adjoint acceptance evidence, and
    reaches the parameters only through the model's own derivative surfaces:
    ``surfaces`` are admitted by the model's derivative contract under the
    binding's authority and ``policy`` (``conditions`` qualify them), with the
    model's own ``route``. ``kind`` is the caller's declaration of the scientific
    meaning of the objective that consumes the design cotangent (it is not
    inferred from the response); the binding's authority must admit
    ``(route, kind)``. The owner's implicit response never widens that
    admission: a surrogate supplies a physical-residual or data-fit design and is
    refused for a solution-map objective.
    """

    authority: ComponentAuthority = eqx.field(static=True)
    slot_semantic_id: str | None = eqx.field(static=True)
    kind: ObjectiveKind = eqx.field(static=True)
    route: DerivativeRoute = eqx.field(static=True)
    surfaces: tuple[DerivativeSurface, ...] = eqx.field(static=True)
    conditions: tuple[str, ...] = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)
    design_structure: jax.tree_util.PyTreeDef = eqx.field(static=True)
    design_shapes: tuple[tuple[tuple[int, ...], str], ...] = eqx.field(static=True)

    def __init__(
        self,
        binding: ComponentBinding,
        /,
        *,
        kind: ObjectiveKind,
        surfaces: Iterable[DerivativeSurface] = (DerivativeSurface.MODEL_PARAMETER,),
        policy: RegularityPolicy | None = None,
    ) -> None:
        from .._model import ComponentBinding

        if not isinstance(binding, ComponentBinding):
            raise TypeError("binding must be a ComponentBinding.")
        if not isinstance(kind, ObjectiveKind):
            raise TypeError("kind must be an ObjectiveKind.")
        request = DifferentiationRequest(surfaces, authority=binding.authority)
        admission = binding.contract(request=request, policy=policy).derivative_admission
        if admission is None or not admission.supported:
            reasons = () if admission is None else admission.reasons
            raise ValueError(
                f"{DERIVATIVE_UNSUPPORTED}: the bound model does not admit "
                f"{tuple(surface.value for surface in request.surfaces)!r} derivatives "
                f"(reasons: {', '.join(reasons) or 'no derivative admission'})."
            )
        if admission.route is not DerivativeRoute.DIRECT:
            raise ValueError(
                "A state-design response differentiates its design with JAX; the "
                f"model's {admission.route.value!r} route is not a direct route."
            )
        if not authority_admits(binding.authority, admission.route, kind):
            raise ValueError(
                f"A {binding.authority.value} component cannot supply the design of "
                f"a state-design response consumed under ({admission.route.value}, "
                f"{kind.value}); the owner's accepted response does not widen what "
                "the component's authority admits."
            )
        parameters, _, _ = partition_parameters(binding.model)
        leaves, structure = jax.tree_util.tree_flatten(parameters)
        if not leaves:
            raise ValueError("The bound model has no PARAMETER leaves to design.")
        self.authority = binding.authority
        self.slot_semantic_id = binding.slot_semantic_id
        self.kind = kind
        self.route = admission.route
        self.surfaces = request.surfaces
        self.conditions = admission.conditions
        self.contract_id = binding.model.model_execution_contract().contract_id
        self.design_structure = structure
        self.design_shapes = tuple(
            (tuple(leaf.shape), jnp.dtype(leaf.dtype).name) for leaf in leaves
        )

    def design(self, binding: ComponentBinding, /) -> PyTree[Array]:
        """The PARAMETER lane of ``binding``, which must be the admitted binding.

        The binding's authority, slot, and model execution contract must equal
        the admitted ones; a different binding (for example an accelerator in
        the admitted surrogate's place) is refused even when its parameter lane
        has the same structure.
        """
        from .._model import ComponentBinding

        if not isinstance(binding, ComponentBinding):
            raise TypeError("binding must be a ComponentBinding.")
        if binding.authority is not self.authority:
            raise ValueError(
                f"The design binding has {binding.authority.value} authority; the "
                f"admission was granted to a {self.authority.value} binding."
            )
        if binding.slot_semantic_id != self.slot_semantic_id:
            raise ValueError(
                f"The design binding fills slot {binding.slot_semantic_id!r}; the "
                f"admission was granted for slot {self.slot_semantic_id!r}."
            )
        contract_id = binding.model.model_execution_contract().contract_id
        if contract_id != self.contract_id:
            raise ValueError(
                f"The design binding's model contract {contract_id!r} is not the "
                f"admitted model contract {self.contract_id!r}."
            )
        parameters, _, _ = partition_parameters(binding.model)
        self.require_design(parameters)
        return parameters

    def require_design(self, design: PyTree[Any], /) -> None:
        """Raise ``ValueError`` unless ``design`` has the admitted lane's layout.

        Arrays carry no binding identity, so this checks PyTree structure, shapes,
        and dtypes only; obtain the design through ``design(binding)``, which
        also checks that the binding is the admitted one.
        """
        leaves, structure = jax.tree_util.tree_flatten(design)
        shapes = tuple(
            (tuple(jnp.shape(leaf)), jnp.result_type(leaf).name) for leaf in leaves
        )
        if structure != self.design_structure or shapes != self.design_shapes:
            raise ValueError(
                "The state-design design is not the PARAMETER lane of the admitted "
                "component."
            )


_Response: TypeAlias = Callable[[PyTree[Any], PyTree[Any], Any], PyTree[Any]]


class _StateAction(StrictModule):
    action: Any
    template: PyTree[Array]
    pullback: bool = eqx.field(static=True)

    def __call__(self, value: PyTree[Any]) -> PyTree[Array]:
        result = self.action(value)
        if self.pullback:
            result = result[0]
        return jax.tree.map(
            lambda leaf, reference: jnp.asarray(leaf, dtype=reference.dtype).reshape(
                reference.shape
            ),
            result,
            self.template,
        )


class StateDesignLinearization(StrictModule):
    """One immutable state/design point and its reusable residual linearization.

    Construct through ``prepare_state_design_linearization``. There is no numeric
    rebinding operation: a changed design, argument, or realization requires a
    newly accepted state. Residual pullbacks are JAX PyTrees, not dense Jacobians.
    ``component`` is the owner's admission of the bound model whose PARAMETER
    lane is the design, or ``None`` for a plain physical design.
    """

    problem: StateDesignProblem
    state: PyTree[Array]
    design: PyTree[Array]
    args: Any
    residual: PyTree[Array]
    state_jacobian: FunctionLinearOperator
    design_pullback: Any
    state_acceptance: StateAcceptanceEvidence
    linear_policy: LinearSolvePolicy
    state_result: StateEquationResult | None
    component: StateDesignComponentAdmission | None

    @property
    def accepted(self) -> Array:
        return self.state_acceptance.accepted


class StateDesignResponseVJP(StrictModule):
    """Response and cotangent together with independent physical solve evidence.

    ``state_acceptance`` is the accepted-primal evidence and
    ``adjoint_acceptance`` the accepted transpose solve (``None`` for a
    design-only response); ``component`` is the admission under which a bound
    model's parameters receive ``design_cotangent``.
    """

    values: PyTree[Array]
    design_cotangent: PyTree[Array]
    adjoint: PyTree[Array]
    linear_result: Any
    state_acceptance: StateAcceptanceEvidence
    adjoint_acceptance: AdjointAcceptanceEvidence | None
    accepted: Array
    component: StateDesignComponentAdmission | None


def _linearize_state_design(
    problem: StateDesignProblem,
    state: PyTree[Any],
    design: PyTree[Any],
    args: Any,
    linear_policy: LinearSolvePolicy,
    state_acceptance: StateAcceptanceEvidence,
    *,
    state_result: StateEquationResult | None = None,
    operator_id: str | None = None,
    component: StateDesignComponentAdmission | None = None,
) -> StateDesignLinearization:
    def residual_function(current_state: PyTree[Any]) -> PyTree[Array]:
        return problem.residual(current_state, design, args)

    residual, state_action = jax.linearize(residual_function, state)
    _, state_pullback = jax.vjp(residual_function, state)
    state_jacobian = FunctionLinearOperator(
        _StateAction(state_action, residual, False),
        source=PyTreeSpace(state),
        target=PyTreeSpace(residual),
        transpose_action=_StateAction(state_pullback, state, True),
        operator_id=(
            f"{problem.problem_id}/state-jacobian" if operator_id is None else operator_id
        ),
        closure_convert=False,
    )
    _, design_pullback = jax.vjp(
        lambda current_design: problem.residual(state, current_design, args), design
    )
    return StateDesignLinearization(
        problem,
        state,
        design,
        args,
        residual,
        state_jacobian,
        design_pullback,
        state_acceptance,
        linear_policy,
        state_result,
        component,
    )


def prepare_state_design_linearization(
    problem: StateDesignProblem,
    design: PyTree[Any],
    initial_state: PyTree[Any],
    /,
    *,
    args: Any = None,
    linear_policy: LinearSolvePolicy | None = None,
    component: StateDesignComponentAdmission | None = None,
) -> StateDesignLinearization:
    """Solve and independently accept physics, then prepare response pullbacks.

    A rejected state returns ``accepted=False``; consumers must not use response
    derivatives as accepted physics. This operation never runs a design optimizer.
    When the design is the PARAMETER lane of a bound model, ``component`` is the
    owner's ``StateDesignComponentAdmission`` of that model and the design must be
    exactly the admitted lane; the admission then travels with every response.
    """
    if not isinstance(problem, StateDesignProblem):
        raise TypeError("problem must be a StateDesignProblem.")
    policy = _default_adjoint_policy() if linear_policy is None else linear_policy
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear_policy must be LinearSolvePolicy or None.")
    if component is not None and not isinstance(component, StateDesignComponentAdmission):
        raise TypeError("component must be a StateDesignComponentAdmission or None.")
    design_ = _validate_real_inexact_tree(design, name="design")
    if component is not None:
        component.require_design(design_)
    initial = _validate_real_inexact_tree(initial_state, name="initial_state")
    solved = problem.solve_state(design_, initial, args=args)
    return _linearize_state_design(
        problem,
        solved.state,
        design_,
        args,
        policy,
        solved.acceptance,
        state_result=solved,
        component=component,
    )


def _response_pullback(
    linearization: StateDesignLinearization,
    response: _Response | None,
    cotangent: PyTree[Any] | None,
    depends_on_state: bool,
    *,
    prepared_adjoint: PreparedLinearSolve | None = None,
) -> StateDesignResponseVJP:
    point = linearization
    function = (
        (lambda state, design, args: point.problem.value(state, design, args)[0])
        if response is None
        else response
    )
    if not callable(function):
        raise TypeError("response must be callable or None.")
    if depends_on_state:
        values, pullback = jax.vjp(
            lambda state, design: function(state, design, point.args),
            point.state,
            point.design,
        )
    else:
        values, pullback = jax.vjp(
            lambda design: function(point.state, design, point.args), point.design
        )
    values = _validate_real_inexact_tree(values, name="response")
    if cotangent is None:
        if jax.tree.structure(values) != jax.tree.structure(jnp.asarray(0.0)):
            raise ValueError("A nonscalar response requires an explicit cotangent.")
        if values.shape != ():
            raise ValueError("A nonscalar response requires an explicit cotangent.")
        cotangent = jnp.ones_like(values)
    cotangent = _validate_real_inexact_tree(cotangent, name="response cotangent")
    if jax.tree.structure(cotangent) != jax.tree.structure(values):
        raise ValueError("Response and cotangent PyTree structures must match.")
    if any(
        left.shape != right.shape or left.dtype != right.dtype
        for left, right in zip(jax.tree.leaves(cotangent), jax.tree.leaves(values))
    ):
        raise ValueError("Response cotangent shapes and dtypes must match the response.")
    if depends_on_state:
        state_gradient, direct = pullback(cotangent)
        result = (
            solve_linear(
                LinearSystem(transpose(point.state_jacobian)),
                state_gradient,
                policy=point.linear_policy,
            )
            if prepared_adjoint is None
            else solve_linear(prepared_adjoint, state_gradient)
        )
        adjoint = result.value
        acceptance = point.problem.acceptance_policy.adjoint_evidence(
            adjoint,
            point.state_jacobian.transpose_mv(adjoint),
            state_gradient,
            result.status,
            admissible=point.state_acceptance.accepted,
            realization_matches=point.state_acceptance.realization_matches,
        )
        residual_part = point.design_pullback(adjoint)[0]
        gradient = jax.tree.map(lambda left, right: left - right, direct, residual_part)
        derivative_accepted = acceptance.accepted
    else:
        gradient = pullback(cotangent)[0]
        adjoint = jax.tree.map(jnp.zeros_like, point.residual)
        result = None
        acceptance = None
        derivative_accepted = jnp.asarray(True)
    current_realization = (
        jnp.asarray(True)
        if point.problem.state_realization is None
        else jnp.all(
            jnp.asarray(
                point.problem.state_realization(point.state, point.design, point.args),
                dtype=jnp.bool_,
            )
        )
    )
    accepted = (
        point.accepted
        & derivative_accepted
        & current_realization
        & _tree_allfinite(values)
        & _tree_allfinite(cotangent)
        & _tree_allfinite(gradient)
    )
    return StateDesignResponseVJP(
        values,
        gradient,
        adjoint,
        result,
        point.state_acceptance,
        acceptance,
        accepted,
        point.component,
    )


def state_design_response_vjp(
    linearization: StateDesignLinearization,
    /,
    response: _Response | None = None,
    cotangent: PyTree[Any] | None = None,
    *,
    depends_on_state: bool = True,
) -> StateDesignResponseVJP:
    """Pull a response cotangent through accepted fixed-realization physics.

    ``response(state, design, args)`` defaults to the scalar objective. A PyTree
    response requires a matching cotangent. ``depends_on_state=False`` declares a
    design-only response and omits the transpose solve. The returned ``accepted``
    flag, not finite fallback values or backend success alone, permits derivative
    use. This is a first-order response derivative, not an optimizer derivative.
    """
    if not isinstance(linearization, StateDesignLinearization):
        raise TypeError("linearization must be StateDesignLinearization.")
    if not isinstance(depends_on_state, bool):
        raise TypeError("depends_on_state must be a static bool.")
    return _response_pullback(linearization, response, cotangent, depends_on_state)


__all__ = [
    "StateDesignComponentAdmission",
    "StateDesignLinearization",
    "StateDesignResponseVJP",
    "prepare_state_design_linearization",
    "state_design_response_vjp",
]
