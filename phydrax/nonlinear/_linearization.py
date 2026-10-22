#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from dis import get_instructions
from inspect import getattr_static
from math import isfinite
from types import CodeType, FunctionType
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import _hidden_arrays
from ..linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    FunctionLinearOperator,
    JacobianLinearOperator,
    LinearizationPolicy,
    prepare_linearization,
    PreparedLinearization,
    PyTreeSpace,
)
from ..linalg._named_blocks import _CoordinateRebasedLinearOperator
from ..sparse import (
    prepare_sparse_linearization,
    PreparedSparseDerivative,
    SparseDerivativePlan,
)
from ..typing import checked, parse
from ._types import _guarded_call, NonlinearSystemProblem


JacobianMode: TypeAlias = Literal[
    "autodiff", "sparse", "directional-finite-difference", "explicit"
]


class JacobianPolicy(StrictModule):
    """Explicit source and numerical controls for nonlinear Jacobian actions."""

    mode: JacobianMode = eqx.field(static=True)
    sparse_plan: SparseDerivativePlan | None
    operator_function: Callable[[PyTree[Any], Any], AbstractLinearOperator] | None
    finite_difference_step: float = eqx.field(static=True)
    linearization_policy: LinearizationPolicy
    operator_transform: (
        Callable[[AbstractLinearOperator], tuple[AbstractLinearOperator, int]] | None
    )
    operator_transform_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        mode: JacobianMode = "autodiff",
        /,
        *,
        sparse_plan: SparseDerivativePlan | None = None,
        operator: Callable[[PyTree[Any], Any], AbstractLinearOperator] | None = None,
        finite_difference_step: float = 1e-6,
        linearization_policy: LinearizationPolicy | None = None,
        operator_transform: Callable[
            [AbstractLinearOperator], tuple[AbstractLinearOperator, int]
        ]
        | None = None,
        operator_transform_id: str | None = None,
    ) -> None:
        mode = parse(mode, JacobianMode, "mode")
        if mode == "sparse" and not isinstance(sparse_plan, SparseDerivativePlan):
            raise TypeError("Sparse Jacobian mode requires a SparseDerivativePlan.")
        if mode != "sparse" and sparse_plan is not None:
            raise ValueError("sparse_plan is only valid for sparse Jacobian mode.")
        if mode == "explicit" and not callable(operator):
            raise TypeError("Explicit Jacobian mode requires an operator callable.")
        if mode != "explicit" and operator is not None:
            raise ValueError("operator is only valid for explicit Jacobian mode.")
        step = float(finite_difference_step)
        if not isfinite(step) or step <= 0.0:
            raise ValueError("finite_difference_step must be finite and positive.")
        linearization = (
            LinearizationPolicy()
            if linearization_policy is None
            else linearization_policy
        )
        if not isinstance(linearization, LinearizationPolicy):
            raise TypeError("linearization_policy must be a LinearizationPolicy or None.")
        if mode != "autodiff" and linearization.rematerialization != "store":
            raise ValueError("Rematerialization requires the autodiff Jacobian source.")
        if operator_transform is None:
            if operator_transform_id is not None:
                raise ValueError("operator_transform_id requires an operator transform.")
        elif not callable(operator_transform):
            raise TypeError("operator_transform must be callable.")
        elif mode != "autodiff":
            raise ValueError(
                "Operator transformation requires the autodiff Jacobian source."
            )
        elif not isinstance(operator_transform_id, str) or not operator_transform_id:
            raise ValueError(
                "An operator transform requires an explicit nonempty identity."
            )
        self.mode = mode
        self.sparse_plan = sparse_plan
        self.operator_function = operator
        self.finite_difference_step = step
        self.linearization_policy = linearization
        self.operator_transform = operator_transform
        self.operator_transform_id = operator_transform_id

    @property
    def policy_id(self) -> str:
        provider = self.operator_function
        provider_type = type(provider)
        code = getattr(provider, "__code__", None)
        payload = {
            "kind": "nonlinear-jacobian-policy",
            "source": self.mode,
            "finite_difference_step": (
                self.finite_difference_step
                if self.mode == "directional-finite-difference"
                else None
            ),
            "sparse_plan": None if self.sparse_plan is None else self.sparse_plan.plan_id,
            "explicit_provider": None
            if provider is None
            else {
                "module": getattr(provider, "__module__", provider_type.__module__),
                "name": getattr(provider, "__qualname__", provider_type.__qualname__),
                "source_line": None if code is None else code.co_firstlineno,
            },
            "rematerialization": self.linearization_policy.rematerialization,
        }
        if self.operator_transform is not None:
            payload["operator_transform"] = self.operator_transform_id
        return canonical_fingerprint(payload)


class PreparedJacobian(StrictModule):
    """Residual value and one reusable Jacobian operator at an accepted state."""

    residual: PyTree
    auxiliary: Any
    operator: AbstractLinearOperator
    sparse_derivative: PreparedSparseDerivative | None
    derivative_id: str = eqx.field(static=True)
    residual_evaluations: int = eqx.field(static=True)
    setup_jvp_evaluations: Array

    @checked
    def __init__(
        self,
        residual: PyTree,
        operator: AbstractLinearOperator,
        /,
        *,
        auxiliary: Any = None,
        sparse_derivative: PreparedSparseDerivative | None = None,
        derivative_id: str,
        residual_evaluations: int = 1,
        setup_jvp_evaluations: int = 0,
    ) -> None:
        if sparse_derivative is not None and not isinstance(
            sparse_derivative, PreparedSparseDerivative
        ):
            raise TypeError("sparse_derivative must be PreparedSparseDerivative or None.")
        identifier = str(derivative_id)
        if not identifier:
            raise ValueError("derivative_id must be non-empty.")
        if not isinstance(setup_jvp_evaluations, int) or isinstance(
            setup_jvp_evaluations, bool
        ):
            raise TypeError("setup_jvp_evaluations must be an exact integer.")
        if setup_jvp_evaluations < 0:
            raise ValueError("setup_jvp_evaluations must be nonnegative.")
        self.residual = residual
        evaluations = int(residual_evaluations)
        if evaluations < 1:
            raise ValueError("residual_evaluations must be positive.")
        self.operator = operator
        self.sparse_derivative = sparse_derivative
        self.auxiliary = auxiliary
        self.derivative_id = identifier
        self.residual_evaluations = evaluations
        self.setup_jvp_evaluations = jnp.asarray(setup_jvp_evaluations, dtype=jnp.int32)


def _jacobian_jvp_actions(operator: AbstractLinearOperator, actions: Array, /) -> Array:
    """Count global JVPs, not arithmetic actions on an assembled Jacobian."""
    if isinstance(operator, _CoordinateRebasedLinearOperator):
        return _jacobian_jvp_actions(operator.operator, actions)
    return (
        actions
        if isinstance(operator, JacobianLinearOperator)
        else jnp.zeros_like(actions)
    )


def _rebase_jacobian_coordinates(
    operator: AbstractLinearOperator,
    /,
) -> _CoordinateRebasedLinearOperator:
    return _CoordinateRebasedLinearOperator(operator)


def _jacobian_solve_operator(
    operator: AbstractLinearOperator,
    /,
) -> AbstractLinearOperator:
    if operator.source.compatible(operator.target):
        return operator
    return _rebase_jacobian_coordinates(operator)


def _jacobian_solve_right_hand_side(
    operator: AbstractLinearOperator,
    residual: PyTree[Any],
    /,
) -> PyTree[Array]:
    if isinstance(operator, _CoordinateRebasedLinearOperator):
        return operator.flatten_target(jax.tree.map(jnp.negative, residual))
    return operator.target.validate(jax.tree.map(jnp.negative, residual))


def _jacobian_solve_direction(
    operator: AbstractLinearOperator,
    value: PyTree[Any],
    /,
) -> PyTree[Array]:
    if isinstance(operator, _CoordinateRebasedLinearOperator):
        return operator.unflatten_source(value)
    return operator.source.validate(value)


def _prepare_autodiff_linearization(
    problem: NonlinearSystemProblem,
    state: PyTree[Any],
    source: AbstractVectorSpace,
    policy: LinearizationPolicy,
    args: Any,
    callback_code: CodeType | None = None,
    /,
) -> PreparedLinearization:
    # Code is static compilation identity, not a snapshot of callback state.
    del callback_code
    # Trace callback arrays as inputs in both cold preparation and refreshes.
    # Concrete closure captures can deduplicate differently from traced inputs,
    # changing the retained tangent tape's PyTree structure.
    function = (
        (lambda candidate: problem.evaluate(candidate, args))
        if problem.has_aux
        else (lambda candidate: problem.residual(candidate, args))
    )
    return prepare_linearization(
        function,
        state,
        source=source,
        has_aux=problem.has_aux,
        target=problem.residual_space,
        policy=policy,
        linearization_id=f"{problem.problem_id}/jacobian/autodiff",
    )


_compiled_autodiff_linearization = eqx.filter_jit(_prepare_autodiff_linearization)


def _keyed_autodiff_state(value: Any, /, *, static: bool = False) -> bool:
    """Prove that state consists only of keyed records and numerical leaves."""
    if value is None or type(value) in (str, bytes, bool, int, float, complex):
        return True
    if eqx.is_array(value):
        return not static
    if type(value) is tuple:
        return all(_keyed_autodiff_state(item, static=static) for item in value)
    if static:
        # Mutable objects in treedef metadata do not get value-based cache keys.
        return False
    if type(value) in (list, dict):
        items = value if type(value) is list else (*value.keys(), *value.values())
        return all(_keyed_autodiff_state(item) for item in items)
    if not isinstance(value, StrictModule):
        # Includes functions, bound methods, partials, and opaque args.
        return False
    if type(value).__getattribute__ is not StrictModule.__getattribute__:
        return False
    attributes = vars(value)
    for field in fields(value):
        if field.name not in attributes:
            return False
        item = attributes[field.name]
        if getattr_static(value, field.name) is not item:
            # A descriptor can read state absent from the declared field.
            return False
        if not _keyed_autodiff_state(item, static=field.metadata.get("static", False)):
            return False
    return True


def _keyed_autodiff_call(callback: Any, state: PyTree[Any], args: Any, /) -> bool:
    """Prove a straight-line numerical callback has no external execution state.

    This deliberately admits only field reads and numerical arithmetic. Calls,
    global/closure reads, descriptors, control flow, and unknown bytecode remain
    request-local rather than assuming purity of their transitive dependencies.
    """
    if not isinstance(callback, StrictModule):
        return False
    function = getattr_static(type(callback), "__call__", None)
    if not isinstance(function, FunctionType):
        return False
    code = function.__code__
    if (
        function.__closure__
        or function.__defaults__
        or function.__kwdefaults__
        or code.co_argcount != 3
        or code.co_kwonlyargcount
        or code.co_flags & 0x0C  # Variadic positional or keyword arguments.
    ):
        return False
    # The marker denotes an array-valued expression, never retained runtime data.
    numeric = object()
    local = dict(zip(code.co_varnames[:3], (callback, state, args), strict=True))
    stack: list[Any] = []

    def is_numeric(value: Any) -> bool:
        return (
            value is numeric
            or eqx.is_array(value)
            or type(value) in (bool, int, float, complex)
        )

    for instruction in get_instructions(function):
        match instruction.opname:
            case "RESUME" | "NOP":
                continue
            case "LOAD_FAST":
                if instruction.argval not in local:
                    return False
                stack.append(local[instruction.argval])
            case "STORE_FAST":
                local[instruction.argval] = stack.pop()
            case "LOAD_CONST":
                if not is_numeric(instruction.argval):
                    return False
                stack.append(instruction.argval)
            case "LOAD_ATTR":
                owner = stack.pop()
                if not isinstance(owner, StrictModule):
                    return False
                if instruction.argval not in {field.name for field in fields(owner)}:
                    return False
                stack.append(vars(owner)[instruction.argval])
            case "BINARY_OP":
                right, left = stack.pop(), stack.pop()
                if not (is_numeric(left) and is_numeric(right)):
                    return False
                stack.append(numeric)
            case "UNARY_NEGATIVE" | "UNARY_POSITIVE" | "UNARY_INVERT":
                if not is_numeric(stack.pop()):
                    return False
                stack.append(numeric)
            case "RETURN_VALUE":
                return len(stack) == 1 and is_numeric(stack[0])
            case _:
                return False
    return False


def _can_share_autodiff_preparation(
    problem: NonlinearSystemProblem, state: PyTree[Any], args: Any, /
) -> bool:
    """Return whether callback execution and args are proven keyed runtime state."""
    return (
        # Domain callbacks execute in the same tape but are not covered by the
        # residual callback's execution-state proof or static code key.
        problem.trial_validity_function is None
        and _keyed_autodiff_state(problem)
        and _keyed_autodiff_state(args)
        and _keyed_autodiff_state(state)
        and _keyed_autodiff_call(problem.residual_function, state, args)
        and not _hidden_arrays((problem, state, args), eqx.is_array)
    )


def _prepare_autodiff_tape(
    problem: NonlinearSystemProblem,
    state: PyTree[Any],
    source: AbstractVectorSpace,
    policy: LinearizationPolicy,
    args: Any,
    /,
) -> PreparedLinearization:
    """Choose request-local or shared compilation for the retained autodiff tape."""
    if not _can_share_autodiff_preparation(problem, state, args):
        # A fresh identity rereads unkeyed state on each host preparation request.
        @eqx.filter_jit
        def prepare_current(
            current_problem: NonlinearSystemProblem,
            current_state: PyTree[Any],
            current_source: AbstractVectorSpace,
            current_policy: LinearizationPolicy,
            current_args: Any,
        ) -> PreparedLinearization:
            return _prepare_autodiff_linearization(
                current_problem,
                current_state,
                current_source,
                current_policy,
                current_args,
            )

        linearization = prepare_current(problem, state, source, policy, args)
    else:
        linearization = _compiled_autodiff_linearization(
            problem,
            state,
            source,
            policy,
            args,
            getattr_static(type(problem.residual_function), "__call__").__code__,
        )
    return linearization


def prepare_jacobian(
    problem: NonlinearSystemProblem,
    state: PyTree[Any],
    policy: JacobianPolicy,
    args: Any = None,
    /,
) -> PreparedJacobian:
    """Prepare one Jacobian action without changing derivative source.

    Cold and refreshed autodiff tapes use compiled preparation. Unproven callback
    execution or args use a request-local boundary; proven numerical PyTrees retain
    shared compilation. Sparse topology and provider operators retain host preparation.
    """
    if not isinstance(problem, NonlinearSystemProblem):
        raise TypeError("problem must be a NonlinearSystemProblem.")
    if not isinstance(policy, JacobianPolicy):
        raise TypeError("policy must be a JacobianPolicy.")
    source = PyTreeSpace(state) if problem.state_space is None else problem.state_space

    if policy.mode == "autodiff":
        linearization = _prepare_autodiff_tape(
            problem, state, source, policy.linearization_policy, args
        )
        operator: AbstractLinearOperator = JacobianLinearOperator(
            linearization,
            operator_id=f"{problem.problem_id}/jacobian",
        )
        setup_work = 0
        if policy.operator_transform is not None:
            transformed, setup_work = policy.operator_transform(operator)
            if not isinstance(transformed, AbstractLinearOperator):
                raise TypeError(
                    "An operator transform must return a native linear operator."
                )
            if not transformed.source.compatible(
                operator.source
            ) or not transformed.target.compatible(operator.target):
                raise ValueError(
                    "An operator transform must preserve source and target identities."
                )
            operator = transformed
        return PreparedJacobian(
            linearization.primal,
            operator,
            auxiliary=linearization.auxiliary,
            derivative_id="autodiff",
            setup_jvp_evaluations=setup_work,
        )

    if policy.mode == "sparse":
        sparse_plan = policy.sparse_plan
        if not (sparse_plan is not None):
            raise RuntimeError("Internal invariant failed: sparse_plan is not None.")
        sparse = (
            prepare_sparse_linearization(sparse_plan, state, args)
            if problem.trial_validity_function is None
            else _guarded_call(
                problem.trial_valid(state, args),
                lambda candidate, arguments: prepare_sparse_linearization(
                    sparse_plan, candidate, arguments
                ),
                state,
                args,
            )
        )
        if not source.compatible(sparse.operator.source):
            raise ValueError(
                "Sparse Jacobian source must match the nonlinear state space."
            )
        if problem.residual_space is not None and not problem.residual_space.compatible(
            sparse.operator.target
        ):
            raise ValueError(
                "Sparse Jacobian target must match the nonlinear residual space."
            )
        if problem.has_aux:
            _, auxiliary = problem.evaluate(state, args)
        else:
            auxiliary = None
        return PreparedJacobian(
            sparse.linearization.primal,
            sparse.operator,
            auxiliary=auxiliary,
            sparse_derivative=sparse,
            derivative_id=f"sparse:{sparse_plan.plan_id}",
            residual_evaluations=1 + int(problem.has_aux),
        )

    residual, auxiliary = problem.evaluate(state, args)
    target = (
        PyTreeSpace(residual)
        if problem.residual_space is None
        else problem.residual_space
    )
    residual = target.validate(residual)
    if policy.mode == "explicit":
        assert policy.operator_function is not None
        operator = (
            policy.operator_function(state, args)
            if problem.trial_validity_function is None
            else _guarded_call(
                problem.trial_valid(state, args), policy.operator_function, state, args
            )
        )
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError(
                "The explicit Jacobian callable must return a linear operator."
            )
        if not source.compatible(operator.source) or not target.compatible(
            operator.target
        ):
            raise ValueError(
                "Explicit Jacobian source and target spaces must match the residual."
            )
        return PreparedJacobian(
            residual,
            operator,
            auxiliary=auxiliary,
            derivative_id="explicit",
        )

    step = jnp.asarray(
        policy.finite_difference_step, dtype=source.flatten(state).real.dtype
    )

    def action(tangent: PyTree[Array]) -> PyTree[Array]:
        candidate = jax.tree.map(
            lambda value, delta: value + step * delta, state, tangent
        )
        candidate = eqx.error_if(
            candidate,
            ~problem.trial_valid(candidate, args),
            "Directional finite-difference Jacobian leaves the residual domain; use autodiff.",
        )
        shifted = problem.residual(candidate, args)
        return jax.tree.map(lambda new, old: (new - old) / step, shifted, residual)

    operator = FunctionLinearOperator(
        action,
        source=source,
        target=target,
        operator_id=f"{problem.problem_id}/jacobian/directional-finite-difference",
    )
    return PreparedJacobian(
        residual,
        operator,
        auxiliary=auxiliary,
        derivative_id="directional-finite-difference",
    )


__all__ = ["JacobianMode", "JacobianPolicy", "PreparedJacobian", "prepare_jacobian"]
