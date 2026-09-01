#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import isfinite
from typing import Any, cast, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree

from .._doc import DOC_KEY0
from .._strict import StrictModule
from .._term import AbstractScalarTerm
from ..enforcement import EnforcementProgram
from ..linalg import ArraySpace
from ..nn.parameters import ParameterSubspace
from ..nonlinear import NonlinearSystemProblem
from ._functional_objective import (
    _FunctionalObjective,
    _PreparedObjective,
    evaluate_prepared_objective,
)


FieldEquilibriumFormulation: TypeAlias = Literal[
    "functional-stationarity",
    "virtual-work",
]


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _named_functions(functions: Mapping[str, Any], /) -> Mapping[str, Any]:
    if not isinstance(functions, Mapping):
        raise TypeError("functions must be a mapping of named fields.")
    if not functions or any(not isinstance(name, str) or not name for name in functions):
        raise ValueError("functions must contain non-empty string field names.")
    return functions


def _parameter_subspace(
    functions: Mapping[str, Any],
    subspace: ParameterSubspace,
    /,
) -> ParameterSubspace:
    if not isinstance(subspace, ParameterSubspace):
        raise TypeError("parameter_subspace must be a ParameterSubspace.")
    subspace.validate_root(functions)
    return subspace


def _sign(value: Any, /) -> float:
    sign = float(value)
    if not isfinite(sign) or sign not in (-1.0, 1.0):
        raise ValueError("sign must be exactly +1 or -1.")
    return sign


def _scalar(value: Any, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != ():
        raise ValueError(f"A scalar action must return shape (); got {scalar.shape}.")
    if jnp.iscomplexobj(scalar):
        raise TypeError("A scalar action must return a real value.")
    return scalar.reshape(())


def _matching_tree(left: Any, right: Any, /) -> bool:
    equal = eqx.tree_equal(left, right)
    return equal if isinstance(equal, bool) else bool(jax.device_get(equal))


def _validated_cotangent(cotangent: PyTree[Any], jets: PyTree[Any], /) -> PyTree[Array]:
    jet_leaves, jet_tree = jax.tree_util.tree_flatten(jets)
    cotangent_leaves, cotangent_tree = jax.tree_util.tree_flatten(cotangent)
    if cotangent_tree != jet_tree:
        raise ValueError("virtual_work must return a cotangent matching the field jets.")
    if not jet_leaves:
        raise ValueError("field_jet must return at least one inexact array leaf.")
    validated: list[Array] = []
    for jet, raw_cotangent in zip(jet_leaves, cotangent_leaves, strict=True):
        if not eqx.is_inexact_array(jet):
            raise TypeError("field_jet leaves must be inexact arrays.")
        value = jnp.asarray(raw_cotangent)
        if value.shape != jet.shape:
            raise ValueError(
                "virtual-work cotangent leaves must match the field-jet shapes."
            )
        if jnp.dtype(value.dtype) != jnp.dtype(jet.dtype):
            raise TypeError(
                "virtual-work cotangent leaves must match the field-jet dtypes."
            )
        validated.append(value)
    return jax.tree_util.tree_unflatten(jet_tree, validated)


class _FunctionalActionResidual(StrictModule):
    action: Callable[[Mapping[str, Any], Any, Any], Any]
    subspace: ParameterSubspace
    realization: Any
    sign: float = eqx.field(static=True)
    formulation: FieldEquilibriumFormulation = eqx.field(static=True)

    def __init__(
        self,
        action: Callable[[Mapping[str, Any], Any, Any], Any],
        subspace: ParameterSubspace,
        realization: Any,
        /,
        *,
        sign: float,
    ):
        self.action = action
        self.subspace = subspace
        self.realization = realization
        self.sign = sign
        self.formulation = "functional-stationarity"

    def __call__(self, state: Array, args: Any, /) -> Array:
        def signed_action(position: Array, /) -> Array:
            functions = self.subspace.reconstruct_vector(position)
            return self.sign * _scalar(self.action(functions, self.realization, args))

        return jax.grad(signed_action)(state)

    def rebase(
        self,
        subspace: ParameterSubspace,
        realization: Any,
        /,
    ) -> _FunctionalActionResidual:
        return type(self)(self.action, subspace, realization, sign=self.sign)


class _FunctionalTermsResidual(StrictModule):
    prepared_objective: _PreparedObjective
    subspace: ParameterSubspace
    sign: float = eqx.field(static=True)
    include_model_losses: bool = eqx.field(static=True)
    formulation: FieldEquilibriumFormulation = eqx.field(static=True)

    def __init__(
        self,
        prepared_objective: _PreparedObjective,
        subspace: ParameterSubspace,
        /,
        *,
        sign: float,
        include_model_losses: bool,
    ):
        self.prepared_objective = prepared_objective
        self.subspace = subspace
        self.sign = sign
        self.include_model_losses = bool(include_model_losses)
        self.formulation = "functional-stationarity"

    @property
    def realization(self) -> _PreparedObjective:
        return self.prepared_objective

    def __call__(self, state: Array, args: Any, /) -> Array:
        del args

        def signed_action(position: Array, /) -> Array:
            functions = self.subspace.reconstruct_vector(position)
            value = evaluate_prepared_objective(
                self.prepared_objective,
                functions,
                include_model_losses=self.include_model_losses,
            ).total
            return self.sign * _scalar(value)

        return jax.grad(signed_action)(state)

    def rebase(
        self,
        subspace: ParameterSubspace,
        realization: Any,
        /,
    ) -> _FunctionalTermsResidual:
        if not isinstance(realization, _PreparedObjective):
            raise TypeError(
                "A terms-based field equilibrium requires its prepared objective "
                "realization."
            )
        return type(self)(
            realization,
            subspace,
            sign=self.sign,
            include_model_losses=self.include_model_losses,
        )


class _VirtualWorkResidual(StrictModule):
    field_jet: Callable[[Mapping[str, Any], Any, Any], PyTree[Any]]
    virtual_work: Callable[[Mapping[str, Any], PyTree[Any], Any, Any], PyTree[Any]]
    subspace: ParameterSubspace
    realization: Any
    formulation: FieldEquilibriumFormulation = eqx.field(static=True)

    def __init__(
        self,
        field_jet: Callable[[Mapping[str, Any], Any, Any], PyTree[Any]],
        virtual_work: Callable[[Mapping[str, Any], PyTree[Any], Any, Any], PyTree[Any]],
        subspace: ParameterSubspace,
        realization: Any,
        /,
    ):
        self.field_jet = field_jet
        self.virtual_work = virtual_work
        self.subspace = subspace
        self.realization = realization
        self.formulation = "virtual-work"

    def __call__(self, state: Array, args: Any, /) -> Array:
        def evaluate_jets(position: Array, /) -> PyTree[Any]:
            functions = self.subspace.reconstruct_vector(position)
            return self.field_jet(functions, self.realization, args)

        jets, pullback = jax.vjp(evaluate_jets, state)
        functions = self.subspace.reconstruct_vector(state)
        cotangent = _validated_cotangent(
            self.virtual_work(functions, jets, self.realization, args),
            jets,
        )
        return pullback(cotangent)[0]

    def rebase(
        self,
        subspace: ParameterSubspace,
        realization: Any,
        /,
    ) -> _VirtualWorkResidual:
        return type(self)(
            self.field_jet,
            self.virtual_work,
            subspace,
            realization,
        )


_FieldResidual: TypeAlias = (
    _FunctionalActionResidual | _FunctionalTermsResidual | _VirtualWorkResidual
)


def _field_residual(problem: NonlinearSystemProblem, /) -> _FieldResidual:
    residual = problem.residual_function
    if not isinstance(
        residual,
        (_FunctionalActionResidual, _FunctionalTermsResidual, _VirtualWorkResidual),
    ):
        raise TypeError("PreparedFieldEquilibrium requires a field-equilibrium residual.")
    return residual


class PreparedFieldEquilibrium(StrictModule):
    """Fixed-realization parameter root problem for named trial fields.

    The nonlinear state is the exact packed coordinate vector selected by the
    underlying :class:`ParameterSubspace`. The stored ``problem`` is the sole
    authoritative residual equation.
    """

    problem: NonlinearSystemProblem
    realization_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)
    formulation: FieldEquilibriumFormulation = eqx.field(static=True)

    def __init__(
        self,
        problem: NonlinearSystemProblem,
        /,
        *,
        realization_id: str,
        provenance_id: str,
        formulation: FieldEquilibriumFormulation,
    ):
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be a NonlinearSystemProblem.")
        residual = _field_residual(problem)
        if formulation != residual.formulation:
            raise ValueError("Prepared field-equilibrium formulation does not match.")
        if problem.state_space is None or problem.residual_space is None:
            raise ValueError(
                "Prepared field equilibrium requires explicit vector spaces."
            )
        if (
            problem.state_space.size != residual.subspace.total_dimension
            or not problem.state_space.compatible(problem.residual_space)
        ):
            raise ValueError(
                "Prepared field-equilibrium spaces must match the parameter subspace."
            )
        self.problem = problem
        self.realization_id = _identifier(realization_id, "realization_id")
        self.provenance_id = _identifier(provenance_id, "provenance_id")
        self.formulation = formulation

    @property
    def subspace(self) -> ParameterSubspace:
        return _field_residual(self.problem).subspace

    @property
    def realization(self) -> Any:
        residual = _field_residual(self.problem)
        if isinstance(residual, _FunctionalTermsResidual):
            return residual.prepared_objective
        return residual.realization

    @property
    def initial_state(self) -> Array:
        return self.subspace.pack()

    @property
    def root(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.subspace.reconstruct(self.subspace.initial))

    def pack(self, root: Mapping[str, Any] | None = None, /) -> Array:
        """Pack the prepared root or a compatible rebased root."""
        if root is None:
            return self.initial_state
        functions = _named_functions(root)
        return self.subspace.rebase(functions).pack()

    def reconstruct(self, state: Array, /) -> Mapping[str, Any]:
        """Reconstruct named functions from one packed nonlinear state."""
        return cast(Mapping[str, Any], self.subspace.reconstruct_vector(state))

    def rebase(
        self,
        root: Mapping[str, Any],
        /,
        *,
        realization: Any,
        realization_id: str,
        provenance_id: str,
    ) -> PreparedFieldEquilibrium:
        """Move the same formulation to a compatible root and identical evidence."""
        functions = _named_functions(root)
        realization_identity = _identifier(realization_id, "realization_id")
        provenance_identity = _identifier(provenance_id, "provenance_id")
        if realization_identity != self.realization_id:
            raise ValueError(
                "Field-equilibrium realization identity changed during rebase."
            )
        if provenance_identity != self.provenance_id:
            raise ValueError("Field-equilibrium provenance changed during rebase.")
        if not _matching_tree(realization, self.realization):
            raise ValueError("Field-equilibrium realization changed during rebase.")
        subspace = self.subspace.rebase(functions)
        residual = _field_residual(self.problem).rebase(subspace, realization)
        return _prepare_residual(
            residual,
            realization_id=self.realization_id,
            provenance_id=self.provenance_id,
            problem_id=self.problem.problem_id,
        )


def _prepare_residual(
    residual: _FieldResidual,
    /,
    *,
    realization_id: str,
    provenance_id: str,
    problem_id: str,
) -> PreparedFieldEquilibrium:
    identifier = _identifier(problem_id, "problem_id")
    initial = residual.subspace.pack()
    space = ArraySpace(
        (residual.subspace.total_dimension,),
        dtype=initial.dtype,
        space_id=f"{identifier}:parameter-space",
    )
    problem = NonlinearSystemProblem(
        residual,
        state_space=space,
        residual_space=space,
        problem_id=identifier,
    )
    return PreparedFieldEquilibrium(
        problem,
        realization_id=realization_id,
        provenance_id=provenance_id,
        formulation=residual.formulation,
    )


def prepare_functional_stationarity(
    functions: Mapping[str, Any],
    objective: (
        Callable[[Mapping[str, Any], Any, Any], Any]
        | AbstractScalarTerm
        | Sequence[AbstractScalarTerm]
    ),
    parameter_subspace: ParameterSubspace,
    /,
    *,
    sign: float = 1.0,
    realization: Any = None,
    realization_id: str,
    provenance_id: str,
    problem_id: str = "field-functional-stationarity",
    key: Any = DOC_KEY0,
    iteration: Any = 0,
    enforcement: EnforcementProgram | None = None,
    evaluation_kwargs: Mapping[str, Any] | None = None,
    include_model_losses: bool = True,
) -> PreparedFieldEquilibrium:
    """Prepare a fixed-realization stationarity root for named functions.

    A callable objective receives ``(functions, realization, args)``. Scalar
    terms are prepared once using their native sampling/integration machinery;
    every residual evaluation then differentiates that same prepared objective.
    """

    named = _named_functions(functions)
    subspace = _parameter_subspace(named, parameter_subspace)
    sign_ = _sign(sign)
    if isinstance(objective, AbstractScalarTerm) or not callable(objective):
        terms = (
            (objective,)
            if isinstance(objective, AbstractScalarTerm)
            else tuple(objective)
        )
        if not terms:
            raise ValueError("objective terms must contain at least one scalar term.")
        if any(not isinstance(term, AbstractScalarTerm) for term in terms):
            raise TypeError("objective terms must all be AbstractScalarTerm values.")
        objective_plan = _FunctionalObjective(
            terms=terms,
            enforcement=enforcement,
            collocation_key=key,
        )
        kwargs = {} if evaluation_kwargs is None else dict(evaluation_kwargs)
        if realization is not None:
            if "realization" in kwargs:
                raise ValueError(
                    "realization was supplied both directly and in evaluation_kwargs."
                )
            kwargs["realization"] = realization
        prepared_objective = objective_plan.prepare_training(
            range(len(terms)),
            scale=1.0,
            evaluation_key=key,
            sampling_key=jr.fold_in(key, 1),
            iteration=iteration,
            evaluation_kwargs=kwargs,
        )
        residual: _FieldResidual = _FunctionalTermsResidual(
            prepared_objective,
            subspace,
            sign=sign_,
            include_model_losses=include_model_losses,
        )
    else:
        if enforcement is not None or evaluation_kwargs is not None:
            raise ValueError(
                "enforcement and evaluation_kwargs apply only to scalar terms."
            )
        residual = _FunctionalActionResidual(
            objective,
            subspace,
            realization,
            sign=sign_,
        )
    return _prepare_residual(
        residual,
        realization_id=realization_id,
        provenance_id=provenance_id,
        problem_id=problem_id,
    )


def prepare_virtual_work_equilibrium(
    functions: Mapping[str, Any],
    field_jet: Callable[[Mapping[str, Any], Any, Any], PyTree[Any]],
    virtual_work: Callable[[Mapping[str, Any], PyTree[Any], Any, Any], PyTree[Any]],
    parameter_subspace: ParameterSubspace,
    realization: Any,
    /,
    *,
    realization_id: str,
    provenance_id: str,
    problem_id: str = "field-virtual-work",
) -> PreparedFieldEquilibrium:
    """Prepare ``J_jet(theta)^T q(jet, state) = 0`` on one realization.

    Both the field-jet pullback and the mechanics cotangent remain inside the
    differentiable residual, so nonlinear follower-load and other nonsymmetric
    tangent contributions are retained by JVP and VJP linearizations.
    """

    named = _named_functions(functions)
    subspace = _parameter_subspace(named, parameter_subspace)
    if not callable(field_jet):
        raise TypeError("field_jet must be callable.")
    if not callable(virtual_work):
        raise TypeError("virtual_work must be callable.")
    residual = _VirtualWorkResidual(
        field_jet,
        virtual_work,
        subspace,
        realization,
    )
    return _prepare_residual(
        residual,
        realization_id=realization_id,
        provenance_id=provenance_id,
        problem_id=problem_id,
    )


__all__ = [
    "FieldEquilibriumFormulation",
    "PreparedFieldEquilibrium",
    "prepare_functional_stationarity",
    "prepare_virtual_work_equilibrium",
]
