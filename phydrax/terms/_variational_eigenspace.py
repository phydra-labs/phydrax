#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DomainFunction

from .._doc import DOC_KEY0
from .._strict import StrictModule
from .._term import AbstractSamplingTerm
from ..integration import (
    IntegrationRealization,
    IntegrationStatus,
    materialize,
    reduce,
)
from ..integration._api import _requires_random_key
from ..linalg import HermitianSpectrum
from ..linalg.eigen import (
    block_rayleigh_trace,
    BlockRayleighEvaluation,
    ReducedRitzResult,
    solve_reduced_ritz,
)
from ..operators import conjugate


FormDensity = Callable[[DomainFunction, DomainFunction], DomainFunction]
EigenspaceAction: TypeAlias = Callable[[DomainFunction], DomainFunction]
MaterializationPolicy: TypeAlias = Literal["fixed", "per_step", "caller"]


def _default_mass(left: DomainFunction, right: DomainFunction, /) -> DomainFunction:
    return conjugate(left) * right


def _identity_action(field: DomainFunction, /) -> DomainFunction:
    return field


def _objective_variables(value: Sequence[str], /) -> tuple[str, ...]:
    variables = tuple(str(name) for name in value)
    if not variables or any(not name for name in variables):
        raise ValueError("objective_vars must contain non-empty field names.")
    if len(set(variables)) != len(variables):
        raise ValueError("objective_vars must not contain duplicates.")
    return variables


def _tolerance(value: float, /) -> float:
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")
    return tolerance


def _weight(value: ArrayLike, /, *, nonnegative: bool) -> float:
    weight = float(value)
    if not math.isfinite(weight):
        raise ValueError("weight must be finite.")
    if nonnegative and weight < 0.0:
        raise ValueError("A residual weight must be non-negative.")
    return weight


def _prepare_materialization(
    target: Any,
    plan: Any,
    materialization_policy: MaterializationPolicy,
    fixed_realization: IntegrationRealization | None,
    fixed_key: Key[Array, ""] | None,
    /,
    *,
    role: str,
) -> tuple[MaterializationPolicy, IntegrationRealization | None]:
    policy = str(materialization_policy).lower()
    if policy not in ("fixed", "per_step", "caller"):
        raise ValueError(
            "materialization_policy must be 'fixed', 'per_step', or 'caller'."
        )
    if policy == "fixed":
        if fixed_realization is None:
            if _requires_random_key(plan):
                if fixed_key is None:
                    raise ValueError(f"A randomized fixed {role} requires fixed_key=.")
                fixed_realization = materialize(target, plan, key=fixed_key)
            else:
                if fixed_key is not None:
                    raise ValueError(
                        f"A deterministic fixed {role} does not consume fixed_key=."
                    )
                fixed_realization = materialize(target, plan)
    elif fixed_realization is not None or fixed_key is not None:
        raise ValueError(
            "fixed_realization/fixed_key require materialization_policy='fixed'."
        )
    return policy, fixed_realization


def _sample_realization(
    target: Any,
    plan: Any,
    materialization_policy: MaterializationPolicy,
    fixed_realization: IntegrationRealization | None,
    key: Key[Array, ""],
    /,
    *,
    role: str,
) -> IntegrationRealization | None:
    if materialization_policy == "fixed":
        if fixed_realization is None:
            raise RuntimeError(f"Fixed {role} has no realization.")
        return fixed_realization
    if materialization_policy == "caller":
        return None
    if _requires_random_key(plan):
        return materialize(target, plan, key=key)
    return materialize(target, plan)


def _resolve_realization(
    target: Any,
    plan: Any,
    materialization_policy: MaterializationPolicy,
    fixed_realization: IntegrationRealization | None,
    batch: IntegrationRealization | None,
    key: Key[Array, ""],
    /,
    *,
    role: str,
) -> IntegrationRealization:
    realization = batch
    if realization is None:
        if materialization_policy == "caller":
            raise ValueError(
                f"Caller-managed {role} requires batch=IntegrationRealization."
            )
        realization = _sample_realization(
            target,
            plan,
            materialization_policy,
            fixed_realization,
            key,
            role=role,
        )
    if not isinstance(realization, IntegrationRealization):
        raise TypeError(f"{role} batch must be an IntegrationRealization.")
    return realization


def _trial_fields(
    objective_vars: tuple[str, ...],
    functions: Mapping[str, DomainFunction],
    /,
    *,
    role: str,
) -> tuple[DomainFunction, ...]:
    fields: list[DomainFunction] = []
    for name in objective_vars:
        if name not in functions:
            raise KeyError(f"Missing {role} field {name!r}.")
        field = functions[name]
        if not isinstance(field, DomainFunction):
            raise TypeError(f"{role} field {name!r} must be a DomainFunction.")
        fields.append(field)
    return tuple(fields)


def _action_fields(
    action: EigenspaceAction,
    fields: tuple[DomainFunction, ...],
    /,
    *,
    role: str,
) -> tuple[DomainFunction, ...]:
    outputs: list[DomainFunction] = []
    for field in fields:
        output = action(field)
        if not isinstance(output, DomainFunction):
            raise TypeError(f"{role} action must return a DomainFunction.")
        outputs.append(output)
    return tuple(outputs)


def _integral(
    density: DomainFunction,
    realization: IntegrationRealization,
    /,
    *,
    role: str,
    **kwargs: Any,
) -> Array:
    if not isinstance(density, DomainFunction):
        raise TypeError(f"{role} forms must return DomainFunction.")
    estimate = reduce(density, realization, **kwargs)
    if estimate.value.dims != ():
        raise ValueError(
            f"{role} forms must reduce to scalar fields; got dims={estimate.value.dims}."
        )
    value = jnp.asarray(estimate.value.data).reshape(())
    return eqx.error_if(
        value,
        estimate.status != int(IntegrationStatus.CONVERGED),
        f"{role} integration did not converge.",
    )


def _form_matrix(
    form: FormDensity,
    left_fields: tuple[DomainFunction, ...],
    right_fields: tuple[DomainFunction, ...],
    realization: IntegrationRealization,
    /,
    *,
    role: str,
    **kwargs: Any,
) -> Array:
    rows = []
    for left in left_fields:
        rows.append(
            jnp.stack(
                tuple(
                    _integral(
                        form(left, right),
                        realization,
                        role=role,
                        **kwargs,
                    )
                    for right in right_fields
                )
            )
        )
    return jnp.stack(tuple(rows))


def _linear_combinations(
    fields: tuple[DomainFunction, ...],
    coefficients: Array,
    /,
) -> tuple[DomainFunction, ...]:
    combinations: list[DomainFunction] = []
    for column in range(int(coefficients.shape[1])):
        combined = coefficients[0, column] * fields[0]
        for row in range(1, len(fields)):
            combined = combined + coefficients[row, column] * fields[row]
        combinations.append(combined)
    return tuple(combinations)


class VariationalEigenspaceEvaluation(StrictModule):
    """One shared-realization stiffness/mass evaluation and block quotient."""

    block: BlockRayleighEvaluation
    provenance: str = eqx.field(static=True)

    @property
    def stiffness(self) -> Array:
        return self.block.stiffness

    @property
    def mass(self) -> Array:
        return self.block.mass

    @property
    def objective(self) -> Array:
        return self.block.objective

    @property
    def valid(self) -> Array:
        return self.block.valid


class VariationalEigenspaceResult(StrictModule):
    """Continuous Ritz modes reconstructed from one variational trial space."""

    evaluation: VariationalEigenspaceEvaluation
    reduced: ReducedRitzResult
    modes: tuple[DomainFunction, ...]
    objective_vars: tuple[str, ...] = eqx.field(static=True)

    @property
    def eigenvalues(self) -> Array:
        return self.reduced.eigenvalues

    @property
    def coefficients(self) -> Array:
        return self.reduced.coefficients

    @property
    def successful(self) -> Array:
        return self.evaluation.valid & self.reduced.successful


class InvariantSubspaceResidualEvaluation(StrictModule):
    """Projected operator and basis-invariant strong-residual evidence."""

    projection: BlockRayleighEvaluation
    residual: BlockRayleighEvaluation
    residual_gram_minimum_eigenvalue: Array
    residual_gram_numerical_rank: Array
    residual_gram_positive_semidefinite: Array
    provenance: str = eqx.field(static=True)

    @property
    def stiffness(self) -> Array:
        return self.projection.stiffness

    @property
    def mass(self) -> Array:
        return self.projection.mass

    @property
    def reduced_operator(self) -> Array:
        return self.projection.solved_stiffness

    @property
    def residual_gram(self) -> Array:
        return self.residual.stiffness

    @property
    def objective(self) -> Array:
        return self.residual.objective

    @property
    def valid(self) -> Array:
        return (
            self.projection.valid
            & self.residual.valid
            & self.residual_gram_positive_semidefinite
        )


class InvariantSubspaceResidualResult(StrictModule):
    """Continuous Ritz modes with absolute and relative strong residuals."""

    evaluation: InvariantSubspaceResidualEvaluation
    reduced: ReducedRitzResult
    modes: tuple[DomainFunction, ...]
    residual_modes: tuple[DomainFunction, ...]
    residual_norms: Array
    relative_residuals: Array
    objective_vars: tuple[str, ...] = eqx.field(static=True)

    @property
    def eigenvalues(self) -> Array:
        return self.reduced.eigenvalues

    @property
    def coefficients(self) -> Array:
        return self.reduced.coefficients

    @property
    def successful(self) -> Array:
        return (
            self.evaluation.valid
            & self.reduced.successful
            & jnp.all(jnp.isfinite(self.residual_norms))
            & jnp.all(jnp.isfinite(self.relative_residuals))
        )

    @property
    def eigenvalue(self) -> Array:
        if len(self.objective_vars) != 1:
            raise ValueError("eigenvalue is available only for one trial field.")
        return self.eigenvalues[0]

    @property
    def mode(self) -> DomainFunction:
        if len(self.objective_vars) != 1:
            raise ValueError("mode is available only for one trial field.")
        return self.modes[0]


class _InvariantSubspaceAssembly(StrictModule):
    evaluation: InvariantSubspaceResidualEvaluation
    fields: tuple[DomainFunction, ...]
    operator_fields: tuple[DomainFunction, ...]
    metric_fields: tuple[DomainFunction, ...]
    residual_fields: tuple[DomainFunction, ...]
    realization: IntegrationRealization


class VariationalEigenspace(AbstractSamplingTerm):
    """Hermitian block Rayleigh objective over named continuous trial fields."""

    objective_vars: tuple[str, ...]
    target: Any
    plan: Any
    stiffness_form: FormDensity
    mass_form: FormDensity
    fixed_realization: IntegrationRealization | None
    weight: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    label: str | None = eqx.field(static=True)
    materialization_policy: MaterializationPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        target: Any,
        stiffness_form: FormDensity,
        objective_vars: Sequence[str],
        mass_form: FormDensity | None = None,
        plan: Any = None,
        weight: ArrayLike = 1.0,
        tolerance: float = 1e-10,
        label: str | None = None,
        materialization_policy: MaterializationPolicy = "fixed",
        fixed_realization: IntegrationRealization | None = None,
        fixed_key: Key[Array, ""] | None = None,
    ):
        if not callable(stiffness_form):
            raise TypeError("stiffness_form must be callable.")
        if mass_form is not None and not callable(mass_form):
            raise TypeError("mass_form must be callable or None.")
        variables = _objective_variables(objective_vars)
        tolerance_ = _tolerance(tolerance)
        weight_ = _weight(weight, nonnegative=False)
        policy, fixed_realization_ = _prepare_materialization(
            target,
            plan,
            materialization_policy,
            fixed_realization,
            fixed_key,
            role="VariationalEigenspace",
        )
        self.objective_vars = variables
        self.target = target
        self.plan = plan
        self.stiffness_form = stiffness_form
        self.mass_form = _default_mass if mass_form is None else mass_form
        self.fixed_realization = fixed_realization_
        self.weight = weight_
        self.tolerance = tolerance_
        self.label = None if label is None else str(label)
        self.materialization_policy = policy

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> IntegrationRealization | None:
        """Materialize one integration realization under the declared policy."""
        return _sample_realization(
            self.target,
            self.plan,
            self.materialization_policy,
            self.fixed_realization,
            key,
            role="VariationalEigenspace",
        )

    def assemble(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: IntegrationRealization | None = None,
        **kwargs: Any,
    ) -> VariationalEigenspaceEvaluation:
        """Assemble both form matrices on one shared realization."""
        realization = _resolve_realization(
            self.target,
            self.plan,
            self.materialization_policy,
            self.fixed_realization,
            batch,
            key,
            role="VariationalEigenspace",
        )
        fields = _trial_fields(
            self.objective_vars,
            functions,
            role="variational eigenspace",
        )
        stiffness = _form_matrix(
            self.stiffness_form,
            fields,
            fields,
            realization,
            role="Variational eigenspace",
            **kwargs,
        )
        mass = _form_matrix(
            self.mass_form,
            fields,
            fields,
            realization,
            role="Variational eigenspace",
            **kwargs,
        )
        block = block_rayleigh_trace(stiffness, mass, tolerance=self.tolerance)
        return VariationalEigenspaceEvaluation(
            block=block,
            provenance=type(realization.plan).__name__,
        )

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: IntegrationRealization | None = None,
        **kwargs: Any,
    ) -> Array:
        """Return the basis-invariant sum of generalized Ritz values."""
        if self.weight == 0.0:
            return jnp.zeros((), dtype=float)
        evaluation = self.assemble(
            functions,
            key=key,
            batch=batch,
            **kwargs,
        )
        objective = eqx.error_if(
            evaluation.objective,
            ~evaluation.valid,
            "Variational eigenspace Gram or Ritz evaluation is invalid.",
        )
        return self.weight * objective

    def ritz(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        count: int | None = None,
        which: str = "smallest-algebraic",
        key: Key[Array, ""] = DOC_KEY0,
        batch: IntegrationRealization | None = None,
        **kwargs: Any,
    ) -> VariationalEigenspaceResult:
        """Extract continuous Ritz modes from the current trial fields."""
        fields = _trial_fields(
            self.objective_vars,
            functions,
            role="variational eigenspace",
        )
        evaluation = self.assemble(
            functions,
            key=key,
            batch=batch,
            **kwargs,
        )
        reduced = solve_reduced_ritz(
            evaluation.stiffness,
            evaluation.mass,
            count=count,
            which=which,
            tolerance=self.tolerance,
        )
        coefficients = reduced.coefficients
        modes = _linear_combinations(fields, coefficients)
        return VariationalEigenspaceResult(
            evaluation=evaluation,
            reduced=reduced,
            modes=modes,
            objective_vars=self.objective_vars,
        )


class InvariantSubspaceResidual(AbstractSamplingTerm):
    """Basis-invariant strong residual for a self-adjoint trial eigenspace."""

    objective_vars: tuple[str, ...]
    target: Any
    plan: Any
    operator_action: EigenspaceAction
    metric_action: EigenspaceAction
    pairing: FormDensity
    residual_pairing: FormDensity
    fixed_realization: IntegrationRealization | None
    weight: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    label: str | None = eqx.field(static=True)
    materialization_policy: MaterializationPolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        target: Any,
        operator_action: EigenspaceAction,
        objective_vars: Sequence[str],
        metric_action: EigenspaceAction | None = None,
        pairing: FormDensity | None = None,
        residual_pairing: FormDensity | None = None,
        plan: Any = None,
        weight: ArrayLike = 1.0,
        tolerance: float = 1e-10,
        label: str | None = None,
        materialization_policy: MaterializationPolicy = "fixed",
        fixed_realization: IntegrationRealization | None = None,
        fixed_key: Key[Array, ""] | None = None,
    ):
        if not callable(operator_action):
            raise TypeError("operator_action must be callable.")
        if metric_action is not None and not callable(metric_action):
            raise TypeError("metric_action must be callable or None.")
        if pairing is not None and not callable(pairing):
            raise TypeError("pairing must be callable or None.")
        if residual_pairing is not None and not callable(residual_pairing):
            raise TypeError("residual_pairing must be callable or None.")
        variables = _objective_variables(objective_vars)
        tolerance_ = _tolerance(tolerance)
        weight_ = _weight(weight, nonnegative=True)
        policy, fixed_realization_ = _prepare_materialization(
            target,
            plan,
            materialization_policy,
            fixed_realization,
            fixed_key,
            role="InvariantSubspaceResidual",
        )
        pairing_ = _default_mass if pairing is None else pairing
        self.objective_vars = variables
        self.target = target
        self.plan = plan
        self.operator_action = operator_action
        self.metric_action = _identity_action if metric_action is None else metric_action
        self.pairing = pairing_
        self.residual_pairing = pairing_ if residual_pairing is None else residual_pairing
        self.fixed_realization = fixed_realization_
        self.weight = weight_
        self.tolerance = tolerance_
        self.label = None if label is None else str(label)
        self.materialization_policy = policy

    def sample(
        self,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> IntegrationRealization | None:
        """Materialize one integration realization under the declared policy."""
        return _sample_realization(
            self.target,
            self.plan,
            self.materialization_policy,
            self.fixed_realization,
            key,
            role="InvariantSubspaceResidual",
        )

    def _assemble(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""],
        batch: IntegrationRealization | None,
        **kwargs: Any,
    ) -> _InvariantSubspaceAssembly:
        realization = _resolve_realization(
            self.target,
            self.plan,
            self.materialization_policy,
            self.fixed_realization,
            batch,
            key,
            role="InvariantSubspaceResidual",
        )
        fields = _trial_fields(
            self.objective_vars,
            functions,
            role="invariant-subspace residual",
        )
        operator_fields = _action_fields(
            self.operator_action,
            fields,
            role="InvariantSubspaceResidual operator",
        )
        metric_fields = _action_fields(
            self.metric_action,
            fields,
            role="InvariantSubspaceResidual metric",
        )
        stiffness = _form_matrix(
            self.pairing,
            fields,
            operator_fields,
            realization,
            role="Invariant-subspace projection",
            **kwargs,
        )
        mass = _form_matrix(
            self.pairing,
            fields,
            metric_fields,
            realization,
            role="Invariant-subspace projection",
            **kwargs,
        )
        projection = block_rayleigh_trace(
            stiffness,
            mass,
            tolerance=self.tolerance,
        )
        reduced_operator = eqx.error_if(
            projection.solved_stiffness,
            ~projection.valid,
            "Invariant-subspace projection is invalid.",
        )
        residual_fields: list[DomainFunction] = []
        for column, operator_field in enumerate(operator_fields):
            residual = operator_field
            for row, metric_field in enumerate(metric_fields):
                residual = residual - reduced_operator[row, column] * metric_field
            residual_fields.append(residual)
        residual_fields_ = tuple(residual_fields)
        residual_gram = _form_matrix(
            self.residual_pairing,
            residual_fields_,
            residual_fields_,
            realization,
            role="Invariant-subspace residual",
            **kwargs,
        )
        residual = block_rayleigh_trace(
            residual_gram,
            projection.mass,
            tolerance=self.tolerance,
        )
        residual_spectrum = HermitianSpectrum(
            jax.lax.stop_gradient(residual_gram),
            tolerance=self.tolerance,
        )
        residual_scale = jnp.maximum(jnp.max(jnp.abs(residual_gram)), 1.0)
        residual_psd = residual_spectrum.valid & (
            residual_spectrum.minimum_eigenvalue >= -self.tolerance * residual_scale
        )
        evaluation = InvariantSubspaceResidualEvaluation(
            projection=projection,
            residual=residual,
            residual_gram_minimum_eigenvalue=(residual_spectrum.minimum_eigenvalue),
            residual_gram_numerical_rank=residual_spectrum.numerical_rank,
            residual_gram_positive_semidefinite=residual_psd,
            provenance=type(realization.plan).__name__,
        )
        return _InvariantSubspaceAssembly(
            evaluation=evaluation,
            fields=fields,
            operator_fields=operator_fields,
            metric_fields=metric_fields,
            residual_fields=residual_fields_,
            realization=realization,
        )

    def assemble(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: IntegrationRealization | None = None,
        **kwargs: Any,
    ) -> InvariantSubspaceResidualEvaluation:
        """Assemble the projected operator and strong residual on one realization."""
        return self._assemble(
            functions,
            key=key,
            batch=batch,
            **kwargs,
        ).evaluation

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: IntegrationRealization | None = None,
        **kwargs: Any,
    ) -> Array:
        """Return the nonnegative basis-invariant strong residual."""
        if self.weight == 0.0:
            return jnp.zeros((), dtype=float)
        evaluation = self.assemble(
            functions,
            key=key,
            batch=batch,
            **kwargs,
        )
        objective = eqx.error_if(
            evaluation.objective,
            ~evaluation.valid,
            "Invariant-subspace residual evaluation is invalid.",
        )
        return self.weight * jnp.maximum(objective, 0.0)

    def ritz(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        count: int | None = None,
        which: str = "smallest-algebraic",
        key: Key[Array, ""] = DOC_KEY0,
        batch: IntegrationRealization | None = None,
        **kwargs: Any,
    ) -> InvariantSubspaceResidualResult:
        """Extract Ritz modes and transformed strong-residual diagnostics."""
        assembly = self._assemble(
            functions,
            key=key,
            batch=batch,
            **kwargs,
        )
        evaluation = assembly.evaluation
        reduced = solve_reduced_ritz(
            evaluation.stiffness,
            evaluation.mass,
            count=count,
            which=which,
            tolerance=self.tolerance,
        )
        coefficients = reduced.coefficients
        adjoint_coefficients = jnp.conj(coefficients.T)
        residual_mode_gram = (
            adjoint_coefficients @ evaluation.residual_gram @ coefficients
        )
        residual_squared = jnp.maximum(
            jnp.real(jnp.diag(residual_mode_gram)),
            0.0,
        )
        residual_norms = jnp.sqrt(residual_squared)
        action_gram = _form_matrix(
            self.residual_pairing,
            assembly.operator_fields,
            assembly.operator_fields,
            assembly.realization,
            role="Invariant-subspace action norm",
            **kwargs,
        )
        metric_gram = _form_matrix(
            self.residual_pairing,
            assembly.metric_fields,
            assembly.metric_fields,
            assembly.realization,
            role="Invariant-subspace metric norm",
            **kwargs,
        )
        action_mode_gram = adjoint_coefficients @ action_gram @ coefficients
        metric_mode_gram = adjoint_coefficients @ metric_gram @ coefficients
        action_norms = jnp.sqrt(jnp.maximum(jnp.real(jnp.diag(action_mode_gram)), 0.0))
        metric_norms = jnp.sqrt(jnp.maximum(jnp.real(jnp.diag(metric_mode_gram)), 0.0))
        tiny = jnp.finfo(residual_norms.dtype).tiny
        scale = action_norms + jnp.abs(reduced.eigenvalues) * metric_norms
        relative_residuals = residual_norms / jnp.maximum(scale, tiny)
        return InvariantSubspaceResidualResult(
            evaluation=evaluation,
            reduced=reduced,
            modes=_linear_combinations(assembly.fields, coefficients),
            residual_modes=_linear_combinations(
                assembly.residual_fields,
                coefficients,
            ),
            residual_norms=residual_norms,
            relative_residuals=relative_residuals,
            objective_vars=self.objective_vars,
        )


__all__ = [
    "EigenspaceAction",
    "FormDensity",
    "InvariantSubspaceResidual",
    "InvariantSubspaceResidualEvaluation",
    "InvariantSubspaceResidualResult",
    "VariationalEigenspace",
    "VariationalEigenspaceEvaluation",
    "VariationalEigenspaceResult",
]
