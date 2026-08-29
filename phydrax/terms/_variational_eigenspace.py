#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import equinox as eqx
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
from ..linalg.eigen import (
    block_rayleigh_trace,
    BlockRayleighEvaluation,
    ReducedRitzResult,
    solve_reduced_ritz,
)
from ..operators import conjugate


FormDensity = Callable[[DomainFunction, DomainFunction], DomainFunction]


def _default_mass(left: DomainFunction, right: DomainFunction, /) -> DomainFunction:
    return conjugate(left) * right


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
    materialization_policy: Literal["fixed", "per_step", "caller"] = eqx.field(
        static=True
    )

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
        materialization_policy: Literal["fixed", "per_step", "caller"] = "fixed",
        fixed_realization: IntegrationRealization | None = None,
        fixed_key: Key[Array, ""] | None = None,
    ):
        if not callable(stiffness_form):
            raise TypeError("stiffness_form must be callable.")
        if mass_form is not None and not callable(mass_form):
            raise TypeError("mass_form must be callable or None.")
        variables = tuple(str(name) for name in objective_vars)
        if not variables or any(not name for name in variables):
            raise ValueError("objective_vars must contain non-empty field names.")
        if len(set(variables)) != len(variables):
            raise ValueError("objective_vars must not contain duplicates.")
        tolerance_ = float(tolerance)
        if not math.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        weight_ = float(weight)
        if not math.isfinite(weight_):
            raise ValueError("weight must be finite.")
        policy = str(materialization_policy).lower()
        if policy not in ("fixed", "per_step", "caller"):
            raise ValueError(
                "materialization_policy must be 'fixed', 'per_step', or 'caller'."
            )
        if policy == "fixed":
            if fixed_realization is None:
                if _requires_random_key(plan):
                    if fixed_key is None:
                        raise ValueError(
                            "A randomized fixed VariationalEigenspace requires fixed_key=."
                        )
                    fixed_realization = materialize(target, plan, key=fixed_key)
                else:
                    if fixed_key is not None:
                        raise ValueError(
                            "A deterministic fixed VariationalEigenspace does not "
                            "consume fixed_key=."
                        )
                    fixed_realization = materialize(target, plan)
        elif fixed_realization is not None or fixed_key is not None:
            raise ValueError(
                "fixed_realization/fixed_key require materialization_policy='fixed'."
            )
        self.objective_vars = variables
        self.target = target
        self.plan = plan
        self.stiffness_form = stiffness_form
        self.mass_form = _default_mass if mass_form is None else mass_form
        self.fixed_realization = fixed_realization
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
        if self.materialization_policy == "fixed":
            if self.fixed_realization is None:
                raise RuntimeError("Fixed VariationalEigenspace has no realization.")
            return self.fixed_realization
        if self.materialization_policy == "caller":
            return None
        if _requires_random_key(self.plan):
            return materialize(self.target, self.plan, key=key)
        return materialize(self.target, self.plan)

    def _realization(
        self,
        batch: IntegrationRealization | None,
        /,
        *,
        key: Key[Array, ""],
    ) -> IntegrationRealization:
        realization = batch
        if realization is None:
            if self.materialization_policy == "caller":
                raise ValueError(
                    "Caller-managed VariationalEigenspace requires "
                    "batch=IntegrationRealization."
                )
            realization = self.sample(key=key)
        if not isinstance(realization, IntegrationRealization):
            raise TypeError(
                "VariationalEigenspace batch must be an IntegrationRealization."
            )
        return realization

    def _fields(
        self,
        functions: Mapping[str, DomainFunction],
        /,
    ) -> tuple[DomainFunction, ...]:
        fields: list[DomainFunction] = []
        for name in self.objective_vars:
            if name not in functions:
                raise KeyError(f"Missing variational eigenspace field {name!r}.")
            field = functions[name]
            if not isinstance(field, DomainFunction):
                raise TypeError(
                    f"Variational eigenspace field {name!r} must be a DomainFunction."
                )
            fields.append(field)
        return tuple(fields)

    def _integral(
        self,
        density: DomainFunction,
        realization: IntegrationRealization,
        /,
        **kwargs: Any,
    ) -> Array:
        if not isinstance(density, DomainFunction):
            raise TypeError("Variational eigenspace forms must return DomainFunction.")
        estimate = reduce(density, realization, **kwargs)
        if estimate.value.dims != ():
            raise ValueError(
                "Variational eigenspace forms must reduce to scalar fields; "
                f"got dims={estimate.value.dims}."
            )
        value = jnp.asarray(estimate.value.data).reshape(())
        return eqx.error_if(
            value,
            estimate.status != int(IntegrationStatus.CONVERGED),
            "Variational eigenspace integration did not converge.",
        )

    def _form_matrix(
        self,
        form: FormDensity,
        fields: tuple[DomainFunction, ...],
        realization: IntegrationRealization,
        /,
        **kwargs: Any,
    ) -> Array:
        rows = []
        for left in fields:
            rows.append(
                jnp.stack(
                    tuple(
                        self._integral(form(left, right), realization, **kwargs)
                        for right in fields
                    )
                )
            )
        return jnp.stack(tuple(rows))

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
        realization = self._realization(batch, key=key)
        fields = self._fields(functions)
        stiffness = self._form_matrix(
            self.stiffness_form,
            fields,
            realization,
            **kwargs,
        )
        mass = self._form_matrix(
            self.mass_form,
            fields,
            realization,
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
        fields = self._fields(functions)
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
        modes: list[DomainFunction] = []
        for mode_index in range(int(coefficients.shape[1])):
            mode = coefficients[0, mode_index] * fields[0]
            for field_index in range(1, len(fields)):
                mode = mode + coefficients[field_index, mode_index] * fields[field_index]
            modes.append(mode)
        return VariationalEigenspaceResult(
            evaluation=evaluation,
            reduced=reduced,
            modes=tuple(modes),
            objective_vars=self.objective_vars,
        )


__all__ = [
    "FormDensity",
    "VariationalEigenspace",
    "VariationalEigenspaceEvaluation",
    "VariationalEigenspaceResult",
]
