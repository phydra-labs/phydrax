#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

import equinox as eqx
from jaxtyping import Array, Key

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..domain import (
    BatchEvaluator,
    DerivativeBackend,
    DerivativeBasis,
    DerivativeMode,
    DerivativeRule,
    DomainFunction,
    GridBatch,
    PointBatch,
)
from ..terms import ResidualPenalty
from ._functional_solver import FunctionalSolver


class _FrozenFieldEvaluator(StrictModule, BatchEvaluator, NonTrainableState):
    field: DomainFunction

    def __init__(self, field: DomainFunction, /):
        if not isinstance(field, DomainFunction):
            raise TypeError("field must be a DomainFunction.")
        self.field = field

    def __call_batch__(
        self,
        batch: PointBatch | GridBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ):
        return self.field(batch, key=key, **kwargs)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        return self.field.func(*args, key=key, **kwargs)


class _FrozenDerivativeRule(StrictModule, DerivativeRule, NonTrainableState):
    rule: DerivativeRule

    def derive(
        self,
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        derived = self.rule.derive(
            var=var,
            axis=axis,
            order=order,
            mode=mode,
            backend=backend,
            basis=basis,
            periodic=periodic,
        )
        return None if derived is None else freeze_domain_function(derived)

    def derive_laplacian(
        self,
        *,
        var: str,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        derived = self.rule.derive_laplacian(
            var=var,
            mode=mode,
            backend=backend,
            basis=basis,
            periodic=periodic,
        )
        return None if derived is None else freeze_domain_function(derived)


def freeze_domain_function(field: DomainFunction, /) -> DomainFunction:
    """Return an equivalent field whose complete evaluator is solver-frozen."""
    return DomainFunction(
        domain=field.domain,
        deps=field.deps,
        func=_FrozenFieldEvaluator(field),
        metadata=field.metadata,
        derivative_rule=(
            None
            if field.derivative_rule is None
            else _FrozenDerivativeRule(field.derivative_rule)
        ),
    )


class FunctionalCorrectionProblem(StrictModule):
    """Correction-only training solver and its unchanged physical objective."""

    training_solver: FunctionalSolver
    physical_solver: FunctionalSolver
    base_functions: frozendict[str, DomainFunction]
    correction_functions: frozendict[str, DomainFunction]
    epsilon: float = eqx.field(static=True)

    def finalize(self, trained: FunctionalSolver, /) -> FunctionalSolver:
        """Bind trained composed fields back to the unscaled physical objective."""
        if not isinstance(trained, FunctionalSolver):
            raise TypeError("trained must be a FunctionalSolver.")
        if (
            trained.discretization_bundle.bundle_id
            != self.training_solver.discretization_bundle.bundle_id
        ):
            raise ValueError("Trained solver does not match this correction problem.")
        return eqx.tree_at(
            lambda solver: solver.functions,
            self.physical_solver,
            trained.functions,
        )


def prepare_functional_correction(
    solver: FunctionalSolver,
    correction_functions: Mapping[str, DomainFunction],
    /,
    *,
    epsilon: float,
) -> FunctionalCorrectionProblem:
    """Prepare exact scaled nonlinear defect correction ``R(u₀+εδu)/ε``."""
    if not isinstance(solver, FunctionalSolver):
        raise TypeError("solver must be a FunctionalSolver.")
    epsilon_ = float(epsilon)
    if not isfinite(epsilon_) or epsilon_ <= 0.0:
        raise ValueError("epsilon must be finite and positive.")
    corrections = frozendict(correction_functions)
    if not corrections:
        raise ValueError("At least one correction field is required.")
    unknown = tuple(name for name in corrections if name not in solver.functions)
    if unknown:
        raise KeyError(f"Correction fields do not exist in the base solver: {unknown!r}.")
    unsupported = tuple(
        type(term).__name__
        for term in solver.terms
        if not isinstance(term, ResidualPenalty)
    )
    if unsupported:
        raise TypeError(
            "Exact functional correction requires a pure ResidualPenalty objective; "
            f"got unsupported terms {unsupported!r}."
        )
    frozen_base = frozendict(
        {name: freeze_domain_function(field) for name, field in solver.functions.items()}
    )
    composed = dict(frozen_base)
    for name, correction in corrections.items():
        if not isinstance(correction, DomainFunction):
            raise TypeError("Correction fields must be DomainFunction values.")
        base = solver.functions[name]
        if not base.domain.same_support(correction.domain):
            raise ValueError(f"Correction field {name!r} has incompatible support.")
        composed[name] = frozen_base[name] + epsilon_ * correction
    scaled_terms = [
        ResidualPenalty(
            term.condition,
            term.source,
            scale=term.scale / (epsilon_**2),
            density=term.density,
            blocks=term.blocks,
            label=term.label,
            data_accuracy_eps=term.data_accuracy_eps,
        )
        for term in solver.terms
    ]
    training_solver = FunctionalSolver(
        functions=composed,
        terms=tuple(scaled_terms),
        evaluation_terms=solver.evaluation_terms,
        enforcement=solver.enforcement,
    )
    physical_solver = FunctionalSolver(
        functions=composed,
        terms=solver.terms,
        evaluation_terms=solver.evaluation_terms,
        enforcement=solver.enforcement,
    )
    return FunctionalCorrectionProblem(
        training_solver,
        physical_solver,
        frozen_base,
        corrections,
        epsilon_,
    )


__all__ = [
    "FunctionalCorrectionProblem",
    "freeze_domain_function",
    "prepare_functional_correction",
]
