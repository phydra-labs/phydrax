#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from .._strict import StrictModule
from ..domain._normalized_density import normalize_density_field
from ..domain._separated_density import SeparatedLogDensityField
from ..integration._api import IntegrationRealization, reduce
from ._fokker_planck_approximation import DensityFokkerPlanckResult


class SeparatedFokkerPlanckPlan(StrictModule):
    """One fixed-rank represented density epoch and independent validation."""

    field: SeparatedLogDensityField
    domain: Any
    realization: IntegrationRealization
    validation_realization: IntegrationRealization
    solver: Any
    validation_operator: Any
    reference: str = eqx.field(static=True)
    state_var: str = eqx.field(static=True)

    def __init__(
        self,
        field: SeparatedLogDensityField,
        domain: Any,
        realization: IntegrationRealization,
        validation_realization: IntegrationRealization,
        /,
        *,
        solver: Any = None,
        validation_operator: Any = None,
        reference: str = "coordinate",
        state_var: str = "x",
    ):
        if not isinstance(field, SeparatedLogDensityField):
            raise TypeError("field must be a SeparatedLogDensityField.")
        if not isinstance(realization, IntegrationRealization) or not isinstance(
            validation_realization, IntegrationRealization
        ):
            raise TypeError("realizations must be IntegrationRealization values.")
        if solver is not None and not callable(solver):
            raise TypeError("solver must be callable or None.")
        if validation_operator is not None and not callable(validation_operator):
            raise TypeError("validation_operator must be callable or None.")
        field.as_domain_function(domain)
        self.field = field
        self.domain = domain
        self.realization = realization
        self.validation_realization = validation_realization
        self.solver = solver
        self.validation_operator = validation_operator
        self.reference = str(reference)
        self.state_var = str(state_var)


def solve_separated_fokker_planck(
    plan: SeparatedFokkerPlanckPlan, /
) -> DensityFokkerPlanckResult:
    """Solve or evaluate one fixed separated-rank normalized density epoch."""
    if not isinstance(plan, SeparatedFokkerPlanckPlan):
        raise TypeError("plan must be a SeparatedFokkerPlanckPlan.")
    represented = plan.field.as_domain_function(plan.domain)
    log_field = (
        represented if plan.solver is None else plan.solver(represented, plan.realization)
    )
    normalized = normalize_density_field(
        log_field,
        plan.realization,
        reference=plan.reference,
        state_var=plan.state_var,
    )
    if plan.validation_operator is None:
        held_out = None
        validation_ok = jnp.asarray(True)
    else:
        residual = plan.validation_operator(normalized.field)
        held_out = reduce(residual * residual, plan.validation_realization)
        validation_ok = held_out.successful & jnp.isfinite(jnp.asarray(held_out.value))
    return DensityFokkerPlanckResult(
        normalized_density=normalized,
        held_out_residual=held_out,
        normalization_error=normalized.normalization.error_estimate,
        valid=normalized.finite & normalized.normalization.successful & validation_ok,
        approximation_kind="separated-rank-normalized-density",
        bounded_non_claim=(
            "Rank is fixed in this execution epoch. Rank enrichment creates a new "
            "represented density and is not differentiated or called a continuum proof."
        ),
    )


__all__ = ["SeparatedFokkerPlanckPlan", "solve_separated_fokker_planck"]
