#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax.numpy as jnp

from .._strict import StrictModule
from ..integration import (
    AdaptiveIntegration,
    CallerIntegration,
    ComponentTarget,
    DensityTarget,
    FixedIntegration,
    IntegrationEstimate,
    IntegrationRealization,
    IntegrationStatus,
    PerStepIntegration,
)
from ..integration._execution import resolve_integration


class _PreparedIntegrationRealization(StrictModule):
    """Internal marker for a realization frozen by objective preparation."""

    realization: IntegrationRealization
    def __init__(self, realization: IntegrationRealization, /):
        self.realization = realization



def prepare_term_realization(
    realization: IntegrationRealization,
    /,
) -> _PreparedIntegrationRealization:
    if not isinstance(realization, IntegrationRealization):
        raise TypeError("realization must be an IntegrationRealization.")
    return _PreparedIntegrationRealization(realization)


def resolve_term_realization(
    source,
    /,
    *,
    key,
    realization: IntegrationRealization | _PreparedIntegrationRealization | None,
) -> IntegrationRealization:
    """Resolve public source semantics plus solver-prepared realizations."""
    if not isinstance(realization, _PreparedIntegrationRealization):
        return resolve_integration(source, key=key, realization=realization)
    return realization.realization

def checked_estimate_field(estimate: IntegrationEstimate, /) -> cx.Field:
    """Return one estimate field after convergence validation."""
    if not isinstance(estimate, IntegrationEstimate):
        raise TypeError("Integrated terms require an IntegrationEstimate.")
    if not isinstance(estimate.value, cx.Field):
        raise TypeError("Integrated term reductions must return a coordax.Field.")
    data = jnp.asarray(estimate.value.data)
    data = eqx.error_if(
        data,
        estimate.status != int(IntegrationStatus.CONVERGED),
        "Term integration did not converge.",
    )
    return cx.Field(data, dims=estimate.value.dims)


def validate_condition_source(on, source, /) -> None:
    """Reject a physical integration source that targets another component."""
    if isinstance(source, PerStepIntegration):
        target = source.target
    elif isinstance(source, FixedIntegration):
        target = source.realization.target
    elif isinstance(source, (CallerIntegration, AdaptiveIntegration)):
        target = source.target
    else:
        raise TypeError("Expected a typed IntegrationSource.")
    while isinstance(target, DensityTarget):
        target = target.base
    if not isinstance(target, ComponentTarget):
        raise TypeError("Condition penalties require a component integration target.")
    if not bool(eqx.tree_equal(on, target.component)):
        raise ValueError(
            "Condition support and integration source component are incompatible."
        )


__all__ = [
    "checked_estimate_field",
    "prepare_term_realization",
    "resolve_term_realization",
    "validate_condition_source",
]
