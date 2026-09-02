# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._term import AbstractSamplingTerm
from ..domain import DomainFunction
from ..integration import (
    CallerIntegration,
    IntegrationRealization,
    IntegrationSource,
    IntegrationStatus,
    reduce,
    resolve_integration,
)


class TargetConsistencyTerm(AbstractSamplingTerm):
    """Measured squared discrepancy against stopped delayed/EMA target functions."""

    field: str = eqx.field(static=True)
    source: IntegrationSource
    weight: float = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        source: IntegrationSource,
        /,
        *,
        weight: ArrayLike = 1.0,
        label: str | None = None,
    ):
        if not field:
            raise ValueError("Target consistency field must be nonempty.")
        self.field = str(field)
        self.source = source
        self.weight = float(weight)
        self.label = None if label is None else str(label)

    def sample(self, *, key: Key[Array, ""] = DOC_KEY0) -> IntegrationRealization | None:
        if isinstance(self.source, CallerIntegration):
            return None
        return resolve_integration(self.source, key=key)

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: IntegrationRealization | None = None,
        target_functions: Mapping[str, DomainFunction] | None = None,
        **kwargs: Any,
    ) -> Array:
        if target_functions is None:
            raise ValueError("TargetConsistencyTerm requires target_functions.")
        if self.field not in functions or self.field not in target_functions:
            raise KeyError(f"Target consistency field {self.field!r} is missing.")
        current = functions[self.field]
        target = target_functions[self.field]
        if not current.domain.same_support(target.domain):
            raise ValueError("Target and training functions must share domain support.")
        difference = current - target
        integrand = difference * difference
        realization = batch
        if realization is None:
            if isinstance(self.source, CallerIntegration):
                raise ValueError(
                    "Caller target consistency requires IntegrationRealization."
                )
            realization = self.sample(key=key)
        elif isinstance(self.source, CallerIntegration):
            realization = resolve_integration(
                self.source, key=key, realization=realization
            )
        if not isinstance(realization, IntegrationRealization):
            raise TypeError("Target consistency batch must be IntegrationRealization.")
        estimate = reduce(integrand, realization, **kwargs)
        value = jnp.asarray(estimate.value.data).reshape(())
        value = eqx.error_if(
            value,
            estimate.status != int(IntegrationStatus.CONVERGED),
            "Target consistency integration did not converge.",
        )
        return jnp.asarray(self.weight, dtype=value.dtype) * value


__all__ = ["TargetConsistencyTerm"]
