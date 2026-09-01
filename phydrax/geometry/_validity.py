#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from .design._schema import DesignState, ParameterSchema


class GeometryValidityDisposition(IntEnum):
    """Runtime disposition of a geometry representation at one design state."""

    VALID = 0
    INVALID = 1
    INCONCLUSIVE = 2


class GeometryValidityEvidence(StrictModule):
    """JAX-safe evidence for state-dependent geometry validity."""

    finite: Array
    conditions_satisfied: Array
    resolved: Array
    margins: Array
    margin_names: tuple[str, ...] = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        finite: Any,
        conditions_satisfied: Any,
        resolved: Any,
        margins: Any = (),
        margin_names: tuple[str, ...] = (),
        contract_id: str,
    ):
        finite_ = jnp.asarray(finite, dtype=bool)
        satisfied_ = jnp.asarray(conditions_satisfied, dtype=bool)
        resolved_ = jnp.asarray(resolved, dtype=bool)
        margins_ = jnp.asarray(margins, dtype=float)
        if finite_.shape != () or satisfied_.shape != () or resolved_.shape != ():
            raise ValueError("Geometry validity flags must be scalar.")
        if margins_.ndim != 1 or margins_.shape != (len(margin_names),):
            raise ValueError(
                "Geometry validity margins must be a vector matching margin_names."
            )
        if not contract_id:
            raise ValueError("Geometry validity contract_id must be non-empty.")
        if any(not name for name in margin_names):
            raise ValueError("Geometry validity margin names must be non-empty.")
        self.finite = finite_
        self.conditions_satisfied = satisfied_
        self.resolved = resolved_
        self.margins = margins_
        self.margin_names = tuple(margin_names)
        self.contract_id = contract_id

    @property
    def disposition(self) -> Array:
        invalid = (~self.finite) | (~self.conditions_satisfied)
        return jnp.where(
            invalid,
            int(GeometryValidityDisposition.INVALID),
            jnp.where(
                self.resolved,
                int(GeometryValidityDisposition.VALID),
                int(GeometryValidityDisposition.INCONCLUSIVE),
            ),
        ).astype(jnp.int32)

    @property
    def accepted(self) -> Array:
        return self.disposition == int(GeometryValidityDisposition.VALID)

    def combined_with(
        self,
        other: GeometryValidityEvidence,
        /,
        *,
        contract_id: str,
    ) -> GeometryValidityEvidence:
        return GeometryValidityEvidence(
            finite=self.finite & other.finite,
            conditions_satisfied=(self.conditions_satisfied & other.conditions_satisfied),
            resolved=self.resolved & other.resolved,
            margins=jnp.concatenate((self.margins, other.margins)),
            margin_names=self.margin_names + other.margin_names,
            contract_id=contract_id,
        )


@runtime_checkable
class GeometryValidityProvider(Protocol):
    """Structural provider of representation-specific runtime validity."""

    def geometry_validity(
        self,
        state: DesignState,
        /,
    ) -> GeometryValidityEvidence: ...


def parameter_validity(
    schema: ParameterSchema,
    state: DesignState,
    /,
) -> GeometryValidityEvidence:
    """Evaluate finiteness and declared physical parameter bounds."""

    if state.schema != schema:
        raise ValueError("Geometry validity state must use the compiled schema.")
    finite = jnp.asarray(True)
    satisfied = jnp.asarray(True)
    margins: list[Array] = []
    names: list[str] = []
    for spec, value in zip(schema.specs, state.values, strict=True):
        finite = finite & jnp.all(jnp.isfinite(value))
        lower, upper = spec.bounds
        scale = jnp.asarray(spec.physical_scale, dtype=value.dtype)
        if lower is not None:
            margin = jnp.min((value - lower) / scale)
            margins.append(margin)
            names.append(f"{spec.parameter_id}:lower")
            satisfied = satisfied & (margin >= 0.0)
        if upper is not None:
            margin = jnp.min((upper - value) / scale)
            margins.append(margin)
            names.append(f"{spec.parameter_id}:upper")
            satisfied = satisfied & (margin >= 0.0)
    margin_array = jnp.stack(margins) if margins else jnp.empty((0,), dtype=float)
    return GeometryValidityEvidence(
        finite=finite,
        conditions_satisfied=satisfied,
        resolved=True,
        margins=margin_array,
        margin_names=tuple(names),
        contract_id="parameter_schema_bounds",
    )


def unrestricted_validity(*, contract_id: str) -> GeometryValidityEvidence:
    """Return resolved evidence for a representation valid over all states."""

    return GeometryValidityEvidence(
        finite=True,
        conditions_satisfied=True,
        resolved=True,
        contract_id=contract_id,
    )


def representation_validity(
    kernel: Any,
    state: DesignState,
    /,
) -> GeometryValidityEvidence:
    """Evaluate a kernel's representation-specific validity contract."""

    if isinstance(kernel, GeometryValidityProvider):
        return kernel.geometry_validity(state)
    certificate = kernel.field_certificate
    if certificate.validity_region == "all_space":
        return unrestricted_validity(contract_id=f"{type(kernel).__name__}:all_space")
    return inconclusive_validity(contract_id=f"{type(kernel).__name__}:restricted")


def combine_validity(
    evidence: tuple[GeometryValidityEvidence, ...],
    /,
    *,
    contract_id: str,
) -> GeometryValidityEvidence:
    """Combine a non-empty family of validity evidence objects."""

    if not evidence:
        raise ValueError("At least one geometry validity evidence object is required.")
    combined = evidence[0]
    for item in evidence[1:]:
        combined = combined.combined_with(
            item,
            contract_id=contract_id,
        )
    if len(evidence) == 1 and combined.contract_id != contract_id:
        combined = GeometryValidityEvidence(
            finite=combined.finite,
            conditions_satisfied=combined.conditions_satisfied,
            resolved=combined.resolved,
            margins=combined.margins,
            margin_names=combined.margin_names,
            contract_id=contract_id,
        )
    return combined


def inconclusive_validity(*, contract_id: str) -> GeometryValidityEvidence:
    """Return evidence for a restricted representation without an evaluator."""

    return GeometryValidityEvidence(
        finite=True,
        conditions_satisfied=True,
        resolved=False,
        contract_id=contract_id,
    )


__all__ = [
    "GeometryValidityDisposition",
    "GeometryValidityEvidence",
    "GeometryValidityProvider",
]
