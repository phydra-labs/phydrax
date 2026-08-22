#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum
from typing import Any

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class DiscretizationRole(StrEnum):
    """Semantic role of one approximation in a simulation."""

    PHYSICAL = "physical"
    TEMPORAL = "temporal"
    RANDOM_INPUT = "random_input"
    DRIVER = "driver"
    ENSEMBLE = "ensemble"
    RESIDUAL = "residual"
    CONTROL = "control"
    OBSERVATION = "observation"
    AUXILIARY = "auxiliary"


class DiscretizationCapability(StrEnum):
    """Structural operation class implemented by a prepared discretization."""

    PROJECTION = "projection"
    RECONSTRUCTION = "reconstruction"
    TRACE = "trace"
    STRONG_DERIVATIVE = "strong_derivative"
    VARIATIONAL_ASSEMBLY = "variational_assembly"
    CONSERVATIVE_FLUX = "conservative_flux"
    BOUNDARY_INTEGRAL = "boundary_integral"
    SPECTRAL_TRANSFORM = "spectral_transform"
    ENTITY_INCIDENCE = "entity_incidence"
    FIELD_TRANSFER = "field_transfer"
    GEOMETRY_REFRESH = "geometry_refresh"
    TOPOLOGY_REFRESH_FIXED_CAPACITY = "topology_refresh_fixed_capacity"
    DIFFERENTIABLE_GEOMETRY = "differentiable_geometry"
    MATRIX_FREE = "matrix_free"
    SPARSE_ASSEMBLY = "sparse_assembly"


def nonempty_identifier(name: str, value: str, /) -> str:
    """Normalize one required identifier."""
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def resolved_identifier(
    name: str,
    value: str | None,
    payload: dict[str, Any],
    /,
) -> str:
    """Use an explicit identifier or fingerprint canonical structural metadata."""
    return (
        canonical_fingerprint(payload)
        if value is None
        else nonempty_identifier(name, value)
    )


def normalized_capabilities(
    capabilities: tuple[DiscretizationCapability, ...] | list[DiscretizationCapability],
    /,
) -> tuple[DiscretizationCapability, ...]:
    """Return unique capabilities in stable lexical order."""
    values = tuple(capabilities)
    if not all(isinstance(value, DiscretizationCapability) for value in values):
        raise TypeError("capabilities must contain DiscretizationCapability values.")
    return tuple(sorted(set(values), key=str))


class DiscretizationKey(StrictModule, NonTrainableState):
    """Stable semantic address for one approximation part."""

    name: str = eqx.field(static=True)
    role: DiscretizationRole = eqx.field(static=True)
    domain_labels: tuple[str, ...] = eqx.field(static=True)
    key_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        role: DiscretizationRole | str,
        /,
        *,
        domain_labels: tuple[str, ...] | list[str] = (),
        key_id: str | None = None,
    ):
        name_ = nonempty_identifier("name", name)
        role_ = DiscretizationRole(role)
        labels = tuple(str(label) for label in domain_labels)
        if any(not label for label in labels):
            raise ValueError("domain_labels must be non-empty strings.")
        if len(set(labels)) != len(labels):
            raise ValueError("domain_labels must be unique.")
        self.name = name_
        self.role = role_
        self.domain_labels = labels
        self.key_id = resolved_identifier(
            "key_id",
            key_id,
            {
                "kind": "discretization-key",
                "name": name_,
                "role": str(role_),
                "domain_labels": list(labels),
            },
        )


class PreparationReport(StrictModule, NonTrainableState):
    """Auditable structural and resource summary for one preparation."""

    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    diagnostics: tuple[str, ...] = eqx.field(static=True)
    resource_counts: tuple[tuple[str, int], ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        capabilities: tuple[DiscretizationCapability, ...]
        | list[DiscretizationCapability] = (),
        diagnostics: tuple[str, ...] | list[str] = (),
        resource_counts: dict[str, int] | tuple[tuple[str, int], ...] = (),
        report_id: str | None = None,
    ):
        capabilities_ = normalized_capabilities(capabilities)
        diagnostics_ = tuple(str(value) for value in diagnostics)
        if any(not value for value in diagnostics_):
            raise ValueError("diagnostics must contain non-empty strings.")
        items = (
            tuple(resource_counts.items())
            if isinstance(resource_counts, dict)
            else tuple(resource_counts)
        )
        counts = tuple(sorted((str(name), int(value)) for name, value in items))
        if any(not name or value < 0 for name, value in counts):
            raise ValueError(
                "resource_counts require non-empty names and non-negative values."
            )
        if len({name for name, _ in counts}) != len(counts):
            raise ValueError("resource_counts names must be unique.")
        self.capabilities = capabilities_
        self.diagnostics = diagnostics_
        self.resource_counts = counts
        self.report_id = resolved_identifier(
            "report_id",
            report_id,
            {
                "kind": "discretization-preparation-report",
                "capabilities": [str(value) for value in capabilities_],
                "diagnostics": list(diagnostics_),
                "resource_counts": [list(value) for value in counts],
            },
        )


__all__ = [
    "DiscretizationCapability",
    "DiscretizationKey",
    "DiscretizationRole",
    "PreparationReport",
]
