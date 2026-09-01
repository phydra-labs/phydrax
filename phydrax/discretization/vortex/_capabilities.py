#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._precision import VortexPrecisionPolicy


_SOURCE_FIELDS = frozenset(
    ("positions", "strength", "active_mask", "core_radius", "volume")
)
_OUTPUT_FIELDS = frozenset(("velocity", "velocity_gradient", "vorticity"))
_DOMAINS = frozenset(("free-space", "periodic", "bounded", "mixed"))
_DERIVATIVES = frozenset(
    (
        "source-position",
        "source-strength",
        "source-core-radius",
        "source-volume",
        "target-position",
    )
)
_TARGET_TOPOLOGIES = frozenset(("same-support", "arbitrary-targets"))
_ACCELERATIONS = frozenset(
    (
        "direct",
        "fixed-cluster",
        "particle-mesh",
        "p3m",
        "ewald",
        "fmm",
        "distributed",
        "none",
    )
)


def _names(
    name: str, values: tuple[str, ...], allowed: frozenset[str], /
) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{name} must contain non-empty values.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} cannot contain duplicates.")
    invalid = tuple(value for value in normalized if value not in allowed)
    if invalid:
        raise ValueError(f"Unsupported {name}: {invalid}.")
    return normalized


def _source_kinds(values: tuple[str, ...], /) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError("source_kinds must contain non-empty values.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("source_kinds cannot contain duplicates.")
    return normalized


class VortexVelocityCapabilities(StrictModule, NonTrainableState):
    """Immutable scientific and execution contract of a velocity backend."""

    dimension: int = eqx.field(static=True)
    source_kinds: tuple[str, ...] = eqx.field(static=True)
    required_source_fields: tuple[str, ...] = eqx.field(static=True)
    supported_fields: tuple[str, ...] = eqx.field(static=True)
    domain: str = eqx.field(static=True)
    precision: VortexPrecisionPolicy
    derivatives: tuple[str, ...] = eqx.field(static=True)
    target_topologies: tuple[str, ...] = eqx.field(static=True)
    acceleration: str = eqx.field(static=True)
    capabilities_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        source_kinds: tuple[str, ...] = ("particle",),
        required_source_fields: tuple[str, ...] = (
            "positions",
            "strength",
            "active_mask",
        ),
        supported_fields: tuple[str, ...] = ("velocity",),
        domain: str = "free-space",
        precision: VortexPrecisionPolicy | None = None,
        derivatives: tuple[str, ...] = (),
        target_topologies: tuple[str, ...] = (
            "same-support",
            "arbitrary-targets",
        ),
        acceleration: str = "none",
    ):
        dimension_ = int(dimension)
        if dimension_ not in (2, 3):
            raise ValueError("Vortex velocity dimension must be 2 or 3.")
        kinds = _source_kinds(source_kinds)
        requirements = _names(
            "required_source_fields", required_source_fields, _SOURCE_FIELDS
        )
        if not {"positions", "strength", "active_mask"}.issubset(requirements):
            raise ValueError(
                "Vortex velocity capabilities must require positions, strength, "
                "and active_mask."
            )
        fields = _names("supported_fields", supported_fields, _OUTPUT_FIELDS)
        domain_ = str(domain).strip()
        if domain_ not in _DOMAINS:
            raise ValueError(
                "Vortex domain must be free-space, periodic, bounded, or mixed."
            )
        precision_ = VortexPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, VortexPrecisionPolicy):
            raise TypeError("precision must be VortexPrecisionPolicy or None.")
        derivatives_ = (
            () if not derivatives else _names("derivatives", derivatives, _DERIVATIVES)
        )
        topologies = _names("target_topologies", target_topologies, _TARGET_TOPOLOGIES)
        acceleration_ = str(acceleration).strip()
        if acceleration_ not in _ACCELERATIONS:
            raise ValueError(f"Unsupported vortex acceleration '{acceleration_}'.")
        if acceleration_ == "particle-mesh" and domain_ not in (
            "periodic",
            "bounded",
        ):
            raise ValueError(
                "Particle-mesh vortex velocity requires periodic or bounded domain."
            )
        self.dimension = dimension_
        self.source_kinds = kinds
        self.required_source_fields = requirements
        self.supported_fields = fields
        self.domain = domain_
        self.precision = precision_
        self.derivatives = derivatives_
        self.target_topologies = topologies
        self.acceleration = acceleration_
        self.capabilities_id = canonical_fingerprint(
            {
                "kind": "vortex-velocity-capabilities",
                "dimension": dimension_,
                "source_kinds": list(kinds),
                "required_source_fields": list(requirements),
                "supported_fields": list(fields),
                "domain": domain_,
                "precision": precision_.policy_id,
                "derivatives": list(derivatives_),
                "target_topologies": list(topologies),
                "acceleration": acceleration_,
            }
        )

    @property
    def requires_core_radius(self) -> bool:
        return "core_radius" in self.required_source_fields

    @property
    def requires_volume(self) -> bool:
        return "volume" in self.required_source_fields


class VortexDiffusionCapabilities(StrictModule, NonTrainableState):
    """Immutable source-property contract of a particle diffusion backend."""

    dimension: int = eqx.field(static=True)
    source_kinds: tuple[str, ...] = eqx.field(static=True)
    required_source_fields: tuple[str, ...] = eqx.field(static=True)
    domain: str = eqx.field(static=True)
    precision: VortexPrecisionPolicy
    derivatives: tuple[str, ...] = eqx.field(static=True)
    topology: str = eqx.field(static=True)
    acceleration: str = eqx.field(static=True)
    capabilities_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        source_kinds: tuple[str, ...] = ("particle",),
        required_source_fields: tuple[str, ...] = (
            "positions",
            "strength",
            "active_mask",
        ),
        domain: str = "free-space",
        precision: VortexPrecisionPolicy | None = None,
        derivatives: tuple[str, ...] = (),
        topology: str = "same-support",
        acceleration: str = "none",
    ):
        dimension_ = int(dimension)
        if dimension_ not in (2, 3):
            raise ValueError("Vortex diffusion dimension must be 2 or 3.")
        kinds = _source_kinds(source_kinds)
        requirements = _names(
            "required_source_fields", required_source_fields, _SOURCE_FIELDS
        )
        if not {"positions", "strength", "active_mask"}.issubset(requirements):
            raise ValueError(
                "Vortex diffusion capabilities must require positions, strength, "
                "and active_mask."
            )
        domain_ = str(domain).strip()
        if domain_ not in _DOMAINS:
            raise ValueError(
                "Vortex domain must be free-space, periodic, bounded, or mixed."
            )
        precision_ = VortexPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, VortexPrecisionPolicy):
            raise TypeError("precision must be VortexPrecisionPolicy or None.")
        derivatives_ = (
            () if not derivatives else _names("derivatives", derivatives, _DERIVATIVES)
        )
        topology_ = str(topology).strip()
        if topology_ != "same-support":
            raise ValueError("Particle diffusion requires same-support topology.")
        acceleration_ = str(acceleration).strip()
        if acceleration_ not in _ACCELERATIONS:
            raise ValueError(f"Unsupported vortex acceleration '{acceleration_}'.")
        self.dimension = dimension_
        self.source_kinds = kinds
        self.required_source_fields = requirements
        self.domain = domain_
        self.precision = precision_
        self.derivatives = derivatives_
        self.topology = topology_
        self.acceleration = acceleration_
        self.capabilities_id = canonical_fingerprint(
            {
                "kind": "vortex-diffusion-capabilities",
                "dimension": dimension_,
                "source_kinds": list(kinds),
                "required_source_fields": list(requirements),
                "domain": domain_,
                "precision": precision_.policy_id,
                "derivatives": list(derivatives_),
                "topology": topology_,
                "acceleration": acceleration_,
            }
        )

    @property
    def requires_core_radius(self) -> bool:
        return "core_radius" in self.required_source_fields

    @property
    def requires_volume(self) -> bool:
        return "volume" in self.required_source_fields


__all__ = ["VortexDiffusionCapabilities", "VortexVelocityCapabilities"]
