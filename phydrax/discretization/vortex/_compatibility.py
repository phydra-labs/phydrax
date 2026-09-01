#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._capabilities import VortexDiffusionCapabilities, VortexVelocityCapabilities
from ._source import VortexSourceState, VortexTargetState


class VortexPropertyRequirements(StrictModule, NonTrainableState):
    """Capability-derived optional source properties needed by a compiled method."""

    core_radius: bool = eqx.field(static=True)
    volume: bool = eqx.field(static=True)
    requirements_id: str = eqx.field(static=True)

    def __init__(self, *, core_radius: bool = False, volume: bool = False):
        core, volume_ = bool(core_radius), bool(volume)
        self.core_radius = core
        self.volume = volume_
        self.requirements_id = canonical_fingerprint(
            {
                "kind": "vortex-property-requirements",
                "core_radius": core,
                "volume": volume_,
            }
        )


class VortexVelocityCompatibility(StrictModule, NonTrainableState):
    """Static proof that one prepared velocity shape satisfies its capabilities."""

    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    source_kind: str = eqx.field(static=True)
    target_topology: str = eqx.field(static=True)
    requested_fields: tuple[str, ...] = eqx.field(static=True)
    capabilities_id: str = eqx.field(static=True)
    compatibility_id: str = eqx.field(static=True)

    def __init__(
        self,
        capabilities: VortexVelocityCapabilities,
        /,
        *,
        source_capacity: int,
        target_capacity: int,
        source_kind: str,
        target_topology: str,
        requested_fields: tuple[str, ...] = ("velocity",),
    ):
        if not isinstance(capabilities, VortexVelocityCapabilities):
            raise TypeError("capabilities must be VortexVelocityCapabilities.")
        sources, targets = int(source_capacity), int(target_capacity)
        if sources <= 0 or targets <= 0:
            raise ValueError("Prepared vortex capacities must be positive.")
        kind = str(source_kind).strip()
        topology = str(target_topology).strip()
        fields = tuple(str(value).strip() for value in requested_fields)
        if kind not in capabilities.source_kinds:
            raise ValueError(
                f"Vortex source kind '{kind}' is unsupported by this backend."
            )
        if topology not in capabilities.target_topologies:
            raise ValueError(
                f"Vortex target topology '{topology}' is unsupported by this backend."
            )
        if not fields or any(not field for field in fields):
            raise ValueError("requested_fields must contain non-empty values.")
        if len(set(fields)) != len(fields):
            raise ValueError("requested_fields cannot contain duplicates.")
        unsupported = tuple(
            field for field in fields if field not in capabilities.supported_fields
        )
        if unsupported:
            raise ValueError(f"Unsupported vortex fields requested: {unsupported}.")
        self.source_capacity = sources
        self.target_capacity = targets
        self.source_kind = kind
        self.target_topology = topology
        self.requested_fields = fields
        self.capabilities_id = capabilities.capabilities_id
        self.compatibility_id = canonical_fingerprint(
            {
                "kind": "vortex-velocity-compatibility",
                "capabilities": capabilities.capabilities_id,
                "source_capacity": sources,
                "target_capacity": targets,
                "source_kind": kind,
                "target_topology": topology,
                "requested_fields": list(fields),
            }
        )


def vortex_property_requirements(
    velocity: VortexVelocityCapabilities,
    diffusion: VortexDiffusionCapabilities,
    /,
) -> VortexPropertyRequirements:
    if not isinstance(velocity, VortexVelocityCapabilities):
        raise TypeError("velocity must be VortexVelocityCapabilities.")
    if not isinstance(diffusion, VortexDiffusionCapabilities):
        raise TypeError("diffusion must be VortexDiffusionCapabilities.")
    return VortexPropertyRequirements(
        core_radius=velocity.requires_core_radius or diffusion.requires_core_radius,
        volume=velocity.requires_volume or diffusion.requires_volume,
    )


def request_fields(request: object, /) -> tuple[str, ...]:
    from ._interfaces import VortexFieldRequest

    if not isinstance(request, VortexFieldRequest):
        raise TypeError("request must be a VortexFieldRequest.")
    fields: list[str] = []
    if request.velocity:
        fields.append("velocity")
    if request.velocity_gradient:
        fields.append("velocity_gradient")
    if request.vorticity:
        fields.append("vorticity")
    return tuple(fields)


def validate_vortex_velocity_evaluation(
    capabilities: VortexVelocityCapabilities,
    compatibility: VortexVelocityCompatibility,
    source: VortexSourceState,
    target: VortexTargetState,
    request: object,
    /,
) -> tuple[VortexSourceState, VortexTargetState]:
    """Validate the static contract and retain the source-index bounds check."""

    if not isinstance(capabilities, VortexVelocityCapabilities):
        raise TypeError("capabilities must be VortexVelocityCapabilities.")
    if not isinstance(compatibility, VortexVelocityCompatibility):
        raise TypeError("compatibility must be VortexVelocityCompatibility.")
    if not isinstance(source, VortexSourceState):
        raise TypeError("source must be a VortexSourceState.")
    if not isinstance(target, VortexTargetState):
        raise TypeError("target must be a VortexTargetState.")
    if compatibility.capabilities_id != capabilities.capabilities_id:
        raise ValueError("Prepared vortex compatibility belongs to another backend.")
    if (
        source.dimension != capabilities.dimension
        or target.dimension != capabilities.dimension
    ):
        raise ValueError("Vortex source/target dimensions do not match the backend.")
    if source.capacity != compatibility.source_capacity:
        raise ValueError("Vortex source capacity does not match the prepared backend.")
    if target.capacity != compatibility.target_capacity:
        raise ValueError("Vortex target capacity does not match the prepared backend.")
    if source.source_kind != compatibility.source_kind:
        raise ValueError("Vortex source kind does not match prepared compatibility.")
    fields = request_fields(request)
    unsupported = tuple(
        field for field in fields if field not in capabilities.supported_fields
    )
    if unsupported:
        raise ValueError(f"Unsupported vortex fields requested: {unsupported}.")
    if capabilities.requires_core_radius and source.core_radius is None:
        raise ValueError("The vortex velocity backend requires source core_radius.")
    if capabilities.requires_volume and source.volume is None:
        raise ValueError("The vortex velocity backend requires source volume.")
    if compatibility.target_topology == "same-support" and target.source_indices is None:
        raise ValueError(
            "Same-support vortex targets require explicit source-index identity."
        )
    if target.source_indices is not None:
        validated = eqx.error_if(
            target.source_indices,
            jnp.any(target.source_indices >= source.capacity),
            "Vortex target source_indices exceed the source capacity.",
        )
        target = eqx.tree_at(lambda state: state.source_indices, target, validated)
    capabilities.precision.validate_coordinates(source.positions)
    capabilities.precision.validate_coordinates(target.positions)
    return source, target


__all__ = [
    "VortexPropertyRequirements",
    "VortexVelocityCompatibility",
    "request_fields",
    "validate_vortex_velocity_evaluation",
    "vortex_property_requirements",
]
