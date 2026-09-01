#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ._capabilities import VortexDiffusionCapabilities, VortexVelocityCapabilities
from ._compatibility import VortexVelocityCompatibility
from ._source import VortexSourceState, VortexTargetState


class VortexFieldRequest(StrictModule, NonTrainableState):
    """Static selection of fields produced by a vortex velocity backend."""

    velocity: bool = eqx.field(static=True)
    velocity_gradient: bool = eqx.field(static=True)
    vorticity: bool = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        velocity: bool = True,
        velocity_gradient: bool = False,
        vorticity: bool = False,
    ):
        velocity_ = bool(velocity)
        gradient_ = bool(velocity_gradient)
        vorticity_ = bool(vorticity)
        if not (velocity_ or gradient_ or vorticity_):
            raise ValueError("A vortex field request must select at least one field.")
        self.velocity = velocity_
        self.velocity_gradient = gradient_
        self.vorticity = vorticity_
        self.request_id = canonical_fingerprint(
            {
                "kind": "vortex-field-request",
                "velocity": velocity_,
                "velocity_gradient": gradient_,
                "vorticity": vorticity_,
            }
        )


DEFAULT_VORTEX_FIELD_REQUEST = VortexFieldRequest()


class VortexVelocityDiagnostics(StrictModule):
    """Backend-independent execution evidence for one field evaluation."""

    source_count: Array
    target_count: Array
    active_interaction_count: Array
    excluded_interaction_count: Array
    coincident_distinct_count: Array
    minimum_core_radius: Array
    inputs_finite: Array
    outputs_finite: Array
    resource_budget_satisfied: Array
    successful: Array
    backend_diagnostics: Any


class VortexVelocityEvaluation(StrictModule):
    """Requested vortex fields together with immutable provenance."""

    velocity: Array | None
    velocity_gradient: Array | None
    vorticity: Array | None
    successful: Array
    backend_id: str = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)
    diagnostics: VortexVelocityDiagnostics


class AbstractVortexVelocityPlan(StrictModule, NonTrainableState):
    """Resource-bounded plan for a dimension-specific vortex field backend."""

    dimension: AbstractAttribute[int]
    plan_id: AbstractAttribute[str]
    capabilities: AbstractAttribute[VortexVelocityCapabilities]

    @abc.abstractmethod
    def prepare(
        self,
        /,
        *,
        source_capacity: int,
        target_capacity: int | None = None,
        source_kind: str = "particle",
        target_topology: str = "same-support",
        request: VortexFieldRequest = DEFAULT_VORTEX_FIELD_REQUEST,
    ) -> AbstractPreparedVortexVelocity:
        """Bind capacities, source kind, target topology, and requested fields."""


class AbstractPreparedVortexVelocity(StrictModule, NonTrainableState):
    """Fixed-shape, JAX-transformable vortex field evaluation."""

    dimension: AbstractAttribute[int]
    source_capacity: AbstractAttribute[int]
    target_capacity: AbstractAttribute[int]
    backend_id: AbstractAttribute[str]
    prepared_id: AbstractAttribute[str]
    capabilities: AbstractAttribute[VortexVelocityCapabilities]
    compatibility: AbstractAttribute[VortexVelocityCompatibility]

    @abc.abstractmethod
    def evaluate(
        self,
        source: VortexSourceState,
        target: VortexTargetState,
        /,
        *,
        request: VortexFieldRequest = DEFAULT_VORTEX_FIELD_REQUEST,
    ) -> VortexVelocityEvaluation:
        """Evaluate requested fields, excluding only explicitly identified selves."""


class VortexDiffusionDiagnostics(StrictModule):
    """Backend-independent evidence for one particle diffusion evaluation."""

    particle_count: Array
    active_interaction_count: Array
    total_rate: Array
    inputs_finite: Array
    outputs_finite: Array
    resource_budget_satisfied: Array
    conservative: Array
    successful: Array
    backend_diagnostics: Any


class VortexDiffusionEvaluation(StrictModule):
    """Particle-strength diffusion rate and provenance."""

    rate: Array
    successful: Array
    backend_id: str = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)
    diagnostics: VortexDiffusionDiagnostics


class AbstractVortexDiffusionPlan(StrictModule, NonTrainableState):
    """Resource-bounded plan for a dimension-specific diffusion backend."""

    dimension: AbstractAttribute[int]
    plan_id: AbstractAttribute[str]
    capabilities: AbstractAttribute[VortexDiffusionCapabilities]

    @abc.abstractmethod
    def prepare(
        self,
        /,
        *,
        capacity: int,
        dimension: int,
    ) -> AbstractPreparedVortexDiffusion:
        """Bind an exact particle capacity after checking dimension and budgets."""


class AbstractPreparedVortexDiffusion(StrictModule, NonTrainableState):
    """Fixed-shape, JAX-transformable particle diffusion evaluation."""

    dimension: AbstractAttribute[int]
    capacity: AbstractAttribute[int]
    backend_id: AbstractAttribute[str]
    prepared_id: AbstractAttribute[str]
    capabilities: AbstractAttribute[VortexDiffusionCapabilities]

    @abc.abstractmethod
    def evaluate(
        self,
        source: VortexSourceState,
        viscosity: ArrayLike,
        /,
    ) -> VortexDiffusionEvaluation:
        """Return the conservative strength rate for the canonical source state."""


__all__ = [
    "AbstractPreparedVortexDiffusion",
    "AbstractPreparedVortexVelocity",
    "AbstractVortexDiffusionPlan",
    "AbstractVortexVelocityPlan",
    "DEFAULT_VORTEX_FIELD_REQUEST",
    "VortexDiffusionDiagnostics",
    "VortexDiffusionEvaluation",
    "VortexFieldRequest",
    "VortexVelocityDiagnostics",
    "VortexVelocityEvaluation",
]
