#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import IntFlag

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._kinematics import (
    ContactKinematicsBatch,
    ContactKinematicsEpoch,
)
from ._materials import ContactMaterialPairTable, ContactPairParameters
from ._route_state import ContactRouteMode, ContactRouteState


class ContactClosureCapability(IntFlag):
    NONE = 0
    POTENTIAL = 1 << 0
    RESIDUAL = 1 << 1
    CONE = 1 << 2
    STATEFUL = 1 << 3
    TRANSPORT = 1 << 4
    DIFFERENTIABLE = 1 << 5
    ROLLING = 1 << 6
    ADHESION = 1 << 7
    WEAR = 1 << 8
    LUBRICATION = 1 << 9


class NormalContactResponse(StrictModule):
    traction: Array
    potential_density: Array
    tangent_stiffness: Array
    active: Array
    admissible: Array
    finite: Array


class TangentialContactResponse(StrictModule):
    traction: Array
    potential_density: Array
    dissipated_power: Array
    stick: Array
    slip: Array
    cone_defect: Array
    finite: Array


class ContactEvolutionResponse(StrictModule):
    mode: Array
    accumulated_slip: Array
    adhesion_damage: Array
    wear_depth: Array
    rate_state: Array
    film_thickness: Array
    finite: Array


class ContactTransportResponse(StrictModule):
    heat_flux: Array
    electrical_current: Array
    mass_flux: Array
    frictional_heat: Array
    finite: Array


class AbstractNormalContactLaw(StrictModule, NonTrainableState):
    @property
    @abc.abstractmethod
    def law_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> ContactClosureCapability:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self,
        kinematics: ContactKinematicsBatch,
        parameters: ContactPairParameters,
        state: ContactRouteState,
        /,
    ) -> NormalContactResponse:
        raise NotImplementedError


class AbstractTangentialContactLaw(StrictModule, NonTrainableState):
    @property
    @abc.abstractmethod
    def law_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> ContactClosureCapability:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self,
        kinematics: ContactKinematicsBatch,
        parameters: ContactPairParameters,
        normal: NormalContactResponse,
        state: ContactRouteState,
        /,
    ) -> TangentialContactResponse:
        raise NotImplementedError


class AbstractInterfaceEvolutionLaw(StrictModule, NonTrainableState):
    @property
    @abc.abstractmethod
    def law_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> ContactClosureCapability:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self,
        kinematics: ContactKinematicsBatch,
        parameters: ContactPairParameters,
        normal: NormalContactResponse,
        tangential: TangentialContactResponse,
        state: ContactRouteState,
        /,
    ) -> ContactEvolutionResponse:
        raise NotImplementedError


class AbstractContactTransportLaw(StrictModule, NonTrainableState):
    @property
    @abc.abstractmethod
    def law_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> ContactClosureCapability:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self,
        kinematics: ContactKinematicsBatch,
        parameters: ContactPairParameters,
        normal: NormalContactResponse,
        tangential: TangentialContactResponse,
        state: ContactRouteState,
        driving_jump: Array,
        /,
    ) -> ContactTransportResponse:
        raise NotImplementedError


class FrictionlessTangentialLaw(AbstractTangentialContactLaw):
    _law_id: str = eqx.field(static=True)

    def __init__(self):
        self._law_id = "frictionless-tangential-contact"

    @property
    def law_id(self) -> str:
        return self._law_id

    @property
    def capabilities(self) -> ContactClosureCapability:
        return (
            ContactClosureCapability.POTENTIAL
            | ContactClosureCapability.RESIDUAL
            | ContactClosureCapability.DIFFERENTIABLE
        )

    def evaluate(
        self,
        kinematics: ContactKinematicsBatch,
        parameters: ContactPairParameters,
        normal: NormalContactResponse,
        state: ContactRouteState,
        /,
    ) -> TangentialContactResponse:
        del parameters, normal, state
        shape = kinematics.tangential_velocity.shape
        zero_vector = jnp.zeros(shape, dtype=kinematics.gap.dtype)
        zero = jnp.zeros(kinematics.gap.shape, dtype=kinematics.gap.dtype)
        return TangentialContactResponse(
            zero_vector,
            zero,
            zero,
            kinematics.valid,
            jnp.zeros_like(kinematics.valid),
            zero,
            jnp.asarray(True),
        )


class IdentityInterfaceEvolution(AbstractInterfaceEvolutionLaw):
    _law_id: str = eqx.field(static=True)

    def __init__(self):
        self._law_id = "identity-contact-evolution"

    @property
    def law_id(self) -> str:
        return self._law_id

    @property
    def capabilities(self) -> ContactClosureCapability:
        return ContactClosureCapability.STATEFUL

    def evaluate(
        self,
        kinematics: ContactKinematicsBatch,
        parameters: ContactPairParameters,
        normal: NormalContactResponse,
        tangential: TangentialContactResponse,
        state: ContactRouteState,
        /,
    ) -> ContactEvolutionResponse:
        del parameters, normal
        mode = jnp.where(
            tangential.slip,
            int(ContactRouteMode.SLIP),
            jnp.where(
                kinematics.valid,
                int(ContactRouteMode.STICK),
                int(ContactRouteMode.OPEN),
            ),
        ).astype(jnp.int32)
        return ContactEvolutionResponse(
            mode,
            state.accumulated_slip,
            state.adhesion_damage,
            state.wear_depth,
            state.rate_state,
            state.film_thickness,
            jnp.asarray(True),
        )


class NoContactTransport(AbstractContactTransportLaw):
    _law_id: str = eqx.field(static=True)

    def __init__(self):
        self._law_id = "no-contact-transport"

    @property
    def law_id(self) -> str:
        return self._law_id

    @property
    def capabilities(self) -> ContactClosureCapability:
        return ContactClosureCapability.NONE

    def evaluate(
        self,
        kinematics: ContactKinematicsBatch,
        parameters: ContactPairParameters,
        normal: NormalContactResponse,
        tangential: TangentialContactResponse,
        state: ContactRouteState,
        driving_jump: Array,
        /,
    ) -> ContactTransportResponse:
        del parameters, normal, tangential, state, driving_jump
        zero = jnp.zeros(kinematics.gap.shape, dtype=kinematics.gap.dtype)
        return ContactTransportResponse(
            zero,
            zero,
            zero,
            zero,
            jnp.asarray(True),
        )


class ContactClosurePlan(StrictModule, NonTrainableState):
    normal: AbstractNormalContactLaw
    tangential: AbstractTangentialContactLaw
    evolution: AbstractInterfaceEvolutionLaw
    transport: AbstractContactTransportLaw
    material_table: ContactMaterialPairTable
    capabilities: ContactClosureCapability = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)

    def __init__(
        self,
        normal: AbstractNormalContactLaw,
        material_table: ContactMaterialPairTable,
        /,
        *,
        tangential: AbstractTangentialContactLaw | None = None,
        evolution: AbstractInterfaceEvolutionLaw | None = None,
        transport: AbstractContactTransportLaw | None = None,
    ):
        if not isinstance(normal, AbstractNormalContactLaw):
            raise TypeError("normal must be a contact normal law.")
        if not isinstance(material_table, ContactMaterialPairTable):
            raise TypeError("material_table must be ContactMaterialPairTable.")
        tangential_ = FrictionlessTangentialLaw() if tangential is None else tangential
        evolution_ = IdentityInterfaceEvolution() if evolution is None else evolution
        transport_ = NoContactTransport() if transport is None else transport
        if not isinstance(tangential_, AbstractTangentialContactLaw):
            raise TypeError("tangential must be a tangential contact law.")
        if not isinstance(evolution_, AbstractInterfaceEvolutionLaw):
            raise TypeError("evolution must be an interface evolution law.")
        if not isinstance(transport_, AbstractContactTransportLaw):
            raise TypeError("transport must be a contact transport law.")
        capabilities = (
            normal.capabilities
            | tangential_.capabilities
            | evolution_.capabilities
            | transport_.capabilities
        )
        self.normal = normal
        self.tangential = tangential_
        self.evolution = evolution_
        self.transport = transport_
        self.material_table = material_table
        self.capabilities = capabilities
        self.closure_id = canonical_fingerprint(
            {
                "kind": "contact-closure-plan",
                "normal": normal.law_id,
                "tangential": tangential_.law_id,
                "evolution": evolution_.law_id,
                "transport": transport_.law_id,
                "materials": material_table.table_id,
                "capabilities": int(capabilities),
            }
        )


class ContactClosureBatchResponse(StrictModule):
    normal: NormalContactResponse
    tangential: TangentialContactResponse
    evolution: ContactEvolutionResponse
    transport: ContactTransportResponse
    total_potential: Array
    total_dissipated_power: Array
    finite: Array
    successful: Array
    batch_id: str = eqx.field(static=True)


class ContactClosureEvidence(StrictModule):
    total_potential: Array
    total_dissipated_power: Array
    maximum_cone_defect: Array
    active_contacts: Array
    finite: Array
    material_complete: Array
    successful: Array
    closure_id: str = eqx.field(static=True)


class ContactClosureEvaluation(StrictModule):
    batches: tuple[ContactClosureBatchResponse, ...]
    candidate_state: ContactRouteState
    evidence: ContactClosureEvidence
    evaluation_id: str = eqx.field(static=True)


def _slice_route_state(
    state: ContactRouteState, start: int, stop: int, /
) -> ContactRouteState:
    return ContactRouteState(
        state.route_keys[start:stop],
        state.valid[start:stop],
        state.mode[start:stop],
        state.accumulated_slip[start:stop],
        state.adhesion_damage[start:stop],
        state.wear_depth[start:stop],
        state.rate_state[start:stop],
        state.film_thickness[start:stop],
        state.state_version,
        state.tangent_dimension,
        state.closure_id,
    )


def evaluate_contact_closure(
    plan: ContactClosurePlan,
    kinematics: ContactKinematicsEpoch,
    state: ContactRouteState,
    /,
    *,
    driving_jump: ArrayLike | None = None,
) -> ContactClosureEvaluation:
    if not isinstance(plan, ContactClosurePlan):
        raise TypeError("plan must be ContactClosurePlan.")
    if not isinstance(kinematics, ContactKinematicsEpoch):
        raise TypeError("kinematics must be ContactKinematicsEpoch.")
    if not isinstance(state, ContactRouteState) or state.closure_id != plan.closure_id:
        raise ValueError("Contact route state belongs to another closure.")
    capacity = sum(batch.capacity for batch in kinematics.batches)
    if state.capacity != capacity:
        raise ValueError("Contact route state capacity does not match kinematics.")
    tangent_dimension = (
        kinematics.batches[0].tangential_velocity.shape[-1]
        if kinematics.batches
        else state.tangent_dimension
    )
    jump = (
        jnp.zeros((capacity, 3), dtype=state.accumulated_slip.dtype)
        if driving_jump is None
        else jnp.asarray(driving_jump, dtype=state.accumulated_slip.dtype)
    )
    if jump.shape[0] != capacity:
        raise ValueError("Contact transport driving jump has invalid capacity.")
    responses = []
    modes = []
    accumulated_slip = []
    adhesion_damage = []
    wear_depth = []
    rate_state = []
    film_thickness = []
    material_complete_values = []
    offset = 0
    for batch in kinematics.batches:
        stop = offset + batch.capacity
        local_state = _slice_route_state(state, offset, stop)
        parameters = plan.material_table.lookup(
            batch.left_material_ids, batch.right_material_ids
        )
        normal = plan.normal.evaluate(batch, parameters, local_state)
        tangential = plan.tangential.evaluate(batch, parameters, normal, local_state)
        evolution = plan.evolution.evaluate(
            batch, parameters, normal, tangential, local_state
        )
        transport = plan.transport.evaluate(
            batch,
            parameters,
            normal,
            tangential,
            local_state,
            jump[offset:stop],
        )
        potential = jnp.sum(
            jnp.where(
                batch.valid,
                normal.potential_density + tangential.potential_density,
                0.0,
            )
            * batch.quadrature_weight
        )
        dissipation = jnp.sum(
            jnp.where(batch.valid, tangential.dissipated_power, 0.0)
            * batch.quadrature_weight
        )
        finite = (
            normal.finite
            & tangential.finite
            & evolution.finite
            & transport.finite
            & jnp.isfinite(potential)
            & jnp.isfinite(dissipation)
        )
        successful = (
            finite
            & jnp.all((~batch.valid) | normal.admissible)
            & jnp.all((~batch.valid) | parameters.mechanical_available)
        )
        responses.append(
            ContactClosureBatchResponse(
                normal,
                tangential,
                evolution,
                transport,
                potential,
                dissipation,
                finite,
                successful,
                batch.batch_id,
            )
        )
        modes.append(evolution.mode)
        accumulated_slip.append(evolution.accumulated_slip)
        adhesion_damage.append(evolution.adhesion_damage)
        wear_depth.append(evolution.wear_depth)
        rate_state.append(evolution.rate_state)
        film_thickness.append(evolution.film_thickness)
        material_complete_values.append(
            jnp.all((~batch.valid) | parameters.mechanical_available)
        )
        offset = stop
    if responses:
        total_potential = sum(
            (value.total_potential for value in responses),
            start=jnp.asarray(0.0, dtype=state.accumulated_slip.dtype),
        )
        total_dissipation = sum(
            (value.total_dissipated_power for value in responses),
            start=jnp.asarray(0.0, dtype=state.accumulated_slip.dtype),
        )
        maximum_cone_defect = jnp.max(
            jnp.concatenate(tuple(value.tangential.cone_defect for value in responses)),
            initial=0.0,
        )
        finite = jnp.all(jnp.stack(tuple(value.finite for value in responses)))
        material_complete = jnp.all(jnp.stack(tuple(material_complete_values)))
        active = sum(
            (jnp.sum(batch.valid, dtype=jnp.int32) for batch in kinematics.batches),
            start=jnp.asarray(0, dtype=jnp.int32),
        )
        candidate = ContactRouteState(
            state.route_keys,
            state.valid,
            jnp.concatenate(tuple(modes)),
            jnp.concatenate(tuple(accumulated_slip)),
            jnp.concatenate(tuple(adhesion_damage)),
            jnp.concatenate(tuple(wear_depth)),
            jnp.concatenate(tuple(rate_state)),
            jnp.concatenate(tuple(film_thickness)),
            state.state_version + 1,
            tangent_dimension,
            plan.closure_id,
        )
    else:
        total_potential = jnp.asarray(0.0, dtype=state.accumulated_slip.dtype)
        total_dissipation = jnp.asarray(0.0, dtype=state.accumulated_slip.dtype)
        maximum_cone_defect = jnp.asarray(0.0, dtype=state.accumulated_slip.dtype)
        finite = jnp.asarray(True)
        material_complete = jnp.asarray(True)
        active = jnp.asarray(0, dtype=jnp.int32)
        candidate = state
    successful = (
        kinematics.evidence.successful
        & finite
        & material_complete
        & (total_dissipation >= 0.0)
    )
    evidence = ContactClosureEvidence(
        total_potential,
        total_dissipation,
        maximum_cone_defect,
        active,
        finite,
        material_complete,
        successful,
        plan.closure_id,
    )
    return ContactClosureEvaluation(
        tuple(responses),
        candidate,
        evidence,
        canonical_fingerprint(
            {
                "kind": "contact-closure-evaluation",
                "closure": plan.closure_id,
                "kinematics": kinematics.epoch_id,
            }
        ),
    )


__all__ = [
    "AbstractContactTransportLaw",
    "AbstractInterfaceEvolutionLaw",
    "AbstractNormalContactLaw",
    "AbstractTangentialContactLaw",
    "ContactClosureBatchResponse",
    "ContactClosureCapability",
    "ContactClosureEvaluation",
    "ContactClosureEvidence",
    "ContactClosurePlan",
    "ContactEvolutionResponse",
    "ContactTransportResponse",
    "FrictionlessTangentialLaw",
    "IdentityInterfaceEvolution",
    "NoContactTransport",
    "NormalContactResponse",
    "TangentialContactResponse",
    "evaluate_contact_closure",
]
