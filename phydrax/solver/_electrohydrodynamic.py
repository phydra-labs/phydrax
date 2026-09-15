#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._admissibility import AdmissibilityHeader, AdmissibilityReason
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import StructuredCochainBridge
from ..discretization.finite_volume._incompressible import (
    FaceVelocity,
    PreparedMACOperators,
)
from ..discretization.finite_volume._mac_electrochemical import mac_cell_to_faces
from ._mac_poisson_nernst_planck import MACPoissonNernstPlanckEvaluation
from ._poisson_nernst_planck import PoissonNernstPlanckEvaluation


class CochainElectrohydrodynamicEvaluation(StrictModule):
    edge_charge: Array
    integrated_edge_force: Array
    edge_force_components: tuple[Array, ...]
    integrated_force_components: tuple[Array, ...]
    edge_power: Array
    packed_power: Array
    power_defect: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class CochainElectrohydrodynamicForcePlan(StrictModule, NonTrainableState):
    """Cochain force diagnostic without claiming MAC face compatibility."""

    bridge: StructuredCochainBridge
    tail_indices: Array
    head_indices: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, bridge: StructuredCochainBridge, /) -> None:
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be StructuredCochainBridge.")
        incidence = bridge.cochain.topology.incidences[0]
        valid = np.asarray(incidence.relation.valid, dtype=bool)
        source = np.asarray(incidence.relation.source_indices)[valid]
        target = np.asarray(incidence.relation.target_indices)[valid]
        signs = np.asarray(incidence.signs)[valid]
        edge_count = bridge.cochain.cell_counts[1]
        tail = np.full(edge_count, -1, dtype=np.int32)
        head = np.full(edge_count, -1, dtype=np.int32)
        tail[target[signs < 0.0]] = source[signs < 0.0]
        head[target[signs > 0.0]] = source[signs > 0.0]
        if np.any(tail < 0) or np.any(head < 0):
            raise ValueError("Cochain force requires complete oriented edges.")
        self.bridge = bridge
        self.tail_indices = jnp.asarray(tail)
        self.head_indices = jnp.asarray(head)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cochain-electrohydrodynamic-force",
                "bridge": bridge.bridge_id,
                "tail": array_tree_fingerprint(tail),
                "head": array_tree_fingerprint(head),
            }
        )

    def evaluate(
        self,
        pnp: PoissonNernstPlanckEvaluation,
        /,
        *,
        edge_velocity: tuple[Array, ...] | None = None,
    ) -> CochainElectrohydrodynamicEvaluation:
        if not isinstance(pnp, PoissonNernstPlanckEvaluation):
            raise TypeError("pnp must be PoissonNernstPlanckEvaluation.")
        charge = pnp.electrochemical.charge_density
        edge_charge = 0.5 * (charge[self.tail_indices] + charge[self.head_indices])
        osmotic_gradient = self.bridge.cochain.exterior_derivative(
            0, pnp.electrochemical.osmotic_pressure
        )
        integrated_force = edge_charge * pnp.electrostatic.electric - osmotic_gradient
        integrated_components = self.bridge.unpack(1, integrated_force)
        measures = self.bridge.unpack(1, self.bridge.cochain.primal_measures[1])
        force_components = tuple(
            force / measure
            for force, measure in zip(integrated_components, measures, strict=True)
        )
        total_force = tuple(
            jnp.sum(force * measure)
            for force, measure in zip(force_components, measures, strict=True)
        )
        if edge_velocity is None:
            edge_power = jnp.asarray(0.0, dtype=integrated_force.dtype)
            packed_power = edge_power
        else:
            if len(edge_velocity) != len(force_components) or any(
                velocity.shape != force.shape
                for velocity, force in zip(edge_velocity, force_components, strict=True)
            ):
                raise ValueError("edge_velocity must match unpacked cochain edge axes.")
            edge_power = sum(
                jnp.sum(velocity * force * measure)
                for velocity, force, measure in zip(
                    edge_velocity, force_components, measures, strict=True
                )
            )
            packed_velocity = self.bridge.pack(1, edge_velocity)
            packed_power = jnp.sum(packed_velocity * integrated_force)
        defect = edge_power - packed_power
        scale = jnp.maximum(jnp.abs(edge_power), 1.0)
        successful = (
            pnp.successful
            & jnp.all(jnp.isfinite(integrated_force))
            & jnp.isfinite(defect)
            & (jnp.abs(defect) <= 256.0 * jnp.finfo(integrated_force.dtype).eps * scale)
        )
        reasons = jnp.where(
            successful,
            jnp.asarray(0, dtype=jnp.uint32),
            jnp.asarray(int(AdmissibilityReason.OUTSIDE_SUPPORT), dtype=jnp.uint32),
        )
        header = AdmissibilityHeader(
            jnp.where(successful, scale - jnp.abs(defect), -1.0),
            reasons,
            self.plan_id,
            canonical_fingerprint({"kind": "cochain-ehd-evidence", "plan": self.plan_id}),
        )
        return CochainElectrohydrodynamicEvaluation(
            edge_charge,
            integrated_force,
            force_components,
            total_force,
            edge_power,
            packed_power,
            defect,
            header,
            self.plan_id,
        )


class MACElectrohydrodynamicEvaluation(StrictModule):
    face_charge: FaceVelocity
    electric_force: FaceVelocity
    osmotic_force: FaceVelocity
    total_force: FaceVelocity
    integrated_force_components: tuple[Array, ...]
    fluid_power: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class MACElectrohydrodynamicForcePlan(StrictModule, NonTrainableState):
    """Native MAC force f = charge E - grad(osmotic pressure), evaluated once."""

    operators: PreparedMACOperators
    plan_id: str = eqx.field(static=True)

    def __init__(self, operators: PreparedMACOperators, /) -> None:
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        self.operators = operators
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-electrohydrodynamic-force",
                "operators": operators.prepared_id,
                "formula": "charge-times-electric-minus-osmotic-gradient",
            }
        )

    def evaluate(
        self,
        pnp: MACPoissonNernstPlanckEvaluation,
        /,
        *,
        face_velocity: FaceVelocity | None = None,
    ) -> MACElectrohydrodynamicEvaluation:
        if not isinstance(pnp, MACPoissonNernstPlanckEvaluation):
            raise TypeError("pnp must be MACPoissonNernstPlanckEvaluation.")
        if pnp.electrostatic.operators_id != self.operators.prepared_id:
            raise ValueError("MAC PNP and force plans use different operator layouts.")
        charge = pnp.electrochemical.charge_density
        osmotic = pnp.electrochemical.osmotic_pressure
        osmotic_gradient = self.operators.gradient(osmotic)
        face_charge = []
        electric_force = []
        osmotic_force = []
        total_force = []
        integrated_force = []
        for axis, (electric, osmotic_component, dual_measure) in enumerate(
            zip(
                pnp.electrostatic.electric_field,
                osmotic_gradient,
                self.operators.face_dual_measures,
                strict=True,
            )
        ):
            tail, head, _ = mac_cell_to_faces(self.operators, charge, axis)
            charge_face = 0.5 * (tail + head)
            electric_component = charge_face * electric
            osmotic_component_force = -osmotic_component
            total_component = electric_component + osmotic_component_force
            face_charge.append(charge_face)
            electric_force.append(electric_component)
            osmotic_force.append(osmotic_component_force)
            total_force.append(total_component)
            integrated_force.append(jnp.sum(dual_measure * total_component))
        if face_velocity is None:
            power = jnp.asarray(0.0, dtype=charge.dtype)
        else:
            velocity = self.operators.validate_velocity(face_velocity)
            power = sum(
                jnp.sum(measure * velocity_component * force_component)
                for measure, velocity_component, force_component in zip(
                    self.operators.face_dual_measures,
                    velocity,
                    total_force,
                    strict=True,
                )
            )
        finite = (
            pnp.header.globally_eligible
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in total_force))
            )
            & jnp.isfinite(power)
        )
        reasons = jnp.where(
            finite,
            jnp.asarray(0, dtype=jnp.uint32),
            jnp.asarray(int(AdmissibilityReason.NONFINITE), dtype=jnp.uint32),
        )
        header = AdmissibilityHeader(
            jnp.where(finite, 1.0, -1.0),
            reasons,
            self.plan_id,
            canonical_fingerprint({"kind": "mac-ehd-evidence", "plan": self.plan_id}),
        )
        return MACElectrohydrodynamicEvaluation(
            tuple(face_charge),
            tuple(electric_force),
            tuple(osmotic_force),
            tuple(total_force),
            tuple(integrated_force),
            power,
            header,
            self.plan_id,
        )


__all__ = [
    "CochainElectrohydrodynamicEvaluation",
    "CochainElectrohydrodynamicForcePlan",
    "MACElectrohydrodynamicEvaluation",
    "MACElectrohydrodynamicForcePlan",
]
