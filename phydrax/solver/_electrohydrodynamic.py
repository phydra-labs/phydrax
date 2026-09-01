#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import StructuredCochainBridge
from ._poisson_nernst_planck import PoissonNernstPlanckEvaluation


class ElectrohydrodynamicCouplingEvaluation(StrictModule):
    edge_charge: Array
    integrated_edge_force: Array
    physical_face_force: tuple[Array, ...]
    total_force: tuple[Array, ...]
    fluid_power: Array
    cochain_power: Array
    power_defect: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class CochainMACTransferPlan(StrictModule, NonTrainableState):
    bridge: StructuredCochainBridge
    tail_indices: Array
    head_indices: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, bridge: StructuredCochainBridge, /):
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
            raise ValueError("Cochain-MAC transfer requires complete oriented edges.")
        self.bridge = bridge
        self.tail_indices = jnp.asarray(tail)
        self.head_indices = jnp.asarray(head)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cochain-mac-electrohydrodynamic-transfer",
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
        face_velocity: tuple[Array, ...] | None = None,
    ) -> ElectrohydrodynamicCouplingEvaluation:
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
        physical_force = tuple(
            force / measure
            for force, measure in zip(integrated_components, measures, strict=True)
        )
        total_force = tuple(
            jnp.sum(force * measure)
            for force, measure in zip(physical_force, measures, strict=True)
        )
        if face_velocity is None:
            fluid_power = jnp.asarray(0.0, dtype=integrated_force.dtype)
            cochain_power = fluid_power
        else:
            if len(face_velocity) != len(physical_force) or any(
                velocity.shape != force.shape
                for velocity, force in zip(face_velocity, physical_force, strict=True)
            ):
                raise ValueError("face_velocity must match unpacked cochain edge axes.")
            fluid_power = sum(
                jnp.sum(velocity * force * measure)
                for velocity, force, measure in zip(
                    face_velocity, physical_force, measures, strict=True
                )
            )
            packed_velocity = self.bridge.pack(1, face_velocity)
            cochain_power = jnp.sum(packed_velocity * integrated_force)
        power_defect = fluid_power - cochain_power
        scale = jnp.maximum(jnp.abs(fluid_power), 1.0)
        successful = (
            pnp.successful
            & jnp.all(jnp.isfinite(integrated_force))
            & jnp.isfinite(power_defect)
            & (
                jnp.abs(power_defect)
                <= 256.0 * jnp.finfo(integrated_force.dtype).eps * scale
            )
        )
        return ElectrohydrodynamicCouplingEvaluation(
            edge_charge,
            integrated_force,
            physical_force,
            total_force,
            fluid_power,
            cochain_power,
            power_defect,
            successful,
            self.plan_id,
        )


__all__ = [
    "CochainMACTransferPlan",
    "ElectrohydrodynamicCouplingEvaluation",
]
