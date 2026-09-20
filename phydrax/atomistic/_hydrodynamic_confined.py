#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..discretization.finite_volume import (
    MACMarkerRouteState,
    PreparedMACMarkerTransfer,
)
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    FunctionLinearOperator,
    OperatorProperties,
)
from ._hydrodynamic_mobility import (
    _active_slots,
    AbstractHydrodynamicMobilityPlan,
    AbstractPreparedHydrodynamicMobility,
    materialize_mobility,
)
from ._system import PreparedAtomisticSystem


class ConfinedFIBMobilityEvidence(StrictModule):
    matrix: Array
    symmetry_residual: Array
    minimum_eigenvalue: Array
    relation_successful: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class ConfinedFIBMobilityPlan(AbstractHydrodynamicMobilityPlan):
    transfer: PreparedMACMarkerTransfer
    inverse_stokes: AbstractLinearOperator
    maximum_particles: int = eqx.field(static=True)
    symmetry_tolerance: float = eqx.field(static=True)
    mobility_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: PreparedMACMarkerTransfer,
        inverse_stokes: AbstractLinearOperator,
        /,
        *,
        maximum_particles: int,
        symmetry_tolerance: float = 1.0e-9,
    ):
        if not isinstance(transfer, PreparedMACMarkerTransfer):
            raise TypeError("transfer must be PreparedMACMarkerTransfer.")
        if not isinstance(inverse_stokes, AbstractLinearOperator):
            raise TypeError("inverse_stokes must be an AbstractLinearOperator.")
        if not inverse_stokes.source.compatible(
            transfer.operators.velocity_space
        ) or not (inverse_stokes.target.compatible(transfer.operators.velocity_space)):
            raise ValueError("inverse_stokes must act on the transfer velocity space.")
        if not inverse_stokes.properties.certifies("self_adjoint") or not (
            inverse_stokes.properties.certifies("positive_definite")
        ):
            raise ValueError("inverse_stokes must certify self-adjoint positivity.")
        maximum = int(maximum_particles)
        tolerance = float(symmetry_tolerance)
        if maximum <= 0 or not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Confined FIB capacities and tolerances must be positive.")
        if transfer.dimension != 3:
            raise ValueError(
                "The atomistic confined FIB adapter requires three dimensions."
            )
        self.transfer = transfer
        self.inverse_stokes = inverse_stokes
        self.maximum_particles = maximum
        self.symmetry_tolerance = tolerance
        self.mobility_id = canonical_fingerprint(
            {
                "kind": "confined-fib-mobility",
                "transfer": transfer.prepared_id,
                "inverse_stokes": inverse_stokes.operator_id,
                "maximum_particles": maximum,
                "symmetry_tolerance": tolerance,
            }
        )

    def prepare(
        self, system: PreparedAtomisticSystem, active_slots: ArrayLike, /
    ) -> PreparedConfinedFIBMobility:
        return PreparedConfinedFIBMobility(self, system, active_slots)


class PreparedConfinedFIBMobility(AbstractPreparedHydrodynamicMobility):
    plan: ConfinedFIBMobilityPlan
    system: PreparedAtomisticSystem
    active_slots: Array
    coordinate_space: ArraySpace
    marker_from_mobility: Array
    mobility_from_marker: Array
    inverse_weights: Array
    route_state: MACMarkerRouteState
    prepared_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ConfinedFIBMobilityPlan,
        system: PreparedAtomisticSystem,
        active_slots: ArrayLike,
        /,
    ):
        slots = _active_slots(plan.maximum_particles, system, active_slots)
        markers = plan.transfer.markers
        if markers.active_count != slots.size:
            raise ValueError(
                "Active atomistic particles and FIB markers must have equal counts."
            )
        particle_ids = np.asarray(system.plan.particle_ids)[np.asarray(slots)]
        marker_indices = np.asarray(markers.active_indices)
        marker_ids = np.asarray(markers.plan.marker_ids)[marker_indices]
        if set(particle_ids.tolist()) != set(marker_ids.tolist()):
            raise ValueError(
                "Active atomistic particle IDs and marker IDs must agree exactly."
            )
        particle_lookup = {
            int(identifier): index for index, identifier in enumerate(particle_ids)
        }
        marker_from = np.asarray(
            [particle_lookup[int(identifier)] for identifier in marker_ids],
            dtype=np.int32,
        )
        mobility_from = np.argsort(marker_from).astype(np.int32)
        weights = np.asarray(markers.plan.quadrature_weight)[marker_indices]
        if np.any(weights <= 0.0) or np.any(~np.isfinite(weights)):
            raise ValueError(
                "Active marker quadrature weights must be finite and positive."
            )
        reference_relation = plan.transfer.relation(markers.reference_position)
        if not bool(np.asarray(reference_relation.successful)):
            raise ValueError(
                "Reference marker positions do not define a certified confined FIB route."
            )
        route_state = plan.transfer.route_state(reference_relation)
        space = ArraySpace(
            (slots.size, 3),
            dtype=system.plan.coordinate_dtype,
            space_id=f"confined-fib:{plan.mobility_id}:coordinates",
        )
        self.plan = plan
        self.system = system
        self.active_slots = slots
        self.coordinate_space = space
        self.marker_from_mobility = jnp.asarray(marker_from)
        self.mobility_from_marker = jnp.asarray(mobility_from)
        self.inverse_weights = 1.0 / jnp.asarray(weights, dtype=space.dtype)
        self.route_state = route_state
        self.route_id = canonical_fingerprint(
            {
                "kind": "confined-fib-marker-route",
                "system": system.prepared_id,
                "transfer": plan.transfer.prepared_id,
                "active_slots": np.asarray(slots).tolist(),
                "marker_from_mobility": marker_from.tolist(),
                "route_state": array_tree_fingerprint(
                    (route_state.face_indices, route_state.valid)
                ),
            }
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-confined-fib-mobility",
                "plan": plan.mobility_id,
                "route": self.route_id,
            }
        )

    def _relation(self, positions: Array):
        markers = self.plan.transfer.markers
        full = markers.reference_position.at[markers.active_indices].set(
            positions[self.marker_from_mobility]
        )
        return self.plan.transfer.relation_on_routes(full, self.route_state)

    def configuration_valid(self, positions: ArrayLike, /) -> Array:
        value = self.coordinate_space.validate(jnp.asarray(positions))
        return self._relation(value).successful & jnp.all(jnp.isfinite(value))

    def operator(self, positions: ArrayLike, /) -> FunctionLinearOperator:
        value = self.coordinate_space.validate(jnp.asarray(positions))
        relation = self._relation(value)

        def action(vector):
            force = self.coordinate_space.validate(vector)
            marker_force_density = (
                force[self.marker_from_mobility] * self.inverse_weights[:, None]
            )
            fluid_force = self.plan.transfer.spread(relation, marker_force_density)
            fluid_velocity = self.plan.inverse_stokes.mv(fluid_force)
            marker_velocity = self.plan.transfer.gather(relation, fluid_velocity)
            result = marker_velocity[self.mobility_from_marker]
            return jnp.where(relation.successful, result, jnp.nan)

        return FunctionLinearOperator(
            action,
            source=self.coordinate_space,
            target=self.coordinate_space,
            transpose_action=action,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "asserted",
                },
            ),
            operator_id=f"{self.prepared_id}:operator",
        )

    def evaluate(
        self, positions: ArrayLike, /, *, maximum_dofs: int
    ) -> ConfinedFIBMobilityEvidence:
        value = self.coordinate_space.validate(jnp.asarray(positions))
        relation = self._relation(value)
        matrix = materialize_mobility(self, value, maximum_dofs=maximum_dofs)
        symmetry = jnp.max(jnp.abs(matrix - matrix.T))
        minimum = jnp.min(jnp.linalg.eigvalsh(0.5 * (matrix + matrix.T)))
        finite = jnp.all(jnp.isfinite(matrix))
        successful = (
            finite
            & relation.successful
            & (symmetry <= self.plan.symmetry_tolerance)
            & (minimum > 0.0)
        )
        return ConfinedFIBMobilityEvidence(
            matrix,
            symmetry,
            minimum,
            relation.successful,
            finite,
            successful,
            self.prepared_id,
        )


__all__ = [
    "ConfinedFIBMobilityEvidence",
    "ConfinedFIBMobilityPlan",
    "PreparedConfinedFIBMobility",
]
