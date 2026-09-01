#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..contact import (
    AbstractNormalContactLaw,
    AcceptedContactState,
    ContactConfiguration,
    ContactEpochTransaction,
    ContactEvaluation,
    ContactQueryPlan,
    ContactStateTransaction,
    ContactSurface,
    CoulombContactLaw,
    FixedEpochContactOperator,
)
from ._geometry import SharpCrackTopology


class CrackFaceContactAdapter(StrictModule, NonTrainableState):
    """Map duplicated sharp-crack traces into the canonical fixed-epoch contact operator."""

    topology: SharpCrackTopology
    operator: FixedEpochContactOperator
    reference_coordinates: Array
    plus_node_ids: Array
    minus_node_ids: Array
    segment_ids: Array
    topology_id: str = eqx.field(static=True)
    mapping_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: SharpCrackTopology,
        normal_law: AbstractNormalContactLaw,
        /,
        *,
        friction_law: CoulombContactLaw | None = None,
        search_radius: float = math.inf,
        adapter_id: str = "sharp-crack-face-contact",
    ):
        if not isinstance(topology, SharpCrackTopology):
            raise TypeError("topology must be SharpCrackTopology.")
        if not isinstance(normal_law, AbstractNormalContactLaw):
            raise TypeError("normal_law must implement AbstractNormalContactLaw.")
        if friction_law is not None and not isinstance(friction_law, CoulombContactLaw):
            raise TypeError("friction_law must be CoulombContactLaw or None.")
        radius = float(search_radius)
        declared_id = str(adapter_id)
        if math.isnan(radius) or radius <= 0.0 or not declared_id:
            raise ValueError("Crack-face search radius and adapter_id are invalid.")
        geometry = topology.geometry
        coordinates = np.asarray(geometry.vertices)
        segments = np.asarray(geometry.segments, dtype=np.int32)
        segment_ids = np.asarray(geometry.segment_ids, dtype=np.int64)
        vertex_count = coordinates.shape[0]
        plus_node_ids = 2 * np.arange(vertex_count, dtype=np.int64)
        minus_node_ids = plus_node_ids + 1
        lengths = np.linalg.norm(
            coordinates[segments[:, 1]] - coordinates[segments[:, 0]], axis=1
        )
        nodal_weights = np.zeros((vertex_count,), dtype=coordinates.dtype)
        np.add.at(nodal_weights, segments[:, 0], 0.5 * lengths)
        np.add.at(nodal_weights, segments[:, 1], 0.5 * lengths)
        if np.any(nodal_weights <= 0.0):
            raise ValueError("Every crack-face vertex must carry positive trace measure.")
        plus_surface = ContactSurface(
            f"{geometry.crack_id}:plus",
            plus_node_ids,
            coordinates,
            segments,
            segment_ids,
            nodal_weights=nodal_weights,
        )
        # Contact normals point from the minus trace toward the plus trace, so the
        # contact facet orientation intentionally follows the crack plus normal.
        minus_surface = ContactSurface(
            f"{geometry.crack_id}:minus",
            minus_node_ids,
            coordinates,
            segments,
            segment_ids,
            nodal_weights=nodal_weights,
        )
        configuration = ContactConfiguration(
            plus_surface,
            minus_surface,
            epoch=topology.topology_version,
            search_radius=radius,
            self_contact=False,
        )
        query = ContactQueryPlan(configuration).execute()
        operator = FixedEpochContactOperator(
            query,
            normal_law,
            friction_law=friction_law,
        )
        self.topology = topology
        self.operator = operator
        self.reference_coordinates = jnp.asarray(coordinates)
        self.plus_node_ids = jnp.asarray(plus_node_ids)
        self.minus_node_ids = jnp.asarray(minus_node_ids)
        self.segment_ids = jnp.asarray(segment_ids)
        self.topology_id = topology.topology_id
        self.mapping_id = canonical_fingerprint(
            {
                "kind": "sharp-crack-face-contact-mapping",
                "declared_id": declared_id,
                "topology": topology.topology_id,
                "geometry": geometry.geometry_id,
                "configuration": configuration.configuration_id,
                "operator": operator.operator_id,
                "segment_ids": segment_ids.tolist(),
            }
        )

    @property
    def configuration(self) -> ContactConfiguration:
        return self.operator.query.configuration

    def accepted_state(
        self,
        previous: AcceptedContactState | None = None,
        /,
    ) -> AcceptedContactState:
        return self.operator.accepted_state(previous)

    def current_coordinates(
        self,
        plus_displacement: ArrayLike,
        minus_displacement: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        plus = jnp.asarray(plus_displacement)
        minus = jnp.asarray(minus_displacement)
        if (
            plus.shape != self.reference_coordinates.shape
            or minus.shape != self.reference_coordinates.shape
        ):
            raise ValueError(
                "Crack-face displacements must preserve the crack-vertex layout."
            )
        return self.reference_coordinates + plus, self.reference_coordinates + minus

    def evaluate(
        self,
        accepted: AcceptedContactState,
        plus_displacement: ArrayLike,
        minus_displacement: ArrayLike,
        /,
        *,
        normal_pressure: ArrayLike | None = None,
    ) -> ContactEvaluation:
        plus, minus = self.current_coordinates(plus_displacement, minus_displacement)
        return self.operator.evaluate(
            accepted,
            plus,
            minus,
            normal_pressure=normal_pressure,
        )

    def attempt(
        self,
        accepted: AcceptedContactState,
        plus_displacement: ArrayLike,
        minus_displacement: ArrayLike,
        /,
        *,
        normal_pressure: ArrayLike | None = None,
    ) -> ContactStateTransaction:
        plus, minus = self.current_coordinates(plus_displacement, minus_displacement)
        return self.operator.attempt(
            accepted,
            plus,
            minus,
            normal_pressure=normal_pressure,
        )

    def attempt_epoch(
        self,
        previous: AcceptedContactState,
        plus_displacement: ArrayLike,
        minus_displacement: ArrayLike,
        /,
        *,
        normal_pressure: ArrayLike | None = None,
    ) -> ContactEpochTransaction:
        """Transfer prior face history into this topology epoch with exact rollback."""

        plus, minus = self.current_coordinates(plus_displacement, minus_displacement)
        return self.operator.attempt_epoch(
            previous,
            plus,
            minus,
            normal_pressure=normal_pressure,
        )


__all__ = ["CrackFaceContactAdapter"]
