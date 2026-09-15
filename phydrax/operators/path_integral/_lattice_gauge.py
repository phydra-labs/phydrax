#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite compact U(1) Wilson measures on explicit oriented cell incidence."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._sampling import SingleCoordinateProposalPayload
from ..._strict import StrictModule
from ...discretization import CellComplexTopology
from ...metrix import FlatTorusStateGeometry
from ._lattice_action import AbstractIncrementalLatticeAction, LatticeActionEvidence


_TWO_PI = 2.0 * jnp.pi


def wrap_u1(angle: ArrayLike, /) -> Array:
    value = jnp.asarray(angle)
    return jnp.mod(value + jnp.pi, _TWO_PI) - jnp.pi


class CompactU1GaugeMeasure(AbstractIncrementalLatticeAction):
    """Finite compact U(1) Wilson measure on one canonical cell topology."""

    topology: CellComplexTopology
    edge_plaquettes: Array
    edge_signs: Array
    edge_valid: Array
    beta: Array
    topology_residual: Array
    valid: Array
    geometry: FlatTorusStateGeometry
    evidence: LatticeActionEvidence
    num_vertices: int = eqx.field(static=True)
    num_edges: int = eqx.field(static=True)
    num_plaquettes: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    configuration_shape: tuple[int, ...] = eqx.field(static=True)
    local_coordinate_shape: tuple[int, ...] = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    def __init__(
        self,
        topology: CellComplexTopology,
        /,
        *,
        beta: float,
    ):
        if not isinstance(topology, CellComplexTopology):
            raise TypeError("topology must be CellComplexTopology.")
        if topology.dimension < 2:
            raise ValueError("Compact U(1) gauge measures require 0-, 1-, and 2-cells.")
        beta_ = float(beta)
        if not np.isfinite(beta_) or beta_ < 0.0:
            raise ValueError("beta must be finite and non-negative.")
        num_vertices = topology.entities(0).count
        num_edges = topology.entities(1).count
        num_plaquettes = topology.entities(2).count
        incidence = topology.incidences[1]
        relation = incidence.relation
        route_valid = np.asarray(relation.valid, dtype=bool)
        source = np.asarray(relation.source_indices, dtype=np.int64)[route_valid]
        target = np.asarray(relation.target_indices, dtype=np.int64)[route_valid]
        signs = np.asarray(incidence.signs)[route_valid]
        counts = np.bincount(source, minlength=num_edges)
        capacity = max(1, int(counts.max(initial=0)))
        edge_plaquettes = np.zeros((num_edges, capacity), dtype=np.int32)
        edge_signs = np.zeros((num_edges, capacity), dtype=float)
        edge_valid = np.zeros((num_edges, capacity), dtype=bool)
        offsets = np.zeros((num_edges,), dtype=np.int32)
        for edge, plaquette, sign in zip(source, target, signs, strict=True):
            slot = offsets[edge]
            edge_plaquettes[edge, slot] = plaquette
            edge_signs[edge, slot] = sign
            edge_valid[edge, slot] = True
            offsets[edge] += 1
        action_id = canonical_fingerprint(
            {
                "kind": "compact-u1-gauge-measure",
                "topology": topology.topology_id,
                "beta": beta_,
                "action_convention": "reduced-wilson-minus-beta-cosine",
            }
        )
        field_space_id = canonical_fingerprint(
            {
                "kind": "compact-u1-edge-angle-space",
                "topology": topology.topology_id,
                "edges": topology.entities(1).entity_set_id,
            }
        )
        self.topology = topology
        self.edge_plaquettes = jnp.asarray(edge_plaquettes)
        self.edge_signs = jnp.asarray(edge_signs)
        self.edge_valid = jnp.asarray(edge_valid)
        self.beta = jnp.asarray(beta_)
        self.topology_residual = jnp.asarray(0.0)
        self.valid = jnp.asarray(True)
        self.geometry = FlatTorusStateGeometry(
            float(_TWO_PI), geometry_id=f"{action_id}:flat-torus"
        )
        self.evidence = LatticeActionEvidence(
            reference_measure="flat-torus",
            real_valued=True,
            bounded_below=True,
            normalizable=True,
            additive_constant=beta_ * num_plaquettes,
            evidence_id=f"{action_id}:measure-evidence",
        )
        self.num_vertices = num_vertices
        self.num_edges = num_edges
        self.num_plaquettes = num_plaquettes
        self.topology_id = topology.topology_id
        self.field_space_id = field_space_id
        self.configuration_shape = (num_edges,)
        self.local_coordinate_shape = (num_edges,)
        self.action_id = action_id
        self.claim = "finite-compact-u1-wilson-measure"

    def _links(self, link_angles: ArrayLike, /) -> Array:
        links = jnp.asarray(link_angles)
        if links.shape != self.configuration_shape:
            raise ValueError(f"link_angles must have shape {self.configuration_shape}.")
        if jnp.iscomplexobj(links):
            raise TypeError("Compact U(1) link angles must be real-valued.")
        return links

    def plaquette_angles(self, link_angles: ArrayLike, /) -> Array:
        links = self._links(link_angles)
        return wrap_u1(self.topology.incidences[1].exterior_derivative().mv(links))

    def action(self, link_angles: PyTree[Array], /) -> Array:
        return -self.beta * jnp.sum(jnp.cos(self.plaquette_angles(link_angles)))

    def canonical_action(self, link_angles: ArrayLike, /) -> Array:
        """Return the Wilson action with its conventional additive constant."""
        return self.action(link_angles) + self.evidence.additive_constant

    def gauge_transform(
        self, link_angles: ArrayLike, vertex_phases: ArrayLike, /
    ) -> Array:
        links = self._links(link_angles)
        phases = jnp.asarray(vertex_phases, dtype=links.dtype)
        if phases.shape != (self.num_vertices,):
            raise ValueError(f"vertex_phases must have shape ({self.num_vertices},).")
        derivative = self.topology.incidences[0].exterior_derivative().mv(phases)
        return wrap_u1(links + derivative)

    def local_delta_action(
        self,
        link_angles: ArrayLike,
        edge: ArrayLike,
        proposed_angle: ArrayLike,
        /,
    ) -> Array:
        links = self._links(link_angles)
        edge_ = jnp.asarray(edge, dtype=jnp.int32)
        proposed = wrap_u1(jnp.asarray(proposed_angle, dtype=links.dtype))
        if edge_.shape != () or proposed.shape != ():
            raise ValueError("local update requires scalar edge and proposed angle.")
        plaquettes = self.edge_plaquettes[edge_]
        signs = self.edge_signs[edge_]
        active = self.edge_valid[edge_]
        old_all = self.plaquette_angles(links)
        old = old_all[plaquettes]
        difference = wrap_u1(proposed - links[edge_])
        new = wrap_u1(old + signs * difference)
        return -self.beta * jnp.sum(jnp.where(active, jnp.cos(new) - jnp.cos(old), 0.0))

    def initialize_incremental(
        self, configuration: PyTree[Array], /
    ) -> tuple[Array, U1GaugeState]:
        links = wrap_u1(self._links(configuration))
        plaquettes = self.plaquette_angles(links)
        value = -self.beta * jnp.sum(jnp.cos(plaquettes))
        state = U1GaugeState(
            link_angles=links,
            plaquette_angles=plaquettes,
            action=value,
            valid=self.valid & jnp.all(jnp.isfinite(links)),
        )
        return value, state

    def propose_incremental(
        self,
        current_configuration: PyTree[Array],
        current_cache: PyTree[Array],
        proposed_configuration: PyTree[Array],
        payload: PyTree[Array],
        /,
    ) -> tuple[Array, U1GaugeState, Array]:
        if not isinstance(current_cache, U1GaugeState):
            raise TypeError("current_cache must be U1GaugeState.")
        if not isinstance(payload, SingleCoordinateProposalPayload):
            raise TypeError("payload must be SingleCoordinateProposalPayload.")
        current = self._links(current_configuration)
        proposed = wrap_u1(self._links(proposed_configuration))
        edge = jnp.asarray(payload.index, dtype=jnp.int32)
        expected = current.at[edge].set(proposed[edge])
        proposal_matches = jnp.all(
            jnp.isclose(
                expected,
                proposed,
                rtol=0.0,
                atol=8.0 * jnp.finfo(proposed.dtype).eps,
            )
        )
        plaquettes = self.edge_plaquettes[edge]
        signs = self.edge_signs[edge]
        active = self.edge_valid[edge]
        old = current_cache.plaquette_angles[plaquettes]
        displacement = wrap_u1(proposed[edge] - current[edge])
        new = wrap_u1(old + signs * displacement)
        updates = jnp.where(active, new - old, 0.0)
        candidate_plaquettes = current_cache.plaquette_angles.at[plaquettes].add(updates)
        delta = -self.beta * jnp.sum(jnp.where(active, jnp.cos(new) - jnp.cos(old), 0.0))
        candidate_action = current_cache.action + delta
        candidate = U1GaugeState(
            link_angles=proposed,
            plaquette_angles=candidate_plaquettes,
            action=candidate_action,
            valid=(
                current_cache.valid
                & proposal_matches
                & jnp.isfinite(delta)
                & jnp.all(jnp.isfinite(proposed))
            ),
        )
        return delta, candidate, candidate.valid

    def select_incremental(
        self,
        current: PyTree[Array],
        proposed: PyTree[Array],
        accepted: Array,
        /,
    ) -> PyTree[Array]:
        return jax.tree_util.tree_map(
            lambda current_value, proposed_value: jnp.where(
                accepted, proposed_value, current_value
            ),
            current,
            proposed,
        )

    def refresh_incremental(
        self, configuration: PyTree[Array], /
    ) -> tuple[Array, U1GaugeState]:
        return self.initialize_incremental(configuration)


class U1GaugeState(StrictModule):
    link_angles: Array
    plaquette_angles: Array
    action: Array
    valid: Array


def initialize_u1_gauge_state(
    measure: CompactU1GaugeMeasure, link_angles: ArrayLike, /
) -> U1GaugeState:
    if not isinstance(measure, CompactU1GaugeMeasure):
        raise TypeError("measure must be CompactU1GaugeMeasure.")
    _, state = measure.initialize_incremental(link_angles)
    return state


def wilson_loop(
    link_angles: ArrayLike, oriented_edge_coefficients: ArrayLike, /
) -> Array:
    links = jnp.asarray(link_angles)
    loop = jnp.asarray(oriented_edge_coefficients)
    if links.ndim != 1 or loop.shape != links.shape:
        raise ValueError("Wilson loop coefficients must match the rank-one link vector.")
    if not jnp.issubdtype(loop.dtype, jnp.integer):
        raise TypeError("Wilson loop coefficients must be integer oriented counts.")
    return jnp.exp(1j * jnp.sum(loop * links))


__all__ = [
    "CompactU1GaugeMeasure",
    "U1GaugeState",
    "initialize_u1_gauge_state",
    "wilson_loop",
    "wrap_u1",
]
