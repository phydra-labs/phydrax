#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._sampling import SingleCoordinateProposalPayload
from ..._strict import StrictModule
from ...discretization import CochainDiscretization
from ...metrix import EuclideanStateGeometry
from ._lattice_action import (
    AbstractIncrementalLatticeAction,
    AbstractLatticeEuclideanAction,
    LatticeActionEvidence,
)


class Phi4ActionCache(StrictModule):
    """Exact dynamic contributions for one scalar-lattice action state."""

    edge_differences: Array
    site_contributions: Array
    action: Array


class Phi4LatticeAction(AbstractLatticeEuclideanAction):
    """Real scalar Euclidean ``phi^4`` action on degree-zero cochains."""

    discretization: CochainDiscretization
    kinetic_scale: Array
    mass_squared: Array
    quartic_coupling: Array
    topology_id: str = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    configuration_shape: tuple[int, ...] = eqx.field(static=True)
    local_coordinate_shape: tuple[int, ...] = eqx.field(static=True)
    geometry: EuclideanStateGeometry
    evidence: LatticeActionEvidence
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: CochainDiscretization,
        /,
        *,
        kinetic_scale: float = 1.0,
        mass_squared: float = 1.0,
        quartic_coupling: float = 1.0,
    ):
        if not isinstance(discretization, CochainDiscretization):
            raise TypeError("discretization must be CochainDiscretization.")
        if discretization.max_degree < 1:
            raise ValueError("A scalar lattice action requires degree-one cells.")
        kinetic = float(kinetic_scale)
        mass = float(mass_squared)
        quartic = float(quartic_coupling)
        if not all(np.isfinite(value) for value in (kinetic, mass, quartic)):
            raise ValueError("Scalar lattice couplings must be finite.")
        if kinetic <= 0.0:
            raise ValueError("kinetic_scale must be positive.")
        if quartic < 0.0:
            raise ValueError("quartic_coupling must be non-negative.")
        dtype = jnp.result_type(discretization.hodge_stars[0].dtype, jnp.float32)
        topology_id = discretization.topology.topology_id
        field_space_id = discretization.field_spaces[0].field_space_id
        configuration_shape = (discretization.cell_counts[0],)
        action_id = canonical_fingerprint(
            {
                "kind": "phi4-lattice-action",
                "prepared_discretization": discretization.prepared_id,
                "kinetic_scale": kinetic,
                "mass_squared": mass,
                "quartic_coupling": quartic,
                "convention": "kinetic-half-mass-half-quartic-quarter",
            }
        )
        normalizable = quartic > 0.0 or (quartic == 0.0 and mass > 0.0)
        bounded_below = normalizable
        self.discretization = discretization
        self.kinetic_scale = jnp.asarray(kinetic, dtype=dtype)
        self.mass_squared = jnp.asarray(mass, dtype=dtype)
        self.quartic_coupling = jnp.asarray(quartic, dtype=dtype)
        self.topology_id = topology_id
        self.field_space_id = field_space_id
        self.configuration_shape = configuration_shape
        self.local_coordinate_shape = configuration_shape
        self.geometry = EuclideanStateGeometry(
            geometry_id=f"{action_id}:euclidean-geometry"
        )
        self.evidence = LatticeActionEvidence(
            reference_measure="lebesgue",
            real_valued=True,
            bounded_below=bounded_below,
            normalizable=normalizable,
            additive_constant=0.0,
            evidence_id=f"{action_id}:measure-evidence",
        )
        self.action_id = action_id

    def _configuration(self, configuration: ArrayLike, /) -> Array:
        field = jnp.asarray(configuration)
        if field.shape != self.configuration_shape:
            raise ValueError(
                f"Scalar field must have shape {self.configuration_shape}; "
                f"got {field.shape}."
            )
        if jnp.iscomplexobj(field):
            raise TypeError("Scalar phi4 fields must be real-valued.")
        return field

    def edge_differences(self, configuration: ArrayLike, /) -> Array:
        """Return the oriented degree-zero exterior derivative."""
        return self.discretization.exterior_derivative(
            0, self._configuration(configuration)
        )

    def site_contributions(self, configuration: ArrayLike, /) -> Array:
        """Return dual-volume-weighted local potential contributions."""
        field = self._configuration(configuration)
        potential = (
            0.5 * self.mass_squared * field**2 + 0.25 * self.quartic_coupling * field**4
        )
        return self.discretization.dual_measures[0] * potential

    def action(self, configuration: PyTree[Any], /) -> Array:
        field = self._configuration(configuration)
        derivative = self.discretization.exterior_derivative(0, field)
        hodge_derivative = self.discretization.apply_hodge(1, derivative)
        kinetic = 0.5 * self.kinetic_scale * jnp.vdot(derivative, hodge_derivative)
        return jnp.real(kinetic) + jnp.sum(self.site_contributions(field))


class LocalPhi4LatticeAction(AbstractIncrementalLatticeAction):
    """Scalar action with exact site-local cache transitions for diagonal Hodge data."""

    base: Phi4LatticeAction
    incident_edges: Array
    incident_signs: Array
    incident_valid: Array
    edge_weights: Array
    topology_id: str = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    configuration_shape: tuple[int, ...] = eqx.field(static=True)
    local_coordinate_shape: tuple[int, ...] = eqx.field(static=True)
    geometry: EuclideanStateGeometry
    evidence: LatticeActionEvidence
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: Phi4LatticeAction,
        incident_edges: ArrayLike,
        incident_signs: ArrayLike,
        incident_valid: ArrayLike,
        /,
    ):
        if not isinstance(base, Phi4LatticeAction):
            raise TypeError("base must be Phi4LatticeAction.")
        edges = jnp.asarray(incident_edges, dtype=jnp.int32)
        signs = jnp.asarray(incident_signs, dtype=base.kinetic_scale.dtype)
        valid = jnp.asarray(incident_valid, dtype=bool)
        if edges.shape != signs.shape or edges.shape != valid.shape:
            raise ValueError("Incident edge, sign, and validity arrays must agree.")
        if edges.shape[0] != base.configuration_shape[0]:
            raise ValueError("Incident routes require one row per scalar site.")
        action_id = canonical_fingerprint(
            {
                "kind": "local-phi4-lattice-action",
                "base": base.action_id,
                "incident_edges": array_tree_fingerprint(np.asarray(edges)),
                "incident_signs": array_tree_fingerprint(np.asarray(signs)),
                "incident_valid": array_tree_fingerprint(np.asarray(valid)),
            }
        )
        self.base = base
        self.incident_edges = edges
        self.incident_signs = signs
        self.incident_valid = valid
        self.edge_weights = base.discretization.hodge_stars[1]
        self.topology_id = base.topology_id
        self.field_space_id = base.field_space_id
        self.configuration_shape = base.configuration_shape
        self.local_coordinate_shape = base.local_coordinate_shape
        self.geometry = base.geometry
        self.evidence = base.evidence
        self.action_id = action_id

    def action(self, configuration: PyTree[Any], /) -> Array:
        return self.base.action(configuration)

    def initialize_incremental(
        self, configuration: PyTree[Any], /
    ) -> tuple[Array, Phi4ActionCache]:
        field = self.base._configuration(configuration)
        differences = self.base.edge_differences(field)
        sites = self.base.site_contributions(field)
        kinetic = (
            0.5 * self.base.kinetic_scale * jnp.sum(self.edge_weights * differences**2)
        )
        value = kinetic + jnp.sum(sites)
        return value, Phi4ActionCache(
            edge_differences=differences,
            site_contributions=sites,
            action=value,
        )

    def propose_incremental(
        self,
        current_configuration: PyTree[Any],
        current_cache: PyTree[Any],
        proposed_configuration: PyTree[Any],
        payload: PyTree[Any],
        /,
    ) -> tuple[Array, Phi4ActionCache, Array]:
        if not isinstance(current_cache, Phi4ActionCache):
            raise TypeError("current_cache must be Phi4ActionCache.")
        if not isinstance(payload, SingleCoordinateProposalPayload):
            raise TypeError("payload must be SingleCoordinateProposalPayload.")
        current = self.base._configuration(current_configuration)
        proposed = self.base._configuration(proposed_configuration)
        index = jnp.asarray(payload.index, dtype=jnp.int32)
        current_flat = jnp.ravel(current)
        proposed_flat = jnp.ravel(proposed)
        displacement = proposed_flat[index] - current_flat[index]
        expected = current_flat.at[index].set(proposed_flat[index])
        proposal_matches = jnp.all(expected == proposed_flat) & jnp.isclose(
            displacement, jnp.asarray(payload.displacement)
        )

        edges = self.incident_edges[index]
        signs = self.incident_signs[index]
        active = self.incident_valid[index]
        old_edge = current_cache.edge_differences[edges]
        new_edge = old_edge + jnp.where(active, signs * displacement, 0.0)
        weights = self.edge_weights[edges]
        delta_kinetic = (
            0.5
            * self.base.kinetic_scale
            * jnp.sum(jnp.where(active, weights * (new_edge**2 - old_edge**2), 0.0))
        )
        new_site = self.base.discretization.dual_measures[0][index] * (
            0.5 * self.base.mass_squared * proposed_flat[index] ** 2
            + 0.25 * self.base.quartic_coupling * proposed_flat[index] ** 4
        )
        old_site = current_cache.site_contributions[index]
        delta = delta_kinetic + new_site - old_site
        edge_update = jnp.where(active, signs * displacement, 0.0)
        candidate_edges = current_cache.edge_differences.at[edges].add(edge_update)
        candidate_sites = current_cache.site_contributions.at[index].set(new_site)
        candidate_action = current_cache.action + delta
        candidate = Phi4ActionCache(
            edge_differences=candidate_edges,
            site_contributions=candidate_sites,
            action=candidate_action,
        )
        valid = (
            proposal_matches
            & (index >= 0)
            & (index < current_flat.size)
            & jnp.isfinite(delta)
            & jnp.all(jnp.isfinite(candidate_edges))
            & jnp.all(jnp.isfinite(candidate_sites))
            & jnp.isfinite(candidate_action)
        )
        return delta, candidate, valid

    def select_incremental(
        self,
        current: PyTree[Any],
        proposed: PyTree[Any],
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
        self, configuration: PyTree[Any], /
    ) -> tuple[Array, Phi4ActionCache]:
        return self.initialize_incremental(configuration)


def prepare_local_phi4_action(
    action: Phi4LatticeAction,
    /,
) -> LocalPhi4LatticeAction:
    """Prepare incident-edge routes for exact scalar single-site updates."""
    if not isinstance(action, Phi4LatticeAction):
        raise TypeError("action must be Phi4LatticeAction.")
    if action.discretization.hodge_matrices[1] is not None:
        raise ValueError("Local phi4 updates require a diagonal degree-one Hodge map.")
    incidence = action.discretization.topology.incidences[0]
    relation = incidence.relation
    valid = np.asarray(relation.valid, dtype=bool)
    source = np.asarray(relation.source_indices, dtype=np.int64)[valid]
    target = np.asarray(relation.target_indices, dtype=np.int64)[valid]
    signs = np.asarray(incidence.signs)[valid]
    num_sites = action.configuration_shape[0]
    counts = np.bincount(source, minlength=num_sites)
    capacity = max(1, int(counts.max(initial=0)))
    incident_edges = np.zeros((num_sites, capacity), dtype=np.int32)
    incident_signs = np.zeros((num_sites, capacity), dtype=float)
    incident_valid = np.zeros((num_sites, capacity), dtype=bool)
    offsets = np.zeros((num_sites,), dtype=np.int32)
    for site, edge, sign in zip(source, target, signs, strict=True):
        slot = offsets[site]
        incident_edges[site, slot] = edge
        incident_signs[site, slot] = sign
        incident_valid[site, slot] = True
        offsets[site] += 1
    return LocalPhi4LatticeAction(
        action,
        incident_edges,
        incident_signs,
        incident_valid,
    )


__all__ = [
    "LocalPhi4LatticeAction",
    "Phi4ActionCache",
    "Phi4LatticeAction",
    "prepare_local_phi4_action",
]
