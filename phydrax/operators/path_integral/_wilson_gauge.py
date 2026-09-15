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
from ..._strict import StrictModule
from ...discretization import CellBoundaryPathPlan
from ...graph import MatrixGaugeLinkSpace
from ...metrix import SpecialUnitaryGroup, UnitaryGroup
from ._lattice_action import AbstractIncrementalLatticeAction, LatticeActionEvidence


class GaugeLinkProposalPayload(StrictModule):
    """Selected edge evidence for one whole-link proposal."""

    edge: Array


class WilsonGaugeActionCache(StrictModule):
    """Exact plaquette holonomies and reduced Wilson contributions."""

    links: Array
    plaquette_holonomies: Array
    plaquette_traces: Array
    action: Array
    valid: Array


class WilsonGaugeAction(AbstractIncrementalLatticeAction):
    """Fundamental-representation Wilson action on ordered matrix-link paths."""

    link_space: MatrixGaugeLinkSpace
    boundaries: CellBoundaryPathPlan
    plaquette_couplings: Array
    edge_paths: Array
    edge_paths_valid: Array
    geometry: Any
    evidence: LatticeActionEvidence
    topology_id: str = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    configuration_shape: tuple[int, ...] = eqx.field(static=True)
    local_coordinate_shape: tuple[int, ...] = eqx.field(static=True)
    num_plaquettes: int = eqx.field(static=True)
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        link_space: MatrixGaugeLinkSpace,
        boundaries: CellBoundaryPathPlan,
        /,
        *,
        plaquette_couplings: ArrayLike,
    ):
        if not isinstance(link_space, MatrixGaugeLinkSpace):
            raise TypeError("link_space must be MatrixGaugeLinkSpace.")
        if not isinstance(boundaries, CellBoundaryPathPlan):
            raise TypeError("boundaries must be CellBoundaryPathPlan.")
        if boundaries.paths.topology_id != link_space.topology.topology_id:
            raise ValueError("Wilson boundaries and link space must share one topology.")
        if not isinstance(link_space.group, (UnitaryGroup, SpecialUnitaryGroup)):
            raise TypeError("WilsonGaugeAction currently supports U(N) and SU(N).")
        num_plaquettes = boundaries.paths.num_paths
        coupling_host = np.asarray(plaquette_couplings, dtype=float)
        if coupling_host.shape == ():
            coupling_host = np.full((num_plaquettes,), float(coupling_host))
        if coupling_host.shape != (num_plaquettes,):
            raise ValueError(
                "plaquette_couplings must be scalar or provide one value per path."
            )
        if np.any(~np.isfinite(coupling_host)) or np.any(coupling_host < 0.0):
            raise ValueError(
                "Wilson plaquette couplings must be finite and non-negative."
            )
        path_edges = np.asarray(boundaries.paths.edge_indices)
        path_valid = np.asarray(boundaries.paths.valid)
        affected: list[list[int]] = [[] for _ in range(link_space.num_edges)]
        for path_index in range(num_plaquettes):
            for edge in np.unique(path_edges[path_index, path_valid[path_index]]):
                affected[int(edge)].append(path_index)
        capacity = max(1, max((len(value) for value in affected), default=0))
        edge_paths = np.zeros((link_space.num_edges, capacity), dtype=np.int32)
        edge_paths_valid = np.zeros((link_space.num_edges, capacity), dtype=bool)
        for edge, path_indices in enumerate(affected):
            edge_paths[edge, : len(path_indices)] = path_indices
            edge_paths_valid[edge, : len(path_indices)] = True
        action_id = canonical_fingerprint(
            {
                "kind": "wilson-gauge-action",
                "link_space": link_space.link_space_id,
                "boundaries": boundaries.boundary_plan_id,
                "couplings": array_tree_fingerprint(coupling_host),
                "representation": "fundamental-normalized-trace",
                "convention": "reduced-wilson-minus-real-trace",
            }
        )
        self.link_space = link_space
        self.boundaries = boundaries
        self.plaquette_couplings = jnp.asarray(coupling_host)
        self.edge_paths = jnp.asarray(edge_paths)
        self.edge_paths_valid = jnp.asarray(edge_paths_valid)
        self.geometry = link_space.geometry
        self.evidence = LatticeActionEvidence(
            reference_measure="product-haar",
            real_valued=True,
            bounded_below=True,
            normalizable=True,
            additive_constant=float(np.sum(coupling_host)),
            evidence_id=f"{action_id}:measure-evidence",
        )
        self.topology_id = link_space.topology.topology_id
        self.field_space_id = link_space.field_space.field_space_id
        self.configuration_shape = link_space.configuration_shape
        self.local_coordinate_shape = link_space.local_coordinate_shape
        self.num_plaquettes = num_plaquettes
        self.action_id = action_id

    def _links(self, links: ArrayLike, /) -> Array:
        values = jnp.asarray(links)
        if values.shape != self.configuration_shape:
            raise ValueError(
                f"links must have shape {self.configuration_shape}; got {values.shape}."
            )
        return values

    def _selected_holonomies(self, links: Array, path_indices: Array, /) -> Array:
        paths = self.boundaries.paths
        identity = jnp.broadcast_to(
            self.link_space.group.identity(dtype=links.dtype),
            (path_indices.shape[0],) + self.link_space.point_shape,
        )

        def step(accumulator, position):
            edges = paths.edge_indices[path_indices, position]
            factors = links[edges]
            inverses = self.link_space.group.inverse(factors)
            oriented = jnp.where(
                (paths.orientations[path_indices, position] > 0)[..., None, None],
                factors,
                inverses,
            )
            product = self.link_space.group.compose(accumulator, oriented)
            active = paths.valid[path_indices, position][..., None, None]
            return jnp.where(active, product, accumulator), None

        holonomies, _ = jax.lax.scan(
            step,
            identity,
            jnp.arange(paths.max_length),
        )
        return holonomies

    def plaquette_holonomies(self, links: ArrayLike, /) -> Array:
        values = self._links(links)
        indices = jnp.arange(self.num_plaquettes, dtype=jnp.int32)
        return self._selected_holonomies(values, indices)

    def plaquette_traces(self, links: ArrayLike, /) -> Array:
        holonomies = self.plaquette_holonomies(links)
        dimension = self.link_space.point_shape[0]
        return jnp.real(jnp.trace(holonomies, axis1=-2, axis2=-1)) / dimension

    def action(self, links: PyTree[Array], /) -> Array:
        return -jnp.sum(self.plaquette_couplings * self.plaquette_traces(links))

    def canonical_action(self, links: ArrayLike, /) -> Array:
        return self.action(links) + self.evidence.additive_constant

    def initialize_incremental(
        self, configuration: PyTree[Array], /
    ) -> tuple[Array, WilsonGaugeActionCache]:
        links = self._links(configuration)
        holonomies = self.plaquette_holonomies(links)
        dimension = self.link_space.point_shape[0]
        traces = jnp.real(jnp.trace(holonomies, axis1=-2, axis2=-1)) / dimension
        value = -jnp.sum(self.plaquette_couplings * traces)
        valid = self.link_space.contains(links) & jnp.isfinite(value)
        return value, WilsonGaugeActionCache(
            links=links,
            plaquette_holonomies=holonomies,
            plaquette_traces=traces,
            action=value,
            valid=valid,
        )

    def propose_incremental(
        self,
        current_configuration: PyTree[Array],
        current_cache: PyTree[Array],
        proposed_configuration: PyTree[Array],
        payload: PyTree[Array],
        /,
    ) -> tuple[Array, WilsonGaugeActionCache, Array]:
        if not isinstance(current_cache, WilsonGaugeActionCache):
            raise TypeError("current_cache must be WilsonGaugeActionCache.")
        if not isinstance(payload, GaugeLinkProposalPayload):
            raise TypeError("payload must be GaugeLinkProposalPayload.")
        current = self._links(current_configuration)
        proposed = self._links(proposed_configuration)
        edge = jnp.asarray(payload.edge, dtype=jnp.int32)
        expected = current.at[edge].set(proposed[edge])
        proposal_matches = jnp.allclose(expected, proposed)
        affected_paths = self.edge_paths[edge]
        affected_valid = self.edge_paths_valid[edge]
        new_holonomies = self._selected_holonomies(proposed, affected_paths)
        dimension = self.link_space.point_shape[0]
        new_traces = jnp.real(jnp.trace(new_holonomies, axis1=-2, axis2=-1)) / dimension
        old_traces = current_cache.plaquette_traces[affected_paths]
        couplings = self.plaquette_couplings[affected_paths]
        delta = -jnp.sum(
            jnp.where(
                affected_valid,
                couplings * (new_traces - old_traces),
                0.0,
            )
        )
        holonomy_updates = jnp.where(
            affected_valid[..., None, None],
            new_holonomies - current_cache.plaquette_holonomies[affected_paths],
            0.0,
        )
        trace_updates = jnp.where(
            affected_valid,
            new_traces - old_traces,
            0.0,
        )
        candidate_holonomies = current_cache.plaquette_holonomies.at[affected_paths].add(
            holonomy_updates
        )
        candidate_traces = current_cache.plaquette_traces.at[affected_paths].add(
            trace_updates
        )
        candidate_action = current_cache.action + delta
        valid = (
            current_cache.valid
            & proposal_matches
            & self.link_space.contains(proposed)
            & jnp.isfinite(delta)
            & jnp.isfinite(candidate_action)
        )
        candidate = WilsonGaugeActionCache(
            links=proposed,
            plaquette_holonomies=candidate_holonomies,
            plaquette_traces=candidate_traces,
            action=candidate_action,
            valid=valid,
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
        self, configuration: PyTree[Array], /
    ) -> tuple[Array, WilsonGaugeActionCache]:
        return self.initialize_incremental(configuration)


__all__ = [
    "GaugeLinkProposalPayload",
    "WilsonGaugeAction",
    "WilsonGaugeActionCache",
]
