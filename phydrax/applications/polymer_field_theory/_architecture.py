#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ContourBlockPlan(StrictModule, NonTrainableState):
    block_id: str = eqx.field(static=True)
    source_node: str = eqx.field(static=True)
    target_node: str = eqx.field(static=True)
    species_index: int = eqx.field(static=True)
    contour_fraction: float = eqx.field(static=True)
    contour_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        block_id: str,
        source_node: str,
        target_node: str,
        species_index: int,
        contour_fraction: float,
        contour_steps: int,
        /,
    ):
        identifier = str(block_id).strip()
        source = str(source_node).strip()
        target = str(target_node).strip()
        species = int(species_index)
        fraction = float(contour_fraction)
        steps = int(contour_steps)
        if (
            not identifier
            or not source
            or not target
            or source == target
            or species < 0
            or not math.isfinite(fraction)
            or fraction <= 0.0
            or steps <= 0
        ):
            raise ValueError("Contour block definition is invalid.")
        self.block_id = identifier
        self.source_node = source
        self.target_node = target
        self.species_index = species
        self.contour_fraction = fraction
        self.contour_steps = steps
        self.plan_id = canonical_fingerprint(
            {
                "kind": "contour-block-plan",
                "block_id": identifier,
                "source_node": source,
                "target_node": target,
                "species_index": species,
                "contour_fraction": fraction,
                "contour_steps": steps,
            }
        )


class PolymerContourArchitecturePlan(StrictModule, NonTrainableState):
    architecture_id: str = eqx.field(static=True)
    blocks: tuple[ContourBlockPlan, ...]
    root_node: str = eqx.field(static=True)
    node_ids: tuple[str, ...] = eqx.field(static=True)
    incident_blocks: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        architecture_id: str,
        blocks: tuple[ContourBlockPlan, ...],
        /,
        *,
        root_node: str,
    ):
        identifier = str(architecture_id).strip()
        values = tuple(blocks)
        root = str(root_node).strip()
        if (
            not identifier
            or not values
            or any(not isinstance(block, ContourBlockPlan) for block in values)
            or len({block.block_id for block in values}) != len(values)
        ):
            raise ValueError("Contour architecture blocks are invalid.")
        nodes = tuple(
            sorted(
                {
                    node
                    for block in values
                    for node in (block.source_node, block.target_node)
                }
            )
        )
        if root not in nodes or len(values) != len(nodes) - 1:
            raise ValueError("Contour architecture must declare one rooted tree.")
        node_index = {node: index for index, node in enumerate(nodes)}
        adjacency: list[list[int]] = [[] for _ in nodes]
        for edge, block in enumerate(values):
            adjacency[node_index[block.source_node]].append(edge)
            adjacency[node_index[block.target_node]].append(edge)
        reached = {root}
        frontier = [root]
        while frontier:
            node = frontier.pop()
            for edge in adjacency[node_index[node]]:
                block = values[edge]
                other = (
                    block.target_node if block.source_node == node else block.source_node
                )
                if other not in reached:
                    reached.add(other)
                    frontier.append(other)
        if reached != set(nodes):
            raise ValueError("Contour architecture must be connected and acyclic.")
        total_fraction = sum(block.contour_fraction for block in values)
        if not math.isclose(total_fraction, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Contour block fractions must sum to one.")
        self.architecture_id = identifier
        self.blocks = values
        self.root_node = root
        self.node_ids = nodes
        self.incident_blocks = tuple(tuple(edges) for edges in adjacency)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polymer-contour-architecture-plan",
                "architecture_id": identifier,
                "blocks": [block.plan_id for block in values],
                "root_node": root,
                "node_ids": list(nodes),
            }
        )

    def incident(self, node: str, /) -> tuple[int, ...]:
        return self.incident_blocks[self.node_ids.index(node)]

    def other_node(self, block_index: int, node: str, /) -> str:
        block = self.blocks[int(block_index)]
        if block.source_node == node:
            return block.target_node
        if block.target_node == node:
            return block.source_node
        raise ValueError("Node is not incident to the requested contour block.")


class PolymerComponentPlan(StrictModule, NonTrainableState):
    component_id: str = eqx.field(static=True)
    architecture: PolymerContourArchitecturePlan
    volume_fraction: float = eqx.field(static=True)
    polymerization_index: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_id: str,
        architecture: PolymerContourArchitecturePlan,
        volume_fraction: float,
        polymerization_index: float,
        /,
    ):
        identifier = str(component_id).strip()
        fraction = float(volume_fraction)
        polymerization = float(polymerization_index)
        if not isinstance(architecture, PolymerContourArchitecturePlan):
            raise TypeError("architecture must be PolymerContourArchitecturePlan.")
        if (
            not identifier
            or not math.isfinite(fraction)
            or fraction <= 0.0
            or not math.isfinite(polymerization)
            or polymerization <= 0.0
        ):
            raise ValueError("Polymer component definition is invalid.")
        self.component_id = identifier
        self.architecture = architecture
        self.volume_fraction = fraction
        self.polymerization_index = polymerization
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polymer-component-plan",
                "component_id": identifier,
                "architecture": architecture.plan_id,
                "volume_fraction": fraction,
                "polymerization_index": polymerization,
            }
        )


class IncompressibleGaussianMixturePlan(StrictModule, NonTrainableState):
    species_ids: tuple[str, ...] = eqx.field(static=True)
    statistical_segment_lengths: Array
    chi_n: Array
    components: tuple[PolymerComponentPlan, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        species_ids: tuple[str, ...],
        statistical_segment_lengths: ArrayLike,
        chi_n: ArrayLike,
        components: tuple[PolymerComponentPlan, ...],
        /,
    ):
        identifiers = tuple(str(value).strip() for value in species_ids)
        lengths = np.asarray(statistical_segment_lengths, dtype=float)
        interactions = np.asarray(chi_n, dtype=float)
        values = tuple(components)
        count = len(identifiers)
        if (
            count < 2
            or any(not value for value in identifiers)
            or len(set(identifiers)) != count
            or lengths.shape != (count,)
            or np.any(~np.isfinite(lengths))
            or np.any(lengths <= 0.0)
            or interactions.shape != (count, count)
            or np.any(~np.isfinite(interactions))
            or not np.allclose(interactions, interactions.T)
            or not np.allclose(np.diag(interactions), 0.0)
            or not values
            or any(not isinstance(value, PolymerComponentPlan) for value in values)
            or len({value.component_id for value in values}) != len(values)
        ):
            raise ValueError("Incompressible Gaussian mixture definition is invalid.")
        if not math.isclose(
            sum(value.volume_fraction for value in values),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Polymer component volume fractions must sum to one.")
        if any(
            block.species_index >= count
            for value in values
            for block in value.architecture.blocks
        ):
            raise ValueError("Contour block references an unknown monomer species.")
        self.species_ids = identifiers
        self.statistical_segment_lengths = jnp.asarray(lengths)
        self.chi_n = jnp.asarray(interactions)
        self.components = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "incompressible-gaussian-mixture-plan",
                "species_ids": list(identifiers),
                "statistical_segment_lengths": array_tree_fingerprint(lengths),
                "chi_n": array_tree_fingerprint(interactions),
                "components": [value.plan_id for value in values],
            }
        )

    @property
    def species_count(self) -> int:
        return len(self.species_ids)


__all__ = [
    "ContourBlockPlan",
    "IncompressibleGaussianMixturePlan",
    "PolymerComponentPlan",
    "PolymerContourArchitecturePlan",
]
