#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState

from ._morton import (
    _morton_decode_host,
    _morton_encode_host,
    MortonAddressPlan,
)


LeafKey = tuple[int, int]


class DyadicTopologyEvidence(NonTrainableState, StrictModule):
    """Structural evidence for a dyadic leaf partition."""

    allocated_cells: jax.Array
    active_leaves: jax.Array
    required_capacity: jax.Array
    antichain: jax.Array
    covering: jax.Array
    two_to_one_balanced: jax.Array
    successful: jax.Array


class DyadicCellTopology(NonTrainableState, StrictModule):
    """Fixed-capacity mixed-level Cartesian cell topology."""

    address_plan: MortonAddressPlan
    prefixes: jax.Array
    levels: jax.Array
    allocated: jax.Array
    leaf_active: jax.Array
    parents: jax.Array
    children: jax.Array
    cell_lower: jax.Array
    cell_upper: jax.Array
    cell_centers: jax.Array
    cell_volumes: jax.Array
    root_slot: jax.Array
    epoch: jax.Array
    evidence: DyadicTopologyEvidence
    require_covering: bool = eqx.field(static=True)
    enforce_two_to_one_balance: bool = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    @property
    def cell_capacity(self) -> int:
        return int(self.prefixes.shape[0])


class DyadicAdaptationEvidence(NonTrainableState, StrictModule):
    """Requested, closed, and accepted adaptive topology changes."""

    requested_refinements: jax.Array
    accepted_refinements: jax.Array
    requested_coarsenings: jax.Array
    accepted_coarsenings: jax.Array
    balance_refinements: jax.Array
    maximum_depth_rejections: jax.Array
    required_capacity: jax.Array
    successful: jax.Array


class DyadicTopologyTransition(NonTrainableState, StrictModule):
    """Atomic accepted-or-previous dyadic topology transition."""

    candidate: DyadicCellTopology
    accepted: DyadicCellTopology
    accepted_candidate: jax.Array
    evidence: DyadicAdaptationEvidence


class AdaptiveDyadicGridPlan(StrictModule):
    """Prepare and adapt a covering quadtree/octree at topology epochs."""

    address_plan: MortonAddressPlan
    cell_capacity: int = eqx.field(static=True)
    require_covering: bool = eqx.field(static=True)
    enforce_two_to_one_balance: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        address_plan: MortonAddressPlan,
        *,
        cell_capacity: int,
        require_covering: bool = True,
        enforce_two_to_one_balance: bool = True,
    ) -> None:
        capacity = int(cell_capacity)
        if capacity < 1:
            raise ValueError("cell_capacity must be positive.")
        object.__setattr__(self, "address_plan", address_plan)
        object.__setattr__(self, "cell_capacity", capacity)
        object.__setattr__(self, "require_covering", bool(require_covering))
        object.__setattr__(
            self, "enforce_two_to_one_balance", bool(enforce_two_to_one_balance)
        )
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "adaptive-dyadic-grid-plan",
                    "address_plan_id": address_plan.plan_id,
                    "cell_capacity": capacity,
                    "require_covering": bool(require_covering),
                    "enforce_two_to_one_balance": bool(enforce_two_to_one_balance),
                }
            ),
        )

    def prepare(self) -> DyadicCellTopology:
        topology, required = self._build_topology({(0, 0)}, epoch=0)
        if required > self.cell_capacity:
            raise ValueError("The root topology exceeds cell_capacity.")
        return topology

    def prepare_leaves(
        self, leaves: Iterable[LeafKey], /, *, epoch: int = 0
    ) -> DyadicCellTopology:
        leaf_set = {(int(level), int(prefix)) for level, prefix in leaves}
        self._validate_leaf_keys(leaf_set)
        topology, required = self._build_topology(leaf_set, epoch=epoch)
        if required > self.cell_capacity:
            raise ValueError(
                f"Dyadic topology requires {required} cells but capacity is "
                f"{self.cell_capacity}."
            )
        if not bool(topology.evidence.successful):
            raise ValueError("Leaf keys violate the requested dyadic invariants.")
        return topology

    def adapt(
        self,
        previous: DyadicCellTopology,
        /,
        *,
        refine_mask: jax.Array | None = None,
        coarsen_mask: jax.Array | None = None,
    ) -> DyadicTopologyTransition:
        if not isinstance(previous, DyadicCellTopology):
            raise TypeError("previous must be DyadicCellTopology.")
        expected = (self.cell_capacity,)
        refine = (
            np.zeros(expected, dtype=bool)
            if refine_mask is None
            else np.asarray(refine_mask, dtype=bool)
        )
        coarsen = (
            np.zeros(expected, dtype=bool)
            if coarsen_mask is None
            else np.asarray(coarsen_mask, dtype=bool)
        )
        if refine.shape != expected or coarsen.shape != expected:
            raise ValueError("Adaptation masks must match cell_capacity.")
        previous_prefix = np.asarray(previous.prefixes)
        previous_level = np.asarray(previous.levels)
        previous_leaf = np.asarray(previous.leaf_active)
        leaves = {
            (int(previous_level[slot]), int(previous_prefix[slot]))
            for slot in np.flatnonzero(previous_leaf)
        }
        original_leaves = set(leaves)
        requested_refine = {
            (int(previous_level[slot]), int(previous_prefix[slot]))
            for slot in np.flatnonzero(previous_leaf & refine)
        }
        requested_coarsen = {
            (int(previous_level[slot]), int(previous_prefix[slot]))
            for slot in np.flatnonzero(previous_leaf & coarsen)
        }
        branching = 1 << self.address_plan.dimension
        accepted_coarsenings = 0
        parent_requests: dict[LeafKey, set[LeafKey]] = {}
        for leaf in requested_coarsen:
            level, prefix = leaf
            if level == 0:
                continue
            parent = (level - 1, prefix >> self.address_plan.dimension)
            parent_requests.setdefault(parent, set()).add(leaf)
        for parent in sorted(parent_requests):
            level, prefix = parent
            children = {
                (level + 1, (prefix << self.address_plan.dimension) | child)
                for child in range(branching)
            }
            if children <= leaves and children <= requested_coarsen:
                leaves.difference_update(children)
                leaves.add(parent)
                accepted_coarsenings += 1
        accepted_refinements = 0
        maximum_depth_rejections = 0
        for leaf in sorted(requested_refine):
            if leaf not in leaves:
                continue
            level, prefix = leaf
            if level >= self.address_plan.maximum_depth:
                maximum_depth_rejections += 1
                continue
            leaves.remove(leaf)
            leaves.update(
                {
                    (level + 1, (prefix << self.address_plan.dimension) | child)
                    for child in range(branching)
                }
            )
            accepted_refinements += 1
        balance_refinements = 0
        if self.enforce_two_to_one_balance:
            leaves, balance_refinements = self._balance(leaves)
        if leaves == original_leaves:
            candidate = previous
            required = int(np.asarray(previous.evidence.required_capacity))
        else:
            candidate, required = self._build_topology(
                leaves, epoch=int(np.asarray(previous.epoch)) + 1
            )
        successful = (
            required <= self.cell_capacity
            and bool(candidate.evidence.successful)
            and maximum_depth_rejections == 0
        )
        if not successful:
            candidate = previous
        evidence = DyadicAdaptationEvidence(
            requested_refinements=jnp.asarray(len(requested_refine), dtype=jnp.int32),
            accepted_refinements=jnp.asarray(accepted_refinements, dtype=jnp.int32),
            requested_coarsenings=jnp.asarray(len(requested_coarsen), dtype=jnp.int32),
            accepted_coarsenings=jnp.asarray(accepted_coarsenings, dtype=jnp.int32),
            balance_refinements=jnp.asarray(balance_refinements, dtype=jnp.int32),
            maximum_depth_rejections=jnp.asarray(
                maximum_depth_rejections, dtype=jnp.int32
            ),
            required_capacity=jnp.asarray(required, dtype=jnp.int32),
            successful=jnp.asarray(successful),
        )
        return DyadicTopologyTransition(
            candidate=candidate,
            accepted=candidate if successful else previous,
            accepted_candidate=jnp.asarray(successful),
            evidence=evidence,
        )

    def _validate_leaf_keys(self, leaves: set[LeafKey]) -> None:
        if not leaves:
            raise ValueError("Dyadic topology requires at least one leaf.")
        for level, prefix in leaves:
            if level < 0 or level > self.address_plan.maximum_depth:
                raise ValueError("Dyadic leaf level is outside the plan depth.")
            if prefix < 0 or prefix >= 1 << (self.address_plan.dimension * level):
                raise ValueError("Dyadic leaf prefix is invalid for its level.")

    def _build_topology(
        self, leaves: set[LeafKey], *, epoch: int
    ) -> tuple[DyadicCellTopology, int]:
        self._validate_leaf_keys(leaves)
        nodes = set(leaves)
        for level, prefix in tuple(leaves):
            current_level, current_prefix = level, prefix
            while current_level > 0:
                current_prefix >>= self.address_plan.dimension
                current_level -= 1
                nodes.add((current_level, current_prefix))
        ordered = sorted(nodes)
        required = len(ordered)
        stored = ordered[: self.cell_capacity]
        slot_by_key = {key: slot for slot, key in enumerate(stored)}
        branching = 1 << self.address_plan.dimension
        prefixes = np.zeros((self.cell_capacity,), dtype=np.uint64)
        levels = np.zeros((self.cell_capacity,), dtype=np.int32)
        allocated = np.zeros((self.cell_capacity,), dtype=bool)
        leaf_active = np.zeros((self.cell_capacity,), dtype=bool)
        parents = -np.ones((self.cell_capacity,), dtype=np.int32)
        children = -np.ones((self.cell_capacity, branching), dtype=np.int32)
        for slot, (level, prefix) in enumerate(stored):
            prefixes[slot] = prefix
            levels[slot] = level
            allocated[slot] = True
            leaf_active[slot] = (level, prefix) in leaves
            if level > 0:
                parents[slot] = slot_by_key.get(
                    (level - 1, prefix >> self.address_plan.dimension), -1
                )
            for child in range(branching):
                children[slot, child] = slot_by_key.get(
                    (level + 1, (prefix << self.address_plan.dimension) | child),
                    -1,
                )
        geometry = self.address_plan.cell_geometry(
            jnp.asarray(prefixes), jnp.asarray(levels)
        )
        volumes = jnp.prod(geometry.upper - geometry.lower, axis=-1)
        antichain = self._is_antichain(leaves)
        covering = self._covers_root(leaves)
        balanced = self._is_two_to_one_balanced(leaves)
        complete = required <= self.cell_capacity
        successful = (
            complete
            and antichain
            and (covering or not self.require_covering)
            and (balanced or not self.enforce_two_to_one_balance)
        )
        topology_id = canonical_fingerprint(
            {
                "kind": "dyadic-cell-topology",
                "plan": self.plan_id,
                "leaves": [list(key) for key in sorted(leaves)],
                "epoch": int(epoch),
            }
        )
        evidence = DyadicTopologyEvidence(
            allocated_cells=jnp.asarray(
                min(required, self.cell_capacity), dtype=jnp.int32
            ),
            active_leaves=jnp.asarray(
                sum(key in leaves for key in stored), dtype=jnp.int32
            ),
            required_capacity=jnp.asarray(required, dtype=jnp.int32),
            antichain=jnp.asarray(antichain),
            covering=jnp.asarray(covering),
            two_to_one_balanced=jnp.asarray(balanced),
            successful=jnp.asarray(successful),
        )
        return (
            DyadicCellTopology(
                address_plan=self.address_plan,
                prefixes=jnp.asarray(prefixes),
                levels=jnp.asarray(levels),
                allocated=jnp.asarray(allocated),
                leaf_active=jnp.asarray(leaf_active),
                parents=jnp.asarray(parents),
                children=jnp.asarray(children),
                cell_lower=jnp.where(
                    jnp.asarray(allocated)[:, None], geometry.lower, 0.0
                ),
                cell_upper=jnp.where(
                    jnp.asarray(allocated)[:, None], geometry.upper, 0.0
                ),
                cell_centers=jnp.where(
                    jnp.asarray(allocated)[:, None], geometry.center, 0.0
                ),
                cell_volumes=jnp.where(jnp.asarray(allocated), volumes, 0.0),
                root_slot=jnp.asarray(slot_by_key.get((0, 0), -1), dtype=jnp.int32),
                epoch=jnp.asarray(epoch, dtype=jnp.int32),
                evidence=evidence,
                require_covering=self.require_covering,
                enforce_two_to_one_balance=self.enforce_two_to_one_balance,
                topology_id=topology_id,
            ),
            required,
        )

    def _balance(self, initial: set[LeafKey]) -> tuple[set[LeafKey], int]:
        leaves = set(initial)
        branching = 1 << self.address_plan.dimension
        added = 0
        changed = True
        while changed:
            changed = False
            to_refine: set[LeafKey] = set()
            for leaf in tuple(leaves):
                level, prefix = leaf
                coordinate = _morton_decode_host(
                    prefix, self.address_plan.dimension, level
                )
                width = 1 << (self.address_plan.maximum_depth - level)
                lower = tuple(value * width for value in coordinate)
                center = [value + width // 2 for value in lower]
                for axis in range(self.address_plan.dimension):
                    for direction in (-1, 1):
                        sample = list(center)
                        sample[axis] = (
                            lower[axis] - 1 if direction < 0 else lower[axis] + width
                        )
                        valid = True
                        for component in range(self.address_plan.dimension):
                            resolution = self.address_plan.resolution
                            if self.address_plan.periodic_axes[component]:
                                sample[component] %= resolution
                            elif sample[component] < 0 or sample[component] >= resolution:
                                valid = False
                        if not valid:
                            continue
                        neighbor = self._containing_leaf(leaves, tuple(sample))
                        if neighbor is not None and level > neighbor[0] + 1:
                            to_refine.add(neighbor)
            for level, prefix in sorted(to_refine):
                if (
                    level >= self.address_plan.maximum_depth
                    or (level, prefix) not in leaves
                ):
                    continue
                leaves.remove((level, prefix))
                leaves.update(
                    {
                        (level + 1, (prefix << self.address_plan.dimension) | child)
                        for child in range(branching)
                    }
                )
                added += 1
                changed = True
        return leaves, added

    def _containing_leaf(
        self, leaves: set[LeafKey], maximum_depth_coordinate: tuple[int, ...]
    ) -> LeafKey | None:
        for level in range(self.address_plan.maximum_depth, -1, -1):
            shift = self.address_plan.maximum_depth - level
            coordinate = tuple(value >> shift for value in maximum_depth_coordinate)
            key = (
                level,
                _morton_encode_host(coordinate, self.address_plan.dimension, level),
            )
            if key in leaves:
                return key
        return None

    def _is_antichain(self, leaves: set[LeafKey]) -> bool:
        for level, prefix in leaves:
            current = prefix
            for ancestor_level in range(level - 1, -1, -1):
                current >>= self.address_plan.dimension
                if (ancestor_level, current) in leaves:
                    return False
        return True

    def _covers_root(self, leaves: set[LeafKey]) -> bool:
        measure = sum(
            2.0 ** (-self.address_plan.dimension * level) for level, _ in leaves
        )
        return self._is_antichain(leaves) and abs(measure - 1.0) <= 1.0e-12

    def _is_two_to_one_balanced(self, leaves: set[LeafKey]) -> bool:
        if not leaves:
            return False
        balanced, additions = self._balance_without_recursion(leaves)
        return balanced and additions == 0

    def _balance_without_recursion(self, leaves: set[LeafKey]) -> tuple[bool, int]:
        violations = 0
        for leaf in leaves:
            level, prefix = leaf
            coordinate = _morton_decode_host(prefix, self.address_plan.dimension, level)
            width = 1 << (self.address_plan.maximum_depth - level)
            lower = tuple(value * width for value in coordinate)
            center = [value + width // 2 for value in lower]
            for axis in range(self.address_plan.dimension):
                for direction in (-1, 1):
                    sample = list(center)
                    sample[axis] = (
                        lower[axis] - 1 if direction < 0 else lower[axis] + width
                    )
                    valid = True
                    for component in range(self.address_plan.dimension):
                        resolution = self.address_plan.resolution
                        if self.address_plan.periodic_axes[component]:
                            sample[component] %= resolution
                        elif sample[component] < 0 or sample[component] >= resolution:
                            valid = False
                    if not valid:
                        continue
                    neighbor = self._containing_leaf(leaves, tuple(sample))
                    if neighbor is not None and abs(level - neighbor[0]) > 1:
                        violations += 1
        return violations == 0, violations


__all__ = [
    "AdaptiveDyadicGridPlan",
    "DyadicAdaptationEvidence",
    "DyadicCellTopology",
    "DyadicTopologyEvidence",
    "DyadicTopologyTransition",
]
