#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg._dense_svd import economy_svd
from ..typing import checked
from ._system import PreparedAtomisticSystem


class DistanceConstraintPlan(StrictModule, NonTrainableState):
    """Iterative SHAKE/RATTLE distance constraints and their rank admission.

    ``maximum_component_atoms`` bounds the explicit per-component Jacobian
    materialization of rank evidence (connected components of the constraint
    graph); a larger component is refused at preparation before any
    allocation.  ``rank_tolerance`` is the smallest admitted ratio of a
    component's smallest to largest mass-weighted Jacobian singular value.
    """

    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_component_atoms: int = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_iterations: int = 32,
        tolerance: float = 1.0e-10,
        maximum_component_atoms: int = 16,
        rank_tolerance: float = 1.0e-8,
    ) -> None:
        iterations = int(maximum_iterations)
        threshold = float(tolerance)
        component = int(maximum_component_atoms)
        rank = float(rank_tolerance)
        if iterations <= 0 or not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("Constraint iterations and tolerance must be positive.")
        if component < 2:
            raise ValueError("maximum_component_atoms must be at least two.")
        if not math.isfinite(rank) or not 0.0 < rank < 1.0:
            raise ValueError("rank_tolerance must lie in (0, 1).")
        self.maximum_iterations = iterations
        self.tolerance = threshold
        self.maximum_component_atoms = component
        self.rank_tolerance = rank
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distance-constraint-plan",
                "maximum_iterations": iterations,
                "tolerance": threshold,
                "maximum_component_atoms": component,
                "rank_tolerance": rank,
            }
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedDistanceConstraints":
        return PreparedDistanceConstraints(self, system)


class ConstraintProjection(StrictModule):
    positions: Array
    momenta: Array
    multipliers: Array
    position_residual: Array
    velocity_residual: Array
    iterations: Array
    successful: Array


class ConstraintRankEvidence(StrictModule):
    """Rank admission of the declared constraints at one explicit geometry.

    ``component_rank`` counts singular values above ``rank_tolerance`` times the
    largest of each component's mass-weighted Jacobian; ``regular`` holds when
    every component has full row rank, i.e. the declared constraints are
    independent at this geometry and ``independent_constraint_count`` is the
    exact removed dimension.
    """

    component_rank: Array
    minimum_relative_singular_value: Array
    regular: Array


def _components(
    indices: np.ndarray, maximum_atoms: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Host component layout of the static constraint graph (bounded, padded)."""
    parent: dict[int, int] = {}

    def find(value: int) -> int:
        root = value
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[value] != root:
            parent[value], value = root, parent[value]
        return root

    for left, right in indices.tolist():
        parent[find(int(left))] = find(int(right))
    groups: dict[int, list[int]] = {}
    for atom in sorted(parent):
        groups.setdefault(find(atom), []).append(atom)
    ordered = sorted(groups.values(), key=lambda atoms: atoms[0])
    width = max((len(atoms) for atoms in ordered), default=0)
    if width > maximum_atoms:
        raise ValueError(
            f"A constraint component spans {width} atoms; the rank-evidence "
            f"materialization limit is {maximum_atoms}."
        )
    rows_by_component: list[list[int]] = [[] for _ in ordered]
    owner = {atom: index for index, atoms in enumerate(ordered) for atom in atoms}
    for row, (left, _) in enumerate(indices.tolist()):
        rows_by_component[owner[int(left)]].append(row)
    height = max((len(rows) for rows in rows_by_component), default=0)
    count = len(ordered)
    atoms = np.zeros((count, max(width, 1)), dtype=np.int32)
    atom_mask = np.zeros((count, max(width, 1)), dtype=np.bool_)
    rows = np.zeros((count, max(height, 1)), dtype=np.int32)
    row_mask = np.zeros((count, max(height, 1)), dtype=np.bool_)
    local_left = np.zeros((count, max(height, 1)), dtype=np.int32)
    local_right = np.zeros((count, max(height, 1)), dtype=np.int32)
    for index, members in enumerate(ordered):
        position = {atom: slot for slot, atom in enumerate(members)}
        atoms[index, : len(members)] = members
        atom_mask[index, : len(members)] = True
        for slot, row in enumerate(rows_by_component[index]):
            rows[index, slot] = row
            row_mask[index, slot] = True
            local_left[index, slot] = position[int(indices[row, 0])]
            local_right[index, slot] = position[int(indices[row, 1])]
    return atoms, atom_mask, rows, row_mask, local_left, local_right


class PreparedDistanceConstraints(StrictModule, NonTrainableState):
    """Prepared SHAKE/RATTLE constraints with constant-rank admission.

    ``independent_constraint_count`` declares every topology constraint
    independent (constant rank).  The declaration is certified only at an
    explicit geometry through ``rank_evidence``; dynamics checks it at
    initialization and after every position projection and refuses a
    trajectory that reaches a rank-changing configuration rather than
    silently changing its degrees of freedom.

    ``independent_constraint_count`` (the topology constraint row count) and
    the padded component arrays' shapes are declared layout.  The prepared
    digest binds the material system, so it is the dynamic
    ``prepared_identity`` (host ``prepared_id``, unchanged digest);
    ``layout_id`` is the content-free executable layout.
    """

    plan: DistanceConstraintPlan
    system: PreparedAtomisticSystem
    component_atoms: Array
    component_atom_mask: Array
    component_rows: Array
    component_row_mask: Array
    component_left: Array
    component_right: Array
    prepared_identity: IdentityWords
    independent_constraint_count: int = eqx.field(static=True)

    @checked
    def __init__(
        self, plan: DistanceConstraintPlan, system: PreparedAtomisticSystem, /
    ) -> None:
        constraint_indices = np.asarray(system.topology.constraint_indices)
        if constraint_indices.size:
            constrained_mobile = np.asarray(system.mobile_mask)[constraint_indices]
            if not np.all(constrained_mobile):
                raise ValueError(
                    "Distance constraints may reference only mobile particles."
                )
        layout = _components(
            constraint_indices.reshape((-1, 2)), plan.maximum_component_atoms
        )
        self.plan = plan
        self.system = system
        (
            self.component_atoms,
            self.component_atom_mask,
            self.component_rows,
            self.component_row_mask,
            self.component_left,
            self.component_right,
        ) = (jnp.asarray(value) for value in layout)
        self.independent_constraint_count = int(constraint_indices.shape[0])
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-distance-constraints",
                "plan": plan.plan_id,
                "system": system.prepared_id,
                "topology": system.topology.topology_id,
                "rank": "constant-full-row-rank-certified-per-geometry",
            }
        )

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical prepared-constraint digest."""
        return self.prepared_identity.fingerprint()

    @property
    def layout_id(self) -> str:
        """Content-free executable layout of these prepared constraints."""
        return canonical_fingerprint(
            {
                "kind": "prepared-distance-constraints-layout",
                "plan": self.plan.plan_id,
                "system": self.system.layout_id,
                "constraints": self.independent_constraint_count,
                "components": list(self.component_atoms.shape),
                "component_rows": list(self.component_rows.shape),
            }
        )

    def rank_evidence(self, positions: ArrayLike, /) -> ConstraintRankEvidence:
        """Per-component mass-weighted Jacobian rank at explicit ``positions``."""
        position = jnp.asarray(positions)
        indices = self.system.topology.constraint_indices
        if indices.shape[0] == 0:
            return ConstraintRankEvidence(
                jnp.zeros((0,), dtype=jnp.int32),
                jnp.asarray(jnp.inf, dtype=position.dtype),
                jnp.asarray(True),
            )
        weights = jnp.sqrt(self.system.inverse_masses.astype(position.dtype))
        atoms = self.component_atoms
        width = atoms.shape[1]
        rows = self.component_rows
        displacement = position[indices[rows, 0]] - position[indices[rows, 1]]
        length = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
        direction = jnp.where(
            (length > 0.0)[..., None],
            displacement / jnp.where(length > 0.0, length, 1.0)[..., None],
            0.0,
        )
        direction = jnp.where(self.component_row_mask[..., None], direction, 0.0)
        left = jax.nn.one_hot(self.component_left, width, dtype=position.dtype)
        right = jax.nn.one_hot(self.component_right, width, dtype=position.dtype)
        local_weights = jnp.where(self.component_atom_mask, weights[atoms], 0.0)
        coefficient = (left - right) * local_weights[:, None, :]
        jacobian = (coefficient[..., None] * direction[..., None, :]).reshape(
            rows.shape + (3 * width,)
        )
        _, singular, _ = economy_svd(jacobian)
        largest = jnp.max(singular, axis=-1, keepdims=True)
        relative = singular / jnp.where(largest > 0.0, largest, 1.0)
        rank = jnp.sum(relative > self.plan.rank_tolerance, axis=-1).astype(jnp.int32)
        required = jnp.sum(self.component_row_mask, axis=-1).astype(jnp.int32)
        padded = jnp.arange(singular.shape[-1])[None, :] < required[:, None]
        smallest = jnp.min(jnp.where(padded, relative, jnp.inf))
        regular = jnp.all(rank == required) & jnp.all(jnp.isfinite(singular))
        return ConstraintRankEvidence(rank, smallest, regular)

    def project_positions(
        self,
        previous_positions: ArrayLike,
        proposed_positions: ArrayLike,
        momenta: ArrayLike,
        /,
    ) -> ConstraintProjection:
        previous = jnp.asarray(previous_positions)
        proposed = jnp.asarray(proposed_positions, dtype=previous.dtype)
        momentum = jnp.asarray(momenta, dtype=previous.dtype)
        indices = self.system.topology.constraint_indices
        targets = self.system.topology.constraint_distances.astype(previous.dtype)
        count = indices.shape[0]
        if count == 0:
            zero = jnp.zeros((), dtype=previous.dtype)
            return ConstraintProjection(
                proposed,
                momentum,
                jnp.zeros((0,), dtype=previous.dtype),
                zero,
                zero,
                jnp.zeros((), dtype=jnp.int32),
                jnp.asarray(True),
            )
        inverse_mass = self.system.inverse_masses.astype(previous.dtype)
        left = indices[:, 0]
        right = indices[:, 1]

        def iteration(_: int | Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            position, multipliers = carry
            displacement = position[left] - position[right]
            squared = jnp.sum(displacement * displacement, axis=-1)
            residual = squared - targets * targets
            denominator = 2.0 * (inverse_mass[left] + inverse_mass[right]) * squared
            valid = denominator > 0.0
            increment = jnp.where(valid, -residual / denominator, 0.0)
            correction = increment[:, None] * displacement
            delta = jnp.zeros_like(position)
            delta = delta.at[left].add(inverse_mass[left, None] * correction)
            delta = delta.at[right].add(-inverse_mass[right, None] * correction)
            position = position + delta
            return position, multipliers + increment

        positions, multipliers = jax.lax.fori_loop(
            0,
            self.plan.maximum_iterations,
            iteration,
            (proposed, jnp.zeros((count,), dtype=previous.dtype)),
        )
        displacement = positions[left] - positions[right]
        distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
        position_residual = jnp.max(jnp.abs(distance - targets))
        projected_momenta, velocity_residual = self.project_momenta(positions, momentum)
        successful = (
            jnp.all(jnp.isfinite(positions))
            & jnp.all(jnp.isfinite(projected_momenta))
            & (position_residual <= self.plan.tolerance)
            & (velocity_residual <= self.plan.tolerance)
            & self.rank_evidence(positions).regular
        )
        return ConstraintProjection(
            positions,
            projected_momenta,
            multipliers,
            position_residual,
            velocity_residual,
            jnp.asarray(self.plan.maximum_iterations, dtype=jnp.int32),
            successful,
        )

    def project_momenta(
        self, positions: ArrayLike, momenta: ArrayLike, /
    ) -> tuple[Array, Array]:
        position = jnp.asarray(positions)
        momentum = jnp.asarray(momenta, dtype=position.dtype)
        indices = self.system.topology.constraint_indices
        count = indices.shape[0]
        if count == 0:
            return momentum, jnp.zeros((), dtype=position.dtype)
        inverse_mass = self.system.inverse_masses.astype(position.dtype)
        left = indices[:, 0]
        right = indices[:, 1]

        def iteration(_: int | Array, value: Array) -> Array:
            displacement = position[left] - position[right]
            relative_velocity = (
                value[left] * inverse_mass[left, None]
                - value[right] * inverse_mass[right, None]
            )
            squared = jnp.sum(displacement * displacement, axis=-1)
            denominator = (inverse_mass[left] + inverse_mass[right]) * squared
            increment = jnp.where(
                denominator > 0.0,
                -jnp.sum(displacement * relative_velocity, axis=-1) / denominator,
                0.0,
            )
            correction = increment[:, None] * displacement
            delta = jnp.zeros_like(value)
            delta = delta.at[left].add(correction)
            delta = delta.at[right].add(-correction)
            return value + delta

        projected = jax.lax.fori_loop(
            0, self.plan.maximum_iterations, iteration, momentum
        )
        displacement = position[left] - position[right]
        relative_velocity = (
            projected[left] * inverse_mass[left, None]
            - projected[right] * inverse_mass[right, None]
        )
        residual = jnp.max(jnp.abs(jnp.sum(displacement * relative_velocity, axis=-1)))
        return projected, residual


__all__ = [
    "ConstraintProjection",
    "ConstraintRankEvidence",
    "DistanceConstraintPlan",
    "PreparedDistanceConstraints",
]
