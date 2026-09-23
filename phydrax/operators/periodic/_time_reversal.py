#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Time-reversal evidence and rank-two Wilson-flow Z2 topology."""

from __future__ import annotations

from math import isfinite, pi

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import ReciprocalMeshPlan
from ._orbital import PreparedPeriodicOrbitalPencil
from ._topology import PeriodicOverlapBundle


def _circular_delta(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.mod(value - reference + 0.5, 1.0) - 0.5


class PeriodicTimeReversalEvidence(StrictModule, NonTrainableState):
    theta_square_residual: Array
    hamiltonian_residual: Array
    overlap_residual: Array
    maximum_pairing_distance: Array
    accepted: Array
    basis_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PeriodicTimeReversalPlan(StrictModule, NonTrainableState):
    pencil: PreparedPeriodicOrbitalPencil
    mesh: ReciprocalMeshPlan
    unitary: Array
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pencil: PreparedPeriodicOrbitalPencil,
        mesh: ReciprocalMeshPlan,
        unitary: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-8,
    ):
        if not isinstance(pencil, PreparedPeriodicOrbitalPencil):
            raise TypeError("pencil must be PreparedPeriodicOrbitalPencil.")
        if not isinstance(mesh, ReciprocalMeshPlan):
            raise TypeError("mesh must be ReciprocalMeshPlan.")
        mesh.require_cell(pencil.plan.basis.cell)
        value = np.asarray(unitary)
        count = pencil.plan.basis.orbital_count
        tolerance_ = float(tolerance)
        if (
            value.shape != (count, count)
            or np.any(~np.isfinite(value))
            or not isfinite(tolerance_)
            or tolerance_ <= 0.0
        ):
            raise ValueError("Time-reversal unitary or tolerance is invalid.")
        unitarity = np.max(np.abs(np.conj(value.T) @ value - np.eye(count)))
        if unitarity > tolerance_:
            raise ValueError("Time-reversal unitary is not unitary within tolerance.")
        self.pencil = pencil
        self.mesh = mesh
        self.unitary = jnp.asarray(value, dtype=jnp.complex128)
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-time-reversal-plan",
                "pencil": pencil.prepared_id,
                "mesh": mesh.mesh_id,
                "unitary": array_tree_fingerprint(value),
                "tolerance": tolerance_,
            }
        )

    def evaluate(self, /) -> PeriodicTimeReversalEvidence:
        points = np.mod(np.asarray(self.mesh.fractional_points), 1.0)
        partner = np.mod(-points, 1.0)
        pair_indices = []
        pair_distances = []
        for point in partner:
            differences = _circular_delta(points, point)
            distances = np.max(np.abs(differences), axis=1)
            index = int(np.argmin(distances))
            pair_indices.append(index)
            pair_distances.append(float(distances[index]))
        evaluation = self.pencil.evaluate(self.mesh.fractional_points)
        hamiltonian = np.asarray(evaluation.hamiltonians)
        overlap = np.asarray(evaluation.overlaps)
        unitary = np.asarray(self.unitary)
        transformed_h = unitary[None] @ np.conj(hamiltonian) @ np.conj(unitary.T)[None]
        transformed_s = unitary[None] @ np.conj(overlap) @ np.conj(unitary.T)[None]
        paired_h = hamiltonian[np.asarray(pair_indices)]
        paired_s = overlap[np.asarray(pair_indices)]
        h_scale = max(float(np.max(np.abs(hamiltonian), initial=0.0)), 1.0)
        s_scale = max(float(np.max(np.abs(overlap), initial=0.0)), 1.0)
        h_residual = float(np.max(np.abs(paired_h - transformed_h)) / h_scale)
        s_residual = float(np.max(np.abs(paired_s - transformed_s)) / s_scale)
        theta_square = float(
            np.max(np.abs(unitary @ np.conj(unitary) + np.eye(unitary.shape[0])))
        )
        pairing = max(pair_distances, default=np.inf)
        accepted = bool(
            theta_square <= self.tolerance
            and h_residual <= self.tolerance
            and s_residual <= self.tolerance
            and pairing <= self.tolerance
            and bool(evaluation.successful)
        )
        evidence_id = canonical_fingerprint(
            {
                "kind": "periodic-time-reversal-evidence",
                "plan": self.plan_id,
                "theta_square_residual": theta_square,
                "hamiltonian_residual": h_residual,
                "overlap_residual": s_residual,
                "maximum_pairing_distance": pairing,
                "accepted": accepted,
            }
        )
        return PeriodicTimeReversalEvidence(
            jnp.asarray(theta_square),
            jnp.asarray(h_residual),
            jnp.asarray(s_residual),
            jnp.asarray(pairing),
            jnp.asarray(accepted),
            self.pencil.plan.basis.basis_id,
            evidence_id,
        )


class PeriodicZ2Result(StrictModule, NonTrainableState):
    transverse_coordinates: Array
    wilson_matrices: Array
    eigenphases: Array
    tracked_phases: Array
    relative_winding: Array
    invariant: Array
    endpoint_kramers_residual: Array
    minimum_link_singular_value: Array
    minimum_direct_gap: Array
    successful: Array
    bundle_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class PeriodicZ2Plan(StrictModule, NonTrainableState):
    """Wilson-flow Z2 plan for one occupied rank-two Kramers pair."""

    bundle: PeriodicOverlapBundle
    time_reversal: PeriodicTimeReversalEvidence
    loop_axis: int = eqx.field(static=True)
    transverse_axis: int = eqx.field(static=True)
    endpoint_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bundle: PeriodicOverlapBundle,
        time_reversal: PeriodicTimeReversalEvidence,
        /,
        *,
        loop_axis: int = 0,
        transverse_axis: int = 1,
        endpoint_tolerance: float = 1.0e-5,
    ):
        if not isinstance(bundle, PeriodicOverlapBundle):
            raise TypeError("bundle must be PeriodicOverlapBundle.")
        if not isinstance(time_reversal, PeriodicTimeReversalEvidence):
            raise TypeError("time_reversal must be PeriodicTimeReversalEvidence.")
        mesh = bundle.connectivity.plan.mesh
        axes = (int(loop_axis), int(transverse_axis))
        tolerance = float(endpoint_tolerance)
        if (
            mesh.rank != 2
            or bundle.manifold.dimension != 2
            or set(axes) != {0, 1}
            or mesh.mesh_shape[axes[1]] % 2 != 0
            or not isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError(
                "Z2 evaluation requires a two-dimensional even mesh and one rank-two occupied Kramers pair."
            )
        if bundle.manifold.spectrum.basis_id != time_reversal.basis_id:
            raise ValueError("Z2 bundle and time-reversal evidence use different bases.")
        self.bundle = bundle
        self.time_reversal = time_reversal
        self.loop_axis = axes[0]
        self.transverse_axis = axes[1]
        self.endpoint_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-z2-plan",
                "bundle": bundle.bundle_id,
                "time_reversal": time_reversal.evidence_id,
                "axes": axes,
                "endpoint_tolerance": tolerance,
            }
        )

    def _cycles(self) -> tuple[tuple[int, ...], ...]:
        connectivity = self.bundle.connectivity.plan
        mesh = connectivity.mesh
        indices = np.asarray(mesh.mesh_indices)
        lookup = {tuple(index): ordinal for ordinal, index in enumerate(indices)}
        source = np.asarray(connectivity.source_indices)
        target = np.asarray(connectivity.target_indices)
        shifts = np.asarray(connectivity.reciprocal_shifts)
        loop_size = mesh.mesh_shape[self.loop_axis]
        transverse_size = mesh.mesh_shape[self.transverse_axis]
        cycles = []
        for transverse in range(transverse_size // 2 + 1):
            edges = []
            for loop in range(loop_size):
                coordinate = [0, 0]
                coordinate[self.loop_axis] = loop
                coordinate[self.transverse_axis] = transverse
                next_coordinate = coordinate.copy()
                next_coordinate[self.loop_axis] = (loop + 1) % loop_size
                source_index = lookup[tuple(coordinate)]
                target_index = lookup[tuple(next_coordinate)]
                expected_shift = np.zeros((2,), dtype=np.int32)
                if loop + 1 == loop_size:
                    expected_shift[self.loop_axis] = 1
                matches = np.flatnonzero(
                    (source == source_index)
                    & (target == target_index)
                    & np.all(shifts == expected_shift[None, :], axis=1)
                )
                if matches.size != 1:
                    raise ValueError(
                        "Regular connectivity does not contain a unique Wilson edge."
                    )
                edges.append(int(matches[0]))
            cycles.append(tuple(edges))
        return tuple(cycles)

    def evaluate(self, /) -> PeriodicZ2Result:
        links = np.asarray(self.bundle.normalized_links)
        cycles = self._cycles()
        matrices = []
        phases = []
        for cycle in cycles:
            loop = np.eye(2, dtype=links.dtype)
            for edge in cycle:
                loop = loop @ links[edge]
            matrices.append(loop)
            phases.append(
                np.sort(np.mod(np.angle(np.linalg.eigvals(loop)) / (2.0 * pi), 1.0))
            )
        phase_array = np.asarray(phases)
        tracked = np.empty_like(phase_array)
        tracked[0] = phase_array[0]
        for index in range(1, phase_array.shape[0]):
            straight = np.sum(
                np.abs(_circular_delta(phase_array[index], tracked[index - 1]))
            )
            swapped_values = phase_array[index, ::-1]
            swapped = np.sum(np.abs(_circular_delta(swapped_values, tracked[index - 1])))
            selected = phase_array[index] if straight <= swapped else swapped_values
            tracked[index] = tracked[index - 1] + _circular_delta(
                selected, np.mod(tracked[index - 1], 1.0)
            )
        endpoint = max(
            abs(float(_circular_delta(phase_array[0, 0], phase_array[0, 1]))),
            abs(float(_circular_delta(phase_array[-1, 0], phase_array[-1, 1]))),
        )
        displacement = tracked[-1] - tracked[0]
        relative = int(np.rint(displacement[0] - displacement[1]))
        invariant = abs(relative) % 2
        successful = bool(
            bool(self.time_reversal.accepted)
            and self.bundle.resolved
            and endpoint <= self.endpoint_tolerance
            and np.all(np.isfinite(tracked))
        )
        matrices_array = np.asarray(matrices)
        result_id = canonical_fingerprint(
            {
                "kind": "periodic-z2-result",
                "plan": self.plan_id,
                "invariant": invariant,
                "arrays": array_tree_fingerprint(
                    {
                        "wilson_matrices": matrices_array,
                        "eigenphases": phase_array,
                        "tracked_phases": tracked,
                    }
                ),
            }
        )
        mesh = self.bundle.connectivity.plan.mesh
        transverse = (
            np.arange(len(cycles), dtype=np.float64)
            / mesh.mesh_shape[self.transverse_axis]
        )
        return PeriodicZ2Result(
            jnp.asarray(transverse),
            jnp.asarray(matrices_array),
            jnp.asarray(phase_array),
            jnp.asarray(tracked),
            jnp.asarray(relative, dtype=jnp.int32),
            jnp.asarray(invariant, dtype=jnp.int32),
            jnp.asarray(endpoint),
            self.bundle.minimum_singular_value,
            self.bundle.manifold.minimum_direct_gap,
            jnp.asarray(successful),
            self.bundle.bundle_id,
            result_id,
        )


__all__ = [
    "PeriodicTimeReversalEvidence",
    "PeriodicTimeReversalPlan",
    "PeriodicZ2Plan",
    "PeriodicZ2Result",
]
