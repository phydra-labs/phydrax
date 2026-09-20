#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Recoupled intertwiners, geometric operators, coherent states, and graph moves."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...operators.quantum.lattice import (
    prepare_su2_sector_basis,
    PreparedSU2SectorBasis,
    SU2CouplingTreePlan,
)
from ._graph import (
    prepare_spin_network,
    PreparedSpinNetwork,
    SpinNetworkEdge,
    SpinNetworkGraphPlan,
)


class SpinNetworkCouplingTree(StrictModule):
    """One ordered left-associated chart in a recoupling-complete vertex atlas."""

    vertex: str = eqx.field(static=True)
    edge_order: tuple[str, ...] = eqx.field(static=True)
    tree_label: str = eqx.field(static=True)
    tree_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertex: str,
        edge_order: Sequence[str],
        tree_label: str,
        /,
    ):
        vertex_ = str(vertex).strip()
        order = tuple(str(value).strip() for value in edge_order)
        label = str(tree_label).strip()
        if (
            not vertex_
            or len(order) < 2
            or any(not value for value in order)
            or len(set(order)) != len(order)
            or not label
        ):
            raise ValueError("Spin-network coupling tree identity is invalid.")
        self.vertex = vertex_
        self.edge_order = order
        self.tree_label = label
        self.tree_id = canonical_fingerprint(
            {
                "kind": "spin-network-coupling-tree",
                "vertex": vertex_,
                "edge_order": order,
                "tree_label": label,
            }
        )


class RecoupledVertexBasis(StrictModule):
    tree: SpinNetworkCouplingTree = eqx.field(static=True)
    basis: PreparedSU2SectorBasis
    canonical_edge_order: tuple[str, ...] = eqx.field(static=True)
    canonical_transform: Array
    basis_id: str = eqx.field(static=True)


def prepare_recoupled_vertex_basis(
    graph: SpinNetworkGraphPlan,
    tree: SpinNetworkCouplingTree,
    /,
) -> RecoupledVertexBasis:
    if not isinstance(graph, SpinNetworkGraphPlan) or not isinstance(
        tree, SpinNetworkCouplingTree
    ):
        raise TypeError("graph and tree must be typed spin-network plans.")
    order_by_vertex = dict(graph.vertex_edge_order)
    if tree.vertex not in order_by_vertex or set(tree.edge_order) != set(
        order_by_vertex[tree.vertex]
    ):
        raise ValueError("Coupling tree does not cover one graph vertex exactly.")
    edge_lookup = {value.label: value for value in graph.edges}
    spins = tuple(edge_lookup[label].twice_spin for label in tree.edge_order)
    basis = prepare_su2_sector_basis(
        SU2CouplingTreePlan(
            tuple(f"{tree.vertex}:{label}" for label in tree.edge_order),
            spins,
            0,
            graph.resources,
        )
    )
    canonical_order = tuple(order_by_vertex[tree.vertex])
    canonical_dimensions = tuple(
        edge_lookup[label].twice_spin + 1 for label in canonical_order
    )
    source_dimensions = tuple(
        edge_lookup[label].twice_spin + 1 for label in tree.edge_order
    )
    source_transform = np.asarray(basis.transform)
    canonical_transform = np.zeros(
        (source_transform.shape[0], math.prod(canonical_dimensions)),
        dtype=np.complex128,
    )
    source_positions = {label: index for index, label in enumerate(tree.edge_order)}
    for source_index, coordinate in enumerate(np.ndindex(*source_dimensions)):
        canonical_coordinate = tuple(
            coordinate[source_positions[label]] for label in canonical_order
        )
        canonical_index = np.ravel_multi_index(canonical_coordinate, canonical_dimensions)
        canonical_transform[:, canonical_index] = source_transform[:, source_index]
    basis_id = canonical_fingerprint(
        {
            "kind": "recoupled-vertex-basis",
            "graph": graph.graph_id,
            "tree": tree.tree_id,
            "basis": basis.prepared_id,
            "canonical_transform": array_tree_fingerprint(canonical_transform),
        }
    )
    return RecoupledVertexBasis(
        tree=tree,
        basis=basis,
        canonical_edge_order=canonical_order,
        canonical_transform=jnp.asarray(canonical_transform),
        basis_id=basis_id,
    )


class SpinNetworkRecouplingEvidence(StrictModule):
    matrix: Array
    source_orthonormality_residual: Array
    target_orthonormality_residual: Array
    unitarity_residual: Array
    accepted: Array
    evidence_id: str = eqx.field(static=True)


def spin_network_recoupling_matrix(
    source: RecoupledVertexBasis,
    target: RecoupledVertexBasis,
    /,
    *,
    tolerance: float = 1e-10,
) -> SpinNetworkRecouplingEvidence:
    if (
        source.tree.vertex != target.tree.vertex
        or source.canonical_edge_order != target.canonical_edge_order
    ):
        raise ValueError(
            "Recoupling bases must describe the same ordered vertex support."
        )
    left = source.canonical_transform
    right = target.canonical_transform
    matrix = right @ jnp.conj(left.T)
    source_residual = jnp.linalg.norm(left @ jnp.conj(left.T) - jnp.eye(left.shape[0]))
    target_residual = jnp.linalg.norm(right @ jnp.conj(right.T) - jnp.eye(right.shape[0]))
    unitarity = jnp.linalg.norm(matrix @ jnp.conj(matrix.T) - jnp.eye(matrix.shape[0]))
    accepted = (
        (source_residual <= tolerance)
        & (target_residual <= tolerance)
        & (unitarity <= tolerance)
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "spin-network-recoupling-evidence",
            "source": source.basis_id,
            "target": target.basis_id,
            "tolerance": float(tolerance),
        }
    )
    return SpinNetworkRecouplingEvidence(
        matrix=matrix,
        source_orthonormality_residual=source_residual,
        target_orthonormality_residual=target_residual,
        unitarity_residual=unitarity,
        accepted=accepted,
        evidence_id=evidence_id,
    )


def _spin_generators(twice_spin: int, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dimension = twice_spin + 1
    spin = 0.5 * twice_spin
    magnetic = np.arange(dimension) - spin
    raising = np.zeros((dimension, dimension), dtype=np.complex128)
    for index, value in enumerate(magnetic[:-1]):
        raising[index + 1, index] = math.sqrt((spin - value) * (spin + value + 1.0))
    lowering = raising.conj().T
    return 0.5 * (raising + lowering), (raising - lowering) / (2.0j), np.diag(magnetic)


def _local_product_operator(
    spins: tuple[int, ...],
    site: int,
    local: np.ndarray,
    /,
) -> np.ndarray:
    factors = [np.eye(spin + 1, dtype=np.complex128) for spin in spins]
    factors[site] = local
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


class SpinNetworkGeometricOperators(StrictModule):
    angle_operators: Array
    volume_operator: Array
    length_operators: Array
    flux_norm_operators: Array
    angle_pairs: tuple[tuple[str, str], ...] = eqx.field(static=True)
    edge_labels: tuple[str, ...] = eqx.field(static=True)
    hermiticity_residuals: Array
    basis_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def prepare_spin_network_geometric_operators(
    graph: SpinNetworkGraphPlan,
    basis: RecoupledVertexBasis,
    /,
) -> SpinNetworkGeometricOperators:
    """Project angle, oriented volume, length, and flux-norm operators."""

    edge_lookup = {value.label: value for value in graph.edges}
    labels = basis.canonical_edge_order
    spins = tuple(edge_lookup[label].twice_spin for label in labels)
    generators = tuple(_spin_generators(value) for value in spins)
    transform = np.asarray(basis.canonical_transform)
    product_generators: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for site in range(len(labels)):
        local = generators[site]
        product = (
            _local_product_operator(spins, site, local[0]),
            _local_product_operator(spins, site, local[1]),
            _local_product_operator(spins, site, local[2]),
        )
        product_generators.append(product)
    pairs = tuple(itertools.combinations(range(len(labels)), 2))
    angles = []
    angle_labels = []
    for first, second in pairs:
        dot = sum(
            product_generators[first][axis] @ product_generators[second][axis]
            for axis in range(3)
        )
        first_norm = 0.25 * spins[first] * (spins[first] + 2)
        second_norm = 0.25 * spins[second] * (spins[second] + 2)
        denominator = math.sqrt(first_norm * second_norm)
        angle = (
            transform @ (dot / denominator if denominator else dot) @ transform.conj().T
        )
        angles.append(angle)
        angle_labels.append((labels[first], labels[second]))
    volume_seed = np.zeros(
        (math.prod(value + 1 for value in spins),) * 2,
        dtype=np.complex128,
    )
    if len(labels) >= 3:
        for permutation in itertools.permutations(range(3)):
            inversions = sum(
                permutation[left] > permutation[right]
                for left in range(3)
                for right in range(left + 1, 3)
            )
            sign = -1.0 if inversions % 2 else 1.0
            volume_seed += sign * (
                product_generators[0][permutation[0]]
                @ product_generators[1][permutation[1]]
                @ product_generators[2][permutation[2]]
            )
    volume_projected = transform @ volume_seed @ transform.conj().T
    volume_projected = 0.5 * (volume_projected + volume_projected.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(volume_projected)
    volume = eigenvectors @ np.diag(np.sqrt(np.abs(eigenvalues))) @ eigenvectors.conj().T
    lengths = []
    flux_norms = []
    for spin in spins:
        eigenvalue = math.sqrt(0.25 * spin * (spin + 2))
        lengths.append(eigenvalue * np.eye(transform.shape[0]))
        flux_norms.append(eigenvalue * np.eye(transform.shape[0]))
    operator_tables = (*angles, volume, *lengths, *flux_norms)
    hermiticity = np.asarray(
        [np.linalg.norm(value - value.conj().T) for value in operator_tables]
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "spin-network-geometric-operators",
            "graph": graph.graph_id,
            "basis": basis.basis_id,
            "angle_pairs": angle_labels,
            "operators": [array_tree_fingerprint(value) for value in operator_tables],
        }
    )
    return SpinNetworkGeometricOperators(
        angle_operators=jnp.asarray(angles),
        volume_operator=jnp.asarray(volume),
        length_operators=jnp.asarray(lengths),
        flux_norm_operators=jnp.asarray(flux_norms),
        angle_pairs=tuple(angle_labels),
        edge_labels=labels,
        hermiticity_residuals=jnp.asarray(hermiticity),
        basis_id=basis.basis_id,
        evidence_id=evidence_id,
    )


class CoherentSpinNetworkEvidence(StrictModule):
    amplitudes: Array
    norm: Array
    closure_vector: Array
    closure_residual: Array
    normalized: Array
    basis_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def coherent_spin_network_state(
    graph: SpinNetworkGraphPlan,
    basis: RecoupledVertexBasis,
    directions: Mapping[str, ArrayLike],
    /,
    *,
    closure_tolerance: float = 1e-8,
) -> CoherentSpinNetworkEvidence:
    """Project Livine–Speziale edge coherent states into one intertwiner basis."""

    edge_lookup = {value.label: value for value in graph.edges}
    if set(directions) != set(basis.canonical_edge_order):
        raise ValueError("One coherent direction is required per incident edge.")
    local_states = []
    closure = np.zeros(3, dtype=np.float64)
    for label in basis.canonical_edge_order:
        edge = edge_lookup[label]
        direction = np.asarray(directions[label], dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("Coherent directions must be finite three-vectors.")
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            raise ValueError("Coherent directions must be nonzero.")
        direction /= norm
        theta = math.acos(np.clip(direction[2], -1.0, 1.0))
        phi = math.atan2(direction[1], direction[0])
        twice_spin = edge.twice_spin
        spin = 0.5 * twice_spin
        coefficients = []
        for ordinal in range(twice_spin + 1):
            magnetic = -spin + ordinal
            coefficients.append(
                math.sqrt(math.comb(twice_spin, ordinal))
                * math.cos(theta / 2.0) ** (spin + magnetic)
                * math.sin(theta / 2.0) ** (spin - magnetic)
                * np.exp(-1j * (spin - magnetic) * phi)
            )
        local_states.append(np.asarray(coefficients))
        orientation = 1.0 if edge.source == basis.tree.vertex else -1.0
        closure += orientation * spin * direction
    product = local_states[0]
    for state in local_states[1:]:
        product = np.kron(product, state)
    amplitudes = np.asarray(basis.canonical_transform) @ product
    norm = float(np.vdot(amplitudes, amplitudes).real)
    if norm > 0.0:
        amplitudes /= math.sqrt(norm)
    closure_residual = float(np.linalg.norm(closure))
    evidence_id = canonical_fingerprint(
        {
            "kind": "coherent-spin-network-evidence",
            "graph": graph.graph_id,
            "basis": basis.basis_id,
            "directions": {
                label: array_tree_fingerprint(np.asarray(directions[label]))
                for label in sorted(directions)
            },
            "closure_tolerance": float(closure_tolerance),
        }
    )
    return CoherentSpinNetworkEvidence(
        amplitudes=jnp.asarray(amplitudes),
        norm=jnp.asarray(norm),
        closure_vector=jnp.asarray(closure),
        closure_residual=jnp.asarray(closure_residual),
        normalized=jnp.asarray(norm > 0.0 and closure_residual <= closure_tolerance),
        basis_id=basis.basis_id,
        evidence_id=evidence_id,
    )


@dataclass(frozen=True, slots=True)
class SpinNetworkRefinement:
    graph: SpinNetworkGraphPlan
    prepared: PreparedSpinNetwork
    embedding: np.ndarray
    source_graph_id: str
    refinement_id: str


def refine_spin_network_edge(
    graph: SpinNetworkGraphPlan,
    edge_label: str,
    new_vertex: str,
    /,
    *,
    immirzi_parameter: float,
    planck_length_squared: float,
) -> SpinNetworkRefinement:
    """Split one edge through an exact bivalent identity intertwiner."""

    target = str(edge_label)
    matches = tuple(value for value in graph.edges if value.label == target)
    if len(matches) != 1 or new_vertex in graph.vertices:
        raise ValueError("Refinement edge is absent/ambiguous or vertex already exists.")
    edge = matches[0]
    left = SpinNetworkEdge(f"{target}:left", edge.source, new_vertex, edge.twice_spin)
    right = SpinNetworkEdge(f"{target}:right", new_vertex, edge.target, edge.twice_spin)
    edges = tuple(value for value in graph.edges if value.label != target) + (left, right)
    orders = {}
    for vertex, order in graph.vertex_edge_order:
        replacement = left.label if vertex == edge.source else right.label
        orders[vertex] = tuple(
            replacement if value == target else value for value in order
        )
    orders[new_vertex] = (left.label, right.label)
    refined_graph = SpinNetworkGraphPlan(
        graph.vertices + (new_vertex,),
        edges,
        orders,
        graph.resources,
    )
    prepared = prepare_spin_network(
        refined_graph,
        immirzi_parameter=immirzi_parameter,
        planck_length_squared=planck_length_squared,
    )
    source_prepared = prepare_spin_network(
        graph,
        immirzi_parameter=immirzi_parameter,
        planck_length_squared=planck_length_squared,
    )
    if prepared.basis_dimension != source_prepared.basis_dimension:
        raise ValueError("Bivalent refinement changed the global intertwiner dimension.")
    embedding = np.eye(prepared.basis_dimension, dtype=np.complex128)
    refinement_id = canonical_fingerprint(
        {
            "kind": "spin-network-edge-refinement",
            "source": graph.graph_id,
            "target": refined_graph.graph_id,
            "edge": target,
            "new_vertex": new_vertex,
        }
    )
    return SpinNetworkRefinement(
        refined_graph,
        prepared,
        embedding,
        graph.graph_id,
        refinement_id,
    )


class RegulatedHamiltonianConstraintPlan(StrictModule):
    move_matrices: Array
    lapse_values: Array
    regularization_id: str = eqx.field(static=True)
    ordering_id: str = eqx.field(static=True)
    graph_change_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        move_matrices: ArrayLike,
        lapse_values: ArrayLike,
        graph_change_ids: Sequence[str],
        /,
        *,
        regularization_id: str,
        ordering_id: str,
    ):
        moves = np.asarray(move_matrices, dtype=np.complex128)
        lapse = np.asarray(lapse_values, dtype=np.float64)
        changes = tuple(str(value).strip() for value in graph_change_ids)
        regularization = str(regularization_id).strip()
        ordering = str(ordering_id).strip()
        if moves.ndim != 3 or moves.shape[1] != moves.shape[2] or not moves.shape[0]:
            raise ValueError(
                "Hamiltonian move matrices must have shape (moves, basis, basis)."
            )
        if lapse.shape != (moves.shape[0],) or len(changes) != moves.shape[0]:
            raise ValueError(
                "Hamiltonian lapse and graph-change rosters must match moves."
            )
        if any(not value for value in changes) or not regularization or not ordering:
            raise ValueError("Hamiltonian regularization identities must be explicit.")
        content = {
            "kind": "regulated-hamiltonian-constraint-plan",
            "move_matrices": array_tree_fingerprint(moves),
            "lapse_values": array_tree_fingerprint(lapse),
            "graph_change_ids": changes,
            "regularization_id": regularization,
            "ordering_id": ordering,
        }
        self.move_matrices = jnp.asarray(moves)
        self.lapse_values = jnp.asarray(lapse)
        self.regularization_id = regularization
        self.ordering_id = ordering
        self.graph_change_ids = changes
        self.plan_id = canonical_fingerprint(content)


class HamiltonianConstraintEvidence(StrictModule):
    matrix: Array
    hermiticity_residual: Array
    commutator_residual: Array
    finite: Array
    claim: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def evaluate_regulated_hamiltonian_constraint(
    plan: RegulatedHamiltonianConstraintPlan,
    /,
    *,
    comparison_lapse_values: ArrayLike | None = None,
) -> HamiltonianConstraintEvidence:
    if not isinstance(plan, RegulatedHamiltonianConstraintPlan):
        raise TypeError("plan must be RegulatedHamiltonianConstraintPlan.")
    matrix = ein.contract("m,mij->ij", plan.lapse_values, plan.move_matrices)
    hermiticity = jnp.linalg.norm(matrix - jnp.conj(matrix.T))
    if comparison_lapse_values is None:
        commutator = jnp.asarray(0.0)
    else:
        comparison = jnp.asarray(comparison_lapse_values, dtype=plan.lapse_values.dtype)
        if comparison.shape != plan.lapse_values.shape:
            raise ValueError("Comparison lapse roster has the wrong shape.")
        other = ein.contract("m,mij->ij", comparison, plan.move_matrices)
        commutator = jnp.linalg.norm(matrix @ other - other @ matrix)
    finite = jnp.all(jnp.isfinite(matrix))
    evidence_id = canonical_fingerprint(
        {
            "kind": "hamiltonian-constraint-evidence",
            "plan": plan.plan_id,
            "comparison": None
            if comparison_lapse_values is None
            else array_tree_fingerprint(np.asarray(comparison_lapse_values)),
        }
    )
    return HamiltonianConstraintEvidence(
        matrix=matrix,
        hermiticity_residual=hermiticity,
        commutator_residual=commutator,
        finite=finite,
        claim="finite-regulated-graph-dependent-constraint-no-unique-dynamics-claim",
        evidence_id=evidence_id,
    )


__all__ = [
    "CoherentSpinNetworkEvidence",
    "HamiltonianConstraintEvidence",
    "RecoupledVertexBasis",
    "RegulatedHamiltonianConstraintPlan",
    "SpinNetworkCouplingTree",
    "SpinNetworkGeometricOperators",
    "SpinNetworkRecouplingEvidence",
    "SpinNetworkRefinement",
    "coherent_spin_network_state",
    "evaluate_regulated_hamiltonian_constraint",
    "prepare_recoupled_vertex_basis",
    "prepare_spin_network_geometric_operators",
    "refine_spin_network_edge",
    "spin_network_recoupling_matrix",
]
