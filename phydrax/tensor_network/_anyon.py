#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod, sqrt

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule


class FusionCoherenceEvidence(StrictModule):
    fusion_associativity_residual: Array
    quantum_dimension_residual: Array
    f_unitarity_residual: Array
    r_unitarity_residual: Array
    pentagon_residual: Array
    hexagon_residual: Array
    tolerance: Array
    coherent: Array
    category_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class FiniteFusionCategory(StrictModule):
    """Finite multiplicity-free unitary braided fusion data in one fixed gauge."""

    labels: tuple[str, ...] = eqx.field(static=True)
    unit_index: int = eqx.field(static=True)
    dual_indices: tuple[int, ...] = eqx.field(static=True)
    fusion_multiplicities: Array
    quantum_dimensions: Array
    f_symbols: Array
    r_symbols: Array
    coherence: FusionCoherenceEvidence
    category_id: str = eqx.field(static=True)

    def __init__(
        self,
        labels: Sequence[str],
        unit_label: str,
        dual_labels: Sequence[str],
        fusion_multiplicities: ArrayLike,
        quantum_dimensions: ArrayLike,
        f_symbols: ArrayLike,
        r_symbols: ArrayLike,
        /,
        *,
        coherence_tolerance: float = 1e-9,
        maximum_simple_objects: int = 8,
    ):
        labels_ = tuple(str(label) for label in labels)
        duals = tuple(str(label) for label in dual_labels)
        maximum = int(maximum_simple_objects)
        tolerance = float(coherence_tolerance)
        if not labels_ or any(not label for label in labels_):
            raise ValueError("Fusion-category labels must be nonempty.")
        if len(set(labels_)) != len(labels_):
            raise ValueError("Fusion-category labels must be unique.")
        if len(labels_) > maximum or maximum < 1:
            raise ValueError("Simple-object count exceeds maximum_simple_objects.")
        if unit_label not in labels_ or len(duals) != len(labels_):
            raise ValueError("Unit and dual labels must belong to the category.")
        if any(label not in labels_ for label in duals):
            raise ValueError("Every dual label must belong to the category.")
        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise ValueError("coherence_tolerance must be finite and nonnegative.")
        count = len(labels_)
        fusion_input = np.asarray(fusion_multiplicities)
        if fusion_input.shape != (count, count, count):
            raise ValueError("fusion_multiplicities must have shape (L, L, L).")
        if not np.all(np.isfinite(fusion_input)) or not np.all(
            fusion_input == np.rint(fusion_input)
        ):
            raise ValueError("Fusion multiplicities must be finite integers.")
        fusion = fusion_input.astype(np.int32)
        if np.any((fusion < 0) | (fusion > 1)):
            raise ValueError("This finite anyon runtime is multiplicity-free.")
        dimensions = np.asarray(quantum_dimensions, dtype=np.float64)
        if dimensions.shape != (count,) or not np.all(np.isfinite(dimensions)):
            raise ValueError(
                "quantum_dimensions must provide one finite value per label."
            )
        if np.any(dimensions <= 0.0):
            raise ValueError("Quantum dimensions must be positive.")
        f_data = np.asarray(f_symbols, dtype=np.complex128)
        r_data = np.asarray(r_symbols, dtype=np.complex128)
        if f_data.shape != (count,) * 6:
            raise ValueError("f_symbols must have shape (L, L, L, L, L, L).")
        if r_data.shape != (count, count, count):
            raise ValueError("r_symbols must have shape (L, L, L).")
        if not np.all(np.isfinite(f_data)) or not np.all(np.isfinite(r_data)):
            raise ValueError("F and R data must be finite.")
        unit = labels_.index(str(unit_label))
        dual_indices = tuple(labels_.index(label) for label in duals)
        for label in range(count):
            expected = np.zeros((count,), dtype=np.int32)
            expected[label] = 1
            if not np.array_equal(fusion[unit, label], expected) or not np.array_equal(
                fusion[label, unit], expected
            ):
                raise ValueError("Fusion data violate the unit law.")
            if dual_indices[dual_indices[label]] != label:
                raise ValueError("Duality must be involutive.")
            if fusion[label, dual_indices[label], unit] != 1:
                raise ValueError("Every object must fuse with its dual to the unit.")
        if not np.array_equal(fusion, np.swapaxes(fusion, 0, 1)):
            raise ValueError("Braided fusion multiplicities must be commutative.")
        associativity = _fusion_associativity_residual(fusion)
        dimension_residual = _quantum_dimension_residual(fusion, dimensions)
        if associativity > 0.0:
            raise ValueError("Fusion multiplicities violate associativity.")
        if dimension_residual > tolerance:
            raise ValueError("Quantum dimensions do not realize the fusion rules.")
        f_support = np.zeros_like(f_data, dtype=np.bool_)
        for first in range(count):
            for second in range(count):
                for third in range(count):
                    for total in range(count):
                        left, right = _channels(fusion, first, second, third, total)
                        for left_channel in left:
                            for right_channel in right:
                                f_support[
                                    first,
                                    second,
                                    third,
                                    total,
                                    left_channel,
                                    right_channel,
                                ] = True
        if np.any(np.abs(f_data[~f_support]) > tolerance):
            raise ValueError("F data are nonzero outside admissible fusion channels.")
        supported_r = fusion.astype("bool")
        if np.any(np.abs(r_data[~supported_r]) > tolerance):
            raise ValueError("R data are nonzero outside admissible fusion channels.")
        r_unitarity = float(np.max(np.abs(np.abs(r_data[supported_r]) - 1.0)))
        f_unitarity = _f_unitarity_residual(fusion, f_data)
        pentagon = _pentagon_residual(fusion, f_data)
        hexagon = _hexagon_residual(fusion, f_data, r_data)
        category_id = canonical_fingerprint(
            {
                "kind": "finite-multiplicity-free-braided-fusion-category",
                "labels": labels_,
                "unit": str(unit_label),
                "duals": duals,
                "fusion": array_tree_fingerprint(fusion),
                "quantum_dimensions": array_tree_fingerprint(dimensions),
                "f_symbols": array_tree_fingerprint(f_data),
                "r_symbols": array_tree_fingerprint(r_data),
            }
        )
        threshold = jnp.asarray(tolerance)
        coherent = jnp.asarray(
            max(
                associativity,
                dimension_residual,
                f_unitarity,
                r_unitarity,
                pentagon,
                hexagon,
            )
            <= tolerance
        )
        self.labels = labels_
        self.unit_index = unit
        self.dual_indices = dual_indices
        self.fusion_multiplicities = jnp.asarray(fusion)
        self.quantum_dimensions = jnp.asarray(dimensions)
        self.f_symbols = jnp.asarray(f_data)
        self.r_symbols = jnp.asarray(r_data)
        self.coherence = FusionCoherenceEvidence(
            fusion_associativity_residual=jnp.asarray(associativity),
            quantum_dimension_residual=jnp.asarray(dimension_residual),
            f_unitarity_residual=jnp.asarray(f_unitarity),
            r_unitarity_residual=jnp.asarray(r_unitarity),
            pentagon_residual=jnp.asarray(pentagon),
            hexagon_residual=jnp.asarray(hexagon),
            tolerance=threshold,
            coherent=coherent,
            category_id=category_id,
            claim="finite-explicit-f-r-data-only",
        )
        self.category_id = category_id

    @property
    def simple_object_count(self) -> int:
        return len(self.labels)

    @property
    def total_quantum_dimension(self) -> Array:
        return jnp.sqrt(jnp.sum(self.quantum_dimensions * self.quantum_dimensions))

    def index(self, label: str, /) -> int:
        target = str(label)
        if target not in self.labels:
            raise ValueError("Label is outside the fusion category.")
        return self.labels.index(target)

    def dual(self, label: str, /) -> str:
        return self.labels[self.dual_indices[self.index(label)]]

    def multiplicity(self, left: str, right: str, output: str, /) -> int:
        indices = self.index(left), self.index(right), self.index(output)
        return int(np.asarray(self.fusion_multiplicities)[indices])


def _fusion_associativity_residual(fusion: np.ndarray, /) -> float:
    left = np.asarray(ein.contract("abx,xcd->abcd", fusion, fusion))
    right = np.asarray(ein.contract("bcx,axd->abcd", fusion, fusion))
    return float(np.max(np.abs(left - right)))


def _quantum_dimension_residual(fusion: np.ndarray, dimensions: np.ndarray, /) -> float:
    fused = np.asarray(ein.contract("abc,c->ab", fusion, dimensions))
    expected = dimensions[:, None] * dimensions[None, :]
    return float(np.max(np.abs(fused - expected)))


def _channels(
    fusion: np.ndarray, first: int, second: int, third: int, total: int, /
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    count = fusion.shape[0]
    left = tuple(
        intermediate
        for intermediate in range(count)
        if fusion[first, second, intermediate] and fusion[intermediate, third, total]
    )
    right = tuple(
        intermediate
        for intermediate in range(count)
        if fusion[second, third, intermediate] and fusion[first, intermediate, total]
    )
    return left, right


def _f_unitarity_residual(fusion: np.ndarray, f_data: np.ndarray, /) -> float:
    count = fusion.shape[0]
    residual = 0.0
    for first in range(count):
        for second in range(count):
            for third in range(count):
                for total in range(count):
                    left, right = _channels(fusion, first, second, third, total)
                    if len(left) != len(right):
                        raise ValueError(
                            "Fusion associativity produced unequal F-matrix bases."
                        )
                    if not left:
                        continue
                    matrix = f_data[
                        first,
                        second,
                        third,
                        total,
                        np.asarray(left)[:, None],
                        np.asarray(right)[None, :],
                    ]
                    identity = np.eye(len(left), dtype=np.complex128)
                    residual = max(
                        residual,
                        float(np.max(np.abs(matrix @ matrix.conj().T - identity))),
                        float(np.max(np.abs(matrix.conj().T @ matrix - identity))),
                    )
    return residual


def _pentagon_residual(fusion: np.ndarray, f_data: np.ndarray, /) -> float:
    count = fusion.shape[0]
    residual = 0.0
    for a in range(count):
        for b in range(count):
            for c in range(count):
                for d in range(count):
                    for total in range(count):
                        for ab in range(count):
                            if not fusion[a, b, ab]:
                                continue
                            for abc in range(count):
                                if not fusion[ab, c, abc] or not fusion[abc, d, total]:
                                    continue
                                for cd in range(count):
                                    if not fusion[c, d, cd]:
                                        continue
                                    for bcd in range(count):
                                        if (
                                            not fusion[b, cd, bcd]
                                            or not fusion[a, bcd, total]
                                        ):
                                            continue
                                        lhs = 0.0j
                                        for bc in range(count):
                                            if (
                                                fusion[b, c, bc]
                                                and fusion[a, bc, abc]
                                                and fusion[bc, d, bcd]
                                            ):
                                                lhs += (
                                                    f_data[a, b, c, abc, ab, bc]
                                                    * f_data[a, bc, d, total, abc, bcd]
                                                    * f_data[b, c, d, bcd, bc, cd]
                                                )
                                        rhs = (
                                            f_data[ab, c, d, total, abc, cd]
                                            * f_data[a, b, cd, total, ab, bcd]
                                        )
                                        residual = max(residual, float(abs(lhs - rhs)))
    return residual


def _f_matrix(
    fusion: np.ndarray,
    f_data: np.ndarray,
    first: int,
    second: int,
    third: int,
    total: int,
    /,
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    left, right = _channels(fusion, first, second, third, total)
    matrix = np.asarray(
        [[f_data[first, second, third, total, l, r] for r in right] for l in left],
        dtype=np.complex128,
    )
    return left, right, matrix


def _hexagon_residual(
    fusion: np.ndarray, f_data: np.ndarray, r_data: np.ndarray, /
) -> float:
    count = fusion.shape[0]
    residual = 0.0
    for a in range(count):
        for b in range(count):
            for c in range(count):
                for total in range(count):
                    left_abc, right_abc, f_abc = _f_matrix(fusion, f_data, a, b, c, total)
                    if not left_abc:
                        continue
                    left_bac, right_bac, f_bac = _f_matrix(fusion, f_data, b, a, c, total)
                    left_bca, right_bca, f_bca = _f_matrix(fusion, f_data, b, c, a, total)
                    if (
                        right_abc != left_bca
                        or left_abc != left_bac
                        or right_bac != right_bca
                    ):
                        raise ValueError(
                            "Fusion channels are incompatible with the first hexagon."
                        )
                    first_braid = np.diag([r_data[a, b, channel] for channel in left_abc])
                    second_braid = (
                        f_bca.conj()
                        @ np.diag([r_data[a, c, channel] for channel in right_bac])
                        @ f_bac.T
                    )
                    sequential = second_braid @ first_braid @ f_abc.conj()
                    direct = np.diag([r_data[a, channel, total] for channel in right_abc])
                    residual = max(residual, float(np.max(np.abs(sequential - direct))))
                    left_acb, right_acb, f_acb = _f_matrix(fusion, f_data, a, c, b, total)
                    left_cab, right_cab, f_cab = _f_matrix(fusion, f_data, c, a, b, total)
                    if (
                        right_abc != right_acb
                        or left_acb != left_cab
                        or left_abc != right_cab
                    ):
                        raise ValueError(
                            "Fusion channels are incompatible with the second hexagon."
                        )
                    sequential_other = (
                        f_cab.T
                        @ np.diag([r_data[a, c, channel] for channel in left_acb])
                        @ f_acb.conj()
                        @ np.diag([r_data[b, c, channel] for channel in right_abc])
                        @ f_abc.T
                    )
                    direct_other = np.diag(
                        [r_data[channel, c, total] for channel in left_abc]
                    )
                    residual = max(
                        residual,
                        float(np.max(np.abs(sequential_other - direct_other))),
                    )
    return residual


def z2_fusion_category() -> FiniteFusionCategory:
    """Return the pointed Z2 input category with its trivial braiding."""
    fusion = np.zeros((2, 2, 2), dtype=np.int32)
    for left in range(2):
        for right in range(2):
            fusion[left, right, left ^ right] = 1
    f_data = _unit_f_symbols(fusion)
    r_data = fusion.astype(np.complex128)
    return FiniteFusionCategory(
        ("1", "s"),
        "1",
        ("1", "s"),
        fusion,
        np.ones((2,)),
        f_data,
        r_data,
    )


def fibonacci_fusion_category() -> FiniteFusionCategory:
    """Return the standard unitary Fibonacci category in a real F gauge."""
    fusion = np.zeros((2, 2, 2), dtype=np.int32)
    fusion[0, 0, 0] = 1
    fusion[0, 1, 1] = 1
    fusion[1, 0, 1] = 1
    fusion[1, 1, 0] = 1
    fusion[1, 1, 1] = 1
    golden = (1.0 + sqrt(5.0)) / 2.0
    f_data = _unit_f_symbols(fusion)
    f_data[1, 1, 1, 1, 0, 0] = 1.0 / golden
    f_data[1, 1, 1, 1, 0, 1] = 1.0 / sqrt(golden)
    f_data[1, 1, 1, 1, 1, 0] = 1.0 / sqrt(golden)
    f_data[1, 1, 1, 1, 1, 1] = -1.0 / golden
    r_data = np.zeros_like(fusion, dtype=np.complex128)
    r_data[0, 0, 0] = 1.0
    r_data[0, 1, 1] = 1.0
    r_data[1, 0, 1] = 1.0
    r_data[1, 1, 0] = np.exp(-4.0j * np.pi / 5.0)
    r_data[1, 1, 1] = np.exp(3.0j * np.pi / 5.0)
    return FiniteFusionCategory(
        ("1", "tau"),
        "1",
        ("1", "tau"),
        fusion,
        np.asarray((1.0, golden)),
        f_data,
        r_data,
        coherence_tolerance=5e-9,
    )


def _unit_f_symbols(fusion: np.ndarray, /) -> np.ndarray:
    count = fusion.shape[0]
    result = np.zeros((count,) * 6, dtype=np.complex128)
    for first in range(count):
        for second in range(count):
            for third in range(count):
                for total in range(count):
                    left, right = _channels(fusion, first, second, third, total)
                    if len(left) == len(right) == 1:
                        result[first, second, third, total, left[0], right[0]] = 1.0
    return result


class StringNetPlan(StrictModule):
    """Finite fusion-basis graph plan with guarded dense reference lowering."""

    category: FiniteFusionCategory
    edge_count: int = eqx.field(static=True)
    vertices: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    plaquette_boundaries: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    maximum_hilbert_dimension: int = eqx.field(static=True)
    maximum_operator_elements: int = eqx.field(static=True)
    hilbert_dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        category: FiniteFusionCategory,
        edge_count: int,
        vertices: Sequence[tuple[int, int, int]],
        plaquette_boundaries: Sequence[Sequence[tuple[int, int]]],
        /,
        *,
        maximum_hilbert_dimension: int = 65_536,
        maximum_operator_elements: int = 10_000_000,
    ):
        if not isinstance(category, FiniteFusionCategory):
            raise TypeError("category must be FiniteFusionCategory.")
        if not bool(np.asarray(category.coherence.coherent)):
            raise ValueError("String-net lowering requires coherent F/R data.")
        edges = int(edge_count)
        vertices_ = tuple(tuple(vertex) for vertex in vertices)
        boundaries = tuple(
            tuple((int(edge), int(orientation)) for edge, orientation in boundary)
            for boundary in plaquette_boundaries
        )
        maximum = int(maximum_hilbert_dimension)
        maximum_operators = int(maximum_operator_elements)
        if edges < 1 or maximum < 1 or maximum_operators < 1:
            raise ValueError("String-net resource capacities must be positive.")
        if any(len(vertex) != 3 for vertex in vertices_):
            raise ValueError("Each oriented trivalent vertex needs three edge indices.")
        if any(not 0 <= edge < edges for vertex in vertices_ for edge in vertex):
            raise ValueError("Vertex edge index lies outside the graph.")
        for boundary in boundaries:
            boundary_edges = tuple(edge for edge, _ in boundary)
            if not boundary or len(set(boundary_edges)) != len(boundary_edges):
                raise ValueError(
                    "Plaquette boundaries must be nonempty simple edge loops."
                )
            if any(not 0 <= edge < edges for edge in boundary_edges):
                raise ValueError("Plaquette boundary edge lies outside the graph.")
            if any(orientation not in (-1, 1) for _, orientation in boundary):
                raise ValueError("Plaquette edge orientations must be +1 or -1.")
        dimension = category.simple_object_count**edges
        if dimension > maximum:
            raise ValueError(
                f"String-net Hilbert dimension {dimension} exceeds capacity {maximum}."
            )
        dense_count = (
            (
                len(vertices_)
                + len(boundaries) * category.simple_object_count
                + len(boundaries)
                + 1
            )
            * dimension
            * dimension
        )
        if dense_count > maximum_operators:
            raise ValueError(
                f"String-net lowering requires {dense_count} dense elements; capacity is {maximum_operators}."
            )
        self.category = category
        self.edge_count = edges
        self.vertices = vertices_
        self.plaquette_boundaries = boundaries
        self.maximum_hilbert_dimension = maximum
        self.maximum_operator_elements = maximum_operators
        self.hilbert_dimension = dimension
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-string-net-fusion-basis-plan",
                "category": category.category_id,
                "edge_count": edges,
                "vertices": vertices_,
                "plaquette_boundaries": boundaries,
                "maximum_hilbert_dimension": maximum,
                "maximum_operator_elements": maximum_operators,
            }
        )


class StringNetEvidence(StrictModule):
    vertex_idempotence_residual: Array
    plaquette_idempotence_residual: Array
    hermiticity_residual: Array
    commutator_residual: Array
    tolerance: Array
    projector_identities: Array
    plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class PreparedStringNetHamiltonian(StrictModule):
    plan: StringNetPlan
    vertex_projectors: Array
    plaquette_projectors: Array
    hamiltonian: Array
    evidence: StringNetEvidence
    prepared_id: str = eqx.field(static=True)


def _embed_boundary_loop(
    local_loops: np.ndarray,
    label: int,
    boundary: tuple[tuple[int, int], ...],
    edge_count: int,
    dual_indices: tuple[int, ...],
    /,
) -> np.ndarray:
    by_edge = {edge: orientation for edge, orientation in boundary}
    label_count = local_loops.shape[1]
    identity = np.eye(label_count, dtype=np.complex128)
    result = np.asarray([[1.0 + 0.0j]])
    for edge in range(edge_count):
        if edge not in by_edge:
            local = identity
        else:
            oriented_label = label if by_edge[edge] == 1 else dual_indices[label]
            local = local_loops[oriented_label]
        result = np.kron(result, local)
    return result


def prepare_string_net_hamiltonian(
    plan: StringNetPlan,
    /,
    *,
    loop_operators: ArrayLike | None = None,
    tolerance: float = 1e-9,
) -> PreparedStringNetHamiltonian:
    """Lower a finite string-net reference using explicit or native fusion loops."""
    if not isinstance(plan, StringNetPlan):
        raise TypeError("plan must be StringNetPlan.")
    tolerance_ = float(tolerance)
    if tolerance_ < 0.0 or not np.isfinite(tolerance_):
        raise ValueError("tolerance must be finite and nonnegative.")
    category = plan.category
    count = category.simple_object_count
    dimension = plan.hilbert_dimension
    fusion = np.asarray(category.fusion_multiplicities)
    basis = np.stack(
        np.unravel_index(np.arange(dimension), (count,) * plan.edge_count), axis=-1
    )
    vertices = []
    for incoming_left, incoming_right, outgoing in plan.vertices:
        valid = fusion[
            basis[:, incoming_left], basis[:, incoming_right], basis[:, outgoing]
        ].astype("float64")
        vertices.append(np.diag(valid).astype(np.complex128))
    vertex_array = (
        np.stack(vertices)
        if vertices
        else np.zeros((0, dimension, dimension), dtype=np.complex128)
    )
    if loop_operators is None:
        f_data = np.asarray(category.f_symbols)
        nonzero_f = f_data[np.abs(f_data) > tolerance_]
        native_category = (
            np.all(np.sum(fusion, axis=2) == 1)
            and np.allclose(np.asarray(category.quantum_dimensions), 1.0)
            and np.allclose(nonzero_f, 1.0)
        )
        if not native_category and plan.plaquette_boundaries:
            raise ValueError(
                "Non-pointed or nontrivially-associated categories require explicit full-boundary loop_operators."
            )
        local_loops = np.swapaxes(fusion, 1, 2).astype(np.complex128)
        loops = (
            np.stack(
                [
                    np.stack(
                        [
                            _embed_boundary_loop(
                                local_loops,
                                label,
                                boundary,
                                plan.edge_count,
                                category.dual_indices,
                            )
                            for label in range(count)
                        ]
                    )
                    for boundary in plan.plaquette_boundaries
                ]
            )
            if plan.plaquette_boundaries
            else np.zeros((0, count, dimension, dimension), dtype=np.complex128)
        )
    else:
        loops = np.asarray(loop_operators, dtype=np.complex128)
        expected = (
            len(plan.plaquette_boundaries),
            count,
            dimension,
            dimension,
        )
        if loops.shape != expected:
            raise ValueError(f"loop_operators must have shape {expected}.")
        if not np.all(np.isfinite(loops)):
            raise ValueError("loop_operators must be finite.")
    fusion_residual = 0.0
    for plaquette in range(loops.shape[0]):
        for left in range(count):
            for right in range(count):
                expected = sum(
                    fusion[left, right, output] * loops[plaquette, output]
                    for output in range(count)
                )
                fusion_residual = max(
                    fusion_residual,
                    float(
                        np.max(
                            np.abs(
                                loops[plaquette, left] @ loops[plaquette, right]
                                - expected
                            )
                        )
                    ),
                )
    if fusion_residual > tolerance_:
        raise ValueError("Plaquette loop operators do not represent the fusion algebra.")
    weights = np.asarray(category.quantum_dimensions)
    normalization = float(np.sum(weights * weights))
    plaquettes = np.asarray(ein.contract("s,psij->pij", weights / normalization, loops))
    identity = np.eye(dimension, dtype=np.complex128)
    hamiltonian = np.zeros_like(identity)
    for projector in vertex_array:
        hamiltonian = hamiltonian + identity - projector
    for projector in plaquettes:
        hamiltonian = hamiltonian + identity - projector
    all_projectors = tuple(vertex_array) + tuple(plaquettes)
    vertex_residual = max(
        (float(np.max(np.abs(value @ value - value))) for value in vertex_array),
        default=0.0,
    )
    plaquette_residual = max(
        (float(np.max(np.abs(value @ value - value))) for value in plaquettes),
        default=0.0,
    )
    hermiticity = max(
        (float(np.max(np.abs(value - value.conj().T))) for value in all_projectors),
        default=0.0,
    )
    commutator = 0.0
    for first, left in enumerate(all_projectors):
        for right in all_projectors[first + 1 :]:
            commutator = max(
                commutator, float(np.max(np.abs(left @ right - right @ left)))
            )
    accepted = (
        max(vertex_residual, plaquette_residual, hermiticity, commutator) <= tolerance_
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-finite-string-net-hamiltonian",
            "plan": plan.plan_id,
            "loops": array_tree_fingerprint(loops),
        }
    )
    return PreparedStringNetHamiltonian(
        plan=plan,
        vertex_projectors=jnp.asarray(vertex_array),
        plaquette_projectors=jnp.asarray(plaquettes),
        hamiltonian=jnp.asarray(hamiltonian),
        evidence=StringNetEvidence(
            vertex_idempotence_residual=jnp.asarray(vertex_residual),
            plaquette_idempotence_residual=jnp.asarray(plaquette_residual),
            hermiticity_residual=jnp.asarray(hermiticity),
            commutator_residual=jnp.asarray(commutator),
            tolerance=jnp.asarray(tolerance_),
            projector_identities=jnp.asarray(accepted),
            plan_id=plan.plan_id,
            claim="finite-fusion-basis-string-net-reference-only",
        ),
        prepared_id=prepared_id,
    )


class AnyonicTensorBlock(StrictModule):
    charges: tuple[str, ...] = eqx.field(static=True)
    data: Array

    def __init__(self, charges: Sequence[str], data: ArrayLike, /):
        charges_ = tuple(str(charge) for charge in charges)
        data_ = jnp.asarray(data)
        if data_.ndim != len(charges_):
            raise ValueError("Anyonic block rank must equal its charge-label count.")
        if not jnp.issubdtype(data_.dtype, jnp.inexact):
            data_ = data_.astype("float64")
        self.charges = charges_
        self.data = data_


class AnyonicTensor(StrictModule):
    """Block-sparse tensor with oriented categorical charge sectors."""

    category: FiniteFusionCategory
    orientations: tuple[int, ...] = eqx.field(static=True)
    blocks: tuple[AnyonicTensorBlock, ...]
    tensor_id: str = eqx.field(static=True)

    def __init__(
        self,
        category: FiniteFusionCategory,
        orientations: Sequence[int],
        blocks: Sequence[AnyonicTensorBlock],
        /,
    ):
        if not isinstance(category, FiniteFusionCategory):
            raise TypeError("category must be FiniteFusionCategory.")
        orientations_ = tuple(orientations)
        blocks_ = tuple(blocks)
        if any(value not in (-1, 1) for value in orientations_):
            raise ValueError("Anyonic tensor orientations must be +1 or -1.")
        if any(not isinstance(block, AnyonicTensorBlock) for block in blocks_):
            raise TypeError("blocks must contain AnyonicTensorBlock values.")
        if any(len(block.charges) != len(orientations_) for block in blocks_):
            raise ValueError("Every block must carry one charge per tensor leg.")
        if any(
            charge not in category.labels for block in blocks_ for charge in block.charges
        ):
            raise ValueError("An anyonic block charge lies outside the category.")
        keys = tuple(block.charges for block in blocks_)
        if len(set(keys)) != len(keys):
            raise ValueError("Anyonic tensor block charge tuples must be unique.")
        self.category = category
        self.orientations = orientations_
        self.blocks = blocks_
        self.tensor_id = canonical_fingerprint(
            {
                "kind": "finite-anyonic-block-tensor",
                "category": category.category_id,
                "orientations": orientations_,
                "blocks": tuple(
                    {
                        "charges": block.charges,
                        "value": array_tree_fingerprint(block.data),
                    }
                    for block in blocks_
                ),
            }
        )


def _contract_expression(
    left_rank: int, right_rank: int, left_axis: int, right_axis: int, /
) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if left_rank + right_rank - 1 > len(alphabet):
        raise ValueError("Anyonic contraction rank exceeds the einsum label capacity.")
    left_labels = list(alphabet[:left_rank])
    contracted = left_labels[left_axis]
    right_labels = list(alphabet[left_rank : left_rank + right_rank])
    right_labels[right_axis] = contracted
    output = left_labels[:left_axis] + left_labels[left_axis + 1 :]
    output += right_labels[:right_axis] + right_labels[right_axis + 1 :]
    return f"{''.join(left_labels)},{''.join(right_labels)}->{''.join(output)}"


def contract_anyonic_tensors(
    left: AnyonicTensor,
    right: AnyonicTensor,
    left_axis: int,
    right_axis: int,
    /,
    *,
    maximum_output_elements: int = 1_000_000,
) -> AnyonicTensor:
    """Contract dual oriented sectors with the categorical quantum-trace weight."""
    if not isinstance(left, AnyonicTensor) or not isinstance(right, AnyonicTensor):
        raise TypeError("left and right must be AnyonicTensor values.")
    if left.category.category_id != right.category.category_id:
        raise ValueError("Anyonic tensors must belong to the same category.")
    left_axis_ = int(left_axis)
    right_axis_ = int(right_axis)
    if not 0 <= left_axis_ < len(left.orientations) or not 0 <= right_axis_ < len(
        right.orientations
    ):
        raise ValueError("Contraction axis lies outside a tensor.")
    if left.orientations[left_axis_] == right.orientations[right_axis_]:
        raise ValueError("Contracted anyonic legs must have opposite orientations.")
    maximum = int(maximum_output_elements)
    if maximum < 1:
        raise ValueError("maximum_output_elements must be positive.")
    expression = _contract_expression(
        len(left.orientations), len(right.orientations), left_axis_, right_axis_
    )
    accumulated: dict[tuple[str, ...], Array] = {}
    output_elements = 0
    for left_block in left.blocks:
        left_charge = left_block.charges[left_axis_]
        expected_right = left.category.dual(left_charge)
        for right_block in right.blocks:
            if right_block.charges[right_axis_] != expected_right:
                continue
            output_charges = (
                left_block.charges[:left_axis_] + left_block.charges[left_axis_ + 1 :]
            )
            output_charges += (
                right_block.charges[:right_axis_] + right_block.charges[right_axis_ + 1 :]
            )
            output_shape = (
                left_block.data.shape[:left_axis_]
                + left_block.data.shape[left_axis_ + 1 :]
            )
            output_shape += (
                right_block.data.shape[:right_axis_]
                + right_block.data.shape[right_axis_ + 1 :]
            )
            block_elements = prod(output_shape)
            if output_charges not in accumulated:
                output_elements += block_elements
            if output_elements > maximum:
                raise ValueError(
                    "Anyonic contraction output exceeds maximum_output_elements."
                )
            weight = left.category.quantum_dimensions[left.category.index(left_charge)]
            contracted = weight * ein.contract(
                expression, left_block.data, right_block.data
            )
            accumulated[output_charges] = (
                contracted
                if output_charges not in accumulated
                else accumulated[output_charges] + contracted
            )
    orientations = left.orientations[:left_axis_] + left.orientations[left_axis_ + 1 :]
    orientations += (
        right.orientations[:right_axis_] + right.orientations[right_axis_ + 1 :]
    )
    blocks = tuple(
        AnyonicTensorBlock(charges, value)
        for charges, value in sorted(accumulated.items())
    )
    return AnyonicTensor(left.category, orientations, blocks)


__all__ = [
    "AnyonicTensor",
    "AnyonicTensorBlock",
    "FiniteFusionCategory",
    "FusionCoherenceEvidence",
    "PreparedStringNetHamiltonian",
    "StringNetEvidence",
    "StringNetPlan",
    "contract_anyonic_tensors",
    "fibonacci_fusion_category",
    "prepare_string_net_hamiltonian",
    "z2_fusion_category",
]
