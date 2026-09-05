#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy.special import eval_jacobi

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._polynomial._orthogonal import legendre_rule_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, DenseLinearOperator, LinearSystem, solve
from .._reference_cell import reference_cell_topology
from ._high_order import (
    lagrange_1d_tabulation,
    SimplexNodalFamily,
)
from ._hp import FiniteElementHPLineage, FiniteElementHPTopology
from ._hp_runtime import (
    finite_element_hp_balance_error,
    FiniteElementHPGeometry,
    FiniteElementHPRefinementResult,
    tensor_trace_interpolation,
)
from ._reference import FiniteElementSpec


class AnisotropicHPattern(StrictModule, NonTrainableState):
    axes: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    child_ordinals: tuple[int, ...] = eqx.field(static=True)
    pattern_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, axes: Sequence[int], /):
        dimension_ = int(dimension)
        axes_ = tuple(sorted(int(axis) for axis in axes))
        if (
            dimension_ not in (2, 3)
            or not axes_
            or any(axis < 0 or axis >= dimension_ for axis in axes_)
            or len(set(axes_)) != len(axes_)
        ):
            raise ValueError("Anisotropic h axes must be unique tensor axes.")
        ordinals = []
        for local_bits in product((0, 1), repeat=len(axes_)):
            ordinal = sum(
                bit << axis for axis, bit in zip(axes_, local_bits, strict=True)
            )
            ordinals.append(ordinal)
        self.axes = axes_
        self.dimension = dimension_
        self.child_ordinals = tuple(ordinals)
        self.pattern_id = canonical_fingerprint(
            {"kind": "anisotropic-h-pattern", "dimension": dimension_, "axes": axes_}
        )


def _tensor_corners(dimension: int) -> np.ndarray:
    if dimension == 1:
        return np.asarray(((0.0,), (1.0,)))
    if dimension == 2:
        return np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    if dimension == 3:
        return np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (1.0, 1.0, 1.0),
                (0.0, 1.0, 1.0),
            )
        )
    raise ValueError("Tensor corners require dimension one, two, or three.")


def _multilinear(vertices: np.ndarray, points: np.ndarray) -> np.ndarray:
    corners = _tensor_corners(points.shape[1])
    factors = np.where(
        corners[None, :, :] == 0.0,
        1.0 - points[:, None, :],
        points[:, None, :],
    )
    return np.prod(factors, axis=-1) @ vertices


def refine_anisotropic_hp_cells(
    topology: FiniteElementHPTopology,
    geometry: FiniteElementHPGeometry,
    marked_cell_ids: ArrayLike,
    pattern: AnisotropicHPattern,
    /,
    *,
    target_degrees: ArrayLike | None = None,
) -> FiniteElementHPRefinementResult:
    if (
        pattern.dimension != topology.dimension
        or geometry.topology_id != topology.topology_id
    ):
        raise ValueError("Anisotropic pattern, topology, and geometry disagree.")
    identifiers = np.asarray(topology.cell_global_ids).copy()
    allocated = np.asarray(topology.allocated).copy()
    active = np.asarray(topology.active).copy()
    degrees = np.asarray(topology.cell_degrees).copy()
    roots = np.asarray(topology.root_cell_ids).copy()
    paths = np.asarray(topology.path_codes).copy()
    levels = np.asarray(topology.levels).copy()
    parents = np.asarray(topology.parent_slots).copy()
    children = np.asarray(topology.child_slots).copy()
    child_valid = np.asarray(topology.child_valid).copy()
    vertices = np.asarray(geometry.cell_vertices).copy()
    lower = np.asarray(geometry.reference_lower).copy()
    upper = np.asarray(geometry.reference_upper).copy()
    slot_by_id = {int(identifiers[slot]): int(slot) for slot in np.flatnonzero(active)}
    marked = np.asarray(marked_cell_ids, dtype=np.int64)
    unknown = set(marked.tolist()) - set(slot_by_id)
    if marked.ndim != 1 or np.unique(marked).size != marked.size or unknown:
        raise ValueError(f"Anisotropic h marked IDs are invalid: {sorted(unknown)!r}.")
    marked_slots = np.asarray(
        sorted(
            (slot_by_id[int(value)] for value in marked),
            key=lambda slot: (int(roots[slot]), int(paths[slot])),
        ),
        dtype=np.int32,
    )
    free = np.flatnonzero(~allocated)
    required = marked_slots.size * len(pattern.child_ordinals)
    if free.size < required:
        raise ValueError("Anisotropic h refinement exceeds forest capacity.")
    requested_degrees = (
        None if target_degrees is None else np.asarray(target_degrees, dtype=np.int32)
    )
    if requested_degrees is not None and requested_degrees.shape != (
        marked_slots.size,
        topology.dimension,
    ):
        raise ValueError("target_degrees must contain one tuple per marked cell.")
    next_global = int(np.max(identifiers[allocated], initial=-1)) + 1
    source_routes = []
    target_routes = []
    relations = []
    marked_set = set(marked_slots.tolist())
    for slot in np.flatnonzero(active):
        if int(slot) not in marked_set:
            source_routes.append(int(slot))
            target_routes.append(int(slot))
            relations.append("unchanged")
    cursor = 0
    corners = _tensor_corners(topology.dimension)
    for marked_index, parent in enumerate(marked_slots):
        parent = int(parent)
        parent_degree = degrees[parent].copy()
        active[parent] = False
        degrees[parent] = 0
        midpoint = 0.5 * (lower[parent] + upper[parent])
        for ordinal in pattern.child_ordinals:
            child = int(free[cursor])
            cursor += 1
            bits = np.asarray(
                tuple((ordinal >> axis) & 1 for axis in range(topology.dimension))
            )
            child_lower = lower[parent].copy()
            child_upper = upper[parent].copy()
            for axis in pattern.axes:
                if bits[axis]:
                    child_lower[axis] = midpoint[axis]
                else:
                    child_upper[axis] = midpoint[axis]
            parent_points = (child_lower - lower[parent]) / (
                upper[parent] - lower[parent]
            )
            parent_scale = (child_upper - child_lower) / (upper[parent] - lower[parent])
            child_points = parent_points + corners * parent_scale
            identifiers[child] = next_global
            next_global += 1
            allocated[child] = True
            active[child] = True
            degrees[child] = (
                parent_degree
                if requested_degrees is None
                else requested_degrees[marked_index]
            )
            roots[child] = roots[parent]
            paths[child] = paths[parent] * topology.child_capacity + ordinal + 1
            levels[child] = levels[parent] + 1
            parents[child] = parent
            children[parent, ordinal] = child
            child_valid[parent, ordinal] = True
            lower[child] = child_lower
            upper[child] = child_upper
            vertices[child] = _multilinear(vertices[parent], child_points)
            source_routes.append(parent)
            target_routes.append(child)
            relations.append("refinement")
    refined_topology = FiniteElementHPTopology(
        topology.cell_kind,
        canonical_fingerprint(
            {
                "kind": "anisotropic-h-topology",
                "source": topology.plan_id,
                "pattern": pattern.pattern_id,
                "marked": marked.tolist(),
            }
        ),
        identifiers,
        allocated,
        active,
        degrees,
        root_cell_ids=roots,
        path_codes=paths,
        levels=levels,
        parent_slots=parents,
        child_slots=children,
        child_valid=child_valid,
    )
    refined_geometry = FiniteElementHPGeometry(refined_topology, vertices, lower, upper)
    if finite_element_hp_balance_error(refined_topology, refined_geometry) > 1:
        raise ValueError("Anisotropic h refinement violates directional 2:1 balance.")
    lineage = FiniteElementHPLineage(
        topology.topology_id,
        refined_topology.topology_id,
        topology.capacity,
        refined_topology.capacity,
        np.asarray(source_routes, dtype=np.int32),
        np.asarray(target_routes, dtype=np.int32),
        tuple(relations),
    )
    return FiniteElementHPRefinementResult(
        refined_topology,
        refined_geometry,
        lineage,
        marked_slots,
        np.empty((0,), dtype=np.int32),
    )


def resize_hp_forest(
    topology: FiniteElementHPTopology,
    geometry: FiniteElementHPGeometry,
    new_capacity: int,
    /,
) -> tuple[FiniteElementHPTopology, FiniteElementHPGeometry]:
    capacity = int(new_capacity)
    if capacity < topology.capacity:
        allocated_slots = np.flatnonzero(np.asarray(topology.allocated))
        if allocated_slots.size and int(np.max(allocated_slots)) >= capacity:
            raise ValueError("Shrink requires compact allocated hp slots.")
    if capacity < 1:
        raise ValueError("hp forest capacity must be positive.")

    def pad(vector, shape, fill):
        result = np.full(shape, fill, dtype=np.asarray(vector).dtype)
        slices = tuple(
            slice(0, min(old, new))
            for old, new in zip(np.asarray(vector).shape, shape, strict=True)
        )
        result[slices] = np.asarray(vector)[slices]
        return result

    identifiers = pad(topology.cell_global_ids, (capacity,), -1)
    allocated = pad(topology.allocated, (capacity,), False)
    active = pad(topology.active, (capacity,), False)
    degrees = pad(topology.cell_degrees, (capacity, topology.dimension), 0)
    roots = pad(topology.root_cell_ids, (capacity,), -1)
    paths = pad(topology.path_codes, (capacity,), -1)
    levels = pad(topology.levels, (capacity,), -1)
    parents = pad(topology.parent_slots, (capacity,), -1)
    children = pad(topology.child_slots, (capacity, topology.child_capacity), -1)
    child_valid = pad(topology.child_valid, (capacity, topology.child_capacity), False)
    resized_topology = FiniteElementHPTopology(
        topology.cell_kind,
        canonical_fingerprint(
            {
                "kind": "resized-hp-forest",
                "source": topology.plan_id,
                "capacity": capacity,
            }
        ),
        identifiers,
        allocated,
        active,
        degrees,
        root_cell_ids=roots,
        path_codes=paths,
        levels=levels,
        parent_slots=parents,
        child_slots=children,
        child_valid=child_valid,
    )
    vertices = pad(
        geometry.cell_vertices,
        (capacity,) + tuple(np.asarray(geometry.cell_vertices).shape[1:]),
        0.0,
    )
    lower = pad(geometry.reference_lower, (capacity, topology.dimension), 0.0)
    upper = pad(geometry.reference_upper, (capacity, topology.dimension), 0.0)
    return resized_topology, FiniteElementHPGeometry(
        resized_topology, vertices, lower, upper
    )


def compact_hp_forest(
    topology: FiniteElementHPTopology,
    geometry: FiniteElementHPGeometry,
    /,
) -> tuple[FiniteElementHPTopology, FiniteElementHPGeometry, Array]:
    allocated = np.flatnonzero(np.asarray(topology.allocated))
    order = sorted(
        allocated.tolist(),
        key=lambda slot: (
            int(np.asarray(topology.levels)[slot]),
            int(np.asarray(topology.root_cell_ids)[slot]),
            int(np.asarray(topology.path_codes)[slot]),
        ),
    )
    old_to_new = np.full((topology.capacity,), -1, dtype=np.int32)
    old_to_new[np.asarray(order)] = np.arange(len(order), dtype=np.int32)
    capacity = topology.capacity
    identifiers = np.full((capacity,), -1, dtype=np.int64)
    allocated_new = np.zeros((capacity,), dtype=bool)
    active = np.zeros((capacity,), dtype=bool)
    degrees = np.zeros((capacity, topology.dimension), dtype=np.int32)
    roots = np.full((capacity,), -1, dtype=np.int64)
    paths = np.full((capacity,), -1, dtype=np.int64)
    levels = np.full((capacity,), -1, dtype=np.int32)
    parents = np.full((capacity,), -1, dtype=np.int32)
    children = np.full((capacity, topology.child_capacity), -1, dtype=np.int32)
    child_valid = np.zeros_like(children, dtype=bool)
    vertices = np.zeros(
        (capacity,) + tuple(np.asarray(geometry.cell_vertices).shape[1:]),
        dtype=np.asarray(geometry.cell_vertices).dtype,
    )
    lower = np.zeros((capacity, topology.dimension), dtype=float)
    upper = np.zeros_like(lower)
    for new, old in enumerate(order):
        identifiers[new] = np.asarray(topology.cell_global_ids)[old]
        allocated_new[new] = True
        active[new] = np.asarray(topology.active)[old]
        degrees[new] = np.asarray(topology.cell_degrees)[old]
        roots[new] = np.asarray(topology.root_cell_ids)[old]
        paths[new] = np.asarray(topology.path_codes)[old]
        levels[new] = np.asarray(topology.levels)[old]
        old_parent = int(np.asarray(topology.parent_slots)[old])
        parents[new] = -1 if old_parent < 0 else old_to_new[old_parent]
        for ordinal, valid in enumerate(np.asarray(topology.child_valid)[old]):
            if valid:
                children[new, ordinal] = old_to_new[
                    int(np.asarray(topology.child_slots)[old, ordinal])
                ]
                child_valid[new, ordinal] = True
        vertices[new] = np.asarray(geometry.cell_vertices)[old]
        lower[new] = np.asarray(geometry.reference_lower)[old]
        upper[new] = np.asarray(geometry.reference_upper)[old]
    compacted = FiniteElementHPTopology(
        topology.cell_kind,
        canonical_fingerprint(
            {"kind": "compacted-hp-forest", "source": topology.plan_id}
        ),
        identifiers,
        allocated_new,
        active,
        degrees,
        root_cell_ids=roots,
        path_codes=paths,
        levels=levels,
        parent_slots=parents,
        child_slots=children,
        child_valid=child_valid,
    )
    return (
        compacted,
        FiniteElementHPGeometry(compacted, vertices, lower, upper),
        jnp.asarray(old_to_new),
    )


class GeometryOrderAdaptation(StrictModule, NonTrainableState):
    source_order: tuple[int, ...] = eqx.field(static=True)
    target_order: tuple[int, ...] = eqx.field(static=True)
    interpolation: Array
    curvature_indicator: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_nodes: ArrayLike,
        target_nodes: ArrayLike,
        source_order: Sequence[int],
        target_order: Sequence[int],
        coordinate_values: ArrayLike,
        /,
    ):
        source = np.asarray(source_nodes)
        target = np.asarray(target_nodes)
        values = np.asarray(coordinate_values)
        source_order_ = tuple(int(value) for value in source_order)
        target_order_ = tuple(int(value) for value in target_order)
        interpolation = np.asarray(tensor_trace_interpolation(source, target))
        if values.shape[-2] != source.shape[0]:
            raise ValueError("Geometry coordinate values do not match source nodes.")
        target_values = ein.contract("qi,...id->...qd", interpolation, values)
        linear = _multilinear(
            values.reshape((-1, values.shape[-1]))[: 2 ** source.shape[1]], target
        )
        indicator = np.max(
            np.linalg.norm(
                target_values.reshape((-1, values.shape[-1])) - linear, axis=-1
            )
        )
        self.source_order = source_order_
        self.target_order = target_order_
        self.interpolation = jnp.asarray(interpolation)
        self.curvature_indicator = jnp.asarray(indicator)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "geometry-order-adaptation",
                "source_order": source_order_,
                "target_order": target_order_,
                "interpolation": array_tree_fingerprint(interpolation),
                "curvature": float(indicator),
            }
        )

    def apply(self, coordinate_values: ArrayLike, /) -> Array:
        values = jnp.asarray(coordinate_values)
        return ein.contract("qi,...id->...qd", self.interpolation, values)


class NIrregularMortarPlan(StrictModule, NonTrainableState):
    coarse_to_patch: tuple[Array, ...]
    patch_weights: tuple[Array, ...]
    reproduction_error: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coarse_nodes: ArrayLike,
        patch_points: Sequence[ArrayLike],
        patch_weights: Sequence[ArrayLike],
        /,
    ):
        nodes = np.asarray(coarse_nodes)
        points = tuple(np.asarray(value) for value in patch_points)
        weights = tuple(np.asarray(value) for value in patch_weights)
        if not points or len(points) != len(weights):
            raise ValueError("n-irregular mortars require points and weights per patch.")
        matrices = tuple(
            np.asarray(tensor_trace_interpolation(nodes, value)) for value in points
        )
        if any(
            weight.shape != (point.shape[0],)
            for point, weight in zip(points, weights, strict=True)
        ):
            raise ValueError("n-irregular patch weights must match patch points.")
        constant_integral = sum(float(np.sum(weight)) for weight in weights)
        coarse_measure = float(np.sum(np.concatenate(weights)))
        error = abs(constant_integral - coarse_measure)
        self.coarse_to_patch = tuple(jnp.asarray(value) for value in matrices)
        self.patch_weights = tuple(jnp.asarray(value) for value in weights)
        self.reproduction_error = jnp.asarray(error)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "n-irregular-mortar",
                "nodes": array_tree_fingerprint(nodes),
                "points": [array_tree_fingerprint(value) for value in points],
                "weights": [array_tree_fingerprint(value) for value in weights],
            }
        )


class TensorCompatibleFamily(StrictModule, NonTrainableState):
    kind: Literal["Hcurl", "Hdiv"] = eqx.field(static=True)
    cell_kind: Literal["quadrilateral", "hexahedron"] = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    component_degrees: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    mapping: str = eqx.field(static=True)

    def __init__(
        self,
        kind: Literal["Hcurl", "Hdiv"],
        cell_kind: Literal["quadrilateral", "hexahedron"],
        degree: int,
        /,
    ):
        p = int(degree)
        dimension = 2 if cell_kind == "quadrilateral" else 3
        if kind not in ("Hcurl", "Hdiv") or p < 1:
            raise ValueError("Compatible tensor families require Hcurl/Hdiv and p >= 1.")
        if kind == "Hcurl":
            degrees = tuple(
                tuple(p - 1 if axis == component else p for axis in range(dimension))
                for component in range(dimension)
            )
            mapping = "covariant_piola"
        else:
            degrees = tuple(
                tuple(p if axis == component else p - 1 for axis in range(dimension))
                for component in range(dimension)
            )
            mapping = "contravariant_piola"
        self.kind = kind
        self.cell_kind = cell_kind
        self.degree = p
        self.component_degrees = degrees
        self.local_dof_count = sum(
            int(np.prod(np.asarray(value) + 1)) for value in degrees
        )
        self.mapping = mapping

    def tabulate(self, points: ArrayLike, /) -> Array:
        points_ = jnp.asarray(points)
        dimension = points_.shape[-1]
        blocks = []
        for component, degrees in enumerate(self.component_degrees):
            exponents = tuple(product(*(range(value + 1) for value in degrees)))
            scalar = jnp.stack(
                [
                    jnp.prod(points_ ** jnp.asarray(exponent), axis=-1)
                    for exponent in exponents
                ],
                axis=-1,
            )
            vector = (
                jnp.zeros(
                    scalar.shape + (dimension,),
                    dtype=scalar.dtype,
                )
                .at[..., component]
                .set(scalar)
            )
            blocks.append(vector)
        return jnp.concatenate(blocks, axis=-2)


def tensor_hcurl_family(
    cell_kind: Literal["quadrilateral", "hexahedron"],
    degree: int,
    /,
) -> TensorCompatibleFamily:
    return TensorCompatibleFamily("Hcurl", cell_kind, degree)


def tensor_hdiv_family(
    cell_kind: Literal["quadrilateral", "hexahedron"],
    degree: int,
    /,
) -> TensorCompatibleFamily:
    return TensorCompatibleFamily("Hdiv", cell_kind, degree)


class TensorDeRhamComplex(StrictModule, NonTrainableState):
    degree: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    gradient: Array
    curl: Array
    divergence: Array
    grad_curl_defect: Array
    curl_div_defect: Array
    complex_id: str = eqx.field(static=True)

    def __init__(self, degree: int, dimension: int, /):
        p = int(degree)
        d = int(dimension)
        if p < 1 or d not in (2, 3):
            raise ValueError("Tensor de Rham complexes require p >= 1 and dimension 2/3.")
        derivative = np.zeros((p, p + 1))
        for power in range(1, p + 1):
            derivative[power - 1, power] = power

        def kron_factors(factors):
            result = factors[0]
            for factor in factors[1:]:
                result = np.kron(result, factor)
            return result

        identity_p = np.eye(p + 1)
        identity_m = np.eye(p)
        grad_blocks = []
        for axis in range(d):
            factors = [identity_p] * d
            factors[axis] = derivative
            grad_blocks.append(kron_factors(factors))
        gradient = np.concatenate(grad_blocks, axis=0)
        if d == 2:
            dx = np.kron(derivative, np.eye(p))
            dy = np.kron(np.eye(p), derivative)
            curl = np.concatenate((-dy, dx), axis=1)
            divergence = curl
            grad_curl = curl @ gradient
            curl_div = np.zeros((1, 1))
        else:
            edge_sizes = [p * (p + 1) * (p + 1)] * 3
            face_sizes = [(p + 1) * p * p] * 3
            curl = np.zeros((sum(face_sizes), sum(edge_sizes)))
            # Component 0: d_y E_z - d_z E_y
            dy_ez = kron_factors((identity_p, derivative, identity_m))
            dz_ey = kron_factors((identity_p, identity_m, derivative))
            curl[0 : face_sizes[0], edge_sizes[0] + edge_sizes[1] :] = dy_ez
            curl[
                0 : face_sizes[0], edge_sizes[0] : edge_sizes[0] + edge_sizes[1]
            ] = -dz_ey
            # Component 1: d_z E_x - d_x E_z
            dz_ex = kron_factors((identity_m, identity_p, derivative))
            dx_ez = kron_factors((derivative, identity_p, identity_m))
            start = face_sizes[0]
            curl[start : start + face_sizes[1], : edge_sizes[0]] = dz_ex
            curl[start : start + face_sizes[1], edge_sizes[0] + edge_sizes[1] :] = -dx_ez
            # Component 2: d_x E_y - d_y E_x
            dx_ey = kron_factors((derivative, identity_m, identity_p))
            dy_ex = kron_factors((identity_m, derivative, identity_p))
            start += face_sizes[1]
            curl[start:, edge_sizes[0] : edge_sizes[0] + edge_sizes[1]] = dx_ey
            curl[start:, : edge_sizes[0]] = -dy_ex
            div_blocks = (
                kron_factors((derivative, identity_m, identity_m)),
                kron_factors((identity_m, derivative, identity_m)),
                kron_factors((identity_m, identity_m, derivative)),
            )
            divergence = np.concatenate(div_blocks, axis=1)
            grad_curl = curl @ gradient
            curl_div = divergence @ curl
        self.degree = p
        self.dimension = d
        self.gradient = jnp.asarray(gradient)
        self.curl = jnp.asarray(curl)
        self.divergence = jnp.asarray(divergence)
        self.grad_curl_defect = jnp.asarray(np.max(np.abs(grad_curl), initial=0.0))
        self.curl_div_defect = jnp.asarray(np.max(np.abs(curl_div), initial=0.0))
        self.complex_id = canonical_fingerprint(
            {
                "kind": "tensor-de-rham-complex",
                "degree": p,
                "dimension": d,
                "gradient": list(gradient.shape),
                "curl": list(curl.shape),
                "divergence": list(divergence.shape),
            }
        )


class TensorPiolaMap(StrictModule, NonTrainableState):
    mapping: Literal["covariant", "contravariant"] = eqx.field(static=True)

    def __init__(self, mapping: Literal["covariant", "contravariant"], /):
        if mapping not in ("covariant", "contravariant"):
            raise ValueError("Piola mapping must be covariant or contravariant.")
        self.mapping = mapping

    def apply(self, jacobian: ArrayLike, values: ArrayLike, /) -> Array:
        matrix = jnp.asarray(jacobian)
        value = jnp.asarray(values)
        if self.mapping == "covariant":
            return jnp.linalg.solve(jnp.swapaxes(matrix, -1, -2), value[..., None])[
                ..., 0
            ]
        determinant = jnp.linalg.det(matrix)
        return ein.contract("...ij,...j->...i", matrix, value) / determinant[..., None]


def _shifted_jacobi(
    degree: int, alpha: float, beta: float, values: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    polynomial = eval_jacobi(degree, alpha, beta, 2.0 * values - 1.0)
    derivative = (
        np.zeros_like(values)
        if degree == 0
        else (degree + alpha + beta + 1.0)
        * eval_jacobi(degree - 1, alpha + 1.0, beta + 1.0, 2.0 * values - 1.0)
    )
    return np.asarray(polynomial), np.asarray(derivative)


def _pyramid_modal_tabulation(
    points: np.ndarray,
    indices: tuple[tuple[int, int, int], ...],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(points, dtype=float)
    height = values[:, 2]
    scale = 1.0 - height
    safe = np.where(scale > 1.0e-12, scale, 1.0)
    first = np.where(scale > 1.0e-12, (values[:, 0] - 0.5 * height) / safe, 0.5)
    second = np.where(scale > 1.0e-12, (values[:, 1] - 0.5 * height) / safe, 0.5)
    modal_values = []
    modal_gradients = []
    for first_degree, second_degree, height_degree in indices:
        maximum = max(first_degree, second_degree)
        first_value, first_derivative = _shifted_jacobi(first_degree, 0.0, 0.0, first)
        second_value, second_derivative = _shifted_jacobi(second_degree, 0.0, 0.0, second)
        height_value, height_derivative = _shifted_jacobi(
            height_degree, 2.0 * maximum + 2.0, 0.0, height
        )
        scale_power = scale**maximum
        mode = first_value * second_value * scale_power * height_value
        x_gradient = (
            first_derivative
            * second_value
            * np.where(maximum > 0, scale ** max(maximum - 1, 0), 1.0)
            * height_value
        )
        y_gradient = (
            first_value
            * second_derivative
            * np.where(maximum > 0, scale ** max(maximum - 1, 0), 1.0)
            * height_value
        )
        first_height_derivative = np.where(scale > 1.0e-12, (first - 0.5) / safe, 0.0)
        second_height_derivative = np.where(scale > 1.0e-12, (second - 0.5) / safe, 0.0)
        scale_derivative = (
            np.zeros_like(scale) if maximum == 0 else -maximum * scale ** (maximum - 1)
        )
        z_gradient = (
            first_derivative
            * first_height_derivative
            * second_value
            * scale_power
            * height_value
            + first_value
            * second_derivative
            * second_height_derivative
            * scale_power
            * height_value
            + first_value * second_value * scale_derivative * height_value
            + first_value * second_value * scale_power * height_derivative
        )
        modal_values.append(mode)
        modal_gradients.append(np.stack((x_gradient, y_gradient, z_gradient), axis=-1))
    return np.stack(tuple(modal_values), axis=-1), np.stack(
        tuple(modal_gradients), axis=1
    )


class HybridReferenceFamily(StrictModule, NonTrainableState):
    """Anisotropic prism and arbitrary-order rational pyramid family."""

    cell_kind: Literal["prism", "pyramid"] = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    orders: tuple[int, int] = eqx.field(static=True)
    nodes: Array
    basis_permutation: tuple[int, ...] = eqx.field(static=True)
    modal_indices: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    coefficients: Array
    condition_number: float = eqx.field(static=True)
    family_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_kind: Literal["prism", "pyramid"],
        degree: int | tuple[int, int],
        /,
    ):
        kind = str(cell_kind)
        if kind == "prism" and isinstance(degree, tuple):
            if len(degree) != 2:
                raise ValueError("Prism degree tuples must be (triangle, axial).")
            triangle_degree, axial_degree = (int(value) for value in degree)
        else:
            triangle_degree = axial_degree = int(degree)
        p = triangle_degree
        q = axial_degree
        if kind not in ("prism", "pyramid") or min(p, q) < 1:
            raise ValueError("Hybrid references require prism/pyramid and degree >= 1.")
        if kind == "pyramid" and p != q:
            raise ValueError("Pyramid orders must be isotropic.")

        if kind == "prism":
            triangle = SimplexNodalFamily("triangle", p)
            rule = legendre_rule_data(q + 1, "lobatto")
            z_nodes = 0.5 * (np.asarray(rule.nodes) + 1.0)
            generated_nodes = np.asarray(
                [
                    (float(point[0]), float(point[1]), float(z))
                    for point in np.asarray(triangle.nodes)
                    for z in z_nodes
                ]
            )
            if (p, q) == (1, 1):
                nodes = np.asarray(reference_cell_topology("prism").vertices)
                permutation = tuple(
                    int(
                        np.flatnonzero(
                            np.max(np.abs(generated_nodes - point), axis=1) <= 2.0e-12
                        )[0]
                    )
                    for point in nodes
                )
            else:
                nodes = generated_nodes
                permutation = tuple(range(nodes.shape[0]))
            modal_indices: tuple[tuple[int, int, int], ...] = ()
            coefficients = np.zeros((0, 0))
            condition_number = 1.0
        else:
            height_rule = legendre_rule_data(p + 1, "lobatto")
            height_nodes = 0.5 * (np.asarray(height_rule.nodes) + 1.0)
            pyramid_nodes = []
            for layer, height in enumerate(height_nodes):
                cross_degree = p - layer
                if cross_degree == 0:
                    cross_nodes = np.asarray((0.5,))
                else:
                    cross_rule = legendre_rule_data(cross_degree + 1, "lobatto")
                    cross_nodes = 0.5 * (np.asarray(cross_rule.nodes) + 1.0)
                scale = 1.0 - height
                pyramid_nodes.extend(
                    (
                        scale * first + 0.5 * height,
                        scale * second + 0.5 * height,
                        height,
                    )
                    for first in cross_nodes
                    for second in cross_nodes
                )
            nodes = (
                np.asarray(reference_cell_topology("pyramid").vertices)
                if p == 1
                else np.asarray(pyramid_nodes)
            )
            permutation = tuple(range(nodes.shape[0]))
            modal_indices = tuple(
                (first, second, height)
                for first in range(p + 1)
                for second in range(p + 1)
                for height in range(p - max(first, second) + 1)
            )
            modal, _modal_gradients = _pyramid_modal_tabulation(nodes, modal_indices)
            if modal.shape[0] != modal.shape[1]:
                raise ValueError("Pyramid node and modal dimensions must match.")
            coefficients = np.linalg.solve(modal, np.eye(modal.shape[0]))
            condition_number = float(np.linalg.cond(modal))

        self.cell_kind = kind
        self.degree = max(p, q)
        self.orders = (p, q)
        self.nodes = jnp.asarray(nodes)
        self.basis_permutation = permutation
        self.modal_indices = modal_indices
        self.coefficients = jnp.asarray(coefficients)
        self.condition_number = condition_number
        self.family_id = canonical_fingerprint(
            {
                "kind": "hybrid-reference-family",
                "cell_kind": kind,
                "degree": max(p, q),
                "orders": (p, q),
                "basis": (
                    "simplex-times-legendre"
                    if kind == "prism"
                    else "bergot-cohen-durufle-rational-pyramid"
                ),
                "nodes": array_tree_fingerprint(nodes),
                "condition_number": condition_number,
            }
        )

    def tabulate_with_gradients(self, points: ArrayLike, /) -> tuple[Array, Array]:
        points_ = jnp.asarray(points)
        if points_.ndim != 2 or points_.shape[-1] != 3:
            raise ValueError("Hybrid reference points must have shape (n, 3).")
        if self.cell_kind == "pyramid":
            modal, modal_gradients = _pyramid_modal_tabulation(
                np.asarray(points_), self.modal_indices
            )
            coefficients = np.asarray(self.coefficients)
            values = modal @ coefficients
            gradients = np.stack(
                tuple(modal_gradients[..., axis] @ coefficients for axis in range(3)),
                axis=-1,
            )
            return jnp.asarray(values), jnp.asarray(gradients)

        triangle_degree, axial_degree = self.orders
        triangle = SimplexNodalFamily("triangle", triangle_degree)
        triangle_values, triangle_gradients = triangle.tabulate(points_[..., :2])
        rule = legendre_rule_data(axial_degree + 1, "lobatto")
        z_nodes = 0.5 * (jnp.asarray(rule.nodes) + 1.0)
        z_values, z_gradients = lagrange_1d_tabulation(z_nodes, points_[..., 2])
        values = ein.contract(
            "qi,qj->qij", triangle_values, z_values, backend="jax"
        ).reshape((points_.shape[0], -1))
        horizontal = ein.contract(
            "qid,qj->qijd", triangle_gradients, z_values, backend="jax"
        )
        vertical = ein.contract(
            "qi,qj->qij", triangle_values, z_gradients, backend="jax"
        )[..., None]
        gradients = jnp.concatenate((horizontal, vertical), axis=-1).reshape(
            (points_.shape[0], -1, 3)
        )
        permutation = jnp.asarray(self.basis_permutation, dtype=jnp.int32)
        return values[:, permutation], gradients[:, permutation]

    def tabulate(self, points: ArrayLike, /) -> Array:
        return self.tabulate_with_gradients(points)[0]

    def _entity_support(self, point: np.ndarray, /) -> frozenset[int]:
        tolerance = 2.0e-10
        if self.cell_kind == "prism":
            height = float(point[2])
            triangle = np.asarray(point[:2])
            barycentric = np.asarray(
                (1.0 - triangle[0] - triangle[1], triangle[0], triangle[1])
            )
            triangle_support = tuple(
                index for index, value in enumerate(barycentric) if value > tolerance
            )
            if height <= tolerance:
                return frozenset(triangle_support)
            if height >= 1.0 - tolerance:
                return frozenset(index + 3 for index in triangle_support)
            return frozenset(
                value for index in triangle_support for value in (index, index + 3)
            )

        height = float(point[2])
        if height >= 1.0 - tolerance:
            return frozenset((4,))
        scale = 1.0 - height
        first = (float(point[0]) - 0.5 * height) / scale
        second = (float(point[1]) - 0.5 * height) / scale
        base_weights = np.asarray(
            (
                (1.0 - first) * (1.0 - second),
                first * (1.0 - second),
                first * second,
                (1.0 - first) * second,
            )
        )
        base_support = frozenset(
            index for index, value in enumerate(base_weights) if value > tolerance
        )
        return base_support if height <= tolerance else frozenset((*base_support, 4))

    def finite_element(self, /, *, conformity: str = "H1") -> FiniteElementSpec:
        if conformity not in ("H1", "L2"):
            raise ValueError("Hybrid finite elements support H1 or L2 conformity.")
        topology = reference_cell_topology(self.cell_kind)
        entities = [[[] for _entity in dimension] for dimension in topology.entities]
        if conformity == "H1":
            entity_sets = [
                tuple(frozenset(entity) for entity in dimension)
                for dimension in topology.entities
            ]
            for dof, point in enumerate(np.asarray(self.nodes)):
                support = self._entity_support(point)
                matches = tuple(
                    (dimension, entities_.index(support))
                    for dimension, entities_ in enumerate(entity_sets)
                    if support in entities_
                )
                if len(matches) != 1:
                    raise ValueError(
                        "Hybrid nodal support has ambiguous entity ownership."
                    )
                dimension, entity = matches[0]
                entities[dimension][entity].append(dof)
        else:
            entities[-1][0].extend(range(self.nodes.shape[0]))
        return FiniteElementSpec(
            "HybridLagrange",
            self.cell_kind,
            self.degree,
            self.nodes,
            tuple(tuple(tuple(values) for values in dimension) for dimension in entities),
            conformity=conformity,
            representation="point_value",
            tabulator=self.tabulate_with_gradients,
            tabulator_id=self.family_id,
        )


class LevelSetCutQuadrature(StrictModule, NonTrainableState):
    points: Array
    weights: Array
    active: Array
    volume_fraction: Array

    def __init__(self, points: ArrayLike, weights: ArrayLike, level_set: ArrayLike, /):
        points_ = jnp.asarray(points)
        weights_ = jnp.asarray(weights)
        phi = jnp.asarray(level_set)
        if weights_.shape != phi.shape or points_.shape[:-1] != phi.shape:
            raise ValueError(
                "Cut quadrature points, weights, and level-set values disagree."
            )
        active = phi <= 0.0
        selected = jnp.where(active, weights_, 0.0)
        self.points = points_
        self.weights = selected
        self.active = active
        self.volume_fraction = jnp.sum(selected) / jnp.sum(weights_)


class TensorDeRhamTransferPlan(StrictModule, NonTrainableState):
    source: TensorDeRhamComplex
    target: TensorDeRhamComplex
    h1_prolongation: Array
    hcurl_prolongation: Array
    hdiv_prolongation: Array
    l2_prolongation: Array
    commuting_gradient_error: Array
    commuting_curl_error: Array
    commuting_divergence_error: Array

    def __init__(self, source: TensorDeRhamComplex, target: TensorDeRhamComplex, /):
        if source.dimension != target.dimension or source.degree > target.degree:
            raise ValueError(
                "Compatible p transfer requires equal dimensions and nested degree."
            )

        def tensor_embedding(source_degrees, target_degrees):
            source_indices = tuple(
                product(*(range(value + 1) for value in source_degrees))
            )
            target_indices = tuple(
                product(*(range(value + 1) for value in target_degrees))
            )
            target_position = {value: index for index, value in enumerate(target_indices)}
            matrix = np.zeros((len(target_indices), len(source_indices)))
            for column, exponent in enumerate(source_indices):
                matrix[target_position[exponent], column] = 1.0
            return matrix

        def block_embedding(source_components, target_components):
            source_widths = [
                int(np.prod(np.asarray(value) + 1)) for value in source_components
            ]
            target_widths = [
                int(np.prod(np.asarray(value) + 1)) for value in target_components
            ]
            matrix = np.zeros((sum(target_widths), sum(source_widths)))
            source_offset = 0
            target_offset = 0
            for source_degrees, target_degrees, source_width, target_width in zip(
                source_components,
                target_components,
                source_widths,
                target_widths,
                strict=True,
            ):
                matrix[
                    target_offset : target_offset + target_width,
                    source_offset : source_offset + source_width,
                ] = tensor_embedding(source_degrees, target_degrees)
                source_offset += source_width
                target_offset += target_width
            return matrix

        source_p = source.degree
        target_p = target.degree
        dimension = source.dimension
        h1 = tensor_embedding((source_p,) * dimension, (target_p,) * dimension)
        hcurl_source = [
            tuple(
                source_p - 1 if axis == component else source_p
                for axis in range(dimension)
            )
            for component in range(dimension)
        ]
        hcurl_target = [
            tuple(
                target_p - 1 if axis == component else target_p
                for axis in range(dimension)
            )
            for component in range(dimension)
        ]
        hcurl = block_embedding(hcurl_source, hcurl_target)
        if dimension == 2:
            hdiv = hcurl
            l2 = tensor_embedding(
                (source_p - 1, source_p - 1),
                (target_p - 1, target_p - 1),
            )
        else:
            hdiv_source = [
                tuple(
                    source_p if axis == component else source_p - 1 for axis in range(3)
                )
                for component in range(3)
            ]
            hdiv_target = [
                tuple(
                    target_p if axis == component else target_p - 1 for axis in range(3)
                )
                for component in range(3)
            ]
            hdiv = block_embedding(hdiv_source, hdiv_target)
            l2 = tensor_embedding(
                (source_p - 1,) * 3,
                (target_p - 1,) * 3,
            )
        self.source = source
        self.target = target
        self.h1_prolongation = jnp.asarray(h1)
        self.hcurl_prolongation = jnp.asarray(hcurl)
        self.hdiv_prolongation = jnp.asarray(hdiv)
        self.l2_prolongation = jnp.asarray(l2)
        self.commuting_gradient_error = jnp.asarray(
            np.max(
                np.abs(
                    np.asarray(target.gradient) @ h1 - hcurl @ np.asarray(source.gradient)
                ),
                initial=0.0,
            )
        )
        curl_target = l2 if dimension == 2 else hdiv
        curl_error = np.max(
            np.abs(
                np.asarray(target.curl) @ hcurl - curl_target @ np.asarray(source.curl)
            ),
            initial=0.0,
        )
        self.commuting_curl_error = jnp.asarray(curl_error)
        if dimension == 2:
            divergence_error = curl_error
        else:
            divergence_error = np.max(
                np.abs(
                    np.asarray(target.divergence) @ hdiv
                    - l2 @ np.asarray(source.divergence)
                ),
                initial=0.0,
            )
        self.commuting_divergence_error = jnp.asarray(divergence_error)


class CompatibleTraceConstraint(StrictModule, NonTrainableState):
    representation: Literal["tangential", "normal"] = eqx.field(static=True)
    prolongation: Array

    def __init__(
        self,
        representation: Literal["tangential", "normal"],
        master_nodes: ArrayLike,
        side_nodes: ArrayLike,
        /,
    ):
        if representation not in ("tangential", "normal"):
            raise ValueError("Compatible trace representation must be tangential/normal.")
        self.representation = representation
        self.prolongation = tensor_trace_interpolation(master_nodes, side_nodes)

    def expand(self, values: ArrayLike, orientation: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        orientation_ = jnp.asarray(orientation)
        result = self.prolongation @ value
        return result * orientation_.reshape(
            orientation_.shape + (1,) * (result.ndim - orientation_.ndim)
        )


class CompatibleMortarPlan(StrictModule, NonTrainableState):
    left_projection: Array
    right_projection: Array
    commuting_error: Array

    def __init__(
        self,
        left_trace: CompatibleTraceConstraint,
        right_trace: CompatibleTraceConstraint,
        differential_left: ArrayLike,
        differential_right: ArrayLike,
        /,
    ):
        left = np.asarray(left_trace.prolongation)
        right = np.asarray(right_trace.prolongation)
        d_left = np.asarray(differential_left)
        d_right = np.asarray(differential_right)
        if left.shape[0] != right.shape[0]:
            raise ValueError("Compatible mortar projections need a common mortar width.")
        self.left_projection = jnp.asarray(left)
        self.right_projection = jnp.asarray(right)
        self.commuting_error = jnp.asarray(
            np.max(np.abs(left @ d_left - right @ d_right), initial=0.0)
        )


class CompatibleAuxiliaryMultigrid(StrictModule, NonTrainableState):
    injection: Array
    auxiliary_inverse: Array

    def __init__(self, injection: ArrayLike, auxiliary_operator: ArrayLike, /):
        injection_ = np.asarray(injection)
        operator = np.asarray(auxiliary_operator)
        if injection_.ndim != 2 or operator.shape != (
            injection_.shape[1],
            injection_.shape[1],
        ):
            raise ValueError("Compatible auxiliary hierarchy shapes are invalid.")
        self.injection = jnp.asarray(injection_)
        self.auxiliary_inverse = jnp.asarray(
            np.linalg.solve(operator, np.eye(operator.shape[0], dtype=operator.dtype))
        )

    def apply(self, residual: ArrayLike, /) -> Array:
        value = jnp.asarray(residual)
        return self.injection @ (
            self.auxiliary_inverse @ (jnp.swapaxes(self.injection, -1, -2) @ value)
        )


class HybridRefinementPlan(StrictModule, NonTrainableState):
    cell_kind: str = eqx.field(static=True)
    child_maps: tuple[tuple[Array, Array], ...]

    def __init__(
        self, cell_kind: str, child_bounds: Sequence[tuple[ArrayLike, ArrayLike]], /
    ):
        kind = str(cell_kind)
        bounds = tuple(
            (jnp.asarray(lower), jnp.asarray(upper)) for lower, upper in child_bounds
        )
        if kind not in ("triangle", "tetrahedron", "prism", "pyramid") or not bounds:
            raise ValueError("Hybrid refinement requires supported cells and child maps.")
        if any(
            lower.shape != upper.shape or jnp.any(lower >= upper)
            for lower, upper in bounds
        ):
            raise ValueError("Hybrid child bounds are invalid.")
        self.cell_kind = kind
        self.child_maps = bounds


class HybridMortarPlan(StrictModule, NonTrainableState):
    left_interpolation: Array
    right_interpolation: Array
    reproduction_error: Array

    def __init__(
        self,
        left_nodes: ArrayLike,
        right_nodes: ArrayLike,
        mortar_points: ArrayLike,
        degree: int,
        /,
    ):
        left = np.asarray(left_nodes)
        right = np.asarray(right_nodes)
        points = np.asarray(mortar_points)
        p = int(degree)
        dimension = points.shape[1]
        exponents = tuple(
            value for value in product(range(p + 1), repeat=dimension) if sum(value) <= p
        )

        def interpolation(nodes):
            vandermonde = np.stack(
                [
                    np.prod(nodes ** np.asarray(exponent), axis=1)
                    for exponent in exponents
                ],
                axis=1,
            )
            evaluation = np.stack(
                [
                    np.prod(points ** np.asarray(exponent), axis=1)
                    for exponent in exponents
                ],
                axis=1,
            )
            return np.linalg.lstsq(
                vandermonde.T,
                evaluation.T,
                rcond=1.0e-15,
            )[0].T

        left_matrix = interpolation(left)
        right_matrix = interpolation(right)
        error = max(
            float(np.max(np.abs(left_matrix @ np.ones(left.shape[0]) - 1.0))),
            float(np.max(np.abs(right_matrix @ np.ones(right.shape[0]) - 1.0))),
        )
        self.left_interpolation = jnp.asarray(left_matrix)
        self.right_interpolation = jnp.asarray(right_matrix)
        self.reproduction_error = jnp.asarray(error)


class UnfittedAggregationPlan(StrictModule, NonTrainableState):
    target_cells: Array
    valid: Array

    def __init__(
        self,
        volume_fractions: ArrayLike,
        neighbours: ArrayLike,
        /,
        *,
        minimum_fraction: float = 0.1,
    ):
        fractions = np.asarray(volume_fractions)
        neighbours_ = np.asarray(neighbours, dtype=np.int32)
        if fractions.ndim != 1 or neighbours_.shape[0] != fractions.size:
            raise ValueError("Unfitted fractions and neighbours disagree.")
        target = np.arange(fractions.size, dtype=np.int32)
        valid = fractions < minimum_fraction
        for cell in np.flatnonzero(valid):
            candidates = neighbours_[cell][neighbours_[cell] >= 0]
            if candidates.size == 0:
                raise ValueError("Small cut cells require one aggregation neighbour.")
            target[cell] = int(candidates[np.argmax(fractions[candidates])])
        self.target_cells = jnp.asarray(target)
        self.valid = jnp.asarray(valid)

    def aggregate(self, content: ArrayLike, /) -> Array:
        value = jnp.asarray(content)
        result = value
        for cell in range(self.target_cells.size):
            if bool(self.valid[cell]):
                target = int(self.target_cells[cell])
                result = result.at[target].add(result[cell])
                result = result.at[cell].set(0.0)
        return result


class ConservativeMovingInterfaceTransfer(StrictModule, NonTrainableState):
    projection: Array

    def __init__(
        self,
        source_basis: ArrayLike,
        target_basis: ArrayLike,
        physical_weights: ArrayLike,
        /,
    ):
        source = np.asarray(source_basis)
        target = np.asarray(target_basis)
        weights = np.asarray(physical_weights)
        if source.shape[0] != target.shape[0] or weights.shape != (source.shape[0],):
            raise ValueError("Moving-interface basis and weights disagree.")
        target_mass = target.T @ (weights[:, None] * target)
        coupling = target.T @ (weights[:, None] * source)
        self.projection = jnp.asarray(np.linalg.solve(target_mass, coupling))

    def apply(self, values: ArrayLike, /) -> Array:
        return self.projection @ jnp.asarray(values)


def physical_mass_projection(
    source_basis: ArrayLike,
    target_basis: ArrayLike,
    quadrature_weights: ArrayLike,
    jacobian_determinant: ArrayLike,
    /,
) -> Array:
    source = jnp.asarray(source_basis)
    target = jnp.asarray(target_basis)
    weights = jnp.asarray(quadrature_weights) * jnp.asarray(jacobian_determinant)
    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape[0] != target.shape[0]
        or weights.shape != (source.shape[0],)
    ):
        raise ValueError("Physical mass projection basis and metric data disagree.")
    target_mass = jnp.swapaxes(target, -1, -2) @ (weights[:, None] * target)
    coupling = jnp.swapaxes(target, -1, -2) @ (weights[:, None] * source)
    space = ArraySpace((target.shape[1],), dtype=target.dtype)
    operator = DenseLinearOperator(target_mass, source=space, target=space)
    columns = tuple(
        solve(LinearSystem(operator), coupling[:, column]).value
        for column in range(coupling.shape[1])
    )
    return jnp.stack(columns, axis=1)


__all__ = [
    "AnisotropicHPattern",
    "compact_hp_forest",
    "CompatibleAuxiliaryMultigrid",
    "CompatibleMortarPlan",
    "CompatibleTraceConstraint",
    "ConservativeMovingInterfaceTransfer",
    "GeometryOrderAdaptation",
    "HybridMortarPlan",
    "HybridRefinementPlan",
    "HybridReferenceFamily",
    "LevelSetCutQuadrature",
    "NIrregularMortarPlan",
    "TensorCompatibleFamily",
    "TensorDeRhamComplex",
    "TensorDeRhamTransferPlan",
    "TensorPiolaMap",
    "tensor_hcurl_family",
    "tensor_hdiv_family",
    "UnfittedAggregationPlan",
    "refine_anisotropic_hp_cells",
    "physical_mass_projection",
    "resize_hp_forest",
]
