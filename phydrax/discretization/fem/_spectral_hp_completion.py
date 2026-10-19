#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import integer
from ...linalg import (
    ArraySpace,
    DenseLinearOperator,
    LinearSystem,
    RHSLayout,
    solve,
)
from ...typing import checked
from . import _hp_runtime as hp_runtime_owner
from ._hp import (
    FiniteElementHPLineage,
    FiniteElementHPLineageKind,
    FiniteElementHPTopology,
)
from ._hp_admission_types import hp_admission_types
from ._hp_forest_preparation import (
    multilinear_cell_map,
    prepare_hp_forest_edit,
    tensor_corner_points,
)


if TYPE_CHECKING:
    from ...meshing._decision import PhysicalErrorEvidence, SolverAwareDecision


class AnisotropicHPattern(StrictModule, NonTrainableState):
    axes: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    child_ordinals: tuple[int, ...] = eqx.field(static=True)
    pattern_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, axes: Sequence[int], /) -> None:
        dimension_ = int(dimension)
        axes_ = tuple(sorted(int(axis) for axis in axes))
        if (
            dimension_ not in (2, 3)
            or not axes_
            or any(axis < 0 or axis >= dimension_ for axis in axes_)
            or len(set(axes_)) != len(axes_)
        ):
            raise ValueError(
                "AnisotropicHPattern.axes must be unique tensor axes; got Anisotropic h axes must be unique tensor axes."
            )
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


@checked
def refine_anisotropic_hp_cells(
    topology: FiniteElementHPTopology,
    geometry: hp_runtime_owner.FiniteElementHPGeometry,
    marked_cell_ids: ArrayLike,
    pattern: AnisotropicHPattern,
    /,
    *,
    target_degrees: ArrayLike | None = None,
) -> hp_runtime_owner.FiniteElementHPRefinementResult:
    if (
        pattern.dimension != topology.dimension
        or geometry.topology_id != topology.topology_id
    ):
        raise ValueError(
            "refine_anisotropic_hp_cells.topology must agree with the declared geometry, topology, and field layout; got Anisotropic pattern, topology, and geometry disagree."
        )
    prepared = prepare_hp_forest_edit(topology)
    identifiers, allocated, active = (
        prepared.identifiers,
        prepared.allocated,
        prepared.active,
    )
    degrees, roots, paths = prepared.degrees, prepared.roots, prepared.paths
    levels, parents = prepared.levels, prepared.parents
    children, child_valid = prepared.children, prepared.child_valid
    vertices = np.array(geometry.cell_vertices, copy=True)
    lower = np.array(geometry.reference_lower, copy=True)
    upper = np.array(geometry.reference_upper, copy=True)
    slot_by_id = {int(identifiers[slot]): int(slot) for slot in np.flatnonzero(active)}
    marked = np.asarray(marked_cell_ids, dtype=np.int64)
    unknown = set(marked.tolist()) - set(slot_by_id)
    if marked.ndim != 1 or np.unique(marked).size != marked.size or unknown:
        raise ValueError(
            f"refine_anisotropic_hp_cells.configuration must have valid anisotropic h marked ids; got Anisotropic h marked IDs are invalid: {sorted(unknown)!r}."
        )
    marked_slots = np.asarray(
        sorted(
            (slot_by_id[int(value)] for value in marked),
            key=lambda slot: (
                integer(
                    roots[integer(slot, "cell slot")],
                    "root cell ID",
                ),
                integer(paths[integer(slot, "cell slot")], "path code"),
            ),
        ),
        dtype=np.int32,
    )
    free = np.flatnonzero(~allocated)
    required = marked_slots.size * len(pattern.child_ordinals)
    if free.size < required:
        raise ValueError(
            "refine_anisotropic_hp_cells.configuration must fit the declared capacity; got Anisotropic h refinement exceeds forest capacity."
        )
    requested_degrees = (
        None if target_degrees is None else np.asarray(target_degrees, dtype=np.int32)
    )
    if requested_degrees is not None and requested_degrees.shape != (
        marked_slots.size,
        topology.dimension,
    ):
        raise ValueError(
            "refine_anisotropic_hp_cells.target_degrees must contain one tuple per marked cell; got target_degrees must contain one tuple per marked cell."
        )
    next_global = int(np.max(identifiers[allocated], initial=-1)) + 1
    source_routes = []
    target_routes = []
    relations: list[FiniteElementHPLineageKind] = []
    marked_set = set(marked_slots.tolist())
    for slot in np.flatnonzero(active):
        if int(slot) not in marked_set:
            source_routes.append(int(slot))
            target_routes.append(int(slot))
            relations.append("unchanged")
    cursor = 0
    corners = tensor_corner_points(topology.dimension)
    for marked_index, parent_value in enumerate(marked_slots.tolist()):
        parent = int(parent_value)
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
            vertices[child] = multilinear_cell_map(vertices[parent], child_points)
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
    refined_geometry = hp_runtime_owner.FiniteElementHPGeometry(
        refined_topology, vertices, lower, upper
    )
    if (
        hp_runtime_owner.finite_element_hp_balance_error(
            refined_topology, refined_geometry
        )
        > 1
    ):
        raise ValueError(
            "refine_anisotropic_hp_cells.configuration must satisfy the declared balance and admission constraints; got Anisotropic h refinement violates directional 2:1 balance."
        )
    lineage = FiniteElementHPLineage(
        topology.topology_id,
        refined_topology.topology_id,
        topology.capacity,
        refined_topology.capacity,
        np.asarray(source_routes, dtype=np.int32),
        np.asarray(target_routes, dtype=np.int32),
        tuple(relations),
    )
    return hp_runtime_owner.FiniteElementHPRefinementResult(
        refined_topology,
        refined_geometry,
        lineage,
        marked_slots,
        np.empty((0,), dtype=np.int32),
    )


@checked
def resize_hp_forest(
    topology: FiniteElementHPTopology,
    geometry: hp_runtime_owner.FiniteElementHPGeometry,
    new_capacity: int,
    /,
) -> tuple[FiniteElementHPTopology, hp_runtime_owner.FiniteElementHPGeometry]:
    capacity = int(new_capacity)
    if capacity < topology.capacity:
        allocated_slots = np.flatnonzero(np.asarray(topology.allocated))
        if allocated_slots.size and int(np.max(allocated_slots)) >= capacity:
            raise ValueError(
                "resize_hp_forest.configuration must provide compact allocated hp slots; got Shrink requires compact allocated hp slots."
            )
    if capacity < 1:
        raise ValueError(
            "resize_hp_forest.configuration must be positive; got hp forest capacity must be positive."
        )

    def pad(vector: ArrayLike, shape: tuple[int, ...], fill: float) -> np.ndarray:
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
    return resized_topology, hp_runtime_owner.FiniteElementHPGeometry(
        resized_topology, vertices, lower, upper
    )


@checked
def compact_hp_forest(
    topology: FiniteElementHPTopology,
    geometry: hp_runtime_owner.FiniteElementHPGeometry,
    /,
) -> tuple[FiniteElementHPTopology, hp_runtime_owner.FiniteElementHPGeometry, Array]:
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
    allocated_new = np.zeros((capacity,), dtype=np.bool_)
    active = np.zeros((capacity,), dtype=np.bool_)
    degrees = np.zeros((capacity, topology.dimension), dtype=np.int32)
    roots = np.full((capacity,), -1, dtype=np.int64)
    paths = np.full((capacity,), -1, dtype=np.int64)
    levels = np.full((capacity,), -1, dtype=np.int32)
    parents = np.full((capacity,), -1, dtype=np.int32)
    children = np.full((capacity, topology.child_capacity), -1, dtype=np.int32)
    child_valid = np.zeros_like(children, dtype=np.bool_)
    vertices = np.zeros(
        (capacity,) + tuple(np.asarray(geometry.cell_vertices).shape[1:]),
        dtype=np.asarray(geometry.cell_vertices).dtype,
    )
    lower = np.zeros((capacity, topology.dimension), dtype=np.float64)
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
        hp_runtime_owner.FiniteElementHPGeometry(compacted, vertices, lower, upper),
        jnp.asarray(old_to_new),
    )


class GeometryOrderAdaptation(StrictModule, NonTrainableState):
    """Reinterpolate the existing coordinate map, not improve source fidelity."""

    source_order: tuple[int, ...] = eqx.field(static=True)
    target_order: tuple[int, ...] = eqx.field(static=True)
    interpolation: Array
    curvature_indicator: Array
    plan_id: str = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)
    solver_decision: SolverAwareDecision | None
    source_revision_id: str | None = eqx.field(static=True)
    target_revision_id: str | None = eqx.field(static=True)
    source_coordinates_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_nodes: ArrayLike,
        target_nodes: ArrayLike,
        source_order: Sequence[int],
        target_order: Sequence[int],
        coordinate_values: ArrayLike,
        /,
        *,
        solver_decision: SolverAwareDecision | None = None,
        source_revision_id: str | None = None,
        target_revision_id: str | None = None,
    ) -> None:

        source = np.asarray(source_nodes)
        target = np.asarray(target_nodes)
        values = np.asarray(coordinate_values)
        source_order_ = tuple(source_order)
        target_order_ = tuple(target_order)
        interpolation = np.asarray(
            hp_runtime_owner.tensor_trace_interpolation(source, target)
        )
        if values.shape[-2] != source.shape[0]:
            raise ValueError(
                "GeometryOrderAdaptation.configuration must match source nodes; got Geometry coordinate values do not match source nodes."
            )
        target_values = ein.contract("qi,...id->...qd", interpolation, values)
        linear = multilinear_cell_map(
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
        plan_id = canonical_fingerprint(
            {
                "kind": "geometry-order-adaptation",
                "source_order": source_order_,
                "target_order": target_order_,
                "interpolation": array_tree_fingerprint(interpolation),
                "curvature": float(indicator),
            }
        )
        source_coordinates = array_tree_fingerprint(values)
        source_coordinates_id = canonical_fingerprint(source_coordinates)
        candidate_id = canonical_fingerprint(
            {
                "kind": "geometry-order-candidate",
                "plan": plan_id,
                "source_nodes": array_tree_fingerprint(source),
                "target_nodes": array_tree_fingerprint(target),
                "source_coordinates": source_coordinates,
            }
        )
        if solver_decision is not None:
            if not isinstance(solver_decision, hp_admission_types().solver_decision):
                raise TypeError(
                    "GeometryOrderAdaptation.solver_decision must be SolverAwareDecision or None; got solver_decision must be SolverAwareDecision or None."
                )
            if not source_revision_id or not target_revision_id:
                raise ValueError(
                    "GeometryOrderAdaptation.configuration must provide prepared revision identities; got Geometry-order admission requires prepared revision identities."
                )
            selected = solver_decision.require_selected(
                source_revision_id, candidate_id, target_revision_id
            )
            if selected.action != hp_admission_types().action.GEOMETRY_ORDER:
                raise ValueError(
                    "GeometryOrderAdaptation.configuration must select a geometry-order action; got Geometry interpolation cannot realize a field-p or h action."
                )
            if selected.error.geometry_error < solver_decision.baseline.geometry_error:
                raise ValueError(
                    "GeometryOrderAdaptation.configuration must use independently certified source-geometry improvement; got Map interpolation cannot claim improved source-geometry fidelity."
                )
            if (
                len(source_order_) != source.shape[1]
                or len(target_order_) != target.shape[1]
                or any(order < 1 for order in source_order_ + target_order_)
                or source.shape[0] != np.prod(np.asarray(source_order_) + 1)
                or target.shape[0] != np.prod(np.asarray(target_order_) + 1)
                or any(
                    new < old
                    for old, new in zip(source_order_, target_order_, strict=True)
                )
            ):
                raise ValueError(
                    "GeometryOrderAdaptation.configuration must provide compatible tensor orders; got Admitted geometry elevation requires compatible tensor orders."
                )
        elif source_revision_id is not None or target_revision_id is not None:
            raise ValueError(
                "GeometryOrderAdaptation.configuration must provide solver admission evidence; got Geometry revision identities require solver admission evidence."
            )
        self.plan_id = plan_id
        self.candidate_id = candidate_id
        self.solver_decision = solver_decision
        self.source_revision_id = source_revision_id
        self.target_revision_id = target_revision_id
        self.source_coordinates_id = source_coordinates_id

    def apply(
        self,
        coordinate_values: ArrayLike,
        /,
        *,
        reanalysis: PhysicalErrorEvidence | None = None,
    ) -> Array:
        if self.solver_decision is not None:
            if (
                canonical_fingerprint(
                    array_tree_fingerprint(np.asarray(coordinate_values))
                )
                != self.source_coordinates_id
            ):
                raise ValueError(
                    "GeometryOrderAdaptation.apply.configuration must match the retained revision and layout; got Admitted geometry interpolation has stale source coordinates."
                )
            if reanalysis is None:
                raise ValueError(
                    "GeometryOrderAdaptation.apply.reanalysis must provide independent physical reanalysis; got Geometry-order publication requires independent physical reanalysis."
                )
            self.solver_decision.require_reanalysis(reanalysis)
            if reanalysis.geometry_error < self.solver_decision.baseline.geometry_error:
                raise ValueError(
                    "GeometryOrderAdaptation.apply.configuration must certify source-fidelity improvement; got Interpolated geometry does not certify source-fidelity improvement."
                )
        elif reanalysis is not None:
            raise ValueError(
                "GeometryOrderAdaptation.apply.reanalysis must provide a solver-aware decision; got Geometry reanalysis requires a solver-aware decision."
            )
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
    ) -> None:
        nodes = np.asarray(coarse_nodes)
        points = tuple(np.asarray(value) for value in patch_points)
        weights = tuple(np.asarray(value) for value in patch_weights)
        if not points or len(points) != len(weights):
            raise ValueError(
                "NIrregularMortarPlan.configuration must provide points and weights per patch; got n-irregular mortars require points and weights per patch."
            )
        matrices = tuple(
            np.asarray(hp_runtime_owner.tensor_trace_interpolation(nodes, value))
            for value in points
        )
        if any(
            weight.shape != (point.shape[0],)
            for point, weight in zip(points, weights, strict=True)
        ):
            raise ValueError(
                "NIrregularMortarPlan.configuration must match patch points; got n-irregular patch weights must match patch points."
            )
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


class LevelSetCutQuadrature(StrictModule, NonTrainableState):
    points: Array
    weights: Array
    active: Array
    volume_fraction: Array

    def __init__(
        self, points: ArrayLike, weights: ArrayLike, level_set: ArrayLike, /
    ) -> None:
        points_ = jnp.asarray(points)
        weights_ = jnp.asarray(weights)
        phi = jnp.asarray(level_set)
        if weights_.shape != phi.shape or points_.shape[:-1] != phi.shape:
            raise ValueError(
                "LevelSetCutQuadrature.points must agree with the declared geometry, topology, and field layout; got Cut quadrature points, weights, and level-set values disagree."
            )
        active = phi <= 0.0
        selected = jnp.where(active, weights_, 0.0)
        self.points = points_
        self.weights = selected
        self.active = active
        self.volume_fraction = jnp.sum(selected) / jnp.sum(weights_)


class HybridRefinementPlan(StrictModule, NonTrainableState):
    cell_kind: str = eqx.field(static=True)
    child_maps: tuple[tuple[Array, Array], ...]

    def __init__(
        self, cell_kind: str, child_bounds: Sequence[tuple[ArrayLike, ArrayLike]], /
    ) -> None:
        kind = str(cell_kind)
        bounds = tuple(
            (jnp.asarray(lower), jnp.asarray(upper)) for lower, upper in child_bounds
        )
        if kind not in ("triangle", "tetrahedron", "prism", "pyramid") or not bounds:
            raise ValueError(
                "HybridRefinementPlan.configuration must provide supported cells and child maps; got Hybrid refinement requires supported cells and child maps."
            )
        if any(
            lower.shape != upper.shape or jnp.any(lower >= upper)
            for lower, upper in bounds
        ):
            raise ValueError(
                "HybridRefinementPlan.configuration must have valid hybrid child bounds; got Hybrid child bounds are invalid."
            )
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
    ) -> None:
        left = np.asarray(left_nodes)
        right = np.asarray(right_nodes)
        points = np.asarray(mortar_points)
        p = int(degree)
        dimension = points.shape[1]
        exponents = tuple(
            value for value in product(range(p + 1), repeat=dimension) if sum(value) <= p
        )

        def interpolation(nodes: np.ndarray) -> np.ndarray:
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
        neighbors: ArrayLike,
        /,
        *,
        minimum_fraction: float = 0.1,
    ) -> None:
        fractions = np.asarray(volume_fractions)
        neighbors_ = np.asarray(neighbors, dtype=np.int32)
        if fractions.ndim != 1 or neighbors_.shape[0] != fractions.size:
            raise ValueError(
                "UnfittedAggregationPlan.neighbors must agree with the declared geometry, topology, and field layout; got Unfitted fractions and neighbors disagree."
            )
        target = np.arange(fractions.size, dtype=np.int32)
        valid = fractions < minimum_fraction
        for cell in np.flatnonzero(valid):
            candidates = neighbors_[cell][neighbors_[cell] >= 0]
            if candidates.size == 0:
                raise ValueError(
                    "UnfittedAggregationPlan.configuration must provide one aggregation neighbor; got Small cut cells require one aggregation neighbor."
                )
            target[cell] = int(candidates[np.argmax(fractions[candidates])])
        for cell in np.flatnonzero(valid):
            root = int(target[cell])
            visited = {int(cell)}
            while valid[root]:
                if root in visited:
                    raise ValueError(
                        "UnfittedAggregationPlan.configuration must have an acyclic aggregation relation; got Cut-cell aggregation targets contain a cycle."
                    )
                visited.add(root)
                root = int(target[root])
            target[cell] = root
        self.target_cells = jnp.asarray(target)
        self.valid = jnp.asarray(valid)

    def aggregate(self, content: ArrayLike, /) -> Array:
        value = jnp.asarray(content)
        trailing = (1,) * (value.ndim - 1)
        moved = jnp.where(
            self.valid.reshape(self.valid.shape + trailing),
            value,
            0.0,
        )
        retained = jnp.where(
            self.valid.reshape(self.valid.shape + trailing),
            0.0,
            value,
        )
        return retained.at[self.target_cells].add(moved)


class ConservativeMovingInterfaceTransfer(StrictModule, NonTrainableState):
    projection: Array

    def __init__(
        self,
        source_basis: ArrayLike,
        target_basis: ArrayLike,
        physical_weights: ArrayLike,
        /,
    ) -> None:
        source = np.asarray(source_basis)
        target = np.asarray(target_basis)
        weights = np.asarray(physical_weights)
        if source.shape[0] != target.shape[0] or weights.shape != (source.shape[0],):
            raise ValueError(
                "ConservativeMovingInterfaceTransfer.configuration must agree with the declared geometry, topology, and field layout; got Moving-interface basis and weights disagree."
            )
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
        raise ValueError(
            "physical_mass_projection.configuration must agree with the declared geometry, topology, and field layout; got Physical mass projection basis and metric data disagree."
        )
    target_mass = jnp.swapaxes(target, -1, -2) @ (weights[:, None] * target)
    coupling = jnp.swapaxes(target, -1, -2) @ (weights[:, None] * source)
    space = ArraySpace((target.shape[1],), dtype=target.dtype)
    operator = DenseLinearOperator(target_mass, source=space, target=space)
    result = solve(
        LinearSystem(operator),
        coupling,
        rhs_layout=RHSLayout((coupling.shape[1],)),
    )
    return eqx.error_if(
        result.value,
        jnp.any(~result.successful) | jnp.any(~jnp.isfinite(result.value)),
        "physical_mass_projection.solve must succeed for every RHS; got failed native mass solve",
    )


__all__ = [
    "AnisotropicHPattern",
    "compact_hp_forest",
    "ConservativeMovingInterfaceTransfer",
    "GeometryOrderAdaptation",
    "HybridMortarPlan",
    "HybridRefinementPlan",
    "LevelSetCutQuadrature",
    "NIrregularMortarPlan",
    "UnfittedAggregationPlan",
    "refine_anisotropic_hp_cells",
    "physical_mass_projection",
    "resize_hp_forest",
]
