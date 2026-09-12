#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scalar Cartesian diffusion on one fixed block-AMR topology."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    AbstractPreconditioner,
    AbstractPreconditionerBuilder,
    GalerkinHierarchyBuilder,
    KernelCertificate,
    LinearSubspace,
    LinearSystem,
    MultigridCyclePolicy,
    MultigridHierarchyBuilder,
    MultigridLevelBuilder,
    NullspacePolicy,
    OperatorCapabilities,
    OperatorProperties,
    PreconditionerProperties,
    PyTreeSpace,
)
from ..amr._composite import CompositeAMRCellLayout
from ..amr._core import BlockHierarchyTopology
from ._diffusion import ConservativeBoundaryCondition, ConservativeBoundaryKind
from ._precision import FiniteVolumePrecisionPolicy


CompositeBoundaryInput: TypeAlias = Mapping[
    str,
    tuple[
        ConservativeBoundaryCondition | ConservativeBoundaryKind,
        ConservativeBoundaryCondition | ConservativeBoundaryKind,
    ],
]
CompositeBoundaryData: TypeAlias = Mapping[str, tuple[Any, Any]]
CompositeCoarseOperatorSource = Literal["direct", "galerkin"]


def _checked(value: Array, invalid: Array, message: str, /) -> Array:
    if isinstance(invalid, jax.core.Tracer):
        return eqx.error_if(value, invalid, message)
    if bool(invalid):
        raise ValueError(message)
    return value


def _condition(
    value: ConservativeBoundaryCondition | ConservativeBoundaryKind,
    /,
) -> ConservativeBoundaryCondition:
    condition = (
        value
        if isinstance(value, ConservativeBoundaryCondition)
        else ConservativeBoundaryCondition(value)
    )
    if condition.kind == "robin":
        raise ValueError(
            "Composite AMR diffusion supports periodic, Dirichlet, and "
            "Neumann boundaries."
        )
    if condition.kind == "dirichlet" and (
        condition.alpha != 1.0 or condition.beta != 0.0
    ):
        raise ValueError(
            "Composite AMR Dirichlet boundaries use the canonical unit value law."
        )
    if condition.kind == "neumann" and (condition.alpha != 0.0 or condition.beta != 1.0):
        raise ValueError(
            "Composite AMR Neumann boundaries use canonical outward flux data."
        )
    return condition


def _boundaries(
    layout: CompositeAMRCellLayout,
    supplied: CompositeBoundaryInput | None,
) -> tuple[tuple[ConservativeBoundaryCondition, ConservativeBoundaryCondition], ...]:
    grid = layout.topology.plan.grid
    values = {} if supplied is None else dict(supplied)
    unknown = set(values).difference(grid.axis_names)
    if unknown:
        raise ValueError(
            f"Composite AMR boundaries reference unknown axes {sorted(unknown)!r}."
        )
    output = []
    for axis, name in enumerate(grid.axis_names):
        periodic = layout.topology.plan.periodic_axes[axis]
        default: tuple[ConservativeBoundaryKind, ConservativeBoundaryKind] = (
            ("periodic", "periodic") if periodic else ("neumann", "neumann")
        )
        raw = values.get(name, default)
        if len(raw) != 2:
            raise ValueError(
                "Each composite AMR axis needs lower and upper boundary laws."
            )
        pair = (_condition(raw[0]), _condition(raw[1]))
        if periodic != (pair[0].kind == pair[1].kind == "periodic"):
            raise ValueError(
                "Composite AMR boundary periodicity must match the tensor grid."
            )
        if (pair[0].kind == "periodic") != (pair[1].kind == "periodic"):
            raise ValueError(
                "Composite AMR periodicity must be declared on both axis sides."
            )
        output.append(pair)
    return tuple(output)


class _CompositeAMRFaceRoutes(StrictModule, NonTrainableState):
    edge_left: Array
    edge_right: Array
    edge_axis: Array
    edge_periodic: Array
    edge_area: Array
    edge_left_distance: Array
    edge_right_distance: Array
    edge_level_jump: Array
    boundary_cells: Array
    boundary_axis: Array
    boundary_side: Array
    boundary_area: Array
    boundary_distance: Array
    route_id: str = eqx.field(static=True)

    def __init__(self, layout: CompositeAMRCellLayout, /):
        topology = layout.topology
        dimension = len(topology.plan.grid.shape)
        finest_shape = topology.plan.global_cell_shapes[-1]
        finest_spacing = topology.plan.level_spacings[-1]
        cell_levels = np.empty((layout.cell_count,), dtype=np.int32)
        cell_widths = np.empty((layout.cell_count, dimension), dtype=float)
        lower_faces: list[dict[int, list[tuple[int, tuple[tuple[int, int], ...]]]]] = [
            {} for _ in range(dimension)
        ]
        upper_faces: list[dict[int, list[tuple[int, tuple[tuple[int, int], ...]]]]] = [
            {} for _ in range(dimension)
        ]

        for level, (level_plan, metadata, mask) in enumerate(
            zip(
                topology.plan.levels,
                topology.levels,
                layout.leaf_mask,
                strict=True,
            )
        ):
            offset = layout.cell_offsets[level]
            count = int(np.count_nonzero(np.asarray(metadata.active)))
            logical = np.asarray(metadata.logical_indices, dtype=np.int32)
            mask_host = np.asarray(mask, dtype=bool)
            block_cells = prod(level_plan.block_shape)
            scale = tuple(
                fine // coarse
                for fine, coarse in zip(
                    finest_shape,
                    topology.plan.global_cell_shapes[level],
                    strict=True,
                )
            )
            start = offset
            stop = offset + level_plan.maximum_blocks * block_cells
            cell_levels[start:stop] = level
            cell_widths[start:stop] = topology.plan.level_spacings[level]
            for slot in range(count):
                block_origin = np.asarray(logical[slot]) * np.asarray(
                    level_plan.block_shape
                )
                for local_value in np.argwhere(mask_host[slot]):
                    local = tuple(int(value) for value in local_value)
                    cell = (
                        offset
                        + slot * block_cells
                        + int(np.ravel_multi_index(local, level_plan.block_shape))
                    )
                    global_index = block_origin + local_value
                    lower = tuple(
                        int(index) * factor
                        for index, factor in zip(global_index, scale, strict=True)
                    )
                    upper = tuple(
                        (int(index) + 1) * factor
                        for index, factor in zip(global_index, scale, strict=True)
                    )
                    for axis in range(dimension):
                        box = tuple(
                            (lower[other], upper[other])
                            for other in range(dimension)
                            if other != axis
                        )
                        lower_faces[axis].setdefault(lower[axis], []).append((cell, box))
                        upper_faces[axis].setdefault(upper[axis], []).append((cell, box))

        edge_counts: dict[tuple[int, int, int, bool], int] = {}

        def box_units(box: tuple[tuple[int, int], ...], /) -> int:
            return prod(stop - start for start, stop in box)

        def overlap_units(
            left: tuple[tuple[int, int], ...],
            right: tuple[tuple[int, int], ...],
            /,
        ) -> int:
            return prod(
                max(0, min(left_stop, right_stop) - max(left_start, right_start))
                for (left_start, left_stop), (right_start, right_stop) in zip(
                    left, right, strict=True
                )
            )

        def add_route(
            left: int,
            right: int,
            axis: int,
            units: int,
            /,
            *,
            periodic: bool,
        ) -> None:
            if left == right:
                return
            key = (left, right, axis, periodic)
            edge_counts[key] = edge_counts.get(key, 0) + units

        def match_faces(
            upper: list[tuple[int, tuple[tuple[int, int], ...]]],
            lower: list[tuple[int, tuple[tuple[int, int], ...]]],
            axis: int,
            /,
            *,
            periodic: bool,
        ) -> None:
            upper_covered = np.zeros((len(upper),), dtype=np.int64)
            lower_covered = np.zeros((len(lower),), dtype=np.int64)
            lower_by_box = {box: index for index, (_, box) in enumerate(lower)}
            exact_upper: set[int] = set()
            exact_lower: set[int] = set()
            for upper_index, (left, box) in enumerate(upper):
                lower_index = lower_by_box.get(box)
                if lower_index is None:
                    continue
                right = lower[lower_index][0]
                units = box_units(box)
                add_route(
                    left,
                    right,
                    axis,
                    units,
                    periodic=periodic,
                )
                upper_covered[upper_index] += units
                lower_covered[lower_index] += units
                exact_upper.add(upper_index)
                exact_lower.add(lower_index)
            remaining_upper = [
                index for index in range(len(upper)) if index not in exact_upper
            ]
            remaining_lower = [
                index for index in range(len(lower)) if index not in exact_lower
            ]
            for upper_index in remaining_upper:
                left, left_box = upper[upper_index]
                for lower_index in remaining_lower:
                    right, right_box = lower[lower_index]
                    units = overlap_units(left_box, right_box)
                    if units == 0:
                        continue
                    add_route(
                        left,
                        right,
                        axis,
                        units,
                        periodic=periodic,
                    )
                    upper_covered[upper_index] += units
                    lower_covered[lower_index] += units
            upper_expected = np.asarray(
                [box_units(box) for _, box in upper], dtype=np.int64
            )
            lower_expected = np.asarray(
                [box_units(box) for _, box in lower], dtype=np.int64
            )
            if not np.array_equal(upper_covered, upper_expected) or not np.array_equal(
                lower_covered, lower_expected
            ):
                raise ValueError(
                    "Composite AMR leaf faces do not form a complete conforming "
                    "same-level/coarse-fine mortar partition."
                )

        boundary_counts: dict[tuple[int, int, int], int] = {}
        for axis in range(dimension):
            extent = finest_shape[axis]
            internal_upper = set(upper_faces[axis]).difference((extent,))
            internal_lower = set(lower_faces[axis]).difference((0,))
            if internal_upper != internal_lower:
                raise ValueError(
                    "Composite AMR leaf coverage has unmatched interior faces."
                )
            for coordinate in sorted(internal_upper):
                match_faces(
                    upper_faces[axis][coordinate],
                    lower_faces[axis][coordinate],
                    axis,
                    periodic=False,
                )
            if topology.plan.periodic_axes[axis]:
                match_faces(
                    upper_faces[axis].get(extent, []),
                    lower_faces[axis].get(0, []),
                    axis,
                    periodic=True,
                )
            else:
                for side, faces in enumerate(
                    (
                        lower_faces[axis].get(0, []),
                        upper_faces[axis].get(extent, []),
                    )
                ):
                    for cell, box in faces:
                        key = (cell, axis, side)
                        boundary_counts[key] = boundary_counts.get(key, 0) + box_units(
                            box
                        )

        edge_keys = tuple(sorted(edge_counts))
        left = np.asarray([key[0] for key in edge_keys], dtype=np.int32)
        right = np.asarray([key[1] for key in edge_keys], dtype=np.int32)
        edge_axis = np.asarray([key[2] for key in edge_keys], dtype=np.int32)
        edge_periodic = np.asarray([key[3] for key in edge_keys], dtype=bool)
        micro_areas = np.asarray(
            [
                prod(finest_spacing[other] for other in range(dimension) if other != axis)
                for axis in edge_axis
            ],
            dtype=float,
        )
        edge_area = (
            np.asarray([edge_counts[key] for key in edge_keys], dtype=float) * micro_areas
        )
        left_distance = (
            0.5 * cell_widths[left, edge_axis] if edge_keys else np.empty((0,))
        )
        right_distance = (
            0.5 * cell_widths[right, edge_axis] if edge_keys else np.empty((0,))
        )
        level_jump = (
            cell_levels[left] != cell_levels[right]
            if edge_keys
            else np.empty((0,), dtype=bool)
        )
        boundary_keys = tuple(sorted(boundary_counts))
        boundary_cells = np.asarray([key[0] for key in boundary_keys], dtype=np.int32)
        boundary_axis = np.asarray([key[1] for key in boundary_keys], dtype=np.int32)
        boundary_side = np.asarray([key[2] for key in boundary_keys], dtype=np.int8)
        boundary_micro_areas = np.asarray(
            [
                prod(finest_spacing[other] for other in range(dimension) if other != axis)
                for axis in boundary_axis
            ],
            dtype=float,
        )
        boundary_area = (
            np.asarray([boundary_counts[key] for key in boundary_keys], dtype=float)
            * boundary_micro_areas
        )
        boundary_distance = (
            0.5 * cell_widths[boundary_cells, boundary_axis]
            if boundary_keys
            else np.empty((0,))
        )
        route_id = canonical_fingerprint(
            {
                "kind": "composite-amr-face-routes",
                "layout": layout.layout_id,
                "edge_left": array_tree_fingerprint(left),
                "edge_right": array_tree_fingerprint(right),
                "edge_axis": array_tree_fingerprint(edge_axis),
                "edge_periodic": array_tree_fingerprint(edge_periodic),
                "edge_area": array_tree_fingerprint(edge_area),
                "edge_left_distance": array_tree_fingerprint(left_distance),
                "edge_right_distance": array_tree_fingerprint(right_distance),
                "edge_level_jump": array_tree_fingerprint(level_jump),
                "boundary_cells": array_tree_fingerprint(boundary_cells),
                "boundary_axis": array_tree_fingerprint(boundary_axis),
                "boundary_side": array_tree_fingerprint(boundary_side),
                "boundary_area": array_tree_fingerprint(boundary_area),
                "boundary_distance": array_tree_fingerprint(boundary_distance),
            }
        )
        dtype = layout.dtype
        self.edge_left = jnp.asarray(left)
        self.edge_right = jnp.asarray(right)
        self.edge_axis = jnp.asarray(edge_axis)
        self.edge_periodic = jnp.asarray(edge_periodic)
        self.edge_area = jnp.asarray(edge_area, dtype=dtype)
        self.edge_left_distance = jnp.asarray(left_distance, dtype=dtype)
        self.edge_right_distance = jnp.asarray(right_distance, dtype=dtype)
        self.edge_level_jump = jnp.asarray(level_jump)
        self.boundary_cells = jnp.asarray(boundary_cells)
        self.boundary_axis = jnp.asarray(boundary_axis)
        self.boundary_side = jnp.asarray(boundary_side)
        self.boundary_area = jnp.asarray(boundary_area, dtype=dtype)
        self.boundary_distance = jnp.asarray(boundary_distance, dtype=dtype)
        self.route_id = route_id


class CompositeAMRDiffusionPlan(StrictModule, NonTrainableState):
    """Homogeneous scalar diffusion policy on one immutable composite layout."""

    layout: CompositeAMRCellLayout
    boundaries: tuple[
        tuple[ConservativeBoundaryCondition, ConservativeBoundaryCondition], ...
    ]
    routes: _CompositeAMRFaceRoutes
    precision: FiniteVolumePrecisionPolicy
    topology_fingerprint: str = eqx.field(static=True)
    route_fingerprint: str = eqx.field(static=True)
    precision_fingerprint: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: CompositeAMRCellLayout,
        /,
        *,
        boundaries: CompositeBoundaryInput | None = None,
        precision: FiniteVolumePrecisionPolicy | None = None,
    ):
        if not isinstance(layout, CompositeAMRCellLayout):
            raise TypeError("Composite AMR diffusion requires CompositeAMRCellLayout.")
        precision_ = (
            FiniteVolumePrecisionPolicy(layout.dtype.name)
            if precision is None
            else precision
        )
        if not isinstance(precision_, FiniteVolumePrecisionPolicy):
            raise TypeError("precision must be FiniteVolumePrecisionPolicy or None.")
        if np.dtype(precision_.storage_dtype) != layout.dtype:
            raise TypeError(
                "Composite AMR layout and finite-volume storage precision must match."
            )
        boundaries_ = _boundaries(layout, boundaries)
        routes = _CompositeAMRFaceRoutes(layout)
        self.layout = layout
        self.boundaries = boundaries_
        self.routes = routes
        self.precision = precision_
        self.topology_fingerprint = layout.topology_fingerprint
        self.route_fingerprint = routes.route_id
        self.precision_fingerprint = precision_.policy_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "composite-amr-diffusion-plan",
                "layout": layout.layout_id,
                "topology": layout.topology_fingerprint,
                "routes": routes.route_id,
                "boundaries": [
                    [lower.condition_id, upper.condition_id]
                    for lower, upper in boundaries_
                ],
                "precision": precision_.policy_id,
            }
        )

    def require_topology(self, topology: BlockHierarchyTopology, /) -> None:
        self.layout.require_topology(topology)

    def prepare(
        self,
        coefficient: ArrayLike | Sequence[ArrayLike],
        /,
    ) -> "PreparedCompositeAMRDiffusion":
        return PreparedCompositeAMRDiffusion(self, coefficient)


class PreparedCompositeAMRDiffusion(AbstractLinearOperator):
    """Measure-self-adjoint ``M^-1 G^T W G`` with inert identity storage rows.

    Interior route weights use the distance-weighted harmonic conductivity.
    Coarse/fine routes are single conservative mortar energies, so their two
    integrated cell contributions cancel exactly.  Minmod FillPatch transfer is
    deliberately absent from this linear operator.
    """

    source: PyTreeSpace
    target: PyTreeSpace
    plan: CompositeAMRDiffusionPlan
    coefficient: tuple[Array, ...]
    edge_weights: Array
    boundary_weights: Array
    boundary_kind: Array
    coefficient_fingerprint: str = eqx.field(static=True)
    numeric_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        plan: CompositeAMRDiffusionPlan,
        coefficient: ArrayLike | Sequence[ArrayLike],
        /,
    ):
        if not isinstance(plan, CompositeAMRDiffusionPlan):
            raise TypeError(
                "Prepared composite diffusion requires CompositeAMRDiffusionPlan."
            )
        coefficient_levels, flat_coefficient = _coefficient(plan, coefficient)
        routes = plan.routes
        left_coefficient = flat_coefficient[routes.edge_left]
        right_coefficient = flat_coefficient[routes.edge_right]
        resistance = (
            routes.edge_left_distance / left_coefficient
            + routes.edge_right_distance / right_coefficient
        )
        edge_weights = (routes.edge_area / resistance).astype(plan.layout.dtype)
        boundary_weights = (
            routes.boundary_area
            * flat_coefficient[routes.boundary_cells]
            / routes.boundary_distance
        ).astype(plan.layout.dtype)
        boundary_kind_host = np.asarray(
            [
                1 if plan.boundaries[int(axis)][int(side)].kind == "dirichlet" else 2
                for axis, side in zip(
                    np.asarray(routes.boundary_axis),
                    np.asarray(routes.boundary_side),
                    strict=True,
                )
            ],
            dtype=np.int8,
        )
        coefficient_fingerprint = canonical_fingerprint(
            {
                "kind": "composite-amr-positive-scalar-coefficient",
                "levels": [array_tree_fingerprint(value) for value in coefficient_levels],
            }
        )
        singular = not bool(np.any(boundary_kind_host == 1))
        component_nullity = plan.layout.component_count if singular else 0
        rank = plan.layout.space.size - component_nullity
        self.source = plan.layout.space
        self.target = plan.layout.space
        self.properties = OperatorProperties(
            self_adjoint=True,
            positive_definite=not singular,
            positive_semidefinite=singular,
            rank=rank,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
                **({"positive_definite": "construction"} if not singular else {}),
                "rank": "construction",
            },
        )
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
            diagonal_assembly=True,
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "prepared-composite-amr-diffusion",
                "plan": plan.plan_id,
                "coefficient": coefficient_fingerprint,
                "topology": plan.topology_fingerprint,
                "routes": plan.route_fingerprint,
                "precision": plan.precision_fingerprint,
            }
        )
        self.plan = plan
        self.coefficient = coefficient_levels
        self.edge_weights = edge_weights
        self.boundary_weights = boundary_weights
        self.boundary_kind = jnp.asarray(boundary_kind_host)
        self.coefficient_fingerprint = coefficient_fingerprint
        self.numeric_fingerprint = canonical_fingerprint(
            {
                "operator": self.operator_id,
                "edge_weights": array_tree_fingerprint(edge_weights),
                "boundary_weights": array_tree_fingerprint(boundary_weights),
            }
        )

    @property
    def layout(self) -> CompositeAMRCellLayout:
        return self.plan.layout

    @property
    def has_constant_nullspace(self) -> bool:
        return not self.properties.positive_definite

    def _stiffness_action(self, values: Array, /) -> Array:
        routes = self.plan.routes
        difference = values[routes.edge_left] - values[routes.edge_right]
        flux = self.edge_weights[:, None] * difference
        result = jnp.zeros_like(values)
        result = result.at[routes.edge_left].add(flux)
        result = result.at[routes.edge_right].add(-flux)
        boundary_flux = self.boundary_weights[:, None] * values[routes.boundary_cells]
        boundary_flux = jnp.where((self.boundary_kind == 1)[:, None], boundary_flux, 0.0)
        return result.at[routes.boundary_cells].add(boundary_flux)

    def mv(self, vector: PyTree[Any], /) -> tuple[Array, ...]:
        values = self.layout.flatten_cells(vector)
        stiffness = self._stiffness_action(values)
        result = jnp.where(
            self.layout.flat_leaf_mask[:, None],
            stiffness / self.layout.cell_measures[:, None],
            values,
        )
        return self.layout.unflatten_cells(result.astype(self.layout.dtype))

    def transpose_mv(self, vector: PyTree[Any], /) -> tuple[Array, ...]:
        values = self.layout.flatten_cells(vector)
        scaled = jnp.where(
            self.layout.flat_leaf_mask[:, None],
            values / self.layout.cell_measures[:, None],
            0.0,
        )
        stiffness = self._stiffness_action(scaled)
        result = jnp.where(self.layout.flat_leaf_mask[:, None], stiffness, values)
        return self.layout.unflatten_cells(result.astype(self.layout.dtype))

    def adjoint_mv(self, vector: PyTree[Any], /) -> tuple[Array, ...]:
        return self.mv(vector)

    def _materialize(self, /) -> Array:
        identity = jnp.eye(self.source.size, dtype=self.layout.dtype)

        def column(coordinates: Array) -> Array:
            return self.target.flatten(self.mv(self.source.unflatten(coordinates)))

        return jax.vmap(column, in_axes=1, out_axes=1)(identity)

    def _assemble_diagonal(self, /) -> Array:
        routes = self.plan.routes
        stiffness_diagonal = jnp.zeros((self.layout.cell_count,), dtype=self.layout.dtype)
        stiffness_diagonal = stiffness_diagonal.at[routes.edge_left].add(
            self.edge_weights
        )
        stiffness_diagonal = stiffness_diagonal.at[routes.edge_right].add(
            self.edge_weights
        )
        stiffness_diagonal = stiffness_diagonal.at[routes.boundary_cells].add(
            jnp.where(self.boundary_kind == 1, self.boundary_weights, 0.0)
        )
        cell_diagonal = jnp.where(
            self.layout.flat_leaf_mask,
            stiffness_diagonal / self.layout.cell_measures,
            1.0,
        )
        return jnp.repeat(cell_diagonal, self.layout.component_count)

    def energy(self, vector: PyTree[Any], /) -> Array:
        checked = self.layout.validate(vector)
        return jnp.real(self.source.inner(checked, self.mv(checked)))

    def face_fluxes(self, vector: PyTree[Any], /) -> Array:
        values = self.layout.flatten_cells(vector)
        routes = self.plan.routes
        flux = self.edge_weights[:, None] * (
            values[routes.edge_left] - values[routes.edge_right]
        )
        return flux.reshape((flux.shape[0],) + self.layout.component_shape)

    def interface_flux_contributions(self, vector: PyTree[Any], /) -> tuple[Array, Array]:
        flux = self.face_fluxes(vector)
        mask = self.plan.routes.edge_level_jump.reshape(
            (self.plan.routes.edge_level_jump.size,)
            + (1,) * len(self.layout.component_shape)
        )
        selected = jnp.where(mask, flux, 0.0)
        return selected, -selected

    def _cell_data(self, value: Any, name: str, /) -> Array:
        if not isinstance(value, (tuple, list)):
            scalar = jnp.asarray(value)
            if scalar.shape == ():
                return jnp.full(
                    (self.layout.cell_count, self.layout.component_count),
                    scalar,
                    dtype=self.layout.dtype,
                )
            if len(self.layout.level_shapes) == 1:
                value = (scalar,)
        if not isinstance(value, (tuple, list)) or len(value) != len(
            self.layout.level_shapes
        ):
            raise ValueError(
                f"Composite AMR {name} must be scalar or contain one array per level."
            )
        arrays = tuple(jnp.asarray(item, dtype=self.layout.dtype) for item in value)
        if any(
            array.shape != shape
            for array, shape in zip(arrays, self.layout.level_shapes, strict=True)
        ):
            raise ValueError(f"Composite AMR {name} level arrays have incorrect shapes.")
        return jnp.concatenate(
            tuple(array.reshape((-1, self.layout.component_count)) for array in arrays),
            axis=0,
        )

    def boundary_rhs_lift(
        self,
        data: CompositeBoundaryData | None = None,
        /,
    ) -> tuple[Array, ...]:
        """Build the affine boundary lift without changing the operator.

        Data are keyed by axis name and lower/upper side.  Each side value is
        either scalar or a tuple of block arrays in the composite layout.
        Neumann data are outward diffusive flux densities.
        """
        supplied = {} if data is None else dict(data)
        unknown = set(supplied).difference(self.layout.topology.plan.grid.axis_names)
        if unknown:
            raise ValueError(
                f"Composite AMR boundary data reference unknown axes {sorted(unknown)!r}."
            )
        routes = self.plan.routes
        integrated = jnp.zeros(
            (self.layout.cell_count, self.layout.component_count),
            dtype=self.layout.dtype,
        )
        for axis, name in enumerate(self.layout.topology.plan.grid.axis_names):
            if name in supplied and self.layout.topology.plan.periodic_axes[axis]:
                raise ValueError(
                    "Periodic composite AMR axes cannot carry physical boundary data."
                )
            values = supplied.get(name, (0.0, 0.0))
            if len(values) != 2:
                raise ValueError(
                    "Composite AMR boundary data need lower and upper values."
                )
            for side in range(2):
                cell_data = self._cell_data(values[side], "boundary data")
                route_mask = (routes.boundary_axis == axis) & (
                    routes.boundary_side == side
                )
                datum = cell_data[routes.boundary_cells]
                condition = self.plan.boundaries[axis][side]
                if condition.kind == "dirichlet":
                    contribution = self.boundary_weights[:, None] * datum
                elif condition.kind == "neumann":
                    contribution = -routes.boundary_area[:, None] * datum
                else:
                    contribution = jnp.zeros_like(datum)
                integrated = integrated.at[routes.boundary_cells].add(
                    jnp.where(route_mask[:, None], contribution, 0.0)
                )
        lift = jnp.where(
            self.layout.flat_leaf_mask[:, None],
            integrated / self.layout.cell_measures[:, None],
            0.0,
        )
        return self.layout.unflatten_cells(lift.astype(self.layout.dtype))

    def prepare_rhs(
        self,
        source: Any = 0.0,
        /,
        *,
        boundary_data: CompositeBoundaryData | None = None,
        project: bool = False,
    ) -> tuple[Array, ...]:
        """Combine a volume source and boundary lift on physical leaf rows."""
        source_values = self._cell_data(source, "source")
        source_values = jnp.where(self.layout.flat_leaf_mask[:, None], source_values, 0.0)
        lift = self.layout.flatten_cells(self.boundary_rhs_lift(boundary_data))
        result = self.layout.unflatten_cells(
            (source_values + lift).astype(self.layout.dtype)
        )
        if project:
            if not self.has_constant_nullspace:
                raise ValueError(
                    "Compatibility projection is only defined for singular diffusion."
                )
            return self.project_compatible_rhs(result)
        return result

    def compatibility_defect(self, rhs: PyTree[Any], /) -> Array:
        checked = self.layout.require_zero_masked(rhs)
        return self.layout.integral(checked)

    def require_compatible_rhs(
        self,
        rhs: PyTree[Any],
        /,
        *,
        tolerance: ArrayLike | None = None,
    ) -> tuple[Array, ...]:
        checked = self.layout.require_zero_masked(rhs)
        if not self.has_constant_nullspace:
            return checked
        defect = self.layout.integral(checked)
        scale = jnp.maximum(
            jnp.sum(
                jnp.abs(
                    jnp.where(
                        self.layout.flat_leaf_mask[:, None],
                        self.layout.flatten_cells(checked)
                        * self.layout.cell_measures[:, None],
                        0.0,
                    )
                ),
                axis=0,
            ),
            1.0,
        )
        threshold = (
            100.0 * jnp.finfo(self.layout.dtype).eps * scale
            if tolerance is None
            else jnp.asarray(tolerance, dtype=self.layout.dtype)
        )
        output = list(checked)
        output[0] = _checked(
            output[0],
            jnp.any(jnp.abs(defect.reshape((-1,))) > threshold),
            "Composite AMR Neumann/periodic right-hand side is incompatible "
            "with the constant nullspace.",
        )
        return tuple(output)

    def project_compatible_rhs(self, rhs: PyTree[Any], /) -> tuple[Array, ...]:
        checked = self.layout.require_zero_masked(rhs)
        if not self.has_constant_nullspace:
            return checked
        return self.layout.zero_mean(checked)

    def zero_mean_gauge(self, vector: PyTree[Any], /) -> tuple[Array, ...]:
        return self.layout.zero_mean(vector)

    def constant_nullspace(self, /) -> LinearSubspace | None:
        if not self.has_constant_nullspace:
            return None
        return LinearSubspace(
            self.source,
            self.layout.constant_mode_coordinates(),
            subspace_id=canonical_fingerprint(
                {"kind": "composite-amr-constant-nullspace", "operator": self.operator_id}
            ),
        )

    def nullspace_policy(
        self,
        /,
        *,
        compatibility: Literal["error", "project"] = "error",
        gauge: Literal["minimum-norm", "project"] = "project",
    ) -> NullspacePolicy | None:
        nullspace = self.constant_nullspace()
        if nullspace is None:
            return None
        certificate = KernelCertificate(
            self,
            nullspace,
            left=nullspace,
            evidence="verified",
            scope="numerical",
            complete=True,
        )
        return NullspacePolicy(
            right=nullspace,
            left=nullspace,
            certificate=certificate,
            compatibility=compatibility,
            gauge=gauge,
        )

    def linear_system(
        self,
        /,
        *,
        compatibility: Literal["error", "project"] = "error",
        gauge: Literal["minimum-norm", "project"] = "project",
    ) -> LinearSystem:
        return LinearSystem(
            self,
            nullspace_policy=self.nullspace_policy(
                compatibility=compatibility, gauge=gauge
            ),
        )


def _coefficient(
    plan: CompositeAMRDiffusionPlan,
    value: ArrayLike | Sequence[ArrayLike],
    /,
) -> tuple[tuple[Array, ...], Array]:
    layout = plan.layout
    coefficient_dtype = np.dtype(plan.precision.flux_dtype)
    if not isinstance(value, (tuple, list)):
        scalar = np.asarray(value)
        if scalar.shape == ():
            raw = tuple(
                np.full(mask.shape, scalar, dtype=coefficient_dtype)
                for mask in layout.leaf_mask
            )
        elif len(layout.leaf_mask) == 1:
            raw = (scalar.astype(coefficient_dtype),)
        else:
            raise ValueError(
                "Composite AMR coefficient must be positive scalar or contain "
                "one scalar array per level."
            )
    else:
        if len(value) != len(layout.leaf_mask):
            raise ValueError(
                "Composite AMR coefficient must be positive scalar or contain "
                "one scalar array per level."
            )
        raw = tuple(np.asarray(item, dtype=coefficient_dtype) for item in value)
    if any(
        array.shape != mask.shape
        for array, mask in zip(raw, layout.leaf_mask, strict=True)
    ):
        raise ValueError("Composite AMR coefficient level arrays have incorrect shapes.")
    normalized = []
    for array, mask in zip(raw, layout.leaf_mask, strict=True):
        mask_host = np.asarray(mask, dtype=bool)
        if np.any(~np.isfinite(array[mask_host])) or np.any(array[mask_host] <= 0.0):
            raise ValueError(
                "Composite AMR leaf coefficients must be finite and strictly positive."
            )
        normalized.append(plan.precision.flux(np.where(mask_host, array, 1.0)))
    levels = tuple(normalized)
    flat = jnp.concatenate(tuple(array.reshape((-1,)) for array in levels))
    return levels, flat


def composite_amr_multigrid_builder(
    operators: Sequence[AbstractLinearOperator],
    smoothers: Sequence[AbstractPreconditioner | AbstractPreconditionerBuilder],
    restrictions: Sequence[AbstractLinearOperator],
    prolongations: Sequence[AbstractLinearOperator],
    /,
    *,
    coarse_operator_source: CompositeCoarseOperatorSource = "galerkin",
    properties: PreconditionerProperties | None = None,
    cycle: Literal["v", "w", "f", "full"] = "v",
    pre_smoothing: int = 1,
    post_smoothing: int = 1,
) -> GalerkinHierarchyBuilder | MultigridHierarchyBuilder:
    """Return a native linalg hierarchy builder without adding AMR state to linalg."""

    operators_ = tuple(operators)
    smoothers_ = tuple(smoothers)
    restrictions_ = tuple(restrictions)
    prolongations_ = tuple(prolongations)
    source = str(coarse_operator_source)
    if source not in ("direct", "galerkin"):
        raise ValueError("Unknown composite AMR coarse-operator source.")
    if not operators_ or not isinstance(operators_[0], PreparedCompositeAMRDiffusion):
        raise TypeError(
            "Composite AMR multigrid requires a prepared composite fine operator."
        )
    transition_count = len(restrictions_)
    if (
        transition_count == 0
        or len(prolongations_) != transition_count
        or len(smoothers_) != transition_count + 1
        or len(operators_) != (1 if source == "galerkin" else transition_count + 1)
    ):
        raise ValueError(
            "Composite AMR multigrid operators, smoothers, and transfers are incomplete."
        )
    cycle_policy = MultigridCyclePolicy(cycle)
    if source == "galerkin":
        return GalerkinHierarchyBuilder(
            tuple(zip(restrictions_, prolongations_, strict=True)),
            smoothers_[:-1],
            smoothers_[-1],
            properties=properties,
            cycle_policy=cycle_policy,
            pre_smoothing=pre_smoothing,
            post_smoothing=post_smoothing,
        )
    levels = []
    for index, (operator, smoother) in enumerate(
        zip(operators_, smoothers_, strict=True)
    ):
        if index == transition_count:
            levels.append(MultigridLevelBuilder(operator, smoother))
        else:
            levels.append(
                MultigridLevelBuilder(
                    operator,
                    smoother,
                    restriction=restrictions_[index],
                    prolongation=prolongations_[index],
                    pre_smoothing=pre_smoothing,
                    post_smoothing=post_smoothing,
                )
            )
    return MultigridHierarchyBuilder(
        tuple(levels), properties=properties, cycle_policy=cycle_policy
    )


__all__ = [
    "CompositeAMRDiffusionPlan",
    "CompositeBoundaryData",
    "CompositeBoundaryInput",
    "CompositeCoarseOperatorSource",
    "PreparedCompositeAMRDiffusion",
    "composite_amr_multigrid_builder",
]
