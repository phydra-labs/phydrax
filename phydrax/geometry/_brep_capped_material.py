#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Conforming source material charts for a circular-capped cylindrical shell."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._brep_capped_cap import CapPoint, CirclePoint, prepare_circular_cap_cells
from ._brep_capped_side import prepare_cylindrical_side_charts
from ._brep_circle_rim_proof import prove_circle_rim_edges
from ._source_array_identity import source_arrays_equal
from ._surface_affine_cell_proof import PeriodicTriangle
from ._surface_source_support import SurfaceSourceCharts
from ._volume_material_map import require_polyhedral_carrier_geometry
from .brep._boolean_core import ChartScalar
from .brep._patches import AbstractSurfacePatch, surface_differential
from .brep._trim_helpers import period_scalar_bounds

if TYPE_CHECKING:
    from ..meshing._result import CellMeshingResult
    from ._source_boundary_records import SourceBoundaryQuery


def _real(value: ChartScalar, /) -> float:
    lower, upper = period_scalar_bounds(value)
    return (float(lower) + float(upper)) / 2


def _point(point: CapPoint, /) -> tuple[float, float, float]:
    if not isinstance(point, CirclePoint):
        return float(point[0]), float(point[1]), float(point[2])
    angle = _real(point.phase)
    curve = point.curve
    values = tuple(float(curve.center_exact[i]) + float(curve.radius_exact) * (
        np.cos(angle) * float(curve.first_axis_exact[i])
        + np.sin(angle) * float(curve.second_axis_exact[i])
    ) for i in range(3))
    return float(values[0]), float(values[1]), float(values[2])


class _FanRows(StrictModule, NonTrainableState):
    indices: Array
    selectors: Array
    apex: Array
    center: Array
    first_axis: Array
    second_axis: Array
    radius: Array
    first: Array
    span: Array


class _SideRows(StrictModule, NonTrainableState):
    patch: int = eqx.field(static=True)
    indices: Array
    corners: Array


class _Block(StrictModule, NonTrainableState):
    corners: Array
    fans: _FanRows
    sides: tuple[_SideRows, ...]


class BRepCappedMaterialMap(StrictModule, NonTrainableState):
    """Current-source cylindrical sides and circular-cap material triangles.

    All cells are proved contained in their actual source trims. Each curved
    fan has a strict interior Jacobian and an injective closed-cell map; its
    apex has a point value but generally no unique derivative. Use
    ``evaluate_points`` for apex degrees of freedom and ``evaluate`` only at
    non-apex quadrature points.
    """

    charts: SurfaceSourceCharts
    patches: tuple[AbstractSurfacePatch, ...]
    blocks: tuple[_Block, ...]
    guard_baselines: tuple[Array, ...]
    guard_groups: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    guard_group_id: str = eqx.field(static=True)
    source_static_values: tuple[Any, ...] = eqx.field(static=True)
    source_tree: Any = eqx.field(static=True)
    carrier_coordinates: Array
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_chart_id: str = eqx.field(static=True)
    binding_identity: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, result: CellMeshingResult, source_query: SourceBoundaryQuery, /) -> None:
        charts = result.surface_source
        if charts is None:
            raise ValueError("Capped material maps require retained source charts.")
        charts.require_bound(result.mesh, result.geometry, source_query)
        charts.require_root(result.mesh, result.geometry)
        certification = result.certification
        if certification is None or certification.inputs.fidelity_tolerance is None:
            raise ValueError("Capped material maps require their original source fidelity tolerance.")
        tolerance = certification.inputs.fidelity_tolerance
        require_polyhedral_carrier_geometry(result.mesh, result.geometry)
        rims = prove_circle_rim_edges(charts, tolerance)
        if not rims:
            raise ValueError("The source has no independently proved circular cap rims.")
        side = prepare_cylindrical_side_charts(charts, tolerance, rims)
        cap = prepare_circular_cap_cells(charts, rims, set(side))
        if set(side) | set(cap.corners) != set(charts.cell_global_ids) or set(side) & set(cap.corners):
            raise ValueError("The material maps do not cover the exact source cell partition.")
        patches = tuple(patch.surface for patch in charts.domain.patches)
        row_of = {cell: row for row, cell in enumerate(charts.cell_global_ids)}
        blocks = []
        for block in result.mesh.blocks:
            ids = [int(value) for value in np.asarray(block.global_ids)]
            corners = np.zeros((len(ids), 3, 3), dtype=np.float64)
            fan_ids: list[int] = []
            selectors: list[np.ndarray] = []
            apex, center, first_axis, second_axis, radius, first, span = ([] for _ in range(7))
            sides: dict[int, list[tuple[int, PeriodicTriangle]]] = {}
            for index, cell in enumerate(ids):
                if cell in side:
                    patch = int(charts.root_patches[row_of[cell]])
                    sides.setdefault(patch, []).append((index, side[cell]))
                    continue
                corners[index] = np.asarray([_point(point) for point in cap.corners[cell]])
                if cell not in cap.fans:
                    continue
                fan, (_, first_slot, last_slot) = cap.fans[cell]
                fan_ids.append(index)
                select = np.zeros((3, 2), dtype=np.float64)
                select[first_slot, 0] = select[last_slot, 1] = 1
                selectors.append(select)
                apex.append(np.asarray(fan.apex))
                center.append(np.asarray(fan.curve.center))
                first_axis.append(np.asarray(fan.curve.first_axis))
                second_axis.append(np.asarray(fan.curve.second_axis))
                radius.append(float(fan.curve.radius))
                first.append(float(fan.first))
                span.append(float(fan.span))
            n = len(fan_ids)
            fan_rows = _FanRows(
                indices=jnp.asarray(fan_ids, dtype=jnp.int32),
                selectors=jnp.asarray(np.asarray(selectors, dtype=np.float64).reshape((n, 3, 2))),
                apex=jnp.asarray(np.asarray(apex, dtype=np.float64).reshape((n, 3))),
                center=jnp.asarray(np.asarray(center, dtype=np.float64).reshape((n, 3))),
                first_axis=jnp.asarray(np.asarray(first_axis, dtype=np.float64).reshape((n, 3))),
                second_axis=jnp.asarray(np.asarray(second_axis, dtype=np.float64).reshape((n, 3))),
                radius=jnp.asarray(radius), first=jnp.asarray(first), span=jnp.asarray(span),
            )
            side_rows = tuple(_SideRows(
                patch=patch,
                indices=jnp.asarray([index for index, _ in values], dtype=jnp.int32),
                corners=jnp.asarray([[[ _real(value) for value in point] for point in triangle]
                                     for _, triangle in values]),
            ) for patch, values in sorted(sides.items()))
            blocks.append(_Block(corners=jnp.asarray(corners), fans=fan_rows, sides=side_rows))
        self.charts = charts
        self.patches = patches
        self.blocks = tuple(blocks)
        self.carrier_coordinates = jnp.asarray(result.geometry.coordinates)
        leaves, self.source_tree = jax.tree.flatten(
            (self.charts, self.patches, self.blocks, self.carrier_coordinates)
        )
        arrays = tuple(jnp.asarray(leaf) for leaf in leaves if eqx.is_array(leaf))
        grouped: dict[tuple[np.dtype, tuple[int, ...]], list[int]] = {}
        for index, array in enumerate(arrays):
            grouped.setdefault((np.dtype(array.dtype), tuple(array.shape)), []).append(index)
        self.guard_groups = tuple(tuple(indices) for indices in grouped.values())
        self.guard_group_id = canonical_fingerprint(self.guard_groups)
        self.guard_baselines = tuple(
            jnp.stack(tuple(arrays[index] for index in indices))
            for indices in self.guard_groups
        )
        self.source_static_values = tuple(leaf for leaf in leaves if not eqx.is_array(leaf))
        self.topology_id = result.mesh.topology_id
        self.geometry_layout_id = result.geometry.geometry_layout_id
        self.source_id = charts.source_id
        self.source_revision = charts.source_revision
        self.source_chart_id = charts.source_chart_id
        self.source_geometry_id = canonical_fingerprint({
            "kind": "brep-capped-material-map", "source": charts.source_chart_id,
            "topology": self.topology_id, "geometry": self.geometry_layout_id,
            "blocks": array_tree_fingerprint(self.blocks),
        })
        self.binding_identity = (
            self.topology_id, self.geometry_layout_id, self.source_geometry_id,
            self.source_id, self.source_revision, self.source_chart_id, self.guard_group_id,
        )

    def evaluate_points(self, block_index: int, reference_points: ArrayLike, /) -> Array:
        return self._evaluate(block_index, reference_points, derivative=False)[0]

    def evaluate(self, block_index: int, reference_points: ArrayLike, /) -> tuple[Array, Array]:
        physical, jacobian = self._evaluate(block_index, reference_points, derivative=True)
        if jacobian is None:
            raise RuntimeError("Capped material evaluation did not produce its Jacobian.")
        return physical, jacobian

    def prepare_rows(
        self, block_index: int, row_indices: np.ndarray, /
    ) -> CappedMaterialRows:
        """Restrict proved maps to fixed, ordered block-local cell rows."""
        return CappedMaterialRows(self, block_index, row_indices)

    def source_changed(self, /) -> Array:
        """Check every source occurrence and its exact, grouped array baseline."""
        if self.binding_identity != (
            self.topology_id, self.geometry_layout_id, self.source_geometry_id,
            self.source_id, self.source_revision, self.source_chart_id, self.guard_group_id,
        ) or self.guard_group_id != canonical_fingerprint(self.guard_groups):
            raise ValueError("Capped material source binding changed after proof.")
        leaves, tree = jax.tree.flatten(
            (self.charts, self.patches, self.blocks, self.carrier_coordinates)
        )
        if tree != self.source_tree:
            raise ValueError("Capped material source structure changed after proof.")
        arrays = tuple(jnp.asarray(leaf) for leaf in leaves if eqx.is_array(leaf))
        static = tuple(leaf for leaf in leaves if not eqx.is_array(leaf))
        if (static != self.source_static_values
                or len(self.guard_groups) != len(self.guard_baselines)
                or sum(len(group) for group in self.guard_groups) != len(arrays)):
            raise ValueError("Capped material source metadata changed after proof.")
        changed = jnp.asarray(False)
        for indices, original in zip(self.guard_groups, self.guard_baselines, strict=True):
            current = tuple(arrays[index] for index in indices)
            if (not current or original.shape[0] != len(current)
                    or any(value.shape != original.shape[1:] or value.dtype != original.dtype
                           for value in current)):
                raise ValueError("Capped material source array layout changed after proof.")
            changed = changed | ~source_arrays_equal(jnp.stack(current), original)
        return changed

    def _evaluate(
        self, block_index: int, reference_points: ArrayLike, /, *,
        derivative: bool, restricted: _Block | None = None,
        restricted_changed: Array | None = None,
    ) -> tuple[Array, Array | None]:
        changed = self.source_changed()
        if restricted_changed is not None:
            changed = changed | restricted_changed
        points = jnp.asarray(reference_points)
        basis = jnp.concatenate((1 - jnp.sum(points, axis=-1, keepdims=True), points), axis=-1)
        block = self.blocks[block_index] if restricted is None else restricted
        physical = ein.contract("qk,ckd->cqd", basis, block.corners)
        jacobian = None
        if derivative:
            gradient = jnp.swapaxes(block.corners[:, 1:] - block.corners[:, :1], 1, 2)
            jacobian = jnp.broadcast_to(gradient[:, None, :, :], (gradient.shape[0], points.shape[0], 3, 2))
        for side in block.sides:
            uv = ein.contract("qk,ckd->cqd", basis, side.corners)
            patch = self.patches[side.patch]
            physical = physical.at[side.indices].set(patch.evaluate(uv))
            if derivative:
                if jacobian is None:
                    raise RuntimeError("Capped material evaluation lost its Jacobian.")
                chart_jacobian = jnp.swapaxes(side.corners[:, 1:] - side.corners[:, :1], 1, 2)
                jacobian = jacobian.at[side.indices].set(ein.contract(
                    "cqdi,cij->cqdj", surface_differential(patch, uv), chart_jacobian
                ))
        fan = block.fans
        local = ein.contract("qk,ckd->cqd", basis, fan.selectors)
        x, y = local[..., 0], local[..., 1]
        q = x + y
        t = y / jnp.where(q == 0, 1, q)
        angle = fan.first[:, None] + fan.span[:, None] * t
        radial = (fan.center[:, None, :] + fan.radius[:, None, None] * (
            jnp.cos(angle)[..., None] * fan.first_axis[:, None, :]
            + jnp.sin(angle)[..., None] * fan.second_axis[:, None, :]
        ) - fan.apex[:, None, :])
        fan_points = fan.apex[:, None, :] + q[..., None] * radial
        physical = physical.at[fan.indices].set(fan_points)
        if derivative:
            if jacobian is None:
                raise RuntimeError("Capped material evaluation lost its Jacobian.")
            tangent = fan.radius[:, None, None] * fan.span[:, None, None] * (
                -jnp.sin(angle)[..., None] * fan.first_axis[:, None, :]
                + jnp.cos(angle)[..., None] * fan.second_axis[:, None, :]
            )
            fan_jac = jnp.stack((radial - t[..., None] * tangent,
                                 radial + (1 - t)[..., None] * tangent), axis=-1)
            transform = jnp.swapaxes(fan.selectors[:, 1:] - fan.selectors[:, :1], 1, 2)
            fan_jac = ein.contract("cqdi,cij->cqdj", fan_jac, transform)
            fan_jac = eqx.error_if(fan_jac, jnp.any(q == 0), "A circle fan has no unique Jacobian at its apex; use evaluate_points.")
            jacobian = jacobian.at[fan.indices].set(fan_jac)
            jacobian = eqx.error_if(jacobian, changed, "Capped material source changed after proof.")
        physical = eqx.error_if(physical, changed, "Capped material source changed after proof.")
        return physical, jacobian


class CappedMaterialRows(StrictModule, NonTrainableState):
    """A fixed row restriction of a proved capped material map."""

    parent: BRepCappedMaterialMap
    block_index: int = eqx.field(static=True)
    block: _Block
    reference_arrays: tuple[Array, ...]
    static_values: tuple[Any, ...] = eqx.field(static=True)
    block_tree: Any = eqx.field(static=True)

    def __init__(
        self, parent: BRepCappedMaterialMap, block_index: int,
        row_indices: np.ndarray, /,
    ) -> None:
        if not isinstance(parent, BRepCappedMaterialMap):
            raise TypeError("A capped workset requires its proved material map.")
        if type(block_index) is not int or not 0 <= block_index < len(parent.blocks):
            raise IndexError("Capped material block index is out of range.")
        rows = np.asarray(row_indices)
        source = parent.blocks[block_index]
        if (rows.ndim != 1 or rows.dtype.kind not in "iu"
                or np.any(rows < 0) or np.any(rows >= source.corners.shape[0])
                or np.any(rows[1:] <= rows[:-1])):
            raise ValueError("Capped material rows must be unique, sorted block-local indices.")
        rows = rows.astype(np.int64, copy=False)
        inverse = np.full(source.corners.shape[0], -1, dtype=np.int64)
        inverse[rows] = np.arange(rows.size)
        fan = source.fans
        fan_indices = np.asarray(fan.indices, dtype=np.int64)
        fan_positions = np.flatnonzero(inverse[fan_indices] >= 0)
        restricted_fan = _FanRows(
            indices=jnp.asarray(inverse[fan_indices[fan_positions]], dtype=jnp.int32),
            selectors=fan.selectors[fan_positions], apex=fan.apex[fan_positions],
            center=fan.center[fan_positions], first_axis=fan.first_axis[fan_positions],
            second_axis=fan.second_axis[fan_positions], radius=fan.radius[fan_positions],
            first=fan.first[fan_positions], span=fan.span[fan_positions],
        )
        sides = []
        for side in source.sides:
            indices = np.asarray(side.indices, dtype=np.int64)
            positions = np.flatnonzero(inverse[indices] >= 0)
            if positions.size:
                sides.append(_SideRows(
                    patch=side.patch,
                    indices=jnp.asarray(inverse[indices[positions]], dtype=jnp.int32),
                    corners=side.corners[positions],
                ))
        self.parent = parent
        self.block_index = block_index
        self.block = _Block(
            corners=source.corners[rows], fans=restricted_fan, sides=tuple(sides),
        )
        leaves, self.block_tree = jax.tree.flatten(self.block)
        self.reference_arrays = tuple(jnp.asarray(leaf) for leaf in leaves if eqx.is_array(leaf))
        self.static_values = tuple(leaf for leaf in leaves if not eqx.is_array(leaf))

    def _changed(self) -> Array:
        leaves, tree = jax.tree.flatten(self.block)
        if tree != self.block_tree:
            raise ValueError("Capped material row restriction changed after proof.")
        arrays = tuple(leaf for leaf in leaves if eqx.is_array(leaf))
        static = tuple(leaf for leaf in leaves if not eqx.is_array(leaf))
        if static != self.static_values or len(arrays) != len(self.reference_arrays):
            raise ValueError("Capped material row restriction metadata changed after proof.")
        changed = jnp.asarray(False)
        for current, original in zip(arrays, self.reference_arrays, strict=True):
            current = jnp.asarray(current)
            if current.shape != original.shape or current.dtype != original.dtype:
                raise ValueError("Capped material row restriction layout changed after proof.")
            changed = changed | ~source_arrays_equal(current, original)
        return changed

    def evaluate_points(self, reference_points: ArrayLike, /) -> Array:
        return self.parent._evaluate(
            self.block_index, reference_points, derivative=False,
            restricted=self.block, restricted_changed=self._changed(),
        )[0]

    def evaluate(self, reference_points: ArrayLike, /) -> tuple[Array, Array]:
        points, jacobian = self.parent._evaluate(
            self.block_index, reference_points, derivative=True,
            restricted=self.block, restricted_changed=self._changed(),
        )
        if jacobian is None:
            raise RuntimeError("Capped material row evaluation did not produce its Jacobian.")
        return points, jacobian
