#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Rank-aware reciprocal meshes, paths, and oriented connectivity."""

from __future__ import annotations

from itertools import combinations, product
from math import isfinite, prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._periodic_cell import PeriodicCell


class ReciprocalResourceError(RuntimeError):
    """A reciprocal preparation exceeds an explicit structural capacity."""


def _require_cell(cell: PeriodicCell) -> PeriodicCell:
    if not isinstance(cell, PeriodicCell):
        raise TypeError("A reciprocal plan requires PeriodicCell.")
    if cell.rank not in (1, 2, 3):
        raise ValueError("Reciprocal plans support lattice rank 1, 2, or 3.")
    return cell


class ReciprocalMeshPlan(StrictModule, NonTrainableState):
    """Unique weighted fractional points bound to one periodic cell."""

    cell: PeriodicCell
    fractional_points: Array
    weights: Array
    mesh_indices: Array
    mesh_shape: tuple[int, ...] = eqx.field(static=True)
    shift: tuple[float, ...] = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    cell_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell: PeriodicCell,
        fractional_points: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        mesh_shape: tuple[int, ...],
        shift: tuple[float, ...],
        mesh_indices: ArrayLike | None = None,
        uniqueness_tolerance: float = 1.0e-12,
    ):
        cell_ = _require_cell(cell)
        points = np.asarray(fractional_points)
        weights_ = np.asarray(weights)
        shape = tuple(mesh_shape)
        shift_ = tuple(float(value) for value in shift)
        tolerance = float(uniqueness_tolerance)
        if (
            points.ndim != 2
            or points.shape[1] != cell_.rank
            or points.shape[0] == 0
            or weights_.shape != (points.shape[0],)
        ):
            raise ValueError(
                "Reciprocal points and weights must have shapes (K, rank) and (K,)."
            )
        if (
            len(shape) != cell_.rank
            or any(value <= 0 for value in shape)
            or len(shift_) != cell_.rank
            or any(not isfinite(value) for value in shift_)
            or not isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError(
                "Reciprocal mesh shape, shift, or uniqueness tolerance is invalid."
            )
        if points.shape[0] != prod(shape):
            raise ValueError(
                "A regular reciprocal mesh requires exactly product(mesh_shape) points."
            )
        if (
            np.any(~np.isfinite(points))
            or np.any(~np.isfinite(weights_))
            or np.any(weights_ <= 0.0)
            or not np.isclose(float(np.sum(weights_)), 1.0, atol=1.0e-12)
        ):
            raise ValueError(
                "Reciprocal points must be finite and weights positive with unit sum."
            )
        inactive = np.logical_not(np.asarray(cell_.periodic_axes))
        if np.any(inactive):
            for axis in np.flatnonzero(inactive):
                if shape[int(axis)] != 1 or shift_[int(axis)] != 0.0:
                    raise ValueError(
                        "Nonperiodic cell axes require one unshifted reciprocal point."
                    )
                if np.any(np.abs(points[:, int(axis)]) > tolerance):
                    raise ValueError(
                        "Nonperiodic cell axes require zero fractional coordinates."
                    )
        wrapped = points - np.floor(points + 0.5)
        quantized = np.rint(wrapped / tolerance).astype(np.int64)
        if np.unique(quantized, axis=0).shape[0] != points.shape[0]:
            raise ValueError(
                "Reciprocal mesh contains points duplicated modulo the lattice."
            )
        if mesh_indices is None:
            indices = np.asarray(
                tuple(product(*(range(value) for value in shape))), dtype=np.int32
            )
        else:
            indices = np.asarray(mesh_indices)
        if (
            indices.shape != (points.shape[0], cell_.rank)
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= np.asarray(shape)[None, :])
            or np.unique(indices, axis=0).shape[0] != indices.shape[0]
        ):
            raise ValueError(
                "mesh_indices must uniquely cover the declared regular grid."
            )
        self.cell = cell_
        self.fractional_points = jnp.asarray(points)
        self.weights = jnp.asarray(weights_, dtype=points.dtype)
        self.mesh_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.mesh_shape = shape
        self.shift = shift_
        self.rank = cell_.rank
        self.cell_id = cell_.cell_id
        self.mesh_id = canonical_fingerprint(
            {
                "kind": "reciprocal-mesh-plan",
                "cell": cell_.cell_id,
                "mesh_shape": list(shape),
                "shift": list(shift_),
                "arrays": array_tree_fingerprint(
                    {"points": points, "weights": weights_, "indices": indices}
                ),
            }
        )

    @classmethod
    def monkhorst_pack(
        cls,
        cell: PeriodicCell,
        mesh_shape: tuple[int, ...],
        /,
        *,
        shift: tuple[float, ...] | None = None,
        dtype=np.float64,
        maximum_points: int = 1_000_000,
    ) -> "ReciprocalMeshPlan":
        cell_ = _require_cell(cell)
        shape = tuple(mesh_shape)
        shift_ = (0.0,) * cell_.rank if shift is None else tuple(float(v) for v in shift)
        if len(shape) != cell_.rank or any(value <= 0 for value in shape):
            raise ValueError(
                "Monkhorst--Pack dimensions must be positive and match cell rank."
            )
        if len(shift_) != cell_.rank:
            raise ValueError("Monkhorst--Pack shift must match cell rank.")
        count = prod(shape)
        if count > int(maximum_points):
            raise ReciprocalResourceError("Monkhorst--Pack mesh exceeds maximum_points.")
        axes = [
            (np.arange(size, dtype=dtype) + 0.5) / size - 0.5 + shift_[axis] / size
            for axis, size in enumerate(shape)
        ]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
            (-1, cell_.rank)
        )
        inactive = np.logical_not(np.asarray(cell_.periodic_axes))
        points[:, inactive] = 0.0
        indices = np.asarray(
            tuple(product(*(range(value) for value in shape))), dtype=np.int32
        )
        weights = np.full((count,), 1.0 / count, dtype=dtype)
        return cls(
            cell_,
            points,
            weights,
            mesh_shape=shape,
            shift=shift_,
            mesh_indices=indices,
        )

    def require_cell(self, cell: PeriodicCell, /) -> None:
        cell_ = _require_cell(cell)
        if cell_.cell_id != self.cell_id:
            raise ValueError("Reciprocal mesh belongs to a different PeriodicCell.")


class ReciprocalPathPlan(StrictModule, NonTrainableState):
    """Ordered reciprocal path with physical reciprocal-space distances."""

    cell: PeriodicCell
    fractional_points: Array
    distances: Array
    labels: tuple[str, ...] = eqx.field(static=True)
    cell_id: str = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    path_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell: PeriodicCell,
        fractional_points: ArrayLike,
        /,
        *,
        labels: tuple[str, ...] | None = None,
        maximum_points: int = 1_000_000,
    ):
        cell_ = _require_cell(cell)
        points = np.asarray(fractional_points)
        if points.ndim != 2 or points.shape[1] != cell_.rank or points.shape[0] < 2:
            raise ValueError(
                "Reciprocal paths require at least two points with shape (K, rank)."
            )
        if points.shape[0] > int(maximum_points):
            raise ReciprocalResourceError("Reciprocal path exceeds maximum_points.")
        if np.any(~np.isfinite(points)):
            raise ValueError("Reciprocal path points must be finite.")
        labels_ = () if labels is None else tuple(str(value).strip() for value in labels)
        if labels_ and (
            len(labels_) != points.shape[0] or any(not value for value in labels_)
        ):
            raise ValueError(
                "Path labels, when supplied, require one nonempty label per point."
            )
        cartesian = points @ np.asarray(cell_.reciprocal_vectors)
        increments = np.linalg.norm(np.diff(cartesian, axis=0), axis=1)
        distances = np.concatenate(
            (np.zeros((1,), dtype=points.dtype), np.cumsum(increments))
        )
        self.cell = cell_
        self.fractional_points = jnp.asarray(points)
        self.distances = jnp.asarray(distances)
        self.labels = labels_
        self.cell_id = cell_.cell_id
        self.rank = cell_.rank
        self.path_id = canonical_fingerprint(
            {
                "kind": "reciprocal-path-plan",
                "cell": cell_.cell_id,
                "labels": list(labels_),
                "arrays": array_tree_fingerprint(
                    {"points": points, "distances": distances}
                ),
            }
        )

    def require_cell(self, cell: PeriodicCell, /) -> None:
        if _require_cell(cell).cell_id != self.cell_id:
            raise ValueError("Reciprocal path belongs to a different PeriodicCell.")


class ReciprocalConnectivityPlan(StrictModule, NonTrainableState):
    """Directed neighbor links and oriented regular-mesh plaquettes."""

    mesh: ReciprocalMeshPlan
    source_indices: Array
    target_indices: Array
    reciprocal_shifts: Array
    reverse_indices: Array
    plaquette_edges: Array
    plaquette_orientations: Array
    mesh_id: str = eqx.field(static=True)
    cell_id: str = eqx.field(static=True)
    connectivity_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: ReciprocalMeshPlan,
        source_indices: ArrayLike,
        target_indices: ArrayLike,
        reciprocal_shifts: ArrayLike,
        reverse_indices: ArrayLike,
        /,
        *,
        plaquette_edges: ArrayLike | None = None,
        plaquette_orientations: ArrayLike | None = None,
        maximum_edges: int = 4_000_000,
        maximum_plaquettes: int = 2_000_000,
        closure_tolerance: float = 1.0e-12,
    ):
        if not isinstance(mesh, ReciprocalMeshPlan):
            raise TypeError("mesh must be ReciprocalMeshPlan.")
        source = np.asarray(source_indices, dtype=np.int64)
        target = np.asarray(target_indices, dtype=np.int64)
        shift = np.asarray(reciprocal_shifts)
        reverse = np.asarray(reverse_indices, dtype=np.int64)
        if (
            source.ndim != 1
            or target.shape != source.shape
            or reverse.shape != source.shape
            or shift.shape != (source.size, mesh.rank)
            or source.size == 0
            or not np.issubdtype(shift.dtype, np.integer)
        ):
            raise ValueError(
                "Connectivity links require source/target/reverse (E,) and shifts (E, rank)."
            )
        if source.size > int(maximum_edges):
            raise ReciprocalResourceError(
                "Reciprocal connectivity exceeds maximum_edges."
            )
        point_count = mesh.fractional_points.shape[0]
        if (
            np.any(source < 0)
            or np.any(source >= point_count)
            or np.any(target < 0)
            or np.any(target >= point_count)
            or np.any(reverse < 0)
            or np.any(reverse >= source.size)
        ):
            raise ValueError("Reciprocal connectivity indices exceed their ranges.")
        indices = np.arange(source.size)
        if (
            not np.array_equal(reverse[reverse], indices)
            or not np.array_equal(source[reverse], target)
            or not np.array_equal(target[reverse], source)
            or not np.array_equal(shift[reverse], -shift)
        ):
            raise ValueError(
                "Connectivity reverse links must be an exact oriented involution."
            )
        link_keys = np.concatenate((source[:, None], target[:, None], shift), axis=1)
        if np.unique(link_keys, axis=0).shape[0] != source.size:
            raise ValueError("Reciprocal connectivity contains duplicate directed links.")
        if plaquette_edges is None:
            plaquettes = np.empty((0, 4), dtype=np.int32)
        else:
            plaquettes = np.asarray(plaquette_edges, dtype=np.int64)
        if plaquette_orientations is None:
            orientations = np.ones_like(plaquettes, dtype=np.int32)
        else:
            orientations = np.asarray(plaquette_orientations, dtype=np.int64)
        if (
            plaquettes.ndim != 2
            or plaquettes.shape[1] != 4
            or orientations.shape != plaquettes.shape
            or np.any(plaquettes < 0)
            or np.any(plaquettes >= source.size)
            or np.any(np.abs(orientations) != 1)
        ):
            raise ValueError(
                "Plaquettes require edge indices and ±1 orientations with shape (P,4)."
            )
        if plaquettes.shape[0] > int(maximum_plaquettes):
            raise ReciprocalResourceError(
                "Reciprocal connectivity exceeds maximum_plaquettes."
            )
        points = np.asarray(mesh.fractional_points)
        displacement = points[target] + shift - points[source]
        tolerance = float(closure_tolerance)
        for edges, signs in zip(plaquettes, orientations, strict=True):
            oriented_displacements = displacement[edges] * signs[:, None]
            if (
                np.max(np.abs(np.sum(oriented_displacements, axis=0)), initial=0.0)
                > tolerance
            ):
                raise ValueError("An oriented reciprocal plaquette does not close.")
            oriented_source = np.where(signs > 0, source[edges], target[edges])
            oriented_target = np.where(signs > 0, target[edges], source[edges])
            if not np.array_equal(np.roll(oriented_source, -1), oriented_target):
                raise ValueError(
                    "Plaquette oriented edges do not form a contiguous cycle."
                )
        self.mesh = mesh
        self.source_indices = jnp.asarray(source, dtype=jnp.int32)
        self.target_indices = jnp.asarray(target, dtype=jnp.int32)
        self.reciprocal_shifts = jnp.asarray(shift, dtype=jnp.int32)
        self.reverse_indices = jnp.asarray(reverse, dtype=jnp.int32)
        self.plaquette_edges = jnp.asarray(plaquettes, dtype=jnp.int32)
        self.plaquette_orientations = jnp.asarray(orientations, dtype=jnp.int32)
        self.mesh_id = mesh.mesh_id
        self.cell_id = mesh.cell_id
        self.connectivity_id = canonical_fingerprint(
            {
                "kind": "reciprocal-connectivity-plan",
                "mesh": mesh.mesh_id,
                "arrays": array_tree_fingerprint(
                    {
                        "source": source,
                        "target": target,
                        "shift": shift,
                        "reverse": reverse,
                        "plaquettes": plaquettes,
                        "orientations": orientations,
                    }
                ),
            }
        )

    @classmethod
    def regular(
        cls,
        mesh: ReciprocalMeshPlan,
        /,
        *,
        axes: tuple[int, ...] | None = None,
        maximum_edges: int = 4_000_000,
        maximum_plaquettes: int = 2_000_000,
    ) -> "ReciprocalConnectivityPlan":
        if not isinstance(mesh, ReciprocalMeshPlan):
            raise TypeError("mesh must be ReciprocalMeshPlan.")
        selected = (
            tuple(
                axis
                for axis, (size, periodic) in enumerate(
                    zip(mesh.mesh_shape, mesh.cell.periodic_axes, strict=True)
                )
                if periodic and size > 1
            )
            if axes is None
            else tuple(axes)
        )
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(axis < 0 or axis >= mesh.rank for axis in selected)
            or any(not mesh.cell.periodic_axes[axis] for axis in selected)
            or any(mesh.mesh_shape[axis] <= 1 for axis in selected)
        ):
            raise ValueError(
                "Regular connectivity axes must be distinct sampled periodic axes."
            )
        indices = np.asarray(mesh.mesh_indices)
        lookup = {tuple(row): index for index, row in enumerate(indices)}
        sources: list[int] = []
        targets: list[int] = []
        shifts: list[tuple[int, ...]] = []
        edge_lookup: dict[tuple[int, int], int] = {}
        for point_index, coordinate in enumerate(indices):
            for axis in selected:
                for direction in (1, -1):
                    raw = coordinate.copy()
                    raw[axis] += direction
                    shift = np.zeros((mesh.rank,), dtype=np.int32)
                    if raw[axis] >= mesh.mesh_shape[axis]:
                        raw[axis] = 0
                        shift[axis] = 1
                    elif raw[axis] < 0:
                        raw[axis] = mesh.mesh_shape[axis] - 1
                        shift[axis] = -1
                    target = lookup[tuple(raw)]
                    edge_lookup[(point_index, direction * (axis + 1))] = len(sources)
                    sources.append(point_index)
                    targets.append(target)
                    shifts.append(tuple(shift))
        if len(sources) > int(maximum_edges):
            raise ReciprocalResourceError(
                "Regular reciprocal connectivity exceeds maximum_edges."
            )
        reverse = []
        for point_index, coordinate in enumerate(indices):
            del coordinate
            for axis in selected:
                forward_target = targets[edge_lookup[(point_index, axis + 1)]]
                reverse.append(edge_lookup[(forward_target, -(axis + 1))])
                backward_target = targets[edge_lookup[(point_index, -(axis + 1))]]
                reverse.append(edge_lookup[(backward_target, axis + 1)])
        plaquettes: list[tuple[int, int, int, int]] = []
        for first_axis, second_axis in combinations(selected, 2):
            for point_index in range(indices.shape[0]):
                edge_first = edge_lookup[(point_index, first_axis + 1)]
                after_first = targets[edge_first]
                edge_second = edge_lookup[(after_first, second_axis + 1)]
                after_second = targets[edge_second]
                edge_back_first = edge_lookup[(after_second, -(first_axis + 1))]
                after_back_first = targets[edge_back_first]
                edge_back_second = edge_lookup[(after_back_first, -(second_axis + 1))]
                plaquettes.append(
                    (edge_first, edge_second, edge_back_first, edge_back_second)
                )
        if len(plaquettes) > int(maximum_plaquettes):
            raise ReciprocalResourceError(
                "Regular reciprocal connectivity exceeds maximum_plaquettes."
            )
        return cls(
            mesh,
            sources,
            targets,
            shifts,
            reverse,
            plaquette_edges=np.asarray(plaquettes, dtype=np.int32).reshape((-1, 4)),
            maximum_edges=maximum_edges,
            maximum_plaquettes=maximum_plaquettes,
        )

    def prepare(self, /) -> "PreparedReciprocalConnectivity":
        return prepare_reciprocal_connectivity(self)


class PreparedReciprocalConnectivity(StrictModule, NonTrainableState):
    """Fixed-shape link displacement data for overlap/topology kernels."""

    plan: ReciprocalConnectivityPlan
    fractional_displacements: Array
    cartesian_displacements: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ReciprocalConnectivityPlan, /):
        if not isinstance(plan, ReciprocalConnectivityPlan):
            raise TypeError("plan must be ReciprocalConnectivityPlan.")
        points = plan.mesh.fractional_points
        fractional = (
            points[plan.target_indices]
            + plan.reciprocal_shifts.astype(points.dtype)
            - points[plan.source_indices]
        )
        cartesian = fractional @ plan.mesh.cell.reciprocal_vectors.astype(points.dtype)
        self.plan = plan
        self.fractional_displacements = fractional
        self.cartesian_displacements = cartesian
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-reciprocal-connectivity",
                "plan": plan.connectivity_id,
            }
        )

    @property
    def edge_count(self) -> int:
        return self.plan.source_indices.size

    @property
    def plaquette_count(self) -> int:
        return self.plan.plaquette_edges.shape[0]


def prepare_reciprocal_connectivity(
    plan: ReciprocalConnectivityPlan, /
) -> PreparedReciprocalConnectivity:
    return PreparedReciprocalConnectivity(plan)


__all__ = [
    "PreparedReciprocalConnectivity",
    "ReciprocalConnectivityPlan",
    "ReciprocalMeshPlan",
    "ReciprocalPathPlan",
    "ReciprocalResourceError",
    "prepare_reciprocal_connectivity",
]
