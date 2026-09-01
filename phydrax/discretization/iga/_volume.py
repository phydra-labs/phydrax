#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._basis import TensorSplineBasisSpec
from ._geometry import NURBSGeometryState


if TYPE_CHECKING:
    from ._topology import PatchAtlas


def _host_array(value: object, /, *, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a concrete real array.") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _fraction(value: float, /) -> Fraction:
    return Fraction.from_float(float(value))


@dataclass(frozen=True, slots=True, order=True)
class BoundaryFacetId:
    """Stable identity of one oriented tensor-block boundary facet."""

    block_id: str
    axis: int
    side: int

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("A boundary facet requires a non-empty block_id.")
        if self.axis < 0:
            raise ValueError("A boundary facet axis must be non-negative.")
        if self.side not in (0, 1):
            raise ValueError("A boundary facet side must be zero or one.")


@dataclass(frozen=True, slots=True)
class TensorNURBSVolume:
    """One fixed tensor NURBS map used by certification and parameterization."""

    block_id: str
    basis: TensorSplineBasisSpec
    geometry: NURBSGeometryState
    patch_id: str
    numeric_revision: str

    def __init__(
        self,
        block_id: str,
        basis: TensorSplineBasisSpec,
        geometry: NURBSGeometryState,
        /,
        *,
        patch_id: str | None = None,
        numeric_revision: str = "initial",
    ):
        identifier = str(block_id)
        revision = str(numeric_revision)
        if not identifier or not revision:
            raise ValueError("Block identity and numeric revision must be non-empty.")
        if not isinstance(basis, TensorSplineBasisSpec):
            raise TypeError("basis must be a TensorSplineBasisSpec.")
        if not isinstance(geometry, NURBSGeometryState):
            raise TypeError("geometry must be a NURBSGeometryState.")
        if geometry.control_shape != basis.control_shape:
            raise ValueError("NURBS geometry and basis control shapes must agree.")
        object.__setattr__(self, "block_id", identifier)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(
            self, "patch_id", identifier if patch_id is None else str(patch_id)
        )
        object.__setattr__(self, "numeric_revision", revision)

    @property
    def parametric_dimension(self) -> int:
        return self.basis.parametric_dimension

    @property
    def ambient_dimension(self) -> int:
        return self.geometry.ambient_dimension

    @property
    def volume_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "tensor-nurbs-volume",
                "block": self.block_id,
                "basis": self.basis.basis_id,
                "patch": self.patch_id,
                "numeric_revision": self.numeric_revision,
                "control_points": array_tree_fingerprint(
                    np.asarray(self.geometry.control_points)
                ),
                "weights": array_tree_fingerprint(np.asarray(self.geometry.weights)),
            }
        )

    @property
    def facets(self) -> tuple[BoundaryFacetId, ...]:
        return tuple(
            BoundaryFacetId(self.block_id, axis, side)
            for axis in range(self.parametric_dimension)
            for side in (0, 1)
        )

    def boundary_data(self, facet: BoundaryFacetId, /) -> tuple[np.ndarray, np.ndarray]:
        """Return the exact boundary control net and positive rational weights."""
        self._validate_facet(facet)
        selector: list[slice | int] = [slice(None)] * self.parametric_dimension
        selector[facet.axis] = 0 if facet.side == 0 else -1
        points = _host_array(
            self.geometry.control_points[tuple(selector)], name="boundary control points"
        )
        weights = _host_array(
            self.geometry.weights[tuple(selector)], name="boundary weights"
        )
        return points, weights

    def homogeneous_boundary(self, facet: BoundaryFacetId, /) -> np.ndarray:
        points, weights = self.boundary_data(facet)
        return np.concatenate((points * weights[..., None], weights[..., None]), axis=-1)

    def normalized_boundary_knots(
        self, facet: BoundaryFacetId, /
    ) -> tuple[tuple[Fraction, ...], ...]:
        self._validate_facet(facet)
        result: list[tuple[Fraction, ...]] = []
        for axis_index, axis in enumerate(self.basis.axes):
            if axis_index == facet.axis:
                continue
            knots = tuple(float(value) for value in np.asarray(axis.knots))
            lower, upper = axis.parameter_interval
            lower_ = _fraction(lower)
            width = _fraction(upper) - lower_
            result.append(tuple((_fraction(value) - lower_) / width for value in knots))
        return tuple(result)

    def control_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        points = _host_array(self.geometry.control_points, name="control points")
        flat = points.reshape((-1, points.shape[-1]))
        return np.min(flat, axis=0), np.max(flat, axis=0)

    def boundary_aabb(self, facet: BoundaryFacetId, /) -> tuple[np.ndarray, np.ndarray]:
        points, _ = self.boundary_data(facet)
        flat = points.reshape((-1, points.shape[-1]))
        return np.min(flat, axis=0), np.max(flat, axis=0)

    def _validate_facet(self, facet: BoundaryFacetId, /) -> None:
        if not isinstance(facet, BoundaryFacetId):
            raise TypeError("facet must be a BoundaryFacetId.")
        if facet.block_id != self.block_id:
            raise ValueError("Boundary facet belongs to a different tensor block.")
        if facet.axis >= self.parametric_dimension:
            raise ValueError("Boundary facet axis is outside the parameter domain.")


@dataclass(frozen=True, slots=True)
class FacetCorrespondence:
    """Exact affine parameter identification between two block facets."""

    left: BoundaryFacetId
    right: BoundaryFacetId
    right_axis_order: tuple[int, ...]
    right_axis_reversed: tuple[bool, ...]
    association_id: str

    def __init__(
        self,
        left: BoundaryFacetId,
        right: BoundaryFacetId,
        /,
        *,
        right_axis_order: tuple[int, ...] | None = None,
        right_axis_reversed: tuple[bool, ...] | None = None,
        association_id: str | None = None,
    ):
        if not isinstance(left, BoundaryFacetId) or not isinstance(
            right, BoundaryFacetId
        ):
            raise TypeError(
                "Facet correspondence endpoints must be BoundaryFacetId values."
            )
        if left == right:
            raise ValueError("A facet cannot be identified with itself.")
        dimension = len(right_axis_order) if right_axis_order is not None else -1
        if right_axis_order is None:
            dimension = len(right_axis_reversed) if right_axis_reversed is not None else 0
            order = tuple(range(dimension))
        else:
            order = tuple(int(value) for value in right_axis_order)
        reversed_ = (
            (False,) * len(order)
            if right_axis_reversed is None
            else tuple(bool(value) for value in right_axis_reversed)
        )
        if len(reversed_) != len(order) or tuple(sorted(order)) != tuple(
            range(len(order))
        ):
            raise ValueError(
                "Facet parameter routing must be a permutation with one reversal flag per axis."
            )
        identifier = association_id or canonical_fingerprint(
            {
                "kind": "facet-correspondence",
                "left": (left.block_id, left.axis, left.side),
                "right": (right.block_id, right.axis, right.side),
                "order": order,
                "reversed": reversed_,
            }
        )
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "right_axis_order", order)
        object.__setattr__(self, "right_axis_reversed", reversed_)
        object.__setattr__(self, "association_id", str(identifier))


@dataclass(frozen=True, slots=True)
class IncidenceCheck:
    """Deterministic outcome of an exact rational facet comparison."""

    matched: bool
    theorem_available: bool
    reason: str
    association_id: str


@dataclass(frozen=True, slots=True)
class BlockComplex:
    """Finite tensor-block complex with explicit exact facet identifications."""

    volumes: tuple[TensorNURBSVolume, ...]
    incidences: tuple[FacetCorrespondence, ...]
    permitted_boundary_contacts: tuple[tuple[BoundaryFacetId, BoundaryFacetId], ...]
    complex_id: str

    def __init__(
        self,
        volumes: tuple[TensorNURBSVolume, ...],
        incidences: tuple[FacetCorrespondence, ...] = (),
        /,
        *,
        permitted_boundary_contacts: tuple[
            tuple[BoundaryFacetId, BoundaryFacetId], ...
        ] = (),
        patch_atlas: PatchAtlas | None = None,
    ):
        values = tuple(volumes)
        if not values or not all(
            isinstance(value, TensorNURBSVolume) for value in values
        ):
            raise TypeError("BlockComplex requires at least one TensorNURBSVolume.")
        identifiers = tuple(value.block_id for value in values)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Tensor block identifiers must be unique.")
        dimension = values[0].parametric_dimension
        ambient = values[0].ambient_dimension
        if any(
            value.parametric_dimension != dimension or value.ambient_dimension != ambient
            for value in values
        ):
            raise ValueError("All blocks in one complex must share dimensions.")
        by_id = {value.block_id: value for value in values}
        routes = tuple(incidences)
        used: set[BoundaryFacetId] = set()
        for route in routes:
            if not isinstance(route, FacetCorrespondence):
                raise TypeError("incidences must contain FacetCorrespondence values.")
            for facet in (route.left, route.right):
                if facet.block_id not in by_id:
                    raise ValueError("Facet incidence references an unknown block.")
                by_id[facet.block_id]._validate_facet(facet)
                if facet in used:
                    raise ValueError("A block facet may occur in at most one incidence.")
                used.add(facet)
            if len(route.right_axis_order) != dimension - 1:
                raise ValueError(
                    "Facet correspondence dimension does not match the blocks."
                )
        contacts: list[tuple[BoundaryFacetId, BoundaryFacetId]] = []
        for first, second in permitted_boundary_contacts:
            if first == second:
                raise ValueError("A permitted contact requires two different facets.")
            for facet in (first, second):
                if facet.block_id not in by_id:
                    raise ValueError("Permitted contact references an unknown block.")
                by_id[facet.block_id]._validate_facet(facet)
            contacts.append(tuple(sorted((first, second))))
        if len(set(contacts)) != len(contacts):
            raise ValueError("Permitted boundary contacts must be unique.")
        atlas_id = None if patch_atlas is None else patch_atlas.atlas_id
        object.__setattr__(self, "volumes", values)
        object.__setattr__(self, "incidences", routes)
        object.__setattr__(self, "permitted_boundary_contacts", tuple(contacts))
        object.__setattr__(
            self,
            "complex_id",
            canonical_fingerprint(
                {
                    "kind": "tensor-nurbs-block-complex",
                    "volumes": [value.volume_id for value in values],
                    "incidences": [value.association_id for value in routes],
                    "permitted_contacts": [
                        (
                            (first.block_id, first.axis, first.side),
                            (second.block_id, second.axis, second.side),
                        )
                        for first, second in contacts
                    ],
                    "patch_atlas": atlas_id,
                }
            ),
        )

    @property
    def parametric_dimension(self) -> int:
        return self.volumes[0].parametric_dimension

    @property
    def ambient_dimension(self) -> int:
        return self.volumes[0].ambient_dimension

    @property
    def exterior_facets(self) -> tuple[BoundaryFacetId, ...]:
        interior = {
            facet
            for incidence in self.incidences
            for facet in (incidence.left, incidence.right)
        }
        return tuple(
            facet
            for volume in self.volumes
            for facet in volume.facets
            if facet not in interior
        )

    def volume(self, block_id: str, /) -> TensorNURBSVolume:
        for value in self.volumes:
            if value.block_id == block_id:
                return value
        raise KeyError(f"Unknown tensor block {block_id!r}.")

    def exact_incidence_checks(self) -> tuple[IncidenceCheck, ...]:
        return tuple(
            exact_facet_incidence(
                self.volume(incidence.left.block_id),
                self.volume(incidence.right.block_id),
                incidence,
            )
            for incidence in self.incidences
        )

    def connected(self) -> bool:
        reached = {self.volumes[0].block_id}
        changed = True
        while changed:
            changed = False
            for incidence in self.incidences:
                left = incidence.left.block_id
                right = incidence.right.block_id
                if left in reached and right not in reached:
                    reached.add(right)
                    changed = True
                elif right in reached and left not in reached:
                    reached.add(left)
                    changed = True
        return len(reached) == len(self.volumes)


def _route_right_boundary(
    values: np.ndarray, correspondence: FacetCorrespondence, /
) -> np.ndarray:
    routed = np.transpose(
        values,
        axes=(*correspondence.right_axis_order, values.ndim - 1),
    )
    for axis, reversed_ in enumerate(correspondence.right_axis_reversed):
        if reversed_:
            routed = np.flip(routed, axis=axis)
    return routed


def _proportional_homogeneous(left: np.ndarray, right: np.ndarray, /) -> bool:
    if left.shape != right.shape:
        return False
    left_flat = left.reshape((-1, left.shape[-1]))
    right_flat = right.reshape((-1, right.shape[-1]))
    left_scale = _fraction(left_flat[0, -1])
    right_scale = _fraction(right_flat[0, -1])
    if left_scale <= 0 or right_scale <= 0:
        return False
    for left_value, right_value in zip(left_flat.flat, right_flat.flat, strict=True):
        if _fraction(left_value) * right_scale != _fraction(right_value) * left_scale:
            return False
    return True


def exact_facet_incidence(
    left_volume: TensorNURBSVolume,
    right_volume: TensorNURBSVolume,
    correspondence: FacetCorrespondence,
    /,
) -> IncidenceCheck:
    """Prove equality from knot vectors and proportional homogeneous control nets."""
    if (
        correspondence.left.block_id != left_volume.block_id
        or correspondence.right.block_id != right_volume.block_id
    ):
        raise ValueError("Facet correspondence endpoints do not match supplied volumes.")
    if left_volume.ambient_dimension != right_volume.ambient_dimension:
        return IncidenceCheck(
            False, False, "ambient dimensions differ", correspondence.association_id
        )
    left_knots = left_volume.normalized_boundary_knots(correspondence.left)
    right_knots_raw = right_volume.normalized_boundary_knots(correspondence.right)
    if len(left_knots) != len(correspondence.right_axis_order):
        return IncidenceCheck(
            False, False, "facet dimensions differ", correspondence.association_id
        )
    right_knots: list[tuple[Fraction, ...]] = []
    for left_axis, right_axis in enumerate(correspondence.right_axis_order):
        values = right_knots_raw[right_axis]
        if correspondence.right_axis_reversed[left_axis]:
            values = tuple(Fraction(1) - value for value in reversed(values))
        right_knots.append(values)
    if left_knots != tuple(right_knots):
        return IncidenceCheck(
            False, True, "normalized knot vectors differ", correspondence.association_id
        )
    left = left_volume.homogeneous_boundary(correspondence.left)
    right = _route_right_boundary(
        right_volume.homogeneous_boundary(correspondence.right), correspondence
    )
    if not _proportional_homogeneous(left, right):
        return IncidenceCheck(
            False,
            True,
            "homogeneous boundary control nets differ",
            correspondence.association_id,
        )
    return IncidenceCheck(
        True, True, "exact rational incidence established", correspondence.association_id
    )


def aabbs_strictly_separated(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    /,
) -> tuple[bool, int | None, float]:
    """Prove disjoint convex hulls when one coordinate interval separates them."""
    first_lower, first_upper = first
    second_lower, second_upper = second
    gaps = np.maximum(second_lower - first_upper, first_lower - second_upper)
    axis = int(np.argmax(gaps))
    gap = float(gaps[axis])
    return gap > 0.0, axis if gap > 0.0 else None, gap


def exact_planar_facet_halfspaces(
    left_volume: TensorNURBSVolume,
    right_volume: TensorNURBSVolume,
    correspondence: FacetCorrespondence,
    /,
) -> tuple[bool, str]:
    """Prove adjacent control hulls occupy opposite halfspaces of a shared plane."""
    if left_volume.ambient_dimension != 3 or left_volume.parametric_dimension != 3:
        return False, "planar halfspace proof currently requires three-dimensional blocks"
    points, _ = left_volume.boundary_data(correspondence.left)
    flat = points.reshape((-1, 3))
    origin = flat[0]
    normal: np.ndarray | None = None
    for first, second in combinations(flat[1:], 2):
        candidate = np.cross(first - origin, second - origin)
        if np.any(candidate != 0.0):
            normal = candidate
            break
    if normal is None:
        return False, "shared facet does not supply an exact plane"
    if np.any((flat - origin) @ normal != 0.0):
        return False, "shared facet control net is not exactly planar"
    left_points = _host_array(
        left_volume.geometry.control_points, name="left control points"
    )
    right_points = _host_array(
        right_volume.geometry.control_points, name="right control points"
    )
    left_signed = (left_points.reshape((-1, 3)) - origin) @ normal
    right_signed = (right_points.reshape((-1, 3)) - origin) @ normal
    left_nonpositive = np.all(left_signed <= 0.0)
    left_nonnegative = np.all(left_signed >= 0.0)
    right_nonpositive = np.all(right_signed <= 0.0)
    right_nonnegative = np.all(right_signed >= 0.0)
    opposite = (left_nonpositive and right_nonnegative) or (
        left_nonnegative and right_nonpositive
    )
    if not opposite:
        return False, "adjacent block control hulls are not in opposite halfspaces"
    left_boundary_count = int(np.count_nonzero(left_signed == 0.0))
    right_boundary_count = int(np.count_nonzero(right_signed == 0.0))
    expected_left = int(np.prod(points.shape[:-1]))
    right_boundary, _ = right_volume.boundary_data(correspondence.right)
    expected_right = int(np.prod(right_boundary.shape[:-1]))
    if left_boundary_count != expected_left or right_boundary_count != expected_right:
        return False, "a non-interface control point lies on the shared plane"
    return True, "exact planar interface separates adjacent control hulls"


__all__ = [
    "BlockComplex",
    "BoundaryFacetId",
    "FacetCorrespondence",
    "IncidenceCheck",
    "TensorNURBSVolume",
    "aabbs_strictly_separated",
    "exact_facet_incidence",
    "exact_planar_facet_halfspaces",
]
