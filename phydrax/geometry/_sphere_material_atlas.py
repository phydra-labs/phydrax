#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepare nonsingular full-sphere material maps bound to physical cells.

The reference triangle follows physical-cell corner ordering. Projective
actions retain binary direction coefficients as rational numbers; runtime
evaluation and coordinate-map fidelity have separate evidence.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from typing import assert_never

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax._host_storage as host_storage_owner
import phydrax._meshcore as meshcore_owner
import phydrax.discretization._cell_geometry_validity as cell_geometry_validity_owner
import phydrax.discretization._coordinate_enclosure as coordinate_enclosure_owner
import phydrax.geometry._embedding_state as embedding_state_owner

# The public meshing module resolves result classes after atlas/transfer imports.
import phydrax.meshing as meshing_owner
import phydrax.meshing._association_state as association_state_owner
from phydrax.geometry._rational import fraction_endpoint, rational_vector

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import CellGeometryElement, CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateLiveStorage,
    reserve_polynomial,
)
from ..discretization._coordinate_enclosure import Expression
from ..discretization._exact_algebraic_geometry import rational_coefficient
from ..linalg._hermitian_spectral import (
    fraction_storage_upper,
)
from ..linalg._small_batched import (
    confirm_exact_action_3,
    exact_adjugate_action,
    exact_cofactor_determinant_2,
    exact_cofactor_determinant_3,
    exact_column_height_3,
    exact_small_linear_actions_id,
    ExactAdjugate3,
    prepare_exact_adjugate_3,
    SmallLinearSolvePlan,
    solve_small_linear,
)
from ..typing import checked, ConvertibleToArray
from ._atlas import AbstractBoundaryMap
from ._domain_numerics import (
    full_sphere_frame,
    gram_spectrum_bounds,
    interval_vector_cross,
    interval_vector_norm,
    matrix_norm_upper,
    sphere_radial_degree_one,
)
from ._interval_enclosure import (
    interval_add,
    interval_divide,
    interval_multiply,
    interval_subtract,
)
from ._meshing_domain import MeshingDomain, PatchCurveUse, PatchPoleUse
from .brep._patches import (
    AbstractSurfacePatch,
    sphere_source_equivalence,
    sphere_source_radius_terms,
)
from .brep._placed import PlacedSurface, source_transform_bounds


type _RationalRows = tuple[tuple[Fraction, ...], ...]
type _CornerToken = tuple[int, int, tuple[str, ...], tuple[Fraction, ...]]


def _sphere_material_query_upper(
    patches: np.ndarray,
    dimensions: np.ndarray,
    indices: np.ndarray,
    parameters: np.ndarray,
    occurrence_paths: tuple[tuple[tuple[str, ...], ...], ...],
    /,
) -> int:
    """Distinct source directions queried by the material-corner cache."""
    return len(
        {
            (
                int(patches[row]),
                int(dimensions[row, corner]),
                occurrence_paths[row][corner],
                int(indices[row, corner]),
                tuple(
                    float(value)
                    for value in parameters[row, corner, : int(dimensions[row, corner])]
                ),
            )
            for row in range(patches.size)
            for corner in range(3)
        }
    )


@dataclass(frozen=True, slots=True, eq=False)
class _SphereMaterialStrata:
    patches: np.ndarray
    dimensions: np.ndarray
    indices: np.ndarray
    parameters: np.ndarray
    occurrence_paths: tuple[tuple[tuple[str, ...], ...], ...]
    entity_ids: tuple[tuple[str, ...], ...]

    def queries_upper(self) -> int:
        return _sphere_material_query_upper(
            self.patches,
            self.dimensions,
            self.indices,
            self.parameters,
            self.occurrence_paths,
        )


def float_enclosure(value: Fraction) -> tuple[float, float]:
    center = float(value)
    if not np.isfinite(center):
        raise ValueError(
            "sphere_projective.coefficient must fit finite binary64; got overflow"
        )
    exact = Fraction(center)
    return (
        float(np.nextafter(center, -np.inf)) if exact > value else center,
        float(np.nextafter(center, np.inf)) if exact < value else center,
    )


@checked
def source_image_enclosure(
    source: AbstractSurfacePatch,
    directions: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    match source:
        case PlacedSurface():
            operation = source.definition
        case AbstractSurfacePatch():
            operation = source
        case invalid:
            assert_never(invalid)
    equivalence = sphere_source_equivalence(operation, budget=budget)
    if equivalence is None:
        raise TypeError(
            "sphere_material.source must have sphere-equivalence evidence; got unsupported expression"
        )
    definition, exact_radius = equivalence
    lower, upper = interval_vector_norm((directions, directions))
    normalized = interval_divide(
        (directions, directions), (lower[..., None], upper[..., None])
    )
    basis = np.column_stack(
        (
            np.asarray(definition.first_axis),
            np.asarray(definition.second_axis),
            np.asarray(definition.axis),
        )
    )
    low, high = source_transform_bounds(
        basis, np.asarray(normalized[0]).T, np.asarray(normalized[1]).T
    )
    radius_lower, radius_upper = float_enclosure(exact_radius)
    low, high = interval_multiply(
        (low.T, high.T), (np.asarray(radius_lower), np.asarray(radius_upper))
    )
    center = np.asarray(definition.center)
    low, high = interval_add((low, high), (center, center))
    low, high = np.asarray(low), np.asarray(high)
    match source:
        case PlacedSurface():
            low, high = source_transform_bounds(
                np.asarray(source.rotation), low.T, high.T
            )
            translation = np.asarray(source.translation)
            low, high = interval_add((low.T, high.T), (translation, translation))
        case AbstractSurfacePatch():
            pass
        case invalid:
            assert_never(invalid)
    return np.asarray(low), np.asarray(high)


@dataclass(frozen=True, slots=True)
class SphereProjectiveTriangleBounds:
    """Store denominator and signed reference-Jacobian bounds on one triangle."""

    denominator_lower: Fraction
    denominator_upper: Fraction
    jacobian_lower: Fraction
    jacobian_upper: Fraction


@dataclass(slots=True)
class SphereProjectiveActionBank:
    """Exact projective actions between two once-authenticated atlases.

    Every map drawn from one bank shares a single authentication of both
    atlases and charges the request ledger that owns the bank. Each target
    cone keeps its exact adjugate, proved once by ``adj(T) T == det(T) I``
    (``prepare_exact_adjugate_3``). A requested coefficient column
    ``T^{-1} r = adj(T) r / det(T)`` is then the unique exact solution without
    another elimination, and it is shared by every source cell incident to
    the binary ray ``r``. Verifying a stored column instead confirms
    ``T c == r`` against the nonsingular authenticated cone.
    """

    source: SphereMaterialCellAtlas
    target: SphereMaterialCellAtlas
    budget: CoordinateEnclosureBudget
    storage: CoordinateLiveStorage
    corners: dict[tuple[bool, int], _RationalRows]
    heights: dict[int, int]
    inverses: dict[int, ExactAdjugate3]
    columns: dict[tuple[int, bytes], tuple[Fraction, ...]]

    @classmethod
    def authenticate(
        cls,
        source: SphereMaterialCellAtlas,
        target: SphereMaterialCellAtlas,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        storage: CoordinateLiveStorage,
    ) -> SphereProjectiveActionBank:
        for atlas in (source, target):
            atlas.require_bound(
                atlas.domain,
                atlas.source_mesh,
                atlas.source_geometry,
                atlas.coordinate_contract,
                budget=budget,
            )
        return cls(source, target, budget, storage, {}, {}, {}, {})

    def require_atlases(
        self, source: SphereMaterialCellAtlas, target: SphereMaterialCellAtlas, /
    ) -> None:
        if source is not self.source or target is not self.target:
            raise ValueError(
                "sphere_projective.actions must belong to the authenticated source and target atlases; got another atlas pair"
            )

    def _retain(self, size: int, /) -> None:
        self.storage.set_bound(self.storage.bound + size)

    def cell_corners(self, target: bool, row: int, /) -> _RationalRows:
        """Return the exact corner rays of one source or target cone."""
        key = target, row
        value = self.corners.get(key)
        if value is None:
            atlas = self.target if target else self.source
            value = tuple(
                rational_vector(corner) for corner in np.asarray(atlas.directions)[row]
            )
            # Nine binary64 rationals and the tuple/dictionary slots.
            self._retain(fraction_storage_upper(9, 2 * 1100) + 512)
            self.corners[key] = value
        return value

    def corner_key(self, target: bool, row: int, column: int, /) -> bytes:
        atlas = self.target if target else self.source
        return np.asarray(atlas.directions)[row, column].tobytes()

    def _inverse(self, row: int, /) -> ExactAdjugate3:
        value = self.inverses.get(row)
        if value is not None:
            return value
        value = prepare_exact_adjugate_3(self.cell_corners(True, row), budget=self.budget)
        if value is None:
            raise ValueError(
                "sphere_projective.target_reference must have rank three; got singular coefficient action"
            )
        self._retain(fraction_storage_upper(10, 3 * value.height + 2) + 512)
        self.inverses[row] = value
        return value

    def target_determinant(self, row: int, /) -> Fraction:
        return self._inverse(row).determinant

    def column(
        self, row: int, ray: tuple[Fraction, ...], key: bytes, /
    ) -> tuple[Fraction, ...]:
        """Return the exact target coefficient column of one source ray."""
        value = self.columns.get((row, key))
        if value is not None:
            return value
        adjugate = self._inverse(row)
        value = exact_adjugate_action(adjugate, ray, budget=self.budget)
        self._retain(fraction_storage_upper(3, 8 * adjugate.height + 64) + 256)
        self.columns[(row, key)] = value
        return value

    def confirm_column(
        self,
        row: int,
        ray: tuple[Fraction, ...],
        key: bytes,
        value: tuple[Fraction, ...],
        /,
    ) -> bool:
        """Decide whether ``value`` is the exact target column of one source ray.

        The target cone determinant is the authenticated nonzero atlas
        determinant, so ``T value == ray`` proves the unique solution.
        """
        known = self.columns.get((row, key))
        if known is not None:
            return known == value
        corners = self.cell_corners(True, row)
        height = self.heights.get(row)
        if height is None:
            if self.target.exact_determinants[row] == 0:
                raise ValueError(
                    "sphere_projective.target_reference must have rank three; got singular coefficient action"
                )
            height = exact_column_height_3(corners, budget=self.budget)
            self.heights[row] = height
        if not confirm_exact_action_3(corners, height, value, ray, budget=self.budget):
            return False
        self._retain(fraction_storage_upper(3, 8 * height + 64) + 256)
        self.columns[(row, key)] = value
        return True


def _sphere_map_rows(
    source: SphereMaterialCellAtlas,
    source_cell_global_id: int,
    target: SphereMaterialCellAtlas,
    target_cell_global_id: int,
    /,
) -> tuple[int, int]:
    """Return both atlas rows after checking they share one source occurrence."""
    source_row, target_row = (
        source.cell_row(source_cell_global_id),
        target.cell_row(target_cell_global_id),
    )
    if (
        source.domain_id,
        source.source_id,
        source.source_revision,
        source.coordinate_contract.spatial_id,
        source.geometry_entity_ids[source_row],
        source.occurrence_paths[source_row],
    ) != (
        target.domain_id,
        target.source_id,
        target.source_revision,
        target.coordinate_contract.spatial_id,
        target.geometry_entity_ids[target_row],
        target.occurrence_paths[target_row],
    ):
        raise ValueError(
            "sphere_projective.source_identity must match across cells; got different occurrence or revision"
        )
    return source_row, target_row


def _sphere_shared_corners(
    source: SphereMaterialCellAtlas,
    source_row: int,
    target: SphereMaterialCellAtlas,
    target_row: int,
    source_corners: _RationalRows,
    target_matrix: _RationalRows,
    /,
) -> tuple[tuple[int | None, ...], list[int]]:
    """Name source corners that are target corners and the remaining columns."""
    target_tokens = {
        token: column for column, token in enumerate(target.corner_tokens[target_row])
    }
    shared = tuple(target_tokens.get(token) for token in source.corner_tokens[source_row])
    unknown = []
    for column, target_column in enumerate(shared):
        if target_column is None:
            unknown.append(column)
        # Source stratum, occurrence and parameter tokens identify a point.
        elif source_corners[column] != tuple(row[target_column] for row in target_matrix):
            raise ValueError(
                "sphere_projective.direction must agree for equal source tokens; got inconsistent realizations"
            )
    return shared, unknown


def _sphere_action_id(
    target_matrix: _RationalRows, source_corners: _RationalRows, unknown: list[int], /
) -> str:
    return exact_small_linear_actions_id(
        target_matrix,
        tuple(
            tuple(source_corners[column][axis] for column in unknown) for axis in range(3)
        ),
    )


def _sphere_map_id(
    source: SphereMaterialCellAtlas,
    target: SphereMaterialCellAtlas,
    source_cell_global_id: int,
    target_cell_global_id: int,
    action_id: str | None,
    shared: tuple[int | None, ...],
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "sphere-projective-reference-map",
            "source": source.atlas_id,
            "target": target.atlas_id,
            "source_cell": source_cell_global_id,
            "target_cell": target_cell_global_id,
            "actions": action_id,
            "shared_source_corners": tuple(
                (column, target_column)
                for column, target_column in enumerate(shared)
                if target_column is not None
            ),
        }
    )


class SphereProjectiveReferenceMap(StrictModule, NonTrainableState):
    """Map source-cell references to target-cell rational coordinates.

    Homogeneous barycentric coordinates define target cone halfplanes; their
    sum is the reference denominator. Coverage uses Fraction constructions.
    """

    source_atlas_id: str = eqx.field(static=True)
    target_atlas_id: str = eqx.field(static=True)
    source_cell_global_id: int = eqx.field(static=True)
    target_cell_global_id: int = eqx.field(static=True)
    geometry_entity_id: str = eqx.field(static=True)
    occurrence_path: tuple[str, ...] = eqx.field(static=True)
    exact_coefficients: _RationalRows = eqx.field(static=True)
    exact_denominator: tuple[Fraction, ...] = eqx.field(static=True)
    orientation_ratio: Fraction = eqx.field(static=True)
    coefficients: Array
    coefficient_lower: Array
    coefficient_upper: Array
    map_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: SphereMaterialCellAtlas,
        source_cell_global_id: int,
        target: SphereMaterialCellAtlas,
        target_cell_global_id: int,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        actions: SphereProjectiveActionBank | None = None,
    ) -> None:
        if actions is None:
            if budget is None:
                raise TypeError(
                    "sphere_projective.budget must name the owning request ledger; got neither a ledger nor an action bank"
                )
            with budget.activate(), budget.live_storage() as storage:
                actions = SphereProjectiveActionBank.authenticate(
                    source, target, budget=budget, storage=storage
                )
                self._construct(
                    source, source_cell_global_id, target, target_cell_global_id, actions
                )
            return
        actions.require_atlases(source, target)
        if budget is not None and budget is not actions.budget:
            raise ValueError(
                "sphere_projective.actions must charge the request budget that owns them; got another budget"
            )
        with actions.budget.activate():
            self._construct(
                source, source_cell_global_id, target, target_cell_global_id, actions
            )

    def _construct(
        self,
        source: SphereMaterialCellAtlas,
        source_cell_global_id: int,
        target: SphereMaterialCellAtlas,
        target_cell_global_id: int,
        actions: SphereProjectiveActionBank,
        /,
    ) -> None:
        budget = actions.budget
        source_row, target_row = _sphere_map_rows(
            source, source_cell_global_id, target, target_cell_global_id
        )
        source_corners = actions.cell_corners(False, source_row)
        target_matrix = tuple(zip(*actions.cell_corners(True, target_row), strict=True))
        with budget.temporary_scope():
            shared, unknown = _sphere_shared_corners(
                source, source_row, target, target_row, source_corners, target_matrix
            )
            corner_actions = {
                column: tuple(Fraction(int(axis == target_column)) for axis in range(3))
                for column, target_column in enumerate(shared)
                if target_column is not None
            }
            determinant = target.exact_determinants[target_row]
            action_id = None
            if unknown:
                # The bank's adjugate identity proves each shared column; the
                # identity still names the same exact coefficient request.
                determinant = actions.target_determinant(target_row)
                action_id = _sphere_action_id(target_matrix, source_corners, unknown)
                for column in unknown:
                    corner_actions[column] = actions.column(
                        target_row,
                        source_corners[column],
                        actions.corner_key(False, source_row, column),
                    )
            columns = tuple(corner_actions[column] for column in range(3))
            coefficients = tuple(
                (
                    columns[0][axis],
                    columns[1][axis] - columns[0][axis],
                    columns[2][axis] - columns[0][axis],
                )
                for axis in range(3)
            )
            self.source_atlas_id, self.target_atlas_id = (
                source.atlas_id,
                target.atlas_id,
            )
            self.source_cell_global_id, self.target_cell_global_id = (
                source_cell_global_id,
                target_cell_global_id,
            )
            self.geometry_entity_id, self.occurrence_path = (
                source.geometry_entity_ids[source_row],
                source.occurrence_paths[source_row],
            )
            self.exact_coefficients = coefficients
            self.exact_denominator = tuple(
                sum((row[column] for row in coefficients), Fraction(0))
                for column in range(3)
            )
            self.orientation_ratio = source.exact_determinants[source_row] / determinant
            enclosures = tuple(
                tuple(float_enclosure(value) for value in row) for row in coefficients
            )
            self.coefficients = jnp.asarray(
                [[float(value) for value in row] for row in coefficients]
            )
            self.coefficient_lower = jnp.asarray(
                [[value[0] for value in row] for row in enclosures]
            )
            self.coefficient_upper = jnp.asarray(
                [[value[1] for value in row] for row in enclosures]
            )
            self.map_id = _sphere_map_id(
                source,
                target,
                source_cell_global_id,
                target_cell_global_id,
                action_id,
                shared,
            )

        _, storage_upper, _, _ = host_storage_owner.retained_host_storage_profile(
            self, {}, meshcore_owner.is_native_host_buffer
        )
        budget.reserve(0, storage_upper)

    def _require_representatives(self) -> None:
        expected = np.asarray(
            [[float(value) for value in row] for row in self.exact_coefficients]
        )
        enclosures = tuple(
            tuple(float_enclosure(value) for value in row)
            for row in self.exact_coefficients
        )
        lower = np.asarray([[value[0] for value in row] for row in enclosures])
        upper = np.asarray([[value[1] for value in row] for row in enclosures])
        denominator = tuple(
            sum((row[column] for row in self.exact_coefficients), Fraction(0))
            for column in range(3)
        )
        if (
            denominator != self.exact_denominator
            or not np.array_equal(np.asarray(self.coefficients), expected)
            or not np.array_equal(np.asarray(self.coefficient_lower), lower)
            or not np.array_equal(np.asarray(self.coefficient_upper), upper)
        ):
            raise ValueError(
                "sphere_projective.coefficients must match their rational representatives; got changed values"
            )

    @checked
    def require_bound(
        self,
        source: SphereMaterialCellAtlas,
        target: SphereMaterialCellAtlas,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        actions: SphereProjectiveActionBank | None = None,
    ) -> None:
        """Verify the retained exact columns against both authenticated cones.

        Each stored target column ``c`` of an unshared source corner ``r`` is
        confirmed by ``T c == r`` for the nonsingular target cone ``T``; shared
        corners must keep their unit columns. The orientation ratio and the
        map identity are re-derived from the same exact inputs.
        """
        self._require_representatives()
        if actions is None:
            if budget is None:
                raise TypeError(
                    "sphere_projective.budget must name the owning request ledger; got neither a ledger nor an action bank"
                )
            with budget.activate(), budget.live_storage() as storage:
                self._verify(
                    source,
                    target,
                    SphereProjectiveActionBank.authenticate(
                        source, target, budget=budget, storage=storage
                    ),
                )
            return
        actions.require_atlases(source, target)
        if budget is not None and budget is not actions.budget:
            raise ValueError(
                "sphere_projective.actions must charge the request budget that owns them; got another budget"
            )
        with actions.budget.activate():
            self._verify(source, target, actions)

    def _verify(
        self,
        source: SphereMaterialCellAtlas,
        target: SphereMaterialCellAtlas,
        actions: SphereProjectiveActionBank,
        /,
    ) -> None:
        budget = actions.budget
        source_row, target_row = _sphere_map_rows(
            source, self.source_cell_global_id, target, self.target_cell_global_id
        )
        source_corners = actions.cell_corners(False, source_row)
        target_matrix = tuple(zip(*actions.cell_corners(True, target_row), strict=True))
        with budget.temporary_scope():
            shared, unknown = _sphere_shared_corners(
                source, source_row, target, target_row, source_corners, target_matrix
            )
            rows = self.exact_coefficients
            # Six column sums, three comparisons per shared unit column, and
            # one orientation product with its comparison.
            budget.reserve(8 + 3 * (3 - len(unknown)))
            columns = (
                tuple(row[0] for row in rows),
                tuple(row[0] + row[1] for row in rows),
                tuple(row[0] + row[2] for row in rows),
            )
            stale = (
                self.source_atlas_id,
                self.target_atlas_id,
                self.geometry_entity_id,
                self.occurrence_path,
            ) != (
                source.atlas_id,
                target.atlas_id,
                source.geometry_entity_ids[source_row],
                source.occurrence_paths[source_row],
            ) or any(
                columns[column]
                != tuple(Fraction(int(axis == target_column)) for axis in range(3))
                for column, target_column in enumerate(shared)
                if target_column is not None
            )
            stale = stale or any(
                not actions.confirm_column(
                    target_row,
                    source_corners[column],
                    actions.corner_key(False, source_row, column),
                    columns[column],
                )
                for column in unknown
            )
            stale = stale or (
                self.orientation_ratio * target.exact_determinants[target_row]
                != source.exact_determinants[source_row]
            )
            if stale or self.map_id != _sphere_map_id(
                source,
                target,
                self.source_cell_global_id,
                self.target_cell_global_id,
                _sphere_action_id(target_matrix, source_corners, unknown)
                if unknown
                else None,
                shared,
            ):
                raise ValueError(
                    "sphere_projective.binding must match both cell expressions; got stale binding"
                )

    def homogeneous(self, reference: Array, /) -> Array:
        return (
            self.coefficients[:, 0]
            + reference[..., 0, None] * self.coefficients[:, 1]
            + reference[..., 1, None] * self.coefficients[:, 2]
        )

    def map(self, reference: Array, /) -> Array:
        coordinates = self.homogeneous(reference)
        return coordinates[..., 1:] / jnp.sum(coordinates, axis=-1)[..., None]

    def reference_expressions(self) -> tuple[Expression, Expression]:
        """Return target rational coordinates in the source-cell reference.

        Denominator and Jacobian admissibility follow from the retained
        projective piece certificate.
        """

        self._require_representatives()
        denominator: coordinate_enclosure_owner.Polynomial = {
            index: value
            for index, value in zip(
                ((0, 0), (1, 0), (0, 1)), self.exact_denominator, strict=True
            )
            if value
        }
        coordinates: list[Expression] = []
        for row in self.exact_coefficients[1:]:
            numerator: coordinate_enclosure_owner.Polynomial = {
                index: value
                for index, value in zip(((0, 0), (1, 0), (0, 1)), row, strict=True)
                if value
            }
            coordinates.append(
                coordinate_enclosure_owner.RationalPolynomial(numerator, denominator)
            )
        return coordinates[0], coordinates[1]

    def differential(self, reference: Array, /) -> Array:
        coordinates = self.homogeneous(reference)
        denominator = jnp.sum(coordinates, axis=-1)
        slope = jnp.sum(self.coefficients[:, 1:], axis=0)
        return (
            self.coefficients[1:, 1:] / denominator[..., None, None]
            - coordinates[..., 1:, None] * slope / denominator[..., None, None] ** 2
        )

    def jacobian(self, reference: Array, /) -> Array:
        denominator = jnp.sum(self.homogeneous(reference), axis=-1)
        return float(self.orientation_ratio) / denominator**3

    def reference_images(
        self, points: _RationalRows, /
    ) -> tuple[tuple[Fraction, tuple[Fraction, Fraction]], ...]:
        """Return each source point's projective denominator and target point.

        Each point is checked inside the source reference triangle and the
        target direction cone once; triangles sharing these points reuse the
        same exact denominators and target coordinates.
        """
        if any(
            len(point) != 2 or any(not isinstance(value, Fraction) for value in point)
            for point in points
        ):
            raise TypeError(
                "sphere_projective.vertices must contain three pairs of Fraction coordinates; got another representation"
            )
        images = []
        for point in points:
            if point[0] < 0 or point[1] < 0 or point[0] + point[1] > 1:
                raise ValueError(
                    "sphere_projective.vertices must lie in the source reference triangle; got exterior coordinate"
                )
            homogeneous = tuple(
                row[0] + row[1] * point[0] + row[2] * point[1]
                for row in self.exact_coefficients
            )
            if any(value < 0 for value in homogeneous):
                raise ValueError(
                    "sphere_projective.vertices must lie in the target direction cone; got negative barycentric coordinate"
                )
            denominator = sum(homogeneous, Fraction(0))
            if denominator <= 0:
                raise ValueError(
                    "sphere_projective.reference must have positive denominator and orientation; got singular or reversed map"
                )
            images.append(
                (
                    denominator,
                    (homogeneous[1] / denominator, homogeneous[2] / denominator),
                )
            )
        return tuple(images)

    def triangle_bounds(
        self, denominators: tuple[Fraction, Fraction, Fraction], /
    ) -> SphereProjectiveTriangleBounds:
        """Bound denominator and Jacobian from three certified vertex images."""
        lower, upper = min(denominators), max(denominators)
        if lower <= 0 or self.orientation_ratio <= 0:
            raise ValueError(
                "sphere_projective.reference must have positive denominator and orientation; got singular or reversed map"
            )
        return SphereProjectiveTriangleBounds(
            lower,
            upper,
            self.orientation_ratio / upper**3,
            self.orientation_ratio / lower**3,
        )

    def certify_triangle(
        self, vertices: _RationalRows, /
    ) -> SphereProjectiveTriangleBounds:
        """Bound denominator and Jacobian over the rational reference triangle."""
        self._require_representatives()
        if len(vertices) != 3:
            raise TypeError(
                "sphere_projective.vertices must contain three pairs of Fraction coordinates; got another representation"
            )
        first, second, third = self.reference_images(vertices)
        return self.triangle_bounds((first[0], second[0], third[0]))


class SphereMaterialInverse(StrictModule):
    reference: Array
    source_residual: Array
    solve_residual: Array
    successful: Array


class _AbstractSphereMaterialAtlasData(StrictModule, NonTrainableState):
    """Store the prepared material arrays and their source-bound evidence."""

    domain: eqx.AbstractVar[MeshingDomain]
    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    coordinate_contract: SpatialCoordinateContract
    cell_global_ids: Array
    physical_rows: Array
    patches: Array
    physical_corner_global_ids: Array
    directions: Array
    centers: Array
    axes: Array
    radii: Array
    radius_terms: Array
    rotations: Array
    translations: Array
    source_slots: Array
    hemisphere_lower: Array
    direction_norm_upper: Array
    jacobian_lower: Array
    normalization_chord_bounds: Array
    hessian_bounds: Array
    source_polynomial_chord_bounds: Array
    source_corner_enclosure_bounds: Array
    source_fidelity_bounds: Array
    coordinate_preparation_work_units: int = eqx.field(static=True)
    coordinate_retained_basis_bytes: int = eqx.field(static=True)
    coordinate_storage_upper_bytes: int = eqx.field(static=True)
    inverse_plan: SmallLinearSolvePlan
    geometry_entity_ids: tuple[str, ...] = eqx.field(static=True)
    occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    corner_tokens: tuple[tuple[_CornerToken, ...], ...] = eqx.field(static=True)
    exact_determinants: tuple[Fraction, ...] = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    coverage_status: str = eqx.field(static=True)
    work_units: int = eqx.field(static=True)


class _PreparedSphereMaterialData(_AbstractSphereMaterialAtlasData):
    domain: MeshingDomain


class SphereMaterialCellAtlas(_AbstractSphereMaterialAtlasData, AbstractBoundaryMap):
    """Prepare a nonsingular full-sphere atlas of physical cells.

    Source strata establish coverage; coordinate-map enclosures establish
    whole-cell fidelity. Preparation binds the material arrays and evidence.
    """

    domain: MeshingDomain
    atlas_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        domain: MeshingDomain,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        cell_patches: ConvertibleToArray,
        corner_source_dimensions: ConvertibleToArray,
        corner_source_indices: ConvertibleToArray,
        corner_source_parameters: ConvertibleToArray,
        corner_source_occurrence_paths: tuple[tuple[tuple[str, ...], ...], ...],
        corner_source_entity_ids: tuple[tuple[str, ...], ...],
        maximum_fidelity: float,
        maximum_cells: int = 100000,
        maximum_work_units: int = 10000000,
        maximum_coordinate_memory_bytes: int = 256 * 1024**2,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:

        limits = embedding_state_owner.MeshCertificateLimits(
            maximum_work_units=maximum_work_units,
            maximum_scratch_bytes=maximum_coordinate_memory_bytes,
            maximum_distance_evaluations=maximum_work_units,
        )
        with embedding_state_owner.coordinate_certificate_request(
            limits, budget=budget
        ) as budget:
            request = _SphereMaterialRequest(
                cell_patches,
                corner_source_dimensions,
                corner_source_indices,
                corner_source_parameters,
                corner_source_occurrence_paths,
                corner_source_entity_ids,
                maximum_fidelity,
                maximum_cells,
                maximum_work_units,
            )
            data = _prepare_sphere_material_data(
                domain, mesh, geometry, coordinate_contract, request, budget=budget
            )
            self.domain = data.domain
            self.source_mesh = data.source_mesh
            self.source_geometry = data.source_geometry
            self.coordinate_contract = data.coordinate_contract
            self.cell_global_ids = data.cell_global_ids
            self.physical_rows = data.physical_rows
            self.patches = data.patches
            self.physical_corner_global_ids = data.physical_corner_global_ids
            self.directions = data.directions
            self.centers = data.centers
            self.axes = data.axes
            self.radii = data.radii
            self.radius_terms = data.radius_terms
            self.rotations = data.rotations
            self.translations = data.translations
            self.source_slots = data.source_slots
            self.hemisphere_lower = data.hemisphere_lower
            self.direction_norm_upper = data.direction_norm_upper
            self.jacobian_lower = data.jacobian_lower
            self.normalization_chord_bounds = data.normalization_chord_bounds
            self.hessian_bounds = data.hessian_bounds
            self.source_polynomial_chord_bounds = data.source_polynomial_chord_bounds
            self.source_corner_enclosure_bounds = data.source_corner_enclosure_bounds
            self.source_fidelity_bounds = data.source_fidelity_bounds
            self.coordinate_preparation_work_units = (
                data.coordinate_preparation_work_units
            )
            self.coordinate_retained_basis_bytes = data.coordinate_retained_basis_bytes
            self.coordinate_storage_upper_bytes = data.coordinate_storage_upper_bytes
            self.inverse_plan = data.inverse_plan
            self.geometry_entity_ids = data.geometry_entity_ids
            self.occurrence_paths = data.occurrence_paths
            self.corner_tokens = data.corner_tokens
            self.exact_determinants = data.exact_determinants
            self.source_geometry_id = data.source_geometry_id
            self.source_topology_id = data.source_topology_id
            self.source_id = data.source_id
            self.source_revision = data.source_revision
            self.domain_id = data.domain_id
            self.coverage_status = data.coverage_status
            self.work_units = data.work_units
            self.atlas_id = _sphere_material_identity(self)

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[SphereMaterialCellAtlas],
        value: SphereMaterialCellAtlas,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> SphereMaterialCellAtlas:
        """Validate stored bindings and reprove their coordinate source on ``budget``."""
        value.require_bound(
            value.domain,
            value.source_mesh,
            value.source_geometry,
            value.coordinate_contract,
            budget=budget,
        )
        return value

    @property
    def num_charts(self) -> int:
        return self.cell_global_ids.shape[0]

    @property
    def reference_dimension(self) -> int:
        return 2

    @property
    def ambient_dimension(self) -> int:
        return 3

    def direction(self, chart_indices: Array, reference: Array, /) -> Array:
        corners = self.directions[chart_indices]
        ray = (
            corners[..., 0, :]
            + reference[..., 0, None] * (corners[..., 1, :] - corners[..., 0, :])
            + reference[..., 1, None] * (corners[..., 2, :] - corners[..., 0, :])
        )
        return ray / jnp.linalg.norm(ray, axis=-1)[..., None]

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        slot = self.source_slots[chart_indices]
        direction = self.direction(chart_indices, reference)
        unit = jnp.sum(self.axes[slot] * direction[..., None, :], axis=-1)
        local = self.centers[slot] + self.radius_terms[slot, 0, None] * unit
        for term in range(1, self.radius_terms.shape[1]):
            local = local + self.radius_terms[slot, term, None] * unit
        return (
            jnp.sum(self.rotations[slot] * local[..., None, :], axis=-1)
            + self.translations[slot]
        )

    def differential(self, chart_indices: Array, reference: Array, /) -> Array:
        slot = self.source_slots[chart_indices]
        corners = self.directions[chart_indices]
        tangent = jnp.stack(
            (
                corners[..., 1, :] - corners[..., 0, :],
                corners[..., 2, :] - corners[..., 0, :],
            ),
            axis=-1,
        )
        ray = corners[..., 0, :] + jnp.sum(tangent * reference[..., None, :], axis=-1)
        norm = jnp.linalg.norm(ray, axis=-1)
        differential = (
            tangent / norm[..., None, None]
            - ray[..., :, None]
            * jnp.sum(ray[..., :, None] * tangent, axis=-2)[..., None, :]
            / norm[..., None, None] ** 3
        )
        unit = self.axes[slot] @ differential
        local = self.radius_terms[slot, 0, None, None] * unit
        for term in range(1, self.radius_terms.shape[1]):
            local = local + self.radius_terms[slot, term, None, None] * unit
        return self.rotations[slot] @ local

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        derivative = self.differential(chart_indices, reference)
        return jnp.linalg.norm(
            jnp.cross(derivative[..., :, 0], derivative[..., :, 1]), axis=-1
        )

    def inverse(self, chart_indices: Array, points: Array, /) -> SphereMaterialInverse:
        """Return radial inverse coordinates and native solve diagnostics.

        Successful solves provide radial coordinates; source membership has
        separate geometric evidence.
        """
        slot = self.source_slots[chart_indices]
        plan = self.inverse_plan
        placement = solve_small_linear(
            plan, self.rotations[slot], points - self.translations[slot]
        )
        # A common radial coefficient cancels in normalized barycentric
        # coordinates; no rounded effective-radius inverse is introduced.
        body = solve_small_linear(
            plan, self.axes[slot], placement.value - self.centers[slot]
        )
        barycentric = solve_small_linear(
            plan, jnp.swapaxes(self.directions[chart_indices], -1, -2), body.value
        )
        denominator = jnp.sum(barycentric.value, axis=-1)
        reference = barycentric.value[..., 1:] / denominator[..., None]
        success = (
            placement.successful
            & body.successful
            & barycentric.successful
            & (denominator > 0)
            & jnp.all(jnp.isfinite(reference), axis=-1)
        )
        return SphereMaterialInverse(
            reference,
            jnp.linalg.norm(self.map(chart_indices, reference) - points, axis=-1),
            jnp.maximum(
                placement.residual_norm,
                jnp.maximum(body.residual_norm, barycentric.residual_norm),
            ),
            success,
        )

    @checked
    def require_bound(
        self,
        domain: MeshingDomain,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:

        if (
            domain.content_domain_id(),
            mesh.mesh_id,
            mesh.topology_id,
            cell_geometry_validity_owner.cell_geometry_id(geometry),
            coordinate_contract.spatial_id,
        ) != (
            self.domain_id,
            self.source_mesh.mesh_id,
            self.source_topology_id,
            self.source_geometry_id,
            self.coordinate_contract.spatial_id,
        ):
            raise ValueError(
                "sphere_material.binding must match source, cells, coordinate map and units; got stale binding"
            )
        if _sphere_material_identity(self) != self.atlas_id:
            raise ValueError(
                "sphere_material.atlas_id must match material maps and evidence; got changed content"
            )
        if budget is not None:
            # Both retained sign bounds are checked for every material cell.
            budget.reserve(self.hemisphere_lower.size + self.jacobian_lower.size)
        if (
            self.coverage_status != "certified"
            or np.any(np.asarray(self.hemisphere_lower) <= 0)
            or np.any(np.asarray(self.jacobian_lower) <= 0)
        ):
            raise ValueError(
                "sphere_material.coverage must be certified with positive hemisphere and Jacobian bounds; got incomplete or singular coverage"
            )
        geometry.resolve(mesh, budget=budget)

    def cell_row(self, cell_global_id: int, /) -> int:
        ids = np.asarray(self.cell_global_ids)
        row = int(np.searchsorted(ids, cell_global_id))
        if row == ids.size or int(ids[row]) != cell_global_id:
            raise ValueError(
                "sphere_material.cell_global_id must identify a prepared cell; got absent identifier"
            )
        return row

    def projective_reference_map(
        self,
        source_cell_global_id: int,
        target: SphereMaterialCellAtlas,
        target_cell_global_id: int,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        actions: SphereProjectiveActionBank | None = None,
    ) -> SphereProjectiveReferenceMap:
        """Return target-cone halfplanes and target coordinates on a source cell."""
        return SphereProjectiveReferenceMap(
            self,
            source_cell_global_id,
            target,
            target_cell_global_id,
            budget=budget,
            actions=actions,
        )


def _sphere_material_identity(atlas: SphereMaterialCellAtlas, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "bound-sphere-material-cell-atlas",
            "domain": atlas.domain_id,
            "source": atlas.source_id,
            "revision": atlas.source_revision,
            "geometry": atlas.source_geometry_id,
            "topology": atlas.source_topology_id,
            "units": atlas.coordinate_contract.spatial_id,
            "coverage": atlas.coverage_status,
            "inverse_plan": atlas.inverse_plan.plan_id,
            "resource_evidence": (
                atlas.work_units,
                atlas.coordinate_preparation_work_units,
                atlas.coordinate_retained_basis_bytes,
                atlas.coordinate_storage_upper_bytes,
            ),
            "material_arrays": array_tree_fingerprint(
                (
                    atlas.cell_global_ids,
                    atlas.physical_rows,
                    atlas.physical_corner_global_ids,
                    atlas.patches,
                    atlas.directions,
                    atlas.centers,
                    atlas.axes,
                    atlas.radii,
                    atlas.rotations,
                    atlas.translations,
                    atlas.radius_terms,
                    atlas.source_slots,
                    atlas.hemisphere_lower,
                    atlas.direction_norm_upper,
                    atlas.jacobian_lower,
                    atlas.normalization_chord_bounds,
                    atlas.hessian_bounds,
                    atlas.source_polynomial_chord_bounds,
                    atlas.source_corner_enclosure_bounds,
                    atlas.source_fidelity_bounds,
                )
            ),
            "entities": atlas.geometry_entity_ids,
            "paths": atlas.occurrence_paths,
            "determinants": tuple(
                (value.numerator, value.denominator) for value in atlas.exact_determinants
            ),
            "corners": tuple(
                tuple(
                    (
                        token[0],
                        token[1],
                        token[2],
                        tuple((value.numerator, value.denominator) for value in token[3]),
                    )
                    for token in row
                )
                for row in atlas.corner_tokens
            ),
        }
    )


def corner_direction(
    domain: MeshingDomain,
    patch: int,
    dimension: int,
    index: int,
    parameters: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, _CornerToken]:
    path = domain.source_occurrences[dimension][index]
    source_index = domain.source_indices[dimension][index]
    if dimension == 0:
        poles: list[PatchPoleUse] = []
        for loop in domain.patches[patch].loops:
            for use in loop:
                match use:
                    case PatchPoleUse():
                        if use.corner == index:
                            poles.append(use)
                    case PatchCurveUse():
                        pass
                    case invalid:
                        assert_never(invalid)
        if not poles or any(
            (use.start[1] > 0) != (poles[0].start[1] > 0) for use in poles
        ):
            raise ValueError(
                "sphere_material.corner must identify a consistent pole use; got absent or inconsistent pole"
            )
        return np.asarray((0.0, 0.0, 1.0 if poles[0].start[1] > 0 else -1.0)), (
            0,
            source_index,
            path,
            (),
        )
    if dimension == 1:
        owner, loop, position = map(int, domain.curve_owners[index])
        if owner != patch or not np.isfinite(parameters[0]):
            raise ValueError(
                "sphere_material.seam must retain its curve owner and finite parameter; got different owner or invalid parameter"
            )
        use = domain.patches[owner].loops[loop][position]
        match use:
            case PatchCurveUse() if (
                min(use.first, use.last) <= parameters[0] <= max(use.first, use.last)
            ):
                pass
            case PatchCurveUse() | PatchPoleUse():
                raise ValueError(
                    "sphere_material.seam_parameter must lie in its declared curve use; got exterior parameter"
                )
            case invalid:
                assert_never(invalid)
        uv = np.asarray(
            use.pcurve.evaluate(jnp.asarray(parameters[0], dtype=jnp.float64))
        ).reshape((2,))
        token = (1, source_index, path, (Fraction(float(parameters[0])),))
    elif dimension == 2:
        if index != patch or not np.all(np.isfinite(parameters)):
            raise ValueError(
                "sphere_material.interior_corner must retain its face and finite parameters; got different face or invalid parameters"
            )
        uv = parameters
        token = (2, source_index, path, tuple(Fraction(float(value)) for value in uv))
    else:
        raise ValueError(
            "sphere_material.corner_dimension must lie between zero and two; got unsupported dimension"
        )
    domain.patches[patch].surface.validate_parameter_box(
        np.stack((uv, uv)), budget=budget
    )
    longitude, latitude = uv
    return np.asarray(
        (
            np.cos(longitude) * np.cos(latitude),
            np.sin(longitude) * np.cos(latitude),
            np.sin(latitude),
        )
    ), token


class _SphereMaterialRequest(StrictModule):
    cell_patches: ConvertibleToArray
    corner_source_dimensions: ConvertibleToArray
    corner_source_indices: ConvertibleToArray
    corner_source_parameters: ConvertibleToArray
    corner_source_occurrence_paths: tuple[tuple[tuple[str, ...], ...], ...]
    corner_source_entity_ids: tuple[tuple[str, ...], ...]
    maximum_fidelity: float
    maximum_cells: int
    maximum_work_units: int


class _SphereMaterialInputs(StrictModule):
    ids: np.ndarray
    cells: np.ndarray
    patches: np.ndarray
    dimensions: np.ndarray
    indices: np.ndarray
    parameters: np.ndarray
    lookup: tuple[dict[tuple[tuple[str, ...], int], int], ...]


class _SphereDirections(StrictModule):
    rays: np.ndarray
    tokens: tuple[tuple[_CornerToken, ...], ...]
    active_patches: tuple[int, ...]


class _SphereSources(StrictModule):
    centers: np.ndarray
    axes: np.ndarray
    radii: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    norms: tuple[float, ...]
    singulars: tuple[float, ...]
    radial_coefficients: np.ndarray


class _SphereRadialBounds(StrictModule):
    hemisphere: np.ndarray
    maximum_norm: np.ndarray
    determinants: tuple[Fraction, ...]
    jacobian: np.ndarray
    normalization: np.ndarray
    hessians: np.ndarray
    slots: dict[int, int]


class _SphereCoordinateBounds(StrictModule):
    polynomial: np.ndarray
    corners: np.ndarray


def _sphere_material_inputs(
    domain: MeshingDomain, mesh: CellMesh, request: _SphereMaterialRequest, /
) -> _SphereMaterialInputs:
    cell_patches = request.cell_patches
    corner_source_dimensions = request.corner_source_dimensions
    corner_source_indices = request.corner_source_indices
    corner_source_parameters = request.corner_source_parameters
    corner_source_occurrence_paths = request.corner_source_occurrence_paths
    corner_source_entity_ids = request.corner_source_entity_ids
    maximum_cells = request.maximum_cells
    maximum_work_units = request.maximum_work_units
    maximum_fidelity = request.maximum_fidelity
    if (
        any(block.cell_kind != "triangle" for block in mesh.blocks)
        or mesh.topological_dimension != 2
        or mesh.ambient_dimension != 3
    ):
        raise ValueError(
            "sphere_material.mesh must contain two-dimensional triangles in three dimensions; got another cell family or dimension"
        )
    ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    count = ids.size
    if (
        not count
        or count > maximum_cells
        or maximum_work_units < 1
        or not np.isfinite(maximum_fidelity)
        or maximum_fidelity < 0
    ):
        raise ValueError(
            "sphere_material.limits must admit nonempty cells, positive work and finite nonnegative fidelity; got invalid limits"
        )
    patches = np.asarray(cell_patches)
    dimensions, indices = (
        np.asarray(corner_source_dimensions),
        np.asarray(corner_source_indices),
    )
    parameters = np.asarray(corner_source_parameters, dtype=np.float64)
    if (
        patches.shape != (count,)
        or not np.issubdtype(patches.dtype, np.integer)
        or dimensions.shape != (count, 3)
        or indices.shape != (count, 3)
        or not np.issubdtype(dimensions.dtype, np.integer)
        or not np.issubdtype(indices.dtype, np.integer)
        or parameters.shape != (count, 3, 2)
    ):
        raise ValueError(
            "sphere_material.strata must match cell and corner shapes with integer indices; got incompatible arrays"
        )
    if (
        np.any(patches < 0)
        or np.any(patches >= len(domain.patches))
        or len(corner_source_occurrence_paths) != count
        or len(corner_source_entity_ids) != count
        or any(
            len(row) != 3
            for row in (*corner_source_occurrence_paths, *corner_source_entity_ids)
        )
    ):
        raise ValueError(
            "sphere_material.source_metadata must align with three corners of every cell; got incompatible paths or entities"
        )
    lookup = tuple(
        {
            (path, index): row
            for row, (path, index) in enumerate(zip(paths, indices_, strict=True))
        }
        for paths, indices_ in zip(
            domain.source_occurrences, domain.source_indices, strict=True
        )
    )
    return _SphereMaterialInputs(
        ids, cells, patches, dimensions, indices, parameters, lookup
    )


def _sphere_material_directions(
    domain: MeshingDomain,
    inputs: _SphereMaterialInputs,
    request: _SphereMaterialRequest,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _SphereDirections:
    count = inputs.ids.size
    patches, dimensions, indices, parameters, lookup = (
        inputs.patches,
        inputs.dimensions,
        inputs.indices,
        inputs.parameters,
        inputs.lookup,
    )
    corner_source_occurrence_paths = request.corner_source_occurrence_paths
    corner_source_entity_ids = request.corner_source_entity_ids
    ledger = budget

    direction_keys: list[tuple[int, int, int, tuple[float, ...]]] = []
    for row in range(count):
        for corner in range(3):
            dimension = int(dimensions[row, corner])
            path = corner_source_occurrence_paths[row][corner]
            if not 0 <= dimension <= 2:
                raise ValueError(
                    "sphere_material.corner_dimension must lie between zero and two; got unsupported stratum"
                )
            index = lookup[dimension].get((path, int(indices[row, corner])))
            if (
                index is None
                or domain.entity_id(dimension, index)
                != corner_source_entity_ids[row][corner]
            ):
                raise ValueError(
                    "sphere_material.corner_metadata must match the domain entity and occurrence; got stale metadata"
                )
            parameter_key = tuple(
                float(value) for value in parameters[row, corner, :dimension]
            )
            direction_keys.append((int(patches[row]), dimension, index, parameter_key))
    # Each distinct source stratum/parameter is queried once by the direction cache.
    queries_upper = _sphere_material_query_upper(
        patches, dimensions, indices, parameters, corner_source_occurrence_paths
    )
    if queries_upper > ledger.ticket.remaining_geometry_queries:
        ledger.ticket.refuse(0, queries_upper=queries_upper)
    ledger.reserve(0, count * 4096)
    rays = np.empty((count, 3, 3), dtype=np.float64)
    tokens: list[tuple[_CornerToken, ...]] = []
    direction_cache: dict[
        tuple[int, int, int, tuple[float, ...]], tuple[np.ndarray, _CornerToken]
    ] = {}
    active_patches = sorted(set(map(int, patches)))
    for patch in active_patches:
        if full_sphere_frame(domain, patch, budget=budget) is None:
            raise ValueError(
                "sphere_material.patch must define a full sphere; got a trimmed or unsupported patch"
            )
    with ledger.ticket.batch(0, queries_upper=queries_upper) as queries:
        for row in range(count):
            members = []
            for corner in range(3):
                key = direction_keys[3 * row + corner]
                prepared_direction = direction_cache.get(key)
                if prepared_direction is None:
                    queries.consume(geometry_queries=1)
                    prepared_direction = corner_direction(
                        domain, key[0], key[1], key[2], parameters[row, corner],
                        budget=budget,
                    )
                    direction_cache[key] = prepared_direction
                ray, token = prepared_direction
                rays[row, corner] = ray
                members.append(token)
            tokens.append(tuple(members))
    return _SphereDirections(rays, tuple(tokens), tuple(active_patches))


def _certify_sphere_material_coverage(
    inputs: _SphereMaterialInputs, directions: _SphereDirections, /
) -> None:
    patches = inputs.patches
    rays, tokens, active_patches = (
        directions.rays,
        directions.tokens,
        directions.active_patches,
    )
    # Source tokens establish incidence; matching direction realizations verify it.
    for patch in active_patches:
        rows = np.flatnonzero(patches == patch)
        values: dict[_CornerToken, np.ndarray] = {}
        chain: dict[tuple[_CornerToken, _CornerToken], int] = {}
        for row in rows:
            for corner, token in enumerate(tokens[row]):
                previous = values.setdefault(token, rays[row, corner])
                if not np.array_equal(previous, rays[row, corner]):
                    raise ValueError(
                        "sphere_material.direction must agree for equal source tokens; got inconsistent realizations"
                    )
            for first, second in ((0, 1), (1, 2), (2, 0)):
                edge = (tokens[row][first], tokens[row][second])
                reverse = (edge[1], edge[0])
                if chain.get(reverse, 0):
                    chain[reverse] -= 1
                else:
                    chain[edge] = chain.get(edge, 0) + 1
        if any(chain.values()):
            raise ValueError(
                "sphere_material.edge_chain must cancel by source tokens; got uncovered edges"
            )
        flat = rays[rows].reshape((-1, 3))
        local_cells = np.arange(flat.shape[0], dtype=np.int64).reshape((-1, 3))
        if not sphere_radial_degree_one((np.zeros(3), 1.0, 1.0), flat, local_cells):
            raise ValueError(
                "sphere_material.radial_chain must have degree one; got failed coverage proof"
            )


def _sphere_material_sources(
    domain: MeshingDomain,
    active_patches: tuple[int, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _SphereSources:
    centers, source_axes, radii, rotations, translations, norms, singulars = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    radial_terms: list[tuple[Fraction, ...]] = []
    for patch in active_patches:
        source = domain.patches[patch].surface
        match source:
            case PlacedSurface():
                operation = source.definition
            case AbstractSurfacePatch():
                operation = source
            case invalid:
                assert_never(invalid)
        equivalence = sphere_source_radius_terms(operation, budget=budget)
        if equivalence is None:
            raise TypeError(
                "sphere_material.source must have sphere-radius expression evidence; got unsupported expression"
            )
        definition, terms = equivalence
        exact_radius = sum(terms, Fraction(0))
        radial_terms.append(terms)
        basis = np.column_stack(
            (
                np.asarray(definition.first_axis),
                np.asarray(definition.second_axis),
                np.asarray(definition.axis),
            )
        )
        match source:
            case PlacedSurface():
                rotation = np.asarray(source.rotation)
                translation = np.asarray(source.translation)
            case AbstractSurfacePatch():
                rotation = np.eye(3, dtype=np.float64)
                translation = np.zeros(3, dtype=np.float64)
            case invalid:
                assert_never(invalid)
        low, _ = gram_spectrum_bounds(basis)
        pose_low, _ = gram_spectrum_bounds(rotation)
        radius = float(exact_radius)
        centers.append(np.asarray(definition.center))
        source_axes.append(basis)
        radii.append(radius)
        rotations.append(rotation)
        translations.append(translation)
        norms.append(sphere_radial_source_norm(source, budget=budget))
        singulars.append(
            float_enclosure(
                abs(exact_radius)
                * Fraction(float(np.nextafter(np.sqrt(low), -np.inf)))
                * Fraction(float(np.nextafter(np.sqrt(pose_low), -np.inf))),
            )[0]
        )
    term_width = max(map(len, radial_terms))
    radial_coefficients = np.asarray(
        [
            [float(value) for value in terms] + [0.0] * (term_width - len(terms))
            for terms in radial_terms
        ],
        dtype=np.float64,
    )
    return _SphereSources(
        np.asarray(centers),
        np.asarray(source_axes),
        np.asarray(radii),
        np.asarray(rotations),
        np.asarray(translations),
        tuple(norms),
        tuple(singulars),
        radial_coefficients,
    )


def _sphere_radial_bound_rows(
    rays: np.ndarray, /, *, budget: CoordinateEnclosureBudget
) -> Iterator[tuple[Fraction, float, float, Fraction] | None]:
    """Share exact direction coefficients within one authenticated preparation."""
    with budget.temporary_scope():
        budget.reserve(0, 1024)
        coordinates: dict[bytes, tuple[Fraction, ...]] = {}
        for triangle in rays:
            corners = []
            for direction in triangle:
                key = direction.tobytes()
                exact = coordinates.get(key)
                if exact is None:
                    # Three genuinely new coefficients; a binary64 Fraction's
                    # numerator/denominator and the cache entry fit this bound.
                    budget.reserve(3, 2048)
                    exact = rational_vector(direction)
                    coordinates[key] = exact
                corners.append(exact)
            matrix = tuple(tuple(corner[axis] for corner in corners) for axis in range(3))
            yield _sphere_radial_bounds_from_matrix(triangle, matrix, budget=budget)


def _sphere_material_radial_bounds(
    domain: MeshingDomain,
    inputs: _SphereMaterialInputs,
    directions: _SphereDirections,
    sources: _SphereSources,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _SphereRadialBounds:
    patches = inputs.patches
    rays, active_patches = directions.rays, directions.active_patches
    norms, singulars = sources.norms, sources.singulars
    ledger = budget
    hemisphere, maximum_norm, determinants, jacobian, normalization, hessians = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    slots = {patch: slot for slot, patch in enumerate(active_patches)}
    for row, radial in enumerate(_sphere_radial_bound_rows(rays, budget=budget)):
        if radial is None:
            raise ValueError(
                "sphere_material.reference_triangle must have a positive hemisphere bound; got failed enclosure"
            )
        determinant, floor, ceiling, unit_error = radial
        ledger.retain_basis((determinant,))
        sign = -1 if domain.patches[int(patches[row])].reversed else 1
        if determinant * sign <= 0:
            raise ValueError(
                "sphere_material.reference_orientation must agree with the source patch; got reversed determinant"
            )
        first = interval_subtract(
            (rays[row, 1], rays[row, 1]), (rays[row, 0], rays[row, 0])
        )
        second = interval_subtract(
            (rays[row, 2], rays[row, 2]), (rays[row, 0], rays[row, 0])
        )
        first = (np.asarray(first[0]), np.asarray(first[1]))
        second = (np.asarray(second[0]), np.asarray(second[1]))
        slot = slots[int(patches[row])]
        radial_defect = max(1 - Fraction(floor), Fraction(ceiling) - 1, Fraction(0))
        chord = float_enclosure(Fraction(norms[slot]) * (radial_defect + unit_error))[1]
        jacobian_floor = float_enclosure(
            Fraction(singulars[slot]) ** 2 * abs(determinant) / Fraction(ceiling) ** 3
        )[0]
        first_norm, second_norm = (
            float(interval_vector_norm(first)[1]),
            float(interval_vector_norm(second)[1]),
        )
        factor = Fraction(6) * Fraction(norms[slot]) / Fraction(floor) ** 2
        hemisphere.append(floor)
        maximum_norm.append(ceiling)
        determinants.append(determinant)
        jacobian.append(jacobian_floor)
        normalization.append(chord)
        hessians.append(
            (
                float_enclosure(factor * Fraction(first_norm) ** 2)[1],
                float_enclosure(factor * Fraction(first_norm) * Fraction(second_norm))[1],
                float_enclosure(factor * Fraction(second_norm) ** 2)[1],
            )
        )
    return _SphereRadialBounds(
        np.asarray(hemisphere),
        np.asarray(maximum_norm),
        tuple(determinants),
        np.asarray(jacobian),
        np.asarray(normalization),
        np.asarray(hessians),
        slots,
    )


class _SphereCoordinateCellBounds(StrictModule):
    polynomial: float
    corners: float


def _sphere_polynomial_chord_defect(
    expressions: tuple[Expression, ...],
    points: _RationalRows,
    variables: tuple[coordinate_enclosure_owner.Polynomial, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    """Bound the coordinate-map difference from its affine corner chord."""
    defect = Fraction(0)
    for component, expression in enumerate(expressions):
        corner_values = tuple(point[component] for point in points)
        chord = coordinate_enclosure_owner.add(
            coordinate_enclosure_owner.constant(corner_values[0], 2, budget=budget),
            coordinate_enclosure_owner.add(
                coordinate_enclosure_owner.scale(
                    variables[0],
                    corner_values[1] - corner_values[0],
                    budget=budget,
                ),
                coordinate_enclosure_owner.scale(
                    variables[1],
                    corner_values[2] - corner_values[0],
                    budget=budget,
                ),
                budget=budget,
            ),
            budget=budget,
        )
        defect += max(
            abs(rational_coefficient(value, role="sphere_material.chord_defect"))
            for value in coordinate_enclosure_owner.expression_bernstein_coefficients(
                coordinate_enclosure_owner.expression_add(
                    expression,
                    coordinate_enclosure_owner.expression_scale(chord, -1, budget=budget),
                    budget=budget,
                ),
                "simplex",
                2,
                budget=budget,
            )
        )
    return defect


def _sphere_coordinate_corner_error(
    source_bounds: tuple[np.ndarray, np.ndarray],
    points: _RationalRows,
) -> float:
    """Keep each actual corner's exact source-enclosure deviation unchanged."""
    low, high = source_bounds
    error = max(
        sum(
            (
                max(
                    abs(value - Fraction(float(low[corner, component]))),
                    abs(value - Fraction(float(high[corner, component]))),
                )
                for component, value in enumerate(point)
            ),
            Fraction(0),
        )
        for corner, point in enumerate(points)
    )
    return fraction_endpoint(error, upper=True, saturate=True)


def _sphere_coordinate_cell_bounds(
    source_bounds: tuple[np.ndarray, np.ndarray],
    element: CellGeometryElement,
    coefficients: _RationalRows,
    variables: tuple[coordinate_enclosure_owner.Polynomial, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _SphereCoordinateCellBounds:
    """Enclose one coordinate map's chord defect and source-corner deviation."""
    points = coordinate_enclosure_owner.affine_simplex_corner_images(
        element, coefficients, budget=budget
    )
    if points is not None:
        points = tuple(
            tuple(
                rational_coefficient(value, role="sphere_material.corner_image")
                for value in point
            )
            for point in points
        )
        # The full admitted source basis is affine, not merely its corner
        # samples. Its map is exactly its chord, so no defect expansion is needed.
        defect = Fraction(0)
    else:
        expressions = coordinate_enclosure_owner.coordinate_expressions(
            element, coefficients, budget=budget
        )
        if expressions is None:
            raise ValueError(
                "sphere_material.coordinate_map must provide rational expressions; got unavailable enclosure"
            )
        points = coordinate_enclosure_owner.coordinate_corner_images(
            element, coefficients, budget=budget
        )
        if points is None:
            raise ValueError(
                "sphere_material.coordinate_map must provide rational corner images; got unavailable enclosure"
            )
        points = tuple(
            tuple(
                rational_coefficient(value, role="sphere_material.corner_image")
                for value in point
            )
            for point in points
        )
        defect = _sphere_polynomial_chord_defect(
            expressions, points, variables, budget=budget
        )
    return _SphereCoordinateCellBounds(
        fraction_endpoint(defect, upper=True, saturate=True),
        _sphere_coordinate_corner_error(source_bounds, points),
    )


def _sphere_material_coordinate_bounds(
    domain: MeshingDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    inputs: _SphereMaterialInputs,
    directions: _SphereDirections,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _SphereCoordinateBounds:
    patches, rays = inputs.patches, directions.rays
    ledger = budget
    elements, routes, coordinate_values = geometry.resolve(mesh, budget=budget)
    # The request ledger owns cumulative coefficient work.
    ledger.reserve(
        0,
        coordinate_values.size * np.dtype(np.float64).itemsize
        + sum(route.size * np.dtype(np.int64).itemsize for route in routes),
    )
    with ledger.activate():
        # Validity, embedding and the material atlas consume the same exact
        # geometry-bound coefficients in this request, not separately rebuilt
        # equal-valued Fraction banks.
        source_bank = coordinate_enclosure_owner.prepared_coordinate_source_bank(
            geometry, budget=budget
        )
        ledger.reserve(
            0, 64 * len(source_bank) + 8 * sum(len(row) for row in source_bank)
        )
        coordinate_bank = tuple(
            tuple(
                rational_coefficient(
                    value,
                    role="CellGeometrySpec.rational_source_coordinates.coefficients",
                )
                for value in row
            )
            for row in source_bank
        )
    polynomial_bounds, corner_bounds = [], []
    # A cell-corner occurrence uses the same source law as every other corner
    # on its patch. Prove that law once per patch and evaluate its interval
    # image in a batch; retain per-occurrence bounds for the full-map checks.
    ledger.reserve(0, 2 * rays.nbytes)
    source_low, source_high = np.empty_like(rays), np.empty_like(rays)
    for patch in directions.active_patches:
        rows = np.flatnonzero(patches == patch)
        with ledger.temporary_scope():
            # Indexing, normalization and placed interval transforms coexist.
            ledger.reserve(0, rows.size * 3 * 3 * 8 * 64)
            low, high = source_image_enclosure(
                domain.patches[int(patch)].surface,
                rays[rows].reshape((-1, 3)),
                budget=budget,
            )
            source_low[rows] = low.reshape((-1, 3, 3))
            source_high[rows] = high.reshape((-1, 3, 3))
    offset = 0
    with ledger.activate():
        variables: tuple[coordinate_enclosure_owner.Polynomial, ...] | None = None
        for element, route in zip(elements, routes, strict=True):
            actual_routes = np.asarray(route, dtype=np.int64)
            # Prove the complete current root affine, not just agreement of
            # sampled corners. The existing batch owner then authenticates the
            # actual element and every route/control operation once.
            if coordinate_enclosure_owner.affine_simplex_source(element, budget=budget):
                with ledger.temporary_scope():
                    count = actual_routes.shape[0]
                    ledger.reserve(0, 512 + 3 * count * (8 + 64))
                    corner_rows = np.arange(3 * count, dtype=np.int64).reshape(count, 3)
                    images = coordinate_enclosure_owner.coordinate_vertex_images(
                        ((element, actual_routes, corner_rows),),
                        coordinate_bank,
                        3 * count,
                        mismatch="sphere_material.affine_corners must retain each actual corner occurrence",
                        budget=budget,
                    )
                    if images is None or any(image is None for image in images):
                        raise ValueError(
                            "sphere_material.coordinate_map must provide complete "
                            "rational affine corner images; got unavailable enclosure"
                        )
                    for row, local in enumerate(actual_routes):
                        with ledger.temporary_scope():
                            # Preserve the existing per-cell scalar/error visits.
                            ledger.reserve(local.size * 3)
                            points = tuple(
                                tuple(
                                    rational_coefficient(
                                        value, role="sphere_material.corner_image"
                                    )
                                    for value in image
                                )
                                for image in images[3 * row : 3 * row + 3]
                                if image is not None
                            )
                            polynomial_bounds.append(0.0)
                            corner_bounds.append(
                                _sphere_coordinate_corner_error(
                                    (source_low[offset], source_high[offset]), points
                                )
                            )
                            offset += 1
            else:
                if variables is None:
                    variables = coordinate_enclosure_owner.axes(2, budget=budget)
                for local in actual_routes:
                    with ledger.temporary_scope():
                        ledger.reserve(local.size * 3)
                        bounds = _sphere_coordinate_cell_bounds(
                            (source_low[offset], source_high[offset]),
                            element,
                            tuple(coordinate_bank[int(index)] for index in local),
                            variables,
                            budget=budget,
                        )
                        polynomial_bounds.append(bounds.polynomial)
                        corner_bounds.append(bounds.corners)
                        offset += 1
    return _SphereCoordinateBounds(
        np.asarray(polynomial_bounds), np.asarray(corner_bounds)
    )


def _prepare_sphere_material_data(
    domain: MeshingDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    coordinate_contract: SpatialCoordinateContract,
    request: _SphereMaterialRequest,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PreparedSphereMaterialData:
    """Prepare source strata, certify coverage and enclose each material map."""
    inputs = _sphere_material_inputs(domain, mesh, request)
    measurement = budget.current_measurement
    directions = _sphere_material_directions(domain, inputs, request, budget=budget)
    _certify_sphere_material_coverage(inputs, directions)
    sources = _sphere_material_sources(domain, directions.active_patches, budget=budget)
    radial = _sphere_material_radial_bounds(
        domain, inputs, directions, sources, budget=budget
    )
    coordinates = _sphere_material_coordinate_bounds(
        domain, mesh, geometry, inputs, directions, budget=budget
    )
    ids, cells, patches = inputs.ids, inputs.cells, inputs.patches
    rays, tokens = directions.rays, directions.tokens
    centers, source_axes, radii = sources.centers, sources.axes, sources.radii
    rotations, translations, radial_coefficients = (
        sources.rotations,
        sources.translations,
        sources.radial_coefficients,
    )
    hemisphere, maximum_norm, determinants = (
        radial.hemisphere,
        radial.maximum_norm,
        radial.determinants,
    )
    jacobian, normalization, hessians, slots = (
        radial.jacobian,
        radial.normalization,
        radial.hessians,
        radial.slots,
    )
    polynomial_bounds, corner_bounds = coordinates.polynomial, coordinates.corners
    maximum_fidelity = request.maximum_fidelity
    work = measurement.work_units
    fidelity = np.nextafter(
        np.asarray(polynomial_bounds)
        + np.asarray(corner_bounds)
        + np.asarray(normalization),
        np.inf,
    )
    if (
        np.any(~np.isfinite(fidelity))
        or np.any(fidelity > maximum_fidelity)
        or np.any(np.asarray(jacobian) <= 0)
    ):
        raise ValueError(
            "sphere_material.coordinate_map must satisfy fidelity and positive Jacobian bounds; got excessive deviation or failed regularity"
        )
    order = np.argsort(ids, kind="stable")
    geometry_id = cell_geometry_validity_owner.cell_geometry_id(geometry)
    entity_ids = tuple(domain.entity_id(2, int(patch)) for patch in patches[order])
    paths = tuple(domain.source_occurrences[2][int(patch)] for patch in patches[order])
    return _PreparedSphereMaterialData(
        domain=domain,
        source_mesh=mesh,
        source_geometry=geometry,
        coordinate_contract=coordinate_contract,
        cell_global_ids=jnp.asarray(ids[order]),
        physical_rows=jnp.asarray(order, dtype=jnp.int64),
        physical_corner_global_ids=jnp.asarray(
            np.asarray(mesh.vertex_global_ids)[cells][order]
        ),
        patches=jnp.asarray(patches[order], dtype=jnp.int32),
        directions=jnp.asarray(rays[order]),
        centers=jnp.asarray(np.asarray(centers)),
        axes=jnp.asarray(np.asarray(source_axes)),
        radii=jnp.asarray(np.asarray(radii)),
        rotations=jnp.asarray(np.asarray(rotations)),
        radius_terms=jnp.asarray(radial_coefficients),
        translations=jnp.asarray(np.asarray(translations)),
        source_slots=jnp.asarray(
            [slots[int(patch)] for patch in patches[order]], dtype=jnp.int32
        ),
        hemisphere_lower=jnp.asarray(np.asarray(hemisphere)[order]),
        direction_norm_upper=jnp.asarray(np.asarray(maximum_norm)[order]),
        jacobian_lower=jnp.asarray(np.asarray(jacobian)[order]),
        normalization_chord_bounds=jnp.asarray(np.asarray(normalization)[order]),
        hessian_bounds=jnp.asarray(np.asarray(hessians)[order]),
        source_polynomial_chord_bounds=jnp.asarray(np.asarray(polynomial_bounds)[order]),
        source_corner_enclosure_bounds=jnp.asarray(np.asarray(corner_bounds)[order]),
        source_fidelity_bounds=jnp.asarray(fidelity[order]),
        geometry_entity_ids=entity_ids,
        occurrence_paths=paths,
        coordinate_preparation_work_units=work,
        coordinate_retained_basis_bytes=measurement.retained_basis_bytes,
        coordinate_storage_upper_bytes=measurement.peak_bytes_upper,
        inverse_plan=SmallLinearSolvePlan(3),
        corner_tokens=tuple(tokens[row] for row in order),
        exact_determinants=tuple(determinants[row] for row in order),
        source_geometry_id=geometry_id,
        source_topology_id=mesh.topology_id,
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        domain_id=domain.domain_id,
        coverage_status="certified",
        work_units=work,
    )


def sphere_material_strata(
    domain: MeshingDomain, result: meshing_owner.CellMeshingResult, /
) -> _SphereMaterialStrata:
    """Validated source strata of accepted physical material-cell corners."""

    if (
        not result.audit.passed
        or result.certification is None
        or not result.certification.passed
    ):
        raise ValueError(
            "sphere_material.result must have passing audit and certification; got unaccepted publication"
        )
    vertex_set, face_set = result.mesh.entity_set(0), result.mesh.entity_set(2)
    associations = [
        value
        for value in result.associations
        if value.association_kind
        in (
            association_state_owner.GeometryAssociationKind.SURFACE,
            association_state_owner.GeometryAssociationKind.BREP,
        )
        and value.source_id == domain.source_id
        and value.source_revision == domain.source_revision
    ]
    vertex = next(
        (
            value
            for value in associations
            if value.target_entity_set_id == vertex_set.entity_set_id
        ),
        None,
    )
    face = next(
        (
            value
            for value in associations
            if value.target_entity_set_id == face_set.entity_set_id
        ),
        None,
    )
    if (
        vertex is None
        or face is None
        or not np.all(np.asarray(vertex.resolved))
        or not np.all(np.asarray(face.resolved))
    ):
        raise ValueError(
            "sphere_material.result.associations must resolve every cell and vertex; got missing source strata"
        )
    vertex_rows = vertex.target_rows(
        np.asarray(result.mesh.vertex_global_ids, dtype=np.int64)
    )
    ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in result.mesh.blocks]
    )
    face_rows = face.target_rows(ids)
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in result.mesh.blocks]
    )
    lookup = {
        (path, index): row
        for row, (path, index) in enumerate(
            zip(domain.source_occurrences[2], domain.source_indices[2], strict=True)
        )
    }
    patches = []
    for row in face_rows:
        patch = lookup.get(
            (
                face.source_entities.occurrence_path(row),
                int(np.asarray(face.source_indices)[row]),
            )
        )
        if (
            patch is None
            or int(np.asarray(face.source_dimensions)[row]) != 2
            or face.source_entities.entity_id(row) != domain.entity_id(2, patch)
        ):
            raise ValueError(
                "sphere_material.result.cell_identity must match its domain face and occurrence; got stale metadata"
            )
        patches.append(patch)
    corner_rows = vertex_rows[cells]
    return _SphereMaterialStrata(
        patches=np.asarray(patches, dtype=np.int64),
        dimensions=np.asarray(vertex.source_dimensions)[corner_rows],
        indices=np.asarray(vertex.source_indices)[corner_rows],
        parameters=np.asarray(vertex.parameters)[corner_rows],
        occurrence_paths=tuple(
            tuple(vertex.source_entities.occurrence_path(row) for row in corners)
            for corners in corner_rows
        ),
        entity_ids=tuple(
            tuple(vertex.source_entities.entity_id(row) for row in corners)
            for corners in corner_rows
        ),
    )


@checked
def sphere_material_atlas_from_result(
    domain: MeshingDomain,
    result: meshing_owner.CellMeshingResult,
    /,
    *,
    maximum_fidelity: float,
    maximum_cells: int = 100_000,
    maximum_work_units: int = 10_000_000,
    maximum_coordinate_memory_bytes: int = 256 * 1024**2,
    budget: CoordinateEnclosureBudget | None = None,
) -> SphereMaterialCellAtlas:
    """Prepare a material atlas from accepted cell and vertex source strata."""
    strata = sphere_material_strata(domain, result)
    return SphereMaterialCellAtlas(
        domain,
        result.mesh,
        result.geometry,
        result.coordinate_contract,
        cell_patches=strata.patches,
        corner_source_dimensions=strata.dimensions,
        corner_source_indices=strata.indices,
        corner_source_parameters=strata.parameters,
        corner_source_occurrence_paths=strata.occurrence_paths,
        corner_source_entity_ids=strata.entity_ids,
        maximum_fidelity=maximum_fidelity,
        maximum_cells=maximum_cells,
        maximum_work_units=maximum_work_units,
        maximum_coordinate_memory_bytes=maximum_coordinate_memory_bytes,
        budget=budget,
    )


__all__ = [
    "SphereMaterialCellAtlas",
    "SphereMaterialInverse",
    "SphereProjectiveReferenceMap",
    "SphereProjectiveTriangleBounds",
    "sphere_material_atlas_from_result",
]


def sphere_cone_cofactors(
    matrix: _RationalRows, /, *, budget: CoordinateEnclosureBudget
) -> _RationalRows:
    """Prepare the three rational source-cone numerator actions."""
    reserve_polynomial(27, 27, 0, 4 * (1024 + 1075) + 2, budget=budget)
    corners = tuple(
        tuple(matrix[axis][corner] for axis in range(3)) for corner in range(3)
    )
    return tuple(
        (
            exact_cofactor_determinant_2((first[1], first[2]), (second[1], second[2])),
            exact_cofactor_determinant_2((first[2], first[0]), (second[2], second[0])),
            exact_cofactor_determinant_2((first[0], first[1]), (second[0], second[1])),
        )
        for first, second in (
            (corners[1], corners[2]),
            (corners[2], corners[0]),
            (corners[0], corners[1]),
        )
    )


def sphere_radial_triangle_bounds(
    directions: np.ndarray, /, *, budget: CoordinateEnclosureBudget
) -> tuple[Fraction, float, float, Fraction] | None:
    """Return radial cone and hemisphere bounds for a reference triangle."""
    budget.reserve(9)
    matrix = tuple(rational_vector(row) for row in directions.T)
    return _sphere_radial_bounds_from_matrix(directions, matrix, budget=budget)


def _sphere_radial_bounds_from_matrix(
    directions: np.ndarray,
    matrix: _RationalRows,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Fraction, float, float, Fraction] | None:
    """Consume exact coefficients constructed from these binary directions."""
    with budget.temporary_scope():
        # Only the scalar determinant is needed: fourteen arithmetic visits
        # and at most eighteen expanded input occurrences. Finite binary64
        # fractions use at most 1024 numerator and 1075 denominator bits;
        # that format bound avoids an unused second coefficient traversal.
        reserve_polynomial(14, 14, 0, 18 * (1024 + 1075) + 4, budget=budget)
        determinant = exact_cofactor_determinant_3(
            (matrix[0][0], matrix[0][1], matrix[0][2]),
            (matrix[1][0], matrix[1][1], matrix[1][2]),
            (matrix[2][0], matrix[2][1], matrix[2][2]),
        )
    if determinant == 0:
        return None
    first = interval_subtract(
        (directions[1], directions[1]), (directions[0], directions[0])
    )
    second = interval_subtract(
        (directions[2], directions[2]), (directions[0], directions[0])
    )
    first = (np.asarray(first[0]), np.asarray(first[1]))
    second = (np.asarray(second[0]), np.asarray(second[1]))
    normal_upper = float(interval_vector_norm(interval_vector_cross(first, second))[1])
    if normal_upper <= 0:
        return None
    floor = float_enclosure(abs(determinant) / Fraction(normal_upper))[0]
    lows, highs = interval_vector_norm((directions, directions))
    ceiling = float(np.max(highs))
    if floor <= 0 or not np.isfinite(ceiling):
        return None
    unit_error = max(
        (
            max(Fraction(float(high)) - 1, 1 - Fraction(float(low)), Fraction(0))
            for low, high in zip(lows, highs, strict=True)
        ),
        default=Fraction(0),
    )
    return determinant, floor, ceiling, unit_error


@checked
def sphere_radial_source_norm(
    source: AbstractSurfacePatch, /, *, budget: CoordinateEnclosureBudget
) -> float:
    """Return the source and placement norm for normalization bounds."""
    match source:
        case PlacedSurface():
            operation = source.definition
        case AbstractSurfacePatch():
            operation = source
        case invalid:
            assert_never(invalid)
    equivalence = sphere_source_radius_terms(operation, budget=budget)
    if equivalence is None:
        raise TypeError(
            "sphere_material.normalization.source must have sphere-radius expression evidence; got unsupported source"
        )
    definition, terms = equivalence
    basis = np.column_stack(
        (
            np.asarray(definition.first_axis),
            np.asarray(definition.second_axis),
            np.asarray(definition.axis),
        )
    )
    match source:
        case PlacedSurface():
            rotation = np.asarray(source.rotation)
        case AbstractSurfacePatch():
            rotation = np.eye(3, dtype=np.float64)
        case invalid:
            assert_never(invalid)
    return float_enclosure(
        abs(sum(terms, Fraction(0)))
        * Fraction(matrix_norm_upper(basis))
        * Fraction(matrix_norm_upper(rotation))
    )[1]
