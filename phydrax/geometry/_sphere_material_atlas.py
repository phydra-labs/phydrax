#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-bound nonsingular material maps of actual full-sphere physical cells.

A radial cell map is a view of its original SpherePatch (and authored placement),
not a replacement sphere or a lat/long ghost-cell correspondence. Its reference
triangle is the actual physical cell's canonical reference triangle. Exact
projective coefficient actions retain original binary direction coefficients;
runtime floating evaluation and physical coordinate fidelity are separate.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from fractions import Fraction
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._coordinate_enclosure import CoordinateEnclosureBudget, Expression
from ..linalg._small_batched import (
    prepare_exact_small_linear_actions,
    SmallLinearSolvePlan,
    solve_small_linear,
)
from ..typing import ConvertibleToArray
from ._atlas import AbstractBoundaryMap
from ._interval_enclosure import (
    interval_add,
    interval_divide,
    interval_multiply,
    interval_subtract,
)
from ._meshing_domain import (
    _full_sphere_frame,
    _gram_spectrum_bounds,
    _interval_vector_cross,
    _interval_vector_norm,
    _matrix_norm_upper,
    _sphere_radial_degree_one,
    MeshingDomain,
    PatchCurveUse,
    PatchPoleUse,
)
from .brep._patches import (
    AbstractSurfacePatch,
    sphere_source_equivalence,
    sphere_source_radius_terms,
)
from .brep._placed import PlacedSurface, source_transform_bounds


if TYPE_CHECKING:
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..discretization._cell_mesh import CellMesh
    from ..meshing._result import CellMeshingResult


type _RationalRows = tuple[tuple[Fraction, ...], ...]
type _CornerToken = tuple[int, int, tuple[str, ...], tuple[Fraction, ...]]


def _fraction_rows(values: np.ndarray) -> _RationalRows:
    return tuple(tuple(Fraction(float(value)) for value in row) for row in values)


def _float_enclosure(value: Fraction) -> tuple[float, float]:
    center = float(value)
    if not np.isfinite(center):
        raise ValueError("A projective coefficient exceeds finite binary64 evaluation.")
    exact = Fraction(center)
    return (
        float(np.nextafter(center, -np.inf)) if exact > value else center,
        float(np.nextafter(center, np.inf)) if exact < value else center,
    )


def _source_image_enclosure(
    source: AbstractSurfacePatch, directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    operation = source.definition if isinstance(source, PlacedSurface) else source
    equivalence = sphere_source_equivalence(operation)
    if equivalence is None:
        raise TypeError(
            "Radial material maps require a proved original sphere source expression."
        )
    definition, exact_radius = equivalence
    lower, upper = _interval_vector_norm((directions, directions))
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
    low, high = source_transform_bounds(basis, normalized[0].T, normalized[1].T)
    radius_lower, radius_upper = _float_enclosure(exact_radius)
    low, high = interval_multiply(
        (low.T, high.T), (np.asarray(radius_lower), np.asarray(radius_upper))
    )
    center = np.asarray(definition.center)
    low, high = interval_add((low, high), (center, center))
    if isinstance(source, PlacedSurface):
        low, high = source_transform_bounds(np.asarray(source.rotation), low.T, high.T)
        translation = np.asarray(source.translation)
        low, high = interval_add((low.T, high.T), (translation, translation))
    return low, high


@dataclass(frozen=True, slots=True)
class SphereProjectiveTriangleBounds:
    """Exact denominator and signed reference-Jacobian bounds on one real piece."""

    denominator_lower: Fraction
    denominator_upper: Fraction
    jacobian_lower: Fraction
    jacobian_upper: Fraction


class SphereProjectiveReferenceMap(StrictModule, NonTrainableState):
    """Exact affine homogeneous coefficients with a genuine rational quotient.

    The input is an actual source cell reference, not rounded gnomonic vertices.
    Three homogeneous barycentric target coordinates are nonnegative target
    cone halfplanes. Their sum is the target reference denominator. Native
    overlap must retain exact Fraction constructions before claiming coverage.
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

    def __init__(
        self,
        source: SphereMaterialCellAtlas,
        source_cell_global_id: int,
        target: SphereMaterialCellAtlas,
        target_cell_global_id: int,
        /,
        *,
        coordinate_budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        source.require_bound(
            source.domain,
            source.source_mesh,
            source.source_geometry,
            source.coordinate_contract,
        )
        target.require_bound(
            target.domain,
            target.source_mesh,
            target.source_geometry,
            target.coordinate_contract,
        )
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
                "Projective material correspondence cannot merge source occurrences or revisions."
            )
        source_corners = _fraction_rows(np.asarray(source.directions)[source_row])
        target_matrix = _fraction_rows(np.asarray(target.directions)[target_row].T)
        target_tokens = {
            token: column for column, token in enumerate(target.corner_tokens[target_row])
        }
        shared = tuple(
            target_tokens.get(token) for token in source.corner_tokens[source_row]
        )
        corner_actions: list[tuple[Fraction, ...] | None] = [None, None, None]
        unknown = []
        for column, target_column in enumerate(shared):
            if target_column is None:
                unknown.append(column)
                continue
            # Only original stratum/occurrence/parameter tokens identify this
            # source point. Equal mesh-local IDs or positions are not authority.
            if source_corners[column] != tuple(
                row[target_column] for row in target_matrix
            ):
                raise ValueError(
                    "One authoritative sphere source point has inconsistent direction realizations."
                )
            corner_actions[column] = tuple(
                Fraction(int(axis == target_column)) for axis in range(3)
            )
        determinant = target.exact_determinants[target_row]
        action_id = None
        if unknown:
            right = tuple(
                tuple(source_corners[column][axis] for column in unknown)
                for axis in range(3)
            )
            prepared = prepare_exact_small_linear_actions(
                target_matrix, right, coordinate_budget=coordinate_budget
            )
            if prepared.actions is None:
                raise ValueError(
                    "The target material reference has no exact nonsingular coefficient action."
                )
            determinant, action_id = prepared.determinant, prepared.input_id
            for slot, column in enumerate(unknown):
                corner_actions[column] = tuple(row[slot] for row in prepared.actions)
        columns: list[tuple[Fraction, ...]] = []
        for action in corner_actions:
            if action is None:
                raise RuntimeError(
                    "A sphere projective map omitted its actual source coefficient column."
                )
            columns.append(action)
        coefficients = tuple(
            (
                columns[0][axis],
                columns[1][axis] - columns[0][axis],
                columns[2][axis] - columns[0][axis],
            )
            for axis in range(3)
        )
        self.source_atlas_id, self.target_atlas_id = source.atlas_id, target.atlas_id
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
            sum((row[column] for row in coefficients), Fraction(0)) for column in range(3)
        )
        self.orientation_ratio = source.exact_determinants[source_row] / determinant
        enclosures = tuple(
            tuple(_float_enclosure(value) for value in row) for row in coefficients
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
        self.map_id = canonical_fingerprint(
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

    def _require_representatives(self) -> None:
        expected = np.asarray(
            [[float(value) for value in row] for row in self.exact_coefficients]
        )
        enclosures = tuple(
            tuple(_float_enclosure(value) for value in row)
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
                "Projective numerical representatives no longer match their authoritative exact coefficients."
            )

    def require_bound(
        self, source: SphereMaterialCellAtlas, target: SphereMaterialCellAtlas, /
    ) -> None:
        self._require_representatives()
        expected = SphereProjectiveReferenceMap(
            source, self.source_cell_global_id, target, self.target_cell_global_id
        )
        if (
            self.source_atlas_id,
            self.target_atlas_id,
            self.geometry_entity_id,
            self.occurrence_path,
            self.exact_coefficients,
            self.exact_denominator,
            self.orientation_ratio,
            self.map_id,
        ) != (
            expected.source_atlas_id,
            expected.target_atlas_id,
            expected.geometry_entity_id,
            expected.occurrence_path,
            expected.exact_coefficients,
            expected.exact_denominator,
            expected.orientation_ratio,
            expected.map_id,
        ):
            raise ValueError(
                "Projective reference map is stale for the original source/target cell expressions."
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
        """Exact target reference coordinates in the actual source reference.

        These are rational arguments, not the affine interpolation of mapped
        corners. Denominator/Jacobian admissibility remains with the retained
        projective piece certificate.
        """
        from ..discretization._coordinate_enclosure import Polynomial, RationalPolynomial

        self._require_representatives()
        denominator: Polynomial = {
            index: value
            for index, value in zip(
                ((0, 0), (1, 0), (0, 1)), self.exact_denominator, strict=True
            )
            if value
        }
        coordinates: list[Expression] = []
        for row in self.exact_coefficients[1:]:
            numerator: Polynomial = {
                index: value
                for index, value in zip(((0, 0), (1, 0), (0, 1)), row, strict=True)
                if value
            }
            coordinates.append(RationalPolynomial(numerator, denominator))
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

    def certify_triangle(
        self, vertices: _RationalRows, /
    ) -> SphereProjectiveTriangleBounds:
        """Bound the complete exact constructed triangle, not three sampled values."""
        self._require_representatives()
        if len(vertices) != 3 or any(
            len(point) != 2 or any(not isinstance(value, Fraction) for value in point)
            for point in vertices
        ):
            raise TypeError(
                "Projective piece vertices must retain exact rational constructions."
            )
        values = []
        for point in vertices:
            if point[0] < 0 or point[1] < 0 or point[0] + point[1] > 1:
                raise ValueError(
                    "A correspondence piece leaves its actual source reference triangle."
                )
            homogeneous = tuple(
                row[0] + row[1] * point[0] + row[2] * point[1]
                for row in self.exact_coefficients
            )
            if any(value < 0 for value in homogeneous):
                raise ValueError(
                    "A correspondence piece leaves its actual target direction cone."
                )
            values.append(sum(homogeneous, Fraction(0)))
        lower, upper = min(values), max(values)
        if lower <= 0 or self.orientation_ratio <= 0:
            raise ValueError(
                "Projective reference correspondence is singular or orientation reversing."
            )
        return SphereProjectiveTriangleBounds(
            lower,
            upper,
            self.orientation_ratio / upper**3,
            self.orientation_ratio / lower**3,
        )


class SphereMaterialInverse(StrictModule):
    reference: Array
    source_residual: Array
    solve_residual: Array
    successful: Array


class _AbstractSphereMaterialAtlasData(StrictModule, NonTrainableState):
    """Private derived data; it is not an admitted material-map owner."""

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
    """Complete nonsingular source atlas of actual scientific physical cells.

    Admission derives all numerical representatives, whole-cell bounds and
    coverage from the original source and original coordinate map. Cached
    positive statuses or material arrays are never constructor inputs.
    """

    domain: MeshingDomain
    atlas_id: str = eqx.field(static=True)

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
        maximum_cells: int = 100_000,
        maximum_work_units: int = 10_000_000,
        maximum_coordinate_memory_bytes: int = 256 * 1024**2,
        coordinate_budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        data = _prepare_sphere_material_data(
            domain,
            mesh,
            geometry,
            coordinate_contract,
            cell_patches=cell_patches,
            corner_source_dimensions=corner_source_dimensions,
            corner_source_indices=corner_source_indices,
            corner_source_parameters=corner_source_parameters,
            corner_source_occurrence_paths=corner_source_occurrence_paths,
            corner_source_entity_ids=corner_source_entity_ids,
            maximum_fidelity=maximum_fidelity,
            maximum_cells=maximum_cells,
            maximum_work_units=maximum_work_units,
            maximum_coordinate_memory_bytes=maximum_coordinate_memory_bytes,
            coordinate_budget=coordinate_budget,
        )
        for field in fields(data):
            setattr(self, field.name, getattr(data, field.name))
        self.atlas_id = _sphere_material_identity(self)

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
        """Radial inverse with native solve status, not an off-source membership certificate."""
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

    def require_bound(
        self,
        domain: MeshingDomain,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        coordinate_contract: SpatialCoordinateContract,
        /,
    ) -> None:
        from ..discretization._cell_geometry_validity import cell_geometry_id

        if (
            domain.domain_id,
            mesh.mesh_id,
            mesh.topology_id,
            cell_geometry_id(geometry),
            coordinate_contract.spatial_id,
        ) != (
            self.domain_id,
            self.source_mesh.mesh_id,
            self.source_topology_id,
            self.source_geometry_id,
            self.coordinate_contract.spatial_id,
        ):
            raise ValueError(
                "Sphere material atlas is stale for its source, cells, coordinate map or units."
            )
        if _sphere_material_identity(self) != self.atlas_id:
            raise ValueError(
                "Sphere material maps or proof evidence have changed since preparation."
            )
        if (
            self.coverage_status != "certified"
            or np.any(np.asarray(self.hemisphere_lower) <= 0)
            or np.any(np.asarray(self.jacobian_lower) <= 0)
        ):
            raise ValueError(
                "Sphere material atlas lacks complete nonsingular reference coverage."
            )

    def cell_row(self, cell_global_id: int, /) -> int:
        ids = np.asarray(self.cell_global_ids)
        row = int(np.searchsorted(ids, cell_global_id))
        if row == ids.size or int(ids[row]) != cell_global_id:
            raise ValueError("The material atlas has no such scientific physical cell.")
        return row

    def projective_reference_map(
        self,
        source_cell_global_id: int,
        target: SphereMaterialCellAtlas,
        target_cell_global_id: int,
        /,
        *,
        coordinate_budget: CoordinateEnclosureBudget | None = None,
    ) -> SphereProjectiveReferenceMap:
        """Exact target-cone halfplanes and target rational reference on an old cell."""
        return SphereProjectiveReferenceMap(
            self,
            source_cell_global_id,
            target,
            target_cell_global_id,
            coordinate_budget=coordinate_budget,
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


def _corner_direction(
    domain: MeshingDomain, patch: int, dimension: int, index: int, parameters: np.ndarray
) -> tuple[np.ndarray, _CornerToken]:
    path = domain.source_occurrences[dimension][index]
    source_index = domain.source_indices[dimension][index]
    if dimension == 0:
        poles = [
            use
            for loop in domain.patches[patch].loops
            for use in loop
            if isinstance(use, PatchPoleUse) and use.corner == index
        ]
        if not poles or any(
            (use.start[1] > 0) != (poles[0].start[1] > 0) for use in poles
        ):
            raise ValueError(
                "A sphere corner must be an explicitly authored pole of this occurrence."
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
                "A sphere seam must retain its actual owning curve parameter and occurrence."
            )
        use = domain.patches[owner].loops[loop][position]
        if not isinstance(use, PatchCurveUse) or not min(
            use.first, use.last
        ) <= parameters[0] <= max(use.first, use.last):
            raise ValueError("A sphere seam parameter leaves its authored source use.")
        uv = np.asarray(
            use.pcurve.evaluate(jnp.asarray(parameters[0], dtype=jnp.float64))
        ).reshape((2,))
        token = (1, source_index, path, (Fraction(float(parameters[0])),))
    elif dimension == 2:
        if index != patch or not np.all(np.isfinite(parameters)):
            raise ValueError(
                "A sphere interior corner must retain its named source-face parameters."
            )
        uv = parameters
        token = (2, source_index, path, tuple(Fraction(float(value)) for value in uv))
    else:
        raise ValueError(
            "Sphere material corners require actual source dimensions zero through two."
        )
    domain.patches[patch].surface.validate_parameter_box(np.stack((uv, uv)))
    longitude, latitude = uv
    return np.asarray(
        (
            np.cos(longitude) * np.cos(latitude),
            np.sin(longitude) * np.cos(latitude),
            np.sin(latitude),
        )
    ), token


def _prepare_sphere_material_data(
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
    maximum_cells: int = 100_000,
    maximum_work_units: int = 10_000_000,
    maximum_coordinate_memory_bytes: int = 256 * 1024**2,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> _PreparedSphereMaterialData:
    """Prepare complete regular material maps from retained accepted source strata."""
    from .._meshcore import charge_native_geometry_queries
    from ..discretization._cell_geometry_validity import cell_geometry_id
    from ..discretization._coordinate_enclosure import (
        add,
        axes,
        constant,
        coordinate_corner_images,
        coordinate_enclosure_budget,
        coordinate_expressions,
        expression_add,
        expression_bernstein_coefficients,
        expression_scale,
        outward,
        prepared_coordinate_source_bank,
        scale,
    )

    if not isinstance(domain, MeshingDomain) or not isinstance(
        coordinate_contract, SpatialCoordinateContract
    ):
        raise TypeError(
            "Sphere atlas preparation requires authoritative source and spatial coordinate owners."
        )
    if (
        any(block.cell_kind != "triangle" for block in mesh.blocks)
        or mesh.topological_dimension != 2
        or mesh.ambient_dimension != 3
    ):
        raise ValueError(
            "Sphere material atlas requires actual physical triangle cells in three dimensions."
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
            "Sphere material atlas exceeds its cell budget or has invalid fidelity/work limits."
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
            "Sphere material strata must follow actual physical-cell canonical corner ordering."
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
            "Sphere material occurrence/entity evidence must align with physical cells."
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
    ledger = (
        coordinate_enclosure_budget(maximum_work_units, maximum_coordinate_memory_bytes)
        if coordinate_budget is None
        else coordinate_budget
    )
    if not isinstance(ledger, CoordinateEnclosureBudget):
        raise TypeError(
            "coordinate_budget must be the original CoordinateEnclosureBudget."
        )
    initial_work = ledger.work_units
    ledger.reserve(0, count * 4096)
    rays = np.empty((count, 3, 3), dtype=np.float64)
    tokens: list[tuple[_CornerToken, ...]] = []
    direction_cache: dict[
        tuple[int, int, int, tuple[float, ...]], tuple[np.ndarray, _CornerToken]
    ] = {}
    active_patches = sorted(set(map(int, patches)))
    for patch in active_patches:
        if _full_sphere_frame(domain, patch) is None:
            raise ValueError(
                "This material atlas requires an authored complete sphere, not a trimmed or inferred cap."
            )
    for row in range(count):
        members = []
        for corner in range(3):
            dimension = int(dimensions[row, corner])
            path = corner_source_occurrence_paths[row][corner]
            if not 0 <= dimension <= 2:
                raise ValueError(
                    "Sphere corner source dimensions must be declared zero-through-two strata."
                )
            index = lookup[dimension].get((path, int(indices[row, corner])))
            if (
                index is None
                or domain.entity_id(dimension, index)
                != corner_source_entity_ids[row][corner]
            ):
                raise ValueError(
                    "Sphere material corner metadata is stale for its authoritative source entity."
                )
            parameter_key = tuple(
                float(value) for value in parameters[row, corner, :dimension]
            )
            key = (int(patches[row]), dimension, index, parameter_key)
            prepared_direction = direction_cache.get(key)
            if prepared_direction is None:
                charge_native_geometry_queries(1)
                prepared_direction = _corner_direction(
                    domain, int(patches[row]), dimension, index, parameters[row, corner]
                )
                direction_cache[key] = prepared_direction
            ray, token = prepared_direction
            rays[row, corner] = ray
            members.append(token)
        tokens.append(tuple(members))
    # Explicit source tokens establish equivalence; equality of geometric
    # directions below is verification, never the source-incidence authority.
    for patch in active_patches:
        rows = np.flatnonzero(patches == patch)
        values: dict[_CornerToken, np.ndarray] = {}
        chain: dict[tuple[_CornerToken, _CornerToken], int] = {}
        for row in rows:
            for corner, token in enumerate(tokens[row]):
                previous = values.setdefault(token, rays[row, corner])
                if not np.array_equal(previous, rays[row, corner]):
                    raise ValueError(
                        "One authored source point has inconsistent material direction realizations."
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
                "Actual physical material cells leave an authored source-edge chain uncovered."
            )
        flat = rays[rows].reshape((-1, 3))
        local_cells = np.arange(flat.shape[0], dtype=np.int64).reshape((-1, 3))
        if not _sphere_radial_degree_one((np.zeros(3), 1.0, 1.0), flat, local_cells):
            raise ValueError(
                "Actual sphere material cells lack exact closed radial degree-one coverage."
            )
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
        operation = source.definition if isinstance(source, PlacedSurface) else source
        equivalence = sphere_source_radius_terms(operation)
        if equivalence is None:
            raise TypeError(
                "A material source must retain a proved original sphere/offset expression."
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
        rotation = (
            np.asarray(source.rotation)
            if isinstance(source, PlacedSurface)
            else np.eye(3)
        )
        translation = (
            np.asarray(source.translation)
            if isinstance(source, PlacedSurface)
            else np.zeros(3)
        )
        low, _ = _gram_spectrum_bounds(basis)
        pose_low, _ = _gram_spectrum_bounds(rotation)
        radius = float(exact_radius)
        centers.append(np.asarray(definition.center))
        source_axes.append(basis)
        radii.append(radius)
        rotations.append(rotation)
        translations.append(translation)
        norms.append(_sphere_radial_source_norm(source))
        singulars.append(
            _float_enclosure(
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
    hemisphere, maximum_norm, determinants, jacobian, normalization, hessians = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    slots = {patch: slot for slot, patch in enumerate(active_patches)}
    for row in range(count):
        radial = _sphere_radial_triangle_bounds(rays[row], ledger)
        if radial is None:
            raise ValueError(
                "The complete sphere reference triangle lacks a positive hemisphere enclosure."
            )
        determinant, floor, ceiling, unit_error = radial
        ledger.retain_basis((determinant,))
        sign = -1 if domain.patches[int(patches[row])].reversed else 1
        if determinant * sign <= 0:
            raise ValueError(
                "Sphere material reference reverses its authored source orientation."
            )
        first = interval_subtract(
            (rays[row, 1], rays[row, 1]), (rays[row, 0], rays[row, 0])
        )
        second = interval_subtract(
            (rays[row, 2], rays[row, 2]), (rays[row, 0], rays[row, 0])
        )
        slot = slots[int(patches[row])]
        radial_defect = max(1 - Fraction(floor), Fraction(ceiling) - 1, Fraction(0))
        chord = _float_enclosure(Fraction(norms[slot]) * (radial_defect + unit_error))[1]
        jacobian_floor = _float_enclosure(
            Fraction(singulars[slot]) ** 2 * abs(determinant) / Fraction(ceiling) ** 3
        )[0]
        first_norm, second_norm = (
            float(_interval_vector_norm(first)[1]),
            float(_interval_vector_norm(second)[1]),
        )
        factor = Fraction(6) * Fraction(norms[slot]) / Fraction(floor) ** 2
        hemisphere.append(floor)
        maximum_norm.append(ceiling)
        determinants.append(determinant)
        jacobian.append(jacobian_floor)
        normalization.append(chord)
        hessians.append(
            (
                _float_enclosure(factor * Fraction(first_norm) ** 2)[1],
                _float_enclosure(factor * Fraction(first_norm) * Fraction(second_norm))[
                    1
                ],
                _float_enclosure(factor * Fraction(second_norm) ** 2)[1],
            )
        )
    elements, routes, coordinate_values = geometry.resolve(mesh)
    # The caller's original ledger remains the sole cumulative coefficient owner.
    ledger.reserve(
        0,
        coordinate_values.size * np.dtype(np.float64).itemsize
        + sum(route.size * np.dtype(np.int64).itemsize for route in routes),
    )
    with ledger.activate():
        coordinate_bank = prepared_coordinate_source_bank(geometry)
    polynomial_bounds, corner_bounds = [], []
    offset = 0
    with ledger.activate():
        variables = axes(2)
        for element, route in zip(elements, routes, strict=True):
            for local in np.asarray(route, dtype=np.int64):
                with ledger.temporary_scope():
                    ledger.reserve(local.size * 3)
                    expressions = coordinate_expressions(
                        element, tuple(coordinate_bank[int(index)] for index in local)
                    )
                    if expressions is None:
                        raise ValueError(
                            "The actual coordinate owner has no exact full-map enclosure."
                        )
                    points = coordinate_corner_images(
                        element, tuple(coordinate_bank[int(index)] for index in local)
                    )
                    if points is None:
                        raise ValueError(
                            "The actual coordinate owner has no exact full-source corner enclosure."
                        )
                    defect = Fraction(0)
                    for component, expression in enumerate(expressions):
                        corner_values = tuple(point[component] for point in points)
                        chord = add(
                            constant(corner_values[0], 2),
                            add(
                                scale(variables[0], corner_values[1] - corner_values[0]),
                                scale(variables[1], corner_values[2] - corner_values[0]),
                            ),
                        )
                        defect += max(
                            abs(value)
                            for value in expression_bernstein_coefficients(
                                expression_add(expression, expression_scale(chord, -1)),
                                "simplex",
                                2,
                            )
                        )
                    source = domain.patches[int(patches[offset])].surface
                    if (
                        sphere_source_equivalence(
                            source.definition
                            if isinstance(source, PlacedSurface)
                            else source
                        )
                        is None
                    ):
                        raise TypeError(
                            "A material source lost its original sphere-operation proof."
                        )
                    low, high = _source_image_enclosure(source, rays[offset])
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
                    polynomial_bounds.append(outward(defect, np.inf))
                    corner_bounds.append(outward(error, np.inf))
                    offset += 1
    work = ledger.work_units - initial_work
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
            "The actual old map misses the requested whole-cell radial source fidelity or regularity."
        )
    order = np.argsort(ids, kind="stable")
    geometry_id = cell_geometry_id(geometry)
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
        coordinate_retained_basis_bytes=ledger.retained_basis_bytes,
        coordinate_storage_upper_bytes=ledger.peak_bytes_upper,
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


def prepare_sphere_material_atlas(
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
    maximum_cells: int = 100_000,
    maximum_work_units: int = 10_000_000,
    maximum_coordinate_memory_bytes: int = 256 * 1024**2,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> SphereMaterialCellAtlas:
    """Admit a material owner only through actual source-bound preparation."""
    return SphereMaterialCellAtlas(
        domain,
        mesh,
        geometry,
        coordinate_contract,
        cell_patches=cell_patches,
        corner_source_dimensions=corner_source_dimensions,
        corner_source_indices=corner_source_indices,
        corner_source_parameters=corner_source_parameters,
        corner_source_occurrence_paths=corner_source_occurrence_paths,
        corner_source_entity_ids=corner_source_entity_ids,
        maximum_fidelity=maximum_fidelity,
        maximum_cells=maximum_cells,
        maximum_work_units=maximum_work_units,
        maximum_coordinate_memory_bytes=maximum_coordinate_memory_bytes,
        coordinate_budget=coordinate_budget,
    )


def sphere_material_atlas_from_result(
    domain: MeshingDomain,
    result: CellMeshingResult,
    /,
    *,
    maximum_fidelity: float,
    maximum_cells: int = 100_000,
    maximum_work_units: int = 10_000_000,
    maximum_coordinate_memory_bytes: int = 256 * 1024**2,
    coordinate_budget: CoordinateEnclosureBudget | None = None,
) -> SphereMaterialCellAtlas:
    """Consume genuine accepted publication metadata; never parse entity-ID strings."""
    from ..meshing._association import GeometryAssociationKind

    if (
        not result.audit.passed
        or result.certification is None
        or not result.certification.passed
    ):
        raise ValueError(
            "Sphere material preparation requires a decided accepted physical publication."
        )
    vertex_set, face_set = result.mesh.entity_set(0), result.mesh.entity_set(2)
    associations = [
        value
        for value in result.associations
        if value.association_kind
        in (GeometryAssociationKind.SURFACE, GeometryAssociationKind.BREP)
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
            "Accepted sphere cells and vertices lack complete declared source strata."
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
            (face.source_occurrence_paths[row], int(np.asarray(face.source_indices)[row]))
        )
        if (
            patch is None
            or int(np.asarray(face.source_dimensions)[row]) != 2
            or face.source_entity_ids[row] != domain.entity_id(2, patch)
        ):
            raise ValueError(
                "Accepted sphere cell identity is stale for its authoritative source occurrence."
            )
        patches.append(patch)
    corner_rows = vertex_rows[cells]
    return prepare_sphere_material_atlas(
        domain,
        result.mesh,
        result.geometry,
        result.coordinate_contract,
        cell_patches=np.asarray(patches, dtype=np.int64),
        corner_source_dimensions=np.asarray(vertex.source_dimensions)[corner_rows],
        corner_source_indices=np.asarray(vertex.source_indices)[corner_rows],
        corner_source_parameters=np.asarray(vertex.parameters)[corner_rows],
        corner_source_occurrence_paths=tuple(
            tuple(vertex.source_occurrence_paths[row] for row in corners)
            for corners in corner_rows
        ),
        corner_source_entity_ids=tuple(
            tuple(vertex.source_entity_ids[row] for row in corners)
            for corners in corner_rows
        ),
        maximum_fidelity=maximum_fidelity,
        maximum_cells=maximum_cells,
        maximum_work_units=maximum_work_units,
        maximum_coordinate_memory_bytes=maximum_coordinate_memory_bytes,
        coordinate_budget=coordinate_budget,
    )


__all__ = [
    "SphereMaterialCellAtlas",
    "SphereMaterialInverse",
    "SphereProjectiveReferenceMap",
    "SphereProjectiveTriangleBounds",
    "prepare_sphere_material_atlas",
    "sphere_material_atlas_from_result",
]


def _sphere_radial_triangle_bounds(
    directions: np.ndarray,
    coordinate_budget: CoordinateEnclosureBudget,
    /,
) -> tuple[Fraction, float, float, Fraction] | None:
    """The complete real cone/hemisphere enclosure shared by admission and edits."""
    coordinate_budget.reserve(9)
    matrix = _fraction_rows(directions.T)
    with coordinate_budget.temporary_scope():
        prepared = prepare_exact_small_linear_actions(
            matrix, ((), (), ()), coordinate_budget=coordinate_budget
        )
    if not prepared.successful:
        return None
    first = interval_subtract(
        (directions[1], directions[1]), (directions[0], directions[0])
    )
    second = interval_subtract(
        (directions[2], directions[2]), (directions[0], directions[0])
    )
    normal_upper = float(_interval_vector_norm(_interval_vector_cross(first, second))[1])
    if normal_upper <= 0:
        return None
    floor = _float_enclosure(abs(prepared.determinant) / Fraction(normal_upper))[0]
    lows, highs = _interval_vector_norm((directions, directions))
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
    return prepared.determinant, floor, ceiling, unit_error


def _sphere_radial_source_norm(source: AbstractSurfacePatch, /) -> float:
    """Original source/placement norm used by the owning normalization bound."""
    operation = source.definition if isinstance(source, PlacedSurface) else source
    equivalence = sphere_source_radius_terms(operation)
    if equivalence is None:
        raise TypeError(
            "Sphere normalization requires its original sphere/offset source."
        )
    definition, terms = equivalence
    basis = np.column_stack(
        (
            np.asarray(definition.first_axis),
            np.asarray(definition.second_axis),
            np.asarray(definition.axis),
        )
    )
    rotation = (
        np.asarray(source.rotation)
        if isinstance(source, PlacedSurface)
        else np.eye(3, dtype=np.float64)
    )
    return _float_enclosure(
        abs(sum(terms, Fraction(0)))
        * Fraction(_matrix_norm_upper(basis))
        * Fraction(_matrix_norm_upper(rotation))
    )[1]
