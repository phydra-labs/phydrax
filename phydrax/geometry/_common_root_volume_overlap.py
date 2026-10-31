from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from fractions import Fraction
from itertools import combinations, permutations
from math import factorial, gcd, lcm
from typing import Any, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .. import _meshcore
from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._polynomial._cubature import cubature_rule_data
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import (
    CellGeometrySpec,
    coordinate_effective_basis,
    PerCellRestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    certify_cell_geometry_validity,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import (
    affine_map,
    axes,
    bernstein_coefficients,
    BernsteinDomain,
    compose_expression,
    constant,
    derivative,
    evaluate_expression,
    ExactScalar,
    Expression,
    expression_equal,
    ExpressionInput,
    linear_combinations,
    multiply,
    polynomial_bounds,
    power,
    RationalPolynomial,
    scale,
    sum_polynomials,
)
from ..discretization._exact_algebraic_geometry import (
    coefficient_endpoint,
    RadicalElement,
)
from ..discretization._nested_reference_types import (
    exact_reference_cell_vertices,
    reference_chart_arguments_identity,
    ReferenceCellMapWitnesses,
    validate_reference_cell_map_witnesses,
)
from ..discretization._reference_cell import reference_cell_topology
from ..discretization.fem._reference import FiniteElementSpec
from ..linalg._exact import determinant_exact, solve_exact
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from . import _supermesh as supermesh_owner
from ._global_embedding import certify_global_embedding, GlobalEmbeddingCertificate


Point = tuple[Fraction, Fraction, Fraction]
Tet = tuple[Point, Point, Point, Point]
_Planes = tuple[tuple[int, int, int, int], ...]


def _as_point(vertices: Sequence[Fraction], /) -> Point:
    if len(vertices) != 3:
        raise ValueError("A reference point must have exactly three coordinates.")
    return vertices[0], vertices[1], vertices[2]


def _as_tetrahedron(vertices: Sequence[Sequence[Fraction]], /) -> Tet:
    if len(vertices) != 4 or any(len(point) != 3 for point in vertices):
        raise ValueError("A reference tetrahedron must contain four 3D vertices.")
    return (
        _as_point(vertices[0]),
        _as_point(vertices[1]),
        _as_point(vertices[2]),
        _as_point(vertices[3]),
    )


def _as_chart_expression(expression: ExpressionInput, /) -> Expression:
    if isinstance(expression, RationalPolynomial):
        return expression
    if isinstance(expression, dict):
        return expression
    return dict(expression)


def _as_chart_arguments(
    arguments: Sequence[ExpressionInput], /
) -> tuple[Expression, Expression, Expression]:
    if len(arguments) != 3:
        raise ValueError("A three-dimensional reference chart needs three outputs.")
    return (
        _as_chart_expression(arguments[0]),
        _as_chart_expression(arguments[1]),
        _as_chart_expression(arguments[2]),
    )


class CommonRootOverlapError(ValueError):
    """Root-reference intersection, coverage, or finite-resolution integration failure."""

    def __init__(
        self,
        message: str,
        status: supermesh_owner.CommonRefinementStatus,
        /,
        *,
        source_reference_defects: Sequence[Fraction] = (),
        target_reference_defects: Sequence[Fraction] = (),
        unmatched_source_rows: Sequence[int] = (),
        unmatched_target_rows: Sequence[int] = (),
        overlapping_source_rows: Sequence[int] = (),
        overlapping_target_rows: Sequence[int] = (),
        integration_error: float | None = None,
        integration_tolerance: float | None = None,
        unresolved_reference_tetrahedron: Sequence[Sequence[Fraction]] = (),
    ) -> None:
        super().__init__(message)
        self.status = status
        self.source_reference_defects = tuple(source_reference_defects)
        self.target_reference_defects = tuple(target_reference_defects)
        self.unmatched_source_rows = tuple(int(value) for value in unmatched_source_rows)
        self.unmatched_target_rows = tuple(int(value) for value in unmatched_target_rows)
        self.overlapping_source_rows = tuple(
            int(value) for value in overlapping_source_rows
        )
        self.overlapping_target_rows = tuple(
            int(value) for value in overlapping_target_rows
        )
        self.integration_error = integration_error
        self.integration_tolerance = integration_tolerance
        self.unresolved_reference_tetrahedron = tuple(
            tuple(str(value) for value in point)
            for point in unresolved_reference_tetrahedron
        )


def _triangular_injectivity_witness(
    expressions: Sequence[Expression], /
) -> tuple[str, ...] | None:
    """Prove injectivity along fibers of two independent affine coordinates."""
    for physical_order in permutations(range(3)):
        first, second, third = (expressions[index] for index in physical_order)
        if (
            isinstance(first, RationalPolynomial)
            or not isinstance(first, Mapping)
            or isinstance(second, RationalPolynomial)
            or not isinstance(second, Mapping)
            or any(sum(exponent) > 1 for exponent in (*first, *second))
        ):
            continue
        rows = []
        for value in (first, second):
            coefficients = []
            for axis in range(3):
                exponent = tuple(int(index == axis) for index in range(3))
                coefficient = value.get(exponent, Fraction(0))
                if isinstance(coefficient, Fraction):
                    coefficients.append(coefficient)
                elif type(coefficient) is int:
                    coefficients.append(Fraction(coefficient))
                else:
                    break
            if len(coefficients) != 3:
                rows = []
                break
            rows.append(tuple(coefficients))
        if len(rows) != 2:
            continue
        first_row, second_row = rows
        null_direction = (
            first_row[1] * second_row[2] - first_row[2] * second_row[1],
            first_row[2] * second_row[0] - first_row[0] * second_row[2],
            first_row[0] * second_row[1] - first_row[1] * second_row[0],
        )
        if not any(null_direction):
            continue
        monotone_derivative = sum_polynomials(
            tuple(
                scale(derivative(third, axis), null_direction[axis]) for axis in range(3)
            ),
            3,
        )
        lower, upper = polynomial_bounds(monotone_derivative, "simplex", 3)
        if lower > 0.0 or upper < 0.0:
            return (
                "triangular-polynomial-injectivity",
                repr(physical_order),
                repr(null_direction),
                format(lower, ".17g"),
                format(upper, ".17g"),
            )
    return None


def _root_embedding_proof(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    expressions: Sequence[tuple[Expression, Expression, Expression]],
    /,
) -> tuple[CellValidityCertificate, GlobalEmbeddingCertificate | None, tuple[str, ...]]:
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    if validity.certified_valid_count != sum(block.cell_count for block in mesh.blocks):
        raise ValueError(
            "Common-root coordinates require certified positive Jacobians on every root cell."
        )
    injectivity = (
        _triangular_injectivity_witness(expressions[0])
        if sum(block.cell_count for block in mesh.blocks) == 1
        else None
    )
    if injectivity is not None:
        return validity, None, injectivity
    embedding = certify_global_embedding(mesh, geometry, validity)
    if embedding.status != "certified":
        raise ValueError(
            "A common curved root needs a computed global-embedding proof or a proved triangular injectivity map."
        )
    return validity, embedding, ("computed-global-embedding", embedding.certificate_id)


class CommonRootVolumeWitness(StrictModule, NonTrainableState):
    """Exact endpoint-to-root reference maps over one authoritative curved root."""

    root_mesh: CellMesh
    root_geometry: CellGeometrySpec
    root_validity: CellValidityCertificate
    root_embedding: GlobalEmbeddingCertificate | None
    root_injectivity_witness: tuple[str, ...] = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    target_mesh_id: str = eqx.field(static=True)
    root_mesh_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    root_geometry_id: str = eqx.field(static=True)
    source_block_names: tuple[str, ...] = eqx.field(static=True)
    source_reference_maps: tuple[ReferenceCellMapWitnesses, ...]
    target_block_names: tuple[str, ...] = eqx.field(static=True)
    target_reference_maps: tuple[ReferenceCellMapWitnesses, ...]
    source_chart_witnesses: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    target_chart_witnesses: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    witness_id: str = eqx.field(static=True)

    def __init__(
        self,
        root_mesh: CellMesh,
        root_geometry: CellGeometrySpec,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        source_reference_maps: Mapping[str, ReferenceCellMapWitnesses],
        target_reference_maps: Mapping[str, ReferenceCellMapWitnesses],
        /,
    ) -> None:
        if not all(
            isinstance(value, CellMesh) for value in (root_mesh, source_mesh, target_mesh)
        ):
            raise TypeError("Common-root overlap requires canonical CellMesh owners.")
        if not all(
            isinstance(value, CellGeometrySpec)
            for value in (root_geometry, source_geometry, target_geometry)
        ):
            raise TypeError(
                "Common-root overlap requires canonical CellGeometrySpec owners."
            )
        if (
            root_mesh.topological_dimension != 3
            or root_mesh.ambient_dimension != 3
            or source_mesh.topological_dimension != 3
            or target_mesh.topological_dimension != 3
            or source_mesh.ambient_dimension != 3
            or target_mesh.ambient_dimension != 3
            or any(
                block.cell_kind != "tetrahedron"
                for mesh in (root_mesh, source_mesh, target_mesh)
                for block in mesh.blocks
            )
        ):
            raise ValueError(
                "Curved common-root volume overlap currently requires tetrahedral meshes."
            )
        source_maps = _canonical_witness_bank(
            source_mesh, source_reference_maps, "source"
        )
        target_maps = _canonical_witness_bank(
            target_mesh, target_reference_maps, "target"
        )
        root_cells = _root_cell_rows(root_mesh)
        root_elements, root_routes, _ = root_geometry.resolve(root_mesh)
        source_elements, source_routes, _ = source_geometry.resolve(source_mesh)
        target_elements, target_routes, _ = target_geometry.resolve(target_mesh)
        _require_exact_source_bank(root_geometry)
        _require_exact_source_bank(source_geometry)
        _require_exact_source_bank(target_geometry)
        source_exprs = _coordinate_expression_rows(
            source_mesh, source_geometry, source_elements, source_routes
        )
        target_exprs = _coordinate_expression_rows(
            target_mesh, target_geometry, target_elements, target_routes
        )
        root_exprs = _coordinate_expression_rows(
            root_mesh, root_geometry, root_elements, root_routes
        )
        root_validity, root_embedding, injectivity_witness = _root_embedding_proof(
            root_mesh, root_geometry, root_exprs
        )
        source_chart_witnesses = _require_root_restrictions(
            root_mesh,
            root_geometry,
            root_cells,
            root_exprs,
            source_mesh,
            source_geometry,
            source_maps,
            source_exprs,
        )
        target_chart_witnesses = _require_root_restrictions(
            root_mesh,
            root_geometry,
            root_cells,
            root_exprs,
            target_mesh,
            target_geometry,
            target_maps,
            target_exprs,
        )
        self.root_mesh = root_mesh
        self.root_geometry = root_geometry
        self.root_validity = root_validity
        self.root_embedding = root_embedding
        self.root_injectivity_witness = injectivity_witness
        self.source_mesh_id = source_mesh.mesh_id
        self.target_mesh_id = target_mesh.mesh_id
        self.root_mesh_id = root_mesh.mesh_id
        self.source_geometry_id = cell_geometry_id(source_geometry)
        self.target_geometry_id = cell_geometry_id(target_geometry)
        self.root_geometry_id = cell_geometry_id(root_geometry)
        self.source_block_names = tuple(block.name for block in source_mesh.blocks)
        self.source_reference_maps = source_maps
        self.target_block_names = tuple(block.name for block in target_mesh.blocks)
        self.target_reference_maps = target_maps
        self.source_chart_witnesses = source_chart_witnesses
        self.target_chart_witnesses = target_chart_witnesses
        self.witness_id = canonical_fingerprint(
            {
                "kind": "common-root-volume-witness",
                "root_mesh": root_mesh.mesh_id,
                "root_geometry": self.root_geometry_id,
                "source_mesh": self.source_mesh_id,
                "source_geometry": self.source_geometry_id,
                "target_mesh": self.target_mesh_id,
                "target_geometry": self.target_geometry_id,
                "source_maps": _witness_identity(self.source_block_names, source_maps),
                "target_maps": _witness_identity(self.target_block_names, target_maps),
                "root_validity": root_validity.certificate_id,
                "root_embedding": None
                if root_embedding is None
                else root_embedding.certificate_id,
                "injectivity_witness": injectivity_witness,
                "source_chart_images": source_chart_witnesses,
                "target_chart_images": target_chart_witnesses,
            }
        )

    def revalidate(
        self,
        root_mesh: CellMesh,
        root_geometry: CellGeometrySpec,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
    ) -> CommonRootVolumeWitness:
        """Rebuild this witness from independently restored geometry owners."""
        if self.source_block_names != tuple(block.name for block in source_mesh.blocks):
            raise ValueError(
                "Stored source reference witnesses do not match restored blocks."
            )
        if self.target_block_names != tuple(block.name for block in target_mesh.blocks):
            raise ValueError(
                "Stored target reference witnesses do not match restored blocks."
            )
        rebuilt = CommonRootVolumeWitness(
            root_mesh,
            root_geometry,
            source_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            dict(zip(self.source_block_names, self.source_reference_maps, strict=True)),
            dict(zip(self.target_block_names, self.target_reference_maps, strict=True)),
        )
        if rebuilt.witness_id != self.witness_id:
            raise ValueError(
                "Stored common-root witness does not match its recomputed geometry proof."
            )
        return rebuilt


class MaterialReferenceVolumeWitness(StrictModule, NonTrainableState):
    """Exact endpoint charts into one root reference mesh for moving volumes.

    Unlike :class:`CommonRootVolumeWitness`, endpoint geometries are independent
    physical deformations of the shared reference cells.  The root geometry
    supplies the reference-volume measure; it is not required to equal either
    endpoint physical map.
    """

    root_mesh: CellMesh
    root_geometry: CellGeometrySpec
    root_validity: CellValidityCertificate
    root_embedding: GlobalEmbeddingCertificate | None
    root_injectivity_witness: tuple[str, ...] = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    target_mesh_id: str = eqx.field(static=True)
    root_mesh_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    root_geometry_id: str = eqx.field(static=True)
    source_block_names: tuple[str, ...] = eqx.field(static=True)
    source_reference_maps: tuple[ReferenceCellMapWitnesses, ...]
    target_block_names: tuple[str, ...] = eqx.field(static=True)
    target_reference_maps: tuple[ReferenceCellMapWitnesses, ...]
    source_chart_witnesses: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    target_chart_witnesses: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    source_validity: CellValidityCertificate
    target_validity: CellValidityCertificate
    source_coordinate_bounds: tuple[tuple[float, ...], tuple[float, ...]] = eqx.field(
        static=True
    )
    target_coordinate_bounds: tuple[tuple[float, ...], tuple[float, ...]] = eqx.field(
        static=True
    )
    maximum_displacement_bound: float = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)
    witness_id: str = eqx.field(static=True)

    def __init__(
        self,
        root_mesh: CellMesh,
        root_geometry: CellGeometrySpec,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        source_reference_maps: Mapping[str, ReferenceCellMapWitnesses],
        target_reference_maps: Mapping[str, ReferenceCellMapWitnesses],
        /,
    ) -> None:
        meshes = (root_mesh, source_mesh, target_mesh)
        geometries = (root_geometry, source_geometry, target_geometry)
        if not all(isinstance(value, CellMesh) for value in meshes):
            raise TypeError(
                "Material-reference overlap requires canonical CellMesh owners."
            )
        if not all(isinstance(value, CellGeometrySpec) for value in geometries):
            raise TypeError(
                "Material-reference overlap requires canonical CellGeometrySpec owners."
            )
        if any(
            mesh.topological_dimension != 3
            or mesh.ambient_dimension != 3
            or any(block.cell_kind != "tetrahedron" for block in mesh.blocks)
            for mesh in meshes
        ):
            raise ValueError(
                "Material-reference volume overlap currently requires tetrahedral meshes."
            )
        source_maps = _canonical_witness_bank(
            source_mesh, source_reference_maps, "source"
        )
        target_maps = _canonical_witness_bank(
            target_mesh, target_reference_maps, "target"
        )
        root_rows = _root_cell_rows(root_mesh)
        root_elements, root_routes, _ = root_geometry.resolve(root_mesh)
        source_elements, source_routes, _ = source_geometry.resolve(source_mesh)
        target_elements, target_routes, _ = target_geometry.resolve(target_mesh)
        for geometry in geometries:
            _require_exact_source_bank(geometry)
        root_expressions = _coordinate_expression_rows(
            root_mesh, root_geometry, root_elements, root_routes
        )
        source_expressions = _coordinate_expression_rows(
            source_mesh, source_geometry, source_elements, source_routes
        )
        target_expressions = _coordinate_expression_rows(
            target_mesh, target_geometry, target_elements, target_routes
        )
        root_validity, root_embedding, root_injectivity = _root_embedding_proof(
            root_mesh, root_geometry, root_expressions
        )
        source_validity = certify_cell_geometry_validity(
            source_geometry, mesh=source_mesh
        )
        target_validity = certify_cell_geometry_validity(
            target_geometry, mesh=target_mesh
        )
        if source_validity.certified_valid_count != sum(
            block.cell_count for block in source_mesh.blocks
        ) or target_validity.certified_valid_count != sum(
            block.cell_count for block in target_mesh.blocks
        ):
            raise ValueError(
                "Material-reference endpoint deformations require certified positive Jacobians."
            )
        source_chart_witnesses = _validate_material_reference_maps(
            root_mesh,
            root_rows,
            source_mesh,
            source_maps,
        )
        target_chart_witnesses = _validate_material_reference_maps(
            root_mesh,
            root_rows,
            target_mesh,
            target_maps,
        )
        reference_id = canonical_fingerprint(
            {
                "kind": "material-reference-domain",
                "root_mesh": root_mesh.mesh_id,
                "root_geometry": cell_geometry_id(root_geometry),
            }
        )
        source_coordinate_bounds = _physical_coordinate_bounds(source_expressions)
        target_coordinate_bounds = _physical_coordinate_bounds(target_expressions)
        source_lower = np.asarray(source_coordinate_bounds[0], dtype=np.float64)
        source_upper = np.asarray(source_coordinate_bounds[1], dtype=np.float64)
        target_lower = np.asarray(target_coordinate_bounds[0], dtype=np.float64)
        target_upper = np.asarray(target_coordinate_bounds[1], dtype=np.float64)
        displacement_axes = tuple(
            max(
                abs(
                    Fraction(float(target_upper[axis]))
                    - Fraction(float(source_lower[axis]))
                ),
                abs(
                    Fraction(float(source_upper[axis]))
                    - Fraction(float(target_lower[axis]))
                ),
            )
            for axis in range(3)
        )
        displacement_squared = sum(
            (value * value for value in displacement_axes), Fraction(0)
        )
        maximum_displacement_bound = float(
            coefficient_endpoint(
                fraction_sqrt_interval(displacement_squared)[1], upper=True
            )
        )
        if not np.isfinite(maximum_displacement_bound):
            raise ValueError(
                "Material endpoint coordinate bounds do not yield a finite displacement enclosure."
            )
        self.root_mesh = root_mesh
        self.root_geometry = root_geometry
        self.root_validity = root_validity
        self.root_embedding = root_embedding
        self.root_injectivity_witness = root_injectivity
        self.source_mesh_id = source_mesh.mesh_id
        self.target_mesh_id = target_mesh.mesh_id
        self.reference_id = reference_id
        self.root_mesh_id = root_mesh.mesh_id
        self.source_geometry_id = cell_geometry_id(source_geometry)
        self.target_geometry_id = cell_geometry_id(target_geometry)
        self.root_geometry_id = cell_geometry_id(root_geometry)
        self.source_block_names = tuple(block.name for block in source_mesh.blocks)
        self.source_reference_maps = source_maps
        self.target_block_names = tuple(block.name for block in target_mesh.blocks)
        self.target_reference_maps = target_maps
        self.source_chart_witnesses = source_chart_witnesses
        self.target_chart_witnesses = target_chart_witnesses
        self.source_validity = source_validity
        self.target_validity = target_validity
        self.source_coordinate_bounds = source_coordinate_bounds
        self.target_coordinate_bounds = target_coordinate_bounds
        self.maximum_displacement_bound = maximum_displacement_bound
        self.witness_id = canonical_fingerprint(
            {
                "kind": "material-reference-volume-witness",
                "reference": reference_id,
                "root_mesh": root_mesh.mesh_id,
                "root_geometry": self.root_geometry_id,
                "source_mesh": self.source_mesh_id,
                "source_geometry": self.source_geometry_id,
                "target_mesh": self.target_mesh_id,
                "target_geometry": self.target_geometry_id,
                "source_maps": _witness_identity(self.source_block_names, source_maps),
                "target_maps": _witness_identity(self.target_block_names, target_maps),
                "root_validity": root_validity.certificate_id,
                "root_embedding": (
                    None if root_embedding is None else root_embedding.certificate_id
                ),
                "root_injectivity": root_injectivity,
                "source_validity": source_validity.certificate_id,
                "target_validity": target_validity.certificate_id,
                "source_chart_images": source_chart_witnesses,
                "target_chart_images": target_chart_witnesses,
                "source_coordinate_bounds": source_coordinate_bounds,
                "target_coordinate_bounds": target_coordinate_bounds,
                "maximum_displacement_bound": maximum_displacement_bound,
            }
        )

    def revalidate(
        self,
        root_mesh: CellMesh,
        root_geometry: CellGeometrySpec,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
    ) -> MaterialReferenceVolumeWitness:
        if self.source_block_names != tuple(block.name for block in source_mesh.blocks):
            raise ValueError(
                "Stored source reference witnesses do not match restored blocks."
            )
        if self.target_block_names != tuple(block.name for block in target_mesh.blocks):
            raise ValueError(
                "Stored target reference witnesses do not match restored blocks."
            )
        rebuilt = MaterialReferenceVolumeWitness(
            root_mesh,
            root_geometry,
            source_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            dict(zip(self.source_block_names, self.source_reference_maps, strict=True)),
            dict(zip(self.target_block_names, self.target_reference_maps, strict=True)),
        )
        if rebuilt.witness_id != self.witness_id:
            raise ValueError(
                "Stored material-reference witness does not match its recomputed geometry proof."
            )
        return rebuilt


class CurvedOverlapIntegrationPolicy(StrictModule, NonTrainableState):
    """Relative and absolute physical-integral error requirements."""

    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative_tolerance: float = 1.0e-10,
        absolute_tolerance: float = 1.0e-14,
    ) -> None:
        relative = float(relative_tolerance)
        absolute = float(absolute_tolerance)
        if not np.isfinite(relative) or not 0.0 < relative < 1.0:
            raise ValueError("relative_tolerance must be finite and in (0, 1).")
        if not np.isfinite(absolute) or absolute < 0.0:
            raise ValueError("absolute_tolerance must be finite and nonnegative.")
        if absolute == 0.0 and relative == 0.0:
            raise ValueError("At least one integration tolerance must be positive.")
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.policy_id = canonical_fingerprint(
            {
                "kind": "curved-overlap-integration-policy",
                "relative_tolerance": relative,
                "absolute_tolerance": absolute,
                "rule": "native-simplex-cubature-degree-8-with-bernstein-integral-enclosure",
            }
        )


class CurvedOverlapIntegrationEvidence(StrictModule, NonTrainableState):
    """Per-overlap and endpoint integral errors and exact reference coverage."""

    overlap_errors: Array
    source_errors: Array
    target_errors: Array
    source_coordinate_bounds: Array
    target_coordinate_bounds: Array
    source_reference_defects: tuple[str, ...] = eqx.field(static=True)
    target_reference_defects: tuple[str, ...] = eqx.field(static=True)
    unmatched_source_rows: Array
    unmatched_target_rows: Array
    root_validity_id: str = eqx.field(static=True)
    witness_id: str = eqx.field(static=True)
    integration_policy: CurvedOverlapIntegrationPolicy
    exact_candidate_count: int = eqx.field(static=True)
    positive_overlap_count: int = eqx.field(static=True)
    exact_polynomial_integral_count: int = eqx.field(static=True)
    enclosed_integral_count: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        overlap_errors: Any,
        source_errors: Any,
        target_errors: Any,
        source_coordinate_bounds: Any,
        target_coordinate_bounds: Any,
        source_reference_defects: Sequence[Fraction],
        target_reference_defects: Sequence[Fraction],
        unmatched_source_rows: Any,
        unmatched_target_rows: Any,
        root_validity_id: str,
        witness_id: str,
        integration_policy: CurvedOverlapIntegrationPolicy,
        exact_candidate_count: int,
        positive_overlap_count: int,
        exact_polynomial_integral_count: int,
        enclosed_integral_count: int,
        /,
    ) -> None:
        overlap = np.asarray(overlap_errors, dtype=np.float64)
        source = np.asarray(source_errors, dtype=np.float64)
        target = np.asarray(target_errors, dtype=np.float64)
        source_bounds = np.asarray(source_coordinate_bounds, dtype=np.float64)
        target_bounds = np.asarray(target_coordinate_bounds, dtype=np.float64)
        unmatched_source = np.asarray(unmatched_source_rows, dtype=np.int64)
        unmatched_target = np.asarray(unmatched_target_rows, dtype=np.int64)
        if (
            overlap.ndim != 2
            or source.ndim != 2
            or target.ndim != 2
            or source.shape[0] != len(source_reference_defects)
            or target.shape[0] != len(target_reference_defects)
            or overlap.shape[1] != source.shape[1]
            or source.shape[1] != target.shape[1]
            or source_bounds.shape != (source.shape[0], 2, 3)
            or target_bounds.shape != (target.shape[0], 2, 3)
            or unmatched_source.ndim != 1
            or unmatched_target.ndim != 1
            or np.any(overlap < 0.0)
            or np.any(source < 0.0)
            or np.any(target < 0.0)
            or np.any(source_bounds[:, 0, :] > source_bounds[:, 1, :])
            or np.any(target_bounds[:, 0, :] > target_bounds[:, 1, :])
            or not all(
                np.all(np.isfinite(values))
                for values in (overlap, source, target, source_bounds, target_bounds)
            )
            or not root_validity_id
            or not witness_id
            or not isinstance(integration_policy, CurvedOverlapIntegrationPolicy)
            or min(
                exact_candidate_count,
                positive_overlap_count,
                exact_polynomial_integral_count,
                enclosed_integral_count,
            )
            < 0
        ):
            raise ValueError("Curved overlap integration evidence is malformed.")
        source_defects = tuple(str(value) for value in source_reference_defects)
        target_defects = tuple(str(value) for value in target_reference_defects)
        self.overlap_errors = jnp.asarray(overlap)
        self.source_errors = jnp.asarray(source)
        self.target_errors = jnp.asarray(target)
        self.source_coordinate_bounds = jnp.asarray(source_bounds)
        self.target_coordinate_bounds = jnp.asarray(target_bounds)
        self.source_reference_defects = source_defects
        self.target_reference_defects = target_defects
        self.unmatched_source_rows = jnp.asarray(unmatched_source)
        self.unmatched_target_rows = jnp.asarray(unmatched_target)
        self.root_validity_id = root_validity_id
        self.witness_id = witness_id
        self.integration_policy = integration_policy
        self.exact_candidate_count = int(exact_candidate_count)
        self.positive_overlap_count = int(positive_overlap_count)
        self.exact_polynomial_integral_count = int(exact_polynomial_integral_count)
        self.enclosed_integral_count = int(enclosed_integral_count)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "curved-overlap-integration-evidence",
                "overlap_errors": array_tree_fingerprint(overlap),
                "source_errors": array_tree_fingerprint(source),
                "target_errors": array_tree_fingerprint(target),
                "source_coordinate_bounds": array_tree_fingerprint(source_bounds),
                "target_coordinate_bounds": array_tree_fingerprint(target_bounds),
                "source_reference_defects": source_defects,
                "target_reference_defects": target_defects,
                "unmatched_source_rows": array_tree_fingerprint(unmatched_source),
                "unmatched_target_rows": array_tree_fingerprint(unmatched_target),
                "root_validity": root_validity_id,
                "witness": witness_id,
                "policy": integration_policy.policy_id,
                "counts": (
                    exact_candidate_count,
                    positive_overlap_count,
                    exact_polynomial_integral_count,
                    enclosed_integral_count,
                ),
            }
        )


def _exact_zero_reference_defects(values: Sequence[Any], role: str, /) -> tuple[str, ...]:
    result = []
    for value in values:
        if value == "interval-certified":
            result.append("interval-certified")
            continue
        if isinstance(value, Fraction):
            zero = value == 0
        elif isinstance(value, RadicalElement):
            zero = value.sign() == 0
        elif isinstance(value, (int, np.integer)) and not isinstance(
            value, (bool, np.bool_)
        ):
            zero = int(value) == 0
        elif isinstance(value, str):
            try:
                zero = Fraction(value) == 0
            except (ValueError, ZeroDivisionError) as error:
                raise TypeError(
                    f"{role} exact reference defects must be rational strings or interval-certified."
                ) from error
        else:
            raise TypeError(
                f"{role} exact reference defects cannot use {type(value).__name__}."
            )
        if not zero:
            raise ValueError(
                "A successful material-reference overlap cannot retain an exact reference coverage defect."
            )
        result.append("0")
    return tuple(result)


class MaterialReferenceVolumeOverlap(StrictModule, NonTrainableState):
    """Sparse target-major material overlap with distinct endpoint volume forms.

    Reference defect entries are exact rational zeroes under an exact topology
    proof, or ``"interval-certified"`` when the reported integration enclosures
    certify coverage for the declared tolerance.
    """

    target_offsets: Array
    source_rows: Array
    reference_overlap_measures: Array
    reference_overlap_errors: Array
    source_overlap_measures: Array
    target_overlap_measures: Array
    source_overlap_errors: Array
    target_overlap_errors: Array
    source_reference_measures: Array
    target_reference_measures: Array
    source_reference_errors: Array
    target_reference_errors: Array
    source_physical_measures: Array
    target_physical_measures: Array
    source_physical_errors: Array
    target_physical_errors: Array
    source_reference_defects: tuple[str, ...] = eqx.field(static=True)
    target_reference_defects: tuple[str, ...] = eqx.field(static=True)
    unmatched_source_rows: Array
    source_coverage_defects: Array
    target_coverage_defects: Array
    source_coverage_tolerances: Array
    target_coverage_tolerances: Array
    source_reference_coverage_tolerances: Array
    target_reference_coverage_tolerances: Array
    source_reference_coverage_defects: Array
    target_reference_coverage_defects: Array
    status: supermesh_owner.CommonRefinementStatus = eqx.field(static=True)
    succeeded: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    unmatched_target_rows: Array
    witness_id: str = eqx.field(static=True)
    root_validity_id: str = eqx.field(static=True)
    source_validity_id: str = eqx.field(static=True)
    target_validity_id: str = eqx.field(static=True)
    source_embedding_id: str = eqx.field(static=True)
    target_embedding_id: str = eqx.field(static=True)
    source_embedding: GlobalEmbeddingCertificate
    target_embedding: GlobalEmbeddingCertificate
    integration_policy: CurvedOverlapIntegrationPolicy
    exact_candidate_count: int = eqx.field(static=True)
    positive_overlap_count: int = eqx.field(static=True)
    exact_polynomial_integral_count: int = eqx.field(static=True)
    enclosed_integral_count: int = eqx.field(static=True)
    overlap_id: str = eqx.field(static=True)

    def __init__(
        self,
        target_offsets: Any,
        source_rows: Any,
        reference_overlap_measures: Any,
        reference_overlap_errors: Any,
        source_overlap_measures: Any,
        target_overlap_measures: Any,
        source_overlap_errors: Any,
        target_overlap_errors: Any,
        source_reference_measures: Any,
        target_reference_measures: Any,
        source_reference_errors: Any,
        target_reference_errors: Any,
        source_physical_measures: Any,
        target_physical_measures: Any,
        source_physical_errors: Any,
        target_physical_errors: Any,
        source_reference_defects: Sequence[Fraction | str],
        target_reference_defects: Sequence[Fraction | str],
        unmatched_source_rows: Any,
        unmatched_target_rows: Any,
        source_coverage_defects: Any,
        target_coverage_defects: Any,
        source_coverage_tolerances: Any,
        target_coverage_tolerances: Any,
        source_reference_coverage_tolerances: Any,
        target_reference_coverage_tolerances: Any,
        source_reference_coverage_defects: Any,
        target_reference_coverage_defects: Any,
        status: supermesh_owner.CommonRefinementStatus,
        succeeded: bool,
        reason: str,
        witness_id: str,
        root_validity_id: str,
        source_validity_id: str,
        target_validity_id: str,
        source_embedding_id: str,
        target_embedding_id: str,
        source_embedding: GlobalEmbeddingCertificate,
        target_embedding: GlobalEmbeddingCertificate,
        integration_policy: CurvedOverlapIntegrationPolicy,
        exact_candidate_count: int,
        positive_overlap_count: int,
        exact_polynomial_integral_count: int,
        enclosed_integral_count: int,
        /,
    ) -> None:
        offsets = np.asarray(target_offsets, dtype=np.int64)
        source_index = np.asarray(source_rows, dtype=np.int64)
        pairs = tuple(
            np.asarray(values, dtype=np.float64)
            for values in (
                reference_overlap_measures,
                reference_overlap_errors,
                source_overlap_measures,
                target_overlap_measures,
                source_overlap_errors,
                target_overlap_errors,
            )
        )
        cells = tuple(
            np.asarray(values, dtype=np.float64)
            for values in (
                source_reference_measures,
                target_reference_measures,
                source_reference_errors,
                target_reference_errors,
                source_physical_measures,
                target_physical_measures,
                source_physical_errors,
                target_physical_errors,
            )
        )
        unmatched_source = np.asarray(unmatched_source_rows, dtype=np.int64)
        unmatched_target = np.asarray(unmatched_target_rows, dtype=np.int64)
        diagnostics = tuple(
            np.asarray(values, dtype=np.float64)
            for values in (
                source_coverage_defects,
                target_coverage_defects,
                source_coverage_tolerances,
                target_coverage_tolerances,
                source_reference_coverage_defects,
                target_reference_coverage_defects,
                source_reference_coverage_tolerances,
                target_reference_coverage_tolerances,
            )
        )
        if (
            offsets.ndim != 1
            or offsets.size < 2
            or offsets[0] != 0
            or offsets[-1] != source_index.size
            or np.any(np.diff(offsets) <= 0)
            or source_index.ndim != 1
            or any(
                values.ndim != 1 or values.shape != source_index.shape for values in pairs
            )
            or len(cells) != 8
            or any(values.ndim != 1 for values in cells)
            or cells[1].shape != (offsets.size - 1,)
            or cells[2].shape != cells[0].shape
            or cells[3].shape != cells[1].shape
            or cells[4].shape != cells[0].shape
            or cells[5].shape != cells[1].shape
            or cells[6].shape != cells[0].shape
            or cells[7].shape != cells[1].shape
            or pairs[0].size == 0
            or any(not np.all(np.isfinite(values)) for values in (*pairs, *cells))
            or any(np.any(values < 0.0) for values in (*pairs, *cells))
            or any(
                np.any(values <= 0.0)
                for values in (
                    pairs[0],
                    pairs[2],
                    pairs[3],
                    cells[0],
                    cells[1],
                    cells[4],
                    cells[5],
                )
            )
            or np.any(source_index < 0)
            or np.any(source_index >= cells[0].size)
            or unmatched_source.ndim != 1
            or unmatched_target.ndim != 1
            or unmatched_source.size != 0
            or unmatched_target.size != 0
            or len(source_reference_defects) != cells[0].size
            or len(target_reference_defects) != cells[1].size
            or diagnostics[0].shape != cells[0].shape
            or diagnostics[1].shape != cells[1].shape
            or diagnostics[2].shape != cells[0].shape
            or diagnostics[3].shape != cells[1].shape
            or diagnostics[4].shape != cells[0].shape
            or diagnostics[5].shape != cells[1].shape
            or diagnostics[6].shape != cells[0].shape
            or diagnostics[7].shape != cells[1].shape
            or np.any(np.abs(diagnostics[0]) > diagnostics[2])
            or np.any(np.abs(diagnostics[1]) > diagnostics[3])
            or np.any(np.abs(diagnostics[4]) > diagnostics[6])
            or np.any(np.abs(diagnostics[5]) > diagnostics[7])
            or any(not np.all(np.isfinite(values)) for values in diagnostics)
            or any(
                np.any(values < 0.0)
                for index, values in enumerate(diagnostics)
                if index in (2, 3, 6, 7)
            )
            or not isinstance(status, supermesh_owner.CommonRefinementStatus)
            or status is not supermesh_owner.CommonRefinementStatus.SUCCESS
            or not succeeded
            or not reason
            or not witness_id
            or not root_validity_id
            or not source_validity_id
            or not target_validity_id
            or not source_embedding_id
            or not target_embedding_id
            or not isinstance(source_embedding, GlobalEmbeddingCertificate)
            or not isinstance(target_embedding, GlobalEmbeddingCertificate)
            or source_embedding.certificate_id != source_embedding_id
            or target_embedding.certificate_id != target_embedding_id
            or not isinstance(integration_policy, CurvedOverlapIntegrationPolicy)
            or min(
                exact_candidate_count,
                positive_overlap_count,
                exact_polynomial_integral_count,
                enclosed_integral_count,
            )
            < 0
        ):
            raise ValueError(
                "Material-reference overlap arrays or evidence are malformed."
            )
        defect_source = _exact_zero_reference_defects(source_reference_defects, "source")
        defect_target = _exact_zero_reference_defects(target_reference_defects, "target")
        if (
            np.unique(source_index).size != cells[0].size
            or exact_candidate_count < positive_overlap_count
            or any(
                np.any(np.diff(source_index[int(start) : int(stop)]) <= 0)
                for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
            )
        ):
            raise ValueError(
                "Material overlap CSR must be complete and target-major canonical."
            )
        arrays = (
            offsets,
            source_index,
            *pairs,
            *cells,
            unmatched_source,
            unmatched_target,
        )
        (
            self.target_offsets,
            self.source_rows,
            self.reference_overlap_measures,
            self.reference_overlap_errors,
            self.source_overlap_measures,
            self.target_overlap_measures,
            self.source_overlap_errors,
            self.target_overlap_errors,
            self.source_reference_measures,
            self.target_reference_measures,
            self.source_reference_errors,
            self.target_reference_errors,
            self.source_physical_measures,
            self.target_physical_measures,
            self.source_physical_errors,
            self.target_physical_errors,
            self.unmatched_source_rows,
            self.unmatched_target_rows,
        ) = tuple(jnp.asarray(value) for value in arrays)
        self.source_reference_defects = defect_source
        self.target_reference_defects = defect_target
        self.witness_id = witness_id
        self.root_validity_id = root_validity_id
        self.source_validity_id = source_validity_id
        self.target_validity_id = target_validity_id
        self.source_embedding_id = source_embedding_id
        self.target_embedding_id = target_embedding_id
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_coverage_defects = jnp.asarray(diagnostics[0])
        self.target_coverage_defects = jnp.asarray(diagnostics[1])
        self.source_coverage_tolerances = jnp.asarray(diagnostics[2])
        self.target_coverage_tolerances = jnp.asarray(diagnostics[3])
        self.source_reference_coverage_defects = jnp.asarray(diagnostics[4])
        self.target_reference_coverage_defects = jnp.asarray(diagnostics[5])
        self.source_reference_coverage_tolerances = jnp.asarray(diagnostics[6])
        self.target_reference_coverage_tolerances = jnp.asarray(diagnostics[7])
        self.status = status
        self.succeeded = bool(succeeded)
        self.reason = str(reason)
        self.integration_policy = integration_policy
        self.exact_candidate_count = int(exact_candidate_count)
        self.positive_overlap_count = int(positive_overlap_count)
        self.exact_polynomial_integral_count = int(exact_polynomial_integral_count)
        self.enclosed_integral_count = int(enclosed_integral_count)
        self.overlap_id = canonical_fingerprint(
            {
                "kind": "material-reference-volume-overlap",
                "target_offsets": array_tree_fingerprint(offsets),
                "source_rows": array_tree_fingerprint(source_index),
                "pair_measures": tuple(array_tree_fingerprint(value) for value in pairs),
                "cell_measures": tuple(array_tree_fingerprint(value) for value in cells),
                "reference_defects": (defect_source, defect_target),
                "unmatched": (
                    array_tree_fingerprint(unmatched_source),
                    array_tree_fingerprint(unmatched_target),
                ),
                "witness": witness_id,
                "validity": (root_validity_id, source_validity_id, target_validity_id),
                "embeddings": (source_embedding_id, target_embedding_id),
                "embedding_certificates": (
                    source_embedding.certificate_id,
                    target_embedding.certificate_id,
                ),
                "coverage_diagnostics": tuple(
                    array_tree_fingerprint(value) for value in diagnostics
                ),
                "status": int(status),
                "reason": reason,
                "policy": integration_policy.policy_id,
                "counts": (
                    exact_candidate_count,
                    positive_overlap_count,
                    exact_polynomial_integral_count,
                    enclosed_integral_count,
                ),
            }
        )


class PhysicalOverlapActionEvidence(NamedTuple):
    """Extensive action output with its observed content and coverage defects."""

    values: Array
    source_total: Array
    target_total: Array
    absolute_conservation_defect: Array
    source_coverage_defects: Array
    target_coverage_defects: Array


def _overlap_field_values(
    refinement: supermesh_owner.PreparedCommonRefinement,
    values: Any,
    count: int,
    name: str,
    /,
) -> Array:
    result = jnp.asarray(values)
    if result.ndim == 0 or result.shape[0] != count:
        raise ValueError(f"{name} must begin with the corresponding cell count.")
    result = eqx.error_if(
        result, ~jnp.all(jnp.isfinite(result)), f"{name} must be finite."
    )
    return result.astype(jnp.result_type(result, jnp.float64))


def apply_physical_overlap_extensive(
    refinement: supermesh_owner.PreparedCommonRefinement,
    source_content: Any,
    /,
) -> Array:
    """Transfer cell content by physical overlap/source measure, without renormalizing gaps."""
    if (
        not isinstance(refinement, supermesh_owner.PreparedCommonRefinement)
        or not refinement.succeeded
    ):
        raise ValueError(
            "Extensive transfer requires a successful prepared common refinement."
        )
    values = _overlap_field_values(
        refinement, source_content, refinement.source_cell_count, "source_content"
    )
    source_rows = refinement.source_cells
    weights = refinement.volumes / refinement.source_measures[source_rows]
    trailing = (1,) * (values.ndim - 1)
    return (
        jnp.zeros((refinement.target_cell_count,) + values.shape[1:], dtype=values.dtype)
        .at[refinement.target_cells]
        .add(values[source_rows] * weights.astype(values.dtype).reshape((-1,) + trailing))
    )


def transpose_physical_overlap_extensive(
    refinement: supermesh_owner.PreparedCommonRefinement,
    target_cotangent: Any,
    /,
) -> Array:
    """Euclidean transpose of :func:`apply_physical_overlap_extensive`."""
    if (
        not isinstance(refinement, supermesh_owner.PreparedCommonRefinement)
        or not refinement.succeeded
    ):
        raise ValueError(
            "Extensive transpose requires a successful prepared common refinement."
        )
    values = _overlap_field_values(
        refinement, target_cotangent, refinement.target_cell_count, "target_cotangent"
    )
    source_rows = refinement.source_cells
    weights = refinement.volumes / refinement.source_measures[source_rows]
    trailing = (1,) * (values.ndim - 1)
    return (
        jnp.zeros((refinement.source_cell_count,) + values.shape[1:], dtype=values.dtype)
        .at[source_rows]
        .add(
            values[refinement.target_cells]
            * weights.astype(values.dtype).reshape((-1,) + trailing)
        )
    )


def apply_physical_overlap_intensive(
    refinement: supermesh_owner.PreparedCommonRefinement,
    source_average: Any,
    /,
) -> Array:
    """Transfer cell averages by physical overlap/target measure, without gap normalization."""
    if (
        not isinstance(refinement, supermesh_owner.PreparedCommonRefinement)
        or not refinement.succeeded
    ):
        raise ValueError(
            "Intensive transfer requires a successful prepared common refinement."
        )
    values = _overlap_field_values(
        refinement, source_average, refinement.source_cell_count, "source_average"
    )
    target_rows = refinement.target_cells
    weights = refinement.volumes / refinement.target_measures[target_rows]
    trailing = (1,) * (values.ndim - 1)
    return (
        jnp.zeros((refinement.target_cell_count,) + values.shape[1:], dtype=values.dtype)
        .at[target_rows]
        .add(
            values[refinement.source_cells]
            * weights.astype(values.dtype).reshape((-1,) + trailing)
        )
    )


def transpose_physical_overlap_intensive(
    refinement: supermesh_owner.PreparedCommonRefinement,
    target_cotangent: Any,
    /,
) -> Array:
    """Euclidean transpose of :func:`apply_physical_overlap_intensive`."""
    if (
        not isinstance(refinement, supermesh_owner.PreparedCommonRefinement)
        or not refinement.succeeded
    ):
        raise ValueError(
            "Intensive transpose requires a successful prepared common refinement."
        )
    values = _overlap_field_values(
        refinement, target_cotangent, refinement.target_cell_count, "target_cotangent"
    )
    target_rows = refinement.target_cells
    weights = refinement.volumes / refinement.target_measures[target_rows]
    trailing = (1,) * (values.ndim - 1)
    return (
        jnp.zeros((refinement.source_cell_count,) + values.shape[1:], dtype=values.dtype)
        .at[refinement.source_cells]
        .add(values[target_rows] * weights.astype(values.dtype).reshape((-1,) + trailing))
    )


def apply_physical_overlap_extensive_with_evidence(
    refinement: supermesh_owner.PreparedCommonRefinement,
    source_content: Any,
    /,
) -> PhysicalOverlapActionEvidence:
    if (
        not isinstance(refinement, supermesh_owner.PreparedCommonRefinement)
        or not refinement.succeeded
    ):
        raise ValueError(
            "Extensive evidence requires a successful prepared common refinement."
        )
    values = _overlap_field_values(
        refinement, source_content, refinement.source_cell_count, "source_content"
    )
    result = apply_physical_overlap_extensive(refinement, values)
    source_total = jnp.sum(values, axis=0)
    target_total = jnp.sum(result, axis=0)
    return PhysicalOverlapActionEvidence(
        result,
        source_total,
        target_total,
        jnp.abs(target_total - source_total),
        refinement.evidence.source_coverage_defects,
        refinement.evidence.target_coverage_defects,
    )


class PhysicalOverlapTransfer(StrictModule, NonTrainableState):
    """Label-free sparse cell-content and cell-average actions over a prepared overlap."""

    refinement: supermesh_owner.PreparedCommonRefinement
    transfer_id: str = eqx.field(static=True)

    def __init__(self, refinement: supermesh_owner.PreparedCommonRefinement, /) -> None:
        if not isinstance(refinement, supermesh_owner.PreparedCommonRefinement):
            raise TypeError("refinement must be PreparedCommonRefinement.")
        if not refinement.succeeded:
            raise ValueError(
                "Physical overlap actions require a successful prepared refinement."
            )
        source = np.asarray(refinement.source_measures, dtype=np.float64)
        target = np.asarray(refinement.target_measures, dtype=np.float64)
        overlap = np.asarray(refinement.volumes, dtype=np.float64)
        if (
            source.shape != (refinement.source_cell_count,)
            or target.shape != (refinement.target_cell_count,)
            or overlap.shape != (refinement.entry_count,)
            or np.any(~np.isfinite(source))
            or np.any(~np.isfinite(target))
            or np.any(~np.isfinite(overlap))
            or np.any(source <= 0.0)
            or np.any(target <= 0.0)
            or np.any(overlap <= 0.0)
        ):
            raise ValueError(
                "Prepared physical overlap measures must be finite and positive."
            )
        self.refinement = refinement
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "physical-overlap-transfer",
                "refinement": refinement.refinement_id,
                "actions": ("extensive-content", "intensive-cell-average"),
            }
        )

    def apply_extensive(self, source_content: Any, /) -> Array:
        return apply_physical_overlap_extensive(self.refinement, source_content)

    def apply_extensive_with_evidence(
        self, source_content: Any, /
    ) -> PhysicalOverlapActionEvidence:
        return apply_physical_overlap_extensive_with_evidence(
            self.refinement, source_content
        )

    def apply_extensive_transpose(self, target_cotangent: Any, /) -> Array:
        return transpose_physical_overlap_extensive(self.refinement, target_cotangent)

    def apply_intensive(self, source_average: Any, /) -> Array:
        return apply_physical_overlap_intensive(self.refinement, source_average)

    def apply_intensive_transpose(self, target_cotangent: Any, /) -> Array:
        return transpose_physical_overlap_intensive(self.refinement, target_cotangent)


class _CellRef(NamedTuple):
    row: int
    global_id: int
    root_id: int
    vertices: Tet
    chart_arguments: tuple[Expression, ...]
    chart_witness: tuple[str, ...]
    corner_image: bool


def _canonical_witness_bank(
    mesh: CellMesh,
    value: Mapping[str, ReferenceCellMapWitnesses],
    role: str,
    /,
) -> tuple[ReferenceCellMapWitnesses, ...]:
    if not isinstance(value, Mapping) or set(value) != {
        block.name for block in mesh.blocks
    }:
        raise ValueError(
            f"{role} root-reference witnesses must name every endpoint block."
        )
    records = []
    for block in mesh.blocks:
        record = value[block.name]
        validate_reference_cell_map_witnesses(record)
        count = block.cell_count
        if (
            record.target_cell_kind != block.cell_kind
            or not np.array_equal(np.asarray(record.target_cell_ids), np.arange(count))
            or np.asarray(record.source_cell_ids).shape != (count,)
        ):
            raise ValueError(
                f"{role} root-reference witness rows must align block-local cell order."
            )
        records.append(record)
    return tuple(records)


def _witness_identity(
    names: tuple[str, ...], maps: tuple[ReferenceCellMapWitnesses, ...], /
) -> tuple[Any, ...]:
    result = []
    for name, record in zip(names, maps, strict=True):
        exact = record.source_reference_exact_vertices
        if exact is None:
            exact = exact_reference_cell_vertices(record)
        result.append(
            (
                name,
                record.target_cell_kind,
                record.source_cell_kind,
                tuple(int(value) for value in np.asarray(record.source_cell_ids)),
                tuple(
                    tuple(tuple(str(value) for value in point) for point in cell)
                    for cell in exact
                ),
                reference_chart_arguments_identity(
                    record.source_reference_chart_arguments
                ),
            )
        )
    return tuple(result)


def _root_cell_rows(mesh: CellMesh, /) -> dict[int, tuple[int, int]]:
    rows: dict[int, tuple[int, int]] = {}
    for block_index, block in enumerate(mesh.blocks):
        for local_row, identifier in enumerate(
            np.asarray(block.global_ids, dtype=np.int64)
        ):
            key = int(identifier)
            if key in rows:
                raise ValueError("Common-root cell global IDs must be globally unique.")
            rows[key] = block_index, local_row
    return rows


def _require_exact_source_bank(geometry: CellGeometrySpec, /) -> None:
    if geometry.source_map is None:
        return
    exact = geometry.source_coordinates()
    rounded = np.asarray(
        [[float(value) for value in row] for row in exact], dtype=np.float64
    )
    if rounded.shape != np.asarray(geometry.coordinates).shape or not np.array_equal(
        rounded, np.asarray(geometry.coordinates, dtype=np.float64)
    ):
        raise ValueError(
            "Exact source-coordinate coefficients must round to the executed geometry bank."
        )


def _coordinate_expression_rows(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    elements: Sequence[Any],
    routes: Sequence[Array],
    /,
) -> tuple[tuple[Expression, Expression, Expression], ...]:
    source = geometry.source_coordinates()
    result = []
    for block, element, block_routes in zip(mesh.blocks, elements, routes, strict=True):
        if not isinstance(element, ScalarCoordinateElement):
            raise ValueError(
                "Common-root overlap requires an exact scalar cell-coordinate map."
            )
        if not isinstance(
            element, (FiniteElementSpec, PerCellRestrictedCellGeometryElement)
        ) and not hasattr(element, "tabulate"):
            raise ValueError("Common-root overlap requires a supported coordinate basis.")
        route_rows = np.asarray(block_routes, dtype=np.int64)
        for row in range(block.cell_count):
            basis = coordinate_effective_basis(element, cell_row=row)
            coefficients = tuple(source[int(index)] for index in route_rows[row])
            if len(basis) != len(coefficients) or len(coefficients[0]) != 3:
                raise ValueError(
                    "Coordinate basis and three-dimensional coefficient rows disagree."
                )
            expressions = linear_combinations(
                basis,
                tuple(
                    tuple(coefficients[dof][axis] for dof in range(len(basis)))
                    for axis in range(3)
                ),
            )
            result.append((expressions[0], expressions[1], expressions[2]))
    return tuple(result)


def _physical_coordinate_bounds(
    expressions: Sequence[tuple[Expression, Expression, Expression]], /
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    lower = np.full((3,), np.inf, dtype=np.float64)
    upper = np.full((3,), -np.inf, dtype=np.float64)
    for row in expressions:
        for axis, expression in enumerate(row):
            low, high = polynomial_bounds(expression, "simplex", 3)
            lower[axis] = min(lower[axis], low)
            upper[axis] = max(upper[axis], high)
    if (
        not expressions
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower > upper)
    ):
        raise ValueError(
            "Material endpoint geometry lacks finite whole-cell coordinate bounds."
        )
    return tuple(float(value) for value in lower), tuple(float(value) for value in upper)


def _chart_arguments(vertices: Tet, /) -> tuple[Expression, Expression, Expression]:
    variables = axes(3)
    origin = vertices[0]

    def coordinate(axis: int) -> Expression:
        return sum_polynomials(
            (
                constant(origin[axis], 3),
                *(
                    scale(
                        variables[column],
                        vertices[column + 1][axis] - origin[axis],
                    )
                    for column in range(3)
                ),
            ),
            3,
        )

    return coordinate(0), coordinate(1), coordinate(2)


def _affine_arguments_from_vertices(
    vertices: Sequence[Sequence[Fraction]], /
) -> tuple[Expression, ...]:
    origin = tuple(vertices[0])
    dimension = len(origin)
    variables = axes(len(vertices) - 1)
    return tuple(
        sum_polynomials(
            (
                constant(origin[axis], len(vertices) - 1),
                *(
                    scale(
                        variables[column],
                        vertices[column + 1][axis] - origin[axis],
                    )
                    for column in range(len(vertices) - 1)
                ),
            ),
            len(vertices) - 1,
        )
        for axis in range(dimension)
    )


def _reference_plane_expression(
    plane: Sequence[int], arguments: Sequence[Expression], /
) -> Expression:
    return sum_polynomials(
        (
            constant(Fraction(int(plane[3])), 3),
            *(scale(arguments[axis], Fraction(int(plane[axis]))) for axis in range(3)),
        ),
        3,
    )


def _bernstein_effective_controls(
    expression: ExpressionInput, domain: BernsteinDomain, dimension: int, /
) -> tuple[Any, ...] | None:
    if isinstance(expression, RationalPolynomial):
        denominator = bernstein_coefficients(expression.denominator, domain, dimension)
        if denominator and all(value > 0 for value in denominator):
            controls = bernstein_coefficients(expression.numerator, domain, dimension)
        elif denominator and all(value < 0 for value in denominator):
            controls = tuple(
                -value
                for value in bernstein_coefficients(
                    expression.numerator, domain, dimension
                )
            )
        else:
            return None
    else:
        controls = bernstein_coefficients(expression, domain, dimension)
    return tuple(controls) if controls else (Fraction(0),)


def _prove_nonpositive_on_reference_simplex(expression: Expression, /) -> tuple[str, ...]:
    """Prove an exact rational/polynomial sign, refining only undecided Bernstein nets."""
    rule = cubature_rule_data("tetrahedron", 8)
    rule_points = np.asarray(rule.points, dtype=np.float64)
    stack = [_canonical_reference_tetrahedron()]
    resolved = 0
    while stack:
        vertices = stack.pop()
        local = compose_expression(expression, _chart_arguments(vertices))
        controls = _bernstein_effective_controls(local, "simplex", 3)
        if controls is not None and max(controls) <= 0:
            resolved += 1
            continue
        center = tuple(
            sum((point[axis] for point in vertices), Fraction(0)) / 4 for axis in range(3)
        )
        if any(
            evaluate_expression(expression, point) > 0 for point in (*vertices, center)
        ):
            raise ValueError(
                "A nonlinear material reference chart is not enclosed by its exact corner tetrahedron."
            )
        if not _has_binary64_tetrahedron_interior(vertices, rule_points):
            raise ValueError(
                "A nonlinear material reference image sign is unresolved at binary64 reference-cell resolution."
            )
        stack.extend(reversed(_subdivide_tetrahedron(vertices)))
    return ("exact-bernstein-nonpositive", f"resolved-subcells:{resolved}")


def _chart_image_tetrahedron_witness(
    arguments: Sequence[Expression],
    vertices: Tet,
    /,
    *,
    allow_curved: bool = False,
) -> tuple[str, ...]:
    """Prove a complete chart image equals its retained corner tetrahedron."""
    if len(arguments) != 3 or _determinant3(vertices) <= 0:
        raise ValueError(
            "A material reference chart needs a positive exact corner tetrahedron."
        )
    denominator_witnesses = []
    for expression in arguments:
        if not isinstance(expression, RationalPolynomial):
            continue
        controls = bernstein_coefficients(expression.denominator, "simplex", 3)
        positive = bool(controls) and all(value > 0 for value in controls)
        negative = bool(controls) and all(value < 0 for value in controls)
        if not (positive or negative):
            raise ValueError(
                "A rational material reference chart has an unresolved denominator sign."
            )
        denominator_witnesses.append(
            "rational-denominator:"
            + ("positive:" if positive else "negative:")
            + ",".join(str(value) for value in controls)
        )
    affine = affine_map(tuple(arguments), 3)
    if affine is not None:
        return (
            "exact-affine-image",
            str(_determinant3(vertices)),
            *denominator_witnesses,
        )

    determinant = _coordinate_jacobian_determinant(arguments)
    determinant_lower, determinant_upper = polynomial_bounds(determinant, "simplex", 3)
    if determinant_lower <= 0.0:
        raise ValueError(
            "A nonlinear material reference chart lacks a certified positive whole-cell Jacobian."
        )

    def curved_witness() -> tuple[str, ...] | None:
        if not allow_curved:
            return None
        injectivity = _triangular_injectivity_witness(arguments)
        if injectivity is None:
            return None
        return (
            "curved-image-triangular-injective",
            *denominator_witnesses,
            "bernstein-positive-jacobian",
            format(determinant_lower, ".17g"),
            format(determinant_upper, ".17g"),
            *injectivity,
        )

    planes = _tetrahedron_planes(vertices)
    canonical = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(
            reference_cell_topology("tetrahedron").vertices, dtype=np.float64
        )
    )
    plane_witnesses = []
    for face_index, plane in enumerate(planes):
        boundary = _reference_plane_expression(plane, arguments)
        try:
            plane_proof = _prove_nonpositive_on_reference_simplex(boundary)
        except ValueError:
            curved = curved_witness()
            if curved is not None:
                return curved
            raise
        plane_witnesses.extend((f"image-plane-{face_index}", *plane_proof))
        face_vertices = tuple(
            canonical[index] for index in range(4) if index != face_index
        )
        face_arguments = _affine_arguments_from_vertices(face_vertices)
        restricted = compose_expression(boundary, face_arguments)
        if not expression_equal(restricted, constant(0, 2)):
            curved = curved_witness()
            if curved is not None:
                return curved
            raise ValueError(
                "A nonlinear material reference chart does not map each complete reference face to its corner face."
            )

    for first, second in combinations(range(4), 2):
        edge_arguments = _affine_arguments_from_vertices(
            (canonical[first], canonical[second])
        )
        mapped_edge = tuple(
            compose_expression(expression, edge_arguments) for expression in arguments
        )
        varying_axis = next(
            (
                axis
                for axis in range(3)
                if vertices[second][axis] != vertices[first][axis]
            ),
            None,
        )
        if varying_axis is None:
            raise ValueError("A nonlinear reference edge has coincident image endpoints.")
        edge_coordinate = scale(
            sum_polynomials(
                (
                    mapped_edge[varying_axis],
                    constant(-vertices[first][varying_axis], 1),
                ),
                1,
            ),
            Fraction(1, 1)
            / (vertices[second][varying_axis] - vertices[first][varying_axis]),
        )
        edge_derivative = derivative(edge_coordinate, 0)
        edge_lower, _ = polynomial_bounds(edge_derivative, "box", 1)
        if edge_lower <= 0.0:
            curved = curved_witness()
            if curved is not None:
                return curved
            raise ValueError(
                "A nonlinear material reference edge lacks a certified endpoint-preserving monotone map."
            )

    for face_index in range(4):
        face_indices = tuple(index for index in range(4) if index != face_index)
        face_vertices = tuple(canonical[index] for index in face_indices)
        face_arguments = _affine_arguments_from_vertices(face_vertices)
        mapped_face = tuple(
            compose_expression(expression, face_arguments) for expression in arguments
        )
        output_face = tuple(vertices[index] for index in face_indices)
        projected_axes = next(
            (
                (first_axis, second_axis)
                for first_axis, second_axis in combinations(range(3), 2)
                if (
                    (output_face[1][first_axis] - output_face[0][first_axis])
                    * (output_face[2][second_axis] - output_face[0][second_axis])
                    - (output_face[1][second_axis] - output_face[0][second_axis])
                    * (output_face[2][first_axis] - output_face[0][first_axis])
                )
            ),
            None,
        )
        if projected_axes is None:
            raise ValueError("A nonlinear material reference face is degenerate.")
        first_axis, second_axis = projected_axes
        determinant_2d = sum_polynomials(
            (
                multiply(
                    derivative(mapped_face[first_axis], 0),
                    derivative(mapped_face[second_axis], 1),
                ),
                scale(
                    multiply(
                        derivative(mapped_face[first_axis], 1),
                        derivative(mapped_face[second_axis], 0),
                    ),
                    -1,
                ),
            ),
            2,
        )
        face_lower, face_upper = polynomial_bounds(determinant_2d, "simplex", 2)
        if face_lower <= 0.0 <= face_upper:
            curved = curved_witness()
            if curved is not None:
                return curved
            raise ValueError(
                "A nonlinear material reference face lacks a certified regular boundary map."
            )

    return (
        *denominator_witnesses,
        *plane_witnesses,
        "bernstein-positive-jacobian",
        format(determinant_lower, ".17g"),
        format(determinant_upper, ".17g"),
        "face-preserving-boundary-homeomorphism",
        "strictly-monotone-oriented-edges",
        "boundary-degree-one-global-injectivity",
    )


def _validate_material_reference_maps(
    root_mesh: CellMesh,
    root_rows: Mapping[int, tuple[int, int]],
    endpoint_mesh: CellMesh,
    witness_maps: tuple[ReferenceCellMapWitnesses, ...],
    /,
) -> tuple[tuple[str, ...], ...]:
    chart_witnesses: list[tuple[str, ...]] = []
    endpoint_row = 0
    for block, witness in zip(endpoint_mesh.blocks, witness_maps, strict=True):
        if witness.source_cell_kind != "tetrahedron":
            raise ValueError(
                "A material-reference tetrahedron must map into root tetrahedra."
            )
        exact_vertices = exact_reference_cell_vertices(witness)
        root_ids = np.asarray(witness.source_cell_ids, dtype=np.int64)
        chart_bank = witness.source_reference_chart_arguments
        for local_row, (root_id, vertices) in enumerate(
            zip(root_ids.tolist(), exact_vertices, strict=True)
        ):
            root_owner = root_rows.get(int(root_id))
            if root_owner is None:
                raise ValueError(
                    "A material reference witness names an absent root cell."
                )
            root_block_index, _ = root_owner
            if root_mesh.blocks[root_block_index].cell_kind != witness.source_cell_kind:
                raise ValueError(
                    "A material reference witness names a root cell of another kind."
                )
            tet = _as_tetrahedron(vertices)
            if _determinant3(tet) <= 0:
                raise ValueError(
                    "Material reference corner tetrahedra must preserve orientation and be nondegenerate."
                )
            arguments = (
                _chart_arguments(tet)
                if chart_bank is None
                else _as_chart_arguments(chart_bank[local_row])
            )
            chart_witnesses.append(
                _chart_image_tetrahedron_witness(arguments, tet, allow_curved=True)
            )
            endpoint_row += 1
    if endpoint_row != sum(block.cell_count for block in endpoint_mesh.blocks):
        raise ValueError("Material reference chart rows do not cover all endpoint cells.")
    return tuple(chart_witnesses)


def _require_root_restrictions(
    root_mesh: CellMesh,
    root_geometry: CellGeometrySpec,
    root_rows: Mapping[int, tuple[int, int]],
    root_expressions: Sequence[tuple[Expression, Expression, Expression]],
    endpoint_mesh: CellMesh,
    endpoint_geometry: CellGeometrySpec,
    witness_maps: tuple[ReferenceCellMapWitnesses, ...],
    endpoint_expressions: Sequence[tuple[Expression, Expression, Expression]],
    /,
) -> tuple[tuple[str, ...], ...]:
    endpoint_row = 0
    chart_witnesses: list[tuple[str, ...]] = []
    for block, witness in zip(endpoint_mesh.blocks, witness_maps, strict=True):
        if witness.source_cell_kind != "tetrahedron":
            raise ValueError(
                "A common-root tetrahedral endpoint must map into root tetrahedra."
            )
        exact_vertices = exact_reference_cell_vertices(witness)
        roots = np.asarray(witness.source_cell_ids, dtype=np.int64)
        for local_row, (root_id, vertices) in enumerate(
            zip(roots.tolist(), exact_vertices, strict=True)
        ):
            root_owner = root_rows.get(int(root_id))
            if root_owner is None:
                raise ValueError(
                    "A reference witness names an absent authoritative root cell."
                )
            root_block_index, root_local_row = root_owner
            root_block = root_mesh.blocks[root_block_index]
            if root_block.cell_kind != witness.source_cell_kind:
                raise ValueError(
                    "A root-reference witness names a root cell of another kind."
                )
            root_row = (
                sum(value.cell_count for value in root_mesh.blocks[:root_block_index])
                + root_local_row
            )
            tet = _as_tetrahedron(vertices)
            chart_arguments = witness.source_reference_chart_arguments
            chart = (
                _chart_arguments(tet)
                if chart_arguments is None
                else _as_chart_arguments(chart_arguments[local_row])
            )
            if _determinant3(tet) <= 0:
                raise ValueError(
                    "Exact common-root tetrahedral charts must preserve reference-cell orientation."
                )
            chart_witnesses.append(_chart_image_tetrahedron_witness(chart, tet))
            expected = tuple(
                compose_expression(value, chart) for value in root_expressions[root_row]
            )
            actual = endpoint_expressions[endpoint_row]
            if any(
                not expression_equal(left, right)
                for left, right in zip(expected, actual, strict=True)
            ):
                raise ValueError(
                    "Endpoint coordinate map is not the witnessed restriction of the authoritative root map."
                )
            points = np.asarray(
                [[float(value) for value in point] for point in tet], dtype=np.float64
            )
            mapped_corners = endpoint_geometry.evaluate(
                endpoint_mesh,
                np.asarray(
                    (int(np.asarray(block.global_ids)[local_row]),), dtype=np.int64
                ),
                np.asarray(
                    reference_cell_topology(block.cell_kind).vertices, dtype=np.float64
                )[None, :, :],
            )[0]
            expected_corners = np.asarray(
                root_geometry.evaluate(
                    root_mesh,
                    np.asarray((int(root_id),), dtype=np.int64),
                    points[None, :, :],
                )[0]
            )
            scale_value = max(1.0, float(np.max(np.abs(expected_corners))))
            if (
                not np.all(np.isfinite(mapped_corners))
                or np.max(np.abs(mapped_corners - expected_corners))
                > 32.0 * np.finfo(np.float64).eps * scale_value
            ):
                raise ValueError(
                    "Endpoint carrier corners disagree with their witnessed root restriction."
                )
            endpoint_row += 1
    if endpoint_row != len(endpoint_expressions):
        raise ValueError(
            "Endpoint coordinate rows are not completely covered by root witnesses."
        )
    return tuple(chart_witnesses)


def _integer_plane(values: Sequence[Fraction], /) -> tuple[int, int, int, int]:
    denominator = lcm(*(value.denominator for value in values))
    row = [int(value * denominator) for value in values]
    divisor = 0
    for value in row:
        divisor = gcd(divisor, abs(value))
    if divisor == 0:
        raise ValueError("Degenerate reference tetrahedron plane.")
    row = [value // divisor for value in row]
    return row[0], row[1], row[2], row[3]


def _tetrahedron_planes(vertices: Tet, /) -> _Planes:
    planes = []
    for omitted in range(4):
        face = [vertices[index] for index in range(4) if index != omitted]
        first = tuple(face[1][axis] - face[0][axis] for axis in range(3))
        second = tuple(face[2][axis] - face[0][axis] for axis in range(3))
        normal = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
        offset = -sum((normal[axis] * face[0][axis] for axis in range(3)), Fraction(0))
        opposite = vertices[omitted]
        if sum((normal[axis] * opposite[axis] for axis in range(3)), offset) > 0:
            normal = tuple(-value for value in normal)
            offset = -offset
        planes.append(_integer_plane((*normal, offset)))
    return tuple(planes)


def _pack_exact_planes(planes: _Planes, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    words: list[int] = []
    offsets = [0]
    signs = []
    for row in planes:
        for value in row:
            signs.append((value > 0) - (value < 0))
            magnitude = abs(value)
            while magnitude:
                words.append(magnitude & 0xFFFFFFFF)
                magnitude >>= 32
            offsets.append(len(words))
    return (
        np.asarray(words, dtype=np.uint32),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(signs, dtype=np.int8).reshape((-1, 4)),
    )


def _pull_target_planes(source: Tet, target: Tet, /) -> _Planes:
    source_origin = source[0]
    matrix = tuple(
        tuple(source[column + 1][row] - source_origin[row] for column in range(3))
        for row in range(3)
    )
    target_planes = _tetrahedron_planes(target)
    pulled = []
    for plane in target_planes:
        normal = plane[:3]
        values = tuple(
            sum(
                (Fraction(normal[row]) * matrix[row][column] for row in range(3)),
                Fraction(0),
            )
            for column in range(3)
        )
        offset = Fraction(plane[3]) + sum(
            (Fraction(normal[row]) * source_origin[row] for row in range(3)), Fraction(0)
        )
        pulled.append(_integer_plane((*values, offset)))
    return tuple(pulled)


def _solve_incident_vertex(plane_ids: Sequence[int], planes: _Planes, /) -> Point:
    for selected in combinations((int(value) for value in plane_ids), 3):
        matrix = [
            [Fraction(planes[index][axis]) for axis in range(3)] for index in selected
        ]
        right = [[Fraction(-planes[index][3])] for index in selected]
        try:
            solution = solve_exact(matrix, right)
        except ValueError:
            continue
        point: Point = (solution[0][0], solution[1][0], solution[2][0])
        if all(
            sum(
                (Fraction(row[axis]) * point[axis] for axis in range(3)), Fraction(row[3])
            )
            <= 0
            for row in planes
        ):
            return point
    raise ValueError(
        "Exact native clip vertex has no feasible nonsingular plane witness."
    )


def _determinant3(vertices: Tet, /) -> Fraction:
    edges = tuple(
        tuple(vertices[column + 1][axis] - vertices[0][axis] for axis in range(3))
        for column in range(3)
    )
    return (
        edges[0][0] * (edges[1][1] * edges[2][2] - edges[1][2] * edges[2][1])
        - edges[1][0] * (edges[0][1] * edges[2][2] - edges[0][2] * edges[2][1])
        + edges[2][0] * (edges[0][1] * edges[1][2] - edges[0][2] * edges[1][1])
    )


def _clip_exact_tetrahedra(first: Tet, second: Tet, /) -> tuple[Tet, ...]:
    canonical: _Planes = (
        (-1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (1, 1, 1, -1),
    )
    planes = canonical + _pull_target_planes(first, second)
    words, offsets, signs = _pack_exact_planes(planes)
    clipped = _meshcore.clip_reference_tetrahedron_exact(words, offsets, signs)
    vertices_local = []
    for row in range(clipped.vertex_incident_offsets.size - 1):
        begin, end = (
            int(value) for value in clipped.vertex_incident_offsets[row : row + 2]
        )
        vertices_local.append(
            _solve_incident_vertex(
                tuple(int(value) for value in clipped.vertex_incident_planes[begin:end]),
                planes,
            )
        )
    if len(vertices_local) < 4:
        return ()
    origin = first[0]
    matrix = tuple(
        tuple(first[column + 1][axis] - origin[axis] for column in range(3))
        for axis in range(3)
    )
    vertices: tuple[Point, ...] = tuple(
        _as_point(
            tuple(
                origin[axis]
                + sum(
                    (matrix[axis][column] * point[column] for column in range(3)),
                    Fraction(0),
                )
                for axis in range(3)
            )
        )
        for point in vertices_local
    )
    center = _as_point(
        tuple(
            sum((point[axis] for point in vertices), Fraction(0)) / len(vertices)
            for axis in range(3)
        )
    )
    pieces: list[Tet] = []
    for face_index in range(clipped.face_planes.size):
        begin, end = (
            int(value) for value in clipped.face_offsets[face_index : face_index + 2]
        )
        face = [int(value) for value in clipped.face_vertices[begin:end]]
        if len(face) < 3:
            raise ValueError("Exact native reference clip returned a non-polygon face.")
        pivot = min(range(len(face)), key=lambda index: vertices[face[index]])
        face = face[pivot:] + face[:pivot]
        for index in range(1, len(face) - 1):
            piece: Tet = (
                center,
                vertices[face[0]],
                vertices[face[index]],
                vertices[face[index + 1]],
            )
            if _determinant3(piece):
                pieces.append(piece)
    if not pieces:
        return ()
    return tuple(pieces)


def _tetrahedron_measure(vertices: Tet, /) -> Fraction:
    return abs(_determinant3(vertices)) / 6


def _cell_records(
    mesh: CellMesh,
    witness_maps: tuple[ReferenceCellMapWitnesses, ...],
    /,
    *,
    allow_curved: bool = False,
) -> tuple[_CellRef, ...]:
    records = []
    row = 0
    for block, witness in zip(mesh.blocks, witness_maps, strict=True):
        exact = exact_reference_cell_vertices(witness)
        root_ids = np.asarray(witness.source_cell_ids, dtype=np.int64)
        chart_bank = witness.source_reference_chart_arguments
        for local, vertices in enumerate(exact):
            tet = _as_tetrahedron(vertices)
            if _determinant3(tet) <= 0:
                raise ValueError(
                    "Root-reference cell witnesses must be positively oriented and nondegenerate."
                )
            chart = (
                _chart_arguments(tet)
                if chart_bank is None
                else _as_chart_arguments(chart_bank[local])
            )
            chart_witness = _chart_image_tetrahedron_witness(
                chart, tet, allow_curved=allow_curved
            )
            records.append(
                _CellRef(
                    row,
                    int(np.asarray(block.global_ids, dtype=np.int64)[local]),
                    int(root_ids[local]),
                    tet,
                    chart,
                    chart_witness,
                    not chart_witness[0].startswith("curved-image-"),
                )
            )
            row += 1
    return tuple(records)


def _reference_image_bounds(
    records: Sequence[_CellRef], /
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty((len(records), 3), dtype=np.float64)
    upper = np.empty_like(lower)
    for index, record in enumerate(records):
        for axis, expression in enumerate(record.chart_arguments):
            low, high = polynomial_bounds(expression, "simplex", 3)
            if not np.isfinite(low) or not np.isfinite(high):
                low, high = 0.0, 1.0
            low = max(0.0, low)
            high = min(1.0, high)
            lower[index, axis] = np.nextafter(low, -np.inf)
            upper[index, axis] = np.nextafter(high, np.inf)
    return lower, upper


def _candidate_pairs(
    source: tuple[_CellRef, ...], target: tuple[_CellRef, ...], /
) -> tuple[tuple[int, int], ...]:
    source_by_root: dict[int, list[_CellRef]] = {}
    target_by_root: dict[int, list[_CellRef]] = {}
    for record in source:
        source_by_root.setdefault(record.root_id, []).append(record)
    for record in target:
        target_by_root.setdefault(record.root_id, []).append(record)
    result = []
    for root_id in sorted(set(source_by_root) & set(target_by_root)):
        first, second = source_by_root[root_id], target_by_root[root_id]
        source_lower, source_upper = _reference_image_bounds(first)
        target_lower, target_upper = _reference_image_bounds(second)
        source_bvh = prepare_bvh(source_lower, source_upper, dtype=jnp.float64)
        target_bvh = prepare_bvh(target_lower, target_upper, dtype=jnp.float64)
        for target_rows, source_rows in bvh_overlap_pair_blocks(target_bvh, source_bvh):
            result.extend(
                (first[int(source_row)].row, second[int(target_row)].row)
                for target_row, source_row in zip(target_rows, source_rows, strict=True)
            )
    return tuple(sorted(result, key=lambda pair: (pair[1], pair[0])))


def _exact_expression_integral(expression: Expression, vertices: Tet, /) -> Any | None:
    arguments = _chart_arguments(vertices)
    local = compose_expression(expression, arguments)
    if isinstance(local, RationalPolynomial):
        return None
    if not isinstance(local, Mapping):
        return None
    determinant = abs(_determinant3(vertices))
    if isinstance(determinant, (int, np.integer)) and not isinstance(
        determinant, (bool, np.bool_)
    ):
        determinant = Fraction(int(determinant))
    if not isinstance(determinant, (Fraction, RadicalElement)):
        raise TypeError("Exact simplex integration requires exact reference vertices.")
    total: Any = Fraction(0)
    for exponent, raw_coefficient in local.items():
        if isinstance(raw_coefficient, (Fraction, RadicalElement)):
            coefficient = raw_coefficient
        elif isinstance(raw_coefficient, (int, np.integer)) and not isinstance(
            raw_coefficient, (bool, np.bool_)
        ):
            coefficient = Fraction(int(raw_coefficient))
        else:
            raise TypeError(
                "Exact polynomial integration requires exact represented coefficients before moment evaluation."
            )
        factor = Fraction(
            factorial(exponent[0]) * factorial(exponent[1]) * factorial(exponent[2]),
            factorial(3 + sum(exponent)),
        )
        total = total + coefficient * factor
    exact = total * determinant
    if not isinstance(exact, (Fraction, RadicalElement)):
        raise TypeError(
            "Exact polynomial integration requires Fraction or RadicalElement output."
        )
    return exact


def _evaluate_expression(expression: Expression, point: Sequence[float], /) -> float:
    def polynomial(values: Mapping[tuple[int, ...], Any]) -> float:
        result = 0.0
        for exponent, coefficient in values.items():
            term = float(coefficient)
            for coordinate, degree in zip(point, exponent, strict=True):
                term *= coordinate**degree
            result += term
        return result

    if isinstance(expression, RationalPolynomial):
        denominator = polynomial(expression.denominator)
        if denominator == 0.0:
            raise ValueError(
                "Curved coordinate map denominator vanishes on an integration point."
            )
        return polynomial(expression.numerator) / denominator
    return polynomial(expression)


def _subdivide_tetrahedron(vertices: Tet, /) -> tuple[Tet, ...]:
    v0, v1, v2, v3 = vertices
    midpoint = lambda first, second: tuple(
        (a + b) / 2 for a, b in zip(first, second, strict=True)
    )
    m01, m02, m03 = midpoint(v0, v1), midpoint(v0, v2), midpoint(v0, v3)
    m12, m13, m23 = midpoint(v1, v2), midpoint(v1, v3), midpoint(v2, v3)
    return (
        (v0, m01, m02, m03),
        (v1, m01, m12, m13),
        (v2, m02, m12, m23),
        (v3, m03, m13, m23),
        (m01, m02, m03, m23),
        (m01, m02, m12, m23),
        (m01, m12, m13, m23),
        (m01, m03, m13, m23),
    )


def _has_binary64_tetrahedron_interior(
    vertices: Tet,
    rule_points: np.ndarray,
    /,
) -> bool:
    rounded = np.asarray(
        [[float(value) for value in point] for point in vertices], dtype=np.float64
    )
    if (
        not np.all(np.isfinite(rounded))
        or np.unique(rounded, axis=0).shape[0] != 4
        or float(_determinant3(vertices)) == 0.0
    ):
        return False
    frame = np.asarray(
        [
            [float(vertices[column + 1][axis] - vertices[0][axis]) for axis in range(3)]
            for column in range(3)
        ],
        dtype=np.float64,
    ).T
    origin = np.asarray([float(value) for value in vertices[0]], dtype=np.float64)
    samples = origin[None, :] + rule_points @ frame.T
    if not np.all(np.isfinite(samples)):
        return False
    return any(
        not any(np.array_equal(sample, vertex) for vertex in rounded)
        for sample in samples
    )


def _adaptive_expression_integral(
    expression: Expression,
    vertices: Tet,
    policy: CurvedOverlapIntegrationPolicy,
    rule_points: np.ndarray,
    rule_weights: np.ndarray,
    /,
) -> tuple[float, float]:
    """Integrate a rational map with outward Bernstein integral enclosures."""

    def record(tet: Tet) -> tuple[Tet, float, float, float, float]:
        arguments = _chart_arguments(tet)
        local = compose_expression(expression, arguments)
        lower, upper = polynomial_bounds(local, "simplex", 3)
        determinant = abs(_determinant3(tet))
        determinant_lower = coefficient_endpoint(determinant, upper=False)
        determinant_upper = coefficient_endpoint(determinant, upper=True)
        measure_low, measure_high = determinant_lower / 6.0, determinant_upper / 6.0
        centroid_estimate = 0.0
        frame = np.asarray(
            [
                [float(tet[column + 1][axis] - tet[0][axis]) for axis in range(3)]
                for column in range(3)
            ],
            dtype=np.float64,
        ).T
        origin = np.asarray([float(value) for value in tet[0]], dtype=np.float64)
        reference_points = origin[None, :] + rule_points @ frame.T
        estimate_local = float(
            np.dot(
                rule_weights,
                np.asarray(
                    [
                        _evaluate_expression(expression, point)
                        for point in reference_points
                    ]
                ),
            )
        )
        centroid_estimate = float(determinant) * estimate_local
        candidate_bounds = (
            float(lower) * measure_low,
            float(lower) * measure_high,
            float(upper) * measure_low,
            float(upper) * measure_high,
        )
        low, high = min(candidate_bounds), max(candidate_bounds)
        estimate = min(high, max(low, centroid_estimate))
        return tet, estimate, low, high, max(estimate - low, high - estimate)

    leaves = [record(vertices)]
    blocked: set[Tet] = set()
    if not _has_binary64_tetrahedron_interior(vertices, rule_points):
        blocked.add(vertices)
    while True:
        estimate = sum(item[1] for item in leaves)
        lower = sum(item[2] for item in leaves)
        upper = sum(item[3] for item in leaves)
        error = max(estimate - lower, upper - estimate)
        scale = max(
            abs(estimate), sum(max(abs(item[2]), abs(item[3])) for item in leaves)
        )
        tolerance = policy.absolute_tolerance + policy.relative_tolerance * scale
        if error <= tolerance:
            return estimate, error
        eligible = [row for row, item in enumerate(leaves) if item[0] not in blocked]
        if not eligible:
            unresolved = max(leaves, key=lambda item: item[4])[0]
            raise CommonRootOverlapError(
                "Rational overlap integration is unresolved at binary64 reference-cell resolution.",
                supermesh_owner.CommonRefinementStatus.PREDICATE_UNCERTAIN,
                integration_error=error,
                integration_tolerance=tolerance,
                unresolved_reference_tetrahedron=unresolved,
            )
        index = max(eligible, key=lambda row: leaves[row][4])
        tet = leaves.pop(index)[0]
        for child in _subdivide_tetrahedron(tet):
            leaves.append(record(child))
            if not _has_binary64_tetrahedron_interior(child, rule_points):
                blocked.add(child)


def _integrate_expression(
    expression: Expression,
    vertices: Tet,
    policy: CurvedOverlapIntegrationPolicy,
    rule_points: np.ndarray,
    rule_weights: np.ndarray,
    /,
) -> tuple[float, float, bool]:
    exact = _exact_expression_integral(expression, vertices)
    if exact is not None:
        lower = coefficient_endpoint(exact, upper=False)
        upper = coefficient_endpoint(exact, upper=True)
        return (lower + upper) * 0.5, (upper - lower) * 0.5, True
    value, error = _adaptive_expression_integral(
        expression, vertices, policy, rule_points, rule_weights
    )
    return value, error, False


def _coordinate_jacobian_determinant(expressions: Sequence[Expression], /) -> Expression:
    dimension = 3
    jacobian = tuple(
        tuple(derivative(expressions[row], column) for column in range(3))
        for row in range(3)
    )
    products = (
        multiply(jacobian[0][0], multiply(jacobian[1][1], jacobian[2][2])),
        scale(multiply(jacobian[0][0], multiply(jacobian[1][2], jacobian[2][1])), -1),
        scale(multiply(jacobian[0][1], multiply(jacobian[1][0], jacobian[2][2])), -1),
        multiply(jacobian[0][1], multiply(jacobian[1][2], jacobian[2][0])),
        multiply(jacobian[0][2], multiply(jacobian[1][0], jacobian[2][1])),
        scale(multiply(jacobian[0][2], multiply(jacobian[1][1], jacobian[2][0])), -1),
    )
    return sum_polynomials(products, dimension)


def _integrand_expressions(
    coordinate_expressions: Sequence[Expression],
    determinant: Expression,
    include_second: bool,
    /,
) -> tuple[Expression, ...]:
    values = [determinant]
    values.extend(multiply(value, determinant) for value in coordinate_expressions)
    if include_second:
        values.extend(
            multiply(multiply(first, second), determinant)
            for first in coordinate_expressions
            for second in coordinate_expressions
        )
    return tuple(values)


def _integrate_piece_bank(
    pieces: Sequence[Tet],
    integrands: Sequence[Expression],
    policy: CurvedOverlapIntegrationPolicy,
    rule_points: np.ndarray,
    rule_weights: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    totals = np.zeros((len(integrands),), dtype=np.float64)
    errors = np.zeros_like(totals)
    exact_count = enclosed_count = 0
    for piece in pieces:
        for index, expression in enumerate(integrands):
            value, error, exact = _integrate_expression(
                expression, piece, policy, rule_points, rule_weights
            )
            totals[index] += value
            errors[index] += error
            exact_count += int(exact)
            enclosed_count += int(not exact)
    return totals, errors, exact_count, enclosed_count


def _physical_cell_boxes(
    records: tuple[_CellRef, ...],
    witness: CommonRootVolumeWitness,
    root_expressions: Sequence[tuple[Expression, Expression, Expression]],
    root_rows: Mapping[int, tuple[int, int]],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty((len(records), 3), dtype=np.float64)
    upper = np.empty_like(lower)
    for record in records:
        block_index, local_row = root_rows[record.root_id]
        global_row = (
            sum(block.cell_count for block in witness.root_mesh.blocks[:block_index])
            + local_row
        )
        arguments = _chart_arguments(record.vertices)
        for axis, expression in enumerate(root_expressions[global_row]):
            local = compose_expression(expression, arguments)
            low, high = polynomial_bounds(local, "simplex", 3)
            lower[record.row, axis] = low
            upper[record.row, axis] = high
    return lower, upper


def _reference_self_overlap(
    records: tuple[_CellRef, ...], /
) -> tuple[int, tuple[int, int] | None]:
    by_root: dict[int, list[_CellRef]] = {}
    for record in records:
        by_root.setdefault(record.root_id, []).append(record)
    candidates = 0
    for cells in by_root.values():
        lower, upper = _reference_image_bounds(cells)
        bvh = prepare_bvh(lower, upper, dtype=jnp.float64)
        for target_rows, source_rows in bvh_overlap_pair_blocks(bvh, bvh):
            for target_row, source_row in zip(target_rows, source_rows, strict=True):
                first, second = int(source_row), int(target_row)
                if first >= second:
                    continue
                candidates += 1
                if _clip_exact_tetrahedra(cells[first].vertices, cells[second].vertices):
                    return candidates, (cells[first].row, cells[second].row)
    return candidates, None


def _materialize_decomposition(
    mesh: CellMesh,
    measures: np.ndarray,
    moments: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    records: tuple[_CellRef, ...],
    /,
) -> Any:
    count = len(records)
    return supermesh_owner._Decomposition(
        count,
        np.concatenate(
            tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
        ),
        np.asarray([record.vertices for record in records], dtype=np.float64),
        None,
        np.arange(count + 1, dtype=np.int64),
        measures,
        moments,
        lower,
        upper,
        0,
        0,
    )


def prepare_common_root_volume_refinement(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    witness: CommonRootVolumeWitness,
    /,
    *,
    policy: supermesh_owner.CommonRefinementPolicy,
    integration_policy: CurvedOverlapIntegrationPolicy,
) -> tuple[supermesh_owner.PreparedCommonRefinement, CurvedOverlapIntegrationEvidence]:
    """Clip exact root-reference tetrahedra and integrate the curved root Jacobian."""
    if not isinstance(witness, CommonRootVolumeWitness):
        raise TypeError("witness must be CommonRootVolumeWitness.")
    if not isinstance(policy, supermesh_owner.CommonRefinementPolicy) or not isinstance(
        integration_policy, CurvedOverlapIntegrationPolicy
    ):
        raise TypeError(
            "Curved overlap requires common-refinement and integration policies."
        )
    if policy.overlap_simplices:
        raise ValueError(
            "Curved common-root overlaps cannot expose straight physical overlap simplices."
        )
    witness = witness.revalidate(
        witness.root_mesh,
        witness.root_geometry,
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
    )
    root_validity = witness.root_validity
    root_validity.require_current(witness.root_geometry, mesh=witness.root_mesh)
    if root_validity.certified_valid_count != sum(
        block.cell_count for block in witness.root_mesh.blocks
    ):
        raise ValueError(
            "Curved common-root overlap requires certified positive root-cell Jacobians."
        )
    root_rows = _root_cell_rows(witness.root_mesh)
    root_elements, root_routes, _ = witness.root_geometry.resolve(witness.root_mesh)
    root_expressions = _coordinate_expression_rows(
        witness.root_mesh, witness.root_geometry, root_elements, root_routes
    )
    root_determinants = tuple(
        _coordinate_jacobian_determinant(row) for row in root_expressions
    )
    source_records = _cell_records(source_mesh, witness.source_reference_maps)
    target_records = _cell_records(target_mesh, witness.target_reference_maps)
    source_self_candidates, source_overlap = _reference_self_overlap(source_records)
    target_self_candidates, target_overlap = _reference_self_overlap(target_records)
    if source_overlap is not None or target_overlap is not None:
        source_pairs = () if source_overlap is None else source_overlap
        target_pairs = () if target_overlap is None else target_overlap
        raise CommonRootOverlapError(
            f"Root-reference endpoint cells overlap in positive measure: {(source_overlap, target_overlap)!r}.",
            supermesh_owner.CommonRefinementStatus.DOUBLE_COVERAGE,
            overlapping_source_rows=source_pairs,
            overlapping_target_rows=target_pairs,
        )
    candidates = _candidate_pairs(source_records, target_records)
    rule = cubature_rule_data("tetrahedron", 8)
    rule_points, rule_weights = (
        np.asarray(rule.points, dtype=np.float64),
        np.asarray(rule.weights, dtype=np.float64),
    )
    source_count, target_count = len(source_records), len(target_records)
    source_cell_values = np.zeros(
        (source_count, 13 if policy.second_moments else 4), dtype=np.float64
    )
    target_cell_values = np.zeros(
        (target_count, 13 if policy.second_moments else 4), dtype=np.float64
    )
    source_cell_errors = np.zeros_like(source_cell_values)
    target_cell_errors = np.zeros_like(target_cell_values)
    integrand_cache: dict[int, tuple[Expression, ...]] = {}
    exact_integral_count = enclosed_integral_count = 0

    def integrands(root_id: int) -> tuple[Expression, ...]:
        nonlocal exact_integral_count, enclosed_integral_count
        if root_id not in integrand_cache:
            block_index, local_row = root_rows[root_id]
            root_row = (
                sum(block.cell_count for block in witness.root_mesh.blocks[:block_index])
                + local_row
            )
            integrand_cache[root_id] = _integrand_expressions(
                root_expressions[root_row],
                root_determinants[root_row],
                policy.second_moments,
            )
        return integrand_cache[root_id]

    def integrate_cell(record: _CellRef) -> tuple[np.ndarray, np.ndarray]:
        nonlocal exact_integral_count, enclosed_integral_count
        values, errors, exact_count, enclosed_count = _integrate_piece_bank(
            (record.vertices,),
            integrands(record.root_id),
            integration_policy,
            rule_points,
            rule_weights,
        )
        exact_integral_count += exact_count
        enclosed_integral_count += enclosed_count
        return values, errors

    for record in source_records:
        source_cell_values[record.row], source_cell_errors[record.row] = integrate_cell(
            record
        )
    for record in target_records:
        target_cell_values[record.row], target_cell_errors[record.row] = integrate_cell(
            record
        )

    overlap_source: list[int] = []
    overlap_target: list[int] = []
    overlap_measures: list[float] = []
    overlap_moments: list[np.ndarray] = []
    overlap_second: list[np.ndarray] = []
    overlap_errors: list[np.ndarray] = []
    reference_overlap_measures: dict[tuple[int, int], Fraction] = {}
    for source_row, target_row in candidates:
        source_record, target_record = (
            source_records[source_row],
            target_records[target_row],
        )
        ref_pieces = _clip_exact_tetrahedra(
            source_record.vertices, target_record.vertices
        )
        if not ref_pieces:
            continue
        reference_measure = sum(
            (_tetrahedron_measure(piece) for piece in ref_pieces), Fraction(0)
        )
        if not reference_measure:
            continue
        values, errors, exact_count, enclosed_count = _integrate_piece_bank(
            ref_pieces,
            integrands(source_record.root_id),
            integration_policy,
            rule_points,
            rule_weights,
        )
        exact_integral_count += exact_count
        enclosed_integral_count += enclosed_count
        overlap_source.append(source_row)
        overlap_target.append(target_row)
        overlap_measures.append(float(values[0]))
        overlap_moments.append(values[1:4])
        if policy.second_moments:
            overlap_second.append(values[4:].reshape((3, 3)))
        overlap_errors.append(errors)
        reference_overlap_measures[(source_row, target_row)] = reference_measure

    source_ref_measures = tuple(
        _tetrahedron_measure(record.vertices) for record in source_records
    )
    target_ref_measures = tuple(
        _tetrahedron_measure(record.vertices) for record in target_records
    )
    source_ref_covered = [Fraction(0) for _ in source_records]
    target_ref_covered = [Fraction(0) for _ in target_records]
    for (source_row, target_row), value in reference_overlap_measures.items():
        source_ref_covered[source_row] += value
        target_ref_covered[target_row] += value
    source_ref_defects = tuple(
        value - expected
        for value, expected in zip(source_ref_covered, source_ref_measures, strict=True)
    )
    target_ref_defects = tuple(
        value - expected
        for value, expected in zip(target_ref_covered, target_ref_measures, strict=True)
    )
    unmatched_source = np.flatnonzero(
        np.asarray([value == 0 for value in source_ref_covered], dtype=np.bool_)
    )
    unmatched_target = np.flatnonzero(
        np.asarray([value == 0 for value in target_ref_covered], dtype=np.bool_)
    )
    source_over = any(value > 0 for value in source_ref_defects)
    target_over = any(value > 0 for value in target_ref_defects)
    source_gap = any(value < 0 for value in source_ref_defects)
    target_gap = any(value < 0 for value in target_ref_defects)
    require_source = policy.coverage in (
        supermesh_owner.CommonRefinementCoverage.COMPLETE,
        supermesh_owner.CommonRefinementCoverage.SOURCE,
    )
    require_target = policy.coverage in (
        supermesh_owner.CommonRefinementCoverage.COMPLETE,
        supermesh_owner.CommonRefinementCoverage.TARGET,
    )
    if (
        source_over
        or target_over
        or require_source
        and source_gap
        or require_target
        and target_gap
    ):
        raise CommonRootOverlapError(
            "Root-reference common refinement has an exact source/target coverage defect: "
            f"{(tuple(map(str, source_ref_defects)), tuple(map(str, target_ref_defects)), unmatched_source.tolist(), unmatched_target.tolist())!r}.",
            supermesh_owner.CommonRefinementStatus.DOUBLE_COVERAGE
            if source_over or target_over
            else supermesh_owner.CommonRefinementStatus.COVERAGE_GAP,
            source_reference_defects=source_ref_defects,
            target_reference_defects=target_ref_defects,
            unmatched_source_rows=unmatched_source.tolist(),
            unmatched_target_rows=unmatched_target.tolist(),
        )

    source_lower, source_upper = _physical_cell_boxes(
        source_records, witness, root_expressions, root_rows
    )
    target_lower, target_upper = _physical_cell_boxes(
        target_records, witness, root_expressions, root_rows
    )
    source_measures, target_measures = source_cell_values[:, 0], target_cell_values[:, 0]
    source_moments, target_moments = (
        source_cell_values[:, 1:4],
        target_cell_values[:, 1:4],
    )
    source_overlap = np.asarray(overlap_source, dtype=np.int64)
    target_overlap = np.asarray(overlap_target, dtype=np.int64)
    overlap_volume = np.asarray(overlap_measures, dtype=np.float64)
    overlap_first = np.asarray(overlap_moments, dtype=np.float64).reshape((-1, 3))
    source_covered = np.bincount(
        source_overlap, weights=overlap_volume, minlength=source_count
    )
    target_covered = np.bincount(
        target_overlap, weights=overlap_volume, minlength=target_count
    )
    source_defects = source_covered - source_measures
    target_defects = target_covered - target_measures
    overlap_error_values = np.asarray(overlap_errors, dtype=np.float64).reshape(
        (-1, source_cell_errors.shape[1])
    )
    source_overlap_error = np.bincount(
        source_overlap, weights=overlap_error_values[:, 0], minlength=source_count
    )
    target_overlap_error = np.bincount(
        target_overlap, weights=overlap_error_values[:, 0], minlength=target_count
    )
    source_tolerance = (
        policy.coverage_tolerance * source_measures
        + source_cell_errors[:, 0]
        + source_overlap_error
        + supermesh_owner._COVERAGE_ROUNDOFF_ULPS
        * np.finfo(np.float64).eps
        * np.linalg.norm(source_upper - source_lower, axis=1) ** 3
    )
    target_tolerance = (
        policy.coverage_tolerance * target_measures
        + target_cell_errors[:, 0]
        + target_overlap_error
        + supermesh_owner._COVERAGE_ROUNDOFF_ULPS
        * np.finfo(np.float64).eps
        * np.linalg.norm(target_upper - target_lower, axis=1) ** 3
    )
    status, reason = supermesh_owner._coverage_status(
        source_defects, target_defects, source_tolerance, target_tolerance, policy
    )
    counts = supermesh_owner._Counts(
        len(candidates), len(overlap_measures), len(candidates), 0, 0, 0
    )
    source_decomposition = _materialize_decomposition(
        source_mesh,
        source_measures,
        source_moments,
        source_lower,
        source_upper,
        source_records,
    )
    target_decomposition = _materialize_decomposition(
        target_mesh,
        target_measures,
        target_moments,
        target_lower,
        target_upper,
        target_records,
    )
    second_moments = (
        np.asarray(overlap_second, dtype=np.float64).reshape((-1, 3, 3))
        if policy.second_moments
        else None
    )
    entries = supermesh_owner._Entries(
        target_overlap,
        source_overlap,
        overlap_volume,
        overlap_first,
        second_moments,
        None,
        None,
    )
    evidence = supermesh_owner.CommonRefinementEvidence(
        status=status,
        reason=reason,
        source_coverage_defects=source_defects,
        target_coverage_defects=target_defects,
        source_measures=source_measures,
        target_measures=target_measures,
        source_tolerances=source_tolerance,
        target_tolerances=target_tolerance,
        counts=counts,
    )
    refinement = supermesh_owner.PreparedCommonRefinement(
        source=source_decomposition,
        target=target_decomposition,
        entries=entries,
        identities=(
            source_mesh.mesh_id,
            target_mesh.mesh_id,
            source_mesh.topology_id,
            target_mesh.topology_id,
        ),
        policy=policy,
        evidence=evidence,
    )
    integration_evidence = CurvedOverlapIntegrationEvidence(
        overlap_error_values,
        source_cell_errors,
        target_cell_errors,
        np.stack((source_lower, source_upper), axis=1),
        np.stack((target_lower, target_upper), axis=1),
        source_ref_defects,
        target_ref_defects,
        unmatched_source,
        unmatched_target,
        root_validity.certificate_id,
        witness.witness_id,
        integration_policy,
        len(candidates) + source_self_candidates + target_self_candidates,
        len(overlap_measures),
        exact_integral_count,
        enclosed_integral_count,
    )
    return refinement, integration_evidence


def _canonical_reference_tetrahedron() -> Tet:
    zero, one = Fraction(0), Fraction(1)
    return (
        (zero, zero, zero),
        (one, zero, zero),
        (zero, one, zero),
        (zero, zero, one),
    )


def _integrate_classified_overlap(
    physical_jacobian: Expression,
    policy: CurvedOverlapIntegrationPolicy,
    rule_points: np.ndarray,
    rule_weights: np.ndarray,
    classify: Any,
    /,
) -> tuple[float, float, int, int]:
    """Share one certified nonnegative integral engine across exact classifiers."""

    def quadrature_estimate(vertices: Tet) -> float:
        rounded = np.asarray(
            [[float(value) for value in point] for point in vertices],
            dtype=np.float64,
        )
        frame = np.asarray(
            [
                [
                    float(vertices[column + 1][axis] - vertices[0][axis])
                    for column in range(3)
                ]
                for axis in range(3)
            ],
            dtype=np.float64,
        )
        points = rounded[0][None, :] + rule_points @ frame.T
        value = float(
            np.dot(
                rule_weights,
                np.asarray(
                    [_evaluate_expression(physical_jacobian, point) for point in points],
                    dtype=np.float64,
                ),
            )
        )
        return float(abs(_determinant3(vertices))) * value

    def record(
        vertices: Tet,
    ) -> tuple[Tet, str, float, float, float, int, int, bool]:
        classification = classify(vertices)
        if classification == "outside":
            return vertices, classification, 0.0, 0.0, 0.0, 0, 0, True
        local_arguments = _chart_arguments(vertices)
        local_integrand = compose_expression(physical_jacobian, local_arguments)
        lower, upper = polynomial_bounds(local_integrand, "simplex", 3)
        determinant = abs(_determinant3(vertices))
        measure_lower = coefficient_endpoint(determinant, upper=False)
        measure_upper = coefficient_endpoint(determinant, upper=True)
        if (
            not np.isfinite(measure_upper)
            or (determinant and measure_upper == 0.0)
            or not np.isfinite(lower)
            or not np.isfinite(upper)
        ):
            lower_integral, upper_integral = 0.0, np.inf
        else:
            lower_product = max(0.0, lower) * measure_lower
            upper_product = max(0.0, upper) * measure_upper
            lower_integral = max(0.0, float(np.nextafter(lower_product, -np.inf)))
            upper_integral = float(np.nextafter(upper_product, np.inf))
        if classification == "boundary":
            value = 0.0 if not np.isfinite(upper_integral) else 0.5 * upper_integral
            return (
                vertices,
                classification,
                value,
                0.0,
                upper_integral,
                0,
                1,
                not _has_binary64_tetrahedron_interior(vertices, rule_points),
            )
        exact = _exact_expression_integral(physical_jacobian, vertices)
        if exact is not None:
            exact_lower = coefficient_endpoint(exact, upper=False)
            exact_upper = coefficient_endpoint(exact, upper=True)
            value = 0.5 * (exact_lower + exact_upper)
            return (
                vertices,
                classification,
                value,
                exact_lower,
                exact_upper,
                1,
                0,
                not _has_binary64_tetrahedron_interior(vertices, rule_points),
            )
        try:
            estimate = quadrature_estimate(vertices)
        except (ArithmeticError, ValueError, ZeroDivisionError) as error:
            raise CommonRootOverlapError(
                "An endpoint physical Jacobian is unresolved on a material overlap cell.",
                supermesh_owner.CommonRefinementStatus.PREDICATE_UNCERTAIN,
                unresolved_reference_tetrahedron=vertices,
            ) from error
        if not np.isfinite(estimate):
            estimate = 0.0
            lower_integral, upper_integral = 0.0, np.inf
        estimate = min(upper_integral, max(lower_integral, estimate))
        return (
            vertices,
            classification,
            estimate,
            lower_integral,
            upper_integral,
            0,
            1,
            not _has_binary64_tetrahedron_interior(vertices, rule_points),
        )

    leaves: dict[int, tuple[Tet, str, float, float, float, int, int, bool]] = {}
    refinement_heap: list[tuple[float, int]] = []
    next_leaf = 0
    exact_count = 0
    enclosed_count = 0
    upper_infinite_count = 0
    value_total = Fraction(0)
    lower_total = Fraction(0)
    upper_total = Fraction(0)

    def add_leaf(
        item: tuple[Tet, str, float, float, float, int, int, bool],
    ) -> int:
        nonlocal next_leaf, exact_count, enclosed_count
        nonlocal upper_infinite_count, value_total, lower_total, upper_total
        identifier = next_leaf
        next_leaf += 1
        leaves[identifier] = item
        value_total += Fraction(float(item[2]))
        lower_total += Fraction(float(item[3]))
        if np.isfinite(item[4]):
            upper_total += Fraction(float(item[4]))
        else:
            upper_infinite_count += 1
        exact_count += item[5]
        enclosed_count += item[6]
        if item[1] != "outside" and not item[7]:
            leaf_error = max(item[2] - item[3], item[4] - item[2])
            heapq.heappush(refinement_heap, (-leaf_error, identifier))
        return identifier

    def remove_leaf(
        identifier: int,
    ) -> tuple[Tet, str, float, float, float, int, int, bool]:
        nonlocal exact_count, enclosed_count, upper_infinite_count
        nonlocal value_total, lower_total, upper_total
        item = leaves.pop(identifier)
        value_total -= Fraction(float(item[2]))
        lower_total -= Fraction(float(item[3]))
        if np.isfinite(item[4]):
            upper_total -= Fraction(float(item[4]))
        else:
            upper_infinite_count -= 1
        exact_count -= item[5]
        enclosed_count -= item[6]
        return item

    add_leaf(record(_canonical_reference_tetrahedron()))
    while True:
        try:
            value = float(value_total)
        except OverflowError:
            value = np.inf
        lower = coefficient_endpoint(lower_total, upper=False, saturate=True)
        upper = (
            np.inf
            if upper_infinite_count
            else coefficient_endpoint(upper_total, upper=True, saturate=True)
        )
        value = min(upper, max(lower, value))
        if np.isfinite(value) and np.isfinite(lower) and np.isfinite(upper):
            error = max(value - lower, upper - value)
            if error > 0.0:
                error = float(np.nextafter(error, np.inf))
        else:
            error = np.inf
        scale_value = max(abs(value), upper)
        relative_error_budget = policy.relative_tolerance * scale_value
        if relative_error_budget:
            relative_error_budget = max(
                0.0, float(np.nextafter(relative_error_budget, -np.inf))
            )
        tolerance = policy.absolute_tolerance + relative_error_budget
        if tolerance:
            tolerance = max(0.0, float(np.nextafter(tolerance, -np.inf)))
        if np.isfinite(error) and np.isfinite(tolerance) and error <= tolerance:
            return value, error, exact_count, enclosed_count
        selected = None
        while refinement_heap:
            _, candidate = heapq.heappop(refinement_heap)
            if candidate in leaves:
                selected = candidate
                break
        if selected is None:
            unresolved = max(
                leaves.values(),
                key=lambda item: max(item[2] - item[3], item[4] - item[2]),
            )[0]
            raise CommonRootOverlapError(
                "Material-reference overlap integration is unresolved at binary64 reference-cell resolution.",
                supermesh_owner.CommonRefinementStatus.PREDICATE_UNCERTAIN,
                integration_error=error,
                integration_tolerance=tolerance,
                unresolved_reference_tetrahedron=unresolved,
            )
        parent = remove_leaf(selected)[0]
        for child in _subdivide_tetrahedron(parent):
            add_leaf(record(child))


def _affine_reference_preimages(
    chart_arguments: Sequence[Expression], pieces: Sequence[Tet], /
) -> tuple[Tet, ...] | None:
    """Exact endpoint-reference preimages of root pieces under an affine chart.

    Returns ``None`` when the chart is not affine with a rational invertible
    matrix or a piece leaves the endpoint reference tetrahedron.
    """
    affine = affine_map(tuple(chart_arguments), 3)
    if affine is None:
        return None
    offsets, rows = affine
    if not all(
        isinstance(value, Fraction) for value in (*offsets, *(v for r in rows for v in r))
    ):
        return None
    preimages: list[Tet] = []
    for piece in pieces:
        local: list[Point] = []
        for point in piece:
            try:
                solved = solve_exact(
                    rows, tuple((point[axis] - offsets[axis],) for axis in range(3))
                )
            except ValueError:
                return None
            values: list[Fraction] = []
            for axis in range(3):
                value = solved[axis][0]
                if not isinstance(value, Fraction):
                    return None
                values.append(value)
            coordinates = _as_point(values)
            if (
                any(value < 0 for value in coordinates)
                or sum(coordinates, Fraction(0)) > 1
            ):
                return None
            local.append(coordinates)
        preimages.append(_as_tetrahedron(local))
    return tuple(preimages)


def _integrate_pulled_reference_overlap(
    chart_arguments: Sequence[Expression],
    physical_jacobian: Expression,
    reference_pieces: Sequence[Tet],
    policy: CurvedOverlapIntegrationPolicy,
    rule_points: np.ndarray,
    rule_weights: np.ndarray,
    /,
) -> tuple[float, float, int, int]:
    """Integrate a nonnegative endpoint Jacobian over clipped reference pieces.

    An affine endpoint chart pulls every piece back exactly to an endpoint
    reference tetrahedron, integrated exactly where the Jacobian is polynomial;
    other charts classify dyadic endpoint reference leaves against the pieces.
    """
    if not reference_pieces:
        return 0.0, 0.0, 0, 0
    preimages = _affine_reference_preimages(chart_arguments, reference_pieces)
    if preimages is not None:
        values, errors, exact_count, enclosed_count = _integrate_piece_bank(
            preimages, (physical_jacobian,), policy, rule_points, rule_weights
        )
        return float(values[0]), float(errors[0]), exact_count, enclosed_count

    def classify(vertices: Tet) -> str:
        local_arguments = _chart_arguments(vertices)
        mapped = tuple(
            compose_expression(expression, local_arguments)
            for expression in chart_arguments
        )
        possible = False
        for piece in reference_pieces:
            contained = True
            excluded = False
            for plane in _tetrahedron_planes(piece):
                inequality = _reference_plane_expression(plane, mapped)
                controls = _bernstein_effective_controls(inequality, "simplex", 3)
                if controls is not None and max(controls) <= 0:
                    continue
                if controls is not None and min(controls) >= 0:
                    excluded = True
                    break
                lower, upper = polynomial_bounds(inequality, "simplex", 3)
                if upper > 0.0:
                    contained = False
                if lower >= 0.0:
                    excluded = True
                    break
            # A leaf excluded by one plane is outside this piece even when the
            # planes checked before it contained the leaf.
            if contained and not excluded:
                return "inside"
            possible |= not excluded
        return "boundary" if possible else "outside"

    return _integrate_classified_overlap(
        physical_jacobian, policy, rule_points, rule_weights, classify
    )


def _triangular_chart_vertical_bounds(
    chart_arguments: Sequence[Expression], /
) -> tuple[Expression, Expression] | None:
    axes_3d = axes(3)
    if not (
        expression_equal(chart_arguments[0], axes_3d[0])
        and expression_equal(chart_arguments[1], axes_3d[1])
    ):
        return None
    axes_2d = axes(2)
    top_height = sum_polynomials(
        (constant(1, 2), scale(axes_2d[0], -1), scale(axes_2d[1], -1)),
        2,
    )
    bottom = compose_expression(
        chart_arguments[2],
        (axes_2d[0], axes_2d[1], constant(0, 2)),
    )
    top = compose_expression(chart_arguments[2], (axes_2d[0], axes_2d[1], top_height))
    return bottom, top


def _triangular_chart_membership_inequalities(
    source_chart: Sequence[Expression],
    target_chart: Sequence[Expression],
    /,
) -> tuple[Expression, Expression] | None:
    target_bounds = _triangular_chart_vertical_bounds(target_chart)
    if target_bounds is None or _triangular_chart_vertical_bounds(source_chart) is None:
        return None
    target_bottom, target_top = target_bounds
    source_x, source_y, source_z = source_chart
    target_bottom_at_source = compose_expression(target_bottom, (source_x, source_y))
    target_top_at_source = compose_expression(target_top, (source_x, source_y))
    return (
        sum_polynomials((target_bottom_at_source, scale(source_z, -1)), 3),
        sum_polynomials((source_z, scale(target_top_at_source, -1)), 3),
    )


def _classify_exact_inequalities(
    inequalities: Sequence[Expression], vertices: Tet, /
) -> str:
    local_arguments = _chart_arguments(vertices)
    for inequality in inequalities:
        local = compose_expression(inequality, local_arguments)
        controls = _bernstein_effective_controls(local, "simplex", 3)
        if controls is not None and max(controls) <= 0:
            continue
        if controls is not None and min(controls) >= 0 and max(controls) > 0:
            return "outside"
        lower, upper = polynomial_bounds(local, "simplex", 3)
        if upper <= 0.0:
            continue
        if lower > 0.0:
            return "outside"
        return "boundary"
    return "inside"


def _integrate_triangular_chart_overlap(
    source_chart: Sequence[Expression],
    target_chart: Sequence[Expression],
    physical_jacobian: Expression,
    policy: CurvedOverlapIntegrationPolicy,
    rule_points: np.ndarray,
    rule_weights: np.ndarray,
    /,
) -> tuple[float, float, int, int]:
    inequalities = _triangular_chart_membership_inequalities(source_chart, target_chart)
    if inequalities is None:
        raise ValueError(
            "Adaptive curved-image intersection requires compatible vertical triangular charts."
        )
    return _integrate_classified_overlap(
        physical_jacobian,
        policy,
        rule_points,
        rule_weights,
        lambda vertices: _classify_exact_inequalities(inequalities, vertices),
    )


def _triangular_chart_contains(
    source_chart: Sequence[Expression], target_chart: Sequence[Expression], /
) -> bool:
    inequalities = _triangular_chart_membership_inequalities(source_chart, target_chart)
    return (
        inequalities is not None
        and _classify_exact_inequalities(inequalities, _canonical_reference_tetrahedron())
        == "inside"
    )


def _box_expression_bounds(
    expression: Expression,
    lower: tuple[Fraction, Fraction, Fraction],
    upper: tuple[Fraction, Fraction, Fraction],
    /,
) -> tuple[float, float]:
    variables = axes(3)
    arguments = tuple(
        sum_polynomials(
            (
                constant(lower[axis], 3),
                scale(variables[axis], upper[axis] - lower[axis]),
            ),
            3,
        )
        for axis in range(3)
    )
    return polynomial_bounds(compose_expression(expression, arguments), "box", 3)


def _parametric_inverse_inclusion(
    source_chart: Sequence[Expression],
    source_vertices: Tet,
    target_chart: Sequence[Expression],
    /,
) -> bool:
    """Prove every source image point has an inverse inside the target simplex."""
    from ..discretization._mapped_locator import (
        _interval_add,
        _interval_multiply,
        _interval_subtract,
    )

    source_arguments = _chart_arguments(source_vertices)
    source_bounds = tuple(
        polynomial_bounds(compose_expression(expression, source_arguments), "simplex", 3)
        for expression in source_chart
    )
    if any(not np.all(np.isfinite(bounds)) for bounds in source_bounds):
        return False
    center = _as_point(
        tuple(
            sum((point[axis] for point in source_vertices), Fraction(0)) / 4
            for axis in range(3)
        )
    )
    source_value = np.asarray(
        [float(evaluate_expression(expression, center)) for expression in source_chart],
        dtype=np.float64,
    )
    target_value = np.asarray(source_value)
    target_reference = np.full((3,), 0.25, dtype=np.float64)
    try:
        for _ in range(16):
            exact_reference = _as_point(
                tuple(Fraction.from_float(float(value)) for value in target_reference)
            )
            mapped = np.asarray(
                [
                    float(evaluate_expression(expression, exact_reference))
                    for expression in target_chart
                ],
                dtype=np.float64,
            )
            jacobian = np.asarray(
                [
                    [
                        float(
                            evaluate_expression(
                                derivative(expression, axis), exact_reference
                            )
                        )
                        for axis in range(3)
                    ]
                    for expression in target_chart
                ],
                dtype=np.float64,
            )
            step = np.linalg.solve(jacobian, mapped - target_value)
            target_reference -= step
            if not np.all(np.isfinite(target_reference)):
                return False
            if float(np.linalg.norm(step, ord=np.inf)) <= 8.0 * np.finfo(float).eps:
                break
        exact_reference = _as_point(
            tuple(Fraction.from_float(float(value)) for value in target_reference)
        )
        jacobian = np.asarray(
            [
                [
                    float(
                        evaluate_expression(derivative(expression, axis), exact_reference)
                    )
                    for axis in range(3)
                ]
                for expression in target_chart
            ],
            dtype=np.float64,
        )
        inverse = np.linalg.solve(jacobian, np.eye(3))
    except (ArithmeticError, FloatingPointError, np.linalg.LinAlgError, ValueError):
        return False
    if not np.all(np.isfinite(inverse)):
        return False
    exact_inverse = tuple(
        tuple(Fraction.from_float(float(value)) for value in row) for row in inverse
    )
    if determinant_exact(exact_inverse) == 0:
        return False

    source_radii = np.asarray(
        [
            max(
                abs(bounds[0] - source_value[axis]),
                abs(bounds[1] - source_value[axis]),
            )
            for axis, bounds in enumerate(source_bounds)
        ],
        dtype=np.float64,
    )
    radius = 4.0 * (np.abs(inverse) @ source_radii)
    radius += 128.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(target_reference))
    lower = _as_point(
        tuple(
            Fraction.from_float(float(np.nextafter(value - extent, -np.inf)))
            for value, extent in zip(target_reference, radius, strict=True)
        )
    )
    upper = _as_point(
        tuple(
            Fraction.from_float(float(np.nextafter(value + extent, np.inf)))
            for value, extent in zip(target_reference, radius, strict=True)
        )
    )
    if any(value <= 0 for value in lower) or sum(upper, Fraction(0)) >= 1:
        return False

    point_bounds = tuple(
        _box_expression_bounds(expression, exact_reference, exact_reference)
        for expression in target_chart
    )
    jacobian_bounds = tuple(
        tuple(
            _box_expression_bounds(derivative(expression, axis), lower, upper)
            for axis in range(3)
        )
        for expression in target_chart
    )
    if any(
        not np.all(np.isfinite(bounds))
        for bounds in (
            *point_bounds,
            *(value for row in jacobian_bounds for value in row),
        )
    ):
        return False
    residual = tuple(
        _interval_subtract(
            point_bounds[axis],
            (float(source_bounds[axis][0]), float(source_bounds[axis][1])),
        )
        for axis in range(3)
    )
    krawczyk_matrix = []
    for row in range(3):
        matrix_row = []
        for column in range(3):
            product = (0.0, 0.0)
            for inner in range(3):
                product = _interval_add(
                    product,
                    _interval_multiply(
                        (float(inverse[row, inner]), float(inverse[row, inner])),
                        jacobian_bounds[inner][column],
                    ),
                )
            identity = (1.0, 1.0) if row == column else (0.0, 0.0)
            matrix_row.append(_interval_subtract(identity, product))
        krawczyk_matrix.append(tuple(matrix_row))
    center_image = []
    for row in range(3):
        correction = (0.0, 0.0)
        for inner in range(3):
            correction = _interval_add(
                correction,
                _interval_multiply(
                    (float(inverse[row, inner]), float(inverse[row, inner])),
                    residual[inner],
                ),
            )
        center_interval = (float(target_reference[row]), float(target_reference[row]))
        center_image.append(_interval_subtract(center_interval, correction))
    for row in range(3):
        image = center_image[row]
        for column in range(3):
            displacement = _interval_subtract(
                (float(lower[column]), float(upper[column])),
                (
                    float(target_reference[column]),
                    float(target_reference[column]),
                ),
            )
            image = _interval_add(
                image,
                _interval_multiply(krawczyk_matrix[row][column], displacement),
            )
        if not (image[0] > float(lower[row]) and image[1] < float(upper[row])):
            return False
    return True


def _chart_image_box_disjoint(
    image_bounds: Sequence[tuple[float, float]],
    chart_arguments: Sequence[Expression],
    source_vertices: Tet,
    rule_points: np.ndarray,
    /,
) -> bool:
    source_scale = max(
        float(np.sqrt(sum(float(a - b) ** 2 for a, b in zip(first, second, strict=True))))
        for first, second in combinations(source_vertices, 2)
    )
    stack = [_canonical_reference_tetrahedron()]
    while stack:
        vertices = stack.pop()
        local_arguments = _chart_arguments(vertices)
        bounds = tuple(
            polynomial_bounds(
                compose_expression(expression, local_arguments), "simplex", 3
            )
            for expression in chart_arguments
        )
        if all(
            np.isfinite(low)
            and np.isfinite(high)
            and (high < image_bounds[axis][0] or image_bounds[axis][1] < low)
            for axis, (low, high) in enumerate(bounds)
        ):
            continue
        target_scale = max(
            float(
                np.sqrt(
                    sum(float(a - b) ** 2 for a, b in zip(first, second, strict=True))
                )
            )
            for first, second in combinations(vertices, 2)
        )
        if target_scale <= source_scale or not _has_binary64_tetrahedron_interior(
            vertices, rule_points
        ):
            return False
        stack.extend(reversed(_subdivide_tetrahedron(vertices)))
    return True


def _exact_affine_target_reference_preimage(
    source_chart: Sequence[Expression],
    target_chart: Sequence[Expression],
    source_vertices: Tet,
    target_vertices: Tet,
    /,
) -> Tet | None:
    target_origin = target_vertices[0]
    matrix = tuple(
        tuple(
            target_vertices[column + 1][row] - target_origin[row] for column in range(3)
        )
        for row in range(3)
    )
    local_vertices = []
    for point in source_vertices:
        right = tuple((point[axis] - target_origin[axis],) for axis in range(3))
        try:
            solved = solve_exact(matrix, right)
        except ValueError:
            return None
        local = _as_point(tuple(solved[axis][0] for axis in range(3)))
        if any(value < 0 for value in local) or sum(local, Fraction(0)) > 1:
            return None
        local_vertices.append(local)
    local_tet = _as_tetrahedron(local_vertices)
    if _determinant3(local_tet) <= 0:
        return None
    arguments = _affine_arguments_from_vertices(local_tet)
    if all(
        expression_equal(compose_expression(target, arguments), source)
        for source, target in zip(source_chart, target_chart, strict=True)
    ):
        return local_tet
    return None


def _tetrahedron_local_points(
    points: Sequence[Point], vertices: Tet, /
) -> tuple[Point, ...] | None:
    """Exact local coordinates of points in the affine frame of one tetrahedron."""
    origin = vertices[0]
    matrix = tuple(
        tuple(vertices[column + 1][row] - origin[row] for column in range(3))
        for row in range(3)
    )
    local: list[Point] = []
    for point in points:
        right = tuple((point[axis] - origin[axis],) for axis in range(3))
        try:
            solved = solve_exact(matrix, right)
        except ValueError:
            return None
        values = tuple(solved[axis][0] for axis in range(3))
        if not all(isinstance(value, Fraction) for value in values):
            return None
        local.append(_as_point(values))
    return tuple(local)


def _shared_chart_reference_pieces(
    source_chart: Sequence[Expression],
    target_chart: Sequence[Expression],
    source_vertices: Tet,
    target_vertices: Tet,
    /,
) -> tuple[tuple[Tet, ...], tuple[Tet, ...]] | None:
    """Exact overlap pieces of two cells whose charts differ by their corner frames.

    When the target chart is the source chart composed with the exact affine
    map between the two corner frames, both charts are one map over the root
    reference and their overlap is the exact clip of the corner tetrahedra. The
    pieces are returned in source- and target-local reference coordinates;
    ``None`` means the charts are not so related.
    """
    target_local = _tetrahedron_local_points(target_vertices, source_vertices)
    if target_local is None:
        return None
    target_in_source = _as_tetrahedron(target_local)
    if _determinant3(target_in_source) <= 0:
        return None
    arguments = _affine_arguments_from_vertices(target_in_source)
    if not all(
        expression_equal(compose_expression(source, arguments), target)
        for source, target in zip(source_chart, target_chart, strict=True)
    ):
        return None
    source_pieces = tuple(
        piece
        for piece in _clip_exact_tetrahedra(
            _canonical_reference_tetrahedron(), target_in_source
        )
        if _tetrahedron_measure(piece)
    )
    target_pieces = []
    for piece in source_pieces:
        local = _tetrahedron_local_points(piece, target_in_source)
        if local is None:
            return None
        target_pieces.append(_as_tetrahedron(local))
    return source_pieces, tuple(target_pieces)


def _integrate_general_chart_overlap(
    source_chart: Sequence[Expression],
    target_chart: Sequence[Expression],
    integrand: Expression,
    target_bounds: Sequence[tuple[float, float]],
    policy: CurvedOverlapIntegrationPolicy,
    rule_points: np.ndarray,
    rule_weights: np.ndarray,
    /,
) -> tuple[float, float, int, int]:
    def classify(vertices: Tet) -> str:
        source_arguments = _chart_arguments(vertices)
        source_bounds = tuple(
            polynomial_bounds(
                compose_expression(expression, source_arguments), "simplex", 3
            )
            for expression in source_chart
        )
        if any(not np.all(np.isfinite(bounds)) for bounds in source_bounds):
            return "boundary"
        if any(
            bounds[1] < target[0] or target[1] < bounds[0]
            for bounds, target in zip(source_bounds, target_bounds, strict=True)
        ):
            return "outside"
        if _chart_image_box_disjoint(source_bounds, target_chart, vertices, rule_points):
            return "outside"
        if _parametric_inverse_inclusion(source_chart, vertices, target_chart):
            return "inside"
        return "boundary"

    return _integrate_classified_overlap(
        integrand, policy, rule_points, rule_weights, classify
    )


def _triangular_partition_covers_reference(records: Sequence[_CellRef], /) -> bool:
    if len(records) != 2 or records[0].root_id != records[1].root_id:
        return False
    first = _triangular_chart_vertical_bounds(records[0].chart_arguments)
    second = _triangular_chart_vertical_bounds(records[1].chart_arguments)
    if first is None or second is None:
        return False
    zero = constant(0, 2)
    axes_2d = axes(2)
    top = sum_polynomials(
        (constant(1, 2), scale(axes_2d[0], -1), scale(axes_2d[1], -1)),
        2,
    )
    return any(
        expression_equal(lower_bounds[0], zero)
        and expression_equal(lower_bounds[1], upper_bounds[0])
        and expression_equal(upper_bounds[1], top)
        for lower_bounds, upper_bounds in ((first, second), (second, first))
    )


def _triangular_side_covers_reference(records: Sequence[_CellRef], /) -> bool:
    if len(records) == 1:
        return (
            records[0].corner_image
            and records[0].vertices == _canonical_reference_tetrahedron()
        )
    return _triangular_partition_covers_reference(records)


def _triangular_affine_vertical_data(
    chart: Sequence[Expression], /
) -> tuple[Expression, Expression, Expression] | None:
    bounds = _triangular_chart_vertical_bounds(chart)
    if bounds is None:
        return None
    derivative_z = derivative(chart[2], 2)
    axes_2d = axes(2)
    slope = compose_expression(derivative_z, (axes_2d[0], axes_2d[1], constant(0, 2)))
    lifted = compose_expression(slope, (axes(3)[0], axes(3)[1]))
    if not expression_equal(lifted, derivative_z):
        return None
    controls = _bernstein_effective_controls(slope, "simplex", 2)
    if controls is None or min(controls) <= 0:
        return None
    return bounds[0], bounds[1], slope


def _exact_bernstein_sign(expression: Expression, dimension: int, /) -> int | None:
    controls = _bernstein_effective_controls(expression, "simplex", dimension)
    if controls is None:
        return None
    if max(controls) <= 0:
        return -1
    if min(controls) >= 0:
        return 1
    return None


def _triangular_affine_overlap_slab(
    source_chart: Sequence[Expression],
    target_chart: Sequence[Expression],
    /,
) -> tuple[str, Expression, Expression] | None:
    source = _triangular_affine_vertical_data(source_chart)
    target = _triangular_affine_vertical_data(target_chart)
    if source is None or target is None:
        return None
    source_bottom, source_top, source_slope = source
    target_bottom, target_top, _ = target
    lower_difference = sum_polynomials((target_bottom, scale(source_bottom, -1)), 2)
    lower_sign = _exact_bernstein_sign(lower_difference, 2)
    if lower_sign == -1:
        lower = constant(0, 2)
    elif lower_sign == 1:
        lower = multiply(lower_difference, power(source_slope, -1))
    else:
        return None
    axes_2d = axes(2)
    base_height = sum_polynomials(
        (constant(1, 2), scale(axes_2d[0], -1), scale(axes_2d[1], -1)),
        2,
    )
    upper_difference = sum_polynomials((target_top, scale(source_bottom, -1)), 2)
    upper_relative_to_cell = sum_polynomials(
        (upper_difference, scale(multiply(source_slope, base_height), -1)), 2
    )
    upper_sign = _exact_bernstein_sign(upper_relative_to_cell, 2)
    if upper_sign == 1:
        upper = base_height
    elif upper_sign == -1:
        upper = multiply(upper_difference, power(source_slope, -1))
    else:
        return None
    width = sum_polynomials((upper, scale(lower, -1)), 2)
    width_controls = _bernstein_effective_controls(width, "simplex", 2)
    if width_controls is None:
        return None
    if max(width_controls) <= 0:
        return "empty", lower, upper
    if min(width_controls) >= 0 and max(width_controls) > 0:
        return "positive", lower, upper
    return None


def _canonical_reference_prism_tetrahedra() -> tuple[Tet, Tet, Tet]:
    p0 = (Fraction(0), Fraction(0), Fraction(0))
    p1 = (Fraction(1), Fraction(0), Fraction(0))
    p2 = (Fraction(0), Fraction(1), Fraction(0))
    p3 = (Fraction(0), Fraction(0), Fraction(1))
    p4 = (Fraction(1), Fraction(0), Fraction(1))
    p5 = (Fraction(0), Fraction(1), Fraction(1))
    return (p0, p1, p2, p5), (p0, p1, p4, p5), (p0, p3, p4, p5)


def _exact_constant_polynomial_ratio(
    numerator: Mapping[tuple[int, ...], ExactScalar],
    denominator: Mapping[tuple[int, ...], ExactScalar],
    /,
) -> Fraction | None:
    def rational_terms(
        expression: Mapping[tuple[int, ...], ExactScalar],
    ) -> dict[tuple[int, ...], Fraction] | None:
        terms: dict[tuple[int, ...], Fraction] = {}
        for exponent, coefficient in expression.items():
            if isinstance(coefficient, Fraction):
                terms[exponent] = coefficient
            elif type(coefficient) is int:
                terms[exponent] = Fraction(coefficient)
            else:
                return None
        return terms

    numerator_terms = rational_terms(numerator)
    denominator_terms = rational_terms(denominator)
    if numerator_terms is None or denominator_terms is None or not denominator_terms:
        return None
    exponents = numerator_terms.keys() | denominator_terms.keys()
    first = next(iter(denominator_terms))
    ratio = numerator_terms.get(first, Fraction(0)) / denominator_terms[first]
    if all(
        numerator_terms.get(exponent, Fraction(0))
        == ratio * denominator_terms.get(exponent, Fraction(0))
        for exponent in exponents
    ):
        return ratio
    return None


def _integrate_triangular_overlap_slab(
    integrand: Expression,
    lower: Expression,
    upper: Expression,
    policy: CurvedOverlapIntegrationPolicy,
    rule_points: np.ndarray,
    rule_weights: np.ndarray,
    /,
) -> tuple[float, float, int, int]:
    variables = axes(3)
    lower_3d = compose_expression(lower, (variables[0], variables[1]))
    upper_3d = compose_expression(upper, (variables[0], variables[1]))
    width = sum_polynomials((upper_3d, scale(lower_3d, -1)), 3)
    coordinate = sum_polynomials((lower_3d, multiply(width, variables[2])), 3)
    pulled_density = compose_expression(
        integrand, (variables[0], variables[1], coordinate)
    )
    if isinstance(width, RationalPolynomial) and isinstance(pulled_density, Mapping):
        ratio = _exact_constant_polynomial_ratio(pulled_density, width.denominator)
        pulled = (
            scale(width.numerator, ratio)
            if ratio is not None
            else multiply(pulled_density, width)
        )
    else:
        pulled = multiply(pulled_density, width)
    relative = max(0.0, float(np.nextafter(policy.relative_tolerance / 3.0, -np.inf)))
    absolute = max(0.0, float(np.nextafter(policy.absolute_tolerance / 3.0, -np.inf)))
    if relative == 0.0:
        raise CommonRootOverlapError(
            "Curved overlap prism integration tolerance is below binary64 resolution.",
            supermesh_owner.CommonRefinementStatus.PREDICATE_UNCERTAIN,
            integration_tolerance=policy.absolute_tolerance,
        )
    subpolicy = CurvedOverlapIntegrationPolicy(
        relative_tolerance=relative, absolute_tolerance=absolute
    )
    value_total = Fraction(0)
    error_total = Fraction(0)
    exact_count = enclosed_count = 0
    for tetrahedron in _canonical_reference_prism_tetrahedra():
        value, error, exact = _integrate_expression(
            pulled, tetrahedron, subpolicy, rule_points, rule_weights
        )
        value_total += Fraction(float(value))
        error_total += Fraction(float(error))
        exact_count += int(exact)
        enclosed_count += int(not exact)
    rounded_value = float(value_total)
    rounded_error = error_total + abs(value_total - Fraction(rounded_value))
    return (
        rounded_value,
        coefficient_endpoint(rounded_error, upper=True),
        exact_count,
        enclosed_count,
    )


def _material_coverage_diagnostics(
    rows: np.ndarray,
    overlap_values: np.ndarray,
    overlap_errors: np.ndarray,
    cell_values: np.ndarray,
    cell_errors: np.ndarray,
    relative_tolerance: float,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covered = [Fraction(0) for _ in cell_values]
    errors = [Fraction(0) for _ in cell_values]
    for row, value, error in zip(rows, overlap_values, overlap_errors, strict=True):
        index = int(row)
        covered[index] += Fraction(float(value))
        errors[index] += Fraction(float(error))
    defects = np.empty_like(cell_values)
    tolerances = np.empty_like(cell_values)
    error_bounds = np.empty_like(cell_values)
    relative = Fraction(float(relative_tolerance))
    for row, (measure, cell_error, overlap_error) in enumerate(
        zip(cell_values, cell_errors, errors, strict=True)
    ):
        defect = covered[row] - Fraction(float(measure))
        defects[row] = (
            0.0 if defect == 0 else coefficient_endpoint(defect, upper=defect > 0)
        )
        tolerance = (
            relative * Fraction(float(measure))
            + Fraction(float(cell_error))
            + overlap_error
        )
        tolerances[row] = (
            0.0 if tolerance == 0 else coefficient_endpoint(tolerance, upper=True)
        )
        error_bounds[row] = (
            0.0 if overlap_error == 0 else coefficient_endpoint(overlap_error, upper=True)
        )
    return defects, tolerances, error_bounds


def prepare_material_reference_volume_overlap(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    reference_witness: MaterialReferenceVolumeWitness,
    /,
    *,
    policy: supermesh_owner.CommonRefinementPolicy,
    integration_policy: CurvedOverlapIntegrationPolicy,
) -> MaterialReferenceVolumeOverlap:
    """Prepare a sparse material-reference overlap with independent physical Jacobians.

    Certified nonlinear/rational charts use exact corner-image and triangular
    slab fast paths where applicable. General intersections classify whole
    parameter subtetrahedra by Bernstein image exclusion and parametric
    Krawczyk inverse inclusion; unresolved boundary leaves carry a rigorous
    integral enclosure through finite binary64 resolution.
    """
    if not isinstance(reference_witness, MaterialReferenceVolumeWitness):
        raise TypeError("reference_witness must be MaterialReferenceVolumeWitness.")
    if not isinstance(policy, supermesh_owner.CommonRefinementPolicy) or not isinstance(
        integration_policy, CurvedOverlapIntegrationPolicy
    ):
        raise TypeError(
            "Material overlap requires common-refinement and integration policies."
        )
    if (
        policy.coverage is not supermesh_owner.CommonRefinementCoverage.COMPLETE
        or policy.overlap_simplices
    ):
        raise ValueError(
            "Material-reference overlap requires complete coverage without physical overlap simplices."
        )
    witness = reference_witness.revalidate(
        reference_witness.root_mesh,
        reference_witness.root_geometry,
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
    )
    root_validity = witness.root_validity
    root_validity.require_current(witness.root_geometry, mesh=witness.root_mesh)
    witness.source_validity.require_current(source_geometry, mesh=source_mesh)
    witness.target_validity.require_current(target_geometry, mesh=target_mesh)
    _root_cell_rows(witness.root_mesh)
    root_elements, root_routes, _ = witness.root_geometry.resolve(witness.root_mesh)
    source_elements, source_routes, _ = source_geometry.resolve(source_mesh)
    target_elements, target_routes, _ = target_geometry.resolve(target_mesh)
    root_expressions = _coordinate_expression_rows(
        witness.root_mesh, witness.root_geometry, root_elements, root_routes
    )
    source_expressions = _coordinate_expression_rows(
        source_mesh, source_geometry, source_elements, source_routes
    )
    target_expressions = _coordinate_expression_rows(
        target_mesh, target_geometry, target_elements, target_routes
    )
    root_determinants = tuple(
        _coordinate_jacobian_determinant(row) for row in root_expressions
    )
    source_determinants = tuple(
        _coordinate_jacobian_determinant(row) for row in source_expressions
    )
    target_determinants = tuple(
        _coordinate_jacobian_determinant(row) for row in target_expressions
    )
    source_records = _cell_records(
        source_mesh, witness.source_reference_maps, allow_curved=True
    )
    target_records = _cell_records(
        target_mesh, witness.target_reference_maps, allow_curved=True
    )
    curved_images = any(
        not record.corner_image or affine_map(tuple(record.chart_arguments), 3) is None
        for record in (*source_records, *target_records)
    )
    reference_overlap_pieces: dict[tuple[int, int], tuple[Tet, ...]] = {}
    reference_overlap_exact: dict[tuple[int, int], Fraction] = {}
    curved_partitioned = False
    curved_general = False
    curved_exact_cover = False
    if curved_images:
        same_chart = (
            len(source_records) == 1
            and len(target_records) == 1
            and source_records[0].root_id == target_records[0].root_id
            and all(
                expression_equal(source, target)
                for source, target in zip(
                    source_records[0].chart_arguments,
                    target_records[0].chart_arguments,
                    strict=True,
                )
            )
        )
        shared_root = (
            len({record.root_id for record in source_records}) == 1
            and len({record.root_id for record in target_records}) == 1
            and source_records[0].root_id == target_records[0].root_id
        )
        exact_root_cover = (
            shared_root
            and _triangular_side_covers_reference(source_records)
            and _triangular_side_covers_reference(target_records)
        )
        triangular_overlap_profile = all(
            _triangular_chart_vertical_bounds(record.chart_arguments) is not None
            for record in (*source_records, *target_records)
        )
        if same_chart:
            curved_exact_cover = True
            source_self_candidates = target_self_candidates = 0
            candidates = ((0, 0),)
            source_self_overlap = target_self_overlap = None
            unit_measure = _tetrahedron_measure(_canonical_reference_tetrahedron())
            reference_overlap_pieces[(0, 0)] = ()
            reference_overlap_exact[(0, 0)] = unit_measure
            source_reference_exact = (unit_measure,)
            target_reference_exact = (unit_measure,)
        elif (
            exact_root_cover
            and triangular_overlap_profile
            and (len(source_records) > 1 or len(target_records) > 1)
        ):
            curved_partitioned = True
            curved_exact_cover = True
            source_self_candidates = target_self_candidates = 0
            source_self_overlap = target_self_overlap = None
            candidates = _candidate_pairs(source_records, target_records)
            reference_overlap_pieces.update({pair: () for pair in candidates})
        else:
            curved_general = True
            curved_exact_cover = exact_root_cover
            source_self_candidates = target_self_candidates = 0
            source_self_overlap = target_self_overlap = None
            candidates = _candidate_pairs(source_records, target_records)
            reference_overlap_pieces.update({pair: () for pair in candidates})
    else:
        source_self_candidates, source_self_overlap = _reference_self_overlap(
            source_records
        )
        target_self_candidates, target_self_overlap = _reference_self_overlap(
            target_records
        )
        if source_self_overlap is not None or target_self_overlap is not None:
            source_pairs = () if source_self_overlap is None else source_self_overlap
            target_pairs = () if target_self_overlap is None else target_self_overlap
            raise CommonRootOverlapError(
                "Material-reference endpoint image cells overlap in positive measure: "
                f"{(source_self_overlap, target_self_overlap)!r}.",
                supermesh_owner.CommonRefinementStatus.DOUBLE_COVERAGE,
                overlapping_source_rows=source_pairs,
                overlapping_target_rows=target_pairs,
            )
        candidates = _candidate_pairs(source_records, target_records)
        for source_row, target_row in candidates:
            source_record = source_records[source_row]
            target_record = target_records[target_row]
            pieces = _clip_exact_tetrahedra(
                source_record.vertices, target_record.vertices
            )
            if not pieces:
                continue
            measure = sum((_tetrahedron_measure(piece) for piece in pieces), Fraction(0))
            if measure:
                reference_overlap_pieces[(source_row, target_row)] = pieces
                reference_overlap_exact[(source_row, target_row)] = measure
        source_reference_exact = tuple(
            _tetrahedron_measure(record.vertices) for record in source_records
        )
        target_reference_exact = tuple(
            _tetrahedron_measure(record.vertices) for record in target_records
        )
    if curved_exact_cover:
        source_ref_defects = tuple(Fraction(0) for _ in source_records)
        target_ref_defects = tuple(Fraction(0) for _ in target_records)
        unmatched_source = np.empty((0,), dtype=np.int64)
        unmatched_target = np.empty((0,), dtype=np.int64)
    elif curved_general:
        source_ref_defects = tuple("interval-certified" for _ in source_records)
        target_ref_defects = tuple("interval-certified" for _ in target_records)
        unmatched_source = np.empty((0,), dtype=np.int64)
        unmatched_target = np.empty((0,), dtype=np.int64)
    else:
        source_covered_exact = [Fraction(0) for _ in source_records]
        target_covered_exact = [Fraction(0) for _ in target_records]
        for (source_row, target_row), value in reference_overlap_exact.items():
            source_covered_exact[source_row] += value
            target_covered_exact[target_row] += value
        source_ref_defects = tuple(
            covered - measure
            for covered, measure in zip(
                source_covered_exact, source_reference_exact, strict=True
            )
        )
        target_ref_defects = tuple(
            covered - measure
            for covered, measure in zip(
                target_covered_exact, target_reference_exact, strict=True
            )
        )
        unmatched_source = np.flatnonzero(
            np.asarray([value == 0 for value in source_covered_exact], dtype=np.bool_)
        )
        unmatched_target = np.flatnonzero(
            np.asarray([value == 0 for value in target_covered_exact], dtype=np.bool_)
        )
    source_exact_defects = tuple(
        value for value in source_ref_defects if isinstance(value, Fraction)
    )
    target_exact_defects = tuple(
        value for value in target_ref_defects if isinstance(value, Fraction)
    )
    exact_defects = tuple(
        value for value in (*source_exact_defects, *target_exact_defects) if value
    )
    if exact_defects:
        positive_defect = any(value > 0 for value in exact_defects)
        raise CommonRootOverlapError(
            "Material-reference common overlap has an exact source/target image coverage defect: "
            f"{(tuple(map(str, source_ref_defects)), tuple(map(str, target_ref_defects)), unmatched_source.tolist(), unmatched_target.tolist())!r}.",
            supermesh_owner.CommonRefinementStatus.DOUBLE_COVERAGE
            if positive_defect
            else supermesh_owner.CommonRefinementStatus.COVERAGE_GAP,
            source_reference_defects=source_exact_defects,
            target_reference_defects=target_exact_defects,
            unmatched_source_rows=unmatched_source.tolist(),
            unmatched_target_rows=unmatched_target.tolist(),
        )

    source_embedding = certify_global_embedding(
        source_mesh, source_geometry, witness.source_validity
    )
    target_embedding = certify_global_embedding(
        target_mesh, target_geometry, witness.target_validity
    )
    if source_embedding.status != "certified" or target_embedding.status != "certified":
        raise CommonRootOverlapError(
            "Material endpoint deformations lack computed global-injectivity evidence.",
            supermesh_owner.CommonRefinementStatus.PREDICATE_UNCERTAIN,
        )

    rule = cubature_rule_data("tetrahedron", 8)
    rule_points = np.asarray(rule.points, dtype=np.float64)
    rule_weights = np.asarray(rule.weights, dtype=np.float64)
    root_rows_ordered = _root_cell_rows(witness.root_mesh)

    def root_row(root_id: int) -> int:
        block_index, local_row = root_rows_ordered[root_id]
        return (
            sum(block.cell_count for block in witness.root_mesh.blocks[:block_index])
            + local_row
        )

    exact_integral_count = 0
    enclosed_integral_count = 0

    def integrate_root_pieces(pieces: Sequence[Tet], root_id: int) -> tuple[float, float]:
        nonlocal exact_integral_count, enclosed_integral_count
        values, errors, exact_count, enclosed_count = _integrate_piece_bank(
            pieces,
            (root_determinants[root_row(root_id)],),
            integration_policy,
            rule_points,
            rule_weights,
        )
        exact_integral_count += exact_count
        enclosed_integral_count += enclosed_count
        return float(values[0]), float(errors[0])

    standard_tetrahedron = _canonical_reference_tetrahedron()

    def integrate_root_chart_cell(record: _CellRef) -> tuple[float, float]:
        nonlocal exact_integral_count, enclosed_integral_count
        chart_determinant = _coordinate_jacobian_determinant(record.chart_arguments)
        root_row_index = root_row(record.root_id)
        composed_root_determinant = compose_expression(
            root_determinants[root_row_index], record.chart_arguments
        )
        integrand = multiply(composed_root_determinant, chart_determinant)
        value, error, exact = _integrate_expression(
            integrand,
            standard_tetrahedron,
            integration_policy,
            rule_points,
            rule_weights,
        )
        exact_integral_count += int(exact)
        enclosed_integral_count += int(not exact)
        return value, error

    source_reference_measures = np.zeros((len(source_records),), dtype=np.float64)
    target_reference_measures = np.zeros((len(target_records),), dtype=np.float64)
    source_reference_errors = np.zeros_like(source_reference_measures)
    target_reference_errors = np.zeros_like(target_reference_measures)
    for record in source_records:
        source_reference_measures[record.row], source_reference_errors[record.row] = (
            integrate_root_pieces((record.vertices,), record.root_id)
            if record.corner_image
            else integrate_root_chart_cell(record)
        )
    for record in target_records:
        target_reference_measures[record.row], target_reference_errors[record.row] = (
            integrate_root_pieces((record.vertices,), record.root_id)
            if record.corner_image
            else integrate_root_chart_cell(record)
        )
    source_physical_measures = np.zeros((len(source_records),), dtype=np.float64)
    target_physical_measures = np.zeros((len(target_records),), dtype=np.float64)
    source_physical_errors = np.zeros_like(source_physical_measures)
    target_physical_errors = np.zeros_like(target_physical_measures)
    for record in source_records:
        value, error, exact = _integrate_expression(
            source_determinants[record.row],
            standard_tetrahedron,
            integration_policy,
            rule_points,
            rule_weights,
        )
        source_physical_measures[record.row] = value
        source_physical_errors[record.row] = error
        exact_integral_count += int(exact)
        enclosed_integral_count += int(not exact)
    for record in target_records:
        value, error, exact = _integrate_expression(
            target_determinants[record.row],
            standard_tetrahedron,
            integration_policy,
            rule_points,
            rule_weights,
        )
        target_physical_measures[record.row] = value
        target_physical_errors[record.row] = error
        exact_integral_count += int(exact)
        enclosed_integral_count += int(not exact)
    if any(
        not np.all(np.isfinite(values)) or np.any(values <= 0.0)
        for values in (
            source_reference_measures,
            target_reference_measures,
            source_physical_measures,
            target_physical_measures,
        )
    ) or any(
        not np.all(np.isfinite(values)) or np.any(values < 0.0)
        for values in (
            source_reference_errors,
            target_reference_errors,
            source_physical_errors,
            target_physical_errors,
        )
    ):
        raise CommonRootOverlapError(
            "A material-reference cell volume or its error is not finitely representable.",
            supermesh_owner.CommonRefinementStatus.PREDICATE_UNCERTAIN,
            integration_error=float(
                max(
                    np.max(source_reference_errors, initial=0.0),
                    np.max(target_reference_errors, initial=0.0),
                    np.max(source_physical_errors, initial=0.0),
                    np.max(target_physical_errors, initial=0.0),
                )
            ),
        )

    overlap_source_rows: list[int] = []
    overlap_target_rows: list[int] = []
    reference_overlap_values: list[float] = []
    reference_overlap_errors: list[float] = []
    source_overlap_values: list[float] = []
    target_overlap_values: list[float] = []
    source_overlap_errors: list[float] = []
    target_overlap_errors: list[float] = []
    source_reference_omitted_bounds = np.zeros_like(source_reference_measures)
    target_reference_omitted_bounds = np.zeros_like(target_reference_measures)
    source_physical_omitted_bounds = np.zeros_like(source_physical_measures)
    target_physical_omitted_bounds = np.zeros_like(target_physical_measures)
    if curved_general:
        source_lower, source_upper = _reference_image_bounds(source_records)
        target_lower, target_upper = _reference_image_bounds(target_records)
        source_image_bounds = tuple(
            tuple(
                (float(source_lower[row, axis]), float(source_upper[row, axis]))
                for axis in range(3)
            )
            for row in range(len(source_records))
        )
        target_image_bounds = tuple(
            tuple(
                (float(target_lower[row, axis]), float(target_upper[row, axis]))
                for axis in range(3)
            )
            for row in range(len(target_records))
        )

    def integrate_curved_pair(
        chart: Sequence[Expression],
        other: _CellRef,
        integrand: Expression,
        other_bounds: Sequence[tuple[float, float]],
        /,
    ) -> tuple[float, float, int, int]:
        if other.corner_image:
            return _integrate_pulled_reference_overlap(
                chart,
                integrand,
                (other.vertices,),
                integration_policy,
                rule_points,
                rule_weights,
            )
        return _integrate_general_chart_overlap(
            chart,
            other.chart_arguments,
            integrand,
            other_bounds,
            integration_policy,
            rule_points,
            rule_weights,
        )

    for (source_row, target_row), pieces in sorted(
        reference_overlap_pieces.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        source_record = source_records[source_row]
        target_record = target_records[target_row]
        affine_preimage = (
            _exact_affine_target_reference_preimage(
                source_record.chart_arguments,
                target_record.chart_arguments,
                source_record.vertices,
                target_record.vertices,
            )
            if curved_general
            and not (source_record.corner_image and target_record.corner_image)
            else None
        )
        shared_pieces = (
            _shared_chart_reference_pieces(
                source_record.chart_arguments,
                target_record.chart_arguments,
                source_record.vertices,
                target_record.vertices,
            )
            if curved_general
            and affine_preimage is None
            and not (source_record.corner_image and target_record.corner_image)
            else None
        )
        if curved_general and source_record.corner_image and target_record.corner_image:
            exact_pieces = _clip_exact_tetrahedra(
                source_record.vertices, target_record.vertices
            )
            if not exact_pieces:
                continue
            exact_measure = sum(
                (_tetrahedron_measure(piece) for piece in exact_pieces),
                Fraction(0),
            )
            root_value, root_error = integrate_root_pieces(
                exact_pieces, source_record.root_id
            )
            source_value, source_error, exact_count, enclosed_count = (
                (
                    source_physical_measures[source_row],
                    source_physical_errors[source_row],
                    0,
                    0,
                )
                if exact_measure == _tetrahedron_measure(source_record.vertices)
                else _integrate_pulled_reference_overlap(
                    source_record.chart_arguments,
                    source_determinants[source_row],
                    exact_pieces,
                    integration_policy,
                    rule_points,
                    rule_weights,
                )
            )
            exact_integral_count += exact_count
            enclosed_integral_count += enclosed_count
            target_value, target_error, exact_count, enclosed_count = (
                (
                    target_physical_measures[target_row],
                    target_physical_errors[target_row],
                    0,
                    0,
                )
                if exact_measure == _tetrahedron_measure(target_record.vertices)
                else _integrate_pulled_reference_overlap(
                    target_record.chart_arguments,
                    target_determinants[target_row],
                    exact_pieces,
                    integration_policy,
                    rule_points,
                    rule_weights,
                )
            )
            exact_integral_count += exact_count
            enclosed_integral_count += enclosed_count
            if root_value <= root_error or any(
                value <= error
                for value, error in (
                    (source_value, source_error),
                    (target_value, target_error),
                )
            ):
                raise CommonRootOverlapError(
                    "A certified corner-image overlap is unresolved at the requested integration tolerance.",
                    supermesh_owner.CommonRefinementStatus.PREDICATE_UNCERTAIN,
                    integration_error=max(root_error, source_error, target_error),
                    unresolved_reference_tetrahedron=source_record.vertices,
                )
        elif affine_preimage is not None:
            root_value = source_reference_measures[source_row]
            root_error = source_reference_errors[source_row]
            source_value = source_physical_measures[source_row]
            source_error = source_physical_errors[source_row]
            if affine_preimage == _canonical_reference_tetrahedron():
                target_value = target_physical_measures[target_row]
                target_error = target_physical_errors[target_row]
            else:
                values, errors, exact_count, enclosed_count = _integrate_piece_bank(
                    (affine_preimage,),
                    (target_determinants[target_row],),
                    integration_policy,
                    rule_points,
                    rule_weights,
                )
                target_value = float(values[0])
                target_error = float(errors[0])
                exact_integral_count += exact_count
                enclosed_integral_count += enclosed_count
            if any(
                value <= error
                for value, error in (
                    (root_value, root_error),
                    (source_value, source_error),
                    (target_value, target_error),
                )
            ):
                raise CommonRootOverlapError(
                    "An exact affine inverse-chart inclusion is not finitely representable.",
                    supermesh_owner.CommonRefinementStatus.PREDICATE_UNCERTAIN,
                    integration_error=max(root_error, source_error, target_error),
                    unresolved_reference_tetrahedron=source_record.vertices,
                )
        elif shared_pieces is not None:
            source_pieces, target_pieces = shared_pieces
            if not source_pieces:
                continue
            root_integrand = multiply(
                compose_expression(
                    root_determinants[root_row(source_record.root_id)],
                    source_record.chart_arguments,
                ),
                _coordinate_jacobian_determinant(source_record.chart_arguments),
            )
            values, errors, exact_count, enclosed_count = _integrate_piece_bank(
                source_pieces,
                (root_integrand, source_determinants[source_row]),
                integration_policy,
                rule_points,
                rule_weights,
            )
            exact_integral_count += exact_count
            enclosed_integral_count += enclosed_count
            root_value, source_value = float(values[0]), float(values[1])
            root_error, source_error = float(errors[0]), float(errors[1])
            values, errors, exact_count, enclosed_count = _integrate_piece_bank(
                target_pieces,
                (target_determinants[target_row],),
                integration_policy,
                rule_points,
                rule_weights,
            )
            exact_integral_count += exact_count
            enclosed_integral_count += enclosed_count
            target_value, target_error = float(values[0]), float(errors[0])
            if any(
                value <= error
                for value, error in (
                    (root_value, root_error),
                    (source_value, source_error),
                    (target_value, target_error),
                )
            ):
                raise CommonRootOverlapError(
                    "An exact shared-chart overlap is not finitely representable.",
                    supermesh_owner.CommonRefinementStatus.PREDICATE_UNCERTAIN,
                    integration_error=max(root_error, source_error, target_error),
                    unresolved_reference_tetrahedron=source_record.vertices,
                )
        elif curved_general:
            root_integrand = multiply(
                compose_expression(
                    root_determinants[root_row(source_record.root_id)],
                    source_record.chart_arguments,
                ),
                _coordinate_jacobian_determinant(source_record.chart_arguments),
            )
            root_value, root_error, exact_count, enclosed_count = integrate_curved_pair(
                source_record.chart_arguments,
                target_record,
                root_integrand,
                target_image_bounds[target_row],
            )
            exact_integral_count += exact_count
            enclosed_integral_count += enclosed_count
            if root_value == 0.0 and root_error == 0.0:
                continue
            source_value, source_error, exact_count, enclosed_count = (
                integrate_curved_pair(
                    source_record.chart_arguments,
                    target_record,
                    source_determinants[source_row],
                    target_image_bounds[target_row],
                )
            )
            exact_integral_count += exact_count
            enclosed_integral_count += enclosed_count
            target_value, target_error, exact_count, enclosed_count = (
                integrate_curved_pair(
                    target_record.chart_arguments,
                    source_record,
                    target_determinants[target_row],
                    source_image_bounds[source_row],
                )
            )
            exact_integral_count += exact_count
            enclosed_integral_count += enclosed_count
            if (
                root_value <= root_error
                or source_value <= source_error
                or target_value <= target_error
            ):
                root_upper = float(
                    np.nextafter(max(root_value, 0.0) + root_error, np.inf)
                )
                source_upper = float(
                    np.nextafter(max(source_value, 0.0) + source_error, np.inf)
                )
                target_upper = float(
                    np.nextafter(max(target_value, 0.0) + target_error, np.inf)
                )
                source_reference_budget = float(
                    np.nextafter(
                        policy.coverage_tolerance * source_reference_measures[source_row]
                        + source_reference_errors[source_row],
                        np.inf,
                    )
                )
                target_reference_budget = float(
                    np.nextafter(
                        policy.coverage_tolerance * target_reference_measures[target_row]
                        + target_reference_errors[target_row],
                        np.inf,
                    )
                )
                source_physical_budget = float(
                    np.nextafter(
                        policy.coverage_tolerance * source_physical_measures[source_row]
                        + source_physical_errors[source_row],
                        np.inf,
                    )
                )
                target_physical_budget = float(
                    np.nextafter(
                        policy.coverage_tolerance * target_physical_measures[target_row]
                        + target_physical_errors[target_row],
                        np.inf,
                    )
                )
                if (
                    root_upper <= source_reference_budget
                    and root_upper <= target_reference_budget
                    and source_upper <= source_physical_budget
                    and target_upper <= target_physical_budget
                ):
                    source_reference_omitted_bounds[source_row] = np.nextafter(
                        source_reference_omitted_bounds[source_row] + root_upper,
                        np.inf,
                    )
                    target_reference_omitted_bounds[target_row] = np.nextafter(
                        target_reference_omitted_bounds[target_row] + root_upper,
                        np.inf,
                    )
                    source_physical_omitted_bounds[source_row] = np.nextafter(
                        source_physical_omitted_bounds[source_row] + source_upper,
                        np.inf,
                    )
                    target_physical_omitted_bounds[target_row] = np.nextafter(
                        target_physical_omitted_bounds[target_row] + target_upper,
                        np.inf,
                    )
                    continue
                raise CommonRootOverlapError(
                    "A general curved material-reference boundary remains unresolved at the declared coverage tolerance.",
                    supermesh_owner.CommonRefinementStatus.PREDICATE_UNCERTAIN,
                    integration_error=max(root_error, source_error, target_error),
                    integration_tolerance=max(
                        source_reference_budget,
                        target_reference_budget,
                        source_physical_budget,
                        target_physical_budget,
                    ),
                    unresolved_reference_tetrahedron=source_record.vertices,
                )
        elif curved_partitioned:
            source_slab = _triangular_affine_overlap_slab(
                source_record.chart_arguments, target_record.chart_arguments
            )
            target_slab = _triangular_affine_overlap_slab(
                target_record.chart_arguments, source_record.chart_arguments
            )
            if (source_slab is not None and source_slab[0] == "empty") or (
                target_slab is not None and target_slab[0] == "empty"
            ):
                continue
            source_image_is_full = _triangular_chart_contains(
                source_record.chart_arguments, target_record.chart_arguments
            )
            target_image_is_full = _triangular_chart_contains(
                target_record.chart_arguments, source_record.chart_arguments
            )
            if source_image_is_full:
                root_value = source_reference_measures[source_row]
                root_error = source_reference_errors[source_row]
            elif target_image_is_full:
                root_value = target_reference_measures[target_row]
                root_error = target_reference_errors[target_row]
            else:
                root_integrand = multiply(
                    compose_expression(
                        root_determinants[root_row(source_record.root_id)],
                        source_record.chart_arguments,
                    ),
                    _coordinate_jacobian_determinant(source_record.chart_arguments),
                )
                if source_slab is not None and source_slab[0] == "positive":
                    root_value, root_error, exact_count, enclosed_count = (
                        _integrate_triangular_overlap_slab(
                            root_integrand,
                            source_slab[1],
                            source_slab[2],
                            integration_policy,
                            rule_points,
                            rule_weights,
                        )
                    )
                else:
                    root_value, root_error, exact_count, enclosed_count = (
                        _integrate_triangular_chart_overlap(
                            source_record.chart_arguments,
                            target_record.chart_arguments,
                            root_integrand,
                            integration_policy,
                            rule_points,
                            rule_weights,
                        )
                    )
                exact_integral_count += exact_count
                enclosed_integral_count += enclosed_count
            if root_value == 0.0 and root_error == 0.0:
                continue
            if root_value <= root_error:
                raise CommonRootOverlapError(
                    "A curved material-reference pair is unresolved at the requested integration tolerance.",
                    supermesh_owner.CommonRefinementStatus.PREDICATE_UNCERTAIN,
                    integration_error=root_error,
                    integration_tolerance=(
                        integration_policy.absolute_tolerance
                        + integration_policy.relative_tolerance * root_value
                    ),
                    unresolved_reference_tetrahedron=source_record.vertices,
                )
            if source_image_is_full:
                source_value = source_physical_measures[source_row]
                source_error = source_physical_errors[source_row]
            elif source_slab is not None and source_slab[0] == "positive":
                source_value, source_error, exact_count, enclosed_count = (
                    _integrate_triangular_overlap_slab(
                        source_determinants[source_row],
                        source_slab[1],
                        source_slab[2],
                        integration_policy,
                        rule_points,
                        rule_weights,
                    )
                )
                exact_integral_count += exact_count
                enclosed_integral_count += enclosed_count
            else:
                source_value, source_error, exact_count, enclosed_count = (
                    _integrate_triangular_chart_overlap(
                        source_record.chart_arguments,
                        target_record.chart_arguments,
                        source_determinants[source_row],
                        integration_policy,
                        rule_points,
                        rule_weights,
                    )
                )
                exact_integral_count += exact_count
                enclosed_integral_count += enclosed_count
            if target_image_is_full:
                target_value = target_physical_measures[target_row]
                target_error = target_physical_errors[target_row]
            elif target_slab is not None and target_slab[0] == "positive":
                target_value, target_error, exact_count, enclosed_count = (
                    _integrate_triangular_overlap_slab(
                        target_determinants[target_row],
                        target_slab[1],
                        target_slab[2],
                        integration_policy,
                        rule_points,
                        rule_weights,
                    )
                )
                exact_integral_count += exact_count
                enclosed_integral_count += enclosed_count
            else:
                target_value, target_error, exact_count, enclosed_count = (
                    _integrate_triangular_chart_overlap(
                        target_record.chart_arguments,
                        source_record.chart_arguments,
                        target_determinants[target_row],
                        integration_policy,
                        rule_points,
                        rule_weights,
                    )
                )
                exact_integral_count += exact_count
                enclosed_integral_count += enclosed_count
            if any(
                value <= error
                for value, error in (
                    (source_value, source_error),
                    (target_value, target_error),
                )
            ):
                raise CommonRootOverlapError(
                    "A curved material-reference endpoint overlap is unresolved at the requested integration tolerance.",
                    supermesh_owner.CommonRefinementStatus.PREDICATE_UNCERTAIN,
                    integration_error=max(source_error, target_error),
                    unresolved_reference_tetrahedron=source_record.vertices,
                )
        else:
            exact_reference_measure = reference_overlap_exact[(source_row, target_row)]
            source_image_is_full = (
                exact_reference_measure == source_reference_exact[source_row]
            )
            target_image_is_full = (
                exact_reference_measure == target_reference_exact[target_row]
            )
            if source_image_is_full:
                root_value = source_reference_measures[source_row]
                root_error = source_reference_errors[source_row]
            elif target_image_is_full:
                root_value = target_reference_measures[target_row]
                root_error = target_reference_errors[target_row]
            else:
                root_value, root_error = integrate_root_pieces(
                    pieces, source_record.root_id
                )
            if source_image_is_full:
                source_value = source_physical_measures[source_row]
                source_error = source_physical_errors[source_row]
            else:
                source_value, source_error, exact_count, enclosed_count = (
                    _integrate_pulled_reference_overlap(
                        source_record.chart_arguments,
                        source_determinants[source_row],
                        pieces,
                        integration_policy,
                        rule_points,
                        rule_weights,
                    )
                )
                exact_integral_count += exact_count
                enclosed_integral_count += enclosed_count
            if target_image_is_full:
                target_value = target_physical_measures[target_row]
                target_error = target_physical_errors[target_row]
            else:
                target_value, target_error, exact_count, enclosed_count = (
                    _integrate_pulled_reference_overlap(
                        target_record.chart_arguments,
                        target_determinants[target_row],
                        pieces,
                        integration_policy,
                        rule_points,
                        rule_weights,
                    )
                )
                exact_integral_count += exact_count
                enclosed_integral_count += enclosed_count
        if not all(
            np.isfinite(value) and value > 0.0
            for value in (root_value, source_value, target_value)
        ):
            raise CommonRootOverlapError(
                "A positive material-reference intersection is not finitely representable in all three volume forms.",
                supermesh_owner.CommonRefinementStatus.PREDICATE_UNCERTAIN,
                integration_error=max(root_error, source_error, target_error),
            )
        overlap_source_rows.append(source_row)
        overlap_target_rows.append(target_row)
        reference_overlap_values.append(root_value)
        reference_overlap_errors.append(root_error)
        source_overlap_values.append(source_value)
        target_overlap_values.append(target_value)
        source_overlap_errors.append(source_error)
        target_overlap_errors.append(target_error)

    source_overlap_index = np.asarray(overlap_source_rows, dtype=np.int64)
    target_overlap_index = np.asarray(overlap_target_rows, dtype=np.int64)
    source_overlap_array = np.asarray(source_overlap_values, dtype=np.float64)
    target_overlap_array = np.asarray(target_overlap_values, dtype=np.float64)
    source_overlap_error_array = np.asarray(source_overlap_errors, dtype=np.float64)
    target_overlap_error_array = np.asarray(target_overlap_errors, dtype=np.float64)
    reference_overlap_array = np.asarray(reference_overlap_values, dtype=np.float64)
    reference_overlap_error_array = np.asarray(reference_overlap_errors, dtype=np.float64)
    if curved_general:
        unmatched_source = np.flatnonzero(
            np.bincount(source_overlap_index, minlength=len(source_records)) == 0
        )
        unmatched_target = np.flatnonzero(
            np.bincount(target_overlap_index, minlength=len(target_records)) == 0
        )
        if unmatched_source.size or unmatched_target.size:
            raise CommonRootOverlapError(
                "General curved material images leave endpoint rows without a positive common intersection.",
                supermesh_owner.CommonRefinementStatus.COVERAGE_GAP,
                unmatched_source_rows=unmatched_source.tolist(),
                unmatched_target_rows=unmatched_target.tolist(),
            )
    (
        source_reference_coverage_defects,
        source_reference_coverage_tolerances,
        source_reference_overlap_error_by_cell,
    ) = _material_coverage_diagnostics(
        source_overlap_index,
        reference_overlap_array,
        reference_overlap_error_array,
        source_reference_measures,
        source_reference_errors,
        policy.coverage_tolerance,
    )
    (
        target_reference_coverage_defects,
        target_reference_coverage_tolerances,
        target_reference_overlap_error_by_cell,
    ) = _material_coverage_diagnostics(
        target_overlap_index,
        reference_overlap_array,
        reference_overlap_error_array,
        target_reference_measures,
        target_reference_errors,
        policy.coverage_tolerance,
    )
    source_reference_coverage_tolerances = np.where(
        source_reference_omitted_bounds > 0.0,
        np.nextafter(
            source_reference_coverage_tolerances + source_reference_omitted_bounds,
            np.inf,
        ),
        source_reference_coverage_tolerances,
    )
    target_reference_coverage_tolerances = np.where(
        target_reference_omitted_bounds > 0.0,
        np.nextafter(
            target_reference_coverage_tolerances + target_reference_omitted_bounds,
            np.inf,
        ),
        target_reference_coverage_tolerances,
    )
    reference_status, reference_reason = supermesh_owner._coverage_status(
        source_reference_coverage_defects,
        target_reference_coverage_defects,
        source_reference_coverage_tolerances,
        target_reference_coverage_tolerances,
        policy,
    )
    if reference_status is not supermesh_owner.CommonRefinementStatus.SUCCESS:
        raise CommonRootOverlapError(
            "Root-Jacobian-weighted material-reference overlap does not certify complete coverage: "
            f"{reference_reason}.",
            reference_status,
            source_reference_defects=source_exact_defects,
            target_reference_defects=target_exact_defects,
            unmatched_source_rows=unmatched_source.tolist(),
            unmatched_target_rows=unmatched_target.tolist(),
            integration_error=float(
                max(
                    np.max(source_reference_overlap_error_by_cell, initial=0.0),
                    np.max(target_reference_overlap_error_by_cell, initial=0.0),
                )
            ),
            integration_tolerance=float(
                max(
                    np.max(source_reference_coverage_tolerances, initial=0.0),
                    np.max(target_reference_coverage_tolerances, initial=0.0),
                )
            ),
        )
    source_defects, source_tolerance, source_overlap_error_by_cell = (
        _material_coverage_diagnostics(
            source_overlap_index,
            source_overlap_array,
            source_overlap_error_array,
            source_physical_measures,
            source_physical_errors,
            policy.coverage_tolerance,
        )
    )
    target_defects, target_tolerance, target_overlap_error_by_cell = (
        _material_coverage_diagnostics(
            target_overlap_index,
            target_overlap_array,
            target_overlap_error_array,
            target_physical_measures,
            target_physical_errors,
            policy.coverage_tolerance,
        )
    )
    source_tolerance = np.where(
        source_physical_omitted_bounds > 0.0,
        np.nextafter(source_tolerance + source_physical_omitted_bounds, np.inf),
        source_tolerance,
    )
    target_tolerance = np.where(
        target_physical_omitted_bounds > 0.0,
        np.nextafter(target_tolerance + target_physical_omitted_bounds, np.inf),
        target_tolerance,
    )
    status, reason = supermesh_owner._coverage_status(
        source_defects,
        target_defects,
        source_tolerance,
        target_tolerance,
        policy,
    )
    if status is not supermesh_owner.CommonRefinementStatus.SUCCESS:
        raise CommonRootOverlapError(
            "Independent source/target physical Jacobians do not certify complete material overlap: "
            f"{reason}.",
            status,
            source_reference_defects=source_exact_defects,
            target_reference_defects=target_exact_defects,
            unmatched_source_rows=unmatched_source.tolist(),
            unmatched_target_rows=unmatched_target.tolist(),
            integration_error=float(
                max(
                    np.max(source_overlap_error_by_cell, initial=0.0),
                    np.max(target_overlap_error_by_cell, initial=0.0),
                )
            ),
        )

    target_counts = np.bincount(target_overlap_index, minlength=len(target_records))
    target_offsets = np.concatenate(
        (np.asarray((0,), dtype=np.int64), np.cumsum(target_counts, dtype=np.int64))
    )
    return MaterialReferenceVolumeOverlap(
        target_offsets,
        source_overlap_index,
        np.asarray(reference_overlap_values, dtype=np.float64),
        np.asarray(reference_overlap_errors, dtype=np.float64),
        source_overlap_array,
        target_overlap_array,
        source_overlap_error_array,
        target_overlap_error_array,
        source_reference_measures,
        target_reference_measures,
        source_reference_errors,
        target_reference_errors,
        source_physical_measures,
        target_physical_measures,
        source_physical_errors,
        target_physical_errors,
        source_ref_defects,
        target_ref_defects,
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        source_defects,
        target_defects,
        source_tolerance,
        target_tolerance,
        source_reference_coverage_tolerances,
        target_reference_coverage_tolerances,
        source_reference_coverage_defects,
        target_reference_coverage_defects,
        status,
        True,
        reason,
        witness.witness_id,
        root_validity.certificate_id,
        witness.source_validity.certificate_id,
        witness.target_validity.certificate_id,
        source_embedding.certificate_id,
        target_embedding.certificate_id,
        source_embedding,
        target_embedding,
        integration_policy,
        len(candidates) + source_self_candidates + target_self_candidates,
        len(overlap_source_rows),
        exact_integral_count,
        enclosed_integral_count,
    )


__all__ = [
    "CommonRootOverlapError",
    "CommonRootVolumeWitness",
    "CurvedOverlapIntegrationEvidence",
    "MaterialReferenceVolumeOverlap",
    "MaterialReferenceVolumeWitness",
    "CurvedOverlapIntegrationPolicy",
    "PhysicalOverlapActionEvidence",
    "PhysicalOverlapTransfer",
    "apply_physical_overlap_extensive",
    "apply_physical_overlap_extensive_with_evidence",
    "apply_physical_overlap_intensive",
    "transpose_physical_overlap_extensive",
    "transpose_physical_overlap_intensive",
    "prepare_common_root_volume_refinement",
    "prepare_material_reference_volume_overlap",
]
