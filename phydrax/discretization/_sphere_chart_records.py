#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Store chart correspondence data and validate its retained scientific bindings."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Hashable
from contextlib import nullcontext
from dataclasses import dataclass, field
from fractions import Fraction
from typing import cast

import equinox as eqx
import numpy as np
from jax import Array
from numpy.typing import NDArray

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry._global_embedding import GlobalEmbeddingCertificate
from ..geometry._sphere_material_atlas import (
    SphereMaterialCellAtlas,
    SphereProjectiveActionBank,
    SphereProjectiveReferenceMap,
    SphereProjectiveTriangleBounds,
)
from ..linalg._hermitian_spectral import fraction_storage_upper, reserve_fraction_work
from ..typing import checked
from ._cell_geometry import CellGeometrySpec
from ._cell_geometry_transition_types import MaterialChartDeformation
from ._cell_geometry_validity import cell_geometry_id, CellValidityCertificate
from ._cell_mesh import CellMesh
from ._coordinate_budget import CoordinateEnclosureBudget
from ._sphere_chart_piece_bank import SphereChartPieceBank, SphereChartPieceBankBuilder


type _Point = tuple[Fraction, Fraction]
type _Triangle = tuple[_Point, _Point, _Point]
type _FloatArray = NDArray[np.float64]


class PreparedSphereChartPiece(StrictModule, NonTrainableState):
    """source-reference triangle with its rational target map."""

    source_cell_global_id: int = eqx.field(static=True)
    target_cell_global_id: int = eqx.field(static=True)
    source_row: int = eqx.field(static=True)
    target_row: int = eqx.field(static=True)
    source_atlas_row: int = eqx.field(static=True)
    target_atlas_row: int = eqx.field(static=True)
    geometry_entity_id: str = eqx.field(static=True)
    occurrence_path: tuple[str, ...] = eqx.field(static=True)
    projective_map: SphereProjectiveReferenceMap
    exact_source_reference_vertices: _Triangle = eqx.field(static=True)
    exact_target_reference_vertices: _Triangle = eqx.field(static=True)
    source_reference_vertices: Array
    target_reference_vertices: Array
    projective_bounds: SphereProjectiveTriangleBounds = eqx.field(static=True)
    source_reference_area: Fraction = eqx.field(static=True)
    target_reference_area: Fraction = eqx.field(static=True)
    native_boundary_labels: tuple[int, ...] = eqx.field(static=True)
    native_polygon_vertex_indices: tuple[int, int, int] = eqx.field(static=True)
    native_clip_id: str = eqx.field(static=True)
    displacement_bound: float = eqx.field(static=True)
    piece_id: str = eqx.field(static=True)


class PreparedSphereChartOccurrence(StrictModule, NonTrainableState):
    """Unmerged scientific occurrence and its whole overlap pieces."""

    geometry_entity_id: str = eqx.field(static=True)
    occurrence_path: tuple[str, ...] = eqx.field(static=True)
    piece_indices: tuple[int, ...] = eqx.field(static=True)
    source_cell_global_ids: tuple[int, ...] = eqx.field(static=True)
    target_cell_global_ids: tuple[int, ...] = eqx.field(static=True)
    coverage_status: str = eqx.field(static=True)


class PreparedSphereChartDeformation(MaterialChartDeformation, NonTrainableState):
    """Accepted full-sphere transition, never a latitude/longitude alias."""

    source_atlas: SphereMaterialCellAtlas
    target_atlas: SphereMaterialCellAtlas
    source_mesh: CellMesh
    target_mesh: CellMesh
    source_geometry: CellGeometrySpec
    target_geometry: CellGeometrySpec
    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    pieces: SphereChartPieceBank
    occurrences: tuple[PreparedSphereChartOccurrence, ...]
    source_reference_coverage: tuple[Fraction, ...] = eqx.field(static=True)
    target_reference_coverage: tuple[Fraction, ...] = eqx.field(static=True)
    source_fidelity_bounds: Array
    target_fidelity_bounds: Array
    source_cell_measures: Array
    target_cell_measures: Array
    source_measure_errors: Array
    target_measure_errors: Array
    source_validity: CellValidityCertificate
    target_validity: CellValidityCertificate
    source_embedding: GlobalEmbeddingCertificate
    target_embedding: GlobalEmbeddingCertificate
    maximum_displacement_bound: float = eqx.field(static=True)
    candidate_pair_count: int = eqx.field(static=True)
    work_units: int = eqx.field(static=True)
    retained_bytes_upper: int = eqx.field(static=True)
    deformation_id: str = eqx.field(static=True)

    @property
    def exact(self) -> bool:
        return False

    @property
    def source_domain_coverage(self) -> str:
        return (
            "certified"
            if self.source_atlas.coverage_status == "certified"
            and all((area == Fraction(1, 2) for area in self.source_reference_coverage))
            else "unresolved"
        )

    @property
    def target_domain_coverage(self) -> str:
        return (
            "certified"
            if self.target_atlas.coverage_status == "certified"
            and all((area == Fraction(1, 2) for area in self.target_reference_coverage))
            else "unresolved"
        )

    @checked
    def require_bound(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> SphereChartValidation:
        """Re-prove the retained correspondence once under ``budget``.

        The returned token binds this proof to the exact deformation object,
        endpoint meshes/maps and request ledger; consumers of the same request
        present it instead of repeating the proof.
        """
        binding = (
            source_mesh.mesh_id,
            target_mesh.mesh_id,
            cell_geometry_id(source_geometry),
            cell_geometry_id(target_geometry),
        )
        if binding != (
            self.source_mesh.mesh_id,
            self.target_mesh.mesh_id,
            self.source_geometry_id,
            self.target_geometry_id,
        ):
            raise ValueError(
                "sphere_chart.require_bound.binding must satisfy the declared geometry and cell identities; got Sphere correspondence is stale for these coordinate maps or cells."
            )
        if budget is None:
            # A restoration owns no request ledger: one explicit ticket,
            # sized from the retained pair and cone counts, owns the proof.
            work_upper, bytes_upper = _sphere_validation_bounds(self)
            ticket = HostBatchTicket(work_upper, bytes_upper=bytes_upper)
            restoration = CoordinateEnclosureBudget(
                ticket, maximum_work_units=work_upper, maximum_memory_bytes=bytes_upper
            )
            with ticket, restoration.activate():
                self._prove(
                    source_mesh,
                    source_geometry,
                    target_mesh,
                    target_geometry,
                    restoration,
                )
            return SphereChartValidation(self, None, binding)
        proven = _PROVEN.get(budget, {}).get(id(self))
        if proven is not None and proven[0] is self and proven[1] == binding:
            self.pieces.authenticate(budget=budget)
            return SphereChartValidation(self, budget, binding, proven[2])
        self._prove(source_mesh, source_geometry, target_mesh, target_geometry, budget)
        return issue_sphere_chart_validation(self, budget)

    def _prove(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        budget: CoordinateEnclosureBudget,
        /,
    ) -> None:
        for atlas, mesh, geometry, validity, embedding in (
            (
                self.source_atlas,
                source_mesh,
                source_geometry,
                self.source_validity,
                self.source_embedding,
            ),
            (
                self.target_atlas,
                target_mesh,
                target_geometry,
                self.target_validity,
                self.target_embedding,
            ),
        ):
            atlas.require_bound(
                atlas.domain, mesh, geometry, atlas.coordinate_contract, budget=budget
            )
            validity.require_bound(geometry, mesh=mesh)
            embedding.binding.require(mesh, geometry)
            if not validity.all_certified or embedding.status != "certified":
                raise ValueError(
                    "sphere_chart.require_bound.pieces must satisfy the material reference contract; got Sphere correspondence requires valid globally embedded coordinate maps."
                )
        if (
            self.source_domain_coverage != "certified"
            or self.target_domain_coverage != "certified"
        ):
            raise ValueError(
                "sphere_chart.require_bound.pieces must satisfy the material reference contract; got Sphere correspondence lacks whole source and target partitions."
            )
        source_coverage = [Fraction(0) for _ in self.source_reference_coverage]
        target_coverage = [Fraction(0) for _ in self.target_reference_coverage]
        # Nothing computed by this proof is retained in its issued token:
        # maps, vertex images, fan-edge products and coverage sums die here.
        with budget.temporary_scope(), budget.live_storage() as storage:
            # Authenticate both atlases once and rebuild each distinct source-target
            # map once; every fan piece then re-proves its own construction.
            actions = SphereProjectiveActionBank.authenticate(
                self.source_atlas, self.target_atlas, budget=budget, storage=storage
            )
            verified: dict[tuple[int, int], SphereProjectiveReferenceMap] = {}
            images: dict[tuple[tuple[int, int], _Point], tuple[Fraction, _Point]] = {}
            geometry = _SpherePieceGeometry()
            # Read each atlas row table once: equinox method access on a whole
            # module re-flattens it, which would be quadratic in the pieces.
            rows = (
                _SphereAtlasRows.of(self.source_atlas),
                _SphereAtlasRows.of(self.target_atlas),
            )
            piece_ids: list[str] = []
            for piece in self.pieces._iter_kernel_pieces(budget=budget):
                piece_ids.append(piece.piece_id)
                source_row, target_row = _require_sphere_piece(
                    self.source_atlas,
                    self.target_atlas,
                    rows,
                    piece,
                    actions,
                    verified,
                    images,
                    geometry,
                    budget,
                )
                source_coverage[source_row] += piece.source_reference_area
                target_coverage[target_row] += piece.target_reference_area
            if (
                tuple(source_coverage) != self.source_reference_coverage
                or tuple(target_coverage) != self.target_reference_coverage
            ):
                raise ValueError(
                    "sphere_chart.require_bound.pieces must satisfy the material reference contract; got Sphere pieces no longer provide their whole coverage evidence."
                )
            budget.reserve(
                len(piece_ids)
                + (
                    sum(
                        np.asarray(value).nbytes
                        for value in (
                            self.source_cell_measures,
                            self.target_cell_measures,
                            self.source_measure_errors,
                            self.target_measure_errors,
                        )
                    )
                    + 63
                ) // 64,
                512 * (len(piece_ids) + len(source_coverage) + len(target_coverage)) + 4096,
            )
            identity = canonical_fingerprint(
                {
                    "kind": "prepared-sphere-projective-deformation",
                    "source": self.source_atlas.atlas_id,
                    "target": self.target_atlas.atlas_id,
                    "piece_bank": self.pieces.bank_id,
                    "pieces": tuple(piece_ids),
                    "source_areas": tuple(
                        ((area.numerator, area.denominator), (0, 1))
                        for area in self.source_reference_coverage
                    ),
                    "target_areas": tuple(
                        ((area.numerator, area.denominator), (0, 1))
                        for area in self.target_reference_coverage
                    ),
                    "measure_values": array_tree_fingerprint(
                        (self.source_cell_measures, self.target_cell_measures)
                    ),
                    "measure_errors": array_tree_fingerprint(
                        (self.source_measure_errors, self.target_measure_errors)
                    ),
                    "embedding": (
                        self.source_embedding.certificate_id,
                        self.target_embedding.certificate_id,
                    ),
                }
            )
            if identity != self.deformation_id:
                raise ValueError(
                    "sphere_chart.require_bound.identity must bind the canonical piece "
                    "bank and retained endpoint evidence; got a stale deformation ID"
                )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PreparedSphereChartDeformation],
        value: PreparedSphereChartDeformation,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> PreparedSphereChartDeformation:
        """Re-earn retained endpoint proofs on the caller ledger when supplied."""
        value.require_bound(
            value.source_mesh,
            value.source_geometry,
            value.target_mesh,
            value.target_geometry,
            budget=budget,
        )
        return value


# Request ledgers -> deformations proven (or issued by their producer) under
# them, with their binding and shared derived results. Entries never reference
# the ledger itself, so a request's proofs die with its ledger.
_PROVEN: weakref.WeakKeyDictionary[
    CoordinateEnclosureBudget,
    dict[
        int,
        tuple[
            PreparedSphereChartDeformation, tuple[str, str, str, str], dict[str, object]
        ],
    ],
] = weakref.WeakKeyDictionary()


def issue_sphere_chart_validation(
    deformation: PreparedSphereChartDeformation,
    budget: CoordinateEnclosureBudget,
    /,
) -> SphereChartValidation:
    """Register one completed proof of ``deformation`` under its request ledger.

    The producer calls this at the end of the request that prepared the
    record, and ``require_bound`` after each charged proof. Consumers of the
    same ledger then reuse it; any other ledger proves again.
    """
    token = SphereChartValidation(
        deformation,
        budget,
        (
            deformation.source_mesh.mesh_id,
            deformation.target_mesh.mesh_id,
            deformation.source_geometry_id,
            deformation.target_geometry_id,
        ),
    )
    _PROVEN.setdefault(budget, {})[id(deformation)] = (
        deformation,
        token.binding,
        token.derived,
    )
    return token


def _sphere_validation_bounds(
    deformation: PreparedSphereChartDeformation, /
) -> tuple[int, int]:
    """Bound cold authentication and proof without uncharged Fraction decoding."""
    buffers, shapes, _ = deformation.pieces._snapshot(None)
    piece_count, pairs = shapes[9][0], shapes[6][0]
    offsets = np.frombuffer(buffers[1], dtype="<i8").reshape(-1, 3)
    atom_bits = max(
        (
            8 * max(int(middle - start - 1), int(stop - middle))
            for start, middle, stop in offsets
        ),
        default=1,
    )
    bank_bytes = sum(len(buffer) for buffer in buffers)
    bank_indices = sum(
        len(buffers[index]) // 8 for index in (1, 3, 4, 5, 6, 8, 9, 10)
    )
    cones = deformation.source_atlas.num_charts + deformation.target_atlas.num_charts
    height = 2 * 1100 + 1
    geometry_bits = 28 * atom_bits + 8
    # A map table row is at least one actual source-target pair. Extra rows
    # for distinct stored payloads only make this bound more conservative.
    work = (
        80 * pairs + 9 * cones + 223 * piece_count
        + bank_indices + 3 * ((bank_bytes + 63) // 64)
        + 8 * shapes[1][0] + 4 * shapes[3][0] + 8192
    )
    storage = (
        cones * (fraction_storage_upper(9, 2 * 1100) + 512)
        + 3 * pairs * (fraction_storage_upper(3, 8 * height + 64) + 256)
        + piece_count * fraction_storage_upper(21, geometry_bits)
        + fraction_storage_upper(64, 8 * height + 64)
        + 8 * bank_bytes + 1024 * shapes[1][0] + 256 * shapes[3][0]
        + 2 * (8192 + fraction_storage_upper(31, atom_bits))
        + 512 * (piece_count + cones) + 1024**2
    )
    return work, storage


@dataclass(frozen=True, slots=True, eq=False)
class SphereChartValidation:
    """A completed ``require_bound`` proof owned by one request ledger.

    It is never stored or serialized: it only lets later consumers inside the
    same request reuse the proof instead of repeating it. Any other deformation
    object, endpoint map or ledger is refused.
    """

    deformation: PreparedSphereChartDeformation
    ledger: CoordinateEnclosureBudget | None
    binding: tuple[str, str, str, str]
    derived: dict[str, object] = field(default_factory=dict)

    @property
    def h1_cache(
        self,
    ) -> dict[Hashable, tuple[NDArray[np.int64], _FloatArray, float]]:
        """Share proved functional rows only for identical element/DOF routes.

        The field owner supplies an actual structural key, excluding the field
        name. This dictionary belongs to the shared request proof, including
        tokens returned by later ``require_bound`` calls on the same ledger.
        """
        return self.derive("sphere-h1-functional-rows", dict)

    def derive[T](self, key: str, factory: Callable[[], T], /) -> T:
        """Compute one request-scoped consequence of this proof at most once.

        The factory charges the owning ledger on first use; later consumers of
        the same request reuse its result. Callers own ``key`` uniqueness.
        """
        if key not in self.derived:
            self.derived[key] = factory()
        return cast(T, self.derived[key])

    def require(
        self,
        deformation: PreparedSphereChartDeformation,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> None:
        deformation.pieces.authenticate(budget=budget)
        issued = _PROVEN.get(budget, {}).get(id(deformation))
        if (
            deformation is not self.deformation
            or self.ledger is None
            or budget is not self.ledger
            or issued is None
            or issued[0] is not deformation
            or issued[1] != self.binding
            or issued[2] is not self.derived
            or (
                source_mesh.mesh_id,
                target_mesh.mesh_id,
                cell_geometry_id(source_geometry),
                cell_geometry_id(target_geometry),
            )
            != self.binding
        ):
            raise ValueError(
                "sphere_chart.validation must belong to this deformation, endpoint maps and request ledger; got a foreign or stale proof"
            )


@dataclass(frozen=True, slots=True)
class _SphereAtlasRows:
    """One validation pass's host view of an atlas's cell identity rows."""

    rows: dict[int, int]
    physical: tuple[int, ...]
    entities: tuple[str, ...]
    paths: tuple[tuple[str, ...], ...]

    @classmethod
    def of(cls, atlas: SphereMaterialCellAtlas, /) -> _SphereAtlasRows:
        return cls(
            {
                int(identifier): row
                for row, identifier in enumerate(np.asarray(atlas.cell_global_ids))
            },
            tuple(int(value) for value in np.asarray(atlas.physical_rows)),
            tuple(atlas.geometry_entity_ids),
            tuple(atlas.occurrence_paths),
        )

    def row(self, cell_global_id: int, /) -> int:
        row = self.rows.get(int(cell_global_id))
        if row is None:
            raise ValueError(
                "sphere_material.cell_global_id must identify a prepared cell; got absent identifier"
            )
        return row


@dataclass
class _SpherePieceGeometry:
    """Exact fan-edge and vertex bounds shared only by one validation pass."""

    wedges: dict[tuple[_Point, _Point], Fraction] = field(default_factory=dict)
    jacobians: dict[tuple[Fraction, Fraction], Fraction] = field(default_factory=dict)

    @staticmethod
    def reserve(
        budget: CoordinateEnclosureBudget | None,
        operations: int,
        count: int,
        values: tuple[Fraction, ...],
    ) -> None:
        bits = sum(
            max(abs(value.numerator).bit_length(), value.denominator.bit_length(), 1)
            for value in values
        )
        reserve_fraction_work(budget, operations, count, 2 * bits + 8)

    def wedge(
        self, first: _Point, second: _Point, budget: CoordinateEnclosureBudget | None
    ) -> Fraction:
        key = first, second
        value = self.wedges.get(key)
        if value is None:
            reverse = self.wedges.get((second, first))
            if reverse is None:
                self.reserve(budget, 3, 1, (*first, *second))
                value = first[0] * second[1] - first[1] * second[0]
            else:
                self.reserve(budget, 1, 1, (reverse,))
                value = -reverse
            self.wedges[key] = value
        return value

    def area(
        self, vertices: _Triangle, budget: CoordinateEnclosureBudget | None
    ) -> Fraction:
        # The same signed shoelace sum as chart_geometry.area, reusing only
        # exact endpoint determinants (including their directed sign).
        values = (
            self.wedge(vertices[0], vertices[1], budget),
            self.wedge(vertices[1], vertices[2], budget),
            self.wedge(vertices[2], vertices[0], budget),
        )
        self.reserve(budget, 3, 1, values)
        return (values[0] + values[1] + values[2]) / 2

    def bounds(
        self,
        ratio: Fraction,
        denominators: tuple[Fraction, Fraction, Fraction],
        budget: CoordinateEnclosureBudget | None,
    ) -> SphereProjectiveTriangleBounds:
        # triangle_bounds depends on the extrema, not the fan triangulation.
        lower, upper = min(denominators), max(denominators)
        if lower <= 0 or ratio <= 0:
            raise ValueError(
                "sphere_projective.reference must have positive denominator and orientation; got singular or reversed map"
            )
        for denominator in (lower, upper):
            key = ratio, denominator
            if key not in self.jacobians:
                self.reserve(budget, 3, 1, key)
                self.jacobians[key] = ratio / denominator**3
        return SphereProjectiveTriangleBounds(
            lower, upper, self.jacobians[ratio, upper], self.jacobians[ratio, lower]
        )


def _require_sphere_piece(
    source_atlas: SphereMaterialCellAtlas,
    target_atlas: SphereMaterialCellAtlas,
    rows: tuple[_SphereAtlasRows, _SphereAtlasRows],
    piece: PreparedSphereChartPiece,
    actions: SphereProjectiveActionBank,
    verified: dict[tuple[int, int], SphereProjectiveReferenceMap],
    images: dict[tuple[tuple[int, int], _Point], tuple[Fraction, _Point]],
    geometry: _SpherePieceGeometry,
    budget: CoordinateEnclosureBudget | None,
    /,
) -> tuple[int, int]:
    """Re-prove one retained fan piece and return its atlas rows."""
    source_row = rows[0].row(piece.source_cell_global_id)
    target_row = rows[1].row(piece.target_cell_global_id)
    if (
        piece.source_atlas_row,
        piece.target_atlas_row,
        piece.source_row,
        piece.target_row,
    ) != (
        source_row,
        target_row,
        rows[0].physical[source_row],
        rows[1].physical[target_row],
    ):
        raise ValueError(
            "sphere_chart.require_bound.binding must satisfy the declared geometry and cell identities; got Sphere piece physical rows do not retain their scientific cell identities."
        )
    mapping = piece.projective_map
    if (
        mapping.source_cell_global_id,
        mapping.target_cell_global_id,
        piece.geometry_entity_id,
        piece.occurrence_path,
    ) != (
        piece.source_cell_global_id,
        piece.target_cell_global_id,
        rows[0].entities[source_row],
        rows[0].paths[source_row],
    ):
        raise ValueError(
            "sphere_chart.require_bound.binding must satisfy the declared geometry and cell identities; got Sphere piece rational map or occurrence changes its scientific endpoints."
        )
    key = piece.source_cell_global_id, piece.target_cell_global_id
    proved = verified.get(key)
    if proved is None:
        with budget.temporary_scope() if budget is not None else nullcontext():
            mapping.require_bound(
                source_atlas, target_atlas, budget=budget, actions=actions
            )
        verified[key] = mapping
    elif proved is not mapping and (
        mapping.source_atlas_id,
        mapping.target_atlas_id,
        mapping.geometry_entity_id,
        mapping.occurrence_path,
        mapping.exact_coefficients,
        mapping.exact_denominator,
        mapping.orientation_ratio,
        mapping.map_id,
    ) != (
        proved.source_atlas_id,
        proved.target_atlas_id,
        proved.geometry_entity_id,
        proved.occurrence_path,
        proved.exact_coefficients,
        proved.exact_denominator,
        proved.orientation_ratio,
        proved.map_id,
    ):
        raise ValueError(
            "sphere_projective.binding must match both cell expressions; got stale binding"
        )
    # Every piece re-proves its exact construction. Fan pieces of one pair share
    # polygon vertices, so each (pair, vertex) image is computed once.
    vertices = piece.exact_source_reference_vertices
    shared = []
    for vertex in vertices:
        image = images.get((key, vertex))
        if image is None:
            # Three homogeneous rows (two products, two sums each), their sum
            # and two quotients.
            geometry.reserve(
                budget,
                16,
                3,
                (
                    *(value for row in mapping.exact_coefficients for value in row),
                    *vertex,
                ),
            )
            (image,) = mapping.reference_images((vertex,))
            images[(key, vertex)] = image
        shared.append(image)
    target: _Triangle = (shared[0][1], shared[1][1], shared[2][1])
    bounds = geometry.bounds(
        mapping.orientation_ratio, (shared[0][0], shared[1][0], shared[2][0]), budget
    )
    source_area = geometry.area(vertices, budget)
    target_area = geometry.area(target, budget)
    # Both signed coverage sums remain visits of this piece's proof.
    # Image/cache allocations are admitted separately, not a second blanket
    # eight-rational allowance on top of their actual retained structures.
    reserve_fraction_work(
        budget,
        2,
        2,
        2
        * max(
            max(value.numerator.bit_length(), value.denominator.bit_length())
            for point in (*vertices, *target)
            for value in point
        )
        + 64,
    )
    if (
        target != piece.exact_target_reference_vertices
        or bounds != piece.projective_bounds
        or piece.source_reference_area != source_area
        or piece.target_reference_area != target_area
    ):
        raise ValueError(
            "sphere_chart.require_bound.pieces must satisfy the material reference contract; got Sphere piece changes its reference construction or rational map."
        )
    if not np.array_equal(
        np.asarray(piece.source_reference_vertices),
        reference_representatives(vertices),
    ) or not np.array_equal(
        np.asarray(piece.target_reference_vertices),
        reference_representatives(piece.exact_target_reference_vertices),
    ):
        raise ValueError(
            "sphere_chart.require_bound.pieces must satisfy the material reference contract; got Sphere piece representatives no longer match the constructions."
        )
    return source_row, target_row


def reference_representatives(vertices: _Triangle) -> _FloatArray:
    return np.asarray(
        [[float(value) for value in point] for point in vertices], dtype=np.float64
    )


class SphereGeometryReconstruction(StrictModule, NonTrainableState):
    """degree-preserving radial-source reconstruction, without UV ghost cells."""

    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    geometry: CellGeometrySpec
    vertex_coordinates: Array
    target_atlas: SphereMaterialCellAtlas
    target_validity: CellValidityCertificate
    target_embedding: GlobalEmbeddingCertificate
    cell_global_ids: Array
    cell_patches: Array
    fidelity_bounds: Array
    node_enclosure_bounds: Array
    node_owner_cells: Array
    node_owner_locals: Array
    continuity_residual: float = eqx.field(static=True)
    evaluation_count: int = eqx.field(static=True)
    preparation_work_units: int = eqx.field(static=True)
    reconstruction_id: str = eqx.field(static=True)

    @checked
    def require_bound(
        self, source: SphereMaterialCellAtlas, target_mesh: CellMesh, /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        source.require_bound(
            source.domain,
            source.source_mesh,
            source.source_geometry,
            source.coordinate_contract,
            budget=budget,
        )
        self.target_atlas.require_bound(
            source.domain, target_mesh, self.geometry, source.coordinate_contract,
            budget=budget,
        )
        if (
            self.source_geometry_id,
            self.source_topology_id,
            self.target_geometry_id,
            self.target_topology_id,
            self.domain_id,
            self.source_id,
            self.source_revision,
        ) != (
            source.source_geometry_id,
            source.source_topology_id,
            cell_geometry_id(self.geometry),
            target_mesh.topology_id,
            source.domain_id,
            source.source_id,
            source.source_revision,
        ):
            raise ValueError(
                "sphere_chart.require_bound.binding must satisfy the declared geometry and cell identities; got Sphere reconstruction is stale for its source, target or material revision."
            )
        identifiers = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks]
        )
        patches = np.empty(identifiers.size, dtype=np.int32)
        patches[np.asarray(self.target_atlas.physical_rows, dtype=np.int64)] = np.asarray(
            self.target_atlas.patches
        )
        if (
            not np.array_equal(np.asarray(self.cell_global_ids), identifiers)
            or not np.array_equal(np.asarray(self.cell_patches), patches)
            or (
                not np.array_equal(
                    np.asarray(self.vertex_coordinates),
                    np.asarray(target_mesh.coordinates),
                )
            )
            or (
                not np.array_equal(
                    np.asarray(self.fidelity_bounds),
                    physical_order(
                        self.target_atlas, self.target_atlas.source_fidelity_bounds
                    ),
                )
            )
        ):
            raise ValueError(
                "sphere_chart.require_bound.binding must satisfy the declared geometry and cell identities; got Sphere reconstruction metadata is stale for its full-map material cells."
            )
        self.target_validity.require_bound(self.geometry, mesh=target_mesh)
        self.target_embedding.binding.require(target_mesh, self.geometry)
        if (
            not self.target_validity.all_certified
            or self.target_embedding.status != "certified"
        ):
            raise ValueError(
                "sphere_chart.require_bound.validity must satisfy the positive cell validity contract; got Sphere reconstruction lacks validity and global embedding."
            )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[SphereGeometryReconstruction], value: SphereGeometryReconstruction, /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> SphereGeometryReconstruction:
        """Re-earn the stored target atlas proof on the supplied caller ledger."""
        mesh = value.target_atlas.source_mesh
        value.target_atlas.require_bound(
            value.target_atlas.domain,
            mesh,
            value.geometry,
            value.target_atlas.coordinate_contract,
            budget=budget,
        )
        value.target_validity.require_bound(value.geometry, mesh=mesh)
        value.target_embedding.binding.require(mesh, value.geometry)
        if (
            not value.target_validity.all_certified
            or value.target_embedding.status != "certified"
        ):
            raise ValueError(
                "sphere_chart.require_bound.validity must satisfy the positive cell validity contract; got Sphere reconstruction lacks validity and global embedding."
            )
        if (
            value.target_geometry_id != cell_geometry_id(value.geometry)
            or value.target_topology_id != mesh.topology_id
        ):
            raise ValueError(
                "sphere_reconstruction.binding must retain its coordinate map and topology; got inconsistent restored identity"
            )
        return value


def physical_order(atlas: SphereMaterialCellAtlas, values: Array) -> _FloatArray:
    result = np.empty(atlas.num_charts, dtype=np.float64)
    result[np.asarray(atlas.physical_rows, dtype=np.int64)] = np.asarray(
        values, dtype=np.float64
    )
    return result
