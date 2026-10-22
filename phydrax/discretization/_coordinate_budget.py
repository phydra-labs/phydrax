#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Explicit coordinate-work counters, cache storage, and resource-refusal evidence.

The domain ledger delegates mechanical accounting to its admitted batch ticket.
Scientific expression owners retain their cache identities and charge order.
"""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import chain
from typing import Callable, cast, get_args, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import numpy as np
from numpy.typing import NDArray

from .._host_storage import (
    declared_integer_storage_bytes,
    declared_object_storage_bytes,
    retained_host_storage_profile,
)
from .._meshcore import is_native_host_buffer
from .._resource_ticket import BatchTicket, ManagedBytesContributor
from .._strict import StrictModule
from .._validation import nonnegative_integer
from ..typing import checked, parse


if TYPE_CHECKING:
    from phydrax.geometry._affine_mapped_contact import _AffineCellImageKey
    from phydrax.geometry._embedding_state import Facets, MeshCertificateCheck
    from phydrax.geometry._exact_polyhedral_geometry import TriangleContactPreparation
    from phydrax.geometry._global_embedding import GlobalEmbeddingCertificate
    from phydrax.geometry._mapped_reference_coverage import MappedReferenceSourceEvidence

    from ..geometry._atlas import CurveTrimLoop
    from ..geometry.brep._correspondence import CurveSurfaceCorrespondence
    from ..geometry.brep._model import _BRepCoedgeProof
    from ..geometry.brep._patches import (
        _TensorDerivativeSource,
        MeridianSpans,
        RationalBezierPiece,
    )
    from ..interchange._cad import _CadExportPreparation, CadCurveApproximation
    from ..lifecycle._meshing_source_records import _StoredRecordContentAdmission
    from ..meshing._association_routes import PlcEntityClasses
    from ..meshing._audit_topology import _PlanarAuditEmbeddingProof
    from ..meshing._hex_generation import _BoxQualityOperands
    from ..meshing._plc_planar_support import PlanarMaps
    from ._cell_geometry_predecessor import _PredecessorContentAdmission
    from ._cell_geometry_validity import (
        _PolynomialDifferential,
        _RootJacobianBernsteinOperands,
        CellValidityCertificate,
    )
    from ._cell_mesh import CellMesh
    from ._coordinate_enclosure import (
        BernsteinCoefficientKey,
        BernsteinDomain,
        CoordinateCornerWeights,
        CoordinateLinearKey,
        CoordinatePointKey,
        CoordinatePreparationKey,
        CoordinateSourceBank,
        ExactScalar,
        Expression,
        OrderedReferenceArguments,
        Polynomial,
        PreparedCoordinateCells,
        PreparedPolynomialArguments,
        PreparedPolynomialCoefficients,
        PreparedReferenceDenominators,
        RationalPolynomial,
        ReferenceArgumentsKey,
    )
    from ._coordinate_identity import ElementIdentityAdmission
    from ._coordinate_operands import (
        BinarySourceRows,
        CoordinateOperandProfile,
        CoordinateOperandProfileBuilder,
        PolynomialOperandEntry,
        PolynomialOperandKey,
        PreparedCoordinateOperands,
    )
    from ._coordinate_source_snapshot import (
        BarycentricSourceSnapshot,
        SplineSourceSnapshot,
    )
    from ._exact_algebraic_geometry import (
        PreparedExactAlgebraicGeometry,
        RadicalElement,
    )
    from ._exact_plc_geometry import (
        _PreparedPlcEntityFrames,
        ExactPlcCellGeometrySource,
        PreparedExactPlcGeometry,
    )
    from ._exact_power_geometry import PreparedExactPowerGeometry
    from ._periodic_exact import ExactIsometry
    from .fem._reference_types import FiniteElementSpec

CoordinateEnclosureResource: TypeAlias = Literal[
    "coefficient_work", "polynomial_storage", "retained_basis"
]
_COORDINATE_ENCLOSURE_RESOURCES: frozenset[str] = frozenset(
    get_args(CoordinateEnclosureResource)
)


@cache
def _coordinate_storage_owner_types() -> tuple[
    type[RationalPolynomial], type[RadicalElement]
]:
    """Resolve the two late-bound coefficient owner classes once per process."""
    from ._coordinate_enclosure import RationalPolynomial
    from ._exact_algebraic_geometry import RadicalElement

    return RationalPolynomial, RadicalElement


def _coordinate_storage_children(value: object, /) -> tuple[int, Iterable[object]]:
    """Return the pending-owner bound and children of one coordinate owner."""
    if type(value) is int:
        return 0, ()
    rational_polynomial, radical_element = _coordinate_storage_owner_types()
    if isinstance(value, dict):
        return 64 + 32 * len(value), chain(value.keys(), value.values())
    if isinstance(value, tuple):
        return 64 + 16 * len(value), value
    if isinstance(value, rational_polynomial):
        return 96, (value.numerator, value.denominator)
    if isinstance(value, Fraction):
        return 96, (value.numerator, value.denominator)
    if isinstance(value, radical_element):
        return 80, (value.terms,)
    return 0, ()


def coordinate_retention_preparation_bounds(
    basis: tuple[object, ...], /
) -> tuple[int, int]:
    """Return fresh-ledger work and peak bytes for retaining a coordinate basis."""
    pending: list[object] = [basis]
    seen: dict[int, object] = {}
    temporary, retained = 256, 256
    while pending:
        value = pending.pop()
        identifier = id(value)
        if identifier in seen and seen[identifier] is value:
            continue
        seen[identifier] = value
        temporary += 128
        retained += declared_object_storage_bytes(value) + 128
        storage, children = _coordinate_storage_children(value)
        temporary += storage
        pending.extend(children)
    # The traversal memo and its pending-owner metadata coexist with the full
    # retained graph at retain_basis's final admission. No sharing is counted twice.
    return len(seen), temporary + retained


class RationalEnclosureError(ValueError):
    """Report an unresolved rational denominator or chart limit."""


class CoordinateEnclosurePhaseEvidence(StrictModule):
    """Record completed work and additional storage observed by one coordinate phase."""

    work_units: int = eqx.field(static=True)
    native_charged_work_units: int = eqx.field(static=True)
    geometry_queries: int = eqx.field(static=True)
    peak_bytes_upper: int = eqx.field(static=True)


class CoordinateEnclosureResourceError(ValueError):
    """Report a pre-operation resource refusal and its phase evidence.

    A measurement or activation scope attaches ``phase_evidence`` on exit.
    The first completed phase is preserved through outer scopes.
    """

    def __init__(
        self,
        resource: CoordinateEnclosureResource,
        limit: int | np.integer,
        requested: int | np.integer,
        completed: int | np.integer,
        /,
        *,
        admission: bool = False,
    ) -> None:
        resource = parse(
            resource,
            CoordinateEnclosureResource,
            "CoordinateEnclosureResourceError.resource",
        )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in (limit, requested, completed)
        ):
            raise TypeError(
                "CoordinateEnclosureResourceError.quantities must be non-boolean integers; "
                f"got limit={limit!r}, requested={requested!r}, completed={completed!r}."
            )
        if not isinstance(admission, bool):
            raise TypeError(
                "CoordinateEnclosureResourceError.admission must be a bool; "
                f"got {admission!r}."
            )
        if (
            min(limit, requested, completed) < 0
            or requested <= limit
            or requested < completed
            or ((requested == completed or completed > limit) and not admission)
        ):
            raise ValueError(
                "CoordinateEnclosureResourceError.quantities must describe a "
                "pre-operation resource refusal; "
                f"got limit={limit}, requested={requested}, completed={completed}, "
                f"admission={admission}."
            )
        self.admission = admission
        self.phase_evidence: CoordinateEnclosurePhaseEvidence | None = None
        self.resource, self.limit, self.requested, self.completed = (
            resource,
            limit,
            requested,
            completed,
        )
        operation = "admission" if admission else "expansion"
        super().__init__(
            f"CoordinateEnclosureBudget.{resource} must fit limit {limit} before "
            f"{operation}; got requested={requested}, completed={completed}."
        )

    @checked
    def record_phase(self, measurement: CoordinateEnclosureMeasurement, /) -> None:
        """Attach the first completed measuring phase that propagated this refusal."""
        if not measurement._closed:
            raise RuntimeError(
                "CoordinateEnclosureResourceError.phase must be closed before recording; "
                "got an active measurement."
            )
        if self.phase_evidence is None:
            self.phase_evidence = CoordinateEnclosurePhaseEvidence(
                work_units=measurement.work_units,
                native_charged_work_units=measurement.native_charged_work_units,
                geometry_queries=measurement.geometry_queries,
                peak_bytes_upper=measurement.peak_bytes_upper,
            )


class CoordinateLiveStorage:
    """Track a live storage bound until its explicit release."""

    @checked
    def __init__(self, budget: CoordinateEnclosureBudget, /) -> None:
        self._budget = budget
        self.bound = 0
        self._closed = False

    def set_bound(self, total_bytes_upper: int, /, *, work: int = 0) -> None:
        if self._closed:
            raise RuntimeError(
                "CoordinateLiveStorage.set_bound must receive an open owner; got closed owner."
            )
        if isinstance(total_bytes_upper, bool) or not isinstance(total_bytes_upper, int):
            raise TypeError(
                "CoordinateLiveStorage.total_bytes_upper must be an explicit integer; "
                f"got {total_bytes_upper!r}."
            )
        if total_bytes_upper < 0:
            raise ValueError(
                "CoordinateLiveStorage.total_bytes_upper must be nonnegative; "
                f"got {total_bytes_upper}."
            )
        if isinstance(work, bool) or not isinstance(work, int):
            raise TypeError(
                f"CoordinateLiveStorage.work must be an explicit integer; got {work!r}."
            )
        if work < 0:
            raise ValueError(
                f"CoordinateLiveStorage.work must be nonnegative; got {work}."
            )
        budget = self._budget
        delta = total_bytes_upper - self.bound
        if delta >= 0:
            budget.reserve(work, delta)
        else:
            if work:
                budget.reserve(work)
            budget.release_temporary_storage(-delta)
        budget._live_bytes_upper += delta
        self.bound = total_bytes_upper

    def close(self) -> None:
        if not self._closed:
            self.set_bound(0)
            self._closed = True


@dataclass(frozen=True, slots=True)
class CoordinateEnclosureStageEvidence:
    """Record bounds and completed coordinate work of one ticket stage."""

    starting_work_units: int
    admitted_work_upper: int
    admitted_bytes_upper: int
    completed_work_units: int
    peak_bytes_upper: int


class CoordinateEnclosureMeasurement:
    """Measure phase-local work and additional live storage.

    Entry snapshots exclude completed predecessors and their still-live storage.
    Nested measurements observe the same admissions without changing accounting.
    """

    @checked
    def __init__(self, budget: CoordinateEnclosureBudget, /) -> None:
        self._budget = budget
        self.starting_work_units = budget.work_units
        self._starting_native_work = budget.native_charged_work_units
        self._query_ticket = budget.ticket
        self._starting_queries = self._query_ticket.geometry_queries
        self._starting_retained_bytes = budget.retained_basis_bytes
        self._starting_storage = (
            budget.retained_basis_bytes + budget.temporary_bytes_upper
        )
        self.required_work_units = 0
        self.peak_bytes_upper = 0
        self._completed_work = 0
        self._completed_native_work = 0
        self._completed_queries = 0
        self._completed_retained_bytes = 0
        self._closed = False

    @property
    def work_units(self) -> int:
        return (
            self._completed_work
            if self._closed
            else self._budget.work_units - self.starting_work_units
        )

    @property
    def native_charged_work_units(self) -> int:
        return (
            self._completed_native_work
            if self._closed
            else self._budget.native_charged_work_units - self._starting_native_work
        )

    @property
    def geometry_queries(self) -> int:
        """Return this ticket's charged queries, excluding separately owned children."""
        return (
            self._completed_queries
            if self._closed
            else self._query_ticket.geometry_queries - self._starting_queries
        )

    @property
    def retained_basis_bytes(self) -> int:
        return (
            self._completed_retained_bytes
            if self._closed
            else self._budget.retained_basis_bytes - self._starting_retained_bytes
        )

    def _observe_requirement(self, requested_work: int, /) -> None:
        required = requested_work - self.starting_work_units
        if required > self.required_work_units:
            self.required_work_units = required

    def _observe_storage(self, storage_upper: int, /) -> None:
        additional = storage_upper - self._starting_storage
        if additional > self.peak_bytes_upper:
            self.peak_bytes_upper = additional

    def _close(self) -> None:
        self._completed_work = self.work_units
        self._completed_native_work = self.native_charged_work_units
        self._completed_queries = self.geometry_queries
        self._completed_retained_bytes = self.retained_basis_bytes
        self._closed = True


class CoordinateEnclosureBudget:
    """Track domain work in exact arithmetic and storage over an admitted batch ticket.

    Structural bounds use declared integer bit lengths and fixed metadata bounds.
    """

    def __init__(
        self,
        ticket: BatchTicket,
        /,
        *,
        maximum_work_units: int | np.integer,
        maximum_memory_bytes: int | np.integer,
    ) -> None:
        self.maximum_work_units = nonnegative_integer(
            maximum_work_units, "CoordinateEnclosureBudget.maximum_work_units"
        )
        self.maximum_memory_bytes = nonnegative_integer(
            maximum_memory_bytes, "CoordinateEnclosureBudget.maximum_memory_bytes"
        )
        self._ticket = ticket
        self._work_storage = ManagedBytesContributor(ticket)
        self._checkpoint_admissions_remaining = 0
        self._checkpoint_work_upper = 0
        self.work_units = 0
        self.required_work_units = 0
        self.native_charged_work_units = 0
        self.retained_basis_bytes = 0
        self.temporary_bytes_upper = 0
        self._live_bytes_upper = 0
        self.peak_bytes_upper = 0
        self._measurements: list[CoordinateEnclosureMeasurement] = []
        self._evidence_measurements: list[CoordinateEnclosureMeasurement] = []
        # Equal element identities share a prepared basis; object identity does
        # not determine cache hits or the sequence of charged operations.
        self.basis_cache: dict[tuple[str, str], tuple[Polynomial, ...] | None] = {}
        self.degree_one_coordinate_element_cache: dict[str, FiniteElementSpec] = {}
        self.basis_profile_cache: dict[tuple[str, str], tuple[int, int, int]] = {}
        self.source_exponent_cache: dict[
            tuple[str, str], tuple[tuple[int, ...], ...]
        ] = {}
        self.source_composition_degree_cache: dict[
            tuple[tuple[str, str], tuple[tuple[int, ...], ...]], int
        ] = {}
        self.affine_simplex_source_cache: dict[tuple[str, str], bool] = {}
        self.affine_injectivity_cache: set[tuple[tuple[ExactScalar, ...], ...]] = set()
        self.root_injectivity_cache: dict[
            tuple[
                CoordinatePreparationKey, str, str, int, str,
                tuple[int, ...], tuple[tuple[Fraction, ...], ...],
            ],
            tuple[int, tuple[MeshCertificateCheck, ...]],
        ] = {}
        self.exact_orientation_preparation_cache: dict[
            tuple[str | tuple[int | Fraction, ...], ...],
            tuple[tuple[int | Fraction, ...], int | Fraction, int],
        ] = {}
        self.exact_orientation_sign_cache: dict[
            tuple[str | tuple[int | Fraction, ...], ...], int
        ] = {}
        self.exact_native_sign_cache: dict[tuple[str, bytes], int] = {}
        self.triangle_contact_batches: dict[
            tuple[tuple[int, ...], str, bytes, str, int, int],
            list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        ] = {}
        self.planar_audit_embedding_proof: _PlanarAuditEmbeddingProof | None = None
        self.exact_predicate_evaluations: dict[str, int] = {}
        self.exact_predicate_cache_hits: dict[str, int] = {}
        self.source_derivative_bounds_cache: dict[
            tuple[str, int, bytes],
            tuple[NDArray[np.float64], NDArray[np.float64]],
        ] = {}
        self.source_homotopy_enclosure_cache: dict[
            tuple[str, str, bytes],
            tuple[tuple[NDArray[np.float64], NDArray[np.float64]], ...],
        ] = {}
        self._source_homotopy_enclosure_storage = CoordinateLiveStorage(self)
        self.source_derivative_box_validation_cache: dict[
            tuple[str, int, bytes], NDArray[np.bool_]
        ] = {}
        self.immutable_bspline_fingerprint_cache: dict[
            tuple[str, tuple[int, ...], tuple[int, ...]], str
        ] = {}
        self.tensor_derivative_source_cache: dict[
            tuple[str, int], _TensorDerivativeSource
        ] = {}
        self.lattice_cache: dict[tuple[str, int], tuple[Polynomial, ...]] = {}
        self.bernstein_lattice_cache: dict[
            tuple[BernsteinDomain, tuple[int, ...], int], tuple[tuple[int, ...], ...]
        ] = {}
        self.bernstein_weight_cache: dict[
            tuple[BernsteinDomain, tuple[int, ...], int, tuple[int, ...]],
            tuple[int, tuple[tuple[int, int], ...]],
        ] = {}
        self.bernstein_weight_entry_cache: dict[tuple[int, int], tuple[int, int]] = {}
        self.bernstein_coefficient_cache: MutableMapping[
            BernsteinCoefficientKey, tuple[ExactScalar, ...]
        ] = {}
        self.reference_monomial_cache: dict[
            tuple[OrderedReferenceArguments, tuple[int, ...]]
            | tuple[OrderedReferenceArguments, tuple[int, ...], bytes],
            Mapping[tuple[int, ...], ExactScalar],
        ] = {}
        self.source_expression_cache: dict[
            tuple[str, str], tuple[Expression, ...] | None
        ] = {}
        self.reference_denominator_group_cache: dict[
            tuple[str, str], PreparedReferenceDenominators
        ] = {}
        self.reference_composition_cache: dict[
            tuple[str, str], tuple[Polynomial, ...]
        ] = {}
        self.prepared_reference_cache: dict[
            ReferenceArgumentsKey, PreparedPolynomialArguments
        ] = {}
        self.corner_weight_cache: dict[
            tuple[str, str], tuple[tuple[str, str], CoordinateCornerWeights]
        ] = {}
        # (source identity, root kind, rational reference point, scalar kinds)
        # -> nonzero source-basis values at that point.
        self.reference_basis_row_cache: dict[
            tuple[tuple[str, str], str, tuple[ExactScalar, ...], bytes],
            tuple[tuple[int, ExactScalar], ...],
        ] = {}
        self.corner_linear_cache: dict[
            CoordinateLinearKey,
            tuple[ExactScalar, ...],
        ] = {}
        self.polynomial_reference_image_cache: dict[
            tuple[CoordinatePreparationKey, tuple[Fraction, ...]],
            tuple[ExactScalar, ...],
        ] = {}
        self.coordinate_point_value_cache: dict[
            CoordinatePointKey, tuple[ExactScalar, ...]
        ] = {}
        self.multi_affine_cache: dict[tuple[str, str], bool] = {}
        self.density_cache: dict[CoordinatePreparationKey, tuple[Expression, ...]] = {}
        self.composed_source_density_cache: dict[
            CoordinatePreparationKey, Expression
        ] = {}
        self.composed_chart_density_cache: dict[tuple[str, str], Expression] = {}
        self.exact_cell_measure_cache: dict[
            CoordinatePreparationKey, tuple[float, float]
        ] = {}
        self.coordinate_cache: dict[CoordinatePreparationKey, tuple[Expression, ...]] = {}
        self.coordinate_corner_cache: dict[
            CoordinatePreparationKey, tuple[tuple[ExactScalar, ...], ...]
        ] = {}
        self.coordinate_vertex_image_cache: dict[
            tuple[str, str, str], tuple[tuple[ExactScalar, ...] | None, ...]
        ] = {}
        self.polynomial_cache: dict[CoordinatePreparationKey, tuple[Polynomial, ...]] = {}
        self.coefficient_preparation_cache: dict[
            CoordinatePreparationKey,
            tuple[CoordinateSourceBank, CoordinatePreparationKey],
        ] = {}
        self.source_bank_cache: dict[tuple[str, str], CoordinateSourceBank] = {}
        self.plc_carrier_binding_cache: set[tuple[str, str, str]] = set()
        self.plc_identity_corner_cache: set[tuple[str, str, str]] = set()
        # Scope keys whose published affine-root carrier this request verified.
        self.published_affine_carrier_cache: set[tuple[str, str, str]] = set()
        self.affine_source_carrier_cache: set[tuple[str, str, str]] = set()
        self.plc_witness_cache: dict[tuple[str, str], PreparedExactPlcGeometry] = {}
        self.power_witness_cache: dict[str, PreparedExactPowerGeometry] = {}
        # Exact isometry-group admissions earned in this request, keyed by the
        # group identity that binds the binary64 generators and metadata.
        self.periodic_generator_cache: dict[
            str, tuple[tuple[ExactIsometry, ...], tuple[int, ...]]
        ] = {}
        # (lifted mesh, numeric version, periodic descriptor) -> the periodic
        # mesh whose quotient rebuild and lift proof this request already earned.
        self.periodic_lifted_mesh_cache: dict[tuple[str, str, str], CellMesh] = {}
        # Exact loop coordinates in loop order -> loop-position triangles.
        self.loop_triangulation_cache: dict[
            tuple[tuple[Fraction, ...], ...], tuple[tuple[int, int, int], ...]
        ] = {}
        # Exact star tetrahedron corners -> signed cofactor determinant.
        self.star_determinant_cache: dict[tuple[tuple[Fraction, ...], ...], Fraction] = {}
        self.exact_triangle_contact_preparation: TriangleContactPreparation | None = None
        self.affine_cell_corner_images_cache: MutableMapping[
            _AffineCellImageKey, CoordinateSourceBank
        ] = {}
        self.affine_coverage_intersection_cache: dict[
            tuple[
                str, str,
                tuple[tuple[ExactScalar, ...], ...],
                tuple[tuple[ExactScalar, ...], ...],
            ],
            Fraction,
        ] = {}
        # Exact projected chart-chain triangle pair -> proved intersection area.
        self.affine_chain_intersection_cache: dict[
            tuple[tuple[tuple[Fraction, ...], ...], tuple[tuple[Fraction, ...], ...]],
            Fraction,
        ] = {}
        self.capture_affine_areas = False
        self.capture_measure_operands = False
        self.affine_intersection_operand_snapshots: dict[
            tuple[tuple[tuple[Fraction, ...], ...], tuple[tuple[Fraction, ...], ...]],
            Fraction,
        ] = {}
        self.simple_source_polygon_convexity_cache: dict[CoordinateSourceBank, bool] = {}
        self.algebraic_witness_cache: dict[
            tuple[str, str], PreparedExactAlgebraicGeometry
        ] = {}
        self.validity_certificate_cache: dict[str, CellValidityCertificate] = {}
        self.predecessor_content_admission_cache: dict[
            int, _PredecessorContentAdmission
        ] = {}
        # Actual record owners and their checked mutable banks, request-local.
        self.stored_record_content_cache: dict[
            tuple[int, str], _StoredRecordContentAdmission
        ] = {}
        self.element_identity_admission_cache: dict[int, ElementIdentityAdmission] = {}
        self.global_embedding_certificate_cache: dict[
            str, GlobalEmbeddingCertificate
        ] = {}
        # id(source bank) -> (that bank, its binary64 predicate view, exported).
        # The strong bank reference keeps the identity key unrecycled.
        self.source_predicate_view_cache: dict[
            int, tuple[object, np.ndarray, bool]
        ] = {}
        # A false row is never a rounded-source substitute. Its float slots
        # are unusable until every exact coordinate of that row was admitted.
        self.source_binary64_rows_cache: dict[
            int, BinarySourceRows
        ] = {}
        self.mapped_reference_source_evidence: dict[
            tuple[str, str, str], MappedReferenceSourceEvidence
        ] = {}
        self.brep_correspondence_cache: dict[str, CurveSurfaceCorrespondence] = {}
        self.cad_curve_approximation_cache: dict[str, CadCurveApproximation] = {}
        self.brep_trim_loop_cache: dict[str, CurveTrimLoop] = {}
        self.brep_coedge_proof_cache: dict[int, _BRepCoedgeProof] = {}
        self.cad_export_preparation_cache: dict[
            tuple[int, int, str], _CadExportPreparation
        ] = {}
        self.polynomial_differential_cache: dict[
            tuple[CoordinatePreparationKey, str, int, bool],
            _PolynomialDifferential,
        ] = {}
        # Root element/bank, cell kind, dimension, node admission -> rigorous
        # Bernstein bounds of its embedded Gram determinant, or None to decline.
        self.embedded_gram_bounds_cache: dict[
            tuple[CoordinatePreparationKey, str, int, int],
            tuple[ExactScalar, ExactScalar] | None,
        ] = {}
        self.volume_determinant_bounds_cache: dict[
            tuple[CoordinatePreparationKey, str, int, int],
            tuple[ExactScalar, ExactScalar] | None,
        ] = {}
        self.root_jacobian_bernstein_operands_cache: dict[
            tuple[CoordinatePreparationKey, str, int, int],
            _RootJacobianBernsteinOperands | None,
        ] = {}
        # Private actual-root numerical nets; no child quality/validity verdict.
        self.root_box_quality_operands_cache: dict[
            CoordinatePreparationKey, _BoxQualityOperands | None
        ] = {}
        self.planar_cell_map_cache: dict[tuple[str, str, str], PlanarMaps] = {}
        self.planar_class_cache: dict[
            tuple[tuple[str, str, str], str], tuple[PlcEntityClasses, ...]
        ] = {}
        self.scope_cache: dict[
            tuple[str, str, str], tuple[bool, NDArray[np.int64], CoordinateSourceBank]
        ] = {}
        self.face_chart_cache: dict[
            tuple[str, str, CoordinateSourceBank, str, tuple[int, ...]]
            | tuple[str, str, CoordinateSourceBank, bytes, str, tuple[int, ...]],
            tuple[tuple[tuple[Expression, ...], BernsteinDomain], ...],
        ] = {}
        self.face_flux_cache: dict[
            tuple[str, str, CoordinateSourceBank, str, tuple[int, ...]]
            | tuple[str, str, CoordinateSourceBank, bytes, str, tuple[int, ...]],
            ExactScalar | None,
        ] = {}
        self.mapped_cell_integral_cache: dict[
            tuple[CoordinatePreparationKey, str, int], Fraction | None
        ] = {}
        self.bspline_span_cache: dict[str, tuple[RationalBezierPiece, ...]] = {}
        self.meridian_span_cache: dict[str, MeridianSpans] = {}
        self.prepared_cell_cache: dict[tuple[str, str, str], PreparedCoordinateCells] = {}
        self.facet_cache: dict[tuple[str, str, str], Facets] = {}
        self.axis_affine_polynomial_cache: MutableMapping[
            str, PreparedPolynomialCoefficients
        ] = {}
        self.barycentric_source_snapshot_cache: dict[str, BarycentricSourceSnapshot] = {}
        self.spline_source_snapshot_cache: dict[str, SplineSourceSnapshot] = {}
        self.retained_objects: dict[int, object] = {}
        self._polynomial_preparation_objects: ChainMap[int, object] | None = None
        self._polynomial_preparation_storage: CoordinateLiveStorage | None = None
        self.retained_owner_objects: dict[int, object] = {}
        self.logical_array_bytes_upper = 0
        self.capture_coordinate_operands = False
        self.capture_polynomial_operands = False
        self.capture_signed_determinant_operands = False
        self.polynomial_operand_snapshots: dict[
            PolynomialOperandKey, PolynomialOperandEntry
        ] = {}
        self.coordinate_operand_profiles: list[CoordinateOperandProfile] = []
        self._prepared_coordinate_operands: PreparedCoordinateOperands | None = None
        self._stage_work_bounds: list[int] = []
        self._stage_memory_bounds: list[int] = []
        self.stage_evidence: list[CoordinateEnclosureStageEvidence] = []
        self._activation_depth = 0
        self._storage_ticket: BatchTicket | None = None
        self._storage_owner: ManagedBytesContributor | None = None
        self._storage_factory: Callable[[int], BatchTicket] | None = None
        self._storage_handed_over = False

    @property
    def ticket(self) -> BatchTicket:
        """Return the explicitly bound stage ticket."""
        return self._ticket

    @property
    def storage_ticket(self) -> BatchTicket | None:
        """Return the request's persistent cache-byte reservation, if present."""
        return self._storage_ticket

    def _scope_storage_contributor(self) -> ManagedBytesContributor | None:
        from .._meshcore import NativeScopeTicket

        if isinstance(self._storage_ticket, NativeScopeTicket):
            return self._storage_owner
        return None

    def _refresh_retained_storage(self) -> None:
        factory = self._storage_factory
        if factory is None:
            raise RuntimeError(
                "CoordinateEnclosureBudget.retained_storage must have an explicit factory; got None."
            )
        prior = self._storage_ticket
        owner = self._storage_owner
        scope_owner = self._scope_storage_contributor()
        if scope_owner is not None and prior is not None and not prior.settled:
            # Keep every existing live token. The scope view adds only growth
            # on the current target and retires actual shrinkage in place.
            scope_owner.set_bound(self.retained_basis_bytes + self.temporary_bytes_upper)
            return
        if owner is not None:
            owner.close()
        self._storage_owner = None
        if prior is not None and not prior.settled:
            prior.settle()
        self._storage_ticket = None
        storage_upper = self.retained_basis_bytes + self.temporary_bytes_upper
        ticket = factory(storage_upper)
        self._storage_ticket = ticket
        self._storage_owner = ticket.managed_storage()
        self._storage_owner.set_bound(storage_upper)

    @contextmanager
    @checked
    def retained_storage(
        self, factory: Callable[[int], BatchTicket], /
    ) -> Iterator[CoordinateEnclosureBudget]:
        """Retain cache bytes across native phases and settle them at request exit."""
        if (
            self._storage_handed_over
            or self._storage_factory is not None
            or not self._ticket.settled
        ):
            raise RuntimeError(
                "CoordinateEnclosureBudget.retained_storage must begin after initial phase settlement without another owner; got active reservation."
            )
        self._storage_factory = factory
        self._work_storage.close()
        primary: BaseException | None = None
        try:
            self._refresh_retained_storage()
            yield self
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                ticket = self._storage_ticket
                owner = self._storage_owner
                if owner is not None:
                    owner.close()
                if ticket is not None and not ticket.settled:
                    ticket.settle()
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(
                    f"Coordinate retained-storage settlement also refused: {error}"
                )
            finally:
                self._storage_ticket = None
                self._storage_owner = None
                self._storage_factory = None

    def handover_retained_storage(
        self,
        receive: Callable[
            [BatchTicket, ManagedBytesContributor, Mapping[int, object], int], None
        ],
        /,
    ) -> None:
        """Transfer the live reservation, without releasing or readmitting bytes.

        The receiver validates and adopts the exact ticket/contributor before
        this ledger detaches it. A failed receiver leaves this owner unchanged.
        No work, query, peak or retained-object evidence is reset.
        """
        ticket, owner = self._storage_ticket, self._storage_owner
        if (
            self._storage_handed_over
            or ticket is None
            or owner is None
            or ticket.settled
            or not self._ticket.settled
            or self._activation_depth
        ):
            raise RuntimeError(
                "CoordinateEnclosureBudget.handover_retained_storage must receive "
                "a live retained reservation between completed phases."
            )
        receive(
            ticket, owner, self.retained_owner_objects, self.logical_array_bytes_upper
        )
        self._storage_handed_over = True
        self._storage_ticket = None
        self._storage_owner = None
        self._storage_factory = None

    @contextmanager
    def bind_ticket(self, ticket: BatchTicket, /) -> Iterator[CoordinateEnclosureBudget]:
        """Bind the next phase after settlement, retaining request counters and caches."""
        if self._storage_handed_over or not self._ticket.settled or self._activation_depth:
            raise RuntimeError(
                "CoordinateEnclosureBudget.bind_ticket must follow prior ticket settlement; got live stage."
            )
        if ticket.settled:
            raise RuntimeError(
                "CoordinateEnclosureBudget.bind_ticket must receive an unsettled ticket; got settled ticket."
            )
        self._work_storage.close()
        self._ticket = ticket
        self._work_storage = ticket.managed_storage()
        self._checkpoint_admissions_remaining = 0
        self._checkpoint_work_upper = self.work_units
        primary: BaseException | None = None
        try:
            with ticket:
                with self.activate():
                    self.reserve(0)
                    self._set_host_storage(
                        self.retained_basis_bytes + self.temporary_bytes_upper
                    )
                    yield self
        except BaseException as error:
            primary = error
            raise
        finally:
            if self._storage_factory is not None:
                try:
                    self._refresh_retained_storage()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note(
                        f"Coordinate retained-storage refresh also refused: {error}"
                    )

    def _set_host_storage(self, storage_upper: int, /) -> None:
        """Count simultaneous live object storage on the bound stage."""
        if self._storage_handed_over:
            raise RuntimeError(
                "CoordinateEnclosureBudget storage must not mutate after handover."
            )
        scope_owner = self._scope_storage_contributor()
        if scope_owner is not None:
            scope_owner.set_bound(storage_upper)
            if not self._ticket.settled:
                self._work_storage.set_bound(0)
            return
        if self._ticket.settled:
            if storage_upper > self.retained_basis_bytes + self.temporary_bytes_upper:
                raise RuntimeError(
                    "CoordinateEnclosureBudget.storage_upper must only release retired "
                    f"capacity; got growth to {storage_upper}."
                )
            # A refused ticket has already released its reservation. Cleanup
            # updates domain lifetimes without debiting that retired ticket.
            return
        retained = self._storage_ticket
        baseline = 0 if retained is None else min(retained.bytes_upper, storage_upper)
        owner = self._storage_owner
        if owner is not None:
            owner.set_bound(baseline)
        self._work_storage.set_bound(storage_upper - baseline)

    def _admit_ticket(self, work: int, storage_upper: int, /) -> None:
        """Check the next work and storage bound against the stage allowance."""
        ticket = self._ticket

        if ticket.settled:
            if work or (
                storage_upper > self.retained_basis_bytes + self.temporary_bytes_upper
            ):
                raise RuntimeError(
                    "CoordinateEnclosureBudget.ticket must only release storage after "
                    "settlement; got new work or growth on a retired ticket."
                )
            return
        # Poll at most 256 admissions or 4096 completed work units apart, and
        # before a single admission of 4096 units or more. The ticket owns the
        # nonrenewing native deadline and its typed refusal evidence.
        if (
            not self._checkpoint_admissions_remaining
            or self.work_units >= self._checkpoint_work_upper
            or work >= 4096
        ):
            ticket.checkpoint()
            self._checkpoint_admissions_remaining = 256
            self._checkpoint_work_upper = self.work_units + 4096
        self._checkpoint_admissions_remaining -= 1
        # The retained owner holds at most the current storage bound, exactly as
        # _set_host_storage splits it; the stage ticket requests only the rest.
        retained = self._storage_ticket
        stage_storage = storage_upper - (
            0 if retained is None else min(retained.bytes_upper, storage_upper)
        )
        if self._scope_storage_contributor() is not None:
            # The live scope owner preadmits this whole bound below, without
            # releasing its old leases or borrowing a fresh byte allowance.
            stage_storage = 0
        if work > ticket.remaining_work_units or stage_storage > ticket.bytes_upper:
            ticket.refuse(work, bytes_upper=stage_storage)

    def reserve(
        self,
        work: int | np.integer,
        storage_upper: int | np.integer = 0,
        /,
        *,
        storage_resource: CoordinateEnclosureResource = "polynomial_storage",
    ) -> None:
        """Admit nonnegative work and storage increments before completing either.

        Invalid increments leave counters and ticket charges unchanged.
        Storage releases belong to the live-owner and temporary-scope boundaries.
        """
        # Exact nonnegative ints and declared literal strings are already
        # canonical; every other value takes the full validating parse.
        if type(work) is not int or work < 0:
            work = nonnegative_integer(work, "CoordinateEnclosureBudget.work")
        if type(storage_upper) is not int or storage_upper < 0:
            storage_upper = nonnegative_integer(
                storage_upper, "CoordinateEnclosureBudget.storage_upper"
            )
        if (
            type(storage_resource) is not str
            or storage_resource not in _COORDINATE_ENCLOSURE_RESOURCES
        ):
            storage_resource = parse(
                storage_resource,
                CoordinateEnclosureResource,
                "CoordinateEnclosureBudget.storage_resource",
            )
        self._reserve_validated(work, storage_upper, storage_resource)

    def _reserve_validated(
        self,
        work: int,
        storage_upper: int,
        storage_resource: CoordinateEnclosureResource,
        /,
    ) -> None:
        """Admit owner-validated increments with the public reservation charge order."""
        if self._storage_handed_over:
            raise RuntimeError(
                "CoordinateEnclosureBudget.reserve must precede retained-storage "
                "handover; got a detached ledger."
            )
        proposed_work = self.work_units + work
        proposed_storage = (
            self.retained_basis_bytes + self.temporary_bytes_upper + storage_upper
        )
        if proposed_work > self.required_work_units:
            self.required_work_units = proposed_work
        for measurement in self._measurements:
            measurement._observe_requirement(proposed_work)

        # Stage entries already include every enclosing bound. The last entry
        # is their minimum; do not rebuild two tuples on every admission.
        work_limit = self.maximum_work_units
        if self._stage_work_bounds:
            work_limit = min(work_limit, self._stage_work_bounds[-1])
        storage_limit = self.maximum_memory_bytes
        if self._stage_memory_bounds:
            storage_limit = min(storage_limit, self._stage_memory_bounds[-1])
        if proposed_work > work_limit:
            raise CoordinateEnclosureResourceError(
                "coefficient_work",
                work_limit,
                proposed_work,
                self.work_units,
                admission=proposed_work == self.work_units
                or self.work_units > work_limit,
            )
        if proposed_storage > storage_limit:
            completed_storage = self.retained_basis_bytes
            if storage_resource != "retained_basis":
                completed_storage += self.temporary_bytes_upper
            raise CoordinateEnclosureResourceError(
                storage_resource,
                storage_limit,
                proposed_storage,
                completed_storage,
                admission=proposed_storage == completed_storage
                or completed_storage > storage_limit,
            )
        self._admit_ticket(work, proposed_storage)
        if storage_upper:
            self._set_host_storage(proposed_storage)
        if work:
            self._ticket.consume(work)
            if self._ticket.native_accounting:
                self.native_charged_work_units += work
        self.work_units = proposed_work
        self.temporary_bytes_upper += storage_upper
        if proposed_storage > self.peak_bytes_upper:
            self.peak_bytes_upper = proposed_storage
        for measurement in self._measurements:
            measurement._observe_storage(proposed_storage)

    def release_temporary_storage(self, amount: int | np.integer, /) -> None:
        """Release a completed temporary allocation without changing charged work.

        The host storage owner is updated before committing the reduced domain
        bound. Retained storage and peak evidence remain unchanged.
        """
        if self._storage_handed_over:
            raise RuntimeError(
                "CoordinateEnclosureBudget.release_temporary_storage must precede "
                "retained-storage handover; got a detached ledger."
            )
        amount = nonnegative_integer(amount, "CoordinateEnclosureBudget.amount")
        remaining = self.temporary_bytes_upper - amount
        if remaining < 0:
            raise RuntimeError(
                "CoordinateEnclosureBudget.amount must fit its temporary storage bound; "
                f"got amount={amount}, temporary_bytes_upper={self.temporary_bytes_upper}."
            )
        self._set_host_storage(self.retained_basis_bytes + remaining)
        self.temporary_bytes_upper = remaining

    def reserve_geometry_queries(self, count: int | np.integer, /) -> None:
        """Count source queries, preserving ticket admission refusal."""
        if self._storage_handed_over:
            raise RuntimeError(
                "CoordinateEnclosureBudget.reserve_geometry_queries must precede "
                "retained-storage handover; got a detached ledger."
            )
        count = nonnegative_integer(count, "CoordinateEnclosureBudget.geometry_queries")

        ticket = self._ticket
        if count > ticket.remaining_geometry_queries:
            ticket.refuse(0, queries_upper=count)
        ticket.consume(geometry_queries=count)

    def admit_work_bound(
        self, maximum_work: int, /, *, storage_upper: int | np.integer = 0
    ) -> None:
        """Dry-admit work and simultaneous storage without counting either as completed."""
        if (
            isinstance(maximum_work, bool)
            or not isinstance(maximum_work, int)
            or maximum_work < 0
        ):
            raise TypeError(
                "CoordinateEnclosureBudget.maximum_work must be a nonnegative explicit "
                f"integer; got {maximum_work!r}."
            )
        storage_upper = nonnegative_integer(
            storage_upper, "CoordinateEnclosureBudget.storage_upper"
        )
        requested = self.work_units + maximum_work
        self.required_work_units = max(self.required_work_units, requested)
        for measurement in self._measurements:
            measurement._observe_requirement(requested)

        limit = self.maximum_work_units
        if self._stage_work_bounds:
            limit = min(limit, self._stage_work_bounds[-1])
        if requested > limit:
            raise CoordinateEnclosureResourceError(
                "coefficient_work",
                limit,
                requested,
                self.work_units,
                admission=requested == self.work_units or self.work_units > limit,
            )
        completed_storage = self.retained_basis_bytes + self.temporary_bytes_upper
        proposed_storage = completed_storage + storage_upper
        storage_limit = self.maximum_memory_bytes
        if self._stage_memory_bounds:
            storage_limit = min(storage_limit, self._stage_memory_bounds[-1])
        if proposed_storage > storage_limit:
            raise CoordinateEnclosureResourceError(
                "polynomial_storage",
                storage_limit,
                proposed_storage,
                completed_storage,
                admission=proposed_storage == completed_storage
                or completed_storage > storage_limit,
            )
        self._admit_ticket(maximum_work, proposed_storage)

    def degree_one_coordinate_element(self, cell_kind: str, /) -> FiniteElementSpec:
        """Retain one declared degree-one reference element per request and cell kind."""
        from ._cell_geometry import coordinate_lagrange_element

        element = self.degree_one_coordinate_element_cache.get(cell_kind)
        if element is None:
            element = coordinate_lagrange_element(cell_kind, 1)
            self.retain_owner(element)
            self.degree_one_coordinate_element_cache[cell_kind] = element
        return element

    @staticmethod
    def _source_homotopy_entry_bytes(
        key: tuple[str, str, bytes],
        value: tuple[tuple[NDArray[np.float64], NDArray[np.float64]], ...],
    ) -> int:
        # Dict slot, complete semantic key and tuples, ndarray headers, and
        # *owned* copied backing bytes. No batch view escapes into this cache.
        return (
            64 + declared_object_storage_bytes(key)
            + sum(declared_object_storage_bytes(atom) for atom in key)
            + declared_object_storage_bytes(value)
            + sum(
                declared_object_storage_bytes(pair)
                + sum(256 + array.nbytes for array in pair)
                for pair in value
            )
        )

    def source_homotopy_enclosure(
        self,
        key: tuple[str, str, bytes],
        value: tuple[tuple[NDArray[np.float64], NDArray[np.float64]], ...],
        /,
        *,
        maximum_entries: int,
    ) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]], ...]:
        """Retain source-only rows under this request's original byte allowance.

        Capacity is an eviction heuristic, never an additional proof quota.
        The live owner survives ticket phases; evicted arrays are not retained
        through the permanent scientific-owner index.
        """
        cache = self.source_homotopy_enclosure_cache
        storage = self._source_homotopy_enclosure_storage
        prior = storage.bound
        size = self._source_homotopy_entry_bytes(key, value)
        # Old and new backing coexist while copying, including on replacement.
        storage.set_bound(max(256, prior) + size, work=1)
        try:
            stored = tuple((lower.copy(), upper.copy()) for lower, upper in value)
            for pair in stored:
                for array in pair:
                    array.setflags(write=False)
        except BaseException:
            storage.set_bound(prior)
            raise
        old = cache.pop(key, None)
        released = 0 if old is None else self._source_homotopy_entry_bytes(key, old)
        del old
        cache[key] = stored
        while len(cache) > maximum_entries:
            evicted_key = next(iter(cache))
            evicted = cache.pop(evicted_key)
            released += self._source_homotopy_entry_bytes(evicted_key, evicted)
            del evicted_key, evicted
        storage.set_bound(max(256, prior) + size - released)
        return stored

    def _profile_admission_bounds(self) -> tuple[int, int] | None:
        """Return absolute work and storage totals that every admission may reach.

        Each limit checked by ``_reserve_validated``, ``_admit_ticket``,
        ``_set_host_storage`` and ``BatchTicket.consume`` is monotone in the
        proposed totals. Below both returned totals no admission in a run of
        nondecreasing proposals can refuse. ``None`` keeps individual
        admissions: a settled stage refuses work and a closed owner cannot grow.
        """
        ticket = self._ticket
        if ticket.settled:
            return None
        work_headroom = self._work_storage.headroom()
        if work_headroom is None:
            return None
        work_limit = self.maximum_work_units
        if self._stage_work_bounds:
            work_limit = min(work_limit, self._stage_work_bounds[-1])
        storage_limit = self.maximum_memory_bytes
        if self._stage_memory_bounds:
            storage_limit = min(storage_limit, self._stage_memory_bounds[-1])
        # The stage ticket receives the bound above the retained baseline.
        stage_bytes = min(ticket.bytes_upper, work_headroom)
        if stage_bytes < 0:
            return None
        storage = stage_bytes
        retained = self._storage_ticket
        if retained is not None:
            scope_owner = self._scope_storage_contributor()
            baseline = retained.bytes_upper if scope_owner is None else scope_owner.bound
            storage += baseline
            owner = self._storage_owner
            if owner is not None:
                owner_headroom = owner.headroom()
                if owner_headroom is None:
                    return None
                if owner_headroom < baseline:
                    storage = min(storage, owner_headroom)
        return (
            min(work_limit, self.work_units + ticket.remaining_work_units),
            min(storage_limit, storage),
        )

    def _commit_profile_admissions(
        self, work: int, storage: int, checkpoint_admissions: int, /
    ) -> None:
        """Commit admissions already proved to fit, as their sequence would.

        Every observed quantity is a running maximum or sum of nondecreasing
        proposals, so the last proposal determines the same counters, peaks,
        ticket charges and live host bound as the individual admissions.
        """
        self._checkpoint_admissions_remaining = checkpoint_admissions
        if not work and not storage:
            return
        proposed_work = self.work_units + work
        proposed_storage = (
            self.retained_basis_bytes + self.temporary_bytes_upper + storage
        )
        if proposed_work > self.required_work_units:
            self.required_work_units = proposed_work
        for measurement in self._measurements:
            measurement._observe_requirement(proposed_work)
        if storage:
            self._set_host_storage(proposed_storage)
        if work:
            self._ticket.consume(work)
            if self._ticket.native_accounting:
                self.native_charged_work_units += work
        self.work_units = proposed_work
        self.temporary_bytes_upper += storage
        if proposed_storage > self.peak_bytes_upper:
            self.peak_bytes_upper = proposed_storage
        for measurement in self._measurements:
            measurement._observe_storage(proposed_storage)

    def _object_storage_profile(
        self,
        root: object,
        existing: Mapping[int, object],
        /,
        *,
        retained: bool,
        operand_profile: CoordinateOperandProfileBuilder | None = None,
        include_operands: bool = False,
    ) -> tuple[dict[int, object], int]:
        """Charge each fresh graph owner one visit and its declared storage.

        Per owner, the admission sequence is one work unit, 128 index bytes,
        the retained-capacity check, then any pending-owner bytes. Owners whose
        admissions provably fit pure precomputed bounds and reach no deadline
        poll are coalesced. The pending prefix is committed before every poll,
        every refusal candidate and every exit, and such owners replay the
        individual admissions, so polls and refusals see the exact original
        prefix and unit prices.
        """
        storage_resource: CoordinateEnclosureResource = (
            "retained_basis" if retained else "polynomial_storage"
        )
        reserve = self._reserve_validated
        reserve(0, 256, storage_resource)
        pending: list[object] = [root]
        seen: dict[int, object] = {}
        size = 256 if retained and not existing else 0
        storage_limit = self.maximum_memory_bytes
        if self._stage_memory_bounds:
            storage_limit = min(storage_limit, self._stage_memory_bounds[-1])
        bounds = self._profile_admission_bounds()
        committed_work = work = self.work_units
        committed_storage = storage = (
            self.retained_basis_bytes + self.temporary_bytes_upper
        )
        admissions = self._checkpoint_admissions_remaining
        poll_work = self._checkpoint_work_upper
        try:
            while pending:
                value = pending.pop()
                identifier = id(value)
                if (identifier in seen and seen[identifier] is value) or (
                    identifier in existing and existing[identifier] is value
                ) or (include_operands and self._has_coordinate_operand_owner(value)):
                    continue
                owner_storage, children = _coordinate_storage_children(value)
                if operand_profile is not None:
                    owner_storage += operand_profile.storage_upper(value)
                # Conservative CPython hash-index slots. The transient traversal
                # index coexists with a retained index; live owners use the same
                # upper bound even though their lifetime is tracked separately.
                owner_size = declared_object_storage_bytes(value) + 128
                count = 3 if owner_storage else 2
                indexed = storage + 128
                if (
                    bounds is not None
                    and admissions >= count
                    and work + 1 < poll_work
                    and work < bounds[0]
                    and indexed + owner_storage <= bounds[1]
                    and not (retained and indexed + size + owner_size > storage_limit)
                ):
                    work += 1
                    storage = indexed + owner_storage
                    admissions -= count
                    size += owner_size
                else:
                    self._commit_profile_admissions(
                        work - committed_work, storage - committed_storage, admissions
                    )
                    committed_work, committed_storage = work, storage
                    reserve(1, 0, "polynomial_storage")
                    reserve(0, 128, storage_resource)
                    size += owner_size
                    if (
                        retained
                        and self.retained_basis_bytes + self.temporary_bytes_upper + size
                        > storage_limit
                    ):
                        raise CoordinateEnclosureResourceError(
                            "retained_basis",
                            storage_limit,
                            self.retained_basis_bytes + self.temporary_bytes_upper + size,
                            self.retained_basis_bytes,
                        )
                    if owner_storage:
                        reserve(0, owner_storage, storage_resource)
                    bounds = self._profile_admission_bounds()
                    committed_work = work = self.work_units
                    committed_storage = storage = (
                        self.retained_basis_bytes + self.temporary_bytes_upper
                    )
                    admissions = self._checkpoint_admissions_remaining
                    poll_work = self._checkpoint_work_upper
                seen[identifier] = value
                if operand_profile is not None:
                    operand_profile.observe(value, children)
                if owner_storage:
                    pending.extend(children)
        finally:
            # Every coalesced owner charges one work unit. Individual
            # admissions leave nothing pending and own the poll counters.
            if work != committed_work:
                self._commit_profile_admissions(
                    work - committed_work, storage - committed_storage, admissions
                )
        return seen, size

    def live_object_storage_upper(self, value: object, /) -> int:
        """Measure a live object graph without retaining it after traversal."""
        with self.temporary_scope():
            _, size = self._object_storage_profile(value, {}, retained=False)
        return size

    def retain_basis(
        self, basis: tuple[object, ...], /, *, operand: bool = False,
        _kernel_operands: bool = False,
    ) -> None:
        """Profile each coefficient owner once during its request retention lifetime."""
        # Strong references prevent recycled IDs from earning another object's
        # profile. A retained root also owns every already-profiled graph member.
        if (
            self.retained_objects.get(id(basis)) is basis
            or self._has_coordinate_operand_owner(basis)
        ):
            return
        from types import MappingProxyType

        from ._coordinate_operands import (
            CoordinateOperandProfile,
            CoordinateOperandProfileBuilder,
        )

        builder = (
            CoordinateOperandProfileBuilder(private_atoms=_kernel_operands)
            if operand and self.capture_coordinate_operands
            else None
        )
        with self.temporary_scope():
            seen, size = self._object_storage_profile(
                basis,
                self.retained_objects,
                retained=True,
                operand_profile=builder,
                include_operands=True,
            )
            # The completed traversal index still coexists with the new graph.
            # Admit that final capacity before committing any retained objects.
            self.reserve(0, size, storage_resource="retained_basis")
            self.temporary_bytes_upper -= size
            self.retained_basis_bytes += size
            self.retained_objects.update(seen)
            self.peak_bytes_upper = max(
                self.peak_bytes_upper,
                self.retained_basis_bytes + self.temporary_bytes_upper,
            )
            if builder is not None and builder.supported:
                # The first actual traversal already paid every owner visit.
                # Promote its live index and immutable-field snapshots rather
                # than copying/scanning the coefficient bank in a later phase.
                self.reserve(0, 1024, storage_resource="retained_basis")
                metadata = (
                    1024 + 128 * len(seen)
                    + 768 * len(builder.owner_fields)
                )
                self.temporary_bytes_upper -= metadata
                self.retained_basis_bytes += metadata
                self.coordinate_operand_profiles.append(
                    CoordinateOperandProfile(
                        basis,
                        MappingProxyType(seen),
                        tuple(builder.owner_fields),
                        self.retained_basis_bytes,
                    )
                )

    def _retain_plc_entity_frames(
        self,
        source: ExactPlcCellGeometrySource,
        frames: _PreparedPlcEntityFrames,
        /,
        *,
        temporary_storage_credit: CoordinateLiveStorage | None,
    ) -> None:
        """Promote only an issued PLC kernel bank's admitted structural owner.

        Ordinary scientific graphs still use ``retain_owner`` and its full
        traversal. The PLC constructor has already admitted and built every
        numeric row under its unchanged first-earner work contract; its private
        seal authenticates that source and all nested immutable row fields.
        """
        from types import MappingProxyType

        from ._coordinate_operands import CoordinateOperandProfile
        from ._exact_plc_geometry import (
            _plc_entity_frame_basis,
            _plc_entity_frame_wrapper_storage_upper,
        )

        count = len(frames.triangles) + len(frames.segments)
        with self.temporary_scope():
            self.reserve(24 + 4 * count, 512 + 64 * count)
            basis = _plc_entity_frame_basis(frames, source=source)
            if (
                self.retained_objects.get(id(frames)) is frames
                or self._has_coordinate_operand_owner(frames)
            ):
                return
            lease = temporary_storage_credit
            # A newly issued source wrapper may own the very same immutable
            # numeric tables. Their existing actual owner indices prove that
            # only wrapper/seal metadata is new; value equality never earns this.
            shared = all(
                self.retained_objects.get(id(table)) is table
                or self._has_coordinate_operand_owner(table)
                for table in (frames.triangles, frames.segments)
            )
            storage = (
                _plc_entity_frame_wrapper_storage_upper(count)
                if shared else frames.storage_bytes_upper
            )
            if (
                not isinstance(lease, CoordinateLiveStorage)
                or lease._budget is not self or lease._closed
                or isinstance(storage, bool) or not isinstance(storage, int)
                or storage <= 0 or lease.bound < storage
                or lease.bound > self._live_bytes_upper
                or lease.bound > self.temporary_bytes_upper
            ):
                raise ValueError(
                    "PLC entity operands must promote their actual open "
                    "same-budget preadmitted numeric-storage owner."
                )
            # No capacity is released and reacquired: the same admitted bytes
            # change lifetime owner atomically, including across scratch scopes.
            self.temporary_bytes_upper -= storage
            self._live_bytes_upper -= storage
            lease.bound -= storage
            self.retained_basis_bytes += storage
            owners = {
                id(frames): frames, id(basis): basis,
                id(frames.triangles): frames.triangles,
                id(frames.segments): frames.segments,
            }
            self.retained_objects.update(owners)
            self.coordinate_operand_profiles.append(
                CoordinateOperandProfile(
                    basis, MappingProxyType(owners), (), self.retained_basis_bytes,
                )
            )


    def retain_operand_metadata(self, size: int, /) -> None:
        """Retain a prebounded operand index/metadata allocation, not a new bank."""
        self.reserve(0, size, storage_resource="retained_basis")
        self.temporary_bytes_upper -= size
        self.retained_basis_bytes += size

    def _has_coordinate_operand_owner(self, value: object, /) -> bool:
        prepared = self._prepared_coordinate_operands
        return prepared is not None and any(
            profile.owners.get(id(value)) is value for profile in prepared._profiles
        )

    def admit_coordinate_operands(self, prepared: PreparedCoordinateOperands, /) -> None:
        """Admit a checked immutable bank's actual retained physical bounds.

        The caller authenticates its earning receipt and immutable input owners.
        First import precedes local numeric preparation; no ambiguous overlapping
        profile union is guessed. Subsequent retention reads the immutable owner
        indices without allocating a duplicate coefficient-ownership table.
        """
        from ._coordinate_operands import (
            _admit_signed_determinant_operand,
            _same_exact_measure_bounds,
            _signed_determinant_operand_key,
        )

        measure_snapshots = prepared._measure_snapshots
        missing_measure = False
        for key, integral in measure_snapshots:
            self.reserve(1)
            current = self.exact_cell_measure_cache.get(key)
            if current is None:
                missing_measure = True
            elif not _same_exact_measure_bounds(current, integral):
                raise ValueError("Coordinate operands conflict with current exact cell measures.")
        for key, witness in prepared._power_witnesses.items():
            self.reserve(1)
            current = self.power_witness_cache.get(key)
            if current is not witness and (
                current is not None or self._prepared_coordinate_operands is prepared
            ):
                raise ValueError("Coordinate operands conflict with the current exact power construction.")
        if self._prepared_coordinate_operands is prepared:
            self._require_polynomial_operand_cache(prepared)
            # The immutable snapshots and current cache cannot change during
            # these local checks. Preserve the missing-entry refusal's original
            # precedence without scanning and charging every measure again.
            if missing_measure:
                raise ValueError("Coordinate exact cell measures changed after operand admission.")
            for key, area in prepared._affine_areas:
                self.reserve(1)
                if self.affine_chain_intersection_cache.get(key) != area:
                    raise ValueError("Coordinate operands conflict with current exact triangle areas.")
            return
        if self.retained_objects or self.retained_owner_objects or self.retained_basis_bytes:
            raise ValueError(
                "CoordinateEnclosureBudget operands must be admitted before new numeric preparation."
            )
        self.reserve(0, prepared._physical_bytes, storage_resource="retained_basis")
        self.temporary_bytes_upper -= prepared._physical_bytes
        self.retained_basis_bytes += prepared._physical_bytes
        self._prepared_coordinate_operands = prepared
        self.source_bank_cache.update(prepared._source_banks)
        self.plc_witness_cache.update(prepared._plc_witnesses)
        self.power_witness_cache.update(prepared._power_witnesses)
        self.coordinate_vertex_image_cache.update(prepared._vertex_images)
        self.plc_carrier_binding_cache.update(prepared._carrier_keys)
        self.plc_identity_corner_cache.update(prepared._identity_keys)
        self.facet_cache.update(prepared._facets)
        self.source_binary64_rows_cache.update(prepared._binary_rows)
        self.affine_chain_intersection_cache.update(prepared._affine_areas)
        self.exact_cell_measure_cache.update(measure_snapshots)
        for key, snapshots, polynomial in prepared._polynomial_snapshots:
            signed_key = _signed_determinant_operand_key(key)
            if signed_key is not None:
                if polynomial:
                    raise ValueError("Signed determinant operands must not carry a coordinate-axis role.")
                _admit_signed_determinant_operand(signed_key, snapshots, self)
                continue
            key = cast("CoordinatePreparationKey", key)
            if key in self.polynomial_cache or key in self.coordinate_cache:
                raise ValueError("Coordinate operands must not overwrite current polynomial preparations.")
            self.reserve(
                sum(len(snapshot) for snapshot in snapshots),
                256 + sum(256 + 96 * len(snapshot) for snapshot in snapshots),
            )
            polynomials: tuple[Polynomial, ...] = tuple(
                {index: coefficient for index, coefficient in snapshot}
                for snapshot in snapshots
            )
            self.retain_basis((key, polynomials))
            if polynomial:
                self.polynomial_cache[key] = polynomials
            self.coordinate_cache[key] = polynomials
            self.polynomial_operand_snapshots[key] = snapshots, polynomial

    def _require_polynomial_operand_cache(
        self, prepared: PreparedCoordinateOperands, /
    ) -> None:
        """Refuse changed current maps rather than overwrite a conflicting cache."""
        from ._coordinate_operands import (
            _require_signed_determinant_cache,
            _signed_determinant_operand_key,
        )

        for key, snapshots, polynomial in prepared._polynomial_snapshots:
            self.reserve(1 + sum(len(snapshot) for snapshot in snapshots))
            signed_key = _signed_determinant_operand_key(key)
            if signed_key is not None:
                if polynomial:
                    raise ValueError("Signed determinant operands must retain their numeric-only role.")
                _require_signed_determinant_cache(signed_key, snapshots, self)
                continue
            key = cast("CoordinatePreparationKey", key)
            caches = (
                (self.polynomial_cache, self.coordinate_cache)
                if polynomial else (self.coordinate_cache,)
            )
            for entries in caches:
                current = entries.get(key)
                if current is None or len(current) != len(snapshots) or any(
                    type(polynomial) is not dict
                    or tuple(polynomial.items()) != snapshot
                    or any(type(coefficient) is not Fraction
                           for coefficient in polynomial.values())
                    for polynomial, snapshot in zip(current, snapshots, strict=True)
                ):
                    raise ValueError("Coordinate operands conflict with current polynomial preparations.")

    @contextmanager
    def discarded_preparation_scope(self) -> Iterator[None]:
        """Retire preparations whose outputs the caller consumes and discards.

        No newly prepared proof, array or scientific owner may escape this
        scope, and preexisting scientific values must not be mutated. Cache
        entries may be inserted, replaced or removed: their original containers,
        entries and retention ledgers are restored even after a refusal.
        Completed work, queries, peaks, stage bounds and deadlines never rewind.

        Request-source homotopy enclosures have their own live-storage owner
        and are not discarded preparations; callers must not prepare them here.
        """
        with self.temporary_scope():
            # Admit the attribute index before constructing it. The second pass
            # and restoration are charged up front, so cleanup cannot need work
            # capacity after a consuming operation exhausts the request.
            fields = len(vars(self))
            self.reserve(2 * fields, 512 + 128 * fields)
            attributes = [
                (name, value)
                for name, value in vars(self).items()
                if isinstance(value, (dict, set, ChainMap))
                and name not in (
                    "exact_predicate_evaluations",
                    "exact_predicate_cache_hits",
                    "source_homotopy_enclosure_cache",
                )
            ]
            contact = self.exact_triangle_contact_preparation
            containers = sum(
                len(value.maps) if isinstance(value, ChainMap) else 1
                for _, value in attributes
            )
            entries = sum(
                sum(len(mapping) for mapping in value.maps)
                if isinstance(value, ChainMap) else len(value)
                for _, value in attributes
            )
            if contact is not None:
                containers += 4
                entries += sum(
                    len(value) for value in (
                        contact.planes, contact.projections, contact.sides, contact.bounds
                    )
                )
            batches = self.triangle_contact_batches
            batch_entries = sum(len(value) for value in batches.values())
            # Copies own only hash/list slots, not the immutable proof graphs
            # already retained by the request. Original batch lists are mutable
            # append-only caches, so snapshot their contents separately.
            self.reserve(
                2 * (containers + entries + len(batches) + batch_entries),
                512 * (containers + len(batches)) + 128 * entries + 32 * batch_entries,
            )
            mappings: list[
                tuple[MutableMapping[object, object], dict[object, object]]
            ] = []
            sets: list[tuple[set[object], set[object]]] = []
            for _, value in attributes:
                if isinstance(value, ChainMap):
                    mappings.extend((mapping, dict(mapping)) for mapping in value.maps)
                elif isinstance(value, dict):
                    mappings.append((value, value.copy()))
                else:
                    sets.append((value, value.copy()))
            if contact is not None:
                for value in (
                    contact.planes, contact.projections, contact.sides, contact.bounds
                ):
                    # Erase heterogeneous key/value types only at this storage
                    # boundary; restoration writes each map's own exact entries.
                    mapping = cast(MutableMapping[object, object], value)
                    mappings.append((mapping, dict(mapping)))
            batch_lists = [(value, value.copy()) for value in batches.values()]
            retained = self.retained_basis_bytes
            logical = self.logical_array_bytes_upper
            capture_operands = self.capture_coordinate_operands
            self.capture_coordinate_operands = False
            try:
                # These three polynomial caches already have a dedicated
                # temporary owner. Nest it rather than copying or closing an
                # enclosing phase's live owner.
                with self.polynomial_preparation_scope():
                    yield
            finally:
                self.capture_coordinate_operands = capture_operands
                for mapping, original in mappings:
                    mapping.clear()
                    mapping.update(original)
                for values, original_values in sets:
                    values.clear()
                    values.update(original_values)
                for values, original_values in batch_lists:
                    values[:] = original_values
                for name, value in attributes:
                    setattr(self, name, value)
                self.exact_triangle_contact_preparation = contact
                self.retained_basis_bytes = retained
                self.logical_array_bytes_upper = logical

    @contextmanager
    def polynomial_preparation_scope(self) -> Iterator[None]:
        """Bound temporary polynomial memos to their consuming phase.

        Previously retained source preparations remain reusable. New affine
        snapshots and Bernstein controls live only until this scope exits;
        callers consume those coefficients before leaving the scope.
        """
        prior_affine = self.axis_affine_polynomial_cache
        prior_bernstein = self.bernstein_coefficient_cache
        prior_corners = self.affine_cell_corner_images_cache
        prior_objects = self._polynomial_preparation_objects
        prior_storage = self._polynomial_preparation_storage
        with self.temporary_scope(), self.live_storage() as storage:
            # Four local dictionaries, ChainMap records and two-owner map lists.
            storage.set_bound(4 * (256 + 256 + 256 + 2 * 16))
            affine: dict[str, PreparedPolynomialCoefficients] = {}
            bernstein: dict[BernsteinCoefficientKey, tuple[ExactScalar, ...]] = {}
            corners: dict[_AffineCellImageKey, CoordinateSourceBank] = {}
            objects: dict[int, object] = {}
            self.axis_affine_polynomial_cache = ChainMap(affine, prior_affine)
            self.bernstein_coefficient_cache = ChainMap(bernstein, prior_bernstein)
            self.affine_cell_corner_images_cache = ChainMap(corners, prior_corners)
            self._polynomial_preparation_objects = ChainMap(
                objects,
                self.retained_objects if prior_objects is None else prior_objects,
            )
            self._polynomial_preparation_storage = storage
            try:
                yield
            finally:
                self.axis_affine_polynomial_cache = prior_affine
                self.bernstein_coefficient_cache = prior_bernstein
                self.affine_cell_corner_images_cache = prior_corners
                self._polynomial_preparation_objects = prior_objects
                self._polynomial_preparation_storage = prior_storage
                affine.clear()
                bernstein.clear()
                corners.clear()
                objects.clear()

    def retain_polynomial_preparation(self, basis: tuple[object, ...], /) -> None:
        """Account for a polynomial memo through its current lifetime owner."""
        storage = self._polynomial_preparation_storage
        objects = self._polynomial_preparation_objects
        if storage is None or objects is None:
            self.retain_basis(basis)
            return
        if objects.get(id(basis)) is basis:
            return
        with self.temporary_scope():
            seen, size = self._object_storage_profile(basis, objects, retained=False)
            storage.set_bound(storage.bound + size)
            objects.maps[0].update(seen)

    def retain_owner(
        self,
        value: object,
        /,
        *,
        temporary_storage_credit: CoordinateLiveStorage | None = None,
    ) -> None:
        """Retain a scientific graph, promoting any admitted live-storage credit.

        Only bytes beyond the credit require new capacity. Completed profiling
        visits remain charged if that capacity is refused; the lease and retained
        graph are changed only after admission succeeds.
        """
        lease = temporary_storage_credit
        if lease is not None:
            if not isinstance(lease, CoordinateLiveStorage):
                raise TypeError(
                    "CoordinateEnclosureBudget.temporary_storage_credit must be a "
                    f"CoordinateLiveStorage owner; got {lease!r}."
                )
            if lease._closed:
                raise RuntimeError(
                    "CoordinateEnclosureBudget.temporary_storage_credit must be open; "
                    "got closed owner."
                )
            if lease._budget is not self:
                raise ValueError(
                    "CoordinateEnclosureBudget.temporary_storage_credit must belong "
                    "to this budget; got foreign owner."
                )
            if isinstance(lease.bound, bool) or not isinstance(lease.bound, int):
                raise TypeError(
                    "CoordinateEnclosureBudget.temporary_storage_credit.bound must "
                    f"be an explicit integer; got {lease.bound!r}."
                )
            if lease.bound < 0:
                raise ValueError(
                    "CoordinateEnclosureBudget.temporary_storage_credit.bound must "
                    f"be nonnegative; got {lease.bound}."
                )
            if (
                lease.bound > self._live_bytes_upper
                or lease.bound > self.temporary_bytes_upper
            ):
                raise RuntimeError(
                    "CoordinateEnclosureBudget.temporary_storage_credit.bound must "
                    "fit admitted live and temporary storage; "
                    f"got bound={lease.bound}, live_bytes_upper={self._live_bytes_upper}, "
                    f"temporary_bytes_upper={self.temporary_bytes_upper}."
                )
        if (
            id(value) in self.retained_owner_objects
            and self.retained_owner_objects[id(value)] is value
        ):
            return
        owners, storage, logical, visits = retained_host_storage_profile(
            value, self.retained_owner_objects, is_native_host_buffer
        )
        credit = 0 if lease is None else min(storage, lease.bound)
        self.reserve(visits)
        self.reserve(0, storage - credit)
        # Promotion preserves the simultaneous live bound already admitted above.
        # Do not wrap it in a temporary scope: the lease reduction must survive
        # this operation and every enclosing scratch-storage scope.
        self.temporary_bytes_upper -= storage
        self._live_bytes_upper -= credit
        if lease is not None:
            lease.bound -= credit
        self.retained_basis_bytes += storage
        self.retained_owner_objects.update(owners)
        self.logical_array_bytes_upper += logical

    @contextmanager
    def bound_stage(
        self,
        maximum_work_units: int,
        maximum_memory_bytes: int,
        /,
        *,
        starting_work_units: int | None = None,
    ) -> Iterator[CoordinateEnclosureBudget]:
        """Bound a stage or a resumed owner's allowance from its starting work count."""
        if (
            isinstance(maximum_work_units, bool)
            or not isinstance(maximum_work_units, int)
            or isinstance(maximum_memory_bytes, bool)
            or not isinstance(maximum_memory_bytes, int)
            or maximum_work_units < 0
            or maximum_memory_bytes < 0
        ):
            raise ValueError(
                "CoordinateEnclosureBudget.bound_stage must receive nonnegative explicit "
                "integer bounds; "
                f"got work={maximum_work_units!r}, memory={maximum_memory_bytes!r}."
            )
        start = self.work_units if starting_work_units is None else starting_work_units
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or start < 0
            or start > self.work_units
        ):
            raise ValueError(
                "CoordinateEnclosureBudget.starting_work_units must be an observed "
                f"nonnegative ledger count; got {start!r} with completed={self.work_units}."
            )
        self._stage_work_bounds.append(
            min(
                start + maximum_work_units,
                self.maximum_work_units,
                *self._stage_work_bounds,
            )
        )
        self._stage_memory_bounds.append(
            min(
                maximum_memory_bytes,
                self.maximum_memory_bytes,
                *self._stage_memory_bounds,
            )
        )
        try:
            self.reserve(0)
            yield self
        finally:
            self._stage_memory_bounds.pop()
            self._stage_work_bounds.pop()

    @contextmanager
    def live_storage(self) -> Iterator[CoordinateLiveStorage]:
        owner = CoordinateLiveStorage(self)
        primary: BaseException | None = None
        try:
            yield owner
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                owner.close()
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(f"Coordinate live-storage release also refused: {error}")

    @contextmanager
    def temporary_scope(self) -> Iterator[None]:
        prior = self.temporary_bytes_upper
        prior_live = self._live_bytes_upper
        primary: BaseException | None = None
        try:
            yield
        except BaseException as error:
            primary = error
            raise
        finally:
            remaining = prior + self._live_bytes_upper - prior_live
            try:
                self._set_host_storage(self.retained_basis_bytes + remaining)
                self.temporary_bytes_upper = remaining
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(
                    f"Coordinate temporary-storage release also refused: {error}"
                )

    @property
    def current_measurement(self) -> CoordinateEnclosureMeasurement:
        """Return the evidence owner's active phase measurement."""
        if not self._evidence_measurements:
            raise RuntimeError(
                "CoordinateEnclosureBudget.current_measurement must have an active measurement; got none."
            )
        return self._evidence_measurements[-1]

    @contextmanager
    def measurement(self) -> Iterator[CoordinateEnclosureMeasurement]:
        """Measure one evidence phase without resetting request counters or caches."""
        measurement = CoordinateEnclosureMeasurement(self)
        self._measurements.append(measurement)
        self._evidence_measurements.append(measurement)
        refusal: CoordinateEnclosureResourceError | None = None
        try:
            yield measurement
        except CoordinateEnclosureResourceError as error:
            refusal = error
            raise
        finally:
            measurement._close()
            self._evidence_measurements.pop()
            self._measurements.pop()
            if refusal is not None:
                refusal.record_phase(measurement)

    @contextmanager
    def activate(self) -> Iterator[CoordinateEnclosureBudget]:
        """Record phase evidence without selecting an ambient ledger."""
        primary: BaseException | None = None
        start = self.work_units
        outermost = self._activation_depth == 0
        measurement = CoordinateEnclosureMeasurement(self) if outermost else None
        if measurement is not None:
            self._measurements.append(measurement)
        self._activation_depth += 1
        try:
            yield self
        except BaseException as error:
            primary = error
            raise
        finally:
            self._activation_depth -= 1
            if measurement is not None:
                measurement._close()
                self._measurements.pop()
                if (
                    isinstance(primary, CoordinateEnclosureResourceError)
                    and not self._evidence_measurements
                ):
                    primary.record_phase(measurement)
                self.stage_evidence.append(
                    CoordinateEnclosureStageEvidence(
                        start,
                        self._ticket.work_upper,
                        self._ticket.bytes_upper,
                        measurement.work_units,
                        measurement.peak_bytes_upper,
                    )
                )


def polynomial_coefficient_storage_upper(dimension: int, bits: int, /) -> int:
    """Canonical polynomial coefficient, key and bounded-integer storage."""
    return 512 + 32 * dimension + 2 * declared_integer_storage_bytes(max(bits, 1))


@checked
def reserve_polynomial(
    work: int,
    terms: int,
    dimension: int,
    bits: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Reserve coefficient work and the declared polynomial object-storage bound."""
    # Dictionary capacity/key tuples/Fraction objects plus two bounded PyLongs.
    per_term = polynomial_coefficient_storage_upper(dimension, bits)
    budget.reserve(work, 256 + terms * per_term)
