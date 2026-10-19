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
from itertools import chain
from typing import Callable, Literal, TYPE_CHECKING, TypeAlias

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
    from phydrax.geometry._embedding_state import Facets

    from ..geometry.brep._patches import (
        _TensorDerivativeSource,
        MeridianSpans,
        RationalBezierPiece,
    )
    from ..meshing._plc_planar_support import PlanarMap
    from ._cell_geometry_validity import _PolynomialDifferential, CellValidityCertificate
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
    )
    from ._coordinate_source_snapshot import (
        BarycentricSourceSnapshot,
        SplineSourceSnapshot,
    )
    from ._exact_algebraic_geometry import PreparedExactAlgebraicGeometry
    from ._exact_plc_geometry import PreparedExactPlcGeometry
    from ._exact_power_geometry import PreparedExactPowerGeometry
    from .fem._reference_types import FiniteElementSpec

CoordinateEnclosureResource: TypeAlias = Literal[
    "coefficient_work", "polynomial_storage", "retained_basis"
]


def _coordinate_storage_children(value: object, /) -> tuple[int, Iterable[object]]:
    """Return the pending-owner bound and children of one coordinate owner."""
    from ._coordinate_enclosure import RationalPolynomial
    from ._exact_algebraic_geometry import RadicalElement

    match value:
        case dict():
            return 64 + 32 * len(value), chain(value.keys(), value.values())
        case tuple():
            return 64 + 16 * len(value), value
        case RationalPolynomial():
            return 96, (value.numerator, value.denominator)
        case Fraction():
            return 96, (value.numerator, value.denominator)
        case RadicalElement():
            return 80, (value.terms,)
        case _:
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
        self.affine_simplex_source_cache: dict[tuple[str, str], bool] = {}
        self.affine_injectivity_cache: set[tuple[tuple[ExactScalar, ...], ...]] = set()
        self.exact_orientation_preparation_cache: dict[
            tuple[str | tuple[int | Fraction, ...], ...],
            tuple[tuple[int | Fraction, ...], int | Fraction, int],
        ] = {}
        self.exact_orientation_sign_cache: dict[
            tuple[str | tuple[int | Fraction, ...], ...], int
        ] = {}
        self.exact_native_sign_cache: dict[tuple[str, bytes], int] = {}
        self.exact_predicate_evaluations: dict[str, int] = {}
        self.exact_predicate_cache_hits: dict[str, int] = {}
        self.source_derivative_bounds_cache: dict[
            tuple[str, int, bytes],
            tuple[NDArray[np.float64], NDArray[np.float64]],
        ] = {}
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
            tuple[str, str], PreparedPolynomialArguments
        ] = {}
        self.corner_weight_cache: dict[
            tuple[str, str], tuple[tuple[str, str], CoordinateCornerWeights]
        ] = {}
        self.corner_linear_cache: dict[
            CoordinateLinearKey,
            tuple[ExactScalar, ...],
        ] = {}
        self.coordinate_point_value_cache: dict[
            CoordinatePointKey, tuple[ExactScalar, ...]
        ] = {}
        self.multi_affine_cache: dict[tuple[str, str], bool] = {}
        self.density_cache: dict[CoordinatePreparationKey, tuple[Expression, ...]] = {}
        self.coordinate_cache: dict[CoordinatePreparationKey, tuple[Expression, ...]] = {}
        self.coordinate_corner_cache: dict[
            CoordinatePreparationKey, tuple[tuple[ExactScalar, ...], ...]
        ] = {}
        self.polynomial_cache: dict[CoordinatePreparationKey, tuple[Polynomial, ...]] = {}
        self.coefficient_preparation_cache: dict[
            CoordinatePreparationKey,
            tuple[CoordinateSourceBank, CoordinatePreparationKey],
        ] = {}
        self.source_bank_cache: dict[tuple[str, str], CoordinateSourceBank] = {}
        self.plc_witness_cache: dict[tuple[str, str], PreparedExactPlcGeometry] = {}
        self.power_witness_cache: dict[str, PreparedExactPowerGeometry] = {}
        self.algebraic_witness_cache: dict[
            tuple[str, str], PreparedExactAlgebraicGeometry
        ] = {}
        self.validity_certificate_cache: dict[str, CellValidityCertificate] = {}
        self.polynomial_differential_cache: dict[
            tuple[CoordinatePreparationKey, str, int, bool],
            _PolynomialDifferential,
        ] = {}
        self.planar_cell_map_cache: dict[tuple[str, str, str], tuple[PlanarMap, ...]] = {}
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
        self.bspline_span_cache: dict[str, tuple[RationalBezierPiece, ...]] = {}
        self.meridian_span_cache: dict[str, MeridianSpans] = {}
        self.prepared_cell_cache: dict[tuple[str, str, str], PreparedCoordinateCells] = {}
        self.facet_cache: dict[tuple[str, str, str], Facets] = {}
        self.axis_affine_polynomial_cache: MutableMapping[
            str, PreparedPolynomialCoefficients
        ] = {}
        self.source_snapshot_identity_cache: dict[int, str] = {}
        self.barycentric_source_snapshot_cache: dict[str, BarycentricSourceSnapshot] = {}
        self.spline_source_snapshot_cache: dict[str, SplineSourceSnapshot] = {}
        self.retained_objects: dict[int, object] = {}
        self._polynomial_preparation_objects: ChainMap[int, object] | None = None
        self._polynomial_preparation_storage: CoordinateLiveStorage | None = None
        self.retained_owner_objects: dict[int, object] = {}
        self.logical_array_bytes_upper = 0
        self._stage_work_bounds: list[int] = []
        self._stage_memory_bounds: list[int] = []
        self.stage_evidence: list[CoordinateEnclosureStageEvidence] = []
        self._activation_depth = 0
        self._storage_ticket: BatchTicket | None = None
        self._storage_owner: ManagedBytesContributor | None = None
        self._storage_factory: Callable[[int], BatchTicket] | None = None

    @property
    def ticket(self) -> BatchTicket:
        """Return the explicitly bound stage ticket."""
        return self._ticket

    @property
    def storage_ticket(self) -> BatchTicket | None:
        """Return the request's persistent cache-byte reservation, if present."""
        return self._storage_ticket

    def _refresh_retained_storage(self) -> None:
        factory = self._storage_factory
        if factory is None:
            raise RuntimeError(
                "CoordinateEnclosureBudget.retained_storage must have an explicit factory; got None."
            )
        prior = self._storage_ticket
        owner = self._storage_owner
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
        if self._storage_factory is not None or not self._ticket.settled:
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

    @contextmanager
    def bind_ticket(self, ticket: BatchTicket, /) -> Iterator[CoordinateEnclosureBudget]:
        """Bind the next phase after settlement, retaining request counters and caches."""
        if not self._ticket.settled or self._activation_depth:
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

    def _admit_ticket(self, work: int, storage_upper: int | None = None, /) -> None:
        """Check the next work and storage bound against the stage allowance."""
        ticket = self._ticket

        if ticket.settled:
            if work or (
                storage_upper is not None
                and storage_upper > self.retained_basis_bytes + self.temporary_bytes_upper
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
        retained = self._storage_ticket
        baseline = 0 if retained is None else retained.bytes_upper
        if work > ticket.remaining_work_units or (
            storage_upper is not None and storage_upper - baseline > ticket.bytes_upper
        ):
            ticket.refuse(
                work,
                bytes_upper=0 if storage_upper is None else storage_upper - baseline,
            )

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
        work = nonnegative_integer(work, "CoordinateEnclosureBudget.work")
        storage_upper = nonnegative_integer(
            storage_upper, "CoordinateEnclosureBudget.storage_upper"
        )
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

    def _object_storage_profile(
        self,
        root: object,
        existing: Mapping[int, object],
        /,
        *,
        retained: bool,
    ) -> tuple[dict[int, object], int]:
        storage_resource: CoordinateEnclosureResource = (
            "retained_basis" if retained else "polynomial_storage"
        )
        self._reserve_validated(0, 256, storage_resource)
        pending: list[object] = [root]
        seen: dict[int, object] = {}
        size = 256 if retained and not existing else 0
        storage_limit = self.maximum_memory_bytes
        if self._stage_memory_bounds:
            storage_limit = min(storage_limit, self._stage_memory_bounds[-1])
        while pending:
            value = pending.pop()
            identifier = id(value)
            if (identifier in seen and seen[identifier] is value) or (
                identifier in existing and existing[identifier] is value
            ):
                continue
            self._reserve_validated(1, 0, "polynomial_storage")
            # Conservative CPython hash-index slots. The transient traversal
            # index coexists with a retained index; live owners use the same
            # upper bound even though their lifetime is tracked separately.
            self._reserve_validated(0, 128, storage_resource)
            size += declared_object_storage_bytes(value) + 128
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
            seen[identifier] = value
            storage, children = _coordinate_storage_children(value)
            if storage:
                self._reserve_validated(0, storage, storage_resource)
                pending.extend(children)
        return seen, size

    def live_object_storage_upper(self, value: object, /) -> int:
        """Measure a live object graph without retaining it after traversal."""
        with self.temporary_scope():
            _, size = self._object_storage_profile(value, {}, retained=False)
        return size

    def retain_basis(self, basis: tuple[object, ...], /) -> None:
        """Profile each coefficient owner once during its request retention lifetime."""
        # Strong references prevent recycled IDs from earning another object's
        # profile. A retained root also owns every already-profiled graph member.
        if self.retained_objects.get(id(basis)) is basis:
            return
        with self.temporary_scope():
            seen, size = self._object_storage_profile(
                basis, self.retained_objects, retained=True
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

    @contextmanager
    def polynomial_preparation_scope(self) -> Iterator[None]:
        """Bound temporary polynomial memos to their consuming phase.

        Previously retained source preparations remain reusable. New affine
        snapshots and Bernstein controls live only until this scope exits;
        callers consume those coefficients before leaving the scope.
        """
        prior_affine = self.axis_affine_polynomial_cache
        prior_bernstein = self.bernstein_coefficient_cache
        prior_objects = self._polynomial_preparation_objects
        prior_storage = self._polynomial_preparation_storage
        with self.temporary_scope(), self.live_storage() as storage:
            # Three local dictionaries, ChainMap records and two-owner map lists.
            storage.set_bound(3 * (256 + 256 + 256 + 2 * 16))
            affine: dict[str, PreparedPolynomialCoefficients] = {}
            bernstein: dict[BernsteinCoefficientKey, tuple[ExactScalar, ...]] = {}
            objects: dict[int, object] = {}
            self.axis_affine_polynomial_cache = ChainMap(affine, prior_affine)
            self.bernstein_coefficient_cache = ChainMap(bernstein, prior_bernstein)
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
                self._polynomial_preparation_objects = prior_objects
                self._polynomial_preparation_storage = prior_storage
                affine.clear()
                bernstein.clear()
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
    per_term = 512 + 32 * dimension + 2 * declared_integer_storage_bytes(max(bits, 1))
    budget.reserve(work, 256 + terms * per_term)
