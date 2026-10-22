# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Caller-owned, receipt-bound coordinate operands; never proof verdicts.

An execution may retain its already computed immutable coordinate banks for its
later consumers. The caller owns their lifetime explicitly. A fresh process or
an archive restoration reconstructs them; neither receipts nor runtime metadata
contain a scientific operand cache.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field as dataclass_field, fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import cast, Literal, NoReturn, TYPE_CHECKING
from weakref import ref, WeakKeyDictionary

import equinox as eqx
import jax
import numpy as np
from numpy.typing import NDArray


if TYPE_CHECKING:
    from .._meshcore import (
        NativeExecutionScope,
        NativeHostStorageWorkspace,
        PlcSourceConstraints,
    )
    from .._physical import SpatialCoordinateContract
    from ..geometry._domain_coverage import PiecewiseLinearDomain
    from ..geometry._embedding_state import Facets
    from ..meshing._audit import CellMeshAuditPolicy
    from ..meshing._contracts import VolumeMeshingSpec
    from ..meshing._measurements import NativeExecutionRecord
    from ..meshing._native_preparation import NativePreparationOverlap
    from ..meshing._volume_generation import NativeVolumeSchedule
    from ..meshing.providers._native_sources import NativePlcSource
    from ._cell_geometry import CellGeometrySpec
    from ._cell_mesh import CellMesh
    from ._coordinate_budget import CoordinateEnclosureBudget, CoordinateLiveStorage
    from ._coordinate_enclosure import (
        CoordinatePreparationKey,
        CoordinateSourceBank,
        ExactScalar,
        Expression,
        Polynomial,
    )
    from ._exact_plc_geometry import (
        _PreparedPlcEntityFrames,
        ExactPlcCellGeometrySource,
        PreparedExactPlcGeometry,
    )
    from ._exact_power_geometry import PreparedExactPowerGeometry

OwnerFields = tuple[tuple[object, type, tuple[str, ...], tuple[object, ...]], ...]
ContainerOwners = tuple[tuple[list[object], tuple[object, ...]], ...]

type PolynomialOperandPolynomials = tuple[
    tuple[tuple[tuple[int, ...], Fraction], ...], ...
]
type PolynomialOperandEntry = tuple[PolynomialOperandPolynomials, bool]
type SignedDeterminantOperandKey = tuple[
    Literal["signed-determinant"],
    CoordinatePreparationKey,
    str,
    int,
]
type PolynomialOperandKey = CoordinatePreparationKey | SignedDeterminantOperandKey
type PolynomialOperandSnapshot = tuple[
    tuple[PolynomialOperandKey, PolynomialOperandPolynomials, bool], ...
]
type BinarySourceRows = tuple[
    CoordinateSourceBank,
    np.ndarray,
    np.ndarray,
    tuple[object, ...],
    tuple[object, ...],
]
type AffineAreaKey = tuple[
    tuple[tuple[Fraction, ...], ...], tuple[tuple[Fraction, ...], ...]
]
type AffineAreaSnapshot = tuple[tuple[AffineAreaKey, Fraction], ...]
type ExactMeasureSnapshot = tuple[
    tuple[CoordinatePreparationKey, tuple[float, float]], ...
]


@dataclass
class _LiveAffineAreaOperands:
    """Canonical affine-area numerics whose earning native scope is still live."""

    scope: NativeExecutionScope
    storage: NativeHostStorageWorkspace
    source: tuple[tuple[str, tuple[int, ...], bytes], ...] | None = None
    _areas: dict[AffineAreaKey, Fraction] = dataclass_field(default_factory=dict)

    def _current(self, budget: CoordinateEnclosureBudget) -> bool:
        from .._meshcore import (
            current_native_execution_scope,
            NativeBatchTicket,
            NativeScopeTicket,
        )

        if _LIVE_AFFINE_AREAS.get() is not self:
            return False

        ticket = budget.ticket
        if not isinstance(ticket, (NativeBatchTicket, NativeScopeTicket)) or ticket.settled:
            return False
        current = current_native_execution_scope()
        while current is not None and current is not self.scope:
            current = current.parent
        owner: NativeExecutionScope | None = ticket.scope
        while owner is not None and owner is not self.scope:
            owner = owner.parent
        return current is not None and owner is not None

    def require_source(
        self, arrays: tuple[np.ndarray, ...], budget: CoordinateEnclosureBudget
    ) -> bool:
        if not self._current(budget) or any(array.dtype.hasobject for array in arrays):
            return False
        budget.reserve(
            sum(array.size for array in arrays), sum(array.nbytes for array in arrays)
        )
        source = tuple((array.dtype.str, array.shape, array.tobytes()) for array in arrays)
        if self.source is None:
            self.storage.retain_owner(source)
            self.source = source
        return source == self.source

    def intersection(self, key: AffineAreaKey, budget: CoordinateEnclosureBudget) -> Fraction:
        from ..geometry._planar_primitives import intersection

        if not self._current(budget):
            return intersection(*key, budget=budget)
        budget.reserve(1, 128)
        area = self._areas.get(key)
        if area is None:
            area = intersection(*key, budget=budget)
            self.storage.retain_owner((key, area))
            self.storage.set_bound(self.storage.bound + 128)
            self._areas[key] = area
        return area


_LIVE_AFFINE_AREAS: ContextVar[_LiveAffineAreaOperands | None] = ContextVar(
    "live_coordinate_affine_areas", default=None
)


def live_affine_area_operands(
    vertices: np.ndarray, facets: np.ndarray, regions: np.ndarray,
    *, budget: CoordinateEnclosureBudget,
) -> _LiveAffineAreaOperands | None:
    """Authenticate current source bytes and the actual native receiving scope."""
    operands = _LIVE_AFFINE_AREAS.get()
    return (
        operands if operands is not None
        and operands.require_source((vertices, facets, regions), budget) else None
    )


def coordinate_input_owner_fields(
    roots: tuple[object, ...],
    budget: CoordinateEnclosureBudget,
    /,
    *,
    known: tuple[CoordinateOperandProfile, ...] = (),
    arrays: list[jax.Array | np.ndarray] | None = None,
    containers: list[tuple[list[object], tuple[object, ...]]] | None = None,
) -> OwnerFields | None:
    """Use the predecessor owner's immutable-JAX/actual-field admission rule."""
    from .._model._artifacts import artifact_value_codec_for
    from .._strict import Strict

    with budget.temporary_scope():
        pending = list(roots)
        seen: dict[int, object] = {}
        result: list[tuple[object, type, tuple[str, ...], tuple[object, ...]]] = []
        budget.reserve(0, 512 + 16 * len(pending))
        while pending:
            owner = pending.pop()
            if seen.get(id(owner)) is owner:
                continue
            budget.reserve(1, 128)
            seen[id(owner)] = owner
            if any(profile.owners.get(id(owner)) is owner for profile in known):
                continue
            if owner is None or type(owner) in (bool, int, float, complex, str, bytes):
                continue
            if isinstance(owner, jax.Array):
                if arrays is not None:
                    arrays.append(owner)
                continue
            if type(owner) is np.ndarray and arrays is not None:
                # Public host banks always require a complete logical-byte
                # rescan, even when their reversible write flag is false.
                if owner.dtype.hasobject or owner.dtype.kind not in "biufc":
                    return None
                arrays.append(owner)
                continue
            if type(owner) is list and containers is not None:
                budget.reserve(len(owner), 64 + 16 * len(owner))
                children = tuple(owner)
                containers.append((cast(list[object], owner), children))
                pending.extend(children)
                continue
            if type(owner) is tuple:
                budget.reserve(0, 64 + 16 * len(owner))
                pending.extend(owner)
                continue
            if type(owner) is dict or type(owner) is MappingProxyType:
                # Mutable container edges are fully validated, not trusted by
                # identity. Every child must separately satisfy this protocol.
                mapping = cast(Mapping[object, object], owner)
                keys = tuple(mapping)
                if any(type(name) is not str for name in keys):
                    return None
                names = cast(tuple[str, ...], keys)
                budget.reserve(len(names), 128 + 32 * len(names))
                values = tuple(mapping[name] for name in names)
                result.append((owner, type(owner), names, values))
                pending.extend(values)
                continue
            if isinstance(owner, tuple):
                field_names = getattr(type(owner), "_fields", None)
                if type(field_names) is not tuple or len(field_names) != len(owner):
                    return None
                budget.reserve(len(field_names), 128 + 64 * len(field_names))
                if (
                    any(type(name) is not str for name in field_names)
                    or len(set(field_names)) != len(field_names)
                    or hasattr(owner, "__dict__")
                ):
                    return None
                field_names = cast(tuple[str, ...], field_names)
                if any(
                    not hasattr(owner, name) or getattr(owner, name) is not owner[index]
                    for index, name in enumerate(field_names)
                ):
                    return None
                names = ("_fields", *field_names)
            elif type(owner) is Fraction:
                names = ("numerator", "denominator")
            elif isinstance(owner, Enum) and type(owner.value) in (bool, int, float, str):
                names = ("name", "value")
            elif is_dataclass(owner) and not isinstance(owner, type):
                names = tuple(member.name for member in fields(owner))
            elif isinstance(owner, Strict) and hasattr(owner, "__dict__"):
                # Strict value objects freeze declared metadata without being
                # Equinox dataclasses. Check every real attribute on admission.
                names = tuple(vars(owner))
            elif artifact_value_codec_for(owner) is not None and hasattr(
                owner, "__dict__"
            ):
                names = tuple(vars(owner))
            else:
                # A NumPy write flag, mutable mapping or foreign object is not
                # an immutable owner. Its normal preparation remains required.
                return None
            budget.reserve(len(names), 128 + 32 * len(names))
            values = tuple(getattr(owner, name) for name in names)
            result.append((owner, type(owner), names, values))
            pending.extend(values)
        return tuple(result)


def _coordinate_input_owner_matches(
    owner: object,
    owner_type: type,
    names: tuple[str, ...],
    values: tuple[object, ...],
    /,
) -> bool:
    """Authenticate one recorded owner without admitting additional work."""
    if type(owner) is dict or type(owner) is MappingProxyType:
        mapping = cast(Mapping[object, object], owner)
        return (
            type(owner) is owner_type
            and tuple(mapping) == names
            and all(
                mapping[name] is value for name, value in zip(names, values, strict=True)
            )
        )
    if isinstance(owner, tuple):
        return (
            type(owner) is owner_type
            and bool(names)
            and names[0] == "_fields"
            and len(owner) + 1 == len(values)
            and getattr(type(owner), "_fields", None) is values[0]
            and all(
                item is values[index + 1] and getattr(owner, names[index + 1]) is item
                for index, item in enumerate(owner)
            )
        )
    if type(owner) is not owner_type or (
        not is_dataclass(owner)
        and not isinstance(owner, Enum)
        and type(owner) is not Fraction
        and tuple(vars(owner)) != names
    ):
        return False
    return all(
        getattr(owner, name) is value for name, value in zip(names, values, strict=True)
    )


def require_coordinate_input_owners(
    owners: OwnerFields,
    budget: CoordinateEnclosureBudget,
    /,
) -> bool:
    for owner, owner_type, names, values in owners:
        budget.reserve(1 + len(names))
        if not _coordinate_input_owner_matches(owner, owner_type, names, values):
            return False
    return True


def _coordinate_input_container_matches(
    owner: list[object],
    children: tuple[object, ...],
    /,
) -> bool:
    """Authenticate one recorded mutable sequence without new admission."""
    return (
        type(owner) is list
        and len(owner) == len(children)
        and all(
            actual is retained for actual, retained in zip(owner, children, strict=True)
        )
    )


def require_coordinate_input_containers(
    containers: ContainerOwners,
    budget: CoordinateEnclosureBudget,
    /,
) -> bool:
    """Authenticate every actual mutable sequence edge, not container equality."""
    with budget.temporary_scope():
        for owner, children in containers:
            budget.reserve(1 + len(children), 128 + 16 * len(children))
            if not _coordinate_input_container_matches(owner, children):
                return False
    return True


@dataclass(frozen=True, slots=True)
class CoordinateOperandProfile:
    """An actual coefficient traversal and its immutable owner index."""

    basis: tuple[object, ...]
    owners: MappingProxyType[int, object]
    owner_fields: OwnerFields
    physical_bytes: int


class CoordinateOperandProfileBuilder:
    """Record immutable ownership during the already charged first traversal."""

    def __init__(self, *, private_atoms: bool = False) -> None:
        self.supported = True
        self.private_atoms = private_atoms
        self.owner_fields: list[
            tuple[object, type, tuple[str, ...], tuple[object, ...]]
        ] = []

    def storage_upper(self, value: object) -> int:
        return (
            768
            if self.supported and not self.private_atoms and type(value) is Fraction
            else 0
        )

    def observe(self, value: object, children: Iterable[object]) -> None:
        if not self.supported:
            return
        if type(value) is Fraction:
            if self.private_atoms:
                # Only kernel-produced opaque banks use this path. Fraction's
                # public API is immutable; unsafe mutation of its private slots
                # is outside this private cache boundary.
                return
            values = tuple(children)
            self.owner_fields.append(
                (value, Fraction, ("numerator", "denominator"), values)
            )
        elif value is not None and type(value) not in (
            tuple,
            bool,
            int,
            float,
            complex,
            str,
            bytes,
        ):
            self.supported = False


@dataclass(frozen=True, slots=True)
class PlcSourceInputWitness:
    """Source-only numerical inputs, not a full native carrier or proof verdict."""

    points: NDArray[np.float64]
    faces: NDArray[np.int32]
    face_group_rows: NDArray[np.int32]
    face_tolerances: NDArray[np.float64]
    segments: NDArray[np.int32]
    segment_group_rows: NDArray[np.int32]
    segment_tolerances: NDArray[np.float64]
    witness_strata: NDArray[np.int8]
    witness_entities: NDArray[np.int32]
    witness_parameters: NDArray[np.float64]
    face_ids: NDArray[np.int64]
    segment_ids: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PlcSourceCoordinateOperands:
    """Actual source-only numerical inputs and their full generating owners."""

    source: NativePlcSource
    specification: VolumeMeshingSpec
    coordinate_contract: SpatialCoordinateContract
    constraints: PlcSourceConstraints
    domain: PiecewiseLinearDomain | None
    witness: PlcSourceInputWitness | None
    schedule: NativeVolumeSchedule
    audit_policy: CellMeshAuditPolicy


@dataclass(frozen=True, slots=True, eq=False)
class _SignedPolynomialIssue:
    snapshot: PolynomialOperandPolynomials
    polynomial: Polynomial


@dataclass(frozen=True, slots=True, eq=False)
class _PlcEntityOperands:
    """Numeric rows attached to their actual source, never source-ID claims."""

    source: ExactPlcCellGeometrySource
    frames: _PreparedPlcEntityFrames


@dataclass(frozen=True, slots=True, eq=False)
class _CoordinateOperandData:
    mesh: CellMesh
    geometry: CellGeometrySpec
    _source_banks: MappingProxyType[tuple[str, str], CoordinateSourceBank]
    _plc_witnesses: MappingProxyType[tuple[str, str], PreparedExactPlcGeometry]
    _vertex_images: MappingProxyType[
        tuple[str, str, str], tuple[tuple[ExactScalar, ...] | None, ...]
    ]
    _carrier_keys: frozenset[tuple[str, str, str]]
    _identity_keys: frozenset[tuple[str, str, str]]
    _profiles: tuple[CoordinateOperandProfile, ...]
    _input_owners: OwnerFields
    _physical_bytes: int
    _facets: MappingProxyType[tuple[str, str, str], Facets]
    _binary_rows: MappingProxyType[int, BinarySourceRows]
    _array_layouts: tuple[tuple[np.ndarray, tuple[object, ...]], ...]
    _polynomial_snapshots: PolynomialOperandSnapshot
    _affine_areas: AffineAreaSnapshot
    _measure_snapshots: ExactMeasureSnapshot
    _input_arrays: tuple[tuple[np.ndarray, str, tuple[int, ...], bytes], ...]
    _input_immutable_arrays: tuple[jax.Array, ...]
    _input_containers: ContainerOwners
    _plc_source: PlcSourceCoordinateOperands | None
    _power_witnesses: MappingProxyType[str, PreparedExactPowerGeometry]
    _coordinate_contract: SpatialCoordinateContract | None
    _plc_entities: tuple[_PlcEntityOperands, ...]


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class PreparedCoordinateOperands(_CoordinateOperandData):
    """An issued reconstruction bank, never a proof verdict or runtime record."""

    earning_receipt: NativeExecutionRecord

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            "PreparedCoordinateOperands must detach at archive/cold boundaries."
        )

    def require(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        paid_receipt: NativeExecutionRecord,
        budget: CoordinateEnclosureBudget,
        /,
        *,
        coordinate_contract: SpatialCoordinateContract | None = None,
    ) -> None:
        _ADMITTED_OPERAND_CONTEXTS.pop(budget, None)
        seal = _ISSUED.get(self) if type(self) is PreparedCoordinateOperands else None
        current: NativeExecutionRecord | None = paid_receipt
        belongs = False
        while current is not None:
            if current is self.earning_receipt:
                belongs = True
                break
            current = current.preparation_evidence
        actual = _operand_fields(self)
        if (
            seal is None
            or len(actual) != len(seal)
            or any(a is not b for a, b in zip(actual, seal, strict=True))
            or mesh is not self.mesh
            or geometry is not self.geometry
            or not belongs
            or coordinate_contract is not self._coordinate_contract
        ):
            raise ValueError(
                "PreparedCoordinateOperands must bind the current source and its exact paid earning receipt."
            )
        self.earning_receipt.require_valid()
        from ..meshing._native_preparation import native_preparation_ledger

        scope = _coordinate_operand_request(budget)
        ledger = native_preparation_ledger(scope)
        if not (
            ledger.receipt_is_accounted(scope, self.earning_receipt)
            or ledger.completed_coordinate_source_is_authenticated(
                scope,
                mesh,
                geometry,
                coordinate_contract,
                paid_receipt,
                self.earning_receipt,
                budget,
            )
        ):
            raise ValueError(
                "PreparedCoordinateOperands earning phase must already be paid by the actual current native request."
            )
        budget.reserve(len(self._input_immutable_arrays))
        if any(array.is_deleted() for array in self._input_immutable_arrays):
            raise ValueError(
                "PreparedCoordinateOperands source immutable arrays must remain live after issue."
            )
        if not require_coordinate_input_containers(self._input_containers, budget):
            raise ValueError(
                "PreparedCoordinateOperands source containers changed after issue."
            )
        for array, dtype, shape, payload in self._input_arrays:
            with budget.temporary_scope():
                budget.reserve(array.size + 1, array.nbytes + 256)
                if (
                    array.dtype.str != dtype
                    or array.shape != shape
                    or array.dtype.hasobject
                    or array.tobytes(order="C") != payload
                ):
                    raise ValueError(
                        "PreparedCoordinateOperands source array changed after issue."
                    )
        if not require_coordinate_input_owners(self._input_owners, budget):
            raise ValueError(
                "PreparedCoordinateOperands source owners changed after issue."
            )
        for profile in self._profiles:
            if not require_coordinate_input_owners(profile.owner_fields, budget):
                raise ValueError(
                    "PreparedCoordinateOperands exact operand owners changed after issue."
                )
        for array, layout in self._array_layouts:
            budget.reserve(1)
            if _immutable_array_layout(array) != layout:
                raise ValueError(
                    "PreparedCoordinateOperands numeric view metadata changed."
                )
        if self._plc_entities:
            from ._exact_plc_geometry import _plc_entity_frame_basis

            for entry in self._plc_entities:
                with budget.temporary_scope():
                    count = len(entry.frames.triangles) + len(entry.frames.segments)
                    budget.reserve(4 * count + 3, 512 + 64 * count)
                    _plc_entity_frame_basis(entry.frames, source=entry.source)
        budget.admit_coordinate_operands(self)
        # This is the actual caller's current operation, never a frame inferred
        # from a saved artifact. Entity lookup reauthenticates it before use.
        _ADMITTED_OPERAND_CONTEXTS[budget] = (
            self,
            mesh,
            geometry,
            paid_receipt,
            coordinate_contract,
        )


def _coordinate_operand_request(
    budget: CoordinateEnclosureBudget, /
) -> NativeExecutionScope:
    """Bind numerical admission to the actual live receiving request."""
    from .._meshcore import (
        current_native_execution_scope,
        NativeBatchTicket,
        NativeScopeTicket,
    )

    ticket = budget.ticket
    actual = current_native_execution_scope()
    if (
        not isinstance(ticket, (NativeBatchTicket, NativeScopeTicket))
        or ticket.scope.evidence is not None
        or actual is None
        or actual.evidence is not None
    ):
        raise ValueError("Coordinate operands require an actual live native request.")
    root = ticket.scope
    while root.parent is not None:
        root = root.parent
    while actual.parent is not None:
        actual = actual.parent
    if actual is not root or root.evidence is not None:
        raise ValueError("Coordinate operands cannot cross native request ownership.")
    return ticket.scope


def _require_coordinate_operand_source(
    geometry: CellGeometrySpec, budget: CoordinateEnclosureBudget, /
) -> None:
    """Reauthenticate the current caller before exposing an imported numeric bank."""
    context = _ADMITTED_OPERAND_CONTEXTS.get(budget)
    if context is None or context[2] is not geometry:
        raise ValueError("Coordinate operands must retain their current source admission.")
    prepared, mesh, _, receipt, contract = context
    prepared.require(mesh, geometry, receipt, budget, coordinate_contract=contract)


def _operand_fields(prepared: PreparedCoordinateOperands, /) -> tuple[object, ...]:
    return (
        *(getattr(prepared, member.name) for member in fields(prepared)),
        *(
            getattr(profile, member.name)
            for profile in prepared._profiles
            for member in fields(profile)
        ),
        *(
            getattr(facet, member.name)
            for facet in prepared._facets.values()
            for member in fields(facet)
        ),
        *(
            getattr(entry, member.name)
            for entry in prepared._plc_entities
            for member in fields(entry)
        ),
    )


_ISSUED: WeakKeyDictionary[PreparedCoordinateOperands, tuple[object, ...]] = (
    WeakKeyDictionary()
)
_ADMITTED_OPERAND_CONTEXTS: WeakKeyDictionary[
    CoordinateEnclosureBudget,
    tuple[
        PreparedCoordinateOperands,
        CellMesh,
        CellGeometrySpec,
        NativeExecutionRecord,
        SpatialCoordinateContract | None,
    ],
] = WeakKeyDictionary()
_ISSUED_MEASURES: WeakKeyDictionary[
    CoordinateEnclosureBudget,
    dict[CoordinatePreparationKey, tuple[float, float]],
] = WeakKeyDictionary()
_ISSUED_PLC_ENTITIES: WeakKeyDictionary[
    CoordinateEnclosureBudget,
    dict[int, _PlcEntityOperands],
] = WeakKeyDictionary()
_ISSUED_SIGNED_POLYNOMIALS: WeakKeyDictionary[
    CoordinateEnclosureBudget,
    dict[SignedDeterminantOperandKey, _SignedPolynomialIssue],
] = WeakKeyDictionary()
_ACTIVE_SIGNED_BATCHES: WeakKeyDictionary[
    CoordinateEnclosureBudget,
    _SignedDeterminantOperandBatch,
] = WeakKeyDictionary()


def _immutable_array_layout(array: np.ndarray, /) -> tuple[object, ...]:
    owner: object = array
    while type(owner) is np.ndarray:
        if owner.flags.writeable:
            raise ValueError(
                "Coordinate operands require immutable byte-backed numeric views."
            )
        owner = owner.base
    if type(owner) is not bytes:
        raise ValueError(
            "Coordinate operands must not trust a reversible NumPy write flag."
        )
    return (
        array.dtype.str,
        array.shape,
        array.strides,
        array.__array_interface__["data"][0],
        array.nbytes,
        id(owner),
        owner,
    )


def seal_coordinate_operand_array(
    array: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    copy: bool = False,
) -> np.ndarray:
    """Seal a final numeric bank on immutable bytes with its real storage cost.

    An artifact can request a private view even when the input already has an
    immutable backing; its array interpretation then cannot be changed through
    the producer's array object. Other final issuers can retain the existing
    immutable bytes and bind the array layout in their own operand seal.
    """
    if array.dtype.hasobject:
        raise ValueError(
            "Coordinate operand arrays must contain numeric values, not object pointers."
        )
    if not copy:
        owner: object = array
        while type(owner) is np.ndarray and not owner.flags.writeable:
            owner = owner.base
        if type(owner) is bytes:
            return array
    with budget.live_storage() as storage:
        # The old working owner remains live until the immutable copy and its
        # two ndarray view headers have been admitted and retained.
        storage.set_bound(array.nbytes + 512 + 16 * array.ndim)
        sealed = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
        budget.retain_owner(sealed, temporary_storage_credit=storage)
        return sealed


def _remember_coordinate_polynomial_operands(
    key: CoordinatePreparationKey,
    coordinates: tuple[Expression, ...],
    budget: CoordinateEnclosureBudget,
    /,
    *,
    polynomial: bool,
) -> None:
    """Seal actual numeric kernel output before mutable working maps escape."""
    if not budget.capture_polynomial_operands or not budget.capture_coordinate_operands:
        return
    if any(type(value) is not dict for value in coordinates):
        return
    polynomials = tuple(value for value in coordinates if isinstance(value, dict))
    budget.reserve(1 + sum(len(value) for value in polynomials))
    if any(
        type(coefficient) is not Fraction
        for value in polynomials
        for coefficient in value.values()
    ):
        return
    existing = budget.polynomial_operand_snapshots.get(key)
    if existing is not None:
        snapshot, was_polynomial = existing
        if len(polynomials) != len(snapshot) or any(
            len(value) != len(terms)
            or any(value.get(index) != coefficient for index, coefficient in terms)
            for value, terms in zip(polynomials, snapshot, strict=True)
        ):
            raise ValueError(
                "Coordinate numeric issue conflicts with an earned snapshot."
            )
        if polynomial and not was_polynomial:
            budget.polynomial_operand_snapshots[key] = snapshot, True
        return
    budget.reserve(
        0,
        256
        + sum(
            128
            + sum(
                128
                + 16 * len(index)
                + (abs(value.numerator).bit_length() + value.denominator.bit_length() + 7)
                // 8
                for index, value in term.items()
                if isinstance(value, Fraction)
            )
            for term in polynomials
        ),
    )
    snapshot = tuple(
        tuple(
            (index, Fraction(value.numerator, value.denominator))
            for index, value in term.items()
            if isinstance(value, Fraction)
        )
        for term in polynomials
    )
    budget.retain_basis((key,), operand=True)
    budget.retain_basis(snapshot, operand=True, _kernel_operands=True)
    budget.polynomial_operand_snapshots[key] = snapshot, polynomial


def _signed_determinant_operand_key(
    value: object,
    /,
) -> SignedDeterminantOperandKey | None:
    if (
        type(value) is tuple
        and len(value) == 4
        and value[0] == "signed-determinant"
        and type(value[1]) is tuple
        and len(value[1]) in (3, 4)
        and type(value[2]) is str
        and type(value[3]) is int
        and value[3] in (1, 2, 3)
    ):
        return cast("SignedDeterminantOperandKey", value)
    return None


def _require_signed_determinant_cache(
    key: SignedDeterminantOperandKey,
    snapshot: PolynomialOperandPolynomials,
    budget: CoordinateEnclosureBudget,
    /,
) -> Polynomial:
    issued = _ISSUED_SIGNED_POLYNOMIALS.get(budget)
    budget.reserve(1 + sum(len(terms) for terms in snapshot))
    entry = None if issued is None else issued.get(key)
    current = None if entry is None else entry.polynomial
    if (
        entry is None
        or entry.snapshot is not snapshot
        or len(snapshot) != 1
        or type(current) is not dict
        or len(current) != len(snapshot[0])
        or any(
            type(index) is not tuple
            or len(index) != key[3]
            or any(type(axis) is not int or axis < 0 for axis in index)
            or type(value) is not Fraction
            or current.get(index) != value
            or type(current.get(index)) is not Fraction
            for index, value in snapshot[0]
        )
    ):
        raise ValueError(
            "Signed determinant operands changed after their actual numeric issue."
        )
    existing = budget.composed_source_density_cache.get(key[1])
    if existing is not None and (
        type(existing) is not dict
        or len(existing) != len(snapshot[0])
        or any(existing.get(index) != value for index, value in snapshot[0])
    ):
        raise ValueError(
            "Signed determinant operands conflict with current source arithmetic."
        )
    return current


def _admit_signed_determinant_operand(
    key: SignedDeterminantOperandKey,
    snapshot: PolynomialOperandPolynomials,
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    """Import only the numeric role of an already authenticated issued artifact."""
    if len(snapshot) != 1:
        raise ValueError(
            "Signed determinant operands must contain exactly one signed polynomial."
        )
    from ._coordinate_budget import polynomial_coefficient_storage_upper

    budget.reserve(1 + len(snapshot[0]))
    current = budget.composed_source_density_cache.get(key[1])
    if current is not None and (
        type(current) is not dict
        or len(current) != len(snapshot[0])
        or any(current.get(index) != value for index, value in snapshot[0])
    ):
        raise ValueError(
            "Signed determinant operands conflict with current numeric preparation."
        )
    issued = _ISSUED_SIGNED_POLYNOMIALS.get(budget)
    previous = None if issued is None else issued.get(key)
    if previous is not None and previous.snapshot is not snapshot:
        raise ValueError(
            "Signed determinant numeric owners conflict on the same complete key."
        )
    if previous is None:
        budget.reserve(
            len(snapshot[0]),
            256
            + 96 * len(snapshot[0])
            + sum(
                polynomial_coefficient_storage_upper(
                    key[3],
                    max(
                        abs(coefficient.numerator).bit_length(),
                        coefficient.denominator.bit_length(),
                    ),
                )
                for _, coefficient in snapshot[0]
            ),
        )
        # The returned map may escape through mapped_density_expression.
        # Its atoms must not alias the private authentication snapshot.
        restored: Polynomial = {
            index: Fraction(coefficient.numerator, coefficient.denominator)
            for index, coefficient in snapshot[0]
        }
        budget.retain_basis((key[1], restored))
    if issued is None:
        budget.retain_operand_metadata(256)
        issued = {}
        _ISSUED_SIGNED_POLYNOMIALS[budget] = issued
    if previous is None:
        budget.retain_operand_metadata(256)
        issued[key] = _SignedPolynomialIssue(snapshot, restored)
    budget.polynomial_operand_snapshots[key] = snapshot, False


def _remember_signed_determinant_operands(
    authority: CoordinatePreparationKey,
    cell_kind: str,
    dimension: int,
    polynomial: Expression | None,
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    """Issue a detached raw signed polynomial, never positivity or validity."""
    if (
        not (
            budget.capture_coordinate_operands
            and budget.capture_signed_determinant_operands
        )
        or type(polynomial) is not dict
    ):
        return
    if cell_kind == "pyramid" or type(dimension) is not int or dimension not in (1, 2, 3):
        return
    key: SignedDeterminantOperandKey = (
        "signed-determinant",
        authority,
        cell_kind,
        dimension,
    )
    budget.reserve(1 + len(polynomial))
    if any(
        not isinstance(index, tuple)
        or type(index) is not tuple
        or len(index) != dimension
        or any(type(axis) is not int or axis < 0 for axis in index)
        or not isinstance(value, Fraction)
        or type(value) is not Fraction
        for index, value in polynomial.items()
    ):
        return
    validated = cast("dict[tuple[int, ...], Fraction]", polynomial)
    previous = budget.polynomial_operand_snapshots.get(key)
    if previous is not None:
        current = _require_signed_determinant_cache(key, previous[0], budget)
        if current != polynomial:
            raise ValueError(
                "Signed determinant numeric issue conflicts with its complete source key."
            )
        return
    budget.reserve(
        0,
        512
        + sum(
            256
            + 32 * len(index)
            + (abs(value.numerator).bit_length() + value.denominator.bit_length() + 7)
            // 8
            for index, value in validated.items()
        ),
    )
    # Unit multiplication and disjoint addition can alias exposed atoms. Copy
    # every atom before assigning the existing private-kernel ownership rule.
    snapshot: PolynomialOperandPolynomials = (
        tuple(
            (index, Fraction(value.numerator, value.denominator))
            for index, value in validated.items()
        ),
    )
    budget.retain_basis((key,), operand=True)
    budget.retain_basis(snapshot, operand=True, _kernel_operands=True)
    _admit_signed_determinant_operand(key, snapshot, budget)


def _closed_signed_element_owners(
    elements: tuple[object, ...],
    budget: CoordinateEnclosureBudget,
    storage: CoordinateLiveStorage,
    /,
) -> dict[int, object] | None:
    """Admit only existing owned lattice/composition programs, not callbacks."""
    from ._cell_geometry import (
        BarycentricCellGeometryElement,
        CoordinateTabulator,
        PolynomialComposedCellGeometryElement,
        RestrictedCellGeometryElement,
    )
    from .fem._reference_types import FiniteElementSpec

    storage.set_bound(1024 + 256 * len(elements))
    pending = list(elements)
    owners: dict[int, object] = {}
    while pending:
        element = pending.pop()
        if owners.get(id(element)) is element:
            continue
        budget.reserve(1)
        storage.set_bound(1024 + 256 * (len(owners) + len(pending) + 3))
        if type(element) is FiniteElementSpec:
            if (
                type(element.tabulator) is not CoordinateTabulator
                or element.cell_kind == "pyramid"
            ):
                return None
        elif type(element) is BarycentricCellGeometryElement:
            pending.append(element.source_element)
        elif type(element) is RestrictedCellGeometryElement:
            pending.append(element.source_element)
        elif type(element) is PolynomialComposedCellGeometryElement:
            pending.extend((element.source_element, element.chart_element))
        else:
            return None
        owners[id(element)] = element
    return owners


class _SignedDeterminantOperandBatch:
    """An explicit, nonportable lifetime for one closed finite-cell operation."""

    def __init__(
        self,
        budget: CoordinateEnclosureBudget,
        root: NativeExecutionScope,
        owners: dict[int, object],
        /,
    ) -> None:
        self._budget = ref(budget)
        self._root = ref(root)
        self._owners = owners
        self._active = True
        self._busy = False

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            "Signed determinant batches must not escape their actual operation."
        )

    def expire(self) -> None:
        self._active = False
        self._owners.clear()

    def _matches(self, budget: CoordinateEnclosureBudget, /) -> bool:
        from .._meshcore import (
            current_native_execution_scope,
            NativeBatchTicket,
            NativeScopeTicket,
        )

        root = self._root()
        ticket = budget.ticket
        if (
            not self._active
            or self._budget() is not budget
            or _ACTIVE_SIGNED_BATCHES.get(budget) is not self
            or root is None
            or root.evidence is not None
            or not isinstance(ticket, (NativeBatchTicket, NativeScopeTicket))
            or ticket.scope.evidence is not None
        ):
            self.expire()
            return False
        actual = current_native_execution_scope()
        while actual is not None and actual.parent is not None:
            actual = actual.parent
        paid = ticket.scope
        while paid.parent is not None:
            paid = paid.parent
        if actual is not root or paid is not root:
            self.expire()
            return False
        return True

    def for_element(self, element: object, /) -> bool:
        """Retire before an unsupported element's properties or callbacks run."""
        budget = self._budget()
        if budget is None or not self._matches(budget):
            return False
        budget.reserve(1)
        if self._owners.get(id(element)) is not element:
            self.expire()
            return False
        return True


@contextmanager
def _signed_determinant_operand_batch(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    budget: CoordinateEnclosureBudget,
    /,
) -> Iterator[_SignedDeterminantOperandBatch | None]:
    """Authenticate a physical caller once around a closed owned-program loop."""
    from .._meshcore import NativeBatchTicket, NativeScopeTicket
    from ._cell_geometry import CellGeometrySpec
    from ._cell_mesh import CellMesh

    previous = _ACTIVE_SIGNED_BATCHES.get(budget)
    if previous is not None:
        previous.expire()
        yield None
        return
    context = _ADMITTED_OPERAND_CONTEXTS.get(budget)
    issued = _ISSUED_SIGNED_POLYNOMIALS.get(budget)
    if (
        context is None
        or not issued
        or type(mesh) is not CellMesh
        or type(geometry) is not CellGeometrySpec
    ):
        yield None
        return
    prepared, actual_mesh, actual_geometry, receipt, contract = context
    if mesh is not actual_mesh or geometry is not actual_geometry or contract is None:
        raise ValueError(
            "Signed determinant batches must bind the actual receiving mesh, geometry and frame."
        )
    ticket = budget.ticket
    if not isinstance(ticket, (NativeBatchTicket, NativeScopeTicket)):
        yield None
        return
    root = ticket.scope
    while root.parent is not None:
        root = root.parent
    prepared.require(mesh, geometry, receipt, budget, coordinate_contract=contract)
    with budget.live_storage() as storage:
        owners = _closed_signed_element_owners(tuple(geometry.elements), budget, storage)
        if owners is None:
            yield None
            return
        batch = _SignedDeterminantOperandBatch(budget, root, owners)
        _ACTIVE_SIGNED_BATCHES[budget] = batch
        try:
            yield batch
            # Retiring the optimization cannot turn a source mutation into a
            # successful result, including mutations during a fallback.
            prepared.require(
                mesh, geometry, receipt, budget, coordinate_contract=contract
            )
        finally:
            batch.expire()
            _ACTIVE_SIGNED_BATCHES.pop(budget, None)


def _signed_determinant_operands_for(
    authority: CoordinatePreparationKey,
    cell_kind: str,
    dimension: int,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    element: object,
    batch: _SignedDeterminantOperandBatch | None = None,
) -> Polynomial | None:
    """Read only an issued numeric role on this actual physical caller's bank."""
    key: SignedDeterminantOperandKey = (
        "signed-determinant",
        authority,
        cell_kind,
        dimension,
    )
    if batch is None:
        batch = _ACTIVE_SIGNED_BATCHES.get(budget)
    if batch is not None and (
        batch._busy or not batch.for_element(element)
    ):
        batch.expire()
        return None
    if batch is not None:
        batch._busy = True
    try:
        issued = _ISSUED_SIGNED_POLYNOMIALS.get(budget)
        budget.reserve(1)
        entry = None if issued is None else issued.get(key)
        if entry is None:
            if batch is not None:
                batch.expire()
            return None
        snapshot = entry.snapshot
    finally:
        if batch is not None:
            batch._busy = False
    if batch is not None:
        if not batch._matches(budget):
            return None
        try:
            return _require_signed_determinant_cache(key, snapshot, budget)
        except BaseException:
            batch.expire()
            raise
    context = _ADMITTED_OPERAND_CONTEXTS.get(budget)
    if context is None:
        if budget._prepared_coordinate_operands is not None:
            raise ValueError("Signed operands require their current source admission.")
        return None
    prepared, mesh, geometry, receipt, contract = context
    if contract is None:
        return None
    prepared.require(
        mesh,
        geometry,
        receipt,
        budget,
        coordinate_contract=contract,
    )
    return _require_signed_determinant_cache(key, snapshot, budget)


def _same_exact_measure_bounds(value: object, expected: tuple[float, float], /) -> bool:
    """Compare the actual kernel-rounded bounds, including signed zero."""
    if type(value) is not tuple or len(value) != 2:
        return False
    left, right = value
    if (
        type(left) is not float
        or type(right) is not float
        or type(expected[0]) is not float
        or type(expected[1]) is not float
    ):
        return False
    return float.hex(left) == float.hex(expected[0]) and float.hex(right) == float.hex(
        expected[1]
    )


def _remember_exact_cell_measure_operand(
    key: CoordinatePreparationKey,
    integral: tuple[float, float],
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    """Issue only an actual freshly computed compact affine numerical result."""
    if not budget.capture_coordinate_operands or not budget.capture_measure_operands:
        return
    budget.reserve(1)
    if (
        len(key) != 4
        or type(key[0]) is not str
        or type(key[1]) is not str
        or not key[0].startswith("affine-rational-measure:")
        or key[2] != ()
        or type(key[3]) is not bytes
        or type(integral) is not tuple
        or len(integral) != 2
        or any(type(value) is not float for value in integral)
        or not all(math.isfinite(value) for value in integral)
        or not integral[0] > integral[1] >= 0
    ):
        return
    issued = _ISSUED_MEASURES.get(budget)
    previous = None if issued is None else issued.get(key)
    if previous is not None:
        if not _same_exact_measure_bounds(previous, integral):
            raise ValueError(
                "Exact coordinate measures conflict with an earned numerical issue."
            )
        return
    # The private index owns only kernel outputs, never a caller's mutable
    # working cache. Both its entry and the numeric payload are admitted.
    budget.retain_operand_metadata(256 + (256 if issued is None else 0))
    budget.retain_basis((key, integral), operand=True, _kernel_operands=True)
    if issued is None:
        issued = {}
        _ISSUED_MEASURES[budget] = issued
    issued[key] = integral


def _retained_coordinate_operand_profiles(
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[CoordinateOperandProfile, ...]:
    """Combine actual already-admitted owners, without another graph traversal."""
    imported = budget._prepared_coordinate_operands
    count = len(budget.coordinate_operand_profiles)
    if imported is not None:
        count += len(imported._profiles)
    budget.reserve(count, 128 + 16 * count)
    return (() if imported is None else imported._profiles) + tuple(
        budget.coordinate_operand_profiles
    )


def _record_plc_entity_frames(
    source: ExactPlcCellGeometrySource,
    frames: _PreparedPlcEntityFrames,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    temporary_storage_credit: CoordinateLiveStorage | None,
) -> None:
    """Retain the actual first-earner kernel bank; publication binds its frame."""
    if not budget.capture_coordinate_operands:
        return
    budget.reserve(1)
    issued = _ISSUED_PLC_ENTITIES.get(budget)
    previous = None if issued is None else issued.get(id(source))
    if previous is not None:
        if previous.source is not source or previous.frames is not frames:
            raise ValueError(
                "PLC entity numeric issue conflicts with its actual source owner."
            )
        return
    budget._retain_plc_entity_frames(
        source,
        frames,
        temporary_storage_credit=temporary_storage_credit,
    )
    # Source ownership is an ordinary scientific graph scan, not a structural
    # numeric-bank exception. Its JAX inputs remain independently authenticated.
    budget.retain_owner(source)
    budget.retain_operand_metadata(256 + (256 if issued is None else 0))
    if issued is None:
        issued = {}
        _ISSUED_PLC_ENTITIES[budget] = issued
    issued[id(source)] = _PlcEntityOperands(source, frames)


_PLC_ENTITY_ARRAY_NAMES = (
    "source_points",
    "source_triangles",
    "source_segments",
    "source_triangle_ids",
    "source_triangle_bounds",
    "source_segment_ids",
    "source_segment_bounds",
)


def _plc_entity_frames_for(
    origin: ExactPlcCellGeometrySource | None,
    arrays: tuple[np.ndarray, ...],
    policy: tuple[str, str, int, int],
    budget: CoordinateEnclosureBudget,
    /,
) -> _PreparedPlcEntityFrames | None:
    """Authenticate a current physical caller before reusing pure numeric rows."""
    if origin is None:
        return None
    context = _ADMITTED_OPERAND_CONTEXTS.pop(budget, None)
    if context is None:
        return None
    prepared, mesh, geometry, receipt, contract = context
    if contract is None:
        return None
    # A prior callback's successful admission is not ambient validation. Recheck
    # all actual mutable owners, arrays, frame and receipt at this operation.
    try:
        prepared.require(
            mesh,
            geometry,
            receipt,
            budget,
            coordinate_contract=contract,
        )
    finally:
        _ADMITTED_OPERAND_CONTEXTS.pop(budget, None)
    for entry in prepared._plc_entities:
        budget.reserve(1)
        if entry.source is not origin:
            continue
        budget.reserve(4)
        if policy != (
            origin.domain_source_id,
            origin.domain_source_revision,
            origin.maximum_work,
            origin.maximum_bits,
        ) or len(arrays) != len(_PLC_ENTITY_ARRAY_NAMES):
            return None
        for name, array in zip(_PLC_ENTITY_ARRAY_NAMES, arrays, strict=True):
            source_array = getattr(origin, name)
            if not isinstance(source_array, jax.Array) or source_array.is_deleted():
                raise ValueError(
                    "PLC entity operand source arrays must remain immutable and live."
                )
            with budget.temporary_scope():
                budget.reserve(source_array.size + 1, 512 + 3 * source_array.nbytes)
                actual = np.asarray(source_array)
                if (
                    type(array) is not np.ndarray
                    or actual.dtype != array.dtype
                    or actual.shape != array.shape
                    or actual.tobytes(order="C") != array.tobytes(order="C")
                ):
                    return None
        return entry.frames
    return None


class CoordinateOperandCollector:
    """Explicit request-context lifetime for a pending and latest issued bank."""

    def __init__(
        self,
        *,
        capture_polynomials: bool = False,
        capture_affine_areas: bool = False,
        capture_measures: bool = False,
        capture_signed_determinants: bool = False,
        source_operands: PreparedCoordinateOperands | None = None,
    ) -> None:
        self._capture_polynomials = capture_polynomials
        self.capture_affine_areas = capture_affine_areas
        self.capture_measures = capture_measures
        self.capture_signed_determinants = capture_signed_determinants
        self.source_operands = source_operands
        self.prepared: PreparedCoordinateOperands | None = None
        self._pending: _CoordinateOperandData | None = None
        self._scope: NativeExecutionScope | None = None
        self._overlap: NativePreparationOverlap | None = None
        self._owner_id: str | None = None

    @property
    def capture_polynomials(self) -> bool:
        return self._capture_polynomials

    @contextmanager
    def live_affine_areas(self, scope: NativeExecutionScope) -> Iterator[None]:
        """Keep only exact clip operands live through fresh downstream proofs.

        Unlike ``complete``, this numerical-only lifetime cannot export a
        receipt or prepare geometry/proof caches. Source bytes and each complete
        rational polygon key are checked; the native workspace owns all bytes
        until publication or failure closes this context.
        """
        from .._meshcore import NativeHostStorageWorkspace

        with NativeHostStorageWorkspace(scope) as storage:
            operands = _LiveAffineAreaOperands(scope, storage)
            storage.retain_owner(operands._areas)
            token = _LIVE_AFFINE_AREAS.set(operands)
            try:
                yield
            finally:
                _LIVE_AFFINE_AREAS.reset(token)
                operands._areas.clear()

    def begin(
        self,
        scope: NativeExecutionScope,
        owner_id: str | None,
        overlap: NativePreparationOverlap,
        /,
    ) -> None:
        if self._scope is not None:
            raise ValueError(
                "CoordinateOperandCollector must finish its previous execution before another begins."
            )
        self._scope, self._owner_id, self._overlap = scope, owner_id, overlap
        self._pending = None

    def capture(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        budget: CoordinateEnclosureBudget,
        scope: NativeExecutionScope,
        /,
        *,
        plc_source: PlcSourceCoordinateOperands | None = None,
        input_owners: tuple[object, ...] = (),
        coordinate_contract: SpatialCoordinateContract | None = None,
    ) -> None:
        from .._meshcore import NativeBatchTicket, NativeScopeTicket

        if not isinstance(budget.ticket, (NativeBatchTicket, NativeScopeTicket)):
            raise ValueError(
                "Coordinate operands must be paid by their actual native earning scope."
            )
        paid: NativeExecutionScope | None = _coordinate_operand_request(budget)
        while paid is not None and paid is not self._scope:
            budget.reserve(1)
            paid = paid.parent
        if paid is None:
            raise ValueError("Coordinate operands belong to another native earning request.")
        ancestor: NativeExecutionScope | None = scope
        while ancestor is not None and ancestor is not self._scope:
            ancestor = ancestor.parent
        if ancestor is None:
            raise ValueError(
                "CoordinateOperandCollector must capture from its actual earning execution scope."
            )
        budget.capture_coordinate_operands = False
        banks = dict(budget.source_bank_cache)
        images = dict(budget.coordinate_vertex_image_cache)
        requested = (*banks.values(), *images.values())
        profiles = _retained_coordinate_operand_profiles(budget)
        if any(
            not any(profile.owners.get(id(bank)) is bank for profile in profiles)
            for bank in requested
        ):
            self._pending = None
            return
        witnesses = dict(budget.plc_witness_cache)
        budget.reserve(
            len(budget.power_witness_cache),
            128 + 64 * len(budget.power_witness_cache),
        )
        power_witnesses = dict(budget.power_witness_cache)
        if power_witnesses:
            from ._exact_power_geometry import (
                _power_witness_operand_basis,
                PreparedExactPowerGeometry,
            )

            for identity, witness in power_witnesses.items():
                budget.reserve(8 + len(identity), 128)
                if type(witness) is not PreparedExactPowerGeometry:
                    raise ValueError(
                        "Power coordinate operands must retain the exact kernel-issued numeric owner."
                    )
                basis = _power_witness_operand_basis(witness)
                if identity != witness.source_id or not any(
                    len(profile.basis) == len(basis)
                    and all(a is b for a, b in zip(profile.basis, basis, strict=True))
                    for profile in profiles
                ):
                    raise ValueError(
                        "Power coordinate operands must bind their actual issued numeric profile."
                    )
        if plc_source is not None:
            budget.retain_owner(plc_source)
        input_arrays: list[jax.Array | np.ndarray] = []
        input_containers: list[tuple[list[object], tuple[object, ...]]] = []
        owners = coordinate_input_owner_fields(
            (
                mesh,
                geometry,
                coordinate_contract,
                plc_source,
                *witnesses.values(),
                *power_witnesses.values(),
                *input_owners,
            ),
            budget,
            known=profiles,
            arrays=input_arrays,
            containers=input_containers,
        )
        if owners is None:
            self._pending = None
            return
        entity_records: tuple[_PlcEntityOperands, ...] = ()
        if coordinate_contract is not None:
            issued_entities = _ISSUED_PLC_ENTITIES.get(budget)
            inherited_entities = (
                ()
                if budget._prepared_coordinate_operands is None
                else budget._prepared_coordinate_operands._plc_entities
            )
            candidates = (
                *inherited_entities,
                *(() if issued_entities is None else issued_entities.values()),
            )
            budget.reserve(len(candidates), 128 + 64 * len(candidates))
            entity_records = tuple(
                entry
                for entry in candidates
                if any(owner is entry.source for owner, _, _, _ in owners)
            )
            from ._exact_plc_geometry import _plc_entity_frame_basis

            for entry in entity_records:
                count = len(entry.frames.triangles) + len(entry.frames.segments)
                with budget.temporary_scope():
                    budget.reserve(4 * count + 3, 512 + 64 * count)
                    _plc_entity_frame_basis(entry.frames, source=entry.source)
        budget.reserve(len(input_arrays), 128 + 16 * len(input_arrays))
        immutable_arrays = tuple(
            array for array in input_arrays if isinstance(array, jax.Array)
        )
        if any(array.is_deleted() for array in immutable_arrays):
            raise ValueError(
                "Coordinate operand capture requires live immutable source arrays."
            )
        array_snapshots = []
        for array in input_arrays:
            if type(array) is np.ndarray:
                budget.reserve(array.size + 1, array.nbytes + 256)
                array_snapshots.append(
                    (array, array.dtype.str, array.shape, array.tobytes(order="C"))
                )
        affine_areas = ()
        if self.capture_affine_areas:
            affine_areas = tuple(budget.affine_intersection_operand_snapshots.items())
            budget.reserve(len(affine_areas), 128 + 64 * len(affine_areas))
            if any(
                budget.affine_chain_intersection_cache.get(key) != value
                for key, value in affine_areas
            ):
                raise ValueError("Coordinate triangle areas changed after numeric issue.")
            budget.capture_coordinate_operands = True
            try:
                budget.retain_basis(affine_areas, operand=True, _kernel_operands=True)
                profiles = _retained_coordinate_operand_profiles(budget)
            finally:
                budget.capture_coordinate_operands = False
        measure_snapshots: ExactMeasureSnapshot = ()
        if self.capture_measures:
            issued_measures = _ISSUED_MEASURES.get(budget)
            count = 0 if issued_measures is None else len(issued_measures)
            budget.reserve(count, 128 + 64 * count)
            measure_snapshots = (
                () if issued_measures is None else tuple(issued_measures.items())
            )
            for key, integral in measure_snapshots:
                budget.reserve(1)
                if not _same_exact_measure_bounds(
                    budget.exact_cell_measure_cache.get(key), integral
                ):
                    raise ValueError(
                        "Coordinate measures changed after their actual numerical issue."
                    )
            budget.capture_coordinate_operands = True
            try:
                budget.retain_basis(
                    measure_snapshots, operand=True, _kernel_operands=True
                )
                profiles = _retained_coordinate_operand_profiles(budget)
            finally:
                budget.capture_coordinate_operands = False
        from ._coordinate_identity import coordinate_scope_key

        polynomial_snapshots = ()
        if self._capture_polynomials or self.capture_signed_determinants:
            # Consume only kernel-sealed maps, never an arbitrary dictionary
            # inserted into or changed in a caller's current work cache.
            budget.reserve(
                len(budget.polynomial_operand_snapshots),
                128 + 64 * len(budget.polynomial_operand_snapshots),
            )
            budget.capture_coordinate_operands = True
            try:
                polynomial_snapshots = tuple(
                    (key, snapshot, polynomial)
                    for key, (
                        snapshot,
                        polynomial,
                    ) in budget.polynomial_operand_snapshots.items()
                    if (
                        self.capture_signed_determinants
                        if _signed_determinant_operand_key(key) is not None
                        else self._capture_polynomials
                    )
                )
                budget.retain_basis(polynomial_snapshots, operand=True)
                for key, snapshot, _ in polynomial_snapshots:
                    signed_key = _signed_determinant_operand_key(key)
                    if signed_key is not None:
                        _require_signed_determinant_cache(signed_key, snapshot, budget)
                        continue
                    current = budget.coordinate_cache.get(
                        cast("CoordinatePreparationKey", key)
                    )
                    budget.reserve(1 + sum(len(value) for value in snapshot))
                    if (
                        current is None
                        or len(current) != len(snapshot)
                        or any(
                            type(value) is not dict
                            or tuple(value.items()) != terms
                            or any(
                                type(coefficient) is not Fraction
                                for coefficient in value.values()
                            )
                            for value, terms in zip(current, snapshot, strict=True)
                        )
                    ):
                        raise ValueError(
                            "Coordinate working maps changed after numeric issue."
                        )
                profiles = _retained_coordinate_operand_profiles(budget)
            finally:
                budget.capture_coordinate_operands = False
        facets = {}
        scope_key = coordinate_scope_key(mesh, geometry)
        facet = budget.facet_cache.get(scope_key)
        arrays = []
        if facet is not None:
            members = {
                member.name: seal_coordinate_operand_array(
                    getattr(facet, member.name),
                    budget=budget,
                    copy=True,
                )
                for member in fields(facet)
            }
            private_facet = replace(facet, **members)
            facets[scope_key] = private_facet
            arrays.extend(members.values())
        binary_rows: dict[int, BinarySourceRows] = {}
        for identity, (
            bank,
            view,
            mask,
            _,
            _,
        ) in budget.source_binary64_rows_cache.items():
            if any(bank is candidate for candidate in requested):
                private_view = seal_coordinate_operand_array(
                    view, budget=budget, copy=True
                )
                private_mask = seal_coordinate_operand_array(
                    mask, budget=budget, copy=True
                )
                binary_rows[identity] = (
                    bank,
                    private_view,
                    private_mask,
                    _immutable_array_layout(private_view),
                    _immutable_array_layout(private_mask),
                )
                arrays.extend((private_view, private_mask))
        layouts = tuple((array, _immutable_array_layout(array)) for array in arrays)
        metadata = (
            1024
            + 128 * (len(banks) + len(images) + len(witnesses) + len(power_witnesses))
            + 512 * len(layouts)
            + sum(256 + 64 * len(names) for _, _, names, _ in owners)
            + sum(len(payload) + 256 for _, _, _, payload in array_snapshots)
            + sum(128 + 16 * len(children) for _, children in input_containers)
            + 128
            + 16 * len(immutable_arrays)
            + 128
            + 64 * len(measure_snapshots)
            + 256 * len(entity_records)
            + 128
            + 16 * len(profiles)
        )
        budget.retain_operand_metadata(metadata)
        self._pending = _CoordinateOperandData(
            mesh,
            geometry,
            MappingProxyType(banks),
            MappingProxyType(witnesses),
            MappingProxyType(images),
            frozenset(budget.plc_carrier_binding_cache),
            frozenset(budget.plc_identity_corner_cache),
            profiles,
            owners,
            budget.retained_basis_bytes,
            MappingProxyType(facets),
            MappingProxyType(binary_rows),
            layouts,
            polynomial_snapshots,
            affine_areas,
            measure_snapshots,
            tuple(array_snapshots),
            immutable_arrays,
            tuple(input_containers),
            plc_source,
            MappingProxyType(power_witnesses),
            coordinate_contract,
            entity_records,
        )

    def complete(self, receipt: NativeExecutionRecord, /) -> None:
        scope, overlap = self._scope, self._overlap
        try:
            if scope is None or overlap is None:
                raise ValueError(
                    "CoordinateOperandCollector must finish an actual begun execution."
                )
            expected = overlap.normalize(scope.completed_evidence())
            earning: NativeExecutionRecord | None = receipt
            while earning is not None and not bool(
                eqx.tree_equal(earning.native_evidence(), expected)
            ):
                earning = earning.preparation_evidence
            if (
                earning is None
                or earning.owner_id is None
                or (self._owner_id is not None and earning.owner_id != self._owner_id)
            ):
                raise ValueError(
                    "CoordinateOperandCollector receipt must contain its actual ended earning phase."
                )
            earning.require_valid()
            if self._pending is None:
                self.prepared = None
                return
            pending = self._pending
            artifact = PreparedCoordinateOperands(
                mesh=pending.mesh,
                geometry=pending.geometry,
                _source_banks=pending._source_banks,
                _plc_witnesses=pending._plc_witnesses,
                _vertex_images=pending._vertex_images,
                _carrier_keys=pending._carrier_keys,
                _identity_keys=pending._identity_keys,
                _profiles=pending._profiles,
                _input_owners=pending._input_owners,
                _physical_bytes=pending._physical_bytes,
                _facets=pending._facets,
                _binary_rows=pending._binary_rows,
                _array_layouts=pending._array_layouts,
                _polynomial_snapshots=pending._polynomial_snapshots,
                _affine_areas=pending._affine_areas,
                _measure_snapshots=pending._measure_snapshots,
                _input_arrays=pending._input_arrays,
                _input_immutable_arrays=pending._input_immutable_arrays,
                _input_containers=pending._input_containers,
                _plc_source=pending._plc_source,
                _power_witnesses=pending._power_witnesses,
                _coordinate_contract=pending._coordinate_contract,
                _plc_entities=pending._plc_entities,
                earning_receipt=earning,
            )
            _ISSUED[artifact] = _operand_fields(artifact)
            self.prepared = artifact
        finally:
            self._scope = self._overlap = self._owner_id = self._pending = None

    @property
    def pending(self) -> bool:
        return self._scope is not None

    def abort(self) -> None:
        """Release pending execution resources, preserving a prior accepted bank."""
        self._scope = self._overlap = self._owner_id = self._pending = None

    def for_result(
        self, mesh: CellMesh, geometry: CellGeometrySpec, /
    ) -> PreparedCoordinateOperands | None:
        prepared = self.prepared
        return (
            prepared
            if prepared is not None
            and prepared.mesh is mesh
            and prepared.geometry is geometry
            else None
        )
