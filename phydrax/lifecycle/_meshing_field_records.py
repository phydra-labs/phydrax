#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Declare physical field semantics for native restart recipes.

Only primitive declarations and accepted arrays cross this boundary. Preparation
is performed by the existing FE/FV owners against the restored accepted carrier.

Cold restore passes its current source-validation owner explicitly. That owner
earns the unchanged epoch-clock content digest in the common replay phase and
authenticates the actual immutable bank before local field-index validation.
Local row comparisons consume admitted bytes; standalone calls retain their
original global content hashing. Accepted-epoch replay agrees local field
admission before computing its original declaration and state-bank hashes.
"""

from __future__ import annotations

import ctypes
import dataclasses
import hashlib
from abc import abstractmethod
from bisect import bisect_left
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import cache
from inspect import getattr_static
from math import prod
from operator import index, itemgetter
from os import getpid
from typing import Any, assert_never, cast, final, Literal, Self, SupportsIndex

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec

from phydrax.discretization.fem._reference_types import (
    AbstractStoredFiniteElementTabulator,
    FiniteElementSpec,
    FiniteElementTabulator,
)

from .._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
from .._differentiation import BranchDifferentiationPolicy
from .._fingerprint import (
    array_bytes_digest,
    canonical_fingerprint,
    canonical_json,
    logical_array_value_collection_digest,
)
from .._frozendict import frozendict
from .._model._structure import model_recipe_array_values, model_structure_recipe
from .._sampling._addressing import derive_key, SampleAddress
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import AdaptiveSimplexParts
from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import CoordinateTabulator
from ..discretization._cell_geometry_transfer import transition_displaced_cell_geometry
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._conservation_boundary import ExtrapolationBoundary
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._spaces import (
    DiscreteFieldSpace,
    EntityDofLayout,
    GlobalCoefficientId,
)
from ..discretization._views import FieldTracePolicy
from ..discretization.fem._form_elements import (
    form_element,
    FormBasis,
    HybridFactors,
    ProxyTabulator,
    TensorFactors,
)
from ..discretization.fem._generic import (
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from ..discretization.fem._high_order import ReferenceNodalFamily, SimplexNodalFamily
from ..discretization.fem._hybrid_reference import HybridReferenceFamily
from ..discretization.fem._precision import FiniteElementPrecisionPolicy
from ..discretization.fem._reference import discontinuous_element, lagrange_element
from ..discretization.finite_volume._cell_polynomial import (
    CellPolynomialReconstructionPlan,
    PreparedCellPolynomialReconstruction,
)
from ..discretization.finite_volume._physical_boundaries import SlipWallBoundary
from ..discretization.finite_volume._reconstruction import PiecewiseConstantReconstruction
from ..discretization.finite_volume._riemann import RusanovFluxPlan
from ..discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from ..discretization.finite_volume._unstructured_dynamics import (
    PreparedUnstructuredFiniteVolumeDynamics,
    UnstructuredFiniteVolumeBoundarySet,
    UnstructuredFiniteVolumeMethodPlan,
)
from ..discretization.finite_volume._unstructured_weno import (
    PreparedUnstructuredWENOZReconstruction,
    UnstructuredWENOZReconstructionPlan,
)
from ..equations._hyperbolic_systems import EulerSystem
from ..equations._materials import IdealGasMaterial
from ..exterior._form_type import FormType, FormValueSpec
from ..linalg._spaces import ArraySpace
from ..meshing._adaptation import MeshAdaptationResult
from ..meshing._collective_failure import agreed_local_step
from ..meshing._entity_keys import key_rows
from ..meshing._initial_certification import InitialCollectiveMeshEvidence
from ..meshing._lineage import CellMeshTransition, identity_lineage
from ..meshing._lineage_records import (
    EntityLineage,
    EntityLineageKind,
    MeshLineage,
    MeshTransitionKind,
)
from ..meshing._measurements import NativeExecutionRecord
from ..meshing._mesh_part import MeshPart
from ..meshing._overset import OversetPartSpec, OversetRegistration
from ..meshing._restart_distribution import (
    SimplexRestartRepack,
    SimplexRestartRepackProof,
    validate_simplex_restart_repack,
)
from ..meshing._result import CellMeshingResult, require_original_meshing_source
from ..meshing._scope import mesh_global_ids, MeshingEntityKind, MeshingScope
from ..meshing.qualification import _polyhedral as polyhedral_owner
from ..meshing.qualification._polyhedral import PolyhedralArchiveOwners
from ..sparse._ops import gather_routes
from ..sparse._relation import RowRelation
from ..typing import checked, parse
from ..units._dimension import DimensionSignature, LENGTH
from ..units._unit import UnitDefinition
from ._meshing_source_records import (
    content_equal,
    register_meshing_source_types,
    stored_record_content_id,
)


def _may_be_derived_coefficient_id(identity: str, /) -> bool:
    """Whether text has the SHA-256 hex form ``canonical_fingerprint`` returns."""
    return len(identity) == 64 and not identity.strip("0123456789abcdef")


@dataclass(frozen=True, slots=True, eq=False)
class MeshingCoefficientIdentityBank:
    """Ordered scientific coefficient identities in one numerical column bank.

    ``coordinates`` is a read-only C-contiguous int64 host array whose row ``r``
    is ``(ordinal, *component)``; components shorter than the widest one are
    padded with trailing ``-1``. Rows are unique and strictly increase in
    ``(ordinal, component)`` order. ``explicit_ids`` lists ``(row, id)``
    overrides in row order; every other row has the ``GlobalCoefficientId``
    fingerprint of the field space, ordinal and component, which overrides never
    repeat. Derived IDs are computed only for requested rows. A derived-form
    override requires a streaming collision check over the array-bounded rows;
    these temporary hash inputs are not serialized manifest data.
    """

    field_space_id: str
    coordinates: np.ndarray
    explicit_ids: tuple[tuple[int, str], ...] = ()

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[Self], value: MeshingCoefficientIdentityBank, /
    ) -> Self:
        """Reconstruct and validate the stored coefficient identities."""
        return cls(value.field_space_id, value.coordinates, value.explicit_ids)

    @classmethod
    def from_field_space(cls: type[Self], space: DiscreteFieldSpace, /) -> Self:
        """Bank every coefficient of an array field space under its derived ID."""
        return cls(space.field_space_id, _field_space_coordinates(space))

    @classmethod
    def from_rows(
        cls: type[Self],
        field_space_id: str,
        rows: tuple[tuple[int, tuple[int, ...]], ...],
        /,
        explicit_ids: tuple[tuple[int, str], ...] = (),
    ) -> Self:
        """Pack ordered ``(ordinal, component)`` rows into the coordinate bank."""
        if type(rows) is not tuple or any(
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not int
            or type(row[1]) is not tuple
            or any(type(value) is not int for value in row[1])
            for row in rows
        ):
            raise TypeError(
                "MeshingCoefficientIdentityBank.rows must be a tuple of (ordinal, component) integer tuples; got malformed rows"
            )
        maximum = np.iinfo(np.int64).max
        if any(
            not 0 <= value <= maximum
            for ordinal, component in rows
            for value in (ordinal, *component)
        ):
            raise ValueError(
                "MeshingCoefficientIdentityBank.rows must contain nonnegative int64 ordinals and components; got out-of-range indices"
            )
        width = 1 + max((len(component) for _, component in rows), default=0)
        if len(rows) * width > DEFAULT_ARRAY_ARCHIVE_LIMITS.max_array_elements:
            raise ValueError(
                "MeshingCoefficientIdentityBank.rows must fit the coordinate array element limit; got oversized rows"
            )
        coordinates = np.full((len(rows), width), -1, dtype=np.int64)
        for position, (ordinal, component) in enumerate(rows):
            coordinates[position, : 1 + len(component)] = (ordinal, *component)
        coordinates.setflags(write=False)
        return cls(field_space_id, coordinates, explicit_ids)

    def __post_init__(self) -> None:
        space = self.field_space_id
        if type(space) is not str or not space or space != space.strip():
            raise ValueError(
                "MeshingCoefficientIdentityBank.field_space_id must be a nonempty trimmed identity; got invalid identity"
            )
        coordinates = self.coordinates
        if (
            type(coordinates) is not np.ndarray
            or coordinates.dtype != np.int64
            or coordinates.ndim != 2
            or coordinates.shape[1] < 1
            or not coordinates.flags.c_contiguous
            or coordinates.flags.writeable
        ):
            raise TypeError(
                "MeshingCoefficientIdentityBank.coordinates must be a read-only C-contiguous two-dimensional int64 host array; got another coordinate representation"
            )
        if coordinates.size > DEFAULT_ARRAY_ARCHIVE_LIMITS.max_array_elements:
            raise ValueError(
                "MeshingCoefficientIdentityBank.coordinates must fit the array element limit; got oversized coordinates"
            )
        count, width = coordinates.shape
        components = coordinates[:, 1:]
        padding = components == -1
        if (
            (count == 0 and width != 1)
            or (count > 0 and width > 1 and bool(padding[:, -1].all()))
            or bool((coordinates[:, 0] < 0).any())
            or bool((components < -1).any())
            or bool((padding[:, :-1] & ~padding[:, 1:]).any())
        ):
            raise ValueError(
                "MeshingCoefficientIdentityBank.coordinates must hold nonnegative ordinals and components padded by trailing -1 to the widest component; got malformed or noncanonical rows"
            )
        if count > 1:
            changed = coordinates[1:] != coordinates[:-1]
            first = changed.argmax(axis=1)
            pairs = np.arange(count - 1)
            if not bool(changed.any(axis=1).all()) or not bool(
                (coordinates[1:][pairs, first] > coordinates[:-1][pairs, first]).all()
            ):
                raise ValueError(
                    "MeshingCoefficientIdentityBank.coordinates must have unique rows in strictly increasing (ordinal, component) order; got duplicate or unordered rows"
                )
        explicit = self.explicit_ids
        if type(explicit) is not tuple or any(
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not int
            or type(entry[1]) is not str
            for entry in explicit
        ):
            raise TypeError(
                "MeshingCoefficientIdentityBank.explicit_ids must be an immutable tuple of (row, coefficient ID) pairs; got malformed overrides"
            )
        identities = {identity for _, identity in explicit}
        if (
            len(identities) != len(explicit)
            or any(
                not identity or identity != identity.strip() for identity in identities
            )
            or any(not 0 <= row < count for row, _ in explicit)
            or any(
                later[0] <= earlier[0] for earlier, later in zip(explicit, explicit[1:])
            )
        ):
            raise ValueError(
                "MeshingCoefficientIdentityBank.explicit_ids must name unique trimmed IDs at strictly increasing in-range rows; got invalid overrides"
            )
        derived_form = {
            identity
            for identity in identities
            if _may_be_derived_coefficient_id(identity)
        }
        if not derived_form:
            # Other text cannot equal a derived SHA-256 hex ID.
            return
        overrides = dict(explicit)
        # Retain only authored overrides and one derived identity at a time.
        # The archive owner bounds the actual recipe, not cumulative hash input.
        for row in range(count):
            identity = overrides.get(row)
            if identity is not None and identity not in derived_form:
                continue
            ordinal, component = self._coordinate(row)
            derived = GlobalCoefficientId(space, ordinal, component).coefficient_id
            if derived == identity or (identity is None and derived in derived_form):
                raise ValueError(
                    "MeshingCoefficientIdentityBank.explicit_ids must differ from every derived coefficient ID; got a repeated or colliding ID"
                )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return (
            self.field_space_id == other.field_space_id
            and self.explicit_ids == other.explicit_ids
            and self.coordinates.shape == other.coordinates.shape
            and bool(np.array_equal(self.coordinates, other.coordinates))
        )

    @property
    def row_count(self) -> int:
        return self.coordinates.shape[0]

    def _coordinate(self, row: int, /) -> tuple[int, tuple[int, ...]]:
        ordinal, *component = self.coordinates[row].tolist()
        length = component.index(-1) if -1 in component else len(component)
        return ordinal, tuple(component[:length])

    def coefficient(self, row: SupportsIndex, /) -> GlobalCoefficientId:
        """Return one row's coefficient identity, deriving only this row's ID."""
        position = index(row)
        if not 0 <= position < self.row_count:
            raise IndexError(
                f"MeshingCoefficientIdentityBank.row must address a banked row; got {position} of {self.row_count}"
            )
        ordinal, component = self._coordinate(position)
        explicit = self.explicit_ids
        slot = bisect_left(explicit, position, key=itemgetter(0))
        return GlobalCoefficientId(
            self.field_space_id,
            ordinal,
            component,
            coefficient_id=explicit[slot][1]
            if slot < len(explicit) and explicit[slot][0] == position
            else None,
        )

    @checked
    def require_field_space(self, space: DiscreteFieldSpace, /) -> None:
        """Require exactly the space's ordered coefficients under derived IDs."""
        self.__post_init__()
        expected = _field_space_coordinates(space)
        if (
            self.field_space_id != space.field_space_id
            or self.explicit_ids
            or self.coordinates.shape != expected.shape
            or not np.array_equal(self.coordinates, expected)
        ):
            raise ValueError(
                "MeshingCoefficientIdentityBank.coordinates must match the field space in order under derived IDs; got missing, reordered, overridden, or foreign identities"
            )


def _field_space_coordinates(space: DiscreteFieldSpace, /) -> np.ndarray:
    """Ordered ``(ordinal, *component)`` rows of every array-space coefficient."""
    if not isinstance(space.vector_space, ArraySpace):
        raise TypeError(
            "DiscreteFieldSpace.vector_space must be an ArraySpace for coefficient rows; got unsupported space"
        )
    shape = space.vector_space.shape
    if not shape:
        raise ValueError(
            "DiscreteFieldSpace.vector_space.shape must have a coefficient-row axis; got scalar space"
        )
    components = prod(shape[1:])
    count = shape[0] * components
    width = len(shape) if count else 1
    if count * width > DEFAULT_ARRAY_ARCHIVE_LIMITS.max_array_elements:
        raise ValueError(
            "DiscreteFieldSpace.vector_space.shape must fit the coefficient bank array element limit; got oversized field space"
        )
    coordinates = np.empty((count, width), dtype=np.int64)
    if count:
        coordinates[:, 0] = np.repeat(np.arange(shape[0], dtype=np.int64), components)
        coordinates[:, 1:] = np.tile(
            np.indices(shape[1:], dtype=np.int64).reshape(len(shape) - 1, components).T,
            (shape[0], 1),
        )
    coordinates.setflags(write=False)
    return coordinates


type MeshingFieldIndexBank = Literal[
    "field/global_coefficient_ids",
    "mesh/vertex_ids",
    "mesh/cell_ids",
    "epoch/vertex_ids",
    "epoch/cell_ids",
]


@dataclass(frozen=True, kw_only=True)
class MeshingFieldIndexBinding:
    """Bind field, mesh-entity, or raw-epoch rows to scientific identities.

    Raw epoch banks include retired records and padding. ``index_values``
    retains bank order; a mesh scope describes membership.
    """

    bank_name: MeshingFieldIndexBank
    source_result_id: str
    source_topology_id: str
    source_geometry_id: str
    source_evidence_id: str | None
    source_epoch_id: str | None
    index_values: jax.Array | np.ndarray
    index_axes: tuple[int, ...]
    scope: MeshingScope | None = None
    field_space_id: str | None = None
    coefficient_bank: MeshingCoefficientIdentityBank | None = None

    @classmethod
    @checked
    def from_stored_fields(cls: type[Self], value: MeshingFieldIndexBinding, /) -> Self:
        """Reconstruct and validate the stored index binding."""
        return cls(
            **{
                member.name: getattr(value, member.name)
                for member in dataclasses.fields(value)
            }
        )

    def __post_init__(self) -> None:
        parse(self.bank_name, MeshingFieldIndexBank, "MeshingFieldIndexBinding.bank_name")
        if any(
            type(value) is not str or not value or value != value.strip()
            for value in (
                self.source_result_id,
                self.source_topology_id,
                self.source_geometry_id,
            )
        ):
            raise ValueError(
                "MeshingFieldIndexBinding.source_result_id must be trimmed and accompany nonempty trimmed topology and geometry IDs; got invalid source identities"
            )
        if not isinstance(
            self.index_values, (jax.Array, np.ndarray)
        ) or self.index_values.dtype != np.dtype("int64"):
            raise TypeError(
                "MeshingFieldIndexBinding.index_values must be an int64 array; got incompatible values"
            )
        raw = self.bank_name.startswith("epoch/")
        axes = (0, 1) if raw else (0,)
        if self.index_axes != axes or self.index_values.ndim != len(axes):
            raise ValueError(
                "MeshingFieldIndexBinding.index_axes must match mesh or raw-bank rank; got incompatible axes"
            )
        if raw and (
            not self.source_evidence_id
            or not self.source_epoch_id
            or self.scope is not None
        ):
            raise ValueError(
                "MeshingFieldIndexBinding.source_evidence_id must bind a native source_epoch_id without a compact scope; got incompatible raw binding"
            )
        if self.bank_name.startswith("mesh/") and type(self.scope) is not MeshingScope:
            raise TypeError(
                "MeshingFieldIndexBinding.scope must be a MeshingScope for mesh rows; got missing or invalid scope"
            )
        if (
            self.bank_name == "field/global_coefficient_ids"
            and self.field_space_id is None
        ):
            raise ValueError(
                "MeshingFieldIndexBinding.field_space_id must be declared for coefficient ordinals; got missing identity"
            )
        if self.field_space_id is not None and (
            type(self.field_space_id) is not str or not self.field_space_id
        ):
            raise ValueError(
                "MeshingFieldIndexBinding.field_space_id must be nonempty; got invalid identity"
            )
        if self.field_space_id is None:
            if self.coefficient_bank is not None:
                raise ValueError(
                    "MeshingFieldIndexBinding.coefficient_bank must be absent without field_space_id; got undeclared coefficient identities"
                )
        elif type(self.coefficient_bank) is not MeshingCoefficientIdentityBank:
            raise TypeError(
                "MeshingFieldIndexBinding.coefficient_bank must be a MeshingCoefficientIdentityBank; got missing or invalid bank"
            )
        else:
            self.coefficient_bank.__post_init__()
            if self.coefficient_bank.field_space_id != self.field_space_id:
                raise ValueError(
                    "MeshingFieldIndexBinding.coefficient_bank.field_space_id must match field_space_id; got foreign bank"
                )


@dataclass(frozen=True, kw_only=True)
class MeshingPolyhedralOwnerBinding:
    """Actual physical plans and their ordered polyhedral scientific banks."""

    source_result_id: str
    source_mesh_id: str
    source_topology_id: str
    source_geometry_id: str
    source_geometry_layout_id: str
    coordinate_contract_id: str
    numeric_version: str
    preparation_ids: frozendict[str, str]
    field_space_ids: frozendict[str, str]
    field_names: frozendict[str, str]
    coefficient_banks: frozendict[str, MeshingCoefficientIdentityBank]
    index_banks: frozendict[str, jax.Array | np.ndarray]
    binding_id: str = dataclasses.field(init=False)
    budget: dataclasses.InitVar[CoordinateEnclosureBudget | None] = None

    def __post_init__(self, budget: CoordinateEnclosureBudget | None = None) -> None:
        names = (
            "source_result_id",
            "source_mesh_id",
            "source_topology_id",
            "source_geometry_id",
            "source_geometry_layout_id",
            "coordinate_contract_id",
            "numeric_version",
        )
        for name in names:
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(
                    f"MeshingPolyhedralOwnerBinding.{name} must retain a trimmed scientific identity; got invalid identity"
                )
        if (
            type(self.preparation_ids) is not frozendict
            or set(self.preparation_ids)
            != {"vem", "finite_volume", "hybrid", "source_star", "source_star_space"}
            or any(
                type(value) is not str or not value
                for value in self.preparation_ids.values()
            )
            or type(self.field_space_ids) is not frozendict
            or set(self.field_space_ids) != {"vem", "source_star"}
            or type(self.field_names) is not frozendict
            or set(self.field_names) != set(self.field_space_ids)
            or any(
                type(name) is not str or not name for name in self.field_names.values()
            )
            or type(self.coefficient_banks) is not frozendict
            or set(self.coefficient_banks) != set(self.field_space_ids)
        ):
            raise ValueError(
                "MeshingPolyhedralOwnerBinding.preparation_ids must retain actual owners and only genuine VEM/source-star coefficient spaces; got incomplete or foreign owners"
            )
        for owner, bank in self.coefficient_banks.items():
            if type(bank) is not MeshingCoefficientIdentityBank:
                raise TypeError(
                    "MeshingPolyhedralOwnerBinding.coefficient_banks must retain canonical coefficient banks; got another owner"
                )
            bank.__post_init__()
            if bank.field_space_id != self.field_space_ids[owner]:
                raise ValueError(
                    "MeshingPolyhedralOwnerBinding.coefficient_banks must match their real field spaces; got a foreign bank"
                )
        if (
            type(self.index_banks) is not frozendict
            or not self.index_banks
            or any(
                type(name) is not str
                or not name
                or not isinstance(array, (jax.Array, np.ndarray))
                or array.dtype.kind not in "biuf"
                for name, array in self.index_banks.items()
            )
        ):
            raise TypeError(
                "MeshingPolyhedralOwnerBinding.index_banks must retain actual named identity, validity, and source-coordinate arrays; got invalid banks"
            )
        object.__setattr__(
            self,
            "binding_id",
            stored_record_content_id(self, excluded="binding_id", budget=budget),
        )

    @classmethod
    def from_stored_fields(
        cls: type[Self],
        value: MeshingPolyhedralOwnerBinding,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> Self:
        rebuilt = cls(
            **{
                member.name: getattr(value, member.name)
                for member in dataclasses.fields(value)
                if member.init
            },
            budget=budget,
        )
        if rebuilt.binding_id != value.binding_id:
            raise ValueError(
                "MeshingPolyhedralOwnerBinding.binding_id must retain complete plan/index content; got changed banks"
            )
        return rebuilt


type MeshingPolyhedralOwner = Literal["vem", "hybrid", "source_star"]
type MeshingPolyhedralAlgebraicRole = Literal["primal", "dual", "physical"]
type MeshingPolyhedralIndexBank = Literal[
    "coefficients", "cell", "active-trace", "cell-trace", "local-facet"
]


@dataclass(frozen=True, kw_only=True)
class MeshingPolyhedralFieldIndexBinding:
    """Select an existing owner's real coefficient/entity/occurrence bank."""

    owner_binding: MeshingPolyhedralOwnerBinding
    owner: MeshingPolyhedralOwner
    bank_name: MeshingPolyhedralIndexBank
    index_axes: tuple[int, ...]

    def __post_init__(self) -> None:
        parse(
            self.owner, MeshingPolyhedralOwner, "MeshingPolyhedralFieldIndexBinding.owner"
        )
        parse(
            self.bank_name,
            MeshingPolyhedralIndexBank,
            "MeshingPolyhedralFieldIndexBinding.bank_name",
        )
        if type(self.owner_binding) is not MeshingPolyhedralOwnerBinding:
            raise TypeError(
                "MeshingPolyhedralFieldIndexBinding.owner_binding must retain the actual scientific owner; got another binding"
            )
        if (
            self.bank_name == "coefficients"
            and self.owner == "hybrid"
            or self.owner == "vem"
            and self.bank_name != "coefficients"
            or self.owner == "source_star"
            and self.bank_name not in ("coefficients", "cell")
        ):
            raise ValueError(
                "MeshingPolyhedralFieldIndexBinding.bank_name must belong to its actual owner; got an invented coefficient/entity layout"
            )
        shape = self.index_shape
        if (
            type(self.index_axes) is not tuple
            or len(self.index_axes) != len(shape)
            or any(type(axis) is not int or axis < 0 for axis in self.index_axes)
            or tuple(sorted(set(self.index_axes))) != self.index_axes
        ):
            raise ValueError(
                "MeshingPolyhedralFieldIndexBinding.index_axes must explicitly locate its ordered owner axes; got invalid axes"
            )

    @property
    def field_space_id(self) -> str | None:
        return (
            self.owner_binding.field_space_ids[self.owner]
            if self.bank_name == "coefficients"
            else None
        )

    @property
    def index_shape(self) -> tuple[int, ...]:
        banks = self.owner_binding.index_banks
        match self.bank_name:
            case "coefficients":
                return (self.owner_binding.coefficient_banks[self.owner].row_count,)
            case "cell":
                # Source-star cell diagnostics use ORIGINAL polyhedral cells.
                return tuple(banks["cell_ids"].shape)
            case "active-trace":
                return tuple(banks["active_face_ids"].shape)
            case "cell-trace":
                # The real direct sum is cells, then selected active faces;
                # both ordered ID banks and their native maps are retained.
                return (banks["cell_ids"].size + banks["active_face_ids"].size,)
            case "local-facet":
                return tuple(banks["cell_faces"].shape)
            case invalid:
                assert_never(invalid)

    @classmethod
    def from_stored_fields(
        cls: type[Self], value: MeshingPolyhedralFieldIndexBinding, /
    ) -> Self:
        return cls(
            owner_binding=value.owner_binding,
            owner=value.owner,
            bank_name=value.bank_name,
            index_axes=value.index_axes,
        )


def _require_field_source_validation(value: Any | None, /) -> None:
    """Refuse caller callbacks; only the concrete common replay owner is valid."""
    if value is None:
        return
    from ._meshing_sources import MeshingSourceValidation

    if type(value) is not MeshingSourceValidation:
        raise TypeError(
            "Field source_validation must be an exact MeshingSourceValidation; got a foreign proof"
        )


@checked
def prepare_meshing_field_index_binding(
    carrier: CellMeshingResult,
    bank_name: MeshingFieldIndexBank,
    /,
    *,
    field_space: DiscreteFieldSpace | None = None,
    source_validation: Any | None = None,
) -> MeshingFieldIndexBinding:
    """Bind rows to the source bank by scientific row identity."""
    _require_field_source_validation(source_validation)
    bank_name = parse(
        bank_name, MeshingFieldIndexBank, "prepare_meshing_field_index_binding.bank_name"
    )
    mesh = carrier.mesh
    evidence = carrier.collective_evidence
    evidence_id = None if evidence is None else evidence.evidence_id
    clocks = (
        None if evidence is None else dict(evidence.logical_arrays).get("epoch/clocks")
    )
    # The restore owner earns this exact digest collectively from the current
    # immutable bank; local validation must authenticate that input, not an ID.
    epoch_id = (
        type(source_validation).field_epoch_clock_digest(source_validation, carrier)
        if source_validation is not None
        else (
            None
            if clocks is None
            else logical_array_value_collection_digest({"epoch/clocks": clocks})
        )
    )
    scope = None
    match bank_name:
        case "field/global_coefficient_ids":
            if field_space is None or not isinstance(
                field_space.vector_space, ArraySpace
            ):
                raise TypeError(
                    "prepare_meshing_field_index_binding.field_space must declare an ArraySpace; got missing or incompatible field space"
                )
            values = np.arange(field_space.vector_space.shape[0], dtype=np.int64)
            values.flags.writeable = False
        case "mesh/vertex_ids" | "mesh/cell_ids":
            degree = 0 if bank_name == "mesh/vertex_ids" else mesh.topological_dimension
            entity_set = mesh.entity_set(degree)
            values = mesh.vertex_global_ids if degree == 0 else entity_set.entity_ids
            scope = MeshingScope(
                mesh.mesh_id,
                mesh.numeric_version,
                MeshingEntityKind.MESH,
                degree,
                entity_set.entity_set_id,
                entity_set.entity_ids,
            )
            if field_space is not None:
                layout = field_space.layout
                if (
                    not isinstance(layout, EntityDofLayout)
                    or layout.entity_set_id != entity_set.entity_set_id
                    or layout.dofs_per_entity != 1
                    or layout.entity_count != values.shape[0]
                ):
                    raise ValueError(
                        "prepare_meshing_field_index_binding.field_space.dof_layout must bind one DOF per entity; got incompatible layout"
                    )
        case "epoch/vertex_ids" | "epoch/cell_ids":
            if evidence is None:
                raise ValueError(
                    "prepare_meshing_field_index_binding.carrier.collective_evidence must bind raw history; got missing evidence"
                )
            bank = dict(evidence.logical_arrays).get(bank_name)
            if bank is None or clocks is None:
                raise ValueError(
                    "prepare_meshing_field_index_binding.carrier.collective_evidence must retain raw index and clock banks; got missing bank"
                )
            if field_space is not None:
                raise ValueError(
                    "prepare_meshing_field_index_binding.field_space must be absent for raw epochs; got coefficient field space"
                )
            values = bank
        case _:
            raise ValueError(
                "prepare_meshing_field_index_binding.bank_name must select a supported bank; got unsupported selector"
            )
    return MeshingFieldIndexBinding(
        bank_name=bank_name,
        source_result_id=carrier.result_id,
        source_topology_id=mesh.topology_id,
        source_geometry_id=carrier.geometry.geometry_layout_id,
        source_evidence_id=evidence_id,
        source_epoch_id=epoch_id,
        index_values=values,
        index_axes=(0, 1) if bank_name.startswith("epoch/") else (0,),
        scope=scope,
        field_space_id=None if field_space is None else field_space.field_space_id,
        coefficient_bank=None
        if field_space is None
        else MeshingCoefficientIdentityBank.from_field_space(field_space),
    )


def _same_index_bank(
    first: jax.Array | np.ndarray,
    second: jax.Array | np.ndarray,
    /,
    *,
    source_validation: Any | None = None,
) -> bool:
    _require_field_source_validation(source_validation)
    if first.shape != second.shape or first.dtype != second.dtype:
        return False
    if source_validation is not None:
        return type(source_validation).array_equal(source_validation, first, second)
    return first is second or (
        logical_array_value_collection_digest({"index-bank": first})
        == logical_array_value_collection_digest({"index-bank": second})
    )


@checked
def validate_meshing_field_index_binding(
    role: MeshingFieldStateRole,
    carrier: CellMeshingResult,
    /,
    *,
    field_space: DiscreteFieldSpace | None = None,
    source_validation: Any | None = None,
) -> None:
    """Prove bank identity against real entity/DOF/raw-source arrays."""
    _require_field_source_validation(source_validation)
    role.__post_init__()
    if role.role in ("state-epoch", "rng-key"):
        return
    binding = role.index_binding
    if type(binding) is MeshingPolyhedralFieldIndexBinding:
        expected_owner, _ = prepare_meshing_polyhedral_owner_binding(
            carrier,
            vem_field_name=binding.owner_binding.field_names["vem"],
            source_validation=source_validation,
        )
        equal = content_equal if source_validation is None else source_validation.equivalent
        if not equal(
            binding.owner_binding, expected_owner, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
        ):
            raise ValueError(
                "MeshingPolyhedralFieldIndexBinding.owner_binding must match the actual prepared carrier plans and index banks; got changed ownership"
            )
        if field_space is not None:
            if binding.bank_name != "coefficients":
                raise ValueError(
                    "MeshingPolyhedralFieldIndexBinding.field_space must belong to a genuine coefficient bank; got a native hybrid entity/occurrence bank"
                )
            binding.owner_binding.coefficient_banks[binding.owner].require_field_space(
                field_space
            )
        return
    if type(binding) is not MeshingFieldIndexBinding:
        raise ValueError(
            "MeshingFieldStateRole.index_binding must declare row identities; got missing binding"
        )
    expected = prepare_meshing_field_index_binding(
        carrier,
        binding.bank_name,
        field_space=field_space,
        source_validation=source_validation,
    )
    identity_fields = (
        "bank_name",
        "source_result_id",
        "source_topology_id",
        "source_geometry_id",
        "source_evidence_id",
        "source_epoch_id",
        "index_axes",
        "field_space_id",
    )
    if any(getattr(binding, name) != getattr(expected, name) for name in identity_fields):
        raise ValueError(
            "MeshingFieldStateRole.index_binding must match source, epoch, geometry and field space; got incompatible identity"
        )
    if not _same_index_bank(
        binding.index_values, expected.index_values, source_validation=source_validation
    ):
        raise ValueError(
            "MeshingFieldStateRole.index_binding.index_values must match source row identity/order; got changed rows"
        )
    if binding.coefficient_bank != expected.coefficient_bank:
        raise ValueError(
            "MeshingFieldStateRole.index_binding.coefficient_bank must retain ordered field-space identities; got changed bank"
        )
    if expected.scope is None:
        if binding.scope is not None:
            raise ValueError(
                "MeshingFieldStateRole.index_binding.scope must be absent for raw or DOF ordinals; got compact scope"
            )
    elif binding.scope is None or (
        binding.scope.source_id != expected.scope.source_id
        or binding.scope.source_revision != expected.scope.source_revision
        or binding.scope.entity_kind != expected.scope.entity_kind
        or binding.scope.entity_dimension != expected.scope.entity_dimension
        or binding.scope.entity_set_id != expected.scope.entity_set_id
        or binding.scope.scope_id != expected.scope.scope_id
        or not _same_index_bank(
            binding.scope.entity_ids,
            expected.scope.entity_ids,
            source_validation=source_validation,
        )
    ):
        raise ValueError(
            "MeshingFieldStateRole.index_binding.scope must match the source mesh scope; got changed entity membership"
        )


@checked
def project_meshing_field_history(
    source_role: MeshingFieldStateRole,
    source_values: jax.Array | np.ndarray,
    raw_role: MeshingFieldStateRole,
    raw_baseline: jax.Array | np.ndarray,
    carrier: CellMeshingResult,
    /,
    *,
    field_space: DiscreteFieldSpace | None = None,
    index_source: CellMeshingResult | None = None,
    transition: CellMeshTransition | None = None,
) -> jax.Array | np.ndarray:
    """Join compact transferred values to raw rows by scientific IDs.

    Only active raw rows are updated. Inactive and retired rows retain the
    supplied indexed baseline. An authenticated accepted ``transition`` routes
    coarsened predecessor children to current parent rows; distinct children may
    share a parent only through that witness. Missing active IDs refuse
    collectively. Local shards never gather the global bank.
    """

    source_role.__post_init__()
    raw_role.__post_init__()
    compact, raw = source_role.index_binding, raw_role.index_binding
    if (
        type(compact) is not MeshingFieldIndexBinding
        or type(raw) is not MeshingFieldIndexBinding
        or (
            (compact.bank_name, raw.bank_name)
            not in (
                ("mesh/vertex_ids", "epoch/vertex_ids"),
                ("mesh/cell_ids", "epoch/cell_ids"),
            )
        )
    ):
        raise ValueError(
            "project_meshing_field_history.index_binding must pair compact entities with allocated epochs; got incompatible banks"
        )
    if raw_role.role not in ("stage-history", "material-history", "rng-counter"):
        raise ValueError(
            "project_meshing_field_history.raw_role.role must be a history role; got coefficient or unsupported role"
        )
    _require_random_stream_pair(source_role, raw_role, "project_meshing_field_history")
    if (
        source_role.field_name != raw_role.field_name
        or source_role.state_epoch != raw_role.state_epoch
        or source_role.value_units != raw_role.value_units
    ):
        raise ValueError(
            "project_meshing_field_history.raw_role must preserve field, units and state epoch; got changed semantics"
        )
    validate_meshing_field_index_binding(source_role, carrier, field_space=field_space)
    raw_source = carrier if index_source is None else index_source
    validate_meshing_field_index_binding(raw_role, raw_source)
    if (
        source_values.shape != source_role.shape
        or source_values.dtype != source_role.dtype
    ):
        raise ValueError(
            "project_meshing_field_history.source_values must match the declared shape and dtype; got incompatible schema"
        )
    if (
        raw_baseline.shape != raw_role.shape
        or raw_baseline.dtype != raw_role.dtype
        or source_values.dtype != raw_role.dtype
    ):
        raise ValueError(
            "project_meshing_field_history.raw_baseline must match the raw role shape and dtype; got incompatible schema"
        )
    if source_role.shape[1:] != raw_role.shape[2:]:
        raise ValueError(
            "project_meshing_field_history.raw_role.shape must preserve compact component axes; got incompatible components"
        )
    if (
        isinstance(compact.index_values, jax.Array)
        and not compact.index_values.is_fully_addressable
    ):
        raise ValueError(
            "project_meshing_field_history.source_role.index_binding.index_values must be addressable; got nonaddressable identity bank"
        )
    if isinstance(source_values, jax.Array) and not source_values.is_fully_addressable:
        raise ValueError(
            "project_meshing_field_history.source_values must be addressable; got nonaddressable values"
        )
    evidence = raw_source.collective_evidence
    if evidence is None:
        raise ValueError(
            "project_meshing_field_history.index_source.collective_evidence must retain native raw banks; got missing evidence"
        )
    active_name = (
        "epoch/vertex_active"
        if raw.bank_name == "epoch/vertex_ids"
        else "epoch/cell_active"
    )
    active = dict(evidence.logical_arrays).get(active_name)
    if (
        active is None
        or active.dtype != np.dtype("bool")
        or active.shape != raw.index_values.shape
    ):
        raise ValueError(
            "project_meshing_field_history.active_mask must match the raw bank; got missing or incompatible mask"
        )
    if isinstance(raw_baseline, jax.Array) and not raw_baseline.is_fully_addressable:
        return _owner_local_projection(
            carrier, compact, raw, source_values, raw_baseline, active, transition
        )
    ids = np.asarray(compact.index_values)
    if np.any(ids < 0) or np.unique(ids).size != ids.size:
        raise ValueError(
            "project_meshing_field_history.compact_ids must be distinct nonnegative identities; got invalid or duplicate IDs"
        )
    ids, source_rows = _projection_identity_rows(carrier, compact, transition)

    def local(
        ids_block: jax.Array | np.ndarray,
        active_block: jax.Array | np.ndarray,
        baseline: jax.Array | np.ndarray,
    ) -> jax.Array:
        queries = np.asarray(ids_block)
        live = np.asarray(active_block)
        if np.any(live & (queries < 0)):
            raise ValueError(
                "project_meshing_field_history.raw_ids must be nonnegative for active rows; got missing identity"
            )
        rows = key_rows(ids[:, None], queries.reshape(-1, 1)).reshape(queries.shape)
        if compact.bank_name == "mesh/vertex_ids":
            live = live & (rows >= 0)
        if np.any(live & (rows < 0)):
            raise ValueError(
                "project_meshing_field_history.compact_ids must contain active raw IDs; got missing transferred identity"
            )
        if ids.size == 0:
            return jnp.asarray(baseline)
        selected = live & (rows >= 0)
        relation = RowRelation(
            np.maximum(rows, 0)[..., None],
            source_size=ids.size,
            valid=selected[..., None],
        )
        values = jnp.squeeze(
            gather_routes(relation, jnp.asarray(source_values)[source_rows]), axis=rows.ndim
        )
        mask = selected.reshape((*selected.shape, *(1 for _ in raw_role.shape[2:])))
        return jnp.where(mask, values, jnp.asarray(baseline))

    if isinstance(raw_baseline, np.ndarray):
        if (
            isinstance(raw.index_values, jax.Array)
            and not raw.index_values.is_fully_addressable
        ):
            raise ValueError(
                "project_meshing_field_history.raw_baseline must match raw-bank addressability; got host baseline with nonaddressable identities"
            )
        if not active.is_fully_addressable:
            raise ValueError(
                "project_meshing_field_history.active_mask must be addressable with a host baseline; got nonaddressable mask"
            )
        return np.asarray(local(raw.index_values, active, raw_baseline))
    if not isinstance(raw.index_values, jax.Array):
        raise TypeError(
            "project_meshing_field_history.raw_ids must be a JAX array for distributed history; got host identities"
        )
    id_shards = {shard.device: shard for shard in raw.index_values.addressable_shards}
    active_shards = {shard.device: shard for shard in active.addressable_shards}
    results = []
    for shard in raw_baseline.addressable_shards:
        index_shard, active_shard = (
            id_shards.get(shard.device),
            active_shards.get(shard.device),
        )
        if (
            index_shard is None
            or active_shard is None
            or shard.index[:2] != index_shard.index
            or active_shard.index != index_shard.index
        ):
            raise ValueError(
                "project_meshing_field_history.raw_baseline must share owner-local axes with raw IDs and the active mask; got incompatible shards"
            )
        results.append(
            jax.device_put(
                local(index_shard.data, active_shard.data, shard.data), shard.device
            )
        )
    return jax.make_array_from_single_device_arrays(
        raw_baseline.shape, raw_baseline.sharding, results
    )


def _require_random_stream_pair(
    first: MeshingFieldStateRole, second: MeshingFieldStateRole, owner: str, /
) -> None:
    """Join RNG counters only to counters of one stream, never to values."""
    if (first.role == "rng-counter") != (second.role == "rng-counter") or (
        first.random_stream != second.random_stream
    ):
        raise ValueError(
            f"{owner}.role must pair rng-counter state with the same random stream; got mixed or foreign stream"
        )


def _require_history_repack(
    receipt: SimplexRestartRepack,
    proof: SimplexRestartRepackProof | None,
) -> None:
    if proof is None:
        validate_simplex_restart_repack(receipt, receipt.requested_target_owners)
    else:
        SimplexRestartRepackProof.require_receipt(proof, receipt)


@checked
def rebind_meshing_field_history(
    role: MeshingFieldStateRole,
    source_values: jax.Array | np.ndarray,
    target: CellMeshingResult,
    receipt: SimplexRestartRepack,
    /,
    *,
    history_index: int,
    proof: SimplexRestartRepackProof | None = None,
) -> tuple[MeshingFieldStateRole, jax.Array]:
    """Rebind raw history through a native repack.

    ``history_index`` selects an authored cell/vertex payload slot. The repack
    validates saved locations, owner assignments, retired records and values.
    A supplied proof authenticates the same immutable source, layout, placement,
    and every numerical binding without repeating all-rank validation.
    """

    role.__post_init__()
    binding = role.index_binding
    if (
        role.role not in ("material-history", "stage-history", "rng-counter")
        or type(binding) is not MeshingFieldIndexBinding
    ):
        raise ValueError(
            "rebind_meshing_field_history.role must declare indexed native history; got unsupported role or missing binding"
        )
    if binding.bank_name not in ("epoch/vertex_ids", "epoch/cell_ids"):
        raise ValueError(
            "rebind_meshing_field_history.role.index_binding.bank_name must select an allocated epoch bank; got compact or unsupported bank"
        )
    if type(history_index) is not int or history_index < 0:
        raise ValueError(
            "rebind_meshing_field_history.history_index must be a nonnegative integer; got invalid payload index"
        )
    source = receipt.source_result
    if source is None:
        raise ValueError(
            "rebind_meshing_field_history.receipt.source_result must retain the accepted source; got missing source"
        )
    validate_meshing_field_index_binding(role, source)
    _require_history_repack(receipt, proof)
    if binding.bank_name == "epoch/vertex_ids":
        saved, transferred = receipt.saved_vertex_history, receipt.vertex_history
        old_ids, new_ids = (
            receipt.saved_states.mesh.vertex_ids,
            receipt.states.mesh.vertex_ids,
        )
    else:
        saved, transferred = receipt.saved_cell_history, receipt.cell_history
        old_ids, new_ids = (
            receipt.saved_states.mesh.cell_ids,
            receipt.states.mesh.cell_ids,
        )
    if history_index >= len(saved) or history_index >= len(transferred):
        raise ValueError(
            "rebind_meshing_field_history.history_index must select retained and transferred payload slots; got absent slot"
        )
    if not _same_index_bank(binding.index_values, old_ids) or not _same_index_bank(
        source_values, saved[history_index]
    ):
        raise ValueError(
            "rebind_meshing_field_history.source_values must match the saved payload and index_binding; got changed rows"
        )
    values = transferred[history_index]
    if (
        source_values.shape != role.shape
        or source_values.dtype != role.dtype
        or values.dtype != role.dtype
    ):
        raise ValueError(
            "rebind_meshing_field_history.transferred_values must preserve dtype and component schema; got incompatible payload"
        )
    target_binding = prepare_meshing_field_index_binding(target, binding.bank_name)
    if not _same_index_bank(target_binding.index_values, new_ids):
        raise ValueError(
            "rebind_meshing_field_history.target.index_binding must match repacked IDs; got foreign target bank"
        )
    rebound = replace(role, shape=values.shape, index_binding=target_binding)
    validate_meshing_field_index_binding(rebound, target)
    return rebound, values


@checked
def recover_meshing_field_history(
    raw_role: MeshingFieldStateRole,
    raw_values: jax.Array | np.ndarray,
    compact_role: MeshingFieldStateRole,
    target: CellMeshingResult,
    receipt: SimplexRestartRepack,
    /,
    *,
    history_index: int,
    field_space: DiscreteFieldSpace | None = None,
    proof: SimplexRestartRepackProof | None = None,
) -> jax.Array | np.ndarray:
    """Recover compact field rows from a repack's owner-assigned raw IDs.

    Owner assignments distinguish ghost duplicates and inactive capacity.
    Cold consumers inspect only addressable native records.
    A supplied proof authenticates retained numerical bindings locally; without
    it, repack validation runs before field recovery.
    """

    raw_role.__post_init__()
    compact_role.__post_init__()
    raw, compact = raw_role.index_binding, compact_role.index_binding
    if (
        type(raw) is not MeshingFieldIndexBinding
        or type(compact) is not MeshingFieldIndexBinding
        or (raw.bank_name, compact.bank_name)
        not in (
            ("epoch/vertex_ids", "mesh/vertex_ids"),
            ("epoch/cell_ids", "mesh/cell_ids"),
        )
    ):
        raise ValueError(
            "recover_meshing_field_history.index_binding must pair allocated epochs with compact entities; got incompatible banks"
        )
    if (
        raw_role.field_name != compact_role.field_name
        or raw_role.state_epoch != compact_role.state_epoch
        or raw_role.value_units != compact_role.value_units
    ):
        raise ValueError(
            "recover_meshing_field_history.compact_role must preserve field, units and epoch; got changed semantics"
        )
    _require_random_stream_pair(raw_role, compact_role, "recover_meshing_field_history")
    if type(history_index) is not int or history_index < 0:
        raise ValueError(
            "recover_meshing_field_history.history_index must be a nonnegative integer; got invalid payload slot"
        )
    _require_history_repack(receipt, proof)
    validate_meshing_field_index_binding(raw_role, target)
    validate_meshing_field_index_binding(compact_role, target, field_space=field_space)
    if raw.bank_name == "epoch/vertex_ids":
        payloads, owners, receipt_ids = (
            receipt.vertex_history,
            receipt.vertex_owners,
            receipt.states.mesh.vertex_ids,
        )
    else:
        payloads, owners, receipt_ids = (
            receipt.cell_history,
            receipt.cell_owners,
            receipt.states.mesh.cell_ids,
        )
    if history_index >= len(payloads) or not _same_index_bank(
        raw_values, payloads[history_index]
    ):
        raise ValueError(
            "recover_meshing_field_history.raw_values must match the retained payload slot; got differing values"
        )
    if (
        not _same_index_bank(raw.index_values, receipt_ids)
        or owners.shape != receipt_ids.shape
    ):
        raise ValueError(
            "recover_meshing_field_history.raw_role.index_binding must match repacked IDs and epoch; got incompatible binding"
        )
    if (
        raw_values.shape != raw_role.shape
        or raw_values.dtype != raw_role.dtype
        or raw_role.dtype != compact_role.dtype
        or raw_role.shape[2:] != compact_role.shape[1:]
    ):
        raise ValueError(
            "recover_meshing_field_history.raw_values must preserve the compact role dtype and component schema; got incompatible payload"
        )
    arrays = (receipt_ids, owners, raw_values)
    if any(
        isinstance(value, jax.Array) and not value.is_fully_addressable
        for value in arrays
    ):
        # Bank validation above binds the caller's values to this sharded payload.
        result = _owner_local_recovery(
            receipt,
            receipt_ids,
            owners,
            payloads[history_index],
            np.asarray(compact.index_values),
        )
        if result.shape != compact_role.shape:
            raise ValueError(
                "recover_meshing_field_history.result.shape must match compact_role.shape; got incompatible rows"
            )
        return result
    if (
        isinstance(compact.index_values, jax.Array)
        and not compact.index_values.is_fully_addressable
    ):
        raise ValueError(
            "recover_meshing_field_history.compact_ids must be addressable by this process; got nonaddressable entity bank"
        )
    identifiers, authority = np.asarray(receipt_ids), np.asarray(owners)
    selected = (identifiers >= 0) & (
        authority == np.arange(identifiers.shape[0], dtype=np.int32)[:, None]
    )
    owner_ids = identifiers[selected]
    if np.unique(owner_ids).size != owner_ids.size:
        raise ValueError(
            "recover_meshing_field_history.owner_ids must be unique; got duplicate scientific IDs"
        )
    queries = np.asarray(compact.index_values)
    rows = key_rows(owner_ids[:, None], queries[:, None])
    if np.any(rows < 0):
        raise ValueError(
            "recover_meshing_field_history.owner_ids must contain compact IDs; got missing history identity"
        )
    if rows.size == 0:
        result = jnp.empty(compact_role.shape, dtype=compact_role.dtype)
    else:
        source_rows = np.flatnonzero(selected.reshape(-1))[rows]
        result = _history_payload_rows(jnp.asarray(raw_values), source_rows)
    return np.asarray(result) if isinstance(raw_values, np.ndarray) else result


@jax.jit
def _history_payload_rows(values: jax.Array, rows: jax.Array | np.ndarray) -> jax.Array:
    """Transport immutable payload bytes before crossing the partition axis."""
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        return jax.lax.complex(
            _history_payload_component_rows(values.real, rows),
            _history_payload_component_rows(values.imag, rows),
        )
    return _history_payload_component_rows(values, rows)


def _history_payload_component_rows(
    values: jax.Array, rows: jax.Array | np.ndarray
) -> jax.Array:
    payload_shape = values.shape[2:]
    packed = values.reshape((*values.shape[:2], prod(payload_shape))).view(jnp.uint8)
    flattened = packed.reshape((prod(values.shape[:2]), packed.shape[-1]))
    selected = flattened[rows]
    return selected.view(values.dtype).reshape((rows.shape[0], *payload_shape))


def _projection_identity_rows(
    carrier: CellMeshingResult,
    compact: MeshingFieldIndexBinding,
    transition: CellMeshTransition | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bind predecessor aliases to accepted target rows, never allocation order."""
    ids = np.asarray(compact.index_values)
    rows = np.arange(ids.size, dtype=np.int64)
    if transition is None:
        return ids, rows
    lineage = transition.lineage
    for item in lineage.entities:
        authenticated = EntityLineage(
            item.dimension, item.source_entity_set_id, item.target_entity_set_id,
            item.source_global_ids, item.target_global_ids, item.relation_kinds,
            created_target_ids=item.created_target_ids,
            deleted_source_ids=item.deleted_source_ids,
        )
        if authenticated.lineage_id != item.lineage_id:
            raise ValueError(
                "project_meshing_field_history.transition must authenticate entity lineage; got changed provenance"
            )
    if MeshLineage(
        lineage.source_topology_id, lineage.target_topology_id, lineage.entities
    ).lineage_id != lineage.lineage_id:
        raise ValueError(
            "project_meshing_field_history.transition must authenticate mesh lineage; got changed provenance"
        )
    rebuilt = CellMeshTransition.from_stored_fields(transition, budget=None)
    if rebuilt.target.result_id != carrier.result_id:
        raise ValueError(
            "project_meshing_field_history.transition must authenticate the current target; got foreign or changed lineage"
        )
    if compact.bank_name == "mesh/vertex_ids":
        vertices = rebuilt.lineage.entity_lineage(0)
        if vertices.target_entity_set_id != carrier.mesh.entity_set(0).entity_set_id:
            raise ValueError(
                "project_meshing_field_history.transition must bind the target vertex entity set; got foreign dimension or provenance"
            )
        merged = np.isin(
            np.asarray(vertices.relation_kinds),
            (int(EntityLineageKind.MERGED_INTO), int(EntityLineageKind.COLLAPSED_INTO)),
        )
        children = np.asarray(vertices.source_global_ids)[merged]
        parents = np.asarray(vertices.target_global_ids)[merged]
    elif rebuilt.coarsened_cell_ids is not None:
        children = np.asarray(rebuilt.coarsened_cell_ids)
        parents = np.asarray(rebuilt.coarsened_into_ids)
    else:
        return ids, rows
    parent_rows = key_rows(ids[:, None], parents[:, None])
    if np.any(parent_rows < 0):
        raise ValueError(
            "project_meshing_field_history.transition must bind current parent identities; got missing parent"
        )
    # Preserved fine/coarse pairs already have their exact current identity.
    changed = children != parents
    aliases = children[changed]
    if np.any(np.isin(aliases, ids)) or np.unique(aliases).size != aliases.size:
        raise ValueError(
            "project_meshing_field_history.transition must give disjoint predecessor identities; got duplicate lineage"
        )
    return np.concatenate((ids, aliases)), np.concatenate((rows, parent_rows[changed]))


def _owner_local_projection(
    carrier: CellMeshingResult,
    compact: MeshingFieldIndexBinding,
    raw: MeshingFieldIndexBinding,
    values: jax.Array | np.ndarray,
    baseline: jax.Array,
    active: jax.Array,
    transition: CellMeshTransition | None = None,
) -> jax.Array:
    """Route current owned compact values to every active allocated identity.

    Raw allocation placement is not the compact mesh closure placement. In
    particular a restart can retain active halo records outside that closure.
    Query those exact IDs against current authoritative compact rows; historical
    inactive rows remain byte-for-byte in their original allocated bank.
    """
    with agreed_local_step(stage="meshing-field-projection-admission"):
        placement = baseline.sharding
        storage = carrier.mesh.storage
        if (
            not isinstance(placement, NamedSharding)
            or len(placement.mesh.axis_names) != 1
            or storage is None
            or not isinstance(raw.index_values, jax.Array)
            or len(baseline.addressable_shards) != 1
        ):
            raise ValueError(
                "project_meshing_field_history.placement must retain one owner-local raw part and compact storage"
            )
        axis = placement.mesh.axis_names[0]
        devices = tuple(placement.mesh.devices.flat)
        count = len(devices)
        parts = AdaptiveSimplexParts(
            devices,
            axis_name=axis,
            neighbor_pairs=tuple(sorted({
                (left, right)
                for rank in range(count)
                for left, right in (
                    (rank, (rank + 1) % count), ((rank + 1) % count, rank)
                )
                if left != right
            })),
        )
        degree = 0 if compact.bank_name == "mesh/vertex_ids" else carrier.mesh.topological_dimension
        ids = np.asarray(compact.index_values)
        owned = np.asarray(storage.entity_owned[degree], dtype=np.bool_)
        if (
            ids.ndim != 1
            or owned.shape != ids.shape
            or np.any(ids < 0)
            or np.unique(ids).size != ids.size
            or not np.array_equal(ids, np.asarray(storage.entity_global_ids[degree]))
        ):
            raise ValueError(
                "project_meshing_field_history.compact_ids must match distinct current storage identities"
            )
        identities, source_rows = _projection_identity_rows(carrier, compact, transition)
        selected = owned[source_rows]
        source_ids = identities[selected]
        source_rows = source_rows[selected]
        rank = storage.partition_index
        if devices[rank] != baseline.addressable_shards[0].device:
            raise ValueError(
                "project_meshing_field_history.placement must match current compact authority"
            )
    capacity = int(
        np.max(multihost_utils.process_allgather(np.int64(max(identities.size, 1))))
    )
    with agreed_local_step(stage="meshing-field-projection-packets"):
        packet_ids = np.full((1, capacity), -1, dtype=np.int64)
        packet_ids[0, :source_ids.size] = source_ids
        packet_values = jnp.zeros((1, capacity, *values.shape[1:]), dtype=values.dtype)
        packet_values = packet_values.at[0, :source_ids.size].set(jnp.asarray(values)[source_rows])
        sharding = NamedSharding(parts.mesh, PartitionSpec(axis))
        membership_ids = np.full((1, capacity), -1, dtype=np.int64)
        membership_ids[0, :identities.size] = identities
    identifiers = jax.make_array_from_process_local_data(
        sharding, packet_ids, (count, capacity)
    )
    authority = jax.make_array_from_process_local_data(
        sharding, np.full((1, capacity), rank, dtype=np.int32), (count, capacity)
    )
    payload = jax.make_array_from_process_local_data(
        sharding, packet_values, (count, capacity, *values.shape[1:])
    )
    queries = jnp.where(active, raw.index_values, -1)
    if degree == 0:
        # Allocation activity can retain orphaned vertices after coarsening.
        # Target membership and current ownership are separate predicates:
        # halo witnesses establish membership, never an invented value owner.
        membership = jax.make_array_from_process_local_data(
            sharding, membership_ids, (count, capacity)
        )
        _, target_copies = _ring_recovery(
            parts, membership, authority,
            jnp.zeros((count, capacity), dtype=jnp.bool_, device=sharding), queries,
        )
        active = active & (target_copies > 0)
    answer, found = _ring_recovery(parts, identifiers, authority, payload, queries)
    with agreed_local_step(stage="meshing-field-projection-identity-coverage"):
        for shard in found.addressable_shards:
            live = np.asarray(active.addressable_shards[0].data)
            if np.any(live & (np.asarray(shard.data) != 1)):
                raise ValueError(
                    "project_meshing_field_history.compact_ids must supply exactly one current owner for every active raw ID; got missing or duplicate transferred identity"
                )
    mask = active.reshape((*active.shape, *(1 for _ in baseline.shape[2:])))
    return jnp.where(mask, answer, baseline)


_SENTINEL = np.iinfo(np.int64).max


@eqx.filter_jit
def _ring_recovery(
    parts: AdaptiveSimplexParts,
    identifiers: jax.Array,
    owners: jax.Array,
    values: jax.Array,
    queries: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Circulate bounded query packets between raw-part owners."""

    axis = parts.axis_name
    ring = tuple(
        (rank, (rank + 1) % parts.part_count) for rank in range(parts.part_count)
    )
    spec = PartitionSpec(axis)

    def local(
        id_block: jax.Array,
        owner_block: jax.Array,
        value_block: jax.Array,
        query_block: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        rank = jax.lax.axis_index(axis)
        ids, authority, payload, query = (
            id_block[0],
            owner_block[0],
            value_block[0],
            query_block[0],
        )
        keys = jnp.where((ids >= 0) & (authority == rank), ids, _SENTINEL)
        order = jnp.argsort(keys, stable=True)
        keys = keys[order]
        answer = jnp.zeros((query.shape[0], *payload.shape[1:]), payload.dtype)
        found = jnp.zeros(query.shape, jnp.int32)

        def visit(
            _: int, carry: tuple[jax.Array, jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            query, answer, found = carry
            position = jnp.minimum(jnp.searchsorted(keys, query), keys.shape[0] - 1)
            match = (query >= 0) & (keys[position] == query)
            mask = match.reshape(match.shape + (1,) * (answer.ndim - 1))
            answer = jnp.where(mask, payload[order[position]], answer)
            found = found + match.astype(jnp.int32)
            return tuple(
                jax.lax.ppermute(value, axis, ring) for value in (query, answer, found)
            )

        _, answer, found = jax.lax.fori_loop(
            0, parts.part_count, visit, (query, answer, found)
        )
        return answer[None], found[None]

    return jax.shard_map(
        local,
        mesh=parts.mesh,
        in_specs=(spec, spec, spec, spec),
        out_specs=(spec, spec),
        check_vma=False,
    )(identifiers, owners, values, queries)


def _owner_local_recovery(
    receipt: SimplexRestartRepack,
    identifiers: jax.Array,
    owners: jax.Array,
    values: jax.Array,
    queries: np.ndarray,
    /,
) -> jax.Array:
    """Recover this process's compact rows from records on each owner.

    Fixed-capacity query packets circulate between neighbors. One host scalar
    exchange agrees on capacity. Every compact ID needs one owner record.
    """

    with agreed_local_step(stage="meshing-field-recovery-query-admission"):
        parts = type(receipt).execution_parts(receipt)
        shards = values.addressable_shards
        if len(shards) != 1 or queries.ndim != 1 or queries.dtype != np.int64:
            raise ValueError(
                "recover_meshing_field_history.queries must be int64 compact IDs for one addressable raw part; got incompatible workset"
            )
    capacity = int(
        np.max(multihost_utils.process_allgather(np.int64(max(queries.shape[0], 1))))
    )
    with agreed_local_step(stage="meshing-field-recovery-query-preparation"):
        packet = np.full((1, capacity), -1, dtype=np.int64)
        packet[0, : queries.shape[0]] = queries
        placement = NamedSharding(parts.mesh, PartitionSpec(parts.axis_name))
    distributed = jax.make_array_from_process_local_data(
        placement, packet, (parts.part_count, capacity)
    )
    answer, found = _ring_recovery(parts, identifiers, owners, values, distributed)
    with agreed_local_step(stage="meshing-field-recovery-owner-coverage"):
        local_found = np.asarray(found.addressable_shards[0].data)[0, : queries.shape[0]]
        if np.any(local_found != 1):
            raise ValueError(
                "recover_meshing_field_history.queries must have one owner record each; got absent or duplicate history identity"
            )
    return answer.addressable_shards[0].data[0, : queries.shape[0]]


type MeshingFieldStateKind = Literal[
    "coefficients",
    "material-history",
    "stage-history",
    "state-epoch",
    "boundary-values",
    "boundary-mask",
    "trace-values",
    "physical-values",
    "dual-values",
    "rng-key",
    "rng-counter",
]


_RANDOM_STREAM_IDENTITY_BANKS = frozenset(
    ("mesh/vertex_ids", "mesh/cell_ids", "epoch/vertex_ids", "epoch/cell_ids")
)
_RANDOM_COUNTER_DTYPES = (np.dtype("uint32"), np.dtype("uint64"))
_WORD_MASK = np.uint64(0xFFFFFFFF)


@cache
def _random_key_shape(implementation: str, /) -> tuple[int, ...]:
    """Key-data shape of one registered JAX PRNG implementation, never executed."""
    data = jax.eval_shape(
        lambda: jax.random.key_data(jax.random.key(0, impl=implementation))
    )
    if data.dtype != jnp.uint32:
        raise ValueError(
            f"MeshingRandomStream.implementation must store uint32 key data; got {data.dtype}"
        )
    return tuple(data.shape)


def _unsigned_words(values: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """High and low 32-bit words of nonnegative 64-bit integers, exactly."""
    words = values.astype(np.uint64)
    high = (words >> np.uint64(32)).astype(np.uint32)
    return high, (words & _WORD_MASK).astype(np.uint32)


@dataclass(frozen=True)
class MeshingRandomStream:
    """Persisted address of one entity-keyed random stream family.

    Entity ``g`` at counter ``n`` draws from ``derive_key(root,
    SampleAddress(namespace, operation, target=target, role=purpose), hi(g),
    lo(g), hi(n), lo(n))``: both 32-bit words of the stable int64 entity ID
    and of the unsigned counter, never a storage slot, rank, or truncated
    word. ``root`` wraps the uint32 key data of the accepted ``rng-key`` array
    ``key_entry_name`` with ``implementation``. A counter names the next
    unconsumed draw event of its entity.
    """

    key_entry_name: str
    implementation: str
    namespace: str
    operation: str
    target: tuple[str, ...]
    purpose: str

    @classmethod
    @checked
    def from_stored_fields(cls: type[Self], value: MeshingRandomStream, /) -> Self:
        """Reconstruct and validate the stored stream address."""
        return cls(
            **{
                member.name: getattr(value, member.name)
                for member in dataclasses.fields(value)
            }
        )

    def __post_init__(self) -> None:
        names = (
            self.key_entry_name,
            self.implementation,
            self.namespace,
            self.operation,
            self.purpose,
        )
        if type(self.target) is not tuple or any(
            type(name) is not str or not name or name != name.strip()
            for name in (*names, *self.target)
        ):
            raise ValueError(
                "MeshingRandomStream.key_entry_name must accompany a trimmed implementation and address; got invalid names"
            )
        _random_key_shape(self.implementation)

    @property
    def address(self) -> SampleAddress:
        """Canonical semantic address of this stream family."""
        return SampleAddress(
            self.namespace, self.operation, target=self.target, role=self.purpose
        )

    @property
    def key_shape(self) -> tuple[int, ...]:
        """Shape of the persisted uint32 root key data."""
        return _random_key_shape(self.implementation)

    def root_key(self, key_data: jax.Array | np.ndarray, /) -> jax.Array:
        """Wrap persisted root key data as a typed JAX key."""
        data = jnp.asarray(key_data)
        if data.dtype != jnp.uint32 or tuple(data.shape) != self.key_shape:
            raise ValueError(
                f"MeshingRandomStream.key_data must be uint32 with shape {self.key_shape}; got {(data.dtype, tuple(data.shape))}"
            )
        return jax.random.wrap_key_data(data, impl=self.implementation)

    def entity_keys(
        self,
        key_data: jax.Array | np.ndarray,
        identities: np.ndarray,
        counters: np.ndarray,
        /,
    ) -> jax.Array:
        """Typed keys of stable entity IDs at their current counters."""
        ids, counts = np.asarray(identities), np.asarray(counters)
        if (
            ids.dtype != np.int64
            or ids.ndim != 1
            or counts.dtype not in _RANDOM_COUNTER_DTYPES
            or counts.shape != ids.shape
        ):
            raise ValueError(
                "MeshingRandomStream.identities must be int64 entity IDs with one unsigned counter each; got incompatible rows"
            )
        if np.any(ids < 0):
            raise ValueError(
                "MeshingRandomStream.identities must be allocated stable IDs; got negative identity"
            )
        root, address = self.root_key(key_data), self.address
        id_high, id_low = _unsigned_words(ids)
        count_high, count_low = _unsigned_words(counts)
        return jax.vmap(lambda a, b, c, d: derive_key(root, address, a, b, c, d))(
            id_high, id_low, count_high, count_low
        )


@dataclass(frozen=True)
class MeshingFieldStateRole:
    """Bind an accepted-array role to a field, schema, and state epoch.

    ``rng-key`` roles hold the uint32 root key data of ``random_stream``;
    ``rng-counter`` roles hold one unsigned counter per stable entity ID of an
    entity bank. RNG state is unitless and never interpolated.
    """

    entry_name: str
    field_name: str
    role: MeshingFieldStateKind
    shape: tuple[int, ...]
    dtype: np.dtype
    state_epoch: int
    value_units: tuple[UnitDefinition, ...] | None = None
    index_binding: (
        MeshingFieldIndexBinding | MeshingPolyhedralFieldIndexBinding | None
    ) = None
    random_stream: MeshingRandomStream | None = None

    @classmethod
    @checked
    def from_stored_fields(cls: type[Self], value: MeshingFieldStateRole, /) -> Self:
        """Reconstruct and validate the stored state role."""
        return cls(
            **{
                member.name: getattr(value, member.name)
                for member in dataclasses.fields(value)
            }
        )

    def __post_init__(self) -> None:
        if any(
            type(name) is not str or not name or name != name.strip()
            for name in (self.entry_name, self.field_name)
        ):
            raise ValueError(
                "MeshingFieldStateRole.entry_name must be nonempty and paired with a trimmed field_name; got invalid name"
            )
        parse(self.role, MeshingFieldStateKind, "MeshingFieldStateRole.role")
        if type(self.shape) is not tuple or any(
            type(n) is not int or n < 0 for n in self.shape
        ):
            raise ValueError(
                "MeshingFieldStateRole.shape must be a tuple of nonnegative extents; got invalid shape"
            )
        if not isinstance(self.dtype, np.dtype) or self.dtype.hasobject:
            raise TypeError(
                "MeshingFieldStateRole.dtype must be a nonobject NumPy dtype; got incompatible dtype"
            )
        if type(self.state_epoch) is not int or self.state_epoch < 0:
            raise ValueError(
                "MeshingFieldStateRole.state_epoch must be a nonnegative integer; got invalid epoch"
            )
        if self.role == "state-epoch" and (
            self.shape != () or self.dtype.kind not in "iu"
        ):
            raise ValueError(
                "MeshingFieldStateRole.shape must describe a scalar epoch with integer dtype; got incompatible schema"
            )
        if self.role == "boundary-mask" and self.dtype != np.dtype("bool"):
            raise ValueError(
                "MeshingFieldStateRole.dtype must be boolean for boundary-mask; got incompatible dtype"
            )
        if self.value_units is not None and (
            type(self.value_units) is not tuple
            or not self.value_units
            or any(type(unit) is not UnitDefinition for unit in self.value_units)
        ):
            raise TypeError(
                "MeshingFieldStateRole.value_units must be an immutable nonempty tuple of units; got incompatible units"
            )
        if self.role == "material-history" and self.value_units is None:
            raise ValueError(
                "MeshingFieldStateRole.value_units must declare material-history units; got missing units"
            )
        self._validate_random_stream()
        if self.role not in ("state-epoch", "rng-key"):
            if type(self.index_binding) is MeshingPolyhedralFieldIndexBinding:
                if (
                    type(self) is not MeshingPolyhedralFieldStateRole
                    or self.role == "rng-counter"
                ):
                    raise ValueError(
                        "MeshingFieldStateRole.index_binding must retain its concrete physical owner; got a polyhedral binding on another role family"
                    )
                self.index_binding.__post_init__()
                index_shape = self.index_binding.index_shape
            elif type(self.index_binding) is MeshingFieldIndexBinding:
                index_shape = self.index_binding.index_values.shape
            else:
                raise ValueError(
                    "MeshingFieldStateRole.index_binding must declare indexed values or history; got missing binding"
                )
            prefix = tuple(
                self.shape[axis]
                for axis in self.index_binding.index_axes
                if axis < len(self.shape)
            )
            if prefix != index_shape:
                raise ValueError(
                    "MeshingFieldStateRole.shape must match index-bank axes; got incompatible payload axes"
                )
            if self.role in ("coefficients", "boundary-values", "boundary-mask") and (
                self.index_binding.bank_name.startswith("epoch/")
                or self.index_binding.field_space_id is None
            ):
                raise ValueError(
                    "MeshingFieldStateRole.index_binding must use compact coefficient identities for coefficients and essential boundaries; got raw or unbound rows"
                )
            if self.role == "rng-counter" and (
                self.index_binding.bank_name not in _RANDOM_STREAM_IDENTITY_BANKS
                or self.index_binding.field_space_id is not None
                or self.shape != index_shape
            ):
                raise ValueError(
                    "MeshingFieldStateRole.index_binding must key one rng-counter per stable entity ID; got coefficient ordinals or component axes"
                )
        elif self.index_binding is not None:
            raise ValueError(
                "MeshingFieldStateRole.index_binding must be absent for state-epoch and rng-key; got indexed scalar"
            )

    def _validate_random_stream(self) -> None:
        stream = self.random_stream
        if self.role not in ("rng-key", "rng-counter"):
            if stream is not None:
                raise ValueError(
                    "MeshingFieldStateRole.random_stream must be absent for physical state; got RNG address"
                )
            return
        if type(stream) is not MeshingRandomStream:
            raise TypeError(
                "MeshingFieldStateRole.random_stream must address RNG state; got missing or invalid stream"
            )
        stream.__post_init__()
        if self.value_units is not None:
            raise ValueError(
                "MeshingFieldStateRole.value_units must be absent for unitless RNG state; got units"
            )
        if self.role == "rng-key" and (
            self.dtype != np.dtype("uint32")
            or self.shape != stream.key_shape
            or stream.key_entry_name != self.entry_name
        ):
            raise ValueError(
                "MeshingFieldStateRole.dtype must hold the uint32 root key data named by its stream; got incompatible rng-key schema"
            )
        if self.role == "rng-counter" and self.dtype not in _RANDOM_COUNTER_DTYPES:
            raise ValueError(
                f"MeshingFieldStateRole.dtype must be uint32 or uint64 for rng-counter; got {self.dtype}"
            )


@dataclass(frozen=True, kw_only=True)
class MeshingPolyhedralFieldStateRole(MeshingFieldStateRole):
    """Explicit primal/dual/physical meaning on an actual polyhedral owner."""

    algebraic_role: MeshingPolyhedralAlgebraicRole
    component_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        MeshingFieldStateRole.__post_init__(self)
        parse(
            self.algebraic_role,
            MeshingPolyhedralAlgebraicRole,
            "MeshingPolyhedralFieldStateRole.algebraic_role",
        )
        binding = self.index_binding
        if type(binding) is not MeshingPolyhedralFieldIndexBinding:
            raise TypeError(
                "MeshingPolyhedralFieldStateRole.index_binding must select its real owner bank; got another binding"
            )
        if self.field_name != binding.owner or self.value_units is None:
            raise ValueError(
                "MeshingPolyhedralFieldStateRole.field_name must name its actual owner and explicitly declare physical units; got missing or foreign semantics"
            )
        if self.role not in (
            "coefficients",
            "physical-values",
            "dual-values",
            "trace-values",
            "stage-history",
            "material-history",
        ) or (
            self.role == "coefficients"
            and self.algebraic_role != "primal"
            or self.role == "dual-values"
            and self.algebraic_role != "dual"
        ):
            raise ValueError(
                "MeshingPolyhedralFieldStateRole.algebraic_role must explicitly match its primal/dual state role; got incompatible semantics"
            )
        component_shape = tuple(
            extent
            for axis, extent in enumerate(self.shape)
            if axis not in binding.index_axes
            and not (self.role == "stage-history" and axis == 0)
        )
        width = prod(component_shape)
        if (
            type(self.component_names) is not tuple
            or any(
                type(name) is not str or not name or name != name.strip()
                for name in self.component_names
            )
            or len(set(self.component_names)) != len(self.component_names)
            or (width > 1 and len(self.component_names) != width)
            or (width == 1 and len(self.component_names) not in (0, 1))
            or len(self.value_units) not in (1, width)
        ):
            raise ValueError(
                "MeshingPolyhedralFieldStateRole.component_names must explicitly label ordered vector/conserved components with matching units; got missing or incompatible component semantics"
            )


type MeshingFieldHistoryPolicy = Literal[
    "none", "accepted-state", "material-history", "multistep"
]
type MeshingFiniteVolumeCoordinatePolicy = Literal[
    "accepted-map", "represented-polyhedral"
]


class MeshingFieldDeclaration(StrictModule, NonTrainableState):
    """Bind physical field semantics and accepted state to one carrier."""

    __strict_abstract__ = True

    part_name: str
    topology_id: str
    geometry_layout_id: str
    field_space_ids: frozendict[str, str]
    value_units: frozendict[str, tuple[UnitDefinition, ...]]
    maximum_derivative_orders: frozendict[str, int]
    branch_policy: BranchDifferentiationPolicy
    trace_policy: FieldTracePolicy
    state_roles: tuple[MeshingFieldStateRole, ...]
    history_policy: MeshingFieldHistoryPolicy
    numeric_version: str
    source_wall_policy: eqx.AbstractVar[OversetPartSpec | None]

    @property
    @abstractmethod
    def owner(self) -> Literal["finite_element", "finite_volume", "polyhedral"]:
        """Return the physical preparation owner."""
        raise NotImplementedError

    @abstractmethod
    def __post_init__(self) -> None:
        """Validate declaration inputs and state bindings."""
        raise NotImplementedError

    @classmethod
    @checked
    def from_stored_fields(cls: type[Self], value: MeshingFieldDeclaration, /) -> Self:
        """Reconstruct stored inputs through the declaration constructor."""
        if type(value) is not cls:
            raise TypeError(
                f"{cls.__name__}.from_stored_fields.value must have the registered declaration type; got {type(value).__name__}"
            )
        expected = {member.name for member in dataclasses.fields(cls)}
        if set(vars(value)) != expected:
            raise ValueError(
                f"{cls.__name__}.from_stored_fields.value must retain the declared field set; got {sorted(vars(value))!r}"
            )
        return cls(
            **{
                member.name: getattr(value, member.name)
                for member in dataclasses.fields(value)
                if member.init
            }
        )


class _FieldSemantics(StrictModule):
    names: tuple[str, ...]
    widths: frozendict[str, int]
    derivative_ceilings: frozendict[str, int]


def _validate_declaration_identity(declaration: MeshingFieldDeclaration, /) -> None:
    for name, value in (
        ("part_name", declaration.part_name),
        ("topology_id", declaration.topology_id),
        ("geometry_layout_id", declaration.geometry_layout_id),
        ("numeric_version", declaration.numeric_version),
    ):
        if type(value) is not str or not value.strip():
            raise ValueError(
                f"MeshingFieldDeclaration.{name} must be a nonempty identity; got {value!r}"
            )


def _validate_field_component_semantics(
    declaration: MeshingFieldDeclaration, name: str, width: int, ceiling: int, /
) -> None:
    units = declaration.value_units[name]
    if (
        type(units) is not tuple
        or len(units) not in (1, width)
        or any(type(unit) is not UnitDefinition for unit in units)
    ):
        raise ValueError(
            f"MeshingFieldDeclaration.value_units.{name} must declare scalar or component units; got {units!r}"
        )
    order = declaration.maximum_derivative_orders[name]
    if type(order) is not int or order < 0:
        raise ValueError(
            f"MeshingFieldDeclaration.maximum_derivative_orders.{name} must be nonnegative; got {order!r}"
        )
    match declaration:
        case MeshingFiniteElementFieldDeclaration():
            if order > ceiling:
                raise ValueError(
                    f"MeshingFieldDeclaration.maximum_derivative_orders.{name} must not exceed reconstruction degree {ceiling}; got {order}"
                )
        case MeshingFiniteVolumeFieldDeclaration():
            # FV degree refusal precedes reconstruction branch/options validation.
            pass
        case _:
            raise TypeError(
                "MeshingFieldDeclaration.owner must select FE or FV component semantics; got unsupported declaration"
            )


def _validate_field_semantics(
    declaration: MeshingFieldDeclaration, semantics: _FieldSemantics, /
) -> None:
    names = semantics.names
    if len(set(names)) != len(names):
        raise ValueError(
            f"MeshingFieldDeclaration.fields must have unique names; got {names!r}"
        )
    for name, mapping in (
        ("field_space_ids", declaration.field_space_ids),
        ("value_units", declaration.value_units),
        ("maximum_derivative_orders", declaration.maximum_derivative_orders),
    ):
        if type(mapping) is not frozendict or set(mapping) != set(names):
            raise ValueError(
                f"MeshingFieldDeclaration.{name} must be immutable and keyed by {names!r}; got {mapping!r}"
            )
    if any(
        type(value) is not str or not value
        for value in declaration.field_space_ids.values()
    ):
        raise ValueError(
            f"MeshingFieldDeclaration.field_space_ids must contain nonempty identities; got {declaration.field_space_ids!r}"
        )
    for name in names:
        _validate_field_component_semantics(
            declaration, name, semantics.widths[name], semantics.derivative_ceilings[name]
        )
    if (
        type(declaration.branch_policy) is not BranchDifferentiationPolicy
        or type(declaration.trace_policy) is not FieldTracePolicy
    ):
        raise TypeError(
            "MeshingFieldDeclaration.branch_policy must be a derivative policy paired with a typed trace_policy; got invalid policy"
        )


def _validate_declaration_roles(
    declaration: MeshingFieldDeclaration, names: tuple[str, ...], /
) -> None:
    parse(
        declaration.history_policy,
        MeshingFieldHistoryPolicy,
        "MeshingFieldDeclaration.history_policy",
    )
    roles = declaration.state_roles
    if type(roles) is not tuple or any(
        type(role) is not MeshingFieldStateRole for role in roles
    ):
        raise TypeError(
            "MeshingFieldDeclaration.state_roles must be immutable field state records; got invalid roles"
        )
    if any(role.field_name not in names for role in roles):
        raise ValueError(
            f"MeshingFieldDeclaration.state_roles must bind declared fields {names!r}; got undeclared field"
        )
    if len({role.entry_name for role in roles}) != len(roles):
        raise ValueError(
            "MeshingFieldDeclaration.state_roles must have unique entry names; got duplicate entry"
        )
    coefficient_names = [role.field_name for role in roles if role.role == "coefficients"]
    if sorted(coefficient_names) != sorted(names):
        raise ValueError(
            f"MeshingFieldDeclaration.state_roles must have one coefficient role per field; got {coefficient_names!r}"
        )
    boundary_values = {
        role.field_name for role in roles if role.role == "boundary-values"
    }
    boundary_masks = {role.field_name for role in roles if role.role == "boundary-mask"}
    if boundary_values != boundary_masks:
        raise ValueError(
            "MeshingFieldDeclaration.state_roles must pair boundary masks and values; got unpaired roles"
        )
    keys = {role.entry_name: role for role in roles if role.role == "rng-key"}
    for role in roles:
        stream = role.random_stream
        key = None if stream is None else keys.get(stream.key_entry_name)
        if role.role == "rng-counter" and (key is None or key.random_stream != stream):
            raise ValueError(
                "MeshingFieldDeclaration.state_roles must declare the rng-key of every rng-counter stream; got missing or foreign root key"
            )
    _validate_declaration_history(declaration, roles)


def _validate_declaration_history(
    declaration: MeshingFieldDeclaration, roles: tuple[MeshingFieldStateRole, ...], /
) -> None:
    histories = [
        role
        for role in roles
        if role.role
        in (
            "material-history",
            "stage-history",
            "state-epoch",
            "rng-key",
            "rng-counter",
        )
    ]
    match declaration.history_policy:
        case "none":
            if histories:
                raise ValueError(
                    "MeshingFieldDeclaration.history_policy must permit retained history; got 'none' with history roles"
                )
        case "accepted-state" | "multistep":
            if not histories:
                raise ValueError(
                    "MeshingFieldDeclaration.state_roles must include history or epoch roles; got none"
                )
        case "material-history":
            if not any(role.role == "material-history" for role in histories):
                raise ValueError(
                    "MeshingFieldDeclaration.state_roles must include material history; got none"
                )
        case _:
            raise ValueError(
                f"MeshingFieldDeclaration.history_policy must select a supported policy; got {declaration.history_policy!r}"
            )
    if len({role.state_epoch for role in roles}) != 1:
        raise ValueError(
            "MeshingFieldDeclaration.state_roles must bind one state epoch; got multiple epochs"
        )
    wall = declaration.source_wall_policy
    if wall is not None and (
        type(wall) is not OversetPartSpec or wall.part_name != declaration.part_name
    ):
        raise ValueError(
            "MeshingFieldDeclaration.source_wall_policy must bind this part; got foreign wall policy"
        )


def _validate_field_elements(
    fields: tuple[FiniteElementFieldSpec, ...], /
) -> _FieldSemantics:
    for field in fields:
        for element in field.elements:
            if element.form_basis is not None:
                if (
                    type(element.form_basis) is not FormBasis
                    or type(element.tabulator) is not ProxyTabulator
                ):
                    raise TypeError(
                        "MeshingFiniteElementFieldDeclaration.elements must pair form basis and proxy tabulator; got invalid pair"
                    )
            elif type(element.tabulator) is ProxyTabulator:
                raise TypeError(
                    "MeshingFiniteElementFieldDeclaration.elements must declare the proxy form basis; got missing basis"
                )
            elif element.tabulator is None:
                if element.family not in ("Lagrange", "DiscontinuousLagrange"):
                    raise ValueError(
                        f"MeshingFiniteElementFieldDeclaration.elements must use a registered basis family; got {element.family!r}"
                    )
            elif type(element.tabulator) not in (
                FiniteElementTabulator,
                CoordinateTabulator,
            ):
                raise TypeError(
                    "MeshingFiniteElementFieldDeclaration.elements must retain portable tabulators; got unresolved callback"
                )
    return _FieldSemantics(
        tuple(field.name for field in fields),
        frozendict(
            {
                field.name: prod(field.component_shape)
                * prod(field.elements[0].value_shape)
                for field in fields
            }
        ),
        frozendict(
            {
                field.name: min(
                    min(element.degree, 1) if element.mapping == "identity" else 1
                    for element in field.elements
                )
                for field in fields
            }
        ),
    )


@final
class MeshingFiniteElementFieldDeclaration(MeshingFieldDeclaration):
    """Declare finite-element fields, precision, and accepted state."""

    finite_element_fields: tuple[FiniteElementFieldSpec, ...]
    precision_policy: FiniteElementPrecisionPolicy
    source_wall_policy: OversetPartSpec | None = None

    @property
    def owner(self) -> Literal["finite_element"]:
        return "finite_element"

    @checked
    def __post_init__(self: MeshingFiniteElementFieldDeclaration) -> None:
        _validate_declaration_identity(self)
        if (
            type(self.finite_element_fields) is not tuple
            or not self.finite_element_fields
            or any(
                type(field) is not FiniteElementFieldSpec
                for field in self.finite_element_fields
            )
            or type(self.precision_policy) is not FiniteElementPrecisionPolicy
        ):
            raise ValueError(
                "MeshingFiniteElementFieldDeclaration.finite_element_fields must declare field specs with a precision_policy; got invalid inputs"
            )
        semantics = _validate_field_elements(self.finite_element_fields)
        _validate_field_semantics(self, semantics)
        _validate_declaration_roles(self, semantics.names)


class _FiniteVolumeNames(StrictModule):
    name: str
    width: int


def _validate_finite_volume_names(
    declaration: MeshingFiniteVolumeFieldDeclaration, /
) -> _FiniteVolumeNames:
    name = declaration.finite_volume_field_name
    components = declaration.finite_volume_component_names
    if (
        type(name) is not str
        or not name
        or type(components) is not tuple
        or not components
        or any(
            type(component) is not str or not component or component != component.strip()
            for component in components
        )
    ):
        raise ValueError(
            "MeshingFiniteVolumeFieldDeclaration.finite_volume_field_name must declare a field with named components; got invalid names"
        )
    return _FiniteVolumeNames(name, len(components))


def _validate_finite_volume_reconstruction(
    declaration: MeshingFiniteVolumeFieldDeclaration, /
) -> _FieldSemantics:
    names = _validate_finite_volume_names(declaration)
    name = names.name
    parse(
        declaration.finite_volume_coordinate_policy,
        MeshingFiniteVolumeCoordinatePolicy,
        "MeshingFiniteVolumeFieldDeclaration.finite_volume_coordinate_policy",
    )
    policy = declaration.reconstruction_policy
    match policy:
        case PiecewiseConstantReconstruction():
            ceiling, branch, options = 0, BranchDifferentiationPolicy.SMOOTH, set()
        case CellPolynomialReconstructionPlan():
            ceiling, branch, options = (
                policy.degree,
                BranchDifferentiationPolicy.SMOOTH,
                {"stencil_direction"},
            )
        case UnstructuredWENOZReconstructionPlan():
            ceiling, branch, options = (
                policy.degree,
                BranchDifferentiationPolicy.BRANCHWISE,
                set(),
            )
        case _:
            raise TypeError(
                "MeshingFiniteVolumeFieldDeclaration.reconstruction_policy must be an unprepared native policy; got invalid owner"
            )
    if declaration.reconstruction_policy_id != policy.plan_id:
        raise ValueError(
            f"MeshingFiniteVolumeFieldDeclaration.reconstruction_policy_id must equal {policy.plan_id!r}; got {declaration.reconstruction_policy_id!r}"
        )
    order = declaration.maximum_derivative_orders.get(name, -1)
    if order > ceiling:
        raise ValueError(
            f"MeshingFieldDeclaration.maximum_derivative_orders.{name} must not exceed reconstruction degree {ceiling}; got {order}"
        )
    if declaration.branch_policy != branch:
        raise ValueError(
            f"MeshingFiniteVolumeFieldDeclaration.branch_policy must equal {branch!r}; got {declaration.branch_policy!r}"
        )
    if (
        type(declaration.reconstruction_options) is not frozendict
        or set(declaration.reconstruction_options) - options
    ):
        raise ValueError(
            f"MeshingFiniteVolumeFieldDeclaration.reconstruction_options must select {options!r}; got {declaration.reconstruction_options!r}"
        )
    return _FieldSemantics(
        (name,), frozendict({name: names.width}), frozendict({name: ceiling})
    )


def _validate_finite_volume_dynamics(
    declaration: MeshingFiniteVolumeFieldDeclaration, /
) -> None:
    allowed = {
        "boundary_face_groups",
        "neighborhood_complete",
        "physical_boundary_faces",
        "closure_evidence_id",
    }
    if (
        type(declaration.finite_volume_options) is not frozendict
        or set(declaration.finite_volume_options) - allowed
    ):
        raise ValueError(
            f"MeshingFiniteVolumeFieldDeclaration.finite_volume_options must select preparation inputs {allowed!r}; got {declaration.finite_volume_options!r}"
        )
    capacity = declaration.finite_volume_owned_face_capacity
    if capacity is not None and (type(capacity) is not int or capacity < 0):
        raise ValueError(
            f"MeshingFiniteVolumeFieldDeclaration.finite_volume_owned_face_capacity must be nonnegative; got {capacity!r}"
        )
    if len(set(declaration.finite_volume_component_names)) != len(
        declaration.finite_volume_component_names
    ):
        raise ValueError(
            "MeshingFiniteVolumeFieldDeclaration.finite_volume_component_names must be unique; got duplicate component"
        )
    if type(declaration.finite_volume_boundaries) is not frozendict:
        raise TypeError(
            "MeshingFiniteVolumeFieldDeclaration.finite_volume_boundaries must be immutable; got mutable boundaries"
        )
    if declaration.finite_volume_system is not None:
        if (
            type(declaration.finite_volume_system) is not EulerSystem
            or type(declaration.finite_volume_flux_policy) is not RusanovFluxPlan
            or declaration.finite_volume_component_names
            != declaration.finite_volume_system.component_names
            or any(
                type(value) not in (ExtrapolationBoundary, SlipWallBoundary)
                for value in declaration.finite_volume_boundaries.values()
            )
        ):
            raise ValueError(
                "MeshingFiniteVolumeFieldDeclaration.finite_volume_system must bind matching components, flux, and boundaries; got incompatible Euler inputs"
            )
    elif (
        declaration.finite_volume_flux_policy is not None
        or declaration.finite_volume_boundaries
    ):
        raise ValueError(
            "MeshingFiniteVolumeFieldDeclaration.finite_volume_system must be declared for dynamics policies; got no system"
        )


@final
class MeshingFiniteVolumeFieldDeclaration(MeshingFieldDeclaration):
    """Declare finite-volume reconstruction, dynamics, and accepted state."""

    finite_volume_field_name: str
    finite_volume_component_names: tuple[str, ...]
    finite_volume_coordinate_policy: MeshingFiniteVolumeCoordinatePolicy
    reconstruction_policy: (
        PiecewiseConstantReconstruction
        | CellPolynomialReconstructionPlan
        | UnstructuredWENOZReconstructionPlan
    )
    reconstruction_policy_id: str
    reconstruction_options: frozendict[str, Any] = frozendict()
    finite_volume_options: frozendict[str, Any] = frozendict()
    finite_volume_owned_face_capacity: int | None = None
    finite_volume_system: EulerSystem | None = None
    finite_volume_flux_policy: RusanovFluxPlan | None = None
    finite_volume_boundaries: frozendict[
        str, ExtrapolationBoundary | SlipWallBoundary
    ] = frozendict()
    source_wall_policy: OversetPartSpec | None = None

    @property
    def owner(self) -> Literal["finite_volume"]:
        return "finite_volume"

    @checked
    def __post_init__(self: MeshingFiniteVolumeFieldDeclaration) -> None:
        _validate_declaration_identity(self)
        semantics = _validate_finite_volume_reconstruction(self)
        _validate_finite_volume_dynamics(self)
        _validate_field_semantics(self, semantics)
        _validate_declaration_roles(self, semantics.names)


@final
class MeshingPolyhedralFieldDeclaration(MeshingFieldDeclaration):
    """Actual quotient VEM, native hybrid, and source-star FE state together."""

    owner_binding: MeshingPolyhedralOwnerBinding
    normalization_reference_length: UnitDefinition | None = None
    source_wall_policy: OversetPartSpec | None = None

    @property
    def owner(self) -> Literal["polyhedral"]:
        return "polyhedral"

    def __post_init__(self) -> None:
        _validate_declaration_identity(self)
        # One unit of L declares the approved normalization: dimensionless
        # primitives/diffusion, numeric-one source/reaction couplings L^-2 and
        # original affine boundary multipliers L^-1. Role units remain explicit.
        reference = self.normalization_reference_length
        if reference is not None and (
            type(reference) is not UnitDefinition or reference.dimension != LENGTH
        ):
            raise ValueError(
                "MeshingPolyhedralFieldDeclaration.normalization_reference_length must declare one exact physical length unit; got a non-length reference"
            )
        binding = self.owner_binding
        if type(binding) is not MeshingPolyhedralOwnerBinding or (
            binding.source_topology_id != self.topology_id
            or binding.source_geometry_layout_id != self.geometry_layout_id
            or binding.numeric_version != self.numeric_version
        ):
            raise ValueError(
                "MeshingPolyhedralFieldDeclaration.owner_binding must bind this source topology, geometry layout and numerical revision; got foreign owner"
            )
        if (
            type(self.field_space_ids) is not frozendict
            or dict(self.field_space_ids) != dict(binding.field_space_ids)
            or type(self.value_units) is not frozendict
            or set(self.value_units) != set(binding.field_space_ids)
            or type(self.maximum_derivative_orders) is not frozendict
            or set(self.maximum_derivative_orders) != set(binding.field_space_ids)
            or any(
                type(order) is not int or not 0 <= order <= 1
                for order in self.maximum_derivative_orders.values()
            )
            or type(self.branch_policy) is not BranchDifferentiationPolicy
            or type(self.trace_policy) is not FieldTracePolicy
        ):
            raise ValueError(
                "MeshingPolyhedralFieldDeclaration.field_space_ids must retain only its real coefficient spaces and declared reconstruction policies; got invented spaces or invalid policies"
            )
        roles = self.state_roles
        if (
            not roles
            or type(roles) is not tuple
            or any(type(role) is not MeshingPolyhedralFieldStateRole for role in roles)
        ):
            raise TypeError(
                "MeshingPolyhedralFieldDeclaration.state_roles must retain explicit owner-bound primal/dual/physical records; got invalid roles"
            )
        if len({role.entry_name for role in roles}) != len(roles):
            raise ValueError(
                "MeshingPolyhedralFieldDeclaration.state_roles must uniquely own every array; got duplicate entries"
            )
        coefficients = {}
        for role in roles:
            role.__post_init__()
            index = role.index_binding
            if type(index) is not MeshingPolyhedralFieldIndexBinding or (
                index.owner_binding is not binding
                and not content_equal(
                    index.owner_binding, binding, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
                )
            ):
                raise ValueError(
                    "MeshingPolyhedralFieldDeclaration.state_roles must retain this complete actual owner binding; got foreign index authority"
                )
            if role.role == "coefficients":
                if role.field_name in coefficients:
                    raise ValueError(
                        "MeshingPolyhedralFieldDeclaration.state_roles must retain one current primal coefficient bank per real space; got duplicate primal bank"
                    )
                coefficients[role.field_name] = role.value_units
        if coefficients != dict(self.value_units):
            raise ValueError(
                "MeshingPolyhedralFieldDeclaration.value_units must match its explicit VEM/source-star primal coefficient roles; got absent or conflicting units"
            )
        _validate_declaration_history(self, roles)


@dataclass(frozen=True, slots=True)
class _PreparedPolyhedralOwnerCache:
    carrier: CellMeshingResult
    binding: MeshingPolyhedralOwnerBinding
    owners: PolyhedralArchiveOwners
    payload: str
    constructors: tuple[tuple[type, str, int], ...]


def _polyhedral_owner_payload(
    binding: MeshingPolyhedralOwnerBinding,
    owners: PolyhedralArchiveOwners,
    budget: CoordinateEnclosureBudget,
    constructors: dict[int, Callable[..., Any]],
    /,
) -> tuple[str, tuple[tuple[type, str, int], ...]]:
    """Seal actual ephemeral outputs through the complete storage-owner traversal."""
    from jax.extend.core import Jaxpr

    from .._host_storage import (
        declared_object_storage_bytes,
        retained_host_storage_profile,
    )
    from .._meshcore import is_native_host_buffer

    digest = hashlib.sha256()

    def program_payload(value: Jaxpr) -> object:
        from .._identity import _CompilerPayload

        class PaidProgramPayload(_CompilerPayload):
            def parameter(self, item: Any, path: str, /) -> Any:
                budget.reserve(1, 1024 + 8 * declared_object_storage_bytes(item))
                return super().parameter(item, path)

            def array(self, item: Any, path: str, /) -> Any:
                size = int(getattr(item, "size", 1))
                storage = int(getattr(item, "nbytes", 32))
                budget.reserve(size, 512 + 2 * storage)
                return super().array(item, path)

            def program(self, item: Any, path: str, /, *, dynamic_captures: bool) -> Any:
                graph = item.jaxpr if hasattr(item, "jaxpr") else item
                count = (
                    len(graph.eqns)
                    + len(graph.invars)
                    + len(graph.outvars)
                    + len(graph.constvars)
                )
                budget.reserve(count, 1024 + 512 * count)
                return super().program(item, path, dynamic_captures=dynamic_captures)

        compiler = PaidProgramPayload()
        root = compiler.program(value, "polyhedral-owner", dynamic_captures=False)
        count = len(compiler.metadata) + len(compiler.programs) + len(compiler.rules)
        budget.reserve(count, 512 + 256 * count)
        return {"kind": "compiler-program", "root": root, **compiler.definitions()}

    classes: dict[int, type] = {}
    methods = (
        "__init__",
        "__new__",
        "__call__",
        "prepare",
        "from_cell_mesh",
        "from_stored_fields",
        "validate_restored",
        "vertex_field_space",
    )

    def observe(value: object) -> None:
        owner_type = type(value)
        budget.reserve(1, 1024 + 4 * declared_object_storage_bytes(value))
        record: list[object] = [owner_type.__module__, owner_type.__qualname__]
        if isinstance(value, Jaxpr):
            record.append(program_payload(value))
        elif isinstance(value, np.ndarray):
            budget.reserve(value.size, 2 * value.nbytes)
            if value.dtype.hasobject:
                if any(not isinstance(device, jax.Device) for device in value.flat):
                    raise TypeError(
                        "Polyhedral owner reuse requires explicit device metadata, not object banks"
                    )
                record.extend(
                    (tuple(value.shape), tuple(id(device) for device in value.flat))
                )
            else:
                record.extend(
                    (
                        tuple(value.shape),
                        value.dtype.str,
                        value.flags.writeable,
                        array_bytes_digest(value),
                    )
                )
        elif isinstance(value, jax.Array):
            if value.is_deleted():
                raise ValueError(
                    "Polyhedral owner reuse must retain live actual JAX banks"
                )
            record.extend((id(value), tuple(value.shape), str(value.dtype)))
        elif isinstance(value, jax.tree_util.PyTreeDef):
            record.extend((value.num_nodes, value.num_leaves))
        elif isinstance(value, jax.Device):
            record.append(id(value))
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            classes[id(owner_type)] = owner_type
            members = dataclasses.fields(value)
            budget.reserve(len(members), 64 * len(members))
            record.append(
                tuple(
                    (member.name, id(getattr(value, member.name))) for member in members
                )
            )
        elif isinstance(value, type):
            classes[id(value)] = value
            record.append(id(value))
        elif type(value) in (float, complex):
            record.append(np.asarray(value).tobytes().hex())
        elif value is None or type(value) in (bool, int, str):
            record.append(value)
        elif isinstance(value, (bytes, bytearray)):
            record.append(bytes(value).hex())
        elif isinstance(value, np.generic):
            record.extend((value.dtype.str, value.tobytes().hex()))
        elif isinstance(value, np.dtype):
            if value.metadata:
                raise TypeError(
                    "Polyhedral owner reuse requires explicit numerical dtype metadata"
                )
            record.append(repr(value.descr))
        elif isinstance(value, (Mapping, tuple, list, set, frozenset)):
            record.append(len(value))
        elif isinstance(value, memoryview):
            budget.reserve(value.nbytes, 2 * value.nbytes)
            record.append(hashlib.sha256(value.tobytes()).hexdigest())
        elif isinstance(value, ctypes.Array):
            if is_native_host_buffer(value):
                record.append(id(value))  # Defined logical NumPy views are sealed above.
            else:
                size = ctypes.sizeof(value)
                budget.reserve(len(value), 2 * size)
                record.append(hashlib.sha256(bytes(value)).hexdigest())
        elif hasattr(value, "__dict__") and not callable(value):
            record.append(tuple(vars(value)))
        digest.update(canonical_json(record).encode("utf-8"))

    with budget.temporary_scope():
        retained_host_storage_profile(
            (binding, owners), {}, is_native_host_buffer, admit_owner=observe
        )
        issued: list[tuple[type, str, int]] = []
        for owner_type in classes.values():
            for name in methods:
                function = getattr_static(owner_type, name, None)
                function = getattr(function, "__func__", function)
                if callable(function):
                    budget.reserve(1, 128)
                    constructors[id(function)] = function
                    issued.append((owner_type, name, id(function)))
        return digest.hexdigest(), tuple(
            sorted(issued, key=lambda row: (id(row[0]), row[1]))
        )


@checked
def prepare_meshing_polyhedral_owner_binding(
    carrier: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
    vem_field_name: str = "vem",
    owners: PolyhedralArchiveOwners | None = None,
    source_validation: Any | None = None,
) -> tuple[MeshingPolyhedralOwnerBinding, PolyhedralArchiveOwners]:
    """Bind actual native plans; supplied plans still need canonical epoch admission."""
    _require_field_source_validation(source_validation)
    from .._meshcore import (
        current_native_execution_scope,
        NativeBatchTicket,
        NativeScopeTicket,
    )
    from . import _meshing_sources as source_owner

    scope = source_owner._source_owner_reconstruction_scope.get()
    if budget is None and scope is not None and scope.process_id == getpid():
        budget = scope.budget
    key = None
    if (
        owners is None
        and budget is not None
        and scope is not None
        and scope.process_id == getpid()
        and scope.budget is budget
        and not budget.ticket.settled
        and isinstance(budget.ticket, (NativeBatchTicket, NativeScopeTicket))
        and current_native_execution_scope() is not None
    ):
        source_owner.register_meshing_source_artifacts()
        scope.request_budgets[id(budget)] = budget
        dependencies = (
            polyhedral_owner.prepare_polyhedral_archive_owners,
            polyhedral_owner.prepare_polyhedral_h1_virtual_element_3d,
            polyhedral_owner.prepare_polyhedral_source_star_fem,
            polyhedral_owner.polyhedral_archive_index_banks,
            polyhedral_owner.UnstructuredFiniteVolumePlan.from_cell_mesh.__func__,
            polyhedral_owner.UnstructuredFiniteVolumePlan.prepare,
            polyhedral_owner.HybridMimeticDiffusion.__init__,
            MeshingPolyhedralOwnerBinding.__init__,
            MeshingCoefficientIdentityBank.from_field_space.__func__,
        )
        for function in dependencies:
            scope.constructors[id(function)] = function
        key = source_owner._source_owner_reconstruction_key(
            (
                carrier,
                vem_field_name,
                id(carrier),
                id(budget),
                tuple(map(id, dependencies)),
            ),
            scope.archive_limits,
            source_validation,
            prepare_meshing_polyhedral_owner_binding,
            scope,
        )
        issued = None if key is None else scope.polyhedral_owners.get(key)
        if key is not None and issued is not None:
            same_constructors = all(
                id(
                    getattr(
                        getattr_static(owner_type, name, None),
                        "__func__",
                        getattr_static(owner_type, name, None),
                    )
                )
                == identifier
                for owner_type, name, identifier in issued.constructors
            )
            if same_constructors:
                payload, _ = _polyhedral_owner_payload(
                    issued.binding, issued.owners, budget, scope.constructors
                )
                if payload != issued.payload:
                    raise ValueError(
                        "MeshingPolyhedralOwnerBinding.preparation must retain its actual issued numerical owners; got changed owner payload"
                    )
                return issued.binding, issued.owners
            del scope.polyhedral_owners[key]
    actual = (
        polyhedral_owner.prepare_polyhedral_archive_owners(carrier, budget=budget)
        if owners is None
        else owners
    )
    star = actual.source_star
    if (
        actual.vem.mesh.mesh_id != carrier.mesh.mesh_id
        or star.source_mesh_id != carrier.mesh.mesh_id
        or star.source_geometry_id != cell_geometry_id(carrier.geometry)
    ):
        raise ValueError(
            "MeshingPolyhedralOwnerBinding.preparation must retain the actual carrier and source maps; got foreign plans"
        )
    spaces = {
        "vem": actual.vem.vertex_field_space(vem_field_name),
        "source_star": star.space.field_spaces[0],
    }
    binding = MeshingPolyhedralOwnerBinding(
        source_result_id=carrier.result_id,
        source_mesh_id=carrier.mesh.mesh_id,
        source_topology_id=carrier.mesh.topology_id,
        source_geometry_id=cell_geometry_id(carrier.geometry),
        source_geometry_layout_id=carrier.geometry.geometry_layout_id,
        coordinate_contract_id=carrier.coordinate_contract.spatial_id,
        numeric_version=carrier.mesh.numeric_version,
        preparation_ids=frozendict(
            {
                "vem": actual.vem.prepared_id,
                "finite_volume": actual.finite_volume.prepared_id,
                "hybrid": actual.hybrid.facet_accumulation.operator_id,
                "source_star": star.plan_id,
                "source_star_space": star.space.prepared_id,
            }
        ),
        field_space_ids=frozendict(
            {owner: space.field_space_id for owner, space in spaces.items()}
        ),
        field_names=frozendict({owner: space.name for owner, space in spaces.items()}),
        coefficient_banks=frozendict(
            {
                owner: MeshingCoefficientIdentityBank.from_field_space(space)
                for owner, space in spaces.items()
            }
        ),
        index_banks=polyhedral_owner.polyhedral_archive_index_banks(actual),
        budget=budget,
    )
    if key is not None and scope is not None and budget is not None:
        payload, constructors = _polyhedral_owner_payload(
            binding, actual, budget, scope.constructors
        )
        issued = _PreparedPolyhedralOwnerCache(
            carrier, binding, actual, payload, constructors
        )
        budget.retain_owner(issued)
        scope.polyhedral_owners[key] = issued
    return binding, actual


@checked
def prepare_meshing_polyhedral_field_declaration(
    carrier: CellMeshingResult,
    binding: MeshingPolyhedralOwnerBinding,
    roles: tuple[MeshingPolyhedralFieldStateRole, ...],
    /,
    *,
    part_name: str,
    history_policy: MeshingFieldHistoryPolicy = "multistep",
    normalization_reference_length: UnitDefinition | None = None,
) -> MeshingPolyhedralFieldDeclaration:
    """Declare explicit scientific state, never infer meaning or units from shape."""
    return MeshingPolyhedralFieldDeclaration(
        part_name=part_name,
        topology_id=carrier.mesh.topology_id,
        geometry_layout_id=carrier.geometry.geometry_layout_id,
        field_space_ids=binding.field_space_ids,
        value_units=frozendict(
            {
                role.field_name: role.value_units
                for role in roles
                if role.role == "coefficients"
            }
        ),
        maximum_derivative_orders=frozendict(
            {name: 0 for name in binding.field_space_ids}
        ),
        branch_policy=BranchDifferentiationPolicy.SMOOTH,
        trace_policy=FieldTracePolicy("cell-sided"),
        state_roles=roles,
        history_policy=history_policy,
        numeric_version=carrier.mesh.numeric_version,
        owner_binding=binding,
        normalization_reference_length=normalization_reference_length,
    )


def _require_execution_endpoint_extension(
    original: CellMeshingResult, updated: CellMeshingResult, /
) -> None:
    """Admit only the same scientific publication with its real continued receipt."""
    previous, current = original.execution_evidence, updated.execution_evidence
    if previous is not None:
        NativeExecutionRecord.require_valid(previous)
    if current is not None:
        NativeExecutionRecord.require_valid(current)
    if content_equal(original, updated, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS):
        return
    if (
        previous is None
        or current is None
        or previous.owner_id is None
        or current.owner_id != previous.owner_id
        or not content_equal(
            CellMeshingResult.with_execution_evidence(original, current),
            updated,
            limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
        )
    ):
        raise ValueError(
            "MeshingAcceptedEpoch.transition.endpoint must change only its owner-bound execution receipt; got changed scientific content or foreign owner"
        )
    # Compare the complete prior payload, not its digest alone. Completed
    # request history is provenance, never same-request endpoint authority.
    pending: list[NativeExecutionRecord] = [current]
    seen: set[int] = set()
    while pending:
        receipt = pending.pop()
        if id(receipt) in seen:
            continue
        seen.add(id(receipt))
        if receipt is previous or content_equal(
            previous, receipt, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
        ):
            return
        pending.extend(
            predecessor
            for predecessor in (
                receipt.preparation_evidence,
                receipt.consumer_evidence,
            )
            if predecessor is not None
        )
    raise ValueError(
        "MeshingAcceptedEpoch.transition.endpoint.execution_evidence must continue the actual preceding receipt; got unrelated execution evidence"
    )


def _replace_meshing_root_children[T](record: T, replacements: Mapping[str, Any], /) -> T:
    """Replace declared root slots without copying their retained child owners."""
    children, structure = jax.tree_util.tree_flatten_with_path(
        record, is_leaf=lambda child: child is not record
    )
    selected = {
        (jax.tree_util.GetAttrKey(name),): value for name, value in replacements.items()
    }
    if not selected.keys() <= {path for path, _ in children}:
        raise RuntimeError(
            "MeshingAcceptedEpoch.transition.endpoints must select declared dynamic root fields; got an absent field"
        )
    return jax.tree_util.tree_unflatten(
        structure, [selected.get(path, child) for path, child in children]
    )


@checked
def rebind_meshing_adaptation_execution_endpoints(
    adaptation: MeshAdaptationResult,
    source: CellMeshingResult,
    target: CellMeshingResult,
    /,
) -> MeshAdaptationResult:
    """Bind exact accepted endpoints after their same-request consumers finish.

    Only execution evidence is outside scientific result/preparation identities.
    Proving that congruence preserves all existing transition proofs; rebuilding a
    preparation would instead execute another request with another allowance.
    """
    _require_execution_endpoint_extension(adaptation.source, source)
    _require_execution_endpoint_extension(adaptation.target, target)
    transition = adaptation.transition
    if transition is not None:
        if (
            transition.source_mesh_id != source.mesh.mesh_id
            or transition.source_topology_id != source.mesh.topology_id
        ):
            raise ValueError(
                "MeshingAcceptedEpoch.transition.source must retain the nested transition source; got changed mesh or topology"
            )
        if transition.target is not adaptation.target:
            _require_execution_endpoint_extension(transition.target, target)
    replacements: dict[str, Any] = {"source": source, "target": target}
    if transition is not None:
        replacements["transition"] = _replace_meshing_root_children(
            transition, {"target": target}
        )
    rebound = _replace_meshing_root_children(adaptation, replacements)
    return rebound


class MeshingGeometryMotionTransition(StrictModule, NonTrainableState):
    """Retain actual endpoint publications and their canonical motion proof."""

    source: CellMeshingResult
    target: CellMeshingResult
    motion: CellMeshTransition
    result_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: CellMeshingResult,
        target: CellMeshingResult,
        motion: CellMeshTransition,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        rebuilt_motion = CellMeshTransition.from_stored_fields(motion, budget=budget)
        if not content_equal(motion, rebuilt_motion, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS):
            raise ValueError(
                "MeshingGeometryMotionTransition.motion must retain its complete canonical proof; got changed fields"
            )
        geometry = motion.geometry_transition
        reconstructed_geometry = transition_displaced_cell_geometry(
            source.mesh, source.geometry, target.mesh, budget=budget,
            _issued_transition=geometry,
        )
        if not content_equal(
            geometry, reconstructed_geometry, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
        ):
            raise ValueError(
                "MeshingGeometryMotionTransition.geometry must match the source-authoritative canonical displacement proof; got changed evidence or reference banks"
            )
        if (
            motion.transition_kind is not MeshTransitionKind.MOTION
            or motion.source_mesh_id != source.mesh.mesh_id
            or motion.source_topology_id != source.mesh.topology_id
            or source.mesh.topology_id != target.mesh.topology_id
            or geometry is None
            or geometry.source_geometry_id != cell_geometry_id(source.geometry)
            or not content_equal(
                motion.target, target, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
            )
            or not content_equal(
                motion.lineage,
                identity_lineage(source.mesh, target.mesh),
                limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
            )
            or any(
                not content_equal(
                    mesh_global_ids(source.mesh, dimension),
                    mesh_global_ids(target.mesh, dimension),
                    limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
                )
                for dimension in range(source.mesh.topological_dimension + 1)
            )
        ):
            raise ValueError(
                "MeshingGeometryMotionTransition must retain actual same-topology endpoints and their canonical geometry motion; got incompatible proof"
            )
        for endpoint in (source, target):
            rebuilt = CellMeshingResult.from_stored_fields(endpoint)
            if not content_equal(rebuilt, endpoint, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS):
                raise ValueError(
                    "MeshingGeometryMotionTransition.endpoint must retain an independently valid publication; got changed fields"
                )
        self.source = source
        self.target = target
        self.motion = motion
        register_meshing_source_types()
        self.result_id = stored_record_content_id(
            self, excluded="result_id", budget=budget
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[Self],
        value: MeshingGeometryMotionTransition,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> Self:
        if type(value) is not cls or set(vars(value)) != {
            "source",
            "target",
            "motion",
            "result_id",
        }:
            raise TypeError(
                "MeshingGeometryMotionTransition must retain its exact fieldset; got invalid fields"
            )
        rebuilt = cls(value.source, value.target, value.motion, budget=budget)
        if rebuilt.result_id != value.result_id:
            raise ValueError(
                "MeshingGeometryMotionTransition.result_id must match full retained content; got changed identity"
            )
        return rebuilt


class MeshingAcceptedEpoch(StrictModule, NonTrainableState):
    """Retain an accepted carrier, physical declarations, and state arrays."""

    carrier: CellMeshingResult
    declarations: tuple[MeshingFieldDeclaration, ...]
    fields: frozendict[str, jax.Array | np.ndarray]
    transition: MeshAdaptationResult | MeshingGeometryMotionTransition | None
    epoch_id: str = eqx.field(static=True)

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[Self],
        value: MeshingAcceptedEpoch,
        /,
        *,
        collective: bool = False,
        source_validation: Any | None = None,
    ) -> Self:
        """Rebuild the accepted epoch and recompute its identity."""
        return cls(
            value.carrier,
            value.declarations,
            value.fields,
            transition=value.transition,
            collective=collective,
            source_validation=source_validation,
        )

    @checked
    def __init__(
        self,
        carrier: CellMeshingResult,
        declarations: tuple[MeshingFieldDeclaration, ...],
        fields: Mapping[str, jax.Array | np.ndarray],
        /,
        *,
        transition: MeshAdaptationResult | MeshingGeometryMotionTransition | None = None,
        collective: bool = False,
        source_validation: Any | None = None,
    ) -> None:
        with (
            agreed_local_step(stage="meshing-accepted-epoch-field-admission")
            if collective
            else nullcontext()
        ):
            if collective and source_validation is None:
                raise ValueError("Collective epoch reconstruction requires current source validation")
            self._initialize_fields(
                carrier, declarations, fields, transition, source_validation
            )
        self.epoch_id = self._identity(collective=collective)

    def _initialize_fields(
        self,
        carrier: CellMeshingResult,
        declarations: tuple[MeshingFieldDeclaration, ...],
        fields: Mapping[str, jax.Array | np.ndarray],
        transition: MeshAdaptationResult | MeshingGeometryMotionTransition | None,
        source_validation: Any | None,
    ) -> None:
        _require_field_source_validation(source_validation)
        if (
            type(declarations) is not tuple
            or not declarations
            or any(
                type(declaration)
                not in (
                    MeshingFiniteElementFieldDeclaration,
                    MeshingFiniteVolumeFieldDeclaration,
                    MeshingPolyhedralFieldDeclaration,
                )
                for declaration in declarations
            )
        ):
            raise TypeError(
                "MeshingAcceptedEpoch.declarations must be a nonempty tuple of concrete field declarations; got invalid declarations"
            )
        if len({declaration.part_name for declaration in declarations}) != 1:
            raise ValueError(
                "MeshingAcceptedEpoch.declarations.part_name must bind one carrier name; got differing names"
            )
        if not isinstance(fields, Mapping) or any(
            type(name) is not str
            or not name
            or not isinstance(value, (jax.Array, np.ndarray))
            for name, value in fields.items()
        ):
            raise TypeError(
                "MeshingAcceptedEpoch.fields must map nonempty names to numerical arrays; got invalid banks"
            )
        if transition is not None:
            equal = content_equal if source_validation is None else source_validation.equivalent
            if not equal(
                transition.target, carrier, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
            ):
                raise ValueError(
                    "MeshingAcceptedEpoch.carrier must match transition.target; got differing carrier"
                )
        self.carrier = carrier
        self.declarations = declarations
        self.fields = frozendict(dict(sorted(fields.items())))
        self.transition = transition
        _validate_field_owner_banks(
            tuple((declaration, carrier) for declaration in declarations),
            self.fields,
            index_source=carrier if carrier.mesh.storage is not None else None,
            source_validation=source_validation,
        )

    def _identity(self, *, collective: bool = False) -> str:
        with (
            agreed_local_step(stage="meshing-accepted-epoch-identity-inputs")
            if collective
            else nullcontext()
        ):
            register_meshing_source_types()
            recipe = model_structure_recipe(self.declarations)
            arrays = model_recipe_array_values(
                self.declarations, recipe, prefix="declarations"
            )
        return canonical_fingerprint(
            {
                "kind": "meshing-accepted-epoch",
                "carrier": self.carrier.result_id,
                "topology": self.carrier.mesh.topology_id,
                "numeric_version": self.carrier.mesh.numeric_version,
                "geometry": cell_geometry_id(self.carrier.geometry),
                "declarations": recipe,
                "declaration_arrays": logical_array_value_collection_digest(arrays),
                "fields": logical_array_value_collection_digest(self.fields),
                "transition": None
                if self.transition is None
                else self.transition.result_id,
            }
        )


@checked
def rebind_meshing_accepted_epoch_execution(
    epoch: MeshingAcceptedEpoch,
    carrier: CellMeshingResult,
    /,
    *,
    transition: MeshAdaptationResult | None,
) -> MeshingAcceptedEpoch:
    """Seal a generation/adaptation epoch after its paid consumer has ended.

    This boundary changes only execution receipts, never reruns geometry or
    field-owner preparation. Actual retained field/declaration bytes must still
    match the earned epoch identity, and adaptation endpoints must continue
    their exact paid predecessor receipts. Motion identities currently include
    endpoint receipts and cannot borrow this receipt-independent cutover.
    """
    if epoch._identity() != epoch.epoch_id:
        raise ValueError(
            "MeshingAcceptedEpoch.execution must retain its exact admitted declarations and fields; got changed epoch payload"
        )
    _require_execution_endpoint_extension(epoch.carrier, carrier)
    previous = epoch.transition
    if previous is None:
        if transition is not None:
            raise ValueError(
                "MeshingAcceptedEpoch.execution must retain its generation history; got an introduced transition"
            )
    elif isinstance(previous, MeshAdaptationResult) and transition is not None:
        expected = rebind_meshing_adaptation_execution_endpoints(
            previous, transition.source, carrier
        )
        if transition.result_id != previous.result_id or not content_equal(
            expected, transition, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
        ):
            raise ValueError(
                "MeshingAcceptedEpoch.execution.transition must retain its exact admitted adaptation; got changed transition science or receipts"
            )
    else:
        raise ValueError(
            "MeshingAcceptedEpoch.execution must retain a receipt-independent generation/adaptation transition; got a missing or incompatible transition"
        )
    return _replace_meshing_root_children(
        epoch, {"carrier": carrier, "transition": transition}
    )


def validate_meshing_accepted_epochs(
    records: Mapping[str, Any], /, *, source_validation: Any | None = None
) -> tuple[MeshingAcceptedEpoch, ...]:
    """Authenticate an ordered field history against its generation and transitions."""
    _require_field_source_validation(source_validation)
    equal = content_equal if source_validation is None else source_validation.equivalent

    accepted = records.get("accepted_data")
    target = records.get("accepted_target")
    if not isinstance(accepted, Mapping) or "accepted_epochs" not in accepted:
        if isinstance(accepted, Mapping) and "neurofluid_transport" in accepted:
            return ()
        if target is not None and not (
            isinstance(accepted, Mapping)
            and ("adaptation_results" in accepted or "adaptation_event" in accepted)
        ):
            raise ValueError(
                "MeshingSourceClosure.accepted_target must retain accepted history; got missing history"
            )
        return ()
    epochs = accepted["accepted_epochs"]
    if (
        type(epochs) is not tuple
        or not epochs
        or any(type(epoch) is not MeshingAcceptedEpoch for epoch in epochs)
    ):
        raise TypeError(
            "accepted_data.accepted_epochs must be a nonempty ordered tuple of MeshingAcceptedEpoch records; got invalid epochs"
        )
    epochs_ = cast(tuple[MeshingAcceptedEpoch, ...], epochs)
    if records.get("generation_part") is not None:
        raise ValueError(
            "accepted_data.accepted_epochs must own generation without generation_part; got duplicate generation"
        )
    if not epochs_[0].declarations:
        raise ValueError(
            "accepted_data.accepted_epochs must bind an initial part name; got absent declarations"
        )
    generation = MeshPart(epochs_[0].declarations[0].part_name, epochs_[0].carrier)
    if any(
        declaration.part_name != generation.name
        for epoch in epochs_
        for declaration in epoch.declarations
    ):
        raise ValueError(
            "accepted_data.accepted_epochs.declarations.part_name must preserve the initial part; got differing part name"
        )
    if epochs_[0].transition is not None or not equal(
        epochs_[0].carrier,
        generation.carrier,
        limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
    ):
        raise ValueError(
            "accepted_data.accepted_epochs must start at generation without a transition; got incompatible initial epoch"
        )
    for previous, epoch in zip(epochs_, epochs_[1:], strict=False):
        transition = epoch.transition
        if (
            transition is None
            or not equal(
                transition.source,
                previous.carrier,
                limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
            )
            or not equal(
                transition.target,
                epoch.carrier,
                limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
            )
        ):
            raise ValueError(
                "accepted_data.accepted_epochs.transition must bind preceding source and current target; got incompatible transition"
            )
    if type(target) is not CellMeshingResult or not equal(
        target,
        epochs_[-1].carrier,
        limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
    ):
        raise ValueError(
            "MeshingSourceClosure.accepted_target must match the final accepted epoch carrier; got differing target"
        )
    if "adaptation_results" in accepted:
        raise ValueError(
            "accepted_data.adaptation_results must be absent when accepted_epochs retain transitions; got duplicate history"
        )
    return epochs_


def _validate_field_owner_banks(
    owners: tuple[tuple[MeshingFieldDeclaration, Any], ...],
    arrays: Mapping[str, Any],
    /,
    *,
    index_source: CellMeshingResult | None = None,
    registration: OversetRegistration | None = None,
    require_complete: bool = True,
    source_validation: Any | None = None,
) -> None:
    """Validate physical ownership and schemas of accepted state banks."""
    _require_field_source_validation(source_validation)
    declared_entries: set[str] = set()
    for declaration, carrier in owners:
        prepared, reconstruction = prepare_meshing_field_owner(
            declaration,
            carrier,
            index_source=index_source,
            source_validation=source_validation,
        )
        match declaration:
            case MeshingFiniteVolumeFieldDeclaration() if (
                declaration.finite_volume_system is not None
            ):
                _prepare_field_dynamics(declaration, prepared, reconstruction)
            case (
                MeshingFiniteElementFieldDeclaration()
                | MeshingFiniteVolumeFieldDeclaration()
                | MeshingPolyhedralFieldDeclaration()
            ):
                pass
            case _:
                raise TypeError(
                    "MeshingFieldDeclaration.owner must select FE or FV state; got unsupported declaration"
                )
        if declaration.source_wall_policy is not None:
            if registration is None:
                raise ValueError(
                    "MeshingFieldDeclaration.source_wall_policy must bind an OversetRegistration; got missing registration"
                )
            specs = {spec.part_name: spec for spec in registration.specs}
            if (
                declaration.source_wall_policy.spec_id
                != specs[declaration.part_name].spec_id
            ):
                raise ValueError(
                    "MeshingFieldDeclaration.source_wall_policy must match registration.specs; got differing policy"
                )
        for role in declaration.state_roles:
            value = arrays.get(role.entry_name)
            if (
                not isinstance(value, (jax.Array, np.ndarray))
                or value.shape != role.shape
                or value.dtype != role.dtype
            ):
                raise ValueError(
                    "MeshingFieldStateRole.array must match the declared shape and dtype; got incompatible array"
                )
            if role.entry_name in declared_entries:
                raise ValueError(
                    "MeshingFieldStateRole.entry_name must have one field owner; got duplicate array ownership"
                )
            declared_entries.add(role.entry_name)
            if role.role == "state-epoch" and (
                value.shape != () or int(np.asarray(value)) != role.state_epoch
            ):
                raise ValueError(
                    "MeshingFieldStateRole.state_epoch must match the accepted state epoch array; got differing epoch"
                )
            stream = role.random_stream
            if role.role == "rng-key" and stream is not None:
                stream.root_key(value)
    if require_complete and set(arrays) != declared_entries:
        raise ValueError(
            "MeshingFieldDeclaration.state_roles must cover every accepted array; got undeclared array"
        )


def meshing_field_artifact_types() -> tuple[tuple[str, tuple[type, ...]], ...]:
    """Return native field artifact admission groups."""
    return (
        (
            "phydrax.lifecycle",
            (
                MeshingAcceptedEpoch,
                MeshingGeometryMotionTransition,
                MeshingFiniteElementFieldDeclaration,
                MeshingFiniteVolumeFieldDeclaration,
                MeshingPolyhedralFieldDeclaration,
                MeshingPolyhedralOwnerBinding,
                MeshingPolyhedralFieldIndexBinding,
                MeshingPolyhedralFieldStateRole,
                MeshingFieldStateRole,
                MeshingFieldIndexBinding,
                MeshingRandomStream,
                MeshingCoefficientIdentityBank,
            ),
        ),
        (
            "phydrax.discretization",
            (
                FiniteElementFieldSpec,
                FiniteElementPrecisionPolicy,
                FiniteElementTabulator,
                FormBasis,
                ProxyTabulator,
                TensorFactors,
                HybridFactors,
                ReferenceNodalFamily,
                SimplexNodalFamily,
                HybridReferenceFamily,
                CellPolynomialReconstructionPlan,
                PiecewiseConstantReconstruction,
                UnstructuredWENOZReconstructionPlan,
                FieldTracePolicy,
                RusanovFluxPlan,
                ExtrapolationBoundary,
                SlipWallBoundary,
                GlobalCoefficientId,
            ),
        ),
        ("phydrax.equations", (EulerSystem, IdealGasMaterial)),
        ("phydrax.exterior", (FormType, FormValueSpec)),
        ("phydrax.core", (BranchDifferentiationPolicy,)),
    )


def _rebuild_finite_element(value: FiniteElementSpec, /) -> FiniteElementSpec | None:
    tabulator = value.tabulator
    if value.form_basis is not None:
        if (
            type(value.form_basis) is not FormBasis
            or type(tabulator) is not ProxyTabulator
        ):
            raise TypeError(
                "FiniteElementSpec.form_basis must pair a form basis with its proxy tabulator; got invalid pair"
            )
        basis = value.form_basis
        return form_element(
            value.cell_kind,
            basis.form_degree,
            basis.order,
            family=basis.family,
            twist=basis.twist,
            proxy=value.value_spec.proxy,
        )
    match tabulator:
        case ProxyTabulator():
            raise TypeError(
                "FiniteElementSpec.form_basis must declare the proxy basis; got missing basis"
            )
        case FiniteElementTabulator():
            match tabulator.source:
                case FiniteElementSpec() as base:
                    if value.family not in (
                        "DiscontinuousLagrange",
                        "DiscontinuousTensorProductLagrange",
                    ):
                        raise ValueError(
                            f"FiniteElementSpec.family must retain the discontinuous scalar family; got {value.family!r}"
                        )
                    if base.value_shape or base.mapping != "identity":
                        raise ValueError(
                            "FiniteElementSpec.tabulator.source must be identity-mapped scalar; got incompatible basis"
                        )
                    entities: list[tuple[tuple[int, ...], ...]] = [
                        tuple(() for _ in dimension) for dimension in base.entity_dofs
                    ]
                    entities[-1] = (tuple(range(base.local_dof_count)),)
                    return FiniteElementSpec(
                        value.family,
                        base.cell_kind,
                        base.degree,
                        base.reference_nodes,
                        tuple(entities),
                        value_spec=base.value_spec,
                        continuity="discontinuous",
                        representation=base.representation,
                        tabulator=FiniteElementTabulator(base, "tabulate"),
                        tabulator_id=f"discontinuous:{base.element_id}",
                    )
                case AbstractStoredFiniteElementTabulator() as source:
                    return source.finite_element_from_stored_fields()
                case _:
                    raise TypeError(
                        "FiniteElementSpec.tabulator.source must support stored element reconstruction; got unsupported source"
                    )
        case None:
            match value.family:
                case "Lagrange":
                    return lagrange_element(value.cell_kind, value.degree)
                case "DiscontinuousLagrange":
                    return discontinuous_element(value.cell_kind, value.degree)
                case _:
                    return None
        case _:
            return None


def _rebuild_field_spec(value: FiniteElementFieldSpec, /) -> FiniteElementFieldSpec:
    if not value.block_names:
        return FiniteElementFieldSpec(
            value.name, value.elements[0], component_shape=value.component_shape
        )
    if all(
        element.element_id == value.elements[0].element_id for element in value.elements
    ):
        return FiniteElementFieldSpec(
            value.name,
            value.elements[0],
            block_names=value.block_names,
            component_shape=value.component_shape,
        )
    return FiniteElementFieldSpec(
        value.name,
        dict(zip(value.block_names, value.elements, strict=True)),
        component_shape=value.component_shape,
    )


def _rebuild_form_proxy(value: ProxyTabulator, /) -> ProxyTabulator:
    if type(value.basis) is not FormBasis or type(value.value_spec) is not FormValueSpec:
        raise TypeError(
            "ProxyTabulator.basis must pair an immutable form basis with its value_spec; got invalid records"
        )
    form = value.value_spec.form_type
    if (
        (form.dimension, form.degree, form.twist)
        != (value.basis.dimension, value.basis.form_degree, value.basis.twist)
        or form.ambient_dimension != form.dimension
        or form.fiber_shape
    ):
        raise ValueError(
            "ProxyTabulator.value_spec must match its basis form type; got incompatible form"
        )
    return ProxyTabulator(value.basis, value.value_spec)


def rebuild_meshing_field_record(
    value: Any, /, *, budget: CoordinateEnclosureBudget | None = None
) -> Any | None:
    """Reconstruct field inputs through their validation boundaries."""
    match value:
        case MeshingCoefficientIdentityBank():
            return MeshingCoefficientIdentityBank.from_stored_fields(value)
        case MeshingGeometryMotionTransition():
            return MeshingGeometryMotionTransition.from_stored_fields(
                value, budget=budget
            )
        case MeshingAcceptedEpoch():
            return MeshingAcceptedEpoch.from_stored_fields(value)
        case (
            MeshingFiniteElementFieldDeclaration()
            | MeshingFiniteVolumeFieldDeclaration()
            | MeshingPolyhedralFieldDeclaration()
        ):
            return type(value).from_stored_fields(value)
        case MeshingPolyhedralOwnerBinding():
            return MeshingPolyhedralOwnerBinding.from_stored_fields(value, budget=budget)
        case MeshingPolyhedralFieldIndexBinding():
            return MeshingPolyhedralFieldIndexBinding.from_stored_fields(value)
        case MeshingFieldStateRole() | MeshingFieldIndexBinding() | MeshingRandomStream():
            return type(value).from_stored_fields(value)
        case GlobalCoefficientId():
            return GlobalCoefficientId(
                value.field_space_id, value.ordinal, value.component
            )
        case DimensionSignature():
            return DimensionSignature.from_stored_fields(value)
        case UnitDefinition():
            return UnitDefinition.from_stored_fields(value)
        case EulerSystem():
            return EulerSystem(value.dimension, material=value.material)
        case IdealGasMaterial():
            return IdealGasMaterial(
                value.gamma,
                value.gas_constant,
                density_floor=value.density_floor,
                pressure_floor=value.pressure_floor,
            )
        case RusanovFluxPlan():
            return RusanovFluxPlan(smooth_epsilon=value.smooth_epsilon)
        case ExtrapolationBoundary():
            return ExtrapolationBoundary()
        case SlipWallBoundary():
            return SlipWallBoundary()
        case FiniteElementFieldSpec():
            return _rebuild_field_spec(value)
        case FiniteElementSpec():
            return _rebuild_finite_element(value)
        case ReferenceNodalFamily():
            return ReferenceNodalFamily(
                value.cell_kind, value.orders, node_set=value.node_set
            )
        case SimplexNodalFamily():
            return SimplexNodalFamily(value.cell_kind, value.order)
        case HybridReferenceFamily():
            return HybridReferenceFamily(
                value.cell_kind,
                value.orders if value.cell_kind == "prism" else value.degree,
            )
        case FiniteElementTabulator():
            return FiniteElementTabulator(value.source, value.method)
        case FormBasis():
            return FormBasis(
                value.dimension,
                value.form_degree,
                value.order,
                value.family,
                value.twist,
                budget=budget,
            )
        case ProxyTabulator():
            return _rebuild_form_proxy(value)
        case FormType():
            return FormType(
                value.dimension,
                value.degree,
                twist=value.twist,
                fiber_shape=value.fiber_shape,
                ambient_dimension=value.ambient_dimension,
            )
        case FormValueSpec():
            return FormValueSpec(value.form_type, proxy=value.proxy)
        case PiecewiseConstantReconstruction():
            return PiecewiseConstantReconstruction()
        case FiniteElementPrecisionPolicy():
            return FiniteElementPrecisionPolicy(
                storage_dtype=value.storage_dtype,
                geometry_dtype=value.geometry_dtype,
                evaluation_dtype=value.evaluation_dtype,
                accumulation_dtype=value.accumulation_dtype,
                output_dtype=value.output_dtype,
                compensated_accumulation=value.compensated_accumulation,
            )
        case CellPolynomialReconstructionPlan():
            return CellPolynomialReconstructionPlan(
                value.degree,
                weight_power=value.weight_power,
                oversampling=value.oversampling,
                rcond=value.rcond,
                condition_limit=value.condition_limit,
            )
        case UnstructuredWENOZReconstructionPlan():
            return UnstructuredWENOZReconstructionPlan(
                value.degree,
                weight_power=value.weight_power,
                oversampling=value.oversampling,
                linear_weights=value.linear_weights,
                epsilon=value.epsilon,
                power=value.power,
                limiter=value.limiter,
            )
        case FieldTracePolicy():
            return FieldTracePolicy(value.kind, sides=value.sides)
        case _:
            return None


type MeshingFieldReconstruction = (
    PiecewiseConstantReconstruction
    | PreparedCellPolynomialReconstruction
    | PreparedUnstructuredWENOZReconstruction
)


class _PreparedFieldOwner(StrictModule):
    owner: UnstructuredFiniteVolumeDiscretization
    reconstruction: MeshingFieldReconstruction


def _prepare_finite_volume_field(
    declaration: MeshingFiniteVolumeFieldDeclaration,
    carrier: Any,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> _PreparedFieldOwner:
    mesh, geometry = carrier.mesh, carrier.geometry
    if declaration.finite_volume_coordinate_policy == "represented-polyhedral":
        if type(mesh.connectivity) is not PolyhedralConnectivity:
            raise ValueError(
                "MeshingFiniteVolumeFieldDeclaration.finite_volume_coordinate_policy must bind polyhedral incidence; got incompatible connectivity"
            )
    periodic = mesh.periodic_topology is not None
    periodic_embedding = None
    if periodic:
        report = carrier.certification
        if report is None or report.embedding is None:
            raise ValueError(
                "MeshingFiniteVolumeFieldDeclaration.periodic_geometry must retain "
                "the accepted carrier's embedding certificate"
            )
        periodic_embedding = report.embedding
    prepared = UnstructuredFiniteVolumePlan.from_cell_mesh(
        mesh,
        field_name=declaration.finite_volume_field_name,
        component_names=declaration.finite_volume_component_names,
        periodic_geometry=geometry if periodic else None,
        periodic_embedding=periodic_embedding,
        budget=budget,
        **declaration.finite_volume_options,
    ).prepare(
        numeric_version=declaration.numeric_version,
        owned_face_capacity=declaration.finite_volume_owned_face_capacity,
        cell_geometry=geometry
        if declaration.finite_volume_coordinate_policy == "accepted-map"
        else None,
        periodic_embedding=periodic_embedding,
        budget=budget,
    )
    match declaration.reconstruction_policy:
        case PiecewiseConstantReconstruction() as policy:
            reconstruction = policy
        case CellPolynomialReconstructionPlan() as policy:
            reconstruction = policy.prepare(
                prepared, **declaration.reconstruction_options
            )
        case UnstructuredWENOZReconstructionPlan() as policy:
            reconstruction = policy.prepare(prepared)
        case _:
            raise TypeError(
                "MeshingFiniteVolumeFieldDeclaration.reconstruction_policy must be a native reconstruction owner; got unsupported policy"
            )
    return _PreparedFieldOwner(prepared, reconstruction)


type _PreparedMeshingFieldOwner = tuple[
    FiniteElementDiscretization
    | UnstructuredFiniteVolumeDiscretization
    | PolyhedralArchiveOwners,
    MeshingFieldReconstruction | None,
]


@checked
def prepare_meshing_field_owner(
    declaration: MeshingFieldDeclaration,
    carrier: Any,
    /,
    *,
    index_source: CellMeshingResult | None = None,
    budget: CoordinateEnclosureBudget | None = None,
    source_validation: Any | None = None,
) -> _PreparedMeshingFieldOwner:
    """Prepare physical fields from the accepted mesh and geometry."""
    _require_field_source_validation(source_validation)
    declaration.__post_init__()
    mesh, geometry = carrier.mesh, carrier.geometry
    if (
        mesh.topology_id != declaration.topology_id
        or geometry.geometry_layout_id != declaration.geometry_layout_id
    ):
        raise ValueError(
            "MeshingFieldDeclaration.topology_id must match the accepted carrier topology and geometry_layout_id; got foreign identities"
        )
    from . import _meshing_sources as source_owner

    scope = source_owner._source_owner_reconstruction_scope.get()
    if budget is None and scope is not None and scope.process_id == getpid():
        budget = scope.budget
    key = None
    if scope is not None and scope.process_id == getpid():
        if budget is not None:
            scope.request_budgets[id(budget)] = budget
        source_owner.register_meshing_source_artifacts()
        key = source_owner._source_owner_reconstruction_key(
            (declaration, carrier, index_source, id(budget)),
            scope.archive_limits,
            source_validation,
            prepare_meshing_field_owner,
            scope,
        )
    retained = None if key is None or scope is None else scope.field_owners.get(key)
    prepared: (
        FiniteElementDiscretization
        | UnstructuredFiniteVolumeDiscretization
        | PolyhedralArchiveOwners
    )
    if retained is not None:
        prepared, reconstruction = retained
    else:
        match declaration:
            case MeshingFiniteElementFieldDeclaration():
                prepared = FiniteElementPlan(
                    mesh,
                    declaration.finite_element_fields,
                    coordinate_spec=geometry,
                    precision_policy=declaration.precision_policy,
                    budget=budget,
                ).prepare(numeric_version=declaration.numeric_version, budget=budget)
                reconstruction = None
            case MeshingFiniteVolumeFieldDeclaration():
                fv = _prepare_finite_volume_field(declaration, carrier, budget=budget)
                prepared, reconstruction = fv.owner, fv.reconstruction
            case MeshingPolyhedralFieldDeclaration():
                expected, prepared = prepare_meshing_polyhedral_owner_binding(
                    carrier,
                    budget=budget,
                    vem_field_name=declaration.owner_binding.field_names["vem"],
                    source_validation=source_validation,
                )
                equal = content_equal if source_validation is None else source_validation.equivalent
                if not equal(
                    declaration.owner_binding,
                    expected,
                    limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
                ):
                    raise ValueError(
                        "MeshingPolyhedralFieldDeclaration.owner_binding must match independently prepared actual plans, coefficient identities and native index banks; got changed scientific ownership"
                    )
                reconstruction = None
            case _:
                raise TypeError(
                    "MeshingFieldDeclaration.owner must select finite-element or finite-volume preparation; got unsupported declaration"
                )
    if isinstance(prepared, PolyhedralArchiveOwners):
        if type(declaration) is not MeshingPolyhedralFieldDeclaration:
            raise TypeError(
                "MeshingPolyhedralFieldDeclaration.preparation must retain its concrete declaration; got a foreign owner"
            )
        if key is not None and scope is not None:
            if budget is not None:
                budget.retain_owner(prepared)
            scope.constructors[id(prepare_meshing_field_owner)] = (
                prepare_meshing_field_owner
            )
            scope.field_owners[key] = (prepared, reconstruction)
        return prepared, reconstruction
    match prepared:
        case FiniteElementDiscretization():
            physical_spaces = prepared.field_spaces
        case UnstructuredFiniteVolumeDiscretization():
            physical_spaces = (prepared.cell_space,)
        case _:
            assert_never(prepared)
    actual = {space.name: space.field_space_id for space in physical_spaces}
    if actual != dict(declaration.field_space_ids):
        raise ValueError(
            "MeshingFieldDeclaration.field_space_ids must match freshly prepared field layouts; got differing identities"
        )
    spaces = {space.name: space for space in physical_spaces}
    for role in declaration.state_roles:
        binding = role.index_binding
        if binding is not None and binding.bank_name.startswith("epoch/"):
            validate_meshing_field_index_binding(
                role,
                carrier if index_source is None else index_source,
                source_validation=source_validation,
            )
        else:
            validate_meshing_field_index_binding(
                role,
                carrier,
                field_space=None
                if binding is None or binding.field_space_id is None
                else spaces[role.field_name],
                source_validation=source_validation,
            )
        if role.role in ("coefficients", "boundary-values"):
            vector_space = spaces[role.field_name].vector_space
            if not isinstance(vector_space, ArraySpace):
                raise TypeError(
                    "MeshingFieldStateRole.field_space.vector_space must be an ArraySpace; got incompatible space"
                )
            expected = vector_space.shape
            if role.shape != expected:
                raise ValueError(
                    "MeshingFieldStateRole.shape must match the prepared coefficient layout; got incompatible shape"
                )
        elif role.role == "boundary-mask":
            if not isinstance(prepared, FiniteElementDiscretization):
                raise ValueError(
                    "MeshingFieldStateRole.role must use FV face policies for FV boundaries; got FE boundary mask"
                )
            dof_map = prepared.dof_maps[prepared.field_index(role.field_name)]
            if role.shape != (dof_map.global_dof_count,):
                raise ValueError(
                    "MeshingFieldStateRole.shape must match FE boundary coefficient ownership; got incompatible mask"
                )
    if key is not None and scope is not None:
        scope.constructors[id(prepare_meshing_field_owner)] = prepare_meshing_field_owner
        scope.field_owners[key] = (prepared, reconstruction)
    return prepared, reconstruction


def _prepare_field_dynamics(
    declaration: MeshingFiniteVolumeFieldDeclaration,
    prepared: Any,
    reconstruction: Any,
    /,
) -> PreparedUnstructuredFiniteVolumeDynamics:

    flux_policy = declaration.finite_volume_flux_policy
    if declaration.finite_volume_system is None or flux_policy is None:
        raise ValueError(
            "MeshingFiniteVolumeFieldDeclaration.finite_volume_system must declare dynamics with a flux policy; got missing system or flux"
        )
    boundaries = UnstructuredFiniteVolumeBoundarySet(
        prepared.boundary_patch_names,
        declaration.finite_volume_boundaries,
    )
    return PreparedUnstructuredFiniteVolumeDynamics(
        declaration.finite_volume_system,
        prepared,
        UnstructuredFiniteVolumeMethodPlan(reconstruction, flux_policy),
        boundaries,
    )


@checked
def prepare_meshing_field_dynamics(
    declaration: MeshingFiniteVolumeFieldDeclaration,
    carrier: Any,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> PreparedUnstructuredFiniteVolumeDynamics:
    """Prepare FV dynamics from stored system, flux, and wall policies."""
    prepared, reconstruction = prepare_meshing_field_owner(
        declaration, carrier, budget=budget
    )
    return _prepare_field_dynamics(declaration, prepared, reconstruction)


class _FieldCarriers(StrictModule):
    carriers: frozendict[str, Any]
    index_source: CellMeshingResult | None
    registration: OversetRegistration | None


def _collective_field_carriers(
    target: CellMeshingResult,
    generation: MeshPart | None,
    declarations: Mapping[str, Any],
    /,
) -> _FieldCarriers:
    if target.mesh.storage is None:
        raise TypeError(
            "field_declarations.collective_target must retain owner-local storage; got dense result"
        )
    original = require_original_meshing_source(target)
    if generation is None:
        if len(declarations) != 1:
            raise ValueError(
                f"MeshingSourceClosure.field_declarations must have one owner for a collective result; got {len(declarations)}"
            )
        return _FieldCarriers(
            frozendict({next(iter(declarations)): target}), target, None
        )
    match generation.carrier:
        case CellMeshingResult() as generation_carrier:
            pass
        case _:
            raise TypeError(
                "field_declarations.generation_part.carrier must be a CellMeshingResult; got incompatible carrier"
            )
    match original:
        case CellMeshingResult():
            if original.result_id != generation_carrier.result_id:
                raise ValueError(
                    "field_declarations.generation_part must bind the collective source; got foreign generation"
                )
        case InitialCollectiveMeshEvidence():
            initial = generation_carrier.collective_evidence
            if (
                not isinstance(initial, InitialCollectiveMeshEvidence)
                or initial.evidence_id != original.evidence_id
                or generation_carrier.mesh.mesh_id != original.mesh_id
            ):
                raise ValueError(
                    "field_declarations.generation_part must bind initial collective evidence; got foreign evidence"
                )
        case _:
            raise TypeError(
                "field_declarations.collective_target must retain a native source; got unsupported source"
            )
    return _FieldCarriers(frozendict({generation.name: target}), target, None)


def _collect_field_carriers(
    records: Mapping[str, Any], declarations: Mapping[str, Any], /
) -> _FieldCarriers:
    registration = records.get("registration")
    if registration is not None:
        match registration:
            case OversetRegistration():
                return _FieldCarriers(
                    frozendict(
                        {part.name: part.carrier for part in registration.assembly.parts}
                    ),
                    None,
                    registration,
                )
            case _:
                raise TypeError(
                    "field_declarations.registration must be an OversetRegistration; got unsupported record"
                )
    target = records.get("collective_target")
    accepted_target = records.get("accepted_target")
    generation = records.get("generation_part")
    if generation is not None and type(generation) is not MeshPart:
        raise TypeError(
            "field_declarations.generation_part must be a MeshPart; got unsupported record"
        )
    if target is not None and accepted_target is not None:
        raise ValueError(
            "MeshingSourceClosure.field_declarations must select one accepted target; got serial and collective targets"
        )
    if target is not None:
        match target:
            case CellMeshingResult():
                return _collective_field_carriers(target, generation, declarations)
            case _:
                raise TypeError(
                    "field_declarations.collective_target must be a CellMeshingResult; got unsupported record"
                )
    if accepted_target is not None:
        if (
            type(accepted_target) is not CellMeshingResult
            or accepted_target.mesh.storage is not None
        ):
            raise TypeError(
                "field_declarations.accepted_target must be a dense CellMeshingResult; got incompatible target"
            )
        if generation is not None:
            raise ValueError(
                "field_declarations.generation_part must be absent for an adapted serial target; got duplicate generation"
            )
        if len(declarations) != 1:
            raise ValueError(
                f"MeshingSourceClosure.field_declarations must have one owner for a serial result; got {len(declarations)}"
            )
        return _FieldCarriers(
            frozendict({next(iter(declarations)): accepted_target}), accepted_target, None
        )
    if generation is not None:
        return _FieldCarriers(
            frozendict({generation.name: generation.carrier}), None, None
        )
    raise ValueError(
        "MeshingSourceClosure.field_declarations must bind an accepted result or registration; got no carrier"
    )


def validate_meshing_field_declarations(
    records: Mapping[str, Any], /, *, source_validation: Any | None = None
) -> None:
    """Validate accepted arrays against their declared carrier and field schemas."""
    _require_field_source_validation(source_validation)
    accepted = records.get("accepted_data")
    declarations = records.get("field_declarations")
    epochs = validate_meshing_accepted_epochs(
        records, source_validation=source_validation
    )
    if epochs:
        if declarations is not None or (
            isinstance(accepted, Mapping) and "fields" in accepted
        ):
            raise ValueError(
                "MeshingSourceClosure.field_declarations must be absent when accepted_epochs retain declarations and banks; got duplicate final-field owner"
            )
        return
    if declarations is None:
        if isinstance(accepted, Mapping) and "fields" in accepted:
            raise ValueError(
                "MeshingSourceClosure.field_declarations must bind accepted physical fields; got missing declarations"
            )
        return

    if not isinstance(declarations, Mapping):
        raise TypeError(
            "MeshingSourceClosure.field_declarations must be a mapping of named declarations; got incompatible container"
        )
    collected = _collect_field_carriers(records, declarations)
    carriers = collected.carriers
    if set(declarations) != set(carriers):
        raise ValueError(
            "MeshingSourceClosure.field_declarations must name each accepted carrier once; got differing carrier keys"
        )
    if not isinstance(accepted, Mapping):
        raise ValueError(
            "field_declarations.accepted_data must retain state arrays; got missing or incompatible data"
        )
    arrays = accepted.get("fields", accepted)
    if not isinstance(arrays, Mapping):
        raise ValueError(
            "field_declarations.accepted_data.fields must be a mapping of arrays; got incompatible container"
        )
    owners: list[tuple[MeshingFieldDeclaration, Any]] = []
    for name, declaration in declarations.items():
        if (
            type(declaration)
            not in (
                MeshingFiniteElementFieldDeclaration,
                MeshingFiniteVolumeFieldDeclaration,
                MeshingPolyhedralFieldDeclaration,
            )
            or declaration.part_name != name
        ):
            raise ValueError(
                "MeshingSourceClosure.field_declarations must map part names to matching concrete declarations; got incompatible record"
            )
        owners.append((declaration, carriers[name]))
    _validate_field_owner_banks(
        tuple(owners),
        arrays,
        index_source=collected.index_source,
        registration=collected.registration,
        require_complete="fields" in accepted,
        source_validation=source_validation,
    )
