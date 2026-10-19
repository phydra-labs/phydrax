#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Declare physical field semantics for native restart recipes.

Only primitive declarations and accepted arrays cross this boundary. Preparation
is performed by the existing FE/FV owners against the restored accepted carrier.
"""

from __future__ import annotations

import dataclasses
import json
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import product
from math import prod
from os import getpid
from typing import Any, assert_never, cast, final, Literal, Self

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

from .._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS, validate_json_nesting
from .._differentiation import BranchDifferentiationPolicy
from .._fingerprint import canonical_fingerprint, logical_array_value_collection_digest
from .._frozendict import frozendict
from .._model._artifacts import ArtifactValueCodec, register_artifact_value_codec
from .._model._structure import model_recipe_array_values, model_structure_recipe
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import AdaptiveSimplexParts
from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import CoordinateTabulator
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._conservation_boundary import ExtrapolationBoundary
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
from ..meshing._entity_keys import key_rows
from ..meshing._initial_certification import InitialCollectiveMeshEvidence
from ..meshing._mesh_part import MeshPart
from ..meshing._overset import OversetPartSpec, OversetRegistration
from ..meshing._restart_distribution import (
    SimplexRestartRepack,
    SimplexRestartRepackProof,
    validate_simplex_restart_repack,
)
from ..meshing._result import CellMeshingResult, require_original_meshing_source
from ..meshing._scope import MeshingEntityKind, MeshingScope
from ..sparse._ops import gather_routes
from ..sparse._relation import RowRelation
from ..typing import checked, parse
from ..units._dimension import DimensionSignature
from ..units._unit import UnitDefinition
from ._meshing_source_records import content_equal, register_meshing_source_types


@dataclass(frozen=True, slots=True)
class MeshingCoefficientIdentityBank:
    """Array-free scientific identities, retaining every explicit ordered row."""

    field_space_id: str
    rows: tuple[tuple[int, tuple[int, ...], str], ...]

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[Self], value: MeshingCoefficientIdentityBank, /
    ) -> Self:
        """Reconstruct and validate the stored coefficient identities."""
        return cls(value.field_space_id, value.rows)

    def __post_init__(self) -> None:
        if (
            type(self.field_space_id) is not str
            or not self.field_space_id
            or self.field_space_id != self.field_space_id.strip()
        ):
            raise ValueError(
                "MeshingCoefficientIdentityBank.field_space_id must be a nonempty trimmed identity; got invalid identity"
            )
        if type(self.rows) is not tuple:
            raise TypeError(
                "MeshingCoefficientIdentityBank.rows must be an immutable tuple; got mutable metadata"
            )
        coordinates, identifiers = set(), set()
        previous = None
        for row in self.rows:
            if (
                type(row) is not tuple
                or len(row) != 3
                or type(row[0]) is not int
                or row[0] < 0
                or type(row[1]) is not tuple
                or any(type(value) is not int or value < 0 for value in row[1])
                or type(row[2]) is not str
                or not row[2]
                or row[2] != row[2].strip()
            ):
                raise ValueError(
                    "MeshingCoefficientIdentityBank.rows must contain nonnegative ordinals/components and trimmed IDs; got malformed row"
                )
            coordinate = row[:2]
            if (
                coordinate in coordinates
                or row[2] in identifiers
                or (previous is not None and coordinate <= previous)
            ):
                raise ValueError(
                    "MeshingCoefficientIdentityBank.rows must have unique ordered coordinates and IDs; got duplicate or unordered row"
                )
            record = GlobalCoefficientId(
                self.field_space_id, row[0], row[1], coefficient_id=row[2]
            )
            if (
                record.field_space_id,
                record.ordinal,
                record.component,
                record.coefficient_id,
            ) != (self.field_space_id, *row):
                raise ValueError(
                    "MeshingCoefficientIdentityBank.rows must preserve coefficient identity inputs; got changed constructor values"
                )
            coordinates.add(coordinate)
            identifiers.add(row[2])
            previous = coordinate

    @checked
    def require_field_space(self, space: DiscreteFieldSpace, /) -> None:
        self.__post_init__()
        expected = tuple(
            (value.ordinal, value.component, value.coefficient_id)
            for value in _coefficient_id_records(space)
        )
        if self.field_space_id != space.field_space_id or self.rows != expected:
            raise ValueError(
                "MeshingCoefficientIdentityBank.rows must match the field space in order; got missing, reordered, or foreign identities"
            )


def _encode_coefficient_identity_bank(
    bank: MeshingCoefficientIdentityBank,
) -> Mapping[str, Any]:

    bank.__post_init__()
    return {
        "field_space_id": bank.field_space_id,
        "rows_json": json.dumps(
            bank.rows, ensure_ascii=True, allow_nan=False, separators=(",", ":")
        ),
    }


def _decode_coefficient_identity_bank(
    payload: Mapping[str, Any],
) -> MeshingCoefficientIdentityBank:

    if set(payload) != {"field_space_id", "rows_json"} or any(
        type(payload[name]) is not str for name in payload
    ):
        raise ValueError(
            "MeshingCoefficientIdentityBank.payload must contain field_space_id and rows_json strings; got invalid fields"
        )
    text = payload["rows_json"]
    if len(text.encode("utf-8")) > DEFAULT_ARRAY_ARCHIVE_LIMITS.max_manifest_bytes:
        raise ValueError(
            "MeshingCoefficientIdentityBank.rows_json must fit the manifest byte limit; got oversized wire data"
        )
    validate_json_nesting(text, DEFAULT_ARRAY_ARCHIVE_LIMITS.max_manifest_nesting)
    values = json.loads(text)
    if type(values) is not list or any(
        type(row) is not list or len(row) != 3 or type(row[1]) is not list
        for row in values
    ):
        raise ValueError(
            "MeshingCoefficientIdentityBank.rows_json must encode ordinal/component/ID rows; got malformed wire records"
        )
    bank = MeshingCoefficientIdentityBank(
        payload["field_space_id"],
        tuple((row[0], tuple(row[1]), row[2]) for row in values),
    )
    if _encode_coefficient_identity_bank(bank) != dict(payload):
        raise ValueError(
            "MeshingCoefficientIdentityBank.payload must use deterministic encoding; got a differing re-encoding"
        )
    return bank


def _register_coefficient_identity_bank_codec() -> None:

    register_artifact_value_codec(
        ArtifactValueCodec(
            "phydrax.lifecycle:MeshingCoefficientIdentityBank",
            MeshingCoefficientIdentityBank,
            _encode_coefficient_identity_bank,
            _decode_coefficient_identity_bank,
        )
    )


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


def _coefficient_id_records(
    space: DiscreteFieldSpace, /
) -> tuple[GlobalCoefficientId, ...]:
    if not isinstance(space.vector_space, ArraySpace):
        raise TypeError(
            "DiscreteFieldSpace.vector_space must be an ArraySpace for coefficient rows; got unsupported space"
        )
    shape = space.vector_space.shape
    if not shape:
        raise ValueError(
            "DiscreteFieldSpace.vector_space.shape must have a coefficient-row axis; got scalar space"
        )
    components = tuple(product(*(range(size) for size in shape[1:])))
    return tuple(
        GlobalCoefficientId(space.field_space_id, ordinal, component)
        for ordinal in range(shape[0])
        for component in components
    )


@checked
def prepare_meshing_field_index_binding(
    carrier: CellMeshingResult,
    bank_name: MeshingFieldIndexBank,
    /,
    *,
    field_space: DiscreteFieldSpace | None = None,
) -> MeshingFieldIndexBinding:
    """Bind rows to the source bank by scientific row identity."""
    bank_name = parse(
        bank_name, MeshingFieldIndexBank, "prepare_meshing_field_index_binding.bank_name"
    )
    mesh = carrier.mesh
    evidence = carrier.collective_evidence
    evidence_id = None if evidence is None else evidence.evidence_id
    clocks = (
        None if evidence is None else dict(evidence.logical_arrays).get("epoch/clocks")
    )
    epoch_id = (
        None
        if clocks is None
        else logical_array_value_collection_digest({"epoch/clocks": clocks})
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
        else MeshingCoefficientIdentityBank(
            field_space.field_space_id,
            tuple(
                (value.ordinal, value.component, value.coefficient_id)
                for value in _coefficient_id_records(field_space)
            ),
        ),
    )


def _same_index_bank(
    first: jax.Array | np.ndarray, second: jax.Array | np.ndarray, /
) -> bool:
    if first.shape != second.shape or first.dtype != second.dtype:
        return False
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
) -> None:
    """Prove bank identity against real entity/DOF/raw-source arrays."""
    role.__post_init__()
    if role.role == "state-epoch":
        return
    binding = role.index_binding
    if binding is None:
        raise ValueError(
            "MeshingFieldStateRole.index_binding must declare row identities; got missing binding"
        )
    expected = prepare_meshing_field_index_binding(
        carrier, binding.bank_name, field_space=field_space
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
    if not _same_index_bank(binding.index_values, expected.index_values):
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
        or not _same_index_bank(binding.scope.entity_ids, expected.scope.entity_ids)
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
) -> jax.Array | np.ndarray:
    """Join compact transferred values to raw rows by scientific IDs.

    Only active raw rows are updated. Inactive and retired rows retain the
    supplied indexed baseline. Missing active IDs refuse. Local shards are
    processed without gathering the global bank.
    """

    source_role.__post_init__()
    raw_role.__post_init__()
    compact, raw = source_role.index_binding, raw_role.index_binding
    if (
        compact is None
        or raw is None
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
    if raw_role.role not in ("stage-history", "material-history"):
        raise ValueError(
            "project_meshing_field_history.raw_role.role must be a history role; got coefficient or unsupported role"
        )
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
    ids = np.asarray(compact.index_values)
    if np.any(ids < 0) or np.unique(ids).size != ids.size:
        raise ValueError(
            "project_meshing_field_history.compact_ids must be distinct nonnegative identities; got invalid or duplicate IDs"
        )

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
            gather_routes(relation, jnp.asarray(source_values)), axis=rows.ndim
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


def _require_history_repack(
    receipt: SimplexRestartRepack,
    proof: SimplexRestartRepackProof | None,
) -> None:
    if proof is None:
        validate_simplex_restart_repack(receipt, receipt.requested_target_owners)
    else:
        proof.require_receipt(receipt)


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
    if role.role not in ("material-history", "stage-history") or binding is None:
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
        raw is None
        or compact is None
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

    parts = receipt.execution_parts()
    shards = values.addressable_shards
    if len(shards) != 1 or queries.ndim != 1 or queries.dtype != np.int64:
        raise ValueError(
            "recover_meshing_field_history.queries must be int64 compact IDs for one addressable raw part; got incompatible workset"
        )
    capacity = int(
        np.max(multihost_utils.process_allgather(np.int64(max(queries.shape[0], 1))))
    )
    packet = np.full((1, capacity), -1, dtype=np.int64)
    packet[0, : queries.shape[0]] = queries
    placement = NamedSharding(parts.mesh, PartitionSpec(parts.axis_name))
    distributed = jax.make_array_from_process_local_data(
        placement, packet, (parts.part_count, capacity)
    )
    answer, found = _ring_recovery(parts, identifiers, owners, values, distributed)
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
]


@dataclass(frozen=True)
class MeshingFieldStateRole:
    """Bind an accepted-array role to a field, schema, and state epoch."""

    entry_name: str
    field_name: str
    role: MeshingFieldStateKind
    shape: tuple[int, ...]
    dtype: np.dtype
    state_epoch: int
    value_units: tuple[UnitDefinition, ...] | None = None
    index_binding: MeshingFieldIndexBinding | None = None

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
        if self.role != "state-epoch":
            if type(self.index_binding) is not MeshingFieldIndexBinding:
                raise ValueError(
                    "MeshingFieldStateRole.index_binding must declare indexed values or history; got missing binding"
                )
            prefix = tuple(
                self.shape[axis]
                for axis in self.index_binding.index_axes
                if axis < len(self.shape)
            )
            if prefix != self.index_binding.index_values.shape:
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
        elif self.index_binding is not None:
            raise ValueError(
                "MeshingFieldStateRole.index_binding must be absent for state-epoch; got indexed scalar"
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
    def owner(self) -> Literal["finite_element", "finite_volume"]:
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
    _validate_declaration_history(declaration, roles)


def _validate_declaration_history(
    declaration: MeshingFieldDeclaration, roles: tuple[MeshingFieldStateRole, ...], /
) -> None:
    histories = [
        role
        for role in roles
        if role.role in ("material-history", "stage-history", "state-epoch")
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


class MeshingAcceptedEpoch(StrictModule, NonTrainableState):
    """Retain an accepted carrier, physical declarations, and state arrays."""

    carrier: CellMeshingResult
    declarations: tuple[MeshingFieldDeclaration, ...]
    fields: frozendict[str, jax.Array | np.ndarray]
    transition: MeshAdaptationResult | None
    epoch_id: str = eqx.field(static=True)

    @classmethod
    @checked
    def from_stored_fields(cls: type[Self], value: MeshingAcceptedEpoch, /) -> Self:
        """Rebuild the accepted epoch and recompute its identity."""
        return cls(
            value.carrier, value.declarations, value.fields, transition=value.transition
        )

    @checked
    def __init__(
        self,
        carrier: CellMeshingResult,
        declarations: tuple[MeshingFieldDeclaration, ...],
        fields: Mapping[str, jax.Array | np.ndarray],
        /,
        *,
        transition: MeshAdaptationResult | None = None,
    ) -> None:
        if (
            type(declarations) is not tuple
            or not declarations
            or any(
                type(declaration)
                not in (
                    MeshingFiniteElementFieldDeclaration,
                    MeshingFiniteVolumeFieldDeclaration,
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
            if not content_equal(
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
        )
        self.epoch_id = self._identity()

    def _identity(self) -> str:
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


def validate_meshing_accepted_epochs(
    records: Mapping[str, Any], /
) -> tuple[MeshingAcceptedEpoch, ...]:
    """Authenticate an ordered field history against its generation and transitions."""

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
    if epochs_[0].transition is not None or not content_equal(
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
            or not content_equal(
                transition.source,
                previous.carrier,
                limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
            )
            or not content_equal(
                transition.target,
                epoch.carrier,
                limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
            )
        ):
            raise ValueError(
                "accepted_data.accepted_epochs.transition must bind preceding source and current target; got incompatible transition"
            )
    if type(target) is not CellMeshingResult or not content_equal(
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
) -> None:
    """Validate physical ownership and schemas of accepted state banks."""
    declared_entries: set[str] = set()
    for declaration, carrier in owners:
        prepared, reconstruction = prepare_meshing_field_owner(
            declaration, carrier, index_source=index_source
        )
        match declaration:
            case MeshingFiniteVolumeFieldDeclaration() if (
                declaration.finite_volume_system is not None
            ):
                _prepare_field_dynamics(declaration, prepared, reconstruction)
            case (
                MeshingFiniteElementFieldDeclaration()
                | MeshingFiniteVolumeFieldDeclaration()
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
    if require_complete and set(arrays) != declared_entries:
        raise ValueError(
            "MeshingFieldDeclaration.state_roles must cover every accepted array; got undeclared array"
        )


def meshing_field_artifact_types() -> tuple[tuple[str, tuple[type, ...]], ...]:
    """Return native field artifact admission groups."""
    _register_coefficient_identity_bank_codec()
    return (
        (
            "phydrax.lifecycle",
            (
                MeshingAcceptedEpoch,
                MeshingFiniteElementFieldDeclaration,
                MeshingFiniteVolumeFieldDeclaration,
                MeshingFieldStateRole,
                MeshingFieldIndexBinding,
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


def rebuild_meshing_field_record(value: Any, /) -> Any | None:
    """Reconstruct field inputs through their validation boundaries."""
    match value:
        case MeshingCoefficientIdentityBank():
            return MeshingCoefficientIdentityBank.from_stored_fields(value)
        case MeshingAcceptedEpoch():
            return MeshingAcceptedEpoch.from_stored_fields(value)
        case (
            MeshingFiniteElementFieldDeclaration() | MeshingFiniteVolumeFieldDeclaration()
        ):
            return type(value).from_stored_fields(value)
        case MeshingFieldStateRole() | MeshingFieldIndexBinding():
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
                value.dimension, value.form_degree, value.order, value.family, value.twist
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
    declaration: MeshingFiniteVolumeFieldDeclaration, carrier: Any, /
) -> _PreparedFieldOwner:
    mesh, geometry = carrier.mesh, carrier.geometry
    if declaration.finite_volume_coordinate_policy == "represented-polyhedral":
        if type(mesh.connectivity) is not PolyhedralConnectivity:
            raise ValueError(
                "MeshingFiniteVolumeFieldDeclaration.finite_volume_coordinate_policy must bind polyhedral incidence; got incompatible connectivity"
            )
    prepared = UnstructuredFiniteVolumePlan.from_cell_mesh(
        mesh,
        field_name=declaration.finite_volume_field_name,
        component_names=declaration.finite_volume_component_names,
        **declaration.finite_volume_options,
    ).prepare(
        numeric_version=declaration.numeric_version,
        owned_face_capacity=declaration.finite_volume_owned_face_capacity,
        cell_geometry=geometry
        if declaration.finite_volume_coordinate_policy == "accepted-map"
        else None,
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
    FiniteElementDiscretization | UnstructuredFiniteVolumeDiscretization,
    MeshingFieldReconstruction | None,
]


@checked
def prepare_meshing_field_owner(
    declaration: MeshingFieldDeclaration,
    carrier: Any,
    /,
    *,
    index_source: CellMeshingResult | None = None,
) -> _PreparedMeshingFieldOwner:
    """Prepare physical fields from the accepted mesh and geometry."""
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
    key = None
    if scope is not None and scope.process_id == getpid():
        source_owner.register_meshing_source_artifacts()
        key = source_owner._source_owner_reconstruction_key(
            (declaration, carrier, index_source),
            scope.archive_limits,
            None,
            prepare_meshing_field_owner,
            scope,
        )
    retained = None if key is None or scope is None else scope.field_owners.get(key)
    prepared: FiniteElementDiscretization | UnstructuredFiniteVolumeDiscretization
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
                ).prepare(numeric_version=declaration.numeric_version)
                reconstruction = None
            case MeshingFiniteVolumeFieldDeclaration():
                fv = _prepare_finite_volume_field(declaration, carrier)
                prepared, reconstruction = fv.owner, fv.reconstruction
            case _:
                raise TypeError(
                    "MeshingFieldDeclaration.owner must select finite-element or finite-volume preparation; got unsupported declaration"
                )
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
                role, carrier if index_source is None else index_source
            )
        else:
            validate_meshing_field_index_binding(
                role,
                carrier,
                field_space=None
                if binding is None or binding.field_space_id is None
                else spaces[role.field_name],
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
    declaration: MeshingFiniteVolumeFieldDeclaration, carrier: Any, /
) -> PreparedUnstructuredFiniteVolumeDynamics:
    """Prepare FV dynamics from stored system, flux, and wall policies."""
    prepared, reconstruction = prepare_meshing_field_owner(declaration, carrier)
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


def validate_meshing_field_declarations(records: Mapping[str, Any], /) -> None:
    """Validate accepted arrays against their declared carrier and field schemas."""
    accepted = records.get("accepted_data")
    declarations = records.get("field_declarations")
    epochs = validate_meshing_accepted_epochs(records)
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
    )
