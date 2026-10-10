#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Explicit physical field declarations for canonical native restart recipes.

Only primitive declarations and accepted arrays cross this boundary. Preparation
is performed by the existing FE/FV owners against the restored accepted carrier.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from itertools import product
from math import prod
from typing import Any, cast

import equinox as eqx
import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from .._differentiation import BranchDifferentiationPolicy
from .._fingerprint import canonical_fingerprint, logical_array_value_collection_digest
from .._frozendict import frozendict
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import AdaptiveSimplexParts
from ..discretization._conservation_boundary import ExtrapolationBoundary
from ..discretization._spaces import (
    DiscreteFieldSpace,
    EntityDofLayout,
    GlobalCoefficientId,
)
from ..discretization._views import FieldTracePolicy
from ..discretization.fem._form_elements import (
    _HybridFactors,
    _ProxyTabulator,
    _TensorFactors,
    form_element,
    FormBasis,
)
from ..discretization.fem._generic import (
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from ..discretization.fem._high_order import ReferenceNodalFamily, SimplexNodalFamily
from ..discretization.fem._precision import FiniteElementPrecisionPolicy
from ..discretization.fem._reference import _FiniteElementTabulator, FiniteElementSpec
from ..discretization.fem._spectral_hp_completion import HybridReferenceFamily
from ..discretization.finite_volume._cell_polynomial import (
    CellPolynomialReconstructionPlan,
)
from ..discretization.finite_volume._physical_boundaries import SlipWallBoundary
from ..discretization.finite_volume._reconstruction import PiecewiseConstantReconstruction
from ..discretization.finite_volume._riemann import RusanovFluxPlan
from ..discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from ..discretization.finite_volume._unstructured_weno import (
    UnstructuredWENOZReconstructionPlan,
)
from ..equations._hyperbolic_systems import EulerSystem
from ..equations._materials import IdealGasMaterial
from ..exterior._form_type import FormType, FormValueSpec
from ..linalg import ArraySpace
from ..meshing._adaptation import MeshAdaptationResult
from ..meshing._overset import OversetPartSpec, OversetRegistration
from ..meshing._restart_distribution import SimplexRestartRepack
from ..meshing._result import CellMeshingResult
from ..meshing._scope import MeshingEntityKind, MeshingScope
from ..units import DimensionSignature, UnitDefinition


@dataclass(frozen=True, slots=True)
class MeshingCoefficientIdentityBank:
    """Array-free scientific identities, retaining every explicit ordered row."""

    field_space_id: str
    rows: tuple[tuple[int, tuple[int, ...], str], ...]

    def __post_init__(self) -> None:
        if (
            type(self.field_space_id) is not str
            or not self.field_space_id
            or self.field_space_id != self.field_space_id.strip()
        ):
            raise ValueError(
                "Coefficient identities require their exact owning field-space identity."
            )
        if type(self.rows) is not tuple:
            raise TypeError(
                "Coefficient identity rows must be immutable ordered metadata."
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
                    "Coefficient identity rows require exact ordinals, components and stored IDs."
                )
            coordinate = row[:2]
            if (
                coordinate in coordinates
                or row[2] in identifiers
                or (previous is not None and coordinate <= previous)
            ):
                raise ValueError(
                    "Coefficient identities must retain unique rows and IDs in their actual canonical order."
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
                    "Coefficient identity metadata changed its actual constructor values."
                )
            coordinates.add(coordinate)
            identifiers.add(row[2])
            previous = coordinate

    def require_field_space(self, space: DiscreteFieldSpace, /) -> None:
        self.__post_init__()
        if type(space) is not DiscreteFieldSpace:
            raise TypeError(
                "Coefficient identity banks require their actual discrete field-space owner."
            )
        expected = tuple(
            (value.ordinal, value.component, value.coefficient_id)
            for value in _coefficient_id_records(space)
        )
        if self.field_space_id != space.field_space_id or self.rows != expected:
            raise ValueError(
                "Coefficient identities are missing, reordered, or foreign to their complete owning field space."
            )


def _encode_coefficient_identity_bank(
    bank: MeshingCoefficientIdentityBank,
) -> Mapping[str, Any]:
    import json

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
    import json

    from .._array_archive import _validate_json_nesting, DEFAULT_ARRAY_ARCHIVE_LIMITS

    if set(payload) != {"field_space_id", "rows_json"} or any(
        type(payload[name]) is not str for name in payload
    ):
        raise ValueError(
            "Coefficient identity wire data requires its exact registered field set."
        )
    text = payload["rows_json"]
    if len(text.encode("utf-8")) > DEFAULT_ARRAY_ARCHIVE_LIMITS.max_manifest_bytes:
        raise ValueError(
            "Coefficient identity wire data exceeds the original manifest byte limit."
        )
    _validate_json_nesting(text, DEFAULT_ARRAY_ARCHIVE_LIMITS.max_manifest_nesting)
    values = json.loads(text)
    if type(values) is not list or any(
        type(row) is not list or len(row) != 3 or type(row[1]) is not list
        for row in values
    ):
        raise ValueError(
            "Coefficient identity wire rows require exact ordinal/component/ID records."
        )
    bank = MeshingCoefficientIdentityBank(
        payload["field_space_id"],
        tuple((row[0], tuple(row[1]), row[2]) for row in values),
    )
    if _encode_coefficient_identity_bank(bank) != dict(payload):
        raise ValueError(
            "Coefficient identity wire data is not its exact canonical encoding."
        )
    return bank


def _register_coefficient_identity_bank_codec() -> None:
    from .._model._artifacts import ArtifactValueCodec, register_artifact_value_codec

    register_artifact_value_codec(
        ArtifactValueCodec(
            "phydrax.lifecycle:MeshingCoefficientIdentityBank",
            MeshingCoefficientIdentityBank,
            _encode_coefficient_identity_bank,
            _decode_coefficient_identity_bank,
        )
    )


@dataclass(frozen=True, kw_only=True)
class MeshingFieldIndexBinding:
    """Exact row meaning from a real field space, mesh entity set or raw epoch.

    Raw epoch banks include allocated retired records and explicit padding.
    They are not coefficient vectors. ``index_values`` retains the original
    bank order; a mesh scope describes membership, never a guessed row order.
    """

    bank_name: str
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

    def __post_init__(self) -> None:
        if self.bank_name not in (
            "field/global_coefficient_ids",
            "mesh/vertex_ids",
            "mesh/cell_ids",
            "epoch/vertex_ids",
            "epoch/cell_ids",
        ):
            raise ValueError(
                "Physical rows require an explicitly supported owning index bank."
            )
        if any(
            type(value) is not str or not value or value != value.strip()
            for value in (
                self.source_result_id,
                self.source_topology_id,
                self.source_geometry_id,
            )
        ):
            raise ValueError(
                "Physical row banks require original source/result/geometry identities."
            )
        if not isinstance(
            self.index_values, (jax.Array, np.ndarray)
        ) or self.index_values.dtype != np.dtype("int64"):
            raise TypeError("Physical row identity banks must retain exact int64 arrays.")
        raw = self.bank_name.startswith("epoch/")
        axes = (0, 1) if raw else (0,)
        if self.index_axes != axes or self.index_values.ndim != len(axes):
            raise ValueError(
                "Physical row axes must be the exact declared mesh or native raw-bank axes."
            )
        if raw and (
            not self.source_evidence_id
            or not self.source_epoch_id
            or self.scope is not None
        ):
            raise ValueError(
                "Allocated raw histories require actual native evidence and epoch authority."
            )
        if self.bank_name.startswith("mesh/") and type(self.scope) is not MeshingScope:
            raise TypeError("Mesh row banks require their real owning entity scope.")
        if (
            self.bank_name == "field/global_coefficient_ids"
            and self.field_space_id is None
        ):
            raise ValueError(
                "Coefficient ordinals require their actual owning field space."
            )
        if self.field_space_id is not None and (
            type(self.field_space_id) is not str or not self.field_space_id
        ):
            raise ValueError(
                "Coefficient bindings require a canonical field-space identity."
            )
        if self.field_space_id is None:
            if self.coefficient_bank is not None:
                raise ValueError(
                    "A history/entity bank cannot claim undeclared coefficient identities."
                )
        elif type(self.coefficient_bank) is not MeshingCoefficientIdentityBank:
            raise TypeError(
                "Coefficient identities require their exact immutable scientific bank."
            )
        else:
            self.coefficient_bank.__post_init__()
            if self.coefficient_bank.field_space_id != self.field_space_id:
                raise ValueError(
                    "Coefficient identity bank belongs to another actual field space."
                )


def _coefficient_id_records(
    space: DiscreteFieldSpace, /
) -> tuple[GlobalCoefficientId, ...]:
    if not isinstance(space.vector_space, ArraySpace):
        raise TypeError(
            "Physical coefficient banks require the existing explicit ArraySpace owner."
        )
    shape = space.vector_space.shape
    if not shape:
        raise ValueError("A physical field space must declare its coefficient-row axis.")
    components = tuple(product(*(range(size) for size in shape[1:])))
    return tuple(
        GlobalCoefficientId(space.field_space_id, ordinal, component)
        for ordinal in range(shape[0])
        for component in components
    )


def prepare_meshing_field_index_binding(
    carrier: CellMeshingResult,
    bank_name: str,
    /,
    *,
    field_space: DiscreteFieldSpace | None = None,
) -> MeshingFieldIndexBinding:
    """Bind rows to the actual source bank, not a payload's shape or row order."""
    if type(carrier) is not CellMeshingResult:
        raise TypeError(
            "Physical index binding requires the actual accepted CellMeshingResult."
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
    if bank_name == "field/global_coefficient_ids":
        if field_space is None or not isinstance(field_space.vector_space, ArraySpace):
            raise TypeError(
                "Coefficient-bank authoring requires its actual prepared DiscreteFieldSpace."
            )
        values = np.arange(field_space.vector_space.shape[0], dtype=np.int64)
        values.flags.writeable = False
    elif bank_name in ("mesh/vertex_ids", "mesh/cell_ids"):
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
                    "Entity-backed coefficients must retain their actual one-DOF entity layout."
                )
    elif bank_name in ("epoch/vertex_ids", "epoch/cell_ids"):
        if evidence is None:
            raise ValueError(
                "Allocated histories require their actual retained native collective source."
            )
        bank = dict(evidence.logical_arrays).get(bank_name)
        if bank is None or clocks is None:
            raise ValueError(
                "Native history authority lacks its exact raw index/clock bank."
            )
        if field_space is not None:
            raise ValueError(
                "An allocated raw epoch bank is not a prepared coefficient field space."
            )
        values = bank
    else:
        raise ValueError("Unknown physical source-index bank.")
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
        raise ValueError("Physical rows have no source-index authority.")
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
            "Physical row binding differs from its actual source/epoch/geometry/field-space authority."
        )
    if not _same_index_bank(binding.index_values, expected.index_values):
        raise ValueError(
            "Physical row identity/order differs from the actual source-index bank."
        )
    if binding.coefficient_bank != expected.coefficient_bank:
        raise ValueError(
            "Physical coefficients lost their complete original ordered field-space identities."
        )
    if expected.scope is None:
        if binding.scope is not None:
            raise ValueError(
                "Raw/DOF ordinals cannot claim an unrelated compact mesh scope."
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
        raise ValueError("Physical entity rows lost their exact owning mesh scope.")


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
    """Join compact transferred values to actual raw rows by scientific IDs.

    Only active raw rows are updated. Allocated inactive/retired/removal rows
    retain the supplied original indexed baseline. Missing active IDs refuse.
    Local raw shards are processed without gathering a complete global bank.
    """
    import jax.numpy as jnp

    from ..meshing._topology_edit import key_rows
    from ..sparse import gather_routes, RowRelation

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
            "History projection requires an explicit compact-entity to allocated-epoch bank pair."
        )
    if raw_role.role not in ("stage-history", "material-history"):
        raise ValueError(
            "Allocated projection targets must be actual history, not coefficient vectors."
        )
    if (
        source_role.field_name != raw_role.field_name
        or source_role.state_epoch != raw_role.state_epoch
        or source_role.value_units != raw_role.value_units
    ):
        raise ValueError(
            "History projection cannot relabel field, physical units or accepted state epoch."
        )
    validate_meshing_field_index_binding(source_role, carrier, field_space=field_space)
    raw_source = carrier if index_source is None else index_source
    validate_meshing_field_index_binding(raw_role, raw_source)
    if (
        source_values.shape != source_role.shape
        or source_values.dtype != source_role.dtype
    ):
        raise ValueError(
            "Transferred compact values differ from their original declared physical schema."
        )
    if (
        raw_baseline.shape != raw_role.shape
        or raw_baseline.dtype != raw_role.dtype
        or source_values.dtype != raw_role.dtype
    ):
        raise ValueError(
            "Raw history baseline must retain its actual payload schema and dtype."
        )
    if source_role.shape[1:] != raw_role.shape[2:]:
        raise ValueError(
            "Compact and raw history component axes must retain the same declared value semantics."
        )
    if (
        isinstance(compact.index_values, jax.Array)
        and not compact.index_values.is_fully_addressable
    ):
        raise ValueError(
            "Compact history projection requires an actual owner-local identity workset."
        )
    if isinstance(source_values, jax.Array) and not source_values.is_fully_addressable:
        raise ValueError(
            "Compact history values must belong to the actual addressable owner workset."
        )
    evidence = raw_source.collective_evidence
    if evidence is None:
        raise ValueError(
            "Raw history projection requires its actual native source evidence."
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
            "Native history projection lacks its actual raw active-row mask."
        )
    ids = np.asarray(compact.index_values)
    if np.any(ids < 0) or np.unique(ids).size != ids.size:
        raise ValueError(
            "Compact history IDs must be distinct actual scientific entities."
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
                "An active raw history row has no allocated scientific identity."
            )
        rows = key_rows(ids[:, None], queries.reshape(-1, 1)).reshape(queries.shape)
        if np.any(live & (rows < 0)):
            raise ValueError(
                "An active raw history ID is absent from its actual transferred compact field."
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
                "A host baseline cannot stand in for a nonaddressable raw bank."
            )
        if not active.is_fully_addressable:
            raise ValueError(
                "A host baseline requires its actual addressable native mask."
            )
        return np.asarray(local(raw.index_values, active, raw_baseline))
    if not isinstance(raw.index_values, jax.Array):
        raise TypeError(
            "Distributed raw history requires the existing JAX source-index bank."
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
                "Raw history payload, identities and active mask require the same actual owner-local axes."
            )
        results.append(
            jax.device_put(
                local(index_shard.data, active_shard.data, shard.data), shard.device
            )
        )
    return jax.make_array_from_single_device_arrays(
        raw_baseline.shape, raw_baseline.sharding, results
    )


def rebind_meshing_field_history(
    role: MeshingFieldStateRole,
    source_values: jax.Array | np.ndarray,
    target: CellMeshingResult,
    receipt: SimplexRestartRepack,
    /,
    *,
    history_index: int,
) -> tuple[MeshingFieldStateRole, jax.Array]:
    """Rebind actual raw history through the existing exact native repack proof.

    ``history_index`` is the explicitly authored cell/vertex payload tuple slot,
    not a slot guessed from a value shape, name or order. The existing receipt
    replays every saved-location, authority, retired record and payload value.
    """
    from ..meshing._restart_distribution import validate_simplex_restart_repack

    role.__post_init__()
    binding = role.index_binding
    if role.role not in ("material-history", "stage-history") or binding is None:
        raise ValueError(
            "Only explicitly indexed native history can be rebound through a forest receipt."
        )
    if binding.bank_name not in ("epoch/vertex_ids", "epoch/cell_ids"):
        raise ValueError("Forest history rebind requires its actual allocated raw bank.")
    if type(history_index) is not int or history_index < 0:
        raise ValueError(
            "A forest history transfer requires its explicit payload tuple index."
        )
    source = receipt.source_result
    if source is None:
        raise ValueError("History repack lacks its actual accepted scientific source.")
    validate_meshing_field_index_binding(role, source)
    validate_simplex_restart_repack(receipt, receipt.requested_target_owners)
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
            "The authored history slot is absent from its actual repack receipt."
        )
    if not _same_index_bank(binding.index_values, old_ids) or not _same_index_bank(
        source_values, saved[history_index]
    ):
        raise ValueError(
            "History/source rows differ from the exact saved indexed payload."
        )
    values = transferred[history_index]
    if (
        source_values.shape != role.shape
        or source_values.dtype != role.dtype
        or values.dtype != role.dtype
    ):
        raise ValueError(
            "History transfer changed the original scientific payload dtype or source schema."
        )
    target_binding = prepare_meshing_field_index_binding(target, binding.bank_name)
    if not _same_index_bank(target_binding.index_values, new_ids):
        raise ValueError(
            "Repacked history rows do not belong to the actual accepted target bank."
        )
    rebound = replace(role, shape=values.shape, index_binding=target_binding)
    validate_meshing_field_index_binding(rebound, target)
    return rebound, values


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
) -> jax.Array | np.ndarray:
    """Recover compact field rows from a proven repack's authoritative raw IDs.

    The existing numerical receipt owns which raw record is authoritative.
    Ghost duplicates, inactive capacity and row positions cannot substitute for
    that owner bank. Cold consumers inspect only addressable native records.
    """
    import jax.numpy as jnp

    from ..meshing._restart_distribution import validate_simplex_restart_repack
    from ..meshing._topology_edit import key_rows
    from ..sparse import gather_routes, RowRelation

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
            "Cold history recovery requires the exact allocated-to-compact entity bank pair."
        )
    if (
        raw_role.field_name != compact_role.field_name
        or raw_role.state_epoch != compact_role.state_epoch
        or raw_role.value_units != compact_role.value_units
    ):
        raise ValueError(
            "Cold history recovery cannot relabel the physical field, units or accepted epoch."
        )
    if type(history_index) is not int or history_index < 0:
        raise ValueError(
            "Cold history recovery requires its explicit repack payload tuple slot."
        )
    validate_simplex_restart_repack(receipt, receipt.requested_target_owners)
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
            "Cold history values differ from their exact proven native payload slot."
        )
    if (
        not _same_index_bank(raw.index_values, receipt_ids)
        or owners.shape != receipt_ids.shape
    ):
        raise ValueError(
            "Cold history identity/authority differs from the actual repacked native bank."
        )
    if (
        raw_values.shape != raw_role.shape
        or raw_values.dtype != raw_role.dtype
        or raw_role.dtype != compact_role.dtype
        or raw_role.shape[2:] != compact_role.shape[1:]
    ):
        raise ValueError(
            "Cold compact recovery changed the original physical payload schema or dtype."
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
                "Owner-local recovery differs from the declared compact physical rows."
            )
        return result
    if (
        isinstance(compact.index_values, jax.Array)
        and not compact.index_values.is_fully_addressable
    ):
        raise ValueError(
            "Compact physical rows must be this process's addressable entity bank."
        )
    identifiers, authority = np.asarray(receipt_ids), np.asarray(owners)
    selected = (identifiers >= 0) & (
        authority == np.arange(identifiers.shape[0], dtype=np.int32)[:, None]
    )
    owner_ids = identifiers[selected]
    if np.unique(owner_ids).size != owner_ids.size:
        raise ValueError("Repacked history has duplicate authoritative scientific IDs.")
    queries = np.asarray(compact.index_values)
    rows = key_rows(owner_ids[:, None], queries[:, None])
    if np.any(rows < 0):
        raise ValueError(
            "A compact scientific field ID is absent from the authoritative raw history."
        )
    if rows.size == 0:
        result = jnp.empty(compact_role.shape, dtype=compact_role.dtype)
    else:
        source_rows = np.flatnonzero(selected.reshape(-1))[rows]
        relation = RowRelation(source_rows[:, None], source_size=identifiers.size)
        flattened = jnp.asarray(raw_values).reshape(
            (identifiers.size, *raw_role.shape[2:])
        )
        result = jnp.squeeze(gather_routes(relation, flattened), axis=1)
    return np.asarray(result) if isinstance(raw_values, np.ndarray) else result


_SENTINEL = np.iinfo(np.int64).max


@eqx.filter_jit
def _ring_recovery(
    parts: AdaptiveSimplexParts,
    identifiers: jax.Array,
    owners: jax.Array,
    values: jax.Array,
    queries: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Circulate bounded query packets; each raw part answers only its authority."""
    import jax.numpy as jnp

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
    """Recover this process's compact rows from authoritative records on every owner.

    Only fixed-capacity query packets circulate between neighboring owners; the
    single host scalar exchange agrees on that capacity. Every compact ID must
    be answered by exactly one authoritative raw record.
    """
    from jax.experimental import multihost_utils

    shards = values.addressable_shards
    if len(shards) != 1 or queries.ndim != 1 or queries.dtype != np.int64:
        raise ValueError(
            "Owner-local recovery requires one addressable raw part and its int64 compact IDs."
        )
    capacity = int(
        np.max(multihost_utils.process_allgather(np.int64(max(queries.shape[0], 1))))
    )
    packet = np.full((1, capacity), -1, dtype=np.int64)
    packet[0, : queries.shape[0]] = queries
    placement = NamedSharding(receipt.parts.mesh, PartitionSpec(receipt.parts.axis_name))
    distributed = jax.make_array_from_process_local_data(
        placement, packet, (receipt.parts.part_count, capacity)
    )
    answer, found = _ring_recovery(
        receipt.parts, identifiers, owners, values, distributed
    )
    local_found = np.asarray(found.addressable_shards[0].data)[0, : queries.shape[0]]
    if np.any(local_found != 1):
        raise ValueError(
            "A compact scientific field ID lacks exactly one authoritative raw history record."
        )
    return answer.addressable_shards[0].data[0, : queries.shape[0]]


@dataclass(frozen=True)
class MeshingFieldStateRole:
    """An authored accepted-array role, never inferred from coefficient shape."""

    entry_name: str
    field_name: str
    role: str
    shape: tuple[int, ...]
    dtype: np.dtype
    state_epoch: int
    value_units: tuple[UnitDefinition, ...] | None = None
    index_binding: MeshingFieldIndexBinding | None = None

    def __post_init__(self) -> None:
        if any(
            type(name) is not str or not name or name != name.strip()
            for name in (self.entry_name, self.field_name)
        ):
            raise ValueError(
                "State roles require explicit entry and physical field names."
            )
        if self.role not in (
            "coefficients",
            "material-history",
            "stage-history",
            "state-epoch",
            "boundary-values",
            "boundary-mask",
            "trace-values",
        ):
            raise ValueError("Unknown physical field state role.")
        if type(self.shape) is not tuple or any(
            type(n) is not int or n < 0 for n in self.shape
        ):
            raise ValueError("State roles require an explicit nonnegative array shape.")
        if not isinstance(self.dtype, np.dtype) or self.dtype.hasobject:
            raise TypeError("State roles require an exact nonobject NumPy dtype.")
        if type(self.state_epoch) is not int or self.state_epoch < 0:
            raise ValueError("State roles require an explicit accepted state epoch.")
        if self.role == "state-epoch" and (
            self.shape != () or self.dtype.kind not in "iu"
        ):
            raise ValueError(
                "Accepted epoch state requires an exact scalar integer array."
            )
        if self.role == "boundary-mask" and self.dtype != np.dtype("bool"):
            raise ValueError(
                "Essential boundary selection requires an exact boolean mask."
            )
        if self.value_units is not None and (
            type(self.value_units) is not tuple
            or not self.value_units
            or any(type(unit) is not UnitDefinition for unit in self.value_units)
        ):
            raise TypeError(
                "Array role units require exact immutable physical unit owners."
            )
        if self.role == "material-history" and self.value_units is None:
            raise ValueError(
                "Material history requires its own explicit units, not inferred state units."
            )
        if self.role != "state-epoch":
            if type(self.index_binding) is not MeshingFieldIndexBinding:
                raise ValueError(
                    "Every physical value/history row requires its explicit owning index-bank binding."
                )
            prefix = tuple(
                self.shape[axis]
                for axis in self.index_binding.index_axes
                if axis < len(self.shape)
            )
            if prefix != self.index_binding.index_values.shape:
                raise ValueError(
                    "Physical payload axes differ from their declared row/entity bank."
                )
            if self.role in ("coefficients", "boundary-values", "boundary-mask") and (
                self.index_binding.bank_name.startswith("epoch/")
                or self.index_binding.field_space_id is None
            ):
                raise ValueError(
                    "Allocated raw banks are not compact physical coefficient or essential-boundary vectors."
                )
        elif self.index_binding is not None:
            raise ValueError("A scalar state epoch is not an indexed physical row bank.")


@dataclass(frozen=True, kw_only=True)
class MeshingFieldDeclaration:
    """One actual accepted carrier's complete FE or FV physical declaration.

    Units, derivatives, traces, accepted-state roles and original owner inputs
    are explicit. ``field_space_ids`` binds the freshly prepared family and DOF
    layout, including global coefficient identities, to the accepted carrier.
    The carrier may be a real generation part, a collective accepted result, or
    a registered assembly part. A standalone declaration never fabricates an
    overset registration to obtain physical authority.
    """

    part_name: str
    owner: str
    topology_id: str
    geometry_layout_id: str
    field_space_ids: frozendict[str, str]
    value_units: frozendict[str, tuple[UnitDefinition, ...]]
    maximum_derivative_orders: frozendict[str, int]
    branch_policy: BranchDifferentiationPolicy
    trace_policy: FieldTracePolicy
    state_roles: tuple[MeshingFieldStateRole, ...]
    history_policy: str
    numeric_version: str
    finite_element_fields: tuple[FiniteElementFieldSpec, ...] = ()
    precision_policy: FiniteElementPrecisionPolicy | None = None
    finite_volume_field_name: str | None = None
    finite_volume_component_names: tuple[str, ...] = ()
    finite_volume_coordinate_policy: str | None = None
    reconstruction_policy: (
        PiecewiseConstantReconstruction
        | CellPolynomialReconstructionPlan
        | UnstructuredWENOZReconstructionPlan
        | None
    ) = None
    reconstruction_policy_id: str | None = None
    reconstruction_options: frozendict[str, Any] = frozendict()
    finite_volume_options: frozendict[str, Any] = frozendict()
    finite_volume_owned_face_capacity: int | None = None
    finite_volume_system: EulerSystem | None = None
    finite_volume_flux_policy: RusanovFluxPlan | None = None
    finite_volume_boundaries: frozendict[
        str, ExtrapolationBoundary | SlipWallBoundary
    ] = frozendict()
    source_wall_policy: OversetPartSpec | None = None

    def __post_init__(self) -> None:
        if any(
            type(x) is not str or not x.strip()
            for x in (
                self.part_name,
                self.topology_id,
                self.geometry_layout_id,
                self.numeric_version,
            )
        ):
            raise ValueError(
                "Field declarations require exact part, topology, geometry and numeric identities."
            )
        if self.owner == "finite_element":
            if (
                not self.finite_element_fields
                or type(self.finite_element_fields) is not tuple
                or any(
                    type(x) is not FiniteElementFieldSpec
                    for x in self.finite_element_fields
                )
                or type(self.precision_policy) is not FiniteElementPrecisionPolicy
                or self.finite_volume_field_name is not None
                or self.finite_volume_component_names
                or self.reconstruction_policy is not None
                or self.finite_volume_options
                or self.reconstruction_options
                or self.finite_volume_owned_face_capacity is not None
                or self.finite_volume_system is not None
                or self.finite_volume_flux_policy is not None
                or self.finite_volume_boundaries
                or self.finite_volume_coordinate_policy is not None
                or self.reconstruction_policy_id is not None
            ):
                raise ValueError(
                    "FE declarations require actual field specs and precision, without FV inputs."
                )
            from ..discretization._cell_geometry import _CoordinateTabulator

            for field in self.finite_element_fields:
                for element in field.elements:
                    if element.form_basis is not None:
                        if (
                            type(element.form_basis) is not FormBasis
                            or type(element.tabulator) is not _ProxyTabulator
                        ):
                            raise TypeError(
                                "Form fields require their exact immutable basis and proxy tabulator."
                            )
                    elif type(element.tabulator) is _ProxyTabulator:
                        raise TypeError(
                            "A form proxy tabulator requires its declared immutable form basis."
                        )
                    elif element.tabulator is None:
                        if element.family not in ("Lagrange", "DiscontinuousLagrange"):
                            raise ValueError(
                                "FE fields require an exact registered native basis family, not a degree-based fallback."
                            )
                    elif type(element.tabulator) not in (
                        _FiniteElementTabulator,
                        _CoordinateTabulator,
                    ):
                        raise TypeError(
                            "FE declarations cannot retain unresolved basis callbacks or live templates."
                        )
            names = tuple(x.name for x in self.finite_element_fields)
            widths = {
                x.name: prod(x.component_shape) * prod(x.elements[0].value_shape)
                for x in self.finite_element_fields
            }
        elif self.owner == "finite_volume":
            if (
                self.finite_element_fields
                or self.precision_policy is not None
                or type(self.finite_volume_field_name) is not str
                or not self.finite_volume_field_name
                or type(self.finite_volume_component_names) is not tuple
                or not self.finite_volume_component_names
                or any(
                    type(name) is not str or not name or name != name.strip()
                    for name in self.finite_volume_component_names
                )
                or type(self.reconstruction_policy)
                not in (
                    PiecewiseConstantReconstruction,
                    CellPolynomialReconstructionPlan,
                    UnstructuredWENOZReconstructionPlan,
                )
            ):
                raise ValueError(
                    "FV declarations require explicit field, components and an unprepared reconstruction policy."
                )
            if self.finite_volume_coordinate_policy not in (
                "accepted-map",
                "represented-polyhedral",
            ):
                raise ValueError(
                    "FV coordinates require an explicit accepted-map or represented-polyhedral policy."
                )
            policy = self.reconstruction_policy
            if policy is None:
                raise ValueError(
                    "FV declarations require an unprepared reconstruction policy."
                )
            if self.reconstruction_policy_id != policy.plan_id:
                raise ValueError(
                    "FV reconstruction family differs from its original scientific policy binding."
                )
            if isinstance(policy, PiecewiseConstantReconstruction):
                ceiling = 0
            elif isinstance(
                policy,
                (CellPolynomialReconstructionPlan, UnstructuredWENOZReconstructionPlan),
            ):
                ceiling = policy.degree
            else:
                raise TypeError(
                    "FV declarations require an exact native reconstruction policy owner."
                )
            if (
                self.maximum_derivative_orders.get(self.finite_volume_field_name, -1)
                > ceiling
            ):
                raise ValueError(
                    "FV derivative declaration exceeds its actual reconstruction family."
                )
            expected_branch = (
                BranchDifferentiationPolicy.BRANCHWISE
                if type(self.reconstruction_policy) is UnstructuredWENOZReconstructionPlan
                else BranchDifferentiationPolicy.SMOOTH
            )
            if self.branch_policy != expected_branch:
                raise ValueError(
                    "FV derivative branch semantics differ from the actual reconstruction owner."
                )
            allowed_reconstruction = (
                {"stencil_direction"}
                if type(self.reconstruction_policy) is CellPolynomialReconstructionPlan
                else set()
            )
            if (
                type(self.reconstruction_options) is not frozendict
                or set(self.reconstruction_options) - allowed_reconstruction
            ):
                raise ValueError(
                    "FV reconstruction options must retain only the original unprepared owner's inputs."
                )
            names = (self.finite_volume_field_name,)
            widths = {
                self.finite_volume_field_name: len(self.finite_volume_component_names)
            }
            allowed = {
                "boundary_face_groups",
                "neighborhood_complete",
                "physical_boundary_faces",
                "closure_evidence_id",
            }
            if (
                type(self.finite_volume_options) is not frozendict
                or set(self.finite_volume_options) - allowed
            ):
                raise ValueError(
                    "FV options must retain only original owning preparation inputs."
                )
            if self.finite_volume_owned_face_capacity is not None and (
                type(self.finite_volume_owned_face_capacity) is not int
                or self.finite_volume_owned_face_capacity < 0
            ):
                raise ValueError(
                    "FV face execution capacity must retain an explicit nonnegative integer."
                )
            if len(set(self.finite_volume_component_names)) != len(
                self.finite_volume_component_names
            ):
                raise ValueError("FV physical components must be uniquely declared.")
            if type(self.finite_volume_boundaries) is not frozendict:
                raise TypeError(
                    "Physical FV boundary policies must be immutable owning declarations."
                )
            if self.finite_volume_system is not None:
                if (
                    type(self.finite_volume_system) is not EulerSystem
                    or type(self.finite_volume_flux_policy) is not RusanovFluxPlan
                    or self.finite_volume_component_names
                    != self.finite_volume_system.component_names
                    or any(
                        type(value) not in (ExtrapolationBoundary, SlipWallBoundary)
                        for value in self.finite_volume_boundaries.values()
                    )
                ):
                    raise ValueError(
                        "Euler continuation requires its actual system, flux and every physical boundary owner."
                    )
            elif (
                self.finite_volume_flux_policy is not None
                or self.finite_volume_boundaries
            ):
                raise ValueError(
                    "FV physical dynamics policies require their explicit owning system."
                )
        else:
            raise ValueError("Unknown registered physical field owner.")
        if len(set(names)) != len(names):
            raise ValueError("Physical field names must be unique per registration part.")
        for mapping in (
            self.field_space_ids,
            self.value_units,
            self.maximum_derivative_orders,
        ):
            if type(mapping) is not frozendict or set(mapping) != set(names):
                raise ValueError(
                    "Physical semantics must be explicitly keyed by every declared field."
                )
        if any(
            type(value) is not str or not value for value in self.field_space_ids.values()
        ):
            raise ValueError(
                "Every physical field requires its original scientific field-space binding."
            )
        for name in names:
            units = self.value_units[name]
            if (
                type(units) is not tuple
                or len(units) not in (1, widths[name])
                or any(type(unit) is not UnitDefinition for unit in units)
            ):
                raise ValueError(
                    "Field units must be explicit for the scalar or every physical component."
                )
            order = self.maximum_derivative_orders[name]
            if type(order) is not int or order < 0:
                raise ValueError(
                    "Coordinate derivative orders must be explicitly nonnegative."
                )
            if self.owner == "finite_element":
                field = next(
                    field for field in self.finite_element_fields if field.name == name
                )
                ceiling = min(
                    min(element.degree, 1) if element.mapping == "identity" else 1
                    for element in field.elements
                )
                if order > ceiling:
                    raise ValueError(
                        "FE derivative declaration exceeds its actual coordinate reconstruction owner."
                    )
        if (
            type(self.branch_policy) is not BranchDifferentiationPolicy
            or type(self.trace_policy) is not FieldTracePolicy
        ):
            raise TypeError(
                "Field derivative and trace semantics require their canonical owners."
            )
        if self.history_policy not in (
            "none",
            "accepted-state",
            "material-history",
            "multistep",
        ):
            raise ValueError("Unknown accepted field history policy.")
        if type(self.state_roles) is not tuple or any(
            type(role) is not MeshingFieldStateRole for role in self.state_roles
        ):
            raise TypeError(
                "Accepted state roles must be exact immutable physical records."
            )
        if any(role.field_name not in names for role in self.state_roles):
            raise ValueError("State roles must bind declared physical fields.")
        if len({role.entry_name for role in self.state_roles}) != len(self.state_roles):
            raise ValueError("Accepted field state entries must have unique roles.")
        coefficient_names = [
            role.field_name for role in self.state_roles if role.role == "coefficients"
        ]
        if sorted(coefficient_names) != sorted(names):
            raise ValueError(
                "Every declared field requires exactly one explicit coefficient role."
            )
        boundary_values = {
            role.field_name for role in self.state_roles if role.role == "boundary-values"
        }
        boundary_masks = {
            role.field_name for role in self.state_roles if role.role == "boundary-mask"
        }
        if boundary_values != boundary_masks:
            raise ValueError(
                "Essential boundary declarations require both actual masks and physical values."
            )
        histories = [
            role
            for role in self.state_roles
            if role.role in ("material-history", "stage-history", "state-epoch")
        ]
        if self.history_policy == "none" and histories:
            raise ValueError(
                "A history-free field cannot retain undeclared history state."
            )
        if self.history_policy != "none" and not histories:
            raise ValueError(
                "Stateful field policies require their actual history or epoch roles."
            )
        if self.history_policy == "material-history" and not any(
            role.role == "material-history" for role in histories
        ):
            raise ValueError(
                "Material history policy requires its actual material history array."
            )
        if len({role.state_epoch for role in self.state_roles}) != 1:
            raise ValueError(
                "A field declaration must bind one atomic accepted state epoch."
            )
        if self.source_wall_policy is not None and (
            type(self.source_wall_policy) is not OversetPartSpec
            or self.source_wall_policy.part_name != self.part_name
        ):
            raise ValueError(
                "Source-wall policy must retain this part's owning original declaration."
            )


class MeshingAcceptedEpoch(StrictModule, NonTrainableState):
    """One actual accepted carrier and all of its primitive physical owners."""

    carrier: CellMeshingResult
    declarations: tuple[MeshingFieldDeclaration, ...]
    fields: frozendict[str, jax.Array | np.ndarray]
    transition: MeshAdaptationResult | None
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        carrier: CellMeshingResult,
        declarations: tuple[MeshingFieldDeclaration, ...],
        fields: Mapping[str, jax.Array | np.ndarray],
        /,
        *,
        transition: MeshAdaptationResult | None = None,
    ) -> None:
        if type(carrier) is not CellMeshingResult:
            raise TypeError("An accepted epoch requires its actual certified carrier.")
        if (
            type(declarations) is not tuple
            or not declarations
            or any(
                type(declaration) is not MeshingFieldDeclaration
                for declaration in declarations
            )
        ):
            raise TypeError(
                "An accepted epoch requires complete exact physical declarations."
            )
        if len({declaration.part_name for declaration in declarations}) != 1:
            raise ValueError(
                "All accepted epoch physical owners must bind the same actual carrier name."
            )
        if not isinstance(fields, Mapping) or any(
            type(name) is not str
            or not name
            or not isinstance(value, (jax.Array, np.ndarray))
            for name, value in fields.items()
        ):
            raise TypeError("Accepted epoch fields require exact named numerical banks.")
        if transition is not None and type(transition) is not MeshAdaptationResult:
            raise TypeError(
                "Accepted epoch transitions require their actual native adaptation."
            )
        if transition is not None:
            from .._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
            from ._meshing_sources import _native_authority_equal

            if not _native_authority_equal(
                transition.target, carrier, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
            ):
                raise ValueError(
                    "Accepted epoch carrier differs from its actual transition target."
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
        from .._model._structure import model_recipe_array_values, model_structure_recipe
        from ..discretization._cell_geometry_validity import cell_geometry_id
        from ._meshing_sources import register_meshing_source_artifacts

        register_meshing_source_artifacts()
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
    """Authenticate one ordered history against its actual original generation."""
    from .._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from ..meshing._assembly import MeshPart
    from ._meshing_sources import _native_authority_equal

    accepted = records.get("accepted_data")
    target = records.get("accepted_target")
    if not isinstance(accepted, Mapping) or "accepted_epochs" not in accepted:
        if isinstance(accepted, Mapping) and "neurofluid_transport" in accepted:
            from ._meshing_sources import _validate_neurofluid_transport_role

            _validate_neurofluid_transport_role(records)
            return ()
        if target is not None and not (
            isinstance(accepted, Mapping)
            and ("adaptation_results" in accepted or "adaptation_event" in accepted)
        ):
            raise ValueError(
                "An accepted target requires its actual complete accepted history."
            )
        return ()
    epochs = accepted["accepted_epochs"]
    if (
        type(epochs) is not tuple
        or not epochs
        or any(type(epoch) is not MeshingAcceptedEpoch for epoch in epochs)
    ):
        raise TypeError(
            "Accepted epochs require one nonempty ordered tuple of exact owning records."
        )
    epochs_ = cast(tuple[MeshingAcceptedEpoch, ...], epochs)
    generation = records.get("generation_part")
    if (
        type(generation) is not MeshPart
        or type(generation.carrier) is not CellMeshingResult
    ):
        raise TypeError("Accepted epochs require their actual original generation part.")
    if any(
        declaration.part_name != generation.name
        for epoch in epochs_
        for declaration in epoch.declarations
    ):
        raise ValueError(
            "Accepted epoch field owners must retain their actual original part name."
        )
    if epochs_[0].transition is not None or not _native_authority_equal(
        epochs_[0].carrier,
        generation.carrier,
        limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
    ):
        raise ValueError(
            "The initial accepted epoch must be the actual original generation."
        )
    for previous, epoch in zip(epochs_, epochs_[1:], strict=False):
        transition = epoch.transition
        if (
            transition is None
            or not _native_authority_equal(
                transition.source,
                previous.carrier,
                limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
            )
            or not _native_authority_equal(
                transition.target,
                epoch.carrier,
                limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
            )
        ):
            raise ValueError(
                "Accepted epoch transition changes its preceding actual source or current target."
            )
    if type(target) is not CellMeshingResult or not _native_authority_equal(
        target,
        epochs_[-1].carrier,
        limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
    ):
        raise ValueError(
            "The accepted target must be the exact final carrier of its authenticated epoch chain."
        )
    if "adaptation_results" in accepted:
        raise ValueError(
            "Accepted epochs are the sole history; retain transitions on their actual epochs."
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
    """Apply the same actual physical ownership law to each declared owner."""
    declared_entries: set[str] = set()
    for declaration, carrier in owners:
        prepared, reconstruction = prepare_meshing_field_owner(
            declaration, carrier, index_source=index_source
        )
        if declaration.finite_volume_system is not None:
            _prepare_field_dynamics(declaration, prepared, reconstruction)
        if declaration.source_wall_policy is not None:
            if registration is None:
                raise ValueError(
                    "A source-wall declaration requires its actual owning registration."
                )
            specs = {spec.part_name: spec for spec in registration.specs}
            if (
                declaration.source_wall_policy.spec_id
                != specs[declaration.part_name].spec_id
            ):
                raise ValueError(
                    "Stored physical source-wall policy differs from registration authority."
                )
        for role in declaration.state_roles:
            value = arrays.get(role.entry_name)
            if (
                not isinstance(value, (jax.Array, np.ndarray))
                or value.shape != role.shape
                or value.dtype != role.dtype
            ):
                raise ValueError(
                    "Accepted physical field array differs from its declared role, shape or dtype."
                )
            if role.entry_name in declared_entries:
                raise ValueError(
                    "A primitive accepted array cannot belong to two physical field owners."
                )
            declared_entries.add(role.entry_name)
            if role.role == "state-epoch" and (
                value.shape != () or int(np.asarray(value)) != role.state_epoch
            ):
                raise ValueError(
                    "Accepted state epoch array differs from its atomic physical state binding."
                )
    if require_complete and set(arrays) != declared_entries:
        raise ValueError(
            "Every primitive physical state array requires its original explicit owner role."
        )


def meshing_field_artifact_types() -> tuple[tuple[str, tuple[type, ...]], ...]:
    """Admission groups for the one canonical artifact registry, not a decoder."""
    _register_coefficient_identity_bank_codec()
    return (
        (
            "phydrax.lifecycle",
            (
                MeshingAcceptedEpoch,
                MeshingFieldDeclaration,
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
                _FiniteElementTabulator,
                FormBasis,
                _ProxyTabulator,
                _TensorFactors,
                _HybridFactors,
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


def rebuild_meshing_field_record(value: Any, /) -> Any | None:
    """Reconstruct authored inputs so the root owner checks all derived fields."""
    if type(value) is MeshingCoefficientIdentityBank:
        return MeshingCoefficientIdentityBank(value.field_space_id, value.rows)
    if type(value) is MeshingAcceptedEpoch:
        return MeshingAcceptedEpoch(
            value.carrier, value.declarations, value.fields, transition=value.transition
        )
    if type(value) in (
        MeshingFieldDeclaration,
        MeshingFieldStateRole,
        MeshingFieldIndexBinding,
    ):
        return type(value)(
            **{member.name: getattr(value, member.name) for member in fields(value)}
        )
    if type(value) is GlobalCoefficientId:
        return GlobalCoefficientId(value.field_space_id, value.ordinal, value.component)
    if type(value) is DimensionSignature:
        return DimensionSignature(value._fraction_terms())
    if type(value) is UnitDefinition:
        return UnitDefinition(
            value.symbol,
            value.dimension,
            value.reference_system_id,
            (value.scale_numerator, value.scale_denominator),
        )
    if type(value) is EulerSystem:
        return EulerSystem(value.dimension, material=value.material)
    if type(value) is IdealGasMaterial:
        return IdealGasMaterial(
            value.gamma,
            value.gas_constant,
            density_floor=value.density_floor,
            pressure_floor=value.pressure_floor,
        )
    if type(value) is RusanovFluxPlan:
        return RusanovFluxPlan(smooth_epsilon=value.smooth_epsilon)
    if type(value) is ExtrapolationBoundary:
        return ExtrapolationBoundary()
    if type(value) is SlipWallBoundary:
        return SlipWallBoundary()
    if type(value) is FiniteElementFieldSpec:
        if not value.block_names:
            return FiniteElementFieldSpec(
                value.name, value.elements[0], component_shape=value.component_shape
            )
        if all(
            element.element_id == value.elements[0].element_id
            for element in value.elements
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
    if type(value) is FiniteElementSpec:
        from ..discretization.fem import _reference

        tabulator = value.tabulator
        if value.form_basis is not None:
            if (
                type(value.form_basis) is not FormBasis
                or type(tabulator) is not _ProxyTabulator
            ):
                raise TypeError(
                    "Form elements require their exact immutable basis and proxy tabulator."
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
        if type(tabulator) is _ProxyTabulator:
            raise TypeError(
                "A form proxy tabulator requires its declared immutable form basis."
            )
        if type(tabulator) is _FiniteElementTabulator:
            if type(tabulator.source) is FiniteElementSpec:
                base = tabulator.source
                if value.family not in (
                    "DiscontinuousLagrange",
                    "DiscontinuousTensorProductLagrange",
                ):
                    raise ValueError(
                        "A cell-local scalar basis must retain its actual discontinuous family."
                    )
                if base.value_shape or base.mapping != "identity":
                    raise ValueError(
                        "Discontinuous scalar fields require their actual identity-mapped base."
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
                    tabulator=_FiniteElementTabulator(base, "tabulate"),
                    tabulator_id=f"discontinuous:{base.element_id}",
                )
            return tabulator.source.finite_element()
        if tabulator is None:
            if value.family == "Lagrange":
                return _reference.lagrange_element(value.cell_kind, value.degree)
            if value.family == "DiscontinuousLagrange":
                return _reference.discontinuous_element(value.cell_kind, value.degree)
        return None
    if type(value) is ReferenceNodalFamily:
        return ReferenceNodalFamily(
            value.cell_kind, value.orders, node_set=value.node_set
        )
    if type(value) is SimplexNodalFamily:
        return SimplexNodalFamily(value.cell_kind, value.order)
    if type(value) is HybridReferenceFamily:
        return HybridReferenceFamily(
            value.cell_kind, value.orders if value.cell_kind == "prism" else value.degree
        )
    if type(value) is _FiniteElementTabulator:
        return _FiniteElementTabulator(value.source, value.method)
    if type(value) is FormBasis:
        return FormBasis(
            value.dimension, value.form_degree, value.order, value.family, value.twist
        )
    if type(value) is _ProxyTabulator:
        if (
            type(value.basis) is not FormBasis
            or type(value.value_spec) is not FormValueSpec
        ):
            raise TypeError(
                "A portable form proxy requires its exact immutable basis and value spec."
            )
        form = value.value_spec.form_type
        if (
            (form.dimension, form.degree, form.twist)
            != (value.basis.dimension, value.basis.form_degree, value.basis.twist)
            or form.ambient_dimension != form.dimension
            or form.fiber_shape
        ):
            raise ValueError(
                "A portable form proxy must retain its actual basis form type."
            )
        return _ProxyTabulator(value.basis, value.value_spec)
    if type(value) is FormType:
        return FormType(
            value.dimension,
            value.degree,
            twist=value.twist,
            fiber_shape=value.fiber_shape,
            ambient_dimension=value.ambient_dimension,
        )
    if type(value) is FormValueSpec:
        return FormValueSpec(value.form_type, proxy=value.proxy)
    if type(value) is PiecewiseConstantReconstruction:
        return PiecewiseConstantReconstruction()
    if type(value) is FiniteElementPrecisionPolicy:
        return FiniteElementPrecisionPolicy(
            **{
                name: getattr(value, name)
                for name in (
                    "storage_dtype",
                    "geometry_dtype",
                    "evaluation_dtype",
                    "accumulation_dtype",
                    "output_dtype",
                    "compensated_accumulation",
                )
            }
        )
    if type(value) is CellPolynomialReconstructionPlan:
        return CellPolynomialReconstructionPlan(
            value.degree,
            **{
                name: getattr(value, name)
                for name in ("weight_power", "oversampling", "rcond", "condition_limit")
            },
        )
    if type(value) is UnstructuredWENOZReconstructionPlan:
        return UnstructuredWENOZReconstructionPlan(
            value.degree,
            **{
                name: getattr(value, name)
                for name in (
                    "weight_power",
                    "oversampling",
                    "linear_weights",
                    "epsilon",
                    "power",
                    "limiter",
                )
            },
        )
    if type(value) is FieldTracePolicy:
        return FieldTracePolicy(value.kind, sides=value.sides)
    return None


def prepare_meshing_field_owner(
    declaration: MeshingFieldDeclaration,
    carrier: Any,
    /,
    *,
    index_source: CellMeshingResult | None = None,
) -> tuple[
    FiniteElementDiscretization | UnstructuredFiniteVolumeDiscretization, Any | None
]:
    """Prepare the actual owner from accepted mesh/geometry, without live templates."""
    declaration.__post_init__()
    mesh, geometry = carrier.mesh, carrier.geometry
    if (
        mesh.topology_id != declaration.topology_id
        or geometry.geometry_layout_id != declaration.geometry_layout_id
    ):
        raise ValueError(
            "Stored physical declaration belongs to another accepted carrier."
        )
    prepared: FiniteElementDiscretization | UnstructuredFiniteVolumeDiscretization
    if declaration.owner == "finite_element":
        prepared = FiniteElementPlan(
            mesh,
            declaration.finite_element_fields,
            coordinate_spec=geometry,
            precision_policy=declaration.precision_policy,
        ).prepare(numeric_version=declaration.numeric_version)
        reconstruction = None
    else:
        field_name = declaration.finite_volume_field_name
        if field_name is None:
            raise ValueError(
                "FV preparation requires its explicitly declared field name."
            )
        if declaration.finite_volume_coordinate_policy == "represented-polyhedral":
            from ..discretization._cell_complex import PolyhedralConnectivity

            if type(mesh.connectivity) is not PolyhedralConnectivity:
                raise ValueError(
                    "Represented polyhedral FV coordinates require actual polyhedral incidence."
                )
        prepared = UnstructuredFiniteVolumePlan.from_cell_mesh(
            mesh,
            field_name=field_name,
            component_names=declaration.finite_volume_component_names,
            **declaration.finite_volume_options,
        ).prepare(
            numeric_version=declaration.numeric_version,
            owned_face_capacity=declaration.finite_volume_owned_face_capacity,
            cell_geometry=geometry
            if declaration.finite_volume_coordinate_policy == "accepted-map"
            else None,
        )
        policy = declaration.reconstruction_policy
        if isinstance(policy, PiecewiseConstantReconstruction):
            reconstruction = policy
        elif isinstance(policy, CellPolynomialReconstructionPlan):
            reconstruction = policy.prepare(
                prepared, **declaration.reconstruction_options
            )
        elif isinstance(policy, UnstructuredWENOZReconstructionPlan):
            reconstruction = policy.prepare(prepared)
        else:
            raise TypeError(
                "FV preparation requires an exact native reconstruction policy owner."
            )
    if isinstance(prepared, FiniteElementDiscretization):
        physical_spaces = prepared.field_spaces
    else:
        physical_spaces = (prepared.cell_space,)
    actual = {space.name: space.field_space_id for space in physical_spaces}
    if actual != dict(declaration.field_space_ids):
        raise ValueError(
            "Fresh physical field family or global coefficient layout differs from its scientific binding."
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
                    "Accepted field coefficients require their actual array coefficient space."
                )
            expected = vector_space.shape
            if role.shape != expected:
                raise ValueError(
                    "Explicit accepted coefficients do not match the freshly prepared field layout."
                )
        elif role.role == "boundary-mask":
            if not isinstance(prepared, FiniteElementDiscretization):
                raise ValueError(
                    "FV physical boundaries require original face groups, not an FE essential mask."
                )
            dof_map = prepared.dof_maps[prepared._field_index(role.field_name)]
            if role.shape != (dof_map.global_dof_count,):
                raise ValueError(
                    "Stored essential boundary mask differs from the actual FE coefficient ownership."
                )
    return prepared, reconstruction


def _prepare_field_dynamics(
    declaration: MeshingFieldDeclaration, prepared: Any, reconstruction: Any, /
) -> Any:
    from ..discretization.finite_volume._unstructured_dynamics import (
        PreparedUnstructuredFiniteVolumeDynamics,
        UnstructuredFiniteVolumeBoundarySet,
        UnstructuredFiniteVolumeMethodPlan,
    )

    flux_policy = declaration.finite_volume_flux_policy
    if (
        declaration.owner != "finite_volume"
        or declaration.finite_volume_system is None
        or flux_policy is None
    ):
        raise ValueError(
            "Physical continuation requires an explicit stored FV system declaration."
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


def prepare_meshing_field_dynamics(
    declaration: MeshingFieldDeclaration, carrier: Any, /
) -> Any:
    """Rebuild actual physical FV dynamics from stored system/flux/wall policies."""
    prepared, reconstruction = prepare_meshing_field_owner(declaration, carrier)
    return _prepare_field_dynamics(declaration, prepared, reconstruction)


def validate_meshing_field_declarations(records: Mapping[str, Any], /) -> None:
    """Bind every stored array to its actual accepted carrier and physical owner."""
    accepted = records.get("accepted_data")
    declarations = records.get("field_declarations")
    epochs = validate_meshing_accepted_epochs(records)
    if epochs:
        if declarations is not None or (
            isinstance(accepted, Mapping) and "fields" in accepted
        ):
            raise ValueError(
                "Accepted epochs own their complete declarations and banks; do not retain a second final-field owner."
            )
        return
    if declarations is None:
        if isinstance(accepted, Mapping) and "fields" in accepted:
            raise ValueError(
                "Accepted physical fields require explicit canonical root field_declarations."
            )
        return
    from ..meshing._assembly import MeshPart
    from ..meshing._initial_certification import InitialCollectiveMeshEvidence
    from ..meshing._result import CellMeshingResult, require_original_meshing_source

    if not isinstance(declarations, Mapping):
        raise TypeError("Physical field declarations must be an explicit owning mapping.")
    registration = records.get("registration")
    index_source: CellMeshingResult | None = None
    if registration is not None:
        if type(registration) is not OversetRegistration:
            raise TypeError(
                "Physical registration must be the exact owning registration."
            )
        carriers = {part.name: part.carrier for part in registration.assembly.parts}
    else:
        target = records.get("collective_target")
        accepted_target = records.get("accepted_target")
        generation = records.get("generation_part")
        if generation is not None and type(generation) is not MeshPart:
            raise TypeError("Generation field authority must retain its actual MeshPart.")
        if target is not None and accepted_target is not None:
            raise ValueError("Physical fields must select one accepted target authority.")
        if target is not None:
            if type(target) is not CellMeshingResult or target.mesh.storage is None:
                raise TypeError(
                    "Collective fields require the actual accepted owner-local result."
                )
            index_source = target
            original = require_original_meshing_source(target)
            if generation is not None:
                if isinstance(original, CellMeshingResult):
                    if original.result_id != generation.carrier.result_id:
                        raise ValueError(
                            "Collective physical fields retain another original generation part."
                        )
                elif isinstance(original, InitialCollectiveMeshEvidence):
                    initial = generation.carrier.collective_evidence
                    if (
                        not isinstance(initial, InitialCollectiveMeshEvidence)
                        or initial.evidence_id != original.evidence_id
                        or generation.carrier.mesh.mesh_id != original.mesh_id
                    ):
                        raise ValueError(
                            "Collective physical fields retain another authored initial source theorem."
                        )
                else:
                    raise TypeError(
                        "Collective physical source must retain its exact native source owner."
                    )
                carriers = {generation.name: target}
            else:
                if len(declarations) != 1:
                    raise ValueError(
                        "One accepted result requires one explicit physical owner declaration."
                    )
                carriers = {next(iter(declarations)): target}
        elif accepted_target is not None:
            if (
                type(accepted_target) is not CellMeshingResult
                or accepted_target.mesh.storage is not None
            ):
                raise TypeError(
                    "Accepted serial fields require their actual dense CellMeshingResult."
                )
            if generation is not None:
                raise ValueError(
                    "Accepted serial fields must not relabel an adapted target as generation."
                )
            if len(declarations) != 1:
                raise ValueError(
                    "One accepted serial result requires one physical owner declaration."
                )
            index_source = accepted_target
            carriers = {next(iter(declarations)): accepted_target}
        elif generation is not None:
            carriers = {generation.name: generation.carrier}
        else:
            raise ValueError(
                "Physical field declarations require an actual accepted result or registration."
            )
    if set(declarations) != set(carriers):
        raise ValueError(
            "Physical fields must declare every actual owning carrier exactly once."
        )
    if not isinstance(accepted, Mapping):
        raise ValueError(
            "Physical field declarations require their accepted primitive state arrays."
        )
    arrays = accepted.get("fields", accepted)
    if not isinstance(arrays, Mapping):
        raise ValueError(
            "Accepted physical fields must retain their explicitly named array roles."
        )
    owners: list[tuple[MeshingFieldDeclaration, Any]] = []
    for name, declaration in declarations.items():
        if (
            type(declaration) is not MeshingFieldDeclaration
            or declaration.part_name != name
        ):
            raise ValueError(
                "Physical field keys require exact matching owning declarations."
            )
        owners.append((declaration, carriers[name]))
    _validate_field_owner_banks(
        tuple(owners),
        arrays,
        index_source=index_source,
        registration=registration,
        require_complete="fields" in accepted,
    )
