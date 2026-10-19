#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
import time
from abc import abstractmethod
from collections.abc import Callable
from typing import ClassVar, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax._model._component as _dependency_phydrax__model__component
import phydrax._model._ports as _dependency_phydrax__model__ports
import phydrax.meshing._adaptation as _dependency_phydrax_meshing__adaptation
import phydrax.meshing._audit as _dependency_phydrax_meshing__audit
import phydrax.meshing._contracts as _dependency_phydrax_meshing__contracts
import phydrax.meshing._metric as _dependency_phydrax_meshing__metric
import phydrax.meshing._mixed_adaptation as _dependency_phydrax_meshing__mixed_adaptation
import phydrax.meshing._optimization as _dependency_phydrax_meshing__optimization
import phydrax.meshing._quality as _dependency_phydrax_meshing__quality
import phydrax.meshing._result as _dependency_phydrax_meshing__result
import phydrax.meshing._scope as _dependency_phydrax_meshing__scope
import phydrax.meshing._sizing as _dependency_phydrax_meshing__sizing

from .._differentiation import ComponentAuthority
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._model import (
    AbstractArrayModel,
    AbstractComponentSlot,
    bind_component,
    ComponentContract,
    ModelPorts,
    PortMapping,
)
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import normalized_identifier
from ..discretization import (
    CellGeometrySpec,
    CellMesh,
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ..optim import OptimizationTermination
from ..typing import (
    checked,
    ConvertibleToArray,
    Dim,
    Float64,
    Identifier,
    Identifiers,
    parse,
    PRNGKey,
)
from ._measurements import measure_phase, NativeMeshingPhaseRecorder


@checked
def mesh_proposal_scope(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    dimension: int,
    entity_ids: ArrayLike | None = None,
    /,
) -> _dependency_phydrax_meshing__scope.MeshingScope:
    """Bind proposal entities to the certified mesh and result revision.

    Proposal values always follow the scope's sorted global-ID order, not the
    mesh's storage order. Omitting IDs selects all entities of the given degree.
    """
    entities = source.mesh.entity_set(dimension)
    scope = _dependency_phydrax_meshing__scope.MeshingScope(
        source.mesh.mesh_id,
        source.result_id,
        _dependency_phydrax_meshing__scope.MeshingEntityKind.MESH,
        dimension,
        entities.entity_set_id,
        entities.entity_ids if entity_ids is None else entity_ids,
    )
    _scope_rows(source, scope)
    return scope


def _scope_rows(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    scope: _dependency_phydrax_meshing__scope.MeshingScope,
) -> np.ndarray:
    mesh = source.mesh
    if (
        scope.entity_kind is not _dependency_phydrax_meshing__scope.MeshingEntityKind.MESH
        or scope.source_id != mesh.mesh_id
        or scope.source_revision != source.result_id
        or scope.entity_dimension > mesh.topological_dimension
    ):
        raise ValueError(
            f"MeshProposal.scope must bind the current mesh revision and compatible mesh entities; got {(scope.source_id, mesh.mesh_id, scope.source_revision, source.result_id, scope.entity_kind, scope.entity_dimension)!r}"
        )
    entities = mesh.entity_set(scope.entity_dimension)
    if scope.entity_set_id != entities.entity_set_id:
        raise ValueError(
            f"MeshProposal.scope.entity_set_id must bind the current entity set; got {(scope.entity_set_id, entities.entity_set_id)!r}"
        )
    identifiers = np.asarray(entities.entity_ids)
    requested = np.asarray(scope.entity_ids)
    order = np.argsort(identifiers)
    positions = np.searchsorted(identifiers[order], requested)
    if np.any(positions >= identifiers.size) or not np.array_equal(
        identifiers[order[np.minimum(positions, identifiers.size - 1)]], requested
    ):
        raise ValueError(
            f"MeshProposal.scope.entity_ids must contain only known source entity IDs; got {requested!r}"
        )
    rows = order[positions]
    if not np.all(np.asarray(entities.active_mask)[rows]):
        raise ValueError(
            f"MeshProposal.scope.entity_ids must contain only active source entities; got {requested!r}"
        )
    return rows


class _AbstractMeshProposal(StrictModule, NonTrainableState):
    """Untrusted proposal values bound to one source scope.

    ``proposer_id`` names the authoring proposer; ``evaluation_id`` names the
    evaluation that produced the values (model, features, address and
    realization) when a learned proposer made them, else ``None``.
    """

    scope: _dependency_phydrax_meshing__scope.MeshingScope
    values: Array
    coordinate_contract_id: str = eqx.field(static=True)
    proposer_id: str = eqx.field(static=True)
    evaluation_id: str | None = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: _dependency_phydrax_meshing__result.CellMeshingResult,
        scope: _dependency_phydrax_meshing__scope.MeshingScope,
        values: ArrayLike,
        *,
        dimension: int,
        value_shape: tuple[int, ...],
        proposer_id: str,
        evaluation_id: str | None = None,
    ) -> None:
        _scope_rows(source, scope)
        if scope.entity_dimension != dimension:
            raise ValueError(
                f"MeshProposal.scope.entity_dimension must match the declared proposal entity dimension; got {(scope.entity_dimension, dimension)!r}"
            )
        data = np.asarray(values, dtype=np.float64)
        if data.shape != (scope.entity_ids.size, *value_shape) or not np.all(
            np.isfinite(data)
        ):
            raise ValueError(
                f"MeshProposal.values must be finite and align with sorted scope IDs; got {(data.shape, scope.entity_ids.size, value_shape, bool(np.all(np.isfinite(data))))!r}"
            )
        self.scope = scope
        _host_values = np.asarray(data)
        (self.values,) = jax.device_put((_host_values,))
        self.coordinate_contract_id = source.coordinate_contract.spatial_id
        self.proposer_id = normalized_identifier(proposer_id, "proposer_id")
        self.evaluation_id = (
            None
            if evaluation_id is None
            else normalized_identifier(evaluation_id, "evaluation_id")
        )
        self.proposal_id = canonical_fingerprint(
            {
                "kind": type(self).__name__,
                "scope": scope.scope_id,
                "values": array_tree_fingerprint(self.values),
                "coordinate_contract": self.coordinate_contract_id,
                "proposer": self.proposer_id,
                "evaluation": self.evaluation_id,
            }
        )


class MeshMarkingProposal(_AbstractMeshProposal):
    """Untrusted cell priorities; positive scores request native bisection."""

    def __init__(
        self,
        source: _dependency_phydrax_meshing__result.CellMeshingResult,
        scope: _dependency_phydrax_meshing__scope.MeshingScope,
        scores: ArrayLike,
        /,
        *,
        proposer_id: str,
        evaluation_id: str | None = None,
    ) -> None:
        super().__init__(
            source,
            scope,
            scores,
            dimension=source.mesh.topological_dimension,
            value_shape=(),
            proposer_id=proposer_id,
            evaluation_id=evaluation_id,
        )


class MeshSizeProposal(_AbstractMeshProposal):
    """Untrusted vertex sizes, expressed in the source coordinate length unit."""

    def __init__(
        self,
        source: _dependency_phydrax_meshing__result.CellMeshingResult,
        scope: _dependency_phydrax_meshing__scope.MeshingScope,
        sizes: ArrayLike,
        /,
        *,
        proposer_id: str,
        evaluation_id: str | None = None,
    ) -> None:
        super().__init__(
            source,
            scope,
            sizes,
            dimension=0,
            value_shape=(),
            proposer_id=proposer_id,
            evaluation_id=evaluation_id,
        )


class MeshMetricProposal(_AbstractMeshProposal):
    """Untrusted vertex tensors; projection repairs symmetry and definiteness."""

    def __init__(
        self,
        source: _dependency_phydrax_meshing__result.CellMeshingResult,
        scope: _dependency_phydrax_meshing__scope.MeshingScope,
        metrics: ArrayLike,
        /,
        *,
        proposer_id: str,
        evaluation_id: str | None = None,
    ) -> None:
        dimension = source.mesh.ambient_dimension
        super().__init__(
            source,
            scope,
            metrics,
            dimension=0,
            value_shape=(dimension, dimension),
            proposer_id=proposer_id,
            evaluation_id=evaluation_id,
        )


class MeshCoordinateProposal(_AbstractMeshProposal):
    """Untrusted vertex targets in one explicit, unchanged coordinate contract."""

    @checked
    def __init__(
        self,
        source: _dependency_phydrax_meshing__result.CellMeshingResult,
        scope: _dependency_phydrax_meshing__scope.MeshingScope,
        coordinates: ArrayLike,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        proposer_id: str,
        evaluation_id: str | None = None,
    ) -> None:
        if coordinate_contract.spatial_id != source.coordinate_contract.spatial_id:
            raise ValueError(
                f"MeshCoordinateProposal.coordinate_contract must match the source coordinate contract; got {(coordinate_contract.spatial_id, source.coordinate_contract.spatial_id)!r}"
            )
        super().__init__(
            source,
            scope,
            coordinates,
            dimension=0,
            value_shape=(source.mesh.ambient_dimension,),
            proposer_id=proposer_id,
            evaluation_id=evaluation_id,
        )


type MeshProposal = (
    MeshMarkingProposal | MeshSizeProposal | MeshMetricProposal | MeshCoordinateProposal
)

MeshProposerKind: TypeAlias = Literal["marking", "size", "metric", "coordinate"]


class _ProposalEntityDim(Dim):
    """Entities of the proposal feature scope."""


class _ProposalFeatureDim(Dim, minimum=1):
    """Independently identified scientific feature columns."""


@final
class MeshProposalFeatures(StrictModule, NonTrainableState):
    """Owner-identified feature columns bound to the result and entity scope."""

    __strict_contract__ = True

    values: Float64[_ProposalEntityDim, _ProposalFeatureDim]
    scope: _dependency_phydrax_meshing__scope.MeshingScope
    source_result_id: Identifier = eqx.field(static=True)
    feature_ids: Identifiers[_ProposalFeatureDim] = eqx.field(static=True)
    feature_owner_id: Identifier = eqx.field(static=True)
    feature_id: Identifier = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: _dependency_phydrax_meshing__result.CellMeshingResult,
        scope: _dependency_phydrax_meshing__scope.MeshingScope,
        values: ArrayLike,
        /,
        *,
        feature_ids: tuple[str, ...],
        feature_owner_id: str,
    ) -> None:
        _scope_rows(source, scope)
        data = np.asarray(values, dtype=np.float64)
        identifiers = tuple(feature_ids)
        if (
            data.ndim != 2
            or data.shape != (scope.entity_ids.size, len(identifiers))
            or not identifiers
            or len(set(identifiers)) != len(identifiers)
            or any(not isinstance(value, str) or not value for value in identifiers)
            or not np.all(np.isfinite(data))
        ):
            raise ValueError(
                f"MeshProposalFeatures.values must contain finite scoped rows and unique nonempty scientific column IDs; got {(data.shape, scope.entity_ids.size, identifiers, bool(np.all(np.isfinite(data))))!r}"
            )
        if not isinstance(feature_owner_id, str) or not feature_owner_id:
            raise ValueError(
                f"MeshProposalFeatures.feature_owner_id must identify the producing scientific owner; got {feature_owner_id!r}"
            )
        _host_values, self.scope = np.asarray(data, dtype=np.float64), scope
        (self.values,) = jax.device_put((_host_values,))
        self.source_result_id, self.feature_ids = source.result_id, identifiers
        self.feature_owner_id = feature_owner_id
        self.feature_id = canonical_fingerprint(
            {
                "kind": "mesh-proposal-features",
                "source": source.result_id,
                "scope": scope.scope_id,
                "columns": identifiers,
                "owner": feature_owner_id,
                "values": array_tree_fingerprint(data),
            }
        )


class AbstractMeshProposer(AbstractComponentSlot):
    """Source of untrusted typed mesh proposals.

    The base is the neutral `DECISION` slot of mesh adaptation: a proposer
    decides where and how a certified mesh should adapt, but it never produces
    a mesh. `propose(source, features, scope=...)` returns one typed marking,
    size, metric or coordinate proposal bound to the source revision; only
    `project_mesh_proposal` and `prepare_mesh_proposal` turn it into a trusted
    candidate, under a `MeshProposalSafetyPolicy` and native adaptation.
    `features` hold one row per scope entity in sorted global-ID order.
    """

    component_authority: ClassVar[ComponentAuthority] = ComponentAuthority.DECISION
    slot_semantic_id: ClassVar[str] = "phydrax.meshing.mesh-proposer"

    proposer_id: eqx.AbstractVar[str]

    @abstractmethod
    def propose(
        self,
        source: _dependency_phydrax_meshing__result.CellMeshingResult,
        features: MeshProposalFeatures,
        /,
        *,
        scope: _dependency_phydrax_meshing__scope.MeshingScope | None = None,
    ) -> MeshProposal:
        raise NotImplementedError


def _model_value_shape(
    size: int | tuple[int, ...] | Literal["scalar"], /
) -> tuple[int, ...]:
    match size:
        case "scalar":
            return ()
        case int():
            return (size,)
        case _:
            return tuple(size)


class LearnedMeshProposer(AbstractMeshProposer):
    """Pointwise learned marking, size, metric or coordinate proposer.

    The model maps one feature row to one proposal value: a marking score per
    cell (`kind="marking"`), a size per vertex (`"size"`), a square coordinate-axis
    metric tensor (`"metric"`), or a source-contract coordinate vector
    (`"coordinate"`). Declared sizes bind the model: `in_size` is the feature width and `out_size`
    produces the value shape; the model must use a pointwise flat binding.
    Deterministic evaluation uses no key; stochastic proposals use explicit,
    scientific entity addresses rather than batch positions. `evaluate(features)` is the differentiable
    per-entity map used for supervised training, for example against marking
    targets derived from `FiniteElementDWRIndicators.absolute` or
    `HighEnthalpyAMREvidence.refine_mask` reordered to the scope's global IDs.

    `propose` wraps the values in the typed proposal of `kind`, which rejects
    non-finite values and stale scopes; the proposal stays untrusted until the
    native projection certifies it. The model is a dynamic child whose arrays
    keep their own roles and is bound to the `AbstractMeshProposer` `DECISION`
    slot; `component_contract()` returns the bound contract.

    `ports` declare the scientific identity of the proposer's values: one
    feature-row port (event shape `(in_size,)`) as the input and the proposal
    value port (scalar, coordinate vector, or square metric tensor event shape)
    as the output. A model declaring ports requires `ports` and an explicit
    `port_mapping` binding its ordered ports to exactly that owner order; a
    model without ports keeps the size checks alone.
    """

    model: AbstractArrayModel
    ports: ModelPorts | None
    port_mapping: PortMapping | None
    kind: MeshProposerKind = eqx.field(static=True)
    spatial_dimension: int | None = eqx.field(static=True)
    proposer_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        model: AbstractArrayModel,
        /,
        *,
        kind: MeshProposerKind,
        proposer_id: str,
        spatial_dimension: int | None = None,
        ports: ModelPorts | None = None,
        port_mapping: PortMapping | None = None,
    ) -> None:
        kind = parse(kind, MeshProposerKind, "kind")
        value_shape: tuple[int, ...]
        match kind:
            case "marking" | "size":
                if spatial_dimension is not None:
                    raise ValueError(
                        f"LearnedMeshProposer.spatial_dimension must be absent for scalar proposers; got {(kind, spatial_dimension)!r}"
                    )
                value_shape = ()
            case "metric" | "coordinate":
                if (
                    isinstance(spatial_dimension, bool)
                    or not isinstance(spatial_dimension, int)
                    or spatial_dimension <= 0
                ):
                    raise ValueError(
                        f"LearnedMeshProposer.spatial_dimension must be a positive integer for metric or coordinate proposals; got {spatial_dimension!r}"
                    )
                value_shape = (
                    (spatial_dimension, spatial_dimension)
                    if kind == "metric"
                    else (spatial_dimension,)
                )
            case _:
                raise ValueError(
                    f"LearnedMeshProposer.kind must select a supported proposal kind; got {kind!r}"
                )
        binding = model.input_binding()
        if binding.batch_mode != "pointwise" or binding.input_mode != "flat":
            raise ValueError(
                f"LearnedMeshProposer.model.input_binding must be pointwise and flat; got {(binding.batch_mode, binding.input_mode)!r}"
            )
        if isinstance(model.in_size, bool) or not isinstance(model.in_size, int):
            raise ValueError(
                f"LearnedMeshProposer.model.in_size must be an integer feature width; got {model.in_size!r}"
            )
        if _model_value_shape(model.out_size) != value_shape:
            raise ValueError(
                f"LearnedMeshProposer.model.out_size must produce the declared proposal value shape; got {(model.out_size, value_shape, kind)!r}"
            )
        identifier = str(proposer_id).strip()
        if not identifier:
            raise ValueError(
                f"LearnedMeshProposer.proposer_id must be nonempty; got {proposer_id!r}"
            )
        site = "LearnedMeshProposer"
        owner_ports = _dependency_phydrax__model__ports.require_port_shapes(
            ports, inputs=((model.in_size,),), outputs=(value_shape,), site=site
        )
        _dependency_phydrax__model__component.bind_positional_component(
            model, AbstractMeshProposer, owner_ports, port_mapping, site=site
        )
        self.model = model
        self.ports = owner_ports
        self.port_mapping = port_mapping
        self.kind = kind
        self.spatial_dimension = spatial_dimension
        self.proposer_id = identifier

    def component_contract(self) -> ComponentContract:
        """Return the model's contract bound to the mesh-proposer slot."""
        return bind_component(
            self.model,
            AbstractMeshProposer,
            owner_ports=self.ports,
            port_mapping=self.port_mapping,
        ).contract()

    @property
    def model_id(self) -> str:
        """Bind current numerical state to the authored proposer specification."""
        return canonical_fingerprint(
            {
                "kind": "learned-mesh-proposer-model",
                "proposer": self.proposer_id,
                "model_type": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
                "kind_of_proposal": self.kind,
                "spatial_dimension": self.spatial_dimension,
                "input_size": self.model.in_size,
                "output_shape": _model_value_shape(self.model.out_size),
                "component_contract": self.component_contract().bound_semantic_id,
                "parameters_and_state": array_tree_fingerprint(self.model),
            }
        )

    def evaluate(self, features: ArrayLike, /) -> Array:
        """Return one proposal value per feature row, shape `(rows,) + value shape`."""
        rows = jnp.asarray(features)
        if rows.ndim != 2 or rows.shape[1] != self.model.in_size:
            raise ValueError(
                f"LearnedMeshProposer.features must have rank two and the declared model feature width; got {(rows.shape, self.model.in_size)!r}"
            )
        binding = self.model.input_binding()
        return jax.vmap(
            lambda row: binding.call(self.model, row, key=None, iter_=None, kwargs={})
        )(rows)

    @checked
    def evaluate_addressed(
        self,
        features: ArrayLike,
        entity_ids: ArrayLike,
        key: PRNGKey,
        /,
        *,
        address_id: str,
    ) -> Array:
        """Evaluate with reproducible keys independent of partition and row order.

        Both halves of each signed 64-bit scientific ID are folded in, avoiding
        collisions between IDs that differ only in their high bits. The authored
        address binds the source revision and feature owner, not the local batch.
        """
        rows = jnp.asarray(features)
        identifiers = jnp.asarray(entity_ids)
        if rows.ndim != 2 or rows.shape[1] != self.model.in_size:
            raise ValueError(
                f"LearnedMeshProposer.features must have rank two and the declared model feature width; got {(rows.shape, self.model.in_size)!r}"
            )
        if identifiers.ndim != 1 or identifiers.shape[0] != rows.shape[0]:
            raise ValueError(
                f"LearnedMeshProposer.entity_ids must have rank one and align with feature rows; got {(identifiers.shape, rows.shape)!r}"
            )
        if identifiers.dtype != jnp.dtype("int64"):
            raise ValueError(
                f"LearnedMeshProposer.entity_ids must use signed 64-bit storage; got {identifiers.dtype!r}"
            )
        if not isinstance(address_id, str) or not address_id:
            raise ValueError(
                f"LearnedMeshProposer.address_id must identify the scientific evaluation; got {address_id!r}"
            )
        digest = canonical_fingerprint(
            {"kind": "mesh-proposal-random-address", "address": address_id}
        )
        addressed = key
        for offset in range(0, len(digest), 8):
            addressed = jax.random.fold_in(
                addressed, int(digest[offset : offset + 8], 16)
            )
        unsigned = identifiers.astype(jnp.uint64)
        low = unsigned.astype(jnp.uint32)
        high = (unsigned >> jnp.uint64(32)).astype(jnp.uint32)
        binding = self.model.input_binding()

        def evaluate_row(row: Array, low_id: Array, high_id: Array) -> Array:
            entity_key = jax.random.fold_in(
                jax.random.fold_in(addressed, high_id), low_id
            )
            return binding.call(self.model, row, key=entity_key, iter_=None, kwargs={})

        return jax.vmap(evaluate_row)(rows, low, high)

    @checked
    def propose(
        self,
        source: _dependency_phydrax_meshing__result.CellMeshingResult,
        features: MeshProposalFeatures,
        /,
        *,
        scope: _dependency_phydrax_meshing__scope.MeshingScope | None = None,
        key: PRNGKey | None = None,
    ) -> MeshProposal:
        if features.source_result_id != source.result_id:
            raise ValueError(
                f"MeshProposalFeatures.source_result_id must bind the current source result revision; got {(features.source_result_id, source.result_id)!r}"
            )
        if (
            self.spatial_dimension is not None
            and self.spatial_dimension != source.mesh.ambient_dimension
        ):
            raise ValueError(
                f"LearnedMeshProposer.spatial_dimension must match source coordinate axes; got {(self.spatial_dimension, source.mesh.ambient_dimension)!r}"
            )
        dimension = source.mesh.topological_dimension if self.kind == "marking" else 0
        scope_ = features.scope if scope is None else scope
        _scope_rows(source, scope_)
        if (
            scope_.scope_id != features.scope.scope_id
            or scope_.entity_dimension != dimension
        ):
            raise ValueError(
                f"MeshProposalFeatures.scope must bind the requested proposal entities; got {(scope_.scope_id, features.scope.scope_id, scope_.entity_dimension, dimension)!r}"
            )
        rows = features.values
        if rows.ndim != 2 or rows.shape[0] != scope_.entity_ids.size:
            raise ValueError(
                f"MeshProposalFeatures.values must hold one row per scope entity in sorted global-ID order; got {(rows.shape, scope_.entity_ids.size)!r}"
            )
        model_id = self.model_id
        address = canonical_fingerprint(
            {
                "kind": "learned-mesh-proposal-evaluation",
                "source": source.result_id,
                "feature_owner": features.feature_owner_id,
                "columns": features.feature_ids,
                "proposer": model_id,
            }
        )
        values = np.asarray(
            self.evaluate(rows)
            if key is None
            else self.evaluate_addressed(rows, scope_.entity_ids, key, address_id=address)
        )
        evaluation_id = canonical_fingerprint(
            {
                "kind": "learned-mesh-proposal",
                "model": model_id,
                "features": features.feature_id,
                "address": address,
                "realization": None
                if key is None
                else array_tree_fingerprint(jax.random.key_data(key)),
            }
        )
        match self.kind:
            case "marking":
                return MeshMarkingProposal(
                    source,
                    scope_,
                    values,
                    proposer_id=self.proposer_id,
                    evaluation_id=evaluation_id,
                )
            case "size":
                return MeshSizeProposal(
                    source,
                    scope_,
                    values,
                    proposer_id=self.proposer_id,
                    evaluation_id=evaluation_id,
                )
            case "metric":
                return MeshMetricProposal(
                    source,
                    scope_,
                    values,
                    proposer_id=self.proposer_id,
                    evaluation_id=evaluation_id,
                )
            case "coordinate":
                return MeshCoordinateProposal(
                    source,
                    scope_,
                    values,
                    source.coordinate_contract,
                    proposer_id=self.proposer_id,
                    evaluation_id=evaluation_id,
                )
            case _:
                raise ValueError(
                    f"LearnedMeshProposer.kind must select a supported proposal kind; got {self.kind!r}"
                )


class MeshProposalSafetyPolicy(StrictModule, NonTrainableState):
    """Trusted, revision-bound constraints, independent of proposer evidence.

    Protected scopes preserve their entities and fix their incident vertices.
    Coordinate bounds and displacement are in source coordinate units. Limits
    are admission limits on candidate payloads, not a process memory quota or
    a preemptive timeout. Wall-time overruns cannot be committed.
    """

    source_result_id: str = eqx.field(static=True)
    protected_scopes: tuple[_dependency_phydrax_meshing__scope.MeshingScope, ...]
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits
    audit_policy: _dependency_phydrax_meshing__audit.CellMeshAuditPolicy
    coordinate_bounds: Array | None
    minimum_size: float = eqx.field(static=True)
    maximum_size: float = eqx.field(static=True)
    maximum_anisotropy: float = eqx.field(static=True)
    maximum_gradation: float = eqx.field(static=True)
    maximum_displacement: float = eqx.field(static=True)
    maximum_marked_cells: int = eqx.field(static=True)
    maximum_optimization_iterations: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: _dependency_phydrax_meshing__result.CellMeshingResult,
        /,
        *,
        minimum_size: float,
        maximum_size: float,
        maximum_displacement: float,
        protected_scopes: tuple[
            _dependency_phydrax_meshing__scope.MeshingScope, ...
        ] = (),
        limits: _dependency_phydrax_meshing__contracts.MeshingLimits | None = None,
        audit_policy: _dependency_phydrax_meshing__audit.CellMeshAuditPolicy
        | None = None,
        coordinate_bounds: ArrayLike | None = None,
        maximum_anisotropy: float = 100.0,
        maximum_gradation: float = 1.3,
        maximum_marked_cells: int = 100_000,
        maximum_optimization_iterations: int = 50,
    ) -> None:
        minimum, maximum = float(minimum_size), float(maximum_size)
        anisotropy, gradation = float(maximum_anisotropy), float(maximum_gradation)
        displacement = float(maximum_displacement)
        if not np.all(
            np.isfinite((minimum, maximum, anisotropy, gradation, displacement))
        ):
            raise ValueError(
                f"MeshProposalSafetyPolicy.bounds must be finite; got {(minimum, maximum, anisotropy, gradation, displacement)!r}"
            )
        if (
            minimum <= 0
            or maximum < minimum
            or anisotropy < 1
            or gradation < 1
            or displacement < 0
        ):
            raise ValueError(
                f"MeshProposalSafetyPolicy.bounds must have positive ordered sizes, anisotropy and gradation at least one, and nonnegative displacement; got {(minimum, maximum, anisotropy, gradation, displacement)!r}"
            )
        normal_floor = np.sqrt(np.finfo(np.float64).tiny)
        if (
            minimum < normal_floor
            or maximum > 1.0 / normal_floor
            or anisotropy > 1.0 / normal_floor
        ):
            raise ValueError(
                f"MeshProposalSafetyPolicy.bounds must admit representable metric eigenvalues; got {(minimum, maximum, anisotropy, normal_floor)!r}"
            )
        marked, iterations = (
            int(maximum_marked_cells),
            int(maximum_optimization_iterations),
        )
        if marked < 0 or iterations <= 0:
            raise ValueError(
                f"MeshProposalSafetyPolicy.capacity must have nonnegative mark capacity and positive optimization iterations; got {(marked, iterations)!r}"
            )
        protected = tuple(protected_scopes)
        for scope in protected:
            _scope_rows(source, scope)
        limits_ = (
            _dependency_phydrax_meshing__contracts.MeshingLimits()
            if limits is None
            else limits
        )
        audit_ = (
            _dependency_phydrax_meshing__audit.CellMeshAuditPolicy()
            if audit_policy is None
            else audit_policy
        )
        bounds = (
            None
            if coordinate_bounds is None
            else np.asarray(coordinate_bounds, dtype=np.float64)
        )
        if bounds is not None:
            if (
                bounds.shape != (2, source.mesh.ambient_dimension)
                or not np.all(np.isfinite(bounds))
                or np.any(bounds[0] > bounds[1])
            ):
                raise ValueError(
                    f"MeshProposalSafetyPolicy.coordinate_bounds must contain finite ordered lower and upper vectors; got {(bounds.shape, bounds, source.mesh.ambient_dimension)!r}"
                )
            if np.any(source.mesh.coordinates < bounds[0]) or np.any(
                source.mesh.coordinates > bounds[1]
            ):
                raise ValueError(
                    f"MeshProposalSafetyPolicy.coordinate_bounds must contain the source coordinates; got {(bounds, np.asarray(source.mesh.coordinates).min(axis=0), np.asarray(source.mesh.coordinates).max(axis=0))!r}"
                )
        self.source_result_id = source.result_id
        self.protected_scopes = protected
        self.limits = limits_
        self.audit_policy = audit_
        _host_coordinate_bounds = None if bounds is None else np.asarray(bounds)
        (self.coordinate_bounds,) = jax.device_put((_host_coordinate_bounds,))
        self.minimum_size, self.maximum_size = minimum, maximum
        self.maximum_anisotropy, self.maximum_gradation = anisotropy, gradation
        self.maximum_displacement = displacement
        self.maximum_marked_cells = marked
        self.maximum_optimization_iterations = iterations
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mesh-proposal-safety-policy",
                "source": source.result_id,
                "protected": tuple(scope.scope_id for scope in protected),
                "limits": limits_.limits_id,
                "audit": audit_.policy_id,
                "bounds": None
                if bounds is None
                else array_tree_fingerprint(self.coordinate_bounds),
                "minimum_size": minimum,
                "maximum_size": maximum,
                "anisotropy": anisotropy,
                "gradation": gradation,
                "displacement": displacement,
                "marks": marked,
                "iterations": iterations,
            }
        )


def _entity_closure(
    mesh: CellMesh, dimension: int, rows: np.ndarray, target: int
) -> np.ndarray:
    for degree in range(dimension, target, -1):
        relation = mesh.topology.incidences[degree - 1].relation
        valid = np.asarray(relation.valid) & np.isin(
            np.asarray(relation.target_indices), rows
        )
        rows = np.unique(np.asarray(relation.source_indices)[valid])
    return rows


def _protected_vertices(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    policy: MeshProposalSafetyPolicy,
) -> np.ndarray:
    fixed = np.zeros(source.mesh.coordinates.shape[0], dtype=np.bool_)
    for scope in policy.protected_scopes:
        rows = _entity_closure(
            source.mesh, scope.entity_dimension, _scope_rows(source, scope), 0
        )
        fixed[rows] = True
    return fixed


def _payload_bytes(value: StrictModule | None) -> int:
    return sum(
        leaf.nbytes
        for leaf in jax.tree_util.tree_leaves(value)
        if isinstance(leaf, (jax.Array, np.ndarray))
    )


def _limit_issues(
    result: _dependency_phydrax_meshing__result.CellMeshingResult,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
) -> tuple[str, ...]:
    counts = result.audit.entity_counts
    observations = (
        ("vertices", result.audit.vertex_count, limits.maximum_vertices),
        ("edges", counts[1] if len(counts) > 1 else 0, limits.maximum_edges),
        ("faces", counts[2] if len(counts) > 2 else 0, limits.maximum_faces),
        ("cells", counts[-1], limits.maximum_cells),
        (
            "connectivity_entries",
            result.audit.connectivity_entries,
            limits.maximum_connectivity_entries,
        ),
        ("data_bytes", _payload_bytes(result), limits.maximum_data_bytes),
    )
    return tuple(
        f"maximum_{name}" for name, actual, maximum in observations if actual > maximum
    )


def _check_binding(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    proposal: MeshProposal,
    policy: MeshProposalSafetyPolicy,
) -> None:
    _scope_rows(source, proposal.scope)
    if policy.source_result_id != source.result_id:
        raise ValueError(
            f"MeshProposalSafetyPolicy.source_result_id must bind the current source result revision; got {(policy.source_result_id, source.result_id)!r}"
        )
    if proposal.coordinate_contract_id != source.coordinate_contract.spatial_id:
        raise ValueError(
            f"MeshProposal.coordinate_contract_id must match the source coordinate contract; got {(proposal.coordinate_contract_id, source.coordinate_contract.spatial_id)!r}"
        )
    if source.coordinate_contract.coordinate_system != "cartesian":
        raise ValueError(
            f"MeshProposal.source.coordinate_system must be Cartesian; got {source.coordinate_contract.coordinate_system!r}"
        )
    source.audit.require_passed()
    if not source.compliance.passed or _limit_issues(source, policy.limits):
        raise ValueError(
            f"MeshProposal.source.compliance must satisfy proposal compliance and capacity limits; got {(source.compliance.issues, policy.limits.limits_id)!r}"
        )
    for scope in policy.protected_scopes:
        _scope_rows(source, scope)


def _project_sizes(
    values: np.ndarray,
    points: np.ndarray,
    edges: np.ndarray,
    policy: MeshProposalSafetyPolicy,
) -> np.ndarray:
    """Clamp to the size bounds, then grade through the metric owner."""
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    graded, _, _ = _dependency_phydrax_meshing__metric.grade_scalar_sizes(
        np.clip(values, policy.minimum_size, policy.maximum_size),
        edges,
        lengths,
        np.full((edges.shape[0],), policy.maximum_gradation),
        _dependency_phydrax_meshing__metric.MetricGradationKind.PHYSICAL,
    )
    return graded


def _project_metric(
    scope: _dependency_phydrax_meshing__scope.MeshingScope,
    raw: np.ndarray,
    points: np.ndarray,
    edges: np.ndarray,
    policy: MeshProposalSafetyPolicy,
) -> tuple[
    _dependency_phydrax_meshing__metric.MeshMetricField,
    _dependency_phydrax_meshing__metric.MetricNormalizationEvidence,
]:
    # Untrusted tensors: the trusted policy explicitly requests symmetric-part
    # repair and indefinite projection, and the evidence reports both.
    return _dependency_phydrax_meshing__metric.normalize_mesh_metric(
        _dependency_phydrax_meshing__metric.MeshMetricSamples(scope, raw),
        policy=_dependency_phydrax_meshing__metric.MetricNormalizationPolicy(
            minimum_size=policy.minimum_size,
            maximum_size=policy.maximum_size,
            maximum_anisotropy=policy.maximum_anisotropy,
            gradation=_dependency_phydrax_meshing__metric.MetricGradationPolicy(
                policy.maximum_gradation
            ),
            symmetrize=True,
            project_indefinite=True,
        ),
        adjacency=edges,
        coordinates=points,
    )


def _coordinate_projector(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    proposal: MeshCoordinateProposal,
    policy: MeshProposalSafetyPolicy,
) -> tuple[Array, Callable[[Array], Array]]:
    points = jnp.asarray(source.mesh.coordinates)
    movable = np.zeros(points.shape[0], dtype=np.bool_)
    movable[_scope_rows(source, proposal.scope)] = True
    fixed = jnp.asarray(~movable | _protected_vertices(source, policy))
    bounds = policy.coordinate_bounds

    def project(values: Array) -> Array:
        if bounds is not None:
            values = jnp.clip(values, bounds[0], bounds[1])
        delta = values - points
        lengths = jnp.linalg.norm(delta, axis=1, keepdims=True)
        factor = jnp.minimum(
            1.0,
            policy.maximum_displacement
            / jnp.maximum(lengths, jnp.finfo(points.dtype).tiny),
        )
        return jnp.where(fixed[:, None], points, points + delta * factor)

    return fixed, project


def _optimization_bounds(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    policy: MeshProposalSafetyPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(source.mesh.coordinates)
    # A per-axis half-width r / sqrt(d) inscribes the box in the trust-region
    # ball, so box projection can never exceed maximum_displacement.
    radius = policy.maximum_displacement / math.sqrt(points.shape[1])
    lower, upper = points - radius, points + radius
    if policy.coordinate_bounds is not None:
        bounds = np.asarray(policy.coordinate_bounds)
        lower, upper = np.maximum(lower, bounds[0]), np.minimum(upper, bounds[1])
    # A source coordinate outside the coordinate bounds cannot move on that axis.
    empty = lower > upper
    return np.where(empty, points, lower), np.where(empty, points, upper)


def _safe_marks(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    scores: np.ndarray,
    policy: MeshProposalSafetyPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the selected and the protection-withheld cell IDs, both sorted.

    Selection takes the highest positive scores, then global ID, within the
    declared capacity. Positive-score cells in a protected cell scope are
    withheld from the route and returned as evidence; closure belongs to
    the route.
    """
    cells = np.concatenate([np.asarray(block.global_ids) for block in source.mesh.blocks])
    order = np.lexsort((cells, -scores))
    eligible = scores > 0.0
    protected = np.zeros_like(eligible)
    for scope in policy.protected_scopes:
        if scope.entity_dimension == source.mesh.topological_dimension:
            protected[_scope_rows(source, scope)] = True
    withheld = np.sort(cells[eligible & protected]).astype(np.int64, copy=False)
    eligible &= ~protected
    capacity = min(
        policy.maximum_marked_cells,
        max(0, policy.limits.maximum_cells - source.audit.entity_counts[-1]),
    )
    # Preserve the planar projection's established allocation bound. In 3D,
    # conformity closure and connectivity admission are owned by native bisection;
    # planar edge/face growth constants are not tetrahedral resource evidence.
    if source.mesh.topological_dimension == 2:
        counts = source.audit.entity_counts
        capacity = min(
            capacity,
            max(0, policy.limits.maximum_vertices - counts[0]),
            max(0, (policy.limits.maximum_cells - counts[2]) // 2),
            max(0, (policy.limits.maximum_faces - counts[2]) // 2),
            max(0, (policy.limits.maximum_edges - counts[1]) // 3),
            max(
                0,
                (
                    policy.limits.maximum_connectivity_entries
                    - source.audit.connectivity_entries
                )
                // 18,
            ),
        )
    selected = order[eligible[order]][:capacity]
    return np.sort(cells[selected]).astype(np.int64, copy=False), withheld


class MeshProposalProjection(StrictModule, NonTrainableState):
    """Projected proposal evidence; this is not a mesh or a certification.

    ``protected_cell_ids`` lists positive-score marks withheld from the route
    because they lie in a protected cell scope.
    """

    proposal: MeshProposal
    policy: MeshProposalSafetyPolicy
    marked_cell_ids: Array
    protected_cell_ids: Array
    size_field: _dependency_phydrax_meshing__sizing.ResolvedSizeField | None
    metric: _dependency_phydrax_meshing__metric.MeshMetricField | None
    metric_evidence: (
        _dependency_phydrax_meshing__metric.MetricNormalizationEvidence | None
    )
    target_coordinates: Array | None
    projection_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        proposal: MeshProposal,
        policy: MeshProposalSafetyPolicy,
        marked_cell_ids: ArrayLike,
        size_field: _dependency_phydrax_meshing__sizing.ResolvedSizeField | None,
        metric: _dependency_phydrax_meshing__metric.MeshMetricField | None,
        target_coordinates: Array | None,
        *,
        protected_cell_ids: ConvertibleToArray = (),
        metric_evidence: _dependency_phydrax_meshing__metric.MetricNormalizationEvidence
        | None = None,
    ) -> None:
        if (metric is None) != (metric_evidence is None):
            raise ValueError(
                f"MeshProposalProjection.metric_evidence must be present exactly when a projected metric is present; got {(metric is not None, metric_evidence is not None)!r}"
            )
        self.proposal, self.policy = proposal, policy
        _host_marked_cell_ids = np.asarray(marked_cell_ids, dtype=np.int64)
        _host_protected_cell_ids = np.asarray(protected_cell_ids, dtype=np.int64)
        (
            self.marked_cell_ids,
            self.protected_cell_ids,
        ) = jax.device_put(
            (
                _host_marked_cell_ids,
                _host_protected_cell_ids,
            )
        )
        self.size_field, self.metric = size_field, metric
        self.metric_evidence = metric_evidence
        self.target_coordinates = target_coordinates
        self.projection_id = canonical_fingerprint(
            {
                "kind": "mesh-proposal-projection",
                "proposal": proposal.proposal_id,
                "policy": policy.policy_id,
                "marks": array_tree_fingerprint(self.marked_cell_ids),
                "protected_marks": array_tree_fingerprint(self.protected_cell_ids),
                "size": None if size_field is None else size_field.field_id,
                "metric": None if metric is None else metric.metric_id,
                "metric_evidence": None
                if metric_evidence is None
                else metric_evidence.evidence_id,
                "coordinates": None
                if target_coordinates is None
                else array_tree_fingerprint(target_coordinates),
            }
        )


@checked
def project_mesh_proposal(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    proposal: MeshProposal,
    policy: MeshProposalSafetyPolicy,
    /,
) -> MeshProposalProjection:
    """Deterministically project untrusted values without generating a mesh.

    Marking scores project onto protected, capacity-bounded cell IDs; the selected
    native family route owns connectivity and conformity closure. Simplex size
    and metric proposals project onto a graded field or normalized tensors for
    the admitted planar or tetrahedral unit-mesh route.
    """
    _check_binding(source, proposal, policy)
    mesh = source.mesh
    rows = _scope_rows(source, proposal.scope)
    sizes, metric, metric_evidence, target = None, None, None, None
    marks = np.empty(0, dtype=np.int64)
    withheld = np.empty(0, dtype=np.int64)
    match proposal:
        case MeshCoordinateProposal():
            _, project = _coordinate_projector(source, proposal, policy)
            target = project(jnp.asarray(mesh.coordinates).at[rows].set(proposal.values))
            # A safe target must itself be a valid optimization reference. Backtrack
            # toward the certified source, rather than accepting an inverted target.
            for _ in range(64):
                if bool(
                    jnp.all(
                        _dependency_phydrax_meshing__quality.evaluate_cell_quality(
                            mesh, target
                        ).sampled_valid
                    )
                ):
                    break
                target = project(0.5 * (target + mesh.coordinates))
            else:
                raise ValueError(
                    f"MeshCoordinateProposal.target_coordinates must admit a certifiable projected target within 64 backtracking checks; got {target.shape!r}"
                )
        case MeshMarkingProposal():
            scores = np.zeros(source.audit.entity_counts[-1], dtype=np.float64)
            scores[rows] = np.asarray(proposal.values)
            marks, withheld = _safe_marks(source, scores, policy)
        case MeshSizeProposal() | MeshMetricProposal():
            kinds = {block.cell_kind for block in mesh.blocks}
            if kinds != {"triangle"} and kinds != {"tetrahedron"}:
                raise ValueError(
                    f"MeshProposal.source.cell_kind must contain simplex blocks for size or metric projection; got {tuple(sorted(kinds))!r}"
                )
            if rows.size != mesh.coordinates.shape[0]:
                raise ValueError(
                    f"MeshProposal.scope.entity_ids must cover every source vertex for size or metric projection; got {(rows.size, mesh.coordinates.shape[0])!r}"
                )
            # Field arrays are scope ordered; connectivity and points are mesh ordered.
            inverse = np.empty(rows.size, dtype=np.int64)
            inverse[rows] = np.arange(rows.size)
            connectivity = mesh.connectivity
            match connectivity:
                case (
                    PolygonalConnectivity()
                    | TetrahedralConnectivity()
                    | PolyhedralConnectivity()
                ):
                    pass
                case _:
                    raise TypeError(
                        f"MeshProposal.source.connectivity must provide declared simplex edge connectivity; got {type(connectivity).__name__!r}"
                    )
            field_edges = inverse[np.asarray(connectivity.edges)]
            field_points = np.asarray(mesh.coordinates)[rows]
            match proposal:
                case MeshSizeProposal():
                    sizes = _dependency_phydrax_meshing__sizing.ResolvedSizeField(
                        _dependency_phydrax_meshing__sizing.SizeFieldDomain.SAMPLE_CLOUD,
                        field_points,
                        _project_sizes(
                            np.asarray(proposal.values), field_points, field_edges, policy
                        ),
                        sample_entity_ids=proposal.scope.entity_ids,
                        source_control_ids=(proposal.proposal_id, policy.policy_id),
                    )
                case MeshMetricProposal():
                    metric, metric_evidence = _project_metric(
                        proposal.scope,
                        np.asarray(proposal.values),
                        field_points,
                        field_edges,
                        policy,
                    )
                case _:
                    raise TypeError(
                        f"MeshProposalProjection.proposal must be a supported typed mesh proposal; got {type(proposal).__name__!r}"
                    )
        case _:
            raise TypeError(
                f"MeshProposalProjection.proposal must be a supported typed mesh proposal; got {type(proposal).__name__!r}"
            )
    return MeshProposalProjection(
        proposal,
        policy,
        marks,
        sizes,
        metric,
        target,
        protected_cell_ids=withheld,
        metric_evidence=metric_evidence,
    )


def _entity_vertex_signatures(
    mesh: CellMesh, dimension: int
) -> tuple[tuple[int, ...], ...]:
    vertices = [{int(identifier)} for identifier in np.asarray(mesh.vertex_global_ids)]
    for degree in range(1, dimension + 1):
        upper = [set() for _ in range(mesh.entity_set(degree).count)]
        relation = mesh.topology.incidences[degree - 1].relation
        valid = np.asarray(relation.valid)
        for lower_row, upper_row in zip(
            np.asarray(relation.source_indices)[valid],
            np.asarray(relation.target_indices)[valid],
            strict=True,
        ):
            upper[int(upper_row)].update(vertices[int(lower_row)])
        vertices = upper
    return tuple(tuple(sorted(values)) for values in vertices)


def _preservation_issues(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    candidate: _dependency_phydrax_meshing__result.CellMeshingResult,
    projection: MeshProposalProjection,
) -> tuple[str, ...]:
    policy = projection.policy
    issues = []
    if candidate.coordinate_contract.spatial_id != source.coordinate_contract.spatial_id:
        issues.append("coordinate_contract")
    fixed = _protected_vertices(source, policy)
    source_ids = np.asarray(source.mesh.vertex_global_ids)
    target_ids = np.asarray(candidate.mesh.vertex_global_ids)
    lookup = {int(identifier): row for row, identifier in enumerate(target_ids)}
    for row in np.flatnonzero(fixed):
        target_row = lookup.get(int(source_ids[row]))
        if target_row is None or not np.array_equal(
            source.mesh.coordinates[row], candidate.mesh.coordinates[target_row]
        ):
            issues.append("protected_vertices")
            break
    signatures = {}
    for scope in policy.protected_scopes:
        dimension = scope.entity_dimension
        if dimension == 0 or source.mesh.topology_id == candidate.mesh.topology_id:
            continue
        # IDs alone do not establish preservation: intermediate entity IDs may
        # be regenerated by refinement. Compare the incident vertex identities.
        source_rows = _scope_rows(source, scope)
        if dimension not in signatures:
            signatures[dimension] = (
                _entity_vertex_signatures(source.mesh, dimension),
                _entity_vertex_signatures(candidate.mesh, dimension),
            )
        source_signatures, target_signatures = signatures[dimension]
        target_entities = candidate.mesh.entity_set(scope.entity_dimension)
        target_lookup = {
            int(identifier): row
            for row, identifier in enumerate(np.asarray(target_entities.entity_ids))
        }
        for wanted, row in zip(np.asarray(scope.entity_ids), source_rows, strict=True):
            target_row = target_lookup.get(int(wanted))
            if target_row is None:
                issues.append("protected_entities")
                break
            if source_signatures[int(row)] != target_signatures[target_row]:
                issues.append("protected_entities")
                break
    bounds = policy.coordinate_bounds
    if bounds is not None and (
        np.any(candidate.mesh.coordinates < bounds[0])
        or np.any(candidate.mesh.coordinates > bounds[1])
    ):
        issues.append("coordinate_bounds")
    if isinstance(projection.proposal, MeshCoordinateProposal):
        if candidate.mesh.topology_id != source.mesh.topology_id:
            issues.append("fixed_topology")
        else:
            delta = np.asarray(candidate.mesh.coordinates) - np.asarray(
                source.mesh.coordinates
            )
            tolerance = (
                16 * np.finfo(delta.dtype).eps * max(1.0, policy.maximum_displacement)
            )
            if np.any(
                np.linalg.norm(delta, axis=1) > policy.maximum_displacement + tolerance
            ):
                issues.append("maximum_displacement")
            movable = np.zeros(delta.shape[0], dtype=np.bool_)
            movable[_scope_rows(source, projection.proposal.scope)] = True
            if np.any(delta[~movable] != 0):
                issues.append("coordinate_scope")
    return tuple(dict.fromkeys(issues))


class MeshProposalTransaction(StrictModule, NonTrainableState):
    """Prepared trusted result and separate safety evidence; source stays intact.

    Marking, size, and metric proposals expose their native `MeshAdaptationResult`
    (transition, lineage, and sparse FE transfer) for the solver's
    FiniteElementTopologyTransaction; commit here promotes only a mesh, never
    solution fields. Explicit rejection returns the identical source.
    """

    source: _dependency_phydrax_meshing__result.CellMeshingResult
    projection: MeshProposalProjection
    trusted_result: _dependency_phydrax_meshing__result.CellMeshingResult
    safety_audit: _dependency_phydrax_meshing__audit.CellMeshAuditReport
    compliance: _dependency_phydrax_meshing__result.MeshingComplianceReport
    adaptation: _dependency_phydrax_meshing__adaptation.MeshAdaptationResult | None
    optimization: _dependency_phydrax_meshing__optimization.MeshOptimizationResult | None
    transaction_id: str = eqx.field(static=True)

    @staticmethod
    @checked
    def coordinate_candidate(
        source: _dependency_phydrax_meshing__result.CellMeshingResult,
        optimization: _dependency_phydrax_meshing__optimization.MeshOptimizationResult,
        /,
    ) -> _dependency_phydrax_meshing__result.CellMeshingResult:
        """Select an accepted optimization mesh or retain the rejected source."""
        match (optimization.accepted, optimization.result):
            case (False, _):
                return source
            case (
                True,
                _dependency_phydrax_meshing__result.CellMeshingResult() as result,
            ):
                return result
            case _:
                raise ValueError(
                    f"MeshOptimizationResult.result must be present for an accepted coordinate proposal; got {(optimization.status, type(optimization.result).__name__)!r}"
                )

    @checked
    def __init__(
        self,
        source: _dependency_phydrax_meshing__result.CellMeshingResult,
        projection: MeshProposalProjection,
        trusted_result: _dependency_phydrax_meshing__result.CellMeshingResult,
        safety_audit: _dependency_phydrax_meshing__audit.CellMeshAuditReport,
        compliance: _dependency_phydrax_meshing__result.MeshingComplianceReport,
        adaptation: _dependency_phydrax_meshing__adaptation.MeshAdaptationResult | None,
        optimization: _dependency_phydrax_meshing__optimization.MeshOptimizationResult
        | None,
    ) -> None:
        _check_binding(source, projection.proposal, projection.policy)
        if (
            safety_audit.mesh_id != trusted_result.mesh.mesh_id
            or safety_audit.policy_id != projection.policy.audit_policy.policy_id
        ):
            raise ValueError(
                f"MeshProposalTransaction.safety_audit must bind the candidate mesh and safety policy; got {(safety_audit.mesh_id, trusted_result.mesh.mesh_id, safety_audit.policy_id, projection.policy.audit_policy.policy_id)!r}"
            )
        if compliance.specification_id != projection.projection_id:
            raise ValueError(
                f"MeshProposalTransaction.compliance must bind the projected proposal; got {(compliance.specification_id, projection.projection_id)!r}"
            )
        if adaptation is not None and (
            adaptation.source.result_id != source.result_id
            or adaptation.target.result_id != trusted_result.result_id
        ):
            raise ValueError(
                f"MeshProposalTransaction.adaptation must bind the source and candidate result revisions; got {(adaptation.source.result_id, source.result_id, adaptation.target.result_id, trusted_result.result_id)!r}"
            )
        match projection.proposal:
            case MeshCoordinateProposal():
                if (
                    optimization is None
                    or adaptation is not None
                    or self.coordinate_candidate(source, optimization).result_id
                    != trusted_result.result_id
                ):
                    raise ValueError(
                        f"MeshProposalTransaction.trusted_result must match the coordinate optimization candidate; got {(trusted_result.result_id, optimization is None, adaptation is not None)!r}"
                    )
            case MeshMarkingProposal() if projection.marked_cell_ids.size == 0:
                if adaptation is not None or trusted_result.result_id != source.result_id:
                    raise ValueError(
                        f"MeshProposalTransaction.trusted_result must retain the source for an empty projected marking; got {(adaptation is not None, trusted_result.result_id, source.result_id)!r}"
                    )
            case MeshMarkingProposal() | MeshSizeProposal() | MeshMetricProposal():
                if adaptation is None or optimization is not None:
                    raise ValueError(
                        f"MeshProposalTransaction.adaptation must provide trusted adaptation evidence without coordinate optimization; got {(adaptation is None, optimization is not None)!r}"
                    )
            case _:
                raise TypeError(
                    f"MeshProposalTransaction.proposal must be a supported typed mesh proposal; got {type(projection.proposal).__name__!r}"
                )
        self.source, self.projection, self.trusted_result = (
            source,
            projection,
            trusted_result,
        )
        self.safety_audit, self.compliance = safety_audit, compliance
        self.adaptation, self.optimization = adaptation, optimization
        self.transaction_id = canonical_fingerprint(
            {
                "kind": "mesh-proposal-transaction",
                "source": source.result_id,
                "projection": projection.projection_id,
                "result": trusted_result.result_id,
                "audit": safety_audit.report_id,
                "compliance": compliance.report_id,
                "adaptation": None if adaptation is None else adaptation.result_id,
                "optimization": None
                if optimization is None
                else optimization.optimization_id,
            }
        )

    @property
    def admissible(self) -> bool:
        return self.safety_audit.passed and self.compliance.passed

    @checked
    def commit(
        self,
        current: _dependency_phydrax_meshing__result.CellMeshingResult,
        /,
        *,
        accept: bool = True,
    ) -> _dependency_phydrax_meshing__result.CellMeshingResult:
        if not isinstance(accept, (bool, np.bool_)):
            raise TypeError(
                f"MeshProposalTransaction.accept must be an explicit host boolean decision; got {type(accept).__name__!r}"
            )
        if current.result_id != self.source.result_id:
            raise ValueError(
                f"MeshProposalTransaction.current must bind the source result revision; got {(current.result_id, self.source.result_id)!r}"
            )
        _check_binding(current, self.projection.proposal, self.projection.policy)
        if not accept:
            return current
        issues = self.compliance.issues + _limit_issues(
            self.trusted_result, self.projection.policy.limits
        )
        issues += _preservation_issues(current, self.trusted_result, self.projection)
        if not self.admissible or issues:
            raise ValueError(
                f"MeshProposalTransaction.admissible must satisfy safety audit, compliance and preservation checks; got {(*self.safety_audit.issues, *issues)!r}"
            )
        self.trusted_result.audit.require_passed()
        return self.trusted_result


def _mesh_scope(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    scope: _dependency_phydrax_meshing__scope.MeshingScope,
    /,
) -> _dependency_phydrax_meshing__scope.MeshingScope:
    """Rebind a result-revision proposal scope to the mesh-revision binding."""
    return _dependency_phydrax_meshing__scope.MeshingScope(
        source.mesh.mesh_id,
        source.mesh.numeric_version,
        _dependency_phydrax_meshing__scope.MeshingEntityKind.MESH,
        scope.entity_dimension,
        scope.entity_set_id,
        scope.entity_ids,
    )


def _adaptation_metric(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    projection: MeshProposalProjection,
    /,
) -> _dependency_phydrax_meshing__metric.MeshMetricField:
    """Projected metric, or the isotropic metric ``I / h**2`` of projected sizes."""
    policy = projection.policy
    scope = _mesh_scope(source, projection.proposal.scope)
    if projection.metric is not None:
        values = np.asarray(projection.metric.values)
    else:
        # ty: ignore[unresolved-attribute]
        sizes = np.asarray(projection.size_field.values, dtype=np.float64)
        dimension = source.mesh.ambient_dimension
        values = np.eye(dimension)[None, :, :] / (sizes**2)[:, None, None]
    return _dependency_phydrax_meshing__metric.MeshMetricField(
        scope,
        values,
        minimum_size=policy.minimum_size,
        maximum_size=policy.maximum_size,
        maximum_anisotropy=policy.maximum_anisotropy,
    )


def _execute_adaptation(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    projection: MeshProposalProjection,
    native_policy: _dependency_phydrax_meshing__adaptation.MeshAdaptationPolicy | None,
    hierarchy: _dependency_phydrax_meshing__adaptation.MeshAdaptationHierarchy,
    layer_columns: _dependency_phydrax_meshing__mixed_adaptation.MixedLayerColumns | None,
    /,
) -> _dependency_phydrax_meshing__adaptation.MeshAdaptationResult:
    policy = projection.policy
    protected = tuple(_mesh_scope(source, scope) for scope in policy.protected_scopes)
    match projection.proposal:
        case MeshMarkingProposal():
            request = _dependency_phydrax_meshing__adaptation.MarkedMeshAdaptation(
                projection.marked_cell_ids,
                hierarchy=hierarchy,
                layer_columns=layer_columns,
            )
            kinds = {block.cell_kind for block in source.mesh.blocks}
            if kinds == {"triangle"} or kinds == {"tetrahedron"}:
                route = _dependency_phydrax_meshing__adaptation.MeshAdaptationRoute.NATIVE_BISECTION
            elif native_policy is not None:
                route = native_policy.route
            else:
                raise ValueError(
                    f"MeshProposal.native_policy must supply a qualified native route for this marked family; got {tuple(sorted(kinds))!r}"
                )
        case MeshSizeProposal() | MeshMetricProposal():
            request = _dependency_phydrax_meshing__adaptation.MetricMeshAdaptation(
                _adaptation_metric(source, projection)
            )
            match (source.mesh.topological_dimension, source.mesh.ambient_dimension):
                case (3, 3):
                    route = _dependency_phydrax_meshing__adaptation.MeshAdaptationRoute.NATIVE_METRIC_3D
                case (2, 2):
                    route = _dependency_phydrax_meshing__adaptation.MeshAdaptationRoute.NATIVE_METRIC_2D
                case _:
                    if native_policy is None:
                        raise ValueError(
                            f"MeshProposal.native_policy must supply a qualified native route for this metric family; got {(source.mesh.topological_dimension, source.mesh.ambient_dimension)!r}"
                        )
                    route = native_policy.route
        case _:
            raise TypeError(
                f"MeshProposalProjection.proposal must be a marking, size or metric proposal for adaptation; got {type(projection.proposal).__name__!r}"
            )
    if native_policy is None:
        # Native publication owns geometric validity; the stricter proposal
        # policy is evaluated below without turning rejection into execution failure.
        native_policy = _dependency_phydrax_meshing__adaptation.MeshAdaptationPolicy(
            route,
            protected_scopes=protected,
            limits=policy.limits,
        )
    if native_policy.route in (
        _dependency_phydrax_meshing__adaptation.MeshAdaptationRoute.MMG,
        _dependency_phydrax_meshing__adaptation.MeshAdaptationRoute.OMEGA_H,
    ):
        raise ValueError(
            f"MeshProposal.native_policy.route must select a native rather than external-provider route; got {native_policy.route!r}"
        )
    if native_policy.limits.limits_id != policy.limits.limits_id:
        raise ValueError(
            f"MeshProposal.native_policy.limits must match the proposal resource limits; got {(native_policy.limits.limits_id, policy.limits.limits_id)!r}"
        )
    if not {scope.scope_id for scope in protected}.issubset(
        scope.scope_id for scope in native_policy.protected_scopes
    ):
        raise ValueError(
            f"MeshProposal.native_policy.protected_scopes must retain every proposal-protected scope; got {(tuple(scope.scope_id for scope in protected), tuple(scope.scope_id for scope in native_policy.protected_scopes))!r}"
        )
    return _dependency_phydrax_meshing__adaptation.execute_mesh_adaptation(
        _dependency_phydrax_meshing__adaptation.prepare_mesh_adaptation(
            source, request, policy=native_policy
        )
    )


@checked
def prepare_mesh_proposal(
    source: _dependency_phydrax_meshing__result.CellMeshingResult,
    proposal: MeshProposal,
    policy: MeshProposalSafetyPolicy,
    /,
    *,
    native_policy: _dependency_phydrax_meshing__adaptation.MeshAdaptationPolicy
    | None = None,
    hierarchy: _dependency_phydrax_meshing__adaptation.MeshAdaptationHierarchy = None,
    layer_columns: _dependency_phydrax_meshing__mixed_adaptation.MixedLayerColumns
    | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> MeshProposalTransaction:
    """Project, execute a native trusted path, audit and prepare atomic promotion.

    Curved nested simplex refinement, material organization and B-Rep associations
    use the native adaptation owner's coordinate/lineage/association transfer.
    A metric route for a new dimension/family must be supplied explicitly and
    pass that owner's admission. Relocation's affine optimization route cannot
    silently flatten a curved map or discard metadata.
    """
    if (hierarchy is not None or layer_columns is not None) and not isinstance(
        proposal, MeshMarkingProposal
    ):
        raise ValueError(
            f"MeshProposal.hierarchy must provide hierarchy and layer context only for marking proposals; got {(type(proposal).__name__, hierarchy is not None, layer_columns is not None)!r}"
        )
    started = time.monotonic()
    with measure_phase(record_phase, "native_preparation"):
        projection = project_mesh_proposal(source, proposal, policy)
        if record_phase is not None:
            jax.block_until_ready(projection)
    adaptation, optimization = None, None
    candidate = source
    match proposal:
        case MeshCoordinateProposal():
            if native_policy is not None:
                raise ValueError(
                    f"MeshCoordinateProposal.native_policy must be absent for coordinate optimization; got {native_policy.route!r}"
                )
            affine = CellGeometrySpec.affine(source.mesh)
            if (
                source.geometry.geometry_layout_id != affine.geometry_layout_id
                or not np.array_equal(source.geometry.coordinates, affine.coordinates)
            ):
                raise ValueError(
                    f"MeshCoordinateProposal.source.geometry must be affine for coordinate optimization; got {(source.geometry.geometry_layout_id, affine.geometry_layout_id)!r}"
                )
            if source.boundary is not None or any(
                (
                    source.patches,
                    source.zones,
                    source.labels,
                    source.attributes,
                    source.associations,
                )
            ):
                raise ValueError(
                    f"MeshCoordinateProposal.source.metadata must be absent when coordinate optimization cannot transfer it; got {(source.boundary is not None, len(source.patches), len(source.zones), len(source.labels), len(source.attributes), len(source.associations))!r}"
                )
            fixed, _ = _coordinate_projector(source, proposal, policy)
            plan = _dependency_phydrax_meshing__optimization.TargetMatrixOptimizationPlan(
                source.mesh,
                target_coordinates=projection.target_coordinates,
                fixed_vertices=fixed,
                coordinate_bounds=_optimization_bounds(source, policy),
                termination=OptimizationTermination(
                    maximum_steps=policy.maximum_optimization_iterations
                ),
            )
            with measure_phase(record_phase, "improvement"):
                optimization = (
                    _dependency_phydrax_meshing__optimization.optimize_cell_mesh(
                        plan,
                        source.coordinate_contract,
                        numeric_version=f"proposal:{projection.projection_id}",
                    )
                )
                if record_phase is not None:
                    jax.block_until_ready(optimization)
            candidate = MeshProposalTransaction.coordinate_candidate(source, optimization)
        case MeshMarkingProposal() if projection.marked_cell_ids.size == 0:
            pass
        case MeshMarkingProposal() | MeshSizeProposal() | MeshMetricProposal():
            with measure_phase(record_phase, "topology_adaptation"):
                adaptation = _execute_adaptation(
                    source, projection, native_policy, hierarchy, layer_columns
                )
                if record_phase is not None:
                    jax.block_until_ready(adaptation)
            candidate = adaptation.target
        case _:
            raise TypeError(
                f"MeshProposalTransaction.proposal must be a supported typed mesh proposal; got {type(proposal).__name__!r}"
            )
    with measure_phase(record_phase, "audit"):
        audit = _dependency_phydrax_meshing__audit.audit_cell_mesh(
            candidate.mesh,
            candidate.geometry,
            policy=policy.audit_policy,
        )
        if record_phase is not None:
            jax.block_until_ready(audit)
    issues = _limit_issues(candidate, policy.limits) + _preservation_issues(
        source, candidate, projection
    )
    if not audit.passed:
        issues += ("safety_audit",)
    if optimization is not None and not optimization.accepted:
        issues += ("mesh_optimization",)
    if adaptation is not None:
        issues += adaptation.compliance.issues
        if not adaptation.status.converged:
            issues += ("native_admission",)
    if (
        _payload_bytes(candidate)
        + _payload_bytes(None if adaptation is None else adaptation.transfer)
        > policy.limits.maximum_data_bytes
    ):
        issues += ("maximum_data_bytes",)
    elapsed = time.monotonic() - started
    if elapsed > policy.limits.maximum_wall_seconds:
        issues += ("maximum_wall_seconds",)
    compliance = _dependency_phydrax_meshing__result.MeshingComplianceReport(
        projection.projection_id,
        issues=tuple(dict.fromkeys(issues)),
        requested=(
            ("maximum_cells", policy.limits.maximum_cells),
            ("maximum_vertices", policy.limits.maximum_vertices),
        ),
        achieved=(
            ("cells", candidate.audit.entity_counts[-1]),
            ("vertices", candidate.audit.vertex_count),
        ),
    )
    return MeshProposalTransaction(
        source,
        projection,
        candidate,
        audit,
        compliance,
        adaptation,
        optimization,
    )


__all__ = [
    "AbstractMeshProposer",
    "LearnedMeshProposer",
    "MeshCoordinateProposal",
    "MeshMarkingProposal",
    "MeshMetricProposal",
    "MeshProposalFeatures",
    "MeshProposal",
    "MeshProposalProjection",
    "MeshProposalSafetyPolicy",
    "MeshProposalTransaction",
    "MeshSizeProposal",
    "mesh_proposal_scope",
    "prepare_mesh_proposal",
    "project_mesh_proposal",
]
