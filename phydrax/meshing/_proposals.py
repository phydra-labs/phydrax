#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
import time
from abc import abstractmethod
from typing import Any, ClassVar, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

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
from .._model._component import bind_positional_component
from .._model._ports import require_port_shapes
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    CellGeometrySpec,
    CellMesh,
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ..optim import OptimizationTermination
from ..typing import checked, Dim, Float64, Identifier, Identifiers, parse
from ._adaptation import (
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationHierarchy,
    MeshAdaptationPolicy,
    MeshAdaptationResult,
    MeshAdaptationRoute,
    MetricMeshAdaptation,
    prepare_mesh_adaptation,
)
from ._audit import audit_cell_mesh, CellMeshAuditPolicy, CellMeshAuditReport
from ._contracts import MeshingLimits
from ._measurements import measure_phase, NativeMeshingPhaseRecorder
from ._metric import (
    _grade_scalar_sizes,
    MeshMetricField,
    MeshMetricSamples,
    MetricGradationKind,
    MetricGradationPolicy,
    MetricNormalizationEvidence,
    MetricNormalizationPolicy,
    normalize_mesh_metric,
)
from ._mixed_adaptation import MixedLayerColumns
from ._optimization import (
    MeshOptimizationResult,
    optimize_cell_mesh,
    TargetMatrixOptimizationPlan,
)
from ._quality import evaluate_cell_quality
from ._result import CellMeshingResult, MeshingComplianceReport
from ._scope import MeshingEntityKind, MeshingScope
from ._sizing import ResolvedSizeField, SizeFieldDomain


def mesh_proposal_scope(
    source: CellMeshingResult,
    dimension: int,
    entity_ids: ArrayLike | None = None,
    /,
) -> MeshingScope:
    """Bind proposal entities to the exact certified mesh and result revision.

    Proposal values always follow the scope's sorted global-ID order, not the
    mesh's storage order. Omitting IDs selects all entities of the given degree.
    """
    if not isinstance(source, CellMeshingResult):
        raise TypeError("source must be CellMeshingResult.")
    entities = source.mesh.entity_set(dimension)
    scope = MeshingScope(
        source.mesh.mesh_id,
        source.result_id,
        MeshingEntityKind.MESH,
        dimension,
        entities.entity_set_id,
        entities.entity_ids if entity_ids is None else entity_ids,
    )
    _scope_rows(source, scope)
    return scope


def _scope_rows(source: CellMeshingResult, scope: MeshingScope) -> np.ndarray:
    if not isinstance(scope, MeshingScope):
        raise TypeError("scope must be MeshingScope.")
    mesh = source.mesh
    if (
        scope.entity_kind is not MeshingEntityKind.MESH
        or scope.source_id != mesh.mesh_id
        or scope.source_revision != source.result_id
        or scope.entity_dimension > mesh.topological_dimension
    ):
        raise ValueError(
            "Proposal scope has a stale or incompatible mesh revision binding."
        )
    entities = mesh.entity_set(scope.entity_dimension)
    if scope.entity_set_id != entities.entity_set_id:
        raise ValueError("Proposal scope has a stale entity-set binding.")
    identifiers = np.asarray(entities.entity_ids)
    requested = np.asarray(scope.entity_ids)
    order = np.argsort(identifiers)
    positions = np.searchsorted(identifiers[order], requested)
    if np.any(positions >= identifiers.size) or not np.array_equal(
        identifiers[order[np.minimum(positions, identifiers.size - 1)]], requested
    ):
        raise ValueError("Proposal scope contains unknown entity IDs.")
    rows = order[positions]
    if not np.all(np.asarray(entities.active_mask)[rows]):
        raise ValueError("Proposal scope contains inactive entities.")
    return rows


class _AbstractMeshProposal(StrictModule, NonTrainableState):
    scope: MeshingScope
    values: Array
    coordinate_contract_id: str = eqx.field(static=True)
    proposer_id: str = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: CellMeshingResult,
        scope: MeshingScope,
        values: ArrayLike,
        *,
        dimension: int,
        value_shape: tuple[int, ...],
        proposer_id: str,
    ) -> None:
        _scope_rows(source, scope)
        if scope.entity_dimension != dimension:
            raise ValueError("Proposal scope has the wrong entity dimension.")
        data = np.asarray(values, dtype=np.float64)
        if data.shape != (scope.entity_ids.size, *value_shape) or not np.all(
            np.isfinite(data)
        ):
            raise ValueError(
                "Proposal values must be finite and aligned with the sorted scope IDs."
            )
        proposer = str(proposer_id).strip()
        if not proposer:
            raise ValueError("proposer_id must be non-empty.")
        self.scope = scope
        self.values = jnp.asarray(data)
        self.coordinate_contract_id = source.coordinate_contract.spatial_id
        self.proposer_id = proposer
        self.proposal_id = canonical_fingerprint(
            {
                "kind": type(self).__name__,
                "scope": scope.scope_id,
                "values": array_tree_fingerprint(self.values),
                "coordinate_contract": self.coordinate_contract_id,
                "proposer": proposer,
            }
        )


class MeshMarkingProposal(_AbstractMeshProposal):
    """Untrusted cell priorities; positive scores request native bisection."""

    def __init__(
        self,
        source: CellMeshingResult,
        scope: MeshingScope,
        scores: ArrayLike,
        /,
        *,
        proposer_id: str,
    ) -> None:
        super().__init__(
            source,
            scope,
            scores,
            dimension=source.mesh.topological_dimension,
            value_shape=(),
            proposer_id=proposer_id,
        )


class MeshSizeProposal(_AbstractMeshProposal):
    """Untrusted vertex sizes, expressed in the source coordinate length unit."""

    def __init__(
        self,
        source: CellMeshingResult,
        scope: MeshingScope,
        sizes: ArrayLike,
        /,
        *,
        proposer_id: str,
    ) -> None:
        super().__init__(
            source, scope, sizes, dimension=0, value_shape=(), proposer_id=proposer_id
        )


class MeshMetricProposal(_AbstractMeshProposal):
    """Untrusted vertex tensors; projection repairs symmetry and definiteness."""

    def __init__(
        self,
        source: CellMeshingResult,
        scope: MeshingScope,
        metrics: ArrayLike,
        /,
        *,
        proposer_id: str,
    ) -> None:
        dimension = source.mesh.ambient_dimension
        super().__init__(
            source,
            scope,
            metrics,
            dimension=0,
            value_shape=(dimension, dimension),
            proposer_id=proposer_id,
        )


class MeshCoordinateProposal(_AbstractMeshProposal):
    """Untrusted vertex targets in one explicit, unchanged coordinate contract."""

    def __init__(
        self,
        source: CellMeshingResult,
        scope: MeshingScope,
        coordinates: ArrayLike,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        proposer_id: str,
    ) -> None:
        if (
            not isinstance(coordinate_contract, SpatialCoordinateContract)
            or coordinate_contract.spatial_id != source.coordinate_contract.spatial_id
        ):
            raise ValueError(
                "Coordinate proposals require the exact source coordinate contract."
            )
        super().__init__(
            source,
            scope,
            coordinates,
            dimension=0,
            value_shape=(source.mesh.ambient_dimension,),
            proposer_id=proposer_id,
        )


MeshProposal = (
    MeshMarkingProposal | MeshSizeProposal | MeshMetricProposal | MeshCoordinateProposal
)

MeshProposerKind: TypeAlias = Literal["marking", "size", "metric", "coordinate"]


class _ProposalEntityDim(Dim):
    """Entities of the exact proposal feature scope."""


class _ProposalFeatureDim(Dim, minimum=1):
    """Independently identified scientific feature columns."""


@final
class MeshProposalFeatures(StrictModule, NonTrainableState):
    """Owner-identified feature columns bound to the exact result and entity scope."""

    __strict_contract__ = True

    values: Float64[_ProposalEntityDim, _ProposalFeatureDim]
    scope: MeshingScope
    source_result_id: Identifier = eqx.field(static=True)
    feature_ids: Identifiers[_ProposalFeatureDim] = eqx.field(static=True)
    feature_owner_id: Identifier = eqx.field(static=True)
    feature_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        source: CellMeshingResult,
        scope: MeshingScope,
        values: ArrayLike,
        /,
        *,
        feature_ids: tuple[str, ...],
        feature_owner_id: str,
    ) -> None:
        if not isinstance(source, CellMeshingResult):
            raise TypeError("source must be CellMeshingResult.")
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
                "Features require finite scoped rows and unique scientific column IDs."
            )
        if not isinstance(feature_owner_id, str) or not feature_owner_id:
            raise ValueError(
                "feature_owner_id must identify the producing scientific owner."
            )
        self.values, self.scope = jnp.asarray(data, dtype=jnp.float64), scope
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
    size, metric or coordinate proposal bound to the exact source revision; only
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
        source: CellMeshingResult,
        features: MeshProposalFeatures,
        /,
        *,
        scope: MeshingScope | None = None,
    ) -> MeshProposal:
        raise NotImplementedError


def _model_value_shape(size: Any, /) -> tuple[int, ...]:
    if size == "scalar":
        return ()
    if isinstance(size, int):
        return (int(size),)
    return tuple(size)


class LearnedMeshProposer(AbstractMeshProposer):
    """Pointwise learned marking, size, metric or coordinate proposer.

    The model maps one feature row to one proposal value: a marking score per
    cell (`kind="marking"`), a size per vertex (`"size"`), a square coordinate-axis
    metric tensor (`"metric"`), or a source-contract coordinate vector
    (`"coordinate"`). Sizes are exact: `in_size` is the feature width and `out_size`
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
                    raise ValueError("Scalar proposers do not take spatial_dimension.")
                value_shape = ()
            case "metric" | "coordinate":
                if (
                    isinstance(spatial_dimension, bool)
                    or not isinstance(spatial_dimension, int)
                    or spatial_dimension <= 0
                ):
                    raise ValueError(
                        "Metric/coordinate proposers require positive spatial_dimension."
                    )
                value_shape = (
                    (spatial_dimension, spatial_dimension)
                    if kind == "metric"
                    else (spatial_dimension,)
                )
            case _:
                raise ValueError(f"Unknown mesh proposer kind {kind!r}.")
        binding = model.input_binding()
        if binding.batch_mode != "pointwise" or binding.input_mode != "flat":
            raise ValueError("Learned mesh proposers require a pointwise flat binding.")
        if isinstance(model.in_size, bool) or not isinstance(model.in_size, int):
            raise ValueError("Learned mesh proposer in_size must be the feature width.")
        if _model_value_shape(model.out_size) != value_shape:
            raise ValueError(
                f"Learned {kind} proposer out_size must produce shape {value_shape}; "
                f"got {model.out_size!r}."
            )
        identifier = str(proposer_id).strip()
        if not identifier:
            raise ValueError("proposer_id must be non-empty.")
        site = "LearnedMeshProposer"
        owner_ports = require_port_shapes(
            ports, inputs=((model.in_size,),), outputs=(value_shape,), site=site
        )
        bind_positional_component(
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
                f"Proposer features must have shape (entities, {self.model.in_size}); "
                f"got {rows.shape}."
            )
        binding = self.model.input_binding()
        return jax.vmap(
            lambda row: binding.call(self.model, row, key=None, iter_=None, kwargs={})
        )(rows)

    def evaluate_addressed(
        self,
        features: ArrayLike,
        entity_ids: ArrayLike,
        key: Array,
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
            raise ValueError("Addressed features must match the model feature width.")
        if identifiers.ndim != 1 or identifiers.shape[0] != rows.shape[0]:
            raise ValueError("Scientific entity IDs must align with feature rows.")
        if identifiers.dtype != jnp.dtype("int64"):
            raise ValueError("Scientific entity IDs require signed 64-bit storage.")
        if not isinstance(address_id, str) or not address_id:
            raise ValueError("address_id must identify the scientific evaluation.")
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
        source: CellMeshingResult,
        features: MeshProposalFeatures,
        /,
        *,
        scope: MeshingScope | None = None,
        key: Array | None = None,
    ) -> MeshProposal:
        if features.source_result_id != source.result_id:
            raise ValueError("Learned features have a stale source result revision.")
        if (
            self.spatial_dimension is not None
            and self.spatial_dimension != source.mesh.ambient_dimension
        ):
            raise ValueError(
                "Proposer coordinate axes must match the source coordinate contract."
            )
        dimension = source.mesh.topological_dimension if self.kind == "marking" else 0
        scope_ = features.scope if scope is None else scope
        _scope_rows(source, scope_)
        if (
            scope_.scope_id != features.scope.scope_id
            or scope_.entity_dimension != dimension
        ):
            raise ValueError(
                "Learned feature scope does not match the requested proposal entities."
            )
        rows = features.values
        if rows.ndim != 2 or rows.shape[0] != scope_.entity_ids.size:
            raise ValueError(
                "Proposer features must hold one row per scope entity in sorted "
                "global-ID order."
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
                    source, scope_, values, proposer_id=evaluation_id
                )
            case "size":
                return MeshSizeProposal(source, scope_, values, proposer_id=evaluation_id)
            case "metric":
                return MeshMetricProposal(
                    source, scope_, values, proposer_id=evaluation_id
                )
            case "coordinate":
                return MeshCoordinateProposal(
                    source,
                    scope_,
                    values,
                    source.coordinate_contract,
                    proposer_id=evaluation_id,
                )
            case _:
                raise ValueError(f"Unknown mesh proposer kind {self.kind!r}.")


class MeshProposalSafetyPolicy(StrictModule, NonTrainableState):
    """Trusted, revision-bound constraints, independent of proposer evidence.

    Protected scopes preserve their entities and fix their incident vertices.
    Coordinate bounds and displacement are in source coordinate units. Limits
    are admission limits on candidate payloads, not a process memory quota or
    a preemptive timeout. Wall-time overruns cannot be committed.
    """

    source_result_id: str = eqx.field(static=True)
    protected_scopes: tuple[MeshingScope, ...]
    limits: MeshingLimits
    audit_policy: CellMeshAuditPolicy
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
        source: CellMeshingResult,
        /,
        *,
        minimum_size: float,
        maximum_size: float,
        maximum_displacement: float,
        protected_scopes: tuple[MeshingScope, ...] = (),
        limits: MeshingLimits | None = None,
        audit_policy: CellMeshAuditPolicy | None = None,
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
            raise ValueError("Safety bounds must be finite.")
        if (
            minimum <= 0
            or maximum < minimum
            or anisotropy < 1
            or gradation < 1
            or displacement < 0
        ):
            raise ValueError(
                "Invalid proposal size, anisotropy, gradation, or displacement bounds."
            )
        normal_floor = np.sqrt(np.finfo(np.float64).tiny)
        if (
            minimum < normal_floor
            or maximum > 1.0 / normal_floor
            or anisotropy > 1.0 / normal_floor
        ):
            raise ValueError("Safety bounds must admit representable metric eigenvalues.")
        marked, iterations = (
            int(maximum_marked_cells),
            int(maximum_optimization_iterations),
        )
        if marked < 0 or iterations <= 0:
            raise ValueError(
                "Mark capacity must be non-negative and optimization iterations positive."
            )
        protected = tuple(protected_scopes)
        for scope in protected:
            _scope_rows(source, scope)
        limits_ = MeshingLimits() if limits is None else limits
        audit_ = CellMeshAuditPolicy() if audit_policy is None else audit_policy
        if not isinstance(limits_, MeshingLimits) or not isinstance(
            audit_, CellMeshAuditPolicy
        ):
            raise TypeError(
                "Safety policy requires MeshingLimits and CellMeshAuditPolicy."
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
                    "coordinate_bounds must contain finite ordered lower/upper vectors."
                )
            if np.any(source.mesh.coordinates < bounds[0]) or np.any(
                source.mesh.coordinates > bounds[1]
            ):
                raise ValueError("Source coordinates lie outside the safety bounds.")
        self.source_result_id = source.result_id
        self.protected_scopes = protected
        self.limits = limits_
        self.audit_policy = audit_
        self.coordinate_bounds = None if bounds is None else jnp.asarray(bounds)
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
    source: CellMeshingResult, policy: MeshProposalSafetyPolicy
) -> np.ndarray:
    fixed = np.zeros(source.mesh.coordinates.shape[0], dtype=np.bool_)
    for scope in policy.protected_scopes:
        rows = _entity_closure(
            source.mesh, scope.entity_dimension, _scope_rows(source, scope), 0
        )
        fixed[rows] = True
    return fixed


def _payload_bytes(value: Any) -> int:
    return sum(
        leaf.nbytes
        for leaf in jax.tree_util.tree_leaves(value)
        if isinstance(leaf, (jax.Array, np.ndarray))
    )


def _limit_issues(result: CellMeshingResult, limits: MeshingLimits) -> tuple[str, ...]:
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


def _check_binding(source: Any, proposal: Any, policy: Any) -> None:
    if not isinstance(source, CellMeshingResult):
        raise TypeError("source must be CellMeshingResult.")
    if not isinstance(
        proposal,
        (
            MeshMarkingProposal,
            MeshSizeProposal,
            MeshMetricProposal,
            MeshCoordinateProposal,
        ),
    ):
        raise TypeError("proposal must be a typed mesh proposal.")
    if not isinstance(policy, MeshProposalSafetyPolicy):
        raise TypeError("policy must be MeshProposalSafetyPolicy.")
    _scope_rows(source, proposal.scope)
    if policy.source_result_id != source.result_id:
        raise ValueError("Safety policy has a stale result revision binding.")
    if proposal.coordinate_contract_id != source.coordinate_contract.spatial_id:
        raise ValueError("Proposal coordinate contract does not match the source.")
    if source.coordinate_contract.coordinate_system != "cartesian":
        raise ValueError("Native proposal routes require Cartesian coordinates.")
    source.audit.require_passed()
    if not source.compliance.passed or _limit_issues(source, policy.limits):
        raise ValueError(
            "Source does not satisfy proposal compliance and capacity limits."
        )
    for scope in policy.protected_scopes:
        _scope_rows(source, scope)


def _project_sizes(values: Any, points: Any, edges: Any, policy: Any) -> np.ndarray:
    """Clamp to the size bounds, then grade exactly through the metric owner."""
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    graded, _, _ = _grade_scalar_sizes(
        np.clip(values, policy.minimum_size, policy.maximum_size),
        edges,
        lengths,
        np.full((edges.shape[0],), policy.maximum_gradation),
        MetricGradationKind.PHYSICAL,
    )
    return graded


def _project_metric(
    scope: Any, raw: Any, points: Any, edges: Any, policy: Any
) -> tuple[MeshMetricField, MetricNormalizationEvidence]:
    # Untrusted tensors: the trusted policy explicitly requests symmetric-part
    # repair and indefinite projection, and the evidence reports both.
    return normalize_mesh_metric(
        MeshMetricSamples(scope, raw),
        policy=MetricNormalizationPolicy(
            minimum_size=policy.minimum_size,
            maximum_size=policy.maximum_size,
            maximum_anisotropy=policy.maximum_anisotropy,
            gradation=MetricGradationPolicy(policy.maximum_gradation),
            symmetrize=True,
            project_indefinite=True,
        ),
        adjacency=edges,
        coordinates=points,
    )


def _coordinate_projector(source: Any, proposal: Any, policy: Any) -> Any:
    points = jnp.asarray(source.mesh.coordinates)
    movable = np.zeros(points.shape[0], dtype=np.bool_)
    movable[_scope_rows(source, proposal.scope)] = True
    fixed = jnp.asarray(~movable | _protected_vertices(source, policy))
    bounds = policy.coordinate_bounds

    def project(values: Any) -> Any:
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


def _optimization_bounds(source: Any, policy: Any) -> tuple[np.ndarray, np.ndarray]:
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
    source: CellMeshingResult, scores: np.ndarray, policy: MeshProposalSafetyPolicy
) -> np.ndarray:
    """Highest positive scores, then global ID; exact closure belongs to the route."""
    cells = np.concatenate([np.asarray(block.global_ids) for block in source.mesh.blocks])
    order = np.lexsort((cells, -scores))
    eligible = scores > 0.0
    for scope in policy.protected_scopes:
        if scope.entity_dimension == source.mesh.topological_dimension:
            eligible[_scope_rows(source, scope)] = False
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
    return np.sort(cells[selected]).astype(np.int64, copy=False)


class MeshProposalProjection(StrictModule, NonTrainableState):
    """Projected proposal evidence; this is not a mesh or a certification."""

    proposal: MeshProposal
    policy: MeshProposalSafetyPolicy
    marked_cell_ids: Array
    size_field: ResolvedSizeField | None
    metric: MeshMetricField | None
    metric_evidence: MetricNormalizationEvidence | None
    target_coordinates: Array | None
    projection_id: str = eqx.field(static=True)

    def __init__(
        self,
        proposal: Any,
        policy: Any,
        marked_cell_ids: Any,
        size_field: Any,
        metric: Any,
        target_coordinates: Any,
        *,
        metric_evidence: Any = None,
    ) -> None:
        if (metric is None) != (metric_evidence is None):
            raise ValueError("A projected metric requires its normalization evidence.")
        self.proposal, self.policy = proposal, policy
        self.marked_cell_ids = jnp.asarray(marked_cell_ids, dtype=jnp.int64)
        self.size_field, self.metric = size_field, metric
        self.metric_evidence = metric_evidence
        self.target_coordinates = target_coordinates
        self.projection_id = canonical_fingerprint(
            {
                "kind": "mesh-proposal-projection",
                "proposal": proposal.proposal_id,
                "policy": policy.policy_id,
                "marks": array_tree_fingerprint(self.marked_cell_ids),
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


def project_mesh_proposal(
    source: CellMeshingResult,
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
    if isinstance(proposal, MeshCoordinateProposal):
        _, project = _coordinate_projector(source, proposal, policy)
        target = project(jnp.asarray(mesh.coordinates).at[rows].set(proposal.values))
        # A safe target must itself be a valid optimization reference. Backtrack
        # toward the certified source, rather than accepting an inverted target.
        for _ in range(64):
            if bool(jnp.all(evaluate_cell_quality(mesh, target).sampled_valid)):
                break
            target = project(0.5 * (target + mesh.coordinates))
        else:
            raise ValueError("Coordinate proposal has no certifiable projected target.")
    else:
        if isinstance(proposal, MeshMarkingProposal):
            scores = np.zeros(source.audit.entity_counts[-1], dtype=np.float64)
            scores[rows] = np.asarray(proposal.values)
            marks = _safe_marks(source, scores, policy)
        else:
            kinds = {block.cell_kind for block in mesh.blocks}
            if kinds != {"triangle"} and kinds != {"tetrahedron"}:
                raise ValueError(
                    "Native size/metric proposal projection requires simplex blocks."
                )
            if rows.size != mesh.coordinates.shape[0]:
                raise ValueError(
                    "Size and metric proposals must cover every source vertex."
                )
            # Field arrays are scope ordered; connectivity and points are mesh ordered.
            inverse = np.empty(rows.size, dtype=np.int64)
            inverse[rows] = np.arange(rows.size)
            connectivity = mesh.connectivity
            if not isinstance(
                connectivity,
                (PolygonalConnectivity, TetrahedralConnectivity, PolyhedralConnectivity),
            ):
                raise TypeError("Simplex proposals require canonical edge connectivity.")
            field_edges = inverse[np.asarray(connectivity.edges)]
            field_points = np.asarray(mesh.coordinates)[rows]
            if isinstance(proposal, MeshSizeProposal):
                sizes = ResolvedSizeField(
                    SizeFieldDomain.SAMPLE_CLOUD,
                    field_points,
                    _project_sizes(
                        np.asarray(proposal.values), field_points, field_edges, policy
                    ),
                    sample_entity_ids=proposal.scope.entity_ids,
                    source_control_ids=(proposal.proposal_id, policy.policy_id),
                )
            else:
                metric, metric_evidence = _project_metric(
                    proposal.scope,
                    np.asarray(proposal.values),
                    field_points,
                    field_edges,
                    policy,
                )
    return MeshProposalProjection(
        proposal,
        policy,
        marks,
        sizes,
        metric,
        target,
        metric_evidence=metric_evidence,
    )


def _entity_vertex_signatures(mesh: CellMesh, dimension: int) -> Any:
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


def _preservation_issues(source: Any, candidate: Any, projection: Any) -> Any:
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

    source: CellMeshingResult
    projection: MeshProposalProjection
    trusted_result: CellMeshingResult
    safety_audit: CellMeshAuditReport
    compliance: MeshingComplianceReport
    adaptation: MeshAdaptationResult | None
    optimization: MeshOptimizationResult | None
    transaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: Any,
        projection: Any,
        trusted_result: Any,
        safety_audit: Any,
        compliance: Any,
        adaptation: Any,
        optimization: Any,
    ) -> None:
        _check_binding(source, projection.proposal, projection.policy)
        if (
            safety_audit.mesh_id != trusted_result.mesh.mesh_id
            or safety_audit.policy_id != projection.policy.audit_policy.policy_id
        ):
            raise ValueError("Safety audit must match the candidate and safety policy.")
        if compliance.specification_id != projection.projection_id:
            raise ValueError("Compliance must be bound to the exact projected proposal.")
        if adaptation is not None and (
            adaptation.source.result_id != source.result_id
            or adaptation.target.result_id != trusted_result.result_id
        ):
            raise ValueError("Proposal adaptation must match the source and candidate.")
        if isinstance(projection.proposal, MeshCoordinateProposal):
            # A failed optimization leaves the exact source as its candidate.
            if (
                optimization is None
                or adaptation is not None
                or (
                    optimization.result.result_id
                    if optimization.accepted
                    else source.result_id
                )
                != trusted_result.result_id
            ):
                raise ValueError(
                    "Coordinate proposal candidate must match its trusted optimization."
                )
        elif projection.marked_cell_ids.size or not isinstance(
            projection.proposal, MeshMarkingProposal
        ):
            if adaptation is None or optimization is not None:
                raise ValueError(
                    "Adaptive proposal candidate requires trusted adaptation evidence."
                )
        elif adaptation is not None or trusted_result.result_id != source.result_id:
            raise ValueError(
                "An empty projected marking must preserve the exact source result."
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

    def commit(
        self, current: CellMeshingResult, /, *, accept: bool = True
    ) -> CellMeshingResult:
        if not isinstance(accept, (bool, np.bool_)):
            raise TypeError("Proposal commit requires an explicit host boolean decision.")
        if (
            not isinstance(current, CellMeshingResult)
            or current.result_id != self.source.result_id
        ):
            raise ValueError("Cannot commit a proposal against a stale current revision.")
        _check_binding(current, self.projection.proposal, self.projection.policy)
        if not accept:
            return current
        issues = self.compliance.issues + _limit_issues(
            self.trusted_result, self.projection.policy.limits
        )
        issues += _preservation_issues(current, self.trusted_result, self.projection)
        if not self.admissible or issues:
            raise ValueError(
                "Proposal is not admissible: "
                + "; ".join((*self.safety_audit.issues, *issues))
            )
        self.trusted_result.audit.require_passed()
        return self.trusted_result


def _mesh_scope(source: CellMeshingResult, scope: MeshingScope, /) -> MeshingScope:
    """Rebind a result-revision proposal scope to the mesh-revision binding."""
    return MeshingScope(
        source.mesh.mesh_id,
        source.mesh.numeric_version,
        MeshingEntityKind.MESH,
        scope.entity_dimension,
        scope.entity_set_id,
        scope.entity_ids,
    )


def _adaptation_metric(
    source: CellMeshingResult, projection: MeshProposalProjection, /
) -> MeshMetricField:
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
    return MeshMetricField(
        scope,
        values,
        minimum_size=policy.minimum_size,
        maximum_size=policy.maximum_size,
        maximum_anisotropy=policy.maximum_anisotropy,
    )


def _execute_adaptation(
    source: CellMeshingResult,
    projection: MeshProposalProjection,
    native_policy: MeshAdaptationPolicy | None,
    hierarchy: MeshAdaptationHierarchy,
    layer_columns: MixedLayerColumns | None,
    /,
) -> MeshAdaptationResult:
    policy = projection.policy
    protected = tuple(_mesh_scope(source, scope) for scope in policy.protected_scopes)
    if isinstance(projection.proposal, MeshMarkingProposal):
        request = MarkedMeshAdaptation(
            projection.marked_cell_ids, hierarchy=hierarchy, layer_columns=layer_columns
        )
        kinds = {block.cell_kind for block in source.mesh.blocks}
        if kinds == {"triangle"} or kinds == {"tetrahedron"}:
            route = MeshAdaptationRoute.NATIVE_BISECTION
        elif native_policy is not None:
            route = native_policy.route
        else:
            raise ValueError(
                "This marked family requires an explicit qualified native policy."
            )
    else:
        request = MetricMeshAdaptation(_adaptation_metric(source, projection))
        match (source.mesh.topological_dimension, source.mesh.ambient_dimension):
            case (3, 3):
                route = MeshAdaptationRoute.NATIVE_METRIC_3D
            case (2, 2):
                route = MeshAdaptationRoute.NATIVE_METRIC_2D
            case _:
                if native_policy is None:
                    raise ValueError(
                        "This metric proposal requires an explicit qualified native route."
                    )
                route = native_policy.route
    if native_policy is None:
        # Native publication owns geometric validity; the stricter proposal
        # policy is evaluated below without turning rejection into execution failure.
        native_policy = MeshAdaptationPolicy(
            route,
            protected_scopes=protected,
            limits=policy.limits,
        )
    if native_policy.route in (MeshAdaptationRoute.MMG, MeshAdaptationRoute.OMEGA_H):
        raise ValueError("Learned proposal execution requires an explicit native route.")
    if native_policy.limits.limits_id != policy.limits.limits_id:
        raise ValueError(
            "Native execution must use the proposal's exact resource limits."
        )
    if not {scope.scope_id for scope in protected}.issubset(
        scope.scope_id for scope in native_policy.protected_scopes
    ):
        raise ValueError("Native execution must preserve every proposal-protected scope.")
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=native_policy)
    )


def prepare_mesh_proposal(
    source: CellMeshingResult,
    proposal: MeshProposal,
    policy: MeshProposalSafetyPolicy,
    /,
    *,
    native_policy: MeshAdaptationPolicy | None = None,
    hierarchy: MeshAdaptationHierarchy = None,
    layer_columns: MixedLayerColumns | None = None,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> MeshProposalTransaction:
    """Project, execute a native trusted path, audit and prepare atomic promotion.

    Curved nested simplex refinement, material organization and B-Rep associations
    use the native adaptation owner's coordinate/lineage/association transfer.
    A metric route for a new dimension/family must be supplied explicitly and
    pass that owner's admission. Relocation's affine optimization route cannot
    silently flatten a curved map or discard metadata.
    """
    if native_policy is not None and not isinstance(native_policy, MeshAdaptationPolicy):
        raise TypeError("native_policy must be MeshAdaptationPolicy or None.")
    if (hierarchy is not None or layer_columns is not None) and not isinstance(
        proposal, MeshMarkingProposal
    ):
        raise ValueError(
            "Refinement hierarchy/layer context belongs only to marking proposals."
        )
    started = time.monotonic()
    with measure_phase(record_phase, "native_preparation"):
        projection = project_mesh_proposal(source, proposal, policy)
        if record_phase is not None:
            jax.block_until_ready(projection)
    adaptation, optimization = None, None
    candidate = source
    if isinstance(proposal, MeshCoordinateProposal):
        if native_policy is not None:
            raise ValueError(
                "Coordinate optimization does not consume an adaptation policy."
            )
        affine = CellGeometrySpec.affine(source.mesh)
        if (
            source.geometry.geometry_layout_id != affine.geometry_layout_id
            or not np.array_equal(source.geometry.coordinates, affine.coordinates)
        ):
            raise ValueError("Coordinate optimization requires affine mesh geometry.")
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
                "Coordinate optimization cannot discard revision-bound metadata."
            )
        fixed, _ = _coordinate_projector(source, proposal, policy)
        plan = TargetMatrixOptimizationPlan(
            source.mesh,
            target_coordinates=projection.target_coordinates,
            fixed_vertices=fixed,
            coordinate_bounds=_optimization_bounds(source, policy),
            termination=OptimizationTermination(
                maximum_steps=policy.maximum_optimization_iterations
            ),
        )
        with measure_phase(record_phase, "improvement"):
            optimization = optimize_cell_mesh(
                plan,
                source.coordinate_contract,
                numeric_version=f"proposal:{projection.projection_id}",
            )
            if record_phase is not None:
                jax.block_until_ready(optimization)
        if optimization.accepted:
            candidate = optimization.result
    elif projection.marked_cell_ids.size or not isinstance(proposal, MeshMarkingProposal):
        with measure_phase(record_phase, "topology_adaptation"):
            adaptation = _execute_adaptation(
                source, projection, native_policy, hierarchy, layer_columns
            )
            if record_phase is not None:
                jax.block_until_ready(adaptation)
        candidate = adaptation.target
    if not isinstance(candidate, CellMeshingResult):
        raise TypeError("Native proposal execution must return CellMeshingResult.")
    with measure_phase(record_phase, "audit"):
        audit = audit_cell_mesh(
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
    compliance = MeshingComplianceReport(
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
