#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any, assert_never, ClassVar, final, Literal

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
from .._validation import canonical_identifier, positive_integer
from ..discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    PolygonalConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._cell_geometry import PerCellRestrictedCellGeometryElement
from ..discretization._cell_geometry_transition_types import cell_geometry_id
from ..discretization._cell_mesh import SimplicialConnectivity
from ..geometry.implicit import AnalyticImplicitProfile
from ..optim import OptimizationTermination
from ..typing import checked, Dim, Float64, Identifier, Identifiers, parse
from ._adaptation import (
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationResult,
    MeshAdaptationRoute,
    MetricMeshAdaptation,
    prepare_mesh_adaptation,
)
from ._association import GeometryAssociationKind
from ._audit import audit_cell_mesh, CellMeshAuditPolicy, CellMeshAuditReport
from ._implicit_association_transfer import ImplicitAssociationTransfer
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
from ._optimization import (
    MeshOptimizationResult,
    optimize_cell_mesh,
    TargetMatrixOptimizationPlan,
)
from ._quality import evaluate_cell_quality
from ._result import CellMeshingResult, MeshingComplianceReport
from ._scope import MeshingEntityKind, MeshingScope
from ._sizing import ResolvedSizeField, SizeFieldDomain
from .providers._native_sources import NativeImplicitSource


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
    """Untrusted proposal values bound to one source scope and evaluation."""

    scope: MeshingScope
    values: Array
    coordinate_contract_id: str = eqx.field(static=True)
    proposer_id: str = eqx.field(static=True)
    evaluation_id: str | None = eqx.field(static=True)
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
        evaluation_id: str | None = None,
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
        proposer = canonical_identifier(proposer_id, "proposer_id")
        evaluation = (
            None
            if evaluation_id is None
            else canonical_identifier(evaluation_id, "evaluation_id")
        )
        self.scope = scope
        self.values = jnp.asarray(data)
        self.coordinate_contract_id = source.coordinate_contract.spatial_id
        self.proposer_id = proposer
        self.evaluation_id = evaluation
        self.proposal_id = canonical_fingerprint(
            {
                "kind": type(self).__name__,
                "scope": scope.scope_id,
                "values": array_tree_fingerprint(self.values),
                "coordinate_contract": self.coordinate_contract_id,
                "proposer": proposer,
                "evaluation": evaluation,
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
        source: CellMeshingResult,
        scope: MeshingScope,
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
        source: CellMeshingResult,
        scope: MeshingScope,
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

    def __init__(
        self,
        source: CellMeshingResult,
        scope: MeshingScope,
        coordinates: ArrayLike,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        proposer_id: str,
        evaluation_id: str | None = None,
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
            evaluation_id=evaluation_id,
        )


MeshProposal = (
    MeshMarkingProposal | MeshSizeProposal | MeshMetricProposal | MeshCoordinateProposal
)

type MeshProposerKind = Literal["marking", "size", "metric", "coordinate"]


class _ProposalEntityDim(Dim):
    """Entities in the source-bound feature table."""


class _ProposalFeatureDim(Dim, minimum=1):
    """Named feature columns in the source-bound feature table."""


@final
class MeshProposalFeatures(StrictModule, NonTrainableState):
    """Named feature columns bound to one exact result revision and scope."""

    __strict_contract__ = True

    values: Float64[_ProposalEntityDim, _ProposalFeatureDim]
    scope: MeshingScope
    source_result_id: Identifier = eqx.field(static=True)
    feature_ids: Identifiers[_ProposalFeatureDim] = eqx.field(static=True)
    feature_owner_id: Identifier = eqx.field(static=True)
    feature_id: Identifier = eqx.field(static=True)

    @checked
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
        _scope_rows(source, scope)
        columns = tuple(
            canonical_identifier(value, "feature_ids") for value in feature_ids
        )
        data = np.asarray(values, dtype=np.float64)
        if (
            not columns
            or len(set(columns)) != len(columns)
            or data.shape != (scope.entity_ids.size, len(columns))
            or not np.all(np.isfinite(data))
        ):
            raise ValueError(
                "Feature values must be finite rows aligned with unique named source-scope columns."
            )
        owner = canonical_identifier(feature_owner_id, "feature_owner_id")
        self.values = jnp.asarray(data, dtype=jnp.float64)
        self.scope = scope
        self.source_result_id = canonical_identifier(source.result_id, "source_result_id")
        self.feature_ids = columns
        self.feature_owner_id = owner
        self.feature_id = canonical_fingerprint(
            {
                "kind": "mesh-proposal-features",
                "source": self.source_result_id,
                "scope": scope.scope_id,
                "columns": columns,
                "owner": owner,
                "values": array_tree_fingerprint(self.values),
            }
        )


class AbstractMeshProposer(AbstractComponentSlot):
    """Source of untrusted typed mesh proposals.

    The base is the neutral `DECISION` slot of mesh adaptation: a proposer
    decides where and how a certified mesh should adapt, but it never produces
    a mesh. `propose(source, features, scope=...)` returns one typed marking,
    size, metric, or coordinate proposal bound to the exact source revision.
    Feature values are immutable and bind their producing owner, named columns,
    exact source result, and sorted entity scope; raw same-shaped rows are refused.
    Only `project_mesh_proposal` and `prepare_mesh_proposal` turn a proposal into
    a trusted candidate under the safety policy and native adaptation.
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
    ) -> (
        MeshMarkingProposal
        | MeshSizeProposal
        | MeshMetricProposal
        | MeshCoordinateProposal
    ):
        raise NotImplementedError


def _model_value_shape(size: Any, /) -> tuple[int, ...]:
    if size == "scalar":
        return ()
    if isinstance(size, int):
        return (int(size),)
    return tuple(size)


class LearnedMeshProposer(AbstractMeshProposer):
    """Pointwise learned marking, size, metric, or coordinate proposer.

    The model maps one row of source-bound, owner-identified features to one
    proposal value. Feature identity binds the exact source revision, entity
    scope, ordered column names, owner, and values before model evaluation.
    `evaluate(features)` remains the differentiable per-entity training map;
    only `propose` accepts a `MeshProposalFeatures` record and produces an
    untrusted, revision-bound proposal.

    The model is a dynamic child whose arrays retain their parameter roles and
    is bound to the `AbstractMeshProposer` `DECISION` slot.
    """

    model: AbstractArrayModel
    ports: ModelPorts | None
    port_mapping: PortMapping | None
    kind: MeshProposerKind = eqx.field(static=True)
    spatial_dimension: int | None = eqx.field(static=True)
    proposer_id: str = eqx.field(static=True)
    feature_width: int = eqx.field(static=True)

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
        dimension: int | None
        value_shape: tuple[int, ...]
        match kind:
            case "marking" | "size":
                if spatial_dimension is not None:
                    raise ValueError(
                        "Scalar marking and size proposers do not accept spatial_dimension."
                    )
                dimension = None
                value_shape = ()
            case "metric":
                if (
                    isinstance(spatial_dimension, bool)
                    or not isinstance(spatial_dimension, int)
                    or spatial_dimension <= 0
                ):
                    raise ValueError(
                        "Metric proposers require a positive spatial_dimension."
                    )
                dimension = spatial_dimension
                value_shape = (dimension, dimension)
            case "coordinate":
                if (
                    isinstance(spatial_dimension, bool)
                    or not isinstance(spatial_dimension, int)
                    or spatial_dimension <= 0
                ):
                    raise ValueError(
                        "Coordinate proposers require a positive spatial_dimension."
                    )
                dimension = spatial_dimension
                value_shape = (dimension,)
            case _:
                assert_never(kind)
        if _model_value_shape(model.out_size) != value_shape:
            raise ValueError(
                f"Learned {kind} proposer out_size must produce shape {value_shape}; "
                f"got {model.out_size!r}."
            )
        identifier = canonical_identifier(proposer_id, "proposer_id")
        site = "LearnedMeshProposer"
        input_width = model.in_size
        if isinstance(input_width, bool) or not isinstance(input_width, int):
            raise TypeError("Learned mesh proposer in_size must be one feature width.")
        feature_width = positive_integer(input_width, "model.in_size")
        owner_ports = require_port_shapes(
            ports, inputs=((feature_width,),), outputs=(value_shape,), site=site
        )
        bind_positional_component(
            model, AbstractMeshProposer, owner_ports, port_mapping, site=site
        )
        self.model = model
        self.ports = owner_ports
        self.port_mapping = port_mapping
        self.kind = kind
        self.spatial_dimension = dimension
        self.feature_width = feature_width
        self.proposer_id = identifier

    def component_contract(self) -> ComponentContract:
        """Return the model's contract bound to the mesh-proposer slot."""
        return bind_component(
            self.model,
            AbstractMeshProposer,
            owner_ports=self.ports,
            port_mapping=self.port_mapping,
        ).contract()

    def evaluate(self, features: ArrayLike, /) -> Array:
        """Return one proposal value per feature row, shape `(rows,) + value shape`."""
        rows = jnp.asarray(features)
        if rows.ndim != 2 or rows.shape[1] != self.feature_width:
            raise ValueError(
                f"Proposer features must have shape (entities, {self.feature_width}); "
                f"got {rows.shape}."
            )
        binding = self.model.input_binding()
        return jax.vmap(
            lambda row: binding.call(self.model, row, key=None, iter_=None, kwargs={})
        )(rows)

    @checked
    def propose(
        self,
        source: CellMeshingResult,
        features: MeshProposalFeatures,
        /,
        *,
        scope: MeshingScope | None = None,
    ) -> (
        MeshMarkingProposal
        | MeshSizeProposal
        | MeshMetricProposal
        | MeshCoordinateProposal
    ):
        if features.source_result_id != source.result_id:
            raise ValueError("Feature rows are stale for this source result.")
        match self.kind:
            case "marking":
                dimension = source.mesh.topological_dimension
            case "size":
                dimension = 0
            case "metric" | "coordinate":
                dimension = 0
                if self.spatial_dimension != source.mesh.ambient_dimension:
                    raise ValueError(
                        "Metric and coordinate proposer dimensions must match the source mesh."
                    )
            case _:
                assert_never(self.kind)
        scope_ = mesh_proposal_scope(source, dimension) if scope is None else scope
        _scope_rows(source, scope_)
        if scope_.entity_dimension != dimension:
            raise ValueError("Proposer scope has the wrong entity dimension.")
        if features.scope.scope_id != scope_.scope_id:
            raise ValueError("Feature rows are bound to a different entity scope.")
        rows = features.values
        if rows.shape != (scope_.entity_ids.size, self.feature_width):
            raise ValueError(
                "Feature columns must match the learned model input width and scope rows."
            )
        values = np.asarray(self.evaluate(rows), dtype=np.float64)
        evaluation_id = canonical_fingerprint(
            {
                "kind": "learned-mesh-proposer-evaluation",
                "proposer": self.proposer_id,
                "model": array_tree_fingerprint(self.model),
                "features": features.feature_id,
                "scope": scope_.scope_id,
                "output": array_tree_fingerprint(values),
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
                assert_never(self.kind)


class MeshProposalSafetyPolicy(StrictModule, NonTrainableState):
    """Trusted, revision-bound constraints, independent of proposer evidence.

    Protected scopes preserve their entities and fix their incident vertices.
    Coordinate bounds and displacement are in source coordinate units.
    """

    source_result_id: str = eqx.field(static=True)
    protected_scopes: tuple[MeshingScope, ...]
    audit_policy: CellMeshAuditPolicy
    coordinate_bounds: Array | None
    minimum_size: float = eqx.field(static=True)
    maximum_size: float = eqx.field(static=True)
    maximum_anisotropy: float = eqx.field(static=True)
    maximum_gradation: float = eqx.field(static=True)
    maximum_displacement: float = eqx.field(static=True)
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
        audit_policy: CellMeshAuditPolicy | None = None,
        coordinate_bounds: ArrayLike | None = None,
        maximum_anisotropy: float = 100.0,
        maximum_gradation: float = 1.3,
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
        iterations = positive_integer(
            maximum_optimization_iterations, "maximum_optimization_iterations"
        )
        protected = tuple(protected_scopes)
        for scope in protected:
            _scope_rows(source, scope)
        audit_ = CellMeshAuditPolicy() if audit_policy is None else audit_policy
        if not isinstance(audit_, CellMeshAuditPolicy):
            raise TypeError("audit_policy must be CellMeshAuditPolicy or None.")
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
        self.audit_policy = audit_
        self.coordinate_bounds = None if bounds is None else jnp.asarray(bounds)
        self.minimum_size, self.maximum_size = minimum, maximum
        self.maximum_anisotropy, self.maximum_gradation = anisotropy, gradation
        self.maximum_displacement = displacement
        self.maximum_optimization_iterations = iterations
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mesh-proposal-safety-policy",
                "source": source.result_id,
                "protected": tuple(scope.scope_id for scope in protected),
                "audit": audit_.policy_id,
                "bounds": None
                if bounds is None
                else array_tree_fingerprint(self.coordinate_bounds),
                "minimum_size": minimum,
                "maximum_size": maximum,
                "anisotropy": anisotropy,
                "gradation": gradation,
                "displacement": displacement,
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
    if not source.compliance.passed:
        raise ValueError("Source does not satisfy scientific proposal compliance.")
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


def _safe_marks(proposal: MeshMarkingProposal, /) -> np.ndarray:
    """Preserve every positive mark in canonical persistent-ID order.

    Native bisection owns protection and conformity closure. Selection here
    cannot silently discard a scientific request to fit a resource allowance.
    """
    cells = np.asarray(proposal.scope.entity_ids, dtype=np.int64)
    return np.sort(cells[np.asarray(proposal.values) > 0])


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

    Marking proposals project onto every positive mark, executed by native
    triangle or tetrahedron bisection. Size and metric proposals project onto a
    graded vertex size field or a normalized vertex metric executed by native
    planar, surface-chart or tetrahedral anisotropic metric adaptation toward
    the unit-mesh criterion.
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
        if len(mesh.blocks) != 1 or (
            mesh.topological_dimension,
            mesh.blocks[0].cell_kind,
        ) not in ((2, "triangle"), (3, "tetrahedron"), (3, "simplex:3")):
            raise ValueError(
                "Native marking, size and metric proposals require one T3 or T4 block."
            )
        if isinstance(proposal, MeshMarkingProposal):
            marks = _safe_marks(proposal)
        else:
            if rows.size != mesh.coordinates.shape[0]:
                raise ValueError(
                    "Size and metric proposals must cover every source vertex."
                )
            match mesh.connectivity:
                case PolygonalConnectivity() | TetrahedralConnectivity() as connectivity:
                    edges = np.asarray(connectivity.edges, dtype=np.int64)
                case SimplicialConnectivity() as connectivity:
                    edges = np.asarray(connectivity.entities[1], dtype=np.int64)
                case _:
                    raise TypeError(
                        "Native simplex proposals require canonical edge connectivity."
                    )
            # Field arrays are scope ordered; connectivity and points are mesh ordered.
            inverse = np.empty(rows.size, dtype=np.int64)
            inverse[rows] = np.arange(rows.size)
            field_edges = inverse[edges]
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
        issues = self.compliance.issues
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


def _proposal_association_transfer(
    source: CellMeshingResult, /
) -> ImplicitAssociationTransfer | None:
    if not source.associations and source.boundary is None:
        return None
    certification = source.certification
    if certification is None:
        raise ValueError(
            "Source-bound proposals require the current certification feature source."
        )
    authority = certification.inputs.feature_source
    if not isinstance(authority, (AnalyticImplicitProfile, NativeImplicitSource)):
        raise ValueError(
            "Proposal association transfer requires a source-bound analytic "
            "implicit authority."
        )
    transfer = ImplicitAssociationTransfer(authority)
    profile = transfer.profile
    if not isinstance(profile, AnalyticImplicitProfile):
        raise ValueError(
            "Native implicit proposal transfer requires its exact authoring profile."
        )
    if any(
        association.association_kind is not GeometryAssociationKind.IMPLICIT
        or (association.source_id, association.source_revision)
        != (profile.source_id, profile.source_revision)
        for association in source.associations
    ):
        raise ValueError(
            "Native proposals cannot discard associations outside the exact "
            "analytic implicit source."
        )
    transfer.validate_source(source)
    return transfer


def _execute_adaptation(
    source: CellMeshingResult,
    projection: MeshProposalProjection,
    association_transfer: ImplicitAssociationTransfer | None,
    /,
) -> MeshAdaptationResult:
    policy = projection.policy
    protected = tuple(_mesh_scope(source, scope) for scope in policy.protected_scopes)
    if isinstance(projection.proposal, MeshMarkingProposal):
        request = MarkedMeshAdaptation(projection.marked_cell_ids)
        route = MeshAdaptationRoute.NATIVE_BISECTION
    else:
        request = MetricMeshAdaptation(_adaptation_metric(source, projection))
        match (source.mesh.topological_dimension, source.mesh.ambient_dimension):
            case (2, 2):
                route = MeshAdaptationRoute.NATIVE_METRIC_2D
            case (3, 3):
                route = MeshAdaptationRoute.NATIVE_METRIC_3D
            case (2, 3):
                if source.surface_source is None and association_transfer is None:
                    raise ValueError(
                        "MeshAdaptationRoute.NATIVE_METRIC_SURFACE requires "
                        "retained SurfaceSourceCharts or source-bound implicit "
                        "chart evidence."
                    )
                route = MeshAdaptationRoute.NATIVE_METRIC_SURFACE
            case _:
                raise ValueError(
                    "Metric proposals require planar triangles, embedded "
                    "surface charts, or three-dimensional tetrahedra."
                )
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source,
            request,
            policy=MeshAdaptationPolicy(
                route,
                protected_scopes=protected,
                association_transfer=association_transfer,
            ),
        )
    )


def prepare_mesh_proposal(
    source: CellMeshingResult,
    proposal: MeshProposal,
    policy: MeshProposalSafetyPolicy,
    /,
) -> MeshProposalTransaction:
    """Project, execute a native trusted path, audit and prepare atomic promotion.

    No caller-supplied mesh or audit is accepted as proposal evidence. Proposal
    execution requires affine input geometry. Topology changes preserve canonical
    lineage organization and accept associations only through their exact source
    transfer; attributes and unsupported association families are refused.
    An empty projected marking is an explicit unchanged-source transaction.
    """
    projection = project_mesh_proposal(source, proposal, policy)
    affine = CellGeometrySpec.affine(source.mesh)
    if (
        source.geometry.geometry_layout_id != affine.geometry_layout_id
        or not np.array_equal(source.geometry.coordinates, affine.coordinates)
    ):
        raise ValueError("Native proposal execution requires affine mesh geometry.")
    if source.attributes:
        raise ValueError("Native proposals cannot discard revision-bound attributes.")
    association_transfer = None
    if isinstance(proposal, MeshCoordinateProposal):
        if source.boundary is not None or any(
            (source.patches, source.zones, source.labels, source.associations)
        ):
            raise ValueError(
                "Coordinate proposals cannot remap revision-bound mesh metadata."
            )
    else:
        association_transfer = _proposal_association_transfer(source)
    adaptation, optimization = None, None
    candidate = source
    if isinstance(proposal, MeshCoordinateProposal):
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
        optimization = optimize_cell_mesh(
            plan,
            source.coordinate_contract,
            numeric_version=f"proposal:{projection.projection_id}",
        )
        if optimization.accepted:
            if optimization.result is None:
                raise RuntimeError("Accepted mesh optimization omitted its result.")
            candidate = optimization.result
    elif projection.marked_cell_ids.size or not isinstance(proposal, MeshMarkingProposal):
        adaptation = _execute_adaptation(source, projection, association_transfer)
        candidate = adaptation.target
    affine_target = CellGeometrySpec.affine(candidate.mesh)
    if (
        candidate.geometry.geometry_layout_id == affine_target.geometry_layout_id
        and np.array_equal(candidate.geometry.coordinates, affine_target.coordinates)
    ):
        quality_coordinates = candidate.mesh.coordinates
    else:
        restriction = candidate.geometry.restriction_source
        # Adaptation binds restrictions to the source's retained root, if any.
        root = source.geometry.restriction_source
        root_geometry_id, root_topology_id = (
            (cell_geometry_id(source.geometry), source.mesh.topology_id)
            if root is None
            else (root.source_geometry_id, root.source_topology_id)
        )
        if (
            restriction is None
            or restriction.source_geometry_id != root_geometry_id
            or restriction.source_topology_id != root_topology_id
            or len(candidate.geometry.elements) != len(candidate.mesh.blocks)
            or any(
                not isinstance(element, PerCellRestrictedCellGeometryElement)
                or not isinstance(block, CellBlock)
                or element.degree != 1
                or element.cell_kind != block.cell_kind
                or element.local_dof_count != block.arity
                for element, block in zip(
                    candidate.geometry.elements, candidate.mesh.blocks, strict=True
                )
            )
        ):
            raise ValueError(
                "Native proposal quality requires an affine or exact P1-restricted target map."
            )
        quality_coordinates = candidate.mesh.coordinates
    quality = evaluate_cell_quality(candidate.mesh, quality_coordinates)
    audit = audit_cell_mesh(
        candidate.mesh,
        candidate.geometry,
        quality,
        policy=policy.audit_policy,
    )
    issues = _preservation_issues(source, candidate, projection)
    if not audit.passed:
        issues += ("safety_audit",)
    if optimization is not None and not optimization.accepted:
        issues += ("mesh_optimization",)
    if adaptation is not None:
        issues += adaptation.compliance.issues
    compliance = MeshingComplianceReport(
        projection.projection_id,
        issues=tuple(dict.fromkeys(issues)),
        requested=candidate.compliance.requested,
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
    "MeshProposal",
    "MeshProposalFeatures",
    "MeshProposalProjection",
    "MeshProposalSafetyPolicy",
    "MeshProposalTransaction",
    "MeshSizeProposal",
    "mesh_proposal_scope",
    "prepare_mesh_proposal",
    "project_mesh_proposal",
]
