#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Declarative runtime and training contracts for neural operators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Sequence, TypeAlias

import jax
import numpy as np
from jax import core as jax_core

from .data import FunctionSamples, OperatorBatch


OperatorGeometryKind: TypeAlias = Literal[
    "abstract",
    "tensor_grid",
    "point_cloud",
    "graph",
    "simplicial",
    "cell_complex",
    "sphere",
    "manifold",
]
OperatorSourceQueryRelation: TypeAlias = Literal[
    "coincident",
    "shared_topology",
    "independent",
]
OperatorAxisRequirement: TypeAlias = Literal[
    "none",
    "uniform",
    "periodic_uniform",
    "periodic_square",
]
OperatorQuadraturePolicy: TypeAlias = Literal[
    "unused",
    "optional",
    "physical_required",
]
OperatorMaskPolicy: TypeAlias = Literal[
    "unsupported",
    "all_valid_only",
    "supported",
]
OperatorTopologyPolicy: TypeAlias = Literal["unused", "optional", "required"]
OperatorCochainPolicy: TypeAlias = Literal["unsupported", "optional", "required"]
OperatorFieldRepresentation: TypeAlias = Literal[
    "generic_channels",
    "scalar",
    "pseudoscalar",
    "vector",
    "covector",
    "tensor",
]
OperatorTrainingRegime: TypeAlias = Literal[
    "task_specific",
    "pretrained_system",
    "task_distribution",
]
OperatorCompatibilityCode: TypeAlias = Literal[
    "SOURCE_COUNT",
    "MULTIPLE_QUERIES_UNSUPPORTED",
    "FIELD_SCHEMA_MISMATCH",
    "UNSUPPORTED_GEOMETRY",
    "UNSUPPORTED_SPATIAL_DIMENSION",
    "INVALID_GLOBAL_CONDITION",
    "MISSING_PHYSICAL_QUADRATURE",
    "MASKED_INPUT_UNSUPPORTED",
    "TOPOLOGY_REQUIRED",
    "TOPOLOGY_UNUSED",
    "TENSOR_GRID_REQUIRED",
    "NONUNIFORM_AXIS",
    "NONPERIODIC_AXIS",
    "NON_SQUARE_GROUP_AXES",
    "AXIS_TOO_SMALL",
    "AXIS_SIZE_DIVISIBILITY",
    "SOURCE_QUERY_RELATION",
    "FIXED_QUERY_REQUIRED",
    "UNKNOWN_SOURCE_REPRESENTATION",
    "UNSUPPORTED_FIELD_REPRESENTATION",
    "UNSUPPORTED_OUTPUT_REPRESENTATION",
    "COCHAIN_SEMANTICS_REQUIRED",
    "COCHAIN_SEMANTICS_UNSUPPORTED",
    "UNSUPPORTED_COCHAIN_SIDE",
    "COCHAIN_DEGREE_MISMATCH",
    "COCHAIN_TOPOLOGY_MISMATCH",
    "COCHAIN_METRIC_REQUIRED",
    "UNSUPPORTED_SYMMETRY_GROUP",
    "RESOLUTION_TRANSFER_UNSUPPORTED",
    "ENCODE_ONCE_DECODE_MANY_UNSUPPORTED",
    "ROLLOUT_UNSUPPORTED",
    "MISSING_PRETRAINED_WEIGHTS",
    "MISSING_TASK_DISTRIBUTION_TRAINING",
    "TRAINING_REGIME_MISMATCH",
]


@dataclass(frozen=True, slots=True)
class OperatorCapabilitySpec:
    """Declared execution envelope for one configured operator architecture."""

    source_geometries: tuple[OperatorGeometryKind, ...]
    query_geometries: tuple[OperatorGeometryKind, ...]
    spatial_dimensions: tuple[int, ...] = ()
    source_query_relations: tuple[OperatorSourceQueryRelation, ...] = ("coincident",)
    requires_fixed_query: bool = False
    axis_requirement: OperatorAxisRequirement = "none"
    quadrature: OperatorQuadraturePolicy = "optional"
    masks: OperatorMaskPolicy = "supported"
    topology: OperatorTopologyPolicy = "unused"
    cochains: OperatorCochainPolicy = "unsupported"
    cochain_sides: tuple[Literal["primal", "dual"], ...] = ("primal",)
    input_representations: tuple[OperatorFieldRepresentation, ...] = (
        "generic_channels",
        "scalar",
    )
    output_representations: tuple[OperatorFieldRepresentation, ...] = (
        "generic_channels",
        "scalar",
    )
    symmetry_groups: tuple[str, ...] = ()
    global_condition_sources: tuple[str, ...] = ()
    minimum_sources: int = 1
    maximum_sources: int | None = None
    minimum_axis_size: int | None = None
    axis_size_divisor: int | None = None
    resolution_transfer: bool = False
    encode_once_decode_many: bool = False
    multiple_queries: bool = False
    autoregressive_rollout: bool = False

    def __post_init__(self):
        if not self.source_geometries or not self.query_geometries:
            raise ValueError(
                "Operator capabilities must declare source and query geometries."
            )
        if any(int(value) <= 0 for value in self.spatial_dimensions):
            raise ValueError("spatial_dimensions must contain positive dimensions.")
        if int(self.minimum_sources) <= 0:
            raise ValueError("minimum_sources must be positive.")
        if self.maximum_sources is not None and int(self.maximum_sources) < int(
            self.minimum_sources
        ):
            raise ValueError("maximum_sources cannot be smaller than minimum_sources.")
        if self.minimum_axis_size is not None and int(self.minimum_axis_size) <= 0:
            raise ValueError("minimum_axis_size must be positive.")
        if self.axis_size_divisor is not None and int(self.axis_size_divisor) <= 0:
            raise ValueError("axis_size_divisor must be positive.")
        if len(set(self.global_condition_sources)) != len(self.global_condition_sources):
            raise ValueError("global_condition_sources must be unique.")
        if self.coarsened_cochain_policy_invalid:
            raise ValueError("Invalid cochain capability policy.")

    @property
    def coarsened_cochain_policy_invalid(self) -> bool:
        return self.cochains not in ("unsupported", "optional", "required") or any(
            side not in ("primal", "dual") for side in self.cochain_sides
        )


@dataclass(frozen=True, slots=True)
class OperatorTrainingRequirement:
    """Training evidence required before an architecture can support its stated claim."""

    regime: OperatorTrainingRegime = "task_specific"
    pretrained_weights_required: bool = False
    corpus_description: str = ""
    claim_scope: str = "task-specific operator learning"

    def __post_init__(self):
        if self.regime == "pretrained_system" and not self.pretrained_weights_required:
            raise ValueError("pretrained_system regimes require pretrained weights.")
        if self.regime == "task_distribution" and not self.corpus_description.strip():
            raise ValueError("task_distribution regimes require a corpus description.")
        if not self.claim_scope.strip():
            raise ValueError("claim_scope must be non-empty.")


@dataclass(frozen=True, slots=True)
class OperatorTrainingEvidence:
    """Concrete training evidence available to one model evaluation."""

    regime: OperatorTrainingRegime
    checkpoint_id: str = ""
    corpus_id: str = ""


@dataclass(frozen=True, slots=True)
class OperatorProblemSpec:
    """Physical requirements that are not inferable from array shapes alone."""

    source_query_relation: OperatorSourceQueryRelation | None = None
    query_is_fixed: bool | None = None
    symmetry_group: str | None = None
    requires_resolution_transfer: bool = False
    requires_encode_once_decode_many: bool = False
    rollout_steps: int = 1

    def __post_init__(self):
        if self.source_query_relation not in (
            None,
            "coincident",
            "shared_topology",
            "independent",
        ):
            raise ValueError(
                "source_query_relation must be 'coincident', 'shared_topology', "
                "'independent', or None."
            )
        if int(self.rollout_steps) <= 0:
            raise ValueError("rollout_steps must be positive.")


@dataclass(frozen=True, slots=True)
class OperatorCompatibilityIssue:
    """One stable machine-readable incompatibility."""

    code: OperatorCompatibilityCode
    message: str
    location: str = ""


@dataclass(frozen=True, slots=True)
class OperatorCompatibilityReport:
    """Structured result of validating a configured operator contract."""

    architecture: str
    configuration: tuple[tuple[str, object], ...]
    issues: tuple[OperatorCompatibilityIssue, ...]

    @property
    def accepted(self) -> bool:
        return not self.issues

    @property
    def runtime_accepted(self) -> bool:
        """Whether geometry, representation, and execution requirements are met."""
        return not any(issue.location != "training" for issue in self.issues)

    @property
    def training_accepted(self) -> bool:
        """Whether evidence satisfies the architecture's stated training claim."""
        return not any(issue.location == "training" for issue in self.issues)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    def require(self) -> None:
        if self.issues:
            detail = "; ".join(f"{issue.code}: {issue.message}" for issue in self.issues)
            raise ValueError(
                f"Operator architecture {self.architecture!r} is incompatible: {detail}"
            )

    def require_runtime(self) -> None:
        """Raise only for geometry, representation, or execution incompatibility."""
        issues = tuple(issue for issue in self.issues if issue.location != "training")
        if issues:
            detail = "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
            raise ValueError(
                f"Operator architecture {self.architecture!r} is incompatible: {detail}"
            )


@dataclass(frozen=True, slots=True)
class ConfiguredOperatorContract:
    """Single source of truth for runtime support and required training evidence."""

    architecture: str
    configuration: tuple[tuple[str, object], ...]
    capabilities: OperatorCapabilitySpec
    training: OperatorTrainingRequirement
    field_specs: tuple[Any, ...] = ()

    def validate(
        self,
        batch: OperatorBatch,
        /,
        *,
        problem: OperatorProblemSpec | None = None,
        training_evidence: OperatorTrainingEvidence | None = None,
        fields: Sequence[Any] = (),
    ) -> OperatorCompatibilityReport:
        supplied_fields = tuple(fields)
        return validate_operator_contract(
            self,
            batch,
            problem=problem,
            training_evidence=training_evidence,
            fields=supplied_fields if supplied_fields else self.field_specs,
            configured_fields=self.field_specs,
        )


def _geometry_kind(samples: FunctionSamples, /) -> OperatorGeometryKind:
    if samples.topology is not None:
        return samples.topology.kind
    if samples.axes:
        if any(axis.basis == "sphere" for axis in samples.axes):
            return "sphere"
        return "tensor_grid"
    if samples.coordinates is not None:
        return "point_cloud"
    return "abstract"


def _coordinate_dimension(samples: FunctionSamples, /) -> int | None:
    if samples.axes:
        return len(samples.axes)
    if samples.coordinates is not None:
        return int(samples.coordinates.shape[-1])
    return None


def _contains_tracer(value: object, /) -> bool:
    return any(
        isinstance(leaf, jax_core.Tracer) for leaf in jax.tree_util.tree_leaves(value)
    )


def _same_geometry(left: FunctionSamples, right: FunctionSamples, /) -> bool:
    if left.sample_shape != right.sample_shape or len(left.axes) != len(right.axes):
        return False
    if left.topology is not None or right.topology is not None:
        if left.topology is None or right.topology is None:
            return False
        return (
            left.topology.graph_fingerprint == right.topology.graph_fingerprint
            and left.topology.entity == right.topology.entity
            and (
                left.topology.sample_entities.shape
                == right.topology.sample_entities.shape
                if _contains_tracer(
                    (
                        left.topology.sample_entities,
                        right.topology.sample_entities,
                    )
                )
                else np.array_equal(
                    np.asarray(left.topology.sample_entities),
                    np.asarray(right.topology.sample_entities),
                )
            )
        )
    if left.axes:
        return all(
            first.name == second.name
            and first.basis == second.basis
            and first.periodic == second.periodic
            and (
                first.nodes.shape == second.nodes.shape
                if _contains_tracer((first.nodes, second.nodes))
                else np.array_equal(np.asarray(first.nodes), np.asarray(second.nodes))
            )
            for first, second in zip(left.axes, right.axes, strict=True)
        )
    if left.coordinates is None or right.coordinates is None:
        return left.coordinates is right.coordinates
    if _contains_tracer((left.coordinates, right.coordinates)):
        return left.coordinates.shape == right.coordinates.shape
    return np.array_equal(np.asarray(left.coordinates), np.asarray(right.coordinates))


def _shared_topology(left: FunctionSamples, right: FunctionSamples, /) -> bool:
    return (
        left.topology is not None
        and right.topology is not None
        and left.topology.graph_fingerprint == right.topology.graph_fingerprint
    )


def _field_schema_signature(field: Any, /) -> tuple[Any, ...]:
    cochain = field.cochain
    output = field.output_spec
    return (
        field.channels,
        field.role,
        field.representation,
        field.source_name,
        field.query_name,
        None if output is None else (output.channels, tuple(output.component_names)),
        tuple(field.component_names),
        tuple(field.physical_dimension),
        tuple(field.scale),
        tuple(field.offset),
        field.required,
        None
        if cochain is None
        else (
            cochain.degree,
            cochain.complex_side,
            cochain.cell_orientation,
            cochain.sampling,
        ),
    )


def _mapped_node_payload(samples: FunctionSamples, key: str, /) -> np.ndarray | None:
    topology = samples.topology
    if topology is None or topology.entity != "node":
        return None
    nodes = topology.graph.nodes
    if not isinstance(nodes, Mapping) or key not in nodes:
        return None
    mapping_values = topology.absolute_sample_entities()
    node_values = nodes[key]
    if any(
        isinstance(leaf, jax_core.Tracer)
        for leaf in jax.tree_util.tree_leaves((mapping_values, node_values))
    ):
        return np.empty((0,), dtype=float)
    mapping = np.asarray(mapping_values)
    valid = mapping >= 0
    values = np.asarray(node_values)
    return values[mapping[valid]] if np.any(valid) else values[:0]


def _all_valid(samples: FunctionSamples, /) -> bool:
    if samples.mask is None or _contains_tracer(samples.mask):
        return True
    return bool(np.all(np.asarray(samples.mask)))


def _uniform_axis(nodes: object, /) -> bool:
    if _contains_tracer(nodes):
        return True
    array = np.asarray(nodes)
    if array.size < 2:
        return False
    spacing = np.diff(array)
    return bool(np.allclose(spacing, np.mean(spacing), rtol=1e-5, atol=1e-8))


def _issue(
    code: OperatorCompatibilityCode,
    message: str,
    location: str = "",
) -> OperatorCompatibilityIssue:
    return OperatorCompatibilityIssue(code, message, location)


def validate_operator_contract(
    contract: ConfiguredOperatorContract,
    batch: OperatorBatch,
    /,
    *,
    problem: OperatorProblemSpec | None = None,
    training_evidence: OperatorTrainingEvidence | None = None,
    fields: Sequence[Any] = (),
    configured_fields: Sequence[Any] = (),
) -> OperatorCompatibilityReport:
    """Validate a runtime batch and explicit physical requirements without guessing."""

    if not isinstance(batch, OperatorBatch):
        raise TypeError("Operator contract validation requires an OperatorBatch.")
    capability = contract.capabilities
    physical = OperatorProblemSpec() if problem is None else problem
    issues: list[OperatorCompatibilityIssue] = []
    configured = tuple(configured_fields)
    supplied = tuple(fields)
    if configured and supplied and supplied != configured:
        supplied_by_name = {field.name: field for field in supplied}
        for expected in configured:
            actual = supplied_by_name.get(expected.name)
            if actual is None or _field_schema_signature(
                actual
            ) != _field_schema_signature(expected):
                issues.append(
                    _issue(
                        "FIELD_SCHEMA_MISMATCH",
                        "task field semantics do not match the configured model field",
                        expected.name,
                    )
                )
    all_sources = tuple(batch.inputs.items())
    source_by_name = dict(all_sources)
    for name in capability.global_condition_sources:
        condition = source_by_name.get(name)
        if condition is None or condition.sample_shape != (1,):
            issues.append(
                _issue(
                    "INVALID_GLOBAL_CONDITION",
                    "configured global conditions require one sampled value per case",
                    name,
                )
            )
    sources = tuple(
        (name, samples)
        for name, samples in all_sources
        if name not in capability.global_condition_sources
    )
    source_count = len(sources)
    if source_count < capability.minimum_sources or (
        capability.maximum_sources is not None
        and source_count > capability.maximum_sources
    ):
        issues.append(
            _issue(
                "SOURCE_COUNT",
                f"supports {capability.minimum_sources}..{capability.maximum_sources or 'many'} "
                f"source fields, received {source_count}",
                "inputs",
            )
        )
    if len(batch.queries) > 1 and not capability.multiple_queries:
        issues.append(
            _issue(
                "MULTIPLE_QUERIES_UNSUPPORTED",
                "the configured architecture accepts exactly one query branch",
                "queries",
            )
        )

    all_samples = sources + tuple(
        (f"query:{name}", samples) for name, samples in batch.queries.items()
    )
    for name, samples in all_samples:
        kind = _geometry_kind(samples)
        accepted = (
            capability.query_geometries
            if name.startswith("query:")
            else capability.source_geometries
        )
        if kind not in accepted:
            issues.append(
                _issue(
                    "UNSUPPORTED_GEOMETRY",
                    f"{kind!r} is not one of {accepted}",
                    name,
                )
            )
        dimension = _coordinate_dimension(samples)
        if (
            dimension is not None
            and capability.spatial_dimensions
            and dimension not in capability.spatial_dimensions
        ):
            issues.append(
                _issue(
                    "UNSUPPORTED_SPATIAL_DIMENSION",
                    f"dimension {dimension} is not one of {capability.spatial_dimensions}",
                    name,
                )
            )
        if (
            capability.quadrature == "physical_required"
            and not name.startswith("query:")
            and not samples.has_physical_quadrature
        ):
            issues.append(
                _issue(
                    "MISSING_PHYSICAL_QUADRATURE",
                    "explicit physical quadrature weights are required",
                    name,
                )
            )
        if capability.masks in ("unsupported", "all_valid_only") and not _all_valid(
            samples
        ):
            issues.append(
                _issue(
                    "MASKED_INPUT_UNSUPPORTED",
                    "masked sample sites are not supported",
                    name,
                )
            )
        if capability.topology == "required" and samples.topology is None:
            issues.append(
                _issue(
                    "TOPOLOGY_REQUIRED",
                    "native graph or simplicial topology is required",
                    name,
                )
            )
        if capability.topology == "unused" and samples.topology is not None:
            issues.append(
                _issue(
                    "TOPOLOGY_UNUSED",
                    "the configured model does not consume attached topology",
                    name,
                )
            )
        if capability.axis_requirement != "none":
            if not samples.axes:
                issues.append(
                    _issue(
                        "TENSOR_GRID_REQUIRED",
                        "explicit tensor-grid axes are required",
                        name,
                    )
                )
            else:
                if any(not _uniform_axis(axis.nodes) for axis in samples.axes):
                    issues.append(
                        _issue(
                            "NONUNIFORM_AXIS",
                            "uniformly spaced axes are required",
                            name,
                        )
                    )
                if capability.axis_requirement in (
                    "periodic_uniform",
                    "periodic_square",
                ) and any(not axis.periodic for axis in samples.axes):
                    issues.append(
                        _issue(
                            "NONPERIODIC_AXIS",
                            "periodic axes are required",
                            name,
                        )
                    )
                if (
                    capability.axis_requirement == "periodic_square"
                    and len({axis.size for axis in samples.axes}) != 1
                ):
                    issues.append(
                        _issue(
                            "NON_SQUARE_GROUP_AXES",
                            "group-action axes must have equal sizes",
                            name,
                        )
                    )
                if capability.minimum_axis_size is not None and any(
                    axis.size < capability.minimum_axis_size for axis in samples.axes
                ):
                    issues.append(
                        _issue(
                            "AXIS_TOO_SMALL",
                            f"axis sizes must be at least {capability.minimum_axis_size}",
                            name,
                        )
                    )
                if capability.axis_size_divisor is not None and any(
                    axis.size % capability.axis_size_divisor for axis in samples.axes
                ):
                    issues.append(
                        _issue(
                            "AXIS_SIZE_DIVISIBILITY",
                            f"axis sizes must be divisible by {capability.axis_size_divisor}",
                            name,
                        )
                    )

    inferred_relation: OperatorSourceQueryRelation = "independent"
    if sources and all(
        _same_geometry(source, query)
        for _, source in sources
        for query in batch.queries.values()
    ):
        inferred_relation = "coincident"
    elif sources and all(
        _shared_topology(source, query)
        for _, source in sources
        for query in batch.queries.values()
    ):
        inferred_relation = "shared_topology"
    required_relation = physical.source_query_relation or inferred_relation
    if required_relation not in capability.source_query_relations:
        issues.append(
            _issue(
                "SOURCE_QUERY_RELATION",
                f"{required_relation!r} is not one of {capability.source_query_relations}",
                "query",
            )
        )
    if (
        problem is not None
        and capability.requires_fixed_query
        and physical.query_is_fixed is not True
    ):
        issues.append(
            _issue(
                "FIXED_QUERY_REQUIRED",
                "the configured model requires one fixed query discretization",
                "query",
            )
        )

    cochain_fingerprints: set[str] = set()
    for field in supplied:
        cochain = field.cochain
        if capability.cochains == "required" and cochain is None:
            issues.append(
                _issue(
                    "COCHAIN_SEMANTICS_REQUIRED",
                    "configured architecture requires explicit cochain field semantics",
                    field.name,
                )
            )
        if capability.cochains == "unsupported" and cochain is not None:
            issues.append(
                _issue(
                    "COCHAIN_SEMANTICS_UNSUPPORTED",
                    "configured architecture does not consume cochain field semantics",
                    field.name,
                )
            )
        if cochain is not None and cochain.complex_side not in capability.cochain_sides:
            issues.append(
                _issue(
                    "UNSUPPORTED_COCHAIN_SIDE",
                    f"{cochain.complex_side!r} is not one of {capability.cochain_sides}",
                    field.name,
                )
            )

        bound_samples: list[tuple[str, FunctionSamples]] = []
        if field.is_source:
            assert field.source_name is not None
            if field.source_name not in batch.inputs:
                issues.append(
                    _issue(
                        "UNKNOWN_SOURCE_REPRESENTATION",
                        f"field was bound to unknown input {field.source_name!r}",
                        field.source_name,
                    )
                )
            else:
                bound_samples.append((field.source_name, batch.input(field.source_name)))
                if field.representation not in capability.input_representations:
                    issues.append(
                        _issue(
                            "UNSUPPORTED_FIELD_REPRESENTATION",
                            f"{field.representation!r} is not one of "
                            f"{capability.input_representations}",
                            field.name,
                        )
                    )
        if field.is_target:
            if field.representation not in capability.output_representations:
                issues.append(
                    _issue(
                        "UNSUPPORTED_OUTPUT_REPRESENTATION",
                        f"{field.representation!r} is not one of "
                        f"{capability.output_representations}",
                        field.name,
                    )
                )
            assert field.query_name is not None
            if field.query_name in batch.queries:
                bound_samples.append(
                    (f"query:{field.query_name}", batch.query(field.query_name))
                )

        if cochain is None:
            continue
        for location, samples in bound_samples:
            topology = samples.topology
            if topology is None or topology.kind != "cell_complex":
                issues.append(
                    _issue(
                        "COCHAIN_TOPOLOGY_MISMATCH",
                        "cochain fields require native cell-complex topology",
                        location,
                    )
                )
                continue
            cochain_fingerprints.add(topology.graph_fingerprint)
            degrees = _mapped_node_payload(samples, "cell_dim")
            if degrees is None or np.any(degrees != cochain.degree):
                issues.append(
                    _issue(
                        "COCHAIN_DEGREE_MISMATCH",
                        f"sampled cells do not all have declared degree {cochain.degree}",
                        location,
                    )
                )
            metric = _mapped_node_payload(samples, "hodge_star")
            if (
                metric is None
                or np.any(~np.isfinite(metric))
                or np.any(metric <= 0.0)
                or samples.quadrature_weights is None
            ):
                issues.append(
                    _issue(
                        "COCHAIN_METRIC_REQUIRED",
                        "cochain fields require positive Hodge-star sample weights",
                        location,
                    )
                )
    if len(cochain_fingerprints) > 1:
        issues.append(
            _issue(
                "COCHAIN_TOPOLOGY_MISMATCH",
                "all cochain source and target fields must share one complex",
                "topology",
            )
        )
    if physical.symmetry_group is not None and physical.symmetry_group not in (
        capability.symmetry_groups
    ):
        issues.append(
            _issue(
                "UNSUPPORTED_SYMMETRY_GROUP",
                f"exact {physical.symmetry_group!r} equivariance is not declared",
                "symmetry",
            )
        )
    if physical.requires_resolution_transfer and not capability.resolution_transfer:
        issues.append(
            _issue(
                "RESOLUTION_TRANSFER_UNSUPPORTED",
                "resolution-transfer execution is required",
                "query",
            )
        )
    if (
        physical.requires_encode_once_decode_many
        and not capability.encode_once_decode_many
    ):
        issues.append(
            _issue(
                "ENCODE_ONCE_DECODE_MANY_UNSUPPORTED",
                "reusable source encoding is required",
                "query",
            )
        )
    if physical.rollout_steps > 1 and not capability.autoregressive_rollout:
        issues.append(
            _issue(
                "ROLLOUT_UNSUPPORTED",
                "autoregressive rollout is required",
                "rollout",
            )
        )

    requirement = contract.training
    if requirement.pretrained_weights_required and (
        training_evidence is None or not training_evidence.checkpoint_id.strip()
    ):
        issues.append(
            _issue(
                "MISSING_PRETRAINED_WEIGHTS",
                "the architecture's stated claim requires a pretrained checkpoint",
                "training",
            )
        )
    if requirement.regime == "task_distribution" and (
        training_evidence is None
        or training_evidence.regime != "task_distribution"
        or not training_evidence.corpus_id.strip()
    ):
        issues.append(
            _issue(
                "MISSING_TASK_DISTRIBUTION_TRAINING",
                "the stated claim requires documented task-distribution training",
                "training",
            )
        )
    if training_evidence is not None and training_evidence.regime != requirement.regime:
        issues.append(
            _issue(
                "TRAINING_REGIME_MISMATCH",
                f"requires {requirement.regime!r}, received {training_evidence.regime!r}",
                "training",
            )
        )

    return OperatorCompatibilityReport(
        architecture=contract.architecture,
        configuration=contract.configuration,
        issues=tuple(issues),
    )


__all__ = [
    "ConfiguredOperatorContract",
    "OperatorAxisRequirement",
    "OperatorCapabilitySpec",
    "OperatorCompatibilityCode",
    "OperatorCompatibilityIssue",
    "OperatorCompatibilityReport",
    "OperatorCochainPolicy",
    "OperatorFieldRepresentation",
    "OperatorGeometryKind",
    "OperatorMaskPolicy",
    "OperatorProblemSpec",
    "OperatorQuadraturePolicy",
    "OperatorSourceQueryRelation",
    "OperatorTopologyPolicy",
    "OperatorTrainingEvidence",
    "OperatorTrainingRegime",
    "OperatorTrainingRequirement",
    "validate_operator_contract",
]
