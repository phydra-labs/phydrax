#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..._fingerprint import canonical_fingerprint
from ..._frozendict import frozendict
from ..._strict import StrictModule
from ...equations._ir import PDEProblemIR
from ...equations._serialize import pde_ir_from_dict, pde_ir_to_dict
from ...graph._operator_topology import OperatorTopologySite
from ...units import DimensionSignature
from .capabilities import (
    OperatorGeometryKind,
    OperatorProblemSpec,
    OperatorQuadraturePolicy,
)
from .data import FunctionSamples, OperatorBatch, OperatorPrediction
from .field import OperatorFieldSpec


def _freeze_json(value: Any, /, *, path: str = "metadata") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float.")
        return value
    if isinstance(value, Mapping):
        items: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings.")
            items[key] = _freeze_json(item, path=f"{path}.{key}")
        return frozendict({key: items[key] for key in sorted(items)})
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} must contain only JSON-compatible immutable values; "
        f"got {type(value).__name__}."
    )


def _thaw_json(value: Any, /) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _problem_to_dict(problem: OperatorProblemSpec, /) -> dict[str, Any]:
    return {
        "source_query_relation": problem.source_query_relation,
        "query_is_fixed": problem.query_is_fixed,
        "symmetry_group": problem.symmetry_group,
        "requires_resolution_transfer": problem.requires_resolution_transfer,
        "requires_encode_once_decode_many": problem.requires_encode_once_decode_many,
        "rollout_steps": problem.rollout_steps,
    }


def _problem_from_dict(value: Mapping[str, Any], /) -> OperatorProblemSpec:
    expected = {
        "source_query_relation",
        "query_is_fixed",
        "symmetry_group",
        "requires_resolution_transfer",
        "requires_encode_once_decode_many",
        "rollout_steps",
    }
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            "Operator problem dictionary must use the current canonical fields; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}."
        )
    return OperatorProblemSpec(
        source_query_relation=value["source_query_relation"],
        query_is_fixed=value["query_is_fixed"],
        symmetry_group=value["symmetry_group"],
        requires_resolution_transfer=bool(value["requires_resolution_transfer"]),
        requires_encode_once_decode_many=bool(value["requires_encode_once_decode_many"]),
        rollout_steps=int(value["rollout_steps"]),
    )


class OperatorQuerySpec(StrictModule):
    """Resolution-independent physical contract for one named query branch."""

    name: str
    geometry_kind: OperatorGeometryKind
    coordinate_components: tuple[str, ...]
    coordinate_dimensions: tuple[DimensionSignature, ...]
    topology_site: OperatorTopologySite | None
    quadrature: OperatorQuadraturePolicy
    fixed_geometry: bool | None

    def __init__(
        self,
        name: str,
        /,
        *,
        geometry_kind: OperatorGeometryKind,
        coordinate_components: Sequence[str],
        coordinate_dimensions: Sequence[DimensionSignature] = (),
        topology_site: OperatorTopologySite | None = None,
        quadrature: OperatorQuadraturePolicy = "optional",
        fixed_geometry: bool | None = None,
    ):
        resolved_name = str(name)
        if not resolved_name:
            raise ValueError("Operator query names must not be empty.")
        components = tuple(str(value) for value in coordinate_components)
        if not components or len(set(components)) != len(components):
            raise ValueError("Query coordinate components must be non-empty and unique.")
        dimensions = tuple(coordinate_dimensions)
        if any(not isinstance(dimension, DimensionSignature) for dimension in dimensions):
            raise TypeError(
                "Query coordinate dimensions must be DimensionSignature values."
            )
        if dimensions and len(dimensions) != len(components):
            raise ValueError(
                "coordinate_dimensions must be empty or provide one signature per component."
            )
        if topology_site not in (
            None,
            "node",
            "edge",
            "face",
            "cell",
            "vertex",
            "point",
            "global",
        ):
            raise ValueError("Unknown operator topology site.")
        if (
            geometry_kind in ("graph", "simplicial", "cell_complex")
            and topology_site is None
        ):
            raise ValueError("Topological queries require a topology site.")
        if (
            geometry_kind not in ("graph", "simplicial", "cell_complex")
            and topology_site is not None
        ):
            raise ValueError("Topology sites are only valid for topological queries.")
        if quadrature not in ("unused", "optional", "physical_required"):
            raise ValueError("Unknown operator query quadrature policy.")
        self.name = resolved_name
        self.geometry_kind = geometry_kind
        self.coordinate_components = components
        self.coordinate_dimensions = dimensions
        self.topology_site = topology_site
        self.quadrature = quadrature
        self.fixed_geometry = fixed_geometry

    @property
    def coordinate_dimension(self) -> int:
        return len(self.coordinate_components)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "geometry_kind": self.geometry_kind,
            "coordinate_components": list(self.coordinate_components),
            "coordinate_dimensions": [
                value.to_dict() for value in self.coordinate_dimensions
            ],
            "topology_site": self.topology_site,
            "quadrature": self.quadrature,
            "fixed_geometry": self.fixed_geometry,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> "OperatorQuerySpec":
        expected = {
            "name",
            "geometry_kind",
            "coordinate_components",
            "coordinate_dimensions",
            "topology_site",
            "quadrature",
            "fixed_geometry",
        }
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing or unknown:
            raise ValueError(
                "Operator query dictionary must use the current canonical fields; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        raw_dimensions = value["coordinate_dimensions"]
        if any(not isinstance(item, Mapping) for item in raw_dimensions):
            raise TypeError(
                "Serialized query coordinate dimensions must be canonical mappings."
            )
        return cls(
            str(value["name"]),
            geometry_kind=value["geometry_kind"],
            coordinate_components=value["coordinate_components"],
            coordinate_dimensions=tuple(
                DimensionSignature.from_dict(item) for item in raw_dimensions
            ),
            topology_site=value["topology_site"],
            quadrature=value["quadrature"],
            fixed_geometry=value["fixed_geometry"],
        )


class OperatorTask(StrictModule):
    """Immutable physical and semantic contract for one operator-learning task."""

    task_id: str
    revision: str
    dimension_basis: tuple[str, ...]
    fields: tuple[OperatorFieldSpec, ...]
    queries: tuple[OperatorQuerySpec, ...]
    problem: OperatorProblemSpec
    pde: PDEProblemIR | None
    metadata: frozendict[str, Any]

    def __init__(
        self,
        task_id: str,
        /,
        *,
        fields: Sequence[OperatorFieldSpec],
        queries: Sequence[OperatorQuerySpec],
        problem: OperatorProblemSpec | None = None,
        pde: PDEProblemIR | None = None,
        dimension_basis: Sequence[str] = (),
        revision: str = "1",
        metadata: Mapping[str, Any] | None = None,
    ):
        resolved_id = str(task_id)
        resolved_revision = str(revision)
        if not resolved_id or not resolved_revision:
            raise ValueError("Operator tasks require non-empty task_id and revision.")
        fields_ = tuple(fields)
        queries_ = tuple(queries)
        if not fields_ or any(
            not isinstance(field, OperatorFieldSpec) for field in fields_
        ):
            raise TypeError("Operator tasks require at least one OperatorFieldSpec.")
        if not queries_ or any(
            not isinstance(query, OperatorQuerySpec) for query in queries_
        ):
            raise TypeError("Operator tasks require at least one OperatorQuerySpec.")
        field_names = tuple(field.name for field in fields_)
        query_names = tuple(query.name for query in queries_)
        if len(set(field_names)) != len(field_names):
            raise ValueError("Operator task field names must be unique.")
        if len(set(query_names)) != len(query_names):
            raise ValueError("Operator task query names must be unique.")
        basis = tuple(dimension_basis)
        if any(not isinstance(value, str) for value in basis):
            raise TypeError("dimension_basis entries must be strings.")
        if len(set(basis)) != len(basis) or any(not value for value in basis):
            raise ValueError("dimension_basis entries must be non-empty and unique.")
        query_lookup = {query.name: query for query in queries_}
        source_bindings: list[str] = []
        output_bindings: list[str] = []
        for field in fields_:
            missing_axes = {
                axis for axis, _, _ in field.dimension.terms if axis not in basis
            }
            if missing_axes:
                raise ValueError(
                    f"Field {field.name!r} dimension uses axes absent from "
                    f"dimension_basis: {sorted(missing_axes)}."
                )
            if field.is_source:
                assert field.source_name is not None
                source_bindings.append(field.source_name)
            if field.is_target:
                assert field.query_name is not None
                if field.query_name not in query_lookup:
                    raise ValueError(
                        f"Target field {field.name!r} references unknown query "
                        f"{field.query_name!r}."
                    )
                output_bindings.append(field.name)
        if len(set(source_bindings)) != len(source_bindings):
            raise ValueError("Operator task source bindings must be unique.")
        if len(set(output_bindings)) != len(output_bindings):
            raise ValueError("Operator task output bindings must be unique.")
        if not output_bindings:
            raise ValueError("Operator tasks require at least one target field.")
        for query in queries_:
            for dimension in query.coordinate_dimensions:
                missing_axes = {
                    axis for axis, _, _ in dimension.terms if axis not in basis
                }
                if missing_axes:
                    raise ValueError(
                        f"Query {query.name!r} coordinate dimension uses axes absent "
                        f"from dimension_basis: {sorted(missing_axes)}."
                    )
        problem_ = OperatorProblemSpec() if problem is None else problem
        if not isinstance(problem_, OperatorProblemSpec):
            raise TypeError("problem must be an OperatorProblemSpec.")
        if problem_.query_is_fixed is True and any(
            query.fixed_geometry is not True for query in queries_
        ):
            raise ValueError(
                "query_is_fixed=True requires fixed_geometry=True on every query."
            )
        if pde is not None:
            if not isinstance(pde, PDEProblemIR):
                raise TypeError("pde must be a PDEProblemIR.")
            self._validate_pde(fields_, queries_, pde)
            pde_dimensions = [
                *(item.dimension for item in pde.coordinates),
                *(item.dimension for item in pde.fields),
                *(item.dimension for item in pde.parameters),
            ]

            def append_expression_dimensions(expression) -> None:
                pde_dimensions.append(expression.dimension)
                for argument in expression.args:
                    append_expression_dimensions(argument)

            for equation in pde.equations:
                append_expression_dimensions(equation.lhs)
                append_expression_dimensions(equation.rhs)
            for condition in pde.conditions:
                append_expression_dimensions(condition.expression)
                append_expression_dimensions(condition.target)
            missing_axes = {
                axis
                for dimension in pde_dimensions
                for axis, _, _ in dimension.terms
                if axis not in basis
            }
            if missing_axes:
                raise ValueError(
                    "Embedded PDE dimensions use axes absent from dimension_basis: "
                    f"{sorted(missing_axes)}."
                )
        frozen_metadata = _freeze_json({} if metadata is None else metadata)
        self.task_id = resolved_id
        self.revision = resolved_revision
        self.dimension_basis = basis
        self.fields = fields_
        self.queries = queries_
        self.problem = problem_
        self.pde = pde
        self.metadata = frozen_metadata

    @staticmethod
    def _validate_pde(
        fields: tuple[OperatorFieldSpec, ...],
        queries: tuple[OperatorQuerySpec, ...],
        pde: PDEProblemIR,
        /,
    ) -> None:
        task_fields = {field.name: field for field in fields}
        query_lookup = {query.name: query for query in queries}
        representation_map = {
            "scalar": "scalar",
            "pseudoscalar": "pseudoscalar",
            "vector": "vector",
            "tensor": "tensor",
        }
        for pde_field in pde.fields:
            if pde_field.name not in task_fields:
                raise ValueError(
                    f"PDE field {pde_field.name!r} is absent from the operator task."
                )
            field = task_fields[pde_field.name]
            if field.channel_count != pde_field.components:
                raise ValueError(
                    f"PDE field {pde_field.name!r} component count disagrees with the task."
                )
            expected_representation = representation_map.get(pde_field.representation)
            if (
                expected_representation is None
                or field.representation != expected_representation
            ):
                raise ValueError(
                    f"PDE field {pde_field.name!r} representation disagrees with the task."
                )
            if field.dimension != pde_field.dimension:
                raise ValueError(
                    f"PDE field {pde_field.name!r} dimension disagrees with the task."
                )
            if field.is_target and pde_field.coordinates:
                assert field.query_name is not None
                query_coordinates = query_lookup[field.query_name].coordinate_components
                if any(name not in query_coordinates for name in pde_field.coordinates):
                    raise ValueError(
                        f"PDE field {pde_field.name!r} coordinates are absent from query "
                        f"{field.query_name!r}."
                    )
        pde_ir_to_dict(pde)

    @property
    def field_by_name(self) -> frozendict[str, OperatorFieldSpec]:
        return frozendict({field.name: field for field in self.fields})

    @property
    def query_by_name(self) -> frozendict[str, OperatorQuerySpec]:
        return frozendict({query.name: query for query in self.queries})

    @property
    def source_fields(self) -> tuple[OperatorFieldSpec, ...]:
        return tuple(field for field in self.fields if field.is_source)

    @property
    def target_fields(self) -> tuple[OperatorFieldSpec, ...]:
        return tuple(field for field in self.fields if field.is_target)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "revision": self.revision,
            "dimension_basis": list(self.dimension_basis),
            "fields": [field.to_dict() for field in self.fields],
            "queries": [query.to_dict() for query in self.queries],
            "problem": _problem_to_dict(self.problem),
            "pde": None if self.pde is None else pde_ir_to_dict(self.pde),
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> "OperatorTask":
        expected = {
            "task_id",
            "revision",
            "dimension_basis",
            "fields",
            "queries",
            "problem",
            "pde",
            "metadata",
        }
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing or unknown:
            raise ValueError(
                "Operator task dictionary must use the current canonical fields; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        pde_value = value["pde"]
        return cls(
            str(value["task_id"]),
            fields=tuple(OperatorFieldSpec.from_dict(item) for item in value["fields"]),
            queries=tuple(OperatorQuerySpec.from_dict(item) for item in value["queries"]),
            problem=_problem_from_dict(value["problem"]),
            pde=None if pde_value is None else pde_ir_from_dict(pde_value),
            dimension_basis=value["dimension_basis"],
            revision=str(value["revision"]),
            metadata=value["metadata"],
        )

    @staticmethod
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

    def validate_batch(self, batch: OperatorBatch, /) -> None:
        """Validate task semantics that are independent of model capabilities."""
        if not isinstance(batch, OperatorBatch):
            raise TypeError("OperatorTask.validate_batch requires an OperatorBatch.")
        declared_sources = {
            field.source_name
            for field in self.source_fields
            if field.source_name is not None
        }
        unknown_sources = tuple(
            name for name in batch.inputs if name not in declared_sources
        )
        if unknown_sources:
            raise ValueError(
                f"Operator batch contains sources absent from the task: {unknown_sources}."
            )
        for field in self.source_fields:
            assert field.source_name is not None
            if field.source_name not in batch.inputs:
                if field.required:
                    raise KeyError(
                        f"Operator batch is missing required source {field.source_name!r}."
                    )
                continue
            samples = batch.inputs[field.source_name]
            if samples.values is None:
                raise ValueError(f"Source {field.source_name!r} has no sampled values.")
            values = samples.values
            sample_ndim = len(samples.sample_shape)
            trailing = values.shape[len(batch.case_shape) + sample_ndim :]
            expected = () if field.channels == "scalar" else (field.channel_count,)
            if tuple(int(size) for size in trailing) != expected:
                raise ValueError(
                    f"Source {field.source_name!r} expected trailing field shape "
                    f"{expected}; got {trailing}."
                )
        expected_queries = tuple(query.name for query in self.queries)
        if set(batch.queries) != set(expected_queries):
            raise ValueError(
                "Operator batch query names must match the task; "
                f"expected {expected_queries}, got {tuple(batch.queries)}."
            )
        for query_spec in self.queries:
            samples = batch.query(query_spec.name)
            actual_kind = self._geometry_kind(samples)
            if actual_kind != query_spec.geometry_kind:
                raise ValueError(
                    f"Query {query_spec.name!r} requires geometry "
                    f"{query_spec.geometry_kind!r}; got {actual_kind!r}."
                )
            if query_spec.topology_site is not None and (
                samples.topology is None
                or samples.topology.site != query_spec.topology_site
            ):
                actual_site = None if samples.topology is None else samples.topology.site
                raise ValueError(
                    f"Query {query_spec.name!r} requires topology site "
                    f"{query_spec.topology_site!r}; got {actual_site!r}."
                )
            coordinates = samples.coordinates_array(flatten=True)
            if int(coordinates.shape[-1]) != query_spec.coordinate_dimension:
                raise ValueError(
                    f"Query {query_spec.name!r} requires coordinate dimension "
                    f"{query_spec.coordinate_dimension}; got {coordinates.shape[-1]}."
                )
            if (
                query_spec.quadrature == "physical_required"
                and not samples.has_physical_quadrature
            ):
                raise ValueError(
                    f"Query {query_spec.name!r} requires physical quadrature weights."
                )
            if query_spec.fixed_geometry is True and samples.geometry_case_shape:
                raise ValueError(
                    f"Query {query_spec.name!r} requires geometry shared by every case."
                )

    def validate_prediction(self, prediction: OperatorPrediction, /) -> None:
        """Validate named physical outputs against this task contract."""
        if not isinstance(prediction, OperatorPrediction):
            raise TypeError(
                "OperatorTask.validate_prediction requires an OperatorPrediction."
            )
        expected_queries = tuple(query.name for query in self.queries)
        if set(prediction.queries) != set(expected_queries):
            raise ValueError(
                "Prediction query names must match the task; "
                f"expected {expected_queries}, got {tuple(prediction.queries)}."
            )
        expected_fields = tuple(field.name for field in self.target_fields)
        if set(prediction.fields) != set(expected_fields):
            raise ValueError(
                "Prediction field names must match the task; "
                f"expected {expected_fields}, got {tuple(prediction.fields)}."
            )
        for specification in self.target_fields:
            output = prediction.field(specification.name)
            assert specification.query_name is not None
            assert specification.output_spec is not None
            if output.query_name != specification.query_name:
                raise ValueError(
                    f"Output field {specification.name!r} is bound to query "
                    f"{output.query_name!r}, expected {specification.query_name!r}."
                )
            if output.spec.to_dict() != specification.output_spec.to_dict():
                raise ValueError(
                    f"Output field {specification.name!r} has the wrong channel contract."
                )


__all__ = [
    "OperatorQuerySpec",
    "OperatorTask",
    "OperatorTopologySite",
]
