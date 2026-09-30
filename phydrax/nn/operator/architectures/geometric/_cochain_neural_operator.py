#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Rigid metric-aware neural operators on typed cochain fields."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import sqrt
from typing import Any, assert_never, ClassVar, final, Literal, TypeAlias

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from phydrax._doc import DOC_KEY0
from phydrax._strict import StrictModule
from phydrax.exterior._complex import ComplexBoundary
from phydrax.exterior._form_type import FormType
from phydrax.graph._cochain_execution import _cochain_metric_valid
from phydrax.graph._cochain_ops import (
    cochain_codifferential,
    cochain_exterior_derivative,
    cochain_harmonic_projection,
    cochain_hodge_laplacian,
)
from phydrax.graph._ir import GraphIR
from phydrax.nn._keys import EvalKey
from phydrax.nn._utils import _get_size
from phydrax.nn.operator.capabilities import ConfiguredOperatorContract
from phydrax.nn.operator.data import (
    OperatorBatch,
    OperatorFieldBatch,
    OperatorPrediction,
)
from phydrax.nn.operator.engine import AbstractOperatorModel
from phydrax.nn.operator.field import OperatorFieldSpec
from phydrax.nn.operator.topology import (
    gather_operator_graph_entities,
    materialize_operator_fields,
)

from .....typing import parse, PRNGKey


_TopologicalRoute: TypeAlias = Literal[
    "self",
    "exterior_derivative",
    "codifferential",
    "lower_laplacian",
    "upper_laplacian",
    "harmonic",
]

_ROUTE_ORDER: tuple[_TopologicalRoute, ...] = (
    "self",
    "exterior_derivative",
    "codifferential",
    "lower_laplacian",
    "upper_laplacian",
    "harmonic",
)


def _named_key(key: PRNGKey, label: str, /) -> PRNGKey:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return jr.fold_in(key, int.from_bytes(digest[:4], "little"))


def _channel_matrix(key: PRNGKey, in_channels: int, out_channels: int, /) -> Array:
    scale = 1.0 / jnp.sqrt(float(max(1, in_channels)))
    return scale * jr.normal(key, (int(in_channels), int(out_channels)))


def _node_degree_mask(
    graph: GraphIR,
    degree: int,
    boundary: ComplexBoundary,
    /,
) -> Array:
    if not isinstance(graph.nodes, Mapping):
        raise ValueError("Cochain operators require named graph-node metadata.")
    mask = jnp.asarray(graph.nodes["cell_dim"]) == int(degree)
    if boundary == "relative":
        mask = mask & ~jnp.asarray(graph.nodes["boundary"], dtype=jnp.bool_)
    if graph.node_mask is not None:
        mask = mask & graph.node_mask
    return mask


@final
class TopologicalRouteConfig(StrictModule):
    """Static admissible routes for a rigid cochain operator block."""

    self_route: bool = eqx.field(static=True)
    exterior_derivative: bool = eqx.field(static=True)
    codifferential: bool = eqx.field(static=True)
    lower_laplacian: bool = eqx.field(static=True)
    upper_laplacian: bool = eqx.field(static=True)
    harmonic: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        self_route: bool = True,
        exterior_derivative: bool = True,
        codifferential: bool = True,
        lower_laplacian: bool = True,
        upper_laplacian: bool = True,
        harmonic: bool = False,
    ) -> None:
        values = (
            bool(self_route),
            bool(exterior_derivative),
            bool(codifferential),
            bool(lower_laplacian),
            bool(upper_laplacian),
            bool(harmonic),
        )
        if not any(values):
            raise ValueError("TopologicalRouteConfig must enable at least one route.")
        (
            self.self_route,
            self.exterior_derivative,
            self.codifferential,
            self.lower_laplacian,
            self.upper_laplacian,
            self.harmonic,
        ) = values

    @property
    def enabled_routes(self) -> tuple[_TopologicalRoute, ...]:
        flags = (
            self.self_route,
            self.exterior_derivative,
            self.codifferential,
            self.lower_laplacian,
            self.upper_laplacian,
            self.harmonic,
        )
        return tuple(
            name for name, enabled in zip(_ROUTE_ORDER, flags, strict=True) if enabled
        )


@final
class TopologicalCochainBlock(StrictModule):
    """One orientation-equivariant block with fixed DEC information routes.

    The topology selects admissible routes. Learned work is restricted to
    cell-wise channel mixing, degree modulation, and an odd residual update.
    """

    route_weights: tuple[Array, ...]
    degree_embeddings: Array
    residual_scales: Array
    route_config: TopologicalRouteConfig = eqx.field(static=True)
    route_names: tuple[_TopologicalRoute, ...] = eqx.field(static=True)
    active_degrees: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    boundary: ComplexBoundary = eqx.field(static=True)
    width: int = eqx.field(static=True)
    norm_epsilon: float = eqx.field(static=True)

    def __init__(
        self,
        width: int,
        active_degrees: Sequence[int],
        /,
        *,
        dimension: int,
        routes: TopologicalRouteConfig | None = None,
        boundary: ComplexBoundary = "absolute",
        norm_epsilon: float = 1e-6,
        residual_scale: float = 0.25,
        key: PRNGKey = DOC_KEY0,
    ) -> None:
        resolved_width = int(width)
        degrees = tuple(sorted(set(active_degrees)))
        if resolved_width <= 0:
            raise ValueError("Topological cochain width must be positive.")
        if not degrees:
            raise ValueError("active_degrees must contain at least one degree.")
        for degree in degrees:
            FormType(dimension, degree)
        if float(norm_epsilon) <= 0.0:
            raise ValueError("norm_epsilon must be positive.")
        route_config = TopologicalRouteConfig() if routes is None else routes
        if not isinstance(route_config, TopologicalRouteConfig):
            raise TypeError("routes must be a TopologicalRouteConfig.")
        policy = parse(boundary, ComplexBoundary, "boundary")
        route_names = route_config.enabled_routes
        self.route_weights = tuple(
            jnp.stack(
                tuple(
                    _channel_matrix(
                        _named_key(key, f"route:{route}:degree:{degree}"),
                        resolved_width,
                        resolved_width,
                    )
                    for degree in degrees
                )
            )
            for route in route_names
        )
        self.degree_embeddings = 0.05 * jr.normal(
            _named_key(key, "degree_embeddings"),
            (len(degrees), resolved_width),
        )
        self.residual_scales = jnp.full(
            (len(degrees),), float(residual_scale), dtype=jnp.float64
        )
        self.route_config = route_config
        self.route_names = route_names
        self.active_degrees = degrees
        self.dimension = int(dimension)
        self.boundary = policy
        self.width = resolved_width
        self.norm_epsilon = float(norm_epsilon)

    def _route_is_admissible(self, name: _TopologicalRoute, degree: int, /) -> bool:
        match name:
            case "self" | "harmonic":
                return True
            case "exterior_derivative" | "lower_laplacian":
                return degree > 0
            case "codifferential" | "upper_laplacian":
                return degree < self.dimension
            case _:
                assert_never(name)

    def _route(
        self,
        name: _TopologicalRoute,
        graph: GraphIR,
        hidden: Array,
        degree: int,
        /,
    ) -> Array:
        name = parse(name, _TopologicalRoute, "route")
        if not self._route_is_admissible(name, degree):
            raise ValueError(
                f"Topological route {name!r} is absent at degree {degree} "
                f"in dimension {self.dimension}."
            )
        if name == "self":
            return hidden
        if name == "exterior_derivative":
            return cochain_exterior_derivative(
                graph,
                hidden,
                degree - 1,
                boundary=self.boundary,
            )
        if name == "codifferential":
            return cochain_codifferential(
                graph,
                hidden,
                degree + 1,
                boundary=self.boundary,
            )
        if name == "lower_laplacian":
            return cochain_hodge_laplacian(
                graph,
                hidden,
                degree,
                part="lower",
                boundary=self.boundary,
            )
        if name == "upper_laplacian":
            return cochain_hodge_laplacian(
                graph,
                hidden,
                degree,
                part="upper",
                boundary=self.boundary,
            )
        if name == "harmonic":
            return cochain_harmonic_projection(
                graph,
                hidden,
                degree,
                boundary=self.boundary,
            )
        assert_never(name)

    def __call__(self, graph: GraphIR, hidden: Any, /) -> Array:
        values = jnp.asarray(hidden)
        if values.ndim != 2 or values.shape[1] != self.width:
            raise ValueError(
                f"Topological hidden values must have shape (cells, {self.width})."
            )
        if (
            not isinstance(graph.nodes, Mapping)
            or values.shape[0] != jnp.asarray(graph.nodes["cell_dim"]).shape[0]
        ):
            raise ValueError("Topological hidden values must align with graph nodes.")
        output = jnp.zeros_like(values)
        for degree_index, degree in enumerate(self.active_degrees):
            degree_mask = _node_degree_mask(graph, degree, self.boundary)[:, None]
            mixed = jnp.zeros_like(values)
            for route_index, route_name in enumerate(self.route_names):
                if not self._route_is_admissible(route_name, degree):
                    continue
                routed = self._route(route_name, graph, values, degree)
                mixed = mixed + routed @ self.route_weights[route_index][degree_index]
            rms = jnp.sqrt(
                jnp.mean(jnp.square(mixed), axis=-1, keepdims=True) + self.norm_epsilon
            )
            degree_gate = 1.0 + 0.1 * jnp.tanh(self.degree_embeddings[degree_index])
            update = (
                self.residual_scales[degree_index] * degree_gate * jnn.tanh(mixed / rms)
            )
            output = output + jnp.where(degree_mask, values + update, 0)
        return eqx.error_if(
            output,
            ~_cochain_metric_valid(graph.cochain_bindings),
            "Native cochain pairing must remain finite and positive definite.",
        )


def _predict_cochain_operator(
    model: Any,
    batch: OperatorBatch,
    key: EvalKey,
    /,
) -> OperatorPrediction:
    values = model.predict_fields(batch, key=key)
    fields = {}
    for name in model.target_names:
        field = model._field(name)
        if field.query_name is None or field.output_spec is None:
            raise ValueError(
                "Cochain prediction targets require query_name and output_spec."
            )
        fields[name] = OperatorFieldBatch(
            values[name],
            query_name=field.query_name,
            spec=field.output_spec,
        )
    return OperatorPrediction(
        fields,
        batch.queries,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


def _cochain_operator_contract(
    model: CochainNeuralOperator,
) -> ConfiguredOperatorContract:
    from phydrax.nn.operator.catalog import operator_architecture_contract

    return replace(
        operator_architecture_contract(model.operator_architecture),
        field_specs=model.fields,
    )


@final
class CochainNeuralOperator(AbstractOperatorModel):
    """Named multi-field neural operator over a shared metric cochain complex.

    All inter-cell communication is an exact sparse DEC route. Trainable maps
    act only on channels at individual cells; native differentials, full Riesz
    maps, boundary policy, and harmonic evidence remain prepared runtime data.
    """

    operator_architecture = "CochainNeuralOperator"
    _operator_contract_builder = staticmethod(_cochain_operator_contract)

    _operator_prediction_builder: ClassVar = staticmethod(_predict_cochain_operator)

    fields: tuple[OperatorFieldSpec, ...]
    source_encoders: tuple[Array, ...]
    blocks: tuple[TopologicalCochainBlock, ...]
    target_decoders: tuple[Array, ...]
    source_names: tuple[str, ...] = eqx.field(static=True)
    target_names: tuple[str, ...] = eqx.field(static=True)
    active_degrees: tuple[int, ...] = eqx.field(static=True)
    default_target: str = eqx.field(static=True)
    boundary: ComplexBoundary = eqx.field(static=True)
    routes: TopologicalRouteConfig = eqx.field(static=True)
    width: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    in_size: tuple[int, ...] = eqx.field(static=True)
    out_size: int | Literal["scalar"] = eqx.field(static=True)

    def __init__(
        self,
        fields: Sequence[OperatorFieldSpec],
        /,
        *,
        active_degrees: Sequence[int] | None = None,
        width: int = 64,
        depth: int = 4,
        routes: TopologicalRouteConfig | None = None,
        boundary: ComplexBoundary = "absolute",
        default_target: str | None = None,
        norm_epsilon: float = 1e-6,
        key: PRNGKey = DOC_KEY0,
    ) -> None:
        specs = tuple(fields)
        if not specs or any(not isinstance(field, OperatorFieldSpec) for field in specs):
            raise TypeError("CochainNeuralOperator requires OperatorFieldSpec fields.")
        if len({field.name for field in specs}) != len(specs):
            raise ValueError("Cochain operator field names must be unique.")
        if any(
            field.form_type is None or field.representation != "cochain"
            for field in specs
        ):
            raise ValueError(
                "Every cochain operator field requires form_type and cochain representation."
            )
        dimensions = {
            (field.form_type.dimension, field.form_type.ambient_dimension)
            for field in specs
            if field.form_type is not None
        }
        if len(dimensions) != 1:
            raise ValueError("Cochain operator fields must share one form dimension.")
        sources = tuple(field.name for field in specs if field.is_source)
        targets = tuple(field.name for field in specs if field.is_target)
        if not sources or not targets:
            raise ValueError("CochainNeuralOperator requires source and target fields.")
        for field in specs:
            if field.is_target:
                if field.output_spec is None:
                    raise ValueError("Target fields require an output_spec.")
                if _get_size(field.output_spec.channels) != field.channel_count:
                    raise ValueError(
                        "Target output channels must match the cochain field channels."
                    )
        inferred_degrees = tuple(
            sorted(
                {field.form_type.degree for field in specs if field.form_type is not None}
            )
        )
        degrees = (
            inferred_degrees
            if active_degrees is None
            else tuple(sorted(set(active_degrees)))
        )
        if not degrees or not set(inferred_degrees).issubset(degrees):
            raise ValueError("active_degrees must include every configured field degree.")
        dimension = next(iter(dimensions))[0]
        for degree in degrees:
            FormType(dimension, degree)
        resolved_width = int(width)
        resolved_depth = int(depth)
        if resolved_width <= 0 or resolved_depth <= 0:
            raise ValueError("Cochain operator width and depth must be positive.")
        route_config = TopologicalRouteConfig() if routes is None else routes
        if not isinstance(route_config, TopologicalRouteConfig):
            raise TypeError("routes must be a TopologicalRouteConfig.")
        policy = parse(boundary, ComplexBoundary, "boundary")
        chosen_target = targets[0] if default_target is None else str(default_target)
        if chosen_target not in targets:
            raise ValueError("default_target must name a configured target field.")

        self.fields = specs
        self.source_names = sources
        self.target_names = targets
        self.active_degrees = degrees
        self.default_target = chosen_target
        self.boundary = policy
        self.routes = route_config
        self.width = resolved_width
        self.depth = resolved_depth
        self.in_size = tuple(self._field(name).channel_count for name in sources)
        default_spec = self._field(chosen_target).output_spec
        if default_spec is None:
            raise ValueError("The default target requires an output_spec.")
        self.out_size = default_spec.channels
        self.source_encoders = tuple(
            _channel_matrix(
                _named_key(key, f"source:{name}"),
                self._field(name).channel_count,
                resolved_width,
            )
            for name in sources
        )
        self.blocks = tuple(
            TopologicalCochainBlock(
                resolved_width,
                degrees,
                dimension=dimension,
                routes=route_config,
                boundary=policy,
                norm_epsilon=norm_epsilon,
                residual_scale=1.0 / sqrt(float(resolved_depth)),
                key=_named_key(key, f"block:{index}"),
            )
            for index in range(resolved_depth)
        )
        self.target_decoders = tuple(
            _channel_matrix(
                _named_key(key, f"target:{name}"),
                resolved_width,
                self._field(name).channel_count,
            )
            for name in targets
        )

    def _field(self, name: str, /) -> OperatorFieldSpec:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(f"Unknown cochain field {name!r}.")

    @property
    def operator_output_specs(self) -> dict[str, Any]:
        return {name: self._field(name).output_spec for name in self.target_names}

    def _encode(self, graph: GraphIR, /) -> Array:
        if not isinstance(graph.nodes, Mapping):
            raise ValueError("Cochain topology graph nodes must be a mapping.")
        node_count = jnp.asarray(graph.nodes["cell_dim"]).shape[0]
        hidden = jnp.zeros((node_count, self.width), dtype=jnp.float64)
        for name, encoder in zip(self.source_names, self.source_encoders, strict=True):
            field = self._field(name)
            if field.form_type is None:
                raise ValueError("Cochain source fields require a form_type.")
            values = jnp.asarray(graph.nodes[f"field:{name}"])
            if values.ndim == 1:
                values = values[:, None]
            degree_mask = _node_degree_mask(graph, field.form_type.degree, self.boundary)[
                :, None
            ]
            hidden = hidden + jnp.where(degree_mask, values @ encoder, 0)
        return hidden

    def predict_fields(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> dict[str, Array]:
        del key
        if not isinstance(batch, OperatorBatch):
            raise TypeError("CochainNeuralOperator requires an OperatorBatch.")
        from phydrax.graph._cochain_residual import _admit_form_type

        graph = materialize_operator_fields(batch, self.fields)
        for field in self.fields:
            if field.form_type is None:
                raise ValueError("Cochain fields require a form_type.")
            graph = _admit_form_type(graph, field.form_type)
        if self.routes.harmonic and (
            not isinstance(graph.nodes, Mapping) or "harmonic_basis" not in graph.nodes
        ):
            raise ValueError(
                "The harmonic route requires a precomputed HarmonicSubspace on the topology."
            )
        metric_valid = _cochain_metric_valid(graph.cochain_bindings)
        hidden = self._encode(graph)
        for block in self.blocks:
            hidden = block(graph, hidden)
        outputs: dict[str, Array] = {}
        for name, decoder in zip(self.target_names, self.target_decoders, strict=True):
            field = self._field(name)
            if field.form_type is None or field.query_name is None:
                raise ValueError("Cochain targets require form_type and query_name.")
            query = batch.query(field.query_name)
            degree_mask = _node_degree_mask(graph, field.form_type.degree, self.boundary)[
                :, None
            ]
            node_values = jnp.where(degree_mask, hidden @ decoder, 0)
            gathered = jnp.asarray(
                gather_operator_graph_entities(
                    query,
                    node_values,
                    case_shape=batch.case_shape,
                )
            )
            if field.channels == "scalar":
                gathered = gathered[..., 0]
            outputs[name] = eqx.error_if(
                gathered,
                ~metric_valid,
                "Native cochain pairing must remain finite and positive definite.",
            )
        return outputs

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        return self.predict_fields(batch, key=key)[self.default_target]

    def __call__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        if not isinstance(batch, OperatorBatch):
            raise TypeError("CochainNeuralOperator requires an OperatorBatch.")
        return self.__call_operator_batch__(batch, key=key)


__all__ = [
    "CochainNeuralOperator",
    "TopologicalCochainBlock",
    "TopologicalRouteConfig",
]
