#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import inspect
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Key

import phydrax.ein as ein

from ...._frozendict import frozendict
from ...._training_objective import _ObjectiveContribution
from ....graph import (
    broadcast_operator_topology,
    cochain_metric_reduce,
    CochainMetricReduction,
    CochainResidualProgram,
)
from ..data import (
    OperatorBatch,
    OperatorPrediction,
    OperatorTargetBatch,
)
from ..metrics import operator_l2_loss
from ..task import OperatorTask
from ..topology import scatter_operator_graph_entities


@dataclass(frozen=True)
class OperatorLossContext:
    """Paired execution-space and physical-space views for one loss evaluation."""

    execution_prediction: OperatorPrediction
    execution_batch: OperatorBatch
    execution_targets: OperatorTargetBatch
    physical_prediction: OperatorPrediction
    physical_batch: OperatorBatch
    physical_targets: OperatorTargetBatch
    normalization: Any = None
    case_log_weights: Array | None = None
    case_mask: Array | None = None
    sampling_probabilities: Array | None = None
    task: Any = None
    target_execution_prediction: OperatorPrediction | None = None
    target_physical_prediction: OperatorPrediction | None = None

    def view(
        self,
        space: Literal["execution", "physical"],
        /,
    ) -> tuple[OperatorPrediction, OperatorBatch, OperatorTargetBatch]:
        if space == "execution":
            return (
                self.execution_prediction,
                self.execution_batch,
                self.execution_targets,
            )
        if space == "physical":
            return (
                self.physical_prediction,
                self.physical_batch,
                self.physical_targets,
            )
        raise ValueError("Loss space must be 'execution' or 'physical'.")


@dataclass(frozen=True)
class CochainResidualInput:
    """Bind one residual-program input to a prediction or source field."""

    kind: Literal["prediction", "source"]
    field: str

    def __post_init__(self):
        if self.kind not in ("prediction", "source"):
            raise ValueError(
                "Cochain residual input kind must be 'prediction' or 'source'."
            )
        if not self.field:
            raise ValueError("Cochain residual input field must be non-empty.")


class AbstractOperatorLossTerm(ABC):
    """One named scalar objective evaluated against a rich operator batch."""

    name: str
    weight: float

    @abstractmethod
    def __call__(
        self,
        model: Any,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        targets: OperatorTargetBatch,
        /,
        *,
        key: Key[Array, ""],
        step: Array,
        training: bool,
        context: OperatorLossContext,
    ) -> Array:
        raise NotImplementedError

    def contribution(
        self,
        model: Any,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        targets: OperatorTargetBatch,
        /,
        *,
        key: Key[Array, ""],
        step: Array,
        training: bool,
        context: OperatorLossContext,
    ) -> _ObjectiveContribution:
        """Return the additive case-supported form of this scalar term."""
        support_count = 1
        for size in batch.case_shape:
            support_count *= int(size)
        value = self(
            model,
            prediction,
            batch,
            targets,
            key=key,
            step=step,
            training=training,
            context=context,
        )
        support = jnp.asarray(support_count, dtype=jnp.asarray(value).real.dtype)
        return _ObjectiveContribution(jnp.asarray(value) * support, support)

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """Stable identity included in exact-resume compatibility checks."""
        raise NotImplementedError


@dataclass(frozen=True)
class CochainResidualLoss(AbstractOperatorLossTerm):
    """Topology-aware physics loss evaluated by a shared cochain residual program."""

    name: str
    program: CochainResidualProgram
    inputs: Mapping[str, CochainResidualInput]
    output: str
    weight: float = 1.0
    reduction: CochainMetricReduction = "graph_mean"
    topology_fingerprint: str | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Cochain residual loss names must be non-empty.")
        if not isinstance(self.program, CochainResidualProgram):
            raise TypeError("program must be a CochainResidualProgram.")
        normalized: dict[str, CochainResidualInput] = {}
        for name, binding in self.inputs.items():
            if not isinstance(binding, CochainResidualInput):
                raise TypeError(
                    f"Cochain residual input {name!r} must be a CochainResidualInput."
                )
            normalized[str(name)] = binding
        if frozenset(normalized) != frozenset(self.program.input_specs):
            raise ValueError(
                "CochainResidualLoss inputs must exactly match the program input schema."
            )
        object.__setattr__(self, "inputs", frozendict(normalized))
        if self.output not in self.program.output_specs:
            raise KeyError(f"Unknown cochain residual output {self.output!r}.")
        if self.reduction not in ("graph_mean", "metric_mean", "metric_sum"):
            raise ValueError(
                "reduction must be 'graph_mean', 'metric_mean', or 'metric_sum'."
            )
        if not jnp.isfinite(self.weight):
            raise ValueError("Cochain residual loss weight must be finite.")
        if self.topology_fingerprint is not None and not self.topology_fingerprint:
            raise ValueError("topology_fingerprint must be non-empty when provided.")

    def _samples_and_values(
        self,
        binding: CochainResidualInput,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        task: OperatorTask | None,
        expected: Any,
        /,
    ) -> tuple[Any, Array]:
        actual_name = binding.field
        task_field = None
        if task is not None:
            if binding.field not in task.field_by_name:
                raise KeyError(f"Unknown task field {binding.field!r}.")
            task_field = task.field_by_name[binding.field]
            if task_field.cochain != expected:
                raise ValueError(
                    f"Task field {binding.field!r} cochain semantics do not match "
                    "the residual program."
                )

        if binding.kind == "prediction":
            if task_field is not None and not task_field.is_target:
                raise ValueError(
                    f"Task field {binding.field!r} is not a predicted target field."
                )
            predicted = prediction.field(actual_name)
            return batch.query(predicted.query_name), predicted.values

        if task_field is not None:
            if not task_field.is_source:
                raise ValueError(f"Task field {binding.field!r} is not a source field.")
            assert task_field.source_name is not None
            actual_name = task_field.source_name
        samples = batch.input(actual_name)
        if samples.values is None:
            raise ValueError(f"Source field {actual_name!r} has no sampled values.")
        return samples, jnp.asarray(samples.values)

    def __call__(
        self,
        model: Any,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        targets: OperatorTargetBatch,
        /,
        *,
        key: Key[Array, ""],
        step: Array,
        training: bool,
        context: OperatorLossContext,
    ) -> Array:
        del model, prediction, batch, targets, step, training
        physical_prediction, physical_batch, _ = context.view("physical")
        task = context.task
        if task is not None and not isinstance(task, OperatorTask):
            raise TypeError("Operator loss context task must be an OperatorTask.")

        reference_samples = None
        reference_fingerprint = None
        full_fields: dict[str, Array] = {}
        coverage_by_input: dict[str, Array] = {}
        for name, expected in self.program.input_specs.items():
            samples, values = self._samples_and_values(
                self.inputs[name],
                physical_prediction,
                physical_batch,
                task,
                expected,
            )
            topology = samples.topology
            if (
                topology is None
                or topology.kind != "cell_complex"
                or topology.entity != "node"
            ):
                raise ValueError(
                    f"Cochain residual input {name!r} requires node-based "
                    "cell-complex topology."
                )
            fingerprint = topology.graph_fingerprint
            if reference_fingerprint is None:
                reference_fingerprint = fingerprint
                reference_samples = samples
            elif fingerprint != reference_fingerprint:
                raise ValueError(
                    "Cochain residual inputs do not share one canonical topology."
                )
            if (
                self.topology_fingerprint is not None
                and fingerprint != self.topology_fingerprint
            ):
                raise ValueError(
                    "Cochain residual topology does not match its declared fingerprint."
                )
            full_fields[name] = scatter_operator_graph_entities(
                samples,
                values,
                case_shape=physical_batch.case_shape,
            )
            coverage_by_input[name] = scatter_operator_graph_entities(
                samples,
                jnp.ones(
                    physical_batch.case_shape + samples.sample_shape,
                    dtype=bool,
                ),
                case_shape=physical_batch.case_shape,
            )

        if reference_samples is None:
            raise ValueError("Cochain residual loss has no bound input samples.")
        assert reference_samples.topology is not None
        topology = broadcast_operator_topology(
            reference_samples.topology,
            physical_batch.case_shape,
        )
        graph = topology.graph
        if not isinstance(graph.nodes, Mapping):
            raise ValueError("Cochain residual topology requires named node metadata.")
        cell_degree = jnp.asarray(graph.nodes["cell_dim"], dtype=jnp.int32)
        valid_nodes = (
            jnp.ones(cell_degree.shape, dtype=bool)
            if graph.node_mask is None
            else jnp.asarray(graph.node_mask, dtype=bool)
        )
        for name, expected in self.program.input_specs.items():
            required = valid_nodes & (cell_degree == expected.degree)
            full_fields[name] = eqx.error_if(
                full_fields[name],
                jnp.any(required & ~jnp.asarray(coverage_by_input[name], dtype=bool)),
                f"Cochain residual input {name!r} does not cover every degree-"
                f"{expected.degree} cell.",
            )

        residual = self.program(graph, full_fields, key=key)[self.output]
        squared = jnp.real(jnp.conj(residual) * residual)
        if squared.ndim > 1:
            squared = jnp.sum(squared, axis=tuple(range(1, squared.ndim)))
        output_spec = self.program.output_specs[self.output]
        active = valid_nodes & (cell_degree == output_spec.degree)
        if "hodge_star" not in graph.nodes:
            raise ValueError("Cochain residual topology requires Hodge-star metadata.")
        metric = jnp.asarray(graph.nodes["hodge_star"], dtype=squared.dtype)
        positions = jnp.arange(cell_degree.shape[0], dtype=jnp.int32)
        ends = jnp.cumsum(jnp.asarray(graph.n_node, dtype=jnp.int32))
        graph_index = jnp.searchsorted(ends, positions, side="right").astype(jnp.int32)
        graph_index = jnp.where(positions < jnp.sum(graph.n_node), graph_index, -1)
        value = cochain_metric_reduce(
            squared,
            metric,
            graph_index,
            n_graph=int(graph.n_node.shape[0]),
            reduction=self.reduction,
            entity_mask=active,
        )
        return jnp.asarray(self.weight, dtype=value.dtype) * value

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "kind": "cochain_residual",
                "name": self.name,
                "weight": self.weight,
                "program": self.program.fingerprint,
                "inputs": {
                    name: {
                        "kind": binding.kind,
                        "field": binding.field,
                    }
                    for name, binding in sorted(self.inputs.items())
                },
                "output": self.output,
                "reduction": self.reduction,
                "topology_fingerprint": self.topology_fingerprint,
                "space": "physical",
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OperatorLossTerm(AbstractOperatorLossTerm):
    """Adapt a custom scalar or explicit per-case callable.

    Canonical dataset case weights make scalar callbacks ambiguous. Weighted
    training therefore requires ``case_reduction="per_case"`` and a ``(case,)``
    result; scalar callbacks fail closed rather than silently dropping weights.
    """

    name: str
    fn: Callable[..., Array]
    weight: float = 1.0
    identity: str | None = None
    space: Literal["execution", "physical"] = "physical"
    case_reduction: Literal["scalar", "per_case"] = "scalar"

    def __post_init__(self):
        if not self.name:
            raise ValueError("Operator loss term names must be non-empty.")
        if not callable(self.fn):
            raise TypeError("Operator loss term fn must be callable.")
        if not jnp.isfinite(self.weight):
            raise ValueError("Operator loss term weight must be finite.")
        if self.space not in ("execution", "physical"):
            raise ValueError("Loss space must be 'execution' or 'physical'.")
        if self.case_reduction not in ("scalar", "per_case"):
            raise ValueError("case_reduction must be 'scalar' or 'per_case'.")

    def __call__(
        self,
        model: Any,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        targets: OperatorTargetBatch,
        /,
        *,
        key: Key[Array, ""],
        step: Array,
        training: bool,
        context: OperatorLossContext,
    ) -> Array:
        del prediction, batch, targets
        selected_prediction, selected_batch, selected_targets = context.view(self.space)
        value = self.fn(
            selected_prediction,
            selected_batch,
            selected_targets,
            model=model,
            key=key,
            step=step,
            training=training,
            context=context,
        )
        array = jnp.asarray(value)
        weighted_cases = context.case_log_weights is not None
        if self.case_reduction == "per_case":
            expected = (selected_batch.case_shape[0],)
            if array.shape != expected:
                raise ValueError(
                    "Custom per-case operator losses must return shape "
                    f"{expected}; got {array.shape}."
                )
            array = _weighted_case_reduction(array, context, "mean")
        elif weighted_cases:
            raise ValueError(
                "A scalar custom OperatorLossTerm cannot consume nonuniform/masked "
                "case weights. Declare case_reduction='per_case' and return one "
                "value per case."
            )
        elif array.ndim != 0:
            raise ValueError(
                "Custom scalar operator losses must return a scalar unless "
                "case_reduction='per_case'."
            )
        return jnp.asarray(self.weight, dtype=array.dtype) * array

    @property
    def fingerprint(self) -> str:
        identity = self.identity
        if identity is None:
            if inspect.isfunction(self.fn) or inspect.ismethod(self.fn):
                identity = f"{self.fn.__module__}.{self.fn.__qualname__}"
            else:
                function_type = type(self.fn)
                identity = f"{function_type.__module__}.{function_type.__qualname__}"
        payload = json.dumps(
            {
                "kind": "custom",
                "name": self.name,
                "weight": self.weight,
                "identity": identity,
                "space": self.space,
                "case_reduction": self.case_reduction,
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SupervisedOperatorLoss(AbstractOperatorLossTerm):
    """Named supervised L² objective in physical or execution space."""

    name: str = "supervised_l2"
    weight: float = 1.0
    prediction_field: str | None = None
    target_field: str | None = None
    relative: bool = False
    squared: bool = True
    reduction: Literal["none", "mean", "sum"] = "mean"
    epsilon: float = 1e-12
    space: Literal["execution", "physical"] = "physical"

    def __post_init__(self):
        if not self.name:
            raise ValueError("Operator loss term names must be non-empty.")
        if not jnp.isfinite(self.weight):
            raise ValueError("Operator loss term weight must be finite.")
        if self.reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
        if self.reduction == "none":
            raise ValueError("Training loss terms must reduce to a scalar.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if self.space not in ("execution", "physical"):
            raise ValueError("Loss space must be 'execution' or 'physical'.")

    def __call__(
        self,
        model: Any,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        targets: OperatorTargetBatch,
        /,
        *,
        key: Key[Array, ""],
        step: Array,
        training: bool,
        context: OperatorLossContext,
    ) -> Array:
        del model, prediction, batch, targets, key, step, training
        selected_prediction, selected_batch, selected_targets = context.view(self.space)
        prediction_name = self.prediction_field
        if prediction_name is None:
            if len(selected_prediction.fields) != 1:
                raise ValueError(
                    "prediction_field is required for multi-output predictions."
                )
            prediction_name = next(iter(selected_prediction.fields))
        target_name = self.target_field
        if target_name is None:
            if prediction_name in selected_targets.fields:
                target_name = prediction_name
            elif len(selected_targets.fields) == 1:
                target_name = next(iter(selected_targets.fields))
            else:
                raise ValueError("target_field is required for multi-target batches.")
        predicted = selected_prediction.field(prediction_name)
        truth = selected_targets.field(target_name)
        if predicted.query_name != truth.query_name:
            raise ValueError(
                f"Prediction {prediction_name!r} and target {target_name!r} "
                "must use the same query."
            )
        query = selected_batch.query(predicted.query_name)
        mask = query.mask_array(case_shape=selected_batch.case_shape)
        trailing = (1,) * (predicted.values.ndim - mask.ndim)
        expanded_mask = mask.reshape(mask.shape + trailing)
        predicted_values = jnp.where(expanded_mask, predicted.values, 0.0)
        target_values = jnp.where(expanded_mask, truth.values, 0.0)
        case_values = operator_l2_loss(
            predicted_values,
            target_values,
            query,
            relative=self.relative,
            squared=self.squared,
            reduction="none",
            eps=self.epsilon,
        )
        value = _weighted_case_reduction(case_values, context, self.reduction)
        return jnp.asarray(self.weight, dtype=jnp.asarray(value).dtype) * value

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "kind": "supervised_l2",
                "name": self.name,
                "weight": self.weight,
                "prediction_field": self.prediction_field,
                "target_field": self.target_field,
                "relative": self.relative,
                "squared": self.squared,
                "reduction": self.reduction,
                "epsilon": self.epsilon,
                "space": self.space,
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SupervisedOperatorRolloutLoss(AbstractOperatorLossTerm):
    """Ordered future-state aliases for one supervised recurrent objective."""

    target_fields: tuple[str, ...]
    time_weights: tuple[float, ...]
    name: str = "supervised_rollout_l2"
    weight: float = 1.0
    reduction: Literal["mean"] = "mean"

    def __post_init__(self):
        fields = tuple(str(field) for field in self.target_fields)
        weights = tuple(float(value) for value in self.time_weights)
        if not self.name:
            raise ValueError("Operator rollout loss names must be non-empty.")
        if not fields or any(not field for field in fields):
            raise ValueError("Supervised rollout target aliases must be non-empty.")
        if len(set(fields)) != len(fields):
            raise ValueError("Supervised rollout target aliases must be unique.")
        if len(weights) != len(fields):
            raise ValueError("time_weights must provide one value per target alias.")
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("Rollout time weights must be finite and nonnegative.")
        if not math.isfinite(float(self.weight)) or float(self.weight) < 0.0:
            raise ValueError(
                "Operator rollout loss weight must be finite and nonnegative."
            )
        if self.reduction != "mean":
            raise ValueError("Rollout reduction must be 'mean'.")
        object.__setattr__(self, "target_fields", fields)
        object.__setattr__(self, "time_weights", weights)
        object.__setattr__(self, "weight", float(self.weight))

    def __call__(
        self,
        model: Any,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        targets: OperatorTargetBatch,
        /,
        *,
        key: Key[Array, ""],
        step: Array,
        training: bool,
        context: OperatorLossContext,
    ) -> Array:
        del model, prediction, batch, targets, key, step, training, context
        raise ValueError(
            "SupervisedOperatorRolloutLoss must be evaluated by task-bound "
            "fit_operator with a rollout route and policy."
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "kind": "supervised_operator_rollout",
                "name": self.name,
                "target_fields": self.target_fields,
                "time_weights": self.time_weights,
                "weight": self.weight,
                "reduction": self.reduction,
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResidualOperatorRolloutLoss(AbstractOperatorLossTerm):
    """Recurrently evaluate one existing target-independent operator residual."""

    residual_term: AbstractOperatorLossTerm
    time_weights: tuple[float, ...]
    name: str = "residual_rollout"
    weight: float = 1.0
    reduction: Literal["mean"] = "mean"

    def __post_init__(self):
        weights = tuple(float(value) for value in self.time_weights)
        if not self.name:
            raise ValueError("Operator rollout loss names must be non-empty.")
        if not isinstance(self.residual_term, AbstractOperatorLossTerm):
            raise TypeError("residual_term must be an AbstractOperatorLossTerm.")
        if isinstance(
            self.residual_term,
            (
                SupervisedOperatorLoss,
                SupervisedOperatorRolloutLoss,
                ResidualOperatorRolloutLoss,
            ),
        ):
            raise TypeError("residual_term must be target-independent and non-recurrent.")
        if not weights:
            raise ValueError("Residual rollout time_weights must be non-empty.")
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("Rollout time weights must be finite and nonnegative.")
        if not math.isfinite(float(self.weight)) or float(self.weight) < 0.0:
            raise ValueError(
                "Operator rollout loss weight must be finite and nonnegative."
            )
        if self.reduction != "mean":
            raise ValueError("Rollout reduction must be 'mean'.")
        object.__setattr__(self, "time_weights", weights)
        object.__setattr__(self, "weight", float(self.weight))

    def __call__(
        self,
        model: Any,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        targets: OperatorTargetBatch,
        /,
        *,
        key: Key[Array, ""],
        step: Array,
        training: bool,
        context: OperatorLossContext,
    ) -> Array:
        del model, prediction, batch, targets, key, step, training, context
        raise ValueError(
            "ResidualOperatorRolloutLoss must be evaluated by task-bound "
            "fit_operator with a rollout route and policy."
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "kind": "residual_operator_rollout",
                "name": self.name,
                "residual_term": self.residual_term.fingerprint,
                "time_weights": self.time_weights,
                "weight": self.weight,
                "reduction": self.reduction,
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _weighted_case_reduction(
    values: Array,
    context: OperatorLossContext,
    reduction: Literal["mean", "sum", "none"],
    /,
) -> Array:
    if context.case_log_weights is None:
        return jnp.sum(values) if reduction == "sum" else jnp.mean(values)
    log_weights = jnp.asarray(context.case_log_weights, dtype=values.dtype)
    mask = (
        jnp.ones(log_weights.shape, dtype=bool)
        if context.case_mask is None
        else jnp.asarray(context.case_mask, dtype=bool)
    )
    probabilities = (
        jnp.ones(log_weights.shape, dtype=values.dtype)
        if context.sampling_probabilities is None
        else jnp.asarray(context.sampling_probabilities, dtype=values.dtype)
    )
    if (
        log_weights.shape != values.shape
        or mask.shape != values.shape
        or probabilities.shape != values.shape
    ):
        raise ValueError("Case weights, mask, and sampling probabilities must align.")
    active = mask & jnp.isfinite(log_weights) & (probabilities > 0.0)
    maximum = jnp.max(jnp.where(active, log_weights, -jnp.inf))
    weights = jnp.where(
        active,
        jnp.exp(log_weights - maximum) / probabilities,
        0.0,
    )
    mass = jnp.sum(weights)
    weights = eqx.error_if(
        weights,
        ~jnp.isfinite(mass) | (mass <= 0.0),
        "Weighted operator reduction requires positive finite case mass.",
    )
    normalized = weights / mass
    return ein.contract("i,i->", normalized, values.reshape((-1,)))


__all__ = [
    "AbstractOperatorLossTerm",
    "CochainResidualInput",
    "CochainResidualLoss",
    "OperatorLossContext",
    "OperatorLossTerm",
    "ResidualOperatorRolloutLoss",
    "SupervisedOperatorLoss",
    "SupervisedOperatorRolloutLoss",
]
