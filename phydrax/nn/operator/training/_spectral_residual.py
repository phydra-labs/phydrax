#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from ...._frozendict import frozendict
from ....equations._spectral_residual import CompiledSpectralResidual
from ..data import OperatorBatch, OperatorPrediction, OperatorTargetBatch
from ._losses import (
    _weighted_case_reduction,
    AbstractOperatorLossTerm,
    OperatorAccumulationKind,
    OperatorLossContext,
)


def _parameter_names(expression) -> frozenset[str]:
    names = (
        frozenset((expression.symbol,))
        if expression.op == "parameter" and expression.symbol is not None
        else frozenset()
    )
    for argument in expression.args:
        names = names | _parameter_names(argument)
    return names


@dataclass(frozen=True)
class SpectralPDEResidualLoss(AbstractOperatorLossTerm):
    """Targetless all-coordinate PDE loss evaluated through a spectral compiler."""

    name: str
    compiled: CompiledSpectralResidual
    prediction_fields: Mapping[str, str]
    parameter_inputs: Mapping[str, str]
    weight: float = 1.0
    query_name: str | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Spectral PDE residual loss names must be non-empty.")
        if not isinstance(self.compiled, CompiledSpectralResidual):
            raise TypeError("compiled must be a CompiledSpectralResidual.")
        predictions = frozendict(
            {str(name): str(field) for name, field in self.prediction_fields.items()}
        )
        if frozenset(predictions) != frozenset(self.compiled.layout.field_names):
            raise ValueError("prediction_fields must exactly cover compiled PDE fields.")
        parameters = frozendict(
            {str(name): str(source) for name, source in self.parameter_inputs.items()}
        )
        known_parameters = frozenset(self.compiled.evaluator.parameter_names)
        if not frozenset(parameters).issubset(known_parameters):
            raise ValueError("parameter_inputs contains an unknown PDE parameter.")
        functional = {
            name
            for name, is_functional in zip(
                self.compiled.evaluator.parameter_names,
                self.compiled.evaluator.parameter_functional,
                strict=True,
            )
            if is_functional
        }
        if not frozenset(parameters).issubset(functional):
            raise ValueError("parameter_inputs may bind only functional PDE parameters.")
        used_parameters = frozenset().union(
            *(
                _parameter_names(expression)
                for expression in self.compiled.evaluator.rhs_expressions
            )
        )
        unresolved_functional = {
            name
            for name, is_functional, default in zip(
                self.compiled.evaluator.parameter_names,
                self.compiled.evaluator.parameter_functional,
                self.compiled.evaluator.parameter_defaults,
                strict=True,
            )
            if is_functional and default is None and name in used_parameters
        }
        if not unresolved_functional.issubset(parameters):
            raise ValueError(
                "parameter_inputs must bind every unresolved functional PDE parameter."
            )
        unresolved_scalar = {
            name
            for name, is_functional, default in zip(
                self.compiled.evaluator.parameter_names,
                self.compiled.evaluator.parameter_functional,
                self.compiled.evaluator.parameter_defaults,
                strict=True,
            )
            if not is_functional and default is None and name in used_parameters
        }
        if unresolved_scalar:
            raise ValueError(
                "Scalar PDE parameters must be fixed during spectral residual compilation."
            )
        if not math.isfinite(float(self.weight)) or float(self.weight) < 0.0:
            raise ValueError(
                "Spectral PDE residual loss weight must be finite and non-negative."
            )
        query = None if self.query_name is None else str(self.query_name)
        if query == "":
            raise ValueError("query_name must be non-empty or None.")
        object.__setattr__(self, "prediction_fields", predictions)
        object.__setattr__(self, "parameter_inputs", parameters)
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "query_name", query)

    def _query(self, prediction: OperatorPrediction, batch: OperatorBatch):
        predicted_queries = {
            prediction.field(field).query_name
            for field in self.prediction_fields.values()
        }
        if self.query_name is not None:
            predicted_queries.add(self.query_name)
        if len(predicted_queries) != 1:
            raise ValueError(
                "Spectral PDE prediction fields must share one canonical query."
            )
        query_name = next(iter(predicted_queries))
        query = batch.query(query_name)
        if not query.axes:
            raise ValueError("Spectral PDE residual loss requires tensor-product axes.")
        if query.geometry_case_shape not in ((), batch.case_shape):
            raise ValueError(
                "Spectral PDE residual loss requires one shared tensor grid."
            )
        # Loaders may materialize all-valid masks; dynamic validation happens in loss.
        expected_shape = self.compiled.discretization.physical_shape
        if query.sample_shape != expected_shape:
            raise ValueError(
                f"Spectral query shape must be {expected_shape}; got {query.sample_shape}."
            )
        expected_names = self.compiled.discretization.plan.axis_names
        if query.axis_names != expected_names:
            raise ValueError(
                f"Spectral query axes must be {expected_names}; got {query.axis_names}."
            )
        expected_periodicity = tuple(
            axis.periodic for axis in self.compiled.discretization.axes
        )
        if tuple(axis.periodic for axis in query.axes) != expected_periodicity:
            raise ValueError(
                "Spectral query periodicity does not match the compiled trial space."
            )
        return query

    def _node_mismatch(self, axes) -> Array:
        mismatch = jnp.asarray(False)
        for source, target in zip(
            axes,
            self.compiled.discretization.axes,
            strict=True,
        ):
            mismatch = mismatch | ~jnp.allclose(
                source.nodes,
                target.nodes,
                rtol=0.0,
                atol=1e-12,
            )
        return mismatch

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
        physical_prediction, physical_batch, _ = context.view("physical")
        query = self._query(physical_prediction, physical_batch)
        case_shape = physical_batch.case_shape
        case_count = prod(case_shape) if case_shape else 1
        flat_fields = {}
        for pde_name, prediction_name in self.prediction_fields.items():
            value = jnp.asarray(physical_prediction.field(prediction_name).values)
            expected = case_shape + self.compiled.layout.field_shape(
                pde_name,
                physical=True,
            )
            if value.shape != expected:
                raise ValueError(
                    f"Prediction field {prediction_name!r} must have shape {expected}; "
                    f"got {value.shape}."
                )
            flat_fields[pde_name] = value.reshape(
                (case_count,)
                + self.compiled.layout.field_shape(
                    pde_name,
                    physical=True,
                )
            )
        states = jax.vmap(self.compiled.project_state)(flat_fields)
        states = eqx.error_if(
            states,
            jnp.any(~query.mask_array(case_shape=case_shape))
            | self._node_mismatch(query.axes),
            "Spectral PDE residual loss requires the complete compiled query grid.",
        )

        flat_parameters = {}
        for parameter_name, input_name in self.parameter_inputs.items():
            samples = physical_batch.input(input_name)
            if samples.values is None:
                raise ValueError(
                    f"Functional parameter input {input_name!r} has no values."
                )
            if (
                not samples.axes
                or samples.geometry_case_shape not in ((), physical_batch.case_shape)
                or samples.axis_names != query.axis_names
                or samples.sample_shape != query.sample_shape
            ):
                raise ValueError(
                    "Functional spectral parameters must use the complete trial grid."
                )
            value = jnp.asarray(samples.values)
            parameter_index = self.compiled.evaluator.parameter_names.index(
                parameter_name
            )
            components = self.compiled.evaluator.parameter_components[parameter_index]
            sample_shape = self.compiled.discretization.physical_shape + (
                () if components == 1 else (components,)
            )
            expected = case_shape + sample_shape
            if value.shape != expected:
                raise ValueError(
                    f"Functional parameter {parameter_name!r} must have shape "
                    f"{expected}; got {value.shape}."
                )
            value = eqx.error_if(
                value,
                jnp.any(~samples.mask_array(case_shape=case_shape))
                | self._node_mismatch(samples.axes),
                "Spectral PDE residual loss requires complete compiled parameter grids.",
            )
            flat_parameters[parameter_name] = value.reshape((case_count,) + sample_shape)

        if flat_parameters:
            energies = jax.vmap(self.compiled.residual_energy)(states, flat_parameters)
        else:
            energies = jax.vmap(lambda state: self.compiled.residual_energy(state, None))(
                states
            )
        value = _weighted_case_reduction(energies, context, "mean")
        return jnp.asarray(self.weight, dtype=value.dtype) * value

    @property
    def accumulation_kind(self) -> OperatorAccumulationKind:
        return "case_mean"

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "kind": "spectral_pde_residual",
                "name": self.name,
                "compiled": self.compiled.compilation_id,
                "prediction_fields": dict(sorted(self.prediction_fields.items())),
                "parameter_inputs": dict(sorted(self.parameter_inputs.items())),
                "weight": self.weight,
                "query_name": self.query_name,
                "space": "physical",
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["SpectralPDEResidualLoss"]
