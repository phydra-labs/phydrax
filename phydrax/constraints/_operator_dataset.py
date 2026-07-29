#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from .._doc import DOC_KEY0
from ..domain._domain import _AbstractDomain
from ..domain._function import DomainFunction
from ..domain._model_function import _ConcatenatedModelCallable
from ..nn.models.core._base import _AbstractOperatorModel
from ..nn.models.core._loss import ModelWithLoss
from ..nn.models.core._operator import OperatorBatch
from ..nn.models.core._operator_metrics import (
    operator_h1_loss,
    operator_l2_loss,
    operator_spectral_loss,
)
from ..nn.models.wrappers._operator_context import OperatorContextModel
from ._base import AbstractConstraint


OperatorLoss = Literal["l2", "h1", "spectral"]


def _operator_callable(function: DomainFunction, /) -> Callable:
    if not isinstance(function.func, _ConcatenatedModelCallable):
        raise TypeError(
            "Operator constraints require a DomainFunction created with Domain.Model(...)."
        )
    model = function.func.raw_model
    if isinstance(model, ModelWithLoss):
        if not isinstance(model.model, _AbstractOperatorModel):
            raise TypeError("Wrapped model is not a PhydraX neural operator.")
        return model
    if not isinstance(model, _AbstractOperatorModel):
        raise TypeError("DomainFunction model is not a PhydraX neural operator.")
    return model


def _metric(
    kind: OperatorLoss,
    prediction: Array,
    target: Array,
    batch: OperatorBatch,
    /,
    *,
    relative: bool,
) -> Array:
    if kind == "l2":
        return operator_l2_loss(
            prediction,
            target,
            batch.require_single_query(),
            relative=relative,
        )
    if kind == "h1":
        return operator_h1_loss(
            prediction,
            target,
            batch.require_single_query(),
            relative=relative,
        )
    if kind == "spectral":
        return operator_spectral_loss(
            prediction,
            target,
            batch.require_single_query(),
            relative=relative,
        )
    raise ValueError("loss must be 'l2', 'h1', or 'spectral'.")


def _batches_tuple(
    batches: OperatorBatch | Sequence[OperatorBatch],
    /,
) -> tuple[OperatorBatch, ...]:
    if isinstance(batches, OperatorBatch):
        return (batches,)
    result = tuple(batches)
    if not result or any(not isinstance(batch, OperatorBatch) for batch in result):
        raise TypeError("batches must contain at least one OperatorBatch.")
    return result


class OperatorDatasetConstraint(AbstractConstraint):
    """Supervised neural-operator data loss over one or more discretizations."""

    batches: tuple[OperatorBatch, ...]
    targets: tuple[Array, ...]
    constraint_vars: tuple[str, ...]
    loss_kind: OperatorLoss
    relative: bool
    weight: Array
    label: str | None

    def __init__(
        self,
        function: str,
        batches: OperatorBatch | Sequence[OperatorBatch],
        targets: Any,
        /,
        *,
        loss: OperatorLoss = "l2",
        relative: bool = True,
        weight: Any = 1.0,
        label: str | None = None,
    ):
        self.batches = _batches_tuple(batches)
        if len(self.batches) == 1:
            target_values = (jnp.asarray(targets),)
        else:
            if not isinstance(targets, Sequence):
                raise TypeError("Multiple operator batches require a target sequence.")
            target_values = tuple(jnp.asarray(target) for target in targets)
        if len(target_values) != len(self.batches):
            raise ValueError("targets and batches must have the same length.")
        self.targets = target_values
        self.constraint_vars = (str(function),)
        self.loss_kind = loss
        self.relative = bool(relative)
        self.weight = jnp.asarray(weight, dtype=float)
        self.label = None if label is None else str(label)
        if loss not in ("l2", "h1", "spectral"):
            raise ValueError("loss must be 'l2', 'h1', or 'spectral'.")

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        function_name = self.constraint_vars[0]
        if function_name not in functions:
            raise KeyError(f"Missing operator function {function_name!r}.")
        model = _operator_callable(functions[function_name])
        keys = jr.split(key, len(self.batches))
        total = jnp.asarray(0.0, dtype=float)
        for batch, target, batch_key in zip(
            self.batches, self.targets, keys, strict=True
        ):
            prediction = model(batch, key=batch_key)
            total = total + _metric(
                self.loss_kind,
                prediction,
                target,
                batch,
                relative=self.relative,
            )
        return self.weight * total / float(len(self.batches))

    def data_metrics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Array]:
        del iter_, kwargs
        model = _operator_callable(functions[self.constraint_vars[0]])
        keys = jr.split(key, len(self.batches))
        absolute = jnp.asarray(0.0, dtype=float)
        relative = jnp.asarray(0.0, dtype=float)
        for batch, target, batch_key in zip(
            self.batches, self.targets, keys, strict=True
        ):
            prediction = model(batch, key=batch_key)
            absolute = absolute + operator_l2_loss(
                prediction, target, batch.require_single_query(), relative=False
            )
            relative = relative + operator_l2_loss(
                prediction, target, batch.require_single_query(), relative=True
            )
        count = float(len(self.batches))
        return {
            "operator_l2": absolute / count,
            "operator_relative_l2": relative / count,
        }


class PhysicsInformedOperatorConstraint(AbstractConstraint):
    """PINO-style residual loss, optionally evaluated on different resolutions."""

    batches: tuple[OperatorBatch, ...]
    residual_fn: Callable[[Array, OperatorBatch], Array]
    constraint_vars: tuple[str, ...]
    loss_kind: OperatorLoss
    weight: Array
    label: str | None

    def __init__(
        self,
        function: str,
        batches: OperatorBatch | Sequence[OperatorBatch],
        residual_fn: Callable[[Array, OperatorBatch], Array],
        /,
        *,
        loss: OperatorLoss = "l2",
        weight: Any = 1.0,
        label: str | None = None,
    ):
        self.batches = _batches_tuple(batches)
        if not callable(residual_fn):
            raise TypeError("residual_fn must be callable.")
        self.residual_fn = residual_fn
        self.constraint_vars = (str(function),)
        self.loss_kind = loss
        self.weight = jnp.asarray(weight, dtype=float)
        self.label = None if label is None else str(label)
        if loss not in ("l2", "h1", "spectral"):
            raise ValueError("loss must be 'l2', 'h1', or 'spectral'.")

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        function_name = self.constraint_vars[0]
        if function_name not in functions:
            raise KeyError(f"Missing operator function {function_name!r}.")
        model = _operator_callable(functions[function_name])
        keys = jr.split(key, len(self.batches))
        total = jnp.asarray(0.0, dtype=float)
        for batch, batch_key in zip(self.batches, keys, strict=True):
            prediction = jnp.asarray(model(batch, key=batch_key))
            residual = jnp.asarray(self.residual_fn(prediction, batch))
            total = total + _metric(
                self.loss_kind,
                residual,
                jnp.zeros_like(residual),
                batch,
                relative=False,
            )
        return self.weight * total / float(len(self.batches))


class DifferentialPhysicsInformedOperatorConstraint(AbstractConstraint):
    """PINO residual loss composed from native PhydraX function operators."""

    batches: tuple[OperatorBatch, ...]
    domain: _AbstractDomain
    coordinate_label: str
    residual_operator: Callable[[DomainFunction], DomainFunction]
    constraint_vars: tuple[str, ...]
    loss_kind: OperatorLoss
    weight: Array
    label: str | None

    def __init__(
        self,
        function: str,
        batches: OperatorBatch | Sequence[OperatorBatch],
        domain: _AbstractDomain,
        coordinate_label: str,
        residual_operator: Callable[[DomainFunction], DomainFunction],
        /,
        *,
        loss: OperatorLoss = "l2",
        weight: Any = 1.0,
        label: str | None = None,
    ):
        self.batches = _batches_tuple(batches)
        self.domain = domain
        self.coordinate_label = str(coordinate_label)
        if self.coordinate_label not in domain.labels:
            raise KeyError(f"Domain has no coordinate label {self.coordinate_label!r}.")
        if not callable(residual_operator):
            raise TypeError("residual_operator must be callable.")
        self.residual_operator = residual_operator
        self.constraint_vars = (str(function),)
        self.loss_kind = loss
        self.weight = jnp.asarray(weight, dtype=float)
        self.label = None if label is None else str(label)
        if loss not in ("l2", "h1", "spectral"):
            raise ValueError("loss must be 'l2', 'h1', or 'spectral'.")
        for batch in self.batches:
            if batch.require_single_query().geometry_case_shape:
                raise ValueError(
                    "Native differential PINO currently requires shared query geometry."
                )

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        function_name = self.constraint_vars[0]
        if function_name not in functions:
            raise KeyError(f"Missing operator function {function_name!r}.")
        model = _operator_callable(functions[function_name])
        keys = jr.split(key, len(self.batches))
        total = jnp.asarray(0.0, dtype=float)
        for batch, batch_key in zip(self.batches, keys, strict=True):
            context = OperatorContextModel(model, batch)
            prediction = context.domain_function(
                self.domain,
                self.coordinate_label,
            )
            residual = self.residual_operator(prediction)
            if not isinstance(residual, DomainFunction):
                raise TypeError("residual_operator must return a DomainFunction.")
            coordinates = batch.require_single_query().coordinates_array(flatten=True)
            flat_residual = jax.vmap(lambda point: residual.func(point, key=batch_key))(
                coordinates
            )
            values = jnp.moveaxis(flat_residual, 0, len(batch.case_shape))
            trailing_shape = tuple(
                int(size) for size in values.shape[len(batch.case_shape) + 1 :]
            )
            values = values.reshape(
                batch.case_shape + batch.require_single_query().sample_shape + trailing_shape
            )
            total = total + _metric(
                self.loss_kind,
                values,
                jnp.zeros_like(values),
                batch,
                relative=False,
            )
        return self.weight * total / float(len(self.batches))


def operator_constraint_suite(
    *constraints: (
        OperatorDatasetConstraint
        | PhysicsInformedOperatorConstraint
        | DifferentialPhysicsInformedOperatorConstraint
    ),
) -> tuple[
    OperatorDatasetConstraint
    | PhysicsInformedOperatorConstraint
    | DifferentialPhysicsInformedOperatorConstraint,
    ...,
]:
    """Validate and group data and physics terms for one operator function."""
    if not constraints:
        raise ValueError("operator_constraint_suite requires at least one constraint.")
    function_name = constraints[0].constraint_vars
    if any(constraint.constraint_vars != function_name for constraint in constraints):
        raise ValueError("All operator constraints in a suite must target one function.")
    return tuple(constraints)


__all__ = [
    "DifferentialPhysicsInformedOperatorConstraint",
    "OperatorDatasetConstraint",
    "PhysicsInformedOperatorConstraint",
    "operator_constraint_suite",
]
