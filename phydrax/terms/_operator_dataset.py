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

from phydrax.domain import ConcatenatedModelEvaluator, Domain, DomainFunction

from .._doc import DOC_KEY0
from .._term import AbstractScalarTerm
from ..nn._loss import ModelWithLoss
from ..nn.operator.adapters import OperatorContextModel
from ..nn.operator.data import OperatorBatch
from ..nn.operator.metrics import (
    operator_h1_loss,
    operator_l2_loss,
    operator_spectral_loss,
)
from ..nn.operator.protocols import OperatorModel


OperatorLoss = Literal["l2", "h1", "spectral"]


def _operator_callable(function: DomainFunction, /) -> Callable:
    if not isinstance(function.func, ConcatenatedModelEvaluator):
        raise TypeError(
            "Operator constraints require a DomainFunction created with Domain.Model(...)."
        )
    model = function.func.raw_model
    if isinstance(model, ModelWithLoss):
        if not isinstance(model.model, OperatorModel):
            raise TypeError("Wrapped model is not a PhydraX neural operator.")
        return model
    if not isinstance(model, OperatorModel):
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


class OperatorDatasetTerm(AbstractScalarTerm):
    """Supervised neural-operator data loss over one or more discretizations."""

    batches: tuple[OperatorBatch, ...]
    targets: tuple[Array, ...]
    fields: tuple[str, ...]
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
        self.fields = (str(function),)
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
        iter_: int | Array | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        function_name = self.fields[0]
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
        iter_: int | Array | None = None,
        **kwargs: Any,
    ) -> dict[str, Array]:
        del iter_, kwargs
        model = _operator_callable(functions[self.fields[0]])
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


class PhysicsInformedOperatorTerm(AbstractScalarTerm):
    """PINO-style residual loss, optionally evaluated on different resolutions."""

    batches: tuple[OperatorBatch, ...]
    residual_fn: Callable[[Array, OperatorBatch], Array]
    fields: tuple[str, ...]
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
        self.fields = (str(function),)
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
        iter_: int | Array | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        function_name = self.fields[0]
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


class DifferentialPhysicsInformedOperatorTerm(AbstractScalarTerm):
    """PINO residual loss composed from native PhydraX function operators."""

    batches: tuple[OperatorBatch, ...]
    domain: Domain
    coordinate_label: str
    residual_operator: Callable[[DomainFunction], DomainFunction]
    fields: tuple[str, ...]
    loss_kind: OperatorLoss
    weight: Array
    label: str | None

    def __init__(
        self,
        function: str,
        batches: OperatorBatch | Sequence[OperatorBatch],
        domain: Domain,
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
        self.fields = (str(function),)
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
        iter_: int | Array | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        function_name = self.fields[0]
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
                batch.case_shape
                + batch.require_single_query().sample_shape
                + trailing_shape
            )
            total = total + _metric(
                self.loss_kind,
                values,
                jnp.zeros_like(values),
                batch,
                relative=False,
            )
        return self.weight * total / float(len(self.batches))


def operator_term_suite(
    *terms: (
        OperatorDatasetTerm
        | PhysicsInformedOperatorTerm
        | DifferentialPhysicsInformedOperatorTerm
    ),
) -> tuple[
    OperatorDatasetTerm
    | PhysicsInformedOperatorTerm
    | DifferentialPhysicsInformedOperatorTerm,
    ...,
]:
    """Validate and group data and physics terms for one operator function."""
    if not terms:
        raise ValueError("operator_term_suite requires at least one term.")
    function_name = terms[0].fields
    if any(term.fields != function_name for term in terms):
        raise ValueError("All operator terms in a suite must target one function.")
    return tuple(terms)


__all__ = [
    "DifferentialPhysicsInformedOperatorTerm",
    "OperatorDatasetTerm",
    "PhysicsInformedOperatorTerm",
    "operator_term_suite",
]
