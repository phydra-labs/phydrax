#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class PhaseFieldMobilityEvaluation(StrictModule):
    tensor: Array
    minimum_eigenvalue: Array
    maximum_eigenvalue: Array
    symmetry_defect: Array
    finite: Array
    positive_semidefinite: Array
    successful: Array
    mobility_id: str = eqx.field(static=True)


class AbstractPhaseFieldMobility(StrictModule, NonTrainableState):
    mobility_id: eqx.AbstractVar[str]
    scalar_kinetics: eqx.AbstractVar[bool]

    @abc.abstractmethod
    def evaluate(
        self,
        previous: Array,
        points: Array,
        time: Array,
        args: object = None,
        /,
    ) -> PhaseFieldMobilityEvaluation:
        raise NotImplementedError

    def quadratic(
        self,
        force: Array,
        previous: Array,
        points: Array,
        time: Array,
        args: object = None,
        /,
    ) -> tuple[Array, PhaseFieldMobilityEvaluation]:
        evaluation = self.evaluate(previous, points, time, args)
        tensor = evaluation.tensor
        if tensor.shape[-2:] == (1, 1):
            norm = ein.contract("...d,...d->...", force, force)
            coefficient = tensor[..., 0, 0]
            coefficient = coefficient.reshape(
                coefficient.shape + (1,) * (norm.ndim - coefficient.ndim)
            )
            value = coefficient * norm
        else:
            value = ein.contract("...d,...de,...e->...", force, tensor, force)
        return value, evaluation


class ScalarPhaseFieldMobility(AbstractPhaseFieldMobility):
    value: Array
    mobility_id: str = eqx.field(static=True)
    scalar_kinetics: bool = eqx.field(static=True)

    def __init__(self, value: ArrayLike, /):
        mobility = np.asarray(value)
        if mobility.shape != () or not np.isfinite(mobility) or mobility <= 0.0:
            raise ValueError("Scalar phase-field mobility must be positive and finite.")
        self.value = jnp.asarray(mobility)
        self.scalar_kinetics = True
        self.mobility_id = canonical_fingerprint(
            {"kind": "scalar-phase-field-mobility", "value": float(mobility)}
        )

    def evaluate(
        self,
        previous: Array,
        points: Array,
        time: Array,
        args: object = None,
        /,
    ) -> PhaseFieldMobilityEvaluation:
        del previous, time, args
        shape = points.shape[:-1] + (1, 1)
        tensor = jnp.broadcast_to(self.value.astype(points.dtype), shape)
        value = self.value.astype(points.dtype)
        finite = jnp.isfinite(value)
        positive = value >= 0.0
        return PhaseFieldMobilityEvaluation(
            tensor,
            value,
            value,
            jnp.asarray(0.0, dtype=value.dtype),
            finite,
            positive,
            finite & positive,
            self.mobility_id,
        )


class TensorPhaseFieldMobility(AbstractPhaseFieldMobility):
    tensor: Array
    mobility_id: str = eqx.field(static=True)
    scalar_kinetics: bool = eqx.field(static=True)

    def __init__(self, tensor: ArrayLike, /, *, tolerance: float = 1.0e-12):
        values = np.asarray(tensor)
        tolerance_ = float(tolerance)
        if (
            values.ndim != 2
            or values.shape[0] != values.shape[1]
            or values.shape[0] == 0
            or np.any(~np.isfinite(values))
            or not np.isfinite(tolerance_)
            or tolerance_ < 0.0
        ):
            raise ValueError("Tensor mobility must be one finite square matrix.")
        symmetric = 0.5 * (values + values.T)
        scale = max(float(np.max(np.abs(symmetric))), 1.0)
        if np.max(np.abs(values - values.T)) > tolerance_ * scale:
            raise ValueError("Tensor mobility must be symmetric.")
        spectrum = np.linalg.eigvalsh(symmetric)
        if np.min(spectrum) < -tolerance_ * scale:
            raise ValueError("Tensor mobility must be positive semidefinite.")
        self.tensor = jnp.asarray(symmetric)
        self.scalar_kinetics = False
        self.mobility_id = canonical_fingerprint(
            {
                "kind": "tensor-phase-field-mobility",
                "tensor": array_tree_fingerprint(symmetric),
                "tolerance": tolerance_,
            }
        )

    def evaluate(
        self,
        previous: Array,
        points: Array,
        time: Array,
        args: object = None,
        /,
    ) -> PhaseFieldMobilityEvaluation:
        del previous, time, args
        if self.tensor.shape != (points.shape[-1], points.shape[-1]):
            raise ValueError("Tensor mobility dimension does not match physical space.")
        tensor = jnp.broadcast_to(
            self.tensor.astype(points.dtype),
            points.shape[:-1] + self.tensor.shape,
        )
        symmetric = 0.5 * (tensor + jnp.swapaxes(tensor, -1, -2))
        eigenvalues = jnp.linalg.eigvalsh(symmetric)
        symmetry = jnp.max(jnp.abs(tensor - jnp.swapaxes(tensor, -1, -2)))
        finite = jnp.all(jnp.isfinite(tensor))
        minimum = jnp.min(eigenvalues)
        maximum = jnp.max(eigenvalues)
        positive = minimum >= -64.0 * jnp.finfo(tensor.dtype).eps * jnp.maximum(
            maximum, 1.0
        )
        return PhaseFieldMobilityEvaluation(
            symmetric,
            minimum,
            maximum,
            symmetry,
            finite,
            positive,
            finite & positive,
            self.mobility_id,
        )


class CallableTensorPhaseFieldMobility(AbstractPhaseFieldMobility):
    evaluator: Callable = eqx.field(static=True)
    mobility_id: str = eqx.field(static=True)
    scalar_kinetics: bool = eqx.field(static=True)

    def __init__(self, evaluator: Callable, /, *, mobility_id: str):
        if not callable(evaluator):
            raise TypeError("Callable mobility requires an evaluator.")
        identifier = str(mobility_id)
        if not identifier:
            raise ValueError("Callable mobility requires a stable mobility_id.")
        self.evaluator = evaluator
        self.scalar_kinetics = False
        self.mobility_id = canonical_fingerprint(
            {"kind": "callable-tensor-mobility", "declared_id": identifier}
        )

    def evaluate(
        self,
        previous: Array,
        points: Array,
        time: Array,
        args: object = None,
        /,
    ) -> PhaseFieldMobilityEvaluation:
        tensor = jnp.asarray(self.evaluator(previous, points, time, args))
        expected = points.shape[:-1] + (points.shape[-1], points.shape[-1])
        if tensor.shape != expected:
            raise ValueError(
                f"Callable mobility must return shape {expected}; got {tensor.shape}."
            )
        symmetric = 0.5 * (tensor + jnp.swapaxes(tensor, -1, -2))
        eigenvalues = jnp.linalg.eigvalsh(symmetric)
        symmetry = jnp.max(jnp.abs(tensor - jnp.swapaxes(tensor, -1, -2)))
        finite = jnp.all(jnp.isfinite(tensor))
        minimum = jnp.min(eigenvalues)
        maximum = jnp.max(eigenvalues)
        positive = minimum >= -64.0 * jnp.finfo(tensor.dtype).eps * jnp.maximum(
            maximum, 1.0
        )
        return PhaseFieldMobilityEvaluation(
            symmetric,
            minimum,
            maximum,
            symmetry,
            finite,
            positive,
            finite & positive,
            self.mobility_id,
        )


def as_phase_field_mobility(
    value: AbstractPhaseFieldMobility | ArrayLike,
    /,
) -> AbstractPhaseFieldMobility:
    if isinstance(value, AbstractPhaseFieldMobility):
        return value
    array = np.asarray(value)
    return (
        ScalarPhaseFieldMobility(array)
        if array.shape == ()
        else TensorPhaseFieldMobility(array)
    )


__all__ = [
    "AbstractPhaseFieldMobility",
    "CallableTensorPhaseFieldMobility",
    "PhaseFieldMobilityEvaluation",
    "ScalarPhaseFieldMobility",
    "TensorPhaseFieldMobility",
    "as_phase_field_mobility",
]
