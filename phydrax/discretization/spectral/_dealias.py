#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import ceil, prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._spectral._fourier import resize_fourier_axis
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._space import TensorSpectralDiscretization, TensorSpectralPlan


DealiasingKind: TypeAlias = Literal["none", "padding", "closure", "filter"]


class DealiasingReport(StrictModule, NonTrainableState):
    """Static exactness, retained shape, and evaluation-shape evidence."""

    kind: DealiasingKind = eqx.field(static=True)
    retained_shape: tuple[int, ...] = eqx.field(static=True)
    evaluation_shape: tuple[int, ...] = eqx.field(static=True)
    maximum_polynomial_degree: int | None = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        kind: DealiasingKind,
        retained_shape: tuple[int, ...],
        evaluation_shape: tuple[int, ...],
        maximum_polynomial_degree: int | None,
        exact: bool,
    ):
        retained = tuple(int(value) for value in retained_shape)
        evaluation = tuple(int(value) for value in evaluation_shape)
        if (
            not retained
            or len(retained) != len(evaluation)
            or any(value <= 0 for value in retained + evaluation)
            or any(
                target < source
                for source, target in zip(retained, evaluation, strict=True)
            )
        ):
            raise ValueError("Dealiasing retained/evaluation shapes are incompatible.")
        degree = (
            None if maximum_polynomial_degree is None else int(maximum_polynomial_degree)
        )
        if degree is not None and degree < 1:
            raise ValueError("maximum_polynomial_degree must be positive or None.")
        self.kind = kind
        self.retained_shape = retained
        self.evaluation_shape = evaluation
        self.maximum_polynomial_degree = degree
        self.exact = bool(exact)
        self.report_id = canonical_fingerprint(
            {
                "kind": "spectral-dealiasing-report",
                "strategy": kind,
                "retained_shape": list(retained),
                "evaluation_shape": list(evaluation),
                "maximum_polynomial_degree": degree,
                "exact": bool(exact),
            }
        )


class AbstractDealiasingPlan(StrictModule, NonTrainableState):
    """Symbolic nonlinear spectral evaluation policy."""

    kind: DealiasingKind = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def prepare(
        self,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        required_polynomial_degree: int | None,
    ) -> "PreparedDealiasingPlan":
        raise NotImplementedError


class NoDealiasingPlan(AbstractDealiasingPlan):
    """Explicitly accept unresolved nonlinear aliases."""

    def __init__(self):
        self.kind = "none"
        self.plan_id = canonical_fingerprint({"kind": "no-spectral-dealiasing"})

    def prepare(
        self,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        required_polynomial_degree: int | None,
    ) -> "PreparedDealiasingPlan":
        degree = required_polynomial_degree
        return PreparedDealiasingPlan(
            self,
            discretization,
            discretization,
            report=DealiasingReport(
                kind="none",
                retained_shape=discretization.modal_shape,
                evaluation_shape=discretization.modal_shape,
                maximum_polynomial_degree=degree,
                exact=degree is not None and degree <= 1,
            ),
        )


class PaddingDealiasingPlan(AbstractDealiasingPlan):
    """Overresolve polynomial products and project to retained modes."""

    maximum_polynomial_degree: int = eqx.field(static=True)

    def __init__(self, maximum_polynomial_degree: int = 2):
        degree = int(maximum_polynomial_degree)
        if degree < 2:
            raise ValueError(
                "Padding dealiasing requires polynomial degree at least two."
            )
        self.maximum_polynomial_degree = degree
        self.kind = "padding"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "padding-spectral-dealiasing",
                "maximum_polynomial_degree": degree,
            }
        )

    def prepare(
        self,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        required_polynomial_degree: int | None,
    ) -> "PreparedDealiasingPlan":
        if required_polynomial_degree is None:
            raise ValueError(
                "Finite padding cannot certify a nonpolynomial spectral expression. "
                "Select an explicit approximate filtering policy."
            )
        required = int(required_polynomial_degree)
        if required > self.maximum_polynomial_degree:
            raise ValueError(
                "Compiled polynomial degree exceeds the padding dealiasing contract."
            )
        factor = 0.5 * float(self.maximum_polynomial_degree + 1)
        target = tuple(
            max(size, int(ceil(factor * size))) for size in discretization.modal_shape
        )
        basis_plans = tuple(
            axis.plan.resized(count)
            for axis, count in zip(discretization.axes, target, strict=True)
        )
        padded_plan = TensorSpectralPlan(
            basis_plans,
            axis_names=discretization.plan.axis_names,
            field_name=discretization.plan.field_name,
            precision=discretization.plan.precision,
        )
        bounds = jnp.stack(tuple(axis.bounds for axis in discretization.axes), axis=1)
        padded = padded_plan.prepare(
            bounds, numeric_version=discretization.numeric_version
        )
        exact = all(axis.family != "sine" for axis in discretization.axes)
        return PreparedDealiasingPlan(
            self,
            discretization,
            padded,
            report=DealiasingReport(
                kind="padding",
                retained_shape=discretization.modal_shape,
                evaluation_shape=target,
                maximum_polynomial_degree=self.maximum_polynomial_degree,
                exact=exact,
            ),
        )


class PolynomialClosureDealiasingPlan(AbstractDealiasingPlan):
    """Represent the complete finite polynomial closure before residual reduction."""

    maximum_polynomial_degree: int = eqx.field(static=True)
    maximum_evaluation_modes: int = eqx.field(static=True)

    def __init__(
        self,
        maximum_polynomial_degree: int = 2,
        /,
        *,
        maximum_evaluation_modes: int = 16_777_216,
    ):
        degree = int(maximum_polynomial_degree)
        maximum = int(maximum_evaluation_modes)
        if degree < 2:
            raise ValueError(
                "Polynomial closure requires polynomial degree at least two."
            )
        if maximum <= 0:
            raise ValueError("maximum_evaluation_modes must be positive.")
        self.maximum_polynomial_degree = degree
        self.maximum_evaluation_modes = maximum
        self.kind = "closure"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polynomial-closure-spectral-dealiasing",
                "maximum_polynomial_degree": degree,
                "maximum_evaluation_modes": maximum,
            }
        )

    def prepare(
        self,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        required_polynomial_degree: int | None,
    ) -> "PreparedDealiasingPlan":
        if required_polynomial_degree is None:
            raise ValueError(
                "Finite polynomial closure cannot certify a nonpolynomial "
                "spectral expression."
            )
        required = int(required_polynomial_degree)
        if required > self.maximum_polynomial_degree:
            raise ValueError(
                "Compiled polynomial degree exceeds the closure dealiasing contract."
            )
        if any(axis.family == "sine" for axis in discretization.axes):
            raise ValueError(
                "A sine basis is not closed under polynomial multiplication."
            )
        from ._constraints import ConstrainedBasisPlan

        if any(
            isinstance(axis.plan, ConstrainedBasisPlan) for axis in discretization.axes
        ):
            raise ValueError(
                "Polynomial closure requires nested unconstrained modal coordinates."
            )
        target = tuple(
            max(size, required * (size - 1) + 1) for size in discretization.modal_shape
        )
        if prod(target) > self.maximum_evaluation_modes:
            raise ValueError(
                "Polynomial closure exceeds maximum_evaluation_modes before "
                "spectral preparation."
            )
        basis_plans = tuple(
            axis.plan.resized(count)
            for axis, count in zip(discretization.axes, target, strict=True)
        )
        closure_plan = TensorSpectralPlan(
            basis_plans,
            axis_names=discretization.plan.axis_names,
            field_name=discretization.plan.field_name,
            precision=discretization.plan.precision,
        )
        bounds = jnp.stack(tuple(axis.bounds for axis in discretization.axes), axis=1)
        closure = closure_plan.prepare(
            bounds,
            numeric_version=discretization.numeric_version,
        )
        return PreparedDealiasingPlan(
            self,
            discretization,
            closure,
            report=DealiasingReport(
                kind="closure",
                retained_shape=discretization.modal_shape,
                evaluation_shape=target,
                maximum_polynomial_degree=required,
                exact=True,
            ),
        )


class ModalFilterPlan(AbstractDealiasingPlan):
    """Approximate axis-separable modal cutoff for general nonlinearities."""

    cutoff_fraction: float = eqx.field(static=True)

    def __init__(self, cutoff_fraction: float = 2.0 / 3.0):
        fraction = float(cutoff_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("cutoff_fraction must lie in (0, 1].")
        self.cutoff_fraction = fraction
        self.kind = "filter"
        self.plan_id = canonical_fingerprint(
            {"kind": "filtered-spectral-dealiasing", "cutoff_fraction": fraction}
        )

    def prepare(
        self,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        required_polynomial_degree: int | None,
    ) -> "PreparedDealiasingPlan":
        masks = []
        for axis in discretization.axes:
            numbers = jnp.abs(axis.modes.mode_numbers)
            maximum = jnp.max(numbers)
            masks.append(numbers <= self.cutoff_fraction * maximum)
        return PreparedDealiasingPlan(
            self,
            discretization,
            discretization,
            modal_masks=tuple(masks),
            report=DealiasingReport(
                kind="filter",
                retained_shape=discretization.modal_shape,
                evaluation_shape=discretization.modal_shape,
                maximum_polynomial_degree=required_polynomial_degree,
                exact=False,
            ),
        )


def _resize_nonfourier_axis(
    coefficients: Array,
    axis: int,
    target_size: int,
    /,
) -> Array:
    source_size = int(coefficients.shape[axis])
    if source_size == target_size:
        return coefficients
    if target_size > source_size:
        padding = [(0, 0)] * coefficients.ndim
        padding[axis] = (0, target_size - source_size)
        return jnp.pad(coefficients, tuple(padding))
    return jnp.take(coefficients, jnp.arange(target_size), axis=axis)


class PreparedDealiasingPlan(StrictModule, NonTrainableState):
    """Prepared modal embedding, evaluation, projection, and filtering actions."""

    plan: AbstractDealiasingPlan
    retained: TensorSpectralDiscretization
    evaluation: TensorSpectralDiscretization
    modal_masks: tuple[Array, ...]
    report: DealiasingReport
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: AbstractDealiasingPlan,
        retained: TensorSpectralDiscretization,
        evaluation: TensorSpectralDiscretization,
        /,
        *,
        modal_masks: tuple[Array, ...] = (),
        report: DealiasingReport,
    ):
        if not isinstance(plan, AbstractDealiasingPlan):
            raise TypeError("plan must be an AbstractDealiasingPlan.")
        if not isinstance(retained, TensorSpectralDiscretization) or not isinstance(
            evaluation, TensorSpectralDiscretization
        ):
            raise TypeError("retained and evaluation must be tensor spectral spaces.")
        masks = tuple(
            jnp.asarray(mask, dtype=bool).reshape((-1,)) for mask in modal_masks
        )
        if masks and (
            len(masks) != len(retained.axes)
            or any(
                mask.shape != (size,)
                for mask, size in zip(masks, retained.modal_shape, strict=True)
            )
        ):
            raise ValueError("modal_masks must align with retained modal axes.")
        if not isinstance(report, DealiasingReport):
            raise TypeError("report must be a DealiasingReport.")
        self.plan = plan
        self.retained = retained
        self.evaluation = evaluation
        self.modal_masks = masks
        self.report = report
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-dealiasing",
                "plan": plan.plan_id,
                "retained": retained.prepared_id,
                "evaluation": evaluation.prepared_id,
                "report": report.report_id,
            }
        )

    def filter(self, coefficients: ArrayLike, /) -> Array:
        result = self.retained._validate_leading(
            coefficients,
            self.retained.modal_shape,
            "Filtered modal coefficients",
        )
        for axis, mask in enumerate(self.modal_masks):
            shape = [1] * result.ndim
            shape[axis] = mask.size
            result = result * mask.reshape(tuple(shape))
        return result

    def embed(self, coefficients: ArrayLike, /) -> Array:
        result = self.filter(coefficients)
        for axis, (source, target) in enumerate(
            zip(self.retained.axes, self.evaluation.axes, strict=True)
        ):
            if source.family == "fourier":
                result = resize_fourier_axis(result, axis, target.mode_count)
            else:
                result = _resize_nonfourier_axis(result, axis, target.mode_count)
        return result

    def restrict(self, coefficients: ArrayLike, /) -> Array:
        result = self.evaluation._validate_leading(
            coefficients,
            self.evaluation.modal_shape,
            "Evaluation modal coefficients",
        )
        for axis, (source, target) in enumerate(
            zip(self.evaluation.axes, self.retained.axes, strict=True)
        ):
            if source.family == "fourier":
                result = resize_fourier_axis(result, axis, target.mode_count)
            else:
                result = _resize_nonfourier_axis(result, axis, target.mode_count)
        return self.filter(result)

    def reconstruct(self, coefficients: ArrayLike, /) -> Array:
        return self.evaluation.reconstruct(self.embed(coefficients))

    def project(self, values: ArrayLike, /) -> Array:
        return self.restrict(self.evaluation.project(values))


__all__ = [
    "AbstractDealiasingPlan",
    "DealiasingKind",
    "DealiasingReport",
    "ModalFilterPlan",
    "NoDealiasingPlan",
    "PaddingDealiasingPlan",
    "PolynomialClosureDealiasingPlan",
    "PreparedDealiasingPlan",
]
