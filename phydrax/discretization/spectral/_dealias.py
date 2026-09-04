#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import ceil, isfinite, prod
from numbers import Integral, Real
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._space import TensorSpectralDiscretization, TensorSpectralPlan
from ._spherical import SphericalSpectralDiscretization, SphericalSpectralPlan
from ._transfer import (
    prepare_spectral_modal_transfer,
    PreparedSpectralModalTransfer,
)


DealiasingKind: TypeAlias = Literal[
    "none",
    "padding",
    "closure",
    "filter",
    "oversampling",
]


class DealiasingReport(StrictModule, NonTrainableState):
    """Static exactness rationale and retained/evaluation-space evidence."""

    kind: DealiasingKind = eqx.field(static=True)
    retained_shape: tuple[int, ...] = eqx.field(static=True)
    evaluation_shape: tuple[int, ...] = eqx.field(static=True)
    maximum_polynomial_degree: int | None = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    reason: str | None = eqx.field(static=True)
    input_bandlimit: int | None = eqx.field(static=True)
    evaluation_bandlimit: int | None = eqx.field(static=True)
    output_bandlimit: int | None = eqx.field(static=True)
    spin: int | None = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        kind: DealiasingKind,
        retained_shape: tuple[int, ...],
        evaluation_shape: tuple[int, ...],
        maximum_polynomial_degree: int | None,
        exact: bool,
        reason: str | None = None,
        input_bandlimit: int | None = None,
        evaluation_bandlimit: int | None = None,
        output_bandlimit: int | None = None,
        spin: int | None = None,
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
        input_limit = None if input_bandlimit is None else int(input_bandlimit)
        evaluation_limit = (
            None if evaluation_bandlimit is None else int(evaluation_bandlimit)
        )
        output_limit = None if output_bandlimit is None else int(output_bandlimit)
        spin_ = None if spin is None else int(spin)
        reason_ = None if reason is None else str(reason).strip()
        if reason is not None and not reason_:
            raise ValueError("Dealiasing exactness reason must be non-empty.")
        if any(
            value is not None and value <= 0
            for value in (input_limit, evaluation_limit, output_limit)
        ):
            raise ValueError("Spherical dealiasing bandlimits must be positive.")
        self.kind = kind
        self.retained_shape = retained
        self.evaluation_shape = evaluation
        self.maximum_polynomial_degree = degree
        self.exact = bool(exact)
        self.reason = reason_
        self.input_bandlimit = input_limit
        self.evaluation_bandlimit = evaluation_limit
        self.output_bandlimit = output_limit
        self.spin = spin_
        self.report_id = canonical_fingerprint(
            {
                "kind": "spectral-dealiasing-report",
                "strategy": kind,
                "retained_shape": list(retained),
                "evaluation_shape": list(evaluation),
                "maximum_polynomial_degree": degree,
                "exact": bool(exact),
                **({} if reason_ is None else {"reason": reason_}),
                "input_bandlimit": input_limit,
                "evaluation_bandlimit": evaluation_limit,
                "output_bandlimit": output_limit,
                "spin": spin_,
            }
        )


class AbstractDealiasingPlan(StrictModule, NonTrainableState):
    """Symbolic nonlinear spectral evaluation policy."""

    kind: DealiasingKind = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def prepare(
        self,
        discretization: TensorSpectralDiscretization | SphericalSpectralDiscretization,
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
        discretization: TensorSpectralDiscretization | SphericalSpectralDiscretization,
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
                retained_shape=_modal_shape(discretization),
                evaluation_shape=_modal_shape(discretization),
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
        discretization: TensorSpectralDiscretization | SphericalSpectralDiscretization,
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
        if isinstance(discretization, SphericalSpectralDiscretization):
            evaluation_limit = (
                self.maximum_polynomial_degree * (discretization.layout.bandlimit - 1) + 1
            )
            padded = SphericalSpectralPlan(
                evaluation_limit,
                sampling=discretization.plan.sampling,
                spin=discretization.plan.spin,
                reality=discretization.plan.reality,
                execution=discretization.plan.execution,
                field_name=discretization.plan.field_name,
                precision=discretization.plan.precision,
                max_precompute_bytes=discretization.plan.max_precompute_bytes,
                max_explicit_eigenbasis_bytes=(
                    discretization.plan.max_explicit_eigenbasis_bytes
                ),
                max_dense_operator_bytes=(discretization.plan.max_dense_operator_bytes),
            ).prepare(
                radius=discretization.radius,
                numeric_version=discretization.numeric_version,
            )
            target = padded.coefficient_shape
            exact = True
        else:
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
            domains = tuple(axis.domain for axis in discretization.axes)
            padded = padded_plan.prepare(
                domains,
                numeric_version=discretization.numeric_version,
            )
            exact = all(axis.family != "sine" for axis in discretization.axes)
        return PreparedDealiasingPlan(
            self,
            discretization,
            padded,
            report=DealiasingReport(
                kind="padding",
                retained_shape=_modal_shape(discretization),
                evaluation_shape=target,
                maximum_polynomial_degree=self.maximum_polynomial_degree,
                exact=exact,
                input_bandlimit=(
                    discretization.layout.bandlimit
                    if isinstance(discretization, SphericalSpectralDiscretization)
                    else None
                ),
                evaluation_bandlimit=(
                    padded.layout.bandlimit
                    if isinstance(padded, SphericalSpectralDiscretization)
                    else None
                ),
                output_bandlimit=(
                    discretization.layout.bandlimit
                    if isinstance(discretization, SphericalSpectralDiscretization)
                    else None
                ),
                spin=(
                    discretization.layout.spin
                    if isinstance(discretization, SphericalSpectralDiscretization)
                    else None
                ),
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
        domains = tuple(axis.domain for axis in discretization.axes)
        closure = closure_plan.prepare(
            domains,
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


class OversamplingDealiasingPlan(AbstractDealiasingPlan):
    """Bounded approximate oversampling for general Fourier nonlinearities."""

    factor: float = eqx.field(static=True)
    maximum_evaluation_modes: int = eqx.field(static=True)

    def __init__(
        self,
        factor: float,
        /,
        *,
        maximum_evaluation_modes: int = 16_777_216,
    ):
        if isinstance(factor, bool) or not isinstance(factor, Real):
            raise TypeError("factor must be a real number.")
        factor_ = float(factor)
        if not isfinite(factor_) or factor_ <= 1.0:
            raise ValueError("factor must be finite and greater than one.")
        if isinstance(maximum_evaluation_modes, bool) or not isinstance(
            maximum_evaluation_modes, Integral
        ):
            raise TypeError("maximum_evaluation_modes must be an integer.")
        maximum = int(maximum_evaluation_modes)
        if maximum <= 0:
            raise ValueError("maximum_evaluation_modes must be positive.")
        self.factor = factor_
        self.maximum_evaluation_modes = maximum
        self.kind = "oversampling"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "oversampling-spectral-dealiasing",
                "factor": factor_,
                "maximum_evaluation_modes": maximum,
            }
        )

    def prepare(
        self,
        discretization: TensorSpectralDiscretization | SphericalSpectralDiscretization,
        /,
        *,
        required_polynomial_degree: int | None,
    ) -> "PreparedDealiasingPlan":
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError(
                "Oversampling dealiasing requires a tensor spectral discretization."
            )
        if any(axis.family != "fourier" for axis in discretization.axes):
            raise ValueError(
                "Oversampling dealiasing supports tensor Fourier bases only."
            )
        numerator, denominator = self.factor.as_integer_ratio()
        target = tuple(
            (numerator * size + denominator - 1) // denominator
            for size in discretization.modal_shape
        )
        if prod(target) > self.maximum_evaluation_modes:
            raise ValueError(
                "Oversampling exceeds maximum_evaluation_modes before spectral "
                "preparation."
            )
        basis_plans = tuple(
            axis.plan.resized(count)
            for axis, count in zip(discretization.axes, target, strict=True)
        )
        evaluation_plan = TensorSpectralPlan(
            basis_plans,
            axis_names=discretization.plan.axis_names,
            field_name=discretization.plan.field_name,
            precision=discretization.plan.precision,
        )
        evaluation = evaluation_plan.prepare(
            tuple(axis.domain for axis in discretization.axes),
            numeric_version=discretization.numeric_version,
        )
        return PreparedDealiasingPlan(
            self,
            discretization,
            evaluation,
            report=DealiasingReport(
                kind="oversampling",
                retained_shape=discretization.modal_shape,
                evaluation_shape=target,
                maximum_polynomial_degree=required_polynomial_degree,
                exact=False,
                reason=(
                    "Finite oversampling approximates nonpolynomial evaluation and "
                    "does not certify alias-free retained modes."
                ),
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
        discretization: TensorSpectralDiscretization | SphericalSpectralDiscretization,
        /,
        *,
        required_polynomial_degree: int | None,
    ) -> "PreparedDealiasingPlan":
        if isinstance(discretization, SphericalSpectralDiscretization):
            maximum = discretization.layout.bandlimit - 1
            mask = discretization.layout.valid_mask & (
                discretization.layout.degrees <= self.cutoff_fraction * maximum
            )
            masks = (mask,)
        else:
            tensor_masks = []
            for axis in discretization.axes:
                numbers = jnp.abs(axis.modes.mode_numbers)
                maximum = jnp.max(numbers)
                tensor_masks.append(numbers <= self.cutoff_fraction * maximum)
            masks = tuple(tensor_masks)
        return PreparedDealiasingPlan(
            self,
            discretization,
            discretization,
            modal_masks=masks,
            report=DealiasingReport(
                kind="filter",
                retained_shape=_modal_shape(discretization),
                evaluation_shape=_modal_shape(discretization),
                maximum_polynomial_degree=required_polynomial_degree,
                exact=False,
                input_bandlimit=(
                    discretization.layout.bandlimit
                    if isinstance(discretization, SphericalSpectralDiscretization)
                    else None
                ),
                evaluation_bandlimit=(
                    discretization.layout.bandlimit
                    if isinstance(discretization, SphericalSpectralDiscretization)
                    else None
                ),
                output_bandlimit=(
                    discretization.layout.bandlimit
                    if isinstance(discretization, SphericalSpectralDiscretization)
                    else None
                ),
                spin=(
                    discretization.layout.spin
                    if isinstance(discretization, SphericalSpectralDiscretization)
                    else None
                ),
            ),
        )


class PreparedDealiasingPlan(StrictModule, NonTrainableState):
    """Prepared modal embedding, evaluation, projection, and filtering actions."""

    plan: AbstractDealiasingPlan
    retained: TensorSpectralDiscretization | SphericalSpectralDiscretization
    evaluation: TensorSpectralDiscretization | SphericalSpectralDiscretization
    embedding: PreparedSpectralModalTransfer
    restriction: PreparedSpectralModalTransfer
    modal_masks: tuple[Array, ...]
    report: DealiasingReport
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: AbstractDealiasingPlan,
        retained: TensorSpectralDiscretization | SphericalSpectralDiscretization,
        evaluation: TensorSpectralDiscretization | SphericalSpectralDiscretization,
        /,
        *,
        modal_masks: tuple[Array, ...] = (),
        report: DealiasingReport,
    ):
        if not isinstance(plan, AbstractDealiasingPlan):
            raise TypeError("plan must be an AbstractDealiasingPlan.")
        tensor_pair = isinstance(retained, TensorSpectralDiscretization) and isinstance(
            evaluation, TensorSpectralDiscretization
        )
        spherical_pair = isinstance(
            retained, SphericalSpectralDiscretization
        ) and isinstance(evaluation, SphericalSpectralDiscretization)
        if not tensor_pair and not spherical_pair:
            raise TypeError("retained and evaluation must use one spectral family.")
        if spherical_pair:
            masks = tuple(jnp.asarray(mask, dtype=bool) for mask in modal_masks)
            if masks and (
                len(masks) != 1 or masks[0].shape != retained.coefficient_shape
            ):
                raise ValueError(
                    "Spherical modal filtering requires one coefficient-shape mask."
                )
        else:
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
        embedding = prepare_spectral_modal_transfer(retained, evaluation)
        restriction = prepare_spectral_modal_transfer(evaluation, retained)
        self.plan = plan
        self.retained = retained
        self.evaluation = evaluation
        self.modal_masks = masks
        self.report = report
        self.embedding = embedding
        self.restriction = restriction
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-dealiasing",
                "plan": plan.plan_id,
                "retained": retained.prepared_id,
                "evaluation": evaluation.prepared_id,
                "embedding": embedding.prepared_id,
                "restriction": restriction.prepared_id,
                "report": report.report_id,
            }
        )

    def filter(self, coefficients: ArrayLike, /) -> Array:
        result = _validate_modal(
            self.retained,
            coefficients,
            "Filtered modal coefficients",
        )
        if isinstance(self.retained, SphericalSpectralDiscretization):
            if self.modal_masks:
                mask = self.modal_masks[0].reshape(
                    self.retained.coefficient_shape
                    + (1,) * (result.ndim - len(self.retained.coefficient_shape))
                )
                result = result * mask
            return self.retained.layout.mask_invalid(result)
        for axis, mask in enumerate(self.modal_masks):
            shape = [1] * result.ndim
            shape[axis] = mask.size
            result = result * mask.reshape(tuple(shape))
        return result

    def embed(self, coefficients: ArrayLike, /) -> Array:
        return self.embedding(self.filter(coefficients))

    def restrict(self, coefficients: ArrayLike, /) -> Array:
        result = _validate_modal(
            self.evaluation,
            coefficients,
            "Evaluation modal coefficients",
        )
        return self.filter(self.restriction(result))

    def reconstruct(self, coefficients: ArrayLike, /) -> Array:
        return self.evaluation.reconstruct(self.embed(coefficients))

    def project(self, values: ArrayLike, /) -> Array:
        return self.restrict(self.evaluation.project(values))


def _modal_shape(
    discretization: TensorSpectralDiscretization | SphericalSpectralDiscretization,
    /,
) -> tuple[int, ...]:
    if isinstance(discretization, TensorSpectralDiscretization):
        return discretization.modal_shape
    return discretization.coefficient_shape


def _validate_modal(
    discretization: TensorSpectralDiscretization | SphericalSpectralDiscretization,
    coefficients: ArrayLike,
    name: str,
    /,
) -> Array:
    shape = _modal_shape(discretization)
    if isinstance(discretization, TensorSpectralDiscretization):
        return discretization._validate_leading(coefficients, shape, name)
    values = jnp.asarray(coefficients)
    if values.ndim < len(shape) or tuple(values.shape[: len(shape)]) != shape:
        raise ValueError(f"{name} must begin with shape {shape}; got {values.shape}.")
    return discretization.layout.mask_invalid(values)


__all__ = [
    "AbstractDealiasingPlan",
    "DealiasingKind",
    "DealiasingReport",
    "ModalFilterPlan",
    "NoDealiasingPlan",
    "PaddingDealiasingPlan",
    "OversamplingDealiasingPlan",
    "PolynomialClosureDealiasingPlan",
    "PreparedDealiasingPlan",
]
