#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fail-closed inference adapters for dark-matter analysis products.

Smooth adapters are restricted to an already selected fixed grid and branch.
Stochastic adapters compare a score estimator with a common-random central
finite difference on one immutable tape.  Topology selection, collision events,
and reactions are always forward evidence and never differentiation targets.
External/reference products are stored behind a constant differentiation contract.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import ScientificArtifactEnvelope
from ...observation import (
    CholeskyCovarianceAction,
    CoordinateLayout,
    LinearObservationPlan,
    TheoryVector,
)
from ...qualification import ReferenceArtifactManifest
from ...uq._metrics import (
    GaussianScaleCalibrator,
    interval_calibration_diagnostics,
    IntervalCalibrationDiagnostics,
)
from ._products import CosmologyProductProvenance
from ._spectral_statistics import (
    SpectralFieldDiscrepancyPlan,
    SpectralFieldDiscrepancyResult,
)


SmoothDarkMatterKind: TypeAlias = Literal["wave-fixed-grid", "mixed-fixed-grid"]


def _identifier(value: str, name: str, /) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty identifier.")
    return normalized


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _parameter_vector(
    value: ArrayLike,
    size: int,
    name: str,
    /,
    *,
    dtype=None,
) -> Array:
    array = jnp.asarray(value, dtype=dtype).reshape((-1,))
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}.")
    if not eqx.is_inexact_array(array):
        array = array.astype("float64")
    return array


def _scalar_flag(value: ArrayLike, name: str, /) -> Array:
    flag = jnp.asarray(value, dtype=jnp.bool_)
    if flag.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return jax.lax.stop_gradient(flag)


class DarkMatterInferenceEvaluation(StrictModule):
    """One product vector and all evidence controlling derivative admission."""

    values: Array
    branch_signature: Array
    finite: Array
    successful: Array
    fixed_grid: Array
    topology_fixed: Array
    event_free: Array
    reaction_free: Array
    derivative_valid: Array
    product_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        branch_signature: ArrayLike,
        /,
        *,
        finite: ArrayLike,
        successful: ArrayLike,
        fixed_grid: ArrayLike,
        topology_fixed: ArrayLike,
        event_free: ArrayLike,
        reaction_free: ArrayLike,
        derivative_valid: ArrayLike,
        product_id: str,
        realization_id: str,
    ):
        values_ = jnp.asarray(values).reshape((-1,))
        if (
            values_.size == 0
            or not eqx.is_inexact_array(values_)
            or jnp.issubdtype(values_.dtype, jnp.complexfloating)
        ):
            raise TypeError(
                "Dark-matter inference values must be a nonempty real inexact vector."
            )
        signature = jax.lax.stop_gradient(
            jnp.asarray(branch_signature, dtype=jnp.int32).reshape((-1,))
        )
        if signature.size == 0:
            raise ValueError("Dark-matter branch signature must be nonempty.")
        self.values = values_
        self.branch_signature = signature
        self.finite = _scalar_flag(finite, "finite")
        self.successful = _scalar_flag(successful, "successful")
        self.fixed_grid = _scalar_flag(fixed_grid, "fixed_grid")
        self.topology_fixed = _scalar_flag(topology_fixed, "topology_fixed")
        self.event_free = _scalar_flag(event_free, "event_free")
        self.reaction_free = _scalar_flag(reaction_free, "reaction_free")
        self.derivative_valid = _scalar_flag(derivative_valid, "derivative_valid")
        self.product_id = _identifier(product_id, "product_id")
        self.realization_id = _identifier(realization_id, "realization_id")

    @property
    def sensitivity_eligible(self) -> Array:
        return (
            self.finite
            & self.successful
            & self.fixed_grid
            & self.topology_fixed
            & self.event_free
            & self.reaction_free
            & self.derivative_valid
            & jnp.all(jnp.isfinite(self.values))
        )


class SmoothFixedGridSensitivityProduct(StrictModule):
    """Product-level JVP with branch-stable central-difference evidence."""

    value: Array
    jvp: Array
    finite_difference: Array
    absolute_residual: Array
    relative_residual: Array
    center_eligible: Array
    stencil_eligible: Array
    branch_stable: Array
    finite: Array
    successful: Array
    epsilon: Array
    kind: SmoothDarkMatterKind = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)


class SmoothFixedGridDarkMatterInferencePlan(StrictModule, NonTrainableState):
    """Differentiate only numeric values on a fixed wave or mixed-matter branch.

    ``topology_sensitive`` is a static refusal for soliton-center selection, vortex
    winding, halo membership, and analogous discrete products.  Dynamic event,
    reaction, topology, branch, or grid changes are checked at the center and both
    common central-difference stencil points.
    """

    evaluator: Callable[[Array], DarkMatterInferenceEvaluation] = eqx.field(static=True)
    expected_branch_signature: Array
    parameter_count: int = eqx.field(static=True)
    output_count: int = eqx.field(static=True)
    kind: SmoothDarkMatterKind = eqx.field(static=True)
    topology_sensitive: bool = eqx.field(static=True)
    evaluator_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: Callable[[Array], DarkMatterInferenceEvaluation],
        expected_branch_signature: ArrayLike,
        /,
        *,
        parameter_count: int,
        output_count: int,
        kind: SmoothDarkMatterKind,
        topology_sensitive: bool,
        evaluator_id: str,
        realization_id: str,
        product_id: str,
    ):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        parameters = _positive_integer(parameter_count, "parameter_count")
        outputs = _positive_integer(output_count, "output_count")
        if kind not in ("wave-fixed-grid", "mixed-fixed-grid"):
            raise ValueError("Unknown smooth dark-matter inference kind.")
        signature = jax.lax.stop_gradient(
            jnp.asarray(expected_branch_signature, dtype=jnp.int32).reshape((-1,))
        )
        if signature.size == 0:
            raise ValueError("expected_branch_signature must be nonempty.")
        evaluator_identity = _identifier(evaluator_id, "evaluator_id")
        realization = _identifier(realization_id, "realization_id")
        product = _identifier(product_id, "product_id")
        self.evaluator = evaluator
        self.expected_branch_signature = signature
        self.parameter_count = parameters
        self.output_count = outputs
        self.kind = kind
        self.topology_sensitive = bool(topology_sensitive)
        self.evaluator_id = evaluator_identity
        self.realization_id = realization
        self.product_id = product
        self.adapter_id = canonical_fingerprint(
            {
                "kind": "smooth-fixed-grid-dark-matter-inference",
                "profile": kind,
                "topology_sensitive": bool(topology_sensitive),
                "evaluator": evaluator_identity,
                "realization": realization,
                "product": product,
                "parameter_count": parameters,
                "output_count": outputs,
                "branch_signature": array_tree_fingerprint(signature),
            }
        )

    def evaluate(self, parameters: ArrayLike, /) -> DarkMatterInferenceEvaluation:
        point = _parameter_vector(parameters, self.parameter_count, "parameters")
        result = self.evaluator(point)
        if not isinstance(result, DarkMatterInferenceEvaluation):
            raise TypeError("evaluator must return DarkMatterInferenceEvaluation.")
        if result.values.shape != (self.output_count,):
            raise ValueError("Dark-matter inference output shape changed.")
        if result.branch_signature.shape != self.expected_branch_signature.shape:
            raise ValueError("Dark-matter inference branch-signature shape changed.")
        if (
            result.realization_id != self.realization_id
            or result.product_id != self.product_id
        ):
            raise ValueError("Dark-matter inference product identity changed.")
        matches = jnp.all(result.branch_signature == self.expected_branch_signature)
        return eqx.tree_at(
            lambda value: value.derivative_valid,
            result,
            result.derivative_valid & matches,
        )

    def values(self, parameters: ArrayLike, /) -> Array:
        return self.evaluate(parameters).values

    def sensitivity(
        self,
        parameters: ArrayLike,
        direction: ArrayLike,
        /,
        *,
        epsilon: float = 1.0e-4,
        absolute_tolerance: float = 1.0e-7,
        relative_tolerance: float = 1.0e-4,
    ) -> SmoothFixedGridSensitivityProduct:
        if self.topology_sensitive:
            raise ValueError(
                "Topology-sensitive soliton/vortex/event products do not admit gradients."
            )
        point = _parameter_vector(parameters, self.parameter_count, "parameters")
        tangent = _parameter_vector(
            direction,
            self.parameter_count,
            "direction",
            dtype=point.dtype,
        )
        step = float(epsilon)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if (
            not np.isfinite(step)
            or step <= 0.0
            or not np.isfinite(absolute)
            or absolute < 0.0
            or not np.isfinite(relative)
            or relative < 0.0
        ):
            raise ValueError("Sensitivity step and tolerances are invalid.")
        value, raw_jvp = jax.jvp(self.values, (point,), (tangent,))
        center = self.evaluate(point)
        lower = self.evaluate(point - step * tangent)
        upper = self.evaluate(point + step * tangent)
        finite_difference = (upper.values - lower.values) / (2.0 * step)
        stencil_eligible = lower.sensitivity_eligible & upper.sensitivity_eligible
        branch_stable = jnp.all(
            lower.branch_signature == center.branch_signature
        ) & jnp.all(upper.branch_signature == center.branch_signature)
        finite = (
            jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(raw_jvp))
            & jnp.all(jnp.isfinite(finite_difference))
        )
        eligible = center.sensitivity_eligible & stencil_eligible & branch_stable & finite
        jvp = eqx.error_if(
            raw_jvp,
            ~eligible,
            "Dark-matter sensitivity crossed a grid, topology, event, reaction, or branch boundary.",
        )
        residual = jnp.abs(jvp - finite_difference)
        scale = jnp.maximum(jnp.maximum(jnp.abs(jvp), jnp.abs(finite_difference)), 1.0)
        relative_residual = residual / scale
        accepted = jnp.all(residual <= absolute + relative * scale)
        return SmoothFixedGridSensitivityProduct(
            value,
            jvp,
            finite_difference,
            residual,
            relative_residual,
            center.sensitivity_eligible,
            stencil_eligible,
            branch_stable,
            finite,
            eligible & accepted,
            jnp.asarray(step, dtype=point.dtype),
            self.kind,
            self.adapter_id,
            self.realization_id,
            self.product_id,
        )


class FixedTapeStochasticEvaluation(StrictModule):
    """Per-draw observable and score values on one immutable stochastic tape."""

    values: Array
    score: Array
    active: Array
    event_occurred: Array
    reaction_occurred: Array
    topology_changed: Array
    finite: Array
    successful: Array
    tape_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        score: ArrayLike,
        active: ArrayLike,
        /,
        *,
        event_occurred: ArrayLike,
        reaction_occurred: ArrayLike,
        topology_changed: ArrayLike,
        finite: ArrayLike,
        successful: ArrayLike,
        tape_id: str,
        product_id: str,
    ):
        value = jnp.asarray(values)
        scores = jnp.asarray(score, dtype=value.real.dtype)
        active_ = jnp.asarray(active, dtype=jnp.bool_).reshape((-1,))
        if (
            value.ndim != 2
            or not eqx.is_inexact_array(value)
            or jnp.issubdtype(value.dtype, jnp.complexfloating)
        ):
            raise ValueError(
                "Fixed-tape values must be a real array with shape (draw, output)."
            )
        if scores.ndim != 2 or scores.shape[0] != value.shape[0]:
            raise ValueError("Fixed-tape score must have shape (draw, parameter).")
        if active_.shape != (value.shape[0],):
            raise ValueError("Fixed-tape active mask must match draw count.")
        masks = tuple(
            jnp.asarray(item, dtype=jnp.bool_).reshape((-1,))
            for item in (event_occurred, reaction_occurred, topology_changed)
        )
        if any(mask.shape != active_.shape for mask in masks):
            raise ValueError("Fixed-tape event/reaction/topology masks must match draws.")
        self.values = value
        self.score = scores
        self.active = jax.lax.stop_gradient(active_)
        self.event_occurred = jax.lax.stop_gradient(masks[0])
        self.reaction_occurred = jax.lax.stop_gradient(masks[1])
        self.topology_changed = jax.lax.stop_gradient(masks[2])
        self.finite = _scalar_flag(finite, "finite")
        self.successful = _scalar_flag(successful, "successful")
        self.tape_id = _identifier(tape_id, "tape_id")
        self.product_id = _identifier(product_id, "product_id")


class StochasticSensitivityEvidence(StrictModule):
    """Score and common-random finite-difference estimates with uncertainty."""

    value: Array
    score_estimate: Array
    score_standard_error: Array
    common_random_finite_difference: Array
    finite_difference_standard_error: Array
    estimator_bias: Array
    combined_standard_error: Array
    paired_bias_standard_error: Array
    bias_z_score: Array
    bias_flag: Array
    normalized_weights: Array
    effective_sample_size: Array
    active_draw_count: Array
    fixed_tape: Array
    event_free: Array
    reaction_free: Array
    topology_fixed: Array
    finite: Array
    successful: Array
    epsilon: Array
    tape_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def _weighted_mean_and_standard_error(
    samples: Array,
    normalized_weights: Array,
    /,
) -> tuple[Array, Array]:
    mean = ein.contract("n,no->o", normalized_weights, samples)
    centered = samples - mean
    squared_weight = jnp.sum(normalized_weights * normalized_weights)
    correction = jnp.maximum(
        1.0 - squared_weight,
        jnp.finfo(normalized_weights.dtype).eps,
    )
    population_variance = (
        ein.contract("n,no,no->o", normalized_weights, centered, centered) / correction
    )
    standard_error = jnp.sqrt(jnp.maximum(population_variance * squared_weight, 0.0))
    return mean, standard_error


class FixedTapeStochasticSensitivityPlan(StrictModule, NonTrainableState):
    """Audit stochastic sensitivity on one stop-gradient common-random tape."""

    evaluator: Callable[[Array, Any], FixedTapeStochasticEvaluation] = eqx.field(
        static=True
    )
    tape: Any
    draw_weights: Array
    parameter_count: int = eqx.field(static=True)
    output_count: int = eqx.field(static=True)
    draw_count: int = eqx.field(static=True)
    minimum_effective_sample_size: float = eqx.field(static=True)
    bias_absolute_tolerance: float = eqx.field(static=True)
    bias_standard_error_multiplier: float = eqx.field(static=True)
    evaluator_id: str = eqx.field(static=True)
    tape_id: str = eqx.field(static=True)
    tape_content_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: Callable[[Array, Any], FixedTapeStochasticEvaluation],
        tape: Any,
        draw_weights: ArrayLike,
        /,
        *,
        parameter_count: int,
        output_count: int,
        minimum_effective_sample_size: float,
        bias_absolute_tolerance: float = 0.0,
        bias_standard_error_multiplier: float = 2.0,
        evaluator_id: str,
        tape_id: str,
        product_id: str,
    ):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        parameters = _positive_integer(parameter_count, "parameter_count")
        outputs = _positive_integer(output_count, "output_count")
        weights_host = np.asarray(draw_weights, dtype=np.float64).reshape((-1,))
        if (
            weights_host.size < 2
            or np.any(~np.isfinite(weights_host))
            or np.any(weights_host < 0.0)
            or not np.any(weights_host > 0.0)
        ):
            raise ValueError(
                "draw_weights must contain at least two finite nonnegative draws."
            )
        minimum = float(minimum_effective_sample_size)
        absolute = float(bias_absolute_tolerance)
        multiplier = float(bias_standard_error_multiplier)
        if (
            not np.isfinite(minimum)
            or not 1.0 <= minimum <= weights_host.size
            or not np.isfinite(absolute)
            or absolute < 0.0
            or not np.isfinite(multiplier)
            or multiplier < 0.0
        ):
            raise ValueError("Stochastic sensitivity evidence thresholds are invalid.")
        self.evaluator = evaluator
        self.tape = jax.tree.map(
            lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
            tape,
        )
        self.draw_weights = jax.lax.stop_gradient(jnp.asarray(weights_host))
        self.parameter_count = parameters
        self.output_count = outputs
        self.draw_count = weights_host.size
        self.minimum_effective_sample_size = minimum
        self.bias_absolute_tolerance = absolute
        self.bias_standard_error_multiplier = multiplier
        self.evaluator_id = _identifier(evaluator_id, "evaluator_id")
        self.tape_id = _identifier(tape_id, "tape_id")
        self.tape_content_id = canonical_fingerprint(
            {
                "kind": "fixed-stochastic-tape-content",
                "arrays": array_tree_fingerprint(self.tape),
            }
        )
        self.product_id = _identifier(product_id, "product_id")
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-tape-stochastic-dark-matter-sensitivity",
                "evaluator": self.evaluator_id,
                "tape_id": self.tape_id,
                "tape_content": self.tape_content_id,
                "product": self.product_id,
                "parameter_count": parameters,
                "output_count": outputs,
                "draw_weights": weights_host.tolist(),
                "minimum_effective_sample_size": minimum,
                "bias_absolute_tolerance": absolute,
                "bias_standard_error_multiplier": multiplier,
            }
        )

    def evaluate(self, parameters: ArrayLike, /) -> FixedTapeStochasticEvaluation:
        point = _parameter_vector(parameters, self.parameter_count, "parameters")
        result = self.evaluator(point, self.tape)
        if not isinstance(result, FixedTapeStochasticEvaluation):
            raise TypeError("evaluator must return FixedTapeStochasticEvaluation.")
        if result.values.shape != (self.draw_count, self.output_count):
            raise ValueError("Fixed-tape observable shape changed.")
        if result.score.shape != (self.draw_count, self.parameter_count):
            raise ValueError("Fixed-tape score shape changed.")
        if result.tape_id != self.tape_id or result.product_id != self.product_id:
            raise ValueError("Fixed-tape stochastic identity changed.")
        return result

    def sensitivity(
        self,
        parameters: ArrayLike,
        direction: ArrayLike,
        /,
        *,
        epsilon: float = 1.0e-4,
    ) -> StochasticSensitivityEvidence:
        point = _parameter_vector(parameters, self.parameter_count, "parameters")
        tangent = _parameter_vector(
            direction,
            self.parameter_count,
            "direction",
            dtype=point.dtype,
        )
        step = float(epsilon)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("epsilon must be finite and positive.")
        center = self.evaluate(point)
        lower = self.evaluate(point - step * tangent)
        upper = self.evaluate(point + step * tangent)
        same_active = jnp.all(center.active == lower.active) & jnp.all(
            center.active == upper.active
        )
        fixed_tape = jnp.asarray(
            center.tape_id == lower.tape_id == upper.tape_id == self.tape_id
        )
        active = center.active
        raw_weights = jnp.where(active, self.draw_weights, 0.0)
        total_weight = jnp.sum(raw_weights)
        normalized = raw_weights / jnp.where(total_weight > 0.0, total_weight, 1.0)
        effective = 1.0 / jnp.maximum(
            jnp.sum(normalized * normalized), jnp.finfo(normalized.dtype).tiny
        )
        center_values = jnp.where(active[:, None], center.values, 0.0)
        lower_values = jnp.where(active[:, None], lower.values, 0.0)
        upper_values = jnp.where(active[:, None], upper.values, 0.0)
        center_score = jnp.where(active[:, None], center.score, 0.0)
        value, _ = _weighted_mean_and_standard_error(center_values, normalized)
        directional_score = ein.contract("np,p->n", center_score, tangent)
        score_samples = jnp.where(
            active[:, None],
            center_values * directional_score[:, None],
            0.0,
        )
        score_estimate, score_se = _weighted_mean_and_standard_error(
            score_samples, normalized
        )
        finite_difference_samples = jnp.where(
            active[:, None],
            (upper_values - lower_values) / (2.0 * step),
            0.0,
        )
        finite_difference, finite_difference_se = _weighted_mean_and_standard_error(
            finite_difference_samples, normalized
        )
        paired_bias_samples = score_samples - finite_difference_samples
        bias, paired_bias_se = _weighted_mean_and_standard_error(
            paired_bias_samples, normalized
        )
        combined = paired_bias_se
        safe_combined = jnp.maximum(combined, jnp.finfo(combined.dtype).tiny)
        z_score = jnp.abs(bias) / safe_combined
        threshold = (
            self.bias_absolute_tolerance
            + self.bias_standard_error_multiplier * paired_bias_se
        )
        bias_flag = jnp.abs(bias) > threshold
        event_free = ~jnp.any(
            active & (center.event_occurred | lower.event_occurred | upper.event_occurred)
        )
        reaction_free = ~jnp.any(
            active
            & (
                center.reaction_occurred
                | lower.reaction_occurred
                | upper.reaction_occurred
            )
        )
        topology_fixed = ~jnp.any(
            active
            & (center.topology_changed | lower.topology_changed | upper.topology_changed)
        )
        evaluations_successful = (
            center.successful
            & lower.successful
            & upper.successful
            & center.finite
            & lower.finite
            & upper.finite
        )
        finite = (
            evaluations_successful
            & same_active
            & fixed_tape
            & (total_weight > 0.0)
            & jnp.all(
                jnp.where(
                    active[:, None],
                    jnp.isfinite(center.values)
                    & jnp.isfinite(lower.values)
                    & jnp.isfinite(upper.values)
                    & jnp.isfinite(center.score),
                    True,
                )
            )
            & jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(score_estimate))
            & jnp.all(jnp.isfinite(score_se))
            & jnp.all(jnp.isfinite(finite_difference))
            & jnp.all(jnp.isfinite(finite_difference_se))
            & jnp.all(jnp.isfinite(bias))
            & jnp.all(jnp.isfinite(paired_bias_se))
            & jnp.all(jnp.isfinite(z_score))
            & jnp.isfinite(effective)
        )
        successful = (
            finite
            & event_free
            & reaction_free
            & topology_fixed
            & (effective >= self.minimum_effective_sample_size)
            & ~jnp.any(bias_flag)
        )
        return StochasticSensitivityEvidence(
            value,
            score_estimate,
            score_se,
            finite_difference,
            finite_difference_se,
            bias,
            combined,
            paired_bias_se,
            z_score,
            bias_flag,
            normalized,
            effective,
            jnp.sum(active.astype(jnp.int32)),
            fixed_tape,
            event_free,
            reaction_free,
            topology_fixed,
            finite,
            successful,
            jnp.asarray(step, dtype=point.dtype),
            self.tape_id,
            self.product_id,
            self.plan_id,
        )


class DarkMatterCoordinateContract(StrictModule, NonTrainableState):
    """Ordered observable coordinates with one explicit unit identity per value."""

    layout: CoordinateLayout
    unit_ids: tuple[str, ...] = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: CoordinateLayout,
        unit_ids: Sequence[str],
        /,
    ):
        if not isinstance(layout, CoordinateLayout):
            raise TypeError("layout must be CoordinateLayout.")
        units = tuple(_identifier(value, "unit_id") for value in unit_ids)
        if len(units) != layout.size:
            raise ValueError("One unit identity is required per observable coordinate.")
        self.layout = layout
        self.unit_ids = units
        self.contract_id = canonical_fingerprint(
            {
                "kind": "dark-matter-coordinate-contract",
                "layout": layout.layout_id,
                "unit_ids": list(units),
            }
        )


class ConstantExternalDarkMatterProduct(StrictModule, NonTrainableState):
    """Checksum-verified, rights-governed, constant external observable vector."""

    values: Array
    coordinates: DarkMatterCoordinateContract
    manifest: ReferenceArtifactManifest
    artifact: ScientificArtifactEnvelope
    provenance: CosmologyProductProvenance
    finite: Array
    successful: Array
    requested_use: tuple[bool, bool, bool, bool] = eqx.field(static=True)
    requested_use_id: str = eqx.field(static=True)
    decoder_id: str = eqx.field(static=True)
    declared_product_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        artifact_bytes: bytes,
        decoder: Callable[[bytes], ArrayLike],
        manifest: ReferenceArtifactManifest,
        artifact: ScientificArtifactEnvelope,
        provenance: CosmologyProductProvenance,
        coordinates: DarkMatterCoordinateContract,
        /,
        *,
        decoder_id: str,
        product_id: str,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ):
        if not callable(decoder):
            raise TypeError("decoder must be callable.")
        if not isinstance(manifest, ReferenceArtifactManifest):
            raise TypeError("manifest must be ReferenceArtifactManifest.")
        if not isinstance(artifact, ScientificArtifactEnvelope):
            raise TypeError("artifact must be ScientificArtifactEnvelope.")
        if not isinstance(provenance, CosmologyProductProvenance):
            raise TypeError("provenance must be CosmologyProductProvenance.")
        if not isinstance(coordinates, DarkMatterCoordinateContract):
            raise TypeError("coordinates must be DarkMatterCoordinateContract.")
        manifest.require_rights(
            commercial_use=commercial_use,
            redistribution=redistribution,
            training_use=training_use,
            export=export,
        )
        manifest.require_uncertainty()
        verified_manifest_id = manifest.verify_bytes(artifact_bytes)
        decoded_host = np.asarray(decoder(artifact_bytes)).reshape((-1,))
        supplied_host = np.asarray(values).reshape((-1,))
        if decoded_host.shape != supplied_host.shape or not np.array_equal(
            decoded_host, supplied_host, equal_nan=True
        ):
            raise ValueError(
                "External values do not match the checksum-verified decoder output."
            )
        requested_use = (
            commercial_use,
            redistribution,
            training_use,
            export,
        )
        requested_use_id = canonical_fingerprint(
            {
                "kind": "external-dark-matter-requested-use",
                "manifest": manifest.manifest_id,
                "commercial_use": commercial_use,
                "redistribution": redistribution,
                "training_use": training_use,
                "export": export,
            }
        )
        if (
            artifact.status != "complete"
            or artifact.content_digest != manifest.checksum
            or artifact.license_id != manifest.license_id
            or manifest.manifest_id not in artifact.parent_artifact_ids
            or provenance.source_kind != "external"
            or provenance.differentiation.supported_surfaces
            or verified_manifest_id != manifest.manifest_id
        ):
            raise ValueError(
                "External dark-matter products require exact checksum/license lineage "
                "and a constant differentiation contract."
            )
        declared = _identifier(product_id, "product_id")
        decoder_identity = _identifier(decoder_id, "decoder_id")
        value = jax.lax.stop_gradient(jnp.asarray(decoded_host).reshape((-1,)))
        if (
            value.size == 0
            or not eqx.is_inexact_array(value)
            or jnp.issubdtype(value.dtype, jnp.complexfloating)
        ):
            raise TypeError(
                "External dark-matter values must be a nonempty real inexact vector."
            )
        if value.shape != (coordinates.layout.size,):
            raise ValueError("External values must match their coordinate contract.")
        finite = jnp.all(jnp.isfinite(value))
        self.values = value
        self.coordinates = coordinates
        self.manifest = manifest
        self.artifact = artifact
        self.provenance = provenance
        self.finite = finite
        self.successful = finite
        self.requested_use = requested_use
        self.requested_use_id = requested_use_id
        self.decoder_id = decoder_identity
        self.declared_product_id = declared
        self.product_id = canonical_fingerprint(
            {
                "kind": "constant-external-dark-matter-product",
                "declared_product_id": declared,
                "manifest": manifest.manifest_id,
                "artifact": artifact.artifact_id,
                "provenance": provenance.provenance_id,
                "requested_use": requested_use_id,
                "decoder": decoder_identity,
                "coordinates": coordinates.contract_id,
                "values": array_tree_fingerprint(decoded_host),
            }
        )

    @property
    def layout(self) -> CoordinateLayout:
        return self.coordinates.layout

    def as_theory_vector(self, /) -> TheoryVector:
        return TheoryVector(self.values, self.coordinates.layout, self.product_id)


class ExternalDarkMatterEmulatorProduct(StrictModule, NonTrainableState):
    """Governed external emulator location/scale values on one held-out design."""

    location: Array
    scale: Array
    coordinates: DarkMatterCoordinateContract
    source: ConstantExternalDarkMatterProduct
    finite: Array
    successful: Array
    product_id: str = eqx.field(static=True)

    def __init__(
        self,
        location: ArrayLike,
        scale: ArrayLike,
        artifact_bytes: bytes,
        decoder: Callable[[bytes], ArrayLike],
        coordinates: DarkMatterCoordinateContract,
        manifest: ReferenceArtifactManifest,
        artifact: ScientificArtifactEnvelope,
        provenance: CosmologyProductProvenance,
        /,
        *,
        decoder_id: str,
        product_id: str,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ):
        location_ = jnp.asarray(location, dtype=jnp.float64).reshape((-1,))
        scale_ = jnp.asarray(scale, dtype=location_.dtype).reshape((-1,))
        if location_.shape != scale_.shape:
            raise ValueError("External emulator location and scale must align.")
        if not isinstance(coordinates, DarkMatterCoordinateContract):
            raise TypeError("coordinates must be DarkMatterCoordinateContract.")
        if location_.shape != (coordinates.layout.size,):
            raise ValueError("External emulator values must match their coordinates.")
        combined_coordinates = DarkMatterCoordinateContract(
            CoordinateLayout(
                tuple(f"location:{label}" for label in coordinates.layout.labels)
                + tuple(f"scale:{label}" for label in coordinates.layout.labels)
            ),
            (*coordinates.unit_ids, *coordinates.unit_ids),
        )
        source = ConstantExternalDarkMatterProduct(
            jnp.concatenate((location_, scale_)),
            artifact_bytes,
            decoder,
            manifest,
            artifact,
            provenance,
            combined_coordinates,
            decoder_id=decoder_id,
            product_id=product_id,
            commercial_use=commercial_use,
            redistribution=redistribution,
            training_use=training_use,
            export=export,
        )
        size = location_.size
        location_constant = source.values[:size]
        scale_constant = source.values[size:]
        finite = (
            source.finite
            & jnp.all(jnp.isfinite(location_constant))
            & jnp.all(jnp.isfinite(scale_constant))
            & jnp.all(scale_constant > 0.0)
        )
        self.location = location_constant
        self.scale = scale_constant
        self.coordinates = coordinates
        self.source = source
        self.finite = finite
        self.successful = finite & source.successful
        self.product_id = canonical_fingerprint(
            {
                "kind": "external-dark-matter-emulator-product",
                "source": source.product_id,
                "coordinates": coordinates.contract_id,
            }
        )

    @property
    def layout(self) -> CoordinateLayout:
        return self.coordinates.layout


class DarkMatterErrorBudget(StrictModule, NonTrainableState):
    """Independent standard-error components retained separately and in quadrature."""

    component_standard_errors: Array
    total_standard_error: Array
    coordinates: DarkMatterCoordinateContract
    component_names: tuple[str, ...] = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    finite: Array
    successful: Array
    budget_id: str = eqx.field(static=True)

    def __init__(
        self,
        components: Mapping[str, ArrayLike],
        coordinates: DarkMatterCoordinateContract,
        /,
        *,
        source_ids: Sequence[str],
    ):
        if not isinstance(components, Mapping) or not components:
            raise TypeError("components must be a nonempty mapping of standard errors.")
        if not isinstance(coordinates, DarkMatterCoordinateContract):
            raise TypeError("coordinates must be DarkMatterCoordinateContract.")
        names = tuple(sorted(_identifier(name, "error component") for name in components))
        if len(set(names)) != len(names):
            raise ValueError("Error-budget component names must be unique.")
        arrays = tuple(
            jax.lax.stop_gradient(
                jnp.asarray(components[name], dtype=jnp.float64).reshape((-1,))
            )
            for name in names
        )
        if (
            not arrays
            or any(value.shape != arrays[0].shape for value in arrays)
            or arrays[0].shape != (coordinates.layout.size,)
        ):
            raise ValueError(
                "Error-budget components must align with their coordinate layout."
            )
        sources = tuple(_identifier(value, "source_id") for value in source_ids)
        if not sources:
            raise ValueError("An error budget requires at least one source ID.")
        stacked = jnp.stack(arrays)
        total = jnp.sqrt(jnp.sum(stacked * stacked, axis=0))
        finite = jnp.all(jnp.isfinite(stacked)) & jnp.all(stacked >= 0.0)
        successful = finite & jnp.all(total > 0.0)
        self.component_standard_errors = stacked
        self.total_standard_error = total
        self.coordinates = coordinates
        self.component_names = names
        self.source_ids = sources
        self.finite = finite
        self.successful = successful
        self.budget_id = canonical_fingerprint(
            {
                "kind": "dark-matter-error-budget",
                "component_names": list(names),
                "source_ids": list(sources),
                "coordinates": coordinates.contract_id,
                "component_standard_errors": array_tree_fingerprint(stacked),
            }
        )

    @property
    def layout(self) -> CoordinateLayout:
        return self.coordinates.layout


class DarkMatterDiscrepancyProduct(StrictModule):
    prediction: Array
    target: Array
    residual: Array
    standardized_residual: Array
    chi_square: Array
    reduced_chi_square: Array
    error_budget: DarkMatterErrorBudget
    covariance: CholeskyCovarianceAction
    spectral: SpectralFieldDiscrepancyResult | None
    provenance: CosmologyProductProvenance
    finite: Array
    successful: Array
    prediction_product_id: str = eqx.field(static=True)
    target_product_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)

    def as_theory_vector(self, /) -> TheoryVector:
        from ...observation import CoordinateLayout

        labels = tuple(
            f"standardized-discrepancy:{index}" for index in range(self.residual.size)
        )
        return TheoryVector(
            self.standardized_residual,
            CoordinateLayout(labels),
            self.product_id,
        )


class DarkMatterDiscrepancyPlan(StrictModule, NonTrainableState):
    """Vector discrepancy plus optional native field-spectrum decomposition."""

    observation: LinearObservationPlan | None
    spectral: SpectralFieldDiscrepancyPlan | None
    degrees_of_freedom: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        degrees_of_freedom: int,
        observation: LinearObservationPlan | None = None,
        spectral: SpectralFieldDiscrepancyPlan | None = None,
    ):
        degrees = _positive_integer(degrees_of_freedom, "degrees_of_freedom")
        if observation is not None and not isinstance(observation, LinearObservationPlan):
            raise TypeError("observation must be LinearObservationPlan or None.")
        if spectral is not None and not isinstance(
            spectral, SpectralFieldDiscrepancyPlan
        ):
            raise TypeError("spectral must be SpectralFieldDiscrepancyPlan or None.")
        self.observation = observation
        self.spectral = spectral
        self.degrees_of_freedom = degrees
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dark-matter-discrepancy-plan",
                "degrees_of_freedom": degrees,
                "observation": "none" if observation is None else observation.plan_id,
                "spectral": "none" if spectral is None else spectral.plan_id,
            }
        )

    def evaluate(
        self,
        prediction: TheoryVector,
        target: TheoryVector | ConstantExternalDarkMatterProduct,
        error_budget: DarkMatterErrorBudget,
        covariance: CholeskyCovarianceAction,
        provenance: CosmologyProductProvenance,
        /,
        *,
        prediction_coordinates: DarkMatterCoordinateContract,
        target_coordinates: DarkMatterCoordinateContract | None = None,
        prediction_source: ConstantExternalDarkMatterProduct | None = None,
        predicted_field: ArrayLike | None = None,
        target_field: ArrayLike | None = None,
    ) -> DarkMatterDiscrepancyProduct:
        if not isinstance(prediction, TheoryVector):
            raise TypeError("prediction must be TheoryVector.")
        if not isinstance(error_budget, DarkMatterErrorBudget):
            raise TypeError("error_budget must be DarkMatterErrorBudget.")
        if not isinstance(provenance, CosmologyProductProvenance):
            raise TypeError("provenance must be CosmologyProductProvenance.")
        if not isinstance(covariance, CholeskyCovarianceAction):
            raise TypeError(
                "covariance must be CholeskyCovarianceAction so marginal and "
                "correlated discrepancy scales remain jointly auditable."
            )
        if not isinstance(prediction_coordinates, DarkMatterCoordinateContract):
            raise TypeError(
                "prediction_coordinates must be DarkMatterCoordinateContract."
            )
        if provenance.source_kind == "external":
            if not isinstance(prediction_source, ConstantExternalDarkMatterProduct):
                raise ValueError(
                    "External discrepancy predictions require a governed ConstantExternalDarkMatterProduct source."
                )
            if (
                prediction_source.provenance.provenance_id != provenance.provenance_id
                or prediction.product_id != prediction_source.product_id
            ):
                raise ValueError(
                    "External discrepancy prediction lineage is inconsistent."
                )
            prediction_parents = (prediction_source.product_id,)
        else:
            if prediction_source is not None:
                raise ValueError(
                    "prediction_source is reserved for external predictions."
                )
            prediction_parents = (prediction.product_id,)
        predicted = (
            self.observation.apply(prediction)
            if self.observation is not None
            else prediction
        )
        if isinstance(target, TheoryVector):
            if not isinstance(target_coordinates, DarkMatterCoordinateContract):
                raise ValueError(
                    "Native target vectors require an explicit coordinate contract."
                )
            target_values = target.values
            target_id = target.product_id
            target_contract = target_coordinates
            target_differentiation = provenance.differentiation
        elif isinstance(target, ConstantExternalDarkMatterProduct):
            if target_coordinates is not None:
                raise ValueError(
                    "External targets already own their coordinate contract."
                )
            target_values = target.values
            target_id = target.product_id
            target_contract = target.coordinates
            target_differentiation = target.provenance.differentiation
        else:
            raise TypeError(
                "target must be TheoryVector or ConstantExternalDarkMatterProduct."
            )
        if (
            prediction_coordinates.layout.layout_id != predicted.layout.layout_id
            or target_contract.contract_id != prediction_coordinates.contract_id
            or error_budget.coordinates.contract_id != prediction_coordinates.contract_id
        ):
            raise ValueError(
                "Prediction, target, and error-budget coordinate/unit contracts disagree."
            )
        if covariance.layout.layout_id != prediction_coordinates.layout.layout_id:
            raise ValueError("Covariance and discrepancy coordinate layouts disagree.")
        if target_values.shape != predicted.values.shape:
            raise ValueError("Prediction and target vectors must align.")
        if jnp.issubdtype(predicted.values.dtype, jnp.complexfloating) or jnp.issubdtype(
            target_values.dtype, jnp.complexfloating
        ):
            raise TypeError("Dark-matter discrepancy vectors must be explicitly real.")
        if error_budget.total_standard_error.shape != predicted.values.shape:
            raise ValueError("Error budget and discrepancy vector must align.")
        marginal_standard_error = jnp.sqrt(jnp.sum(covariance.lower_cholesky**2, axis=1))
        covariance_consistent = jnp.all(
            jnp.abs(marginal_standard_error - error_budget.total_standard_error)
            <= 64.0
            * jnp.finfo(marginal_standard_error.dtype).eps
            * jnp.maximum(marginal_standard_error, 1.0)
        )
        residual = eqx.error_if(
            predicted.values - target_values,
            ~covariance_consistent,
            "Covariance marginals and discrepancy error budget disagree.",
        )
        standardized = covariance.whiten(residual)
        chi_square = jnp.sum(standardized * standardized)
        reduced = chi_square / self.degrees_of_freedom
        if self.spectral is None:
            if predicted_field is not None or target_field is not None:
                raise ValueError("Field inputs require a spectral discrepancy plan.")
            spectral_product = None
            spectral_finite = jnp.asarray(True)
            spectral_successful = jnp.asarray(True)
        else:
            if predicted_field is None or target_field is None:
                raise ValueError("Spectral discrepancy requires both fields.")
            spectral_product = self.spectral.evaluate(
                predicted_field,
                target_field,
                predicted.product_id,
                target_id,
            )
            spectral_finite = spectral_product.finite
            spectral_successful = spectral_product.successful
        finite = (
            error_budget.finite
            & jnp.all(jnp.isfinite(residual))
            & jnp.all(jnp.isfinite(standardized))
            & jnp.isfinite(chi_square)
            & spectral_finite
        )
        successful = finite & error_budget.successful & spectral_successful
        parent_product_ids = tuple(
            dict.fromkeys(
                (
                    *prediction_parents,
                    predicted.product_id,
                    target_id,
                    error_budget.budget_id,
                    covariance.action_id,
                )
            )
        )
        product_provenance = CosmologyProductProvenance(
            producer=("phydrax.applications.cosmology.DarkMatterDiscrepancyPlan"),
            producer_version="native",
            model_form_id=provenance.model_form_id,
            request_id=provenance.request_id,
            numerical_policy_id=self.plan_id,
            physics_policy_id=provenance.physics_policy_id,
            scale_id=provenance.scale_id,
            source_kind="native",
            differentiation=provenance.differentiation.meet(target_differentiation),
            parent_product_ids=parent_product_ids,
        )
        product_id = canonical_fingerprint(
            {
                "kind": "dark-matter-discrepancy-product",
                "plan": self.plan_id,
                "prediction": predicted.product_id,
                "target": target_id,
                "budget": error_budget.budget_id,
                "covariance": covariance.action_id,
                "provenance": product_provenance.provenance_id,
            }
        )
        return DarkMatterDiscrepancyProduct(
            predicted.values,
            target_values,
            residual,
            standardized,
            chi_square,
            reduced,
            error_budget,
            covariance,
            spectral_product,
            product_provenance,
            finite,
            successful,
            predicted.product_id,
            target_id,
            product_id,
        )


class DarkMatterEmulatorCalibrationProduct(StrictModule, NonTrainableState):
    """Held-out Gaussian scale calibration with governed source lineage."""

    location: Array
    raw_scale: Array
    calibrated_scale: Array
    standardized_residual: Array
    calibration_effective_sample_size: Array
    calibrator: GaussianScaleCalibrator
    diagnostics: IntervalCalibrationDiagnostics
    reference: ConstantExternalDarkMatterProduct
    emulator_source: ExternalDarkMatterEmulatorProduct | None
    provenance: CosmologyProductProvenance
    finite: Array
    successful: Array
    emulator_product_id: str = eqx.field(static=True)
    split_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)


class DarkMatterEmulatorCalibrationPlan(StrictModule, NonTrainableState):
    nominal_coverage: float = eqx.field(static=True)
    maximum_coverage_gap: float = eqx.field(static=True)
    minimum_calibration_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        nominal_coverage: float = 0.9,
        maximum_coverage_gap: float = 0.1,
        minimum_calibration_count: int = 8,
    ):
        coverage = float(nominal_coverage)
        gap = float(maximum_coverage_gap)
        count = _positive_integer(minimum_calibration_count, "minimum_calibration_count")
        if not 0.0 < coverage < 1.0 or not 0.0 <= gap < 1.0:
            raise ValueError("Emulator calibration coverage policy is invalid.")
        self.nominal_coverage = coverage
        self.maximum_coverage_gap = gap
        self.minimum_calibration_count = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dark-matter-emulator-calibration-plan",
                "nominal_coverage": coverage,
                "maximum_coverage_gap": gap,
                "minimum_calibration_count": count,
            }
        )

    def fit(
        self,
        location: ArrayLike,
        raw_scale: ArrayLike,
        reference: ConstantExternalDarkMatterProduct,
        provenance: CosmologyProductProvenance,
        /,
        *,
        emulator_coordinates: DarkMatterCoordinateContract,
        emulator_product_id: str,
        split_id: str,
        mask: ArrayLike | None = None,
        weights: ArrayLike | None = None,
    ) -> DarkMatterEmulatorCalibrationProduct:
        if not isinstance(provenance, CosmologyProductProvenance):
            raise TypeError("provenance must be CosmologyProductProvenance.")
        if provenance.source_kind == "external":
            raise ValueError(
                "External emulator calibration requires fit_external with a governed source."
            )
        return self._fit(
            location,
            raw_scale,
            reference,
            provenance,
            _identifier(emulator_product_id, "emulator_product_id"),
            emulator_coordinates,
            split_id=split_id,
            mask=mask,
            weights=weights,
            emulator_source=None,
        )

    def fit_external(
        self,
        emulator: ExternalDarkMatterEmulatorProduct,
        reference: ConstantExternalDarkMatterProduct,
        /,
        *,
        split_id: str,
        mask: ArrayLike | None = None,
        weights: ArrayLike | None = None,
    ) -> DarkMatterEmulatorCalibrationProduct:
        if not isinstance(emulator, ExternalDarkMatterEmulatorProduct):
            raise TypeError("emulator must be ExternalDarkMatterEmulatorProduct.")
        if not isinstance(reference, ConstantExternalDarkMatterProduct):
            raise TypeError("reference must be ConstantExternalDarkMatterProduct.")
        if emulator.coordinates.contract_id != reference.coordinates.contract_id:
            raise ValueError(
                "External emulator and calibration reference coordinate/unit contracts disagree."
            )
        if not bool(emulator.successful):
            raise ValueError("External emulator source evidence is unsuccessful.")
        return self._fit(
            emulator.location,
            emulator.scale,
            reference,
            emulator.source.provenance,
            emulator.product_id,
            emulator.coordinates,
            split_id=split_id,
            mask=mask,
            weights=weights,
            emulator_source=emulator,
        )

    def _fit(
        self,
        location: ArrayLike,
        raw_scale: ArrayLike,
        reference: ConstantExternalDarkMatterProduct,
        provenance: CosmologyProductProvenance,
        emulator_product_id: str,
        emulator_coordinates: DarkMatterCoordinateContract,
        /,
        *,
        split_id: str,
        mask: ArrayLike | None,
        weights: ArrayLike | None,
        emulator_source: ExternalDarkMatterEmulatorProduct | None,
    ) -> DarkMatterEmulatorCalibrationProduct:
        if not isinstance(reference, ConstantExternalDarkMatterProduct):
            raise TypeError("reference must be ConstantExternalDarkMatterProduct.")
        if not isinstance(provenance, CosmologyProductProvenance):
            raise TypeError("provenance must be CosmologyProductProvenance.")
        if not isinstance(emulator_coordinates, DarkMatterCoordinateContract):
            raise TypeError("emulator_coordinates must be DarkMatterCoordinateContract.")
        if emulator_coordinates.contract_id != reference.coordinates.contract_id:
            raise ValueError(
                "Emulator and calibration reference coordinate/unit contracts disagree."
            )
        external_source = isinstance(emulator_source, ExternalDarkMatterEmulatorProduct)
        if (provenance.source_kind == "external") != external_source:
            raise ValueError("Emulator provenance/source admission is inconsistent.")
        if external_source and (
            emulator_source.source.provenance.provenance_id != provenance.provenance_id
        ):
            raise ValueError("External emulator provenance identity changed.")
        split = _identifier(split_id, "split_id")
        center = jnp.asarray(location, dtype=jnp.float64).reshape((-1,))
        scale = jnp.asarray(raw_scale, dtype=center.dtype).reshape((-1,))
        if center.shape != reference.values.shape or scale.shape != center.shape:
            raise ValueError("Emulator location, scale, and reference must align.")
        active = (
            jnp.ones(center.shape, dtype=jnp.bool_)
            if mask is None
            else jnp.asarray(mask, dtype=jnp.bool_).reshape((-1,))
        )
        weight_array = (
            jnp.ones(center.shape, dtype=center.dtype)
            if weights is None
            else jnp.asarray(weights, dtype=center.dtype).reshape((-1,))
        )
        if active.shape != center.shape or weight_array.shape != center.shape:
            raise ValueError("Calibration mask/weights must align with outputs.")
        invalid_weight = active & ((~jnp.isfinite(weight_array)) | (weight_array < 0.0))
        if bool(jnp.any(invalid_weight)):
            raise ValueError("Active calibration weights must be finite and nonnegative.")
        active = active & (weight_array > 0.0)
        effective_weight = jnp.where(active, weight_array, 0.0)
        weight_sum = jnp.sum(effective_weight)
        effective_count = weight_sum**2 / jnp.maximum(
            jnp.sum(effective_weight * effective_weight),
            jnp.finfo(center.dtype).tiny,
        )
        if float(effective_count) < self.minimum_calibration_count:
            raise ValueError(
                "Held-out emulator calibration effective sample size is too small."
            )
        calibrator = GaussianScaleCalibrator.fit(
            center,
            scale,
            reference.values,
            mask=active,
            weights=weight_array,
        )
        calibrated = calibrator.transform(scale)
        quantile = jsp.special.ndtri(
            jnp.asarray(
                0.5 * (1.0 + self.nominal_coverage),
                dtype=center.dtype,
            )
        )
        diagnostics = interval_calibration_diagnostics(
            center - quantile * calibrated,
            center + quantile * calibrated,
            reference.values,
            nominal_coverage=self.nominal_coverage,
            mask=active,
            weights=weight_array,
        )
        standardized = (reference.values - center) / calibrated
        emulator_successful = (
            emulator_source.successful if external_source else jnp.asarray(True)
        )
        finite = (
            reference.finite
            & emulator_successful
            & diagnostics.valid
            & jnp.isfinite(effective_count)
            & jnp.all(jnp.where(active, jnp.isfinite(center), True))
            & jnp.all(
                jnp.where(
                    active,
                    jnp.isfinite(scale) & (scale > 0.0),
                    True,
                )
            )
            & jnp.all(jnp.where(active, jnp.isfinite(standardized), True))
        )
        successful = (
            finite
            & reference.successful
            & emulator_successful
            & (diagnostics.absolute_coverage_gap <= self.maximum_coverage_gap)
        )
        product_provenance = CosmologyProductProvenance(
            producer=("phydrax.applications.cosmology.DarkMatterEmulatorCalibrationPlan"),
            producer_version="native",
            model_form_id=provenance.model_form_id,
            request_id=provenance.request_id,
            numerical_policy_id=self.plan_id,
            physics_policy_id=provenance.physics_policy_id,
            scale_id=provenance.scale_id,
            source_kind="native",
            differentiation=provenance.differentiation.meet(
                reference.provenance.differentiation
            ),
            parent_product_ids=(
                emulator_product_id,
                reference.product_id,
            ),
        )
        product_id = canonical_fingerprint(
            {
                "kind": "dark-matter-emulator-calibration-product",
                "plan": self.plan_id,
                "split": split,
                "reference": reference.product_id,
                "emulator": emulator_product_id,
                "coordinates": emulator_coordinates.contract_id,
                "location": array_tree_fingerprint(np.asarray(center)),
                "raw_scale": array_tree_fingerprint(np.asarray(scale)),
                "active": array_tree_fingerprint(np.asarray(active)),
                "weights": array_tree_fingerprint(np.asarray(weight_array)),
                "provenance": product_provenance.provenance_id,
            }
        )
        return DarkMatterEmulatorCalibrationProduct(
            center,
            scale,
            calibrated,
            standardized,
            effective_count,
            calibrator,
            diagnostics,
            reference,
            emulator_source,
            product_provenance,
            finite,
            successful,
            emulator_product_id,
            split,
            product_id,
        )


__all__ = [
    "ConstantExternalDarkMatterProduct",
    "DarkMatterCoordinateContract",
    "DarkMatterDiscrepancyPlan",
    "DarkMatterDiscrepancyProduct",
    "DarkMatterEmulatorCalibrationPlan",
    "DarkMatterEmulatorCalibrationProduct",
    "DarkMatterErrorBudget",
    "DarkMatterInferenceEvaluation",
    "FixedTapeStochasticEvaluation",
    "ExternalDarkMatterEmulatorProduct",
    "FixedTapeStochasticSensitivityPlan",
    "SmoothDarkMatterKind",
    "SmoothFixedGridDarkMatterInferencePlan",
    "SmoothFixedGridSensitivityProduct",
    "StochasticSensitivityEvidence",
]
