#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-branch uncertainty composition for relativistic evolution."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..compact_objects._inverse import (
    FixedBranchInverseAdapter,
    FixedBranchModelEvaluation,
    FixedBranchSensitivityEvidence,
)


_MULTIFIDELITY_KINDS = (
    "exact-baseline",
    "perturbative-correction",
    "rom-correction",
    "full-simulation-correction",
)


class NumericalErrorRecord(StrictModule, NonTrainableState):
    """Declared componentwise numerical error bound and its qualification state."""

    absolute_error_bound: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    method_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        absolute_error_bound: ArrayLike,
        /,
        *,
        converged: ArrayLike,
        physically_valid: ArrayLike,
        qualified: ArrayLike,
        method_id: str,
        realization_id: str,
        evidence_id: str,
    ):
        bound_host = np.asarray(absolute_error_bound)
        if np.iscomplexobj(bound_host):
            raise TypeError("Numerical error bounds must be real-valued.")
        if bound_host.size == 0 or np.any(np.isnan(bound_host)) or np.any(bound_host < 0):
            raise ValueError(
                "Numerical error bounds must be nonempty, nonnegative, and not NaN."
            )
        bound = jax.lax.stop_gradient(jnp.asarray(bound_host).reshape((-1,)))
        flags = tuple(
            jnp.asarray(value, dtype=bool)
            for value in (converged, physically_valid, qualified)
        )
        if any(value.shape != () for value in flags):
            raise ValueError("Numerical error evidence flags must be scalar.")
        method = _identifier(method_id, "method_id")
        realization = _identifier(realization_id, "realization_id")
        evidence = _identifier(evidence_id, "evidence_id")
        self.absolute_error_bound = bound
        self.finite = jnp.all(jnp.isfinite(bound))
        self.converged, self.physically_valid, self.qualified = flags
        self.method_id = method
        self.realization_id = realization
        self.evidence_id = evidence
        self.record_id = canonical_fingerprint(
            {
                "kind": "nr-numerical-error-record",
                "absolute_error_bound": array_tree_fingerprint(bound),
                "converged": bool(np.asarray(flags[0])),
                "physically_valid": bool(np.asarray(flags[1])),
                "qualified": bool(np.asarray(flags[2])),
                "method_id": method,
                "realization_id": realization,
                "evidence_id": evidence,
            }
        )

    @property
    def authoritative(self) -> Array:
        """Whether this bound is finite and supported for scientific use."""

        return self.finite & self.converged & self.physically_valid & self.qualified


class ModelDiscrepancyRecord(StrictModule, NonTrainableState):
    """Additive bias and low-rank covariance kept separate from numerical error."""

    mean: Array
    covariance_factor: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    model_id: str = eqx.field(static=True)
    calibration_evidence_id: str = eqx.field(static=True)
    validation_evidence_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    discrepancy_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean: ArrayLike,
        covariance_factor: ArrayLike | None = None,
        /,
        *,
        converged: ArrayLike,
        physically_valid: ArrayLike,
        qualified: ArrayLike,
        model_id: str,
        calibration_evidence_id: str,
        validation_evidence_id: str,
        support_id: str,
    ):
        mean_host = np.asarray(mean)
        if np.iscomplexobj(mean_host):
            raise TypeError("Model discrepancy means must be real-valued.")
        mean_ = jax.lax.stop_gradient(jnp.asarray(mean_host).reshape((-1,)))
        if not eqx.is_inexact_array(mean_):
            raise TypeError("Model discrepancy means must be inexact arrays.")
        if covariance_factor is None:
            factor = jnp.zeros((mean_.size, 0), dtype=mean_.dtype)
        else:
            factor_host = np.asarray(covariance_factor)
            if np.iscomplexobj(factor_host):
                raise TypeError("Model discrepancy covariance factors must be real-valued.")
            factor = jax.lax.stop_gradient(
                jnp.asarray(factor_host, dtype=mean_.dtype)
            )
        if mean_.size == 0 or factor.ndim != 2 or factor.shape[0] != mean_.size:
            raise ValueError(
                "Discrepancy factor must have one row per nonempty mean value."
            )
        flags = tuple(
            jnp.asarray(value, dtype=bool)
            for value in (converged, physically_valid, qualified)
        )
        if any(value.shape != () for value in flags):
            raise ValueError("Model discrepancy evidence flags must be scalar.")
        model = _identifier(model_id, "model_id")
        calibration = _identifier(calibration_evidence_id, "calibration_evidence_id")
        validation = _identifier(validation_evidence_id, "validation_evidence_id")
        support = _identifier(support_id, "support_id")
        finite = jnp.all(jnp.isfinite(mean_)) & jnp.all(jnp.isfinite(factor))
        self.mean = mean_
        self.covariance_factor = factor
        self.finite = finite
        self.converged, self.physically_valid, self.qualified = flags
        self.model_id = model
        self.calibration_evidence_id = calibration
        self.validation_evidence_id = validation
        self.support_id = support
        self.discrepancy_id = canonical_fingerprint(
            {
                "kind": "nr-model-discrepancy",
                "mean": array_tree_fingerprint(mean_),
                "covariance_factor": array_tree_fingerprint(factor),
                "converged": bool(np.asarray(flags[0])),
                "physically_valid": bool(np.asarray(flags[1])),
                "qualified": bool(np.asarray(flags[2])),
                "model_id": model,
                "calibration_evidence_id": calibration,
                "validation_evidence_id": validation,
                "support_id": support,
            }
        )

    @property
    def covariance(self) -> Array:
        return ein.contract("ik,jk->ij", self.covariance_factor, self.covariance_factor)

    @property
    def authoritative(self) -> Array:
        return self.finite & self.converged & self.physically_valid & self.qualified


def smooth_grhd_inverse_adapter(
    evaluator: Callable[[Array], FixedBranchModelEvaluation],
    expected_branch_signature: ArrayLike,
    /,
    *,
    parameter_count: int,
    output_count: int,
    realization_id: str,
    branch_id: str,
    adapter_id: str,
    evaluator_semantic_id: str | None = None,
    evaluator_numeric_id: str | None = None,
) -> FixedBranchInverseAdapter:
    """Bind a shock-free GRHD trajectory with a fixed recovery/flux branch."""

    return FixedBranchInverseAdapter(
        evaluator,
        expected_branch_signature,
        parameter_count=parameter_count,
        output_count=output_count,
        model_kind="smooth-grhd",
        realization_id=realization_id,
        branch_id=branch_id,
        adapter_id=adapter_id,
        evaluator_semantic_id=evaluator_semantic_id,
        evaluator_numeric_id=evaluator_numeric_id,
    )


def smooth_nr_inverse_adapter(
    evaluator: Callable[[Array], FixedBranchModelEvaluation],
    expected_branch_signature: ArrayLike,
    /,
    *,
    parameter_count: int,
    output_count: int,
    realization_id: str,
    branch_id: str,
    adapter_id: str,
    evaluator_semantic_id: str | None = None,
    evaluator_numeric_id: str | None = None,
) -> FixedBranchInverseAdapter:
    """Bind fixed-grid, fixed-step NR parameters without topology/event selection."""

    return FixedBranchInverseAdapter(
        evaluator,
        expected_branch_signature,
        parameter_count=parameter_count,
        output_count=output_count,
        model_kind="smooth-numerical-relativity",
        realization_id=realization_id,
        branch_id=branch_id,
        adapter_id=adapter_id,
        evaluator_semantic_id=evaluator_semantic_id,
        evaluator_numeric_id=evaluator_numeric_id,
    )


def z4c_step_model_evaluation(
    result: Any,
    values: ArrayLike,
    /,
    *,
    realization_id: str,
    branch_id: str,
) -> FixedBranchModelEvaluation:
    """Adapt one fixed-grid ``Z4cStepResult`` to the shared inverse contract."""

    return FixedBranchModelEvaluation(
        values,
        jnp.asarray(result.status, dtype=jnp.int32).reshape((-1,)),
        finite=jnp.all(jnp.asarray(result.finite)),
        converged=jnp.all(jnp.asarray(result.converged)),
        physically_valid=jnp.all(jnp.asarray(result.physically_valid)),
        qualified=jnp.all(jnp.asarray(result.qualified)),
        derivative_valid=jnp.all(jnp.asarray(result.derivative_valid)),
        shock_free=True,
        event_free=True,
        topology_fixed=True,
        realization_id=realization_id,
        branch_id=branch_id,
    )


def grhd_model_evaluation(
    evaluation: Any,
    values: ArrayLike,
    branch_signature: ArrayLike,
    /,
    *,
    shock_free: ArrayLike,
    realization_id: str,
    branch_id: str,
) -> FixedBranchModelEvaluation:
    """Adapt smooth Valencia GRHD output with independent shock evidence."""

    return FixedBranchModelEvaluation(
        values,
        branch_signature,
        finite=jnp.all(jnp.asarray(evaluation.finite)),
        converged=jnp.all(jnp.asarray(evaluation.converged)),
        physically_valid=jnp.all(jnp.asarray(evaluation.physically_valid)),
        qualified=jnp.all(jnp.asarray(evaluation.qualified)),
        derivative_valid=jnp.all(jnp.asarray(evaluation.derivative_valid)),
        shock_free=jnp.all(jnp.asarray(shock_free)),
        event_free=True,
        topology_fixed=True,
        realization_id=realization_id,
        branch_id=branch_id,
    )


def grhd_recovery_model_evaluation(
    recovery: Any,
    values: ArrayLike,
    /,
    *,
    shock_free: ArrayLike,
    realization_id: str,
    branch_id: str,
) -> FixedBranchModelEvaluation:
    """Adapt GRHD primitive recovery while requiring separate shock-free evidence."""

    signature = jnp.concatenate(
        (
            jnp.asarray(recovery.status, dtype=jnp.int32).reshape((-1,)),
            jnp.asarray(
                recovery.candidates.selected_branch, dtype=jnp.int32
            ).reshape((-1,)),
            jnp.asarray(recovery.atmosphere.applied, dtype=jnp.int32).reshape((-1,)),
        )
    )
    return FixedBranchModelEvaluation(
        values,
        signature,
        finite=jnp.all(jnp.asarray(recovery.finite)),
        converged=jnp.all(jnp.asarray(recovery.converged)),
        physically_valid=jnp.all(jnp.asarray(recovery.physically_valid)),
        qualified=jnp.all(jnp.asarray(recovery.qualified)),
        derivative_valid=jnp.all(jnp.asarray(recovery.derivative_valid)),
        shock_free=jnp.all(jnp.asarray(shock_free)),
        event_free=True,
        topology_fixed=True,
        realization_id=realization_id,
        branch_id=branch_id,
    )


class RelativisticMultifidelityEvaluation(StrictModule):
    """Exact baseline plus explicitly declared perturbative/ROM/full corrections."""

    term_values: Array
    native_value: Array
    discrepancy_mean: Array
    value: Array
    numerical_error_bound: Array
    discrepancy_covariance: Array
    term_finite: Array
    term_converged: Array
    term_physically_valid: Array
    term_qualified: Array
    term_derivative_valid: Array
    term_shock_free: Array
    term_event_free: Array
    term_topology_fixed: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    numerical_error_available: Array
    model_discrepancy_available: Array
    uncertainty_qualified: Array
    branch_signature: Array
    plan_id: str = eqx.field(static=True)


class RelativisticMultifidelityPlan(StrictModule, NonTrainableState):
    """Four-level additive composition with no implicit fidelity substitution.

    The exact component is the sole baseline. The perturbative, ROM and
    full-simulation components are corrections supplied as such by the caller; the
    plan never subtracts, chooses, or replaces a level. Unknown uncertainty is
    represented by unavailable evidence and an infinite bound, never by zero.
    """

    components: tuple[
        FixedBranchInverseAdapter,
        FixedBranchInverseAdapter,
        FixedBranchInverseAdapter,
        FixedBranchInverseAdapter,
    ]
    numerical_errors: tuple[
        NumericalErrorRecord | None,
        NumericalErrorRecord | None,
        NumericalErrorRecord | None,
        NumericalErrorRecord | None,
    ]
    model_discrepancy: ModelDiscrepancyRecord | None
    parameter_count: int = eqx.field(static=True)
    output_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        exact: FixedBranchInverseAdapter,
        perturbative_correction: FixedBranchInverseAdapter,
        rom_correction: FixedBranchInverseAdapter,
        full_simulation_correction: FixedBranchInverseAdapter,
        /,
        *,
        numerical_errors: tuple[
            NumericalErrorRecord | None,
            NumericalErrorRecord | None,
            NumericalErrorRecord | None,
            NumericalErrorRecord | None,
        ] = (None, None, None, None),
        model_discrepancy: ModelDiscrepancyRecord | None = None,
        plan_id: str,
    ):
        components = (
            exact,
            perturbative_correction,
            rom_correction,
            full_simulation_correction,
        )
        if any(not isinstance(item, FixedBranchInverseAdapter) for item in components):
            raise TypeError("Every multifidelity component must be a fixed-branch adapter.")
        if tuple(item.model_kind for item in components) != _MULTIFIDELITY_KINDS:
            raise ValueError(
                "Multifidelity components must be exact baseline, perturbative, ROM, "
                "and full-simulation corrections in that order."
            )
        if len({item.parameter_count for item in components}) != 1 or len(
            {item.output_count for item in components}
        ) != 1:
            raise ValueError("Multifidelity components must share parameter/output shape.")
        errors = tuple(numerical_errors)
        if len(errors) != 4 or any(
            item is not None and not isinstance(item, NumericalErrorRecord)
            for item in errors
        ):
            raise TypeError("numerical_errors must contain four records or None values.")
        output_count = exact.output_count
        for component, error in zip(components, errors, strict=True):
            if error is not None and (
                error.absolute_error_bound.shape != (output_count,)
                or error.realization_id != component.realization_id
            ):
                raise ValueError(
                    "Numerical error shape and realization must match its component."
                )
        if model_discrepancy is not None and not isinstance(
            model_discrepancy, ModelDiscrepancyRecord
        ):
            raise TypeError("model_discrepancy must be a ModelDiscrepancyRecord or None.")
        if model_discrepancy is not None and model_discrepancy.mean.shape != (
            output_count,
        ):
            raise ValueError("Model discrepancy must match the multifidelity output.")
        declared = _identifier(plan_id, "plan_id")
        self.components = components
        self.numerical_errors = errors
        self.model_discrepancy = model_discrepancy
        self.parameter_count = exact.parameter_count
        self.output_count = output_count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "relativistic-multifidelity-composition",
                "declared_id": declared,
                "components": [item.adapter_id for item in components],
                "numerical_errors": [
                    None if item is None else item.record_id for item in errors
                ],
                "model_discrepancy": None
                if model_discrepancy is None
                else model_discrepancy.discrepancy_id,
            }
        )

    def evaluate(self, parameters: ArrayLike, /) -> RelativisticMultifidelityEvaluation:
        evaluations = tuple(item.evaluate(parameters) for item in self.components)
        term_values = jnp.stack([item.values for item in evaluations])
        native = jnp.sum(term_values, axis=0)
        discrepancy_available = jnp.asarray(self.model_discrepancy is not None)
        if self.model_discrepancy is None:
            discrepancy_mean = jnp.zeros_like(native)
            discrepancy_covariance = jnp.full(
                (self.output_count, self.output_count),
                jnp.nan,
                dtype=native.real.dtype,
            )
            discrepancy_authoritative = jnp.asarray(False)
        else:
            discrepancy_mean = self.model_discrepancy.mean.astype(native.dtype)
            discrepancy_covariance = self.model_discrepancy.covariance
            discrepancy_authoritative = self.model_discrepancy.authoritative
        value = native + discrepancy_mean

        error_available = jnp.asarray(all(item is not None for item in self.numerical_errors))
        error_bounds = [
            jnp.full((self.output_count,), jnp.inf, dtype=native.real.dtype)
            if item is None
            else item.absolute_error_bound.astype(native.real.dtype)
            for item in self.numerical_errors
        ]
        total_error = jnp.sum(jnp.stack(error_bounds), axis=0)
        error_authoritative = jnp.all(
            jnp.stack(
                [
                    jnp.asarray(False) if item is None else item.authoritative
                    for item in self.numerical_errors
                ]
            )
        )

        term_finite = jnp.stack([item.finite for item in evaluations])
        term_converged = jnp.stack([item.converged for item in evaluations])
        term_physical = jnp.stack([item.physically_valid for item in evaluations])
        term_qualified = jnp.stack([item.qualified for item in evaluations])
        term_derivative = jnp.stack(
            [item.sensitivity_eligible for item in evaluations]
        )
        term_shock_free = jnp.stack([item.shock_free for item in evaluations])
        term_event_free = jnp.stack([item.event_free for item in evaluations])
        term_topology_fixed = jnp.stack([item.topology_fixed for item in evaluations])
        finite = jnp.all(term_finite) & jnp.all(jnp.isfinite(value))
        converged = finite & jnp.all(term_converged)
        physical = converged & jnp.all(term_physical)
        qualified = physical & jnp.all(term_qualified)
        derivative = qualified & jnp.all(term_derivative)
        uncertainty_qualified = (
            qualified
            & error_available
            & error_authoritative
            & discrepancy_available
            & discrepancy_authoritative
        )
        signature = jnp.concatenate(
            [item.branch_signature for item in evaluations], axis=0
        )
        return RelativisticMultifidelityEvaluation(
            term_values,
            native,
            discrepancy_mean,
            value,
            total_error,
            discrepancy_covariance,
            term_finite,
            term_converged,
            term_physical,
            term_qualified,
            term_derivative,
            term_shock_free,
            term_event_free,
            term_topology_fixed,
            finite,
            converged,
            physical,
            qualified,
            derivative,
            error_available,
            discrepancy_available,
            uncertainty_qualified,
            signature,
            self.plan_id,
        )

    def _fixed_evaluation(self, parameters: ArrayLike, /) -> FixedBranchModelEvaluation:
        result = self.evaluate(parameters)
        return FixedBranchModelEvaluation(
            result.value,
            result.branch_signature,
            finite=result.finite,
            converged=result.converged,
            physically_valid=result.physically_valid,
            qualified=result.qualified,
            derivative_valid=result.derivative_valid,
            shock_free=jnp.all(result.term_shock_free),
            event_free=jnp.all(result.term_event_free),
            topology_fixed=jnp.all(result.term_topology_fixed),
            realization_id=self.plan_id,
            branch_id="exact+perturbative+rom+full-simulation",
        )

    def sensitivity(
        self,
        parameters: ArrayLike,
        direction: ArrayLike,
        /,
        *,
        cotangent: ArrayLike | None = None,
        epsilon: float = 1.0e-4,
    ) -> FixedBranchSensitivityEvidence:
        """Audit the additive composition on all four unchanged branches."""

        signature = jnp.concatenate(
            [item.expected_branch_signature for item in self.components], axis=0
        )
        adapter = FixedBranchInverseAdapter(
            self._fixed_evaluation,
            signature,
            parameter_count=self.parameter_count,
            output_count=self.output_count,
            model_kind="relativistic-multifidelity",
            realization_id=self.plan_id,
            branch_id="exact+perturbative+rom+full-simulation",
            adapter_id=self.plan_id,
            evaluator_semantic_id=self.plan_id,
            evaluator_numeric_id=self.plan_id,
        )
        return adapter.sensitivity(
            parameters,
            direction,
            cotangent=cotangent,
            epsilon=epsilon,
        )


class LearnedClosureCandidate(StrictModule, NonTrainableState):
    """A frozen learned additive correction; it has no native-replacement API."""

    model: Callable[[Any], ArrayLike]
    derivative_supported: bool = eqx.field(static=True)
    native_model_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    training_realization_id: str = eqx.field(static=True)
    differentiation_evidence_id: str | None = eqx.field(static=True)
    role: str = eqx.field(static=True, default="additive-correction-only")
    candidate_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: Callable[[Any], ArrayLike],
        /,
        *,
        native_model_id: str,
        model_id: str,
        training_realization_id: str,
        derivative_supported: bool = False,
        differentiation_evidence_id: str | None = None,
    ):
        if not callable(model):
            raise TypeError("Learned closure model must be callable.")
        derivative = bool(derivative_supported)
        if derivative != (differentiation_evidence_id is not None):
            raise ValueError(
                "Derivative support and differentiation evidence must be supplied together."
            )
        native = _identifier(native_model_id, "native_model_id")
        model_identity = _identifier(model_id, "model_id")
        training = _identifier(training_realization_id, "training_realization_id")
        differentiation = (
            None
            if differentiation_evidence_id is None
            else _identifier(
                differentiation_evidence_id, "differentiation_evidence_id"
            )
        )
        self.model = model
        self.derivative_supported = derivative
        self.native_model_id = native
        self.model_id = model_identity
        self.training_realization_id = training
        self.differentiation_evidence_id = differentiation
        self.candidate_id = canonical_fingerprint(
            {
                "kind": "nr-learned-additive-closure-candidate",
                "native_model_id": native,
                "model_id": model_identity,
                "training_realization_id": training,
                "derivative_supported": derivative,
                "differentiation_evidence_id": differentiation,
                "model_arrays": array_tree_fingerprint(model),
                "role": "additive-correction-only",
            }
        )


class LearnedClosureAdmissionEvidence(StrictModule, NonTrainableState):
    """Independent conservation, admissibility, and requested-use rights evidence."""

    conservation_residual: Array
    admissibility_margin: Array
    conservation_tolerance: float = eqx.field(static=True)
    finite: bool = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    physically_valid: bool = eqx.field(static=True)
    qualified: bool = eqx.field(static=True)
    conservation_passed: bool = eqx.field(static=True)
    admissibility_passed: bool = eqx.field(static=True)
    rights_authorized: bool = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)
    conservation_evidence_id: str = eqx.field(static=True)
    admissibility_evidence_id: str = eqx.field(static=True)
    rights_evidence_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        conservation_residual: ArrayLike,
        admissibility_margin: ArrayLike,
        /,
        *,
        conservation_tolerance: float,
        converged: bool,
        physically_valid: bool,
        qualified: bool,
        rights_authorized: bool,
        candidate_id: str,
        conservation_evidence_id: str,
        admissibility_evidence_id: str,
        rights_evidence_id: str,
    ):
        residual = jax.lax.stop_gradient(
            jnp.asarray(conservation_residual).reshape((-1,))
        )
        margin = jax.lax.stop_gradient(jnp.asarray(admissibility_margin).reshape((-1,)))
        if residual.size == 0 or margin.size == 0:
            raise ValueError("Learned closure evidence arrays must be nonempty.")
        if jnp.issubdtype(residual.dtype, jnp.complexfloating) or jnp.issubdtype(
            margin.dtype, jnp.complexfloating
        ):
            raise TypeError("Learned closure evidence must be real-valued.")
        tolerance = float(conservation_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("conservation_tolerance must be finite and nonnegative.")
        candidate = _identifier(candidate_id, "candidate_id")
        conservation_id = _identifier(
            conservation_evidence_id, "conservation_evidence_id"
        )
        admissibility_id = _identifier(
            admissibility_evidence_id, "admissibility_evidence_id"
        )
        rights_id = _identifier(rights_evidence_id, "rights_evidence_id")
        finite = bool(
            np.all(np.isfinite(np.asarray(residual)))
            and np.all(np.isfinite(np.asarray(margin)))
        )
        conservation_passed = bool(
            finite and np.max(np.abs(np.asarray(residual))) <= tolerance
        )
        admissibility_passed = bool(finite and np.min(np.asarray(margin)) >= 0.0)
        convergence = bool(converged)
        physical = bool(physically_valid)
        qualification = bool(qualified)
        rights = bool(rights_authorized)
        self.conservation_residual = residual
        self.admissibility_margin = margin
        self.conservation_tolerance = tolerance
        self.finite = finite
        self.converged = convergence
        self.physically_valid = physical
        self.qualified = qualification
        self.conservation_passed = conservation_passed
        self.admissibility_passed = admissibility_passed
        self.rights_authorized = rights
        self.candidate_id = candidate
        self.conservation_evidence_id = conservation_id
        self.admissibility_evidence_id = admissibility_id
        self.rights_evidence_id = rights_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "nr-learned-closure-admission-evidence",
                "candidate_id": candidate,
                "conservation_residual": array_tree_fingerprint(residual),
                "admissibility_margin": array_tree_fingerprint(margin),
                "conservation_tolerance": tolerance.hex(),
                "finite": finite,
                "converged": convergence,
                "physically_valid": physical,
                "qualified": qualification,
                "conservation_passed": conservation_passed,
                "admissibility_passed": admissibility_passed,
                "rights_authorized": rights,
                "conservation_evidence_id": conservation_id,
                "admissibility_evidence_id": admissibility_id,
                "rights_evidence_id": rights_id,
            }
        )

    @property
    def eligible(self) -> bool:
        return (
            self.finite
            and self.converged
            and self.physically_valid
            and self.qualified
            and self.conservation_passed
            and self.admissibility_passed
            and self.rights_authorized
        )


class LearnedClosureAdmission(StrictModule, NonTrainableState):
    """Fail-closed host decision binding one correction to one native model."""

    evidence: LearnedClosureAdmissionEvidence
    admitted: bool = eqx.field(static=True)
    refusal_reasons: tuple[str, ...] = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)
    native_model_id: str = eqx.field(static=True)
    admission_id: str = eqx.field(static=True)

    def __init__(
        self,
        evidence: LearnedClosureAdmissionEvidence,
        /,
        *,
        candidate_id: str,
        native_model_id: str,
        admitted: bool,
        refusal_reasons: tuple[str, ...],
    ):
        if not isinstance(evidence, LearnedClosureAdmissionEvidence):
            raise TypeError("evidence must be LearnedClosureAdmissionEvidence.")
        candidate = _identifier(candidate_id, "candidate_id")
        native = _identifier(native_model_id, "native_model_id")
        reasons = tuple(refusal_reasons)
        if any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError("Learned closure refusal reasons must be nonempty strings.")
        admitted_ = bool(admitted)
        if (admitted_ and reasons) or (not admitted_ and not reasons):
            raise ValueError(
                "An admission has no refusals; a refusal has explicit reasons."
            )
        self.evidence = evidence
        self.admitted = admitted_
        self.refusal_reasons = reasons
        self.candidate_id = candidate
        self.native_model_id = native
        self.admission_id = canonical_fingerprint(
            {
                "kind": "nr-learned-closure-admission",
                "candidate_id": candidate,
                "native_model_id": native,
                "evidence_id": evidence.evidence_id,
                "admitted": admitted_,
                "refusal_reasons": list(reasons),
            }
        )


class LearnedClosureEvaluation(StrictModule):
    """Native value, learned correction, and combined value retained separately."""

    native_value: Array
    learned_correction: Array
    combined_value: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    candidate_id: str = eqx.field(static=True)
    native_model_id: str = eqx.field(static=True)
    admission_id: str = eqx.field(static=True)


def admit_learned_closure(
    candidate: LearnedClosureCandidate,
    evidence: LearnedClosureAdmissionEvidence,
    /,
) -> LearnedClosureAdmission:
    """Admit only evidence-complete additive corrections, never replacements."""

    if not isinstance(candidate, LearnedClosureCandidate):
        raise TypeError("candidate must be a LearnedClosureCandidate.")
    if not isinstance(evidence, LearnedClosureAdmissionEvidence):
        raise TypeError("evidence must be LearnedClosureAdmissionEvidence.")
    if evidence.candidate_id != candidate.candidate_id:
        raise ValueError("Learned closure evidence belongs to another candidate.")
    reasons: list[str] = []
    if not evidence.finite:
        reasons.append("nonfinite-evidence")
    if not evidence.converged:
        reasons.append("evidence-nonconverged")
    if not evidence.physically_valid:
        reasons.append("physical-validation-failed")
    if not evidence.qualified:
        reasons.append("qualification-failed")
    if not evidence.conservation_passed:
        reasons.append("conservation-failed")
    if not evidence.admissibility_passed:
        reasons.append("admissibility-failed")
    if not evidence.rights_authorized:
        reasons.append("requested-use-rights-missing")
    return LearnedClosureAdmission(
        evidence,
        candidate_id=candidate.candidate_id,
        native_model_id=candidate.native_model_id,
        admitted=not reasons,
        refusal_reasons=tuple(reasons),
    )


def apply_admitted_learned_closure(
    candidate: LearnedClosureCandidate,
    admission: LearnedClosureAdmission,
    native_value: ArrayLike,
    model_inputs: Any,
    /,
) -> LearnedClosureEvaluation:
    """Add an admitted correction while retaining and requiring native physics."""

    if not isinstance(candidate, LearnedClosureCandidate) or not isinstance(
        admission, LearnedClosureAdmission
    ):
        raise TypeError("A learned candidate and its admission are required.")
    if (
        admission.candidate_id != candidate.candidate_id
        or admission.native_model_id != candidate.native_model_id
    ):
        raise ValueError("Learned candidate and admission identities do not match.")
    if not admission.admitted:
        raise ValueError("An unadmitted learned closure cannot enter the physics path.")
    native = jnp.asarray(native_value)
    if not eqx.is_inexact_array(native):
        raise TypeError("Native physics values must be an inexact array.")
    correction = jnp.asarray(candidate.model(model_inputs), dtype=native.dtype)
    if correction.shape != native.shape:
        raise ValueError("Learned correction shape must match the native physics value.")
    combined = native + correction
    finite = jnp.all(jnp.isfinite(native)) & jnp.all(jnp.isfinite(correction))
    converged = finite & jnp.asarray(admission.evidence.converged)
    physical = converged & jnp.asarray(admission.evidence.physically_valid)
    qualified = physical & jnp.asarray(admission.evidence.qualified)
    derivative = qualified & jnp.asarray(candidate.derivative_supported)
    return LearnedClosureEvaluation(
        native,
        correction,
        combined,
        finite,
        converged,
        physical,
        qualified,
        derivative,
        candidate.candidate_id,
        candidate.native_model_id,
        admission.admission_id,
    )


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


__all__ = [
    "LearnedClosureAdmission",
    "LearnedClosureAdmissionEvidence",
    "LearnedClosureCandidate",
    "LearnedClosureEvaluation",
    "ModelDiscrepancyRecord",
    "NumericalErrorRecord",
    "RelativisticMultifidelityEvaluation",
    "RelativisticMultifidelityPlan",
    "admit_learned_closure",
    "apply_admitted_learned_closure",
    "grhd_model_evaluation",
    "grhd_recovery_model_evaluation",
    "smooth_grhd_inverse_adapter",
    "smooth_nr_inverse_adapter",
    "z4c_step_model_evaluation",
]
