#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Branch-local inverse adapters for stationary compact-object calculations.

The adapters in this module differentiate an already selected numerical branch. They
never differentiate branch selection, event localization, shock formation, or topology
changes.  A sensitivity is usable only when the center and both finite-difference
stencils retain the declared branch and carry affirmative physical derivative evidence.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._identity import callable_payload
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...metrix._metric_domain import MetricDomainEvidence


class FixedBranchModelEvaluation(StrictModule):
    """One fixed-shape observable with explicit scientific and smoothness evidence."""

    values: Array
    branch_signature: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    shock_free: Array
    event_free: Array
    topology_fixed: Array
    realization_id: str = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        branch_signature: ArrayLike,
        /,
        *,
        finite: ArrayLike,
        converged: ArrayLike,
        physically_valid: ArrayLike,
        qualified: ArrayLike,
        derivative_valid: ArrayLike,
        shock_free: ArrayLike = True,
        event_free: ArrayLike = True,
        topology_fixed: ArrayLike = True,
        realization_id: str,
        branch_id: str,
    ):
        values_ = jnp.asarray(values).reshape((-1,))
        signature = jax.lax.stop_gradient(
            jnp.asarray(branch_signature, dtype=jnp.int32).reshape((-1,))
        )
        if values_.size == 0 or signature.size == 0:
            raise ValueError("Fixed-branch values and branch signature must be nonempty.")
        if not eqx.is_inexact_array(values_):
            raise TypeError("Fixed-branch observable values must be inexact arrays.")
        flags = tuple(
            jnp.asarray(value, dtype=bool)
            for value in (
                finite,
                converged,
                physically_valid,
                qualified,
                derivative_valid,
                shock_free,
                event_free,
                topology_fixed,
            )
        )
        if any(value.shape != () for value in flags):
            raise ValueError("Fixed-branch evidence flags must be scalar arrays.")
        self.values = values_
        self.branch_signature = signature
        (
            self.finite,
            self.converged,
            self.physically_valid,
            self.qualified,
            self.derivative_valid,
            self.shock_free,
            self.event_free,
            self.topology_fixed,
        ) = flags
        self.realization_id = _identifier(realization_id, "realization_id")
        self.branch_id = _identifier(branch_id, "branch_id")

    @property
    def sensitivity_eligible(self) -> Array:
        """Whether this evaluation is smooth and admitted on its selected branch."""

        return (
            self.finite
            & self.converged
            & self.physically_valid
            & self.qualified
            & self.derivative_valid
            & self.shock_free
            & self.event_free
            & self.topology_fixed
            & jnp.all(jnp.isfinite(self.values))
        )


class FixedBranchSensitivityEvidence(StrictModule):
    """JVP, VJP pairing, and central-difference evidence for one selected branch."""

    value: Array
    jvp: Array
    vjp: Array
    finite_difference: Array
    jvp_finite_difference_residual: Array
    vjp_pairing_residual: Array
    center_eligible: Array
    stencil_eligible: Array
    branch_stable: Array
    finite: Array
    derivative_valid: Array
    epsilon: Array
    adapter_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)


class FixedBranchInverseAdapter(StrictModule, NonTrainableState):
    """A host-bound evaluator whose numeric branch signature must remain unchanged.

    Callable ``StrictModule`` instances are content-addressed by the repository
    identity substrate. Opaque Python callables require explicit semantic and numeric
    revision IDs; a name, type, or captured array leaves are never treated as
    executable identity.
    """

    evaluator: Callable[[Array], FixedBranchModelEvaluation]
    expected_branch_signature: Array
    parameter_count: int = eqx.field(static=True)
    output_count: int = eqx.field(static=True)
    model_kind: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)
    evaluator_semantic_id: str = eqx.field(static=True)
    evaluator_numeric_id: str = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: Callable[[Array], FixedBranchModelEvaluation],
        expected_branch_signature: ArrayLike,
        /,
        *,
        parameter_count: int,
        output_count: int,
        model_kind: str,
        realization_id: str,
        branch_id: str,
        adapter_id: str,
        evaluator_semantic_id: str | None = None,
        evaluator_numeric_id: str | None = None,
    ):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        parameters = int(parameter_count)
        outputs = int(output_count)
        if parameters <= 0 or outputs <= 0:
            raise ValueError("Fixed-branch parameter and output counts must be positive.")
        signature = jax.lax.stop_gradient(
            jnp.asarray(expected_branch_signature, dtype=jnp.int32).reshape((-1,))
        )
        if signature.size == 0 or bool(jnp.any(~jnp.isfinite(signature))):
            raise ValueError("Expected branch signature must be a finite nonempty array.")
        kind = _identifier(model_kind, "model_kind")
        realization = _identifier(realization_id, "realization_id")
        branch = _identifier(branch_id, "branch_id")
        declared = _identifier(adapter_id, "adapter_id")
        evaluator_identity = callable_payload(
            evaluator,
            semantic_id=evaluator_semantic_id,
            numeric_id=evaluator_numeric_id,
        )
        self.evaluator = evaluator
        self.expected_branch_signature = signature
        self.parameter_count = parameters
        self.output_count = outputs
        self.model_kind = kind
        self.realization_id = realization
        self.branch_id = branch
        self.evaluator_semantic_id = evaluator_identity["semantic_content_id"]
        self.evaluator_numeric_id = evaluator_identity["numeric_content_id"]
        self.adapter_id = canonical_fingerprint(
            {
                "kind": "fixed-branch-inverse-adapter",
                "declared_id": declared,
                "model_kind": kind,
                "realization_id": realization,
                "branch_id": branch,
                "evaluator": evaluator_identity,
                "parameter_count": parameters,
                "output_count": outputs,
                "branch_signature": array_tree_fingerprint(signature),
            }
        )

    def evaluate(self, parameters: ArrayLike, /) -> FixedBranchModelEvaluation:
        values = _parameter_vector(parameters, self.parameter_count)
        result = self.evaluator(values)
        if not isinstance(result, FixedBranchModelEvaluation):
            raise TypeError(
                "A fixed-branch evaluator must return FixedBranchModelEvaluation."
            )
        if result.values.shape != (self.output_count,):
            raise ValueError(
                f"Fixed-branch observable must have shape {(self.output_count,)}."
            )
        if result.branch_signature.shape != self.expected_branch_signature.shape:
            raise ValueError("Fixed-branch signature shape changed during evaluation.")
        if (
            result.realization_id != self.realization_id
            or result.branch_id != self.branch_id
        ):
            raise ValueError(
                "Fixed-branch evaluation identity does not match its adapter."
            )
        branch_matches = jnp.all(
            result.branch_signature == self.expected_branch_signature
        )
        return eqx.tree_at(
            lambda value: (value.qualified, value.derivative_valid),
            result,
            (
                result.qualified & branch_matches,
                result.derivative_valid & branch_matches,
            ),
        )

    def values(self, parameters: ArrayLike, /) -> Array:
        """Evaluate only the numeric observable on the preselected branch."""

        return self.evaluate(parameters).values

    def sensitivity(
        self,
        parameters: ArrayLike,
        direction: ArrayLike,
        /,
        *,
        cotangent: ArrayLike | None = None,
        epsilon: float = 1.0e-4,
    ) -> FixedBranchSensitivityEvidence:
        """Audit a branch-local derivative without exposing invalid tangents.

        The AD operations evaluate only ``values``. Their tangents are masked unless
        center and stencil evaluations are finite, converged, physically valid,
        qualified, derivative-valid, shock-free, event-free, fixed-topology, and on
        exactly the expected branch signature.
        """

        point = _parameter_vector(parameters, self.parameter_count)
        tangent = _parameter_vector(direction, self.parameter_count, dtype=point.dtype)
        step = float(epsilon)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("epsilon must be finite and positive.")
        value, raw_jvp = jax.jvp(self.values, (point,), (tangent,))
        dual = (
            jnp.ones((self.output_count,), dtype=value.dtype)
            if cotangent is None
            else jnp.asarray(cotangent, dtype=value.dtype).reshape((-1,))
        )
        if dual.shape != (self.output_count,):
            raise ValueError(f"cotangent must have shape {(self.output_count,)}.")
        _, pullback = jax.vjp(self.values, point)
        raw_vjp = pullback(dual)[0]
        minus_point = point - step * tangent
        plus_point = point + step * tangent
        minus = self.evaluate(minus_point)
        center = self.evaluate(point)
        plus = self.evaluate(plus_point)
        raw_difference = (plus.values - minus.values) / (2.0 * step)

        expected = self.expected_branch_signature
        branch_stable = jnp.all(center.branch_signature == expected)
        branch_stable = branch_stable & jnp.all(minus.branch_signature == expected)
        branch_stable = branch_stable & jnp.all(plus.branch_signature == expected)
        center_eligible = center.sensitivity_eligible
        stencil_eligible = minus.sensitivity_eligible & plus.sensitivity_eligible
        finite = (
            jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(raw_jvp))
            & jnp.all(jnp.isfinite(raw_vjp))
            & jnp.all(jnp.isfinite(raw_difference))
        )
        valid = center_eligible & stencil_eligible & branch_stable & finite
        nan_output = jnp.full_like(raw_jvp, jnp.nan)
        nan_input = jnp.full_like(raw_vjp, jnp.nan)
        audited_jvp = jnp.where(valid, raw_jvp, nan_output)
        audited_vjp = jnp.where(valid, raw_vjp, nan_input)
        audited_difference = jnp.where(valid, raw_difference, nan_output)
        jvp_residual = jnp.where(
            valid,
            jnp.max(jnp.abs(raw_jvp - raw_difference), initial=0.0),
            jnp.asarray(jnp.nan, dtype=point.real.dtype),
        )
        pairing = jnp.sum(dual * raw_jvp) - jnp.sum(raw_vjp * tangent)
        pairing_residual = jnp.where(
            valid,
            jnp.abs(pairing),
            jnp.asarray(jnp.nan, dtype=point.real.dtype),
        )
        return FixedBranchSensitivityEvidence(
            value,
            audited_jvp,
            audited_vjp,
            audited_difference,
            jvp_residual,
            pairing_residual,
            center_eligible,
            stencil_eligible,
            branch_stable,
            finite,
            valid,
            jnp.asarray(step, dtype=point.real.dtype),
            self.adapter_id,
            self.realization_id,
            self.branch_id,
        )


def kerr_geometry_inverse_adapter(
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
    """Bind smooth Kerr metric/curvature observables on one declared chart branch."""

    return FixedBranchInverseAdapter(
        evaluator,
        expected_branch_signature,
        parameter_count=parameter_count,
        output_count=output_count,
        model_kind="kerr-geometry",
        realization_id=realization_id,
        branch_id=branch_id,
        adapter_id=adapter_id,
        evaluator_semantic_id=evaluator_semantic_id,
        evaluator_numeric_id=evaluator_numeric_id,
    )


def kerr_thermodynamics_inverse_adapter(
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
    """Bind one nonsingular Kerr equilibrium ensemble and horizon branch."""

    return FixedBranchInverseAdapter(
        evaluator,
        expected_branch_signature,
        parameter_count=parameter_count,
        output_count=output_count,
        model_kind="kerr-thermodynamics",
        realization_id=realization_id,
        branch_id=branch_id,
        adapter_id=adapter_id,
        evaluator_semantic_id=evaluator_semantic_id,
        evaluator_numeric_id=evaluator_numeric_id,
    )


def simple_qnm_root_inverse_adapter(
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
    """Bind one simple, isolated QNM root without differentiating root selection."""

    return FixedBranchInverseAdapter(
        evaluator,
        expected_branch_signature,
        parameter_count=parameter_count,
        output_count=output_count,
        model_kind="simple-qnm-root",
        realization_id=realization_id,
        branch_id=branch_id,
        adapter_id=adapter_id,
        evaluator_semantic_id=evaluator_semantic_id,
        evaluator_numeric_id=evaluator_numeric_id,
    )


def fixed_branch_model_evaluation(
    values: ArrayLike,
    branch_signature: ArrayLike,
    /,
    *,
    status: Any,
    realization_id: str,
    branch_id: str,
    shock_free: ArrayLike = True,
    event_free: ArrayLike = True,
    topology_fixed: ArrayLike = True,
) -> FixedBranchModelEvaluation:
    """Adapt a standard relativity status-evidence record without hiding failures."""

    return FixedBranchModelEvaluation(
        values,
        branch_signature,
        finite=jnp.all(jnp.asarray(status.finite)),
        converged=jnp.all(jnp.asarray(status.converged)),
        physically_valid=jnp.all(jnp.asarray(status.physically_valid)),
        qualified=jnp.all(jnp.asarray(status.qualified)),
        derivative_valid=jnp.all(jnp.asarray(status.derivative_valid)),
        shock_free=shock_free,
        event_free=event_free,
        topology_fixed=topology_fixed,
        realization_id=realization_id,
        branch_id=branch_id,
    )


def kerr_geometry_model_evaluation(
    values: ArrayLike,
    domain: MetricDomainEvidence,
    branch_signature: ArrayLike,
    /,
    *,
    realization_id: str,
    branch_id: str,
) -> FixedBranchModelEvaluation:
    """Adapt exact Kerr values and their ``MetricDomainEvidence``."""
    if not isinstance(domain, MetricDomainEvidence):
        raise TypeError("domain must be MetricDomainEvidence.")

    return FixedBranchModelEvaluation(
        values,
        branch_signature,
        finite=jnp.all(jnp.asarray(domain.finite)),
        converged=True,
        physically_valid=jnp.all(jnp.asarray(domain.physically_valid)),
        qualified=jnp.all(jnp.asarray(domain.qualified)),
        derivative_valid=jnp.all(jnp.asarray(domain.derivative_valid)),
        realization_id=realization_id,
        branch_id=branch_id,
    )


def kerr_thermodynamics_model_evaluation(
    result: Any,
    values: ArrayLike,
    /,
    *,
    realization_id: str,
    branch_id: str,
) -> FixedBranchModelEvaluation:
    """Adapt a Kerr equilibrium result without merging thermodynamic ensembles."""

    return FixedBranchModelEvaluation(
        values,
        jnp.asarray(result.branch.code, dtype=jnp.int32).reshape((-1,)),
        finite=jnp.all(jnp.asarray(result.finite)),
        converged=jnp.all(jnp.asarray(result.converged)),
        physically_valid=jnp.all(jnp.asarray(result.physically_valid)),
        qualified=jnp.all(jnp.asarray(result.qualified)),
        derivative_valid=jnp.all(jnp.asarray(result.derivative_valid)),
        realization_id=realization_id,
        branch_id=branch_id,
    )


def qnm_root_model_evaluation(
    result: Any,
    /,
    *,
    realization_id: str,
) -> FixedBranchModelEvaluation:
    """Realify one simple QNM root while retaining its solver branch identity."""

    frequency = jnp.asarray(result.angular_frequency)
    separation = jnp.asarray(result.separation_constant)
    values = jnp.stack(
        (
            jnp.real(frequency),
            jnp.imag(frequency),
            jnp.real(separation),
            jnp.imag(separation),
        )
    )
    return FixedBranchModelEvaluation(
        values,
        jnp.asarray(result.status, dtype=jnp.int32).reshape((-1,)),
        finite=jnp.all(jnp.asarray(result.finite)),
        converged=jnp.all(jnp.asarray(result.converged)),
        physically_valid=jnp.all(jnp.asarray(result.physically_valid)),
        qualified=jnp.all(jnp.asarray(result.qualified)),
        derivative_valid=jnp.all(jnp.asarray(result.derivative_valid)),
        realization_id=realization_id,
        branch_id=result.branch_id,
    )


def _parameter_vector(
    value: ArrayLike,
    count: int,
    /,
    *,
    dtype: Any = None,
) -> Array:
    array = jnp.asarray(value, dtype=dtype).reshape((-1,))
    if array.shape != (count,):
        raise ValueError(f"Parameter vectors must have shape {(count,)}.")
    if not eqx.is_inexact_array(array) or jnp.issubdtype(
        array.dtype, jnp.complexfloating
    ):
        raise TypeError("Inverse parameters and directions must be real inexact arrays.")
    return array


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


__all__ = [
    "FixedBranchInverseAdapter",
    "FixedBranchModelEvaluation",
    "FixedBranchSensitivityEvidence",
    "fixed_branch_model_evaluation",
    "kerr_geometry_model_evaluation",
    "kerr_geometry_inverse_adapter",
    "kerr_thermodynamics_model_evaluation",
    "kerr_thermodynamics_inverse_adapter",
    "simple_qnm_root_inverse_adapter",
    "qnm_root_model_evaluation",
]
