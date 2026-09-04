#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ._les_closures import (
    AlgebraicLESInputs,
    AlgebraicLESResult,
    LESParameterProvenance,
    ResolvedLESFilter,
)


_Differentiation = Literal["smooth", "branchwise"]


class DynamicLESProvenance(StrictModule, NonTrainableState):
    """Bind a dynamic procedure to distinct resolved and test filters.

    ``test_filter_ratio`` is the directional physical-width ratio
    ``Delta_test / Delta_resolved`` in the order declared by ``axis_names``.
    A scalar ratio is expanded to all three directions. The ratio is metadata;
    this object performs no filtering and accepts no dealiasing object.
    """

    parameter_provenance: LESParameterProvenance
    test_filter: ResolvedLESFilter
    test_filter_ratio: tuple[float, float, float] = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameter_provenance: LESParameterProvenance,
        test_filter: ResolvedLESFilter,
        test_filter_ratio: ArrayLike,
        /,
    ):
        if not isinstance(parameter_provenance, LESParameterProvenance):
            raise TypeError("parameter_provenance must be LESParameterProvenance.")
        if not isinstance(test_filter, ResolvedLESFilter):
            raise TypeError("test_filter must be a ResolvedLESFilter.")
        resolved_filter = parameter_provenance.resolved_filter
        if resolved_filter.filter_id == test_filter.filter_id:
            raise ValueError("Resolved and test LES filters must be distinct.")
        if resolved_filter.dimension != test_filter.dimension:
            raise ValueError("Resolved and test LES filter dimensions must match.")
        if resolved_filter.axis_names != test_filter.axis_names:
            raise ValueError("Resolved and test LES filter axis order must match.")
        if resolved_filter.topology != test_filter.topology:
            raise ValueError("Resolved and test LES filter topologies must match.")
        if resolved_filter.boundary_class != test_filter.boundary_class:
            raise ValueError("Resolved and test LES filter boundary classes must match.")
        if resolved_filter.commutation_status != test_filter.commutation_status:
            raise ValueError("Resolved and test LES commutation semantics must match.")

        ratio_array = jnp.asarray(test_filter_ratio)
        if isinstance(ratio_array, jax.core.Tracer):
            raise TypeError("The test-filter ratio must be concrete provenance metadata.")
        ratio = np.asarray(ratio_array, dtype=float)
        if ratio.shape == ():
            ratio = np.repeat(ratio[None], 3)
        if ratio.shape != (3,):
            raise ValueError("The test-filter ratio must be scalar or have shape (3,).")
        if np.any(~np.isfinite(ratio)) or np.any(ratio <= 1.0):
            raise ValueError(
                "Every directional test-filter ratio must be finite and > 1."
            )
        ratio_tuple = tuple(float(value) for value in ratio)

        self.parameter_provenance = parameter_provenance
        self.test_filter = test_filter
        self.test_filter_ratio = ratio_tuple
        self.provenance_id = canonical_fingerprint(
            {
                "kind": "dynamic-les-provenance",
                "parameter_provenance": parameter_provenance.provenance_id,
                "resolved_filter": resolved_filter.filter_id,
                "test_filter": test_filter.filter_id,
                "test_filter_ratio": list(ratio_tuple),
            }
        )

    @property
    def resolved_filter(self) -> ResolvedLESFilter:
        """Return the typed filter defining the resolved field."""
        return self.parameter_provenance.resolved_filter


class DynamicLESInputs(StrictModule):
    """Already filtered Germano tensors and local algebraic-model inputs.

    The backend supplies ``leonard_tensor`` and ``modeled_tensor`` with common
    trailing shape ``(3, 3)`` and the exact same leading field shape. The
    modeled tensor is the coefficient-free Germano tensor ``M`` in
    ``L_deviatoric = C_d M_deviatoric``. No spatial filter is invoked here.
    ``accepted_update_mask`` is consumed only by history averaging; a scalar
    mask applies to the complete field.
    """

    leonard_tensor: Array
    modeled_tensor: Array
    algebraic_inputs: AlgebraicLESInputs
    provenance: DynamicLESProvenance
    accepted_update_mask: Array

    def __init__(
        self,
        leonard_tensor: ArrayLike,
        modeled_tensor: ArrayLike,
        algebraic_inputs: AlgebraicLESInputs,
        provenance: DynamicLESProvenance,
        /,
        *,
        accepted_update_mask: ArrayLike,
    ):
        if not isinstance(algebraic_inputs, AlgebraicLESInputs):
            raise TypeError("algebraic_inputs must be AlgebraicLESInputs.")
        if not isinstance(provenance, DynamicLESProvenance):
            raise TypeError("provenance must be DynamicLESProvenance.")
        leonard = _inexact_array(leonard_tensor)
        modeled = _inexact_array(modeled_tensor)
        if leonard.ndim < 2 or leonard.shape[-2:] != (3, 3):
            raise ValueError("The Leonard tensor must have trailing shape (3, 3).")
        if modeled.shape != leonard.shape:
            raise ValueError("Leonard and modeled Germano tensors must have equal shape.")
        gradient = algebraic_inputs.velocity_gradient
        if gradient.shape != leonard.shape:
            raise ValueError(
                "The velocity-gradient and Germano tensor field shapes must match."
            )
        _require_finite(leonard, "Leonard tensor")
        _require_finite(modeled, "modeled Germano tensor")
        _require_finite(gradient, "velocity gradient")
        _require_finite(
            algebraic_inputs.filter_scale.directional_widths,
            "resolved LES filter scale",
        )

        leading_shape = leonard.shape[:-2]
        width_shape = algebraic_inputs.filter_scale.directional_widths.shape[:-1]
        if np.broadcast_shapes(width_shape, leading_shape) != leading_shape:
            raise ValueError(
                "The resolved LES filter scale does not broadcast to the tensor field."
            )

        accepted = jnp.asarray(accepted_update_mask)
        if accepted.dtype != jnp.bool_:
            raise TypeError("accepted_update_mask must have boolean dtype.")
        if accepted.shape not in ((), leading_shape):
            raise ValueError(
                "accepted_update_mask must be scalar or match the tensor leading shape."
            )
        self.leonard_tensor = leonard
        self.modeled_tensor = modeled
        self.algebraic_inputs = algebraic_inputs
        self.provenance = provenance
        self.accepted_update_mask = accepted


class LagrangianDynamicLESState(StrictModule, NonTrainableState):
    """Immutable restart state for accepted history-averaging updates."""

    averaged_numerator: Array
    averaged_denominator: Array
    initialized_mask: Array
    accepted_updates: Array
    rejected_updates: Array
    continuation_id: str = eqx.field(static=True)

    def __init__(
        self,
        averaged_numerator: ArrayLike,
        averaged_denominator: ArrayLike,
        initialized_mask: ArrayLike,
        accepted_updates: ArrayLike,
        rejected_updates: ArrayLike,
        continuation_id: str,
        /,
    ):
        numerator = _inexact_array(averaged_numerator)
        denominator = _inexact_array(averaged_denominator)
        initialized = jnp.asarray(initialized_mask)
        accepted = jnp.asarray(accepted_updates)
        rejected = jnp.asarray(rejected_updates)
        if denominator.shape != numerator.shape or initialized.shape != numerator.shape:
            raise ValueError("Lagrangian LES state field shapes must match.")
        if initialized.dtype != jnp.bool_:
            raise TypeError("Lagrangian initialized_mask must have boolean dtype.")
        for count, name in (
            (accepted, "accepted_updates"),
            (rejected, "rejected_updates"),
        ):
            if count.shape != () or not jnp.issubdtype(count.dtype, jnp.integer):
                raise TypeError(f"Lagrangian {name} must be a scalar integer.")
        if not isinstance(continuation_id, str) or not continuation_id.strip():
            raise ValueError("Lagrangian continuation_id must be a non-empty string.")
        _require_finite(numerator, "Lagrangian numerator")
        _require_finite(denominator, "Lagrangian denominator")
        if _is_concrete(denominator) and np.any(np.asarray(denominator) < 0.0):
            raise ValueError("Lagrangian denominator history must be nonnegative.")
        if _is_concrete(accepted) and int(np.asarray(accepted)) < 0:
            raise ValueError("Lagrangian accepted_updates must be nonnegative.")
        if _is_concrete(rejected) and int(np.asarray(rejected)) < 0:
            raise ValueError("Lagrangian rejected_updates must be nonnegative.")
        self.averaged_numerator = numerator
        self.averaged_denominator = denominator
        self.initialized_mask = initialized
        self.accepted_updates = accepted
        self.rejected_updates = rejected
        self.continuation_id = continuation_id.strip()


class GermanoLeastSquaresEvidence(StrictModule):
    """Auditable contractions, policy activity, and differentiation semantics."""

    pointwise_numerator: Array
    pointwise_denominator: Array
    averaged_numerator: Array
    averaged_denominator: Array
    effective_denominator: Array
    unconstrained_coefficient: Array
    zero_denominator_count: Array
    regularization_activity_count: Array
    backscatter_activity_count: Array
    backscatter_limit_count: Array
    accepted_update_count: Array
    rejected_update_count: Array
    finite: Array
    averaging: str = eqx.field(static=True)
    regularization: str = eqx.field(static=True)
    backscatter: str = eqx.field(static=True)
    differentiation: _Differentiation = eqx.field(static=True)
    dynamic_provenance_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class DynamicLESResult(StrictModule):
    """Dynamic coefficient, ready algebraic stress, evidence, and restart state.

    ``coefficient`` is ``C_d`` in
    ``nu_t = C_d Delta^2 sqrt(2 S:S)``. Consequently it may be signed when the
    selected backscatter policy permits it. ``prepared_algebraic_stress`` is an
    already evaluated, backend-ready stress result; it is not a filter plan.
    """

    coefficient: Array
    prepared_algebraic_stress: AlgebraicLESResult
    evidence: GermanoLeastSquaresEvidence
    continuation_state: LagrangianDynamicLESState | None


class AbstractDynamicLESAveraging(StrictModule):
    """Pure averaging policy for the numerator and denominator separately."""

    name: AbstractAttribute[str]
    differentiation: AbstractAttribute[_Differentiation]
    averaging_id: AbstractAttribute[str]

    @abc.abstractmethod
    def average(
        self,
        numerator: Array,
        denominator: Array,
        inputs: DynamicLESInputs,
        state: LagrangianDynamicLESState | None,
        continuation_id: str,
        /,
    ) -> tuple[
        Array,
        Array,
        LagrangianDynamicLESState | None,
        Array,
        Array,
    ]:
        """Average contractions and return per-call accepted/rejected counts."""


class GlobalDynamicLESAveraging(AbstractDynamicLESAveraging, NonTrainableState):
    """Average contractions over every leading field axis."""

    name: str = eqx.field(static=True)
    differentiation: _Differentiation = eqx.field(static=True)
    averaging_id: str = eqx.field(static=True)

    def __init__(self):
        self.name = "global"
        self.differentiation = "smooth"
        self.averaging_id = _policy_id("dynamic-les-averaging", {"name": self.name})

    def average(
        self,
        numerator: Array,
        denominator: Array,
        inputs: DynamicLESInputs,
        state: LagrangianDynamicLESState | None,
        continuation_id: str,
        /,
    ) -> tuple[Array, Array, None, Array, Array]:
        del inputs, continuation_id
        _reject_state(state, self.name)
        axes = tuple(range(numerator.ndim))
        if axes:
            numerator = jnp.mean(numerator, axis=axes)
            denominator = jnp.mean(denominator, axis=axes)
        zero = jnp.asarray(0, dtype=jnp.int32)
        return numerator, denominator, None, zero, zero


class HomogeneousPlaneDynamicLESAveraging(AbstractDynamicLESAveraging, NonTrainableState):
    """Average over declared physical axes while retaining broadcast dimensions."""

    axis_names: tuple[str, ...] = eqx.field(static=True)
    name: str = eqx.field(static=True)
    differentiation: _Differentiation = eqx.field(static=True)
    averaging_id: str = eqx.field(static=True)

    def __init__(self, axis_names: tuple[str, ...], /):
        if not isinstance(axis_names, tuple) or not axis_names:
            raise TypeError("Homogeneous averaging axis_names must be a non-empty tuple.")
        if any(not isinstance(axis, str) or not axis.strip() for axis in axis_names):
            raise TypeError("Homogeneous averaging axis names must be non-empty strings.")
        axes = tuple(axis.strip() for axis in axis_names)
        if len(set(axes)) != len(axes) or len(axes) > 3:
            raise ValueError(
                "Homogeneous averaging axes must be unique and at most three."
            )
        self.axis_names = axes
        self.name = "homogeneous-plane"
        self.differentiation = "smooth"
        self.averaging_id = _policy_id(
            "dynamic-les-averaging", {"name": self.name, "axis_names": list(axes)}
        )

    def average(
        self,
        numerator: Array,
        denominator: Array,
        inputs: DynamicLESInputs,
        state: LagrangianDynamicLESState | None,
        continuation_id: str,
        /,
    ) -> tuple[Array, Array, None, Array, Array]:
        del continuation_id
        _reject_state(state, self.name)
        if numerator.ndim != 3:
            raise ValueError(
                "Homogeneous-plane averaging requires exactly three spatial axes."
            )
        declared = inputs.provenance.resolved_filter.axis_names
        if any(axis not in declared for axis in self.axis_names):
            raise ValueError("A homogeneous averaging axis is absent from the filter.")
        axes = tuple(declared.index(axis) for axis in self.axis_names)
        averaged_numerator = jnp.mean(numerator, axis=axes, keepdims=True)
        averaged_denominator = jnp.mean(denominator, axis=axes, keepdims=True)
        zero = jnp.asarray(0, dtype=jnp.int32)
        return averaged_numerator, averaged_denominator, None, zero, zero


class LocalKernelDynamicLESAveraging(AbstractDynamicLESAveraging, NonTrainableState):
    """Periodic local averaging with fixed, normalized, nonnegative 3-D weights."""

    kernel_weights: tuple[tuple[tuple[float, ...], ...], ...] = eqx.field(static=True)
    kernel_shape: tuple[int, int, int] = eqx.field(static=True)
    name: str = eqx.field(static=True)
    differentiation: _Differentiation = eqx.field(static=True)
    averaging_id: str = eqx.field(static=True)

    def __init__(self, kernel_weights: ArrayLike, /):
        array = jnp.asarray(kernel_weights)
        if isinstance(array, jax.core.Tracer):
            raise TypeError("Local averaging kernel weights must be concrete.")
        weights = np.asarray(array, dtype=float)
        if weights.ndim != 3:
            raise ValueError("Local averaging kernel weights must be three-dimensional.")
        if any(size <= 0 or size % 2 == 0 for size in weights.shape):
            raise ValueError("Local averaging kernel extents must be positive and odd.")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("Local averaging weights must be finite and nonnegative.")
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("Local averaging kernel must have positive total weight.")
        normalized = weights / total
        nested = tuple(
            tuple(tuple(float(value) for value in row) for row in plane)
            for plane in normalized
        )
        self.kernel_weights = nested
        self.kernel_shape = tuple(int(size) for size in weights.shape)
        self.name = "local-periodic-kernel"
        self.differentiation = "smooth"
        self.averaging_id = _policy_id(
            "dynamic-les-averaging",
            {"name": self.name, "kernel_weights": nested},
        )

    def average(
        self,
        numerator: Array,
        denominator: Array,
        inputs: DynamicLESInputs,
        state: LagrangianDynamicLESState | None,
        continuation_id: str,
        /,
    ) -> tuple[Array, Array, None, Array, Array]:
        del continuation_id
        _reject_state(state, self.name)
        if numerator.ndim != 3:
            raise ValueError(
                "Local kernel averaging requires exactly three spatial axes."
            )
        if inputs.provenance.resolved_filter.boundary_class != "periodic":
            raise ValueError("Periodic local kernel averaging requires periodic filters.")
        weights = jnp.asarray(self.kernel_weights, dtype=numerator.dtype)
        center = tuple(size // 2 for size in self.kernel_shape)
        averaged_numerator = jnp.zeros_like(numerator)
        averaged_denominator = jnp.zeros_like(denominator)
        for index in np.ndindex(self.kernel_shape):
            weight = weights[index]
            shift = tuple(center[axis] - index[axis] for axis in range(3))
            averaged_numerator = averaged_numerator + weight * jnp.roll(
                numerator, shift, axis=(0, 1, 2)
            )
            averaged_denominator = averaged_denominator + weight * jnp.roll(
                denominator, shift, axis=(0, 1, 2)
            )
        zero = jnp.asarray(0, dtype=jnp.int32)
        return averaged_numerator, averaged_denominator, None, zero, zero


class LagrangianDynamicLESAveraging(AbstractDynamicLESAveraging, NonTrainableState):
    """Accepted-update exponential history averaging at every field point.

    ``relaxation`` is the weight of a newly accepted sample. Rejected entries
    preserve both histories exactly. Previously uninitialized accepted entries
    take the first sample without blending it with artificial zeros.
    """

    relaxation: float = eqx.field(static=True)
    name: str = eqx.field(static=True)
    differentiation: _Differentiation = eqx.field(static=True)
    averaging_id: str = eqx.field(static=True)

    def __init__(self, relaxation: float, /):
        if not isinstance(relaxation, (int, float)):
            raise TypeError("Lagrangian relaxation must be a real scalar.")
        value = float(relaxation)
        if not np.isfinite(value) or value <= 0.0 or value > 1.0:
            raise ValueError("Lagrangian relaxation must be finite and in (0, 1].")
        self.relaxation = value
        self.name = "lagrangian-history"
        self.differentiation = "branchwise"
        self.averaging_id = _policy_id(
            "dynamic-les-averaging", {"name": self.name, "relaxation": value}
        )

    def average(
        self,
        numerator: Array,
        denominator: Array,
        inputs: DynamicLESInputs,
        state: LagrangianDynamicLESState | None,
        continuation_id: str,
        /,
    ) -> tuple[Array, Array, LagrangianDynamicLESState, Array, Array]:
        if state is None:
            raise TypeError(
                "Lagrangian averaging requires explicit initialized continuation state."
            )
        if state.continuation_id != continuation_id:
            raise ValueError("Lagrangian LES continuation state is incompatible.")
        if state.averaged_numerator.shape != numerator.shape:
            raise ValueError("Lagrangian LES continuation state shape is incompatible.")
        accepted_mask = jnp.broadcast_to(inputs.accepted_update_mask, numerator.shape)
        first = accepted_mask & ~state.initialized_mask
        subsequent = accepted_mask & state.initialized_mask
        blended_numerator = (
            1.0 - self.relaxation
        ) * state.averaged_numerator + self.relaxation * numerator
        blended_denominator = (
            1.0 - self.relaxation
        ) * state.averaged_denominator + self.relaxation * denominator
        next_numerator = jnp.where(
            first,
            numerator,
            jnp.where(subsequent, blended_numerator, state.averaged_numerator),
        )
        next_denominator = jnp.where(
            first,
            denominator,
            jnp.where(subsequent, blended_denominator, state.averaged_denominator),
        )
        accepted_count = _count(accepted_mask).astype(state.accepted_updates.dtype)
        rejected_count = _count(~accepted_mask).astype(state.rejected_updates.dtype)
        next_state = LagrangianDynamicLESState(
            next_numerator,
            next_denominator,
            state.initialized_mask | accepted_mask,
            state.accepted_updates + accepted_count,
            state.rejected_updates + rejected_count,
            continuation_id,
        )
        return (
            next_numerator,
            next_denominator,
            next_state,
            accepted_count,
            rejected_count,
        )


class AbstractDenominatorRegularization(StrictModule):
    """Policy for making the averaged Germano denominator usable."""

    name: AbstractAttribute[str]
    differentiation: AbstractAttribute[_Differentiation]
    regularization_id: AbstractAttribute[str]

    @abc.abstractmethod
    def apply(
        self, numerator: Array, denominator: Array, /
    ) -> tuple[Array, Array, Array, Array]:
        """Return raw coefficient, effective denominator, zero count, activity count."""


class ExactDenominatorRegularization(
    AbstractDenominatorRegularization, NonTrainableState
):
    """Use the exact quotient and define the zero-denominator branch as zero."""

    name: str = eqx.field(static=True)
    differentiation: _Differentiation = eqx.field(static=True)
    regularization_id: str = eqx.field(static=True)

    def __init__(self):
        self.name = "exact-zero-branch"
        self.differentiation = "branchwise"
        self.regularization_id = _policy_id(
            "dynamic-les-denominator", {"name": self.name}
        )

    def apply(
        self, numerator: Array, denominator: Array, /
    ) -> tuple[Array, Array, Array, Array]:
        active = denominator > 0.0
        safe_denominator = jnp.where(active, denominator, jnp.ones_like(denominator))
        coefficient = jnp.where(
            active,
            numerator / safe_denominator,
            jnp.zeros_like(numerator),
        )
        zero_count = _count(~active)
        return coefficient, denominator, zero_count, jnp.asarray(0, dtype=jnp.int32)


class AdditiveDenominatorRegularization(
    AbstractDenominatorRegularization, NonTrainableState
):
    """Apply an explicit dimensional Tikhonov shift to every denominator.

    ``shift`` has the same units as ``M:M``. It is never inferred from dtype or
    data and therefore cannot masquerade as a dimensionless numerical epsilon.
    """

    shift: float = eqx.field(static=True)
    name: str = eqx.field(static=True)
    differentiation: _Differentiation = eqx.field(static=True)
    regularization_id: str = eqx.field(static=True)

    def __init__(self, shift: float, /):
        if not isinstance(shift, (int, float)):
            raise TypeError("Denominator regularization shift must be a real scalar.")
        value = float(shift)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Denominator regularization shift must be finite and > 0.")
        self.shift = value
        self.name = "additive-tikhonov"
        self.differentiation = "smooth"
        self.regularization_id = _policy_id(
            "dynamic-les-denominator", {"name": self.name, "shift": value}
        )

    def apply(
        self, numerator: Array, denominator: Array, /
    ) -> tuple[Array, Array, Array, Array]:
        effective = denominator + self.shift
        coefficient = numerator / effective
        zero_count = _count(denominator == 0.0)
        activity = jnp.asarray(denominator.size, dtype=jnp.int32)
        return coefficient, effective, zero_count, activity


class AbstractBackscatterPolicy(StrictModule):
    """Explicit policy applied to the unconstrained dynamic coefficient."""

    name: AbstractAttribute[str]
    differentiation: AbstractAttribute[_Differentiation]
    backscatter_id: AbstractAttribute[str]

    @abc.abstractmethod
    def apply(self, coefficient: Array, /) -> tuple[Array, Array, Array]:
        """Return selected coefficient, negative count, and limited count."""


class AllowSignedBackscatter(AbstractBackscatterPolicy, NonTrainableState):
    """Preserve signed coefficients without a branch in the coefficient map."""

    name: str = eqx.field(static=True)
    differentiation: _Differentiation = eqx.field(static=True)
    backscatter_id: str = eqx.field(static=True)

    def __init__(self):
        self.name = "allow-signed"
        self.differentiation = "smooth"
        self.backscatter_id = _policy_id("dynamic-les-backscatter", {"name": self.name})

    def apply(self, coefficient: Array, /) -> tuple[Array, Array, Array]:
        negative_count = _count(coefficient < 0.0)
        return coefficient, negative_count, jnp.asarray(0, dtype=jnp.int32)


class NonnegativeBackscatterClip(AbstractBackscatterPolicy, NonTrainableState):
    """Suppress backscatter with a branchwise zero derivative at the boundary."""

    name: str = eqx.field(static=True)
    differentiation: _Differentiation = eqx.field(static=True)
    backscatter_id: str = eqx.field(static=True)

    def __init__(self):
        self.name = "nonnegative-clip"
        self.differentiation = "branchwise"
        self.backscatter_id = _policy_id("dynamic-les-backscatter", {"name": self.name})

    def apply(self, coefficient: Array, /) -> tuple[Array, Array, Array]:
        negative = coefficient < 0.0
        selected = jnp.where(coefficient > 0.0, coefficient, jnp.zeros_like(coefficient))
        activity = _count(negative)
        return selected, activity, activity


class BoundedFractionBackscatter(AbstractBackscatterPolicy, NonTrainableState):
    """Bound negative ``C_d`` to a declared fraction of a reference coefficient."""

    maximum_fraction: float = eqx.field(static=True)
    reference_coefficient: float = eqx.field(static=True)
    minimum_coefficient: float = eqx.field(static=True)
    name: str = eqx.field(static=True)
    differentiation: _Differentiation = eqx.field(static=True)
    backscatter_id: str = eqx.field(static=True)

    def __init__(self, maximum_fraction: float, reference_coefficient: float, /):
        if not isinstance(maximum_fraction, (int, float)) or not isinstance(
            reference_coefficient, (int, float)
        ):
            raise TypeError("Backscatter fraction and reference must be real scalars.")
        fraction = float(maximum_fraction)
        reference = float(reference_coefficient)
        if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
            raise ValueError("Backscatter maximum_fraction must be finite and in [0, 1].")
        if not np.isfinite(reference) or reference <= 0.0:
            raise ValueError("Backscatter reference_coefficient must be finite and > 0.")
        minimum = -fraction * reference
        self.maximum_fraction = fraction
        self.reference_coefficient = reference
        self.minimum_coefficient = minimum
        self.name = "bounded-fraction"
        self.differentiation = "branchwise"
        self.backscatter_id = _policy_id(
            "dynamic-les-backscatter",
            {
                "name": self.name,
                "maximum_fraction": fraction,
                "reference_coefficient": reference,
            },
        )

    def apply(self, coefficient: Array, /) -> tuple[Array, Array, Array]:
        negative_count = _count(coefficient < 0.0)
        limited = coefficient < self.minimum_coefficient
        selected = jnp.where(
            limited,
            jnp.asarray(self.minimum_coefficient, dtype=coefficient.dtype),
            coefficient,
        )
        return selected, negative_count, _count(limited)


class DynamicSmagorinskyPlan(StrictModule, NonTrainableState):
    """Unbound Germano least-squares and dynamic Smagorinsky policy plan."""

    averaging: AbstractDynamicLESAveraging
    regularization: AbstractDenominatorRegularization
    backscatter: AbstractBackscatterPolicy
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        averaging: AbstractDynamicLESAveraging,
        regularization: AbstractDenominatorRegularization,
        backscatter: AbstractBackscatterPolicy,
        /,
    ):
        if not isinstance(averaging, AbstractDynamicLESAveraging):
            raise TypeError("averaging must be a dynamic LES averaging policy.")
        if not isinstance(regularization, AbstractDenominatorRegularization):
            raise TypeError("regularization must be a denominator policy.")
        if not isinstance(backscatter, AbstractBackscatterPolicy):
            raise TypeError("backscatter must be a backscatter policy.")
        self.averaging = averaging
        self.regularization = regularization
        self.backscatter = backscatter
        self.model_id = canonical_fingerprint(
            {
                "kind": "dynamic-smagorinsky-model",
                "coefficient_convention": "Cd-multiplies-Delta2-full-strain",
                "germano_trace": "deviatoric",
                "averaging": averaging.averaging_id,
                "regularization": regularization.regularization_id,
                "backscatter": backscatter.backscatter_id,
            }
        )

    def prepare(
        self, provenance: DynamicLESProvenance, /
    ) -> PreparedDynamicSmagorinskyPlan:
        """Bind the complete filter pair and parameter provenance."""
        return PreparedDynamicSmagorinskyPlan(self, provenance)


class PreparedDynamicSmagorinskyPlan(StrictModule, NonTrainableState):
    """Filter-pair-bound dynamic procedure that never performs filtering."""

    averaging: AbstractDynamicLESAveraging
    regularization: AbstractDenominatorRegularization
    backscatter: AbstractBackscatterPolicy
    provenance: DynamicLESProvenance
    model_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    differentiation: _Differentiation = eqx.field(static=True)

    def __init__(
        self,
        plan: DynamicSmagorinskyPlan,
        provenance: DynamicLESProvenance,
        /,
    ):
        if not isinstance(plan, DynamicSmagorinskyPlan):
            raise TypeError("plan must be DynamicSmagorinskyPlan.")
        if not isinstance(provenance, DynamicLESProvenance):
            raise TypeError("provenance must be DynamicLESProvenance.")
        self.averaging = plan.averaging
        self.regularization = plan.regularization
        self.backscatter = plan.backscatter
        self.provenance = provenance
        self.model_id = plan.model_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-dynamic-smagorinsky",
                "model": plan.model_id,
                "dynamic_provenance": provenance.provenance_id,
                "parameter_provenance": provenance.parameter_provenance.provenance_id,
                "resolved_filter": provenance.resolved_filter.filter_id,
                "test_filter": provenance.test_filter.filter_id,
                "test_filter_ratio": list(provenance.test_filter_ratio),
            }
        )
        policies = (
            plan.averaging.differentiation,
            plan.regularization.differentiation,
            plan.backscatter.differentiation,
        )
        self.differentiation = (
            "smooth" if all(value == "smooth" for value in policies) else "branchwise"
        )

    def initial_state(self, inputs: DynamicLESInputs, /) -> LagrangianDynamicLESState:
        """Create explicit zero history for a compatible Lagrangian field."""
        self._validate_inputs(inputs)
        if not isinstance(self.averaging, LagrangianDynamicLESAveraging):
            raise TypeError("Only Lagrangian averaging has continuation state.")
        shape = inputs.leonard_tensor.shape[:-2]
        dtype = jnp.result_type(
            inputs.leonard_tensor,
            inputs.modeled_tensor,
            float,
        )
        continuation_id = self._continuation_id(shape)
        return LagrangianDynamicLESState(
            jnp.zeros(shape, dtype=dtype),
            jnp.zeros(shape, dtype=dtype),
            jnp.zeros(shape, dtype=bool),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            continuation_id,
        )

    def evaluate(
        self,
        inputs: DynamicLESInputs,
        state: LagrangianDynamicLESState | None = None,
        /,
    ) -> DynamicLESResult:
        """Infer ``C_d`` and evaluate its algebraic deviatoric stress."""
        self._validate_inputs(inputs)
        leading_shape = inputs.leonard_tensor.shape[:-2]
        continuation_id = self._continuation_id(leading_shape)
        leonard = _deviatoric(inputs.leonard_tensor)
        modeled = _deviatoric(inputs.modeled_tensor)
        pointwise_numerator = ein.contract(
            "...ij,...ij->...", leonard, modeled, backend="jax"
        )
        pointwise_denominator = ein.contract(
            "...ij,...ij->...", modeled, modeled, backend="jax"
        )
        (
            averaged_numerator,
            averaged_denominator,
            continuation_state,
            accepted_count,
            rejected_count,
        ) = self.averaging.average(
            pointwise_numerator,
            pointwise_denominator,
            inputs,
            state,
            continuation_id,
        )
        (
            unconstrained,
            effective_denominator,
            zero_count,
            regularization_count,
        ) = self.regularization.apply(averaged_numerator, averaged_denominator)
        (
            coefficient,
            backscatter_count,
            backscatter_limit_count,
        ) = self.backscatter.apply(unconstrained)
        algebraic_result = _evaluate_dynamic_stress(coefficient, inputs.algebraic_inputs)
        finite = _all_finite(
            coefficient,
            algebraic_result.kinematic_viscosity,
            algebraic_result.specific_deviatoric_stress,
            algebraic_result.energy_transfer,
        )
        evidence = GermanoLeastSquaresEvidence(
            pointwise_numerator=pointwise_numerator,
            pointwise_denominator=pointwise_denominator,
            averaged_numerator=averaged_numerator,
            averaged_denominator=averaged_denominator,
            effective_denominator=effective_denominator,
            unconstrained_coefficient=unconstrained,
            zero_denominator_count=zero_count,
            regularization_activity_count=regularization_count,
            backscatter_activity_count=backscatter_count,
            backscatter_limit_count=backscatter_limit_count,
            accepted_update_count=accepted_count,
            rejected_update_count=rejected_count,
            finite=finite,
            averaging=self.averaging.name,
            regularization=self.regularization.name,
            backscatter=self.backscatter.name,
            differentiation=self.differentiation,
            dynamic_provenance_id=self.provenance.provenance_id,
            prepared_id=self.prepared_id,
        )
        return DynamicLESResult(
            coefficient=coefficient,
            prepared_algebraic_stress=algebraic_result,
            evidence=evidence,
            continuation_state=continuation_state,
        )

    def _validate_inputs(self, inputs: DynamicLESInputs, /) -> None:
        if not isinstance(inputs, DynamicLESInputs):
            raise TypeError("inputs must be DynamicLESInputs.")
        if inputs.provenance.provenance_id != self.provenance.provenance_id:
            raise ValueError("Dynamic LES input and prepared filter provenance differ.")

    def _continuation_id(self, shape: tuple[int, ...], /) -> str:
        return canonical_fingerprint(
            {
                "kind": "dynamic-les-continuation",
                "prepared_dynamic_model": self.prepared_id,
                "field_shape": list(shape),
            }
        )


def _inexact_array(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.result_type(array, float))
    return array


def _is_concrete(value: Array, /) -> bool:
    return not isinstance(value, jax.core.Tracer)


def _require_finite(value: Array, name: str, /) -> None:
    if _is_concrete(value) and np.any(~np.isfinite(np.asarray(value))):
        raise ValueError(f"{name} must be finite.")


def _deviatoric(tensor: Array, /) -> Array:
    trace = jnp.trace(tensor, axis1=-2, axis2=-1)
    identity = jnp.eye(3, dtype=tensor.dtype)
    return tensor - trace[..., None, None] * identity / 3.0


def _count(mask: Array, /) -> Array:
    return jnp.sum(mask.astype(jnp.int32))


def _all_finite(*values: Array) -> Array:
    result = jnp.asarray(True)
    for value in values:
        result = result & jnp.all(jnp.isfinite(value))
    return result


def _positive_square_root(value: Array, /) -> Array:
    active = value > 0.0
    safe = jnp.where(active, value, jnp.ones_like(value))
    return jnp.where(active, jnp.sqrt(safe), jnp.zeros_like(value))


def _evaluate_dynamic_stress(
    coefficient: Array, inputs: AlgebraicLESInputs, /
) -> AlgebraicLESResult:
    gradient = inputs.velocity_gradient
    strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
    trace = jnp.trace(strain, axis1=-2, axis2=-1)
    identity = jnp.eye(3, dtype=gradient.dtype)
    deviatoric_strain = strain - trace[..., None, None] * identity / 3.0
    strain_squared = ein.contract("...ij,...ij->...", strain, strain, backend="jax")
    magnitude = _positive_square_root(2.0 * strain_squared)
    width = inputs.filter_scale.equivalent_width
    kinematic_viscosity = coefficient * width * width * magnitude
    stress = -2.0 * kinematic_viscosity[..., None, None] * deviatoric_strain
    energy_transfer = -ein.contract("...ij,...ij->...", stress, strain, backend="jax")
    return AlgebraicLESResult(
        kinematic_viscosity=kinematic_viscosity,
        specific_deviatoric_stress=stress,
        energy_transfer=energy_transfer,
    )


def _reject_state(
    state: LagrangianDynamicLESState | None, averaging_name: str, /
) -> None:
    if state is not None:
        raise TypeError(f"{averaging_name} averaging does not accept continuation state.")


def _policy_id(kind: str, payload: dict[str, object], /) -> str:
    return canonical_fingerprint({"kind": kind, **payload})


__all__ = [
    "AbstractBackscatterPolicy",
    "AbstractDenominatorRegularization",
    "AbstractDynamicLESAveraging",
    "AdditiveDenominatorRegularization",
    "AllowSignedBackscatter",
    "BoundedFractionBackscatter",
    "DynamicLESInputs",
    "DynamicLESProvenance",
    "DynamicLESResult",
    "DynamicSmagorinskyPlan",
    "ExactDenominatorRegularization",
    "GermanoLeastSquaresEvidence",
    "GlobalDynamicLESAveraging",
    "HomogeneousPlaneDynamicLESAveraging",
    "LagrangianDynamicLESAveraging",
    "LagrangianDynamicLESState",
    "LocalKernelDynamicLESAveraging",
    "NonnegativeBackscatterClip",
    "PreparedDynamicSmagorinskyPlan",
]
