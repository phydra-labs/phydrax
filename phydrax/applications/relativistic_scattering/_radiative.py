#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite scalar one-loop and local real/virtual subtraction references."""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...integration._rules import GaussLegendreRule, ReferenceIntervalRule


class RadiativeStatus(IntEnum):
    """Status for finite loop and subtraction evaluations."""

    SUCCESS = 0
    NONFINITE = 1
    INVALID_KINEMATICS = 2


class ScalarBubblePlan(StrictModule, NonTrainableState):
    """Fixed Gauss-Legendre representation of a finite scalar bubble."""

    nodes: Array
    weights: Array
    renormalization_scale_squared: Array
    epsilon: Array
    order: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        order: int = 96,
        renormalization_scale_squared: float = 1.0,
        epsilon: float = 1.0e-15,
        max_points: int = 256,
    ):
        order_ = int(order)
        maximum = int(max_points)
        scale = float(renormalization_scale_squared)
        epsilon_ = float(epsilon)
        if order_ < 2 or order_ > maximum or maximum < 2:
            raise ValueError("Scalar bubble quadrature exceeds its fixed point guard.")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("Renormalization scale squared must be finite and positive.")
        if not math.isfinite(epsilon_) or epsilon_ <= 0.0:
            raise ValueError("Feynman epsilon must be finite and positive.")
        data = ReferenceIntervalRule(GaussLegendreRule(order_)).materialize()
        self.nodes = data.points[:, 0]
        self.weights = data.weights
        self.renormalization_scale_squared = jnp.asarray(scale)
        self.epsilon = jnp.asarray(epsilon_)
        self.order = order_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-scalar-bubble",
                "order": order_,
                "renormalization_scale_squared": scale,
                "epsilon": epsilon_,
                "max_points": maximum,
            }
        )


class ScalarBubbleResult(StrictModule):
    """Finite MS-bar scalar bubble convention and numerical evidence."""

    value: Array
    quadrature_values: Array
    status: Array
    minimum_denominator_magnitude: Array
    plan_id: str = eqx.field(static=True)
    normalization: str = eqx.field(static=True)


def finite_scalar_bubble(
    invariant: ArrayLike,
    mass_one_squared: ArrayLike,
    mass_two_squared: ArrayLike,
    plan: ScalarBubblePlan,
    /,
) -> ScalarBubbleResult:
    r"""Evaluate ``-1/(16 pi^2) int_0^1 log(Delta(x)/mu^2) dx``."""
    invariant_, mass_one, mass_two = jnp.broadcast_arrays(
        jnp.asarray(invariant),
        jnp.asarray(mass_one_squared),
        jnp.asarray(mass_two_squared),
    )
    trailing = (1,) * invariant_.ndim
    x = plan.nodes.reshape((plan.order,) + trailing)
    weights = plan.weights.reshape((plan.order,) + trailing)
    delta = (
        x * mass_one
        + (1.0 - x) * mass_two
        - x * (1.0 - x) * invariant_
        - 1.0j * plan.epsilon
    )
    values = -jnp.log(delta / plan.renormalization_scale_squared) / (16.0 * jnp.pi**2)
    result = jnp.sum(weights * values, axis=0)
    valid_input = (mass_one >= 0.0) & (mass_two >= 0.0)
    finite = jnp.all(jnp.isfinite(values), axis=0) & jnp.isfinite(result)
    status = jnp.where(
        ~valid_input,
        int(RadiativeStatus.INVALID_KINEMATICS),
        jnp.where(
            finite,
            int(RadiativeStatus.SUCCESS),
            int(RadiativeStatus.NONFINITE),
        ),
    )
    return ScalarBubbleResult(
        result,
        values,
        status.astype(jnp.int32),
        jnp.min(jnp.abs(delta), axis=0),
        plan.plan_id,
        "finite-msbar-b0-over-16pi-squared",
    )


class RealVirtualSubtractionPlan(StrictModule, NonTrainableState):
    """Fixed quadrature and slicing cutoff for a plus-distribution reference."""

    nodes: Array
    weights: Array
    cutoff: Array
    order: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        cutoff: float = 1.0e-6,
        order: int = 128,
        max_points: int = 256,
    ):
        cutoff_ = float(cutoff)
        order_ = int(order)
        maximum = int(max_points)
        if not math.isfinite(cutoff_) or cutoff_ <= 0.0 or cutoff_ >= 1.0:
            raise ValueError("Subtraction cutoff must lie strictly between zero and one.")
        if order_ < 2 or order_ > maximum or maximum < 2:
            raise ValueError("Subtraction quadrature exceeds its fixed point guard.")
        data = ReferenceIntervalRule(GaussLegendreRule(order_)).materialize()
        self.nodes = data.points[:, 0]
        self.weights = data.weights
        self.cutoff = jnp.asarray(cutoff_)
        self.order = order_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "real-virtual-plus-subtraction",
                "cutoff": cutoff_,
                "order": order_,
                "max_points": maximum,
                "kernel": "(1+z^2)/(1-z)",
            }
        )


class RealVirtualSubtractionResult(StrictModule):
    """Separately divergent-like pieces and their finite cancellation evidence."""

    real: Array
    local_counterterm: Array
    virtual: Array
    subtracted_real: Array
    unresolved_remainder: Array
    total: Array
    cancellation_residual: Array
    status: Array
    cutoff: Array
    plan_id: str = eqx.field(static=True)
    prescription: str = eqx.field(static=True)


def qed_splitting_kernel(z: ArrayLike, /) -> Array:
    """Unregularized fermion-to-fermion QED splitting kernel."""
    z_ = jnp.asarray(z)
    return (1.0 + z_**2) / (1.0 - z_)


def real_virtual_subtraction(
    measurement: Callable[[Array], Array],
    plan: RealVirtualSubtractionPlan,
    /,
    *,
    finite_virtual: ArrayLike = 0.0,
) -> RealVirtualSubtractionResult:
    """Apply local plus subtraction and exhibit exact real/virtual cancellation."""
    endpoint = jnp.asarray(measurement(jnp.asarray(1.0)))
    z = (1.0 - plan.cutoff) * plan.nodes
    weights = (1.0 - plan.cutoff) * plan.weights
    measured = jnp.asarray(measurement(z))
    if measured.shape[:1] != z.shape:
        raise ValueError(
            "Subtraction measurements must preserve the quadrature node axis."
        )
    kernel = qed_splitting_kernel(z)
    output_shape = (z.shape[0],) + (1,) * (measured.ndim - 1)
    weighted_kernel = (weights * kernel).reshape(output_shape)
    real = jnp.sum(weighted_kernel * measured, axis=0)
    counterterm = jnp.sum(weighted_kernel * endpoint, axis=0)
    subtracted = jnp.sum(weighted_kernel * (measured - endpoint), axis=0)
    tail_z = 1.0 - plan.cutoff + plan.cutoff * plan.nodes
    tail_weights = plan.cutoff * plan.weights
    tail_measured = jnp.asarray(measurement(tail_z))
    tail_shape = (tail_z.shape[0],) + (1,) * (tail_measured.ndim - 1)
    unresolved = jnp.sum(
        (tail_weights * qed_splitting_kernel(tail_z)).reshape(tail_shape)
        * (tail_measured - endpoint),
        axis=0,
    )
    virtual = -counterterm + unresolved + jnp.asarray(finite_virtual) * endpoint
    total = real + virtual
    reference = subtracted + unresolved + jnp.asarray(finite_virtual) * endpoint
    residual = jnp.max(jnp.abs(total - reference))
    finite = (
        jnp.all(jnp.isfinite(measured))
        & jnp.all(jnp.isfinite(tail_measured))
        & jnp.all(jnp.isfinite(real))
        & jnp.all(jnp.isfinite(virtual))
        & jnp.all(jnp.isfinite(total))
    )
    status = jnp.where(
        finite, int(RadiativeStatus.SUCCESS), int(RadiativeStatus.NONFINITE)
    )
    return RealVirtualSubtractionResult(
        real,
        counterterm,
        virtual,
        subtracted,
        unresolved,
        total,
        residual,
        status.astype(jnp.int32),
        plan.cutoff,
        plan.plan_id,
        "local-qed-plus-distribution",
    )


__all__ = [
    "RadiativeStatus",
    "RealVirtualSubtractionPlan",
    "RealVirtualSubtractionResult",
    "ScalarBubblePlan",
    "ScalarBubbleResult",
    "finite_scalar_bubble",
    "qed_splitting_kernel",
    "real_virtual_subtraction",
]
