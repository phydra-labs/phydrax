#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import zeta
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ModeSumRegularizationParameters(StrictModule, NonTrainableState):
    """First-order one-sided mode-sum subtraction parameters.

    For ``L = ell + 1/2``, supplied retarded modes are regularized as
    ``F_ret[ell] - A L - B - C/L`` and the finite parameter ``D`` is subtracted
    once after summation.  The parameters must be supplied in a declared gauge,
    worldline, component basis, and radial-side limit; they are not universal.
    """

    A: Array
    B: Array
    C: Array
    D: Array
    side: str = eqx.field(static=True)
    gauge: str = eqx.field(static=True)
    worldline_id: str = eqx.field(static=True)
    component_basis: str = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        A,
        B,
        C,
        D,
        /,
        *,
        side,
        gauge,
        worldline_id,
        component_basis,
    ):
        values = [np.asarray(value, dtype=float) for value in (A, B, C, D)]
        if any(value.shape != values[0].shape for value in values[1:]) or any(
            np.any(~np.isfinite(value)) for value in values
        ):
            raise ValueError("Mode-sum A, B, C, and D must be finite matching arrays.")
        side_value = str(side).strip().lower()
        if side_value not in ("plus", "minus", "average"):
            raise ValueError("side must be plus, minus, or average.")
        labels = tuple(
            str(value).strip() for value in (gauge, worldline_id, component_basis)
        )
        if not all(labels):
            raise ValueError("Gauge, worldline, and component-basis labels are required.")
        self.A = jnp.asarray(values[0])
        self.B = jnp.asarray(values[1])
        self.C = jnp.asarray(values[2])
        self.D = jnp.asarray(values[3])
        self.side = side_value
        self.gauge, self.worldline_id, self.component_basis = labels
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "first-order-mode-sum-regularization",
                "content": array_tree_fingerprint(
                    {"A": values[0], "B": values[1], "C": values[2], "D": values[3]}
                ),
                "side": side_value,
                "gauge": labels[0],
                "worldline_id": labels[1],
                "component_basis": labels[2],
            }
        )


class ModeSumTailEvidence(StrictModule):
    fitted_coefficients: Array
    fitted_modes: Array
    fit_residual: Array
    relative_fit_residual: Array
    tail_correction: Array
    tail_fraction: Array
    normal_matrix_determinant: Array
    fit_start_ell: Array
    finite: Array
    fitted: Array
    qualified: Array
    derivative_valid: Array
    asymptotic_model: str = eqx.field(static=True)
    calculator_id: str = eqx.field(static=True)


class FirstOrderSelfForceResult(StrictModule):
    retarded_modes: Array
    regularized_modes: Array
    finite_partial_sum: Array
    self_force: Array
    regularization: ModeSumRegularizationParameters
    tail: ModeSumTailEvidence
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    status: Array
    calculator_id: str = eqx.field(static=True)
    theory_scope: str = eqx.field(static=True)


class FirstOrderSelfForceModeSum(StrictModule, NonTrainableState):
    """Fixed-capacity first-order regularized self-force mode-sum calculator.

    The unresolved tail is fit to ``c2/L^2 + c4/L^4`` over a fixed high-ell
    window and summed analytically with Hurwitz zeta functions.  Qualification
    exposes both the fit defect and the extrapolated tail fraction.  This class
    does not construct regularization parameters or claim second-order
    self-force validity.
    """

    regularization: ModeSumRegularizationParameters
    ell_max: int = eqx.field(static=True)
    tail_window: int = eqx.field(static=True)
    tail_fit_tolerance: float = eqx.field(static=True)
    maximum_tail_fraction: float = eqx.field(static=True)
    calculator_id: str = eqx.field(static=True)

    def __init__(
        self,
        regularization: ModeSumRegularizationParameters,
        ell_max,
        /,
        *,
        tail_window=6,
        tail_fit_tolerance=5.0e-3,
        maximum_tail_fraction=0.25,
    ):
        if not isinstance(regularization, ModeSumRegularizationParameters):
            raise TypeError("regularization must be ModeSumRegularizationParameters.")
        ell_max_value = int(ell_max)
        tail_window_value = int(tail_window)
        if ell_max_value < 3 or not 3 <= tail_window_value <= ell_max_value + 1:
            raise ValueError(
                "ell_max must be at least three and tail_window must fit the "
                "mode capacity."
            )
        if (
            not np.isfinite(tail_fit_tolerance)
            or tail_fit_tolerance <= 0.0
            or not np.isfinite(maximum_tail_fraction)
            or maximum_tail_fraction <= 0.0
        ):
            raise ValueError("Tail qualification tolerances must be positive.")
        self.regularization = regularization
        self.ell_max = ell_max_value
        self.tail_window = tail_window_value
        self.tail_fit_tolerance = float(tail_fit_tolerance)
        self.maximum_tail_fraction = float(maximum_tail_fraction)
        self.calculator_id = canonical_fingerprint(
            {
                "kind": "first-order-regularized-self-force-mode-sum",
                "regularization": regularization.parameter_id,
                "ell_max": ell_max_value,
                "tail_window": tail_window_value,
                "tail_fit_tolerance": float(tail_fit_tolerance),
                "maximum_tail_fraction": float(maximum_tail_fraction),
                "tail_model": "c2/L^2+c4/L^4",
            }
        )

    def calculate(self, retarded_modes: ArrayLike, /) -> FirstOrderSelfForceResult:
        modes = jnp.asarray(retarded_modes)
        component_shape = self.regularization.A.shape
        expected_shape = (self.ell_max + 1, *component_shape)
        if modes.shape != expected_shape:
            raise ValueError(
                f"retarded_modes must have fixed shape {expected_shape}, "
                f"got {modes.shape}."
            )
        ell = jnp.arange(self.ell_max + 1, dtype=modes.real.dtype)
        angular_order = ell + 0.5
        broadcast_shape = (self.ell_max + 1,) + (1,) * len(component_shape)
        angular_order_broadcast = angular_order.reshape(broadcast_shape)
        regularized = (
            modes
            - angular_order_broadcast * self.regularization.A
            - self.regularization.B
            - self.regularization.C / angular_order_broadcast
        )
        finite_partial_sum = jnp.sum(regularized, axis=0) - self.regularization.D

        tail_orders = jnp.asarray((2.0, 4.0), dtype=modes.real.dtype)
        tail_angular_order = angular_order[-self.tail_window :]
        design = tail_angular_order[:, None] ** (-tail_orders[None, :])
        flattened = regularized[-self.tail_window :].reshape(self.tail_window, -1)
        normal = contract("lp,lq->pq", design, design)
        right_hand_side = contract("lp,lc->pc", design, flattened)
        determinant = normal[0, 0] * normal[1, 1] - normal[0, 1] * normal[1, 0]
        coefficient_2 = (
            normal[1, 1] * right_hand_side[0] - normal[0, 1] * right_hand_side[1]
        ) / determinant
        coefficient_4 = (
            normal[0, 0] * right_hand_side[1] - normal[1, 0] * right_hand_side[0]
        ) / determinant
        coefficient_flat = jnp.stack((coefficient_2, coefficient_4))
        fitted_flat = contract("lp,pc->lc", design, coefficient_flat)
        fit_defect_flat = flattened - fitted_flat
        fit_scale = jnp.maximum(
            jnp.sqrt(jnp.mean(jnp.abs(flattened) ** 2, axis=0)), 1.0e-30
        )
        fit_residual_flat = jnp.sqrt(jnp.mean(jnp.abs(fit_defect_flat) ** 2, axis=0))
        relative_fit_flat = fit_residual_flat / fit_scale

        first_omitted_order = jnp.asarray(self.ell_max + 1.5, dtype=modes.real.dtype)
        zeta_values = jnp.asarray(
            (zeta(2.0, first_omitted_order), zeta(4.0, first_omitted_order))
        )
        tail_flat = contract("p,pc->c", zeta_values, coefficient_flat)
        tail_correction = tail_flat.reshape(component_shape)
        self_force = finite_partial_sum + tail_correction
        coefficient = coefficient_flat.reshape((2, *component_shape))
        fitted_modes = fitted_flat.reshape((self.tail_window, *component_shape))
        fit_residual = fit_residual_flat.reshape(component_shape)
        relative_fit = relative_fit_flat.reshape(component_shape)
        tail_fraction = _component_norm(tail_correction) / jnp.maximum(
            _component_norm(self_force), 1.0e-30
        )
        finite = jnp.all(
            jnp.isfinite(
                jnp.concatenate(
                    (
                        jnp.ravel(jnp.real(regularized)),
                        jnp.ravel(jnp.imag(regularized)),
                        jnp.ravel(jnp.real(self_force)),
                        jnp.ravel(jnp.imag(self_force)),
                        jnp.ravel(relative_fit),
                    )
                )
            )
        )
        fitted = jnp.abs(determinant) > 1.0e-30
        fit_qualified = jnp.all(relative_fit <= self.tail_fit_tolerance)
        fraction_qualified = tail_fraction <= self.maximum_tail_fraction
        tail_qualified = finite & fitted & fit_qualified & fraction_qualified
        derivative_valid = finite & fitted & tail_qualified
        tail_evidence = ModeSumTailEvidence(
            coefficient,
            fitted_modes,
            fit_residual,
            relative_fit,
            tail_correction,
            tail_fraction,
            determinant,
            jnp.asarray(self.ell_max + 1 - self.tail_window, dtype=jnp.int32),
            finite,
            fitted,
            tail_qualified,
            derivative_valid,
            "even inverse powers c2/L^2+c4/L^4 with Hurwitz-zeta remainder",
            self.calculator_id,
        )
        physically_valid = finite & fitted
        converged = tail_qualified
        qualified = physically_valid & converged & derivative_valid
        status = jnp.where(
            ~finite,
            4,
            jnp.where(
                ~fitted,
                3,
                jnp.where(~fit_qualified, 2, jnp.where(~fraction_qualified, 1, 0)),
            ),
        ).astype(jnp.int32)
        return FirstOrderSelfForceResult(
            modes,
            regularized,
            finite_partial_sum,
            self_force,
            self.regularization,
            tail_evidence,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            status,
            self.calculator_id,
            "first-order point-particle mode sum in the declared gauge and side limit",
        )


def _component_norm(value):
    return jnp.sqrt(jnp.sum(jnp.abs(value) ** 2))


__all__ = [
    "FirstOrderSelfForceModeSum",
    "FirstOrderSelfForceResult",
    "ModeSumRegularizationParameters",
    "ModeSumTailEvidence",
]
