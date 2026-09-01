#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ...nonlinear import NonlinearTermination
from ...operators.mechanics import (
    finite_strain_kinematics,
    FiniteStrainKinematics,
    HyperelasticLaw,
)


_PLANE_STRESS_LINEAR_SOLVE = SmallLinearSolvePlan(2)
_DEFAULT_LOG_STRETCH_BOUNDS = (-8.0, 8.0)
_DEFAULT_PRESSURE_BOUNDS = (-1.0e12, 1.0e12)


def _determinant_2d(value: Array, /) -> Array:
    return value[..., 0, 0] * value[..., 1, 1] - value[..., 0, 1] * value[..., 1, 0]


class PlaneStressFailure(IntEnum):
    """Terminal evidence for a local plane-stress reduction."""

    OK = 0
    INVALID_INPUT = 1
    NO_BRACKET = 2
    MAX_STEPS = 3
    SINGULAR_TANGENT = 4
    NONFINITE = 5
    BASE_LAW_REJECTED = 6


class PlaneStressKinematics(StrictModule):
    """Block-diagonal three-dimensional kinematics with explicit thickness data."""

    in_plane_deformation_gradient: Array
    deformation_gradient: Array
    finite_strain: FiniteStrainKinematics
    log_thickness_stretch: Array
    thickness_stretch: Array
    reference_thickness: Array
    current_thickness: Array
    admissible: Array
    dimension: int = eqx.field(static=True)
    kinematics: str = eqx.field(static=True)

    def __init__(
        self,
        deformation_gradient: ArrayLike,
        log_thickness_stretch: ArrayLike,
        reference_thickness: ArrayLike = 1.0,
        /,
    ):
        deformation = jnp.asarray(deformation_gradient)
        if deformation.shape[-2:] != (2, 2):
            raise ValueError("Plane-stress deformation gradients must end in 2x2.")
        if not jnp.issubdtype(deformation.dtype, jnp.inexact):
            deformation = deformation.astype(jnp.result_type(deformation.dtype, float))
        batch_shape = deformation.shape[:-2]
        eta = jnp.asarray(log_thickness_stretch, dtype=deformation.dtype)
        thickness = jnp.asarray(reference_thickness, dtype=deformation.dtype)
        try:
            eta = jnp.broadcast_to(eta, batch_shape)
            thickness = jnp.broadcast_to(thickness, batch_shape)
        except ValueError as error:
            raise ValueError(
                "Plane-stress thickness data must broadcast to the deformation batch."
            ) from error
        stretch = jnp.exp(eta)
        embedded = jnp.zeros(batch_shape + (3, 3), dtype=deformation.dtype)
        embedded = embedded.at[..., :2, :2].set(deformation)
        embedded = embedded.at[..., 2, 2].set(stretch)
        finite_strain = finite_strain_kinematics(embedded)
        admissible = (
            finite_strain.admissible
            & jnp.isfinite(eta)
            & jnp.isfinite(stretch)
            & jnp.isfinite(thickness)
            & (stretch > 0.0)
            & (thickness > 0.0)
        )
        self.in_plane_deformation_gradient = deformation
        self.deformation_gradient = embedded
        self.finite_strain = finite_strain
        self.log_thickness_stretch = eta
        self.thickness_stretch = stretch
        self.reference_thickness = thickness
        self.current_thickness = thickness * stretch
        self.admissible = admissible
        self.dimension = 2
        self.kinematics = "plane_stress"


class BlockDiagonalPlaneStressReductionResponse(StrictModule):
    """Reduced membrane response with root, implicit-derivative, and failure evidence."""

    kinematics: PlaneStressKinematics
    reference_energy_density: Array
    first_piola: Array
    cauchy_stress: Array
    condensed_tangent: Array
    log_stretch_sensitivity: Array
    residual: Array
    bracket_residual: Array
    successful: Array
    failure: Array


class BlockDiagonalPlaneStressReductionPlan(StrictModule, NonTrainableState):
    """Safeguarded scalar ``P33 = 0`` reduction for block-diagonal kinematics.

    ``root_policy=None`` selects a bracket-certified safeguarded Newton solve.  It
    never selects an open or unguarded Newton iteration.  The returned energy,
    first Piola stress, and condensed tangent are reference-area quantities and
    therefore scale with the explicit reference thickness ``h0``.
    """

    root_policy: NonlinearTermination
    log_stretch_bounds: tuple[float, float] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        root_policy: NonlinearTermination | None = None,
        log_stretch_bounds: tuple[float, float] = _DEFAULT_LOG_STRETCH_BOUNDS,
    ):
        policy = (
            NonlinearTermination(
                absolute_residual=1.0e-10,
                relative_residual=1.0e-10,
                absolute_step=1.0e-12,
                relative_step=1.0e-10,
                maximum_steps=50,
            )
            if root_policy is None
            else root_policy
        )
        if not isinstance(policy, NonlinearTermination):
            raise TypeError("root_policy must be NonlinearTermination or None.")
        lower, upper = _ordered_finite_bounds(
            log_stretch_bounds, name="log_stretch_bounds"
        )
        self.root_policy = policy
        self.log_stretch_bounds = (lower, upper)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "block-diagonal-plane-stress-reduction",
                "log_stretch_bounds": (lower, upper),
                "absolute_residual": policy.absolute_residual,
                "relative_residual": policy.relative_residual,
                "absolute_step": policy.absolute_step,
                "relative_step": policy.relative_step,
                "maximum_steps": policy.maximum_steps,
            }
        )

    def evaluate(
        self,
        deformation_gradient: ArrayLike,
        law: HyperelasticLaw,
        /,
        *,
        reference_thickness: ArrayLike = 1.0,
    ) -> BlockDiagonalPlaneStressReductionResponse:
        if not isinstance(law, HyperelasticLaw):
            raise TypeError(
                "law must be a pure HyperelasticLaw; mixed pressure laws require "
                "CoupledPlaneStressIncompressiblePlan."
            )
        deformation = jnp.asarray(deformation_gradient)
        if deformation.shape[-2:] != (2, 2):
            raise ValueError("Plane-stress deformation gradients must end in 2x2.")
        if not jnp.issubdtype(deformation.dtype, jnp.inexact):
            deformation = deformation.astype(jnp.result_type(deformation.dtype, float))
        batch_shape = deformation.shape[:-2]
        thickness = jnp.asarray(reference_thickness, dtype=deformation.dtype)
        try:
            thickness = jnp.broadcast_to(thickness, batch_shape)
        except ValueError as error:
            raise ValueError(
                "reference_thickness must broadcast to the deformation batch."
            ) from error
        flat_deformation = deformation.reshape((-1, 2, 2))
        flat_thickness = thickness.reshape((-1,))
        outputs = jax.vmap(lambda value, h0: self._evaluate_point(value, law, h0))(
            flat_deformation,
            flat_thickness,
        )
        (
            eta,
            energy,
            first_piola,
            cauchy,
            tangent,
            sensitivity,
            residual,
            bracket_residual,
            successful,
            failure,
        ) = outputs
        kinematics = PlaneStressKinematics(
            deformation,
            eta.reshape(batch_shape),
            thickness,
        )
        return BlockDiagonalPlaneStressReductionResponse(
            kinematics,
            energy.reshape(batch_shape),
            first_piola.reshape(batch_shape + (2, 2)),
            cauchy.reshape(batch_shape + (2, 2)),
            tangent.reshape(batch_shape + (2, 2, 2, 2)),
            sensitivity.reshape(batch_shape + (2, 2)),
            residual.reshape(batch_shape),
            bracket_residual.reshape(batch_shape + (2,)),
            successful.reshape(batch_shape),
            failure.reshape(batch_shape),
        )

    def _evaluate_point(self, deformation: Array, law: HyperelasticLaw, h0: Array):
        lower = jnp.asarray(self.log_stretch_bounds[0], dtype=deformation.dtype)
        upper = jnp.asarray(self.log_stretch_bounds[1], dtype=deformation.dtype)

        def embedded(eta):
            value = jnp.zeros((3, 3), dtype=deformation.dtype)
            value = value.at[:2, :2].set(deformation)
            return value.at[2, 2].set(jnp.exp(eta))

        def residual(eta):
            return law.evaluate(embedded(eta)).first_piola[2, 2]

        def solve(function, initial):
            lower_residual = function(lower)
            upper_residual = function(upper)
            initial_residual = function(initial)
            initial_converged = jnp.isfinite(initial_residual) & (
                jnp.abs(initial_residual) <= self.root_policy.absolute_residual
            )
            bracketed = (
                jnp.isfinite(lower_residual)
                & jnp.isfinite(upper_residual)
                & (lower_residual * upper_residual <= 0.0)
            )

            def body(_, carry):
                left, right, left_residual, state = carry
                midpoint = 0.5 * (left + right)
                midpoint_residual = function(midpoint)
                same_side = left_residual * midpoint_residual > 0.0
                next_left = jnp.where(same_side, midpoint, left)
                next_right = jnp.where(same_side, right, midpoint)
                next_left_residual = jnp.where(
                    same_side, midpoint_residual, left_residual
                )
                return next_left, next_right, next_left_residual, midpoint

            _, _, _, root = jax.lax.fori_loop(
                0,
                self.root_policy.maximum_steps,
                body,
                (lower, upper, lower_residual, initial),
            )
            return jnp.where(
                initial_converged,
                initial,
                jnp.where(bracketed, root, initial),
            )

        def tangent_solve(linearized, right_hand_side):
            derivative = jax.grad(linearized)(jnp.zeros_like(right_hand_side))
            safe = jnp.where(jnp.abs(derivative) > 0.0, derivative, 1.0)
            return right_hand_side / safe

        initial = jnp.clip(jnp.asarray(0.0, dtype=deformation.dtype), lower, upper)
        eta = jax.lax.custom_root(residual, initial, solve, tangent_solve)
        lower_response = law.evaluate(embedded(lower))
        upper_response = law.evaluate(embedded(upper))
        lower_residual = lower_response.first_piola[2, 2]
        upper_residual = upper_response.first_piola[2, 2]
        bracket_residual = jnp.stack((lower_residual, upper_residual))
        bracket_finite = jnp.all(jnp.isfinite(bracket_residual))
        bracketed = bracket_finite & (lower_residual * upper_residual <= 0.0)

        base = law.evaluate(embedded(eta))
        root_residual = base.first_piola[2, 2]
        thickness_stretch = jnp.exp(eta)
        tangent = base.tangent
        denominator = tangent[2, 2, 2, 2] * thickness_stretch
        sensitivity = -tangent[2, 2, :2, :2] / jnp.where(
            jnp.abs(denominator) > 0.0,
            denominator,
            1.0,
        )
        condensed = tangent[:2, :2, :2, :2] + (
            tangent[:2, :2, 2, 2][..., None, None]
            * thickness_stretch
            * sensitivity[None, None, ...]
        )
        scaled_energy = h0 * base.reference_energy_density
        scaled_first_piola = h0 * base.first_piola[:2, :2]
        scaled_tangent = h0 * condensed
        cauchy = base.cauchy_stress[:2, :2]

        determinant = _determinant_2d(deformation)
        input_valid = (
            jnp.all(jnp.isfinite(deformation))
            & jnp.isfinite(h0)
            & (h0 > 0.0)
            & jnp.isfinite(determinant)
            & (determinant > 0.0)
        )
        endpoint_law_valid = lower_response.admissible & upper_response.admissible
        base_valid = base.admissible
        finite_root = (
            jnp.isfinite(eta)
            & jnp.isfinite(root_residual)
            & jnp.isfinite(denominator)
            & jnp.all(jnp.isfinite(condensed))
        )
        initial_norm = jnp.maximum(
            jnp.minimum(jnp.abs(lower_residual), jnp.abs(upper_residual)),
            jnp.asarray(1.0e-30, dtype=deformation.dtype),
        )
        tolerance = self.root_policy.residual_threshold(initial_norm)
        converged = jnp.abs(root_residual) <= tolerance
        tangent_valid = jnp.abs(denominator) > jnp.asarray(
            1.0e-12, dtype=deformation.dtype
        ) * jnp.maximum(jnp.max(jnp.abs(tangent)), 1.0)
        failure = jnp.where(
            ~input_valid,
            int(PlaneStressFailure.INVALID_INPUT),
            jnp.where(
                ~endpoint_law_valid,
                int(PlaneStressFailure.BASE_LAW_REJECTED),
                jnp.where(
                    ~bracket_finite,
                    int(PlaneStressFailure.NONFINITE),
                    jnp.where(
                        ~bracketed,
                        int(PlaneStressFailure.NO_BRACKET),
                        jnp.where(
                            ~base_valid,
                            int(PlaneStressFailure.BASE_LAW_REJECTED),
                            jnp.where(
                                ~finite_root,
                                int(PlaneStressFailure.NONFINITE),
                                jnp.where(
                                    ~converged,
                                    int(PlaneStressFailure.MAX_STEPS),
                                    jnp.where(
                                        ~tangent_valid,
                                        int(PlaneStressFailure.SINGULAR_TANGENT),
                                        int(PlaneStressFailure.OK),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        successful = failure == int(PlaneStressFailure.OK)
        scaled_energy = jnp.where(successful, scaled_energy, jnp.nan)
        scaled_first_piola = jnp.where(successful, scaled_first_piola, jnp.nan)
        cauchy = jnp.where(successful, cauchy, jnp.nan)
        scaled_tangent = jnp.where(successful, scaled_tangent, jnp.nan)
        sensitivity = jnp.where(successful, sensitivity, jnp.nan)
        return (
            eta,
            scaled_energy,
            scaled_first_piola,
            cauchy,
            scaled_tangent,
            sensitivity,
            root_residual,
            bracket_residual,
            successful,
            failure,
        )


class CoupledPlaneStressIncompressibleResponse(StrictModule):
    """Coupled thickness-pressure response with condensed in-plane tangent evidence."""

    thickness_stretch: Array
    pressure: Array
    residual: Array
    first_piola: Array
    condensed_tangent: Array
    successful: Array
    failure: Array


class CoupledPlaneStressIncompressiblePlan(StrictModule, NonTrainableState):
    """Bounded two-unknown reduction of ``P33`` and incompressibility.

    Exact closure solves ``[P33, g(F)] = 0``.  With finite ``bulk_modulus`` it
    solves ``[P33, g(F) - p/K] = 0``.  This is a genuinely coupled solve and is
    never routed through the scalar plane-stress reduction.
    """

    root_policy: NonlinearTermination
    log_stretch_bounds: tuple[float, float] = eqx.field(static=True)
    pressure_bounds: tuple[float, float] = eqx.field(static=True)
    volumetric_constraint: Callable[[Array], Array] = eqx.field(static=True)
    bulk_modulus: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        root_policy: NonlinearTermination | None = None,
        log_stretch_bounds: tuple[float, float] = _DEFAULT_LOG_STRETCH_BOUNDS,
        pressure_bounds: tuple[float, float] = _DEFAULT_PRESSURE_BOUNDS,
        volumetric_constraint: Callable[[Array], Array] | None = None,
        bulk_modulus: float | None = None,
    ):
        if volumetric_constraint is None or not callable(volumetric_constraint):
            raise TypeError("volumetric_constraint must be callable.")
        policy = (
            NonlinearTermination(
                absolute_residual=1.0e-10,
                relative_residual=1.0e-10,
                absolute_step=1.0e-12,
                relative_step=1.0e-10,
                maximum_steps=50,
            )
            if root_policy is None
            else root_policy
        )
        if not isinstance(policy, NonlinearTermination):
            raise TypeError("root_policy must be NonlinearTermination or None.")
        eta_bounds = _ordered_finite_bounds(log_stretch_bounds, name="log_stretch_bounds")
        pressure_bounds_ = _ordered_finite_bounds(pressure_bounds, name="pressure_bounds")
        bulk = None if bulk_modulus is None else float(bulk_modulus)
        if bulk is not None and (not np.isfinite(bulk) or bulk <= 0.0):
            raise ValueError("bulk_modulus must be a positive finite scalar or None.")
        self.root_policy = policy
        self.log_stretch_bounds = eta_bounds
        self.pressure_bounds = pressure_bounds_
        self.volumetric_constraint = volumetric_constraint
        self.bulk_modulus = bulk
        self.plan_id = canonical_fingerprint(
            {
                "kind": "coupled-plane-stress-incompressibility",
                "log_stretch_bounds": eta_bounds,
                "pressure_bounds": pressure_bounds_,
                "bulk_modulus": bulk,
                "maximum_steps": policy.maximum_steps,
                "absolute_residual": policy.absolute_residual,
                "relative_residual": policy.relative_residual,
            }
        )

    def evaluate(
        self,
        deformation_gradient: ArrayLike,
        law: Any,
        /,
        *,
        reference_thickness: ArrayLike = 1.0,
    ) -> CoupledPlaneStressIncompressibleResponse:
        from ._mixed_hyperelastic import MixedHyperelasticLaw

        if not isinstance(law, MixedHyperelasticLaw):
            raise TypeError("law must be MixedHyperelasticLaw.")
        if law.volumetric_constraint is not self.volumetric_constraint:
            raise ValueError(
                "Plan and mixed law must share the identical volumetric constraint."
            )
        law_bulk = law.bulk_modulus
        if (law_bulk is None) != (self.bulk_modulus is None) or (
            law_bulk is not None and float(law_bulk) != self.bulk_modulus
        ):
            raise ValueError("Plan and mixed law bulk-modulus semantics differ.")
        deformation = jnp.asarray(deformation_gradient)
        if deformation.shape[-2:] != (2, 2):
            raise ValueError("Coupled plane-stress gradients must end in 2x2.")
        if not jnp.issubdtype(deformation.dtype, jnp.inexact):
            deformation = deformation.astype(jnp.result_type(deformation.dtype, float))
        batch_shape = deformation.shape[:-2]
        thickness = jnp.asarray(reference_thickness, dtype=deformation.dtype)
        try:
            thickness = jnp.broadcast_to(thickness, batch_shape)
        except ValueError as error:
            raise ValueError(
                "reference_thickness must broadcast to the deformation batch."
            ) from error
        outputs = jax.vmap(
            lambda value, h0: self._evaluate_coupled_point(value, law, h0)
        )(
            deformation.reshape((-1, 2, 2)),
            thickness.reshape((-1,)),
        )
        stretch, pressure, residual, stress, tangent, successful, failure = outputs
        return CoupledPlaneStressIncompressibleResponse(
            stretch.reshape(batch_shape),
            pressure.reshape(batch_shape),
            residual.reshape(batch_shape + (2,)),
            stress.reshape(batch_shape + (2, 2)),
            tangent.reshape(batch_shape + (2, 2, 2, 2)),
            successful.reshape(batch_shape),
            failure.reshape(batch_shape),
        )

    def _evaluate_coupled_point(self, deformation: Array, law: Any, h0: Array):
        lower = jnp.asarray(
            (self.log_stretch_bounds[0], self.pressure_bounds[0]),
            dtype=deformation.dtype,
        )
        upper = jnp.asarray(
            (self.log_stretch_bounds[1], self.pressure_bounds[1]),
            dtype=deformation.dtype,
        )

        def embedded(eta, in_plane=deformation):
            value = jnp.zeros((3, 3), dtype=in_plane.dtype)
            value = value.at[:2, :2].set(in_plane)
            return value.at[2, 2].set(jnp.exp(eta))

        def equations(state, in_plane=deformation):
            eta, pressure = state
            value = embedded(eta, in_plane)
            first_piola = law.first_piola(value, pressure)
            constraint = law.constraint(value, pressure)
            return jnp.stack((first_piola[2, 2], constraint))

        state, root_residual, linear_success = _bounded_two_root(
            lambda state: equations(state),
            lower,
            upper,
            self.root_policy,
        )
        eta, pressure = state
        value = embedded(eta)
        first_piola_3d = law.first_piola(value, pressure)
        law_response = law.evaluate(value, pressure)
        first_piola = h0 * first_piola_3d[:2, :2]

        blocks = law.block_tangent(value, pressure)
        stretch = jnp.exp(eta)
        root_jacobian = jnp.stack(
            (
                jnp.stack(
                    (
                        blocks.deformation_deformation[2, 2, 2, 2] * stretch,
                        blocks.deformation_pressure[2, 2],
                    )
                ),
                jnp.stack(
                    (
                        blocks.pressure_deformation[2, 2] * stretch,
                        blocks.pressure_pressure,
                    )
                ),
            )
        )
        in_plane_jacobian = jnp.stack(
            (
                blocks.deformation_deformation[2, 2, :2, :2].reshape((4,)),
                blocks.pressure_deformation[:2, :2].reshape((4,)),
            )
        )
        implicit = solve_small_linear(
            _PLANE_STRESS_LINEAR_SOLVE,
            root_jacobian,
            -in_plane_jacobian,
        )
        direct = h0 * blocks.deformation_deformation[:2, :2, :2, :2].reshape((4, 4))
        unknown_derivative = h0 * jnp.stack(
            (
                blocks.deformation_deformation[:2, :2, 2, 2] * stretch,
                blocks.deformation_pressure[:2, :2],
            ),
            axis=-1,
        ).reshape((4, 2))
        condensed = (direct + unknown_derivative @ implicit.value).reshape((2, 2, 2, 2))

        determinant = _determinant_2d(deformation)
        input_valid = (
            jnp.all(jnp.isfinite(deformation))
            & jnp.isfinite(h0)
            & (h0 > 0.0)
            & jnp.isfinite(determinant)
            & (determinant > 0.0)
        )
        base_valid = law_response.evidence.valid
        finite = (
            jnp.all(jnp.isfinite(state))
            & jnp.all(jnp.isfinite(root_residual))
            & jnp.all(jnp.isfinite(first_piola))
            & jnp.all(jnp.isfinite(condensed))
        )
        initial_residual = equations(0.5 * (lower + upper))
        threshold = (
            self.root_policy.absolute_residual
            + self.root_policy.relative_residual
            * jnp.sqrt(jnp.sum(initial_residual * initial_residual))
        )
        converged = jnp.sqrt(jnp.sum(root_residual * root_residual)) <= threshold
        failure = jnp.where(
            ~input_valid,
            int(PlaneStressFailure.INVALID_INPUT),
            jnp.where(
                ~base_valid,
                int(PlaneStressFailure.BASE_LAW_REJECTED),
                jnp.where(
                    ~finite,
                    int(PlaneStressFailure.NONFINITE),
                    jnp.where(
                        ~converged | ~linear_success,
                        int(PlaneStressFailure.MAX_STEPS),
                        jnp.where(
                            ~implicit.successful,
                            int(PlaneStressFailure.SINGULAR_TANGENT),
                            int(PlaneStressFailure.OK),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        successful = failure == int(PlaneStressFailure.OK)
        first_piola = jnp.where(successful, first_piola, jnp.nan)
        condensed = jnp.where(successful, condensed, jnp.nan)
        return (
            jnp.exp(eta),
            pressure,
            root_residual,
            first_piola,
            condensed,
            successful,
            failure,
        )


def _ordered_finite_bounds(bounds, /, *, name: str) -> tuple[float, float]:
    if len(bounds) != 2:
        raise ValueError(f"{name} must contain two endpoints.")
    lower, upper = (float(value) for value in bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"{name} must be finite and strictly ordered.")
    return lower, upper


def _bounded_two_root(function, lower, upper, policy):
    state = 0.5 * (lower + upper)
    residual = function(state)
    initial_norm = jnp.sqrt(jnp.sum(residual * residual))
    threshold = policy.absolute_residual + policy.relative_residual * initial_norm
    active = jnp.isfinite(initial_norm) & (initial_norm > threshold)
    linear_success = jnp.asarray(True)
    rates = jnp.asarray((1.0, 0.5, 0.25, 0.125, 0.0625), dtype=state.dtype)

    def body(_, carry):
        current, current_residual, current_active, solves_successful = carry
        jacobian = jax.jacfwd(function)(current)
        direction = solve_small_linear(
            _PLANE_STRESS_LINEAR_SOLVE,
            jacobian,
            -current_residual,
        )
        candidates = jnp.clip(
            current[None, :] + rates[:, None] * direction.value[None, :],
            lower,
            upper,
        )
        candidate_residuals = jax.vmap(function)(candidates)
        norms = jnp.sqrt(jnp.sum(candidate_residuals * candidate_residuals, axis=-1))
        norms = jnp.where(jnp.isfinite(norms), norms, jnp.inf)
        best = jnp.argmin(norms)
        candidate = candidates[best]
        candidate_residual = candidate_residuals[best]
        current_norm = jnp.sqrt(jnp.sum(current_residual * current_residual))
        accepted = direction.successful & (norms[best] < current_norm)
        update = current_active & accepted
        next_state = jnp.where(update, candidate, current)
        next_residual = jnp.where(update, candidate_residual, current_residual)
        next_norm = jnp.sqrt(jnp.sum(next_residual * next_residual))
        next_active = current_active & accepted & (next_norm > threshold)
        return (
            next_state,
            next_residual,
            next_active,
            solves_successful & (~current_active | direction.successful),
        )

    state, residual, active, linear_success = jax.lax.fori_loop(
        0,
        policy.maximum_steps,
        body,
        (state, residual, active, linear_success),
    )
    return state, residual, linear_success & ~active


__all__ = [
    "BlockDiagonalPlaneStressReductionPlan",
    "BlockDiagonalPlaneStressReductionResponse",
    "CoupledPlaneStressIncompressiblePlan",
    "CoupledPlaneStressIncompressibleResponse",
    "PlaneStressFailure",
    "PlaneStressKinematics",
]
