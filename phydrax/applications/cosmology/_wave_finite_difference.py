#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Periodic, cell-centered finite-difference actions for complex wave matter."""

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._tensor_support import PreparedTensorGrid
from ...linalg import (
    FFTLinearTransform,
    FunctionLinearOperator,
    GMRES,
    IdentityLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    solve,
    TolerancePolicy,
)


class WaveFiniteDifferenceState(StrictModule):
    """Native-complex cell wavefunction at one explicit coordinate time."""

    psi: Array
    coordinate_time: Array


class WaveFiniteDifferencePolicy(StrictModule, NonTrainableState):
    """Residual, conservation, phase, and nonlinear acceptance gates."""

    solve_relative_tolerance: float = eqx.field(static=True)
    solve_absolute_tolerance: float = eqx.field(static=True)
    maximum_solve_steps: int = eqx.field(static=True)
    norm_relative_tolerance: float = eqx.field(static=True)
    self_adjoint_tolerance: float = eqx.field(static=True)
    maximum_phase_radians: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        solve_relative_tolerance: float = 1.0e-10,
        solve_absolute_tolerance: float = 1.0e-12,
        maximum_solve_steps: int = 400,
        norm_relative_tolerance: float = 1.0e-9,
        self_adjoint_tolerance: float = 1.0e-11,
        maximum_phase_radians: float = 0.75,
    ):
        values = tuple(
            float(value)
            for value in (
                solve_relative_tolerance,
                solve_absolute_tolerance,
                norm_relative_tolerance,
                self_adjoint_tolerance,
                maximum_phase_radians,
            )
        )
        steps = int(maximum_solve_steps)
        if (
            not all(isfinite(value) and value >= 0.0 for value in values[:-1])
            or not isfinite(values[-1])
            or values[-1] <= 0.0
            or steps < 1
        ):
            raise ValueError("Finite-difference wave policy values are invalid.")
        self.solve_relative_tolerance = values[0]
        self.solve_absolute_tolerance = values[1]
        self.maximum_solve_steps = steps
        self.norm_relative_tolerance = values[2]
        self.self_adjoint_tolerance = values[3]
        self.maximum_phase_radians = values[4]
        self.policy_id = canonical_fingerprint(
            {
                "kind": "wave-finite-difference-policy",
                "solve_relative_tolerance": values[0],
                "solve_absolute_tolerance": values[1],
                "maximum_solve_steps": steps,
                "norm_relative_tolerance": values[2],
                "self_adjoint_tolerance": values[3],
                "maximum_phase_radians": values[4],
            }
        )


class WaveContactSelfInteractionPlan(StrictModule, NonTrainableState):
    """Separately identified Gross--Pitaevskii contact phase policy."""

    coupling: float = eqx.field(static=True)
    retained_mode_fraction: float = eqx.field(static=True)
    maximum_dealiasing_defect: float = eqx.field(static=True)
    energy_relative_tolerance: float = eqx.field(static=True)
    maximum_phase_radians: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coupling: float,
        /,
        *,
        retained_mode_fraction: float = 2.0 / 3.0,
        maximum_dealiasing_defect: float = 0.1,
        energy_relative_tolerance: float = 1.0e-9,
        maximum_phase_radians: float = 0.5,
    ):
        values = tuple(
            float(value)
            for value in (
                coupling,
                retained_mode_fraction,
                maximum_dealiasing_defect,
                energy_relative_tolerance,
                maximum_phase_radians,
            )
        )
        if (
            not all(isfinite(value) for value in values)
            or not 0.0 < values[1] <= 1.0
            or values[2] < 0.0
            or values[3] < 0.0
            or values[4] <= 0.0
        ):
            raise ValueError("Contact self-interaction policy values are invalid.")
        self.coupling = values[0]
        self.retained_mode_fraction = values[1]
        self.maximum_dealiasing_defect = values[2]
        self.energy_relative_tolerance = values[3]
        self.maximum_phase_radians = values[4]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "wave-contact-self-interaction",
                "coupling": values[0],
                "retained_mode_fraction": values[1],
                "maximum_dealiasing_defect": values[2],
                "energy_relative_tolerance": values[3],
                "maximum_phase_radians": values[4],
            }
        )


class WaveContactActionResult(StrictModule):
    state: WaveFiniteDifferenceState
    candidate_state: WaveFiniteDifferenceState
    maximum_phase: Array
    input_truncation_defect: Array
    aliasing_defect: Array
    dealiasing_defect: Array
    energy_relative_error: Array
    finite: Array
    successful: Array
    contact_plan_id: str = eqx.field(static=True)


class WaveFiniteDifferenceDiagnostics(StrictModule):
    initial_norm: Array
    final_norm: Array
    norm_relative_error: Array
    cayley_relative_residual: Array
    self_adjoint_residual: Array
    maximum_kinetic_phase: Array
    maximum_potential_phase: Array
    maximum_contact_phase: Array
    contact_input_truncation_defect: Array
    contact_aliasing_defect: Array
    contact_dealiasing_defect: Array
    contact_energy_relative_error: Array
    finite: Array
    phase_resolved: Array
    contact_accepted: Array
    accepted: Array
    status: Array
    status_meanings: tuple[str, ...] = eqx.field(
        static=True,
        default=(
            "accepted",
            "non_finite",
            "linear_solve_failed",
            "cayley_residual",
            "self_adjoint_residual",
            "norm_drift",
            "phase_unresolved",
            "contact_gate_failed",
        ),
    )


class WaveFiniteDifferenceResult(StrictModule):
    state: WaveFiniteDifferenceState
    candidate_state: WaveFiniteDifferenceState
    diagnostics: WaveFiniteDifferenceDiagnostics
    kinetic_solve: LinearSolveResult
    successful: Array
    prepared_id: str = eqx.field(static=True)


class PeriodicWaveFiniteDifferencePlan(StrictModule, NonTrainableState):
    """Physics and qualification policy for a periodic complex FD wave grid."""

    boson_mass: float = eqx.field(static=True)
    reduced_planck_constant: float = eqx.field(static=True)
    policy: WaveFiniteDifferencePolicy
    contact: WaveContactSelfInteractionPlan | None
    dtype: np.dtype = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        boson_mass: float,
        /,
        *,
        reduced_planck_constant: float = 1.0,
        policy: WaveFiniteDifferencePolicy | None = None,
        contact: WaveContactSelfInteractionPlan | None = None,
        dtype: Any = np.complex128,
    ):
        mass = float(boson_mass)
        hbar = float(reduced_planck_constant)
        policy_ = WaveFiniteDifferencePolicy() if policy is None else policy
        dtype_ = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))
        if not isfinite(mass) or mass <= 0.0 or not isfinite(hbar) or hbar <= 0.0:
            raise ValueError(
                "Wave finite-difference physical coefficients must be positive."
            )
        if not isinstance(policy_, WaveFiniteDifferencePolicy):
            raise TypeError("policy must be WaveFiniteDifferencePolicy or None.")
        if contact is not None and not isinstance(
            contact, WaveContactSelfInteractionPlan
        ):
            raise TypeError("contact must be WaveContactSelfInteractionPlan or None.")
        if not np.issubdtype(dtype_, np.complexfloating):
            raise TypeError("Wave finite-difference storage must use a complex dtype.")
        self.boson_mass = mass
        self.reduced_planck_constant = hbar
        self.policy = policy_
        self.contact = contact
        self.dtype = dtype_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-wave-finite-difference-plan",
                "boson_mass": mass,
                "reduced_planck_constant": hbar,
                "policy": policy_.policy_id,
                "contact": None if contact is None else contact.plan_id,
                "dtype": dtype_.str,
                "stencil": "cell-centered-second-order-periodic",
            }
        )

    def prepare(
        self, grid: PreparedTensorGrid, /
    ) -> "PreparedPeriodicWaveFiniteDifference":
        return PreparedPeriodicWaveFiniteDifference(self, grid)


class PreparedPeriodicWaveFiniteDifference(StrictModule, NonTrainableState):
    """Weighted self-adjoint FD kinetic operator and global Cayley action."""

    plan: PeriodicWaveFiniteDifferencePlan
    grid: PreparedTensorGrid
    spacings: tuple[float, ...] = eqx.field(static=True)
    kinetic_operator: FunctionLinearOperator
    fft_transforms: tuple[FFTLinearTransform, ...]
    retained_masks: tuple[Array, ...]
    kinetic_eigenvalues: Array
    solve_policy: LinearSolvePolicy
    coordinate_convention: str = eqx.field(static=True)
    time_level_convention: str = eqx.field(static=True)
    action_convention: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: PeriodicWaveFiniteDifferencePlan, grid: PreparedTensorGrid, /
    ):
        if not isinstance(plan, PeriodicWaveFiniteDifferencePlan):
            raise TypeError("plan must be PeriodicWaveFiniteDifferencePlan.")
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("grid must be PreparedTensorGrid.")
        if not 1 <= len(grid.shape) <= 3:
            raise ValueError("Periodic FD wave grids support one to three dimensions.")
        if any(axis.primary_entity != "interval" for axis in grid.structured_axes):
            raise ValueError("Periodic FD waves require cell-centered interval axes.")
        if not all(axis.periodic for axis in grid.axes):
            raise ValueError("Periodic FD waves reject physical or isolated boundaries.")
        widths = tuple(
            np.asarray(axis.interval_widths, dtype=np.float64)
            for axis in grid.structured_axes
        )
        if any(
            values.size == 0
            or np.any(~np.isfinite(values))
            or np.any(values <= 0.0)
            or not np.all(values == values[0])
            for values in widths
        ):
            raise ValueError("Periodic FD waves require finite uniform axis spacing.")
        spacings = tuple(float(values[0]) for values in widths)
        field = grid.field_space(
            "periodic-complex-wavefunction",
            dtype=plan.dtype,
            representation="cell_average",
            conformity="discontinuous",
        )
        space = field.vector_space

        def kinetic_action(psi):
            output = jnp.zeros_like(psi)
            for axis, spacing in enumerate(spacings):
                output = (
                    output
                    + (
                        2.0 * psi
                        - jnp.roll(psi, 1, axis=axis)
                        - jnp.roll(psi, -1, axis=axis)
                    )
                    / spacing**2
                )
            return output

        kinetic = FunctionLinearOperator(
            kinetic_action,
            source=space,
            target=space,
            transpose_action=kinetic_action,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_semidefinite=True,
                rank=space.size - 1,
                evidence={
                    "self_adjoint": "construction",
                    "positive_semidefinite": "construction",
                    "rank": "construction",
                },
            ),
            operator_id=canonical_fingerprint(
                {
                    "kind": "periodic-fd-negative-laplacian",
                    "grid": grid.prepared_id,
                    "spacings": spacings,
                    "dtype": plan.dtype.str,
                }
            ),
        )
        transforms = tuple(
            FFTLinearTransform(count, dtype=plan.dtype) for count in grid.shape
        )
        fraction = 1.0 if plan.contact is None else plan.contact.retained_mode_fraction
        masks = tuple(
            jnp.asarray(
                np.abs(np.fft.fftfreq(count) * count) <= fraction * max(1, count // 2)
            )
            for count in grid.shape
        )
        real_dtype = np.empty((), dtype=plan.dtype).real.dtype
        eigenvalues = jnp.zeros(grid.shape, dtype=real_dtype)
        for axis, (count, spacing) in enumerate(zip(grid.shape, spacings, strict=True)):
            modes = np.fft.fftfreq(count) * count
            values = 4.0 * np.sin(np.pi * modes / count) ** 2 / spacing**2
            shape = [1] * len(grid.shape)
            shape[axis] = count
            eigenvalues = eigenvalues + jnp.asarray(
                values.reshape(tuple(shape)),
                dtype=real_dtype,
            )
        self.plan = plan
        self.grid = grid
        self.spacings = spacings
        self.kinetic_operator = kinetic
        self.fft_transforms = transforms
        self.retained_masks = masks
        self.kinetic_eigenvalues = eigenvalues
        self.solve_policy = LinearSolvePolicy(
            GMRES(restart=min(40, max(4, space.size))),
            tolerance=TolerancePolicy(
                relative=plan.policy.solve_relative_tolerance,
                absolute=plan.policy.solve_absolute_tolerance,
                max_steps=plan.policy.maximum_solve_steps,
            ),
        )
        self.coordinate_convention = "periodic-cell-centered-cartesian-code-length"
        self.time_level_convention = "explicit-caller-coordinate-time"
        self.action_convention = (
            "drift_factor and kick_factor are caller-integrated code-time actions"
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-wave-finite-difference",
                "plan": plan.plan_id,
                "grid": grid.prepared_id,
                "kinetic": kinetic.operator_id,
                "transforms": [value.transform_id for value in transforms],
                "kinetic_eigenvalues": canonical_fingerprint(
                    np.asarray(eigenvalues).tolist()
                ),
                "coordinate_convention": self.coordinate_convention,
                "time_level_convention": self.time_level_convention,
                "action_convention": self.action_convention,
            }
        )

    @property
    def dtype(self) -> np.dtype:
        return self.plan.dtype

    @property
    def real_dtype(self) -> np.dtype:
        return np.empty((), dtype=self.dtype).real.dtype

    @property
    def boson_mass(self) -> float:
        return self.plan.boson_mass

    @property
    def reduced_planck_constant(self) -> float:
        return self.plan.reduced_planck_constant

    def _validate_state(
        self, state: WaveFiniteDifferenceState, /
    ) -> WaveFiniteDifferenceState:
        if not isinstance(state, WaveFiniteDifferenceState):
            raise TypeError("state must be WaveFiniteDifferenceState.")
        psi = jnp.asarray(state.psi)
        time = jnp.asarray(state.coordinate_time, dtype=self.real_dtype)
        if psi.shape != self.grid.shape or np.dtype(psi.dtype) != self.dtype:
            raise ValueError(
                f"Wavefunction must have shape {self.grid.shape} and dtype {self.dtype}."
            )
        if time.shape != ():
            raise ValueError("Wave coordinate time must be scalar.")
        psi = eqx.error_if(
            psi,
            ~jnp.all(jnp.isfinite(psi)) | ~jnp.isfinite(time) | (self.norm(psi) <= 0.0),
            "Wave FD state must be finite and have positive norm.",
        )
        return WaveFiniteDifferenceState(psi, time)

    def initialize(
        self,
        psi: ArrayLike,
        coordinate_time: ArrayLike = 0.0,
        /,
    ) -> WaveFiniteDifferenceState:
        return self._validate_state(
            WaveFiniteDifferenceState(
                jnp.asarray(psi, dtype=self.dtype),
                jnp.asarray(coordinate_time, dtype=self.real_dtype),
            )
        )

    def norm(self, psi: ArrayLike, /) -> Array:
        values = jnp.asarray(psi, dtype=self.dtype)
        return jnp.real(self.kinetic_operator.source.inner(values, values))

    def density(self, state: WaveFiniteDifferenceState, /) -> Array:
        checked = self._validate_state(state)
        return self.boson_mass * jnp.abs(checked.psi) ** 2

    def _axis_transform(self, values: Array, axis: int, *, inverse: bool) -> Array:
        moved = jnp.moveaxis(values, axis, -1)
        leading = moved.shape[:-1]
        flat = moved.reshape((-1, moved.shape[-1]))
        transform = self.fft_transforms[axis]
        action = transform.synthesize if inverse else transform.analyze
        transformed = jax.vmap(action)(flat)
        return jnp.moveaxis(
            transformed.reshape(leading + (transformed.shape[-1],)), -1, axis
        )

    def _modal(self, values: Array, /) -> Array:
        coefficients = values.astype(self.dtype)
        for axis in range(len(self.grid.shape)):
            coefficients = self._axis_transform(coefficients, axis, inverse=False)
        return coefficients

    def _physical(self, coefficients: Array, /) -> Array:
        values = coefficients
        for axis in reversed(range(len(self.grid.shape))):
            values = self._axis_transform(values, axis, inverse=True)
        return values

    def _truncate(self, coefficients: Array, /) -> Array:
        retained = coefficients
        for axis, mask in enumerate(self.retained_masks):
            shape = [1] * retained.ndim
            shape[axis] = mask.size
            retained = jnp.where(mask.reshape(tuple(shape)), retained, 0.0)
        return retained

    def _qualified_contact_density(
        self,
        psi: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        coefficients = self._modal(psi)
        retained_psi = self._truncate(coefficients)
        coefficient_norm = jnp.sqrt(jnp.sum(jnp.abs(coefficients) ** 2))
        input_defect = jnp.sqrt(
            jnp.sum(jnp.abs(coefficients - retained_psi) ** 2)
        ) / jnp.where(coefficient_norm > 0.0, coefficient_norm, 1.0)
        qualified_psi = self._physical(retained_psi)
        density = jnp.abs(qualified_psi) ** 2
        density_coefficients = self._modal(density.astype(self.dtype))
        retained_density = self._truncate(density_coefficients)
        density_norm = jnp.sqrt(jnp.sum(jnp.abs(density_coefficients) ** 2))
        aliasing_defect = jnp.sqrt(
            jnp.sum(jnp.abs(density_coefficients - retained_density) ** 2)
        ) / jnp.where(density_norm > 0.0, density_norm, 1.0)
        filtered_density = jnp.real(self._physical(retained_density)).astype(
            self.real_dtype
        )
        return filtered_density, input_defect, aliasing_defect

    def _maximum_kinetic_phase(self, psi: Array, drift_factor: ArrayLike, /) -> Array:
        drift = jnp.asarray(drift_factor, dtype=self.real_dtype)
        coefficients = self._modal(psi)
        power = jnp.abs(coefficients) ** 2
        occupied = power >= 1.0e-12 * jnp.max(power)
        action = self.reduced_planck_constant * drift / (2.0 * self.boson_mass)
        cayley_phase = 2.0 * jnp.arctan(0.5 * jnp.abs(action) * self.kinetic_eigenvalues)
        return jnp.max(jnp.where(occupied, cayley_phase, 0.0))

    def _self_adjoint_residual(self, psi: Array, /) -> Array:
        probe = jnp.roll(psi, 1, axis=0)
        kinetic_psi = self.kinetic_operator.mv(psi)
        kinetic_probe = self.kinetic_operator.mv(probe)
        left = self.kinetic_operator.source.inner(psi, kinetic_probe)
        right = self.kinetic_operator.source.inner(kinetic_psi, probe)
        scale = jnp.maximum(jnp.maximum(jnp.abs(left), jnp.abs(right)), 1.0)
        return jnp.abs(left - right) / scale

    def kinetic_drift(
        self,
        state: WaveFiniteDifferenceState,
        drift_factor: ArrayLike,
        /,
        *,
        end_coordinate_time: ArrayLike | None = None,
    ) -> tuple[WaveFiniteDifferenceState, LinearSolveResult, Array, Array]:
        checked = self._validate_state(state)
        drift = jnp.asarray(drift_factor, dtype=self.real_dtype)
        if drift.shape != ():
            raise ValueError("Kinetic drift factor must be scalar.")
        drift = eqx.error_if(drift, ~jnp.isfinite(drift), "Kinetic drift must be finite.")
        action = self.reduced_planck_constant * drift / (2.0 * self.boson_mass)
        alpha = 0.5 * action
        identity = IdentityLinearOperator(self.kinetic_operator.source)
        left = identity + (1j * alpha) * self.kinetic_operator
        right = identity + (-1j * alpha) * self.kinetic_operator
        right_hand_side = right.mv(checked.psi)
        solved = solve(LinearSystem(left), right_hand_side, policy=self.solve_policy)
        candidate = jnp.asarray(solved.value, dtype=self.dtype)
        residual = left.mv(candidate) - right_hand_side
        residual_norm = jnp.sqrt(jnp.real(left.target.inner(residual, residual)))
        rhs_norm = jnp.sqrt(jnp.real(left.target.inner(right_hand_side, right_hand_side)))
        relative_residual = residual_norm / jnp.where(rhs_norm > 0.0, rhs_norm, 1.0)
        if end_coordinate_time is None:
            time = eqx.error_if(
                checked.coordinate_time,
                drift != 0.0,
                "Nonzero kinetic action requires explicit end_coordinate_time.",
            )
        else:
            time = jnp.asarray(end_coordinate_time, dtype=self.real_dtype)
            if time.shape != ():
                raise ValueError("End coordinate time must be scalar.")
            time = eqx.error_if(
                time,
                ~jnp.isfinite(time)
                | ((drift != 0.0) & (time <= checked.coordinate_time)),
                "Nonzero kinetic action requires a finite advancing end time.",
            )
        return (
            WaveFiniteDifferenceState(candidate, time),
            solved,
            relative_residual,
            self._self_adjoint_residual(checked.psi),
        )

    def potential_kick(
        self,
        state: WaveFiniteDifferenceState,
        potential: ArrayLike,
        kick_factor: ArrayLike,
        /,
        *,
        fraction: ArrayLike = 1.0,
    ) -> tuple[WaveFiniteDifferenceState, Array]:
        """Apply a stop-gradient external potential phase on the fixed grid."""
        checked = self._validate_state(state)
        values = jnp.asarray(potential)
        if values.shape != self.grid.shape:
            raise ValueError("Potential must have the exact FD grid shape.")
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise TypeError("Potential must have a real floating dtype.")
        values = values.astype(self.real_dtype)
        kick = jnp.asarray(kick_factor, dtype=self.real_dtype)
        amount = jnp.asarray(fraction, dtype=self.real_dtype)
        if kick.shape != () or amount.shape != ():
            raise ValueError("Potential kick factor and fraction must be scalar.")
        values = eqx.error_if(
            values,
            ~jnp.all(jnp.isfinite(values))
            | ~jnp.isfinite(kick)
            | ~jnp.isfinite(amount)
            | (amount < 0.0)
            | (amount > 1.0),
            "Potential inputs must be finite and fraction must lie in [0, 1].",
        )
        values = jax.lax.stop_gradient(values)
        phase = amount * self.boson_mass * kick * values / self.reduced_planck_constant
        return (
            WaveFiniteDifferenceState(
                checked.psi * jnp.exp(-1j * phase), checked.coordinate_time
            ),
            jnp.max(jnp.abs(phase)),
        )

    def contact_kick(
        self,
        state: WaveFiniteDifferenceState,
        action_factor: ArrayLike,
        /,
        *,
        fraction: ArrayLike = 1.0,
    ) -> WaveContactActionResult:
        contact = self.plan.contact
        if contact is None:
            raise ValueError("This wave FD plan has no contact self-interaction.")
        checked = self._validate_state(state)
        action = jnp.asarray(action_factor, dtype=self.real_dtype)
        amount = jnp.asarray(fraction, dtype=self.real_dtype)
        if action.shape != () or amount.shape != ():
            raise ValueError("Contact action factor and fraction must be scalar.")
        action = eqx.error_if(
            action,
            ~jnp.isfinite(action)
            | ~jnp.isfinite(amount)
            | (amount < 0.0)
            | (amount > 1.0),
            "Contact action must be finite and its fraction must lie in [0, 1].",
        )
        filtered_density, input_defect, aliasing_defect = self._qualified_contact_density(
            checked.psi
        )
        phase = (
            amount
            * contact.coupling
            * action
            * filtered_density
            / self.reduced_planck_constant
        )
        candidate = checked.psi * jnp.exp(-1j * phase)
        (
            final_filtered_density,
            final_input_defect,
            final_aliasing_defect,
        ) = self._qualified_contact_density(candidate)
        input_defect = jnp.maximum(input_defect, final_input_defect)
        aliasing_defect = jnp.maximum(aliasing_defect, final_aliasing_defect)
        dealiasing_defect = jnp.maximum(input_defect, aliasing_defect)
        weights = self.grid.quadrature_weights.astype(self.real_dtype)
        initial_energy = 0.5 * contact.coupling * jnp.sum(weights * filtered_density**2)
        final_energy = (
            0.5 * contact.coupling * jnp.sum(weights * final_filtered_density**2)
        )
        energy_scale = jnp.maximum(jnp.abs(initial_energy), 1.0)
        energy_error = jnp.abs(final_energy - initial_energy) / energy_scale
        maximum_phase = jnp.max(jnp.abs(phase))
        finite = (
            jnp.all(jnp.isfinite(candidate))
            & jnp.isfinite(input_defect)
            & jnp.isfinite(aliasing_defect)
            & jnp.isfinite(dealiasing_defect)
            & jnp.isfinite(energy_error)
            & jnp.isfinite(maximum_phase)
        )
        accepted = (
            finite
            & (input_defect <= contact.maximum_dealiasing_defect)
            & (aliasing_defect <= contact.maximum_dealiasing_defect)
            & (energy_error <= contact.energy_relative_tolerance)
            & (maximum_phase <= contact.maximum_phase_radians)
        )
        committed = jnp.where(accepted, candidate, checked.psi)
        return WaveContactActionResult(
            WaveFiniteDifferenceState(committed, checked.coordinate_time),
            WaveFiniteDifferenceState(candidate, checked.coordinate_time),
            maximum_phase,
            input_defect,
            aliasing_defect,
            dealiasing_defect,
            energy_error,
            finite,
            accepted,
            contact.plan_id,
        )

    def step(
        self,
        state: WaveFiniteDifferenceState,
        potential: ArrayLike,
        drift_factor: ArrayLike,
        kick_factor: ArrayLike,
        /,
        *,
        end_coordinate_time: ArrayLike | None = None,
        contact_action_factor: ArrayLike | None = None,
    ) -> WaveFiniteDifferenceResult:
        """Apply symmetric potential/contact/Cayley/contact/potential actions."""
        initial = self._validate_state(state)
        initial_norm = self.norm(initial.psi)
        first, first_phase = self.potential_kick(
            initial, potential, kick_factor, fraction=0.5
        )
        contact_phase = jnp.asarray(0.0, dtype=self.real_dtype)
        input_defect = jnp.asarray(0.0, dtype=self.real_dtype)
        aliasing_defect = jnp.asarray(0.0, dtype=self.real_dtype)
        dealiasing_defect = jnp.asarray(0.0, dtype=self.real_dtype)
        contact_energy_error = jnp.asarray(0.0, dtype=self.real_dtype)
        contact_accepted = jnp.asarray(True)
        if contact_action_factor is not None:
            if self.plan.contact is None:
                raise ValueError(
                    "contact_action_factor requires a WaveContactSelfInteractionPlan."
                )
            first_contact = self.contact_kick(
                first,
                contact_action_factor,
                fraction=0.5,
            )
            first = first_contact.candidate_state
            contact_phase = first_contact.maximum_phase
            input_defect = first_contact.input_truncation_defect
            aliasing_defect = first_contact.aliasing_defect
            dealiasing_defect = first_contact.dealiasing_defect
            contact_energy_error = first_contact.energy_relative_error
            contact_accepted = first_contact.successful
        kinetic_phase = self._maximum_kinetic_phase(first.psi, drift_factor)
        kinetic, solved, cayley_residual, self_adjoint_residual = self.kinetic_drift(
            first,
            drift_factor,
            end_coordinate_time=end_coordinate_time,
        )
        if contact_action_factor is not None:
            second_contact = self.contact_kick(
                kinetic,
                contact_action_factor,
                fraction=0.5,
            )
            kinetic = second_contact.candidate_state
            contact_phase = jnp.maximum(
                contact_phase,
                second_contact.maximum_phase,
            )
            input_defect = jnp.maximum(
                input_defect,
                second_contact.input_truncation_defect,
            )
            aliasing_defect = jnp.maximum(
                aliasing_defect,
                second_contact.aliasing_defect,
            )
            dealiasing_defect = jnp.maximum(
                dealiasing_defect,
                second_contact.dealiasing_defect,
            )
            contact_energy_error = jnp.maximum(
                contact_energy_error,
                second_contact.energy_relative_error,
            )
            contact_accepted = contact_accepted & second_contact.successful
        candidate, second_phase = self.potential_kick(
            kinetic, potential, kick_factor, fraction=0.5
        )
        final_norm = self.norm(candidate.psi)
        norm_error = jnp.abs(final_norm - initial_norm) / initial_norm
        potential_phase = jnp.maximum(first_phase, second_phase)
        phase_resolved = (
            jnp.maximum(kinetic_phase, potential_phase)
            <= self.plan.policy.maximum_phase_radians
        )
        finite = (
            jnp.all(jnp.isfinite(candidate.psi))
            & jnp.isfinite(final_norm)
            & jnp.isfinite(cayley_residual)
            & jnp.isfinite(self_adjoint_residual)
            & jnp.isfinite(kinetic_phase)
        )
        linear_closed = solved.successful & jnp.all(solved.diagnostics.converged)
        cayley_closed = cayley_residual <= max(
            self.plan.policy.solve_relative_tolerance,
            self.plan.policy.solve_absolute_tolerance,
        )
        self_adjoint = self_adjoint_residual <= self.plan.policy.self_adjoint_tolerance
        norm_conserved = norm_error <= self.plan.policy.norm_relative_tolerance
        accepted = (
            finite
            & linear_closed
            & cayley_closed
            & self_adjoint
            & norm_conserved
            & phase_resolved
            & contact_accepted
        )
        status = jnp.where(
            ~finite,
            1,
            jnp.where(
                ~linear_closed,
                2,
                jnp.where(
                    ~cayley_closed,
                    3,
                    jnp.where(
                        ~self_adjoint,
                        4,
                        jnp.where(
                            ~norm_conserved,
                            5,
                            jnp.where(
                                ~phase_resolved,
                                6,
                                jnp.where(~contact_accepted, 7, 0),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        committed = WaveFiniteDifferenceState(
            jnp.where(accepted, candidate.psi, initial.psi),
            jnp.where(accepted, candidate.coordinate_time, initial.coordinate_time),
        )
        diagnostics = WaveFiniteDifferenceDiagnostics(
            initial_norm=initial_norm,
            final_norm=final_norm,
            norm_relative_error=norm_error,
            cayley_relative_residual=cayley_residual,
            self_adjoint_residual=self_adjoint_residual,
            maximum_kinetic_phase=kinetic_phase,
            maximum_potential_phase=potential_phase,
            maximum_contact_phase=contact_phase,
            contact_input_truncation_defect=input_defect,
            contact_aliasing_defect=aliasing_defect,
            contact_dealiasing_defect=dealiasing_defect,
            contact_energy_relative_error=contact_energy_error,
            finite=finite,
            phase_resolved=phase_resolved,
            contact_accepted=contact_accepted,
            accepted=accepted,
            status=status,
        )
        return WaveFiniteDifferenceResult(
            state=committed,
            candidate_state=candidate,
            diagnostics=diagnostics,
            kinetic_solve=solved,
            successful=accepted,
            prepared_id=self.prepared_id,
        )


__all__ = [
    "PeriodicWaveFiniteDifferencePlan",
    "PreparedPeriodicWaveFiniteDifference",
    "WaveContactActionResult",
    "WaveContactSelfInteractionPlan",
    "WaveFiniteDifferenceDiagnostics",
    "WaveFiniteDifferencePolicy",
    "WaveFiniteDifferenceResult",
    "WaveFiniteDifferenceState",
]
