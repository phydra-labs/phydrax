#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class KerrNewmanModeParameters(StrictModule):
    frequency: Array
    horizon_radii: Array
    horizon_angular_velocity: Array
    horizon_electric_potential: Array
    surface_gravity: Array
    separation_constant: Array
    radial_decay_rate: Array
    infinity_power: Array
    horizon_exponent: Array
    superradiant_detuning: Array
    finite: Array
    converged: Array
    physically_valid: Array
    bound_state: Array
    superradiant: Array
    qualified: Array
    derivative_valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class MassiveFieldQuasiBoundResult(StrictModule):
    frequency: Array
    hydrogenic_seed: Array
    hydrogenic_seed_residual: Array
    radial_log_derivative: Array
    shooting_residual: Array
    residual_norm: Array
    iterations: Array
    parameters: KerrNewmanModeParameters
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class MassiveFieldQuasiBoundPlan(StrictModule, NonTrainableState):
    """Fixed-grid charged massive-scalar quasi-bound-state shooting plan.

    The solve uses the separated charged Klein--Gordon radial equation on a
    Kerr--Newman background.  Its angular eigenvalue retains the first
    small-spheroidicity correction to the spherical eigenvalue; qualification
    is restricted to that regime rather than claiming an exact angular solve.
    """

    black_hole_mass: Array
    black_hole_spin: Array
    black_hole_charge: Array
    field_mass: Array
    field_charge: Array
    radial_nodes: Array
    ell: int = eqx.field(static=True)
    azimuthal: int = eqx.field(static=True)
    overtone: int = eqx.field(static=True)
    newton_steps: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    finite_difference_step: float = eqx.field(static=True)
    spheroidicity_limit: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        black_hole_mass,
        black_hole_spin,
        black_hole_charge,
        field_mass,
        radial_nodes,
        /,
        *,
        field_charge=0.0,
        ell=0,
        azimuthal=0,
        overtone=0,
        newton_steps=8,
        residual_tolerance=1.0e-7,
        finite_difference_step=2.0e-5,
        spheroidicity_limit=0.25,
    ):
        mass = float(np.asarray(black_hole_mass))
        spin = float(np.asarray(black_hole_spin))
        charge = float(np.asarray(black_hole_charge))
        particle_mass = float(np.asarray(field_mass))
        particle_charge = float(np.asarray(field_charge))
        nodes = np.asarray(radial_nodes, dtype=np.float64)
        ell_value = int(ell)
        azimuthal_value = int(azimuthal)
        overtone_value = int(overtone)
        discriminant = mass * mass - spin * spin - charge * charge
        if (
            not np.all(np.isfinite((mass, spin, charge, particle_mass, particle_charge)))
            or mass <= 0.0
            or particle_mass <= 0.0
            or discriminant <= 0.0
        ):
            raise ValueError(
                "A quasi-bound plan requires a finite subextremal Kerr--Newman black hole and a positive field mass."
            )
        outer_horizon = mass + np.sqrt(discriminant)
        if (
            nodes.ndim != 1
            or nodes.size < 8
            or np.any(~np.isfinite(nodes))
            or np.any(np.diff(nodes) <= 0.0)
            or nodes[0] <= outer_horizon
        ):
            raise ValueError(
                "radial_nodes must be a finite increasing exterior grid with at least eight nodes."
            )
        if ell_value < 0 or abs(azimuthal_value) > ell_value or overtone_value < 0:
            raise ValueError("Mode indices require ell >= |m| and overtone >= 0.")
        if int(newton_steps) <= 0:
            raise ValueError("newton_steps must be positive.")
        if (
            not np.isfinite(residual_tolerance)
            or residual_tolerance <= 0.0
            or not np.isfinite(finite_difference_step)
            or finite_difference_step <= 0.0
            or not np.isfinite(spheroidicity_limit)
            or spheroidicity_limit <= 0.0
        ):
            raise ValueError(
                "Solver tolerances and spheroidicity_limit must be positive."
            )
        self.black_hole_mass = jnp.asarray(mass)
        self.black_hole_spin = jnp.asarray(spin)
        self.black_hole_charge = jnp.asarray(charge)
        self.field_mass = jnp.asarray(particle_mass)
        self.field_charge = jnp.asarray(particle_charge)
        self.radial_nodes = jnp.asarray(nodes)
        self.ell = ell_value
        self.azimuthal = azimuthal_value
        self.overtone = overtone_value
        self.newton_steps = int(newton_steps)
        self.residual_tolerance = float(residual_tolerance)
        self.finite_difference_step = float(finite_difference_step)
        self.spheroidicity_limit = float(spheroidicity_limit)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "massive-field-kerr-newman-quasi-bound-shooting",
                "black_hole_mass": mass,
                "black_hole_spin": spin,
                "black_hole_charge": charge,
                "field_mass": particle_mass,
                "field_charge": particle_charge,
                "mode": (ell_value, azimuthal_value, overtone_value),
                "radial_nodes": array_tree_fingerprint(nodes),
                "newton_steps": int(newton_steps),
                "residual_tolerance": float(residual_tolerance),
                "finite_difference_step": float(finite_difference_step),
                "spheroidicity_limit": float(spheroidicity_limit),
                "angular_model": "first-order-small-massive-spheroidicity",
            }
        )

    def hydrogenic_seed(self) -> Array:
        """Return the charged Coulombic small-coupling frequency seed."""
        principal = self.ell + self.overtone + 1
        effective_coupling = (
            self.black_hole_mass * self.field_mass
            - self.field_charge * self.black_hole_charge
        )
        binding_argument = 1.0 - (effective_coupling / principal) ** 2
        real_frequency = self.field_mass * jnp.sqrt(binding_argument)
        horizon = self._horizon_quantities()
        threshold = self.azimuthal * horizon[2] + self.field_charge * horizon[3]
        growth_scale = (
            jnp.abs(effective_coupling) ** (4 * self.ell + 5) / self.black_hole_mass
        )
        imaginary_frequency = (threshold - real_frequency) * growth_scale
        return real_frequency + 1.0j * imaginary_frequency

    def mode_parameters(self, frequency: ArrayLike, /) -> KerrNewmanModeParameters:
        omega = jnp.asarray(frequency) + 0.0j
        inner, outer, angular_velocity, electric_potential, surface_gravity = (
            self._horizon_quantities()
        )
        separation = self._separation_constant(omega)
        decay = _positive_real_sqrt(self.field_mass**2 - omega**2 + 0.0j)
        infinity_power = (
            self.black_hole_mass * (2.0 * omega**2 - self.field_mass**2)
            - self.field_charge * self.black_hole_charge * omega
        ) / decay - 1.0
        exponent = (
            -1.0j
            * (
                (outer**2 + self.black_hole_spin**2) * omega
                - self.azimuthal * self.black_hole_spin
                - self.field_charge * self.black_hole_charge * outer
            )
            / (outer - inner)
        )
        detuning = (
            self.azimuthal * angular_velocity
            + self.field_charge * electric_potential
            - jnp.real(omega)
        )
        effective_coupling = (
            self.black_hole_mass * self.field_mass
            - self.field_charge * self.black_hole_charge
        )
        finite = jnp.all(
            jnp.isfinite(
                jnp.asarray(
                    (
                        jnp.real(omega),
                        jnp.imag(omega),
                        jnp.real(decay),
                        jnp.imag(decay),
                        jnp.real(separation),
                        jnp.imag(separation),
                    )
                )
            )
        )
        bound_state = (
            (effective_coupling > 0.0)
            & (jnp.real(omega) > 0.0)
            & (jnp.real(omega) < self.field_mass)
            & (jnp.real(decay) > 0.0)
        )
        spheroidicity = jnp.abs(self.black_hole_spin**2 * (self.field_mass**2 - omega**2))
        physically_valid = bound_state & (spheroidicity <= self.spheroidicity_limit)
        converged = finite
        qualified = finite & physically_valid
        derivative_valid = qualified & (jnp.abs(decay) > 1.0e-12)
        superradiant = bound_state & (detuning > 0.0)
        status = jnp.where(
            ~finite,
            3,
            jnp.where(~bound_state, 2, jnp.where(~physically_valid, 1, 0)),
        ).astype(jnp.int32)
        return KerrNewmanModeParameters(
            omega,
            jnp.asarray((inner, outer)),
            angular_velocity,
            electric_potential,
            surface_gravity,
            separation,
            decay,
            infinity_power,
            exponent,
            detuning,
            finite,
            converged,
            physically_valid,
            bound_state,
            superradiant,
            qualified,
            derivative_valid,
            status,
            self.plan_id,
        )

    def solve(self) -> MassiveFieldQuasiBoundResult:
        """Solve the complex outgoing-decay residual with fixed Newton capacity."""
        seed = self.hydrogenic_seed()
        initial_residual = self._shoot(seed)[1]
        initial = (
            seed,
            initial_residual,
            jnp.asarray(False),
            jnp.asarray(0, dtype=jnp.int32),
        )

        def newton_step(_, carry):
            omega, residual, done, iterations = carry
            jacobian, determinant = self._residual_jacobian(omega)
            right_hand_side = -jnp.asarray((jnp.real(residual), jnp.imag(residual)))
            safe_determinant = jnp.where(jnp.abs(determinant) > 1.0e-20, determinant, 1.0)
            step_real = (
                right_hand_side[0] * jacobian[1, 1] - jacobian[0, 1] * right_hand_side[1]
            ) / safe_determinant
            step_imag = (
                jacobian[0, 0] * right_hand_side[1] - right_hand_side[0] * jacobian[1, 0]
            ) / safe_determinant
            raw_step = step_real + 1.0j * step_imag
            trust_radius = 0.2 * self.field_mass
            scale = jnp.minimum(
                1.0, trust_radius / jnp.maximum(jnp.abs(raw_step), 1.0e-30)
            )
            usable = (
                ~done
                & jnp.isfinite(step_real)
                & jnp.isfinite(step_imag)
                & (jnp.abs(determinant) > 1.0e-20)
            )
            candidate = omega + scale * raw_step
            next_omega = jnp.where(usable, candidate, omega)
            next_residual = self._shoot(next_omega)[1]
            next_done = done | (jnp.abs(next_residual) <= self.residual_tolerance)
            next_iterations = iterations + usable.astype(jnp.int32)
            return next_omega, next_residual, next_done, next_iterations

        omega, residual, converged, iterations = jax.lax.fori_loop(
            0, self.newton_steps, newton_step, initial
        )
        radial_log_derivative, residual = self._shoot(omega)
        parameters = self.mode_parameters(omega)
        jacobian, determinant = self._residual_jacobian(omega)
        residual_norm = jnp.abs(residual)
        finite = parameters.finite & jnp.all(
            jnp.isfinite(
                jnp.concatenate(
                    (
                        jnp.real(radial_log_derivative),
                        jnp.imag(radial_log_derivative),
                        jnp.ravel(jacobian),
                    )
                )
            )
        )
        converged = converged | (residual_norm <= self.residual_tolerance)
        derivative_valid = finite & (jnp.abs(determinant) > 1.0e-12)
        qualified = finite & converged & parameters.physically_valid & derivative_valid
        status = jnp.where(
            ~finite,
            4,
            jnp.where(
                ~parameters.physically_valid,
                3,
                jnp.where(~derivative_valid, 2, jnp.where(~converged, 1, 0)),
            ),
        ).astype(jnp.int32)
        return MassiveFieldQuasiBoundResult(
            omega,
            seed,
            initial_residual,
            radial_log_derivative,
            residual,
            residual_norm,
            iterations,
            parameters,
            finite,
            converged,
            parameters.physically_valid,
            qualified,
            derivative_valid,
            status,
            self.plan_id,
        )

    def _horizon_quantities(self):
        root = jnp.sqrt(
            self.black_hole_mass**2 - self.black_hole_spin**2 - self.black_hole_charge**2
        )
        inner = self.black_hole_mass - root
        outer = self.black_hole_mass + root
        area_factor = outer**2 + self.black_hole_spin**2
        angular_velocity = self.black_hole_spin / area_factor
        electric_potential = self.black_hole_charge * outer / area_factor
        surface_gravity = (outer - inner) / (2.0 * area_factor)
        return inner, outer, angular_velocity, electric_potential, surface_gravity

    def _separation_constant(self, frequency):
        ell = self.ell
        azimuthal = self.azimuthal
        angular_average = (2 * ell * (ell + 1) - 2 * azimuthal**2 - 1) / (
            (2 * ell - 1) * (2 * ell + 3)
        )
        return ell * (ell + 1) + angular_average * self.black_hole_spin**2 * (
            self.field_mass**2 - frequency**2
        )

    def _riccati_derivative(self, radius, log_derivative, frequency):
        delta = (
            radius**2
            - 2.0 * self.black_hole_mass * radius
            + self.black_hole_spin**2
            + self.black_hole_charge**2
        )
        delta_prime = 2.0 * (radius - self.black_hole_mass)
        radial_phase = (
            (radius**2 + self.black_hole_spin**2) * frequency
            - self.azimuthal * self.black_hole_spin
            - self.field_charge * self.black_hole_charge * radius
        )
        separation = self._separation_constant(frequency)
        potential = (
            self.field_mass**2 * radius**2
            + self.black_hole_spin**2 * frequency**2
            - 2 * self.azimuthal * self.black_hole_spin * frequency
            + separation
        )
        return (
            -(log_derivative**2)
            - delta_prime / delta * log_derivative
            - radial_phase**2 / delta**2
            + potential / delta
        )

    def _shoot(self, frequency):
        parameters = self.mode_parameters(frequency)
        outer = parameters.horizon_radii[1]
        start = self.radial_nodes[0]
        initial_log_derivative = parameters.horizon_exponent / (start - outer)
        intervals = jnp.stack((self.radial_nodes[:-1], self.radial_nodes[1:]), axis=-1)

        def step(log_derivative, interval):
            left, right = interval
            width = right - left
            midpoint = left + 0.5 * width
            k1 = self._riccati_derivative(left, log_derivative, frequency)
            k2 = self._riccati_derivative(
                midpoint, log_derivative + 0.5 * width * k1, frequency
            )
            k3 = self._riccati_derivative(
                midpoint, log_derivative + 0.5 * width * k2, frequency
            )
            k4 = self._riccati_derivative(right, log_derivative + width * k3, frequency)
            candidate = log_derivative + width / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return candidate, candidate

        terminal, tail = jax.lax.scan(step, initial_log_derivative, intervals)
        path = jnp.concatenate((initial_log_derivative[None], tail))
        target = (
            -parameters.radial_decay_rate
            + parameters.infinity_power / self.radial_nodes[-1]
        )
        return path, terminal - target

    def _residual_jacobian(self, frequency):
        step = self.finite_difference_step * (1.0 + jnp.abs(frequency))
        residual_real_plus = self._shoot(frequency + step)[1]
        residual_real_minus = self._shoot(frequency - step)[1]
        residual_imag_plus = self._shoot(frequency + 1.0j * step)[1]
        residual_imag_minus = self._shoot(frequency - 1.0j * step)[1]
        derivative_real = (residual_real_plus - residual_real_minus) / (2.0 * step)
        derivative_imag = (residual_imag_plus - residual_imag_minus) / (2.0 * step)
        jacobian = jnp.asarray(
            (
                (jnp.real(derivative_real), jnp.real(derivative_imag)),
                (jnp.imag(derivative_real), jnp.imag(derivative_imag)),
            )
        )
        determinant = jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
        return jacobian, determinant


class ExcitationResidueResult(StrictModule):
    pole_frequency: Array
    residue: Array
    nearest_pole_separation: Array
    finite: Array
    converged: Array
    physically_valid: Array
    simple_poles: Array
    qualified: Array
    derivative_valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class GreenFunctionAssembly(StrictModule):
    coordinate: Array
    values: Array
    finite: Array
    converged: Array
    causal: Array
    stable_poles: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    status: Array
    domain: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)
    branch_cut_included: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class ExcitationResiduePlan(StrictModule, NonTrainableState):
    """Residues ``N(omega_n) / dW/domega(omega_n)`` at simple mode poles."""

    pole_frequency: Array
    excitation_numerator: Array
    wronskian_derivative: Array
    derivative_floor: float = eqx.field(static=True)
    pole_separation_floor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pole_frequency,
        excitation_numerator,
        wronskian_derivative,
        /,
        *,
        derivative_floor=1.0e-12,
        pole_separation_floor=1.0e-10,
    ):
        poles = np.asarray(pole_frequency, dtype=np.complex128)
        numerators = np.asarray(excitation_numerator, dtype=np.complex128)
        derivatives = np.asarray(wronskian_derivative, dtype=np.complex128)
        if (
            poles.ndim != 1
            or poles.size == 0
            or numerators.shape != poles.shape
            or derivatives.shape != poles.shape
            or np.any(~np.isfinite(poles))
            or np.any(~np.isfinite(numerators))
            or np.any(~np.isfinite(derivatives))
        ):
            raise ValueError(
                "Pole, numerator, and Wronskian arrays must be finite vectors."
            )
        if derivative_floor <= 0.0 or pole_separation_floor <= 0.0:
            raise ValueError("Residue qualification floors must be positive.")
        self.pole_frequency = jnp.asarray(poles)
        self.excitation_numerator = jnp.asarray(numerators)
        self.wronskian_derivative = jnp.asarray(derivatives)
        self.derivative_floor = float(derivative_floor)
        self.pole_separation_floor = float(pole_separation_floor)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "simple-pole-excitation-residues",
                "content": array_tree_fingerprint(
                    {
                        "pole_frequency": poles,
                        "excitation_numerator": numerators,
                        "wronskian_derivative": derivatives,
                    }
                ),
                "derivative_floor": float(derivative_floor),
                "pole_separation_floor": float(pole_separation_floor),
            }
        )

    def residues(self) -> ExcitationResidueResult:
        residue = self.excitation_numerator / self.wronskian_derivative
        distances = jnp.abs(self.pole_frequency[:, None] - self.pole_frequency[None, :])
        distances = jnp.where(
            jnp.eye(self.pole_frequency.size, dtype=jnp.bool_), jnp.inf, distances
        )
        nearest = jnp.min(distances, axis=1)
        nearest = jnp.where(self.pole_frequency.size == 1, jnp.inf, nearest)
        finite = jnp.all(jnp.isfinite(residue))
        simple = jnp.all(jnp.abs(self.wronskian_derivative) > self.derivative_floor)
        separated = jnp.all(nearest > self.pole_separation_floor)
        converged = finite
        physically_valid = simple & separated
        qualified = finite & converged & physically_valid
        derivative_valid = qualified
        status = jnp.where(
            ~finite, 3, jnp.where(~simple, 2, jnp.where(~separated, 1, 0))
        ).astype(jnp.int32)
        return ExcitationResidueResult(
            self.pole_frequency,
            residue,
            nearest,
            finite,
            converged,
            physically_valid,
            simple,
            qualified,
            derivative_valid,
            status,
            self.plan_id,
        )

    def time_domain(
        self, times: ArrayLike, source_weights: ArrayLike | None = None, /
    ) -> GreenFunctionAssembly:
        time = jnp.asarray(times)
        if time.ndim != 1:
            raise ValueError("Green-function times must be a one-dimensional grid.")
        weights = self._source_weights(source_weights)
        residue_result = self.residues()
        basis = jnp.exp(-1.0j * time[:, None] * self.pole_frequency[None, :])
        values = contract("tm,m->t", basis, residue_result.residue * weights)
        values = jnp.where(time >= 0.0, values, 0.0j)
        finite = jnp.all(jnp.isfinite(values))
        causal = jnp.all(jnp.where(time < 0.0, jnp.abs(values) == 0.0, True))
        stable = jnp.all(jnp.imag(self.pole_frequency) < 0.0)
        converged = residue_result.converged
        physically_valid = causal & stable & residue_result.physically_valid
        qualified = finite & converged & physically_valid
        derivative_valid = qualified
        status = jnp.where(
            ~finite,
            4,
            jnp.where(
                ~causal,
                3,
                jnp.where(~stable, 2, jnp.where(~residue_result.qualified, 1, 0)),
            ),
        ).astype(jnp.int32)
        return GreenFunctionAssembly(
            time,
            values,
            finite,
            converged,
            causal,
            stable,
            physically_valid,
            qualified,
            derivative_valid,
            status,
            "retarded-time",
            "simple-pole modal contribution; continuum and branch-cut tail excluded",
            False,
            self.plan_id,
        )

    def frequency_domain(
        self, frequencies: ArrayLike, source_weights: ArrayLike | None = None, /
    ) -> GreenFunctionAssembly:
        frequency = jnp.asarray(frequencies) + 0.0j
        if frequency.ndim != 1:
            raise ValueError("Green-function frequencies must be a one-dimensional grid.")
        weights = self._source_weights(source_weights)
        residue_result = self.residues()
        resolvent = 1.0 / (frequency[:, None] - self.pole_frequency[None, :])
        values = contract("fm,m->f", resolvent, residue_result.residue * weights)
        finite = jnp.all(jnp.isfinite(values))
        stable = jnp.all(jnp.imag(self.pole_frequency) < 0.0)
        converged = residue_result.converged
        physically_valid = residue_result.physically_valid
        qualified = finite & converged & physically_valid
        derivative_valid = qualified
        status = jnp.where(~finite, 2, jnp.where(~residue_result.qualified, 1, 0)).astype(
            jnp.int32
        )
        return GreenFunctionAssembly(
            frequency,
            values,
            finite,
            converged,
            jnp.asarray(True),
            stable,
            physically_valid,
            qualified,
            derivative_valid,
            status,
            "frequency-resolvent",
            "simple-pole modal contribution; continuum and branch-cut tail excluded",
            False,
            self.plan_id,
        )

    def _source_weights(self, source_weights):
        if source_weights is None:
            return jnp.ones_like(self.pole_frequency)
        weights = jnp.asarray(source_weights)
        if weights.shape != self.pole_frequency.shape:
            raise ValueError("source_weights must match the pole-frequency vector.")
        return weights


class NonlinearRingdownResult(StrictModule):
    times: Array
    linear_signal: Array
    quadratic_modes: Array
    total_signal: Array
    pair_detuning: Array
    resonant_pairs: Array
    finite: Array
    converged: Array
    stable: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class QuadraticRingdownPlan(StrictModule, NonTrainableState):
    """Second-order mode coupling for damped complex ringdown amplitudes.

    Each child mode solves ``db_k/dt + i omega_k b_k = sum C_kij a_i a_j``
    with zero second-order initial data.  The analytic Duhamel factor is used in
    a sinc form, including the exactly resonant limit without a singular branch.
    """

    pole_frequency: Array
    linear_amplitude: Array
    coupling: Array
    resonance_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pole_frequency,
        linear_amplitude,
        coupling,
        /,
        *,
        resonance_tolerance=1.0e-8,
    ):
        poles = np.asarray(pole_frequency, dtype=np.complex128)
        amplitudes = np.asarray(linear_amplitude, dtype=np.complex128)
        coefficients = np.asarray(coupling, dtype=np.complex128)
        if (
            poles.ndim != 1
            or poles.size == 0
            or amplitudes.shape != poles.shape
            or coefficients.shape != (poles.size, poles.size, poles.size)
            or np.any(~np.isfinite(poles))
            or np.any(~np.isfinite(amplitudes))
            or np.any(~np.isfinite(coefficients))
        ):
            raise ValueError(
                "Ringdown poles, amplitudes, and child-parent-parent coupling must be finite and shape-compatible."
            )
        if resonance_tolerance <= 0.0:
            raise ValueError("resonance_tolerance must be positive.")
        self.pole_frequency = jnp.asarray(poles)
        self.linear_amplitude = jnp.asarray(amplitudes)
        self.coupling = jnp.asarray(coefficients)
        self.resonance_tolerance = float(resonance_tolerance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "quadratic-ringdown-duhamel",
                "content": array_tree_fingerprint(
                    {
                        "pole_frequency": poles,
                        "linear_amplitude": amplitudes,
                        "coupling": coefficients,
                    }
                ),
                "resonance_tolerance": float(resonance_tolerance),
            }
        )

    def evaluate(self, times: ArrayLike, /) -> NonlinearRingdownResult:
        time = jnp.asarray(times)
        if time.ndim != 1:
            raise ValueError("Ringdown times must be a one-dimensional grid.")
        linear_basis = jnp.exp(-1.0j * time[:, None] * self.pole_frequency[None, :])
        linear_signal = contract("tm,m->t", linear_basis, self.linear_amplitude)
        pair_frequency = self.pole_frequency[:, None] + self.pole_frequency[None, :]
        detuning = pair_frequency[None, :, :] - self.pole_frequency[:, None, None]
        phase_argument = 0.5 * time[:, None, None, None] * detuning[None, :, :, :]
        safe_argument = jnp.where(
            jnp.abs(phase_argument) < self.resonance_tolerance,
            1.0 + 0.0j,
            phase_argument,
        )
        sinc = jnp.where(
            jnp.abs(phase_argument) < self.resonance_tolerance,
            1.0 - phase_argument**2 / 6.0 + phase_argument**4 / 120.0,
            jnp.sin(phase_argument) / safe_argument,
        )
        response = (
            time[:, None, None, None]
            * jnp.exp(
                -1.0j
                * time[:, None, None, None]
                * (
                    self.pole_frequency[None, :, None, None]
                    + 0.5 * detuning[None, :, :, :]
                )
            )
            * sinc
        )
        quadratic_modes = contract(
            "tkij,kij,i,j->tk",
            response,
            self.coupling,
            self.linear_amplitude,
            self.linear_amplitude,
        )
        total_signal = linear_signal + jnp.sum(quadratic_modes, axis=1)
        resonant = jnp.abs(detuning) <= self.resonance_tolerance
        finite = jnp.all(
            jnp.isfinite(
                jnp.concatenate(
                    (
                        jnp.real(total_signal),
                        jnp.imag(total_signal),
                        jnp.ravel(jnp.real(quadratic_modes)),
                        jnp.ravel(jnp.imag(quadratic_modes)),
                    )
                )
            )
        )
        stable = jnp.all(jnp.imag(self.pole_frequency) < 0.0)
        converged = finite
        physically_valid = stable
        qualified = finite & converged & physically_valid
        derivative_valid = qualified
        status = jnp.where(~finite, 2, jnp.where(~stable, 1, 0)).astype(jnp.int32)
        return NonlinearRingdownResult(
            time,
            linear_signal,
            quadratic_modes,
            total_signal,
            detuning,
            resonant,
            finite,
            converged,
            stable,
            physically_valid,
            qualified,
            derivative_valid,
            status,
            self.plan_id,
        )


def _positive_real_sqrt(value):
    root = jnp.sqrt(value)
    return jnp.where(
        (jnp.real(root) < 0.0) | ((jnp.real(root) == 0.0) & (jnp.imag(root) < 0.0)),
        -root,
        root,
    )


__all__ = [
    "ExcitationResiduePlan",
    "ExcitationResidueResult",
    "GreenFunctionAssembly",
    "KerrNewmanModeParameters",
    "MassiveFieldQuasiBoundPlan",
    "MassiveFieldQuasiBoundResult",
    "NonlinearRingdownResult",
    "QuadraticRingdownPlan",
]
