#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Gas diffusion: Epstein–Plesset dissolution and pinned surface nanobubbles.

Laplace pressure enters consistently: the gas amount is `(p0 + 2σ/R) V/(R_u T)`
and Henry's law gives the interface concentration `c_s (p0 + 2σ/R)/p0`.
Bulk-nanobubble stabilization mechanisms are not represented.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_never, Literal, TypeAlias

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import fixed_field, parameter_field
from .._validation import positive_integer
from ..solver import DifferentialProblem, DifferentialSolution, solve_diffrax
from ..typing import checked, parse
from ._contracts import MOLAR_GAS_CONSTANT, scalar_parameter
from ._interface import TolmanCorrectionPolicy
from ._status import BubbleDynamicsStatus
from ._validity import BubbleValidityPolicy


EpsteinPlessetRoute: TypeAlias = Literal["quasi_static", "full_history"]
SurfaceBubbleContact: TypeAlias = Literal["pinned", "unpinned"]


def _positive(value: float, name: str, /) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return number


def _save_schedule(save_times: ArrayLike, /) -> np.ndarray:
    times = np.asarray(save_times, dtype=np.float64)
    if times.ndim != 1 or times.shape[0] == 0 or not np.all(np.isfinite(times)):
        raise ValueError("save_times must be a finite non-empty rank-1 array.")
    if times[0] < 0.0 or times[-1] <= 0.0:
        raise ValueError("save_times must be nonnegative and end after t = 0.")
    if times.shape[0] > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("save_times must be strictly increasing.")
    return times


class GasSolutionProperties(StrictModule):
    """Dissolved-gas transport and saturation properties of the liquid.

    `saturation_concentration` (mol m⁻³) is the Henry's-law concentration in
    equilibrium with the pure gas at `ambient_pressure`; `saturation_ratio` is
    the far-field concentration divided by it (1 is saturated).
    """

    diffusivity: Array = parameter_field()
    saturation_concentration: Array = parameter_field()
    saturation_ratio: Array = parameter_field()
    surface_tension: Array = parameter_field()
    ambient_pressure: Array = parameter_field()
    temperature: Array = parameter_field()

    def __init__(
        self,
        diffusivity: ArrayLike,
        saturation_concentration: ArrayLike,
        saturation_ratio: ArrayLike,
        surface_tension: ArrayLike,
        ambient_pressure: ArrayLike,
        temperature: ArrayLike,
        /,
    ) -> None:
        self.diffusivity = scalar_parameter(diffusivity, "diffusivity", lower=0.0)
        self.saturation_concentration = scalar_parameter(
            saturation_concentration, "saturation_concentration", lower=0.0
        )
        self.saturation_ratio = scalar_parameter(
            saturation_ratio, "saturation_ratio", lower=0.0, inclusive=True
        )
        self.surface_tension = scalar_parameter(
            surface_tension, "surface_tension", lower=0.0, inclusive=True
        )
        self.ambient_pressure = scalar_parameter(ambient_pressure, "ambient_pressure", lower=0.0)
        self.temperature = scalar_parameter(temperature, "temperature", lower=0.0)

    def gas_molar_density(self) -> Array:
        """Ideal-gas molar density at the ambient pressure."""
        return self.ambient_pressure / (MOLAR_GAS_CONSTANT * self.temperature)


def _surface_tension(
    properties: GasSolutionProperties,
    tolman: TolmanCorrectionPolicy | None,
    radius: Array,
    /,
) -> Array:
    if tolman is None:
        return jnp.broadcast_to(properties.surface_tension, jnp.shape(radius))
    return tolman.surface_tension(properties.surface_tension, radius)


class DissolutionEvidence(StrictModule):
    """Lifetime, conservation, saturation and continuum-validity evidence.

    `lifetime` is infinite when the bubble did not dissolve before the final
    save time. `amount_residual` compares the gas amount implied by the final
    geometry with the integrated diffusive flux, relative to the initial amount.
    `saturation_margin` is `(c_interface − c_∞)/c_s` at the start and the end.
    """

    solver_successful: Array
    accepted_steps: Array
    rejected_steps: Array
    dissolved: Array
    lifetime: Array
    amount_residual: Array
    initial_saturation_margin: Array
    final_saturation_margin: Array
    max_laplace_ratio: Array
    max_knudsen: Array | None
    max_tolman_ratio: Array | None
    continuum_support: Array
    history_kernel_error: float = eqx.field(static=True)
    route: str = eqx.field(static=True)


class GasDissolutionState(StrictModule):
    """Bubble radius, integrated amount change and diffusion-history state."""

    radius: Array
    amount_change: Array
    history: Array


class GasDissolutionResult(StrictModule):
    """Radius and gas-amount trajectory of a dissolving spherical bubble."""

    times: Array
    radius: Array
    amount: Array
    interface_concentration: Array
    valid: Array
    terminal_state: GasDissolutionState
    terminal_time: Array
    status: Array
    evidence: DissolutionEvidence
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        """Whether the solve reached the final time or complete dissolution."""
        return (
            (self.status == int(BubbleDynamicsStatus.SUCCESS))
            | (self.status == int(BubbleDynamicsStatus.DISSOLVED))
        ) & self.evidence.solver_successful


def _history_kernel(
    terms: int, horizon: float, lag_ratio: float, /
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sum-of-exponentials `1/√(π t) ≈ Σ w_j exp(−λ_j t)` on `[lag_ratio·T, T]`.

    Trapezoidal rule on `1/√(πt) = (1/π) ∫ exp(−e^u t + u/2) du`; the maximal
    relative kernel error over a logarithmic lag grid is returned as evidence.
    """
    shortest = lag_ratio * horizon
    lower = np.log(1.0e-12 / horizon)
    upper = np.log(36.0 / shortest)
    nodes = np.linspace(lower, upper, terms)
    step = nodes[1] - nodes[0]
    rates = np.exp(nodes)
    weights = step / np.pi * np.exp(nodes / 2.0)
    lags = np.geomspace(shortest, horizon, 512)
    approximation = np.exp(-np.outer(lags, rates)) @ weights
    exact = 1.0 / np.sqrt(np.pi * lags)
    error = float(np.max(np.abs(approximation / exact - 1.0)))
    return rates, weights, error


class EpsteinPlessetPlan(StrictModule):
    """Isothermal diffusive dissolution or growth of a free spherical bubble.

    Routes (static, recorded as provenance):

    - `"quasi_static"`: flux `D Δc/R` of the quasi-stationary concentration
      field. With zero surface tension its lifetime is the closed form
      `ρ_g R₀²/(2 D c_s (1 − f))`.
    - `"full_history"`: Epstein & Plesset (1950) fixed-sphere diffusion with
      the complete Duhamel history of the Laplace-dependent interface
      concentration, `J = D Δc/R + √D [Δc(0)/√(πt) + ∫₀ᵗ Δc'(s)/√(π(t−s)) ds]`;
      boundary motion in the diffusion field is neglected as in the original
      derivation. The history kernel is a sum of `history_terms` exponentials
      with its reported relative error on lags in `[lag_ratio·T, T]`.

    Time is integrated in `s = √(t/t_D)` with `t_D = R₀²/D`, which regularizes
    the `1/√t` start-up flux exactly.
    """

    properties: GasSolutionProperties
    tolman: TolmanCorrectionPolicy | None
    validity: BubbleValidityPolicy
    save_times: Array = fixed_field()
    history_rates: Array = fixed_field()
    history_weights: Array = fixed_field()
    route: EpsteinPlessetRoute = eqx.field(static=True)
    dissolution_radius_ratio: float = eqx.field(static=True)
    history_kernel_error: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        properties: GasSolutionProperties,
        save_times: ArrayLike,
        /,
        *,
        route: EpsteinPlessetRoute,
        initial_radius: float,
        tolman: TolmanCorrectionPolicy | None = None,
        validity: BubbleValidityPolicy | None = None,
        dissolution_radius_ratio: float = 1.0e-3,
        history_terms: int = 96,
        history_lag_ratio: float = 1.0e-10,
        relative_tolerance: float = 1.0e-9,
        absolute_tolerance: float = 1.0e-11,
        maximum_steps: int = 16384,
    ) -> None:
        if tolman is not None and not isinstance(tolman, TolmanCorrectionPolicy):
            raise TypeError("tolman must be a TolmanCorrectionPolicy or None.")
        support = BubbleValidityPolicy() if validity is None else validity
        if not isinstance(support, BubbleValidityPolicy):
            raise TypeError("validity must be a BubbleValidityPolicy or None.")
        selected = parse(route, EpsteinPlessetRoute, "route")
        times = _save_schedule(save_times)
        radius = _positive(initial_radius, "initial_radius")
        ratio = _positive(dissolution_radius_ratio, "dissolution_radius_ratio")
        if ratio >= 1.0:
            raise ValueError("dissolution_radius_ratio must be below 1.")
        terms = positive_integer(history_terms, "history_terms")
        lag_ratio = _positive(history_lag_ratio, "history_lag_ratio")
        match selected:
            case "quasi_static":
                rates = np.zeros((0,))
                weights = np.zeros((0,))
                kernel_error = 0.0
            case "full_history":
                if terms < 8:
                    raise ValueError("history_terms must be at least 8.")
                horizon = float(times[-1]) * float(properties.diffusivity) / radius**2
                rates, weights, kernel_error = _history_kernel(terms, horizon, lag_ratio)
            case _:
                assert_never(selected)
        self.properties = properties
        self.tolman = tolman
        self.validity = support
        self.save_times = jnp.asarray(times, dtype=jnp.float64)
        self.history_rates = jnp.asarray(rates, dtype=jnp.float64)
        self.history_weights = jnp.asarray(weights, dtype=jnp.float64)
        self.route = selected
        self.dissolution_radius_ratio = ratio
        self.history_kernel_error = kernel_error
        self.relative_tolerance = _positive(relative_tolerance, "relative_tolerance")
        self.absolute_tolerance = _positive(absolute_tolerance, "absolute_tolerance")
        self.maximum_steps = positive_integer(maximum_steps, "maximum_steps")
        self.plan_id = canonical_fingerprint(
            {
                "kind": "epstein-plesset-plan",
                "route": selected,
                "save_times": times,
                "initial_radius": radius,
                "tolman": tolman is not None,
                "validity": support.policy_id,
                "dissolution_radius_ratio": ratio,
                "history_terms": int(rates.shape[0]),
                "history_lag_ratio": lag_ratio,
                "relative_tolerance": self.relative_tolerance,
                "absolute_tolerance": self.absolute_tolerance,
                "maximum_steps": self.maximum_steps,
            }
        )

    def gas_amount(self, radius: Array, /) -> Array:
        """Gas amount of a free spherical bubble including its Laplace pressure."""
        properties = self.properties
        sigma = _surface_tension(properties, self.tolman, radius)
        pressure = properties.ambient_pressure + 2.0 * sigma / radius
        return pressure * 4.0 * jnp.pi * radius**3 / (3.0 * MOLAR_GAS_CONSTANT * properties.temperature)

    def interface_concentration(self, radius: Array, /) -> Array:
        """Henry's-law concentration at the Laplace-pressurized interface."""
        properties = self.properties
        sigma = _surface_tension(properties, self.tolman, radius)
        return properties.saturation_concentration * (
            1.0 + 2.0 * sigma / (radius * properties.ambient_pressure)
        )

    def prepare(self, initial_radius: ArrayLike, /) -> PreparedEpsteinPlesset:
        """Initial state, diffusion time scale and admissibility."""
        radius = jnp.asarray(initial_radius, dtype=jnp.float64)
        state = GasDissolutionState(
            radius,
            jnp.zeros_like(radius),
            jnp.zeros(self.history_rates.shape, dtype=jnp.float64),
        )
        amount = self.gas_amount(radius)
        admissible = (radius > 0.0) & jnp.isfinite(amount) & (amount > 0.0)
        return PreparedEpsteinPlesset(
            self, state, radius**2 / self.properties.diffusivity, admissible
        )


class PreparedEpsteinPlesset(StrictModule):
    """Initial dissolution state and the diffusion time scale `R₀²/D`."""

    plan: EpsteinPlessetPlan
    initial_state: GasDissolutionState
    diffusion_time: Array
    admissible: Array


def _dissolution_field(
    prepared: PreparedEpsteinPlesset, root_time: Array, flat: Array, /
) -> Array:
    """Nondimensional `d/ds` of `[R/R₀, Δn/n₀, history/c_s]`, `s = √(t/t_D)`."""
    plan = prepared.plan
    properties = plan.properties
    initial_radius = prepared.initial_state.radius
    saturation = properties.saturation_concentration
    radius = flat[0] * initial_radius
    history = flat[2:]
    amount_slope = jax.grad(plan.gas_amount)(radius)
    concentration_slope = jax.grad(plan.interface_concentration)(radius)
    far_field = properties.saturation_ratio * saturation
    excess = (plan.interface_concentration(radius) - far_field) / saturation
    initial_excess = (plan.interface_concentration(initial_radius) - far_field) / saturation
    # s·J̃ with J = (D c_s/R₀) J̃; the start-up term Δc₀/√(π t̃) times s is finite.
    quasi_static = root_time * excess * initial_radius / radius
    match plan.route:
        case "quasi_static":
            scaled_flux = quasi_static
        case "full_history":
            scaled_flux = (
                quasi_static
                + initial_excess / jnp.sqrt(jnp.pi)
                + root_time * jnp.sum(plan.history_weights * history)
            )
        case _:
            assert_never(plan.route)
    amount_rate = -8.0 * jnp.pi * radius**2 * initial_radius * saturation * scaled_flux
    radius_rate = amount_rate / amount_slope
    history_rate = (
        -2.0 * root_time * plan.history_rates * history
        + concentration_slope * radius_rate / saturation
    )
    initial_amount = plan.gas_amount(initial_radius)
    return jnp.concatenate(
        (
            jnp.stack((radius_rate / initial_radius, amount_rate / initial_amount)),
            history_rate,
        )
    )


class _RadiusFloor(StrictModule):
    ratio: float = eqx.field(static=True)

    def __call__(self, t: Array, y: Array, args: object, **kwargs: object) -> Array:
        del t, args, kwargs
        return y[0] - self.ratio


class _NonfiniteState(StrictModule):
    def __call__(self, t: Array, y: Array, args: object, **kwargs: object) -> Array:
        del t, args, kwargs
        return ~jnp.all(jnp.isfinite(y)) | (y[0] <= 0.0)


def _terminal_solve(
    field: Callable[[Array, Array, object], Array],
    initial: Array,
    end: Array,
    conditions: tuple[StrictModule, ...],
    directions: tuple[bool | None, ...],
    *,
    solver: dfx.AbstractSolver,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_steps: int,
    problem_id: str,
) -> DifferentialSolution:
    problem = DifferentialProblem(field, initial, t0=0.0, t1=end, problem_id=problem_id)
    return solve_diffrax(
        problem,
        save_times=end[None],
        solver=solver,
        event=dfx.Event(
            conditions,
            root_finder=optx.Newton(rtol=1.0e-10, atol=1.0e-10),
            direction=directions,
        ),
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        dense=True,
        max_steps=maximum_steps,
        throw=False,
        solver_configuration_id=problem_id,
    )


def _dissolution_solver(route: EpsteinPlessetRoute, /) -> dfx.AbstractSolver:
    match route:
        case "quasi_static":
            return dfx.Tsit5()
        case "full_history":
            # The sum-of-exponentials history modes span many decades of rates.
            return dfx.Kvaerno5()
        case _:
            assert_never(route)


def _terminal_status(solution: DifferentialSolution, dissolved: Array, /) -> Array:
    failed = ~solution.backend_successful
    exhausted = solution.backend_result == dfx.RESULTS.max_steps_reached
    return jnp.where(
        failed,
        jnp.where(
            exhausted,
            int(BubbleDynamicsStatus.MAX_STEPS),
            int(BubbleDynamicsStatus.SOLVER_FAILURE),
        ),
        jnp.where(
            dissolved,
            int(BubbleDynamicsStatus.DISSOLVED),
            jnp.where(
                solution.event_terminated,
                int(BubbleDynamicsStatus.INVALID_STATE),
                int(BubbleDynamicsStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)


def _extreme(values: Array, valid: Array, /) -> Array:
    return jnp.max(jnp.where(valid, values, -jnp.inf))


@eqx.filter_jit
def solve_epstein_plesset(prepared: PreparedEpsteinPlesset, /) -> GasDissolutionResult:
    """Integrate Epstein–Plesset dissolution or growth to the last save time."""
    if not isinstance(prepared, PreparedEpsteinPlesset):
        raise TypeError("prepared must be a PreparedEpsteinPlesset.")
    plan = prepared.plan
    properties = plan.properties
    initial_radius = prepared.initial_state.radius
    initial = jnp.concatenate(
        (
            jnp.stack((jnp.ones_like(initial_radius), jnp.zeros_like(initial_radius))),
            prepared.initial_state.history / properties.saturation_concentration,
        )
    )
    root_times = jnp.sqrt(plan.save_times / prepared.diffusion_time)

    def field(time: Array, flat: Array, args: object) -> Array:
        del args
        return _dissolution_field(prepared, time, flat)

    solution = _terminal_solve(
        field,
        initial,
        root_times[-1],
        (_RadiusFloor(plan.dissolution_radius_ratio), _NonfiniteState()),
        (False, None),
        solver=_dissolution_solver(plan.route),
        relative_tolerance=plan.relative_tolerance,
        absolute_tolerance=plan.absolute_tolerance,
        maximum_steps=plan.maximum_steps,
        problem_id=f"{plan.plan_id}:dissolution",
    )
    mask = jnp.stack(tuple(jnp.asarray(value, dtype=jnp.bool_) for value in solution.event_mask))
    dissolved = solution.event_terminated & mask[0] & solution.backend_successful
    status = jnp.where(
        prepared.admissible,
        _terminal_status(solution, dissolved),
        int(BubbleDynamicsStatus.INVALID_EQUILIBRIUM),
    ).astype(jnp.int32)
    terminal_root = solution.terminal_time
    valid = (root_times <= terminal_root) & prepared.admissible
    values = solution.evaluate(jnp.clip(root_times, 0.0, terminal_root))
    radius = jnp.where(valid, values[:, 0] * initial_radius, jnp.nan)
    safe_radius = jnp.where(valid, radius, initial_radius)
    amount = jnp.where(valid, jax.vmap(plan.gas_amount)(safe_radius), jnp.nan)
    concentration = jnp.where(valid, jax.vmap(plan.interface_concentration)(safe_radius), jnp.nan)
    terminal = solution.terminal_state
    terminal_radius = terminal[0] * initial_radius
    initial_amount = plan.gas_amount(initial_radius)
    amount_residual = (
        plan.gas_amount(terminal_radius) - initial_amount - terminal[1] * initial_amount
    ) / initial_amount
    far_field = properties.saturation_ratio * properties.saturation_concentration
    margin_initial = (
        plan.interface_concentration(initial_radius) - far_field
    ) / properties.saturation_concentration
    margin_final = (
        plan.interface_concentration(terminal_radius) - far_field
    ) / properties.saturation_concentration
    rows = jnp.concatenate((safe_radius, terminal_radius[None]))
    row_valid = jnp.concatenate((valid, jnp.asarray([True])))
    sigma = _surface_tension(properties, plan.tolman, rows)
    laplace = 2.0 * sigma / (rows * properties.ambient_pressure)
    knudsen = plan.validity.knudsen_number(
        properties.temperature, properties.ambient_pressure + 2.0 * sigma / rows, rows
    )
    tolman_ratio = plan.validity.tolman_ratio(rows)
    max_knudsen = None if knudsen is None else _extreme(knudsen, row_valid)
    max_tolman = None if tolman_ratio is None else _extreme(tolman_ratio, row_valid)
    continuum = jnp.asarray(True)
    if max_knudsen is not None:
        continuum = continuum & (max_knudsen <= plan.validity.knudsen_limit)
    if max_tolman is not None:
        continuum = continuum & (max_tolman <= plan.validity.tolman_ratio_limit)
    terminal_time = terminal_root**2 * prepared.diffusion_time
    evidence = DissolutionEvidence(
        solution.backend_successful,
        jnp.asarray(solution.stats["num_accepted_steps"], dtype=jnp.int32),
        jnp.asarray(solution.stats["num_rejected_steps"], dtype=jnp.int32),
        dissolved,
        jnp.where(dissolved, terminal_time, jnp.inf),
        amount_residual,
        margin_initial,
        margin_final,
        _extreme(laplace, row_valid),
        max_knudsen,
        max_tolman,
        continuum,
        history_kernel_error=plan.history_kernel_error,
        route=plan.route,
    )
    return GasDissolutionResult(
        plan.save_times,
        radius,
        amount,
        concentration,
        valid,
        GasDissolutionState(
            terminal_radius, terminal[1] * initial_amount, terminal[2:] * properties.saturation_concentration
        ),
        terminal_time,
        status,
        evidence,
        plan_id=plan.plan_id,
    )


def quasi_static_dissolution_time(
    initial_radius: ArrayLike,
    properties: GasSolutionProperties,
    /,
) -> Array:
    """Closed-form quasi-static lifetime `ρ_g R₀²/(2 D c_s (1 − f))`.

    Valid only for the quasi-static route without surface tension or history.
    """
    radius = jnp.asarray(initial_radius, dtype=jnp.float64)
    return (
        properties.gas_molar_density()
        * radius**2
        / (
            2.0
            * properties.diffusivity
            * properties.saturation_concentration
            * (1.0 - properties.saturation_ratio)
        )
    )


def popov_flux_factor(gas_contact_angle: ArrayLike, /, *, nodes: int = 64) -> Array:
    """Popov (2005) total diffusive flux factor of a spherical cap on a wall.

    `ṅ = −π a D Δc f(θ)` for footprint radius `a` and cap angle `θ` measured
    inside the cap: `f = sinθ/(1 + cosθ) + 4∫₀^∞ (1 + cosh 2θτ)/sinh 2πτ ·
    tanh((π − θ)τ) dτ`; `f(π/2) = 2` (hemisphere) and `f(0) = 4/π` (disk).
    """
    angle = jnp.asarray(gas_contact_angle, dtype=jnp.float64)
    unit, weights = np.polynomial.legendre.leggauss(nodes)
    mapped = (unit + 1.0) / 2.0
    tau = jnp.asarray(mapped / (1.0 - mapped), dtype=jnp.float64)
    jacobian = jnp.asarray(0.5 * weights / (1.0 - mapped) ** 2, dtype=jnp.float64)
    angle_ = angle[..., None]
    # (1 + cosh 2θτ)/sinh 2πτ written with decaying exponentials for large τ.
    numerator = jnp.exp(-2.0 * jnp.pi * tau) + 0.5 * (
        jnp.exp((2.0 * angle_ - 2.0 * jnp.pi) * tau) + jnp.exp((-2.0 * angle_ - 2.0 * jnp.pi) * tau)
    )
    denominator = 0.5 * (1.0 - jnp.exp(-4.0 * jnp.pi * tau))
    integrand = numerator / denominator * jnp.tanh((jnp.pi - angle_) * tau)
    return jnp.sin(angle) / (1.0 + jnp.cos(angle)) + 4.0 * jnp.sum(jacobian * integrand, axis=-1)


class SurfaceBubbleState(StrictModule):
    """Footprint diameter and gas-side cap angle of a surface bubble."""

    footprint_diameter: Array
    gas_contact_angle: Array
    amount_change: Array

    @property
    def liquid_contact_angle(self) -> Array:
        """Contact angle measured through the liquid, `π − θ_gas`."""
        return jnp.pi - self.gas_contact_angle


class SurfaceBubbleEquilibrium(StrictModule):
    """Lohse–Zhang equilibrium of a pinned cap and its linear stability.

    `sin θ_gas = a/R_c` with `R_c = 2σ/(ζ p0) − 2δ` (Tolman length `δ`, zero
    without a Tolman policy); without Tolman this is `sin θ_gas = ζ L/L_c`,
    `L_c = 4σ/p0`. `stability_derivative = ∂θ̇/∂θ` is negative when stable.
    """

    exists: Array
    gas_contact_angle: Array
    liquid_contact_angle: Array
    curvature_radius: Array
    stability_derivative: Array
    stable: Array


def _cap_volume(curvature_radius: Array, gas_angle: Array, /) -> Array:
    return (
        jnp.pi
        * curvature_radius**3
        / 3.0
        * (2.0 + jnp.cos(gas_angle))
        * (1.0 - jnp.cos(gas_angle)) ** 2
    )


class PinnedSurfaceBubblePlan(StrictModule):
    """Diffusive evolution of a spherical-cap surface bubble (Lohse & Zhang 2015).

    Geometry uses the footprint diameter `L` (footprint radius `a = L/2`) and
    the gas-side cap angle `θ` (the liquid-side contact angle is `π − θ`);
    `R_c = a/sin θ`. Laplace pressure `2σ/R_c` sets both the gas amount and the
    Henry's-law interface concentration; the quasi-static flux is Popov's
    `ṅ = −π a D Δc f(θ)` with oversaturation `ζ = c_∞/c_s − 1`.

    - `"pinned"`: `L` fixed, `θ` evolves; stops with `SUPPORT_EXIT` when the cap
      reaches a hemisphere (beyond it the pinned branch is unstable) and with
      `DISSOLVED` when `θ` falls below `dissolution_ratio·θ₀`.
    - `"unpinned"`: `θ` fixed, `L` evolves; `DISSOLVED` below
      `dissolution_ratio·L₀`.
    """

    properties: GasSolutionProperties
    oversaturation: Array = parameter_field()
    tolman: TolmanCorrectionPolicy | None
    validity: BubbleValidityPolicy
    save_times: Array = fixed_field()
    contact: SurfaceBubbleContact = eqx.field(static=True)
    dissolution_ratio: float = eqx.field(static=True)
    flux_nodes: int = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        properties: GasSolutionProperties,
        oversaturation: ArrayLike,
        save_times: ArrayLike,
        /,
        *,
        contact: SurfaceBubbleContact,
        tolman: TolmanCorrectionPolicy | None = None,
        validity: BubbleValidityPolicy | None = None,
        dissolution_ratio: float = 1.0e-3,
        flux_nodes: int = 64,
        relative_tolerance: float = 1.0e-9,
        absolute_tolerance: float = 1.0e-11,
        maximum_steps: int = 16384,
    ) -> None:
        if tolman is not None and not isinstance(tolman, TolmanCorrectionPolicy):
            raise TypeError("tolman must be a TolmanCorrectionPolicy or None.")
        support = BubbleValidityPolicy() if validity is None else validity
        if not isinstance(support, BubbleValidityPolicy):
            raise TypeError("validity must be a BubbleValidityPolicy or None.")
        selected = parse(contact, SurfaceBubbleContact, "contact")
        times = _save_schedule(save_times)
        ratio = _positive(dissolution_ratio, "dissolution_ratio")
        if ratio >= 1.0:
            raise ValueError("dissolution_ratio must be below 1.")
        self.properties = properties
        self.oversaturation = scalar_parameter(oversaturation, "oversaturation", lower=-1.0)
        self.tolman = tolman
        self.validity = support
        self.save_times = jnp.asarray(times, dtype=jnp.float64)
        self.contact = selected
        self.dissolution_ratio = ratio
        self.flux_nodes = positive_integer(flux_nodes, "flux_nodes")
        self.relative_tolerance = _positive(relative_tolerance, "relative_tolerance")
        self.absolute_tolerance = _positive(absolute_tolerance, "absolute_tolerance")
        self.maximum_steps = positive_integer(maximum_steps, "maximum_steps")
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pinned-surface-bubble-plan",
                "contact": selected,
                "save_times": times,
                "tolman": tolman is not None,
                "validity": support.policy_id,
                "dissolution_ratio": ratio,
                "flux_nodes": self.flux_nodes,
                "relative_tolerance": self.relative_tolerance,
                "absolute_tolerance": self.absolute_tolerance,
                "maximum_steps": self.maximum_steps,
            }
        )

    def laplace_pressure(self, curvature_radius: Array, /) -> Array:
        """Laplace pressure `2σ(R_c)/R_c` of the cap."""
        sigma = _surface_tension(self.properties, self.tolman, curvature_radius)
        return 2.0 * sigma / curvature_radius

    def gas_amount(self, footprint_diameter: Array, gas_angle: Array, /) -> Array:
        """Gas amount in the cap including its Laplace pressure."""
        properties = self.properties
        curvature = footprint_diameter / (2.0 * jnp.sin(gas_angle))
        pressure = properties.ambient_pressure + self.laplace_pressure(curvature)
        return (
            pressure * _cap_volume(curvature, gas_angle)
            / (MOLAR_GAS_CONSTANT * properties.temperature)
        )

    def concentration_excess(self, footprint_diameter: Array, gas_angle: Array, /) -> Array:
        """`(c_interface − c_∞)/c_s = Δp_L/p0 − ζ`."""
        curvature = footprint_diameter / (2.0 * jnp.sin(gas_angle))
        return (
            self.laplace_pressure(curvature) / self.properties.ambient_pressure
            - self.oversaturation
        )

    def amount_rate(self, footprint_diameter: Array, gas_angle: Array, /) -> Array:
        """Popov diffusive amount rate `−π a D c_s (Δp_L/p0 − ζ) f(θ)`."""
        properties = self.properties
        return (
            -jnp.pi
            * footprint_diameter
            / 2.0
            * properties.diffusivity
            * properties.saturation_concentration
            * self.concentration_excess(footprint_diameter, gas_angle)
            * popov_flux_factor(gas_angle, nodes=self.flux_nodes)
        )

    def angle_rate(self, footprint_diameter: Array, gas_angle: Array, /) -> Array:
        """Pinned `θ̇ = ṅ/(∂n/∂θ)` at fixed footprint."""
        slope = jax.grad(self.gas_amount, argnums=1)(footprint_diameter, gas_angle)
        return self.amount_rate(footprint_diameter, gas_angle) / slope

    def equilibrium(self, footprint_diameter: ArrayLike, /) -> SurfaceBubbleEquilibrium:
        """Pinned equilibrium angle on the flat (stable) branch and its stability."""
        diameter = jnp.asarray(footprint_diameter, dtype=jnp.float64)
        properties = self.properties
        tolman_length = (
            jnp.zeros_like(diameter) if self.tolman is None else self.tolman.tolman_length
        )
        curvature = (
            2.0 * properties.surface_tension / (self.oversaturation * properties.ambient_pressure)
            - 2.0 * tolman_length
        )
        sine = diameter / (2.0 * curvature)
        exists = (self.oversaturation > 0.0) & (curvature > 0.0) & (sine > 0.0) & (sine <= 1.0)
        angle = jnp.arcsin(jnp.clip(sine, 0.0, 1.0))
        derivative = jax.grad(self.angle_rate, argnums=1)(diameter, angle)
        return SurfaceBubbleEquilibrium(
            exists,
            angle,
            jnp.pi - angle,
            curvature,
            derivative,
            exists & (derivative < 0.0),
        )

    def prepare(
        self, footprint_diameter: ArrayLike, liquid_contact_angle: ArrayLike, /
    ) -> PreparedSurfaceBubble:
        """Initial cap from its footprint diameter and liquid-side contact angle."""
        diameter = jnp.asarray(footprint_diameter, dtype=jnp.float64)
        gas_angle = jnp.pi - jnp.asarray(liquid_contact_angle, dtype=jnp.float64)
        state = SurfaceBubbleState(diameter, gas_angle, jnp.zeros_like(diameter))
        admissible = (
            (diameter > 0.0)
            & (gas_angle > 0.0)
            & (gas_angle < jnp.pi / 2.0)
            & jnp.isfinite(self.gas_amount(diameter, gas_angle))
        )
        return PreparedSurfaceBubble(
            self, state, (diameter / 2.0) ** 2 / self.properties.diffusivity, admissible
        )


class PreparedSurfaceBubble(StrictModule):
    """Initial cap, diffusion time `a²/D` and admissibility."""

    plan: PinnedSurfaceBubblePlan
    initial_state: SurfaceBubbleState
    diffusion_time: Array
    admissible: Array


class SurfaceBubbleEvidence(StrictModule):
    """Lifetime, conservation, equilibrium and continuum evidence of a surface bubble."""

    solver_successful: Array
    accepted_steps: Array
    rejected_steps: Array
    dissolved: Array
    lifetime: Array
    amount_residual: Array
    initial_saturation_margin: Array
    final_saturation_margin: Array
    equilibrium: SurfaceBubbleEquilibrium
    max_laplace_ratio: Array
    max_knudsen: Array | None
    max_tolman_ratio: Array | None
    continuum_support: Array


class SurfaceBubbleResult(StrictModule):
    """Trajectory of the cap geometry with explicit contact-angle conventions."""

    times: Array
    footprint_diameter: Array
    gas_contact_angle: Array
    liquid_contact_angle: Array
    curvature_radius: Array
    amount: Array
    valid: Array
    terminal_state: SurfaceBubbleState
    terminal_time: Array
    status: Array
    evidence: SurfaceBubbleEvidence
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        """Whether the solve reached the final time or complete dissolution."""
        return (
            (self.status == int(BubbleDynamicsStatus.SUCCESS))
            | (self.status == int(BubbleDynamicsStatus.DISSOLVED))
        ) & self.evidence.solver_successful


class _CapFloor(StrictModule):
    ratio: float = eqx.field(static=True)
    index: int = eqx.field(static=True)

    def __call__(self, t: Array, y: Array, args: object, **kwargs: object) -> Array:
        del t, args, kwargs
        return y[self.index] - self.ratio


class _Hemisphere(StrictModule):
    initial_angle: Array

    def __call__(self, t: Array, y: Array, args: object, **kwargs: object) -> Array:
        del t, args, kwargs
        return jnp.pi / 2.0 - y[1] * self.initial_angle


def _surface_field(prepared: PreparedSurfaceBubble, flat: Array, /) -> Array:
    """Nondimensional rates of `[L/L₀, θ/θ₀, Δn/n₀]` in time units of `a₀²/D`."""
    plan = prepared.plan
    initial = prepared.initial_state
    diameter = flat[0] * initial.footprint_diameter
    angle = flat[1] * initial.gas_contact_angle
    amount_rate = plan.amount_rate(diameter, angle)
    match plan.contact:
        case "pinned":
            slope = jax.grad(plan.gas_amount, argnums=1)(diameter, angle)
            diameter_rate = jnp.zeros_like(diameter)
            angle_rate = amount_rate / slope
        case "unpinned":
            slope = jax.grad(plan.gas_amount, argnums=0)(diameter, angle)
            diameter_rate = amount_rate / slope
            angle_rate = jnp.zeros_like(angle)
        case _:
            assert_never(plan.contact)
    initial_amount = plan.gas_amount(initial.footprint_diameter, initial.gas_contact_angle)
    return prepared.diffusion_time * jnp.stack(
        (
            diameter_rate / initial.footprint_diameter,
            angle_rate / initial.gas_contact_angle,
            amount_rate / initial_amount,
        )
    )


@eqx.filter_jit
def solve_surface_bubble(prepared: PreparedSurfaceBubble, /) -> SurfaceBubbleResult:
    """Integrate a pinned or unpinned surface bubble to the last save time."""
    if not isinstance(prepared, PreparedSurfaceBubble):
        raise TypeError("prepared must be a PreparedSurfaceBubble.")
    plan = prepared.plan
    properties = plan.properties
    initial = prepared.initial_state
    scaled_times = plan.save_times / prepared.diffusion_time

    def field(time: Array, flat: Array, args: object) -> Array:
        del time, args
        return _surface_field(prepared, flat)

    match plan.contact:
        case "pinned":
            shrinking_index = 1
        case "unpinned":
            shrinking_index = 0
        case _:
            assert_never(plan.contact)
    conditions: tuple[StrictModule, ...] = (
        _CapFloor(plan.dissolution_ratio, shrinking_index),
        _Hemisphere(initial.gas_contact_angle),
        _NonfiniteState(),
    )
    solution = _terminal_solve(
        field,
        jnp.stack((jnp.ones_like(initial.footprint_diameter), jnp.ones_like(initial.footprint_diameter), jnp.zeros_like(initial.footprint_diameter))),
        scaled_times[-1],
        conditions,
        (False, False, None),
        solver=dfx.Tsit5(),
        relative_tolerance=plan.relative_tolerance,
        absolute_tolerance=plan.absolute_tolerance,
        maximum_steps=plan.maximum_steps,
        problem_id=f"{plan.plan_id}:surface",
    )
    mask = jnp.stack(tuple(jnp.asarray(value, dtype=jnp.bool_) for value in solution.event_mask))
    event = solution.event_terminated & solution.backend_successful
    dissolved = event & mask[0]
    hemisphere = event & mask[1]
    status = jnp.where(
        hemisphere,
        int(BubbleDynamicsStatus.SUPPORT_EXIT),
        _terminal_status(solution, dissolved),
    )
    status = jnp.where(
        prepared.admissible, status, int(BubbleDynamicsStatus.INVALID_EQUILIBRIUM)
    ).astype(jnp.int32)
    terminal_scaled = solution.terminal_time
    valid = (scaled_times <= terminal_scaled) & prepared.admissible
    values = solution.evaluate(jnp.clip(scaled_times, 0.0, terminal_scaled))
    diameter = values[:, 0] * initial.footprint_diameter
    angle = values[:, 1] * initial.gas_contact_angle
    safe_diameter = jnp.where(valid, diameter, initial.footprint_diameter)
    safe_angle = jnp.where(valid, angle, initial.gas_contact_angle)
    curvature = safe_diameter / (2.0 * jnp.sin(safe_angle))
    amount = jax.vmap(plan.gas_amount)(safe_diameter, safe_angle)
    terminal = solution.terminal_state
    terminal_state = SurfaceBubbleState(
        terminal[0] * initial.footprint_diameter,
        terminal[1] * initial.gas_contact_angle,
        terminal[2]
        * plan.gas_amount(initial.footprint_diameter, initial.gas_contact_angle),
    )
    initial_amount = plan.gas_amount(initial.footprint_diameter, initial.gas_contact_angle)
    amount_residual = (
        plan.gas_amount(terminal_state.footprint_diameter, terminal_state.gas_contact_angle)
        - initial_amount
        - terminal_state.amount_change
    ) / initial_amount
    rows_curvature = jnp.concatenate(
        (
            curvature,
            (terminal_state.footprint_diameter / (2.0 * jnp.sin(terminal_state.gas_contact_angle)))[None],
        )
    )
    row_valid = jnp.concatenate((valid, jnp.asarray([True])))
    laplace = jax.vmap(plan.laplace_pressure)(rows_curvature) / properties.ambient_pressure
    knudsen = plan.validity.knudsen_number(
        properties.temperature,
        properties.ambient_pressure * (1.0 + laplace),
        rows_curvature,
    )
    tolman_ratio = plan.validity.tolman_ratio(rows_curvature)
    max_knudsen = None if knudsen is None else _extreme(knudsen, row_valid)
    max_tolman = None if tolman_ratio is None else _extreme(tolman_ratio, row_valid)
    continuum = jnp.asarray(True)
    if max_knudsen is not None:
        continuum = continuum & (max_knudsen <= plan.validity.knudsen_limit)
    if max_tolman is not None:
        continuum = continuum & (max_tolman <= plan.validity.tolman_ratio_limit)
    terminal_time = terminal_scaled * prepared.diffusion_time
    evidence = SurfaceBubbleEvidence(
        solution.backend_successful,
        jnp.asarray(solution.stats["num_accepted_steps"], dtype=jnp.int32),
        jnp.asarray(solution.stats["num_rejected_steps"], dtype=jnp.int32),
        dissolved,
        jnp.where(dissolved, terminal_time, jnp.inf),
        amount_residual,
        plan.concentration_excess(initial.footprint_diameter, initial.gas_contact_angle),
        plan.concentration_excess(
            terminal_state.footprint_diameter, terminal_state.gas_contact_angle
        ),
        plan.equilibrium(initial.footprint_diameter),
        _extreme(laplace, row_valid),
        max_knudsen,
        max_tolman,
        continuum,
    )
    nan = jnp.nan
    return SurfaceBubbleResult(
        plan.save_times,
        jnp.where(valid, diameter, nan),
        jnp.where(valid, angle, nan),
        jnp.where(valid, jnp.pi - angle, nan),
        jnp.where(valid, curvature, nan),
        jnp.where(valid, amount, nan),
        valid,
        terminal_state,
        terminal_time,
        status,
        evidence,
        plan_id=plan.plan_id,
    )


__all__ = [
    "DissolutionEvidence",
    "EpsteinPlessetPlan",
    "EpsteinPlessetRoute",
    "GasDissolutionResult",
    "GasDissolutionState",
    "GasSolutionProperties",
    "PinnedSurfaceBubblePlan",
    "PreparedEpsteinPlesset",
    "PreparedSurfaceBubble",
    "SurfaceBubbleContact",
    "SurfaceBubbleEquilibrium",
    "SurfaceBubbleEvidence",
    "SurfaceBubbleResult",
    "SurfaceBubbleState",
    "popov_flux_factor",
    "quasi_static_dissolution_time",
    "solve_epstein_plesset",
    "solve_surface_bubble",
]
