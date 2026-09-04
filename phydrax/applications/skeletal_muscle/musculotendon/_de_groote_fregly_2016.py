#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""De Groote--Fregly 2016 compliant-tendon muscle formulations.

Equations are independently derived from De Groote, Kinney, Rao & Fregly,
Annals of Biomedical Engineering 44 (2016), Eqs. 1--7 and Online Supplement
Eqs. S1--S31, DOI 10.1007/s10439-016-1591-9. The explicit runtime implements
Supplement formulation 1 (normalized tendon force is a state). The separately
named implicit runtime solves formulation 3's algebraic path constraint with
Phydrax's implicit-root sensitivity owner; the paper itself treated its scaled
force rate as an optimization control.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....nonlinear import implicit_root_result, NonlinearSystemProblem


_SOURCE_DOI = "10.1007/s10439-016-1591-9"


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _positive_vector(value: ArrayLike, name: str, /, *, dtype: Any | None = None) -> Array:
    array = jnp.asarray(value, dtype=dtype)
    if dtype is None and not jnp.issubdtype(array.dtype, jnp.floating):
        array = array.astype(float)
    host = np.asarray(array)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty rank-1 array.")
    if not np.all(np.isfinite(host)) or np.any(host <= 0.0):
        raise ValueError(f"{name} must be positive and finite.")
    return array


def _positive_scalar(value: ArrayLike, name: str, dtype: Any, /) -> Array:
    array = jnp.asarray(value, dtype=dtype)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    host = float(np.asarray(array))
    if not isfinite(host) or host <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return array


def _finite_scalar(value: ArrayLike, name: str, dtype: Any, /) -> Array:
    array = jnp.asarray(value, dtype=dtype)
    if array.shape != () or not isfinite(float(np.asarray(array))):
        raise ValueError(f"{name} must be finite and scalar.")
    return array


class DeGrooteFregly2016Parameters(StrictModule):
    """Trainable SI scaling and dimensionless source coefficients.

    The five muscle-specific vectors are, respectively, N, m, m, rad, and m/s.
    Every numeric field is a dynamic JAX leaf. Curve defaults reproduce Table 1
    of the online supplement; time constants and smoothing reproduce manuscript
    Eqs. 1--2. ``implicit_force_rate_scale_per_s=10`` is manuscript Eq. 14.
    """

    maximum_isometric_force_N: Array
    optimal_fiber_length_m: Array
    tendon_slack_length_m: Array
    pennation_angle_at_optimum_rad: Array
    maximum_fiber_velocity_m_per_s: Array
    activation_time_constant_s: Array
    deactivation_time_constant_s: Array
    activation_smoothing: Array
    tendon_stiffness: Array
    tendon_c1: Array
    tendon_c2: Array
    tendon_c3: Array
    active_force_length_b1: Array
    active_force_length_b2: Array
    active_force_length_b3: Array
    active_force_length_b4: Array
    passive_stiffness: Array
    passive_strain: Array
    force_velocity_d1: Array
    force_velocity_d2: Array
    force_velocity_d3: Array
    force_velocity_d4: Array
    implicit_force_rate_scale_per_s: Array

    def __init__(
        self,
        maximum_isometric_force_N: ArrayLike,
        optimal_fiber_length_m: ArrayLike,
        tendon_slack_length_m: ArrayLike,
        pennation_angle_at_optimum_rad: ArrayLike,
        maximum_fiber_velocity_m_per_s: ArrayLike,
        /,
        *,
        activation_time_constant_s: ArrayLike = 0.015,
        deactivation_time_constant_s: ArrayLike = 0.060,
        activation_smoothing: ArrayLike = 0.1,
        tendon_stiffness: ArrayLike = 35.0,
        tendon_c1: ArrayLike = 0.200,
        tendon_c2: ArrayLike = 0.995,
        tendon_c3: ArrayLike = 0.250,
        active_force_length_b1: ArrayLike = (0.815, 0.433, 0.100),
        active_force_length_b2: ArrayLike = (1.055, 0.717, 1.000),
        active_force_length_b3: ArrayLike = (0.162, -0.030, 0.354),
        active_force_length_b4: ArrayLike = (0.063, 0.200, 0.000),
        passive_stiffness: ArrayLike = 4.0,
        passive_strain: ArrayLike = 0.6,
        force_velocity_d1: ArrayLike = -0.318,
        force_velocity_d2: ArrayLike = -8.149,
        force_velocity_d3: ArrayLike = -0.374,
        force_velocity_d4: ArrayLike = 0.886,
        implicit_force_rate_scale_per_s: ArrayLike = 10.0,
    ):
        force = _positive_vector(maximum_isometric_force_N, "maximum_isometric_force_N")
        dtype = force.dtype
        optimal = _positive_vector(
            optimal_fiber_length_m, "optimal_fiber_length_m", dtype=dtype
        )
        slack = _positive_vector(
            tendon_slack_length_m, "tendon_slack_length_m", dtype=dtype
        )
        pennation = jnp.asarray(pennation_angle_at_optimum_rad, dtype=dtype)
        velocity = _positive_vector(
            maximum_fiber_velocity_m_per_s,
            "maximum_fiber_velocity_m_per_s",
            dtype=dtype,
        )
        expected = force.shape
        if optimal.shape != expected or slack.shape != expected or velocity.shape != expected:
            raise ValueError("Muscle-specific parameter vectors must have equal shape.")
        if pennation.shape != expected:
            raise ValueError("pennation_angle_at_optimum_rad must match muscle capacity.")
        pennation_host = np.asarray(pennation)
        if (
            not np.all(np.isfinite(pennation_host))
            or np.any(pennation_host < 0.0)
            or np.any(pennation_host >= 0.5 * np.pi)
        ):
            raise ValueError("Pennation angles must be finite and lie in [0, pi/2).")

        curve_arrays = tuple(
            jnp.asarray(value, dtype=dtype)
            for value in (
                active_force_length_b1,
                active_force_length_b2,
                active_force_length_b3,
                active_force_length_b4,
            )
        )
        if any(array.shape != (3,) for array in curve_arrays):
            raise ValueError("Active force-length coefficient arrays must have shape (3,).")
        if not all(np.all(np.isfinite(np.asarray(array))) for array in curve_arrays):
            raise ValueError("Active force-length coefficients must be finite.")
        length_grid = np.linspace(0.4, 1.6, 257)
        b3_host = np.asarray(curve_arrays[2])
        b4_host = np.asarray(curve_arrays[3])
        if np.any(np.abs(b3_host[None, :] + length_grid[:, None] * b4_host) < 1.0e-8):
            raise ValueError("Active force-length Gaussian widths vanish on [0.4, 1.6].")

        self.maximum_isometric_force_N = force
        self.optimal_fiber_length_m = optimal
        self.tendon_slack_length_m = slack
        self.pennation_angle_at_optimum_rad = pennation
        self.maximum_fiber_velocity_m_per_s = velocity
        self.activation_time_constant_s = _positive_scalar(
            activation_time_constant_s, "activation_time_constant_s", dtype
        )
        self.deactivation_time_constant_s = _positive_scalar(
            deactivation_time_constant_s, "deactivation_time_constant_s", dtype
        )
        self.activation_smoothing = _positive_scalar(
            activation_smoothing, "activation_smoothing", dtype
        )
        self.tendon_stiffness = _positive_scalar(
            tendon_stiffness, "tendon_stiffness", dtype
        )
        self.tendon_c1 = _positive_scalar(tendon_c1, "tendon_c1", dtype)
        self.tendon_c2 = _finite_scalar(tendon_c2, "tendon_c2", dtype)
        self.tendon_c3 = _positive_scalar(tendon_c3, "tendon_c3", dtype)
        self.active_force_length_b1 = curve_arrays[0]
        self.active_force_length_b2 = curve_arrays[1]
        self.active_force_length_b3 = curve_arrays[2]
        self.active_force_length_b4 = curve_arrays[3]
        self.passive_stiffness = _positive_scalar(
            passive_stiffness, "passive_stiffness", dtype
        )
        self.passive_strain = _positive_scalar(
            passive_strain, "passive_strain", dtype
        )
        self.force_velocity_d1 = _finite_scalar(
            force_velocity_d1, "force_velocity_d1", dtype
        )
        self.force_velocity_d2 = _finite_scalar(
            force_velocity_d2, "force_velocity_d2", dtype
        )
        self.force_velocity_d3 = _finite_scalar(
            force_velocity_d3, "force_velocity_d3", dtype
        )
        self.force_velocity_d4 = _finite_scalar(
            force_velocity_d4, "force_velocity_d4", dtype
        )
        if float(np.asarray(self.force_velocity_d1)) == 0.0 or float(
            np.asarray(self.force_velocity_d2)
        ) == 0.0:
            raise ValueError("force_velocity_d1 and force_velocity_d2 must be nonzero.")
        self.implicit_force_rate_scale_per_s = _positive_scalar(
            implicit_force_rate_scale_per_s,
            "implicit_force_rate_scale_per_s",
            dtype,
        )

    @property
    def muscle_capacity(self) -> int:
        return int(self.maximum_isometric_force_N.shape[0])


class DeGrooteFregly2016State(StrictModule):
    """Activation and normalized tendon-force state (dimensionless)."""

    activation: Array
    normalized_tendon_force: Array

    def __init__(
        self, activation: ArrayLike, normalized_tendon_force: ArrayLike, /
    ):
        activation_ = jnp.asarray(activation)
        tendon_force = jnp.asarray(normalized_tendon_force, dtype=activation_.dtype)
        if activation_.ndim != 1 or activation_.shape != tendon_force.shape:
            raise ValueError("State fields must have one common non-empty rank-1 shape.")
        if activation_.size == 0 or not jnp.issubdtype(activation_.dtype, jnp.floating):
            raise TypeError("State fields must use a non-empty floating-point vector.")
        self.activation = activation_
        self.normalized_tendon_force = tendon_force

    @classmethod
    def resting(
        cls,
        muscle_capacity: int,
        /,
        *,
        dtype: Any = jnp.float64,
    ) -> DeGrooteFregly2016State:
        capacity = int(muscle_capacity)
        if capacity <= 0:
            raise ValueError("muscle_capacity must be positive.")
        return cls(
            jnp.full((capacity,), 0.01, dtype=dtype),
            jnp.zeros((capacity,), dtype=dtype),
        )


class DeGrooteFregly2016Rates(StrictModule):
    """Source-equation state rates: activation 1/s and normalized force 1/s."""

    activation_per_s: Array
    normalized_tendon_force_per_s: Array


class DeGrooteFregly2016Evidence(StrictModule):
    """Per-muscle equation, geometry, energy, and admissibility evidence."""

    tendon_constitutive_residual_normalized: Array
    force_velocity_inverse_residual_normalized: Array
    tendon_rate_residual_per_s: Array
    force_equilibrium_residual_normalized: Array
    length_closure_residual_m: Array
    pennation_closure_residual_m: Array
    tendon_energy_J: Array
    passive_fiber_energy_J: Array
    tendon_energy_rate_W: Array
    tendon_power_W: Array
    tendon_energy_residual_W: Array
    passive_energy_rate_W: Array
    passive_power_W: Array
    passive_energy_residual_W: Array
    route_power_W: Array
    series_plus_fiber_power_W: Array
    power_balance_residual_W: Array
    finite: Array
    state_admissible: Array
    geometry_admissible: Array
    residuals_satisfied: Array
    energy_consistent: Array
    successful: Array
    source_doi: str = eqx.field(static=True, default=_SOURCE_DOI)


class DeGrooteFregly2016Evaluation(StrictModule):
    """Explicit formulation-1 constitutive evaluation in SI and normalized units."""

    activation: Array
    normalized_tendon_force: Array
    tendon_force_N: Array
    normalized_tendon_length: Array
    tendon_length_m: Array
    tendon_velocity_m_per_s: Array
    normalized_fiber_length: Array
    fiber_length_m: Array
    normalized_fiber_velocity: Array
    fiber_velocity_m_per_s: Array
    pennation_angle_rad: Array
    cosine_pennation: Array
    active_force_length: Array
    passive_force_length: Array
    force_velocity: Array
    normalized_fiber_force: Array
    fiber_force_N: Array
    rates: DeGrooteFregly2016Rates
    evidence: DeGrooteFregly2016Evidence
    force_owner: str = eqx.field(static=True, default="de-groote-fregly-2016")
    force_sign: str = eqx.field(static=True, default="positive-is-tensile")

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class DeGrooteFregly2016StepEvidence(StrictModule):
    """Whole-state explicit Euler transaction evidence."""

    source_evidence: DeGrooteFregly2016Evidence
    candidate_evidence: DeGrooteFregly2016Evidence
    excitation_admissible: Array
    time_step_admissible: Array
    candidate_state_admissible: Array
    finite: Array
    successful: Array
    integration_scheme: str = eqx.field(static=True, default="forward-euler")


class DeGrooteFregly2016Candidate(StrictModule):
    """Uncommitted explicit formulation-1 step retaining its rollback state."""

    previous_state: DeGrooteFregly2016State
    candidate_state: DeGrooteFregly2016State
    evaluation: DeGrooteFregly2016Evaluation
    evidence: DeGrooteFregly2016StepEvidence
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.successful


def de_groote_fregly_2016_tendon_force_length(
    parameters: DeGrooteFregly2016Parameters,
    normalized_tendon_length: ArrayLike,
    /,
) -> Array:
    """Supplement Eq. S1, returning normalized tendon force."""

    length = jnp.asarray(
        normalized_tendon_length, dtype=parameters.maximum_isometric_force_N.dtype
    )
    return (
        parameters.tendon_c1
        * jnp.exp(parameters.tendon_stiffness * (length - parameters.tendon_c2))
        - parameters.tendon_c3
    )


def de_groote_fregly_2016_inverse_tendon_force_length(
    parameters: DeGrooteFregly2016Parameters,
    normalized_tendon_force: ArrayLike,
    /,
) -> Array:
    """Supplement Eq. S7, inverse normalized tendon force-length curve."""

    force = jnp.asarray(
        normalized_tendon_force, dtype=parameters.maximum_isometric_force_N.dtype
    )
    return (
        jnp.log((force + parameters.tendon_c3) / parameters.tendon_c1)
        / parameters.tendon_stiffness
        + parameters.tendon_c2
    )


def de_groote_fregly_2016_active_force_length(
    parameters: DeGrooteFregly2016Parameters,
    normalized_fiber_length: ArrayLike,
    /,
) -> Array:
    """Supplement Eq. S2, the sum of three normalized Gaussian terms."""

    length = jnp.asarray(
        normalized_fiber_length, dtype=parameters.maximum_isometric_force_N.dtype
    )
    width = (
        parameters.active_force_length_b3
        + length[..., None] * parameters.active_force_length_b4
    )
    normalized = (length[..., None] - parameters.active_force_length_b2) / width
    return jnp.sum(
        parameters.active_force_length_b1 * jnp.exp(-0.5 * normalized * normalized),
        axis=-1,
    )


def de_groote_fregly_2016_passive_force_length(
    parameters: DeGrooteFregly2016Parameters,
    normalized_fiber_length: ArrayLike,
    /,
) -> Array:
    """Supplement Eq. S3, without clipping its smooth slack-region extension."""

    length = jnp.asarray(
        normalized_fiber_length, dtype=parameters.maximum_isometric_force_N.dtype
    )
    numerator = jnp.exp(
        parameters.passive_stiffness
        * (length - 1.0)
        / parameters.passive_strain
    ) - 1.0
    return numerator / jnp.expm1(parameters.passive_stiffness)


def de_groote_fregly_2016_force_velocity(
    parameters: DeGrooteFregly2016Parameters,
    normalized_fiber_velocity: ArrayLike,
    /,
) -> Array:
    """Supplement Eq. S4; positive velocity is fiber lengthening."""

    velocity = jnp.asarray(
        normalized_fiber_velocity, dtype=parameters.maximum_isometric_force_N.dtype
    )
    return (
        parameters.force_velocity_d1
        * jnp.arcsinh(
            parameters.force_velocity_d2 * velocity + parameters.force_velocity_d3
        )
        + parameters.force_velocity_d4
    )


def de_groote_fregly_2016_inverse_force_velocity(
    parameters: DeGrooteFregly2016Parameters,
    normalized_force_velocity: ArrayLike,
    /,
) -> Array:
    """Supplement Eq. S13, the analytic inverse of Eq. S4."""

    force = jnp.asarray(
        normalized_force_velocity, dtype=parameters.maximum_isometric_force_N.dtype
    )
    return (
        jnp.sinh(
            (force - parameters.force_velocity_d4)
            / parameters.force_velocity_d1
        )
        - parameters.force_velocity_d3
    ) / parameters.force_velocity_d2


def _activation_rate(
    parameters: DeGrooteFregly2016Parameters,
    activation: Array,
    independent_excitation: Array,
    /,
) -> Array:
    transition = 0.5 * jnp.tanh(
        parameters.activation_smoothing * (independent_excitation - activation)
    )
    scale = 0.5 + 1.5 * activation
    inverse_time = (
        (transition + 0.5) / (parameters.activation_time_constant_s * scale)
        + scale * (-transition + 0.5) / parameters.deactivation_time_constant_s
    )
    return inverse_time * (independent_excitation - activation)


def _tendon_energy(
    parameters: DeGrooteFregly2016Parameters,
    normalized_tendon_length: Array,
    /,
) -> Array:
    exponent = parameters.tendon_stiffness * (
        normalized_tendon_length - parameters.tendon_c2
    )
    reference_exponent = parameters.tendon_stiffness * (1.0 - parameters.tendon_c2)
    normalized_integral = (
        parameters.tendon_c1
        / parameters.tendon_stiffness
        * (jnp.exp(exponent) - jnp.exp(reference_exponent))
        - parameters.tendon_c3 * (normalized_tendon_length - 1.0)
    )
    return (
        parameters.maximum_isometric_force_N
        * parameters.tendon_slack_length_m
        * normalized_integral
    )


def _passive_energy(
    parameters: DeGrooteFregly2016Parameters,
    normalized_fiber_length: Array,
    /,
) -> Array:
    exponent = (
        parameters.passive_stiffness
        * (normalized_fiber_length - 1.0)
        / parameters.passive_strain
    )
    normalized_integral = (
        parameters.passive_strain
        / parameters.passive_stiffness
        * jnp.expm1(exponent)
        - (normalized_fiber_length - 1.0)
    ) / jnp.expm1(parameters.passive_stiffness)
    return (
        parameters.maximum_isometric_force_N
        * parameters.optimal_fiber_length_m
        * normalized_integral
    )


class DeGrooteFregly2016Plan(StrictModule):
    """Explicit supplement formulation-1 plan with fixed muscle capacity/mask."""

    parameters: DeGrooteFregly2016Parameters
    muscle_names: tuple[str, ...] = eqx.field(static=True)
    muscle_mask: tuple[bool, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameters: DeGrooteFregly2016Parameters,
        muscle_names: Sequence[str],
        /,
        *,
        muscle_mask: Sequence[bool] | None = None,
        model_id: str | None = None,
    ):
        if not isinstance(parameters, DeGrooteFregly2016Parameters):
            raise TypeError("parameters must be DeGrooteFregly2016Parameters.")
        names = tuple(_identifier(name, "muscle name") for name in muscle_names)
        if len(names) != parameters.muscle_capacity or len(set(names)) != len(names):
            raise ValueError("muscle_names must uniquely fill the parameter capacity.")
        mask = (
            (True,) * len(names)
            if muscle_mask is None
            else tuple(bool(value) for value in muscle_mask)
        )
        if len(mask) != len(names) or not any(mask):
            raise ValueError("muscle_mask must match capacity and enable a muscle.")
        generated = canonical_fingerprint(
            {
                "kind": "de-groote-fregly-2016-explicit-tendon-force",
                "source_doi": _SOURCE_DOI,
                "muscle_names": list(names),
                "muscle_mask": list(mask),
            }
        )
        self.parameters = parameters
        self.muscle_names = names
        self.muscle_mask = mask
        self.model_id = generated if model_id is None else _identifier(model_id, "model_id")

    def prepare(
        self, state: DeGrooteFregly2016State | None = None, /
    ) -> PreparedDeGrooteFregly2016Musculotendon:
        initial = (
            DeGrooteFregly2016State.resting(
                self.parameters.muscle_capacity,
                dtype=self.parameters.maximum_isometric_force_N.dtype,
            )
            if state is None
            else state
        )
        return PreparedDeGrooteFregly2016Musculotendon(self, initial)


class PreparedDeGrooteFregly2016Musculotendon(StrictModule):
    """Prepared explicit compliant-tendon runtime and atomic commit owner."""

    plan: DeGrooteFregly2016Plan
    reference_state: DeGrooteFregly2016State
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: DeGrooteFregly2016Plan,
        reference_state: DeGrooteFregly2016State,
        /,
    ):
        if not isinstance(plan, DeGrooteFregly2016Plan):
            raise TypeError("plan must be DeGrooteFregly2016Plan.")
        self.plan = plan
        self._validate_state(reference_state)
        self.reference_state = reference_state
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-de-groote-fregly-2016-explicit",
                "model": plan.model_id,
                "capacity": plan.parameters.muscle_capacity,
                "dtype": np.dtype(reference_state.activation.dtype).str,
            }
        )

    @property
    def parameters(self) -> DeGrooteFregly2016Parameters:
        return self.plan.parameters

    @property
    def muscle_capacity(self) -> int:
        return self.parameters.muscle_capacity

    def _validate_state(self, state: DeGrooteFregly2016State, /) -> None:
        if not isinstance(state, DeGrooteFregly2016State):
            raise TypeError("state must be DeGrooteFregly2016State.")
        expected = (self.plan.parameters.muscle_capacity,)
        if state.activation.shape != expected:
            raise ValueError(f"State must have fixed shape {expected}.")
        if state.activation.dtype != self.plan.parameters.maximum_isometric_force_N.dtype:
            raise TypeError("State dtype must match the prepared parameter dtype.")

    def _input(self, value: ArrayLike, name: str, /) -> Array:
        array = jnp.asarray(value, dtype=self.parameters.maximum_isometric_force_N.dtype)
        expected = (self.muscle_capacity,)
        if array.shape != expected:
            raise ValueError(f"{name} must have fixed shape {expected}.")
        return array

    def evaluate(
        self,
        state: DeGrooteFregly2016State,
        independent_excitation: ArrayLike,
        musculotendon_length_m: ArrayLike,
        musculotendon_velocity_m_per_s: ArrayLike,
        /,
    ) -> DeGrooteFregly2016Evaluation:
        """Evaluate manuscript Eqs. 1--7 and supplement formulation 1."""

        self._validate_state(state)
        parameters = self.parameters
        activation = state.activation
        tendon_force_normalized = state.normalized_tendon_force
        excitation_ = self._input(
            independent_excitation, "independent_excitation"
        )
        length_mt = self._input(musculotendon_length_m, "musculotendon_length_m")
        velocity_mt = self._input(
            musculotendon_velocity_m_per_s, "musculotendon_velocity_m_per_s"
        )
        mask = jnp.asarray(self.plan.muscle_mask, dtype=bool)
        tiny = jnp.finfo(activation.dtype).tiny
        force_argument_valid = tendon_force_normalized + parameters.tendon_c3 > tiny
        safe_force = jnp.where(
            force_argument_valid,
            tendon_force_normalized,
            -parameters.tendon_c3 + tiny,
        )
        normalized_tendon_length = de_groote_fregly_2016_inverse_tendon_force_length(
            parameters, safe_force
        )
        tendon_length = parameters.tendon_slack_length_m * normalized_tendon_length
        fixed_height = (
            parameters.optimal_fiber_length_m
            * jnp.sin(parameters.pennation_angle_at_optimum_rad)
        )
        along_tendon = length_mt - tendon_length
        fiber_length = jnp.sqrt(fixed_height * fixed_height + along_tendon * along_tendon)
        safe_fiber_length = jnp.maximum(fiber_length, tiny)
        cosine = along_tendon / safe_fiber_length
        sine = fixed_height / safe_fiber_length
        pennation = jnp.arctan2(sine, cosine)
        normalized_fiber_length = fiber_length / parameters.optimal_fiber_length_m
        active_length = de_groote_fregly_2016_active_force_length(
            parameters, normalized_fiber_length
        )
        passive_length = de_groote_fregly_2016_passive_force_length(
            parameters, normalized_fiber_length
        )
        active_denominator = activation * active_length
        active_valid = active_denominator > tiny
        safe_active = jnp.where(active_valid, active_denominator, 1.0)
        safe_cosine = jnp.where(cosine > tiny, cosine, 1.0)
        normalized_fiber_force_from_tendon = tendon_force_normalized / safe_cosine
        force_velocity_required = (
            normalized_fiber_force_from_tendon - passive_length
        ) / safe_active
        normalized_fiber_velocity = de_groote_fregly_2016_inverse_force_velocity(
            parameters, force_velocity_required
        )
        force_velocity = de_groote_fregly_2016_force_velocity(
            parameters, normalized_fiber_velocity
        )
        fiber_velocity = (
            normalized_fiber_velocity * parameters.maximum_fiber_velocity_m_per_s
        )
        tendon_velocity = velocity_mt - fiber_velocity / safe_cosine
        tendon_curve_slope = (
            parameters.tendon_c1
            * parameters.tendon_stiffness
            * jnp.exp(
                parameters.tendon_stiffness
                * (normalized_tendon_length - parameters.tendon_c2)
            )
        )
        normalized_tendon_force_rate = (
            tendon_curve_slope
            * tendon_velocity
            / parameters.tendon_slack_length_m
        )
        activation_rate = _activation_rate(parameters, activation, excitation_)
        normalized_fiber_force = (
            activation * active_length * force_velocity + passive_length
        )
        tendon_force = parameters.maximum_isometric_force_N * tendon_force_normalized
        fiber_force = parameters.maximum_isometric_force_N * normalized_fiber_force
        passive_force = parameters.maximum_isometric_force_N * passive_length

        tendon_curve_force = de_groote_fregly_2016_tendon_force_length(
            parameters, normalized_tendon_length
        )
        tendon_constitutive_residual = (
            tendon_force_normalized - tendon_curve_force
        )
        force_velocity_inverse_residual = (
            force_velocity_required - force_velocity
        )
        tendon_rate_residual = normalized_tendon_force_rate - (
            tendon_curve_slope
            * tendon_velocity
            / parameters.tendon_slack_length_m
        )
        force_residual = tendon_force_normalized - normalized_fiber_force * cosine
        length_residual = length_mt - (tendon_length + fiber_length * cosine)
        pennation_residual = fiber_length * sine - fixed_height
        tendon_energy = _tendon_energy(parameters, normalized_tendon_length)
        passive_energy = _passive_energy(parameters, normalized_fiber_length)
        tendon_energy_rate = tendon_force * tendon_velocity
        tendon_power = tendon_force * tendon_velocity
        tendon_energy_residual = tendon_energy_rate - tendon_power
        passive_energy_rate = passive_force * fiber_velocity
        passive_power = passive_force * fiber_velocity
        passive_energy_residual = passive_energy_rate - passive_power
        route_power = tendon_force * velocity_mt
        series_plus_fiber_power = tendon_power + fiber_force * fiber_velocity
        power_residual = route_power - series_plus_fiber_power

        state_admissible = (
            (activation >= 0.01)
            & (activation <= 1.0)
            & (tendon_force_normalized >= 0.0)
            & (tendon_force_normalized <= 3.0)
            & (excitation_ >= 0.0)
            & (excitation_ <= 1.0)
        )
        geometry_admissible = (
            force_argument_valid
            & active_valid
            & (along_tendon > 0.0)
            & (cosine > 0.0)
            & (normalized_fiber_length >= 0.4)
            & (normalized_fiber_length <= 1.6)
        )
        arrays = (
            activation,
            tendon_force_normalized,
            tendon_force,
            normalized_tendon_length,
            tendon_length,
            tendon_velocity,
            normalized_fiber_length,
            fiber_length,
            normalized_fiber_velocity,
            fiber_velocity,
            pennation,
            cosine,
            active_length,
            passive_length,
            force_velocity,
            normalized_fiber_force,
            fiber_force,
            activation_rate,
            normalized_tendon_force_rate,
            tendon_constitutive_residual,
            force_velocity_inverse_residual,
            tendon_rate_residual,
            force_residual,
            length_residual,
            pennation_residual,
            tendon_energy,
            passive_energy,
            route_power,
            series_plus_fiber_power,
            power_residual,
        )
        finite = jnp.all(
            jnp.stack(tuple(jnp.isfinite(value) for value in arrays), axis=0),
            axis=0,
        )
        eps = jnp.finfo(activation.dtype).eps
        force_scale = jnp.maximum(1.0, jnp.abs(tendon_force_normalized))
        rate_scale = jnp.maximum(
            1.0, jnp.abs(normalized_tendon_force_rate)
        )
        length_scale = jnp.maximum(1.0, jnp.abs(length_mt))
        power_scale = jnp.maximum(
            1.0, jnp.maximum(jnp.abs(route_power), jnp.abs(series_plus_fiber_power))
        )
        residuals_satisfied = (
            (
                jnp.abs(tendon_constitutive_residual)
                <= 128.0 * eps * force_scale
            )
            & (
                jnp.abs(force_velocity_inverse_residual)
                <= 128.0 * eps * force_scale
            )
            & (jnp.abs(tendon_rate_residual) <= 128.0 * eps * rate_scale)
            & (jnp.abs(force_residual) <= 128.0 * eps * force_scale)
            & (jnp.abs(length_residual) <= 128.0 * eps * length_scale)
            & (jnp.abs(pennation_residual) <= 128.0 * eps * length_scale)
        )
        energy_consistent = (
            (jnp.abs(tendon_energy_residual) <= 128.0 * eps * power_scale)
            & (jnp.abs(passive_energy_residual) <= 128.0 * eps * power_scale)
            & (jnp.abs(power_residual) <= 256.0 * eps * power_scale)
        )
        successful = (~mask) | (
            finite
            & state_admissible
            & geometry_admissible
            & residuals_satisfied
            & energy_consistent
        )
        evidence = DeGrooteFregly2016Evidence(
            tendon_constitutive_residual,
            force_velocity_inverse_residual,
            tendon_rate_residual,
            force_residual,
            length_residual,
            pennation_residual,
            tendon_energy,
            passive_energy,
            tendon_energy_rate,
            tendon_power,
            tendon_energy_residual,
            passive_energy_rate,
            passive_power,
            passive_energy_residual,
            route_power,
            series_plus_fiber_power,
            power_residual,
            finite,
            state_admissible,
            geometry_admissible,
            residuals_satisfied,
            energy_consistent,
            successful,
        )
        return DeGrooteFregly2016Evaluation(
            activation,
            tendon_force_normalized,
            jnp.where(mask, tendon_force, 0.0),
            normalized_tendon_length,
            tendon_length,
            tendon_velocity,
            normalized_fiber_length,
            fiber_length,
            normalized_fiber_velocity,
            fiber_velocity,
            pennation,
            cosine,
            active_length,
            passive_length,
            force_velocity,
            normalized_fiber_force,
            fiber_force,
            DeGrooteFregly2016Rates(activation_rate, normalized_tendon_force_rate),
            evidence,
        )

    def candidate(
        self,
        state: DeGrooteFregly2016State,
        independent_excitation: ArrayLike,
        musculotendon_length_m: ArrayLike,
        musculotendon_velocity_m_per_s: ArrayLike,
        time_step_s: ArrayLike,
        /,
    ) -> DeGrooteFregly2016Candidate:
        """Propose one explicit formulation-1 Euler step under constant inputs."""

        source = self.evaluate(
            state,
            independent_excitation,
            musculotendon_length_m,
            musculotendon_velocity_m_per_s,
        )
        dt = jnp.asarray(
            time_step_s, dtype=self.parameters.maximum_isometric_force_N.dtype
        )
        if dt.shape != ():
            raise ValueError("time_step_s must be scalar.")
        mask = jnp.asarray(self.plan.muscle_mask, dtype=bool)
        proposed_activation = state.activation + dt * source.rates.activation_per_s
        proposed_force = (
            state.normalized_tendon_force
            + dt * source.rates.normalized_tendon_force_per_s
        )
        candidate_state = DeGrooteFregly2016State(
            jnp.where(mask, proposed_activation, state.activation),
            jnp.where(mask, proposed_force, state.normalized_tendon_force),
        )
        length = self._input(musculotendon_length_m, "musculotendon_length_m")
        velocity = self._input(
            musculotendon_velocity_m_per_s, "musculotendon_velocity_m_per_s"
        )
        candidate_evaluation = self.evaluate(
            candidate_state,
            independent_excitation,
            length + dt * velocity,
            velocity,
        )
        excitation_ = self._input(
            independent_excitation, "independent_excitation"
        )
        excitation_admissible = jnp.all(
            (~mask)
            | (
                jnp.isfinite(excitation_)
                & (excitation_ >= 0.0)
                & (excitation_ <= 1.0)
            )
        )
        time_step_admissible = jnp.isfinite(dt) & (dt > 0.0)
        candidate_state_admissible = jnp.all(
            (~mask)
            | (
                (candidate_state.activation >= 0.01)
                & (candidate_state.activation <= 1.0)
                & (candidate_state.normalized_tendon_force >= 0.0)
                & (candidate_state.normalized_tendon_force <= 3.0)
            )
        )
        finite = (
            jnp.all(jnp.isfinite(candidate_state.activation))
            & jnp.all(jnp.isfinite(candidate_state.normalized_tendon_force))
            & jnp.all(source.evidence.finite | ~mask)
            & jnp.all(candidate_evaluation.evidence.finite | ~mask)
        )
        successful = (
            excitation_admissible
            & time_step_admissible
            & candidate_state_admissible
            & finite
            & jnp.all(source.successful)
            & jnp.all(candidate_evaluation.successful)
        )
        evidence = DeGrooteFregly2016StepEvidence(
            source.evidence,
            candidate_evaluation.evidence,
            excitation_admissible,
            time_step_admissible,
            candidate_state_admissible,
            finite,
            successful,
        )
        return DeGrooteFregly2016Candidate(
            state,
            candidate_state,
            candidate_evaluation,
            evidence,
            self.prepared_id,
        )

    def commit(
        self, candidate: DeGrooteFregly2016Candidate, /
    ) -> DeGrooteFregly2016State:
        """Atomically accept the full candidate or return its untouched source."""

        if not isinstance(candidate, DeGrooteFregly2016Candidate):
            raise TypeError("candidate must be DeGrooteFregly2016Candidate.")
        if candidate.prepared_id != self.prepared_id:
            raise ValueError("Candidate belongs to another prepared musculotendon model.")
        return jax.tree.map(
            lambda proposed, previous: jnp.where(
                candidate.evidence.successful, proposed, previous
            ),
            candidate.candidate_state,
            candidate.previous_state,
        )


class DeGrooteFregly2016ImplicitTendonForcePlan(StrictModule):
    """Separate formulation-3 plan solved through Phydrax implicit roots."""

    parameters: DeGrooteFregly2016Parameters
    muscle_names: tuple[str, ...] = eqx.field(static=True)
    muscle_mask: tuple[bool, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameters: DeGrooteFregly2016Parameters,
        muscle_names: Sequence[str],
        /,
        *,
        muscle_mask: Sequence[bool] | None = None,
        model_id: str | None = None,
    ):
        explicit = DeGrooteFregly2016Plan(
            parameters,
            muscle_names,
            muscle_mask=muscle_mask,
        )
        generated = canonical_fingerprint(
            {
                "kind": "de-groote-fregly-2016-implicit-tendon-force",
                "source_doi": _SOURCE_DOI,
                "muscle_names": list(explicit.muscle_names),
                "muscle_mask": list(explicit.muscle_mask),
                "root_owner": "phydrax.nonlinear.implicit_root_result",
            }
        )
        self.parameters = parameters
        self.muscle_names = explicit.muscle_names
        self.muscle_mask = explicit.muscle_mask
        self.model_id = generated if model_id is None else _identifier(model_id, "model_id")

    def prepare(
        self, state: DeGrooteFregly2016State | None = None, /
    ) -> PreparedDeGrooteFregly2016ImplicitTendonForce:
        initial = (
            DeGrooteFregly2016State.resting(
                self.parameters.muscle_capacity,
                dtype=self.parameters.maximum_isometric_force_N.dtype,
            )
            if state is None
            else state
        )
        return PreparedDeGrooteFregly2016ImplicitTendonForce(self, initial)


class DeGrooteFregly2016ImplicitEvidence(StrictModule):
    """Formulation-3 algebraic-root, constitutive, and sensitivity evidence."""

    constitutive: DeGrooteFregly2016StepEvidence
    scaled_force_rate_control: Array
    algebraic_residual: Array
    nonlinear_status: Array
    finite: Array
    residual_satisfied: Array
    successful: Array
    sensitivity_owner: str = eqx.field(
        static=True, default="phydrax.nonlinear.implicit_root_result"
    )
    source_equation: str = eqx.field(static=True, default="supplement-S24--S28")


class DeGrooteFregly2016ImplicitCandidate(StrictModule):
    """Uncommitted formulation-3 step with full-state rollback."""

    previous_state: DeGrooteFregly2016State
    candidate_state: DeGrooteFregly2016State
    evaluation: DeGrooteFregly2016Evaluation
    evidence: DeGrooteFregly2016ImplicitEvidence
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class PreparedDeGrooteFregly2016ImplicitTendonForce(StrictModule):
    """Prepared formulation-3 algebraic root with implicit JVP/VJP ownership."""

    plan: DeGrooteFregly2016ImplicitTendonForcePlan
    constitutive: PreparedDeGrooteFregly2016Musculotendon
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: DeGrooteFregly2016ImplicitTendonForcePlan,
        reference_state: DeGrooteFregly2016State,
        /,
    ):
        if not isinstance(plan, DeGrooteFregly2016ImplicitTendonForcePlan):
            raise TypeError("plan must be DeGrooteFregly2016ImplicitTendonForcePlan.")
        explicit_plan = DeGrooteFregly2016Plan(
            plan.parameters,
            plan.muscle_names,
            muscle_mask=plan.muscle_mask,
        )
        self.plan = plan
        self.constitutive = explicit_plan.prepare(reference_state)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-de-groote-fregly-2016-implicit",
                "model": plan.model_id,
                "capacity": plan.parameters.muscle_capacity,
                "dtype": np.dtype(reference_state.activation.dtype).str,
            }
        )

    def _algebraic_residual(
        self,
        scaled_force_rate: Array,
        state: DeGrooteFregly2016State,
        musculotendon_length_m: Array,
        musculotendon_velocity_m_per_s: Array,
        /,
    ) -> Array:
        parameters = self.plan.parameters
        tendon_force = state.normalized_tendon_force
        normalized_tendon_length = de_groote_fregly_2016_inverse_tendon_force_length(
            parameters, tendon_force
        )
        tendon_length = parameters.tendon_slack_length_m * normalized_tendon_length
        fixed_height = (
            parameters.optimal_fiber_length_m
            * jnp.sin(parameters.pennation_angle_at_optimum_rad)
        )
        along = musculotendon_length_m - tendon_length
        fiber_length = jnp.sqrt(fixed_height * fixed_height + along * along)
        cosine = along / fiber_length
        normalized_fiber_length = fiber_length / parameters.optimal_fiber_length_m
        tendon_curve_slope = (
            parameters.tendon_c1
            * parameters.tendon_stiffness
            * jnp.exp(
                parameters.tendon_stiffness
                * (normalized_tendon_length - parameters.tendon_c2)
            )
        )
        normalized_force_rate = (
            parameters.implicit_force_rate_scale_per_s * scaled_force_rate
        )
        tendon_velocity = (
            parameters.tendon_slack_length_m
            * normalized_force_rate
            / tendon_curve_slope
        )
        fiber_velocity = (musculotendon_velocity_m_per_s - tendon_velocity) * cosine
        normalized_fiber_velocity = (
            fiber_velocity / parameters.maximum_fiber_velocity_m_per_s
        )
        active_length = de_groote_fregly_2016_active_force_length(
            parameters, normalized_fiber_length
        )
        passive_length = de_groote_fregly_2016_passive_force_length(
            parameters, normalized_fiber_length
        )
        force_velocity = de_groote_fregly_2016_force_velocity(
            parameters, normalized_fiber_velocity
        )
        fiber_force = (
            state.activation * active_length * force_velocity + passive_length
        )
        mask = jnp.asarray(self.plan.muscle_mask, dtype=bool)
        return jnp.where(mask, tendon_force - fiber_force * cosine, scaled_force_rate)

    def candidate(
        self,
        state: DeGrooteFregly2016State,
        independent_excitation: ArrayLike,
        musculotendon_length_m: ArrayLike,
        musculotendon_velocity_m_per_s: ArrayLike,
        time_step_s: ArrayLike,
        /,
    ) -> DeGrooteFregly2016ImplicitCandidate:
        """Solve S25 for S24's scaled force rate, then propose one Euler step."""

        source = self.constitutive.evaluate(
            state,
            independent_excitation,
            musculotendon_length_m,
            musculotendon_velocity_m_per_s,
        )
        length = self.constitutive._input(
            musculotendon_length_m, "musculotendon_length_m"
        )
        velocity = self.constitutive._input(
            musculotendon_velocity_m_per_s, "musculotendon_velocity_m_per_s"
        )
        scale = self.plan.parameters.implicit_force_rate_scale_per_s
        initial = source.rates.normalized_tendon_force_per_s / scale

        def residual(scaled_force_rate, runtime_args):
            del runtime_args
            return self._algebraic_residual(
                scaled_force_rate, state, length, velocity
            )

        problem = NonlinearSystemProblem(
            residual,
            problem_id=f"{self.prepared_id}:formulation-3-force-rate",
        )
        nonlinear = implicit_root_result(problem, initial)
        scaled_force_rate = nonlinear.state
        algebraic_residual = self._algebraic_residual(
            scaled_force_rate, state, length, velocity
        )
        dt = jnp.asarray(
            time_step_s, dtype=self.plan.parameters.maximum_isometric_force_N.dtype
        )
        if dt.shape != ():
            raise ValueError("time_step_s must be scalar.")
        mask = jnp.asarray(self.plan.muscle_mask, dtype=bool)
        candidate_state = DeGrooteFregly2016State(
            jnp.where(
                mask,
                state.activation + dt * source.rates.activation_per_s,
                state.activation,
            ),
            jnp.where(
                mask,
                state.normalized_tendon_force + dt * scale * scaled_force_rate,
                state.normalized_tendon_force,
            ),
        )
        candidate_evaluation = self.constitutive.evaluate(
            candidate_state,
            independent_excitation,
            length + dt * velocity,
            velocity,
        )
        excitation_ = self.constitutive._input(
            independent_excitation, "independent_excitation"
        )
        excitation_admissible = jnp.all(
            (~mask)
            | (
                jnp.isfinite(excitation_)
                & (excitation_ >= 0.0)
                & (excitation_ <= 1.0)
            )
        )
        time_step_admissible = jnp.isfinite(dt) & (dt > 0.0)
        state_admissible = jnp.all(
            (~mask)
            | (
                (candidate_state.activation >= 0.01)
                & (candidate_state.activation <= 1.0)
                & (candidate_state.normalized_tendon_force >= 0.0)
                & (candidate_state.normalized_tendon_force <= 3.0)
            )
        )
        finite = (
            jnp.all(jnp.isfinite(scaled_force_rate))
            & jnp.all(jnp.isfinite(algebraic_residual))
            & jnp.all(jnp.isfinite(candidate_state.activation))
            & jnp.all(jnp.isfinite(candidate_state.normalized_tendon_force))
        )
        residual_scale = jnp.maximum(1.0, jnp.abs(state.normalized_tendon_force))
        residual_satisfied = jnp.all(
            (~mask)
            | (
                jnp.abs(algebraic_residual)
                <= 256.0
                * jnp.finfo(algebraic_residual.dtype).eps
                * residual_scale
            )
        )
        step_successful = (
            excitation_admissible
            & time_step_admissible
            & state_admissible
            & finite
            & jnp.all(source.successful)
            & jnp.all(candidate_evaluation.successful)
        )
        step_evidence = DeGrooteFregly2016StepEvidence(
            source.evidence,
            candidate_evaluation.evidence,
            excitation_admissible,
            time_step_admissible,
            state_admissible,
            finite,
            step_successful,
        )
        successful = nonlinear.successful & residual_satisfied & step_successful
        evidence = DeGrooteFregly2016ImplicitEvidence(
            step_evidence,
            scaled_force_rate,
            algebraic_residual,
            nonlinear.status,
            finite,
            residual_satisfied,
            successful,
        )
        return DeGrooteFregly2016ImplicitCandidate(
            state,
            candidate_state,
            candidate_evaluation,
            evidence,
            self.prepared_id,
        )

    def commit(
        self, candidate: DeGrooteFregly2016ImplicitCandidate, /
    ) -> DeGrooteFregly2016State:
        """Atomically accept the implicit candidate or return its full source."""

        if not isinstance(candidate, DeGrooteFregly2016ImplicitCandidate):
            raise TypeError("candidate must be DeGrooteFregly2016ImplicitCandidate.")
        if candidate.prepared_id != self.prepared_id:
            raise ValueError("Candidate belongs to another implicit prepared model.")
        return jax.tree.map(
            lambda proposed, previous: jnp.where(
                candidate.evidence.successful, proposed, previous
            ),
            candidate.candidate_state,
            candidate.previous_state,
        )


__all__ = [
    "DeGrooteFregly2016Candidate",
    "DeGrooteFregly2016Evaluation",
    "DeGrooteFregly2016Evidence",
    "DeGrooteFregly2016ImplicitCandidate",
    "DeGrooteFregly2016ImplicitEvidence",
    "DeGrooteFregly2016ImplicitTendonForcePlan",
    "DeGrooteFregly2016Parameters",
    "DeGrooteFregly2016Plan",
    "DeGrooteFregly2016Rates",
    "DeGrooteFregly2016State",
    "DeGrooteFregly2016StepEvidence",
    "PreparedDeGrooteFregly2016ImplicitTendonForce",
    "PreparedDeGrooteFregly2016Musculotendon",
    "de_groote_fregly_2016_active_force_length",
    "de_groote_fregly_2016_force_velocity",
    "de_groote_fregly_2016_inverse_force_velocity",
    "de_groote_fregly_2016_inverse_tendon_force_length",
    "de_groote_fregly_2016_passive_force_length",
    "de_groote_fregly_2016_tendon_force_length",
]
