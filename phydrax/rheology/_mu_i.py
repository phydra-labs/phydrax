#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Pressure-dependent granular friction, derived from primary literature.

Jop, Forterre & Pouliquen (2006), Nature 441:727–730,
https://doi.org/10.1038/nature04801, Eqs. (1)–(3).
Barker & Gray (2017), JFM 828:5–32,
https://doi.org/10.1017/jfm.2017.428, Eqs. (6.3)–(6.4): the final
partial regularization has TWO branches, not a Jop middle/high splice.
Barker et al. (2015), JFM 779:794–818,
https://doi.org/10.1017/jfm.2015.412, Eq. (2.43), restated as Eq. (3.9)
in Barker & Gray (2017). Its incompressible two-dimensional criterion
is a diagnostic, never an admission certificate for a 3-D material.
"""

from __future__ import annotations

from math import isfinite
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..nonlinear import LocalRootPlan
from ..typing import checked, parse


MuIRegularizationKind: TypeAlias = Literal["none", "barker-gray-2017"]


@final
class MuIParameters(StrictModule):
    """Scalar dynamic coefficients; all leaves remain differentiable.

    Barker coefficients are explicit caller input, with no guessed defaults.
    The lower neutral point is computed from the original Jop curve.
    """

    mu_s: Array
    mu_2: Array
    I_0: Array
    grain_diameter_m: Array
    grain_density_kg_m3: Array
    barker_alpha: Array | None
    barker_mu_infinity: Array | None

    @checked
    def __init__(
        self,
        mu_s: ArrayLike,
        mu_2: ArrayLike,
        I_0: ArrayLike,
        grain_diameter_m: ArrayLike,
        grain_density_kg_m3: ArrayLike,
        /,
        *,
        barker_alpha: ArrayLike | None = None,
        barker_mu_infinity: ArrayLike | None = None,
    ) -> None:
        values = tuple(
            jnp.asarray(value)
            for value in (mu_s, mu_2, I_0, grain_diameter_m, grain_density_kg_m3)
        )
        if any(value.shape != () for value in values):
            raise ValueError(
                "MuIParameters coefficients must be scalar; vmap per-point parameters."
            )
        dtype = jnp.result_type(*values)
        if not jnp.issubdtype(dtype, jnp.floating):
            dtype = jnp.dtype(jnp.float32)
        values = tuple(value.astype(dtype) for value in values)
        invalid = ~jnp.all(jnp.isfinite(jnp.stack(values)))
        invalid = invalid | (values[0] <= 0) | (values[1] <= values[0])
        invalid = invalid | (values[2] <= 0) | (values[3] <= 0) | (values[4] <= 0)
        values = eqx.error_if(
            values,
            invalid,
            "MuIParameters must have 0 < mu_s < mu_2 and positive I_0, diameter and density.",
        )
        if (barker_alpha is None) != (barker_mu_infinity is None):
            raise ValueError("Both Barker–Gray coefficients must be supplied together.")
        alpha = None if barker_alpha is None else jnp.asarray(barker_alpha, dtype=dtype)
        infinity = (
            None
            if barker_mu_infinity is None
            else jnp.asarray(barker_mu_infinity, dtype=dtype)
        )
        if alpha is not None and infinity is not None:
            if alpha.shape != () or infinity.shape != ():
                raise ValueError("Barker–Gray coefficients must be scalar.")
            alpha, infinity = eqx.error_if(
                (alpha, infinity),
                ~jnp.isfinite(alpha)
                | (alpha <= 0)
                | (alpha >= 2)
                | ~jnp.isfinite(infinity)
                | (infinity < 0),
                "Barker–Gray requires 0 < alpha < 2 and mu_infinity >= 0.",
            )
        self.mu_s, self.mu_2, self.I_0 = values[:3]
        self.grain_diameter_m, self.grain_density_kg_m3 = values[3:]
        self.barker_alpha = alpha
        self.barker_mu_infinity = infinity


@final
class MuIPackingFractionLaw(StrictModule):
    """Diagnostic empirical steady target Phi_max - b I; not a flow rule."""

    maximum_packing_fraction: float = eqx.field(static=True)
    dilatancy_rate: float = eqx.field(static=True)

    def __init__(self, maximum_packing_fraction: float, dilatancy_rate: float, /) -> None:
        if not isfinite(maximum_packing_fraction) or not 0 < maximum_packing_fraction < 1:
            raise ValueError("Maximum packing fraction must be finite and in (0,1).")
        if not isfinite(dilatancy_rate) or dilatancy_rate < 0:
            raise ValueError("Dilatancy rate must be finite and nonnegative.")
        self.maximum_packing_fraction = maximum_packing_fraction
        self.dilatancy_rate = dilatancy_rate

    def evaluate(self, inertial_number: ArrayLike, /) -> Array:
        return self.maximum_packing_fraction - self.dilatancy_rate * jnp.asarray(
            inertial_number
        )


@final
class MuILawEvaluation(StrictModule):
    inertial_number: Array
    friction_coefficient: Array
    apparent_viscosity_pa_s: Array
    shear_stress_magnitude_pa: Array
    packing_fraction: Array | None
    pressure_active: Array
    separated: Array
    viscosity_defined: Array
    hadamard_margin: Array
    hadamard_applicable: Array
    finite: Array
    successful: Array
    neutral_point_valid: Array
    lower_neutral_inertial_number: Array
    law_id: str = eqx.field(static=True)


def _jop_neutral_condition(x: Array, parameters: MuIParameters) -> Array:
    delta = parameters.mu_2 - parameters.mu_s
    mu = parameters.mu_s + delta * x
    r = delta * x * (1 - x) / mu
    return 4 * r * r - 4 * r + mu * mu * (1 - r / 2) ** 2


def _lower_neutral(parameters: MuIParameters) -> tuple[Array, Array]:
    # Fixed-capacity bracketing in x=I/(I0+I). Missing/narrow neutral
    # intervals are explicitly unsupported, never replaced by a fitted number.
    x = jnp.linspace(0.0, 1.0, 257, dtype=parameters.mu_s.dtype)
    c = _jop_neutral_condition(x, parameters)
    crosses = (c[:-1] > 0) & (c[1:] <= 0)
    index = jnp.argmax(crosses)
    lower, upper = x[index], x[index + 1]

    def bisect(_: int | Array, bracket: tuple[Array, Array]) -> tuple[Array, Array]:
        lo, hi = bracket
        mid = (lo + hi) / 2
        above = _jop_neutral_condition(mid, parameters) > 0
        return jnp.where(above, mid, lo), jnp.where(above, hi, mid)

    lower, upper = jax.lax.fori_loop(0, 32, bisect, (lower, upper))
    guess = jax.lax.stop_gradient((lower + upper) / 2)
    root_plan = LocalRootPlan(
        maximum_steps=4, tolerance=1e-6, plan_id="mu-i-jop-lower-neutral"
    )

    def residual(value: Array) -> Array:
        return _jop_neutral_condition(value, parameters)

    root, diagnostics = root_plan.solve_with_diagnostics(residual, guess)
    valid = jnp.any(crosses) & diagnostics.converged & (root > 0) & (root < 1)
    operand = jnp.where(valid, root, 0.5)
    return parameters.I_0 * operand / (1 - operand), valid


@final
class MuILaw(StrictModule):
    """Static curve structure with explicit dynamic material parameters.

    evaluate supports jit/vmap and first-order coefficient/rate/pressure AD on
    strict branches. Positive pressure at zero rate has undefined viscosity;
    separation has zero stress and zero viscosity. No pressure or rate floor.
    The Hadamard diagnostic applies only to a flowing, incompressible 2-D
    friction-only interpretation; it does not certify MPM or 3-D FLIP.
    """

    regularization: MuIRegularizationKind = eqx.field(static=True)
    packing_fraction_law: MuIPackingFractionLaw | None = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        /,
        *,
        regularization: MuIRegularizationKind = "none",
        packing_fraction_law: MuIPackingFractionLaw | None = None,
    ) -> None:
        kind = parse(regularization, MuIRegularizationKind, "regularization")
        self.regularization = kind
        self.packing_fraction_law = packing_fraction_law
        self.law_id = canonical_fingerprint(
            {
                "kind": "mu-i-law",
                "regularization": kind,
                "packing": None
                if packing_fraction_law is None
                else {
                    "maximum": packing_fraction_law.maximum_packing_fraction,
                    "slope": packing_fraction_law.dilatancy_rate,
                },
            }
        )

    @checked
    def evaluate(
        self,
        shear_rate_s_inv: ArrayLike,
        pressure_pa: ArrayLike,
        parameters: MuIParameters,
        /,
    ) -> MuILawEvaluation:
        rate, pressure = jnp.broadcast_arrays(
            jnp.asarray(shear_rate_s_inv), jnp.asarray(pressure_pa)
        )
        valid = jnp.isfinite(rate) & (rate >= 0) & jnp.isfinite(pressure)
        active = pressure > 0
        separated = valid & ~active
        flowing = active & (rate > 0) & valid
        p = jnp.where(active, pressure, 1.0)
        inertial = jnp.where(
            active,
            rate
            * parameters.grain_diameter_m
            * jnp.sqrt(parameters.grain_density_kg_m3 / p),
            0.0,
        )
        mu = parameters.mu_s + (parameters.mu_2 - parameters.mu_s) * inertial / (
            parameters.I_0 + inertial
        )
        slope = (
            (parameters.mu_2 - parameters.mu_s)
            * parameters.I_0
            / (parameters.I_0 + inertial) ** 2
        )
        neutral = jnp.asarray(0.0, dtype=inertial.dtype)
        neutral_valid = jnp.asarray(True)
        match self.regularization:
            case "none":
                pass
            case "barker-gray-2017":
                alpha, infinity = parameters.barker_alpha, parameters.barker_mu_infinity
                if alpha is None or infinity is None:
                    raise ValueError(
                        "Barker–Gray law requires explicit alpha and mu_infinity parameters."
                    )
                neutral, neutral_valid = _lower_neutral(parameters)
                upper = (
                    parameters.mu_s * parameters.I_0
                    + parameters.mu_2 * inertial
                    + infinity * inertial**2
                ) / (parameters.I_0 + inertial)
                upper_slope = (
                    (parameters.mu_2 - parameters.mu_s) * parameters.I_0
                    + infinity * inertial * (2 * parameters.I_0 + inertial)
                ) / (parameters.I_0 + inertial) ** 2
                match_mu = (
                    parameters.mu_s * parameters.I_0
                    + parameters.mu_2 * neutral
                    + infinity * neutral**2
                ) / (parameters.I_0 + neutral)
                # Log form avoids overflow in A=I_N1 exp(alpha/mu_match²).
                positive_i = jnp.where(inertial > 0, inertial, neutral)
                logarithm = jnp.log(neutral / positive_i) + alpha / match_mu**2
                low_branch = inertial <= neutral
                log_operand = jnp.where(low_branch, logarithm, 1.0)
                low = jnp.sqrt(alpha / log_operand)
                low = jnp.where(inertial > 0, low, 0.0)
                low_slope = low / (2 * positive_i * log_operand)
                mu = jnp.where(low_branch, low, upper)
                slope = jnp.where(low_branch, low_slope, upper_slope)
            case _:
                assert_never(self.regularization)
        mu_operand = jnp.where(mu > 0, mu, 1.0)
        r = inertial * slope / mu_operand
        margin = -(4 * r**2 - 4 * r + mu**2 * (1 - r / 2) ** 2)
        mu = jnp.where(active, mu, 0.0)
        stress = jnp.where(active, pressure * mu, 0.0)
        eta = pressure * mu / jnp.where(flowing, rate, 1.0)
        eta = jnp.where(active, jnp.where(flowing, eta, jnp.inf), 0.0)
        defined = valid & (~active | flowing) & neutral_valid
        finite = jnp.isfinite(eta) & jnp.isfinite(stress) & jnp.isfinite(inertial)
        packing = (
            None
            if self.packing_fraction_law is None
            else self.packing_fraction_law.evaluate(inertial)
        )
        return MuILawEvaluation(
            inertial,
            mu,
            eta,
            stress,
            packing,
            active,
            separated,
            defined,
            margin,
            flowing & neutral_valid,
            finite,
            defined & finite,
            neutral_valid,
            neutral,
            self.law_id,
        )
