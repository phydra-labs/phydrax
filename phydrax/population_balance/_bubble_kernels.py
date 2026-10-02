#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Turbulent bubble coalescence and breakage kernels for sectional population balances.

The kernels evaluate on a `BubbleSectionalPlan`: strictly increasing pivot
diameters ``d_i`` whose gas volumes ``v_i = π d_i³/6`` are the additive pivots
consumed by `ConservativeSectionalSolver`. Coalescence kernels return the
symmetric number-based matrix ``K_ij`` (m³ s⁻¹, loss ``N_i Σ_j K_ij N_j``);
the breakage kernel returns the per-bubble frequency ``g_j`` (s⁻¹) and the
daughter-number matrix ``D_ij`` whose columns conserve gas volume,
``Σ_i v_i D_ij = v_j``. Every result carries `BubbleKernelEvidence`: the
dissipation rate used, the Weber range ``We = ρ ε^{2/3} d^{5/3}/σ`` over the
pivots, inertial-subrange applicability (Kolmogorov length ``η = (ν³/ε)^{1/4}``
and an optional integral scale), the largest exponential argument and whether
it left the float64 range (the factor is then represented as zero, never
clipped), finiteness, and a status.

Sources (equations only):

- Prince & Blanch, AIChE J. 36 (1990) 1485, doi:10.1002/aic.690361004.
- Lehr, Millies & Mewes, AIChE J. 48 (2002) 2426, doi:10.1002/aic.690481103.
- Luo & Svendsen, AIChE J. 42 (1996) 1225, doi:10.1002/aic.690420505.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import gammainc, gammaincc
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import fixed_field, NonTrainableState, parameter_field
from .._validation import positive_integer
from ..ein import contract
from ..typing import as_host_array, checked, Dim, Float64, HostFloat64, Scalar


_EXPONENT_LIMIT: Final = -math.log(float(np.finfo(np.float64).tiny))
"""Largest ``x`` for which ``exp(-x)`` is a normal float64 number."""

# Luo–Svendsen closed form: shape parameters s_k = (8 − 3k)/11 and binomial
# weights of (1 + ξ)² = 1 + 2ξ + ξ², k = 0, 1, 2.
_EDDY_SHAPES: Final = (8.0 / 11.0, 5.0 / 11.0, 2.0 / 11.0)
_EDDY_WEIGHTS: Final = tuple(
    weight * math.gamma(shape)
    for weight, shape in zip((1.0, 2.0, 1.0), _EDDY_SHAPES, strict=True)
)


class _BubbleSectionDim(Dim, minimum=2):
    """Pivot sections of one bubble population."""


class _BreakageFractionDim(Dim, minimum=1):
    """Nodes of the breakage volume-fraction quadrature."""


class BubbleKernelStatus(IntEnum):
    """Outcome of one kernel evaluation, in increasing severity.

    `SUCCESS` and `EXPONENT_OUT_OF_RANGE` are successful: in the latter at least
    one factor ``exp(-x)`` had ``x`` beyond the float64 normal range and is
    represented as (sub)normal or zero, which is accurate in absolute terms.
    `OUTSIDE_SUPPORT` (Weber range or inertial subrange outside the declared
    support) and `NONFINITE` are refusals.
    """

    SUCCESS = 0
    EXPONENT_OUT_OF_RANGE = 1
    OUTSIDE_SUPPORT = 2
    NONFINITE = 3


def _physical_scalar(
    value: ArrayLike,
    name: str,
    /,
    *,
    lower: float = 0.0,
    inclusive: bool = False,
    upper: float | None = None,
) -> Array:
    """Validate one finite scalar coefficient on the host; return a float64 leaf."""
    host = as_host_array(value, HostFloat64[Scalar], name)
    number = float(host)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    if (number < lower) if inclusive else (number <= lower):
        raise ValueError(f"{name} must be {'>=' if inclusive else '>'} {lower}.")
    if upper is not None and number >= upper:
        raise ValueError(f"{name} must be < {upper}.")
    return jnp.asarray(host, dtype=jnp.float64)


def _weber_support(value: tuple[float, float] | None, /) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("weber_support must be a (minimum, maximum) tuple or None.")
    lower, upper = float(value[0]), float(value[1])
    if math.isnan(lower) or math.isnan(upper) or lower < 0.0 or upper <= lower:
        raise ValueError("weber_support must satisfy 0 <= minimum < maximum.")
    return lower, upper


class TurbulentBubblyLiquid(StrictModule):
    """Continuous-phase state seen by turbulent bubble kernels (SI units).

    ``density`` ρ, ``surface_tension`` σ, ``kinematic_viscosity`` ν and the
    turbulent ``dissipation_rate`` ε are inferable parameter leaves. The optional
    ``integral_length_scale`` (fixed leaf) only bounds the inertial-subrange
    evidence from above; the Kolmogorov length ``(ν³/ε)^{1/4}`` bounds it below.
    """

    __strict_contract__ = True

    density: Float64[Scalar] = parameter_field()
    surface_tension: Float64[Scalar] = parameter_field()
    kinematic_viscosity: Float64[Scalar] = parameter_field()
    dissipation_rate: Float64[Scalar] = parameter_field()
    integral_length_scale: Float64[Scalar] | None = fixed_field()

    def __init__(
        self,
        density: ArrayLike,
        surface_tension: ArrayLike,
        kinematic_viscosity: ArrayLike,
        dissipation_rate: ArrayLike,
        /,
        *,
        integral_length_scale: ArrayLike | None = None,
    ) -> None:
        rho = _physical_scalar(density, "density")
        sigma = _physical_scalar(surface_tension, "surface_tension")
        nu = _physical_scalar(kinematic_viscosity, "kinematic_viscosity")
        epsilon = _physical_scalar(dissipation_rate, "dissipation_rate")
        scale = (
            None
            if integral_length_scale is None
            else _physical_scalar(integral_length_scale, "integral_length_scale")
        )
        self.density = rho
        self.surface_tension = sigma
        self.kinematic_viscosity = nu
        self.dissipation_rate = epsilon
        self.integral_length_scale = scale

    def kolmogorov_length(self) -> Array:
        """Kolmogorov dissipation length ``η = (ν³/ε)^{1/4}``."""
        return (self.kinematic_viscosity**3 / self.dissipation_rate) ** 0.25

    def weber_numbers(self, diameters: Array, /) -> Array:
        """Turbulent Weber numbers ``ρ ε^{2/3} d^{5/3}/σ`` of the given diameters."""
        return (
            self.density
            * jnp.cbrt(self.dissipation_rate) ** 2
            * diameters ** (5.0 / 3.0)
            / self.surface_tension
        )


def _fixed_pivot_weights(volumes: np.ndarray, daughters: np.ndarray, /) -> np.ndarray:
    """Place each daughter volume on the pivots (sections × daughters).

    Inside the pivot range the two bracketing pivots receive the lever-rule
    weights, which preserve both number and volume; a daughter below the
    smallest pivot moves its whole volume to that pivot (number is not kept).
    """
    count = volumes.size
    upper = np.clip(np.searchsorted(volumes, daughters, side="left"), 1, count - 1)
    lower = upper - 1
    fraction = (daughters - volumes[lower]) / (volumes[upper] - volumes[lower])
    below = daughters < volumes[0]
    columns = np.arange(daughters.size)
    weights = np.zeros((count, daughters.size), dtype=np.float64)
    weights[lower, columns] = np.where(below, daughters / volumes[0], 1.0 - fraction)
    weights[upper, columns] += np.where(below, 0.0, fraction)
    return weights


class BubbleSectionalPlan(StrictModule, NonTrainableState):
    """Sectional bubble pivots and the fixed breakage-fraction quadrature.

    ``diameters`` (m) must be positive and strictly increasing; ``volumes`` are
    the gas volumes ``π d³/6`` (m³) to pass as `ConservativeSectionalSolver`
    pivots. Binary breakage of a parent ``v_j`` into ``f v_j`` and
    ``(1 − f) v_j`` is integrated over ``f ∈ (0, 1/2]`` with a
    ``breakage_order``-point Gauss–Legendre rule in ``t`` after the substitution
    ``f = t³/2``, which removes the ``f^{2/3}`` endpoint singularity of the
    surface-energy increase. The weights sum to 1/2.
    ``daughter_assignment[i, j, k]`` is the fixed-pivot share of the daughter
    pair of node ``k`` of parent ``j`` placed on pivot ``i``; its
    volume moment equals ``v_j`` for every ``(j, k)``.
    """

    __strict_contract__ = True

    diameters: Float64[_BubbleSectionDim]
    volumes: Float64[_BubbleSectionDim]
    breakage_fractions: Float64[_BreakageFractionDim]
    breakage_weights: Float64[_BreakageFractionDim]
    daughter_assignment: Float64[
        _BubbleSectionDim, _BubbleSectionDim, _BreakageFractionDim
    ]
    plan_id: str = eqx.field(static=True)

    def __init__(self, diameters: ArrayLike, /, *, breakage_order: int = 32) -> None:
        values = as_host_array(diameters, HostFloat64[_BubbleSectionDim], "diameters")
        if not np.all(np.isfinite(values)) or values[0] <= 0.0:
            raise ValueError("Bubble diameters must be finite and positive.")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError("Bubble diameters must be strictly increasing.")
        order = positive_integer(breakage_order, "breakage_order")
        volumes = np.pi * values**3 / 6.0
        if np.any(np.diff(volumes) <= 0.0):
            raise ValueError("Bubble volumes must be strictly increasing.")
        nodes, weights = np.polynomial.legendre.leggauss(order)
        unit = 0.5 * (nodes + 1.0)
        fractions = 0.5 * unit**3
        fraction_weights = 0.75 * weights * unit**2
        small = volumes[:, None] * fractions[None, :]
        large = volumes[:, None] * (1.0 - fractions[None, :])
        count = values.size
        assignment = (
            _fixed_pivot_weights(volumes, small.ravel())
            + _fixed_pivot_weights(volumes, large.ravel())
        ).reshape(count, count, order)
        self.diameters = jnp.asarray(values)
        self.volumes = jnp.asarray(volumes)
        self.breakage_fractions = jnp.asarray(fractions)
        self.breakage_weights = jnp.asarray(fraction_weights)
        self.daughter_assignment = jnp.asarray(assignment)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bubble-sectional-plan",
                "diameters": values.tolist(),
                "breakage_order": order,
            }
        )


class BubbleKernelEvidence(StrictModule):
    """Applicability and numerical evidence shared by every bubble kernel.

    ``maximum_exponent_argument`` is the largest ``x`` of the model's
    ``exp(-x)`` factors; ``exponent_out_of_range`` reports that it exceeded
    the float64 normal range. ``successful`` is ``finite`` and inside support.
    """

    dissipation_rate: Array
    kolmogorov_length: Array
    minimum_weber: Array
    maximum_weber: Array
    weber_within_support: Array
    inertial_subrange: Array
    maximum_exponent_argument: Array
    exponent_out_of_range: Array
    finite: Array
    status: Array
    successful: Array


class BubbleCoalescenceKernelResult(StrictModule):
    """Coalescence kernel ``K = collision_frequency · efficiency`` (m³ s⁻¹).

    ``symmetry_residual`` is ``max|K − Kᵀ|/max|K|`` (zero by construction).
    """

    kernel: Array
    collision_frequency: Array
    efficiency: Array
    symmetry_residual: Array
    evidence: BubbleKernelEvidence
    plan_id: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)


class BubbleBreakageKernelResult(StrictModule):
    """Breakage frequency (s⁻¹), daughter-number matrix and partial rates.

    ``partial_rates[j, k]`` is the rate density per unit volume fraction at node
    ``k`` of parent ``j``; ``daughter_number[:, j]`` counts sectional daughters
    per breakage event of section ``j`` (identity where ``frequency`` is zero).
    ``daughter_volume_residual`` is the largest relative volume defect
    ``|Σ_i v_i D_ij − v_j|/v_j`` over breaking sections.
    """

    frequency: Array
    daughter_number: Array
    partial_rates: Array
    daughter_volume_residual: Array
    evidence: BubbleKernelEvidence
    plan_id: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)


def _evidence(
    plan: BubbleSectionalPlan,
    liquid: TurbulentBubblyLiquid,
    weber_support: tuple[float, float] | None,
    exponent_argument: Array,
    finite: Array,
    /,
) -> BubbleKernelEvidence:
    diameters = plan.diameters
    kolmogorov = liquid.kolmogorov_length()
    weber = liquid.weber_numbers(diameters)
    minimum_weber, maximum_weber = jnp.min(weber), jnp.max(weber)
    weber_within = (
        jnp.asarray(True)
        if weber_support is None
        else (minimum_weber >= weber_support[0]) & (maximum_weber <= weber_support[1])
    )
    inertial = jnp.all(diameters > kolmogorov)
    if liquid.integral_length_scale is not None:
        inertial = inertial & jnp.all(diameters < liquid.integral_length_scale)
    out_of_range = exponent_argument > _EXPONENT_LIMIT
    within = weber_within & inertial
    status = jnp.where(
        ~finite,
        BubbleKernelStatus.NONFINITE.value,
        jnp.where(
            ~within,
            BubbleKernelStatus.OUTSIDE_SUPPORT.value,
            jnp.where(
                out_of_range,
                BubbleKernelStatus.EXPONENT_OUT_OF_RANGE.value,
                BubbleKernelStatus.SUCCESS.value,
            ),
        ),
    ).astype(jnp.int32)
    return BubbleKernelEvidence(
        liquid.dissipation_rate,
        kolmogorov,
        minimum_weber,
        maximum_weber,
        weber_within,
        inertial,
        exponent_argument,
        out_of_range,
        finite,
        status,
        finite & within,
    )


def _symmetry_residual(kernel: Array, /) -> Array:
    scale = jnp.maximum(jnp.max(jnp.abs(kernel)), jnp.finfo(kernel.dtype).tiny)
    return jnp.max(jnp.abs(kernel - kernel.T)) / scale


def _pair_sums(diameters: Array, /) -> tuple[Array, Array]:
    """``d_i + d_j`` and ``d_i^{2/3} + d_j^{2/3}``, exactly symmetric."""
    two_thirds = jnp.cbrt(diameters) ** 2
    return (
        diameters[:, None] + diameters[None, :],
        two_thirds[:, None] + two_thirds[None, :],
    )


class PrinceBlanchCoalescenceKernel(StrictModule):
    """Prince & Blanch (1990) turbulent coalescence with film-drainage efficiency.

    ``K_ij = (θ^T_ij + θ^LS_ij) λ_ij`` with the turbulent collision rate (Eq. 8)
    ``θ^T = C₁ π (d_i + d_j)² ε^{1/3} (d_i^{2/3} + d_j^{2/3})^{1/2}``, ``C₁ = 0.089``;
    the optional laminar-shear (orthokinetic) rate
    ``θ^LS = (4/3)(r_i + r_j)³ γ̇ = (d_i + d_j)³ γ̇/6`` when ``shear_rate`` γ̇ is
    declared; and the efficiency ``λ = exp(−t_ij/τ_ij)`` (Eq. 16) with drainage
    time ``t = (r_ij³ ρ/(16σ))^{1/2} ln(h₀/h_f)`` (Eq. 18) and turbulent contact
    time ``τ = r_ij^{2/3} ε^{-1/3}`` (Eq. 20). The equivalent radius is
    ``r_ij = 2 r_i r_j/(r_i + r_j)`` (Chesters–Hofman), which reduces to the
    bubble radius for equal bubbles as the paper's text requires.
    ``h₀ = 10⁻⁴ m`` and ``h_f = 10⁻⁸ m`` are the paper's air–water values.
    Buoyancy-driven collisions are not modeled.
    """

    __strict_contract__ = True

    liquid: TurbulentBubblyLiquid
    collision_coefficient: Float64[Scalar] = parameter_field()
    initial_film_thickness: Float64[Scalar] = parameter_field()
    critical_film_thickness: Float64[Scalar] = parameter_field()
    shear_rate: Float64[Scalar] | None = parameter_field()
    weber_support: tuple[float, float] | None = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        liquid: TurbulentBubblyLiquid,
        /,
        *,
        collision_coefficient: ArrayLike = 0.089,
        initial_film_thickness: ArrayLike = 1.0e-4,
        critical_film_thickness: ArrayLike = 1.0e-8,
        shear_rate: ArrayLike | None = None,
        weber_support: tuple[float, float] | None = None,
    ) -> None:
        coefficient = _physical_scalar(collision_coefficient, "collision_coefficient")
        initial = _physical_scalar(initial_film_thickness, "initial_film_thickness")
        critical = _physical_scalar(critical_film_thickness, "critical_film_thickness")
        if float(critical) >= float(initial):
            raise ValueError(
                "critical_film_thickness must be smaller than initial_film_thickness."
            )
        shear = (
            None
            if shear_rate is None
            else _physical_scalar(shear_rate, "shear_rate", inclusive=True)
        )
        support = _weber_support(weber_support)
        self.liquid = liquid
        self.collision_coefficient = coefficient
        self.initial_film_thickness = initial
        self.critical_film_thickness = critical
        self.shear_rate = shear
        self.weber_support = support
        self.kernel_id = canonical_fingerprint(
            {
                "kind": "prince-blanch-coalescence",
                "laminar_shear": shear is not None,
                "weber_support": support,
            }
        )

    def evaluate(self, plan: BubbleSectionalPlan, /) -> BubbleCoalescenceKernelResult:
        """Dense coalescence kernel over the plan pivots."""
        liquid = self.liquid
        diameters = plan.diameters
        pair_sum, pair_two_thirds = _pair_sums(diameters)
        epsilon_third = jnp.cbrt(liquid.dissipation_rate)
        collision = (
            self.collision_coefficient
            * jnp.pi
            * pair_sum**2
            * epsilon_third
            * jnp.sqrt(pair_two_thirds)
        )
        if self.shear_rate is not None:
            collision = collision + pair_sum**3 * self.shear_rate / 6.0
        radius = diameters[:, None] * diameters[None, :] / pair_sum
        drainage = jnp.sqrt(
            radius**3 * liquid.density / (16.0 * liquid.surface_tension)
        ) * jnp.log(self.initial_film_thickness / self.critical_film_thickness)
        contact = jnp.cbrt(radius) ** 2 / epsilon_third
        argument = drainage / contact
        efficiency = jnp.exp(-argument)
        kernel = collision * efficiency
        finite = (
            jnp.all(jnp.isfinite(kernel))
            & jnp.all(jnp.isfinite(collision))
            & jnp.all(jnp.isfinite(efficiency))
        )
        evidence = _evidence(
            plan,
            liquid,
            self.weber_support,
            jnp.max(argument),
            finite,
        )
        return BubbleCoalescenceKernelResult(
            kernel,
            collision,
            efficiency,
            _symmetry_residual(kernel),
            evidence,
            plan.plan_id,
            self.kernel_id,
        )


class LehrCoalescenceKernel(StrictModule):
    """Lehr, Millies & Mewes (2002) critical-approach-velocity coalescence.

    ``K_ij = (π/4)(d_i + d_j)² min(u′, u_crit) exp(−((α_max/α)^{1/3} − 1)²)`` with
    the characteristic velocity ``u′ = max(√2 ε^{1/3} (d_i^{2/3} + d_j^{2/3})^{1/2},
    |u_i − u_j|)``; the rise-velocity difference enters only when per-section
    ``rise_velocities`` are passed to `evaluate`. Collisions faster than
    ``u_crit`` bounce, so the efficiency is ``min(u′, u_crit)/u′``; the
    void-fraction factor (mean bubble spacing relative to diameter) is part of
    the collision frequency. Defaults ``u_crit = 0.08 m/s`` (air–water) and
    ``α_max = 0.6``.
    """

    __strict_contract__ = True

    liquid: TurbulentBubblyLiquid
    gas_volume_fraction: Float64[Scalar] = parameter_field()
    critical_velocity: Float64[Scalar] = parameter_field()
    maximum_packing_fraction: Float64[Scalar] = parameter_field()
    weber_support: tuple[float, float] | None = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        liquid: TurbulentBubblyLiquid,
        gas_volume_fraction: ArrayLike,
        /,
        *,
        critical_velocity: ArrayLike = 0.08,
        maximum_packing_fraction: ArrayLike = 0.6,
        weber_support: tuple[float, float] | None = None,
    ) -> None:
        packing = _physical_scalar(
            maximum_packing_fraction, "maximum_packing_fraction", upper=1.0
        )
        holdup = _physical_scalar(
            gas_volume_fraction, "gas_volume_fraction", upper=float(packing)
        )
        velocity = _physical_scalar(critical_velocity, "critical_velocity")
        support = _weber_support(weber_support)
        self.liquid = liquid
        self.gas_volume_fraction = holdup
        self.critical_velocity = velocity
        self.maximum_packing_fraction = packing
        self.weber_support = support
        self.kernel_id = canonical_fingerprint(
            {"kind": "lehr-coalescence", "weber_support": support}
        )

    def evaluate(
        self,
        plan: BubbleSectionalPlan,
        /,
        *,
        rise_velocities: ArrayLike | None = None,
    ) -> BubbleCoalescenceKernelResult:
        """Dense coalescence kernel; ``rise_velocities`` (m/s) align with the pivots."""
        liquid = self.liquid
        diameters = plan.diameters
        pair_sum, pair_two_thirds = _pair_sums(diameters)
        velocity = (
            jnp.sqrt(2.0) * jnp.cbrt(liquid.dissipation_rate) * jnp.sqrt(pair_two_thirds)
        )
        if rise_velocities is not None:
            rise = jnp.asarray(rise_velocities, dtype=jnp.float64)
            if rise.shape != diameters.shape:
                raise ValueError("rise_velocities must align with the plan pivots.")
            velocity = jnp.maximum(velocity, jnp.abs(rise[:, None] - rise[None, :]))
        argument = (
            jnp.cbrt(self.maximum_packing_fraction / self.gas_volume_fraction) - 1.0
        ) ** 2
        cross_section = 0.25 * jnp.pi * pair_sum**2 * jnp.exp(-argument)
        limited = jnp.minimum(velocity, self.critical_velocity)
        collision = cross_section * velocity
        kernel = cross_section * limited
        efficiency = limited / velocity
        finite = (
            jnp.all(jnp.isfinite(kernel))
            & jnp.all(jnp.isfinite(collision))
            & jnp.all(jnp.isfinite(efficiency))
        )
        evidence = _evidence(
            plan,
            liquid,
            self.weber_support,
            argument,
            finite,
        )
        return BubbleCoalescenceKernelResult(
            kernel,
            collision,
            efficiency,
            _symmetry_residual(kernel),
            evidence,
            plan.plan_id,
            self.kernel_id,
        )


def _eddy_integral(scale: Array, minimum_ratio: Array, /) -> Array:
    upper = scale * minimum_ratio ** (-11.0 / 3.0)
    total = jnp.zeros(
        jnp.broadcast_shapes(scale.shape, minimum_ratio.shape), dtype=jnp.float64
    )
    for shape, weight in zip(_EDDY_SHAPES, _EDDY_WEIGHTS, strict=True):
        # Difference the smaller tail: P below the median, Q above it.
        lower_tail = gammainc(shape, scale)
        mass = jnp.where(
            lower_tail < 0.5,
            gammainc(shape, upper) - lower_tail,
            gammaincc(shape, scale) - gammaincc(shape, upper),
        )
        total = total + weight * scale ** (-shape) * mass
    return (3.0 / 11.0) * total


def luo_svendsen_eddy_integral(scale: ArrayLike, minimum_ratio: ArrayLike, /) -> Array:
    """Closed form of ``∫_{ξmin}^1 (1 + ξ)² ξ^{-11/3} exp(−b ξ^{-11/3}) dξ``.

    With ``u = b ξ^{-11/3}`` each monomial of ``(1 + ξ)² = Σ_k c_k ξ^k``
    (``c = 1, 2, 1``) integrates to
    ``(3/11) b^{-s_k} Γ(s_k) [Q(s_k, b) − Q(s_k, b ξmin^{-11/3})]`` with
    ``s_k = (8 − 3k)/11`` and ``Q`` the regularized upper incomplete gamma
    function (evaluated through the smaller tail). Requires ``b > 0`` and
    ``0 < ξmin ≤ 1``; arguments broadcast. Differentiable in both arguments.
    """
    b = jnp.asarray(scale, dtype=jnp.float64)
    ratio = jnp.asarray(minimum_ratio, dtype=jnp.float64)
    b = eqx.error_if(
        b,
        jnp.any(~(b > 0.0)) | jnp.any(~((ratio > 0.0) & (ratio <= 1.0))),
        "Luo–Svendsen integral requires b > 0 and 0 < minimum_ratio <= 1.",
    )
    return _eddy_integral(b, ratio)


class LuoSvendsenBreakageKernel(StrictModule):
    """Luo & Svendsen (1996) binary turbulent breakage.

    The partial rate per unit breakage volume fraction ``f`` (Eq. 27) is
    ``Ω(f; d) = C₄ (1 − α)(ε/d²)^{1/3} ∫_{ξmin}^1 (1 + ξ)² ξ^{-11/3}
    exp(−12 c_f σ/(β ρ ε^{2/3} d^{5/3} ξ^{11/3})) dξ`` with
    ``c_f = f^{2/3} + (1 − f)^{2/3} − 1`` and ``ξmin = λ_min/d``,
    ``λ_min = C₅ η``. The breakage frequency is ``g(d) = ½ ∫_0^1 Ω df`` and the
    daughter pair distribution is ``Ω(f)/g`` over ``f ∈ (0, 1/2]``; both use the
    plan quadrature, and the ξ integral uses `luo_svendsen_eddy_integral`.
    Sections with ``ξmin ≥ 1`` (bubble below the smallest breaking eddy) do not
    break. Defaults ``C₄ = 0.923``, ``β = 2.045``, ``C₅ = 11.4``.
    """

    __strict_contract__ = True

    liquid: TurbulentBubblyLiquid
    gas_volume_fraction: Float64[Scalar] = parameter_field()
    collision_coefficient: Float64[Scalar] = parameter_field()
    eddy_velocity_coefficient: Float64[Scalar] = parameter_field()
    minimum_eddy_ratio: Float64[Scalar] = parameter_field()
    weber_support: tuple[float, float] | None = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        liquid: TurbulentBubblyLiquid,
        gas_volume_fraction: ArrayLike,
        /,
        *,
        collision_coefficient: ArrayLike = 0.923,
        eddy_velocity_coefficient: ArrayLike = 2.045,
        minimum_eddy_ratio: ArrayLike = 11.4,
        weber_support: tuple[float, float] | None = None,
    ) -> None:
        holdup = _physical_scalar(
            gas_volume_fraction, "gas_volume_fraction", inclusive=True, upper=1.0
        )
        coefficient = _physical_scalar(collision_coefficient, "collision_coefficient")
        beta = _physical_scalar(eddy_velocity_coefficient, "eddy_velocity_coefficient")
        ratio = _physical_scalar(minimum_eddy_ratio, "minimum_eddy_ratio")
        support = _weber_support(weber_support)
        self.liquid = liquid
        self.gas_volume_fraction = holdup
        self.collision_coefficient = coefficient
        self.eddy_velocity_coefficient = beta
        self.minimum_eddy_ratio = ratio
        self.weber_support = support
        self.kernel_id = canonical_fingerprint(
            {"kind": "luo-svendsen-breakage", "weber_support": support}
        )

    def evaluate(self, plan: BubbleSectionalPlan, /) -> BubbleBreakageKernelResult:
        """Breakage frequencies and volume-conserving daughter matrix on the plan."""
        liquid = self.liquid
        diameters = plan.diameters
        fractions = plan.breakage_fractions
        epsilon = liquid.dissipation_rate
        minimum_ratio = self.minimum_eddy_ratio * liquid.kolmogorov_length() / diameters
        breaking = minimum_ratio < 1.0
        # ξmin = 1 gives an exactly empty eddy range, so larger ratios are
        # evaluated there: those sections break at rate zero by the model.
        ratio = jnp.minimum(minimum_ratio, 1.0)[:, None]
        surface = fractions ** (2.0 / 3.0) + jnp.expm1(
            (2.0 / 3.0) * jnp.log1p(-fractions)
        )
        scale = (
            12.0
            * liquid.surface_tension
            * surface[None, :]
            / (
                self.eddy_velocity_coefficient
                * liquid.density
                * jnp.cbrt(epsilon) ** 2
                * diameters[:, None] ** (5.0 / 3.0)
            )
        )
        prefactor = (
            self.collision_coefficient
            * (1.0 - self.gas_volume_fraction)
            * jnp.cbrt(epsilon / diameters**2)
        )
        partial = prefactor[:, None] * _eddy_integral(scale, ratio)
        weighted = partial * plan.breakage_weights[None, :]
        frequency = jnp.sum(weighted, axis=1)
        active = frequency > 0.0
        denominator = jnp.where(active, frequency, 1.0)
        daughters = jnp.where(
            active[None, :],
            contract("ijk,jk->ij", plan.daughter_assignment, weighted)
            / denominator[None, :],
            jnp.eye(diameters.size, dtype=frequency.dtype),
        )
        volumes = plan.volumes
        defect = jnp.abs(volumes @ daughters - volumes) / volumes
        volume_residual = jnp.max(jnp.where(active, defect, 0.0))
        finite = (
            jnp.all(jnp.isfinite(partial))
            & jnp.all(jnp.isfinite(frequency))
            & jnp.all(jnp.isfinite(daughters))
        )
        evidence = _evidence(
            plan,
            liquid,
            self.weber_support,
            jnp.max(jnp.where(breaking[:, None], scale, 0.0)),
            finite,
        )
        return BubbleBreakageKernelResult(
            frequency,
            daughters,
            partial,
            volume_residual,
            evidence,
            plan.plan_id,
            self.kernel_id,
        )


__all__ = [
    "BubbleBreakageKernelResult",
    "BubbleCoalescenceKernelResult",
    "BubbleKernelEvidence",
    "BubbleKernelStatus",
    "BubbleSectionalPlan",
    "LehrCoalescenceKernel",
    "LuoSvendsenBreakageKernel",
    "PrinceBlanchCoalescenceKernel",
    "TurbulentBubblyLiquid",
    "luo_svendsen_eddy_integral",
]
