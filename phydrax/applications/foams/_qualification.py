#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned primary-source references and closed-form foam geometries.

Every reference records its convention and source. Closed-form geometries
(standard double bubble, catenoid branch) are exact continuum solutions used to
qualify discrete equilibria; they are not solver fallbacks.

Catenoid convention (Goldstein, Pesci, Raufaste and Shemilt, *Geometry of
catenoidal soap film collapse induced by boundary deformation*, Phys. Rev. E
104, 035105, 2021): coaxial rings of radius ``R`` at ``z = +/- d``, the film
``r = a cosh(z / a)``, scaled ``alpha = a / R`` and ``D = d / R`` with
``alpha cosh(D / alpha) = 1``. The two branches merge at ``x_c tanh x_c = 1``,
``D_c = x_c / cosh x_c = 0.6627...`` and ``alpha_c = 1 / cosh x_c = 0.5524...``;
the scaled area ``A = area / (2 pi R^2) = alpha D + alpha^2 sinh(2 D / alpha) / 2``
equals ``1.199...`` at ``D_c``, and the two-disc (Goldschmidt) solution has less
area than the stable catenoid for ``D > D* = 0.528``. In ring separation over
ring diameter, the limit reads ``2 d / (2 R) = 0.6627``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import final

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier, positive_finite_float
from ...nonlinear import Brent, NonlinearTermination, scalar_root, ScalarRootProblem


_GOLDSTEIN_2021 = (
    "R. E. Goldstein, A. I. Pesci, C. Raufaste, J. D. Shemilt, Phys. Rev. E 104, "
    "035105 (2021)"
)
_KUSNER_SULLIVAN_1996 = (
    "R. Kusner, J. M. Sullivan, Forma 11, 233-242 (1996); D. Weaire, R. Phelan, "
    "Phil. Mag. Lett. 69, 107-110 (1994)"
)


@final
class FoamReferenceValue(StrictModule, NonTrainableState):
    """One pinned reference value with its convention and provenance."""

    name: str = eqx.field(static=True)
    value: float = eqx.field(static=True)
    uncertainty: float = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    source: str = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        value: float,
        uncertainty: float,
        convention: str,
        source: str,
        /,
    ) -> None:
        name_ = canonical_identifier(name, "name")
        value_ = float(value)
        spread = float(uncertainty)
        if not math.isfinite(value_) or not math.isfinite(spread) or spread < 0.0:
            raise ValueError("Reference value and uncertainty must be finite.")
        self.name = name_
        self.value = value_
        self.uncertainty = spread
        self.convention = canonical_identifier(convention, "convention")
        self.source = canonical_identifier(source, "source")
        self.reference_id = canonical_fingerprint(
            {
                "kind": "foam-reference-value",
                "name": name_,
                "value": value_,
                "uncertainty": spread,
                "convention": self.convention,
                "source": self.source,
            }
        )


def _solve_scalar(
    function: Callable[[Array, object], Array], lower: float, upper: float, /
) -> float:
    problem = ScalarRootProblem(
        function, bracket=(lower, upper), problem_id="foam-reference-root"
    )
    result = scalar_root(
        problem,
        method=Brent(),
        termination=NonlinearTermination(
            absolute_residual=1.0e-15,
            relative_residual=0.0,
            absolute_step=1.0e-15,
            relative_step=1.0e-15,
        ),
    )
    if not bool(result.successful):
        raise RuntimeError("Reference root solve did not converge.")
    return float(result.root)


def catenoid_critical_parameters() -> tuple[float, float, float]:
    """``(x_c, D_c, alpha_c)`` from ``x_c tanh x_c = 1`` (exact to roundoff)."""
    critical = _solve_scalar(lambda x, _: x * jnp.tanh(x) - 1.0, 0.5, 2.0)
    return critical, critical / math.cosh(critical), 1.0 / math.cosh(critical)


def catenoid_stable_neck_ratio(half_separation_ratio: float, /) -> float:
    """Stable-branch ``alpha = a / R`` for ``D = d / R < D_c``."""
    ratio = positive_finite_float(half_separation_ratio, "half_separation_ratio")
    _, critical_ratio, critical_alpha = catenoid_critical_parameters()
    if ratio >= critical_ratio:
        raise ValueError(
            f"No catenoid exists for D = {ratio} >= D_c = {critical_ratio:.10f}."
        )

    def residual(alpha: Array, _: object) -> Array:
        return alpha * jnp.cosh(ratio / alpha) - 1.0

    return _solve_scalar(residual, critical_alpha, 1.0)


def catenoid_area_ratio(half_separation_ratio: float, neck_ratio: float, /) -> float:
    """``area / (2 pi R^2) = alpha D + alpha^2 sinh(2 D / alpha) / 2``."""
    ratio = positive_finite_float(half_separation_ratio, "half_separation_ratio")
    alpha = positive_finite_float(neck_ratio, "neck_ratio")
    return alpha * ratio + 0.5 * alpha * alpha * math.sinh(2.0 * ratio / alpha)


@final
class StandardDoubleBubble(StrictModule, NonTrainableState):
    """Closed-form equal-tension double bubble with outer radii ``R1``, ``R2``.

    Three spherical caps meet at 120 degrees on a circle; the separating cap
    has radius ``R1 R2 / |R1 - R2|`` so ``1 / R_P = 1 / R2 - 1 / R1`` (infinite
    for equal bubbles). With effective film tension ``gamma`` (``2 sigma`` for a
    soap film) the pressures are ``2 gamma / R_i`` and the energy is
    ``gamma`` times the total film area.
    """

    radius_first: float = eqx.field(static=True)
    radius_second: float = eqx.field(static=True)
    interface_radius: float = eqx.field(static=True)
    ring_radius: float = eqx.field(static=True)
    center_distance: float = eqx.field(static=True)
    volume_first: float = eqx.field(static=True)
    volume_second: float = eqx.field(static=True)
    film_area: float = eqx.field(static=True)
    effective_tension: float = eqx.field(static=True)

    def __init__(
        self, radius_first: float, radius_second: float, effective_tension: float, /
    ) -> None:
        first = positive_finite_float(radius_first, "radius_first")
        second = positive_finite_float(radius_second, "radius_second")
        tension = positive_finite_float(effective_tension, "effective_tension")
        distance = math.sqrt(first * first + second * second - first * second)
        offset_first = (distance * distance + first * first - second * second) / (
            2.0 * distance
        )
        offset_second = distance - offset_first
        ring = math.sqrt(first * first - offset_first * offset_first)

        def cap(height: float, radius: float, /) -> float:
            return math.pi * height * height * (3.0 * radius - height) / 3.0

        volume_first = 4.0 * math.pi * first**3 / 3.0 - cap(first - offset_first, first)
        volume_second = 4.0 * math.pi * second**3 / 3.0 - cap(
            second - offset_second, second
        )
        area = 2.0 * math.pi * (first * (first + offset_first) + second * (second + offset_second))
        if math.isclose(first, second, rel_tol=1.0e-14):
            interface = math.inf
            area += math.pi * ring * ring
        else:
            interface = first * second / abs(first - second)
            height = interface - math.sqrt(interface * interface - ring * ring)
            lens = cap(height, interface)
            sign = 1.0 if first > second else -1.0
            volume_first -= sign * lens
            volume_second += sign * lens
            area += 2.0 * math.pi * interface * height
        self.radius_first = first
        self.radius_second = second
        self.interface_radius = interface
        self.ring_radius = ring
        self.center_distance = distance
        self.volume_first = volume_first
        self.volume_second = volume_second
        self.film_area = area
        self.effective_tension = tension

    @property
    def pressures(self) -> tuple[float, float]:
        """Laplace pressures ``2 gamma / R_i`` relative to the ambient."""
        return (
            2.0 * self.effective_tension / self.radius_first,
            2.0 * self.effective_tension / self.radius_second,
        )

    @property
    def energy(self) -> float:
        return self.effective_tension * self.film_area


def foam_reference_values() -> tuple[FoamReferenceValue, ...]:
    """Pinned references used by the foam qualification campaign."""
    _, critical_ratio, critical_alpha = catenoid_critical_parameters()
    flat_kelvin = (6.0 + 12.0 * math.sqrt(3.0)) / (8.0 * math.sqrt(2.0)) ** (2.0 / 3.0)
    return (
        FoamReferenceValue(
            "catenoid.critical-half-separation-ratio",
            critical_ratio,
            0.0,
            "D_c = d / R for rings of radius R at z = +/- d (x_c tanh x_c = 1)",
            _GOLDSTEIN_2021,
        ),
        FoamReferenceValue(
            "catenoid.critical-neck-ratio",
            critical_alpha,
            0.0,
            "alpha_c = a / R with r = a cosh(z / a)",
            _GOLDSTEIN_2021,
        ),
        FoamReferenceValue(
            "catenoid.critical-area-ratio",
            catenoid_area_ratio(critical_ratio, critical_alpha),
            0.0,
            "area / (2 pi R^2) of the critical catenoid",
            _GOLDSTEIN_2021,
        ),
        FoamReferenceValue(
            "catenoid.goldschmidt-crossover-ratio",
            0.528,
            0.0005,
            "D* = d / R above which two discs have less area than the stable catenoid",
            _GOLDSTEIN_2021,
        ),
        FoamReferenceValue(
            "kelvin.relaxed-cell-area",
            5.306,
            0.0005,
            "cell surface area over V^(2/3) per unit-volume cell of the relaxed "
            "periodic Kelvin foam (faces counted by both cells)",
            _KUSNER_SULLIVAN_1996,
        ),
        FoamReferenceValue(
            "weaire-phelan.relaxed-cell-area",
            5.288,
            0.0005,
            "cell surface area over V^(2/3) per unit-volume cell of the relaxed "
            "periodic Weaire-Phelan foam (faces counted by both cells)",
            _KUSNER_SULLIVAN_1996,
        ),
        FoamReferenceValue(
            "kelvin.flat-truncated-octahedron-area",
            flat_kelvin,
            0.0,
            "surface area over V^(2/3) of the flat-faced truncated octahedron",
            "exact polyhedral geometry",
        ),
    )


__all__ = [
    "FoamReferenceValue",
    "StandardDoubleBubble",
    "catenoid_area_ratio",
    "catenoid_critical_parameters",
    "catenoid_stable_neck_ratio",
    "foam_reference_values",
]
