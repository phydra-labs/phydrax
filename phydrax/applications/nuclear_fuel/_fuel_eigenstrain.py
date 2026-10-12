#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Irradiation eigenstrains of sintered UO2 fuel pellets.

``UO2Densification`` and ``FuelRelocation`` return linear strains and
``UO2SolidSwelling`` a volumetric strain. Burnup enters as
``burnup_mwd_per_kghm``; for UO2 the heavy metal is uranium, so it equals the
MWd/kgU and GWd/MTU of the FRAPCON sources.

Primary sources:

- Luscher, Geelhood, Porter, PNNL-19417 Rev. 2 (2015), Section 2.6 and
  Appendix A.1.6 (FUDENS).
- Geelhood et al., FRAPCON-4.0, PNNL-19418 Vol. 1 Rev. 2 (2015), Section 2.4
  (fuel deformation, relocation Eq. (2.141), GTRLOC; relocation recovery).
- Geelhood et al., MatLib-1.2.1, PNNL-35702 (2024), Sections 2.1.7-2.1.8.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...materials import PropertyValidityRange
from ...nonlinear import LocalRootPlan
from ._constants import SECONDS_PER_DAY
from ._material_math import zero_safe_log_power
from ._material_parameters import finite_parameter, float64_array, positive_parameter


_WATT_PER_KILOWATT = 1.0e3
_PERCENT = 100.0
# FUDENS (PNNL-19417 Rev. 2, Appendix A.1.6) evaluates densification only for
# bu >= 1728 MW s/kgU; 1728 / 86400 s/d = 0.02 MWd/kgU.
_FUDENS_ONSET_BURNUP = 1728.0 / SECONDS_PER_DAY
# FUDENS searches the burnup offset B in [0, 6] MWd/kgU and falls back when the
# maximum linear change reaches 3 - 4.27e-3 percent; the resintering density
# fraction 3 (3 - 4.27e-3) / 100 bounds the source's solvable domain.
_FUDENS_MAXIMUM_RESINTERING_FRACTION = 3.0 * (3.0 - 4.27e-3) / _PERCENT
_DENSIFICATION_ROOT = LocalRootPlan(
    maximum_steps=80, tolerance=1.0e-13, plan_id="uo2-densification-burnup-offset"
)


@final
class FuelDensificationUpdate(StrictModule):
    """Candidate FUDENS strain with its root diagnostics; negative is shrinkage."""

    linear_strain: Array
    linear_strain_increment: Array
    root_converged: Array
    root_residual: Array


@final
class UO2Densification(StrictModule):
    """In-reactor densification linear strain of UO2 (FRAPCON-4.0 FUDENS).

    PNNL-19417 Rev. 2 Eqs. (2.29) and (2.32), with the resintering-test route
    preferred by Regulatory Guide 1.126::

        (dL/L)_m = 100 drho_RS / (3 rho_0)                         [%]
        dL/L = -(dL/L)_m + 1.0 exp(-3 (Bu + B)) + 2.0 exp(-35 (Bu + B))  [%]

    ``Bu`` in MWd/kgU, ``drho_RS`` the density increase of a 1973 K, 24 h
    resintering test and ``rho_0`` the as-fabricated density [kg/m^3]. The
    offset ``B`` makes ``dL/L = 0`` at zero burnup; with ``y = exp(-3 B)`` it
    solves ``y + 2 y^(35/3) = (dL/L)_m``, a convex increasing scalar root
    solved by a local Newton iteration to convergence (FUDENS stops its Newton
    iteration for ``B`` at a 2e-4 % relative change). FUDENS returns zero
    below 0.02 MWd/kgU. The resintering route of FRAPCON-4.0 has no temperature
    dependence (MATPRO's 1000 K split of Eqs. (2.27)-(2.28) is not used by
    FRAPCON-4.0); the sintering-temperature fallback of Eqs. (2.30)-(2.31) for
    fuel without a resintering test is not provided. The strain is linear and
    isotropic, decreases monotonically with burnup, and is bounded by
    ``-drho_RS / (3 rho_0)``, so FUDENS's rule of never reducing the densified
    state is satisfied by construction.
    """

    slow_amplitude_percent: Array
    slow_rate: Array
    fast_amplitude_percent: Array
    fast_rate: Array
    onset_burnup: Array
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        slow_amplitude_percent: ArrayLike = 1.0,
        slow_rate: ArrayLike = 3.0,
        fast_amplitude_percent: ArrayLike = 2.0,
        fast_rate: ArrayLike = 35.0,
        onset_burnup: ArrayLike = _FUDENS_ONSET_BURNUP,
    ) -> None:
        parameters = {
            "slow_amplitude_percent": positive_parameter(
                slow_amplitude_percent, "slow_amplitude_percent"
            ),
            "slow_rate": positive_parameter(slow_rate, "slow_rate"),
            "fast_amplitude_percent": positive_parameter(
                fast_amplitude_percent, "fast_amplitude_percent"
            ),
            "fast_rate": positive_parameter(fast_rate, "fast_rate"),
            "onset_burnup": finite_parameter(onset_burnup, "onset_burnup"),
        }
        if parameters["fast_rate"] <= parameters["slow_rate"]:
            raise ValueError("fast_rate must exceed slow_rate.")
        self.slow_amplitude_percent = jnp.asarray(parameters["slow_amplitude_percent"])
        self.slow_rate = jnp.asarray(parameters["slow_rate"])
        self.fast_amplitude_percent = jnp.asarray(parameters["fast_amplitude_percent"])
        self.fast_rate = jnp.asarray(parameters["fast_rate"])
        self.onset_burnup = jnp.asarray(parameters["onset_burnup"])
        # The source states applicability only to pore structures like those of
        # Freshley et al. (1976) (PNNL-19417 Rev. 2 Section 2.6.2); the validity
        # range is the domain where FUDENS finds its burnup offset.
        self.validity = PropertyValidityRange(
            {
                "resintering_density_fraction": (
                    0.0,
                    _FUDENS_MAXIMUM_RESINTERING_FRACTION,
                    "1",
                )
            },
            source_id="pnnl-19417-rev2:appendix-a.1.6-fudens",
        )
        self.source_id = "pnnl-19417-rev2:eq-2.29-2.32"
        self.law_id = canonical_fingerprint(
            {
                "kind": "uo2-densification-fudens-resintering",
                "parameters": parameters,
                "source": self.source_id,
            }
        )

    def maximum_linear_strain(
        self, resintering_density_change: ArrayLike, as_fabricated_density: ArrayLike, /
    ) -> Array:
        """Asymptotic linear strain ``-drho_RS / (3 rho_0)`` [-]."""

        change, density = jnp.broadcast_arrays(
            float64_array(resintering_density_change),
            float64_array(as_fabricated_density),
        )
        return -change / (3.0 * density)

    def linear_strain(
        self,
        burnup_mwd_per_kghm: ArrayLike,
        resintering_density_change: ArrayLike,
        as_fabricated_density: ArrayLike,
        /,
    ) -> Array:
        """Densification linear strain [-] (negative) at burnup [MWd/kgHM].

        ``resintering_density_change`` and ``as_fabricated_density`` are in
        kg/m^3.
        """

        strain, _, _ = self._strain_with_diagnostics(
            burnup_mwd_per_kghm, resintering_density_change, as_fabricated_density
        )
        return strain

    def _strain_with_diagnostics(
        self,
        burnup_mwd_per_kghm: ArrayLike,
        resintering_density_change: ArrayLike,
        as_fabricated_density: ArrayLike,
        /,
    ) -> tuple[Array, Array, Array]:
        burnup, change, density = jnp.broadcast_arrays(
            float64_array(burnup_mwd_per_kghm),
            float64_array(resintering_density_change),
            float64_array(as_fabricated_density),
        )
        maximum_percent = -_PERCENT * self.maximum_linear_strain(change, density)
        exponent = self.fast_rate / self.slow_rate

        def solve_offset(target_percent: Array) -> tuple[Array, Array, Array]:
            def residual(y: Array) -> Array:
                return (
                    self.slow_amplitude_percent * y
                    + self.fast_amplitude_percent * zero_safe_log_power(y, exponent)
                    - target_percent
                )

            # Newton from the right of the root of a convex increasing function
            # converges monotonically: y0 = target / a1 is right of the root, and
            # so is y0 = 1 whenever the root lies in (0, 1].
            scaled = target_percent / self.slow_amplitude_percent
            initial = jnp.where(
                target_percent
                < self.slow_amplitude_percent + self.fast_amplitude_percent,
                jnp.minimum(scaled, 1.0),
                scaled,
            )
            root, diagnostics = _DENSIFICATION_ROOT.solve_with_diagnostics(
                residual, initial
            )
            return root, diagnostics.converged, diagnostics.residual

        roots, converged, residual = jax.vmap(solve_offset)(maximum_percent.reshape(-1))
        shape = maximum_percent.shape
        y = roots.reshape(shape)
        percent = (
            -maximum_percent
            + self.slow_amplitude_percent * y * jnp.exp(-self.slow_rate * burnup)
            + self.fast_amplitude_percent
            * zero_safe_log_power(y, exponent)
            * jnp.exp(-self.fast_rate * burnup)
        )
        below_onset = burnup < self.onset_burnup
        strain = jnp.where(below_onset, 0.0, percent / _PERCENT)
        # Below the onset burnup the strain is zero whatever the offset root.
        return strain, converged.reshape(shape) | below_onset, residual.reshape(shape)

    def update(
        self,
        burnup_mwd_per_kghm: ArrayLike,
        resintering_density_change: ArrayLike,
        as_fabricated_density: ArrayLike,
        accepted_linear_strain: ArrayLike,
        /,
    ) -> FuelDensificationUpdate:
        """FUDENS candidate update that never reverses accepted densification.

        Source Appendix A.1.6 returns only the additional shrinkage when its
        magnitude exceeds ``prvden``, so even a lower candidate burnup cannot
        reverse the accepted shrinkage. The root convergence and residual are
        returned for the step acceptance decision; below the onset burnup the
        root does not enter the strain, so the update reports converged.
        """

        predicted, converged, residual = self._strain_with_diagnostics(
            burnup_mwd_per_kghm, resintering_density_change, as_fabricated_density
        )
        previous = float64_array(accepted_linear_strain)
        candidate = jnp.minimum(previous, predicted)
        return FuelDensificationUpdate(
            linear_strain=candidate,
            linear_strain_increment=candidate - previous,
            root_converged=converged,
            root_residual=residual,
        )


@final
class UO2SolidSwelling(StrictModule):
    """Solid fission product swelling volumetric strain of UO2 (FRAPCON-4.0).

    FRAPCON-4.0 (PNNL-19418 Vol. 1 Rev. 2, Section 2.4, "Radial Deformation")
    and MatLib-1.2.1 Eq. (2-17): no swelling up to 6 GWd/MTU (swelling into
    as-fabricated porosity), then 0.062 % dV/V per GWd/MTU up to 80 GWd/MTU and
    0.086 % dV/V per GWd/MTU beyond::

        dV/V = 0.00062 (min(Bu, 80) - 6)^+ + 0.00086 (Bu - 80)^+

    Athermal; the strain is volumetric and piecewise linear (kinks at the
    onset and transition burnups).
    """

    low_burnup_rate: Array
    high_burnup_rate: Array
    onset_burnup: Array
    transition_burnup: Array
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        low_burnup_rate: ArrayLike = 6.2e-4,
        high_burnup_rate: ArrayLike = 8.6e-4,
        onset_burnup: ArrayLike = 6.0,
        transition_burnup: ArrayLike = 80.0,
    ) -> None:
        parameters = {
            "low_burnup_rate": finite_parameter(low_burnup_rate, "low_burnup_rate"),
            "high_burnup_rate": finite_parameter(high_burnup_rate, "high_burnup_rate"),
            "onset_burnup": finite_parameter(onset_burnup, "onset_burnup"),
            "transition_burnup": finite_parameter(transition_burnup, "transition_burnup"),
        }
        if not parameters["onset_burnup"] < parameters["transition_burnup"]:
            raise ValueError("onset_burnup must be below transition_burnup.")
        self.low_burnup_rate = jnp.asarray(parameters["low_burnup_rate"])
        self.high_burnup_rate = jnp.asarray(parameters["high_burnup_rate"])
        self.onset_burnup = jnp.asarray(parameters["onset_burnup"])
        self.transition_burnup = jnp.asarray(parameters["transition_burnup"])
        # MatLib-1.2.1 Section 2.1.8.3: 0-100 GWd/MTU, 90-98 %TD (the model has
        # no density dependence, so only the burnup range is checked).
        self.validity = PropertyValidityRange(
            {"burnup_mwd_per_kghm": (0.0, 100.0, "MWd/kgHM")},
            source_id="pnnl-35702:2.1.8.3",
        )
        self.source_id = "pnnl-19418-vol1-rev2:2.4:pnnl-35702-eq-2-17"
        self.law_id = canonical_fingerprint(
            {
                "kind": "uo2-solid-swelling-frapcon",
                "parameters": parameters,
                "source": self.source_id,
            }
        )

    def volumetric_strain(self, burnup_mwd_per_kghm: ArrayLike, /) -> Array:
        """Solid swelling volumetric strain ``dV/V`` [-] at burnup [MWd/kgHM]."""

        burnup = float64_array(burnup_mwd_per_kghm)
        low = jnp.clip(
            burnup - self.onset_burnup, 0.0, self.transition_burnup - self.onset_burnup
        )
        high = jnp.maximum(burnup - self.transition_burnup, 0.0)
        return self.low_burnup_rate * low + self.high_burnup_rate * high


@final
class FuelRelocation(StrictModule):
    """Pellet fragment relocation of FRAPCON-4.0 (GTRLOC, FRAPCON-3.5 model).

    PNNL-19418 Vol. 1 Rev. 2, Eq. (2.141): the gap closure as a fraction of the
    as-fabricated pellet-cladding gap is ``0.055`` below 0.0937 GWd/MTU and::

        dG/G = 0.055 + min(reloc, reloc (0.5795 + 0.2447 ln Bu))

    above, with ``reloc = 0.345`` below 20 kW/m, ``0.345 + (P - 20)/200``
    between 20 and 40 kW/m, and ``0.445`` above (the report prints "kW/ft";
    Fig. 2.16 plots the three regimes at 15, 30 and 45 kW/m with plateaus 0.40,
    0.45 and 0.50, fixing the unit as kW/m). ``P`` is the local linear heat
    generation rate and ``Bu`` the local burnup.

    The relocation strain is the linear diametral strain
    ``dG/G * G_0 / D_0`` (pellet radius increase over pellet radius), with
    ``G_0`` the as-fabricated cold diametral gap and ``D_0`` the as-fabricated
    pellet diameter. FRAPCON uses the full relocation for the thermal and gas
    pressure gap, while the mechanical pellet keeps only the permanent part:
    ``recoverable_fraction`` (50 % by default) must be taken up by other fuel
    expansion and cladding creepdown before hard contact loads the cladding
    (Section 2.4, "Rigid Pellet Fuel Deformation in FRACAS-I"; Section 4
    developer option "relocation recovery fraction").
    """

    initial_closure: Array
    onset_burnup: Array
    low_power_closure: Array
    power_slope_per_kw_per_m: Array
    lower_power_kw_per_m: Array
    upper_power_kw_per_m: Array
    burnup_intercept: Array
    burnup_log_coefficient: Array
    recoverable_fraction: Array
    validity: PropertyValidityRange
    source_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        initial_closure: ArrayLike = 0.055,
        onset_burnup: ArrayLike = 0.0937,
        low_power_closure: ArrayLike = 0.345,
        power_slope_per_kw_per_m: ArrayLike = 1.0 / 200.0,
        lower_power_kw_per_m: ArrayLike = 20.0,
        upper_power_kw_per_m: ArrayLike = 40.0,
        burnup_intercept: ArrayLike = 0.5795,
        burnup_log_coefficient: ArrayLike = 0.2447,
        recoverable_fraction: ArrayLike = 0.5,
    ) -> None:
        parameters = {
            "initial_closure": finite_parameter(initial_closure, "initial_closure"),
            "onset_burnup": positive_parameter(onset_burnup, "onset_burnup"),
            "low_power_closure": finite_parameter(low_power_closure, "low_power_closure"),
            "power_slope_per_kw_per_m": finite_parameter(
                power_slope_per_kw_per_m, "power_slope_per_kw_per_m"
            ),
            "lower_power_kw_per_m": finite_parameter(
                lower_power_kw_per_m, "lower_power_kw_per_m"
            ),
            "upper_power_kw_per_m": finite_parameter(
                upper_power_kw_per_m, "upper_power_kw_per_m"
            ),
            "burnup_intercept": finite_parameter(burnup_intercept, "burnup_intercept"),
            "burnup_log_coefficient": finite_parameter(
                burnup_log_coefficient, "burnup_log_coefficient"
            ),
            "recoverable_fraction": finite_parameter(
                recoverable_fraction, "recoverable_fraction"
            ),
        }
        if not parameters["lower_power_kw_per_m"] < parameters["upper_power_kw_per_m"]:
            raise ValueError("lower_power_kw_per_m must be below upper_power_kw_per_m.")
        if not 0.0 <= parameters["recoverable_fraction"] <= 1.0:
            raise ValueError("recoverable_fraction must lie in [0, 1].")
        self.initial_closure = jnp.asarray(parameters["initial_closure"])
        self.onset_burnup = jnp.asarray(parameters["onset_burnup"])
        self.low_power_closure = jnp.asarray(parameters["low_power_closure"])
        self.power_slope_per_kw_per_m = jnp.asarray(
            parameters["power_slope_per_kw_per_m"]
        )
        self.lower_power_kw_per_m = jnp.asarray(parameters["lower_power_kw_per_m"])
        self.upper_power_kw_per_m = jnp.asarray(parameters["upper_power_kw_per_m"])
        self.burnup_intercept = jnp.asarray(parameters["burnup_intercept"])
        self.burnup_log_coefficient = jnp.asarray(parameters["burnup_log_coefficient"])
        self.recoverable_fraction = jnp.asarray(parameters["recoverable_fraction"])
        # FRAPCON-4.0 Section 1: validated up to a rod-average burnup of
        # 62 GWd/MTU. The report states no power bound for GTRLOC.
        self.validity = PropertyValidityRange(
            {"burnup_mwd_per_kghm": (0.0, 62.0, "MWd/kgHM")},
            source_id="pnnl-19418-vol1-rev2:1",
        )
        self.source_id = "pnnl-19418-vol1-rev2:eq-2.141:section-2.4"
        self.law_id = canonical_fingerprint(
            {
                "kind": "fuel-relocation-frapcon-gtrloc",
                "parameters": parameters,
                "source": self.source_id,
            }
        )

    def gap_closure_fraction(
        self, linear_heat_rate: ArrayLike, burnup_mwd_per_kghm: ArrayLike, /
    ) -> Array:
        """Relocation gap closure ``dG/G`` [-] at local LHGR [W/m] and burnup."""

        power, burnup = jnp.broadcast_arrays(
            float64_array(linear_heat_rate), float64_array(burnup_mwd_per_kghm)
        )
        power_kw_per_m = power / _WATT_PER_KILOWATT
        closure = self.low_power_closure + self.power_slope_per_kw_per_m * jnp.clip(
            power_kw_per_m - self.lower_power_kw_per_m,
            0.0,
            self.upper_power_kw_per_m - self.lower_power_kw_per_m,
        )
        relocating = burnup > self.onset_burnup
        safe_burnup = jnp.where(relocating, burnup, 1.0)
        burnup_factor = jnp.minimum(
            1.0,
            self.burnup_intercept + self.burnup_log_coefficient * jnp.log(safe_burnup),
        )
        return self.initial_closure + jnp.where(relocating, closure * burnup_factor, 0.0)

    def diametral_strain(
        self,
        linear_heat_rate: ArrayLike,
        burnup_mwd_per_kghm: ArrayLike,
        as_fabricated_diametral_gap: ArrayLike,
        as_fabricated_pellet_diameter: ArrayLike,
        /,
    ) -> Array:
        """Full relocation linear diametral strain [-] (thermal and gas gap)."""

        gap, diameter = jnp.broadcast_arrays(
            float64_array(as_fabricated_diametral_gap),
            float64_array(as_fabricated_pellet_diameter),
        )
        return self.gap_closure_fraction(linear_heat_rate, burnup_mwd_per_kghm) * (
            gap / diameter
        )

    def mechanical_diametral_strain(
        self,
        linear_heat_rate: ArrayLike,
        burnup_mwd_per_kghm: ArrayLike,
        as_fabricated_diametral_gap: ArrayLike,
        as_fabricated_pellet_diameter: ArrayLike,
        /,
    ) -> Array:
        """Unrecoverable fraction of the algebraic relocation strain [-].

        ``(1 - recoverable_fraction)`` of `diametral_strain`: the recoverable
        remainder closes without cladding load before hard contact. This is
        the source correlation evaluated at the given power and burnup, not a
        history update: the published GTRLOC equation specifies neither cold
        activation nor retention on power decrease, so the caller supplies the
        permanent-relocation history rule.
        """

        return (1.0 - self.recoverable_fraction) * self.diametral_strain(
            linear_heat_rate,
            burnup_mwd_per_kghm,
            as_fabricated_diametral_gap,
            as_fabricated_pellet_diameter,
        )


__all__ = [
    "FuelDensificationUpdate",
    "FuelRelocation",
    "UO2Densification",
    "UO2SolidSwelling",
]
