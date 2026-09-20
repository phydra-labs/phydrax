#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_UIB2016_SOURCE = "doi:10.1103/PhysRevD.95.064024:UIB2016v2"


class AlignedBinaryRemnantStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_INPUT = 1
    OUTSIDE_CALIBRATION = 2
    UNPHYSICAL_FIT = 3


class AlignedBinaryRemnantPlan(StrictModule, NonTrainableState):
    """Qualified domain for the UIB2016v2 nonprecessing BBH remnant fit."""

    maximum_mass_ratio: float = eqx.field(static=True)
    maximum_spin_magnitude: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_mass_ratio: float = 8.0,
        maximum_spin_magnitude: float = 0.8,
    ):
        ratio = float(maximum_mass_ratio)
        spin = float(maximum_spin_magnitude)
        if not math.isfinite(ratio) or ratio < 1.0 or ratio > 8.0:
            raise ValueError(
                "maximum_mass_ratio must lie in the qualified interval [1, 8]."
            )
        if not math.isfinite(spin) or spin <= 0.0 or spin > 0.8:
            raise ValueError(
                "maximum_spin_magnitude must lie in the qualified interval (0, 0.8]."
            )
        self.maximum_mass_ratio = ratio
        self.maximum_spin_magnitude = spin
        self.source_id = _UIB2016_SOURCE
        self.plan_id = canonical_fingerprint(
            {
                "kind": "aligned-binary-remnant-plan",
                "fit": _UIB2016_SOURCE,
                "maximum_mass_ratio": ratio,
                "maximum_spin_magnitude": spin,
                "mass_unit": "same-as-input",
                "spin_convention": "dimensionless-aligned-primary-secondary",
            }
        )

    def evaluate(
        self,
        primary_mass: ArrayLike,
        secondary_mass: ArrayLike,
        primary_spin: ArrayLike,
        secondary_spin: ArrayLike,
        /,
    ) -> AlignedBinaryRemnantResult:
        return evaluate_aligned_binary_remnant(
            self,
            primary_mass,
            secondary_mass,
            primary_spin,
            secondary_spin,
        )


class AlignedBinaryRemnantResult(StrictModule):
    symmetric_mass_ratio: Array
    final_mass: Array
    final_mass_fraction: Array
    final_dimensionless_spin: Array
    radiated_energy_fraction: Array
    finite: Array
    physically_valid: Array
    derivative_valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(AlignedBinaryRemnantStatus.SUCCESS)


def _uib2016_final_spin(
    eta: Array,
    primary_fraction: Array,
    secondary_fraction: Array,
    primary_spin: Array,
    secondary_spin: Array,
    /,
) -> Array:
    eta2 = eta * eta
    eta3 = eta2 * eta
    delta = primary_fraction - secondary_fraction
    spin_difference = primary_spin - secondary_spin
    effective_spin = (
        primary_fraction**2 * primary_spin + secondary_fraction**2 * secondary_spin
    ) / (primary_fraction**2 + secondary_fraction**2)
    effective_spin2 = effective_spin * effective_spin
    effective_spin3 = effective_spin2 * effective_spin
    denominator = (
        1.0
        + (-0.9142232693081653 + 2.3191363426522633 * eta - 9.710576749140989 * eta3)
        * effective_spin
    )
    equal_spin = (primary_fraction**2 + secondary_fraction**2) * effective_spin + (
        (
            -0.8561951310209386 * eta
            - 0.09939065676370885 * eta2
            + 1.668810429851045 * eta3
        )
        * effective_spin
        + (
            0.5881660363307388 * eta
            - 2.149269067519131 * eta2
            + 3.4768263932898678 * eta3
        )
        * effective_spin2
        + (
            0.142443244743048 * eta
            - 0.9598353840147513 * eta2
            + 1.9595643107593743 * eta3
        )
        * effective_spin3
    ) / denominator
    nonspinning = (
        3.4641016151377544 * eta + 20.0830030082033 * eta2 - 12.333573402277912 * eta3
    ) / (1.0 + 7.2388440419467335 * eta)
    unequal_spin = (
        0.3223660562764661
        * spin_difference
        * delta
        * (1.0 + 9.332575956437443 * eta)
        * eta2
        - 0.059808322561702126 * spin_difference**2 * eta3
        + 2.3170397514509933
        * spin_difference
        * delta
        * (1.0 - 3.2624649875884852 * eta)
        * eta3
        * effective_spin
    )
    return nonspinning + equal_spin + unequal_spin


def _uib2016_radiated_energy(
    eta: Array,
    primary_fraction: Array,
    secondary_fraction: Array,
    primary_spin: Array,
    secondary_spin: Array,
    /,
) -> Array:
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    delta = primary_fraction - secondary_fraction
    spin_difference = primary_spin - secondary_spin
    effective_spin = (
        primary_fraction**2 * primary_spin + secondary_fraction**2 * secondary_spin
    ) / (primary_fraction**2 + secondary_fraction**2)
    effective_spin2 = effective_spin * effective_spin
    effective_spin3 = effective_spin2 * effective_spin
    nonspinning = (
        0.057190958417936644 * eta
        + 0.5609904135313374 * eta2
        - 0.84667563764404 * eta3
        + 3.145145224278187 * eta4
    )
    spin_numerator = (
        1.0
        + (-0.13084389181783257 - 1.1387311580238488 * eta + 5.49074464410971 * eta2)
        * effective_spin
        + (-0.17762802148331427 + 2.176667900182948 * eta2) * effective_spin2
        + (-0.6320191645391563 + 4.952698546796005 * eta - 10.023747993978121 * eta2)
        * effective_spin3
    )
    spin_denominator = (
        1.0
        + (-0.9919475346968611 + 0.367620218664352 * eta + 4.274567337924067 * eta2)
        * effective_spin
    )
    unequal_spin = (
        -0.09803730445895877
        * spin_difference
        * delta
        * (1.0 - 3.2283713377939134 * eta)
        * eta2
        + 0.01118530335431078 * spin_difference**2 * eta3
        - 0.01978238971523653
        * spin_difference
        * delta
        * (1.0 - 4.91667749015812 * eta)
        * eta
        * effective_spin
    )
    return nonspinning * spin_numerator / spin_denominator + unequal_spin


def evaluate_aligned_binary_remnant(
    plan: AlignedBinaryRemnantPlan,
    primary_mass: ArrayLike,
    secondary_mass: ArrayLike,
    primary_spin: ArrayLike,
    secondary_spin: ArrayLike,
    /,
) -> AlignedBinaryRemnantResult:
    """Evaluate final mass and spin for a quasicircular nonprecessing BBH."""

    if not isinstance(plan, AlignedBinaryRemnantPlan):
        raise TypeError("plan must be AlignedBinaryRemnantPlan.")
    raw_inputs = tuple(
        jnp.asarray(value)
        for value in (primary_mass, secondary_mass, primary_spin, secondary_spin)
    )
    if any(
        value.shape != ()
        or jnp.iscomplexobj(value)
        or jnp.issubdtype(value.dtype, jnp.bool_)
        for value in raw_inputs
    ):
        raise TypeError("Remnant masses and spins must be real numeric scalars.")
    dtype = jnp.result_type(*raw_inputs, 1.0)
    primary, secondary, spin_primary, spin_secondary = tuple(
        value.astype(dtype) for value in raw_inputs
    )
    inputs = jnp.stack((primary, secondary, spin_primary, spin_secondary))
    finite_inputs = jnp.all(jnp.isfinite(inputs))
    mass_support = (primary >= secondary) & (secondary > 0.0)
    safe_primary = jnp.where(primary > 0.0, primary, 1.0)
    safe_secondary = jnp.where(secondary > 0.0, secondary, 1.0)
    total_mass = safe_primary + safe_secondary
    primary_fraction = safe_primary / total_mass
    secondary_fraction = safe_secondary / total_mass
    eta = primary_fraction * secondary_fraction
    mass_ratio = safe_primary / safe_secondary
    spin_support = (jnp.abs(spin_primary) <= plan.maximum_spin_magnitude) & (
        jnp.abs(spin_secondary) <= plan.maximum_spin_magnitude
    )
    physical_spin_support = (jnp.abs(spin_primary) <= 1.0) & (
        jnp.abs(spin_secondary) <= 1.0
    )
    calibration_support = (
        finite_inputs
        & mass_support
        & (mass_ratio <= plan.maximum_mass_ratio)
        & spin_support
    )
    final_spin = _uib2016_final_spin(
        eta,
        primary_fraction,
        secondary_fraction,
        spin_primary,
        spin_secondary,
    )
    radiated = _uib2016_radiated_energy(
        eta,
        primary_fraction,
        secondary_fraction,
        spin_primary,
        spin_secondary,
    )
    final_fraction = 1.0 - radiated
    final_mass = total_mass * final_fraction
    finite = finite_inputs & jnp.all(
        jnp.isfinite(jnp.stack((eta, final_spin, radiated, final_fraction, final_mass)))
    )
    physically_valid = (
        finite
        & mass_support
        & physical_spin_support
        & (eta > 0.0)
        & (eta <= 0.25)
        & (jnp.abs(final_spin) < 1.0)
        & (radiated >= 0.0)
        & (radiated < 1.0)
        & (final_mass > 0.0)
    )
    successful = calibration_support & physically_valid
    derivative_valid = (
        successful
        & (primary > secondary)
        & (mass_ratio < plan.maximum_mass_ratio)
        & (jnp.abs(spin_primary) < plan.maximum_spin_magnitude)
        & (jnp.abs(spin_secondary) < plan.maximum_spin_magnitude)
    )
    status = jnp.where(
        successful,
        int(AlignedBinaryRemnantStatus.SUCCESS),
        jnp.where(
            ~finite_inputs,
            int(AlignedBinaryRemnantStatus.NONFINITE_INPUT),
            jnp.where(
                ~calibration_support,
                int(AlignedBinaryRemnantStatus.OUTSIDE_CALIBRATION),
                int(AlignedBinaryRemnantStatus.UNPHYSICAL_FIT),
            ),
        ),
    ).astype(jnp.int32)
    invalid = jnp.asarray(jnp.nan, dtype=final_mass.dtype)
    return AlignedBinaryRemnantResult(
        jnp.where(successful, eta, invalid),
        jnp.where(successful, final_mass, invalid),
        jnp.where(successful, final_fraction, invalid),
        jnp.where(successful, final_spin, invalid),
        jnp.where(successful, radiated, invalid),
        finite,
        physically_valid,
        derivative_valid,
        status,
        plan.plan_id,
        plan.source_id,
    )


__all__ = [
    "AlignedBinaryRemnantPlan",
    "AlignedBinaryRemnantResult",
    "AlignedBinaryRemnantStatus",
    "evaluate_aligned_binary_remnant",
]
