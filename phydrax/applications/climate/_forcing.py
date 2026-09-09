#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


MYHRE_1998_SOURCE = "https://doi.org/10.1029/98GL01908"


def _overlap(methane: Array, nitrous_oxide: Array, /) -> Array:
    product = methane * nitrous_oxide
    return 0.47 * jnp.log1p(2.01e-5 * product**0.75 + 5.31e-15 * methane * product**1.52)


class ClimateForcingResult(StrictModule):
    components: Array
    total: Array
    successful: Array
    names: tuple[str, ...] = eqx.field(static=True)
    unit: str = eqx.field(static=True, default="W m^-2")


class Myhre1998Forcing(StrictModule):
    """Myhre et al. (1998) / IPCC TAR simplified adjusted radiative forcing.

    CO2 uses ppm, CH4/N2O use ppb. Methane overlap holds N2O at the
    reference concentration; N2O overlap holds CH4 at reference, exactly as
    the published separate-gas approximation. This is not ERF and omits
    methane oxidation/ozone/stratospheric-water effects and updated forcing
    fits. External channels must provide any such contributions explicitly.
    """

    external_names: tuple[str, ...] = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    source: str = eqx.field(static=True)
    forcing_id: str = eqx.field(static=True)

    def __init__(self, external_names: tuple[str, ...] = (), /):
        gas_names = (
            "CO2_logarithmic",
            "CH4_square_root",
            "CH4_N2O_overlap",
            "N2O_square_root",
            "N2O_CH4_overlap",
        )
        names = tuple(external_names)
        if (
            any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
            or set(names).intersection(gas_names)
        ):
            raise ValueError(
                "External forcing names must be nonempty, unique and distinct from physical gas channels."
            )
        self.external_names = names
        self.component_names = gas_names + names
        self.source = MYHRE_1998_SOURCE
        self.forcing_id = canonical_fingerprint(
            {
                "kind": "myhre-1998-tar-forcing",
                "source": self.source,
                "components": self.component_names,
                "overlap_convention": "other-gas-at-background",
                "unit": "W m^-2",
            }
        )

    def evaluate(
        self,
        concentrations: Array,
        background: Array,
        external: Array,
        gas_forcing: Array,
        roles: tuple[str, ...],
        /,
    ) -> ClimateForcingResult:
        if (
            concentrations.shape != (3,)
            or background.shape != (3,)
            or gas_forcing.shape != (3,)
            or external.shape != (len(self.external_names),)
        ):
            raise ValueError(
                "Forcing inputs must match three admitted gases and the named external channels."
            )
        if len(roles) != 3 or any(
            role not in ("emissions", "concentration", "forcing") for role in roles
        ):
            raise ValueError("Each gas requires exactly one supported driver role.")
        forcing_driven = jnp.asarray(tuple(role == "forcing" for role in roles))
        concentrations = jnp.where(forcing_driven, background, concentrations)
        gas_forcing = jnp.where(forcing_driven, gas_forcing, 0.0)
        c, m, n = concentrations
        c0, m0, n0 = background
        baseline_overlap = _overlap(m0, n0)
        co2 = 5.35 * jnp.log(c / c0)
        methane = 0.036 * (jnp.sqrt(m) - jnp.sqrt(m0))
        methane_overlap = -(_overlap(m, n0) - baseline_overlap)
        nitrous = 0.12 * (jnp.sqrt(n) - jnp.sqrt(n0))
        nitrous_overlap = -(_overlap(m0, n) - baseline_overlap)
        # A prescribed gas forcing replaces both that gas's direct and overlap
        # term. External channels are additive, never implicitly gas emissions.
        if roles[0] == "forcing":
            co2 = gas_forcing[0]
        if roles[1] == "forcing":
            methane, methane_overlap = gas_forcing[1], jnp.zeros_like(methane_overlap)
        if roles[2] == "forcing":
            nitrous, nitrous_overlap = gas_forcing[2], jnp.zeros_like(nitrous_overlap)
        components = jnp.concatenate(
            (
                jnp.stack((co2, methane, methane_overlap, nitrous, nitrous_overlap)),
                external,
            )
        )
        valid = (
            jnp.all(jnp.isfinite(components))
            & jnp.all(concentrations > 0.0)
            & jnp.all(background > 0.0)
        )
        names = list(self.component_names)
        for gas, index, role in zip(("CO2", "CH4", "N2O"), (0, 1, 3), roles, strict=True):
            if role == "forcing":
                names[index] = f"{gas}_prescribed"
        return ClimateForcingResult(components, jnp.sum(components), valid, tuple(names))


__all__ = ["ClimateForcingResult", "MYHRE_1998_SOURCE", "Myhre1998Forcing"]
