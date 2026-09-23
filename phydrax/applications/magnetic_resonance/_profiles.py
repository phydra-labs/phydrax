#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Separate exact NMR, EPR, and static-site muSR profile boundaries."""

from __future__ import annotations

import equinox as eqx

from ..._strict import StrictModule
from ._spin_system import (
    HyperfineCoupling,
    MagneticResonanceSpinSystem,
    prepare_spin_system,
    PreparedMagneticResonanceSystem,
)


class ExactSingleCrystalNMRProfile(StrictModule):
    """Exact small finite nuclear-spin single-crystal profile."""

    system: MagneticResonanceSpinSystem
    profile_id: str = eqx.field(static=True)

    def __init__(self, system: MagneticResonanceSpinSystem, /):
        if not isinstance(system, MagneticResonanceSpinSystem):
            raise TypeError("system must be a MagneticResonanceSpinSystem.")
        if any(site.isotope.particle_kind != "nucleus" for site in system.sites):
            raise ValueError("The NMR profile accepts nuclear spin sites only.")
        self.system = system
        self.profile_id = "magnetic-resonance:nmr:exact-small-single-crystal"

    def prepare(self, /) -> PreparedMagneticResonanceSystem:
        return prepare_spin_system(self.system)


class ExactSingleCrystalEPRProfile(StrictModule):
    """Exact small single-crystal electron-spin profile with full hyperfine terms."""

    system: MagneticResonanceSpinSystem
    profile_id: str = eqx.field(static=True)

    def __init__(self, system: MagneticResonanceSpinSystem, /):
        if not isinstance(system, MagneticResonanceSpinSystem):
            raise TypeError("system must be a MagneticResonanceSpinSystem.")
        electron_count = sum(
            site.isotope.particle_kind == "electron" for site in system.sites
        )
        if electron_count != 1:
            raise ValueError("The EPR profile requires exactly one electron spin site.")
        if any(
            site.isotope.particle_kind not in ("electron", "nucleus")
            for site in system.sites
        ):
            raise ValueError("The EPR profile accepts electron and nuclear sites only.")
        sites = {site.site_id: site for site in system.sites}
        for interaction in system.interactions:
            if isinstance(interaction, HyperfineCoupling):
                left = sites[interaction.site_a]
                right = sites[interaction.site_b]
                if (
                    left.isotope.particle_kind != "electron"
                    and right.isotope.particle_kind != "electron"
                ):
                    raise ValueError(
                        "EPR hyperfine terms must include the electron site."
                    )
        self.system = system
        self.profile_id = "magnetic-resonance:epr:exact-small-single-crystal"

    def prepare(self, /) -> PreparedMagneticResonanceSystem:
        return prepare_spin_system(self.system)


class ExactStaticSiteMuonSpinRotationProfile(StrictModule):
    """Exact static-site positive-muon spin-rotation profile."""

    system: MagneticResonanceSpinSystem
    profile_id: str = eqx.field(static=True)

    def __init__(self, system: MagneticResonanceSpinSystem, /):
        if not isinstance(system, MagneticResonanceSpinSystem):
            raise TypeError("system must be a MagneticResonanceSpinSystem.")
        muon_count = sum(
            site.isotope.particle_kind == "positive-muon" for site in system.sites
        )
        if muon_count != 1:
            raise ValueError(
                "The static-site muSR profile requires exactly one positive muon."
            )
        self.system = system
        self.profile_id = "magnetic-resonance:musr:exact-small-static-site"

    def prepare(self, /) -> PreparedMagneticResonanceSystem:
        return prepare_spin_system(self.system)


__all__ = [
    "ExactSingleCrystalEPRProfile",
    "ExactSingleCrystalNMRProfile",
    "ExactStaticSiteMuonSpinRotationProfile",
]
