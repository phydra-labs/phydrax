#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cross-application promotion portfolios derived from the canonical catalog."""

from __future__ import annotations

from dataclasses import dataclass

from .._fingerprint import canonical_fingerprint
from ._catalog import CapabilityCatalog, CapabilityDisposition


_PORTFOLIO_PATTERNS = {
    "flow-multiphysics": (
        "aerothermodynamics",
        "compressible_flow",
        "free_boundary",
        "hydrodynamics",
        "incompressible_flow",
        "phase_field",
        "radiation_transport",
        "reacting_flow",
        "thermofluids",
        "two_phase_flow",
        "vortex_flow",
        "phase-field",
        "radiation-transport",
        "reacting-flow",
    ),
    "atomistic-chemistry-polymer": (
        "atomistic",
        "chemistry",
        "polymer",
        "soft-matter",
    ),
    "battery-energy": (
        "battery",
        "building_energy",
        "electrical_machines",
        "energy_planning",
        "power",
    ),
    "nuclear-reactor-tokamak": ("nuclear", "reactor_physics", "tokamak"),
    "earth-atmosphere-ocean-climate": (
        "atmosphere",
        "climate",
        "geophysics",
        "ocean",
        "porous_media",
    ),
    "gr-astrophysics-dark-qft": (
        "astrophysics",
        "compact_objects",
        "cosmology",
        "curved_spacetime_qft",
        "dark",
        "lattice_field",
        "numerical_relativity",
        "qft",
    ),
    "hep-provider-composition": (
        "hep",
        "particle_physics",
        "collider_physics",
        "particle-spectrum",
    ),
    "biology-medical": (
        "cardiovascular",
        "cellular_mechanics",
        "electrophysiology",
        "magnetic_resonance",
        "medical",
        "neuro",
        "nucleic_acid",
        "protein_folding",
        "radiation_biophysics",
        "skeletal_muscle",
        "systems_biology",
    ),
    "finance": ("finance",),
    "robotics": ("robotics",),
}


@dataclass(frozen=True, slots=True)
class ApplicationPromotionPortfolio:
    name: str
    capability_ids: tuple[str, ...]
    profile_ids: tuple[str, ...]
    blockers: tuple[str, ...]
    ready_for_release_review: bool
    portfolio_id: str
    def to_record(self) -> dict[str, object]:
        return {
            "kind": "application-promotion-portfolio",
            "name": self.name,
            "capability_ids": list(self.capability_ids),
            "profile_ids": list(self.profile_ids),
            "blockers": list(self.blockers),
            "ready_for_release_review": self.ready_for_release_review,
            "portfolio_id": self.portfolio_id,
        }



def application_promotion_portfolios(
    catalog: CapabilityCatalog | None = None,
    /,
) -> tuple[ApplicationPromotionPortfolio, ...]:
    """Derive honest application portfolios without creating a release claim."""

    if catalog is None:
        from ._builtin_catalog import builtin_capability_catalog

        catalog = builtin_capability_catalog()
    if not isinstance(catalog, CapabilityCatalog):
        raise TypeError("catalog must be CapabilityCatalog.")
    portfolios = []
    for name, patterns in _PORTFOLIO_PATTERNS.items():
        declarations = tuple(
            declaration
            for declaration in catalog.declarations
            if any(
                pattern in declaration.capability
                or pattern in declaration.owner
                for pattern in patterns
            )
        )
        capability_ids = tuple(sorted(value.capability for value in declarations))
        profile_ids = tuple(
            sorted(
                profile.profile_id
                for declaration in declarations
                for profile in declaration.profiles
            )
        )
        blockers = []
        if not declarations:
            blockers.append("empty-portfolio")
        if any(
            value.disposition is CapabilityDisposition.RESEARCH
            for value in declarations
        ):
            blockers.append("research-capabilities-present")
        if any(
            value.disposition is CapabilityDisposition.CANDIDATE and not value.evidence
            for value in declarations
        ):
            blockers.append("candidate-evidence-incomplete")
        if not profile_ids:
            blockers.append("no-exact-candidate-profiles")
        if any(
            value.disposition is not CapabilityDisposition.RELEASED
            for value in declarations
        ):
            blockers.append("release-authorization-absent")
        blockers_ = tuple(sorted(set(blockers)))
        payload = {
            "kind": "application-promotion-portfolio",
            "name": name,
            "capabilities": list(capability_ids),
            "profiles": list(profile_ids),
            "blockers": list(blockers_),
        }
        portfolios.append(
            ApplicationPromotionPortfolio(
                name,
                capability_ids,
                profile_ids,
                blockers_,
                not blockers_,
                canonical_fingerprint(payload),
            )
        )
    return tuple(portfolios)


__all__ = ["ApplicationPromotionPortfolio", "application_promotion_portfolios"]
