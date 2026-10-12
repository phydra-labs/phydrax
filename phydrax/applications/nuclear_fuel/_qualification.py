#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Candidate capability profiles of the nuclear-fuel application.

Profiles declare candidate support only; they never stand in for release
evidence.
"""

from __future__ import annotations

from types import MappingProxyType

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._validation import canonical_identifier
from ...qualification import CapabilityProfile, SupportTuple
from ...typing import parse
from ._bundles import FuelModelBundle
from ._identity import AxialInteraction, CoolantBoundaryKind, FuelGeometryReduction


VERIFICATION_GATES = tuple(
    f"nuclear-fuel.verification.V{index}" for index in range(1, 16)
)
SCIENTIFIC_RELEASE_GATES = (
    "nuclear-fuel.ifpe-data-rights",
    "nuclear-fuel.independent-locked-evaluation",
    "nuclear-fuel.ancestry-and-fabrication-split",
    "nuclear-fuel.precommitted-observable-criteria",
    "nuclear-fuel.runtime-precision-resources",
)
# Material system shared by the family and exact execution declarations.
_MATERIAL_SYSTEM = MappingProxyType(
    {
        "fuel": "uo2-pellet",
        "cladding": "zircaloy-4",
        "heat_treatment": "stress-relief-annealed",
    }
)


def nuclear_fuel_profiles() -> tuple[CapabilityProfile, ...]:
    """Declare the source-bound model family without a default spectrum.

    The family entry has no executable bundle and serves discovery only;
    :func:`nuclear_fuel_execution_profile` binds an actual prepared model.
    """
    return (
        CapabilityProfile(
            "nuclear-fuel.uo2-zircaloy.candidate",
            "phydrax",
            "candidate",
            (
                SupportTuple(
                    "nuclear-fuel.performance",
                    {
                        "support_kind": "source-bound-family-declaration",
                        **_MATERIAL_SYSTEM,
                        "model_family": "mechanistic-gas-effective-diffusivity",
                        "scattering_input": "required-spectrum-averaged-source-datum",
                        "regulatory_use": False,
                    },
                ),
            ),
            required_gates=(*VERIFICATION_GATES, *SCIENTIFIC_RELEASE_GATES),
            released=False,
        ),
    )


def nuclear_fuel_execution_profile(
    bundle: FuelModelBundle,
    /,
    *,
    reduction: FuelGeometryReduction,
    coolant: CoolantBoundaryKind,
    axial_interaction: AxialInteraction,
    prepared_id: str,
    runtime_id: str,
    resource_evidence_id: str,
    precision_evidence_id: str,
    dtype: str,
    channel_profile_id: str | None = None,
) -> CapabilityProfile:
    """Bind candidate support to one prepared model and measured runtime budget.

    The supplied IDs identify that model and its runtime evidence; they satisfy
    no release gate. Release goes through the signed release index.
    """
    if not isinstance(bundle, FuelModelBundle):
        raise TypeError(
            "Execution support requires the actual source-bound FuelModelBundle."
        )
    reduction_ = parse(reduction, FuelGeometryReduction, "reduction")
    coolant_ = parse(coolant, CoolantBoundaryKind, "coolant")
    interaction = parse(axial_interaction, AxialInteraction, "axial_interaction")
    prepared_id = canonical_identifier(prepared_id, "prepared_id")
    runtime_id = canonical_identifier(runtime_id, "runtime_id")
    resource_evidence_id = canonical_identifier(
        resource_evidence_id, "resource_evidence_id"
    )
    precision_evidence_id = canonical_identifier(
        precision_evidence_id, "precision_evidence_id"
    )
    if dtype != "float64":
        raise ValueError(
            "Prepared production fuel rods support explicit float64 precision only."
        )
    dependencies = (nuclear_fuel_profiles()[0].profile_id,)
    if coolant_ == "coupled-channel":
        if channel_profile_id is None:
            raise ValueError(
                "Coupled-channel execution requires its exact admitted channel profile ID."
            )
        dependencies = (
            *dependencies,
            canonical_identifier(channel_profile_id, "channel_profile_id"),
        )
    elif channel_profile_id is not None:
        raise ValueError(
            "A channel profile dependency belongs only to coupled-channel execution."
        )
    source = bundle.radial_power.scattering_source_id
    support = SupportTuple(
        "nuclear-fuel.performance",
        {
            "support_kind": "exact-prepared-execution",
            "bundle_id": bundle.bundle_id,
            "prepared_id": prepared_id,
            "parameter_values_id": canonical_fingerprint(array_tree_fingerprint(bundle)),
            **_MATERIAL_SYSTEM,
            "reduction": reduction_,
            "coolant": coolant_,
            "axial_interaction": interaction,
            "normal_contact": "fischer-burmeister",
            "gas_modes": bundle.fission_gas.mode_count,
            "scattering_source_id": source,
            "scattering_cross_section_barn_hex": float(
                bundle.radial_power.scattering_cross_section_barn
            ).hex(),
            "radial_power_law_id": bundle.radial_power.law_id,
            "gap_law_id": bundle.gap.law_id,
            "applicability_policy": bundle.applicability,
            "helium_law_id": bundle.helium.law_id,
            "helium_production_plan_id": bundle.helium_source.production.plan_id,
            "helium_production_provenance_id": bundle.helium_source.production.provenance.provenance_id,
            "helium_inventory_network_id": bundle.helium_source.inventory.network_id,
            "helium_inventory_reference_volume_m3_hex": float(
                bundle.helium_source.reference_volume
            ).hex(),
            "bore_emissivity_source_id": bundle.bore_emissivity.source_id,
            "deposition_source_id": bundle.deposition.source_id,
            # A frictionless bundle has no friction source.
            **(
                {}
                if bundle.friction is None
                else {"friction_source_id": bundle.friction.source_id}
            ),
            "runtime_id": runtime_id,
            "dtype": dtype,
            "resource_evidence_id": resource_evidence_id,
            "precision_evidence_id": precision_evidence_id,
            "regulatory_use": False,
        },
    )
    return CapabilityProfile(
        "nuclear-fuel.uo2-zircaloy.execution.candidate",
        "phydrax",
        "candidate",
        (support,),
        dependencies=dependencies,
        required_gates=(*VERIFICATION_GATES, *SCIENTIFIC_RELEASE_GATES),
        released=False,
    )


__all__ = ["nuclear_fuel_profiles", "nuclear_fuel_execution_profile"]
