#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact single-host candidate tuples backed by retained omniphysics evidence."""

from __future__ import annotations

from ._catalog import (
    CapabilityDeclaration,
    CapabilityDisposition,
    EvidenceAssessment,
    EvidenceDimension,
    EvidenceState,
)
from ._registry import CapabilityProfile, SupportTuple


_CONTROL_IDS = {
    "materials": "807c42a5203dbe5cd356fb269149235e75f1e2444dab75ff80160bb7dcae1a29",
    "manufacturing": "12331c67708cd3c53d340eaafd652085a3b7ca11f08c6fc8648ea09c18e1d4fb",
    "frequency": "e3b46768a09ed1e897348865477455df7cd319c0cc3f2f30400a5cb8d59c58d7",
    "population-balance": "abbe5f2856e0d3dd75b02cb8b2aa5b11c8e9144a0d953640ad7f504a0e2814bd",
    "system-modeling": "c5d21f61333ef3cb83737be578ab3a6aff7f50bceac51b7bfdc9715cd00da60e",
    "rheology": "bbb1255be89b04653cec27da899a6f3c734d094f5dfc4c304cb7ab86900fae71",
    "interfacial-transport": "0a28a4f988594c6e5266f230c947e75351c2cd8211b7e06ccfe96901d9e036ed",
    "structural-dynamics": "bd6807569c936ee2cba52d46636c9bb45dd2ca5b5fb2ba0400e520542ce0ca46",
    "correlation": "456e90fc764315b1a10603bddbb22f69e77839edf38f888a0bd84770af2a5935",
    "electrohydrodynamics": "4b5359b506f0f6b01d60352b2e63c5a72bec0ffe9f12412b7977f1801fb2fbc1",
    "phoresis": "c72239a6dba8f933f26ac1e61cc867128ecac1000cacf873a2850459f5d85a27",
    "smart-materials": "7fd0b308e6af00f2117faaea0a27035e1385616fb0c4d1522a5251782c75d644",
    "chemo-mechanics": "4d9cadf1a5d208237c1089acbb5acca29e477372f00f9ecb2aa658c0aa0294d9",
    "tribology": "3bde5b9e57105ad7591871aa6647012ed6f2ef932a5b0bafc4ab90def3c268d5",
    "thermal": "0b3d6c114dd9e8b91f57ff479c56737bba75cde11a2d8c5be06cb3b4dce7bc57",
    "membranes": "e01330c2c8634c2b62db1084f352b7b23ba569c177a3d89ba6986cd39e4b548a",
    "surface-chemistry": "9f993c87892a67816b0cafae22358da70e3e5b71ea91864dc0f4516bfd00d9fe",
    "optomechanics": "ed2195a67c5d0e51b457e744312b518a1eeb4e6abc8cb797e0ec01339253f78e",
    "acoustics": "9d883d5d94ce2fff068242d9c34e1e54c89c11b8d91cacc0f921a443bf12600c",
    "electrochemistry": "0696fe25312d880ffe9105d5b4bd7a7d30ab83df3727afb99f79247d5444d22f",
    "process-systems": "56ef8fd9dc92e5b30101ad1a53d8f028f19d1b1d6570b66c919aaec3d22fd500",
}
_REFINEMENT_IDS = {
    "materials": "a4dc3ea842fb12cd10c743027d4ca56fe49eb6c2113e9f7f5629377e4dee3938",
    "manufacturing": "a4dc3ea842fb12cd10c743027d4ca56fe49eb6c2113e9f7f5629377e4dee3938",
    "thermal": "a4dc3ea842fb12cd10c743027d4ca56fe49eb6c2113e9f7f5629377e4dee3938",
    "structural-dynamics": "149c51a248faa7adcfeda57b2f26a3c91cdd04bf7d1b0ab42e08d5274113f847",
    "correlation": "149c51a248faa7adcfeda57b2f26a3c91cdd04bf7d1b0ab42e08d5274113f847",
    "acoustics": "149c51a248faa7adcfeda57b2f26a3c91cdd04bf7d1b0ab42e08d5274113f847",
    "smart-materials": "149c51a248faa7adcfeda57b2f26a3c91cdd04bf7d1b0ab42e08d5274113f847",
    "optomechanics": "149c51a248faa7adcfeda57b2f26a3c91cdd04bf7d1b0ab42e08d5274113f847",
    "rheology": "cd44e709e29880522e033ed10e254066fb7f6f1c55260d999af5893f962e8372",
    "electrohydrodynamics": "cd44e709e29880522e033ed10e254066fb7f6f1c55260d999af5893f962e8372",
    "interfacial-transport": "cd44e709e29880522e033ed10e254066fb7f6f1c55260d999af5893f962e8372",
}
_PROVIDER_ID = "8228fef85cc35bf25c8270140cb8f6ea01c8ae806e86d0fb1983841d2ebf9a61"
_APPLICATION_IDS = {
    "manufacturing": "8c13d188be5a88cc20f7d482f192870b35d4402d72907988ab221a33f3bd8e14",
}


_SPECS = (
    (
        "materials",
        "materials.spatial-icme",
        "phydrax.materials",
        "phydrax.materials.SpatialICMEModel",
        "process-structure-property-relaxation",
    ),
    (
        "manufacturing",
        "manufacturing.scheduled-spatial-runtime",
        "phydrax.manufacturing",
        "phydrax.manufacturing.ManufacturingRuntime",
        "moving-source-control-volume-ledger",
    ),
    (
        "frequency",
        "frequency.second-order-complex",
        "phydrax.frequency",
        "phydrax.frequency.CompiledFrequencySystem",
        "dense-second-order-frequency-system",
    ),
    (
        "population-balance",
        "population-balance.conservative-sectional",
        "phydrax.population_balance",
        "phydrax.population_balance.ConservativeSectionalSolver",
        "one-coordinate-fixed-pivot-overflow",
    ),
    (
        "system-modeling",
        "system-modeling.linear-acausal",
        "phydrax.system_modeling",
        "phydrax.system_modeling.compile_linear_acausal_system",
        "linear-square-across-through-system",
    ),
    (
        "rheology",
        "rheology.spatial-conformation",
        "phydrax.rheology",
        "phydrax.rheology.SpatialConformationSolver",
        "fixed-mesh-implicit-transport",
    ),
    (
        "interfacial-transport",
        "interfacial-transport.bulk-surface",
        "phydrax.interfacial_transport",
        "phydrax.interfacial_transport.CoupledBulkSurfaceTransport",
        "fixed-topology-single-species",
    ),
    (
        "structural-dynamics",
        "structural-dynamics.linear-modal-transient",
        "phydrax.structural_dynamics",
        "phydrax.structural_dynamics.LinearStructuralSystem",
        "linear-symmetric-newmark",
    ),
    (
        "correlation",
        "correlation.modal-frf",
        "phydrax.correlation",
        "phydrax.correlation.correlate_modes",
        "exact-pairing-sixteen-candidate-limit",
    ),
    (
        "electrohydrodynamics",
        "electrohydrodynamics.operator-coupled",
        "phydrax.electrohydrodynamics",
        "phydrax.electrohydrodynamics.CoupledElectrohydrodynamicSolver",
        "fixed-mesh-electrostatic-creeping-flow",
    ),
    (
        "phoresis",
        "phoresis.oseen-cloud",
        "phydrax.phoresis",
        "phydrax.phoresis.HydrodynamicPhoreticSolver",
        "three-dimensional-nonoverlapping-far-field",
    ),
    (
        "smart-materials",
        "smart-materials.spatial-piezoelectric",
        "phydrax.smart_materials",
        "phydrax.smart_materials.SpatialPiezoelectricSystem",
        "linear-reciprocal-piezoelectric",
    ),
    (
        "chemo-mechanics",
        "chemo-mechanics.conservative-spatial",
        "phydrax.chemo_mechanics",
        "phydrax.chemo_mechanics.SpatialChemoMechanicalSystem",
        "linear-onsager-fixed-mesh",
    ),
    (
        "tribology",
        "tribology.mass-conserving-ehl",
        "phydrax.tribology",
        "phydrax.tribology.MassConservingEHLSolver",
        "one-dimensional-elrod-compressible",
    ),
    (
        "thermal",
        "thermal.diffuse-gray-enclosure",
        "phydrax.thermal_systems",
        "phydrax.thermal_systems.DiffuseGrayEnclosure",
        "closed-reciprocal-gray-enclosure",
    ),
    (
        "membranes",
        "membranes.segmented-crossflow",
        "phydrax.membranes",
        "phydrax.membranes.CrossflowMembraneModule",
        "cocurrent-solution-diffusion",
    ),
    (
        "surface-chemistry",
        "surface-chemistry.segmented-catalytic",
        "phydrax.surface_chemistry",
        "phydrax.surface_chemistry.SegmentedCatalyticReactor",
        "plug-flow-mass-action",
    ),
    (
        "optomechanics",
        "optomechanics.spatial-stop",
        "phydrax.optomechanics",
        "phydrax.optomechanics.SpatialOptomechanicalSystem",
        "linear-thermal-structural-optical",
    ),
    (
        "acoustics",
        "acoustics.vibroacoustic",
        "phydrax.acoustics",
        "phydrax.acoustics.VibroacousticSystem",
        "linear-reciprocal-frequency-domain",
    ),
    (
        "electrochemistry",
        "electrochemistry.porous-electrode",
        "phydrax.electrochemistry",
        "phydrax.electrochemistry.PorousElectrodeSystem",
        "single-reaction-fixed-mesh",
    ),
    (
        "process-systems",
        "process-systems.equation-oriented",
        "phydrax.process_systems",
        "phydrax.process_systems.EquationOrientedFlowsheet",
        "square-differentiable-steady-system",
    ),
)

_DEPTHS = {
    "frequency": "reduced-system",
    "system-modeling": "reduced-system",
    "correlation": "reduced-system",
    "phoresis": "reduced-system",
    "process-systems": "reduced-system",
    "materials": "spatial-single-physics",
    "manufacturing": "spatial-single-physics",
    "population-balance": "spatial-single-physics",
    "rheology": "spatial-single-physics",
    "structural-dynamics": "spatial-single-physics",
    "thermal": "spatial-single-physics",
    "surface-chemistry": "spatial-single-physics",
    "interfacial-transport": "spatial-coupled",
    "electrohydrodynamics": "spatial-coupled",
    "smart-materials": "spatial-coupled",
    "chemo-mechanics": "spatial-coupled",
    "tribology": "spatial-coupled",
    "membranes": "spatial-coupled",
    "optomechanics": "spatial-coupled",
    "acoustics": "spatial-coupled",
    "electrochemistry": "spatial-coupled",
}


def qualified_omniphysics_profiles() -> tuple[CapabilityProfile, ...]:
    return tuple(
        CapabilityProfile(
            f"{capability}.candidate",
            "phydrax",
            "candidate-2026-09-20",
            (
                SupportTuple(
                    capability,
                    {
                        "depth": _DEPTHS[family],
                        "topology": "fixed",
                        "execution": "single-host-cpu-float64",
                        "workflow": "forward",
                        "scope": scope,
                    },
                ),
            ),
            required_gates=(
                "numerical-control",
                "single-host-provider",
                "source-review",
            ),
        )
        for family, capability, _, _, scope in _SPECS
    )


def qualified_omniphysics_declarations() -> tuple[CapabilityDeclaration, ...]:
    profiles = {
        profile.capability: profile for profile in qualified_omniphysics_profiles()
    }
    declarations = []
    for family, capability, owner, symbol, _ in _SPECS:
        numerical_ids = [_CONTROL_IDS[family]]
        if family in _REFINEMENT_IDS:
            numerical_ids.append(_REFINEMENT_IDS[family])
        scientific = (
            EvidenceAssessment(
                EvidenceDimension.SCIENTIFIC,
                EvidenceState.PASSED,
                evidence_ids=(_APPLICATION_IDS[family],),
                reason="analytic-application-reference-passed",
            )
            if family in _APPLICATION_IDS
            else EvidenceAssessment(
                EvidenceDimension.SCIENTIFIC,
                EvidenceState.BLOCKED,
                reason="no-independent-experimental-validation",
            )
        )
        declarations.append(
            CapabilityDeclaration(
                capability,
                owner,
                CapabilityDisposition.CANDIDATE,
                domain_maturity="implementation-qualified-candidate",
                public_symbols=(symbol,),
                profiles=(profiles[capability],),
                evidence=(
                    EvidenceAssessment(
                        EvidenceDimension.IMPLEMENTATION,
                        EvidenceState.PASSED,
                        evidence_ids=(_CONTROL_IDS[family],),
                        reason="executable-spatial-control-passed",
                    ),
                    EvidenceAssessment(
                        EvidenceDimension.NUMERICAL,
                        EvidenceState.PASSED,
                        evidence_ids=tuple(numerical_ids),
                        reason="retained-control-and-available-refinement-passed",
                    ),
                    EvidenceAssessment(
                        EvidenceDimension.HARDWARE_PROVIDER,
                        EvidenceState.PASSED,
                        evidence_ids=(_PROVIDER_ID,),
                        reason="single-host-arm64-cpu-execution-passed",
                    ),
                    scientific,
                    EvidenceAssessment(
                        EvidenceDimension.RIGHTS_SECURITY,
                        EvidenceState.BLOCKED,
                        reason="technical-source-review-complete-legal-approval-pending",
                    ),
                    EvidenceAssessment(
                        EvidenceDimension.RELEASE,
                        EvidenceState.BLOCKED,
                        reason="candidate-not-release-authorized",
                    ),
                ),
                documentation=("docs/guides_omniphysics_program.md",),
                intended_uses=("bounded-single-host-engineering-evaluation",),
                nonclaims=(
                    "not-release-authorized",
                    "no-distributed-hardware-evidence",
                    "no-unlisted-regime-or-topology-support",
                ),
            )
        )
    return tuple(declarations)


__all__ = ["qualified_omniphysics_declarations", "qualified_omniphysics_profiles"]
