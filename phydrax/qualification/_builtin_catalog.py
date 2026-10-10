#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lazy assembly of the repository's built-in capability inventory."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib import import_module
from types import ModuleType

from ._catalog import (
    CapabilityCatalog,
    CapabilityDeclaration,
    CapabilityDisposition,
    declarations_from_profiles,
    EvidenceAssessment,
    EvidenceDimension,
    EvidenceState,
)
from ._qualified_profiles import (
    qualified_omniphysics_declarations,
    qualified_omniphysics_profiles,
)
from ._registry import CapabilityProfile, SupportTuple


# Specific owners precede aggregate ledgers. Duplicate content-addressed profiles are
# intentionally collapsed without changing the owner selected by the first producer.
_PROFILE_PROVIDERS = (
    ("phydrax.qualification._core_portfolio", "core_candidate_profiles"),
    ("phydrax.discretization.meshfree._profiles", "meshfree_candidate_profiles"),
    ("phydrax.atomistic._mace_profiles", "mace_candidate_profiles"),
    ("phydrax.linalg._svd_qualification", "singular_subspace_candidate_profiles"),
    ("phydrax.acoustics._bubbly", "bubbly_medium_candidate_profiles"),
    ("phydrax.applications.foams._profiles", "foam_candidate_profiles"),
    (
        "phydrax.applications.soap_film_tunnel._profiles",
        "soap_film_tunnel_candidate_profiles",
    ),
    ("phydrax.bubble_dynamics._cloud_profiles", "bubble_cloud_candidate_profiles"),
    ("phydrax.bubble_dynamics._profiles", "bubble_dynamics_candidate_profiles"),
    (
        "phydrax.discretization.lattice_boltzmann",
        "color_gradient_candidate_profiles",
    ),
    (
        "phydrax.geometry.multiregion_surface._profiles",
        "multiregion_surface_candidate_profiles",
    ),
    (
        "phydrax.interfacial_transport._core",
        "interfacial_transport_candidate_profiles",
    ),
    (
        "phydrax.optics.wave._thin_film_qualification",
        "thin_film_interference_candidate_profiles",
    ),
    (
        "phydrax.rendering._thin_film_qualification",
        "thin_film_appearance_candidate_profiles",
    ),
    (
        "phydrax.threshold_dynamics._profiles",
        "threshold_dynamics_candidate_profiles",
    ),
    ("phydrax.applications.battery._dfn", "battery_dfn_candidate_profiles"),
    ("phydrax.nuclear._transport", "nuclear_transport_candidate_profiles"),
    (
        "phydrax.applications.conformal_bootstrap._qualification",
        "conformal_bootstrap_candidate_profiles",
    ),
    ("phydrax.applications.fuzzy_space._qualification", "fuzzy_space_candidate_profiles"),
    (
        "phydrax.applications.magnetic_resonance._qualification",
        "magnetic_resonance_candidate_profiles",
    ),
    ("phydrax.applications.magnetism._qualification", "magnetism_candidate_profiles"),
    (
        "phydrax.applications.numerical_relativity._ads_qualification",
        "ads_conformal_candidate_profiles",
    ),
    (
        "phydrax.applications.phase_field._coupled_profiles",
        "coupled_phase_field_candidate_profiles",
    ),
    ("phydrax.applications.phase_field._profiles", "phase_field_candidate_profiles"),
    (
        "phydrax.applications.phase_field._stationary_qualification",
        "stationary_soliton_candidate_profiles",
    ),
    (
        "phydrax.applications.radiation_transport._qualification",
        "radiation_transport_candidate_profiles",
    ),
    (
        "phydrax.applications.reacting_flow._qualification",
        "reacting_flow_candidate_profiles",
    ),
    ("phydrax.applications.reactor_physics._qualification", "reactor_candidate_profiles"),
    (
        "phydrax.applications.semiconductor._production_qualification",
        "semiconductor_candidate_profiles",
    ),
    ("phydrax.applications.spin_foam._qualification", "spin_foam_candidate_profiles"),
    (
        "phydrax.applications.spin_network._qualification",
        "spin_network_candidate_profiles",
    ),
    (
        "phydrax.applications.superconductivity._qualification",
        "superconductivity_candidate_profiles",
    ),
    (
        "phydrax.applications.supersymmetric_lattice._qualification",
        "supersymmetric_lattice_candidate_profiles",
    ),
    ("phydrax.applications.tokamak._qualification", "tokamak_candidate_profiles"),
    (
        "phydrax.chemistry.periodic._embedding_qualification",
        "green_embedding_candidate_profiles",
    ),
    (
        "phydrax.chemistry.periodic._lattice_qualification",
        "lattice_material_candidate_profiles",
    ),
    ("phydrax.chemistry.periodic._qualification", "periodic_candidate_profiles"),
    (
        "phydrax.chemistry.spectroscopy._qualification",
        "material_spectroscopy_candidate_profiles",
    ),
    ("phydrax.imaging._qualification", "imaging_candidate_profiles"),
    ("phydrax.nuclear._qualification", "nuclear_candidate_profiles"),
    (
        "phydrax.operators.quantum.lattice._qualification",
        "quantum_lattice_candidate_profiles",
    ),
    (
        "phydrax.optics.wave._envelope_qualification",
        "envelope_propagation_candidate_profiles",
    ),
    (
        "phydrax.particle_physics._spectrum_qualification",
        "particle_spectrum_candidate_profiles",
    ),
    ("phydrax.solver._calabi_yau_qualification", "calabi_yau_candidate_profiles"),
    (
        "phydrax.solver._maxwell_qualification",
        "maxwell_far_field_candidate_profiles",
    ),
    ("phydrax.solver._maxwell_qualification", "maxwell_antenna_candidate_profiles"),
    ("phydrax.solver._maxwell_qualification", "maxwell_dispersion_candidate_profiles"),
    (
        "phydrax.solver._maxwell_qualification",
        "maxwell_moving_charge_candidate_profiles",
    ),
    (
        "phydrax.solver._maxwell_qualification",
        "maxwell_frequency_moving_charge_candidate_profiles",
    ),
    ("phydrax.solver._pic_qualification", "pic_qed_cascade_candidate_profiles"),
    ("phydrax.solver._pic_qualification", "pic_polarized_qed_candidate_profiles"),
    (
        "phydrax.solver._pic_qualification",
        "pic_radiation_reaction_candidate_profiles",
    ),
    ("phydrax.solver._pic_qualification", "pic_spectral_candidate_profiles"),
    ("phydrax.solver._pic_qualification", "pic_boosted_frame_candidate_profiles"),
    ("phydrax.solver._pic_qualification", "pic_distributed_candidate_profiles"),
    (
        "phydrax.solver._pic_qualification",
        "pic_dispersive_self_consistent_candidate_profiles",
    ),
    (
        "phydrax.electromagnetics._qualification",
        "electromagnetic_radiation_candidate_profiles",
    ),
    (
        "phydrax.optics.transport._qualification",
        "optical_transport_candidate_profiles",
    ),
    ("phydrax.applications.accelerator._qualification", "accelerator_candidate_profiles"),
    (
        "phydrax.qualification._radiation_release_matrix",
        "radiation_release_matrix_candidate_profiles",
    ),
    ("phydrax.applications._biophysical_qualification", "biophysical_candidate_profiles"),
    ("phydrax.applications._soft_matter_qualification", "soft_matter_candidate_profiles"),
    (
        "phydrax.applications._condensed_matter_evidence",
        "condensed_matter_candidate_profiles",
    ),
    (
        "phydrax.applications._condensed_matter_evidence",
        "condensed_matter_frontier_candidate_profiles",
    ),
)


def _public_owner(module_name: str, /) -> str:
    parts = module_name.split(".")
    if parts[-1].startswith("_"):
        parts.pop()
    return ".".join(parts)


def _provider(module_name: str, function_name: str, /) -> Callable[[], Iterable[object]]:
    module = import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(
            f"Capability provider {module_name}:{function_name} is not callable."
        )
    return function


def builtin_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Load all owner-local and aggregate evidence-free candidate profiles lazily."""

    profiles: dict[str, CapabilityProfile] = {}
    for module_name, function_name in _PROFILE_PROVIDERS:
        values = _provider(module_name, function_name)()
        for profile in values:
            if not isinstance(profile, CapabilityProfile):
                raise TypeError(
                    f"Capability provider {module_name}:{function_name} returned "
                    f"{type(profile).__name__}, not CapabilityProfile."
                )
            existing = profiles.get(profile.profile_id)
            if existing is not None and existing.to_record() != profile.to_record():
                raise ValueError(f"Conflicting candidate profile {profile.profile_id}.")
            profiles[profile.profile_id] = profile
    for profile in qualified_omniphysics_profiles():
        profiles[profile.profile_id] = profile
    for declaration in _native_meshing_declarations():
        if declaration.disposition is CapabilityDisposition.CANDIDATE:
            for profile in declaration.profiles:
                profiles[profile.profile_id] = profile
    return tuple(sorted(profiles.values(), key=lambda item: item.profile_id))


def _candidate_declarations() -> tuple[CapabilityDeclaration, ...]:
    profiles_by_id: dict[str, CapabilityProfile] = {}
    owner_by_capability: dict[str, str] = {}
    for module_name, function_name in _PROFILE_PROVIDERS:
        owner = _public_owner(module_name)
        values = _provider(module_name, function_name)()
        for profile in values:
            if not isinstance(profile, CapabilityProfile):
                raise TypeError(
                    f"Capability provider {module_name}:{function_name} returned "
                    f"{type(profile).__name__}, not CapabilityProfile."
                )
            if profile.profile_id in profiles_by_id:
                continue
            profiles_by_id[profile.profile_id] = profile
            owner_by_capability.setdefault(profile.capability, owner)
    return declarations_from_profiles(
        profiles_by_id.values(),
        owners=owner_by_capability,
        nonclaim="candidate-not-release-authorized",
    )


def _operator_declarations() -> tuple[CapabilityDeclaration, ...]:
    from phydrax.nn.operator.catalog import OPERATOR_ARCHITECTURE_STATUSES

    declarations = []
    for name, status in OPERATOR_ARCHITECTURE_STATUSES.items():
        slug = "".join(
            character.lower() if character.isalnum() else "-" for character in name
        ).strip("-")
        declarations.append(
            CapabilityDeclaration(
                f"nn.operator.{slug}",
                "phydrax.nn.operator",
                CapabilityDisposition.RESEARCH,
                domain_maturity=status.tier,
                documentation=("docs/api/nn/architectures.md",),
                intended_uses=("operator-learning-research",),
                nonclaims=("no-scenario-specific-release-profile",),
            )
        )
    return tuple(declarations)


def _rom_declarations() -> tuple[CapabilityDeclaration, ...]:
    from phydrax.rom import rom_capability_catalog, ROMMaturity

    declarations = []
    for entry in rom_capability_catalog():
        if entry.maturity is ROMMaturity.INTERNAL:
            disposition = CapabilityDisposition.INTERNAL
        else:
            disposition = CapabilityDisposition.CANDIDATE
        profile = entry.profile(provider="phydrax")
        declarations.append(
            CapabilityDeclaration(
                profile.capability,
                "phydrax.rom",
                disposition,
                domain_maturity=entry.maturity.name.lower().replace("_", "-"),
                profiles=()
                if disposition is CapabilityDisposition.INTERNAL
                else (profile,),
                documentation=("docs/guides_reduced_order_modeling.md",),
                intended_uses=("bounded-reduced-order-modeling",),
                nonclaims=("not-release-authorized",)
                if disposition is CapabilityDisposition.CANDIDATE
                else (),
            )
        )
    return tuple(declarations)


def _research_platform_declarations() -> tuple[CapabilityDeclaration, ...]:
    specifications = (
        (
            "particle.discretization-platform",
            "phydrax.discretization.particle",
            "experimental",
            "docs/guides_particle_qualification.md",
            "no-released-particle-method-profile",
        ),
        (
            "tensor-network.platform",
            "phydrax.tensor_network",
            "experimental",
            "docs/guides_tensor_platform.md",
            "no-signed-release-profile",
        ),
        (
            "boundary.integral-platform",
            "phydrax.operators.integral",
            "Q0",
            "docs/guides_boundary_platform.md",
            "Q1-through-Q3-not-qualified",
        ),
        (
            "qft.frontier-platform",
            "phydrax.applications.lattice_field",
            "research",
            "docs/guides_qft_production.md",
            "finite-controls-do-not-establish-continuum-physics",
        ),
        (
            "platform.exterior-calculus",
            "phydrax.exterior",
            "analytic-control",
            "docs/guides_exterior_calculus.md",
            "realizations-beyond-qualified-scenarios-not-claimed",
        ),
        (
            "platform.finance",
            "phydrax.finance",
            "research",
            "docs/guides_finance.md",
            "no-live-market-or-regulatory-release-profile",
        ),
        (
            "platform.materials",
            "phydrax.materials",
            "analytic-control",
            "docs/guides_omniphysics_program.md",
            "spatial-icme-not-implementation-closed",
        ),
        (
            "platform.manufacturing",
            "phydrax.manufacturing",
            "analytic-control",
            "docs/guides_omniphysics_program.md",
            "spatial-process-runtime-not-implementation-closed",
        ),
        (
            "platform.frequency",
            "phydrax.frequency",
            "analytic-control",
            "docs/guides_omniphysics_program.md",
            "advanced-frequency-runtime-not-implementation-closed",
        ),
        (
            "platform.population-balance",
            "phydrax.population_balance",
            "analytic-control",
            "docs/guides_omniphysics_program.md",
            "general-population-balance-not-implementation-closed",
        ),
        (
            "platform.system-modeling",
            "phydrax.system_modeling",
            "semantic",
            "docs/guides_omniphysics_program.md",
            "acausal-compiler-not-implementation-closed",
        ),
        (
            "platform.rheology",
            "phydrax.rheology",
            "local-constitutive",
            "docs/guides_omniphysics_program.md",
            "spatial-rheology-not-implementation-closed",
        ),
        (
            "platform.interfacial-transport",
            "phydrax.interfacial_transport",
            "local-constitutive",
            "docs/guides_omniphysics_program.md",
            "surface-pde-not-implementation-closed",
        ),
        (
            "platform.structural-dynamics",
            "phydrax.structural_dynamics",
            "analytic-control",
            "docs/guides_omniphysics_program.md",
            "engineering-dynamics-not-implementation-closed",
        ),
        (
            "platform.correlation",
            "phydrax.correlation",
            "analytic-control",
            "docs/guides_omniphysics_program.md",
            "test-correlation-workflow-not-implementation-closed",
        ),
        (
            "platform.electrohydrodynamics",
            "phydrax.electrohydrodynamics",
            "local-constitutive",
            "docs/guides_omniphysics_program.md",
            "coupled-ehd-not-implementation-closed",
        ),
        (
            "platform.phoresis",
            "phydrax.phoresis",
            "analytic-control",
            "docs/guides_omniphysics_program.md",
            "resolved-phoresis-not-implementation-closed",
        ),
        (
            "platform.smart-materials",
            "phydrax.smart_materials",
            "local-constitutive",
            "docs/guides_omniphysics_program.md",
            "spatial-smart-materials-not-implementation-closed",
        ),
        (
            "platform.chemo-mechanics",
            "phydrax.chemo_mechanics",
            "local-constitutive",
            "docs/guides_omniphysics_program.md",
            "spatial-chemo-mechanics-not-implementation-closed",
        ),
        (
            "platform.tribology",
            "phydrax.tribology",
            "analytic-control",
            "docs/guides_omniphysics_program.md",
            "mass-conserving-ehl-not-implementation-closed",
        ),
        (
            "platform.thermal-systems",
            "phydrax.thermal_systems",
            "analytic-control",
            "docs/guides_omniphysics_program.md",
            "spatial-thermal-systems-not-implementation-closed",
        ),
        (
            "platform.membranes",
            "phydrax.membranes",
            "local-constitutive",
            "docs/guides_omniphysics_program.md",
            "spatial-membrane-modules-not-implementation-closed",
        ),
        (
            "platform.surface-chemistry",
            "phydrax.surface_chemistry",
            "local-constitutive",
            "docs/guides_omniphysics_program.md",
            "spatial-catalytic-reactors-not-implementation-closed",
        ),
        (
            "platform.optomechanics",
            "phydrax.optomechanics",
            "local-constitutive",
            "docs/guides_omniphysics_program.md",
            "coupled-optomechanics-not-implementation-closed",
        ),
        (
            "platform.acoustics",
            "phydrax.acoustics",
            "analytic-control",
            "docs/guides_omniphysics_program.md",
            "spatial-acoustics-not-implementation-closed",
        ),
        (
            "platform.electrochemistry",
            "phydrax.electrochemistry",
            "local-constitutive",
            "docs/guides_omniphysics_program.md",
            "spatial-electrochemistry-not-implementation-closed",
        ),
        (
            "platform.process-systems",
            "phydrax.process_systems",
            "analytic-control",
            "docs/guides_omniphysics_program.md",
            "equation-oriented-flowsheets-not-implementation-closed",
        ),
        (
            "platform.numerical-interoperability",
            "phydrax.solver.coupling",
            "research",
            "docs/guides_numerical_interoperability.md",
            "method-combinations-beyond-qualified-scenarios-not-claimed",
        ),
    )
    return tuple(
        CapabilityDeclaration(
            capability,
            owner,
            CapabilityDisposition.RESEARCH,
            domain_maturity=maturity,
            documentation=(documentation,),
            intended_uses=("bounded-research",),
            nonclaims=("umbrella-platform-broader-than-qualified-tuples",),
        )
        for capability, owner, maturity, documentation, _ in specifications
    )


def _compressible_kinetic_declarations() -> tuple[CapabilityDeclaration, ...]:
    return (
        CapabilityDeclaration(
            "kinetic.compressible-entropic",
            "phydrax.discretization",
            CapabilityDisposition.RESEARCH,
            domain_maturity="research-production-closure",
            public_symbols=(
                "phydrax.discretization.PositiveCompressibleKineticPlan",
                "phydrax.discretization.FullRangeQuasiEquilibriumPlan",
                "phydrax.discretization.FilteredD3Q33Plan",
                "phydrax.discretization.CompressibleKineticRuntimePlan",
                "phydrax.discretization.guided_d3q39_plan",
                "phydrax.discretization.entropic_d3q343_plan",
            ),
            documentation=("docs/guides_lattice_boltzmann.md",),
            intended_uses=(
                "bounded-compressible-kinetic-research",
                "explicit-support-production-qualification",
            ),
            nonclaims=(
                "no-universal-mach-or-stability-claim",
                "no-blanket-release-across-model-mesh-physics-products",
                "entropy-stabilization-is-not-turbulence-closure",
            ),
        ),
    )


def _privacy_declarations() -> tuple[CapabilityDeclaration, ...]:
    support = tuple(
        SupportTuple(
            "privacy.control-plane",
            {
                "unit": "operator-case",
                "adjacency": "add-or-remove-one",
                "trust-model": "central",
                "sampling": "poisson",
                "mechanism": "gaussian-dp-sgd",
                "accountant": accountant,
                "accountant-configuration": (
                    "pld-discretization-1e-4"
                    if accountant == "pld"
                    else "rdp-default-orders"
                ),
                "accounting-provider-version": "0.6.0",
                "microbatch-size": microbatch_size,
                "dtype": dtype,
                "process-count": 1,
                "static-case-schema": "fixed",
                "data-source": "in-memory-case-source",
                "device-count": 1,
                "randomness": "research-prng",
                "prng-implementation": "threefry2x32",
                "provider-version": "2.0.0",
            },
        )
        for accountant in ("pld", "rdp")
        for dtype in ("float32", "float64")
        for microbatch_size in ("none", 1)
    )
    profile = CapabilityProfile(
        "privacy.operator-case-dpsgd.research",
        "jax-privacy",
        support,
        required_gates=(
            "artifact-redaction",
            "finite-precision",
            "independent-review",
            "mechanism-accounting",
            "neighboring-dataset",
            "randomness",
        ),
        released=False,
    )
    return (
        CapabilityDeclaration(
            "privacy.control-plane",
            "phydrax.privacy",
            CapabilityDisposition.RESEARCH,
            domain_maturity="research",
            profiles=(profile,),
            public_symbols=(
                "phydrax.privacy.AccountingMethod",
                "phydrax.privacy.DPSGDPlan",
                "phydrax.privacy.MechanismTrace",
                "phydrax.privacy.NeighboringRelation",
                "phydrax.privacy.PrivateDataScope",
                "phydrax.privacy.PrivateTrainingPlan",
                "phydrax.privacy.PrivacyBudget",
                "phydrax.privacy.PrivacyCertificate",
                "phydrax.privacy.PrivacyDefinition",
                "phydrax.privacy.PrivacyGuarantee",
                "phydrax.privacy.PrivacyReleaseLedger",
                "phydrax.privacy.PrivacyReleaseReceipt",
                "phydrax.privacy.PrivacyUnit",
                "phydrax.privacy.RandomnessAssurance",
                "phydrax.privacy.TrustModel",
                "phydrax.privacy.account_mechanism_trace",
                "phydrax.privacy.account_mechanism_traces",
                "phydrax.privacy.certify_private_release",
                "phydrax.privacy.dp_event_from_record",
                "phydrax.privacy.dp_event_to_record",
            ),
            documentation=("docs/guides_privacy.md",),
            intended_uses=("bounded-private-training-research",),
            nonclaims=(
                "central-case-level-add-remove-only",
                "research-prng-not-public-release-authorized",
            ),
        ),
    )


def _application_declarations() -> tuple[CapabilityDeclaration, ...]:
    import phydrax.applications as applications

    declarations = []
    for name in applications.__all__:
        value = getattr(applications, name)
        if not isinstance(value, ModuleType):
            continue
        module_name = getattr(value, "__name__", f"phydrax.applications.{name}")
        slug = name.replace("_", "-")
        declarations.append(
            CapabilityDeclaration(
                f"application.{slug}",
                module_name,
                CapabilityDisposition.RESEARCH,
                domain_maturity="application-research",
                public_symbols=(f"phydrax.applications.{name}",),
                intended_uses=("bounded-application-research",),
                nonclaims=("no-umbrella-application-release-profile",),
            )
        )
    return tuple(declarations)


def _native_meshing_combination(
    name: str,
    source: str,
    operation: str,
    cell_family: str,
    geometry_order: int,
    controls: str,
    placement: str,
    derivative: str,
    transfer: str,
    /,
    *,
    admission: str,
    source_references: tuple[str, ...],
    required_evidence: tuple[str, ...],
    nonclaims: tuple[str, ...] = (),
) -> CapabilityDeclaration:
    """Declare exact admission or an incomplete obligation, never a product.

    Profiles remain unsigned and unreleased. Source-inspected admission is
    separate from corpus completion, independent certification, and release or
    leadership evidence; none of those evidence gates is inferred from code.
    """
    capability = f"meshing.native.{name}"
    admitted = admission == "implemented-admitted"
    support = SupportTuple(
        capability,
        {
            "source": source,
            "operation": operation,
            "cell_family": cell_family,
            "geometry_order": geometry_order,
            "controls": controls,
            "placement": placement,
            "derivative_contract": derivative,
            "transfer_contract": transfer,
            "implementation_admission": admission,
        },
    )
    requirements = (
        "source-request-policy-runtime-bound-identities",
        "licensed-digested-small-and-large-positive-corpus",
        "mandatory-positive-native-corpus-completion",
        "adversarial-corpus-with-independent-fault-adequate-oracles",
        "native-engine-free-install-import-and-execution",
        "independent-validity-embedding-fidelity-coverage-semantics",
        "atomic-resource-refusal-and-accepted-state-rollback",
        "frozen-like-for-like-baseline-contracts",
        "repeated-phase-timing-compile-memory-and-failure-tails",
        "end-to-end-time-to-fixed-physical-error-or-qoi",
        "authenticated-release-and-preregistered-leadership-review",
        *required_evidence,
    )
    profile = CapabilityProfile(
        f"{capability}.candidate" if admitted else f"{capability}.requirements",
        "phydrax-native-meshing",
        (support,),
        required_gates=requirements,
        released=False,
    )
    return CapabilityDeclaration(
        capability,
        "phydrax.meshing",
        CapabilityDisposition.CANDIDATE if admitted else CapabilityDisposition.RESEARCH,
        domain_maturity=(
            "source-inspected-native-admission"
            if admitted
            else "mandatory-native-combination-incomplete"
        ),
        profiles=(profile,),
        evidence=(
            EvidenceAssessment(
                EvidenceDimension.IMPLEMENTATION,
                EvidenceState.UNASSESSED if admitted else EvidenceState.BLOCKED,
                reason=admission,
            ),
            EvidenceAssessment(
                EvidenceDimension.OPERATIONS,
                EvidenceState.UNASSESSED,
                reason="mandatory-positive-native-corpus-completion-not-established",
            ),
            EvidenceAssessment(
                EvidenceDimension.SCIENTIFIC,
                EvidenceState.UNASSESSED,
                reason="independent-route-and-consumer-certification-not-established",
            ),
            EvidenceAssessment(
                EvidenceDimension.DERIVATIVE,
                EvidenceState.UNASSESSED,
                reason="only-the-declared-fixed-epoch-contract-may-be-assessed",
            ),
            EvidenceAssessment(
                EvidenceDimension.HARDWARE_PROVIDER,
                EvidenceState.UNASSESSED,
                reason="engine-free-placement-execution-not-established",
            ),
            EvidenceAssessment(
                EvidenceDimension.PERFORMANCE,
                EvidenceState.UNASSESSED,
                reason="like-for-like-leadership-target-not-observed",
            ),
            EvidenceAssessment(
                EvidenceDimension.RELEASE,
                EvidenceState.BLOCKED,
                reason="no-authenticated-release-evidence",
            ),
        ),
        documentation=("docs/guides_meshing.md", "docs/api/meshing.md"),
        intended_uses=(
            "bounded-native-meshing-evaluation",
            *(f"source-reference:{path}" for path in source_references),
            *(f"required-evidence:{requirement}" for requirement in requirements),
        ),
        nonclaims=(
            "no-cartesian-product-of-source-operation-family-order-controls-placement",
            "source-inspection-is-not-an-observed-qualified-run",
            "safe-refusal-does-not-complete-a-mandatory-positive-workflow",
            "external-provider-success-is-not-native-evidence",
            "missing-optional-comparison-engine-is-missing-dependency",
            "focused-test-and-tooling-passes-are-not-final-qualification-artifacts",
            "final-like-for-like-leadership-campaign-not-run",
            "no-release-or-measured-superiority-claim",
            "undeclared-derivative-and-transfer-families-not-supported",
            *nonclaims,
        ),
    )


def _native_meshing_declarations() -> tuple[CapabilityDeclaration, ...]:
    """Exact admitted slices and still-open W15 mandatory combined workflows."""
    return (
        _native_meshing_combination(
            "planar-affine-generation",
            "NativePlanarSource:piecewise-linear-loops-and-embedded-segments",
            "planar_constrained_delaunay:generate-certify-publish",
            "triangle",
            1,
            "region-and-edge-uniform-sizing+protected-corners-and-curves+holes",
            "host-in-process",
            "none-through-topology",
            "none-generation-only",
            admission="implemented-admitted",
            source_references=("phydrax/meshing/providers/_native_planar.py",),
            required_evidence=(
                "hole-area-and-embedded-segment-coverage",
                "planar-manufactured-pde",
            ),
            nonclaims=("no-region-patch-periodic-or-layer-controls",),
        ),
        _native_meshing_combination(
            "curve-affine-generation",
            "NativeCurveSource:parametric-atlas-or-straight-segment-network",
            "curve_arc_length:generate-certify-publish",
            "interval",
            1,
            "uniform-or-curvature-sizing+protected-curves+shared-junctions",
            "host-in-process",
            "none-through-topology",
            "none-generation-only",
            admission="implemented-admitted",
            source_references=("phydrax/meshing/providers/_native_curve.py",),
            required_evidence=("independent-arc-length-and-continuous-chord-deviation",),
            nonclaims=("no-proximity-sizing-or-high-order-interval-generation",),
        ),
        _native_meshing_combination(
            "parametric-affine-surface",
            "NativeSurfaceSource:revision-bound-patches-curves-and-corners",
            "parametric_surface:shared-curve-chart-generation-certify-publish",
            "triangle",
            1,
            "uniform-patch-and-curve-sizing+trims+seams+poles+protected-strata",
            "host-in-process",
            "none-through-topology",
            "none-generation-only",
            admission="implemented-admitted",
            source_references=(
                "phydrax/meshing/providers/_native_surface.py",
                "phydrax/meshing/_domain.py",
            ),
            required_evidence=(
                "trim-and-shared-curve-coverage",
                "continuous-patch-fidelity",
                "surface-manufactured-pde",
            ),
            nonclaims=(
                "no-material-interface-region-patch-periodic-or-layer-controls",
                "sampled-fidelity-is-not-a-continuous-certificate",
            ),
        ),
        _native_meshing_combination(
            "implicit-affine-surface",
            "NativeImplicitSource:compiled-three-dimensional-field-and-lattice",
            "implicit_surface:adaptive-discovery-generate-certify-publish",
            "triangle",
            1,
            "one-whole-surface-uniform-size+whole-surface-fidelity",
            "host-in-process",
            "none-through-adaptive-discovery",
            "none-generation-only",
            admission="implemented-admitted",
            source_references=("phydrax/meshing/providers/_implicit.py",),
            required_evidence=(
                "source-bound-enclosure-completeness-and-zero-set-topology",
            ),
            nonclaims=(
                "no-volume-generation-implied-by-surface-route",
                "no-quality-target-material-periodic-or-layer-controls",
            ),
        ),
        _native_meshing_combination(
            "plc-affine-generation",
            "NativePlcSource:oriented-nonconvex-plc-with-cavity-and-material-facets",
            "plc_tetrahedral:recover-classify-refine-certify-publish",
            "tetrahedron",
            1,
            "whole-domain-uniform-size+protected-plc-entities+region-seeds+enabled-exact-region-material-role-controls+facet-adjacency-patches",
            "host-in-process",
            "none-through-topology",
            "none-generation-only",
            admission="implemented-admitted",
            source_references=(
                "phydrax/meshing/providers/_native_volume.py",
                "phydrax/meshing/_volume_generation.py",
            ),
            required_evidence=(
                "immutable-and-subdividable-plc-recovery",
                "sliver-and-hard-quality-tails",
                "independent-cavity-material-volume-and-interface-coverage",
                "region-material-role-and-facet-adjacency-preservation",
                "contradictory-region-seed-and-patch-adjacency-refusal",
                "manufactured-diffusion",
            ),
            nonclaims=(
                "no-periodic-or-layer-controls",
                "disabled-declared-region-controls-not-admitted",
                "delaunay-status-is-not-constraint-recovery-or-sliver-certification",
            ),
        ),
        _native_meshing_combination(
            "occupied-image-compartment-construction",
            "CompartmentMeshingSource:occupied-image-cell-material-interpretation",
            "prepare_compartment_complex+generate_compartment_volume",
            "tetrahedron",
            1,
            "shared-material-facets-and-junctions+physical-image-affine+declared-outer-surface",
            "host-in-process",
            "none-through-label-topology",
            "none-construction-only-current-region-evidence",
            admission="implemented-admitted",
            source_references=("phydrax/meshing/_compartments.py",),
            required_evidence=(
                "whole-cell-material-and-independent-outer-coverage",
                "oblique-reflected-affine-and-tiny-material-positive-cases",
            ),
            nonclaims=(
                "construction-record-is-not-provider-CellMeshingResult",
                "occupied-image-cell-interpretation-is-not-interpolated-label-or-implicit-domain-semantics",
            ),
        ),
        _native_meshing_combination(
            "occupied-image-compartment-publication",
            "CompartmentMeshingSource:occupied-image-cell-material-interpretation",
            "image_material_tetrahedral:generate-certify-publish",
            "tetrahedron",
            1,
            "whole-image-uniform-size+enabled-exact-region-scopes+ordered-interface-patches+shared-junctions",
            "host-in-process",
            "none-through-label-topology",
            "none-generation-only-current-region-evidence",
            admission="implemented-admitted",
            source_references=("phydrax/meshing/providers/_native_compartment.py",),
            required_evidence=(
                "whole-cell-material-outer-and-interface-coverage",
                "current-source-region-evidence-and-neurofluid-readmission",
            ),
            nonclaims=(
                "no-periodic-or-layer-controls",
                "no-interpolated-label-or-implicit-domain-semantics",
                "multiregion-exterior-patch-requires-region-resolved-scopes",
            ),
        ),
        _native_meshing_combination(
            "surface-envelope-affine-publication",
            "NativeSurfaceEnvelopeSource:accepted-unsigned-distance-envelope-and-explicit-solid-region",
            "surface_envelope_tetrahedral:fill-certify-publish",
            "tetrahedron",
            1,
            "repaired-source-identity+whole-domain-uniform-size+explicit-topology-feature-change-permissions+enabled-exact-region-controls+facet-adjacency-patches",
            "host-in-process",
            "none-through-repair-topology",
            "none-generation-only",
            admission="implemented-admitted",
            source_references=(
                "phydrax/meshing/_surface_envelope.py",
                "phydrax/meshing/providers/_native_sources.py",
            ),
            required_evidence=(
                "two-directed-distance-bounds-and-source-inclusion",
                "retained-repair-evidence-topology-change-and-volume-coverage",
            ),
            nonclaims=(
                "unsigned-thickening-does-not-infer-the-intended-interior-of-a-dirty-shell",
                "not-exact-original-source-conformity",
                "no-periodic-or-layer-controls",
                "original-soup-semantics-not-inferred-for-repaired-region-and-facet-controls",
            ),
        ),
        _native_meshing_combination(
            "structured-hexahedron-publication",
            "NativeStructuredSource:explicit-six-face-blocks-and-authoritative-boundary-query",
            "structured_transfinite:conforming-glue-certify-publish",
            "hexahedron",
            1,
            "uniform-sizing+exact-logical-counts+conforming-block-interfaces+fixed-boundary",
            "host-preparation+jax-interior-optimization",
            "none-through-construction",
            "none-generation-only",
            admission="implemented-admitted",
            source_references=(
                "phydrax/meshing/providers/_native_structured.py",
                "phydrax/meshing/_multiblock.py",
            ),
            required_evidence=(
                "independent-source-boundary-fidelity-and-global-embedding",
                "shared-face-orientation-and-pure-hex-family",
                "fem-fv-manufactured-solve",
            ),
            nonclaims=(
                "no-source-specific-region-patch-periodic-layer-or-volume-seed-controls",
                "no-nonsweepable-all-hex-claim",
            ),
        ),
        _native_meshing_combination(
            "swept-hexahedron-publication",
            "NativeSweepSource:pure-quad-profile-explicit-extrusion-and-authoritative-boundary-query",
            "sweep:extrude-certify-publish",
            "hexahedron",
            1,
            "uniform-sizing+physical-layer-stations+fixed-profile-identity",
            "host-in-process",
            "none-through-construction",
            "none-generation-only",
            admission="implemented-admitted",
            source_references=(
                "phydrax/meshing/providers/_native_structured.py",
                "phydrax/meshing/_sweep.py",
            ),
            required_evidence=(
                "independent-source-boundary-fidelity-volume-and-global-embedding",
                "fem-fv-manufactured-solve",
            ),
            nonclaims=(
                "no-source-specific-region-patch-periodic-layer-or-volume-seed-controls",
                "no-automatic-cap-matching-or-nonsweepable-all-hex-claim",
            ),
        ),
        _native_meshing_combination(
            "planar-dual-quadrilateral-generation",
            "NativePlanarSource:piecewise-linear-loops-and-embedded-segments",
            "planar_dual_quad:constrained-triangulate-dual-extract-certify-publish",
            "quadrilateral",
            1,
            "region-and-edge-uniform-sizing+protected-corners-and-curves+holes+subdividable-boundary",
            "host-in-process",
            "none-through-topology",
            "none-generation-only-subdivision-lineage",
            admission="implemented-admitted",
            source_references=(
                "phydrax/meshing/providers/_native_dual.py",
                "phydrax/meshing/_quad_generation.py",
            ),
            required_evidence=(
                "independent-bilinear-jacobian-global-embedding-and-boundary-coverage",
                "pure-quad-family-and-protected-feature-subdivision",
                "planar-manufactured-pde",
            ),
            nonclaims=(
                "no-region-patch-periodic-or-layer-controls",
                "no-curved-integer-grid-or-immutable-boundary-extraction-claim",
            ),
        ),
        _native_meshing_combination(
            "plc-dual-hexahedron-generation",
            "NativePlcSource:oriented-nonconvex-plc-with-subdividable-facets",
            "plc_dual_hex:multizone-tetrahedral-dual-extract-certify-publish",
            "hexahedron",
            1,
            "whole-domain-uniform-size+protected-plc-entities+material-regions+exact-facet-adjacency+pure-family",
            "host-in-process",
            "none-through-topology",
            "none-generation-only-subdivision-lineage",
            admission="implemented-admitted",
            source_references=(
                "phydrax/meshing/providers/_native_dual.py",
                "phydrax/meshing/_hex_generation.py",
            ),
            required_evidence=(
                "independent-trilinear-jacobian-global-embedding-and-domain-volume",
                "pure-hex-family-material-incidence-and-subdivided-boundary-coverage",
                "fem-fv-manufactured-solve",
            ),
            nonclaims=(
                "fixed-immutable-triangular-plc-boundaries-not-admitted",
                "no-periodic-layer-or-curved-general-all-hex-combination-claim",
                "affine-dual-subdivision-does-not-close-the-curved-general-all-hex-research-corpus",
            ),
        ),
        _native_meshing_combination(
            "plc-balanced-grid-hexahedron-generation",
            "NativePlcSource:oriented-nonconvex-plc-with-subdividable-facets",
            "plc_balanced_grid_hex:balanced-grid-reconcile-place-certify-publish",
            "hexahedron",
            1,
            "whole-domain-uniform-size+protected-plc-entities+material-regions+facet-adjacency+pure-family+balanced-grid-schedule",
            "host-in-process",
            "none-through-integer-topology",
            "none-generation-only",
            admission="implemented-admitted",
            source_references=(
                "phydrax/meshing/providers/_native_dual.py",
                "phydrax/meshing/_hex_generation.py",
            ),
            required_evidence=(
                "balanced-grid-transition-conformity-and-positive-map",
                "independent-material-cavity-volume-and-source-coverage",
                "fem-fv-manufactured-solve",
            ),
            nonclaims=(
                "fixed-immutable-triangular-plc-boundaries-not-admitted",
                "no-periodic-layer-or-curved-coordinate-map-combination-claim",
            ),
        ),
        _native_meshing_combination(
            "plc-frame-grid-hexahedron-generation",
            "NativePlcSource:oriented-nonconvex-plc-with-subdividable-facets",
            "plc_frame_grid_hex:frame-field-integer-extract-place-certify-publish",
            "hexahedron",
            1,
            "whole-domain-uniform-size+protected-plc-entities+material-regions+facet-adjacency+pure-family+frame-grid-schedule",
            "host-in-process",
            "none-through-integer-topology",
            "none-generation-only",
            admission="implemented-admitted",
            source_references=(
                "phydrax/meshing/providers/_native_dual.py",
                "phydrax/meshing/_hex_generation.py",
            ),
            required_evidence=(
                "frame-field-integer-transition-and-singularity-consistency",
                "independent-global-embedding-material-cavity-volume-and-source-coverage",
                "fem-fv-manufactured-solve",
            ),
            nonclaims=(
                "fixed-immutable-triangular-plc-boundaries-not-admitted",
                "no-periodic-layer-or-curved-coordinate-map-combination-claim",
            ),
        ),
        _native_meshing_combination(
            "mapped-balanced-grid-hexahedron-generation",
            "NativeMappedHexSource:independent-reference-plc-and-degree-two-MappedReferenceDomain",
            "mapped_balanced_grid_hex:reference-grid-exact-root-map-certify-publish",
            "hexahedron",
            2,
            "one-uniform-physical-size+exact-mapped-root-feature-ids+source-bound-region-labels+balanced-grid-schedule",
            "host-in-process",
            "fixed-source-map-only-no-topology-gradient",
            "none-generation-only-preserved-CellGeometrySpec",
            admission="implemented-admitted",
            source_references=(
                "phydrax/meshing/providers/_native_dual.py",
                "phydrax/meshing/providers/_native_sources.py",
                "phydrax/meshing/_hex_generation.py",
            ),
            required_evidence=(
                "exact-source-degree-and-root-domain-binding",
                "continuous-mapped-domain-coverage-and-original-source-fidelity",
                "independent-positive-jacobian-global-embedding-and-physical-pde",
            ),
            nonclaims=(
                "no-physical-region-hole-seeds-without-owning-inverse-query",
                "no-mapped-region-patch-periodic-or-layer-controls",
                "nodal-residual-and-creator-assertion-are-not-coverage-certificates",
            ),
        ),
        _native_meshing_combination(
            "mapped-frame-grid-hexahedron-generation",
            "NativeMappedHexSource:independent-reference-plc-and-degree-two-MappedReferenceDomain",
            "mapped_frame_grid_hex:reference-frame-grid-exact-root-map-certify-publish",
            "hexahedron",
            2,
            "one-uniform-physical-size+exact-mapped-root-feature-ids+source-bound-region-labels+frame-grid-schedule",
            "host-in-process",
            "fixed-source-map-only-no-topology-gradient",
            "none-generation-only-preserved-CellGeometrySpec",
            admission="implemented-admitted",
            source_references=(
                "phydrax/meshing/providers/_native_dual.py",
                "phydrax/meshing/providers/_native_sources.py",
                "phydrax/meshing/_hex_generation.py",
            ),
            required_evidence=(
                "frame-field-integer-transition-and-root-domain-binding",
                "continuous-mapped-domain-coverage-and-original-source-fidelity",
                "independent-positive-jacobian-global-embedding-and-physical-pde",
            ),
            nonclaims=(
                "no-physical-region-hole-seeds-without-owning-inverse-query",
                "no-mapped-region-patch-periodic-or-layer-controls",
                "nodal-residual-and-creator-assertion-are-not-coverage-certificates",
            ),
        ),
        _native_meshing_combination(
            "plc-hex-dominant-generation",
            "NativePlcSource:oriented-nonconvex-plc-with-subdividable-facets",
            "plc_hex_dominant:hex-core-pyramid-simplex-transition-certify-publish",
            "hexahedron+pyramid+tetrahedron",
            1,
            "explicit-mixed-family-policy+uniform-size+material-regions+facet-adjacency+declared-core-fraction",
            "host-in-process",
            "none-through-topology",
            "none-generation-only",
            admission="implemented-admitted",
            source_references=("phydrax/meshing/providers/_native_dual.py",),
            required_evidence=(
                "explicit-family-counts-and-transition-face-conformity",
                "independent-domain-coverage-and-material-volume",
                "mixed-family-fem-fv-consumer",
            ),
            nonclaims=(
                "hex-dominant-output-does-not-satisfy-a-pure-all-hex-request",
                "fixed-immutable-triangular-plc-boundaries-not-admitted",
                "no-periodic-or-layer-controls",
            ),
        ),
        _native_meshing_combination(
            "periodic-affine-triangle-generation",
            "NativePeriodicSource:translational-quotient-triangle-carrier",
            "periodic_delaunay:generate-certify-publish",
            "triangle",
            1,
            "whole-region-uniform-size+represented-corner-edge-orbits+represented-material-regions",
            "host-in-process",
            "none-through-topology",
            "none-generation-only",
            admission="implemented-admitted",
            source_references=("phydrax/meshing/providers/_native_periodic.py",),
            required_evidence=(
                "quotient-incidence-and-measure",
                "repeated-representatives-and-winding-edges",
                "native-publication-reload-and-periodic-pde",
            ),
            nonclaims=(
                "no-additional-periodic-constraints-patch-layer-or-high-order-controls",
            ),
        ),
        _native_meshing_combination(
            "periodic-affine-tetrahedron-generation",
            "NativePeriodicSource:translational-quotient-tetrahedron-carrier",
            "periodic_delaunay:generate-certify-publish",
            "tetrahedron",
            1,
            "whole-region-uniform-size+represented-corner-edge-orbits+represented-material-regions",
            "host-in-process",
            "none-through-topology",
            "none-generation-only",
            admission="implemented-admitted",
            source_references=("phydrax/meshing/providers/_native_periodic.py",),
            required_evidence=(
                "quotient-incidence-volume-and-region-coverage",
                "native-publication-reload-and-periodic-pde",
            ),
            nonclaims=(
                "no-additional-periodic-constraints-patch-layer-or-high-order-controls",
            ),
        ),
        _native_meshing_combination(
            "plc-affine-polyhedral-generation",
            "NativePlcSource:oriented-nonconvex-plc-with-cavity-and-material-facets",
            "plc_restricted_power:generate-certify-publish",
            "polyhedron",
            1,
            "whole-domain-uniform-size+protected-plc-entities+region-seeds+enabled-exact-region-material-role-controls+facet-adjacency-patches+unweighted-native-sites",
            "host-in-process",
            "none-through-topology",
            "none-generation-only",
            admission="implemented-admitted",
            source_references=(
                "phydrax/meshing/providers/_native_polyhedral.py",
                "phydrax/meshing/_polyhedral_generation.py",
            ),
            required_evidence=(
                "independent-volume-moments-reciprocity-and-conditioning",
                "nonconvex-material-coverage-and-hard-diameter",
                "region-material-role-and-facet-adjacency-preservation",
            ),
            nonclaims=(
                "no-periodic-layer-or-curved-map-controls",
                "disabled-declared-region-controls-not-admitted",
                "no-vem-fv-or-remap-evidence-inferred-from-generation",
            ),
        ),
        _native_meshing_combination(
            "affine-prism-tetrahedron-layer-core",
            "NativeLayerCoreSource:accepted-prism-layers-and-fixed-triangulated-core-plc",
            "layer_core:immutable-cap-recovery-fill-certify-publish",
            "prism+tetrahedron",
            1,
            "whole-core-uniform-size+exact-cap-vertex-ids+oriented-facets+explicit-material-incidence",
            "host-in-process",
            "none-through-topology",
            "none-generation-only",
            admission="implemented-admitted",
            source_references=(
                "phydrax/meshing/providers/_native_layer.py",
                "phydrax/meshing/_layer_core.py",
            ),
            required_evidence=(
                "exact-immutable-cap-identity-and-no-gap-duplicate",
                "combined-domain-material-volume-and-interface-coverage",
            ),
            nonclaims=(
                "no-additional-region-patch-or-periodic-controls",
                "supplied-accepted-layer-realization-is-not-automatic-wall-to-layer-admission",
            ),
        ),
        _native_meshing_combination(
            "structured-quadrilateral-construction",
            "TransfiniteBlock:four-oriented-native-boundary-curves",
            "generate_structured_block:coons-map-and-independent-certificates",
            "quadrilateral",
            1,
            "exact-opposing-interval-counts+fixed-boundaries+bounded-elliptic-placement",
            "host-preparation+jax-interior-optimization",
            "none-through-construction",
            "none-construction-only",
            admission="implemented-admitted",
            source_references=("phydrax/meshing/_structured.py",),
            required_evidence=(
                "independent-bilinear-jacobian-global-embedding-and-boundary-residual",
                "pure-quad-fem-manufactured-solve",
            ),
            nonclaims=(
                "construction-record-is-not-provider-CellMeshingResult",
                "no-general-quad-or-curved-coordinate-element-claim",
            ),
        ),
        _native_meshing_combination(
            "structured-hexahedron-construction",
            "TransfiniteBlock:six-oriented-native-boundary-surface-patches",
            "generate_structured_block:gordon-hall-map-and-independent-certificates",
            "hexahedron",
            1,
            "exact-logical-face-edge-counts+fixed-boundaries+bounded-elliptic-placement",
            "host-preparation+jax-interior-optimization",
            "none-through-construction",
            "none-construction-only",
            admission="implemented-admitted",
            source_references=("phydrax/meshing/_structured.py",),
            required_evidence=(
                "independent-trilinear-jacobian-global-embedding-and-boundary-residual",
                "pure-hex-fem-fv-manufactured-solve",
            ),
            nonclaims=(
                "construction-record-is-not-provider-CellMeshingResult",
                "no-nonsweepable-all-hex-or-curved-coordinate-element-claim",
            ),
        ),
        _native_meshing_combination(
            "swept-hexahedron-construction",
            "CellMesh:pure-quadrilateral-profile-and-explicit-extrusion-map",
            "generate_sweep:extrude-and-independent-certificates",
            "hexahedron",
            1,
            "physical-layer-stations+explicit-axis-origin+shared-source-vertex-identities",
            "host-in-process",
            "none-through-construction",
            "none-construction-only",
            admission="implemented-admitted",
            source_references=("phydrax/meshing/_sweep.py",),
            required_evidence=(
                "positive-sweep-jacobian-volume-and-global-embedding",
                "pure-hex-fem-fv-manufactured-solve",
            ),
            nonclaims=(
                "construction-record-is-not-provider-CellMeshingResult",
                "no-automatic-cap-matching-or-nonsweepable-all-hex-claim",
            ),
        ),
        _native_meshing_combination(
            "planar-metric-transfer",
            "CellMeshingResult:affine-planar-triangle",
            "prepare_mesh_adaptation+execute_mesh_adaptation:native_metric_2d",
            "triangle",
            1,
            "spd-anisotropic-vertex-metric+protected-feature-classes+organization",
            "host-in-process",
            "fixed-topology-only-no-event-gradient",
            "p1-vertex-transfer-with-measured-integral-defect",
            admission="implemented-admitted",
            source_references=(
                "phydrax/meshing/_adaptation.py",
                "phydrax/meshing/_local_metric.py",
            ),
            required_evidence=(
                "metric-unit-length-and-quality",
                "constant-linear-reproduction-and-integral-change",
                "time-to-manufactured-error-vs-uniform",
            ),
            nonclaims=(
                "p1-interpolation-is-not-arbitrary-conservative-state-remap",
                "no-three-dimensional-or-curved-metric-route-implied",
            ),
        ),
        _native_meshing_combination(
            "tetrahedron-bisection-transfer",
            "CellMeshingResult:affine-tetrahedra-and-complete-bisection-hierarchy",
            "prepare_mesh_adaptation+execute_mesh_adaptation:native_bisection-refine-coarsen",
            "tetrahedron",
            1,
            "marked-cells+feature-protection+patch-zone-label-lineage",
            "host-in-process",
            "fixed-epoch-only-no-topology-gradient",
            "p1-nested-topology-transfer",
            admission="implemented-admitted",
            source_references=(
                "phydrax/meshing/_adaptation.py",
                "phydrax/meshing/_bisection.py",
            ),
            required_evidence=(
                "complete-family-coarsening-and-refine-coarsen-roundtrip",
                "constant-linear-reproduction-and-lineage-completeness",
            ),
            nonclaims=(
                "boundary-models-and-attributes-require-explicit-remap",
                "no-arbitrary-field-material-history-transfer-implied",
            ),
        ),
        _native_meshing_combination(
            "mandatory-planar-curved-features",
            "native-curves-and-trimmed-surface-domain",
            "generate-curve-adapt-certify-solve",
            "quadrilateral",
            2,
            "curved-protected-features+trim-holes+exact-pure-family",
            "host-in-process",
            "fixed-epoch-geometry-and-pde-only",
            "h1-field-and-geometry-preserving-transition",
            admission="mandatory-combination-not-closed",
            source_references=(
                "phydrax/meshing/_quad_generation.py",
                "phydrax/meshing/_curving.py",
            ),
            required_evidence=(
                "feature-conforming-nontrivial-quad-positive-corpus",
                "continuous-curved-global-embedding",
                "planar-and-surface-pde-convergence",
            ),
            nonclaims=(
                "affine-barycentric-dual-extraction-does-not-close-curved-integer-grid-research",
            ),
        ),
        _native_meshing_combination(
            "mandatory-implicit-volume",
            "certified-implicit-domain-with-complete-enclosure-queries",
            "adaptive-discovery-tetrahedral-generation-metric-adaptation-solve",
            "tetrahedron",
            1,
            "thin-and-hidden-components+anisotropic-metric+hard-source-fidelity",
            "host-in-process",
            "fixed-epoch-field-derivative-only",
            "h1-field+bounded-conservative-cell-content",
            admission="mandatory-combination-not-closed",
            source_references=(
                "phydrax/meshing/_implicit_volume.py",
                "phydrax/meshing/_tetra_metric.py",
            ),
            required_evidence=(
                "complete-box-classification-and-boundary-recovery",
                "hidden-component-and-tangential-zero-positive-cases",
                "source-fidelity-and-physical-error-convergence",
            ),
            nonclaims=("enclosed-workset-is-not-an-accepted-volume-mesh",),
        ),
        _native_meshing_combination(
            "mandatory-labeled-image",
            "physical-affine-labeled-image-with-declared-interpretation",
            "junction-complex-conforming-tetrahedra-multicomponent-transport",
            "tetrahedron",
            1,
            "anisotropic-oblique-voxels+tiny-required-regions+material-junctions",
            "host-in-process",
            "none-through-label-topology",
            "conservative-compartment-inventory-and-interface-flux",
            admission="mandatory-combination-not-closed",
            source_references=(
                "phydrax/meshing/_compartments.py",
                "phydrax/applications/neurofluid/_model.py",
            ),
            required_evidence=(
                "shared-junction-incidence-and-region-coverage",
                "independent-interface-flux-and-inventory-balance",
                "physical-units-and-image-extent-background",
            ),
            nonclaims=(
                "voxel-label-semantics-are-not-continuous-implicit-certification",
            ),
        ),
        _native_meshing_combination(
            "mandatory-compartment-lifecycle",
            "native-compartment-source-and-current-source-region-evidence",
            "generate-refine-coarsen-move-reclassify-readmit-transport",
            "tetrahedron",
            1,
            "material-adjacency+oriented-interfaces+explicit-source-revision-update",
            "host-in-process",
            "legal-fixed-topology-motion-only",
            "bulk-network-and-positive-conservative-inventory",
            admission="mandatory-combination-not-closed",
            source_references=(
                "phydrax/meshing/_compartments.py",
                "phydrax/meshing/_motion.py",
                "phydrax/applications/neurofluid/_model.py",
            ),
            required_evidence=(
                "current-coverage-and-cell-assignment-revalidation",
                "stale-copied-evidence-refusal",
                "ambiguous-reclassification-atomic-rollback",
            ),
        ),
        _native_meshing_combination(
            "mandatory-imperfect-surface",
            "defective-discrete-surface-with-explicit-repair-wrap-or-envelope-policy",
            "repair-envelope-native-volume-certify-consumer-solve",
            "tetrahedron",
            1,
            "declared-topology-change+two-sided-envelope-distance+source-revision",
            "host-in-process",
            "none-through-repair-topology",
            "consumer-state-with-explicit-geometry-change",
            admission="mandatory-combination-not-closed",
            source_references=(
                "phydrax/meshing/_surface_envelope.py",
                "phydrax/meshing/providers/_native_sources.py",
                "phydrax/meshing/_volume_generation.py",
            ),
            required_evidence=(
                "independent-two-sided-distance-and-topology-change",
                "nonwatertight-positive-envelope-corpus",
                "volume-and-consumer-solve",
            ),
            nonclaims=(
                "exact-source-conformity-not-inferred-from-envelope-success",
                "underspecified-defects-remain-explicit-research-obligations",
            ),
        ),
        _native_meshing_combination(
            "mandatory-native-cad",
            "native-step-iges-brep-with-exact-geometry-topology-units-and-occurrences",
            "construct-decode-boolean-partition-query-mesh-curve-solve",
            "tetrahedron",
            2,
            "trims+material-partition+nonrational-quadric-intersection+lineage",
            "host-in-process",
            "fixed-branch-certified-query-only",
            "association-preserving-h1-field-transfer",
            admission="mandatory-combination-not-closed",
            source_references=(
                "phydrax/geometry/brep/_model.py",
                "phydrax/meshing/_curving.py",
                "phydrax/interchange/_catalog.py",
            ),
            required_evidence=(
                "native-persistence-and-step-iges-independent-reader-roundtrips",
                "complete-intersection-branches-and-pcurve-consistency",
                "exact-export-refusal-and-explicit-bounded-approximation",
                "engine-free-high-order-manufactured-solve",
            ),
            nonclaims=(
                "local-projection-residual-is-not-complete-intersection-discovery",
                "tessellation-is-not-exact-cad-authority",
                "implicit-curved-brep-text-branches-are-refused-not-fabricated",
                "singular-arrangements-remain-research-obligations",
            ),
        ),
        _native_meshing_combination(
            "mandatory-periodic-combination",
            "translational-quotient-multimaterial-source",
            "generate-publish-reload-orbit-refine-coarsen-solve",
            "tetrahedron",
            2,
            "periodic+multimaterial+curved+protected-winding-edges",
            "host-in-process",
            "fixed-epoch-only",
            "compatible-vector-flux-and-conservative-transfer",
            admission="implemented-admitted",
            source_references=(
                "phydrax/meshing/providers/_native_periodic.py",
                "phydrax/meshing/_periodic.py",
            ),
            required_evidence=(
                "repeated-representatives-and-distinct-winding-edge-publication",
                "quotient-incidence-composition-and-measure",
                "periodic-manufactured-pde-and-commuting-transfer",
            ),
            nonclaims=(
                "affine-periodic-simplex-route-does-not-admit-every-curved-layer-combination",
                "post-hoc-node-matching-is-not-periodic-construction",
                "historical-curved-periodic-narrow-gap-source-remains-an-immutable-exact-negative",
            ),
        ),
        _native_meshing_combination(
            "mandatory-hybrid-layers",
            "native-wall-and-complete-multiregion-core-plc",
            "advance-layers-fixed-cap-core-fill-curve-adapt-solve",
            "prism+pyramid+tetrahedron+hexahedron",
            2,
            "layers+narrow-gap+material-interface+immutable-oriented-cap+feature-protection",
            "host-in-process",
            "fixed-epoch-only",
            "geometry-preserving-mixed-h1-and-conservative-inventory",
            admission="implemented-admitted",
            source_references=(
                "phydrax/meshing/_boundary_layer.py",
                "phydrax/meshing/_layer_core.py",
                "phydrax/meshing/_curving.py",
                "phydrax/meshing/_mixed_adaptation.py",
            ),
            required_evidence=(
                "active-thickness-growth-and-cap-identity",
                "no-gap-duplicate-and-shared-high-order-face-orientation",
                "continuous-interior-jacobian-and-global-embedding",
                "curved-mixed-adaptation-and-flow-compatible-solve",
            ),
            nonclaims=(
                "arbitrary-opposing-feature-merge-and-every-periodic-layer-family-not-claimed",
                "historical-curved-periodic-narrow-gap-source-remains-an-immutable-exact-negative",
                "corrected-exact-x-orbit-source-does-not-generalize-to-arbitrary-periodic-layers",
                "positive-corners-do-not-certify-curved-interiors",
            ),
        ),
        _native_meshing_combination(
            "mandatory-general-all-hex",
            "nonsweepable-multiply-connected-curved-mechanical-part-with-material-assembly",
            "decompose-field-solve-integer-extract-place-certify-fem-fv",
            "hexahedron",
            2,
            "cavities+curved-feature-intersections+material-interfaces+pure-family",
            "host-in-process",
            "none-through-integer-topology",
            "h1-field-and-conservative-fv-remap",
            admission="mandatory-curved-general-all-hex-research-gate-not-closed",
            source_references=(
                "phydrax/meshing/_quad_generation.py",
                "phydrax/meshing/_hex_generation.py",
                "phydrax/meshing/_structured.py",
                "phydrax/meshing/_multiblock.py",
            ),
            required_evidence=(
                "mandatory-nonsweepable-all-hex-positive-corpus",
                "parity-singularity-and-positive-global-embedding",
                "exact-family-policy-and-fem-fv-consumption",
            ),
            nonclaims=(
                "boxes-sweeps-multiblocks-or-dual-subdivision-do-not-close-general-all-hex",
                "no-silent-tetrahedral-or-hex-dominant-substitution",
                "research-gate-remains-incomplete-not-removed-from-mandatory-scope",
            ),
        ),
        _native_meshing_combination(
            "mandatory-polyhedral-system",
            "nonconvex-plc-with-material-interfaces-and-weighted-sites",
            "restricted-power-generate-certify-vem-fv-conservative-remap",
            "polyhedron",
            1,
            "domain-restriction+reciprocal-material-faces+conditioning+positive-volumes",
            "host-in-process",
            "fixed-epoch-only",
            "conservative-cell-content-and-bounded-second-order-remap",
            admission="mandatory-combination-not-closed",
            source_references=(
                "phydrax/meshing/providers/_native_polyhedral.py",
                "phydrax/meshing/_polyhedral_generation.py",
            ),
            required_evidence=(
                "independent-nonconvex-volume-moments-and-face-reciprocity",
                "conditioning-and-vem-fv-manufactured-solve",
                "uncovered-double-covered-remap-refusal",
            ),
            nonclaims=("bounding-box-voronoi-is-not-nonconvex-domain-conformity",),
        ),
        _native_meshing_combination(
            "mandatory-distributed-lifecycle",
            "native-owner-local-non-fully-addressable-plc-volume",
            "generate-refine-coarsen-repartition-transfer-checkpoint-changed-placement-restart-solve",
            "tetrahedron",
            1,
            "cross-owner-cavities+complete-coarsening-families+collective-resource-acceptance",
            "real-multi-process-multi-device-owner-local",
            "fixed-epoch-only",
            "all-field-material-history-and-conservative-fe-fv-state",
            admission="mandatory-distributed-generation-and-coarsening-not-closed",
            source_references=(
                "phydrax/meshing/_device_generation.py",
                "phydrax/meshing/_distribution.py",
                "phydrax/discretization/_cell_mesh.py",
            ),
            required_evidence=(
                "actual-addressable-transfer-and-per-rank-memory-instrumentation",
                "global-logical-identity-and-collective-certificate-coverage",
                "partial-corrupt-shard-refusal-and-changed-placement-continued-solve",
                "strong-weak-scaling-neighbor-traffic-and-migration",
            ),
            nonclaims=(
                "partitioned-storage-or-simulated-placement-is-not-distributed-generation",
                "process-local-device-seed-refinement-is-not-distributed-plc-recovery",
                "no-mandatory-global-host-gather",
            ),
        ),
        _native_meshing_combination(
            "mandatory-moving-overset-interpolative",
            "native-affine-triangle-mesh-assembly-with-declared-solid-and-overset-boundaries",
            "hole-cut-donor-resolve-motion-refresh-rebind-pde-update",
            "triangle",
            1,
            "moving-registrations+fringe-layers+deterministic-donor-priority+orphan-refusal",
            "host-in-process",
            "fixed-registration-field-query-transpose-only",
            "interpolative-pointwise-field-query",
            admission="mandatory-combination-not-closed",
            source_references=(
                "phydrax/meshing/_overset.py",
                "phydrax/meshing/_assembly.py",
            ),
            required_evidence=(
                "independent-hole-and-donor-classification",
                "motion-uncovered-state-transport-and-atomic-rebind",
                "accepted-pde-update-and-transpose-duality",
            ),
            nonclaims=("interpolative-transfer-is-not-conservative",),
        ),
        _native_meshing_combination(
            "mandatory-moving-overset-conservative",
            "native-affine-triangle-mesh-assembly-with-declared-solid-and-overset-boundaries",
            "hole-cut-donor-resolve-common-refinement-motion-rebind-pde-update",
            "triangle",
            1,
            "moving-registrations+fringe-cell-coverage+positive-inventory",
            "host-in-process",
            "fixed-registration-only",
            "separately-certified-common-refinement-cell-average-remap",
            admission="mandatory-combination-not-closed",
            source_references=(
                "phydrax/meshing/_overset.py",
                "phydrax/discretization/finite_volume/_unstructured_remap.py",
            ),
            required_evidence=(
                "common-refinement-coverage-and-inventory-balance",
                "motion-uncovered-state-transport-and-atomic-rebind",
                "accepted-conservative-pde-update",
            ),
            nonclaims=("interpolative-donor-evidence-cannot-certify-conservative-mode",),
        ),
        _native_meshing_combination(
            "mandatory-design-learned-loop",
            "native-geometry-and-prepared-solver-epoch",
            "fixed-epoch-derivative-trusted-proposal-native-event-transfer-independent-reanalysis",
            "triangle",
            1,
            "h-and-anisotropic-metric-alternatives+physical-objective+geometry-state-constraints",
            "host-preparation+jax-fixed-epoch-execution",
            "certified-fixed-epoch-jvp-vjp-no-topology-event-gradient",
            "complete-registered-field-material-history-state",
            admission="mandatory-combination-not-closed",
            source_references=(
                "phydrax/meshing/_decision.py",
                "phydrax/meshing/_proposals.py",
                "phydrax/geometry/design/_qualification.py",
            ),
            required_evidence=(
                "independent-primal-adjoint-directional-derivative-and-branch-margin",
                "native-event-transfer-and-mandatory-independent-physical-reanalysis",
                "preregistered-objective-vs-uniform-and-analytic-adaptation",
                "learned-proposal-trusted-acceptance-and-atomic-rollback",
            ),
            nonclaims=(
                "estimator-output-is-not-measured-physical-improvement",
                "h-p-metric-order-flags-do-not-imply-every-combination",
                "learned-proposals-do-not-replace-certification-or-reanalysis",
            ),
        ),
    )


def builtin_capability_catalog() -> CapabilityCatalog:
    """Return the canonical built-in candidate/research inventory.

    The result is intentionally not a release index and contains no implied release
    claim.  Release remains an authenticated operation over exact profiles.
    """

    declarations: dict[str, CapabilityDeclaration] = {}
    for declaration in (
        *_candidate_declarations(),
        *_operator_declarations(),
        *_rom_declarations(),
        *_research_platform_declarations(),
        *_compressible_kinetic_declarations(),
        *_privacy_declarations(),
        *qualified_omniphysics_declarations(),
        *_application_declarations(),
        *_native_meshing_declarations(),
    ):
        existing = declarations.get(declaration.capability)
        if existing is not None:
            if existing.declaration_id != declaration.declaration_id:
                raise ValueError(
                    f"Conflicting capability declarations for {declaration.capability}."
                )
            continue
        declarations[declaration.capability] = declaration
    return CapabilityCatalog(tuple(declarations.values()))


__all__ = ["builtin_candidate_profiles", "builtin_capability_catalog"]
