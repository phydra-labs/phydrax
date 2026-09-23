#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lazy assembly of the repository's built-in capability inventory."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType

from ._catalog import (
    CapabilityCatalog,
    CapabilityDeclaration,
    CapabilityDisposition,
    declarations_from_profiles,
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


def _provider(module_name: str, function_name: str, /) -> Callable[[], object]:
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
        profile = entry.profile(provider="phydrax", version="candidate")
        declarations.append(
            CapabilityDeclaration(
                profile.capability,
                "phydrax.rom",
                disposition,
                domain_maturity=entry.maturity.value,
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
        "2.0.0",
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
