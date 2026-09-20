#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Built-in fine-grained closure requirements for omniphysics families."""

from __future__ import annotations

from ._catalog import CapabilityCatalog
from ._closure_matrix import CapabilityClosureMatrix
from ._closure_requirement import CapabilityClosureRequirement, CapabilityGapResolution
from ._closure_taxonomy import (
    CapabilityDepth,
    CarrierRepresentation,
    ClosureDisposition,
    CouplingLocation,
    ExecutionRegime,
    PhysicsField,
    TopologyRegime,
    WorkflowClass,
)
from ._qualified_profiles import _CONTROL_IDS, _REFINEMENT_IDS


_FAMILIES = {
    "materials": (
        (
            PhysicsField.SOLID_MECHANICS,
            PhysicsField.THERMAL,
            PhysicsField.CHEMICAL_SPECIES,
        ),
        (CarrierRepresentation.CONTINUUM_VOLUME, CarrierRepresentation.ATOMISTIC),
        ("exa-ca",),
    ),
    "manufacturing": (
        (PhysicsField.SOLID_MECHANICS, PhysicsField.THERMAL),
        (CarrierRepresentation.CONTINUUM_VOLUME,),
        ("adamantine", "additive-foam"),
    ),
    "frequency": (
        (PhysicsField.ELECTRIC, PhysicsField.MAGNETIC, PhysicsField.ACOUSTIC),
        (CarrierRepresentation.REDUCED_SYSTEM, CarrierRepresentation.NETWORK),
        ("palace",),
    ),
    "population-balance": (
        (PhysicsField.FLUID_MECHANICS, PhysicsField.CHEMICAL_SPECIES),
        (CarrierRepresentation.CONTINUUM_VOLUME, CarrierRepresentation.PARTICLE),
        ("mfix",),
    ),
    "system-modeling": (
        (
            PhysicsField.SOLID_MECHANICS,
            PhysicsField.FLUID_MECHANICS,
            PhysicsField.ELECTRIC,
        ),
        (CarrierRepresentation.NETWORK, CarrierRepresentation.REDUCED_SYSTEM),
        ("project-chrono",),
    ),
    "rheology": (
        (PhysicsField.FLUID_MECHANICS,),
        (CarrierRepresentation.CONTINUUM_VOLUME,),
        ("openfoam",),
    ),
    "interfacial-transport": (
        (PhysicsField.FLUID_MECHANICS, PhysicsField.CHEMICAL_SPECIES),
        (CarrierRepresentation.INTERFACE_SURFACE,),
        ("openfoam",),
    ),
    "structural-dynamics": (
        (PhysicsField.SOLID_MECHANICS, PhysicsField.ACOUSTIC),
        (CarrierRepresentation.CONTINUUM_VOLUME, CarrierRepresentation.REDUCED_SYSTEM),
        ("ross",),
    ),
    "correlation": (
        (PhysicsField.SOLID_MECHANICS, PhysicsField.ACOUSTIC),
        (CarrierRepresentation.REDUCED_SYSTEM,),
        ("pylife", "ross"),
    ),
    "electrohydrodynamics": (
        (PhysicsField.FLUID_MECHANICS, PhysicsField.ELECTRIC, PhysicsField.MAGNETIC),
        (CarrierRepresentation.CONTINUUM_VOLUME, CarrierRepresentation.INTERFACE_SURFACE),
        ("bernaise",),
    ),
    "phoresis": (
        (
            PhysicsField.FLUID_MECHANICS,
            PhysicsField.ELECTRIC,
            PhysicsField.THERMAL,
            PhysicsField.ACOUSTIC,
            PhysicsField.OPTICAL,
        ),
        (CarrierRepresentation.PARTICLE,),
        ("py-stokes",),
    ),
    "smart-materials": (
        (
            PhysicsField.SOLID_MECHANICS,
            PhysicsField.ELECTRIC,
            PhysicsField.MAGNETIC,
            PhysicsField.THERMAL,
        ),
        (CarrierRepresentation.CONTINUUM_VOLUME,),
        ("sfepy",),
    ),
    "chemo-mechanics": (
        (PhysicsField.SOLID_MECHANICS, PhysicsField.CHEMICAL_SPECIES),
        (CarrierRepresentation.CONTINUUM_VOLUME,),
        ("opengeosys",),
    ),
    "tribology": (
        (
            PhysicsField.SOLID_MECHANICS,
            PhysicsField.FLUID_MECHANICS,
            PhysicsField.THERMAL,
        ),
        (CarrierRepresentation.INTERFACE_SURFACE,),
        ("ross",),
    ),
    "thermal": (
        (PhysicsField.THERMAL, PhysicsField.IONIZING_RADIATION),
        (CarrierRepresentation.CONTINUUM_VOLUME, CarrierRepresentation.INTERFACE_SURFACE),
        ("openfoam",),
    ),
    "membranes": (
        (
            PhysicsField.FLUID_MECHANICS,
            PhysicsField.CHEMICAL_SPECIES,
            PhysicsField.ELECTRIC,
        ),
        (CarrierRepresentation.INTERFACE_SURFACE, CarrierRepresentation.NETWORK),
        ("openfoam",),
    ),
    "surface-chemistry": (
        (PhysicsField.CHEMICAL_SPECIES, PhysicsField.THERMAL),
        (CarrierRepresentation.INTERFACE_SURFACE, CarrierRepresentation.NETWORK),
        ("cantera",),
    ),
    "optomechanics": (
        (PhysicsField.OPTICAL, PhysicsField.SOLID_MECHANICS, PhysicsField.THERMAL),
        (CarrierRepresentation.CONTINUUM_VOLUME,),
        ("palace",),
    ),
    "acoustics": (
        (
            PhysicsField.ACOUSTIC,
            PhysicsField.FLUID_MECHANICS,
            PhysicsField.SOLID_MECHANICS,
        ),
        (CarrierRepresentation.CONTINUUM_VOLUME,),
        ("sfepy",),
    ),
    "electrochemistry": (
        (PhysicsField.CHEMICAL_SPECIES, PhysicsField.ELECTRIC, PhysicsField.THERMAL),
        (CarrierRepresentation.CONTINUUM_VOLUME, CarrierRepresentation.INTERFACE_SURFACE),
        ("pybamm",),
    ),
    "process-systems": (
        (
            PhysicsField.FLUID_MECHANICS,
            PhysicsField.THERMAL,
            PhysicsField.CHEMICAL_SPECIES,
        ),
        (CarrierRepresentation.NETWORK,),
        ("cantera",),
    ),
}

_COUPLED_FAMILIES = frozenset(
    (
        "acoustics",
        "chemo-mechanics",
        "electrochemistry",
        "electrohydrodynamics",
        "interfacial-transport",
        "membranes",
        "optomechanics",
        "smart-materials",
    )
)
_REDUCED_FAMILIES = frozenset(
    ("correlation", "frequency", "phoresis", "process-systems", "system-modeling")
)


def _requirement(family, fields, carriers, source_ids, symbols, name, depth):
    return CapabilityClosureRequirement.create(
        f"{family}-{name}",
        minimum_depth=depth,
        physical_fields=fields,
        carriers=carriers,
        coupling_locations=(CouplingLocation.BULK, CouplingLocation.BOUNDARY),
        execution_regimes=(ExecutionRegime.STATIC, ExecutionRegime.TRANSIENT),
        topology_regimes=(TopologyRegime.FIXED,),
        workflow_classes=(WorkflowClass.FORWARD, WorkflowClass.QUALIFICATION),
        required_benchmarks=("docs/data/omniphysics_qualification.json",),
        required_evidence_dimensions=("implementation", "numerical"),
        required_providers=("jax-cpu-arm64",),
        required_public_symbols=symbols,
        required_documents=("docs/guides_omniphysics_program.md",),
        source_ids=source_ids,
        rationale="exact-single-host-implementation-closure",
    )


def builtin_omniphysics_closure_matrices(
    catalog: CapabilityCatalog,
) -> tuple[CapabilityClosureMatrix, ...]:
    matrices = []
    for family, (fields, carriers, source_ids) in _FAMILIES.items():
        declarations = tuple(
            value
            for value in catalog.declarations
            if value.capability.startswith(f"{family}.")
            and value.domain_maturity == "implementation-qualified-candidate"
        )
        capabilities = tuple(value.capability for value in declarations)
        symbols = tuple(
            sorted(
                symbol
                for declaration in declarations
                for symbol in declaration.public_symbols
            )
        )
        implementation_depth = (
            CapabilityDepth.REDUCED_SYSTEM
            if family in _REDUCED_FAMILIES
            else CapabilityDepth.SPATIAL_COUPLED
            if family in _COUPLED_FAMILIES
            else CapabilityDepth.SPATIAL_SINGLE_PHYSICS
        )
        implementation_name = (
            "reduced-execution"
            if implementation_depth is CapabilityDepth.REDUCED_SYSTEM
            else "spatial-execution"
        )
        semantic = _requirement(
            family,
            fields,
            carriers,
            source_ids,
            symbols,
            "analytic-control",
            CapabilityDepth.ANALYTIC_CONTROL,
        )
        implementation = _requirement(
            family,
            fields,
            carriers,
            source_ids,
            symbols,
            implementation_name,
            implementation_depth,
        )
        evidence_ids = (_CONTROL_IDS[family],)
        if family in _REFINEMENT_IDS:
            evidence_ids = (*evidence_ids, _REFINEMENT_IDS[family])
        implemented = bool(capabilities)
        actual_depth = implementation_depth

        def resolution(requirement):
            return CapabilityGapResolution.create(
                requirement.requirement_id,
                ClosureDisposition.IMPLEMENTED
                if implemented
                else ClosureDisposition.MISSING,
                actual_depth if implemented else CapabilityDepth.SEMANTIC,
                capability_ids=capabilities,
                evidence_ids=evidence_ids if implemented else (),
                provider_ids=("jax-cpu-arm64",) if implemented else (),
                rationale="exact-qualified-candidate-implemented"
                if implemented
                else "native-owner-missing",
                release_authorized=False,
            )

        matrices.append(
            CapabilityClosureMatrix.create(
                family,
                (semantic, implementation),
                (resolution(semantic), resolution(implementation)),
            )
        )
    return tuple(matrices)


__all__ = ["builtin_omniphysics_closure_matrices"]
