#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Built-in source atlas and closure matrices for the omniphysics program."""

from __future__ import annotations

from ._catalog import CapabilityCatalog
from ._closure import (
    CapabilityClosureMatrix,
    CapabilityClosureRequirement,
    CapabilityGapResolution,
    CarrierRepresentation,
    ClosureDisposition,
    CouplingLocation,
    ExecutionRegime,
    PhysicsField,
    SourceAbsorptionLedger,
    SourceReference,
    SourceReuseClass,
    TopologyRegime,
    WorkflowClass,
)


_SOURCE_SPECS = (
    (
        "adamantine",
        "https://github.com/adamantine-sim/adamantine",
        "Apache-2.0-WITH-LLVM-exception",
        "permissive",
        ("additive-manufacturing", "moving-heat-source", "material-state"),
    ),
    (
        "exa-ca",
        "https://github.com/LLNL/ExaCA",
        "MIT",
        "permissive",
        ("grain-growth", "thermal-history-transfer"),
    ),
    (
        "bernaise",
        "https://github.com/gautelinga/BERNAISE",
        "MIT",
        "permissive",
        ("electrohydrodynamics", "phase-field", "surface-charge"),
    ),
    (
        "py-stokes",
        "https://github.com/rajeshrinet/pystokes",
        "MIT",
        "permissive",
        ("phoresis", "stokesian-dynamics"),
    ),
    (
        "sfepy",
        "https://github.com/sfepy/sfepy",
        "BSD-3-Clause",
        "permissive",
        ("piezoelectricity", "finite-element"),
    ),
    (
        "ross",
        "https://github.com/petrobras/ross",
        "Apache-2.0",
        "permissive",
        ("rotordynamics", "bearings", "seals"),
    ),
    (
        "pylife",
        "https://github.com/boschresearch/pylife",
        "Apache-2.0",
        "permissive",
        ("fatigue", "load-collectives"),
    ),
    (
        "pybamm",
        "https://github.com/pybamm-team/PyBaMM",
        "BSD-3-Clause",
        "permissive",
        ("electrochemistry", "porous-electrode"),
    ),
    (
        "cantera",
        "https://github.com/Cantera/cantera",
        "BSD-3-Clause",
        "permissive",
        ("surface-chemistry", "thermochemistry"),
    ),
    (
        "openfoam",
        "https://github.com/OpenFOAM/OpenFOAM-dev",
        "GPL-3.0-or-later",
        "strong-copyleft",
        ("multiphase-flow", "combustion", "industrial-cfd"),
    ),
    (
        "additive-foam",
        "https://github.com/ORNL/AdditiveFOAM",
        "GPL-3.0-or-later",
        "strong-copyleft",
        ("additive-manufacturing", "heat-source-calibration"),
    ),
    (
        "laserbeam-foam",
        "https://github.com/laserbeamfoam/LaserbeamFoam",
        "GPL-3.0-or-later",
        "strong-copyleft",
        ("melt-pool", "laser-ray-tracing"),
    ),
    (
        "openfast",
        "https://github.com/OpenFAST/openfast",
        "Apache-2.0",
        "permissive",
        ("wind-energy", "aero-hydro-servo-elastic"),
    ),
    (
        "opengeosys",
        "https://github.com/ufz/ogs",
        "BSD-3-Clause",
        "permissive",
        ("thmc", "geotechnical", "subsurface"),
    ),
    (
        "project-chrono",
        "https://github.com/projectchrono/chrono",
        "BSD-3-Clause",
        "permissive",
        ("multibody", "contact", "vehicles"),
    ),
    (
        "mfix",
        "https://github.com/NREL/MFiX",
        "source-available",
        "source-available",
        ("multiphase-reactors", "tfm-dem-pic"),
    ),
    (
        "palace",
        "https://github.com/awslabs/palace",
        "Apache-2.0",
        "permissive",
        ("electromagnetics", "ports", "frequency-domain"),
    ),
    (
        "warpx",
        "https://github.com/BLAST-WarpX/warpx",
        "BSD-3-Clause",
        "permissive",
        ("plasma", "particle-in-cell"),
    ),
    (
        "simvascular",
        "https://github.com/SimVascular/SimVascular",
        "BSD-3-Clause",
        "permissive",
        ("medical-devices", "vascular-flow"),
    ),
    (
        "jsbsim",
        "https://github.com/JSBSim-Team/jsbsim",
        "LGPL-2.1-or-later",
        "weak-copyleft",
        ("flight-dynamics", "aircraft-systems"),
    ),
    (
        "opm-flow",
        "https://github.com/OPM/opm-simulators",
        "GPL-3.0-or-later",
        "strong-copyleft",
        ("reservoir", "wells", "schedules"),
    ),
)


def builtin_source_absorption_ledger() -> SourceAbsorptionLedger:
    return SourceAbsorptionLedger.create(
        tuple(
            SourceReference.create(
                source_id,
                url,
                "unpinned",
                licence,
                SourceReuseClass(reuse),
                concepts=concepts,
                code_inspected=False,
                copying_permitted=SourceReuseClass(reuse)
                in (SourceReuseClass.PERMISSIVE, SourceReuseClass.PUBLIC_DOMAIN),
                provider_only=SourceReuseClass(reuse)
                in (SourceReuseClass.STRONG_COPYLEFT, SourceReuseClass.SOURCE_AVAILABLE),
                reviewer="pending-source-review",
            )
            for source_id, url, licence, reuse, concepts in _SOURCE_SPECS
        )
    )


_FAMILY_SPECS = (
    (
        "materials",
        (
            PhysicsField.SOLID_MECHANICS,
            PhysicsField.THERMAL,
            PhysicsField.CHEMICAL_SPECIES,
        ),
        (CarrierRepresentation.CONTINUUM_VOLUME, CarrierRepresentation.ATOMISTIC),
        "materials",
    ),
    (
        "manufacturing",
        (PhysicsField.SOLID_MECHANICS, PhysicsField.THERMAL),
        (CarrierRepresentation.CONTINUUM_VOLUME,),
        "manufacturing",
    ),
    (
        "rheology",
        (PhysicsField.FLUID_MECHANICS,),
        (CarrierRepresentation.CONTINUUM_VOLUME,),
        "rheology",
    ),
    (
        "interfacial-transport",
        (PhysicsField.FLUID_MECHANICS, PhysicsField.CHEMICAL_SPECIES),
        (CarrierRepresentation.INTERFACE_SURFACE,),
        "interfacial-transport",
    ),
    (
        "electrohydrodynamics",
        (PhysicsField.FLUID_MECHANICS, PhysicsField.ELECTRIC, PhysicsField.MAGNETIC),
        (CarrierRepresentation.CONTINUUM_VOLUME, CarrierRepresentation.INTERFACE_SURFACE),
        "electrohydrodynamics",
    ),
    (
        "phoresis",
        (
            PhysicsField.FLUID_MECHANICS,
            PhysicsField.ELECTRIC,
            PhysicsField.THERMAL,
            PhysicsField.ACOUSTIC,
            PhysicsField.OPTICAL,
        ),
        (CarrierRepresentation.PARTICLE,),
        "phoresis",
    ),
    (
        "smart-materials",
        (
            PhysicsField.SOLID_MECHANICS,
            PhysicsField.ELECTRIC,
            PhysicsField.MAGNETIC,
            PhysicsField.THERMAL,
        ),
        (CarrierRepresentation.CONTINUUM_VOLUME,),
        "smart-materials",
    ),
    (
        "chemo-mechanics",
        (PhysicsField.SOLID_MECHANICS, PhysicsField.CHEMICAL_SPECIES),
        (CarrierRepresentation.CONTINUUM_VOLUME,),
        "chemo-mechanics",
    ),
    (
        "tribology",
        (
            PhysicsField.SOLID_MECHANICS,
            PhysicsField.FLUID_MECHANICS,
            PhysicsField.THERMAL,
        ),
        (CarrierRepresentation.INTERFACE_SURFACE,),
        "tribology",
    ),
    (
        "thermal",
        (PhysicsField.THERMAL, PhysicsField.IONIZING_RADIATION),
        (CarrierRepresentation.CONTINUUM_VOLUME, CarrierRepresentation.INTERFACE_SURFACE),
        "thermal",
    ),
    (
        "electrochemistry",
        (PhysicsField.CHEMICAL_SPECIES, PhysicsField.ELECTRIC, PhysicsField.THERMAL),
        (CarrierRepresentation.CONTINUUM_VOLUME, CarrierRepresentation.INTERFACE_SURFACE),
        "electrochemistry",
    ),
    (
        "acoustics",
        (
            PhysicsField.ACOUSTIC,
            PhysicsField.FLUID_MECHANICS,
            PhysicsField.SOLID_MECHANICS,
        ),
        (CarrierRepresentation.CONTINUUM_VOLUME,),
        "acoustics",
    ),
    (
        "process-systems",
        (
            PhysicsField.FLUID_MECHANICS,
            PhysicsField.THERMAL,
            PhysicsField.CHEMICAL_SPECIES,
        ),
        (CarrierRepresentation.NETWORK,),
        "process-systems",
    ),
    (
        "structural-dynamics",
        (PhysicsField.SOLID_MECHANICS, PhysicsField.ACOUSTIC),
        (CarrierRepresentation.CONTINUUM_VOLUME, CarrierRepresentation.REDUCED_SYSTEM),
        "structural-dynamics",
    ),
    (
        "optomechanics",
        (PhysicsField.OPTICAL, PhysicsField.SOLID_MECHANICS, PhysicsField.THERMAL),
        (CarrierRepresentation.CONTINUUM_VOLUME,),
        "optomechanics",
    ),
)


def builtin_omniphysics_closure_matrices(
    catalog: CapabilityCatalog,
) -> tuple[CapabilityClosureMatrix, ...]:
    matrices = []
    for family, fields, carriers, prefix in _FAMILY_SPECS:
        requirement = CapabilityClosureRequirement.create(
            f"{family}-mainstream-native-surface",
            physical_fields=fields,
            carriers=carriers,
            coupling_locations=(CouplingLocation.BULK, CouplingLocation.BOUNDARY),
            execution_regimes=(ExecutionRegime.STATIC, ExecutionRegime.TRANSIENT),
            topology_regimes=(TopologyRegime.FIXED,),
            workflow_classes=(WorkflowClass.FORWARD, WorkflowClass.QUALIFICATION),
            rationale="mainstream-engineering-closure",
        )
        capabilities = tuple(
            value.capability
            for value in catalog.declarations
            if value.capability.startswith(prefix)
        )
        disposition = (
            ClosureDisposition.CANDIDATE if capabilities else ClosureDisposition.MISSING
        )
        resolution = CapabilityGapResolution.create(
            requirement.requirement_id,
            disposition,
            capability_ids=capabilities,
            rationale="candidate-profile-present"
            if capabilities
            else "native-owner-missing",
        )
        matrices.append(
            CapabilityClosureMatrix.create(family, (requirement,), (resolution,))
        )
    return tuple(matrices)


__all__ = ["builtin_omniphysics_closure_matrices", "builtin_source_absorption_ledger"]
