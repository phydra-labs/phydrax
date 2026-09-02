#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reacting-flow commercial qualification candidate producer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from phydrax.qualification._reference import ReferenceArtifactManifest
from tools._commercial_qualification import (
    assemble_candidate_profile,
    availability_observation,
    build_cli_parser,
    GateDefinition,
    make_candidate_artifact,
    RouteDefinition,
    run_cli,
    with_observation,
)


CAPABILITY = "reacting-flow"


def _gate(name: str, category: str, description: str, /) -> GateDefinition:
    return GateDefinition(name, category, description)


_RESOURCE_GATE = _gate(
    "resource-preflight",
    "performance",
    "The exact species, reaction, state, and workspace capacities fit declared resources.",
)
_RESTART_GATE = _gate(
    "reacting-restart",
    "operational",
    "Accepted flow, chemistry, runtime, and continuation state restart exactly.",
)


ROUTES: dict[str, RouteDefinition] = {
    "thermodynamics": RouteDefinition(
        "thermodynamics",
        (
            _gate(
                "thermo-energy-inversion",
                "scientific",
                "Mixture temperature inversion closes the internal-energy residual.",
            ),
            _gate(
                "thermo-enthalpy-consistency",
                "scientific",
                "Species and mixture enthalpy retain formation-energy consistency.",
            ),
            _gate(
                "thermo-derivatives",
                "scientific",
                "Caloric and ideal-mixture derivatives match the governed reference.",
            ),
            _RESOURCE_GATE,
            _gate(
                "thermo-domain-admission",
                "operational",
                "Every evaluated state lies in the exact species thermodynamic domain.",
            ),
        ),
        (
            "phydrax.applications.reacting_flow._thermodynamics:ReactingGasModel",
            "phydrax.applications.reacting_flow._thermodynamics:TemperatureInversionEvidence",
        ),
    ),
    "transport": RouteDefinition(
        "transport",
        (
            _gate(
                "transport-mixture-properties",
                "scientific",
                "Viscosity, conductivity, and binary diffusion match governed references.",
            ),
            _gate(
                "transport-diffusive-closure",
                "scientific",
                "Mixture-averaged correction velocity closes total diffusive mass flux.",
            ),
            _gate(
                "transport-stefan-maxwell",
                "scientific",
                "Stefan-Maxwell velocity and conservation residuals meet tolerance.",
            ),
            _RESOURCE_GATE,
            _gate(
                "transport-route-admission",
                "operational",
                "The exact mixture-averaged or Stefan-Maxwell route executes as declared.",
            ),
        ),
        (
            "phydrax.applications.reacting_flow._transport:MixtureAveragedTransportPlan",
            "phydrax.applications.reacting_flow._transport:StefanMaxwellTransportPlan",
            "phydrax.applications.reacting_flow._transport:StefanMaxwellEvidence",
        ),
    ),
    "mechanism": RouteDefinition(
        "mechanism",
        (
            _gate(
                "mechanism-elements",
                "scientific",
                "Compiled reaction rates preserve every declared element.",
            ),
            _gate(
                "mechanism-charge",
                "scientific",
                "Compiled reaction rates preserve declared charge.",
            ),
            _gate(
                "mechanism-energy",
                "scientific",
                "Species production closes the chemical energy source.",
            ),
            _gate(
                "mechanism-reference-rates",
                "scientific",
                "Forward, reverse, and net rates agree with governed references.",
            ),
            _RESOURCE_GATE,
            _gate(
                "mechanism-feature-admission",
                "operational",
                "Every mechanism feature is explicitly supported by the compiled route.",
            ),
        ),
        (
            "phydrax.applications.reacting_flow._mechanism:ChemicalMechanismCompiler",
            "phydrax.applications.reacting_flow._mechanism:CompiledMechanismEvidence",
            "phydrax.applications.reacting_flow._mechanism:ChemicalMechanismFeatureReport",
        ),
    ),
    "state": RouteDefinition(
        "state",
        (
            _gate(
                "state-species-closure",
                "scientific",
                "S-1 independent species reconstruct total density and all mass fractions.",
            ),
            _gate(
                "state-energy-closure",
                "scientific",
                "Primitive-conserved round-trip closes total and internal energy.",
            ),
            _gate(
                "state-admissibility",
                "scientific",
                "Density, species densities, pressure, and temperature remain admissible.",
            ),
            _RESOURCE_GATE,
            _RESTART_GATE,
        ),
        (
            "phydrax.applications.reacting_flow._state:ReactiveConservedLayout",
            "phydrax.applications.reacting_flow._state:ReactiveConservedEvidence",
            "phydrax.applications.reacting_flow._state:ReactiveEulerSystem",
        ),
    ),
    "strang": RouteDefinition(
        "strang",
        (
            _gate(
                "strang-order",
                "scientific",
                "Transport-chemistry-transport refinement attains second order.",
            ),
            _gate(
                "strang-atomic-commit",
                "scientific",
                "A rejected chemistry or transport stage cannot partially commit state.",
            ),
            _gate(
                "strang-conservation",
                "scientific",
                "Accepted macro-steps preserve elements, mass, and total energy.",
            ),
            _RESOURCE_GATE,
            _RESTART_GATE,
        ),
        (
            "phydrax.applications.reacting_flow._advance:ReactiveStrangPlan",
            "phydrax.applications.reacting_flow._advance:ReactiveAdvanceEvidence",
        ),
    ),
    "imex": RouteDefinition(
        "imex",
        (
            _gate(
                "reactive-imex-order",
                "scientific",
                "Reactive IMEX refinement attains its declared temporal order.",
            ),
            _gate(
                "reactive-imex-chemistry-residual",
                "scientific",
                "Implicit trapezoidal chemistry meets its nonlinear residual criterion.",
            ),
            _gate(
                "reactive-imex-conservation",
                "scientific",
                "Transport and implicit chemistry retain element and energy closure.",
            ),
            _RESOURCE_GATE,
            _RESTART_GATE,
        ),
        (
            "phydrax.applications.reacting_flow._advance:ReactiveIMEXPlan",
            "phydrax.applications.reacting_flow._advance:ReactiveAdvanceEvidence",
        ),
    ),
    "cantera-boundary": RouteDefinition(
        "cantera-boundary",
        (
            _gate(
                "cantera-state-reference",
                "scientific",
                "Thermodynamic states match the governed Cantera boundary reference.",
            ),
            _gate(
                "cantera-rate-reference",
                "scientific",
                "Chemical production rates match the governed Cantera boundary reference.",
            ),
            _gate(
                "cantera-feature-report",
                "scientific",
                "Imported features exactly match the supported Cantera YAML subset.",
            ),
            _RESOURCE_GATE,
            _gate(
                "cantera-provider",
                "operational",
                "A real observed Cantera provider executes the host-only reference boundary.",
            ),
            _gate(
                "cantera-nondifferentiable-boundary",
                "operational",
                "Cantera remains outside differentiable and compiled solver execution.",
            ),
        ),
        (
            "phydrax.applications.reacting_flow._cantera:CanteraYAMLAdapter",
            "phydrax.applications.reacting_flow._cantera:CanteraReferenceAdapter",
            "phydrax.applications.reacting_flow._cantera:CanteraReferenceState",
        ),
    ),
    "low-mach": RouteDefinition(
        "low-mach",
        (
            _gate(
                "low-mach-divergence-source",
                "scientific",
                "Thermal and compositional expansion close the low-Mach divergence source.",
            ),
            _gate(
                "low-mach-pressure-separation",
                "scientific",
                "Thermodynamic and mechanical pressures retain separate ownership.",
            ),
            _gate(
                "low-mach-species-energy",
                "scientific",
                "Species and enthalpy transport close mass and energy budgets.",
            ),
            _RESOURCE_GATE,
            _gate(
                "low-mach-route-independence",
                "operational",
                "The reacting low-Mach formulation does not inherit an incompressible MAC route.",
            ),
        ),
        (
            "phydrax.applications.reacting_flow._low_mach:LowMachReactingFormulation",
            "phydrax.applications.reacting_flow._low_mach:LowMachConstraintEvidence",
        ),
    ),
    "statistics": RouteDefinition(
        "statistics",
        (
            _gate(
                "reactive-favre-statistics",
                "scientific",
                "Reactive Favre means and stresses retain density weighting.",
            ),
            _gate(
                "reactive-species-statistics",
                "scientific",
                "Species moments preserve closure and elemental structure.",
            ),
            _gate(
                "reactive-closure-targets",
                "scientific",
                "Instantaneous closure targets preserve species mass and chemical energy.",
            ),
            _RESOURCE_GATE,
            _gate(
                "reactive-statistics-merge",
                "operational",
                "Block statistics merge from additive records without sample reconstruction.",
            ),
        ),
        (
            "phydrax.applications.reacting_flow._statistics:ReactiveFlowStatisticsPlan",
            "phydrax.applications.reacting_flow._statistics:ReactiveClosureTargetPlan",
        ),
    ),
}


def produce_candidate(
    route: str,
    request: Mapping[str, object],
    /,
    *,
    reference_manifest: ReferenceArtifactManifest | Mapping[str, object] | None = None,
    reference_payload: bytes | None = None,
) -> dict[str, object]:
    """Produce one exact reacting-flow candidate artifact."""

    if route not in ROUTES:
        raise ValueError(f"Unknown reacting-flow qualification route {route!r}.")
    prepared = dict(request)
    if route == "cantera-boundary":
        availability = request.get("availability")
        if availability is not None and not isinstance(availability, Mapping):
            raise TypeError("availability must be a mapping.")
        prepared = with_observation(
            prepared,
            "cantera-provider",
            availability_observation(
                availability,
                provider="cantera",
                minimum_devices=1,
                require_hardware=False,
            ),
        )
    return make_candidate_artifact(
        CAPABILITY,
        ROUTES[route],
        prepared,
        reference_manifest=reference_manifest,
        reference_payload=reference_payload,
        extra_record={
            "application": "reacting-flow",
            "inherits_incompressible_dns": False,
            "external_provider_in_solver": False,
        },
    )


def assemble_profile(
    artifacts: Sequence[Mapping[str, object]],
    /,
    *,
    name: str = "reacting-flow.candidate",
    provider: str = "phydrax",
) -> dict[str, object]:
    return assemble_candidate_profile(artifacts, name=name, provider=provider)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_cli_parser(
        "Create unsigned reacting-flow qualification candidates.",
        ROUTES,
        profile_name="reacting-flow.candidate",
    )
    run_cli(parser, ROUTES, CAPABILITY, argv, producer=produce_candidate)


if __name__ == "__main__":
    main()
