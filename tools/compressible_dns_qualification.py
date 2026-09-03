#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unsigned route qualification for compressible DNS candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from phydrax._fingerprint import canonical_fingerprint
from phydrax.applications.compressible_flow._contracts import (
    CompressibleQualificationEvidence,
)
from phydrax.qualification._reference import ReferenceArtifactManifest
from tools._commercial_qualification import (
    assemble_candidate_profile,
    build_cli_parser,
    GATE_CATEGORIES,
    GateDefinition,
    make_candidate_artifact,
    RouteDefinition,
    run_cli,
)


CAPABILITY = "compressible-dns"


def _gate(name: str, category: str, description: str, /) -> GateDefinition:
    return GateDefinition(name, category, description)


_RESOURCE_GATE = _gate(
    "resource-preflight",
    "performance",
    "The prepared route fits its exact state, stage, fallback, and device resource limits.",
)
_RESTART_GATE = _gate(
    "restart-continuation",
    "operational",
    "Accepted state, runtime ledgers, and temporal continuation restart exactly.",
)


ROUTES: dict[str, RouteDefinition] = {
    "smooth-dgsem": RouteDefinition(
        "smooth-dgsem",
        (
            _gate(
                "dgsem-smooth-wave",
                "scientific",
                "Tensor DGSEM reproduces all admitted smooth Euler wave families.",
            ),
            _gate(
                "dgsem-entropy-balance",
                "scientific",
                "Split-form volume and surface entropy ledgers close to tolerance.",
            ),
            _gate(
                "dgsem-viscous-manufactured",
                "scientific",
                "Viscous DGSEM converges on the manufactured Navier-Stokes source.",
            ),
            _RESOURCE_GATE,
            _RESTART_GATE,
        ),
        (
            "phydrax.applications.compressible_flow._production:SmoothCompressibleProductionPlan",
            "phydrax.applications.compressible_flow._qualification:CompressibleReferenceWavePlan",
            "phydrax.applications.compressible_flow._qualification:ManufacturedViscousNSPlan",
        ),
    ),
    "smooth-fv": RouteDefinition(
        "smooth-fv",
        (
            _gate(
                "fv-smooth-wave",
                "scientific",
                "Structured or mapped FV reproduces admitted smooth Euler waves.",
            ),
            _gate(
                "fv-viscous-manufactured",
                "scientific",
                "Finite-volume viscous residuals converge on the manufactured solution.",
            ),
            _gate(
                "fv-positivity-inactive",
                "scientific",
                "Smooth qualification remains on the primary flux without hidden fallback.",
            ),
            _RESOURCE_GATE,
            _RESTART_GATE,
        ),
        (
            "phydrax.applications.compressible_flow._production:StructuredFVCompressibleProductionPlan",
            "phydrax.applications.compressible_flow._qualification:CompressibleReferenceWavePlan",
            "phydrax.applications.compressible_flow._qualification:ManufacturedViscousNSPlan",
        ),
    ),
    "forcing": RouteDefinition(
        "forcing",
        (
            _gate(
                "forcing-mass-ledger",
                "scientific",
                "Mass injection equals the conservative density source.",
            ),
            _gate(
                "forcing-momentum-ledger",
                "scientific",
                "Momentum injection and body acceleration close exactly.",
            ),
            _gate(
                "forcing-energy-ledger",
                "scientific",
                "Mechanical and thermal forcing close the total-energy work ledger.",
            ),
            _RESOURCE_GATE,
            _RESTART_GATE,
        ),
        ("phydrax.applications.compressible_flow._forcing:CompressibleForcingPlan",),
    ),
    "budgets": RouteDefinition(
        "budgets",
        (
            _gate(
                "mass-budget",
                "scientific",
                "Mass change closes against boundary and source ledgers.",
            ),
            _gate(
                "momentum-budget",
                "scientific",
                "Momentum change closes against flux, stress, and forcing ledgers.",
            ),
            _gate(
                "energy-budget",
                "scientific",
                "Total-energy change closes against heat, work, and boundary ledgers.",
            ),
            _RESOURCE_GATE,
            _RESTART_GATE,
        ),
        ("phydrax.applications.compressible_flow._diagnostics:CompressibleBudgetPlan",),
    ),
    "favre-statistics": RouteDefinition(
        "favre-statistics",
        (
            _gate(
                "favre-merge",
                "scientific",
                "Merged raw moments equal a single full-domain accumulation.",
            ),
            _gate(
                "favre-density-weighting",
                "scientific",
                "Favre means and stresses use exact density-weighted normalization.",
            ),
            _gate(
                "wall-thermal-units",
                "scientific",
                "Wall thermal units retain their declared dimensional normalization.",
            ),
            _RESOURCE_GATE,
            _gate(
                "statistics-restart",
                "operational",
                "Additive raw moments restart without reconstructing samples.",
            ),
        ),
        (
            "phydrax.applications.compressible_flow._diagnostics:CompressiblePlaneStatisticsPlan",
            "phydrax.applications.compressible_flow._diagnostics:CompressibleRawMoments",
        ),
    ),
    "boundaries": RouteDefinition(
        "boundaries",
        (
            _gate(
                "characteristic-wave-selection",
                "scientific",
                "Boundary states freeze incoming and pass outgoing characteristics exactly.",
            ),
            _gate(
                "boundary-reflection",
                "scientific",
                "Measured reflected characteristic energy meets the declared bound.",
            ),
            _gate(
                "boundary-conservation-ledger",
                "scientific",
                "Boundary fluxes close the global conservative budget.",
            ),
            _RESOURCE_GATE,
            _gate(
                "boundary-route-execution",
                "operational",
                "Every face executes the exact configured boundary policy.",
            ),
        ),
        (
            "phydrax.applications.compressible_flow._boundary:CharacteristicNonreflectingBoundaryPlan",
            "phydrax.applications.compressible_flow._boundary:CharacteristicReflectionLedger",
        ),
    ),
    "sponge": RouteDefinition(
        "sponge",
        (
            _gate(
                "sponge-conservative-rate",
                "scientific",
                "Sponge source components equal their conservative relaxation rates.",
            ),
            _gate(
                "sponge-entropy-ledger",
                "scientific",
                "Sponge entropy production is finite and has the admitted sign.",
            ),
            _gate(
                "sponge-interior-invariance",
                "scientific",
                "The zero-profile interior remains unchanged.",
            ),
            _RESOURCE_GATE,
            _RESTART_GATE,
        ),
        ("phydrax.applications.compressible_flow._boundary:CompressibleSpongePlan",),
    ),
    "imex": RouteDefinition(
        "imex",
        (
            _gate(
                "imex-order",
                "scientific",
                "Additive IMEX refinement attains its declared temporal order.",
            ),
            _gate(
                "imex-stiff-stage",
                "scientific",
                "Implicit viscous stages meet their nonlinear and linear residual criteria.",
            ),
            _gate(
                "imex-conservation",
                "scientific",
                "Explicit and implicit stage partitions close the conservative budget.",
            ),
            _RESOURCE_GATE,
            _RESTART_GATE,
        ),
        (
            "phydrax.applications.compressible_flow._production:AdditiveIMEXCompressibleFixedStepAdapter",
            "phydrax.solver._conservation_temporal:ConservationIMEXMethod",
        ),
    ),
    "all-speed": RouteDefinition(
        "all-speed",
        (
            _gate(
                "low-mach-pressure-scaling",
                "scientific",
                "Acoustic dissipation follows the declared O(M) low-Mach scaling.",
            ),
            _gate(
                "all-speed-vortex",
                "scientific",
                "Low-Mach vortical flow avoids spurious pressure contamination.",
            ),
            _gate(
                "all-speed-shock-switch",
                "scientific",
                "Declared shock-ledger selection and stage admissibility choose "
                "the exact robust fallback route.",
            ),
            _RESOURCE_GATE,
            _gate(
                "all-speed-route-label",
                "operational",
                "Every accepted step retains the exact all-speed route label.",
            ),
        ),
        (
            "phydrax.applications.compressible_flow._contracts:AllSpeedCompressiblePolicy",
            "phydrax.applications.compressible_flow._all_speed:AllSpeedHLLFluxPlan",
            "phydrax.applications.compressible_flow._all_speed:ShockAwareAllSpeedFluxPlan",
        ),
    ),
    "boundary-layer": RouteDefinition(
        "boundary-layer",
        (
            _gate(
                "finite-x-inflow",
                "scientific",
                "Finite-x inflow matches its compressible boundary-layer similarity state.",
            ),
            _gate(
                "wall-shear",
                "scientific",
                "Wall shear and skin-friction observables meet reference uncertainty.",
            ),
            _gate(
                "boundary-layer-budgets",
                "scientific",
                "Finite-x boundary-layer integral budgets close to tolerance.",
            ),
            _RESOURCE_GATE,
            _gate(
                "finite-x-boundary-ownership",
                "operational",
                "Inflow, wall, outflow, and spanwise policies retain exact ownership.",
            ),
        ),
        (
            "phydrax.applications.compressible_flow._contracts:FiniteXBoundaryLayerInflowPlan",
            "phydrax.applications.compressible_flow._contracts:FiniteXBoundaryLayerCaseSpec",
        ),
    ),
    "slow-growth": RouteDefinition(
        "slow-growth",
        (
            _gate(
                "slow-growth-chain-rule",
                "scientific",
                "Slow-growth conservative sources satisfy the exact primitive chain rule.",
            ),
            _gate(
                "slow-growth-budget",
                "scientific",
                "Mass, momentum, and energy slow-growth source ledgers close.",
            ),
            _gate(
                "slow-growth-finite-x-comparison",
                "scientific",
                "Modeled temporal slow growth agrees with governed finite-x evidence.",
            ),
            _gate(
                "slow-growth-jvp-vjp",
                "scientific",
                "Analytic JVP and VJP agree with the exact prepared source linearization.",
            ),
            _RESOURCE_GATE,
            _gate(
                "slow-growth-restart",
                "operational",
                "Accepted baseflow continuation and source history restart exactly.",
            ),
        ),
        (
            "phydrax.applications.compressible_flow._slow_growth:TemporalSlowGrowthModelPlan",
            "phydrax.applications.compressible_flow._slow_growth:SpatialSlowGrowthModelPlan",
            "phydrax.applications.compressible_flow._slow_growth:PreparedSlowGrowthSource",
            "phydrax.applications.compressible_flow._slow_growth:SlowGrowthFiniteXEvidence",
        ),
    ),
    "shock": RouteDefinition(
        "shock",
        (
            _gate(
                "shock-conservation",
                "scientific",
                "Shock-tube mass, momentum, and energy remain conservative.",
            ),
            _gate(
                "shock-positivity",
                "scientific",
                "Density and pressure remain admissible through every accepted stage.",
            ),
            _gate(
                "shock-fallback-ledger",
                "scientific",
                "Every pressure-sensor, primary-admissibility, and robust HLL "
                "fallback decision is explicitly recorded.",
            ),
            _gate(
                "shock-reference-error",
                "scientific",
                "Shock location and state errors meet the governed reference uncertainty.",
            ),
            _RESOURCE_GATE,
            _gate(
                "shock-route-execution",
                "operational",
                "The exact sensor, reconstruction, and fallback route are executed.",
            ),
        ),
        (
            "phydrax.applications.compressible_flow._contracts:ShockResolvingPolicy",
            "phydrax.applications.compressible_flow._contracts:ShockRouteLedger",
        ),
    ),
    "material": RouteDefinition(
        "material",
        (
            _gate(
                "material-schema-standard-state",
                "scientific",
                "Component, species, phase, ordering, and gas standard pressure "
                "have one canonical identity.",
            ),
            _gate(
                "material-density-energy-inversion",
                "scientific",
                "Canonical density-energy recovery closes with successful domain evidence.",
            ),
            _gate(
                "material-sound-entropy-characteristics",
                "scientific",
                "Frozen-composition sound speed, entropy variables, and "
                "characteristics derive from the same Helmholtz model.",
            ),
            _gate(
                "material-real-fluid-provenance",
                "scientific",
                "Real-fluid parameters, roots, stability, and phase-equilibrium "
                "evidence retain canonical provenance.",
            ),
            _RESOURCE_GATE,
            _gate(
                "material-domain-admission",
                "operational",
                "The exact homogeneous model, domain, gas system, and phase "
                "boundary are admitted.",
            ),
        ),
        (
            "phydrax.equations._chemical_components:ChemicalComponentCatalog",
            "phydrax.equations._chemical_species:ChemicalSpeciesSchema",
            "phydrax.equations._chemical_thermodynamics:PolynomialSpeciesThermodynamicsPlan",
            "phydrax.equations._homogeneous_thermodynamics:HomogeneousHelmholtzPlan",
            "phydrax.equations._peng_robinson:PengRobinsonResidualHelmholtzTerm",
            "phydrax.equations._gas_dynamics:HomogeneousMixtureEulerSystem",
            "phydrax.solver._phase_equilibrium:FixedTwoPhaseTPFlashPlan",
        ),
    ),
}


def _bind_application_evidence(
    artifact: Mapping[str, object], request: Mapping[str, object], /
) -> dict[str, object]:
    support = artifact["support_tuple"]
    run_spec = artifact["resolved_run_spec"]
    if not isinstance(support, Mapping) or not isinstance(run_spec, Mapping):
        raise TypeError("Candidate admission records must be mappings.")
    attributes = support["attributes"]
    if not isinstance(attributes, Mapping):
        raise TypeError("SupportTuple attributes must be a mapping.")
    method_id = str(attributes.get("method", run_spec["prepared_configuration_id"]))
    case_id = str(request.get("case_id", run_spec["prepared_configuration_id"]))
    gates = artifact["gates"]
    if not isinstance(gates, Mapping):
        raise TypeError("Candidate gates must be a mapping.")
    checks: list[tuple[str, bool]] = []
    for category in GATE_CATEGORIES:
        values = gates[category]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError("Candidate gate categories must be sequences.")
        for value in values:
            if not isinstance(value, Mapping):
                raise TypeError("Candidate gates must be mappings.")
            checks.append((str(value["name"]), value["outcome"] == "passed"))
    route_evidence = CompressibleQualificationEvidence(
        case_id,
        str(artifact["route"]),
        method_id,
        tuple(checks),
    )
    route_record = {
        "kind": "compressible-qualification-evidence",
        "case_id": route_evidence.case_id,
        "route_label": route_evidence.route_label,
        "method_id": route_evidence.method_id,
        "support_tuple_id": route_evidence.support_tuple_id,
        "checks": [list(value) for value in route_evidence.checks],
        "qualification_ready": route_evidence.qualification_ready,
        "dns_claimed": route_evidence.dns_claimed,
        "signed": route_evidence.signed,
        "released": route_evidence.released,
        "evidence_id": route_evidence.evidence_id,
    }
    core = {name: value for name, value in artifact.items() if name != "artifact_id"}
    extra = dict(core["extra"])
    extra["application_route_evidence"] = route_record
    extra["dns_support_inherited"] = False
    if artifact["route"] == "slow-growth":
        extra["claims_spatial_dns"] = False
    core["extra"] = extra
    return {**core, "artifact_id": canonical_fingerprint(core)}


def produce_candidate(
    route: str,
    request: Mapping[str, object],
    /,
    *,
    reference_manifest: ReferenceArtifactManifest | Mapping[str, object] | None = None,
    reference_payload: bytes | None = None,
) -> dict[str, object]:
    """Produce one compressible candidate without inheriting a DNS claim."""

    if route not in ROUTES:
        raise ValueError(f"Unknown compressible DNS qualification route {route!r}.")
    artifact = make_candidate_artifact(
        CAPABILITY,
        ROUTES[route],
        request,
        reference_manifest=reference_manifest,
        reference_payload=reference_payload,
        extra_record={
            "application": "compressible-flow",
            "fidelity": "dns-candidate",
            "dns_support_inherited": False,
        },
    )
    return _bind_application_evidence(artifact, request)


def assemble_profile(
    artifacts: Sequence[Mapping[str, object]],
    /,
    *,
    name: str = "compressible-dns.candidate",
    provider: str = "phydrax",
) -> dict[str, object]:
    return assemble_candidate_profile(artifacts, name=name, provider=provider)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_cli_parser(
        "Create unsigned compressible DNS qualification candidates.",
        ROUTES,
        profile_name="compressible-dns.candidate",
    )
    run_cli(parser, ROUTES, CAPABILITY, argv, producer=produce_candidate)


if __name__ == "__main__":
    main()
