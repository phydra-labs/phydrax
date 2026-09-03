#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Commercial incompressible route qualification candidate producer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from phydrax.qualification._reference import ReferenceArtifactManifest
from tools._commercial_qualification import (
    assemble_candidate_profile,
    build_cli_parser,
    GateDefinition,
    make_candidate_artifact,
    RouteDefinition,
    run_cli,
)


CAPABILITY = "incompressible-flow"


def _gate(name: str, category: str, description: str, /) -> GateDefinition:
    return GateDefinition(name, category, description)


_PRESSURE_API = (
    "phydrax.solver._mac_pressure_operator:MACPressureOperatorSpec",
    "phydrax.solver._mac_pressure_operator:MACWeightedPressureAction",
    "phydrax.solver._mac_pressure_operator:execute_weighted_pressure_iteration",
)
_RESOURCE_GATE = _gate(
    "resource-preflight",
    "performance",
    "The prepared route fits its declared memory and iteration resource limits.",
)


ROUTES: dict[str, RouteDefinition] = {
    "weighted-pressure": RouteDefinition(
        "weighted-pressure",
        (
            _gate(
                "weighted-residual",
                "scientific",
                "The closure-aware matrix-free weighted pressure residual meets tolerance.",
            ),
            _gate(
                "coefficient-positivity",
                "scientific",
                "Every face pressure coefficient is finite and strictly positive.",
            ),
            _gate(
                "nullspace-compatibility",
                "scientific",
                "The weighted right-hand side satisfies its declared pressure nullspace.",
            ),
            _RESOURCE_GATE,
            _gate(
                "weighted-route-execution",
                "operational",
                "The selected weighted pressure route executes without route substitution.",
            ),
        ),
        _PRESSURE_API,
    ),
    "open-pressure": RouteDefinition(
        "open-pressure",
        (
            _gate(
                "robin-closure",
                "scientific",
                "Open pressure Robin traces satisfy their declared alpha-beta closure.",
            ),
            _gate(
                "open-boundary-flux",
                "scientific",
                "Open-boundary pressure flux and compatibility defects meet tolerance.",
            ),
            _gate(
                "open-pressure-residual",
                "scientific",
                "The open pressure solve residual meets its exact norm criterion.",
            ),
            _RESOURCE_GATE,
            _gate(
                "open-route-execution",
                "operational",
                "The open-boundary route executes with its prepared closure unchanged.",
            ),
        ),
        _PRESSURE_API + ("phydrax.solver._mac_pressure_operator:MACPressureRobinSide",),
    ),
    "mapped-pressure": RouteDefinition(
        "mapped-pressure",
        (
            _gate(
                "mapped-geometric-conservation",
                "scientific",
                "Mapped pressure geometry satisfies the discrete geometric conservation law.",
            ),
            _gate(
                "mapped-weighted-residual",
                "scientific",
                "The mapped weighted pressure residual meets tolerance.",
            ),
            _gate(
                "mapped-coefficient-epoch",
                "scientific",
                "Pressure coefficients bind the exact mapped geometry epoch.",
            ),
            _RESOURCE_GATE,
            _gate(
                "mapped-route-execution",
                "operational",
                "Mapped pressure executes without falling back to a Cartesian surrogate.",
            ),
        ),
        _PRESSURE_API,
    ),
    "ale-pressure": RouteDefinition(
        "ale-pressure",
        (
            _gate(
                "ale-geometric-conservation",
                "scientific",
                "ALE stage geometry satisfies the discrete geometric conservation law.",
            ),
            _gate(
                "ale-weighted-residual",
                "scientific",
                "The moving-geometry weighted pressure residual meets tolerance.",
            ),
            _gate(
                "ale-epoch-refresh",
                "scientific",
                "Geometry-dependent preconditioning refreshes at the exact accepted epoch.",
            ),
            _RESOURCE_GATE,
            _gate(
                "ale-restart-continuation",
                "operational",
                "ALE pressure restart preserves accepted geometry and pressure epochs.",
            ),
        ),
        _PRESSURE_API + ("phydrax.solver._mac_ale:MACALEGeometryPlan",),
    ),
    "immersed-pressure": RouteDefinition(
        "immersed-pressure",
        (
            _gate(
                "immersed-pressure-residual",
                "scientific",
                "Immersed pressure and constraint residuals meet the bound regime tolerance.",
            ),
            _gate(
                "immersed-load-balance",
                "scientific",
                "Hydrodynamic loads satisfy the immersed action-reaction balance.",
            ),
            _gate(
                "immersed-reference-campaign",
                "scientific",
                "All reference cases required by the exact immersed regime are covered.",
            ),
            _RESOURCE_GATE,
            _gate(
                "immersed-regime-admission",
                "operational",
                "The immersed body regime and reference campaign are admitted exactly.",
            ),
        ),
        _PRESSURE_API
        + (
            "phydrax.applications.incompressible_flow._immersed_support:ImmersedBodyRegimePlan",
            "phydrax.applications.incompressible_flow._immersed_qualification:ImmersedReferenceCampaignPlan",
        ),
    ),
    "mac-controller": RouteDefinition(
        "mac-controller",
        (
            _gate(
                "controller-target-tracking",
                "scientific",
                "The accepted MAC control response meets the declared integral target.",
            ),
            _gate(
                "controller-mass-flux",
                "scientific",
                "Controlled stages retain the discrete mass-flux constraint.",
            ),
            _gate(
                "controller-conditioning",
                "scientific",
                "The finite response basis passes the declared conditioning criterion.",
            ),
            _RESOURCE_GATE,
            _gate(
                "controller-restart",
                "operational",
                "Controller integral and multistep continuation state round-trip exactly.",
            ),
        ),
        (
            "phydrax.applications.incompressible_flow._control:MACFlowControlPlan",
            "phydrax.applications.incompressible_flow._control:PreparedMACFlowControl",
        ),
    ),
    "ou-fluid": RouteDefinition(
        "ou-fluid",
        (
            _gate(
                "ou-covariance",
                "scientific",
                "OU coefficient covariance matches the exact stationary transition law.",
            ),
            _gate(
                "ou-subdivision",
                "scientific",
                "OU realization subdivision preserves the same pathwise endpoint.",
            ),
            _gate(
                "ou-solenoidal-basis",
                "scientific",
                "The forced Fourier basis is Hermitian, solenoidal, and spectrally restricted.",
            ),
            _gate(
                "ou-fluid-energy-budget",
                "scientific",
                "Fluid energy change closes against viscous dissipation and OU work.",
            ),
            _RESOURCE_GATE,
            _gate(
                "ou-realization-replay",
                "operational",
                "Restart and replay retain the exact OU realization and accepted flow state.",
            ),
        ),
        (
            "phydrax.applications.incompressible_flow._forcing:SolenoidalOUForcingPlan",
            "phydrax.applications.incompressible_flow._production:prepare_ou_forced_periodic_method",
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
    """Produce one route-exact incompressible candidate artifact."""

    if route not in ROUTES:
        raise ValueError(f"Unknown incompressible commercial route {route!r}.")
    return make_candidate_artifact(
        CAPABILITY,
        ROUTES[route],
        request,
        reference_manifest=reference_manifest,
        reference_payload=reference_payload,
        extra_record={"application": "incompressible-flow", "route_family": "commercial"},
    )


def assemble_profile(
    artifacts: Sequence[Mapping[str, object]],
    /,
    *,
    name: str = "incompressible-flow.commercial-candidate",
    provider: str = "phydrax",
) -> dict[str, object]:
    return assemble_candidate_profile(artifacts, name=name, provider=provider)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_cli_parser(
        "Create unsigned commercial incompressible qualification candidates.",
        ROUTES,
        profile_name="incompressible-flow.commercial-candidate",
    )
    run_cli(parser, ROUTES, CAPABILITY, argv, producer=produce_candidate)


if __name__ == "__main__":
    main()
