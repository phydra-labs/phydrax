#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


PolymerModelKind = Literal[
    "kremer-grest-particle",
    "bead-spring-slip-spring",
    "hard-sphere-colloid",
    "tube-continuum",
]
EntanglementRoute = Literal["none", "native-ppa", "z1plus"]
ReptationRoute = Literal[
    "none",
    "particle-observables",
    "doi-edwards",
    "likhtman-mcleish",
    "slip-spring",
    "glamm",
]
HydrodynamicRoute = Literal[
    "none",
    "constant",
    "free-space-rpy",
    "periodic-rpy",
    "positive-split-periodic-rpy",
    "confined-fib",
    "rpy-lubrication",
]
DrivenFlowRoute = Literal[
    "equilibrium",
    "overdamped-affine",
    "sllod-shear",
    "lees-edwards-sllod",
    "kr-extension",
    "constitutive-flow",
]


_SUPPORTED: frozenset[tuple[str, str, str, str, str]] = frozenset(
    {
        (
            "kremer-grest-particle",
            "native-ppa",
            "particle-observables",
            "none",
            "equilibrium",
        ),
        (
            "kremer-grest-particle",
            "z1plus",
            "particle-observables",
            "none",
            "equilibrium",
        ),
        (
            "kremer-grest-particle",
            "native-ppa",
            "particle-observables",
            "none",
            "sllod-shear",
        ),
        (
            "kremer-grest-particle",
            "native-ppa",
            "particle-observables",
            "none",
            "lees-edwards-sllod",
        ),
        (
            "kremer-grest-particle",
            "native-ppa",
            "particle-observables",
            "none",
            "kr-extension",
        ),
        (
            "kremer-grest-particle",
            "none",
            "particle-observables",
            "constant",
            "equilibrium",
        ),
        (
            "kremer-grest-particle",
            "none",
            "particle-observables",
            "constant",
            "overdamped-affine",
        ),
        ("bead-spring-slip-spring", "none", "slip-spring", "none", "equilibrium"),
        ("hard-sphere-colloid", "none", "none", "free-space-rpy", "equilibrium"),
        ("hard-sphere-colloid", "none", "none", "free-space-rpy", "overdamped-affine"),
        ("hard-sphere-colloid", "none", "none", "periodic-rpy", "equilibrium"),
        ("hard-sphere-colloid", "none", "none", "periodic-rpy", "overdamped-affine"),
        (
            "hard-sphere-colloid",
            "none",
            "none",
            "positive-split-periodic-rpy",
            "equilibrium",
        ),
        (
            "hard-sphere-colloid",
            "none",
            "none",
            "positive-split-periodic-rpy",
            "overdamped-affine",
        ),
        ("hard-sphere-colloid", "none", "none", "confined-fib", "equilibrium"),
        ("hard-sphere-colloid", "none", "none", "confined-fib", "overdamped-affine"),
        ("hard-sphere-colloid", "none", "none", "rpy-lubrication", "equilibrium"),
        ("hard-sphere-colloid", "none", "none", "rpy-lubrication", "overdamped-affine"),
        ("tube-continuum", "none", "doi-edwards", "none", "equilibrium"),
        ("tube-continuum", "none", "likhtman-mcleish", "none", "equilibrium"),
        ("tube-continuum", "none", "glamm", "none", "constitutive-flow"),
    }
)


class PolymerProductionRegime(StrictModule, NonTrainableState):
    model: PolymerModelKind = eqx.field(static=True)
    entanglement: EntanglementRoute = eqx.field(static=True)
    reptation: ReptationRoute = eqx.field(static=True)
    hydrodynamics: HydrodynamicRoute = eqx.field(static=True)
    driven_flow: DrivenFlowRoute = eqx.field(static=True)
    regime_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: PolymerModelKind,
        entanglement: EntanglementRoute,
        reptation: ReptationRoute,
        hydrodynamics: HydrodynamicRoute,
        driven_flow: DrivenFlowRoute,
        /,
    ):
        allowed = (
            (
                model,
                {
                    "kremer-grest-particle",
                    "bead-spring-slip-spring",
                    "hard-sphere-colloid",
                    "tube-continuum",
                },
            ),
            (entanglement, {"none", "native-ppa", "z1plus"}),
            (
                reptation,
                {
                    "none",
                    "particle-observables",
                    "doi-edwards",
                    "likhtman-mcleish",
                    "slip-spring",
                    "glamm",
                },
            ),
            (
                hydrodynamics,
                {
                    "none",
                    "constant",
                    "free-space-rpy",
                    "periodic-rpy",
                    "positive-split-periodic-rpy",
                    "confined-fib",
                    "rpy-lubrication",
                },
            ),
            (
                driven_flow,
                {
                    "equilibrium",
                    "overdamped-affine",
                    "sllod-shear",
                    "lees-edwards-sllod",
                    "kr-extension",
                    "constitutive-flow",
                },
            ),
        )
        if any(value not in choices for value, choices in allowed):
            raise ValueError("Polymer production regime contains an unknown coordinate.")
        self.model = model
        self.entanglement = entanglement
        self.reptation = reptation
        self.hydrodynamics = hydrodynamics
        self.driven_flow = driven_flow
        self.regime_id = canonical_fingerprint(
            {
                "kind": "polymer-production-regime",
                "model": model,
                "entanglement": entanglement,
                "reptation": reptation,
                "hydrodynamics": hydrodynamics,
                "driven_flow": driven_flow,
            }
        )

    @property
    def coordinates(self) -> tuple[str, str, str, str, str]:
        return (
            self.model,
            self.entanglement,
            self.reptation,
            self.hydrodynamics,
            self.driven_flow,
        )


class PolymerRegimeDecision(StrictModule, NonTrainableState):
    regime: PolymerProductionRegime
    supported: bool = eqx.field(static=True)
    exclusions: tuple[str, ...] = eqx.field(static=True)
    decision_id: str = eqx.field(static=True)

    def require_supported(self) -> PolymerProductionRegime:
        if not self.supported:
            detail = "; ".join(self.exclusions)
            raise ValueError(f"Unsupported polymer production regime: {detail}")
        return self.regime


def decide_polymer_production_regime(
    regime: PolymerProductionRegime, /
) -> PolymerRegimeDecision:
    if not isinstance(regime, PolymerProductionRegime):
        raise TypeError("regime must be PolymerProductionRegime.")
    supported = regime.coordinates in _SUPPORTED
    exclusions: list[str] = []
    if regime.entanglement != "none" and regime.hydrodynamics in {
        "free-space-rpy",
        "periodic-rpy",
        "positive-split-periodic-rpy",
        "confined-fib",
        "rpy-lubrication",
    }:
        exclusions.append(
            "entangled particle melts with long-range hydrodynamic interactions are not qualified"
        )
    if (
        regime.driven_flow.startswith("sllod")
        or regime.driven_flow == "lees-edwards-sllod"
    ):
        if regime.hydrodynamics != "none":
            exclusions.append("SLLOD is the underdamped no-HI route")
    if regime.driven_flow == "overdamped-affine" and regime.hydrodynamics == "none":
        exclusions.append("overdamped affine flow requires an explicit mobility route")
    if not supported and not exclusions:
        exclusions.append(
            "the exact coordinate tuple is absent from the qualified support matrix"
        )
    decision_id = canonical_fingerprint(
        {
            "kind": "polymer-regime-decision",
            "regime": regime.regime_id,
            "supported": supported,
            "exclusions": exclusions,
        }
    )
    return PolymerRegimeDecision(regime, supported, tuple(exclusions), decision_id)


def polymer_production_support_matrix() -> tuple[tuple[str, str, str, str, str], ...]:
    return tuple(sorted(_SUPPORTED))


__all__ = [
    "DrivenFlowRoute",
    "EntanglementRoute",
    "HydrodynamicRoute",
    "PolymerModelKind",
    "PolymerProductionRegime",
    "PolymerRegimeDecision",
    "ReptationRoute",
    "decide_polymer_production_regime",
    "polymer_production_support_matrix",
]
