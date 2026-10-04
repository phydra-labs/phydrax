# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exact unreleased envelopes for streamed standard-MACE execution.

Profiles are closure exceptions for native standard MACE, image-aware learned
graphs and bounded architecture-specialized execution. They declare evidence
obligations, not measured success, scientific accuracy or release authority.
"""

from __future__ import annotations

from itertools import product
from typing import get_args, Literal, TypeAlias

from ..qualification._registry import CapabilityProfile, SupportTuple, SupportValue
from ..typing import parse


MACEGeometry: TypeAlias = Literal[
    "finite", "orthorhombic", "triclinic", "partial-periodic"
]
MACEPrecision: TypeAlias = Literal["float32", "float64"]
MACEDerivative: TypeAlias = Literal[
    "energy-force", "energy-force-stress", "mixed-parameter", "coordinate-hvp"
]


_COMMON_GATES = (
    "compiler-resource-envelope",
    "derivative-reference",
    "implementation-complete",
    "independent-review",
    "numerical-reference",
    "operations-restart",
    "rights-and-security",
    "scientific-validation",
)


def mace_support(
    geometry: MACEGeometry,
    precision: MACEPrecision,
    derivative: MACEDerivative,
    /,
    *,
    route: str,
    device: str,
    atoms: int = 32,
    edges: int = 4096,
    channels: int = 8,
    correlation: int = 3,
    angular_degree: int = 3,
    layers: int = 2,
) -> dict[str, SupportValue]:
    """Declare one bounded conjunction; never infer basis from channel shape."""
    selected_geometry = parse(geometry, MACEGeometry, "geometry")
    selected_precision = parse(precision, MACEPrecision, "precision")
    selected_derivative = parse(derivative, MACEDerivative, "derivative")
    for name, value in (
        ("atoms", atoms),
        ("edges", edges),
        ("channels", channels),
        ("correlation", correlation),
        ("angular_degree", angular_degree),
        ("layers", layers),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive declared capacity.")
    if not route or not device:
        raise ValueError("route and device must name the declared execution boundary.")
    if selected_geometry == "finite" and selected_derivative == "energy-force-stress":
        raise ValueError(
            "Bulk stress requires an explicitly supplied periodic cell volume."
        )
    return {
        "architecture": "standard-mace",
        "closure-exception": "streamed-atomistic-execution",
        "geometry": selected_geometry,
        "precision": selected_precision,
        "derivative": selected_derivative,
        "route": route,
        "device": device,
        "provider": "phydrax-native",
        "maximum-atoms": atoms,
        "maximum-edges": edges,
        "channels": channels,
        "correlation": correlation,
        "angular-degree": angular_degree,
        "layers": layers,
        "image-topology": "explicit-stable-directed-shifts",
        "basis-identity": "declared-real-o3-layout",
        "parameterization": "original-w-fixed-u",
        "stress-convention": "tensile-positive-strain-gradient-per-cell-volume",
        "partial-periodic-volume": "explicit-three-dimensional-embedding",
        "topology-derivative": "frozen-candidates-and-images",
        "release-claim": False,
    }


def _native_supports(route: str, /) -> tuple[SupportTuple, ...]:
    capability = "atomistic.mace-inference"
    return tuple(
        SupportTuple(
            capability,
            mace_support(geometry, precision, derivative, route=route, device="cpu"),
        )
        for geometry, precision, derivative in product(
            get_args(MACEGeometry), get_args(MACEPrecision), get_args(MACEDerivative)
        )
        if not (geometry == "finite" and derivative == "energy-force-stress")
    )


def mace_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Candidate envelopes stay unreleased until independent exact gates pass."""
    exact = CapabilityProfile(
        "atomistic.mace-inference.native-exact.profile",
        "phydrax",
        "candidate",
        _native_supports("native-exact"),
        required_gates=(*_COMMON_GATES, "source-checkpoint-fidelity", "capacity-scaling"),
        released=False,
    )
    tabulated = CapabilityProfile(
        "atomistic.mace-inference.native-tabulated.profile",
        "phydrax",
        "candidate",
        tuple(
            SupportTuple(
                "atomistic.mace-inference",
                mace_support(
                    geometry,
                    precision,
                    "energy-force" if geometry == "finite" else "energy-force-stress",
                    route="native-tabulated",
                    device="cpu",
                ),
            )
            for geometry, precision in product(
                get_args(MACEGeometry), get_args(MACEPrecision)
            )
        ),
        required_gates=(
            *_COMMON_GATES,
            "source-checkpoint-fidelity",
            "radial-value-derivative-fidelity",
            "cutoff-knot-regularity",
            "capacity-scaling",
        ),
        released=False,
    )
    accelerated = CapabilityProfile(
        "atomistic.mace-inference.pallas-cuda.profile",
        "phydrax",
        "candidate",
        tuple(
            SupportTuple(
                "atomistic.mace-inference",
                mace_support(
                    geometry, precision, derivative, route="pallas", device="cuda"
                ),
            )
            for geometry, precision, derivative in product(
                get_args(MACEGeometry), get_args(MACEPrecision), get_args(MACEDerivative)
            )
            if not (geometry == "finite" and derivative == "energy-force-stress")
        ),
        required_gates=(
            *_COMMON_GATES,
            "actual-cuda-execution",
            "source-checkpoint-fidelity",
            "capacity-scaling",
            "useful-matched-performance",
        ),
        released=False,
    )
    distributed = CapabilityProfile(
        "atomistic.mace-inference.distributed.profile",
        "phydrax",
        "candidate",
        tuple(
            SupportTuple(
                "atomistic.mace-inference",
                {
                    **mace_support(
                        geometry,
                        precision,
                        "energy-force-stress",
                        route="distributed-native",
                        device="cpu",
                    ),
                    "owner-count": 2,
                    "collective": "jax-owner-local-halo",
                    "migration": "complete-accepted-state-transaction",
                },
            )
            for geometry, precision in product(
                get_args(MACEGeometry), get_args(MACEPrecision)
            )
            if geometry != "finite"
        ),
        required_gates=(
            *_COMMON_GATES,
            "actual-two-device-execution",
            "owner-local-source-parity",
            "exact-reverse-halo",
            "migration-restart",
            "shared-cell-cotangent",
        ),
        released=False,
    )
    return exact, tabulated, accelerated, distributed


__all__ = ["mace_candidate_profiles", "mace_support"]
