#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Distributed execution qualification with physical-provider fail closure."""

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


CAPABILITY = "distributed-execution"


def _gate(name: str, category: str, description: str, /) -> GateDefinition:
    return GateDefinition(name, category, description)


_SPECTRAL_API = (
    "phydrax.discretization.spectral._distributed:SpectralMeshTopology",
    "phydrax.discretization.spectral._distributed:DistributedSpectralExecutionPlan",
    "phydrax.discretization.spectral._distributed:SpectralGlobalDiagnostics",
)
_LINE_API = (
    "phydrax.linalg._distributed_line:StructuredSolveTopologyPlan",
    "phydrax.linalg._distributed_line:DistributedLineSolvePlan",
    "phydrax.linalg._distributed_line:PreparedDistributedLineSolve",
)
_MULTI_DEVICE_GATE = _gate(
    "multi-device-execution",
    "operational",
    "The route executes on at least two observed physical devices without simulation.",
)
_RESOURCE_GATE = _gate(
    "resource-preflight",
    "performance",
    "The exact global shape and sharding topology fit the declared device budget.",
)


ROUTES: dict[str, RouteDefinition] = {
    "slab": RouteDefinition(
        "slab",
        (
            _gate(
                "slab-roundtrip",
                "scientific",
                "Distributed slab forward and inverse transforms satisfy round-trip tolerance.",
            ),
            _gate(
                "slab-parseval",
                "scientific",
                "Slab physical and modal energies satisfy Parseval normalization.",
            ),
            _gate(
                "slab-global-reduction",
                "scientific",
                "Global slab reductions equal the deterministic full-domain reference.",
            ),
            _RESOURCE_GATE,
            _MULTI_DEVICE_GATE,
        ),
        _SPECTRAL_API,
        dependency_scope="deployment",
    ),
    "pencil": RouteDefinition(
        "pencil",
        (
            _gate(
                "pencil-roundtrip",
                "scientific",
                "Pencil all-to-all transforms satisfy the exact round-trip criterion.",
            ),
            _gate(
                "pencil-transpose",
                "scientific",
                "Every prepared pencil transpose preserves the global field.",
            ),
            _gate(
                "pencil-global-reduction",
                "scientific",
                "Pencil reductions equal the deterministic global reference.",
            ),
            _RESOURCE_GATE,
            _MULTI_DEVICE_GATE,
        ),
        _SPECTRAL_API
        + ("phydrax.discretization.spectral._distributed:SpectralTranspose",),
        dependency_scope="deployment",
    ),
    "padded": RouteDefinition(
        "padded",
        (
            _gate(
                "padded-roundtrip",
                "scientific",
                "Distributed padded transforms preserve retained modal coefficients.",
            ),
            _gate(
                "padded-alias-suppression",
                "scientific",
                "Distributed padding suppresses aliased products at the declared cutoff.",
            ),
            _gate(
                "padded-hermitian-closure",
                "scientific",
                "Padded execution retains the declared Hermitian spectral closure.",
            ),
            _RESOURCE_GATE,
            _MULTI_DEVICE_GATE,
        ),
        _SPECTRAL_API,
        dependency_scope="deployment",
    ),
    "channel": RouteDefinition(
        "channel",
        (
            _gate(
                "channel-transform-roundtrip",
                "scientific",
                "Distributed Fourier-channel transforms satisfy route-exact round-trip tolerance.",
            ),
            _gate(
                "channel-line-solve",
                "scientific",
                "The wall-normal distributed line solve satisfies its residual criterion.",
            ),
            _gate(
                "channel-global-reduction",
                "scientific",
                "Channel diagnostics reduce over the complete global domain.",
            ),
            _RESOURCE_GATE,
            _MULTI_DEVICE_GATE,
        ),
        _SPECTRAL_API + _LINE_API,
        dependency_scope="deployment",
    ),
    "global-reductions": RouteDefinition(
        "global-reductions",
        (
            _gate(
                "global-sum",
                "scientific",
                "Distributed total equals the deterministic full-domain sum.",
            ),
            _gate(
                "global-l2",
                "scientific",
                "Distributed L2 norm equals the deterministic full-domain norm.",
            ),
            _gate(
                "global-maximum",
                "scientific",
                "Distributed maximum equals the full-domain maximum.",
            ),
            _RESOURCE_GATE,
            _MULTI_DEVICE_GATE,
        ),
        _SPECTRAL_API,
        dependency_scope="deployment",
    ),
    "line-local": RouteDefinition(
        "line-local",
        (
            _gate(
                "line-local-residual",
                "scientific",
                "Line-contiguous local solves satisfy the exact tridiagonal residual.",
            ),
            _gate(
                "line-local-batch-equivalence",
                "scientific",
                "Transverse sharding equals the unsharded batched line reference.",
            ),
            _gate(
                "line-local-resource",
                "performance",
                "Local factors fit the declared factor and workspace budget.",
            ),
            _gate(
                "line-local-execution",
                "operational",
                "The line axis remains local and no split-line algorithm is substituted.",
            ),
        ),
        _LINE_API
        + ("phydrax.linalg._distributed_line:PreparedTransverseBatchLineSolve",),
        dependency_scope="deployment",
    ),
    "partitioned-thomas": RouteDefinition(
        "partitioned-thomas",
        (
            _gate(
                "thomas-residual",
                "scientific",
                "Partitioned Thomas reconstruction satisfies the global line residual.",
            ),
            _gate(
                "thomas-interface",
                "scientific",
                "Reduced interface values agree across every contiguous partition.",
            ),
            _gate(
                "thomas-uneven-tail",
                "scientific",
                "Uneven final partitions retain the full physical line.",
            ),
            _RESOURCE_GATE,
            _MULTI_DEVICE_GATE,
        ),
        _LINE_API,
        dependency_scope="deployment",
    ),
    "spike": RouteDefinition(
        "spike",
        (
            _gate(
                "spike-residual",
                "scientific",
                "SPIKE reconstruction satisfies the global line residual.",
            ),
            _gate(
                "spike-interface",
                "scientific",
                "SPIKE reduced-interface elimination meets its determinant criterion.",
            ),
            _gate(
                "spike-communication",
                "scientific",
                "Observed neighbor and collective rounds match the prepared SPIKE route.",
            ),
            _RESOURCE_GATE,
            _MULTI_DEVICE_GATE,
        ),
        _LINE_API,
        dependency_scope="deployment",
    ),
    "pcr": RouteDefinition(
        "pcr",
        (
            _gate(
                "pcr-residual",
                "scientific",
                "Parallel cyclic reduction satisfies the global line residual.",
            ),
            _gate(
                "pcr-padding",
                "scientific",
                "PCR internal power-of-two padding does not change physical-line values.",
            ),
            _gate(
                "pcr-communication",
                "scientific",
                "Observed collective rounds match the prepared PCR topology.",
            ),
            _RESOURCE_GATE,
            _MULTI_DEVICE_GATE,
        ),
        _LINE_API,
        dependency_scope="deployment",
    ),
    "topology-restart": RouteDefinition(
        "topology-restart",
        (
            _gate(
                "restart-state-equivalence",
                "scientific",
                "Restored canonical state meets the admitted bitwise or tolerance relation.",
            ),
            _gate(
                "restart-direct-shards",
                "scientific",
                "Canonical byte ranges restore directly into complete destination shards.",
            ),
            _gate(
                "restart-resource-bound",
                "performance",
                "Direct restoration stays within the declared segment and staging bounds.",
            ),
            _MULTI_DEVICE_GATE,
            _gate(
                "restart-admission",
                "operational",
                "The exact source-target relation is admitted by the exact restart policy.",
            ),
            _gate(
                "restart-cache-exclusion",
                "operational",
                "Execution caches never participate in topology-changing restoration.",
            ),
        ),
        (
            "phydrax.lifecycle._restart_topology:TopologyRestartRelation",
            "phydrax.lifecycle._restart_topology:RestartAdmission",
            "phydrax.lifecycle._restart_topology:prepare_direct_restore",
            "phydrax.lifecycle._restart_topology:execute_direct_restore",
        ),
        dependency_scope="deployment",
    ),
    "multiblock-extruded": RouteDefinition(
        "multiblock-extruded",
        (
            _gate(
                "extruded-axis-invariance",
                "scientific",
                "The extruded transform commutes with every certified block coupling.",
            ),
            _gate(
                "multiblock-mortar-continuity",
                "scientific",
                "Global multiblock mortar continuity meets tolerance.",
            ),
            _gate(
                "multiblock-global-residual",
                "scientific",
                "The global coupled solve satisfies its declared residual.",
            ),
            _RESOURCE_GATE,
            _MULTI_DEVICE_GATE,
        ),
        (
            "phydrax.linalg._distributed_line:ExtrudedAxisInvarianceCertificate",
            "phydrax.linalg._distributed_line:MultiblockExtrudedReductionPlan",
        ),
        dependency_scope="deployment",
    ),
    "scale-resource": RouteDefinition(
        "scale-resource",
        (
            _gate(
                "scale-solution-equivalence",
                "scientific",
                "Scaled execution retains the admitted solution tolerance.",
            ),
            _gate(
                "observed-resource-record",
                "performance",
                "At least one exact observed resource record is bound.",
            ),
            _gate(
                "forecast-resource-record",
                "performance",
                "Forecast bounds bind included observations and the exact forecast model.",
            ),
            _MULTI_DEVICE_GATE,
            _gate(
                "scale-topology-executed",
                "operational",
                "The recorded scale topology is the physically executed topology.",
            ),
        ),
        (
            "phydrax.qualification._evidence:ObservedResourceRecord",
            "phydrax.qualification._evidence:ForecastResourceRecord",
            "phydrax.discretization.spectral._distributed:SpectralResourceReport",
        ),
        dependency_scope="deployment",
    ),
}

_MULTI_DEVICE_ROUTES = frozenset(route for route in ROUTES if route != "line-local")


def _resource_presence(
    request: Mapping[str, object], field: str, reason: str, /
) -> bool | dict[str, object]:
    values = request.get(field, ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be a sequence.")
    return True if values else {"unavailable_reason": reason}


def produce_candidate(
    route: str,
    request: Mapping[str, object],
    /,
    *,
    reference_manifest: ReferenceArtifactManifest | Mapping[str, object] | None = None,
    reference_payload: bytes | None = None,
) -> dict[str, object]:
    """Produce one distributed candidate, never passing simulated hardware."""

    if route not in ROUTES:
        raise ValueError(f"Unknown distributed execution route {route!r}.")
    prepared = dict(request)
    if route in _MULTI_DEVICE_ROUTES:
        availability = request.get("availability")
        if availability is not None and not isinstance(availability, Mapping):
            raise TypeError("availability must be a mapping.")
        prepared = with_observation(
            prepared,
            "multi-device-execution",
            availability_observation(
                availability,
                provider="jax-multi-device",
                minimum_devices=2,
                require_hardware=True,
            ),
        )
    if route == "scale-resource":
        prepared = with_observation(
            prepared,
            "observed-resource-record",
            _resource_presence(
                prepared,
                "observed_resource_records",
                "observed-resource-record-not-supplied",
            ),
        )
        prepared = with_observation(
            prepared,
            "forecast-resource-record",
            _resource_presence(
                prepared,
                "forecast_resource_records",
                "forecast-resource-record-not-supplied",
            ),
        )
    return make_candidate_artifact(
        CAPABILITY,
        ROUTES[route],
        prepared,
        reference_manifest=reference_manifest,
        reference_payload=reference_payload,
        extra_record={
            "execution": "physical-provider-only",
            "simulated_qualification_permitted": False,
        },
    )


def assemble_profile(
    artifacts: Sequence[Mapping[str, object]],
    /,
    *,
    name: str = "distributed-execution.candidate",
    provider: str = "phydrax",
) -> dict[str, object]:
    return assemble_candidate_profile(artifacts, name=name, provider=provider)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_cli_parser(
        "Create unsigned distributed-execution qualification candidates.",
        ROUTES,
        profile_name="distributed-execution.candidate",
    )
    run_cli(parser, ROUTES, CAPABILITY, argv, producer=produce_candidate)


if __name__ == "__main__":
    main()
