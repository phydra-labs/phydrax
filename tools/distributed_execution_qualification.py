#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Distributed spectral qualification with physical-provider fail closure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from phydrax.discretization.spectral._qualification import (
    distributed_spectral_candidate_profiles,
    distributed_spectral_support_tuples,
)
from phydrax.qualification._evidence import SupportDependency
from phydrax.qualification._reference import ReferenceArtifactManifest
from phydrax.qualification._registry import SupportTuple
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


CAPABILITY = "distributed-spectral-execution"
_OWNER_PROFILE = distributed_spectral_candidate_profiles()[0]
_OWNER_SUPPORT_BY_ROUTE = {
    dict(value.attributes)["route"]: value
    for value in distributed_spectral_support_tuples()
}


def _gate(name: str, category: str, description: str, /) -> GateDefinition:
    return GateDefinition(name, category, description)


_SPECTRAL_API = (
    "phydrax.discretization.spectral._distributed:SpectralMeshTopology",
    "phydrax.discretization.spectral._distributed:DistributedSpectralExecutionPlan",
    "phydrax.discretization.spectral._distributed:SpectralGlobalDiagnostics",
    "phydrax.discretization.spectral._distributed:SpectralResourceReport",
    "phydrax.discretization.spectral._distributed:SpectralTranspose",
)
_CHANNEL_API = _SPECTRAL_API + (
    "phydrax.discretization.spectral._distributed:"
    "DistributedSpectralExecutionPlan.execute_channel",
)
_FORWARD_GATE = _gate(
    "forward-reference",
    "scientific",
    "The distributed forward transform agrees with an independent full-complex reference.",
)
_INVERSE_GATE = _gate(
    "inverse-reference",
    "scientific",
    "The distributed inverse transform agrees with an independent full-complex reference.",
)
_DIRECTIONAL_JVP_GATE = _gate(
    "directional-jvp",
    "scientific",
    "Forward- and inverse-direction JVPs agree with independent linear references.",
)
_HILBERT_ADJOINT_GATE = _gate(
    "hilbert-adjoint",
    "scientific",
    "The complex Hilbert adjoints satisfy the declared normalization pairing.",
)
_PRECISION_GATE = _gate(
    "precision-policy",
    "scientific",
    "Execution honors the exact transform, storage, and accumulation precision policy.",
)
_PAYLOAD_GATE = _gate(
    "payload-admission",
    "scientific",
    "Every exercised payload has one exact admitted shape including the state shape.",
)
_STAGE_IDENTITY_GATE = _gate(
    "stage-identity",
    "operational",
    "Executed forward, inverse, and padded stage sequences match their fingerprints.",
)
_TOPOLOGY_GATE = _gate(
    "process-qualified-topology",
    "operational",
    "Observed devices are bound to the executing JAX process topology.",
)
_NO_GATHER_GATE = _gate(
    "no-host-gather",
    "operational",
    "Execution completes without an implicit host or global gather.",
)
_DECLARED_RESOURCE_GATE = _gate(
    "declared-resource",
    "performance",
    "Observed logical storage, workspace, liveness, and collective traffic fit the declared budget.",
)
_COMPILER_MEMORY_GATE = _gate(
    "compiler-memory",
    "performance",
    "Compiler and runtime memory observations remain within the retained bound.",
)
_PHYSICAL_MULTI_DEVICE_GATE = _gate(
    "physical-multi-device",
    "operational",
    "The route executes on at least two process-qualified physical devices.",
)
_PHYSICAL_MULTI_HOST_GATE = _gate(
    "physical-multi-host",
    "operational",
    "The route executes across at least two observed physical hosts and processes.",
)
_CHANNEL_EXECUTION_GATE = _gate(
    "channel-execution",
    "scientific",
    "execute_channel agrees with an independent horizontal channel reference.",
)
_CHANNEL_LAYOUT_GATE = _gate(
    "channel-horizontal-layout",
    "operational",
    "The declared Fourier–Chebyshev–Fourier horizontal layout is preserved.",
)
_CHANNEL_PARTITION_GATE = _gate(
    "channel-horizontal-partition",
    "operational",
    "Only the ordered horizontal Fourier axes use the prepared regular partition.",
)
_CHANNEL_ZERO_MODE_GATE = _gate(
    "channel-atomic-zero-mode",
    "scientific",
    "The atomic horizontal zero mode matches the independent channel reference.",
)
_SPECTRAL_GATES = (
    _FORWARD_GATE,
    _INVERSE_GATE,
    _DIRECTIONAL_JVP_GATE,
    _HILBERT_ADJOINT_GATE,
    _PRECISION_GATE,
    _PAYLOAD_GATE,
    _STAGE_IDENTITY_GATE,
    _TOPOLOGY_GATE,
    _NO_GATHER_GATE,
    _DECLARED_RESOURCE_GATE,
    _COMPILER_MEMORY_GATE,
    _PHYSICAL_MULTI_DEVICE_GATE,
)
_CHANNEL_GATES = (
    _CHANNEL_EXECUTION_GATE,
    _CHANNEL_ZERO_MODE_GATE,
    _PRECISION_GATE,
    _PAYLOAD_GATE,
    _CHANNEL_LAYOUT_GATE,
    _CHANNEL_PARTITION_GATE,
    _TOPOLOGY_GATE,
    _NO_GATHER_GATE,
    _DECLARED_RESOURCE_GATE,
    _COMPILER_MEMORY_GATE,
    _PHYSICAL_MULTI_DEVICE_GATE,
)
_MULTI_HOST_GATES = (
    _FORWARD_GATE,
    _TOPOLOGY_GATE,
    _NO_GATHER_GATE,
    _DECLARED_RESOURCE_GATE,
    _PHYSICAL_MULTI_DEVICE_GATE,
    _PHYSICAL_MULTI_HOST_GATE,
)


def _route(name: str, /) -> RouteDefinition:
    return RouteDefinition(
        name,
        _SPECTRAL_GATES,
        _SPECTRAL_API,
        dependency_scope="deployment",
    )


ROUTES: dict[str, RouteDefinition] = {
    name: _route(name)
    for name in (
        "slab-roundtrip",
        "pencil-roundtrip",
        "padded-dealias",
        "global-reductions",
    )
}
ROUTES["channel-horizontal"] = RouteDefinition(
    "channel-horizontal",
    _CHANNEL_GATES,
    _CHANNEL_API,
    dependency_scope="deployment",
)
ROUTES["scale-resource"] = RouteDefinition(
    "scale-resource",
    _SPECTRAL_GATES,
    _SPECTRAL_API
    + (
        "phydrax.qualification._evidence:ObservedResourceRecord",
        "phydrax.qualification._evidence:ForecastResourceRecord",
    ),
    dependency_scope="deployment",
)
ROUTES["multi-host"] = RouteDefinition(
    "multi-host",
    _MULTI_HOST_GATES,
    _SPECTRAL_API,
    dependency_scope="deployment",
)


def _resource_presence(
    request: Mapping[str, object], field: str, reason: str, /
) -> bool | dict[str, object]:
    values = request.get(field, ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be a sequence.")
    return True if values else {"unavailable_reason": reason}


def _availability(request: Mapping[str, object], /) -> Mapping[str, object] | None:
    value = request.get("availability")
    if value is not None and not isinstance(value, Mapping):
        raise TypeError("availability must be a mapping.")
    return value


def _process_topology_observation(
    availability: Mapping[str, object] | None, /
) -> bool | dict[str, object]:
    if availability is None:
        return {"unavailable_reason": "jax-topology-availability-not-recorded"}
    if availability.get("simulated", False) is True:
        return {"unavailable_reason": "jax-topology-simulation-is-not-qualification"}
    if availability.get("forced", False) is True:
        return {"unavailable_reason": "jax-topology-forced-device-is-not-qualification"}
    if availability.get("process_qualified") is not True:
        return {"unavailable_reason": "jax-topology-is-not-process-qualified"}
    process_count = availability.get("process_count")
    if type(process_count) is not int or process_count < 1:
        return {"unavailable_reason": "jax-topology-process-count-not-observed"}
    return True


def _physical_multi_device_observation(
    availability: Mapping[str, object] | None, /
) -> bool | dict[str, object]:
    if availability is not None and availability.get("forced", False) is True:
        return {
            "unavailable_reason": "jax-multi-device-forced-device-is-not-qualification"
        }
    observed = availability_observation(
        availability,
        provider="jax-multi-device",
        minimum_devices=2,
        require_hardware=True,
    )
    if observed is not True:
        return observed
    if availability is None or availability.get("physical") is not True:
        return {"unavailable_reason": "jax-multi-device-physical-devices-not-observed"}
    if availability.get("process_qualified") is not True:
        return {"unavailable_reason": "jax-multi-device-is-not-process-qualified"}
    return True


def _physical_multi_host_observation(
    availability: Mapping[str, object] | None, /
) -> bool | dict[str, object]:
    device_observation = _physical_multi_device_observation(availability)
    if device_observation is not True:
        return device_observation
    if availability is None:
        return {"unavailable_reason": "jax-multi-host-availability-not-recorded"}
    if availability.get("same_host", False) is True:
        return {"unavailable_reason": "jax-multi-host-same-host-is-not-qualification"}
    process_count = availability.get("process_count")
    host_count = availability.get("host_count")
    if type(process_count) is not int or process_count < 2:
        return {"unavailable_reason": "jax-multi-host-requires-two-processes"}
    if type(host_count) is not int or host_count < 2:
        return {"unavailable_reason": "jax-multi-host-requires-two-physical-hosts"}
    return True


def _require_owner_admission(route: str, request: Mapping[str, object], /) -> None:
    support_value = request.get("support_tuple")
    if isinstance(support_value, SupportTuple):
        support = support_value
    elif isinstance(support_value, Mapping):
        support = SupportTuple.from_record(support_value)
    else:
        raise TypeError("support_tuple must be a SupportTuple or serialized mapping.")
    dependency_value = request.get("support_dependency")
    if isinstance(dependency_value, SupportDependency):
        dependency = dependency_value
    elif isinstance(dependency_value, Mapping):
        dependency = SupportDependency.from_record(dependency_value)
    else:
        raise TypeError(
            "support_dependency must be a SupportDependency or serialized mapping."
        )
    expected = _OWNER_SUPPORT_BY_ROUTE[route]
    if support.support_tuple_id != expected.support_tuple_id:
        raise ValueError(
            "Distributed spectral evidence must bind the exact owner-local SupportTuple."
        )
    if (
        dependency.profile_id != _OWNER_PROFILE.profile_id
        or dependency.support_tuple_id != expected.support_tuple_id
    ):
        raise ValueError(
            "Distributed spectral evidence cannot inherit a consumer capability profile."
        )


def produce_candidate(
    route: str,
    request: Mapping[str, object],
    /,
    *,
    reference_manifest: ReferenceArtifactManifest | Mapping[str, object] | None = None,
    reference_payload: bytes | None = None,
) -> dict[str, object]:
    """Produce one owner-local candidate with fail-closed physical evidence."""

    if route not in ROUTES:
        raise ValueError(f"Unknown distributed spectral route {route!r}.")
    _require_owner_admission(route, request)
    prepared = dict(request)
    availability = _availability(request)
    prepared = with_observation(
        prepared,
        "process-qualified-topology",
        _process_topology_observation(availability),
    )
    prepared = with_observation(
        prepared,
        "physical-multi-device",
        _physical_multi_device_observation(availability),
    )
    if route == "multi-host":
        prepared = with_observation(
            prepared,
            "physical-multi-host",
            _physical_multi_host_observation(availability),
        )
    if route == "scale-resource":
        prepared = with_observation(
            prepared,
            "declared-resource",
            _resource_presence(
                prepared,
                "observed_resource_records",
                "declared-resource-record-not-supplied",
            ),
        )
        prepared = with_observation(
            prepared,
            "compiler-memory",
            _resource_presence(
                prepared,
                "forecast_resource_records",
                "compiler-memory-record-not-supplied",
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
            "forced_device_qualification_permitted": False,
            "les_evidence_inherited": False,
            "pic_evidence_inherited": False,
            "same_host_multi_host_qualification_permitted": False,
            "simulated_qualification_permitted": False,
        },
    )


def assemble_profile(
    artifacts: Sequence[Mapping[str, object]],
    /,
    *,
    name: str = "distributed-spectral-execution.profile",
    provider: str = "phydrax",
) -> dict[str, object]:
    return assemble_candidate_profile(artifacts, name=name, provider=provider)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_cli_parser(
        "Create unsigned distributed-spectral qualification candidates.",
        ROUTES,
        profile_name="distributed-spectral-execution.profile",
    )
    run_cli(parser, ROUTES, CAPABILITY, argv, producer=produce_candidate)


if __name__ == "__main__":
    main()
