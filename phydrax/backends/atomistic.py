#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Admission owner for accelerated atomistic Pallas kernels.

The accelerated MACE edge coupling is written once against the Pallas Mosaic
GPU backend (``jax.experimental.pallas.mosaic_gpu``) with lane semantics: one
warpgroup of 128 lanes owns a channel tile. ``"cuda"`` compiles those kernels
for the active NVIDIA device. ``"cpu_interpret"`` executes the same kernel
bodies through the Mosaic GPU interpreter on CPU, which simulates GPU memory
spaces and threads; it is semantic reference verification of the kernels and
never GPU correctness, performance or memory qualification. No target
substitutes ordinary JAX for a kernel.
"""

from __future__ import annotations

from importlib.metadata import version
from typing import final, get_args, Literal, TypeAlias

import equinox as eqx
import jax

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._validation import nonnegative_integer, positive_integer
from ..sparse._execution import RelationAccumulation
from ..typing import parse
from ._types import (
    BackendAvailability,
    BackendCapabilities,
    BackendDistributionCapabilities,
)


AtomisticAccelerationTarget: TypeAlias = Literal["cuda", "cpu_interpret"]
AtomisticKernelPrecision: TypeAlias = Literal["float32", "float64"]
AtomisticKernelLowering: TypeAlias = Literal[
    "pallas_mosaic_gpu", "pallas_mosaic_gpu_interpret"
]
MACECouplingRole: TypeAlias = Literal[
    "coefficient", "radial", "source", "harmonic", "receiver"
]

WARPGROUP_LANES = 128
_BACKEND = "pallas-atomistic"
# Mosaic GPU's maintained targets are Hopper and Blackwell, and lowering was
# verified only at architecture 9.0. Upstream pre-Hopper code paths are not
# admitted without their own compile evidence.
_CUDA_MINIMUM_COMPUTE_CAPABILITY = (9, 0)
_REQUIREMENT = (
    "JAX Pallas Mosaic GPU on an NVIDIA CUDA device with compute capability >= "
    f"{_CUDA_MINIMUM_COMPUTE_CAPABILITY[0]}.{_CUDA_MINIMUM_COMPUTE_CAPABILITY[1]}, "
    "or the explicit CPU Mosaic GPU interpreter for reference verification"
)


def mace_coupling_problem_kind(role: MACECouplingRole, /) -> str:
    """Backend problem kind of one MACE edge-coupling kernel role."""
    return f"atomistic.mace_edge_coupling.{parse(role, MACECouplingRole, 'role')}"


PALLAS_ATOMISTIC_CAPABILITIES = BackendCapabilities(
    backend=_BACKEND,
    problem_kinds=tuple(
        mace_coupling_problem_kind(role) for role in get_args(MACECouplingRole)
    ),
    execution="device",
    host_only=False,
    supports_matrix_free=True,
    supports_assembled=False,
    coordinate_dtypes=get_args(AtomisticKernelPrecision),
    distribution=BackendDistributionCapabilities(
        array_model="global",
        scopes=("single_device",),
        platforms=get_args(AtomisticAccelerationTarget),
        transformations=("jit", "jvp", "linearize", "vjp", "transpose", "vmap"),
        communicator_model="jax",
    ),
)


@final
class AtomisticKernelLimits(StrictModule):
    """Declared admission envelope of the accelerated atomistic kernels.

    ``maximum_workspace_bytes`` bounds the planned per-program register working
    set computed from the static coupling structure and tiles. It is a planning
    bound, not a measured occupancy, register or shared-memory figure; those
    remain compiler-determined and are reported only when observed.
    ``maximum_reduction_programs`` bounds the fixed program count of strided
    cross-lane reductions, so reduction partials never scale with edges.
    """

    maximum_degree: int = eqx.field(static=True)
    maximum_paths: int = eqx.field(static=True)
    maximum_coefficients: int = eqx.field(static=True)
    maximum_tile: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)
    maximum_reduction_programs: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_degree: int,
        maximum_paths: int,
        maximum_coefficients: int,
        maximum_tile: int,
        maximum_workspace_bytes: int,
        maximum_derivative_order: int,
        maximum_reduction_programs: int,
    ) -> None:
        self.maximum_degree = positive_integer(maximum_degree, "maximum_degree")
        self.maximum_paths = positive_integer(maximum_paths, "maximum_paths")
        self.maximum_coefficients = positive_integer(
            maximum_coefficients, "maximum_coefficients"
        )
        self.maximum_tile = positive_integer(maximum_tile, "maximum_tile")
        self.maximum_workspace_bytes = positive_integer(
            maximum_workspace_bytes, "maximum_workspace_bytes"
        )
        self.maximum_derivative_order = positive_integer(
            maximum_derivative_order, "maximum_derivative_order"
        )
        self.maximum_reduction_programs = positive_integer(
            maximum_reduction_programs, "maximum_reduction_programs"
        )


ATOMISTIC_KERNEL_LIMITS = AtomisticKernelLimits(
    maximum_degree=4,
    maximum_paths=64,
    maximum_coefficients=4096,
    maximum_tile=1024,
    maximum_workspace_bytes=192 * 1024,
    maximum_derivative_order=2,
    maximum_reduction_programs=1024,
)


def _cuda_device() -> tuple[jax.Device | None, str]:
    if jax.default_backend() != "gpu":
        return None, f"the active JAX backend is {jax.default_backend()!r}, not a GPU"
    device = jax.devices()[0]
    if device.client.platform != "cuda":
        return None, (
            f"the active GPU client is {device.client.platform!r}; only NVIDIA CUDA "
            "is an implemented accelerated atomistic target"
        )
    return device, "an NVIDIA CUDA device is the active JAX backend"


def _compute_capability(device: jax.Device, /) -> tuple[int, int]:
    major, minor = str(device.compute_capability).split(".")
    return int(major), int(minor)


def pallas_atomistic_availability(
    target: AtomisticAccelerationTarget, /
) -> BackendAvailability:
    """Return whether the accelerated atomistic kernels can execute on ``target``."""
    target_ = parse(target, AtomisticAccelerationTarget, "target")
    match target_:
        case "cpu_interpret":
            available = True
            reason = (
                "the CPU Mosaic GPU interpreter executes the kernel bodies for "
                "reference verification; this is not GPU qualification"
            )
        case "cuda":
            device, reason = _cuda_device()
            available = device is not None
            if device is not None:
                capability = _compute_capability(device)
                if capability < _CUDA_MINIMUM_COMPUTE_CAPABILITY:
                    available = False
                    reason = (
                        f"device {device.device_kind!r} has compute capability "
                        f"{capability[0]}.{capability[1]}, below the admitted minimum"
                    )
        case _:
            raise ValueError(f"Unsupported atomistic acceleration target {target_!r}.")
    return BackendAvailability(
        capabilities=PALLAS_ATOMISTIC_CAPABILITIES,
        available=available,
        requirement=_REQUIREMENT,
        reason=reason,
        versions=(("jax", jax.__version__), ("jaxlib", version("jaxlib"))),
    )


@final
class AtomisticKernelTarget(StrictModule):
    """Resolved device, lowering and runtime identity of one kernel target."""

    target: AtomisticAccelerationTarget = eqx.field(static=True)
    platform: str = eqx.field(static=True)
    lowering: AtomisticKernelLowering = eqx.field(static=True)
    device_kind: str = eqx.field(static=True)
    compute_capability: str | None = eqx.field(static=True)
    platform_version: str = eqx.field(static=True)
    jax_version: str = eqx.field(static=True)
    jaxlib_version: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __init__(self, target: AtomisticAccelerationTarget, /) -> None:
        target_ = parse(target, AtomisticAccelerationTarget, "target")
        pallas_atomistic_availability(target_).require(
            mace_coupling_problem_kind("receiver")
        )
        match target_:
            case "cuda":
                device, _ = _cuda_device()
                if device is None:
                    raise RuntimeError("The admitted CUDA device disappeared.")
                platform = "cuda"
                lowering: AtomisticKernelLowering = "pallas_mosaic_gpu"
                major, minor = _compute_capability(device)
                capability: str | None = f"{major}.{minor}"
            case "cpu_interpret":
                device = jax.devices("cpu")[0]
                platform = "cpu"
                lowering = "pallas_mosaic_gpu_interpret"
                capability = None
            case _:
                raise ValueError(
                    f"Unsupported atomistic acceleration target {target_!r}."
                )
        self.target = target_
        self.platform = platform
        self.lowering = lowering
        self.device_kind = device.device_kind
        self.compute_capability = capability
        self.platform_version = device.client.platform_version
        self.jax_version = jax.__version__
        self.jaxlib_version = version("jaxlib")
        self.target_id = canonical_fingerprint(
            {
                "kind": "atomistic-kernel-target",
                "backend": _BACKEND,
                "target": target_,
                "platform": platform,
                "lowering": lowering,
                "device_kind": self.device_kind,
                "compute_capability": capability,
                "platform_version": self.platform_version,
                "jax": self.jax_version,
                "jaxlib": self.jaxlib_version,
            }
        )


@final
class AtomisticKernelRequest(StrictModule):
    """Static structural, tiling, extent, precision and derivative facts to admit.

    ``channel_capacity`` is the channel multiplicity padded to whole channel
    tiles; padded channels carry zero operands and are sliced from results.
    ``receiver_extent``, ``source_extent`` and ``edge_extent`` are the
    independent receiver-slot, local-source-row and lane extents of one kernel
    fragment; ``fragment_bytes`` is the declared logical bytes of the largest
    single kernel bind over that fragment (operands plus result), admitted
    against the caller's explicit ``fragment_budget_bytes``.
    """

    precision: AtomisticKernelPrecision = eqx.field(static=True)
    accumulation: RelationAccumulation = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    channel_tile: int = eqx.field(static=True)
    channel_capacity: int = eqx.field(static=True)
    receiver_tile: int = eqx.field(static=True)
    edge_tile: int = eqx.field(static=True)
    reduction_programs: int = eqx.field(static=True)
    receiver_extent: int = eqx.field(static=True)
    source_extent: int = eqx.field(static=True)
    edge_extent: int = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    path_count: int = eqx.field(static=True)
    coefficient_count: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    fragment_bytes: int = eqx.field(static=True)
    fragment_budget_bytes: int = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        precision: AtomisticKernelPrecision,
        accumulation: RelationAccumulation,
        channels: int,
        channel_tile: int,
        receiver_tile: int,
        edge_tile: int,
        reduction_programs: int,
        receiver_extent: int,
        source_extent: int,
        edge_extent: int,
        maximum_degree: int,
        path_count: int,
        coefficient_count: int,
        workspace_bytes: int,
        fragment_bytes: int,
        fragment_budget_bytes: int,
        maximum_derivative_order: int,
    ) -> None:
        self.precision = parse(precision, AtomisticKernelPrecision, "precision")
        self.accumulation = parse(accumulation, RelationAccumulation, "accumulation")
        self.channels = positive_integer(channels, "channels")
        self.channel_tile = positive_integer(channel_tile, "channel_tile")
        self.channel_capacity = -(-self.channels // self.channel_tile) * self.channel_tile
        self.receiver_tile = positive_integer(receiver_tile, "receiver_tile")
        self.edge_tile = positive_integer(edge_tile, "edge_tile")
        self.reduction_programs = positive_integer(
            reduction_programs, "reduction_programs"
        )
        self.receiver_extent = positive_integer(receiver_extent, "receiver_extent")
        self.source_extent = positive_integer(source_extent, "source_extent")
        self.edge_extent = positive_integer(edge_extent, "edge_extent")
        self.maximum_degree = nonnegative_integer(maximum_degree, "maximum_degree")
        self.path_count = nonnegative_integer(path_count, "path_count")
        self.coefficient_count = nonnegative_integer(
            coefficient_count, "coefficient_count"
        )
        self.workspace_bytes = nonnegative_integer(workspace_bytes, "workspace_bytes")
        self.fragment_bytes = nonnegative_integer(fragment_bytes, "fragment_bytes")
        self.fragment_budget_bytes = positive_integer(
            fragment_budget_bytes, "fragment_budget_bytes"
        )
        self.maximum_derivative_order = nonnegative_integer(
            maximum_derivative_order, "maximum_derivative_order"
        )


def _require_accumulation(request: AtomisticKernelRequest, /) -> None:
    match request.accumulation:
        case "deterministic" | "compensated":
            return
        case "fast":
            raise ValueError(
                "Accelerated atomistic kernels are receiver-owned and implement only "
                "ordered 'deterministic' or per-event 'compensated' accumulation; "
                "'fast' atomic scatter is not an implemented kernel route."
            )
        case _:
            raise ValueError(f"Unsupported accumulation {request.accumulation!r}.")


def _require_tiles(
    request: AtomisticKernelRequest, limits: AtomisticKernelLimits, /
) -> None:
    tile = request.channel_tile
    if tile % WARPGROUP_LANES or tile & (tile - 1):
        raise ValueError(
            f"channel_tile must be a power-of-two multiple of the {WARPGROUP_LANES} "
            f"warpgroup lanes; got {tile}."
        )
    for name, value in (
        ("channel_tile", tile),
        ("receiver_tile", request.receiver_tile),
        ("edge_tile", request.edge_tile),
    ):
        if value > limits.maximum_tile:
            raise ValueError(
                f"{name}={value} exceeds the admitted maximum tile {limits.maximum_tile}."
            )
    if request.reduction_programs > limits.maximum_reduction_programs:
        raise ValueError(
            f"reduction_programs={request.reduction_programs} exceeds the admitted "
            f"maximum {limits.maximum_reduction_programs}."
        )


def _require_structure(
    request: AtomisticKernelRequest, limits: AtomisticKernelLimits, /
) -> None:
    if request.maximum_degree > limits.maximum_degree:
        raise ValueError(
            f"Coupling degree {request.maximum_degree} exceeds the admitted maximum "
            f"degree {limits.maximum_degree}."
        )
    if not 1 <= request.path_count <= limits.maximum_paths:
        raise ValueError(
            f"Path count {request.path_count} is outside the admitted range "
            f"1..{limits.maximum_paths}."
        )
    if not 1 <= request.coefficient_count <= limits.maximum_coefficients:
        raise ValueError(
            f"Coefficient count {request.coefficient_count} is outside the admitted "
            f"range 1..{limits.maximum_coefficients}."
        )
    if request.workspace_bytes > limits.maximum_workspace_bytes:
        raise ValueError(
            f"Planned per-program workspace {request.workspace_bytes} bytes exceeds the "
            f"admitted {limits.maximum_workspace_bytes} bytes; reduce the channel tile."
        )
    if request.maximum_derivative_order > limits.maximum_derivative_order:
        raise ValueError(
            f"Derivative order {request.maximum_derivative_order} exceeds the admitted "
            f"order {limits.maximum_derivative_order}."
        )
    if request.fragment_bytes > request.fragment_budget_bytes:
        raise ValueError(
            f"One kernel fragment declares {request.fragment_bytes} bytes of operands and "
            f"results, above the declared fragment budget of "
            f"{request.fragment_budget_bytes} bytes; use smaller receiver/edge fragments."
        )


@final
class AtomisticKernelAdmission(StrictModule):
    """Accepted request on one resolved target, with its qualification scope."""

    request: AtomisticKernelRequest
    target: AtomisticKernelTarget
    limits: AtomisticKernelLimits
    qualification_scope: str = eqx.field(static=True)
    admission_id: str = eqx.field(static=True)

    def __init__(
        self,
        request: AtomisticKernelRequest,
        target: AtomisticKernelTarget,
        /,
        *,
        limits: AtomisticKernelLimits = ATOMISTIC_KERNEL_LIMITS,
    ) -> None:
        if not isinstance(request, AtomisticKernelRequest):
            raise TypeError("request must be an AtomisticKernelRequest.")
        if not isinstance(target, AtomisticKernelTarget):
            raise TypeError("target must be an AtomisticKernelTarget.")
        if not isinstance(limits, AtomisticKernelLimits):
            raise TypeError("limits must be AtomisticKernelLimits.")
        _require_accumulation(request)
        _require_tiles(request, limits)
        _require_structure(request, limits)
        match target.target:
            case "cpu_interpret":
                scope = (
                    "CPU Mosaic GPU interpreter reference verification of the kernel "
                    "bodies; not GPU correctness, performance or memory qualification"
                )
            case "cuda":
                scope = (
                    f"runtime-admitted CUDA candidate on {target.device_kind!r} "
                    f"(compute capability {target.compute_capability}) at "
                    f"{request.precision}; numerical, derivative and performance "
                    "qualification of this (device, precision) tuple is separate evidence"
                )
            case _:
                raise ValueError(f"Unsupported atomistic target {target.target!r}.")
        self.request = request
        self.target = target
        self.limits = limits
        self.qualification_scope = scope
        self.admission_id = canonical_fingerprint(
            {
                "kind": "atomistic-kernel-admission",
                "target_id": target.target_id,
                "request": {
                    "precision": request.precision,
                    "accumulation": request.accumulation,
                    "channels": request.channels,
                    "channel_tile": request.channel_tile,
                    "channel_capacity": request.channel_capacity,
                    "receiver_tile": request.receiver_tile,
                    "edge_tile": request.edge_tile,
                    "reduction_programs": request.reduction_programs,
                    "receiver_extent": request.receiver_extent,
                    "source_extent": request.source_extent,
                    "edge_extent": request.edge_extent,
                    "maximum_degree": request.maximum_degree,
                    "path_count": request.path_count,
                    "coefficient_count": request.coefficient_count,
                    "workspace_bytes": request.workspace_bytes,
                    "fragment_bytes": request.fragment_bytes,
                    "fragment_budget_bytes": request.fragment_budget_bytes,
                    "maximum_derivative_order": request.maximum_derivative_order,
                },
                "limits": {
                    "maximum_degree": limits.maximum_degree,
                    "maximum_paths": limits.maximum_paths,
                    "maximum_coefficients": limits.maximum_coefficients,
                    "maximum_tile": limits.maximum_tile,
                    "maximum_workspace_bytes": limits.maximum_workspace_bytes,
                    "maximum_derivative_order": limits.maximum_derivative_order,
                    "maximum_reduction_programs": limits.maximum_reduction_programs,
                },
            }
        )


__all__ = [
    "ATOMISTIC_KERNEL_LIMITS",
    "AtomisticAccelerationTarget",
    "AtomisticKernelAdmission",
    "AtomisticKernelLimits",
    "AtomisticKernelLowering",
    "AtomisticKernelPrecision",
    "AtomisticKernelRequest",
    "AtomisticKernelTarget",
    "MACECouplingRole",
    "mace_coupling_problem_kind",
    "pallas_atomistic_availability",
    "PALLAS_ATOMISTIC_CAPABILITIES",
    "WARPGROUP_LANES",
]
