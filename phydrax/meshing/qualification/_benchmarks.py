# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Scientific benchmark phases with measurement supplied by the caller."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import assert_never, final, Literal, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._bvh import (
    bvh_nearest_items,
    bvh_overlap_pairs,
    bvh_overlap_pairs_host,
    BVHBuildKind,
    BVHBuildPolicy,
    BVHNearestResult,
    BVHPairResult,
    PackedBVH,
    prepare_bvh,
    refit_packed_bvh_bounds,
)
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ...discretization import CellGeometrySpec, CellMesh
from ...discretization._adaptive_simplex import (
    _compiled_coarsen,
    _compiled_refine,
    AdaptiveSimplexLayout,
    AdaptiveSimplexState,
)
from ...discretization.fem import _topology_transfer as projection_kernels
from ...discretization.fem._generic import (
    assemble_masked_finite_element,
    MaskedFiniteElementPlan,
)
from ...discretization.fem._topology_transfer import (
    FiniteElementL2Projection,
    PreparedL2ProjectionTarget,
)
from ...typing import checked, parse
from .._curving import straight_geometry
from .._device_metric import (
    _compiled_adapt,
    adapt_device_metric,
    DeviceMetricLayout,
    DeviceMetricState,
    DeviceMetricUpdate,
)
from .._metric import MeshMetricField
from .._optimization import (
    MeshOptimizationResult,
    minimize_free_coordinates,
    optimize_cell_mesh,
    TargetMatrixOptimizationPlan,
)
from .._quality import BlockQuality, standard_block_quality
from .._result import CellMeshingResult
from ..providers._mmg import MmgAdaptationPlan, MmgProvider
from ..providers._omega_h import _carrier, OmegaHOptions


class BenchmarkMeasure(Protocol):
    def __call__[T](self, call: Callable[[], T], /) -> tuple[T, float]: ...


class BenchmarkCompilerMeasure(Protocol):
    def __call__(
        self, entry: Callable[..., object], /, *arguments: object
    ) -> dict[str, object]: ...


type BenchmarkFailure = Callable[[str, str, dict[str, object]], Exception]


def _clustered_boxes(count: int, seed: int, /) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.random((max(1, count // 256), 3))
    lower = centers[rng.integers(0, centers.shape[0], count)] + 0.02 * (
        rng.standard_normal((count, 3))
    )
    return lower, lower + 0.004 + 0.004 * rng.random((count, 3))


@eqx.filter_jit
def _refit_and_query(
    bvh: PackedBVH, lower: ArrayLike, upper: ArrayLike, queries: ArrayLike
) -> BVHNearestResult:
    refitted = refit_packed_bvh_bounds(bvh, lower, upper)
    return bvh_nearest_items(refitted, queries, k=4)


def qualify_bvh_scaling(
    resolution: int, *, measure: BenchmarkMeasure
) -> dict[str, object]:
    """Measure every BVH build strategy, refit, and nearest-item consumer."""
    item_count = 1024 * resolution
    lower, upper = _clustered_boxes(item_count, resolution)
    moved_lower = lower + 0.001
    queries = np.random.default_rng(7).random((1024, 3))
    stages: dict[str, float] = {}
    quality: dict[str, int] = {}
    for kind in BVHBuildKind:
        policy = BVHBuildPolicy(kind, leaf_size=8)
        bvh, stages[f"build_{kind.value}"] = measure(
            lambda policy=policy: prepare_bvh(lower, upper, policy=policy)
        )
        _, stages[f"refit_query_cold_{kind.value}"] = measure(
            lambda bvh=bvh: jax.block_until_ready(
                _refit_and_query(bvh, moved_lower, moved_lower + 0.006, queries)
            )
        )
        _, stages[f"refit_query_warm_{kind.value}"] = measure(
            lambda bvh=bvh: jax.block_until_ready(
                _refit_and_query(bvh, moved_lower, moved_lower + 0.006, queries)
            )
        )
        quality[f"depth_{kind.value}"] = bvh.max_depth
        quality[f"nodes_{kind.value}"] = bvh.node_count
    return {
        "resolution": resolution,
        "stages_seconds": stages,
        "quality": quality,
        "counts": {"items": item_count, "queries": queries.shape[0]},
    }


@eqx.filter_jit
def _device_overlap_pairs(
    first: PackedBVH, second: PackedBVH, *, capacity: int
) -> BVHPairResult:
    return bvh_overlap_pairs(first, second, capacity=capacity)


@checked
def qualify_overlap_pair_scaling(
    resolution: int, *, measure: BenchmarkMeasure, failure: BenchmarkFailure
) -> dict[str, object]:
    """Compare host and bounded device pair search on shared clustered boxes."""
    item_count = 1024 * resolution
    first_lower, first_upper = _clustered_boxes(item_count, resolution)
    second_lower = first_lower + 0.004 * np.random.default_rng(
        resolution
    ).standard_normal(first_lower.shape)
    second_upper = second_lower + (first_upper - first_lower)
    policy = BVHBuildPolicy(BVHBuildKind.SAH, leaf_size=8)
    first = prepare_bvh(first_lower, first_upper, policy=policy, dtype=np.float64)
    second = prepare_bvh(second_lower, second_upper, policy=policy, dtype=np.float64)
    (host_first, _), host_seconds = measure(lambda: bvh_overlap_pairs_host(first, second))
    capacity = max(1, 2 * host_first.shape[0])
    _, device_cold = measure(
        lambda: jax.block_until_ready(
            _device_overlap_pairs(first, second, capacity=capacity)
        )
    )
    device, device_warm = measure(
        lambda: jax.block_until_ready(
            _device_overlap_pairs(first, second, capacity=capacity)
        )
    )
    if bool(device.overflow) or int(device.count) != host_first.shape[0]:
        raise failure(
            "benchmark_overlap_pairs",
            "Device overlap pair count or capacity disagrees with the host search.",
            {"device": device, "host_first": host_first},
        )
    return {
        "resolution": resolution,
        "stages_seconds": {
            "host": host_seconds,
            "device_cold": device_cold,
            "device_warm": device_warm,
        },
        "counts": {"items": item_count, "pairs": host_first.shape[0]},
    }


@eqx.filter_jit
@checked
def benchmark_boundary_layer_quality(
    coordinates: Array, connectivity: Mapping[str, Array]
) -> dict[str, BlockQuality]:
    """Evaluate native corner quality for the declared family connectivity banks."""
    return {
        kind: standard_block_quality(kind, coordinates[rows], None)
        for kind, rows in connectivity.items()
    }


@eqx.filter_jit
@checked
def benchmark_adapted_quality(
    coordinates: Array, connectivity: Mapping[str, Array]
) -> dict[str, BlockQuality]:
    """Evaluate adapted-mesh quality with an independent cold compiler cache."""
    return {
        kind: standard_block_quality(kind, coordinates[rows], None)
        for kind, rows in connectivity.items()
    }


def benchmark_projection_compilation_counts() -> dict[str, int]:
    """Snapshot the target-factor, solve, action, and tabulation compiler caches."""
    entries = {
        "_factor_target_mass": projection_kernels._factor_target_mass,
        "_mass_condition": projection_kernels._mass_condition,
        "_mixed_mass_action": projection_kernels._mixed_mass_action,
        "_target_mass_solve": projection_kernels._target_mass_solve,
        "_tabulate_basis": projection_kernels._tabulate_basis,
    }
    # Equinox compiler caches are runtime diagnostics, not numerical leaves.
    return {name: entry._cached._cache_size() for name, entry in entries.items()}  # ty: ignore[unresolved-attribute]


@checked
def measure_projection_compilation(
    prepared: PreparedL2ProjectionTarget,
    projection: FiniteElementL2Projection,
    block: Array,
    target_size: int,
    *,
    compiler: BenchmarkCompilerMeasure,
) -> dict[str, dict[str, object]]:
    """Measure the mixed-mass block action and the matching retained target solve."""
    mass = projection.mixed_mass
    return {
        "l2_mixed_mass_action_block3": compiler(
            projection_kernels._mixed_mass_action,
            mass.relation,
            mass.coefficients,
            block,
        ),
        "l2_target_solve_block3": compiler(
            projection_kernels._target_mass_solve,
            prepared.factorization,
            jnp.zeros((target_size, block.shape[1])),
        ),
    }


@checked
def measure_bisection_compilation(
    layout: AdaptiveSimplexLayout,
    state: AdaptiveSimplexState,
    plan: MaskedFiniteElementPlan,
    empty: Array,
    *,
    compiler: BenchmarkCompilerMeasure,
) -> dict[str, dict[str, object]]:
    """Measure both topology transitions and their masked finite-element consumer."""
    return {
        "refine": compiler(_compiled_refine, layout, state, empty),
        "coarsen": compiler(_compiled_coarsen, layout, state, empty),
        "masked_fe_assembly": compiler(assemble_masked_finite_element, plan, state.mesh),
    }


def benchmark_bisection_compilation_counts() -> dict[str, int]:
    """Snapshot kernel compilation counts for a fixed-capacity bisection cycle."""
    entries = {
        "refine": _compiled_refine,
        "coarsen": _compiled_coarsen,
        "masked_fe_assembly": assemble_masked_finite_element,
    }
    return {name: entry._cached._cache_size() for name, entry in entries.items()}  # ty: ignore[unresolved-attribute]


@final
class BenchmarkMetricExecution(StrictModule):
    cold: DeviceMetricUpdate
    warm: DeviceMetricUpdate
    cost: dict[str, object]
    cold_seconds: float = eqx.field(static=True)
    warm_seconds: float = eqx.field(static=True)


@checked
def execute_benchmark_device_metric(
    layout: DeviceMetricLayout,
    state: DeviceMetricState,
    *,
    measure: BenchmarkMeasure,
    compiler: BenchmarkCompilerMeasure,
) -> BenchmarkMetricExecution:
    """Lower the admitted device bucket, then execute its cold and warm paths."""
    cost = compiler(_compiled_adapt, layout, state)
    cold, cold_seconds = measure(
        lambda: jax.block_until_ready(adapt_device_metric(layout, state))
    )
    warm, warm_seconds = measure(
        lambda: jax.block_until_ready(adapt_device_metric(layout, state))
    )
    return BenchmarkMetricExecution(
        cold=cold,
        warm=warm,
        cost=cost,
        cold_seconds=cold_seconds,
        warm_seconds=warm_seconds,
    )


def benchmark_device_metric_compilation_count() -> int:
    """Snapshot the measured device-metric entry after its certified host commit."""
    return _compiled_adapt._cached._cache_size()  # ty: ignore[unresolved-attribute]


@final
class BenchmarkOptimizationExecution(StrictModule):
    result: MeshOptimizationResult
    cost: dict[str, object]
    cold_seconds: float = eqx.field(static=True)
    warm_seconds: float = eqx.field(static=True)
    minimization_seconds: float = eqx.field(static=True)


@checked
def execute_benchmark_optimization(
    plan: TargetMatrixOptimizationPlan,
    mesh: CellMesh,
    free: Array,
    contract: SpatialCoordinateContract,
    *,
    measure: BenchmarkMeasure,
    compiler: BenchmarkCompilerMeasure,
) -> BenchmarkOptimizationExecution:
    """Measure the fixed-coordinate minimizer and its certified cold/warm consumers."""
    arguments = (
        plan.energy,
        plan.method,
        plan.termination,
        jnp.asarray(mesh.coordinates, dtype=jnp.float64),
        free,
        plan.lower[free],
        plan.upper[free],
    )
    cost = compiler(minimize_free_coordinates, *arguments)
    result, cold = measure(lambda: optimize_cell_mesh(plan, contract))
    _, warm = measure(lambda: optimize_cell_mesh(plan, contract))
    _, warm_minimization = measure(
        lambda: jax.block_until_ready(minimize_free_coordinates(*arguments))
    )
    return BenchmarkOptimizationExecution(
        result=result,
        cost=cost,
        cold_seconds=cold,
        warm_seconds=warm,
        minimization_seconds=warm_minimization,
    )


def _squeezed(points: np.ndarray, /) -> np.ndarray:
    """Gaussian contraction retaining the authored inverted core and thin shell."""
    offset = points - 0.5
    weight = np.exp(-8.0 * np.sum(offset * offset, axis=1, keepdims=True))
    return points - 1.2 * offset * weight


@checked
def prepare_squeezed_benchmark_geometry(mesh: CellMesh, degree: int) -> CellGeometrySpec:
    """Prepare straight high-order nodes and apply the authored Gaussian squeeze."""
    straight = straight_geometry(mesh, degree)
    names = straight.block_names
    return CellGeometrySpec(
        dict(zip(names, straight.elements, strict=True)),
        dict(zip(names, straight.geometry_dofs, strict=True)),
        _squeezed(np.asarray(straight.coordinates, dtype=np.float64)),
    )


type BenchmarkProviderKind = Literal["mmg", "omega_h"]


@final
class BenchmarkProviderInput(StrictModule):
    """Prepared provider request and binary array payload for worker measurements."""

    request: MmgAdaptationPlan | MeshMetricField
    arrays: dict[str, np.ndarray]


@checked
def prepare_benchmark_provider_input(
    source: CellMeshingResult,
    metric: MeshMetricField,
    provider_name: BenchmarkProviderKind,
    executable: str,
) -> BenchmarkProviderInput:
    """Prepare the same classified worker request used by provider execution."""
    provider = parse(provider_name, BenchmarkProviderKind, "provider_name")
    match provider:
        case "mmg":
            plan = MmgProvider(executable=executable).plan(source, metric=metric)
            return BenchmarkProviderInput(request=plan, arrays=plan._encoding.arrays)
        case "omega_h":
            carrier = _carrier(source, metric, (), OmegaHOptions())
            return BenchmarkProviderInput(request=metric, arrays=carrier.arrays)
        case _:
            assert_never(provider)


__all__ = [
    "BenchmarkMeasure",
    "BenchmarkCompilerMeasure",
    "BenchmarkFailure",
    "BenchmarkMetricExecution",
    "BenchmarkOptimizationExecution",
    "qualify_bvh_scaling",
    "qualify_overlap_pair_scaling",
    "benchmark_boundary_layer_quality",
    "benchmark_adapted_quality",
    "benchmark_device_metric_compilation_count",
    "benchmark_projection_compilation_counts",
    "measure_projection_compilation",
    "measure_bisection_compilation",
    "benchmark_bisection_compilation_counts",
    "execute_benchmark_device_metric",
    "execute_benchmark_optimization",
    "prepare_squeezed_benchmark_geometry",
    "BenchmarkProviderInput",
    "prepare_benchmark_provider_input",
    "BenchmarkProviderKind",
]
