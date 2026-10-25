#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Isolated BRep containment, moments, and orientation-audit comparisons."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING

import jax
import numpy as np

from phydrax.geometry.brep import BRepQueryResourceError
from phydrax.meshing.qualification import (
    benchmark_box,
    corpus_route_points,
    corpus_route_query,
    qualify_brep_containment,
    qualify_brep_moments,
    qualify_brep_orientation_audit,
    trimmed_polynomial_nurbs_box,
)
from tools.meshing_qualification._runtime import peak_resident_bytes, StageLedger


if TYPE_CHECKING:
    from phydrax.geometry._interval_enclosure import NativeIntervalProgram

type _IntervalBounds = tuple[np.ndarray, np.ndarray]
type _HomotopyInputs = tuple[
    list[tuple[_IntervalBounds, _IntervalBounds]],
    list[tuple[_IntervalBounds, np.ndarray, _IntervalBounds]],
    list[tuple[np.ndarray, _IntervalBounds]],
]


def _measure_samples[T](
    call: Callable[[], T], /, *, repeats: int
) -> tuple[T, dict[str, object]]:
    """Keep clean samples separate from profiled compilation/accounting processes."""

    def synchronized_call() -> T:
        return jax.block_until_ready(call())

    if os.environ.get("PHYDRAX_MESHING_ACCOUNTING_DIAGNOSTICS") == "1":
        diagnostics: dict[str, object] = {}
        result = StageLedger(diagnostics).run("accounting_diagnostic", synchronized_call)
        return result, {
            "headline_timing_available": False,
            "diagnostic_measurements": diagnostics["stage_measurements"],
            "diagnostic_timing_is_headline": False,
            "peak_resident_bytes": peak_resident_bytes(),
        }
    started = perf_counter()
    result = synchronized_call()
    first_execution = perf_counter() - started
    warm: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        result = synchronized_call()
        warm.append(perf_counter() - started)
    return result, {
        "headline_timing_available": True,
        "first_execution_seconds": first_execution,
        "warm_samples_seconds": warm,
        "warm_median_seconds": float(np.median(np.asarray(warm, dtype=np.float64))),
        "peak_resident_bytes": peak_resident_bytes(),
        "diagnostic_measurements": None,
        "diagnostic_not_run_reason": "accounting diagnostics require a separate process",
    }


def benchmark_cell_validity(resolution: int, repeats: int) -> dict[str, object]:
    """Measure fresh exact P2 validity on independent maps, not global embedding."""
    from fractions import Fraction

    from phydrax import model_array_bytes
    from phydrax.discretization._cell_geometry import owned_coordinate_preparation
    from phydrax.discretization._cell_geometry_validity import (
        cell_validity_preparation_bounds,
        CellValidityCertificate,
        CellValidityPolicy,
        certify_cell_geometry_validity,
    )
    from phydrax.discretization._cell_mesh import CellMesh
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.meshing import meshcore_available
    from phydrax.meshing._curving import straight_geometry

    if not meshcore_available():
        raise RuntimeError(
            "cell-validity requires the installed native meshcore provider"
        )
    started = perf_counter()
    reference = np.asarray(
        reference_cell_topology("tetrahedron").vertices, dtype=np.float64
    )
    mesh = CellMesh.from_tetrahedra(
        np.tile(reference, (resolution, 1)),
        np.arange(4 * resolution, dtype=np.int64).reshape((-1, 4)),
    )
    straight = straight_geometry(mesh, 2)
    points = np.asarray(straight.coordinates, dtype=np.float64)
    x, y, z = points.T
    alpha = Fraction(1, 16)
    a = float(alpha)
    geometry = straight.with_coordinates(
        np.stack((x + a * x * y, y + a * y * z, z + a * z * x), axis=1)
    )
    policy = CellValidityPolicy()
    work, memory, queries = cell_validity_preparation_bounds(geometry, policy, mesh=mesh)
    preparation = perf_counter() - started

    def prove() -> tuple[CellValidityCertificate, int, int]:
        # A new request each time: warm samples reuse numerical preparation,
        # never a previously issued validity certificate.
        with owned_coordinate_preparation(
            work, memory, maximum_geometry_queries=queries
        ) as budget:
            certificate = certify_cell_geometry_validity(
                geometry, mesh=mesh, policy=policy, budget=budget
            )
            return certificate, budget.work_units, budget.peak_bytes_upper

    (certificate, actual_work, peak), timings = _measure_samples(prove, repeats=repeats)
    # det J = 1+a(x+y+z)+a²(xy+yz+zx)+2a³xyz. Its degree-three
    # simplex Bernstein extrema are 1 and 1+a+a²/2+a³/3.
    exact_upper = Fraction(1) + alpha + alpha**2 / 2 + alpha**3 / 3
    upper = float(exact_upper)
    if Fraction(upper) < exact_upper:
        upper = float(np.nextafter(upper, np.inf))
    if (
        not certificate.all_certified
        or np.any(np.asarray(certificate.depth) != 0)
        or np.any(np.asarray(certificate.determinant_lower) != 1.0)
        or np.any(np.asarray(certificate.determinant_upper) != upper)
    ):
        raise RuntimeError("P2 validity must match the independent exact cubic controls")
    return {
        "status": "completed",
        "scope": "fresh-local-bernstein-validity-not-embedding-or-workflow",
        "resolution": resolution,
        "resolution_semantics": "independent-quadratic-tetrahedral-maps",
        "repeats": repeats,
        "preparation_seconds": preparation,
        "compilation": "ahead-of-time-meshcore-build",
        "work_units": actual_work,
        "managed_peak_bytes": peak,
        "logical_retained_bytes": model_array_bytes((mesh, geometry, certificate)),
        "certificate_id": certificate.certificate_id,
        "determinant_lower": 1.0,
        "determinant_upper": upper,
        **timings,
    }


def benchmark_brep_preconditioners(resolution: int, repeats: int) -> dict[str, object]:
    """Separate actual CAD preconditioner preparation, compilation and execution."""
    from phydrax.geometry.brep._intersection_kernel import (
        _preconditioner_kernel,
        _PRECONDITIONER_ROWS,
        _preconditioner_rows,
        approximate_inverses,
    )

    maximum_rows = _PRECONDITIONER_ROWS[-1]
    full, remainder = divmod(resolution, maximum_rows)
    buckets = sorted(
        ({maximum_rows} if full else set())
        | ({_preconditioner_rows(remainder)} if remainder else set())
    )
    rng = np.random.default_rng(971)
    records: list[dict[str, object]] = []
    for dimension in (3, 4):
        started = perf_counter()
        identity = np.eye(dimension, dtype=np.float64)
        matrices = rng.uniform(
            -0.25, 0.25, size=(resolution, dimension, dimension)
        ).astype(np.float64)
        matrices += 2.0 * identity[None]
        preparation = perf_counter() - started
        programs: list[dict[str, object]] = []
        kernel = _preconditioner_kernel(dimension)
        for rows in buckets:
            representative = np.broadcast_to(
                identity, (rows, dimension, dimension)
            ).copy()
            started = perf_counter()
            lowered = kernel.lower(representative)
            lowering = perf_counter() - started
            started = perf_counter()
            compiled = lowered.compile()
            compilation = perf_counter() - started
            memory = compiled.memory_analysis()
            programs.append(
                {
                    "rows": rows,
                    "lowering_seconds": lowering,
                    "compile_seconds": compilation,
                    "temporary_bytes": None
                    if memory is None
                    else memory.temp_size_in_bytes,
                    "output_bytes": None
                    if memory is None
                    else memory.output_size_in_bytes,
                    "argument_bytes": None
                    if memory is None
                    else memory.argument_size_in_bytes,
                    "generated_code_bytes": None
                    if memory is None
                    else memory.generated_code_size_in_bytes,
                    "memory_measurement": "unsupported"
                    if memory is None
                    else "compiler-analysis",
                }
            )
        result, timings = _measure_samples(
            lambda: approximate_inverses(matrices), repeats=repeats
        )
        # This is a host-only independent reference, not a replacement solver.
        reference = np.linalg.solve(matrices, np.broadcast_to(identity, matrices.shape))
        np.testing.assert_allclose(result, reference, rtol=2e-15, atol=2e-15)
        records.append(
            {
                "dimension": dimension,
                "active_rows": resolution,
                "evaluated_rows": full * maximum_rows
                + (_preconditioner_rows(remainder) if remainder else 0),
                "preparation_seconds": preparation,
                "programs": programs,
                "logical_retained_bytes": matrices.nbytes + result.nbytes,
                "maximum_reference_error": float(np.max(np.abs(result - reference))),
                **timings,
            }
        )
    return {
        "status": "completed",
        "scope": "cad-preconditioner-not-whole-workflow",
        "resolution": resolution,
        "resolution_semantics": "actual-matrices",
        "repeats": repeats,
        "cache_policy": "compile-required-shapes-before-boundary-execution",
        "records": records,
    }


def benchmark_brep_interval_program(resolution: int, repeats: int) -> dict[str, object]:
    """Compare prepared native and interpreted interval execution on identical boxes."""
    from dataclasses import replace

    import jax.numpy as jnp
    from jax import Array

    from phydrax._meshcore import (
        meshcore_provider_installed,
        NativeExecutionLimits,
        NativeExecutionScope,
    )
    from phydrax.geometry._interval_enclosure import prepare_interval_function

    if not meshcore_provider_installed():
        raise ModuleNotFoundError("interval benchmark requires the meshcore provider")

    def coordinates(point: Array) -> Array:
        x, y, z = point[0], point[1], point[2]
        return (
            jnp.cos(x) * jnp.sin(y) + jnp.sin(x) * jnp.cos(y) + jnp.sum(point * point) - z
        )

    started = perf_counter()
    rng = np.random.default_rng(971)
    lower = rng.uniform(-1.0, 1.0, size=(resolution, 3)).astype(np.float64)
    upper = lower + 0.01
    input_preparation = perf_counter() - started
    started = perf_counter()
    prepared = prepare_interval_function(
        coordinates, 3, batch_capacity=min(64, resolution)
    )
    preparation = perf_counter() - started
    native = prepared.native
    if native is None:
        raise RuntimeError("interval benchmark requires a lowerable native program")
    interpreted = replace(prepared, native=None)
    native_bounds, native_times = _measure_samples(
        lambda: prepared.evaluate(lower, upper),
        repeats=repeats,
    )
    reference_bounds, reference_times = _measure_samples(
        lambda: interpreted.evaluate(lower, upper),
        repeats=repeats,
    )
    # Consecutive chunk results can coexist at the assignment boundary.
    capacity = 4096 + 4 * native.template_lower.nbytes
    started = perf_counter()
    with NativeExecutionScope(NativeExecutionLimits(0, 0, 0, capacity, np.inf)) as scope:
        scope.observe_managed_admission()
        scope_preparation = perf_counter() - started
        admitted_bounds, admitted_times = _measure_samples(
            lambda: prepared.evaluate(lower, upper),
            repeats=repeats,
        )
        started = perf_counter()
    scope_settlement = perf_counter() - started
    evidence = scope.evidence
    if evidence is None or evidence.managed_admission_peak_bytes_upper is None:
        raise RuntimeError("interval benchmark requires complete managed-memory evidence")
    for actual, reference in zip(admitted_bounds, reference_bounds, strict=True):
        np.testing.assert_array_equal(actual.view(np.uint64), reference.view(np.uint64))
    for actual, reference in zip(native_bounds, reference_bounds, strict=True):
        np.testing.assert_array_equal(actual.view(np.uint64), reference.view(np.uint64))
    # Independent nominal host samples check inclusion, not only backend agreement.
    midpoint = (lower + upper) * 0.5
    x, y, z = midpoint.T
    sample = (
        np.cos(x) * np.sin(y)
        + np.sin(x) * np.cos(y)
        + np.sum(midpoint * midpoint, axis=1)
        - z
    )
    if np.any(sample < native_bounds[0]) or np.any(sample > native_bounds[1]):
        raise RuntimeError("interval benchmark excluded an independent interior sample")
    program_bytes = sum(
        bank.nbytes
        for bank in (
            native.instructions,
            native.indices,
            native.parameters,
            native.template_lower,
            native.template_upper,
        )
    )
    return {
        "status": "completed",
        "scope": "prepared-interval-program-not-whole-workflow",
        "resolution": resolution,
        "resolution_semantics": "input-boxes",
        "repeats": repeats,
        "batch_capacity": prepared.batch_capacity,
        "input_preparation_seconds": input_preparation,
        "tracing_and_lowering_seconds": preparation,
        "compilation": "ahead-of-time-meshcore-build",
        "native_program_bytes": program_bytes,
        "native_workspace_bytes": native.template_lower.nbytes
        + native.template_upper.nbytes,
        "logical_retained_numeric_bytes": (
            program_bytes
            + lower.nbytes
            + upper.nbytes
            + sum(bound.nbytes for bound in native_bounds)
        ),
        "bit_identical_to_interpreter": True,
        "native": native_times,
        "native_with_resource_scope": admitted_times,
        "resource_scope_preparation_seconds": scope_preparation,
        "resource_scope_settlement_seconds": scope_settlement,
        "native_managed_peak_bytes": evidence.managed_admission_peak_bytes_upper,
        "interpreter": reference_times,
    }


def benchmark_brep_quadratic_bounds(resolution: int, repeats: int) -> dict[str, object]:
    """Measure the native bound filter independently of complete certificates."""
    from phydrax._meshcore import simplex_quadratic_bounds

    started = perf_counter()
    rng = np.random.default_rng(971)
    coefficients = rng.integers(-20, 21, size=(resolution, 3, 4, 4)).astype(np.float64)
    coefficients = (coefficients + coefficients.swapaxes(-1, -2)) / 16.0
    transform = np.broadcast_to(np.eye(4, dtype=np.float64), (resolution, 4, 4)).copy()
    transform[:, 0, :2] = 0.5
    preparation = perf_counter() - started
    bounds, timings = _measure_samples(
        lambda: simplex_quadratic_bounds(
            coefficients, coefficients, transform, transform
        ),
        repeats=repeats,
    )
    lower, upper = bounds
    # Small dyadic coefficients and half-coordinate maps make these ordinary
    # host matrix products exact; this reference is independent of the kernel.
    controls = transform[:, None] @ coefficients @ transform[:, None].swapaxes(-1, -2)
    exact_lower = controls.min(axis=(-1, -2))
    exact_upper = controls.max(axis=(-1, -2))
    if np.any(lower > exact_lower) or np.any(upper < exact_upper):
        raise RuntimeError(
            "native quadratic bounds must contain the exact dyadic controls"
        )
    return {
        "status": "completed",
        "scope": "native-simplex-bound-filter-not-global-embedding",
        "resolution": resolution,
        "resolution_semantics": "quadratic-tetrahedron-maps",
        "repeats": repeats,
        "preparation_seconds": preparation,
        "logical_retained_bytes": coefficients.nbytes
        + transform.nbytes
        + lower.nbytes
        + upper.nbytes,
        "compilation": "ahead-of-time-meshcore-build",
        "maximum_outward_slack": float(
            max(np.max(exact_lower - lower), np.max(upper - exact_upper))
        ),
        **timings,
    }


def benchmark_brep_containment(resolution: int, repeats: int) -> dict[str, object]:
    """Retain fixed-corpus measurements and declared-limit refusal evidence."""
    campaigns: dict[str, object] = {}
    refusal: dict[str, object] | None = None
    for route in ("sphere", "revolution", "affine", "nurbs"):
        if refusal is not None:
            campaigns[route] = {
                "status": "not-attempted",
                "reason": "containment-case-stopped-after-resource-refusal",
                "timing_is_headline": False,
                "seconds": None,
            }
            continue
        query = corpus_route_query(route)
        band = corpus_route_points(query, route)
        count = resolution**3
        outside = np.empty((count, 3), dtype=np.float64)
        outside[:, 0] = np.linspace(10, 11, count, dtype=np.float64)
        outside[:, 1:] = 10
        points = jax.device_put(np.concatenate((band, outside)))
        try:
            result = qualify_brep_containment(
                query, points, sampling=partial(_measure_samples, repeats=repeats)
            )
        except BRepQueryResourceError as error:
            faces = query.solid_faces[0]
            refusal = {
                "route": route,
                "resource": error.resource,
                "requested": error.requested,
                "remaining": error.remaining,
                "point_count": points.shape[0],
                "solid_count": 1,
                "face_count": len(faces),
                "coedge_count": sum(
                    len(query.faces[face].coedge_edges) for face in faces
                ),
                "model_id": query.model.model_id,
                "policy_id": query.policy.policy_id,
                "policy_limits": {
                    "maximum_operations": query.policy.maximum_operations,
                    "maximum_points": query.policy.maximum_points,
                    "maximum_scratch_bytes": query.policy.maximum_scratch_bytes,
                    "maximum_subdivisions": query.policy.maximum_subdivisions,
                    "maximum_depth": query.policy.maximum_depth,
                },
            }
            campaigns[route] = {
                "status": "resource-refused",
                "refusal": refusal,
                "seconds": None,
                "timing_is_headline": False,
                "headline_timing_available": False,
                "threshold_met": False,
                "decisions_identical": None,
                "baseline": None,
                "boundary_filter": None,
            }
        else:
            campaigns[route] = {"status": "completed", **result}
    return {
        "resolution": resolution,
        "repeats": repeats,
        "status": "completed" if refusal is None else "resource-refused",
        "seconds": None,
        "headline_timing_available": (
            refusal is None
            and os.environ.get("PHYDRAX_MESHING_ACCOUNTING_DIAGNOSTICS") != "1"
        ),
        "threshold_met": False if refusal is not None else None,
        "timing_is_headline": (
            refusal is None
            and os.environ.get("PHYDRAX_MESHING_ACCOUNTING_DIAGNOSTICS") != "1"
        ),
        "refusal": refusal,
        "campaigns": campaigns,
    }


def benchmark_brep_containment_campaign(
    resolutions: tuple[int, ...],
    repeats: int,
    /,
    *,
    run_case: Callable[[int], dict[str, object]],
) -> list[dict[str, object]]:
    """Preserve completed cells without retrying a refused or larger request."""
    results: list[dict[str, object]] = []
    refused_resolution: int | None = None
    for resolution in resolutions:
        if refused_resolution is not None and resolution >= refused_resolution:
            results.append(
                {
                    "resolution": resolution,
                    "repeats": repeats,
                    "status": "not-attempted",
                    "reason": "same-or-larger-containment-request-exceeds-declared-policy",
                    "refused_resolution": refused_resolution,
                    "seconds": None,
                    "timing_is_headline": False,
                    "headline_timing_available": False,
                    "threshold_met": False,
                }
            )
            continue
        result = run_case(resolution)
        results.append(result)
        if result["status"] == "resource-refused":
            refused_resolution = resolution
    return results


def benchmark_brep_moments(resolution: int, repeats: int) -> dict[str, object]:
    result = qualify_brep_moments(
        trimmed_polynomial_nurbs_box(),
        resolution,
        sampling=partial(_measure_samples, repeats=repeats),
    )
    return {"resolution": resolution, "repeats": repeats, **result}


def benchmark_brep_orientation_audit(resolution: int, repeats: int) -> dict[str, object]:
    result = qualify_brep_orientation_audit(
        benchmark_box(),
        sampling=partial(_measure_samples, repeats=repeats),
    )
    return {"resolution": resolution, "repeats": repeats, **result}


def _homotopy_reference_product(
    left: _IntervalBounds, right: _IntervalBounds
) -> _IntervalBounds:
    from phydrax.geometry._interval_enclosure import interval_add, interval_multiply

    products = interval_multiply(
        (left[0][:, :, None], left[1][:, :, None]),
        (right[0][None, :, :], right[1][None, :, :]),
    )
    shape = (left[0].shape[0], right[0].shape[1])
    total = (np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.float64))
    for index in range(left[0].shape[1]):
        total = interval_add(
            total,
            (np.asarray(products[0])[:, index, :], np.asarray(products[1])[:, index, :]),
        )
    return np.asarray(total[0]), np.asarray(total[1])


def _homotopy_ordered_product(
    left: _IntervalBounds, right: _IntervalBounds
) -> _IntervalBounds:
    from phydrax.geometry._interval_enclosure import interval_dot

    lower = np.empty((left[0].shape[0], right[0].shape[1]), dtype=np.float64)
    upper = np.empty_like(lower)
    for row in range(lower.shape[0]):
        for column in range(lower.shape[1]):
            low, high = interval_dot(
                (left[0][row], left[1][row]),
                (right[0][:, column], right[1][:, column]),
            )
            lower[row, column], upper[row, column] = low, high
    return lower, upper


def _homotopy_reference_spd(
    jacobian: _IntervalBounds, projection: np.ndarray, transform: _IntervalBounds
) -> bool:
    from phydrax.geometry._interval_enclosure import (
        interval_add,
        interval_multiply,
        interval_subtract,
    )

    lower, upper = _homotopy_reference_product(
        _homotopy_reference_product((projection, projection), jacobian), transform
    )
    off = interval_multiply(
        interval_add((lower[0, 1], upper[0, 1]), (lower[1, 0], upper[1, 0])),
        (np.asarray(0.5), np.asarray(0.5)),
    )
    determinant = interval_subtract(
        interval_multiply((lower[0, 0], upper[0, 0]), (lower[1, 1], upper[1, 1])),
        interval_multiply(off, off),
    )
    return bool(lower[0, 0] > 0 and determinant[0] > 0)


def _homotopy_reference_net(
    direction: np.ndarray, net: _IntervalBounds
) -> tuple[float, float]:
    from phydrax.geometry._interval_enclosure import interval_add, interval_multiply

    products = interval_multiply((direction[None, :], direction[None, :]), net)
    total = (
        np.zeros(net[0].shape[0], dtype=np.float64),
        np.zeros(net[0].shape[0], dtype=np.float64),
    )
    for axis in range(3):
        total = interval_add(
            total, (np.asarray(products[0])[:, axis], np.asarray(products[1])[:, axis])
        )
    return float(np.min(total[0])), float(np.max(total[1]))


def _require_homotopy_endpoint_bytes(
    actual: _IntervalBounds, expected: _IntervalBounds
) -> None:
    for value, reference in zip(actual, expected, strict=True):
        if (
            value.shape != reference.shape
            or value.dtype != reference.dtype
            or value.tobytes() != reference.tobytes()
        ):
            raise AssertionError("Ordered interval endpoint bytes differ")


def _homotopy_operand_sets(resolution: int) -> _HomotopyInputs:
    left_base = np.asarray(((1.0, -2.0, 0.5), (0.25, 3.0, -0.125)), dtype=np.float64)
    right_base = np.asarray(((0.5, 0.25), (-2.0, 1.0), (0.125, -0.5)), dtype=np.float64)
    jacobian_base = np.asarray(
        ((1.0, 0.125), (0.25, 2.0), (0.125, -0.25)), dtype=np.float64
    )
    projection = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float64)
    identity = np.eye(2, dtype=np.float64)
    direction_base = np.asarray((0.25, -1.0, 0.5), dtype=np.float64)
    controls_base = np.arange(18, dtype=np.float64).reshape(6, 3) / 16.0 - 0.5
    left_pattern = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
    right_pattern = np.arange(1, 7, dtype=np.float64).reshape(3, 2)
    jacobian_pattern = np.asarray(
        ((1.0, -1.0), (0.5, 1.0), (-0.5, 0.25)), dtype=np.float64
    )
    direction_pattern = np.asarray((0.5, -0.25, 0.125), dtype=np.float64)
    control_pattern = np.arange(1, 19, dtype=np.float64).reshape(6, 3)
    linear_sets: list[tuple[_IntervalBounds, _IntervalBounds]] = []
    spd_sets: list[tuple[_IntervalBounds, np.ndarray, _IntervalBounds]] = []
    net_sets: list[tuple[np.ndarray, _IntervalBounds]] = []
    signatures: tuple[set[bytes], set[bytes], set[bytes]] = (set(), set(), set())
    for index in range(resolution):
        phase = np.float64(index) / np.float64(resolution)
        left_center = left_base + phase * (2.0**-12) * left_pattern
        right_center = right_base - phase * (2.0**-12) * right_pattern
        left = (left_center - 0.01, left_center + 0.01)
        right = (right_center - 0.02, right_center + 0.02)
        linear_sets.append((left, right))
        signatures[0].add(left[0].tobytes() + right[0].tobytes())
        jacobian_center = jacobian_base + phase * 1.0e-3 * jacobian_pattern
        jacobian = (jacobian_center - 0.01, jacobian_center + 0.01)
        spd_sets.append((jacobian, projection, (identity, identity)))
        signatures[1].add(jacobian[0].tobytes())
        direction = direction_base + phase * 1.0e-3 * direction_pattern
        net_center = controls_base + phase * 1.0e-3 * control_pattern
        net = (net_center - 0.01, net_center + 0.01)
        net_sets.append((direction, net))
        signatures[2].add(direction.tobytes() + net[0].tobytes())
    if any(len(values) != resolution for values in signatures):
        raise RuntimeError("homotopy resolution must contain distinct operand sets")
    return linear_sets, spd_sets, net_sets


def _measure_homotopy_program(
    name: str,
    operand_shape: str,
    factory: Callable[[], NativeIntervalProgram],
    workload: Callable[[], int],
    reference: Callable[[], int],
    resolution: int,
    repeats: int,
) -> dict[str, object]:
    from phydrax import _meshcore

    started = perf_counter()
    program = factory()
    preparation_seconds = perf_counter() - started
    _, unscoped = _measure_samples(workload, repeats=repeats)
    _, reference_timings = _measure_samples(reference, repeats=repeats)
    scope = _meshcore.NativeExecutionScope(
        _meshcore.NativeExecutionLimits(0, 0, 0, 8 * 1024**2, 120.0)
    )
    started = perf_counter()
    with scope:
        scope_preparation_seconds = perf_counter() - started
        _, scoped = _measure_samples(workload, repeats=repeats)
        settlement_started = perf_counter()
    settlement_seconds = perf_counter() - settlement_started
    return {
        "case": name,
        "operand_shape": operand_shape,
        "operand_sets": resolution,
        "program_preparation_seconds": preparation_seconds,
        "compilation": "existing meshcore AOT library; no runtime machine-code compilation",
        "logical_retained_program_bytes": sum(
            bank.nbytes
            for bank in (
                program.instructions,
                program.indices,
                program.parameters,
                program.template_lower,
                program.template_upper,
            )
        ),
        "instruction_bytes": program.instructions.nbytes,
        "program_cache_policy": "shape-only keys; no source operands retained",
        "unscoped_timings": unscoped,
        "reference_timings": reference_timings,
        "native_scope_timings": scoped,
        "native_scope_preparation_seconds": scope_preparation_seconds,
        "native_scope_settlement_seconds": settlement_seconds,
        "native_scope_peak_bytes": scope.completed_evidence().memory_counters.peak_bytes,
        "native_scope_peak_measurement": "actual active NativeExecutionScope peak_bytes",
    }


def benchmark_homotopy_arithmetic(resolution: int, repeats: int) -> dict[str, object]:
    """Measure native homotopy arithmetic leaves, not embedding qualification."""
    from phydrax import _meshcore
    from phydrax.geometry import _mapped_embedding as mapped

    if resolution < 1 or repeats < 1:
        raise ValueError("resolution and repeats must be positive")
    if not _meshcore.meshcore_provider_installed():
        raise ModuleNotFoundError(
            "homotopy arithmetic requires the installed meshcore provider"
        )
    started = perf_counter()
    linear_sets, spd_sets, net_sets = _homotopy_operand_sets(resolution)
    input_preparation_seconds = perf_counter() - started
    mapped._homotopy_product_program.cache_clear()
    mapped._homotopy_spd_program.cache_clear()
    mapped._homotopy_net_program.cache_clear()

    def run_linear(reference: bool) -> int:
        operation = _homotopy_reference_product if reference else mapped._homotopy_linear
        for left, right in linear_sets:
            operation(left, right)
        return resolution

    def run_spd(reference: bool) -> int:
        operation = _homotopy_reference_spd if reference else mapped._homotopy_spd
        for jacobian, projection, transform in spd_sets:
            operation(jacobian, projection, transform)
        return resolution

    def run_net(reference: bool) -> int:
        operation = _homotopy_reference_net if reference else mapped._homotopy_net_range
        for direction, net in net_sets:
            operation(direction, net)
        return resolution

    records = [
        _measure_homotopy_program(
            "matrix-2x3x2",
            "2x3 times 3x2",
            partial(mapped._homotopy_product_program, 2, 3, 2),
            partial(run_linear, False),
            partial(run_linear, True),
            resolution,
            repeats,
        ),
        _measure_homotopy_program(
            "strict-SPD",
            "projected 2x2 strict-SPD",
            mapped._homotopy_spd_program,
            partial(run_spd, False),
            partial(run_spd, True),
            resolution,
            repeats,
        ),
        _measure_homotopy_program(
            "quadratic-net-projection",
            "six-control net",
            partial(mapped._homotopy_net_program, 6),
            partial(run_net, False),
            partial(run_net, True),
            resolution,
            repeats,
        ),
    ]
    for left, right in linear_sets:
        actual = mapped._homotopy_linear(left, right)
        _require_homotopy_endpoint_bytes(actual, _homotopy_reference_product(left, right))
        _require_homotopy_endpoint_bytes(actual, _homotopy_ordered_product(left, right))
    for jacobian, projection, transform in spd_sets:
        expected = _homotopy_reference_spd(jacobian, projection, transform)
        if (
            not expected
            or mapped._homotopy_spd(jacobian, projection, transform) is not expected
        ):
            raise AssertionError(
                "Strict-SPD decision differs from the positive interval reference"
            )
    for direction, net in net_sets:
        actual = np.asarray(mapped._homotopy_net_range(direction, net), dtype=np.float64)
        expected_range = np.asarray(
            _homotopy_reference_net(direction, net), dtype=np.float64
        )
        if actual.tobytes() != expected_range.tobytes():
            raise AssertionError("Net projection endpoint bytes differ")
    return {
        "status": "completed",
        "scope": "homotopy-arithmetic-leaf-diagnostic-not-qualification",
        "resolution": resolution,
        "repeats": repeats,
        "resolution_semantics": "distinct fixed-size operand sets per matrix, SPD, and six-control net leaf",
        "input_preparation_seconds": input_preparation_seconds,
        "distinct_operand_sets_verified": True,
        "reference_validation": {
            "matrix_ordered_endpoint_bytes_match": True,
            "matrix_previous_vectorized_endpoint_bytes_match": True,
            "strict_spd_positive_matches_numpy_interval_owner": True,
            "net_projection_endpoint_bytes_match": True,
        },
        "records": records,
    }
