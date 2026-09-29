"""Numerical-interoperability benchmark campaign (one function per campaign row).

Rows are registered in `ROWS`; later phases append their row functions. Each
row separates host preparation, lowering and compilation, and warmed repeated
execution, and reports compiler and logical retained-byte evidence. A row case
whose correctness evidence fails (a refused duality check, an unaccepted or
unsuccessful solve, an invalid derivative) raises before any of its timings is
reported, so the campaign exits nonzero instead of timing a wrong answer.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)

import phydrax as phx
from phydrax.discretization import SimplicialLocationPolicy
from phydrax.discretization.fem import prepare_finite_element_field_reconstruction
from phydrax.solver.coupling._interface_quadrature import (
    prepare_boundary_panel_resampling,
)


jax.config.update("jax_enable_x64", True)

type RowFunction = Callable[[bool, int, int], list[dict[str, Any]]]


def _require_correct(case: str, /, **gates: bool) -> None:
    """Refuse a case whose correctness evidence failed before reporting its timings."""
    failed = sorted(name for name, passed in gates.items() if not passed)
    if failed:
        raise RuntimeError(
            f"Benchmark case {case!r} failed its correctness gates: {', '.join(failed)}."
        )


def _square_mesh(
    resolution: int, /, *, x_offset: float = 0.0
) -> phx.discretization.CellMesh:
    vertices = np.asarray(
        [
            (x_offset + i / resolution, j / resolution)
            for j in range(resolution + 1)
            for i in range(resolution + 1)
        ]
    )
    triangles = []
    for j in range(resolution):
        for i in range(resolution):
            corner = j * (resolution + 1) + i
            triangles.append((corner, corner + 1, corner + resolution + 2))
            triangles.append((corner, corner + resolution + 2, corner + resolution + 1))
    return phx.discretization.CellMesh(
        jnp.asarray(vertices),
        (
            phx.discretization.CellBlock(
                "cells", "triangle", jnp.asarray(np.asarray(triangles, np.int32))
            ),
        ),
    )


def _compiled_timing(
    function: Callable[[jax.Array], jax.Array],
    argument: jax.Array,
    /,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    jitted = jax.jit(function)
    compiled, compilation = measure_lower_and_compile(
        lambda: jitted.lower(argument), lambda lowered: lowered.compile()
    )
    _, execution = measure_repeated(
        lambda: compiled(argument), warmup=warmup, repeats=repeats
    )
    return {
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "warm_execution": execution.to_dict(),
        "compiler": _compiler_record(compiled),
    }


def _compiler_record(compiled: Any, /) -> dict[str, int | None]:
    compiler = compiler_evidence(
        compiled.cost_analysis(), compiled.memory_analysis(), source="xla"
    )
    return {
        "flops": compiler.flops,
        "bytes_accessed": compiler.bytes_accessed,
        "argument_bytes": compiler.argument_bytes,
        "temporary_bytes": compiler.temporary_bytes,
        "output_bytes": compiler.output_bytes,
        "generated_code_bytes": compiler.generated_code_bytes,
    }


def _phased_timing(
    function: Callable[[jax.Array], jax.Array],
    argument: jax.Array,
    /,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Lowering, compilation, first synchronized run, warm runs, compiler bytes."""
    jitted = jax.jit(function)
    compiled, compilation = measure_lower_and_compile(
        lambda: jitted.lower(argument), lambda lowered: lowered.compile()
    )
    _, first = measure_synchronized(lambda: compiled(argument))
    _, execution = measure_repeated(
        lambda: compiled(argument), warmup=warmup, repeats=repeats
    )
    return {
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "first_synchronized_seconds": first,
        "warm_execution": execution.to_dict(),
        "compiler": _compiler_record(compiled),
    }


def _prepared_query_case(
    degree: int, query_count: int, resolution: int, warmup: int, repeats: int, /
) -> dict[str, Any]:
    discretization = phx.discretization.FiniteElementPlan(
        _square_mesh(resolution),
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", degree)
        ),
    ).prepare()
    reconstruction, reconstruction_seconds = measure_host(
        lambda: prepare_finite_element_field_reconstruction(
            discretization, "u", location_policy=SimplicialLocationPolicy(64, 16, 1)
        )
    )
    rng = np.random.default_rng(degree * 1000 + query_count)
    points = rng.uniform(0.05, 0.95, size=(query_count, 2))
    query, query_seconds = measure_host(lambda: reconstruction.prepare_query(points))
    nodes = np.asarray(discretization.dof_maps[0].dof_coordinates)
    state = jnp.asarray(np.sin(3.0 * nodes[:, 0]) * np.cos(2.0 * nodes[:, 1]))
    cotangent = jnp.asarray(rng.normal(size=query.output_shape))
    query_points = jnp.asarray(points)
    cold = _compiled_timing(
        lambda coefficients: reconstruction.evaluate(coefficients, query_points).values,
        state,
        warmup=warmup,
        repeats=repeats,
    )
    reused = _compiled_timing(
        lambda coefficients: query.apply(coefficients),
        state,
        warmup=warmup,
        repeats=repeats,
    )
    transpose = _compiled_timing(
        lambda values: query.transpose(values), cotangent, warmup=warmup, repeats=repeats
    )
    duality = query.duality_evidence(state, cotangent)
    _require_correct("prepared-query", duality_valid=bool(duality.valid))
    return {
        "degree": degree,
        "query_count": query_count,
        "resolution": resolution,
        "coefficients": int(reconstruction.coefficient_shape[0]),
        "reconstruction_preparation_seconds": reconstruction_seconds,
        "query_preparation_seconds": query_seconds,
        "cold_locate_and_apply": cold,
        "prepared_apply": reused,
        "prepared_transpose": transpose,
        "logical_route_bytes": logical_array_bytes(query.route),
        "logical_reconstruction_bytes": logical_array_bytes(reconstruction),
        "duality_valid": bool(duality.valid),
        "duality_residual": float(duality.residual),
    }


def prepared_query_row(smoke: bool, warmup: int, repeats: int, /) -> list[dict[str, Any]]:
    """Row 1: prepared query cold versus reused apply/transpose.

    Scales the query count and the local polynomial degree on a fixed mesh.
    """
    degrees = (1, 2) if smoke else (1, 2, 3)
    counts = (8, 64) if smoke else (16, 256, 2048)
    resolution = 4 if smoke else 16
    return [
        _prepared_query_case(degree, count, resolution, warmup, repeats)
        for degree in degrees
        for count in counts
    ]


def _strong_form_space(
    method: str, resolution: int, /
) -> tuple[Any, jax.Array, Any, jax.Array]:
    """Reconstruction, smooth coefficients, a face trace, and its coefficients."""
    d = phx.discretization
    match method:
        case "finite-difference-bspline":
            grid = d.TensorGridPlan(
                (d.UniformAxisSpec(resolution), d.UniformAxisSpec(resolution)),
                axis_names=("x", "y"),
            ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
            requests = tuple(
                d.DerivativeRequest(
                    f"d{name}", grid, name, derivative_order=1, accuracy_order=4
                )
                for name in ("x", "y")
            )
            discretization = d.FiniteDifferencePlan(
                grid, requests, field_name="u"
            ).prepare()
            reconstruction = d.prepare_finite_difference_field_reconstruction(
                discretization, interpolation=d.BSplineGridInterpolation(3)
            )
            axes = tuple(
                np.asarray(values)
                for values in grid.primary_entity_layout.coordinates_by_axis
            )
            x, y = np.meshgrid(*axes, indexing="ij")
            state = jnp.asarray(np.sin(3.0 * x) * np.cos(2.0 * y))
            norm = d.SBPGridNorm(
                tuple(
                    d.SBPDerivativePlan(grid, name, interior_order=4).prepare()
                    for name in ("x", "y")
                )
            )
            trace = discretization.prepare_side_trace(
                "u", discretization.exterior_facet_domain, norm=norm
            )
            return reconstruction, state, trace, state.reshape(-1)
        case "spectral-chebyshev-legendre":
            space = d.TensorSpectralPlan(
                (d.ChebyshevBasisPlan(resolution), d.LegendreBasisPlan(resolution)),
                axis_names=("x", "y"),
                field_name="u",
            ).prepare((d.AxisDomain.interval(0.0, 1.0), d.AxisDomain.interval(0.0, 1.0)))
            reconstruction = d.prepare_spectral_field_reconstruction(space)
            axes = tuple(np.asarray(axis.nodes) for axis in space.axes)
            x, y = np.meshgrid(*axes, indexing="ij")
            state = space.project(jnp.asarray(np.sin(3.0 * x) * np.cos(2.0 * y)))
            trace = space.prepare_side_trace(
                "u",
                space.exterior_facet_domain,
                rule=d.FacetTraceRule(points=resolution),
            )
            return reconstruction, state, trace, state
        case _:
            raise ValueError(f"Unknown strong-form benchmark method {method!r}.")


def _strong_form_query_case(
    method: str, resolution: int, query_count: int, warmup: int, repeats: int, /
) -> dict[str, Any]:
    (reconstruction, state, trace, trace_state), preparation_seconds = measure_host(
        lambda: _strong_form_space(method, resolution)
    )
    rng = np.random.default_rng(resolution * 1000 + query_count)
    points = rng.uniform(0.02, 0.98, size=(query_count, 2))
    query, query_seconds = measure_host(lambda: reconstruction.prepare_query(points))
    cotangent = jnp.asarray(rng.normal(size=query.output_shape))
    trace_cotangent = jnp.asarray(rng.normal(size=trace.output_shape))
    query_points = jnp.asarray(points)
    cold = _compiled_timing(
        lambda coefficients: reconstruction.evaluate(coefficients, query_points).values,
        state,
        warmup=warmup,
        repeats=repeats,
    )
    reused = _compiled_timing(
        lambda coefficients: query.apply(coefficients),
        state,
        warmup=warmup,
        repeats=repeats,
    )
    transpose = _compiled_timing(
        lambda values: query.transpose(values), cotangent, warmup=warmup, repeats=repeats
    )
    trace_apply = _compiled_timing(
        lambda coefficients: trace.apply(coefficients),
        trace_state,
        warmup=warmup,
        repeats=repeats,
    )
    trace_transpose = _compiled_timing(
        lambda values: trace.dual_pullback(values),
        trace_cotangent,
        warmup=warmup,
        repeats=repeats,
    )
    duality = query.duality_evidence(state, cotangent)
    _require_correct("strong-form-query", duality_valid=bool(duality.valid))
    return {
        "method": method,
        "resolution": resolution,
        "query_count": query_count,
        "coefficients": int(np.prod(reconstruction.coefficient_shape)),
        "trace_sites": int(np.prod(trace.output_shape)),
        "space_and_trace_preparation_seconds": preparation_seconds,
        "query_preparation_seconds": query_seconds,
        "cold_locate_and_apply": cold,
        "prepared_apply": reused,
        "prepared_transpose": transpose,
        "trace_apply": trace_apply,
        "trace_dual_pullback": trace_transpose,
        "logical_route_bytes": logical_array_bytes(query.route),
        "logical_trace_route_bytes": logical_array_bytes(trace.route),
        "duality_valid": bool(duality.valid),
        "duality_residual": float(duality.residual),
    }


def strong_form_query_row(
    smoke: bool, warmup: int, repeats: int, /
) -> list[dict[str, Any]]:
    """Row 2: FD/SBP and global spectral queries and face traces.

    Scales the grid or mode resolution and the query count for B-spline FD
    queries with SBP boundary traces and Chebyshev--Legendre synthesis with
    sum-factorized face traces.
    """
    methods = ("finite-difference-bspline", "spectral-chebyshev-legendre")
    resolutions = (16,) if smoke else (16, 64, 128)
    counts = (8, 64) if smoke else (16, 256, 2048)
    return [
        _strong_form_query_case(method, resolution, count, warmup, repeats)
        for method in methods
        for resolution in resolutions
        for count in counts
    ]


def _cube_tetrahedra(resolution: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Freudenthal tetrahedralization of the unit cube with positive orientation."""
    axis = np.linspace(0.0, 1.0, resolution + 1)
    vertices = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)

    def index(i: int, j: int, k: int) -> int:
        return (i * (resolution + 1) + j) * (resolution + 1) + k

    cells = []
    for i, j, k in itertools.product(range(resolution), repeat=3):
        for order in itertools.permutations(range(3)):
            corner = np.zeros(3, dtype=np.int64)
            path = [index(i, j, k)]
            for direction in order:
                corner[direction] += 1
                path.append(index(i + corner[0], j + corner[1], k + corner[2]))
            points = vertices[path]
            if np.linalg.det(points[1:] - points[0]) < 0.0:
                path[1], path[2] = path[2], path[1]
            cells.append(path)
    return vertices, np.asarray(cells, dtype=np.int32)


def _outward_boundary(
    vertices: np.ndarray, cells: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Compact outward boundary triangles of a tetrahedral mesh."""
    owners: dict[tuple[int, ...], list[tuple[tuple[int, int, int], int]]] = {}
    for cell in cells:
        for opposite in range(4):
            face = tuple(int(value) for value in np.delete(cell, opposite))
            owners.setdefault(tuple(sorted(face)), []).append(
                ((face[0], face[1], face[2]), int(cell[opposite]))
            )
    faces = []
    for entries in owners.values():
        if len(entries) != 1:
            continue
        (first, second, third), opposite = entries[0]
        normal = np.cross(
            vertices[second] - vertices[first], vertices[third] - vertices[first]
        )
        inward = normal @ (vertices[opposite] - vertices[first]) > 0.0
        faces.append((first, third, second) if inward else (first, second, third))
    faces_ = np.asarray(sorted(faces), dtype=np.int32)
    used, compact = np.unique(faces_, return_inverse=True)
    return vertices[used], compact.reshape(faces_.shape).astype(np.int32)


def _conductivity_rebind_case(
    resolution: int, warmup: int, repeats: int, /
) -> dict[str, Any]:
    vertices, cells = _cube_tetrahedra(resolution)
    surface_vertices, surface_faces = _outward_boundary(vertices, cells)
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        jnp.asarray(vertices), jnp.asarray(cells)
    )
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("tetrahedron", 1)
    )
    fem = phx.discretization.FiniteElementPlan(mesh, field).prepare(
        numeric_version=f"cube-{resolution}"
    )
    surface = phx.geometry.MeshRegion(
        jnp.asarray(surface_vertices),
        jnp.asarray(surface_faces),
        feature_id=f"cube-surface-{resolution}",
    )
    # Declared, bounded quadrature: the row measures rebinding, not BEM accuracy.
    quadrature = phx.operators.LaplaceSingleLayerDP0GalerkinPolicy3D(
        regular_order=3,
        singular_order=3,
        near_order=3,
        near_ratio=1.0,
        absolute_tolerance=2.0e-3,
        relative_tolerance=2.0e-3,
    )
    calderon, calderon_seconds = measure_host(
        lambda: phx.operators.prepare_scalar_calderon_dp0_3d(
            surface, policy=quadrature, numeric_version=f"cube-{resolution}"
        )
    )
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.FGMRES(restart=60, stagnation_iterations=60),
        tolerance=phx.linalg.TolerancePolicy(relative=1.0e-10, absolute=0.0),
        differentiation=phx.linalg.DifferentiationPolicy("mathematical"),
        failure=phx.linalg.FailurePolicy("status"),
    )
    prepared, preparation_seconds = measure_host(
        lambda: phx.solver.prepare_scalar_laplace_fem_bem_3d(
            fem,
            surface,
            calderon,
            interface_selection=phx.discretization.EntitySelection.from_subset(
                mesh.topology.entity_sets[2], "boundary"
            ),
            linear=policy,
        )
    )
    source = jnp.ones((vertices.shape[0],))
    conductivity = jnp.asarray(1.0 + 0.5 * np.sin(np.arange(cells.shape[0])))
    weights = jnp.asarray(np.cos(np.arange(vertices.shape[0])))

    def observation(kappa: jax.Array) -> jax.Array:
        return prepared.solve(source, conductivity=kappa).interior_coefficients @ weights

    result = prepared.solve(source, conductivity=conductivity)
    _require_correct(
        "conductivity-rebind",
        valid=bool(result.valid),
        derivative_valid=bool(result.derivative_valid),
    )
    return {
        "resolution": resolution,
        "cells": int(cells.shape[0]),
        "interface_faces": int(surface_faces.shape[0]),
        "calderon_preparation_seconds": calderon_seconds,
        "fem_bem_preparation_seconds": preparation_seconds,
        "rebound_primal": _compiled_timing(
            observation, conductivity, warmup=warmup, repeats=repeats
        ),
        "rebound_gradient": _compiled_timing(
            jax.grad(observation), conductivity, warmup=warmup, repeats=repeats
        ),
        "logical_prepared_bytes": logical_array_bytes(prepared),
        "valid": bool(result.valid),
        "derivative_valid": bool(result.derivative_valid),
        "linear_iterations": int(result.linear_result.diagnostics.iterations),
        "relative_block_residual": float(result.relative_block_residual),
    }


def conductivity_rebind_row(
    smoke: bool, warmup: int, repeats: int, /
) -> list[dict[str, Any]]:
    """Row 5: coefficient rebinding and adjoint solves of the scalar FEM-BEM owner.

    The conductivity is a runtime argument: the compiled primal and gradient
    rebind the prepared solve numerically, so host preparation is paid once.
    Scales the tetrahedral resolution and interface panel count.
    """
    resolutions = (1, 2) if smoke else (1, 2, 3)
    return [
        _conductivity_rebind_case(resolution, warmup, repeats)
        for resolution in resolutions
    ]


def _isogeometric_annulus(spans: int, /) -> tuple[Any, np.ndarray]:
    """Exact rational quarter annulus 1 <= r <= 2 with `spans` spans per axis."""
    iga = phx.discretization.iga
    knots = np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    half = np.sqrt(0.5)
    homogeneous = np.asarray(((1.0, 0.0, 1.0), (half, half, half), (0.0, 1.0, 1.0)))
    for value in np.arange(1, spans) / spans:
        span = np.searchsorted(knots, value, side="right") - 1
        refined = []
        for index in range(homogeneous.shape[0] + 1):
            if index <= span - 2:
                refined.append(homogeneous[index])
            elif index > span:
                refined.append(homogeneous[index - 1])
            else:
                alpha = (value - knots[index]) / (knots[index + 2] - knots[index])
                refined.append(
                    alpha * homogeneous[index] + (1.0 - alpha) * homogeneous[index - 1]
                )
        homogeneous = np.asarray(refined)
        knots = np.sort(np.append(knots, value))
    radial = iga.BSplineGrid.open_uniform(2, spans, interval=(0.0, 1.0))
    radii = 1.0 + np.asarray(radial.greville_abscissae)
    arc = homogeneous[:, :2] / homogeneous[:, 2:]
    controls = radii[:, None, None] * arc[None, :, :]
    plan = iga.IsogeometricPlan.isoparametric(
        (radial, iga.BSplineGrid(jnp.asarray(knots), 2)),
        iga.NURBSGeometryState(
            jnp.asarray(controls),
            jnp.asarray(np.ones(radii.shape)[:, None] * homogeneous[None, :, 2]),
        ),
        axis_names=("r", "theta"),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(3),
    )
    return plan.prepare(numeric_version=f"annulus-{spans}"), controls


def _isogeometric_query_case(
    spans: int, query_count: int, warmup: int, repeats: int, /
) -> dict[str, Any]:
    geometry = phx.geometry
    region = (
        (geometry.Ball((0.0, 0.0), 2.0) - geometry.Ball((0.0, 0.0), 1.0))
        & geometry.Orthotope((1.0, 1.0), (2.0, 2.0))
    ).compile()
    (prepared, controls), discretization_seconds = measure_host(
        lambda: _isogeometric_annulus(spans)
    )
    reconstruction, reconstruction_seconds = measure_host(
        lambda: phx.discretization.iga.prepare_isogeometric_field_reconstruction(
            prepared, "u", support_geometry=region
        )
    )
    rng = np.random.default_rng(spans * 1000 + query_count)
    radius = rng.uniform(1.0, 2.0, query_count)
    angle = rng.uniform(0.0, 0.5 * np.pi, query_count)
    points = np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=-1)
    query, query_seconds = measure_host(
        lambda: reconstruction.prepare_query(points, derivative=(1, 0))
    )
    trace, trace_seconds = measure_host(
        lambda: prepared.prepare_side_trace(
            "u",
            prepared.exterior_facet_domain,
            rule=phx.discretization.FacetTraceRule(points=4),
        )
    )
    state = jnp.asarray(np.sin(controls[..., 0]) * np.cos(controls[..., 1]))
    cotangent = jnp.asarray(rng.normal(size=query.output_shape))
    trace_cotangent = jnp.asarray(rng.normal(size=trace.output_shape))
    query_points = jnp.asarray(points)
    cold = _compiled_timing(
        lambda coefficients: (
            reconstruction.derivative(coefficients, query_points, (1, 0)).values
        ),
        state,
        warmup=warmup,
        repeats=repeats,
    )
    reused = _compiled_timing(
        lambda coefficients: query.apply(coefficients),
        state,
        warmup=warmup,
        repeats=repeats,
    )
    transpose = _compiled_timing(
        lambda values: query.transpose(values), cotangent, warmup=warmup, repeats=repeats
    )
    trace_apply = _compiled_timing(
        lambda coefficients: trace.apply(coefficients),
        state,
        warmup=warmup,
        repeats=repeats,
    )
    trace_transpose = _compiled_timing(
        lambda values: trace.dual_pullback(values),
        trace_cotangent,
        warmup=warmup,
        repeats=repeats,
    )
    duality = query.duality_evidence(state, cotangent)
    _require_correct("isogeometric-query", duality_valid=bool(duality.valid))
    return {
        "spans_per_axis": spans,
        "query_count": query_count,
        "coefficients": int(np.prod(reconstruction.coefficient_shape)),
        "trace_sites": int(np.prod(trace.output_shape)),
        "discretization_preparation_seconds": discretization_seconds,
        "reconstruction_preparation_seconds": reconstruction_seconds,
        "query_preparation_seconds": query_seconds,
        "trace_preparation_seconds": trace_seconds,
        "cold_inverse_map_and_gradient": cold,
        "prepared_gradient_apply": reused,
        "prepared_gradient_transpose": transpose,
        "trace_apply": trace_apply,
        "trace_dual_pullback": trace_transpose,
        "logical_route_bytes": logical_array_bytes(query.route),
        "logical_trace_route_bytes": logical_array_bytes(trace.route),
        "duality_valid": bool(duality.valid),
        "duality_residual": float(duality.residual),
    }


def isogeometric_query_row(
    smoke: bool, warmup: int, repeats: int, /
) -> list[dict[str, Any]]:
    """IGA row: inverse-map queries and patch-boundary traces.

    Scales the spans per axis of an exact rational quarter annulus and the
    query count; the cold path re-inverts the NURBS map on every call.
    """
    spans = (2,) if smoke else (2, 8, 16)
    counts = (8, 64) if smoke else (16, 256, 2048)
    return [
        _isogeometric_query_case(span, count, warmup, repeats)
        for span in spans
        for count in counts
    ]


def _finite_volume_face_trace_case(
    resolution: int, warmup: int, repeats: int, /
) -> dict[str, Any]:
    d = phx.discretization
    mesh = _square_mesh(resolution)
    discretization = d.UnstructuredFiniteVolumePlan(
        np.asarray(mesh.coordinates), triangles=np.asarray(mesh.blocks[0].vertices)
    ).prepare()
    name = discretization.cell_space.name
    rule = d.FacetTraceRule(points=2)
    domain = discretization.integration_domain("exterior_facet")
    centers = np.asarray(discretization.cell_centers)
    state = jnp.asarray(
        (np.sin(3.0 * centers[:, 0]) * np.cos(2.0 * centers[:, 1]))[:, None]
    )
    polynomial, polynomial_seconds = measure_host(
        lambda: d.CellPolynomialReconstructionPlan(1).prepare(discretization)
    )
    weno, weno_seconds = measure_host(
        lambda: d.UnstructuredWENOZReconstructionPlan(2).prepare(discretization)
    )
    average, average_seconds = measure_host(
        lambda: discretization.prepare_side_trace(name, domain, rule=rule)
    )
    linear, linear_seconds = measure_host(
        lambda: discretization.prepare_side_trace(
            name, domain, rule=rule, reconstruction=polynomial
        )
    )
    nonlinear, nonlinear_seconds = measure_host(
        lambda: discretization.prepare_nonlinear_face_trace(
            name, domain, rule=rule, reconstruction=weno
        )
    )
    rng = np.random.default_rng(resolution)
    cotangent = jnp.asarray(rng.normal(size=linear.output_shape))
    tangent = jnp.asarray(rng.normal(size=state.shape))
    covector = rng.normal(size=linear.output_shape)
    duality = abs(
        float(jnp.vdot(linear.apply(state), covector))
        - float(jnp.vdot(state, linear.dual_pullback(jnp.asarray(covector))))
    )
    return {
        "resolution": resolution,
        "cells": discretization.cell_count,
        "selected_facets": int(domain.entity_indices.size),
        "sites": int(np.prod(linear.output_shape[:2])),
        "cell_polynomial_preparation_seconds": polynomial_seconds,
        "weno_preparation_seconds": weno_seconds,
        "cell_average_trace_preparation_seconds": average_seconds,
        "face_state_trace_preparation_seconds": linear_seconds,
        "weno_face_trace_preparation_seconds": nonlinear_seconds,
        "cell_average_apply": _compiled_timing(
            lambda values: average.apply(values),
            state,
            warmup=warmup,
            repeats=repeats,
        ),
        "face_state_apply": _compiled_timing(
            lambda values: linear.apply(values),
            state,
            warmup=warmup,
            repeats=repeats,
        ),
        "face_state_dual_pullback": _compiled_timing(
            lambda values: linear.dual_pullback(values),
            cotangent,
            warmup=warmup,
            repeats=repeats,
        ),
        "weno_face_state_apply": _compiled_timing(
            lambda values: nonlinear.apply(values),
            state,
            warmup=warmup,
            repeats=repeats,
        ),
        "weno_linearized_jvp": _compiled_timing(
            lambda direction: nonlinear.linearize(state).jvp(direction),
            tangent,
            warmup=warmup,
            repeats=repeats,
        ),
        "weno_every_face_reference": _compiled_timing(
            lambda values: weno.reconstruct(values)[0],
            state,
            warmup=warmup,
            repeats=repeats,
        ),
        "logical_face_state_route_bytes": logical_array_bytes(linear.route),
        "logical_cell_average_route_bytes": logical_array_bytes(average.route),
        "face_state_duality_residual": duality,
    }


def finite_volume_face_trace_row(
    smoke: bool, warmup: int, repeats: int, /
) -> list[dict[str, Any]]:
    """FV row: boundary face traces of cell averages and reconstructions.

    Scales the triangulated square; the selected boundary facets grow like the
    square root of the cell count, and the per-facet routes with them. The
    reference evaluates the owner's WENO-Z reconstruction on every face.
    """
    resolutions = (4, 8) if smoke else (8, 16, 32)
    return [
        _finite_volume_face_trace_case(resolution, warmup, repeats)
        for resolution in resolutions
    ]


def _manufactured(points: jax.Array) -> jax.Array:
    return jnp.sin(0.5 * jnp.pi * points[..., 0]) * jnp.exp(points[..., 1])


def _manufactured_source(points: jax.Array, args: object) -> jax.Array:
    del args
    return (0.25 * jnp.pi**2 - 1.0) * _manufactured(points)


def _poisson_owner(
    resolution: int, degree: int, /, *, x_offset: float, interface: float | None
) -> tuple[Any, Any]:
    """Compiled P_k Poisson owner on a unit square and its facets on ``x = interface``.

    Every boundary DOF is Dirichlet except the interior of the interface
    segment, which the coupling law owns.
    """
    d = phx.discretization
    space = d.FiniteElementPlan(
        _square_mesh(resolution, x_offset=x_offset),
        d.FiniteElementFieldSpec("u", d.lagrange_element("triangle", degree)),
    ).prepare()
    coordinates = np.asarray(space.dof_maps[0].dof_coordinates)
    dirichlet = np.array(space.dof_maps[0].boundary_dof_mask, dtype=np.bool_)
    domain = None
    if interface is not None:
        exterior = space.exterior_facet_domain
        probe = space.prepare_side_trace("u", exterior, rule=d.FacetTraceRule(points=2))
        facets = np.all(np.isclose(np.asarray(probe.sites)[..., 0], interface), axis=1)
        edges = space.mesh.topology.entity_sets[1]
        mask = np.zeros((edges.count,), dtype=np.bool_)
        mask[np.asarray(exterior.entity_indices)[facets]] = True
        domain = space.integration_domain(
            "exterior_facet", d.EntitySelection(edges, mask)
        )
        dirichlet &= ~(
            np.isclose(coordinates[:, 0], interface)
            & (coordinates[:, 1] > 1.0e-12)
            & (coordinates[:, 1] < 1.0 - 1.0e-12)
        )
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "poisson",
            "u",
            (
                phx.equations.DiffusionAction("u"),
                phx.equations.SourceAction(
                    "u",
                    phx.equations.coefficient(_manufactured_source, coefficient_id="f"),
                ),
            ),
        ),
        space,
        constraint=d.dirichlet_constraint(space, "u", boundary_mask=dirichlet),
        dirichlet_values=_manufactured,
    )
    return problem, domain


def _cut_binding(
    minus_field: str, plus_field: str, /
) -> tuple[phx.solver.coupling.InterfaceBinding, phx.domain.SubdomainCover]:
    """Two-sided paired-support binding of the cut ``x = 1`` of ``[0, 2] x [0, 1]``."""
    c = phx.solver.coupling
    plate = phx.domain.HyperRectangle(np.zeros(2), np.asarray([2.0, 1.0]))
    cover = phx.domain.cartesian_subdomain_cover(plate, "x", (2, 1), cover_id="plate")
    pairing = cover.pairings[0]
    witness = pairing.component.sample(phx.domain.PointSampling(8), key=jax.random.key(1))
    binding = c.InterfaceBinding(
        "cut",
        c.InterfaceSource.paired_support(cover, pairing.pairing_id),
        "two-sided",
        tuple(
            c.InterfaceEndpoint(
                role,
                c.PairedSupportAttachment(cover, pairing.pairing_id, patch, witness),
                fields={"value": field},
            )
            for role, patch, field in (
                ("minus", pairing.left_patch_id, minus_field),
                ("plus", pairing.right_patch_id, plus_field),
            )
        ),
    )
    return binding, cover


def _mortar_constraint(law: Any, component: str, /) -> Any:
    """The mortar constraint block ``B_s`` reading the component's full field."""
    for contribution in law.contributions:
        if (
            isinstance(contribution, phx.solver.coupling.LinearContribution)
            and contribution.target.block == "constraint"
            and contribution.source.owner == component
        ):
            return contribution.operator
    raise ValueError(f"The mortar publishes no constraint block of {component!r}.")


def _interface_coupling_case(
    minus: int, plus: int, degree: int, warmup: int, repeats: int, /
) -> dict[str, Any]:
    c = phx.solver.coupling
    (left, left_domain), left_seconds = measure_host(
        lambda: _poisson_owner(minus, degree, x_offset=0.0, interface=1.0)
    )
    (right, right_domain), right_seconds = measure_host(
        lambda: _poisson_owner(plus, degree, x_offset=1.0, interface=1.0)
    )
    components = {
        "minus": c.VariationalComponent("minus", left, field="u"),
        "plus": c.VariationalComponent("plus", right, field="u"),
    }
    binding, cover = _cut_binding(
        components["minus"].field_space_id("u"), components["plus"].field_space_id("u")
    )
    sides = (
        c.TransmissionSide("minus", "minus", "u", left_domain),
        c.TransmissionSide("plus", "plus", "u", right_domain),
    )
    mortar, mortar_seconds = measure_host(
        lambda: c.ScalarTransmissionLaw(
            "mortar",
            binding,
            sides,
            c.MortarImposition(c.MortarMultiplier("side-trace", side="plus")),
        ).prepare(components, (cover,))
    )
    matching_seconds = None
    if minus == plus:
        _, matching_seconds = measure_host(
            lambda: c.ScalarTransmissionLaw(
                "matching", binding, sides, c.MatchingElimination(eliminated="plus")
            ).prepare(components, (cover,))
        )
    nitsche, nitsche_seconds = measure_host(
        lambda: c.ScalarTransmissionLaw(
            "nitsche", binding, sides, c.NitscheImposition(penalty_factor=2.0)
        ).prepare(components, (cover,))
    )
    evidence = mortar.evidence
    nitsche_evidence = nitsche.evidence
    if not isinstance(evidence, c.MortarEvidence) or not isinstance(
        nitsche_evidence, c.NitscheEvidence
    ):
        raise TypeError("The interface laws publish no mortar or Nitsche evidence.")
    rng = np.random.default_rng(1000 * minus + plus)
    multiplier = jnp.asarray(rng.normal(size=(evidence.multiplier_dimension,)))
    record: dict[str, Any] = {
        "degree": degree,
        "minus_resolution": minus,
        "plus_resolution": plus,
        "multiplier_dimension": evidence.multiplier_dimension,
        "numerical_rank": evidence.numerical_rank,
        "inf_sup": evidence.inf_sup,
        "common_segments": evidence.coverage.segment_count,
        "quadrature_exact_degree": evidence.quadrature_exact_degree,
        "minus_owner_preparation_seconds": left_seconds,
        "plus_owner_preparation_seconds": right_seconds,
        "mortar_preparation_seconds": mortar_seconds,
        "matching_preparation_seconds": matching_seconds,
        "nitsche_preparation_seconds": nitsche_seconds,
        "nitsche_penalty_range": list(nitsche_evidence.penalty_range),
        "logical_mortar_bytes": logical_array_bytes(mortar),
        "logical_nitsche_bytes": logical_array_bytes(nitsche),
    }
    nitsche_rows = nitsche.contributions[0]
    if not isinstance(nitsche_rows, c.ResidualContribution):
        raise TypeError("The Nitsche law publishes no residual contribution.")
    plus_state = jnp.asarray(rng.normal(size=(right.full_space.size,)))
    record["nitsche_row_action"] = _compiled_timing(
        lambda value: nitsche_rows.residual.evaluate((value, plus_state), None)[0],
        jnp.asarray(rng.normal(size=(left.full_space.size,))),
        warmup=warmup,
        repeats=repeats,
    )
    for role, owner in (("minus", left), ("plus", right)):
        operator = _mortar_constraint(mortar, role)
        state = jnp.asarray(rng.normal(size=(owner.full_space.size,)))
        forward = operator.mv(state)
        backward = operator.transpose_mv(multiplier)
        record[f"{role}_full_coefficients"] = owner.full_space.size
        record[f"{role}_constraint_action"] = _compiled_timing(
            lambda value: operator.mv(value), state, warmup=warmup, repeats=repeats
        )
        record[f"{role}_transpose_action"] = _compiled_timing(
            lambda value: operator.transpose_mv(value),
            multiplier,
            warmup=warmup,
            repeats=repeats,
        )
        record[f"{role}_duality_residual"] = float(
            jnp.abs(forward @ multiplier - state @ backward)
            / (jnp.linalg.norm(forward) * jnp.linalg.norm(multiplier))
        )
    return record


def interface_coupling_row(
    smoke: bool, warmup: int, repeats: int, /
) -> list[dict[str, Any]]:
    """Interface-coupling row: mortar/matching/Nitsche preparation and actions.

    Scales the interface resolution with matching and 2:3 nonmatching facet
    partitions and varies the two sides' resolutions independently. Matching
    elimination is prepared only where the partitions coincide; Nitsche
    preparation includes both owners' trace-inverse certification.
    """
    degrees = (1,) if smoke else (1, 2)
    pairs = (
        ((4, 4), (4, 6))
        if smoke
        else ((8, 8), (8, 12), (16, 16), (16, 24), (32, 32), (32, 48), (8, 32), (32, 8))
    )
    return [
        _interface_coupling_case(minus, plus, degree, warmup, repeats)
        for degree in degrees
        for minus, plus in pairs
    ]


def _coupled_assembly_case(
    resolution: int, degree: int, warmup: int, repeats: int, /
) -> dict[str, Any]:
    """Generic single-component coupled path versus the owner's native system."""
    c = phx.solver.coupling
    (problem, _), owner_seconds = measure_host(
        lambda: _poisson_owner(resolution, degree, x_offset=0.0, interface=None)
    )
    plan = c.CoupledProblemPlan(
        "single-owner",
        components=(c.VariationalComponent("square", problem, field="u"),),
        bindings=(),
        laws=(),
    )
    prepared, preparation_seconds = measure_host(lambda: c.prepare_coupled_problem(plan))
    (system, rhs), system_seconds = measure_synchronized(prepared.linear_system)
    (native, native_rhs), native_system_seconds = measure_synchronized(
        problem.linear_system
    )
    size = problem.state_space.size
    state = jnp.asarray(np.random.default_rng(resolution).normal(size=(size,)))

    def coupled_action(value: jax.Array) -> jax.Array:
        return system.operator.mv(((value,),))[0][0]

    reference_action = native.operator.mv(state)
    discrepancy = jnp.linalg.norm(
        coupled_action(state) - reference_action
    ) / jnp.linalg.norm(reference_action)
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        materialization=phx.linalg.MaterializationPolicy(
            max_entries=4 * size * size, max_bytes=32 * size * size
        ),
    )
    generic, generic_solve = measure_repeated(
        lambda: phx.linalg.solve(system, rhs, policy=policy),
        warmup=warmup,
        repeats=repeats,
    )
    reference, native_solve = measure_repeated(
        lambda: phx.linalg.solve(native, native_rhs, policy=policy),
        warmup=warmup,
        repeats=repeats,
    )
    certified, certified_solve = measure_repeated(
        lambda: c.solve_coupled_problem(prepared, policy=policy),
        warmup=warmup,
        repeats=repeats,
    )
    _require_correct(
        "coupled-assembly",
        coupled_successful=bool(generic.successful),
        native_successful=bool(reference.successful),
        coupled_accepted=bool(certified.accepted),
    )
    return {
        "degree": degree,
        "resolution": resolution,
        "unknowns": size,
        "dtype": str(state.dtype),
        "owner_compilation_seconds": owner_seconds,
        "coupled_preparation_seconds": preparation_seconds,
        "coupled_linear_system_seconds": system_seconds,
        "native_linear_system_seconds": native_system_seconds,
        "coupled_action": _compiled_timing(
            coupled_action, state, warmup=warmup, repeats=repeats
        ),
        "native_action": _compiled_timing(
            lambda value: native.operator.mv(value), state, warmup=warmup, repeats=repeats
        ),
        "relative_action_discrepancy": float(discrepancy),
        "coupled_solve": generic_solve.to_dict(),
        "native_solve": native_solve.to_dict(),
        "certified_coupled_solve": certified_solve.to_dict(),
        "solution_discrepancy": float(
            jnp.max(jnp.abs(generic.value[0][0] - reference.value))
        ),
        "native_successful": bool(reference.successful),
        "coupled_accepted": bool(certified.accepted),
        "logical_prepared_bytes": logical_array_bytes(prepared),
    }


def coupled_assembly_row(
    smoke: bool, warmup: int, repeats: int, /
) -> list[dict[str, Any]]:
    """Coupled-assembly row: generic coupled assembly against the native owner.

    One Dirichlet Poisson owner is solved through the generic coupled path
    (preparation, block ``linear_system``, block operator action, certified
    solve) and through its own native ``linear_system`` with the same dtype
    and dense policy; scales the mesh resolution and element degree.
    """
    cases = ((4, 1), (8, 1)) if smoke else ((8, 1), (16, 1), (32, 1), (8, 2), (16, 2))
    return [
        _coupled_assembly_case(resolution, degree, warmup, repeats)
        for resolution, degree in cases
    ]


_HOLE, _INTERFACE, _OUTER = 0.5, 1.0, 1.5

type _TraceOwner = (
    phx.discretization.FiniteElementDiscretization
    | phx.discretization.VirtualElementDiscretization
)


def _square_band_grid(
    inner: float, outer: float, cells: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Counterclockwise quadrilaterals of ``inner <= |x|_inf <= outer``.

    ``cells`` quadrilaterals span ``outer - inner`` (``inner = 0`` meshes the
    full square); every square side is a union of whole cell edges.
    """
    spacing = (outer - inner) / cells
    count = round(2.0 * outer / spacing)
    axis = np.linspace(-outer, outer, count + 1)
    grid = np.stack(np.meshgrid(axis, axis, indexing="xy"), axis=-1).reshape(-1, 2)
    index = np.arange((count + 1) ** 2).reshape(count + 1, count + 1)
    quads = np.stack(
        (index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]), axis=-1
    ).reshape(-1, 4)
    quads = quads[np.max(np.abs(np.mean(grid[quads], axis=1)), axis=1) > inner]
    used, local = np.unique(quads, return_inverse=True)
    return grid[used], local.reshape(quads.shape).astype(np.int32)


def _on_square(points: np.ndarray, half_width: float, /) -> np.ndarray:
    return np.isclose(np.max(np.abs(points), axis=-1), half_width)


def _facets_on_square(
    space: _TraceOwner, half_width: float, /
) -> phx.discretization.IntegrationDomain:
    """The owner's exterior facets on the square ``|x|_inf = half_width``."""
    d = phx.discretization
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace("u", exterior, rule=d.FacetTraceRule(points=2))
    facets = np.all(_on_square(np.asarray(probe.sites), half_width), axis=1)
    edges = space.mesh.topology.entity_sets[1]
    mask = np.zeros((edges.count,), dtype=np.bool_)
    mask[np.asarray(exterior.entity_indices)[facets]] = True
    return space.integration_domain("exterior_facet", d.EntitySelection(edges, mask))


def _square_band_cover(
    bands: tuple[tuple[str, float, float], ...], cover_id: str, /
) -> phx.domain.SubdomainCover:
    """Analytic cover of concentric square bands; consecutive bands are paired."""
    domain = phx.domain

    def square(half_width: float, /) -> phx.domain.HyperRectangle:
        return domain.HyperRectangle(np.full(2, -half_width), np.full(2, half_width))

    def identity(region: phx.domain.Domain, /) -> dict[str, phx.domain.DomainFunction]:
        return {"x": region.Function("x")(lambda x: x)}

    def band(inner: float, outer: float, /) -> Callable[[jax.Array], jax.Array]:
        def support(x: jax.Array) -> jax.Array:
            size = jnp.max(jnp.abs(x))
            return ((size >= inner) & (size <= outer)).astype(jnp.float64)

        return support

    def normal(x: jax.Array) -> jax.Array:
        axis = jnp.argmax(jnp.abs(x))
        return jnp.where(jnp.arange(2) == axis, jnp.sign(x), 0.0)

    window = square(bands[-1][2])
    patches = tuple(
        domain.SubdomainPatch(
            window,
            window.component(),
            window.Function("x")(band(inner, outer)),
            identity(window),
            identity(window),
            patch_id=patch_id,
        )
        for patch_id, inner, outer in bands
    )
    pairings = tuple(
        domain.PairedSupport(
            square(first[2]).component({"x": domain.Boundary()}),
            identity(square(first[2])),
            identity(square(first[2])),
            pairing_id=f"{first[0]}|{second[0]}",
            left_patch_id=first[0],
            right_patch_id=second[0],
            normal=square(first[2]).Function("x")(normal),
        )
        for first, second in zip(bands[:-1], bands[1:], strict=True)
    )
    return domain.SubdomainCover(window, patches, pairings, cover_id=cover_id)


def _square_band_binding(
    cover: phx.domain.SubdomainCover,
    pairing_id: str,
    endpoints: tuple[tuple[str, dict[str, str]], tuple[str, dict[str, str]]],
    /,
) -> phx.solver.coupling.InterfaceBinding:
    """Two-sided binding ``(minus, plus)`` of one square: its normal points outward."""
    c = phx.solver.coupling
    pairing = cover.pairing(pairing_id)
    witness = pairing.component.sample(
        phx.domain.PointSampling(16), key=jax.random.key(3)
    )
    patches = (pairing.left_patch_id, pairing.right_patch_id)
    return c.InterfaceBinding(
        pairing_id,
        c.InterfaceSource.paired_support(cover, pairing_id),
        "two-sided",
        tuple(
            c.InterfaceEndpoint(
                role,
                c.PairedSupportAttachment(cover, pairing_id, patch, witness),
                fields=fields,
            )
            for (role, fields), patch in zip(endpoints, patches, strict=True)
        ),
    )


def _spectral_element_owner(
    name: str,
    points: np.ndarray,
    quads: np.ndarray,
    degree: int,
    /,
    *,
    reaction: bool = False,
) -> tuple[
    phx.solver.coupling.VariationalComponent,
    phx.discretization.FiniteElementDiscretization,
]:
    """GLL tensor Lagrange quadrilaterals of one degree, coupled on every boundary.

    ``reaction`` adds a unit mass term, which removes the constant kernel of a
    pure-Neumann owner coupled to a bounded exterior.
    """
    d = phx.discretization
    space = d.FiniteElementPlan(
        d.CellMesh(
            jnp.asarray(points),
            (d.CellBlock("cells", "quadrilateral", jnp.asarray(quads)),),
        ),
        d.FiniteElementFieldSpec("u", d.lagrange_element("quadrilateral", degree)),
    ).prepare()
    diffusion = phx.equations.DiffusionAction("u")
    actions = (
        (diffusion, phx.equations.MassAction("u", 1.0)) if reaction else (diffusion,)
    )
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm("laplace", "u", actions),
        space,
    )
    return phx.solver.coupling.VariationalComponent(name, problem, field="u"), space


def _harmonic_dipole(points: jax.Array) -> jax.Array:
    """``u = x / (x^2 + y^2)``: harmonic away from the origin, decaying."""
    return points[..., 0] / jnp.sum(points**2, axis=-1)


def _virtual_element_ring(
    cells: int, degree: int, /
) -> tuple[
    phx.solver.coupling.VariationalComponent, phx.discretization.IntegrationDomain
]:
    """Virtual elements on perturbed quadrilaterals of ``0.5 <= |x|_inf <= 1``.

    The hole carries the Dirichlet data of ``_harmonic_dipole``; the outer
    square of the ring is coupled.
    """
    d = phx.discretization
    points, quads = _square_band_grid(_HOLE, _INTERFACE, cells)
    size = np.max(np.abs(points), axis=1)
    interior = (size > _HOLE + 1.0e-9) & (size < _INTERFACE - 1.0e-9)
    spacing = (_INTERFACE - _HOLE) / cells
    jitter = np.random.default_rng(20260928).uniform(-0.2, 0.2, points.shape) * spacing
    points = points + np.where(interior[:, None], jitter, 0.0)
    mesh = d.CellMesh.from_polygons(
        jnp.asarray(points), tuple(np.asarray(cell) for cell in quads)
    )
    space = d.VirtualElementPlan(
        mesh, d.VirtualElementFieldSpec("u", d.conforming_h1_virtual_element(degree))
    ).prepare()
    connectivity = mesh.connectivity
    if not isinstance(connectivity, d.PolygonalConnectivity):
        raise TypeError("Virtual elements live on polygon meshes.")
    dofs = np.asarray(space.dof_map.default_dof_points)
    boundary = np.zeros((dofs.shape[0],), dtype=np.bool_)
    vertices = np.asarray(connectivity.boundary_vertices, dtype=np.bool_)
    boundary[: vertices.shape[0]] = vertices
    for edge in np.flatnonzero(np.asarray(connectivity.boundary_edges)):
        start = vertices.shape[0] + edge * (degree - 1)
        boundary[start : start + degree - 1] = True
    problem = phx.equations.compile_virtual_element_problem(
        phx.equations.VirtualElementForm(
            "laplace", "u", (phx.equations.DiffusionAction("u"),)
        ),
        space,
        constraint=d.virtual_element_dirichlet_constraint(
            space, "u", boundary_mask=boundary & _on_square(dofs, _HOLE)
        ),
        dirichlet_values=_harmonic_dipole,
    )
    return (
        phx.solver.coupling.VariationalComponent("inner", problem, field="u"),
        _facets_on_square(space, _INTERFACE),
    )


def _square_exterior_owner(
    panels_per_side: int, /
) -> phx.solver.coupling.GalerkinBoundaryComponent:
    """Galerkin boundary operator on the outer square with counterclockwise panels."""
    corners = np.asarray(
        [[-_OUTER, -_OUTER], [_OUTER, -_OUTER], [_OUTER, _OUTER], [-_OUTER, _OUTER]]
    )
    fractions = np.arange(panels_per_side)[:, None] / panels_per_side
    vertices = np.concatenate(
        [
            corners[side] + fractions * (corners[(side + 1) % 4] - corners[side])
            for side in range(4)
        ]
    )
    galerkin = phx.operators.prepare_scalar_laplace_galerkin_2d(
        phx.operators.ClosedPolygonalCurve2D(vertices, source_id="outer-square"),
        policy=phx.operators.ScalarLaplaceGalerkinPolicy2D(regular_order=8),
    )
    return phx.solver.coupling.GalerkinBoundaryComponent("exterior", galerkin)


def _boundary_integral_law(
    cover: phx.domain.SubdomainCover,
    pairing_id: str,
    volume: phx.solver.coupling.VariationalComponent,
    domain: phx.discretization.IntegrationDomain,
    exterior: phx.solver.coupling.GalerkinBoundaryComponent,
    /,
) -> tuple[
    phx.solver.coupling.InterfaceBinding,
    phx.solver.coupling.BoundaryIntegralTransmissionLaw,
]:
    """Binding and bordered Johnson--Nedelec law of a volume owner and the exterior."""
    c = phx.solver.coupling
    binding = _square_band_binding(
        cover,
        pairing_id,
        (
            ("volume", {"value": volume.field_space_id("u")}),
            ("exterior", {"conormal": exterior.field_space_id("conormal")}),
        ),
    )
    law = c.BoundaryIntegralTransmissionLaw(
        "boundary",
        binding,
        c.TransmissionSide("volume", volume.name, "u", domain),
        c.BoundaryIntegralSide("exterior", exterior.name),
    )
    return binding, law


def _boundary_integral_case(
    panels_per_facet: int, cells: int, degree: int, warmup: int, repeats: int, /
) -> dict[str, Any]:
    """SEM square owner and 2-D Galerkin exterior joined by the boundary-integral law."""
    c = phx.solver.coupling
    points, quads = _square_band_grid(0.0, _OUTER, cells)
    (volume, space), owner_seconds = measure_host(
        lambda: _spectral_element_owner("square", points, quads, degree, reaction=True)
    )
    domain = _facets_on_square(space, _OUTER)
    exterior, exterior_seconds = measure_host(
        lambda: _square_exterior_owner(2 * cells * panels_per_facet)
    )
    cover = _square_band_cover(
        (("square", 0.0, _OUTER), ("exterior", _OUTER, 3.0)), "full-square"
    )
    binding, law = _boundary_integral_law(
        cover, "square|exterior", volume, domain, exterior
    )
    plan = c.CoupledProblemPlan(
        "sem-bem-square", components=(volume, exterior), bindings=(binding,), laws=(law,)
    )
    prepared, preparation_seconds = measure_host(
        lambda: c.prepare_coupled_problem(plan, interface_owners=(cover,))
    )
    (prepared_law,) = prepared.laws
    evidence = prepared_law.evidence
    if not isinstance(evidence, c.BoundaryIntegralEvidence):
        raise TypeError("The boundary law publishes no BoundaryIntegralEvidence.")
    components = {component.name: component for component in prepared.components}
    _, cold_law_seconds = measure_synchronized(lambda: law.prepare(components, (cover,)))
    # The law's GLL side trace (degree + 1 points for degree >= 2), prepared
    # outside the cold projection-and-resampling measurement.
    trace = volume.prepare_side_trace(
        "u",
        domain,
        rule=phx.discretization.FacetTraceRule(
            "gauss-lobatto-legendre", points=degree + 1
        ),
    )
    galerkin = exterior.galerkin
    vertices = np.asarray(galerkin.curve.vertices)
    ends = np.asarray(galerkin.curve.panel_vertices)
    panels = (
        vertices[ends[:, 0]],
        vertices[ends[:, 1]] - vertices[ends[:, 0]],
        -np.asarray(galerkin.curve.normals),
    )

    def cold_projection() -> Any:
        projection = phx.operators.prepare_boundary_trace_projection_2d(
            galerkin.spaces, order=evidence.projection_order
        )
        return projection, prepare_boundary_panel_resampling(
            trace, panels, np.asarray(projection.sample_points), policy=law.quadrature
        )

    _, cold_projection_seconds = measure_synchronized(cold_projection)
    operator = prepared.weak_operator(None)
    state_space, row_space = prepared.state_space, prepared.row_space
    rng = np.random.default_rng(panels_per_facet)
    state = jnp.asarray(rng.normal(size=(state_space.size,)))
    rows = jnp.asarray(rng.normal(size=(row_space.size,)))

    def forward(coordinates: jax.Array) -> jax.Array:
        return row_space.flatten(operator.mv(state_space.unflatten(coordinates)))

    def transpose(coordinates: jax.Array) -> jax.Array:
        return state_space.flatten(
            operator.transpose_mv(row_space.unflatten(coordinates))
        )

    forward_value, transpose_value = forward(state), transpose(rows)
    return {
        "volume_degree": degree,
        "volume_cells_per_side": 2 * cells,
        "panels_per_facet": panels_per_facet,
        "panel_count": evidence.panel_count,
        "state_size": state_space.size,
        "row_size": row_space.size,
        "trace_degree": evidence.trace_degree,
        "projection_order": evidence.projection_order,
        "projection_exact_degree": evidence.projection_exact_degree,
        "coverage_maximum_gap": float(evidence.coverage.maximum_gap),
        "coverage_maximum_normal_defect": float(evidence.coverage.maximum_normal_defect),
        "owner_compilation_seconds": owner_seconds,
        "exterior_preparation_seconds": exterior_seconds,
        "coupled_preparation_seconds": preparation_seconds,
        "cold_law_preparation_seconds": cold_law_seconds,
        "cold_projection_resampling_seconds": cold_projection_seconds,
        "forward_action": _compiled_timing(
            forward, state, warmup=warmup, repeats=repeats
        ),
        "transpose_action": _compiled_timing(
            transpose, rows, warmup=warmup, repeats=repeats
        ),
        "duality_residual": float(
            jnp.abs(forward_value @ rows - state @ transpose_value)
            / (jnp.linalg.norm(forward_value) * jnp.linalg.norm(rows))
        ),
        "logical_law_bytes": logical_array_bytes(prepared_law),
        "logical_prepared_bytes": logical_array_bytes(prepared),
    }


def boundary_integral_law_row(
    smoke: bool, warmup: int, repeats: int, /
) -> list[dict[str, Any]]:
    """Boundary-integral row: SEM-BEM law preparation and weak ``A``/``A^T`` actions.

    A degree-4 GLL spectral-element square ``[-1.5, 1.5]^2`` (4 x 4 cells, no
    Dirichlet rows, unit reaction so the coupled operator has no kernel) couples to a 2-D Galerkin exterior on its boundary; the
    volume is fixed while the boundary panels per volume facet grow. Cold
    re-preparation of the law and of its trace projection plus panel
    resampling is recorded beside the prepared weak-operator actions.
    """
    refinements = (1, 2) if smoke else (1, 2, 4, 8)
    return [
        _boundary_integral_case(panels, 2, 4, warmup, repeats) for panels in refinements
    ]


def _sem_vem_bem_case(outer_cells: int, /) -> dict[str, Any]:
    """VEM inner ring, SEM outer ring, and BEM exterior in one certified solve."""
    c = phx.solver.coupling
    outer_degree, inner_cells, inner_degree = 2, 2 * outer_cells, 1
    (inner, inner_domain), inner_seconds = measure_host(
        lambda: _virtual_element_ring(inner_cells, inner_degree)
    )
    points, quads = _square_band_grid(_INTERFACE, _OUTER, outer_cells)
    (outer, space), outer_seconds = measure_host(
        lambda: _spectral_element_owner("outer", points, quads, outer_degree)
    )
    facets_per_side = round(2.0 * _OUTER * outer_cells / (_OUTER - _INTERFACE))
    exterior, exterior_seconds = measure_host(
        lambda: _square_exterior_owner(facets_per_side)
    )
    cover = _square_band_cover(
        (
            ("inner-ring", _HOLE, _INTERFACE),
            ("outer-ring", _INTERFACE, _OUTER),
            ("exterior", _OUTER, 3.0),
        ),
        "square-annulus",
    )
    internal = _square_band_binding(
        cover,
        "inner-ring|outer-ring",
        (
            ("inner", {"value": inner.field_space_id("u")}),
            ("outer", {"value": outer.field_space_id("u")}),
        ),
    )
    boundary, law = _boundary_integral_law(
        cover, "outer-ring|exterior", outer, _facets_on_square(space, _OUTER), exterior
    )
    plan = c.CoupledProblemPlan(
        "sem-vem-bem",
        components=(inner, outer, exterior),
        bindings=(internal, boundary),
        laws=(
            c.ScalarTransmissionLaw(
                "internal",
                internal,
                (
                    c.TransmissionSide("inner", "inner", "u", inner_domain),
                    c.TransmissionSide(
                        "outer", "outer", "u", _facets_on_square(space, _INTERFACE)
                    ),
                ),
                c.MortarImposition(c.MortarMultiplier("side-trace", side="inner")),
            ),
            law,
        ),
    )
    prepared, preparation_seconds = measure_host(
        lambda: c.prepare_coupled_problem(plan, interface_owners=(cover,))
    )
    size = prepared.state_space.size
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        materialization=phx.linalg.MaterializationPolicy(
            max_entries=4 * size * size, max_bytes=32 * size * size
        ),
    )
    solution, solve_seconds = measure_synchronized(
        lambda: c.solve_coupled_problem(prepared, policy=policy)
    )
    gated_defects: dict[str, dict[str, float]] = {}
    for report in solution.interfaces:
        gated = np.asarray(report.gated, dtype=np.bool_)
        values = np.asarray(report.values)[gated]
        scales = np.asarray(report.scales)[gated]
        gated_defects[report.law_id] = {
            "maximum_value": float(np.max(values)),
            "maximum_scaled": float(np.max(values / scales)),
        }
    panel_count = None
    for prepared_law in prepared.laws:
        if isinstance(prepared_law.evidence, c.BoundaryIntegralEvidence):
            panel_count = prepared_law.evidence.panel_count
    _require_correct(
        "sem-vem-bem-flagship",
        native_successful=bool(solution.native_successful),
        accepted=bool(solution.accepted),
    )
    return {
        "outer_cells": outer_cells,
        "outer_degree": outer_degree,
        "inner_cells": inner_cells,
        "inner_degree": inner_degree,
        "panels_per_facet": 1,
        "panel_count": panel_count,
        "unknowns": size,
        "inner_preparation_seconds": inner_seconds,
        "outer_preparation_seconds": outer_seconds,
        "exterior_preparation_seconds": exterior_seconds,
        "coupled_preparation_seconds": preparation_seconds,
        "first_solve_seconds": solve_seconds,
        "native_successful": bool(solution.native_successful),
        "accepted": bool(solution.accepted),
        "tolerance": solution.tolerance,
        "gated_defects": gated_defects,
        "logical_prepared_bytes": logical_array_bytes(prepared),
    }


def sem_vem_bem_flagship_row(
    smoke: bool, warmup: int, repeats: int, /
) -> list[dict[str, Any]]:
    """Flagship row: VEM, SEM, and BEM owners in one mortar/boundary-law solve.

    The inner ring (degree-1 virtual elements, hole Dirichlet data), the
    outer ring (degree-2 GLL spectral elements), and the Galerkin exterior
    (one panel per spectral facet) are coupled by a side-trace mortar and the
    boundary-integral law; one dense-LU certified solve per h-level. Only the
    first synchronized solve is timed, so ``warmup``/``repeats`` are unused.
    """
    del warmup, repeats
    return [_sem_vem_bem_case(cells) for cells in ((1,) if smoke else (1, 2))]


# --- Coupled parameter bindings: FE-VEM heat conduction plate ----------------------------

_PLATE_SOURCE = 2.0
_PLATE_WATT = phx.units.derived_unit("W", ((phx.units.JOULE, 1), (phx.units.SECOND, -1)))
_PLATE_CONDUCTIVITY = phx.units.derived_unit(
    "W/(m K)", ((_PLATE_WATT, 1), (phx.units.METER, -1), (phx.units.KELVIN, -1))
)
_PLATE_HEAT_FLUX = phx.units.derived_unit(
    "W/m^2", ((_PLATE_WATT, 1), (phx.units.METER, -2))
)
_PLATE_SENSORS = np.asarray(
    [[0.3, 0.4, 0.0], [0.7, 0.6, 0.0], [1.4, 0.3, 0.0], [1.8, 0.7, 0.0]]
)
_PLATE_PARAMETERS = ("conductivity-left", "conductivity-right", "heat-flux")
_PLATE_TRUTH = (1.3, 0.7, 0.5)


def _plate_temperature(points: np.ndarray, theta: np.ndarray, /) -> np.ndarray:
    """Host analytic ``u(x)`` for ``theta = (kappa_left, kappa_right, heat_flux)``."""
    kappa_left, kappa_right, flux = (float(value) for value in theta)
    s = _PLATE_SOURCE
    x = np.asarray(points, dtype=np.float64)[..., 0]
    left = (-0.5 * s * x**2 + (2.0 * s + flux) * x) / kappa_left
    right = (-0.5 * s * (x - 1.0) ** 2 + (s + flux) * (x - 1.0)) / kappa_right + (
        1.5 * s + flux
    ) / kappa_left
    return np.where(x <= 1.0, left, right)


def _plate_bricks(bricks: int, /) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Distorted running-bond hexagons and quadrilaterals on ``[1, 2] x [0, 1]``."""
    columns = 2 * bricks + 1
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, 1.0, columns), np.linspace(0.0, 1.0, bricks + 1), indexing="xy"
    )
    line = np.arange(bricks + 1)[:, None]
    midpoint = (np.arange(columns) % 2 == 1)[None, :]
    interior = (line > 0) & (line < bricks)
    grid_y = grid_y + np.where(midpoint & interior, 0.15 * (-1.0) ** line / bricks, 0.0)
    mapped_y = grid_y + 0.06 * np.sin(2.0 * np.pi * grid_y) * (1.0 - 0.5 * grid_x)
    mapped_x = grid_x + 0.05 * np.sin(np.pi * grid_x) * np.sin(np.pi * grid_y)
    points = np.stack((1.0 + mapped_x, mapped_y), axis=-1).reshape(-1, 2)
    cells: list[np.ndarray] = []
    for row in range(bricks):
        bounds = (
            [0, *range(1, columns - 1, 2), columns - 1]
            if row % 2
            else list(range(0, columns, 2))
        )
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
            lower = [row * columns + column for column in range(start, stop + 1)]
            upper = [
                (row + 1) * columns + column for column in range(stop, start - 1, -1)
            ]
            cells.append(np.asarray(lower + upper, dtype=np.int32))
    return points, tuple(cells)


def _plate_facets(
    space: phx.discretization.FiniteElementDiscretization
    | phx.discretization.VirtualElementDiscretization,
    x: float,
    /,
) -> phx.discretization.IntegrationDomain:
    """The owner's exterior facets on the vertical line ``x``."""
    d = phx.discretization
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace("u", exterior, rule=d.FacetTraceRule(points=2))
    facets = np.all(np.isclose(np.asarray(probe.sites)[..., 0], x), axis=1)
    edges = space.mesh.topology.entity_sets[1]
    mask = np.zeros((edges.count,), dtype=np.bool_)
    mask[np.asarray(exterior.entity_indices)[facets]] = True
    return space.integration_domain("exterior_facet", d.EntitySelection(edges, mask))


def _plate_input(arguments: object, name: str, /) -> jax.Array:
    if not isinstance(arguments, Mapping):
        raise TypeError("Owner user arguments must be a mapping.")
    return jnp.asarray(arguments[name])


def _plate_source(points: jax.Array, args: object) -> jax.Array:
    del args
    return _PLATE_SOURCE * jnp.ones(points.shape[:-1])


def _plate_left_conductivity(points: jax.Array, context: object) -> jax.Array:
    if not isinstance(context, phx.equations.FiniteElementExecutionContext):
        raise TypeError("FE coefficients receive the execution context.")
    return _plate_input(context.user_args, "conductivity") * jnp.ones(points.shape[:-1])


def _plate_right_conductivity(points: jax.Array, args: object) -> jax.Array:
    return _plate_input(args, "conductivity") * jnp.ones(points.shape[:-1])


def _plate_heat_flux(points: jax.Array, args: object) -> jax.Array:
    return _plate_input(args, "heat-flux") * jnp.ones(points.shape[:-1])


def _plate_components(level: int, /) -> tuple[tuple[Any, Any], Any, Any, Any]:
    """P1 triangles on ``[0, 1]^2`` and k=1 virtual elements on ``[1, 2] x [0, 1]``."""
    c, d, e = phx.solver.coupling, phx.discretization, phx.equations
    fe_space = d.FiniteElementPlan(
        _square_mesh(4 * 2**level),
        d.FiniteElementFieldSpec("u", d.lagrange_element("triangle", 1)),
    ).prepare()
    nodes = np.asarray(fe_space.dof_maps[0].dof_coordinates)
    source = e.SourceAction("u", e.coefficient(_plate_source, coefficient_id="source"))
    fe = e.compile_finite_element_problem(
        e.FiniteElementForm(
            "left-conduction",
            "u",
            (
                e.DiffusionAction(
                    "u",
                    e.coefficient(
                        _plate_left_conductivity, coefficient_id="conductivity-left"
                    ),
                ),
                source,
            ),
        ),
        fe_space,
        constraint=d.dirichlet_constraint(
            fe_space, "u", boundary_mask=np.isclose(nodes[:, 0], 0.0)
        ),
        dirichlet_values=0.0,
    )
    points, cells = _plate_bricks(3 * 2**level)
    vem_space = d.VirtualElementPlan(
        d.CellMesh.from_polygons(jnp.asarray(points), cells),
        d.VirtualElementFieldSpec("u", d.conforming_h1_virtual_element(1)),
    ).prepare()
    vem = e.compile_virtual_element_problem(
        e.VirtualElementForm(
            "right-conduction",
            "u",
            (
                e.DiffusionAction(
                    "u",
                    e.coefficient(
                        _plate_right_conductivity, coefficient_id="conductivity-right"
                    ),
                ),
                source,
                e.BoundaryLoadAction(
                    "u",
                    e.coefficient(_plate_heat_flux, coefficient_id="heat-flux"),
                    action_id="right-wall-heat-flux",
                    domain=_plate_facets(vem_space, 2.0),
                ),
            ),
        ),
        vem_space,
    )
    triangles = c.VariationalComponent("triangles", fe, field="u")
    polygons = c.VariationalComponent("polygons", vem, field="u")
    plate = phx.domain.HyperRectangle(np.zeros(2), np.asarray([2.0, 1.0]))
    cover = phx.domain.cartesian_subdomain_cover(plate, "x", (2, 1), cover_id="plate")
    pairing = cover.pairings[0]
    witness = pairing.component.sample(phx.domain.PointSampling(8), key=jax.random.key(1))
    binding = c.InterfaceBinding(
        "cut",
        c.InterfaceSource.paired_support(cover, pairing.pairing_id),
        "two-sided",
        tuple(
            c.InterfaceEndpoint(
                component.name,
                c.PairedSupportAttachment(cover, pairing.pairing_id, patch, witness),
                fields={"value": component.field_space_id("u")},
            )
            for component, patch in (
                (triangles, pairing.left_patch_id),
                (polygons, pairing.right_patch_id),
            )
        ),
    )
    law = c.ScalarTransmissionLaw(
        "transmission",
        binding,
        (
            c.TransmissionSide(
                "triangles", "triangles", "u", _plate_facets(fe_space, 1.0)
            ),
            c.TransmissionSide(
                "polygons", "polygons", "u", _plate_facets(vem_space, 1.0)
            ),
        ),
        c.MortarImposition(c.MortarMultiplier("side-trace", side="polygons")),
    )
    return (triangles, polygons), binding, law, cover


def _plate_port(port_id: str, component_id: str, unit: Any, /) -> phx.ValuePort:
    return phx.ValuePort(
        port_id,
        event_shape=(),
        component_ids=(component_id,),
        representation="scalar",
        dimensions=(unit.dimension,),
    )


def _plate_plan(level: int, /) -> tuple[phx.solver.coupling.CoupledProblemPlan, Any]:
    """The plate with bound conductivities and heat flux and four temperature sensors."""
    c, m = phx.solver.coupling, phx.measurement
    components, binding, law, cover = _plate_components(level)
    parameters = (
        c.ParameterBinding(
            "conductivity-left",
            _plate_port("thermal-conductivity-left", "kappa", _PLATE_CONDUCTIVITY),
            targets=(c.RuntimeInput("triangles", "conductivity"),),
            role="coefficient",
            derivative=phx.DerivativeSurface.PHYSICAL_PARAMETER,
        ),
        c.ParameterBinding(
            "conductivity-right",
            _plate_port("thermal-conductivity-right", "kappa", _PLATE_CONDUCTIVITY),
            targets=(c.RuntimeInput("polygons", "conductivity"),),
            role="coefficient",
            derivative=phx.DerivativeSurface.PHYSICAL_PARAMETER,
        ),
        c.ParameterBinding(
            "heat-flux",
            _plate_port("right-wall-heat-flux", "g", _PLATE_HEAT_FLUX),
            targets=(c.RuntimeInput("polygons", "heat-flux"),),
            role="boundary",
            derivative=phx.DerivativeSurface.SOLVER_ARGUMENT,
        ),
    )
    temperature = m.QuantitySpec(
        "benchmark", "temperature", "temperature", phx.units.KELVIN, "temperature"
    )
    contract = phx.SpatialCoordinateContract(phx.units.METER)
    point = m.SamplingSemantics(m.SpatialSamplingKind.POINT)
    observations = tuple(
        c.FieldPointObservation(
            f"{component}-sensors",
            component,
            "u",
            quantity=temperature,
            support=m.PointSampleSupport(sensors, names, contract),
            sampling=point,
            field_unit=phx.units.KELVIN,
        )
        for component, sensors, names in (
            ("triangles", _PLATE_SENSORS[:2], ("a", "b")),
            ("polygons", _PLATE_SENSORS[2:], ("c", "d")),
        )
    )
    plan = c.CoupledProblemPlan(
        "benchmark-plate",
        components=components,
        bindings=(binding,),
        laws=(law,),
        parameters=parameters,
        observations=observations,
    )
    return plan, cover


def _prepared_plate(level: int, /) -> tuple[Any, dict[str, Any]]:
    """Prepare the plate once; owner compilation and coupled preparation timed apart."""
    (plan, cover), owner_seconds = measure_host(lambda: _plate_plan(level))
    reference = {
        name: jnp.asarray(value)
        for name, value in zip(_PLATE_PARAMETERS, _PLATE_TRUTH, strict=True)
    }
    prepared, preparation_seconds = measure_host(
        lambda: phx.solver.coupling.prepare_coupled_problem(
            plan, interface_owners=(cover,), parameters=reference
        )
    )
    return prepared, {
        "level": level,
        "unknowns": prepared.state_space.size,
        "owner_compilation_seconds": owner_seconds,
        "coupled_preparation_seconds": preparation_seconds,
        "logical_prepared_bytes": logical_array_bytes(prepared),
    }


def _plate_solve(prepared: Any, /) -> Callable[[jax.Array], Any]:
    """Certified solve with ``theta`` bound; dense LU derivative solves reuse its factors."""
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        differentiation=phx.linalg.DifferentiationPolicy("mathematical"),
        derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(route="primal-factors"),
        materialization=phx.linalg.MaterializationPolicy(
            max_entries=16_000_000, max_bytes=256 * 1024 * 1024
        ),
    )

    def solve(theta: jax.Array) -> Any:
        return phx.solver.coupling.solve_coupled_problem(
            prepared,
            parameters={name: theta[i] for i, name in enumerate(_PLATE_PARAMETERS)},
            policy=policy,
        )

    return solve


def _plate_sensors(solution: Any, /) -> jax.Array:
    return jnp.concatenate(
        [
            solution.observation("triangles-sensors").values,
            solution.observation("polygons-sensors").values,
        ]
    )


def _coupled_derivative_case(level: int, warmup: int, repeats: int, /) -> dict[str, Any]:
    prepared, record = _prepared_plate(level)
    solve = _plate_solve(prepared)
    truth = np.asarray(_PLATE_TRUTH)
    data = jnp.asarray(_plate_temperature(_PLATE_SENSORS, truth))
    theta = jnp.asarray([1.0, 1.0, 0.3])
    direction = jnp.asarray([0.3, -0.2, 0.5])

    def predictions(values: jax.Array) -> jax.Array:
        return _plate_sensors(solve(values))

    def misfit(values: jax.Array) -> jax.Array:
        return 0.5 * jnp.sum((predictions(values) - data) ** 2)

    def tangent(values: jax.Array) -> jax.Array:
        return jax.jvp(misfit, (values,), (direction,))[1]

    # Time the compiled paths before any evidence evaluation fills their caches.
    timings = {
        "rebound_primal": _compiled_timing(
            predictions, theta, warmup=warmup, repeats=repeats
        ),
        "adjoint_gradient": _compiled_timing(
            jax.grad(misfit), theta, warmup=warmup, repeats=repeats
        ),
        "tangent": _compiled_timing(tangent, theta, warmup=warmup, repeats=repeats),
    }
    jitted_solve = jax.jit(solve)
    jitted_misfit = jax.jit(misfit)
    solution = jitted_solve(theta)
    gradient = np.asarray(jax.jit(jax.grad(misfit))(theta))
    step = 1.0e-6
    central = np.asarray(
        [
            (jitted_misfit(theta + step * unit) - jitted_misfit(theta - step * unit))
            / (2.0 * step)
            for unit in jnp.eye(3, dtype=theta.dtype)
        ]
    )
    directional = float(jax.jit(tangent)(theta))
    expected_directional = float(central @ np.asarray(direction))
    at_truth = np.asarray(_plate_sensors(jitted_solve(jnp.asarray(truth))))
    capability = solution.derivative_capability
    _require_correct(
        "coupled-derivative",
        native_successful=bool(solution.native_successful),
        accepted=bool(solution.accepted),
        derivative_valid=bool(solution.derivative_valid),
    )
    return {
        **record,
        "derivative_solve": "dense LU factors of the primal solve",
        **timings,
        "admitted": [name for name, _ in capability.admitted],
        "route": capability.derivative_contract.route.value,
        "native_successful": bool(solution.native_successful),
        "accepted": bool(solution.accepted),
        "derivative_valid": bool(solution.derivative_valid),
        "gradient_relative_error_vs_central": float(
            np.linalg.norm(gradient - central) / np.linalg.norm(central)
        ),
        "tangent_relative_error_vs_central": abs(directional - expected_directional)
        / abs(expected_directional),
        "max_sensor_error_at_truth": float(np.max(np.abs(at_truth - np.asarray(data)))),
    }


def coupled_derivative_row(
    smoke: bool, warmup: int, repeats: int, /
) -> list[dict[str, Any]]:
    """Row 5 (coupled): coefficient rebinding with primal, adjoint, and tangent solves.

    The FE-VEM plate binds two conductivities (physical parameters) and a
    boundary heat flux (solver argument). Host preparation is timed once; the
    jitted primal with new parameter values, ``jax.grad`` of a sensor misfit
    (adjoint), and ``jax.jvp`` (tangent) then rebind the prepared problem
    numerically. Dense LU derivative solves reuse the primal factors. The
    gradient and tangent are compared with host central differences.
    """
    levels = (0,) if smoke else (0, 1, 2)
    return [_coupled_derivative_case(level, warmup, repeats) for level in levels]


def _coupled_repeated_bind_case(
    level: int, warmup: int, repeats: int, /
) -> dict[str, Any]:
    prepared, record = _prepared_plate(level)
    solve = _plate_solve(prepared)
    traces = [0]

    def counted(theta: jax.Array) -> Any:
        traces[0] += 1
        return solve(theta)

    jitted = jax.jit(counted)
    values = (_PLATE_TRUTH, (1.0, 1.0, 0.3), (0.8, 1.5, 1.1), (2.0, 0.5, 0.0))
    thetas = tuple(jnp.asarray(value) for value in values)
    _, cold_seconds = measure_synchronized(lambda: jitted(thetas[0]))
    cycle = itertools.cycle(thetas)
    _, warm = measure_repeated(
        lambda: jitted(next(cycle)), warmup=warmup, repeats=repeats
    )
    solutions = tuple(jitted(theta) for theta in thetas)
    errors = [
        float(
            np.max(
                np.abs(
                    np.asarray(_plate_sensors(solution))
                    - _plate_temperature(_PLATE_SENSORS, np.asarray(value))
                )
            )
        )
        for solution, value in zip(solutions, values, strict=True)
    ]
    same_problem_id = all(
        solution.derivative_capability.owner_id == prepared.problem_id
        for solution in solutions
    )
    accepted = all(bool(solution.accepted) for solution in solutions)
    _require_correct(
        "coupled-repeated-bind", same_problem_id=same_problem_id, accepted=accepted
    )
    return {
        **record,
        "cold_first_call_seconds": cold_seconds,
        "warm_rebound_solve": warm.to_dict(),
        "distinct_parameter_values": len(values),
        "traces": traces[0],
        "same_problem_id": same_problem_id,
        "accepted": accepted,
        "max_sensor_error_per_value": errors,
    }


def coupled_repeated_bind_row(
    smoke: bool, warmup: int, repeats: int, /
) -> list[dict[str, Any]]:
    """Coupled repeated-bind row: many parameter values, one prepared plate.

    Compares the cold first call (trace, compile, solve) with warmed solves
    that cycle through distinct conductivity and heat-flux values in one
    compiled function. ``traces`` counts traces (one means no retrace), and
    every solution's derivative capability names the same ``problem_id``, so
    no value was re-prepared. Sensor errors are against the host analytic field.
    """
    levels = (0,) if smoke else (0, 1, 2)
    return [_coupled_repeated_bind_case(level, warmup, repeats) for level in levels]


# --- Bounded execution scaling (strip chains) -------------------------------------------------


def _harmonic(points: jax.Array) -> jax.Array:
    """``u = sin(x / 2) exp(y / 2)``: harmonic, so every strip is source free."""
    return jnp.sin(0.5 * points[..., 0]) * jnp.exp(0.5 * points[..., 1])


def _host_harmonic(points: np.ndarray, /) -> np.ndarray:
    return np.sin(0.5 * points[..., 0]) * np.exp(0.5 * points[..., 1])


def _no_source(points: jax.Array, args: object) -> jax.Array:
    del args
    return jnp.zeros(points.shape[:-1], dtype=points.dtype)


def _band_mesh(x0: float, width: int, cells: int, /) -> phx.discretization.CellMesh:
    """Right triangles on ``[x0, x0 + width] x [0, 1]``, ``cells`` per unit length."""
    columns = width * cells
    xs = np.linspace(x0, x0 + width, columns + 1)
    ys = np.linspace(0.0, 1.0, cells + 1)
    points = np.stack(np.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)
    triangles = [
        triangle
        for j in range(cells)
        for i in range(columns)
        for a in (j * (columns + 1) + i,)
        for triangle in (
            (a, a + 1, a + columns + 2),
            (a, a + columns + 2, a + columns + 1),
        )
    ]
    return phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "cells", "triangle", np.asarray(triangles, dtype=np.int32)
            ),
        ),
    )


def _laplace_band(
    x0: float, width: int, cells: int, degree: int, cuts: tuple[float, ...], /
) -> tuple[Any, Any]:
    """P_k Laplace owner on a band; interface rows of ``cuts`` stay free."""
    d = phx.discretization
    space = d.FiniteElementPlan(
        _band_mesh(x0, width, cells),
        d.FiniteElementFieldSpec("u", d.lagrange_element("triangle", degree)),
    ).prepare()
    points = np.asarray(space.dof_maps[0].dof_coordinates)
    dirichlet = np.array(space.dof_maps[0].boundary_dof_mask, dtype=np.bool_)
    for x in cuts:
        dirichlet &= ~(
            np.isclose(points[:, 0], x)
            & (points[:, 1] > 1.0e-12)
            & (points[:, 1] < 1.0 - 1.0e-12)
        )
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "laplace",
            "u",
            (
                phx.equations.DiffusionAction("u"),
                phx.equations.SourceAction(
                    "u", phx.equations.coefficient(_no_source, coefficient_id="f")
                ),
            ),
        ),
        space,
        constraint=d.dirichlet_constraint(space, "u", boundary_mask=dirichlet),
        dirichlet_values=_harmonic,
    )
    return problem, space


def _cut_domain(space: Any, x: float, /) -> phx.discretization.IntegrationDomain:
    d = phx.discretization
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace("u", exterior, rule=d.FacetTraceRule(points=2))
    on = np.all(np.isclose(np.asarray(probe.sites)[..., 0], x), axis=1)
    edges = space.mesh.topology.entity_sets[1]
    mask = np.zeros((edges.count,), dtype=np.bool_)
    mask[np.asarray(exterior.entity_indices)[on]] = True
    return space.integration_domain("exterior_facet", d.EntitySelection(edges, mask))


def _strip_chain(
    count: int,
    cells: int,
    degree: int,
    imposition: Any,
    /,
    *,
    observations: Callable[[tuple[Any, ...]], tuple[Any, ...]] = lambda _: (),
) -> tuple[Any, Any]:
    """``count`` unit strips joined on ``x = 1, ..., count - 1`` by one law each."""
    c = phx.solver.coupling
    plate = phx.domain.HyperRectangle(np.zeros(2), np.asarray([float(count), 1.0]))
    cover = phx.domain.cartesian_subdomain_cover(
        plate, "x", (count, 1), cover_id=f"strips-{count}"
    )
    strips: list[tuple[Any, dict[float, Any]]] = []
    for k in range(count):
        cuts = tuple(x for x in (float(k), float(k + 1)) if 0.0 < x < count)
        problem, space = _laplace_band(float(k), 1, cells, degree, cuts)
        component = c.VariationalComponent(f"strip{k:03d}", problem, field="u")
        strips.append((component, {x: _cut_domain(space, x) for x in cuts}))
    bindings, laws = [], []
    for k, pairing in enumerate(cover.pairings):
        witness = pairing.component.sample(
            phx.domain.PointSampling(8), key=jax.random.key(1)
        )
        (minus, minus_cuts), (plus, plus_cuts) = strips[k], strips[k + 1]
        binding = c.InterfaceBinding(
            f"cut{k:03d}",
            c.InterfaceSource.paired_support(cover, pairing.pairing_id),
            "two-sided",
            tuple(
                c.InterfaceEndpoint(
                    role,
                    c.PairedSupportAttachment(cover, pairing.pairing_id, patch, witness),
                    fields={"value": owner.field_space_id("u")},
                )
                for role, patch, owner in (
                    ("minus", pairing.left_patch_id, minus),
                    ("plus", pairing.right_patch_id, plus),
                )
            ),
        )
        bindings.append(binding)
        laws.append(
            c.ScalarTransmissionLaw(
                f"law{k:03d}",
                binding,
                (
                    c.TransmissionSide("minus", minus.name, "u", minus_cuts[k + 1.0]),
                    c.TransmissionSide("plus", plus.name, "u", plus_cuts[k + 1.0]),
                ),
                imposition,
            )
        )
    components = tuple(component for component, _ in strips)
    plan = c.CoupledProblemPlan(
        f"strip-chain-{count}",
        components=components,
        bindings=tuple(bindings),
        laws=tuple(laws),
        observations=observations(components),
    )
    return plan, cover


def _with_execution(plan: Any, execution: Any, /) -> Any:
    c = phx.solver.coupling
    return c.CoupledProblemPlan(
        plan.plan_id,
        components=plan.components,
        bindings=plan.bindings,
        laws=plan.laws,
        observations=plan.observations,
        resources=c.CoupledResourcePolicy(execution=execution),
    )


def _dense_policy(size: int, /) -> phx.linalg.LinearSolvePolicy:
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        materialization=phx.linalg.MaterializationPolicy(
            max_entries=4 * size * size, max_bytes=32 * size * size
        ),
    )


def _workset_record(workset: Any, /) -> dict[str, Any]:
    estimate = workset.estimate
    return {
        "kind": workset.kind,
        "members": len(workset.members),
        "signature_id": workset.signature_id,
        "bucket_count": estimate.bucket_count,
        "lane_capacity": estimate.lane_capacity,
        "padded_lanes": estimate.padded_lanes,
        "lane_data_bytes": estimate.lane_data_bytes,
        "lane_input_bytes": estimate.lane_input_bytes,
        "lane_output_bytes": estimate.lane_output_bytes,
        "traced_intermediate_bytes": estimate.traced_intermediate_bytes,
        "working_set_bytes": estimate.working_set_bytes,
    }


def _execution_record(
    prepared: Any, state: jax.Array, policy: Any, warmup: int, repeats: int, /
) -> tuple[dict[str, Any], Any]:
    """Compiled residual and operator action, certified solve, retained bytes."""
    space, rows = prepared.state_space, prepared.row_space
    system, _ = prepared.linear_system()
    solution, first_solve = measure_synchronized(
        lambda: phx.solver.coupling.solve_coupled_problem(prepared, policy=policy)
    )
    _, warm_solve = measure_repeated(
        lambda: phx.solver.coupling.solve_coupled_problem(prepared, policy=policy),
        warmup=warmup,
        repeats=repeats,
    )
    _require_correct("coupled-execution", accepted=bool(solution.accepted))
    return {
        "residual": _phased_timing(
            lambda value: rows.flatten(prepared.residual(space.unflatten(value))),
            state,
            warmup=warmup,
            repeats=repeats,
        ),
        "operator_action": _phased_timing(
            lambda value: space.flatten(system.operator.mv(space.unflatten(value))),
            state,
            warmup=warmup,
            repeats=repeats,
        ),
        "certified_solve_first_seconds": first_solve,
        "certified_solve_warm": warm_solve.to_dict(),
        "accepted": bool(solution.accepted),
        "logical_prepared_bytes": logical_array_bytes(prepared),
    }, solution


def _conforming_difference(
    count: int, cells: int, degree: int, solution: Any, prepared: Any, /
) -> tuple[dict[str, Any], float]:
    """Specialized product: one native owner on the union band, same data.

    On matching meshes the side-trace mortar enforces nodal continuity, so the
    coupled discrete solution is the conforming owner's solution.
    """
    (problem, _), owner_seconds = measure_host(
        lambda: _laplace_band(0.0, count, cells, degree, ())
    )
    (system, rhs), system_seconds = measure_synchronized(lambda: problem.linear_system())
    size = problem.state_space.size
    result, first_solve = measure_synchronized(
        lambda: phx.linalg.solve(system, rhs, policy=_dense_policy(size))
    )
    full = np.asarray(problem.expand(result.value, None))
    union = np.asarray(problem.discretization.dof_maps[0].dof_coordinates)
    index = {tuple(np.round(point, 10)): row for row, point in enumerate(union)}
    difference = 0.0
    for component in prepared.components:
        coordinates = np.asarray(
            component.problem.discretization.dof_maps[0].dof_coordinates
        )
        rows = [index[tuple(np.round(point, 10))] for point in coordinates]
        values = np.asarray(solution.field(component.name, "u"))
        difference = max(difference, float(np.max(np.abs(values - full[rows]))))
    _require_correct("conforming-owner", native_successful=bool(result.successful))
    return {
        "owner_compilation_seconds": owner_seconds,
        "native_linear_system_seconds": system_seconds,
        "native_solve_first_seconds": first_solve,
        "native_unknowns": size,
        "native_successful": bool(result.successful),
        "max_nodal_error_vs_exact": float(np.max(np.abs(full - _host_harmonic(union)))),
    }, difference


def _homogeneous_worksets_case(
    count: int, cells: int, degree: int, capacity: int, warmup: int, repeats: int, /
) -> dict[str, Any]:
    c = phx.solver.coupling
    mortar = c.MortarImposition(c.MortarMultiplier("side-trace", side="plus"))
    (plan, cover), owner_seconds = measure_host(
        lambda: _strip_chain(count, cells, degree, mortar)
    )
    reference, reference_seconds = measure_host(
        lambda: c.prepare_coupled_problem(plan, interface_owners=(cover,))
    )
    execution = c.CoupledExecutionPolicy(lane_capacity=capacity)
    lanes, lane_seconds = measure_host(
        lambda: c.prepare_coupled_problem(
            _with_execution(plan, execution), interface_owners=(cover,)
        )
    )
    size = reference.state_space.size
    state = jnp.asarray(np.random.default_rng(count).normal(size=(size,)))
    policy = _dense_policy(size)
    per_component, expected = _execution_record(reference, state, policy, warmup, repeats)
    lane_record, solution = _execution_record(lanes, state, policy, warmup, repeats)
    specialized, difference = _conforming_difference(
        count, cells, degree, solution, lanes
    )
    worksets = lanes.worksets
    if worksets is None:
        raise RuntimeError("A declared lane execution prepares worksets.")
    rows = reference.row_space
    return {
        "components": count,
        "cells_per_strip": cells,
        "degree": degree,
        "lane_capacity": capacity,
        "unknowns": size,
        "multiplier_unknowns": sum(
            block.space.size for law in reference.laws for block in law.state_blocks
        ),
        "owner_compilation_seconds": owner_seconds,
        "per_component_preparation_seconds": reference_seconds,
        "lane_preparation_seconds": lane_seconds,
        "worksets": [
            _workset_record(workset)
            for workset in (*worksets.components, *worksets.interfaces)
        ],
        "ungrouped_components": count - len(worksets.grouped_components),
        "lane_data_bytes": worksets.lane_data_bytes,
        "max_working_set_bytes": worksets.max_working_set_bytes,
        "per_component": per_component,
        "lanes": lane_record,
        "residual_difference": float(
            jnp.max(
                jnp.abs(
                    rows.flatten(
                        reference.residual(reference.state_space.unflatten(state))
                    )
                    - rows.flatten(lanes.residual(lanes.state_space.unflatten(state)))
                )
            )
        ),
        "solution_difference": float(
            jnp.max(
                jnp.abs(
                    reference.state_space.flatten(expected.state)
                    - lanes.state_space.flatten(solution.state)
                )
            )
        ),
        "specialized_conforming_owner": specialized,
        "difference_to_conforming_owner": difference,
    }


def homogeneous_worksets_row(
    smoke: bool,
    warmup: int,
    repeats: int,
    /,
    *,
    max_components: int | None = None,
) -> list[dict[str, Any]]:
    """Row 8/4: homogeneous lane worksets versus per-component execution.

    Chains of identical P_k Laplace strips coupled by side-trace mortars scale
    the component count, then the lane capacity (working-set bound), then the
    DOFs/order and multiplier size at fixed count. Each case measures host
    preparation of both executions, lowering/compilation/first/warm runs and
    compiler bytes of the residual and operator action, the certified solve,
    retained bytes, and the lane estimates, and compares the coupled solution
    with the specialized conforming owner on the union band (identical
    discrete problem on matching meshes).

    ``max_components`` caps the component-count axis (and the count of the
    lane-capacity axis) for bounded full-size runs; ``None`` runs 4 to 64.
    """
    if max_components is not None and max_components < 4:
        raise ValueError("max_components must be at least 4.")
    limit = 64 if max_components is None else max_components
    capacity_count = min(32, limit)
    cases = (
        ((4, 2, 1, 4), (8, 2, 1, 4))
        if smoke
        else (
            *((count, 4, 1, 8) for count in (4, 8, 16, 32, 64) if count <= limit),
            *((capacity_count, 4, 1, capacity) for capacity in (2, 32)),
            *(
                (8, cells, degree, 8)
                for cells, degree in ((2, 1), (8, 1), (4, 2), (8, 2))
            ),
        )
    )
    return [
        _homogeneous_worksets_case(count, cells, degree, capacity, warmup, repeats)
        for count, cells, degree, capacity in cases
    ]


_PARITY_DEVICES = 4
_PARITY_CHILD = "PHYDRAX_BENCHMARK_FORCED_HOST_DEVICES"


def _forced_host_environment(devices: int, /) -> dict[str, str]:
    """This environment on ``devices`` forced CPU host devices, other XLA flags kept."""
    prefix = "--xla_force_host_platform_device_count="
    flags = [
        flag
        for flag in os.environ.get("XLA_FLAGS", "").split()
        if not flag.startswith(prefix)
    ]
    return {
        **os.environ,
        # Forced host devices exist only on the CPU platform.
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": " ".join((*flags, f"{prefix}{devices}")),
        _PARITY_CHILD: str(devices),
    }


def execution_group_parity_row(
    smoke: bool, warmup: int, repeats: int, /
) -> list[dict[str, Any]]:
    """Row 9: lanes placed on a four-device execution group (functional only).

    With fewer devices the row reruns itself once in a subprocess on forced CPU
    host devices; that subprocess refuses rather than recursing if the devices
    still do not appear. It reports equality with per-component single-group
    execution, never hardware performance.
    """
    if jax.device_count() < _PARITY_DEVICES:
        if _PARITY_CHILD in os.environ:
            raise RuntimeError(
                f"Forcing {_PARITY_DEVICES} host devices left {jax.device_count()} "
                f"{jax.default_backend()} devices; the execution group cannot be placed."
            )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--rows",
            "execution-group-parity",
            "--warmup",
            str(warmup),
            "--repeats",
            str(repeats),
            *(("--smoke",) if smoke else ()),
        ]
        completed = subprocess.run(
            command,
            env=_forced_host_environment(_PARITY_DEVICES),
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        return json.loads(completed.stdout)["rows"]["execution-group-parity"]
    from phydrax._execution_runtime import ExecutionRuntime

    c = phx.solver.coupling
    count = 6 if smoke else 10
    mortar = c.MortarImposition(c.MortarMultiplier("side-trace", side="plus"))
    plan, cover = _strip_chain(count, 2, 1, mortar)
    group = ExecutionRuntime.current().root_group.spec
    reference = c.prepare_coupled_problem(plan, interface_owners=(cover,))
    placed = c.prepare_coupled_problem(
        _with_execution(
            plan,
            c.CoupledExecutionPolicy(
                lane_capacity=_PARITY_DEVICES, execution_group=group
            ),
        ),
        interface_owners=(cover,),
    )
    size = reference.state_space.size
    state = jnp.asarray(np.random.default_rng(9).normal(size=(size,)))
    policy = _dense_policy(size)
    record, solution = _execution_record(placed, state, policy, warmup, repeats)
    expected = c.solve_coupled_problem(reference, policy=policy)
    rows = reference.row_space
    return [
        {
            "claim": "functional parity only; forced host devices, no hardware performance",
            "devices": jax.device_count(),
            "execution_group_devices": group.device_count,
            "components": count,
            "placed": record,
            "residual_difference": float(
                jnp.max(
                    jnp.abs(
                        rows.flatten(
                            reference.residual(reference.state_space.unflatten(state))
                        )
                        - rows.flatten(
                            placed.residual(placed.state_space.unflatten(state))
                        )
                    )
                )
            ),
            "solution_difference": float(
                jnp.max(
                    jnp.abs(
                        reference.state_space.flatten(expected.state)
                        - placed.state_space.flatten(solution.state)
                    )
                )
            ),
        }
    ]


def _sensor_observations(sensors: int, /) -> Callable[[tuple[Any, ...]], tuple[Any, ...]]:
    """Point temperature sensors on a ``sqrt(sensors)`` grid inside every strip."""
    c, m = phx.solver.coupling, phx.measurement
    side = int(round(np.sqrt(sensors)))
    local = (np.arange(side) + 0.5) / side
    grid = np.stack(np.meshgrid(local, local, indexing="xy"), -1).reshape(-1, 2)
    quantity = m.QuantitySpec(
        "benchmark", "temperature", "temperature", phx.units.KELVIN, "temperature"
    )
    contract = phx.SpatialCoordinateContract(phx.units.METER)
    point = m.SamplingSemantics(m.SpatialSamplingKind.POINT)

    def observations(components: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(
            c.FieldPointObservation(
                f"{component.name}-sensors",
                component.name,
                "u",
                quantity=quantity,
                support=m.PointSampleSupport(
                    np.column_stack(
                        (grid[:, 0] + k, grid[:, 1], np.zeros((grid.shape[0],)))
                    ),
                    tuple(f"p{index}" for index in range(grid.shape[0])),
                    contract,
                ),
                sampling=point,
                field_unit=phx.units.KELVIN,
            )
            for k, component in enumerate(components)
        )

    return observations


def _observation_count_case(sensors: int, warmup: int, repeats: int, /) -> dict[str, Any]:
    c = phx.solver.coupling
    count = 4
    mortar = c.MortarImposition(c.MortarMultiplier("side-trace", side="plus"))
    plan, cover = _strip_chain(
        count, 8, 1, mortar, observations=_sensor_observations(sensors)
    )
    components = {component.name: component for component in plan.components}
    prepared_observations, observation_seconds = measure_host(
        lambda: tuple(binding.prepare(components) for binding in plan.observations)
    )
    prepared = c.prepare_coupled_problem(plan, interface_owners=(cover,))
    policy = _dense_policy(prepared.state_space.size)
    solution = c.solve_coupled_problem(prepared, policy=policy)
    fields = {(name, "u"): solution.field(name, "u") for name in components}
    arguments = prepared.arguments()
    flat = jnp.concatenate([fields[(name, "u")] for name in components])
    sizes = np.cumsum([fields[(name, "u")].size for name in components])[:-1]

    def observe(values: jax.Array) -> jax.Array:
        split = dict(zip(fields, jnp.split(values, sizes), strict=True))
        return jnp.concatenate(
            [
                observation.evaluate(split, arguments).values
                for observation in prepared_observations
            ]
        )

    predicted = np.asarray(observe(flat))
    points = np.concatenate(
        [np.asarray(binding.points)[:, :2] for binding in plan.observations]
    )
    _require_correct("coupled-observation-count", accepted=bool(solution.accepted))
    return {
        "sensors_per_component": int(predicted.size // count),
        "sensors": int(predicted.size),
        "observation_preparation_seconds": observation_seconds,
        "evaluation": _phased_timing(observe, flat, warmup=warmup, repeats=repeats),
        "logical_observation_bytes": logical_array_bytes(prepared_observations),
        "max_sensor_error_vs_exact": float(
            np.max(np.abs(predicted - _host_harmonic(points)))
        ),
        "accepted": bool(solution.accepted),
    }


def coupled_observation_count_row(
    smoke: bool, warmup: int, repeats: int, /
) -> list[dict[str, Any]]:
    """Observation-count row: point sensors per strip on a four-strip mortar chain.

    Scales the sensor count; measures host observation preparation (point
    location), the compiled evaluation of every observation from the solved
    fields, retained observation bytes, and the error against the exact field.
    """
    counts = (4, 64) if smoke else (4, 64, 256, 1024)
    return [_observation_count_case(sensors, warmup, repeats) for sensors in counts]


def _temporal_policy() -> phx.solver.DAESolvePolicy:
    termination = phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-10,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=12,
    )
    return phx.solver.DAESolvePolicy(
        method=phx.solver.BDFMethod(2),
        nonlinear_termination=termination,
        initialization_termination=termination,
    )


def _temporal_samples_case(
    transient: Any, steps: int, warmup: int, repeats: int, /
) -> dict[str, Any]:
    c = phx.solver.coupling
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 0.2, steps + 1), time_id=f"steps-{steps}"
    )
    zero = transient.state_space.zeros()
    policy = _temporal_policy()
    solution, first = measure_synchronized(
        lambda: c.solve_coupled_transient(transient, zero, grid, policy=policy)
    )
    _, warm = measure_repeated(
        lambda: c.solve_coupled_transient(transient, zero, grid, policy=policy),
        warmup=warmup,
        repeats=repeats,
    )
    certificate = solution.certificate
    _require_correct(
        "coupled-temporal-samples",
        native_successful=bool(solution.native_successful),
        accepted=bool(solution.accepted),
    )
    return {
        "samples": steps + 1,
        "first_synchronized_seconds": first,
        "warm_execution": warm.to_dict(),
        "certified_rows": list(certificate.residual_norms.shape),
        "max_certified_defect_ratio": float(
            jnp.max(certificate.residual_norms / certificate.scales)
        ),
        "history_bytes": logical_array_bytes(solution.dae),
        "native_successful": bool(solution.native_successful),
        "accepted": bool(solution.accepted),
    }


def coupled_temporal_samples_row(
    smoke: bool, warmup: int, repeats: int, /
) -> list[dict[str, Any]]:
    """Temporal row: BDF2 transient of two strips joined by matching elimination.

    The prepared transient (capacity blocks, eliminated interface rows, one
    index-one DAE) is reused while the number of time samples grows; records
    the host preparation once and, per sample count, the first synchronized
    and warmed certified solves, certificate extent, and retained history.
    """
    c = phx.solver.coupling
    (plan, cover), owner_seconds = measure_host(
        lambda: _strip_chain(2, 4, 1, c.MatchingElimination(eliminated="plus"))
    )
    prepared = c.prepare_coupled_problem(plan, interface_owners=(cover,))
    names = tuple(component.name for component in prepared.components)
    transient, transient_seconds = measure_host(
        lambda: c.prepare_coupled_transient(
            prepared,
            fields=tuple(c.TransientField(name, "u") for name in names),
            arguments=lambda time, parameters: dict.fromkeys(names),
            arguments_id="steady-harmonic-walls",
        )
    )
    steps = (8, 16) if smoke else (8, 32, 128)
    return [
        {
            "owner_compilation_seconds": owner_seconds,
            "transient_preparation_seconds": transient_seconds,
            **_temporal_samples_case(transient, count, warmup, repeats),
        }
        for count in steps
    ]


def _coupled_transition_case(
    transition: Any, members: int, warmup: int, repeats: int, /
) -> dict[str, Any]:
    """One certified window of ``members`` independent states (an ensemble forecast)."""
    size = transition.state_size
    states = jnp.asarray(
        np.random.default_rng(members).normal(scale=0.1, size=(members, size))
    )

    def window(values: jax.Array) -> Any:
        return jax.vmap(lambda state: transition.step(0.0, 0.05, state, {}))(values)

    timing = _compiled_timing(
        lambda values: window(values).accepted,
        states,
        warmup=warmup,
        repeats=repeats,
    )
    evidence = jax.jit(window)(states)
    all_accepted = bool(jnp.all(evidence.successful))
    _require_correct("coupled-transition", all_accepted=all_accepted)
    return {
        "members": members,
        "state_size": size,
        **timing,
        "all_accepted": all_accepted,
        "max_certified_defect_ratio": float(jnp.max(evidence.residual_ratio)),
    }


def coupled_transition_row(
    smoke: bool, warmup: int, repeats: int, /
) -> list[dict[str, Any]]:
    """Transition row: certified coupled windows as a discrete Markov transition.

    The strip transient of the temporal row is prepared once as a
    `PreparedCoupledTransition` (one native DAE solve on a template window of
    four BDF steps, rebound to each interval); records the host preparation and,
    per ensemble size, lowering, compilation, and warmed execution of one
    certified window for every member, as the ensemble filter's forecast runs it.
    """
    c = phx.solver.coupling
    (plan, cover), owner_seconds = measure_host(
        lambda: _strip_chain(2, 4, 1, c.MatchingElimination(eliminated="plus"))
    )
    prepared = c.prepare_coupled_problem(plan, interface_owners=(cover,))
    names = tuple(component.name for component in prepared.components)
    transient = c.prepare_coupled_transient(
        prepared,
        fields=tuple(c.TransientField(name, "u") for name in names),
        arguments=lambda time, parameters: dict.fromkeys(names),
        arguments_id="steady-harmonic-walls",
        parameters={},
    )
    transition, transition_seconds = measure_host(
        lambda: c.prepare_coupled_transition(
            transient, step=0.05, policy=_temporal_policy(), parameters={}, substeps=4
        )
    )
    members = (4, 16) if smoke else (4, 16, 64)
    return [
        {
            "owner_compilation_seconds": owner_seconds,
            "transition_preparation_seconds": transition_seconds,
            "logical_transition_bytes": logical_array_bytes(transition),
            **_coupled_transition_case(transition, count, warmup, repeats),
        }
        for count in members
    ]


ROWS: dict[str, RowFunction] = {
    "prepared-query": prepared_query_row,
    "strong-form-query": strong_form_query_row,
    "conductivity-rebind": conductivity_rebind_row,
    "isogeometric-query": isogeometric_query_row,
    "finite-volume-face-trace": finite_volume_face_trace_row,
    "interface-coupling": interface_coupling_row,
    "coupled-assembly": coupled_assembly_row,
    "boundary-integral-law": boundary_integral_law_row,
    "sem-vem-bem-flagship": sem_vem_bem_flagship_row,
    "coupled-derivative": coupled_derivative_row,
    "coupled-repeated-bind": coupled_repeated_bind_row,
    "homogeneous-worksets": homogeneous_worksets_row,
    "execution-group-parity": execution_group_parity_row,
    "coupled-observation-count": coupled_observation_count_row,
    "coupled-temporal-samples": coupled_temporal_samples_row,
    "coupled-transition": coupled_transition_row,
}


def _run_row(name: str, arguments: argparse.Namespace, /) -> list[dict[str, Any]]:
    """Run one registered row; only the worksets row takes the component cap."""
    match name:
        case "homogeneous-worksets":
            return homogeneous_worksets_row(
                arguments.smoke,
                arguments.warmup,
                arguments.repeats,
                max_components=arguments.max_components,
            )
        case _:
            return ROWS[name](arguments.smoke, arguments.warmup, arguments.repeats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", choices=tuple(ROWS), default=tuple(ROWS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-components",
        type=int,
        default=None,
        help="Cap the homogeneous-worksets component count (default: 64).",
    )
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.repeats < 1:
        raise ValueError("warmup >= 0 and repeats >= 1 are required.")
    payload = {
        "benchmark": "numerical-interoperability",
        "smoke": bool(arguments.smoke),
        "max_components": arguments.max_components,
        "environment": capture_environment().to_dict(),
        "rows": {name: _run_row(name, arguments) for name in arguments.rows},
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
