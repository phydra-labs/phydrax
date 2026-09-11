#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import (
    logical_array_bytes,
    measure_repeated,
    measure_synchronized,
    synchronize,
)


@dataclass(frozen=True)
class SphericalSpectralBenchmarkRecord:
    schema_version: int
    bandlimit: int
    sampling: str
    execution: str
    radius: float
    physical_points: int
    padded_coefficients: int
    logical_modes: int
    precompute_bytes: int
    prepare_ms: float
    project_first_jit_ms: float
    project_steady_ms: float
    laplacian_first_jit_ms: float
    laplacian_steady_ms: float
    dynamic_point_count: int
    dynamic_direction_bytes: int
    dynamic_directions_are_runtime_inputs: bool
    dynamic_evaluate_first_jit_ms: float
    dynamic_evaluate_steady_ms: float
    dynamic_evaluate_checksum: float
    dynamic_grid_reconstruction_error: float
    fixed_sample_capacity: int
    fixed_sample_active_count: int
    fixed_sample_design_bytes: int
    fixed_sample_factor_bytes: int
    fixed_sample_condition_number: float
    fixed_directions_are_prepared: bool
    fixed_sample_prepare_ms: float
    fixed_sample_evaluate_first_ms: float
    fixed_sample_evaluate_steady_ms: float
    fixed_sample_evaluate_checksum: float
    fixed_sample_grid_reconstruction_error: float
    roundtrip_error: float
    area_defect: float
    laplacian_error: float
    invalid_capacity_error: float
    finite: bool

    @property
    def passed(self) -> bool:
        return (
            self.finite
            and self.roundtrip_error <= 1e-9
            and self.area_defect <= 1e-10
            and self.laplacian_error <= 1e-9
            and self.dynamic_grid_reconstruction_error <= 1e-9
            and self.fixed_sample_grid_reconstruction_error <= 1e-9
            and self.invalid_capacity_error <= 1e-11
        )


def _measure(function, argument, repeats):
    compiled = eqx.filter_jit(function)
    value, first_seconds = measure_synchronized(lambda: compiled(argument))
    value, distribution = measure_repeated(
        lambda: compiled(argument),
        warmup=0,
        repeats=repeats,
    )
    return (
        value,
        1_000.0 * first_seconds,
        1_000.0 * float(distribution.mean_seconds),
    )


def _measure_uncompiled(function, argument, repeats):
    value, first_seconds = measure_synchronized(lambda: function(argument))
    value, distribution = measure_repeated(
        lambda: function(argument),
        warmup=0,
        repeats=repeats,
    )
    return (
        value,
        1_000.0 * first_seconds,
        1_000.0 * float(distribution.mean_seconds),
    )


def _tree_checksum(tree: Any, /) -> float:
    return sum(
        float(jnp.sum(jnp.abs(leaf)))
        for leaf in jax.tree_util.tree_leaves(tree)
        if isinstance(leaf, jax.Array)
    )


def _compiled_case(
    function, arguments, /, *, repeats: int, reference: Any | None = None
) -> dict[str, Any]:
    lowered = eqx.filter_jit(function).lower(*arguments)
    started = time.perf_counter()
    executable = lowered.compile()
    compile_ms = 1e3 * (time.perf_counter() - started)
    value = executable(*arguments)
    synchronize(value)
    started = time.perf_counter()
    for _ in range(repeats):
        value = executable(*arguments)
        synchronize(value)
    steady_ms = 1e3 * (time.perf_counter() - started) / repeats
    memory = executable.compiled.memory_analysis()
    metrics = {
        "compile_ms": compile_ms,
        "steady_ms": steady_ms,
        "checksum": _tree_checksum(value),
        "output_bytes": logical_array_bytes(value),
        "retained_bytes": (
            None if memory is None else int(memory.generated_code_size_in_bytes)
        ),
        "workspace_bytes": None if memory is None else int(memory.temp_size_in_bytes),
    }
    if reference is not None:
        value_leaves, value_structure = jax.tree_util.tree_flatten(value)
        reference_leaves, reference_structure = jax.tree_util.tree_flatten(reference)
        if value_structure != reference_structure:
            raise ValueError(
                "Benchmark result and reference must have the same structure."
            )
        metrics["reference_max_abs_error"] = max(
            (
                float(jnp.max(jnp.abs(actual - expected)))
                for actual, expected in zip(value_leaves, reference_leaves, strict=True)
            ),
            default=0.0,
        )
    return metrics


def _solid_reference(function, coefficients, points, bandlimit):
    offset = bandlimit - 1
    values = jnp.zeros(
        (points.shape[0], coefficients.shape[2]),
        dtype=jnp.result_type(coefficients, jnp.complex64),
    )
    for degree in range(bandlimit):
        for order in range(-degree, degree + 1):
            basis = function(degree, order, points)
            values = values + basis[:, None] * coefficients[degree, order + offset]
    return values


def spherical_function_closure_metrics(
    bandlimit: int, /, *, repeats: int
) -> dict[str, Any]:
    """Bounded spin-point and solid-harmonic timings and resource counters."""
    from phydrax.discretization import spectral

    limit = min(max(int(bandlimit), 2), 8)
    point_count = 4 * limit
    theta = jnp.linspace(0.2, jnp.pi - 0.2, point_count)
    phi = jnp.linspace(-jnp.pi, jnp.pi, point_count)
    radius = jnp.linspace(0.6, 1.4, point_count)
    directions = jnp.stack(
        (
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ),
        axis=-1,
    )
    points = radius[:, None] * directions
    degree = min(5, limit - 1)
    order = min(2, degree)
    scalar_reference = (
        radius**degree * phx.special.sph_harm_y_cart(degree, order, directions),
        radius ** (-degree - 1) * phx.special.sph_harm_y_cart(degree, order, directions),
    )
    metrics: dict[str, Any] = {
        "bandlimit": limit,
        "point_count": point_count,
        "solid_scalar": _compiled_case(
            lambda vectors: (
                phx.special.solid_harmonic_regular(degree, order, vectors),
                phx.special.solid_harmonic_irregular(degree, order, vectors),
            ),
            (points,),
            repeats=repeats,
            reference=scalar_reference,
        ),
        "solid_synthesis": {},
    }
    coefficient_grid = jnp.arange(
        limit * (2 * limit - 1) * 2,
        dtype=jnp.float64,
    ).reshape((limit, 2 * limit - 1, 2))
    coefficients = (coefficient_grid + 1.0j * coefficient_grid[::-1]) / (
        coefficient_grid.size + 1.0
    )
    for kind, function in (
        ("regular", phx.special.solid_harmonic_regular),
        ("irregular", phx.special.solid_harmonic_irregular),
    ):
        started = time.perf_counter()
        prepared = spectral.SolidHarmonicPlan(limit, kind=kind, reality=False).prepare()
        synchronize(prepared)
        prepare_ms = 1e3 * (time.perf_counter() - started)
        modal = prepared.plan.layout.mask_invalid(coefficients)
        reference = _solid_reference(function, modal, points, limit)
        case = _compiled_case(
            prepared.evaluate,
            (modal, points),
            repeats=repeats,
            reference=reference,
        )
        case.update(
            {
                "prepare_ms": prepare_ms,
                "prepared_array_bytes": logical_array_bytes(prepared),
                "resources": dict(prepared.preparation.resource_counts),
            }
        )
        metrics["solid_synthesis"][kind] = case
    spin = min(2, limit - 1)
    started = time.perf_counter()
    spin_space = phx.discretization.SphericalSpectralPlan(
        limit,
        spin=spin,
        reality=False,
        precision=spectral.SpectralPrecisionPolicy(jnp.complex128),
    ).prepare()
    synchronize(spin_space)
    spin_prepare_ms = 1e3 * (time.perf_counter() - started)
    spin_coefficients = spin_space.layout.mask_invalid(coefficients)
    east = jnp.stack(
        (-jnp.sin(phi), jnp.cos(phi), jnp.zeros_like(phi)),
        axis=-1,
    )
    north = jnp.stack(
        (
            -jnp.cos(theta) * jnp.cos(phi),
            -jnp.cos(theta) * jnp.sin(phi),
            jnp.sin(theta),
        ),
        axis=-1,
    )
    frame_angle = jnp.linspace(-0.7, 0.8, point_count)
    angular_reference = spin_space.evaluate_angles(
        spin_coefficients,
        theta,
        phi,
    )
    frame_phase = jnp.exp(-1.0j * spin * frame_angle)[:, None]
    metrics["spin_point_evaluation"] = {
        "spin": spin,
        "prepare_ms": spin_prepare_ms,
        "prepared_array_bytes": logical_array_bytes(spin_space),
        "resources": dict(spin_space.preparation.resource_counts),
        "angles": _compiled_case(
            lambda modal, polar, azimuth, frame: (
                spin_space.evaluate_angles(modal, polar, azimuth),
                spin_space.evaluate_angles(
                    modal,
                    polar,
                    azimuth,
                    frame_angle=frame,
                ),
            ),
            (spin_coefficients, theta, phi, frame_angle),
            repeats=repeats,
            reference=(angular_reference, angular_reference * frame_phase),
        ),
        "cartesian": _compiled_case(
            lambda modal, vectors, east_, north_: spin_space.evaluate(
                modal,
                vectors,
                tangent_frame=(east_, north_),
            ),
            (spin_coefficients, directions, east, north),
            repeats=repeats,
            reference=angular_reference,
        ),
    }
    return metrics


def run_spherical_spectral_benchmark(
    bandlimit: int = 16,
    /,
    *,
    sampling: str = "mw",
    execution: str = "recursive",
    radius: float = 1.0,
    repeats: int = 5,
) -> SphericalSpectralBenchmarkRecord:
    limit = int(bandlimit)
    repeat_count = int(repeats)
    if limit < 2 or repeat_count < 1:
        raise ValueError("bandlimit must exceed one and repeats must be positive.")
    started = time.perf_counter()
    space = phx.discretization.SphericalSpectralPlan(
        limit,
        sampling=sampling,
        execution=execution,
    ).prepare(radius=radius)
    jax.block_until_ready(space.transform.theta)
    prepare_ms = 1e3 * (time.perf_counter() - started)
    theta, phi = jnp.meshgrid(
        space.transform.theta,
        space.transform.phi,
        indexing="ij",
    )
    del phi
    values = jnp.cos(theta)
    coefficients, project_first, project_steady = _measure(
        space.project,
        values,
        repeat_count,
    )
    laplacian, laplacian_first, laplacian_steady = _measure(
        space.laplacian,
        values,
        repeat_count,
    )
    dynamic_values, dynamic_first, dynamic_steady = _measure(
        lambda inputs: space.evaluate(inputs[0], inputs[1]),
        (coefficients, space.points),
        repeat_count,
    )
    from phydrax.discretization import spectral

    started = time.perf_counter()
    fixed_samples = spectral.SphericalSamplePlan(
        space.points,
        weights=space.quadrature_weights.reshape((-1,)),
        tikhonov=1e-12,
    ).prepare(space)
    jax.block_until_ready(fixed_samples.design)
    fixed_sample_prepare_ms = 1e3 * (time.perf_counter() - started)
    fixed_values, fixed_first, fixed_steady = _measure_uncompiled(
        fixed_samples.evaluate,
        coefficients,
        repeat_count,
    )
    reconstructed = space.reconstruct(coefficients)
    contaminated = coefficients.at[0, 0].set(jnp.nan + 1j * jnp.inf)
    inert = space.reconstruct(contaminated)
    expected_area = 4.0 * jnp.pi * float(radius) ** 2
    expected_laplacian = -2.0 / float(radius) ** 2 * values
    roundtrip_error = jnp.max(jnp.abs(reconstructed - values))
    area_defect = jnp.abs(jnp.sum(space.quadrature_weights) - expected_area)
    laplacian_error = jnp.max(jnp.abs(laplacian - expected_laplacian))
    invalid_error = jnp.max(jnp.abs(inert - reconstructed))
    flattened_reconstruction = reconstructed.reshape((-1,))
    dynamic_error = jnp.max(jnp.abs(dynamic_values - flattened_reconstruction))
    fixed_sample_error = jnp.max(jnp.abs(fixed_values - flattened_reconstruction))
    finite = bool(
        jnp.all(jnp.isfinite(coefficients))
        & jnp.all(jnp.isfinite(laplacian))
        & jnp.all(jnp.isfinite(inert))
        & jnp.all(jnp.isfinite(dynamic_values))
        & jnp.all(jnp.isfinite(fixed_values))
    )
    resources = dict(space.preparation.resource_counts)
    return SphericalSpectralBenchmarkRecord(
        schema_version=1,
        bandlimit=limit,
        sampling=space.transform.sampling,
        execution=space.transform.execution,
        radius=float(radius),
        physical_points=resources["physical_points"],
        padded_coefficients=resources["padded_coefficients"],
        logical_modes=resources["logical_modes"],
        precompute_bytes=resources["precompute_bytes"],
        prepare_ms=float(prepare_ms),
        project_first_jit_ms=float(project_first),
        project_steady_ms=float(project_steady),
        laplacian_first_jit_ms=float(laplacian_first),
        laplacian_steady_ms=float(laplacian_steady),
        dynamic_point_count=int(space.points.shape[0]),
        dynamic_direction_bytes=int(space.points.nbytes),
        dynamic_directions_are_runtime_inputs=True,
        dynamic_evaluate_first_jit_ms=float(dynamic_first),
        dynamic_evaluate_steady_ms=float(dynamic_steady),
        dynamic_evaluate_checksum=float(jnp.sum(jnp.abs(dynamic_values))),
        dynamic_grid_reconstruction_error=float(dynamic_error),
        fixed_sample_capacity=fixed_samples.report.sample_capacity,
        fixed_sample_active_count=fixed_samples.report.active_count,
        fixed_sample_design_bytes=fixed_samples.report.design_bytes,
        fixed_sample_factor_bytes=fixed_samples.report.factor_bytes,
        fixed_sample_condition_number=fixed_samples.report.condition_number,
        fixed_directions_are_prepared=True,
        fixed_sample_prepare_ms=float(fixed_sample_prepare_ms),
        fixed_sample_evaluate_first_ms=float(fixed_first),
        fixed_sample_evaluate_steady_ms=float(fixed_steady),
        fixed_sample_evaluate_checksum=float(jnp.sum(jnp.abs(fixed_values))),
        fixed_sample_grid_reconstruction_error=float(fixed_sample_error),
        roundtrip_error=float(roundtrip_error),
        area_defect=float(area_defect),
        laplacian_error=float(laplacian_error),
        invalid_capacity_error=float(invalid_error),
        finite=finite,
    )


def spherical_completion_metrics(
    discretization: phx.discretization.SphericalSpectralDiscretization,
    /,
    *,
    nside: int,
) -> dict[str, int | float]:
    """Bounded scattered/HEALPix, Wigner, and CG capacity counters."""
    from phydrax.discretization import spectral

    samples = spectral.SphericalSamplePlan.healpix(
        int(nside), ordering="nested", tikhonov=1e-12
    ).prepare(discretization)
    rotation = spectral.SphericalRotationPlan(discretization).prepare()
    coupling = spectral.SphericalClebschGordanPlan(
        discretization,
        discretization,
        output_bandlimit=2 * discretization.layout.bandlimit - 1,
    ).prepare()
    identity_blocks = rotation.wigner_d(jnp.zeros((3,)))
    return {
        "sample_capacity": samples.report.sample_capacity,
        "active_sample_count": samples.report.active_count,
        "sample_design_bytes": samples.report.design_bytes,
        "sample_factor_bytes": samples.report.factor_bytes,
        "sample_condition_number": samples.report.condition_number,
        "wigner_block_bytes": int(identity_blocks.nbytes),
        "cg_coupling_count": coupling.report.coupling_count,
        "cg_coefficient_bytes": coupling.report.coefficient_bytes,
        "cg_recurrence_residual": coupling.report.recurrence_residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exact-sampling spherical spectral workflows."
    )
    parser.add_argument("--bandlimit", type=int, default=16)
    parser.add_argument("--sampling", choices=("mw", "mwss", "dh", "gl"), default="mw")
    parser.add_argument(
        "--execution", choices=("recursive", "precomputed"), default="recursive"
    )
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    bandlimit = 4 if arguments.smoke else arguments.bandlimit
    repeats = 1 if arguments.smoke else arguments.repeats
    record = run_spherical_spectral_benchmark(
        bandlimit,
        sampling=arguments.sampling,
        execution=arguments.execution,
        radius=arguments.radius,
        repeats=repeats,
    )
    payload = json.dumps(
        {
            **asdict(record),
            "function_space_closure": spherical_function_closure_metrics(
                bandlimit,
                repeats=repeats,
            ),
            "passed": record.passed,
        },
        indent=2,
    )
    if arguments.output is not None:
        temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
        temporary.write_text(payload + "\n")
        os.replace(temporary, arguments.output)
    print(payload)
    if not record.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
