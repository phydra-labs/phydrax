# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Explicit compilation evidence for raw cuEST FFI operations, not whole-SCF JIT.

The caller owns native admission and independent scientific qualification. Recipe
arrays and numerical operands are dynamic executable inputs; only result shapes,
target names, and immutable scalar policy attributes are captured. Factorization,
AO conversions, host references, and sample validation are outside kernel timers.
XLA estimates exclude opaque vendor arithmetic, caches, and private workspaces.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from math import isfinite
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from benchmarks._runtime import (
    compiler_evidence,
    DurationDistribution,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_synchronized,
    synchronize,
)
from phydrax.chemistry.electronic_structure._mean_field import RestrictedMeanFieldState
from phydrax.chemistry.electronic_structure._molecular_hf import MolecularGradientResult
from phydrax.operators.quantum.gaussian._cuest import _density_factors, CuestGeometry
from phydrax.operators.quantum.gaussian._provider import (
    MolecularDensityFactorizationEvidence,
)


type _Attributes = dict[str, np.int64 | np.float64]
type _HostArray = NDArray[np.float64]

_RECIPE_NAMES = (
    "geometry_key",
    "orbital_shell_metadata",
    "orbital_exponents",
    "orbital_coefficients",
    "auxiliary_shell_metadata",
    "auxiliary_exponents",
    "auxiliary_coefficients",
    "positions_bohr",
    "nuclear_charges",
)


def _make_kernel(
    target: str,
    shapes: tuple[jax.ShapeDtypeStruct, ...],
    attributes: _Attributes,
    /,
) -> Callable[[tuple[Array, ...]], tuple[Array, ...]]:
    """Capture no geometry, state, guard, layout, or numerical array."""

    def kernel(arguments: tuple[Array, ...]) -> tuple[Array, ...]:
        return tuple(
            jax.ffi.ffi_call(target, shapes, has_side_effect=True)(
                *arguments, **attributes
            )
        )

    return kernel


def _validate_outputs(
    status: int,
    actual: tuple[_HostArray, ...],
    expected: tuple[_HostArray, ...],
    names: tuple[str, ...],
    /,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Fail closed after synchronization, never within a timed kernel call."""
    checks = {"status_zero": status == 0}
    errors: dict[str, float] = {}
    for name, value, reference in zip(names, actual, expected, strict=True):
        finite = bool(np.all(np.isfinite(value)) and np.all(np.isfinite(reference)))
        checks[f"{name}_finite"] = finite
        matches = finite and value.shape == reference.shape
        checks[f"{name}_matches_host"] = bool(
            matches and np.allclose(value, reference, rtol=rtol, atol=atol)
        )
        if matches:
            errors[name] = float(np.max(np.abs(value - reference), initial=0.0))
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"Raw cuEST compiler sample failed qualification: {failed}")
    return {"scientific_checks": checks, "maximum_absolute_errors": errors}


def _optional_measured(value: Array, /) -> float | None:
    result = float(value)
    return result if isfinite(result) else None


def measure_cuest_kernels(
    bound: CuestGeometry,
    state: RestrictedMeanFieldState,
    gradient: MolecularGradientResult,
    /,
    *,
    repeats: int,
) -> dict[str, Any]:
    """Lower, compile, and execute OE/J/K/contracted-gradient raw FFI kernels.

    Every first and warm sample is synchronized inside its timer, then checked
    outside it. A failed status, nonfinite value, or scientific mismatch raises;
    this helper cannot turn a failed raw comparison into a successful report.
    Native binding/component registration must already have completed. The
    external memory observation window is owned by the caller.
    """
    # Deferred to reuse the benchmark's declared gates without an import cycle.
    from benchmarks.cuest_molecular import (
        _DF_FORCE_ATOL,
        _DF_FORCE_RTOL,
        _DF_JK_ATOL,
        _DF_JK_RTOL,
        _OE_MATRIX_ATOL,
        _OE_MATRIX_RTOL,
    )

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Raw cuEST kernel repeats must be a positive integer.")
    bound._validate_bound()
    if state.integral_definition_id != bound.prepared.definition_id:
        raise ValueError("Compiler study state belongs to another integral definition.")
    if gradient.state_id != state.state_id:
        raise ValueError("Compiler study gradient belongs to another SCF state.")
    if not bool(state.evidence.converged) or not bool(gradient.successful):
        raise ValueError("Compiler study requires converged SCF and qualified gradients.")

    layout, plan = bound.prepared.orbital_layout, bound.prepared.plan
    occupied = np.flatnonzero(np.asarray(state.occupations) > 0.0)

    def prepare_factors() -> tuple[Array, MolecularDensityFactorizationEvidence]:
        candidate = (
            jnp.sqrt(state.occupations[occupied])[:, None]
            * state.coefficients[:, occupied].T
        )
        return _density_factors(
            state.density, plan.density_factorization_tolerance, candidate
        )

    (factors, certification), factor_seconds = measure_synchronized(prepare_factors)
    # Canonical Pulay density: occupations and energies weight orbital columns.
    weighted = (
        state.coefficients * (state.occupations * state.orbital_energies)[None, :]
    ) @ state.coefficients.T
    vendor_density = layout.density_to_vendor(state.density)
    vendor_factors = layout.factors_to_vendor(factors)
    vendor_weighted = layout.energy_weighted_density_to_vendor(weighted)
    recipe = bound._arguments()
    synchronize((recipe, vendor_density, vendor_factors, vendor_weighted))

    # References are the already independently qualified canonical host path.
    # These warm native geometry/cache state; first below means first executable
    # invocation after explicit compile, NOT cold vendor initialization.
    one = bound.one_electron()
    jk = bound.jk(state.density, factors=factors)
    synchronize((one, jk))
    if not bool(one.successful) or not bool(jk.successful):
        raise RuntimeError("Canonical cuEST compiler-study reference unsuccessful.")

    matrix = jax.ShapeDtypeStruct(state.density.shape, jnp.float64)
    force = jax.ShapeDtypeStruct(bound.positions.shape, jnp.float64)
    status = jax.ShapeDtypeStruct((), jnp.int32)
    attributes: _Attributes = {
        "representation": np.int64(layout.pure),
        "fitting_cutoff": np.float64(plan.fitting_cutoff),
        "relative_conditioning": np.int64(plan.relative_conditioning),
        "inversion": np.int64(0 if plan.fitting_algorithm == "qr" else 1),
        "screening": np.float64(plan.screening_threshold),
        "variable_workspace_bytes": np.int64(plan.variable_workspace_bytes),
        "workspace_limit_bytes": np.int64(plan.maximum_workspace_bytes),
        "cache_capacity": np.int64(plan.maximum_cached_geometries),
        "partition": np.int64(bound.prepared.partition),
    }
    configurations = (
        (
            "one_electron",
            "phydrax_cuest_one_electron",
            (matrix, matrix, matrix, status),
            recipe,
            _RECIPE_NAMES,
            (one.overlap, one.kinetic, one.nuclear_attraction),
            ("overlap", "kinetic", "nuclear_attraction"),
            _OE_MATRIX_RTOL,
            _OE_MATRIX_ATOL,
        ),
        (
            "jk",
            "phydrax_cuest_jk",
            (matrix, matrix, status),
            recipe + (vendor_density, vendor_factors),
            _RECIPE_NAMES + ("vendor_density", "vendor_factors"),
            (jk.coulomb, jk.exchange),
            ("coulomb", "exchange"),
            _DF_JK_RTOL,
            _DF_JK_ATOL,
        ),
        (
            "gradient",
            "phydrax_cuest_gradient",
            (force, force, force, status),
            recipe + (vendor_density, vendor_factors, vendor_weighted),
            _RECIPE_NAMES + ("vendor_density", "vendor_factors", "vendor_pulay_density"),
            (
                gradient.one_electron,
                gradient.two_electron,
                gradient.pulay,
                gradient.gradient,
                gradient.forces,
            ),
            ("one_electron", "two_electron", "pulay", "total_gradient", "forces"),
            _DF_FORCE_RTOL,
            _DF_FORCE_ATOL,
        ),
    )
    results: dict[str, Any] = {}
    for (
        name,
        target,
        shapes,
        arguments,
        input_names,
        references,
        names,
        rtol,
        atol,
    ) in configurations:
        kernel = jax.jit(_make_kernel(target, shapes, attributes))
        executable, timing = measure_lower_and_compile(
            lambda: kernel.lower(arguments), lambda lowered: lowered.compile()
        )
        memory = executable.memory_analysis()
        cost = executable.cost_analysis()
        missing = []
        if cost is None:
            missing.append("Compiled.cost_analysis() returned None")
        else:
            for field in ("flops", "bytes accessed"):
                if field not in cost:
                    missing.append(f"Compiled.cost_analysis() omitted {field!r}")
        if memory is None:
            missing.append("Compiled.memory_analysis() returned None")
        evidence = compiler_evidence(
            cost,
            memory,
            source="JAX compiled raw cuEST FFI executable cost_analysis/memory_analysis",
            unavailable_reason="; ".join(missing) if missing else None,
        )
        expected = tuple(np.asarray(value) for value in references)

        def validate(sample: tuple[Array, ...]) -> dict[str, Any]:
            values = sample[:-1]
            if name == "gradient":
                actual = tuple(np.asarray(value) for value in values)
                total = actual[0] + actual[1] + actual[2] + np.asarray(gradient.nuclear)
                actual += (total, -total)
            else:
                actual = tuple(
                    np.asarray(layout.matrix_from_vendor(value)) for value in values
                )
            return _validate_outputs(
                int(sample[-1]), actual, expected, names, rtol=rtol, atol=atol
            )

        sample, first_seconds = measure_synchronized(lambda: executable(arguments))
        first_validation = validate(sample)
        warm_seconds: list[float] = []
        warm_validation: list[dict[str, Any]] = []
        for _ in range(repeats):
            sample, elapsed = measure_synchronized(lambda: executable(arguments))
            warm_seconds.append(elapsed)
            warm_validation.append(validate(sample))
        results[name] = {
            "target": target,
            "timings_seconds": {
                "lowering": timing.lowering_seconds,
                "compile": timing.compilation_seconds,
                "first_execution": first_seconds,
                "warm": DurationDistribution(tuple(warm_seconds)).to_seconds_dict(),
            },
            "compiler": {
                **asdict(evidence),
                "raw_cost_analysis": None if cost is None else dict(cost),
                "estimated_device_memory_bytes": evidence.estimated_device_memory_bytes,
                "scope": "XLA-visible executable buffers and estimates only",
                "excludes": [
                    "opaque vendor arithmetic and data traffic",
                    "vendor/private workspaces",
                    "native geometry caches",
                    "CUDA driver/JIT and process allocations",
                ],
                "zero_xla_temporary_bytes_is_not_zero_vendor_memory": True,
            },
            "dynamic_inputs": [
                {
                    "name": label,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "logical_bytes": logical_array_bytes(value),
                }
                for label, value in zip(input_names, arguments, strict=True)
            ],
            "dynamic_input_logical_bytes": logical_array_bytes(arguments),
            "output_logical_bytes": logical_array_bytes(sample),
            "tolerances": {"relative": rtol, "absolute": atol},
            "first_validation": first_validation,
            "warm_validation": warm_validation,
        }

    return {
        "scope": "representative raw FFI kernels; not whole-SCF JIT or end-to-end speedup",
        "first_execution_scope": "first explicit-compiled executable call; native geometry/cache already warm",
        "timing_excludes": [
            "factor preparation/certification",
            "AO-layout transforms",
            "reference generation",
            "status/finite/scientific checks and host transfers for validation",
        ],
        "factor_preparation": {
            "scope": "native _density_factors certifies actual SCF D, occupied-orbital candidate only",
            "seconds": factor_seconds,
            "candidate_rows": int(occupied.size),
            "actual_rank_rows": int(factors.shape[0]),
            "factor_logical_bytes": logical_array_bytes(factors),
            "vendor_factor_logical_bytes": logical_array_bytes(vendor_factors),
            "certified_factor_bytes": int(certification.factor_bytes),
            "used_eigensolver": bool(certification.used_eigensolver),
            "minimum_eigenvalue": _optional_measured(certification.minimum_eigenvalue),
            "negative_roundoff_magnitude": _optional_measured(
                certification.negative_roundoff_magnitude
            ),
            "minimum_eigenvalue_lower_bound": float(
                certification.minimum_eigenvalue_lower_bound
            ),
            "reconstruction_residual": float(certification.reconstruction_residual),
            "tolerance": float(certification.tolerance),
            "unmeasured_eigenvalue_fields_are_null": True,
        },
        "scientific_checks": {"all_first_and_warm_samples_qualified": True},
        "kernels": results,
    }


__all__ = ["measure_cuest_kernels"]
