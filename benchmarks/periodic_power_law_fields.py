#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np
from jax import Array

from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_repeated,
    measure_synchronized,
)
from phydrax.discretization import (
    AxisDomain,
    FourierBasisPlan,
    PreparedRetainedBandRestriction,
    RetainedBandRestrictionPlan,
    TensorSpectralDiscretization,
    TensorSpectralPlan,
)
from phydrax.stochastic import PeriodicPowerLawFieldPlan, PreparedPeriodicPowerLawField


def _prepare(
    resolution: int, chunk: int, /
) -> tuple[PreparedPeriodicPowerLawField, TensorSpectralDiscretization]:
    domain = (AxisDomain.periodic(0.0, 2 * np.pi),) * 2
    space = TensorSpectralPlan((FourierBasisPlan(resolution),) * 2).prepare(domain)
    prepared = PeriodicPowerLawFieldPlan.fractional_gaussian(
        0.5, spatial_dimension=2, component_shape=(2,), solenoidal=True
    ).prepare(space, member_chunk_size=chunk)
    target = TensorSpectralPlan((FourierBasisPlan(max(3, resolution // 2)),) * 2).prepare(
        domain
    )
    return prepared, target


@eqx.filter_jit
def _apply_restriction(
    restriction: PreparedRetainedBandRestriction, coefficients: Array, /
) -> Array:
    return jax.vmap(restriction)(coefficients)


def benchmark_case(
    resolution: int, members: int, chunk: int, repeats: int
) -> dict[str, Any]:
    (prepared, target), preparation_seconds = measure_synchronized(
        lambda: _prepare(resolution, chunk)
    )
    key = jax.random.key(710)
    indices = np.arange(members, dtype=np.uint32)
    cold_sample, sampling_first_seconds = measure_synchronized(
        lambda: prepared.sample(key, indices)
    )
    warm_sample, sampling_distribution = measure_repeated(
        lambda: prepared.sample(key, indices), warmup=1, repeats=repeats
    )
    np.testing.assert_array_equal(cold_sample.coefficients, warm_sample.coefficients)
    cold_restriction, restriction_first_seconds = measure_synchronized(
        lambda: prepared.restrict(warm_sample, target)
    )
    warm_restriction, restriction_distribution = measure_repeated(
        lambda: prepared.restrict(warm_sample, target), warmup=1, repeats=repeats
    )
    np.testing.assert_array_equal(
        cold_restriction.coefficients, warm_restriction.coefficients
    )

    # Official compiler memory is available for the public prepared restriction
    # callable. It does not include the host realization/evidence construction.
    restriction = RetainedBandRestrictionPlan(prepared.discretization, target).prepare()
    # Equinox's decorator stubs omit its documented lowering method.
    executable = _apply_restriction.lower(  # ty: ignore[unresolved-attribute]
        restriction, warm_sample.coefficients
    ).compile()
    restriction_evidence = compiler_evidence(
        executable.compiled.cost_analysis(),
        executable.compiled.memory_analysis(),
        source="public-prepared-retained-band-restriction",
    )
    sampling_evidence = compiler_evidence(
        None,
        None,
        source="public-periodic-power-law-sample",
        unavailable_reason="Public sample has host admission and key identity work; no public lowering route.",
    )

    # Independent signed-mode gather, including every target Nyquist plane.
    source = np.asarray(warm_sample.coefficients)
    expected = np.zeros((members, *target.modal_shape, 2), dtype=source.dtype)
    for index in itertools.product(*(range(n) for n in target.modal_shape)):
        if any(n % 2 == 0 and i == n // 2 for i, n in zip(index, target.modal_shape)):
            continue
        signed = tuple(
            i if i <= (n - 1) // 2 else i - n for i, n in zip(index, target.modal_shape)
        )
        source_index = tuple(k % resolution for k in signed)
        expected[(slice(None), *index)] = source[(slice(None), *source_index)]
    gather_matches = np.array_equal(warm_restriction.coefficients, expected)

    # Leray rank is d-1=1 in 2D. Normal latent quadratic energy has variance
    # <= 2*E[energy]^2; five standard errors is a conservative finite-sample gate.
    modes = np.fft.fftfreq(resolution) * resolution
    kx, ky = np.meshgrid(modes, modes, indexing="ij")
    squared = kx**2 + ky**2
    active = squared > 0
    if resolution % 2 == 0:
        active[resolution // 2, :] = False
        active[:, resolution // 2] = False
    expected_energy = float((2 * np.pi) ** 2 * np.sum(squared[active] ** -1.5))
    observed_energy = float(np.mean(np.sum(np.abs(source) ** 2, axis=(1, 2, 3))))
    energy_tolerance = 5 * expected_energy * np.sqrt(2 / members)
    energy_matches = bool(abs(observed_energy - expected_energy) <= energy_tolerance)
    passed = bool(
        np.all(np.isfinite(source))
        and gather_matches
        and energy_matches
        and np.all(
            np.asarray(warm_restriction.removed_coefficient_energy)
            >= np.asarray(warm_sample.removed_coefficient_energy)
        )
    )
    return {
        "axes": {
            "resolution": resolution,
            "members": members,
            "chunk": chunk,
            "repeats": repeats,
        },
        "ids": {
            "plan": prepared.plan.plan_id,
            "field": prepared.field_id,
            "prepared": prepared.prepared_id,
        },
        "host_preparation_seconds": preparation_seconds,
        "logical_retained_preparation_bytes": logical_array_bytes((prepared, target)),
        "phases": [
            {
                "operation": "public_sample",
                "first_call_including_compile_seconds": sampling_first_seconds,
                "warm": sampling_distribution.to_seconds_dict(),
                "compiler": asdict(sampling_evidence),
                "logical_retained_output_bytes": logical_array_bytes(warm_sample),
            },
            {
                "operation": "public_restrict",
                "first_call_including_compile_seconds": restriction_first_seconds,
                "warm": restriction_distribution.to_seconds_dict(),
                "compiler": asdict(restriction_evidence),
                "logical_retained_output_bytes": logical_array_bytes(warm_restriction),
            },
        ],
        "numerical_check": {
            "signed_mode_gather_matches": gather_matches,
            "expected_modal_energy": expected_energy,
            "observed_modal_energy": observed_energy,
            "five_sigma_energy_tolerance": energy_tolerance,
            "energy_matches": energy_matches,
        },
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    if (
        arguments.resolution < 4
        or min(arguments.members, arguments.chunk, arguments.repeats) < 1
    ):
        raise ValueError("Require resolution >= 4, members, chunk, and repeats >= 1.")
    case = benchmark_case(
        arguments.resolution, arguments.members, arguments.chunk, arguments.repeats
    )
    root = Path(__file__).resolve().parent.parent
    identity = capture_benchmark_identity(root, Path(__file__), tuple(case))
    print(
        json.dumps(
            {
                "benchmark_identity": identity.to_dict(),
                "environment": capture_environment().to_dict(),
                "cases": [case],
                "passed": case["passed"],
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    if not case["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
