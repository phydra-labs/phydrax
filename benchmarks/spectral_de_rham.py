#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded spectral complex campaign: python -m benchmarks.spectral_de_rham."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.discretization import AxisDomain, FourierBasisPlan, TensorSpectralPlan
from phydrax.discretization.spectral import (
    FourierDeRhamComplex,
    SphericalDeRhamComplex,
    SphericalSpectralPlan,
)

from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)


def _fourier(capacity: int) -> FourierDeRhamComplex:
    space = TensorSpectralPlan(
        (FourierBasisPlan(capacity), FourierBasisPlan(capacity)), axis_names=("x", "y")
    ).prepare(
        (AxisDomain.periodic(0.0, 2.0 * np.pi), AxisDomain.periodic(0.0, 2.0 * np.pi))
    )
    return FourierDeRhamComplex(space, nyquist_policy="zero-self-conjugate")


def _sphere(capacity: int) -> SphericalDeRhamComplex:
    return SphericalDeRhamComplex(
        SphericalSpectralPlan(capacity, sampling="gl").prepare(radius=1.7)
    )


def _measure(
    realization: FourierDeRhamComplex | SphericalDeRhamComplex,
    preparation: float,
    capacity: int,
    kind: str,
) -> dict[str, Any]:
    space = realization.hilbert_complex().space(1)
    value = jnp.sin(0.13 * jnp.arange(space.size, dtype=jnp.float64)).astype(
        space.structure().dtype
    )

    def operation(state: Array) -> Array:
        decomposition = realization.hodge_decomposition(1, state)
        laplacian = realization.hodge_laplacian(1, state)
        return jnp.stack(
            (
                decomposition.exact,
                decomposition.coexact,
                decomposition.harmonic,
                laplacian,
            )
        )

    prepared = jax.jit(operation)
    compiled, timing = measure_lower_and_compile(
        lambda: prepared.lower(value), lambda lowered: lowered.compile()
    )
    cold, cold_seconds = measure_synchronized(lambda: compiled(value))
    warm, distribution = measure_repeated(lambda: compiled(value), warmup=1, repeats=5)
    np.testing.assert_allclose(
        np.sum(np.asarray(warm[:3]), axis=0), np.asarray(value), atol=2e-11
    )
    np.testing.assert_allclose(np.asarray(cold), np.asarray(warm), atol=0.0)
    if isinstance(realization, FourierDeRhamComplex):
        numbers = np.fft.fftfreq(capacity) * capacity
        wave = np.stack(np.meshgrid(numbers, numbers, indexing="ij"), axis=-1).reshape(
            (-1, 2)
        )
        active = np.asarray(realization.active_modes)
        eigenvalues = np.repeat(np.sum(wave[active] ** 2, axis=-1), 2)
    elif isinstance(realization, SphericalDeRhamComplex):
        ell = np.arange(1, capacity)
        eigenvalues = np.repeat(np.repeat(ell * (ell + 1) / 1.7**2, 2 * ell + 1), 2)
    else:
        raise TypeError("This campaign measures Fourier and spherical realizations only.")
    np.testing.assert_allclose(
        np.asarray(warm[3]), eigenvalues * np.asarray(value), atol=2e-10
    )
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-spectral-complex",
    )
    return {
        "kind": kind,
        "capacity": capacity,
        "coordinate_counts": realization.cell_counts,
        "preparation_seconds": preparation,
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "cold_seconds": cold_seconds,
        "warm": distribution.to_seconds_dict(),
        "compiler": asdict(evidence),
        "retained_preparation_bytes": logical_array_bytes(realization),
        "analytic_laplacian_defect": float(
            np.max(np.abs(np.asarray(warm[3]) - eigenvalues * np.asarray(value)))
        ),
    }


def main() -> None:
    rows = []
    for capacity in (4, 8, 16):
        realization, preparation = measure_synchronized(lambda: _fourier(capacity))
        rows.append(_measure(realization, preparation, capacity, "fourier"))
    for capacity in (3, 5, 8):
        sphere, preparation = measure_synchronized(lambda: _sphere(capacity))
        rows.append(_measure(sphere, preparation, capacity, "sphere"))
    root = Path(__file__).resolve().parent.parent
    identity = capture_benchmark_identity(root, Path(__file__), tuple(rows[0]))
    print(
        json.dumps(
            {
                "benchmark_identity": identity.to_dict(),
                "environment": capture_environment().to_dict(),
                "cases": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
