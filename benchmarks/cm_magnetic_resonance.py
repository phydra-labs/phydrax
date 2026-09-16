#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from _runtime import capture_environment, logical_array_bytes, measure_repeated

from phydrax.applications import magnetic_resonance as mr


def _density(state):
    value = jnp.asarray(state, dtype=complex)
    return jnp.outer(value, jnp.conj(value))


def _case(site_count: int, sample_count: int, repeats: int) -> dict[str, Any]:
    isotope = mr.ResonanceIsotope(
        "benchmark-spin",
        "nucleus",
        0.5,
        2.0 * math.pi * 100.0,
    )
    system = mr.MagneticResonanceSpinSystem(
        tuple(mr.SpinSite(f"s{index}", isotope) for index in range(site_count)),
        (0.0, 0.0, 1.0),
        resource_policy=mr.MagneticResonanceResourcePolicy(
            maximum_hilbert_dimension=256,
            maximum_density_elements=65_536,
        ),
        system_id=f"benchmark-{site_count}-spin",
    )
    prepared, prepare_times = measure_repeated(
        lambda: mr.prepare_spin_system(system),
        warmup=0,
        repeats=repeats,
    )
    plus_x = jnp.asarray([1.0, 1.0], dtype=complex) / jnp.sqrt(2.0)
    product = plus_x
    for _ in range(site_count - 1):
        product = jnp.kron(product, plus_x)
    density = _density(product)
    plan = mr.AcquisitionPlan(1.0 / 1024.0, sample_count)
    fid, acquisition_times = measure_repeated(
        lambda: mr.acquire_fid(prepared, density, plan),
        warmup=1,
        repeats=repeats,
    )
    spectrum, fft_times = measure_repeated(
        lambda: mr.fid_spectrum(fid),
        warmup=1,
        repeats=repeats,
    )
    return {
        "axes": {
            "site_count": site_count,
            "hilbert_dimension": prepared.layout.dimension,
            "density_elements": prepared.layout.dimension**2,
            "sample_count": sample_count,
        },
        "timing": {
            "prepare": prepare_times.to_seconds_dict(),
            "acquisition": acquisition_times.to_seconds_dict(),
            "fft": fft_times.to_seconds_dict(),
        },
        "logical_bytes": {
            "prepared": logical_array_bytes(prepared),
            "initial_density": logical_array_bytes(density),
            "fid": logical_array_bytes(fid),
            "spectrum": logical_array_bytes(spectrum),
        },
        "scientific_residual": {
            "hamiltonian_hermiticity": float(prepared.evidence.hermiticity_residual),
            "maximum_trace": float(jnp.max(fid.trace_residuals)),
            "step_unitarity": float(fid.step_unitarity_residual),
        },
        "successful": bool(prepared.evidence.valid & fid.valid & spectrum.finite),
    }


def run(*, repeats: int) -> dict[str, Any]:
    return {
        "environment": capture_environment().to_dict(),
        "profile_id": "magnetic-resonance:nmr:exact-small-single-crystal",
        "cases": [
            _case(site_count, sample_count, repeats)
            for site_count, sample_count in ((1, 256), (2, 512), (4, 1024))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    payload = json.dumps(run(repeats=arguments.repeats), indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
