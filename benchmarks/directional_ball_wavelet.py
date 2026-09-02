#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from _runtime import (
    capture_environment,
    logical_array_bytes,
    measure_host,
    measure_repeated,
    measure_synchronized,
)

from phydrax.discretization import (
    DirectionalBallWaveletPlan,
    FourierLaguerrePlan,
    RadialLaguerrePlan,
    SphericalHarmonicPlan,
)


def _configuration(
    bandlimit: int,
    radial_bandlimit: int,
    directional_bandlimit: int,
    execution: str,
    /,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    def prepare():
        radial = RadialLaguerrePlan(radial_bandlimit, tau=0.8)
        angular = SphericalHarmonicPlan(bandlimit, reality=False)
        fourier = FourierLaguerrePlan(radial, angular)
        return DirectionalBallWaveletPlan(
            fourier,
            directional_bandlimit=directional_bandlimit,
            wigner_execution=execution,
        )

    plan, preparation_seconds = measure_host(prepare)
    degree = jnp.arange(bandlimit)[:, None]
    order = jnp.arange(-(bandlimit - 1), bandlimit)[None, :]
    valid = jnp.abs(order) <= degree
    modes = (
        jr.normal(jr.key(100 + bandlimit), plan.fourier_laguerre.coefficient_shape)
        + 1j * jr.normal(jr.key(200 + bandlimit), plan.fourier_laguerre.coefficient_shape)
    ) * valid[None, ...]
    values = plan.fourier_laguerre.synthesis(modes)
    analyze = eqx.filter_jit(lambda transform, field: transform.analysis(field))
    synthesize = eqx.filter_jit(
        lambda transform, coefficients: transform.synthesis(coefficients)
    )

    coefficients, analysis_first_seconds = measure_synchronized(
        lambda: analyze(plan, values)
    )
    _, analysis_execution = measure_repeated(
        lambda: analyze(plan, values),
        warmup=warmup,
        repeats=repeats,
    )
    reconstructed, synthesis_first_seconds = measure_synchronized(
        lambda: synthesize(plan, coefficients)
    )
    _, synthesis_execution = measure_repeated(
        lambda: synthesize(plan, coefficients),
        warmup=warmup,
        repeats=repeats,
    )
    roundtrip_error = float(jnp.max(jnp.abs(reconstructed - values)))
    modal_energy = jnp.sum(jnp.abs(modes) ** 2)
    sample_quadrature_energy = jnp.sum(
        plan.fourier_laguerre.radial.quadrature_weights[:, None, None]
        * plan.fourier_laguerre.angular.theta_quadrature_weights[None, :, None]
        * plan.fourier_laguerre.angular.phi_quadrature_weights[None, None, :]
        * jnp.abs(values) ** 2
    )
    sample_quadrature_parseval_relative_residual = float(
        jnp.abs(modal_energy - sample_quadrature_energy)
        / jnp.maximum(modal_energy, jnp.finfo(modal_energy.dtype).tiny)
    )
    if roundtrip_error > 1.0e-10:
        raise RuntimeError("directional ball-wavelet benchmark failed round trip.")
    if plan.admissibility_defect > 1.0e-11:
        raise RuntimeError("directional ball-wavelet benchmark failed admissibility.")

    full_detail_elements = (
        plan.scale_count
        * radial_bandlimit
        * (2 * directional_bandlimit - 1)
        * bandlimit
        * (2 * bandlimit - 1)
    )
    multiresolution_detail_elements = sum(
        shape[0] * shape[1] * shape[2] * shape[3] for shape in plan.detail_shapes
    )
    return {
        "bandlimit": bandlimit,
        "radial_bandlimit": radial_bandlimit,
        "directional_bandlimit": directional_bandlimit,
        "wigner_execution": execution,
        "preparation_seconds": preparation_seconds,
        "analysis_first_jit_seconds": analysis_first_seconds,
        "synthesis_first_jit_seconds": synthesis_first_seconds,
        "analysis_execution": analysis_execution.to_milliseconds_dict(),
        "synthesis_execution": synthesis_execution.to_milliseconds_dict(),
        "persistent_bytes": plan.persistent_bytes,
        "reported_output_bytes": plan.output_bytes(),
        "actual_output_bytes": logical_array_bytes(coefficients),
        "workspace_bytes": plan.workspace_bytes(),
        "estimated_peak_bytes": plan.estimated_peak_bytes(),
        "scale_count": plan.scale_count,
        "full_resolution_detail_elements": full_detail_elements,
        "multiresolution_detail_elements": multiresolution_detail_elements,
        "roundtrip_error": roundtrip_error,
        "sample_quadrature_parseval_relative_residual": (
            sample_quadrature_parseval_relative_residual
        ),
        "admissibility_defect": plan.admissibility_defect,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".json"),
    )
    arguments = parser.parse_args()
    configurations = (
        ((4, 4, 2, "recursive"),)
        if arguments.quick
        else (
            (4, 4, 2, "recursive"),
            (4, 4, 2, "precomputed"),
            (8, 8, 3, "recursive"),
            (16, 16, 3, "recursive"),
        )
    )
    cases = [
        _configuration(
            *configuration,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
        for configuration in configurations
    ]
    radial_256, radial_preparation_seconds = measure_host(
        lambda: RadialLaguerrePlan(256, tau=0.8)
    )
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "radial_certification": {
            "radial_bandlimit": 256,
            "preparation_seconds": radial_preparation_seconds,
            "precompute_bytes": radial_256.precompute_bytes,
            "orthogonality_defect": radial_256.orthogonality_defect,
        },
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    arguments.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
