#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Decision benchmark for raw condensed-matter response and instrument work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from _runtime import capture_environment, logical_array_bytes, measure_repeated

from phydrax import units
from phydrax.chemistry.spectroscopy._instrument import (
    apply_spectral_instrument,
    prepare_spectral_instrument,
    SpectralInstrumentKind,
    SpectralInstrumentPlan,
)
from phydrax.chemistry.spectroscopy._response import (
    SpectralResponseEvidence,
    SpectralResponseProduct,
    SpectralResponseRepresentation,
)
from phydrax.chemistry.spectroscopy._scattering import ElasticNeutronScatteringPlan


def run(
    *, grid_points: int, channels: int, atoms: int, q_points: int, repeats: int
) -> dict[str, Any]:
    grid = np.linspace(0.0, 20.0, grid_points)
    values = np.exp(-0.5 * ((grid[None, :] - 10.0) / 0.8) ** 2)
    values = np.broadcast_to(values, (channels, grid_points)).copy()
    response = SpectralResponseProduct(
        grid,
        values,
        np.ones(grid_points, dtype=bool),
        units.ELECTRONVOLT,
        units.ONE,
        tuple(f"channel[{index}]" for index in range(channels)),
        SpectralResponseRepresentation.DENSITY,
        "benchmark-response",
        "benchmark-source",
        SpectralResponseEvidence(0.0, 0.0, 0.0, 0.0, True),
    )
    kernel = np.exp(-0.5 * (np.linspace(-3.0, 3.0, 129) / 0.7) ** 2)
    instrument = prepare_spectral_instrument(
        SpectralInstrumentPlan(
            SpectralInstrumentKind.STATIONARY_KERNEL,
            "benchmark-response",
            channels,
            grid_points,
            kernel_capacity=kernel.size,
            convolution_method="fft",
            area_tolerance=2.0e-2,
        ),
        kernel,
    )
    positions = np.column_stack(
        (np.linspace(0.0, 1.0, atoms, endpoint=False), np.zeros(atoms), np.zeros(atoms))
    )
    positive_q = np.column_stack(
        (
            np.linspace(0.1, 10.0, q_points // 2),
            np.zeros(q_points // 2),
            np.zeros(q_points // 2),
        )
    )
    q = np.concatenate((positive_q, -positive_q), axis=0)
    reverse = np.concatenate(
        (np.arange(q_points // 2, q_points), np.arange(q_points // 2))
    )
    scattering = ElasticNeutronScatteringPlan(
        positions,
        np.zeros((atoms, 3, 3)),
        reverse,
        "benchmark-cell",
        q_capacity=q_points,
        atom_capacity=atoms,
        symmetry_tolerance=1.0e-10,
    )

    instrument_result, instrument_timing = measure_repeated(
        lambda: apply_spectral_instrument(instrument, response), warmup=1, repeats=repeats
    )
    scattering_result, scattering_timing = measure_repeated(
        lambda: scattering.evaluate(
            q, np.ones(atoms), "benchmark-lengths", ("sha256:benchmark",)
        ),
        warmup=1,
        repeats=repeats,
    )
    return {
        "environment": capture_environment().to_dict(),
        "axes": {
            "grid_points": grid_points,
            "channels": channels,
            "atoms": atoms,
            "q_points": q_points,
        },
        "raw_response_bytes": logical_array_bytes(response),
        "instrument": {
            "timing": instrument_timing.to_milliseconds_dict(),
            "output_bytes": logical_array_bytes(instrument_result),
            "area_residual": float(instrument_result.evidence.area_residual),
            "successful": bool(instrument_result.evidence.successful),
        },
        "elastic_scattering": {
            "timing": scattering_timing.to_milliseconds_dict(),
            "output_bytes": logical_array_bytes(scattering_result),
            "friedel_residual": float(scattering_result.evidence.friedel_residual),
            "successful": bool(scattering_result.evidence.successful),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-points", type=int, default=4096)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--q-points", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.q_points <= 0 or arguments.q_points % 2:
        raise ValueError("--q-points must be a positive even integer.")
    if (
        min(arguments.grid_points, arguments.channels, arguments.atoms, arguments.repeats)
        <= 0
    ):
        raise ValueError("Benchmark sizes and repeats must be positive.")
    report = run(
        grid_points=arguments.grid_points,
        channels=arguments.channels,
        atoms=arguments.atoms,
        q_points=arguments.q_points,
        repeats=arguments.repeats,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
