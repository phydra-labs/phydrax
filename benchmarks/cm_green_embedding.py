#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
from _runtime import capture_environment, measure_repeated

from phydrax.operators.quantum._impurity import AndersonBath, ImpurityEnvironment
from phydrax.solver._impurity import ImpuritySolveRequest, solve_all_sector_ed_impurity


def _case(frequency_count: int, bath_sites: int, repeats: int) -> dict:
    labels = jnp.arange(-frequency_count // 2, frequency_count // 2)
    energies = jnp.linspace(-1.0, 1.0, bath_sites) if bath_sites else jnp.asarray([])
    couplings = jnp.full((bath_sites,), 0.4 / max(bath_sites, 1) ** 0.5)
    request = ImpuritySolveRequest(
        0.0,
        2.0,
        1.0,
        8.0,
        labels,
        ImpurityEnvironment(bath=AndersonBath(energies, couplings)),
    )
    result, timing = measure_repeated(
        lambda: solve_all_sector_ed_impurity(request), warmup=1, repeats=repeats
    )
    jax.block_until_ready(result.green.values)
    return {
        "frequency_count": frequency_count,
        "bath_sites": bath_sites,
        "sector_count": result.evidence.sector_count,
        "total_state_count": result.evidence.total_state_count,
        "steady": timing.to_dict(),
        "causality_residual": float(result.evidence.causality_residual),
        "moment_residual": float(result.evidence.moment_residual),
        "dyson_residual": float(result.evidence.dyson_residual),
        "density_residual": float(result.evidence.density_residual),
        "valid": bool(result.evidence.valid),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--frequencies", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--bath-sites", type=int, nargs="+", default=[0, 1, 3])
    arguments = parser.parse_args()
    report = {
        "environment": capture_environment().to_dict(),
        "cases": [
            _case(frequencies, bath, arguments.repeats)
            for frequencies in arguments.frequencies
            for bath in arguments.bath_sites
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
