#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax.numpy as jnp

from phydrax.operators.quantum._impurity import AndersonBath, ImpurityEnvironment
from phydrax.solver._impurity import ImpuritySolveRequest, solve_all_sector_ed_impurity


def main() -> None:
    interaction = 4.0
    result = solve_all_sector_ed_impurity(
        ImpuritySolveRequest(
            0.0,
            interaction,
            interaction / 2.0,
            8.0,
            jnp.arange(-32, 32),
            ImpurityEnvironment(bath=AndersonBath(jnp.asarray([]), jnp.asarray([]))),
        )
    )
    print(
        json.dumps(
            {
                "valid": bool(result.evidence.valid),
                "density": float(result.density),
                "causality_residual": float(result.evidence.causality_residual),
                "moment_residual": float(result.evidence.moment_residual),
                "dyson_residual": float(result.evidence.dyson_residual),
                "density_residual": float(result.evidence.density_residual),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
