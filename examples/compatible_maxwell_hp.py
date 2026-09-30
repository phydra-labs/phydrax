#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import jax.numpy as jnp

import phydrax as phx


def run() -> dict[str, tuple[int, int] | float]:
    mesh = phx.discretization.CellMesh(
        jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (1.0, 1.0, 1.0),
                (0.0, 1.0, 1.0),
            ),
            dtype=jnp.float64,
        ),
        (
            phx.discretization.CellBlock(
                "hexahedra",
                "hexahedron",
                jnp.arange(8, dtype=jnp.int32)[None, :],
            ),
        ),
    )
    complex_ = phx.discretization.fem.FiniteElementDeRhamComplex(
        mesh,
        family="tensor-trimmed",
        order=4,
        twist="untwisted",
    )
    counts = complex_.cell_counts
    scalar = jnp.arange(counts[0], dtype=jnp.float64)
    circulation = jnp.arange(counts[1], dtype=jnp.float64)
    dd_scalar = complex_.exterior_derivative(1, complex_.exterior_derivative(0, scalar))
    dd_circulation = complex_.exterior_derivative(
        2, complex_.exterior_derivative(1, circulation)
    )
    return {
        "gradient_shape": (counts[1], counts[0]),
        "curl_shape": (counts[2], counts[1]),
        "divergence_shape": (counts[3], counts[2]),
        "curl_gradient_defect": float(jnp.max(jnp.abs(dd_scalar), initial=0.0)),
        "divergence_curl_defect": float(jnp.max(jnp.abs(dd_circulation), initial=0.0)),
    }


if __name__ == "__main__":
    print(run())
