#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def run():
    triangle = phx.discretization.fem.SimplexModalFamily("triangle", 4)
    prism = phx.discretization.fem.HybridReferenceFamily("prism", 3)
    mortar = phx.discretization.fem.HybridMortarPlan(
        triangle.nodes[:, :1],
        jnp.linspace(0.0, 1.0, 4)[:, None],
        jnp.linspace(0.0, 1.0, 7)[:, None],
        1,
    )
    return {
        "triangle_nodes": triangle.nodes.shape[0],
        "prism_nodes": prism.nodes.shape[0],
        "constant_reproduction_error": float(mortar.reproduction_error),
    }


if __name__ == "__main__":
    print(run())
