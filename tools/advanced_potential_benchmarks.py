#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def run_advanced_potential_benchmarks(
    *,
    panels_per_chart: int = 4,
    quadrature_order: int = 6,
) -> dict[str, Any]:
    """Run deterministic holomorphic and boundary-layer correctness workloads."""

    polynomial = phx.equations.HolomorphicPolynomialPotential(1, 2)
    polynomial = eqx.tree_at(
        lambda value: value.coefficient_real,
        polynomial,
        jnp.asarray([[0.0, 0.0, 1.0]]),
    )
    harmonic = phx.equations.HarmonicPotential2D(polynomial)
    point = jnp.asarray([0.2, -0.1])
    started = time.perf_counter()
    harmonic_residual = jnp.trace(jax.hessian(harmonic)(point))
    jax.block_until_ready(harmonic_residual)
    holomorphic_ms = 1e3 * (time.perf_counter() - started)

    geometry = phx.geometry.Circle((0.0, 0.0), 1.0).compile()
    panelization = phx.operators.BoundaryPanelization2D(
        geometry.boundary_atlas,
        panels_per_chart=int(panels_per_chart),
        quadrature_order=int(quadrature_order),
        geometry=geometry,
    )
    started = time.perf_counter()
    result = phx.solver.solve_interior_laplace_dirichlet_2d(
        panelization,
        jnp.ones((panelization.node_count,)),
    )
    center_value = result.potential(jnp.asarray([0.0, 0.0]))
    jax.block_until_ready(center_value)
    boundary_ms = 1e3 * (time.perf_counter() - started)
    center_error = jnp.abs(center_value - 1.0)
    harmonic_certificate = phx.equations.trial_space_certificate(
        phx.domain.HyperRectangle((-1.0, -1.0), (1.0, 1.0)).Model("x")(
            harmonic
        )
    )
    layer_certificate = phx.equations.trial_space_certificate(
        phx.domain.GeometryDomain(geometry).Model("x")(result.potential)
    )
    passed = bool(
        jnp.abs(harmonic_residual) <= 1e-11
        and result.valid
        and result.boundary_residual_norm <= 1e-9
        and center_error <= 5e-4
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "holomorphic": {
            "laplace_residual": float(harmonic_residual),
            "wall_ms": holomorphic_ms,
            "certificate_id": harmonic_certificate.certificate_id,
        },
        "boundary_layer": {
            "node_count": panelization.node_count,
            "boundary_residual": float(result.boundary_residual_norm),
            "center_error": float(center_error),
            "wall_ms": boundary_ms,
            "pde_exactness": layer_certificate.exactness,
            "validity_region": layer_certificate.validity_region,
            "approximation_id": result.approximation.approximation_id,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark holomorphic and boundary layer potentials."
    )
    parser.add_argument("--panels-per-chart", type=int, default=4)
    parser.add_argument("--quadrature-order", type=int, default=6)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_advanced_potential_benchmarks(
                panels_per_chart=arguments.panels_per_chart,
                quadrature_order=arguments.quadrature_order,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
