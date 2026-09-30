#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.discretization import iga
from phydrax.discretization.iga._certificate import (
    CertificateDisposition,
    certify_tensor_nurbs,
)
from phydrax.discretization.iga._volume import TensorNURBSVolume
from phydrax.linalg import MaterializationPolicy, materialize


def _geometry_certificate() -> dict[str, object]:
    grid = iga.BSplineGrid.open_uniform(2, 1)
    coordinates = grid.greville_abscissae
    xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    geometry = iga.NURBSGeometryState(
        jnp.stack((xx, yy), axis=-1),
        jnp.ones((grid.coefficient_count, grid.coefficient_count)),
    )
    plan = iga.IsogeometricPlan.isoparametric(
        (grid, grid),
        geometry,
        quadrature_policy=iga.IsogeometricQuadraturePolicy(3),
    )
    certificate = certify_tensor_nurbs(
        TensorNURBSVolume("unit-square", plan.basis, geometry)
    )
    return {
        "passed": certificate.disposition is CertificateDisposition.PASS,
        "certificate_id": certificate.certificate_id,
        "cell_count": len(certificate.cells),
        "diagnostics": [value.code for value in certificate.diagnostics],
    }


def _compatible_complex() -> dict[str, object]:
    grid = iga.BSplineGrid.open_uniform(2, 2)
    complex_ = iga.SplineDeRhamComplex((grid, grid))
    defect = float(jnp.max(complex_.d_squared_defects))
    return {
        "passed": defect <= 1.0e-13,
        "complex_id": complex_.complex_id,
        "d_squared_defect": defect,
    }


def _annulus(point: Array) -> Array:
    angle = 2.0 * jnp.pi * point[1]
    return (1.0 + point[0]) * jnp.stack((jnp.cos(angle), jnp.sin(angle)))


def _mapped_annulus() -> dict[str, object]:
    radial = iga.BSplineGrid.open_uniform(2, 2, interval=(0.0, 1.0))
    angular = iga.BSplineGrid.open_uniform(1, 4, interval=(0.0, 1.0))
    complex_ = iga.SplineDeRhamComplex(
        (radial, angular),
        periodic=(False, True),
        geometry=_annulus,
        geometry_id="qualification-annulus",
        quadrature_degree=18,
    )

    def harmonic(point: Array) -> Array:
        return jnp.stack((-point[1], point[0])) / jnp.sum(point * point)

    values = complex_.interpolant(1, harmonic)
    laplacian = jax.jit(lambda value: complex_.hodge_laplacian(1, value))(values)
    harmonic_defect = float(jnp.max(jnp.abs(laplacian)))
    d0, d1 = (
        np.asarray(materialize(operator, MaterializationPolicy()))
        for operator in complex_.hilbert_complex().differentials
    )
    betti_one = (
        complex_.dof_count(1) - np.linalg.matrix_rank(d0) - np.linalg.matrix_rank(d1)
    )
    trace = complex_.trace(0, "upper")
    scalar = jnp.linspace(-0.3, 0.8, complex_.dof_count(0), dtype=jnp.float64)
    left = trace.target.differential(0).mv(trace.maps[0].mv(scalar))
    right = trace.maps[1].mv(complex_.exterior_derivative(0, scalar))
    trace_defect = float(jnp.max(jnp.abs(left - right)))
    return {
        "passed": betti_one == 1 and harmonic_defect < 2e-8 and trace_defect < 1e-12,
        "betti_one": int(betti_one),
        "harmonic_defect": harmonic_defect,
        "trace_commuting_defect": trace_defect,
        "realization_id": complex_.realization_id,
    }


def _prepared_metric_refresh() -> dict[str, object]:
    grid = iga.BSplineGrid.open_uniform(1, 2, interval=(0.0, 1.0))

    def identity(point: Array) -> Array:
        return point

    prepared = iga.SplineDeRhamComplex(
        (grid, grid), geometry=identity, geometry_id="qualification-metric-binding"
    )
    coefficients = jnp.ones((prepared.dof_count(0),), dtype=jnp.float64)

    def energy(scale: Array) -> Array:
        def stretch(point: Array) -> Array:
            return point.at[0].set(scale * point[0])

        refreshed = prepared.refresh_geometry(stretch)
        return coefficients @ refreshed.hodge_star(0, coefficients)

    def step(total: Array, scale: Array) -> tuple[Array, Array]:
        value = energy(scale)
        return total + value, value

    scales = jnp.asarray([1.0, 1.2, 1.7], dtype=jnp.float64)
    _, values = jax.jit(
        lambda factors: jax.lax.scan(step, jnp.asarray(0.0, dtype=jnp.float64), factors)
    )(scales)
    value, gradient = jax.jit(jax.value_and_grad(energy))(
        jnp.asarray(1.7, dtype=jnp.float64)
    )
    value_defect = float(jnp.max(jnp.abs(values - scales)))
    gradient_defect = float(jnp.abs(gradient - 1.0))
    return {
        "passed": value_defect < 2e-11 and gradient_defect < 2e-10,
        "scan_energy_defect": value_defect,
        "geometry_gradient_defect": gradient_defect,
        "refreshed_energy": float(value),
        "realization_id": prepared.realization_id,
    }


def _thb() -> dict[str, object]:
    from phydrax.discretization.iga._thb import THBHierarchy, THBLevel

    hierarchy = THBHierarchy(
        (
            # ty: ignore[invalid-argument-type]
            THBLevel(0, "coarse", (True,), (True, False)),
            # ty: ignore[invalid-argument-type]
            THBLevel(1, "fine", (True, True), (False, True, True)),
        ),
        (np.asarray(((1.0, 0.0), (0.5, 0.5), (0.0, 1.0))),),
    )
    certificate = hierarchy.certify()
    return {
        "passed": certificate.passed,
        "hierarchy_id": hierarchy.hierarchy_id,
        "certificate_id": certificate.certificate_id,
        "partition_defect": certificate.partition_defect,
        "rank": certificate.rank,
        "basis_count": certificate.basis_count,
    }


def run() -> dict[str, object]:
    cases = {
        "geometry_certificate": _geometry_certificate(),
        "compatible_complex": _compatible_complex(),
        "mapped_annulus": _mapped_annulus(),
        "prepared_metric_refresh": _prepared_metric_refresh(),
        "thb_basis": _thb(),
    }
    return {
        "kind": "iga-closure-substrate-qualification",
        "passed": all(bool(value["passed"]) for value in cases.values()),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/iga_closure_qualification.json"),
    )
    arguments = parser.parse_args()
    report = run()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
