#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from fractions import Fraction
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
from phydrax.geometry.brep import TSplineSurfacePatch
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


def _tspline_local_refinement() -> dict[str, object]:
    span_count = 8
    breaks = tuple(Fraction(index, span_count) for index in range(span_count + 1))
    source_mesh = iga.TMesh2D.tensor_product(breaks, breaks)
    greville_points = tuple(
        anchor.local_knots.greville_parameter for anchor in source_mesh.anchors
    )
    exact_points = tuple((u, v, u * v) for u, v in greville_points)
    numeric_points = np.asarray(
        [[float(value) for value in row] for row in exact_points], dtype=np.float64
    )
    exact_weights = tuple(
        Fraction(1) + v / 5 for _, v in greville_points
    )
    numeric_weights = np.asarray(
        [float(weight) for weight in exact_weights], dtype=np.float64
    )
    source_patch = TSplineSurfacePatch(
        source_mesh,
        numeric_points,
        numeric_weights,
        exact_control_points=exact_points,
        exact_weights=exact_weights,
    )
    trim = iga.UVTrimmedSurface(
        iga.UVTrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    )
    plan = iga.TSplineIsogeometricPlan(
        iga.TSplineGeometryState(source_patch),
        iga.TSplineH1FieldSpec(
            "u", weights_from_geometry=True, anchor_count=source_mesh.coefficient_count
        ),
        trim=trim,
        quadrature_policy=iga.IsogeometricQuadraturePolicy(5),
    )
    marked = next(
        cell
        for cell in source_mesh.cells
        if cell.parameter_bounds
        == (Fraction(3, 8), Fraction(4, 8), Fraction(3, 8), Fraction(4, 8))
    )
    distant = next(
        cell
        for cell in source_mesh.cells
        if cell.parameter_bounds
        == (Fraction(0), Fraction(1, 8), Fraction(0), Fraction(1, 8))
    )
    refinement = plan.refine((marked.cell_id,))
    target_plan = refinement.target_plan
    target_mesh = target_plan.geometry.mesh
    target_cells = {cell.cell_id: cell for cell in target_mesh.cells}
    tensor_cell_count = (len(target_mesh.axis_breaks[0]) - 1) * (
        len(target_mesh.axis_breaks[1]) - 1
    )
    locality = target_cells[distant.cell_id].parameter_bounds == distant.parameter_bounds
    nonsingular_source = (
        target_plan.exact_geometry_certificate.passed
        and target_plan.geometry_certificate.passed
    )
    local_structure = target_mesh.cell_count < tensor_cell_count and target_mesh.t_junction_count > 0
    target_patch = target_plan.geometry.surface_patch
    shared_source = (
        target_patch.source_mesh.mesh_id == source_mesh.mesh_id
        and target_patch.exact_source_control_points == source_patch.exact_source_control_points
        and target_patch.exact_source_weights == source_patch.exact_source_weights
    )
    target_patch.validate_exact_transfer_chain()
    transfer_chain_valid = (
        len(target_patch.transfer_meshes) == len(target_patch.transfer_plans) + 1
        and target_patch.transfer_meshes[0].mesh_id == source_mesh.mesh_id
        and target_patch.transfer_meshes[-1].mesh_id == target_mesh.mesh_id
    )
    checks = {
        "target_basis_certificate": target_plan.geometry.surface_patch.basis_certificate.passed,
        "exact_and_numeric_physical_maps": nonsingular_source,
        "local_non_tensor_structure": local_structure,
        "distant_cell_unchanged": locality,
        "source_definition_retained": shared_source,
        "exact_transfer_chain": transfer_chain_valid,
    }
    return {
        "passed": all(checks.values()),
        "source_mesh_id": source_mesh.mesh_id,
        "target_mesh_id": target_mesh.mesh_id,
        "target_basis_certificate_id": target_plan.geometry.surface_patch.basis_certificate.certificate_id,
        "exact_geometry_certificate_id": target_plan.exact_geometry_certificate.certificate_id,
        "numeric_geometry_certificate_id": target_plan.geometry_certificate.certificate_id,
        "marked_cell_ids": list(refinement.basis_refinement.marked_cell_ids),
        "closure_cell_ids": list(refinement.basis_refinement.closure_cell_ids),
        "source_cells": source_mesh.cell_count,
        "target_cells": target_mesh.cell_count,
        "cartesian_cells_at_target_breaks": tensor_cell_count,
        "target_t_junctions": target_mesh.t_junction_count,
        "transfer_route_count": refinement.basis_refinement.transfer_plan.route_count,
        "transfer_mesh_ids": [mesh.mesh_id for mesh in target_patch.transfer_meshes],
        "checks": checks,
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
        "tspline_local_refinement": _tspline_local_refinement(),
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
