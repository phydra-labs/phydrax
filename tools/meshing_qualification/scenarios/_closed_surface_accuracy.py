"""Independent host P1 energy-error measurement on the certified source cells."""

from __future__ import annotations

import numpy as np

from phydrax.discretization.fem._generic import FiniteElementDiscretization
from phydrax.geometry._brep_capped_material import BRepCappedMaterialMap
from phydrax.geometry._brep_sphere_material import BRepSphereMaterialMap
from phydrax.geometry._brep_torus_material import BRepTorusMaterialMap


_MaterialMap = BRepSphereMaterialMap | BRepTorusMaterialMap | BRepCappedMaterialMap


def _energy_integrals(
    space: FiniteElementDiscretization,
    material: _MaterialMap,
    geometry: str,
    values: np.ndarray,
    nodal: np.ndarray,
    order: int,
) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = (nodes + 1.0) / 2.0, weights / 2.0
    points = np.array([(s, (1 - s) * t) for s in nodes for t in nodes])
    qw = np.array(
        [
            weights[i] * weights[j] * (1 - s)
            for i, s in enumerate(nodes)
            for j in range(order)
        ]
    )
    gradient = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    sums = np.zeros(2)
    for block_index, (block, routes, transforms) in enumerate(
        zip(
            space.mesh.blocks,
            space.dof_maps[0].cell_dofs,
            space.dof_maps[0].cell_transforms,
            strict=True,
        )
    ):
        reference_nodes = np.asarray(space.elements[0][block_index].reference_nodes)
        if not np.array_equal(
            reference_nodes, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        ):
            raise ValueError("Closed oracle requires canonical P1 reference-node order.")
        route_array = np.asarray(routes, dtype=np.int64)
        transform_array = np.asarray(transforms)
        if route_array.shape != (block.cell_count, 3) or transform_array.shape != (
            block.cell_count,
            3,
            3,
        ):
            raise ValueError("Closed oracle requires scalar triangular P1 cell routes.")
        local = np.einsum("cij,cj->ci", transform_array, values[route_array])
        interp = np.einsum("cij,cj->ci", transform_array, nodal[route_array])
        side = np.zeros(block.cell_count, dtype=bool)
        if geometry == "capped_cylinder":
            if not isinstance(material, BRepCappedMaterialMap):
                raise ValueError("Capped cylinder requires a capped material map.")
            for chart in material.blocks[block_index].sides:
                side[np.asarray(chart.indices, dtype=np.int64)] = True
        for start in range(0, len(points), 4):
            p = points[start : start + 4]
            physical, jacobian = material.evaluate(block_index, p)
            x, jac = np.asarray(physical), np.asarray(jacobian)
            metric = np.einsum("cqdi,cqdj->cqij", jac, jac)
            det = np.linalg.det(metric)
            if not np.all(np.isfinite(det)) or np.any(det <= 0):
                raise ValueError("Closed oracle encountered a singular source metric.")
            inverse = np.linalg.inv(metric)
            phi = np.column_stack((1 - p.sum(axis=1), p))
            if geometry == "sphere":
                radius = np.linalg.norm(x, axis=-1)
                normal = x / radius[..., None]
                exact = normal[..., 0]
                grad = (np.array([1.0, 0.0, 0.0]) - exact[..., None] * normal) / radius[
                    ..., None
                ]
            elif geometry == "torus":
                rho = np.linalg.norm(x[..., :2], axis=-1)
                normal = (
                    np.concatenate(
                        (
                            (rho - 2.0)[..., None] * x[..., :2] / rho[..., None],
                            x[..., 2:3],
                        ),
                        axis=-1,
                    )
                    / 0.6
                )
                exact = x[..., 2]
                grad = np.array([0.0, 0.0, 1.0]) - normal[..., 2:3] * normal
            elif geometry == "capped_cylinder":
                z = x[..., 2]
                exact = z**2 * (1.5 - z) ** 2
                derivative = 2 * z * (1.5 - z) * (1.5 - 2 * z)
                grad = np.zeros_like(x)
                grad[..., 2] = derivative * side[:, None]
            else:
                raise ValueError("Unknown closed manufactured field.")
            exact_ref = np.einsum("cqd,cqdi->cqi", grad, jac)
            for i, coeff in enumerate((local, interp)):
                v = np.einsum("qi,ci->cq", phi, coeff)
                dv = np.einsum("ia,ci->ca", gradient, coeff)[:, None, :]
                diff = dv - exact_ref
                density = (v - exact) ** 2 + np.einsum(
                    "cqi,cqij,cqj->cq", diff, inverse, diff
                )
                sums[i] += np.sum(
                    density * np.sqrt(det) * qw[None, start : start + len(p)]
                )
    return np.sqrt(sums)


def closed_surface_accuracy(
    space: FiniteElementDiscretization,
    material: _MaterialMap | None,
    geometry: str,
    values: np.ndarray,
    nodal: np.ndarray,
) -> dict[str, float | bool]:
    """Compare energy error with nodal interpolation and check reference convergence.

    This numerical qualification is conditional on Galerkin and sufficiently small
    assembly quadrature error; it is not a rigorous interval bound.
    """
    if material is None:
        raise ValueError("Closed surface oracle requires a source material map.")
    coarse = _energy_integrals(space, material, geometry, values, nodal, 4)
    fine = _energy_integrals(space, material, geometry, values, nodal, 6)
    precision = float(np.max(np.abs(fine - coarse)))
    scale = max(float(fine[0]), float(fine[1]), np.finfo(float).tiny)
    if not np.all(np.isfinite(fine)) or precision > 1e-4 * scale:
        raise RuntimeError(
            f"Closed surface reference quadrature did not converge: {coarse}, {fine}."
        )
    allowance = 2 * precision
    return {
        "energy_error": float(fine[0]),
        "interpolant_energy_error": float(fine[1]),
        "reference_quadrature_difference": precision,
        "energy_error_allowance": float(allowance),
        "energy_accuracy_passed": bool(fine[0] <= fine[1] + allowance),
    }
