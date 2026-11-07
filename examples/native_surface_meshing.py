#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native sphere meshing -> embedded H1 -> zero-mean Laplace-Beltrami solve.

The manufactured spherical harmonic u(x)=x_0 has -Delta_S u=2u on the unit
sphere. Two independently generated native meshes demonstrate convergence;
no OCCT/Gmsh, ambient-volume surrogate, reaction regularization or area-only
substitute is used. The scalar FE gradient is J (J.T J)^-1 grad_ref and the
surface density is sqrt(det(J.T J)). An exact linear constraint removes the
constant nullspace by prescribing zero FE surface mean.

Set ``PHYDRAX_MESHCORE_LIBRARY`` to the ABI-matched native library and run
``python examples/native_surface_meshing.py`` from the repository root.
"""

from __future__ import annotations

import json
import os

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.geometry import MeshingDomain
from phydrax.geometry.brep import brep_sphere
from phydrax.linalg import ArraySpace, ConstraintMap
from phydrax.sparse import EdgeRelation, SparseCoordinateOperator


def sphere() -> MeshingDomain:
    model = brep_sphere(
        1.0,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
        source_id="example:native-sphere",
    )
    return MeshingDomain.from_brep(model)


def generate(size: float) -> phx.meshing.CellMeshingResult:
    domain = sphere()
    scope = phx.meshing.MeshingScope.whole(domain, 2)
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, size, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        maximum_deviation=0.05,
    )
    result = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("parametric_surface")
        )
        .plan(
            domain,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    result.audit.require_passed()
    certification = result.certification
    charts = result.surface_source
    if certification is None or certification.fidelity is None or charts is None:
        raise RuntimeError("The native sphere requires source/global certificates.")
    certification.require_passed()
    witness = certification.inputs.fidelity_witness
    if witness is None or witness.witness_id != charts.chart_witness.witness_id:
        raise RuntimeError("The fidelity witness must match the retained source charts.")
    if not result.compliance.passed or result.provider.name != "phydrax-native":
        raise RuntimeError("The native sphere did not satisfy its meshing request.")
    return result


def zero_mean_constraint(
    space: phx.discretization.FiniteElementDiscretization,
) -> tuple[ConstraintMap, np.ndarray]:
    full = space.field_spaces[0].vector_space
    if not isinstance(full, ArraySpace):
        raise TypeError("Scalar surface H1 requires an ArraySpace.")
    weights = np.zeros(full.size, dtype=np.float64)
    for index, geometry in enumerate(
        space.evaluate_geometry("u", space.default_runtime.coordinates)
    ):
        local = np.asarray(
            phx.ein.contract(
                "ql,cq->cl",
                np.asarray(geometry.basis_values),
                np.asarray(geometry.physical_weights),
            )
        )
        routes = np.asarray(space.dof_maps[0].cell_dofs[index])
        np.add.at(weights, routes.reshape(-1), local.reshape(-1))
    pivot = int(np.argmax(weights))
    if not weights[pivot] > 0.0:
        raise RuntimeError(
            "Surface mean constraint requires positive integrated basis mass."
        )
    kept = np.flatnonzero(np.arange(weights.size) != pivot)
    reduced = ArraySpace((kept.size,), dtype=jnp.float64)
    columns = np.tile(np.arange(kept.size, dtype=np.int32), 2)
    rows = np.concatenate((kept, np.full(kept.size, pivot, dtype=np.int64)))
    relation = EdgeRelation(
        columns, rows, source_size=kept.size, target_size=weights.size
    )
    data = np.concatenate(
        (np.ones(kept.size, dtype=np.float64), -weights[kept] / weights[pivot])
    )
    identity = (
        f"sphere-zero-mean:{space.mesh.geometry_id}:"
        f"{phx.array_tree_fingerprint(weights)['sha256']}"
    )
    prolongation = SparseCoordinateOperator(
        relation, data, source=reduced, target=full, operator_id=identity
    )
    return phx.discretization.affine_dof_constraint(
        space, "u", prolongation, constraint_id=identity
    ), weights


def sphere_error(
    space: phx.discretization.FiniteElementDiscretization, values: Array
) -> float:
    """Independent radial-lift L2 norm on the exact sphere, not facet area error."""
    rule = phx.integration.ReferenceTriangleRule(phx.integration.GaussLegendreRule(5))
    data = phx.integration.reference_rule_data(rule)
    total = 0.0
    coordinates = np.asarray(space.mesh.coordinates)
    values_host = np.asarray(values)
    for index, block in enumerate(space.mesh.blocks):
        geometry = space.evaluate_block_geometry(
            "u", index, space.default_runtime.coordinates, data.points, data.weights
        )
        routes = np.asarray(space.dof_maps[0].cell_dofs[index])
        approximate = np.asarray(
            phx.ein.contract(
                "ql,cl->cq",
                np.asarray(geometry.basis_values),
                values_host[routes],
            )
        )
        points = np.asarray(geometry.physical_points)
        radius = np.linalg.norm(points, axis=-1)
        exact = points[..., 0] / radius
        corners = coordinates[np.asarray(block.vertices)]
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normal /= np.linalg.norm(normal, axis=-1)[:, None]
        density = (
            np.abs(np.asarray(phx.ein.contract("cd,cqd->cq", normal, points))) / radius**3
        )
        total += float(
            np.sum(
                np.asarray(geometry.physical_weights)
                * density
                * (approximate - exact) ** 2
            )
        )
    return float(np.sqrt(total))


def solve(result: phx.meshing.CellMeshingResult) -> tuple[float, float]:
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    constraint, weights = zero_mean_constraint(space)

    def forcing(points: Array, context: object) -> Array:
        del context
        return 2.0 * points[..., 0] / jnp.linalg.norm(points, axis=-1)

    form = phx.equations.FiniteElementForm(
        "sphere-laplace-beltrami",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    forcing, coefficient_id="unit-sphere-degree-one-harmonic"
                ),
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form, space, constraint=constraint
    )
    operator, rhs = problem.linear_system()
    solved = phx.linalg.solve(operator, rhs)
    if not bool(jnp.all(solved.successful)):
        raise RuntimeError(
            "The constrained native Laplace-Beltrami solve did not converge."
        )
    values = problem.expand(solved.value)
    mean = float(weights @ np.asarray(values) / np.sum(weights))
    if abs(mean) > 1e-10:
        raise RuntimeError(
            "The Laplace-Beltrami solution violates its zero-mean constraint."
        )
    return sphere_error(space, values), mean


def main() -> None:
    library = os.environ.get("PHYDRAX_MESHCORE_LIBRARY")
    if not library:
        raise RuntimeError("Set PHYDRAX_MESHCORE_LIBRARY to the native meshcore library.")
    phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
    reports = []
    errors: list[float] = []
    for size in (0.8, 0.4):
        result = generate(size)
        certification = result.certification
        charts = result.surface_source
        if certification is None or certification.fidelity is None or charts is None:
            raise RuntimeError("Native sphere source certification is mandatory.")
        error, mean = solve(result)
        errors.append(error)
        reports.append(
            {
                "requested_size": size,
                "triangles": sum(block.cell_count for block in result.mesh.blocks),
                "certification_passed": certification.passed,
                "source_id": charts.source_id,
                "source_revision": charts.source_revision,
                "result_id": result.result_id,
                "topology_id": result.mesh.topology_id,
                "coordinate_dtype": str(np.asarray(result.mesh.coordinates).dtype),
                "chart_witness_id": charts.chart_witness.witness_id,
                "fidelity_certificate_id": certification.fidelity.certificate_id,
                "provider": result.provider.name,
                "backend_version": result.runtime.actual_version,
                "sphere_l2_error": error,
                "surface_mean": mean,
            }
        )
    if not errors[1] < errors[0]:
        raise RuntimeError(
            "Native sphere refinement did not improve the manufactured PDE error."
        )
    print(
        json.dumps(
            {
                "native_library": library,
                "pde": "zero-mean Laplace-Beltrami",
                "convergence": reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
