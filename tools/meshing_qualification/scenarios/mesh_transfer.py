from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._meshcore import meshcore_available

from ._resources import measured as _measured


def _contract() -> phx.SpatialCoordinateContract:
    return phx.SpatialCoordinateContract(phx.units.MILLIMETER)


def _quality_summary(report: phx.meshing.CellQualityReport, /) -> dict[str, object]:
    return {
        "minimum_mean_ratio": report.minimum_mean_ratio,
        "maximum_aspect_ratio": report.maximum_aspect_ratio,
        "minimum_scaled_jacobian": report.minimum_scaled_jacobian,
        "minimum_angle": report.minimum_angle,
        "maximum_angle": report.maximum_angle,
        "sampled_invalid_count": report.sampled_invalid_count,
    }


def _result_identity(result: phx.meshing.CellMeshingResult, /) -> dict[str, object]:
    return {
        "result_id": result.result_id,
        "topology_id": result.mesh.topology_id,
        "vertices": result.mesh.coordinates.shape[0],
        "cells": sum(block.cell_count for block in result.mesh.blocks),
        "audit_passed": result.audit.passed,
    }


def _missing_dependency(dependency: str, detail: str, /) -> dict[str, object]:
    """Explicit record of a scenario that cannot run in this environment."""
    return {
        "status": "missing-dependency",
        "dependency": dependency,
        "detail": detail,
        "runtime": _runtime_identity(),
    }


_MESHCORE_DETAIL = "Set PHYDRAX_MESHCORE_LIBRARY or install phydrax[meshcore]."


_HEXAHEDRON_CORNERS = np.asarray(
    (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ),
    dtype=np.int32,
)


def _unit_lattice_cells(
    cells_per_axis: int, dimension: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Unit square/cube lattice points and quadrilateral/hexahedron corners."""
    pass  # Source fixture helpers are owned by this scenario package.

    count = cells_per_axis
    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(*(axis,) * dimension, indexing="ij"), axis=-1).reshape(
        (-1, dimension)
    )
    index = np.arange((count + 1) ** dimension, dtype=np.int32).reshape(
        (count + 1,) * dimension
    )
    offsets = np.unique(_HEXAHEDRON_CORNERS[:, :dimension], axis=0, return_index=True)[1]
    corners = _HEXAHEDRON_CORNERS[np.sort(offsets), :dimension]
    cells = np.stack(
        [
            index[tuple(slice(start, start + count) for start in corner)].reshape((-1,))
            for corner in corners
        ],
        axis=1,
    )
    return points, cells


def _jittered_cube_tetrahedra(
    cells_per_axis: int, seed: int, /
) -> phx.discretization.CellMesh:
    """Kuhn tetrahedra of the unit cube with interior vertices jittered in place."""
    pass  # Source fixture helpers are owned by this scenario package.

    points, tetrahedra = _kuhn_tetrahedra(cells_per_axis)
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    jitter = np.random.default_rng(seed).uniform(-0.1, 0.1, points.shape)
    points[interior] += jitter[interior] / cells_per_axis
    return phx.discretization.CellMesh.from_tetrahedra(points, tetrahedra)


def _supermesh_case(
    source: phx.discretization.CellMesh, target: phx.discretization.CellMesh, /
) -> dict[str, object]:
    """Certify one common refinement and recount its measure and moment ledgers."""
    contract = _contract()
    source_result, source_certify = _measured(
        lambda: phx.meshing.certify_cell_mesh(source, contract)
    )
    target_result, target_certify = _measured(
        lambda: phx.meshing.certify_cell_mesh(target, contract)
    )
    policy = phx.geometry.CommonRefinementPolicy(overlap_simplices=True)
    refinement, refinement_record = _measured(
        lambda: phx.geometry.prepare_common_refinement(source, target, policy=policy)
    )
    # ty: ignore[unresolved-attribute]
    evidence = refinement.evidence
    # ty: ignore[unresolved-attribute]
    if not refinement.succeeded:
        raise RuntimeError(
            # ty: ignore[unresolved-attribute]
            f"Common refinement failed: {refinement.status.name}: {evidence.reason}."
        )
    # ty: ignore[unresolved-attribute]
    dimension = refinement.dimension
    # ty: ignore[unresolved-attribute]
    volumes = np.asarray(refinement.volumes, dtype=np.float64)
    # ty: ignore[unresolved-attribute]
    moments = np.asarray(refinement.first_moments, dtype=np.float64)
    ledgers = {}
    for role, cells, measures, first in (
        (
            "source",
            # ty: ignore[unresolved-attribute]
            refinement.source_cells,
            # ty: ignore[unresolved-attribute]
            refinement.source_measures,
            # ty: ignore[unresolved-attribute]
            refinement.source_first_moments,
        ),
        (
            "target",
            # ty: ignore[unresolved-attribute]
            refinement.target_cells,
            # ty: ignore[unresolved-attribute]
            refinement.target_measures,
            # ty: ignore[unresolved-attribute]
            refinement.target_first_moments,
        ),
    ):
        cells = np.asarray(cells, dtype=np.int64)
        measures = np.asarray(measures, dtype=np.float64)
        first = np.asarray(first, dtype=np.float64)
        covered = np.zeros_like(measures)
        np.add.at(covered, cells, volumes)
        covered_moments = np.zeros_like(first)
        np.add.at(covered_moments, cells, moments)
        total = float(np.sum(measures))
        ledgers[role] = {
            "measure": total,
            "total_measure_relative_defect": abs(float(np.sum(volumes)) - total) / total,
            "maximum_cell_measure_relative_defect": float(
                np.max(np.abs(covered - measures) / measures)
            ),
            # Cell centroid integrals: the overlap first moments of each cell sum
            # to the cell's own first moment.
            "maximum_cell_first_moment_relative_defect": float(
                np.max(np.abs(covered_moments - first)) / np.max(np.abs(first))
            ),
        }
    # The unit square/cube has measure one and centroid (1/2, ..., 1/2).
    domain = {
        "measure_defect": abs(float(np.sum(volumes)) - 1.0),
        "first_moment_defect": float(np.max(np.abs(np.sum(moments, axis=0) - 0.5))),
    }
    # ty: ignore[unresolved-attribute]
    offsets = np.asarray(refinement.simplex_offsets, dtype=np.int64)
    # ty: ignore[unresolved-attribute]
    simplices = np.asarray(refinement.simplices, dtype=np.float64)
    signed = np.linalg.det(simplices[:, 1:] - simplices[:, :1]) / np.prod(
        np.arange(1, dimension + 1)
    )
    partition = np.zeros_like(volumes)
    np.add.at(partition, np.repeat(np.arange(volumes.size), np.diff(offsets)), signed)
    partition_defect = float(np.max(np.abs(partition - volumes) / volumes))
    defects = (
        *(
            value
            for ledger in ledgers.values()
            for key, value in ledger.items()
            if key != "measure"
        ),
        *domain.values(),
        partition_defect,
        evidence.maximum_relative_source_defect,
        evidence.maximum_relative_target_defect,
    )
    if (
        max(defects) > 1.0e-12
        or evidence.source_gap_count
        or evidence.target_gap_count
        or evidence.source_double_count
        or evidence.target_double_count
        or evidence.uncertain_predicate_count
        or evidence.invalid_cell_count
        or evidence.intersection_failure_count
        # ty: ignore[unresolved-attribute]
        or not (source_result.audit.passed and target_result.audit.passed)
    ):
        raise RuntimeError(
            f"Common refinement ledgers failed: {ledgers}, {domain}, "
            f"partition {partition_defect}, evidence {evidence.reason}."
        )
    return {
        # ty: ignore[invalid-argument-type]
        "source": {**_result_identity(source_result), "mesh_id": source.mesh_id},
        # ty: ignore[invalid-argument-type]
        "target": {**_result_identity(target_result), "mesh_id": target.mesh_id},
        "identities": {
            # ty: ignore[unresolved-attribute]
            "refinement_id": refinement.refinement_id,
            "evidence_id": evidence.evidence_id,
            "policy_id": policy.policy_id,
            # ty: ignore[unresolved-attribute]
            "source_topology_id": refinement.source_topology_id,
            # ty: ignore[unresolved-attribute]
            "target_topology_id": refinement.target_topology_id,
        },
        "quality": {
            # ty: ignore[unresolved-attribute]
            "source": _quality_summary(source_result.quality),
            # ty: ignore[unresolved-attribute]
            "target": _quality_summary(target_result.quality),
        },
        "coverage": {
            # ty: ignore[unresolved-attribute]
            "status": refinement.status.name,
            "reason": evidence.reason,
            "maximum_relative_source_defect": evidence.maximum_relative_source_defect,
            "maximum_relative_target_defect": evidence.maximum_relative_target_defect,
            "source_gap_count": evidence.source_gap_count,
            "target_gap_count": evidence.target_gap_count,
            "source_double_count": evidence.source_double_count,
            "target_double_count": evidence.target_double_count,
        },
        "conservation": {
            **ledgers,
            "unit_domain": domain,
            "simplex_partition_maximum_relative_defect": partition_defect,
        },
        "pairs": {
            # ty: ignore[unresolved-attribute]
            "entries": refinement.entry_count,
            "overlap_simplices": simplices.shape[0],
        },
        "predicates": {
            "mode": policy.predicate_mode.value,
            "uncertain": evidence.uncertain_predicate_count,
            "invalid_cells": evidence.invalid_cell_count,
            "intersection_failures": evidence.intersection_failure_count,
        },
        "resources": {
            "source_certify": source_certify,
            "target_certify": target_certify,
            "common_refinement": refinement_record,
        },
    }


def qualify_supermesh() -> dict[str, object]:
    if not meshcore_available():
        return _missing_dependency("phydrax-meshcore", _MESHCORE_DETAIL)
    pass  # Source fixture helpers are owned by this scenario package.

    square_points, quadrilaterals = _unit_lattice_cells(5, 2)
    cube_points, hexahedra = _unit_lattice_cells(2, 3)
    meshes = {
        "triangles-quadrilaterals-2d": (
            _jittered_unit_square_triangles(6, 1),
            phx.discretization.CellMesh(
                square_points,
                (phx.discretization.CellBlock("quads", "quadrilateral", quadrilaterals),),
            ),
        ),
        "triangles-triangles-2d": (
            _jittered_unit_square_triangles(6, 1),
            _jittered_unit_square_triangles(5, 2),
        ),
        "tetrahedra-tetrahedra-3d": (
            _jittered_cube_tetrahedra(3, 3),
            _jittered_cube_tetrahedra(2, 4),
        ),
        "hexahedra-tetrahedra-3d": (
            phx.discretization.CellMesh(
                cube_points,
                (phx.discretization.CellBlock("hexahedra", "hexahedron", hexahedra),),
            ),
            _jittered_cube_tetrahedra(3, 3),
        ),
    }
    cases = {name: _supermesh_case(*pair) for name, pair in meshes.items()}
    sections = ("source", "target", "quality", "conservation", "resources")
    return {
        "status": "passed",
        "source": {name: case["source"] for name, case in cases.items()},
        "target": {name: case["target"] for name, case in cases.items()},
        "runtime": _runtime_identity(),
        "quality": {
            # ty: ignore[not-subscriptable]
            "before": {name: case["quality"]["source"] for name, case in cases.items()},
            # ty: ignore[not-subscriptable]
            "after": {name: case["quality"]["target"] for name, case in cases.items()},
        },
        "conservation": {name: case["conservation"] for name, case in cases.items()},
        "resources": {name: case["resources"] for name, case in cases.items()},
        "cases": {
            name: {key: value for key, value in case.items() if key not in sections}
            for name, case in cases.items()
        },
    }


def _remap_cell_averages(discretization: Any, function: Any, /) -> np.ndarray:
    """Quadrature cell averages of ``function`` on one FV discretization."""
    weights = np.where(
        np.asarray(discretization.cell_quadrature_valid),
        np.asarray(discretization.cell_quadrature_weights),
        0.0,
    )
    values = function(np.asarray(discretization.cell_quadrature_points))
    return np.sum(weights * values, axis=1) / np.sum(weights, axis=1)


def _remap_linear(points: np.ndarray, /) -> np.ndarray:
    return 0.5 + 2.0 * points[..., 0] - 3.0 * points[..., 1]


def qualify_remap_p1() -> dict[str, object]:
    if not meshcore_available():
        return _missing_dependency("phydrax-meshcore", _MESHCORE_DETAIL)
    pass  # Source fixture helpers are owned by this scenario package.

    discretization = phx.discretization
    contract = _contract()
    points, quadrilaterals = _unit_lattice_cells(6, 2)
    triangles = _jittered_unit_square_triangles(5, 3)
    source, source_record = _measured(
        lambda: discretization.UnstructuredFiniteVolumePlan(
            points, quadrilaterals=quadrilaterals
        ).prepare()
    )
    target, target_record = _measured(
        lambda: discretization.UnstructuredFiniteVolumePlan(
            np.asarray(triangles.coordinates),
            triangles=np.asarray(triangles.blocks[0].vertices),
        ).prepare()
    )
    # ty: ignore[unresolved-attribute]
    source_result = phx.meshing.certify_cell_mesh(source.mesh, contract)
    # ty: ignore[unresolved-attribute]
    target_result = phx.meshing.certify_cell_mesh(target.mesh, contract)
    prepared, refinement_record = _measured(
        lambda: discretization.prepare_unstructured_conservative_remap(
            # ty: ignore[invalid-argument-type]
            source,
            # ty: ignore[invalid-argument-type]
            target,
            provenance="qualification-remap-p1",
        )
    )
    # ty: ignore[unresolved-attribute]
    if not prepared.succeeded:
        # ty: ignore[unresolved-attribute]
        raise RuntimeError(f"P0 remap preparation failed: {prepared.reason}.")
    limited_plan, limited_record = _measured(
        lambda: discretization.UnstructuredSecondOrderRemapPlan(
            # ty: ignore[unresolved-attribute]
            prepared.plan,
            # ty: ignore[unresolved-attribute]
            prepared.refinement,
            # ty: ignore[invalid-argument-type]
            source,
        )
    )
    unlimited_plan, unlimited_record = _measured(
        lambda: discretization.UnstructuredSecondOrderRemapPlan(
            # ty: ignore[unresolved-attribute]
            prepared.plan,
            # ty: ignore[unresolved-attribute]
            prepared.refinement,
            # ty: ignore[invalid-argument-type]
            source,
            limiter=discretization.UnstructuredRemapLimiter.NONE,
        )
    )
    # ty: ignore[unresolved-attribute]
    source_volumes = np.asarray(source.cell_volumes, dtype=np.float64)
    # ty: ignore[unresolved-attribute]
    target_volumes = np.asarray(target.cell_volumes, dtype=np.float64)
    smooth = jnp.asarray(_remap_cell_averages(source, _remap_field))
    smooth_exact = _remap_cell_averages(target, _remap_field)
    linear = jnp.asarray(_remap_cell_averages(source, _remap_linear))
    linear_exact = _remap_cell_averages(target, _remap_linear)
    # ty: ignore[unresolved-attribute]
    centers = np.asarray(source.cell_centers)
    step = jnp.asarray(np.where(centers[:, 0] + 0.3 * centers[:, 1] > 0.55, 2.0, -1.0))
    constant = jnp.ones_like(smooth)

    def relative_l1(values: Any, exact: Any) -> Any:
        return float(
            np.sum(target_volumes * np.abs(np.asarray(values) - exact))
            / np.sum(target_volumes * np.abs(exact))
        )

    def mass_residual(source_values: Any, target_values: Any) -> Any:
        return abs(
            float(np.sum(target_volumes * np.asarray(target_values)))
            - float(np.sum(source_volumes * np.asarray(source_values)))
        ) / float(np.sum(source_volumes * np.abs(np.asarray(source_values))))

    p0, p0_record = _measured(
        lambda: jax.block_until_ready(
            {
                # ty: ignore[unresolved-attribute]
                name: prepared.plan.apply(values)
                for name, values in (
                    ("constant", constant),
                    ("smooth", smooth),
                    ("linear", linear),
                    ("step", step),
                )
            }
        )
    )
    p1, p1_record = _measured(
        lambda: jax.block_until_ready(
            {
                # ty: ignore[unresolved-attribute]
                name: limited_plan.apply(values)
                for name, values in (
                    ("constant", constant),
                    ("smooth", smooth),
                    ("linear", linear),
                    ("step", step),
                )
            }
        )
    )
    # ty: ignore[unresolved-attribute]
    unlimited_linear = unlimited_plan.apply(linear)
    # ty: ignore[not-subscriptable, unresolved-attribute]
    p0_defect = prepared.plan.conservation_defect(smooth, p0["smooth"])
    fv = {
        "p0": {
            # ty: ignore[not-subscriptable]
            "constant_error": float(np.max(np.abs(np.asarray(p0["constant"]) - 1.0))),
            # ty: ignore[not-subscriptable]
            "smooth_mass_residual": mass_residual(smooth, p0["smooth"]),
            # ty: ignore[not-subscriptable]
            "step_mass_residual": mass_residual(step, p0["step"]),
            "plan_conservation_defect": float(np.max(np.abs(np.asarray(p0_defect)))),
            # ty: ignore[not-subscriptable]
            "smooth_relative_l1_error": relative_l1(p0["smooth"], smooth_exact),
            # ty: ignore[not-subscriptable]
            "linear_relative_l1_error": relative_l1(p0["linear"], linear_exact),
            # ty: ignore[not-subscriptable]
            "step_bounds": [float(jnp.min(p0["step"])), float(jnp.max(p0["step"]))],
        },
        "limited_p1": {
            "constant_error": float(
                # ty: ignore[not-subscriptable]
                np.max(np.abs(np.asarray(p1["constant"].values) - 1.0))
            ),
            # ty: ignore[not-subscriptable]
            "smooth_mass_residual": mass_residual(smooth, p1["smooth"].values),
            # ty: ignore[not-subscriptable]
            "step_mass_residual": mass_residual(step, p1["step"].values),
            "step_conservation_residual_after": float(
                # ty: ignore[not-subscriptable]
                np.abs(np.asarray(p1["step"].conservation_residual_after))
            ),
            # ty: ignore[not-subscriptable]
            "step_restored": bool(p1["step"].restored),
            # ty: ignore[not-subscriptable]
            "smooth_relative_l1_error": relative_l1(p1["smooth"].values, smooth_exact),
            # ty: ignore[not-subscriptable]
            "linear_relative_l1_error": relative_l1(p1["linear"].values, linear_exact),
            "step_bounds": [
                # ty: ignore[not-subscriptable]
                float(jnp.min(p1["step"].values)),
                # ty: ignore[not-subscriptable]
                float(jnp.max(p1["step"].values)),
            ],
            # ty: ignore[not-subscriptable]
            "step_limited_cells": int(p1["step"].limited_count),
            # ty: ignore[not-subscriptable]
            "step_minimum_limiter_factor": float(p1["step"].minimum_limiter_factor),
            # ty: ignore[unresolved-attribute]
            "gradient_minimum_rank": int(np.min(np.asarray(limited_plan.gradient_rank))),
            "gradient_maximum_condition": float(
                # ty: ignore[unresolved-attribute]
                np.max(np.asarray(limited_plan.gradient_condition))
            ),
        },
        "unlimited_p1": {
            "linear_maximum_error": float(
                np.max(np.abs(np.asarray(unlimited_linear.values) - linear_exact))
            ),
            "linear_mass_residual": mass_residual(linear, unlimited_linear.values),
        },
    }
    step_bounds = fv["limited_p1"]["step_bounds"]
    if (
        max(
            fv["p0"]["constant_error"],
            fv["p0"]["smooth_mass_residual"],
            fv["p0"]["step_mass_residual"],
            fv["limited_p1"]["constant_error"],
            fv["limited_p1"]["smooth_mass_residual"],
            fv["limited_p1"]["step_mass_residual"],
            fv["unlimited_p1"]["linear_maximum_error"],
            fv["unlimited_p1"]["linear_mass_residual"],
        )
        > 1.0e-12
        or step_bounds[0] < -1.0 - 1.0e-12
        or step_bounds[1] > 2.0 + 1.0e-12
        or not fv["limited_p1"]["step_restored"]
        or fv["limited_p1"]["step_limited_cells"] == 0
        or fv["limited_p1"]["smooth_relative_l1_error"]
        >= fv["p0"]["smooth_relative_l1_error"]
    ):
        raise RuntimeError(f"Finite-volume remap qualification failed: {fv}.")

    # FEM L2 projection of a continuous P1 field between jittered triangle meshes.
    fem_source_mesh = _jittered_unit_square_triangles(7, 1)
    fem_target_mesh = _jittered_unit_square_triangles(5, 2)

    def p1_space(mesh: Any) -> Any:
        return discretization.FiniteElementPlan(
            mesh,
            discretization.FiniteElementFieldSpec(
                "u", discretization.lagrange_element("triangle", 1)
            ),
        ).prepare()

    fem_source, fem_target = p1_space(fem_source_mesh), p1_space(fem_target_mesh)
    fem_refinement, fem_refinement_record = _measured(
        lambda: phx.geometry.prepare_common_refinement(
            fem_source_mesh,
            fem_target_mesh,
            policy=phx.geometry.CommonRefinementPolicy(overlap_simplices=True),
        )
    )
    # ty: ignore[unresolved-attribute]
    if not fem_refinement.succeeded:
        # ty: ignore[unresolved-attribute]
        raise RuntimeError(f"L2 projection refinement failed: {fem_refinement.status}.")
    prepared_target, target_record = _measured(
        lambda: discretization.prepare_l2_projection_target(fem_target, field_name="u")
    )
    transfer, transfer_record = _measured(
        lambda: discretization.prepare_l2_projection_transfer(
            fem_source,
            # ty: ignore[invalid-argument-type]
            prepared_target,
            # ty: ignore[invalid-argument-type]
            fem_refinement,
            field_name="u",
        )
    )
    source_dofs = np.asarray(fem_source.dof_maps[0].dof_coordinates)
    target_dofs = np.asarray(fem_target.dof_maps[0].dof_coordinates)
    source_weights = np.asarray(
        fem_source.mass.mv(jnp.ones((fem_source.dof_maps[0].global_dof_count,)))
    )
    target_weights = np.asarray(
        fem_target.mass.mv(jnp.ones((fem_target.dof_maps[0].global_dof_count,)))
    )
    fem_smooth = _remap_field(source_dofs)
    projected, projection_record = _measured(
        lambda: jax.block_until_ready(
            (
                # ty: ignore[unresolved-attribute]
                transfer.apply(jnp.ones_like(jnp.asarray(fem_smooth))),
                # ty: ignore[unresolved-attribute]
                transfer.apply(jnp.asarray(_remap_linear(source_dofs))),
                # ty: ignore[unresolved-attribute]
                transfer.apply(jnp.asarray(fem_smooth)),
            )
        )
    )
    source_integral = float(np.dot(source_weights, fem_smooth))
    fem = {
        "flags": {
            # ty: ignore[unresolved-attribute]
            "preserves_constants": transfer.preserves_constants,
            # ty: ignore[unresolved-attribute]
            "preserves_linear": transfer.preserves_linear,
            # ty: ignore[unresolved-attribute]
            "conservative": transfer.conservative,
            # ty: ignore[unresolved-attribute]
            "positivity_preserving": transfer.positivity_preserving,
        },
        # ty: ignore[not-subscriptable]
        "constant_error": float(np.max(np.abs(np.asarray(projected[0]) - 1.0))),
        "linear_maximum_error": float(
            # ty: ignore[not-subscriptable]
            np.max(np.abs(np.asarray(projected[1]) - _remap_linear(target_dofs)))
        ),
        "smooth_integral_relative_residual": abs(
            # ty: ignore[not-subscriptable]
            float(np.dot(target_weights, np.asarray(projected[2]))) - source_integral
        )
        / abs(source_integral),
        "smooth_nodal_maximum_error": float(
            # ty: ignore[not-subscriptable]
            np.max(np.abs(np.asarray(projected[2]) - _remap_field(target_dofs)))
        ),
        # ty: ignore[unresolved-attribute]
        "target_mass_condition": float(prepared_target.mass_condition.value),
        # ty: ignore[unresolved-attribute]
        "factorization_status": int(prepared_target.factorization.status),
    }
    if (
        not (
            # ty: ignore[unresolved-attribute]
            transfer.preserves_constants
            # ty: ignore[unresolved-attribute]
            and transfer.preserves_linear
            # ty: ignore[unresolved-attribute]
            and transfer.conservative
        )
        # ty: ignore[unresolved-attribute]
        or transfer.positivity_preserving
        or max(
            fem["constant_error"],
            fem["linear_maximum_error"],
            fem["smooth_integral_relative_residual"],
        )
        > 1.0e-10
    ):
        raise RuntimeError(f"L2 projection qualification failed: {fem}.")
    # ty: ignore[unresolved-attribute]
    evidence = prepared.evidence
    return {
        "status": "passed",
        "source": {
            "finite_volume": {
                **_result_identity(source_result),
                # ty: ignore[unresolved-attribute]
                "prepared_id": source.prepared_id,
                # ty: ignore[unresolved-attribute]
                "geometry_id": source.geometry_id,
            },
            "finite_element": {
                "mesh_id": fem_source_mesh.mesh_id,
                "topology_id": fem_source_mesh.topology_id,
                "prepared_id": fem_source.prepared_id,
                "dofs": source_dofs.shape[0],
            },
        },
        "target": {
            "finite_volume": {
                **_result_identity(target_result),
                # ty: ignore[unresolved-attribute]
                "prepared_id": target.prepared_id,
                # ty: ignore[unresolved-attribute]
                "geometry_id": target.geometry_id,
            },
            "finite_element": {
                "mesh_id": fem_target_mesh.mesh_id,
                "topology_id": fem_target_mesh.topology_id,
                "prepared_id": fem_target.prepared_id,
                "dofs": target_dofs.shape[0],
            },
        },
        "identities": {
            # ty: ignore[unresolved-attribute]
            "refinement_id": prepared.refinement.refinement_id,
            # ty: ignore[unresolved-attribute]
            "remap_id": prepared.remap_id,
            # ty: ignore[unresolved-attribute]
            "p0_plan_id": prepared.plan.plan_id,
            # ty: ignore[unresolved-attribute]
            "limited_p1_plan_id": limited_plan.plan_id,
            # ty: ignore[unresolved-attribute]
            "unlimited_p1_plan_id": unlimited_plan.plan_id,
            # ty: ignore[unresolved-attribute]
            "l2_refinement_id": fem_refinement.refinement_id,
            # ty: ignore[unresolved-attribute]
            "l2_transfer_id": transfer.transfer_id,
        },
        "runtime": _runtime_identity(),
        "quality": {
            "before": _quality_summary(source_result.quality),
            "after": _quality_summary(target_result.quality),
        },
        "conservation": {
            "finite_volume": fv,
            "coverage": {
                "maximum_relative_source_defect": evidence.maximum_relative_source_defect,
                "maximum_relative_target_defect": evidence.maximum_relative_target_defect,
                # ty: ignore[unresolved-attribute]
                "entries": prepared.refinement.entry_count,
            },
        },
        "transfer": {"l2_projection": fem},
        "resources": {
            "source_prepare": source_record,
            "target_prepare": target_record,
            "refinement_and_p0_plan": refinement_record,
            "limited_p1_plan": limited_record,
            "unlimited_p1_plan": unlimited_record,
            "p0_apply": p0_record,
            "limited_p1_apply": p1_record,
            "l2_refinement": fem_refinement_record,
            "l2_target_prepare": target_record,
            "l2_transfer_prepare": transfer_record,
            "l2_apply": projection_record,
        },
    }


def _bisection_meshes() -> dict[str, phx.discretization.CellMesh]:
    """A 4 x 4 right-triangle lattice of the unit square and 2^3 Kuhn cubes."""
    pass  # Source fixture helpers are owned by this scenario package.

    axis = np.linspace(0.0, 1.0, 5)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    spatial, tetrahedra = _kuhn_tetrahedra(2)
    return {
        "triangles": phx.discretization.CellMesh.from_triangles(
            points, _planar_triangles(4)
        ),
        "tetrahedra": phx.discretization.CellMesh.from_tetrahedra(spatial, tetrahedra),
    }


def _bisection_cell_ids(mesh: phx.discretization.CellMesh, /) -> np.ndarray:
    return np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)


def _bisection_corner_cells(mesh: phx.discretization.CellMesh, /) -> np.ndarray:
    """Cells touching the vertex nearest the origin (repeated local refinement)."""
    corner = np.argmin(np.linalg.norm(np.asarray(mesh.coordinates), axis=1))
    touching = np.any(np.asarray(mesh.blocks[0].vertices) == corner, axis=1)
    return np.sort(_bisection_cell_ids(mesh)[touching])


def _bisection_adapt(
    route: phx.meshing.MeshAdaptationRoute,
    source: phx.meshing.CellMeshingResult,
    refine: np.ndarray,
    coarsen: np.ndarray,
    hierarchy: Any,
    /,
    *,
    device_policy: phx.discretization.AdaptiveSimplexPolicy | None = None,
) -> tuple[phx.meshing.MeshAdaptationResult, dict[str, dict[str, float]]]:
    request = phx.meshing.MarkedMeshAdaptation(
        np.asarray(refine, dtype=np.int64),
        np.asarray(coarsen, dtype=np.int64),
        hierarchy=hierarchy,
    )
    policy = phx.meshing.MeshAdaptationPolicy(route, device_policy=device_policy)
    prepared, preparation = _measured(
        lambda: phx.meshing.prepare_mesh_adaptation(source, request, policy=policy)
    )
    # ty: ignore[invalid-argument-type]
    result, execution = _measured(lambda: phx.meshing.execute_mesh_adaptation(prepared))
    # ty: ignore[unresolved-attribute]
    if not result.target.audit.passed:
        raise RuntimeError(f"{route.name} produced a target that failed its audit.")
    # ty: ignore[invalid-return-type]
    return result, {"prepare": preparation, "execute": execution}


def _bisection_retained_bytes(tree: Any, /) -> int:
    """Bytes of the distinct array leaves an adaptation artifact retains."""
    leaves = {
        id(leaf): leaf
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, (jax.Array, np.ndarray))
    }
    return sum(leaf.nbytes for leaf in leaves.values())


def _bisection_lineage(result: phx.meshing.MeshAdaptationResult, /) -> dict[str, object]:
    """Every entity is an identity, a lineage relation, or created/deleted."""
    source, target = result.source.mesh, result.target.mesh
    lineage = result.lineage
    # ty: ignore[unresolved-attribute]
    dimensions = tuple(record.dimension for record in lineage.entities)
    width = np.asarray(source.blocks[0].vertices).shape[1]
    complete = (
        dimensions == tuple(range(width))
        # ty: ignore[unresolved-attribute]
        and lineage.source_topology_id == source.topology_id
        # ty: ignore[unresolved-attribute]
        and lineage.target_topology_id == target.topology_id
    )
    relations = {}
    # ty: ignore[unresolved-attribute]
    for record in lineage.entities:
        before = np.asarray(source.entity_set(record.dimension).entity_ids)
        after = np.asarray(target.entity_set(record.dimension).entity_ids)
        created = np.asarray(record.created_target_ids)
        deleted = np.asarray(record.deleted_source_ids)
        covered_after = np.concatenate(
            (before, np.asarray(record.target_global_ids), created)
        )
        covered_before = np.concatenate(
            (after, np.asarray(record.source_global_ids), deleted)
        )
        complete = complete and bool(
            np.all(np.isin(after, covered_after))
            and np.all(np.isin(before, covered_before))
            and np.all(np.isin(created, after) & ~np.isin(created, before))
            and np.all(np.isin(deleted, before) & ~np.isin(deleted, after))
        )
        relations[f"dimension_{record.dimension}"] = {
            "relations": record.relation_kinds.shape[0],
            "created": created.shape[0],
            "deleted": deleted.shape[0],
        }
    if not complete:
        raise RuntimeError("Bisection lineage does not account for every entity.")
    # ty: ignore[unresolved-attribute]
    return {"lineage_id": lineage.lineage_id, "complete": complete, **relations}


def _bisection_evidence(evidence: Any, /) -> dict[str, object]:
    return {
        "evidence_id": evidence.evidence_id,
        "requested_refinements": evidence.requested_refinements,
        "accepted_refinements": evidence.accepted_refinements,
        "rejected_refinements": evidence.rejected_refinement_ids.shape[0],
        "admissibility_tests": evidence.admissibility_tests,
        "bisections": evidence.bisections,
        "closure_iterations": evidence.closure_iterations,
        "created_vertices": evidence.created_vertices,
        "maximum_generation": evidence.maximum_generation,
        "initially_compatible": evidence.initially_compatible,
        "incompatible_facets": evidence.incompatible_facets,
        "uniform_refinement_applied": evidence.uniform_refinement_applied,
        "requested_coarsenings": evidence.requested_coarsenings,
        "coarsened_vertices": evidence.coarsened_vertices,
        "coarsening_passes": evidence.coarsening_passes,
        "restored_cells": evidence.restored_cells,
        "rejected_coarsenings": evidence.rejected_coarsening_ids.shape[0],
    }


def _bisection_p1_integral(mesh: phx.discretization.CellMesh, values: Any, /) -> float:
    """Exact integral of the P1 interpolant of vertex values."""
    cells = np.asarray(mesh.blocks[0].vertices)
    corners = np.asarray(mesh.coordinates)[cells]
    edges = corners[:, 1:] - corners[:, :1]
    width = cells.shape[1]
    measures = np.abs(np.linalg.det(edges)) / (2.0 if width == 3 else 6.0)
    return float(np.sum(measures * np.mean(np.asarray(values)[cells], axis=1)))


def _bisection_transfer(result: phx.meshing.MeshAdaptationResult, /) -> dict[str, object]:
    """Claims, P1 linear reproduction, and P1 integral change of one transfer."""
    transfer = result.transfer
    source = np.asarray(result.source.mesh.coordinates)
    target = np.asarray(result.target.mesh.coordinates)
    slope = np.linspace(0.5, 1.5, source.shape[1])
    # ty: ignore[unresolved-attribute]
    linear = np.asarray(transfer.apply(source @ slope + 0.25))
    reproduction = float(np.max(np.abs(linear - (target @ slope + 0.25))))
    # ty: ignore[unresolved-attribute]
    constant = np.asarray(transfer.apply(np.ones((source.shape[0],))))
    smooth = np.sin(3.0 * source[:, 0]) * np.cos(2.0 * source[:, -1])
    # ty: ignore[unresolved-attribute]
    moved = np.asarray(transfer.apply(smooth))
    residual = _bisection_p1_integral(result.target.mesh, moved) - _bisection_p1_integral(
        result.source.mesh, smooth
    )
    if (
        # ty: ignore[unresolved-attribute]
        not transfer.preserves_constants
        # ty: ignore[unresolved-attribute]
        or not transfer.preserves_linear
        or reproduction > 1.0e-12
        or float(np.max(np.abs(constant - 1.0))) > 1.0e-14
    ):
        raise RuntimeError("The bisection transfer does not reproduce P1 fields.")
    return {
        # ty: ignore[unresolved-attribute]
        "transfer_id": transfer.transfer_id,
        # ty: ignore[unresolved-attribute]
        "preserves_constants": transfer.preserves_constants,
        # ty: ignore[unresolved-attribute]
        "preserves_linear": transfer.preserves_linear,
        # ty: ignore[unresolved-attribute]
        "conservative": transfer.conservative,
        # ty: ignore[unresolved-attribute]
        "positivity_preserving": transfer.positivity_preserving,
        "linear_reproduction_error": reproduction,
        "constant_reproduction_error": float(np.max(np.abs(constant - 1.0))),
        "p1_integral_residual": residual,
        "retained_bytes": _bisection_retained_bytes(transfer),
    }


def _bisection_shape_classes(
    source: phx.meshing.CellMeshingResult, rounds: int, cycles: int, /
) -> dict[str, object]:
    """Uniform class bound, then repeated corner refinement against it.

    Newest-vertex (Maubach) bisection generates finitely many similarity
    classes; the first uniform generations visit them all, so locally refined
    meshes never fall below the uniform bound.
    """
    route = phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
    angle = source.quality.minimum_angle
    mean_ratio = source.quality.minimum_mean_ratio
    current, hierarchy = source, None
    for _ in range(rounds):
        cells = np.sort(_bisection_cell_ids(current.mesh))
        # ty: ignore[invalid-argument-type]
        result, _ = _bisection_adapt(route, current, cells, (), hierarchy)
        current, hierarchy = result.target, result.hierarchy
        angle = min(angle, current.quality.minimum_angle)
        mean_ratio = min(mean_ratio, current.quality.minimum_mean_ratio)
    current, hierarchy = source, None
    angles, ratios = [], []
    for _ in range(cycles):
        marks = _bisection_corner_cells(current.mesh)
        # ty: ignore[invalid-argument-type]
        result, _ = _bisection_adapt(route, current, marks, (), hierarchy)
        current, hierarchy = result.target, result.hierarchy
        angles.append(current.quality.minimum_angle)
        ratios.append(current.quality.minimum_mean_ratio)
    bounded = min(angles) >= angle * (1.0 - 1.0e-9) and min(ratios) >= 0.999 * mean_ratio
    # ty: ignore[unresolved-attribute]
    if not bounded or result.evidence.maximum_generation < cycles:
        raise RuntimeError("Repeated bisection left the uniform shape-class bound.")
    return {
        "uniform_rounds": rounds,
        "uniform_minimum_angle_bound": angle,
        "uniform_minimum_mean_ratio_bound": mean_ratio,
        "corner_cycles": cycles,
        "corner_minimum_angles": angles,
        "corner_minimum_mean_ratios": ratios,
        "minimum_angle_ratio_to_source": min(angles) / source.quality.minimum_angle,
        "minimum_mean_ratio_ratio_to_source": min(ratios)
        / source.quality.minimum_mean_ratio,
        # ty: ignore[unresolved-attribute]
        "maximum_generation": result.evidence.maximum_generation,
        "final_cells": current.mesh.blocks[0].cell_count,
        "bounded": bounded,
    }


def _bisection_round_trip(
    mesh: phx.discretization.CellMesh, rounds: int, cycles: int, /
) -> dict[str, object]:
    route = phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
    source, certification = _measured(
        lambda: phx.meshing.certify_cell_mesh(mesh, _contract())
    )
    marks = np.sort(_bisection_cell_ids(mesh))[::3]
    # ty: ignore[invalid-argument-type]
    first, first_stages = _bisection_adapt(route, source, marks, (), None)
    corner = _bisection_corner_cells(first.target.mesh)
    second, second_stages = _bisection_adapt(
        route,
        first.target,
        corner,
        # ty: ignore[invalid-argument-type]
        (),
        first.hierarchy,
    )
    everything = np.sort(_bisection_cell_ids(second.target.mesh))
    restored, restore_stages = _bisection_adapt(
        route,
        second.target,
        # ty: ignore[invalid-argument-type]
        (),
        everything,
        second.hierarchy,
    )
    back = restored.target.mesh
    identical = (
        back.topology_id == mesh.topology_id
        and np.array_equal(np.asarray(back.coordinates), np.asarray(mesh.coordinates))
        and np.array_equal(
            np.asarray(back.vertex_global_ids), np.asarray(mesh.vertex_global_ids)
        )
        and np.array_equal(_bisection_cell_ids(back), _bisection_cell_ids(mesh))
    )
    statuses = (first.status, second.status, restored.status)
    complete = phx.meshing.MeshAdaptationStatus.COMPLETE
    if not identical or first.status is not complete or second.status is not complete:
        raise RuntimeError("The bisection refine/coarsen round trip is not exact.")
    refinement = _bisection_transfer(first)
    # ty: ignore[invalid-argument-type]
    if not refinement["conservative"] or abs(refinement["p1_integral_residual"]) > 1e-13:
        raise RuntimeError("Refinement transfer does not conserve P1 integrals.")
    return {
        # ty: ignore[invalid-argument-type]
        "source": _result_identity(source),
        "refined": _result_identity(second.target),
        "target": _result_identity(restored.target),
        "statuses": [status.value for status in statuses],
        "restores_source_topology": identical,
        "quality": {
            # ty: ignore[unresolved-attribute]
            "before": _quality_summary(source.quality),
            "refined": _quality_summary(second.target.quality),
            "after": _quality_summary(restored.target.quality),
        },
        "transfer": {
            "refinement": refinement,
            "local_refinement": _bisection_transfer(second),
            "coarsening": _bisection_transfer(restored),
        },
        "lineage": {
            "refinement": _bisection_lineage(first),
            "local_refinement": _bisection_lineage(second),
            "coarsening": _bisection_lineage(restored),
        },
        "evidence": {
            "refinement": _bisection_evidence(first.evidence),
            "local_refinement": _bisection_evidence(second.evidence),
            "coarsening": _bisection_evidence(restored.evidence),
        },
        # ty: ignore[unresolved-attribute]
        "hierarchy_id": second.hierarchy.hierarchy_id,
        # ty: ignore[invalid-argument-type]
        "shape_classes": _bisection_shape_classes(source, rounds, cycles),
        "resources": {
            "certify": certification,
            "refinement": first_stages,
            "local_refinement": second_stages,
            "coarsening": restore_stages,
            "hierarchy_retained_bytes": _bisection_retained_bytes(second.hierarchy),
            "target_mesh_retained_bytes": _bisection_retained_bytes(second.target.mesh),
        },
    }


def qualify_bisection() -> dict[str, object]:
    """NATIVE_BISECTION refine/coarsen round trips of triangles and tetrahedra."""
    meshes = _bisection_meshes()
    dimensions = {
        "triangles": _bisection_round_trip(meshes["triangles"], 4, 12),
        "tetrahedra": _bisection_round_trip(meshes["tetrahedra"], 3, 6),
    }
    return {
        "status": "passed",
        "route": phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION.value,
        "source": {name: record["source"] for name, record in dimensions.items()},
        "target": {name: record["target"] for name, record in dimensions.items()},
        "quality": {
            name: {
                # ty: ignore[not-subscriptable]
                "before": record["quality"]["before"],
                # ty: ignore[not-subscriptable]
                "after": record["quality"]["after"],
            }
            for name, record in dimensions.items()
        },
        "transfer": {name: record["transfer"] for name, record in dimensions.items()},
        "conservation": {
            name: {
                stage: values["p1_integral_residual"]
                # ty: ignore[unresolved-attribute]
                for stage, values in record["transfer"].items()
            }
            for name, record in dimensions.items()
        },
        "resources": {name: record["resources"] for name, record in dimensions.items()},
        "dimensions": dimensions,
        "runtime": _runtime_identity(),
    }


from .._runtime import runtime_identity as _runtime_identity


def _planar_triangles(cells_per_axis: int, /) -> np.ndarray:
    index = np.arange((cells_per_axis + 1) ** 2, dtype=np.int32).reshape(
        (cells_per_axis + 1, cells_per_axis + 1)
    )
    lower, right, upper, left = (
        index[:-1, :-1].reshape(-1),
        index[1:, :-1].reshape(-1),
        index[1:, 1:].reshape(-1),
        index[:-1, 1:].reshape(-1),
    )
    return np.concatenate(
        (np.stack((lower, right, upper), axis=1), np.stack((lower, upper, left), axis=1))
    )


def _jittered_unit_square_triangles(
    cells_per_axis: int, seed: int, /
) -> phx.discretization.CellMesh:
    axis = np.linspace(0.0, 1.0, cells_per_axis + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    jitter = np.random.default_rng(seed).uniform(-0.2, 0.2, points.shape)
    points[interior] += jitter[interior] / cells_per_axis
    return phx.discretization.CellMesh.from_triangles(
        points, _planar_triangles(cells_per_axis)
    )


def _remap_field(points: np.ndarray, /) -> np.ndarray:
    x, y = points[..., 0], points[..., 1]
    return np.sin(2.0 * np.pi * x) * np.cos(np.pi * y) + x * x


def _kuhn_tetrahedra(cells_per_axis: int, /) -> tuple[np.ndarray, np.ndarray]:
    from itertools import permutations

    axis = np.linspace(0.0, 1.0, cells_per_axis + 1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        (-1, 3)
    )
    vertex_rows = np.arange((cells_per_axis + 1) ** 3, dtype=np.int64).reshape(
        (cells_per_axis + 1,) * 3
    )
    tetrahedra = []
    for cube in np.ndindex((cells_per_axis,) * 3):
        for order in permutations(range(3)):
            point = np.asarray(cube, dtype=np.int64)
            path = [point.copy()]
            for coordinate_axis in order:
                point = point.copy()
                point[coordinate_axis] += 1
                path.append(point)
            tetrahedron = np.asarray(
                [vertex_rows[tuple(row)] for row in path], dtype=np.int32
            )
            if np.linalg.det(points[tetrahedron[1:]] - points[tetrahedron[0]]) < 0.0:
                tetrahedron[[1, 2]] = tetrahedron[[2, 1]]
            tetrahedra.append(tetrahedron)
    return points, np.asarray(tetrahedra, dtype=np.int32)
