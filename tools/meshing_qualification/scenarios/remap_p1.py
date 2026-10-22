# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import Any, final

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.geometry import CommonRefinementEvidence
from phydrax.meshing import meshcore_available
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._fixtures import (
    _jittered_unit_square_triangles,
    _remap_field,
)
from tools.meshing_qualification._records import _quality_summary, _result_identity
from tools.meshing_qualification._runtime import (
    _contract,
    _measured,
    _missing_dependency,
    _runtime_identity,
    qualification_failure,
    StageMeasurement,
)
from tools.meshing_qualification.scenarios.supermesh import (
    _MESHCORE_DETAIL,
    _unit_lattice_cells,
)


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


@final
class _RemapFiniteVolumePreparation(phx.StrictModule):
    source: phx.discretization.UnstructuredFiniteVolumeDiscretization
    target: phx.discretization.UnstructuredFiniteVolumeDiscretization
    source_result: phx.meshing.CellMeshingResult
    target_result: phx.meshing.CellMeshingResult
    conservative_plan: phx.discretization.UnstructuredConservativeRemapPlan
    common_refinement: phx.geometry.PreparedCommonRefinement
    limited_plan: phx.discretization.UnstructuredSecondOrderRemapPlan
    unlimited_plan: phx.discretization.UnstructuredSecondOrderRemapPlan
    source_record: StageMeasurement
    target_record: StageMeasurement
    refinement_record: StageMeasurement
    limited_record: StageMeasurement
    unlimited_record: StageMeasurement
    evidence: CommonRefinementEvidence
    remap_id: str


def _prepare_remap_finite_volume() -> _RemapFiniteVolumePreparation:
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

    source_result = phx.meshing.certify_cell_mesh(source.mesh, contract)

    target_result = phx.meshing.certify_cell_mesh(target.mesh, contract)
    prepared, refinement_record = _measured(
        lambda: discretization.prepare_unstructured_conservative_remap(
            source,
            target,
            provenance="qualification-remap-p1",
        )
    )

    if not prepared.succeeded:
        raise qualification_failure(
            "qualify_remap_p1",
            (f"P0 remap preparation failed: {prepared.reason}."),
            {"prepared": prepared},
        )
    conservative_plan = prepared.plan
    if conservative_plan is None:
        raise qualification_failure(
            "qualify_remap_p1.prepared.plan",
            "be present after successful preparation",
            {"prepared": prepared},
        )
    match prepared.route:
        case phx.geometry.PreparedCommonRefinement() as common_refinement:
            pass
        case route:
            raise qualification_failure(
                "qualify_remap_p1.prepared.route",
                "be a common refinement for second-order reconstruction",
                {"route": type(route).__name__},
            )
    if not isinstance(prepared.evidence, CommonRefinementEvidence):
        raise qualification_failure(
            "qualify_remap_p1",
            ("Affine P1 remap qualification requires affine common-refinement evidence."),
            {
                "CommonRefinementEvidence": CommonRefinementEvidence,
                "prepared": prepared,
            },
        )
    limited_plan, limited_record = _measured(
        lambda: discretization.UnstructuredSecondOrderRemapPlan(
            conservative_plan,
            common_refinement,
            source,
        )
    )
    unlimited_plan, unlimited_record = _measured(
        lambda: discretization.UnstructuredSecondOrderRemapPlan(
            conservative_plan,
            common_refinement,
            source,
            limiter=discretization.UnstructuredRemapLimiter.NONE,
        )
    )
    return _RemapFiniteVolumePreparation(
        source=source,
        target=target,
        source_result=source_result,
        target_result=target_result,
        conservative_plan=conservative_plan,
        common_refinement=common_refinement,
        limited_plan=limited_plan,
        unlimited_plan=unlimited_plan,
        source_record=source_record,
        target_record=target_record,
        refinement_record=refinement_record,
        limited_record=limited_record,
        unlimited_record=unlimited_record,
        evidence=prepared.evidence,
        remap_id=prepared.remap_id,
    )


@final
class _RemapFiniteVolumeExecution(phx.StrictModule):
    fv: dict[str, Any]
    p0_record: StageMeasurement
    p1_record: StageMeasurement


def _execute_remap_finite_volume(
    preparation: _RemapFiniteVolumePreparation,
) -> _RemapFiniteVolumeExecution:
    source = preparation.source
    target = preparation.target
    conservative_plan = preparation.conservative_plan
    limited_plan = preparation.limited_plan
    unlimited_plan = preparation.unlimited_plan
    source_volumes = np.asarray(source.cell_volumes, dtype=np.float64)

    target_volumes = np.asarray(target.cell_volumes, dtype=np.float64)
    smooth = jnp.asarray(_remap_cell_averages(source, _remap_field))
    smooth_exact = _remap_cell_averages(target, _remap_field)
    linear = jnp.asarray(_remap_cell_averages(source, _remap_linear))
    linear_exact = _remap_cell_averages(target, _remap_linear)

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

    apply_p0 = conservative_plan.apply
    p0, p0_record = _measured(
        lambda: jax.block_until_ready(
            {
                name: apply_p0(values)
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

    unlimited_linear = unlimited_plan.apply(linear)
    p0_defect = conservative_plan.conservation_defect(smooth, p0["smooth"])
    fv = {
        "p0": {
            "constant_error": float(np.max(np.abs(np.asarray(p0["constant"]) - 1.0))),
            "smooth_mass_residual": mass_residual(smooth, p0["smooth"]),
            "step_mass_residual": mass_residual(step, p0["step"]),
            "plan_conservation_defect": float(np.max(np.abs(np.asarray(p0_defect)))),
            "smooth_relative_l1_error": relative_l1(p0["smooth"], smooth_exact),
            "linear_relative_l1_error": relative_l1(p0["linear"], linear_exact),
            "step_bounds": [float(jnp.min(p0["step"])), float(jnp.max(p0["step"]))],
        },
        "limited_p1": {
            "constant_error": float(
                np.max(np.abs(np.asarray(p1["constant"].values) - 1.0))
            ),
            "smooth_mass_residual": mass_residual(smooth, p1["smooth"].values),
            "step_mass_residual": mass_residual(step, p1["step"].values),
            "step_conservation_residual_after": float(
                np.abs(np.asarray(p1["step"].conservation_residual_after))
            ),
            "step_restored": bool(p1["step"].restored),
            "smooth_relative_l1_error": relative_l1(p1["smooth"].values, smooth_exact),
            "linear_relative_l1_error": relative_l1(p1["linear"].values, linear_exact),
            "step_bounds": [
                float(jnp.min(p1["step"].values)),
                float(jnp.max(p1["step"].values)),
            ],
            "step_limited_cells": int(p1["step"].limited_count),
            "step_minimum_limiter_factor": float(p1["step"].minimum_limiter_factor),
            "gradient_minimum_rank": int(np.min(np.asarray(limited_plan.gradient_rank))),
            "gradient_maximum_condition": float(
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
        raise qualification_failure(
            "qualify_remap_p1",
            (f"Finite-volume remap qualification failed: {fv}."),
            {"fv": fv, "step_bounds": step_bounds},
        )
    return _RemapFiniteVolumeExecution(fv=fv, p0_record=p0_record, p1_record=p1_record)


@final
class _RemapFiniteElementProjection(phx.StrictModule):
    fem: dict[str, Any]
    source_identity: dict[str, str | int]
    target_identity: dict[str, str | int]
    refinement_id: str
    transfer_id: str
    l2_refinement_record: StageMeasurement
    l2_target_record: StageMeasurement
    l2_transfer_record: StageMeasurement
    l2_apply_record: StageMeasurement
    retained_bytes: int
    working_bytes: int


def _project_remap_finite_element() -> _RemapFiniteElementProjection:
    discretization = phx.discretization
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
    fem_refinement, l2_refinement_record = _measured(
        lambda: phx.geometry.prepare_common_refinement(
            fem_source_mesh,
            fem_target_mesh,
            policy=phx.geometry.CommonRefinementPolicy(overlap_simplices=True),
        )
    )

    if not fem_refinement.succeeded:
        raise qualification_failure(
            "qualify_remap_p1",
            (f"L2 projection refinement failed: {fem_refinement.status}."),
            {"fem_refinement": fem_refinement},
        )
    prepared_target, l2_target_record = _measured(
        lambda: discretization.prepare_l2_projection_target(fem_target, field_name="u")
    )
    transfer, l2_transfer_record = _measured(
        lambda: discretization.prepare_l2_projection_transfer(
            fem_source,
            prepared_target,
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
    projected, l2_apply_record = _measured(
        lambda: jax.block_until_ready(
            (
                transfer.apply(jnp.ones_like(jnp.asarray(fem_smooth))),
                transfer.apply(jnp.asarray(_remap_linear(source_dofs))),
                transfer.apply(jnp.asarray(fem_smooth)),
            )
        )
    )
    source_integral = float(np.dot(source_weights, fem_smooth))
    fem = {
        "flags": {
            "preserves_constants": transfer.preserves_constants,
            "preserves_linear": transfer.preserves_linear,
            "conservative": transfer.conservative,
            "positivity_preserving": transfer.positivity_preserving,
        },
        "constant_error": float(np.max(np.abs(np.asarray(projected[0]) - 1.0))),
        "linear_maximum_error": float(
            np.max(np.abs(np.asarray(projected[1]) - _remap_linear(target_dofs)))
        ),
        "smooth_integral_relative_residual": abs(
            float(np.dot(target_weights, np.asarray(projected[2]))) - source_integral
        )
        / abs(source_integral),
        "smooth_nodal_maximum_error": float(
            np.max(np.abs(np.asarray(projected[2]) - _remap_field(target_dofs)))
        ),
        "target_mass_condition": float(prepared_target.mass_condition.value),
        "factorization_status": int(prepared_target.factorization.status),
    }
    if (
        not (
            transfer.preserves_constants
            and transfer.preserves_linear
            and transfer.conservative
        )
        or transfer.positivity_preserving
        or max(
            fem["constant_error"],
            fem["linear_maximum_error"],
            fem["smooth_integral_relative_residual"],
        )
        > 1.0e-10
    ):
        raise qualification_failure(
            "qualify_remap_p1",
            (f"L2 projection qualification failed: {fem}."),
            {"fem": fem, "transfer": transfer},
        )
    return _RemapFiniteElementProjection(
        fem=fem,
        source_identity={
            "mesh_id": fem_source_mesh.mesh_id,
            "topology_id": fem_source_mesh.topology_id,
            "prepared_id": fem_source.prepared_id,
            "dofs": source_dofs.shape[0],
        },
        target_identity={
            "mesh_id": fem_target_mesh.mesh_id,
            "topology_id": fem_target_mesh.topology_id,
            "prepared_id": fem_target.prepared_id,
            "dofs": target_dofs.shape[0],
        },
        refinement_id=fem_refinement.refinement_id,
        transfer_id=transfer.transfer_id,
        l2_refinement_record=l2_refinement_record,
        l2_target_record=l2_target_record,
        l2_transfer_record=l2_transfer_record,
        l2_apply_record=l2_apply_record,
        retained_bytes=fem_refinement.evidence.retained_bytes,
        working_bytes=fem_refinement.evidence.working_bytes,
    )


def qualify_remap_p1() -> dict[str, object]:
    if not meshcore_available():
        return _missing_dependency("phydrax-meshcore", _MESHCORE_DETAIL)
    preparation = _prepare_remap_finite_volume()
    source = preparation.source
    target = preparation.target
    source_result = preparation.source_result
    target_result = preparation.target_result
    conservative_plan = preparation.conservative_plan
    common_refinement = preparation.common_refinement
    limited_plan = preparation.limited_plan
    unlimited_plan = preparation.unlimited_plan
    source_record = preparation.source_record
    target_record = preparation.target_record
    refinement_record = preparation.refinement_record
    limited_record = preparation.limited_record
    unlimited_record = preparation.unlimited_record
    evidence = preparation.evidence
    execution = _execute_remap_finite_volume(preparation)
    fv = execution.fv
    p0_record = execution.p0_record
    p1_record = execution.p1_record
    projection = _project_remap_finite_element()
    fem = projection.fem
    l2_refinement_record = projection.l2_refinement_record
    l2_target_record = projection.l2_target_record
    l2_transfer_record = projection.l2_transfer_record
    l2_apply_record = projection.l2_apply_record
    return {
        "status": "passed",
        "source": {
            "finite_volume": {
                **_result_identity(source_result),
                "prepared_id": source.prepared_id,
                "geometry_id": source.geometry_id,
            },
            "finite_element": projection.source_identity,
        },
        "target": {
            "finite_volume": {
                **_result_identity(target_result),
                "prepared_id": target.prepared_id,
                "geometry_id": target.geometry_id,
            },
            "finite_element": projection.target_identity,
        },
        "identities": {
            "refinement_id": common_refinement.refinement_id,
            "remap_id": preparation.remap_id,
            "p0_plan_id": conservative_plan.plan_id,
            "limited_p1_plan_id": limited_plan.plan_id,
            "unlimited_p1_plan_id": unlimited_plan.plan_id,
            "l2_refinement_id": projection.refinement_id,
            "l2_transfer_id": projection.transfer_id,
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
                "entries": common_refinement.entry_count,
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
            "l2_refinement": l2_refinement_record,
            "l2_target_prepare": l2_target_record,
            "l2_transfer_prepare": l2_transfer_record,
            "l2_apply": l2_apply_record,
            "refinement_retained_bytes": evidence.retained_bytes,
            "refinement_working_bytes": evidence.working_bytes,
            "l2_refinement_retained_bytes": projection.retained_bytes,
            "l2_refinement_working_bytes": projection.working_bytes,
        },
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_remap_p1()
