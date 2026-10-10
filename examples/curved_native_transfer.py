#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact quadratic shear, native projected bisection and atomic state transfer.

A tiny linear marking model is fitted to measured cellwise curvature indicators;
it supplies scores, never connectivity or proof. P2 coordinate and field maps
are prepared explicitly. The reaction PDE u = y**2 + a*x, material cell averages
and nodal history cross the native nested transfer. An independently changed
load rejects the entire candidate before a valid reanalysis accepts it.
Fixed-epoch geometry/PDE/transfer JVPs are checked against finite differences
and VJP duality; adaptation/acceptance invalidate those derivatives.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization import CellGeometrySpec, FiniteElementRuntimeData
from phydrax.geometry.design import DerivativeTier, DesignQualificationEvidence
from phydrax.lifecycle import CompositionRebind
from phydrax.meshing import (
    FixedEpochDerivativeEvidence,
    LearnedMeshProposer,
    mesh_proposal_scope,
    MeshProposalFeatures,
    MeshProposalSafetyPolicy,
    prepare_mesh_proposal,
)


SHEAR = 0.1


class LinearMarker(phx.AbstractArrayModel):
    """Least-squares fitted pointwise scores; native policy remains authoritative."""

    weight: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, weight: jax.Array) -> None:
        self.weight = weight
        self.in_size = weight.size
        self.out_size = "scalar"

    def __call__(self, x: jax.Array, /, *, key: jax.Array | None = None) -> jax.Array:
        del key
        return self.weight @ x


def shear(points: jax.Array, amplitude: jax.Array) -> jax.Array:
    return points.at[..., 0].add(amplitude * points[..., 1] ** 2)


def exact(points: jax.Array, amplitude: jax.Array) -> jax.Array:
    return points[..., 1] ** 2 + amplitude * points[..., 0]


def load(points: jax.Array, context: Any) -> jax.Array:
    return exact(points, context.user_args)


def compile_problem(space: Any) -> Any:
    return phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "curved-reaction",
            "u",
            (
                phx.equations.MassAction("u", 1.0),
                phx.equations.SourceAction(
                    "u",
                    phx.equations.coefficient(
                        load, coefficient_id="curved-reaction-load"
                    ),
                ),
            ),
        ),
        space,
    )


def solve(compiled: Any, context: Any) -> jax.Array:
    operator, rhs = compiled.linear_system(context)
    result = phx.linalg.solve(
        operator,
        rhs,
        policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.DenseLU(),
            derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(
                route="primal-factors",
                relative_tolerance=1.0e-12,
                absolute_tolerance=1.0e-13,
            ),
        ),
    )
    return compiled.expand(result.value, context)


def integrate(space: Any, values: jax.Array, amplitude: float) -> tuple[float, float]:
    data = phx.integration.reference_rule_data(
        phx.integration.ReferenceTriangleRule(phx.integration.GaussLegendreRule(6))
    )
    geometry = space.evaluate_block_geometry(
        "u", 0, space.default_runtime.coordinates, data.points, data.weights
    )
    dofs = space.dof_maps[0].cell_dofs[0]
    discrete = phx.ein.contract("ql,cl->cq", geometry.basis_values, values[dofs])
    defect = discrete - exact(geometry.physical_points, jnp.asarray(amplitude))
    return float(jnp.sqrt(jnp.sum(geometry.physical_weights * defect**2))), float(
        jnp.sum(geometry.physical_weights * discrete)
    )


def numerical_derivative(
    function: Callable[[jax.Array], jax.Array],
    parameter: jax.Array,
    /,
    *,
    label: str,
) -> dict[str, float]:
    direction = jnp.asarray(0.7, dtype=jnp.float64)
    value, tangent = jax.jvp(function, (parameter,), (direction,))
    cotangent = jnp.linspace(-0.3, 0.8, value.size, dtype=jnp.float64).reshape(
        value.shape
    )
    _, pullback = jax.vjp(function, parameter)
    duality = float(
        jnp.abs(jnp.vdot(tangent, cotangent) - direction * pullback(cotangent)[0])
    )
    step = 1.0e-5
    finite_difference = (
        function(parameter + step * direction) - function(parameter - step * direction)
    ) / (2.0 * step)
    discrepancy = float(jnp.max(jnp.abs(tangent - finite_difference)))
    if (
        not np.isfinite(discrepancy + duality)
        or discrepancy > 2.0e-8
        or duality > 2.0e-10
    ):
        raise RuntimeError(
            f"{label}: fixed-route finite-difference error={discrepancy}, "
            f"JVP/VJP duality error={duality}."
        )
    return {"finite_difference_error": discrepancy, "JVP_VJP_duality_error": duality}


def run() -> dict[str, Any]:
    axis = np.linspace(0.0, 1.0, 3)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    cells = []
    for i in range(2):
        for j in range(2):
            a = i * 3 + j
            cells.extend(((a, a + 3, a + 4), (a, a + 4, a + 1)))
    mesh = phx.meshing.canonicalize_cell_mesh(
        phx.discretization.CellMesh.from_triangles(
            points, np.asarray(cells, dtype=np.int32)
        )
    )
    coordinate_element = phx.discretization.coordinate_lagrange_element("triangle", 2)
    coordinate_space = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec("coordinate", coordinate_element),
    ).prepare()
    coordinate_map = coordinate_space.dof_maps[0]
    base_coordinates = coordinate_map.dof_coordinates
    geometry = CellGeometrySpec(
        {mesh.blocks[0].name: coordinate_element},
        {mesh.blocks[0].name: coordinate_map.cell_dofs[0]},
        shear(base_coordinates, jnp.asarray(SHEAR)),
    )
    moved = mesh.with_coordinates(
        shear(mesh.coordinates, jnp.asarray(SHEAR)), numeric_version="quadratic-shear"
    )
    source = phx.meshing.certify_cell_mesh(
        moved, phx.SpatialCoordinateContract.si(), geometry=geometry
    )
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 2)
    )
    history = phx.discretization.FiniteElementFieldSpec(
        "history", phx.discretization.lagrange_element("triangle", 2)
    )
    source_space = phx.discretization.FiniteElementPlan(
        source.mesh, field, coordinate_spec=source.geometry
    ).prepare()
    compiled = compile_problem(source_space)
    context = phx.equations.FiniteElementExecutionContext(
        source_space.default_runtime, user_args=jnp.asarray(SHEAR)
    )
    values = solve(compiled, context)
    values.block_until_ready()
    initial_error, initial_integral = integrate(source_space, values, SHEAR)
    if initial_error > 1.0e-10:
        raise RuntimeError("The manufactured curved reaction solve failed.")
    # Measured physical-linear interpolation defect at each cell corner centroid.
    corners = np.asarray(source.mesh.coordinates)[
        np.asarray(source.mesh.blocks[0].vertices)
    ]
    centroids = corners.mean(axis=1)
    targets = np.mean(corners[..., 1] ** 2, axis=1) - centroids[:, 1] ** 2
    features = np.stack((np.ones_like(targets), centroids[:, 1]), axis=1)
    order = np.argsort(np.asarray(source.mesh.blocks[0].global_ids), kind="stable")
    weights = np.linalg.lstsq(features, targets, rcond=None)[0]
    proposer = LearnedMeshProposer(
        LinearMarker(jnp.asarray(weights)),
        kind="marking",
        proposer_id="fitted-quadratic-defect-marker",
    )
    bound = MeshProposalFeatures(
        source,
        mesh_proposal_scope(source, 2),
        features[order],
        feature_ids=("constant-feature", "physical-centroid-y"),
        feature_owner_id="manufactured-quadratic-defect",
    )
    safety = MeshProposalSafetyPolicy(
        source,
        minimum_size=0.05,
        maximum_size=1.0,
        maximum_displacement=0.0,
        maximum_marked_cells=2,
        limits=phx.meshing.MeshingLimits(
            maximum_cells=64, maximum_vertices=64, maximum_wall_seconds=180.0
        ),
    )
    proposed = prepare_mesh_proposal(
        source,
        proposer.propose(source, bound),
        safety,
        native_policy=phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION, limits=safety.limits
        ),
    )
    adaptation = proposed.adaptation
    if (
        adaptation is None
        or not adaptation.status.converged
        or not proposed.safety_audit.passed
    ):
        raise RuntimeError("The projected native proposal was not admitted.")
    geometry_transition = adaptation.geometry_transition
    if geometry_transition is None:
        raise RuntimeError(
            "Curved native adaptation requires its exact coordinate transition."
        )
    target = adaptation.target
    target_space = phx.discretization.FiniteElementPlan(
        target.mesh, field, coordinate_spec=target.geometry
    ).prepare()
    target_problem = compile_problem(target_space)
    history_values = source_space.dof_maps[0].dof_coordinates[:, 1] + 0.5
    materials = phx.equations.MaterialTransaction(
        (
            phx.equations.MaterialState(
                phx.equations.MaterialSiteId("cell-density"),
                "constant-density",
                jnp.full((source.mesh.blocks[0].cell_count,), 2.0, dtype=jnp.float64),
            ),
        )
    )
    accepted = phx.solver.FiniteElementAcceptedState(
        (values, history_values),
        0.0,
        0,
        source.mesh.topology_id,
        source_space.prepared_id,
        compiled.compilation_id,
        materials=materials,
    )
    parents = phx.solver.refinement_parent_cells(adaptation)
    if parents is None:
        raise RuntimeError("A nested parent witness is required.")
    reanalyses: list[dict[str, float]] = []
    rebinds: list[CompositionRebind] = []

    def transfer_materials(before: Any, lineage: Any, args: object) -> Any:
        del lineage, args
        state = before.state("cell-density")
        return phx.equations.MaterialTransaction(
            (
                phx.equations.MaterialState(
                    state.site_id,
                    state.model_id,
                    state.committed[parents],
                    state_version=state.state_version + 1,
                ),
            )
        )

    def capture(rebind: CompositionRebind) -> CompositionRebind:
        rebinds.append(rebind)
        return rebind

    def certify(
        candidate: Any,
        fields: tuple[jax.Array, ...],
        material: Any,
        lineage: Any,
        args: float,
    ) -> bool:
        del candidate, lineage
        # Independent candidate solve and high-order physical quadrature; a changed
        # load genuinely violates the PDE and must leave every accepted owner intact.
        independent_context = phx.equations.FiniteElementExecutionContext(
            target_space.default_runtime, user_args=jnp.asarray(args)
        )
        recomputed = solve(target_problem, independent_context)
        error, integral = integrate(target_space, fields[0], args)
        update = float(jnp.max(jnp.abs(recomputed - fields[0])))
        history_error = float(
            jnp.max(
                jnp.abs(
                    fields[1] - (target_space.dof_maps[0].dof_coordinates[:, 1] + 0.5)
                )
            )
        )
        density_error = float(
            jnp.max(jnp.abs(material.state("cell-density").committed - 2.0))
        )
        reanalyses.append(
            {
                "physical_L2_error": error,
                "recomputed_solution_defect": update,
                "integral": integral,
                "history_error": history_error,
                "material_error": density_error,
            }
        )
        return (
            max(
                error,
                update,
                history_error,
                density_error,
                abs(integral - initial_integral),
            )
            < 1.0e-9
        )

    transaction = phx.solver.FiniteElementTopologyTransaction(
        certify,
        fields=(field, history),
        material_transfer=transfer_materials,
        composition_rebind=capture,
    )
    rejected = transaction.execute(accepted, source.mesh, adaptation, 2.0 * SHEAR)
    if (
        bool(rejected.committed)
        or rejected.state is not accepted
        or rejected.mesh is not source.mesh
    ):
        raise RuntimeError(
            "Failed physical reanalysis did not atomically retain the source."
        )
    result = transaction.execute(accepted, source.mesh, adaptation, SHEAR)
    if (
        not bool(result.committed)
        or result.receipt is None
        or not result.receipt.published
    ):
        raise RuntimeError(result.diagnostics)
    transfer = result.transfers[0]
    base_coordinates = jnp.asarray(base_coordinates)
    reference = jnp.asarray(((0.2, 0.2), (0.6, 0.1), (0.1, 0.7)), dtype=jnp.float64)
    barycentric = np.concatenate(
        (1.0 - np.asarray(reference).sum(axis=1, keepdims=True), np.asarray(reference)),
        axis=1,
    )
    coordinate_element = target.geometry.elements[0]
    tabulate = getattr(coordinate_element, "tabulate", None)
    if not callable(tabulate):
        raise TypeError("The shear-map oracle requires a tabulatable coordinate element.")
    basis, gradients = tabulate(reference)
    routes_array = np.asarray(target.geometry.geometry_dofs[0])
    coordinate_nodes = np.asarray(target.geometry.coordinates)[routes_array]
    mapped = np.asarray(
        phx.ein.contract("ql,cld->cqd", np.asarray(basis), coordinate_nodes)
    )
    target_corners = np.asarray(target.mesh.coordinates)[
        np.asarray(target.mesh.blocks[0].vertices)
    ]
    unsheared_corners = np.asarray(
        shear(jnp.asarray(target_corners), jnp.asarray(-SHEAR))
    )
    expected_reference = phx.ein.contract("ql,cld->cqd", barycentric, unsheared_corners)
    expected_map = np.asarray(shear(jnp.asarray(expected_reference), jnp.asarray(SHEAR)))
    geometry_oracle_error = float(np.max(np.abs(mapped - expected_map)))
    jacobians = np.asarray(
        phx.ein.contract("qld,cla->cqad", np.asarray(gradients), coordinate_nodes)
    )
    minimum_determinant = float(np.min(np.linalg.det(jacobians)))
    if geometry_oracle_error > 1.0e-12 or minimum_determinant <= 0.0:
        raise RuntimeError("The independent exact shear-map oracle failed.")

    def realized_coordinates(amplitude: jax.Array) -> jax.Array:
        return shear(base_coordinates, amplitude)

    def fixed_pde(amplitude: jax.Array) -> jax.Array:
        runtime = FiniteElementRuntimeData(
            source.mesh,
            realized_coordinates(amplitude),
            numeric_version="fixed-shear-epoch",
            geometry_layout_id=source.geometry.geometry_layout_id,
        )
        return solve(
            compiled,
            phx.equations.FiniteElementExecutionContext(runtime, user_args=amplitude),
        )

    def fixed_transfer(amplitude: jax.Array) -> jax.Array:
        # Nested reference restrictions do not change with this shear; this
        # qualified fixed donor map is deliberately not a retopology derivative.
        return transfer.transfer.apply(fixed_pde(amplitude))

    parameter = jnp.asarray(SHEAR, dtype=jnp.float64)
    stages = (realized_coordinates, fixed_pde, fixed_transfer)
    ids = (
        source.geometry.geometry_layout_id,
        compiled.compilation_id,
        transfer.transfer_id,
    )
    diagnostics = [
        numerical_derivative(stage, parameter, label=label)
        for label, stage in zip(("geometry", "pde", "transfer"), stages, strict=True)
    ]
    margins = (
        ("shear-design-bound", 0.25 - abs(SHEAR)),
        ("positive-map-determinant", minimum_determinant),
    )
    qualifications = tuple(
        DesignQualificationEvidence(
            DerivativeTier.Q1,
            source_space.prepared_id,
            source.result_id,
            plan_id,
            event_ids=tuple(name for name, _ in margins),
            event_margins=tuple(value for _, value in margins),
            gradient_error=diagnostic["finite_difference_error"],
            valid=True,
        )
        for plan_id, diagnostic in zip(ids, diagnostics, strict=True)
    )
    routes = tuple(zip(("geometry", "pde", "transfer"), ids, strict=True))
    derivative = FixedEpochDerivativeEvidence(
        source.result_id,
        "quadratic-shear-amplitude",
        geometry=qualifications[0],
        pde=qualifications[1],
        transfer=qualifications[2],
        routes=routes,
    )
    derivative.require_valid(
        source.result_id,
        "quadratic-shear-amplitude",
        routes=routes,
        event_margins=margins,
    )
    invalidation = derivative.invalidation_issues(
        target.result_id,
        "quadratic-shear-amplitude",
        routes=routes,
        event_margins=margins,
        topology_event_id=adaptation.result_id,
    )
    if "stopped-topology-event" not in invalidation:
        raise RuntimeError("Adaptation must invalidate the fixed-epoch derivative.")
    geometry_evidence = geometry_transition.evidence
    if (
        not geometry_evidence.exact
        or abs(geometry_evidence.target_measure - 1.0) > 1.0e-10
    ):
        raise RuntimeError("Native bisection did not preserve the quadratic shear map.")
    return {
        "source_cells": source.mesh.blocks[0].cell_count,
        "target_cells": target.mesh.blocks[0].cell_count,
        "geometry_order": 2,
        "field_order": 2,
        "geometry_exact": geometry_evidence.exact,
        "mapped_measure": geometry_evidence.target_measure,
        "initial_physical_error": initial_error,
        "geometry_oracle_error": geometry_oracle_error,
        "minimum_sampled_determinant": minimum_determinant,
        "material_total_content": 2.0 * geometry_evidence.target_measure,
        "failed_reanalysis": reanalyses[0],
        "accepted_reanalysis": reanalyses[1],
        "conservation_error": abs(reanalyses[1]["integral"] - initial_integral),
        "rollback_retained_all_state": rejected.state is accepted,
        "composition_receipt": result.receipt.receipt_id,
        "transferred_entries": rebinds[-1].source.entry_ids,
        "fixed_epoch_derivatives": dict(
            zip(("geometry", "pde", "transfer"), diagnostics, strict=True)
        ),
        "guard_margins": margins,
        "event_invalidation": invalidation,
        "derivative_evidence": derivative.evidence_id,
        "claim": "Small manufactured qualification only; no learned superiority claim.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
