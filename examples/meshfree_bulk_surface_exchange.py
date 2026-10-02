# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Conservative Langmuir equilibrium on an actual intrinsic sphere graph.

The amount solver explicitly chooses the positive Shepard reconstruction, which
reproduces constants only; no polynomial-exactness or monotonicity of signed MLS
is claimed. Bulk quadrature is deposition-compatible so the homogeneous
Langmuir solution is an independent analytical reference. The surface metric
is explicitly relaxed/nonnegative and its actual moment slack is reported.
Geometry queries relocate only at a host window boundary; the moving-query
ledger is not a differentiated within-window geometry solve.
"""

from __future__ import annotations

import argparse
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from benchmarks._runtime import logical_array_bytes
from phydrax import StrictModule
from phydrax.discretization import (
    PointCloudPlan,
    prepare_point_cloud_field_reconstruction,
)
from phydrax.discretization.meshfree import (
    ImplicitSurfaceGeometry,
    MeshfreeMetricPolicy,
    PreparedMeshfreeExteriorCalculus,
    PreparedSurfacePointCloud,
    SurfacePointCloudPlan,
    SurfaceQuadraturePolicy,
)
from phydrax.geometry import Ball
from phydrax.interfacial_transport import AdsorptionKinetics
from phydrax.metrix import RegularLevelSetManifold
from phydrax.solver.coupling import (
    ContributionEndpoint,
    CouplingWindow,
    LangmuirAdsorptionFlux,
    MeshfreeBulkSurfaceMethod,
    MeshfreeComponent,
    MethodParticipantState,
    SurfaceExchangeLaw,
)
from phydrax.sparse import SparseCoordinateOperator
from phydrax.typing import Dim, Float


class _WorkflowBulkDim(Dim):
    """Extensive bulk amount coordinates."""


class _WorkflowSurfaceDim(Dim):
    """Extensive surface amount coordinates."""


@final
class PreparedBulkSurfaceWorkflow(StrictModule):
    __strict_contract__ = True
    method: MeshfreeBulkSurfaceMethod
    bulk: MeshfreeComponent
    surface: PreparedSurfacePointCloud
    surface_component: MeshfreeComponent
    graph: PreparedMeshfreeExteriorCalculus
    law: SurfaceExchangeLaw
    initial: tuple[Float[_WorkflowBulkDim], Float[_WorkflowSurfaceDim]]
    equilibrium_bulk: float = eqx.field(static=True)
    equilibrium_surface: float = eqx.field(static=True)


def prepare_workflow(
    *, size: int = 128, dimension: int = 3, seed: int = 0
) -> PreparedBulkSurfaceWorkflow:
    """Host preparation; ``size`` is total bulk plus surface point capacity."""
    if dimension != 3:
        raise ValueError("The intrinsic sphere workflow requires dimension=3.")
    if isinstance(size, bool) or not isinstance(size, int) or size < 128:
        raise ValueError("size must be an integer total capacity of at least 128.")
    rng = np.random.default_rng(seed)
    count = size // 4
    z = 1.0 - 2.0 * (np.arange(count) + 0.5) / count
    angle = np.pi * (3.0 - np.sqrt(5.0)) * np.arange(count) + rng.uniform(0.0, 2 * np.pi)
    radius_xy = np.sqrt(1 - z * z)
    points = jnp.asarray(
        np.stack((radius_xy * np.cos(angle), radius_xy * np.sin(angle), z), axis=-1)
    )
    manifold = RegularLevelSetManifold(
        lambda x: jnp.reshape(jnp.sum(x * x) - 1.0, (1,)),
        ambient_dimension=3,
        codimension=1,
    )
    surface = SurfacePointCloudPlan(
        points,
        ImplicitSurfaceGeometry(manifold),
        min(16, count),
        quadrature=SurfaceQuadraturePolicy(
            "normalized-density", density=jnp.ones((count,)), total_area=4 * np.pi
        ),
    ).prepare()
    edge_radius = min(1.5, 5.0 / np.sqrt(count))
    graph = surface.conservative_exterior(
        edge_radius,
        maximum_pairs=min(count * 32, count * (count - 1) // 2),
        metric_policy=MeshfreeMetricPolicy(
            "nonnegative",
            acceptance="relaxed",
            tolerance=1e-7,
            slack_penalty=100.0,
            maximum_steps=4096,
        ),
    )
    query_radius = min(0.8, 4.0 / np.sqrt(count))
    bulk_count = size - count
    shells = [
        np.asarray(surface.points) * (1.0 - layer * query_radius / 3.0)
        for layer in range(3)
    ]
    bulk_points = np.concatenate(shells)
    remainder = bulk_count - bulk_points.shape[0]
    if remainder:
        bulk_points = np.concatenate(
            (
                bulk_points,
                np.asarray(surface.points)[:remainder] * (1.0 - query_radius / 6.0),
            )
        )
    support = Ball((0.0, 0.0, 0.0), 1.05).compile()
    volume = 4 * np.pi / 3
    provisional = PointCloudPlan(
        bulk_points,
        np.full((bulk_count,), volume / bulk_count),
        neighbors=min(36, bulk_count),
    ).prepare()
    view = prepare_point_cloud_field_reconstruction(
        provisional,
        support_geometry=support,
        radius=query_radius,
        capacity=min(128, bulk_count),
        reconstruction="shepard",
    )
    route = view.prepare_query(graph.points)
    deposition = np.asarray(route.transpose(surface.measures))
    if not np.all(deposition > 0):
        raise ValueError(
            "The analytical workflow requires coverage of every bulk deposition site."
        )
    volumes = volume * deposition / np.sum(deposition)
    cloud = PointCloudPlan(bulk_points, volumes, neighbors=min(36, bulk_count)).prepare()
    view = prepare_point_cloud_field_reconstruction(
        cloud,
        support_geometry=support,
        radius=query_radius,
        capacity=min(128, bulk_count),
        reconstruction="shepard",
    )
    query = view.prepare_query(graph.points)
    weights = sum(pair[1] for pair in cloud.derivative_weights)
    full = cloud.field_spaces[0].vector_space
    native = SparseCoordinateOperator(
        cloud.relation, -jnp.asarray(volumes)[:, None] * weights, source=full, target=full
    )
    bulk = MeshfreeComponent(
        cloud, native, view, volumes, name="bulk", owner_id=cloud.prepared_id
    )
    surface_view = surface.prepare_field_reconstruction(support_geometry=support)
    laplace = surface.laplace_beltrami
    surface_native = SparseCoordinateOperator(
        laplace.relation,
        -surface.measures[:, None] * laplace.coefficients,
        source=laplace.source,
        target=laplace.target,
    )
    surface_component = MeshfreeComponent(
        surface,
        surface_native,
        surface_view,
        surface.measures,
        name="surface",
        owner_id=surface_view.reconstruction_id,
    )
    kinetics = AdsorptionKinetics(2.0, 0.5, 1.0)
    method = MeshfreeBulkSurfaceMethod(
        query, graph, volumes, kinetics, surface_diffusivity=0.02
    )
    law = SurfaceExchangeLaw(
        ContributionEndpoint("bulk", "concentration"),
        ContributionEndpoint("surface", "concentration"),
        query,
        surface.measures,
        surface.normals,
        LangmuirAdsorptionFlux(kinetics),
    )
    initial = (jnp.asarray(volumes), 0.1 * surface.measures)
    total = volume + 0.1 * 4 * np.pi
    equilibrium_constant = 4.0
    a = volume * equilibrium_constant
    b = volume + 4 * np.pi * equilibrium_constant - total * equilibrium_constant
    bulk_reference = 2 * total / (b + np.sqrt(b * b + 4 * a * total))
    surface_reference = (
        equilibrium_constant
        * bulk_reference
        / (1 + equilibrium_constant * bulk_reference)
    )
    return PreparedBulkSurfaceWorkflow(
        method,
        bulk,
        surface,
        surface_component,
        graph,
        law,
        initial,
        float(bulk_reference),
        float(surface_reference),
    )


def run_workflow(
    *, size: int = 128, dimension: int = 3, seed: int = 0
) -> dict[str, float | int | bool | str]:
    prepared = prepare_workflow(size=size, dimension=dimension, seed=seed)
    participant = prepared.method.participant(substeps=1)
    initial = participant.initial_state(prepared.initial)

    def window_step(
        state: MethodParticipantState, index: Array
    ) -> tuple[MethodParticipantState, tuple[Array, Array, Array, Array]]:
        start = index.astype(jnp.float64) * 5.0
        result = participant.advance_window(
            CouplingWindow(index, start, start + 5.0), state, (), None
        )
        # Native amount steps already reject inadmissible candidates atomically.
        next_state = jax.lax.cond(
            result.successful, lambda _: result.candidate_state, lambda _: state, None
        )
        total = jnp.sum(next_state.native[0]) + jnp.sum(next_state.native[1])
        return next_state, (
            result.successful,
            result.residual_norm,
            result.iterations,
            total,
        )

    final, ledger = eqx.filter_jit(
        lambda state: jax.lax.scan(window_step, state, jnp.arange(80, dtype=jnp.int32))
    )(initial)
    successful, residuals, iterations, totals = ledger
    bulk = final.native[0] / prepared.method.bulk_volumes
    surface = final.native[1] / prepared.method.surface_measures
    reference_error = jnp.maximum(
        jnp.max(jnp.abs(bulk - prepared.equilibrium_bulk)),
        jnp.max(jnp.abs(surface - prepared.equilibrium_surface)),
    )
    flux = prepared.method.transport.structure.exchange_flux(bulk, surface)
    initial_total = jnp.sum(prepared.initial[0]) + jnp.sum(prepared.initial[1])
    lowered = prepared.law.prepare(
        {"bulk": prepared.bulk, "surface": prepared.surface_component}, ()
    )
    law_defects = lowered.certificate.defects(
        {("bulk", "concentration"): bulk, ("surface", "concentration"): surface},
        (),
        {},
    )
    # Rigid rotation is an actual source-preserving geometry change, prepared at
    # the host boundary. No topology relocation occurs inside the numerical scan.
    theta = 1e-3
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    moved_points = np.asarray(prepared.surface.points) @ rotation.T
    moved_surface = SurfacePointCloudPlan(
        jnp.asarray(moved_points),
        prepared.surface.plan.geometry,
        prepared.surface.plan.neighbors,
        quadrature=prepared.surface.plan.quadrature,
    ).prepare()
    moved_view = moved_surface.prepare_field_reconstruction(
        support_geometry=prepared.bulk.reconstruction.support_geometry,
    )
    moved_laplace = moved_surface.laplace_beltrami
    moved_operator = SparseCoordinateOperator(
        moved_laplace.relation,
        -moved_surface.measures[:, None] * moved_laplace.coefficients,
        source=moved_laplace.source,
        target=moved_laplace.target,
    )
    moved_component = MeshfreeComponent(
        moved_surface,
        moved_operator,
        moved_view,
        moved_surface.measures,
        name="surface",
        owner_id=moved_view.reconstruction_id,
    )
    moved_components = {"bulk": prepared.bulk, "surface": moved_component}
    refreshed = prepared.law.refresh_at_window(
        moved_components,
        moved_surface.points,
        moved_surface.measures,
        moved_surface.normals,
        window_start=400.0,
        geometry_time=399.95,
        geometry_epoch=1,
        maximum_displacement=0.01,
        maximum_lag=0.1,
    )
    refreshed_prepared = refreshed.prepare(moved_components, ())
    refreshed_defects = refreshed_prepared.certificate.defects(
        {("bulk", "concentration"): bulk, ("surface", "concentration"): surface},
        (),
        {},
    )
    refreshed_density = refreshed.flux.evaluate(
        refreshed.query.apply(bulk),
        surface,
        refreshed.query.points,
        refreshed.normals,
        None,
    )
    refreshed_rows = refreshed.query.transpose(refreshed.measures * refreshed_density)
    refreshed_balance = jnp.abs(
        jnp.sum(refreshed_rows) - jnp.sum(refreshed.measures * refreshed_density)
    )
    return {
        "equilibrium_error": float(reference_error),
        "conservation_error": float(
            jnp.maximum(jnp.max(jnp.abs(totals - initial_total)), refreshed_balance)
        ),
        "flux_error": float(jnp.max(jnp.abs(flux))),
        "accepted": bool(
            jnp.all(successful)
            & law_defects.accepted(1e-10)
            & refreshed_defects.accepted(1e-10)
        ),
        "nonlinear_residual": float(jnp.max(residuals)),
        "nonlinear_iterations": int(jnp.sum(iterations)),
        "points": size,
        "window_lag_error": abs(refreshed.evidence.lag - (400.0 - 399.95)),
        "window_lag": refreshed.evidence.lag,
        "query_displacement": refreshed.evidence.displacement,
        "query_constant_error": refreshed.evidence.constant_reproduction_error,
        "metric_moment_slack": float(
            jnp.max(jnp.abs(prepared.graph.metric_result.normalized_residual))
        ),
        "metric_exact": bool(prepared.graph.metric_result.exact),
        "positive_deposition": bool(jnp.all(prepared.method.query.route.weights >= 0)),
        "geometry_differentiated_within_window": refreshed.evidence.geometry_differentiated_within_window,
        "domain": "unit sphere interface and deposition-compatible unit-ball bulk",
        "oracle_provenance": "Langmuir isotherm plus closed total-amount quadratic, independent of Newton iterations",
        "retained_bytes": logical_array_bytes(prepared),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(run_workflow(size=args.size, dimension=args.dimension, seed=args.seed))


if __name__ == "__main__":
    main()
