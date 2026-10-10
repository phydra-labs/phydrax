# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Adaptive meshfree refinement through epoch transactions, and learned corrections.

Bulk: ``-Laplace(u) = f`` on the unit square with the nonpolynomial boundary
layer ``u = exp(-x / eps) (1 + sin(pi y) / 4)``. Each adaptive level computes
the strong residual at off-node probes, marks with Dörfler bulk chasing,
proposes deterministic edge-midpoint children (boundary children are
projected onto the square), transfers the solution with the joint
conservative/constant-preserving transfer, solves on the candidate cloud,
and commits or rejects one ``adaptive-refinement`` epoch. A coarsening-only
candidate that raises the indicator is rejected and publishes nothing; a
capacity-refused proposal never becomes an epoch.

Surface: ``-Laplace_S(u) + u = f`` on the unit sphere with the zonal peak
``u = exp(kappa x . a)``; the indicator is the degree-3 versus degree-4 PHS
Laplace-Beltrami difference, children are projected onto the sphere.

Both are compared with uniform refinement at equal point count (and the
observed wall time is reported). Indicators are indicators: no reliability
bound is claimed; errors are measured against the analytic solutions.

Learned correction: an edge-feature model's candidate metric weights are
projected onto the full moment-constraint set. The signed minimum-norm
projection keeps every moment exact but breaks the positivity margin, so it is
refused (no weights published); the explicitly selected nonnegative conic
correction restores the margin with every moment still exact.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import TypedDict

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.linalg as la
from examples.meshfree_learned_metric_correction import (
    run_training as run_correction_training,
    TrainingReport,
)
from phydrax.discretization import (
    PointBoundaryCondition,
    PointBoundaryPlan,
    PointCloudPlan,
    PointCloudPoissonPlan,
    PointCollocationStability,
    PreparedPointCloudDiscretization,
    TopologyEpoch,
    TransferGeometryBinding,
)
from phydrax.discretization.meshfree import (
    adaptation_acceptance,
    commit_meshfree_epoch,
    degree_difference_indicator,
    ImplicitSurfaceGeometry,
    LocalStencilPolicy,
    mark_points,
    meshfree_fill_measures,
    MeshfreeAdaptationAcceptance,
    MeshfreeAdaptationPolicy,
    MeshfreeAdaptationProposal,
    MeshfreeAdaptiveSupport,
    MeshfreeEpochChange,
    MeshfreeErrorIndicator,
    MeshfreeExteriorCalculusPlan,
    MeshfreeMarkingPolicy,
    MeshfreeMetricCorrectionPlan,
    MeshfreeMetricCorrectionPolicy,
    MeshfreeProbeJet,
    prepare_adaptation_transfer,
    PreparedSurfacePointCloud,
    probe_residual_indicator,
    propose_adaptation,
    stage_meshfree_epoch,
    SurfaceEllipticSystem,
    SurfacePointCloudPlan,
    SurfaceQuadraturePolicy,
)
from phydrax.lifecycle import Composition, CompositionEntry, CompositionRole
from phydrax.metrix import RegularLevelSetManifold


class AdaptationTransaction(TypedDict):
    composition: Composition
    published: bool
    values: Array | None
    seconds: float
    acceptance: MeshfreeAdaptationAcceptance
    transfer_status: str
    conservation_residual: float
    indicator_after: MeshfreeErrorIndicator | None


class CoarseningReport(TypedDict):
    published: bool
    refusals: str
    unchanged: bool


class BulkReport(TypedDict):
    bulk_uniform_points: list[int]
    bulk_uniform_max_errors: list[float]
    bulk_uniform_solve_seconds: list[float]
    bulk_adaptive_points: list[int]
    bulk_adaptive_max_errors: list[float]
    bulk_adaptive_cumulative_solve_seconds: list[float]
    bulk_epochs_published: list[bool]
    bulk_epoch_refusals: list[str]
    bulk_epoch_stability: list[str]
    bulk_uniform_stability: list[str]
    bulk_uniform_error_at_equal_points: float
    bulk_adaptive_gain_at_equal_points: float
    bulk_rejected_published: bool
    bulk_rejected_refusals: str
    bulk_rejected_source_unchanged: bool
    bulk_capacity_status: str


class SurfaceReport(TypedDict):
    surface_uniform_points: list[int]
    surface_uniform_relative_l2: list[float]
    surface_uniform_solve_seconds: list[float]
    surface_adaptive_points: list[int]
    surface_adaptive_relative_l2: list[float]
    surface_proposal_statuses: list[str]
    surface_uniform_error_at_equal_points: float
    surface_adaptive_gain_at_equal_points: float


class LearnedReport(TrainingReport):
    learned_signed_status: str
    learned_signed_moments_exact: bool
    learned_signed_max_moment_residual: float
    learned_signed_sign_margin: float
    learned_signed_weights_published: bool
    learned_constrained_status: str
    learned_constrained_provider: str
    learned_constrained_max_moment_residual: float
    learned_constrained_min_weight: float
    learned_constrained_coercive: bool
    learned_constrained_derivative: bool


class AdaptiveWorkflowReport(BulkReport, SurfaceReport, LearnedReport):
    pass


EPSILON = 0.04
KAPPA = 6.0
OWNER = "adaptive-boundary-layer"


# --- Bulk boundary layer -----------------------------------------------------------------


def layer_exact(points: np.ndarray) -> np.ndarray:
    return np.exp(-points[:, 0] / EPSILON) * (1.0 + 0.25 * np.sin(np.pi * points[:, 1]))


def layer_source(points: Array | np.ndarray) -> Array:
    x, y = jnp.asarray(points)[:, 0], jnp.asarray(points)[:, 1]
    profile = 1.0 + 0.25 * jnp.sin(jnp.pi * y)
    return -jnp.exp(-x / EPSILON) * (
        profile / EPSILON**2 - 0.25 * jnp.pi**2 * jnp.sin(jnp.pi * y)
    )


def layer_residual(points: Array, jet: MeshfreeProbeJet) -> Array:
    return -(jet.hessian[:, 0, 0] + jet.hessian[:, 1, 1]) - layer_source(points)


def square_cloud(count: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jittered Cartesian cloud with exact boundary rows and outward normals."""
    axis = np.linspace(0.0, 1.0, count)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    boundary = np.any((points == 0.0) | (points == 1.0), axis=1)
    spacing = 1.0 / (count - 1)
    jitter = np.random.default_rng(seed).uniform(
        -0.15 * spacing, 0.15 * spacing, (int(np.sum(~boundary)), 2)
    )
    points[~boundary] += jitter
    normals = np.zeros_like(points)
    for axis_index in range(2):
        normals[:, axis_index] = np.where(
            points[:, axis_index] == 0.0,
            -1.0,
            np.where(points[:, axis_index] == 1.0, 1.0, 0.0),
        )
    normals[boundary] /= np.linalg.norm(normals[boundary], axis=1, keepdims=True)
    return points, boundary, normals


def square_projection(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest point on the unit-square boundary and its outward side normal."""
    gaps = np.stack(
        (points[:, 0], 1.0 - points[:, 0], points[:, 1], 1.0 - points[:, 1]), axis=1
    )
    side = np.argmin(gaps, axis=1)
    projected, normals = points.copy(), np.zeros_like(points)
    for index, (axis, value, sign) in enumerate(
        ((0, 0.0, -1.0), (0, 1.0, 1.0), (1, 0.0, -1.0), (1, 1.0, 1.0))
    ):
        rows = side == index
        projected[rows, axis] = value
        normals[rows, axis] = sign
    return projected, normals


def bulk_cloud(
    points: np.ndarray,
    boundary: np.ndarray,
    normals: np.ndarray,
    ids: np.ndarray,
    /,
    *,
    degree: int = 3,
    neighbors: int = 30,
) -> PreparedPointCloudDiscretization:
    """Cubic-augmented PHS stencils of about three times the basis size.

    Candidate clouds mask refused rows so refusal is evidence, not an exception.
    """
    return PointCloudPlan(
        jnp.asarray(points),
        jnp.asarray(meshfree_fill_measures(points, 1.0, intrinsic_dimension=2)),
        boundary_mask=jnp.asarray(boundary),
        boundary_normals=jnp.asarray(normals),
        stencil=LocalStencilPolicy(
            approximation="phs-rbf-fd",
            polynomial_degree=degree,
            phs_power=3,
            acceptance="mask",
        ),
        neighbors=neighbors,
        point_ids=jnp.asarray(ids),
        maximum_candidates=min(points.shape[0], 1024),
    ).prepare()


def solve_layer(
    cloud: PreparedPointCloudDiscretization,
) -> tuple[Array, bool, PointCollocationStability | None, float]:
    """Square collocation with the owner's spectral stability as evidence."""
    points = np.asarray(cloud.points)
    boundary = np.asarray(cloud.plan.boundary_mask)
    rows = np.flatnonzero(boundary)
    start = time.perf_counter()
    plan = PointCloudPoissonPlan(
        cloud,
        PointBoundaryPlan(
            (
                PointBoundaryCondition(
                    "dirichlet", rows, layer_exact(points)[rows], label="walls"
                ),
            ),
            row_count=points.shape[0],
        ),
        linear_policy=la.LinearSolvePolicy(
            la.GMRES(restart=min(40, points.shape[0])),
            tolerance=la.TolerancePolicy(relative=1e-9, absolute=1e-10, max_steps=1000),
            preconditioning=la.PreconditioningPolicy(
                la.ILUPreconditionerBuilder(), refresh="numeric"
            ),
            failure=la.FailurePolicy("status"),
        ),
        stability="diagnostic",
    )
    prepared = plan.prepare(1.0)
    result = prepared.solve(layer_source(points))
    return (
        result.values,
        bool(result.successful),
        prepared.stability,
        time.perf_counter() - start,
    )


def layer_error(cloud: PreparedPointCloudDiscretization, values: Array) -> float:
    return float(
        np.max(np.abs(np.asarray(values) - layer_exact(np.asarray(cloud.points))))
    )


def _entry(
    value: object,
    entry_id: str,
    role: CompositionRole,
    structure: str,
    revision: str,
    *dependencies: CompositionEntry,
) -> CompositionEntry:
    return CompositionEntry(
        value,
        entry_id=entry_id,
        role=role,
        owner_id=OWNER,
        structure_id=structure,
        revision_id=revision,
        semantics_id=entry_id.split("/")[0],
        dependencies=tuple(item.binding("structure") for item in dependencies),
    )


def layer_composition(
    epoch: TopologyEpoch, cloud: PreparedPointCloudDiscretization, values: Array
) -> Composition:
    root = _entry(epoch, "cloud/epoch", "topology", epoch.epoch_id, "initial")
    return Composition(
        (
            root,
            _entry(
                values,
                "field/solution",
                "physical-state",
                epoch.epoch_id,
                "solution",
                root,
            ),
            _entry(
                cloud,
                "support/cloud",
                "discretization",
                epoch.epoch_id,
                cloud.prepared_id,
                root,
            ),
        ),
        boundary_id=f"adaptive-{epoch.index}",
    )


def _epoch(index: int) -> TopologyEpoch:
    return TopologyEpoch(index, f"square-cloud-{index}", "boundary-layer", "serial")


def adaptation_transaction(
    source: Composition,
    proposal: MeshfreeAdaptationProposal,
    indicator: MeshfreeErrorIndicator,
    /,
) -> AdaptationTransaction:
    """Candidate cloud, transfer, solve, indicator, acceptance and epoch commit."""
    cloud = source.value("support/cloud")
    epoch = source.value("cloud/epoch")
    if not isinstance(cloud, PreparedPointCloudDiscretization):
        raise TypeError("Adaptive bulk composition requires a prepared point cloud.")
    if not isinstance(epoch, TopologyEpoch):
        raise TypeError("Adaptive bulk composition requires a topology epoch.")
    target_epoch = _epoch(epoch.index + 1)
    candidate = bulk_cloud(
        np.asarray(proposal.target_points),
        np.asarray(proposal.target_boundary),
        np.asarray(proposal.target_normals),
        np.asarray(proposal.target_ids),
        degree=proposal.target_degree,
        neighbors=proposal.target_neighbors,
    )
    transfer = prepare_adaptation_transfer(
        cloud,
        proposal,
        geometry=TransferGeometryBinding(
            epoch.geometry_id,
            target_epoch.geometry_id,
            "topology-correspondence",
            source_topology_id=epoch.topology_id,
            target_topology_id=target_epoch.topology_id,
            coverage_defect=None,
        ),
    )
    if candidate.report.refused_rows == 0:
        values, successful, stability, seconds = solve_layer(candidate)
        after = probe_residual_indicator(candidate, values, layer_residual)
    else:
        values, successful, stability, seconds, after = None, False, None, 0.0, None
    acceptance = adaptation_acceptance(
        proposal,
        transfer,
        candidate.report,
        solve_successful=successful,
        stability=stability,
        indicator_before=indicator if after is not None else None,
        indicator_after=after,
    )
    change = MeshfreeEpochChange(
        epoch, target_epoch, cause="adaptive-refinement", proposal=proposal
    )
    root = CompositionEntry(
        target_epoch,
        entry_id="cloud/epoch",
        role="topology",
        owner_id=OWNER,
        structure_id=target_epoch.epoch_id,
        revision_id=change.change_id,
        semantics_id="cloud",
    )
    rebuilt = _entry(
        candidate,
        "support/cloud",
        "discretization",
        target_epoch.epoch_id,
        candidate.prepared_id,
        root,
    )
    remap = (
        {}
        if not transfer.admitted
        else {"field/solution": transfer.epoch_transition(epoch, target_epoch)}
    )
    if transfer.admitted:
        staged = stage_meshfree_epoch(
            source, change, epoch_entry="cloud/epoch", remap=remap, reprepare=(rebuilt,)
        )
        receipt = commit_meshfree_epoch(staged, accepted_boundary=acceptance.accepted)
        composition, published = receipt.composition, receipt.published
        residual = float(np.max(np.abs(np.asarray(receipt.conservation_residuals))))
    else:
        composition, published, residual = source, False, math.inf
    return {
        "composition": composition,
        "published": published,
        "values": values,
        "seconds": seconds,
        "acceptance": acceptance,
        "transfer_status": transfer.evidence.status.name,
        "conservation_residual": residual,
        "indicator_after": None if after is None else after,
    }


def run_bulk(
    *, start: int = 16, levels: int = 4, uniform: tuple[int, ...] = (16, 24, 34, 48)
) -> BulkReport:
    uniform_points, uniform_errors, uniform_seconds, uniform_stability = [], [], [], []
    for count in uniform:
        points, boundary, normals = square_cloud(count)
        cloud = bulk_cloud(points, boundary, normals, np.arange(points.shape[0]))
        values, successful, stability, seconds = solve_layer(cloud)
        if not successful:
            raise RuntimeError(f"Uniform solve with {points.shape[0]} points failed.")
        uniform_points.append(points.shape[0])
        uniform_errors.append(layer_error(cloud, values))
        uniform_seconds.append(seconds)
        uniform_stability.append("unassessed" if stability is None else stability.outcome)

    points, boundary, normals = square_cloud(start)
    cloud = bulk_cloud(points, boundary, normals, np.arange(points.shape[0]))
    values, successful, _, seconds = solve_layer(cloud)
    if not successful:
        raise RuntimeError("Initial adaptive solve failed.")
    composition = layer_composition(_epoch(0), cloud, values)
    adaptive_points, adaptive_errors = [points.shape[0]], [layer_error(cloud, values)]
    adaptive_seconds, published, refusals, stabilities = [seconds], [], [], []
    marking_policy = MeshfreeMarkingPolicy("dorfler", fraction=0.9, maximum_marked=800)
    adaptation = MeshfreeAdaptationPolicy(total_measure=1.0, maximum_points=8_000)
    for _ in range(levels):
        indicator = probe_residual_indicator(cloud, values, layer_residual)
        marking = mark_points(indicator, marking_policy)
        proposal = propose_adaptation(
            MeshfreeAdaptiveSupport.from_point_cloud(cloud),
            indicator,
            marking,
            adaptation,
            boundary_projection=square_projection,
        )
        outcome = adaptation_transaction(composition, proposal, indicator)
        published.append(bool(outcome["published"]))
        refusals.append(",".join(outcome["acceptance"].refusals))
        stabilities.append(str(outcome["acceptance"].stability_outcome))
        if not outcome["published"]:
            break
        composition = outcome["composition"]
        cloud = composition.value("support/cloud")
        if not isinstance(cloud, PreparedPointCloudDiscretization):
            raise TypeError(
                "Published adaptive composition requires a prepared point cloud."
            )
        next_values = outcome["values"]
        if next_values is None:
            raise RuntimeError(
                "A published adaptive epoch requires a solved candidate field."
            )
        values = next_values
        adaptive_points.append(int(np.asarray(cloud.points).shape[0]))
        adaptive_errors.append(layer_error(cloud, values))
        adaptive_seconds.append(adaptive_seconds[-1] + float(outcome["seconds"]))

    rejected = _rejected_coarsening(composition, values)
    capacity = _capacity_refusal(cloud, values)
    equal = _interpolated_error(uniform_points, uniform_errors, adaptive_points[-1])
    return {
        "bulk_uniform_points": uniform_points,
        "bulk_uniform_max_errors": uniform_errors,
        "bulk_uniform_solve_seconds": uniform_seconds,
        "bulk_adaptive_points": adaptive_points,
        "bulk_adaptive_max_errors": adaptive_errors,
        "bulk_adaptive_cumulative_solve_seconds": adaptive_seconds,
        "bulk_epochs_published": published,
        "bulk_epoch_refusals": refusals,
        "bulk_epoch_stability": stabilities,
        "bulk_uniform_stability": uniform_stability,
        "bulk_uniform_error_at_equal_points": equal,
        "bulk_adaptive_gain_at_equal_points": equal / adaptive_errors[-1],
        "bulk_rejected_published": rejected["published"],
        "bulk_rejected_refusals": rejected["refusals"],
        "bulk_rejected_source_unchanged": rejected["unchanged"],
        "bulk_capacity_status": capacity,
    }


def _rejected_coarsening(composition: Composition, values: Array) -> CoarseningReport:
    """Blind coarsening raises the residual indicator and is refused whole.

    A zero indicator marks nothing and makes every interior point a removal
    candidate; the staged epoch is judged against the residual indicator.
    """
    cloud = composition.value("support/cloud")
    if not isinstance(cloud, PreparedPointCloudDiscretization):
        raise TypeError("Adaptive coarsening requires a prepared point cloud.")
    indicator = probe_residual_indicator(cloud, values, layer_residual)
    blind = MeshfreeErrorIndicator(
        np.zeros(indicator.values.shape[0]),
        indicator.stable_ids,
        kind="probe-residual",
        probe_count=0,
    )
    proposal = propose_adaptation(
        MeshfreeAdaptiveSupport.from_point_cloud(cloud),
        blind,
        mark_points(blind, MeshfreeMarkingPolicy()),
        MeshfreeAdaptationPolicy(
            total_measure=1.0, coarsen_fraction=1.0, maximum_removed=10_000
        ),
        boundary_projection=square_projection,
    )
    outcome = adaptation_transaction(composition, proposal, indicator)
    return {
        "published": bool(outcome["published"]),
        "refusals": ",".join(outcome["acceptance"].refusals),
        "unchanged": outcome["composition"] is composition,
    }


def _capacity_refusal(cloud: PreparedPointCloudDiscretization, values: Array) -> str:
    """A proposal beyond its declared point capacity is refused before staging."""
    indicator = probe_residual_indicator(cloud, values, layer_residual)
    proposal = propose_adaptation(
        MeshfreeAdaptiveSupport.from_point_cloud(cloud),
        indicator,
        mark_points(indicator, MeshfreeMarkingPolicy("dorfler", fraction=0.9)),
        MeshfreeAdaptationPolicy(total_measure=1.0, maximum_inserted=4),
        boundary_projection=square_projection,
    )
    return proposal.status.name


def _interpolated_error(points: list[int], errors: list[float], count: int) -> float:
    """Log-log interpolation of the uniform error curve at ``count`` points."""
    return float(np.exp(np.interp(np.log(count), np.log(points), np.log(errors))))


# --- Curved surface ----------------------------------------------------------------------


# The peak axis avoids the Fibonacci poles, whose spiral sampling is locally
# irregular; the field is zonal about this axis.
PEAK_AXIS = np.asarray([0.6, 0.0, 0.8])


def sphere_exact(points: np.ndarray) -> np.ndarray:
    return np.exp(KAPPA * (points @ PEAK_AXIS))


def sphere_source(points: np.ndarray) -> np.ndarray:
    """``-Laplace_S(u) + u`` for the zonal ``u(t)``, ``t = x . a``:
    ``Laplace_S u = (1 - t^2) u'' - 2 t u'``."""
    t = points @ PEAK_AXIS
    laplace = np.exp(KAPPA * t) * (KAPPA**2 * (1.0 - t * t) - 2.0 * KAPPA * t)
    return -laplace + sphere_exact(points)


def fibonacci_sphere(count: int) -> np.ndarray:
    index = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / count
    azimuth = index * math.pi * (3.0 - math.sqrt(5.0))
    radius = np.sqrt(1.0 - z * z)
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))


def sphere_projection(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = points / np.linalg.norm(points, axis=1, keepdims=True)
    return unit, unit


def sphere_marking() -> MeshfreeMarkingPolicy:
    """Maximum-fraction marking of the steep cap as a whole.

    The degree-difference indicator of ``exp(kappa t)`` spans about ``e^12``,
    so Dörfler bulk chasing marks only a handful of points at the very top.
    The resulting small refined caps put their 2:1 transition bands where the
    solution derivatives are still large; irregular PHS stencils in those
    bands (stencil amplification grew about fourfold) inject coherent
    truncation error that the global screened solve spreads, and the error
    rose (Dörfler 0.5/0.7 measured 0.76x and 0.37x the uniform accuracy at
    equal points). Marking a fixed fraction refines the steep region whole.
    """
    return MeshfreeMarkingPolicy("maximum", fraction=0.2)


def sphere_adaptation(*, maximum_points: int) -> MeshfreeAdaptationPolicy:
    """Three children per refined point and grading 2.5 keep two levels inside
    the declared cap.

    The library default grading 2.05 widens graded bands, which removed the
    bulk boundary-layer stall, but on the sphere the second level then exceeds
    the 768-point cap (measured: CAPACITY_REFUSED after 192 -> 364 points, gain
    0.92 at equal points), while grading 2.5 measured 192 -> 355 -> 747 points
    with gain 1.37. Wider bands cost points; under a fixed cap the declared
    grading is the trade-off.
    """
    return MeshfreeAdaptationPolicy(
        total_measure=4.0 * math.pi,
        children_per_point=3,
        grading=2.5,
        maximum_points=maximum_points,
    )


def sphere_cloud(
    points: np.ndarray, degree: int, neighbors: int
) -> PreparedSurfacePointCloud:
    def constraint(point: Array) -> Array:
        return jnp.asarray([jnp.dot(point, point) - 1.0])

    measures = meshfree_fill_measures(points, 4.0 * math.pi, intrinsic_dimension=2)
    return SurfacePointCloudPlan(
        jnp.asarray(points),
        ImplicitSurfaceGeometry(
            RegularLevelSetManifold(
                constraint, ambient_dimension=3, codimension=1, manifold_id="unit-sphere"
            ),
            certified_tube_radius=0.5,
            geometry_id="unit-sphere",
        ),
        neighbors=neighbors,
        quadrature=SurfaceQuadraturePolicy("supplied", measures=jnp.asarray(measures)),
        stencil_policy=LocalStencilPolicy(
            approximation="phs-rbf-fd",
            polynomial_degree=degree,
            phs_power=3,
            chunk_rows=64,
        ),
    ).prepare()


def solve_sphere(surface: PreparedSurfacePointCloud) -> tuple[Array, float]:
    points = np.asarray(surface.points)
    start = time.perf_counter()
    system = SurfaceEllipticSystem(
        (surface,), diffusivity=1.0, reaction=1.0, system_id="screened-sphere"
    )
    result = la.solve(
        system.linear_system,
        system.rhs((jnp.asarray(sphere_source(points)),), (None,)),
        policy=la.LinearSolvePolicy(
            la.GMRES(restart=min(200, points.shape[0])),
            tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=4000),
            failure=la.FailurePolicy("status"),
        ),
    )
    if not bool(result.successful):
        raise RuntimeError(f"Sphere solve failed with status {int(result.status)}.")
    return system.split(result.value)[0], time.perf_counter() - start


def sphere_error(surface: PreparedSurfacePointCloud, values: Array) -> float:
    """Measure-weighted relative L2 error against the analytic field."""
    exact = sphere_exact(np.asarray(surface.points))
    measures = np.asarray(surface.measures)
    difference = np.asarray(values) - exact
    return float(np.sqrt(np.sum(measures * difference**2) / np.sum(measures * exact**2)))


def run_surface(
    *, start: int = 256, levels: int = 3, uniform: tuple[int, ...] = (256, 512, 1024)
) -> SurfaceReport:
    neighbors = 30
    uniform_errors, uniform_seconds = [], []
    for count in uniform:
        surface = sphere_cloud(fibonacci_sphere(count), 3, neighbors)
        values, seconds = solve_sphere(surface)
        uniform_errors.append(sphere_error(surface, values))
        uniform_seconds.append(seconds)

    points = fibonacci_sphere(start)
    ids = np.arange(start)
    adaptive_points, adaptive_errors, statuses = [], [], []
    for level in range(levels + 1):
        surface = sphere_cloud(points, 3, neighbors)
        values, _ = solve_sphere(surface)
        adaptive_points.append(points.shape[0])
        adaptive_errors.append(sphere_error(surface, values))
        if level == levels:
            break
        higher = sphere_cloud(points, 4, neighbors)
        measures = np.asarray(surface.measures)
        indicator = degree_difference_indicator(
            surface.laplace_beltrami.mv, higher.laplace_beltrami.mv, values, measures, ids
        )
        support = MeshfreeAdaptiveSupport(
            np.asarray(surface.points),
            ids,
            kind="surface",
            intrinsic_dimension=2,
            degree=3,
            neighbors=neighbors,
        )
        proposal = propose_adaptation(
            support,
            indicator,
            mark_points(indicator, sphere_marking()),
            sphere_adaptation(maximum_points=uniform[-1]),
            manifold_projection=sphere_projection,
        )
        statuses.append(proposal.status.name)
        if not proposal.admitted:
            break
        points = np.asarray(proposal.target_points)
        ids = np.asarray(proposal.target_ids)
    equal = _interpolated_error(list(uniform), uniform_errors, adaptive_points[-1])
    return {
        "surface_uniform_points": list(uniform),
        "surface_uniform_relative_l2": uniform_errors,
        "surface_uniform_solve_seconds": uniform_seconds,
        "surface_adaptive_points": adaptive_points,
        "surface_adaptive_relative_l2": adaptive_errors,
        "surface_proposal_statuses": statuses,
        "surface_uniform_error_at_equal_points": equal,
        "surface_adaptive_gain_at_equal_points": equal / adaptive_errors[-1],
    }


# --- Learned metric correction -----------------------------------------------------------


def run_learned(
    *, size: int = 6, seed: int = 0, margin: float = 1e-2, training_steps: int = 6
) -> LearnedReport:
    """Learned candidate -> full-moment projection -> audits -> refusal -> conic
    route, then training of a learned correction through the projection and the
    implicit conservation adjoint (``examples/meshfree_learned_metric_correction.py``)."""
    spacing = 1.0 / (size - 1)
    grid = (
        np.stack(
            np.meshgrid(np.arange(size), np.arange(size), indexing="ij"), axis=-1
        ).reshape((-1, 2))
        * spacing
    )
    interior = np.all((grid > 1e-9) & (grid < 1.0 - 1e-9), axis=1)
    jitter = np.random.default_rng(seed).uniform(
        -0.1 * spacing, 0.1 * spacing, grid.shape
    )
    points = grid + interior[:, None] * jitter
    exterior = MeshfreeExteriorCalculusPlan(
        points,
        2.05 * spacing,
        16 * size * size,
        node_volumes=np.full(size * size, spacing * spacing),
        dirichlet=~interior,
    ).prepare()
    lengths = np.asarray(exterior.lengths)
    # A frozen edge-feature model stands in for any learned law: its output is
    # only a candidate, never trusted before projection and audit.
    candidate = np.asarray(exterior.metric_result.weights) + 1e-2 * np.tanh(
        3.0 * lengths / spacing
    )
    signed = MeshfreeMetricCorrectionPlan(
        exterior, policy=MeshfreeMetricCorrectionPolicy(margin=margin)
    ).correct(candidate)
    constrained = MeshfreeMetricCorrectionPlan(
        exterior, policy=MeshfreeMetricCorrectionPolicy(margin=margin, sign="constrained")
    ).correct(candidate)
    return {
        "learned_signed_status": signed.status.name,
        "learned_signed_moments_exact": signed.evidence.moments_exact,
        "learned_signed_max_moment_residual": signed.evidence.maximum_moment_residual,
        "learned_signed_sign_margin": signed.evidence.sign_margin,
        "learned_signed_weights_published": signed.weights is not None,
        "learned_constrained_status": constrained.status.name,
        "learned_constrained_provider": constrained.evidence.provider,
        "learned_constrained_max_moment_residual": (
            constrained.evidence.maximum_moment_residual
        ),
        "learned_constrained_min_weight": constrained.evidence.minimum_weight,
        "learned_constrained_coercive": constrained.evidence.coercive,
        "learned_constrained_derivative": constrained.evidence.derivative_available,
        **run_correction_training(size=size, seed=seed, steps=training_steps),
    }


def run_workflow(*, quick: bool = False) -> AdaptiveWorkflowReport:
    bulk = run_bulk(start=12, levels=2, uniform=(12, 17, 24, 34)) if quick else run_bulk()
    surface = (
        run_surface(start=192, levels=2, uniform=(192, 384, 768))
        if quick
        else run_surface()
    )
    return {**bulk, **surface, **run_learned()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(run_workflow(quick=arguments.quick), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
