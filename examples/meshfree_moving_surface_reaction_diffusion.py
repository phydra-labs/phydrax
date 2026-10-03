# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Growing sphere: stage-correct native IMEX, tangential ALE shift, atomic remap.

The sphere follows the authoritative chart R(t)=1+0.2t. Stage geometry is the
native surface refresh modulo similarity on a smooth fixed-radius support
envelope, so measures, normals and curvature are evaluated at every stage of
the selected IMEX tableau and the support trust is the declared displacement
envelope. The uniform analytical solution is exp(-k*t)/R(t)^2. The mesh shift
is a tangential mesh velocity whose relative content flux is integrated by the
same native stage rule. The fixed-edge positive native sparse graph diffusion
is conservative; this driver does not claim that its edge conductances are a
polynomial-exact Laplace--Beltrami discretization.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, get_args, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.discretization import CochainDiscretization, TopologyEpoch
from phydrax.discretization.meshfree import (
    ChartMotion,
    ImplicitSurfaceGeometry,
    LocalStencilPolicy,
    MeshfreeCapacityMap,
    MeshfreeCapacityPolicy,
    MeshfreeFunctional,
    MeshfreeMetricPolicy,
    MeshfreeNeighborhoodPlan,
    MovingGeometryRefresh,
    MovingSurfaceEpochResult,
    MovingSurfaceEvidence,
    MovingSurfacePlan,
    MovingSurfaceState,
    PointTransferRequest,
    prepare_local_stencils,
    PreparedPointTransfer,
    PreparedSurfacePointCloud,
    PrescribedVelocityMotion,
    SmoothSupportEnvelope,
    SurfaceGeometryProvider,
    SurfaceMeshShift,
    SurfacePointCloudPlan,
    SurfaceQuadraturePolicy,
    SurfaceResamplingPolicy,
    SurfaceShiftPolicy,
    SurfaceTransferPlan,
)
from phydrax.graph import graph_to_cochain_complex, GraphIR
from phydrax.linalg import ArraySpace, FunctionLinearOperator
from phydrax.metrix import RegularLevelSetManifold
from phydrax.solver.advanced import AdditiveIMEXScheme


WorkflowMetric = float | int | bool | str | None

GROWTH = 0.2
REACTION_RATE = 0.1

_COMPILED_STEP = eqx.filter_jit(MovingSurfacePlan.step)


def _radius(time: float | Array) -> float | Array:
    return 1 + GROWTH * time


def _sphere_points(size: int, seed: int) -> np.ndarray:
    index = np.arange(size)
    z = 1 - 2 * (index + 0.5) / size
    angle = np.pi * (3 - np.sqrt(5)) * index + np.random.default_rng(seed).uniform(
        0, 2 * np.pi
    )
    radial = np.sqrt(1 - z * z)
    return np.column_stack((radial * np.cos(angle), radial * np.sin(angle), z))


def _spacing(count: int, /) -> float:
    """Mean sample spacing of ``count`` points on the unit sphere."""
    return math.sqrt(4 * math.pi / count)


def _unit_surface(unit: np.ndarray, /) -> PreparedSurfacePointCloud:
    def constraint(point: Array) -> Array:
        return jnp.asarray([jnp.dot(point, point) - 1])

    count = unit.shape[0]
    spacing = _spacing(count)
    source = RegularLevelSetManifold(
        constraint, ambient_dimension=3, codimension=1, manifold_id="unit-sphere"
    )
    # A smooth fixed-radius support: stencil weights stay smooth while samples
    # move, and refresh admits any anchored displacement below the envelope.
    envelope = SmoothSupportEnvelope(1.75 * spacing, 0.25 * spacing)
    return SurfacePointCloudPlan(
        jnp.asarray(unit),
        ImplicitSurfaceGeometry(
            source, certified_tube_radius=0.5, geometry_id="unit-sphere"
        ),
        min(48, count),
        quadrature=SurfaceQuadraturePolicy(
            "normalized-density",
            density=jnp.ones(count, dtype=jnp.float64),
            total_area=4 * math.pi,
        ),
        stencil_policy=LocalStencilPolicy(polynomial_degree=2, support=envelope),
        require_tube=True,
    ).prepare()


def _edge_pairs(points: Array | np.ndarray, /) -> np.ndarray:
    count = points.shape[0]
    neighborhood = MeshfreeNeighborhoodPlan(points, min(8, count - 1)).prepare()
    indices = np.asarray(neighborhood.relation.source_indices)
    rows = np.broadcast_to(np.arange(count)[:, None], indices.shape)
    pairs = np.unique(
        np.sort(np.column_stack((rows.reshape(-1), indices.reshape(-1))), axis=1), axis=0
    )
    return pairs[pairs[:, 0] != pairs[:, 1]]


def _content_diffusion(
    cochain: CochainDiscretization, measures: Array, concentration: Array
) -> Array:
    return -measures * cochain.hodge_laplacian(0, concentration)


def _chart(reference: Array, time: Array) -> Array:
    return _radius(time) * reference


def _reaction(time: Array, points: Array, concentration: Array, args: Any, /) -> Array:
    return -REACTION_RATE * concentration


def _prepare_plan(
    unit: np.ndarray,
    seed: int,
    /,
    *,
    epoch_index: int,
    capacity: MeshfreeCapacityMap,
    method: AdditiveIMEXScheme,
) -> MovingSurfacePlan:
    surface = _unit_surface(unit)
    reference = surface.points
    pairs = _edge_pairs(reference)
    i, j = pairs[:, 0], pairs[:, 1]
    active = reference.shape[0]
    lengths = np.linalg.norm(np.asarray(reference)[j] - np.asarray(reference)[i], axis=1)
    base_measures = surface.measures
    conductance = 0.02 * (4 * np.pi / active) / lengths**2
    graph = GraphIR(
        nodes=reference,
        edges={"conductance": jnp.asarray(conductance)},
        senders=i,
        receivers=j,
        n_node=[active],
        n_edge=[pairs.shape[0]],
    )
    cochain = graph_to_cochain_complex(
        graph,
        edge_weight_key="conductance",
        node_measure=base_measures,
        edge_semantics="undirected_once",
    )

    # Two-dimensional extensive graph diffusion is invariant under dilation.
    content_diffusion = eqx.Partial(_content_diffusion, cochain, base_measures)

    space = ArraySpace((active,), dtype=np.float64)
    diffusion = FunctionLinearOperator(
        content_diffusion,
        transpose_action=content_diffusion,
        source=space,
        target=space,
        operator_id=f"{surface.neighborhood.neighborhood_id}:native-cochain-content-action",
    )

    geometry = SurfaceGeometryProvider(surface, diffusion, mode="similarity")

    chart = eqx.Partial(_chart, reference)

    epoch = TopologyEpoch(
        epoch_index,
        f"sphere:{seed}:{active}",
        surface.neighborhood.neighborhood_id,
        capacity.mapping_id,
    )
    return MovingSurfacePlan(
        geometry,
        ChartMotion(
            chart,
            law_id=f"growing-sphere-chart:{surface.neighborhood.neighborhood_id}",
        ),
        _reaction,
        method=method,
        epoch=epoch,
        capacity=capacity,
        history_capacity=12,
        plan_id=f"growing-sphere-native-cochain:{seed}:{surface.neighborhood.neighborhood_id}",
    )


def prepare_workflow(
    *,
    size: int = 48,
    dimension: int = 3,
    seed: int = 0,
    method: AdditiveIMEXScheme = "ars-222",
) -> tuple[MovingSurfacePlan, MovingSurfaceState]:
    """Separate bounded native relation preparation from device steps."""
    if dimension != 3:
        raise ValueError("Growing-sphere workflow requires ambient dimension=3.")
    if size < 32:
        raise ValueError("Workflow requires total capacity of at least 32.")
    active = 3 * size // 4
    capacity = MeshfreeCapacityPolicy((size,)).allocate(active)
    plan = _prepare_plan(
        _sphere_points(active, seed),
        seed,
        epoch_index=0,
        capacity=capacity,
        method=method,
    )
    chart = plan.motion
    if not isinstance(chart, ChartMotion):
        raise TypeError("The growing-sphere workflow uses its authoritative chart.")
    state = plan.initialize(chart.positions(jnp.asarray(0.0)), np.ones(active))
    return plan, state


def _cross_transfer(
    old_points: Array, old_measures: Array, target: MovingGeometryRefresh
) -> PreparedPointTransfer:
    new_points, new_measures = target.points, target.measures
    neighborhood = MeshfreeNeighborhoodPlan(
        old_points, min(8, old_points.shape[0]), targets=new_points
    ).prepare()
    stencil = prepare_local_stencils(
        neighborhood,
        old_points,
        new_points,
        (
            MeshfreeFunctional(
                ((0, 0, 0),), jnp.asarray([1.0]), name="cross-target-value"
            ),
        ),
        LocalStencilPolicy(polynomial_degree=1),
    )
    prepared = SurfaceTransferPlan.from_stencils(
        stencil,
        old_measures,
        new_measures,
        target_normals=target.normals,
        request=PointTransferRequest("conservative-signed"),
    ).prepare()
    if not prepared.admitted:
        raise ValueError(f"Cross-epoch transfer refused: {prepared.evidence.status}.")
    return prepared


def prepare_shift_plan(
    plan: MovingSurfacePlan, /, *, strength: float = 1.0
) -> MovingSurfacePlan:
    """Same-epoch runtime that grows the sphere while redistributing its mesh.

    The chart is replaced by its radial velocity ``0.2 x/|x|`` with integrated
    coordinates, so the tangential mesh shift can move them; the relative
    content flux uses the surface exterior's exact nonnegative metric. The
    conservative measure law advances the measures of record with the same
    graph divergence as that flux, so the stage-quadrature GCL is admitted by
    construction and the refreshed measures report the geometric drift.
    """
    provider = plan.geometry
    if not isinstance(provider, SurfaceGeometryProvider):
        raise TypeError("The growing-sphere workflow uses a surface geometry provider.")
    surface = provider.surface
    count = surface.points.shape[0]
    exterior = surface.conservative_exterior(
        2.5 * _spacing(count),
        count * (count - 1) // 2,
        metric_policy=MeshfreeMetricPolicy(sign="nonnegative"),
    )
    lengths = np.asarray(exterior.lengths)
    lengths = lengths[lengths > 0]
    shift = SurfaceMeshShift(
        SurfaceShiftPolicy(
            strength=strength, target_separation=1.1 * float(np.min(lengths))
        ),
        exterior,
    )

    def growth(time: Array, points: Array, args: Any) -> Array:
        return GROWTH * points / jnp.linalg.norm(points, axis=1)[:, None]

    return MovingSurfacePlan(
        provider,
        PrescribedVelocityMotion(
            growth, tangential="normal-only", law_id="growing-sphere-velocity"
        ),
        plan.reaction,
        method=plan.tableau,
        epoch=plan.epoch,
        capacity=plan.capacity,
        history_capacity=plan.history_capacity,
        plan_id=f"{plan.plan_id}:shift",
        shift=shift,
        require_positivity=True,
        measure_law="conservative",
    )


def perform_shift(
    shift_plan: MovingSurfacePlan,
    state: MovingSurfaceState,
    /,
    *,
    steps: int = 4,
    step_size: ArrayLike = 0.01,
) -> tuple[MovingSurfaceState, list[MovingSurfaceEvidence]]:
    """Grow and redistribute the mesh: one native stage rule for all of it."""
    evidence = []
    dt = jnp.asarray(step_size, dtype=state.content.dtype)
    for _ in range(steps):
        result = _COMPILED_STEP(shift_plan, state, dt)
        evidence.append(result.evidence)
        state = result.state
    return state, evidence


class EpochRoutes(NamedTuple):
    """Frozen next-epoch routes in ``MovingSurfacePlan.transition_epoch`` order."""

    transfer: PreparedPointTransfer
    target_epoch: TopologyEpoch
    points: Array
    normals: Array
    capacity: MeshfreeCapacityMap
    histories: list[PreparedPointTransfer]
    history_points: Array


def prepare_epoch_transition(
    state: MovingSurfaceState,
    /,
    *,
    seed: int = 0,
    method: AdditiveIMEXScheme = "ars-222",
) -> tuple[MovingSurfacePlan, EpochRoutes] | None:
    """Host resampling, next-epoch preparation and every live-history route.

    Target measures of the current state and of every live history come from
    the next epoch's native geometry, so its stage-quadrature GCL starts
    consistent. ``None`` reports a nonconverged resampling.
    """
    time = float(np.asarray(state.time))
    radius = _radius(time)

    def project(points: Array) -> Array:
        return radius * points / jnp.linalg.norm(points, axis=1)[:, None]

    def residual(points: Array) -> Array:
        return jnp.linalg.norm(points, axis=1) - radius

    neighbors = MeshfreeNeighborhoodPlan(state.points, 2).prepare()
    nearest = np.asarray(neighbors.relation.source_indices)[:, 1]
    row = (seed + state.epoch.index) % state.points.shape[0]
    midpoint = project((state.points[row] + state.points[nearest[row]])[None, :] / 2)
    probes = jnp.concatenate((state.points, midpoint), axis=0)
    gap = float(
        np.min(np.linalg.norm(np.asarray(state.points) - np.asarray(midpoint)[0], axis=1))
    )

    def quality(points: Array) -> tuple[Array, Array]:
        neighborhood = MeshfreeNeighborhoodPlan(points, min(8, points.shape[0])).prepare()
        stencil = prepare_local_stencils(
            neighborhood,
            points,
            points,
            (MeshfreeFunctional(((0, 0, 0),), jnp.asarray([1.0])),),
            LocalStencilPolicy(polynomial_degree=1, acceptance="mask"),
        )
        return stencil.evidence.condition, stencil.evidence.amplification

    repair = SurfaceResamplingPolicy(
        maximum_fill=0.5 * gap, minimum_separation=0.1 * gap, maximum_iterations=8
    ).repair(
        state.points,
        probes,
        MeshfreeCapacityPolicy((state.capacity.capacity,)),
        project,
        residual,
        quality,
    )
    if not repair.converged:
        return None
    target_plan = _prepare_plan(
        np.asarray(repair.points) / radius,
        seed,
        epoch_index=state.epoch.index + 1,
        capacity=repair.capacity,
        method=method,
    )
    chart = target_plan.motion
    if not isinstance(chart, ChartMotion):
        raise TypeError("The growing-sphere workflow uses its authoritative chart.")
    current = target_plan.geometry(chart.positions(state.time), state.time, None)
    transfer = _cross_transfer(state.points, state.measures, current)
    histories: list[PreparedPointTransfer] = []
    history_points: list[Array] = []
    for slot in state.live_slots():
        historical = target_plan.geometry(
            chart.positions(state.history_times[slot]), state.history_times[slot], None
        )
        histories.append(
            _cross_transfer(
                state.history_points[slot], state.history_measures[slot], historical
            )
        )
        history_points.append(historical.points)
    return target_plan, EpochRoutes(
        transfer,
        target_plan.epoch,
        current.points,
        current.normals,
        repair.capacity,
        histories,
        jnp.stack(history_points),
    )


def perform_epoch_transition(
    plan: MovingSurfacePlan,
    state: MovingSurfaceState,
    /,
    *,
    seed: int = 0,
    method: AdditiveIMEXScheme = "ars-222",
) -> tuple[MovingSurfacePlan, MovingSurfaceEpochResult]:
    """Atomic cutover of the current field and every live history."""
    prepared = prepare_epoch_transition(state, seed=seed, method=method)
    if prepared is None:
        return plan, MovingSurfaceEpochResult(
            state, False, jnp.zeros((1,), dtype=jnp.float64)
        )
    target_plan, routes = prepared
    result = plan.transition_epoch(state, *routes)
    return (target_plan if result.successful else plan), result


def run_workflow(
    *,
    size: int = 48,
    dimension: int = 3,
    seed: int = 0,
    method: AdditiveIMEXScheme = "ars-222",
) -> dict[str, WorkflowMetric]:
    plan, state = prepare_workflow(
        size=size, dimension=dimension, seed=seed, method=method
    )
    dt = jnp.asarray(0.01, dtype=state.content.dtype)
    conservation = measure_residual = gcl_defect = 0.0
    successful = True
    for _ in range(8):
        result = _COMPILED_STEP(plan, state, dt)
        successful = successful and bool(np.asarray(result.evidence.successful))
        conservation = max(
            conservation, abs(float(np.asarray(result.evidence.conservation_residual)))
        )
        measure_residual = max(
            measure_residual, float(np.asarray(result.evidence.measure_rate_residual))
        )
        gcl_defect = max(gcl_defect, float(np.asarray(result.evidence.gcl_defect)))
        state = result.state
    time = float(np.asarray(state.time))
    exact = np.exp(-REACTION_RATE * time) / _radius(time) ** 2
    error = float(np.max(np.abs(np.asarray(state.concentration) - exact)))
    checkpoint = plan.checkpoint(state)
    attempted = _COMPILED_STEP(plan, state, -dt)
    successful = successful and not bool(np.asarray(attempted.evidence.successful))
    rollback = plan.rollback(checkpoint)
    rollback_error = float(np.max(np.abs(np.asarray(rollback.content - state.content))))
    shift_residual = shift_gcl = shift_drift = shift_cfl = epoch_residual = 0.0
    shift_minimum = math.inf
    initial_content = float(np.sum(np.asarray(state.content)))
    cycle_reaction_content = 0.0
    for _ in range(2):
        state, shifts = perform_shift(prepare_shift_plan(plan), state, step_size=dt)
        for item in shifts:
            successful = successful and bool(np.asarray(item.successful))
            cycle_reaction_content += float(np.asarray(item.reaction_content))
            shift_residual = max(
                shift_residual, abs(float(np.asarray(item.conservation_residual)))
            )
            shift_gcl = max(shift_gcl, float(np.asarray(item.gcl_defect)))
            shift_drift = max(shift_drift, float(np.asarray(item.geometric_drift)))
            shift_cfl = max(shift_cfl, float(np.asarray(item.transport_cfl)))
            shift_minimum = min(
                shift_minimum, float(np.asarray(item.minimum_concentration))
            )
        plan, transitioned = perform_epoch_transition(
            plan, state, seed=seed, method=method
        )
        successful = successful and transitioned.successful
        epoch_residual = max(
            epoch_residual,
            float(np.max(np.abs(np.asarray(transitioned.conservation_residuals)))),
        )
        state = transitioned.state
        if transitioned.successful:
            continued = _COMPILED_STEP(plan, state, dt)
            successful = successful and bool(np.asarray(continued.evidence.successful))
            cycle_reaction_content += float(
                np.asarray(continued.evidence.reaction_content)
            )
            conservation = max(
                conservation,
                abs(float(np.asarray(continued.evidence.conservation_residual))),
            )
            gcl_defect = max(gcl_defect, float(np.asarray(continued.evidence.gcl_defect)))
            state = continued.state
    return {
        "size": size,
        "dimension": dimension,
        "seed": seed,
        "method": method,
        "successful": successful,
        "domain": "growing sphere R(t)=1+0.2t in ambient R^3",
        "oracle_provenance": "analytical uniform exp(-0.1t)/R(t)^2 before remap; extensive source ledger during repeated epochs",
        "retained_bytes": None,
        "concentration_error": error,
        "conservation_residual": conservation,
        "measure_rate_residual": measure_residual,
        "gcl_defect": gcl_defect,
        "shift_conservation_residual": shift_residual,
        "shift_gcl_defect": shift_gcl,
        "shift_geometric_drift": shift_drift,
        "shift_transport_cfl": shift_cfl,
        "shift_minimum_concentration": shift_minimum,
        "epoch_conservation_residual": epoch_residual,
        "repeated_cycle_conservation_residual": abs(
            float(np.sum(np.asarray(state.content)))
            - initial_content
            - cycle_reaction_content
        ),
        "rollback_error": rollback_error,
        "minimum_concentration": float(np.min(np.asarray(state.concentration))),
        "history_count": int(np.asarray(state.history_count)),
        "accepted_steps": int(np.asarray(state.accepted_steps)),
        "epoch": state.epoch.index,
        "active_capacity": state.capacity.active_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--method",
        choices=get_args(AdditiveIMEXScheme),
        default="ars-222",
    )
    options = parser.parse_args()
    print(
        run_workflow(
            size=options.size,
            dimension=options.dimension,
            seed=options.seed,
            method=options.method,
        )
    )


if __name__ == "__main__":
    main()
