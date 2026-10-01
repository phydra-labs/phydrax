# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Growing sphere: extensive native IMEX, tangential ALE shift, and atomic remap.

The uniform analytical solution is exp(-k*t)/R(t)^2. The fixed-edge positive
native sparse graph diffusion is conservative; this driver does not claim that
its edge conductances are a polynomial-exact Laplace--Beltrami discretization.
"""

from __future__ import annotations

import argparse
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.discretization._topology_epoch import TopologyEpoch
from phydrax.discretization.meshfree._capacity import (
    MeshfreeCapacityMap,
    MeshfreeCapacityPolicy,
)
from phydrax.discretization.meshfree._moving import (
    MovingGeometryRefresh,
    MovingSurfaceEpochResult,
    MovingSurfacePlan,
    MovingSurfaceState,
)
from phydrax.discretization.meshfree._neighbors import MeshfreeNeighborhoodPlan
from phydrax.discretization.meshfree._resampling import SurfaceResamplingPolicy
from phydrax.discretization.meshfree._shifting import (
    surface_relative_advection,
    SurfaceRelativeAdvectionResult,
    SurfaceShiftPolicy,
)
from phydrax.discretization.meshfree._stencils import (
    LocalStencilPolicy,
    MeshfreeFunctional,
    prepare_local_stencils,
)
from phydrax.discretization.meshfree._surface_transfer import (
    PreparedSurfaceTransfer,
    SurfaceTransferPlan,
)
from phydrax.graph import graph_to_cochain_complex, GraphIR
from phydrax.linalg import ArraySpace, FunctionLinearOperator
from phydrax.sparse import SparseCoordinateOperator


WorkflowMetric = float | int | bool | str | None


def _sphere_points(size: int, seed: int) -> np.ndarray:
    index = np.arange(size)
    z = 1 - 2 * (index + 0.5) / size
    angle = np.pi * (3 - np.sqrt(5)) * index + np.random.default_rng(seed).uniform(
        0, 2 * np.pi
    )
    radial = np.sqrt(1 - z * z)
    return np.column_stack((radial * np.cos(angle), radial * np.sin(angle), z))


def _prepare_plan(
    unit: np.ndarray,
    seed: int,
    /,
    *,
    epoch: TopologyEpoch | None = None,
    capacity: MeshfreeCapacityMap | None = None,
) -> tuple[MovingSurfacePlan, Array, str]:
    active = unit.shape[0]
    neighborhood = MeshfreeNeighborhoodPlan(unit, min(8, active - 1)).prepare()
    indices = np.asarray(neighborhood.relation.source_indices)
    row = np.broadcast_to(np.arange(active)[:, None], indices.shape)
    pairs = np.unique(
        np.sort(np.column_stack((row.reshape(-1), indices.reshape(-1))), axis=1), axis=0
    )
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    i, j = pairs[:, 0], pairs[:, 1]
    lengths = np.linalg.norm(unit[j] - unit[i], axis=1)
    conductance = 0.02 * (4 * np.pi / active) / lengths**2
    reference = jnp.asarray(unit)
    base_measures = jnp.full((active,), 4 * jnp.pi / active)
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

    def content_diffusion(concentration: Array) -> Array:
        return -base_measures * cochain.hodge_laplacian(0, concentration)

    space = ArraySpace((active,), dtype=np.float64)
    diffusion = FunctionLinearOperator(
        content_diffusion,
        transpose_action=content_diffusion,
        source=space,
        target=space,
        operator_id=f"{neighborhood.neighborhood_id}:native-content-diffusion",
    )
    growth, reaction_rate = 0.2, 0.1

    def refresh(
        state: MovingSurfaceState, time: Array, dt: Array, args: Any, /
    ) -> MovingGeometryRefresh:
        radius = 1 + growth * time
        points = reference * radius
        measures = base_measures * radius**2
        # Uniform dilation preserves every neighbor ordering EXACTLY. This is
        # a similarity certificate, not inverse-curvature reach estimation.
        similarity = jnp.max(jnp.abs(points / radius - reference)) <= 1e-12
        radial = jnp.max(jnp.abs(jnp.linalg.norm(points, axis=1) - radius)) <= 1e-12
        tube = (radius > 0) & radial  # Exact sphere signed-distance tube, on surface.
        return MovingGeometryRefresh(
            points,
            measures,
            reference,
            diffusion,
            2 * growth * radius * base_measures,
            similarity,
            tube,
            jnp.all(jnp.isfinite(points)),
        )

    def reaction(time: Array, points: Array, concentration: Array, args: Any, /) -> Array:
        return -reaction_rate * concentration

    if epoch is None:
        if capacity is None:
            raise ValueError(
                "Initial epoch preparation requires an explicit capacity map."
            )
        epoch = TopologyEpoch(
            0,
            f"sphere:{seed}:{active}",
            neighborhood.neighborhood_id,
            capacity.mapping_id,
        )
    plan = MovingSurfacePlan(
        refresh,
        reaction,
        epoch=epoch,
        history_capacity=12,
        plan_id=f"growing-sphere:{seed}:{neighborhood.neighborhood_id}",
    )
    return plan, jnp.asarray(pairs, dtype=jnp.int32), neighborhood.neighborhood_id


def prepare_workflow(
    *, size: int = 48, dimension: int = 3, seed: int = 0
) -> tuple[MovingSurfacePlan, MovingSurfaceState, Array]:
    """Separate bounded native relation preparation from device steps."""
    if dimension != 3:
        raise ValueError("Growing-sphere workflow requires ambient dimension=3.")
    if size < 16:
        raise ValueError("Workflow requires total capacity of at least 16.")
    active = max(12, 3 * size // 4)
    unit = _sphere_points(active, seed)
    capacity = MeshfreeCapacityPolicy((size,)).allocate(active)
    plan, pairs, _topology_id = _prepare_plan(unit, seed, capacity=capacity)
    epoch = plan.epoch
    state = plan.initialize(
        unit, np.full(active, 4 * np.pi / active), unit, np.ones(active), capacity, epoch
    )
    return plan, state, pairs


def reprepare_workflow(
    state: MovingSurfaceState, /, *, seed: int = 0
) -> MovingSurfacePlan:
    """Prepare the next fixed-epoch geometry/diffusion owner after host remap."""
    radius = 1 + 0.2 * float(np.asarray(state.time))
    plan, _pairs, _topology_id = _prepare_plan(
        np.asarray(state.points) / radius, seed, epoch=state.epoch
    )
    return plan


def _cross_transfer(
    old_points: Array, new_points: Array, old_measures: Array, new_measures: Array
) -> PreparedSurfaceTransfer:
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
    return SurfaceTransferPlan.from_stencils(
        stencil, old_measures, new_measures
    ).prepare()


def perform_shift(
    state: MovingSurfaceState, /, *, step_size: float = 0.01
) -> tuple[MovingSurfaceState, SurfaceRelativeAdvectionResult]:
    radius = 1 + 0.2 * float(np.asarray(state.time))
    neighborhood = MeshfreeNeighborhoodPlan(
        state.points, min(8, state.points.shape[0] - 1)
    ).prepare()
    indices = np.asarray(neighborhood.relation.source_indices)
    rows = np.broadcast_to(np.arange(state.points.shape[0])[:, None], indices.shape)
    pairs = np.unique(
        np.sort(np.column_stack((rows.reshape(-1), indices.reshape(-1))), axis=1), axis=0
    )
    pairs = jnp.asarray(pairs[pairs[:, 0] != pairs[:, 1]], dtype=jnp.int32)

    def project(points: Array) -> Array:
        return radius * points / jnp.linalg.norm(points, axis=1)[:, None]

    def residual(points: Array) -> Array:
        return jnp.linalg.norm(points, axis=1) - radius

    lengths = jnp.linalg.norm(
        state.points[pairs[:, 1]] - state.points[pairs[:, 0]], axis=1
    )
    shift = SurfaceShiftPolicy(
        strength=0.01,
        target_separation=1.2 * float(np.min(np.asarray(lengths))),
        maximum_displacement=0.01,
    ).propose(state.points, state.normals, pairs, step_size, project, residual)
    edge_metric = jnp.full((pairs.shape[0],), 0.1)
    graph = GraphIR(
        nodes=state.points,
        edges={"weight": edge_metric},
        senders=pairs[:, 0],
        receivers=pairs[:, 1],
        n_node=np.asarray([state.points.shape[0]]),
        n_edge=np.asarray([pairs.shape[0]]),
    )
    native = graph_to_cochain_complex(
        graph,
        edge_weight_key="weight",
        node_measure=state.measures,
        edge_semantics="undirected_once",
    )
    incidence = native.hilbert_complex().differential(0)
    if not isinstance(incidence, SparseCoordinateOperator):
        raise TypeError(
            "The native graph differential must retain sparse incidence structure."
        )
    advection = surface_relative_advection(
        state.concentration,
        state.measures,
        state.points,
        incidence,
        edge_metric,
        jnp.zeros_like(state.points),
        shift.mesh_velocity,
        step_size,
        require_positivity=True,
    )
    advection = eqx.tree_at(
        lambda a: a.successful, advection, shift.successful & advection.successful
    )
    if not bool(np.asarray(shift.successful & advection.successful)):
        return state, advection
    last = int(np.asarray(state.history_count)) - 1
    shifted = eqx.tree_at(
        lambda s: (s.points, s.normals, s.content, s.history_content, s.history_points),
        state,
        (
            shift.points,
            shift.points / radius,
            advection.content,
            state.history_content.at[last].set(advection.content),
            state.history_points.at[last].set(shift.points),
        ),
    )
    return shifted, advection


def perform_epoch_transition(
    plan: MovingSurfacePlan, state: MovingSurfaceState, /, *, seed: int = 0
) -> MovingSurfaceEpochResult:
    """Host resampling and complete historical transfer, separately callable."""
    radius = 1 + 0.2 * float(np.asarray(state.time))

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
        return MovingSurfaceEpochResult(state, False, jnp.zeros((1,), dtype=jnp.float64))
    new_points = repair.points
    new_measures = jnp.full(
        (new_points.shape[0],), 4 * jnp.pi * radius**2 / new_points.shape[0]
    )
    transfer = _cross_transfer(state.points, new_points, state.measures, new_measures)
    count = int(np.asarray(state.history_count))
    histories: list[PreparedSurfaceTransfer] = []
    history_points: list[Array] = []
    for index in range(count):
        historical_radius = 1 + 0.2 * state.history_times[index]
        target_points = new_points / radius * historical_radius
        target_measures = jnp.full(
            (new_points.shape[0],),
            4 * jnp.pi * historical_radius**2 / new_points.shape[0],
        )
        histories.append(
            _cross_transfer(
                state.history_points[index],
                target_points,
                state.history_measures[index],
                target_measures,
            )
        )
        history_points.append(target_points)
    next_epoch = state.epoch.index + 1
    target_epoch = TopologyEpoch(
        next_epoch,
        f"sphere:{seed}:resampled:{next_epoch}",
        transfer.transfer.transfer_id,
        repair.capacity.mapping_id,
    )
    return plan.transition_epoch(
        state,
        transfer,
        target_epoch,
        new_points,
        new_points / radius,
        repair.capacity,
        histories,
        jnp.stack(history_points),
    )


def run_workflow(
    *, size: int = 48, dimension: int = 3, seed: int = 0
) -> dict[str, WorkflowMetric]:
    plan, state, _pairs = prepare_workflow(size=size, dimension=dimension, seed=seed)
    dt = 0.01
    conservation = measure_residual = 0.0
    successful = True
    for _ in range(8):
        result = plan.step(state, dt)
        successful = successful and bool(np.asarray(result.evidence.successful))
        conservation = max(
            conservation, abs(float(np.asarray(result.evidence.conservation_residual)))
        )
        measure_residual = max(
            measure_residual, float(np.asarray(result.evidence.measure_rate_residual))
        )
        state = result.state
    time = float(np.asarray(state.time))
    radius = 1 + 0.2 * time
    exact = np.exp(-0.1 * time) / radius**2
    error = float(np.max(np.abs(np.asarray(state.concentration) - exact)))
    checkpoint = plan.checkpoint(state)
    attempted = plan.step(state, -dt)
    successful = successful and not bool(np.asarray(attempted.evidence.successful))
    rollback = plan.rollback(checkpoint)
    rollback_error = float(np.max(np.abs(np.asarray(rollback.content - state.content))))
    shift_residual = epoch_residual = 0.0
    initial_content = float(np.sum(np.asarray(state.content)))
    cycle_reaction_content = 0.0
    for _ in range(2):
        state, advection = perform_shift(state, step_size=dt)
        successful = successful and bool(np.asarray(advection.successful))
        shift_residual = max(
            shift_residual, abs(float(np.asarray(advection.conservation_residual)))
        )
        transitioned = perform_epoch_transition(plan, state, seed=seed)
        successful = successful and transitioned.successful
        epoch_residual = max(
            epoch_residual,
            float(np.max(np.abs(np.asarray(transitioned.conservation_residuals)))),
        )
        state = transitioned.state
        if transitioned.successful:
            plan = reprepare_workflow(state, seed=seed)
            continued = plan.step(state, dt)
            successful = successful and bool(np.asarray(continued.evidence.successful))
            cycle_reaction_content += float(
                np.asarray(continued.evidence.reaction_content)
            )
            conservation = max(
                conservation,
                abs(float(np.asarray(continued.evidence.conservation_residual))),
            )
            state = continued.state
    return {
        "size": size,
        "dimension": dimension,
        "seed": seed,
        "successful": successful,
        "domain": "growing sphere R(t)=1+0.2t in ambient R^3",
        "oracle_provenance": "analytical uniform exp(-0.1t)/R(t)^2 before remap; extensive source ledger during repeated epochs",
        "retained_bytes": None,
        "concentration_error": error,
        "conservation_residual": conservation,
        "measure_rate_residual": measure_residual,
        "shift_conservation_residual": shift_residual,
        "epoch_conservation_residual": epoch_residual,
        "repeated_cycle_conservation_residual": abs(
            float(np.sum(np.asarray(state.content)))
            - initial_content
            - cycle_reaction_content
        ),
        "rollback_error": rollback_error,
        "minimum_concentration": float(np.min(np.asarray(state.concentration))),
        "history_count": int(np.asarray(state.history_count)),
        "epoch": state.epoch.index,
        "active_capacity": state.capacity.active_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()
    print(run_workflow(size=options.size, dimension=options.dimension, seed=options.seed))


if __name__ == "__main__":
    main()
