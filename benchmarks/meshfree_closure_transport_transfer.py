# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Q9 bulk evolution/transport and Q13 joint transfer/physical topology workloads.

Every workload calls the public meshfree facade, measures its phases separately
with the shared ``PhaseRecorder`` and returns independent scientific evidence
beside timing:

- bulk ADR on periodic GMLS clouds against the exact advection-diffusion-
  relaxation mode (spatial order on measured fill distance, temporal order
  against a fine SSPRK54 reference of the same semidiscrete system);
- conservative graph transport (upwind/reconstructed/limited) with a declared
  inflow pulse, closed swirl conservation/positivity, the ALE/GCL ledger and
  the declared CFL/positivity/support/topology refusals;
- audited point transfers (signed, positive, joint moments) on tensor Simpson
  quadratures and their declared coverage/obstruction/infeasibility refusals;
- committed multiregion merge and pinch events consumed as one all-history
  meshfree epoch transaction, with rollback.

Workloads never set JAX configuration; the driver owns precision.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from benchmarks._runtime import logical_array_bytes
from benchmarks.meshfree_scaling import (
    declare_reservation,
    fill_distance,
    MeshfreeConfig,
    PhaseRecorder,
    unit_cube_probes,
)
from phydrax.discretization import PointCloudPlan, PreparedPointCloudDiscretization
from phydrax.discretization.meshfree import (
    commit_meshfree_epoch,
    ConservativeTransport,
    LocalStencilPolicy,
    MeshfreeDiffusionLaw,
    MeshfreeEpochReceipt,
    MeshfreeEvolutionPlan,
    MeshfreeEvolutionStatus,
    MeshfreeExteriorCalculusPlan,
    MeshfreeFunctional,
    MeshfreeMotion,
    MeshfreeNeighborhoodPlan,
    MeshfreeReactionLaw,
    MeshfreeSheetSupport,
    MeshfreeSurfaceEventEpoch,
    PointTransferPlan,
    PointTransferRequest,
    PointTransferStatus,
    prepare_local_stencils,
    PreparedMeshfreeEvolution,
    PreparedMeshfreeExteriorCalculus,
    PreparedPointTransfer,
    stage_meshfree_epoch,
    surface_event_epoch,
    TransportRefreshStatus,
    TransportScheme,
)
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.geometry.multiregion_surface import (
    apply_surface_events,
    MergeProposal,
    MultiRegionRemeshPlan,
    MultiRegionSurfaceCapacityPlan,
    MultiRegionSurfaceSeed,
    MultiRegionSurfaceState,
    MultiRegionSurfaceTopology,
    PreparedMultiRegionSurface,
    propose_merges,
    propose_pinches,
    propose_remesh,
    RegionSplitProposal,
    seed_catenoid,
    seed_sphere,
    SurfaceEventPassResult,
    SurfaceEventPolicy,
    SurfaceMergePolicy,
)
from phydrax.lifecycle import Composition, CompositionEntry
from phydrax.solver import FixedStepProblem, solve_fixed_step
from phydrax.sparse import EdgeRelation


BulkTemporalMethod: TypeAlias = Literal["ssprk33", "ars-222"]
ReferenceMethod: TypeAlias = Literal["ssprk33", "ssprk54", "ars-222"]
TransferCase: TypeAlias = Literal[
    "conservative-signed",
    "conservative-positive",
    "joint-nonnegative-constant",
    "joint-signed-degree2",
]
TransferRefusal: TypeAlias = Literal[
    "uncovered-source",
    "uncovered-target",
    "measure-obstruction",
    "infeasible-left-null",
    "infeasible-farkas",
]
type FieldLaw = Callable[[Array, Array, Any], Array]


def _result(
    workload: str,
    capacity: int,
    requested: int,
    seed: int,
    dimension: int,
    recorder: PhaseRecorder,
    metrics: dict[str, Any],
    retained: Any,
    /,
    *,
    oracle: str,
    consumer: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "workload": workload,
        "capacity": capacity,
        "requested_capacity": requested,
        "seed": seed,
        "dimension": dimension,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": metrics,
        "retained_bytes": logical_array_bytes(retained),
        "oracle_provenance": oracle,
        "consumer": consumer,
        **extra,
    }


def _require_float64(config: MeshfreeConfig, workload: str, /) -> None:
    if config.precision != "float64":
        raise ValueError(
            f"{workload} is declared only for float64 data; got {config.precision}."
        )


def _require_dimension(config: MeshfreeConfig, dimension: int, workload: str, /) -> None:
    _require_float64(config, workload)
    if config.dimension != dimension:
        raise ValueError(
            f"{workload} is declared in dimension {dimension}; got {config.dimension}."
        )


def _measured_fill(points: np.ndarray, seed: int, provenance: str, /) -> dict[str, Any]:
    cloud = np.asarray(points, dtype=np.float64)
    return fill_distance(
        cloud,
        unit_cube_probes(cloud.shape[1], 4 * cloud.shape[0], seed),
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Q9 collocation ADR: exact advection-diffusion-relaxation mode
#
# c = 1 + 0.5 exp(-(8 pi^2 k + r) t) sin(2 pi (x - u t)) cos(2 pi (y - v t))
# solves c_t + u . grad c = k lap c - r (c - 1) on the periodic unit square.

_ADR_VELOCITY = np.asarray([1.0, 0.5], dtype=np.float64)
_ADR_DIFFUSIVITY = 0.05
_ADR_RELAXATION = 1.0
_ADR_FINAL_TIME = 0.2
_ADR_JITTER = 0.2
_PERIODIC = MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 10, periodic_axes=(True, True))
# SSPRK33 real-axis absolute-stability interval [-2.5127, 0]. The 2-D
# five-point Laplacian radius 8 k / h^2 sets the explicit diffusion step
# 2.5127 h^2 / (8 k); jittered GMLS stencils may exceed it, so a declared
# half of that step is used. The native power-iteration radius of the implicit
# part is recorded beside it as an estimate only (it is not a bound).
_SSPRK33_REAL_INTERVAL = 2.5127
_EXPLICIT_FRACTION = 0.5
_PERIODIC_FILL = (
    "max Euclidean nearest-node distance over scrambled Sobol probes of [0,1]^2 "
    "(non-periodic nearest; an upper estimate of the periodic fill distance)"
)


def adr_exact(time: Array | float, points: Array, /) -> Array:
    shifted = points - time * jnp.asarray(_ADR_VELOCITY, dtype=jnp.float64)
    decay = jnp.exp(-(8.0 * jnp.pi**2 * _ADR_DIFFUSIVITY + _ADR_RELAXATION) * time)
    return 1.0 + 0.5 * decay * jnp.sin(2.0 * jnp.pi * shifted[:, 0]) * jnp.cos(
        2.0 * jnp.pi * shifted[:, 1]
    )


def _relaxation(time: Array, points: Array, value: Array, args: Any) -> Array:
    del time, points, args
    return -_ADR_RELAXATION * (value - 1.0)


def _uniform_field(vector: np.ndarray, /) -> FieldLaw:
    def field(time: Array, points: Array, args: Any) -> Array:
        del time, args
        return jnp.broadcast_to(jnp.asarray(vector, dtype=jnp.float64), points.shape)

    return field


def lattice_side(capacity: int, minimum: int, /) -> int:
    side = int(round(math.sqrt(capacity)))
    if side < minimum:
        raise ValueError(f"Capacity {capacity} is below the {minimum}^2 lattice minimum.")
    return side


def _periodic_lattice(
    side: int, jitter: float, seed: int, /, *, offset: float = 0.5
) -> np.ndarray:
    spacing = 1.0 / side
    axis = (np.arange(side, dtype=np.float64) + offset) * spacing
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    noise = np.random.default_rng(seed).uniform(-1.0, 1.0, points.shape)
    return np.mod(points + jitter * spacing * noise, 1.0)


def _adr_cloud(
    side: int, seed: int, recorder: PhaseRecorder, /
) -> PreparedPointCloudDiscretization:
    points = _periodic_lattice(side, _ADR_JITTER, seed)
    weights = np.full(points.shape[0], 1.0 / side**2, dtype=np.float64)
    recorder.unavailable(
        "search",
        "PointCloudPlan.prepare fuses the periodic Morton search with the local fit",
    )
    return recorder.run(
        "local-fit",
        lambda: PointCloudPlan(
            points,
            weights,
            stencil=LocalStencilPolicy(polynomial_degree=3),
            address=_PERIODIC,
        ).prepare(),
        scope="periodic-gmls-degree-3",
    )


def _adr_evolution(
    cloud: PreparedPointCloudDiscretization, plan_id: str, /
) -> PreparedMeshfreeEvolution:
    return MeshfreeEvolutionPlan(
        cloud,
        velocity=_uniform_field(_ADR_VELOCITY),
        diffusion=MeshfreeDiffusionLaw(_ADR_DIFFUSIVITY, law_id="isotropic-k"),
        reaction=MeshfreeReactionLaw(_relaxation, law_id="linear-relaxation"),
        plan_id=plan_id,
    ).prepare()


def _rollout(
    evolution: PreparedMeshfreeEvolution,
    method: ReferenceMethod,
    initial: Array,
    final: float,
    steps: int,
    /,
) -> tuple[Array, bool, int]:
    match method:
        case "ssprk33" | "ssprk54":
            solution = solve_fixed_step(
                FixedStepProblem(
                    evolution.ssprk_method(method),
                    initial,
                    t0=0.0,
                    t1=final,
                    step_size=final / steps,
                ),
                save_every=steps,
            )
            return solution.states[-1], bool(solution.successful), 0
        case "ars-222":
            solution = solve_fixed_step(
                FixedStepProblem(
                    evolution.imex_method("ars-222"),
                    initial,
                    t0=0.0,
                    t1=final,
                    step_size=final / steps,
                ),
                save_every=steps,
                evidence_retention="steps",
            )
            evidence = solution.evidence
            iterations = (
                0
                if evidence is None or evidence.steps is None
                else int(np.sum(np.asarray(evidence.steps["stage_iterations"])))
            )
            return solution.states[-1], bool(solution.successful), iterations


def _explicit_steps(
    evolution: PreparedMeshfreeEvolution,
    initial: Array,
    final: float,
    minimum: int,
    spacing: float,
    recorder: PhaseRecorder,
    /,
) -> tuple[int, float]:
    estimate = recorder.run(
        "solve",
        lambda: evolution.spectral_estimate(0.0, initial),
        scope="implicit-part-spectral-estimate",
    )
    limit = (
        _EXPLICIT_FRACTION
        * _SSPRK33_REAL_INTERVAL
        * spacing**2
        / (8.0 * _ADR_DIFFUSIVITY)
    )
    return max(minimum, math.ceil(final / limit)), float(estimate.radius)


def measure_bulk_adr_spatial(
    method: BulkTemporalMethod, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Collocation ADR error at one measured fill distance (temporal error negligible)."""
    _require_dimension(config, 2, "bulk ADR")
    side = lattice_side(capacity, 8)
    count = side * side
    reservation = config.check_capacity(count, degree=3)
    recorder = PhaseRecorder()
    cloud = _adr_cloud(side, seed, recorder)
    evolution = recorder.run(
        "assembly",
        lambda: _adr_evolution(cloud, f"q9-adr-spatial-{method}-{count}"),
        scope="semidiscrete-adr",
    )
    nodes = cloud.points
    initial = evolution.initial_state(adr_exact(0.0, nodes))
    _, compiled = recorder.compiled_action(
        lambda state: evolution.rate(0.0, state),
        initial,
        budget_bytes=config.resource_bytes,
        repeats=config.repeats,
        scope="semidiscrete-rate",
    )
    # Only the explicit route estimates the implicit-part radius.
    radius: float | None = None
    match method:
        case "ssprk33":
            steps, radius = _explicit_steps(
                evolution, initial, _ADR_FINAL_TIME, 40, 1.0 / side, recorder
            )
        case "ars-222":
            # Diffusion is implicit; 40 steps keep the O(dt^2) temporal error far
            # below the spatial error of every declared resolution.
            steps = 40
    final, successful, iterations = recorder.run(
        "solve",
        lambda: _rollout(evolution, method, initial, _ADR_FINAL_TIME, steps),
        scope=f"{method}-rollout",
    )
    exact = adr_exact(_ADR_FINAL_TIME, nodes)
    error = jnp.abs(final - exact)
    weights = cloud.quadrature_weights
    fill = _measured_fill(np.asarray(nodes), seed, _PERIODIC_FILL)
    metrics = {
        "successful": successful,
        "max_error": float(jnp.max(error)),
        "weighted_l2_error": float(jnp.sqrt(jnp.sum(weights * error**2))),
        "relative_content_drift": float(
            jnp.abs(jnp.sum(weights * final) - jnp.sum(weights * initial))
            / jnp.sum(weights * initial)
        ),
        "fill_distance": fill["fill_distance"],
        "separation_distance": fill["separation_distance"],
        "steps": steps,
        "step_size": _ADR_FINAL_TIME / steps,
        "implicit_spectral_radius_estimate": radius,
        "stage_iterations": iterations,
        "point_count": count,
    }
    return _result(
        f"bulk-adr-spatial-{method}",
        count,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        (cloud, evolution),
        oracle="exact periodic advection-diffusion-relaxation mode (closed form)",
        consumer="phydrax.discretization.meshfree.MeshfreeEvolutionPlan (collocation route)",
        fill=fill,
        reserved_working_set_bytes=reservation,
        **compiled,
    )


_TEMPORAL_REFINEMENTS = (1, 2, 4, 8)
_REFERENCE_FACTOR = 8


def measure_bulk_adr_temporal(
    method: BulkTemporalMethod, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Temporal error of one method over four step sizes on one fixed cloud."""
    _require_dimension(config, 2, "bulk ADR")
    side = lattice_side(capacity, 8)
    count = side * side
    reservation = config.check_capacity(count, degree=3)
    recorder = PhaseRecorder()
    cloud = _adr_cloud(side, seed, recorder)
    evolution = recorder.run(
        "assembly",
        lambda: _adr_evolution(cloud, f"q9-adr-temporal-{method}-{count}"),
        scope="semidiscrete-adr",
    )
    nodes = cloud.points
    initial = evolution.initial_state(adr_exact(0.0, nodes))
    # The coarsest step stays inside the declared explicit stability step.
    base, radius = _explicit_steps(
        evolution, initial, _ADR_FINAL_TIME, 10, 1.0 / side, recorder
    )
    reference_steps = base * _TEMPORAL_REFINEMENTS[-1] * _REFERENCE_FACTOR
    reference, reference_ok, _ = recorder.run(
        "solve",
        lambda: _rollout(evolution, "ssprk54", initial, _ADR_FINAL_TIME, reference_steps),
        scope="ssprk54-reference",
    )
    scales: list[float] = []
    errors: list[float] = []
    successes: list[bool] = []
    iterations = 0
    for factor in _TEMPORAL_REFINEMENTS:
        steps = base * factor
        state, successful, used = recorder.run(
            "solve",
            lambda steps=steps: _rollout(
                evolution, method, initial, _ADR_FINAL_TIME, steps
            ),
            scope=f"{method}-{steps}",
        )
        scales.append(_ADR_FINAL_TIME / steps)
        errors.append(float(jnp.max(jnp.abs(state - reference))))
        successes.append(successful)
        iterations += used
    exact = adr_exact(_ADR_FINAL_TIME, nodes)
    pairwise = [
        math.log(errors[index] / errors[index + 1]) / math.log(2.0)
        for index in range(len(errors) - 1)
        if errors[index] > 0 and errors[index + 1] > 0
    ]
    metrics = {
        "successful": all(successes) and reference_ok,
        "coarsest_error": errors[0],
        "finest_error": errors[-1],
        "minimum_pairwise_order": min(pairwise) if pairwise else None,
        "reference_steps": reference_steps,
        "reference_error_vs_exact": float(jnp.max(jnp.abs(reference - exact))),
        "implicit_spectral_radius_estimate": radius,
        "stage_iterations": iterations,
        "point_count": count,
    }
    return _result(
        f"bulk-adr-temporal-{method}",
        count,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        (cloud, evolution),
        oracle=f"SSPRK54 reference of the same semidiscrete system at {reference_steps} "
        "steps (temporal error only; spatial error cancels)",
        consumer="phydrax.discretization.meshfree.PreparedMeshfreeEvolution."
        + ("ssprk_method" if method == "ssprk33" else "imex_method"),
        temporal_refinement={
            "step_sizes": scales,
            "errors": errors,
            "pairwise_orders": pairwise,
            "metric": "max-abs-error-vs-ssprk54-reference",
        },
        reserved_working_set_bytes=reservation,
    )


# ---------------------------------------------------------------------------
# Q9 conservative graph transport on Cartesian lattices.
#
# The signed exact metric on jittered clouds is refused upstream (TransportP5
# blocker), so graph rows use lattices whose control volumes close the metric.

_PULSE_SPEED = np.asarray([1.0, 0.5], dtype=np.float64)
_PULSE_FINAL_TIME = 0.4
_CLOSED_FINAL_TIME = 0.25
_GRAPH_DIFFUSIVITY = 0.01
_LATTICE_FILL = "max nearest-node distance over scrambled Sobol probes of [0,1]^2"


def _square_lattice(side: int, /) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    spacing = 1.0 / (side - 1)
    axis = np.arange(side, dtype=np.float64) * spacing
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    edge = np.isclose(points, 0.0) | np.isclose(points, 1.0)
    volumes = spacing**2 * np.prod(np.where(edge, 0.5, 1.0), axis=1)
    return points, volumes, edge.any(axis=1), spacing


def _graph_owners(
    side: int, scheme: TransportScheme, recorder: PhaseRecorder, /
) -> tuple[PreparedMeshfreeExteriorCalculus, ConservativeTransport, np.ndarray]:
    points, volumes, boundary, spacing = _square_lattice(side)
    # Prescribed lattice faces carry no moment rows: declare their outward
    # control-volume face areas (corners split between both faces).
    lower = np.isclose(points, 0.0)
    upper = np.isclose(points, 1.0)
    share = np.where((lower | upper).sum(axis=1, keepdims=True) > 1, 0.5, 1.0) * spacing
    areas = np.where(lower, -share, np.where(upper, share, 0.0))
    exterior = recorder.run(
        "geometry",
        lambda: MeshfreeExteriorCalculusPlan(
            points,
            1.1 * spacing,
            4 * points.shape[0],
            node_volumes=volumes,
            dirichlet=boundary,
            boundary_area_vectors=areas,
        ).prepare(),
        scope="radius-graph-and-moment-metric",
    )
    reason = (
        "MeshfreeExteriorCalculusPlan.prepare fuses the radius relation search, "
        "rank certificate and moment-metric solve (recorded under geometry)"
    )
    for phase in ("search", "rank-certificate", "conic"):
        recorder.unavailable(phase, reason)
    reconstruction = (
        None
        if scheme == "upwind"
        else recorder.run(
            "local-fit",
            lambda: PointCloudPlan(
                points, volumes, stencil=LocalStencilPolicy(polynomial_degree=2)
            ).prepare(),
            scope="gmls-degree-2-reconstruction",
        )
    )
    transport = ConservativeTransport(
        exterior, scheme=scheme, reconstruction=reconstruction
    )
    return exterior, transport, points


def _pulse(points: Array, /) -> Array:
    radial = (points[:, 0] + 0.1) ** 2 + (points[:, 1] - 0.3) ** 2
    return jnp.exp(-30.0 * radial) * (1.0 + 0.5 * jnp.sin(3.0 * points[:, 1]))


def pulse_exact(time: Array | float, points: Array, /) -> Array:
    return _pulse(points - time * jnp.asarray(_PULSE_SPEED, dtype=jnp.float64))


def _pulse_inflow(time: Array, points: Array, args: Any) -> Array:
    del args
    return pulse_exact(time, points)


def _swirl(time: Array, points: Array, args: Any) -> Array:
    """Divergence-free flow tangent to every face of the unit square."""
    del time, args
    x, y = points[:, 0], points[:, 1]
    return jnp.stack(
        (
            jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y),
            -jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y),
        ),
        axis=1,
    )


def _no_inflow(time: Array, points: Array, args: Any) -> Array:
    del time, args
    return jnp.zeros(points.shape[0], dtype=jnp.float64)


def _block(points: Array, /) -> Array:
    inside = (jnp.abs(points[:, 0] - 0.35) < 0.15) & (jnp.abs(points[:, 1] - 0.5) < 0.15)
    return inside.astype(jnp.float64)


def measure_graph_inflow(
    scheme: TransportScheme, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Inflow pulse advected at the scheme's forward-Euler step bound (fixed CFL)."""
    _require_dimension(config, 2, "graph transport")
    side = lattice_side(capacity, 9)
    count = side * side
    reservation = config.check_capacity(count, degree=2)
    recorder = PhaseRecorder()
    exterior, transport, points = _graph_owners(side, scheme, recorder)
    velocity = _uniform_field(_PULSE_SPEED)
    evolution = recorder.run(
        "assembly",
        lambda: MeshfreeEvolutionPlan(
            transport,
            velocity=velocity,
            inflow=_pulse_inflow,
            # Genuine inflow nodes follow the declared inflow as prescribed rows:
            # boundary nodes have no tangential graph edges, so weak upwind
            # inflow would lag the inflow state by O(h).
            inflow_treatment="strong",
            plan_id=f"q9-graph-pulse-{scheme}-{count}",
        ).prepare(),
        scope="graph-evolution",
    )
    nodes = exterior.points
    initial = evolution.initial_state(pulse_exact(0.0, nodes))
    certificate = evolution.transport_cfl(0.0, initial, 1.0)
    bound = float(certificate.step_bound)
    # SSPRK33 has SSP coefficient one: the forward-Euler bound is its step bound.
    steps = math.ceil(_PULSE_FINAL_TIME / bound)
    step_size = _PULSE_FINAL_TIME / steps
    _, compiled = recorder.compiled_action(
        lambda state: evolution.rate(0.0, state),
        initial,
        budget_bytes=config.resource_bytes,
        repeats=config.repeats,
        scope="graph-rate",
    )
    solution = recorder.run(
        "solve",
        lambda: solve_fixed_step(
            FixedStepProblem(
                evolution.ssprk_method("ssprk33"),
                initial,
                t0=0.0,
                t1=_PULSE_FINAL_TIME,
                step_size=step_size,
            ),
            save_every=steps,
        ),
        scope="ssprk33-rollout",
    )
    final = evolution.fields(solution.states[-1]).concentration
    exact = pulse_exact(_PULSE_FINAL_TIME, nodes)
    volumes = exterior.node_volumes
    error = jnp.abs(final - exact)
    nodal_velocity = velocity(jnp.asarray(0.0), nodes, None)
    ledger_start = transport.rate(
        evolution.fields(initial).concentration,
        nodal_velocity,
        inflow=pulse_exact(0.0, nodes),
    )
    ledger_end = transport.rate(
        final, nodal_velocity, inflow=pulse_exact(_PULSE_FINAL_TIME, nodes)
    )
    fill = _measured_fill(np.asarray(points), seed, _LATTICE_FILL)
    metrics = {
        "successful": bool(solution.successful),
        "metric_accepted": bool(exterior.metric_result.accepted),
        "l1_error": float(jnp.sum(volumes * error)),
        "max_error": float(jnp.max(error)),
        "minimum_concentration": float(jnp.min(final)),
        "content_ratio": float(
            evolution.total_content(solution.states[-1])
            / evolution.total_content(initial)
        ),
        "maximum_ledger_residual": max(
            float(jnp.abs(ledger_start.conservation_residual)),
            float(jnp.abs(ledger_end.conservation_residual)),
        ),
        "limiter_switched": int(ledger_end.limiter_switched),
        "cfl_certified": bool(certificate.certified),
        "step_bound": bound,
        "steps": steps,
        "fill_distance": fill["fill_distance"],
        "separation_distance": fill["separation_distance"],
        "point_count": count,
        "edge_count": int(exterior.endpoint_displacements.shape[0]),
    }
    return _result(
        f"bulk-graph-inflow-{scheme}",
        count,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        (exterior, transport),
        oracle="exact translated nonpolynomial pulse with exact inflow state (closed form)",
        consumer="phydrax.discretization.meshfree.ConservativeTransport + MeshfreeEvolutionPlan",
        fill=fill,
        seed_role="seed selects only fill-distance probes; the lattice is deterministic",
        reserved_working_set_bytes=reservation,
        **compiled,
    )


def measure_graph_closed(
    method: BulkTemporalMethod, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Limited transport of a discontinuous block in a closed swirl: mass and bounds."""
    _require_dimension(config, 2, "graph transport")
    side = lattice_side(capacity, 9)
    count = side * side
    reservation = config.check_capacity(count, degree=2)
    recorder = PhaseRecorder()
    exterior, transport, _ = _graph_owners(side, "limited", recorder)
    positivity = method == "ssprk33"
    diffusion = (
        None
        if method == "ssprk33"
        else MeshfreeDiffusionLaw(_GRAPH_DIFFUSIVITY, law_id="graph-isotropic-k")
    )
    evolution = recorder.run(
        "assembly",
        lambda: MeshfreeEvolutionPlan(
            transport,
            velocity=_swirl,
            inflow=_no_inflow,
            diffusion=diffusion,
            positivity=positivity,
            plan_id=f"q9-graph-closed-{method}-{count}",
        ).prepare(),
        scope="graph-evolution",
    )
    initial = evolution.initial_state(_block(exterior.points))
    bound = float(evolution.transport_cfl(0.0, initial, 1.0).step_bound)
    steps = math.ceil(_CLOSED_FINAL_TIME / bound)
    step_size = _CLOSED_FINAL_TIME / steps
    method_object = (
        evolution.ssprk_method("ssprk33")
        if method == "ssprk33"
        else evolution.imex_method("ars-222")
    )
    solution = recorder.run(
        "solve",
        lambda: solve_fixed_step(
            FixedStepProblem(
                method_object, initial, t0=0.0, t1=_CLOSED_FINAL_TIME, step_size=step_size
            ),
            save_every=1,
        ),
        scope=f"{method}-rollout",
    )
    states = solution.states
    mass = jnp.asarray([evolution.total_content(state) for state in states])
    concentration = jnp.stack([evolution.fields(state).concentration for state in states])
    # A velocity-derived volume flux is O(h^2)-accurate but not discretely
    # solenoidal: each SSP forward-Euler stage bounds c_i by its previous
    # maximum times (1 + dt max(-div_h u, 0)) (TransportP5), so after n steps the
    # declared upper bound is max(c_0) (1 + dt kappa)^n, not max(c_0).
    divergence = (
        transport.volume_rate(_swirl(jnp.asarray(0.0), exterior.points, None))
        / exterior.node_volumes
    )
    kappa = float(jnp.max(jnp.maximum(-divergence, 0.0)))
    growth = (1.0 + step_size * kappa) ** jnp.arange(concentration.shape[0])
    upper = jnp.max(concentration[0]) * growth
    metrics = {
        "successful": bool(solution.successful),
        "metric_accepted": bool(exterior.metric_result.accepted),
        "relative_mass_drift": float(jnp.max(jnp.abs(mass - mass[0])) / mass[0]),
        "minimum_concentration": float(jnp.min(concentration)),
        "maximum_overshoot": float(jnp.max(jnp.max(concentration, axis=1) - upper)),
        "maximum_compression_rate": kappa,
        "compression_bound": float(growth[-1]),
        "uncompensated_overshoot": float(jnp.max(concentration) - 1.0),
        "steps": steps,
        "step_bound": bound,
        "point_count": count,
    }
    return _result(
        f"bulk-graph-closed-limited-{method}",
        count,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        (exterior, transport),
        oracle="exact invariants: closed tangential flow conserves content; the "
        "limited forward-Euler certificate bounds values to the initial range [0, 1]",
        consumer="phydrax.discretization.meshfree.ConservativeTransport(scheme='limited')",
        seed_role="seed is unused; the lattice and block are deterministic",
        reserved_working_set_bytes=reservation,
    )


def _shell_lattice(side: int, seed: int, /) -> PreparedPointCloudDiscretization:
    """Periodic lattice whose 21 neighbors close a distance shell (positive trust)."""
    points = _periodic_lattice(side, 0.005, seed, offset=0.98)
    return PointCloudPlan(
        points,
        np.full(points.shape[0], 1.0 / side**2, dtype=np.float64),
        stencil=LocalStencilPolicy(polynomial_degree=3),
        neighbors=21,
        address=_PERIODIC,
    ).prepare()


def measure_bulk_ale(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """ALE seam motion: GCL free stream, material-minus-mesh rate, accuracy, rebase."""
    _require_dimension(config, 2, "bulk ALE")
    side = lattice_side(capacity, 8)
    count = side * side
    reservation = config.check_capacity(count, degree=3, neighbors=21)
    recorder = PhaseRecorder()
    recorder.unavailable(
        "search",
        "PointCloudPlan.prepare fuses the periodic Morton search with the local fit",
    )
    recorder.unavailable(
        "numeric-refresh",
        "in-stage fixed-support refresh is fused into every ALE stage of the solve",
    )
    cloud = recorder.run(
        "local-fit", lambda: _shell_lattice(side, seed), scope="shell-lattice"
    )
    points = cloud.points
    weights = cloud.quadrature_weights
    trust = float(jnp.min(cloud.trust_radius))
    shift = np.asarray([0.4 * trust, -0.2 * trust], dtype=np.float64)
    free_stream = recorder.run(
        "assembly",
        lambda: MeshfreeEvolutionPlan(
            cloud,
            motion=MeshfreeMotion(
                "ale", mesh_velocity=_uniform_field(shift), law_id="translate"
            ),
            plan_id=f"q9-ale-gcl-{count}",
        ).prepare(),
        scope="gcl-translation",
    )
    uniform = free_stream.initial_state(jnp.ones(count, dtype=jnp.float64))
    gcl_state, gcl_ok, _ = recorder.run(
        "solve",
        lambda: _rollout(free_stream, "ssprk33", uniform, 1.0, 4),
        scope="gcl-translation",
    )
    gcl = free_stream.fields(gcl_state)
    material = _uniform_field(_ADR_VELOCITY)
    # Total mesh displacement over the run is half the fixed-support trust.
    mesh_velocity = (
        0.5 * trust / _ADR_FINAL_TIME * np.asarray([1.0, 0.0], dtype=np.float64)
    )
    ale = recorder.run(
        "assembly",
        lambda: MeshfreeEvolutionPlan(
            cloud,
            velocity=material,
            diffusion=MeshfreeDiffusionLaw(_ADR_DIFFUSIVITY, law_id="isotropic-k"),
            reaction=MeshfreeReactionLaw(_relaxation, law_id="linear-relaxation"),
            motion=MeshfreeMotion(
                "ale",
                mesh_velocity=_uniform_field(mesh_velocity),
                law_id="seam-translation",
            ),
            plan_id=f"q9-ale-adr-{count}",
        ).prepare(),
        scope="ale-adr",
    )
    relative = recorder.run(
        "assembly",
        lambda: MeshfreeEvolutionPlan(
            cloud,
            velocity=_uniform_field(_ADR_VELOCITY - mesh_velocity),
            diffusion=MeshfreeDiffusionLaw(_ADR_DIFFUSIVITY, law_id="isotropic-k"),
            reaction=MeshfreeReactionLaw(_relaxation, law_id="linear-relaxation"),
            plan_id=f"q9-relative-adr-{count}",
        ).prepare(),
        scope="eulerian-relative-velocity",
    )
    start = adr_exact(0.0, points)
    initial = ale.initial_state(start)
    ale_rate = ale.rate(0.0, initial)[:count] / weights
    identity = float(jnp.max(jnp.abs(ale_rate - relative.rate(0.0, start))))
    final_time = _ADR_FINAL_TIME
    steps, _ = _explicit_steps(ale, initial, final_time, 20, 1.0 / side, recorder)
    state, motion_ok, _ = recorder.run(
        "solve",
        lambda: _rollout(ale, "ssprk33", initial, final_time, steps),
        scope="ale-adr-rollout",
    )
    moved = ale.fields(state)
    error = float(
        jnp.max(jnp.abs(moved.concentration - adr_exact(final_time, moved.points)))
    )
    successor = recorder.run(
        "epoch-commit", lambda: ale.rebase(state), scope="support-rebase"
    )
    repacked = ale.repacked(state)
    step = 0.5 * float(successor.capacity.support_trust) / float(mesh_velocity[0])
    continued = recorder.run(
        "solve",
        lambda: successor.ssprk_method("ssprk33").step(
            jnp.asarray(0), jnp.asarray(final_time), repacked, jnp.asarray(step), None
        ),
        scope="rebased-continuation",
    )
    metrics = {
        "gcl_successful": gcl_ok,
        "gcl_free_stream_error": float(jnp.max(jnp.abs(gcl.concentration - 1.0))),
        "gcl_volume_relative_error": float(
            jnp.max(jnp.abs(gcl.volumes - weights) / weights)
        ),
        "gcl_position_error": float(
            jnp.max(jnp.abs(gcl.points - (points + jnp.asarray(shift))))
        ),
        "material_minus_mesh_rate_error": identity,
        "motion_successful": motion_ok,
        "ale_max_error": error,
        "relative_content_drift": float(
            jnp.abs(ale.total_content(state) - ale.total_content(initial))
            / ale.total_content(initial)
        ),
        "points_crossing_seam": int(jnp.sum(moved.points[:, 0] >= 1.0)),
        "rebased_step_accepted": bool(continued.successful),
        "support_trust": trust,
        "point_count": count,
    }
    return _result(
        "bulk-ale-motion",
        count,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        (cloud, ale),
        oracle="exact GCL invariants (uniform state, translated volumes/points), the "
        "material-minus-mesh identity, and the exact ADR mode at moved nodes",
        consumer="phydrax.discretization.meshfree.MeshfreeMotion('ale') + "
        "PreparedMeshfreeEvolution.rebase",
        reserved_working_set_bytes=reservation,
    )


# ---------------------------------------------------------------------------
# Q9 declared refusals: CFL admission, positivity, support trust, topology trust.


def _closed_attempt(
    scheme: TransportScheme,
    multiple: float,
    capacity: int,
    config: MeshfreeConfig,
    recorder: PhaseRecorder,
    /,
) -> tuple[dict[str, Any], int, Any]:
    _require_dimension(config, 2, "graph transport refusal")
    side = lattice_side(capacity, 9)
    count = side * side
    config.check_capacity(count, degree=2)
    exterior, transport, _ = _graph_owners(side, scheme, recorder)
    evolution = recorder.run(
        "assembly",
        lambda: MeshfreeEvolutionPlan(
            transport,
            velocity=_swirl,
            inflow=_no_inflow,
            positivity=True,
            plan_id=f"q9-refusal-{scheme}-{count}",
        ).prepare(),
        scope="graph-evolution",
    )
    initial = evolution.initial_state(_block(exterior.points))
    bound = float(evolution.transport_cfl(0.0, initial, 1.0).step_bound)
    step = multiple * bound
    certificate = transport.cfl(_swirl(jnp.asarray(0.0), exterior.points, None), step)
    method = evolution.ssprk_method("ssprk33")
    attempt = recorder.run(
        "solve",
        lambda: method.step(
            jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(step), None
        ),
        scope="refused-step",
    )
    admission = evolution.admission(attempt.candidate_state)
    rollout = recorder.run(
        "solve",
        lambda: solve_fixed_step(
            FixedStepProblem(method, initial, t0=0.0, t1=2.0 * step, step_size=step)
        ),
        scope="refused-rollout",
    )
    status = int(admission.status)
    metrics = {
        "cfl": float(certificate.cfl),
        "cfl_certified": bool(certificate.certified),
        "cfl_admitted": bool(certificate.admitted),
        "step_refused": not bool(attempt.successful),
        "negative_state_status": status == int(MeshfreeEvolutionStatus.NEGATIVE_STATE),
        "admission_status": status,
        "candidate_minimum": float(jnp.min(attempt.candidate_state)),
        "candidate_kept_raw": bool(float(jnp.min(attempt.candidate_state)) < 0.0),
        "accepted_state_held": bool(jnp.all(attempt.accepted_state == initial)),
        "rollout_refused": not bool(rollout.successful),
        "rollout_state_held": bool(jnp.all(rollout.states[-1] == initial)),
        "rolled_back": not bool(rollout.successful)
        and bool(jnp.all(rollout.states[-1] == initial)),
        "step_multiple_of_bound": multiple,
        "point_count": count,
    }
    statuses = {"admission": MeshfreeEvolutionStatus(status).name}
    return metrics, count, (statuses, (exterior, transport))


def measure_cfl_refusal(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Upwind step four times its forward-Euler bound: CFL refused, step rolled back."""
    recorder = PhaseRecorder()
    metrics, count, (statuses, retained) = _closed_attempt(
        "upwind", 4.0, capacity, config, recorder
    )
    return _result(
        "bulk-cfl-refusal",
        count,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        retained,
        oracle="declared TransportCFL.admitted=False and MeshfreeEvolutionStatus.NEGATIVE_STATE",
        consumer="phydrax.discretization.meshfree.ConservativeTransport.cfl + admission",
        statuses=statuses,
    )


def measure_positivity_refusal(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Unlimited reconstruction at the upwind bound: uncertified, negative state refused."""
    recorder = PhaseRecorder()
    metrics, count, (statuses, retained) = _closed_attempt(
        "reconstructed", 1.0, capacity, config, recorder
    )
    return _result(
        "bulk-positivity-refusal",
        count,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        retained,
        oracle="declared TransportCFL.certified=False for 'reconstructed' and "
        "MeshfreeEvolutionStatus.NEGATIVE_STATE without clipping",
        consumer="phydrax.discretization.meshfree.ConservativeTransport(scheme='reconstructed')",
        statuses=statuses,
    )


def measure_support_refusal(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """ALE motion four times the fixed-support trust: SUPPORT_EXCEEDED and rollback."""
    _require_dimension(config, 2, "bulk ALE refusal")
    side = lattice_side(capacity, 8)
    count = side * side
    config.check_capacity(count, degree=3, neighbors=21)
    recorder = PhaseRecorder()
    cloud = recorder.run(
        "local-fit", lambda: _shell_lattice(side, seed), scope="shell-lattice"
    )
    trust = float(jnp.max(cloud.trust_radius))
    evolution = recorder.run(
        "assembly",
        lambda: MeshfreeEvolutionPlan(
            cloud,
            diffusion=MeshfreeDiffusionLaw(_ADR_DIFFUSIVITY, law_id="isotropic-k"),
            motion=MeshfreeMotion(
                "ale",
                mesh_velocity=_uniform_field(np.asarray([4.0 * trust, 0.0])),
                law_id="beyond-trust",
            ),
            plan_id=f"q9-support-refusal-{count}",
        ).prepare(),
        scope="ale-beyond-trust",
    )
    initial = evolution.initial_state(adr_exact(0.0, cloud.points))
    method = evolution.ssprk_method("ssprk33")
    attempt = recorder.run(
        "solve",
        lambda: method.step(
            jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(1.0), None
        ),
        scope="refused-step",
    )
    admission = evolution.admission(attempt.candidate_state)
    rollout = recorder.run(
        "solve",
        lambda: solve_fixed_step(
            FixedStepProblem(method, initial, t0=0.0, t1=2.0, step_size=1.0)
        ),
        scope="refused-rollout",
    )
    status = int(admission.status)
    metrics = {
        "step_refused": not bool(attempt.successful),
        "support_exceeded_status": status
        == int(MeshfreeEvolutionStatus.SUPPORT_EXCEEDED),
        "admission_status": status,
        "support_accepted": bool(admission.support_accepted),
        "accepted_state_held": bool(jnp.all(attempt.accepted_state == initial)),
        "rollout_refused": not bool(rollout.successful),
        "rollout_state_held": bool(jnp.all(rollout.states[-1] == initial)),
        "rolled_back": not bool(rollout.successful)
        and bool(jnp.all(rollout.states[-1] == initial)),
        "support_trust": trust,
        "point_count": count,
    }
    return _result(
        "bulk-support-refusal",
        count,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        (cloud, evolution),
        oracle="declared MeshfreeEvolutionStatus.SUPPORT_EXCEEDED with the state held",
        consumer="phydrax.discretization.meshfree.PreparedMeshfreeEvolution.admission",
        statuses={"admission": MeshfreeEvolutionStatus(status).name},
    )


def measure_topology_refusal(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Graph refresh beyond the topology trust margin: refused, owner unchanged."""
    _require_dimension(config, 2, "graph transport refusal")
    side = lattice_side(capacity, 9)
    count = side * side
    config.check_capacity(count, degree=2)
    recorder = PhaseRecorder()
    exterior, transport, _ = _graph_owners(side, "upwind", recorder)
    margin = float(exterior.topology_trust_margin)
    moved = exterior.points + 2.0 * margin
    refresh = recorder.run(
        "numeric-refresh", lambda: transport.refresh(moved), scope="beyond-topology-trust"
    )
    status = int(refresh.status)
    metrics = {
        "refresh_refused": not bool(refresh.accepted),
        "topology_trust_status": status
        == int(TransportRefreshStatus.TOPOLOGY_TRUST_EXCEEDED),
        "refresh_status": status,
        "owner_unchanged": bool(
            jnp.all(refresh.transport.exterior.points == exterior.points)
        ),
        "topology_trust_margin": margin,
        "point_count": count,
    }
    return _result(
        "bulk-topology-refusal",
        count,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        (exterior, transport),
        oracle="declared TransportRefreshStatus.TOPOLOGY_TRUST_EXCEEDED; refusal "
        "returns the unchanged owner",
        consumer="phydrax.discretization.meshfree.ConservativeTransport.refresh",
        statuses={"refresh": TransportRefreshStatus(status).name},
    )


# ---------------------------------------------------------------------------
# Q13 point transfer on tensor quadratures of the unit cube.

_TRANSFER_TOLERANCE = 1e-10
# Declared max-norm amplification max_r sum_e |t_e|: above it a transfer's
# accuracy order is unbounded and the plan refuses (AMPLIFICATION_EXCEEDED).
TRANSFER_LEBESGUE_BOUND = 10.0


def _tensor_grid(
    nodes: np.ndarray, weights: np.ndarray, dimension: int, /
) -> tuple[np.ndarray, np.ndarray]:
    axes = np.meshgrid(*([nodes] * dimension), indexing="ij")
    points = np.stack([axis.reshape(-1) for axis in axes], axis=1)
    factors = np.meshgrid(*([weights] * dimension), indexing="ij")
    measure = np.prod(
        np.stack([factor.reshape(-1) for factor in factors], axis=1), axis=1
    )
    return points, measure


def simpson_grid(intervals: int, dimension: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Tensor composite Simpson nodes/weights of [0,1]^d (exact for degree 3 per axis)."""
    weights = np.where(np.arange(intervals + 1) % 2 == 1, 4.0, 2.0)
    weights[[0, -1]] = 1.0
    return _tensor_grid(
        np.linspace(0.0, 1.0, intervals + 1), weights / (3.0 * intervals), dimension
    )


def midpoint_grid(cells: int, dimension: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Tensor midpoint nodes/weights of [0,1]^d (exact for degree 1 per axis)."""
    return _tensor_grid(
        (np.arange(cells, dtype=np.float64) + 0.5) / cells,
        np.full(cells, 1.0 / cells),
        dimension,
    )


def target_intervals(capacity: int, dimension: int, /) -> int:
    """Even Simpson interval count whose (m+1)^d targets are nearest ``capacity``."""
    return 2 * max(1, int(round((capacity ** (1.0 / dimension) - 1.0) / 2.0)))


def source_intervals(target: int, /) -> int:
    return target + 2 * max(1, int(round(target / 4)))


def transfer_routes(capacity: int, dimension: int, neighbors: int, /) -> int:
    """Controlling route capacity of one size-sweep transfer row."""
    return (target_intervals(capacity, dimension) + 1) ** dimension * neighbors


def _smooth(points: np.ndarray, /) -> np.ndarray:
    value = np.sin(2.0 * np.pi * points[:, 0])
    for axis in range(1, points.shape[1]):
        value = value * np.cos(np.pi * points[:, axis])
    return value


def _request(case: TransferCase, /) -> PointTransferRequest:
    match case:
        case "conservative-signed":
            return PointTransferRequest("conservative-signed")
        case "conservative-positive":
            return PointTransferRequest("conservative-positive")
        case "joint-nonnegative-constant":
            return PointTransferRequest("joint", nonnegative=True)
        case "joint-signed-degree2":
            return PointTransferRequest("joint", moment_degree=2)


def _value_functional(dimension: int, /) -> MeshfreeFunctional:
    return MeshfreeFunctional(((0,) * dimension,), (1.0,), name="cross-target-value")


def _stencil_transfer(
    sources: np.ndarray,
    targets: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    request: PointTransferRequest,
    neighbors: int,
    recorder: PhaseRecorder,
    /,
) -> PreparedPointTransfer:
    dimension = sources.shape[1]
    neighborhood = recorder.run(
        "search",
        lambda: MeshfreeNeighborhoodPlan(sources, neighbors, targets=targets).prepare(),
        scope="cross-target-knn",
    )
    stencils = recorder.run(
        "local-fit",
        lambda: prepare_local_stencils(
            neighborhood,
            sources,
            targets,
            (_value_functional(dimension),),
            # Linear base: a quadratic local fit is rank deficient at face
            # targets whose nearest lattice sources span only two planes; the
            # declared request, not the base, carries the moment equations.
            LocalStencilPolicy(polynomial_degree=1),
        ),
        scope="gmls-degree-1-cross-target-value",
    )
    return recorder.run(
        "transfer",
        lambda: PointTransferPlan.from_stencils(
            stencils,
            old,
            new,
            request=request,
            tolerance=_TRANSFER_TOLERANCE,
            lebesgue_bound=TRANSFER_LEBESGUE_BOUND,
        ).prepare(),
        scope="declared-transfer-solve-and-audit",
    )


def _transfer_evidence_metrics(prepared: PreparedPointTransfer, /) -> dict[str, Any]:
    evidence = prepared.evidence
    moments = np.asarray(evidence.moment_residual)
    return {
        "admitted": bool(prepared.admitted),
        "status": int(evidence.status),
        "provider_status": int(evidence.provider_status),
        "audited_conservation_residual": float(
            np.max(np.abs(np.asarray(evidence.conservation_residual)), initial=0.0)
        ),
        "audited_constant_residual": float(
            np.max(np.abs(np.asarray(evidence.constant_residual)), initial=0.0)
        ),
        "audited_moment_residual": float(np.max(np.abs(moments), initial=0.0)),
        "minimum_coefficient": float(evidence.minimum_coefficient),
        "correction_norm": float(evidence.correction_norm),
        "objective_value": float(evidence.objective_value),
        "obstruction_defect": float(evidence.obstruction_defect),
        "witness_residual": float(evidence.witness_residual),
        "witness_margin": float(evidence.witness_margin),
        "uncovered_source_count": len(evidence.uncovered_sources),
        "uncovered_target_count": len(evidence.uncovered_targets),
        "route_count": int(np.asarray(evidence.coefficients).shape[0]),
        "lebesgue_constant": float(evidence.lebesgue_constant),
    }


def _transfer_labels(prepared: PreparedPointTransfer, /) -> dict[str, str]:
    evidence = prepared.evidence
    return {
        "status": evidence.status.name,
        "provider": evidence.provider,
        "witness_kind": evidence.witness_kind,
    }


def measure_point_transfer(
    case: TransferCase, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Audited stencil-based transfer between non-nested tensor Simpson quadratures."""
    _require_float64(config, "point transfer")
    dimension = config.dimension
    neighbors = config.neighbors
    target_m = target_intervals(capacity, dimension)
    sources, old = simpson_grid(source_intervals(target_m), dimension)
    targets, new = simpson_grid(target_m, dimension)
    routes = targets.shape[0] * neighbors
    reservation = config.check_capacity(sources.shape[0], degree=2, neighbors=neighbors)
    # Routes carry coefficient, offset, equation and conic slack storage.
    declare_reservation(
        8 * routes * (2 * dimension + 24), config, scope="point-transfer route system"
    )
    recorder = PhaseRecorder()
    prepared = _stencil_transfer(
        sources, targets, old, new, _request(case), neighbors, recorder
    )
    metrics = _transfer_evidence_metrics(prepared)
    smooth = _smooth(sources)
    extra: dict[str, Any] = {}
    if prepared.transfer is not None:
        transfer = prepared.transfer
        mapped, compiled = recorder.compiled_action(
            lambda values: prepared.apply(values),
            jnp.asarray(smooth),
            budget_bytes=config.resource_bytes,
            repeats=config.repeats,
            scope="frozen-transfer-apply",
        )
        extra.update(compiled)
        mapped = np.asarray(mapped)
        constant = np.asarray(prepared.apply(np.ones(sources.shape[0])))
        linear = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(prepared.apply(sources[:, axis])) - targets[:, axis]
                    )
                )
            )
            for axis in range(dimension)
        )
        quadratic = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(prepared.apply(sources[:, axis] ** 2))
                        - targets[:, axis] ** 2
                    )
                )
            )
            for axis in range(dimension)
        )
        positive = np.asarray(prepared.apply(1.0 + smooth))
        cotangent = np.cos(3.0 * targets[:, 0])
        dual = transfer.dual_pullback_operator
        if dual is None:
            raise ValueError(
                "An admitted point transfer must publish its coordinate dual."
            )
        pairing = float(np.vdot(mapped, cotangent))
        pulled = np.asarray(dual.mv(jnp.asarray(cotangent)))
        pairing_scale = np.linalg.norm(mapped) * np.linalg.norm(
            cotangent
        ) + np.linalg.norm(smooth) * np.linalg.norm(pulled)
        metrics.update(
            {
                "independent_content_defect": float(
                    abs(np.vdot(new, mapped) - np.vdot(old, smooth))
                    / np.vdot(old, np.abs(smooth))
                ),
                "constant_reproduction_error": float(np.max(np.abs(constant - 1.0))),
                "linear_reproduction_error": linear,
                "quadratic_reproduction_error": quadratic,
                "remap_max_error": float(np.max(np.abs(mapped - _smooth(targets)))),
                "positive_field_minimum": float(np.min(positive)),
                "dual_identity_residual": abs(pairing - float(np.vdot(smooth, pulled)))
                / max(float(pairing_scale), 1e-300),
                "dual_pairing": pairing,
                "dual_pairing_scale": float(pairing_scale),
            }
        )
    else:
        reason = "transfer refused; no frozen operator exists to execute"
        for phase in ("lowering", "compilation", "first", "warm", "jvp", "vjp"):
            recorder.unavailable(phase, reason)
    fill = _measured_fill(
        sources,
        seed,
        "source fill: max nearest-source distance over scrambled Sobol probes of [0,1]^d",
    )
    metrics.update(
        {
            "fill_distance": fill["fill_distance"],
            "separation_distance": fill["separation_distance"],
            "source_count": sources.shape[0],
            "target_count": targets.shape[0],
            "route_neighbors": neighbors,
        }
    )
    return _result(
        f"point-transfer-{case}",
        routes,
        capacity,
        seed,
        dimension,
        recorder,
        metrics,
        prepared,
        oracle="independent host NumPy: content of mapped field vs source quadrature, "
        "T1=1, exact polynomial images, smooth-field point error at targets, dual pairing",
        consumer="phydrax.discretization.meshfree.PointTransferPlan.from_stencils",
        labels=_transfer_labels(prepared),
        fill=fill,
        seed_role="seed selects only fill-distance probes; quadratures are deterministic",
        reserved_working_set_bytes=reservation,
        **extra,
    )


def _uncovered_target_transfer(
    sources: np.ndarray,
    targets: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    neighbors: int,
    recorder: PhaseRecorder,
    /,
) -> PreparedPointTransfer:
    """Radius-limited routes: the exterior target has no source within 3 h_s."""
    dimension = sources.shape[1]
    exterior = np.full((1, dimension), 1.5, dtype=np.float64)
    requested = np.concatenate((targets, exterior))
    # The exterior target takes half the last interior target's measure, so
    # the total measures still agree and only coverage can refuse.
    measures = np.concatenate((new, np.zeros(1, dtype=np.float64)))
    measures[-2:] = 0.5 * new[-1]
    neighborhood = recorder.run(
        "search",
        lambda: MeshfreeNeighborhoodPlan(sources, neighbors, targets=requested).prepare(),
        scope="cross-target-knn",
    )
    relation = neighborhood.relation
    indices = np.asarray(relation.source_indices)
    valid = np.asarray(relation.valid)
    distance = np.linalg.norm(sources[indices] - requested[:, None, :], axis=2)
    spacing = float(np.min(np.diff(np.unique(sources[:, 0]))))
    keep = valid & (distance <= 3.0 * spacing * math.sqrt(dimension))
    rows, slots = np.nonzero(keep)
    columns = indices[rows, slots]
    inverse = 1.0 / np.maximum(distance[rows, slots], 1e-12)
    totals = np.bincount(rows, weights=inverse, minlength=requested.shape[0])
    coefficients = inverse / totals[rows]
    return recorder.run(
        "transfer",
        lambda: PointTransferPlan(
            EdgeRelation(
                columns.astype(np.int32),
                rows.astype(np.int32),
                source_size=sources.shape[0],
                target_size=requested.shape[0],
            ),
            coefficients,
            old,
            measures,
            source_id="simpson-sources",
            target_id="simpson-targets-with-exterior",
            request=PointTransferRequest("joint"),
            tolerance=_TRANSFER_TOLERANCE,
        ).prepare(),
        scope="declared-transfer-solve-and-audit",
    )


def measure_transfer_refusal(
    case: TransferRefusal, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Declared transfer refusal with its audited witness; nothing is executed."""
    _require_float64(config, "point transfer refusal")
    dimension = config.dimension
    neighbors = config.neighbors
    target_m = target_intervals(capacity, dimension)
    recorder = PhaseRecorder()
    expected: PointTransferStatus
    match case:
        case "uncovered-source":
            # Sources three times finer per axis and the minimal unisolvent k:
            # most sources lie in no target's support.
            k = math.comb(dimension + 2, 2)
            sources, old = simpson_grid(3 * target_m, dimension)
            targets, new = simpson_grid(target_m, dimension)
            prepared = _stencil_transfer(
                sources,
                targets,
                old,
                new,
                PointTransferRequest("conservative-signed"),
                k,
                recorder,
            )
            expected = PointTransferStatus.UNCOVERED_SOURCE
        case "uncovered-target":
            sources, old = simpson_grid(target_m + 2, dimension)
            targets, new = simpson_grid(target_m, dimension)
            prepared = _uncovered_target_transfer(
                sources, targets, old, new, neighbors, recorder
            )
            expected = PointTransferStatus.UNCOVERED_TARGET
        case "measure-obstruction":
            # Material dilation disguised as resampling: target total measure 1.1.
            sources, old = simpson_grid(source_intervals(target_m), dimension)
            targets, new = simpson_grid(target_m, dimension)
            prepared = _stencil_transfer(
                sources,
                targets,
                old,
                1.1 * new,
                PointTransferRequest("joint", nonnegative=True),
                neighbors,
                recorder,
            )
            expected = PointTransferStatus.MEASURE_OBSTRUCTION
        case "infeasible-left-null":
            # Midpoint rules of different widths integrate x^2 differently.
            sources, old = midpoint_grid(source_intervals(target_m), dimension)
            targets, new = midpoint_grid(target_m, dimension)
            prepared = _stencil_transfer(
                sources,
                targets,
                old,
                new,
                PointTransferRequest("joint", moment_degree=2),
                neighbors,
                recorder,
            )
            expected = PointTransferStatus.INFEASIBLE
        case "infeasible-farkas":
            # Positive linear-exact transfer from a 1.5x finer midpoint grid on
            # nearest-source routes has no nonnegative solution (independently
            # confirmed infeasible by a HiGHS LP in 1-D and 2-D); a certified
            # Farkas ray must be reported, not an unresolved provider failure.
            sources, old = midpoint_grid(source_intervals(target_m), dimension)
            targets, new = midpoint_grid(target_m, dimension)
            prepared = _stencil_transfer(
                sources,
                targets,
                old,
                new,
                PointTransferRequest("joint", moment_degree=1, nonnegative=True),
                neighbors,
                recorder,
            )
            expected = PointTransferStatus.INFEASIBLE
    metrics = _transfer_evidence_metrics(prepared)
    metrics.update(
        {
            "expected_status": prepared.evidence.status is expected,
            "transfer_withheld": prepared.transfer is None,
            "witness_margin_magnitude": abs(float(prepared.evidence.witness_margin)),
            "source_count": sources.shape[0],
            "target_count": targets.shape[0],
        }
    )
    return _result(
        f"transfer-refusal-{case}",
        metrics["route_count"],
        capacity,
        seed,
        dimension,
        recorder,
        metrics,
        prepared,
        oracle=f"declared PointTransferStatus.{expected.name} with its independently "
        "audited witness (coverage, measure summation, left-null or Farkas ray)",
        consumer="phydrax.discretization.meshfree.PointTransferPlan",
        labels={**_transfer_labels(prepared), "expected_status": expected.name},
    )


# ---------------------------------------------------------------------------
# Q13 physical multiregion events consumed as meshfree epochs.

_OWNER = "q13-meshfree-surfactant"
_EPOCH, _SUPPORT = "foam/epoch", "foam/support"
_CURRENT, _CLOCK = "foam/surfactant", "foam/clock"
_HISTORIES = ("foam/surfactant-history/0", "foam/surfactant-history/1")


def merge_seed(subdivisions: int, /) -> MultiRegionSurfaceSeed:
    """Two unit icosphere bubbles separated by a thin ambient gap."""
    first = seed_sphere(1.0, subdivisions=subdivisions)
    second = seed_sphere(1.0, center=(2.04, 0.0, 0.0), subdivisions=subdivisions)
    count = first.positions.shape[0]
    return MultiRegionSurfaceSeed(
        np.concatenate((first.positions, second.positions)),
        np.concatenate((first.faces, second.faces + count)),
        np.concatenate(
            (
                np.tile((0, 2), (first.faces.shape[0], 1)),
                np.tile((1, 2), (second.faces.shape[0], 1)),
            )
        ),
        ("left", "right", "ambient"),
        ("finite", "finite", "boundary"),
        source=f"two-bubbles-s{subdivisions}",
    )


def merge_capacity_plan(
    seed: MultiRegionSurfaceSeed, /
) -> MultiRegionSurfaceCapacityPlan:
    counts = seed.counts()
    return MultiRegionSurfaceCapacityPlan(
        vertex_capacity=2 * counts.vertex,
        edge_capacity=2 * counts.edge,
        face_capacity=2 * counts.face,
        region_capacity=counts.region + 2,
        region_pair_capacity=2 * counts.region_pair + 2,
        maximum_edge_valence=3,
        maximum_vertex_region_pairs=9,
        resource_id="q13-merge",
        event_capacity=16,
    )


def pinch_seed(ring_points: int, rows: int, /) -> MultiRegionSurfaceSeed:
    return seed_catenoid(1.0, 0.7, ring_points=ring_points, rows=rows, neck_radius=0.04)


def pinch_capacity_plan(
    seed: MultiRegionSurfaceSeed, /
) -> MultiRegionSurfaceCapacityPlan:
    return seed.capacity_plan(resource_id="q13-pinch", headroom=2.0, event_capacity=32)


def _surfactant_state(
    topology: MultiRegionSurfaceTopology, base: MultiRegionSurfaceState, /
) -> MultiRegionSurfaceState:
    areas = np.asarray(
        PreparedMultiRegionSurface(topology, base).slot_areas(base.positions)
    )
    x = np.asarray(base.positions)[:, 0]
    concentration = 3.0e-7 * (1.0 + 0.1 * x)
    return MultiRegionSurfaceState(
        topology,
        base.positions,
        sheet_fields=(concentration[:, None] * areas)[:, :, None],
        sheet_field_names=("surfactant",),
    )


def _composition(
    topology: MultiRegionSurfaceTopology,
    state: MultiRegionSurfaceState,
    /,
    *,
    poisoned: bool = False,
) -> Composition:
    """Epoch, support, current field, two live histories and a controller."""
    support = MeshfreeSheetSupport(topology, state)
    epoch = CompositionEntry(
        support.epoch,
        entry_id=_EPOCH,
        role="topology",
        owner_id=_OWNER,
        structure_id=support.epoch.epoch_id,
        revision_id=support.epoch.epoch_id,
        semantics_id="foam-sheet-epoch",
    )
    bound = (epoch.binding("structure"),)
    current = support.concentration(state.sheet_fields[..., 0])
    entries = [
        epoch,
        CompositionEntry(
            support,
            entry_id=_SUPPORT,
            role="discretization",
            owner_id=_OWNER,
            structure_id=support.support_id,
            revision_id=support.support_id,
            semantics_id="sheet-slot-support",
            dependencies=bound,
        ),
    ]
    for entry_id, scale in ((_CURRENT, 1.0), (_HISTORIES[0], 0.95), (_HISTORIES[1], 0.9)):
        value = scale * current
        if poisoned and entry_id == _HISTORIES[1]:
            value = value.at[0].set(jnp.nan)
        entries.append(
            CompositionEntry(
                value,
                entry_id=entry_id,
                role="physical-state" if entry_id == _CURRENT else "history",
                owner_id=_OWNER,
                structure_id=support.epoch.epoch_id,
                revision_id=f"{entry_id}:{support.epoch.epoch_id}",
                semantics_id="surfactant-concentration",
                dependencies=bound,
            )
        )
    entries.append(
        CompositionEntry(
            jnp.asarray(1.0e-3, dtype=jnp.float64),
            entry_id=_CLOCK,
            role="optimizer-state",
            owner_id=_OWNER,
            structure_id="scalar-step",
            revision_id="step-0",
            semantics_id="step-size-controller",
        )
    )
    return Composition(entries, boundary_id="accepted-step")


def _event_epoch_entries(
    event: MeshfreeSurfaceEventEpoch, /
) -> tuple[CompositionEntry, CompositionEntry]:
    target_epoch = CompositionEntry(
        event.change.target,
        entry_id=_EPOCH,
        role="topology",
        owner_id=_OWNER,
        structure_id=event.change.target.epoch_id,
        revision_id=event.change.change_id,
        semantics_id="foam-sheet-epoch",
    )
    support = CompositionEntry(
        event.target,
        entry_id=_SUPPORT,
        role="discretization",
        owner_id=_OWNER,
        structure_id=event.target.support_id,
        revision_id=event.target.support_id,
        semantics_id="sheet-slot-support",
        dependencies=(target_epoch.binding("structure"),),
    )
    return target_epoch, support


def _advance(
    composition: Composition,
    topology: MultiRegionSurfaceTopology,
    state: MultiRegionSurfaceState,
    result: SurfaceEventPassResult,
    recorder: PhaseRecorder,
    scope: str,
    /,
) -> tuple[MeshfreeSurfaceEventEpoch, MeshfreeEpochReceipt]:
    event = recorder.run(
        "transfer", lambda: surface_event_epoch(topology, state, result), scope=scope
    )
    transition = event.transition()
    _, support = _event_epoch_entries(event)

    def transaction() -> MeshfreeEpochReceipt:
        candidate = stage_meshfree_epoch(
            composition,
            event.change,
            epoch_entry=_EPOCH,
            remap={name: transition for name in (_CURRENT, *_HISTORIES)},
            reprepare=(support,),
        )
        return commit_meshfree_epoch(candidate, accepted_boundary=True)

    receipt = recorder.run("epoch-commit", transaction, scope=scope)
    return event, receipt


def _content(composition: Composition, entry_id: str, /) -> float:
    support = composition.value(_SUPPORT)
    return float(jnp.vdot(support.measures, composition.value(entry_id)))


def _drift(composition: Composition, entry_id: str, expected: float, /) -> float:
    return abs(_content(composition, entry_id) - expected) / expected


def _residuals_within(receipt: MeshfreeEpochReceipt, /) -> bool:
    residuals = np.abs(np.asarray(receipt.conservation_residuals))
    return bool(np.all(residuals <= np.asarray(receipt.content_tolerances)))


def merge_face_capacity(level: int, /) -> int:
    """Declared face capacity of the two-bubble topology at icosphere ``level``."""
    return merge_capacity_plan(merge_seed(level)).face_capacity


def merge_subdivisions(capacity: int, /) -> int:
    """Largest icosphere level whose two-bubble face capacity fits ``capacity``."""
    level = 1
    while merge_face_capacity(level + 1) <= capacity:
        level += 1
    return level


def measure_surface_merge(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Physical film merge, then a refinement continuation, as two meshfree epochs."""
    _require_dimension(config, 3, "multiregion surface events")
    level = merge_subdivisions(capacity)
    recorder = PhaseRecorder()

    def build() -> tuple[
        MultiRegionSurfaceTopology, MultiRegionSurfaceState, Composition
    ]:
        bubbles = merge_seed(level)
        topology = bubbles.topology(merge_capacity_plan(bubbles))
        state = _surfactant_state(topology, bubbles.state(topology))
        return topology, state, _composition(topology, state)

    topology, state, composition = recorder.run(
        "geometry", build, scope="two-bubble-seed"
    )
    plan = merge_capacity_plan(merge_seed(level))
    initial = _content(composition, _CURRENT)
    histories = [_content(composition, name) for name in _HISTORIES]
    search = recorder.run(
        "search",
        lambda: propose_merges(
            PreparedMultiRegionSurface(topology, state),
            state,
            SurfaceMergePolicy(("ambient",), merge_distance=0.4),
        ),
        scope="merge-proposals",
    )
    merges = [item for item in search.proposals if isinstance(item, MergeProposal)]
    # Every proposal goes to one pass: the authority applies its own local
    # guards (a proposal may be refused, e.g. LABEL_ORIENTATION_INCONSISTENT)
    # and commits the first admissible merge; zipped films invalidate the rest.
    merged = recorder.run(
        "geometry",
        lambda: apply_surface_events(topology, state, merges),
        scope="physical-merge-commit",
    )
    if not merged.committed:
        raise ValueError(
            f"The multiregion authority committed no merge at icosphere level {level} "
            f"({len(merges)} merge proposals from {len(search.proposals)} candidates); "
            "no physical event exists to consume."
        )
    event, receipt = _advance(composition, topology, state, merged, recorder, "merge")
    published = receipt.composition
    authority = event.target.concentration(merged.state.sheet_fields[..., 0])
    mismatch = float(
        jnp.max(jnp.abs(published.value(_CURRENT) - authority))
        / jnp.max(jnp.abs(authority))
    )
    refine = MultiRegionRemeshPlan(
        minimum_edge_length=0.05 / 2 ** (level - 1),
        maximum_edge_length=0.55 / 2 ** (level - 1),
        operations=("split",),
    )
    refined = recorder.run(
        "geometry",
        lambda: apply_surface_events(
            merged.topology,
            merged.state,
            propose_remesh(
                PreparedMultiRegionSurface(merged.topology, merged.state),
                merged.state,
                refine,
            ),
        ),
        scope="continuation-refinement-commit",
    )
    second, continued = _advance(
        published, merged.topology, merged.state, refined, recorder, "continuation"
    )
    transition = second.transition()
    values = jnp.asarray(published.value(_CURRENT))
    _, tangent = recorder.run(
        "jvp",
        lambda: jax.jvp(
            lambda v: transition.apply(v).values, (values,), (jnp.ones_like(values),)
        ),
        scope="frozen-transfer-value",
    )
    _, pullback = jax.vjp(lambda v: transition.apply(v).values, values)
    cotangent = jnp.linspace(-1.0, 1.0, second.target.measures.shape[0])
    reverse = recorder.run(
        "vjp", lambda: pullback(cotangent)[0], scope="frozen-transfer-value"
    )
    final = continued.composition
    lineage = event.change.lineage
    second_lineage = second.change.lineage
    if lineage is None or second_lineage is None:
        raise ValueError("A surface-event epoch must carry its physical lineage.")
    metrics = {
        "published": bool(receipt.published),
        "continuation_published": bool(continued.published),
        "merge_in_lineage": "MERGE" in lineage.event_kinds,
        "topology_changed": bool(lineage.topology_changed),
        "ccd_certified": bool(jnp.all(lineage.ccd_certified)),
        "transfer_conservative": bool(event.transfer.evidence.conservative),
        "transfer_nonnegative": bool(event.transfer.evidence.nonnegative),
        "constant_preserving": bool(event.transfer.evidence.constant_preserving),
        "area_obstruction_defect": float(event.transfer.evidence.obstruction_defect),
        "authority_mismatch": mismatch,
        "wall_slots": sum(label == ("left", "right") for label in event.target.labels),
        "epoch_index": int(final.value(_EPOCH).index),
        "content_drift": _drift(final, _CURRENT, initial),
        "history_drift": max(
            _drift(final, name, value)
            for name, value in zip(_HISTORIES, histories, strict=True)
        ),
        "every_history_remapped": tuple(sorted(receipt.remapped))
        == tuple(sorted((_CURRENT, *_HISTORIES))),
        "receipt_residuals_within_tolerance": _residuals_within(receipt)
        and _residuals_within(continued),
        "controller_carried": final.value(_CLOCK) is composition.value(_CLOCK),
        "value_jvp_finite": bool(jnp.all(jnp.isfinite(tangent))),
        "value_vjp_matches_pullback": float(
            jnp.max(jnp.abs(reverse - transition.pullback(cotangent)))
        ),
        "subdivisions": level,
        "face_capacity": plan.face_capacity,
        "edge_capacity": plan.edge_capacity,
        "support_points": int(event.target.measures.shape[0]),
    }
    return _result(
        "surface-merge",
        plan.face_capacity,
        capacity,
        seed,
        3,
        recorder,
        metrics,
        (event, second),
        oracle="the multiregion transaction's committed sheet content (physical "
        "authority) and exact content invariants of every live history",
        consumer="phydrax.discretization.meshfree.surface_event_epoch + "
        "stage_meshfree_epoch/commit_meshfree_epoch",
        labels={
            "event_kinds": list(lineage.event_kinds),
            "continuation_kinds": sorted(set(second_lineage.event_kinds)),
        },
        seed_role="seed is unused; icosphere bubbles are deterministic",
    )


def pinch_face_capacity(rows: int, /) -> int:
    """Declared face capacity of the catenoid with ``rows`` rows (3 rows/2 ring points)."""
    return pinch_capacity_plan(pinch_seed(_ring(rows), rows)).face_capacity


def pinch_shape(capacity: int, /) -> tuple[int, int]:
    """Catenoid ring points and rows whose declared face capacity fits ``capacity``."""
    rows = 8
    while pinch_face_capacity(rows + 2) <= capacity:
        rows += 2
    return _ring(rows), rows


def _ring(rows: int, /) -> int:
    return 3 * rows // 2


def measure_surface_pinch(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Catenoid neck pinch-off with a core region split, chained as meshfree epochs."""
    _require_dimension(config, 3, "multiregion surface events")
    ring, rows = pinch_shape(capacity)
    recorder = PhaseRecorder()

    def build() -> tuple[
        MultiRegionSurfaceSeed,
        MultiRegionSurfaceTopology,
        MultiRegionSurfaceState,
        Composition,
    ]:
        catenoid = pinch_seed(ring, rows)
        topology = catenoid.topology(pinch_capacity_plan(catenoid))
        state = _surfactant_state(topology, catenoid.state(topology))
        return catenoid, topology, state, _composition(topology, state)

    catenoid, topology, state, composition = recorder.run(
        "geometry", build, scope="catenoid-seed"
    )
    plan = pinch_capacity_plan(catenoid)
    initial = _content(composition, _CURRENT)
    histories = [_content(composition, name) for name in _HISTORIES]
    policy = SurfaceEventPolicy(
        fixed_vertex_ids=catenoid.vertex_set("ring-lower")
        + catenoid.vertex_set("ring-upper")
    )
    collapse = MultiRegionRemeshPlan(
        minimum_edge_length=0.06, maximum_edge_length=1.0, operations=("collapse",)
    )
    epochs = 0
    published = True
    residuals = True
    pinch: MeshfreeSurfaceEventEpoch | None = None
    # Collapses of one pass mostly conflict, so the pass count scales with the
    # neck resolution. Each committed pass removes at least one collapse
    # candidate, so the initial candidate count (+1 for the pinch pass) bounds
    # the passes; a pass committing nothing is a deterministic stall.
    budget = 1 + len(
        propose_remesh(PreparedMultiRegionSurface(topology, state), state, collapse)
    )
    for index in range(budget):
        prepared = PreparedMultiRegionSurface(topology, state)
        proposals = recorder.run(
            "search",
            lambda prepared=prepared: [
                *propose_pinches(prepared, state, maximum_neck_perimeter=0.3),
                RegionSplitProposal("core"),
                *propose_remesh(prepared, state, collapse),
            ],
            scope=f"pass-{index}",
        )
        result = recorder.run(
            "geometry",
            lambda proposals=proposals: apply_surface_events(
                topology, state, proposals, policy=policy
            ),
            scope=f"physical-pass-{index}",
        )
        if not result.committed:
            break
        event, receipt = _advance(
            composition, topology, state, result, recorder, f"pass-{index}"
        )
        published = published and bool(receipt.published)
        residuals = residuals and _residuals_within(receipt)
        if not receipt.published:
            break
        composition, topology, state = receipt.composition, result.topology, result.state
        epochs += 1
        lineage = event.change.lineage
        if lineage is not None and "PINCH" in lineage.event_kinds:
            pinch = event
            break
    lineage = None if pinch is None else pinch.change.lineage
    parents = () if lineage is None else lineage.region_parents
    metrics = {
        "pinched": lineage is not None,
        "every_epoch_published": published,
        "receipt_residuals_within_tolerance": residuals,
        "epochs": epochs,
        "core_split_into_children": sum(
            1 for _, sources in parents if sources == ("core",)
        )
        >= 2,
        "core_removed": lineage is not None and "core" in lineage.removed_region_ids,
        "content_drift": _drift(composition, _CURRENT, initial),
        "history_drift": max(
            _drift(composition, name, value)
            for name, value in zip(_HISTORIES, histories, strict=True)
        ),
        "ring_points": ring,
        "rows": rows,
        "face_capacity": plan.face_capacity,
        "edge_capacity": plan.edge_capacity,
    }
    return _result(
        "surface-pinch",
        plan.face_capacity,
        capacity,
        seed,
        3,
        recorder,
        metrics,
        composition,
        oracle="the multiregion transaction's committed pinch/region-split lineage "
        "(physical authority) and exact content invariants of every live history",
        consumer="phydrax.discretization.meshfree.surface_event_epoch + "
        "stage_meshfree_epoch/commit_meshfree_epoch",
        labels={
            "region_ids": list(topology.region_ids),
            "region_parents": [
                f"{child}<-{'+'.join(sources)}" for child, sources in parents
            ],
        },
        seed_role="seed is unused; the catenoid is deterministic",
    )


def measure_merge_rollback(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """A nonfinite live history fails the merge epoch; the source is returned unchanged."""
    _require_dimension(config, 3, "multiregion surface events")
    level = merge_subdivisions(capacity)
    recorder = PhaseRecorder()
    bubbles = merge_seed(level)
    plan = merge_capacity_plan(bubbles)
    topology = bubbles.topology(plan)
    state = _surfactant_state(topology, bubbles.state(topology))
    composition = _composition(topology, state, poisoned=True)
    current = np.asarray(composition.value(_CURRENT))
    search = recorder.run(
        "search",
        lambda: propose_merges(
            PreparedMultiRegionSurface(topology, state),
            state,
            SurfaceMergePolicy(("ambient",), merge_distance=0.4),
        ),
        scope="merge-proposals",
    )
    merges = [item for item in search.proposals if isinstance(item, MergeProposal)]
    merged = recorder.run(
        "geometry",
        lambda: apply_surface_events(topology, state, merges[:1]),
        scope="physical-merge-commit",
    )
    _, receipt = _advance(
        composition, topology, state, merged, recorder, "poisoned-merge"
    )
    metrics = {
        "physical_event_committed": bool(merged.committed),
        "epoch_refused": not bool(receipt.published),
        "failed_only_poisoned_history": tuple(receipt.failed) == (_HISTORIES[1],),
        "source_composition_returned": receipt.composition is composition,
        "current_field_unchanged": bool(
            np.array_equal(np.asarray(receipt.composition.value(_CURRENT)), current)
        ),
        "value_derivative_withheld": not bool(receipt.value_derivative_available),
        "subdivisions": level,
        "face_capacity": plan.face_capacity,
    }
    return _result(
        "surface-merge-rollback",
        plan.face_capacity,
        capacity,
        seed,
        3,
        recorder,
        metrics,
        composition,
        oracle="transaction invariant: any failed history returns the source composition object",
        consumer="phydrax.discretization.meshfree.commit_meshfree_epoch",
        labels={"failed": list(receipt.failed)},
        seed_role="seed is unused; icosphere bubbles are deterministic",
    )


def measure_history_route_refusal(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Staging an epoch without a route for one live history is refused before work."""
    _require_dimension(config, 3, "multiregion surface events")
    level = merge_subdivisions(capacity)
    recorder = PhaseRecorder()
    bubbles = merge_seed(level)
    plan = merge_capacity_plan(bubbles)
    topology = bubbles.topology(plan)
    state = _surfactant_state(topology, bubbles.state(topology))
    composition = _composition(topology, state)
    search = propose_merges(
        PreparedMultiRegionSurface(topology, state),
        state,
        SurfaceMergePolicy(("ambient",), merge_distance=0.4),
    )
    merges = [item for item in search.proposals if isinstance(item, MergeProposal)]
    merged = apply_surface_events(topology, state, merges[:1])
    event = recorder.run(
        "transfer", lambda: surface_event_epoch(topology, state, merged), scope="merge"
    )
    transition = event.transition()
    _, support = _event_epoch_entries(event)
    message = ""
    # The ONE documented typed refusal: stage_meshfree_epoch raises ValueError
    # naming every live history without its own route; anything else re-raises.
    try:
        stage_meshfree_epoch(
            composition,
            event.change,
            epoch_entry=_EPOCH,
            remap={name: transition for name in (_CURRENT, _HISTORIES[0])},
            reprepare=(support,),
        )
    except ValueError as error:
        if "own route" not in str(error):
            raise
        message = str(error)
    metrics = {
        "refused_with_documented_error": bool(message),
        "names_missing_history": _HISTORIES[1] in message,
        "refused_naming_missing_history": bool(message) and _HISTORIES[1] in message,
        "face_capacity": plan.face_capacity,
    }
    return _result(
        "epoch-refusal-history-route",
        plan.face_capacity,
        capacity,
        seed,
        3,
        recorder,
        metrics,
        composition,
        oracle="declared ValueError: every epoch-bound live history needs its own route",
        consumer="phydrax.discretization.meshfree.stage_meshfree_epoch",
        labels={"refusal": message},
        seed_role="seed is unused; icosphere bubbles are deterministic",
    )


TRANSPORT_TRANSFER_WORKLOADS: dict[
    str, Callable[[int, int, MeshfreeConfig], dict[str, Any]]
] = {
    "bulk-adr-spatial-ssprk33": partial(measure_bulk_adr_spatial, "ssprk33"),
    "bulk-adr-spatial-ars222": partial(measure_bulk_adr_spatial, "ars-222"),
    "bulk-adr-temporal-ssprk33": partial(measure_bulk_adr_temporal, "ssprk33"),
    "bulk-adr-temporal-ars222": partial(measure_bulk_adr_temporal, "ars-222"),
    "bulk-graph-inflow-upwind": partial(measure_graph_inflow, "upwind"),
    "bulk-graph-inflow-reconstructed": partial(measure_graph_inflow, "reconstructed"),
    "bulk-graph-inflow-limited": partial(measure_graph_inflow, "limited"),
    "bulk-graph-closed-limited-ssprk33": partial(measure_graph_closed, "ssprk33"),
    "bulk-graph-closed-limited-ars222": partial(measure_graph_closed, "ars-222"),
    "bulk-ale-motion": measure_bulk_ale,
    "bulk-cfl-refusal": measure_cfl_refusal,
    "bulk-positivity-refusal": measure_positivity_refusal,
    "bulk-support-refusal": measure_support_refusal,
    "bulk-topology-refusal": measure_topology_refusal,
    "point-transfer-conservative-signed": partial(
        measure_point_transfer, "conservative-signed"
    ),
    "point-transfer-conservative-positive": partial(
        measure_point_transfer, "conservative-positive"
    ),
    "point-transfer-joint-nonnegative-constant": partial(
        measure_point_transfer, "joint-nonnegative-constant"
    ),
    "point-transfer-joint-signed-degree2": partial(
        measure_point_transfer, "joint-signed-degree2"
    ),
    "transfer-refusal-uncovered-source": partial(
        measure_transfer_refusal, "uncovered-source"
    ),
    "transfer-refusal-uncovered-target": partial(
        measure_transfer_refusal, "uncovered-target"
    ),
    "transfer-refusal-measure-obstruction": partial(
        measure_transfer_refusal, "measure-obstruction"
    ),
    "transfer-refusal-infeasible-left-null": partial(
        measure_transfer_refusal, "infeasible-left-null"
    ),
    "transfer-refusal-infeasible-farkas": partial(
        measure_transfer_refusal, "infeasible-farkas"
    ),
    "surface-merge": measure_surface_merge,
    "surface-pinch": measure_surface_pinch,
    "surface-merge-rollback": measure_merge_rollback,
    "epoch-refusal-history-route": measure_history_route_refusal,
}
