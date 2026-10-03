# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Q10/Q11 phase-separated flow and mechanics workloads (P16).

Q10 covers the periodic Taylor–Green flow on the exterior one-complex, the
bounded generalized-Stokes manufactured problem, and Lagrangian GMLS particle
flow with its SPH and measure-transfer interoperability. Q11 covers small- and
finite-strain elasticity, the Herrmann mixed form, closed-surface Stokes, and
the monolithic fluid–structure step. Every workload calls the public package
API, separates its preparation and solve phases with ``PhaseRecorder``, and
reports its scientific evidence against an independent analytic oracle.
Expected-refusal workloads report the documented typed status or exception.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from benchmarks._runtime import logical_array_bytes
from benchmarks.meshfree_scaling import (
    declare_reservation,
    DeclaredCapacityRefusal,
    fill_distance,
    MeshfreeConfig,
    PhaseRecorder,
    unit_cube_probes,
)
from phydrax import discretization as dsc, linalg as la
from phydrax.discretization import (
    MortonAddressPlan,
    PointBoundaryCondition,
    PointBoundaryPlan,
    PointCloudPlan,
    PreparedPointCloudDiscretization,
)
from phydrax.discretization.meshfree import (
    ImplicitSurfaceGeometry,
    LocalStencilPolicy,
    MaterialParticleMeasure,
    MechanicsStatus,
    MeshfreeCoercivityPolicy,
    MeshfreeEdgeRelationPlan,
    MeshfreeElasticityPlan,
    MeshfreeExteriorCalculusPlan,
    MeshfreeGeneralizedStokesPlan,
    MeshfreeHyperelasticPlan,
    MeshfreeMetricPolicy,
    minimum_image_edge_charts,
    PointTransferStatus,
    PreparedMeshfreeExteriorCalculus,
    SurfacePointCloudPlan,
    SurfaceQuadraturePolicy,
    SurfaceTangentCalculus,
)
from phydrax.metrix import EuclideanStateGeometry, RegularLevelSetManifold
from phydrax.nonlinear import NonlinearTermination
from phydrax.operators.mechanics import (
    LinearElasticityTensor,
    NeoHookeanLaw,
    NeoHookeanParameters,
)
from phydrax.solver import (
    CompatibleIncompressibleProjection,
    CompatibleProjectionStatus,
    CompatibleVariableDensityProjection,
    FixedStepProblem,
    IncompressibleProjectionResult,
    MeshfreeFlowEvidence,
    MeshfreeFlowStatus,
    MeshfreeIncompressibleFlowPlan,
    MeshfreeLagrangianFlowPlan,
    MeshfreeLagrangianStatus,
    MeshfreeMeasureTransferPlan,
    MeshfreeSPHReconstruction,
    MeshfreeSurfaceStokesPlan,
    PreparedMeshfreeIncompressibleFlow,
    PreparedMeshfreeLagrangianFlow,
    solve_fixed_step,
)


# ---------------------------------------------------------------------------
# Shared declarations


def _require_float64(config: MeshfreeConfig, /) -> None:
    if config.precision != "float64":
        raise ValueError("Flow and mechanics workloads declare float64 support only.")


def _side(capacity: int, minimum: int, /) -> int:
    """Lattice side whose square is the largest controlling count within ``capacity``."""
    side = math.isqrt(capacity)
    if side < minimum:
        raise ValueError(f"capacity {capacity} is below the {minimum}^2 lattice minimum.")
    return side


def _history_reservation(
    config: MeshfreeConfig, values_per_step: int, steps: int, /, *, scope: str
) -> int:
    """Declared float64 bytes of a retained step history, admitted before execution."""
    return declare_reservation(8 * values_per_step * (steps + 1), config, scope=scope)


def _unit_square_fill(points: np.ndarray, seed: int, /) -> dict[str, Any]:
    return fill_distance(
        points,
        unit_cube_probes(2, 4 * points.shape[0], seed),
        provenance="max nearest-sample distance over scrambled Sobol probes of [0,1]^2 "
        "(a lower estimate of the supremum)",
    )


def _weighted_rms(weights: Array, values: Array, /) -> float:
    squared = values**2 if values.ndim == 1 else jnp.sum(values**2, axis=1)
    return float(jnp.sqrt(jnp.sum(weights * squared) / jnp.sum(weights)))


def _relative(actual: Array, expected: Array, weights: Array, /) -> float:
    return _weighted_rms(weights, actual - expected) / _weighted_rms(weights, expected)


def _centered(weights: Array, values: Array, /) -> Array:
    return values - jnp.sum(weights * values) / jnp.sum(weights)


def _trapezoid_plate(side: int, seed: int, /) -> PreparedPointCloudDiscretization:
    """Jittered unit plate (interior jitter 0.15 h) with trapezoid volumes and
    boundary measures (corners: unit bisector normal with measure ``h/sqrt 2``)."""
    axis = np.linspace(0.0, 1.0, side, dtype=np.float64)
    grid = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack([value.reshape(-1) for value in grid], axis=1)
    faces = np.count_nonzero(np.isclose(points, 0.0) | np.isclose(points, 1.0), axis=1)
    spacing = 1.0 / (side - 1)
    interior = faces == 0
    points[interior] += (
        np.random.default_rng(seed).uniform(-0.15, 0.15, (np.count_nonzero(interior), 2))
        * spacing
    )
    boundary = faces > 0
    normals = np.where(np.isclose(points, 1.0), 1.0, 0.0) - np.where(
        np.isclose(points, 0.0), 1.0, 0.0
    )
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0)
    measure = np.where(faces == 2, spacing / np.sqrt(2.0), spacing)
    return PointCloudPlan(
        points,
        spacing**2 * 0.5**faces,
        boundary_mask=boundary,
        boundary_normals=normals,
        boundary_quadrature_weights=np.where(boundary, measure, 0.0),
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
    ).prepare()


def _face_rows(points: np.ndarray, axis: int, value: float, /) -> np.ndarray:
    """Rows on one face excluding the two corners (their normal is the bisector)."""
    other = points[:, 1 - axis]
    return np.flatnonzero(
        np.isclose(points[:, axis], value)
        & ~np.isclose(other, 0.0)
        & ~np.isclose(other, 1.0)
    )


# ---------------------------------------------------------------------------
# Q10 periodic Taylor–Green flow on the exterior one-complex

_TG_LENGTH = 2.0 * math.pi
_TG_VISCOSITY = 0.05
_TG_FINAL_TIME = 1.0
_TG_CYCLE_TOLERANCE = 0.15


def _tg_velocity(points: Array, time: float, /) -> Array:
    x, y = points[:, 0], points[:, 1]
    return math.exp(-2.0 * _TG_VISCOSITY * time) * jnp.stack(
        (jnp.sin(x) * jnp.cos(y), -jnp.cos(x) * jnp.sin(y)), axis=1
    )


def _tg_pressure(points: Array, time: float, /) -> Array:
    x, y = points[:, 0], points[:, 1]
    # (u . grad) u = -grad p: u u_x + v u_y = sin(2x)/2 = -p_x.
    return (
        0.25
        * math.exp(-4.0 * _TG_VISCOSITY * time)
        * (jnp.cos(2.0 * x) + jnp.cos(2.0 * y))
    )


def _tg_exterior(
    count: int,
    recorder: PhaseRecorder,
    /,
    *,
    coercivity: MeshfreeCoercivityPolicy | None = None,
) -> tuple[PreparedMeshfreeExteriorCalculus, MortonAddressPlan]:
    """Periodic lattice exterior complex with the signed exact metric (phases recorded)."""
    spacing = _TG_LENGTH / count
    lattice = np.stack(
        np.meshgrid(np.arange(count), np.arange(count), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    points = (lattice.astype(np.float64) + 0.5) * spacing
    address = MortonAddressPlan(
        (0.0, 0.0), (_TG_LENGTH, _TG_LENGTH), 12, periodic_axes=(True, True)
    )
    capacity = 4 * count * count
    relation = recorder.run(
        "search",
        lambda: MeshfreeEdgeRelationPlan(
            points, 1.1 * spacing, capacity, address=address
        ).prepare(),
        scope="periodic-radius-edge-relation",
    )
    charts = recorder.run(
        "geometry",
        lambda: minimum_image_edge_charts(relation, points),
        scope="minimum-image-edge-charts",
    )
    exterior = recorder.run(
        "assembly",
        lambda: MeshfreeExteriorCalculusPlan(
            points,
            1.1 * spacing,
            capacity,
            node_volumes=np.full(count * count, spacing**2, dtype=np.float64),
            intrinsic_displacements=charts,
            metric_policy=MeshfreeMetricPolicy(),
            coercivity_policy=MeshfreeCoercivityPolicy()
            if coercivity is None
            else coercivity,
        ).prepare(edge_relation=relation),
        scope="exterior-signed-exact-metric-hodge-coercivity",
    )
    recorder.unavailable(
        "rank-certificate",
        "Metric rank/row-equilibration certificate is fused into "
        "MeshfreeExteriorCalculusPlan.prepare (recorded as assembly)",
    )
    recorder.unavailable(
        "ordering-fill",
        "Coercivity AMD ordering and no-shift Cholesky are fused into "
        "MeshfreeExteriorCalculusPlan.prepare (recorded as assembly)",
    )
    recorder.unavailable(
        "conic",
        "The signed exact metric is a native minimum-norm solve; no conic program",
    )
    return exterior, address


def _tg_flow(
    exterior: PreparedMeshfreeExteriorCalculus,
    address: MortonAddressPlan,
    recorder: PhaseRecorder,
    /,
) -> PreparedMeshfreeIncompressibleFlow:
    cloud = recorder.run(
        "local-fit",
        lambda: PointCloudPlan(
            np.asarray(exterior.points),
            np.asarray(exterior.node_volumes),
            address=address,
            stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
        ).prepare(),
        scope="phs3-reconstruction-cloud",
    )
    return recorder.run(
        "assembly",
        lambda: MeshfreeIncompressibleFlowPlan(
            exterior,
            cloud,
            viscosity=_TG_VISCOSITY,
            domain_betti_number=2,
            method="ars-222",
            transport_scheme="reconstructed",
            cycle_tolerance=_TG_CYCLE_TOLERANCE,
        ).prepare(),
        scope="incompressible-flow-plan",
    )


def measure_incompressible_taylor_green(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Periodic Taylor–Green rollout to ``t = 1`` with ``dt = 1/n`` (ARS-222)."""
    _require_float64(config)
    count = _side(capacity, 8)
    nodes = count * count
    steps = count
    step_size = _TG_FINAL_TIME / steps
    reservation = config.check_capacity(nodes, degree=3)
    # Retained state history: velocity (2), density, pressure, 2 flux slots per node.
    reservation += _history_reservation(
        config, 6 * nodes, steps, scope="Taylor-Green rollout history"
    )
    recorder = PhaseRecorder()
    exterior, address = _tg_exterior(count, recorder)
    flow = _tg_flow(exterior, address, recorder)
    points = flow.points
    initial = flow.initialize(_tg_velocity(points, 0.0))
    problem = FixedStepProblem(
        flow,
        initial.state,
        t0=0.0,
        t1=_TG_FINAL_TIME,
        step_size=step_size,
        state_geometry=EuclideanStateGeometry(),
    )
    solution = recorder.run_repeated(
        "solve",
        lambda: solve_fixed_step(problem, evidence_retention="steps"),
        repeats=config.repeats,
        scope="solve_fixed_step rollout (first occurrence includes tracing/compilation)",
    )
    retained = solution.evidence
    if retained is None:
        raise ValueError("Taylor-Green rollout requires retained step evidence.")
    evidence = retained.steps
    if not isinstance(evidence, MeshfreeFlowEvidence):
        raise TypeError(
            "Taylor-Green rollout requires retained MeshfreeFlowEvidence steps."
        )
    final = jax.tree.map(lambda leaf: leaf[-1], solution.states)
    volumes = exterior.node_volumes

    projection = flow.plan.projection

    def project_result(edges: Array) -> IncompressibleProjectionResult:
        if isinstance(projection, CompatibleIncompressibleProjection):
            return projection.project(edges)
        if isinstance(projection, CompatibleVariableDensityProjection):
            return projection.project(edges, initial.state.density)
        raise TypeError("Taylor-Green flow requires a compatible pressure projection.")

    generator = np.random.default_rng(seed)
    noise = jnp.asarray(
        generator.standard_normal(exterior.lengths.shape), dtype=jnp.float64
    )

    def project(edges: Array) -> Array:
        return project_result(edges).velocity

    projected, execution = recorder.compiled_action(
        project,
        noise,
        budget_bytes=config.resource_bytes,
        repeats=config.repeats,
        scope="compatible-pressure-projection",
    )
    reprojected = project_result(projected)
    noise_projection = project_result(noise)
    noise_cycles = flow.classify_edge_velocity(noise_projection.candidate_velocity)
    energy_after = np.asarray(evidence.kinetic_energy_after, dtype=np.float64)
    energy_before = float(evidence.kinetic_energy_before[0])
    exact_ratio = math.exp(-4.0 * _TG_VISCOSITY * _TG_FINAL_TIME)
    velocity_error = final.velocity - _tg_velocity(points, _TG_FINAL_TIME)
    # The step pressure balances the momentum update of the last step: a
    # midpoint quantity compared with the exact pressure at t1 - dt/2.
    pressure_error = _centered(volumes, final.pressure) - _centered(
        volumes, _tg_pressure(points, _TG_FINAL_TIME - 0.5 * step_size)
    )
    mass_before = np.asarray(evidence.mass_before, dtype=np.float64)
    mass_after = np.asarray(evidence.mass_after, dtype=np.float64)
    fill = fill_distance(
        np.asarray(points) / _TG_LENGTH,
        unit_cube_probes(2, 4 * nodes, seed),
        provenance="lattice scaled to [0,1)^2; max nearest-sample distance over scrambled "
        "Sobol probes (non-periodic lower estimate), in units of the period",
    )
    metrics: dict[str, Any] = {
        "nodes": nodes,
        "lattice_spacing": _TG_LENGTH / count,
        "fill_distance": fill["fill_distance"] * _TG_LENGTH,
        "steps_requested": steps,
        "step_size": step_size,
        "initial_status": MeshfreeFlowStatus(int(initial.evidence.status)).name,
        "initial_nodal_divergence_after": float(initial.evidence.nodal_divergence_after),
        "steps_accepted": int(jnp.sum(evidence.successful)),
        "all_steps_accepted": bool(jnp.all(evidence.successful)),
        "run_successful": bool(solution.successful),
        "relative_energy_error": abs(
            float(energy_after[-1]) / energy_before / exact_ratio - 1.0
        ),
        "energy_nonincreasing": bool(
            np.all(np.diff(np.concatenate(([energy_before], energy_after))) <= 0.0)
        ),
        "velocity_rms_error": _weighted_rms(volumes, velocity_error),
        "velocity_max_error": float(jnp.max(jnp.abs(velocity_error))),
        "pressure_rms_error": _weighted_rms(volumes, pressure_error),
        "max_graph_divergence": float(jnp.max(evidence.graph_divergence_norm)),
        "max_nodal_divergence": float(jnp.max(evidence.nodal_divergence_after)),
        "max_momentum": float(jnp.max(jnp.abs(evidence.momentum_after))),
        "relative_mass_defect": float(
            np.max(np.abs(mass_after - mass_before)) / np.max(np.abs(mass_before))
        ),
        "max_pressure_residual": float(jnp.max(evidence.pressure_residual_norm)),
        "pressure_iterations": int(jnp.sum(evidence.pressure_iterations)),
        "max_cfl": float(jnp.max(evidence.cfl)),
        "smooth_cycle_fraction": float(jnp.max(evidence.cycles.nonphysical_fraction)),
        "smooth_flow_classified_physical": bool(jnp.all(evidence.cycles.physical)),
        "cycle_tolerance": _TG_CYCLE_TOLERANCE,
        "cycle_space_dimension": flow.plan.cycle_space_dimension,
        "artificial_cycle_dimension": int(evidence.cycles.artificial_cycle_dimension),
        "noise_cycle_fraction": float(noise_cycles.nonphysical_fraction),
        "noise_classified_physical": bool(noise_cycles.physical),
        "projection_status": CompatibleProjectionStatus(
            int(noise_projection.status)
        ).name,
        "projection_idempotence_defect": float(
            jnp.linalg.norm(reprojected.velocity - projected) / jnp.linalg.norm(projected)
        ),
        "oracle_provenance": "closed-form Taylor-Green vortex u = e^{-2 nu t}(sin x cos y, "
        "-cos x sin y), p = e^{-4 nu t}(cos 2x + cos 2y)/4, KE ratio e^{-4 nu t}",
    }
    return {
        "workload": "incompressible-taylor-green",
        "capacity": nodes,
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": 2,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": metrics,
        "compiler": execution["compiler"],
        "retained_bytes": logical_array_bytes((exterior, flow, solution)),
        "reserved_bytes": reservation,
        "oracle_provenance": metrics["oracle_provenance"],
        "consumer": "phydrax.solver.MeshfreeIncompressibleFlowPlan + solve_fixed_step",
    }


def measure_incompressible_refusals(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """CFL refusal with rollback/resume and an incompatible pressure source."""
    _require_float64(config)
    count = _side(capacity, 8)
    nodes = count * count
    reservation = config.check_capacity(nodes, degree=3)
    recorder = PhaseRecorder()
    exterior, address = _tg_exterior(count, recorder)
    flow = _tg_flow(exterior, address, recorder)
    initial = flow.initialize(_tg_velocity(flow.points, 0.0)).state
    accepted = recorder.run(
        "solve", lambda: flow.advance(initial, 0.0, 0.05), scope="admitted-step"
    )
    state = accepted.state
    # 50 time units is ~100x the CFL-admitted step for |u| = 1, h = 2 pi / n.
    refused = recorder.run(
        "solve", lambda: flow.advance(state, 0.05, 50.0), scope="cfl-violating-step"
    )
    resumed = recorder.run(
        "solve", lambda: flow.advance(refused.state, 0.05, 0.05), scope="resumed-step"
    )
    projection = flow.plan.projection
    edges = jnp.asarray(
        np.random.default_rng(seed).standard_normal(exterior.lengths.shape),
        dtype=jnp.float64,
    )

    # A uniform source does not integrate to zero against the degree-zero
    # Hodge measure of the closed (periodic) component: incompatible.
    def incompatible_projection() -> IncompressibleProjectionResult:
        target = jnp.ones(nodes, dtype=jnp.float64)
        if isinstance(projection, CompatibleIncompressibleProjection):
            return projection.project(edges, target_divergence=target)
        if isinstance(projection, CompatibleVariableDensityProjection):
            return projection.project(edges, state.density, target_divergence=target)
        raise TypeError(
            "Flow refusal diagnostics require a compatible pressure projection."
        )

    incompatible = recorder.run(
        "solve", incompatible_projection, scope="incompatible-pressure-source"
    )
    metrics: dict[str, Any] = {
        "nodes": nodes,
        "admitted_status": MeshfreeFlowStatus(int(accepted.evidence.status)).name,
        "refused_status": MeshfreeFlowStatus(int(refused.evidence.status)).name,
        "cfl_refused": int(refused.evidence.status)
        == int(MeshfreeFlowStatus.CFL_REFUSED),
        "refused_cfl": float(refused.evidence.cfl),
        "refused_state_unchanged": bool(
            jnp.all(refused.state.velocity == state.velocity)
            & jnp.all(refused.state.pressure == state.pressure)
            & jnp.all(refused.state.volume_flux == state.volume_flux)
        ),
        "resumed_accepted": int(resumed.evidence.status)
        == int(MeshfreeFlowStatus.ACCEPTED),
        "incompatible_status": CompatibleProjectionStatus(int(incompatible.status)).name,
        "incompatible_source_refused": int(incompatible.status)
        == int(CompatibleProjectionStatus.INCOMPATIBLE_SOURCE),
        "incompatible_velocity_unchanged": bool(jnp.all(incompatible.velocity == edges)),
        "incompatible_pressure_nan": bool(jnp.all(jnp.isnan(incompatible.pressure))),
        "incompatible_compatibility_residual": float(incompatible.compatibility_residual),
    }
    return {
        "workload": "incompressible-refusals",
        "capacity": nodes,
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": 2,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": metrics,
        "retained_bytes": logical_array_bytes((exterior, flow)),
        "reserved_bytes": reservation,
        "oracle_provenance": "documented MeshfreeFlowStatus.CFL_REFUSED and "
        "CompatibleProjectionStatus.INCOMPATIBLE_SOURCE",
        "consumer": "PreparedMeshfreeIncompressibleFlow.advance; "
        "CompatibleIncompressibleProjection.project",
    }


# Declared symbolic-work budget of the expected coercivity refusal: below the
# AMD Cholesky work of any admitted periodic lattice, so the native factor must
# refuse before allocation instead of factoring with a truncated budget.
COERCIVITY_REFUSAL_SYMBOLIC_WORK = 1_000


def measure_exterior_coercivity_refusal(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """The coercivity factor of a declared, insufficient budget refuses before allocation."""
    _require_float64(config)
    count = _side(capacity, 8)
    nodes = count * count
    reservation = config.check_capacity(nodes, degree=3)
    recorder = PhaseRecorder()
    policy = MeshfreeCoercivityPolicy(
        factorization=la.SparseFactorizationPolicy(
            "cholesky",
            ordering="approximate-minimum-degree",
            max_symbolic_work=COERCIVITY_REFUSAL_SYMBOLIC_WORK,
        )
    )
    message: str | None = None
    try:
        _tg_exterior(count, recorder, coercivity=policy)
    except la.LinearCapabilityError as error:  # The documented typed refusal only.
        message = str(error)
    metrics: dict[str, Any] = {
        "nodes": nodes,
        "declared_symbolic_work": COERCIVITY_REFUSAL_SYMBOLIC_WORK,
        "coercivity_refused": message is not None,
        "refused_before_allocation": message is not None
        and "refused before allocation" in message,
        "refusal_message": message,
    }
    return {
        "workload": "exterior-coercivity-refusal",
        "capacity": nodes,
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": 2,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": metrics,
        "retained_bytes": 0,
        "reserved_bytes": reservation,
        "oracle_provenance": "documented phydrax.linalg.LinearCapabilityError of the "
        "sparse symbolic factorization budget",
        "consumer": "phydrax.discretization.meshfree.MeshfreeCoercivityPolicy",
    }


# ---------------------------------------------------------------------------
# Q10 bounded generalized Stokes (manufactured velocity/pressure/traction)

_STOKES_VISCOSITY = 1.0


def _stokes_velocity(x: Array) -> Array:
    return jnp.stack(
        (
            jnp.sin(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1]),
            -jnp.cos(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
        )
    )


def _stokes_pressure(x: Array) -> Array:
    return jnp.cos(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1])


def _stokes_stress(x: Array) -> Array:
    gradient = jax.jacfwd(_stokes_velocity)(x)
    return _STOKES_VISCOSITY * (gradient + gradient.T) - _stokes_pressure(x) * jnp.eye(
        2, dtype=jnp.float64
    )


def _stokes_force(x: Array) -> Array:
    return -jnp.trace(jax.jacfwd(_stokes_stress)(x), axis1=1, axis2=2)


def measure_incompressible_stokes_bounded(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Clamped manufactured Stokes on a jittered unit square (PSPG mixed owner)."""
    _require_float64(config)
    side = _side(capacity, 5)
    nodes = side * side
    reservation = config.check_capacity(nodes, degree=3)
    recorder = PhaseRecorder()
    cloud = recorder.run(
        "local-fit", lambda: _trapezoid_plate(side, seed), scope="phs3-plate-cloud"
    )
    recorder.unavailable("search", "Fused into PointCloudPlan.prepare (local-fit)")
    x = cloud.points
    exact = jax.vmap(_stokes_velocity)(x)
    pressure = jax.vmap(_stokes_pressure)(x)
    rows = np.flatnonzero(np.asarray(cloud.plan.boundary_mask))
    boundary = PointBoundaryPlan(
        tuple(
            PointBoundaryCondition(
                "dirichlet", rows, exact[rows, a], label=f"wall-{a}", component=a
            )
            for a in range(2)
        ),
        row_count=nodes,
        components=2,
    )
    prepared = recorder.run(
        "assembly",
        lambda: MeshfreeGeneralizedStokesPlan(
            cloud, boundary, shear_modulus=_STOKES_VISCOSITY
        ).prepare(),
        scope="mixed-operator-and-block-ilu-preconditioner",
    )
    recorder.unavailable(
        "ordering-fill", "Block ILU setup is fused into the mixed plan prepare (assembly)"
    )
    force = jax.vmap(_stokes_force)(x)
    result = recorder.run_repeated(
        "solve",
        lambda: prepared.solve(force),
        repeats=config.repeats,
        scope="native-gmres-saddle-solve",
    )
    weights = cloud.quadrature_weights
    host = np.asarray(x)
    face_errors: list[float] = []
    face_scale: list[float] = []
    stress = jax.vmap(_stokes_stress)(x)
    for axis, value, sign in (
        (0, 0.0, -1.0),
        (0, 1.0, 1.0),
        (1, 0.0, -1.0),
        (1, 1.0, 1.0),
    ):
        face = _face_rows(host, axis, value)
        normal = np.zeros(2, dtype=np.float64)
        normal[axis] = sign
        expected = stress[face] @ jnp.asarray(normal)
        face_errors.append(
            float(jnp.max(jnp.abs(result.boundary_traction[face] - expected)))
        )
        face_scale.append(float(jnp.max(jnp.abs(expected))))
    fill = _unit_square_fill(host, seed)
    metrics: dict[str, Any] = {
        "nodes": nodes,
        "fill_distance": fill["fill_distance"],
        "status": MechanicsStatus(int(result.status)).name,
        "accepted": bool(result.successful),
        "linear_iterations": int(result.linear.diagnostics.iterations),
        "residual_within_tolerance": bool(
            (result.momentum_residual_norm <= result.residual_tolerance)
            & (result.pressure_residual_norm <= result.residual_tolerance)
        ),
        "velocity_rms_error": _weighted_rms(weights, result.field - exact),
        "velocity_relative_error": _relative(result.field, exact, weights),
        "pressure_rms_error": _weighted_rms(
            weights, result.pressure - _centered(weights, pressure)
        ),
        "pressure_relative_error": _relative(
            result.pressure, _centered(weights, pressure), weights
        ),
        "volumetric_defect": float(result.volumetric_defect),
        "max_divergence": float(jnp.max(jnp.abs(result.divergence))),
        "pressure_oscillation": float(result.pressure_oscillation),
        "compatibility_residual": float(result.compatibility_residual),
        "traction_face_relative_error": max(face_errors) / max(face_scale),
        "oracle_provenance": "manufactured u = (sin pi x cos pi y, -cos pi x sin pi y), "
        "p = cos pi x cos pi y; forcing and traction by autodiff",
    }
    return {
        "workload": "incompressible-stokes-bounded",
        "capacity": nodes,
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": 2,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": metrics,
        "retained_bytes": logical_array_bytes((cloud, prepared)),
        "reserved_bytes": reservation,
        "oracle_provenance": metrics["oracle_provenance"],
        "consumer": "phydrax.discretization.meshfree.MeshfreeGeneralizedStokesPlan",
    }


# ---------------------------------------------------------------------------
# Q10 Lagrangian GMLS particle flow, SPH and measure-transfer interoperability

_LAGRANGIAN_WAVE = 2.0 * math.pi
_LAGRANGIAN_SHELLS = 21  # complete square-lattice distance shells
_LAGRANGIAN_STEPS = 3


def _unit_periodic() -> MortonAddressPlan:
    return MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 10, periodic_axes=(True, True))


def _cell_lattice(side: int, /) -> np.ndarray:
    axis = (np.arange(side, dtype=np.float64) + 0.5) / side
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((x.ravel(), y.ravel()), axis=1)


def _lagrangian_vortex(points: np.ndarray, /) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    return np.stack(
        (
            np.sin(_LAGRANGIAN_WAVE * x) * np.cos(_LAGRANGIAN_WAVE * y),
            -np.cos(_LAGRANGIAN_WAVE * x) * np.sin(_LAGRANGIAN_WAVE * y),
        ),
        axis=1,
    )


def _gradient_perturbation(points: np.ndarray, /) -> np.ndarray:
    """Gradient of ``phi = 0.1 cos(2 pi x) cos(2 pi y)``."""
    x, y = points[:, 0], points[:, 1]
    return (
        -0.1
        * _LAGRANGIAN_WAVE
        * np.stack(
            (
                np.sin(_LAGRANGIAN_WAVE * x) * np.cos(_LAGRANGIAN_WAVE * y),
                np.cos(_LAGRANGIAN_WAVE * x) * np.sin(_LAGRANGIAN_WAVE * y),
            ),
            axis=1,
        )
    )


def _lagrangian_record(
    workload: str,
    side: int,
    capacity: int,
    seed: int,
    recorder: PhaseRecorder,
    metrics: dict[str, Any],
    retained: Any,
    reservation: int,
    oracle: str,
    consumer: str,
    /,
) -> dict[str, Any]:
    return {
        "workload": workload,
        "capacity": side * side,
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": 2,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": metrics,
        "retained_bytes": logical_array_bytes(retained),
        "reserved_bytes": reservation,
        "oracle_provenance": oracle,
        "consumer": consumer,
    }


def _volume_flow(
    points: np.ndarray, recorder: PhaseRecorder, viscosity: float, /
) -> PreparedMeshfreeLagrangianFlow:
    return recorder.run(
        "local-fit",
        lambda: MeshfreeLagrangianFlowPlan(
            "quadrature-volume",
            reference_density=1.0,
            address=_unit_periodic(),
            neighbors=_LAGRANGIAN_SHELLS,
            viscosity=viscosity,
        ).prepare(points, volumes=np.full(points.shape[0], 1.0 / points.shape[0])),
        scope="anchored-gmls-support-epoch",
    )


def measure_lagrangian_quadrature_volume(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Weak GMLS projection of a gradient perturbation and a short Taylor–Green run."""
    _require_float64(config)
    side = _side(capacity, 8)
    count = side * side
    reservation = config.check_capacity(count, degree=3, neighbors=_LAGRANGIAN_SHELLS)
    recorder = PhaseRecorder()
    recorder.unavailable("search", "Fused into the Lagrangian support-epoch prepare")
    recorder.unavailable(
        "assembly",
        "Per-step cloud refresh and Poisson assembly are fused into "
        "PreparedMeshfreeLagrangianFlow.step (recorded as solve)",
    )
    points = _cell_lattice(side)
    # Inviscid, unforced: the predictor is the input, so the projection alone
    # must remove exactly the gradient perturbation.
    inviscid = _volume_flow(points, recorder, 0.0)
    vortex = _lagrangian_vortex(points)
    perturbation = _gradient_perturbation(points)
    perturbed = inviscid.initialize(vortex + perturbation)
    projected = recorder.run(
        "solve",
        lambda: inviscid.step(perturbed, 0.01),
        scope="gradient-perturbation-step",
    )
    removed = vortex + perturbation - np.asarray(projected.candidate.velocity)
    vortex_energy = 0.5 * float(np.sum(np.asarray(perturbed.masses)[:, None] * vortex**2))
    flow = _volume_flow(points, recorder, 0.01)
    step_size = 0.01 * 12.0 / side
    state = flow.initialize(vortex)
    statuses: list[int] = []
    mass_defects: list[float] = []
    divergences: list[float] = []
    energy_monotone: list[bool] = []
    volume_defects: list[float] = []
    for index in range(_LAGRANGIAN_STEPS):
        result = recorder.run(
            "solve",
            lambda: flow.step(state, step_size),
            scope=f"taylor-green-step-{index}",
        )
        evidence = result.evidence
        statuses.append(int(evidence.status))
        mass_defects.append(
            abs(float(evidence.mass_after) - float(evidence.mass_before))
            / float(evidence.mass_before)
        )
        divergences.append(float(evidence.divergence_after))
        energy_monotone.append(
            float(evidence.kinetic_energy_after)
            <= float(evidence.kinetic_energy_predicted) * (1.0 + 1e-12)
        )
        volume_defects.append(
            abs(float(evidence.volume_after) - float(evidence.volume_before))
            / float(evidence.volume_before)
        )
        flow, state = result.flow, result.state
    metrics: dict[str, Any] = {
        "particles": count,
        "projection_status": MeshfreeLagrangianStatus(
            int(projected.evidence.status)
        ).name,
        "projection_weak_divergence_ratio": float(projected.evidence.divergence_after)
        / float(projected.evidence.divergence_before),
        "projection_removed_gradient_error": float(
            np.linalg.norm(removed - perturbation) / np.linalg.norm(perturbation)
        ),
        "projection_energy_relative_error": abs(
            float(projected.evidence.kinetic_energy_after) / vortex_energy - 1.0
        ),
        "steps_accepted": sum(
            status == int(MeshfreeLagrangianStatus.ACCEPTED) for status in statuses
        ),
        "all_steps_accepted": all(
            status == int(MeshfreeLagrangianStatus.ACCEPTED) for status in statuses
        ),
        "step_statuses": [MeshfreeLagrangianStatus(status).name for status in statuses],
        "max_relative_mass_defect": max(mass_defects),
        "max_weak_divergence_after": max(divergences),
        "projection_energy_nonincreasing": all(energy_monotone),
        "max_relative_volume_defect": max(volume_defects),
        "step_size": step_size,
    }
    return _lagrangian_record(
        "lagrangian-quadrature-volume",
        side,
        capacity,
        seed,
        recorder,
        metrics,
        (flow, state),
        reservation,
        "closed-form gradient perturbation grad(0.1 cos 2 pi x cos 2 pi y) of a periodic "
        "Taylor-Green vortex; constant masses (exact mass ledger)",
        "phydrax.solver.MeshfreeLagrangianFlowPlan('quadrature-volume')",
    )


def _sph(points: np.ndarray, side: int, /) -> MeshfreeSPHReconstruction:
    count = points.shape[0]
    spacing = 1.0 / side
    particles = dsc.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), 1.0 / count), ambient_dimension=2
    ).prepare()
    box = dsc.ParticleBox(
        jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        jnp.asarray([1.0, 1.0], dtype=jnp.float64),
    )
    verlet = dsc.VerletParticleNeighborhoodPlan(
        dsc.DenseParticleNeighborhoodPlan(count * (count - 1) // 2, box=box),
        2.6 * spacing,
        0.4 * spacing,
    ).prepare(particles)
    return MeshfreeSPHReconstruction(
        particles, verlet, dsc.WendlandC2SPHKernel(2), 1.3 * spacing
    )


def measure_lagrangian_material_mass_sph(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """SPH versus GMLS density/divergence, measure conversion and material-mass steps."""
    _require_float64(config)
    side = _side(capacity, 8)
    count = side * side
    reservation = config.check_capacity(count, degree=3, neighbors=_LAGRANGIAN_SHELLS)
    # Dense all-pairs SPH candidate relation: i32 pairs plus f64 geometry per pair.
    reservation += declare_reservation(
        (count * (count - 1) // 2) * (2 * 4 + 4 * 8),
        config,
        scope="dense SPH pair relation",
    )
    recorder = PhaseRecorder()
    points = _cell_lattice(side)
    sph = recorder.run("search", lambda: _sph(points, side), scope="sph-verlet-relation")
    cache = recorder.run(
        "search", lambda: sph.initialize(points), scope="verlet-initialize"
    )
    cloud = recorder.run(
        "local-fit",
        lambda: PointCloudPlan(
            points,
            np.full(count, 1.0 / count),
            stencil=LocalStencilPolicy(polynomial_degree=3),
            neighbors=_LAGRANGIAN_SHELLS,
            address=_unit_periodic(),
        ).prepare(),
        scope="gmls-cloud",
    )
    masses = np.full(count, 1.0 / count)
    x, y = points[:, 0], points[:, 1]
    velocity = 0.1 * np.stack(
        (np.sin(_LAGRANGIAN_WAVE * x), np.sin(_LAGRANGIAN_WAVE * y)), axis=1
    )
    divergence = (
        0.1
        * _LAGRANGIAN_WAVE
        * (np.cos(_LAGRANGIAN_WAVE * x) + np.cos(_LAGRANGIAN_WAVE * y))
    )
    comparison = recorder.run(
        "transfer",
        lambda: sph.compare(cloud, points, velocity, masses, masses, cache),
        scope="sph-gmls-reconstruction-exchange",
    )
    conversion = recorder.run(
        "transfer",
        lambda: sph.convert(points, masses, masses, cache, target="material-mass"),
        scope="measure-conversion",
    )
    norm = float(np.sqrt(np.mean(divergence**2)))
    population = dsc.ParticlePopulationPlan(sph.particles).initialize()
    material = MaterialParticleMeasure(
        sph.particles, population, 1.0, neighbors=sph.neighbors, reference=cache
    )
    flow = recorder.run(
        "local-fit",
        lambda: MeshfreeLagrangianFlowPlan(
            "material-mass",
            reference_density=1.0,
            address=_unit_periodic(),
            neighbors=_LAGRANGIAN_SHELLS,
            material=material,
            sph=sph,
        ).prepare(points),
        scope="material-mass-support-epoch",
    )
    state = flow.initialize(_lagrangian_vortex(points))
    initial_masses = np.asarray(state.masses)
    exact_mass: list[bool] = []
    density_mismatch: list[float] = []
    statuses: list[int] = []
    for index in range(_LAGRANGIAN_STEPS):
        result = recorder.run(
            "solve",
            lambda: flow.step(state, 0.01 * 12.0 / side),
            scope=f"material-step-{index}",
        )
        evidence = result.evidence
        statuses.append(int(evidence.status))
        exact_mass.append(
            float(evidence.mass_after) == float(evidence.mass_before)
            and bool(np.array_equal(np.asarray(result.state.masses), initial_masses))
        )
        density_mismatch.append(
            math.nan
            if evidence.sph_density_mismatch is None
            else float(evidence.sph_density_mismatch)
        )
        flow, state = result.flow, result.state
    metrics: dict[str, Any] = {
        "particles": count,
        "sph_neighbors_successful": bool(comparison.neighbors_successful),
        "sph_partition_density_mismatch": float(comparison.density_mismatch),
        "gmls_divergence_relative_error": float(
            np.sqrt(np.mean((np.asarray(comparison.gmls_divergence) - divergence) ** 2))
        )
        / norm,
        "sph_divergence_relative_error": float(
            np.sqrt(np.mean((np.asarray(comparison.sph_divergence) - divergence) ** 2))
        )
        / norm,
        "sph_gmls_divergence_mismatch_relative": float(comparison.divergence_mismatch)
        / norm,
        "conversion_mass_defect": abs(float(conversion.mass_defect)),
        "conversion_masses_unchanged": bool(
            np.array_equal(np.asarray(conversion.masses), masses)
        ),
        "material_steps_accepted": all(
            status == int(MeshfreeLagrangianStatus.ACCEPTED) for status in statuses
        ),
        "material_step_statuses": [
            MeshfreeLagrangianStatus(status).name for status in statuses
        ],
        "material_mass_exact": all(exact_mass),
        "material_sph_volume_mismatch": max(density_mismatch),
    }
    return _lagrangian_record(
        "lagrangian-material-mass-sph",
        side,
        capacity,
        seed,
        recorder,
        metrics,
        (sph, cloud, flow, state),
        reservation,
        "closed-form divergence of 0.1(sin 2 pi x, sin 2 pi y); material masses; "
        "V = m / rho_SPH identity",
        "phydrax.solver.MeshfreeSPHReconstruction + MeshfreeLagrangianFlowPlan('material-mass')",
    )


def measure_lagrangian_measure_transfer(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Conservative-positive material-mass -> quadrature-volume transfer ledger."""
    _require_float64(config)
    side = _side(capacity, 8)
    count = side * side
    target_side = side // 2
    reservation = config.check_capacity(count, degree=2, neighbors=9)
    recorder = PhaseRecorder()
    generator = np.random.default_rng(seed)
    source = _cell_lattice(side) + generator.uniform(-0.1, 0.1, (count, 2)) / side
    source_volumes = np.full(count, 1.0 / count)
    density = 1.0 + 0.2 * np.sin(_LAGRANGIAN_WAVE * source[:, 0])
    masses = density * source_volumes
    velocity = _lagrangian_vortex(source)
    target = _cell_lattice(target_side)
    prepared = recorder.run(
        "transfer",
        lambda: MeshfreeMeasureTransferPlan(
            source,
            source_volumes,
            target,
            np.full(target.shape[0], 1.0 / target.shape[0]),
            source_measure="material-mass",
            target_measure="quadrature-volume",
            address=_unit_periodic(),
        ).prepare(),
        scope="cross-target-stencils-and-audited-transfer",
    )
    recorder.unavailable(
        "local-fit", "Cross-target stencils are fused into the transfer prepare"
    )
    admitted = bool(prepared.admitted)
    metrics: dict[str, Any] = {
        "particles": count,
        "target_points": target.shape[0],
        "transfer_status": prepared.status.name,
        "transfer_admitted": admitted,
    }
    if admitted:
        moved = recorder.run(
            "transfer",
            lambda: prepared.apply(masses, velocity),
            scope="apply-mass-momentum",
        )
        moved_masses = np.asarray(moved.masses)
        momentum = moved_masses[:, None] * np.asarray(moved.velocity)
        source_momentum = np.sum(masses[:, None] * velocity, axis=0)
        metrics.update(
            {
                "transfer_positive": bool(moved.positive),
                "relative_mass_defect": abs(float(np.sum(moved_masses) - np.sum(masses)))
                / float(np.sum(masses)),
                "momentum_defect": float(
                    np.max(np.abs(np.sum(momentum, axis=0) - source_momentum))
                ),
                "reported_density_mismatch": float(moved.density_mismatch),
                "center_of_mass_defect": float(
                    jnp.max(jnp.abs(moved.center_of_mass_defect))
                ),
            }
        )
    return _lagrangian_record(
        "lagrangian-measure-transfer",
        side,
        capacity,
        seed,
        recorder,
        metrics,
        prepared,
        reservation,
        "exact source totals of mass and momentum (host NumPy sums)",
        "phydrax.solver.MeshfreeMeasureTransferPlan",
    )


def measure_lagrangian_refusals(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Courant refusal with kept state and resume; uncovered-source transfer refusal."""
    _require_float64(config)
    side = _side(capacity, 8)
    count = side * side
    reservation = config.check_capacity(count, degree=3, neighbors=_LAGRANGIAN_SHELLS)
    recorder = PhaseRecorder()
    points = _cell_lattice(side)
    flow = _volume_flow(points, recorder, 0.0)
    state = flow.initialize(_lagrangian_vortex(points))
    # |u| = 1 and h = 1/side: dt = 0.3 is a Courant number of 0.3 side >> 0.5.
    refused = recorder.run(
        "solve", lambda: flow.step(state, 0.3), scope="courant-violating"
    )
    kept = refused.state
    continued = recorder.run(
        "solve", lambda: refused.flow.step(kept, 0.01 * 12.0 / side), scope="resumed-step"
    )
    generator = np.random.default_rng(seed)
    source = _cell_lattice(side) + generator.uniform(-0.1, 0.1, (count, 2)) / side
    half = _cell_lattice(side // 2) * np.asarray([0.5, 1.0])
    uncovered = recorder.run(
        "transfer",
        lambda: MeshfreeMeasureTransferPlan(
            source,
            np.full(count, 1.0 / count),
            half,
            np.full(half.shape[0], 0.5 / half.shape[0]),
            source_measure="material-mass",
            target_measure="quadrature-volume",
            address=_unit_periodic(),
        ).prepare(),
        scope="uncovered-target",
    )
    apply_refused = False
    try:
        uncovered.apply(np.full(count, 1.0 / count), _lagrangian_vortex(source))
    except ValueError as error:  # Documented: apply refuses a non-admitted transfer.
        if "UNCOVERED_SOURCE" not in str(error):
            raise
        apply_refused = True
    metrics: dict[str, Any] = {
        "particles": count,
        "refused_status": MeshfreeLagrangianStatus(int(refused.evidence.status)).name,
        "courant_refused": int(refused.evidence.status)
        == int(MeshfreeLagrangianStatus.COURANT_REFUSED),
        "refused_state_kept": all(
            bool(np.array_equal(np.asarray(a), np.asarray(b)))
            for a, b in (
                (kept.positions, state.positions),
                (kept.velocity, state.velocity),
                (kept.masses, state.masses),
                (kept.volumes, state.volumes),
                (kept.pressure, state.pressure),
                (kept.time, state.time),
            )
        ),
        "continued_accepted": int(continued.evidence.status)
        == int(MeshfreeLagrangianStatus.ACCEPTED),
        "transfer_status": uncovered.status.name,
        "uncovered_source_refused": uncovered.status
        is PointTransferStatus.UNCOVERED_SOURCE,
        "uncovered_apply_refused": apply_refused,
    }
    return _lagrangian_record(
        "lagrangian-refusals",
        side,
        capacity,
        seed,
        recorder,
        metrics,
        (flow, kept),
        reservation,
        "documented MeshfreeLagrangianStatus.COURANT_REFUSED and "
        "PointTransferStatus.UNCOVERED_SOURCE",
        "PreparedMeshfreeLagrangianFlow.step; MeshfreeMeasureTransferPlan",
    )


# ---------------------------------------------------------------------------
# Q11 small-strain, finite-strain and mixed elasticity on a jittered plate

_LAMBDA, _MU = 1.0, 0.5


def _hooke_tensor() -> LinearElasticityTensor:
    return LinearElasticityTensor.isotropic(2, lame_lambda=_LAMBDA, shear_modulus=_MU)


def _smooth_displacement(x: Array) -> Array:
    return jnp.stack(
        (0.1 * jnp.sin(2.0 * x[0]) * jnp.exp(x[1]), 0.05 * jnp.cos(x[0] + 2.0 * x[1]))
    )


def _hooke_stress(x: Array) -> Array:
    gradient = jax.jacfwd(_smooth_displacement)(x)
    strain = 0.5 * (gradient + gradient.T)
    return (
        _LAMBDA * jnp.trace(strain) * jnp.eye(2, dtype=jnp.float64) + 2.0 * _MU * strain
    )


def _traction_face_boundary(
    cloud: PreparedPointCloudDiscretization, /
) -> PointBoundaryPlan:
    """Exact traction on ``x = 1`` (including corners); exact displacement elsewhere."""
    x = cloud.points
    exact = jax.vmap(_smooth_displacement)(x)
    right = np.isclose(np.asarray(x[:, 0]), 1.0)
    rows = np.flatnonzero(right)
    rest = np.flatnonzero(np.asarray(cloud.plan.boundary_mask) & ~right)
    traction = jax.vmap(_hooke_stress)(x[rows])[:, :, 0]
    normals = np.tile(np.asarray([[1.0, 0.0]]), (rows.size, 1))
    conditions: list[PointBoundaryCondition] = []
    for a in range(2):
        conditions.append(
            PointBoundaryCondition(
                "neumann",
                rows,
                traction[:, a],
                label=f"traction-{a}",
                component=a,
                normals=normals,
            )
        )
        conditions.append(
            PointBoundaryCondition(
                "dirichlet", rest, exact[rest, a], label=f"clamp-{a}", component=a
            )
        )
    return PointBoundaryPlan(conditions, row_count=x.shape[0], components=2)


def _mechanics_record(
    workload: str,
    side: int,
    capacity: int,
    seed: int,
    recorder: PhaseRecorder,
    metrics: dict[str, Any],
    retained: Any,
    reservation: int,
    oracle: str,
    consumer: str,
    /,
) -> dict[str, Any]:
    return {
        "workload": workload,
        "capacity": side * side,
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": 2,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": metrics,
        "retained_bytes": logical_array_bytes(retained),
        "reserved_bytes": reservation,
        "oracle_provenance": oracle,
        "consumer": consumer,
    }


def measure_elasticity_manufactured(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Nonpolynomial manufactured displacement/traction and rigid-body zero strain."""
    _require_float64(config)
    side = _side(capacity, 5)
    count = side * side
    reservation = config.check_capacity(count, degree=3)
    recorder = PhaseRecorder()
    cloud = recorder.run("local-fit", lambda: _trapezoid_plate(side, seed), scope="plate")
    recorder.unavailable("search", "Fused into PointCloudPlan.prepare (local-fit)")
    prepared = recorder.run(
        "assembly",
        lambda: MeshfreeElasticityPlan(
            cloud, _traction_face_boundary(cloud), _hooke_tensor()
        ).prepare(),
        scope="block-elasticity-and-auxiliary-preconditioner",
    )
    recorder.unavailable(
        "ordering-fill",
        "Auxiliary hierarchy setup is fused into elasticity prepare (assembly)",
    )
    x = cloud.points
    force = jax.vmap(
        lambda y: -jnp.trace(jax.jacfwd(_hooke_stress)(y), axis1=1, axis2=2)
    )(x)
    result = recorder.run_repeated(
        "solve",
        lambda: prepared.solve(force),
        repeats=config.repeats,
        scope="gmres-auxiliary",
    )
    exact = jax.vmap(_smooth_displacement)(x)
    weights = cloud.quadrature_weights
    host = np.asarray(x)
    # Traction on the clamped face y = 0 (n = -e_y) is computed, not imposed:
    # on x = 1 it is the prescribed Neumann datum and would prove nothing.
    bottom = _face_rows(host, 1, 0.0)
    face_normals = jnp.tile(jnp.asarray([[0.0, -1.0]], dtype=jnp.float64), (count, 1))
    traction = prepared.traction(result.candidate_displacement, face_normals)[bottom]
    expected = -jax.vmap(_hooke_stress)(x[bottom])[:, :, 1]
    rigid = jnp.stack((0.3 - 0.7 * x[:, 1], -0.2 + 0.7 * x[:, 0]), axis=1)
    fill = _unit_square_fill(host, seed)
    metrics: dict[str, Any] = {
        "points": count,
        "fill_distance": fill["fill_distance"],
        "status": MechanicsStatus(int(result.status)).name,
        "accepted": bool(result.successful),
        "linear_iterations": int(result.block.linear_result.diagnostics.iterations),
        "displacement_max_error": float(
            jnp.max(jnp.abs(result.candidate_displacement - exact))
        ),
        "displacement_relative_error": _relative(
            result.candidate_displacement, exact, weights
        ),
        "traction_face_relative_error": float(
            jnp.max(jnp.abs(traction - expected)) / jnp.max(jnp.abs(expected))
        ),
        "force_balance_defect": float(result.force_balance_defect),
        "clapeyron_defect": float(result.clapeyron_defect),
        "rigid_max_strain": float(jnp.max(jnp.abs(prepared.strain(rigid)))),
        "rigid_max_stress": float(jnp.max(jnp.abs(prepared.stress(rigid)))),
        "rigid_strain_energy": float(prepared.strain_energy(rigid)),
    }
    return _mechanics_record(
        "elasticity-manufactured",
        side,
        capacity,
        seed,
        recorder,
        metrics,
        (cloud, prepared),
        reservation,
        "manufactured u = (0.1 sin 2x e^y, 0.05 cos(x+2y)); Hooke stress, body force "
        "and traction by autodiff; rigid motion (0.3-0.7y, -0.2+0.7x)",
        "phydrax.discretization.meshfree.MeshfreeElasticityPlan",
    )


def _rollers(
    cloud: PreparedPointCloudDiscretization, pull: float, /
) -> PointBoundaryPlan:
    """Rollers on ``x = 0`` and ``y = 0``; traction ``(pull, 0)`` on ``x = 1``."""
    x = np.asarray(cloud.points)
    left, right = np.isclose(x[:, 0], 0.0), np.isclose(x[:, 0], 1.0)
    bottom, top = np.isclose(x[:, 1], 0.0), np.isclose(x[:, 1], 1.0)

    def traction(
        label: str,
        mask: np.ndarray,
        normal: tuple[float, float],
        component: int,
        value: float,
    ) -> PointBoundaryCondition:
        rows = np.flatnonzero(mask)
        return PointBoundaryCondition(
            "neumann",
            rows,
            value,
            label=label,
            component=component,
            normals=np.tile(np.asarray(normal, dtype=np.float64), (rows.size, 1)),
        )

    sides = ~left & ~right
    return PointBoundaryPlan(
        (
            PointBoundaryCondition("dirichlet", np.flatnonzero(left), label="roller-x"),
            traction("pull", right, (1.0, 0.0), 0, pull),
            traction("free-x-bottom", bottom & sides, (0.0, -1.0), 0, 0.0),
            traction("free-x-top", top & sides, (0.0, 1.0), 0, 0.0),
            PointBoundaryCondition(
                "dirichlet", np.flatnonzero(bottom), label="roller-y", component=1
            ),
            traction("free-y-top", top, (0.0, 1.0), 1, 0.0),
            traction("free-y-left", left & ~bottom & ~top, (-1.0, 0.0), 1, 0.0),
            traction("free-y-right", right & ~bottom & ~top, (1.0, 0.0), 1, 0.0),
        ),
        row_count=x.shape[0],
        components=2,
    )


def _uniaxial_neo_hookean(pull: float, /) -> tuple[float, float, float]:
    """Independent host Newton for ``P_xx = pull, P_yy = 0`` of the neo-Hookean law."""
    stretch = np.ones(2, dtype=np.float64)
    for _ in range(50):
        a, b = stretch
        log_j = np.log(a * b)
        residual = np.asarray(
            [
                _MU * (a - 1.0 / a) + _LAMBDA * log_j / a - pull,
                _MU * (b - 1.0 / b) + _LAMBDA * log_j / b,
            ]
        )
        cross = _LAMBDA / (a * b)
        jacobian = np.asarray(
            [
                [_MU * (1.0 + a**-2) + _LAMBDA * (1.0 - log_j) / a**2, cross],
                [cross, _MU * (1.0 + b**-2) + _LAMBDA * (1.0 - log_j) / b**2],
            ]
        )
        stretch = stretch - np.linalg.solve(jacobian, residual)
    a, b = stretch
    log_j = np.log(a * b)
    energy = 0.5 * _MU * (a**2 + b**2 - 2.0) - _MU * log_j + 0.5 * _LAMBDA * log_j**2
    return float(a), float(b), float(energy)


_TENSION_PULL = 0.05
_FINITE_PULL = 0.25
_LOAD_STEPS = 4


def measure_elasticity_closed_form(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Homogeneous plate tension: linear and neo-Hookean closed forms, energy and work."""
    _require_float64(config)
    side = _side(capacity, 5)
    count = side * side
    reservation = config.check_capacity(count, degree=3)
    recorder = PhaseRecorder()
    cloud = recorder.run("local-fit", lambda: _trapezoid_plate(side, seed), scope="plate")
    recorder.unavailable("search", "Fused into PointCloudPlan.prepare (local-fit)")
    x = cloud.points
    zero = jnp.zeros((count, 2), dtype=jnp.float64)
    boundary = _rollers(cloud, _TENSION_PULL)
    linear = recorder.run(
        "assembly",
        lambda: MeshfreeElasticityPlan(cloud, boundary, _hooke_tensor()).prepare(),
        scope="linear-elasticity",
    )
    result = recorder.run("solve", lambda: linear.solve(zero), scope="linear-tension")
    stiff = _LAMBDA + 2.0 * _MU
    axial = _TENSION_PULL * stiff / (4.0 * _MU * (_LAMBDA + _MU))
    lateral = -_LAMBDA * axial / stiff
    exact = jnp.stack((axial * x[:, 0], lateral * x[:, 1]), axis=1)
    analytic_energy = 0.5 * _TENSION_PULL * axial
    law = NeoHookeanLaw(
        NeoHookeanParameters(
            jnp.asarray(_MU, dtype=jnp.float64), jnp.asarray(_LAMBDA, dtype=jnp.float64)
        )
    )
    finite = recorder.run(
        "assembly",
        lambda: MeshfreeHyperelasticPlan(
            cloud, boundary, law, load_steps=_LOAD_STEPS
        ).prepare(),
        scope="neo-hookean",
    )
    small = 1e-3
    gentle = recorder.run(
        "solve",
        lambda: finite.solve(zero, boundary_values={"pull": small}),
        scope="neo-hookean-small-load",
    )
    reference = linear.solve(zero, boundary_values={"pull": small})
    stretched = recorder.run(
        "solve",
        lambda: finite.solve(zero, boundary_values={"pull": _FINITE_PULL}),
        scope="neo-hookean-finite-load",
    )
    a, b, energy = _uniaxial_neo_hookean(_FINITE_PULL)
    homogeneous = jnp.stack(((a - 1.0) * x[:, 0], (b - 1.0) * x[:, 1]), axis=1)
    metrics: dict[str, Any] = {
        "points": count,
        "linear_status": MechanicsStatus(int(result.status)).name,
        "linear_accepted": bool(result.successful),
        "linear_displacement_error": float(
            jnp.max(jnp.abs(result.candidate_displacement - exact))
        ),
        "linear_energy_relative_error": abs(
            float(result.strain_energy) / analytic_energy - 1.0
        ),
        "linear_work_relative_error": abs(
            float(result.external_work) / (2.0 * analytic_energy) - 1.0
        ),
        "clapeyron_defect": float(result.clapeyron_defect),
        "force_balance_defect": float(result.force_balance_defect),
        "small_load_accepted": bool(gentle.successful),
        "small_load_relative_gap": float(
            jnp.max(jnp.abs(gentle.displacement - reference.displacement))
            / jnp.max(jnp.abs(reference.displacement))
        ),
        "finite_status": MechanicsStatus(int(stretched.status)).name,
        "finite_accepted": bool(stretched.successful),
        "finite_displacement_error": float(
            jnp.max(jnp.abs(stretched.displacement - homogeneous))
        ),
        "finite_energy_relative_error": abs(
            float(stretched.stored_energy) / energy - 1.0
        ),
        "finite_min_jacobian": float(jnp.min(stretched.jacobian)),
        "finite_jacobian_positive": bool(jnp.min(stretched.jacobian) > 0.0),
        "energy_work_defect": float(stretched.energy_work_defect),
        "newton_iterations": np.asarray(stretched.step_iterations).tolist(),
    }
    return _mechanics_record(
        "elasticity-closed-form",
        side,
        capacity,
        seed,
        recorder,
        metrics,
        (cloud, linear, finite),
        reservation,
        "closed-form plane-strain uniaxial stress (U = t eps_xx / 2, W = 2U) and an "
        "independent host Newton solve of the homogeneous neo-Hookean stretches",
        "MeshfreeElasticityPlan + MeshfreeHyperelasticPlan",
    )


_MIXED_MEAN_PRESSURE = 0.2
MIXED_COMPRESSIBILITIES: tuple[tuple[str, float], ...] = (
    ("kappa-1", 1.0),
    ("kappa-1e-2", 1e-2),
    ("kappa-1e-8", 1e-8),
)


def _solenoidal(x: Array) -> Array:
    return 0.05 * jnp.stack(
        (
            jnp.sin(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1]),
            -jnp.cos(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
        )
    )


def _mixed_potential(x: Array) -> Array:
    return (
        -0.1 * jnp.cos(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1]) / (2.0 * jnp.pi**2)
        + _MIXED_MEAN_PRESSURE * (x[0] ** 2 + x[1] ** 2) / 4.0
    )


def _mixed_pressure(x: Array) -> Array:
    return 0.1 * jnp.cos(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1]) + _MIXED_MEAN_PRESSURE


def _mixed_displacement(x: Array, kappa: float) -> Array:
    """``u_s - kappa grad psi`` with ``lap psi = p``: ``div u + kappa p = 0`` exactly."""
    return _solenoidal(x) - kappa * jax.grad(_mixed_potential)(x)


def _mixed_force(x: Array, kappa: float) -> Array:
    def cauchy(y: Array) -> Array:
        gradient = jax.jacfwd(_mixed_displacement)(y, kappa)
        return _MU * (gradient + gradient.T) - _mixed_pressure(y) * jnp.eye(
            2, dtype=jnp.float64
        )

    return -jnp.trace(jax.jacfwd(cauchy)(x), axis1=1, axis2=2)


def measure_elasticity_mixed_locking(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Clamped Herrmann mixed form for kappa = 1, 1e-2, 1e-8 (no-locking evidence)."""
    _require_float64(config)
    side = _side(capacity, 5)
    count = side * side
    reservation = config.check_capacity(count, degree=3)
    recorder = PhaseRecorder()
    cloud = recorder.run("local-fit", lambda: _trapezoid_plate(side, seed), scope="plate")
    recorder.unavailable("search", "Fused into PointCloudPlan.prepare (local-fit)")
    recorder.unavailable(
        "ordering-fill", "Block ILU setup is fused into the mixed plan prepare (assembly)"
    )
    x = cloud.points
    pressure = jax.vmap(_mixed_pressure)(x)
    rows = np.flatnonzero(np.asarray(cloud.plan.boundary_mask))
    weights = cloud.quadrature_weights
    metrics: dict[str, Any] = {"points": count}
    for label, kappa in MIXED_COMPRESSIBILITIES:
        exact = jax.vmap(_mixed_displacement, in_axes=(0, None))(x, kappa)
        boundary = PointBoundaryPlan(
            tuple(
                PointBoundaryCondition(
                    "dirichlet", rows, exact[rows, a], label=f"clamp-{a}", component=a
                )
                for a in range(2)
            ),
            row_count=count,
            components=2,
        )
        prepared = recorder.run(
            "assembly",
            lambda: MeshfreeGeneralizedStokesPlan(
                cloud, boundary, shear_modulus=_MU, compressibility=kappa
            ).prepare(),
            scope=label,
        )
        force = jax.vmap(_mixed_force, in_axes=(0, None))(x, kappa)
        result = recorder.run("solve", lambda: prepared.solve(force), scope=label)
        mean = float(jnp.sum(weights * result.pressure) / jnp.sum(weights))
        metrics.update(
            {
                f"{label}_status": MechanicsStatus(int(result.status)).name,
                f"{label}_accepted": bool(result.successful),
                f"{label}_linear_iterations": int(result.linear.diagnostics.iterations),
                f"{label}_displacement_relative_error": _relative(
                    result.field, exact, weights
                ),
                f"{label}_pressure_relative_error": _relative(
                    result.pressure, pressure, weights
                ),
                f"{label}_mean_pressure_error": abs(mean - _MIXED_MEAN_PRESSURE),
                f"{label}_compatibility_residual": float(result.compatibility_residual),
                f"{label}_volumetric_defect": float(result.volumetric_defect),
                f"{label}_pressure_oscillation": float(result.pressure_oscillation),
            }
        )
    # Locking would amplify the displacement error as kappa = 1/lambda -> 0.
    metrics["locking_ratio"] = (
        metrics["kappa-1e-8_displacement_relative_error"]
        / metrics["kappa-1_displacement_relative_error"]
    )
    return _mechanics_record(
        "elasticity-mixed-locking",
        side,
        capacity,
        seed,
        recorder,
        metrics,
        cloud,
        reservation,
        "manufactured Herrmann pair u = u_s - kappa grad psi, p = lap psi (mean 0.2) with "
        "div u + kappa p = 0 exactly; forcing by autodiff",
        "phydrax.discretization.meshfree.MeshfreeGeneralizedStokesPlan(compressibility=kappa)",
    )


def measure_mechanics_refusals(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Newton-budget rollback, floating-body and unstabilized mixed-form refusals."""
    _require_float64(config)
    side = _side(capacity, 5)
    count = side * side
    reservation = config.check_capacity(count, degree=3)
    recorder = PhaseRecorder()
    cloud = recorder.run("local-fit", lambda: _trapezoid_plate(side, seed), scope="plate")
    zero = jnp.zeros((count, 2), dtype=jnp.float64)
    law = NeoHookeanLaw(
        NeoHookeanParameters(
            jnp.asarray(_MU, dtype=jnp.float64), jnp.asarray(_LAMBDA, dtype=jnp.float64)
        )
    )
    strict = recorder.run(
        "assembly",
        lambda: MeshfreeHyperelasticPlan(
            cloud,
            _rollers(cloud, _TENSION_PULL),
            law,
            load_steps=2,
            termination=NonlinearTermination(
                absolute_residual=1e-14, relative_residual=1e-15, maximum_steps=1
            ),
        ).prepare(),
        scope="one-iteration-newton-budget",
    )
    refused = recorder.run(
        "solve",
        lambda: strict.solve(zero, boundary_values={"pull": _FINITE_PULL}),
        scope="refused-increment",
    )
    host = np.asarray(cloud.points)
    boundary_rows = np.flatnonzero(np.asarray(cloud.plan.boundary_mask))
    normals = np.asarray(cloud.plan.boundary_normals)[boundary_rows]
    free = PointBoundaryPlan(
        tuple(
            PointBoundaryCondition(
                "neumann",
                boundary_rows,
                0.0,
                label=f"free-{a}",
                component=a,
                normals=normals,
            )
            for a in range(2)
        ),
        row_count=host.shape[0],
        components=2,
    )
    floating_message: str | None = None
    try:
        MeshfreeElasticityPlan(cloud, free, _hooke_tensor())
    except ValueError as error:  # Documented: floating systems are refused, never gauged.
        if "floats" not in str(error):
            raise
        floating_message = str(error)
    clamp = PointBoundaryPlan(
        tuple(
            PointBoundaryCondition(
                "dirichlet", boundary_rows, label=f"clamp-{a}", component=a
            )
            for a in range(2)
        ),
        row_count=host.shape[0],
        components=2,
    )
    unstable_message: str | None = None
    try:
        MeshfreeGeneralizedStokesPlan(cloud, clamp, shear_modulus=_MU, stabilization=0.0)
    except (
        ValueError
    ) as error:  # Documented: equal-order collocation is not inf-sup stable.
        if "inf-sup" not in str(error):
            raise
        unstable_message = str(error)
    metrics: dict[str, Any] = {
        "points": count,
        "newton_refused_status": MechanicsStatus(int(refused.status)).name,
        "newton_solve_refused": int(refused.status) == int(MechanicsStatus.SOLVE_REFUSED),
        "rolled_back_load_factor": float(refused.accepted_load_factor),
        "rolled_back_to_zero_load": float(refused.accepted_load_factor) == 0.0,
        "committed_max_displacement": float(jnp.max(jnp.abs(refused.displacement))),
        "candidate_max_displacement": float(
            jnp.max(jnp.abs(refused.candidate_displacement))
        ),
        "floating_body_refused": floating_message is not None,
        "floating_body_message": floating_message,
        "unstabilized_mixed_refused": unstable_message is not None,
        "unstabilized_mixed_message": unstable_message,
    }
    return _mechanics_record(
        "mechanics-refusals",
        side,
        capacity,
        seed,
        recorder,
        metrics,
        (cloud, strict),
        reservation,
        "documented MechanicsStatus.SOLVE_REFUSED rollback; documented ValueError "
        "refusals for floating components and unstabilized equal-order mixed rows",
        "MeshfreeHyperelasticPlan; MeshfreeElasticityPlan; MeshfreeGeneralizedStokesPlan",
    )


# ---------------------------------------------------------------------------
# Q11 closed-surface Stokes on the unit sphere

_SPHERE_HARMONIC = np.asarray(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.5], [0.0, 0.5, 0.0]], dtype=np.float64
)
_SURFACE_VISCOSITY = 1.0
_SURFACE_REACTION = 1.0
_SURFACE_NEIGHBORS = 30
_SURFACE_DEGREE = 4


def _sphere_calculus(points: Array, recorder: PhaseRecorder, /) -> SurfaceTangentCalculus:
    def constraint(point: Array) -> Array:
        return jnp.asarray([jnp.dot(point, point) - 1.0])

    source = RegularLevelSetManifold(
        constraint, ambient_dimension=3, codimension=1, manifold_id="unit-sphere"
    )
    prepared = recorder.run(
        "local-fit",
        lambda: SurfacePointCloudPlan(
            points,
            ImplicitSurfaceGeometry(source, geometry_id="unit-sphere"),
            _SURFACE_NEIGHBORS,
            quadrature=SurfaceQuadraturePolicy("tangent-voronoi"),
            stencil_policy=LocalStencilPolicy(
                polynomial_degree=_SURFACE_DEGREE, chunk_rows=128
            ),
        ).prepare(),
        scope="surface-search-geometry-quadrature-gmls",
    )
    recorder.unavailable(
        "search", "Surface neighbor search is fused into SurfacePointCloudPlan.prepare"
    )
    recorder.unavailable(
        "geometry",
        "Implicit normals/projectors are fused into SurfacePointCloudPlan.prepare",
    )
    return recorder.run(
        "assembly", lambda: SurfaceTangentCalculus(prepared), scope="tangent-calculus"
    )


def _sphere_fields(points: Array, normals: Array, /) -> tuple[Array, Array]:
    """``grad_S Y`` and ``n x grad_S Y`` of the trace-free quadratic harmonic ``Y``."""
    harmonic = jnp.asarray(_SPHERE_HARMONIC)
    ambient = 2.0 * points @ harmonic
    gradient = ambient - jnp.sum(ambient * normals, axis=-1, keepdims=True) * normals
    return gradient, jnp.cross(normals, gradient)


def _sphere_probes(count: int, seed: int, /) -> np.ndarray:
    probes = np.random.default_rng(seed + 7919).standard_normal((count, 3))
    return probes / np.linalg.norm(probes, axis=1, keepdims=True)


def _surface_reservation(capacity: int, config: MeshfreeConfig, /) -> int:
    """Declared local-fit reservation of the intrinsic 2-D degree-4 surface fit.

    ``MeshfreeConfig.check_capacity`` sizes an ambient-dimension basis; the
    surface fit is intrinsic (sheet dimension 2), so it is declared here with
    the same planning formula and the same ``max_points`` refusal.
    """
    if capacity < _SURFACE_NEIGHBORS:
        raise ValueError("Surface capacity must cover the declared neighborhood.")
    if capacity > config.max_points:
        raise DeclaredCapacityRefusal(
            f"Requested {capacity} points exceeds the declared max_points={config.max_points}."
        )
    basis = math.comb(2 + _SURFACE_DEGREE, _SURFACE_DEGREE)
    chunk = 128
    return declare_reservation(
        8
        * (
            capacity * _SURFACE_NEIGHBORS * (3 + basis + 12)
            + chunk * (_SURFACE_NEIGHBORS + basis) ** 2 * 16
            + capacity * chunk * (3 + 8)
        ),
        config,
        scope="surface local-fit planning",
    )


def measure_surface_stokes_sphere(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Strain-form surface Stokes/Brinkman with a closed-form solenoidal oracle."""
    from examples.meshfree_surface_laplace_beltrami import sphere_points

    _require_float64(config)
    reservation = _surface_reservation(capacity, config)
    # GMRES(400) Krylov basis over velocity (3N) and pressure/multiplier (2N) unknowns.
    reservation += declare_reservation(
        8 * 401 * 5 * capacity, config, scope="surface Stokes GMRES(400) basis"
    )
    recorder = PhaseRecorder()
    calculus = _sphere_calculus(sphere_points(capacity, seed), recorder)
    plan = recorder.run(
        "assembly",
        lambda: MeshfreeSurfaceStokesPlan(
            calculus, viscosity=_SURFACE_VISCOSITY, reaction=_SURFACE_REACTION
        ),
        scope="strain-form-saddle-and-prepared-gmres",
    )
    surface = calculus.surface
    points, normals, measures = surface.points, surface.normals, surface.measures
    gradient, rotated = _sphere_fields(points, normals)
    pressure = points[:, 2]
    pressure_gradient = (
        jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64) - pressure[:, None] * normals
    )
    # -nu (Delta_B U + K U) + r U + grad p with Delta_B U = -5 U, K = 1.
    forcing = (4.0 * _SURFACE_VISCOSITY + _SURFACE_REACTION) * rotated + pressure_gradient
    result = recorder.run_repeated(
        "solve", lambda: plan.solve(forcing), repeats=config.repeats, scope="gmres"
    )
    exact_pressure = pressure - jnp.sum(measures * pressure) / jnp.sum(measures)
    # int Y^2 = (8 pi / 15) tr(C^2) for trace-free C; int |U|^2 = 6 int Y^2;
    # 2 int E:E = 4 int |U|^2 since 2 Div E(U) = -4 U on the unit sphere.
    exact_dissipation = (
        4.0
        * _SURFACE_VISCOSITY
        * 6.0
        * (8.0 * math.pi / 15.0)
        * float(np.trace(_SPHERE_HARMONIC @ _SPHERE_HARMONIC))
    )
    killing = jnp.cross(
        jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64), points.shape),
        points,
    )
    identity = 2.0 * calculus.tensor_divergence.mv(plan.strain(rotated))
    fill = fill_distance(
        np.asarray(points),
        _sphere_probes(4 * capacity, seed),
        provenance="max chordal nearest-sample distance over uniform random sphere probes",
    )
    metrics: dict[str, Any] = {
        "nodes": capacity,
        "fill_distance": fill["fill_distance"],
        "successful": bool(result.successful),
        "linear_status": int(result.status),
        "linear_iterations": int(result.diagnostics.iterations),
        "relative_residual": float(result.diagnostics.relative_residual),
        "velocity_relative_error": _relative(result.velocity, rotated, measures),
        "pressure_relative_error": _relative(result.pressure, exact_pressure, measures),
        "normal_residual": float(result.normal_residual),
        "divergence_residual": float(result.divergence_residual),
        "gauge_residual": float(result.gauge_residual),
        "dissipation_relative_error": abs(
            float(result.dissipation) / exact_dissipation - 1.0
        ),
        "killing_max_strain": float(jnp.max(jnp.abs(plan.strain(killing)))),
        "killing_dissipation": float(plan.dissipation(killing)),
        "curvature_identity_relative_error": _relative(
            identity, -4.0 * rotated, measures
        ),
        "gradient_mode_identity_relative_error": _relative(
            2.0 * calculus.tensor_divergence.mv(plan.strain(gradient)),
            -10.0 * gradient,
            measures,
        ),
    }
    return {
        "workload": "surface-stokes-sphere",
        "capacity": capacity,
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": 3,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": metrics,
        "retained_bytes": logical_array_bytes((calculus, plan)),
        "reserved_bytes": reservation,
        "oracle_provenance": "vector spherical harmonic U = n x grad_S Y (Y = x^T C x "
        "trace-free), p = z; closed-form dissipation; rigid rotation e_z x X",
        "consumer": "phydrax.solver.MeshfreeSurfaceStokesPlan",
    }


def measure_surface_stokes_refusal(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """A starved Krylov budget publishes MAXIMUM_STEPS_REACHED with finite fields."""
    from examples.meshfree_surface_laplace_beltrami import sphere_points

    _require_float64(config)
    reservation = _surface_reservation(capacity, config)
    recorder = PhaseRecorder()
    calculus = _sphere_calculus(sphere_points(capacity, seed), recorder)
    starved = recorder.run(
        "assembly",
        lambda: MeshfreeSurfaceStokesPlan(
            calculus,
            viscosity=_SURFACE_VISCOSITY,
            reaction=_SURFACE_REACTION,
            linear_policy=la.LinearSolvePolicy(
                la.GMRES(restart=5),
                tolerance=la.TolerancePolicy(
                    relative=1e-12, absolute=1e-14, max_steps=10
                ),
            ),
        ),
        scope="ten-step-gmres-budget",
    )
    surface = calculus.surface
    _, rotated = _sphere_fields(surface.points, surface.normals)
    result = recorder.run(
        "solve",
        lambda: starved.solve((4.0 * _SURFACE_VISCOSITY + _SURFACE_REACTION) * rotated),
        scope="starved-solve",
    )
    metrics: dict[str, Any] = {
        "nodes": capacity,
        "linear_status": la.LinearSolveStatus(int(result.status)).name,
        "maximum_steps_reported": int(result.status)
        == int(la.LinearSolveStatus.MAXIMUM_STEPS_REACHED),
        "refused_not_successful": not bool(result.successful),
        "refused_fields_finite": bool(result.finite),
    }
    return {
        "workload": "surface-stokes-refusal",
        "capacity": capacity,
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": 3,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": metrics,
        "retained_bytes": logical_array_bytes((calculus, starved)),
        "reserved_bytes": reservation,
        "oracle_provenance": "documented LinearSolveStatus.MAXIMUM_STEPS_REACHED",
        "consumer": "phydrax.solver.MeshfreeSurfaceStokesPlan(linear_policy=...)",
    }


# ---------------------------------------------------------------------------
# Q11 monolithic quasi-static fluid–structure step


def _fsi_resolution(capacity: int, /) -> tuple[int, int, int]:
    """Largest fluid resolution (multiple of 4, solid 3/4 of it) within ``capacity`` points."""
    # The regions' 30-point PHS3 stencils need at least 7 x 7 solid samples.
    fluid = 8
    if (fluid + 1) ** 2 + (3 * fluid // 4 + 1) ** 2 > capacity:
        raise ValueError("capacity is below the smallest fluid-structure pair (8, 6).")
    while (fluid + 5) ** 2 + (3 * (fluid + 4) // 4 + 1) ** 2 <= capacity:
        fluid += 4
    solid = 3 * fluid // 4
    return fluid, solid, (fluid + 1) ** 2 + (solid + 1) ** 2


def measure_fluid_structure_step(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Force, delivered/received power and elastic power of one monolithic FSI step."""
    import examples.meshfree_fluid_structure as consumer
    from phydrax.solver.coupling import solve_coupled_problem

    _require_float64(config)
    fluid, solid, points = _fsi_resolution(capacity)
    # Cloud and traction-ghost values for both components, plus interface mortars.
    unknowns = 2 * (points + fluid - 1 + solid - 1) + 2 * (fluid + 1)
    budget = min(config.working_set_bytes, config.resource_bytes)
    reservation = declare_reservation(
        8 * unknowns * unknowns, config, scope="FSI dense LU"
    )
    recorder = PhaseRecorder()
    prepared = recorder.run(
        "assembly",
        lambda: consumer.prepare_step(fluid, solid),
        scope="clouds-traces-mortar-coupled-problem",
    )
    recorder.unavailable(
        "local-fit", "Region clouds and reconstructions are fused into prepare_step"
    )
    policy = la.LinearSolvePolicy(
        la.DenseLU(),
        materialization=la.MaterializationPolicy(
            max_entries=budget // 8, max_bytes=budget
        ),
    )
    solution = recorder.run(
        "solve",
        lambda: solve_coupled_problem(prepared, policy=policy),
        scope="monolithic-dense-lu",
    )
    interface = solution.interface("wetted-interface")
    accepted = bool(solution.accepted)
    values = np.abs(np.asarray(interface.values, dtype=np.float64))
    scales = np.maximum(
        np.asarray(interface.scales, dtype=np.float64), np.finfo(np.float64).tiny
    )
    gated = np.asarray(interface.gated, dtype=bool)
    metrics: dict[str, Any] = {
        "fluid_resolution": fluid,
        "solid_resolution": solid,
        "points": points,
        "accepted": accepted,
        "native_successful": bool(solution.native_successful),
        "original_equation_residuals": {
            certificate.component: {
                "residual_norm": float(certificate.residual_norm),
                "scale": float(certificate.scale),
                "accepted": bool(certificate.accepted),
            }
            for certificate in solution.components
        },
        "max_gated_interface_defect": float(np.max(values[gated] / scales[gated]))
        if bool(np.any(gated))
        else math.nan,
    }
    if accepted:
        result = recorder.run(
            "output",
            lambda: consumer.resultants(prepared, solution),
            scope="vector-interface-resultants",
        )
        elastic = recorder.run(
            "output",
            lambda: consumer.elastic_power(prepared, solution),
            scope="solid-elastic-power",
        )
        fluid_power, solid_power = float(result.minus_power), float(result.plus_power)
        metrics.update(
            {
                "force_x": float(result.force[0]),
                "force_y": float(result.force[1]),
                "fluid_power": fluid_power,
                "solid_power": solid_power,
                "elastic_power": elastic,
                "power_balance_defect": abs(fluid_power + solid_power)
                / max(abs(solid_power), np.finfo(np.float64).tiny),
                "elastic_work_gap": abs(elastic - solid_power)
                / max(abs(solid_power), np.finfo(np.float64).tiny),
            }
        )
    return {
        "workload": "fluid-structure-step",
        "capacity": points,
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": 2,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": metrics,
        "retained_bytes": logical_array_bytes(prepared),
        "reserved_bytes": reservation,
        "oracle_provenance": "work consistency: delivered + received interface power = 0 "
        "under the mortar kinematic constraint; received power = solid elastic power",
        "consumer": "examples.meshfree_fluid_structure.prepare_step + "
        "phydrax.solver.coupling.solve_coupled_problem",
    }


def _isotropic_stress(
    lam: float, mu: float, field: Callable[[Array], Array], /
) -> Callable[[Array], Array]:
    def sigma(point: Array) -> Array:
        gradient = jax.jacfwd(field)(point)
        strain = 0.5 * (gradient + gradient.T)
        return 2.0 * mu * strain + lam * jnp.trace(strain) * jnp.eye(2, dtype=jnp.float64)

    return sigma


def _divergence_force(
    lam: float, mu: float, field: Callable[[Array], Array], /
) -> Callable[[Array], Array]:
    sigma = _isotropic_stress(lam, mu, field)

    def force(point: Array) -> Array:
        return -jnp.trace(jax.jacfwd(sigma)(point), axis1=1, axis2=2)

    return force


def _manufactured_fluid_velocity(point: Array) -> Array:
    """Smooth velocity vanishing with its traction derivatives on the walls y = 0, 1."""
    x, y = point[0], point[1]
    return (
        jnp.sin(jnp.pi * y) ** 2 * (2.0 - x) * jnp.stack((1.0 + 0.5 * x, 0.5 - 0.25 * x))
    )


def measure_fluid_structure_manufactured(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Smooth manufactured FSI step: solid-rate error and the solid energy identity.

    The fluid velocity is ``U`` and the solid rate ``w = U + (x-1)(2-x) Z(y)``
    with ``Z`` balancing the tractions at ``x = 1``; both vanish on the corner
    walls, so ``int sigma : eps(w) = int f . w + int_Gamma (sigma n_s) . w`` holds
    at the scheme order (no corner singularity, unlike the physical channel).
    """
    import examples.meshfree_fluid_structure as consumer
    from phydrax.discretization.meshfree import isotropic_elasticity_coefficients
    from phydrax.solver.coupling import (
        MeshfreeTraceComponent,
        PreparedCoupledProblem,
        solve_coupled_problem,
    )

    _require_float64(config)
    resolution = math.isqrt(capacity // 2) - 1
    if resolution < 8:
        raise ValueError("capacity is below two 9 x 9 manufactured regions.")
    points = 2 * (resolution + 1) ** 2
    unknowns = 2 * (points + 2 * (resolution - 1)) + 2 * (resolution + 1)
    budget = min(config.working_set_bytes, config.resource_bytes)
    reservation = declare_reservation(
        8 * unknowns * unknowns, config, scope="FSI dense LU"
    )
    fluid_law = (consumer.BULK_VISCOSITY, consumer.VISCOSITY)
    solid_law = (
        consumer.STEP * consumer.LAME_LAMBDA,
        consumer.STEP * consumer.SHEAR_MODULUS,
    )

    def solid_rate(point: Array) -> Array:
        on = jnp.stack((jnp.asarray(1.0, dtype=jnp.float64), point[1]))
        normal = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
        mismatch = (
            _isotropic_stress(*fluid_law, _manufactured_fluid_velocity)(on) @ normal
            - _isotropic_stress(*solid_law, _manufactured_fluid_velocity)(on) @ normal
        )
        lam, mu = solid_law
        correction = jnp.stack((mismatch[0] / (lam + 2.0 * mu), mismatch[1] / mu))
        return (
            _manufactured_fluid_velocity(point)
            + (point[0] - 1.0) * (2.0 - point[0]) * correction
        )

    recorder = PhaseRecorder()

    def component(
        name: str,
        x0: float,
        side: str,
        law: tuple[float, float],
        field: Callable[[Array], Array],
        /,
    ) -> MeshfreeTraceComponent:
        region_points = jnp.asarray(consumer.region(x0, resolution, side)[0].points)
        return consumer.block_component(
            name,
            x0,
            resolution,
            side,
            isotropic_elasticity_coefficients(*law, 2),
            np.asarray(jax.vmap(field)(region_points)),
            source=np.asarray(jax.vmap(_divergence_force(*law, field))(region_points)),
        )

    fluid = recorder.run(
        "local-fit",
        lambda: component("fluid", 0.0, "left", fluid_law, _manufactured_fluid_velocity),
        scope="fluid-cloud-block-system-trace",
    )
    solid = recorder.run(
        "local-fit",
        lambda: component("solid", 1.0, "right", solid_law, solid_rate),
        scope="solid-cloud-block-system-trace",
    )
    prepared: PreparedCoupledProblem = recorder.run(
        "assembly",
        lambda: consumer.coupled_step(fluid, solid),
        scope="mortar-coupled-problem",
    )
    policy = la.LinearSolvePolicy(
        la.DenseLU(),
        materialization=la.MaterializationPolicy(
            max_entries=budget // 8, max_bytes=budget
        ),
    )
    solution = recorder.run(
        "solve",
        lambda: solve_coupled_problem(prepared, policy=policy),
        scope="monolithic-dense-lu",
    )
    accepted = bool(solution.accepted)
    solid_cloud, fluid_cloud = solid.owner, fluid.owner
    metrics: dict[str, Any] = {
        "resolution": resolution,
        "points": points,
        "accepted": accepted,
        "native_successful": bool(solution.native_successful),
        "original_equation_residuals": {
            certificate.component: {
                "residual_norm": float(certificate.residual_norm),
                "scale": float(certificate.scale),
                "accepted": bool(certificate.accepted),
            }
            for certificate in solution.components
        },
        "fill_distance": _unit_square_fill(np.asarray(fluid_cloud.points), seed)[
            "fill_distance"
        ],
    }
    if accepted:
        rate = jnp.stack(
            [solution.field("solid", field) for field in consumer.FIELDS], axis=1
        )
        exact = jax.vmap(solid_rate)(solid_cloud.points)
        supplied = float(
            jnp.sum(
                solid_cloud.quadrature_weights
                * jnp.sum(
                    jax.vmap(_divergence_force(*solid_law, solid_rate))(
                        solid_cloud.points
                    )
                    * rate,
                    axis=1,
                )
            )
        )
        stored = recorder.run(
            "output",
            lambda: consumer.elastic_power(prepared, solution),
            scope="solid-elastic-power",
        )
        result = recorder.run(
            "output",
            lambda: consumer.resultants(prepared, solution),
            scope="vector-interface-resultants",
        )
        received = float(result.plus_power)
        metrics.update(
            {
                "solid_rate_relative_error": float(jnp.max(jnp.abs(rate - exact)))
                / float(jnp.max(jnp.abs(exact))),
                "energy_balance_gap": abs(stored - received - supplied) / abs(stored),
                "elastic_power": stored,
                "received_power": received,
                "body_force_power": supplied,
                "power_balance_defect": abs(float(result.minus_power) + received)
                / max(abs(received), np.finfo(np.float64).tiny),
            }
        )
    return {
        "workload": "fluid-structure-manufactured",
        "capacity": points,
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": 2,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": metrics,
        "retained_bytes": logical_array_bytes(prepared),
        "reserved_bytes": reservation,
        "oracle_provenance": "manufactured fluid velocity and traction-balancing solid "
        "rate vanishing on the corner walls; body forces -div(sigma) by autodiff",
        "consumer": "examples.meshfree_fluid_structure.block_component/coupled_step + "
        "phydrax.solver.coupling.solve_coupled_problem",
    }


FLOW_MECHANICS_WORKLOADS: dict[
    str, Callable[[int, int, MeshfreeConfig], dict[str, Any]]
] = {
    "incompressible-taylor-green": measure_incompressible_taylor_green,
    "incompressible-refusals": measure_incompressible_refusals,
    "exterior-coercivity-refusal": measure_exterior_coercivity_refusal,
    "incompressible-stokes-bounded": measure_incompressible_stokes_bounded,
    "lagrangian-quadrature-volume": measure_lagrangian_quadrature_volume,
    "lagrangian-material-mass-sph": measure_lagrangian_material_mass_sph,
    "lagrangian-measure-transfer": measure_lagrangian_measure_transfer,
    "lagrangian-refusals": measure_lagrangian_refusals,
    "elasticity-manufactured": measure_elasticity_manufactured,
    "elasticity-closed-form": measure_elasticity_closed_form,
    "elasticity-mixed-locking": measure_elasticity_mixed_locking,
    "mechanics-refusals": measure_mechanics_refusals,
    "surface-stokes-sphere": measure_surface_stokes_sphere,
    "surface-stokes-refusal": measure_surface_stokes_refusal,
    "fluid-structure-step": measure_fluid_structure_step,
    "fluid-structure-manufactured": measure_fluid_structure_manufactured,
}
