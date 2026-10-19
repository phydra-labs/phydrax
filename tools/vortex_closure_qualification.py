#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json

import jax
import jax.numpy as jnp

import phydrax as phx


position = jnp.asarray(
    ((-0.6, -0.2), (-0.2, 0.3), (0.3, -0.4), (0.7, 0.2)), dtype=jnp.float64
)
strength = jnp.asarray((0.5, -0.3, 0.8, -0.4), dtype=jnp.float64)
core = jnp.full((4,), 0.08, dtype=jnp.float64)
target_position = jnp.asarray(((0.9, 0.9), (-0.9, -0.9)), dtype=jnp.float64)
source = phx.discretization.VortexSourceState(position, strength, core_radius=core)
target = phx.discretization.VortexTargetState(target_position)
direct = (
    phx.operators.GaussianDirectVortexPlan2D(maximum_sources=4, maximum_targets=2)
    .prepare(source_capacity=4, target_capacity=2, target_topology="arbitrary-targets")
    .evaluate(source, target)
)
fmm = (
    phx.operators.VortexFMMPlan(
        position,
        jnp.asarray((-1.0, -1.0), dtype=jnp.float64),
        jnp.asarray((1.0, 1.0), dtype=jnp.float64),
        depth=2,
        expansion_order=1,
        leaf_capacity=4,
    )
    .prepare(source_capacity=4, target_capacity=2, target_topology="arbitrary-targets")
    .evaluate(source, target)
)
if fmm.velocity is None or direct.velocity is None:
    raise ValueError("vortex-closure.velocity must be available; got None")
fmm_relative_error = jnp.linalg.norm(fmm.velocity - direct.velocity) / jnp.maximum(
    jnp.linalg.norm(direct.velocity), 1.0e-12
)

ewald_position = jnp.asarray(((0.25, 0.5), (0.75, 0.5)), dtype=jnp.float64)
ewald_source = phx.discretization.VortexSourceState(
    ewald_position, jnp.asarray((1.0, -1.0), dtype=jnp.float64)
)
ewald_target = phx.discretization.VortexTargetState(
    ewald_position, source_indices=jnp.arange(2, dtype=jnp.int32)
)
ewald = (
    phx.operators.PeriodicVortexEwaldPlan(
        jnp.asarray((1.0, 1.0), dtype=jnp.float64),
        splitting_parameter=6.0,
        real_image_radius=2,
        reciprocal_mode_radius=5,
    )
    .prepare(source_capacity=2, target_capacity=2)
    .evaluate(ewald_source, ewald_target)
)

population = phx.discretization.VortexPopulationPlan(3, 2)
population_state, journal = population.initialize(
    jnp.zeros((3, 2), dtype=jnp.float64),
    jnp.zeros((3,), dtype=jnp.float64),
    jnp.ones((3,), dtype=jnp.float64),
    jnp.ones((3,), dtype=jnp.float64),
    active_mask=jnp.zeros((3,), dtype=jnp.bool_),
)
inserted = population.insert(
    population_state, journal, jnp.asarray((0.0, 0.0), dtype=jnp.float64), 1.0, 0.2, 0.5
)
split = population.split(
    inserted.accepted, inserted.journal, 0, jnp.asarray((0.1, 0.0), dtype=jnp.float64)
)
population_defect = jnp.abs(jnp.sum(split.accepted.strength) - 1.0)

topology = phx.discretization.VortexRingSheetTopology(
    4,
    jnp.asarray((0, 1, 2, 3), dtype=jnp.int32),
    jnp.asarray((1, 2, 3, 0), dtype=jnp.int32),
    jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
    jnp.asarray(((1, 1, 1, 1),), dtype=jnp.int32),
)
ring = phx.discretization.VortexRingSheetState(
    topology,
    jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=jnp.float64,
    ),
    jnp.asarray((1.0,), dtype=jnp.float64),
    jnp.full((4,), 0.05, dtype=jnp.float64),
)
wake = phx.solver.VortexWakeIntegratorPlan("rk3", core_diffusivity=0.01).step(
    ring,
    lambda targets, time, args: jnp.broadcast_to(
        jnp.asarray((0.1, 0.0, 0.0), dtype=jnp.float64), targets.positions.shape
    ),
    0.0,
    0.1,
)

random_direct = phx.operators.GaussianDirectVortexPlan2D(maximum_sources=2).prepare(
    source_capacity=2, target_capacity=2
)
random_source = phx.discretization.VortexSourceState(
    ewald_position,
    jnp.asarray((1.0, -1.0), dtype=jnp.float64),
    core_radius=jnp.full((2,), 0.1, dtype=jnp.float64),
    volume=jnp.ones((2,), dtype=jnp.float64),
)
# ty: ignore[call-non-callable]
random = phx.applications.vortex_flow.RandomVortexSolverPlan(
    random_direct, 0.01, 4, antithetic=True
).step(
    # ty: ignore[call-non-callable]
    phx.applications.vortex_flow.RandomVortexSolverPlan(
        random_direct, 0.01, 4, antithetic=True
    ).initialize(random_source),
    jax.random.key(4),
    0.01,
)

metrics = {
    "fmm_relative_error": float(fmm_relative_error),
    "fmm_tail_bound": float(fmm.diagnostics.backend_diagnostics.geometric_tail_bound),
    "ewald_compatibility_residual": float(
        ewald.diagnostics.backend_diagnostics.compatibility_residual
    ),
    "population_strength_defect": float(population_defect),
    "wake_circulation_defect": float(jnp.abs(wake.evidence.circulation_residual)),
    "random_antithetic_mean_defect": float(random.evidence.weak_moment_residual),
}
passed = bool(
    metrics["fmm_relative_error"] < 0.3
    and metrics["ewald_compatibility_residual"] < 1.0e-12
    and metrics["population_strength_defect"] < 1.0e-12
    and metrics["wake_circulation_defect"] < 1.0e-12
    and metrics["random_antithetic_mean_defect"] < 1.0e-12
    and fmm.successful
    and ewald.successful
    and split.successful
    and wake.successful
    and random.successful
)
print(json.dumps({"campaign": "vortex-closure", "passed": passed, **metrics}, indent=2))
if not passed:
    raise SystemExit(1)
