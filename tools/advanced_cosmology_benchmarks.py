"""Compile and steady-state benchmarks for advanced cosmology contracts."""

from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp

import phydrax as phx


def _block(tree) -> None:
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _measure(function, *args):
    start = time.perf_counter()
    value = function(*args)
    _block(value)
    return value, time.perf_counter() - start


def _baryon_case(cosmo):
    count = 8
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(count, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.EulerSystem(1)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "advanced-baryon-benchmark",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(grid.axis_names),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.HLLCFluxPlan(),
        ),
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.3, maximum_retries=0),
    )
    gravity = phx.solver.NewtonianSelfGravityPlan(0.01).prepare(
        phx.solver.prepare_balance_law_transport(runtime)
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count),
        jnp.full((count,), 1.0 / count),
        ambient_dimension=1,
    ).prepare()
    transfer = phx.discretization.ParticleGridSplatPlan(grid).prepare(particles)
    particle_gravity = phx.solver.ParticleMeshGravityPlan(gravity, transfer)
    scale = cosmo.CosmologyScaleContract(
        cosmo.CODE_COSMOLOGY_SCALE.length_unit,
        cosmo.CODE_COSMOLOGY_SCALE.mass_unit,
        cosmo.CODE_COSMOLOGY_SCALE.time_unit,
    )
    kdk = cosmo.CosmologicalKDKPlan(particles, (1.0,), scale=scale)
    gas = cosmo.ComovingEulerPlan(dynamics, substeps=8)
    plan = cosmo.CosmologicalGasParticleGravityPlan(
        gas, kdk, particle_gravity, [0.5, 0.51]
    )
    background = cosmo.FLRWBackground(1.0, 1.0, scale=scale)
    gas_average = jnp.zeros((count, 3)).at[:, 0].set(1.0).at[:, 2].set(1.0)
    positions = ((jnp.arange(count) + 0.5) / count)[:, None]
    state = cosmo.CosmologicalGasParticleState(
        gas.initialize(gas_average, 0.5),
        kdk.initialize(positions, jnp.zeros_like(positions), 0.5),
    )
    return background, plan, state


def main() -> None:
    cosmo = phx.applications.cosmology
    background = cosmo.FLRWBackground(1.0, 0.3, dark_energy_w0=-0.9, dark_energy_wa=0.1)
    distance = cosmo.FLRWDistancePlan(light_speed=1.0, order=64)
    redshifts = jnp.linspace(0.01, 3.0, 4096)
    distance_function = jax.jit(lambda z: distance.evaluate(background, z))
    _, distance_compile = _measure(distance_function, redshifts)
    _, distance_steady = _measure(distance_function, redshifts)

    provenance = cosmo.CosmologyProductProvenance(
        producer="advanced-benchmark",
        producer_version="native",
        model_form_id=background.model_form_id,
        request_id="advanced-benchmark-power",
        numerical_policy_id="advanced-benchmark-grid",
        physics_policy_id="linear-total-matter",
        scale_id=background.scale.scale_id,
        source_kind="native",
        differentiation="native-parameter",
    )
    scales = jnp.linspace(0.2, 1.0, 64)
    k = jnp.geomspace(0.05, 50000.0, 512)
    values = scales[:, None] ** 2 / (1.0 + k[None, :] ** 2)
    power = cosmo.MatterPowerTable(
        scales,
        k,
        values,
        cosmo.MatterPowerDescriptor("total_matter", "total_matter"),
        background.scale,
        provenance,
        background.realization,
    )
    card = cosmo.CorrectionModelCard(
        name="benchmark-boost",
        model_version="native",
        source_reference="benchmark",
        calibration_id="none",
        denominator_stage="linear",
        output_stage="nonlinear",
        scale_factor_domain=(0.2, 1.0),
        wavenumber_domain=(0.05, 50000.0),
        expected_error="not calibrated",
        license_id="internal",
    )
    correction = cosmo.MultiplicativeMatterPowerCorrectionPlan(
        scales,
        k,
        1.0 + 0.1 * jnp.broadcast_to(k[None, :] / (1.0 + k[None, :]), values.shape),
        card,
        differentiation="native-parameter",
    )
    correction_function = jax.jit(
        lambda strength: correction.apply(power, strength=strength)
    )
    corrected, correction_compile = _measure(correction_function, jnp.asarray(1.0))
    _, correction_steady = _measure(correction_function, jnp.asarray(1.0))

    variance_plan = cosmo.LinearVariancePlan(1.0)
    masses = jnp.geomspace(1.0e-4, 10.0, 128)
    variance_function = jax.jit(
        lambda mass: variance_plan.sigma(background, power, mass, 1.0)
    )
    _, variance_compile = _measure(variance_function, masses)
    _, variance_steady = _measure(variance_function, masses)

    radial = cosmo.RadialGrid(jnp.linspace(0.05, 2.0, 128))
    distribution = cosmo.RedshiftDistribution(
        radial,
        jnp.exp(-(((radial.redshifts - 0.8) / 0.25) ** 2)),
        "benchmark",
    )
    tracer = cosmo.LinearDensityTracer(distribution, 1.5)
    angular_plan = cosmo.LimberAngularPowerPlan(jnp.arange(10, 1010, 10), 1)
    angular_function = jax.jit(
        lambda table: angular_plan.predict(background, distance, table, (tracer,))
    )
    _, angular_compile = _measure(angular_function, corrected.power)
    angular_result, angular_steady = _measure(angular_function, corrected.power)

    force_plan = cosmo.PeriodicImageForcePlan(
        (1.0, 1.0, 1.0), 1.0, softening=0.01, image_shells=1
    )
    coordinate = (jnp.arange(32, dtype=float) + 0.5) / 32.0
    positions = jnp.stack(
        (coordinate, jnp.mod(3.0 * coordinate, 1.0), jnp.mod(7.0 * coordinate, 1.0)),
        axis=-1,
    )
    force_function = jax.jit(
        lambda value: force_plan.acceleration(value, jnp.ones((32,)))
    )
    _, force_compile = _measure(force_function, positions)
    force_result, force_steady = _measure(force_function, positions)

    baryon_background, baryon_plan, baryon_state = _baryon_case(cosmo)
    baryon_function = jax.jit(lambda state: baryon_plan.rollout(baryon_background, state))
    baryon_result, baryon_compile = _measure(baryon_function, baryon_state)
    _, baryon_steady = _measure(baryon_function, baryon_state)

    ell = jnp.arange(2, 3002)
    cmb_values = jnp.zeros((1, ell.size, 4, 4))
    diagonal = 1.0 / (ell.astype(float) * (ell + 1.0))
    for field in range(4):
        cmb_values = cmb_values.at[0, :, field, field].set(diagonal)
    cmb_table = cosmo.CmbSpectrumTable(
        ell,
        cmb_values,
        ("scalar",),
        provenance,
        background.realization,
    )
    cmb_plan = cosmo.CmbSpectrumTransformPlan(
        (0,), ((0, 0), (0, 1), (1, 1), (2, 2), (3, 3)), use_d_ell=True
    )
    cmb_function = jax.jit(cmb_plan.pack)
    _, cmb_compile = _measure(cmb_function, cmb_table)
    cmb_result, cmb_steady = _measure(cmb_function, cmb_table)

    report = {
        "distance_queries": int(redshifts.size),
        "distance_compile_seconds": distance_compile,
        "distance_steady_seconds": distance_steady,
        "correction_shape": list(values.shape),
        "correction_compile_seconds": correction_compile,
        "correction_steady_seconds": correction_steady,
        "variance_masses": int(masses.size),
        "variance_compile_seconds": variance_compile,
        "variance_steady_seconds": variance_steady,
        "limber_multipoles": int(angular_plan.multipoles.size),
        "limber_compile_seconds": angular_compile,
        "limber_steady_seconds": angular_steady,
        "limber_successful": bool(angular_result.successful),
        "force_particles": int(positions.shape[0]),
        "force_compile_seconds": force_compile,
        "force_steady_seconds": force_steady,
        "force_finite": bool(jnp.all(jnp.isfinite(force_result))),
        "baryon_cells": int(baryon_state.gas.cell_average.shape[0]),
        "baryon_compile_seconds": baryon_compile,
        "baryon_steady_seconds": baryon_steady,
        "baryon_successful": bool(baryon_result.successful),
        "cmb_multipoles": int(ell.size),
        "cmb_compile_seconds": cmb_compile,
        "cmb_steady_seconds": cmb_steady,
        "cmb_output_shape": list(cmb_result.shape),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
