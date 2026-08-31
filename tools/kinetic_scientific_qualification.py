"""Physics, differentiation, restart, AMR, and execution qualification for kinetics."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import opt_einsum as oe

import phydrax as phx
from benchmarks._runtime import capture_environment
from phydrax.discretization.finite_difference._boundary import HaloPlan
from phydrax.discretization.finite_difference._distributed import (
    DistributedHaloSchedule,
)
from phydrax.discretization.finite_difference._stencil import StencilFootprint
from phydrax.discretization.lattice_boltzmann._collision import quadratic_equilibrium
from phydrax.discretization.lattice_boltzmann._forcing import guo_raw_source
from phydrax.discretization.lattice_boltzmann._interfacial import (
    continuum_surface_force,
    static_contact_angle_normal,
)
from phydrax.equations._materials import IdealGasMaterial
from phydrax.equations._transport_closures import ConstantTransport
from tools.lattice_boltzmann_qualification import _d3q19_case, _shear_decay_case


def _grid(shape, *, periodic=True):
    dimension = len(shape)
    return phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic)
            for count in shape
        ),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(
        jnp.asarray(
            (
                (0.0,) * dimension,
                (1.0,) * dimension,
            )
        )
    )


def _collision_forcing_case():
    baseline = _shear_decay_case()
    d3q19 = _d3q19_case()
    lattice = phx.discretization.D2Q9()
    precision = phx.discretization.LatticeBoltzmannPrecisionPolicy()
    velocity = jnp.broadcast_to(jnp.asarray((0.02, -0.01)), (8, 8, 2))
    force = jnp.broadcast_to(jnp.asarray((1.0e-5, -2.0e-5)), velocity.shape)
    source = guo_raw_source(velocity, force, lattice, precision)
    source_mass = float(jnp.max(jnp.abs(jnp.sum(source, axis=-1))))
    source_momentum = jnp.max(
        jnp.abs(oe.contract("...q,qd->...d", source, lattice.velocities) - force)
    )
    return {
        "shear_decay": baseline,
        "d3q19_shear": d3q19,
        "guo_mass_residual": source_mass,
        "guo_momentum_residual": float(source_momentum),
        "passed": bool(
            baseline["passed"]
            and d3q19["passed"]
            and source_mass <= 2.0e-13
            and source_momentum <= 2.0e-13
        ),
    }


def _boundary_geometry_case():
    discretization = phx.discretization.LatticeBoltzmannPlan(
        _grid((32, 32)), phx.discretization.D2Q9()
    ).prepare()
    prepared = phx.discretization.prepare_lattice_boltzmann_link_geometry(
        discretization,
        phx.geometry.Circle((0.5, 0.5), 0.2).compile(),
        body_name="circle",
    )
    coordinates = jnp.asarray(((0.0, 0.0), (0.0, 0.5), (0.0, 1.0)))
    profile = phx.equations.WomersleyVelocityProfilePlan(2, 0)
    parameters = phx.equations.WomersleyVelocityParameters(
        jnp.asarray((0.0, 0.5)),
        0.5,
        2.0 * jnp.pi,
        2.0,
        0.02,
    )
    initial = profile(0.0, coordinates, parameters)
    periodic = profile(1.0, coordinates, parameters)
    wall_error = float(jnp.max(jnp.abs(initial[[0, 2], 0])))
    center_error = float(jnp.abs(initial[1, 0] - 0.02))
    periodic_error = float(jnp.max(jnp.abs(periodic - initial)))
    return {
        "blocked_link_count": prepared.evidence.blocked_link_count,
        "geometry_normal_residual": prepared.evidence.maximum_normal_residual,
        "geometry_margin": prepared.evidence.minimum_cell_distance_margin,
        "womersley_wall_error": wall_error,
        "womersley_centerline_error": center_error,
        "womersley_periodicity_error": periodic_error,
        "passed": bool(
            prepared.evidence.passed
            and wall_error <= 2.0e-12
            and center_error <= 2.0e-12
            and periodic_error <= 2.0e-12
        ),
    }


def _multiphase_case():
    lattice = phx.discretization.D2Q9()
    coordinate = jnp.arange(96, dtype=jnp.float64) - 48.0
    x, y = jnp.meshgrid(coordinate, coordinate, indexing="ij")
    radius = 24.0
    distance = jnp.sqrt(x * x + y * y)
    phase = jnp.tanh((radius - distance) / 3.0)
    surface = continuum_surface_force(phase, lattice, 0.04)
    interface = jnp.abs(distance - radius) < 1.0
    curvature = float(jnp.mean(jnp.abs(surface.curvature[interface])))
    curvature_error = abs(curvature - 1.0 / radius) / (1.0 / radius)
    closure = phx.equations.BinaryPhaseThermodynamicClosure()
    thermodynamics = phx.discretization.PreparedBinaryKineticThermodynamics(
        closure,
        lattice,
        phx.equations.ThermodynamicForceRepresentation.CHEMICAL_POTENTIAL_GRADIENT,
    ).evaluate(phase, phx.equations.BinaryThermodynamicParameters(0.08, 0.12))
    chemical_force_sum = float(
        jnp.max(jnp.abs(jnp.sum(thermodynamics.chemical_force_density, axis=(0, 1))))
    )
    stress_force_sum = float(
        jnp.max(jnp.abs(jnp.sum(thermodynamics.stress_force_density, axis=(0, 1))))
    )
    interface_normal = jnp.asarray([[[1.0, 0.0]]])
    wall_normal = jnp.asarray([[[0.0, 1.0]]])
    angle = jnp.asarray(jnp.pi / 3.0)
    imposed = static_contact_angle_normal(
        interface_normal,
        wall_normal,
        angle,
        jnp.asarray([[True]]),
    )
    contact_error = float(
        jnp.abs(jnp.sum(imposed * wall_normal, axis=-1)[0, 0] - jnp.cos(angle))
    )
    representation_errors = []
    smooth_parameters = phx.equations.BinaryThermodynamicParameters(0.08, 0.001)
    prepared_thermodynamics = phx.discretization.PreparedBinaryKineticThermodynamics(
        closure,
        lattice,
        phx.equations.ThermodynamicForceRepresentation.CHEMICAL_POTENTIAL_GRADIENT,
    )
    for resolution in (32, 64):
        smooth_coordinate = (jnp.arange(resolution, dtype=jnp.float64) + 0.5) / resolution
        smooth_x, smooth_y = jnp.meshgrid(
            smooth_coordinate,
            smooth_coordinate,
            indexing="ij",
        )
        smooth_phase = (
            0.4 * jnp.sin(2.0 * jnp.pi * smooth_x) * jnp.cos(2.0 * jnp.pi * smooth_y)
        )
        smooth_fields = prepared_thermodynamics.evaluate(
            smooth_phase,
            smooth_parameters,
            cell_size=1.0 / resolution,
        )
        representation_errors.append(
            float(jnp.max(jnp.abs(smooth_fields.force_representation_residual)))
        )
    representation_order = float(
        jnp.log2(representation_errors[0] / representation_errors[1])
    )
    return {
        "curvature_relative_error": curvature_error,
        "chemical_force_sum": chemical_force_sum,
        "stress_force_sum": stress_force_sum,
        "contact_angle_cosine_error": contact_error,
        "force_representation_residual_maximum": float(
            jnp.max(jnp.abs(thermodynamics.force_representation_residual))
        ),
        "force_representation_errors": representation_errors,
        "force_representation_order": representation_order,
        "passed": bool(
            curvature_error <= 0.2
            and chemical_force_sum <= 2.0e-12
            and stress_force_sum <= 2.0e-12
            and contact_error <= 2.0e-12
            and representation_order >= 1.7
        ),
    }


def _dvm_case():
    material = IdealGasMaterial(1.4, 1.0)
    transport = ConstantTransport(0.03, 0.04)
    records = {}
    passed = True
    for name, quadrature in (
        ("d2v17", phx.discretization.d2v17_quadrature()),
        ("d2v37", phx.discretization.d2v37_off_lattice_quadrature()),
    ):
        method = phx.equations.SmoothCompressibleD2VKineticMethod(
            quadrature, material, transport
        )
        conserved = jnp.asarray((1.0, 0.03, -0.02, 2.5))
        _, evidence = method.equilibrium_with_evidence(conserved)
        residual = float(jnp.max(jnp.abs(evidence.conserved_residual)))
        flux_residual = float(jnp.max(jnp.abs(evidence.total_energy_flux_residual)))
        realizable = bool(evidence.realizability.realizable)
        records[name] = {
            "quadrature_residual": quadrature.certification.maximum_residual,
            "conserved_residual": residual,
            "energy_flux_residual": flux_residual,
            "realizable": realizable,
            "transport_kind": quadrature.transport_kind,
        }
        passed = passed and realizable and residual <= 2.0e-6 and flux_residual <= 2.0e-6
    return {"methods": records, "passed": passed}


def _differentiation_case():
    lattice = phx.discretization.D2Q9()
    precision = phx.discretization.LatticeBoltzmannPrecisionPolicy()
    velocity = jnp.broadcast_to(jnp.asarray((0.02, -0.01)), (8, 8, 2))

    def objective(amplitude):
        force = jnp.broadcast_to(
            jnp.asarray((amplitude, -0.5 * amplitude)), velocity.shape
        )
        source = guo_raw_source(velocity, force, lattice, precision)
        return jnp.sum(source * source)

    amplitude = jnp.asarray(1.0e-4)
    tangent = jax.grad(objective)(amplitude)
    step = jnp.asarray(1.0e-6)
    finite_difference = (objective(amplitude + step) - objective(amplitude - step)) / (
        2.0 * step
    )
    force_gradient_error = float(jnp.abs(tangent - finite_difference))

    closure = phx.equations.BinaryPhaseThermodynamicClosure()
    phase = jnp.linspace(-0.8, 0.8, 64).reshape((8, 8))

    def energy(bulk):
        parameters = phx.equations.BinaryThermodynamicParameters(bulk, 0.1)
        gradient = jnp.zeros(phase.shape + (2,))
        laplacian = jnp.zeros_like(phase)
        return jnp.sum(
            closure.evaluate_local(
                phase, gradient, laplacian, parameters
            ).bulk_energy_density
        )

    bulk = jnp.asarray(0.08)
    bulk_gradient = jax.grad(energy)(bulk)
    bulk_step = jnp.asarray(1.0e-5)
    bulk_finite_difference = (energy(bulk + bulk_step) - energy(bulk - bulk_step)) / (
        2.0 * bulk_step
    )
    bulk_gradient_error = float(jnp.abs(bulk_gradient - bulk_finite_difference))
    return {
        "force_gradient_error": force_gradient_error,
        "bulk_gradient_error": bulk_gradient_error,
        "passed": bool(force_gradient_error <= 2.0e-12 and bulk_gradient_error <= 2.0e-9),
    }


def _execution_case():
    devices = tuple(jax.devices())
    if len(devices) >= 4:
        selected = devices[:4]
        partition_shape = (2, 2)
    elif len(devices) >= 2:
        selected = devices[:2]
        partition_shape = (2, 1)
    else:
        selected = devices[:1]
        partition_shape = (1, 1)
    shape = tuple(4 * value for value in partition_shape)
    grid = _grid(shape)
    discretization = phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D2Q9()
    ).prepare()
    compiled = phx.equations.compile_lattice_boltzmann_problem(
        phx.equations.LatticeBoltzmannProblem("distributed-qualification", 2),
        discretization,
        phx.discretization.LatticeBoltzmannMethodPlan(
            phx.discretization.BGKCollisionPlan()
        ),
        phx.discretization.LatticeBoltzmannBoundaryPlan(),
        time_step=0.01,
    )
    generic_halo = DistributedHaloSchedule(
        shape,
        partition_shape,
        HaloPlan(
            StencilFootprint(("x", "y"), (1, 1), (1, 1)),
            distributed_neighbors=True,
        ),
        periodic_axes=(True, True),
        devices=selected,
        mesh_axis_prefix="kinetic_scientific",
    )
    distributed = phx.discretization.PreparedDistributedLatticeBoltzmannDynamics(
        compiled.dynamics,
        phx.discretization.LatticeBoltzmannHaloSchedule(
            discretization.velocity_set,
            generic_halo,
        ),
    )
    parameters = phx.discretization.LatticeBoltzmannRuntimeParameters(0.01)
    coordinates = grid.points.reshape(shape + (2,))
    velocity = jnp.stack(
        (
            0.01 * jnp.sin(2.0 * jnp.pi * coordinates[..., 1]),
            jnp.zeros(shape),
        ),
        axis=-1,
    )
    initial = compiled.initialize_state(1.0, velocity, parameters)
    result = distributed.realize(
        initial,
        step_count=3,
        args=parameters,
        verify_equivalence=True,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    jax.block_until_ready(result.final_populations)
    return {
        "device_count": len(selected),
        "partition_shape": list(partition_shape),
        "population_axis_partitioned": distributed.execution.metadata.population_axis_partitioned,
        "maximum_absolute_error": float(result.equivalence.maximum_absolute_error),
        "populations_equivalent": bool(result.equivalence.populations_equivalent),
        "failures_equivalent": bool(result.equivalence.failures_equivalent),
        "diagnostics_equivalent": bool(result.equivalence.diagnostics_equivalent),
        "passed": bool(result.equivalence.equivalent),
    }


def _amr_case():
    lattice = phx.discretization.D2Q9()
    precision = phx.discretization.LatticeBoltzmannPrecisionPolicy()
    transfer = phx.discretization.LatticeBoltzmannAMRTransferPlan(lattice)
    prepared = phx.discretization.PreparedLatticeBoltzmannAMRTransfer(
        transfer,
        precision,
        phx.discretization.LatticeBoltzmannScaling(0.25, 0.01, 1.0),
        phx.discretization.LatticeBoltzmannScaling(0.125, 0.005, 1.0),
    )
    coordinate = jnp.arange(4, dtype=jnp.float64)
    x, y = jnp.meshgrid(coordinate, coordinate, indexing="ij")
    density = 1.0 + 1.0e-3 * jnp.sin(2.0 * jnp.pi * x / 4.0)
    velocity = jnp.stack(
        (
            0.01 * jnp.sin(2.0 * jnp.pi * y / 4.0),
            jnp.zeros_like(x),
        ),
        axis=-1,
    )
    coarse = quadratic_equilibrium(density, velocity, lattice, precision)
    fine, prolongation = prepared.prolong(
        coarse, jnp.asarray(1.0), jnp.asarray(2.0 / 3.0)
    )
    recovered, restriction = prepared.restrict(
        fine, jnp.asarray(1.0), jnp.asarray(2.0 / 3.0)
    )
    error = float(jnp.max(jnp.abs(recovered - coarse)))
    return {
        "roundtrip_error": error,
        "mass_defect": float(restriction.mass_defect),
        "momentum_defect": float(restriction.momentum_defect),
        "minimum_population": float(restriction.minimum_population),
        "nonequilibrium_scale": float(restriction.nonequilibrium_scale),
        "passed": bool(
            prolongation.successful and restriction.successful and error <= 3.0e-12
        ),
    }


def _checkpoint_case():
    grid = _grid((6, 6))
    discretization = phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D2Q9()
    ).prepare()
    compiled = phx.equations.compile_lattice_boltzmann_problem(
        phx.equations.LatticeBoltzmannProblem("checkpoint-qualification", 2),
        discretization,
        phx.discretization.LatticeBoltzmannMethodPlan(
            phx.discretization.BGKCollisionPlan()
        ),
        phx.discretization.LatticeBoltzmannBoundaryPlan(),
        time_step=0.01,
    )
    parameters = phx.discretization.LatticeBoltzmannRuntimeParameters(0.01)
    state = compiled.initialize_state(1.0, jnp.asarray((0.01, 0.0)), parameters)
    plan = phx.discretization.KineticCheckpointPlan(
        compiled.dynamics.prepared_id,
        compiled.dynamics.program_manifest,
        topology_id=compiled.boundary.boundary_id,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "kinetic.phxcheckpoint"
        written = phx.discretization.write_kinetic_checkpoint(
            path,
            plan,
            jnp.asarray(0.0),
            jnp.asarray(0, dtype=jnp.int32),
            state,
            args=parameters,
        )
        restored = phx.discretization.read_kinetic_checkpoint(
            path,
            plan,
            state,
            args_template=parameters,
        )
    error = float(jnp.max(jnp.abs(restored.state - state)))
    return {
        "state_error": error,
        "payload_identity_matches": written.payload_id == restored.payload_id,
        "passed": error == 0.0 and written.payload_id == restored.payload_id,
    }


def qualification():
    cases = {
        "collision_forcing": _collision_forcing_case(),
        "boundary_geometry": _boundary_geometry_case(),
        "multiphase": _multiphase_case(),
        "discrete_velocity": _dvm_case(),
        "differentiation": _differentiation_case(),
        "production_execution": _execution_case(),
        "amr_interface": _amr_case(),
        "checkpoint": _checkpoint_case(),
    }
    return {
        "environment": capture_environment().to_dict(),
        "evidence_levels": {
            "invariant_complete": all(
                bool(cases[name]["passed"])
                for name in (
                    "collision_forcing",
                    "discrete_velocity",
                    "amr_interface",
                    "checkpoint",
                )
            ),
            "physics_qualified": all(
                bool(cases[name]["passed"])
                for name in (
                    "collision_forcing",
                    "boundary_geometry",
                    "multiphase",
                    "discrete_velocity",
                    "amr_interface",
                )
            ),
            "differentiation_qualified": bool(cases["differentiation"]["passed"]),
            "execution_qualified": bool(cases["production_execution"]["passed"]),
            "deployment_qualified": False,
        },
        "cases": cases,
        "passed": all(bool(case["passed"]) for case in cases.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/kinetic_scientific_qualification.json"),
    )
    arguments = parser.parse_args()
    payload = qualification()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
