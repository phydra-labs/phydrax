"""Deterministic invariant qualification for advanced kinetic capabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe

import phydrax as phx
from benchmarks._runtime import capture_environment
from phydrax.discretization.finite_difference._boundary import HaloPlan
from phydrax.discretization.finite_difference._distributed import (
    DistributedHaloSchedule,
)
from phydrax.discretization.finite_difference._stencil import StencilFootprint
from phydrax.discretization.lattice_boltzmann._collision import quadratic_equilibrium
from phydrax.discretization.lattice_boltzmann._colour_gradient import (
    recolour_populations,
)
from phydrax.discretization.lattice_boltzmann._execution import (
    LatticeBoltzmannExecutionStep,
    ReferenceLatticeBoltzmannExecutionPlan,
)
from phydrax.discretization.lattice_boltzmann._moments import (
    populations_from_raw_moments,
    raw_moments,
)
from phydrax.discretization.lattice_boltzmann._species import (
    species_equilibrium,
    species_raw_moments,
)
from phydrax.discretization.lattice_boltzmann._thermal import (
    thermal_equilibrium,
    thermal_raw_moments,
)


def _maximum_absolute(value) -> float:
    return float(jnp.max(jnp.abs(jnp.asarray(value))))


def _lattice_case() -> dict[str, object]:
    lattice = phx.discretization.D3Q27()
    velocities = np.asarray(lattice.velocities)
    opposite_residual = np.max(
        np.abs(velocities[np.asarray(lattice.opposite)] + velocities)
    )
    passed = bool(
        opposite_residual == 0
        and lattice.capability_evidence.nearest_neighbor
        and lattice.capability_evidence.tensor_product
        and lattice.capability_evidence.hydrodynamic_isotropy_order >= 4
    )
    return {
        "lattice": lattice.name,
        "population_count": lattice.population_count,
        "opposite_residual": int(opposite_residual),
        "capability_evidence_id": lattice.capability_evidence.evidence_id,
        "passed": passed,
    }


def _moment_case() -> dict[str, object]:
    lattice = phx.discretization.D3Q27()
    precision = phx.discretization.LatticeBoltzmannPrecisionPolicy()
    basis = phx.discretization.MomentBasisPlan().prepare(lattice, precision)
    populations = jnp.asarray(lattice.weights) * jnp.linspace(
        0.95, 1.05, lattice.population_count
    )
    recovered = populations_from_raw_moments(
        raw_moments(populations, basis, precision), basis, precision
    )
    residual = _maximum_absolute(recovered - populations)
    tolerance = 2.0e-13
    return {
        "basis_id": basis.basis_id,
        "round_trip_residual": residual,
        "tolerance": tolerance,
        "passed": residual <= tolerance,
    }


def _collision_case() -> dict[str, object]:
    lattice = phx.discretization.D2Q9()
    precision = phx.discretization.LatticeBoltzmannPrecisionPolicy()
    density = jnp.asarray([[1.0, 1.05], [0.95, 1.02]])
    velocity = jnp.broadcast_to(jnp.asarray((0.01, -0.015)), density.shape + (2,))
    equilibrium = quadratic_equilibrium(density, velocity, lattice, precision)
    perturbation = jnp.asarray((0.0, 1.0, 1.0, -1.0, -1.0, 0.5, 0.5, -0.5, -0.5)) * 1.0e-6
    populations = equilibrium + perturbation
    basis = phx.discretization.MomentBasisPlan()
    spectrum = phx.discretization.RelaxationSpectrumPlan(default_rate=1.1)
    plans = {
        "mrt": phx.discretization.MRTCollisionPlan(basis, spectrum),
        "regularized": phx.discretization.RegularizedCollisionPlan(),
        "smagorinsky": phx.discretization.SmagorinskyCollisionPlan(),
        "central_moment": phx.discretization.CentralMomentCollisionPlan(basis, spectrum),
        "cumulant": phx.discretization.CumulantCollisionPlan(basis, spectrum),
        "kbc": phx.discretization.KBCCollisionPlan(basis),
        "entropic": phx.discretization.EntropicCollisionPlan(),
    }
    records = {}
    passed = True
    for name, collision in plans.items():
        method = phx.discretization.LatticeBoltzmannMethodPlan(collision).prepare(
            lattice, precision
        )
        result = method.collide(
            populations,
            density,
            velocity,
            jnp.zeros_like(velocity),
            jnp.asarray(1.2),
            lattice,
            precision,
        )
        record = {
            "mass_error": float(result.diagnostics.mass_error),
            "momentum_error": float(result.diagnostics.momentum_error),
            "minimum_population": float(result.diagnostics.minimum_population),
            "entropy_residual_maximum": float(
                jnp.max(result.diagnostics.entropy_residual)
            ),
            "root_residual": float(jnp.max(result.diagnostics.root_residual)),
            "successful": bool(result.successful),
        }
        records[name] = record
        passed = passed and bool(result.successful)
        passed = passed and record["mass_error"] <= 2.0e-12
        passed = passed and record["momentum_error"] <= 2.0e-12
    return {
        "families": records,
        "conservation_tolerance": 2.0e-12,
        "passed": passed,
    }


def _boundary_case() -> dict[str, object]:
    owner_enum = phx.discretization.LatticeBoltzmannLinkOwner
    shape = (3, 3, 9)
    owner = np.full(shape, int(owner_enum.LOCAL), dtype=np.int8)
    owner[0, 1, 1] = int(owner_enum.HALFWAY)
    parameter = np.full(shape, -1, dtype=np.int32)
    axis = np.full(shape, -1, dtype=np.int8)
    sign = np.zeros(shape, dtype=np.int8)
    body = np.full(shape, -1, dtype=np.int32)
    body[0, 1, 1] = 0
    fraction = np.zeros(shape)
    fraction[0, 1, 1] = 0.5
    topology = phx.discretization.CompiledLatticeBoltzmannLinkTopology(
        owner,
        parameter,
        axis,
        sign,
        body,
        fraction,
        np.ones(shape[:-1], dtype=bool),
        topology_id="kinetic-expansion-qualification",
    )
    streamed = topology.commit(
        topology.begin(jnp.zeros(shape)),
        jnp.ones(shape),
        phx.discretization.LatticeBoltzmannBoundaryStage.STREAM,
        (owner_enum.LOCAL, owner_enum.PERIODIC, owner_enum.HALO),
    )
    local_written = int(jnp.sum(streamed.written))
    expected = int(np.sum(owner == int(owner_enum.LOCAL)))
    wall_unwritten = not bool(streamed.written[0, 1, 1])
    return {
        "topology_id": topology.topology_id,
        "stream_written_count": local_written,
        "expected_stream_count": expected,
        "wall_link_unwritten_before_wall_stage": wall_unwritten,
        "passed": local_written == expected and wall_unwritten,
    }


def _transfer_case() -> dict[str, object]:
    lattice = phx.discretization.D2Q9()
    transfer = phx.discretization.LatticeBoltzmannAMRTransferPlan(lattice)
    fine = jnp.broadcast_to(jnp.asarray(lattice.weights), (8, 8, 9))
    coarse, restriction = transfer.restrict(fine)
    prolonged, prolongation = transfer.prolong(coarse)
    residual = _maximum_absolute(prolonged - fine)
    return {
        "coarse_shape": list(coarse.shape),
        "restriction_mass_defect": float(restriction.mass_defect),
        "restriction_momentum_defect": float(restriction.momentum_defect),
        "prolongation_mass_defect": float(prolongation.mass_defect),
        "prolongation_momentum_defect": float(prolongation.momentum_defect),
        "round_trip_residual": residual,
        "passed": bool(
            restriction.successful and prolongation.successful and residual <= 1e-14
        ),
    }


def _geometry_transfer_case() -> dict[str, object]:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D2Q9()
    ).prepare()
    source = phx.discretization.LatticeBoltzmannGeometryEpoch.from_mask(
        discretization,
        np.ones(grid.shape, dtype=bool),
        source_id="qualification-all-fluid",
    )
    target_mask = np.ones(grid.shape, dtype=bool)
    target_mask[3, 4] = False
    target = phx.discretization.LatticeBoltzmannGeometryEpoch.from_mask(
        discretization,
        target_mask,
        source_id="qualification-covered-cell",
        topology_epoch=1,
    )
    transfer = phx.discretization.LatticeBoltzmannPopulationTransferPlan(source, target)
    populations = jnp.broadcast_to(
        discretization.velocity_set.weights,
        discretization.population_shape,
    )
    result = transfer.transfer(populations)
    return {
        "mass_residual": float(result.evidence.mass_residual),
        "momentum_residual": _maximum_absolute(result.evidence.momentum_residual),
        "minimum_population": float(result.evidence.minimum_population),
        "covered_count": result.evidence.covered_count,
        "uncovered_count": result.evidence.uncovered_count,
        "passed": bool(result.evidence.passed),
    }


def _immersed_boundary_case() -> dict[str, object]:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D2Q9()
    ).prepare()
    plan = phx.discretization.ImmersedBoundaryForcingPlan(
        discretization, iteration_count=12
    )
    result = plan.apply(
        jnp.zeros((8, 8, 2)),
        jnp.ones((8, 8)),
        jnp.asarray(((0.5, 0.5),)),
        jnp.asarray(((0.01, 0.0),)),
        jnp.asarray((float(discretization.cell_size),)),
        jnp.asarray(0.01),
        body_indices=jnp.asarray((0,)),
        body_centers=jnp.asarray(((0.5, 0.5),)),
    )
    return {
        "maximum_velocity_residual": float(result.evidence.maximum_velocity_residual),
        "partition_of_unity_residual": float(result.evidence.partition_of_unity_residual),
        "force_balance_residual": float(result.ledger.force_balance_residual),
        "iteration_count": int(result.evidence.iteration_count),
        "converged": bool(result.evidence.converged),
        "passed": bool(result.evidence.successful and result.evidence.converged),
    }


def _multiphysics_case() -> dict[str, object]:
    lattice = phx.discretization.D2Q9()
    precision = phx.discretization.LatticeBoltzmannPrecisionPolicy()
    weights = jnp.asarray(lattice.weights)
    red = jnp.asarray([[0.7, 0.3], [0.4, 0.8]])
    blue = 1.0 - red
    total = jnp.broadcast_to(weights, red.shape + (lattice.population_count,))
    normal = jnp.broadcast_to(jnp.asarray((0.6, 0.8)), red.shape + (2,))
    recoloured = recolour_populations(total, red, blue, normal, lattice, 0.7)
    recolour_closure = _maximum_absolute(
        recoloured.red_populations + recoloured.blue_populations - total
    )

    velocity = jnp.broadcast_to(jnp.asarray((0.02, -0.01)), red.shape + (2,))
    energy = jnp.asarray([[2.0, 2.5], [1.5, 3.0]])
    thermal = thermal_equilibrium(energy, velocity, lattice, precision)
    recovered_energy, energy_flux = thermal_raw_moments(thermal, lattice, precision)
    thermal_residual = max(
        _maximum_absolute(recovered_energy - energy),
        _maximum_absolute(energy_flux - energy[..., None] * velocity),
    )

    concentration = jnp.stack((red, blue), axis=-1)
    species = species_equilibrium(concentration, velocity, lattice, precision)
    recovered_species, species_flux = species_raw_moments(species, lattice, precision)
    species_residual = max(
        _maximum_absolute(recovered_species - concentration),
        _maximum_absolute(
            species_flux - oe.contract("...s,...d->...sd", concentration, velocity)
        ),
    )
    tolerance = 2.0e-13
    return {
        "recolouring_closure_residual": recolour_closure,
        "thermal_moment_residual": thermal_residual,
        "species_moment_residual": species_residual,
        "tolerance": tolerance,
        "passed": max(recolour_closure, thermal_residual, species_residual) <= tolerance,
    }


def _discrete_velocity_case() -> dict[str, object]:
    d2v17 = phx.discretization.d2v17_quadrature()
    d2v37 = phx.discretization.d2v37_off_lattice_quadrature()
    passed = bool(
        d2v17.certification.passed
        and d2v37.certification.passed
        and d2v17.transport_kind == "integer_lattice"
        and d2v37.transport_kind == "off_lattice"
    )
    return {
        "d2v17": {
            "maximum_residual": d2v17.certification.maximum_residual,
            "transport_kind": d2v17.transport_kind,
        },
        "d2v37": {
            "maximum_residual": d2v37.certification.maximum_residual,
            "transport_kind": d2v37.transport_kind,
        },
        "passed": passed,
    }


def _qualification_execution_step(step_index, time, populations, step_size, args):
    del time, step_size
    shifts = (
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
    )
    routed = jnp.stack(
        tuple(
            jnp.roll(populations[..., direction], shift=shift, axis=(0, 1))
            for direction, shift in enumerate(shifts)
        ),
        axis=-1,
    )
    candidate = routed + args
    successful = jnp.asarray(True)
    return LatticeBoltzmannExecutionStep(
        candidate,
        candidate,
        successful,
        jnp.asarray(0.0),
        jnp.asarray(populations.size, dtype=jnp.int32),
        {"mass": jnp.sum(candidate), "step": step_index},
    )


def _execution_case() -> dict[str, object]:
    lattice = phx.discretization.D2Q9()
    aa_plan = phx.discretization.AALatticeBoltzmannPlan(lattice)
    canonical = jnp.arange(4 * 5 * 9, dtype=jnp.float64).reshape((4, 5, 9))
    even = aa_plan.encode(canonical, parity=0)
    odd = aa_plan.encode(canonical, parity=1)
    even_residual = _maximum_absolute(aa_plan.canonical(even) - canonical)
    odd_residual = _maximum_absolute(aa_plan.canonical(odd) - canonical)
    replay = phx.solver.FixedStepReplayPolicy("block", block_size=4)

    devices = tuple(jax.devices())
    if len(devices) >= 4:
        selected_devices = devices[:4]
        partition_shape = (2, 2)
    elif len(devices) >= 2:
        selected_devices = devices[:2]
        partition_shape = (2, 1)
    else:
        selected_devices = devices[:1]
        partition_shape = (1, 1)
    global_shape = tuple(4 * count for count in partition_shape)
    generic_halo = DistributedHaloSchedule(
        global_shape,
        partition_shape,
        HaloPlan(
            StencilFootprint(("x", "y"), (1, 1), (1, 1)),
            distributed_neighbors=True,
        ),
        periodic_axes=(True, True),
        devices=selected_devices,
        mesh_axis_prefix="kinetic_qualification",
    )
    halo = phx.discretization.LatticeBoltzmannHaloSchedule(lattice, generic_halo)
    reference = ReferenceLatticeBoltzmannExecutionPlan(
        lattice,
        _qualification_execution_step,
        step_id="kinetic-qualification-periodic-stream",
    )
    sharded_plan = phx.discretization.ShardedLatticeBoltzmannExecutionPlan(
        reference, halo
    )
    population_shape = global_shape + (lattice.population_count,)
    initial = jnp.arange(np.prod(population_shape), dtype=jnp.float64).reshape(
        population_shape
    ) / float(np.prod(population_shape))
    realized = sharded_plan.realize(
        initial,
        step_count=3,
        step_size=jnp.asarray(1.0),
        args=jnp.asarray(0.25),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    jax.block_until_ready(realized.populations)
    distributed_equivalent = bool(realized.equivalence.equivalent)
    fused = phx.discretization.FusedLatticeBoltzmannExecutionPlan(reference)
    fused_result = fused.realize(
        initial,
        step_count=3,
        step_size=jnp.asarray(1.0),
        args=jnp.asarray(0.25),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    jax.block_until_ready(fused_result.populations)
    fused_equivalent = bool(fused_result.equivalence.equivalent)
    return {
        "even_parity_residual": even_residual,
        "odd_parity_residual": odd_residual,
        "checkpoint_identity_distinguishes_parity": aa_plan.checkpoint(
            even, "same-state"
        ).identity
        != aa_plan.checkpoint(odd, "same-state").identity,
        "replay_mode": replay.mode,
        "replay_block_size": replay.block_size,
        "device_count": len(selected_devices),
        "partition_shape": list(partition_shape),
        "population_axis_partitioned": (
            sharded_plan.metadata.population_axis_partitioned
        ),
        "sharded_reference_equivalent": distributed_equivalent,
        "sharded_population_equivalent": bool(
            realized.equivalence.populations_equivalent
        ),
        "sharded_failure_equivalent": bool(realized.equivalence.failures_equivalent),
        "sharded_diagnostics_equivalent": bool(
            realized.equivalence.diagnostics_equivalent
        ),
        "sharded_maximum_absolute_error": float(
            realized.equivalence.maximum_absolute_error
        ),
        "fused_reference_equivalent": fused_equivalent,
        "fused_maximum_absolute_error": float(
            fused_result.equivalence.maximum_absolute_error
        ),
        "passed": (
            even_residual == 0.0
            and odd_residual == 0.0
            and distributed_equivalent
            and fused_equivalent
            and not sharded_plan.metadata.population_axis_partitioned
        ),
    }


def qualification() -> dict[str, object]:
    cases = {
        "lattice": _lattice_case(),
        "moments": _moment_case(),
        "collisions": _collision_case(),
        "boundary_ownership": _boundary_case(),
        "amr_transfer": _transfer_case(),
        "immersed_boundary": _immersed_boundary_case(),
        "geometry_transfer": _geometry_transfer_case(),
        "multiphysics_moments": _multiphysics_case(),
        "discrete_velocity": _discrete_velocity_case(),
        "execution": _execution_case(),
    }
    return {
        "environment": capture_environment().to_dict(),
        "scope": {
            "status": "advanced invariant qualification",
            "stability_claim": False,
            "gradient_export": "JAX only",
            "iree": "forward inference only",
        },
        "cases": cases,
        "passed": all(bool(case["passed"]) for case in cases.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/kinetic_expansion_qualification.json"),
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
