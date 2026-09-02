#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


@dataclass(frozen=True)
class AdvancedPICQualification:
    population_active: int
    population_mass_defect: float
    collision_momentum_defect: float
    collision_energy_defect: float
    reduced_energy_defect: float
    reduced_gauss_defect: float
    locator_inside: int
    compile_and_first_ms: float
    steady_ms: float
    successful: bool


def run(*, smoke=False):
    count = 16 if smoke else 64
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(count, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(8), jnp.ones((8,)), ambient_dimension=1
    ).prepare()
    population = phx.discretization.ParticlePopulationPlan(particles).initialize()
    velocity = jnp.stack(
        (jnp.linspace(-0.2, 0.2, 8), jnp.zeros((8,)), jnp.zeros((8,))), axis=-1
    )
    collision_plan = phx.discretization.pic.collisions.CoulombCollisionPlan(
        1.0, maximum_probability=0.2
    )

    @jax.jit
    def collide(values, key):
        return collision_plan.collide(
            values,
            population.mass,
            population.active,
            population.incarnation,
            key,
            0.1,
        )

    started = perf_counter()
    collision = collide(velocity, jr.key(3))
    jax.block_until_ready(collision.accepted_velocity)
    first_ms = 1.0e3 * (perf_counter() - started)
    repetitions = 3 if smoke else 10
    started = perf_counter()
    for index in range(repetitions):
        collision = collide(velocity, jr.key(index + 10))
    jax.block_until_ready(collision.accepted_velocity)
    steady_ms = 1.0e3 * (perf_counter() - started) / repetitions

    field_plan = phx.solver.CompatibleMaxwell1DPlan(grid)
    field = field_plan.initialize()
    current = (
        jnp.zeros((count,)),
        jnp.sin(2.0 * jnp.pi * jnp.arange(count) / count) * 1.0e-3,
        jnp.zeros((count,)),
    )
    old_energy = field_plan.energy(field)
    field, diagnostics = field_plan.step(field, current, 0.1 * field_plan.stable_dt)
    energy_defect = (
        field_plan.energy(field)
        - old_energy
        + 0.1 * field_plan.stable_dt * diagnostics.source_power
    )

    mesh = phx.discretization.CellMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        (phx.discretization.CellBlock("tri", "triangle", jnp.asarray(((0, 1, 2),))),),
    )
    finite_element = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    located = phx.discretization.PreparedSimplicialCellLocator(
        phx.discretization.fem.prepare_finite_element_cell_map(finite_element, 0),
        finite_element.default_runtime.coordinates,
        phx.discretization.SimplicialLocationPolicy(1, 8, 3),
    ).locate(jnp.asarray(((0.2, 0.2), (0.3, 0.1))))
    successful = bool(
        collision.successful
        and diagnostics.successful
        and located.successful.all()
        and collision.momentum_defect < 1.0e-10
        and jnp.abs(collision.energy_defect) < 1.0e-10
    )
    return AdvancedPICQualification(
        int(jnp.sum(population.active)),
        float(jnp.sum(population.mass) - jnp.sum(particles.masses)),
        float(collision.momentum_defect),
        float(collision.energy_defect),
        float(energy_defect),
        float(diagnostics.electric_constraint_linf),
        int(jnp.sum(located.inside)),
        first_ms,
        steady_ms,
        successful,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(smoke=args.smoke)
    payload = json.dumps(asdict(report), indent=2)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report.successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
