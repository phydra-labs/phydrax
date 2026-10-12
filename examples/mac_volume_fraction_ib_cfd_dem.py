#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Real bounded VF-IBM sphere settling, without a drag closure or fitted force.

Run: python examples/mac_volume_fraction_ib_cfd_dem.py --steps 800 --cells 36
Smoke: python examples/mac_volume_fraction_ib_cfd_dem.py --steps 4 --cells 12

The periodic box uses submerged gravity (weight minus Archimedes buoyancy),
not a periodic hydrostatic pressure. The sphere stays away from the box faces;
no periodic shape images are implied. Four cells across the default diameter
is a coarse candidate comparison, not a resolved terminal-drag qualification.
"""

from __future__ import annotations

import argparse
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.equations import MACVolumeFractionIBCFDEMCouplingPlan
from phydrax.solver import MACVolumeFractionIBState


DIAMETER = 0.5
FLUID_DENSITY = 1.0
SOLID_DENSITY = 10.0
DYNAMIC_VISCOSITY = 0.5
GRAVITY = 0.5
BOX_LENGTH = 4.5
STEP_SIZE = 0.0025


class SettlingTrajectory(NamedTuple):
    state: MACVolumeFractionIBState
    time: Array
    position: Array
    relative_speed: Array
    successful: Array
    projection_successful: Array
    impulse_residual: Array
    torque_residual: Array
    geometry_unresolved: Array
    hydrodynamic_drag: Array


def make_settling_case(
    cells: int = 36,
) -> tuple[MACVolumeFractionIBCFDEMCouplingPlan, MACVolumeFractionIBState]:
    """Prepare one sphere and an explicitly bounded dense signed-query route."""
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(cells, periodic=True) for _ in range(3)
        ),
        axis_names=("x", "y", "z"),
    ).prepare(
        np.asarray(
            ((0.0, 0.0, 0.0), (BOX_LENGTH, BOX_LENGTH, BOX_LENGTH)), dtype=np.float64
        )
    )
    fv = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(fv).prepare()
    projection = phx.solver.MACVariableDensityProjectionPlan(
        operators, tolerance=1.0e-9, maximum_iterations=300
    )
    fluid = phx.equations.compile_mac_variable_density_flow(
        phx.equations.MACVariableDensityFlowProblem(3, DYNAMIC_VISCOSITY),
        phx.discretization.MACVariableDensityPlan(
            phx.discretization.MACMomentumPlan(operators).prepare()
        ).prepare(),
        projection,
    )
    volume = np.pi * DIAMETER**3 / 6.0
    particles = phx.discretization.ParticleSetPlan(
        np.asarray((0,), dtype=np.int64),
        np.asarray((SOLID_DENSITY * volume,), dtype=np.float64),
        ambient_dimension=3,
    ).prepare()
    materials = phx.equations.DEMMaterialTable(
        np.asarray((1.0e5,), dtype=np.float64),
        np.asarray((0.25,), dtype=np.float64),
        np.asarray(((0.9,),), dtype=np.float64),
        np.asarray(((0.0,),), dtype=np.float64),
    )
    # Exact physical weight-minus-buoyancy load; no fitted drag is applied.
    effective_gravity = GRAVITY * (1.0 - FLUID_DENSITY / SOLID_DENSITY)
    dem = phx.equations.compile_discrete_element_problem(
        phx.equations.DiscreteElementProblemIR(
            "vf-ib-settling-sphere",
            materials,
            gravity=np.asarray((0.0, 0.0, -effective_gravity), dtype=np.float64),
        ),
        particles,
        phx.discretization.RigidSphereSetPlan(
            np.asarray((DIAMETER / 2.0,), dtype=np.float64), np.zeros(1, dtype=np.int32)
        ),
        phx.discretization.CompliantDEMMethodPlan(
            phx.discretization.DEMContactModelPlan(
                phx.discretization.LinearSpringDashpotNormalPlan(100.0)
            ),
            maximum_contact_routes=1,
        ),
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1),
    )
    dem_state = dem.initialize_state(
        0.0,
        np.asarray(((1.5, 1.5, 2.0),), dtype=np.float64),
        np.zeros((1, 3), dtype=np.float64),
        orientation=np.asarray(((1.0, 0.0, 0.0, 0.0),), dtype=np.float64),
    )
    subdivisions = 4
    cut = phx.equations.MACSDFCutFractionPlan(
        operators,
        dem.dynamics.shapes,
        subdivisions=subdivisions,
        maximum_unresolved_fraction=1.0,
        maximum_queries=cells**3 * subdivisions**3,
    )
    coupling = MACVolumeFractionIBCFDEMCouplingPlan(
        fluid,
        dem.dynamics,
        cut,
        projection,
        fluid_density=FLUID_DENSITY,
        solid_density=SOLID_DENSITY,
    )
    zero = tuple(jnp.zeros(layout.shape, dtype=jnp.float64) for layout in fv.face_layouts)
    state = MACVolumeFractionIBState.initialize(
        coupling,
        dem_state,
        fluid.pack_state(jnp.full(fv.cell_shape, FLUID_DENSITY, dtype=jnp.float64), zero),
    )
    return coupling, state


@eqx.filter_jit
def settle(
    coupling: MACVolumeFractionIBCFDEMCouplingPlan,
    initial: MACVolumeFractionIBState,
    steps: int,
    step_size: float = STEP_SIZE,
) -> SettlingTrajectory:
    """Advance the real fluid/DEM transaction with fixed trajectory capacity."""
    operators = coupling.cut_fraction.operators

    def advance(
        state: MACVolumeFractionIBState,
        index: Array,
    ) -> tuple[
        MACVolumeFractionIBState,
        tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    ]:
        time = index.astype(jnp.float64) * step_size
        result = phx.solver.advance_mac_volume_fraction_ib_cfd_dem(
            coupling, state, time, jnp.asarray(step_size, dtype=jnp.float64)
        )
        accepted = result.accepted_state
        velocity = coupling.fluid.physical_state(
            time + step_size, accepted.fluid_state
        ).velocity
        measures = operators.face_dual_measures[2]
        mean_fluid_vertical = jnp.sum(measures * velocity[2]) / jnp.sum(measures)
        relative = mean_fluid_vertical - accepted.dem_state.kinematics.velocity[0, 2]
        return accepted, (
            accepted.dem_state.kinematics.position[0],
            relative,
            result.successful,
            result.pressure_projection.successful,
            result.evaluation.impulse_residual,
            result.evaluation.torque_residual,
            jnp.max(result.candidate_cut_fraction.unresolved_fraction),
            result.evaluation.body_force[0, 2],
        )

    state, history = jax.lax.scan(advance, initial, jnp.arange(steps, dtype=jnp.int32))
    position, relative, successful, projected, impulse, torque, unresolved, drag = history
    return SettlingTrajectory(
        state,
        (jnp.arange(steps, dtype=jnp.float64) + 1.0) * step_size,
        position,
        relative,
        successful,
        projected,
        impulse,
        torque,
        unresolved,
        drag,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=36)
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()
    if args.cells < 4 or args.steps < 1:
        raise ValueError("At least four cells per axis and one step are required.")
    coupling, state = make_settling_case(args.cells)
    trajectory = settle(coupling, state, args.steps)
    if not bool(jnp.all(trajectory.successful)):
        raise RuntimeError(
            "A fluid/geometry/DEM attempt failed; accepted state was retained."
        )
    print(f"accepted_steps={int(trajectory.state.accepted_steps)}")
    print(f"position={np.asarray(trajectory.position[-1])}")
    print(f"relative_settling_speed={float(trajectory.relative_speed[-1]):.8g}")
    print(
        f"maximum_unresolved_fraction={float(jnp.max(trajectory.geometry_unresolved)):.8g}"
    )
    print(
        f"maximum_impulse_residual={float(jnp.max(jnp.abs(trajectory.impulse_residual))):.3g}"
    )


if __name__ == "__main__":
    main()
