#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._physical import RelativityScaleContract
from phydrax.applications.numerical_relativity._boundaries import PeriodicBoundary
from phydrax.applications.numerical_relativity._derivatives import FourthOrderDerivatives
from phydrax.applications.numerical_relativity._einstein_vlasov import (
    EinsteinVlasovConstraintSolveEvidence,
    EinsteinVlasovConstraintSolveResult,
    EinsteinVlasovMatterPlan,
)
from phydrax.applications.numerical_relativity._enforcement import (
    Z4cAlgebraicEnforcement,
)
from phydrax.applications.numerical_relativity._gauge import GeodesicGauge
from phydrax.applications.numerical_relativity._grid import FixedGridGeometry
from phydrax.applications.numerical_relativity._state import flat_z4c_state
from phydrax.applications.numerical_relativity._z4c import (
    z4c_adm_geometry,
    Z4cSystem,
)
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.discretization.particle._relativistic_stress_transfer import (
    RelativisticStressDepositPlan,
)
from phydrax.metrix import RelativityConvention
from phydrax.units import KILOGRAM


def test_constraint_admission_stage_exact_coupling_and_atomic_commit_workflow():
    shape = (5, 5, 5)
    grid = FixedGridGeometry(shape, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), periodic=True)
    transfer_grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(5) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))))
    support = phx.discretization.ParticleSetPlan(
        jnp.asarray((101, 202), dtype=jnp.int64),
        jnp.ones((2,)),
        ambient_dimension=3,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(transfer_grid).prepare(support)
    scale = RelativityScaleContract.geometric(KILOGRAM)
    convention = RelativityConvention.canonical()
    units = RelativisticUnitContract(scale, convention)
    stress = RelativisticStressDepositPlan(
        splat,
        units,
        jnp.asarray((13,), dtype=jnp.int32),
        jnp.asarray((1.0,)),
        topology_id=grid.grid_id,
        mass_shell_relative_tolerance=1.0e-5,
        conservation_tolerance=1.0e-5,
        frame_momentum_relative_tolerance=1.0e-5,
    )
    system = Z4cSystem(
        scale,
        convention,
        chart_id="cartesian",
        constraint_tolerance=1.0e-3,
    )
    spatial_coordinates = jnp.moveaxis(grid.coordinates, 0, -1)
    coordinates = jnp.concatenate((jnp.zeros(shape + (1,)), spatial_coordinates), axis=-1)

    def frame_provider(geometry, time, scale_factor):
        return LocalRelativisticFramePlan.from_adm(
            geometry,
            units,
            coordinates.at[..., 0].set(time),
            time,
            scale_factor,
            observer_id="workflow-eulerian-observer",
            orientation_id="right-handed-future",
        )

    initial_geometry = z4c_adm_geometry(
        system,
        grid,
        flat_z4c_state(shape, grid_id=grid.grid_id),
        snapshot_token=0,
    )
    initial_frame = frame_provider(initial_geometry, jnp.asarray(0.0), jnp.asarray(1.0))
    particles = stress.initialize(
        jnp.full((2,), 1.0e-30),
        jnp.asarray(((1.5, 2.0, 2.0), (2.5, 2.0, 2.0))),
        jnp.asarray(((0.2, 0.0, 0.0), (-0.2, 0.0, 0.0))),
        jnp.full((2,), 13, dtype=jnp.int32),
        initial_frame,
    )
    plan = EinsteinVlasovMatterPlan(
        system,
        grid,
        FourthOrderDerivatives(shape, grid.spacing),
        GeodesicGauge(),
        PeriodicBoundary(),
        Z4cAlgebraicEnforcement(maximum_correction=1.0),
        stress,
        frame_provider,
        frame_provider_id="workflow-adm-frame-provider",
        time_step=0.01,
        source_tolerance=1.0e-4,
        mass_shell_tolerance=1.0e-4,
        energy_condition_tolerance=1.0e-6,
        maximum_consecutive_failures=2,
    )

    def constraint_solver(z4c, particle_state, projection, geometry):
        del particle_state, projection, geometry
        return EinsteinVlasovConstraintSolveResult(
            z4c,
            EinsteinVlasovConstraintSolveEvidence(
                0.0,
                jnp.zeros((3,)),
                0.0,
                jnp.zeros((3,)),
                1,
                True,
                True,
                solver_id="workflow-manufactured-solve",
            ),
        )

    state = plan.initialize(
        flat_z4c_state(shape, grid_id=grid.grid_id),
        particles,
        constraint_solver,
    )
    first = plan.advance(state)
    second = plan.advance(first.accepted)

    assert bool(first.successful) and bool(second.successful)
    assert int(second.accepted.accepted_steps) == 2
    np.testing.assert_allclose(second.accepted.time, 0.02, rtol=0.0, atol=1.0e-8)
    assert (
        first.endpoint_stress.projection.snapshot_token
        != second.endpoint_stress.projection.snapshot_token
    )
    assert bool(
        second.endpoint_stress.projection.compatible_with(second.endpoint_geometry)
    )
    assert bool(second.evidence.resource_valid)
    assert bool(second.evidence.strong_field_supported)
    assert bool(second.evidence.qualified)
    np.testing.assert_allclose(
        jnp.sum(second.endpoint_stress.source_integrals.momentum_covector),
        0.0,
        atol=1.0e-10,
    )
