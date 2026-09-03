#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization._axis_domain import AxisDomain
from phydrax.discretization.spectral._basis import FourierBasisPlan
from phydrax.discretization.spectral._space import TensorSpectralPlan
from phydrax.equations._barotropic_beta_plane import BarotropicBetaPlane
from phydrax.statistical_dynamics._beta_plane import BetaPlaneCumulantSystem
from phydrax.statistical_dynamics._cumulants import DenseCumulantState, ForcingCovariance
from phydrax.statistical_dynamics._interactions import InteractionPartition


def _problem(count=8, *, beta=2.0):
    space = TensorSpectralPlan(
        (FourierBasisPlan(count), FourierBasisPlan(count)),
        axis_names=("x", "y"),
        field_name="vorticity",
    ).prepare(
        (
            AxisDomain.periodic(0.0, 2.0 * jnp.pi),
            AxisDomain.periodic(0.0, 2.0 * jnp.pi),
        )
    )
    return BarotropicBetaPlane(space, beta=beta)


def test_beta_plane_rossby_wave_inversion_and_budgets():
    problem = _problem()
    space = problem.discretization
    x = space.axes[0].nodes[:, None]
    y = space.axes[1].nodes[None, :]
    wave = space.project(jnp.cos(x) + 0.0 * y)

    psi = problem.streamfunction(wave)
    velocity = problem.velocity(wave, physical=True)
    linear = problem.linear_tendency(wave)
    budgets = problem.budgets(wave)
    advanced = problem.prepare_etdrk(4).step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        wave,
        jnp.asarray(0.1),
        None,
    )

    np.testing.assert_allclose(space.reconstruct(psi), -jnp.cos(x) + 0.0 * y, atol=1e-11)
    np.testing.assert_allclose(velocity[..., 0], 0.0, atol=1e-11)
    np.testing.assert_allclose(velocity[..., 1], jnp.sin(x) + 0.0 * y, atol=1e-11)
    np.testing.assert_allclose(
        space.reconstruct(linear),
        -problem.beta * (jnp.sin(x) + 0.0 * y),
        atol=1e-11,
    )
    np.testing.assert_allclose(
        advanced.accepted_state,
        jnp.exp(0.1 * problem.linear_diagonal) * wave,
        atol=1e-11,
    )
    np.testing.assert_allclose(budgets.kinetic_energy, 0.25, atol=1e-11)
    np.testing.assert_allclose(budgets.enstrophy, 0.25, atol=1e-11)
    np.testing.assert_allclose(budgets.energy_rate, 0.0, atol=1e-11)
    np.testing.assert_allclose(budgets.enstrophy_rate, 0.0, atol=1e-11)
    assert bool(budgets.successful)


def test_dealiased_jacobian_conserves_energy_and_enstrophy():
    problem = _problem(beta=0.0)
    space = problem.discretization
    x = space.axes[0].nodes[:, None]
    y = space.axes[1].nodes[None, :]
    vorticity = space.project(jnp.cos(x) + 0.4 * jnp.cos(2.0 * y) + 0.3 * jnp.sin(x + y))
    nonlinear = problem.nonlinear_tendency(vorticity)
    budgets = problem.budgets(
        vorticity,
        tendency=nonlinear,
        nonlinear_tendency=nonlinear,
    )

    np.testing.assert_allclose(budgets.nonlinear_energy_rate, 0.0, atol=2e-11)
    np.testing.assert_allclose(budgets.nonlinear_enstrophy_rate, 0.0, atol=2e-11)


def test_hermitian_masks_close_and_ql_gql_reach_exact_limits():
    problem = _problem()
    space = problem.discretization
    partition = InteractionPartition.zonal_mean(
        space,
        zonal_axis=0,
        admissibility_mask=problem.admissibility_mask,
    )
    assert bool(partition.mask_is_closed())
    x = space.axes[0].nodes[:, None]
    y = space.axes[1].nodes[None, :]
    state = problem.project_state(
        space.project(jnp.cos(y) + jnp.cos(x + y) + 0.25 * jnp.sin(2.0 * x - y))
    )

    ql = partition.select(problem.bilinear_tendency, state, model="ql")
    gql = partition.select(problem.bilinear_tendency, state, model="gql")
    np.testing.assert_allclose(ql, gql, atol=2e-11)

    all_low = InteractionPartition.from_wavenumber_cutoff(
        space,
        100,
        admissibility_mask=problem.admissibility_mask,
    )
    nl = all_low.select(problem.bilinear_tendency, state, model="nl")
    gql_limit = all_low.select(problem.bilinear_tendency, state, model="gql")
    continued = partition.select(
        problem.bilinear_tendency,
        state,
        model="ql",
        interaction_coordinate=1.0,
    )
    np.testing.assert_allclose(gql_limit, nl, atol=2e-11)
    np.testing.assert_allclose(continued, problem.nonlinear_tendency(state), atol=2e-11)


def test_beta_plane_coordinates_drive_exact_prepared_gce2_owner():
    problem = _problem(6, beta=0.0)
    partition = InteractionPartition.from_wavenumber_cutoff(
        problem.discretization,
        1,
        axes=(0, 1),
        admissibility_mask=problem.admissibility_mask,
    )
    system = BetaPlaneCumulantSystem(problem, partition)
    forcing = ForcingCovariance(
        jnp.zeros((system.layout.eddy_dimension, system.layout.eddy_dimension))
    )
    prepared = system.prepare(forcing, closure="gce2", time_step=1.0e-3)
    values = jnp.linspace(-0.2, 0.3, system.coordinates.coordinate_size)
    vorticity = system.coordinates.from_coordinates(values)
    expected_selected = system.coordinates.to_coordinates(
        partition.select(
            problem.bilinear_tendency,
            vorticity,
            model="gql",
        )
    )
    actual_selected = prepared.plan.dynamics(values)
    full_nonlinear = system.coordinates.to_coordinates(
        problem.bilinear_tendency(vorticity, vorticity)
    )
    np.testing.assert_allclose(actual_selected, expected_selected, atol=2.0e-11)
    assert float(jnp.max(jnp.abs(actual_selected - full_nonlinear))) > 1.0e-12
    state = DenseCumulantState(
        jnp.zeros((system.layout.mean_dimension,)),
        jnp.zeros((system.layout.eddy_dimension, system.layout.eddy_dimension)),
        layout_id=system.layout.layout_id,
    )

    result = prepared.step(state)

    assert bool(result.evidence.accepted)
    np.testing.assert_allclose(result.state.mean, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.state.covariance, 0.0, atol=1e-12)
