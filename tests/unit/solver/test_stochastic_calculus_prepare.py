import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.solver._rough_prepare import (
    prepare_rough_evolution,
    RoughEvolutionPolicy,
    solve_prepared_rough,
)
from phydrax.stochastic._calculus import stratonovich_correction


def test_stratonovich_correction_is_jittable_and_differentiable_on_fixed_route():
    vector_fields = lambda state: jnp.stack((state, 2.0 * state), axis=-1)
    compiled = jax.jit(
        lambda state: stratonovich_correction(vector_fields, state).correction
    )
    state = jnp.asarray([0.4])
    correction = compiled(state)
    derivative = jax.jacfwd(compiled)(state)

    assert jnp.allclose(correction, 2.5 * state)
    assert jnp.allclose(derivative, jnp.asarray([[2.5]]))


def test_projected_stratonovich_route_retains_geometry_evidence():
    geometry = phx.metrix.EuclideanStateGeometry()
    result = stratonovich_correction(
        lambda state: state[..., None],
        jnp.asarray([0.7]),
        geometry=geometry,
    )

    assert result.valid
    assert result.geometry_id == geometry.geometry_id
    assert jnp.allclose(result.tangent_residual, 0.0)
    assert jnp.allclose(result.correction, jnp.asarray([0.35]))


def test_rough_preparation_selects_existing_davie_route_and_executes():
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: 0.8 * state[..., None],
        jnp.asarray([1.0]),
        driver_dimension=1,
        problem_id="prepared-linear-rde",
    )
    control = phx.stochastic.GeometricRoughPath.from_values(
        jnp.asarray([0.0, 0.5, 1.0]),
        jnp.asarray([[0.0], [0.5], [1.0]]),
    )
    prepared = prepare_rough_evolution(
        problem,
        control,
        RoughEvolutionPolicy(
            order=2,
            p_variation_upper=2.0,
            vector_field_regularity=3.0,
            candidate_solvers=(phx.solver.Davie(), phx.solver.RoughEuler()),
        ),
    )
    solution = solve_prepared_rough(prepared, save_times=jnp.asarray([1.0]))

    assert prepared.solver.solver_name == "Davie"
    assert solution.solver_id == "rough-solver:davie:v1"
    assert solution.valid[0]


def test_rough_preparation_fails_closed_on_insufficient_regularity():
    problem = phx.solver.RoughDifferentialProblem(
        lambda time, state, args: state[..., None],
        jnp.asarray([1.0]),
        driver_dimension=1,
        problem_id="invalid-regularity-rde",
    )
    control = phx.stochastic.GeometricRoughPath.from_values(
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([[0.0], [1.0]]),
    )
    with pytest.raises(ValueError, match="No rough solver"):
        prepare_rough_evolution(
            problem,
            control,
            RoughEvolutionPolicy(
                order=2,
                p_variation_upper=3.0,
                vector_field_regularity=3.0,
            ),
        )
