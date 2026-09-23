import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax.continuation._stability_pencil import (
    ContinuationStabilityPencil,
    GeneralizedPencilStabilityAnalyzer,
    hopf_point_evidence,
    HopfContinuationAdapter,
)


def test_projected_rectangular_residual_requires_declared_lift_and_project():
    residual = lambda state, coordinate, args: jnp.asarray(
        [state[0] + state[1], state[0] - state[1], 2.0 * state[0]]
    )
    pencil = ContinuationStabilityPencil.projected_residual(
        residual,
        lambda value: value,
        lambda value: value[:2],
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        pencil_id="rectangular-projection",
    )
    matrix, mass = pencil.matrices(jnp.zeros((2,)), 0.0)
    assert mass is None
    assert jnp.allclose(matrix, jnp.asarray([[1.0, 1.0], [1.0, -1.0]]))
    assert jax.jit(lambda state: pencil.matrices(state, 0.0)[0])(
        jnp.zeros((2,))
    ).shape == (2, 2)


def test_real_block_hopf_locus_has_local_frequency_and_phase_evidence():
    pencil = ContinuationStabilityPencil(
        lambda state, parameters, args: (
            jnp.asarray([[parameters[0], -1.0], [1.0, parameters[0]]]),
            jnp.eye(2),
        ),
        pencil_id="rotation-pencil",
    )
    adapter = HopfContinuationAdapter(
        lambda state, parameters, args: state - parameters,
        pencil,
        lambda first, second, args: jnp.asarray([first, second]),
        jnp.asarray([1.0, 0.0]),
        coordinate_lower=-1.0,
        coordinate_upper=1.0,
        problem_id="rotation-hopf",
    )
    scale = jnp.sqrt(0.5)
    state = (
        jnp.asarray([0.0, 0.0]),
        jnp.asarray(0.0),
        jnp.asarray([scale, 0.0]),
        jnp.asarray([0.0, -scale]),
        jnp.asarray(1.0),
    )
    residual = adapter.problem.residual(state, 0.0)
    assert all(jnp.allclose(value, 0.0, atol=1.0e-7) for value in residual)
    evidence = hopf_point_evidence(adapter, state, 0.0)
    assert evidence.frequency > 0
    assert evidence.normalization_residual < 1.0e-7
    assert evidence.phase_residual < 1.0e-7


def test_generalized_pencil_omits_hopf_indicator_without_conjugate_pair():
    pencil = ContinuationStabilityPencil(
        lambda state, coordinate, args: (
            jnp.asarray([[coordinate, 0.0], [0.0, -1.0]]),
            None,
        ),
        pencil_id="purely-real-pencil",
    )
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, coordinate, args: (
            jnp.asarray([[coordinate, 0.0], [0.0, -1.0]]) @ state
        ),
        problem_id="purely-real-stability",
    )
    evidence = GeneralizedPencilStabilityAnalyzer(pencil).analyze(
        problem,
        jnp.zeros(2),
        jnp.asarray(0.25),
    )

    assert bool(evidence.successful)
    assert int(evidence.conjugate_pair_count) == 0
    assert jnp.isnan(evidence.leading_complex_real_part)
