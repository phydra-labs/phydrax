import jax.numpy as jnp

from phydrax.enforcement import (
    BoxProjection,
    PositiveSemidefiniteProjection,
    SecondOrderConeProjection,
    SimplexProjection,
)


def test_closed_feasibility_projections_certify_their_results():
    box = BoxProjection(0.0, 1.0)
    box_value, box_evidence = box.project(jnp.asarray([-2.0, 0.25, 4.0]))
    assert jnp.allclose(box_value, jnp.asarray([0.0, 0.25, 1.0]))
    assert box_evidence.certified

    simplex = SimplexProjection()
    simplex_value, simplex_evidence = simplex.project(jnp.asarray([-1.0, 2.0, 3.0]))
    assert jnp.all(simplex_value >= 0.0)
    assert jnp.allclose(jnp.sum(simplex_value), 1.0)
    assert simplex_evidence.feasible
    assert simplex_evidence.optimality_certified
    assert not simplex_evidence.derivative_certified


def test_cone_and_psd_projections_report_independent_feasibility():
    cone = SecondOrderConeProjection(3)
    cone_value, cone_evidence = cone.project(jnp.asarray([-1.0, 2.0, 0.0]))
    assert cone_value[0] >= jnp.linalg.norm(cone_value[1:]) - 1e-6
    assert cone_evidence.feasible

    psd = PositiveSemidefiniteProjection(2)
    matrix = jnp.asarray([[1.0, 2.0], [2.0, -3.0]])
    psd_value, psd_evidence = psd.project(matrix)
    assert jnp.min(jnp.linalg.eigvalsh(psd_value)) >= -1e-6
    assert psd_evidence.feasible
