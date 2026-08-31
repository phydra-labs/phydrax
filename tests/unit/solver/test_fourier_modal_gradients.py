from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.spectral import LatticeHarmonicPlan
from phydrax.solver.maxwell import fourier_modal as fm


def _operator():
    harmonics = LatticeHarmonicPlan.parallelogramic((1,), (3,)).prepare(
        jnp.asarray(((1.0, 0.0),))
    )
    material = fm.FrequencyMaxwellMaterial(2.25, material_id="film")
    prepared_material = fm.prepare_fourier_material(
        material,
        harmonics,
        fm.DirectFourierFactorizationPlan(),
    )
    operator = fm.prepare_layer_operator(
        prepared_material,
        harmonics,
        jnp.asarray(2.0 * jnp.pi),
        jnp.asarray((0.0, 0.0)),
    )
    policy = fm.BoundaryCascadePolicy(
        doublings=5,
        initializer_order=7,
        paired_error=False,
        relative_tolerance=1e-7,
    )
    return operator, policy


def test_boundary_thickness_gradient_matches_finite_difference() -> None:
    operator, policy = _operator()

    def objective(thickness):
        relation = fm.prepare_layer_boundary(operator, thickness, policy)
        return jnp.real(jnp.sum(jnp.abs(relation.a) ** 2))

    thickness = jnp.asarray(0.17)
    automatic = jax.grad(objective)(thickness)
    step = 1e-5
    finite_difference = (objective(thickness + step) - objective(thickness - step)) / (
        2.0 * step
    )
    np.testing.assert_allclose(
        np.asarray(automatic),
        np.asarray(finite_difference),
        rtol=2e-4,
        atol=2e-6,
    )


def test_modal_and_boundary_relations_agree_for_uniform_layer() -> None:
    operator, policy = _operator()
    boundary = fm.prepare_layer_boundary(operator, 0.1, policy)
    modal = fm.prepare_modal_boundary(operator, 0.1)
    np.testing.assert_allclose(
        np.asarray(boundary.a), np.asarray(modal.boundary.a), rtol=1e-8, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(boundary.b), np.asarray(modal.boundary.b), rtol=1e-8, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(boundary.c), np.asarray(modal.boundary.c), rtol=1e-8, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(boundary.d), np.asarray(modal.boundary.d), rtol=1e-8, atol=1e-9
    )
