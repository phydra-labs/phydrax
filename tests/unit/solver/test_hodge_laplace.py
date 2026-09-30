#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.divergence_free_neural_field import velocity
from examples.hodge_laplace_mixed import square_complex
from examples.learned_hodge_darcy import train
from examples.maxwell_cavity_whitney import cube_complex
from phydrax.discretization import (
    CochainDiscretization,
    cubical_cell_complex,
    DiagonalHodge,
)
from phydrax.linalg import harmonic_subspace
from phydrax.solver import HodgeLaplacePlan, maxwell_cavity_modes
from phydrax.solver._hodge_laplace import HodgeLaplaceFormulation


pytestmark = [pytest.mark.strict_jax, pytest.mark.filterwarnings("error")]


@pytest.mark.parametrize("formulation", ["mixed", "primal"])
def test_hodge_plan_convergence(formulation: HodgeLaplaceFormulation) -> None:
    """The exact one-form d(sin(pi x)sin(pi y)) has eigenvalue 2 pi²."""
    errors: list[float] = []
    for n in (4, 8):
        realization = square_complex(n)
        hilbert = realization.hilbert_complex(boundary="relative")
        active = realization.active_indices(0, boundary="relative")
        coordinates = np.asarray(
            [(i / n, j / n) for j in range(n + 1) for i in range(n + 1)], dtype=np.float64
        )
        nodal = jnp.asarray(
            np.sin(np.pi * coordinates[:, 0]) * np.sin(np.pi * coordinates[:, 1])
        )
        exact = hilbert.differential(0).mv(nodal[active])
        source = (2.0 * np.pi**2) * exact
        harmonic = harmonic_subspace(hilbert, 1, expected_dimension=0)
        plan = HodgeLaplacePlan(
            realization,
            1,
            boundary="relative",
            formulation=formulation,
            harmonic=harmonic,
        )
        result = plan.solve(source)
        assert bool(result.successful), result.solve_result.status
        error = result.u - exact
        errors.append(
            float(
                jnp.sqrt(
                    hilbert.space(1).inner(error, error)
                    / hilbert.space(1).inner(exact, exact)
                )
            )
        )
        assert float(result.residual_norm) < 1e-7
        assert float(result.harmonic_defect) < 1e-10
        if formulation == "mixed":
            assert result.sigma is not None
    assert errors[1] < 0.45 * errors[0], errors
    assert errors[1] < 0.1, errors


def test_cavity_spectrum() -> None:
    """The first PEC cube cluster is the threefold eigenvalue 2 pi²."""
    errors: list[float] = []
    for n in (2, 3):
        result = maxwell_cavity_modes(cube_complex(n), count=3)
        assert bool(result.successful), result.status
        values = np.asarray(result.eigenvalues)
        assert np.all(values > 0.0)
        assert np.max(np.asarray(result.relative_residuals)) < 1e-6
        errors.append(float(np.max(np.abs(values / (2.0 * np.pi**2) - 1.0))))
    assert errors[1] < errors[0], errors
    assert errors[1] < 0.35, errors


def test_learned_hodge_conservation() -> None:
    """Positive metric learning preserves the source-free Darcy equation at every step."""
    weights, losses, conservation = train(40)
    assert bool(jnp.all(jnp.isfinite(weights)))
    assert float(losses[-1]) < 0.1 * float(losses[0])
    assert float(jnp.max(conservation)) < 1e-11


def test_hodge_harmonic_load_requires_mixed_multiplier() -> None:
    topology = cubical_cell_complex((3, 3), periodic=True).topology
    realization = CochainDiscretization(
        topology,
        tuple(
            DiagonalHodge(jnp.ones((entity.count,), dtype=jnp.float64))
            for entity in topology.entity_sets
        ),
    )
    hilbert = realization.hilbert_complex(boundary="absolute")
    harmonic = harmonic_subspace(hilbert, 0, expected_dimension=1)
    load = jnp.ones((hilbert.space(0).size,), dtype=jnp.float64)
    mixed = HodgeLaplacePlan(
        realization, 0, boundary="absolute", formulation="mixed", harmonic=harmonic
    )
    result = mixed.solve(load)
    assert bool(result.successful), result.solve_result.status
    assert result.p is not None
    np.testing.assert_allclose(np.asarray(result.u), 0.0, atol=1e-9)
    represented_load = harmonic.basis @ result.p
    np.testing.assert_allclose(np.asarray(represented_load), np.asarray(load), atol=1e-9)
    primal = HodgeLaplacePlan(
        realization, 0, boundary="absolute", formulation="primal", harmonic=harmonic
    )
    with pytest.raises(eqx.EquinoxRuntimeError):
        primal.solve(load)


def test_neural_potential_has_analytic_solenoidal_velocity() -> None:
    parameters = (
        jnp.asarray([[1.0, 1.0, 0.0]], dtype=jnp.float64),
        jnp.zeros((1,), dtype=jnp.float64),
        jnp.asarray([[0.0], [0.0], [1.0]], dtype=jnp.float64),
    )
    point = jnp.asarray([0.3, -0.1, 0.7], dtype=jnp.float64)
    amplitude = 1.0 / np.cosh(0.2) ** 2
    np.testing.assert_allclose(
        np.asarray(velocity(parameters, point)), [amplitude, -amplitude, 0.0], atol=1e-12
    )
    derivative = jax.jacfwd(velocity, argnums=1)(parameters, point)
    assert abs(float(jnp.trace(derivative))) < 1e-12


def test_whitney_polynomial_derivatives_are_finite_at_coordinate_zeros() -> None:
    from jax import Array

    from phydrax.discretization.fem import form_element

    element = form_element("triangle", 0, 1, family="trimmed", proxy="scalar")
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64)
    values, derivatives = element.tabulate(vertices)
    np.testing.assert_allclose(values, np.eye(3, dtype=np.float64), atol=1e-13)
    expected = np.asarray([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    np.testing.assert_allclose(
        derivatives, np.broadcast_to(expected, (3, 3, 2)), atol=1e-13
    )

    def nodal_values(point: Array, /) -> Array:
        return element.tabulate(point[None, :])[0][0]

    differentiated = jax.jacfwd(nodal_values)(vertices[0])
    np.testing.assert_allclose(differentiated, expected, atol=1e-13)
