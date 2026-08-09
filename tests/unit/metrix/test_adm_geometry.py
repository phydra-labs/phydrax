#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from math import pi

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _chart(name="adm_spacetime"):
    return phx.metrix.CoordinateChart(name, ("t", "x", "y", "z"))


@pytest.mark.parametrize(
    ("convention", "timelike_sign"),
    (("mostly_plus", -1.0), ("mostly_minus", 1.0)),
)
def test_adm_normal_and_projector_obey_signed_hypersurface_identities(
    convention,
    timelike_sign,
):
    metric = phx.metrix.adm_metric(
        lambda q: 1.3 + 0.1 * q[0],
        lambda q: jnp.array([0.2, -0.1 * q[1], 0.05]),
        lambda q: jnp.array(
            [[1.4, 0.1, 0.0], [0.1, 1.8, -0.05], [0.0, -0.05, 2.1]]
        ),
        chart=_chart(),
        convention=convention,
    )
    point = jnp.array([0.2, -0.3, 0.1, 0.4])
    normal = phx.metrix.adm_normal_vector(metric, point)
    conormal = phx.metrix.adm_normal_covector(metric, point)
    projector = phx.metrix.adm_spacetime_projector(metric, point)

    assert normal[0] > 0.0
    assert jnp.allclose(metric(point) @ normal, conormal)
    assert jnp.allclose(normal @ conormal, timelike_sign)
    assert jnp.allclose(projector @ normal, 0.0)
    assert jnp.allclose(projector @ projector, projector)
    assert jnp.allclose(jnp.trace(projector), 3.0)


def _inhomogeneous_metric(convention="mostly_plus"):
    def matrix(coordinates):
        time, x, _, _ = coordinates
        lapse = jnp.exp(0.2 * x)
        spatial = jnp.diag(
            jnp.array(
                [
                    jnp.exp(2.0 * (0.2 * time + 0.1 * x)),
                    jnp.exp(2.0 * (-0.1 * time + 0.05 * x)),
                    jnp.exp(2.0 * (0.15 * time - 0.08 * x)),
                ]
            )
        )
        signsafe = phx.metrix.adm_metric(
            lambda q: lapse + 0.0 * q[0],
            lambda q: jnp.zeros((3,), dtype=q.dtype),
            lambda q: spatial + 0.0 * q[0],
            chart=_chart("inhomogeneous"),
            convention=convention,
        )
        return signsafe(coordinates)

    return phx.metrix.LorentzianMetric(
        matrix,
        chart=_chart("inhomogeneous"),
        convention=convention,
    )


@pytest.mark.parametrize("convention", ("mostly_plus", "mostly_minus"))
def test_adm_gauss_codazzi_constraints_match_spacetime_einstein_projections(
    convention,
):
    metric = _inhomogeneous_metric(convention)
    point = jnp.array([0.3, 0.2, 0.1, -0.1])
    decomposition = phx.metrix.decompose_adm_metric(metric, point)
    normal = phx.metrix.adm_normal_vector(metric, point)
    einstein = phx.metrix.einstein_tensor(metric, point)

    hamiltonian = phx.metrix.adm_hamiltonian_constraint(
        metric,
        point,
        einstein_coupling=0.0,
    )
    momentum = phx.metrix.adm_momentum_constraint(
        metric,
        point,
        einstein_coupling=0.0,
    )
    expected_hamiltonian = 2.0 * jnp.einsum(
        "i,ij,j->",
        normal,
        einstein,
        normal,
    )
    expected_momentum = -decomposition.spatial_inverse @ jnp.einsum(
        "m,mj->j",
        normal,
        einstein[:, 1:],
    )

    assert jnp.allclose(hamiltonian, expected_hamiltonian, atol=1e-10)
    assert jnp.allclose(momentum, expected_momentum, atol=1e-10)


def test_adm_extrinsic_curvature_and_sourced_constraints_match_flat_flrw():
    chart = _chart("flat_flrw")
    expansion_rate = 0.2

    def metric_from_rate(rate):
        return phx.metrix.LorentzianMetric(
            lambda q: jnp.diag(
                jnp.array(
                    [
                        -1.0,
                        jnp.exp(2.0 * rate * q[0]),
                        jnp.exp(2.0 * rate * q[0]),
                        jnp.exp(2.0 * rate * q[0]),
                    ]
                )
            ),
            chart=chart,
        )

    point = jnp.array([0.3, 0.1, -0.2, 0.4])
    metric = metric_from_rate(expansion_rate)
    decomposition = phx.metrix.decompose_adm_metric(metric, point)
    extrinsic = phx.metrix.adm_extrinsic_curvature(metric, point)
    coupling = 8.0 * pi
    energy_density = 3.0 * expansion_rate**2 / coupling
    constraints = phx.metrix.adm_constraint_residuals(
        metric,
        point,
        energy_density=energy_density,
        momentum_density=jnp.zeros((3,)),
        einstein_coupling=coupling,
    )

    def vacuum_hamiltonian(rate):
        return phx.metrix.adm_hamiltonian_constraint(
            metric_from_rate(rate),
            point,
            einstein_coupling=0.0,
        )

    derivative = jax.jit(jax.grad(vacuum_hamiltonian))(jnp.asarray(expansion_rate))

    assert jnp.allclose(
        extrinsic,
        -expansion_rate * decomposition.spatial_metric,
    )
    assert jnp.allclose(constraints.hamiltonian, 0.0, atol=1e-10)
    assert jnp.allclose(constraints.momentum, 0.0, atol=1e-10)
    assert constraints.maximum_absolute < 1e-10
    assert jnp.allclose(derivative, 12.0 * expansion_rate)


def test_adm_constraint_sources_reject_incompatible_shapes():
    metric = phx.metrix.minkowski_metric(_chart())
    points = jnp.zeros((2, 4))

    with pytest.raises(ValueError, match="energy_density"):
        phx.metrix.adm_hamiltonian_constraint(
            metric,
            points,
            energy_density=jnp.zeros((2, 1)),
        )
    with pytest.raises(ValueError, match="momentum_density"):
        phx.metrix.adm_momentum_constraint(
            metric,
            points,
            momentum_density=jnp.zeros((2, 2)),
        )
