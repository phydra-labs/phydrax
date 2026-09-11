import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


SOURCES = jnp.asarray(
    [[-0.7, -0.6, -0.5], [-0.5, 0.6, 0.4], [0.55, -0.45, 0.5], [0.7, 0.65, -0.4]]
)
TARGETS = jnp.asarray([[0.05, 0.1, 0.2], [0.2, -0.1, -0.15], [-0.1, 0.2, -0.2]])
STRENGTHS = jnp.asarray([0.8, -0.4, 0.3, 1.1])


def _direct(kernel, parameter=0.0):
    radii = jnp.linalg.norm(TARGETS[:, None, :] - SOURCES[None, :, :], axis=-1)
    if kernel == "laplace":
        numerator = jnp.ones_like(radii)
    elif kernel == "helmholtz":
        numerator = jnp.exp(1j * parameter * radii)
    else:
        numerator = jnp.exp(-parameter * radii)
    return jnp.sum(numerator * STRENGTHS[None, :] / (4.0 * jnp.pi * radii), axis=1)


@pytest.mark.parametrize(
    ("plan_type", "keyword", "kernel"),
    [
        (phx.operators.LaplaceMultipolePlan3D, {}, "laplace"),
        (phx.operators.HelmholtzMultipolePlan3D, {"wavenumber": 0.7}, "helmholtz"),
        (
            phx.operators.ModifiedHelmholtzMultipolePlan3D,
            {"decay": 0.7},
            "modified-helmholtz",
        ),
    ],
)
def test_complete_multipole_pipelines_match_direct(plan_type, keyword, kernel):
    prepared = plan_type(
        SOURCES,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        reference_targets=TARGETS,
        depth=2,
        expansion_order=3,
        **keyword,
    ).prepare()
    result = prepared.evaluate(SOURCES, STRENGTHS, TARGETS)
    expected = _direct(kernel, 0.7)
    np.testing.assert_allclose(result.values, expected, rtol=4e-3, atol=4e-4)
    assert bool(result.successful)
    assert int(result.p2m_count) == SOURCES.shape[0]
    assert int(result.m2m_count) > 0
    assert int(result.m2l_count) > 0
    assert int(result.l2l_count) > 0
    assert int(result.l2p_count) == TARGETS.shape[0]
    assert int(result.p2p_count) > 0
    assert bool(result.capacity.successful)
    assert bool(result.truncation.well_separated)


def test_laplace_elementary_translations_compose():
    prepared = phx.operators.LaplaceMultipolePlan3D(
        SOURCES,
        [-2.0, -2.0, -2.0],
        [2.0, 2.0, 2.0],
        reference_targets=TARGETS,
        depth=2,
        expansion_order=4,
    ).prepare()
    child = jnp.asarray([-0.2, 0.1, 0.0])
    parent = jnp.asarray([0.0, 0.0, 0.0])
    multipole = prepared.p2m(SOURCES, STRENGTHS, child)
    translated = prepared.m2m(multipole, child, parent)
    exterior = jnp.asarray([[2.5, -0.2, 0.4], [2.2, 1.4, -0.8]])
    direct = jnp.sum(
        STRENGTHS[None, :]
        / (
            4.0
            * jnp.pi
            * jnp.linalg.norm(exterior[:, None, :] - SOURCES[None, :, :], axis=-1)
        ),
        axis=1,
    )
    np.testing.assert_allclose(
        prepared.multipole_to_point(translated, parent, exterior),
        direct,
        rtol=6e-3,
        atol=2e-5,
    )

    local_center = jnp.asarray([2.4, 0.1, -0.1])
    local = prepared.m2l(translated, parent, local_center)
    nearby = local_center + jnp.asarray([[0.02, 0.01, -0.01], [-0.03, 0.02, 0.01]])
    direct_nearby = jnp.sum(
        STRENGTHS[None, :]
        / (
            4.0
            * jnp.pi
            * jnp.linalg.norm(nearby[:, None, :] - SOURCES[None, :, :], axis=-1)
        ),
        axis=1,
    )
    np.testing.assert_allclose(
        prepared.l2p(local, local_center, nearby),
        direct_nearby,
        rtol=4e-3,
        atol=4e-5,
    )
