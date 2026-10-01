# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.meshfree._exterior import (
    MeshfreeBoundaryQuadrature,
    MeshfreeExteriorCalculusPlan,
)
from phydrax.discretization.meshfree._exterior_transport import (
    MeshfreeAdvection,
    MeshfreeAdvectionStatus,
)


def test_oriented_upwind_uses_outflow_cfl_and_extensive_sources() -> None:
    exterior = MeshfreeExteriorCalculusPlan(
        np.asarray([[-1.0], [0.0], [1.0]]),
        1.1,
        3,
        node_volumes=np.asarray([1.0, 1.0, 1.0]),
        dirichlet=np.asarray([True, False, True]),
    ).prepare()
    advection = MeshfreeAdvection(exterior)
    result = advection.step(np.asarray([1.0, 2.0, 3.0]), np.asarray([2.0, -1.0]), 0.25)
    np.testing.assert_allclose(result.value, [0.5, 3.25, 2.25], atol=1e-12)
    np.testing.assert_allclose(result.node_outflow, [2.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(result.cfl, 0.5, atol=1e-12)
    np.testing.assert_allclose(result.conservation_residual, 0.0, atol=1e-12)
    assert bool(result.positivity_admitted)
    supplied = advection.step(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([2.0, -1.0]),
        0.25,
        source=np.asarray([0.0, 4.0, 0.0]),
    )
    np.testing.assert_allclose(
        supplied.content_after - supplied.content_before, 1.0, atol=1e-12
    )
    np.testing.assert_allclose(supplied.conservation_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        exterior.divergence(np.asarray([2.0, -1.0])), [2.0, -3.0, 1.0], atol=1e-12
    )


def test_cfl_violation_returns_unmodified_candidate_not_clipped_positivity() -> None:
    exterior = MeshfreeExteriorCalculusPlan(
        np.asarray([[-1.0], [0.0], [1.0]]),
        1.1,
        3,
        node_volumes=np.asarray([1.0, 1.0, 1.0]),
        dirichlet=np.asarray([True, False, True]),
    ).prepare()
    result = MeshfreeAdvection(exterior).step(
        np.asarray([1.0, 2.0, 3.0]), np.asarray([2.0, -1.0]), 0.75
    )
    assert int(result.status) == int(MeshfreeAdvectionStatus.CFL_EXCEEDED)
    assert not bool(result.positivity_admitted)
    np.testing.assert_allclose(result.value, [-0.5, 5.75, 0.75], atol=1e-12)
    np.testing.assert_allclose(result.conservation_residual, 0.0, atol=1e-12)


def test_zero_weights_split_anchoring_even_if_geometric_graph_is_connected() -> None:
    exterior = MeshfreeExteriorCalculusPlan(
        np.asarray([[-1.0], [0.0], [1.0]]),
        1.1,
        3,
        node_volumes=np.asarray([1.0, 1.0, 1.0]),
        dirichlet=np.asarray([True, False, True]),
    ).prepare()
    disconnected = exterior.stiffness_evidence(jnp.asarray([0.0, 0.0]))
    assert not bool(disconnected.anchored_components)
    assert not bool(disconnected.maximum_principle)
    assert not bool(disconnected.spd)
    anchored = exterior.stiffness_evidence(jnp.asarray([1.0, 0.0]))
    assert bool(anchored.anchored_components)
    assert bool(anchored.maximum_principle)


def test_natural_flux_requires_declared_boundary_quadrature() -> None:
    exterior = MeshfreeExteriorCalculusPlan(
        np.asarray([[-1.0], [0.0], [1.0]]),
        1.1,
        3,
        node_volumes=np.asarray([1.0, 1.0, 1.0]),
        dirichlet=np.asarray([True, False, True]),
    ).prepare()
    with pytest.raises(ValueError, match="declared boundary quadrature"):
        exterior.natural_boundary_load(2.0)
    quadrature = MeshfreeBoundaryQuadrature(np.asarray([0.5, 0.0, 0.5]))
    boundary = MeshfreeExteriorCalculusPlan(
        np.asarray([[-1.0], [0.0], [1.0]]),
        1.1,
        3,
        node_volumes=np.asarray([1.0, 1.0, 1.0]),
        dirichlet=np.asarray([True, False, True]),
        boundary_quadrature=quadrature,
    ).prepare()
    np.testing.assert_allclose(
        boundary.natural_boundary_load(np.asarray([2.0, 100.0, 4.0])), [-1.0, 0.0, -2.0]
    )


def test_harmonic_arithmetic_and_supplied_edge_diffusion_agree_with_flux_law() -> None:
    exterior = MeshfreeExteriorCalculusPlan(
        np.asarray([[-1.0], [0.0], [1.0]]),
        1.1,
        3,
        node_volumes=np.asarray([1.0, 1.0, 1.0]),
        dirichlet=np.asarray([True, False, True]),
    ).prepare()
    values = jnp.asarray([0.0, 1.0, 3.0])
    harmonic = exterior.diffusion(np.asarray([1.0, 3.0, 5.0]), averaging="harmonic")
    arithmetic = exterior.diffusion(np.asarray([1.0, 3.0, 5.0]), averaging="arithmetic")
    supplied = exterior.diffusion(np.asarray([1.5, 3.75]), averaging="supplied")
    np.testing.assert_allclose(harmonic.mv(values), [1.5, 6.0, -7.5], atol=1e-8)
    np.testing.assert_allclose(arithmetic.mv(values), [2.0, 6.0, -8.0], atol=1e-8)
    np.testing.assert_allclose(supplied.mv(values), harmonic.mv(values), atol=1e-8)


def test_zero_diffusivity_is_a_constitutive_zero_not_a_zero_hodge_mass() -> None:
    exterior = MeshfreeExteriorCalculusPlan(
        np.asarray([[-1.0], [0.0], [1.0]]),
        1.1,
        3,
        node_volumes=np.asarray([1.0, 1.0, 1.0]),
        dirichlet=np.asarray([True, False, True]),
    ).prepare()
    diffusion = exterior.diffusion(np.asarray([1.0, 0.0]), averaging="supplied")
    np.testing.assert_allclose(
        diffusion.mv(np.asarray([0.0, 1.0, 3.0])), [1.0, -1.0, 0.0], atol=1e-8
    )
    np.testing.assert_allclose(
        diffusion.conservation_residual(np.asarray([0.0, 1.0, 3.0])), 0.0, atol=1e-8
    )
    assert bool(diffusion.evidence.maximum_principle)
    assert not bool(diffusion.native_active)
    zero = exterior.diffusion(np.asarray([0.0, 0.0, 0.0]), averaging="harmonic")
    np.testing.assert_allclose(zero.mv(np.asarray([0.0, 1.0, 3.0])), [0.0, 0.0, 0.0])
    assert not bool(zero.evidence.spd)
