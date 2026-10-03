# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.meshfree._exterior import (
    MeshfreeBoundaryQuadrature,
    MeshfreeExteriorCalculusPlan,
    PreparedMeshfreeExteriorCalculus,
)
from phydrax.discretization.meshfree._exterior_transport import (
    edge_upwind_content,
    MeshfreeAdvection,
    MeshfreeAdvectionResult,
    MeshfreeAdvectionStatus,
)


def _line_exterior() -> PreparedMeshfreeExteriorCalculus:
    return MeshfreeExteriorCalculusPlan(
        np.asarray([[-1.0], [0.0], [1.0]]),
        1.1,
        3,
        node_volumes=np.asarray([1.0, 1.0, 1.0]),
        dirichlet=np.asarray([True, False, True]),
    ).prepare()


def _euler_step(
    exterior: PreparedMeshfreeExteriorCalculus,
    values: jax.Array | np.ndarray,
    flux: jax.Array | np.ndarray,
    dt: jax.Array | float,
) -> MeshfreeAdvectionResult:
    return edge_upwind_content(
        values,
        exterior.node_volumes,
        exterior.incidence,
        flux,
        dt,
        metric_nonnegative=exterior.metric_result.nonnegative,
        metric_accepted=exterior.metric_result.accepted,
    )


def test_upwind_rate_uses_outflow_and_extensive_sources_without_a_time_step() -> None:
    exterior = _line_exterior()
    advection = MeshfreeAdvection(exterior)
    rate = advection.rate(np.asarray([1.0, 2.0, 3.0]), np.asarray([2.0, -1.0]))
    # Hand upwind ledger with unit volumes: donors give edge fluxes (2, -3).
    np.testing.assert_allclose(rate.edge_flux, [2.0, -3.0], atol=1e-12)
    np.testing.assert_allclose(rate.content_rate, [-2.0, 5.0, -3.0], atol=1e-12)
    np.testing.assert_allclose(rate.node_outflow, [2.0, 0.0, 1.0], atol=1e-12)
    # Forward-Euler positivity bound min V/outflow, a spatial certificate only.
    np.testing.assert_allclose(rate.stable_step, 0.5, atol=1e-12)
    np.testing.assert_allclose(rate.conservation_residual, 0.0, atol=1e-12)
    assert int(rate.status) == int(MeshfreeAdvectionStatus.ACCEPTED)
    supplied = advection.rate(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([2.0, -1.0]),
        source=np.asarray([0.0, 4.0, 0.0]),
    )
    np.testing.assert_allclose(jnp.sum(supplied.content_rate), 4.0, atol=1e-12)
    np.testing.assert_allclose(supplied.conservation_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        exterior.divergence(np.asarray([2.0, -1.0])), [2.0, -3.0, 1.0], atol=1e-12
    )


def test_euler_certificate_uses_outflow_cfl() -> None:
    exterior = _line_exterior()
    result = _euler_step(
        exterior, np.asarray([1.0, 2.0, 3.0]), np.asarray([2.0, -1.0]), 0.25
    )
    np.testing.assert_allclose(result.value, [0.5, 3.25, 2.25], atol=1e-12)
    np.testing.assert_allclose(result.cfl, 0.5, atol=1e-12)
    assert bool(result.positivity_admitted)


def test_cfl_refusal_keeps_raw_candidate_and_holds_the_public_state() -> None:
    values = np.asarray([1.0, 2.0, 3.0])
    result = _euler_step(_line_exterior(), values, np.asarray([2.0, -1.0]), 0.75)
    assert int(result.status) == int(MeshfreeAdvectionStatus.CFL_EXCEEDED)
    assert not bool(result.accepted)
    assert not bool(result.positivity_admitted)
    # The unclipped candidate remains inspectable, including its negative node.
    np.testing.assert_allclose(result.candidate_value, [-0.5, 5.75, 0.75], atol=1e-12)
    np.testing.assert_allclose(result.candidate_content, [-0.5, 5.75, 0.75], atol=1e-12)
    np.testing.assert_allclose(result.conservation_residual, 0.0, atol=1e-12)
    # The refused public state is the unchanged input, not a clipped update.
    np.testing.assert_array_equal(result.value, values)
    np.testing.assert_array_equal(result.content, values)


def test_refused_advection_has_no_valid_derivative_through_its_public_state() -> None:
    exterior = _line_exterior()
    values = jnp.asarray([1.0, 2.0, 3.0])
    flux = jnp.asarray([2.0, -1.0])

    def public_total(dt: jax.Array) -> jax.Array:
        return jnp.sum(_euler_step(exterior, values, flux, dt).value)

    def candidate_center(dt: jax.Array) -> jax.Array:
        return _euler_step(exterior, values, flux, dt).candidate_value[1]

    # The nodal content rate (-2, 5, -3) is independent of dt.
    accepted = jax.grad(public_total)(jnp.asarray(0.25))
    np.testing.assert_allclose(accepted, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        jax.grad(lambda dt: _euler_step(exterior, values, flux, dt).value[1])(
            jnp.asarray(0.25)
        ),
        5.0,
        atol=1e-12,
    )
    refused = jax.grad(public_total)(jnp.asarray(0.75))
    assert bool(jnp.isnan(refused))
    # Raw candidate diagnostics stay ordinary differentiable functions of dt.
    np.testing.assert_allclose(
        jax.grad(candidate_center)(jnp.asarray(0.75)), 5.0, atol=1e-12
    )


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
