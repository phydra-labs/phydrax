#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.cavity_quantum import (
    AdaptiveHcurlCapabilityPlan,
    CavityDipoleCouplingPlan,
    CavityParticipationPlan,
    HcurlCapabilityStatus,
    MaxwellBlochPlan,
    MaxwellBlochState,
    MaxwellBlochVectorField,
    MaxwellEigenmodeNormalizationPlan,
    MaxwellLindbladPlan,
    MaxwellLindbladState,
    MaxwellLindbladVectorField,
    PreparedAdaptiveHcurlCapability,
    PurcellLoweringPlan,
)
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._topology_epoch import TopologyEpoch


def _tetrahedral_mesh() -> Any:
    return CellMesh.from_tetrahedra(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
    )


def test_qft_contracts() -> None:
    plan = MaxwellEigenmodeNormalizationPlan(
        jnp.eye(2),
        jnp.eye(2),
        # ty: ignore[invalid-argument-type]
        [3.0, 5.0],
        hbar=2.0,
        equipartition_tolerance=1e-12,
    )
    electric = jnp.asarray(((1.0, 0.0), (0.0, 2.0)), dtype="complex128")
    magnetic = electric
    result = eqx.filter_jit(plan.prepare().normalize)(electric, magnetic)

    np.testing.assert_allclose(result.evidence.total_energies, [6.0, 10.0])
    np.testing.assert_allclose(result.evidence.normalization_residuals, 0.0, atol=1e-12)
    assert bool(jnp.all(result.evidence.successful))
    assert len(set(result.mode_ids)) == 2
    with pytest.raises(TypeError, match="maximum_modes"):
        MaxwellEigenmodeNormalizationPlan(
            jnp.eye(1),
            jnp.eye(1),
            # ty: ignore[invalid-argument-type]
            [1.0],
            maximum_modes=True,
        )
    first = MaxwellEigenmodeNormalizationPlan(
        jnp.eye(1),
        jnp.eye(1),
        # ty: ignore[invalid-argument-type]
        [1.0],
        maximum_modes=1,
    )
    second = MaxwellEigenmodeNormalizationPlan(
        jnp.eye(1),
        jnp.eye(1),
        # ty: ignore[invalid-argument-type]
        [1.0],
        maximum_modes=2,
    )
    assert first.plan_id != second.plan_id
    modes = (
        # ty: ignore[invalid-argument-type]
        MaxwellEigenmodeNormalizationPlan(jnp.eye(1), jnp.eye(1), [3.0], hbar=2.0)
        .prepare()
        .normalize(jnp.ones((1, 1)), jnp.ones((1, 1)))
    )
    participation = CavityParticipationPlan({"dielectric": jnp.eye(1)}).evaluate(modes)
    coupling = CavityDipoleCouplingPlan(
        jnp.ones((1, 1)),
        # ty: ignore[invalid-argument-type]
        [4.0],
        emitter_id="emitter-a",
        hbar_joule_second=2.0,
    ).evaluate(modes)

    np.testing.assert_allclose(participation.participation_ratios, [[0.5]])
    np.testing.assert_allclose(
        coupling.interaction_energies, 2.0 * coupling.angular_couplings
    )
    assert coupling.angular_coupling_unit == "rad s^-1"
    assert coupling.interaction_energy_unit == "J"
    assert bool(jnp.all(coupling.successful))
    plan = PurcellLoweringPlan(10.0, 10.0, 2.0, 0.25)
    result = plan.lower(0.5)

    np.testing.assert_allclose(result.induced_decay_rate, 0.5)
    np.testing.assert_allclose(result.total_decay_rate, 0.75)
    np.testing.assert_allclose(result.purcell_factor, 2.0)
    np.testing.assert_allclose(result.lamb_shift_angular_frequency, 0.0)
    assert result.rate_unit == "rad s^-1"
    assert bool(result.successful)
    bloch_plan = MaxwellBlochPlan(
        cavity_detuning=0.1,
        emitter_detuning=-0.2,
        angular_coupling=0.3,
        cavity_energy_decay_rate=0.05,
        transverse_decay_rate=0.02,
        longitudinal_decay_rate=0.01,
    )
    bloch_state = MaxwellBlochState(0.2 + 0.1j, 0.1, -0.2, -0.8)
    bloch_rate = MaxwellBlochVectorField(bloch_plan)(jnp.asarray(0.0), bloch_state, None)
    assert bool(jnp.isfinite(bloch_rate.cavity_amplitude))
    assert bool(
        jnp.all(
            jnp.isfinite(
                jnp.asarray(
                    (bloch_rate.bloch_u, bloch_rate.bloch_v, bloch_rate.inversion)
                )
            )
        )
    )
    assert bool(bloch_plan.evidence(bloch_state).successful)
    assert (
        bloch_plan.prepare(
            bloch_state, t0=0.0, t1=0.1
        ).state_coordinates.evidence.domain_kind
        == "full"
    )

    lowering = jnp.asarray(((0.0, 1.0), (0.0, 0.0)), dtype="complex128")
    lindblad_plan = MaxwellLindbladPlan(
        jnp.diag(jnp.asarray((0.0, 1.0))),
        lowering,
        (jnp.sqrt(0.2) * lowering)[None, ...],
        angular_coupling=0.4,
        cavity_detuning=0.1,
        cavity_energy_decay_rate=0.3,
    )
    state = MaxwellLindbladState(0.2 + 0.05j, jnp.diag(jnp.asarray((0.7, 0.3))))
    rate = MaxwellLindbladVectorField(lindblad_plan)(jnp.asarray(0.0), state, None)

    np.testing.assert_allclose(jnp.trace(rate.density_matrix), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        rate.density_matrix, jnp.conj(rate.density_matrix.T), atol=1e-12
    )
    assert bool(lindblad_plan.evidence(state).successful)
    assert (
        lindblad_plan.prepare(
            state, t0=0.0, t1=0.1
        ).state_coordinates.evidence.domain_kind
        == "full"
    )
    mesh = _tetrahedral_mesh()
    high_order = AdaptiveHcurlCapabilityPlan(mesh, requested_polynomial_order=2)
    adaptive = AdaptiveHcurlCapabilityPlan(
        mesh, requested_polynomial_order=1, require_adaptation=True
    )
    native = AdaptiveHcurlCapabilityPlan(mesh, requested_polynomial_order=1)
    resource_refused = AdaptiveHcurlCapabilityPlan(mesh, maximum_edges=5)

    assert high_order.evidence.status == int(HcurlCapabilityStatus.SUPPORTED)
    assert high_order.evidence.assembly_supported
    high_complex = high_order.prepare().space
    assert high_complex.hilbert_complex().space(1).size == 20
    assert adaptive.evidence.status == int(HcurlCapabilityStatus.SUPPORTED)
    assert adaptive.prepare().evidence.adaptation_supported
    assert resource_refused.evidence.status == int(
        HcurlCapabilityStatus.RESOURCE_LIMIT_EXCEEDED
    )
    with pytest.raises(NotImplementedError, match="resources"):
        resource_refused.prepare()
    prepared = native.prepare()
    assert prepared.space.hilbert_complex().space(1).size == 6
    assert prepared.evidence.assembly_supported
    assert not prepared.evidence.adaptation_supported


def _hcurl_moments(space: Any, constant: np.ndarray, rotation: np.ndarray) -> Any:
    """Canonical one-form moments, including non-edge higher-order functionals."""
    return space.interpolant(
        1,
        lambda points: (
            jnp.asarray(constant)[None, :]
            + jnp.cross(jnp.asarray(rotation)[None, :], points)
            + (0.25 * points if space.order > 1 else 0.0)
        ),
    ).values


def _refined_cavity(mesh: Any) -> Any:
    import phydrax as phx

    certified = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    return phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            certified,
            phx.meshing.MarkedMeshAdaptation(np.asarray((0,), dtype=np.int64)),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
            ),
        )
    )


@pytest.mark.parametrize("order", (1, 2))
def test_adaptive_hcurl_transfer_preserves_circulation_and_curl(order: int) -> None:
    mesh = _tetrahedral_mesh()
    prepared = AdaptiveHcurlCapabilityPlan(
        mesh, requested_polynomial_order=order, require_adaptation=True
    ).prepare()
    adaptation = _refined_cavity(mesh)
    constant, rotation = np.asarray((0.3, -1.2, 0.5)), np.asarray((0.7, 0.2, -0.4))
    field = _hcurl_moments(prepared.space, constant, rotation)

    result = prepared.adapt(adaptation, field, accepted_boundary=True)

    assert result.committed and result.receipt is not None and result.receipt.published
    successor = result.capability.space
    np.testing.assert_allclose(
        result.edge_integrals,
        _hcurl_moments(successor, constant, rotation),
        atol=1e-13,
    )
    # A canonical zero-form differential stays curl-free at every order.
    potential = prepared.space.interpolant(
        0, lambda points: (points @ jnp.asarray((1.0, -2.0, 0.5)))[:, None]
    ).values
    gradient = prepared.adapt(
        adaptation,
        prepared.space.exterior_derivative(0, potential),
        accepted_boundary=True,
    )
    np.testing.assert_allclose(
        successor.exterior_derivative(1, gradient.edge_integrals), 0.0, atol=1e-13
    )
    assert result.transfer is not None and result.transfer.evidence.passed
    assert result.transfer.evidence.defect("commuting") <= (
        result.transfer.evidence.tolerance
    )
    assert result.capability.epoch.index == prepared.epoch.index + 1


def test_adaptive_hcurl_rejection_retains_the_accepted_epoch() -> None:
    mesh = _tetrahedral_mesh()
    prepared = AdaptiveHcurlCapabilityPlan(mesh, require_adaptation=True).prepare()
    field = _hcurl_moments(
        prepared.space, np.asarray((1.0, 0.0, 0.0)), np.asarray((0.0, 0.0, 1.0))
    )

    result = prepared.adapt(_refined_cavity(mesh), field, accepted_boundary=False)

    assert not result.committed
    assert result.capability is prepared
    np.testing.assert_array_equal(result.edge_integrals, field)
    assert result.receipt is not None and not result.receipt.published
    assert result.receipt.composition.composition_id == (
        result.receipt.source_composition_id
    )


def test_adaptive_hcurl_rejects_stale_geometry_before_publication() -> None:
    mesh = _tetrahedral_mesh()
    prepared = AdaptiveHcurlCapabilityPlan(mesh, require_adaptation=True).prepare()
    field = _hcurl_moments(
        prepared.space, np.asarray((1.0, 0.0, 0.0)), np.asarray((0.0, 0.0, 1.0))
    )
    before = np.asarray(field).copy()
    stale_epoch = TopologyEpoch(
        prepared.epoch.index,
        "WRONGGEOMETRY",
        prepared.epoch.topology_id,
        prepared.epoch.partition_id,
    )
    stale = PreparedAdaptiveHcurlCapability(
        prepared.space, prepared.evidence, stale_epoch, prepared.prepared_id
    )

    with pytest.raises(ValueError, match="geometry"):
        stale.adapt(_refined_cavity(mesh), field, accepted_boundary=True)

    np.testing.assert_array_equal(field, before)


def test_high_order_hcurl_resource_refusal_retains_all_form_moments() -> None:
    mesh = _tetrahedral_mesh()
    prepared = AdaptiveHcurlCapabilityPlan(
        mesh, requested_polynomial_order=2, require_adaptation=True, maximum_edges=20
    ).prepare()
    field = _hcurl_moments(
        prepared.space, np.asarray((0.3, -0.2, 0.7)), np.asarray((0.0, 0.0, 1.0))
    )

    result = prepared.adapt(_refined_cavity(mesh), field, accepted_boundary=True)

    assert not result.committed
    assert result.diagnostics == "hcurl-resource-limit-exceeded"
    assert result.capability is prepared
    np.testing.assert_array_equal(result.edge_integrals, field)
