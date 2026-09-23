import numpy as np
import pytest

from phydrax.discretization import (
    PeriodicCell,
    ReciprocalConnectivityPlan,
    ReciprocalMeshPlan,
)
from phydrax.operators.periodic import (
    identity_cross_k_connection,
    periodic_translation_family_from_dense_blocks,
    PeriodicBandManifold,
    PeriodicBlochGauge,
    PeriodicChernPlan,
    PeriodicChernRefinementEvidence,
    PeriodicCrossKConnection,
    PeriodicOrbitalBasisPlan,
    PeriodicOrbitalPencilPlan,
    PeriodicOverlapBundle,
    PeriodicSpectrumPlan,
    PeriodicWilsonPlan,
)
from phydrax.units import ANGSTROM, ELECTRONVOLT


def _ssh(intercell=1.0, intracell=0.4, points=8):
    cell = PeriodicCell([[1.0]])
    basis = PeriodicOrbitalBasisPlan(
        cell,
        ("A", "B"),
        [[0.0], [0.5]],
        ANGSTROM,
        PeriodicBlochGauge("lattice"),
    )
    blocks = np.zeros((3, 2, 1, 2, 1), dtype="complex128")
    blocks[1, 0, 0, 1, 0] = intracell
    blocks[1, 1, 0, 0, 0] = intracell
    blocks[0, 0, 0, 1, 0] = intercell
    blocks[2, 1, 0, 0, 0] = intercell
    family = periodic_translation_family_from_dense_blocks([[-1], [0], [1]], blocks)
    pencil = PeriodicOrbitalPencilPlan.orthonormal(
        basis, family.plan, family.state, ELECTRONVOLT
    ).prepare()
    mesh = ReciprocalMeshPlan.monkhorst_pack(cell, (points,))
    spectrum = PeriodicSpectrumPlan(pencil, mesh).evaluate()
    connectivity = ReciprocalConnectivityPlan.regular(mesh).prepare()
    connection = identity_cross_k_connection(pencil, connectivity)
    return pencil, mesh, spectrum, connectivity, connection


def test_ssh_wilson_loop_returns_raw_links_and_quantized_zak_phase():
    _, mesh, spectrum, connectivity, connection = _ssh()
    manifold = PeriodicBandManifold(spectrum, [0])
    bundle = PeriodicOverlapBundle(manifold, connection)
    forward_edges = np.arange(0, 2 * mesh.fractional_points.shape[0], 2)
    wilson = PeriodicWilsonPlan(bundle, forward_edges).evaluate()

    np.testing.assert_allclose(abs(float(wilson.zak_phase)), np.pi, atol=1.0e-8)
    assert wilson.raw_link_determinant_phases.shape == forward_edges.shape
    assert np.min(np.asarray(wilson.link_singular_values)) > 0.0
    assert bool(wilson.successful)


def test_topology_rejects_gap_and_link_rank_ambiguity():
    pencil, mesh, _, connectivity, _ = _ssh(intercell=1.0, intracell=1.0)
    points = np.arange(8, dtype="float64")[:, None] / 8.0
    explicit_mesh = ReciprocalMeshPlan(
        mesh.cell,
        points,
        np.full(8, 1.0 / 8.0),
        mesh_shape=(8,),
        shift=(0.0,),
    )
    touching = PeriodicSpectrumPlan(pencil, explicit_mesh).evaluate()
    with pytest.raises(ValueError, match="not isolated"):
        PeriodicBandManifold(touching, [0], gap_tolerance=1.0e-10)

    manifold = PeriodicBandManifold(PeriodicSpectrumPlan(pencil, mesh).evaluate(), [0])
    zero_connection = PeriodicCrossKConnection(
        connectivity,
        np.zeros_like(np.asarray(connectivity.plan.source_indices))[:, None, None]
        * np.ones((1, 2, 2)),
        pencil.plan.basis.basis_id,
        "deliberately-singular-test",
    )
    with pytest.raises(ValueError, match="rank deficient"):
        PeriodicOverlapBundle(manifold, zero_connection)


def test_trivial_two_dimensional_chern_retains_plaquette_and_refinement_evidence():
    cell = PeriodicCell(np.eye(2))
    basis = PeriodicOrbitalBasisPlan(
        cell,
        ("lower", "upper"),
        [[0.0, 0.0], [0.0, 0.0]],
        ANGSTROM,
        PeriodicBlochGauge("lattice"),
    )
    family = periodic_translation_family_from_dense_blocks(
        [[0, 0]],
        np.diag([-1.0, 1.0]).reshape((1, 2, 1, 2, 1)),
    )
    pencil = PeriodicOrbitalPencilPlan.orthonormal(
        basis, family.plan, family.state, ELECTRONVOLT
    ).prepare()
    mesh = ReciprocalMeshPlan.monkhorst_pack(cell, (3, 3))
    spectrum = PeriodicSpectrumPlan(pencil, mesh).evaluate()
    connectivity = ReciprocalConnectivityPlan.regular(mesh).prepare()
    bundle = PeriodicOverlapBundle(
        PeriodicBandManifold(spectrum, [0]),
        identity_cross_k_connection(pencil, connectivity),
    )
    refinement = PeriodicChernRefinementEvidence((2, 2), (3, 3), 0.0, 0.0, 1.0e-10)
    chern = PeriodicChernPlan(
        bundle, refinement=refinement, require_refinement=True
    ).evaluate()

    np.testing.assert_allclose(chern.plaquette_phases, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(chern.raw_chern, 0.0, atol=1.0e-13)
    assert chern.mesh_shape == (3, 3)
    assert bool(chern.successful)

    failed = PeriodicChernRefinementEvidence((2, 2), (3, 3), 0.0, 0.2, 1.0e-3)
    with pytest.raises(ValueError, match="refinement"):
        PeriodicChernPlan(bundle, refinement=failed, require_refinement=True)
