import numpy as np

from phydrax.chemistry.periodic._observables import (
    fermi_surface_evidence,
    PeriodicDensityOfStatesPlan,
    PeriodicProjectedDOSPlan,
    PeriodicProjectorGroups,
    PeriodicVelocityPlan,
)
from phydrax.chemistry.periodic._orbital_model import (
    PeriodicBlochGauge,
    PeriodicOrbitalBasisPlan,
    PeriodicOrbitalPencilPlan,
)
from phydrax.chemistry.periodic._spectrum import (
    ChebyshevMomentPlan,
    PeriodicSpectrumPlan,
)
from phydrax.discretization import PeriodicCell, ReciprocalMeshPlan
from phydrax.operators.periodic import periodic_translation_family_from_dense_blocks
from phydrax.units import ANGSTROM, conversion_factor, ELECTRONVOLT, JOULE, METER


HBAR = 1.054_571_817e-34


def _scalar_chain(*, overlap=0.0):
    cell = PeriodicCell([[2.0]])
    basis = PeriodicOrbitalBasisPlan(
        cell,
        ("s",),
        [[0.0]],
        ANGSTROM,
        PeriodicBlochGauge("lattice"),
    )
    h = periodic_translation_family_from_dense_blocks(
        [[-1], [0], [1]],
        np.asarray([[[[[-0.25]]]], [[[[0.2]]]], [[[[-0.25]]]]]),
    )
    s = periodic_translation_family_from_dense_blocks(
        [[-1], [0], [1]],
        np.asarray([[[[[overlap]]]], [[[[1.0]]]], [[[[overlap]]]]]),
    )
    pencil = PeriodicOrbitalPencilPlan(
        basis,
        h.plan,
        h.state,
        s.plan,
        s.state,
        ELECTRONVOLT,
    ).prepare()
    mesh = ReciprocalMeshPlan.monkhorst_pack(cell, (16,))
    spectrum = PeriodicSpectrumPlan(pencil, mesh).evaluate()
    return pencil, mesh, spectrum


def test_generalized_velocity_contains_dS_and_is_physical_cartesian():
    pencil, mesh, spectrum = _scalar_chain(overlap=0.1)
    velocity = PeriodicVelocityPlan(pencil, spectrum).evaluate()
    q = np.asarray(mesh.fractional_points[:, 0])
    h = 0.2 - 0.5 * np.cos(2.0 * np.pi * q)
    s = 1.0 + 0.2 * np.cos(2.0 * np.pi * q)
    dh = np.pi * np.sin(2.0 * np.pi * q)
    ds = -0.4 * np.pi * np.sin(2.0 * np.pi * q)
    reciprocal_right_inverse_angstrom = 1.0 / np.pi
    scale = (
        float(conversion_factor(ELECTRONVOLT, JOULE))
        * float(conversion_factor(ANGSTROM, METER))
        / HBAR
    )
    expected = (dh - (h / s) * ds) / s * reciprocal_right_inverse_angstrom * scale

    np.testing.assert_allclose(velocity.band_velocities[:, 0, 0], expected, rtol=2.0e-6)
    np.testing.assert_allclose(
        velocity.velocity_matrices[:, 0, 0, 0], expected, rtol=2.0e-6
    )
    assert velocity.derivative_basis == "cartesian-wavevector-m-per-s"
    assert bool(velocity.successful)


def test_dos_and_metric_grouped_pdos_preserve_state_partition():
    cell = PeriodicCell([[3.0]])
    basis = PeriodicOrbitalBasisPlan(
        cell,
        ("left", "right"),
        [[0.0], [0.4]],
        ANGSTROM,
        PeriodicBlochGauge("atomic"),
    )
    h = periodic_translation_family_from_dense_blocks(
        [[0]],
        np.asarray([[[[[-1.0]], [[0.2]]], [[[0.2]], [[1.0]]]]]).reshape(1, 2, 1, 2, 1),
    )
    pencil = PeriodicOrbitalPencilPlan.orthonormal(
        basis, h.plan, h.state, ELECTRONVOLT
    ).prepare()
    mesh = ReciprocalMeshPlan.monkhorst_pack(cell, (4,))
    spectrum = PeriodicSpectrumPlan(pencil, mesh).evaluate()
    grid = np.linspace(-3.0, 3.0, 4001)
    dos = PeriodicDensityOfStatesPlan(spectrum, grid, 0.08).evaluate()
    groups = PeriodicProjectorGroups(("left", "right"), [0, 1], basis.basis_id)
    pdos = PeriodicProjectedDOSPlan(pencil, spectrum, groups, grid, 0.08).evaluate()

    np.testing.assert_allclose(
        np.trapezoid(np.asarray(dos.density), grid), 2.0, atol=2.0e-5
    )
    np.testing.assert_allclose(pdos.total_density, dos.density, atol=2.0e-12)
    np.testing.assert_allclose(
        np.sum(pdos.band_group_weights, axis=-1), 1.0, atol=2.0e-12
    )
    assert pdos.metric_route == "generalized-mulliken-partition"
    assert bool(pdos.successful)


def test_fermi_surface_reports_unresolved_corner_without_silent_interpolation():
    _, mesh, spectrum = _scalar_chain()
    resolved = fermi_surface_evidence(mesh, spectrum, 0.2)
    corner_energy = float(spectrum.energies[0, 0])
    unresolved = fermi_surface_evidence(mesh, spectrum, corner_energy)

    assert bool(resolved.complete)
    assert not bool(unresolved.complete)
    assert np.any(np.asarray(unresolved.unresolved_mask))


def test_bounded_chebyshev_candidate_uses_sparse_orthonormal_recurrence():
    pencil, mesh, _ = _scalar_chain()
    moments = ChebyshevMomentPlan(
        pencil,
        mesh,
        4,
        -0.3,
        0.7,
        maximum_operator_applications=1_000,
    ).evaluate()

    np.testing.assert_allclose(moments.moments[:3], [1.0, 0.0, 0.0], atol=2.0e-12)
    assert bool(moments.successful)
