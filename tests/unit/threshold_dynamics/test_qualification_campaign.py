#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np

import phydrax.threshold_dynamics as td
from tools.threshold_dynamics_qualification import (
    _foam_cell_cost,
    _icosphere,
    _kelvin_bcc_seeds,
    _mesh_cosine,
    _periodic_seed_evidence,
    _sparse_growth,
    _surface_p1_operators,
    _weaire_phelan_a15_seeds,
    _zero_width_cost,
)


def test_scaled_grain_campaign_admits_every_required_candidate() -> None:
    # Same grain density, kernel width, capacity, and brick size as the 512^2
    # campaign. This exercises admission rather than weakening overflow safety.
    _, result, _, _ = _sparse_growth(7, 2, 188, 1, 32, 12, 4, 3.0)
    sparse = result.evidence.sparse

    assert sparse is not None
    assert int(result.status) == int(td.ThresholdDynamicsStatus.SUCCESS)
    assert not bool(np.any(np.asarray(sparse.candidate_overflow)))
    assert int(np.max(np.asarray(sparse.required_candidates))) <= 32


def test_periodic_foam_seeds_have_physical_topology_and_cost_counting() -> None:
    kelvin = _periodic_seed_evidence(_kelvin_bcc_seeds(), (14,) * 16)
    phelan = _periodic_seed_evidence(
        _weaire_phelan_a15_seeds(), (12, 12, 14, 14, 14, 14, 14, 14)
    )

    np.testing.assert_allclose(kelvin["region_volumes"], 1.0 / 16.0, atol=2.0e-12)
    np.testing.assert_allclose(kelvin["total_volume"], 1.0, atol=2.0e-12)
    np.testing.assert_allclose(phelan["total_volume"], 1.0, atol=2.0e-12)
    assert phelan["face_counts"] == [12, 12, 14, 14, 14, 14, 14, 14]

    cells = 16
    cell_volume = 1.0 / cells
    expected_cost = 5.3
    once_counted_film_area = 0.5 * cells * expected_cost * cell_volume ** (2.0 / 3.0)
    np.testing.assert_allclose(
        _foam_cell_cost([once_counted_film_area], cells), expected_cost
    )
    widths = [0.04, 0.06, 0.08]
    finite_width = [expected_cost - 2.0 * width for width in widths]
    extrapolated, fit_error = _zero_width_cost(widths, finite_width)
    np.testing.assert_allclose(extrapolated, expected_cost)
    np.testing.assert_allclose(fit_error, 0.0, atol=1.0e-14)


def test_mesh_cap_angle_uses_its_polyhedral_area_measure() -> None:
    vertices, faces = _icosphere(1)
    mass, _ = _surface_p1_operators(vertices, faces)
    lumped = np.asarray(mass.sum(axis=1)).reshape(-1)

    np.testing.assert_allclose(_mesh_cosine(lumped, np.zeros(len(vertices))), -1.0)
    np.testing.assert_allclose(_mesh_cosine(lumped, np.ones(len(vertices))), 1.0)
    labels = np.where(vertices[:, 2] > 0.0, 0, 1)
    cosine = _mesh_cosine(lumped, labels)
    complement = _mesh_cosine(lumped, 1 - labels)
    np.testing.assert_allclose(cosine + complement, 0.0, atol=2.0e-15)
