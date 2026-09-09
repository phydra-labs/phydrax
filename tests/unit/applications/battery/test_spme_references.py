#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

from tools._battery_spme_references import (
    conservative_projection,
    FARADAY,
    paper_reference,
    SyntheticSpmeData,
)


def test_independent_paper_solver_conserves_species_and_exact_current_integrals():
    times = (0.0, 0.1, 0.2, 0.3)
    result = paper_reference(
        times, times, (0.2, 0.0, -0.1), radial_cells=8, region_cells=(4, 3, 4)
    )
    negative = result["negative_amount_mol"].sum(axis=-1)
    positive = result["positive_amount_mol"].sum(axis=-1)
    electrolyte = result["electrolyte_amount_mol"].sum(axis=-1)
    charge = np.asarray((0.0, 0.02, 0.02, 0.01))
    np.testing.assert_allclose(
        FARADAY * (negative - negative[0]), charge, atol=2e-10, rtol=1e-8
    )
    np.testing.assert_allclose(
        FARADAY * (positive - positive[0]), -charge, atol=2e-10, rtol=1e-8
    )
    np.testing.assert_allclose(electrolyte, electrolyte[0], atol=1e-14, rtol=1e-12)
    np.testing.assert_allclose(
        negative + positive + electrolyte,
        negative[0] + positive[0] + electrolyte[0],
        atol=1e-13,
    )
    # Boundaries are observed from the left, while subsequent dynamics use right holds.
    np.testing.assert_array_equal(result["current_a"], (0.2, 0.2, 0.0, -0.1))


def test_independent_equilibrium_has_no_voltage_or_inventory_drift():
    data = SyntheticSpmeData()
    result = paper_reference(
        (0.0, 0.5, 1.0),
        (0.0, 1.0),
        (0.0,),
        data=data,
        radial_cells=8,
        region_cells=(4, 3, 4),
    )
    np.testing.assert_allclose(result["voltage_v"], data.ocp[1] - data.ocp[0], atol=1e-12)
    for field in ("negative_amount_mol", "positive_amount_mol", "electrolyte_amount_mol"):
        np.testing.assert_allclose(
            result[field],
            np.broadcast_to(result[field][0], result[field].shape),
            atol=1e-14,
        )


@pytest.mark.parametrize("spherical", (False, True))
def test_nonuniform_overlap_projection_preserves_extensive_inventory(spherical):
    source = np.asarray((0.0, 0.1, 0.4, 1.0))
    target = np.asarray((0.0, 0.3, 0.7, 1.0))
    values = np.asarray(((2.0, 4.0, 1.0), (3.0, 2.0, 5.0)))
    projected = conservative_projection(source, target, values, spherical=spherical)
    power = 3 if spherical else 1
    np.testing.assert_allclose(
        projected @ np.diff(target**power), values @ np.diff(source**power), atol=1e-14
    )
    with pytest.raises(ValueError):
        conservative_projection(source, (0.0, 0.5, 0.9), values, spherical=spherical)


def test_reference_refuses_unsupported_current_instead_of_clipping():
    with pytest.raises(ValueError):
        paper_reference((0.0, 1.0), (0.0, 1.0), (0.6,))
