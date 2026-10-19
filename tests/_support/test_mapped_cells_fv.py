"""Independent polynomial references for the shared mapped FV fixture."""

import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax.discretization import UnstructuredFiniteVolumePlan
from tests._support import mapped_cells


# A degree-two polynomial map integrated on a unit cell has no truncation error.
# The existing mapped FV qualification admits 2e-14 absolute round-off at this scale.
MAPPED_VOLUME_ROUNDOFF = 2.0e-14


def test_fv_volume_map_preserves_inputs_and_polynomial_values() -> None:
    points = np.asarray(((0.2, 0.5, 0.7), (0.0, 1.0, 0.0)), dtype=np.float64)
    before = points.copy()
    mapped = mapped_cells.fv_volume_map(points)
    np.testing.assert_array_equal(points, before)
    np.testing.assert_allclose(
        mapped,
        ((0.2, 0.5195, 0.7), (0.0, 1.05, 0.0)),
        rtol=0.0,
        atol=MAPPED_VOLUME_ROUNDOFF,
    )
    assert mapped.dtype == np.dtype(np.float64)


def test_curved_fv_volume_retains_identity_and_integrated_density() -> None:
    mesh, geometry = mapped_cells.curved_fv_volume("hexahedron")
    prepared = UnstructuredFiniteVolumePlan.from_cell_mesh(mesh).prepare(
        cell_geometry=geometry
    )
    np.testing.assert_array_equal(prepared.cell_global_ids, (17,))
    # Integral over [0,1]^3 of det(D map) = integral(1 + 0.1*y) = 1.05.
    np.testing.assert_allclose(
        prepared.cell_volumes, (1.05,), rtol=0.0, atol=MAPPED_VOLUME_ROUNDOFF
    )
    np.testing.assert_array_equal(
        mesh.coordinates,
        mapped_cells.fv_volume_map(
            np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (1.0, 0.0, 1.0),
                    (1.0, 1.0, 1.0),
                    (0.0, 1.0, 1.0),
                ),
                dtype=np.float64,
            )
        ),
    )


def test_mapped_volume_reference_detects_erased_polynomial_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def identity_map(points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(points, dtype=np.float64).copy()

    monkeypatch.setattr(mapped_cells, "fv_volume_map", identity_map)
    mesh, geometry = mapped_cells.curved_fv_volume("hexahedron")
    prepared = UnstructuredFiniteVolumePlan.from_cell_mesh(mesh).prepare(
        cell_geometry=geometry
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            prepared.cell_volumes, (1.05,), rtol=0.0, atol=MAPPED_VOLUME_ROUNDOFF
        )
