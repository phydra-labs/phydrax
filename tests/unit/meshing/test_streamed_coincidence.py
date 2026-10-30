import numpy as np
import pytest

from phydrax.meshing._audit_topology import _coincident_representatives


@pytest.mark.parametrize(
    ("points", "tolerance", "expected", "coincident_count"),
    [
        pytest.param(
            np.asarray(
                ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (3.0, 0.0), (3.0, 0.0), (10.0, 0.0)),
                dtype=np.float64,
            ),
            0.6,
            np.asarray((0, 0, 0, 3, 3, 5), dtype=np.int64),
            5,
            id="transitive-components-and-isolated-vertex",
        ),
        pytest.param(
            np.concatenate(
                (
                    np.zeros((128, 3), dtype=np.float64),
                    np.full((128, 3), 10.0, dtype=np.float64),
                )
            ),
            0.0,
            np.concatenate(
                (np.zeros(128, dtype=np.int64), np.full(128, 128, dtype=np.int64))
            ),
            256,
            id="complete-dense-coincidence-components",
        ),
        pytest.param(
            np.asarray(((0.0,), (2.0,), (4.0,)), dtype=np.float64),
            0.0,
            np.asarray((0, 1, 2), dtype=np.int64),
            0,
            id="separate-source-vertices",
        ),
    ],
)
def test_streamed_coincidence_preserves_complete_topology_and_source(
    points: np.ndarray,
    tolerance: float,
    expected: np.ndarray,
    coincident_count: int,
) -> None:
    original = points.copy()
    representatives, count = _coincident_representatives(points, tolerance)
    np.testing.assert_array_equal(representatives, expected)
    np.testing.assert_array_equal(points, original)
    assert count == coincident_count
