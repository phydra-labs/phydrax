import numpy as np
import pytest

from phydrax.geometry import PlanarEmbedding


@pytest.mark.parametrize(
    "origin",
    (
        pytest.param((0.0, 0.0), id="wrong-extent"),
        pytest.param((0.0, np.inf, 0.0), id="nonfinite-origin"),
    ),
)
def test_planar_embedding_refuses_invalid_origin(origin: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="PlanarEmbedding.origin"):
        PlanarEmbedding(origin, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@pytest.mark.parametrize(
    "y_axis,normal",
    (
        pytest.param((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), id="dependent-axes"),
        pytest.param((0.0, 1.0, 0.0), (0.0, 0.0, -1.0), id="left-handed-frame"),
    ),
)
def test_planar_embedding_refuses_invalid_frame(
    y_axis: tuple[float, float, float], normal: tuple[float, float, float]
) -> None:
    with pytest.raises(ValueError, match="PlanarEmbedding.axes"):
        PlanarEmbedding((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), y_axis, normal)


@pytest.mark.parametrize(
    "points",
    (
        pytest.param(np.zeros((2, 2), dtype=np.float64), id="wrong-world-extent"),
        pytest.param(
            np.asarray(((0.0, np.nan, 0.0),), dtype=np.float64),
            id="nonfinite-world-point",
        ),
    ),
)
def test_planar_embedding_world_queries_refuse_invalid_points(points: np.ndarray) -> None:
    embedding = PlanarEmbedding(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    )
    with pytest.raises(ValueError, match="PlanarEmbedding.points"):
        embedding.to_planar(points)
    with pytest.raises(ValueError, match="PlanarEmbedding.points"):
        embedding.plane_residual(points)


def test_planar_embedding_projects_off_plane_points_and_reports_signed_residual() -> None:
    embedding = PlanarEmbedding(
        (3.0, -2.0, 5.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
    )
    world = np.asarray(((7.0, -1.0, 7.0), (-1.0, -3.0, 4.0)), dtype=np.float64)
    np.testing.assert_array_equal(embedding.to_planar(world), ((1.0, 2.0), (-1.0, -1.0)))
    np.testing.assert_array_equal(embedding.plane_residual(world), (4.0, -4.0))
    np.testing.assert_array_equal(
        embedding.to_world(embedding.to_planar(world)),
        ((3.0, -1.0, 7.0), (3.0, -3.0, 4.0)),
    )


def test_planar_embedding_identity_is_reproducible_for_equal_declared_frames() -> None:
    frame = ((3.0, -2.0, 5.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    assert PlanarEmbedding(*frame).embedding_id == PlanarEmbedding(*frame).embedding_id
    displaced = PlanarEmbedding((4.0, -2.0, 5.0), *frame[1:])
    assert displaced.embedding_id != PlanarEmbedding(*frame).embedding_id
