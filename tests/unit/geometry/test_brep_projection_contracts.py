"""Projection result identity, fiber evidence, and validation ordering."""

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.geometry.brep import BRepProjectionResult, BRepProjectionStatus


def _fiber_result(
    source_revision: str,
    *,
    point: tuple[float, float, float] = (np.nan, np.nan, np.nan),
    residual: float = 1.0,
    occurrence_paths: tuple[tuple[str, ...], ...] = (),
) -> BRepProjectionResult:
    return BRepProjectionResult(
        source_revision,
        np.asarray((2,), dtype=np.int8),
        np.asarray((0,), dtype=np.int32),
        np.zeros((1, 3), dtype=np.float64),
        np.asarray((point,), dtype=np.float64),
        np.asarray(((0.0, np.nan),), dtype=np.float64),
        np.asarray((residual,), dtype=np.float64),
        np.asarray((BRepProjectionStatus.NON_ISOLATED,), dtype=np.int8),
        np.full((1, 3), np.nan, dtype=np.float64),
        np.full((1, 2, 3), np.nan, dtype=np.float64),
        occurrence_paths=occurrence_paths,
    )


def test_projection_fiber_retains_identity_without_selecting_a_point() -> None:
    result = _fiber_result("source")

    assert not np.asarray(result.resolved)[0]
    assert result.entity_ids() == ("source:face:0",)
    assert result.source_occurrence_paths == ((),)
    assert np.asarray(result.residuals)[0] == 1.0
    assert np.all(np.isnan(np.asarray(result.points)))
    assert np.any(np.isnan(np.asarray(result.parameters)))


def test_projection_revision_normalization_preserves_result_identity() -> None:
    normalized = _fiber_result("source")
    spaced = _fiber_result(" source ")

    assert spaced.source_revision == normalized.source_revision
    assert spaced.result_id == normalized.result_id


@pytest.mark.parametrize("residual", (-1.0, np.inf), ids=("negative", "nonfinite"))
def test_projection_fiber_refuses_invalid_distance(residual: float) -> None:
    with pytest.raises(ValueError, match="BRepProjectionResult.NON_ISOLATED"):
        _fiber_result("source", residual=residual)


def test_projection_fiber_refuses_a_selected_point() -> None:
    with pytest.raises(ValueError, match="BRepProjectionResult.NON_ISOLATED"):
        _fiber_result("source", point=(0.0, 0.0, 0.0))


def test_projection_occurrence_table_validation_precedes_revision_validation() -> None:
    with pytest.raises(ValueError, match="BRepProjectionResult.occurrence_paths"):
        _fiber_result("", occurrence_paths=(("part",), ("part",)))


def test_projection_host_banks_preserve_bytes_and_identity_without_eager_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    banks = {
        "dimensions": np.asarray((2, -1), dtype=np.int8),
        "indices": np.asarray((3, -1), dtype=np.int32),
        "queries": np.asarray(((0.0, -0.0, 0.25), (1.0, 2.0, 3.0)), dtype=np.float64),
        "points": np.asarray(
            ((0.0, -0.0, 0.5), (np.nan, np.nan, np.nan)), dtype=np.float64
        ),
        "parameters": np.asarray(((0.25, 0.75), (np.nan, np.nan)), dtype=np.float64),
        "residuals": np.asarray((0.25, np.inf), dtype=np.float64),
        "status": np.asarray(
            (BRepProjectionStatus.UNIQUE, BRepProjectionStatus.FAILED), dtype=np.int8
        ),
        "normals": np.asarray(
            ((0.0, 0.0, 1.0), (np.nan, np.nan, np.nan)), dtype=np.float64
        ),
        "tangents": np.asarray(
            (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), ((np.nan,) * 3, (np.nan,) * 3)),
            dtype=np.float64,
        ),
    }
    occurrences = np.asarray((0, -1), dtype=np.int32)
    paths = (("part",),)
    expected_id = canonical_fingerprint(
        {
            "kind": "brep-projection-result",
            "source_revision": "source",
            **{
                name: array_tree_fingerprint(banks[name])
                for name in (
                    "dimensions",
                    "indices",
                    "queries",
                    "points",
                    "parameters",
                    "status",
                )
            },
            "occurrence_indices": array_tree_fingerprint(occurrences),
            "occurrence_paths": [list(path) for path in paths],
        }
    )
    with monkeypatch.context() as host_only:
        host_only.setattr(jnp, "asarray", None)
        result = BRepProjectionResult(
            "source",
            banks["dimensions"],
            banks["indices"],
            banks["queries"],
            banks["points"],
            banks["parameters"],
            banks["residuals"],
            banks["status"],
            banks["normals"],
            banks["tangents"],
            occurrence_indices=occurrences,
            occurrence_paths=paths,
        )

    for name, bank in banks.items():
        actual = np.asarray(getattr(result, name))
        assert actual.dtype == bank.dtype
        assert actual.shape == bank.shape
        assert actual.tobytes() == bank.tobytes()
    assert np.asarray(result.occurrence_indices).tobytes() == occurrences.tobytes()
    assert result.result_id == expected_id
    assert result.entity_ids() == ('source:face:3:occurrence:["part"]', "")
    assert result.source_occurrence_paths == (("part",), ())
