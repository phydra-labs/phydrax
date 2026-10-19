"""White-box admission contracts for rational surface insertion batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from phydrax.meshing._surface_generation import RestrictionBatch


@dataclass
class _Restrictions:
    required: np.ndarray
    edges: np.ndarray
    parameters: np.ndarray
    insert_edges: np.ndarray


def _restrictions() -> _Restrictions:
    return _Restrictions(
        np.asarray((True, False), dtype=np.bool_),
        np.asarray(((0, 1), (-1, -1)), dtype=np.int64),
        np.asarray(((1, 2), (0, 0)), dtype=np.int64),
        np.asarray(((0, 1), (-1, -1)), dtype=np.int32),
    )


def _batch(restrictions: _Restrictions) -> RestrictionBatch:
    return RestrictionBatch(
        np.asarray(((0.5, 0.0), (0.2, 0.2)), dtype=np.float64),
        np.zeros((2,), dtype=np.float64),
        np.asarray((0, 1), dtype=np.int32),
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        np.asarray((True, False), dtype=np.bool_),
        np.asarray((True, True), dtype=np.bool_),
        restrictions.insert_edges,
        restrictions.required,
        restrictions.edges,
        restrictions.parameters,
    )


def test_restriction_batch_retains_rational_roots_and_execution_edge_identity() -> None:
    restrictions = _restrictions()
    batch = _batch(restrictions)
    assert batch.required is restrictions.required
    assert batch.edges is restrictions.edges
    assert batch.parameters is restrictions.parameters
    assert batch.insert_edges is restrictions.insert_edges
    np.testing.assert_array_equal(
        batch.edges[batch.required], batch.insert_edges[batch.required]
    )
    np.testing.assert_array_equal(batch.parameters[batch.required], ((1, 2),))


@pytest.mark.parametrize(
    ("field", "values"),
    (
        ("required", np.asarray((1, 0), dtype=np.int32)),
        ("required", np.asarray(((True,), (False,)), dtype=np.bool_)),
        ("edges", np.asarray(((0, 1), (-1, -1)), dtype=np.int32)),
        ("edges", np.asarray(((0, 1, 2), (-1, -1, -1)), dtype=np.int64)),
        ("parameters", np.asarray(((1, 2), (0, 0)), dtype=np.int32)),
        ("parameters", np.asarray(((1, 2), (0, 0), (0, 0)), dtype=np.int64)),
    ),
    ids=(
        "required-dtype",
        "required-rank",
        "edge-dtype",
        "edge-width",
        "parameter-dtype",
        "parameter-count",
    ),
)
def test_restriction_batch_refuses_misaligned_rational_banks(
    field: str, values: np.ndarray
) -> None:
    restrictions = _restrictions()
    setattr(restrictions, field, values)
    with pytest.raises(ValueError, match="RestrictionBatch.restrictions"):
        _batch(restrictions)


def test_restriction_batch_refuses_a_root_on_a_different_execution_edge() -> None:
    restrictions = _restrictions()
    restrictions.insert_edges[0] = (1, 2)
    with pytest.raises(ValueError, match="RestrictionBatch.restrictions"):
        _batch(restrictions)


@pytest.mark.parametrize(
    ("field", "values"),
    (
        ("edges", np.asarray(((0, 1), (0, 1)), dtype=np.int64)),
        ("parameters", np.asarray(((1, 2), (1, 2)), dtype=np.int64)),
    ),
    ids=("unused-edge", "unused-parameter"),
)
def test_restriction_batch_refuses_roots_on_unrestricted_insertions(
    field: str, values: np.ndarray
) -> None:
    restrictions = _restrictions()
    setattr(restrictions, field, values)
    with pytest.raises(ValueError, match="RestrictionBatch.restrictions"):
        _batch(restrictions)
