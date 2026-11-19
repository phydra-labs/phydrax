"""Per-row static banks of a geometry association: one validated host leaf each."""

from __future__ import annotations

import pickle
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.meshing._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._model_codec import (
    MODEL_ARCHIVE_LIMITS,
    read_model_artifact,
    restore_model_archive_value,
    write_model_artifact,
)


_BANKS = (
    "source_entity_ids",
    "source_roles",
    "source_occurrence_paths",
    "source_region_sides",
    "source_parameter_intervals",
    "target_parameter_intervals",
    "target_reference_pieces",
    "source_parameter_points",
)


def _association() -> GeometryAssociation:
    return GeometryAssociation(
        GeometryAssociationKind.CURVE,
        "source",
        "revision",
        "edges",
        np.asarray((7, 9, 11)),
        ("curve-a", "curve-b", "curve-a"),
        np.asarray((0.0, 0.0, 0.0)),
        exact=True,
        source_roles=(GeometrySourceEntityRole.EDGE,) * 3,
        source_occurrence_paths=(("assembly", "a"), ("assembly", "b"), ("assembly", "a")),
        source_parameter_intervals=((Fraction(1, 3), Fraction(5, 3)), None, None),
        source_parameter_points=(None, Fraction(7, 11), Fraction(1, 2)),
        target_parameter_intervals=((Fraction(1, 4), Fraction(3, 4)), None, None),
    )


def test_row_banks_are_tuples_held_as_one_static_leaf() -> None:
    association = _association()
    for name in _BANKS:
        bank = getattr(association, name)
        plain = tuple(bank)
        assert isinstance(bank, tuple), name
        assert bank == plain and hash(bank) == hash(plain), name
        assert jax.tree_util.tree_leaves(bank) == [bank], name
        restored = pickle.loads(pickle.dumps(bank))
        assert restored == bank and type(restored) is type(bank), name
    assert association.source_entity_ids == ("curve-a", "curve-b", "curve-a")
    # Static fields stay out of the dynamic leaves.
    assert all(
        isinstance(leaf, jax.Array) for leaf in jax.tree_util.tree_leaves(association)
    )


def test_pickled_association_keeps_its_identity() -> None:
    association = _association()
    restored = pickle.loads(pickle.dumps(association))
    assert restored.association_id == association.association_id
    assert restored.source_occurrence_paths == association.source_occurrence_paths


def test_archived_association_restores_its_identity(tmp_path: Path) -> None:
    association = _association()
    path = write_model_artifact(
        tmp_path / "association.phx", association, limits=MODEL_ARCHIVE_LIMITS
    )
    restored = read_model_artifact(
        path,
        expected_type=GeometryAssociation,
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    assert type(restored) is GeometryAssociation
    assert restored.association_id == association.association_id
    for name in _BANKS:
        assert getattr(restored, name) == getattr(association, name), name


@pytest.mark.parametrize(
    "rows",
    (
        {"source_occurrence_paths": ((jnp.asarray(1),), ("b",), ("c",))},
        {"source_roles": (jnp.asarray(1), None, None)},
        {"source_parameter_points": (jnp.asarray(0.5), None, None)},
        {"source_region_sides": ((jnp.asarray(0), None), (None, None), (None, None))},
    ),
    ids=("occurrence-path", "role", "parameter-point", "region-side"),
)
def test_rows_holding_a_jax_array_are_refused(rows: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GeometryAssociation(
            GeometryAssociationKind.CURVE,
            "source",
            "revision",
            "edges",
            np.asarray((7, 9, 11)),
            ("a", "b", "c"),
            np.asarray((0.0, 0.0, 0.0)),
            resolved=np.zeros((3,), dtype=np.bool_),
            **rows,  # ty: ignore[invalid-argument-type]
        )
