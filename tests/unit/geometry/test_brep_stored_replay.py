"""Stored B-Rep replay retains declared source state and rejects corrupt evidence."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.geometry.brep import (
    brep_sphere,
    BRepTessellationPolicy,
    prepare_brep_query,
    PreparedBRepQuery,
)
from phydrax.lifecycle._meshing_sources import validate_meshing_source_closure


@pytest.fixture(scope="module")
def stored_query() -> PreparedBRepQuery:
    model = brep_sphere(
        1.0,
        coordinate_contract=SpatialCoordinateContract.si(),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    return prepare_brep_query(model, certify_on_host=False)


def test_query_stored_replay_retains_device_certification_role(
    stored_query: PreparedBRepQuery,
) -> None:
    rebuilt = PreparedBRepQuery.from_stored_fields(stored_query)
    assert rebuilt.certify_on_host is False
    assert rebuilt.query_id == stored_query.query_id
    assert rebuilt.model_id == stored_query.model_id
    validate_meshing_source_closure(rebuilt)


def test_query_stored_replay_rejects_changed_declared_identity(
    stored_query: PreparedBRepQuery,
) -> None:
    different_model = brep_sphere(
        2.0,
        coordinate_contract=stored_query.model.coordinate_contract,
        tessellation=BRepTessellationPolicy(realize=False),
    )
    corrupted = eqx.tree_at(
        lambda value: value.model,
        stored_query,
        different_model,
    )
    with pytest.raises(ValueError, match=r"PreparedBRepQuery\.stored_fields"):
        PreparedBRepQuery.from_stored_fields(corrupted)


def test_registered_query_replay_rejects_changed_derived_geometry(
    stored_query: PreparedBRepQuery,
) -> None:
    corrupted = eqx.tree_at(
        lambda value: value.face_boxes,
        stored_query,
        stored_query.face_boxes + jnp.asarray(0.125, dtype=jnp.float64),
    )
    # The dyadic perturbation changes derived geometry while leaving
    # declared query IDs intact; registered replay must compare stored arrays.
    with pytest.raises(ValueError, match="source"):
        validate_meshing_source_closure(corrupted)
