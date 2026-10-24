#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.contact._cone import ContactConeBranch
from phydrax.applications.contact._manifold import (
    ContactFrameTransport,
    ContactManifoldObservation,
    ContactManifoldPolicy,
    ContactManifoldState,
    record_contact_response,
    update_contact_manifold,
)


_CATALOGUE = "manifold-test-catalogue"


def _frames(count: int) -> tuple[np.ndarray, np.ndarray]:
    normal = np.tile(np.asarray((0.0, 0.0, 1.0)), (count, 1))
    basis = np.tile(np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))), (count, 1, 1))
    return normal, basis


def _observation(
    keys: tuple[int, ...],
    *,
    normal: Any | None = None,
    basis: Any | None = None,
    valid: Any | None = None,
    material: Any = 0,
    geometry: int = 0,
    catalogue: str = _CATALOGUE,
) -> ContactManifoldObservation:
    count = len(keys)
    default_normal, default_basis = _frames(count)
    witness = np.zeros((count, 2, 3))
    witness[:, 0, 0] = np.asarray(keys, dtype=np.float64)
    return ContactManifoldObservation(
        np.asarray(keys, dtype=np.int64),
        np.ones((count,), dtype=np.bool_) if valid is None else np.asarray(valid),
        default_normal if normal is None else np.asarray(normal),
        default_basis if basis is None else np.asarray(basis),
        witness,
        material,
        geometry,
        catalogue_id=catalogue,
    )


def _recorded(
    state: ContactManifoldState, keys: tuple[int, ...], impulses: Any
) -> ContactManifoldState:
    count = len(keys)
    return record_contact_response(
        state,
        np.asarray(keys, dtype=np.int64),
        np.asarray(impulses, dtype=np.float64),
        np.full((count,), int(ContactConeBranch.STICKING), dtype=np.int8),
        np.zeros((count, 2)),
        step_size=0.1,
    )


def _occupied_keys(state: ContactManifoldState) -> np.ndarray:
    return np.asarray(state.route_keys)[np.asarray(state.occupied)]


def test_birth_continuation_retention_death_and_resurrection() -> None:
    policy = ContactManifoldPolicy(retention_steps=2)
    state = ContactManifoldState.empty(4, catalogue_id=_CATALOGUE)
    born = update_contact_manifold(state, _observation((20, 10)), policy=policy)
    stored = _recorded(born.commit(), (10, 20), ((1.0, 0.2, 0.0), (2.0, 0.0, 0.1)))
    continued = update_contact_manifold(stored, _observation((10, 30)), policy=policy)
    retained = continued.commit()
    gone_once = update_contact_manifold(retained, _observation((10,)), policy=policy)
    resurrected = update_contact_manifold(
        gone_once.commit(), _observation((10, 30)), policy=policy
    )

    assert bool(born.successful) and int(born.born_count) == 2
    np.testing.assert_array_equal(_occupied_keys(born.commit()), (10, 20))
    np.testing.assert_allclose(continued.warm_start[0], (1.0, 0.2, 0.0))
    np.testing.assert_allclose(continued.warm_start[1], 0.0)
    np.testing.assert_array_equal(_occupied_keys(retained), (10, 20, 30))
    assert not bool(np.asarray(retained.active)[list(_occupied_keys(retained)).index(20)])
    assert 20 in np.asarray(gone_once.died_keys).tolist()
    assert int(resurrected.resurrected_count) == 1
    assert 30 in np.asarray(resurrected.resurrected_keys).tolist()
    assert int(resurrected.candidate.transaction_revision) == int(
        gone_once.candidate.transaction_revision
    ) + 1


def test_observation_permutation_gives_identical_history() -> None:
    policy = ContactManifoldPolicy()
    state = _recorded(
        update_contact_manifold(
            ContactManifoldState.empty(4, catalogue_id=_CATALOGUE),
            _observation((5, 7, 9)),
            policy=policy,
        ).commit(),
        (5, 7, 9),
        ((1.0, 0.0, 0.0), (2.0, 0.1, 0.0), (3.0, 0.0, 0.2)),
    )
    forward = update_contact_manifold(state, _observation((5, 7, 9)), policy=policy)
    reverse = update_contact_manifold(state, _observation((9, 7, 5)), policy=policy)

    for name in ("route_keys", "occupied", "active", "impulse", "witness", "age"):
        np.testing.assert_array_equal(
            getattr(forward.candidate, name), getattr(reverse.candidate, name)
        )
    np.testing.assert_allclose(reverse.warm_start[0], forward.warm_start[2])


def test_overflow_duplicates_stale_geometry_and_catalogue_refuse() -> None:
    policy = ContactManifoldPolicy()
    state = ContactManifoldState.empty(2, catalogue_id=_CATALOGUE)
    overflow = update_contact_manifold(state, _observation((1, 2, 3)), policy=policy)
    duplicate = update_contact_manifold(state, _observation((4, 4)), policy=policy)
    advanced = update_contact_manifold(
        state, _observation((1,), geometry=3), policy=policy
    ).commit()
    stale = update_contact_manifold(advanced, _observation((1,), geometry=2), policy=policy)

    assert bool(overflow.active_overflow) and not bool(overflow.successful)
    assert int(duplicate.duplicate_count) == 2 and not bool(duplicate.successful)
    assert bool(stale.stale_geometry) and not bool(stale.successful)
    assert overflow.rollback() is state
    with pytest.raises(ValueError, match="not transferable across catalogues"):
        update_contact_manifold(
            state, _observation((1,), catalogue="other-catalogue"), policy=policy
        )


def test_material_revision_change_resets_warm_start() -> None:
    policy = ContactManifoldPolicy()
    state = _recorded(
        update_contact_manifold(
            ContactManifoldState.empty(2, catalogue_id=_CATALOGUE),
            _observation((1,)),
            policy=policy,
        ).commit(),
        (1,),
        ((1.0, 0.5, 0.0),),
    )
    changed = update_contact_manifold(state, _observation((1,), material=4), policy=policy)

    assert int(changed.material_changed_count) == 1
    np.testing.assert_array_equal(changed.warm_start, 0.0)
    assert int(changed.candidate.branch[0]) == int(ContactConeBranch.UNCLASSIFIED)


def _rotation_about_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)))


def _rotation_about_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))


def _world(normal: np.ndarray, basis: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    return normal * impulse[0] + basis @ impulse[1:]


def test_normal_parallel_transport_is_basis_independent_and_objective() -> None:
    policy = ContactManifoldPolicy(frame_transport=ContactFrameTransport.NORMAL_PARALLEL)
    impulse = np.asarray((1.0, 0.3, -0.2))
    normal, basis = _frames(1)
    state = _recorded(
        update_contact_manifold(
            ContactManifoldState.empty(2, catalogue_id=_CATALOGUE),
            _observation((1,)),
            policy=policy,
        ).commit(),
        (1,),
        (impulse,),
    )
    old_world = _world(normal[0], basis[0], impulse)

    twisted_basis = _rotation_about_z(0.7) @ basis[0]
    twisted = update_contact_manifold(
        state, _observation((1,), basis=twisted_basis[None]), policy=policy
    )
    np.testing.assert_allclose(
        _world(normal[0], twisted_basis, np.asarray(twisted.warm_start[0])),
        old_world,
        atol=1.0e-12,
    )

    rotation = _rotation_about_x(0.4)
    rotated = update_contact_manifold(
        state,
        _observation(
            (1,),
            normal=(rotation @ normal[0])[None],
            basis=(rotation @ basis[0])[None],
        ),
        policy=policy,
    )
    np.testing.assert_allclose(rotated.warm_start[0], impulse, atol=1.0e-12)

    flipped = update_contact_manifold(
        state, _observation((1,), normal=-normal), policy=policy
    )
    assert int(flipped.frame_reset_count) == 1
    np.testing.assert_array_equal(flipped.warm_start, 0.0)


def test_material_frame_transport_preserves_local_components() -> None:
    policy = ContactManifoldPolicy(frame_transport=ContactFrameTransport.MATERIAL_FRAME)
    impulse = np.asarray((1.0, -0.3, 0.1))
    normal, basis = _frames(1)
    state = _recorded(
        update_contact_manifold(
            ContactManifoldState.empty(2, catalogue_id=_CATALOGUE),
            _observation((1,)),
            policy=policy,
        ).commit(),
        (1,),
        (impulse,),
    )
    rotation = _rotation_about_z(0.9)
    moved = update_contact_manifold(
        state, _observation((1,), basis=(rotation @ basis[0])[None]), policy=policy
    )

    np.testing.assert_allclose(moved.warm_start[0], impulse)
    np.testing.assert_allclose(
        _world(normal[0], rotation @ basis[0], np.asarray(moved.warm_start[0])),
        rotation @ _world(normal[0], basis[0], impulse),
        atol=1.0e-12,
    )


def test_update_and_record_are_jit_compatible() -> None:
    policy = ContactManifoldPolicy()
    state = ContactManifoldState.empty(3, catalogue_id=_CATALOGUE)
    observation = _observation((3, 1))

    def step(value: ContactManifoldState) -> ContactManifoldState:
        candidate = update_contact_manifold(value, observation, policy=policy).commit()
        return record_contact_response(
            candidate,
            observation.route_keys,
            jnp.ones((2, 3)),
            jnp.full((2,), int(ContactConeBranch.SLIDING), dtype=jnp.int8),
            jnp.ones((2, 2)),
            step_size=0.5,
        )

    eager = step(state)
    compiled = eqx.filter_jit(step)(state)

    for name in ("route_keys", "impulse", "slip", "branch", "transaction_revision"):
        np.testing.assert_array_equal(getattr(eager, name), getattr(compiled, name))
    np.testing.assert_allclose(eager.slip[np.asarray(eager.occupied)], 0.5)
