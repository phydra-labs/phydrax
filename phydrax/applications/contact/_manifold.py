#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generic mechanical contact-manifold history on device.

A :class:`ContactManifoldState` retains, per canonical route key of one
static route catalogue, the last witness geometry, contact frame, warm-start
impulse, fixed-branch label, accumulated slip, age, retention and material
revision.  Updates are device sorted-key joins with deterministic birth,
death, retention and eviction; active-history overflow, duplicate keys,
stale geometry revisions and nonfinite history refuse.  Warm starts and slip
history are transported through the physical world vectors into the new
contact frame.  The state is mechanical impulse history; closure-law route
state (:mod:`._route_state`) and monolithic lagged friction remain distinct.
"""

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import ConvertibleToArray
from ._cone import ContactConeBranch
from ._route_state import match_route_keys


class ContactFrameTransport(IntEnum):
    """How retained local vectors follow a changed contact frame.

    ``NORMAL_PARALLEL`` rotates the physical world vector by the minimal
    rotation carrying the old normal onto the new one, then expresses it in the
    new tangent basis; an exactly reversed normal has no defined transport and
    resets that route's history.  ``MATERIAL_FRAME`` declares that both frames
    are attached to the same material carrier (for example frames built from
    body-fixed axes); the world vector then follows the carrier rotation
    ``F_new F_old^T`` and local components are preserved.
    """

    NORMAL_PARALLEL = 0
    MATERIAL_FRAME = 1


class ContactManifoldPolicy(StrictModule, NonTrainableState):
    """Retention and frame-transport policy of a mechanical contact manifold.

    An inactive record survives ``retention_steps - 1`` further updates before
    it dies; active records always take priority and active overflow refuses.
    ``frame_transport`` declares how retained impulses and slips follow a
    changed contact frame.
    """

    retention_steps: int = eqx.field(static=True)
    frame_transport: ContactFrameTransport = eqx.field(static=True)

    def __init__(
        self,
        *,
        retention_steps: int = 2,
        frame_transport: ContactFrameTransport = ContactFrameTransport.NORMAL_PARALLEL,
    ) -> None:
        steps = int(retention_steps)
        if steps < 0:
            raise ValueError("retention_steps must be nonnegative.")
        self.retention_steps = steps
        self.frame_transport = ContactFrameTransport(frame_transport)


class ContactManifoldState(StrictModule, NonTrainableState):
    """Fixed-capacity key-sorted mechanical history of one route catalogue."""

    route_keys: Array
    occupied: Array
    active: Array
    witness: Array
    normal: Array
    tangent_basis: Array
    impulse: Array
    branch: Array
    slip: Array
    age: Array
    retention: Array
    material_revision: Array
    geometry_revision: Array
    transaction_revision: Array
    capacity: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    tangent_dimension: int = eqx.field(static=True)
    witness_count: int = eqx.field(static=True)
    catalogue_id: str = eqx.field(static=True)

    def __init__(
        self,
        route_keys: ConvertibleToArray,
        occupied: ConvertibleToArray,
        active: ConvertibleToArray,
        witness: ConvertibleToArray,
        normal: ConvertibleToArray,
        tangent_basis: ConvertibleToArray,
        impulse: ConvertibleToArray,
        branch: ConvertibleToArray,
        slip: ConvertibleToArray,
        age: ConvertibleToArray,
        retention: ConvertibleToArray,
        material_revision: ConvertibleToArray,
        geometry_revision: ConvertibleToArray,
        transaction_revision: ConvertibleToArray,
        /,
        *,
        catalogue_id: str,
    ) -> None:
        keys = jnp.asarray(route_keys)
        witness_ = jnp.asarray(witness)
        if not jnp.issubdtype(witness_.dtype, jnp.floating):
            witness_ = witness_.astype(jnp.float64)
        dtype = witness_.dtype
        if keys.ndim != 1 or not jnp.issubdtype(keys.dtype, jnp.integer):
            raise TypeError("route_keys must be one integer capacity vector.")
        capacity = int(keys.shape[0])
        if capacity <= 0:
            raise ValueError("Contact manifold capacity must be positive.")
        if witness_.ndim != 3 or witness_.shape[0] != capacity:
            raise ValueError("witness must have shape (capacity, witnesses, ambient).")
        witnesses, ambient = int(witness_.shape[1]), int(witness_.shape[2])
        basis = jnp.asarray(tangent_basis, dtype=dtype)
        if basis.ndim != 3 or basis.shape[:2] != (capacity, ambient):
            raise ValueError("tangent_basis must have shape (capacity, ambient, T).")
        tangent = int(basis.shape[2])
        if ambient not in (2, 3) or tangent != ambient - 1:
            raise ValueError("Contact manifolds need ambient 2 or 3 and T = ambient-1.")
        identifier = str(catalogue_id)
        if not identifier:
            raise ValueError("catalogue_id must be nonempty.")
        shapes = {
            "occupied": (capacity,),
            "active": (capacity,),
            "normal": (capacity, ambient),
            "impulse": (capacity, 1 + tangent),
            "branch": (capacity,),
            "slip": (capacity, tangent),
            "age": (capacity,),
            "retention": (capacity,),
            "material_revision": (capacity,),
        }
        values = {
            "occupied": jnp.asarray(occupied, dtype=jnp.bool_),
            "active": jnp.asarray(active, dtype=jnp.bool_),
            "normal": jnp.asarray(normal, dtype=dtype),
            "impulse": jnp.asarray(impulse, dtype=dtype),
            "branch": jnp.asarray(branch, dtype=jnp.int8),
            "slip": jnp.asarray(slip, dtype=dtype),
            "age": jnp.asarray(age, dtype=jnp.int32),
            "retention": jnp.asarray(retention, dtype=jnp.int32),
            "material_revision": jnp.asarray(material_revision, dtype=jnp.int64),
        }
        for name, shape in shapes.items():
            if values[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}.")
        geometry = jnp.asarray(geometry_revision, dtype=jnp.int64)
        transaction = jnp.asarray(transaction_revision, dtype=jnp.int64)
        if geometry.shape != () or transaction.shape != ():
            raise ValueError("Manifold revisions must be scalars.")
        self.route_keys = keys
        self.occupied = values["occupied"]
        self.active = values["active"]
        self.witness = witness_
        self.normal = values["normal"]
        self.tangent_basis = basis
        self.impulse = values["impulse"]
        self.branch = values["branch"]
        self.slip = values["slip"]
        self.age = values["age"]
        self.retention = values["retention"]
        self.material_revision = values["material_revision"]
        self.geometry_revision = geometry
        self.transaction_revision = transaction
        self.capacity = capacity
        self.ambient_dimension = ambient
        self.tangent_dimension = tangent
        self.witness_count = witnesses
        self.catalogue_id = identifier

    @classmethod
    def empty(
        cls,
        capacity: int,
        /,
        *,
        catalogue_id: str,
        ambient_dimension: int = 3,
        witness_count: int = 2,
        dtype: DTypeLike = jnp.float64,
    ) -> ContactManifoldState:
        count = int(capacity)
        ambient = int(ambient_dimension)
        witnesses = int(witness_count)
        if witnesses <= 0:
            raise ValueError("witness_count must be positive.")
        tangent = ambient - 1
        return cls(
            jnp.zeros((count,), dtype=jnp.int64),
            jnp.zeros((count,), dtype=jnp.bool_),
            jnp.zeros((count,), dtype=jnp.bool_),
            jnp.zeros((count, witnesses, ambient), dtype=dtype),
            jnp.zeros((count, ambient), dtype=dtype),
            jnp.zeros((count, ambient, tangent), dtype=dtype),
            jnp.zeros((count, 1 + tangent), dtype=dtype),
            jnp.full((count,), int(ContactConeBranch.UNCLASSIFIED), dtype=jnp.int8),
            jnp.zeros((count, tangent), dtype=dtype),
            jnp.zeros((count,), dtype=jnp.int32),
            jnp.zeros((count,), dtype=jnp.int32),
            jnp.zeros((count,), dtype=jnp.int64),
            jnp.asarray(0, dtype=jnp.int64),
            jnp.asarray(0, dtype=jnp.int64),
            catalogue_id=catalogue_id,
        )


class ContactManifoldObservation(StrictModule):
    """Current canonical routes with witnesses, frames and material revisions."""

    route_keys: Array
    valid: Array
    normal: Array
    tangent_basis: Array
    witness: Array
    material_revision: Array
    geometry_revision: Array
    catalogue_id: str = eqx.field(static=True)

    def __init__(
        self,
        route_keys: ConvertibleToArray,
        valid: ConvertibleToArray,
        normal: ConvertibleToArray,
        tangent_basis: ConvertibleToArray,
        witness: ConvertibleToArray,
        material_revision: ConvertibleToArray,
        geometry_revision: ConvertibleToArray,
        /,
        *,
        catalogue_id: str,
    ) -> None:
        keys = jnp.asarray(route_keys)
        if keys.ndim != 1 or not jnp.issubdtype(keys.dtype, jnp.integer):
            raise TypeError("route_keys must be one integer vector.")
        count = int(keys.shape[0])
        normal_ = jnp.asarray(normal)
        if not jnp.issubdtype(normal_.dtype, jnp.floating):
            normal_ = normal_.astype(jnp.float64)
        if normal_.ndim != 2 or normal_.shape[0] != count:
            raise ValueError("normal must have shape (routes, ambient).")
        ambient = int(normal_.shape[1])
        basis = jnp.asarray(tangent_basis, dtype=normal_.dtype)
        if basis.shape != (count, ambient, ambient - 1):
            raise ValueError("tangent_basis must have shape (routes, ambient, T).")
        witness_ = jnp.asarray(witness, dtype=normal_.dtype)
        if (
            witness_.ndim != 3
            or witness_.shape[0] != count
            or (witness_.shape[2] != ambient)
        ):
            raise ValueError("witness must have shape (routes, witnesses, ambient).")
        mask = jnp.asarray(valid, dtype=jnp.bool_)
        if mask.shape != (count,):
            raise ValueError("valid must have route shape.")
        material = jnp.asarray(material_revision, dtype=jnp.int64)
        if material.shape == ():
            material = jnp.full((count,), material)
        if material.shape != (count,):
            raise ValueError("material_revision must be scalar or per route.")
        geometry = jnp.asarray(geometry_revision, dtype=jnp.int64)
        if geometry.shape != ():
            raise ValueError("geometry_revision must be one scalar.")
        identifier = str(catalogue_id)
        if not identifier:
            raise ValueError("catalogue_id must be nonempty.")
        self.route_keys = keys
        self.valid = mask
        self.normal = normal_
        self.tangent_basis = basis
        self.witness = witness_
        self.material_revision = material
        self.geometry_revision = geometry
        self.catalogue_id = identifier

    @property
    def capacity(self) -> int:
        return int(self.route_keys.shape[0])


class ContactManifoldTransition(StrictModule):
    """Atomic candidate history; ``commit`` only after the full step succeeds."""

    previous: ContactManifoldState
    candidate: ContactManifoldState
    warm_start: Array
    born_keys: Array
    died_keys: Array
    resurrected_keys: Array
    born_count: Array
    died_count: Array
    resurrected_count: Array
    material_changed_count: Array
    frame_reset_count: Array
    duplicate_count: Array
    active_overflow: Array
    stale_geometry: Array
    frame_residual: Array
    finite: Array
    successful: Array

    @property
    def state(self) -> ContactManifoldState:
        return self.candidate

    def commit(self, /) -> ContactManifoldState:
        return self.candidate

    def rollback(self, /) -> ContactManifoldState:
        return self.previous


def _frame_rotation(old_normal: Array, new_normal: Array, /) -> tuple[Array, Array]:
    """Minimal rotations carrying old normals onto new normals, and definedness."""

    dtype = new_normal.dtype
    cosine = jnp.sum(old_normal * new_normal, axis=-1)
    defined = cosine > -1.0 + jnp.sqrt(jnp.finfo(dtype).eps)
    ambient = new_normal.shape[-1]
    if ambient == 2:
        sine = old_normal[:, 0] * new_normal[:, 1] - old_normal[:, 1] * new_normal[:, 0]
        rotation = jnp.stack(
            (
                jnp.stack((cosine, -sine), axis=-1),
                jnp.stack((sine, cosine), axis=-1),
            ),
            axis=-2,
        )
        return rotation, defined
    axis = jnp.cross(old_normal, new_normal)
    zero = jnp.zeros_like(cosine)
    skew = jnp.stack(
        (
            jnp.stack((zero, -axis[:, 2], axis[:, 1]), axis=-1),
            jnp.stack((axis[:, 2], zero, -axis[:, 0]), axis=-1),
            jnp.stack((-axis[:, 1], axis[:, 0], zero), axis=-1),
        ),
        axis=-2,
    )
    factor = 1.0 / jnp.where(defined, 1.0 + cosine, 1.0)
    rotation = (
        jnp.eye(3, dtype=dtype)[None] + skew + factor[:, None, None] * (skew @ skew)
    )
    return rotation, defined


def _transport(
    policy: ContactManifoldPolicy,
    old_normal: Array,
    old_basis: Array,
    old_impulse: Array,
    old_slip: Array,
    new_normal: Array,
    new_basis: Array,
    /,
) -> tuple[Array, Array, Array]:
    if policy.frame_transport == ContactFrameTransport.MATERIAL_FRAME:
        return old_impulse, old_slip, jnp.ones(old_impulse.shape[:1], dtype=jnp.bool_)
    rotation, defined = _frame_rotation(old_normal, new_normal)
    world_impulse = (
        old_normal * old_impulse[:, :1] + (old_basis @ old_impulse[:, 1:, None])[..., 0]
    )
    world_slip = (old_basis @ old_slip[..., None])[..., 0]
    moved_impulse = (rotation @ world_impulse[..., None])[..., 0]
    moved_slip = (rotation @ world_slip[..., None])[..., 0]
    impulse = jnp.concatenate(
        (
            jnp.sum(new_normal * moved_impulse, axis=-1, keepdims=True),
            (jnp.swapaxes(new_basis, -1, -2) @ moved_impulse[..., None])[..., 0],
        ),
        axis=-1,
    )
    slip = (jnp.swapaxes(new_basis, -1, -2) @ moved_slip[..., None])[..., 0]
    return impulse, slip, defined


def _padded_sorted(keys: Array, selected: Array, /) -> Array:
    sentinel = jnp.iinfo(keys.dtype).max
    ordered = jnp.sort(jnp.where(selected, keys, sentinel))
    return jnp.where(ordered == sentinel, -1, ordered)


def update_contact_manifold(
    state: ContactManifoldState,
    observation: ContactManifoldObservation,
    /,
    *,
    policy: ContactManifoldPolicy,
) -> ContactManifoldTransition:
    """Join current routes into history; return the atomic candidate transition."""

    if not isinstance(state, ContactManifoldState):
        raise TypeError("state must be ContactManifoldState.")
    if not isinstance(observation, ContactManifoldObservation):
        raise TypeError("observation must be ContactManifoldObservation.")
    if not isinstance(policy, ContactManifoldPolicy):
        raise TypeError("policy must be ContactManifoldPolicy.")
    if observation.catalogue_id != state.catalogue_id:
        raise ValueError("Route keys are not transferable across catalogues.")
    if (
        observation.normal.shape[1] != state.ambient_dimension
        or observation.witness.shape[1] != state.witness_count
    ):
        raise ValueError("Observation geometry does not match the manifold schema.")
    capacity = state.capacity
    count = observation.capacity
    dtype = state.impulse.dtype
    valid = observation.valid
    keys = observation.route_keys.astype(state.route_keys.dtype)
    found, source, duplicate, old_continued = match_route_keys(
        keys, valid, state.route_keys, state.occupied
    )
    material_changed = found & (
        state.material_revision[source] != observation.material_revision
    )
    transported_impulse, transported_slip, defined = _transport(
        policy,
        state.normal[source],
        state.tangent_basis[source],
        state.impulse[source],
        state.slip[source],
        observation.normal.astype(dtype),
        observation.tangent_basis.astype(dtype),
    )
    frame_reset = found & ~material_changed & ~defined
    inherit = found & ~material_changed & defined
    warm = jnp.where(inherit[:, None], transported_impulse, 0.0)
    warm = jnp.where(valid[:, None], warm, 0.0)
    new_slip = jnp.where(inherit[:, None], transported_slip, 0.0)
    new_branch = jnp.where(
        inherit, state.branch[source], int(ContactConeBranch.UNCLASSIFIED)
    ).astype(jnp.int8)
    new_age = jnp.where(found, state.age[source] + 1, 0).astype(jnp.int32)
    resurrected = found & ~state.active[source]
    born = valid & ~found

    remaining = state.retention - 1
    retained = state.occupied & ~old_continued & (remaining > 0)

    # Candidate records: current observations first, then retained history.
    present = jnp.concatenate((valid, retained))
    active = jnp.concatenate((valid, jnp.zeros((capacity,), dtype=jnp.bool_)))
    record_keys = jnp.concatenate((keys, state.route_keys))
    record_retention = jnp.concatenate(
        (
            jnp.full((count,), policy.retention_steps, dtype=jnp.int32),
            remaining.astype(jnp.int32),
        )
    )
    record_age = jnp.concatenate((new_age, state.age + 1))
    priority = jnp.lexsort(
        (record_keys, -record_age, -record_retention, ~active, ~present)
    )
    top = priority[:capacity]
    selected = present[top]
    order = jnp.lexsort((record_keys[top], ~selected))
    final = top[order]
    occupied = selected[order]

    def gather(current: Array, history: Array) -> Array:
        merged = jnp.concatenate((current, history.astype(current.dtype)), axis=0)
        values = merged[final]
        mask = occupied.reshape(occupied.shape + (1,) * (values.ndim - 1))
        return jnp.where(mask, values, jnp.zeros_like(values))

    kept = jnp.zeros((count + capacity,), dtype=jnp.bool_).at[top].set(selected)
    active_count = jnp.sum(valid, dtype=jnp.int32)
    active_overflow = active_count > capacity
    candidate = ContactManifoldState(
        gather(keys, state.route_keys),
        occupied,
        gather(valid, jnp.zeros((capacity,), dtype=jnp.bool_)),
        gather(observation.witness.astype(dtype), state.witness),
        gather(observation.normal.astype(dtype), state.normal),
        gather(observation.tangent_basis.astype(dtype), state.tangent_basis),
        gather(warm, state.impulse),
        gather(new_branch, state.branch),
        gather(new_slip, state.slip),
        gather(new_age, state.age + 1),
        gather(record_retention[:count], remaining),
        gather(observation.material_revision, state.material_revision),
        observation.geometry_revision,
        state.transaction_revision + 1,
        catalogue_id=state.catalogue_id,
    )
    died = state.occupied & ~old_continued & ~kept[count:]
    duplicate_count = jnp.sum(duplicate, dtype=jnp.int32)
    stale_geometry = observation.geometry_revision < state.geometry_revision
    observed_basis = jnp.where(valid[:, None, None], observation.tangent_basis, 0.0)
    observed_normal = jnp.where(valid[:, None], observation.normal, 0.0)
    normal_residual = jnp.max(
        jnp.abs(
            (jnp.swapaxes(observed_basis, -1, -2) @ observed_normal[..., None])[..., 0]
        ),
        initial=0.0,
    )
    gram = jnp.swapaxes(observed_basis, -1, -2) @ observed_basis
    identity = jnp.eye(state.tangent_dimension, dtype=gram.dtype)[None]
    orthogonal_residual = jnp.max(
        jnp.where(valid[:, None, None], jnp.abs(gram - identity), 0.0), initial=0.0
    )
    unit_residual = jnp.max(
        jnp.where(valid, jnp.abs(jnp.sum(observed_normal**2, axis=-1) - 1.0), 0.0),
        initial=0.0,
    )
    frame_residual = jnp.maximum(
        jnp.maximum(normal_residual, orthogonal_residual), unit_residual
    ).astype(dtype)
    frame_tolerance = jnp.sqrt(jnp.finfo(dtype).eps)
    finite = (
        jnp.all(jnp.isfinite(candidate.witness))
        & jnp.all(jnp.isfinite(candidate.normal))
        & jnp.all(jnp.isfinite(candidate.tangent_basis))
        & jnp.all(jnp.isfinite(candidate.impulse))
        & jnp.all(jnp.isfinite(candidate.slip))
        & jnp.all(jnp.isfinite(warm))
    )
    successful = (
        finite
        & (duplicate_count == 0)
        & ~active_overflow
        & ~stale_geometry
        & (frame_residual <= frame_tolerance)
    )
    return ContactManifoldTransition(
        previous=state,
        candidate=candidate,
        warm_start=warm,
        born_keys=_padded_sorted(keys, born),
        died_keys=_padded_sorted(state.route_keys, died),
        resurrected_keys=_padded_sorted(keys, resurrected),
        born_count=jnp.sum(born, dtype=jnp.int32),
        died_count=jnp.sum(died, dtype=jnp.int32),
        resurrected_count=jnp.sum(resurrected, dtype=jnp.int32),
        material_changed_count=jnp.sum(material_changed, dtype=jnp.int32),
        frame_reset_count=jnp.sum(frame_reset, dtype=jnp.int32),
        duplicate_count=duplicate_count,
        active_overflow=active_overflow,
        stale_geometry=stale_geometry,
        frame_residual=frame_residual,
        finite=finite,
        successful=successful,
    )


def record_contact_response(
    state: ContactManifoldState,
    route_keys: ConvertibleToArray,
    impulse: ConvertibleToArray,
    branch: ConvertibleToArray,
    slip_velocity: ConvertibleToArray,
    /,
    *,
    step_size: ConvertibleToArray,
    valid: ConvertibleToArray | None = None,
) -> ContactManifoldState:
    """Record accepted response impulses, branches and slip by sorted-key join.

    Impulses and slip rates are expressed in the frames already stored for
    their keys (the frames of the observation that produced the response).
    Responses for keys absent from the history are not recorded.
    """

    if not isinstance(state, ContactManifoldState):
        raise TypeError("state must be ContactManifoldState.")
    keys = jnp.asarray(route_keys).astype(state.route_keys.dtype)
    count = int(keys.shape[0])
    dtype = state.impulse.dtype
    impulse_ = jnp.asarray(impulse, dtype=dtype)
    branch_ = jnp.asarray(branch, dtype=jnp.int8)
    slip_rate = jnp.asarray(slip_velocity, dtype=dtype)
    mask = (
        jnp.ones((count,), dtype=jnp.bool_)
        if valid is None
        else jnp.asarray(valid, dtype=jnp.bool_)
    )
    if (
        keys.ndim != 1
        or impulse_.shape != (count, 1 + state.tangent_dimension)
        or branch_.shape != (count,)
        or slip_rate.shape != (count, state.tangent_dimension)
        or mask.shape != (count,)
    ):
        raise ValueError("Response keys, impulses, branches and slips disagree.")
    step = jnp.asarray(step_size, dtype=dtype)
    if step.shape != ():
        raise ValueError("step_size must be one scalar.")
    found, source, _, _ = match_route_keys(state.route_keys, state.occupied, keys, mask)
    recorded_impulse = jnp.where(found[:, None], impulse_[source], state.impulse)
    recorded_branch = jnp.where(found, branch_[source], state.branch)
    recorded_slip = jnp.where(
        found[:, None], state.slip + step * slip_rate[source], state.slip
    )
    return ContactManifoldState(
        state.route_keys,
        state.occupied,
        state.active,
        state.witness,
        state.normal,
        state.tangent_basis,
        recorded_impulse,
        recorded_branch,
        recorded_slip,
        state.age,
        state.retention,
        state.material_revision,
        state.geometry_revision,
        state.transaction_revision + 1,
        catalogue_id=state.catalogue_id,
    )


__all__ = [
    "ContactFrameTransport",
    "ContactManifoldObservation",
    "ContactManifoldPolicy",
    "ContactManifoldState",
    "ContactManifoldTransition",
    "record_contact_response",
    "update_contact_manifold",
]
