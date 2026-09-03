#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import CellMesh
from ....discretization._cell_complex import TetrahedralConnectivity


def _name(value: str, description: str, /) -> str:
    resolved = str(value)
    if not resolved:
        raise ValueError(f"{description} must be non-empty.")
    return resolved


def _pair(value: Sequence[str], description: str, /) -> tuple[str, str]:
    pair = tuple(str(item) for item in value)
    if len(pair) != 2 or not pair[0] or not pair[1] or pair[0] == pair[1]:
        raise ValueError(f"{description} entries must name two distinct roles.")
    return tuple(sorted(pair))


def _components(faces: np.ndarray, /) -> int:
    if faces.shape[0] == 0:
        return 0
    edge_owners: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(first), int(second))))
            edge_owners.setdefault(edge, []).append(face_index)
    neighbours = [set() for _ in range(faces.shape[0])]
    for owners in edge_owners.values():
        for first in owners:
            neighbours[first].update(owner for owner in owners if owner != first)
    unseen = set(range(faces.shape[0]))
    count = 0
    while unseen:
        count += 1
        pending = [unseen.pop()]
        while pending:
            current = pending.pop()
            attached = unseen.intersection(neighbours[current])
            unseen.difference_update(attached)
            pending.extend(attached)
    return count


def _closure_components(vertices: set[int], edges: np.ndarray, /) -> int:
    if not vertices:
        return 0
    neighbours = {vertex: set() for vertex in vertices}
    for first, second in edges:
        first_ = int(first)
        second_ = int(second)
        if first_ in vertices and second_ in vertices:
            neighbours[first_].add(second_)
            neighbours[second_].add(first_)
    unseen = set(vertices)
    count = 0
    while unseen:
        count += 1
        pending = [unseen.pop()]
        while pending:
            current = pending.pop()
            attached = unseen.intersection(neighbours[current])
            unseen.difference_update(attached)
            pending.extend(attached)
    return count


class BoundaryRoleAssignment(StrictModule, NonTrainableState):
    """Named ownership of exterior triangular facets without numeric role codes."""

    name: str = eqx.field(static=True)
    face_indices: Array
    assignment_id: str = eqx.field(static=True)

    def __init__(self, name: str, face_indices: ArrayLike, /):
        role_name = _name(name, "Boundary role name")
        indices = np.asarray(face_indices, dtype=np.int32)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("Boundary role face_indices must be a non-empty vector.")
        if np.any(indices < 0) or np.unique(indices).size != indices.size:
            raise ValueError(
                "Boundary role face indices must be unique and non-negative."
            )
        indices = np.sort(indices)
        self.name = role_name
        self.face_indices = jnp.asarray(indices)
        self.assignment_id = canonical_fingerprint(
            {
                "kind": "cardiac-boundary-role-assignment",
                "name": role_name,
                "faces": array_tree_fingerprint(indices),
            }
        )


class CardiacBoundaryProfile(StrictModule, NonTrainableState):
    """Extensible semantic contract for a particular cardiac boundary profile.

    Roles are user-defined strings.  The profile specifies only relationships that
    are necessary for a workflow; it deliberately does not assign universal
    integer labels or imply semantics for unlisted roles.
    """

    name: str = eqx.field(static=True)
    required_roles: tuple[str, ...] = eqx.field(static=True)
    connected_roles: tuple[str, ...] = eqx.field(static=True)
    disjoint_closure_pairs: tuple[tuple[str, str], ...] = eqx.field(static=True)
    shared_closure_pairs: tuple[tuple[str, str], ...] = eqx.field(static=True)
    exhaustive: bool = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        /,
        *,
        required_roles: Sequence[str],
        connected_roles: Sequence[str] = (),
        disjoint_closure_pairs: Sequence[Sequence[str]] = (),
        shared_closure_pairs: Sequence[Sequence[str]] = (),
        exhaustive: bool = False,
    ):
        profile_name = _name(name, "Boundary profile name")
        required = tuple(_name(role, "Required role") for role in required_roles)
        connected = tuple(_name(role, "Connected role") for role in connected_roles)
        if not required or len(set(required)) != len(required):
            raise ValueError("required_roles must be non-empty and unique.")
        if len(set(connected)) != len(connected) or not set(connected).issubset(required):
            raise ValueError("connected_roles must be a unique subset of required_roles.")
        disjoint = tuple(
            sorted(
                {_pair(pair, "Disjoint closure pair") for pair in disjoint_closure_pairs}
            )
        )
        shared = tuple(
            sorted({_pair(pair, "Shared closure pair") for pair in shared_closure_pairs})
        )
        declared = set(required)
        if any(not set(pair).issubset(declared) for pair in (*disjoint, *shared)):
            raise ValueError("Closure-pair roles must be declared in required_roles.")
        if set(disjoint).intersection(shared):
            raise ValueError("A role pair cannot be both disjoint and shared in closure.")
        self.name = profile_name
        self.required_roles = required
        self.connected_roles = connected
        self.disjoint_closure_pairs = disjoint
        self.shared_closure_pairs = shared
        self.exhaustive = bool(exhaustive)
        self.profile_id = canonical_fingerprint(
            {
                "kind": "cardiac-boundary-profile",
                "name": profile_name,
                "required": list(required),
                "connected": list(connected),
                "disjoint_closure": [list(pair) for pair in disjoint],
                "shared_closure": [list(pair) for pair in shared],
                "exhaustive": bool(exhaustive),
            }
        )


class BoundaryRoleEvidence(StrictModule, NonTrainableState):
    """Topology evidence emitted after boundary-role qualification."""

    role_component_counts: Array
    shared_closure_component_counts: Array
    assigned_face_count: Array
    exterior_face_count: Array
    unassigned_face_count: Array
    successful: Array

    def __init__(
        self,
        role_component_counts: ArrayLike,
        shared_closure_component_counts: ArrayLike,
        assigned_face_count: ArrayLike,
        exterior_face_count: ArrayLike,
        unassigned_face_count: ArrayLike,
        successful: ArrayLike,
        /,
    ):
        self.role_component_counts = jnp.asarray(role_component_counts, dtype=jnp.int32)
        self.shared_closure_component_counts = jnp.asarray(
            shared_closure_component_counts, dtype=jnp.int32
        )
        self.assigned_face_count = jnp.asarray(assigned_face_count, dtype=jnp.int32)
        self.exterior_face_count = jnp.asarray(exterior_face_count, dtype=jnp.int32)
        self.unassigned_face_count = jnp.asarray(unassigned_face_count, dtype=jnp.int32)
        self.successful = jnp.asarray(successful, dtype=bool)


class CardiacBoundaryRoles(StrictModule, NonTrainableState):
    """Qualified affine-P1 tetrahedral boundary roles and closure evidence."""

    mesh: CellMesh
    profile: CardiacBoundaryProfile
    assignments: tuple[BoundaryRoleAssignment, ...]
    role_names: tuple[str, ...] = eqx.field(static=True)
    face_owner: Array
    role_vertex_masks: Array
    evidence: BoundaryRoleEvidence
    roles_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        assignments: Mapping[str, ArrayLike] | Sequence[BoundaryRoleAssignment],
        /,
        *,
        profile: CardiacBoundaryProfile,
    ):
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be a CellMesh.")
        if mesh.topological_dimension != 3 or not all(
            block.cell_kind == "tetrahedron" for block in mesh.blocks
        ):
            raise ValueError("Cardiac boundary roles require an affine tetrahedral mesh.")
        if not isinstance(mesh.connectivity, TetrahedralConnectivity):
            raise TypeError("Cardiac boundary roles require tetrahedral connectivity.")
        if not isinstance(profile, CardiacBoundaryProfile):
            raise TypeError("profile must be a CardiacBoundaryProfile.")
        if isinstance(assignments, Mapping):
            normalized = tuple(
                BoundaryRoleAssignment(name, indices)
                for name, indices in sorted(
                    assignments.items(), key=lambda item: str(item[0])
                )
            )
        else:
            normalized = tuple(assignments)
        if not normalized or not all(
            isinstance(assignment, BoundaryRoleAssignment) for assignment in normalized
        ):
            raise TypeError("assignments must contain BoundaryRoleAssignment values.")
        names = tuple(assignment.name for assignment in normalized)
        if len(set(names)) != len(names):
            raise ValueError("Boundary role names must be unique.")
        if not set(profile.required_roles).issubset(names):
            missing = sorted(set(profile.required_roles).difference(names))
            raise ValueError(f"Boundary profile is missing required roles: {missing}.")

        connectivity = mesh.connectivity
        faces = np.asarray(connectivity.faces, dtype=np.int32)
        exterior_mask = np.asarray(connectivity.boundary_faces, dtype=bool)
        owner = np.full((faces.shape[0],), -1, dtype=np.int32)
        role_vertex_masks = np.zeros(
            (len(normalized), mesh.coordinates.shape[0]), dtype=bool
        )
        component_counts: list[int] = []
        for role_index, assignment in enumerate(normalized):
            indices = np.asarray(assignment.face_indices, dtype=np.int32)
            if np.any(indices >= faces.shape[0]) or np.any(~exterior_mask[indices]):
                raise ValueError(
                    f"Boundary role {assignment.name!r} contains a non-exterior face."
                )
            if np.any(owner[indices] >= 0):
                raise ValueError("Boundary roles must have disjoint face ownership.")
            owner[indices] = role_index
            role_faces = faces[indices]
            role_vertex_masks[role_index, np.unique(role_faces)] = True
            component_counts.append(_components(role_faces))
        name_to_index = {name: index for index, name in enumerate(names)}
        for role in profile.connected_roles:
            count = component_counts[name_to_index[role]]
            if count != 1:
                raise ValueError(
                    f"Boundary role {role!r} must have one edge-connected component; "
                    f"found {count}."
                )

        for first, second in profile.disjoint_closure_pairs:
            overlap = (
                role_vertex_masks[name_to_index[first]]
                & role_vertex_masks[name_to_index[second]]
            )
            if np.any(overlap):
                raise ValueError(
                    f"Boundary roles {first!r} and {second!r} must have disjoint closures."
                )

        edges = np.asarray(connectivity.edges, dtype=np.int32)
        shared_counts: list[int] = []
        for first, second in profile.shared_closure_pairs:
            shared_vertices = set(
                np.flatnonzero(
                    role_vertex_masks[name_to_index[first]]
                    & role_vertex_masks[name_to_index[second]]
                ).tolist()
            )
            count = _closure_components(shared_vertices, edges)
            shared_edge_count = sum(
                int(edge[0]) in shared_vertices and int(edge[1]) in shared_vertices
                for edge in edges
            )
            if count != 1 or shared_edge_count == 0:
                raise ValueError(
                    f"Boundary roles {first!r} and {second!r} must share one "
                    "edge-connected closure component."
                )
            shared_counts.append(count)

        assigned = int(np.count_nonzero(owner >= 0))
        exterior = int(np.count_nonzero(exterior_mask))
        unassigned = int(np.count_nonzero(exterior_mask & (owner < 0)))
        if profile.exhaustive and unassigned:
            raise ValueError(
                f"Exhaustive boundary profile leaves {unassigned} exterior faces unassigned."
            )
        evidence = BoundaryRoleEvidence(
            np.asarray(component_counts, dtype=np.int32),
            np.asarray(shared_counts, dtype=np.int32),
            assigned,
            exterior,
            unassigned,
            True,
        )
        self.mesh = mesh
        self.profile = profile
        self.assignments = normalized
        self.role_names = names
        self.face_owner = jnp.asarray(owner)
        self.role_vertex_masks = jnp.asarray(role_vertex_masks)
        self.evidence = evidence
        self.roles_id = canonical_fingerprint(
            {
                "kind": "qualified-cardiac-boundary-roles",
                "mesh": mesh.mesh_id,
                "profile": profile.profile_id,
                "assignments": [assignment.assignment_id for assignment in normalized],
            }
        )

    def role_index(self, name: str, /) -> int:
        role_name = str(name)
        if role_name not in self.role_names:
            raise KeyError(f"Unknown cardiac boundary role {role_name!r}.")
        return self.role_names.index(role_name)

    def face_indices(self, name: str, /) -> Array:
        """Return canonical global face indices owned by ``name``."""
        return self.assignments[self.role_index(name)].face_indices

    def vertex_mask(self, name: str, /) -> Array:
        """Return the P1 nodal closure mask for ``name``."""
        return self.role_vertex_masks[self.role_index(name)]

    def vertex_indices(self, name: str, /) -> Array:
        """Return P1 nodal closure indices for ``name``."""
        return jnp.flatnonzero(self.vertex_mask(name))


def left_ventricular_boundary_profile(
    *,
    endocardium: str = "lv-endocardium",
    epicardium: str = "epicardium",
    base: str = "base",
    apex: str | None = None,
    exhaustive: bool = True,
) -> CardiacBoundaryProfile:
    """Return an LV wall profile, optionally including a distinct apical cap."""

    endocardium_ = _name(endocardium, "Endocardium role")
    epicardium_ = _name(epicardium, "Epicardium role")
    base_ = _name(base, "Base role")
    roles = (endocardium_, epicardium_, base_)
    shared = ((endocardium_, base_), (epicardium_, base_))
    disjoint = ((endocardium_, epicardium_),)
    if apex is not None:
        apex_ = _name(apex, "Apex role")
        roles = (*roles, apex_)
        shared = (*shared, (endocardium_, apex_), (epicardium_, apex_))
        disjoint = (*disjoint, (apex_, base_))
    if len(set(roles)) != len(roles):
        raise ValueError("LV foundation roles must have distinct names.")
    return CardiacBoundaryProfile(
        "left-ventricular-wall-foundation",
        required_roles=roles,
        connected_roles=roles,
        disjoint_closure_pairs=disjoint,
        shared_closure_pairs=shared,
        exhaustive=exhaustive,
    )


def _distinct_role_names(
    named_roles: Sequence[tuple[str, str]],
    /,
) -> tuple[str, ...]:
    roles = tuple(_name(value, description) for description, value in named_roles)
    if len(set(roles)) != len(roles):
        raise ValueError("Cardiac boundary profile roles must have distinct names.")
    return roles


def biventricular_boundary_profile(
    *,
    lv_endocardium: str = "lv-endocardium",
    rv_endocardium: str = "rv-endocardium",
    epicardium: str = "epicardium",
    apex: str = "apex",
    base: str = "base",
    exhaustive: bool = True,
) -> CardiacBoundaryProfile:
    """Return a two-cavity ventricular wall profile with apical/basal caps."""

    lv, rv, epi, apex_, base_ = _distinct_role_names(
        (
            ("LV endocardium role", lv_endocardium),
            ("RV endocardium role", rv_endocardium),
            ("Epicardium role", epicardium),
            ("Apex role", apex),
            ("Base role", base),
        )
    )
    return CardiacBoundaryProfile(
        "biventricular-wall-foundation",
        required_roles=(lv, rv, epi, apex_, base_),
        connected_roles=(lv, rv, epi, apex_, base_),
        disjoint_closure_pairs=(
            (lv, rv),
            (lv, epi),
            (rv, epi),
            (apex_, base_),
        ),
        shared_closure_pairs=(
            (lv, apex_),
            (rv, apex_),
            (epi, apex_),
            (lv, base_),
            (rv, base_),
            (epi, base_),
        ),
        exhaustive=exhaustive,
    )


def atrial_boundary_profile(
    *,
    left_endocardium: str = "la-endocardium",
    right_endocardium: str = "ra-endocardium",
    epicardium: str = "atrial-epicardium",
    mitral_plane: str = "mitral-plane",
    tricuspid_plane: str = "tricuspid-plane",
    left_openings: Sequence[str] = (),
    right_openings: Sequence[str] = (),
    exhaustive: bool = True,
) -> CardiacBoundaryProfile:
    """Return a biatrial wall profile with explicit valve and venous caps."""

    left_openings_ = tuple(str(role) for role in left_openings)
    right_openings_ = tuple(str(role) for role in right_openings)
    named = (
        ("LA endocardium role", left_endocardium),
        ("RA endocardium role", right_endocardium),
        ("Atrial epicardium role", epicardium),
        ("Mitral plane role", mitral_plane),
        ("Tricuspid plane role", tricuspid_plane),
        *(
            (f"Left opening role {index}", role)
            for index, role in enumerate(left_openings_)
        ),
        *(
            (f"Right opening role {index}", role)
            for index, role in enumerate(right_openings_)
        ),
    )
    left, right, epi, mitral, tricuspid, *openings = _distinct_role_names(named)
    left_caps = (mitral, *openings[: len(left_openings_)])
    right_caps = (tricuspid, *openings[len(left_openings_) :])
    caps = (*left_caps, *right_caps)
    return CardiacBoundaryProfile(
        "biatrial-wall-foundation",
        required_roles=(left, right, epi, *caps),
        connected_roles=(left, right, epi, *caps),
        disjoint_closure_pairs=(
            (left, right),
            (left, epi),
            (right, epi),
            *tuple(
                (caps[first], caps[second])
                for first in range(len(caps))
                for second in range(first + 1, len(caps))
            ),
        ),
        shared_closure_pairs=(
            *tuple((left, cap) for cap in left_caps),
            *tuple((right, cap) for cap in right_caps),
            *tuple((epi, cap) for cap in caps),
        ),
        exhaustive=exhaustive,
    )


def whole_heart_boundary_profile(
    *,
    lv_endocardium: str = "lv-endocardium",
    rv_endocardium: str = "rv-endocardium",
    la_endocardium: str = "la-endocardium",
    ra_endocardium: str = "ra-endocardium",
    epicardium: str = "epicardium",
    mitral_plane: str = "mitral-plane",
    tricuspid_plane: str = "tricuspid-plane",
    aortic_plane: str = "aortic-plane",
    pulmonary_plane: str = "pulmonary-plane",
    pulmonary_vein_openings: Sequence[str] = (),
    vena_cava_openings: Sequence[str] = (),
    exhaustive: bool = True,
) -> CardiacBoundaryProfile:
    """Return four-chamber wall roles with explicit valve and vessel caps."""

    pulmonary_veins = tuple(str(role) for role in pulmonary_vein_openings)
    venae_cavae = tuple(str(role) for role in vena_cava_openings)
    named = (
        ("LV endocardium role", lv_endocardium),
        ("RV endocardium role", rv_endocardium),
        ("LA endocardium role", la_endocardium),
        ("RA endocardium role", ra_endocardium),
        ("Epicardium role", epicardium),
        ("Mitral plane role", mitral_plane),
        ("Tricuspid plane role", tricuspid_plane),
        ("Aortic plane role", aortic_plane),
        ("Pulmonary plane role", pulmonary_plane),
        *(
            (f"Pulmonary vein role {index}", role)
            for index, role in enumerate(pulmonary_veins)
        ),
        *((f"Vena cava role {index}", role) for index, role in enumerate(venae_cavae)),
    )
    lv, rv, la, ra, epi, mitral, tricuspid, aortic, pulmonary, *vessels = (
        _distinct_role_names(named)
    )
    pv_caps = tuple(vessels[: len(pulmonary_veins)])
    vc_caps = tuple(vessels[len(pulmonary_veins) :])
    caps = (mitral, tricuspid, aortic, pulmonary, *pv_caps, *vc_caps)
    endocardia = (lv, rv, la, ra)
    endocardial_disjoint = tuple(
        (endocardia[first], endocardia[second])
        for first in range(len(endocardia))
        for second in range(first + 1, len(endocardia))
    )
    cap_disjoint = tuple(
        (caps[first], caps[second])
        for first in range(len(caps))
        for second in range(first + 1, len(caps))
    )
    return CardiacBoundaryProfile(
        "whole-heart-wall-foundation",
        required_roles=(*endocardia, epi, *caps),
        connected_roles=(*endocardia, epi, *caps),
        disjoint_closure_pairs=(
            *endocardial_disjoint,
            *((endocardium, epi) for endocardium in endocardia),
            *cap_disjoint,
        ),
        shared_closure_pairs=(
            (lv, mitral),
            (la, mitral),
            (rv, tricuspid),
            (ra, tricuspid),
            (lv, aortic),
            (epi, aortic),
            (rv, pulmonary),
            (epi, pulmonary),
            *((la, cap) for cap in pv_caps),
            *((epi, cap) for cap in pv_caps),
            *((ra, cap) for cap in vc_caps),
            *((epi, cap) for cap in vc_caps),
        ),
        exhaustive=exhaustive,
    )


__all__ = [
    "BoundaryRoleAssignment",
    "BoundaryRoleEvidence",
    "CardiacBoundaryProfile",
    "CardiacBoundaryRoles",
    "atrial_boundary_profile",
    "biventricular_boundary_profile",
    "left_ventricular_boundary_profile",
    "whole_heart_boundary_profile",
]
