#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified current geometry, image branches and group reconstruction for CVs."""

from __future__ import annotations

from collections import deque
from enum import IntFlag, StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._identity import IdentityWords
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import PeriodicCell
from .._neighborhood_bundle import AtomisticNeighborhoodView
from .._potential_program import AtomisticPotentialContext
from .._system import PreparedAtomisticSystem


class CollectiveVariableStatus(IntFlag):
    """Explicit CV failure and regularity causes; zero is a regular success."""

    ZERO_LENGTH = 1
    COLLINEAR = 2
    ALIGNMENT_GAP = 4
    IMAGE_AMBIGUOUS = 8
    BRANCH_UNCERTIFIED = 16
    NONFINITE = 32
    SITE_REALIZATION = 64
    ZERO_DISTANCE = 128
    NEIGHBORHOOD = 256


# Causes after which the CV value itself is not defined; the remaining flags
# (zero length, collinear, alignment gap, zero distance) keep a finite value
# whose first derivative does not exist.
VALUE_FAILURES = int(
    CollectiveVariableStatus.IMAGE_AMBIGUOUS
    | CollectiveVariableStatus.BRANCH_UNCERTIFIED
    | CollectiveVariableStatus.NONFINITE
    | CollectiveVariableStatus.SITE_REALIZATION
    | CollectiveVariableStatus.NEIGHBORHOOD
)


class CollectiveVariableImageBranch(StrEnum):
    """How periodic coordinates of one CV are placed on a single branch.

    ``MINIMUM_IMAGE`` evaluates every point-pair vector by the certified
    minimum image.  ``UNWRAPPED`` uses the accepted material (unwrapped)
    coordinates supplied with the geometry.  ``TOPOLOGY`` rebuilds a group by
    walking a spanning tree of its declared bonds with per-bond minimum images
    and checks every cycle closure.  ``ANCHORED`` places every member by its
    minimum image from the first member under a declared maximum extent.
    """

    MINIMUM_IMAGE = "minimum-image"
    UNWRAPPED = "unwrapped"
    TOPOLOGY = "topology"
    ANCHORED = "anchored"


def status_flag(condition: ArrayLike, flag: CollectiveVariableStatus, /) -> Array:
    return jnp.where(jnp.asarray(condition), jnp.int32(int(flag)), jnp.int32(0))


def reduce_status(status: ArrayLike, /) -> Array:
    """Bitwise union of an array of status words."""
    words = jnp.asarray(status, dtype=jnp.int32)
    return jax.lax.reduce(
        words, jnp.int32(0), jax.lax.bitwise_or, tuple(range(words.ndim))
    )


def safe_norm(vector: ArrayLike, /, *, axis: int = -1) -> tuple[Array, Array]:
    """Euclidean norm with a finite zero-length gradient and its regularity mask."""

    value = jnp.asarray(vector)
    squared = jnp.sum(value * value, axis=axis)
    positive = squared > 0.0
    norm = jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1.0)), 0.0)
    return norm, positive


class CollectiveVariableGeometry(StrictModule):
    """Current coordinates, cell and image branch consumed by one CV evaluation.

    ``positions`` are the wrapped coordinates of the selected coordinate
    domain owner, ``unwrapped_positions`` the accepted material coordinates.
    ``unwrapped_declared`` is false only when a caller built the geometry from
    wrapped coordinates alone; group CVs on the unwrapped branch then refuse.
    ``cell_vectors`` are the current (possibly strained) lattice rows.
    """

    positions: Array
    unwrapped_positions: Array
    site_positions: Array | None
    cell: PeriodicCell | None
    cell_vectors: Array | None
    neighborhoods: tuple[AtomisticNeighborhoodView, ...]
    site_successful: Array
    unwrapped_declared: bool = eqx.field(static=True)

    def __init__(
        self,
        positions: ArrayLike,
        /,
        *,
        unwrapped_positions: ArrayLike | None = None,
        cell: PeriodicCell | None = None,
        cell_vectors: ArrayLike | None = None,
        site_positions: ArrayLike | None = None,
        neighborhoods: tuple[AtomisticNeighborhoodView, ...] = (),
        site_successful: ArrayLike = True,
    ) -> None:
        position = jnp.asarray(positions)
        if position.ndim != 2 or position.shape[-1] != 3:
            raise ValueError("CV geometry positions must have shape (capacity, 3).")
        unwrapped = (
            position if unwrapped_positions is None else jnp.asarray(unwrapped_positions)
        )
        if unwrapped.shape != position.shape:
            raise ValueError("Unwrapped CV positions must align with positions.")
        if cell is not None and not isinstance(cell, PeriodicCell):
            raise TypeError("cell must be a PeriodicCell or None.")
        vectors = None if cell_vectors is None else jnp.asarray(cell_vectors)
        if vectors is not None and (vectors.ndim != 2 or vectors.shape[-1] != 3):
            raise ValueError("CV cell vectors must have shape (rank, 3).")
        if vectors is not None and vectors.shape[0] == 0:
            vectors = None
        if (
            cell is not None
            and vectors is not None
            and vectors.shape != cell.vectors.shape
        ):
            raise ValueError("Current cell vectors must align with the prepared cell.")
        if any(not isinstance(view, AtomisticNeighborhoodView) for view in neighborhoods):
            raise TypeError(
                "neighborhoods must contain AtomisticNeighborhoodView values."
            )
        self.positions = position
        self.unwrapped_positions = unwrapped
        self.site_positions = (
            None if site_positions is None else jnp.asarray(site_positions)
        )
        self.cell = cell
        self.cell_vectors = vectors
        self.neighborhoods = tuple(neighborhoods)
        self.site_successful = jnp.asarray(site_successful, dtype=jnp.bool_)
        self.unwrapped_declared = unwrapped_positions is not None or not _periodic(cell)

    @classmethod
    def from_context(
        cls, context: AtomisticPotentialContext, /
    ) -> "CollectiveVariableGeometry":
        """Bind the program's current (strain-mapped) geometry of one term.

        The program contract makes ``context.unwrapped_positions`` the
        material coordinates of the bound configuration.
        """

        return cls(
            context.positions,
            unwrapped_positions=context.unwrapped_positions,
            cell=context.cell,
            cell_vectors=context.cell_vectors if context.cell is not None else None,
            site_positions=context.site_positions,
            neighborhoods=context.neighborhoods,
            site_successful=context.site_state.successful,
        )

    @property
    def periodic(self) -> bool:
        return _periodic(self.cell)

    @property
    def vectors(self) -> Array | None:
        if self.cell is None:
            return None
        return self.cell.vectors if self.cell_vectors is None else self.cell_vectors

    def volume(self) -> Array:
        vectors = self.vectors
        if vectors is None or vectors.shape != (3, 3):
            raise ValueError(
                "Cell-volume quantities require a full three-dimensional cell."
            )
        return jnp.abs(jnp.sum(vectors[0] * jnp.cross(vectors[1], vectors[2])))

    def unique_image_radius(self) -> Array:
        """Half the shortest nonzero lattice translation of the current cell.

        Uses the owning cell's prepared image stencil.  A minimum-image vector
        strictly shorter than this radius is the unique shortest image.
        """

        cell = self.cell
        if cell is None or not self.periodic:
            return jnp.asarray(jnp.inf, dtype=self.positions.dtype)
        shifts = jnp.asarray(cell.image_shifts, dtype=self.positions.dtype)
        periodic = jnp.asarray(cell.periodic_mask)
        shifts = jnp.where(periodic[None, :], shifts, 0.0)
        translations = shifts @ jax.lax.stop_gradient(
            cell.vectors if self.cell_vectors is None else self.cell_vectors
        )
        length, nonzero = safe_norm(translations)
        return 0.5 * jnp.min(jnp.where(nonzero, length, jnp.inf))

    def minimum_image(self, displacement: ArrayLike, /) -> tuple[Array, Array]:
        """Certified minimum image and its uniqueness margin (positive = unique)."""

        vector = jnp.asarray(displacement)
        cell = self.cell
        if cell is None or not self.periodic:
            return vector, jnp.full(vector.shape[:-1], jnp.inf, dtype=vector.dtype)
        image = (
            cell.minimum_image(vector)
            if self.cell_vectors is None
            else cell.minimum_image_with_vectors(vector, self.cell_vectors)
        )
        length, _ = safe_norm(jax.lax.stop_gradient(image))
        return image, self.unique_image_radius() - length

    def neighborhood(self, requirement_id: str, /) -> AtomisticNeighborhoodView:
        for view in self.neighborhoods:
            if view.requirement_id == requirement_id:
                return view
        raise ValueError(
            "The CV geometry carries no view for this neighborhood requirement."
        )


def _periodic(cell: PeriodicCell | None, /) -> bool:
    return cell is not None and any(cell.periodic_axes)


class PreparedGroupBranch(StrictModule, NonTrainableState):
    """One CV group's prepared single-branch reconstruction.

    For ``TOPOLOGY`` the spanning tree (``tree_parent``/``tree_child`` in
    breadth-first order, local indices) and the nontree ``closures`` come from
    the system's bonds among the group members.  A disconnected group or a
    group without bonds is refused at preparation.

    PyTree: ``branch`` is a static selector; the anchored ``maximum_extent``
    is a dynamic scalar (``None`` for other branches) and the exact
    canonical branch digest is the dynamic ``branch_identity`` (``branch_id``
    decodes it on the host).
    """

    branch: CollectiveVariableImageBranch = eqx.field(static=True)
    indices: Array
    tree_parent: Array
    tree_child: Array
    closures: Array
    maximum_extent: Array | None
    branch_identity: IdentityWords

    def __init__(
        self,
        branch: CollectiveVariableImageBranch,
        indices: np.ndarray,
        system: PreparedAtomisticSystem,
        /,
        *,
        maximum_extent: float | None = None,
    ) -> None:
        members = np.asarray(indices, dtype=np.int64).reshape((-1,))
        if members.size == 0 or len(set(members.tolist())) != members.size:
            raise ValueError("A CV group requires unique members.")
        parent = np.zeros((0,), dtype=np.int32)
        child = np.zeros((0,), dtype=np.int32)
        closures = np.zeros((0, 2), dtype=np.int32)
        extent = None
        if branch is CollectiveVariableImageBranch.TOPOLOGY:
            parent, child, closures = _spanning_tree(members, system)
        elif branch is CollectiveVariableImageBranch.ANCHORED:
            if maximum_extent is None:
                raise ValueError(
                    "Anchored reconstruction requires a declared maximum_extent."
                )
            extent = float(maximum_extent)
            if not np.isfinite(extent) or extent <= 0.0:
                raise ValueError("maximum_extent must be finite and positive.")
            cell = system.cell
            if cell is not None and extent >= cell.unique_image_radius:
                raise ValueError(
                    "The declared extent is not below the cell's unique-image radius; "
                    "anchored reconstruction cannot be certified."
                )
        elif maximum_extent is not None:
            raise ValueError("maximum_extent belongs only to anchored reconstruction.")
        if branch is CollectiveVariableImageBranch.MINIMUM_IMAGE:
            raise ValueError(
                "Groups are placed by an unwrapped, topology or anchored branch."
            )
        self.branch = branch
        self.indices = jnp.asarray(members, dtype=jnp.int32)
        self.tree_parent = jnp.asarray(parent, dtype=jnp.int32)
        self.tree_child = jnp.asarray(child, dtype=jnp.int32)
        self.closures = jnp.asarray(closures, dtype=jnp.int32)
        self.maximum_extent = None if extent is None else jnp.asarray(extent)
        self.branch_identity = IdentityWords.of(
            {
                "kind": "cv-group-branch",
                "branch": branch.value,
                "members": members.tolist(),
                "tree": [parent.tolist(), child.tolist()],
                "closures": closures.tolist(),
                "maximum_extent": extent,
                "system": system.prepared_id,
            }
        )

    @property
    def branch_id(self) -> str:
        """Host decoding of the canonical group-branch digest."""
        return self.branch_identity.fingerprint()

    def coordinates(
        self, geometry: CollectiveVariableGeometry, /
    ) -> tuple[Array, Array, Array]:
        """Return ``(coordinates (g, 3), branch margin, status)`` of the group."""

        index = self.indices
        dtype = geometry.positions.dtype
        if not geometry.periodic:
            return (
                geometry.positions[index],
                jnp.asarray(jnp.inf, dtype=dtype),
                jnp.int32(0),
            )
        if self.branch is CollectiveVariableImageBranch.UNWRAPPED:
            status = (
                jnp.int32(0)
                if geometry.unwrapped_declared
                else jnp.int32(int(CollectiveVariableStatus.BRANCH_UNCERTIFIED))
            )
            return (
                geometry.unwrapped_positions[index],
                jnp.asarray(jnp.inf, dtype=dtype),
                status,
            )
        positions = geometry.positions[index]
        extent = self.maximum_extent
        if self.branch is CollectiveVariableImageBranch.ANCHORED and extent is not None:
            vector, margin = geometry.minimum_image(positions - positions[0])
            length, _ = safe_norm(jax.lax.stop_gradient(vector))
            extent_margin = jnp.minimum(
                jnp.min(margin), extent.astype(dtype) - jnp.max(length)
            )
            radius_margin = geometry.unique_image_radius() - extent.astype(dtype)
            certified = (extent_margin >= 0.0) & (radius_margin > 0.0)
            return (
                positions[0] + vector,
                jnp.minimum(extent_margin, radius_margin),
                status_flag(~certified, CollectiveVariableStatus.BRANCH_UNCERTIFIED),
            )
        bonds, bond_margin = geometry.minimum_image(
            positions[self.tree_child] - positions[self.tree_parent]
        )

        def place(placed: Array, edge: tuple[Array, Array, Array]) -> tuple[Array, None]:
            parent, child, vector = edge
            return placed.at[child].set(placed[parent] + vector), None

        start = jnp.zeros_like(positions).at[0].set(positions[0])
        rebuilt, _ = jax.lax.scan(
            place, start, (self.tree_parent, self.tree_child, bonds)
        )
        margin = jnp.min(bond_margin, initial=jnp.inf)
        closure_ok = jnp.asarray(True)
        if self.closures.shape[0]:
            left, right = self.closures[:, 0], self.closures[:, 1]
            reference, closure_margin = geometry.minimum_image(
                positions[left] - positions[right]
            )
            mismatch, _ = safe_norm(
                jax.lax.stop_gradient(rebuilt[left] - rebuilt[right] - reference)
            )
            scale = jnp.maximum(geometry.unique_image_radius(), 1.0)
            tolerance = 1.0e3 * jnp.finfo(dtype).eps * scale
            closure_ok = jnp.all(mismatch <= tolerance)
            margin = jnp.minimum(margin, jnp.min(closure_margin))
        certified = (margin > 0.0) & closure_ok
        return (
            rebuilt,
            margin,
            status_flag(~certified, CollectiveVariableStatus.BRANCH_UNCERTIFIED),
        )


def _spanning_tree(
    members: np.ndarray, system: PreparedAtomisticSystem, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bonds = np.asarray(system.topology.bond_indices, dtype=np.int64).reshape((-1, 2))
    local = {int(slot): position for position, slot in enumerate(members.tolist())}
    edges = [
        (local[int(a)], local[int(b)])
        for a, b in bonds
        if int(a) in local and int(b) in local
    ]
    if members.size > 1 and not edges:
        raise ValueError(
            "Topology reconstruction requires declared bonds among the group."
        )
    adjacency: dict[int, list[int]] = {index: [] for index in range(members.size)}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen = {0}
    parents: list[int] = []
    children: list[int] = []
    tree: set[tuple[int, int]] = set()
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacency[node]):
            if neighbor not in seen:
                seen.add(neighbor)
                parents.append(node)
                children.append(neighbor)
                tree.add((min(node, neighbor), max(node, neighbor)))
                queue.append(neighbor)
    if len(seen) != members.size:
        raise ValueError(
            "Topology reconstruction requires a bond-connected group; split it into "
            "connected groups or declare another branch."
        )
    closures = sorted({(min(a, b), max(a, b)) for a, b in edges} - tree)
    return (
        np.asarray(parents, dtype=np.int32),
        np.asarray(children, dtype=np.int32),
        np.asarray(closures, dtype=np.int32).reshape((-1, 2)),
    )


__all__ = [
    "CollectiveVariableGeometry",
    "CollectiveVariableImageBranch",
    "CollectiveVariableStatus",
    "PreparedGroupBranch",
]
