#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Radial screening, relaxation handoff and same-definition physical ranking.

Radial screening
----------------
A `RadialScreenPolicy` fixes the source radial-comparison semantics.  For
every retained atom ``i`` (atoms of atomic number one are removed as centres
*and* neighbours when ``include_hydrogens`` is false) the *row* of ``i`` is the
list of distances ``|r_j + n H - r_i| < cutoff`` over every retained atom
``j`` and lattice translation ``n`` except ``(j, n) = (i, 0)``; periodic self
images are neighbours.  The image box is enumerated completely: with
fractional coordinates wrapped into ``[0, 1)`` and cell heights ``h_k``, a
displacement shorter than ``cutoff`` has ``|f_k| <= cutoff / h_k``, so
``|n_k| <= floor(cutoff / h_k) + 1`` is exhaustive.  A box larger than
``max_images``, more than ``max_atoms`` retained atoms or a row with more than
``max_neighbors`` entries is refused (`RadialScreenCapacityError`); a
truncated search is never scored.  Coincident distinct atoms are refused.

Rows are ordered by ascending distance (``untyped`` and ``center-typed``) or
by ``(neighbour atomic number, distance)`` (``neighbor-typed``).  For two
distances the mismatch is ``max(d1, d2) / min(d1, d2)``.  Two rows are
compared rank by rank; a rank present in only one row is compared against the
cutoff (the source pads missing ranks with the cutoff, so a neighbour count
difference is *not* an infinite score: a neighbour just inside the cutoff
contributes a mismatch near one and the score is continuous in the
geometry).  Under ``neighbor-typed`` ranks are counted within each neighbour
species and a species group missing in one row is compared against the
cutoff.  Under ``center-typed`` and ``neighbor-typed`` rows of different
centre species are incompatible (infinite mismatch); ``untyped`` ignores
species.  The row mismatch is the largest pairwise mismatch.

The directed mismatch ``A -> B`` is the largest, over atoms of ``A``, of the
smallest row mismatch against compatible atoms of ``B`` (infinite when an
atom has no compatible partner, one when ``A`` has no retained atoms); the
score is ``max(A -> B, B -> A) - 1``, a symmetric Hausdorff comparison of
atom environments, not a bijection.  The threshold is inclusive: a pair
matches when ``score <= nextafter(threshold, +inf)`` (the source advances its
FP32 threshold one representable value outward; this is the float64 analog).

Greedy screening visits a pool in order and assigns each structure to the
first retained representative (in retention order) it matches, otherwise
retains it.  The relation is not transitive (``A ~ B``, ``B ~ C`` and
``A !~ C`` retain ``A`` and ``C``), decisions depend on order, and decisions
are independent of the pair chunk size.  A radial match is a screen, never a
proof of crystallographic equivalence; `screen_dedup_rank` reports it next to
the certified `DedupBank` decision, which alone merges structures.

Ranking
-------
`rank_crystal_candidates` compares relaxed structures of one Hamiltonian
definition (semantic program identity and numeric revision) in one energy
unit.  Without a `ChemicalPotentialReference` every candidate must have the
same formula-unit composition and the ranking value is the energy per formula
unit; with a reference the value is ``(E - sum_Z N_Z mu_Z) / N_atoms``.
Unqualified candidates are excluded with recorded reasons.  An energy ranking
is a static-lattice ordering; it is not finite-temperature polymorph
stability.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import partial
from typing import final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...typing import Dim, HostBool, HostFloat64, HostInt32, HostInt64, parse
from ...units import ANGSTROM, conversion_factor, UnitDefinition
from .._campaign import atomistic_hamiltonian_revision
from .._potential_program import AbstractPreparedAtomisticHamiltonian
from .._relaxation import (
    _method_id,
    AtomisticDeformationChart,
    AtomisticRelaxationConvergence,
    AtomisticRelaxationPlan,
    AtomisticRelaxationStatus,
    AtomisticSymmetryChart,
    RelaxationMethod,
)
from ._asu import AsymmetricUnitPlan, AsymmetricUnitState, ExpandedCrystal
from ._dedup import DedupBank, DedupCandidate, DedupSubmission, submit_to_dedup_bank
from ._equivalence import (
    _copy_molecules,
    _interplanar_spacings,
    crystal_structure_from_expansion,
    CrystalStructure,
)
from ._input import MolecularComponent
from ._symmetry_basis import (
    atomic_state_from_expansion,
    AtomicSymmetryPlan,
    symmetry_residual,
)


RadialScreenMode: TypeAlias = Literal["untyped", "center-typed", "neighbor-typed"]
"""``untyped`` ignores species; ``center-typed`` compares only rows of equal
centre species; ``neighbor-typed`` additionally separates neighbour ranks by
species."""
RadialCapacity: TypeAlias = Literal["atoms", "images", "neighbors"]
RadialRole: TypeAlias = Literal["representative", "duplicate"]
CrystalRelaxationChartKind: TypeAlias = Literal[
    "variable-cell", "symmetry-atomic", "symmetry-rigid"
]
RankingBasis: TypeAlias = Literal["energy-per-formula-unit", "grand-energy-per-atom"]

RANKING_NONCLAIM = (
    "Static-lattice energy ordering of relaxed structures under one Hamiltonian "
    "definition; it is not finite-temperature polymorph stability (no vibrational, "
    "configurational or thermal-expansion free energy) and not a likelihood."
)

_TYPE_PAD = int(np.iinfo(np.int32).max)


class RadialStructureDim(Dim, minimum=0):
    """Structures of one fingerprint pool."""


class RadialAtomDim(Dim, minimum=1):
    """Retained-atom capacity of one fingerprint."""


class RadialNeighborDim(Dim, minimum=1):
    """Neighbour capacity of one fingerprint row."""


class RadialScreenCapacityError(ValueError):
    """A declared radial-screen capacity would truncate a complete search."""

    capacity: RadialCapacity
    structure_id: str

    def __init__(
        self, capacity: RadialCapacity, structure_id: str, message: str, /
    ) -> None:
        self.capacity = capacity
        self.structure_id = structure_id
        super().__init__(message)


def _positive_int(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


def _finite(value: float, name: str, /) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty string.")
    return value


def _optional_tolerance(value: float | None, name: str, /) -> float | None:
    if value is None:
        return None
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


# -- radial screening ---------------------------------------------------------------


@final
class RadialScreenPolicy(StrictModule):
    """Cutoff, typing mode, hydrogen selection, inclusive threshold and capacities.

    ``cutoff`` is in the structure length unit.  ``max_atoms`` bounds retained
    atoms per structure, ``max_images`` the complete image box and
    ``max_neighbors`` the row length; every bound refuses rather than
    truncates.  Device memory of one fingerprint chunk is
    ``chunk * max_atoms**2 * max_images`` distances.
    """

    cutoff: float = eqx.field(static=True)
    mode: RadialScreenMode = eqx.field(static=True)
    include_hydrogens: bool = eqx.field(static=True)
    threshold: float = eqx.field(static=True)
    max_atoms: int = eqx.field(static=True)
    max_images: int = eqx.field(static=True)
    max_neighbors: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        cutoff: float,
        mode: RadialScreenMode,
        include_hydrogens: bool,
        threshold: float,
        max_atoms: int,
        max_images: int,
        max_neighbors: int,
    ) -> None:
        cutoff_ = _finite(cutoff, "cutoff")
        if cutoff_ <= 0.0:
            raise ValueError("cutoff must be positive.")
        mode_ = parse(mode, RadialScreenMode, "mode")
        if not isinstance(include_hydrogens, bool):
            raise TypeError("include_hydrogens must be a bool.")
        threshold_ = _finite(threshold, "threshold")
        if threshold_ < 0.0:
            raise ValueError("threshold must be nonnegative.")
        atoms = _positive_int(max_atoms, "max_atoms")
        images = _positive_int(max_images, "max_images")
        neighbors = _positive_int(max_neighbors, "max_neighbors")
        self.cutoff = cutoff_
        self.mode = mode_
        self.include_hydrogens = include_hydrogens
        self.threshold = threshold_
        self.max_atoms = atoms
        self.max_images = images
        self.max_neighbors = neighbors
        self.policy_id = canonical_fingerprint(
            {
                "kind": "crystal-radial-screen-policy",
                "cutoff": cutoff_,
                "mode": mode_,
                "include_hydrogens": include_hydrogens,
                "threshold": threshold_,
                "max_atoms": atoms,
                "max_images": images,
                "max_neighbors": neighbors,
            }
        )

    @property
    def match_limit(self) -> float:
        """Inclusive comparison value: ``threshold`` advanced one float64 ulp."""
        return float(np.nextafter(self.threshold, np.inf))


@final
class RadialFingerprints(StrictModule, NonTrainableState):
    """Radial rows of a pool of structures under one policy (host arrays).

    ``distances`` are the ordered rows padded with the cutoff,
    ``neighbor_types`` their neighbour atomic numbers (pads hold the int32
    maximum), ``center_types`` the retained centre atomic numbers,
    ``atom_mask`` the retained slots, ``neighbor_counts`` the row lengths,
    ``source_atoms`` the original structure atom of each slot (``-1`` padded)
    and ``image_bounds`` the exhaustive per-axis image bound ``m_k``.
    """

    __strict_contract__ = True

    distances: HostFloat64[RadialStructureDim, RadialAtomDim, RadialNeighborDim]
    neighbor_types: HostInt32[RadialStructureDim, RadialAtomDim, RadialNeighborDim]
    center_types: HostInt32[RadialStructureDim, RadialAtomDim]
    atom_mask: HostBool[RadialStructureDim, RadialAtomDim]
    neighbor_counts: HostInt32[RadialStructureDim, RadialAtomDim]
    source_atoms: HostInt64[RadialStructureDim, RadialAtomDim]
    image_bounds: HostInt64[RadialStructureDim, Literal[3]]
    policy: RadialScreenPolicy
    structure_ids: tuple[str, ...] = eqx.field(static=True)
    content_ids: tuple[str, ...] = eqx.field(static=True)

    @property
    def structure_count(self) -> int:
        return len(self.structure_ids)


def _fingerprint_rows(
    fractional: Array,
    mask: Array,
    types: Array,
    vectors: Array,
    shifts: Array,
    image_mask: Array,
    /,
    *,
    cutoff: float,
    capacity: int,
    typed: bool,
) -> tuple[Array, Array, Array, Array, Array]:
    """Return ordered rows, neighbour types, centre types, counts and coincidence."""
    atoms = mask.shape[0]
    images = shifts.shape[0]
    cartesian = contract("ai,ij->aj", fractional, vectors, backend="jax")
    translations = contract("mi,ij->mj", shifts, vectors, backend="jax")
    displacement = (
        cartesian[None, :, None, :]
        + translations[None, None, :, :]
        - cartesian[:, None, None, :]
    )
    distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
    zero_image = jnp.all(shifts == 0.0, axis=-1)
    self_pair = jnp.eye(atoms, dtype=jnp.bool_)[:, :, None] & zero_image[None, None, :]
    valid = (
        mask[:, None, None] & mask[None, :, None] & image_mask[None, None, :] & ~self_pair
    )
    coincident = jnp.any(valid & (distance <= 0.0))
    inside = valid & (distance < cutoff)
    counts = jnp.sum(inside, axis=(1, 2), dtype=jnp.int32)
    key = jnp.where(inside, distance, jnp.inf).reshape((atoms, atoms * images))
    if atoms * images < capacity:
        key = jnp.pad(
            key, ((0, 0), (0, capacity - atoms * images)), constant_values=jnp.inf
        )
    negative, index = jax.lax.top_k(-key, capacity)
    nearest = -negative
    present = jnp.isfinite(nearest)
    rows = jnp.where(present, nearest, jnp.asarray(cutoff, dtype=nearest.dtype))
    neighbor = jnp.clip(index // images, 0, atoms - 1)
    neighbor_types = jnp.where(present, types[neighbor], jnp.int32(_TYPE_PAD))
    if typed:
        order = jnp.lexsort((rows, neighbor_types), axis=-1)
        rows = jnp.take_along_axis(rows, order, axis=-1)
        neighbor_types = jnp.take_along_axis(neighbor_types, order, axis=-1)
    centers = jnp.where(mask, types, jnp.int32(_TYPE_PAD))
    return rows, neighbor_types, centers, counts, coincident


@partial(jax.jit, static_argnames=("cutoff", "capacity", "typed"))
def _fingerprint_chunk(
    fractional: Array,
    mask: Array,
    types: Array,
    vectors: Array,
    shifts: Array,
    image_mask: Array,
    *,
    cutoff: float,
    capacity: int,
    typed: bool,
) -> tuple[Array, Array, Array, Array, Array]:
    rows = partial(_fingerprint_rows, cutoff=cutoff, capacity=capacity, typed=typed)
    return jax.vmap(rows)(fractional, mask, types, vectors, shifts, image_mask)


def _image_box(bounds: np.ndarray, /) -> np.ndarray:
    axes = [np.arange(-int(m), int(m) + 1, dtype=np.int64) for m in bounds]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return grid.reshape((-1, 3))


def radial_fingerprints(
    structures: Sequence[CrystalStructure],
    policy: RadialScreenPolicy,
    /,
    *,
    chunk_size: int,
) -> RadialFingerprints:
    """Compute radial rows of every structure in fixed-size device chunks.

    Host preparation removes hydrogens when requested, wraps fractional
    coordinates and enumerates each complete image box; one compiled kernel
    of shape ``(chunk_size, max_atoms, max_images)`` evaluates every chunk,
    so rows do not depend on chunking.
    """
    if not isinstance(policy, RadialScreenPolicy):
        raise TypeError("policy must be a RadialScreenPolicy.")
    chunk = _positive_int(chunk_size, "chunk_size")
    structures_ = tuple(structures)
    if any(not isinstance(structure, CrystalStructure) for structure in structures_):
        raise TypeError("structures must be CrystalStructure values.")
    atoms, images, capacity = policy.max_atoms, policy.max_images, policy.max_neighbors
    count = len(structures_)
    # Lanes are padded to whole chunks; a padding lane has no atoms and an
    # identity cell, so one compiled kernel serves every chunk.
    lanes = -(-count // chunk) * chunk
    fractional = np.zeros((lanes, atoms, 3), dtype=np.float64)
    mask = np.zeros((lanes, atoms), dtype=np.bool_)
    types = np.full((lanes, atoms), _TYPE_PAD, dtype=np.int32)
    source = np.full((count, atoms), -1, dtype=np.int64)
    vectors = np.broadcast_to(np.eye(3), (lanes, 3, 3)).copy()
    shifts = np.zeros((lanes, images, 3), dtype=np.float64)
    image_mask = np.zeros((lanes, images), dtype=np.bool_)
    bounds = np.zeros((count, 3), dtype=np.int64)
    for s, structure in enumerate(structures_):
        keep = (
            np.arange(structure.atom_count)
            if policy.include_hydrogens
            else np.flatnonzero(structure.species != 1)
        )
        if keep.shape[0] > atoms:
            raise RadialScreenCapacityError(
                "atoms",
                structure.structure_id,
                f"Structure {structure.structure_id} retains {keep.shape[0]} atoms, "
                f"above max_atoms={atoms}.",
            )
        heights = _interplanar_spacings(structure.vectors)
        if not np.all(np.isfinite(heights)) or np.any(heights <= 0.0):
            raise ValueError(f"Structure {structure.structure_id} has a degenerate cell.")
        bound = np.floor(policy.cutoff / heights).astype(np.int64) + 1
        box = _image_box(bound)
        if box.shape[0] > images:
            raise RadialScreenCapacityError(
                "images",
                structure.structure_id,
                f"Structure {structure.structure_id} needs {box.shape[0]} images for a "
                f"complete search, above max_images={images}.",
            )
        wrapped = structure.fractional[keep]
        fractional[s, : keep.shape[0]] = wrapped - np.floor(wrapped)
        mask[s, : keep.shape[0]] = True
        types[s, : keep.shape[0]] = structure.species[keep].astype(np.int32)
        source[s, : keep.shape[0]] = keep
        vectors[s] = structure.vectors
        shifts[s, : box.shape[0]] = box
        image_mask[s, : box.shape[0]] = True
        bounds[s] = bound
    rows = np.zeros((count, atoms, capacity), dtype=np.float64)
    neighbor_types = np.zeros((count, atoms, capacity), dtype=np.int32)
    centers = np.zeros((count, atoms), dtype=np.int32)
    counts = np.zeros((count, atoms), dtype=np.int32)
    typed = policy.mode == "neighbor-typed"
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        width = stop - start
        lane = slice(start, start + chunk)
        result = _fingerprint_chunk(
            jnp.asarray(fractional[lane]),
            jnp.asarray(mask[lane]),
            jnp.asarray(types[lane]),
            jnp.asarray(vectors[lane]),
            jnp.asarray(shifts[lane]),
            jnp.asarray(image_mask[lane]),
            cutoff=policy.cutoff,
            capacity=capacity,
            typed=typed,
        )
        block_rows, block_types, block_centers, block_counts, coincident = (
            np.asarray(value) for value in jax.device_get(result)
        )
        for offset in range(width):
            structure = structures_[start + offset]
            if bool(coincident[offset]):
                raise ValueError(
                    f"Structure {structure.structure_id} has coincident distinct atoms."
                )
            if int(np.max(block_counts[offset])) > capacity:
                raise RadialScreenCapacityError(
                    "neighbors",
                    structure.structure_id,
                    f"Structure {structure.structure_id} has a row of "
                    f"{int(np.max(block_counts[offset]))} neighbours, above "
                    f"max_neighbors={capacity}.",
                )
        rows[start:stop] = block_rows[:width]
        neighbor_types[start:stop] = block_types[:width]
        centers[start:stop] = block_centers[:width]
        counts[start:stop] = block_counts[:width]
    retained = mask[:count].copy()
    for array in (rows, neighbor_types, centers, retained, counts, source, bounds):
        array.setflags(write=False)
    return RadialFingerprints(
        rows,
        neighbor_types,
        centers,
        retained,
        counts,
        source,
        bounds,
        policy,
        tuple(structure.structure_id for structure in structures_),
        tuple(structure.content_id for structure in structures_),
    )


def _ratio(left: Array, right: Array, /) -> Array:
    return jnp.maximum(left, right) / jnp.minimum(left, right)


def _typed_row(
    left: Array, left_types: Array, right: Array, right_types: Array, cutoff: Array, /
) -> Array:
    """Row mismatch with ranks counted within neighbour species."""
    positions = jnp.arange(left.shape[0])

    def side(
        kinds: Array, others: Array, other_kinds: Array
    ) -> tuple[Array, Array, Array]:
        rank = positions - jnp.searchsorted(kinds, kinds, side="left")
        first = jnp.searchsorted(other_kinds, kinds, side="left")
        available = jnp.searchsorted(other_kinds, kinds, side="right") - first
        real = kinds != _TYPE_PAD
        matched = real & (rank < available)
        partner = others[jnp.clip(first + rank, 0, others.shape[0] - 1)]
        return real, matched, partner

    real, matched, partner = side(left_types, right, right_types)
    forward = jnp.where(
        real, jnp.where(matched, _ratio(left, partner), _ratio(left, cutoff)), 1.0
    )
    real_right, matched_right, _ = side(right_types, left, left_types)
    backward = jnp.where(real_right & ~matched_right, _ratio(right, cutoff), 1.0)
    return jnp.maximum(jnp.max(forward), jnp.max(backward))


def _pair_ratio(
    left: Array,
    left_types: Array,
    left_centers: Array,
    left_mask: Array,
    right: Array,
    right_types: Array,
    right_centers: Array,
    right_mask: Array,
    /,
    *,
    cutoff: float,
    mode: RadialScreenMode,
) -> Array:
    """Return ``1 + score`` of one structure pair."""
    if mode == "neighbor-typed":
        limit = jnp.asarray(cutoff, dtype=left.dtype)
        rows = jax.vmap(
            jax.vmap(_typed_row, (None, None, 0, 0, None)),
            (0, 0, None, None, None),
        )(left, left_types, right, right_types, limit)
    else:
        rows = jnp.max(_ratio(left[:, None, :], right[None, :, :]), axis=-1)
    if mode != "untyped":
        rows = jnp.where(left_centers[:, None] == right_centers[None, :], rows, jnp.inf)
    rows = jnp.where(left_mask[:, None] & right_mask[None, :], rows, jnp.inf)
    forward = jnp.max(jnp.where(left_mask, jnp.min(rows, axis=1), 1.0))
    backward = jnp.max(jnp.where(right_mask, jnp.min(rows, axis=0), 1.0))
    return jnp.maximum(forward, backward)


@partial(jax.jit, static_argnames=("cutoff", "mode"))
def _score_pairs(
    left: tuple[Array, Array, Array, Array],
    right: tuple[Array, Array, Array, Array],
    left_index: Array,
    right_index: Array,
    *,
    cutoff: float,
    mode: RadialScreenMode,
) -> Array:
    gathered_left = tuple(value[left_index] for value in left)
    gathered_right = tuple(value[right_index] for value in right)
    ratio = jax.vmap(partial(_pair_ratio, cutoff=cutoff, mode=mode))(
        *gathered_left, *gathered_right
    )
    return ratio - 1.0


def _device_pool(
    fingerprints: RadialFingerprints, /
) -> tuple[Array, Array, Array, Array]:
    return (
        jnp.asarray(fingerprints.distances),
        jnp.asarray(fingerprints.neighbor_types),
        jnp.asarray(fingerprints.center_types),
        jnp.asarray(fingerprints.atom_mask),
    )


def _scores(
    policy: RadialScreenPolicy,
    left: tuple[Array, Array, Array, Array],
    right: tuple[Array, Array, Array, Array],
    left_index: np.ndarray,
    right_index: np.ndarray,
    chunk: int,
    /,
) -> np.ndarray:
    total = left_index.shape[0]
    result = np.zeros((total,), dtype=np.float64)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        lanes = np.zeros((2, chunk), dtype=np.int32)
        lanes[0, : stop - start] = left_index[start:stop]
        lanes[1, : stop - start] = right_index[start:stop]
        values = _score_pairs(
            left,
            right,
            jnp.asarray(lanes[0]),
            jnp.asarray(lanes[1]),
            cutoff=policy.cutoff,
            mode=policy.mode,
        )
        result[start:stop] = np.asarray(jax.device_get(values))[: stop - start]
    return result


def radial_scores(
    left: RadialFingerprints,
    right: RadialFingerprints,
    pairs: ArrayLike,
    /,
    *,
    pair_chunk_size: int,
) -> np.ndarray:
    """Return the score of every ``(left, right)`` index pair (``inf`` if incompatible)."""
    if not isinstance(left, RadialFingerprints) or not isinstance(
        right, RadialFingerprints
    ):
        raise TypeError("left and right must be RadialFingerprints.")
    if left.policy.policy_id != right.policy.policy_id:
        raise ValueError("Fingerprints of different radial policies are not comparable.")
    chunk = _positive_int(pair_chunk_size, "pair_chunk_size")
    indices = np.asarray(pairs)
    if indices.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if (
        not np.issubdtype(indices.dtype, np.integer)
        or indices.ndim != 2
        or indices.shape[1] != 2
    ):
        raise ValueError("pairs must be integer indices of shape (pairs, 2).")
    if (
        np.any(indices < 0)
        or np.any(indices[:, 0] >= left.structure_count)
        or np.any(indices[:, 1] >= right.structure_count)
    ):
        raise ValueError("pairs index outside the fingerprint pools.")
    return _scores(
        left.policy,
        _device_pool(left),
        _device_pool(right),
        indices[:, 0].astype(np.int32),
        indices[:, 1].astype(np.int32),
        chunk,
    )


@final
class RadialScreenDecision(StrictModule):
    """Greedy screen decision of one pool structure.

    A duplicate's ``score`` is its score against ``representative_id``; a
    representative's ``score`` is its smallest score against the
    representatives retained before it (``None`` for the first).
    """

    structure_id: str = eqx.field(static=True)
    index: int = eqx.field(static=True)
    role: RadialRole = eqx.field(static=True)
    representative_id: str = eqx.field(static=True)
    score: float | None = eqx.field(static=True)


@final
class RadialScreenResult(StrictModule):
    """Greedy radial screen of one ordered pool (a screen, not an equivalence proof)."""

    decisions: tuple[RadialScreenDecision, ...]
    representative_ids: tuple[str, ...] = eqx.field(static=True)
    pairs_scored: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)


def screen_radial(
    fingerprints: RadialFingerprints, /, *, pair_chunk_size: int
) -> RadialScreenResult:
    """Greedy, order-dependent radial screen of a pool in its given order."""
    if not isinstance(fingerprints, RadialFingerprints):
        raise TypeError("fingerprints must be RadialFingerprints.")
    chunk = _positive_int(pair_chunk_size, "pair_chunk_size")
    policy = fingerprints.policy
    limit = policy.match_limit
    pool = _device_pool(fingerprints)
    retained: list[int] = []
    decisions: list[RadialScreenDecision] = []
    scored = 0
    for candidate, structure_id in enumerate(fingerprints.structure_ids):
        match: int | None = None
        matched_score = math.inf
        nearest: float | None = None
        for start in range(0, len(retained), chunk):
            block = np.asarray(retained[start : start + chunk], dtype=np.int32)
            values = _scores(
                policy,
                pool,
                pool,
                np.full(block.shape, candidate, dtype=np.int32),
                block,
                chunk,
            )
            scored += block.shape[0]
            low = float(np.min(values))
            nearest = low if nearest is None else min(nearest, low)
            hits = np.flatnonzero(values <= limit)
            if hits.size:
                match = int(block[hits[0]])
                matched_score = float(values[hits[0]])
                break
        if match is None:
            retained.append(candidate)
            decisions.append(
                RadialScreenDecision(
                    structure_id, candidate, "representative", structure_id, nearest
                )
            )
        else:
            decisions.append(
                RadialScreenDecision(
                    structure_id,
                    candidate,
                    "duplicate",
                    fingerprints.structure_ids[match],
                    matched_score,
                )
            )
    return RadialScreenResult(
        tuple(decisions),
        tuple(fingerprints.structure_ids[index] for index in retained),
        scored,
        policy.policy_id,
    )


# -- ranking ---------------------------------------------------------------------------


@final
class CrystalRelaxationEvidence(StrictModule):
    """Accepted-iterate relaxation evidence of one ranking candidate.

    ``status`` names the `AtomisticRelaxationStatus`; ``converged`` is the
    declared convergence and ``full_converged`` the full (Cartesian force and,
    for variable cells, stress) residual test.  ``symmetry_deviation`` is the
    largest per-atom distance from the exact images of the source setting
    (zero up to round-off for a symmetry chart; for an unconstrained run it
    measures broken symmetry, and the setting is provenance only).
    ``symmetry_projected_norm`` is the tangent-operator projected residual of
    a symmetry chart.  Residuals are in the Hamiltonian's units; the
    deviation is in angstrom.
    """

    status: str = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    full_converged: bool = eqx.field(static=True)
    evaluation_successful: bool = eqx.field(static=True)
    projected_residual: float = eqx.field(static=True)
    full_force_residual: float = eqx.field(static=True)
    full_stress_residual: float | None = eqx.field(static=True)
    evaluations: int = eqx.field(static=True)
    chart: CrystalRelaxationChartKind = eqx.field(static=True)
    relaxation_id: str = eqx.field(static=True)
    setting_id: str = eqx.field(static=True)
    symmetry_deviation: float = eqx.field(static=True)
    symmetry_projected_norm: float | None = eqx.field(static=True)


def _composition(structure: CrystalStructure, /) -> tuple[tuple[int, int], ...]:
    values, counts = np.unique(structure.species, return_counts=True)
    return tuple((int(v), int(c)) for v, c in zip(values, counts, strict=True))


@final
class CrystalRankingCandidate(StrictModule, NonTrainableState):
    """One structure with its energy under one declared Hamiltonian definition.

    ``energy`` is the total cell energy in ``energy_unit`` (``None`` when no
    accepted evaluation exists).  ``hamiltonian_id`` is the semantic program
    identity and ``numeric_revision_id`` its numeric revision; together they
    define the Hamiltonian.  ``program_prepared_id`` records the resource
    binding as provenance only.  ``formula_units`` divides the cell
    composition into ``formula``.  ``energy_uncertainty`` is a declared
    standard uncertainty of ``energy``.  ``source_structure_id`` and
    ``lineage_id`` keep the pre-relaxation identity.
    """

    structure: CrystalStructure
    energy: float | None = eqx.field(static=True)
    energy_unit: str = eqx.field(static=True)
    formula_units: int = eqx.field(static=True)
    formula: tuple[tuple[int, int], ...] = eqx.field(static=True)
    hamiltonian_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    program_prepared_id: str = eqx.field(static=True)
    ordinal: int = eqx.field(static=True)
    relaxation: CrystalRelaxationEvidence | None
    energy_uncertainty: float | None = eqx.field(static=True)
    source_structure_id: str | None = eqx.field(static=True)
    lineage_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        structure: CrystalStructure,
        /,
        *,
        energy: float | None,
        energy_unit: str,
        formula_units: int,
        hamiltonian_id: str,
        numeric_revision_id: str,
        program_prepared_id: str,
        ordinal: int,
        relaxation: CrystalRelaxationEvidence | None = None,
        energy_uncertainty: float | None = None,
        source_structure_id: str | None = None,
        lineage_id: str | None = None,
    ) -> None:
        if not isinstance(structure, CrystalStructure):
            raise TypeError("structure must be a CrystalStructure.")
        energy_ = None if energy is None else _finite(energy, "energy")
        units = _positive_int(formula_units, "formula_units")
        composition = _composition(structure)
        if any(count % units for _, count in composition):
            raise ValueError("formula_units must divide every element count of the cell.")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("ordinal must be a nonnegative integer.")
        if relaxation is not None and not isinstance(
            relaxation, CrystalRelaxationEvidence
        ):
            raise TypeError("relaxation must be CrystalRelaxationEvidence or None.")
        uncertainty = _optional_tolerance(energy_uncertainty, "energy_uncertainty")
        self.structure = structure
        self.energy = energy_
        self.energy_unit = _identifier(energy_unit, "energy_unit")
        self.formula_units = units
        self.formula = tuple((number, count // units) for number, count in composition)
        self.hamiltonian_id = _identifier(hamiltonian_id, "hamiltonian_id")
        self.numeric_revision_id = _identifier(numeric_revision_id, "numeric_revision_id")
        self.program_prepared_id = _identifier(program_prepared_id, "program_prepared_id")
        self.ordinal = ordinal
        self.relaxation = relaxation
        self.energy_uncertainty = uncertainty
        self.source_structure_id = (
            None
            if source_structure_id is None
            else _identifier(source_structure_id, "source_structure_id")
        )
        self.lineage_id = (
            None if lineage_id is None else _identifier(lineage_id, "lineage_id")
        )

    @property
    def structure_id(self) -> str:
        return self.structure.structure_id


@final
class ChemicalPotentialReference(StrictModule):
    """Declared elemental chemical potentials for comparing compositions.

    The ranking value becomes the grand energy per atom
    ``(E - sum_Z N_Z mu_Z) / N_atoms``; every element present must have a
    potential in ``energy_unit``.  The reference is a declared input, not a
    derived phase-stability statement.
    """

    potentials: tuple[tuple[int, float], ...] = eqx.field(static=True)
    energy_unit: str = eqx.field(static=True)
    name: str = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)

    def __init__(
        self, potentials: Mapping[int, float], /, *, energy_unit: str, name: str
    ) -> None:
        values = []
        for number, value in sorted(dict(potentials).items()):
            if isinstance(number, bool) or not isinstance(number, int) or number < 1:
                raise ValueError(
                    "Chemical potentials are keyed by positive atomic numbers."
                )
            values.append((number, _finite(value, "chemical potential")))
        if not values:
            raise ValueError("A chemical-potential reference needs at least one element.")
        self.potentials = tuple(values)
        self.energy_unit = _identifier(energy_unit, "energy_unit")
        self.name = _identifier(name, "name")
        self.reference_id = canonical_fingerprint(
            {
                "kind": "crystal-chemical-potential-reference",
                "potentials": [[number, value] for number, value in values],
                "energy_unit": self.energy_unit,
                "name": self.name,
            }
        )


@final
class CrystalRankingPolicy(StrictModule):
    """Qualification rules and the optional chemical-potential reference.

    ``require_converged`` excludes candidates without a converged relaxation;
    ``require_full_minimum`` additionally excludes projected-only minima of
    constrained charts; ``maximum_symmetry_deviation`` and
    ``maximum_uncertainty`` (in ranking-value units) are optional bounds.
    """

    require_converged: bool = eqx.field(static=True)
    require_full_minimum: bool = eqx.field(static=True)
    maximum_symmetry_deviation: float | None = eqx.field(static=True)
    maximum_uncertainty: float | None = eqx.field(static=True)
    chemical_potentials: ChemicalPotentialReference | None
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        require_converged: bool = True,
        require_full_minimum: bool = False,
        maximum_symmetry_deviation: float | None = None,
        maximum_uncertainty: float | None = None,
        chemical_potentials: ChemicalPotentialReference | None = None,
    ) -> None:
        if not isinstance(require_converged, bool) or not isinstance(
            require_full_minimum, bool
        ):
            raise TypeError("require_converged and require_full_minimum must be bools.")
        if require_full_minimum and not require_converged:
            raise ValueError("require_full_minimum needs require_converged.")
        if chemical_potentials is not None and not isinstance(
            chemical_potentials, ChemicalPotentialReference
        ):
            raise TypeError(
                "chemical_potentials must be a ChemicalPotentialReference or None."
            )
        symmetry = _optional_tolerance(
            maximum_symmetry_deviation, "maximum_symmetry_deviation"
        )
        uncertainty = _optional_tolerance(maximum_uncertainty, "maximum_uncertainty")
        self.require_converged = require_converged
        self.require_full_minimum = require_full_minimum
        self.maximum_symmetry_deviation = symmetry
        self.maximum_uncertainty = uncertainty
        self.chemical_potentials = chemical_potentials
        self.policy_id = canonical_fingerprint(
            {
                "kind": "crystal-ranking-policy",
                "require_converged": require_converged,
                "require_full_minimum": require_full_minimum,
                "maximum_symmetry_deviation": symmetry,
                "maximum_uncertainty": uncertainty,
                "chemical_potentials": (
                    None
                    if chemical_potentials is None
                    else chemical_potentials.reference_id
                ),
            }
        )


@final
class CrystalRankingEntry(StrictModule):
    """One ranked candidate: 1-based ``rank``, ranking ``value`` and ``relative`` to the best."""

    rank: int = eqx.field(static=True)
    candidate: CrystalRankingCandidate
    value: float = eqx.field(static=True)
    relative: float = eqx.field(static=True)
    uncertainty: float | None = eqx.field(static=True)

    @property
    def structure_id(self) -> str:
        return self.candidate.structure_id


@final
class CrystalRankingExclusion(StrictModule):
    """A candidate left out of the ranking and the recorded reason."""

    structure_id: str = eqx.field(static=True)
    ordinal: int = eqx.field(static=True)
    reason: str = eqx.field(static=True)


@final
class CrystalRanking(StrictModule):
    """Same-definition ranking with exclusions, definition and nonclaim.

    ``basis`` names the ranking value; ``formula`` is the shared formula unit
    (``None`` under a chemical-potential reference).  Ties are broken by
    ``(ordinal, structure_id)``.  ``nonclaim`` states what the ordering is not.
    """

    entries: tuple[CrystalRankingEntry, ...]
    excluded: tuple[CrystalRankingExclusion, ...]
    basis: RankingBasis = eqx.field(static=True)
    hamiltonian_id: str | None = eqx.field(static=True)
    numeric_revision_id: str | None = eqx.field(static=True)
    energy_unit: str | None = eqx.field(static=True)
    formula: tuple[tuple[int, int], ...] | None = eqx.field(static=True)
    reference_id: str | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    nonclaim: str = eqx.field(static=True)


class _Admitted:
    """Host bookkeeping of one admission (never part of a result identity)."""

    def __init__(self) -> None:
        self.qualified: list[tuple[CrystalRankingCandidate, float, float | None]] = []
        self.excluded: list[CrystalRankingExclusion] = []


class _RankingDefinition(NamedTuple):
    """The shared definition every ranked value of one admission is measured in."""

    basis: RankingBasis
    hamiltonian_id: str | None
    numeric_revision_id: str | None
    energy_unit: str | None
    formula: tuple[tuple[int, int], ...] | None
    reference_id: str | None


def _admit(
    candidates: Sequence[CrystalRankingCandidate], policy: CrystalRankingPolicy, /
) -> tuple[_Admitted, _RankingDefinition]:
    if not isinstance(policy, CrystalRankingPolicy):
        raise TypeError("policy must be a CrystalRankingPolicy.")
    candidates_ = tuple(candidates)
    if any(
        not isinstance(candidate, CrystalRankingCandidate) for candidate in candidates_
    ):
        raise TypeError("candidates must be CrystalRankingCandidate values.")
    identities = [candidate.structure_id for candidate in candidates_]
    if len(set(identities)) != len(identities):
        raise ValueError("A ranking names each structure identity once.")
    reference = policy.chemical_potentials
    basis: RankingBasis = (
        "energy-per-formula-unit" if reference is None else "grand-energy-per-atom"
    )
    reference_id = None if reference is None else reference.reference_id
    admitted = _Admitted()
    if not candidates_:
        return admitted, _RankingDefinition(basis, None, None, None, None, reference_id)
    first = candidates_[0]
    for name in ("hamiltonian_id", "numeric_revision_id", "energy_unit"):
        values = {getattr(candidate, name) for candidate in candidates_}
        if len(values) != 1:
            raise ValueError(
                f"Ranking refuses mixed {name} values {sorted(values)}: energies of different "
                "Hamiltonian definitions or units are not comparable."
            )
    potentials: dict[int, float] | None
    formula: tuple[tuple[int, int], ...] | None
    if reference is None:
        potentials = None
        formulas = {candidate.formula for candidate in candidates_}
        if len(formulas) != 1:
            raise ValueError(
                "Ranking refuses different formula-unit compositions without an explicit "
                "ChemicalPotentialReference."
            )
        formula = first.formula
    else:
        reference_potentials = dict(reference.potentials)
        potentials = reference_potentials
        formula = None
        if reference.energy_unit != first.energy_unit:
            raise ValueError("The chemical-potential reference uses another energy unit.")
        missing = {
            number
            for candidate in candidates_
            for number, _ in candidate.formula
            if number not in reference_potentials
        }
        if missing:
            raise ValueError(
                f"The chemical-potential reference lacks elements {sorted(missing)}."
            )
    definition = _RankingDefinition(
        basis,
        first.hamiltonian_id,
        first.numeric_revision_id,
        first.energy_unit,
        formula,
        reference_id,
    )
    for candidate in candidates_:
        reason = _disqualification(candidate, policy)
        if reason is None and candidate.energy is not None:
            if potentials is None:
                divisor = float(candidate.formula_units)
                value = candidate.energy / divisor
            else:
                divisor = float(candidate.structure.atom_count)
                shift = sum(
                    potentials[number] * count * candidate.formula_units
                    for number, count in candidate.formula
                )
                value = (candidate.energy - shift) / divisor
            uncertainty = (
                None
                if candidate.energy_uncertainty is None
                else candidate.energy_uncertainty / divisor
            )
            if policy.maximum_uncertainty is None or (
                uncertainty is not None and uncertainty <= policy.maximum_uncertainty
            ):
                admitted.qualified.append((candidate, value, uncertainty))
                continue
            reason = "uncertainty"
        admitted.excluded.append(
            CrystalRankingExclusion(
                candidate.structure_id, candidate.ordinal, reason or "no-energy"
            )
        )
    return admitted, definition


def _disqualification(
    candidate: CrystalRankingCandidate, policy: CrystalRankingPolicy, /
) -> str | None:
    if candidate.energy is None:
        return "no-energy"
    relaxation = candidate.relaxation
    if relaxation is not None and not relaxation.evaluation_successful:
        return "evaluation-failed"
    if policy.require_converged:
        if relaxation is None:
            return "no-relaxation-evidence"
        if not relaxation.converged:
            return f"not-converged:{relaxation.status}"
        if policy.require_full_minimum and not relaxation.full_converged:
            return "projected-minimum-only"
    if policy.maximum_symmetry_deviation is not None and (
        relaxation is None
        or relaxation.symmetry_deviation > policy.maximum_symmetry_deviation
    ):
        return "symmetry-deviation"
    return None


def _ordered(
    qualified: Sequence[tuple[CrystalRankingCandidate, float, float | None]], /
) -> list[tuple[CrystalRankingCandidate, float, float | None]]:
    return sorted(
        qualified, key=lambda item: (item[1], item[0].ordinal, item[0].structure_id)
    )


def _ranking(
    qualified: Sequence[tuple[CrystalRankingCandidate, float, float | None]],
    excluded: Sequence[CrystalRankingExclusion],
    definition: _RankingDefinition,
    policy: CrystalRankingPolicy,
    /,
) -> CrystalRanking:
    ordered = _ordered(qualified)
    best = ordered[0][1] if ordered else 0.0
    entries = tuple(
        CrystalRankingEntry(rank + 1, candidate, value, value - best, uncertainty)
        for rank, (candidate, value, uncertainty) in enumerate(ordered)
    )
    return CrystalRanking(
        entries,
        tuple(sorted(excluded, key=lambda item: (item.ordinal, item.structure_id))),
        definition.basis,
        definition.hamiltonian_id,
        definition.numeric_revision_id,
        definition.energy_unit,
        definition.formula,
        definition.reference_id,
        policy.policy_id,
        RANKING_NONCLAIM,
    )


def rank_crystal_candidates(
    candidates: Sequence[CrystalRankingCandidate], /, *, policy: CrystalRankingPolicy
) -> CrystalRanking:
    """Rank qualified candidates of one Hamiltonian definition by their declared value.

    Mixed Hamiltonian identities, numeric revisions or energy units are
    refused; different formula-unit compositions are refused unless the
    policy supplies a `ChemicalPotentialReference`.  Candidates failing the
    qualification rules are excluded with their reason.
    """
    admitted, definition = _admit(candidates, policy)
    return _ranking(admitted.qualified, admitted.excluded, definition, policy)


# -- relaxation handoff -------------------------------------------------------------------


@final
class CrystalRelaxationInput(StrictModule):
    """One ASU candidate to relax: state, its expansion and lineage identities."""

    state: AsymmetricUnitState
    expansion: ExpandedCrystal
    structure_id: str = eqx.field(static=True)
    lineage_id: str = eqx.field(static=True)
    ordinal: int = eqx.field(static=True)

    def __init__(
        self,
        state: AsymmetricUnitState,
        expansion: ExpandedCrystal,
        /,
        *,
        structure_id: str,
        lineage_id: str,
        ordinal: int,
    ) -> None:
        if not isinstance(state, AsymmetricUnitState) or not isinstance(
            expansion, ExpandedCrystal
        ):
            raise TypeError(
                "state and expansion must be an AsymmetricUnitState and ExpandedCrystal."
            )
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("ordinal must be a nonnegative integer.")
        self.state = state
        self.expansion = expansion
        self.structure_id = _identifier(structure_id, "structure_id")
        self.lineage_id = _identifier(lineage_id, "lineage_id")
        self.ordinal = ordinal


def _formula_units(plan: AsymmetricUnitPlan, /) -> int:
    """Return ``Z``: the gcd of the integral molecules per cell of every component."""
    counts = []
    for component in plan.components:
        value = plan.molecules_per_cell(component.component_id)
        if value.denominator != 1:
            raise ValueError("Molecules per cell must be integral for a formula unit.")
        counts.append(int(value))
    return math.gcd(*counts)


def _plan_species(plan: AsymmetricUnitPlan, /) -> np.ndarray:
    return np.asarray(
        [
            plan.components[int(component)].atomic_numbers[int(local)]
            for component, local in zip(plan.atom_component, plan.atom_local, strict=True)
        ],
        dtype=np.int64,
    )


def _relaxed_structure(
    plan: AsymmetricUnitPlan,
    molecules: Sequence[tuple[MolecularComponent, tuple[int, ...]]],
    positions: np.ndarray,
    vectors: np.ndarray,
    structure_id: str,
    /,
) -> CrystalStructure:
    fractional = np.linalg.solve(vectors.T, positions.T).T
    return CrystalStructure(
        vectors,
        fractional,
        _plan_species(plan),
        structure_id=structure_id,
        molecules=molecules,
    )


def _angstrom_to_system(
    length_unit: UnitDefinition,
    chart: AtomisticDeformationChart | AtomisticSymmetryChart,
    /,
) -> float:
    """Exact factor converting angstrom crystal geometry to the Hamiltonian's length.

    Refuses a length unit without a declared calibration to the angstrom (for
    example an uncalibrated reduced unit) and a symmetry chart under any
    length unit other than the angstrom, since that chart realizes the
    angstrom-valued ASU geometry of the plan itself.
    """
    try:
        factor = conversion_factor(ANGSTROM, length_unit)
    except ValueError as error:
        raise ValueError(
            "Crystal geometry is angstrom-valued, but the Hamiltonian measures length "
            f"in {length_unit.symbol!r} of reference system "
            f"{length_unit.reference_system_id!r} with no calibration to the "
            "angstrom; declare the Hamiltonian's physical length unit before "
            "relaxing crystal candidates."
        ) from error
    if isinstance(chart, AtomisticSymmetryChart) and factor != 1:
        raise ValueError(
            "An AtomisticSymmetryChart realizes the plan's angstrom-valued ASU "
            "geometry directly, so its Hamiltonian must measure length in angstrom; "
            f"this Hamiltonian uses {length_unit.symbol!r}. Use an angstrom unit "
            "system or an AtomisticDeformationChart, whose handoff converts units."
        )
    return float(factor)


# Retain the single batched handoff: unit conversion, accepted-state selection and
# ranking evidence must describe the same relaxation. Numerical iteration is owned
# by the native relaxation plan, not duplicated in this orchestration boundary.
def relax_crystal_candidates(
    plan: AsymmetricUnitPlan,
    inputs: Sequence[CrystalRelaxationInput],
    program: AbstractPreparedAtomisticHamiltonian,
    chart: AtomisticDeformationChart | AtomisticSymmetryChart,
    /,
    *,
    system_atoms: ArrayLike,
    method: RelaxationMethod,
    convergence: AtomisticRelaxationConvergence,
    maximum_rounds: int,
    target_pressure: float = 0.0,
    maximum_growths: int = 8,
) -> tuple[CrystalRankingCandidate, ...]:
    """Relax every candidate in one batched native relaxation and hand off ranking inputs.

    ``program`` is one prepared Hamiltonian bound to the expanded-atom system;
    expanded atom ``a`` occupies system slot ``system_atoms[a]``.  The chart
    choice is explicit: an `AtomisticDeformationChart` relaxes positions and
    cell without symmetry (the source setting stays provenance and the
    post-relaxation symmetry deviation is measured), an `AtomisticSymmetryChart`
    over this plan keeps the symmetry and reports projected and full
    residuals.  Energies come from ``accepted_evaluation`` of the same
    program; each relaxed structure gets a deterministic identity derived
    from its source identity and the relaxation definition.

    Crystal geometry is angstrom-valued.  Under an `AtomisticDeformationChart`
    the start positions and cell are converted exactly to the Hamiltonian's
    length unit and the relaxed geometry back to angstrom; an
    `AtomisticSymmetryChart` requires an angstrom Hamiltonian.  A length unit
    without a calibration to the angstrom is refused before any numerical
    work.  Residuals, ``target_pressure`` and the convergence policy are in
    the Hamiltonian's units.
    """
    if not isinstance(plan, AsymmetricUnitPlan):
        raise TypeError("plan must be an AsymmetricUnitPlan.")
    inputs_ = tuple(inputs)
    if not inputs_:
        raise ValueError("relax_crystal_candidates needs at least one input.")
    if any(not isinstance(item, CrystalRelaxationInput) for item in inputs_):
        raise TypeError("inputs must be CrystalRelaxationInput values.")
    if any(item.expansion.plan_id != plan.plan_id for item in inputs_):
        raise ValueError("Every input expansion must belong to the plan.")
    identities = [item.structure_id for item in inputs_]
    if len(set(identities)) != len(identities):
        raise ValueError("Relaxation inputs name each structure identity once.")
    if not isinstance(program, AbstractPreparedAtomisticHamiltonian):
        raise TypeError("program must be a prepared atomistic Hamiltonian.")
    atoms = np.asarray(system_atoms)
    if not np.issubdtype(atoms.dtype, np.integer) or atoms.shape != (plan.atom_count,):
        raise ValueError(f"system_atoms must be {plan.atom_count} integer system slots.")
    system = program.system
    active = np.flatnonzero(np.asarray(system.active_mask, dtype=np.bool_))
    if np.unique(atoms).shape[0] != atoms.shape[0] or not np.array_equal(
        np.sort(atoms), active
    ):
        raise ValueError("system_atoms must map every active system atom exactly once.")
    if not np.array_equal(
        np.asarray(system.plan.atomic_numbers)[atoms], _plan_species(plan)
    ):
        raise ValueError("Expanded atoms and system atoms differ in element.")
    if isinstance(chart, AtomisticSymmetryChart):
        if chart.layout.unit.plan_id != plan.plan_id:
            raise ValueError("The symmetry chart belongs to another plan.")
        if not np.array_equal(chart.system_atoms, atoms):
            raise ValueError(
                "The symmetry chart maps expanded atoms to other system slots."
            )
        kind: CrystalRelaxationChartKind = (
            "symmetry-atomic" if chart.atomic else "symmetry-rigid"
        )
    elif isinstance(chart, AtomisticDeformationChart):
        kind = "variable-cell"
    else:
        raise TypeError(
            "chart must be an AtomisticDeformationChart or AtomisticSymmetryChart."
        )
    length_unit = system.plan.units.scale.length_unit
    to_system = _angstrom_to_system(length_unit, chart)
    groups = len(inputs_)
    relaxation = AtomisticRelaxationPlan(
        program,
        chart,
        method=method,
        convergence=convergence,
        target_pressure=target_pressure,
        group_count=groups,
    ).prepare()
    if isinstance(chart, AtomisticSymmetryChart):
        layout = chart.layout
        if isinstance(layout, AtomicSymmetryPlan):
            states = [
                atomic_state_from_expansion(layout, item.state, item.expansion)
                for item in inputs_
            ]
        else:
            states = [item.state for item in inputs_]
        start = relaxation.init(
            chart_states=jax.tree.map(lambda *leaves: jnp.stack(leaves), *states)
        )
    else:
        dtype = np.dtype(system.plan.coordinate_dtype)
        positions = np.zeros((groups, system.capacity, 3), dtype=dtype)
        vectors = np.zeros((groups, 3, 3), dtype=dtype)
        for group, item in enumerate(inputs_):
            positions[group, atoms] = to_system * np.asarray(
                jax.device_get(item.expansion.positions), dtype=np.float64
            )
            vectors[group] = to_system * np.asarray(
                jax.device_get(item.expansion.cell.vectors), dtype=np.float64
            )
        start = relaxation.init(positions=positions, cell_vectors=vectors)
    runtime, final = relaxation.solve(
        start, maximum_rounds=maximum_rounds, maximum_growths=maximum_growths
    )
    revision = atomistic_hamiltonian_revision(program)
    energy_unit = system.plan.units.scale.energy_unit.unit_id
    formula_units = _formula_units(plan)
    molecules = _copy_molecules(plan)
    chart_layout = chart.layout if isinstance(chart, AtomisticSymmetryChart) else None
    atomic_plan = (
        chart_layout
        if isinstance(chart_layout, AtomicSymmetryPlan)
        else AtomicSymmetryPlan(plan)
    )
    definition_id = canonical_fingerprint(
        {
            "kind": "crystal-relaxation-definition",
            "hamiltonian": revision.semantic_id,
            "numeric_revision": revision.revision_id,
            "chart": chart.chart_id,
            "method": _method_id(method),
            "convergence": convergence.convergence_id,
            "target_pressure": float(target_pressure),
            "length_unit": length_unit.unit_id,
        }
    )
    status = np.asarray(final.status)
    accepted = np.asarray(final.optimizer.has_accepted)
    projected = np.asarray(final.projected_residual)
    full_force = np.asarray(final.full_force_residual)
    full_stress = np.asarray(final.full_stress_residual)
    full_converged = np.asarray(final.full_converged)
    evaluations = np.asarray(final.evaluations)
    candidates = []
    for group, item in enumerate(inputs_):
        relaxed_id = canonical_fingerprint(
            {
                "kind": "crystal-relaxed-structure",
                "source": item.structure_id,
                "lineage": item.lineage_id,
                "relaxation": definition_id,
            }
        )
        energy: float | None = None
        successful = False
        projected_norm: float | None = None
        if bool(accepted[group]):
            evaluation = runtime.accepted_evaluation(final, group)
            energy = float(
                np.asarray(jax.device_get(evaluation.energy), dtype=np.float64).reshape(
                    ()
                )
            )
            successful = bool(
                np.all(np.asarray(evaluation.successful))
            ) and math.isfinite(energy)
            if final.cell_vectors is None:
                raise RuntimeError("Crystal relaxation lost its accepted periodic cell.")
            realized = final.positions[group]
            cell = final.cell_vectors[group]
            expanded = (
                np.asarray(jax.device_get(realized), dtype=np.float64)[atoms] / to_system
            )
            vectors_ = np.asarray(jax.device_get(cell), dtype=np.float64) / to_system
            structure = _relaxed_structure(
                plan, molecules, expanded, vectors_, relaxed_id
            )
            if isinstance(chart, AtomisticSymmetryChart):
                evidence = runtime.symmetry_evidence(final, group)
                deviation = float(jnp.max(evidence.deviation))
                projected_norm = float(evidence.residual.projected_norm)
            else:
                deviation = float(
                    jnp.max(
                        symmetry_residual(
                            atomic_plan, jnp.asarray(expanded), jnp.asarray(vectors_)
                        )
                    )
                )
            if not successful:
                energy = None
        else:
            structure = crystal_structure_from_expansion(
                plan, item.expansion, structure_id=relaxed_id
            )
            deviation = float(
                jnp.max(
                    symmetry_residual(
                        atomic_plan, item.expansion.positions, item.expansion.cell.vectors
                    )
                )
            )
        evidence_ = CrystalRelaxationEvidence(
            AtomisticRelaxationStatus(int(status[group])).name.lower(),
            int(status[group]) == int(AtomisticRelaxationStatus.CONVERGED),
            bool(full_converged[group]),
            successful,
            float(projected[group]),
            float(full_force[group]),
            float(full_stress[group]) if chart.variable_cell else None,
            int(evaluations[group]),
            kind,
            definition_id,
            plan.setting.setting_id,
            deviation,
            projected_norm,
        )
        candidates.append(
            CrystalRankingCandidate(
                structure,
                energy=energy,
                energy_unit=energy_unit,
                formula_units=formula_units,
                hamiltonian_id=revision.semantic_id,
                numeric_revision_id=revision.revision_id,
                program_prepared_id=program.prepared_id,
                ordinal=item.ordinal,
                relaxation=evidence_,
                source_structure_id=item.structure_id,
                lineage_id=item.lineage_id,
            )
        )
    return tuple(candidates)


# -- composition --------------------------------------------------------------------------


@final
class CrystalScreeningRecord(StrictModule):
    """Radial screen and certified bank outcome of one qualified candidate.

    ``certified_representative_id`` is the bank's greedy representative or
    the representative of the candidate's component; ``comparisons`` counts
    the certified comparisons recorded for its decision.
    """

    structure_id: str = eqx.field(static=True)
    radial_role: RadialRole = eqx.field(static=True)
    radial_representative_id: str = eqx.field(static=True)
    radial_score: float | None = eqx.field(static=True)
    certified_role: str = eqx.field(static=True)
    certified_representative_id: str = eqx.field(static=True)
    comparisons: int = eqx.field(static=True)


@final
class CrystalScreeningOutcome(StrictModule):
    """Radial screen, certified bank submission and ranking of confirmed representatives.

    ``screen_misses`` lists ``(structure_id, representative_id)`` pairs the
    bank certified as equivalent although the radial screen kept the
    structure as a representative; ``unconfirmed_proposals`` lists radial
    duplicates ``(structure_id, proposed_id)`` whose certified representative
    is not the certified representative of the proposed one (a radial false
    positive or an ``UNDETERMINED`` comparison, which never merges).
    ``certified_comparisons`` counts certified comparisons of this submission.
    """

    screen: RadialScreenResult
    submission: DedupSubmission
    ranking: CrystalRanking
    records: tuple[CrystalScreeningRecord, ...]
    screen_misses: tuple[tuple[str, str], ...] = eqx.field(static=True)
    unconfirmed_proposals: tuple[tuple[str, str], ...] = eqx.field(static=True)
    certified_comparisons: int = eqx.field(static=True)


def screen_dedup_rank(
    candidates: Sequence[CrystalRankingCandidate],
    bank: DedupBank,
    /,
    *,
    screen: RadialScreenPolicy,
    ranking: CrystalRankingPolicy,
    fingerprint_chunk_size: int,
    pair_chunk_size: int,
) -> CrystalScreeningOutcome:
    """Radial screen, certified persistent dedup and ranking of confirmed representatives.

    Candidates are admitted under the ranking rules (mixed definitions are
    refused, unqualified candidates excluded), ordered by the declared
    priority ``(value, ordinal, structure_id)``, screened radially in that
    order and submitted to the bank with the ranking value as ascending
    priority.  Only the certified bank decision merges structures
    (``UNDETERMINED`` never merges); greedy representatives or component
    representatives of this submission are ranked and every other qualified
    candidate is excluded as ``duplicate-of:<id>``.  One bank must hold one
    ranking definition; its priorities are the ranking values.
    """
    if not isinstance(bank, DedupBank):
        raise TypeError("bank must be a DedupBank.")
    if not isinstance(screen, RadialScreenPolicy):
        raise TypeError("screen must be a RadialScreenPolicy.")
    if bank.policy.priority_order != "ascending":
        raise ValueError("Energy priorities need an ascending dedup bank.")
    admitted, definition = _admit(candidates, ranking)
    ordered = _ordered(admitted.qualified)
    fingerprints = radial_fingerprints(
        [candidate.structure for candidate, _, _ in ordered],
        screen,
        chunk_size=fingerprint_chunk_size,
    )
    radial = screen_radial(fingerprints, pair_chunk_size=pair_chunk_size)
    submission = submit_to_dedup_bank(
        bank,
        [
            DedupCandidate(candidate.structure, priority=value, ordinal=candidate.ordinal)
            for candidate, value, _ in ordered
        ],
    )
    decisions = {decision.structure_id: decision for decision in submission.decisions}
    if submission.bank.policy.mode == "greedy":
        representative_of = {
            identity: decision.representative_id
            for identity, decision in decisions.items()
        }
    else:
        representative_of = {}
        for _, representative, members in submission.bank.components():
            for member in members:
                representative_of[member] = representative
    ranked = []
    excluded = list(admitted.excluded)
    records = []
    misses = []
    unconfirmed = []
    for (candidate, value, uncertainty), radial_decision in zip(
        ordered, radial.decisions, strict=True
    ):
        identity = candidate.structure_id
        decision = decisions[identity]
        representative = representative_of[identity]
        if representative == identity:
            ranked.append((candidate, value, uncertainty))
        else:
            excluded.append(
                CrystalRankingExclusion(
                    identity, candidate.ordinal, f"duplicate-of:{representative}"
                )
            )
        if radial_decision.role == "representative" and representative != identity:
            misses.append((identity, representative))
        if (
            radial_decision.role == "duplicate"
            and representative != representative_of[radial_decision.representative_id]
        ):
            unconfirmed.append((identity, radial_decision.representative_id))
        records.append(
            CrystalScreeningRecord(
                identity,
                radial_decision.role,
                radial_decision.representative_id,
                radial_decision.score,
                decision.role,
                representative,
                len(decision.comparisons),
            )
        )
    head = submission.bank.head.sequence
    certified = sum(
        len(decision.comparisons)
        for decision in submission.decisions
        if submission.published and decision.sequence == head
    )
    return CrystalScreeningOutcome(
        radial,
        submission,
        _ranking(ranked, excluded, definition, ranking),
        tuple(records),
        tuple(misses),
        tuple(unconfirmed),
        certified,
    )


__all__ = [
    "ChemicalPotentialReference",
    "CrystalRanking",
    "CrystalRankingCandidate",
    "CrystalRankingEntry",
    "CrystalRankingExclusion",
    "CrystalRankingPolicy",
    "CrystalRelaxationChartKind",
    "CrystalRelaxationEvidence",
    "CrystalRelaxationInput",
    "CrystalScreeningOutcome",
    "CrystalScreeningRecord",
    "RadialFingerprints",
    "RadialScreenCapacityError",
    "RadialScreenDecision",
    "RadialScreenMode",
    "RadialScreenPolicy",
    "RadialScreenResult",
    "RANKING_NONCLAIM",
    "rank_crystal_candidates",
    "radial_fingerprints",
    "radial_scores",
    "relax_crystal_candidates",
    "screen_dedup_rank",
    "screen_radial",
]
