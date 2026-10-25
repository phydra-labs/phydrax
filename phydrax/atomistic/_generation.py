#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native atomistic generation: support, sampler lanes, filters and bounded streams.

This module composes generation; it owns no flow, diffusion, packing or
optimization solver.  Its four layers are:

* ``AtomisticGenerationSupport``: the static chemistry/cell/representation ABI
  of generated proposals (fixed chemistry or a bounded composition over an
  explicit element vocabulary; nonperiodic, fixed-cell or variable-cell
  periodic support; units, masses, charge and topology admission and the
  typed `AtomisticConditioningSpec` of its condition ports).
* ``AbstractPreparedAtomisticSampler``: a native array-bearing lane transition
  ``vacant -> admit -> advance* -> terminal -> emit`` with declared method
  state, resources and numeric revision.  One-shot samplers are terminal after
  ``admit``; stateful samplers (packers, iterative solvers) retain their whole
  continuation in the lane state.  Keys and ordinals are supplied by the
  caller, never derived from lane or completion order.  `map_admitted_lanes`
  evaluates per-lane work for admitted lanes only, in bounded vmapped chunks.
* typed condition schedules/transforms and pure proposal filters: every
  rejection keeps its reason bits and the proposal itself.  Condition sources
  and transforms declare semantic identities (stream identity) and numeric
  revisions (continuation identity), like samplers.
* ``AtomisticGenerationStreamPlan``: a bounded, resumable stream over one
  sampler or an ordinal-routed family of samplers.  Free lanes take the next
  ordinals in order (bounded by the pending bank); keys are derived from
  ``(campaign, support, sampler, ordinal, role)``; terminal proposals wait in
  a bounded pending bank addressed by ``ordinal mod capacity`` and publish in
  ordinal order below a durable watermark.  The complete state (cursor,
  lanes, bank, receipt chain and root key) is one array-bearing PyTree,
  persisted through the typed array-artifact owner.  Unlimited source-style
  iteration is an explicit chain of finite continuation segments (`extend`);
  a reached target stops admission but retains every in-flight lane and
  pending result, so a later segment continues them exactly.
* placement: ``prepare(placement=AtomisticGenerationPlacement(group, ...))``
  runs the same scheduler over the participants of an execution group.  One
  global ordinal window, accepted quota, attempt budget and publication
  authority remain; every participant holds the replicated cursor, pending
  bank, root key and receipt chain plus its own lanes.  Each transition is
  one fixed-shape exchange of harvested bank rows, free lanes and cursor
  consensus; every participant inserts all rows (each slot was reserved by
  admission, which is the credit of the bounded bank) and derives the same
  allocation of the next ordinals to ``(participant, member, lane)``.  The
  authority's sink receipt is agreed through one exchange before every
  participant commits.  A placed continuation persists through the lifecycle
  distributed commit (`publish_checkpoint` / `restore_checkpoint`; same stream
  profile and placement on restore).  Without a placement the stream is its
  own single participant of exactly this scheduler.

Generation pseudotime is never physical time, and a generated geometry is a
proposal: geometric screening and filter acceptance are not physical
qualification, likelihood or Boltzmann weight.
"""

from __future__ import annotations

import abc
import functools
import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from enum import IntEnum, IntFlag, StrEnum
from os import PathLike
from pathlib import Path
from typing import Any, assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from .._fingerprint import canonical_fingerprint, canonical_json
from .._identity import callable_payload, IdentityWords, NumericRevision
from .._jax_config import x64_enabled
from .._sampling._addressing import derive_key, SampleAddress
from .._strict import StrictModule
from .._trainable import _hidden_arrays, ExplicitFreeze, NonTrainableState
from ..ein import contract
from ..lifecycle._array_artifact import (
    ArrayArtifactProvenance,
    ArrayArtifactReceipt,
    read_typed_array_artifact,
    write_typed_array_artifact,
)
from ..lifecycle._chunk_repository import ArtifactRepository, ChunkEncoding
from ..lifecycle._models import CheckpointManifest
from ..linalg import SmallLinearSolvePlan, solve_small_linear
from ..typing import parse, PRNGKey
from ..units import UnitDefinition
from ._conditioning import AtomisticConditioning, AtomisticConditioningSpec
from ._generation_distribution import (
    AtomisticGenerationCapacityError,
    AtomisticGenerationDivergenceError,
    AtomisticGenerationPlacement,
    publish_placement_checkpoint,
    restore_placement_checkpoint,
)
from ._structure_store import (
    AtomisticStoreCommit,
    AtomisticStoreLimits,
    AtomisticStructureRecord,
    AtomisticStructureSource,
    AtomisticStructureStoreSchema,
    AtomisticStructureStoreWriter,
    open_atomistic_structure_store,
)


GENERATION_ADDRESS_NAMESPACE = "phydrax.atomistic.generation"
_SMALL3 = SmallLinearSolvePlan(3)
_WORD = 2**32
_MAXIMUM_FILTERS = 30


class AtomisticChemistryAdmission(StrEnum):
    """Whether generation keeps chemistry fixed or proposes a bounded composition."""

    FIXED = "fixed"
    VARIABLE_COMPOSITION = "variable-composition"


class AtomisticCellAdmission(StrEnum):
    """Periodic support of generated proposals."""

    NONPERIODIC = "nonperiodic"
    FIXED_CELL = "fixed-cell"
    VARIABLE_CELL = "variable-cell"


class AtomisticCoordinateRepresentation(StrEnum):
    """Primary coordinate representation of a sampler state.

    ``FRACTIONAL`` coordinates live on the unit flat torus with an explicit
    cell; ``CARTESIAN`` coordinates are nonperiodic.  Emitted blocks always
    carry Cartesian positions; periodic blocks also carry their row cell.
    """

    CARTESIAN = "cartesian"
    FRACTIONAL = "fractional"


class AtomisticTopologyAdmission(StrEnum):
    """Bond/molecule identity of generated proposals.

    ``FIXED`` binds a declared topology to fixed chemistry.  ``NOT_INFERRED``
    proposals carry no bonds, charge state, stereochemistry or molecule
    identity; any such claim needs an explicit qualifier filter.
    """

    FIXED = "fixed"
    NOT_INFERRED = "not-inferred"


class AtomisticProposalStatus(IntFlag):
    """Typed rejection bits of one proposal; ``ACCEPTED`` is zero."""

    ACCEPTED = 0
    SAMPLER_FAILURE = 1
    NONFINITE = 2
    CONDITION_INVALID = 4
    DECODER_INVALID = 8
    CELL_INVALID = 16
    STEP_BUDGET_EXHAUSTED = 32
    CAPACITY_REFUSED = 64
    SPECIES_INVALID = 128
    COMPOSITION_INVALID = 256
    FILTER_REJECTED = 512
    CHEMISTRY_UNQUALIFIED = 1024
    GEOMETRY_INVALID = 2048


class AtomisticGenerationStopReason(IntEnum):
    """Terminal reason of a stream continuation segment."""

    RUNNING = 0
    TARGET_REACHED = 1
    BUDGET_EXHAUSTED = 2
    ZERO_TARGET = 3
    ZERO_BUDGET = 4


ConditionSchedulePolicy: TypeAlias = Literal["cycle", "bounded"]


def _nonempty(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _positive_int(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if int(value) < 1:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _nonnegative_int(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if int(value) < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return int(value)


def _positive_float(value: float, name: str, /) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _require_x64() -> None:
    if not x64_enabled():
        raise ValueError(
            "Atomistic generation streams address proposals by 64-bit ordinals; "
            "enable jax_enable_x64."
        )


def _tree_where(mask: Array, new: Any, old: Any, /) -> Any:
    def select(left: Array, right: Array) -> Array:
        shape = mask.shape + (1,) * (jnp.ndim(left) - mask.ndim)
        return jnp.where(jnp.reshape(mask, shape), left, right)

    return jax.tree.map(select, new, old)


def _require_visible_numeric_state(function: Callable[..., Any], context: str, /) -> None:
    """Refuse an opaque callable whose numerical state is not visible PyTree leaves.

    The canonical hidden-state walk (``_hidden_arrays`` with ``eqx.is_array``,
    the walk `callable_payload` applies to plain functions) finds every array
    of any dtype, including integer tables and typed PRNG keys, hidden in
    closure cells, defaults, read globals, partial arguments, bound instances
    or static fields.  The walk exempts a visible `ExplicitFreeze` subtree,
    whose hidden state no numeric revision binds, so such holders are refused
    as opaque callables.  Declared IDs never authorize hidden arrays.
    """
    found = _hidden_arrays(function, eqx.is_array)
    if found:
        routes = "; ".join(
            f"{path or '<callable>'}: {' -> '.join(trail)}" for path, trail in found
        )
        raise ValueError(
            f"{context} hides numerical arrays that are not visible PyTree leaves "
            f"({routes}); hold them as dynamic fields of a StrictModule or "
            "PyTree callable."
        )
    holders = jax.tree.leaves(
        function, is_leaf=lambda node: isinstance(node, ExplicitFreeze)
    )
    if any(isinstance(node, ExplicitFreeze) for node in holders):
        raise ValueError(
            f"{context} contains an ExplicitFreeze holder; its hidden state cannot be "
            "bound by a numeric revision."
        )


def _callable_identity(
    function: Callable[..., Any],
    semantic_id: str | None,
    numeric_id: str | None,
    /,
    *,
    context: str,
) -> tuple[str, IdentityWords | None]:
    """Semantic content ID of a bound callable and its declared numeric words.

    `callable_payload` owns the identity: StrictModule callables and plain
    stateless module functions are content-addressed (its payload refuses
    hidden arrays of either) and take no declared IDs; every other callable is
    opaque, declares both identities and must expose all numerical state as
    visible PyTree leaves (`_require_visible_numeric_state`).  The declared
    numeric ID is returned as dynamic `IdentityWords` (None when
    content-addressed) so a changed declaration binds the numeric revision
    without entering any executable's static structure.
    """
    payload = callable_payload(function, semantic_id=semantic_id, numeric_id=numeric_id)
    if semantic_id is None and numeric_id is None:
        return payload["semantic_content_id"], None
    _require_visible_numeric_state(function, context)
    return payload["semantic_content_id"], IdentityWords.of(
        {
            "kind": "declared-callable-numeric-content",
            "semantic_id": payload["semantic_content_id"],
            "numeric_id": payload["numeric_content_id"],
        }
    )


def _callable_numeric_content(
    function: Callable[..., Any],
    declared: IdentityWords | None,
    /,
    *,
    context: str,
) -> dict[str, Any]:
    """Current numeric content of a bound callable, for a `NumericRevision`.

    A content-addressed callable contributes its `callable_payload` numeric ID:
    a StrictModule's structure and every array it holds, or nothing for a plain
    function.  An opaque callable is re-admitted and contributes its declared
    numeric words and every array leaf it exposes (floating, integer, boolean
    or typed PRNG keys, encoded by the `NumericRevision` payload owner), so
    replacing visible state changes the revision even under an unchanged
    declaration.
    """
    if declared is None:
        return {"content": callable_payload(function)["numeric_content_id"]}
    _require_visible_numeric_state(function, context)
    return {
        "declared": declared.words,
        "leaves": tuple(jax.tree.leaves(eqx.filter(function, eqx.is_array))),
    }


def ordinal_words(ordinals: ArrayLike, /) -> tuple[Array, Array]:
    """Exact ``(high, low)`` uint32 words of nonnegative int64 ordinals."""
    value = jnp.asarray(ordinals)
    if not jnp.issubdtype(value.dtype, jnp.integer):
        raise TypeError("ordinals must be integers.")
    value = value.astype(jnp.int64)
    return (
        jnp.right_shift(value, 32).astype(jnp.uint32),
        jnp.bitwise_and(value, _WORD - 1).astype(jnp.uint32),
    )


def atomistic_proposal_id(
    campaign_id: str, support_id: str, sampler_id: str, ordinal: int, /
) -> str:
    """Stable semantic identity of one generated proposal.

    Shared with the campaign owner: a proposal is the same object wherever,
    and in whichever lane, chunk, rank or completion order, it is computed.
    """
    value = _nonnegative_int(ordinal, "ordinal")
    return canonical_fingerprint(
        {
            "kind": "atomistic-generated-proposal",
            "campaign_id": _nonempty(campaign_id, "campaign_id"),
            "support_id": _nonempty(support_id, "support_id"),
            "sampler_id": _nonempty(sampler_id, "sampler_id"),
            "ordinal": value,
        }
    )


def proposal_address(
    campaign_id: str, support_id: str, sampler_id: str, /, *, role: str
) -> SampleAddress:
    """Semantic RNG address of one proposal role."""
    return SampleAddress(
        GENERATION_ADDRESS_NAMESPACE,
        "proposal",
        target=(campaign_id, support_id, sampler_id),
        role=role,
    )


def proposal_keys(
    root_key: PRNGKey,
    address: SampleAddress,
    ordinals: ArrayLike,
    /,
) -> Array:
    """Keys ``derive_key(root, address, ordinal_hi, ordinal_lo)`` per ordinal."""
    high, low = ordinal_words(ordinals)
    return jax.vmap(lambda hi, lo: derive_key(root_key, address, hi, lo))(high, low)


def role_keys(keys: Array, role: str, /) -> Array:
    """Independent per-lane substreams of one sampler role, derived from lane keys."""
    address = SampleAddress(GENERATION_ADDRESS_NAMESPACE, "sampler-role", role=role)
    return jax.vmap(lambda key: derive_key(key, address))(keys)


def map_admitted_lanes(
    function: Callable[..., Any],
    selected: ArrayLike,
    inputs: Sequence[Any],
    vacant: Any,
    /,
    *,
    lane_chunk: int,
) -> Any:
    """Evaluate ``function`` on admitted lanes only, in bounded vmapped chunks.

    ``inputs`` are lane-major PyTrees (leading axis ``lanes``) and
    ``function(*lane_inputs)`` returns one lane's output PyTree; ``vacant`` is
    the lane-major output whose rows of unselected lanes are returned
    unchanged.  Admitted lanes are compacted in lane order and evaluated by a
    ``lax.while_loop`` over ``ceil(admitted / lane_chunk)`` chunks of
    ``min(lane_chunk, lanes)`` vmapped lanes, so resident work is bounded by
    the chunk and zero admitted lanes run nothing.  Inputs of unselected lanes
    are never read; padding slots of the last chunk repeat the first admitted
    lane and are discarded.  A lane's output therefore depends only on its own
    inputs, never on its lane position, chunk size or chunk mates.
    """
    mask = jnp.asarray(selected, dtype=jnp.bool_)
    if mask.ndim != 1:
        raise ValueError("selected must hold one admission flag per lane.")
    lanes = int(mask.shape[0])
    chunk = min(_positive_int(lane_chunk, "lane_chunk"), lanes)
    arguments = tuple(inputs)
    for leaf in jax.tree.leaves((arguments, vacant)):
        if jnp.shape(leaf)[:1] != (lanes,):
            raise ValueError("Admitted-lane inputs and outputs must be lane-major.")
    count = jnp.sum(mask, dtype=jnp.int32)
    padded = -(-lanes // chunk) * chunk
    order = jnp.argsort((~mask).astype(jnp.int32), stable=True).astype(jnp.int32)
    order = jnp.concatenate((order, jnp.zeros((padded - lanes,), dtype=jnp.int32)))
    offsets = jnp.arange(chunk, dtype=jnp.int32)

    def remaining(carry: tuple[Array, Any]) -> Array:
        return carry[0] * chunk < count

    def evaluate(carry: tuple[Array, Any]) -> tuple[Array, Any]:
        index, outputs = carry
        position = index * chunk + offsets
        admitted = position < count
        lane = jnp.where(admitted, order[position], order[0])
        values = jax.vmap(function)(*jax.tree.map(lambda value: value[lane], arguments))
        target = jnp.where(admitted, lane, lanes)
        outputs = jax.tree.map(
            lambda output, value: output.at[target].set(
                value.astype(output.dtype), mode="drop"
            ),
            outputs,
            values,
        )
        return index + 1, outputs

    _, outputs = jax.lax.while_loop(
        remaining, evaluate, (jnp.asarray(0, dtype=jnp.int32), vacant)
    )
    return outputs


# -- support ------------------------------------------------------------------------------


def cell_heights(cells: ArrayLike, /) -> tuple[Array, Array]:
    """Volume ``|a . (b x c)|`` and perpendicular heights of ``(..., 3, 3)`` row cells."""
    vectors = jnp.asarray(cells)
    if vectors.shape[-2:] != (3, 3):
        raise ValueError("cells must end in (3, 3).")
    a, b, c = vectors[..., 0, :], vectors[..., 1, :], vectors[..., 2, :]
    normals = jnp.stack((jnp.cross(b, c), jnp.cross(c, a), jnp.cross(a, b)), axis=-2)
    volume = jnp.abs(jnp.sum(a * normals[..., 0, :], axis=-1))
    areas = jnp.sqrt(jnp.sum(normals * normals, axis=-1))
    heights = volume[..., None] / jnp.maximum(areas, jnp.finfo(vectors.dtype).tiny)
    return volume, heights


@final
class AtomisticCellBounds(StrictModule, NonTrainableState):
    """Explicit positive-volume cell admission of periodic proposals.

    ``minimum_height`` bounds every perpendicular lattice height
    ``V / |a_j x a_k|``; ``maximum_condition`` bounds the native small-solve
    condition estimate of the row cell; volumes are in cubed length units.
    """

    minimum_height: float = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    minimum_volume: float = eqx.field(static=True)
    maximum_volume: float = eqx.field(static=True)
    bounds_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_height: float,
        maximum_condition: float,
        minimum_volume: float,
        maximum_volume: float,
    ) -> None:
        height = _positive_float(minimum_height, "minimum_height")
        condition = float(maximum_condition)
        if not np.isfinite(condition) or condition < 1.0:
            raise ValueError("maximum_condition must be finite and at least one.")
        lower = _positive_float(minimum_volume, "minimum_volume")
        upper = _positive_float(maximum_volume, "maximum_volume")
        if upper < lower:
            raise ValueError("maximum_volume must not be below minimum_volume.")
        self.minimum_height = height
        self.maximum_condition = condition
        self.minimum_volume = lower
        self.maximum_volume = upper
        self.bounds_id = canonical_fingerprint(
            {
                "kind": "atomistic-generation-cell-bounds",
                "minimum_height": height,
                "maximum_condition": condition,
                "minimum_volume": lower,
                "maximum_volume": upper,
            }
        )

    def evaluate(self, cells: ArrayLike, /) -> tuple[Array, Array, Array, Array]:
        """Return ``(admitted, heights, condition, volume)`` for ``(L, 3, 3)`` row cells."""
        vectors = jnp.asarray(cells)
        if vectors.shape[-2:] != (3, 3):
            raise ValueError("cells must end in (3, 3).")
        volume, heights = cell_heights(vectors)
        identity = jnp.broadcast_to(jnp.eye(3, dtype=vectors.dtype), vectors.shape)
        solved = solve_small_linear(_SMALL3, vectors, identity)
        condition = solved.condition_estimate
        finite = jnp.all(jnp.isfinite(vectors), axis=(-2, -1))
        admitted = (
            finite
            & solved.successful
            & jnp.all(heights >= self.minimum_height, axis=-1)
            & (condition <= self.maximum_condition)
            & (volume >= self.minimum_volume)
            & (volume <= self.maximum_volume)
        )
        return admitted, heights, condition, volume


@final
class AtomisticCompositionBounds(StrictModule, NonTrainableState):
    """Per-element and total active-atom count bounds of a variable composition."""

    elements: tuple[int, ...] = eqx.field(static=True)
    minimum_counts: tuple[int, ...] = eqx.field(static=True)
    maximum_counts: tuple[int, ...] = eqx.field(static=True)
    minimum_atoms: int = eqx.field(static=True)
    maximum_atoms: int = eqx.field(static=True)
    bounds_id: str = eqx.field(static=True)

    def __init__(
        self,
        counts: Mapping[int, tuple[int, int]],
        /,
        *,
        minimum_atoms: int,
        maximum_atoms: int,
    ) -> None:
        items = sorted((int(key), value) for key, value in counts.items())
        if not items:
            raise ValueError("A composition declares at least one element.")
        elements, lower, upper = [], [], []
        for number, (low, high) in items:
            if number <= 0:
                raise ValueError("Elements are positive atomic numbers.")
            low_ = _nonnegative_int(low, "minimum count")
            high_ = _nonnegative_int(high, "maximum count")
            if high_ < low_:
                raise ValueError("Element count bounds must be ordered.")
            elements.append(number)
            lower.append(low_)
            upper.append(high_)
        minimum = _nonnegative_int(minimum_atoms, "minimum_atoms")
        maximum = _positive_int(maximum_atoms, "maximum_atoms")
        if maximum < minimum or sum(lower) > maximum or sum(upper) < minimum:
            raise ValueError("Composition bounds admit no active-atom count.")
        self.elements = tuple(elements)
        self.minimum_counts = tuple(lower)
        self.maximum_counts = tuple(upper)
        self.minimum_atoms = minimum
        self.maximum_atoms = maximum
        self.bounds_id = canonical_fingerprint(
            {
                "kind": "atomistic-composition-bounds",
                "elements": elements,
                "minimum": lower,
                "maximum": upper,
                "minimum_atoms": minimum,
                "maximum_atoms": maximum,
            }
        )

    def evaluate(self, species: ArrayLike, mask: ArrayLike, /) -> Array:
        values = jnp.asarray(species)
        active = jnp.asarray(mask, dtype=jnp.bool_)
        elements = jnp.asarray(self.elements, dtype=values.dtype)
        counts = jnp.sum(
            active[..., None] & (values[..., None] == elements), axis=-2, dtype=jnp.int32
        )
        total = jnp.sum(active, axis=-1, dtype=jnp.int32)
        known = jnp.all(
            ~active | jnp.any(values[..., None] == elements, axis=-1), axis=-1
        )
        return (
            known
            & jnp.all(counts >= jnp.asarray(self.minimum_counts), axis=-1)
            & jnp.all(counts <= jnp.asarray(self.maximum_counts), axis=-1)
            & (total >= self.minimum_atoms)
            & (total <= self.maximum_atoms)
        )


def _admit_species_vocabulary(
    species_vocabulary: Sequence[int], vocabulary_masses: Sequence[float], /
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Declared species vocabulary and its one-positive-mass-per-element table."""
    vocabulary = tuple(int(value) for value in species_vocabulary)
    masses = tuple(float(value) for value in vocabulary_masses)
    if not vocabulary or len(set(vocabulary)) != len(vocabulary):
        raise ValueError("species_vocabulary must list unique atomic numbers.")
    if any(value <= 0 for value in vocabulary):
        raise ValueError("species_vocabulary holds positive atomic numbers.")
    if len(masses) != len(vocabulary) or any(
        not np.isfinite(value) or value <= 0.0 for value in masses
    ):
        raise ValueError("vocabulary_masses declares one positive mass per element.")
    return vocabulary, masses


def _admit_fixed_slots(
    capacity: int,
    vocabulary: tuple[int, ...],
    fixed_species: Sequence[int] | None,
    fixed_mask: Sequence[bool] | None,
    composition: AtomisticCompositionBounds | None,
    /,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Fixed chemistry slots: every slot declared, vocabulary atoms, zero padding.

    Fixed chemistry declares species and mask (never composition bounds) over
    exactly ``capacity`` slots with at least one active atom.
    """
    if fixed_species is None or fixed_mask is None or composition is not None:
        raise ValueError(
            "Fixed chemistry declares fixed_species and fixed_mask, no composition."
        )
    species = tuple(int(value) for value in fixed_species)
    mask = tuple(bool(value) for value in fixed_mask)
    if len(species) != capacity or len(mask) != capacity or not any(mask):
        raise ValueError("Fixed chemistry covers every slot with an active atom.")
    for number, active in zip(species, mask, strict=True):
        if active and number not in vocabulary:
            raise ValueError("Fixed species lie outside the vocabulary.")
        if not active and number != 0:
            raise ValueError("Padding slots use atomic number zero.")
    return species, mask


def _admit_fixed_bonds(
    capacity: int,
    mask: tuple[bool, ...],
    bonds: tuple[tuple[int, int], ...],
    topology: AtomisticTopologyAdmission,
    /,
) -> None:
    """Declared fixed bonds join distinct active slots under FIXED topology."""
    for a, b in bonds:
        if a == b or not (0 <= a < capacity and 0 <= b < capacity):
            raise ValueError("Fixed bonds join distinct slots.")
        if not (mask[a] and mask[b]):
            raise ValueError("Fixed bonds join active slots.")
    if bonds and topology is not AtomisticTopologyAdmission.FIXED:
        raise ValueError("Declared bonds require FIXED topology admission.")


def _admit_variable_composition(
    capacity: int,
    vocabulary: tuple[int, ...],
    fixed_species: Sequence[int] | None,
    fixed_mask: Sequence[bool] | None,
    composition: AtomisticCompositionBounds | None,
    topology: AtomisticTopologyAdmission,
    bonds: tuple[tuple[int, int], ...],
    total_charge: int | None,
    /,
) -> None:
    """Variable composition: vocabulary-bounded counts and no inferred chemistry.

    Composition bounds replace fixed slots, draw only vocabulary elements and
    fit the atom capacity; topology, bonds and total charge stay undeclared.
    """
    if fixed_species is not None or fixed_mask is not None:
        raise ValueError("Variable composition declares no fixed species.")
    if not isinstance(composition, AtomisticCompositionBounds):
        raise ValueError("Variable composition requires composition bounds.")
    if not set(composition.elements) <= set(vocabulary):
        raise ValueError("Composition elements must belong to the vocabulary.")
    if composition.maximum_atoms > capacity:
        raise ValueError("Composition maximum exceeds the atom capacity.")
    if topology is not AtomisticTopologyAdmission.NOT_INFERRED or bonds:
        raise ValueError(
            "Variable composition infers no topology; bonds, charge state and "
            "molecule identity need an explicit qualifier."
        )
    if total_charge is not None:
        raise ValueError(
            "Variable composition has no fixed total charge; declare a "
            "charge qualifier filter instead."
        )


def _admit_cell_support(
    cells: AtomisticCellAdmission,
    cell: ArrayLike | None,
    cell_bounds: AtomisticCellBounds | None,
    /,
) -> tuple[AtomisticCoordinateRepresentation, tuple[tuple[float, ...], ...] | None]:
    """Coordinate representation and bound row cell of a cell admission.

    Nonperiodic supports are Cartesian and declare no cell.  Periodic supports
    are fractional under explicit ``cell_bounds``; a fixed cell is a finite
    ``(3, 3)`` row lattice its bounds admit, and a variable cell binds none.
    """
    cell_: tuple[tuple[float, ...], ...] | None = None
    if cells is AtomisticCellAdmission.NONPERIODIC:
        if cell is not None or cell_bounds is not None:
            raise ValueError("Nonperiodic supports declare no cell.")
        representation = AtomisticCoordinateRepresentation.CARTESIAN
    else:
        if not isinstance(cell_bounds, AtomisticCellBounds):
            raise ValueError("Periodic supports declare explicit cell_bounds.")
        representation = AtomisticCoordinateRepresentation.FRACTIONAL
        if cells is AtomisticCellAdmission.FIXED_CELL:
            if cell is None:
                raise ValueError("A fixed-cell support binds its row cell.")
            matrix = np.asarray(cell, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise ValueError("cell must be a finite (3, 3) row lattice.")
            admitted, *_ = cell_bounds.evaluate(jnp.asarray(matrix)[None])
            if not bool(admitted[0]):
                raise ValueError("The fixed cell violates its declared cell_bounds.")
            cell_ = tuple(tuple(float(v) for v in row) for row in matrix)
        elif cell is not None:
            raise ValueError("A variable-cell support binds no fixed cell.")
    return representation, cell_


@final
class AtomisticGenerationSupport(StrictModule, NonTrainableState):
    """Static chemistry, cell, representation, unit and condition ABI of proposals.

    Fixed chemistry binds ``fixed_species``/``fixed_mask`` (atomic numbers per
    slot, zero padding), its masses and, with ``FIXED`` topology, the declared
    bonds and total charge.  Variable composition draws species from the
    explicit ``species_vocabulary`` under ``composition`` bounds; its masses
    come from the declared ``vocabulary_masses`` table and its topology is
    ``NOT_INFERRED``.  Periodic supports use the ``FRACTIONAL`` representation
    on the unit torus with explicit ``cell_bounds``; a fixed-cell support binds
    its row ``cell``.
    """

    atom_capacity: int = eqx.field(static=True)
    species_vocabulary: tuple[int, ...] = eqx.field(static=True)
    vocabulary_masses: tuple[float, ...] = eqx.field(static=True)
    mass_provenance: str = eqx.field(static=True)
    chemistry: AtomisticChemistryAdmission = eqx.field(static=True)
    fixed_species: tuple[int, ...] | None = eqx.field(static=True)
    fixed_mask: tuple[bool, ...] | None = eqx.field(static=True)
    composition: AtomisticCompositionBounds | None
    topology: AtomisticTopologyAdmission = eqx.field(static=True)
    fixed_bonds: tuple[tuple[int, int], ...] = eqx.field(static=True)
    total_charge: int | None = eqx.field(static=True)
    cell_admission: AtomisticCellAdmission = eqx.field(static=True)
    representation: AtomisticCoordinateRepresentation = eqx.field(static=True)
    cell: tuple[tuple[float, ...], ...] | None = eqx.field(static=True)
    cell_bounds: AtomisticCellBounds | None
    length_unit: UnitDefinition
    conditioning: AtomisticConditioningSpec
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        atom_capacity: int,
        species_vocabulary: Sequence[int],
        vocabulary_masses: Sequence[float],
        mass_provenance: str,
        chemistry: AtomisticChemistryAdmission | str,
        length_unit: UnitDefinition,
        conditioning: AtomisticConditioningSpec | None = None,
        fixed_species: Sequence[int] | None = None,
        fixed_mask: Sequence[bool] | None = None,
        composition: AtomisticCompositionBounds | None = None,
        topology: AtomisticTopologyAdmission
        | str = AtomisticTopologyAdmission.NOT_INFERRED,
        fixed_bonds: Sequence[tuple[int, int]] = (),
        total_charge: int | None = None,
        cell_admission: AtomisticCellAdmission | str = AtomisticCellAdmission.NONPERIODIC,
        cell: ArrayLike | None = None,
        cell_bounds: AtomisticCellBounds | None = None,
    ) -> None:
        capacity = _positive_int(atom_capacity, "atom_capacity")
        vocabulary, masses = _admit_species_vocabulary(
            species_vocabulary, vocabulary_masses
        )
        provenance = _nonempty(mass_provenance, "mass_provenance")
        chemistry_ = parse(
            AtomisticChemistryAdmission(chemistry)
            if isinstance(chemistry, str)
            else chemistry,
            AtomisticChemistryAdmission,
            "chemistry",
        )
        topology_ = parse(
            AtomisticTopologyAdmission(topology)
            if isinstance(topology, str)
            else topology,
            AtomisticTopologyAdmission,
            "topology",
        )
        cells = parse(
            AtomisticCellAdmission(cell_admission)
            if isinstance(cell_admission, str)
            else cell_admission,
            AtomisticCellAdmission,
            "cell_admission",
        )
        if not isinstance(length_unit, UnitDefinition):
            raise TypeError("length_unit must be a UnitDefinition.")
        spec = AtomisticConditioningSpec() if conditioning is None else conditioning
        if not isinstance(spec, AtomisticConditioningSpec):
            raise TypeError("conditioning must be an AtomisticConditioningSpec.")
        species_: tuple[int, ...] | None = None
        mask_: tuple[bool, ...] | None = None
        bonds = tuple((int(a), int(b)) for a, b in fixed_bonds)
        match chemistry_:
            case AtomisticChemistryAdmission.FIXED:
                species_, mask_ = _admit_fixed_slots(
                    capacity, vocabulary, fixed_species, fixed_mask, composition
                )
                _admit_fixed_bonds(capacity, mask_, bonds, topology_)
            case AtomisticChemistryAdmission.VARIABLE_COMPOSITION:
                _admit_variable_composition(
                    capacity,
                    vocabulary,
                    fixed_species,
                    fixed_mask,
                    composition,
                    topology_,
                    bonds,
                    total_charge,
                )
        representation, cell_ = _admit_cell_support(cells, cell, cell_bounds)
        self.atom_capacity = capacity
        self.species_vocabulary = vocabulary
        self.vocabulary_masses = masses
        self.mass_provenance = provenance
        self.chemistry = chemistry_
        self.fixed_species = species_
        self.fixed_mask = mask_
        self.composition = composition
        self.topology = topology_
        self.fixed_bonds = bonds
        self.total_charge = None if total_charge is None else int(total_charge)
        self.cell_admission = cells
        self.representation = representation
        self.cell = cell_
        self.cell_bounds = cell_bounds
        self.length_unit = length_unit
        self.conditioning = spec
        self.support_id = canonical_fingerprint(
            {
                "kind": "atomistic-generation-support",
                "atom_capacity": capacity,
                "vocabulary": vocabulary,
                "masses": masses,
                "mass_provenance": provenance,
                "chemistry": chemistry_.value,
                "fixed_species": species_,
                "fixed_mask": mask_,
                "composition": None if composition is None else composition.bounds_id,
                "topology": topology_.value,
                "bonds": bonds,
                "total_charge": self.total_charge,
                "cell_admission": cells.value,
                "representation": representation.value,
                "cell": cell_,
                "cell_bounds": None if cell_bounds is None else cell_bounds.bounds_id,
                "length_unit": length_unit.unit_id,
                "conditioning": spec.spec_id,
            }
        )

    @property
    def periodic(self) -> bool:
        return self.cell_admission is not AtomisticCellAdmission.NONPERIODIC

    def masses(self, species: ArrayLike, mask: ArrayLike, /) -> Array:
        """Declared masses of vocabulary species (zero on padding)."""
        values = jnp.asarray(species)
        active = jnp.asarray(mask, dtype=jnp.bool_)
        vocabulary = jnp.asarray(self.species_vocabulary, dtype=values.dtype)
        table = jnp.asarray(self.vocabulary_masses)
        index = jnp.argmax(values[..., None] == vocabulary, axis=-1)
        return jnp.where(active, table[index], 0.0)

    def admit(self, block: AtomisticProposalBlock, /) -> Array:
        """Support admission bits of every block row (zero when admitted)."""
        status = jnp.zeros(block.status.shape, dtype=jnp.int32)
        active = block.atom_mask
        finite = jnp.all(
            jnp.where(active[..., None], jnp.isfinite(block.positions), True),
            axis=(-2, -1),
        )
        status = status | jnp.where(finite, 0, int(AtomisticProposalStatus.NONFINITE))
        vocabulary = jnp.asarray(self.species_vocabulary, dtype=block.species.dtype)
        known = jnp.all(
            ~active | jnp.any(block.species[..., None] == vocabulary, axis=-1), axis=-1
        )
        padding = jnp.all(active | (block.species == 0), axis=-1)
        status = status | jnp.where(
            known & padding, 0, int(AtomisticProposalStatus.SPECIES_INVALID)
        )
        if self.fixed_species is not None and self.fixed_mask is not None:
            exact = jnp.all(
                (
                    block.species
                    == jnp.asarray(self.fixed_species, dtype=block.species.dtype)
                )
                & (active == jnp.asarray(self.fixed_mask)),
                axis=-1,
            )
            status = status | jnp.where(
                exact, 0, int(AtomisticProposalStatus.SPECIES_INVALID)
            )
        if self.composition is not None:
            composed = self.composition.evaluate(block.species, active)
            status = status | jnp.where(
                composed, 0, int(AtomisticProposalStatus.COMPOSITION_INVALID)
            )
        if self.cell_bounds is not None:
            if block.cell is None:
                return status | int(AtomisticProposalStatus.CELL_INVALID)
            admitted, *_ = self.cell_bounds.evaluate(block.cell)
            if self.cell is not None:
                admitted = admitted & jnp.all(
                    block.cell == jnp.asarray(self.cell, dtype=block.cell.dtype),
                    axis=(-2, -1),
                )
            status = status | jnp.where(
                admitted, 0, int(AtomisticProposalStatus.CELL_INVALID)
            )
        elif block.cell is not None:
            status = status | int(AtomisticProposalStatus.CELL_INVALID)
        return status.astype(jnp.int32)


# -- proposals and sampler lanes -----------------------------------------------------------


@final
class AtomisticPayloadSpec(StrictModule, NonTrainableState):
    """Named per-proposal payload arrays a sampler emits (shapes exclude the lane axis)."""

    names: tuple[str, ...] = eqx.field(static=True)
    shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    dtypes: tuple[str, ...] = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)

    def __init__(self, entries: Sequence[tuple[str, Sequence[int], str]] = (), /) -> None:
        names, shapes, dtypes = [], [], []
        for name, shape, dtype in entries:
            names.append(_nonempty(name, "payload name"))
            shapes.append(tuple(int(size) for size in shape))
            dtypes.append(np.dtype(dtype).name)
        if len(set(names)) != len(names):
            raise ValueError("Payload names must be unique.")
        if any(size < 0 for shape in shapes for size in shape):
            raise ValueError("Payload shapes must be nonnegative.")
        self.names = tuple(names)
        self.shapes = tuple(shapes)
        self.dtypes = tuple(dtypes)
        self.spec_id = canonical_fingerprint(
            {
                "kind": "atomistic-proposal-payload-spec",
                "entries": [
                    list(item) for item in zip(names, shapes, dtypes, strict=True)
                ],
            }
        )

    def zeros(self, rows: int, /) -> tuple[Array, ...]:
        return tuple(
            jnp.zeros((rows, *shape), dtype=dtype)
            for shape, dtype in zip(self.shapes, self.dtypes, strict=True)
        )


@final
class AtomisticProposalBlock(StrictModule):
    """One proposal per lane: Cartesian geometry, species, optional cell and payload.

    ``positions`` are unwrapped Cartesian ``(L, N, 3)``; ``cell`` is the
    ``(L, 3, 3)`` row lattice of periodic supports.  ``status`` carries
    `AtomisticProposalStatus` bits (zero accepted).  ``payload`` follows the
    sampler's `AtomisticPayloadSpec` with a leading lane axis.
    """

    ordinals: Array
    positions: Array
    species: Array
    atom_mask: Array
    cell: Array | None
    status: Array
    payload: tuple[Array, ...]
    sampler_id: str = eqx.field(static=True)
    payload_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        ordinals: ArrayLike,
        positions: ArrayLike,
        species: ArrayLike,
        atom_mask: ArrayLike,
        cell: ArrayLike | None,
        status: ArrayLike,
        payload: Sequence[Array],
        /,
        *,
        sampler_id: str,
        payload_names: Sequence[str],
    ) -> None:
        position = jnp.asarray(positions)
        lanes = position.shape[0] if position.ndim == 3 else -1
        if position.ndim != 3 or position.shape[-1] != 3:
            raise ValueError("positions must have shape (lanes, atoms, 3).")
        if not jnp.issubdtype(position.dtype, jnp.floating):
            raise TypeError("positions must be real floating values.")
        species_ = jnp.asarray(species).astype(jnp.int32)
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_)
        if species_.shape != position.shape[:2] or mask.shape != position.shape[:2]:
            raise ValueError("species and atom_mask must have shape (lanes, atoms).")
        cell_ = None if cell is None else jnp.asarray(cell, dtype=position.dtype)
        if cell_ is not None and cell_.shape != (lanes, 3, 3):
            raise ValueError("cell must have shape (lanes, 3, 3).")
        status_ = jnp.asarray(status).astype(jnp.int32)
        ordinals_ = jnp.asarray(ordinals).astype(jnp.int64)
        if status_.shape != (lanes,) or ordinals_.shape != (lanes,):
            raise ValueError("status and ordinals hold one value per lane.")
        payload_ = tuple(jnp.asarray(value) for value in payload)
        names = tuple(payload_names)
        if len(names) != len(payload_) or any(
            value.shape[:1] != (lanes,) for value in payload_
        ):
            raise ValueError("payload must align with its names and the lane axis.")
        self.ordinals = ordinals_
        self.positions = position
        self.species = species_
        self.atom_mask = mask
        self.cell = cell_
        self.status = status_
        self.payload = payload_
        self.sampler_id = _nonempty(sampler_id, "sampler_id")
        self.payload_names = names

    @property
    def lanes(self) -> int:
        return int(self.positions.shape[0])

    @property
    def accepted(self) -> Array:
        return self.status == 0

    def value(self, name: str, /) -> Array:
        if name not in self.payload_names:
            raise KeyError(f"Proposal payload {name!r} is not declared.")
        return self.payload[self.payload_names.index(name)]

    def with_status(self, status: ArrayLike, /) -> AtomisticProposalBlock:
        return AtomisticProposalBlock(
            self.ordinals,
            self.positions,
            self.species,
            self.atom_mask,
            self.cell,
            status,
            self.payload,
            sampler_id=self.sampler_id,
            payload_names=self.payload_names,
        )


@final
class AtomisticSamplerLaneState(StrictModule):
    """Fixed-capacity lanes of one prepared sampler.

    ``active`` lanes own an admitted ordinal; ``terminal`` lanes have a final
    proposal ready for ``emit``.  ``key_data`` holds the caller-supplied lane
    keys (default implementation) so the state persists as plain arrays;
    ``method`` is the sampler's numeric continuation with a leading lane axis.
    """

    active: Array
    terminal: Array
    ordinals: Array
    key_data: Array
    method: Any
    sampler_id: str = eqx.field(static=True)

    @property
    def lanes(self) -> int:
        return int(self.active.shape[0])

    @property
    def keys(self) -> Array:
        return jr.wrap_key_data(self.key_data)


class AbstractPreparedAtomisticSampler(StrictModule, NonTrainableState):
    """Native lane transition producing atomistic proposals.

    Subclasses implement the method hooks; the base owns lane masking, so
    lanes outside an ``admit`` mask and inactive or terminal lanes during
    ``advance`` stay bitwise unchanged.  ``stateful`` samplers must implement
    `advance_method`; one-shot samplers are terminal after `admit`.
    ``idempotent`` declares a pure one-shot sampler whose proposal is a pure
    function of ``(key, ordinal, conditioning)`` and may be recomputed from a
    watermark; stateful samplers never are.
    """

    sampler_id: eqx.AbstractVar[str]
    support: eqx.AbstractVar[AtomisticGenerationSupport]
    payload_spec: eqx.AbstractVar[AtomisticPayloadSpec]
    stateful: eqx.AbstractVar[bool]
    idempotent: eqx.AbstractVar[bool]

    @abc.abstractmethod
    def vacant_method(self, lanes: int, /) -> Any:
        """Vacant method state with leading lane axis ``lanes``."""

    @abc.abstractmethod
    def initialize_method(
        self,
        keys: Array,
        ordinals: Array,
        conditioning: AtomisticConditioning | None,
        selected: Array,
        /,
    ) -> Any:
        """Method state of lanes started at ``(key, ordinal, conditioning)``.

        ``selected`` is the admission mask.  Rows of unselected lanes are
        discarded by `admit`, so a sampler may (and periodic samplers do, through
        `map_admitted_lanes`) skip every scientific and random computation for
        them and return their vacant rows.
        """

    def advance_method(
        self, method: Any, running: Array, steps: int, /
    ) -> tuple[Any, Array]:
        """Advance running lanes up to ``steps`` transitions; return ``(method, done)``."""
        raise NotImplementedError(f"{type(self).__name__} is a one-shot sampler.")

    @abc.abstractmethod
    def emit_method(self, method: Any, ordinals: Array, /) -> AtomisticProposalBlock:
        """Proposal block of the current method state of every lane."""

    @abc.abstractmethod
    def signature_facts(self) -> dict[str, str | int]:
        """Static executable facts: layout, capacities, architecture, precision."""

    @abc.abstractmethod
    def numeric_revision(self) -> NumericRevision:
        """Numeric revision of the sampler's dynamic model/parameter content."""

    def select_method(self, mask: Array, new: Any, old: Any, /) -> Any:
        """Per-lane selection of method states (override for non lane-major leaves)."""
        return _tree_where(mask, new, old)

    def vacant(self, lanes: int, /) -> AtomisticSamplerLaneState:
        count = _positive_int(lanes, "lanes")
        return AtomisticSamplerLaneState(
            jnp.zeros((count,), dtype=jnp.bool_),
            jnp.zeros((count,), dtype=jnp.bool_),
            jnp.zeros((count,), dtype=jnp.int64),
            jr.key_data(jr.split(jr.key(0), count)),
            self.vacant_method(count),
            self.sampler_id,
        )

    def _require_lanes(self, state: AtomisticSamplerLaneState, /) -> None:
        if not isinstance(state, AtomisticSamplerLaneState):
            raise TypeError("state must be an AtomisticSamplerLaneState.")
        if state.sampler_id != self.sampler_id:
            raise ValueError("Lane state belongs to another sampler.")

    def admit(
        self,
        state: AtomisticSamplerLaneState,
        mask: ArrayLike,
        keys: Array,
        ordinals: ArrayLike,
        conditioning: AtomisticConditioning | None,
        /,
    ) -> AtomisticSamplerLaneState:
        """Start masked lanes at their ordinals; other lanes stay bitwise unchanged."""
        self._require_lanes(state)
        lanes = state.lanes
        selected = jnp.asarray(mask, dtype=jnp.bool_)
        ordinal = jnp.asarray(ordinals).astype(jnp.int64)
        if selected.shape != (lanes,) or ordinal.shape != (lanes,):
            raise ValueError("mask and ordinals hold one value per lane.")
        if keys.shape != (lanes,) or not jax.dtypes.issubdtype(
            keys.dtype, jax.dtypes.prng_key
        ):
            raise TypeError("keys must be one typed PRNG key per lane.")
        if jr.key_data(keys).shape != state.key_data.shape:
            raise TypeError("keys must use the default PRNG implementation.")
        spec = self.support.conditioning
        if spec.fields:
            if not isinstance(conditioning, AtomisticConditioning):
                raise ValueError("This sampler requires typed conditioning.")
            if (
                conditioning.spec.spec_id != spec.spec_id
                or conditioning.case_count != lanes
                or conditioning.atom_capacity != self.support.atom_capacity
            ):
                raise ValueError(
                    "Conditioning must follow the support spec with one case per lane."
                )
        elif conditioning is not None:
            raise ValueError("This support declares no condition ports.")
        started = self.initialize_method(keys, ordinal, conditioning, selected)
        method = self.select_method(selected, started, state.method)
        terminal = jnp.where(selected, jnp.asarray(not self.stateful), state.terminal)
        return AtomisticSamplerLaneState(
            state.active | selected,
            terminal,
            jnp.where(selected, ordinal, state.ordinals),
            jnp.where(selected[:, None], jr.key_data(keys), state.key_data),
            method,
            self.sampler_id,
        )

    def advance(
        self, state: AtomisticSamplerLaneState, steps: int, /
    ) -> AtomisticSamplerLaneState:
        self._require_lanes(state)
        count = _nonnegative_int(steps, "steps")
        if not self.stateful or count == 0:
            return state
        running = state.active & ~state.terminal
        method, done = jax.lax.cond(
            jnp.any(running),
            lambda: self.advance_method(state.method, running, count),
            lambda: (state.method, state.terminal),
        )
        return AtomisticSamplerLaneState(
            state.active,
            state.terminal | (running & jnp.asarray(done, dtype=jnp.bool_)),
            state.ordinals,
            state.key_data,
            self.select_method(running, method, state.method),
            self.sampler_id,
        )

    def terminal(self, state: AtomisticSamplerLaneState, /) -> Array:
        self._require_lanes(state)
        return state.active & state.terminal

    def emit(self, state: AtomisticSamplerLaneState, /) -> AtomisticProposalBlock:
        self._require_lanes(state)
        block = self.emit_method(state.method, state.ordinals)
        if block.sampler_id != self.sampler_id or block.lanes != state.lanes:
            raise ValueError("emit_method must return one row per lane of this sampler.")
        if block.payload_names != self.payload_spec.names:
            raise ValueError("emit_method payload does not follow payload_spec.")
        return block

    def release(
        self, state: AtomisticSamplerLaneState, mask: ArrayLike, /
    ) -> AtomisticSamplerLaneState:
        """Vacate masked lanes (their method state is kept until re-admission)."""
        self._require_lanes(state)
        selected = jnp.asarray(mask, dtype=jnp.bool_)
        return AtomisticSamplerLaneState(
            state.active & ~selected,
            state.terminal & ~selected,
            state.ordinals,
            state.key_data,
            state.method,
            self.sampler_id,
        )


# -- routing, conditions and filters --------------------------------------------------------


class AbstractOrdinalRoute(StrictModule, NonTrainableState):
    """Pure ordinal-addressed choice of a family member (e.g. a space group)."""

    route_id: eqx.AbstractVar[str]
    member_count: eqx.AbstractVar[int]

    @abc.abstractmethod
    def members(self, root_key: PRNGKey, ordinals: Array, /) -> Array:
        """Member index of every ordinal; a pure function of root key and ordinal."""


@final
class AtomisticFixedRoute(AbstractOrdinalRoute):
    """Every ordinal uses member zero (a family of one)."""

    route_id: str = eqx.field(static=True)
    member_count: int = eqx.field(static=True)

    def __init__(self) -> None:
        self.route_id = canonical_fingerprint({"kind": "atomistic-fixed-route"})
        self.member_count = 1

    def members(self, root_key: PRNGKey, ordinals: Array, /) -> Array:
        del root_key
        return jnp.zeros(jnp.shape(ordinals), dtype=jnp.int32)


@final
class AtomisticWeightedRoute(AbstractOrdinalRoute):
    """Ordinal-addressed categorical choice under declared member weights.

    The weights are a caller-admitted prior with explicit provenance; they are
    never inferred from member names or indices.
    """

    weights: tuple[float, ...] = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    member_count: int = eqx.field(static=True)

    def __init__(self, weights: Sequence[float], /, *, provenance: str) -> None:
        values = tuple(float(value) for value in weights)
        if not values or any(not np.isfinite(v) or v < 0.0 for v in values):
            raise ValueError("Route weights must be finite and nonnegative.")
        total = sum(values)
        if total <= 0.0:
            raise ValueError("Route weights need positive total mass.")
        self.weights = tuple(value / total for value in values)
        self.provenance = _nonempty(provenance, "provenance")
        self.member_count = len(values)
        self.route_id = canonical_fingerprint(
            {
                "kind": "atomistic-weighted-route",
                "weights": self.weights,
                "provenance": self.provenance,
            }
        )

    def members(self, root_key: PRNGKey, ordinals: Array, /) -> Array:
        address = SampleAddress(
            GENERATION_ADDRESS_NAMESPACE, "route", target=(self.route_id,), role="member"
        )
        keys = proposal_keys(root_key, address, ordinals)
        logits = jnp.log(jnp.asarray(self.weights))
        return jax.vmap(lambda key: jr.categorical(key, logits))(keys).astype(jnp.int32)


class AbstractConditionSource(StrictModule, NonTrainableState):
    """Typed per-ordinal condition values of one conditioning spec.

    ``source_id`` is the semantic identity (it enters the stream identity);
    `numeric_revision` binds the numeric values the source currently holds, so
    a continuation conditioned under other values refuses to restore.
    """

    source_id: eqx.AbstractVar[str]
    spec: eqx.AbstractVar[AtomisticConditioningSpec]

    @abc.abstractmethod
    def conditioning(
        self, ordinals: Array, keys: Array, /, *, atom_capacity: int
    ) -> AtomisticConditioning:
        """Conditioning with one case per ordinal."""

    @abc.abstractmethod
    def numeric_revision(self) -> NumericRevision:
        """Numeric revision of the source's dynamic condition content."""


@final
class AtomisticConditionSchedule(AbstractConditionSource):
    """Declared condition rows indexed by ordinal (``cycle`` or ``bounded``).

    Row ``k`` belongs to ordinal ``first_ordinal + k``.  ``bounded`` streams
    must stay inside the declared rows (checked at stream start/extend);
    ``cycle`` repeats rows ``(ordinal - first_ordinal) mod rows``.  The spec,
    policy, first ordinal and row count are the semantic ``source_id``; the
    row values are its numeric revision.
    """

    spec: AtomisticConditioningSpec
    rows: tuple[Array, ...]
    row_count: int = eqx.field(static=True)
    policy: ConditionSchedulePolicy = eqx.field(static=True)
    first_ordinal: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        spec: AtomisticConditioningSpec,
        values: Mapping[str, ArrayLike],
        /,
        *,
        policy: ConditionSchedulePolicy,
        first_ordinal: int = 0,
    ) -> None:
        if not isinstance(spec, AtomisticConditioningSpec):
            raise TypeError("spec must be an AtomisticConditioningSpec.")
        policy_ = parse(policy, ConditionSchedulePolicy, "policy")
        if set(values) != set(spec.names):
            raise ValueError("A condition schedule supplies every declared field.")
        arrays = []
        count = None
        for field in spec.fields:
            if field.level.value != "case":
                raise ValueError("Condition schedules supply case-level fields only.")
            array = jnp.asarray(values[field.name], dtype=field.dtype.value)
            if array.ndim < 1 or array.shape[1:] != field.port.event_shape:
                raise ValueError(f"Rows of {field.name!r} must follow its port event.")
            if count is None:
                count = array.shape[0]
            if array.shape[0] != count or count < 1:
                raise ValueError("Every field supplies the same positive row count.")
            arrays.append(array)
        if count is None:
            raise ValueError("A condition schedule needs at least one field.")
        self.spec = spec
        self.rows = tuple(arrays)
        self.row_count = int(count)
        self.policy = policy_
        self.first_ordinal = _nonnegative_int(first_ordinal, "first_ordinal")
        self.source_id = canonical_fingerprint(
            {
                "kind": "atomistic-condition-schedule",
                "spec": spec.spec_id,
                "row_count": self.row_count,
                "policy": policy_,
                "first_ordinal": self.first_ordinal,
            }
        )

    def conditioning(
        self, ordinals: Array, keys: Array, /, *, atom_capacity: int
    ) -> AtomisticConditioning:
        del keys
        offset = jnp.asarray(ordinals).astype(jnp.int64) - self.first_ordinal
        if self.policy == "bounded":
            # Vacant lanes may carry stale ordinals; admitted ones are in range.
            index = jnp.clip(offset, 0, self.row_count - 1)
        else:
            index = offset % self.row_count
        return AtomisticConditioning(
            self.spec,
            {
                field.name: row[index]
                for field, row in zip(self.spec.fields, self.rows, strict=True)
            },
            case_count=int(index.shape[0]),
            atom_capacity=atom_capacity,
        )

    def numeric_revision(self) -> NumericRevision:
        return NumericRevision(self.source_id, self.rows)


@final
class AtomisticConditionTransform(StrictModule, NonTrainableState):
    """Pure typed condition map with declared input and output specs.

    Replaces mutating before/after-condition hooks: ``function(values, keys)``
    maps the named input arrays (and per-lane keys of its own address) to the
    named output arrays; the result is re-admitted under ``output``.
    ``function`` must hold its numeric parameters as visible dynamic PyTree
    leaves (callables hiding arrays of any dtype are refused, with or without
    declared IDs) and is identified by `callable_payload`: a StrictModule or a
    plain stateless module function is content-addressed, any other callable
    must declare ``function_semantic_id`` and ``function_numeric_id``.  Its
    semantic identity enters ``transform_id``; its numeric content (the
    parameters a StrictModule holds, or the declared numeric ID and visible
    array leaves of an opaque callable) is the transform's `numeric_revision`.
    """

    input_spec: AtomisticConditioningSpec
    output_spec: AtomisticConditioningSpec
    function: Callable[[Mapping[str, Array], Array], Mapping[str, Array]]
    function_numeric: IdentityWords | None
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        input_spec: AtomisticConditioningSpec,
        output_spec: AtomisticConditioningSpec,
        function: Callable[[Mapping[str, Array], Array], Mapping[str, Array]],
        /,
        *,
        transform_id: str,
        function_semantic_id: str | None = None,
        function_numeric_id: str | None = None,
    ) -> None:
        if not isinstance(input_spec, AtomisticConditioningSpec) or not isinstance(
            output_spec, AtomisticConditioningSpec
        ):
            raise TypeError("Transform specs must be AtomisticConditioningSpec values.")
        if not callable(function):
            raise TypeError("function must be callable.")
        declared_id = _nonempty(transform_id, "transform_id")
        function_id, declared = _callable_identity(
            function,
            function_semantic_id,
            function_numeric_id,
            context="AtomisticConditionTransform function",
        )
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.function = function
        self.function_numeric = declared
        self.transform_id = canonical_fingerprint(
            {
                "kind": "atomistic-condition-transform",
                "declared": declared_id,
                "input": input_spec.spec_id,
                "output": output_spec.spec_id,
                "function": function_id,
            }
        )

    def numeric_revision(self) -> NumericRevision:
        """Numeric revision of the parameters the transform function holds."""
        return NumericRevision(
            self.transform_id,
            _callable_numeric_content(
                self.function,
                self.function_numeric,
                context="AtomisticConditionTransform function",
            ),
        )

    def __call__(
        self, conditioning: AtomisticConditioning, keys: Array, /
    ) -> AtomisticConditioning:
        if conditioning.spec.spec_id != self.input_spec.spec_id:
            raise ValueError("Conditioning does not follow the transform input spec.")
        inputs = {
            name: conditioning.value(name)
            for name in self.input_spec.names
            if conditioning.present(name)
        }
        outputs = self.function(inputs, role_keys(keys, self.transform_id))
        return AtomisticConditioning(
            self.output_spec,
            dict(outputs),
            case_count=conditioning.case_count,
            atom_capacity=conditioning.atom_capacity,
        )


@final
class AtomisticFilterEvidence(StrictModule):
    """Per-row keep decision and one named diagnostic value."""

    keep: Array
    value: Array


class AbstractProposalFilter(StrictModule, NonTrainableState):
    """Pure proposal filter; rejected rows are retained with this filter's bit."""

    filter_id: eqx.AbstractVar[str]
    status_bit: eqx.AbstractVar[int]

    @abc.abstractmethod
    def evaluate(self, block: AtomisticProposalBlock, /) -> AtomisticFilterEvidence:
        """Keep decision and diagnostic of every row."""


@final
class AtomisticMinimumDistanceFilter(AbstractProposalFilter):
    """Reject proposals with any active pair closer than ``minimum_distance``.

    Periodic proposals are checked over explicit lattice images
    ``n in [-image_extent, image_extent]^3`` of the reduced fractional
    differences; the stencil is complete for the threshold only when every
    perpendicular height ``h`` satisfies ``(image_extent - 1/2) h >=
    minimum_distance``, otherwise the row is rejected as undetermined rather
    than accepted on a truncated search.  Self-images of an atom count.
    """

    minimum_distance: float = eqx.field(static=True)
    image_extent: int = eqx.field(static=True)
    filter_id: str = eqx.field(static=True)
    status_bit: int = eqx.field(static=True)

    def __init__(self, minimum_distance: float, /, *, image_extent: int = 2) -> None:
        self.minimum_distance = _positive_float(minimum_distance, "minimum_distance")
        self.image_extent = _positive_int(image_extent, "image_extent")
        self.status_bit = int(AtomisticProposalStatus.GEOMETRY_INVALID)
        self.filter_id = canonical_fingerprint(
            {
                "kind": "atomistic-minimum-distance-filter",
                "minimum_distance": self.minimum_distance,
                "image_extent": self.image_extent,
            }
        )

    def evaluate(self, block: AtomisticProposalBlock, /) -> AtomisticFilterEvidence:
        positions = block.positions
        mask = block.atom_mask
        count = positions.shape[1]
        delta = positions[:, None, :, :] - positions[:, :, None, :]
        pair = mask[:, :, None] & mask[:, None, :]
        if block.cell is None:
            off_diagonal = ~jnp.eye(count, dtype=jnp.bool_)
            distance = jnp.linalg.norm(delta, axis=-1)
            closest = jnp.min(
                jnp.where(pair & off_diagonal, distance, jnp.inf), axis=(-2, -1)
            )
            return AtomisticFilterEvidence(closest >= self.minimum_distance, closest)
        extent = self.image_extent
        transpose = jnp.broadcast_to(
            jnp.swapaxes(block.cell, -1, -2)[:, None, None], delta.shape[:-1] + (3, 3)
        )
        solved = solve_small_linear(_SMALL3, transpose, delta[..., :, None])
        fractional = solved.value[..., 0]
        reduced = fractional - jax.lax.stop_gradient(jnp.round(fractional))
        wrapped = contract("lpqi,lij->lpqj", reduced, block.cell)
        grid = np.stack(
            np.meshgrid(*(np.arange(-extent, extent + 1),) * 3, indexing="ij"), axis=-1
        ).reshape((-1, 3))
        shifts = jnp.asarray(grid, dtype=positions.dtype)
        translations = contract("si,lij->lsj", shifts, block.cell)
        image = wrapped[:, None] + translations[:, :, None, None, :]
        zero = jnp.asarray(np.all(grid == 0, axis=-1))
        diagonal = jnp.eye(count, dtype=jnp.bool_)
        excluded = zero[:, None, None] & diagonal[None]
        distance = jnp.linalg.norm(image, axis=-1)
        closest = jnp.min(
            jnp.where(pair[:, None] & ~excluded[None], distance, jnp.inf),
            axis=(-3, -2, -1),
        )
        # Reduced fractional differences lie in [-1/2, 1/2]; an image closer than
        # d needs |n_i| <= 1/2 + d / h_i, so the stencil is complete when
        # extent >= 1/2 + d / min(h).
        _, heights = cell_heights(block.cell)
        complete = jnp.all(
            heights * (extent - 0.5) >= self.minimum_distance, axis=-1
        ) & jnp.all(solved.successful, axis=(-2, -1))
        return AtomisticFilterEvidence(
            complete & (closest >= self.minimum_distance), closest
        )


@final
class AtomisticChargeBalanceQualifier(AbstractProposalFilter):
    """Declared oxidation-state charge balance of a variable composition.

    A proposal qualifies when some assignment of each atom to one of its
    element's declared oxidation states sums to ``total_charge``.  This is an
    explicit chemistry qualifier, not bond, valence or stability evidence.
    The bounded dynamic program runs over the declared charge window.
    """

    elements: tuple[int, ...] = eqx.field(static=True)
    states: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    total_charge: int = eqx.field(static=True)
    maximum_atoms: int = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    filter_id: str = eqx.field(static=True)
    status_bit: int = eqx.field(static=True)

    def __init__(
        self,
        oxidation_states: Mapping[int, Sequence[int]],
        /,
        *,
        provenance: str,
        maximum_atoms: int,
        total_charge: int = 0,
    ) -> None:
        items = sorted(
            (int(key), tuple(int(v) for v in value))
            for key, value in oxidation_states.items()
        )
        if not items or any(not states for _, states in items):
            raise ValueError("Every declared element needs at least one oxidation state.")
        self.elements = tuple(number for number, _ in items)
        self.states = tuple(tuple(sorted(set(states))) for _, states in items)
        self.total_charge = int(total_charge)
        self.maximum_atoms = _positive_int(maximum_atoms, "maximum_atoms")
        self.provenance = _nonempty(provenance, "provenance")
        self.status_bit = int(AtomisticProposalStatus.CHEMISTRY_UNQUALIFIED)
        self.filter_id = canonical_fingerprint(
            {
                "kind": "atomistic-charge-balance-qualifier",
                "elements": self.elements,
                "states": self.states,
                "total_charge": self.total_charge,
                "maximum_atoms": self.maximum_atoms,
                "provenance": self.provenance,
            }
        )

    def evaluate(self, block: AtomisticProposalBlock, /) -> AtomisticFilterEvidence:
        bound = self.maximum_atoms * max(abs(v) for states in self.states for v in states)
        width = 2 * bound + 1
        elements = jnp.asarray(self.elements, dtype=block.species.dtype)
        mask = block.atom_mask
        known = jnp.all(
            ~mask | jnp.any(block.species[..., None] == elements, axis=-1), axis=-1
        )
        index = jnp.argmax(block.species[..., None] == elements, axis=-1)
        values = sorted({value for states in self.states for value in states})
        allowed = np.zeros((len(self.elements), len(values)), dtype=bool)
        for row, states in enumerate(self.states):
            for value in states:
                allowed[row, values.index(value)] = True
        step_table = jnp.asarray(allowed)

        def lane(index_row: Array, mask_row: Array) -> Array:
            reachable = jnp.zeros((width,), dtype=jnp.bool_).at[bound].set(True)

            def atom(carry: Array, item: tuple[Array, Array]) -> tuple[Array, None]:
                element, active = item
                states = step_table[element]
                shifted = jnp.zeros_like(carry)
                for column, offset in enumerate(values):
                    moved = jnp.roll(carry, offset)
                    if offset > 0:
                        moved = moved.at[:offset].set(False)
                    elif offset < 0:
                        moved = moved.at[offset:].set(False)
                    shifted = shifted | (moved & states[column])
                return jnp.where(active, shifted, carry), None

            reachable, _ = jax.lax.scan(atom, reachable, (index_row, mask_row))
            target = self.total_charge + bound
            return reachable[target] if 0 <= target < width else jnp.asarray(False)

        balanced = jax.vmap(lane)(index, mask)
        count = jnp.sum(mask, axis=-1)
        keep = known & balanced & (count <= self.maximum_atoms)
        return AtomisticFilterEvidence(keep, count.astype(block.positions.dtype))


def screen_atomistic_proposals(
    block: AtomisticProposalBlock,
    support: AtomisticGenerationSupport,
    filters: Sequence[AbstractProposalFilter],
    /,
) -> tuple[Array, Array, Array]:
    """Pure admission and filter screen shared by streams and campaigns.

    Returns ``(status, filter_bits, filter_values)``: ``status`` is the
    sampler status combined with `AtomisticGenerationSupport.admit`; filters
    run in declared order and reject only rows still admitted, setting their
    ``status_bit``, their index bit in ``filter_bits`` and
    ``FILTER_REJECTED``.  ``filter_values`` holds every filter's diagnostic
    (``(lanes, filters)``) whether or not the row was rejected.
    """
    items = tuple(filters)
    if len(items) > _MAXIMUM_FILTERS or any(
        not isinstance(item, AbstractProposalFilter) for item in items
    ):
        raise ValueError(f"filters must be at most {_MAXIMUM_FILTERS} proposal filters.")
    status = block.status | support.admit(block)
    bits = jnp.zeros(status.shape, dtype=jnp.int32)
    values = []
    for index, item in enumerate(items):
        evidence = item.evaluate(block)
        rejected = (status == 0) & ~evidence.keep
        bits = bits | jnp.where(rejected, 1 << index, 0)
        status = status | jnp.where(rejected, item.status_bit, 0)
        values.append(evidence.value.astype(block.positions.dtype))
    status = status | jnp.where(
        bits != 0, int(AtomisticProposalStatus.FILTER_REJECTED), 0
    )
    filter_values = (
        jnp.stack(values, axis=-1)
        if values
        else jnp.zeros((block.lanes, 0), dtype=block.positions.dtype)
    )
    return status.astype(jnp.int32), bits.astype(jnp.int32), filter_values


# -- family, bank and stream ----------------------------------------------------------------


@final
class AtomisticSamplerFamily(StrictModule, NonTrainableState):
    """Ordinal-routed samplers of one stream sharing a support kind and payload.

    Members may differ in static shape (e.g. one packer per admitted space-group
    setting); payload specs must agree.  A single sampler is a family of one.
    """

    members: tuple[AbstractPreparedAtomisticSampler, ...]
    route: AbstractOrdinalRoute
    family_id: str = eqx.field(static=True)

    def __init__(
        self,
        members: Sequence[AbstractPreparedAtomisticSampler],
        /,
        *,
        route: AbstractOrdinalRoute | None = None,
    ) -> None:
        samplers = tuple(members)
        if not samplers or any(
            not isinstance(member, AbstractPreparedAtomisticSampler)
            for member in samplers
        ):
            raise TypeError("members must be prepared atomistic samplers.")
        if len({member.sampler_id for member in samplers}) != len(samplers):
            raise ValueError("Family members must be distinct samplers.")
        specs = {member.payload_spec.spec_id for member in samplers}
        if len(specs) != 1:
            raise ValueError("Family members must emit one payload spec.")
        conditioning = {member.support.conditioning.spec_id for member in samplers}
        if len(conditioning) != 1:
            raise ValueError("Family members must share one conditioning spec.")
        if len({member.support.periodic for member in samplers}) != 1:
            raise ValueError("Family members must agree on periodic support.")
        route_ = AtomisticFixedRoute() if route is None else route
        if route_.member_count != len(samplers):
            raise ValueError("The route must choose among exactly the family members.")
        self.members = samplers
        self.route = route_
        self.family_id = canonical_fingerprint(
            {
                "kind": "atomistic-sampler-family",
                "members": [member.sampler_id for member in samplers],
                "route": route_.route_id,
            }
        )

    @property
    def atom_capacity(self) -> int:
        return max(member.support.atom_capacity for member in self.members)

    @property
    def payload_spec(self) -> AtomisticPayloadSpec:
        return self.members[0].payload_spec


@final
class AtomisticProposalBank(StrictModule):
    """Bounded pending results addressed by ``ordinal mod capacity``."""

    occupied: Array
    ordinals: Array
    members: Array
    positions: Array
    species: Array
    atom_mask: Array
    cell: Array | None
    status: Array
    filter_bits: Array
    filter_values: Array
    payload: tuple[Array, ...]

    @property
    def capacity(self) -> int:
        return int(self.occupied.shape[0])


@final
class AtomisticGenerationCursor(StrictModule):
    """Durable counters of one stream (0-d integer arrays).

    ``first_ordinal`` starts the chain, ``watermark`` is the first unpublished
    ordinal and ``next_ordinal`` the first unadmitted one.  ``target``
    (accepted proposals) and ``budget`` (the absolute exclusive ordinal limit)
    are the continuation limits of the current segment chain; ``segment``
    counts `extend` calls.  ``active`` counts admitted, unharvested lanes over
    every participant (in-flight work that a stop retains) and ``receipts``
    the committed publication receipts.  Every participant of a placed stream
    holds the same cursor.
    """

    first_ordinal: Array
    next_ordinal: Array
    watermark: Array
    committed_accepted: Array
    committed_rejected: Array
    target: Array
    budget: Array
    segment: Array
    transitions: Array
    stop_reason: Array
    active: Array
    receipts: Array


@final
class AtomisticGenerationState(StrictModule):
    """Complete array-bearing stream continuation.

    ``root_key_data``, ``bank``, ``cursor`` and ``receipt_digest`` (the
    SHA-256 chain over committed receipts) are replicated on every
    participant of a placed stream; ``lanes`` are this participant's own
    sampler lanes, including admitted lanes retained across a stop.
    """

    root_key_data: Array
    lanes: tuple[AtomisticSamplerLaneState, ...]
    bank: AtomisticProposalBank
    cursor: AtomisticGenerationCursor
    receipt_digest: Array
    stream_id: str = eqx.field(static=True)

    @property
    def stop_reason(self) -> AtomisticGenerationStopReason:
        return AtomisticGenerationStopReason(int(self.cursor.stop_reason))


@final
class AtomisticGeneratedProposal(StrictModule, NonTrainableState):
    """Host record of one terminal proposal in its stream.

    ``positions``/``species`` hold the active atoms only; ``status`` bits and
    ``filter_bits`` (one bit per declared filter index) retain every rejection
    reason, so rejected proposals are published, never silently dropped.
    """

    proposal_id: str = eqx.field(static=True)
    ordinal: int = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    sampler_id: str = eqx.field(static=True)
    member: int = eqx.field(static=True)
    status: int = eqx.field(static=True)
    filter_bits: int = eqx.field(static=True)
    positions: np.ndarray
    species: np.ndarray
    slots: np.ndarray
    cell: np.ndarray | None
    filter_values: np.ndarray
    payload: tuple[tuple[str, np.ndarray], ...]

    @property
    def accepted(self) -> bool:
        return self.status == 0

    def value(self, name: str, /) -> np.ndarray:
        for key, value in self.payload:
            if key == name:
                return value
        raise KeyError(name)


@final
class AtomisticGenerationReceipt(StrictModule, NonTrainableState):
    """Durable publication of one ordinal-contiguous run of proposals."""

    sink_id: str = eqx.field(static=True)
    first_ordinal: int = eqx.field(static=True)
    proposal_ids: tuple[str, ...] = eqx.field(static=True)
    durable_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        sink_id: str,
        first_ordinal: int,
        proposal_ids: Sequence[str],
        durable_id: str,
    ) -> None:
        ids = tuple(_nonempty(value, "proposal_id") for value in proposal_ids)
        if not ids:
            raise ValueError("A receipt covers at least one proposal.")
        self.sink_id = _nonempty(sink_id, "sink_id")
        self.first_ordinal = _nonnegative_int(first_ordinal, "first_ordinal")
        self.proposal_ids = ids
        self.durable_id = _nonempty(durable_id, "durable_id")

    @property
    def count(self) -> int:
        return len(self.proposal_ids)


class AbstractAtomisticGenerationSink(abc.ABC):
    """Host publication owner of committed proposals (idempotent by proposal id)."""

    @property
    @abc.abstractmethod
    def sink_id(self) -> str: ...

    @abc.abstractmethod
    def publish(
        self, proposals: Sequence[AtomisticGeneratedProposal], /
    ) -> AtomisticGenerationReceipt | None:
        """Durably publish all proposals, or nothing (``None`` = backpressure)."""

    @abc.abstractmethod
    def published(self, proposal_ids: Sequence[str], /) -> tuple[bool, ...]:
        """Durable membership of proposal identities (restart reconciliation)."""


@final
class AtomisticMemoryGenerationSink(AbstractAtomisticGenerationSink):
    """Explicit host in-memory output policy (not durable across processes).

    Publication is idempotent by proposal id; ``records`` keeps publication
    order.  Use a store sink when results must survive a restart.
    """

    def __init__(self, sink_id: str, /) -> None:
        self._sink_id = _nonempty(sink_id, "sink_id")
        self._records: list[AtomisticGeneratedProposal] = []
        self._known: set[str] = set()

    @property
    def sink_id(self) -> str:
        return self._sink_id

    @property
    def records(self) -> tuple[AtomisticGeneratedProposal, ...]:
        return tuple(self._records)

    def publish(
        self, proposals: Sequence[AtomisticGeneratedProposal], /
    ) -> AtomisticGenerationReceipt | None:
        items = tuple(proposals)
        if not items:
            raise ValueError("A publication contains at least one proposal.")
        for item in items:
            if item.proposal_id not in self._known:
                self._records.append(item)
                self._known.add(item.proposal_id)
        ids = tuple(item.proposal_id for item in items)
        return AtomisticGenerationReceipt(
            sink_id=self._sink_id,
            first_ordinal=items[0].ordinal,
            proposal_ids=ids,
            durable_id=canonical_fingerprint(
                {
                    "kind": "atomistic-memory-sink-receipt",
                    "sink": self._sink_id,
                    "ids": ids,
                }
            ),
        )

    def published(self, proposal_ids: Sequence[str], /) -> tuple[bool, ...]:
        return tuple(value in self._known for value in proposal_ids)


ProposalRecordBuilder: TypeAlias = Callable[
    [AtomisticStructureStoreSchema, AtomisticGeneratedProposal], AtomisticStructureRecord
]


class AtomisticStructureStoreGenerationSink(AbstractAtomisticGenerationSink):
    """Structure-store sink: one append revision per published run.

    The appended commit is the durable receipt.  ``record_builder`` maps a
    proposal to one typed record whose ``record_id`` is the proposal id.
    """

    def __init__(
        self,
        root: str | PathLike[str],
        writer: AtomisticStructureStoreWriter,
        record_builder: ProposalRecordBuilder,
        /,
        *,
        limits: AtomisticStoreLimits,
    ) -> None:
        if not isinstance(writer, AtomisticStructureStoreWriter):
            raise TypeError("writer must be an AtomisticStructureStoreWriter.")
        if not callable(record_builder):
            raise TypeError("record_builder must be callable.")
        self._root = Path(root)
        self._writer = writer
        self._builder = record_builder
        self._limits = limits
        source = open_atomistic_structure_store(self._root, limits=limits)
        self._head: AtomisticStoreCommit = source.commit
        self._known = set(source.record_ids)

    @property
    def sink_id(self) -> str:
        return self._head.store_id

    @property
    def head(self) -> AtomisticStoreCommit:
        return self._head

    def publish(
        self, proposals: Sequence[AtomisticGeneratedProposal], /
    ) -> AtomisticGenerationReceipt | None:
        items = tuple(proposals)
        if not items:
            raise ValueError("A publication contains at least one proposal.")
        fresh = [item for item in items if item.proposal_id not in self._known]
        if fresh:
            schema = self._head.schema
            records = [self._builder(schema, item) for item in fresh]
            if any(
                record.record_id != item.proposal_id
                for record, item in zip(records, fresh, strict=True)
            ):
                raise ValueError("Store records must use the proposal id as record id.")
            self._head = self._writer.append(
                self._head, AtomisticStructureSource(schema, records)
            )
            self._known.update(item.proposal_id for item in fresh)
        return AtomisticGenerationReceipt(
            sink_id=self.sink_id,
            first_ordinal=items[0].ordinal,
            proposal_ids=tuple(item.proposal_id for item in items),
            durable_id=self._head.commit_id,
        )

    def published(self, proposal_ids: Sequence[str], /) -> tuple[bool, ...]:
        return tuple(value in self._known for value in proposal_ids)


@final
class AtomisticGenerationStreamPlan(StrictModule, NonTrainableState):
    """Bounded resumable stream over a sampler family.

    ``lanes_per_member`` fixes resident lanes of every member;
    ``pending_capacity`` bounds the pending bank (at least the total lane
    count).  ``steps_per_transition`` bounds stateful work between harvests.
    Filters are applied in declared order to support-admitted proposals.
    """

    family: AtomisticSamplerFamily
    conditions: AbstractConditionSource | None
    transforms: tuple[AtomisticConditionTransform, ...]
    filters: tuple[AbstractProposalFilter, ...]
    campaign_id: str = eqx.field(static=True)
    lanes_per_member: int = eqx.field(static=True)
    pending_capacity: int = eqx.field(static=True)
    steps_per_transition: int = eqx.field(static=True)
    stream_id: str = eqx.field(static=True)

    def __init__(
        self,
        samplers: AbstractPreparedAtomisticSampler | AtomisticSamplerFamily,
        /,
        *,
        campaign_id: str,
        lanes_per_member: int,
        pending_capacity: int,
        steps_per_transition: int = 1,
        conditions: AbstractConditionSource | None = None,
        transforms: Sequence[AtomisticConditionTransform] = (),
        filters: Sequence[AbstractProposalFilter] = (),
    ) -> None:
        family = (
            samplers
            if isinstance(samplers, AtomisticSamplerFamily)
            else AtomisticSamplerFamily((samplers,))
        )
        if not isinstance(family, AtomisticSamplerFamily):
            raise TypeError("samplers must be a prepared sampler or a sampler family.")
        lanes = _positive_int(lanes_per_member, "lanes_per_member")
        pending = _positive_int(pending_capacity, "pending_capacity")
        if pending < lanes * len(family.members):
            raise ValueError("pending_capacity must cover every resident lane.")
        steps = _positive_int(steps_per_transition, "steps_per_transition")
        transforms_ = tuple(transforms)
        filters_ = tuple(filters)
        if len(filters_) > _MAXIMUM_FILTERS:
            raise ValueError(f"At most {_MAXIMUM_FILTERS} filters are admitted.")
        spec = family.members[0].support.conditioning
        if spec.fields:
            if not isinstance(conditions, AbstractConditionSource):
                raise ValueError("A conditioned support requires a condition source.")
            expected = conditions.spec.spec_id
            for transform in transforms_:
                if transform.input_spec.spec_id != expected:
                    raise ValueError("Condition transforms must chain their specs.")
                expected = transform.output_spec.spec_id
            if expected != spec.spec_id:
                raise ValueError(
                    "Conditions must arrive at the support conditioning spec."
                )
        elif conditions is not None or transforms_:
            raise ValueError(
                "An unconditioned support takes no conditions or transforms."
            )
        self.family = family
        self.conditions = conditions
        self.transforms = transforms_
        self.filters = filters_
        self.campaign_id = _nonempty(campaign_id, "campaign_id")
        self.lanes_per_member = lanes
        self.pending_capacity = pending
        self.steps_per_transition = steps
        self.stream_id = canonical_fingerprint(
            {
                "kind": "atomistic-generation-stream",
                "campaign_id": self.campaign_id,
                "family": family.family_id,
                "supports": [member.support.support_id for member in family.members],
                "conditions": None if conditions is None else conditions.source_id,
                "transforms": [item.transform_id for item in transforms_],
                "filters": [item.filter_id for item in filters_],
                "lanes_per_member": lanes,
                "pending_capacity": pending,
                "steps_per_transition": steps,
            }
        )

    def prepare(
        self, *, placement: AtomisticGenerationPlacement | None = None
    ) -> PreparedAtomisticGenerationStream:
        """Bind the stream; without ``placement`` it is its own single participant."""
        _require_x64()
        return PreparedAtomisticGenerationStream(self, placement=placement)


def _scalar(value: int, /) -> Array:
    return jnp.asarray(value, dtype=jnp.int64)


# Cursor words, receipt-agreement words and the bytes of each agreed receipt
# identity text (sink id, durable id).
_CURSOR_WORDS = 12
_RECEIPT_WORDS = 5
_RECEIPT_TEXT_BYTES = 256
_DIGEST_BYTES = 32


class _ReceiptAgreement(IntEnum):
    """Authority row of one receipt agreement."""

    ABSENT = 0
    PRESENT = 1
    OVERSIZED = 2
    INVALID = 3


def _vacant_bank(
    plan: AtomisticGenerationStreamPlan, rows: int, /
) -> AtomisticProposalBank:
    """Empty bank-layout rows; the pending bank and every exchange share the layout."""
    atoms = plan.family.atom_capacity
    dtype = jnp.float64 if x64_enabled() else jnp.float32
    periodic = plan.family.members[0].support.periodic
    return AtomisticProposalBank(
        occupied=jnp.zeros((rows,), dtype=jnp.bool_),
        ordinals=jnp.full((rows,), -1, dtype=jnp.int64),
        members=jnp.zeros((rows,), dtype=jnp.int32),
        positions=jnp.zeros((rows, atoms, 3), dtype=dtype),
        species=jnp.zeros((rows, atoms), dtype=jnp.int32),
        atom_mask=jnp.zeros((rows, atoms), dtype=jnp.bool_),
        cell=jnp.zeros((rows, 3, 3), dtype=dtype) if periodic else None,
        status=jnp.zeros((rows,), dtype=jnp.int32),
        filter_bits=jnp.zeros((rows,), dtype=jnp.int32),
        filter_values=jnp.zeros((rows, len(plan.filters)), dtype=dtype),
        payload=plan.family.payload_spec.zeros(rows),
    )


def _bank_bytes(plan: AtomisticGenerationStreamPlan, rows: int, /) -> int:
    shapes = jax.eval_shape(functools.partial(_vacant_bank, plan, rows))
    return sum(
        math.prod(leaf.shape) * leaf.dtype.itemsize for leaf in jax.tree.leaves(shapes)
    )


def _receipt_origin(stream_id: str, /) -> Array:
    digest = hashlib.sha256(f"atomistic-generation-receipts:{stream_id}".encode())
    return jnp.asarray(np.frombuffer(digest.digest(), dtype=np.uint8).copy())


def _chain_receipt(digest: Array, receipt: AtomisticGenerationReceipt, /) -> Array:
    """Next receipt-chain digest after committing ``receipt``."""
    chain = hashlib.sha256(np.asarray(digest, dtype=np.uint8).tobytes())
    chain.update(
        canonical_json(
            {
                "sink_id": receipt.sink_id,
                "first_ordinal": receipt.first_ordinal,
                "count": receipt.count,
                "durable_id": receipt.durable_id,
            }
        ).encode("utf-8")
    )
    return jnp.asarray(np.frombuffer(chain.digest(), dtype=np.uint8).copy())


@final
class PreparedAtomisticGenerationStream(StrictModule, NonTrainableState):
    """Host transactions of one stream: start, segment, ready, commit, extend, persist.

    ``placement`` is None for a stream that is its own single participant; an
    `AtomisticGenerationPlacement` spreads the lanes over the participants of
    an execution group, every participant calling the same transactions in
    the same order (module docs).  ``round_bytes`` (gathered bytes of the
    largest exchange) and ``replicated_bytes`` (cursor, pending bank, root key
    and receipt chain resident on every participant) are charged against the
    placement's declared bounds when the stream is prepared.
    """

    plan: AtomisticGenerationStreamPlan
    placement: AtomisticGenerationPlacement | None
    atom_capacity: int = eqx.field(static=True)
    round_bytes: int = eqx.field(static=True)
    replicated_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        plan: AtomisticGenerationStreamPlan,
        /,
        *,
        placement: AtomisticGenerationPlacement | None = None,
    ) -> None:
        if not isinstance(plan, AtomisticGenerationStreamPlan):
            raise TypeError("plan must be an AtomisticGenerationStreamPlan.")
        if placement is not None and not isinstance(
            placement, AtomisticGenerationPlacement
        ):
            raise TypeError("placement must be an AtomisticGenerationPlacement or None.")
        participants = 1 if placement is None else placement.participants
        rows = len(plan.family.members) * plan.lanes_per_member
        if plan.pending_capacity < participants * rows:
            raise ValueError(
                "pending_capacity must cover every resident lane of every participant."
            )
        consensus = 8 * _CURSOR_WORDS + jr.key_data(jr.key(0)).nbytes + _DIGEST_BYTES
        transition_bytes = _bank_bytes(plan, rows) + rows + consensus
        receipt_bytes = 8 * _RECEIPT_WORDS + 2 * _RECEIPT_TEXT_BYTES
        round_bytes = participants * max(transition_bytes, receipt_bytes)
        replicated_bytes = _bank_bytes(plan, plan.pending_capacity) + consensus
        if placement is not None:
            placement.admit(round_bytes=round_bytes, replicated_bytes=replicated_bytes)
        self.plan = plan
        self.placement = placement
        self.atom_capacity = plan.family.atom_capacity
        self.round_bytes = round_bytes
        self.replicated_bytes = replicated_bytes

    # -- identity and templates ------------------------------------------------

    @property
    def stream_id(self) -> str:
        return self.plan.stream_id

    def identity(self) -> dict[str, str]:
        """Structure identities a persisted continuation must match on restore.

        ``samplers`` binds the members' scientific sampler identities.
        ``revisions`` binds the numeric revisions of every executed
        parameter-bearing owner: the sampler members, the condition source and
        each condition transform.  Pending and active lanes were conditioned
        and drawn under exactly these values, so any change refuses an exact
        restore.  ``executables`` binds the ordered typed `signature_facts` of
        the members (lane layout, capacities, precision): a sampler identity is
        independent of these execution choices, so a continuation written under
        another executable layout refuses rather than restoring into it.
        Continuations whose recorded identities predate this binding refuse
        and must be prepared again.
        """
        plan = self.plan
        members = plan.family.members
        return {
            "stream": plan.stream_id,
            "samplers": canonical_fingerprint([m.sampler_id for m in members]),
            "revisions": canonical_fingerprint(
                {
                    "samplers": [m.numeric_revision().revision_id for m in members],
                    "conditions": (
                        None
                        if plan.conditions is None
                        else plan.conditions.numeric_revision().revision_id
                    ),
                    "transforms": [
                        item.numeric_revision().revision_id for item in plan.transforms
                    ],
                }
            ),
            "executables": canonical_fingerprint(
                {
                    "kind": "atomistic-generation-executables",
                    "members": [dict(m.signature_facts()) for m in members],
                }
            ),
        }

    def template(self) -> AtomisticGenerationState:
        """Restore template: vacant lanes, empty bank, zero cursor and receipt chain."""
        zero = _scalar(0)
        return AtomisticGenerationState(
            root_key_data=jr.key_data(jr.key(0)),
            lanes=tuple(
                member.vacant(self.plan.lanes_per_member)
                for member in self.plan.family.members
            ),
            bank=_vacant_bank(self.plan, self.plan.pending_capacity),
            cursor=AtomisticGenerationCursor(*([zero] * _CURSOR_WORDS)),
            receipt_digest=jnp.zeros((_DIGEST_BYTES,), dtype=jnp.uint8),
            stream_id=self.plan.stream_id,
        )

    def _require_state(self, state: AtomisticGenerationState, /) -> None:
        if not isinstance(state, AtomisticGenerationState):
            raise TypeError("state must be an AtomisticGenerationState.")
        if state.stream_id != self.plan.stream_id:
            raise ValueError("Generation state belongs to another stream.")

    def _participant(self) -> int:
        return 0 if self.placement is None else self.placement.position

    def _authority(self) -> bool:
        return self.placement is None or self.placement.is_authority

    # -- lifecycle --------------------------------------------------------------

    def _require_limit(self, first: int, limit: int, /) -> None:
        """Admit the absolute ordinal window ``[first, limit)``."""
        if limit >= 2**62:
            raise ValueError("attempt ordinals must stay below 2**62.")
        source = self.plan.conditions
        if (
            isinstance(source, AtomisticConditionSchedule)
            and source.policy == "bounded"
            and (
                first < source.first_ordinal
                or limit > source.first_ordinal + source.row_count
            )
        ):
            raise ValueError(
                "A bounded condition schedule admits only ordinals of its declared rows."
            )

    def start(
        self,
        root_key: PRNGKey,
        /,
        *,
        target: int,
        attempt_budget: int,
        first_ordinal: int = 0,
    ) -> AtomisticGenerationState:
        """Start at ``first_ordinal``; zero target or budget stops at once without drawing.

        Proposals are ordinal-addressed, so a stream over ``[0, 4)`` followed by
        an independent stream over ``[4, 8)`` reproduces ``[0, 8)`` exactly.
        Every participant of a placed stream starts with identical arguments;
        the first exchange refuses divergent cursors or root keys.
        """
        goal = _nonnegative_int(target, "target")
        budget = _nonnegative_int(attempt_budget, "attempt_budget")
        first = _nonnegative_int(first_ordinal, "first_ordinal")
        self._require_limit(first, first + budget)
        if (
            not isinstance(root_key, jax.Array)
            or not jax.dtypes.issubdtype(root_key.dtype, jax.dtypes.prng_key)
            or root_key.shape != ()
        ):
            raise TypeError("root_key must be one typed JAX key from jax.random.key.")
        template = self.template()
        reason = AtomisticGenerationStopReason.RUNNING
        if goal == 0:
            reason = AtomisticGenerationStopReason.ZERO_TARGET
        elif budget == 0:
            reason = AtomisticGenerationStopReason.ZERO_BUDGET
        cursor = eqx.tree_at(
            lambda c: (
                c.first_ordinal,
                c.next_ordinal,
                c.watermark,
                c.target,
                c.budget,
                c.stop_reason,
            ),
            template.cursor,
            (
                _scalar(first),
                _scalar(first),
                _scalar(first),
                _scalar(goal),
                _scalar(first + budget),
                _scalar(int(reason)),
            ),
        )
        return AtomisticGenerationState(
            root_key_data=jr.key_data(root_key),
            lanes=template.lanes,
            bank=template.bank,
            cursor=cursor,
            receipt_digest=_receipt_origin(self.plan.stream_id),
            stream_id=self.plan.stream_id,
        )

    def extend(
        self, state: AtomisticGenerationState, /, *, target: int, attempt_budget: int
    ) -> AtomisticGenerationState:
        """Next finite continuation segment with raised, never lowered, limits.

        ``target`` counts accepted proposals and ``attempt_budget`` attempts,
        both over the whole chain from its first ordinal.
        """
        self._require_state(state)
        goal = _nonnegative_int(target, "target")
        first = int(state.cursor.first_ordinal)
        limit = first + _nonnegative_int(attempt_budget, "attempt_budget")
        if goal < int(state.cursor.target) or limit < int(state.cursor.budget):
            raise ValueError("A continuation segment cannot lower its target or budget.")
        self._require_limit(first, limit)
        cursor = eqx.tree_at(
            lambda c: (c.target, c.budget, c.segment, c.stop_reason),
            state.cursor,
            (
                _scalar(goal),
                _scalar(limit),
                state.cursor.segment + 1,
                _scalar(int(AtomisticGenerationStopReason.RUNNING)),
            ),
        )
        return eqx.tree_at(lambda s: s.cursor, state, cursor)

    def _root(self, state: AtomisticGenerationState, /) -> PRNGKey:
        return jr.wrap_key_data(state.root_key_data)

    # -- harvest and exchange ---------------------------------------------------

    def _harvest(
        self, state: AtomisticGenerationState, /
    ) -> tuple[AtomisticProposalBank, tuple[AtomisticSamplerLaneState, ...]]:
        """This participant's lanes as bank rows; terminal rows are occupied.

        Harvested lanes are released; every other lane stays bitwise unchanged.
        """
        layout = _vacant_bank(self.plan, self.plan.lanes_per_member)
        rows = []
        lanes = []
        for index, member in enumerate(self.plan.family.members):
            current = state.lanes[index]
            done = member.terminal(current)
            block = member.emit(current)
            status, bits, values = screen_atomistic_proposals(
                block, member.support, self.plan.filters
            )
            pad = self.atom_capacity - block.positions.shape[1]
            cell = None
            if layout.cell is not None:
                if block.cell is None:
                    raise ValueError(
                        "Periodic proposal publication requires cell vectors."
                    )
                cell = block.cell.astype(layout.cell.dtype)
            rows.append(
                AtomisticProposalBank(
                    occupied=done,
                    ordinals=block.ordinals,
                    members=jnp.full(done.shape, index, dtype=jnp.int32),
                    positions=jnp.pad(block.positions, ((0, 0), (0, pad), (0, 0))).astype(
                        layout.positions.dtype
                    ),
                    species=jnp.pad(block.species, ((0, 0), (0, pad))),
                    atom_mask=jnp.pad(block.atom_mask, ((0, 0), (0, pad))),
                    cell=cell,
                    status=status,
                    filter_bits=bits,
                    filter_values=values.astype(layout.filter_values.dtype),
                    payload=tuple(
                        value.astype(like.dtype)
                        for value, like in zip(block.payload, layout.payload, strict=True)
                    ),
                )
            )
            lanes.append(member.release(current, done))
        merged = jax.tree.map(lambda *parts: jnp.concatenate(parts), *rows)
        return merged, tuple(lanes)

    def _insert(
        self, bank: AtomisticProposalBank, rows: AtomisticProposalBank, /
    ) -> AtomisticProposalBank:
        """Write occupied rows at ``ordinal mod capacity`` (reserved by admission)."""
        capacity = self.plan.pending_capacity
        slots = jnp.where(rows.occupied, rows.ordinals % capacity, capacity)
        return jax.tree.map(
            lambda target, value: target.at[slots].set(
                value.astype(target.dtype), mode="drop"
            ),
            bank,
            rows,
        )

    def _consensus(
        self, state: AtomisticGenerationState, /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Replicated cursor words, root key data and receipt chain of this participant."""
        words = np.asarray(
            [np.asarray(leaf) for leaf in jax.tree.leaves(jax.device_get(state.cursor))],
            dtype=np.int64,
        )
        return (
            words,
            np.asarray(state.root_key_data),
            np.asarray(state.receipt_digest),
        )

    def _exchange(
        self,
        state: AtomisticGenerationState,
        rows: AtomisticProposalBank,
        free: Array,
        /,
    ) -> tuple[AtomisticProposalBank, np.ndarray]:
        """Rows and ``(participants, lanes)`` free masks of every participant.

        A placed stream runs its one exchange here and refuses participants
        whose replicated cursor, root key or receipt chain differ.
        """
        free_host = np.asarray(free)
        if self.placement is None:
            return rows, free_host[None]
        leaves, treedef = jax.tree.flatten(jax.device_get(rows))
        count = len(leaves)
        gathered = self.placement.gather((*leaves, free_host, *self._consensus(state)))
        if any(not np.all(words == words[:1]) for words in gathered[count + 1 :]):
            raise AtomisticGenerationDivergenceError(
                "Participants hold different replicated stream states; every "
                "participant must start, extend, publish and commit identically."
            )
        merged = jax.tree.unflatten(
            treedef, [value.reshape(-1, *value.shape[2:]) for value in gathered[:count]]
        )
        return merged, gathered[count]

    # -- admission ----------------------------------------------------------------

    def _contiguous(self, state: AtomisticGenerationState, /) -> tuple[int, int, int]:
        """Host scan from the watermark: ``(ready count, ready accepted, cut)``.

        ``cut`` is the ordinal after which nothing may be published because
        the target is reached inside the ready run (``-1`` when not reached).
        """
        bank = state.bank
        occupied = np.asarray(bank.occupied)
        ordinals = np.asarray(bank.ordinals)
        status = np.asarray(bank.status)
        capacity = self.plan.pending_capacity
        watermark = int(state.cursor.watermark)
        needed = int(state.cursor.target) - int(state.cursor.committed_accepted)
        if needed <= 0:
            # The target is already committed: nothing more is publishable.
            return 0, 0, watermark - 1
        ready = accepted = 0
        cut = -1
        for offset in range(capacity):
            ordinal = watermark + offset
            slot = ordinal % capacity
            if not (occupied[slot] and ordinals[slot] == ordinal):
                break
            ready += 1
            if status[slot] == 0:
                accepted += 1
                if accepted >= needed:
                    cut = ordinal
                    break
        return ready, accepted, cut

    def _admit(
        self, state: AtomisticGenerationState, free: np.ndarray, /
    ) -> AtomisticGenerationState:
        """Allocate the next ordinals of the window to free lanes of every participant.

        Free lanes of each member are ordered participant-major, then by lane;
        candidates take them in ordinal order until one finds its routed member
        full, so admission stays contiguous.  Every participant derives the same
        allocation and admits only its own picks; keys and conditions depend on
        the ordinal alone.
        """
        cursor = state.cursor
        next_ordinal = int(cursor.next_ordinal)
        limit = min(
            int(cursor.budget), int(cursor.watermark) + self.plan.pending_capacity
        )
        if next_ordinal >= limit:
            return state
        family = self.plan.family
        width = self.plan.lanes_per_member
        available = [
            [
                (participant, lane)
                for participant in range(free.shape[0])
                for lane in np.flatnonzero(
                    free[participant, index * width : (index + 1) * width]
                ).tolist()
            ]
            for index in range(len(family.members))
        ]
        window = min(limit - next_ordinal, sum(len(values) for values in available))
        if window == 0:
            return state
        root = self._root(state)
        candidates = jnp.arange(next_ordinal, next_ordinal + window, dtype=jnp.int64)
        routes = np.asarray(family.route.members(root, candidates))
        me = self._participant()
        taken = [0] * len(family.members)
        chosen: list[list[tuple[int, int]]] = [[] for _ in family.members]
        admitted = 0
        for offset in range(window):
            member = int(routes[offset])
            if taken[member] == len(available[member]):
                break
            participant, lane = available[member][taken[member]]
            taken[member] += 1
            admitted += 1
            if participant == me:
                chosen[member].append((lane, next_ordinal + offset))
        if admitted == 0:
            return state
        lanes = list(state.lanes)
        for index, picks in enumerate(chosen):
            if not picks:
                continue
            sampler = family.members[index]
            current = lanes[index]
            count = current.lanes
            mask_host = np.zeros((count,), dtype=bool)
            ordinal_host = np.asarray(current.ordinals).copy()
            for lane, ordinal in picks:
                mask_host[lane] = True
                ordinal_host[lane] = ordinal
            ordinals = jnp.asarray(ordinal_host, dtype=jnp.int64)
            # Vacant lanes get the address of their own recorded ordinal; only
            # masked lanes are read by admit.
            keys = proposal_keys(
                root,
                proposal_address(
                    self.plan.campaign_id,
                    sampler.support.support_id,
                    sampler.sampler_id,
                    role="proposal",
                ),
                ordinals,
            )
            conditioning = None
            if self.plan.conditions is not None:
                conditioning = self.plan.conditions.conditioning(
                    ordinals, keys, atom_capacity=sampler.support.atom_capacity
                )
                for transform in self.plan.transforms:
                    conditioning = transform(conditioning, keys)
            lanes[index] = sampler.admit(
                current, jnp.asarray(mask_host), keys, ordinals, conditioning
            )
        cursor = eqx.tree_at(
            lambda c: (c.next_ordinal, c.active),
            cursor,
            (_scalar(next_ordinal + admitted), cursor.active + admitted),
        )
        return eqx.tree_at(lambda s: (s.lanes, s.cursor), state, (tuple(lanes), cursor))

    # -- transitions ----------------------------------------------------------------

    def _stopped(self, state: AtomisticGenerationState, /) -> AtomisticGenerationState:
        """Record the stop reason once the target or the budget is settled.

        A reached target stops admission only: admitted lanes and pending
        results stay in the continuation, so `extend` resumes them exactly and
        no drawn ordinal or physical progress is discarded.
        """
        cursor = state.cursor
        if int(cursor.stop_reason) != int(AtomisticGenerationStopReason.RUNNING):
            return state
        _, _, cut = self._contiguous(state)
        if cut >= 0 or int(cursor.committed_accepted) >= int(cursor.target):
            reason = AtomisticGenerationStopReason.TARGET_REACHED
        elif int(cursor.next_ordinal) >= int(cursor.budget) and int(cursor.active) == 0:
            reason = AtomisticGenerationStopReason.BUDGET_EXHAUSTED
        else:
            return state
        return eqx.tree_at(lambda s: s.cursor.stop_reason, state, _scalar(int(reason)))

    def transition(self, state: AtomisticGenerationState, /) -> AtomisticGenerationState:
        """One harvest -> exchange -> stop check -> admit -> advance transaction."""
        self._require_state(state)
        if int(state.cursor.stop_reason) != int(AtomisticGenerationStopReason.RUNNING):
            return state
        rows, lanes = _harvest_jit(self, state)
        free = jnp.concatenate([~current.active for current in lanes])
        merged, free_all = self._exchange(state, rows, free)
        bank = _insert_jit(self, state.bank, merged)
        cursor = eqx.tree_at(
            lambda c: c.active, state.cursor, _scalar(int(np.count_nonzero(~free_all)))
        )
        state = eqx.tree_at(
            lambda s: (s.lanes, s.bank, s.cursor), state, (lanes, bank, cursor)
        )
        state = self._stopped(state)
        if int(state.cursor.stop_reason) != int(AtomisticGenerationStopReason.RUNNING):
            return state
        state = self._admit(state, free_all)
        lanes = tuple(
            _advance_jit(member, current, self.plan.steps_per_transition)
            for member, current in zip(self.plan.family.members, state.lanes, strict=True)
        )
        cursor = eqx.tree_at(
            lambda c: c.transitions, state.cursor, state.cursor.transitions + 1
        )
        state = eqx.tree_at(lambda s: (s.lanes, s.cursor), state, (lanes, cursor))
        return self._stopped(state)

    def segment(
        self, state: AtomisticGenerationState, /, *, maximum_transitions: int
    ) -> AtomisticGenerationState:
        """At most ``maximum_transitions`` transitions (bounded resident work)."""
        count = _nonnegative_int(maximum_transitions, "maximum_transitions")
        for _ in range(count):
            if int(state.cursor.stop_reason) != int(
                AtomisticGenerationStopReason.RUNNING
            ):
                break
            state = self.transition(state)
        return state

    # -- publication --------------------------------------------------------------

    def ready(
        self, state: AtomisticGenerationState, /
    ) -> tuple[AtomisticGeneratedProposal, ...]:
        """Ordinal-contiguous publishable proposals from the watermark."""
        self._require_state(state)
        ready, _, cut = self._contiguous(state)
        watermark = int(state.cursor.watermark)
        if cut >= 0:
            ready = cut - watermark + 1
        bank = jax.device_get(state.bank)
        family = self.plan.family
        capacity = self.plan.pending_capacity
        records = []
        for offset in range(ready):
            ordinal = watermark + offset
            slot = ordinal % capacity
            member = family.members[int(bank.members[slot])]
            mask = np.asarray(bank.atom_mask[slot])
            slots = np.flatnonzero(mask)
            records.append(
                AtomisticGeneratedProposal(
                    proposal_id=atomistic_proposal_id(
                        self.plan.campaign_id,
                        member.support.support_id,
                        member.sampler_id,
                        ordinal,
                    ),
                    ordinal=ordinal,
                    campaign_id=self.plan.campaign_id,
                    support_id=member.support.support_id,
                    sampler_id=member.sampler_id,
                    member=int(bank.members[slot]),
                    status=int(bank.status[slot]),
                    filter_bits=int(bank.filter_bits[slot]),
                    positions=np.asarray(bank.positions[slot][slots]),
                    species=np.asarray(bank.species[slot][slots]),
                    slots=slots.astype(np.int64),
                    cell=None if bank.cell is None else np.asarray(bank.cell[slot]),
                    filter_values=np.asarray(bank.filter_values[slot]),
                    payload=tuple(
                        (name, np.asarray(value[slot]))
                        for name, value in zip(
                            family.payload_spec.names, bank.payload, strict=True
                        )
                    ),
                )
            )
        return tuple(records)

    def _receipt_error(
        self,
        state: AtomisticGenerationState,
        ready: Sequence[AtomisticGeneratedProposal],
        receipt: AtomisticGenerationReceipt,
        /,
    ) -> str | None:
        """Why ``receipt`` does not cover the ready run at the watermark (None if it does)."""
        if receipt.first_ordinal != int(state.cursor.watermark) or receipt.count > len(
            ready
        ):
            return "The receipt does not cover the ready run at the watermark."
        expected = tuple(item.proposal_id for item in ready[: receipt.count])
        if expected != receipt.proposal_ids:
            return "The receipt names other proposals than the ready run."
        return None

    def commit(
        self, state: AtomisticGenerationState, receipt: AtomisticGenerationReceipt, /
    ) -> AtomisticGenerationState:
        """Advance the watermark over a durably published ordinal run.

        The receipt extends the replicated receipt chain; in a placed stream
        every participant commits the agreed receipt identically.
        """
        self._require_state(state)
        if not isinstance(receipt, AtomisticGenerationReceipt):
            raise TypeError("receipt must be an AtomisticGenerationReceipt.")
        ready = self.ready(state)
        error = self._receipt_error(state, ready, receipt)
        if error is not None:
            raise ValueError(error)
        published = ready[: receipt.count]
        accepted = sum(1 for item in published if item.accepted)
        capacity = self.plan.pending_capacity
        slots = jnp.asarray(
            [item.ordinal % capacity for item in published], dtype=jnp.int32
        )
        bank = state.bank
        bank = eqx.tree_at(
            lambda b: (b.occupied, b.ordinals),
            bank,
            (
                bank.occupied.at[slots].set(False),
                bank.ordinals.at[slots].set(-1),
            ),
        )
        cursor = state.cursor
        cursor = eqx.tree_at(
            lambda c: (
                c.watermark,
                c.committed_accepted,
                c.committed_rejected,
                c.receipts,
            ),
            cursor,
            (
                _scalar(receipt.first_ordinal + receipt.count),
                cursor.committed_accepted + accepted,
                cursor.committed_rejected + (receipt.count - accepted),
                cursor.receipts + 1,
            ),
        )
        return self._stopped(
            eqx.tree_at(
                lambda s: (s.bank, s.cursor, s.receipt_digest),
                state,
                (bank, cursor, _chain_receipt(state.receipt_digest, receipt)),
            )
        )

    def _require_sink(self, sink: AbstractAtomisticGenerationSink | None, /) -> None:
        if self._authority():
            if not isinstance(sink, AbstractAtomisticGenerationSink):
                raise TypeError("The publication authority must supply its sink.")
        elif sink is not None:
            raise ValueError(
                "Only the publication authority hosts the sink; other participants "
                "pass None."
            )

    def _agree(
        self,
        state: AtomisticGenerationState,
        ready: Sequence[AtomisticGeneratedProposal],
        receipt: AtomisticGenerationReceipt | None,
        /,
    ) -> AtomisticGenerationReceipt | None:
        """The authority's receipt as every participant's identical receipt.

        Placed streams run one exchange; an invalid or oversized authority
        receipt is refused on every participant alike (the durable publication
        is then committed by identity through `reconcile`).
        """
        if self.placement is None:
            return receipt
        words = np.zeros((_RECEIPT_WORDS,), dtype=np.int64)
        text = np.zeros((2 * _RECEIPT_TEXT_BYTES,), dtype=np.uint8)
        if receipt is not None:
            sink_id = receipt.sink_id.encode("utf-8")
            durable = receipt.durable_id.encode("utf-8")
            if self._receipt_error(state, ready, receipt) is not None:
                words[0] = _ReceiptAgreement.INVALID
            elif max(len(sink_id), len(durable)) > _RECEIPT_TEXT_BYTES:
                words[0] = _ReceiptAgreement.OVERSIZED
            else:
                words[:] = (
                    _ReceiptAgreement.PRESENT,
                    receipt.count,
                    receipt.first_ordinal,
                    len(sink_id),
                    len(durable),
                )
                text[: len(sink_id)] = np.frombuffer(sink_id, dtype=np.uint8)
                text[_RECEIPT_TEXT_BYTES : _RECEIPT_TEXT_BYTES + len(durable)] = (
                    np.frombuffer(durable, dtype=np.uint8)
                )
        all_words, all_text = self.placement.gather((words, text))
        if np.any(all_words[1:] != 0) or np.any(all_text[1:] != 0):
            raise AtomisticGenerationDivergenceError(
                "Only the publication authority holds a sink receipt."
            )
        agreed, identity = all_words[0], all_text[0]
        decision = _ReceiptAgreement(int(agreed[0]))
        match decision:
            case _ReceiptAgreement.ABSENT:
                return None
            case _ReceiptAgreement.OVERSIZED:
                raise AtomisticGenerationCapacityError(
                    f"A sink receipt identity exceeds {_RECEIPT_TEXT_BYTES} bytes."
                )
            case _ReceiptAgreement.INVALID:
                raise ValueError(
                    "The authority's sink receipt does not cover the ready run."
                )
            case _ReceiptAgreement.PRESENT:
                count = int(agreed[1])
                return AtomisticGenerationReceipt(
                    sink_id=bytes(identity[: int(agreed[3])]).decode("utf-8"),
                    first_ordinal=int(agreed[2]),
                    proposal_ids=tuple(item.proposal_id for item in ready[:count]),
                    durable_id=bytes(
                        identity[
                            _RECEIPT_TEXT_BYTES : _RECEIPT_TEXT_BYTES + int(agreed[4])
                        ]
                    ).decode("utf-8"),
                )
            case _:
                assert_never(decision)

    def publish(
        self,
        state: AtomisticGenerationState,
        sink: AbstractAtomisticGenerationSink | None,
        /,
    ) -> tuple[AtomisticGenerationState, AtomisticGenerationReceipt | None]:
        """Publish the ready run through the authority's sink; commit on its receipt.

        The authority supplies ``sink``; other participants pass None and
        return the same agreed receipt.  ``None`` is backpressure: nothing
        durable, the watermark and every pending result stay.
        """
        self._require_state(state)
        self._require_sink(sink)
        ready = self.ready(state)
        if not ready:
            return state, None
        receipt = None if sink is None else sink.publish(ready)
        agreed = self._agree(state, ready, receipt)
        if agreed is None:
            return state, None
        return self.commit(state, agreed), agreed

    def run(
        self,
        state: AtomisticGenerationState,
        sink: AbstractAtomisticGenerationSink | None,
        /,
        *,
        maximum_transitions: int,
    ) -> AtomisticGenerationState:
        """Bounded transitions with publication after each; returns the continuation."""
        count = _nonnegative_int(maximum_transitions, "maximum_transitions")
        self._require_sink(sink)
        state, _ = self.publish(state, sink)
        for _ in range(count):
            if self.finished(state):
                break
            state = self.transition(state)
            state, _ = self.publish(state, sink)
        return state

    def finished(self, state: AtomisticGenerationState, /) -> bool:
        """Stopped with every publishable pending result committed."""
        self._require_state(state)
        if int(state.cursor.stop_reason) == int(AtomisticGenerationStopReason.RUNNING):
            return False
        return not self.ready(state)

    def shortfall(self, state: AtomisticGenerationState, /) -> int:
        """Accepted proposals still missing from the target (zero when reached)."""
        return max(0, int(state.cursor.target) - int(state.cursor.committed_accepted))

    def reconcile(
        self,
        state: AtomisticGenerationState,
        sink: AbstractAtomisticGenerationSink | None,
        /,
    ) -> AtomisticGenerationState:
        """Restart reconciliation: commit ready proposals the sink already holds.

        A crash between a durable sink publication and the next checkpoint
        leaves published proposals pending; they are committed again by
        identity, never republished as new records.  The authority supplies
        ``sink``; other participants pass None and commit the agreed prefix.
        """
        self._require_state(state)
        self._require_sink(sink)
        ready = self.ready(state)
        if not ready:
            return state
        receipt = None
        if sink is not None:
            known = sink.published([item.proposal_id for item in ready])
            prefix = 0
            while prefix < len(ready) and known[prefix]:
                prefix += 1
            if prefix:
                receipt = AtomisticGenerationReceipt(
                    sink_id=sink.sink_id,
                    first_ordinal=ready[0].ordinal,
                    proposal_ids=tuple(item.proposal_id for item in ready[:prefix]),
                    durable_id="reconciled:" + sink.sink_id,
                )
        agreed = self._agree(state, ready, receipt)
        if agreed is None:
            return state
        return self.commit(state, agreed)

    # -- persistence --------------------------------------------------------------

    def _provenance(self) -> ArrayArtifactProvenance:
        return ArrayArtifactProvenance(
            "phydrax.atomistic.generation-stream",
            (self.plan.stream_id,),
            # Declared support profiles, not released scientific qualifications.
            tuple(sorted({m.support.support_id for m in self.plan.family.members})),
            tuple(
                sorted({m.support.length_unit.unit_id for m in self.plan.family.members})
            ),
        )

    def _require_local(self) -> None:
        if self.placement is not None:
            raise ValueError(
                "A placed stream persists through publish_checkpoint and "
                "restore_checkpoint."
            )

    def write_checkpoint(
        self, path: str | PathLike[str], state: AtomisticGenerationState, /
    ) -> ArrayArtifactReceipt:
        """Persist the complete continuation through the typed array-artifact owner."""
        self._require_local()
        self._require_state(state)
        return write_typed_array_artifact(
            Path(path),
            jax.device_get(state),
            artifact_kind="atomistic-generation-state",
            provenance=self._provenance(),
            structure_ids=self.identity(),
        )

    def read_checkpoint(self, path: str | PathLike[str], /) -> AtomisticGenerationState:
        """Restore a continuation; any stream, sampler or revision change refuses."""
        self._require_local()
        restored, _ = read_typed_array_artifact(
            Path(path),
            jax.device_get(self.template()),
            artifact_kind="atomistic-generation-state",
            provenance=self._provenance(),
            structure_ids=self.identity(),
        )
        return jax.tree.map(jnp.asarray, restored)

    def _require_placement(self) -> AtomisticGenerationPlacement:
        if self.placement is None:
            raise ValueError(
                "Only a placed stream persists through the distributed lifecycle "
                "commit; use write_checkpoint for a single-participant stream."
            )
        return self.placement

    def _checkpoint_ids(
        self, placement: AtomisticGenerationPlacement, /
    ) -> tuple[str, str, str]:
        """``(analysis, numeric revision, execution)`` identities of a placed checkpoint."""
        identity = self.identity()
        execution = canonical_fingerprint(
            {
                "kind": "atomistic-generation-execution",
                "stream": identity["stream"],
                "samplers": identity["samplers"],
                "executables": identity["executables"],
                "placement": placement.placement_id,
            }
        )
        return identity["stream"], identity["revisions"], execution

    def publish_checkpoint(
        self,
        repository: ArtifactRepository,
        state: AtomisticGenerationState,
        /,
        *,
        checkpoint_id: str,
        writer_id: str,
        attempt_id: str | None = None,
        parent_manifest: CheckpointManifest | None = None,
        encoding: ChunkEncoding = "identity",
        limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    ) -> CheckpointManifest:
        """Collective durable continuation of a placed stream with one global commit.

        Every participant calls this at the same transaction boundary.  The
        replicated root key, pending bank, cursor (allocation and counters)
        and receipt chain are written once; each participant's lanes
        (including retained in-flight work) are its own block.  No exchange is
        in flight between transactions, so no further transport state exists.
        """
        placement = self._require_placement()
        self._require_state(state)
        shared = (state.root_key_data, state.bank, state.cursor, state.receipt_digest)
        for leaf in jax.tree.leaves((shared, state.lanes)):
            if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
                raise TypeError(
                    "Generation continuations persist key data, not typed keys."
                )
        analysis, revision, execution = self._checkpoint_ids(placement)
        return publish_placement_checkpoint(
            repository,
            placement,
            checkpoint_id,
            analysis_id=analysis,
            revision_id=revision,
            execution_id=execution,
            replicated=tuple(jax.tree.leaves(jax.device_get(shared))),
            participant=tuple(jax.tree.leaves(jax.device_get(state.lanes))),
            writer_id=writer_id,
            attempt_id=attempt_id,
            parent_manifest=parent_manifest,
            encoding=encoding,
            limits=limits,
        )

    def restore_checkpoint(
        self,
        repository: ArtifactRepository,
        checkpoint_id: str,
        /,
        *,
        parent_manifest: CheckpointManifest | None = None,
        limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
    ) -> AtomisticGenerationState:
        """Restore a globally committed placed continuation on the same placement.

        The current restore guarantee is the same stream profile (including
        lanes and pending capacity), sampler, condition-source and transform
        revisions and placement; any change is refused with
        `AtomisticGenerationCheckpointError`.  Fresh-run proposal identities are
        independent of placement and lane count.
        """
        placement = self._require_placement()
        template = jax.device_get(self.template())
        shared, shared_def = jax.tree.flatten(
            (
                template.root_key_data,
                template.bank,
                template.cursor,
                template.receipt_digest,
            )
        )
        owned, owned_def = jax.tree.flatten(template.lanes)
        analysis, revision, execution = self._checkpoint_ids(placement)
        restored_shared, restored_owned, _ = restore_placement_checkpoint(
            repository,
            placement,
            checkpoint_id,
            analysis_id=analysis,
            revision_id=revision,
            execution_id=execution,
            replicated=tuple(shared),
            participant=tuple(owned),
            parent_manifest=parent_manifest,
            limits=limits,
        )
        root, bank, cursor, digest = jax.tree.unflatten(
            shared_def, [jnp.asarray(value) for value in restored_shared]
        )
        lanes = jax.tree.unflatten(
            owned_def, [jnp.asarray(value) for value in restored_owned]
        )
        return AtomisticGenerationState(
            root_key_data=root,
            lanes=lanes,
            bank=bank,
            cursor=cursor,
            receipt_digest=digest,
            stream_id=self.plan.stream_id,
        )


@eqx.filter_jit
def _harvest_jit(
    stream: PreparedAtomisticGenerationStream, state: AtomisticGenerationState
) -> tuple[AtomisticProposalBank, tuple[AtomisticSamplerLaneState, ...]]:
    return stream._harvest(state)


@eqx.filter_jit
def _insert_jit(
    stream: PreparedAtomisticGenerationStream,
    bank: AtomisticProposalBank,
    rows: AtomisticProposalBank,
) -> AtomisticProposalBank:
    return stream._insert(bank, rows)


@eqx.filter_jit
def _advance_jit(
    sampler: AbstractPreparedAtomisticSampler,
    state: AtomisticSamplerLaneState,
    steps: int,
) -> AtomisticSamplerLaneState:
    return sampler.advance(state, steps)


__all__ = [
    "AbstractAtomisticGenerationSink",
    "AbstractConditionSource",
    "AbstractOrdinalRoute",
    "AbstractPreparedAtomisticSampler",
    "AbstractProposalFilter",
    "AtomisticCellAdmission",
    "AtomisticCellBounds",
    "AtomisticChargeBalanceQualifier",
    "AtomisticChemistryAdmission",
    "AtomisticCompositionBounds",
    "AtomisticConditionSchedule",
    "AtomisticConditionTransform",
    "AtomisticCoordinateRepresentation",
    "AtomisticFilterEvidence",
    "AtomisticFixedRoute",
    "AtomisticGeneratedProposal",
    "AtomisticGenerationCursor",
    "AtomisticGenerationReceipt",
    "AtomisticGenerationState",
    "AtomisticGenerationStopReason",
    "AtomisticGenerationStreamPlan",
    "AtomisticGenerationSupport",
    "AtomisticMemoryGenerationSink",
    "AtomisticMinimumDistanceFilter",
    "AtomisticPayloadSpec",
    "AtomisticProposalBank",
    "AtomisticProposalBlock",
    "AtomisticProposalStatus",
    "AtomisticSamplerFamily",
    "AtomisticSamplerLaneState",
    "AtomisticStructureStoreGenerationSink",
    "AtomisticTopologyAdmission",
    "AtomisticWeightedRoute",
    "GENERATION_ADDRESS_NAMESPACE",
    "PreparedAtomisticGenerationStream",
    "atomistic_proposal_id",
    "map_admitted_lanes",
    "ordinal_words",
    "proposal_address",
    "proposal_keys",
    "role_keys",
    "screen_atomistic_proposals",
]
