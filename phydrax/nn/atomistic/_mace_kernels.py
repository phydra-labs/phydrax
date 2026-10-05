#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Accelerated Pallas Mosaic GPU kernels for MACE's radial-weighted ``uvu`` coupling.

The accelerated operation is the seeded receiver aggregation of ONE prepared
streamed edge fragment. On a fragment of ``F`` lanes, ``R`` receiver slots and
``S`` fragment-local source rows it computes, for every receiver slot ``r``,

    A[r, (b, o), k] = seed[r] + sum_{l -> r} a_l sum_{p -> b} R[l, p, k]
                      sum_{(o, i', j)} C_p[o, i', j] H[s(l), i', k] Y[l, j],

the radial-weighted O(3) coupling of the fragment's source rows ``H`` with its
lane harmonics ``Y``, streamed in stable lane order into the incoming
``(high, correction)`` accumulator of each receiver slot. Receiver, source and
lane extents are independent; nothing in the kernel family is shaped by the
graph's node or edge capacity. The streamed relation owns the fragment
schedule, carries each receiver's seeded accumulator across its fragments,
generates local radial weights and harmonics inside the fragment and applies
the receiver epilogue once after the final fragment. Fragment source rows are
the caller's gather of loop-invariant node features, so the source cotangent
returns to the single node cotangent through that gather's transpose.

Static model structure lowers into the kernels: packed column placement, the
coupling rows ``(o, i', j)`` of every nonzero entry of the executed coefficient
tables (their constructor-prepared support, imported residuals included), and
tiles. Coupling coefficient values, radial path weights, source rows,
harmonics, lane masks and fragment routing are explicit array operands, so a
parameter update changes the numeric revision without changing the executable
signature.

Kernels use Mosaic GPU lane semantics: one warpgroup owns a channel tile of a
multiple of 128 lanes, and every operand is laid out channel-minor
(``[row, column, channel]``) so each gathered row is one contiguous vector.
Packed multiplicity-major layouts convert to and from that view outside the
kernels with fragment-sized transposes; channels pad to whole tiles with zeros.

Ownership:

- ``"receiver"``: one program owns a receiver-slot tile and a channel tile,
  streams each slot's target-major lanes in stable order into its seed and
  commits every slot exactly once.
- ``"source"``: the transpose in the source rows, owned by fragment-local
  sources over the fragment's source-major lane order; each source row is
  written exactly once.
- ``"radial"``, ``"harmonic"``: lane-owned cotangents that recompute the local
  coupling per lane instead of retaining lane messages.
- ``"coefficient"``: a fixed number of reduction programs stride the lanes;
  its partials are ``[1 or 2, programs, channel tiles, coefficients]`` (each
  program's high and, when compensated, correction) regardless of the
  fragment's lane count.

The lane-reducing roles (receiver, source, coefficient) also accept a seeded
``(high, correction)`` accumulator, which carries one declared accumulation
through every active slot of a multi-slot tangent.

No kernel materializes per-lane messages or graph-wide edge data.
"""

from __future__ import annotations

from collections.abc import Callable, Set
from functools import partial
from typing import Any, final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax._src import callback as _jax_callback, effects as _jax_effects
from jax.core import ShapedArray
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import core as jax_core
from jax.interpreters import mlir
from jax.typing import ArrayLike

from ..._execution_resources import ExecutionResourceEvidence
from ..._execution_sampling import compiled_memory_estimate
from ..._fingerprint import canonical_fingerprint
from ..._identity import ExecutableSignature
from ..._numerics._compensated import two_sum
from ..._strict import StrictModule
from ..._validation import nonnegative_integer, positive_integer
from ...backends._types import BackendUnavailableError
from ...backends.atomistic import (
    AtomisticAccelerationTarget,
    AtomisticKernelAdmission,
    AtomisticKernelPrecision,
    AtomisticKernelRequest,
    AtomisticKernelTarget,
    MACECouplingRole,
)
from ...sparse._execution import KeyGroupAccumulation, RelationAccumulation
from ...sparse._streamed import StreamedFragment
from ...typing import parse
from ..operator.layers import O3TensorProduct
from ..operator.representations import O3IrrepLayout
from ._mace_kernel_derivatives import (
    coupling_input_roles,
    MACE_COUPLING_INPUT_SLOTS,
    MACE_COUPLING_SCHEDULE_OPERANDS,
    MACE_REDUCING_ROLES,
    register_mace_coupling_derivatives,
)


class MACECouplingRow(NamedTuple):
    """One executed (nonzero) coupling-table entry.

    Columns address the channel-minor operand views; ``component`` is the
    ``(output, source, harmonic)`` position in the path's dense table.
    """

    message_column: int
    source_column: int
    harmonic_column: int
    coefficient: int
    component: tuple[int, int, int]


class MACELayoutBlock(NamedTuple):
    """Packed offset, component count and channel-minor column base of a block."""

    offset: int
    dimension: int
    column: int


def _irrep_layout(layout: object, role: str, /) -> O3IrrepLayout:
    if not isinstance(layout, O3IrrepLayout):
        raise ValueError(
            f"The accelerated MACE coupling needs a real-spherical O3IrrepLayout {role} "
            "layout; Cartesian O3Representation products are not an admitted kernel route."
        )
    return layout


def _uniform_multiplicity(layout: O3IrrepLayout, expected: int, role: str, /) -> None:
    for block in layout.blocks:
        if block.multiplicity != expected:
            raise ValueError(
                f"The accelerated MACE coupling requires every {role} block to have "
                f"multiplicity {expected}; block {block.name!r} has {block.multiplicity}."
            )


def _layout_blocks(layout: O3IrrepLayout, /) -> tuple[MACELayoutBlock, ...]:
    blocks: list[MACELayoutBlock] = []
    column = 0
    for block in layout.blocks:
        blocks.append(MACELayoutBlock(layout.offset(block.name), block.dimension, column))
        column += block.dimension
    return tuple(blocks)


def _columns(blocks: tuple[MACELayoutBlock, ...], /) -> int:
    return sum(block.dimension for block in blocks)


@final
class MACEEdgeCouplingSpec(StrictModule):
    """Static kernel structure lowered from one admitted ``uvu`` tensor product.

    Coupling rows are exactly the constructor-prepared
    ``O3TensorProduct.coefficient_support``: every nonzero entry of the executed
    tables, including imported residuals where the native coupling is zero.
    The support is static structure, so tables that differ in support lower to
    different rows, coefficient counts, resources and ``spec_id``.
    """

    channels: int = eqx.field(static=True)
    source_blocks: tuple[MACELayoutBlock, ...] = eqx.field(static=True)
    harmonic_width: int = eqx.field(static=True)
    message_blocks: tuple[MACELayoutBlock, ...] = eqx.field(static=True)
    paths: tuple[tuple[MACECouplingRow, ...], ...] = eqx.field(static=True)
    coefficient_count: int = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    tensor_product_plan_id: str = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)

    def __init__(self, tensor_product: O3TensorProduct, /) -> None:
        if not isinstance(tensor_product, O3TensorProduct):
            raise TypeError("tensor_product must be an O3TensorProduct.")
        plan = tensor_product.plan
        source = _irrep_layout(plan.left_representation, "source")
        harmonic = _irrep_layout(plan.right_representation, "harmonic")
        message = _irrep_layout(plan.output_representation, "message")
        channels = source.blocks[0].multiplicity
        _uniform_multiplicity(source, channels, "source")
        _uniform_multiplicity(harmonic, 1, "harmonic")
        _uniform_multiplicity(message, channels, "message")
        source_blocks = _layout_blocks(source)
        message_blocks = _layout_blocks(message)
        paths: list[tuple[MACECouplingRow, ...]] = []
        coefficient = 0
        for index, (path, instruction, support) in enumerate(
            zip(
                plan.paths,
                plan.instructions,
                tensor_product.coefficient_support,
                strict=True,
            )
        ):
            if path.connection_mode != "uvu" or not path.weighted:
                raise ValueError(
                    "The accelerated MACE coupling admits only externally weighted 'uvu' "
                    f"paths; path {path.left!r} x {path.right!r} -> {path.output!r} is "
                    f"{path.connection_mode!r} with weighted={path.weighted}."
                )
            if instruction.weight_offset != index * channels:
                raise ValueError(
                    "Accelerated radial weights must be path-major with one channel row "
                    "per path."
                )
            if not support:
                raise ValueError(
                    f"Path {index} of the accelerated MACE coupling has an all-zero "
                    "coefficient table; omit the path instead."
                )
            right = harmonic.blocks[instruction.right_block]
            source_base = source_blocks[instruction.left_block].column
            message_base = message_blocks[instruction.output_block].column
            harmonic_base = harmonic.offset(right.name)
            paths.append(
                tuple(
                    MACECouplingRow(
                        message_base + o,
                        source_base + i,
                        harmonic_base + j,
                        coefficient + position,
                        (o, i, j),
                    )
                    for position, (o, i, j) in enumerate(support)
                )
            )
            coefficient += len(support)
        if not paths:
            raise ValueError("The accelerated MACE coupling needs at least one path.")
        self.channels = channels
        self.source_blocks = source_blocks
        self.harmonic_width = harmonic.packed_size
        self.message_blocks = message_blocks
        self.paths = tuple(paths)
        self.coefficient_count = coefficient
        self.maximum_degree = max(
            block.degree
            for layout in (source, harmonic, message)
            for block in layout.blocks
        )
        self.tensor_product_plan_id = plan.plan_id
        self.spec_id = canonical_fingerprint(
            {
                "kind": "mace-edge-coupling-kernel-structure",
                "tensor_product_plan_id": plan.plan_id,
                "channels": channels,
                "source_blocks": [list(block) for block in source_blocks],
                "harmonic_width": self.harmonic_width,
                "message_blocks": [list(block) for block in message_blocks],
                "paths": [[list(row) for row in rows] for rows in paths],
            }
        )

    @property
    def source_columns(self) -> int:
        return _columns(self.source_blocks)

    @property
    def message_columns(self) -> int:
        return _columns(self.message_blocks)

    @property
    def source_width(self) -> int:
        return self.channels * self.source_columns

    @property
    def message_width(self) -> int:
        return self.channels * self.message_columns

    @property
    def radial_width(self) -> int:
        return self.channels * len(self.paths)

    @property
    def coefficient_support(self) -> tuple[tuple[tuple[int, int, int], ...], ...]:
        """Per-path ``(output, source, harmonic)`` table entries the rows execute."""
        return tuple(tuple(row.component for row in rows) for rows in self.paths)

    def coefficients(self, tensor_product: O3TensorProduct, /) -> Array:
        """Flat coupling-coefficient operand gathered from the prepared tables.

        Every executed table entry is gathered. A table carrying a nonzero entry
        outside the recorded support (for example after an ``eqx.tree_at``
        replacement of the prepared tables) is refused in-graph rather than
        silently truncated; numeric tables are never read on the host.
        """
        if not isinstance(tensor_product, O3TensorProduct):
            raise TypeError("tensor_product must be an O3TensorProduct.")
        if tensor_product.plan.plan_id != self.tensor_product_plan_id:
            raise ValueError(
                "The tensor product does not realize the plan this kernel structure was "
                "lowered from."
            )
        if tensor_product.coefficient_support != self.coefficient_support:
            raise ValueError(
                "The tensor product's coefficient support differs from the support this "
                "kernel structure was lowered from; lower a structure for its tables."
            )
        if tensor_product.internal_weights:
            raise ValueError(
                "The accelerated MACE coupling consumes external per-edge radial weights; "
                "an internally weighted tensor product is not this operation."
            )
        pieces = []
        outside = jnp.zeros((), dtype=jnp.bool_)
        for rows, table in zip(
            self.paths, tensor_product.prepared.coefficients, strict=True
        ):
            index = np.asarray([row.component for row in rows], dtype=np.int32)
            pieces.append(table[index[:, 0], index[:, 1], index[:, 2]])
            unsupported = np.ones(table.shape, dtype=bool)
            unsupported[index[:, 0], index[:, 1], index[:, 2]] = False
            outside = outside | jnp.any(jnp.where(unsupported, table, 0) != 0)
        return eqx.error_if(
            jnp.concatenate(pieces),
            outside,
            "A prepared O(3) coefficient table has a nonzero entry outside its recorded "
            "coefficient_support; prepare the tensor product again.",
        )

    def workspace_elements(self, plan: MACEKernelPlan, /) -> dict[MACECouplingRole, int]:
        """Planned live per-program elements of every role at ``plan`` tiles.

        Counts carried (seeded) accumulators, memoized gathered rows of one
        lane, the path weights and product temporaries, and the strided
        coefficient program's scalar accumulators. This is the declared
        planning bound; actual register and spill allocation is
        compiler-determined.
        """
        if not isinstance(plan, MACEKernelPlan):
            raise TypeError("plan must be a MACEKernelPlan.")
        rows = tuple(row for path in self.paths for row in path)
        sources = len({row.source_column for row in rows})
        messages = len({row.message_column for row in rows})
        carried = 2 if plan.accumulation == "compensated" else 1
        tile = plan.channel_tile
        return {
            "receiver": tile * (carried * self.message_columns + sources + 2),
            "source": tile * (carried * self.source_columns + messages + 2),
            "radial": tile * (sources + messages + 2),
            "harmonic": tile * (sources + messages + 2) + self.harmonic_width,
            "coefficient": tile * (sources + messages + 2)
            + carried * self.coefficient_count,
        }

    def workspace_bytes(self, plan: MACEKernelPlan, /) -> int:
        """Largest planned per-program workspace over all roles, in bytes."""
        itemsize = np.dtype(plan.precision).itemsize
        return max(self.workspace_elements(plan).values()) * itemsize

    def to_channel_minor(
        self, blocks: tuple[MACELayoutBlock, ...], values: Array, capacity: int, /
    ) -> Array:
        """Packed ``[rows, C (2l+1) ...]`` to zero-padded ``[rows, columns, capacity]``."""
        rows = values.shape[0]
        pieces = [
            jnp.swapaxes(
                values[
                    :, block.offset : block.offset + self.channels * block.dimension
                ].reshape(rows, self.channels, block.dimension),
                1,
                2,
            )
            for block in blocks
        ]
        joined = jnp.concatenate(pieces, axis=1)
        return jnp.pad(joined, ((0, 0), (0, 0), (0, capacity - self.channels)))

    def from_channel_minor(
        self, blocks: tuple[MACELayoutBlock, ...], values: Array, /
    ) -> Array:
        """Inverse of `to_channel_minor`; padded channels are dropped."""
        rows = values.shape[0]
        pieces = [
            jnp.swapaxes(
                values[:, block.column : block.column + block.dimension, : self.channels],
                1,
                2,
            ).reshape(rows, self.channels * block.dimension)
            for block in blocks
        ]
        return jnp.concatenate(pieces, axis=1)


@final
class MACEKernelPlan(StrictModule):
    """Static execution choice of the accelerated MACE fragment coupling.

    ``channel_tile`` is a power-of-two multiple of the 128 warpgroup lanes.
    ``receiver_tile`` owner rows (receiver slots or fragment sources) and
    ``edge_tile`` lanes share one program. ``reduction_programs`` fixes the
    program count of the coefficient cotangent's strided lane reduction.
    ``fragment_budget_bytes`` is the caller's explicit admission budget for
    the declared operand and result bytes of one kernel bind over one
    fragment; a fragment above it is refused, never split by the kernel.
    """

    target: AtomisticAccelerationTarget = eqx.field(static=True)
    precision: AtomisticKernelPrecision = eqx.field(static=True)
    accumulation: RelationAccumulation = eqx.field(static=True)
    receiver_tile: int = eqx.field(static=True)
    edge_tile: int = eqx.field(static=True)
    channel_tile: int = eqx.field(static=True)
    reduction_programs: int = eqx.field(static=True)
    fragment_budget_bytes: int = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        target: AtomisticAccelerationTarget,
        precision: AtomisticKernelPrecision,
        receiver_tile: int,
        edge_tile: int,
        channel_tile: int,
        reduction_programs: int,
        fragment_budget_bytes: int,
        accumulation: RelationAccumulation = "deterministic",
        maximum_derivative_order: int = 2,
    ) -> None:
        self.target = parse(target, AtomisticAccelerationTarget, "target")
        self.precision = parse(precision, AtomisticKernelPrecision, "precision")
        self.accumulation = parse(accumulation, RelationAccumulation, "accumulation")
        self.receiver_tile = positive_integer(receiver_tile, "receiver_tile")
        self.edge_tile = positive_integer(edge_tile, "edge_tile")
        self.channel_tile = positive_integer(channel_tile, "channel_tile")
        self.reduction_programs = positive_integer(
            reduction_programs, "reduction_programs"
        )
        self.fragment_budget_bytes = positive_integer(
            fragment_budget_bytes, "fragment_budget_bytes"
        )
        self.maximum_derivative_order = nonnegative_integer(
            maximum_derivative_order, "maximum_derivative_order"
        )

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.precision)

    def channel_capacity(self, spec: MACEEdgeCouplingSpec, /) -> int:
        """Channel multiplicity padded to whole channel tiles."""
        return -(-spec.channels // self.channel_tile) * self.channel_tile


@final
class MACEFragmentExtent(StrictModule):
    """Independent receiver-slot, local-source and lane extents of one fragment."""

    receivers: int = eqx.field(static=True)
    sources: int = eqx.field(static=True)
    edges: int = eqx.field(static=True)

    def __init__(self, *, receivers: int, sources: int, edges: int) -> None:
        self.receivers = positive_integer(receivers, "receivers")
        self.sources = positive_integer(sources, "sources")
        self.edges = positive_integer(edges, "edges")


def fragment_extent(fragment: StreamedFragment, /) -> MACEFragmentExtent:
    """Extents of one fragment, or of every fragment of a stacked schedule."""
    if not isinstance(fragment, StreamedFragment):
        raise TypeError("fragment must be a StreamedFragment.")
    return MACEFragmentExtent(
        receivers=int(fragment.receiver_ids.shape[-1]),
        sources=int(fragment.source_ids.shape[-1]),
        edges=int(fragment.lane_routes.shape[-1]),
    )


def _slot_shape(
    role: MACECouplingRole,
    spec: MACEEdgeCouplingSpec,
    plan: MACEKernelPlan,
    receivers: int,
    sources: int,
    edges: int,
    /,
) -> tuple[int, ...]:
    capacity = plan.channel_capacity(spec)
    match role:
        case "coefficient":
            return (spec.coefficient_count,)
        case "radial":
            return (edges, len(spec.paths), capacity)
        case "source":
            return (sources, spec.source_columns, capacity)
        case "harmonic":
            return (edges, spec.harmonic_width)
        case "receiver":
            return (receivers, spec.message_columns, capacity)
        case _:
            raise ValueError(f"Unsupported MACE coupling role {role!r}.")


def _routing_shapes(
    receivers: int, sources: int, edges: int, /
) -> tuple[tuple[int, ...], ...]:
    """Kernel-order routing operand shapes; see `_routing_operands`."""
    return (
        (edges,),
        (edges,),
        (edges,),
        (receivers,),
        (receivers,),
        (edges,),
        (sources,),
        (sources,),
    )


_ROLES: tuple[MACECouplingRole, ...] = (
    "receiver",
    "source",
    "radial",
    "harmonic",
    "coefficient",
)


def _partial_pairs(plan: MACEKernelPlan, /) -> int:
    """Carried coefficient-partial components: high, plus correction if compensated."""
    return 2 if plan.accumulation == "compensated" else 1


@final
class MACEFragmentResources(StrictModule):
    """Declared logical bytes of the accelerated coupling over one fragment.

    Every figure follows from the static extents, coupling structure (the
    executed coefficient support), tiles and precision; none is shaped by a
    graph's node or edge capacity. ``role_bind_bytes`` counts each kernel
    role's routing operands, input slots and result; the reducing roles
    (receiver, source, coefficient) are counted in their seeded form, with the
    incoming ``(high, correction)`` seed and pair result that forward binds and
    multi-slot tangent chains execute, and the coefficient role also with its
    fixed ``[1 or 2, programs, channel tiles, coefficients]`` partials (two
    under compensated accumulation: each program's high and correction).
    ``peak_bind_bytes`` is their maximum and the admitted fragment figure.
    ``layout_bytes`` counts the channel-minor copies the boundary makes of the
    packed source rows, radial weights, seed and result. Compiler temporaries,
    kernel registers and shared memory are not included; compiled-executable
    memory is separate evidence.
    """

    receivers: int = eqx.field(static=True)
    sources: int = eqx.field(static=True)
    edges: int = eqx.field(static=True)
    channel_capacity: int = eqx.field(static=True)
    reduction_programs: int = eqx.field(static=True)
    routing_bytes: int = eqx.field(static=True)
    slot_bytes: tuple[tuple[str, int], ...] = eqx.field(static=True)
    seed_bytes: int = eqx.field(static=True)
    coefficient_partial_bytes: int = eqx.field(static=True)
    role_bind_bytes: tuple[tuple[str, int], ...] = eqx.field(static=True)
    peak_bind_bytes: int = eqx.field(static=True)
    layout_bytes: int = eqx.field(static=True)
    program_workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        spec: MACEEdgeCouplingSpec,
        plan: MACEKernelPlan,
        extent: MACEFragmentExtent,
        /,
    ) -> None:
        itemsize = plan.dtype.itemsize
        receivers, sources, edges = extent.receivers, extent.sources, extent.edges
        capacity = plan.channel_capacity(spec)
        slots = {
            role: int(np.prod(_slot_shape(role, spec, plan, receivers, sources, edges)))
            * itemsize
            for role in _ROLES
        }
        routing = 4 * sum(
            int(np.prod(shape)) for shape in _routing_shapes(receivers, sources, edges)
        )
        seed = 2 * slots["receiver"]
        partials = (
            _partial_pairs(plan)
            * plan.reduction_programs
            * (capacity // plan.channel_tile)
            * spec.coefficient_count
            * itemsize
        )
        binds: dict[str, int] = {}
        for role in _ROLES:
            inputs = sum(slots[slot] for slot in coupling_input_roles(role))
            if role in MACE_REDUCING_ROLES:
                # Seed in and (high, correction) pair out.
                binds[role] = routing + inputs + 2 * (2 * slots[role])
            else:
                binds[role] = routing + inputs + slots[role]
            if role == "coefficient":
                binds[role] += partials
        self.receivers = receivers
        self.sources = sources
        self.edges = edges
        self.channel_capacity = capacity
        self.reduction_programs = plan.reduction_programs
        self.routing_bytes = routing
        self.slot_bytes = tuple(sorted(slots.items()))
        self.seed_bytes = seed
        self.coefficient_partial_bytes = partials
        self.role_bind_bytes = tuple(sorted(binds.items()))
        self.peak_bind_bytes = max(binds.values())
        self.layout_bytes = slots["source"] + slots["radial"] + 2 * seed
        self.program_workspace_bytes = spec.workspace_bytes(plan)


# Kernel bodies -----------------------------------------------------------------


class _Tile(NamedTuple):
    start: Array
    size: int


def _strided(value: Array, tile: _Tile, /) -> Array:
    """``value`` in the warpgroup-strided lane layout of the channel carry."""
    return plgpu.layout_cast(value, plgpu.Layout.WG_STRIDED((tile.size,), vec_size=1))


def _lanes(tile: _Tile, dtype: np.dtype, /) -> Array:
    """Zero channel vector in the warpgroup-strided lane layout."""
    return _strided(jnp.zeros((tile.size,), dtype=dtype), tile)


def _row(ref: jax.Ref, row: Array, column: int, tile: _Tile, /) -> Array:
    return ref[row, column, pl.ds(tile.start, tile.size)]


def _channel_tile(plan: MACEKernelPlan, block: Array, /) -> _Tile:
    return _Tile(
        pl.multiple_of(block * plan.channel_tile, plan.channel_tile), plan.channel_tile
    )


type _Carry = tuple[tuple[Array, ...], tuple[Array, ...]]


def _accumulate(
    carry: _Carry, contributions: dict[int, Array], live: Array, compensated: bool, /
) -> _Carry:
    high, correction = carry
    next_high = list(high)
    next_correction = list(correction)
    for column, value in contributions.items():
        masked = jnp.where(live, value, jnp.zeros_like(value))
        if compensated:
            total, error = two_sum(next_high[column], masked)
            next_high[column] = total
            next_correction[column] = next_correction[column] + error
        else:
            next_high[column] = next_high[column] + masked
    return tuple(next_high), tuple(next_correction)


class _Owners(NamedTuple):
    """Owner rows of one owner-streamed role and their stable lane order.

    Owner ``row`` streams positions ``offsets[row] + i`` for ``i < counts[row]``;
    a position is a lane itself when ``order`` is None (owner-major lanes) and
    ``order[position]`` otherwise.
    """

    count: int
    offsets: jax.Ref
    counts: jax.Ref
    order: jax.Ref | None
    partners: jax.Ref
    active: jax.Ref


def _stream_owners(
    spec: MACEEdgeCouplingSpec,
    plan: MACEKernelPlan,
    role: MACECouplingRole,
    owners: _Owners,
    coefficient: jax.Ref,
    radial: jax.Ref,
    partner_values: jax.Ref,
    harmonic: jax.Ref,
    seed: jax.Ref | None,
    output: jax.Ref,
) -> None:
    """Receiver- or source-owned streamed aggregation of one owner/channel tile."""
    owner_block = lax.axis_index("owner")
    tile = _channel_tile(plan, lax.axis_index("channel"))
    window = pl.ds(tile.start, tile.size)
    coefficients = tuple(coefficient[q] for q in range(spec.coefficient_count))
    receiver = role == "receiver"
    owned_count = spec.message_columns if receiver else spec.source_columns
    compensated = plan.accumulation == "compensated"

    def route_step(start: Array, index: Array, carry: _Carry) -> _Carry:
        position = start + index
        lane = position if owners.order is None else owners.order[position]
        partner = owners.partners[lane]
        live = owners.active[lane] != 0
        partner_cache: dict[int, Array] = {}
        harmonic_cache: dict[int, Array] = {}
        contributions: dict[int, Array] = {}
        for path, rows in enumerate(spec.paths):
            weights = radial[lane, path, window]
            products: dict[int, Array] = {}
            for row in rows:
                owned = row.message_column if receiver else row.source_column
                other = row.source_column if receiver else row.message_column
                if other not in partner_cache:
                    partner_cache[other] = _row(partner_values, partner, other, tile)
                if row.harmonic_column not in harmonic_cache:
                    harmonic_cache[row.harmonic_column] = harmonic[
                        lane, row.harmonic_column
                    ]
                term = (
                    coefficients[row.coefficient] * harmonic_cache[row.harmonic_column]
                ) * partner_cache[other]
                products[owned] = (
                    term if owned not in products else products[owned] + term
                )
            for owned, product in products.items():
                value = weights * product
                contributions[owned] = (
                    value if owned not in contributions else contributions[owned] + value
                )
        return _accumulate(carry, contributions, live, compensated)

    def owner(local: Array, state: Array) -> Array:
        row = owner_block * plan.receiver_tile + local

        @pl.when(row < owners.count)
        def _() -> None:
            if seed is None:
                zeros = tuple(_lanes(tile, plan.dtype) for _ in range(owned_count))
                initial: _Carry = (zeros, zeros if compensated else ())
            else:
                # The incoming accumulator is the stream's start: every lane is
                # added to it in order, never to a collapsed fragment subtotal.
                initial = (
                    tuple(
                        _strided(seed[0, row, column, window], tile)
                        for column in range(owned_count)
                    ),
                    tuple(
                        _strided(seed[1, row, column, window], tile)
                        for column in range(owned_count)
                    )
                    if compensated
                    else (),
                )
            high, correction = lax.fori_loop(
                0, owners.counts[row], partial(route_step, owners.offsets[row]), initial
            )
            for column in range(owned_count):
                if seed is None:
                    output[row, column, window] = (
                        high[column] + correction[column] if compensated else high[column]
                    )
                else:
                    output[0, row, column, window] = high[column]
                    output[1, row, column, window] = (
                        correction[column]
                        if compensated
                        else seed[1, row, column, window]
                    )

        return state

    lax.fori_loop(0, plan.receiver_tile, owner, jnp.int32(0))


def _receiver_kernel(
    spec: MACEEdgeCouplingSpec,
    plan: MACEKernelPlan,
    receiver_count: int,
    seeded: bool,
    offsets: jax.Ref,
    counts: jax.Ref,
    sources: jax.Ref,
    active: jax.Ref,
    coefficient: jax.Ref,
    radial: jax.Ref,
    source: jax.Ref,
    harmonic: jax.Ref,
    *seed_and_output: jax.Ref,
) -> None:
    seed, output = seed_and_output if seeded else (None, seed_and_output[0])
    _stream_owners(
        spec,
        plan,
        "receiver",
        _Owners(receiver_count, offsets, counts, None, sources, active),
        coefficient,
        radial,
        source,
        harmonic,
        seed,
        output,
    )


def _source_kernel(
    spec: MACEEdgeCouplingSpec,
    plan: MACEKernelPlan,
    source_count: int,
    seeded: bool,
    order: jax.Ref,
    offsets: jax.Ref,
    counts: jax.Ref,
    slots: jax.Ref,
    active: jax.Ref,
    coefficient: jax.Ref,
    radial: jax.Ref,
    receiver: jax.Ref,
    harmonic: jax.Ref,
    *seed_and_output: jax.Ref,
) -> None:
    seed, output = seed_and_output if seeded else (None, seed_and_output[0])
    _stream_owners(
        spec,
        plan,
        "source",
        _Owners(source_count, offsets, counts, order, slots, active),
        coefficient,
        radial,
        receiver,
        harmonic,
        seed,
        output,
    )


class _Lane(NamedTuple):
    lane: Array
    source: Array
    receiver: Array
    live: Array


def _lane(
    lane: Array, edge_count: int, sources: jax.Ref, slots: jax.Ref, active: jax.Ref, /
) -> _Lane:
    safe = jnp.minimum(lane, edge_count - 1)
    live = (lane < edge_count) & (active[safe] != 0)
    return _Lane(safe, sources[safe], slots[safe], live)


def _lane_factors(
    edge: _Lane,
    row: MACECouplingRow,
    tile: _Tile,
    source: jax.Ref,
    message: jax.Ref,
    caches: tuple[dict[int, Array], dict[int, Array]],
    /,
) -> tuple[Array, Array]:
    """Gathered source and receiver-cotangent channel rows of one coupling row."""
    source_cache, message_cache = caches
    if row.source_column not in source_cache:
        source_cache[row.source_column] = _row(
            source, edge.source, row.source_column, tile
        )
    if row.message_column not in message_cache:
        message_cache[row.message_column] = _row(
            message, edge.receiver, row.message_column, tile
        )
    return source_cache[row.source_column], message_cache[row.message_column]


def _radial_kernel(
    spec: MACEEdgeCouplingSpec,
    plan: MACEKernelPlan,
    edge_count: int,
    sources: jax.Ref,
    slots: jax.Ref,
    active: jax.Ref,
    coefficient: jax.Ref,
    source: jax.Ref,
    harmonic: jax.Ref,
    message: jax.Ref,
    output: jax.Ref,
) -> None:
    edge_block = lax.axis_index("edge")
    tile = _channel_tile(plan, lax.axis_index("channel"))

    def step(local: Array, state: Array) -> Array:
        lane = edge_block * plan.edge_tile + local
        edge = _lane(lane, edge_count, sources, slots, active)

        @pl.when(lane < edge_count)
        def _() -> None:
            caches: tuple[dict[int, Array], dict[int, Array]] = ({}, {})
            for path, rows in enumerate(spec.paths):
                total: Array | None = None
                for row in rows:
                    source_values, message_values = _lane_factors(
                        edge, row, tile, source, message, caches
                    )
                    scale = (
                        coefficient[row.coefficient]
                        * harmonic[edge.lane, row.harmonic_column]
                    )
                    term = scale * (source_values * message_values)
                    total = term if total is None else total + term
                if total is None:
                    raise RuntimeError("A MACE coupling path has no coupling rows.")
                output[edge.lane, path, pl.ds(tile.start, tile.size)] = jnp.where(
                    edge.live, total, jnp.zeros_like(total)
                )

        return state

    lax.fori_loop(0, plan.edge_tile, step, jnp.int32(0))


def _harmonic_kernel(
    spec: MACEEdgeCouplingSpec,
    plan: MACEKernelPlan,
    edge_count: int,
    channel_capacity: int,
    sources: jax.Ref,
    slots: jax.Ref,
    active: jax.Ref,
    coefficient: jax.Ref,
    radial: jax.Ref,
    source: jax.Ref,
    message: jax.Ref,
    output: jax.Ref,
) -> None:
    edge_block = lax.axis_index("edge")
    used = tuple(sorted({row.harmonic_column for rows in spec.paths for row in rows}))
    position = {column: index for index, column in enumerate(used)}
    zero = jnp.zeros((), dtype=plan.dtype)

    def step(local: Array, state: Array) -> Array:
        lane = edge_block * plan.edge_tile + local
        edge = _lane(lane, edge_count, sources, slots, active)

        @pl.when(lane < edge_count)
        def _() -> None:
            def channel_block(block: Array, sums: tuple[Array, ...]) -> tuple[Array, ...]:
                tile = _channel_tile(plan, block)
                caches: tuple[dict[int, Array], dict[int, Array]] = ({}, {})
                updated = list(sums)
                for path, rows in enumerate(spec.paths):
                    weights = radial[edge.lane, path, pl.ds(tile.start, tile.size)]
                    for row in rows:
                        source_values, message_values = _lane_factors(
                            edge, row, tile, source, message, caches
                        )
                        index = position[row.harmonic_column]
                        updated[index] = updated[index] + coefficient[
                            row.coefficient
                        ] * jnp.sum(weights * (source_values * message_values))
                return tuple(updated)

            sums = lax.fori_loop(
                0,
                channel_capacity // plan.channel_tile,
                channel_block,
                tuple(zero for _ in used),
            )
            for column in range(spec.harmonic_width):
                value = sums[position[column]] if column in position else zero
                output[edge.lane, column] = jnp.where(edge.live, value, zero)

        return state

    lax.fori_loop(0, plan.edge_tile, step, jnp.int32(0))


def _coefficient_kernel(
    spec: MACEEdgeCouplingSpec,
    plan: MACEKernelPlan,
    edge_count: int,
    sources: jax.Ref,
    slots: jax.Ref,
    active: jax.Ref,
    radial: jax.Ref,
    source: jax.Ref,
    harmonic: jax.Ref,
    message: jax.Ref,
    output: jax.Ref,
) -> None:
    """Fixed-program strided lane reduction of the coefficient cotangent.

    Program ``g`` visits lanes ``g, g + G, g + 2G, ...`` in order, so the
    partials are ``[1 or 2, G, channel tiles, coefficients]`` for any lane
    count: each program's ``high`` and, under compensated accumulation, its
    uncollapsed ``correction``.
    """
    program = lax.axis_index("program")
    channel_block = lax.axis_index("channel")
    tile = _channel_tile(plan, channel_block)
    programs = plan.reduction_programs
    compensated = plan.accumulation == "compensated"
    zero = jnp.zeros((), dtype=plan.dtype)

    def step(stride: Array, carry: _Carry) -> _Carry:
        edge = _lane(program + stride * programs, edge_count, sources, slots, active)
        caches: tuple[dict[int, Array], dict[int, Array]] = ({}, {})
        high = list(carry[0])
        correction = list(carry[1])
        for path, rows in enumerate(spec.paths):
            weights = radial[edge.lane, path, pl.ds(tile.start, tile.size)]
            for row in rows:
                source_values, message_values = _lane_factors(
                    edge, row, tile, source, message, caches
                )
                value = jnp.where(
                    edge.live,
                    harmonic[edge.lane, row.harmonic_column]
                    * jnp.sum(weights * (source_values * message_values)),
                    zero,
                )
                if compensated:
                    total, error = two_sum(high[row.coefficient], value)
                    high[row.coefficient] = total
                    correction[row.coefficient] = correction[row.coefficient] + error
                else:
                    high[row.coefficient] = high[row.coefficient] + value
        return tuple(high), tuple(correction)

    zeros = tuple(zero for _ in range(spec.coefficient_count))
    high, correction = lax.fori_loop(
        0, -(-edge_count // programs), step, (zeros, zeros if compensated else ())
    )
    for index in range(spec.coefficient_count):
        output[0, program, channel_block, index] = high[index]
        if compensated:
            output[1, program, channel_block, index] = correction[index]


# Primitive ---------------------------------------------------------------------


class _InterpretedKernelEffect(_jax_effects.Effect):
    """Ordered host simulation of one interpreted kernel bind.

    The CPU Mosaic GPU interpreter executes every kernel through ordered host
    IO callbacks over one process-global simulated device, so binds must never
    overlap: the effect is ordered. A bind's result is a pure function of its
    operands, so loops, conditionals, custom derivatives and checkpoints may
    stage it. JAX does not recompute effectful equations inside a checkpoint;
    under reverse modes each interpreted bind's result is therefore kept as a
    replay residual, which CUDA binds (no effect) avoid by recomputation.
    """


_INTERPRETED_KERNEL = _InterpretedKernelEffect()
for _registry in (
    _jax_effects.ordered_effects,
    _jax_effects.lowerable_effects,
    _jax_effects.control_flow_allowed_effects,
    _jax_effects.custom_derivatives_allowed_effects,
    _jax_effects.remat_allowed_effects,
):
    _registry.add_type(_InterpretedKernelEffect)


def _result_shape(
    role: MACECouplingRole,
    spec: MACEEdgeCouplingSpec,
    plan: MACEKernelPlan,
    receivers: int,
    sources: int,
    edges: int,
    seeded: bool,
    /,
) -> tuple[int, ...]:
    shape = _slot_shape(role, spec, plan, receivers, sources, edges)
    return (2, *shape) if seeded else shape


def _abstract(
    *operands: ShapedArray,
    role: MACECouplingRole,
    spec: MACEEdgeCouplingSpec,
    plan: MACEKernelPlan,
    receiver_count: int,
    source_count: int,
    edge_count: int,
    seeded: bool,
    derivative_order: int,
) -> tuple[ShapedArray, Set[jax_core.Effect]]:
    extents = (receiver_count, source_count, edge_count)
    routing = operands[:MACE_COUPLING_SCHEDULE_OPERANDS]
    for index, (value, expected) in enumerate(
        zip(routing, _routing_shapes(*extents), strict=True)
    ):
        if value.shape != expected or value.dtype != jnp.int32:
            raise ValueError(
                f"MACE fragment routing operand {index} must be int32{list(expected)}; "
                f"got {value.dtype}{list(value.shape)}."
            )
    end = MACE_COUPLING_SCHEDULE_OPERANDS + MACE_COUPLING_INPUT_SLOTS
    for slot, value in zip(
        coupling_input_roles(role),
        operands[MACE_COUPLING_SCHEDULE_OPERANDS:end],
        strict=True,
    ):
        expected = _slot_shape(slot, spec, plan, *extents)
        if value.shape != expected or value.dtype != plan.dtype:
            raise ValueError(
                f"MACE coupling slot {slot!r} must be {plan.precision}{list(expected)}; "
                f"got {value.dtype}{list(value.shape)}."
            )
    if seeded:
        if role not in MACE_REDUCING_ROLES or len(operands) != end + 1:
            raise ValueError(
                "Only the lane-reducing receiver, source and coefficient roles take "
                "one incoming seed operand."
            )
        expected = _result_shape(role, spec, plan, *extents, True)
        if operands[end].shape != expected or operands[end].dtype != plan.dtype:
            raise ValueError(
                f"The MACE fragment seed must be {plan.precision}{list(expected)}; got "
                f"{operands[end].dtype}{list(operands[end].shape)}."
            )
    elif len(operands) != end:
        raise ValueError("An unseeded MACE coupling bind takes no seed operand.")
    output = ShapedArray(_result_shape(role, spec, plan, *extents, seeded), plan.dtype)
    match plan.target:
        case "cuda":
            return output, jax_core.no_effects
        case "cpu_interpret":
            # The Mosaic GPU interpreter simulates the device through ordered
            # host IO callbacks; a bind declares one replayable interpreter
            # effect in their place (see `_InterpretedKernelEffect`).
            kernel = jax.make_jaxpr(
                partial(
                    _coupling_impl,
                    role=role,
                    spec=spec,
                    plan=plan,
                    receiver_count=receiver_count,
                    source_count=source_count,
                    edge_count=edge_count,
                    seeded=seeded,
                    derivative_order=derivative_order,
                )
            )(*(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in operands))
            foreign = [
                effect
                for effect in kernel.effects
                if not isinstance(effect, _jax_callback.OrderedIOEffect)
            ]
            if foreign:
                raise RuntimeError(
                    "The Mosaic GPU interpreter kernel carries effects other than its "
                    f"ordered host simulation: {foreign}."
                )
            return output, {_INTERPRETED_KERNEL}
        case _:
            raise ValueError(f"Unsupported atomistic target {plan.target!r}.")


def _kernel(
    body: Callable[..., None],
    shape: tuple[int, ...],
    grid: tuple[int, ...],
    names: tuple[str, ...],
    plan: MACEKernelPlan,
    /,
) -> Callable[..., Array]:
    return plgpu.kernel(
        body,
        out_type=jax.ShapeDtypeStruct(shape, plan.dtype),
        grid=grid,
        grid_names=names,
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Lane
        ),
        interpret=plgpu.InterpretGPUParams() if plan.target == "cpu_interpret" else None,
    )


def _reduce_coefficient_partials(
    plan: MACEKernelPlan, partials: Array, seed: Array | None, /
) -> Array:
    """Reduce ``[1 or 2, programs, tiles, Q]`` partials into the coefficient result.

    ``deterministic`` sums the program highs in a fixed order onto the seed's
    high. ``compensated`` streams every program's high and correction, in
    program order, through one ``two_sum`` accumulator started at the seed; an
    unseeded result collapses it once, a seeded one returns the pair.
    """
    width = partials.shape[-1]
    if plan.accumulation != "compensated":
        total = jnp.sum(partials[0], axis=(0, 1))
        if seed is None:
            return total
        return jnp.stack((seed[0] + total, seed[1]))
    events = jnp.swapaxes(partials.reshape(2, -1, width), 0, 1).reshape(-1, width)
    initial = (
        (jnp.zeros((width,), partials.dtype), jnp.zeros((width,), partials.dtype))
        if seed is None
        else (seed[0], seed[1])
    )

    def step(
        carry: tuple[Array, Array], event: Array
    ) -> tuple[tuple[Array, Array], None]:
        total, error = two_sum(carry[0], event)
        return (total, carry[1] + error), None

    (high, correction), _ = lax.scan(step, initial, events)
    if seed is None:
        return high + correction
    return jnp.stack((high, correction))


def _coupling_impl(
    *operands: Array,
    role: MACECouplingRole,
    spec: MACEEdgeCouplingSpec,
    plan: MACEKernelPlan,
    receiver_count: int,
    source_count: int,
    edge_count: int,
    seeded: bool,
    derivative_order: int,
) -> Array:
    del derivative_order
    (
        lane_slots,
        lane_sources,
        lane_use,
        receiver_offsets,
        receiver_counts,
        source_lanes,
        source_offsets,
        source_counts,
    ) = operands[:MACE_COUPLING_SCHEDULE_OPERANDS]
    end = MACE_COUPLING_SCHEDULE_OPERANDS + MACE_COUPLING_INPUT_SLOTS
    slots = dict(
        zip(
            coupling_input_roles(role),
            operands[MACE_COUPLING_SCHEDULE_OPERANDS:end],
            strict=True,
        )
    )
    capacity = plan.channel_capacity(spec)
    channels = capacity // plan.channel_tile
    shape = _result_shape(
        role, spec, plan, receiver_count, source_count, edge_count, seeded
    )
    lanes = -(-edge_count // plan.edge_tile)
    match role:
        case "receiver":
            body = partial(_receiver_kernel, spec, plan, receiver_count, seeded)
            owners = -(-receiver_count // plan.receiver_tile)
            return _kernel(body, shape, (owners, channels), ("owner", "channel"), plan)(
                receiver_offsets,
                receiver_counts,
                lane_sources,
                lane_use,
                slots["coefficient"],
                slots["radial"],
                slots["source"],
                slots["harmonic"],
                *operands[end:],
            )
        case "source":
            body = partial(_source_kernel, spec, plan, source_count, seeded)
            owners = -(-source_count // plan.receiver_tile)
            return _kernel(body, shape, (owners, channels), ("owner", "channel"), plan)(
                source_lanes,
                source_offsets,
                source_counts,
                lane_slots,
                lane_use,
                slots["coefficient"],
                slots["radial"],
                slots["receiver"],
                slots["harmonic"],
                *operands[end:],
            )
        case "radial":
            body = partial(_radial_kernel, spec, plan, edge_count)
            return _kernel(body, shape, (lanes, channels), ("edge", "channel"), plan)(
                lane_sources,
                lane_slots,
                lane_use,
                slots["coefficient"],
                slots["source"],
                slots["harmonic"],
                slots["receiver"],
            )
        case "harmonic":
            body = partial(_harmonic_kernel, spec, plan, edge_count, capacity)
            return _kernel(body, shape, (lanes,), ("edge",), plan)(
                lane_sources,
                lane_slots,
                lane_use,
                slots["coefficient"],
                slots["radial"],
                slots["source"],
                slots["receiver"],
            )
        case "coefficient":
            body = partial(_coefficient_kernel, spec, plan, edge_count)
            partials = _kernel(
                body,
                (
                    _partial_pairs(plan),
                    plan.reduction_programs,
                    channels,
                    spec.coefficient_count,
                ),
                (plan.reduction_programs, channels),
                ("program", "channel"),
                plan,
            )(
                lane_sources,
                lane_slots,
                lane_use,
                slots["radial"],
                slots["source"],
                slots["harmonic"],
                slots["receiver"],
            )
            return _reduce_coefficient_partials(
                plan, partials, operands[end] if seeded else None
            )
        case _:
            raise ValueError(f"Unsupported MACE coupling role {role!r}.")


def _require_platform(plan: MACEKernelPlan, platform: str, /) -> None:
    match plan.target:
        case "cuda":
            required = "cuda"
        case "cpu_interpret":
            required = "cpu"
        case _:
            raise ValueError(f"Unsupported atomistic target {plan.target!r}.")
    if platform != required:
        raise BackendUnavailableError(
            "pallas-atomistic",
            f"atomistic.mace_edge_coupling on {platform}",
            f"target {plan.target!r} executes only on the {required!r} platform",
            "no accelerated kernel or substitute route is declared for this platform",
        )


def _eager_coupling(*operands: Array, **params: Any) -> Array:
    platform = jax.default_backend()
    _require_platform(params["plan"], "cuda" if platform == "gpu" else platform)
    return _coupling_impl(*operands, **params)


def _platform_lowering(platform: str, /) -> Callable[..., Any]:
    lowered = mlir.lower_fun(_coupling_impl, multiple_results=False)

    def rule(context: mlir.LoweringRuleContext, *values: Any, **params: Any) -> Any:
        plan: MACEKernelPlan = params["plan"]
        _require_platform(plan, platform)
        if plan.target != "cpu_interpret":
            return lowered(context, *values, **params)
        # Thread the bind's interpreter token through the simulation's ordered
        # host callbacks and hand the advanced token back under its own effect.
        ordered = _jax_callback._OrderedIOEffect
        simulation = context.replace(
            tokens_in=mlir.TokenSet(
                {ordered: context.tokens_in.get(_INTERPRETED_KERNEL)}
            ),
            tokens_out=None,
        )
        result = lowered(simulation, *values, **params)
        context.set_tokens_out(
            mlir.TokenSet({_INTERPRETED_KERNEL: simulation.tokens_out.get(ordered)})
        )
        return result

    return rule


_coupling_p = jax_core.Primitive("phydrax_mace_fragment_coupling")
_coupling_p.def_impl(_eager_coupling)
_coupling_p.def_effectful_abstract_eval(_abstract)
for _platform in ("cpu", "cuda", "rocm", "tpu"):
    mlir.register_lowering(_coupling_p, _platform_lowering(_platform), platform=_platform)
register_mace_coupling_derivatives(_coupling_p)


# Public boundary -----------------------------------------------------------------


def _float_operand(
    value: ArrayLike, name: str, shape: tuple[int, ...], plan: MACEKernelPlan, /
) -> Array:
    array = jnp.asarray(value)
    if array.dtype != plan.dtype:
        raise TypeError(
            f"{name} must have the plan precision {plan.precision}; got {array.dtype}. "
            "Convert explicitly before the accelerated coupling."
        )
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    return array


def _index_operand(value: ArrayLike, name: str, shape: tuple[int, ...], /) -> Array:
    array = jnp.asarray(value)
    if array.dtype != jnp.bool_ and not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError(f"Fragment field {name} must be integer or boolean.")
    if array.shape != shape:
        raise ValueError(
            f"Fragment field {name} must have shape {shape} for one fragment; got "
            f"{array.shape}. Stream stacked fragments one at a time."
        )
    return array.astype(jnp.int32)


def _routing_operands(
    fragment: StreamedFragment, extent: MACEFragmentExtent, /
) -> tuple[Array, ...]:
    """Kernel-order routing of one fragment (see `_routing_shapes`)."""
    fields = (
        ("lane_slots", fragment.lane_slots),
        ("lane_local_sources", fragment.lane_local_sources),
        ("lane_use", fragment.lane_use),
        ("receiver_lane_offsets", fragment.receiver_lane_offsets),
        ("receiver_lane_counts", fragment.receiver_lane_counts),
        ("source_lanes", fragment.source_lanes),
        ("source_lane_offsets", fragment.source_lane_offsets),
        ("source_lane_counts", fragment.source_lane_counts),
    )
    return tuple(
        _index_operand(value, name, shape)
        for (name, value), shape in zip(
            fields,
            _routing_shapes(extent.receivers, extent.sources, extent.edges),
            strict=True,
        )
    )


class _Routing(NamedTuple):
    """Routing arrays of one fragment (or stacked fragments) to check."""

    receiver_ids: Array
    source_ids: Array
    source_valid: Array
    lane_slots: Array
    lane_local_sources: Array
    lane_use: Array
    receiver_lane_offsets: Array
    receiver_lane_counts: Array
    source_lanes: Array
    source_lane_offsets: Array
    source_lane_counts: Array


def _routing_violation(routing: _Routing, /) -> Array:
    """Whether one fragment's routing breaks the kernel's ownership contract."""
    edges = routing.lane_slots.shape[0]
    receivers = routing.receiver_ids.shape[0]
    sources = routing.source_ids.shape[0]
    lanes = jnp.arange(edges, dtype=jnp.int32)
    slots, local = routing.lane_slots, routing.lane_local_sources
    offsets, counts = routing.receiver_lane_offsets, routing.receiver_lane_counts
    order = routing.source_lanes
    source_offsets, source_counts = (
        routing.source_lane_offsets,
        routing.source_lane_counts,
    )
    total = jnp.sum(counts, dtype=jnp.int32)
    structural = lanes < total
    # Target-major lanes: receiver slots with lanes own consecutive segments
    # (an empty slot's offset is irrelevant).
    bad = jnp.any(counts < 0) | (total > edges)
    bad = bad | jnp.any(
        (counts > 0) & (offsets != jnp.cumsum(counts, dtype=jnp.int32) - counts)
    )
    owner = jnp.clip(slots, 0, receivers - 1)
    bad = bad | jnp.any(
        structural
        & (
            (slots < 0)
            | (slots >= receivers)
            | (lanes < offsets[owner])
            | (lanes >= offsets[owner] + counts[owner])
        )
    )
    bad = bad | jnp.any(routing.lane_use & ~structural)
    # Source-major positions: sources with lanes own consecutive segments of a
    # permutation of exactly the structural lanes.
    source_ends = jnp.cumsum(source_counts, dtype=jnp.int32)
    bad = bad | jnp.any(source_counts < 0) | (source_ends[-1] != total)
    bad = bad | jnp.any(
        (source_counts > 0) & (source_offsets != source_ends - source_counts)
    )
    bad = bad | jnp.any((source_counts > 0) & ~routing.source_valid)
    group = jnp.clip(
        jnp.searchsorted(source_ends, lanes, side="right").astype(jnp.int32),
        0,
        sources - 1,
    )
    safe_order = jnp.clip(order, 0, edges - 1)
    bad = bad | jnp.any(
        structural & ((order < 0) | (order >= total) | (local[safe_order] != group))
    )
    visits = (
        jnp.zeros((edges,), jnp.int32).at[safe_order].add(structural.astype(jnp.int32))
    )
    return bad | jnp.any(structural & (visits != 1))


def require_fragment_routing(fragment: StreamedFragment, /) -> StreamedFragment:
    """Refuse fragments whose routing breaks the kernel's ownership contract.

    Checks one fragment, or every fragment of a schedule stacked along leading
    axes: receiver slots own consecutive target-major lane segments, every lane
    in use lies in a receiver segment, fragment sources own consecutive
    segments of a source-major permutation of exactly those lanes, and source
    groups with lanes are valid. Run it once per prepared topology epoch; the
    accelerated fragment entry trusts its prepared routing.
    """
    if not isinstance(fragment, StreamedFragment):
        raise TypeError("fragment must be a StreamedFragment.")
    routing = _Routing(
        *(
            jnp.asarray(getattr(fragment, name), dtype=jnp.bool_)
            if name in ("source_valid", "lane_use")
            else jnp.asarray(getattr(fragment, name), dtype=jnp.int32)
            for name in _Routing._fields
        )
    )
    check: Callable[[_Routing], Array] = _routing_violation
    for _ in range(routing.lane_slots.ndim - 1):
        check = jax.vmap(check)
    checked = eqx.error_if(
        jnp.asarray(fragment.lane_slots),
        jnp.any(check(routing)),
        "A MACE kernel fragment routing breaks the receiver/source lane ownership "
        "contract.",
    )
    return eqx.tree_at(lambda value: value.lane_slots, fragment, checked)


@final
class MACEKernelEvidence(StrictModule):
    """Selected accelerated route, its identities and declared fragment resources."""

    admission: AtomisticKernelAdmission
    signature: ExecutableSignature
    resources: MACEFragmentResources
    spec_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    workspace_elements: tuple[tuple[str, int], ...] = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    padded_channels: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        admission: AtomisticKernelAdmission,
        signature: ExecutableSignature,
        resources: MACEFragmentResources,
        spec: MACEEdgeCouplingSpec,
        plan: MACEKernelPlan,
        topology_id: str,
    ) -> None:
        self.admission = admission
        self.signature = signature
        self.resources = resources
        self.spec_id = spec.spec_id
        self.topology_id = topology_id
        self.workspace_elements = tuple(sorted(spec.workspace_elements(plan).items()))
        self.workspace_bytes = spec.workspace_bytes(plan)
        self.padded_channels = plan.channel_capacity(spec) - spec.channels


@final
class MACEAcceleratedCoupling(StrictModule):
    """Admitted accelerated MACE fragment coupling on one resolved kernel target.

    Construction resolves the target and refuses unavailable devices.
    `fragment` admits the static structure and extents of each call, then
    executes the seeded receiver kernel through one JAX primitive whose JVP,
    transpose and batching rules execute the source, radial, harmonic and
    coefficient kernels over the same fragment.
    """

    plan: MACEKernelPlan
    target: AtomisticKernelTarget

    def __init__(self, plan: MACEKernelPlan, /) -> None:
        if not isinstance(plan, MACEKernelPlan):
            raise TypeError("plan must be a MACEKernelPlan.")
        self.plan = plan
        self.target = AtomisticKernelTarget(plan.target)

    def resources(
        self, spec: MACEEdgeCouplingSpec, extent: MACEFragmentExtent, /
    ) -> MACEFragmentResources:
        """Declared logical bytes of this coupling over one fragment of ``extent``."""
        if not isinstance(spec, MACEEdgeCouplingSpec):
            raise TypeError("spec must be a MACEEdgeCouplingSpec.")
        if not isinstance(extent, MACEFragmentExtent):
            raise TypeError("extent must be a MACEFragmentExtent.")
        return MACEFragmentResources(spec, self.plan, extent)

    def admit(
        self, spec: MACEEdgeCouplingSpec, extent: MACEFragmentExtent, /
    ) -> AtomisticKernelAdmission:
        """Admit ``spec`` over fragments of ``extent`` at this plan's tiles and budget."""
        plan = self.plan
        resources = self.resources(spec, extent)
        request = AtomisticKernelRequest(
            precision=plan.precision,
            accumulation=plan.accumulation,
            channels=spec.channels,
            channel_tile=plan.channel_tile,
            receiver_tile=plan.receiver_tile,
            edge_tile=plan.edge_tile,
            reduction_programs=plan.reduction_programs,
            receiver_extent=extent.receivers,
            source_extent=extent.sources,
            edge_extent=extent.edges,
            maximum_degree=spec.maximum_degree,
            path_count=len(spec.paths),
            coefficient_count=spec.coefficient_count,
            workspace_bytes=spec.workspace_bytes(plan),
            fragment_bytes=resources.peak_bind_bytes,
            fragment_budget_bytes=plan.fragment_budget_bytes,
            maximum_derivative_order=plan.maximum_derivative_order,
        )
        return AtomisticKernelAdmission(request, self.target)

    def executable_signature(
        self, spec: MACEEdgeCouplingSpec, extent: MACEFragmentExtent, topology_id: str, /
    ) -> ExecutableSignature:
        """Static executable identity; numeric operands never enter it.

        ``topology_id`` is the caller's prepared fragment-schedule binding.
        """
        plan = self.plan
        receivers, sources, edges = extent.receivers, extent.sources, extent.edges
        return ExecutableSignature(
            shapes=(
                ("source_rows", (sources, spec.source_width)),
                ("harmonics", (edges, spec.harmonic_width)),
                ("radial", (edges, spec.radial_width)),
                ("seed", (2, receivers, spec.message_width)),
                ("coefficient", (spec.coefficient_count,)),
            ),
            dtypes=(("float", plan.precision), ("index", "int32")),
            topology_ids=(("fragments", topology_id),),
            capacities=(
                ("receivers", receivers),
                ("sources", sources),
                ("edges", edges),
                ("receiver_tile", plan.receiver_tile),
                ("edge_tile", plan.edge_tile),
                ("channel_tile", plan.channel_tile),
                ("channel_capacity", plan.channel_capacity(spec)),
                ("reduction_programs", plan.reduction_programs),
                ("fragment_budget_bytes", plan.fragment_budget_bytes),
                ("maximum_derivative_order", plan.maximum_derivative_order),
            ),
            algorithm_facts=(
                ("kernel_structure", spec.spec_id),
                ("accumulation", plan.accumulation),
            ),
            backend_facts=(
                ("target", self.target.target_id),
                ("lowering", self.target.lowering),
                ("lowering_semantics", "mosaic_gpu_lane"),
            ),
        )

    def evidence(
        self, spec: MACEEdgeCouplingSpec, extent: MACEFragmentExtent, topology_id: str, /
    ) -> MACEKernelEvidence:
        """Admission, executable identity and declared resources of one binding."""
        return MACEKernelEvidence(
            admission=self.admit(spec, extent),
            signature=self.executable_signature(spec, extent, topology_id),
            resources=self.resources(spec, extent),
            spec=spec,
            plan=self.plan,
            topology_id=topology_id,
        )

    def resource_evidence(
        self, compiled: jax.stages.Compiled, /
    ) -> ExecutionResourceEvidence:
        """XLA compiled-memory evidence of an executable that runs these kernels.

        Covers XLA-assigned argument/output/temporary/code buffers of the
        transformed route actually compiled by the caller. Kernel registers and
        shared memory are on-chip and outside this analysis.
        """
        estimate = compiled_memory_estimate(compiled)
        if estimate is None:
            raise ValueError(
                "The compiled executable exposes no XLA memory analysis; no compiler "
                "memory evidence can be recorded for the accelerated coupling."
            )
        return ExecutionResourceEvidence(
            per_device_peak_bytes=estimate.device_bytes
            if estimate.peak_bytes is None
            else estimate.peak_bytes,
            dtypes=(self.plan.precision,),
            backends=(f"pallas-atomistic:{self.target.lowering}",),
            memory_basis="compiler_analysis",
        )

    def fragment(
        self,
        tensor_product: O3TensorProduct,
        fragment: StreamedFragment,
        source_rows: ArrayLike,
        harmonics: ArrayLike,
        radial: ArrayLike,
        seed: KeyGroupAccumulation,
        /,
    ) -> KeyGroupAccumulation:
        """Stream one fragment's lanes into its receiver slots' seeded accumulators.

        ``source_rows`` ``[S, source width]`` are the packed rows of the
        fragment's sources in ``fragment.source_ids`` order; ``harmonics``
        ``[F, harmonic width]`` and ``radial`` ``[F, radial width]`` are the
        lanes' local operands; ``seed`` holds packed ``[R, message width]``
        high/correction accumulators of the receiver slots. Lanes not in
        ``fragment.lane_use`` contribute nothing. Returns the updated
        accumulators: ``deterministic`` adds every lane to ``high`` in stable
        order and returns ``correction`` unchanged; ``compensated`` applies
        one error-free ``two_sum`` per lane and carries its residual.
        """
        if not isinstance(fragment, StreamedFragment):
            raise TypeError("fragment must be a StreamedFragment.")
        if not isinstance(seed, KeyGroupAccumulation):
            raise TypeError("seed must be a KeyGroupAccumulation.")
        plan = self.plan
        if fragment.accumulation != plan.accumulation:
            raise ValueError(
                f"The fragment schedule accumulates {fragment.accumulation!r} but the "
                f"kernel plan declares {plan.accumulation!r}; one stream has one order."
            )
        spec = MACEEdgeCouplingSpec(tensor_product)
        extent = fragment_extent(fragment)
        self.admit(spec, extent)
        receivers, sources, edges = extent.receivers, extent.sources, extent.edges
        capacity = plan.channel_capacity(spec)
        coefficients = _float_operand(
            spec.coefficients(tensor_product),
            "tensor-product coefficients",
            (spec.coefficient_count,),
            plan,
        )
        radial_ = _float_operand(radial, "radial", (edges, spec.radial_width), plan)
        radial_ = jnp.pad(
            radial_.reshape(edges, len(spec.paths), spec.channels),
            ((0, 0), (0, 0), (0, capacity - spec.channels)),
        )
        source = spec.to_channel_minor(
            spec.source_blocks,
            _float_operand(
                source_rows, "source_rows", (sources, spec.source_width), plan
            ),
            capacity,
        )
        message = (receivers, spec.message_width)
        initial = jnp.stack(
            (
                spec.to_channel_minor(
                    spec.message_blocks,
                    _float_operand(seed.high, "seed.high", message, plan),
                    capacity,
                ),
                spec.to_channel_minor(
                    spec.message_blocks,
                    _float_operand(seed.correction, "seed.correction", message, plan),
                    capacity,
                ),
            )
        )
        result = _coupling_p.bind(
            *_routing_operands(fragment, extent),
            coefficients,
            radial_,
            source,
            _float_operand(harmonics, "harmonics", (edges, spec.harmonic_width), plan),
            initial,
            role="receiver",
            spec=spec,
            plan=plan,
            receiver_count=receivers,
            source_count=sources,
            edge_count=edges,
            seeded=True,
            derivative_order=0,
        )
        return KeyGroupAccumulation(
            spec.from_channel_minor(spec.message_blocks, result[0]),
            spec.from_channel_minor(spec.message_blocks, result[1]),
        )


__all__ = [
    "fragment_extent",
    "MACEAcceleratedCoupling",
    "MACECouplingRow",
    "MACEEdgeCouplingSpec",
    "MACEFragmentExtent",
    "MACEFragmentResources",
    "MACEKernelEvidence",
    "MACEKernelPlan",
    "MACELayoutBlock",
    "require_fragment_routing",
]
