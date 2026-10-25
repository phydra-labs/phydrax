#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-faithful MACE interaction, product, and layer kernels.

One layer is split at the only nonlocal boundary of the architecture. Each
directed edge evaluates the radial weights, real harmonics and ``uvu`` coupling
of its sender's ``linear_up`` features (plus a scalar density and, in the first
layer, the ZBL share). Each receiver then owns its complete epilogue over the
summed messages: message linear, neighbor or density normalization, species
self-connection or residual, symmetric product, product linear, and readout.
The epilogue runs once per receiver after all of its edges, so only inter-layer
node states and readout energies leave a receiver tile.

Three interaction families are admitted exactly as the pinned source defines
them:

* the standard real-agnostic kinds (plain, residual, density, density-residual)
  with the species-conditioned ``skip_tp`` self-connection;
* the attention-residual kind, whose edge weights read the radial features and
  a 64-channel scalar projection of both endpoint states through a fixed
  ``256, 256, 256`` unit-variance network, with a plain residual linear;
* the residual-nonlinear kind, whose biased layer-normalized edge networks read
  the radial features and species source/target embeddings, normalizes by
  ``density * beta + alpha``, adds a receiver-local linear residual, applies the
  normalized gated nonlinearity and a final linear, with a plain residual linear.

Endpoint data an edge needs beyond its sender's coupling input (the attention
projection) travels as explicit source/receiver context rows, so the streamed
and accelerated fragment routes evaluate the same equations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import assert_never, final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._model import register_artifact_value
from ..._strict import StrictModule
from ..._validation import positive_finite_float
from ...ein import contract
from ...sparse._execution import KeyGroupAccumulation
from ...sparse._streamed import StreamedFragment
from ...special import RealCartesianHarmonics
from ...typing import parse, PRNGKey
from ..operator.layers import O3IrrepLinear, O3TensorProduct
from ..operator.representations import O3IrrepBlock, O3IrrepLayout, O3Parity
from ._mace_kernels import MACEAcceleratedCoupling, MACEEdgeCouplingSpec
from ._mace_readout import (
    MACEInvariantReadoutProduct,
    MACEPairRepulsion,
    MACEReadout,
)
from ._radial import LayerNormRadialMLP, RadialEmbedding, RadialMLP
from ._symmetric_contraction import SymmetricContraction


MACEInteractionKind: TypeAlias = Literal[
    "real-agnostic",
    "real-agnostic-residual",
    "real-agnostic-density",
    "real-agnostic-density-residual",
    "real-agnostic-attention-residual",
    "real-agnostic-residual-nonlinear",
]
MACEInteractionFamily: TypeAlias = Literal["standard", "attention", "nonlinear"]
MACECutoffPlacement: TypeAlias = Literal["embedding", "weights"]
MACEParityConvention: TypeAlias = Literal["o3", "so3"]

# Fixed by the source attention-residual block, independent of radial_MLP.
ATTENTION_PROJECTION_WIDTH = 64
ATTENTION_RADIAL_WIDTHS = (256, 256, 256)
# Fixed hidden width of the source nonlinear block's density network.
NONLINEAR_DENSITY_WIDTH = 64
# Source initial values of the nonlinear block's density normalization.
NONLINEAR_DENSITY_ALPHA = 20.0
NONLINEAR_DENSITY_BETA = 0.0


def interaction_traits(kind: MACEInteractionKind, /) -> tuple[bool, bool]:
    """``(residual, density_normalized)`` of one admitted interaction kind."""
    match kind:
        case "real-agnostic":
            return False, False
        case "real-agnostic-residual":
            return True, False
        case "real-agnostic-density":
            return False, True
        case "real-agnostic-density-residual":
            return True, True
        case "real-agnostic-attention-residual":
            return True, False
        case "real-agnostic-residual-nonlinear":
            return True, True
        case unreachable:
            assert_never(unreachable)


def interaction_family(kind: MACEInteractionKind, /) -> MACEInteractionFamily:
    match kind:
        case (
            "real-agnostic"
            | "real-agnostic-residual"
            | "real-agnostic-density"
            | "real-agnostic-density-residual"
        ):
            return "standard"
        case "real-agnostic-attention-residual":
            return "attention"
        case "real-agnostic-residual-nonlinear":
            return "nonlinear"
        case unreachable:
            assert_never(unreachable)


def irrep_block_name(degree: int, parity: O3Parity, /) -> str:
    """Canonical name of the hidden/target block of one ``(l, parity)``."""
    return f"l{degree}{'e' if parity == 1 else 'o'}"


def natural_parity(degree: int, /) -> O3Parity:
    return 1 if degree % 2 == 0 else -1


def convention_parity(degree: int, convention: MACEParityConvention, /) -> O3Parity:
    """``(-1)^l`` under O(3); every block is even under the source SO(3) option."""
    match convention:
        case "o3":
            return natural_parity(degree)
        case "so3":
            return 1
        case unreachable:
            assert_never(unreachable)


def mace_irrep_layout(
    multiplicity: int,
    maximum_degree: int,
    /,
    *,
    parity: MACEParityConvention = "o3",
) -> O3IrrepLayout:
    """``multiplicity x (l, p_l)`` for ``l = 0..maximum_degree`` in degree order."""
    return O3IrrepLayout(
        tuple(
            O3IrrepBlock(
                irrep_block_name(degree, convention_parity(degree, parity)),
                degree,
                convention_parity(degree, parity),
                multiplicity=multiplicity,
            )
            for degree in range(maximum_degree + 1)
        )
    )


def harmonic_layout(
    maximum_degree: int, /, *, parity: MACEParityConvention = "o3"
) -> O3IrrepLayout:
    """Real spherical-harmonic layout ``1 x (l, p_l)``, degree-major."""
    return O3IrrepLayout(
        tuple(
            O3IrrepBlock(f"y{degree}", degree, convention_parity(degree, parity))
            for degree in range(maximum_degree + 1)
        )
    )


def gate_layout(target: O3IrrepLayout, /) -> O3IrrepLayout:
    """Input layout of the source gate over ``target``: scalars and gates, then gated.

    The source sorts ``scalars + gates + gated`` and merges the leading scalar
    groups, giving ``C (1 + G) x 0e`` (``G`` gated blocks) followed by the gated
    blocks of ``target`` in order.
    """
    scalar = target.blocks[0]
    if (scalar.degree, scalar.parity) != (0, 1):
        raise ValueError("A gated layout begins with its even scalar block.")
    gated = target.blocks[1:]
    if any(block.degree == 0 for block in gated):
        raise ValueError("The gate admits exactly one leading scalar block.")
    gates = sum(block.multiplicity for block in gated)
    return O3IrrepLayout(
        (
            O3IrrepBlock("g0e", 0, 1, multiplicity=scalar.multiplicity + gates),
            *gated,
        )
    )


def to_channels(layout: O3IrrepLayout, values: Array, /) -> Array:
    """Reshape packed ``C x irreps`` values to per-channel ``[..., C, sum(2l+1)]``."""
    return jnp.concatenate(layout.split(values), axis=-1)


def from_channels(layout: O3IrrepLayout, values: Array, /) -> Array:
    """Inverse of `to_channels` for one equal-multiplicity layout."""
    blocks = []
    start = 0
    for block in layout.blocks:
        blocks.append(values[..., start : start + block.dimension])
        start += block.dimension
    return layout.join(blocks)


def convolution_paths(
    input_layout: O3IrrepLayout,
    harmonics: O3IrrepLayout,
    target: O3IrrepLayout,
    /,
) -> tuple[O3IrrepLayout, tuple[tuple[str, str, str], ...]]:
    """Source ``uvu`` coupling paths and their message layout.

    Paths enumerate input blocks, then harmonic blocks, then output degrees
    ``|l1 - l2| .. l1 + l2`` that occur in ``target``. Message blocks are then
    stably sorted by ``(l, parity)`` with odd before even, the source message
    order that fixes both the radial-weight columns and the message linear rows.
    """
    allowed = {(block.degree, block.parity) for block in target.blocks}
    declared: list[tuple[int, int, str, str]] = []
    for left in input_layout.blocks:
        for right in harmonics.blocks:
            parity = left.parity * right.parity
            for degree in range(
                abs(left.degree - right.degree), left.degree + right.degree + 1
            ):
                if (degree, parity) in allowed:
                    declared.append((degree, parity, left.name, right.name))
    ordered = sorted(
        enumerate(declared), key=lambda item: (item[1][0], item[1][1], item[0])
    )
    multiplicity = input_layout.blocks[0].multiplicity
    blocks: list[O3IrrepBlock] = []
    paths: list[tuple[str, str, str]] = []
    for position, (_, (degree, parity, left, right)) in enumerate(ordered):
        name = f"m{position}-{irrep_block_name(degree, 1 if parity == 1 else -1)}"
        blocks.append(
            O3IrrepBlock(
                name, degree, 1 if parity == 1 else -1, multiplicity=multiplicity
            )
        )
        paths.append((left, right, name))
    return O3IrrepLayout(blocks), tuple(paths)


def linear_paths(
    in_layout: O3IrrepLayout, out_layout: O3IrrepLayout, /
) -> tuple[tuple[int, int], ...]:
    """Input-major ``(input block, output block)`` pairs of identical ``(l, p)``."""
    return tuple(
        (left, right)
        for left, source in enumerate(in_layout.blocks)
        for right, target in enumerate(out_layout.blocks)
        if (source.degree, source.parity) == (target.degree, target.parity)
    )


# The species self-connection enumerates the same input-major pairs.
self_connection_paths = linear_paths


@final
class MACEPlainLinear(StrictModule):
    """Species-agnostic equivariant linear whose unmatched outputs are exactly zero.

    The source e3nn ``Linear`` between arbitrary layouts: output block ``k``
    receives ``sum_paths W x / sqrt(fan_in_k)`` over input blocks of identical
    ``(l, parity)``, and output blocks without any matching input stay zero (for
    example the higher blocks of a first-layer scalar input). ``linear`` executes
    the matched output blocks; its weights are the PARAMETER leaves.
    """

    in_layout: O3IrrepLayout
    out_layout: O3IrrepLayout
    linear: O3IrrepLinear
    matched: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        in_layout: O3IrrepLayout,
        out_layout: O3IrrepLayout,
        /,
        *,
        weights: Sequence[ArrayLike] | None = None,
        dtype: DTypeLike,
        key: PRNGKey | None = None,
    ) -> None:
        if not isinstance(in_layout, O3IrrepLayout) or not isinstance(
            out_layout, O3IrrepLayout
        ):
            raise TypeError("Plain linear layouts must be O3IrrepLayout values.")
        pairs = linear_paths(in_layout, out_layout)
        matched = tuple(sorted({right for _, right in pairs}))
        if not matched:
            raise ValueError("A plain linear needs at least one matching block.")
        sub = O3IrrepLayout(tuple(out_layout.blocks[index] for index in matched))
        if weights is None:
            if key is None:
                raise ValueError("Initialize a plain linear with weights or a key.")
            linear = O3IrrepLinear(in_layout, sub, dtype=dtype, key=key)
        else:
            linear = O3IrrepLinear(in_layout, sub, weights=weights, dtype=dtype)
        self.in_layout = in_layout
        self.out_layout = out_layout
        self.linear = linear
        self.matched = matched

    @property
    def weights(self) -> tuple[Array, ...]:
        return self.linear.weights

    def __call__(self, values: Array, /) -> Array:
        mixed = self.linear.out_layout.split(self.linear(values))
        leading = values.shape[:-1]
        outputs = []
        position = 0
        for index, block in enumerate(self.out_layout.blocks):
            if position < len(self.matched) and self.matched[position] == index:
                outputs.append(mixed[position])
                position += 1
            else:
                outputs.append(
                    jnp.zeros(
                        leading + (block.multiplicity, block.dimension),
                        dtype=values.dtype,
                    )
                )
        return self.out_layout.join(outputs)


@final
class MACESelfConnection(StrictModule):
    """Species-conditioned equivariant linear map (the source ``skip_tp``).

    Each path couples one input block to the output block of identical
    ``(l, parity)`` through ``W[u, species, w]``; an output block receives
    ``sum_paths W x / sqrt(sum_paths C_in * S)``. Output blocks without a
    matching input are exactly zero, as in the source fully connected product.
    """

    in_layout: O3IrrepLayout
    out_layout: O3IrrepLayout
    weights: tuple[Array, ...]
    paths: tuple[tuple[int, int], ...] = eqx.field(static=True)
    species_count: int = eqx.field(static=True)

    def __init__(
        self,
        in_layout: O3IrrepLayout,
        out_layout: O3IrrepLayout,
        weights: Sequence[ArrayLike],
        /,
        *,
        species_count: int,
    ) -> None:
        if not isinstance(in_layout, O3IrrepLayout) or not isinstance(
            out_layout, O3IrrepLayout
        ):
            raise TypeError("Self-connection layouts must be O3IrrepLayout values.")
        paths = self_connection_paths(in_layout, out_layout)
        arrays = tuple(jnp.asarray(weight) for weight in weights)
        expected = tuple(
            (
                in_layout.blocks[left].multiplicity,
                species_count,
                out_layout.blocks[right].multiplicity,
            )
            for left, right in paths
        )
        if tuple(array.shape for array in arrays) != expected:
            raise ValueError(f"Self-connection weights must have shapes {expected}.")
        dtypes = {array.dtype for array in arrays}
        if len(dtypes) != 1 or next(iter(dtypes)) not in (
            jnp.dtype(jnp.float32),
            jnp.dtype(jnp.float64),
        ):
            raise TypeError("Self-connection weights must share float32 or float64.")
        self.in_layout = in_layout
        self.out_layout = out_layout
        self.weights = arrays
        self.paths = paths
        self.species_count = species_count

    @classmethod
    def initialize(
        cls,
        in_layout: O3IrrepLayout,
        out_layout: O3IrrepLayout,
        /,
        *,
        species_count: int,
        key: PRNGKey,
        dtype: DTypeLike,
    ) -> MACESelfConnection:
        """Draw unit-normal weights, the original fully connected initialization."""
        paths = self_connection_paths(in_layout, out_layout)
        keys = jr.split(key, max(len(paths), 1))
        weights = tuple(
            jr.normal(
                path_key,
                (
                    in_layout.blocks[left].multiplicity,
                    species_count,
                    out_layout.blocks[right].multiplicity,
                ),
                dtype=dtype,
            )
            for path_key, (left, right) in zip(keys, paths)
        )
        return cls(in_layout, out_layout, weights, species_count=species_count)

    def __call__(self, values: Array, species: Array, /) -> Array:
        """``values[..., P_in]`` with integer ``species[...]`` -> ``[..., P_out]``."""
        inputs = self.in_layout.split(values)
        leading = values.shape[:-1]
        fan_in = [0] * len(self.out_layout.blocks)
        for left, right in self.paths:
            fan_in[right] += self.in_layout.blocks[left].multiplicity * self.species_count
        dtype = values.dtype
        outputs = [
            jnp.zeros(leading + (block.multiplicity, block.dimension), dtype=dtype)
            for block in self.out_layout.blocks
        ]
        for (left, right), weight in zip(self.paths, self.weights, strict=True):
            selected = jnp.moveaxis(jnp.take(weight, species, axis=1), 0, -2)
            scale = jnp.sqrt(jnp.asarray(fan_in[right], dtype=dtype))
            outputs[right] = (
                outputs[right]
                + contract("...ui,...uw->...wi", inputs[left], selected.astype(dtype))
                / scale
            )
        return self.out_layout.join(outputs)


@final
class MACEGate(StrictModule):
    """Source normalized gated equivariant nonlinearity over one target layout.

    Input ``C (1 + G) x 0e + gated`` (see `gate_layout`): the first ``C``
    scalars become ``scalar_scale * silu(x)``; the next ``G`` groups of ``C``
    gates become ``gate_scale * sigmoid(x)`` and multiply, channel by channel,
    the components of the corresponding gated block. The output is the target
    layout. The two scales are the declared second-moment normalization
    constants of the activations.
    """

    target_layout: O3IrrepLayout
    input_layout: O3IrrepLayout
    scalar_scale: float = eqx.field(static=True)
    gate_scale: float = eqx.field(static=True)

    def __init__(
        self,
        target_layout: O3IrrepLayout,
        /,
        *,
        scalar_scale: float,
        gate_scale: float,
    ) -> None:
        if not isinstance(target_layout, O3IrrepLayout):
            raise TypeError("target_layout must be an O3IrrepLayout.")
        self.target_layout = target_layout
        self.input_layout = gate_layout(target_layout)
        self.scalar_scale = positive_finite_float(scalar_scale, "scalar_scale")
        self.gate_scale = positive_finite_float(gate_scale, "gate_scale")

    def __call__(self, values: Array, /) -> Array:
        blocks = self.input_layout.split(values)
        leading = blocks[0]
        channels = self.target_layout.blocks[0].multiplicity
        dtype = values.dtype
        scalars = jnp.asarray(self.scalar_scale, dtype=dtype) * jax.nn.silu(
            leading[..., :channels, :]
        )
        gates = jnp.asarray(self.gate_scale, dtype=dtype) * jax.nn.sigmoid(
            leading[..., channels:, 0]
        )
        outputs = [scalars]
        offset = 0
        for block, gated in zip(self.target_layout.blocks[1:], blocks[1:], strict=True):
            gate = gates[..., offset : offset + block.multiplicity]
            outputs.append(gated * gate[..., None])
            offset += block.multiplicity
        return self.target_layout.join(outputs)


class MACESourceRows(NamedTuple):
    """Node-local source payload of one layer; the halo-exchanged state.

    ``context`` carries sender data the edge weights read beyond the coupling
    input (the attention projection), or ``None``.
    """

    features: Array
    species: Array
    context: Array | None = None


class MACEReceiverRows(NamedTuple):
    """Receiver-local payload of one layer epilogue.

    ``context`` carries receiver data the edge weights or the epilogue read
    (the attention projection, or the nonlinear block's lifted state), or
    ``None``.
    """

    features: Array
    species: Array
    mask: Array
    context: Array | None = None


@final
class MACEEdgeGeometry(StrictModule):
    """Shared per-edge radial features and real harmonics of one model."""

    radial: RadialEmbedding
    harmonics: RealCartesianHarmonics
    cutoff_placement: MACECutoffPlacement = eqx.field(static=True)

    def __init__(
        self,
        radial: RadialEmbedding,
        harmonics: RealCartesianHarmonics,
        /,
        *,
        cutoff_placement: MACECutoffPlacement,
    ) -> None:
        if not isinstance(radial, RadialEmbedding):
            raise TypeError("radial must be a RadialEmbedding.")
        if not isinstance(harmonics, RealCartesianHarmonics):
            raise TypeError("harmonics must be RealCartesianHarmonics.")
        if (
            harmonics.normalization != "fully_normalized"
            or harmonics.argument != "direction"
        ):
            raise ValueError(
                "MACE edge harmonics are fully normalized real direction harmonics."
            )
        self.radial = radial
        self.harmonics = harmonics
        self.cutoff_placement = cutoff_placement

    def radial_features(
        self, distance: Array, sender: Array, receiver: Array, /
    ) -> tuple[Array, Array | None]:
        """Radial-network input and the cutoff applied to its outputs, if separate."""
        match self.cutoff_placement:
            case "embedding":
                return self.radial.unmasked(distance, sender, receiver), None
            case "weights":
                transform = self.radial.transform
                transformed = (
                    distance
                    if transform is None
                    else transform(distance, sender, receiver)
                )
                return self.radial.basis(transformed), self.radial.cutoff(distance)
            case unreachable:
                assert_never(unreachable)


class MACEEdgeWeights(NamedTuple):
    """Cutoff-applied coupling weights and (density kinds) edge density of edges."""

    weights: Array
    density: Array | None


def _with_cutoff(value: Array, cutoff: Array | None, /) -> Array:
    return value if cutoff is None else value * cutoff[..., None]


@final
class MACEInteraction(StrictModule):
    """One standard source interaction block in its four admitted variants."""

    linear_up: O3IrrepLinear
    tensor_product: O3TensorProduct
    radial: RadialMLP
    linear_down: O3IrrepLinear
    self_connection: MACESelfConnection
    density: RadialMLP | None
    kind: MACEInteractionKind = eqx.field(static=True)
    average_neighbor_count: float = eqx.field(static=True)

    def __init__(
        self,
        linear_up: O3IrrepLinear,
        tensor_product: O3TensorProduct,
        radial: RadialMLP,
        linear_down: O3IrrepLinear,
        self_connection: MACESelfConnection,
        /,
        *,
        kind: MACEInteractionKind,
        average_neighbor_count: float,
        density: RadialMLP | None = None,
    ) -> None:
        kind_ = parse(kind, MACEInteractionKind, "kind")
        if interaction_family(kind_) != "standard":
            raise ValueError(f"Interaction kind {kind_!r} is not a standard kind.")
        _require_coupling(linear_up, tensor_product, linear_down)
        if radial.output_width != tensor_product.plan.parameter_count:
            raise ValueError("The radial network must emit every coupling path weight.")
        if density is not None and (
            density.widths != (radial.input_width, 1)
            or density.postprocess != "tanh-square"
        ):
            raise ValueError(
                "Density normalization is a tanh-squared linear edge scalar."
            )
        residual, density_normalized = interaction_traits(kind_)
        target = linear_down.out_layout
        if residual:
            if self_connection.in_layout.layout_id != linear_up.in_layout.layout_id:
                raise ValueError("A residual self-connection reads the layer input.")
        elif (
            self_connection.in_layout.layout_id != target.layout_id
            or self_connection.out_layout.layout_id != target.layout_id
        ):
            raise ValueError("A non-residual self-connection maps the target layout.")
        if density_normalized != (density is not None):
            raise ValueError(f"Interaction kind {kind_!r} density network mismatch.")
        self.linear_up = linear_up
        self.tensor_product = tensor_product
        self.radial = radial
        self.linear_down = linear_down
        self.self_connection = self_connection
        self.density = density
        self.kind = kind_
        self.average_neighbor_count = positive_finite_float(
            average_neighbor_count, "average_neighbor_count"
        )

    @property
    def residual(self) -> bool:
        return interaction_traits(self.kind)[0]

    @property
    def input_layout(self) -> O3IrrepLayout:
        return self.linear_up.in_layout

    @property
    def message_layout(self) -> O3IrrepLayout:
        return self.linear_down.in_layout

    @property
    def target_layout(self) -> O3IrrepLayout:
        return self.linear_down.out_layout

    @property
    def residual_layout(self) -> O3IrrepLayout | None:
        return self.self_connection.out_layout if self.residual else None

    def source_rows(
        self, features: Array, species: Array, atom_mask: Array, /
    ) -> MACESourceRows:
        lifted = self.linear_up(features)
        zero = jnp.zeros((), dtype=lifted.dtype)
        return MACESourceRows(jnp.where(atom_mask[:, None], lifted, zero), species)

    def receiver_context(self, features: Array, /) -> Array | None:
        return None

    def edge_weights(
        self,
        radial_features: Array,
        cutoff: Array | None,
        source_context: Array | None,
        receiver_context: Array | None,
        sender: Array,
        receiver: Array,
        /,
    ) -> MACEEdgeWeights:
        weights = _with_cutoff(self.radial(radial_features), cutoff)
        density = None
        if self.density is not None:
            density = self.density(radial_features)[..., 0]
            density = density if cutoff is None else density * cutoff
        return MACEEdgeWeights(weights, density)

    def receiver_message(
        self,
        aggregate: dict[str, Array],
        species: Array,
        context: Array | None,
        /,
    ) -> Array:
        """Normalized, self-connected receiver message in the target layout."""
        message = self.linear_down(aggregate["coupling"])
        if self.density is None:
            message = message / self.average_neighbor_count
        else:
            message = message / (aggregate["density"][..., None] + 1.0)
        if self.residual:
            return message
        return self.self_connection(message, species)

    def residual_state(self, features: Array, species: Array, /) -> Array | None:
        """Species-conditioned residual of the layer input, for residual kinds."""
        if not self.residual:
            return None
        return self.self_connection(features, species)


@final
class MACEAttentionInteraction(StrictModule):
    """The source attention-residual interaction block.

    Edge weights are ``FC([radial, down(x_s), down(x_r)])`` with the
    unit-variance normalized-SiLU network of widths ``radial + 128, 256, 256,
    256, paths`` and ``down`` a plain linear map of the layer input onto
    ``64 x 0e``; the summed message is ``linear(m) / avg_num_neighbors`` and the
    residual is a plain species-agnostic linear map of the layer input.
    """

    linear_up: O3IrrepLinear
    attention_down: O3IrrepLinear
    tensor_product: O3TensorProduct
    radial: RadialMLP
    linear_down: O3IrrepLinear
    skip: MACEPlainLinear
    average_neighbor_count: float = eqx.field(static=True)

    def __init__(
        self,
        linear_up: O3IrrepLinear,
        attention_down: O3IrrepLinear,
        tensor_product: O3TensorProduct,
        radial: RadialMLP,
        linear_down: O3IrrepLinear,
        skip: MACEPlainLinear,
        /,
        *,
        average_neighbor_count: float,
    ) -> None:
        _require_coupling(linear_up, tensor_product, linear_down)
        down = attention_down.out_layout
        if (
            attention_down.in_layout.layout_id != linear_up.in_layout.layout_id
            or len(down.blocks) != 1
            or (down.blocks[0].degree, down.blocks[0].parity) != (0, 1)
            or down.blocks[0].multiplicity != ATTENTION_PROJECTION_WIDTH
        ):
            raise ValueError(
                f"The attention projection maps the layer input onto "
                f"{ATTENTION_PROJECTION_WIDTH} x 0e."
            )
        if radial.widths[1:-1] != ATTENTION_RADIAL_WIDTHS or (
            radial.output_width != tensor_product.plan.parameter_count
        ):
            raise ValueError(
                "The attention radial network has fixed 256, 256, 256 hidden widths "
                "and emits every coupling path weight."
            )
        if skip.in_layout.layout_id != linear_up.in_layout.layout_id:
            raise ValueError("The attention residual reads the layer input.")
        self.linear_up = linear_up
        self.attention_down = attention_down
        self.tensor_product = tensor_product
        self.radial = radial
        self.linear_down = linear_down
        self.skip = skip
        self.average_neighbor_count = positive_finite_float(
            average_neighbor_count, "average_neighbor_count"
        )

    @property
    def kind(self) -> MACEInteractionKind:
        return "real-agnostic-attention-residual"

    @property
    def residual(self) -> bool:
        return True

    @property
    def density(self) -> None:
        return None

    @property
    def input_layout(self) -> O3IrrepLayout:
        return self.linear_up.in_layout

    @property
    def message_layout(self) -> O3IrrepLayout:
        return self.linear_down.in_layout

    @property
    def target_layout(self) -> O3IrrepLayout:
        return self.linear_down.out_layout

    @property
    def residual_layout(self) -> O3IrrepLayout:
        return self.skip.out_layout

    def _projection(self, features: Array, atom_mask: Array | None, /) -> Array:
        down = self.attention_down(features)
        if atom_mask is None:
            return down
        return jnp.where(atom_mask[:, None], down, jnp.zeros((), dtype=down.dtype))

    def source_rows(
        self, features: Array, species: Array, atom_mask: Array, /
    ) -> MACESourceRows:
        lifted = self.linear_up(features)
        zero = jnp.zeros((), dtype=lifted.dtype)
        return MACESourceRows(
            jnp.where(atom_mask[:, None], lifted, zero),
            species,
            self._projection(features, atom_mask),
        )

    def receiver_context(self, features: Array, /) -> Array:
        return self._projection(features, None)

    def edge_weights(
        self,
        radial_features: Array,
        cutoff: Array | None,
        source_context: Array | None,
        receiver_context: Array | None,
        sender: Array,
        receiver: Array,
        /,
    ) -> MACEEdgeWeights:
        if source_context is None or receiver_context is None:
            raise ValueError("Attention edges need both endpoint projections.")
        augmented = jnp.concatenate(
            [radial_features, source_context, receiver_context], axis=-1
        )
        return MACEEdgeWeights(_with_cutoff(self.radial(augmented), cutoff), None)

    def receiver_message(
        self,
        aggregate: dict[str, Array],
        species: Array,
        context: Array | None,
        /,
    ) -> Array:
        return self.linear_down(aggregate["coupling"]) / self.average_neighbor_count

    def residual_state(self, features: Array, species: Array, /) -> Array:
        return self.skip(features)


@final
class MACENonlinearInteraction(StrictModule):
    """The source residual-nonlinear interaction block.

    Edge networks read ``[radial, E_src[s_j], E_tgt[s_i]]`` with species
    embeddings ``E[s] = W[s] / sqrt(S)``; both the weight network and the
    ``tanh(.)^2`` density network are biased ``Linear -> LayerNorm -> SiLU``
    stacks. The receiver forms ``linear_1(m) / (beta * density + alpha) +
    linear_res(linear_up(x_i))``, applies the gate and ``linear_2``; the residual
    is a plain linear map of the layer input. ``alpha`` and ``beta`` are
    trainable scalars.
    """

    linear_up: O3IrrepLinear
    source_embedding: Array
    target_embedding: Array
    tensor_product: O3TensorProduct
    radial: LayerNormRadialMLP
    density: LayerNormRadialMLP
    linear_down: O3IrrepLinear
    residual_linear: MACEPlainLinear
    gate: MACEGate
    output_linear: O3IrrepLinear
    skip: MACEPlainLinear
    density_alpha: Array
    density_beta: Array

    def __init__(
        self,
        linear_up: O3IrrepLinear,
        source_embedding: ArrayLike,
        target_embedding: ArrayLike,
        tensor_product: O3TensorProduct,
        radial: LayerNormRadialMLP,
        density: LayerNormRadialMLP,
        linear_down: O3IrrepLinear,
        residual_linear: MACEPlainLinear,
        gate: MACEGate,
        output_linear: O3IrrepLinear,
        skip: MACEPlainLinear,
        density_alpha: ArrayLike,
        density_beta: ArrayLike,
        /,
    ) -> None:
        _require_coupling(linear_up, tensor_product, linear_down)
        source = jnp.asarray(source_embedding)
        target = jnp.asarray(target_embedding)
        alpha = jnp.asarray(density_alpha)
        beta = jnp.asarray(density_beta)
        dtype = source.dtype
        if (
            source.ndim != 2
            or target.shape != source.shape
            or any(value.dtype != dtype for value in (target, alpha, beta))
            or alpha.shape != ()
            or beta.shape != ()
        ):
            raise ValueError(
                "Species embeddings must share one (species, channels) shape and the "
                "density normalization is two scalars, all in one dtype."
            )
        channels = source.shape[1]
        edge_width = radial.input_width
        if density.input_width != edge_width or edge_width <= 2 * channels:
            raise ValueError("Both edge networks read radial and endpoint embeddings.")
        if density.widths[1:] != (NONLINEAR_DENSITY_WIDTH, 1) or (
            density.postprocess != "tanh-square"
        ):
            raise ValueError(
                "The nonlinear density network is a tanh-squared 64-wide edge scalar."
            )
        if radial.output_width != tensor_product.plan.parameter_count:
            raise ValueError("The radial network must emit every coupling path weight.")
        nonlinear = gate.input_layout
        if (
            linear_down.out_layout.layout_id != nonlinear.layout_id
            or residual_linear.in_layout.layout_id != linear_up.out_layout.layout_id
            or residual_linear.out_layout.layout_id != nonlinear.layout_id
            or output_linear.in_layout.layout_id != gate.target_layout.layout_id
            or output_linear.out_layout.layout_id != gate.target_layout.layout_id
            or skip.in_layout.layout_id != linear_up.in_layout.layout_id
        ):
            raise ValueError("The nonlinear interaction layouts are inconsistent.")
        if gate.target_layout.blocks[0].multiplicity != channels:
            raise ValueError("Species embeddings have one entry per target channel.")
        self.linear_up = linear_up
        self.source_embedding = source
        self.target_embedding = target
        self.tensor_product = tensor_product
        self.radial = radial
        self.density = density
        self.linear_down = linear_down
        self.residual_linear = residual_linear
        self.gate = gate
        self.output_linear = output_linear
        self.skip = skip
        self.density_alpha = alpha
        self.density_beta = beta

    @property
    def kind(self) -> MACEInteractionKind:
        return "real-agnostic-residual-nonlinear"

    @property
    def residual(self) -> bool:
        return True

    @property
    def input_layout(self) -> O3IrrepLayout:
        return self.linear_up.in_layout

    @property
    def message_layout(self) -> O3IrrepLayout:
        return self.linear_down.in_layout

    @property
    def target_layout(self) -> O3IrrepLayout:
        return self.gate.target_layout

    @property
    def residual_layout(self) -> O3IrrepLayout:
        return self.skip.out_layout

    @property
    def species_count(self) -> int:
        return self.source_embedding.shape[0]

    def source_rows(
        self, features: Array, species: Array, atom_mask: Array, /
    ) -> MACESourceRows:
        lifted = self.linear_up(features)
        zero = jnp.zeros((), dtype=lifted.dtype)
        return MACESourceRows(jnp.where(atom_mask[:, None], lifted, zero), species)

    def receiver_context(self, features: Array, /) -> Array:
        """The receiver's own lifted state, read by ``linear_res``."""
        return self.linear_up(features)

    def edge_weights(
        self,
        radial_features: Array,
        cutoff: Array | None,
        source_context: Array | None,
        receiver_context: Array | None,
        sender: Array,
        receiver: Array,
        /,
    ) -> MACEEdgeWeights:
        dtype = self.source_embedding.dtype
        fan_in = jnp.sqrt(jnp.asarray(self.species_count, dtype=dtype))
        augmented = jnp.concatenate(
            [
                radial_features,
                self.source_embedding[sender] / fan_in,
                self.target_embedding[receiver] / fan_in,
            ],
            axis=-1,
        )
        weights = _with_cutoff(self.radial(augmented), cutoff)
        density = self.density(augmented)[..., 0]
        return MACEEdgeWeights(weights, density if cutoff is None else density * cutoff)

    def receiver_message(
        self,
        aggregate: dict[str, Array],
        species: Array,
        context: Array | None,
        /,
    ) -> Array:
        if context is None:
            raise ValueError("The nonlinear epilogue needs the receiver's lifted state.")
        normalization = (
            aggregate["density"][..., None] * self.density_beta + self.density_alpha
        )
        message = self.linear_down(aggregate["coupling"]) / normalization
        message = message + self.residual_linear(context)
        return self.output_linear(self.gate(message))

    def residual_state(self, features: Array, species: Array, /) -> Array:
        return self.skip(features)


MACEAnyInteraction: TypeAlias = (
    MACEInteraction | MACEAttentionInteraction | MACENonlinearInteraction
)


def _require_coupling(
    linear_up: O3IrrepLinear, tensor_product: O3TensorProduct, linear_down: O3IrrepLinear
) -> None:
    plan = tensor_product.plan
    if tensor_product.weight is not None:
        raise ValueError("The MACE coupling takes radial weights per edge.")
    if plan.left_representation.layout_id != linear_up.out_layout.layout_id:
        raise ValueError("linear_up must produce the coupling source layout.")
    if plan.output_representation.layout_id != linear_down.in_layout.layout_id:
        raise ValueError("The message linear must consume the coupling message layout.")


@final
class MACEProduct(StrictModule):
    """Symmetric product basis, product linear, and optional residual add."""

    contraction: SymmetricContraction
    linear: O3IrrepLinear
    target_layout: O3IrrepLayout
    self_connection: bool = eqx.field(static=True)

    def __init__(
        self,
        contraction: SymmetricContraction,
        linear: O3IrrepLinear,
        target_layout: O3IrrepLayout,
        /,
        *,
        self_connection: bool,
    ) -> None:
        if not isinstance(contraction, SymmetricContraction):
            raise TypeError("contraction must be a SymmetricContraction.")
        if not isinstance(linear, O3IrrepLinear):
            raise TypeError("linear must be an O3IrrepLinear.")
        if not isinstance(self_connection, bool):
            raise TypeError("self_connection must be a bool.")
        plan = contraction.plan
        _require_channel_view(plan.input_layout, target_layout, "product input")
        _require_channel_view(plan.output_layout, linear.in_layout, "product output")
        if linear.in_layout.layout_id != linear.out_layout.layout_id:
            raise ValueError("The product linear maps the product layout onto itself.")
        self.contraction = contraction
        self.linear = linear
        self.target_layout = target_layout
        self.self_connection = self_connection

    @property
    def output_layout(self) -> O3IrrepLayout:
        return self.linear.out_layout

    def __call__(
        self, message: Array, species: Array, residual: Array | None, /
    ) -> Array:
        """``message[N, P_target]``, ``species[N]`` -> node state ``[N, P_out]``."""
        contracted = self.contraction(to_channels(self.target_layout, message), species)
        state = self.linear(from_channels(self.output_layout, contracted))
        if self.self_connection and residual is not None:
            return state + residual
        return state


def _require_channel_view(
    channel: O3IrrepLayout, packed: O3IrrepLayout, name: str, /
) -> None:
    view = tuple((block.degree, block.parity) for block in channel.blocks)
    declared = tuple((block.degree, block.parity) for block in packed.blocks)
    multiplicities = {block.multiplicity for block in packed.blocks}
    if view != declared or len(multiplicities) != 1:
        raise ValueError(f"The {name} channel view does not match its packed layout.")


@final
class MACELayer(StrictModule):
    """One interaction, its product, readout, and (first layer) ZBL edge term."""

    interaction: MACEAnyInteraction
    product: MACEProduct
    readout: MACEReadout | None
    readout_product: MACEInvariantReadoutProduct | None
    pair_repulsion: MACEPairRepulsion | None

    def __init__(
        self,
        interaction: MACEAnyInteraction,
        product: MACEProduct,
        /,
        *,
        readout: MACEReadout | None,
        readout_product: MACEInvariantReadoutProduct | None = None,
        pair_repulsion: MACEPairRepulsion | None = None,
    ) -> None:
        if not isinstance(
            interaction,
            (MACEInteraction, MACEAttentionInteraction, MACENonlinearInteraction),
        ):
            raise TypeError("interaction must be an admitted MACE interaction.")
        if not isinstance(product, MACEProduct):
            raise TypeError("product must be a MACEProduct.")
        if product.target_layout.layout_id != interaction.target_layout.layout_id:
            raise ValueError("The product must consume the interaction target layout.")
        residual_layout = interaction.residual_layout
        if (
            residual_layout is not None
            and residual_layout.layout_id != product.output_layout.layout_id
        ):
            raise ValueError("A residual must produce the product output layout.")
        if readout_product is not None and readout is None:
            raise ValueError("The invariant readout product feeds a nonlinear readout.")
        scalar = product.output_layout.blocks[0]
        if (scalar.degree, scalar.parity) != (0, 1):
            raise ValueError("Layer outputs begin with their scalar block.")
        self.interaction = interaction
        self.product = product
        self.readout = readout
        self.readout_product = readout_product
        self.pair_repulsion = pair_repulsion

    @property
    def output_layout(self) -> O3IrrepLayout:
        return self.product.output_layout

    def readout_scalars(self, state: Array, species: Array, /) -> Array:
        """Scalar channels entering this layer's readout."""
        if self.readout_product is not None:
            return self.readout_product(to_channels(self.output_layout, state), species)
        return state[..., : self.output_layout.blocks[0].multiplicity]

    def readout_energy(self, state: Array, species: Array, head: int, /) -> Array:
        """Unscaled per-node readout energy of one head (zero without readout)."""
        if self.readout is None:
            return jnp.zeros(state.shape[:-1], dtype=state.dtype)
        return self.readout(self.readout_scalars(state, species), head)


def admit_mace_coupling(
    coupling: MACEAcceleratedCoupling, layer: MACELayer, dtype: DTypeLike, /
) -> None:
    """Refuse an accelerated coupling that cannot execute ``layer`` exactly.

    Lowering the layer's coupling structure refuses non-``uvu``, unweighted
    or non-uniform layouts here; the kernel admits tiles, extents and budget
    when each fragment binds (static, before any execution). Every admitted
    interaction family feeds the same coupling: its edge weights (including
    attention projections or species embeddings) are evaluated lane-locally by
    the exact family equations before the kernel runs.
    """
    if not isinstance(coupling, MACEAcceleratedCoupling):
        raise TypeError("The accelerated coupling must be a MACEAcceleratedCoupling.")
    if coupling.plan.dtype != jnp.dtype(dtype):
        raise ValueError(
            f"The accelerated coupling computes in {coupling.plan.precision}; the model "
            f"computes in {jnp.dtype(dtype).name}."
        )
    MACEEdgeCouplingSpec(layer.interaction.tensor_product)


def fragment_messages(
    coupling: MACEAcceleratedCoupling,
    layer: MACELayer,
    fragment: StreamedFragment,
    seed: dict[str, KeyGroupAccumulation],
    source_features: Array,
    harmonics: Array,
    weights: Array,
    density: Array | None,
    pair_energy: Array | None,
    /,
) -> dict[str, KeyGroupAccumulation]:
    """Seeded fragment aggregates of one layer from fragment-local edge factors.

    The coupling runs the accelerated fragment kernel over the fragment's
    distinct source rows; density and ZBL scalars use the substrate's seeded
    reduction over the same lanes.
    """
    messages = {
        "coupling": coupling.fragment(
            layer.interaction.tensor_product,
            fragment,
            source_features[fragment.source_ids],
            harmonics,
            weights,
            seed["coupling"],
        )
    }
    if density is not None:
        messages["density"] = fragment.reduce(density, seed["density"])
    if pair_energy is not None:
        messages["pair"] = fragment.reduce(pair_energy, seed["pair"])
    return messages


@final
class MACEStreamedLayer(StrictModule):
    """Edge function or fragment aggregator and receiver epilogue of one layer.

    Methods are used unbound as the substrate's pure callbacks, so every array
    reaches them through this dynamic parameter PyTree. ``coupling`` (an
    admitted accelerated kernel) selects `fragment` aggregation.
    """

    layer: MACELayer
    geometry: MACEEdgeGeometry
    coupling: MACEAcceleratedCoupling | None
    head: int = eqx.field(static=True)

    def __init__(
        self,
        layer: MACELayer,
        geometry: MACEEdgeGeometry,
        head: int,
        /,
        *,
        coupling: MACEAcceleratedCoupling | None = None,
    ) -> None:
        self.layer = layer
        self.geometry = geometry
        self.coupling = coupling
        self.head = head

    def payload_shapes(
        self, dtype: DTypeLike, /
    ) -> tuple[dict[str, jax.ShapeDtypeStruct], dict[str, jax.ShapeDtypeStruct]]:
        """Declared one-event message and one-receiver output structures."""
        dtype_ = jnp.dtype(dtype)
        scalar = jax.ShapeDtypeStruct((), dtype_)
        interaction = self.layer.interaction
        message = {
            "coupling": jax.ShapeDtypeStruct(
                (interaction.message_layout.packed_size,), dtype_
            )
        }
        if interaction.density is not None:
            message["density"] = scalar
        if self.layer.pair_repulsion is not None:
            message["pair"] = scalar
        output = {
            "energy": scalar,
            "features": jax.ShapeDtypeStruct(
                (self.layer.output_layout.packed_size,), dtype_
            ),
        }
        return message, output

    def edge(
        self,
        source: MACESourceRows,
        receiver: MACEReceiverRows,
        vector: Array,
        /,
    ) -> dict[str, Array]:
        """Message of one directed edge with displacement ``r_recv - r_src (+ image)``."""
        distance = jnp.sqrt(jnp.sum(vector * vector, axis=-1))
        features, cutoff = self.geometry.radial_features(
            distance, source.species, receiver.species
        )
        harmonics = self.geometry.harmonics(vector).astype(features.dtype)
        interaction = self.layer.interaction
        edge = interaction.edge_weights(
            features,
            cutoff,
            source.context,
            receiver.context,
            source.species,
            receiver.species,
        )
        message = {
            "coupling": interaction.tensor_product(
                source.features, harmonics, edge.weights
            )
        }
        if edge.density is not None:
            message["density"] = edge.density
        pair = self.layer.pair_repulsion
        if pair is not None:
            message["pair"] = pair.edge_energy(distance, source.species, receiver.species)
        return message

    def fragment(
        self,
        sources: MACESourceRows,
        receivers: MACEReceiverRows,
        vectors: Array,
        fragment: StreamedFragment,
        seed: dict[str, KeyGroupAccumulation],
        /,
    ) -> dict[str, KeyGroupAccumulation]:
        """Seeded aggregates of one fragment; radial and harmonic factors stay local."""
        if self.coupling is None:
            raise ValueError("Fragment aggregation needs an accelerated coupling.")
        distance = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
        sender = sources.species[fragment.lane_sources]
        receiver = receivers.species[fragment.lane_slots]
        features, cutoff = self.geometry.radial_features(distance, sender, receiver)
        edge = self.layer.interaction.edge_weights(
            features,
            cutoff,
            None if sources.context is None else sources.context[fragment.lane_sources],
            None if receivers.context is None else receivers.context[fragment.lane_slots],
            sender,
            receiver,
        )
        pair = self.layer.pair_repulsion
        return fragment_messages(
            self.coupling,
            self.layer,
            fragment,
            seed,
            sources.features,
            self.geometry.harmonics(vectors).astype(features.dtype),
            edge.weights,
            edge.density,
            None if pair is None else pair.edge_energy(distance, sender, receiver),
        )

    def epilogue(
        self, receiver: MACEReceiverRows, aggregate: dict[str, Array], /
    ) -> dict[str, Array]:
        """Complete receiver update and readout after its final edge."""
        species = receiver.species
        mask = receiver.mask
        interaction = self.layer.interaction
        message = interaction.receiver_message(aggregate, species, receiver.context)
        residual = interaction.residual_state(receiver.features, species)
        state = self.layer.product(
            message[None], species[None], None if residual is None else residual[None]
        )[0]
        energy = self.layer.readout_energy(state[None], species[None], self.head)[0]
        if self.layer.pair_repulsion is not None:
            energy = energy + aggregate["pair"]
        zero = jnp.zeros((), dtype=state.dtype)
        return {
            "energy": jnp.where(mask, energy, zero),
            "features": jnp.where(mask, state, zero),
        }


__all__ = [
    "admit_mace_coupling",
    "ATTENTION_PROJECTION_WIDTH",
    "ATTENTION_RADIAL_WIDTHS",
    "convention_parity",
    "convolution_paths",
    "fragment_messages",
    "from_channels",
    "gate_layout",
    "harmonic_layout",
    "interaction_family",
    "interaction_traits",
    "irrep_block_name",
    "linear_paths",
    "mace_irrep_layout",
    "MACEAnyInteraction",
    "MACEAttentionInteraction",
    "MACECutoffPlacement",
    "MACEEdgeGeometry",
    "MACEEdgeWeights",
    "MACEGate",
    "MACEInteraction",
    "MACEInteractionFamily",
    "MACEInteractionKind",
    "MACELayer",
    "MACENonlinearInteraction",
    "MACEParityConvention",
    "MACEPlainLinear",
    "MACEProduct",
    "MACEReceiverRows",
    "MACESelfConnection",
    "MACESourceRows",
    "MACEStreamedLayer",
    "natural_parity",
    "NONLINEAR_DENSITY_ALPHA",
    "NONLINEAR_DENSITY_BETA",
    "NONLINEAR_DENSITY_WIDTH",
    "self_connection_paths",
    "to_channels",
]

for _artifact in (
    MACEAttentionInteraction,
    MACEEdgeGeometry,
    MACEGate,
    MACEInteraction,
    MACELayer,
    MACENonlinearInteraction,
    MACEPlainLinear,
    MACEProduct,
    MACEReceiverRows,
    MACESelfConnection,
    MACESourceRows,
    MACEStreamedLayer,
):
    register_artifact_value(
        f"phydrax.nn.atomistic.internal:{_artifact.__name__}", _artifact
    )
del _artifact
