#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-faithful MACE interaction, product, and layer kernels.

One layer is split at the only nonlocal boundary of the architecture. Each
directed edge evaluates the radial weights, real harmonics and ``uvu`` coupling
of its sender's ``linear_up`` features (plus a scalar density and, in the first
layer, the ZBL share). Each receiver then owns its complete epilogue over the
summed messages: ``linear_down``, neighbor or density normalization, species
self-connection or residual, symmetric product, product linear, and readout.
The epilogue runs once per receiver after all of its edges, so only inter-layer
node states and readout energies leave a receiver tile.
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
    MACELinearReadout,
    MACENonlinearReadout,
    MACEPairRepulsion,
)
from ._radial import RadialEmbedding, RadialMLP
from ._symmetric_contraction import SymmetricContraction


MACEInteractionKind: TypeAlias = Literal[
    "real-agnostic",
    "real-agnostic-residual",
    "real-agnostic-density",
    "real-agnostic-density-residual",
]
MACECutoffPlacement: TypeAlias = Literal["embedding", "weights"]


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
        case unreachable:
            assert_never(unreachable)


def irrep_block_name(degree: int, parity: O3Parity, /) -> str:
    """Canonical name of the hidden/target block of one ``(l, parity)``."""
    return f"l{degree}{'e' if parity == 1 else 'o'}"


def natural_parity(degree: int, /) -> O3Parity:
    return 1 if degree % 2 == 0 else -1


def mace_irrep_layout(multiplicity: int, maximum_degree: int, /) -> O3IrrepLayout:
    """``multiplicity x (l, (-1)^l)`` for ``l = 0..maximum_degree`` in degree order."""
    return O3IrrepLayout(
        tuple(
            O3IrrepBlock(
                irrep_block_name(degree, natural_parity(degree)),
                degree,
                natural_parity(degree),
                multiplicity=multiplicity,
            )
            for degree in range(maximum_degree + 1)
        )
    )


def harmonic_layout(maximum_degree: int, /) -> O3IrrepLayout:
    """Real spherical-harmonic layout ``1 x (l, (-1)^l)``, degree-major."""
    return O3IrrepLayout(
        tuple(
            O3IrrepBlock(f"y{degree}", degree, natural_parity(degree))
            for degree in range(maximum_degree + 1)
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
    order that fixes both the radial-weight columns and ``linear_down`` rows.
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


def self_connection_paths(
    in_layout: O3IrrepLayout, out_layout: O3IrrepLayout, /
) -> tuple[tuple[int, int], ...]:
    """Input-major ``(input block, output block)`` pairs of identical ``(l, p)``."""
    return tuple(
        (left, right)
        for left, source in enumerate(in_layout.blocks)
        for right, target in enumerate(out_layout.blocks)
        if (source.degree, source.parity) == (target.degree, target.parity)
    )


class MACESourceRows(NamedTuple):
    """Node-local source payload of one layer; the halo-exchanged state."""

    features: Array
    species: Array


class MACEReceiverRows(NamedTuple):
    """Receiver-local payload of one layer epilogue."""

    features: Array
    species: Array
    mask: Array


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


@final
class MACEInteraction(StrictModule):
    """One source interaction block in its exhaustive admitted variants."""

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
        plan = tensor_product.plan
        if tensor_product.weight is not None:
            raise ValueError("The MACE coupling takes radial weights per edge.")
        if plan.left_representation.layout_id != linear_up.out_layout.layout_id:
            raise ValueError("linear_up must produce the coupling source layout.")
        if linear_up.in_layout.layout_id != linear_up.out_layout.layout_id:
            raise ValueError("linear_up maps the layer input layout onto itself.")
        if plan.output_representation.layout_id != linear_down.in_layout.layout_id:
            raise ValueError("linear_down must consume the coupling message layout.")
        if radial.output_width != plan.parameter_count:
            raise ValueError("The radial network must emit every coupling path weight.")
        if density is not None and (
            density.widths != (radial.input_width, 1)
            or density.postprocess != "tanh-square"
        ):
            raise ValueError(
                "Density normalization is a tanh-squared linear edge scalar."
            )
        kind_ = parse(kind, MACEInteractionKind, "kind")
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

    def edge_message(
        self,
        source: Array,
        harmonics: Array,
        radial_features: Array,
        cutoff: Array | None,
        /,
    ) -> dict[str, Array]:
        """Per-edge coupling message and, for density kinds, edge density."""
        weights = self.radial(radial_features)
        if cutoff is not None:
            weights = weights * cutoff
        message = {"coupling": self.tensor_product(source, harmonics, weights)}
        if self.density is not None:
            density = self.density(radial_features)[..., 0]
            message["density"] = density if cutoff is None else density * cutoff
        return message

    def receiver_message(self, aggregate: dict[str, Array], species: Array, /) -> Array:
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

    interaction: MACEInteraction
    product: MACEProduct
    readout: MACELinearReadout | MACENonlinearReadout | None
    readout_product: MACEInvariantReadoutProduct | None
    pair_repulsion: MACEPairRepulsion | None

    def __init__(
        self,
        interaction: MACEInteraction,
        product: MACEProduct,
        /,
        *,
        readout: MACELinearReadout | MACENonlinearReadout | None,
        readout_product: MACEInvariantReadoutProduct | None = None,
        pair_repulsion: MACEPairRepulsion | None = None,
    ) -> None:
        if not isinstance(interaction, MACEInteraction):
            raise TypeError("interaction must be a MACEInteraction.")
        if not isinstance(product, MACEProduct):
            raise TypeError("product must be a MACEProduct.")
        if product.target_layout.layout_id != interaction.target_layout.layout_id:
            raise ValueError("The product must consume the interaction target layout.")
        if interaction.residual:
            if (
                interaction.self_connection.out_layout.layout_id
                != product.output_layout.layout_id
            ):
                raise ValueError("A residual must produce the product output layout.")
        if readout_product is not None and not isinstance(readout, MACENonlinearReadout):
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

    def readout_energy(self, state: Array, species: Array, head: int, /) -> Array:
        """Unscaled per-node readout energy of one head (zero without readout)."""
        if self.readout is None:
            return jnp.zeros(state.shape[:-1], dtype=state.dtype)
        if self.readout_product is not None:
            scalars = self.readout_product(
                to_channels(self.output_layout, state), species
            )
        else:
            scalars = state[..., : self.output_layout.blocks[0].multiplicity]
        return self.readout(scalars, head)


def admit_mace_coupling(
    coupling: MACEAcceleratedCoupling, layer: MACELayer, dtype: DTypeLike, /
) -> None:
    """Refuse an accelerated coupling that cannot execute ``layer`` exactly.

    Lowering the layer's coupling structure refuses non-``uvu``, unweighted
    or non-uniform layouts here; the kernel admits tiles, extents and budget
    when each fragment binds (static, before any execution).
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
        message = self.layer.interaction.edge_message(
            source.features, harmonics, features, cutoff
        )
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
        interaction = self.layer.interaction
        weights = interaction.radial(features)
        density = (
            None if interaction.density is None else interaction.density(features)[..., 0]
        )
        if cutoff is not None:
            weights = weights * cutoff[..., None]
            density = None if density is None else density * cutoff
        pair = self.layer.pair_repulsion
        return fragment_messages(
            self.coupling,
            self.layer,
            fragment,
            seed,
            sources.features,
            self.geometry.harmonics(vectors).astype(features.dtype),
            weights,
            density,
            None if pair is None else pair.edge_energy(distance, sender, receiver),
        )

    def epilogue(
        self, receiver: MACEReceiverRows, aggregate: dict[str, Array], /
    ) -> dict[str, Array]:
        """Complete receiver update and readout after its final edge."""
        species = receiver.species
        mask = receiver.mask
        interaction = self.layer.interaction
        message = interaction.receiver_message(aggregate, species)
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
    "convolution_paths",
    "fragment_messages",
    "from_channels",
    "harmonic_layout",
    "interaction_traits",
    "irrep_block_name",
    "mace_irrep_layout",
    "MACECutoffPlacement",
    "MACEEdgeGeometry",
    "MACEInteraction",
    "MACEInteractionKind",
    "MACELayer",
    "MACEProduct",
    "MACEReceiverRows",
    "MACESelfConnection",
    "MACESourceRows",
    "MACEStreamedLayer",
    "natural_parity",
    "self_connection_paths",
    "to_channels",
]

for _artifact in (
    MACEEdgeGeometry,
    MACEInteraction,
    MACELayer,
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
