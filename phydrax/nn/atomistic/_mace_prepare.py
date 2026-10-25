#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""System-bound frozen MACE inference with identity-bound folds and tables.

Preparation binds one exact `MACEPotential` at one parameter revision and one
active species subset, then folds only genuinely linear, species-local stages:

* first-layer sources, edge contexts and residuals depend on species alone
  (absent a joint embedding, whose conditioning addend makes them
  case-dependent), so ``linear_up(embedding)``, the attention projections or
  nonlinear lifted receiver state, and the residual map become per-species
  rows;
* a standard non-residual receiver message ``self_connection(linear_down(m) /
  avg)`` is linear in ``m`` with no activation, residual, head, or field
  between the two maps, so it becomes one per-species matrix per target block
  (density normalization divides afterwards, a scalar that commutes with both
  maps); attention and residual-nonlinear receivers (plain skips, the
  ``beta * density + alpha`` normalization and the gate) run their exact
  equations;
* symmetric products bind merged ``C = P W`` coefficients to the original
  ``W`` revision;
* optionally, radial networks are replaced by qualified smooth tables for the
  active ordered species pairs. Tables realize standard interactions only:
  attention edge networks read node-feature projections and the
  residual-nonlinear edge networks are biased layer-normalized stacks over
  species embeddings, so a tabulated preparation of either refuses with
  `UnsupportedMACEPreparedRoute` before any payload is built.

Every fold is FIXED data bound to the source numeric revision; the prepared
form is inference-only and refuses stale weights. Folding changes rounding
order, so the prepared realization has its own identity, which records each
layer's family and executed folds. Tables admit only first coordinate
derivatives; Hessians and force training use the exact model. Restoration and
publication run `PreparedMACEPotential.validate`, which re-admits the
realization, re-derives every executed payload from the embedded exact model
and the admitted table declaration; recorded identities are recomputed, never
trusted.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, assert_never, final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._identity import NumericRevision
from ..._model import register_artifact_value
from ..._model._fixed_binding import require_identical, require_realized
from ..._strict import StrictModule
from ..._trainable import ExplicitFreeze, NonTrainableState
from ...atomistic._conditioning import (
    AtomisticConditioningSpec,
    AtomisticModelOutputs,
    AtomisticObservable,
    AtomisticObservableRequest,
)
from ...atomistic._model_inputs import AtomisticGraphInputs
from ...atomistic._potential import (
    AbstractPreparedAtomisticPotential,
    atomistic_model_revision,
    AtomisticPotentialCapabilities,
    AtomisticPotentialRequirements,
)
from ...ein import contract
from ...sparse._execution import KeyGroupAccumulation
from ...sparse._streamed import (
    PreparedStreamedRelation,
    StreamedFragment,
    StreamedPayloadSpec,
)
from ...typing import checked, Dim, Int32, parse
from ._mace import MACELayerContext, MACENodeStates, MACEPotential
from ._mace_interaction import (
    admit_mace_coupling,
    fragment_messages,
    from_channels,
    interaction_family,
    MACEAttentionInteraction,
    MACEEdgeGeometry,
    MACEEdgeWeights,
    MACEInteraction,
    MACELayer,
    MACENonlinearInteraction,
    MACESourceRows,
    to_channels,
)
from ._mace_kernels import MACEAcceleratedCoupling
from ._mace_les import MACELESPotential
from ._radial import RadialMLP
from ._radial_projection import (
    prepare_radial_tables,
    PreparedRadialTables,
    RadialSpeciesBinding,
    RadialTableDeclaration,
)
from ._symmetric_contraction import MergedSymmetricContraction


MACERadialRealization: TypeAlias = Literal["exact", "tabulated"]
MACEPreparedRouteRefusal: TypeAlias = Literal[
    "latent-ewald-composition",
    "tabulated-output-cutoff",
    "tabulated-attention-edge-network",
    "tabulated-nonlinear-edge-network",
    "context-energy-conditioning",
]


class StaleMACEPreparation(ValueError):
    """A prepared MACE realization is bound to different source parameters."""


class UnsupportedMACEPreparedRoute(ValueError):
    """A model family has no exact realization on the requested prepared route.

    ``reason`` names the refusal; it is raised before any payload is prepared
    (or, for a context energy, before any evaluation), never replaced by
    running the exact model under the prepared name.
    """

    reason: MACEPreparedRouteRefusal

    def __init__(self, reason: MACEPreparedRouteRefusal, message: str, /) -> None:
        super().__init__(f"[{reason}] {message}")
        self.reason = reason


class MACEActiveSpeciesDim(Dim):
    """Model species, mapped to their active-subset position."""


class MACERawSpeciesDim(Dim):
    """Raw species identifiers ``0..maximum``."""


@final
class MACEActiveSpecies(StrictModule, NonTrainableState):
    """Binding of raw species identifiers to model and active-subset indices."""

    __strict_contract__ = True

    model_lookup: Int32[MACERawSpeciesDim]
    active_lookup: Int32[MACEActiveSpeciesDim]
    active_species: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, model: Sequence[int], active: Sequence[int], /) -> None:
        model_ = tuple(model)
        active_ = tuple(active)
        if not active_ or len(set(active_)) != len(active_):
            raise ValueError("Active species must be distinct and non-empty.")
        unknown = sorted(set(active_) - set(model_))
        if unknown:
            raise ValueError(f"Active species {unknown} are outside the model domain.")
        raw = np.full((max(model_) + 1,), -1, dtype=np.int32)
        raw[np.asarray(model_, dtype=np.int64)] = np.arange(len(model_), dtype=np.int32)
        positions = np.full((len(model_),), -1, dtype=np.int32)
        for index, value in enumerate(active_):
            positions[model_.index(value)] = index
        self.model_lookup = jnp.asarray(raw)
        self.active_lookup = jnp.asarray(positions)
        self.active_species = active_

    def indices(self, identifiers: Array, mask: Array, /) -> tuple[Array, Array]:
        """``(model, active)`` indices; inactive or unknown active particles fail."""
        raw = jnp.asarray(identifiers, dtype=jnp.int32)
        maximum = self.model_lookup.shape[0] - 1
        inside = (raw >= 0) & (raw <= maximum)
        model = jnp.where(inside, self.model_lookup[jnp.clip(raw, 0, maximum)], -1)
        active = jnp.where(model >= 0, self.active_lookup[jnp.maximum(model, 0)], -1)
        active = eqx.error_if(
            active,
            jnp.any(mask & (active < 0)),
            "Particle species is outside the prepared MACE active species.",
        )
        zero = jnp.zeros_like(model)
        return jnp.where(mask, model, zero), jnp.where(mask, active, zero)


@final
class PreparedMACELayer(StrictModule):
    """Folded, merged and (optionally) tabulated realization of one layer.

    Fixed by its `PreparedMACEPotential` holder, which freezes the embedded exact
    layer on purpose. Species-indexed fold rows are ``None`` where the fold does
    not apply and the exact map runs instead.
    """

    layer: MACELayer
    contraction: MergedSymmetricContraction
    readout_contraction: MergedSymmetricContraction | None
    source_rows: Array | None
    source_context_rows: Array | None
    receiver_context_rows: Array | None
    residual_rows: Array | None
    message_maps: tuple[Array, ...] | None
    radial_table: PreparedRadialTables | None
    density_table: PreparedRadialTables | None


class _LayerFolds(NamedTuple):
    """Every species-indexed fold of one layer, ``None`` where it does not apply."""

    source_rows: Array | None
    source_context_rows: Array | None
    receiver_context_rows: Array | None
    residual_rows: Array | None
    message_maps: tuple[Array, ...] | None


def _prepared_folds(prepared: PreparedMACELayer, /) -> _LayerFolds:
    return _LayerFolds(
        prepared.source_rows,
        prepared.source_context_rows,
        prepared.receiver_context_rows,
        prepared.residual_rows,
        prepared.message_maps,
    )


def _active_model_indices(
    potential: MACEPotential, active: Sequence[int], /
) -> np.ndarray:
    species = potential.configuration.species
    return np.asarray([species.index(value) for value in active], dtype=np.int32)


def _fold_sources(
    potential: MACEPotential, layer: MACELayer, model_indices: np.ndarray, /
) -> tuple[Array, Array | None, Array | None, Array | None]:
    """First-layer species rows of sources, edge contexts and the residual.

    Valid only without a joint embedding: the layer input is then the species
    embedding alone, so every map of it is a function of species.
    """
    species = jnp.asarray(model_indices)
    mask = jnp.ones(species.shape, dtype=jnp.bool_)
    embedded = potential.init_features(species, mask, MACELayerContext(None, None))
    interaction = layer.interaction
    sources = interaction.source_rows(embedded, species, mask)
    return (
        sources.features,
        sources.context,
        interaction.receiver_context(embedded),
        interaction.residual_state(embedded, species),
    )


def _fold_message(
    interaction: MACEInteraction, model_indices: np.ndarray, /
) -> tuple[Array, ...]:
    """Per-species ``self_connection o linear_down / avg`` maps of a plain layer."""
    down = interaction.linear_down
    connection = interaction.self_connection
    target = interaction.target_layout
    species = jnp.asarray(model_indices)
    dtype = down.weights[0].dtype
    maps = []
    for block_index, block in enumerate(target.blocks):
        path = connection.paths.index((block_index, block_index))
        weight = connection.weights[path][:, species, :]
        connection_scale = jnp.sqrt(
            jnp.asarray(block.multiplicity * connection.species_count, dtype=dtype)
        )
        down_weight = down.weights[block_index]
        down_scale = jnp.sqrt(jnp.asarray(down_weight.shape[1], dtype=dtype))
        normalization = (
            jnp.ones((), dtype=dtype)
            if interaction.density is not None
            else jnp.asarray(interaction.average_neighbor_count, dtype=dtype)
        )
        maps.append(
            contract("usw,uf->swf", weight, down_weight)
            / (connection_scale * down_scale * normalization)
        )
    return tuple(maps)


def _layer_folds(
    potential: MACEPotential, index: int, model_indices: np.ndarray, /
) -> _LayerFolds:
    """Every species-indexed fold of one layer of ``potential``.

    The message fold assumes the standard species self-connection directly
    after ``linear_down``; attention and residual-nonlinear receivers keep their
    exact epilogue.
    """
    layer = potential.layers[index]
    interaction = layer.interaction
    sources = source_context = receiver_context = residual = None
    if index == 0 and potential.joint_embedding is None:
        sources, source_context, receiver_context, residual = _fold_sources(
            potential, layer, model_indices
        )
    message_maps = (
        _fold_message(interaction, model_indices)
        if isinstance(interaction, MACEInteraction) and not interaction.residual
        else None
    )
    return _LayerFolds(sources, source_context, receiver_context, residual, message_maps)


def _require_tabulable(potential: MACEPotential, /) -> None:
    """Refuse a tabulated preparation of any layer tables cannot realize exactly."""
    if potential.configuration.cutoff_placement != "embedding":
        raise UnsupportedMACEPreparedRoute(
            "tabulated-output-cutoff",
            "Radial tables realize the cutoff-embedded network only; this "
            "model applies its cutoff to network outputs.",
        )
    for index, layer in enumerate(potential.layers):
        interaction = layer.interaction
        if isinstance(interaction, MACEAttentionInteraction):
            raise UnsupportedMACEPreparedRoute(
                "tabulated-attention-edge-network",
                f"Layer {index} is an attention-residual interaction whose edge "
                "network reads node-feature projections of both endpoints; its "
                "weights are not a function of distance and species, so no radial "
                "table realizes them. Prepare it with radial='exact'.",
            )
        if isinstance(interaction, MACENonlinearInteraction):
            raise UnsupportedMACEPreparedRoute(
                "tabulated-nonlinear-edge-network",
                f"Layer {index} is a residual-nonlinear interaction whose weight and "
                "density networks are biased layer-normalized stacks over radial "
                "features and trainable species embeddings; the radial table owner "
                "tabulates only bias-free radial MLPs of radial features. Prepare "
                "it with radial='exact'.",
            )


def _admit_realization(
    potential: MACEPotential, radial: MACERadialRealization, /
) -> None:
    """Refuse a realization x family combination with no exact prepared route."""
    match radial:
        case "exact":
            return
        case "tabulated":
            _require_tabulable(potential)
        case unreachable:
            assert_never(unreachable)


def _standard_interaction(layer: MACELayer, index: int, /) -> MACEInteraction:
    """The standard interaction of a tabulated layer (admitted by `_require_tabulable`)."""
    interaction = layer.interaction
    if not isinstance(interaction, MACEInteraction):
        raise UnsupportedMACEPreparedRoute(
            "tabulated-attention-edge-network"
            if isinstance(interaction, MACEAttentionInteraction)
            else "tabulated-nonlinear-edge-network",
            f"Layer {index} {interaction.kind} edge networks have no radial table.",
        )
    return interaction


def _table_rows(potential: MACEPotential, active: Sequence[int], /) -> tuple[int, ...]:
    """Tables bind active rows by ascending position in the model domain."""
    domain = potential.configuration.species
    return tuple(sorted(domain.index(value) for value in active))


def _density_declaration(
    declaration: RadialTableDeclaration, /
) -> RadialTableDeclaration:
    """tanh(x^2) does not commute with interpolation, so the density table
    always stores postprocessed (projected-width) outputs."""
    return RadialTableDeclaration(
        declaration.grid_min, declaration.node_count, layout="projected-width"
    )


def _prepare_layer(
    potential: MACEPotential,
    index: int,
    model_indices: np.ndarray,
    radial: MACERadialRealization,
    declaration: RadialTableDeclaration | None,
    active: Sequence[int],
    /,
) -> PreparedMACELayer:
    layer = potential.layers[index]
    folds = _layer_folds(potential, index, model_indices)
    radial_table = density_table = None
    match radial:
        case "exact":
            pass
        case "tabulated":
            if declaration is None:
                raise ValueError(
                    "Tabulated radial realization needs a table declaration."
                )
            interaction = _standard_interaction(layer, index)
            domain = potential.configuration.species
            rows = _table_rows(potential, active)
            radial_table = prepare_radial_tables(
                potential.geometry.radial,
                interaction.radial,
                declaration,
                species_domain=domain,
                active_species=rows,
            )
            if interaction.density is not None:
                density_table = prepare_radial_tables(
                    potential.geometry.radial,
                    interaction.density,
                    _density_declaration(declaration),
                    species_domain=domain,
                    active_species=rows,
                )
        case unreachable:
            assert_never(unreachable)
    return PreparedMACELayer(
        layer=layer,
        contraction=layer.product.contraction.merged(),
        readout_contraction=None
        if layer.readout_product is None
        else layer.readout_product.contraction.merged(),
        source_rows=folds.source_rows,
        source_context_rows=folds.source_context_rows,
        receiver_context_rows=folds.receiver_context_rows,
        residual_rows=folds.residual_rows,
        message_maps=folds.message_maps,
        radial_table=radial_table,
        density_table=density_table,
    )


@final
class PreparedMACEStreamedLayer(StrictModule):
    """Edge function or fragment aggregator and receiver epilogue of one prepared layer.

    Edge and receiver rows carry model species (and the family's edge
    contexts); receiver rows additionally carry active-subset species for the
    folded per-species maps. ``coupling`` selects accelerated `fragment`
    aggregation, which consumes the same lane-local edge weights.
    """

    prepared: PreparedMACELayer
    geometry: MACEEdgeGeometry
    coupling: MACEAcceleratedCoupling | None
    head: int = eqx.field(static=True)
    first: bool = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedMACELayer,
        geometry: MACEEdgeGeometry,
        head: int,
        first: bool,
        /,
        *,
        coupling: MACEAcceleratedCoupling | None = None,
    ) -> None:
        self.prepared = prepared
        self.geometry = geometry
        self.coupling = coupling
        self.head = head
        self.first = first

    def payload_shapes(self, dtype: np.dtype, /) -> tuple[dict[str, Any], dict[str, Any]]:
        scalar = jax.ShapeDtypeStruct((), dtype)
        layer = self.prepared.layer
        message: dict[str, Any] = {
            "coupling": jax.ShapeDtypeStruct(
                (layer.interaction.message_layout.packed_size,), dtype
            )
        }
        if layer.interaction.density is not None:
            message["density"] = scalar
        if layer.pair_repulsion is not None:
            message["pair"] = scalar
        output = {
            "energy": scalar,
            "features": jax.ShapeDtypeStruct((layer.output_layout.packed_size,), dtype),
        }
        return message, output

    def edge_weights(
        self,
        distance: Array,
        sender: Array,
        receiver: Array,
        source_context: Array | None,
        receiver_context: Array | None,
        /,
    ) -> MACEEdgeWeights:
        """Coupling weights and edge density from tables or the exact family networks."""
        table = self.prepared.radial_table
        if table is None:
            features, cutoff = self.geometry.radial_features(distance, sender, receiver)
            return self.prepared.layer.interaction.edge_weights(
                features, cutoff, source_context, receiver_context, sender, receiver
            )
        density_table = self.prepared.density_table
        return MACEEdgeWeights(
            table.evaluate(distance, sender, receiver),
            None
            if density_table is None
            else density_table.evaluate(distance, sender, receiver)[..., 0],
        )

    def edge(
        self, source: MACESourceRows, receiver: _PreparedReceiverRows, vector: Array, /
    ) -> dict[str, Array]:
        distance = jnp.sqrt(jnp.sum(vector * vector, axis=-1))
        edge = self.edge_weights(
            distance, source.species, receiver.species, source.context, receiver.context
        )
        harmonics = self.geometry.harmonics(vector).astype(edge.weights.dtype)
        layer = self.prepared.layer
        message = {
            "coupling": layer.interaction.tensor_product(
                source.features, harmonics, edge.weights
            )
        }
        if edge.density is not None:
            message["density"] = edge.density
        if layer.pair_repulsion is not None:
            message["pair"] = layer.pair_repulsion.edge_energy(
                distance, source.species, receiver.species
            )
        return message

    def fragment(
        self,
        sources: MACESourceRows,
        receivers: _PreparedReceiverRows,
        vectors: Array,
        fragment: StreamedFragment,
        seed: dict[str, KeyGroupAccumulation],
        /,
    ) -> dict[str, KeyGroupAccumulation]:
        """Seeded aggregates of one fragment; edge-weight and harmonic factors stay local."""
        if self.coupling is None:
            raise ValueError("Fragment aggregation needs an accelerated coupling.")
        distance = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
        sender = sources.species[fragment.lane_sources]
        receiver = receivers.species[fragment.lane_slots]
        edge = self.edge_weights(
            distance,
            sender,
            receiver,
            None if sources.context is None else sources.context[fragment.lane_sources],
            None if receivers.context is None else receivers.context[fragment.lane_slots],
        )
        pair = self.prepared.layer.pair_repulsion
        return fragment_messages(
            self.coupling,
            self.prepared.layer,
            fragment,
            seed,
            sources.features,
            self.geometry.harmonics(vectors).astype(edge.weights.dtype),
            edge.weights,
            edge.density,
            None if pair is None else pair.edge_energy(distance, sender, receiver),
        )

    def epilogue(
        self, receiver: _PreparedReceiverRows, aggregate: dict[str, Array], /
    ) -> dict[str, Array]:
        prepared = self.prepared
        layer = prepared.layer
        interaction = layer.interaction
        species = receiver.species
        message = _receiver_message(
            prepared, aggregate, species, receiver.active, receiver.context
        )
        residual = (
            prepared.residual_rows[receiver.active]
            if prepared.residual_rows is not None
            else interaction.residual_state(receiver.features, species)
        )
        contracted = prepared.contraction(
            to_channels(layer.product.target_layout, message)[None], species[None]
        )[0]
        state = layer.product.linear(from_channels(layer.output_layout, contracted))
        if layer.product.self_connection and residual is not None:
            state = state + residual
        energy = _readout_energy(prepared, state, species, self.head)
        if layer.pair_repulsion is not None:
            energy = energy + aggregate["pair"]
        zero = jnp.zeros((), dtype=state.dtype)
        return {
            "energy": jnp.where(receiver.mask, energy, zero),
            "features": jnp.where(receiver.mask, state, zero),
        }


class _PreparedReceiverRows(NamedTuple):
    """Receiver payload of one prepared layer epilogue.

    ``context`` is the family's receiver context (attention projection or the
    nonlinear lifted state), or ``None``.
    """

    features: Array
    species: Array
    active: Array
    mask: Array
    context: Array | None = None


def _receiver_message(
    prepared: PreparedMACELayer,
    aggregate: dict[str, Array],
    species: Array,
    active: Array,
    context: Array | None,
    /,
) -> Array:
    interaction = prepared.layer.interaction
    if prepared.message_maps is None:
        return interaction.receiver_message(aggregate, species, context)
    if not isinstance(interaction, MACEInteraction):
        raise ValueError("Folded message maps realize a standard interaction only.")
    blocks = interaction.message_layout.split(aggregate["coupling"])
    target = interaction.target_layout
    mixed = []
    for maps, matches in zip(
        prepared.message_maps, interaction.linear_down.sources, strict=True
    ):
        stacked = jnp.concatenate([blocks[index] for index in matches], axis=-2)
        mixed.append(contract("wf,fd->wd", maps[active], stacked))
    message = target.join(mixed)
    if interaction.density is None:
        return message
    return message / (aggregate["density"] + 1.0)


def _readout_energy(
    prepared: PreparedMACELayer, state: Array, species: Array, head: int, /
) -> Array:
    layer = prepared.layer
    readout = layer.readout
    if readout is None:
        return jnp.zeros((), dtype=state.dtype)
    if layer.readout_product is not None and prepared.readout_contraction is not None:
        contracted = prepared.readout_contraction(
            to_channels(layer.output_layout, state)[None], species[None]
        )[0]
        scalars = layer.readout_product.linear(contracted[..., 0])
    else:
        scalars = state[: layer.output_layout.blocks[0].multiplicity]
    return readout(scalars, head)


@final
class PreparedMACEPotential(AbstractPreparedAtomisticPotential, ExplicitFreeze):
    """Inference-only MACE realization bound to one source parameter revision.

    It embeds the exact potential it was prepared from and intentionally freezes
    it (every array below is FIXED); `validate` re-admits the realization,
    recomputes that potential's revision and every executed fold, merge,
    duplicate and table, and refuses a stale binding. Training uses the exact
    `MACEPotential`. ``edge_coupling`` selects an admitted accelerated coupling
    kernel for each streamed fragment's aggregation (edge weights of every
    interaction family are evaluated lane-locally before it); the shared
    streamed relation owns the tiling, seeded spill and single receiver
    epilogue either way. Joint-embedding conditioning, the embedding-readout
    energy, requested node/graph embeddings and atom-energy completeness follow
    the exact model's `evaluate_graph` contract.
    """

    model: MACEPotential
    layers: tuple[PreparedMACELayer, ...]
    species: MACEActiveSpecies
    edge_coupling: MACEAcceleratedCoupling | None
    radial: MACERadialRealization = eqx.field(static=True)
    source_revision_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        model: MACEPotential,
        layers: Sequence[PreparedMACELayer],
        species: MACEActiveSpecies,
        /,
        *,
        radial: MACERadialRealization,
        edge_coupling: MACEAcceleratedCoupling | None,
    ) -> None:
        revision = atomistic_model_revision(model)
        _admit_realization(model, radial)
        _admit_edge_coupling(edge_coupling, model)
        self.model = model
        self.layers = tuple(layers)
        self.species = species
        self.edge_coupling = edge_coupling
        self.radial = radial
        self.source_revision_id = revision.revision_id
        self.capabilities = model.capabilities
        self.requirements = model.requirements
        self.prepared_id = _prepared_identity(
            model, revision.revision_id, species, radial, self.layers, edge_coupling
        )

    @property
    def conditioning(self) -> AtomisticConditioningSpec:
        """Typed conditioning of the embedded model (empty without joint embedding)."""
        return self.model.conditioning

    @property
    def supported_observables(self) -> tuple[AtomisticObservable, ...]:
        return self.model.supported_observables

    def revision(self) -> NumericRevision:
        """The bound source revision; refuses when the embedded model drifted."""
        revision = atomistic_model_revision(self.model)
        if revision.revision_id != self.source_revision_id:
            raise StaleMACEPreparation("The prepared MACE realization is stale.")
        return revision

    def validate(self) -> None:
        """Re-run the exact model's validators and re-derive every executed payload.

        This is a host boundary over concrete arrays, run at publication and
        restoration, never inside a numerical call. The realization is
        re-admitted for the model's interaction families; duplicated layers
        must equal the embedded model exactly; folds, merged coefficients and
        tables must be finite and reproduce a fresh preparation to rounding
        under the admitted table declaration; ``prepared_id`` (which records the
        architecture declaration, joint embedding, families and folds) is
        recomputed.
        """
        self.model.validate()
        revision = self.revision()
        model = self.model
        radial = parse(self.radial, MACERadialRealization, "radial")
        _admit_realization(model, radial)
        active = self.species.active_species
        require_identical(
            self.species,
            MACEActiveSpecies(model.configuration.species, active),
            "The prepared species binding",
            StaleMACEPreparation,
        )
        require_identical(
            (self.capabilities, self.requirements),
            (model.capabilities, model.requirements),
            "The prepared capabilities and requirements",
            StaleMACEPreparation,
        )
        if len(self.layers) != model.interaction_count:
            raise ValueError("A prepared MACE declares one layer per interaction.")
        declaration = _admitted_declaration(self.layers, radial)
        indices = _active_model_indices(model, active)
        for index, prepared in enumerate(self.layers):
            _validate_layer(model, index, prepared, indices, active, declaration)
        _admit_edge_coupling(self.edge_coupling, model)
        identity = _prepared_identity(
            model,
            revision.revision_id,
            self.species,
            radial,
            self.layers,
            self.edge_coupling,
        )
        if identity != self.prepared_id:
            raise StaleMACEPreparation(
                "The prepared MACE identity does not match its payload."
            )

    @checked
    def evaluate_graph(
        self,
        inputs: AtomisticGraphInputs,
        request: AtomisticObservableRequest | None = None,
        /,
    ) -> AtomisticModelOutputs:
        """Case energies, per-atom energies and requested embeddings, prepared.

        Requests, the joint-embedding conditioning (missing or differently
        declared fields refuse before execution), the embedding-readout atom
        energy and the node/graph embedding observables follow the exact
        model's `evaluate_graph`.
        """
        model = self.model
        inputs.require_compatible(model.scale, self.capabilities.species_kind)
        if request is not None:
            request.require_supported(
                model.supported_observables, "PreparedMACEPotential"
            )
        graph = inputs.graph
        mask = inputs.atom_mask
        species, active = self.species.indices(inputs.species_ids, mask)
        ir = graph.graph
        if ir.edge_mask is None:
            raise ValueError("MACE requires an explicit masked edge relation.")
        joint = model.joint_features(inputs.conditioning, inputs.atom_cases, mask)
        vectors = jnp.asarray(ir.edges["displacement"]).astype(model.embedding.dtype)
        edge_active = jnp.asarray(ir.edge_mask, dtype=jnp.bool_)
        relation = model.bound_relation(inputs)
        states = self._node_states(
            relation, vectors, edge_active, species, active, mask, joint
        )
        energy, atomic = model.case_energies(
            relation,
            states.atom_energy,
            inputs.atom_cases,
            inputs.case_count,
            inputs.atom_capacity,
        )
        successful = inputs.successful
        requested = () if request is None else request.observables
        return AtomisticModelOutputs(
            energy,
            atomic,
            successful,
            additional=model.feature_observables(inputs, states, requested, successful),
        )

    def energy(self, context: Any, /) -> tuple[Array, Any]:
        """Scalar energy of one prepared single-case system context."""
        if self.model.joint_embedding is not None:
            raise UnsupportedMACEPreparedRoute(
                "context-energy-conditioning",
                "A joint-embedding MACE consumes typed conditioning, which a system "
                "potential context does not carry; evaluate it through "
                "evaluate_graph with conditioned AtomisticGraphInputs.",
            )
        system = context.system
        graph = context.graph
        if graph is None:
            raise ValueError("Prepared MACE requires a directed graph context.")
        outputs = self.evaluate_graph(
            AtomisticGraphInputs(
                graph,
                system.plan.units.scale,
                species_kind=self.capabilities.species_kind,
            )
        )
        return outputs.energy[0], (outputs.atom_energy[0], outputs.successful[0])

    def _layer_sources(
        self, index: int, features: Array, species: Array, active: Array, mask: Array, /
    ) -> MACESourceRows:
        prepared = self.layers[index]
        rows = prepared.source_rows
        if rows is None:
            return prepared.layer.interaction.source_rows(features, species, mask)
        zero = jnp.zeros((), dtype=rows.dtype)
        context = prepared.source_context_rows
        return MACESourceRows(
            jnp.where(mask[:, None], rows[active], zero),
            species,
            None if context is None else jnp.where(mask[:, None], context[active], zero),
        )

    def _layer_receivers(
        self, index: int, features: Array, species: Array, active: Array, mask: Array, /
    ) -> _PreparedReceiverRows:
        prepared = self.layers[index]
        rows = prepared.receiver_context_rows
        context = (
            prepared.layer.interaction.receiver_context(features)
            if rows is None
            else rows[active]
        )
        return _PreparedReceiverRows(features, species, active, mask, context)

    def _node_states(
        self,
        relation: PreparedStreamedRelation,
        vectors: Array,
        edge_active: Array,
        species: Array,
        active: Array,
        mask: Array,
        joint: Array | None,
        /,
    ) -> MACENodeStates:
        model = self.model
        dtype = model.embedding.dtype
        head = model.energy_reference.head
        initial = MACELayerContext(joint, None)
        features = model.init_features(
            species, mask, initial, accumulation=relation.plan.accumulation
        )
        context = (
            initial
            if model.embedding_readout is None
            else MACELayerContext(joint, model.embedding_readout(features, head))
        )
        energies = []
        states = []
        successful = jnp.asarray(True)
        for index, prepared in enumerate(self.layers):
            streamed = PreparedMACEStreamedLayer(
                prepared,
                model.geometry,
                head,
                index == 0,
                coupling=self.edge_coupling,
            )
            message, output = streamed.payload_shapes(np.dtype(dtype))
            payload = StreamedPayloadSpec(message, output)
            sources = self._layer_sources(index, features, species, active, mask)
            receivers = self._layer_receivers(index, features, species, active, mask)
            if self.edge_coupling is None:
                result = relation.evaluate(
                    payload,
                    PreparedMACEStreamedLayer.edge,
                    PreparedMACEStreamedLayer.epilogue,
                    streamed,
                    sources,
                    receivers,
                    vectors,
                    edge_active=edge_active,
                )
            else:
                result = relation.evaluate_fragments(
                    payload,
                    PreparedMACEStreamedLayer.fragment,
                    PreparedMACEStreamedLayer.epilogue,
                    streamed,
                    sources,
                    receivers,
                    vectors,
                    edge_active=edge_active,
                )
            features = result.receiver_outputs["features"]
            states.append(features)
            energies.append(result.receiver_outputs["energy"])
            successful = successful & result.evidence.successful
        atom_energy = model.atom_energies(
            species, mask, jnp.stack(energies), features, context
        )
        return MACENodeStates(atom_energy, successful, tuple(states))


def _prepared_identity(
    model: MACEPotential,
    revision_id: str,
    species: MACEActiveSpecies,
    radial: MACERadialRealization,
    layers: Sequence[PreparedMACELayer],
    edge_coupling: MACEAcceleratedCoupling | None,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "prepared-mace-potential",
            "architecture": model.architecture_id,
            "source_revision": revision_id,
            "active_species": list(species.active_species),
            "radial": radial,
            "joint_embedding": model.joint_embedding is not None,
            "embedding_readout": model.embedding_readout is not None,
            "layers": [
                {
                    "family": interaction_family(layer.layer.interaction.kind),
                    "folds": [
                        name
                        for name, value in zip(
                            _LayerFolds._fields, _prepared_folds(layer), strict=True
                        )
                        if value is not None
                    ],
                    "tables": [
                        None
                        if layer.radial_table is None
                        else layer.radial_table.table_id,
                        None
                        if layer.density_table is None
                        else layer.density_table.table_id,
                    ],
                    "contraction": layer.contraction.binding_id,
                }
                for layer in layers
            ],
            "edge_coupling": None
            if edge_coupling is None
            else edge_coupling.target.target_id,
        }
    )


def _admitted_declaration(
    layers: Sequence[PreparedMACELayer], radial: MACERadialRealization, /
) -> RadialTableDeclaration | None:
    """Re-admit the one table declaration a tabulated preparation was made under."""
    match radial:
        case "exact":
            return None
        case "tabulated":
            table = layers[0].radial_table if layers else None
            if table is None:
                raise ValueError("A tabulated prepared realization needs radial tables.")
            declaration = table.declaration
            return RadialTableDeclaration(
                declaration.grid_min, declaration.node_count, layout=declaration.layout
            )
        case unreachable:
            assert_never(unreachable)


def _validate_layer(
    model: MACEPotential,
    index: int,
    prepared: PreparedMACELayer,
    model_indices: np.ndarray,
    active: Sequence[int],
    declaration: RadialTableDeclaration | None,
    /,
) -> None:
    """Compare every executed field of one prepared layer with its preparation."""
    layer = model.layers[index]
    require_identical(
        prepared.layer, layer, f"Prepared layer {index}", StaleMACEPreparation
    )
    prepared.contraction.validate(layer.product.contraction)
    match (layer.readout_product, prepared.readout_contraction):
        case (None, None):
            pass
        case (product, MergedSymmetricContraction() as merged) if product is not None:
            merged.validate(product.contraction)
        case _:
            raise StaleMACEPreparation(
                f"Prepared layer {index} readout contraction does not match the model."
            )
    require_realized(
        _prepared_folds(prepared),
        _layer_folds(model, index, model_indices),
        f"Prepared layer {index} source, context, residual and message folds",
        StaleMACEPreparation,
    )
    if declaration is None:
        if prepared.radial_table is not None or prepared.density_table is not None:
            raise ValueError("An exact prepared realization carries no radial tables.")
        return
    interaction = _standard_interaction(layer, index)
    binding = RadialSpeciesBinding(
        model.configuration.species,
        _table_rows(model, active),
        pair_dependent=model.geometry.radial.pair_dependent,
    )
    _validate_table(
        prepared.radial_table, model, interaction.radial, declaration, binding, "radial"
    )
    if interaction.density is None:
        if prepared.density_table is not None:
            raise ValueError(
                "A layer without density normalization carries no density table."
            )
        return
    _validate_table(
        prepared.density_table,
        model,
        interaction.density,
        _density_declaration(declaration),
        binding,
        "density",
    )


def _validate_table(
    table: PreparedRadialTables | None,
    model: MACEPotential,
    mlp: RadialMLP,
    declaration: RadialTableDeclaration,
    binding: RadialSpeciesBinding,
    name: str,
    /,
) -> None:
    if not isinstance(table, PreparedRadialTables):
        raise ValueError(f"A tabulated prepared layer needs its {name} table.")
    table.validate(model.geometry.radial, mlp)
    if (
        table.declaration.declaration_id != declaration.declaration_id
        or table.binding.binding_id != binding.binding_id
    ):
        raise StaleMACEPreparation(
            f"The prepared {name} table is not the admitted declaration and species binding."
        )


def prepare_mace_potential(
    potential: MACEPotential,
    /,
    *,
    active_species: Sequence[int] | None = None,
    radial: MACERadialRealization = "exact",
    tables: RadialTableDeclaration | None = None,
    edge_coupling: MACEAcceleratedCoupling | None = None,
) -> PreparedMACEPotential:
    """Bind one exact MACE potential at its current revision for frozen inference.

    This is a host boundary over concrete parameters. ``active_species`` (raw
    identifiers, default every model species) bounds folded and tabulated
    species data; particles of other species refuse at evaluation. Every
    interaction family admits the exact radial realization (with or without
    ``edge_coupling``); ``radial="tabulated"`` admits standard interactions
    only and otherwise raises `UnsupportedMACEPreparedRoute` before preparing
    anything. A `MACELESPotential` has no prepared realization.
    """
    if isinstance(potential, MACELESPotential):
        raise UnsupportedMACEPreparedRoute(
            "latent-ewald-composition",
            "MACELESPotential has no prepared realization: its latent-charge "
            "readouts and latent Ewald energy are part of the model, and preparing "
            "only the embedded MACEPotential would drop them. Evaluate the exact "
            "MACELESPotential.",
        )
    if not isinstance(potential, MACEPotential):
        raise TypeError("potential must be a MACEPotential.")
    if edge_coupling is not None and not isinstance(
        edge_coupling, MACEAcceleratedCoupling
    ):
        raise TypeError("edge_coupling must be a MACEAcceleratedCoupling or None.")
    match radial:
        case "exact":
            if tables is not None:
                raise ValueError("Exact radial realization takes no table declaration.")
        case "tabulated":
            if not isinstance(tables, RadialTableDeclaration):
                raise TypeError(
                    "Tabulated radial realization needs a RadialTableDeclaration."
                )
        case unreachable:
            assert_never(unreachable)
    potential.validate()
    _admit_realization(potential, radial)
    species = potential.configuration.species
    active = tuple(species if active_species is None else active_species)
    binding = MACEActiveSpecies(species, active)
    indices = _active_model_indices(potential, active)
    layers = tuple(
        _prepare_layer(potential, index, indices, radial, tables, active)
        for index in range(potential.interaction_count)
    )
    return PreparedMACEPotential(
        potential,
        layers,
        binding,
        radial=radial,
        edge_coupling=edge_coupling,
    )


def _admit_edge_coupling(
    coupling: MACEAcceleratedCoupling | None, model: MACEPotential, /
) -> None:
    if coupling is None:
        return
    for layer in model.layers:
        admit_mace_coupling(coupling, layer, model.embedding.dtype)


__all__ = [
    "MACEActiveSpecies",
    "MACEPreparedRouteRefusal",
    "MACERadialRealization",
    "PreparedMACELayer",
    "PreparedMACEPotential",
    "PreparedMACEStreamedLayer",
    "prepare_mace_potential",
    "StaleMACEPreparation",
    "UnsupportedMACEPreparedRoute",
]

register_artifact_value(
    "phydrax.nn.atomistic:PreparedMACEPotential", PreparedMACEPotential
)
for _artifact in (
    MACEActiveSpecies,
    PreparedMACELayer,
    PreparedMACEStreamedLayer,
    _PreparedReceiverRows,
):
    register_artifact_value(
        f"phydrax.nn.atomistic.internal:{_artifact.__name__}", _artifact
    )
del _artifact
