#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""System-bound frozen MACE inference with identity-bound folds and tables.

Preparation binds one exact `MACEPotential` at one parameter revision and one
active species subset, then folds only genuinely linear, species-local stages:

* first-layer sources and residuals depend on species alone, so
  ``linear_up(embedding)`` and the residual self-connection become per-species
  rows;
* a non-residual receiver message ``self_connection(linear_down(m) / avg)`` is
  linear in ``m`` with no activation, residual, head, or field between the two
  maps, so it becomes one per-species matrix per target block (density
  normalization divides afterwards, a scalar that commutes with both maps);
* symmetric products bind merged ``C = P W`` coefficients to the original
  ``W`` revision;
* optionally, radial networks are replaced by qualified smooth tables for the
  active ordered species pairs.

Every fold is FIXED data bound to the source numeric revision; the prepared
form is inference-only and refuses stale weights. Folding changes rounding
order, so the prepared realization has its own identity. Tables admit only
first coordinate derivatives; Hessians and force training use the exact model.
Restoration and publication run `PreparedMACEPotential.validate`, which
re-derives every executed payload from the embedded exact model and the
admitted table declaration; recorded identities are recomputed, never trusted.
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
from ..._strict import StrictModule
from ..._trainable import ExplicitFreeze, NonTrainableState
from ...atomistic._graph import AtomisticGraph
from ...atomistic._potential import (
    AbstractPreparedAtomisticPotential,
    atomistic_potential_revision,
    AtomisticPotentialCapabilities,
    AtomisticPotentialRequirements,
    AtomisticSpeciesKind,
)
from ...ein import contract
from ...sparse._execution import KeyGroupAccumulation
from ...sparse._streamed import (
    PreparedStreamedRelation,
    StreamedFragment,
    StreamedPayloadSpec,
)
from ...typing import Dim, Int32, parse
from ._fixed_binding import require_identical, require_realized
from ._mace import MACEPotential
from ._mace_interaction import (
    admit_mace_coupling,
    fragment_messages,
    from_channels,
    MACEEdgeGeometry,
    MACELayer,
    MACESourceRows,
    to_channels,
)
from ._mace_kernels import MACEAcceleratedCoupling
from ._radial import RadialMLP
from ._radial_projection import (
    prepare_radial_tables,
    PreparedRadialTables,
    RadialSpeciesBinding,
    RadialTableDeclaration,
)
from ._symmetric_contraction import MergedSymmetricContraction


MACERadialRealization: TypeAlias = Literal["exact", "tabulated"]


class StaleMACEPreparation(ValueError):
    """A prepared MACE realization is bound to different source parameters."""


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
    layer on purpose.
    """

    layer: MACELayer
    contraction: MergedSymmetricContraction
    readout_contraction: MergedSymmetricContraction | None
    source_rows: Array | None
    residual_rows: Array | None
    message_maps: tuple[Array, ...] | None
    radial_table: PreparedRadialTables | None
    density_table: PreparedRadialTables | None


def _active_model_indices(
    potential: MACEPotential, active: Sequence[int], /
) -> np.ndarray:
    species = potential.configuration.species
    return np.asarray([species.index(value) for value in active], dtype=np.int32)


def _fold_sources(
    potential: MACEPotential, layer: MACELayer, model_indices: np.ndarray, /
) -> tuple[Array, Array | None]:
    """First-layer species rows of ``linear_up`` and of the residual, if any."""
    species = jnp.asarray(model_indices)
    mask = jnp.ones(species.shape, dtype=jnp.bool_)
    embedded = potential.init_features(species, mask)
    sources = layer.interaction.linear_up(embedded)
    residual = layer.interaction.residual_state(embedded, species)
    return sources, residual


def _fold_message(layer: MACELayer, model_indices: np.ndarray, /) -> tuple[Array, ...]:
    """Per-species ``self_connection o linear_down / avg`` maps of a plain layer."""
    interaction = layer.interaction
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
) -> tuple[Array | None, Array | None, tuple[Array, ...] | None]:
    """Source rows, residual rows and message maps folded for one layer."""
    layer = potential.layers[index]
    source_rows = residual_rows = None
    if index == 0:
        source_rows, residual_rows = _fold_sources(potential, layer, model_indices)
    message_maps = (
        None if layer.interaction.residual else _fold_message(layer, model_indices)
    )
    return source_rows, residual_rows, message_maps


def _require_tabulable(potential: MACEPotential, /) -> None:
    if potential.configuration.cutoff_placement != "embedding":
        raise ValueError(
            "Radial tables realize the cutoff-embedded network only; this "
            "model applies its cutoff to network outputs."
        )


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
    interaction = layer.interaction
    source_rows, residual_rows, message_maps = _layer_folds(
        potential, index, model_indices
    )
    radial_table = density_table = None
    match radial:
        case "exact":
            pass
        case "tabulated":
            if declaration is None:
                raise ValueError(
                    "Tabulated radial realization needs a table declaration."
                )
            _require_tabulable(potential)
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
        source_rows=source_rows,
        residual_rows=residual_rows,
        message_maps=message_maps,
        radial_table=radial_table,
        density_table=density_table,
    )


@final
class PreparedMACEStreamedLayer(StrictModule):
    """Edge function or fragment aggregator and receiver epilogue of one prepared layer.

    Edge and receiver rows carry model species; receiver rows additionally
    carry active-subset species for the folded per-species maps. ``coupling``
    selects accelerated `fragment` aggregation.
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

    def radial(
        self, distance: Array, sender: Array, receiver: Array, /
    ) -> tuple[Array, Array | None]:
        """Coupling weights and edge density from tables or the exact networks."""
        interaction = self.prepared.layer.interaction
        table = self.prepared.radial_table
        if table is None:
            features, cutoff = self.geometry.radial_features(distance, sender, receiver)
            weights = interaction.radial(features)
            density = None
            if interaction.density is not None:
                density = interaction.density(features)[..., 0]
                if cutoff is not None:
                    density = density * cutoff
            if cutoff is not None:
                weights = weights * cutoff[..., None]
            return weights, density
        weights = table.evaluate(distance, sender, receiver)
        density_table = self.prepared.density_table
        density = (
            None
            if density_table is None
            else density_table.evaluate(distance, sender, receiver)[..., 0]
        )
        return weights, density

    def edge(
        self, source: MACESourceRows, receiver: _PreparedReceiverRows, vector: Array, /
    ) -> dict[str, Array]:
        distance = jnp.sqrt(jnp.sum(vector * vector, axis=-1))
        weights, density = self.radial(distance, source.species, receiver.species)
        harmonics = self.geometry.harmonics(vector).astype(weights.dtype)
        layer = self.prepared.layer
        message = {
            "coupling": layer.interaction.tensor_product(
                source.features, harmonics, weights
            )
        }
        if density is not None:
            message["density"] = density
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
        """Seeded aggregates of one fragment; radial and harmonic factors stay local."""
        if self.coupling is None:
            raise ValueError("Fragment aggregation needs an accelerated coupling.")
        distance = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
        sender = sources.species[fragment.lane_sources]
        receiver = receivers.species[fragment.lane_slots]
        weights, density = self.radial(distance, sender, receiver)
        pair = self.prepared.layer.pair_repulsion
        return fragment_messages(
            self.coupling,
            self.prepared.layer,
            fragment,
            seed,
            sources.features,
            self.geometry.harmonics(vectors).astype(weights.dtype),
            weights,
            density,
            None if pair is None else pair.edge_energy(distance, sender, receiver),
        )

    def epilogue(
        self, receiver: _PreparedReceiverRows, aggregate: dict[str, Array], /
    ) -> dict[str, Array]:
        prepared = self.prepared
        layer = prepared.layer
        interaction = layer.interaction
        species = receiver.species
        message = _receiver_message(prepared, aggregate, species, receiver.active)
        residual = None
        if interaction.residual:
            residual = (
                prepared.residual_rows[receiver.active]
                if prepared.residual_rows is not None
                else interaction.self_connection(receiver.features, species)
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
    """Receiver payload of one prepared layer epilogue."""

    features: Array
    species: Array
    active: Array
    mask: Array


def _receiver_message(
    prepared: PreparedMACELayer,
    aggregate: dict[str, Array],
    species: Array,
    active: Array,
    /,
) -> Array:
    interaction = prepared.layer.interaction
    coupling = aggregate["coupling"]
    if prepared.message_maps is None:
        message = interaction.linear_down(coupling)
        if interaction.density is None:
            return message / interaction.average_neighbor_count
        return message / (aggregate["density"] + 1.0)
    blocks = interaction.message_layout.split(coupling)
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
    it (every array below is FIXED); `validate` recomputes that potential's
    revision and every executed fold, merge, duplicate and table, and refuses a
    stale binding. Training uses the exact
    `MACEPotential`. ``edge_coupling`` selects an admitted accelerated coupling
    kernel for each streamed fragment's aggregation; the shared streamed
    relation owns the tiling, seeded spill and single receiver epilogue either
    way.
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
        revision = atomistic_potential_revision(model)
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

    def revision(self) -> NumericRevision:
        """The bound source revision; refuses when the embedded model drifted."""
        revision = atomistic_potential_revision(self.model)
        if revision.revision_id != self.source_revision_id:
            raise StaleMACEPreparation("The prepared MACE realization is stale.")
        return revision

    def validate(self) -> None:
        """Re-run the exact model's validators and re-derive every executed payload.

        This is a host boundary over concrete arrays, run at publication and
        restoration, never inside a numerical call. Duplicated
        layers must equal the embedded model exactly; folds, merged
        coefficients and tables must be finite and reproduce a fresh preparation
        to rounding under the admitted table declaration; ``prepared_id`` is
        recomputed.
        """
        self.model.validate()
        revision = self.revision()
        model = self.model
        radial = parse(self.radial, MACERadialRealization, "radial")
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

    def graph_energy(
        self,
        species_ids: Array,
        atom_mask: Array,
        atom_cases: Array,
        case_count: int,
        atom_capacity: int,
        graph: AtomisticGraph,
        /,
    ) -> tuple[Array, Array]:
        """Case and per-atom energies with the prepared realization."""
        model = self.model
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_).reshape((-1,))
        species, active = self.species.indices(
            jnp.asarray(species_ids).reshape((-1,)), mask
        )
        ir = graph.graph
        if ir.edge_mask is None:
            raise ValueError("MACE requires an explicit masked edge relation.")
        vectors = jnp.asarray(ir.edges["displacement"]).astype(model.embedding.dtype)
        edge_active = jnp.asarray(ir.edge_mask, dtype=jnp.bool_)
        topology = graph.topology
        relation = (
            topology.streamed
            if model.streaming is None
            else topology.prepare_streamed(model.streaming)
        )
        energies = self._streamed(relation, vectors, edge_active, species, active, mask)
        atom_energy = model.atom_energies(species, mask, energies)
        reduction = jnp.dtype(model.precision.reduction_dtype)
        total = (
            jnp.zeros((case_count,), dtype=reduction)
            .at[jnp.asarray(atom_cases).reshape((-1,))]
            .add(atom_energy)
        )
        output = jnp.dtype(model.precision.output_dtype)
        return (
            total.astype(output),
            atom_energy.reshape((case_count, atom_capacity)).astype(output),
        )

    def energy(self, context: Any, /) -> tuple[Array, Any]:
        """Scalar energy of one prepared single-case system context."""
        system = context.system
        graph = context.graph
        if graph is None:
            raise ValueError("Prepared MACE requires a directed graph context.")
        species = (
            system.plan.atomic_numbers
            if self.capabilities.species_kind is AtomisticSpeciesKind.ATOMIC_NUMBER
            else context.species
        )
        energy, atom_energy = self.graph_energy(
            species,
            system.active_mask,
            jnp.zeros((system.capacity,), dtype=jnp.int32),
            1,
            system.capacity,
            graph,
        )
        successful = graph.valid[0] & jnp.all(jnp.isfinite(energy))
        return energy[0], (atom_energy[0], successful)

    def _layer_features(
        self, index: int, features: Array, species: Array, active: Array, mask: Array, /
    ) -> MACESourceRows:
        prepared = self.layers[index]
        if prepared.source_rows is not None:
            lifted = prepared.source_rows[active]
        else:
            lifted = prepared.layer.interaction.linear_up(features)
        zero = jnp.zeros((), dtype=lifted.dtype)
        return MACESourceRows(jnp.where(mask[:, None], lifted, zero), species)

    def _streamed(
        self,
        relation: PreparedStreamedRelation,
        vectors: Array,
        edge_active: Array,
        species: Array,
        active: Array,
        mask: Array,
        /,
    ) -> Array:
        model = self.model
        dtype = model.embedding.dtype
        features = model.init_features(species, mask)
        energies = []
        for index, prepared in enumerate(self.layers):
            streamed = PreparedMACEStreamedLayer(
                prepared,
                model.geometry,
                model.energy_reference.head,
                index == 0,
                coupling=self.edge_coupling,
            )
            message, output = streamed.payload_shapes(np.dtype(dtype))
            payload = StreamedPayloadSpec(message, output)
            sources = self._layer_features(index, features, species, active, mask)
            receivers = _PreparedReceiverRows(features, species, active, mask)
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
            energies.append(result.receiver_outputs["energy"])
        return jnp.stack(energies)


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
            "tables": [
                [
                    None if layer.radial_table is None else layer.radial_table.table_id,
                    None if layer.density_table is None else layer.density_table.table_id,
                ]
                for layer in layers
            ],
            "contractions": [layer.contraction.binding_id for layer in layers],
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
    interaction = layer.interaction
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
    source_rows, residual_rows, message_maps = _layer_folds(model, index, model_indices)
    require_realized(
        (prepared.source_rows, prepared.residual_rows, prepared.message_maps),
        (source_rows, residual_rows, message_maps),
        f"Prepared layer {index} source, residual and message folds",
        StaleMACEPreparation,
    )
    if declaration is None:
        if prepared.radial_table is not None or prepared.density_table is not None:
            raise ValueError("An exact prepared realization carries no radial tables.")
        return
    _require_tabulable(model)
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
    species data; particles of other species refuse at evaluation.
    """
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
    "MACERadialRealization",
    "PreparedMACELayer",
    "PreparedMACEPotential",
    "PreparedMACEStreamedLayer",
    "prepare_mace_potential",
    "StaleMACEPreparation",
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
