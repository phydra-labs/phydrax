#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native MACELES: MACE with latent readout heads and a latent Ewald energy.

The pinned source ``MACELES`` is a ``ScaleShiftMACE`` with one latent readout
per energy readout (the same linear or nonlinear class, freshly parameterized),
read from the same layer states and selected head but never scale-shifted. The
latent channel ``q_i = sum_k latent_k(h_i^(k))`` drives a separately defined
latent Ewald energy (`LatentEwaldDefinition`, owned by
``atomistic._latent_electrostatics``): Gaussian-smeared finite pair sums or
periodic nonzero reciprocal modes with the source normalization, self policy
and branch rule. The total energy is ``E_MACE + E_latent``.

The latent energy is global. Exactly as in the source, the per-atom energies
are the scoped local MACE decomposition (``E0 + scaled interaction``) and omit
the latent term, so outputs set ``atom_energy_complete=False`` and report the
latent energy as a case-level ledger component rather than inventing atomic
shares. Latent charges are typed latent channels, never physical partial
charges. Born effective charges follow the source ``BEC`` projection,
including the latent-charge response ``dq/dr``.

The latent Ewald action is prepared once, explicitly, for one per-case atom
capacity and resource declaration, and reused by every evaluation; a graph of
another capacity is refused as a stale binding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import assert_never, final, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from ..._differentiation import DerivativeContract
from ..._doc import DOC_KEY0
from ..._fingerprint import canonical_fingerprint
from ..._model import register_artifact_value
from ..._model._fixed_binding import require_identical
from ...atomistic._conditioning import (
    AtomisticConditioning,
    AtomisticConditioningSpec,
    AtomisticConditionLevel,
    AtomisticModelOutputs,
    AtomisticObservable,
    AtomisticObservableRequest,
    AtomisticObservableValue,
)
from ...atomistic._feature_execution import AtomisticSystemStage
from ...atomistic._graph import AtomisticGraphExecutionPlan
from ...atomistic._latent_electrostatics import (
    LatentEwaldDefinition,
    LatentEwaldEvaluation,
    LatentEwaldResources,
    PreparedLatentEwald,
)
from ...atomistic._model_inputs import AtomisticGraphInputs
from ...atomistic._potential import (
    AbstractAtomisticPotential,
    AtomisticPotentialCapabilities,
)
from ...atomistic._types import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
)
from ...sparse._streamed import PreparedStreamedRelation, StreamedRelationEvidence
from ...typing import checked, PRNGKey
from ...units import ELEMENTARY_CHARGE
from ._constructed_admission import require_constructed
from ._mace import MACEArchitecture, MACELayerContext, MACENodeStates, MACEPotential
from ._mace_interaction import MACESourceRows
from ._mace_readout import (
    MACELinearReadout,
    MACENonlinearBiasReadout,
    MACENonlinearReadout,
    MACEReadout,
)


_METHOD_ID = "negative-position-gradient-of-mace-plus-latent-ewald-energy"

MACELESLatentReadout = MACELinearReadout | MACENonlinearReadout


class MACELESNodeState(NamedTuple):
    """Owned MACE node states plus the running latent charge of evaluated layers."""

    features: Array
    latent: Array


class MACELESLayerUpdate(NamedTuple):
    """One layered MACELES interaction: states, MACE layer energies, evidence."""

    features: MACELESNodeState
    energy: Array
    evidence: StreamedRelationEvidence


def _latent_readout(
    readout: MACEReadout | None, index: int, key: PRNGKey, /
) -> MACELESLatentReadout | None:
    """Fresh latent copy of one energy readout (the source ``_copy_mace_readout``)."""
    match readout:
        case None:
            return None
        case MACELinearReadout():
            channels, heads = readout.weight.shape
            return MACELinearReadout.initialize(
                channels, heads, key=key, dtype=readout.weight.dtype
            )
        case MACENonlinearReadout():
            return MACENonlinearReadout.initialize(
                readout.first.shape[0],
                readout.width,
                readout.head_count,
                activation_scale=readout.activation_scale,
                key=key,
                dtype=readout.first.dtype,
            )
        case MACENonlinearBiasReadout():
            raise ValueError(
                f"Layer {index}: the source MACELES copies only linear and nonlinear "
                "readouts; a nonlinear-bias readout has no latent counterpart."
            )
        case unreachable:
            assert_never(unreachable)


def _check_latent_readouts(
    mace: MACEPotential, readouts: Sequence[MACELESLatentReadout | None], /
) -> tuple[MACELESLatentReadout | None, ...]:
    values = tuple(readouts)
    if len(values) != mace.interaction_count:
        raise ValueError("MACELES declares one latent readout slot per interaction.")
    if mace.configuration.readout_correlation is not None:
        raise ValueError(
            "The source MACELES latent copy has no invariant readout product."
        )
    dtype = mace.embedding.dtype
    for index, (layer, latent) in enumerate(zip(mace.layers, values, strict=True)):
        energy = layer.readout
        match energy, latent:
            case None, None:
                valid = True
            case MACELinearReadout(), MACELinearReadout():
                valid = (
                    latent.weight.shape == energy.weight.shape
                    and latent.weight.dtype == dtype
                )
            case MACENonlinearReadout(), MACENonlinearReadout():
                valid = (
                    latent.first.shape == energy.first.shape
                    and latent.second.shape == energy.second.shape
                    and latent.activation_scale == energy.activation_scale
                    and latent.first.dtype == dtype
                )
            case _:
                valid = False
        if not valid:
            raise ValueError(
                f"Layer {index}: the latent readout must be a parameter copy of the "
                "energy readout's class and shape."
            )
    return values


@final
class MACELESPotential(AbstractAtomisticPotential):
    """MACE energy plus the latent Ewald energy of learned latent charges.

    ``mace`` is the complete scale-shifted MACE potential (its energy is the
    local part); ``latent_readouts`` holds one latent copy per energy readout
    slot; ``latent_ewald`` is the explicitly prepared latent Ewald action of
    one latent channel, bound to the model scale and one per-case atom
    capacity.
    """

    mace: MACEPotential
    latent_readouts: tuple[MACELESLatentReadout | None, ...]
    latent_ewald: PreparedLatentEwald
    architecture_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        mace: MACEPotential,
        latent_readouts: Sequence[MACELESLatentReadout | None],
        latent_ewald: PreparedLatentEwald,
        /,
    ) -> None:
        if not isinstance(mace, MACEPotential):
            raise TypeError("mace must be a MACEPotential.")
        if not isinstance(latent_ewald, PreparedLatentEwald):
            raise TypeError("latent_ewald must be a PreparedLatentEwald.")
        if mace.configuration.energy_scaling != "scale-shift":
            raise ValueError("The source MACELES is a scale-shifted MACE.")
        definition = latent_ewald.definition
        if definition.channels != 1:
            raise ValueError("MACELES drives one latent channel.")
        if latent_ewald.scale.scale_id != mace.scale.scale_id:
            raise ValueError(
                "The latent Ewald action must be prepared in the MACE scale contract."
            )
        readouts = _check_latent_readouts(mace, latent_readouts)
        self.mace = mace
        self.latent_readouts = readouts
        self.latent_ewald = latent_ewald
        self.architecture_id = canonical_fingerprint(
            {
                "kind": "maceles-architecture",
                "mace": mace.architecture_id,
                "latent_readouts": [
                    None if readout is None else type(readout).__name__
                    for readout in readouts
                ],
                "latent_ewald": definition.definition_id,
            }
        )
        self.method_id = _METHOD_ID

    @classmethod
    def initialize(
        cls,
        mace: MACEPotential,
        definition: LatentEwaldDefinition,
        /,
        *,
        atom_capacity: int,
        resources: LatentEwaldResources,
        key: PRNGKey = DOC_KEY0,
    ) -> MACELESPotential:
        """Fresh latent readouts copying every energy readout class of ``mace``."""
        keys = jr.split(key, mace.interaction_count)
        readouts = tuple(
            _latent_readout(layer.readout, index, layer_key)
            for index, (layer, layer_key) in enumerate(zip(mace.layers, keys))
        )
        return cls(
            mace,
            readouts,
            definition.prepare_graph(mace.scale, atom_capacity, resources=resources),
        )

    @property
    def configuration(self) -> MACEArchitecture:
        return self.mace.configuration

    @property
    def scale(self) -> AtomisticScaleContract:
        return self.mace.scale

    @property
    def precision(self) -> AtomisticPrecisionPolicy:
        return self.mace.precision

    @property
    def definition(self) -> LatentEwaldDefinition:
        return self.latent_ewald.definition

    @property
    def conditioning(self) -> AtomisticConditioningSpec:
        return self.mace.conditioning

    @property
    def capabilities(self) -> AtomisticPotentialCapabilities:
        """A global latent energy: no complete local decomposition or local deltas."""
        return AtomisticPotentialCapabilities(
            orthorhombic_periodic=True,
            triclinic_periodic=True,
            cell_derivative=True,
            local_energy=False,
            species_kind=self.mace.configuration.species_kind,
        )

    @property
    def supported_observables(self) -> tuple[AtomisticObservable, ...]:
        return (
            AtomisticObservable.ENERGY,
            AtomisticObservable.ATOM_ENERGY,
            AtomisticObservable.GLOBAL_ENERGY_COMPONENT,
            AtomisticObservable.LATENT_CHARGES,
            AtomisticObservable.BORN_CHARGES,
            AtomisticObservable.NODE_EMBEDDINGS,
            AtomisticObservable.GRAPH_EMBEDDINGS,
        )

    @property
    def derivative_contract(self) -> DerivativeContract | None:
        """The MACE contract met with MACE latent heads composed into latent Ewald.

        The latent energy is smooth in positions and charges; with a prepared
        reciprocal mode set the cell derivative holds almost everywhere (modes
        cross the declared sphere discretely).
        """
        local = self.mace.derivative_contract
        if local is None:
            return None
        latent = self.definition.derivative_contract(
            periodic=self.latent_ewald.modes is not None
        )
        return local.meet(local.compose(latent))

    def validate(self) -> None:
        """Re-run the MACE, latent Ewald and composition validators.

        This is a host boundary over concrete arrays, run at publication and
        restoration. The MACE validators run first; the prepared latent Ewald
        action replays its definition, resources, scale and reciprocal mode set
        (`PreparedLatentEwald.validate`) and its scale must duplicate the MACE
        scale. Each latent readout must be exactly what the latent copy of its
        energy readout constructs (class, static fields, shapes, compute
        dtype) with finite values; trainable values are otherwise free. The
        composition and its architecture and method identities are rebuilt.
        """
        mace = self.mace
        latent = self.latent_ewald
        if not isinstance(mace, MACEPotential):
            raise TypeError("mace must be a MACEPotential.")
        if not isinstance(latent, PreparedLatentEwald):
            raise TypeError("latent_ewald must be a PreparedLatentEwald.")
        if not isinstance(self.latent_readouts, tuple):
            raise TypeError("latent_readouts must be a tuple.")
        mace.validate()
        latent.validate()
        require_identical(
            latent.scale, mace.scale, "The MACELES latent Ewald scale", ValueError
        )
        readouts = _check_latent_readouts(mace, self.latent_readouts)
        dtype = np.dtype(mace.embedding.dtype)
        for index, (layer, readout) in enumerate(zip(mace.layers, readouts, strict=True)):
            require_constructed(
                readout,
                eqx.filter_eval_shape(_latent_readout, layer.readout, index, jr.key(0)),
                dtype,
                f"MACELES latent readout {index}",
            )
        rebuilt = MACELESPotential(mace, readouts, latent)
        if (
            rebuilt.architecture_id != self.architecture_id
            or self.method_id != _METHOD_ID
        ):
            raise ValueError("The MACELES potential does not match its identities.")

    def _validate_batch(self, batch: AtomisticBatch, /) -> None:
        self.mace._validate_batch(batch)

    def _latent_increment(
        self, index: int, state: Array, species: Array, /
    ) -> Array | None:
        """Latent readout of layer ``index`` (``None`` for an empty slot)."""
        readout = self.latent_readouts[index]
        if readout is None:
            return None
        layer = self.mace.layers[index]
        return readout(
            layer.readout_scalars(state, species), self.mace.energy_reference.head
        )

    def latent_charges(self, species: Array, states: MACENodeStates, /) -> Array:
        """Per-atom latent channel ``sum_k latent_k`` of the selected head."""
        dtype = self.mace.embedding.dtype
        total = jnp.zeros(states.atom_energy.shape, dtype=dtype)
        for index, state in enumerate(states.states):
            increment = self._latent_increment(index, state, species)
            if increment is not None:
                total = total + increment
        return total

    # Owner-region layered decomposition (`atomistic._feature_execution`): the
    # MACE layers with a running latent-charge column (owned rows only, never
    # exchanged). The model is NONLOCAL: its global payload is the masked
    # latent charge, consumed by the runtime's distributed latent Ewald term
    # of ``self.definition``; the executor refuses it without that term.

    @property
    def interaction_count(self) -> int:
        return self.mace.interaction_count

    @property
    def system_stages(self) -> tuple[AtomisticSystemStage, ...]:
        return ()

    def species_indices(self, species_ids: Array, atom_mask: Array, /) -> Array:
        return self.mace.species_indices(species_ids, atom_mask)

    def layer_context(
        self,
        conditioning: AtomisticConditioning | None,
        atom_cases: Array,
        species: Array,
        atom_mask: Array,
        /,
        *,
        key: PRNGKey | None = None,
    ) -> MACELayerContext:
        return self.mace.layer_context(
            conditioning, atom_cases, species, atom_mask, key=key
        )

    def admitted(
        self, species: Array, atom_mask: Array, context: MACELayerContext, /
    ) -> Array:
        return self.mace.admitted(species, atom_mask, context)

    def system_partials(
        self,
        stage: int,
        features: MACELESNodeState | None,
        species: Array,
        atom_mask: Array,
        context: MACELayerContext,
        /,
    ) -> None:
        del features, species, atom_mask, context
        raise ValueError(f"MACELES declares no system stage {stage}.")

    def apply_system_context(
        self,
        stage: int,
        features: MACELESNodeState | None,
        system_context: None,
        species: Array,
        atom_mask: Array,
        context: MACELayerContext,
        /,
    ) -> tuple[MACELESNodeState | None, MACELayerContext]:
        del features, system_context, species, atom_mask, context
        raise ValueError(f"MACELES declares no system stage {stage}.")

    def init_features(
        self, species: Array, atom_mask: Array, context: MACELayerContext, /
    ) -> MACELESNodeState:
        features = self.mace.init_features(species, atom_mask, context)
        return MACELESNodeState(
            features, jnp.zeros(features.shape[:1], dtype=self.mace.embedding.dtype)
        )

    def prepare_sources(
        self,
        index: int,
        features: MACELESNodeState,
        species: Array,
        atom_mask: Array,
        context: MACELayerContext,
        /,
    ) -> MACESourceRows:
        """The MACE sources only: latent charges stay on their owners."""
        return self.mace.prepare_sources(
            index, features.features, species, atom_mask, context
        )

    def update_receivers(
        self,
        index: int,
        relation: PreparedStreamedRelation,
        vectors: Array,
        sources: MACESourceRows,
        features: MACELESNodeState,
        species: Array,
        atom_mask: Array,
        context: MACELayerContext,
        /,
        *,
        edge_active: Array | None = None,
        route_identity: tuple[Array, ...] | None = None,
    ) -> MACELESLayerUpdate:
        """One MACE layer plus its latent readout of the updated states."""
        update = self.mace.update_receivers(
            index,
            relation,
            vectors,
            sources,
            features.features,
            species,
            atom_mask,
            context,
            edge_active=edge_active,
            route_identity=route_identity,
        )
        increment = self._latent_increment(index, update.features, species)
        latent = features.latent if increment is None else features.latent + increment
        return MACELESLayerUpdate(
            MACELESNodeState(update.features, latent), update.energy, update.evidence
        )

    def atom_energies(
        self,
        species: Array,
        atom_mask: Array,
        layer_energies: Array,
        features: MACELESNodeState,
        context: MACELayerContext,
        /,
    ) -> Array:
        """The scoped local MACE atom energies (the latent energy is global)."""
        return self.mace.atom_energies(
            species, atom_mask, layer_energies, features.features, context
        )

    def global_payload(
        self,
        features: MACELESNodeState,
        species: Array,
        atom_mask: Array,
        context: MACELayerContext,
        /,
    ) -> Array:
        """Masked latent charges of the owned rows (the latent Ewald payload)."""
        del species, context
        return jnp.where(
            jnp.asarray(atom_mask, dtype=jnp.bool_),
            features.latent,
            jnp.zeros((), dtype=features.latent.dtype),
        )

    def system_energy(self, context: MACELayerContext, /) -> None:
        del context
        return None

    def owner_fields(
        self,
        features: MACELESNodeState,
        species: Array,
        atom_mask: Array,
        context: MACELayerContext,
        /,
    ) -> Mapping[AtomisticObservable, Array]:
        """Latent charges are the global payload, not a domain-term field."""
        del features, species, atom_mask, context
        return {}

    def _require_binding(self, inputs: AtomisticGraphInputs, /) -> None:
        if inputs.atom_capacity != self.latent_ewald.atom_capacity:
            raise ValueError(
                "The prepared latent Ewald action is bound to atom capacity "
                f"{self.latent_ewald.atom_capacity}; prepare it again for "
                f"{inputs.atom_capacity}."
            )

    def _latent_energy(
        self, inputs: AtomisticGraphInputs, charges: Array, /
    ) -> LatentEwaldEvaluation:
        graph = inputs.graph
        return self.latent_ewald.energy(
            charges,
            inputs.positions,
            inputs.atom_mask,
            graph.cell_vectors,
            graph.topology.periodic_mask,
        )

    def _born_charges(
        self, inputs: AtomisticGraphInputs, species: Array, /
    ) -> AtomisticObservableValue:
        """Source BEC with the latent-charge response through the whole network."""
        mace = self.mace
        graph = inputs.graph
        ir = graph.graph
        if ir.senders is None or ir.receivers is None or ir.edge_mask is None:
            raise ValueError("Born charges need the bound graph's explicit edges.")
        reference = inputs.positions
        displacement = jnp.asarray(ir.edges["displacement"])
        senders = jnp.asarray(ir.senders)
        receivers = jnp.asarray(ir.receivers)
        edge_active = jnp.asarray(ir.edge_mask, dtype=jnp.bool_)
        relation = mace.bound_relation(inputs)
        mask = inputs.atom_mask
        joint = mace.joint_features(inputs.conditioning, inputs.atom_cases, mask)

        def charge_function(positions: Array, /) -> Array:
            # Edge displacements are r_receiver - r_sender (+ fixed images), so
            # moving atoms changes them by the same linear map.
            delta = positions - reference
            moved = displacement + (delta[receivers] - delta[senders]).astype(
                displacement.dtype
            )
            states = mace.node_states(
                species, mask, relation, moved, edge_active=edge_active, joint=joint
            )
            return self.latent_charges(species, states)

        born = self.latent_ewald.born_charges(
            charge_function,
            reference,
            mask,
            graph.cell_vectors,
            graph.topology.periodic_mask,
        )
        return born.value(inputs.case_count, inputs.atom_capacity)

    @checked
    def evaluate_graph(
        self,
        inputs: AtomisticGraphInputs,
        request: AtomisticObservableRequest | None = None,
        /,
    ) -> AtomisticModelOutputs:
        """Total energy, scoped atom energies and the latent ledger on one graph.

        ``energy`` is the MACE energy plus the latent Ewald energy; ``atom_energy``
        is the scoped local MACE decomposition (``atom_energy_complete=False``).
        The latent energy is always reported as a case-level global energy
        component, and its evaluation record is attached as evidence. Latent
        charges, Born charges and embeddings are returned when requested.
        """
        if request is not None:
            request.require_supported(self.supported_observables, "MACELESPotential")
        self._require_binding(inputs)
        mace = self.mace
        species, states, relation = mace.graph_node_states(inputs)
        local, atomic = mace.case_energies(
            relation,
            states.atom_energy,
            inputs.atom_cases,
            inputs.case_count,
            inputs.atom_capacity,
        )
        charges = jnp.where(
            inputs.atom_mask,
            self.latent_charges(species, states),
            jnp.zeros((), dtype=mace.embedding.dtype),
        )
        latent = self._latent_energy(inputs, charges)
        output = jnp.dtype(mace.precision.output_dtype)
        energy = local + latent.energy.astype(output)
        successful = inputs.successful
        requested = () if request is None else request.observables
        additional = [
            latent.global_energy_value(inputs.atom_capacity),
            *mace.feature_observables(inputs, states, requested, successful),
        ]
        if AtomisticObservable.LATENT_CHARGES in requested:
            additional.append(
                AtomisticObservableValue(
                    AtomisticObservable.LATENT_CHARGES,
                    self.definition.latent_port(),
                    charges.astype(output),
                    successful,
                    level=AtomisticConditionLevel.ATOM,
                    case_count=inputs.case_count,
                    atom_capacity=inputs.atom_capacity,
                    unit=ELEMENTARY_CHARGE,
                )
            )
        if AtomisticObservable.BORN_CHARGES in requested:
            additional.append(self._born_charges(inputs, species))
        return AtomisticModelOutputs(
            energy,
            atomic,
            successful & latent.successful,
            additional=tuple(additional),
            atom_energy_complete=False,
            evidence=(latent,),
        )

    def __call__(
        self,
        structure: AtomicStructure | AtomisticBatch,
        execution: AtomisticGraphExecutionPlan,
        /,
    ) -> Array:
        if isinstance(structure, AtomicStructure):
            return self.energy(AtomisticBatch.from_structure(structure), execution)[0]
        if isinstance(structure, AtomisticBatch):
            return self.energy(structure, execution)
        raise TypeError("MACELESPotential expects AtomicStructure or AtomisticBatch.")


__all__ = [
    "MACELESLatentReadout",
    "MACELESLayerUpdate",
    "MACELESNodeState",
    "MACELESPotential",
]

register_artifact_value("phydrax.nn.atomistic:MACELESPotential", MACELESPotential)
