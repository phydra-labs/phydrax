#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from phydrax.ein import contract

from ..._differentiation import (
    DerivativeContract,
    DerivativeRegularity,
    DerivativeRoute,
    DerivativeSurface,
    GradientLevel,
    SurfaceDerivative,
)
from ..._doc import DOC_KEY0
from ..._fingerprint import canonical_fingerprint
from ..._model import register_artifact_value
from ..._model._fixed_binding import require_identical
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float, positive_integer
from ...atomistic._conditioning import (
    AtomisticModelOutputs,
    AtomisticObservable,
    AtomisticObservableRequest,
)
from ...atomistic._graph import (
    AtomisticGraphExecutionPlan,
    case_energy_totals,
)
from ...atomistic._hessian import (
    atomistic_edge_distance_locus,
    ATOMISTIC_NONCOINCIDENT_SUPPORT,
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
    AtomisticSpeciesKind,
)
from ...sparse._streamed import PreparedStreamedRelation, StreamedPayloadSpec
from ...typing import checked, PRNGKey
from ..layers import Linear
from ..parameters import IdentityTransform
from ._constructed_admission import require_atomistic_contracts, require_constructed


# The exact atomwise-update norm ``|V v|`` is a cone wherever a gated vector
# channel vanishes (for example by inversion symmetry of an atom environment).
# The locus is model-internal: no graph evidence certifies a point off it.
_VECTOR_NORM_ZERO_LOCUS = "atomistic-nonsmooth-locus:painn-vector-norm-zero"
_METHOD_ID = "negative-position-gradient-of-total-painn-energy"


class _PaiNNConfiguration(StrictModule, NonTrainableState):
    radial_frequencies: Array
    cutoff: float = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    interaction_count: int = eqx.field(static=True)
    radial_basis_count: int = eqx.field(static=True)
    maximum_species_id: int = eqx.field(static=True)
    species_kind: AtomisticSpeciesKind = eqx.field(static=True)

    def __init__(
        self,
        *,
        cutoff: float,
        feature_count: int,
        interaction_count: int,
        radial_basis_count: int,
        maximum_species_id: int,
        species_kind: AtomisticSpeciesKind,
        dtype: np.dtype[Any],
    ) -> None:
        cutoff_value = positive_finite_float(cutoff, "cutoff")
        features = positive_integer(feature_count, "feature_count")
        interactions = positive_integer(interaction_count, "interaction_count")
        radial_count = positive_integer(radial_basis_count, "radial_basis_count")
        maximum_z = positive_integer(maximum_species_id, "maximum_species_id")
        if not isinstance(species_kind, AtomisticSpeciesKind):
            raise TypeError("species_kind must be AtomisticSpeciesKind.")
        # Fixed sinc frequencies 1..R of the radial basis, in the compute dtype.
        self.radial_frequencies = jnp.arange(1, radial_count + 1, dtype=dtype)
        self.cutoff = cutoff_value
        self.feature_count = features
        self.interaction_count = interactions
        self.radial_basis_count = radial_count
        self.maximum_species_id = maximum_z
        self.species_kind = species_kind


def _architecture_id(
    scale: AtomisticScaleContract,
    precision: AtomisticPrecisionPolicy,
    configuration: _PaiNNConfiguration,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "painn-architecture",
            "scale": scale.scale_id,
            "precision": precision.policy_id,
            "cutoff": configuration.cutoff,
            "feature_count": configuration.feature_count,
            "interaction_count": configuration.interaction_count,
            "radial_basis_count": configuration.radial_basis_count,
            "maximum_species_id": configuration.maximum_species_id,
            "species_kind": configuration.species_kind.value,
        }
    )


def _readout_hidden(feature_count: int, key: PRNGKey, /) -> Linear:
    return Linear(
        in_size=feature_count,
        out_size=feature_count,
        activation=jax.nn.silu,
        rwf=False,
        weight_transform=IdentityTransform(),
        key=key,
    )


def _readout_energy(feature_count: int, key: PRNGKey, /) -> Linear:
    return Linear(
        in_size=feature_count,
        out_size="scalar",
        rwf=False,
        weight_transform=IdentityTransform(),
        key=key,
    )


class _PaiNNInteraction(StrictModule):
    filter_in: Linear
    filter_out: Linear
    message_in: Linear
    message_out: Linear
    vector_u: Linear
    vector_v: Linear
    update_in: Linear
    update_out: Linear

    def __init__(self, feature_count: int, radial_basis_count: int, key: PRNGKey) -> None:
        keys = jr.split(key, 8)
        identity = IdentityTransform()
        self.filter_in = Linear(
            in_size=radial_basis_count,
            out_size=feature_count,
            activation=jax.nn.silu,
            rwf=False,
            weight_transform=identity,
            key=keys[0],
        )
        self.filter_out = Linear(
            in_size=feature_count,
            out_size=3 * feature_count,
            rwf=False,
            weight_transform=identity,
            key=keys[1],
        )
        self.message_in = Linear(
            in_size=feature_count,
            out_size=feature_count,
            activation=jax.nn.silu,
            rwf=False,
            weight_transform=identity,
            key=keys[2],
        )
        self.message_out = Linear(
            in_size=feature_count,
            out_size=3 * feature_count,
            rwf=False,
            weight_transform=identity,
            key=keys[3],
        )
        self.vector_u = Linear(
            in_size=feature_count,
            out_size=feature_count,
            rwf=False,
            use_bias=False,
            weight_transform=identity,
            key=keys[4],
        )
        self.vector_v = Linear(
            in_size=feature_count,
            out_size=feature_count,
            rwf=False,
            use_bias=False,
            weight_transform=identity,
            key=keys[5],
        )
        self.update_in = Linear(
            in_size=2 * feature_count,
            out_size=feature_count,
            activation=jax.nn.silu,
            rwf=False,
            weight_transform=identity,
            key=keys[6],
        )
        self.update_out = Linear(
            in_size=feature_count,
            out_size=3 * feature_count,
            rwf=False,
            weight_transform=identity,
            key=keys[7],
        )

    def atomwise_update(self, scalar: Array, vector: Array, /) -> tuple[Array, Array]:
        """Apply the norm-conditioned canonical three-head PaiNN atomwise update."""

        vector_u = self.vector_u(vector)
        vector_v = self.vector_v(vector)
        squared_norm = contract("ndf,ndf->nf", vector_v, vector_v)
        tiny = jnp.asarray(jnp.finfo(vector.dtype).tiny, dtype=vector.dtype)
        vector_norm = jnp.where(
            squared_norm > 0.0,
            jnp.sqrt(jnp.maximum(squared_norm, tiny)),
            0.0,
        )
        invariant = contract("ndf,ndf->nf", vector_u, vector_v)
        update = self.update_out(
            self.update_in(jnp.concatenate((scalar, vector_norm), axis=-1))
        )
        scalar_scalar, scalar_vector, vector_vector = jnp.split(update, 3, axis=-1)
        return (
            scalar + scalar_scalar + scalar_vector * invariant,
            vector + vector_vector[:, None, :] * vector_u,
        )

    def __call__(
        self,
        potential: PaiNNPotential,
        scalar: Array,
        vector: Array,
        edges: PreparedStreamedRelation,
        edge_data: tuple[Array, Array],
        edge_active: Array,
        /,
    ) -> tuple[Array, Array]:
        feature_count = scalar.shape[-1]
        node = jax.ShapeDtypeStruct((feature_count,), scalar.dtype)
        node_vector = jax.ShapeDtypeStruct((3, feature_count), vector.dtype)
        payload = StreamedPayloadSpec(
            message={"scalar": node, "vector": node_vector},
            output=(node, node_vector),
        )
        scalar, vector = edges.evaluate(
            payload,
            _painn_message,
            _painn_update,
            (potential, self),
            (scalar, vector),
            (scalar, vector),
            edge_data,
            edge_active=edge_active,
        ).receiver_outputs
        if scalar.shape[-1] != feature_count:
            raise RuntimeError("PaiNN interaction changed its scalar feature width.")
        return scalar, vector


def _painn_message(
    parameters: tuple[PaiNNPotential, _PaiNNInteraction],
    sender: tuple[Array, Array],
    receiver: tuple[Array, Array],
    edge: tuple[Array, Array],
    /,
) -> dict[str, Array]:
    """Filtered scalar and equivariant vector message of one directed edge."""
    potential, interaction = parameters
    del receiver
    sender_scalar, sender_vector = sender
    distance, direction = edge
    radial, cutoff_weight = potential._radial_basis(distance)
    filtered = interaction.filter_out(interaction.filter_in(radial)) * cutoff_weight
    message = interaction.message_out(interaction.message_in(sender_scalar)) * filtered
    scalar_coefficient, vector_coefficient, direction_coefficient = jnp.split(
        message, 3, axis=-1
    )
    return {
        "scalar": scalar_coefficient,
        "vector": vector_coefficient[None, :] * sender_vector
        + direction_coefficient[None, :] * direction[:, None],
    }


def _painn_update(
    parameters: tuple[PaiNNPotential, _PaiNNInteraction],
    receiver: tuple[Array, Array],
    aggregate: dict[str, Array],
    /,
) -> tuple[Array, Array]:
    """Residual message sum followed by the atomwise update, once per receiver."""
    _potential, interaction = parameters
    scalar, vector = receiver
    updated_scalar, updated_vector = interaction.atomwise_update(
        (scalar + aggregate["scalar"])[None], (vector + aggregate["vector"])[None]
    )
    return updated_scalar[0], updated_vector[0]


class PaiNNPotential(AbstractAtomisticPotential):
    """PaiNN scalar energy potential on finite or explicit periodic-image graphs.

    The candidate topology is fixed by the prepared graph topology: finite
    batches use the dense all-pairs layout, periodic (orthorhombic, triclinic or
    partially periodic) batches use explicit integer image routes, including
    nonzero self images. Geometry and cell vectors only change differentiable
    displacement payloads and smooth cutoff weights; no edge is truncated or
    rebuilt inside a derivative, so forces and the first strain derivative
    (stress) are exact. The cosine envelope is only C1 at the cutoff:
    coordinate second derivatives jump where an edge crosses it. Every
    interaction evaluates each directed edge message once and each receiver
    update once on the streamed schedule the graph topology prepared for its
    execution plan.
    """

    embedding: Array
    interactions: tuple[_PaiNNInteraction, ...]
    readout_hidden: Linear
    readout_energy: Linear
    configuration: _PaiNNConfiguration
    scale: AtomisticScaleContract
    precision: AtomisticPrecisionPolicy
    architecture_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: AtomisticScaleContract,
        /,
        *,
        cutoff: float,
        feature_count: int = 64,
        interaction_count: int = 3,
        radial_basis_count: int = 20,
        maximum_species_id: int = 118,
        species_kind: AtomisticSpeciesKind = AtomisticSpeciesKind.ATOMIC_NUMBER,
        precision: AtomisticPrecisionPolicy | None = None,
        key: PRNGKey = DOC_KEY0,
    ) -> None:
        if not isinstance(scale, AtomisticScaleContract):
            raise TypeError("scale must be an AtomisticScaleContract.")
        precision_ = AtomisticPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, AtomisticPrecisionPolicy):
            raise TypeError("precision must be an AtomisticPrecisionPolicy or None.")
        compute_dtype = jnp.dtype(precision_.compute_dtype)
        configuration = _PaiNNConfiguration(
            cutoff=cutoff,
            feature_count=feature_count,
            interaction_count=interaction_count,
            radial_basis_count=radial_basis_count,
            maximum_species_id=maximum_species_id,
            species_kind=species_kind,
            dtype=compute_dtype,
        )
        features = configuration.feature_count
        interactions = configuration.interaction_count
        keys = jr.split(key, interactions + 3)
        embedding = jr.normal(
            keys[0],
            (configuration.maximum_species_id + 1, features),
            dtype=compute_dtype,
        ) / jnp.sqrt(jnp.asarray(features, dtype=precision_.compute_dtype))
        interaction_modules = tuple(
            _PaiNNInteraction(features, configuration.radial_basis_count, keys[index + 1])
            for index in range(interactions)
        )
        readout_hidden = _readout_hidden(features, keys[-2])
        readout_energy = _readout_energy(features, keys[-1])

        def cast_inexact(value: Any) -> Any:
            if eqx.is_inexact_array(value):
                return value.astype(compute_dtype)
            return value

        self.embedding = embedding
        self.interactions = jax.tree_util.tree_map(cast_inexact, interaction_modules)
        self.readout_hidden = jax.tree_util.tree_map(cast_inexact, readout_hidden)
        self.readout_energy = jax.tree_util.tree_map(cast_inexact, readout_energy)
        self.configuration = configuration
        self.scale = scale
        self.precision = precision_
        self.architecture_id = _architecture_id(scale, precision_, configuration)
        self.method_id = _METHOD_ID

    def validate(self) -> None:
        """Rerun every owning check on this (possibly restored) potential.

        This is a host boundary over concrete arrays. Scale and precision must
        equal their constructor rebuild. The configuration is rebuilt by its
        own constructor from its declared fields (exact positive integer
        counts, finite positive float cutoff, species kind) and must equal what
        is stored, fixed radial frequencies ``1..R`` in the compute dtype
        included. The species embedding, every interaction and both readouts
        must be exactly what the constructor builds for that declaration
        (structure, sizes, SiLU/identity activations, identity weight
        transforms without weight factorization, dense weight and bias shapes)
        with finite values in the compute dtype. The recorded architecture and
        method identities are recomputed. Nothing is reinitialized; any
        mismatch refuses with ``ValueError`` or ``TypeError``.
        """
        require_atomistic_contracts(self.scale, self.precision, "PaiNN")
        configuration = self.configuration
        if not isinstance(configuration, _PaiNNConfiguration):
            raise TypeError("configuration must be a PaiNN configuration.")
        if type(configuration.cutoff) is not float:
            raise TypeError("The PaiNN cutoff must be a float.")
        dtype = np.dtype(self.precision.compute_dtype)
        require_identical(
            configuration,
            _PaiNNConfiguration(
                cutoff=configuration.cutoff,
                feature_count=configuration.feature_count,
                interaction_count=configuration.interaction_count,
                radial_basis_count=configuration.radial_basis_count,
                maximum_species_id=configuration.maximum_species_id,
                species_kind=configuration.species_kind,
                dtype=dtype,
            ),
            "PaiNN configuration",
            ValueError,
        )
        features = configuration.feature_count
        require_constructed(
            self.embedding,
            jax.ShapeDtypeStruct((configuration.maximum_species_id + 1, features), dtype),
            dtype,
            "PaiNN species embedding",
        )
        if len(self.interactions) != configuration.interaction_count:
            raise ValueError("PaiNN interaction count differs from its configuration.")
        interaction = eqx.filter_eval_shape(
            _PaiNNInteraction, features, configuration.radial_basis_count, DOC_KEY0
        )
        for index, layer in enumerate(self.interactions):
            require_constructed(layer, interaction, dtype, f"PaiNN interaction {index}")
        require_constructed(
            self.readout_hidden,
            eqx.filter_eval_shape(_readout_hidden, features, DOC_KEY0),
            dtype,
            "PaiNN hidden readout",
        )
        require_constructed(
            self.readout_energy,
            eqx.filter_eval_shape(_readout_energy, features, DOC_KEY0),
            dtype,
            "PaiNN energy readout",
        )
        if (
            self.architecture_id
            != _architecture_id(self.scale, self.precision, configuration)
            or self.method_id != _METHOD_ID
        ):
            raise ValueError(
                "The PaiNN potential does not match its recorded identities."
            )

    @property
    def capabilities(self) -> AtomisticPotentialCapabilities:
        # The radius-local energy consumes explicit image routes; strain enters
        # only through cell-dependent image displacements (first derivative).
        return AtomisticPotentialCapabilities(
            orthorhombic_periodic=True,
            triclinic_periodic=True,
            cell_derivative=True,
            species_kind=self.configuration.species_kind,
        )

    @property
    def derivative_contract(self) -> DerivativeContract:
        """Truthful coordinate and parameter regularity of the PaiNN energy.

        Edge directions require non-coincident positions. The cosine envelope
        is C1 at the cutoff, and the exact vector norm of the atomwise update is
        only C0 where a vector channel vanishes, so the energy is a C0 map with
        smooth pieces: forces are almost-everywhere derivatives and every
        Hessian or higher response requires an explicit almost-everywhere
        ``RegularityPolicy``. The cutoff join is certified from the bound graph;
        the vector-norm locus remains an unresolved condition.
        """
        regularity = DerivativeRegularity.piecewise_smooth(
            continuity=0,
            conditions=(
                atomistic_edge_distance_locus(self.configuration.cutoff),
                _VECTOR_NORM_ZERO_LOCUS,
            ),
            support=ATOMISTIC_NONCOINCIDENT_SUPPORT,
        )
        return DerivativeContract(
            (
                SurfaceDerivative(
                    DerivativeSurface.INPUT, GradientLevel.ALMOST_EVERYWHERE
                ),
                SurfaceDerivative(
                    DerivativeSurface.MODEL_PARAMETER, GradientLevel.ALMOST_EVERYWHERE
                ),
            ),
            route=DerivativeRoute.DIRECT,
            regularity=regularity,
        )

    @checked
    def _validate_batch(self, batch: AtomisticBatch, /) -> None:
        if batch.scale.scale_id != self.scale.scale_id:
            raise ValueError(
                "Potential and structure must share one exact scale contract."
            )
        if batch.positions.dtype != jnp.dtype(self.precision.coordinate_dtype):
            raise ValueError(
                "Batch coordinate dtype does not match the PaiNN precision contract."
            )

    def _radial_basis(self, distance: Array, /) -> tuple[Array, Array]:
        """Sinc radial basis and cosine envelope, broadcast over distance axes."""
        dtype = jnp.dtype(self.precision.compute_dtype)
        radius = jnp.asarray(distance, dtype=dtype)
        cutoff = jnp.asarray(self.configuration.cutoff, dtype=dtype)
        scaled = radius / cutoff
        frequencies = self.configuration.radial_frequencies
        basis = (jnp.pi * frequencies / cutoff) * jnp.sinc(
            scaled[..., None] * frequencies
        )
        envelope = jnp.where(
            scaled < 1.0,
            0.5 * (jnp.cos(jnp.pi * scaled) + 1.0),
            0.0,
        )
        return basis, envelope[..., None]

    @checked
    def evaluate_graph(
        self,
        inputs: AtomisticGraphInputs,
        request: AtomisticObservableRequest | None = None,
        /,
    ) -> AtomisticModelOutputs:
        inputs.require_compatible(self.scale, self.capabilities.species_kind)
        if request is not None:
            request.require_supported(
                (AtomisticObservable.ENERGY, AtomisticObservable.ATOM_ENERGY),
                "PaiNNPotential",
            )
        graph = inputs.graph
        case_count, atom_capacity = inputs.case_count, inputs.atom_capacity
        numbers = inputs.species_ids
        numbers = eqx.error_if(
            numbers,
            jnp.any(numbers > self.configuration.maximum_species_id),
            "Species ID exceeds PaiNNPotential.maximum_species_id.",
        )
        mask = inputs.atom_mask
        # One prepared candidate schedule serves every interaction; the cutoff
        # mask is runtime route activity, never a schedule change.
        edges = graph.topology.streamed
        accumulation = edges.plan.accumulation
        # Ordered policies select one validated atom row per step, so embedding
        # cotangents add in atom order rather than by unordered scatter additions.
        rows = (
            self.embedding[numbers]
            if accumulation == "fast" or not numbers.size
            else jax.lax.map(lambda number: self.embedding[number], numbers)
        )
        scalar = rows.astype(self.precision.compute_dtype)
        vector = jnp.zeros(
            (scalar.shape[0], 3, self.configuration.feature_count),
            dtype=self.precision.compute_dtype,
        )
        scalar = scalar * mask[:, None]
        ir = graph.graph
        if ir.edge_mask is None:
            raise ValueError("PaiNN requires an explicit masked edge relation.")
        edge_data = (
            jnp.asarray(ir.edges["distance"])[:, 0],
            jnp.asarray(ir.edges["direction"], dtype=scalar.dtype),
        )
        for interaction in self.interactions:
            scalar, vector = interaction(
                self, scalar, vector, edges, edge_data, ir.edge_mask
            )
            scalar = scalar * mask[:, None]
            vector = vector * mask[:, None, None]
        atom_energy = self.readout_energy(self.readout_hidden(scalar))
        atom_energy = atom_energy * mask.astype(atom_energy.dtype)
        total_energy = case_energy_totals(
            atom_energy.astype(self.precision.reduction_dtype),
            inputs.atom_cases,
            case_count,
            accumulation,
        )
        return AtomisticModelOutputs(
            total_energy.astype(self.precision.output_dtype),
            atom_energy.reshape((case_count, atom_capacity)).astype(
                self.precision.output_dtype
            ),
            inputs.successful,
        )

    def __call__(
        self,
        structure: AtomicStructure | AtomisticBatch,
        execution: AtomisticGraphExecutionPlan,
        /,
    ) -> Array:
        if isinstance(structure, AtomicStructure):
            batch = AtomisticBatch.from_structure(structure)
            return self.energy(batch, execution)[0]
        if isinstance(structure, AtomisticBatch):
            return self.energy(structure, execution)
        raise TypeError("PaiNNPotential expects AtomicStructure or AtomisticBatch.")


register_artifact_value("phydrax.nn.atomistic:PaiNNPotential", PaiNNPotential)
register_artifact_value(
    "phydrax.nn.atomistic.internal:PaiNNConfiguration", _PaiNNConfiguration
)
register_artifact_value(
    "phydrax.nn.atomistic.internal:PaiNNInteraction", _PaiNNInteraction
)


__all__ = ["PaiNNPotential"]
