#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
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
from ..._validation import nonnegative_integer, positive_finite_float, positive_integer
from ...atomistic._conditioning import (
    AtomisticModelOutputs,
    AtomisticObservable,
    AtomisticObservableRequest,
)
from ...atomistic._graph import (
    AtomisticGraphExecutionPlan,
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
from ..operator.layers import o3_gated_activation, O3TensorProduct, O3TensorProductPlan
from ..operator.representations import O3Features, O3Representation
from ..parameters import IdentityTransform
from ._constructed_admission import require_atomistic_contracts, require_constructed


_METHOD_ID = "negative-position-gradient-of-total-nequip-energy"
# Cartesian O(3) features of degrees zero through two.
_MAXIMUM_DEGREE = 2


def _hidden_representation(feature_count: int, /) -> O3Representation:
    return O3Representation(
        scalars=feature_count,
        pseudoscalars=feature_count,
        vectors=feature_count,
        pseudovectors=feature_count,
        tensors=feature_count,
        pseudotensors=feature_count,
    )


def _edge_representation() -> O3Representation:
    return O3Representation(scalars=1, vectors=1, tensors=1)


def _tensor_product_plan(
    representation: O3Representation,
    edge_representation: O3Representation,
    maximum_tensor_product_parameters: int,
    /,
) -> O3TensorProductPlan:
    return O3TensorProductPlan(
        representation,
        edge_representation,
        representation,
        maximum_parameters=maximum_tensor_product_parameters,
    )


class _NequIPConfiguration(StrictModule, NonTrainableState):
    radial_frequencies: Array
    hidden_representation: O3Representation
    edge_representation: O3Representation
    tensor_product_plan_ids: tuple[str, ...] = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    interaction_count: int = eqx.field(static=True)
    radial_basis_count: int = eqx.field(static=True)
    maximum_species_id: int = eqx.field(static=True)
    species_kind: AtomisticSpeciesKind = eqx.field(static=True)
    maximum_tensor_product_parameters: int = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        cutoff: float,
        feature_count: int,
        interaction_count: int,
        radial_basis_count: int,
        maximum_species_id: int,
        species_kind: AtomisticSpeciesKind,
        maximum_tensor_product_parameters: int,
        dtype: np.dtype[Any],
    ) -> None:
        cutoff_value = positive_finite_float(cutoff, "cutoff")
        features = positive_integer(feature_count, "feature_count")
        interactions = positive_integer(interaction_count, "interaction_count")
        radial_count = positive_integer(radial_basis_count, "radial_basis_count")
        maximum_z = positive_integer(maximum_species_id, "maximum_species_id")
        if not isinstance(species_kind, AtomisticSpeciesKind):
            raise TypeError("species_kind must be AtomisticSpeciesKind.")
        tensor_product_limit = nonnegative_integer(
            maximum_tensor_product_parameters, "maximum_tensor_product_parameters"
        )
        hidden_representation = _hidden_representation(features)
        edge_representation = _edge_representation()
        # Every interaction couples the same layouts under the same budget.
        plan = _tensor_product_plan(
            hidden_representation, edge_representation, tensor_product_limit
        )
        # Fixed sinc frequencies 1..R of the radial basis, in the compute dtype.
        self.radial_frequencies = jnp.arange(1, radial_count + 1, dtype=dtype)
        self.hidden_representation = hidden_representation
        self.edge_representation = edge_representation
        self.tensor_product_plan_ids = (plan.plan_id,) * interactions
        self.cutoff = cutoff_value
        self.feature_count = features
        self.interaction_count = interactions
        self.radial_basis_count = radial_count
        self.maximum_species_id = maximum_z
        self.species_kind = species_kind
        self.maximum_tensor_product_parameters = tensor_product_limit
        self.maximum_degree = _MAXIMUM_DEGREE


def _architecture_id(
    scale: AtomisticScaleContract,
    precision: AtomisticPrecisionPolicy,
    configuration: _NequIPConfiguration,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "nequip-architecture",
            "scope": "cartesian-degree-at-most-two",
            "scale": scale.scale_id,
            "precision": precision.policy_id,
            "cutoff": configuration.cutoff,
            "feature_count": configuration.feature_count,
            "interaction_count": configuration.interaction_count,
            "radial_basis_count": configuration.radial_basis_count,
            "maximum_species_id": configuration.maximum_species_id,
            "species_kind": configuration.species_kind.value,
            "maximum_tensor_product_parameters": (
                configuration.maximum_tensor_product_parameters
            ),
            "tensor_product_plans": configuration.tensor_product_plan_ids,
        }
    )


def _scalar_linear(
    in_size: int, out_size: int, *, activated: bool, key: PRNGKey
) -> Linear:
    return Linear(
        in_size=in_size,
        out_size=out_size,
        activation=jax.nn.silu if activated else None,
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


def _require_tensor_product(
    product: O3TensorProduct,
    plan: O3TensorProductPlan,
    dtype: np.dtype[Any],
    name: str,
    /,
) -> None:
    """Refuse an interaction coupling its owner does not rebuild exactly."""
    if not isinstance(product, O3TensorProduct):
        raise TypeError(f"{name} tensor product must be an O3TensorProduct.")
    # Every static plan field is compared, not only the recorded plan_id.
    require_identical(product.plan, plan, f"{name} tensor-product plan", ValueError)
    if (
        product.internal_weights
        or product.weight is not None
        or product.coefficient_origin != "native"
        or product.coefficient_tolerance is not None
    ):
        raise ValueError(
            f"{name} tensor product must use native couplings and radial path weights."
        )
    product.validate()
    if any(np.dtype(table.dtype) != dtype for table in product.prepared.coefficients):
        raise ValueError(f"{name} coupling tables are not in the compute dtype.")


class _SpeciesSelfConnection(StrictModule):
    representation: O3Representation
    weights: tuple[Array, ...]

    def __init__(
        self,
        representation: O3Representation,
        maximum_species_id: int,
        /,
        *,
        dtype: jnp.dtype,
        key: PRNGKey,
    ) -> None:
        counts = (
            representation.scalars,
            representation.pseudoscalars,
            representation.vectors,
            representation.pseudovectors,
            representation.tensors,
            representation.pseudotensors,
        )
        keys = jr.split(key, 6)
        weights = []
        for count, block_key in zip(counts, keys, strict=True):
            scale = 1.0 / math.sqrt(float(count))
            value = scale * jr.normal(
                block_key,
                (maximum_species_id + 1, count, count),
                dtype=dtype,
            )
            weights.append(value)
        self.representation = representation
        self.weights = tuple(weights)

    def __call__(self, values: Array, species_ids: Array, /) -> Array:
        features = self.representation.split(values)
        selected = tuple(weight[species_ids] for weight in self.weights)
        return self.representation.join(
            O3Features(
                scalars=contract("noi,ni->no", selected[0], features.scalars),
                pseudoscalars=contract("noi,ni->no", selected[1], features.pseudoscalars),
                vectors=contract("noi,nic->noc", selected[2], features.vectors),
                pseudovectors=contract(
                    "noi,nic->noc", selected[3], features.pseudovectors
                ),
                tensors=contract("noi,nicd->nocd", selected[4], features.tensors),
                pseudotensors=contract(
                    "noi,nicd->nocd", selected[5], features.pseudotensors
                ),
            )
        )


class _NequIPInteraction(StrictModule):
    tensor_product: O3TensorProduct
    radial_in: Linear
    radial_out: Linear
    self_connection: _SpeciesSelfConnection
    representation: O3Representation

    def __init__(
        self,
        representation: O3Representation,
        edge_representation: O3Representation,
        radial_basis_count: int,
        maximum_species_id: int,
        maximum_tensor_product_parameters: int,
        /,
        *,
        dtype: jnp.dtype,
        key: PRNGKey,
    ) -> None:
        plan = _tensor_product_plan(
            representation, edge_representation, maximum_tensor_product_parameters
        )
        radial_in_key, radial_out_key, self_key = jr.split(key, 3)
        self.tensor_product = O3TensorProduct(plan, internal_weights=False, dtype=dtype)
        self.radial_in = _scalar_linear(
            radial_basis_count, representation.scalars, activated=True, key=radial_in_key
        )
        self.radial_out = _scalar_linear(
            representation.scalars,
            plan.parameter_count,
            activated=False,
            key=radial_out_key,
        )
        self.self_connection = _SpeciesSelfConnection(
            representation,
            maximum_species_id,
            dtype=dtype,
            key=self_key,
        )
        self.representation = representation

    def __call__(
        self,
        potential: NequIPPotential,
        values: Array,
        species_ids: Array,
        edges: PreparedStreamedRelation,
        edge_data: tuple[Array, Array],
        edge_active: Array,
        node_mask: Array,
        /,
    ) -> Array:
        node = jax.ShapeDtypeStruct(values.shape[1:], values.dtype)
        activated = edges.evaluate(
            StreamedPayloadSpec(message=node, output=node),
            _nequip_message,
            _nequip_update,
            (potential, self),
            values,
            (values, species_ids),
            edge_data,
            edge_active=edge_active,
        ).receiver_outputs
        return activated * node_mask[:, None]


def _nequip_message(
    parameters: tuple[NequIPPotential, _NequIPInteraction],
    sender: Array,
    receiver: tuple[Array, Array],
    edge: tuple[Array, Array],
    /,
) -> Array:
    """Radially weighted tensor-product message of one directed edge."""
    potential, interaction = parameters
    del receiver
    distance, direction = edge
    radial, cutoff_envelope = potential._radial_basis(distance)
    path_weights = interaction.radial_out(interaction.radial_in(radial))
    path_weights = path_weights * cutoff_envelope
    return interaction.tensor_product(
        sender, potential._edge_features(direction[None])[0], path_weights
    )


def _nequip_update(
    parameters: tuple[NequIPPotential, _NequIPInteraction],
    receiver: tuple[Array, Array],
    aggregate: Array,
    /,
) -> Array:
    """Species self-connection plus message sum, then gated activation, per receiver."""
    _potential, interaction = parameters
    values, species_id = receiver
    connected = interaction.self_connection(values[None], species_id[None])[0] + aggregate
    return o3_gated_activation(connected[None], interaction.representation)[0]


class NequIPPotential(AbstractAtomisticPotential):
    """Low-degree NequIP scalar energy potential on finite or periodic-image graphs.

    This is a Cartesian O(3) implementation with degrees zero through two. It
    consumes the same prepared graph topology as PaiNN: finite dense all-pairs
    routes, or explicit integer periodic image routes for orthorhombic,
    triclinic and partially periodic cells. Geometry and cell vectors only
    change differentiable edge payloads and smooth weights, so forces and the
    first strain derivative (stress) are exact; the cosine envelope is only C1
    at the cutoff, so coordinate second derivatives jump where an edge crosses
    it. Every interaction evaluates each directed edge message once and each
    receiver update once on the streamed schedule the graph topology prepared
    for its execution plan.
    """

    embedding: Array
    interactions: tuple[_NequIPInteraction, ...]
    readout_hidden: Linear
    readout_energy: Linear
    configuration: _NequIPConfiguration
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
        feature_count: int = 32,
        interaction_count: int = 3,
        radial_basis_count: int = 20,
        maximum_species_id: int = 118,
        species_kind: AtomisticSpeciesKind = AtomisticSpeciesKind.ATOMIC_NUMBER,
        maximum_tensor_product_parameters: int = 10_000_000,
        precision: AtomisticPrecisionPolicy | None = None,
        key: PRNGKey = DOC_KEY0,
    ) -> None:
        if not isinstance(scale, AtomisticScaleContract):
            raise TypeError("scale must be an AtomisticScaleContract.")
        precision_ = AtomisticPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, AtomisticPrecisionPolicy):
            raise TypeError("precision must be an AtomisticPrecisionPolicy or None.")
        compute_dtype = jnp.dtype(precision_.compute_dtype)
        configuration = _NequIPConfiguration(
            cutoff=cutoff,
            feature_count=feature_count,
            interaction_count=interaction_count,
            radial_basis_count=radial_basis_count,
            maximum_species_id=maximum_species_id,
            species_kind=species_kind,
            maximum_tensor_product_parameters=maximum_tensor_product_parameters,
            dtype=compute_dtype,
        )
        features = configuration.feature_count
        interactions = configuration.interaction_count
        keys = jr.split(key, interactions + 3)
        embedding = jr.normal(
            keys[0],
            (configuration.maximum_species_id + 1, features),
            dtype=compute_dtype,
        ) / jnp.sqrt(jnp.asarray(features, dtype=compute_dtype))
        interaction_modules = tuple(
            _NequIPInteraction(
                configuration.hidden_representation,
                configuration.edge_representation,
                configuration.radial_basis_count,
                configuration.maximum_species_id,
                configuration.maximum_tensor_product_parameters,
                dtype=compute_dtype,
                key=keys[index + 1],
            )
            for index in range(interactions)
        )
        readout_hidden = _scalar_linear(features, features, activated=True, key=keys[-2])
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
        counts, finite positive float cutoff, species kind, nonnegative
        tensor-product budget) and must equal what is stored: fixed radial
        frequencies ``1..R`` in the compute dtype, the hidden and edge O(3)
        representations, the degree bound and every tensor-product plan
        identity. Every interaction must hold the hidden representation, a
        tensor product whose plan equals the owner's rebuild of the declared
        layouts and budget, externally path-weighted with native couplings in
        the compute dtype that `O3TensorProduct.validate` re-admits, and radial
        maps, species self-connection, species embedding and readouts that are
        exactly what the constructor builds for that declaration (structure,
        sizes, SiLU/identity activations, identity weight transforms without
        weight factorization, dense shapes) with finite compute-dtype values.
        The recorded architecture and method identities are recomputed.
        Nothing is reinitialized; any mismatch refuses with ``ValueError`` or
        ``TypeError``.
        """
        require_atomistic_contracts(self.scale, self.precision, "NequIP")
        configuration = self.configuration
        if not isinstance(configuration, _NequIPConfiguration):
            raise TypeError("configuration must be a NequIP configuration.")
        if type(configuration.cutoff) is not float:
            raise TypeError("The NequIP cutoff must be a float.")
        dtype = np.dtype(self.precision.compute_dtype)
        require_identical(
            configuration,
            _NequIPConfiguration(
                cutoff=configuration.cutoff,
                feature_count=configuration.feature_count,
                interaction_count=configuration.interaction_count,
                radial_basis_count=configuration.radial_basis_count,
                maximum_species_id=configuration.maximum_species_id,
                species_kind=configuration.species_kind,
                maximum_tensor_product_parameters=(
                    configuration.maximum_tensor_product_parameters
                ),
                dtype=dtype,
            ),
            "NequIP configuration",
            ValueError,
        )
        features = configuration.feature_count
        maximum_z = configuration.maximum_species_id
        representation = configuration.hidden_representation
        plan = _tensor_product_plan(
            representation,
            configuration.edge_representation,
            configuration.maximum_tensor_product_parameters,
        )
        require_constructed(
            self.embedding,
            jax.ShapeDtypeStruct((maximum_z + 1, features), dtype),
            dtype,
            "NequIP species embedding",
        )
        if len(self.interactions) != configuration.interaction_count:
            raise ValueError("NequIP interaction count differs from its configuration.")
        radial_in = eqx.filter_eval_shape(
            _scalar_linear,
            configuration.radial_basis_count,
            features,
            activated=True,
            key=DOC_KEY0,
        )
        radial_out = eqx.filter_eval_shape(
            _scalar_linear, features, plan.parameter_count, activated=False, key=DOC_KEY0
        )
        self_connection = eqx.filter_eval_shape(
            _SpeciesSelfConnection, representation, maximum_z, dtype=dtype, key=DOC_KEY0
        )
        for index, layer in enumerate(self.interactions):
            name = f"NequIP interaction {index}"
            require_identical(
                layer.representation, representation, f"{name} representation", ValueError
            )
            _require_tensor_product(layer.tensor_product, plan, dtype, name)
            require_constructed(layer.radial_in, radial_in, dtype, f"{name} radial input")
            require_constructed(
                layer.radial_out, radial_out, dtype, f"{name} radial path weights"
            )
            require_constructed(
                layer.self_connection,
                self_connection,
                dtype,
                f"{name} species self-connection",
            )
        require_constructed(
            self.readout_hidden,
            eqx.filter_eval_shape(
                _scalar_linear, features, features, activated=True, key=DOC_KEY0
            ),
            dtype,
            "NequIP hidden readout",
        )
        require_constructed(
            self.readout_energy,
            eqx.filter_eval_shape(_readout_energy, features, DOC_KEY0),
            dtype,
            "NequIP energy readout",
        )
        if (
            self.architecture_id
            != _architecture_id(self.scale, self.precision, configuration)
            or self.method_id != _METHOD_ID
        ):
            raise ValueError(
                "The NequIP potential does not match its recorded identities."
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
        """Truthful coordinate and parameter regularity of the NequIP energy.

        Edge directions require non-coincident positions. Gates use GELU, tanh
        and sigmoid of an epsilon-regularized norm, so pieces are smooth; the
        cosine envelope is C1 at the cutoff. Forces are classical, while a
        Hessian (order two) or a mixed third-order response is an
        almost-everywhere derivative that needs an explicit
        ``RegularityPolicy(allow_almost_everywhere=True)``; the cutoff join is
        certified at each point from the bound graph.
        """
        return DerivativeContract(
            (
                SurfaceDerivative(DerivativeSurface.INPUT, GradientLevel.SMOOTH),
                SurfaceDerivative(
                    DerivativeSurface.MODEL_PARAMETER, GradientLevel.SMOOTH
                ),
            ),
            route=DerivativeRoute.DIRECT,
            regularity=DerivativeRegularity.piecewise_smooth(
                continuity=1,
                conditions=(atomistic_edge_distance_locus(self.configuration.cutoff),),
                support=ATOMISTIC_NONCOINCIDENT_SUPPORT,
            ),
        )

    @checked
    def _validate_batch(self, batch: AtomisticBatch, /) -> None:
        if batch.scale.scale_id != self.scale.scale_id:
            raise ValueError(
                "Potential and structure must share one exact scale contract."
            )
        if batch.positions.dtype != jnp.dtype(self.precision.coordinate_dtype):
            raise ValueError(
                "Batch coordinate dtype does not match the NequIP precision contract."
            )

    def _radial_basis(self, distance: Array, /) -> tuple[Array, Array]:
        """Enveloped sinc radial basis and cosine envelope, broadcast over distances."""
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
        return basis * envelope[..., None], envelope

    def _edge_features(self, direction: Array, /) -> Array:
        dtype = jnp.dtype(self.precision.compute_dtype)
        unit = jnp.asarray(direction, dtype=dtype)
        identity = jnp.eye(3, dtype=dtype)
        outer = contract("ei,ej->eij", unit, unit)
        tensor = jnp.sqrt(jnp.asarray(1.5, dtype=dtype)) * (
            outer - identity[None, :, :] / 3.0
        )
        edge_count = unit.shape[0]
        empty_scalar = jnp.zeros((edge_count, 0), dtype=dtype)
        empty_vector = jnp.zeros((edge_count, 0, 3), dtype=dtype)
        empty_tensor = jnp.zeros((edge_count, 0, 3, 3), dtype=dtype)
        return self.configuration.edge_representation.join(
            O3Features(
                scalars=jnp.ones((edge_count, 1), dtype=dtype),
                pseudoscalars=empty_scalar,
                vectors=unit[:, None, :],
                pseudovectors=empty_vector,
                tensors=tensor[:, None, :, :],
                pseudotensors=empty_tensor,
            )
        )

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
                "NequIPPotential",
            )
        graph = inputs.graph
        case_count, atom_capacity = inputs.case_count, inputs.atom_capacity
        ir = graph.graph
        if ir.edge_mask is None:
            raise ValueError("NequIP requires explicit edge masks.")
        numbers = inputs.species_ids
        numbers = eqx.error_if(
            numbers,
            jnp.any(numbers > self.configuration.maximum_species_id),
            "Species ID exceeds NequIPPotential.maximum_species_id.",
        )
        node_mask = inputs.atom_mask.astype(self.precision.compute_dtype)
        node_count = numbers.shape[0]
        packed_size = self.configuration.hidden_representation.packed_size
        values = jnp.zeros((node_count, packed_size), dtype=self.precision.compute_dtype)
        scalar = self.embedding[numbers].astype(self.precision.compute_dtype)
        values = values.at[:, : self.configuration.feature_count].set(scalar)
        values = values * node_mask[:, None]
        # One prepared candidate schedule serves every interaction; the cutoff
        # mask is runtime route activity, never a schedule change.
        edges = graph.topology.streamed
        edge_data = (
            jnp.asarray(ir.edges["distance"])[:, 0],
            jnp.asarray(ir.edges["direction"]),
        )
        for interaction in self.interactions:
            values = interaction(
                self,
                values,
                numbers,
                edges,
                edge_data,
                ir.edge_mask,
                node_mask,
            )
        invariant_scalar = self.configuration.hidden_representation.split(values).scalars
        atom_energy = self.readout_energy(self.readout_hidden(invariant_scalar))
        atom_energy = atom_energy * node_mask.astype(atom_energy.dtype)
        total_energy = (
            jnp.zeros((case_count,), dtype=self.precision.reduction_dtype)
            .at[inputs.atom_cases]
            .add(atom_energy.astype(self.precision.reduction_dtype))
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
        raise TypeError("NequIPPotential expects AtomicStructure or AtomisticBatch.")


register_artifact_value("phydrax.nn.atomistic:NequIPPotential", NequIPPotential)
register_artifact_value(
    "phydrax.nn.atomistic.internal:NequIPConfiguration", _NequIPConfiguration
)
register_artifact_value(
    "phydrax.nn.atomistic.internal:NequIPInteraction", _NequIPInteraction
)
register_artifact_value(
    "phydrax.nn.atomistic.internal:NequIPSpeciesSelfConnection", _SpeciesSelfConnection
)


__all__ = ["NequIPPotential"]
