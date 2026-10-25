#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native trainable MACE scalar-energy potential.

The architecture follows the admitted real-agnostic MACE source families
exactly: a species embedding (optionally plus a joint categorical/continuous
case/atom embedding and its linear embedding readout), statically declared
interaction layers (standard plain/residual/density kinds, attention-residual
and residual-nonlinear), symmetric product bases with their original
species-conditioned weights ``W`` and fixed coupling bases ``U``, linear,
nonlinear and nonlinear-bias invariant readouts, atomic reference energies with
optional per-head scale and shift, and optional ZBL repulsion. Radial features
use a Bessel, Gaussian or Chebyshev basis with an optional Agnesi or Soft
distance transform; harmonics and hidden irreps follow the O(3) natural-parity
or the source SO(3) all-even convention; coupling inputs may use declared edge
irreps (also in the first layer). Every layer streams over the shared prepared
relation: edges evaluate radial weights, real harmonics and the ``uvu``
coupling; each receiver then runs its complete epilogue once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import assert_never, final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike

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
from ..._model._ports import ValuePort
from ..._strict import StrictModule
from ..._trainable import NonTrainableState, partition_parameters
from ..._validation import (
    canonical_identifier,
    finite_real_scalar,
    nonnegative_integer,
    positive_finite_float,
    positive_integer,
)
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
from ...sparse import RelationAccumulation
from ...sparse._streamed import (
    PreparedStreamedRelation,
    StreamedPayloadSpec,
    StreamedRelationEvidence,
    StreamedRelationPlan,
    StreamedRelationResources,
)
from ...special import RealCartesianHarmonics
from ...typing import checked, Dim, Int32, parse, PRNGKey
from ...units import ANGSTROM, conversion_factor
from ..operator.layers import (
    O3IrrepLinear,
    O3TensorProduct,
    O3TensorProductPath,
    O3TensorProductPlan,
)
from ..operator.representations import O3IrrepLayout
from ._mace_embedding import (
    joint_embedding_identity,
    mace_embedding_conditioning,
    MACEEmbeddingFeature,
    MACEJointEmbedding,
    require_mace_conditioning,
)
from ._mace_interaction import (
    admit_mace_coupling,
    ATTENTION_PROJECTION_WIDTH,
    ATTENTION_RADIAL_WIDTHS,
    convolution_paths,
    gate_layout,
    harmonic_layout,
    interaction_family,
    interaction_traits,
    mace_irrep_layout,
    MACEAnyInteraction,
    MACEAttentionInteraction,
    MACECutoffPlacement,
    MACEEdgeGeometry,
    MACEGate,
    MACEInteraction,
    MACEInteractionKind,
    MACELayer,
    MACENonlinearInteraction,
    MACEParityConvention,
    MACEPlainLinear,
    MACEProduct,
    MACEReceiverRows,
    MACESelfConnection,
    MACESourceRows,
    MACEStreamedLayer,
    NONLINEAR_DENSITY_ALPHA,
    NONLINEAR_DENSITY_BETA,
    NONLINEAR_DENSITY_WIDTH,
)
from ._mace_kernels import (
    MACEAcceleratedCoupling,
    MACEEdgeCouplingSpec,
    MACEFragmentExtent,
    MACEFragmentResources,
)
from ._mace_readout import (
    covalent_radii_angstrom,
    MACEEnergyReference,
    MACEEnergyScaling,
    MACEInvariantReadoutProduct,
    MACELinearReadout,
    MACENonlinearBiasReadout,
    MACENonlinearReadout,
    MACEPairRepulsion,
    MACEReadout,
)
from ._radial import (
    AgnesiTransform,
    BesselRadialBasis,
    ChebyshevRadialBasis,
    DistanceTransform,
    GaussianRadialBasis,
    LayerNormRadialMLP,
    normalized_sigmoid_scale,
    normalized_silu_scale,
    PolynomialCutoff,
    RadialBasis,
    RadialEmbedding,
    RadialMLP,
    SoftTransform,
)
from ._symmetric_contraction import SymmetricContraction, SymmetricContractionPlan


MACEDistanceTransform: TypeAlias = Literal["none", "agnesi", "soft"]
MACERadialBasisKind: TypeAlias = Literal["bessel", "gaussian", "chebyshev"]
MACEReadoutVariant: TypeAlias = Literal["nonlinear", "nonlinear-bias"]

_METHOD_ID = "negative-position-gradient-of-total-mace-energy"

# ZBL envelopes join at species-pair covalent-radius sums carried as dynamic
# fixed arrays; the graph evidence cannot name that radius statically.
_ZBL_ENVELOPE_LOCUS = "atomistic-nonsmooth-locus:mace-zbl-covalent-radius-sum"
# The nonlinear interaction divides by ``beta * density + alpha`` with trainable
# scalars; the energy is singular where that denominator vanishes.
_NONLINEAR_DENOMINATOR_LOCUS = (
    "atomistic-singular-locus:mace-nonlinear-density-denominator-zero"
)

NODE_FEATURES_REPRESENTATION = "phydrax-native-real-harmonic-packed-irreps"


class MACESpeciesLookupDim(Dim):
    """Raw species identifiers ``0..maximum`` mapped to model species."""


class MACELayerUpdate(NamedTuple):
    """Updated receiver states, unscaled layer energies, streamed evidence and bounds.

    ``resources`` are the substrate's declared persistent schedule, fragment,
    receiver, spill/replay and cotangent bytes; ``kernel_resources`` the
    accelerated kernel's declared per-fragment bind bytes (``None`` on the
    portable route). Neither is a compiler measurement.
    """

    features: Array
    energy: Array
    evidence: StreamedRelationEvidence
    resources: StreamedRelationResources
    kernel_resources: MACEFragmentResources | None


class MACENodeStates(NamedTuple):
    """Per-atom energies, streamed success and every layer's output state."""

    atom_energy: Array
    successful: Array
    states: tuple[Array, ...]


class MACELayerContext(NamedTuple):
    """Owner layer context of MACE: the conditioning-dependent node terms.

    ``joint`` is the per-atom joint-embedding addend and ``embedding_energy``
    the per-atom embedding-readout energy of the layer-0 states; each is
    present exactly when the architecture declares it (both ``None`` for an
    unconditioned MACE, whose numerics are then the plain species embedding).
    """

    joint: Array | None
    embedding_energy: Array | None


@final
class MACEArchitecture(StrictModule, NonTrainableState):
    """Immutable declaration of one MACE architecture.

    Layer ``i`` reads ``C x 0e`` (``i = 0``) or the hidden irreps
    ``C x (l, p_l), l <= hidden_degree``; ``linear_up`` lifts that state onto
    the coupling input (the declared edge irreps ``E x (l, p_l), l <=
    edge_feature_degree`` for ``i > 0``; ``E x 0e`` in the first layer when
    ``edge_irreps_first``; otherwise the layer input), which couples with real
    harmonics up to ``edge_degree`` into ``C x (l, p_l), l <= edge_degree``. Its
    product emits the hidden irreps, except the last product of a
    multi-interaction model (scalars unless ``keep_last_layer_irreps``) and the
    single product of a one-interaction model (scalars unless
    ``readout_correlation``). ``parity`` ``"o3"`` uses ``p_l = (-1)^l``;
    ``"so3"`` is the source ``use_so3`` convention with every block even.

    ``readout_width`` is the per-head hidden width of the final nonlinear
    readout (``readout_variant`` ``"nonlinear"`` or ``"nonlinear-bias"``) and
    is ``None`` exactly when the only readout is linear. ``gate_scales`` are
    the ``(silu, sigmoid)`` normalization constants of the residual-nonlinear
    interaction gate. ``embedding_features`` declare a joint embedding added to
    the species embedding; ``embedding_readout`` adds its linear energy readout.
    """

    species: tuple[int, ...] = eqx.field(static=True)
    species_kind: AtomisticSpeciesKind = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    radial_basis: MACERadialBasisKind = eqx.field(static=True)
    radial_basis_count: int = eqx.field(static=True)
    cutoff_power: int = eqx.field(static=True)
    distance_transform: MACEDistanceTransform = eqx.field(static=True)
    agnesi_parameters: tuple[float, float, float] | None = eqx.field(static=True)
    soft_alpha: float | None = eqx.field(static=True)
    cutoff_placement: MACECutoffPlacement = eqx.field(static=True)
    parity: MACEParityConvention = eqx.field(static=True)
    channel_count: int = eqx.field(static=True)
    hidden_degree: int = eqx.field(static=True)
    edge_degree: int = eqx.field(static=True)
    edge_channel_count: int | None = eqx.field(static=True)
    edge_feature_degree: int | None = eqx.field(static=True)
    edge_irreps_first: bool = eqx.field(static=True)
    interactions: tuple[MACEInteractionKind, ...] = eqx.field(static=True)
    correlations: tuple[int, ...] = eqx.field(static=True)
    radial_widths: tuple[int, ...] = eqx.field(static=True)
    readout_width: int | None = eqx.field(static=True)
    readout_variant: MACEReadoutVariant = eqx.field(static=True)
    radial_activation_scale: float = eqx.field(static=True)
    readout_activation_scale: float | None = eqx.field(static=True)
    gate_scales: tuple[float, float] | None = eqx.field(static=True)
    average_neighbor_count: float = eqx.field(static=True)
    heads: tuple[str, ...] = eqx.field(static=True)
    head: str = eqx.field(static=True)
    energy_scaling: MACEEnergyScaling = eqx.field(static=True)
    pair_repulsion: bool = eqx.field(static=True)
    readout_correlation: int | None = eqx.field(static=True)
    last_readout_only: bool = eqx.field(static=True)
    keep_last_layer_irreps: bool = eqx.field(static=True)
    agnostic_product: bool = eqx.field(static=True)
    embedding_features: tuple[MACEEmbeddingFeature, ...] = eqx.field(static=True)
    embedding_readout: bool = eqx.field(static=True)
    architecture_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        species: Sequence[int],
        cutoff: float,
        radial_basis_count: int,
        cutoff_power: int,
        channel_count: int,
        hidden_degree: int,
        edge_degree: int,
        interactions: Sequence[MACEInteractionKind],
        correlations: Sequence[int],
        radial_widths: Sequence[int],
        average_neighbor_count: float,
        readout_width: int | None = None,
        species_kind: AtomisticSpeciesKind = AtomisticSpeciesKind.ATOMIC_NUMBER,
        radial_basis: MACERadialBasisKind = "bessel",
        distance_transform: MACEDistanceTransform = "none",
        agnesi_parameters: tuple[float, float, float] | None = None,
        soft_alpha: float | None = None,
        cutoff_placement: MACECutoffPlacement = "embedding",
        parity: MACEParityConvention = "o3",
        edge_channel_count: int | None = None,
        edge_feature_degree: int | None = None,
        edge_irreps_first: bool = False,
        readout_variant: MACEReadoutVariant = "nonlinear",
        radial_activation_scale: float | None = None,
        readout_activation_scale: float | None = None,
        gate_scales: tuple[float, float] | None = None,
        heads: Sequence[str] = ("Default",),
        head: str | None = None,
        energy_scaling: MACEEnergyScaling = "unscaled",
        pair_repulsion: bool = False,
        readout_correlation: int | None = None,
        last_readout_only: bool = False,
        keep_last_layer_irreps: bool = False,
        agnostic_product: bool = False,
        embedding_features: Sequence[MACEEmbeddingFeature] = (),
        embedding_readout: bool = False,
    ) -> None:
        fields = _architecture_fields(
            species=species,
            species_kind=species_kind,
            cutoff=cutoff,
            radial_basis=radial_basis,
            radial_basis_count=radial_basis_count,
            cutoff_power=cutoff_power,
            distance_transform=distance_transform,
            agnesi_parameters=agnesi_parameters,
            soft_alpha=soft_alpha,
            cutoff_placement=cutoff_placement,
            parity=parity,
            channel_count=channel_count,
            hidden_degree=hidden_degree,
            edge_degree=edge_degree,
            edge_channel_count=edge_channel_count,
            edge_feature_degree=edge_feature_degree,
            edge_irreps_first=edge_irreps_first,
            interactions=interactions,
            correlations=correlations,
            radial_widths=radial_widths,
            readout_width=readout_width,
            readout_variant=readout_variant,
            radial_activation_scale=radial_activation_scale,
            readout_activation_scale=readout_activation_scale,
            gate_scales=gate_scales,
            average_neighbor_count=average_neighbor_count,
            heads=heads,
            head=head,
            energy_scaling=energy_scaling,
            pair_repulsion=pair_repulsion,
            readout_correlation=readout_correlation,
            last_readout_only=last_readout_only,
            keep_last_layer_irreps=keep_last_layer_irreps,
            agnostic_product=agnostic_product,
            embedding_features=embedding_features,
            embedding_readout=embedding_readout,
        )
        for name, value in fields._asdict().items():
            setattr(self, name, value)
        self.architecture_id = _architecture_id(fields)

    def _fields(self) -> _ArchitectureFields:
        return _architecture_fields(
            **{name: getattr(self, name) for name in _ArchitectureFields._fields}
        )

    def validate(self) -> None:
        """Re-run every declaration check, e.g. after constructor-bypassing restore."""
        if _architecture_id(self._fields()) != self.architecture_id:
            raise ValueError("MACEArchitecture fields do not match their identity.")

    @property
    def interaction_count(self) -> int:
        return len(self.interactions)

    @property
    def species_count(self) -> int:
        return len(self.species)

    @property
    def contraction_species_count(self) -> int:
        return 1 if self.agnostic_product else len(self.species)

    @property
    def conditioning(self) -> AtomisticConditioningSpec:
        """Typed conditioning inputs consumed by the joint embedding (may be empty)."""
        return mace_embedding_conditioning(self.embedding_features)

    def layout(self, multiplicity: int, degree: int, /) -> O3IrrepLayout:
        """``multiplicity x (l, p_l)`` for ``l <= degree`` under the parity convention."""
        return mace_irrep_layout(multiplicity, degree, parity=self.parity)

    def input_layout(self, index: int, /) -> O3IrrepLayout:
        degree = 0 if index == 0 else self.hidden_degree
        return self.layout(self.channel_count, degree)

    def output_degree(self, index: int, /) -> int:
        last = index == self.interaction_count - 1
        if not last:
            return self.hidden_degree
        if self.interaction_count == 1:
            return 0 if self.readout_correlation is None else self.hidden_degree
        return self.hidden_degree if self.keep_last_layer_irreps else 0

    def output_layout(self, index: int, /) -> O3IrrepLayout:
        return self.layout(self.channel_count, self.output_degree(index))

    def edge_layout(self, index: int, /) -> O3IrrepLayout:
        """Coupling input (``linear_up`` output) of one layer."""
        edges = self.edge_channel_count
        if index == 0:
            if edges is not None and self.edge_irreps_first:
                return self.layout(edges, 0)
            return self.input_layout(0)
        if edges is None or self.edge_feature_degree is None:
            return self.input_layout(index)
        return self.layout(edges, self.edge_feature_degree)

    def target_layout(self) -> O3IrrepLayout:
        return self.layout(self.channel_count, self.edge_degree)

    def harmonics_layout(self) -> O3IrrepLayout:
        return harmonic_layout(self.edge_degree, parity=self.parity)

    def message_payload_width(self) -> int:
        """Largest per-event message or per-receiver output element count."""
        widths = [1 + self.output_layout(0).packed_size]
        for index in range(self.interaction_count):
            message, _ = convolution_paths(
                self.edge_layout(index), self.harmonics_layout(), self.target_layout()
            )
            widths.append(message.packed_size + 2)
            widths.append(self.output_layout(index).packed_size + 1)
        return max(widths)

    def readout_kind(self, index: int, /) -> Literal["linear", "nonlinear", "none"]:
        """Readout of one layer in source order."""
        last = index == self.interaction_count - 1
        if self.interaction_count == 1:
            return "linear" if self.readout_correlation is None else "nonlinear"
        if last:
            return "nonlinear"
        return "none" if self.last_readout_only else "linear"


class _ArchitectureFields(NamedTuple):
    """Canonical validated declaration, in identity order."""

    species: tuple[int, ...]
    species_kind: AtomisticSpeciesKind
    cutoff: float
    radial_basis: MACERadialBasisKind
    radial_basis_count: int
    cutoff_power: int
    distance_transform: MACEDistanceTransform
    agnesi_parameters: tuple[float, float, float] | None
    soft_alpha: float | None
    cutoff_placement: MACECutoffPlacement
    parity: MACEParityConvention
    channel_count: int
    hidden_degree: int
    edge_degree: int
    edge_channel_count: int | None
    edge_feature_degree: int | None
    edge_irreps_first: bool
    interactions: tuple[MACEInteractionKind, ...]
    correlations: tuple[int, ...]
    radial_widths: tuple[int, ...]
    readout_width: int | None
    readout_variant: MACEReadoutVariant
    radial_activation_scale: float
    readout_activation_scale: float | None
    gate_scales: tuple[float, float] | None
    average_neighbor_count: float
    heads: tuple[str, ...]
    head: str
    energy_scaling: MACEEnergyScaling
    pair_repulsion: bool
    readout_correlation: int | None
    last_readout_only: bool
    keep_last_layer_irreps: bool
    agnostic_product: bool
    embedding_features: tuple[MACEEmbeddingFeature, ...]
    embedding_readout: bool


def _architecture_fields(
    *,
    species: Sequence[int],
    species_kind: AtomisticSpeciesKind,
    cutoff: float,
    radial_basis: MACERadialBasisKind,
    radial_basis_count: int,
    cutoff_power: int,
    distance_transform: MACEDistanceTransform,
    agnesi_parameters: tuple[float, float, float] | None,
    soft_alpha: float | None,
    cutoff_placement: MACECutoffPlacement,
    parity: MACEParityConvention,
    channel_count: int,
    hidden_degree: int,
    edge_degree: int,
    edge_channel_count: int | None,
    edge_feature_degree: int | None,
    edge_irreps_first: bool,
    interactions: Sequence[MACEInteractionKind],
    correlations: Sequence[int],
    radial_widths: Sequence[int],
    readout_width: int | None,
    readout_variant: MACEReadoutVariant,
    radial_activation_scale: float | None,
    readout_activation_scale: float | None,
    gate_scales: tuple[float, float] | None,
    average_neighbor_count: float,
    heads: Sequence[str],
    head: str | None,
    energy_scaling: MACEEnergyScaling,
    pair_repulsion: bool,
    readout_correlation: int | None,
    last_readout_only: bool,
    keep_last_layer_irreps: bool,
    agnostic_product: bool,
    embedding_features: Sequence[MACEEmbeddingFeature],
    embedding_readout: bool,
) -> _ArchitectureFields:
    """Validate and canonicalize one architecture declaration."""
    if not isinstance(species_kind, AtomisticSpeciesKind):
        raise TypeError("species_kind must be AtomisticSpeciesKind.")
    species_ = _species(species, species_kind)
    basis = parse(radial_basis, MACERadialBasisKind, "radial_basis")
    transform = parse(distance_transform, MACEDistanceTransform, "distance_transform")
    placement = parse(cutoff_placement, MACECutoffPlacement, "cutoff_placement")
    parity_ = parse(parity, MACEParityConvention, "parity")
    scaling = parse(energy_scaling, MACEEnergyScaling, "energy_scaling")
    variant = parse(readout_variant, MACEReadoutVariant, "readout_variant")
    kinds = tuple(
        parse(kind, MACEInteractionKind, "interactions") for kind in interactions
    )
    orders = tuple(positive_integer(value, "correlations") for value in correlations)
    if not kinds or len(orders) != len(kinds):
        raise ValueError("MACE declares one correlation per interaction (at least one).")
    widths = tuple(positive_integer(value, "radial_widths") for value in radial_widths)
    hidden = nonnegative_integer(hidden_degree, "hidden_degree")
    edge = nonnegative_integer(edge_degree, "edge_degree")
    if hidden > edge:
        raise ValueError("hidden_degree cannot exceed edge_degree.")
    count = positive_integer(radial_basis_count, "radial_basis_count")
    if basis == "gaussian" and count < 2:
        raise ValueError("A Gaussian radial basis declares at least two functions.")
    agnesi, alpha = _transform_parameters(transform, agnesi_parameters, soft_alpha)
    if (transform != "none" or pair_repulsion) and (
        species_kind is not AtomisticSpeciesKind.ATOMIC_NUMBER
    ):
        raise ValueError(
            "Distance transforms and ZBL repulsion read covalent radii of atomic numbers."
        )
    edge_channels, edge_features = _edge_irreps(
        edge_channel_count, edge_feature_degree, hidden
    )
    first = _flag(edge_irreps_first, "edge_irreps_first")
    if first and edge_channels is None:
        raise ValueError("edge_irreps_first needs declared edge irreps.")
    names = tuple(canonical_identifier(value, "heads") for value in heads)
    if not names or len(set(names)) != len(names):
        raise ValueError("MACE heads must be distinct, non-empty identifiers.")
    selected = names[0] if head is None else canonical_identifier(head, "head")
    if selected not in names:
        raise ValueError(f"MACE head {selected!r} is not a declared head.")
    readout_order = (
        None
        if readout_correlation is None
        else positive_integer(readout_correlation, "readout_correlation")
    )
    nonlinear = _readout_contract(
        len(kinds), readout_order, readout_width, bool(last_readout_only)
    )
    if variant == "nonlinear-bias" and not nonlinear:
        raise ValueError("A nonlinear-bias readout variant needs a nonlinear readout.")
    keep = _flag(keep_last_layer_irreps, "keep_last_layer_irreps")
    if keep and len(kinds) == 1:
        raise ValueError(
            "keep_last_layer_irreps applies only to multi-interaction models; the "
            "single product's output is declared by readout_correlation."
        )
    radial_scale = (
        normalized_silu_scale()
        if radial_activation_scale is None
        else positive_finite_float(radial_activation_scale, "radial_activation_scale")
    )
    if nonlinear:
        readout_scale: float | None = (
            normalized_silu_scale()
            if readout_activation_scale is None
            else positive_finite_float(
                readout_activation_scale, "readout_activation_scale"
            )
        )
    elif readout_activation_scale is not None:
        raise ValueError("A model without a nonlinear readout has no readout activation.")
    else:
        readout_scale = None
    gates = _gate_scales(
        gate_scales,
        any(interaction_family(kind) == "nonlinear" for kind in kinds),
    )
    features = tuple(embedding_features)
    if any(not isinstance(feature, MACEEmbeddingFeature) for feature in features):
        raise TypeError("embedding_features must be MACEEmbeddingFeature values.")
    if len({feature.name for feature in features}) != len(features):
        raise ValueError("Joint-embedding feature names must be distinct.")
    embedding_readout_ = _flag(embedding_readout, "embedding_readout")
    if embedding_readout_ and not features:
        raise ValueError("An embedding readout needs a declared joint embedding.")
    return _ArchitectureFields(
        species=species_,
        species_kind=species_kind,
        cutoff=positive_finite_float(cutoff, "cutoff"),
        radial_basis=basis,
        radial_basis_count=count,
        cutoff_power=positive_integer(cutoff_power, "cutoff_power"),
        distance_transform=transform,
        agnesi_parameters=agnesi,
        soft_alpha=alpha,
        cutoff_placement=placement,
        parity=parity_,
        channel_count=positive_integer(channel_count, "channel_count"),
        hidden_degree=hidden,
        edge_degree=edge,
        edge_channel_count=edge_channels,
        edge_feature_degree=edge_features,
        edge_irreps_first=first,
        interactions=kinds,
        correlations=orders,
        radial_widths=widths,
        readout_width=None
        if readout_width is None
        else positive_integer(readout_width, "readout_width"),
        readout_variant=variant,
        radial_activation_scale=radial_scale,
        readout_activation_scale=readout_scale,
        gate_scales=gates,
        average_neighbor_count=positive_finite_float(
            average_neighbor_count, "average_neighbor_count"
        ),
        heads=names,
        head=selected,
        energy_scaling=scaling,
        pair_repulsion=_flag(pair_repulsion, "pair_repulsion"),
        readout_correlation=readout_order,
        last_readout_only=_flag(last_readout_only, "last_readout_only"),
        keep_last_layer_irreps=keep,
        agnostic_product=_flag(agnostic_product, "agnostic_product"),
        embedding_features=features,
        embedding_readout=embedding_readout_,
    )


def _flag(value: bool, name: str, /) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")
    return value


def _species(values: Sequence[int], kind: AtomisticSpeciesKind, /) -> tuple[int, ...]:
    match kind:
        case AtomisticSpeciesKind.ATOMIC_NUMBER:
            species = tuple(positive_integer(value, "species") for value in values)
            if any(value > 118 for value in species):
                raise ValueError("Atomic-number species must lie in 1..118.")
        case AtomisticSpeciesKind.ATOM_TYPE_ID:
            species = tuple(nonnegative_integer(value, "species") for value in values)
        case unreachable:
            assert_never(unreachable)
    if not species or len(set(species)) != len(species):
        raise ValueError("MACE species must be distinct and non-empty.")
    return species


def _transform_parameters(
    transform: MACEDistanceTransform,
    agnesi: tuple[float, float, float] | None,
    alpha: float | None,
    /,
) -> tuple[tuple[float, float, float] | None, float | None]:
    if transform != "agnesi" and agnesi is not None:
        raise ValueError("agnesi_parameters require distance_transform='agnesi'.")
    if transform != "soft" and alpha is not None:
        raise ValueError("soft_alpha requires distance_transform='soft'.")
    match transform:
        case "none":
            return None, None
        case "agnesi":
            if agnesi is None or len(agnesi) != 3:
                raise ValueError("The Agnesi transform declares (a, q, p).")
            a, q, p = (finite_real_scalar(value, "agnesi_parameters") for value in agnesi)
            return (a, q, p), None
        case "soft":
            if alpha is None:
                raise ValueError("The Soft transform declares its steepness alpha.")
            return None, finite_real_scalar(alpha, "soft_alpha")
        case unreachable:
            assert_never(unreachable)


def _edge_irreps(
    channels: int | None, degree: int | None, hidden: int, /
) -> tuple[int | None, int | None]:
    if (channels is None) != (degree is None):
        raise ValueError("Edge irreps declare both edge_channel_count and degree.")
    if channels is None or degree is None:
        return None, None
    channels_ = positive_integer(channels, "edge_channel_count")
    degree_ = nonnegative_integer(degree, "edge_feature_degree")
    if degree_ > hidden:
        raise ValueError(
            "Edge irreps above the hidden degree would be lifted from absent hidden "
            "blocks; such sources are not an admitted MACE family."
        )
    return channels_, degree_


def _gate_scales(
    scales: tuple[float, float] | None, required: bool, /
) -> tuple[float, float] | None:
    if not required:
        if scales is not None:
            raise ValueError("gate_scales belong to residual-nonlinear interactions.")
        return None
    if scales is None:
        return normalized_silu_scale(), normalized_sigmoid_scale()
    if len(scales) != 2:
        raise ValueError("gate_scales declares (silu, sigmoid) normalization constants.")
    silu, sigmoid = scales
    return (
        positive_finite_float(silu, "gate_scales"),
        positive_finite_float(sigmoid, "gate_scales"),
    )


def _readout_contract(
    interaction_count: int,
    readout_correlation: int | None,
    readout_width: int | None,
    last_readout_only: bool,
    /,
) -> bool:
    """Validate the readout declaration; return whether a nonlinear readout exists."""
    if interaction_count == 1:
        if readout_correlation is None:
            if readout_width is not None:
                raise ValueError(
                    "A one-interaction model without readout_correlation has only a "
                    "linear readout; readout_width must be None."
                )
            if last_readout_only:
                raise ValueError("A one-interaction model needs its linear readout.")
            return False
        if readout_width is None:
            raise ValueError("The invariant readout product feeds a nonlinear readout.")
        return True
    if readout_correlation is not None:
        raise ValueError("readout_correlation is admitted only for one interaction.")
    if readout_width is None:
        raise ValueError("Multi-interaction models need readout_width.")
    return True


def _architecture_id(fields: _ArchitectureFields, /) -> str:
    payload: dict[str, object] = dict(fields._asdict())
    payload["species_kind"] = fields.species_kind.value
    payload["embedding_features"] = [
        feature.feature_id for feature in fields.embedding_features
    ]
    return canonical_fingerprint({"kind": "mace-architecture-declaration", **payload})


@final
class MACESpeciesTable(StrictModule, NonTrainableState):
    """Fixed lookup from raw species identifiers to model species indices."""

    __strict_contract__ = True

    lookup: Int32[MACESpeciesLookupDim]

    def __init__(self, species: Sequence[int], /) -> None:
        values = tuple(species)
        table = np.full((max(values) + 1,), -1, dtype=np.int32)
        table[np.asarray(values, dtype=np.int64)] = np.arange(len(values), dtype=np.int32)
        self.lookup = jnp.asarray(table)

    def indices(self, identifiers: Array, mask: Array, /) -> Array:
        """Model species of active particles; unknown active species fail closed."""
        raw = jnp.asarray(identifiers, dtype=jnp.int32)
        maximum = self.lookup.shape[0] - 1
        inside = (raw >= 0) & (raw <= maximum)
        index = jnp.where(inside, self.lookup[jnp.clip(raw, 0, maximum)], -1)
        index = eqx.error_if(
            index,
            jnp.any(mask & (index < 0)),
            "Particle species is outside the MACE model species domain.",
        )
        return jnp.where(mask, index, 0)


def _compute_dtype(precision: AtomisticPrecisionPolicy, /) -> np.dtype:
    dtype = np.dtype(precision.compute_dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("MACE computes in float32 or float64.")
    return dtype


def mace_species_radii(
    architecture: MACEArchitecture, scale: AtomisticScaleContract, /
) -> np.ndarray:
    """Source covalent radii of the model species in the model length unit."""
    factor = float(conversion_factor(ANGSTROM, scale.length_unit))
    return covalent_radii_angstrom(architecture.species) * factor


def mace_radial_basis(architecture: MACEArchitecture, dtype: np.dtype, /) -> RadialBasis:
    """Native realization of the declared radial basis."""
    match architecture.radial_basis:
        case "bessel":
            return BesselRadialBasis.native(
                architecture.cutoff, architecture.radial_basis_count, dtype=dtype
            )
        case "gaussian":
            return GaussianRadialBasis.native(
                architecture.cutoff, architecture.radial_basis_count, dtype=dtype
            )
        case "chebyshev":
            return ChebyshevRadialBasis(
                architecture.cutoff, architecture.radial_basis_count, dtype=dtype
            )
        case unreachable:
            assert_never(unreachable)


def _native_transform(
    architecture: MACEArchitecture,
    scale: AtomisticScaleContract,
    dtype: np.dtype,
    /,
) -> DistanceTransform | None:
    match architecture.distance_transform:
        case "none":
            return None
        case "agnesi":
            parameters = architecture.agnesi_parameters
            if parameters is None:
                raise RuntimeError("The Agnesi transform lost its parameters.")
            a, q, p = parameters
            return AgnesiTransform(
                mace_species_radii(architecture, scale),
                atomic_numbers=architecture.species,
                a=a,
                q=q,
                p=p,
                dtype=dtype,
            )
        case "soft":
            alpha = architecture.soft_alpha
            if alpha is None:
                raise RuntimeError("The Soft transform lost its steepness.")
            return SoftTransform(
                mace_species_radii(architecture, scale),
                atomic_numbers=architecture.species,
                alpha=alpha,
                dtype=dtype,
            )
        case unreachable:
            assert_never(unreachable)


def _native_geometry(
    architecture: MACEArchitecture,
    scale: AtomisticScaleContract,
    dtype: np.dtype,
    /,
) -> MACEEdgeGeometry:
    return MACEEdgeGeometry(
        RadialEmbedding(
            mace_radial_basis(architecture, dtype),
            PolynomialCutoff(architecture.cutoff, architecture.cutoff_power),
            transform=_native_transform(architecture, scale, dtype),
        ),
        RealCartesianHarmonics(
            architecture.edge_degree, normalization="fully_normalized"
        ),
        cutoff_placement=architecture.cutoff_placement,
    )


def mace_tensor_product_plan(
    architecture: MACEArchitecture,
    index: int,
    /,
    *,
    path_scales: Sequence[float] | None = None,
) -> O3TensorProductPlan:
    """Coupling plan of one layer; ``path_scales`` carry imported basis signs."""
    message, paths = convolution_paths(
        architecture.edge_layout(index),
        architecture.harmonics_layout(),
        architecture.target_layout(),
    )
    scales = (1.0,) * len(paths) if path_scales is None else tuple(path_scales)
    if len(scales) != len(paths):
        raise ValueError("path_scales must give one scale per coupling path.")
    return O3TensorProductPlan(
        architecture.edge_layout(index),
        architecture.harmonics_layout(),
        message,
        paths=tuple(
            O3TensorProductPath(
                left, right, output, connection_mode="uvu", path_scale=scale
            )
            for (left, right, output), scale in zip(paths, scales, strict=True)
        ),
    )


def mace_radial_widths(architecture: MACEArchitecture, index: int, /) -> tuple[int, ...]:
    """Complete layer widths of one interaction's edge-weight network."""
    parameters = mace_tensor_product_plan(architecture, index).parameter_count
    basis = architecture.radial_basis_count
    match interaction_family(architecture.interactions[index]):
        case "standard":
            return (basis, *architecture.radial_widths, parameters)
        case "attention":
            return (
                basis + 2 * ATTENTION_PROJECTION_WIDTH,
                *ATTENTION_RADIAL_WIDTHS,
                parameters,
            )
        case "nonlinear":
            return (
                basis + 2 * architecture.channel_count,
                *architecture.radial_widths,
                parameters,
            )
        case unreachable:
            assert_never(unreachable)


def mace_contraction_plan(
    architecture: MACEArchitecture,
    input_degree: int,
    output_degree: int,
    correlation: int,
    dtype: np.dtype,
    /,
) -> SymmetricContractionPlan:
    return SymmetricContractionPlan.native(
        architecture.layout(1, input_degree),
        architecture.layout(1, output_degree),
        correlation,
        species_count=architecture.contraction_species_count,
        channel_count=architecture.channel_count,
        dtype=dtype,
    )


def mace_gate(architecture: MACEArchitecture, /) -> MACEGate:
    scales = architecture.gate_scales
    if scales is None:
        raise RuntimeError("A residual-nonlinear interaction lost its gate scales.")
    return MACEGate(
        architecture.target_layout(), scalar_scale=scales[0], gate_scale=scales[1]
    )


def _native_interaction(
    architecture: MACEArchitecture,
    index: int,
    dtype: np.dtype,
    key: PRNGKey,
    /,
) -> MACEAnyInteraction:
    keys = jr.split(key, 10)
    kind = architecture.interactions[index]
    inputs = architecture.input_layout(index)
    outputs = architecture.output_layout(index)
    edges = architecture.edge_layout(index)
    target = architecture.target_layout()
    plan = mace_tensor_product_plan(architecture, index)
    message, _ = convolution_paths(edges, architecture.harmonics_layout(), target)
    widths = mace_radial_widths(architecture, index)
    linear_up = O3IrrepLinear(inputs, edges, dtype=dtype, key=keys[0])
    coupling = O3TensorProduct(plan, internal_weights=False, dtype=dtype)
    match interaction_family(kind):
        case "standard":
            residual, density_normalized = interaction_traits(kind)
            return MACEInteraction(
                linear_up,
                coupling,
                RadialMLP.initialize(
                    widths,
                    key=keys[1],
                    dtype=dtype,
                    activation_scale=architecture.radial_activation_scale,
                ),
                O3IrrepLinear(message, target, dtype=dtype, key=keys[2]),
                MACESelfConnection.initialize(
                    inputs if residual else target,
                    outputs if residual else target,
                    species_count=architecture.species_count,
                    key=keys[3],
                    dtype=dtype,
                ),
                kind=kind,
                average_neighbor_count=architecture.average_neighbor_count,
                density=RadialMLP.initialize(
                    (architecture.radial_basis_count, 1),
                    key=keys[4],
                    dtype=dtype,
                    activation_scale=architecture.radial_activation_scale,
                    postprocess="tanh-square",
                )
                if density_normalized
                else None,
            )
        case "attention":
            return MACEAttentionInteraction(
                linear_up,
                O3IrrepLinear(
                    inputs,
                    architecture.layout(ATTENTION_PROJECTION_WIDTH, 0),
                    dtype=dtype,
                    key=keys[1],
                ),
                coupling,
                RadialMLP.initialize(
                    widths,
                    key=keys[2],
                    dtype=dtype,
                    activation_scale=architecture.radial_activation_scale,
                ),
                O3IrrepLinear(message, target, dtype=dtype, key=keys[3]),
                MACEPlainLinear(inputs, outputs, dtype=dtype, key=keys[4]),
                average_neighbor_count=architecture.average_neighbor_count,
            )
        case "nonlinear":
            gate = mace_gate(architecture)
            nonlinear = gate_layout(target)
            species = architecture.species_count
            channels = architecture.channel_count
            # Source species embeddings start uniform in [-1e-3, 1e-3].
            return MACENonlinearInteraction(
                linear_up,
                jr.uniform(keys[1], (species, channels), dtype, -1.0e-3, 1.0e-3),
                jr.uniform(keys[2], (species, channels), dtype, -1.0e-3, 1.0e-3),
                coupling,
                LayerNormRadialMLP.initialize(widths, key=keys[3], dtype=dtype),
                LayerNormRadialMLP.initialize(
                    (widths[0], NONLINEAR_DENSITY_WIDTH, 1),
                    key=keys[4],
                    dtype=dtype,
                    postprocess="tanh-square",
                ),
                O3IrrepLinear(message, nonlinear, dtype=dtype, key=keys[5]),
                MACEPlainLinear(edges, nonlinear, dtype=dtype, key=keys[6]),
                gate,
                O3IrrepLinear(target, target, dtype=dtype, key=keys[7]),
                MACEPlainLinear(inputs, outputs, dtype=dtype, key=keys[8]),
                jnp.asarray(NONLINEAR_DENSITY_ALPHA, dtype=dtype),
                jnp.asarray(NONLINEAR_DENSITY_BETA, dtype=dtype),
            )
        case unreachable:
            assert_never(unreachable)


def _native_layer(
    architecture: MACEArchitecture,
    scale: AtomisticScaleContract,
    index: int,
    dtype: np.dtype,
    key: PRNGKey,
    /,
) -> MACELayer:
    keys = jr.split(key, 5)
    residual, _ = interaction_traits(architecture.interactions[index])
    outputs = architecture.output_layout(index)
    target = architecture.target_layout()
    product = MACEProduct(
        SymmetricContraction.initialize(
            mace_contraction_plan(
                architecture,
                architecture.edge_degree,
                architecture.output_degree(index),
                architecture.correlations[index],
                dtype,
            ),
            key=keys[1],
            dtype=dtype,
        ),
        O3IrrepLinear(outputs, outputs, dtype=dtype, key=keys[2]),
        target,
        self_connection=index > 0 or residual,
    )
    return MACELayer(
        _native_interaction(architecture, index, dtype, keys[0]),
        product,
        readout=_native_readout(architecture, index, dtype, keys[3]),
        readout_product=_native_readout_product(architecture, dtype, keys[4])
        if architecture.readout_correlation is not None
        else None,
        pair_repulsion=MACEPairRepulsion(
            architecture.species, scale, power=architecture.cutoff_power, dtype=dtype
        )
        if index == 0 and architecture.pair_repulsion
        else None,
    )


def _native_readout(
    architecture: MACEArchitecture, index: int, dtype: np.dtype, key: PRNGKey, /
) -> MACEReadout | None:
    heads = len(architecture.heads)
    match architecture.readout_kind(index):
        case "none":
            return None
        case "linear":
            return MACELinearReadout.initialize(
                architecture.channel_count, heads, key=key, dtype=dtype
            )
        case "nonlinear":
            width = architecture.readout_width
            scale = architecture.readout_activation_scale
            if width is None or scale is None:
                raise RuntimeError(
                    "A nonlinear readout lost its declared width or scale."
                )
            match architecture.readout_variant:
                case "nonlinear":
                    return MACENonlinearReadout.initialize(
                        architecture.channel_count,
                        width,
                        heads,
                        activation_scale=scale,
                        key=key,
                        dtype=dtype,
                    )
                case "nonlinear-bias":
                    return MACENonlinearBiasReadout.initialize(
                        architecture.channel_count,
                        width,
                        heads,
                        activation_scale=scale,
                        key=key,
                        dtype=dtype,
                    )
                case unreachable:
                    assert_never(unreachable)
        case unreachable:
            assert_never(unreachable)


def _native_readout_product(
    architecture: MACEArchitecture, dtype: np.dtype, key: PRNGKey, /
) -> MACEInvariantReadoutProduct:
    correlation = architecture.readout_correlation
    if correlation is None:
        raise RuntimeError("The invariant readout product needs readout_correlation.")
    contraction_key, linear_key = jr.split(key)
    channel_layout = architecture.layout(1, architecture.hidden_degree)
    scalars = architecture.layout(architecture.channel_count, 0)
    return MACEInvariantReadoutProduct(
        SymmetricContraction.initialize(
            mace_contraction_plan(
                architecture, architecture.hidden_degree, 0, correlation, dtype
            ),
            key=contraction_key,
            dtype=dtype,
        ),
        O3IrrepLinear(scalars, scalars, dtype=dtype, key=linear_key),
        channel_layout,
    )


def mace_node_feature_port(
    architecture: MACEArchitecture, level: AtomisticConditionLevel, /
) -> ValuePort:
    """Port of the concatenated inter-layer node states (or their case sums).

    Components follow the native packed layout of every layer's output in
    layer order: block, then channel, then real-harmonic component in the
    native fully normalized basis (not the source e3nn component order).
    """
    components = tuple(
        f"layer{index}.{block.name}.c{channel}.m{component}"
        for index in range(architecture.interaction_count)
        for block in architecture.output_layout(index).blocks
        for channel in range(block.multiplicity)
        for component in range(block.dimension)
    )
    semantic = (
        "mace.node-features"
        if level is AtomisticConditionLevel.ATOM
        else "mace.graph-features-sum"
    )
    return ValuePort(
        semantic,
        event_shape=(len(components),),
        component_ids=components,
        representation=NODE_FEATURES_REPRESENTATION,
        normalization_id="identity",
    )


@final
class MACEPotential(AbstractAtomisticPotential):
    """Native trainable MACE scalar-energy potential.

    Construct with only ``key`` for the source family's random initialization,
    or with all of ``embedding``, ``geometry`` and ``layers`` (plus the joint
    embedding and embedding readout when declared), as the trusted source
    reconstruction does; both routes run the same owning validators.
    ``streaming`` is an explicit streamed-relation tiling; ``None`` reuses the
    schedule each bound graph topology already prepared for its epoch.

    ``acceleration`` is an execution policy, not architecture: an admitted
    accelerated coupling kernel aggregates each streamed fragment's coupling
    from fragment-local edge weights (evaluated by the exact family equations,
    including attention projections and species embeddings) and harmonics
    (density and ZBL use the substrate's seeded reduction over the same lanes),
    and the receiver epilogue still runs once per receiver after its final
    fragment. It never enters the architecture identity or numeric revision;
    every parameter keeps its source role, so the accelerated route is
    trainable.
    """

    embedding: Array
    joint_embedding: MACEJointEmbedding | None
    embedding_readout: MACELinearReadout | None
    geometry: MACEEdgeGeometry
    layers: tuple[MACELayer, ...]
    energy_reference: MACEEnergyReference
    species_table: MACESpeciesTable
    configuration: MACEArchitecture
    scale: AtomisticScaleContract
    precision: AtomisticPrecisionPolicy
    streaming: StreamedRelationPlan | None
    acceleration: MACEAcceleratedCoupling | None
    architecture_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: AtomisticScaleContract,
        architecture: MACEArchitecture,
        /,
        *,
        atomic_energies: ArrayLike,
        energy_scale: ArrayLike | None = None,
        energy_shift: ArrayLike | None = None,
        precision: AtomisticPrecisionPolicy | None = None,
        streaming: StreamedRelationPlan | None = None,
        acceleration: MACEAcceleratedCoupling | None = None,
        embedding: ArrayLike | None = None,
        geometry: MACEEdgeGeometry | None = None,
        layers: Sequence[MACELayer] | None = None,
        joint_embedding: MACEJointEmbedding | None = None,
        embedding_readout: MACELinearReadout | None = None,
        key: PRNGKey = DOC_KEY0,
    ) -> None:
        if not isinstance(scale, AtomisticScaleContract):
            raise TypeError("scale must be an AtomisticScaleContract.")
        if not isinstance(architecture, MACEArchitecture):
            raise TypeError("architecture must be a MACEArchitecture.")
        precision_ = AtomisticPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, AtomisticPrecisionPolicy):
            raise TypeError("precision must be an AtomisticPrecisionPolicy or None.")
        dtype = _compute_dtype(precision_)
        supplied = (embedding is not None, geometry is not None, layers is not None)
        if any(supplied) and not all(supplied):
            raise ValueError("Supply all of embedding, geometry, and layers, or none.")
        if embedding is None or geometry is None or layers is None:
            if joint_embedding is not None or embedding_readout is not None:
                raise ValueError(
                    "Joint-embedding parameters are supplied with a reconstruction."
                )
            keys = jr.split(key, architecture.interaction_count + 3)
            embedding_ = jr.normal(
                keys[0],
                (architecture.species_count, architecture.channel_count),
                dtype=dtype,
            )
            geometry_ = _native_geometry(architecture, scale, dtype)
            layers_ = tuple(
                _native_layer(architecture, scale, index, dtype, keys[index + 3])
                for index in range(architecture.interaction_count)
            )
            joint_ = (
                MACEJointEmbedding.initialize(
                    architecture.embedding_features,
                    architecture.channel_count,
                    key=keys[1],
                    dtype=dtype,
                )
                if architecture.embedding_features
                else None
            )
            embedding_readout_ = (
                MACELinearReadout.initialize(
                    architecture.channel_count,
                    len(architecture.heads),
                    key=keys[2],
                    dtype=dtype,
                )
                if architecture.embedding_readout
                else None
            )
        else:
            embedding_ = jnp.asarray(embedding, dtype=dtype)
            geometry_ = geometry
            layers_ = tuple(layers)
            joint_ = joint_embedding
            embedding_readout_ = embedding_readout
        reference = MACEEnergyReference(
            atomic_energies,
            heads=architecture.heads,
            head=architecture.head,
            scaling=architecture.energy_scaling,
            scale=energy_scale,
            shift=energy_shift,
            species_count=architecture.species_count,
            dtype=precision_.reduction_dtype,
        )
        if streaming is not None and not isinstance(streaming, StreamedRelationPlan):
            raise TypeError("streaming must be a StreamedRelationPlan or None.")
        _check_components(
            architecture,
            scale,
            dtype,
            embedding_,
            joint_,
            embedding_readout_,
            geometry_,
            layers_,
        )
        _check_acceleration(acceleration, layers_, dtype)
        self.embedding = embedding_
        self.joint_embedding = joint_
        self.embedding_readout = embedding_readout_
        self.geometry = geometry_
        self.layers = layers_
        self.energy_reference = reference
        self.species_table = MACESpeciesTable(architecture.species)
        self.configuration = architecture
        self.scale = scale
        self.precision = precision_
        self.streaming = streaming
        self.acceleration = acceleration
        self.architecture_id = _model_identity(
            architecture, scale, precision_, reference, geometry_, layers_, joint_
        )
        self.method_id = _METHOD_ID

    def validate(self) -> None:
        """Run every owning validator on this (possibly restored) potential.

        This is a host boundary over concrete arrays. It re-checks the
        architecture declaration, layouts, coupling plans and their executed
        coefficient tables and support, symmetric-contraction bases, product
        graphs and term maps (finiteness, equivariance, source evidence and
        recomputed identities), every fixed radial field (basis arrays, cutoff,
        transform radii and constants, harmonics) and ZBL field rebuilt from the
        stored arrays, parameter shapes, dtypes and finiteness, the joint
        embedding declaration, reference energies, species lookup, and the
        recorded scientific identity.
        """
        architecture = self.configuration
        architecture.validate()
        dtype = _compute_dtype(self.precision)
        _check_components(
            architecture,
            self.scale,
            dtype,
            self.embedding,
            self.joint_embedding,
            self.embedding_readout,
            self.geometry,
            self.layers,
        )
        _check_acceleration(self.acceleration, self.layers, dtype)
        reference = MACEEnergyReference(
            np.asarray(jax.device_get(self.energy_reference.atomic_energies)),
            heads=architecture.heads,
            head=architecture.head,
            scaling=architecture.energy_scaling,
            scale=None
            if architecture.energy_scaling == "unscaled"
            else np.asarray(jax.device_get(self.energy_reference.scale)),
            shift=None
            if architecture.energy_scaling == "unscaled"
            else np.asarray(jax.device_get(self.energy_reference.shift)),
            species_count=architecture.species_count,
            dtype=self.precision.reduction_dtype,
        )
        if reference.reference_id != self.energy_reference.reference_id:
            raise ValueError("MACE reference energies do not match their identity.")
        expected = MACESpeciesTable(architecture.species).lookup
        if not np.array_equal(
            np.asarray(jax.device_get(expected)),
            np.asarray(jax.device_get(self.species_table.lookup)),
        ):
            raise ValueError("The MACE species lookup does not match its species.")
        parameters = jax.tree_util.tree_leaves(partition_parameters(self)[0])
        if not all(
            bool(np.all(np.isfinite(jax.device_get(leaf)))) for leaf in parameters
        ):
            raise ValueError("MACE parameters must be finite.")
        identity = _model_identity(
            architecture,
            self.scale,
            self.precision,
            self.energy_reference,
            self.geometry,
            self.layers,
            self.joint_embedding,
        )
        if identity != self.architecture_id or self.method_id != _METHOD_ID:
            raise ValueError("The MACE potential does not match its recorded identities.")

    def with_acceleration(
        self, acceleration: MACEAcceleratedCoupling | None, /
    ) -> MACEPotential:
        """The same potential (identities, parameters) under another execution policy."""
        _check_acceleration(acceleration, self.layers, _compute_dtype(self.precision))
        return eqx.tree_at(
            lambda model: model.acceleration,
            self,
            acceleration,
            is_leaf=lambda value: value is None,
        )

    @property
    def capabilities(self) -> AtomisticPotentialCapabilities:
        return AtomisticPotentialCapabilities(
            orthorhombic_periodic=True,
            triclinic_periodic=True,
            cell_derivative=True,
            species_kind=self.configuration.species_kind,
        )

    @property
    def conditioning(self) -> AtomisticConditioningSpec:
        """Typed conditioning this potential consumes (empty without joint embedding)."""
        return self.configuration.conditioning

    @property
    def supported_observables(self) -> tuple[AtomisticObservable, ...]:
        return (
            AtomisticObservable.ENERGY,
            AtomisticObservable.ATOM_ENERGY,
            AtomisticObservable.NODE_EMBEDDINGS,
            AtomisticObservable.GRAPH_EMBEDDINGS,
        )

    @property
    def derivative_contract(self) -> DerivativeContract | None:
        """Truthful regularity of the exact (unaccelerated) MACE energy.

        Bessel, Gaussian and Chebyshev radial features, Agnesi transforms with
        ``a >= 0`` and Soft transforms, real spherical harmonics, normalized
        SiLU, ``tanh(x^2)``, layer-normalized (``epsilon > 0``) and gated
        networks, the ``1 + density`` normalization and polynomial products are
        smooth off coincident positions; the polynomial cutoff (and the optional
        ZBL envelope at covalent-radius sums) joins C2. The residual-nonlinear
        normalization ``beta * density + alpha`` has trainable scalars and is
        singular where it vanishes, a declared locus outside the support.
        Coordinate Hessians are classical; a third-order (mixed HVP-training)
        response is an almost-everywhere derivative needing an explicit policy.
        An accelerated coupling (which refuses order three) and an Agnesi
        transform with ``a < 0`` (whose denominator may vanish) are not declared
        here.
        """
        architecture = self.configuration
        if self.acceleration is not None:
            return None
        agnesi = architecture.agnesi_parameters
        if agnesi is not None and agnesi[0] < 0.0:
            return None
        conditions = (
            (atomistic_edge_distance_locus(architecture.cutoff),)
            + ((_ZBL_ENVELOPE_LOCUS,) if architecture.pair_repulsion else ())
            + (
                (_NONLINEAR_DENOMINATOR_LOCUS,)
                if any(
                    interaction_family(kind) == "nonlinear"
                    for kind in architecture.interactions
                )
                else ()
            )
        )
        return DerivativeContract(
            (
                SurfaceDerivative(DerivativeSurface.INPUT, GradientLevel.SMOOTH),
                SurfaceDerivative(
                    DerivativeSurface.MODEL_PARAMETER, GradientLevel.SMOOTH
                ),
            ),
            route=DerivativeRoute.DIRECT,
            regularity=DerivativeRegularity.piecewise_smooth(
                continuity=2,
                conditions=conditions,
                support=ATOMISTIC_NONCOINCIDENT_SUPPORT,
            ),
        )

    @property
    def interaction_count(self) -> int:
        return self.configuration.interaction_count

    @property
    def system_stages(self) -> tuple[AtomisticSystemStage, ...]:
        """MACE has no system reductions: conditioning enters node-locally."""
        return ()

    def species_indices(self, species_ids: ArrayLike, atom_mask: ArrayLike, /) -> Array:
        """Model species indices of active particles (zero on inactive slots)."""
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_)
        return self.species_table.indices(jnp.asarray(species_ids), mask)

    def _require_context(self, context: MACELayerContext, /) -> MACELayerContext:
        if not isinstance(context, MACELayerContext):
            raise TypeError("MACE layered hooks consume a MACELayerContext.")
        if (context.joint is None) != (self.joint_embedding is None):
            raise ValueError(
                "The joint-embedding addend is required exactly when the model "
                "declares a joint embedding; build the context with layer_context "
                "from the declared conditioning."
            )
        return context

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
        """Joint-embedding addend and embedding-readout energy of these rows.

        Owner local: ``atom_cases`` index the case-level conditioning fields of
        each row. Missing or differently declared fields refuse (see
        `joint_features`); an unconditioned MACE yields an empty context.
        """
        del key
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_)
        joint = self.joint_features(conditioning, atom_cases, mask)
        if self.embedding_readout is None:
            return MACELayerContext(joint, None)
        initial = self.init_features(species, mask, MACELayerContext(joint, None))
        return MACELayerContext(
            joint, self.embedding_readout(initial, self.energy_reference.head)
        )

    def admitted(
        self, species: Array, atom_mask: Array, context: MACELayerContext, /
    ) -> Array:
        """Always admitted: unknown species fail closed in `species_indices`."""
        del species, atom_mask, context
        return jnp.asarray(True)

    def system_partials(
        self,
        stage: int,
        features: Array | None,
        species: Array,
        atom_mask: Array,
        context: MACELayerContext,
        /,
    ) -> None:
        del features, species, atom_mask, context
        raise ValueError(f"MACE declares no system stage {stage}.")

    def apply_system_context(
        self,
        stage: int,
        features: Array | None,
        system_context: None,
        species: Array,
        atom_mask: Array,
        context: MACELayerContext,
        /,
    ) -> tuple[Array | None, MACELayerContext]:
        del features, system_context, species, atom_mask, context
        raise ValueError(f"MACE declares no system stage {stage}.")

    def init_features(
        self,
        species: Array,
        atom_mask: Array,
        context: MACELayerContext,
        /,
        *,
        accumulation: RelationAccumulation = "fast",
    ) -> Array:
        """Layer-0 node states ``C x 0e``: the normalized species embedding.

        ``species`` are species-table indices. Ordered ``accumulation`` selects
        one atom's embedding row per step, so embedding cotangents add in atom
        order instead of by unordered duplicate-index scatter additions. A model
        with a joint embedding adds the context's per-atom addend (see
        `layer_context`); a context without it is refused rather than silently
        omitting the conditioning.
        """
        joint = self._require_context(context).joint
        dtype = self.embedding.dtype
        fan_in = jnp.sqrt(jnp.asarray(self.embedding.shape[0], dtype=dtype))
        rows = (
            self.embedding[species]
            if accumulation == "fast" or not species.size
            else jax.lax.map(lambda index: self.embedding[index], species)
        )
        features = rows / fan_in
        if joint is not None:
            features = features + joint.astype(dtype)
        return jnp.where(atom_mask[:, None], features, jnp.zeros((), dtype=dtype))

    def joint_features(
        self,
        conditioning: AtomisticConditioning | None,
        atom_cases: Array,
        atom_mask: Array,
        /,
    ) -> Array | None:
        """Per-atom joint-embedding addend, or ``None`` without a joint embedding."""
        require_mace_conditioning(self.configuration.embedding_features, conditioning)
        if self.joint_embedding is None or conditioning is None:
            return None
        return self.joint_embedding(conditioning, jnp.asarray(atom_cases), atom_mask)

    def prepare_sources(
        self,
        index: int,
        features: Array,
        species: Array,
        atom_mask: Array,
        context: MACELayerContext,
        /,
    ) -> MACESourceRows:
        """Node-local ``linear_up`` source states (and edge context) of one layer."""
        del context
        return self.layers[index].interaction.source_rows(features, species, atom_mask)

    def update_receivers(
        self,
        index: int,
        relation: PreparedStreamedRelation,
        vectors: Array,
        sources: MACESourceRows,
        features: Array,
        species: Array,
        atom_mask: Array,
        context: MACELayerContext,
        /,
        *,
        edge_active: Array | None = None,
        route_identity: tuple[Array, ...] | None = None,
    ) -> MACELayerUpdate:
        """Stream one layer: every receiver's complete update and readout energy.

        ``vectors`` are edge displacements ``r_receiver - r_source (+ n H)`` in
        route order; the relation's sources are ``sources`` rows and its
        receivers the rows of ``features``. Energies are unscaled per-node
        readout terms (the first layer also carries ZBL). MACE reads neither
        the layer context nor route identities here.
        """
        del context, route_identity
        dtype = self.embedding.dtype
        layer = self.layers[index]
        streamed = MACEStreamedLayer(
            layer,
            self.geometry,
            self.energy_reference.head,
            coupling=self.acceleration,
        )
        message, output = streamed.payload_shapes(dtype)
        payload = StreamedPayloadSpec(message, output)
        receivers = MACEReceiverRows(
            features, species, atom_mask, layer.interaction.receiver_context(features)
        )
        if self.acceleration is None:
            result = relation.evaluate(
                payload,
                MACEStreamedLayer.edge,
                MACEStreamedLayer.epilogue,
                streamed,
                sources,
                receivers,
                jnp.asarray(vectors).astype(dtype),
                edge_active=edge_active,
            )
        else:
            result = relation.evaluate_fragments(
                payload,
                MACEStreamedLayer.fragment,
                MACEStreamedLayer.epilogue,
                streamed,
                sources,
                receivers,
                jnp.asarray(vectors).astype(dtype),
                edge_active=edge_active,
            )
        outputs = result.receiver_outputs
        kernel_resources = None
        if self.acceleration is not None:
            schedule = relation.schedule
            kernel_resources = self.acceleration.resources(
                MACEEdgeCouplingSpec(layer.interaction.tensor_product),
                MACEFragmentExtent(
                    receivers=schedule.receiver_tile,
                    sources=schedule.edge_tile,
                    edges=schedule.edge_tile,
                ),
            )
        return MACELayerUpdate(
            outputs["features"],
            outputs["energy"],
            result.evidence,
            result.resources,
            kernel_resources,
        )

    def atom_energies(
        self,
        species: Array,
        atom_mask: Array,
        layer_energies: Array,
        features: Array,
        context: MACELayerContext,
        /,
    ) -> Array:
        """Per-atom ``E0 + scale * sum(layer terms) + shift (+ embedding readout)``.

        The source adds the embedding-readout energy to its reference energy
        unscaled; it is a genuine per-atom term, so it enters the atom energy.
        It is the context's ``embedding_energy`` (required when declared);
        ``features`` (the final states) do not enter the MACE readout sum.
        """
        del features
        if not isinstance(context, MACELayerContext):
            raise TypeError("MACE atom energies consume a MACELayerContext.")
        embedding_energy = context.embedding_energy
        reduction = jnp.dtype(self.precision.reduction_dtype)
        interaction = jnp.sum(layer_energies.astype(reduction), axis=0)
        energy = self.energy_reference.atom_energy(species, interaction)
        if (embedding_energy is None) != (self.embedding_readout is None):
            raise ValueError("The embedding-readout energy is required when declared.")
        if embedding_energy is not None:
            energy = energy + embedding_energy.astype(reduction)
        return jnp.where(atom_mask, energy, jnp.zeros((), dtype=reduction))

    def global_payload(
        self,
        features: Array,
        species: Array,
        atom_mask: Array,
        context: MACELayerContext,
        /,
    ) -> None:
        """MACE is local: no global epilogue."""
        del features, species, atom_mask, context
        return None

    def system_energy(self, context: MACELayerContext, /) -> None:
        del context
        return None

    def owner_fields(
        self,
        features: Array,
        species: Array,
        atom_mask: Array,
        context: MACELayerContext,
        /,
    ) -> Mapping[AtomisticObservable, Array]:
        """MACE predicts no per-atom field consumed by domain terms."""
        del features, species, atom_mask, context
        return {}

    def case_energies(
        self,
        relation: PreparedStreamedRelation,
        atom_energy: Array,
        atom_cases: Array,
        case_count: int,
        atom_capacity: int,
        /,
    ) -> tuple[Array, Array]:
        """Output-dtype case totals and ``(case, atom)`` energies of flat atoms.

        Atoms are flat case-major rows of ``atom_capacity`` slots; totals
        follow the ``relation`` accumulation policy.
        """
        reduction = jnp.dtype(self.precision.reduction_dtype)
        total = case_energy_totals(
            atom_energy.astype(reduction),
            jnp.asarray(atom_cases),
            case_count,
            relation.plan.accumulation,
        )
        output = jnp.dtype(self.precision.output_dtype)
        return (
            total.astype(output),
            atom_energy.reshape((case_count, atom_capacity)).astype(output),
        )

    def node_states(
        self,
        species: Array,
        atom_mask: Array,
        relation: PreparedStreamedRelation,
        vectors: Array,
        /,
        *,
        edge_active: Array | None = None,
        joint: Array | None = None,
    ) -> MACENodeStates:
        """Per-atom energies, success evidence and every layer's output state."""
        initial = MACELayerContext(joint, None)
        features = self.init_features(
            species, atom_mask, initial, accumulation=relation.plan.accumulation
        )
        context = (
            initial
            if self.embedding_readout is None
            else MACELayerContext(
                joint,
                self.embedding_readout(features, self.energy_reference.head),
            )
        )
        energies = []
        states = []
        successful = jnp.asarray(True)
        for index in range(self.interaction_count):
            sources = self.prepare_sources(index, features, species, atom_mask, context)
            update = self.update_receivers(
                index,
                relation,
                vectors,
                sources,
                features,
                species,
                atom_mask,
                context,
                edge_active=edge_active,
            )
            features = update.features
            states.append(features)
            energies.append(update.energy)
            successful = successful & update.evidence.successful
        atom_energy = self.atom_energies(
            species, atom_mask, jnp.stack(energies), features, context
        )
        return MACENodeStates(atom_energy, successful, tuple(states))

    def node_energies(
        self,
        species: Array,
        atom_mask: Array,
        relation: PreparedStreamedRelation,
        vectors: Array,
        /,
        *,
        edge_active: Array | None = None,
        joint: Array | None = None,
    ) -> tuple[Array, Array]:
        """Per-atom energies and the conjunction of every layer's success evidence."""
        result = self.node_states(
            species, atom_mask, relation, vectors, edge_active=edge_active, joint=joint
        )
        return result.atom_energy, result.successful

    @checked
    def _validate_batch(self, batch: AtomisticBatch, /) -> None:
        if batch.scale.scale_id != self.scale.scale_id:
            raise ValueError(
                "Potential and structure must share one exact scale contract."
            )
        if batch.positions.dtype != jnp.dtype(self.precision.coordinate_dtype):
            raise ValueError(
                "Batch coordinate dtype does not match the MACE precision contract."
            )

    def bound_relation(self, inputs: AtomisticGraphInputs, /) -> PreparedStreamedRelation:
        """The streamed relation this model executes on one bound graph."""
        topology = inputs.graph.topology
        return (
            topology.streamed
            if self.streaming is None
            else topology.prepare_streamed(self.streaming)
        )

    def graph_node_states(
        self, inputs: AtomisticGraphInputs, /
    ) -> tuple[Array, MACENodeStates, PreparedStreamedRelation]:
        """Admitted species indices and node states of one bound graph."""
        inputs.require_compatible(self.scale, self.capabilities.species_kind)
        graph = inputs.graph
        mask = inputs.atom_mask
        species = self.species_indices(inputs.species_ids, mask)
        ir = graph.graph
        if ir.edge_mask is None:
            raise ValueError("MACE requires an explicit masked edge relation.")
        relation = self.bound_relation(inputs)
        joint = self.joint_features(inputs.conditioning, inputs.atom_cases, mask)
        states = self.node_states(
            species,
            mask,
            relation,
            jnp.asarray(ir.edges["displacement"]),
            edge_active=jnp.asarray(ir.edge_mask, dtype=jnp.bool_),
            joint=joint,
        )
        return species, states, relation

    def feature_observables(
        self,
        inputs: AtomisticGraphInputs,
        states: MACENodeStates,
        requested: Sequence[AtomisticObservable],
        successful: Array,
        /,
    ) -> tuple[AtomisticObservableValue, ...]:
        """Requested node/graph embedding observables of evaluated node states."""
        values: list[AtomisticObservableValue] = []
        wants_nodes = AtomisticObservable.NODE_EMBEDDINGS in requested
        wants_graph = AtomisticObservable.GRAPH_EMBEDDINGS in requested
        if not (wants_nodes or wants_graph):
            return ()
        cases = inputs.case_count
        capacity = inputs.atom_capacity
        output = jnp.dtype(self.precision.output_dtype)
        nodes = jnp.concatenate(states.states, axis=-1)
        if wants_nodes:
            values.append(
                AtomisticObservableValue(
                    AtomisticObservable.NODE_EMBEDDINGS,
                    mace_node_feature_port(
                        self.configuration, AtomisticConditionLevel.ATOM
                    ),
                    nodes.astype(output),
                    successful,
                    level=AtomisticConditionLevel.ATOM,
                    case_count=cases,
                    atom_capacity=capacity,
                )
            )
        if wants_graph:
            reduction = jnp.dtype(self.precision.reduction_dtype)
            masked = jnp.where(
                inputs.atom_mask[:, None], nodes, jnp.zeros((), dtype=nodes.dtype)
            )
            graph = jnp.sum(
                masked.astype(reduction).reshape((cases, capacity, -1)), axis=1
            )
            values.append(
                AtomisticObservableValue(
                    AtomisticObservable.GRAPH_EMBEDDINGS,
                    mace_node_feature_port(
                        self.configuration, AtomisticConditionLevel.CASE
                    ),
                    graph.astype(output),
                    successful,
                    level=AtomisticConditionLevel.CASE,
                    case_count=cases,
                    atom_capacity=capacity,
                )
            )
        return tuple(values)

    @checked
    def evaluate_graph(
        self,
        inputs: AtomisticGraphInputs,
        request: AtomisticObservableRequest | None = None,
        /,
    ) -> AtomisticModelOutputs:
        """Case energies, per-atom energies and requested embeddings on one graph.

        The topology's epoch-bound streamed schedule is reused for every layer
        (or prepared once for an explicit ``streaming`` plan); the cutoff mask
        is a runtime route mask. Failed streamed schedules poison their outputs,
        so an unsuccessful evaluation is non-finite rather than plausible. A
        joint embedding consumes its declared conditioning fields; missing or
        differently declared fields refuse before execution and category
        violations make their cases unsuccessful.
        """
        if request is not None:
            request.require_supported(self.supported_observables, "MACEPotential")
        _, states, relation = self.graph_node_states(inputs)
        energy, atomic = self.case_energies(
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
            additional=self.feature_observables(inputs, states, requested, successful),
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
        raise TypeError("MACEPotential expects AtomicStructure or AtomisticBatch.")


def _model_identity(
    architecture: MACEArchitecture,
    scale: AtomisticScaleContract,
    precision: AtomisticPrecisionPolicy,
    reference: MACEEnergyReference,
    geometry: MACEEdgeGeometry,
    layers: Sequence[MACELayer],
    joint: MACEJointEmbedding | None,
    /,
) -> str:
    """Scientific identity of everything outside the PARAMETER revision.

    Fixed coupling coefficients (including imported path signs), coupling bases
    ``U``, radial bases and transforms, and reference energies change the
    function without changing parameters, so their content identities enter.
    """
    return canonical_fingerprint(
        {
            "kind": "mace-architecture",
            "declaration": architecture.architecture_id,
            "scale": scale.scale_id,
            "precision": precision.policy_id,
            "energy_reference": reference.reference_id,
            "radial": geometry.radial.embedding_id,
            "joint_embedding": joint_embedding_identity(joint),
            "couplings": [
                layer.interaction.tensor_product.coefficient_id for layer in layers
            ],
            "contractions": [layer.product.contraction.plan.plan_id for layer in layers],
            "readout_contractions": [
                None
                if layer.readout_product is None
                else layer.readout_product.contraction.plan.plan_id
                for layer in layers
            ],
            "pair_repulsion": [
                None
                if layer.pair_repulsion is None
                else layer.pair_repulsion.constants_id
                for layer in layers
            ],
        }
    )


def _check_components(
    architecture: MACEArchitecture,
    scale: AtomisticScaleContract,
    dtype: np.dtype,
    embedding: Array,
    joint: MACEJointEmbedding | None,
    embedding_readout: MACELinearReadout | None,
    geometry: MACEEdgeGeometry,
    layers: Sequence[MACELayer],
    /,
) -> None:
    """Owning validation shared by construction, reconstruction and restore."""
    if embedding.shape != (architecture.species_count, architecture.channel_count):
        raise ValueError("The species embedding must have shape (species, channels).")
    if embedding.dtype != dtype:
        raise TypeError("The species embedding must use the compute dtype.")
    _check_joint_embedding(architecture, dtype, joint, embedding_readout)
    _check_geometry(architecture, geometry, dtype)
    if len(layers) != architecture.interaction_count:
        raise ValueError("MACE declares one layer per interaction.")
    for index, layer in enumerate(layers):
        if not isinstance(layer, MACELayer):
            raise TypeError("layers must contain MACELayer values.")
        _check_layer(architecture, scale, index, layer, dtype)


def _check_joint_embedding(
    architecture: MACEArchitecture,
    dtype: np.dtype,
    joint: MACEJointEmbedding | None,
    readout: MACELinearReadout | None,
    /,
) -> None:
    features = architecture.embedding_features
    if (joint is None) != (not features):
        raise ValueError("A joint embedding is present exactly when declared.")
    if joint is not None and (
        not isinstance(joint, MACEJointEmbedding)
        or joint.features != features
        or joint.channel_count != architecture.channel_count
        or joint.projection.dtype != dtype
    ):
        raise ValueError("The joint embedding does not match its declaration.")
    if (readout is None) != (not architecture.embedding_readout):
        raise ValueError("An embedding readout is present exactly when declared.")
    if readout is not None and (
        not isinstance(readout, MACELinearReadout)
        or readout.weight.shape != (architecture.channel_count, len(architecture.heads))
        or readout.weight.dtype != dtype
    ):
        raise ValueError("The embedding readout does not match its declaration.")


def _check_acceleration(
    acceleration: MACEAcceleratedCoupling | None,
    layers: Sequence[MACELayer],
    dtype: np.dtype,
    /,
) -> None:
    """Admit an accelerated coupling for every layer's exact coupling structure."""
    if acceleration is None:
        return
    for layer in layers:
        admit_mace_coupling(acceleration, layer, dtype)


def _check_geometry(
    architecture: MACEArchitecture, geometry: MACEEdgeGeometry, dtype: np.dtype, /
) -> None:
    if not isinstance(geometry, MACEEdgeGeometry):
        raise TypeError("geometry must be a MACEEdgeGeometry.")
    radial = geometry.radial
    harmonics = geometry.harmonics
    if not isinstance(radial, RadialEmbedding) or not isinstance(
        harmonics, RealCartesianHarmonics
    ):
        raise TypeError("MACE edge geometry needs a RadialEmbedding and harmonics.")
    expected_basis = {
        "bessel": BesselRadialBasis,
        "gaussian": GaussianRadialBasis,
        "chebyshev": ChebyshevRadialBasis,
    }[architecture.radial_basis]
    if (
        not isinstance(radial.basis, expected_basis)
        or radial.radius != architecture.cutoff
        or radial.basis_count != architecture.radial_basis_count
        or radial.cutoff.power != architecture.cutoff_power
        or radial.dtype != dtype
        or harmonics.maximum_degree != architecture.edge_degree
        or harmonics.normalization != "fully_normalized"
        or harmonics.argument != "direction"
        or geometry.cutoff_placement != architecture.cutoff_placement
    ):
        raise ValueError("The MACE edge geometry does not match the architecture.")
    transform = radial.transform
    match architecture.distance_transform:
        case "none":
            if transform is not None:
                raise ValueError("The architecture declares no distance transform.")
        case "agnesi":
            if (
                not isinstance(transform, AgnesiTransform)
                or (transform.a, transform.q, transform.p)
                != architecture.agnesi_parameters
            ):
                raise ValueError("The Agnesi transform does not match its declaration.")
        case "soft":
            if (
                not isinstance(transform, SoftTransform)
                or transform.alpha != architecture.soft_alpha
            ):
                raise ValueError("The Soft transform does not match its declaration.")
        case unreachable:
            assert_never(unreachable)
    if transform is not None and transform.atomic_numbers != architecture.species:
        raise ValueError("Transform radii must follow the model species order.")
    # Every executed fixed radial field is rebuilt from the stored arrays, so
    # the embedding identity entering the model identity is never a stale label.
    radial.validate()


def _check_layer(
    architecture: MACEArchitecture,
    scale: AtomisticScaleContract,
    index: int,
    layer: MACELayer,
    dtype: np.dtype,
    /,
) -> None:
    interaction = layer.interaction
    kind = architecture.interactions[index]
    residual, _ = interaction_traits(kind)
    inputs = architecture.input_layout(index)
    outputs = architecture.output_layout(index)
    target = architecture.target_layout()
    expected_plan = mace_tensor_product_plan(
        architecture,
        index,
        path_scales=tuple(
            path.path_scale for path in interaction.tensor_product.plan.paths
        ),
    )
    plan = interaction.tensor_product.plan
    # Every static plan field is compared, not only the recorded plan_id.
    require_identical(plan, expected_plan, f"Layer {index} coupling plan", ValueError)
    if any(abs(path.path_scale) != 1.0 for path in plan.paths):
        raise ValueError("MACE coupling path scales are basis signs of magnitude one.")
    # The owner re-binds the executed tables, recomputing their support and
    # identity; the compute dtype of the tables is checked with the layer arrays.
    interaction.tensor_product.validate()
    if (
        interaction.kind != kind
        or interaction.input_layout.layout_id != inputs.layout_id
        or interaction.linear_up.out_layout.layout_id
        != architecture.edge_layout(index).layout_id
        or interaction.target_layout.layout_id != target.layout_id
    ):
        raise ValueError(f"Layer {index} interaction does not match the architecture.")
    _check_interaction(architecture, index, interaction, outputs)
    _check_product(architecture, index, layer.product, outputs, residual)
    _check_readouts(architecture, index, layer)
    if (layer.pair_repulsion is not None) != (index == 0 and architecture.pair_repulsion):
        raise ValueError(
            "ZBL repulsion belongs to the first layer exactly when declared."
        )
    pair = layer.pair_repulsion
    if pair is not None:
        # Reconstruction re-admits the realized constants against the universal
        # ZBL values; every executed field (atomic numbers, exponents, envelope,
        # unit factors) and the recomputed identity must equal what executes.
        expected = MACEPairRepulsion(
            architecture.species,
            scale,
            power=architecture.cutoff_power,
            dtype=dtype,
            covalent_radii=np.asarray(jax.device_get(pair.covalent_radii)),
            coefficients=np.asarray(jax.device_get(pair.coefficients)),
            screening=np.asarray(jax.device_get(pair.screening)),
        )
        require_identical(
            pair, expected, "ZBL constants of the model species and units", ValueError
        )
    if any(array.dtype != dtype for array in _layer_compute_arrays(layer)):
        raise TypeError(
            f"Layer {index} parameters and runtime tables must use the compute dtype."
        )


def _check_interaction(
    architecture: MACEArchitecture,
    index: int,
    interaction: MACEAnyInteraction,
    outputs: O3IrrepLayout,
    /,
) -> None:
    """Family-specific structure of one interaction against the declaration."""
    widths = mace_radial_widths(architecture, index)
    target = architecture.target_layout()
    match interaction:
        case MACEInteraction():
            residual, _ = interaction_traits(interaction.kind)
            connection = interaction.self_connection
            expected_out = outputs if residual else target
            valid = (
                interaction.radial.widths == widths
                and interaction.radial.activation_scale
                == architecture.radial_activation_scale
                and interaction.average_neighbor_count
                == architecture.average_neighbor_count
                and connection.out_layout.layout_id == expected_out.layout_id
                and connection.species_count == architecture.species_count
                and (
                    interaction.density is None
                    or interaction.density.activation_scale
                    == architecture.radial_activation_scale
                )
            )
        case MACEAttentionInteraction():
            valid = (
                interaction.radial.widths == widths
                and interaction.radial.activation_scale
                == architecture.radial_activation_scale
                and interaction.average_neighbor_count
                == architecture.average_neighbor_count
                and interaction.attention_down.out_layout.layout_id
                == architecture.layout(ATTENTION_PROJECTION_WIDTH, 0).layout_id
                and interaction.skip.out_layout.layout_id == outputs.layout_id
            )
        case MACENonlinearInteraction():
            gate = mace_gate(architecture)
            valid = (
                interaction.radial.widths == widths
                and interaction.density.widths == (widths[0], NONLINEAR_DENSITY_WIDTH, 1)
                and interaction.source_embedding.shape
                == (architecture.species_count, architecture.channel_count)
                and interaction.gate.target_layout.layout_id
                == gate.target_layout.layout_id
                and interaction.gate.scalar_scale == gate.scalar_scale
                and interaction.gate.gate_scale == gate.gate_scale
                and interaction.skip.out_layout.layout_id == outputs.layout_id
            )
        case unreachable:
            assert_never(unreachable)
    if not valid:
        raise ValueError(
            f"Layer {index} {interaction.kind} interaction structure differs."
        )


def _interaction_arrays(interaction: MACEAnyInteraction, /) -> tuple[Array, ...]:
    """Every parameter and fixed table of one interaction."""
    shared = (
        *interaction.linear_up.weights,
        *interaction.linear_down.weights,
        *interaction.tensor_product.prepared.coefficients,
    )
    match interaction:
        case MACEInteraction():
            return (
                *shared,
                *interaction.radial.weights,
                *interaction.self_connection.weights,
                *(() if interaction.density is None else interaction.density.weights),
            )
        case MACEAttentionInteraction():
            return (
                *shared,
                *interaction.attention_down.weights,
                *interaction.radial.weights,
                *interaction.skip.weights,
            )
        case MACENonlinearInteraction():
            return (
                *shared,
                interaction.source_embedding,
                interaction.target_embedding,
                *jax.tree_util.tree_leaves(interaction.radial),
                *jax.tree_util.tree_leaves(interaction.density),
                *interaction.residual_linear.weights,
                *interaction.output_linear.weights,
                *interaction.skip.weights,
                interaction.density_alpha,
                interaction.density_beta,
            )
        case unreachable:
            assert_never(unreachable)


def _readout_arrays(readout: MACEReadout | None, /) -> tuple[Array, ...]:
    match readout:
        case None:
            return ()
        case MACELinearReadout():
            return (readout.weight,)
        case MACENonlinearReadout():
            return (readout.first, readout.second)
        case MACENonlinearBiasReadout():
            return (
                readout.first,
                readout.middle,
                readout.middle_bias,
                readout.second,
                readout.bias,
            )
        case unreachable:
            assert_never(unreachable)


def _layer_compute_arrays(layer: MACELayer, /) -> tuple[Array, ...]:
    """Original parameters and explicit compute-dtype numerical tables."""
    arrays: list[Array] = [
        *_interaction_arrays(layer.interaction),
        *(weight for block in layer.product.contraction.weights for weight in block),
        *(
            basis.coefficients
            for block in layer.product.contraction.plan.bases
            for basis in block
        ),
        *layer.product.linear.weights,
        *_readout_arrays(layer.readout),
    ]
    product = layer.readout_product
    if product is not None:
        arrays.extend(weight for block in product.contraction.weights for weight in block)
        arrays.extend(
            basis.coefficients
            for block in product.contraction.plan.bases
            for basis in block
        )
        arrays.extend(product.linear.weights)
    pair = layer.pair_repulsion
    if pair is not None:
        arrays.extend(
            (
                pair.atomic_numbers,
                pair.covalent_radii,
                pair.coefficients,
                pair.exponents,
                pair.screening,
            )
        )
    return tuple(arrays)


def _check_product(
    architecture: MACEArchitecture,
    index: int,
    product: MACEProduct,
    outputs: O3IrrepLayout,
    residual: bool,
    /,
) -> None:
    plan = product.contraction.plan
    if (
        product.output_layout.layout_id != outputs.layout_id
        or plan.input_layout.layout_id
        != architecture.layout(1, architecture.edge_degree).layout_id
        or plan.output_layout.layout_id
        != architecture.layout(1, architecture.output_degree(index)).layout_id
        or plan.correlation != architecture.correlations[index]
        or plan.species_count != architecture.contraction_species_count
        or plan.channel_count != architecture.channel_count
        or product.self_connection != (index > 0 or residual)
    ):
        raise ValueError(f"Layer {index} product does not match the architecture.")
    # Fixed bases, product graph and term maps are re-admitted by their owner,
    # so the model's architecture identity never trusts a recorded plan_id.
    product.contraction.validate()


def _check_readouts(
    architecture: MACEArchitecture, index: int, layer: MACELayer, /
) -> None:
    heads = len(architecture.heads)
    channels = architecture.channel_count
    readout = layer.readout
    match architecture.readout_kind(index):
        case "none":
            valid = readout is None
        case "linear":
            valid = isinstance(readout, MACELinearReadout) and readout.weight.shape == (
                channels,
                heads,
            )
        case "nonlinear":
            expected = (
                MACENonlinearReadout
                if architecture.readout_variant == "nonlinear"
                else MACENonlinearBiasReadout
            )
            valid = (
                isinstance(readout, expected)
                and isinstance(readout, (MACENonlinearReadout, MACENonlinearBiasReadout))
                and readout.first.shape[0] == channels
                and readout.head_count == heads
                and readout.width == architecture.readout_width
                and readout.activation_scale == architecture.readout_activation_scale
            )
        case unreachable:
            assert_never(unreachable)
    if not valid:
        raise ValueError(f"Layer {index} readout does not match the architecture.")
    product = layer.readout_product
    if (product is None) != (architecture.readout_correlation is None):
        raise ValueError(
            "The invariant readout product is declared by readout_correlation."
        )
    if product is not None and (
        product.correlation != architecture.readout_correlation
        or product.contraction.plan.species_count
        != architecture.contraction_species_count
        or product.contraction.plan.channel_count != channels
        or product.channel_layout.layout_id
        != architecture.layout(1, architecture.hidden_degree).layout_id
        or product.contraction.plan.input_layout.layout_id
        != product.channel_layout.layout_id
        or product.contraction.plan.output_layout.layout_id
        != architecture.layout(1, 0).layout_id
    ):
        raise ValueError("The invariant readout product does not match the architecture.")
    if product is not None:
        product.contraction.validate()


__all__ = [
    "mace_contraction_plan",
    "mace_gate",
    "mace_node_feature_port",
    "mace_radial_basis",
    "mace_radial_widths",
    "mace_species_radii",
    "mace_tensor_product_plan",
    "MACEArchitecture",
    "MACEDistanceTransform",
    "MACELayerContext",
    "MACELayerUpdate",
    "MACENodeStates",
    "MACEPotential",
    "MACERadialBasisKind",
    "MACEReadoutVariant",
    "MACESpeciesTable",
    "NODE_FEATURES_REPRESENTATION",
]

register_artifact_value("phydrax.nn.atomistic:MACEArchitecture", MACEArchitecture)
register_artifact_value("phydrax.nn.atomistic:MACEPotential", MACEPotential)
register_artifact_value("phydrax.nn.atomistic:MACELayerUpdate", MACELayerUpdate)
register_artifact_value(
    "phydrax.nn.atomistic.internal:MACESpeciesTable", MACESpeciesTable
)
