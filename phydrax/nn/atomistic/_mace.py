#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native trainable MACE scalar-energy potential.

The architecture follows the standard real-agnostic MACE family exactly: a
species embedding, statically declared interaction layers (plain, residual,
density-normalized, or both), symmetric product bases with their original
species-conditioned weights ``W`` and fixed coupling bases ``U``, linear and
nonlinear invariant readouts, atomic reference energies with optional per-head
scale and shift, and optional ZBL repulsion. Every layer streams over the shared
prepared relation: edges evaluate radial weights, real harmonics and the
``uvu`` coupling; each receiver then runs its complete epilogue once.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import assert_never, final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._doc import DOC_KEY0
from ..._fingerprint import canonical_fingerprint
from ..._model import register_artifact_value
from ..._strict import StrictModule
from ..._trainable import NonTrainableState, partition_parameters
from ..._validation import (
    canonical_identifier,
    finite_real_scalar,
    nonnegative_integer,
    positive_finite_float,
    positive_integer,
)
from ...atomistic._graph import (
    AtomisticGraph,
    AtomisticGraphExecutionPlan,
    AtomisticGraphTopology,
    prepare_atomistic_graph_topology,
    realize_atomistic_graph,
)
from ...atomistic._potential import (
    AbstractAtomisticPotential,
    AtomisticPotentialCapabilities,
    AtomisticSpeciesKind,
)
from ...atomistic._types import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
)
from ...sparse._streamed import (
    PreparedStreamedRelation,
    StreamedPayloadSpec,
    StreamedRelationEvidence,
    StreamedRelationPlan,
    StreamedRelationResources,
)
from ...special import RealCartesianHarmonics
from ...typing import checked, Dim, Int32, parse, PRNGKey
from ..operator.layers import (
    O3IrrepLinear,
    O3TensorProduct,
    O3TensorProductPath,
    O3TensorProductPlan,
)
from ..operator.representations import O3IrrepLayout
from ._fixed_binding import require_identical
from ._mace_interaction import (
    admit_mace_coupling,
    convolution_paths,
    harmonic_layout,
    interaction_traits,
    mace_irrep_layout,
    MACECutoffPlacement,
    MACEEdgeGeometry,
    MACEInteraction,
    MACEInteractionKind,
    MACELayer,
    MACEProduct,
    MACEReceiverRows,
    MACESelfConnection,
    MACESourceRows,
    MACEStreamedLayer,
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
    MACENonlinearReadout,
    MACEPairRepulsion,
)
from ._radial import (
    AgnesiTransform,
    BesselRadialBasis,
    normalized_silu_scale,
    PolynomialCutoff,
    RadialEmbedding,
    RadialMLP,
)
from ._symmetric_contraction import SymmetricContraction, SymmetricContractionPlan


MACEDistanceTransform: TypeAlias = Literal["none", "agnesi"]

_METHOD_ID = "negative-position-gradient-of-total-mace-energy"


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


@final
class MACEArchitecture(StrictModule, NonTrainableState):
    """Immutable declaration of one MACE architecture.

    Layer ``i`` reads ``C x 0e`` (``i = 0``) or the hidden irreps
    ``C x (l, (-1)^l), l <= hidden_degree``, couples with real harmonics up to
    ``edge_degree`` into ``C x (l, (-1)^l), l <= edge_degree``, and its product
    emits the hidden irreps, except the last product, which emits ``C x 0e``.
    With ``readout_correlation`` (one interaction only), the single product
    emits the hidden irreps and the separate invariant readout product of that
    correlation feeds the nonlinear readout. ``readout_width`` is the per-head
    hidden width of the nonlinear readout and is ``None`` exactly when the only
    readout is linear (one interaction without ``readout_correlation``).
    """

    species: tuple[int, ...] = eqx.field(static=True)
    species_kind: AtomisticSpeciesKind = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    radial_basis_count: int = eqx.field(static=True)
    cutoff_power: int = eqx.field(static=True)
    distance_transform: MACEDistanceTransform = eqx.field(static=True)
    agnesi_parameters: tuple[float, float, float] | None = eqx.field(static=True)
    cutoff_placement: MACECutoffPlacement = eqx.field(static=True)
    channel_count: int = eqx.field(static=True)
    hidden_degree: int = eqx.field(static=True)
    edge_degree: int = eqx.field(static=True)
    interactions: tuple[MACEInteractionKind, ...] = eqx.field(static=True)
    correlations: tuple[int, ...] = eqx.field(static=True)
    radial_widths: tuple[int, ...] = eqx.field(static=True)
    readout_width: int | None = eqx.field(static=True)
    radial_activation_scale: float = eqx.field(static=True)
    readout_activation_scale: float | None = eqx.field(static=True)
    average_neighbor_count: float = eqx.field(static=True)
    heads: tuple[str, ...] = eqx.field(static=True)
    head: str = eqx.field(static=True)
    energy_scaling: MACEEnergyScaling = eqx.field(static=True)
    pair_repulsion: bool = eqx.field(static=True)
    readout_correlation: int | None = eqx.field(static=True)
    last_readout_only: bool = eqx.field(static=True)
    agnostic_product: bool = eqx.field(static=True)
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
        distance_transform: MACEDistanceTransform = "none",
        agnesi_parameters: tuple[float, float, float] | None = None,
        cutoff_placement: MACECutoffPlacement = "embedding",
        radial_activation_scale: float | None = None,
        readout_activation_scale: float | None = None,
        heads: Sequence[str] = ("Default",),
        head: str | None = None,
        energy_scaling: MACEEnergyScaling = "unscaled",
        pair_repulsion: bool = False,
        readout_correlation: int | None = None,
        last_readout_only: bool = False,
        agnostic_product: bool = False,
    ) -> None:
        fields = _architecture_fields(
            species=species,
            species_kind=species_kind,
            cutoff=cutoff,
            radial_basis_count=radial_basis_count,
            cutoff_power=cutoff_power,
            distance_transform=distance_transform,
            agnesi_parameters=agnesi_parameters,
            cutoff_placement=cutoff_placement,
            channel_count=channel_count,
            hidden_degree=hidden_degree,
            edge_degree=edge_degree,
            interactions=interactions,
            correlations=correlations,
            radial_widths=radial_widths,
            readout_width=readout_width,
            radial_activation_scale=radial_activation_scale,
            readout_activation_scale=readout_activation_scale,
            average_neighbor_count=average_neighbor_count,
            heads=heads,
            head=head,
            energy_scaling=energy_scaling,
            pair_repulsion=pair_repulsion,
            readout_correlation=readout_correlation,
            last_readout_only=last_readout_only,
            agnostic_product=agnostic_product,
        )
        self.species = fields.species
        self.species_kind = fields.species_kind
        self.cutoff = fields.cutoff
        self.radial_basis_count = fields.radial_basis_count
        self.cutoff_power = fields.cutoff_power
        self.distance_transform = fields.distance_transform
        self.agnesi_parameters = fields.agnesi_parameters
        self.cutoff_placement = fields.cutoff_placement
        self.channel_count = fields.channel_count
        self.hidden_degree = fields.hidden_degree
        self.edge_degree = fields.edge_degree
        self.interactions = fields.interactions
        self.correlations = fields.correlations
        self.radial_widths = fields.radial_widths
        self.readout_width = fields.readout_width
        self.radial_activation_scale = fields.radial_activation_scale
        self.readout_activation_scale = fields.readout_activation_scale
        self.average_neighbor_count = fields.average_neighbor_count
        self.heads = fields.heads
        self.head = fields.head
        self.energy_scaling = fields.energy_scaling
        self.pair_repulsion = fields.pair_repulsion
        self.readout_correlation = fields.readout_correlation
        self.last_readout_only = fields.last_readout_only
        self.agnostic_product = fields.agnostic_product
        self.architecture_id = _architecture_id(fields)

    def validate(self) -> None:
        """Re-run every declaration check, e.g. after constructor-bypassing restore."""
        fields = _architecture_fields(
            species=self.species,
            species_kind=self.species_kind,
            cutoff=self.cutoff,
            radial_basis_count=self.radial_basis_count,
            cutoff_power=self.cutoff_power,
            distance_transform=self.distance_transform,
            agnesi_parameters=self.agnesi_parameters,
            cutoff_placement=self.cutoff_placement,
            channel_count=self.channel_count,
            hidden_degree=self.hidden_degree,
            edge_degree=self.edge_degree,
            interactions=self.interactions,
            correlations=self.correlations,
            radial_widths=self.radial_widths,
            readout_width=self.readout_width,
            radial_activation_scale=self.radial_activation_scale,
            readout_activation_scale=self.readout_activation_scale,
            average_neighbor_count=self.average_neighbor_count,
            heads=self.heads,
            head=self.head,
            energy_scaling=self.energy_scaling,
            pair_repulsion=self.pair_repulsion,
            readout_correlation=self.readout_correlation,
            last_readout_only=self.last_readout_only,
            agnostic_product=self.agnostic_product,
        )
        if _architecture_id(fields) != self.architecture_id:
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

    def input_layout(self, index: int, /) -> O3IrrepLayout:
        degree = 0 if index == 0 else self.hidden_degree
        return mace_irrep_layout(self.channel_count, degree)

    def output_layout(self, index: int, /) -> O3IrrepLayout:
        last = index == self.interaction_count - 1
        degree = 0 if last and self.readout_correlation is None else self.hidden_degree
        return mace_irrep_layout(self.channel_count, degree)

    def target_layout(self) -> O3IrrepLayout:
        return mace_irrep_layout(self.channel_count, self.edge_degree)

    def harmonics_layout(self) -> O3IrrepLayout:
        return harmonic_layout(self.edge_degree)

    def message_payload_width(self) -> int:
        """Largest per-event message or per-receiver output element count."""
        widths = [1 + self.output_layout(0).packed_size]
        for index in range(self.interaction_count):
            message, _ = convolution_paths(
                self.input_layout(index), self.harmonics_layout(), self.target_layout()
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
    radial_basis_count: int
    cutoff_power: int
    distance_transform: MACEDistanceTransform
    agnesi_parameters: tuple[float, float, float] | None
    cutoff_placement: MACECutoffPlacement
    channel_count: int
    hidden_degree: int
    edge_degree: int
    interactions: tuple[MACEInteractionKind, ...]
    correlations: tuple[int, ...]
    radial_widths: tuple[int, ...]
    readout_width: int | None
    radial_activation_scale: float
    readout_activation_scale: float | None
    average_neighbor_count: float
    heads: tuple[str, ...]
    head: str
    energy_scaling: MACEEnergyScaling
    pair_repulsion: bool
    readout_correlation: int | None
    last_readout_only: bool
    agnostic_product: bool


def _architecture_fields(
    *,
    species: Sequence[int],
    species_kind: AtomisticSpeciesKind,
    cutoff: float,
    radial_basis_count: int,
    cutoff_power: int,
    distance_transform: MACEDistanceTransform,
    agnesi_parameters: tuple[float, float, float] | None,
    cutoff_placement: MACECutoffPlacement,
    channel_count: int,
    hidden_degree: int,
    edge_degree: int,
    interactions: Sequence[MACEInteractionKind],
    correlations: Sequence[int],
    radial_widths: Sequence[int],
    readout_width: int | None,
    radial_activation_scale: float | None,
    readout_activation_scale: float | None,
    average_neighbor_count: float,
    heads: Sequence[str],
    head: str | None,
    energy_scaling: MACEEnergyScaling,
    pair_repulsion: bool,
    readout_correlation: int | None,
    last_readout_only: bool,
    agnostic_product: bool,
) -> _ArchitectureFields:
    """Validate and canonicalize one architecture declaration."""
    if not isinstance(species_kind, AtomisticSpeciesKind):
        raise TypeError("species_kind must be AtomisticSpeciesKind.")
    species_ = _species(species, species_kind)
    transform = parse(distance_transform, MACEDistanceTransform, "distance_transform")
    placement = parse(cutoff_placement, MACECutoffPlacement, "cutoff_placement")
    scaling = parse(energy_scaling, MACEEnergyScaling, "energy_scaling")
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
    agnesi = _agnesi_parameters(transform, agnesi_parameters)
    if (transform == "agnesi" or pair_repulsion) and (
        species_kind is not AtomisticSpeciesKind.ATOMIC_NUMBER
    ):
        raise ValueError("Agnesi transforms and ZBL repulsion need atomic numbers.")
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
    return _ArchitectureFields(
        species=species_,
        species_kind=species_kind,
        cutoff=positive_finite_float(cutoff, "cutoff"),
        radial_basis_count=positive_integer(radial_basis_count, "radial_basis_count"),
        cutoff_power=positive_integer(cutoff_power, "cutoff_power"),
        distance_transform=transform,
        agnesi_parameters=agnesi,
        cutoff_placement=placement,
        channel_count=positive_integer(channel_count, "channel_count"),
        hidden_degree=hidden,
        edge_degree=edge,
        interactions=kinds,
        correlations=orders,
        radial_widths=widths,
        readout_width=None
        if readout_width is None
        else positive_integer(readout_width, "readout_width"),
        radial_activation_scale=radial_scale,
        readout_activation_scale=readout_scale,
        average_neighbor_count=positive_finite_float(
            average_neighbor_count, "average_neighbor_count"
        ),
        heads=names,
        head=selected,
        energy_scaling=scaling,
        pair_repulsion=_flag(pair_repulsion, "pair_repulsion"),
        readout_correlation=readout_order,
        last_readout_only=_flag(last_readout_only, "last_readout_only"),
        agnostic_product=_flag(agnostic_product, "agnostic_product"),
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


def _agnesi_parameters(
    transform: MACEDistanceTransform,
    parameters: tuple[float, float, float] | None,
    /,
) -> tuple[float, float, float] | None:
    match transform:
        case "none":
            if parameters is not None:
                raise ValueError("agnesi_parameters require distance_transform='agnesi'.")
            return None
        case "agnesi":
            if parameters is None or len(parameters) != 3:
                raise ValueError("The Agnesi transform declares (a, q, p).")
            a, q, p = (
                finite_real_scalar(value, "agnesi_parameters") for value in parameters
            )
            return (a, q, p)
        case unreachable:
            assert_never(unreachable)


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


def _native_geometry(
    architecture: MACEArchitecture, dtype: np.dtype, /
) -> MACEEdgeGeometry:
    basis = BesselRadialBasis.native(
        architecture.cutoff, architecture.radial_basis_count, dtype=dtype
    )
    cutoff = PolynomialCutoff(architecture.cutoff, architecture.cutoff_power)
    transform = None
    if architecture.agnesi_parameters is not None:
        a, q, p = architecture.agnesi_parameters
        transform = AgnesiTransform(
            covalent_radii_angstrom(architecture.species),
            atomic_numbers=architecture.species,
            a=a,
            q=q,
            p=p,
            dtype=dtype,
        )
    return MACEEdgeGeometry(
        RadialEmbedding(basis, cutoff, transform=transform),
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
        architecture.input_layout(index),
        architecture.harmonics_layout(),
        architecture.target_layout(),
    )
    scales = (1.0,) * len(paths) if path_scales is None else tuple(path_scales)
    if len(scales) != len(paths):
        raise ValueError("path_scales must give one scale per coupling path.")
    return O3TensorProductPlan(
        architecture.input_layout(index),
        architecture.harmonics_layout(),
        message,
        paths=tuple(
            O3TensorProductPath(
                left, right, output, connection_mode="uvu", path_scale=scale
            )
            for (left, right, output), scale in zip(paths, scales, strict=True)
        ),
    )


def _native_contraction_plan(
    architecture: MACEArchitecture,
    input_degree: int,
    output_degree: int,
    correlation: int,
    /,
) -> SymmetricContractionPlan:
    return SymmetricContractionPlan.native(
        mace_irrep_layout(1, input_degree),
        mace_irrep_layout(1, output_degree),
        correlation,
        species_count=architecture.contraction_species_count,
        channel_count=architecture.channel_count,
    )


def _native_layer(
    architecture: MACEArchitecture,
    scale: AtomisticScaleContract,
    index: int,
    dtype: np.dtype,
    key: PRNGKey,
    /,
) -> MACELayer:
    keys = jr.split(key, 10)
    kind = architecture.interactions[index]
    residual, density_normalized = interaction_traits(kind)
    inputs = architecture.input_layout(index)
    outputs = architecture.output_layout(index)
    target = architecture.target_layout()
    plan = mace_tensor_product_plan(architecture, index)
    message, _ = convolution_paths(inputs, architecture.harmonics_layout(), target)
    radial_widths = (
        architecture.radial_basis_count,
        *architecture.radial_widths,
        plan.parameter_count,
    )
    interaction = MACEInteraction(
        O3IrrepLinear(inputs, inputs, dtype=dtype, key=keys[0]),
        O3TensorProduct(plan, internal_weights=False, dtype=dtype),
        RadialMLP.initialize(
            radial_widths,
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
    product = MACEProduct(
        SymmetricContraction.initialize(
            _native_contraction_plan(
                architecture,
                architecture.edge_degree,
                outputs.blocks[-1].degree,
                architecture.correlations[index],
            ),
            key=keys[5],
            dtype=dtype,
        ),
        O3IrrepLinear(outputs, outputs, dtype=dtype, key=keys[6]),
        target,
        self_connection=index > 0 or residual,
    )
    return MACELayer(
        interaction,
        product,
        readout=_native_readout(architecture, index, dtype, keys[7]),
        readout_product=_native_readout_product(architecture, dtype, keys[8])
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
) -> MACELinearReadout | MACENonlinearReadout | None:
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
            return MACENonlinearReadout.initialize(
                architecture.channel_count,
                width,
                heads,
                activation_scale=scale,
                key=key,
                dtype=dtype,
            )
        case unreachable:
            assert_never(unreachable)


def _native_readout_product(
    architecture: MACEArchitecture, dtype: np.dtype, key: PRNGKey, /
) -> MACEInvariantReadoutProduct:
    correlation = architecture.readout_correlation
    if correlation is None:
        raise RuntimeError("The invariant readout product needs readout_correlation.")
    contraction_key, linear_key = jr.split(key)
    channel_layout = mace_irrep_layout(1, architecture.hidden_degree)
    scalars = mace_irrep_layout(architecture.channel_count, 0)
    return MACEInvariantReadoutProduct(
        SymmetricContraction.initialize(
            _native_contraction_plan(
                architecture, architecture.hidden_degree, 0, correlation
            ),
            key=contraction_key,
            dtype=dtype,
        ),
        O3IrrepLinear(scalars, scalars, dtype=dtype, key=linear_key),
        channel_layout,
    )


@final
class MACEPotential(AbstractAtomisticPotential):
    """Native trainable MACE scalar-energy potential.

    Construct with only ``key`` for the source family's random initialization,
    or with all of ``embedding``, ``geometry`` and ``layers`` (as the trusted
    source reconstruction does); both routes run the same owning validators.
    ``streaming`` is an explicit streamed-relation tiling; ``None`` reuses the
    schedule each bound graph topology already prepared for its epoch.

    ``acceleration`` is an execution policy, not architecture: an admitted
    accelerated coupling kernel aggregates each streamed fragment's coupling
    from fragment-local radial weights and harmonics (density and ZBL use the
    substrate's seeded reduction over the same lanes), and the receiver
    epilogue still runs once per receiver after its final fragment. It never
    enters the architecture identity or numeric revision; every parameter keeps
    its source role, so the accelerated route is trainable.
    """

    embedding: Array
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
            keys = jr.split(key, architecture.interaction_count + 1)
            embedding_ = jr.normal(
                keys[0],
                (architecture.species_count, architecture.channel_count),
                dtype=dtype,
            )
            geometry_ = _native_geometry(architecture, dtype)
            layers_ = tuple(
                _native_layer(architecture, scale, index, dtype, keys[index + 1])
                for index in range(architecture.interaction_count)
            )
        else:
            embedding_ = jnp.asarray(embedding, dtype=dtype)
            geometry_ = geometry
            layers_ = tuple(layers)
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
        _check_components(architecture, scale, dtype, embedding_, geometry_, layers_)
        _check_acceleration(acceleration, layers_, dtype)
        self.embedding = embedding_
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
            architecture, scale, precision_, reference, geometry_, layers_
        )
        self.method_id = _METHOD_ID

    def validate(self) -> None:
        """Run every owning validator on this (possibly restored) potential.

        This is a host boundary over concrete arrays. It re-checks the
        architecture declaration, layouts, coupling plans and their executed
        coefficient tables and support, symmetric-contraction bases, product
        graphs and term maps (finiteness, equivariance, source evidence and
        recomputed identities), every fixed radial field (Bessel frequencies and
        prefactor, cutoff, Agnesi radii and constants, harmonics) and ZBL field
        rebuilt from the stored arrays, parameter shapes, dtypes and finiteness,
        reference energies, species lookup, and the recorded scientific identity.
        """
        architecture = self.configuration
        architecture.validate()
        dtype = _compute_dtype(self.precision)
        _check_components(
            architecture, self.scale, dtype, self.embedding, self.geometry, self.layers
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
    def interaction_count(self) -> int:
        return self.configuration.interaction_count

    def species_indices(self, species_ids: ArrayLike, atom_mask: ArrayLike, /) -> Array:
        """Model species indices of active particles (zero on inactive slots)."""
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_)
        return self.species_table.indices(jnp.asarray(species_ids), mask)

    def init_features(self, species: Array, atom_mask: Array, /) -> Array:
        """Layer-0 node states ``C x 0e``: the normalized species embedding."""
        dtype = self.embedding.dtype
        fan_in = jnp.sqrt(jnp.asarray(self.embedding.shape[0], dtype=dtype))
        features = self.embedding[species] / fan_in
        return jnp.where(atom_mask[:, None], features, jnp.zeros((), dtype=dtype))

    def prepare_sources(
        self, index: int, features: Array, species: Array, atom_mask: Array, /
    ) -> MACESourceRows:
        """Node-local ``linear_up`` source states of one layer."""
        lifted = self.layers[index].interaction.linear_up(features)
        zero = jnp.zeros((), dtype=lifted.dtype)
        return MACESourceRows(jnp.where(atom_mask[:, None], lifted, zero), species)

    def update_receivers(
        self,
        index: int,
        relation: PreparedStreamedRelation,
        vectors: Array,
        sources: MACESourceRows,
        features: Array,
        species: Array,
        atom_mask: Array,
        /,
        *,
        edge_active: Array | None = None,
    ) -> MACELayerUpdate:
        """Stream one layer: every receiver's complete update and readout energy.

        ``vectors`` are edge displacements ``r_receiver - r_source (+ n H)`` in
        route order; the relation's sources are ``sources`` rows and its
        receivers the rows of ``features``. Energies are unscaled per-node
        readout terms (the first layer also carries ZBL).
        """
        dtype = self.embedding.dtype
        streamed = MACEStreamedLayer(
            self.layers[index],
            self.geometry,
            self.energy_reference.head,
            coupling=self.acceleration,
        )
        message, output = streamed.payload_shapes(dtype)
        payload = StreamedPayloadSpec(message, output)
        if self.acceleration is None:
            result = relation.evaluate(
                payload,
                MACEStreamedLayer.edge,
                MACEStreamedLayer.epilogue,
                streamed,
                sources,
                MACEReceiverRows(features, species, atom_mask),
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
                MACEReceiverRows(features, species, atom_mask),
                jnp.asarray(vectors).astype(dtype),
                edge_active=edge_active,
            )
        outputs = result.receiver_outputs
        kernel_resources = None
        if self.acceleration is not None:
            schedule = relation.schedule
            kernel_resources = self.acceleration.resources(
                MACEEdgeCouplingSpec(self.layers[index].interaction.tensor_product),
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
        self, species: Array, atom_mask: Array, layer_energies: Array, /
    ) -> Array:
        """Per-atom energy ``E0 + scale * sum(layer terms) + shift`` (zero if masked)."""
        reduction = jnp.dtype(self.precision.reduction_dtype)
        interaction = jnp.sum(layer_energies.astype(reduction), axis=0)
        energy = self.energy_reference.atom_energy(species, interaction)
        return jnp.where(atom_mask, energy, jnp.zeros((), dtype=reduction))

    def node_energies(
        self,
        species: Array,
        atom_mask: Array,
        relation: PreparedStreamedRelation,
        vectors: Array,
        /,
        *,
        edge_active: Array | None = None,
    ) -> tuple[Array, Array]:
        """Per-atom energies and the conjunction of every layer's success evidence."""
        features = self.init_features(species, atom_mask)
        energies = []
        successful = jnp.asarray(True)
        for index in range(self.interaction_count):
            sources = self.prepare_sources(index, features, species, atom_mask)
            update = self.update_receivers(
                index,
                relation,
                vectors,
                sources,
                features,
                species,
                atom_mask,
                edge_active=edge_active,
            )
            features = update.features
            energies.append(update.energy)
            successful = successful & update.evidence.successful
        return self.atom_energies(species, atom_mask, jnp.stack(energies)), successful

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
        """Case energies and per-atom energies on one bound atomistic graph.

        The topology's epoch-bound streamed schedule is reused for every layer
        (or prepared once for an explicit ``streaming`` plan); the cutoff mask
        is a runtime route mask. Failed streamed schedules poison their outputs,
        so an unsuccessful evaluation is non-finite rather than plausible.
        """
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_).reshape((-1,))
        species = self.species_indices(jnp.asarray(species_ids).reshape((-1,)), mask)
        ir = graph.graph
        if ir.edge_mask is None:
            raise ValueError("MACE requires an explicit masked edge relation.")
        topology = graph.topology
        relation = (
            topology.streamed
            if self.streaming is None
            else topology.prepare_streamed(self.streaming)
        )
        atom_energy, _ = self.node_energies(
            species,
            mask,
            relation,
            jnp.asarray(ir.edges["displacement"]),
            edge_active=jnp.asarray(ir.edge_mask, dtype=jnp.bool_),
        )
        reduction = jnp.dtype(self.precision.reduction_dtype)
        total = (
            jnp.zeros((case_count,), dtype=reduction)
            .at[jnp.asarray(atom_cases).reshape((-1,))]
            .add(atom_energy)
        )
        output = jnp.dtype(self.precision.output_dtype)
        return (
            total.astype(output),
            atom_energy.reshape((case_count, atom_capacity)).astype(output),
        )

    def _energy_unchecked(
        self,
        batch: AtomisticBatch,
        positions: Array,
        execution: AtomisticGraphExecutionPlan,
        /,
        *,
        topology: AtomisticGraphTopology | None = None,
        cell_vectors: Array | None = None,
    ) -> tuple[Array, Array, AtomisticGraph]:
        coordinate = jnp.asarray(positions, dtype=self.precision.coordinate_dtype)
        species = self._batch_species(batch)
        graph = realize_atomistic_graph(
            batch,
            execution,
            cutoff=self.configuration.cutoff,
            positions=coordinate,
            topology=topology,
            cell_vectors=cell_vectors,
        )
        energy, atom_energy = self.graph_energy(
            species,
            batch.atom_mask,
            batch.atom_cases,
            batch.case_count,
            batch.atom_capacity,
            graph,
        )
        return energy, atom_energy, graph

    def _batch_species(self, batch: AtomisticBatch, /) -> Array:
        match self.configuration.species_kind:
            case AtomisticSpeciesKind.ATOMIC_NUMBER:
                return eqx.error_if(
                    batch.atomic_numbers,
                    jnp.any(batch.atom_mask & ~batch.element_mask),
                    "Atomic-number MACE cannot evaluate non-element particles.",
                )
            case AtomisticSpeciesKind.ATOM_TYPE_ID:
                return batch.atom_type_ids
            case unreachable:
                assert_never(unreachable)

    def energy(
        self,
        batch: AtomisticBatch,
        execution: AtomisticGraphExecutionPlan,
        /,
        *,
        positions: Array | None = None,
        topology: AtomisticGraphTopology | None = None,
    ) -> Array:
        """Evaluate typed-scale total energies, failing closed on graph overflow.

        Periodic batches without ``topology`` prepare their image topology once
        on the host from the batch's concrete reference geometry.
        """
        self._validate_batch(batch)
        coordinate = batch.positions if positions is None else positions
        prepared = (
            prepare_atomistic_graph_topology(
                batch, execution, cutoff=self.configuration.cutoff
            )
            if topology is None and batch.has_periodic_metadata
            else topology
        )
        energy, _, graph = self._energy_unchecked(
            batch, coordinate, execution, topology=prepared
        )
        return graph.require_success(energy)

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
    /,
) -> str:
    """Scientific identity of everything outside the PARAMETER revision.

    Fixed coupling coefficients (including imported path signs), coupling bases
    ``U``, radial frequencies and transforms, and reference energies change the
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
    geometry: MACEEdgeGeometry,
    layers: Sequence[MACELayer],
    /,
) -> None:
    """Owning validation shared by construction, reconstruction and restore."""
    if embedding.shape != (architecture.species_count, architecture.channel_count):
        raise ValueError("The species embedding must have shape (species, channels).")
    if embedding.dtype != dtype:
        raise TypeError("The species embedding must use the compute dtype.")
    _check_geometry(architecture, geometry, dtype)
    if len(layers) != architecture.interaction_count:
        raise ValueError("MACE declares one layer per interaction.")
    for index, layer in enumerate(layers):
        if not isinstance(layer, MACELayer):
            raise TypeError("layers must contain MACELayer values.")
        _check_layer(architecture, scale, index, layer, dtype)


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
    if (
        radial.radius != architecture.cutoff
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
    match architecture.agnesi_parameters:
        case None:
            if transform is not None:
                raise ValueError("The architecture declares no distance transform.")
        case (a, q, p):
            if transform is None or (transform.a, transform.q, transform.p) != (a, q, p):
                raise ValueError("The Agnesi transform does not match its declaration.")
            if transform.atomic_numbers != architecture.species:
                raise ValueError("Agnesi radii must follow the model species order.")
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
        or interaction.target_layout.layout_id != target.layout_id
        or interaction.average_neighbor_count != architecture.average_neighbor_count
    ):
        raise ValueError(f"Layer {index} interaction does not match the architecture.")
    expected_widths = (
        architecture.radial_basis_count,
        *architecture.radial_widths,
        plan.parameter_count,
    )
    if (
        interaction.radial.widths != expected_widths
        or interaction.radial.activation_scale != architecture.radial_activation_scale
    ):
        raise ValueError(f"Layer {index} radial network does not match the architecture.")
    connection = interaction.self_connection
    expected_out = outputs if residual else target
    if (
        connection.out_layout.layout_id != expected_out.layout_id
        or connection.species_count != architecture.species_count
    ):
        raise ValueError(
            f"Layer {index} self-connection does not match the architecture."
        )
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


def _layer_compute_arrays(layer: MACELayer, /) -> tuple[Array, ...]:
    """Original parameters and the runtime numerical tables this model owns.

    Fixed symmetric-contraction coupling bases are excluded deliberately: their
    owner stores them host-exact in float64 (their identity is that exact table)
    and converts them explicitly to the feature dtype at evaluation.
    """
    interaction = layer.interaction
    arrays: list[Array] = [
        *interaction.linear_up.weights,
        *interaction.linear_down.weights,
        *interaction.radial.weights,
        *interaction.self_connection.weights,
        *interaction.tensor_product.prepared.coefficients,
        *(() if interaction.density is None else interaction.density.weights),
        *(weight for block in layer.product.contraction.weights for weight in block),
        *layer.product.linear.weights,
    ]
    match layer.readout:
        case None:
            pass
        case MACELinearReadout():
            arrays.append(layer.readout.weight)
        case MACENonlinearReadout():
            arrays.extend((layer.readout.first, layer.readout.second))
        case unreachable:
            assert_never(unreachable)
    product = layer.readout_product
    if product is not None:
        arrays.extend(weight for block in product.contraction.weights for weight in block)
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
        != mace_irrep_layout(1, architecture.edge_degree).layout_id
        or plan.output_layout.layout_id
        != mace_irrep_layout(1, outputs.blocks[-1].degree).layout_id
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
            valid = (
                isinstance(readout, MACENonlinearReadout)
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
        != mace_irrep_layout(1, architecture.hidden_degree).layout_id
        or product.contraction.plan.input_layout.layout_id
        != product.channel_layout.layout_id
        or product.contraction.plan.output_layout.layout_id
        != mace_irrep_layout(1, 0).layout_id
    ):
        raise ValueError("The invariant readout product does not match the architecture.")
    if product is not None:
        product.contraction.validate()


__all__ = [
    "MACEArchitecture",
    "MACEDistanceTransform",
    "MACELayerUpdate",
    "MACEPotential",
    "MACESpeciesTable",
    "mace_tensor_product_plan",
]

register_artifact_value("phydrax.nn.atomistic:MACEArchitecture", MACEArchitecture)
register_artifact_value("phydrax.nn.atomistic:MACEPotential", MACEPotential)
register_artifact_value("phydrax.nn.atomistic:MACELayerUpdate", MACELayerUpdate)
register_artifact_value(
    "phydrax.nn.atomistic.internal:MACESpeciesTable", MACESpeciesTable
)
