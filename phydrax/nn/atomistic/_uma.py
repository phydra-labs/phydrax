#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native UMA/eSCN-MD scalar-energy potential (fairchem ``bb0a062`` algorithms).

Forward pass of one bound graph, per case ``b`` with task ``t_b``:

1. ``csd_b = silu(mix_csd([charge_b, spin_b, dataset_t]))``; with experts,
   ``c_b = route(composition_b, csd_b)`` and every mixture linear becomes the
   mixed operator ``sum_e c_be W_e`` (never an ensemble of predictions).
2. ``x_i[l=0] = sphere_embedding[Z_i] + csd_b``, other degrees zero.
3. Edge-degree embedding: every edge ``i <- j`` (source vector ``v = r_j - r_i``
   plus image) adds ``D(v)^T[:, m=0] radial(x_edge) env(|v|/r_c) / 5`` with
   ``x_edge = [gauss(|v|), source_emb[Z_j], target_emb[Z_i]]``.
4. Each block: ``h = norm_1(x)`` with ``csd`` added to ``l = 0``; ``x <- x +
   sum_j D^T env SO2_2(act(SO2_1(D [h_j, h_i], x_edge)))``; ``x <- x +
   atomwise(norm_2(x))``; then declared charge/spin channel balancing.
5. ``x <- norm(x)``; per-atom raw energy ``e_i = head(x_i[l=0])``.
6. ``E_b = rmsd_t sum_i e_i + mean_t + sum_i ref_t[Z_i]`` (the source task
   normalizer and element references, in the reduction dtype).

Forces and stress are the framework's conservative derivatives of ``E``.
Source direct force/stress heads are separate typed ``DIRECT_FORCES`` /
``DIRECT_STRESS`` outputs and never relabelled as conservative responses.

Graph selection is the source's scientific nearest-neighbour rule (fairchem
``radius_graph_pbc_v2`` + ``get_max_neighbors_mask``): squared distance at
most ``cutoff^2``, nonzero, then per receiver the ``k`` nearest plus every
route within ``0.01 A^2`` of the ``(k+1)``-th (or exactly ``k`` when strict).
It is a declared requirement (`UMAPotential.requirements`), not a buffer
limit. Evaluation requires a graph whose canonical
``PreparedImageNeighborSelection`` binding is exactly this policy; its
scientific and capacity failures reach the case status through the graph.

Layered owner-region execution (`UMAPotential` implements the domain
``AtomisticLayeredModel`` hooks) runs the same functions on owned rows:
interaction ``0`` is the edge-degree embedding (sources exchange species),
interaction ``k + 1`` is block ``k`` (sources exchange ``norm_1(x) + csd``;
receivers convolve on the owner relation, then apply the node-local
atomwise update). System stages carry every per-case sum: the prologue
combines owner atom and species counts, routes the experts and realizes the
blocks; each block is followed by charge/spin channel balancing from the
combined channel sums. The task-normalizer mean is a per-case energy outside
every atom energy (`UMAPotential.system_energy`). The nearest-neighbour
selection is the domain role's, applied to the owner relation's edges. The
keyed roll gauge draws each owner edge's roll from the evaluation key and the
edge's canonical route identity, so rolls do not depend on the owner layout.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._differentiation import (
    DerivativeContract,
    DerivativeRegularity,
    DerivativeRoute,
    DerivativeSurface,
    GradientLevel,
    SurfaceDerivative,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._identity import IdentityWords, NumericRevision
from ..._model import register_artifact_value
from ..._model._fixed_binding import require_identical
from ..._model._ports import ValuePort
from ..._strict import StrictModule
from ..._trainable import (
    ExplicitFreeze,
    NonTrainableState,
    ParameterOwner,
    partition_parameters,
)
from ..._validation import (
    nonnegative_integer,
    positive_finite_float,
    positive_integer,
)
from ...atomistic._conditioning import (
    AtomisticConditioning,
    AtomisticConditionLevel,
    AtomisticModelOutputs,
    AtomisticObservable,
    AtomisticObservableRequest,
    AtomisticObservableValue,
)
from ...atomistic._feature_execution import AtomisticSystemStage
from ...atomistic._graph import AtomisticGraphExecutionPlan, case_energy_totals
from ...atomistic._hessian import (
    atomistic_edge_distance_locus,
    ATOMISTIC_NONCOINCIDENT_SUPPORT,
)
from ...atomistic._model_inputs import AtomisticGraphInputs
from ...atomistic._neighborhood_bundle import (
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
)
from ...atomistic._potential import (
    AbstractAtomisticModel,
    AbstractAtomisticPotential,
    AbstractPreparedAtomisticModel,
    atomistic_model_revision,
    AtomisticPotentialCapabilities,
    AtomisticPotentialRequirements,
)
from ...atomistic._types import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
    AtomisticSpeciesKind,
)
from ...discretization.particle._image_neighborhood import ImageNeighborSelectionPolicy
from ...sparse._streamed import (
    PreparedStreamedRelation,
    StreamedPayloadSpec,
    StreamedRelationEvidence,
    StreamedRelationPlan,
    StreamedRelationResult,
)
from ...typing import Bool, checked, Dim, Float, Identifier, parse, PRNGKey
from ...units import ANGSTROM, derived_unit, ELECTRONVOLT, ENERGY, LENGTH
from ._uma_conditioning import (
    apply_channel_balance,
    balance_channels,
    case_atom_counts,
    case_reduce,
    case_sums,
    channel_balance_correction,
    species_counts,
    UMA_CHARGE_FIELD,
    UMA_SPIN_FIELD,
    UMA_TASK_FIELD,
    UMAChargeSpinEmbeddingKind,
    UMACoefficientNorm,
    UMACompositionPolicy,
    UMAConditionSchema,
    UMAFrozen,
    UMAMoLEBinding,
    UMAMoLERouter,
    UMASystemEmbedding,
    UMATaskDim,
)
from ._uma_layers import (
    apply_linear,
    bind_shared_mixtures,
    merge_mixtures,
    mixture_linears,
    polynomial_envelope,
    resolve_mixtures,
    UMABlock,
    UMAEdgeActivation,
    UMAEdgeDegreeEmbedding,
    UMAFeedForward,
    UMAGaussianSmearing,
    UMAGridAtomwise,
    UMALinear,
    UMALinearMap,
    UMAMixtureLinear,
    UMANormalization,
    UMANormalizationKind,
    UMASharedMixture,
    UMASO3Linear,
)
from ._uma_so3 import (
    uma_edge_wigner,
    uma_keyed_edge_rolls,
    UMACoefficientLayout,
    UMAEdgeRollGauge,
)


# Source mole_type values with an executable meaning: SO(2) linears of every
# order, or only the m = 0 linear ("all"/"notso2" cannot run in the source).
UMAMixtureKind: TypeAlias = Literal["so2", "so2m0"]
UMAEnergyHeadKind: TypeAlias = Literal["mlp", "linear"]
UMAHeadWrapper: TypeAlias = Literal["shared", "dataset-experts"]
UMAForceRoute: TypeAlias = Literal["conservative", "direct"]
UMAStressReduce: TypeAlias = Literal["sum", "mean"]
UMAIsolatedAtomRoute: TypeAlias = Literal["network", "external-reference-lookup"]
UMARollRealization: TypeAlias = Literal[
    "roll-invariant",
    "fixed-gauge-s2-quadrature-dependent",
    "keyed-route-uniform-roll-s2",
]
# "sum": the potential is the summed head (source reduce="sum"). "mean": the
# source reports sum / n as its task energy but differentiates the unreduced
# sum for forces and stress; the potential is that differentiated sum.
UMAEnergyReduction: TypeAlias = Literal["sum", "mean"]

# "potential": a scalar potential exists (UMAPotential). "property": mean-reduced
# source heads without gradient forces only report properties (UMAPropertyModel).
UMAModelKind: TypeAlias = Literal["potential", "property"]
# Property models have no force method: their outputs are source reports.
_PROPERTY_METHOD_ID = "uma-source-reported-energy-and-direct-outputs-without-potential"
_METHOD_ID = "negative-position-gradient-of-total-uma-energy"
# Mean-reduced heads: the force-generating potential is rmsd * sum_i e_i, the
# function the source differentiates; the source task energy is a separate report.
_MEAN_METHOD_ID = (
    "negative-position-gradient-of-uma-unreduced-head-sum-potential"
    ":source-task-energy-is-reported-energy"
)


def _method_id(architecture: UMAArchitecture, /) -> str:
    """Force method: the potential's definition and any roll-dependent realization."""
    if architecture.model_kind == "property":
        base = _PROPERTY_METHOD_ID
    elif architecture.energy_reduction == "sum":
        base = _METHOD_ID
    else:
        base = _MEAN_METHOD_ID
    if architecture.roll_realization == "roll-invariant":
        return base
    return f"{base}:edge-roll-gauge={architecture.roll_gauge}"


# Source constants: edge-degree aggregation rescale ("sqrt avg degree") and
# envelope exponent, fixed in eSCNMDBackbone.__init__; tie band in A^2.
UMA_EDGE_DEGREE_RESCALE = 5.0
UMA_ENVELOPE_EXPONENT = 5
UMA_DEGENERACY_TOLERANCE = 0.01

_FORCE_UNIT = derived_unit("eV/angstrom", ((ELECTRONVOLT, 1), (ANGSTROM, -1)))
_STRESS_UNIT = derived_unit("eV/angstrom^3", ((ELECTRONVOLT, 1), (ANGSTROM, -3)))
UMA_DIRECT_FORCE_PORT = ValuePort(
    "atomistic.direct-force",
    event_shape=(3,),
    component_ids=("x", "y", "z"),
    representation="cartesian-vector-direct-prediction",
    dimensions=(ENERGY / LENGTH,) * 3,
    normalization_id="identity",
)
UMA_DIRECT_STRESS_PORT = ValuePort(
    "atomistic.direct-stress",
    event_shape=(3, 3),
    component_ids=("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"),
    representation="cartesian-rank2-direct-prediction",
    dimensions=(ENERGY / LENGTH**3,) * 9,
    normalization_id="identity",
)
UMA_REPORTED_ENERGY_PORT = ValuePort(
    "uma.reported-task-energy",
    event_shape=(),
    component_ids=("energy",),
    representation=(
        "fairchem-mean-reduced-task-energy:rmsd_t*sum_i(e_i)/n+mean_t+sum_i(ref_t[Z_i])"
    ),
    dimensions=(ENERGY,),
    normalization_id="uma-task-normalizer-then-element-references",
)
# fairchem cg_change_mat(2): rows are the (l=0, l=1, l=2) components.
_CG_CHANGE_RANK_TWO = np.array(
    [
        [3**-0.5, 0, 0, 0, 3**-0.5, 0, 0, 0, 3**-0.5],
        [0, 0, 0, 0, 0, 2**-0.5, 0, -(2**-0.5), 0],
        [0, 0, -(2**-0.5), 0, 0, 0, 2**-0.5, 0, 0],
        [0, 2**-0.5, 0, -(2**-0.5), 0, 0, 0, 0, 0],
        [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
        [0, 2**-0.5, 0, 2**-0.5, 0, 0, 0, 0, 0],
        [-(6**-0.5), 0, 0, 0, 2 * 6**-0.5, 0, 0, 0, -(6**-0.5)],
        [0, 0, 0, 0, 0, 2**-0.5, 0, 2**-0.5, 0],
        [-(2**-0.5), 0, 0, 0, 0, 0, 0, 0, 2**-0.5],
    ],
    dtype=np.float64,
)


def _channel_range(value: Sequence[int], name: str, /) -> tuple[int, int]:
    channels = sorted(nonnegative_integer(item, name) for item in value)
    if not channels:
        return (0, 0)
    if channels != list(range(channels[0], channels[-1] + 1)):
        raise ValueError(f"{name} must be contiguous.")
    return (channels[0], channels[-1] + 1)


def _admit_tasks(
    tasks: Sequence[str], dataset_mapping: Sequence[tuple[str, str]] | None, /
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...] | None]:
    """Sorted unique task names and the source dataset map keyed by exactly them."""
    task_names = tuple(sorted(str(task) for task in tasks))
    if not task_names or len(set(task_names)) != len(task_names):
        raise ValueError("UMA tasks must be unique and nonempty.")
    if dataset_mapping is None:
        return task_names, None
    mapping = tuple((str(key), str(value)) for key, value in dataset_mapping)
    keys = [key for key, _ in mapping]
    if (
        len(set(keys)) != len(keys)
        or not {value for _, value in mapping} <= set(keys)
        or tuple(sorted(keys)) != task_names
    ):
        raise ValueError(
            "dataset_mapping keys must be the tasks and its values a subset."
        )
    return task_names, mapping


def _balanced_channels(
    charge_channels: Sequence[int],
    spin_channels: Sequence[int],
    sphere_channels: int,
    /,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Contiguous charge and spin balancing ranges inside the sphere channels."""
    charge = _channel_range(charge_channels, "charge_channels")
    spin = _channel_range(spin_channels, "spin_channels")
    for _, end in (charge, spin):
        if end > sphere_channels:
            raise ValueError("Balanced channels must lie within sphere_channels.")
    return charge, spin


def _mixture_layers(
    mixture_layers: Sequence[int] | None, layer_count: int, experts: int, /
) -> tuple[int, ...]:
    """Mixed blocks (every block by default); none are mixed without experts."""
    layers = (
        tuple(range(layer_count))
        if mixture_layers is None
        else tuple(
            sorted({nonnegative_integer(i, "mixture layer") for i in mixture_layers})
        )
    )
    if any(layer >= layer_count for layer in layers):
        raise ValueError("Mixture layers must name existing blocks.")
    return layers if experts else ()


def _head_routing(
    head_wrapper: UMAHeadWrapper,
    head_experts: Sequence[int],
    energy_head_of_task: Sequence[int] | None,
    task_count: int,
    /,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Dataset expert and energy head of every task code."""
    experts = tuple(nonnegative_integer(i, "head expert") for i in head_experts)
    if (head_wrapper == "dataset-experts") != bool(experts) or (
        experts and len(experts) != task_count
    ):
        raise ValueError("Dataset-expert heads map every task code to one expert.")
    head_of_task = (
        (0,) * task_count
        if energy_head_of_task is None
        else tuple(nonnegative_integer(i, "energy head") for i in energy_head_of_task)
    )
    if len(head_of_task) != task_count:
        raise ValueError("energy_head_of_task names one energy head per task code.")
    return experts, head_of_task


def _roll_realization(
    edge_activation: UMAEdgeActivation, roll_gauge: UMAEdgeRollGauge, /
) -> UMARollRealization:
    """The edge-roll realization the activation and gauge define, or a refusal."""
    # Gate activations commute with edge rolls, so a fixed gauge realizes the
    # source function exactly; the S2 grid activation is roll-invariant only up
    # to quadrature and the source draws a fresh uniform roll per edge per
    # forward pass, so the fixed gauge is a distinct deterministic realization.
    if roll_gauge == "keyed-route-uniform-roll" and edge_activation == "gate":
        raise ValueError(
            "Gate-activated UMA is roll invariant; a keyed roll gauge would only "
            "add a key requirement and is not admitted."
        )
    if edge_activation == "gate":
        return "roll-invariant"
    if roll_gauge == "keyed-route-uniform-roll":
        return "keyed-route-uniform-roll-s2"
    return "fixed-gauge-s2-quadrature-dependent"


# Declared fields of the architecture fingerprint, in record order.
_ARCHITECTURE_RECORD_FIELDS = (
    "max_elements",
    "sphere_channels",
    "lmax",
    "mmax",
    "grid_resolution",
    "cutoff",
    "max_neighbors",
    "strict_neighbors",
    "edge_channels",
    "distance_basis",
    "layer_count",
    "hidden_channels",
    "normalization",
    "edge_activation",
    "feed_forward",
    "charge_spin_embedding",
    "charge_spin_trainable",
    "dataset_trainable",
    "dataset_mapping",
    "tasks",
    "charge_channels",
    "spin_channels",
    "experts",
    "mixture_kind",
    "mixture_layers",
    "mixture_sharing",
    "composition_embedding",
    "composition_policy",
    "coefficient_norm",
    "energy_head",
    "energy_reduction",
    "head_wrapper",
    "head_experts",
    "energy_head_of_task",
    "force_route",
    "direct_force_head",
    "direct_stress_head",
    "isolated_atoms",
    "model_kind",
    "roll_gauge",
    "roll_realization",
)


@final
class UMAArchitecture(StrictModule, NonTrainableState):
    """Complete static UMA/eSCN-MD declaration.

    Every field is a source configuration value or a named scientific policy;
    execution-only source settings (checkpointing, chunking, backends, Wigner
    kernels) are not part of the architecture.
    """

    __strict_contract__ = True

    max_elements: int = eqx.field(static=True)
    sphere_channels: int = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    mmax: int = eqx.field(static=True)
    grid_resolution: int | None = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    max_neighbors: int = eqx.field(static=True)
    strict_neighbors: bool = eqx.field(static=True)
    edge_channels: int = eqx.field(static=True)
    distance_basis: int = eqx.field(static=True)
    layer_count: int = eqx.field(static=True)
    hidden_channels: int = eqx.field(static=True)
    normalization: UMANormalizationKind = eqx.field(static=True)
    edge_activation: UMAEdgeActivation = eqx.field(static=True)
    feed_forward: UMAFeedForward = eqx.field(static=True)
    charge_spin_embedding: UMAChargeSpinEmbeddingKind = eqx.field(static=True)
    charge_spin_trainable: bool = eqx.field(static=True)
    dataset_trainable: bool = eqx.field(static=True)
    dataset_mapping: tuple[tuple[str, str], ...] | None = eqx.field(static=True)
    tasks: tuple[str, ...] = eqx.field(static=True)
    charge_channels: tuple[int, int] = eqx.field(static=True)
    spin_channels: tuple[int, int] = eqx.field(static=True)
    experts: int = eqx.field(static=True)
    mixture_kind: UMAMixtureKind = eqx.field(static=True)
    mixture_layers: tuple[int, ...] = eqx.field(static=True)
    composition_embedding: bool = eqx.field(static=True)
    composition_policy: UMACompositionPolicy = eqx.field(static=True)
    coefficient_norm: UMACoefficientNorm = eqx.field(static=True)
    energy_head: UMAEnergyHeadKind = eqx.field(static=True)
    head_wrapper: UMAHeadWrapper = eqx.field(static=True)
    head_experts: tuple[int, ...] = eqx.field(static=True)
    energy_head_of_task: tuple[int, ...] = eqx.field(static=True)
    force_route: UMAForceRoute = eqx.field(static=True)
    direct_force_head: bool = eqx.field(static=True)
    direct_stress_head: UMAStressReduce | None = eqx.field(static=True)
    isolated_atoms: UMAIsolatedAtomRoute = eqx.field(static=True)
    model_kind: UMAModelKind = eqx.field(static=True)
    roll_gauge: UMAEdgeRollGauge = eqx.field(static=True)
    roll_realization: UMARollRealization = eqx.field(static=True)
    mixture_sharing: bool = eqx.field(static=True)
    energy_reduction: UMAEnergyReduction = eqx.field(static=True)
    species_kind: AtomisticSpeciesKind = eqx.field(static=True)
    architecture_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_elements: int,
        sphere_channels: int,
        lmax: int,
        mmax: int,
        cutoff: float,
        max_neighbors: int,
        edge_channels: int,
        distance_basis: int,
        layer_count: int,
        hidden_channels: int,
        normalization: UMANormalizationKind,
        edge_activation: UMAEdgeActivation,
        feed_forward: UMAFeedForward,
        charge_spin_embedding: UMAChargeSpinEmbeddingKind,
        tasks: Sequence[str],
        dataset_mapping: Sequence[tuple[str, str]] | None,
        grid_resolution: int | None = None,
        strict_neighbors: bool = False,
        charge_spin_trainable: bool = False,
        dataset_trainable: bool = False,
        charge_channels: Sequence[int] = (),
        spin_channels: Sequence[int] = (),
        experts: int = 0,
        mixture_kind: UMAMixtureKind = "so2",
        mixture_layers: Sequence[int] | None = None,
        mixture_sharing: bool = False,
        composition_embedding: bool = False,
        composition_policy: UMACompositionPolicy = "mean",
        coefficient_norm: UMACoefficientNorm = "softmax-plus-0.005",
        energy_head: UMAEnergyHeadKind = "mlp",
        energy_reduction: UMAEnergyReduction = "sum",
        head_wrapper: UMAHeadWrapper = "shared",
        head_experts: Sequence[int] = (),
        energy_head_of_task: Sequence[int] | None = None,
        force_route: UMAForceRoute = "conservative",
        direct_force_head: bool = False,
        direct_stress_head: UMAStressReduce | None = None,
        isolated_atoms: UMAIsolatedAtomRoute = "network",
        model_kind: UMAModelKind = "potential",
        roll_gauge: UMAEdgeRollGauge = "two-chart-nlerp-zero-roll",
    ) -> None:
        layout = UMACoefficientLayout(lmax, mmax)
        self.max_elements = positive_integer(max_elements, "max_elements")
        self.sphere_channels = positive_integer(sphere_channels, "sphere_channels")
        if self.sphere_channels % 2:
            raise ValueError("UMA sphere_channels must be even (sinusoidal embeddings).")
        self.lmax = layout.lmax
        self.mmax = layout.mmax
        if self.lmax == 0:
            raise ValueError("UMA gating needs lmax >= 1.")
        self.grid_resolution = (
            None if grid_resolution is None else positive_integer(grid_resolution, "grid")
        )
        self.cutoff = positive_finite_float(cutoff, "cutoff")
        self.max_neighbors = positive_integer(max_neighbors, "max_neighbors")
        self.strict_neighbors = bool(strict_neighbors)
        self.edge_channels = positive_integer(edge_channels, "edge_channels")
        self.distance_basis = positive_integer(distance_basis, "distance_basis")
        self.layer_count = positive_integer(layer_count, "layer_count")
        self.hidden_channels = positive_integer(hidden_channels, "hidden_channels")
        self.normalization = parse(normalization, UMANormalizationKind, "normalization")
        self.edge_activation = parse(
            edge_activation, UMAEdgeActivation, "edge_activation"
        )
        self.feed_forward = parse(feed_forward, UMAFeedForward, "feed_forward")
        self.charge_spin_embedding = parse(
            charge_spin_embedding, UMAChargeSpinEmbeddingKind, "charge_spin_embedding"
        )
        self.charge_spin_trainable = bool(charge_spin_trainable)
        self.dataset_trainable = bool(dataset_trainable)
        self.tasks, self.dataset_mapping = _admit_tasks(tasks, dataset_mapping)
        self.charge_channels, self.spin_channels = _balanced_channels(
            charge_channels, spin_channels, self.sphere_channels
        )
        self.experts = nonnegative_integer(experts, "experts")
        self.mixture_kind = parse(mixture_kind, UMAMixtureKind, "mixture_kind")
        self.mixture_layers = _mixture_layers(
            mixture_layers, self.layer_count, self.experts
        )
        self.mixture_sharing = bool(mixture_sharing) and self.experts > 0
        self.composition_embedding = bool(composition_embedding) and self.experts > 0
        self.composition_policy = parse(
            composition_policy, UMACompositionPolicy, "composition_policy"
        )
        self.coefficient_norm = parse(
            coefficient_norm, UMACoefficientNorm, "coefficient_norm"
        )
        self.energy_head = parse(energy_head, UMAEnergyHeadKind, "energy_head")
        self.energy_reduction = parse(
            energy_reduction, UMAEnergyReduction, "energy_reduction"
        )
        self.head_wrapper = parse(head_wrapper, UMAHeadWrapper, "head_wrapper")
        self.head_experts, self.energy_head_of_task = _head_routing(
            self.head_wrapper, head_experts, energy_head_of_task, len(self.tasks)
        )
        self.force_route = parse(force_route, UMAForceRoute, "force_route")
        self.direct_force_head = bool(direct_force_head)
        self.direct_stress_head = (
            None
            if direct_stress_head is None
            else parse(direct_stress_head, UMAStressReduce, "direct_stress_head")
        )
        self.isolated_atoms = parse(
            isolated_atoms, UMAIsolatedAtomRoute, "isolated_atoms"
        )
        self.model_kind = parse(model_kind, UMAModelKind, "model_kind")
        if self.model_kind == "property" and self.energy_reduction != "mean":
            raise ValueError(
                "Only mean-reduced source energy heads lack a scalar potential; a "
                "summed head is always a UMAPotential."
            )
        self.roll_gauge = parse(roll_gauge, UMAEdgeRollGauge, "roll_gauge")
        self.roll_realization = _roll_realization(self.edge_activation, self.roll_gauge)
        self.species_kind = AtomisticSpeciesKind.ATOMIC_NUMBER
        record = {name: getattr(self, name) for name in _ARCHITECTURE_RECORD_FIELDS}
        record["kind"] = "uma-escn-md-architecture"
        record["edge_degree_rescale"] = UMA_EDGE_DEGREE_RESCALE
        record["envelope_exponent"] = UMA_ENVELOPE_EXPONENT
        record["degeneracy_tolerance"] = UMA_DEGENERACY_TOLERANCE
        self.architecture_id = canonical_fingerprint(record)

    @property
    def layout(self) -> UMACoefficientLayout:
        return UMACoefficientLayout(self.lmax, self.mmax)

    @property
    def selection_policy(self) -> ImageNeighborSelectionPolicy:
        """The source's scientific nearest-neighbour graph rule."""
        return ImageNeighborSelectionPolicy(
            self.cutoff,
            self.max_neighbors,
            degeneracy_tolerance=UMA_DEGENERACY_TOLERANCE,
            measure="squared-distance",
            strict=self.strict_neighbors,
            radius_boundary="closed",
            coincidence_threshold=0.0,
        )

    @property
    def dataset_rows(self) -> tuple[int, ...]:
        """Dataset-embedding row (source key order) used by each task code."""
        if self.dataset_mapping is None:
            return ()
        keys = [key for key, _ in self.dataset_mapping]
        mapping = dict(self.dataset_mapping)
        return tuple(keys.index(mapping[task]) for task in self.tasks)


class UMAElementDim(Dim):
    pass


@final
class UMATaskReferences(StrictModule, NonTrainableState):
    """Source task normalizers and element references, indexed by task code.

    ``energy_*`` are the energy task's ``Normalizer`` (``x rmsd + mean``) and
    ``ElementReferences`` (summed per atomic number, absent references are
    exact zeros). ``force_*``/``stress_*`` are the *direct* head normalizers;
    conservative forces and stress follow from the denormalized energy.
    """

    __strict_contract__ = True

    energy_available: Bool[UMATaskDim]
    energy_rmsd: Float[UMATaskDim]
    energy_mean: Float[UMATaskDim]
    element_references: Float[UMATaskDim, UMAElementDim]
    force_available: Bool[UMATaskDim]
    force_rmsd: Float[UMATaskDim]
    force_mean: Float[UMATaskDim]
    stress_available: Bool[UMATaskDim]
    stress_rmsd: Float[UMATaskDim]
    stress_mean: Float[UMATaskDim]
    complete_atom_energy: bool = eqx.field(static=True)
    references_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        *,
        tasks: int,
        elements: int,
        energy: Sequence[tuple[int, float, float, np.ndarray | None]],
        forces: Sequence[tuple[int, float, float]] = (),
        stress: Sequence[tuple[int, float, float]] = (),
    ) -> None:
        count = positive_integer(tasks, "tasks")
        size = positive_integer(elements, "elements")

        def table(
            entries: Sequence[tuple[int, float, float]], name: str, /
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            available = np.zeros((count,), dtype=np.bool_)
            rmsd = np.ones((count,), dtype=np.float64)
            mean = np.zeros((count,), dtype=np.float64)
            for code, scale, shift in entries:
                code_ = nonnegative_integer(code, f"{name} task")
                if code_ >= count or available[code_]:
                    raise ValueError(f"Each task has at most one {name} normalizer.")
                if not (np.isfinite(scale) and np.isfinite(shift)) or scale == 0.0:
                    raise ValueError(f"{name} normalizers must be finite with rmsd != 0.")
                available[code_] = True
                rmsd[code_] = float(scale)
                mean[code_] = float(shift)
            return available, rmsd, mean

        energy_available, energy_rmsd, energy_mean = table(
            [(code, scale, shift) for code, scale, shift, _ in energy], "energy"
        )
        references = np.zeros((count, size), dtype=np.float64)
        for code, _, _, values in energy:
            if values is None:
                continue
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 1 or array.shape[0] < size or not np.all(np.isfinite(array)):
                raise ValueError("Element references must cover every model element.")
            if np.any(array[size:] != 0.0):
                raise ValueError(
                    "Element references beyond the model's element table must be zero."
                )
            references[code] = array[:size]
        force_available, force_rmsd, force_mean = table(forces, "force")
        stress_available, stress_rmsd, stress_mean = table(stress, "stress")
        self.energy_available = jnp.asarray(energy_available)
        self.energy_rmsd = jnp.asarray(energy_rmsd)
        self.energy_mean = jnp.asarray(energy_mean)
        self.element_references = jnp.asarray(references)
        self.force_available = jnp.asarray(force_available)
        self.force_rmsd = jnp.asarray(force_rmsd)
        self.force_mean = jnp.asarray(force_mean)
        self.stress_available = jnp.asarray(stress_available)
        self.stress_rmsd = jnp.asarray(stress_rmsd)
        self.stress_mean = jnp.asarray(stress_mean)
        self.complete_atom_energy = bool(np.all(energy_mean == 0.0))
        self.references_id = canonical_fingerprint(
            {
                "kind": "uma-task-references",
                "energy": [
                    energy_available.tolist(),
                    energy_rmsd.tolist(),
                    energy_mean.tolist(),
                    references.tolist(),
                ],
                "forces": [
                    force_available.tolist(),
                    force_rmsd.tolist(),
                    force_mean.tolist(),
                ],
                "stress": [
                    stress_available.tolist(),
                    stress_rmsd.tolist(),
                    stress_mean.tolist(),
                ],
            }
        )


@final
class UMAEnergyHead(StrictModule):
    """Source ``MLP_EFS_Head``/``MLP_Energy_Head`` (3 linears) or ``Linear_Energy_Head``."""

    linears: tuple[UMALinearMap, ...]

    def __init__(self, linears: Sequence[UMALinearMap], /) -> None:
        linears_ = tuple(linears)
        if len(linears_) not in (1, 3) or linears_[-1].out_features != 1:
            raise ValueError("UMA energy heads have one or three linears to one output.")
        if any(linear.bias is None for linear in linears_):
            raise ValueError("UMA energy head linears are biased.")
        self.linears = linears_

    def __call__(self, scalars: Array, task: Array, /) -> Array:
        hidden = scalars
        for index, linear in enumerate(self.linears):
            hidden = apply_linear(linear, hidden, task)
            if index < len(self.linears) - 1:
                hidden = jax.nn.silu(hidden)
        return hidden[..., 0]


@final
class UMADirectStressHead(StrictModule):
    """Source ``MLP_Stress_Head``: scalar MLP trace plus ``SO3_Linear`` ``l = 2`` part."""

    scalar: UMAEnergyHead
    anisotropic: UMASO3Linear

    def __init__(self, scalar: UMAEnergyHead, anisotropic: UMASO3Linear, /) -> None:
        if len(scalar.linears) != 3 or anisotropic.lmax != 2:
            raise ValueError("The stress head has a 3-linear scalar MLP and an l=2 map.")
        self.scalar = scalar
        self.anisotropic = anisotropic


class _SourceRows(NamedTuple):
    """Block source payload: normalized hidden states and species of each source."""

    features: Array
    species: Array


class _SpeciesRows(NamedTuple):
    """Edge-degree source payload: the embedding reads only source species."""

    species: Array


class _ReceiverRows(NamedTuple):
    features: Array
    residual: Array
    species: Array
    case: Array
    mask: Array


class _EdgeRows(NamedTuple):
    displacement: Array
    roll: Array | None


@final
class UMAEdgeGeometry(StrictModule):
    """Per-edge invariant features, source Wigner pair and envelope."""

    distance: UMAGaussianSmearing
    source_embedding: Array
    target_embedding: Array
    layout: UMACoefficientLayout
    cutoff: float = eqx.field(static=True)

    def __init__(
        self,
        distance: UMAGaussianSmearing,
        source_embedding: Array,
        target_embedding: Array,
        layout: UMACoefficientLayout,
        cutoff: float,
        /,
    ) -> None:
        self.distance = distance
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.layout = layout
        self.cutoff = cutoff

    def __call__(
        self, edge: _EdgeRows, source: Array, target: Array, /
    ) -> tuple[Array, Array, Array]:
        """``(x_edge, wigner, envelope * wigner_inverse)`` of one edge.

        ``edge.displacement`` is the native ``r_receiver - r_source (+ image)``;
        the source edge vector is its negation. ``edge.roll`` is the explicit
        roll of the keyed gauge, or ``None`` for the zero roll.
        """
        vector = -edge.displacement
        squared = jnp.sum(vector * vector)
        positive = squared > 0.0
        distance = jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1.0)), 0.0)
        features = jnp.concatenate(
            [
                self.distance(distance),
                self.source_embedding[source],
                self.target_embedding[target],
            ]
        )
        wigner, inverse = uma_edge_wigner(vector, self.layout, roll=edge.roll)
        envelope = polynomial_envelope(
            distance / self.cutoff, exponent=UMA_ENVELOPE_EXPONENT
        )
        return features, wigner, inverse * envelope


@final
class _EdgeDegreeStream(StrictModule):
    embedding: UMAEdgeDegreeEmbedding
    geometry: UMAEdgeGeometry

    def edge(
        self, source: _SpeciesRows, receiver: _ReceiverRows, edge: _EdgeRows, /
    ) -> Array:
        features, _, inverse = self.geometry(edge, source.species, receiver.species)
        return self.embedding.message(
            inverse, features, receiver.case, self.geometry.layout.m_sizes[0]
        ).astype(receiver.residual.dtype)

    def epilogue(self, receiver: _ReceiverRows, aggregate: Array, /) -> Array:
        return jnp.where(receiver.mask, receiver.residual + aggregate, 0.0)


@final
class _BlockStream(StrictModule):
    block: UMABlock
    geometry: UMAEdgeGeometry

    def edge(
        self, source: _SourceRows, receiver: _ReceiverRows, edge: _EdgeRows, /
    ) -> Array:
        features, wigner, inverse = self.geometry(edge, source.species, receiver.species)
        return self.block.edgewise.message(
            source.features,
            receiver.features,
            wigner,
            inverse,
            features,
            receiver.case,
        )

    def epilogue(self, receiver: _ReceiverRows, aggregate: Array, /) -> Array:
        return jnp.where(receiver.mask, receiver.residual + aggregate, 0.0)


def _node_blocks(
    function: Callable[..., Array], arrays: Sequence[Array], block: int, /
) -> Array:
    """Apply ``function`` to bounded row blocks of node arrays (``lax.map``)."""
    rows = arrays[0].shape[0]
    count = -(-rows // block)
    padding = count * block - rows
    shaped = tuple(
        jnp.pad(array, ((0, padding),) + ((0, 0),) * (array.ndim - 1)).reshape(
            (count, block) + array.shape[1:]
        )
        for array in arrays
    )
    output = jax.lax.map(lambda values: function(*values), shaped)
    return output.reshape((count * block,) + output.shape[2:])[:rows]


def _compute_dtype(precision: AtomisticPrecisionPolicy, /) -> np.dtype:
    dtype = np.dtype(precision.compute_dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("UMA computes in float32 or float64.")
    return dtype


@final
class UMANetwork(StrictModule, ParameterOwner):
    """The shared UMA trunk, heads and source references of one declaration.

    Both model kinds evaluate exactly this network: `UMAPotential` (a scalar
    potential exists) and `UMAPropertyModel` (the source only reports
    intensive/direct properties). ``streaming`` tiles every edge pass;
    ``node_block`` bounds the atom rows of each feed-forward evaluation.
    """

    configuration: UMAArchitecture
    schema: UMAConditionSchema
    sphere_embedding: Array
    source_embedding: Array
    target_embedding: Array
    system: UMASystemEmbedding
    router: UMAMoLERouter | None
    distance: UMAGaussianSmearing
    edge_degree: UMAEdgeDegreeEmbedding
    blocks: tuple[UMABlock, ...]
    shared_mixtures: tuple[UMAMixtureLinear, ...]
    norm: UMANormalization
    energy_heads: tuple[UMAEnergyHead, ...]
    force_head: UMASO3Linear | None
    stress_head: UMADirectStressHead | None
    references: UMATaskReferences
    streaming: StreamedRelationPlan | None
    node_block: int = eqx.field(static=True)

    def __init__(
        self,
        architecture: UMAArchitecture,
        /,
        *,
        dtype: DTypeLike,
        sphere_embedding: ArrayLike,
        source_embedding: ArrayLike,
        target_embedding: ArrayLike,
        system: UMASystemEmbedding,
        router: UMAMoLERouter | None,
        edge_degree: UMAEdgeDegreeEmbedding,
        blocks: Sequence[UMABlock],
        norm: UMANormalization,
        energy_heads: Sequence[UMAEnergyHead],
        references: UMATaskReferences,
        force_head: UMASO3Linear | None = None,
        stress_head: UMADirectStressHead | None = None,
        streaming: StreamedRelationPlan | None = None,
        node_block: int = 1024,
        shared_mixtures: Sequence[UMAMixtureLinear] = (),
    ) -> None:
        if not isinstance(architecture, UMAArchitecture):
            raise TypeError("architecture must be a UMAArchitecture.")
        dtype_ = np.dtype(dtype)
        if dtype_ not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("UMA computes in float32 or float64.")
        if streaming is not None and not isinstance(streaming, StreamedRelationPlan):
            raise TypeError("streaming must be a StreamedRelationPlan or None.")
        self.configuration = architecture
        self.schema = UMAConditionSchema(architecture.tasks)
        self.sphere_embedding = jnp.asarray(sphere_embedding, dtype=dtype_)
        self.source_embedding = jnp.asarray(source_embedding, dtype=dtype_)
        self.target_embedding = jnp.asarray(target_embedding, dtype=dtype_)
        self.system = system
        self.router = router
        self.distance = UMAGaussianSmearing(
            architecture.cutoff, architecture.distance_basis, dtype=dtype_
        )
        self.edge_degree = edge_degree
        self.blocks = tuple(blocks)
        self.shared_mixtures = tuple(shared_mixtures)
        self.norm = norm
        self.energy_heads = tuple(energy_heads)
        self.force_head = force_head
        self.stress_head = stress_head
        self.references = references
        self.streaming = streaming
        self.node_block = positive_integer(node_block, "node_block")
        _check_components(self, dtype_)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.sphere_embedding.dtype)

    def validate(self) -> None:
        """Re-run every component validator and recompute fixed tables."""
        _check_components(self, self.dtype)
        for block in self.blocks:
            if block.edgewise.grid is not None:
                block.edgewise.grid.validate()
            grid = getattr(block.atomwise, "grid", None)
            if grid is not None:
                grid.validate()
        parameters = jax.tree_util.tree_leaves(partition_parameters(self)[0])
        if not all(
            bool(np.all(np.isfinite(jax.device_get(leaf)))) for leaf in parameters
        ):
            raise ValueError("UMA parameters must be finite.")

    def property_observables(self) -> tuple[AtomisticObservable, ...]:
        """Typed outputs beyond the scalar potential this network produces."""
        observables = []
        if self.force_head is not None:
            observables.append(AtomisticObservable.DIRECT_FORCES)
        if self.stress_head is not None:
            observables.append(AtomisticObservable.DIRECT_STRESS)
        if self.configuration.energy_reduction == "mean":
            observables.append(AtomisticObservable.REPORTED_ENERGY)
        return tuple(observables)

    @property
    def composition_routed(self) -> bool:
        """Routing reads per-case species counts (a composition embedding)."""
        return self.router is not None and self.router.composition is not None

    def mixture_coefficients(
        self,
        system: Array,
        species_counts: Array | None,
        atom_counts: Array,
        /,
    ) -> Array | None:
        """Per-case expert coefficients ``(B, E)``, or ``None`` without experts.

        ``species_counts`` ``(B, max_elements)`` and ``atom_counts`` ``(B,)`` are
        complete per-case sums over every atom of the case; ``species_counts``
        is read only by composition routing. A case is active when it has atoms.
        """
        if self.router is None:
            return None
        counts = species_counts if self.router.composition is not None else None
        return self.router.coefficients(system, counts, atom_counts > 0)

    @property
    def bound_blocks(self) -> tuple[UMABlock, ...]:
        """Blocks with parameter-shared mixtures (source ``moe_single``) bound."""
        if not self.shared_mixtures:
            return self.blocks
        return bind_shared_mixtures(self.blocks, self.shared_mixtures)

    def realized_blocks(self, coefficients: Array | None, /) -> tuple[UMABlock, ...]:
        blocks = self.bound_blocks
        if coefficients is None:
            return blocks
        return resolve_mixtures(blocks, coefficients)

    def head_coefficients(self, dtype: np.dtype, /) -> Array | None:
        """One-hot expert coefficients ``(T, E)`` of the dataset-expert head wrapper."""
        experts = self.configuration.head_experts
        if not experts:
            return None
        width = mixture_linears(self.energy_heads[0])[0].expert_count
        return jax.nn.one_hot(jnp.asarray(experts), width, dtype=dtype)


def _model_precision(
    network: UMANetwork,
    scale: AtomisticScaleContract,
    precision: AtomisticPrecisionPolicy | None,
    /,
) -> AtomisticPrecisionPolicy:
    if not isinstance(scale, AtomisticScaleContract):
        raise TypeError("scale must be an AtomisticScaleContract.")
    if not isinstance(network, UMANetwork):
        raise TypeError("network must be a UMANetwork.")
    policy = AtomisticPrecisionPolicy() if precision is None else precision
    if _compute_dtype(policy) != network.dtype:
        raise ValueError("The precision compute dtype differs from the network dtype.")
    return policy


def _validate_scaled_batch(
    scale: AtomisticScaleContract,
    precision: AtomisticPrecisionPolicy,
    batch: AtomisticBatch,
    /,
) -> None:
    if batch.scale.scale_id != scale.scale_id:
        raise ValueError("Model and structure must share one exact scale contract.")
    if batch.positions.dtype != jnp.dtype(precision.coordinate_dtype):
        raise ValueError("Batch coordinate dtype does not match the UMA precision.")


def _capabilities(*, potential: bool) -> AtomisticPotentialCapabilities:
    return AtomisticPotentialCapabilities(
        conservative_energy=potential,
        local_energy=potential,
        orthorhombic_periodic=True,
        triclinic_periodic=True,
        cell_derivative=potential,
        species_kind=AtomisticSpeciesKind.ATOMIC_NUMBER,
    )


def _requirements(configuration: UMAArchitecture, /) -> AtomisticPotentialRequirements:
    return AtomisticPotentialRequirements(
        neighborhoods=(
            AtomisticNeighborhoodRequirement(
                configuration.architecture_id,
                AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH,
                configuration.cutoff,
                selection=configuration.selection_policy,
            ),
        ),
    )


def _coordinate_regularity(
    configuration: UMAArchitecture, coordinate_dtype: str, /
) -> DerivativeRegularity:
    """Smooth pieces of the executed radius selection and edge-frame formulas.

    Capped-neighbor membership can jump inside the radial cutoff. Internal
    frame-floor/blend joins remain unresolved evidence; finite S2 quadrature
    does not cancel arbitrary edge-gauge jumps. Discrete conditions, topology,
    material route identities and a keyed roll realization stay fixed.
    """
    conditions = (
        atomistic_edge_distance_locus(configuration.cutoff),
        atomistic_edge_distance_locus(1.0e-12),
        atomistic_edge_distance_locus(
            math.sqrt(np.finfo(np.dtype(coordinate_dtype)).tiny)
        ),
        "atomistic-nonsmooth-locus:uma-neighbor-selection-membership",
        "atomistic-nonsmooth-locus:uma-edge-frame-normalization-and-blend-joins",
    )
    if configuration.edge_activation == "s2":
        conditions += ("atomistic-nonsmooth-locus:uma-edge-roll-gauge-join",)
    return DerivativeRegularity.piecewise_smooth(
        continuity=-1,
        support=ATOMISTIC_NONCOINCIDENT_SUPPORT,
        conditions=conditions,
    )


@final
class UMAPotential(AbstractAtomisticPotential):
    """Native trainable UMA/eSCN-MD scalar-energy potential.

    The potential is the summed head energy (source ``reduce="sum"``) or, for
    mean-reduced heads whose forces are gradients, the differentiated sum
    ``rmsd * sum_i e_i`` (the source's task energy is then `REPORTED_ENERGY`).
    Build with `uma_model_from_source`.
    """

    network: UMANetwork
    configuration: UMAArchitecture
    scale: AtomisticScaleContract
    precision: AtomisticPrecisionPolicy
    architecture_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: AtomisticScaleContract,
        network: UMANetwork,
        /,
        *,
        precision: AtomisticPrecisionPolicy | None = None,
    ) -> None:
        policy = _model_precision(network, scale, precision)
        if network.configuration.model_kind != "potential":
            raise ValueError(
                "This UMA declaration has no scalar potential; use UMAPropertyModel."
            )
        self.network = network
        self.configuration = network.configuration
        self.scale = scale
        self.precision = policy
        self.architecture_id = _model_identity(network, scale, policy)
        self.method_id = _method_id(network.configuration)

    def validate(self) -> None:
        """Re-run the owning validators on this (possibly restored) potential."""
        self.network.validate()
        if (
            self.configuration.architecture_id
            != self.network.configuration.architecture_id
            or _model_identity(self.network, self.scale, self.precision)
            != self.architecture_id
            or self.method_id != _method_id(self.configuration)
        ):
            raise ValueError("The UMA potential does not match its recorded identities.")

    @property
    def capabilities(self) -> AtomisticPotentialCapabilities:
        return _capabilities(potential=True)

    @property
    def derivative_contract(self) -> DerivativeContract:
        """Branchwise response, with explicit selection/gauge and router conditions."""
        pnorm = (
            self.network.router is not None
            and self.configuration.coefficient_norm == "pnorm-batch-count"
        )
        return DerivativeContract(
            (
                SurfaceDerivative(
                    DerivativeSurface.INPUT, GradientLevel.ALMOST_EVERYWHERE
                ),
                SurfaceDerivative(
                    DerivativeSurface.MODEL_PARAMETER,
                    GradientLevel.ALMOST_EVERYWHERE if pnorm else GradientLevel.SMOOTH,
                    conditions=("atomistic-nonsmooth-locus:uma-router-pnorm-hidden-zero",)
                    if pnorm
                    else (),
                ),
            ),
            route=DerivativeRoute.DIRECT,
            regularity=_coordinate_regularity(
                self.configuration, self.precision.coordinate_dtype
            ),
        )

    @property
    def requirements(self) -> AtomisticPotentialRequirements:
        return _requirements(self.configuration)

    @property
    def supported_observables(self) -> tuple[AtomisticObservable, ...]:
        return (
            AtomisticObservable.ENERGY,
            AtomisticObservable.ATOM_ENERGY,
            *self.network.property_observables(),
        )

    @property
    def schema(self) -> UMAConditionSchema:
        return self.network.schema

    def merged(self, binding: UMAMoLEBinding, /) -> PreparedUMAMergedModel:
        """Explicitly prepare the merged MoLE realization of one exact binding."""
        return PreparedUMAMergedModel(self, binding)

    @checked
    def _validate_batch(self, batch: AtomisticBatch, /) -> None:
        _validate_scaled_batch(self.scale, self.precision, batch)

    @checked
    def evaluate_graph(
        self,
        inputs: AtomisticGraphInputs,
        request: AtomisticObservableRequest | None = None,
        /,
    ) -> AtomisticModelOutputs:
        """Case energies, scoped atom energies and requested typed outputs."""
        return _evaluate(self, inputs, request, self.network.realized_blocks, None)

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
        raise TypeError("UMAPotential expects AtomicStructure or AtomisticBatch.")

    # --- layered owner-region execution (AtomisticLayeredModel) ------------

    @property
    def interaction_count(self) -> int:
        """The edge-degree embedding followed by one interaction per block."""
        return self.configuration.layer_count + 1

    @property
    def system_stages(self) -> tuple[AtomisticSystemStage, ...]:
        """Composition routing (prologue) and per-block channel balancing."""
        return tuple(plan.stage for plan in _stage_plans(self.network))

    def layer_context(
        self,
        conditioning: AtomisticConditioning | None,
        atom_cases: Array,
        species: Array,
        atom_mask: Array,
        /,
        *,
        key: PRNGKey | None = None,
    ) -> UMALayerContext:
        """Replicated case conditions and ``csd`` embeddings of one owner region.

        ``key`` is the evaluation's explicit PRNG key. The keyed roll gauge
        requires it (no roll is drawn implicitly) and draws each route's roll
        from it and the route's canonical identity, exactly as on one graph;
        other gauges ignore it.
        """
        network = self.network
        keyed = self.configuration.roll_gauge == "keyed-route-uniform-roll"
        if keyed and key is None:
            raise ValueError(
                "The keyed UMA roll gauge needs an explicit PRNG key on its owner "
                "conditioning; no roll is drawn implicitly."
            )
        conditions = _require_conditioning(network, conditioning)
        charge = jnp.asarray(conditions.value(UMA_CHARGE_FIELD))
        spin = jnp.asarray(conditions.value(UMA_SPIN_FIELD))
        task = jnp.asarray(conditions.value(UMA_TASK_FIELD))
        system = network.system(charge, spin, task).astype(network.dtype)
        blocks = None if _stage_plans(network) else network.realized_blocks(None)
        return UMALayerContext(
            jnp.asarray(atom_cases, dtype=jnp.int32),
            charge,
            spin,
            task,
            system,
            None,
            blocks,
            key if keyed else None,
        )

    def species_indices(self, species_ids: ArrayLike, atom_mask: ArrayLike, /) -> Array:
        """Executor species of raw atomic numbers: the element-table row itself.

        Inactive slots read zero. Active out-of-table numbers are kept, so
        `admitted` reports them; every hook reads only in-table rows.
        """
        mask = jnp.asarray(atom_mask, dtype=jnp.bool_)
        return jnp.where(mask, jnp.asarray(species_ids, dtype=jnp.int32), 0)

    def admitted(
        self, species: Array, atom_mask: Array, context: UMALayerContext, /
    ) -> Array:
        """Owned atoms lie in the element table and every task has an energy.

        A task without an energy normalizer has no UMA potential; an active
        species outside the table has no embedding. Either makes this owner's
        evaluation unsuccessful (no runtime error inside the owner region).
        """
        raw = jnp.asarray(species, dtype=jnp.int32)
        inside = (raw >= 0) & (raw < self.configuration.max_elements)
        return jnp.all(~atom_mask | inside) & jnp.all(
            _energy_available(self.network, context.task)
        )

    def system_partials(
        self,
        stage: int,
        features: Array | None,
        species: Array,
        atom_mask: Array,
        context: UMALayerContext,
        /,
    ) -> UMACompositionPartials | UMABalancePartials:
        """This owner's per-case sums over its owned active atoms."""
        network = self.network
        configuration = self.configuration
        plan = _stage_plans(network)[stage]
        cases = context.system.shape[0]
        if plan.block is None:
            rows = _layered_species(configuration, species, atom_mask)
            atoms = case_sums(
                atom_mask.astype(jnp.int32), atom_mask, context.atom_cases, cases
            )
            table = (
                case_sums(
                    jax.nn.one_hot(rows, configuration.max_elements, dtype=jnp.int32),
                    atom_mask,
                    context.atom_cases,
                    cases,
                )
                if network.composition_routed
                else None
            )
            return UMACompositionPartials(atoms, table)
        if features is None:
            raise ValueError("Channel balancing reads the block's output features.")

        def channel_sums(channels: tuple[int, int], /) -> Array:
            start, end = channels
            return case_sums(
                features[:, 0, start:end], atom_mask, context.atom_cases, cases
            )

        return UMABalancePartials(
            channel_sums(configuration.charge_channels) if plan.charge else None,
            channel_sums(configuration.spin_channels) if plan.spin else None,
        )

    def apply_system_context(
        self,
        stage: int,
        features: Array | None,
        system_context: UMACompositionPartials | UMABalancePartials,
        species: Array,
        atom_mask: Array,
        context: UMALayerContext,
        /,
    ) -> tuple[Array | None, UMALayerContext]:
        """Consume one stage's complete per-case sums.

        Composition: route the experts from the complete species and atom
        counts and realize every block with the per-case coefficients.
        Balancing: shift the declared channels of owned active atoms.
        """
        network = self.network
        configuration = self.configuration
        plan = _stage_plans(network)[stage]
        if plan.block is None:
            if not isinstance(system_context, UMACompositionPartials):
                raise TypeError("The composition stage consumes UMACompositionPartials.")
            coefficients = network.mixture_coefficients(
                context.system, system_context.species_counts, system_context.atom_counts
            )
            return features, UMALayerContext(
                context.atom_cases,
                context.charge,
                context.spin,
                context.task,
                context.system,
                system_context.atom_counts,
                network.realized_blocks(coefficients),
                context.key,
            )
        if not isinstance(system_context, UMABalancePartials):
            raise TypeError("A balancing stage consumes UMABalancePartials.")
        if features is None:
            raise ValueError("Channel balancing updates the block's output features.")
        if plan.charge:
            features = _balance(
                features,
                system_context.charge,
                context.charge,
                configuration.charge_channels,
                0.0,
                atom_mask,
                context,
            )
        if plan.spin:
            features = _balance(
                features,
                system_context.spin,
                context.spin,
                configuration.spin_channels,
                1.0,
                atom_mask,
                context,
            )
        return features, context

    def init_features(
        self, species: Array, atom_mask: Array, context: UMALayerContext, /
    ) -> Array:
        """``x[l = 0] = sphere_embedding[Z] + csd`` of owned rows."""
        rows = _layered_species(self.configuration, species, atom_mask)
        return _initial_features(self.network, rows, atom_mask, context.node_system)

    def prepare_sources(
        self,
        index: int,
        features: Array,
        species: Array,
        atom_mask: Array,
        context: UMALayerContext,
        /,
    ) -> _SpeciesRows | _SourceRows:
        """Owned source payload of one interaction (exchanged to receivers).

        The edge-degree embedding reads only source species; block ``k`` reads
        the normalized hidden states ``norm_1(x) + csd`` and species.
        """
        rows = _layered_species(self.configuration, species, atom_mask)
        if index == 0:
            return _SpeciesRows(rows)
        block = context.realized()[index - 1]
        return _SourceRows(
            _block_hidden(block, features, context.node_system, atom_mask), rows
        )

    def update_receivers(
        self,
        index: int,
        relation: PreparedStreamedRelation,
        vectors: Array,
        sources: _SpeciesRows | _SourceRows,
        features: Array,
        species: Array,
        atom_mask: Array,
        context: UMALayerContext,
        /,
        *,
        edge_active: Array | None = None,
        route_identity: tuple[Array, ...] | None = None,
    ) -> UMALayerUpdate:
        """Stream one interaction into owned receivers.

        ``vectors`` are ``r_receiver - r_source (+ n H)`` in route order over
        ``relation`` (owned receiver rows, exchanged source columns). Block
        ``k`` convolves the receiver's own ``norm_1(x) + csd`` with its source
        columns, adds the aggregate to ``x`` and applies the node-local
        ``x + atomwise(norm_2(x))``. ``route_identity`` are the routes'
        canonical identity fields (the single-graph ``route_identity_fields``
        order); the keyed roll gauge draws every route's roll from them and
        the context key, so an edge has the same roll on every owner layout.
        """
        network = self.network
        rows = _layered_species(self.configuration, species, atom_mask)
        geometry = _edge_geometry(network)
        rolls = None
        if context.key is not None:
            if route_identity is None:
                raise ValueError(
                    "The keyed UMA roll gauge needs the routes' canonical identities."
                )
            rolls = uma_keyed_edge_rolls(context.key, route_identity, dtype=network.dtype)
        edges = _EdgeRows(jnp.asarray(vectors).astype(network.dtype), rolls)
        active = None if edge_active is None else jnp.asarray(edge_active, jnp.bool_)
        if index == 0:
            if not isinstance(sources, _SpeciesRows):
                raise TypeError("The edge-degree embedding reads species source rows.")
            result = _stream(
                relation,
                _EdgeDegreeStream(network.edge_degree, geometry),
                sources,
                _ReceiverRows(features, features, rows, context.atom_cases, atom_mask),
                edges,
                active,
            )
            updated = result.receiver_outputs
        else:
            if not isinstance(sources, _SourceRows):
                raise TypeError("A UMA block reads hidden-state source rows.")
            block = context.realized()[index - 1]
            hidden = _block_hidden(block, features, context.node_system, atom_mask)
            result = _stream(
                relation,
                _BlockStream(block, geometry),
                sources,
                _ReceiverRows(hidden, features, rows, context.atom_cases, atom_mask),
                edges,
                active,
            )
            updated = _block_atomwise(
                network, block, result.receiver_outputs, context.atom_cases, atom_mask
            )
        reduction = jnp.dtype(self.precision.reduction_dtype)
        return UMALayerUpdate(
            updated, jnp.zeros((features.shape[0],), dtype=reduction), result.evidence
        )

    def atom_energies(
        self,
        species: Array,
        atom_mask: Array,
        layer_energies: Array,
        features: Array,
        context: UMALayerContext,
        /,
    ) -> Array:
        """Owned atom energies: final norm, task head, normalizer and references.

        ``layer_energies`` are UMA's identically zero per-interaction terms; the
        energy is read out from ``features`` only.
        """
        network = self.network
        reduction = jnp.dtype(self.precision.reduction_dtype)
        rows = _layered_species(self.configuration, species, atom_mask)
        node_task = _task_codes(self.configuration, context.task)[context.atom_cases]
        raw = _raw_head_energies(
            network, _final_features(network, features, atom_mask), node_task, reduction
        )
        return _potential_atom_energies(
            network, raw, node_task, rows, atom_mask, reduction
        )

    def global_payload(
        self,
        features: Array,
        species: Array,
        atom_mask: Array,
        context: UMALayerContext,
        /,
    ) -> None:
        """UMA is local after its system stages: no global epilogue payload."""
        return None

    def system_energy(self, context: UMALayerContext, /) -> Array:
        """Replicated energy outside every atom energy, summed over cases.

        The summed head's task-normalizer mean of each case (zero for
        mean-reduced heads). The executor counts it once; it is independent of
        positions and cell.
        """
        reduction = jnp.dtype(self.precision.reduction_dtype)
        case_task = _task_codes(self.configuration, context.task)
        return jnp.sum(_case_energy_shift(self.network, case_task, reduction))

    def owner_fields(
        self,
        features: Array,
        species: Array,
        atom_mask: Array,
        context: UMALayerContext,
        /,
    ) -> Mapping[AtomisticObservable, Array]:
        """UMA predicts no per-atom field consumed by domain terms."""
        del features, species, atom_mask, context
        return {}


@final
class UMAPropertyModel(AbstractAtomisticModel):
    """UMA source configuration with no scalar potential: typed reports only.

    Arises when the source energy heads are mean-reduced and some dataset's
    forces are not their gradient (direct forces, or a ``Linear_Energy_Head``):
    the source then never forms an extensive energy. Outputs are the source
    `REPORTED_ENERGY` and any `DIRECT_FORCES`/`DIRECT_STRESS`; ``ENERGY`` and
    ``ATOM_ENERGY`` requests refuse before execution and no energy is invented.
    """

    network: UMANetwork
    configuration: UMAArchitecture
    scale: AtomisticScaleContract
    precision: AtomisticPrecisionPolicy
    architecture_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: AtomisticScaleContract,
        network: UMANetwork,
        /,
        *,
        precision: AtomisticPrecisionPolicy | None = None,
    ) -> None:
        policy = _model_precision(network, scale, precision)
        if network.configuration.model_kind != "property":
            raise ValueError(
                "This UMA declaration has a scalar potential; use UMAPotential."
            )
        self.network = network
        self.configuration = network.configuration
        self.scale = scale
        self.precision = policy
        self.architecture_id = _model_identity(network, scale, policy)
        self.method_id = _method_id(network.configuration)

    def validate(self) -> None:
        """Re-run the owning validators on this (possibly restored) model."""
        self.network.validate()
        if (
            self.configuration.architecture_id
            != self.network.configuration.architecture_id
            or _model_identity(self.network, self.scale, self.precision)
            != self.architecture_id
            or self.method_id != _method_id(self.configuration)
        ):
            raise ValueError(
                "The UMA property model does not match its recorded identities."
            )

    @property
    def capabilities(self) -> AtomisticPotentialCapabilities:
        return _capabilities(potential=False)

    @property
    def requirements(self) -> AtomisticPotentialRequirements:
        return _requirements(self.configuration)

    @property
    def supported_observables(self) -> tuple[AtomisticObservable, ...]:
        return self.network.property_observables()

    @property
    def schema(self) -> UMAConditionSchema:
        return self.network.schema

    def merged(self, binding: UMAMoLEBinding, /) -> PreparedUMAMergedModel:
        """Explicitly prepare the merged MoLE realization of one exact binding."""
        return PreparedUMAMergedModel(self, binding)

    @checked
    def _validate_batch(self, batch: AtomisticBatch, /) -> None:
        _validate_scaled_batch(self.scale, self.precision, batch)

    @checked
    def evaluate_graph(
        self,
        inputs: AtomisticGraphInputs,
        request: AtomisticObservableRequest | None = None,
        /,
    ) -> AtomisticModelOutputs:
        """Requested typed reports; ``request=None`` yields every supported report."""
        return _evaluate(self, inputs, request, self.network.realized_blocks, None)


UMAModel: TypeAlias = UMAPotential | UMAPropertyModel


@final
class _FrozenUMA(StrictModule, ExplicitFreeze):
    """The exact raw UMA model a prepared realization intentionally freezes."""

    value: UMAModel

    def __init__(self, value: UMAModel, /) -> None:
        self.value = value


@final
class _FixedBlocks(StrictModule, ExplicitFreeze):
    """Merged parameter-bearing blocks intentionally fixed by their binding."""

    value: tuple[UMABlock, ...]

    def __init__(self, value: tuple[UMABlock, ...], /) -> None:
        self.value = value


class StaleUMAMergeError(ValueError):
    """A merged MoLE realization no longer matches its model revision or binding."""


@final
class PreparedUMAMergedModel(AbstractPreparedAtomisticModel):
    """A UMA model with every mixture merged for one exact composition/condition binding.

    One implementation serves both model kinds: it keeps the raw model's kind,
    capabilities and supported observables (a merged `UMAPropertyModel` still
    refuses ``ENERGY`` and returns unavailable energy; a merged `UMAPotential`
    returns its potential through `evaluate_graph`). ``model`` is the exact raw
    model, frozen in ``frozen_model``; its scientific architecture and method
    identities, scale, precision, configuration and batch validation come from
    it through the prepared base. Merged operators are fixed leaves bound to
    the raw model's numeric revision (``source_revision_id``); ``prepared_id``
    identifies this realization (raw architecture and method, source revision
    and binding). Both digests depend on parameter values, so they are carried
    as dynamic `IdentityWords` leaves (``source_revision_identity``,
    ``prepared_identity``) decoded only at host boundaries: a re-merge after a
    parameter update keeps the same PyTree structure. Cases whose species
    counts or conditions differ from the binding fail (no fallback to
    routing); a changed model must be re-prepared (`require_current`).
    """

    frozen_model: _FrozenUMA
    merged_blocks: _FixedBlocks
    binding: UMAMoLEBinding
    source_revision_identity: IdentityWords
    prepared_identity: IdentityWords

    def __init__(self, model: UMAModel, binding: UMAMoLEBinding, /) -> None:
        if not isinstance(model, (UMAPotential, UMAPropertyModel)):
            raise TypeError("model must be a UMAPotential or UMAPropertyModel.")
        if not isinstance(binding, UMAMoLEBinding):
            raise TypeError("binding must be a UMAMoLEBinding.")
        _admit_binding(model, binding)
        merged = merge_mixtures(
            model.network.bound_blocks, _binding_coefficients(model.network, binding)
        )
        revision = atomistic_model_revision(model).revision_id
        self.frozen_model = _FrozenUMA(model)
        self.merged_blocks = _FixedBlocks(merged)
        self.binding = binding
        self.source_revision_identity = IdentityWords.from_fingerprint(revision)
        self.prepared_identity = IdentityWords.from_fingerprint(
            _merged_identity(model, revision, binding)
        )

    @property
    def source_revision_id(self) -> str:
        """The bound raw-model numeric revision (host boundary)."""
        return self.source_revision_identity.fingerprint()

    @property
    def prepared_id(self) -> str:
        """The identity of this merged realization (host boundary)."""
        return self.prepared_identity.fingerprint()

    @property
    def model(self) -> UMAModel:
        """The exact raw model this realization merges."""
        return self.frozen_model.value

    @property
    def capabilities(self) -> AtomisticPotentialCapabilities:
        return self.model.capabilities

    @property
    def derivative_contract(self) -> DerivativeContract | None:
        """The frozen merge has coordinate response, not a trainable parameter surface."""
        if isinstance(self.model, UMAPropertyModel):
            return None
        return DerivativeContract(
            (
                SurfaceDerivative(
                    DerivativeSurface.INPUT, GradientLevel.ALMOST_EVERYWHERE
                ),
            ),
            route=DerivativeRoute.DIRECT,
            regularity=_coordinate_regularity(
                self.model.configuration, self.precision.coordinate_dtype
            ),
        )

    @property
    def requirements(self) -> AtomisticPotentialRequirements:
        return self.model.requirements

    @property
    def supported_observables(self) -> tuple[AtomisticObservable, ...]:
        return self.model.supported_observables

    @property
    def model_kind(self) -> UMAModelKind:
        return self.model.configuration.model_kind

    @property
    def schema(self) -> UMAConditionSchema:
        return self.model.schema

    def require_current(self, model: UMAModel, /) -> None:
        """Refuse a model whose revision differs from the one these merges bind."""
        if atomistic_model_revision(model).revision_id != self.source_revision_id:
            raise StaleUMAMergeError(
                "The UMA model changed since its MoLE merge; prepare it again."
            )

    def validate(self) -> None:
        """Re-run the raw model's validators and re-derive every merged operator.

        This is a host boundary over concrete arrays, run at publication and
        restoration, never inside a numerical call. The binding is replayed
        through its constructor and re-admitted for the raw model (composition
        policy, element table, task code, experts); the raw model's numeric
        revision must equal ``source_revision_id``; the merged operators must
        reproduce a fresh merge of that binding exactly, with identical
        structure and dtype; ``prepared_id`` is recomputed, never trusted.
        """
        model = self.model
        model.validate()
        _admit_binding(model, self.binding)
        revision = atomistic_model_revision(model).revision_id
        require_identical(
            self.source_revision_identity,
            IdentityWords.from_fingerprint(revision),
            "The bound source revision",
            StaleUMAMergeError,
        )
        require_identical(
            self.merged_blocks.value,
            merge_mixtures(
                model.network.bound_blocks,
                _binding_coefficients(model.network, self.binding),
            ),
            "Merged MoLE operators",
            StaleUMAMergeError,
        )
        require_identical(
            self.prepared_identity,
            IdentityWords.from_fingerprint(
                _merged_identity(model, revision, self.binding)
            ),
            "The merged UMA identity",
            StaleUMAMergeError,
        )

    def revision(self) -> NumericRevision:
        """The raw model's numeric revision; refuses when it drifted from the merge."""
        revision = atomistic_model_revision(self.model)
        if revision.revision_id != self.source_revision_id:
            raise StaleUMAMergeError(
                "The UMA model changed since its MoLE merge; prepare it again."
            )
        return revision

    @checked
    def evaluate_graph(
        self,
        inputs: AtomisticGraphInputs,
        request: AtomisticObservableRequest | None = None,
        /,
    ) -> AtomisticModelOutputs:
        """Evaluate with merged operators; unbound cases are unsuccessful.

        The raw model's kind governs outputs: a property kind refuses
        ``ENERGY``/``ATOM_ENERGY`` before execution and reports no energy.
        """
        merged = self.merged_blocks.value
        return _evaluate(self.model, inputs, request, lambda _: merged, self.binding)


def _admit_binding(model: UMAModel, binding: UMAMoLEBinding, /) -> None:
    """Refuse a binding that is not canonical or not admissible for ``model``."""
    binding.validate()
    configuration = model.configuration
    if model.network.router is None:
        raise TypeError("Only a UMA model with experts has a merged realization.")
    if binding.composition_policy != configuration.composition_policy:
        raise ValueError("The binding's composition policy differs from the model's.")
    if not 0 <= binding.task < len(configuration.tasks):
        raise ValueError("The binding names an unknown task code.")
    if binding.composition[-1][0] >= configuration.max_elements:
        raise ValueError("A bound element exceeds the model's element table.")


def _merged_identity(
    model: UMAModel, revision_id: str, binding: UMAMoLEBinding, /
) -> str:
    return canonical_fingerprint(
        {
            "kind": "prepared-uma-merged-mole",
            "architecture": model.architecture_id,
            "method": model.method_id,
            "source_revision": revision_id,
            "binding": binding.binding_id,
        }
    )


def _binding_coefficients(network: UMANetwork, binding: UMAMoLEBinding, /) -> Array:
    """Expert coefficients ``(E,)`` of one bound system, exactly as routed."""
    router = network.router
    if router is None:
        raise ValueError("A MoLE binding needs a network with experts.")
    system = network.system(
        jnp.asarray([binding.total_charge]),
        jnp.asarray([binding.spin_index]),
        jnp.asarray([binding.task]),
    )
    counts = (
        jnp.asarray(binding.counts(network.configuration.max_elements))[None]
        if router.composition is not None
        else None
    )
    active = jnp.ones((binding.batch_count,), dtype=jnp.bool_)
    return router.coefficients(system, counts, active)[0]


def _edge_rolls(
    configuration: UMAArchitecture, inputs: AtomisticGraphInputs, dtype: np.dtype, /
) -> Array | None:
    """Explicit per-route rolls of the keyed gauge, or ``None`` (zero roll).

    Routes are addressed by the canonical ``inputs.route_identity_fields``
    (material particle identities, explicit case identities and
    representation-invariant image shifts), never by storage slots; the
    explicit input key is the only randomness and stays fixed for one
    evaluation and its derivatives.
    """
    if configuration.roll_gauge != "keyed-route-uniform-roll":
        return None
    key = inputs.key
    if key is None:
        raise ValueError(
            "The keyed UMA roll gauge needs an explicit PRNG key on its graph inputs; "
            "no roll is drawn implicitly."
        )
    return uma_keyed_edge_rolls(key, inputs.route_identity_fields, dtype=dtype)


def _require_selected_graph(
    configuration: UMAArchitecture, inputs: AtomisticGraphInputs, /
) -> None:
    """Refuse a graph not bound to the declared nearest-neighbour selection."""
    selection_id = inputs.graph.selection_id
    if selection_id != configuration.selection_policy.policy_id:
        raise ValueError(
            "UMA evaluates only graphs bound to its declared nearest-neighbour "
            f"selection (UMAPotential.requirements.selection); got {selection_id!r}."
        )


def _require_conditioning(
    network: UMANetwork, conditioning: AtomisticConditioning | None, /
) -> AtomisticConditioning:
    """The declared charge, spin-index and task conditions, or a refusal."""
    if conditioning is None or conditioning.spec.spec_id != network.schema.spec.spec_id:
        raise ValueError(
            "UMA evaluation needs conditioning of its declared charge, spin-index and "
            "task fields (UMAConditionSchema.conditioning)."
        )
    return conditioning


def _model_species(
    configuration: UMAArchitecture, species_ids: ArrayLike, mask: Array, /
) -> Array:
    """Element-table rows of active atoms (zero on inactive slots).

    An active species outside the table is a runtime error of every value
    computed from these rows.
    """
    raw = jnp.asarray(species_ids, dtype=jnp.int32)
    inside = (raw >= 0) & (raw < configuration.max_elements)
    raw = eqx.error_if(
        raw,
        jnp.any(mask & ~inside),
        "Particle species is outside the UMA element table.",
    )
    return jnp.where(mask, raw, 0)


def _layered_species(
    configuration: UMAArchitecture, species_ids: ArrayLike, mask: Array, /
) -> Array:
    """Element-table rows of active owned atoms; zero elsewhere.

    Out-of-table species are reported by `UMAPotential.admitted` as an
    unsuccessful owner, so these rows stay in the table without a runtime error.
    """
    raw = jnp.asarray(species_ids, dtype=jnp.int32)
    inside = (raw >= 0) & (raw < configuration.max_elements)
    return jnp.where(mask & inside, raw, 0)


def _task_codes(configuration: UMAArchitecture, task: Array, /) -> Array:
    return jnp.clip(task, 0, len(configuration.tasks) - 1)


def _energy_available(network: UMANetwork, task: Array, /) -> Array:
    """Per-case availability of the task's energy normalizer."""
    return network.references.energy_available[_task_codes(network.configuration, task)]


def _edge_geometry(network: UMANetwork, /) -> UMAEdgeGeometry:
    configuration = network.configuration
    return UMAEdgeGeometry(
        network.distance,
        network.source_embedding,
        network.target_embedding,
        configuration.layout,
        configuration.cutoff,
    )


def _stream(
    relation: PreparedStreamedRelation,
    callbacks: _EdgeDegreeStream | _BlockStream,
    sources: _SpeciesRows | _SourceRows,
    receivers: _ReceiverRows,
    edges: _EdgeRows,
    edge_active: Array | None,
    /,
) -> StreamedRelationResult:
    """One edge pass: ``sources`` are the relation's columns, ``receivers`` its rows."""
    shape = jax.ShapeDtypeStruct(receivers.residual.shape[1:], receivers.residual.dtype)
    return relation.evaluate(
        StreamedPayloadSpec(shape, shape),
        type(callbacks).edge,
        type(callbacks).epilogue,
        callbacks,
        sources,
        receivers,
        edges,
        edge_active=edge_active,
    )


def _initial_features(
    network: UMANetwork, species: Array, mask: Array, node_system: Array, /
) -> Array:
    """``x[l = 0] = sphere_embedding[Z] + csd`` on active atoms, higher degrees zero."""
    configuration = network.configuration
    dtype = network.dtype
    features = jnp.zeros(
        (species.shape[0], configuration.layout.full_size, configuration.sphere_channels),
        dtype=dtype,
    )
    features = features.at[:, 0, :].set(network.sphere_embedding[species] + node_system)
    return jnp.where(mask[:, None, None], features, jnp.zeros((), dtype=dtype))


def _block_hidden(
    block: UMABlock, features: Array, node_system: Array, mask: Array, /
) -> Array:
    """Edge input of one block: ``norm_1(x)`` with ``csd`` added to ``l = 0``."""
    hidden = block.norm_1(features)
    hidden = hidden.at[:, 0, :].add(node_system)
    return jnp.where(mask[:, None, None], hidden, jnp.zeros((), dtype=features.dtype))


def _block_atomwise(
    network: UMANetwork,
    block: UMABlock,
    features: Array,
    atom_cases: Array,
    mask: Array,
    /,
) -> Array:
    """Node-local residual ``x + atomwise(norm_2(x))`` over bounded row blocks."""
    atomwise = block.atomwise
    norm_2 = block.norm_2
    update = _node_blocks(
        lambda rows, row_cases: atomwise(norm_2(rows), row_cases),
        (features, atom_cases),
        network.node_block,
    )
    return jnp.where(
        mask[:, None, None], features + update, jnp.zeros((), dtype=features.dtype)
    )


def _final_features(network: UMANetwork, features: Array, mask: Array, /) -> Array:
    return jnp.where(
        mask[:, None, None], network.norm(features), jnp.zeros((), dtype=features.dtype)
    )


def _raw_head_energies(
    network: UMANetwork, features: Array, node_task: Array, reduction: DTypeLike, /
) -> Array:
    """Per-atom raw head energy ``e_i`` of each atom's task head (reduction dtype)."""
    configuration = network.configuration
    head_coefficients = network.head_coefficients(network.dtype)
    heads = tuple(
        head if head_coefficients is None else resolve_mixtures(head, head_coefficients)
        for head in network.energy_heads
    )
    head_index = jnp.asarray(configuration.energy_head_of_task, dtype=jnp.int32)[
        node_task
    ]
    raw = jnp.stack(
        [head(features[:, 0, :], node_task) for head in heads],
        axis=0,
    )
    return jnp.take_along_axis(raw, head_index[None, :], axis=0)[0].astype(reduction)


def _potential_atom_energies(
    network: UMANetwork,
    raw: Array,
    node_task: Array,
    species: Array,
    mask: Array,
    reduction: DTypeLike,
    /,
) -> Array:
    """Atom energies of the scalar potential (zero on inactive slots).

    Summed heads: ``rmsd_t e_i + ref_t[Z_i]``. Source ``reduce="mean"`` with
    gradient forces differentiates ``rmsd_t sum_i e_i``, the potential; the
    task energy is then `REPORTED_ENERGY`.
    """
    references = network.references
    energy = raw * references.energy_rmsd[node_task].astype(reduction)
    if network.configuration.energy_reduction == "sum":
        energy = energy + references.element_references[node_task, species].astype(
            reduction
        )
    return jnp.where(mask, energy, jnp.zeros((), dtype=reduction))


def _case_energy_shift(
    network: UMANetwork, case_task: Array, reduction: DTypeLike, /
) -> Array:
    """Per-case energy outside every atom energy: the summed task normalizer mean."""
    if network.configuration.energy_reduction == "sum":
        return network.references.energy_mean[case_task].astype(reduction)
    return jnp.zeros(case_task.shape, dtype=reduction)


class _UMAStagePlan(NamedTuple):
    """One declared system stage: composition routing or channel balancing."""

    stage: AtomisticSystemStage
    block: int | None
    charge: bool
    spin: bool


def _stage_plans(network: UMANetwork, /) -> tuple[_UMAStagePlan, ...]:
    """The model's system stages in execution order.

    Interaction ``0`` is the edge-degree embedding and interaction ``k + 1``
    is block ``k``. The prologue sums atom counts (balancing, ``pnorm``) and,
    with a composition embedding, species counts, then routes the experts and
    realizes every block. Each block is followed by its charge and spin
    balancing; overlapping channel ranges balance in two ordered stages so
    the spin sums read the charge-balanced channels, as in the source.
    """
    configuration = network.configuration
    charge = configuration.charge_channels[1] > configuration.charge_channels[0]
    spin = configuration.spin_channels[1] > configuration.spin_channels[0]
    plans: list[_UMAStagePlan] = []
    if network.router is not None or charge or spin:
        plans.append(
            _UMAStagePlan(
                AtomisticSystemStage("uma-composition-routing", -1), None, False, False
            )
        )
    overlap = (
        charge
        and spin
        and max(configuration.charge_channels[0], configuration.spin_channels[0])
        < min(configuration.charge_channels[1], configuration.spin_channels[1])
    )
    for block in range(configuration.layer_count):
        after = block + 1
        if overlap:
            plans.append(
                _UMAStagePlan(
                    AtomisticSystemStage(f"uma-charge-balance-{block}", after),
                    block,
                    True,
                    False,
                )
            )
            plans.append(
                _UMAStagePlan(
                    AtomisticSystemStage(f"uma-spin-balance-{block}", after),
                    block,
                    False,
                    True,
                )
            )
        elif charge or spin:
            plans.append(
                _UMAStagePlan(
                    AtomisticSystemStage(f"uma-channel-balance-{block}", after),
                    block,
                    charge,
                    spin,
                )
            )
    return tuple(plans)


@final
class UMALayerContext(StrictModule):
    """Owner-local UMA context of one layered domain evaluation (case axis kept).

    ``charge``, ``spin``, ``task`` ``(B,)`` and the system embedding ``system``
    ``(B, C)`` are replicated per case; ``atom_cases`` ``(R,)`` names each
    owned row's case. The composition stage sets ``atom_counts`` ``(B,)`` to
    every case's complete atom count and ``blocks`` to the blocks realized
    with the routed per-case expert coefficients. Before it both are ``None``;
    a model without system stages carries its bound blocks from the start.
    ``key`` is the evaluation's explicit roll key of the keyed roll gauge
    (``None`` for every other gauge).
    """

    __strict_contract__ = True

    atom_cases: Array
    charge: Array
    spin: Array
    task: Array
    system: Array
    atom_counts: Array | None
    blocks: tuple[UMABlock, ...] | None
    key: PRNGKey | None

    def __init__(
        self,
        atom_cases: Array,
        charge: Array,
        spin: Array,
        task: Array,
        system: Array,
        atom_counts: Array | None,
        blocks: tuple[UMABlock, ...] | None,
        key: PRNGKey | None,
        /,
    ) -> None:
        cases = system.shape[0]
        if system.ndim != 2 or any(
            value.shape != (cases,) for value in (charge, spin, task)
        ):
            raise ValueError("UMA case conditions and system embeddings disagree.")
        if atom_cases.ndim != 1 or not jnp.issubdtype(atom_cases.dtype, jnp.integer):
            raise ValueError("atom_cases must be one integer case index per row.")
        if atom_counts is not None and (
            atom_counts.shape != (cases,)
            or not jnp.issubdtype(atom_counts.dtype, jnp.integer)
        ):
            raise ValueError("atom_counts must be one integer count per case.")
        self.atom_cases = atom_cases
        self.charge = charge
        self.spin = spin
        self.task = task
        self.system = system
        self.atom_counts = atom_counts
        self.blocks = blocks
        self.key = key

    @property
    def node_system(self) -> Array:
        """Each owned row's system embedding ``(R, C)``."""
        return self.system[self.atom_cases]

    def realized(self) -> tuple[UMABlock, ...]:
        if self.blocks is None:
            raise ValueError(
                "UMA blocks are realized by the composition stage; run it first."
            )
        return self.blocks


class UMACompositionPartials(NamedTuple):
    """One owner's per-case atom counts ``(B,)`` and species counts ``(B, Z)``."""

    atom_counts: Array
    species_counts: Array | None


class UMABalancePartials(NamedTuple):
    """One owner's per-case sums of the balanced charge/spin scalar channels."""

    charge: Array | None
    spin: Array | None


class UMALayerUpdate(NamedTuple):
    """Updated receiver states, per-receiver layer energies and streamed evidence.

    UMA energy is read out once after the last block, so every interaction's
    layer energy is exactly zero.
    """

    features: Array
    energy: Array
    evidence: StreamedRelationEvidence


def _balance(
    features: Array,
    sums: Array | None,
    target: Array,
    channels: tuple[int, int],
    offset: float,
    mask: Array,
    context: UMALayerContext,
    /,
) -> Array:
    """Apply one channel balance from the complete per-case channel sums."""
    if sums is None or context.atom_counts is None:
        raise ValueError("Channel balancing needs complete channel sums and atom counts.")
    correction = channel_balance_correction(sums, context.atom_counts, target, offset)
    return apply_channel_balance(features, correction[context.atom_cases], mask, channels)


def _reported_task_energy(
    network: UMANetwork,
    raw: Array,
    node_task: Array,
    case_task: Array,
    species: Array,
    mask: Array,
    counts: Array,
    cases: int,
    capacity: int,
    reduction: DTypeLike,
    /,
) -> Array:
    """Source mean-reduced task energy ``(B,)`` in the reduction dtype."""
    # fairchem compute_energy(reduce="mean") -> Normalizer.denorm ->
    # ElementReferences.undo_refs: rmsd_t * (sum_i e_i) / n + mean_t + sum_i ref_t[Z_i].
    references = network.references
    raw_sum = case_reduce(raw, mask, cases, capacity)
    atoms = jnp.maximum(counts, 1).astype(reduction)
    return (
        references.energy_rmsd[case_task].astype(reduction) * raw_sum / atoms
        + references.energy_mean[case_task].astype(reduction)
        + case_reduce(
            references.element_references[node_task, species].astype(reduction),
            mask,
            cases,
            capacity,
        )
    )


def _direct_forces(
    network: UMANetwork,
    features: Array,
    node_task: Array,
    mask: Array,
    reduction: DTypeLike,
    /,
) -> Array:
    """Source direct force head ``(R, 3)``: its ``l = 1`` output, task-denormalized."""
    assert network.force_head is not None
    references = network.references
    direct = network.force_head(features[:, :4, :])[:, 1:4, 0].astype(reduction)
    direct = (
        direct * references.force_rmsd[node_task, None]
        + references.force_mean[node_task, None]
    )
    return jnp.where(mask[:, None], direct, jnp.zeros((), dtype=reduction))


def _direct_stress(
    network: UMANetwork,
    features: Array,
    node_task: Array,
    case_task: Array,
    mask: Array,
    counts: Array,
    cases: int,
    capacity: int,
    dtype: np.dtype,
    reduction: DTypeLike,
    /,
) -> Array:
    """Source ``MLP_Stress_Head`` ``(B, 3, 3)``: scalar trace plus ``l = 2`` part."""
    configuration = network.configuration
    assert network.stress_head is not None and configuration.direct_stress_head
    head = network.stress_head
    head_coefficients = network.head_coefficients(dtype)
    scalar_head = (
        head.scalar
        if head_coefficients is None
        else resolve_mixtures(head.scalar, head_coefficients)
    )
    trace = case_reduce(
        scalar_head(features[:, 0, :], node_task).astype(reduction),
        mask,
        cases,
        capacity,
    )
    anisotropic = case_reduce(
        head.anisotropic(features[:, :9, :])[:, 4:9, 0].astype(reduction),
        mask,
        cases,
        capacity,
    )
    if configuration.direct_stress_head == "mean":
        atoms = jnp.maximum(counts, 1).astype(reduction)
        trace = trace / atoms
        anisotropic = anisotropic / atoms[:, None]
    decomposed = jnp.concatenate(
        [trace[:, None], jnp.zeros((cases, 3), dtype=reduction), anisotropic], axis=1
    )
    stress = decomposed @ jnp.asarray(_CG_CHANGE_RANK_TWO, dtype=reduction)
    references = network.references
    stress = (
        stress * references.stress_rmsd[case_task, None]
        + references.stress_mean[case_task, None]
    )
    return stress.reshape(cases, 3, 3)


def _evaluate(
    owner: UMAPotential | UMAPropertyModel,
    inputs: AtomisticGraphInputs,
    request: AtomisticObservableRequest | None,
    realize: Callable[[Array | None], tuple[UMABlock, ...]],
    binding: UMAMoLEBinding | None,
    /,
) -> AtomisticModelOutputs:
    """Evaluate the shared network for either model kind.

    A property model has no potential: unsupported ``ENERGY``/``ATOM_ENERGY``
    requests refuse before any numerical work, and its outputs carry no
    energy. ``request=None`` means the scalar energy for a potential and
    every supported report for a property model.

    The body is the single ordered source forward (admission, routing,
    edge-degree and block streams, energy readout); its steps share the
    per-case conditioning and row arrays, so only the independent source
    report heads are separate functions.
    """
    potential = isinstance(owner, UMAPotential)
    if request is not None:
        request.require_supported(owner.supported_observables, type(owner).__name__)
    model = owner.network
    configuration = model.configuration
    inputs.require_compatible(owner.scale, owner.capabilities.species_kind)
    conditioning = _require_conditioning(model, inputs.conditioning)
    dtype = jnp.dtype(_compute_dtype(owner.precision))
    reduction = jnp.dtype(owner.precision.reduction_dtype)
    output_dtype = jnp.dtype(owner.precision.output_dtype)
    graph = inputs.graph
    ir = graph.graph
    if ir.edge_mask is None:
        raise ValueError("UMA requires an explicit masked edge relation.")
    topology = graph.topology
    relation: PreparedStreamedRelation = (
        topology.streamed
        if model.streaming is None
        else topology.prepare_streamed(model.streaming)
    )
    cases = inputs.case_count
    capacity = inputs.atom_capacity
    mask = inputs.atom_mask
    species = _model_species(configuration, inputs.species_ids, mask)
    atom_cases = jnp.asarray(inputs.atom_cases, dtype=jnp.int32)
    charge = jnp.asarray(conditioning.value(UMA_CHARGE_FIELD))
    spin = jnp.asarray(conditioning.value(UMA_SPIN_FIELD))
    task = jnp.asarray(conditioning.value(UMA_TASK_FIELD))
    _require_selected_graph(configuration, inputs)
    edge_active = jnp.asarray(ir.edge_mask, dtype=jnp.bool_)
    successful = inputs.successful & _energy_available(model, task)
    counts = case_atom_counts(mask, cases, capacity)
    if (
        configuration.isolated_atoms == "external-reference-lookup"
        and topology.lattice_rank == 0
    ):
        # The source prediction unit substitutes external reference tables for a
        # single non-periodic atom; that table is not part of this model.
        successful = successful & (counts != 1)
    system = model.system(charge, spin, task).astype(dtype)
    full_counts = (
        species_counts(species, mask, cases, capacity, configuration.max_elements)
        if binding is not None or model.composition_routed
        else None
    )
    coefficients = model.mixture_coefficients(system, full_counts, counts)
    if binding is not None:
        assert full_counts is not None
        successful = successful & binding.matches(full_counts, charge, spin, task)
    blocks = realize(coefficients)
    geometry = _edge_geometry(model)
    vectors = jnp.asarray(ir.edges["displacement"]).astype(dtype)
    edges = _EdgeRows(vectors, _edge_rolls(configuration, inputs, dtype))
    node_system = system[atom_cases]
    features = _initial_features(model, species, mask, node_system)
    features = _stream(
        relation,
        _EdgeDegreeStream(model.edge_degree, geometry),
        _SpeciesRows(species),
        _ReceiverRows(features, features, species, atom_cases, mask),
        edges,
        edge_active,
    ).receiver_outputs
    for block in blocks:
        hidden = _block_hidden(block, features, node_system, mask)
        features = _stream(
            relation,
            _BlockStream(block, geometry),
            _SourceRows(hidden, species),
            _ReceiverRows(hidden, features, species, atom_cases, mask),
            edges,
            edge_active,
        ).receiver_outputs
        features = _block_atomwise(model, block, features, atom_cases, mask)
        features = balance_channels(
            features, charge, mask, cases, capacity, configuration.charge_channels, 0.0
        )
        features = balance_channels(
            features, spin, mask, cases, capacity, configuration.spin_channels, 1.0
        )
    features = _final_features(model, features, mask)
    case_task = _task_codes(configuration, task)
    node_task = case_task[atom_cases]
    raw = _raw_head_energies(model, features, node_task, reduction)
    references = model.references
    energy = atom_energy = None
    complete = False
    if potential:
        atom_energy = _potential_atom_energies(
            model, raw, node_task, species, mask, reduction
        )
        complete = (
            configuration.energy_reduction == "mean" or references.complete_atom_energy
        )
        energy = case_energy_totals(
            atom_energy, atom_cases, cases, relation.plan.accumulation
        ) + _case_energy_shift(model, case_task, reduction)
    additional: list[AtomisticObservableValue] = []
    requested = (
        request.observables
        if request is not None
        else (() if potential else owner.supported_observables)
    )
    if AtomisticObservable.REPORTED_ENERGY in requested:
        reported = _reported_task_energy(
            model,
            raw,
            node_task,
            case_task,
            species,
            mask,
            counts,
            cases,
            capacity,
            reduction,
        )
        additional.append(
            AtomisticObservableValue(
                AtomisticObservable.REPORTED_ENERGY,
                UMA_REPORTED_ENERGY_PORT,
                reported.astype(output_dtype),
                successful,
                level=AtomisticConditionLevel.CASE,
                case_count=cases,
                atom_capacity=capacity,
                unit=ELECTRONVOLT,
            )
        )
    if AtomisticObservable.DIRECT_FORCES in requested:
        direct = _direct_forces(model, features, node_task, mask, reduction)
        additional.append(
            AtomisticObservableValue(
                AtomisticObservable.DIRECT_FORCES,
                UMA_DIRECT_FORCE_PORT,
                direct.astype(output_dtype),
                successful & references.force_available[case_task],
                level=AtomisticConditionLevel.ATOM,
                case_count=cases,
                atom_capacity=capacity,
                unit=_FORCE_UNIT,
            )
        )
    if AtomisticObservable.DIRECT_STRESS in requested:
        stress = _direct_stress(
            model,
            features,
            node_task,
            case_task,
            mask,
            counts,
            cases,
            capacity,
            dtype,
            reduction,
        )
        additional.append(
            AtomisticObservableValue(
                AtomisticObservable.DIRECT_STRESS,
                UMA_DIRECT_STRESS_PORT,
                stress.astype(output_dtype),
                successful & references.stress_available[case_task],
                level=AtomisticConditionLevel.CASE,
                case_count=cases,
                atom_capacity=capacity,
                unit=_STRESS_UNIT,
            )
        )
    return AtomisticModelOutputs(
        None if energy is None else energy.astype(output_dtype),
        None
        if atom_energy is None
        else atom_energy.reshape(cases, capacity).astype(output_dtype),
        successful,
        additional=tuple(additional),
        atom_energy_complete=complete,
    )


def _linear_kind(
    linear: object,
    mixture: bool,
    experts: int,
    name: str,
    shared: dict[tuple[int, int], int] | None,
    /,
) -> None:
    """One linear slot: plain, a mixture, or (``moe_single``) a shared mixture.

    The source shares MOLE layers through a cache keyed by ``(in, out, bias)``;
    a biased layer's key holds its own bias tensor, so only bias-free mixtures
    of equal ``(in, out)`` are one shared layer.
    """
    if not mixture:
        if not isinstance(linear, UMALinear):
            raise ValueError(f"{name} must be a plain linear.")
        return
    if (
        shared is not None
        and isinstance(linear, (UMAMixtureLinear, UMASharedMixture))
        and (linear.bias is None)
    ):
        if not isinstance(linear, UMASharedMixture) or linear.expert_count != experts:
            raise ValueError(f"{name} must reference the shared bias-free mixture.")
        key = (linear.in_features, linear.out_features)
        if shared.setdefault(key, linear.index) != linear.index:
            raise ValueError(f"{name} references another shared mixture of its shape.")
        return
    if not isinstance(linear, UMAMixtureLinear) or linear.expert_count != experts:
        raise ValueError(f"{name} must be a {experts}-expert mixture linear.")


def _check_block_mixtures(
    block: UMABlock,
    kind: UMAMixtureKind | None,
    experts: int,
    index: int,
    shared: dict[tuple[int, int], int] | None,
    /,
) -> None:
    """Exactly the source ``mole_type`` replacement set of one block is mixed."""
    match kind:
        case None:
            m0_mixed = orders_mixed = False
        case "so2m0":
            m0_mixed, orders_mixed = True, False
        case "so2":
            m0_mixed = orders_mixed = True
    edgewise = block.edgewise
    for conv_name, conv in (("first", edgewise.first), ("second", edgewise.second)):
        _linear_kind(conv.m0, m0_mixed, experts, f"block {index} {conv_name} m0", shared)
        for order, linear in enumerate(conv.orders, start=1):
            _linear_kind(
                linear,
                orders_mixed,
                experts,
                f"block {index} {conv_name} m{order}",
                shared,
            )
        if conv.radial is not None:
            for layer, linear in enumerate(conv.radial.linears):
                _linear_kind(
                    linear,
                    False,
                    experts,
                    f"block {index} {conv_name} radial {layer}",
                    None,
                )
    atomwise = block.atomwise
    linears = (
        atomwise.linears if isinstance(atomwise, UMAGridAtomwise) else (atomwise.scalar,)
    )
    for layer, linear in enumerate(linears):
        _linear_kind(linear, False, experts, f"block {index} atomwise {layer}", None)


def _check_trunk_declaration(model: UMANetwork, dtype: np.dtype, /) -> None:
    """Element tables, block count, MoLE router and dataset embedding."""
    configuration = model.configuration
    elements = configuration.max_elements
    for name, table, width in (
        ("sphere_embedding", model.sphere_embedding, configuration.sphere_channels),
        ("source_embedding", model.source_embedding, configuration.edge_channels),
        ("target_embedding", model.target_embedding, configuration.edge_channels),
    ):
        if table.shape != (elements, width) or table.dtype != dtype:
            raise ValueError(f"{name} must be ({elements}, {width}) {dtype}.")
    if len(model.blocks) != configuration.layer_count:
        raise ValueError("The UMA potential needs one block per declared layer.")
    if (model.router is None) != (configuration.experts == 0):
        raise ValueError("Experts and the MoLE router must be declared together.")
    if model.router is not None:
        if model.router.expert_count != configuration.experts:
            raise ValueError("The router emits a different expert count.")
        if (model.router.composition is not None) != configuration.composition_embedding:
            raise ValueError("The composition embedding does not match its declaration.")
        if (
            model.router.composition_policy != configuration.composition_policy
            or model.router.normalization != configuration.coefficient_norm
        ):
            raise ValueError("Router policies do not match the architecture.")
    if (model.system.dataset is None) != (configuration.dataset_mapping is None):
        raise ValueError("The dataset embedding does not match its declaration.")


def _check_blocks(
    model: UMANetwork, layout: UMACoefficientLayout, edge_input: int, /
) -> dict[tuple[int, int], int] | None:
    """Every block against the declaration, in block order.

    Returns the shared-mixture slot referenced by each bias-free mixture shape
    (source ``moe_single``), or ``None`` when mixtures are not shared.
    """
    configuration = model.configuration
    channels = configuration.sphere_channels
    shared: dict[tuple[int, int], int] | None = (
        {} if configuration.mixture_sharing else None
    )
    for index, block in enumerate(model.blocks):
        mixed = index in configuration.mixture_layers
        _check_block_mixtures(
            block,
            configuration.mixture_kind if mixed else None,
            configuration.experts,
            index,
            shared,
        )
        first = block.edgewise.first
        if (
            first.layout.layout_id != layout.layout_id
            or first.channels_in != 2 * channels
            or first.channels_out != configuration.hidden_channels
            or block.edgewise.activation != configuration.edge_activation
            or first.radial is None
            or first.radial.input_width != edge_input
        ):
            raise ValueError(f"Block {index} edgewise layer is inconsistent.")
        if block.edgewise.second.channels_out != channels:
            raise ValueError(f"Block {index} second SO(2) convolution width differs.")
        for norm in (block.norm_1, block.norm_2):
            if norm.kind != configuration.normalization or norm.channels != channels:
                raise ValueError(f"Block {index} normalization is inconsistent.")
        is_grid = hasattr(block.atomwise, "grid")
        if is_grid != (configuration.feed_forward == "grid"):
            raise ValueError(f"Block {index} feed-forward kind differs.")
    return shared


def _check_energy_heads(model: UMANetwork, /) -> None:
    """Reachable energy heads of the declared kind and expert wrapper."""
    configuration = model.configuration
    if len(model.energy_heads) != max(configuration.energy_head_of_task) + 1:
        raise ValueError("Every declared energy head must exist and be reachable.")
    wrapped = configuration.head_wrapper == "dataset-experts"
    if wrapped and len(model.energy_heads) != 1:
        raise ValueError("A dataset-expert wrapper is one mixture head.")
    for head in model.energy_heads:
        if len(head.linears) != (3 if configuration.energy_head == "mlp" else 1):
            raise ValueError("An energy head does not match its declared kind.")
        for linear in head.linears:
            if isinstance(linear, UMAMixtureLinear) != wrapped:
                raise ValueError("Energy head mixtures must match the expert wrapper.")
            if (
                isinstance(linear, UMAMixtureLinear)
                and max(configuration.head_experts) >= linear.expert_count
            ):
                raise ValueError("A task maps to an expert the head does not have.")


def _check_direct_heads(model: UMANetwork, /) -> None:
    """Source direct force and stress heads exist exactly as declared."""
    configuration = model.configuration
    if (model.force_head is not None) != configuration.direct_force_head:
        raise ValueError("The direct force head does not match its declaration.")
    if model.force_head is not None and (
        model.force_head.lmax != 1
        or model.force_head.weight.shape[1:] != (1, configuration.sphere_channels)
    ):
        raise ValueError("The direct force head is SO3_Linear(C, 1, lmax=1).")
    if (model.stress_head is not None) != (configuration.direct_stress_head is not None):
        raise ValueError("The direct stress head does not match its declaration.")


def _check_shared_mixtures(
    model: UMANetwork, shared: dict[tuple[int, int], int] | None, /
) -> None:
    """The stored shared mixtures are exactly the referenced bias-free shapes."""
    references_used = {} if shared is None else shared
    if sorted(references_used.values()) != list(range(len(model.shared_mixtures))):
        raise ValueError("Every shared mixture is referenced, and only those exist.")
    for (fan_in, fan_out), slot in references_used.items():
        target = model.shared_mixtures[slot]
        if (
            target.in_features != fan_in
            or target.out_features != fan_out
            or target.bias is not None
            or target.expert_count != model.configuration.experts
        ):
            raise ValueError("A shared mixture differs from its references.")


def _check_components(model: UMANetwork, dtype: np.dtype, /) -> None:
    """Every network component against its declaration, in construction order."""
    configuration = model.configuration
    layout = configuration.layout
    channels = configuration.sphere_channels
    _check_trunk_declaration(model, dtype)
    edge_input = configuration.distance_basis + 2 * configuration.edge_channels
    if model.edge_degree.radial.input_width != edge_input or (
        model.edge_degree.radial.output_width != (layout.lmax + 1) * channels
    ):
        raise ValueError("The edge-degree radial MLP has inconsistent widths.")
    shared = _check_blocks(model, layout, edge_input)
    if model.norm.kind != configuration.normalization or model.norm.channels != channels:
        raise ValueError("The final normalization is inconsistent.")
    _check_energy_heads(model)
    _check_direct_heads(model)
    tasks = len(configuration.tasks)
    if model.references.energy_available.shape != (tasks,) or (
        model.references.element_references.shape != (tasks, configuration.max_elements)
    ):
        raise ValueError("Task references must cover every task and element.")
    if model.schema.tasks != configuration.tasks:
        raise ValueError("The condition schema does not match the tasks.")
    _check_shared_mixtures(model, shared)


def _model_identity(
    model: UMANetwork,
    scale: AtomisticScaleContract,
    precision: AtomisticPrecisionPolicy,
    /,
) -> str:
    """Scientific identity outside the PARAMETER revision (fixed tables and refs)."""
    grids = []
    for block in model.blocks:
        grid = block.edgewise.grid
        grids.append(None if grid is None else grid.grid_id)
        atom_grid = getattr(block.atomwise, "grid", None)
        grids.append(None if atom_grid is None else atom_grid.grid_id)
    frozen = tuple(
        component
        for component in (model.system.charge, model.system.spin, model.system.dataset)
        if isinstance(component, UMAFrozen)
    )
    fixed = array_tree_fingerprint(jax.device_get(frozen))
    return canonical_fingerprint(
        {
            "kind": f"uma-{model.configuration.model_kind}-model",
            "architecture": model.configuration.architecture_id,
            "scale": scale.scale_id,
            "precision": precision.policy_id,
            "references": model.references.references_id,
            "grids": grids,
            "fixed_embeddings": fixed,
        }
    )


for _type in (
    UMAArchitecture,
    UMATaskReferences,
    UMAEnergyHead,
    UMADirectStressHead,
    UMAEdgeGeometry,
    UMANetwork,
    UMAPotential,
    UMAPropertyModel,
    PreparedUMAMergedModel,
    _FrozenUMA,
    _FixedBlocks,
):
    register_artifact_value(f"phydrax.nn.atomistic:{_type.__name__}", _type)


__all__ = [
    "PreparedUMAMergedModel",
    "StaleUMAMergeError",
    "UMA_DEGENERACY_TOLERANCE",
    "UMA_DIRECT_FORCE_PORT",
    "UMA_DIRECT_STRESS_PORT",
    "UMA_EDGE_DEGREE_RESCALE",
    "UMA_ENVELOPE_EXPONENT",
    "UMAArchitecture",
    "UMABalancePartials",
    "UMACompositionPartials",
    "UMADirectStressHead",
    "UMAEdgeGeometry",
    "UMAEnergyHead",
    "UMAEnergyHeadKind",
    "UMAForceRoute",
    "UMAHeadWrapper",
    "UMAIsolatedAtomRoute",
    "UMALayerContext",
    "UMALayerUpdate",
    "UMAMixtureKind",
    "UMAModel",
    "UMAModelKind",
    "UMANetwork",
    "UMAPotential",
    "UMAPropertyModel",
    "UMA_REPORTED_ENERGY_PORT",
    "UMAStressReduce",
    "UMATaskReferences",
]
