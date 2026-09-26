#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Omega_h metric adaptation through a persistent native worker.

The worker (``native/providers/omega_h``) keeps Omega_h and its MPI world alive
across calls and exchanges binary arrays. Named organization is carried as
Omega_h geometric classification: cell classes for blocks/zones/labels, facet
classes for patches/zones/labels, and feature-angle classes for every other
boundary or region interface. Omega_h renumbers global IDs, so lineage is
unknown and coincident IDs never identify preserved entities.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._external_runtime import NativeWorkerCall, NativeWorkerPolicy
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._identity import SemanticProvenance
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellBlock, CellMesh, CellPartition
from .._assembly import MeshPart
from .._association import BRepAssociationTransfer
from .._canonical import canonicalize_cell_mesh, certify_cell_mesh
from .._contracts import (
    MeshingCapability,
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
    MeshingOperation,
    MeshingProviderInfo,
    MeshingSourceKind,
)
from .._distribution import MeshDistribution, MeshPartitionKind
from .._metric import MeshMetricField
from .._organization import MeshLabel, MeshPatch, MeshZone
from .._result import CellMeshingResult, MeshingRuntimeInfo
from .._scope import MeshingEntityKind, MeshingScope
from .._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)
from ._worker import ProviderWorker


_LOCAL_INDEX_LIMIT = np.iinfo(np.int32).max
_TARGET_REVISION = "omega_h-adapted"
_CELL_KINDS = {2: "triangle", 3: "tetrahedron"}
_BASE_OUTPUTS = frozenset(
    {
        "vertex_global_ids",
        "coordinates",
        "vertex_owner_ranks",
        "vertex_owner_indices",
        "metric",
        "cell_global_ids",
        "cells",
        "cell_owner_ranks",
        "cell_owner_indices",
        "cell_class_ids",
        "facet_vertices",
        "facet_class_ids",
        "facet_global_ids",
        "facet_owner_ranks",
    }
)


def _real(value: Any, name: str, /) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _integer(value: Any, name: str, minimum: int, maximum: int, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}].")
    return int(value)


def _flag(value: Any, name: str, /) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")
    return value


class OmegaHFieldTransfer(StrEnum):
    """Omega_h transfer of one declared field through adaptation."""

    # Vertex values, OMEGA_H_LINEAR_INTERP: exact for affine fields.
    LINEAR = "linear"
    # Cell densities, OMEGA_H_CONSERVE: integrals conserved per region class.
    CONSERVE = "conserve"


def _on_vertices(transfer: OmegaHFieldTransfer, /) -> bool:
    """Whether a transfer carries vertex values (otherwise cell values)."""
    match transfer:
        case OmegaHFieldTransfer.LINEAR:
            return True
        case OmegaHFieldTransfer.CONSERVE:
            return False
        case _:
            raise ValueError(f"Unsupported Omega_h field transfer {transfer!r}.")


class OmegaHOptions(StrictModule, NonTrainableState):
    """Explicit Omega_h ``AdaptOpts`` targets, passed to the library verbatim.

    Lengths are edge lengths measured in the metric. ``None`` quality targets
    select Omega_h's dimension defaults (0.3/0.4 in 2-D, 0.2/0.3 in 3-D); the
    adaptation evidence reports the effective values. ``gradation_rate`` is
    Omega_h's metric-gradation rate applied to the requested metric (not a
    neighboring-edge size ratio); ``None`` adapts to the metric ungraded.
    ``feature_angle`` (radians) splits unnamed boundaries at sharp hinges.
    """

    feature_angle: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    gradation_rate: float | None = eqx.field(static=True)
    min_length_desired: float = eqx.field(static=True)
    max_length_desired: float = eqx.field(static=True)
    max_length_allowed: float = eqx.field(static=True)
    min_quality_allowed: float | None = eqx.field(static=True)
    min_quality_desired: float | None = eqx.field(static=True)
    nsliver_layers: int = eqx.field(static=True)
    should_refine: bool = eqx.field(static=True)
    should_coarsen: bool = eqx.field(static=True)
    should_swap: bool = eqx.field(static=True)
    should_coarsen_slivers: bool = eqx.field(static=True)
    should_prevent_coarsen_flip: bool = eqx.field(static=True)
    options_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        feature_angle: float = math.pi / 4,
        maximum_iterations: int = 100,
        gradation_rate: float | None = 1.0,
        min_length_desired: float = 1.0 / math.sqrt(2.0),
        max_length_desired: float = math.sqrt(2.0),
        max_length_allowed: float = 2.0 * math.sqrt(2.0),
        min_quality_allowed: float | None = None,
        min_quality_desired: float | None = None,
        nsliver_layers: int = 4,
        should_refine: bool = True,
        should_coarsen: bool = True,
        should_swap: bool = True,
        should_coarsen_slivers: bool = True,
        should_prevent_coarsen_flip: bool = False,
    ):
        angle = _real(feature_angle, "feature_angle")
        if not 0.0 < angle < math.pi:
            raise ValueError("feature_angle must lie in (0, pi) radians.")
        iterations = _integer(maximum_iterations, "maximum_iterations", 1, 1_000_000)
        rate = None if gradation_rate is None else _real(gradation_rate, "gradation_rate")
        if rate is not None and rate <= 0.0:
            raise ValueError("gradation_rate must be positive or None.")
        lengths = tuple(
            _real(value, name)
            for value, name in (
                (min_length_desired, "min_length_desired"),
                (max_length_desired, "max_length_desired"),
                (max_length_allowed, "max_length_allowed"),
            )
        )
        if not 0.0 < lengths[0] < lengths[1] <= lengths[2]:
            raise ValueError(
                "Length targets require 0 < min_length_desired < max_length_desired "
                "<= max_length_allowed."
            )
        qualities = tuple(
            None if value is None else _real(value, name)
            for value, name in (
                (min_quality_allowed, "min_quality_allowed"),
                (min_quality_desired, "min_quality_desired"),
            )
        )
        if any(value is not None and not 0.0 <= value <= 1.0 for value in qualities) or (
            None not in qualities and qualities[0] > qualities[1]
        ):
            raise ValueError(
                "Quality targets require 0 <= min_quality_allowed <= "
                "min_quality_desired <= 1."
            )
        layers = _integer(nsliver_layers, "nsliver_layers", 0, 99)
        flags = tuple(
            _flag(value, name)
            for value, name in (
                (should_refine, "should_refine"),
                (should_coarsen, "should_coarsen"),
                (should_swap, "should_swap"),
                (should_coarsen_slivers, "should_coarsen_slivers"),
                (should_prevent_coarsen_flip, "should_prevent_coarsen_flip"),
            )
        )
        self.feature_angle = angle
        self.maximum_iterations = iterations
        self.gradation_rate = rate
        self.min_length_desired, self.max_length_desired, self.max_length_allowed = (
            lengths
        )
        self.min_quality_allowed, self.min_quality_desired = qualities
        self.nsliver_layers = layers
        (
            self.should_refine,
            self.should_coarsen,
            self.should_swap,
            self.should_coarsen_slivers,
            self.should_prevent_coarsen_flip,
        ) = flags
        self.options_id = canonical_fingerprint(
            {"kind": "omega-h-options", **self.worker_record()}
        )

    def worker_record(self) -> dict[str, Any]:
        """The adaptation parameters exactly as the worker receives them."""
        return {
            "feature_angle": self.feature_angle,
            "maximum_iterations": self.maximum_iterations,
            "gradation_rate": self.gradation_rate,
            "adapt": {
                "min_length_desired": self.min_length_desired,
                "max_length_desired": self.max_length_desired,
                "max_length_allowed": self.max_length_allowed,
                "min_quality_allowed": self.min_quality_allowed,
                "min_quality_desired": self.min_quality_desired,
                "nsliver_layers": self.nsliver_layers,
                "should_refine": self.should_refine,
                "should_coarsen": self.should_coarsen,
                "should_swap": self.should_swap,
                "should_coarsen_slivers": self.should_coarsen_slivers,
                "should_prevent_coarsen_flip": self.should_prevent_coarsen_flip,
            },
        }


class OmegaHField(StrictModule, NonTrainableState):
    """One field declared for transfer through Omega_h adaptation.

    ``LINEAR`` fields hold vertex values and ``CONSERVE`` fields cell
    densities (integrals are conserved per region class). Rows follow the
    order of ``scope.entity_ids`` (sorted global IDs), like MeshMetricField;
    the scope must cover every vertex or cell of the adapted revision.
    ``diffusion_tolerance`` (``CONSERVE`` only) is Omega_h's relative tolerance
    for diffusing the conservation correction within a region; ``None``
    applies it cell by cell.
    """

    name: str = eqx.field(static=True)
    transfer: OmegaHFieldTransfer = eqx.field(static=True)
    scope: MeshingScope
    values: Array
    diffusion_tolerance: float | None = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        transfer: OmegaHFieldTransfer,
        scope: MeshingScope,
        values: ArrayLike,
        /,
        *,
        diffusion_tolerance: float | None = None,
    ):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Omega_h field names must be nonempty strings.")
        if not isinstance(transfer, OmegaHFieldTransfer):
            raise TypeError("transfer must be OmegaHFieldTransfer.")
        if not isinstance(scope, MeshingScope):
            raise TypeError("scope must be MeshingScope.")
        array = np.asarray(values)
        if array.dtype.kind != "f":
            raise TypeError("Omega_h field values must be floating point.")
        array = array.astype(np.float64)
        if array.ndim not in (1, 2) or array.shape[0] != scope.entity_ids.shape[0]:
            raise ValueError("Omega_h field values need one row (or scalar) per entity.")
        if array.ndim == 2 and not 1 <= array.shape[1] <= 64:
            raise ValueError("Omega_h fields carry 1 to 64 components.")
        if not np.all(np.isfinite(array)):
            raise ValueError("Omega_h field values must be finite.")
        match transfer:
            case OmegaHFieldTransfer.LINEAR:
                if scope.entity_dimension != 0 or diffusion_tolerance is not None:
                    raise ValueError(
                        "LINEAR fields are vertex fields without diffusion tolerance."
                    )
                tolerance = None
            case OmegaHFieldTransfer.CONSERVE:
                if scope.entity_dimension == 0:
                    raise ValueError("CONSERVE fields are cell densities.")
                tolerance = (
                    None
                    if diffusion_tolerance is None
                    else _real(diffusion_tolerance, "diffusion_tolerance")
                )
                if tolerance is not None and tolerance <= 0.0:
                    raise ValueError("diffusion_tolerance must be positive or None.")
            case _:
                raise ValueError(f"Unsupported Omega_h field transfer {transfer!r}.")
        self.name = name.strip()
        self.transfer = transfer
        self.scope = scope
        self.values = jnp.asarray(array)
        self.diffusion_tolerance = tolerance
        self.field_id = canonical_fingerprint(
            {
                "kind": "omega-h-field",
                "name": self.name,
                "transfer": transfer.value,
                "scope": scope.scope_id,
                "values": array_tree_fingerprint(array),
                "diffusion_tolerance": tolerance,
            }
        )


class OmegaHTransferredField(StrictModule, NonTrainableState):
    """A declared field after adaptation, bound to the adapted target."""

    name: str = eqx.field(static=True)
    transfer: OmegaHFieldTransfer = eqx.field(static=True)
    scope: MeshingScope
    values: Array
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        transfer: OmegaHFieldTransfer,
        scope: MeshingScope,
        values: ArrayLike,
        /,
    ):
        array = np.asarray(values, dtype=np.float64)
        if not isinstance(transfer, OmegaHFieldTransfer) or not isinstance(
            scope, MeshingScope
        ):
            raise TypeError("Transferred fields need a transfer kind and target scope.")
        if array.shape[:1] != scope.entity_ids.shape or not np.all(np.isfinite(array)):
            raise ValueError("Transferred field values must be finite scope rows.")
        self.name, self.transfer, self.scope = str(name), transfer, scope
        self.values = jnp.asarray(array)
        self.field_id = canonical_fingerprint(
            {
                "kind": "omega-h-transferred-field",
                "name": self.name,
                "transfer": transfer.value,
                "scope": scope.scope_id,
                "values": array_tree_fingerprint(array),
            }
        )


class OmegaHFieldEvidence(StrictModule, NonTrainableState):
    """Transfer method and, for conservative fields, Omega_h's owned integrals."""

    name: str = eqx.field(static=True)
    transfer: OmegaHFieldTransfer = eqx.field(static=True)
    method: str = eqx.field(static=True)
    diffusion_tolerance: float | None = eqx.field(static=True)
    integral_before: Array | None
    integral_after: Array | None
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        field: OmegaHField,
        method: str,
        integral_before: ArrayLike | None,
        integral_after: ArrayLike | None,
        /,
    ):
        if not isinstance(field, OmegaHField):
            raise TypeError("field must be OmegaHField.")
        before = (
            None if integral_before is None else np.asarray(integral_before, np.float64)
        )
        after = None if integral_after is None else np.asarray(integral_after, np.float64)
        conservative = field.transfer is OmegaHFieldTransfer.CONSERVE
        if (before is None) == conservative or (after is None) == conservative:
            raise ValueError("Integrals are reported exactly for conservative fields.")
        self.name, self.transfer, self.method = field.name, field.transfer, str(method)
        self.diffusion_tolerance = field.diffusion_tolerance
        self.integral_before = None if before is None else jnp.asarray(before)
        self.integral_after = None if after is None else jnp.asarray(after)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "omega-h-field-evidence",
                "field": field.field_id,
                "method": self.method,
                "integral_before": None if before is None else before.tolist(),
                "integral_after": None if after is None else after.tolist(),
            }
        )


class OmegaHClassification(StrictModule, NonTrainableState):
    """Omega_h class IDs of the named organization carried through adaptation.

    Cell class ``k`` (class_dim = d) is the region of block ``cell_blocks[k]``,
    zone ``cell_zones[k]`` (or none), and labels ``cell_labels[k]``. Facet
    class ``k`` (class_dim = d - 1) lies in patches ``facet_patches[k]``, zone
    ``facet_zones[k]``, and labels ``facet_labels[k]``. Facet classes at or
    beyond ``len(facet_patches)`` are unnamed boundary or region-interface
    components split at hinges sharper than ``feature_angle``.
    """

    cell_blocks: tuple[str, ...] = eqx.field(static=True)
    cell_zones: tuple[str | None, ...] = eqx.field(static=True)
    cell_labels: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    facet_patches: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    facet_zones: tuple[str | None, ...] = eqx.field(static=True)
    facet_labels: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    feature_angle: float = eqx.field(static=True)
    classification_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        cell_blocks: tuple[str, ...],
        cell_zones: tuple[str | None, ...],
        cell_labels: tuple[tuple[str, ...], ...],
        facet_patches: tuple[tuple[str, ...], ...],
        facet_zones: tuple[str | None, ...],
        facet_labels: tuple[tuple[str, ...], ...],
        feature_angle: float,
    ):
        if (
            not cell_blocks
            or len(cell_zones) != len(cell_blocks)
            or len(cell_labels) != len(cell_blocks)
            or len(facet_zones) != len(facet_patches)
            or len(facet_labels) != len(facet_patches)
        ):
            raise ValueError("Omega_h class tables must align by class ID.")
        self.cell_blocks = tuple(cell_blocks)
        self.cell_zones = tuple(cell_zones)
        self.cell_labels = tuple(tuple(value) for value in cell_labels)
        self.facet_patches = tuple(tuple(value) for value in facet_patches)
        self.facet_zones = tuple(facet_zones)
        self.facet_labels = tuple(tuple(value) for value in facet_labels)
        self.feature_angle = _real(feature_angle, "feature_angle")
        self.classification_id = canonical_fingerprint(
            {
                "kind": "omega-h-classification",
                "cell_blocks": self.cell_blocks,
                "cell_zones": self.cell_zones,
                "cell_labels": self.cell_labels,
                "facet_patches": self.facet_patches,
                "facet_zones": self.facet_zones,
                "facet_labels": self.facet_labels,
                "feature_angle": self.feature_angle,
            }
        )


class OmegaHPartition(StrictModule, NonTrainableState):
    """One rank's ghosted Omega_h partition in native local order.

    Residence is membership in the rank's arrays; owners are (rank, local
    index) pairs, and ``*_owned`` masks mark the authoritative copies (the
    rest are ghosts). Facets are those classified on model faces (class_dim =
    d - 1). Global IDs identify entities of this output revision only.
    """

    rank: int = eqx.field(static=True)
    vertex_global_ids: Array
    coordinates: Array
    metric: Array
    vertex_owner_ranks: Array
    vertex_owner_indices: Array
    vertex_owned: Array
    cell_global_ids: Array
    cells: Array
    cell_class_ids: Array
    cell_owner_ranks: Array
    cell_owner_indices: Array
    cell_owned: Array
    facet_global_ids: Array
    facet_vertices: Array
    facet_class_ids: Array
    facet_owner_ranks: Array
    facet_owned: Array
    fields: tuple[Array, ...]
    partition_id: str = eqx.field(static=True)

    def __init__(self, rank: int, arrays: _RankArrays, /):
        if not isinstance(arrays, _RankArrays):
            raise TypeError("arrays must be validated Omega_h rank arrays.")
        self.rank = _integer(rank, "rank", 0, _LOCAL_INDEX_LIMIT)
        self.vertex_global_ids = jnp.asarray(arrays.vertex_global_ids)
        self.coordinates = jnp.asarray(arrays.coordinates)
        self.metric = jnp.asarray(arrays.metric)
        self.vertex_owner_ranks = jnp.asarray(arrays.vertex_owner_ranks)
        self.vertex_owner_indices = jnp.asarray(arrays.vertex_owner_indices)
        self.vertex_owned = jnp.asarray(arrays.vertex_owner_ranks == rank)
        self.cell_global_ids = jnp.asarray(arrays.cell_global_ids)
        self.cells = jnp.asarray(arrays.cells)
        self.cell_class_ids = jnp.asarray(arrays.cell_class_ids)
        self.cell_owner_ranks = jnp.asarray(arrays.cell_owner_ranks)
        self.cell_owner_indices = jnp.asarray(arrays.cell_owner_indices)
        self.cell_owned = jnp.asarray(arrays.cell_owner_ranks == rank)
        self.facet_global_ids = jnp.asarray(arrays.facet_global_ids)
        self.facet_vertices = jnp.asarray(arrays.facet_vertices)
        self.facet_class_ids = jnp.asarray(arrays.facet_class_ids)
        self.facet_owner_ranks = jnp.asarray(arrays.facet_owner_ranks)
        self.facet_owned = jnp.asarray(arrays.facet_owner_ranks == rank)
        self.fields = tuple(jnp.asarray(values) for values in arrays.fields)
        self.partition_id = canonical_fingerprint(
            {
                "kind": "omega-h-partition",
                "rank": self.rank,
                "arrays": array_tree_fingerprint(tuple(arrays)),
            }
        )

    @property
    def vertex_ghosts(self) -> Array:
        return ~self.vertex_owned

    @property
    def cell_ghosts(self) -> Array:
        return ~self.cell_owned

    @property
    def facet_ghosts(self) -> Array:
        return ~self.facet_owned


class OmegaHAdaptationEvidence(StrictModule, NonTrainableState):
    """What Omega_h was asked to do, what it achieved, and which session ran it.

    Quality is Omega_h's element quality measured in the metric (1 for the
    regular simplex); lengths are edge lengths in the adapted metric (the
    extremes of Omega_h's length histogram).
    """

    options: OmegaHOptions
    min_quality_allowed: float = eqx.field(static=True)
    min_quality_desired: float = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    minimum_quality: float = eqx.field(static=True)
    maximum_quality: float = eqx.field(static=True)
    minimum_length: float = eqx.field(static=True)
    maximum_length: float = eqx.field(static=True)
    vertex_count: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    facet_count: int = eqx.field(static=True)
    generated_facet_classes: int = eqx.field(static=True)
    fields: tuple[OmegaHFieldEvidence, ...]
    ranks: int = eqx.field(static=True)
    session_id: str = eqx.field(static=True)
    identity_id: str = eqx.field(static=True)
    sequence: int = eqx.field(static=True)
    peak_rss_bytes: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        options: OmegaHOptions,
        fields: tuple[OmegaHFieldEvidence, ...],
        record: Mapping[str, Any],
        call: NativeWorkerCall,
        identity_id: str,
        /,
    ):
        if not isinstance(options, OmegaHOptions):
            raise TypeError("options must be OmegaHOptions.")
        if not all(isinstance(value, OmegaHFieldEvidence) for value in fields):
            raise TypeError("fields must contain OmegaHFieldEvidence values.")
        self.options = options
        self.min_quality_allowed = float(record["options"]["min_quality_allowed"])
        self.min_quality_desired = float(record["options"]["min_quality_desired"])
        self.iterations = int(record["iterations"])
        self.minimum_quality = float(record["minimum_quality"])
        self.maximum_quality = float(record["maximum_quality"])
        self.minimum_length = float(record["minimum_length"])
        self.maximum_length = float(record["maximum_length"])
        self.vertex_count = int(record["vertex_count"])
        self.cell_count = int(record["cell_count"])
        self.facet_count = int(record["facet_count"])
        self.generated_facet_classes = int(
            record["classification"]["generated_facet_classes"]
        )
        self.fields = tuple(fields)
        self.ranks = int(record["ranks"])
        self.session_id = str(call.evidence["session_id"])
        self.identity_id = str(identity_id)
        self.sequence = call.sequence
        self.peak_rss_bytes = int(call.evidence["peak_rss_bytes"])
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "omega-h-adaptation-evidence",
                "options": options.options_id,
                "record": dict(record),
                "fields": [value.evidence_id for value in fields],
                "identity": self.identity_id,
            }
        )


class OmegaHAdaptationResult(StrictModule, NonTrainableState):
    """Adapted Omega_h carrier with native rank residence and transfer evidence.

    ``partitions`` always holds every rank's ghosted partition. ``target``,
    ``metric`` and ``fields`` describe the certified global carrier; they are
    present for serial runs and for distributed runs with ``gather=True``,
    which also yields the ``distribution`` of Omega_h's ownership and ghosts.
    """

    source_id: str = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    partitions: tuple[OmegaHPartition, ...]
    classification: OmegaHClassification
    evidence: OmegaHAdaptationEvidence
    target: CellMeshingResult | None
    metric: MeshMetricField | None
    fields: tuple[OmegaHTransferredField, ...]
    distribution: MeshDistribution | None
    provider: MeshingProviderInfo
    runtime: MeshingRuntimeInfo
    provenance: SemanticProvenance
    derivative_mode: MeshingDerivativeMode = eqx.field(static=True)
    # Omega_h renumbers globals; coincident IDs never identify preserved
    # entities and must not be used to fabricate a transfer.
    lineage_status: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: CellMeshingResult,
        metric_id: str,
        partitions: tuple[OmegaHPartition, ...],
        classification: OmegaHClassification,
        evidence: OmegaHAdaptationEvidence,
        target: _GatheredTarget | None,
        distribution: MeshDistribution | None,
        provider: MeshingProviderInfo,
        runtime: MeshingRuntimeInfo,
        provenance: SemanticProvenance,
        /,
    ):
        if len(partitions) != evidence.ranks or any(
            partition.rank != rank for rank, partition in enumerate(partitions)
        ):
            raise ValueError("Omega_h results need one partition per rank, in order.")
        if target is None and distribution is not None:
            raise ValueError("A distribution requires the gathered target.")
        self.source_id = source.result_id
        self.source_mesh_id = source.mesh.mesh_id
        self.metric_id = str(metric_id)
        self.partitions = tuple(partitions)
        self.classification = classification
        self.evidence = evidence
        self.target = None if target is None else target.result
        self.metric = None if target is None else target.metric
        self.fields = () if target is None else target.fields
        self.distribution = distribution
        self.provider, self.runtime, self.provenance = provider, runtime, provenance
        self.derivative_mode = MeshingDerivativeMode.NONDIFFERENTIABLE
        self.lineage_status = "unknown"
        self.result_id = canonical_fingerprint(
            {
                "kind": "omega-h-adaptation-result",
                "source": self.source_id,
                "metric": self.metric_id,
                "partitions": [value.partition_id for value in self.partitions],
                "classification": classification.classification_id,
                "evidence": evidence.evidence_id,
                "target": None if self.target is None else self.target.result_id,
                "fields": [value.field_id for value in self.fields],
                "distribution": (
                    None if distribution is None else distribution.distribution_id
                ),
                "provenance": provenance.semantic_id,
            }
        )


class _RankArrays(NamedTuple):
    """One rank's verified output in native order (NumPy, explicit dtypes)."""

    vertex_global_ids: np.ndarray
    coordinates: np.ndarray
    metric: np.ndarray
    vertex_owner_ranks: np.ndarray
    vertex_owner_indices: np.ndarray
    cell_global_ids: np.ndarray
    cells: np.ndarray
    cell_class_ids: np.ndarray
    cell_owner_ranks: np.ndarray
    cell_owner_indices: np.ndarray
    facet_global_ids: np.ndarray
    facet_vertices: np.ndarray
    facet_class_ids: np.ndarray
    facet_owner_ranks: np.ndarray
    fields: tuple[np.ndarray, ...]


class _Carrier(NamedTuple):
    """Worker input in sorted-vertex-ID order plus the class tables."""

    arrays: dict[str, np.ndarray]
    classification: OmegaHClassification


class _GatheredTarget(NamedTuple):
    result: CellMeshingResult
    metric: MeshMetricField
    fields: tuple[OmegaHTransferredField, ...]
    cell_owner_ranks: np.ndarray
    cell_global_ids: np.ndarray


def _failure(category: MeshingFailureCategory, message: str, /) -> MeshingFailure:
    return MeshingFailure(category, message, stage="omega_h")


def _conversion(message: str, /) -> MeshingFailure:
    return _failure(
        MeshingFailureCategory.CONVERSION_FAILED, f"Omega_h output: {message}"
    )


def _rows_of(identifiers: np.ndarray, requested: ArrayLike, /) -> np.ndarray:
    """Rows of the unique ``identifiers`` holding each ``requested`` ID."""
    wanted = np.asarray(requested, dtype=np.int64)
    order = np.argsort(identifiers, kind="stable")
    located = np.minimum(
        np.searchsorted(identifiers, wanted, sorter=order), identifiers.size - 1
    )
    rows = order[located]
    if not np.array_equal(identifiers[rows], wanted):
        raise ValueError("A scope names entities outside the source mesh.")
    return rows


def _entity_scope(mesh: CellMesh, dimension: int, identifiers: np.ndarray, /):
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        mesh.entity_set(dimension).entity_set_id,
        identifiers,
    )


def _facet_rows(mesh: CellMesh, /) -> np.ndarray:
    """Local vertex rows of the facet entity set, in entity-set order."""
    match mesh.topological_dimension:
        case 2:
            return np.asarray(mesh.connectivity.edges, dtype=np.int64)
        case 3:
            return np.asarray(mesh.connectivity.faces, dtype=np.int64)
        case _:
            raise ValueError("Omega_h facets exist for 2-D and 3-D simplices only.")


def _source_mesh(
    source: CellMeshingResult, transfer: BRepAssociationTransfer | None, /
) -> tuple[CellMesh, int]:
    mesh = source.mesh
    dimension = mesh.topological_dimension
    kind = _CELL_KINDS.get(dimension)
    if (
        kind is None
        or mesh.ambient_dimension != dimension
        or any(
            not isinstance(block, CellBlock) or block.cell_kind != kind
            for block in mesh.blocks
        )
    ):
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
            "Omega_h adapts affine planar triangles and volume tetrahedra only.",
        )
    if (
        source.boundary is not None
        or source.attributes
        or (source.associations and transfer is None)
    ):
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
            "Omega_h carries blocks, zones, patches, labels, and B-Rep associations "
            "through an association transfer; boundary surface models, attributes, "
            "and untransferred geometry associations cannot follow adaptation.",
        )
    if source.associations:
        transfer.source_associations(source)
    if any(
        value.scope.entity_dimension not in (dimension - 1, dimension)
        for value in (*source.zones, *source.labels)
    ) or any(patch.scope.entity_dimension != dimension - 1 for patch in source.patches):
        raise _failure(
            MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
            "Omega_h classification carries cell and facet zones/labels and facet "
            "patches only.",
        )
    return mesh, dimension


def _classes(keys: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Canonical (lexicographic) class table and each row's class ID."""
    table, inverse = np.unique(keys, axis=0, return_inverse=True)
    return table, inverse.reshape(-1).astype(np.int32)


def _cell_classes(
    source: CellMeshingResult, cell_ids: np.ndarray, blocks: np.ndarray, /
) -> tuple[np.ndarray, tuple, tuple, tuple]:
    """Region class per cell: one class per distinct (block, zone, labels)."""
    dimension = source.mesh.topological_dimension
    zones = tuple(
        zone for zone in source.zones if zone.scope.entity_dimension == dimension
    )
    labels = tuple(
        label for label in source.labels if label.scope.entity_dimension == dimension
    )
    zone_index = np.full(cell_ids.shape, -1, dtype=np.int64)
    membership = np.zeros((cell_ids.size, len(labels)), dtype=np.int64)
    for index, zone in enumerate(zones):
        zone_index[_rows_of(cell_ids, zone.scope.entity_ids)] = index
    for index, label in enumerate(labels):
        membership[_rows_of(cell_ids, label.scope.entity_ids), index] = 1
    table, classes = _classes(np.column_stack((blocks, zone_index, membership)))
    names = tuple(block.name for block in source.mesh.blocks)
    return (
        classes,
        tuple(names[row[0]] for row in table),
        tuple(None if row[1] < 0 else zones[row[1]].name for row in table),
        tuple(tuple(labels[k].name for k in np.flatnonzero(row[2:])) for row in table),
    )


def _facet_classes(
    source: CellMeshingResult, inverse: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, tuple, tuple, tuple]:
    """Facet class per named facet: one per distinct (zone, patches, labels)."""
    mesh = source.mesh
    dimension = mesh.topological_dimension - 1
    facet_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    patches = source.patches
    zones = tuple(
        zone for zone in source.zones if zone.scope.entity_dimension == dimension
    )
    labels = tuple(
        label for label in source.labels if label.scope.entity_dimension == dimension
    )
    zone_index = np.full(facet_ids.shape, -1, dtype=np.int64)
    membership = np.zeros((facet_ids.size, len(patches) + len(labels)), dtype=np.int64)
    for index, patch in enumerate(patches):
        membership[_rows_of(facet_ids, patch.scope.entity_ids), index] = 1
    for index, zone in enumerate(zones):
        zone_index[_rows_of(facet_ids, zone.scope.entity_ids)] = index
    for index, label in enumerate(labels):
        membership[_rows_of(facet_ids, label.scope.entity_ids), len(patches) + index] = 1
    named = (zone_index >= 0) | np.any(membership > 0, axis=1)
    keys = np.column_stack((zone_index, membership))[named]
    table, classes = _classes(keys) if keys.size else (keys, np.zeros(0, np.int32))
    rows = inverse[_facet_rows(mesh)[named]].astype(np.int32)
    return (
        rows.reshape(-1, mesh.topological_dimension),
        classes,
        tuple(
            tuple(patches[k].name for k in np.flatnonzero(row[1 : 1 + len(patches)]))
            for row in table
        ),
        tuple(None if row[0] < 0 else zones[row[0]].name for row in table),
        tuple(
            tuple(labels[k].name for k in np.flatnonzero(row[1 + len(patches) :]))
            for row in table
        ),
    )


def _checked_metric(mesh: CellMesh, metric: MeshMetricField, vertex_ids: np.ndarray, /):
    scope = metric.scope
    if (
        scope.source_id != mesh.mesh_id
        or scope.source_revision != mesh.numeric_version
        or scope.entity_kind is not MeshingEntityKind.MESH
        or scope.entity_dimension != 0
        or scope.entity_set_id != mesh.entity_set(0).entity_set_id
        or not np.array_equal(np.asarray(scope.entity_ids), np.sort(vertex_ids))
    ):
        raise ValueError("Metric scope must cover exactly this mesh revision's vertices.")
    dimension = mesh.topological_dimension
    values = np.asarray(metric.values, dtype=np.float64)
    if values.shape != (vertex_ids.size, dimension, dimension):
        raise ValueError("Metric matrix dimension must match the Omega_h mesh dimension.")
    eigenvalues = np.linalg.eigvalsh(values)
    if (
        np.any(eigenvalues < metric.maximum_size**-2 * (1 - 1e-12))
        or np.any(eigenvalues > metric.minimum_size**-2 * (1 + 1e-12))
        or np.any(
            np.sqrt(eigenvalues[:, -1] / eigenvalues[:, 0])
            > metric.maximum_anisotropy * (1 + 1e-12)
        )
    ):
        raise ValueError("Input metrics exceed their declared size/anisotropy bounds.")
    # INRIA lower-triangular row-major packing: xx, xy, yy[, xz, yz, zz].
    return values[:, *np.tril_indices(dimension)]


def _field_rows(mesh: CellMesh, field: OmegaHField, identifiers: np.ndarray, /):
    dimension = 0 if _on_vertices(field.transfer) else mesh.topological_dimension
    scope = field.scope
    if (
        scope.source_id != mesh.mesh_id
        or scope.source_revision != mesh.numeric_version
        or scope.entity_kind is not MeshingEntityKind.MESH
        or scope.entity_dimension != dimension
        or scope.entity_set_id != mesh.entity_set(dimension).entity_set_id
        or scope.entity_ids.shape != identifiers.shape
    ):
        raise ValueError(
            f"Field {field.name!r} must cover every "
            f"{'vertex' if dimension == 0 else 'cell'} of this mesh revision."
        )
    # Scope rows follow sorted IDs; the worker receives native carrier order.
    return np.asarray(field.values, dtype=np.float64)[
        _rows_of(np.asarray(scope.entity_ids, dtype=np.int64), identifiers)
    ]


def _carrier(
    source: CellMeshingResult,
    metric: MeshMetricField,
    fields: tuple[OmegaHField, ...],
    options: OmegaHOptions,
    /,
) -> _Carrier:
    mesh = source.mesh
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    order = np.argsort(vertex_ids, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    packed = _checked_metric(mesh, metric, vertex_ids)
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    blocks = np.repeat(
        np.arange(len(mesh.blocks)), [block.cell_count for block in mesh.blocks]
    )
    cell_classes, cell_blocks, cell_zones, cell_labels = _cell_classes(
        source, cell_ids, blocks
    )
    (
        facet_vertices,
        facet_classes,
        facet_patches,
        facet_zones,
        facet_labels,
    ) = _facet_classes(source, inverse)
    arrays = {
        "vertex_global_ids": vertex_ids[order],
        "coordinates": np.asarray(mesh.coordinates, dtype=np.float64)[order],
        "cells": inverse[cells].astype(np.int32),
        "cell_global_ids": cell_ids,
        "cell_class_ids": cell_classes,
        "metric": np.ascontiguousarray(packed),
        "facet_vertices": facet_vertices,
        "facet_class_ids": facet_classes,
    }
    for index, field in enumerate(fields):
        identifiers = vertex_ids[order] if _on_vertices(field.transfer) else cell_ids
        values = _field_rows(mesh, field, identifiers)
        arrays[f"field-{index}"] = np.ascontiguousarray(
            values.reshape(values.shape[0], -1)
        )
    classification = OmegaHClassification(
        cell_blocks=cell_blocks,
        cell_zones=cell_zones,
        cell_labels=cell_labels,
        facet_patches=facet_patches,
        facet_zones=facet_zones,
        facet_labels=facet_labels,
        feature_angle=options.feature_angle,
    )
    return _Carrier(arrays, classification)


def _refuse_resources(mesh: CellMesh, limits: MeshingLimits, /) -> None:
    vertices = mesh.coordinates.shape[0]
    cells = sum(block.cell_count for block in mesh.blocks)
    connectivity = sum(block.vertices.size for block in mesh.blocks)
    if (
        vertices > limits.maximum_vertices
        or cells > limits.maximum_cells
        or connectivity > limits.maximum_connectivity_entries
    ):
        raise _failure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Omega_h input exceeds configured entity limits.",
        )


def _refuse_bytes(arrays: Mapping[str, np.ndarray], limits: MeshingLimits, /) -> None:
    # Payload plus a manifest allowance; the exchange enforces the exact bound.
    estimate = sum(value.nbytes for value in arrays.values()) + 4096 + 256 * len(arrays)
    if estimate > limits.maximum_data_bytes:
        raise _failure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Omega_h input exceeds maximum_data_bytes.",
        )


def _require(condition: bool, message: str, /) -> None:
    if not condition:
        raise _conversion(message)


def _array(arrays: Mapping[str, np.ndarray], name: str, dtype, shape: tuple, /):
    value = arrays[name]
    _require(
        value.dtype == dtype
        and value.ndim == len(shape)
        and all(
            expected < 0 or size == expected
            for size, expected in zip(value.shape, shape, strict=True)
        ),
        f"array {name!r} has an unexpected dtype or shape",
    )
    return np.asarray(value)


def _rank_arrays(
    arrays: Mapping[str, np.ndarray],
    dimension: int,
    fields: tuple[OmegaHField, ...],
    ranks: int,
    /,
) -> _RankArrays:
    expected = _BASE_OUTPUTS | {f"field-{index}" for index in range(len(fields))}
    _require(set(arrays) == expected, "rank arrays do not match the protocol")
    vertex_ids = _array(arrays, "vertex_global_ids", np.int64, (-1,))
    cell_ids = _array(arrays, "cell_global_ids", np.int64, (-1,))
    facet_ids = _array(arrays, "facet_global_ids", np.int64, (-1,))
    nv, nc, nf = vertex_ids.size, cell_ids.size, facet_ids.size
    width = dimension * (dimension + 1) // 2
    metric = _array(arrays, "metric", np.float64, (nv, width))
    unpacked = np.zeros((nv, dimension, dimension), dtype=np.float64)
    rows, columns = np.tril_indices(dimension)
    unpacked[:, rows, columns] = metric
    unpacked[:, columns, rows] = metric
    result = _RankArrays(
        vertex_ids,
        _array(arrays, "coordinates", np.float64, (nv, dimension)),
        unpacked,
        _array(arrays, "vertex_owner_ranks", np.int32, (nv,)),
        _array(arrays, "vertex_owner_indices", np.int32, (nv,)),
        cell_ids,
        _array(arrays, "cells", np.int32, (nc, dimension + 1)),
        _array(arrays, "cell_class_ids", np.int32, (nc,)),
        _array(arrays, "cell_owner_ranks", np.int32, (nc,)),
        _array(arrays, "cell_owner_indices", np.int32, (nc,)),
        facet_ids,
        _array(arrays, "facet_vertices", np.int32, (nf, dimension)),
        _array(arrays, "facet_class_ids", np.int32, (nf,)),
        _array(arrays, "facet_owner_ranks", np.int32, (nf,)),
        tuple(
            _array(
                arrays,
                f"field-{index}",
                np.float64,
                (
                    nv if _on_vertices(field.transfer) else nc,
                    math.prod(field.values.shape[1:]),
                ),
            ).reshape((-1, *field.values.shape[1:]))
            for index, field in enumerate(fields)
        ),
    )
    _require(
        np.all(np.isfinite(result.coordinates)) and np.all(np.isfinite(metric)),
        "coordinates and metric must be finite",
    )
    _require(
        np.all((result.cells >= 0) & (result.cells < nv))
        and np.all((result.facet_vertices >= 0) & (result.facet_vertices < nv)),
        "connectivity references missing local vertices",
    )
    _require(
        min(
            np.min(vertex_ids, initial=0),
            np.min(cell_ids, initial=0),
            np.min(facet_ids, initial=0),
            np.min(result.cell_class_ids, initial=0),
            np.min(result.facet_class_ids, initial=0),
        )
        >= 0,
        "global and class IDs must be nonnegative",
    )
    _require(
        all(
            np.all((owners >= 0) & (owners < ranks))
            for owners in (
                result.vertex_owner_ranks,
                result.cell_owner_ranks,
                result.facet_owner_ranks,
            )
        ),
        "owner ranks are out of range",
    )
    _require(
        all(np.unique(ids).size == ids.size for ids in (vertex_ids, cell_ids, facet_ids)),
        "local global IDs must be unique per rank",
    )
    return result


def _owner_rows(
    counts: np.ndarray,
    owner_ranks: np.ndarray,
    owner_indices: np.ndarray,
    identifiers: np.ndarray,
    kind: str,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve every resident copy onto its owner copy (vectorized)."""
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    resident_rank = np.repeat(np.arange(counts.size), counts)
    local = np.arange(identifiers.size) - offsets[resident_rank]
    indices = owner_indices.astype(np.int64)
    _require(
        np.all((indices >= 0) & (indices < counts[owner_ranks])),
        f"{kind} owner indices are out of range",
    )
    rows = offsets[owner_ranks] + indices
    _require(
        np.array_equal(identifiers[rows], identifiers)
        and np.array_equal(owner_ranks[rows], resident_rank[rows])
        and np.array_equal(indices[rows], local[rows]),
        f"{kind} owner references do not resolve to authoritative copies",
    )
    owned = owner_ranks == resident_rank
    _require(
        np.unique(identifiers[owned]).size == np.count_nonzero(owned),
        f"{kind} global IDs have more than one owner",
    )
    return rows, owned


def _merged(ranks: Sequence[_RankArrays], /):
    """Concatenate ranks, verify every copy against its owner, keep owners."""
    columns = [item._asdict() for item in ranks]
    counts = {
        name: np.asarray([column[name].shape[0] for column in columns], dtype=np.int64)
        for name in ("vertex_global_ids", "cell_global_ids", "facet_global_ids")
    }

    def stacked(name: str) -> np.ndarray:
        return np.concatenate([column[name] for column in columns])

    vertex_ids = stacked("vertex_global_ids")
    vertex_rows, vertex_owned = _owner_rows(
        counts["vertex_global_ids"],
        stacked("vertex_owner_ranks"),
        stacked("vertex_owner_indices"),
        vertex_ids,
        "vertex",
    )
    coordinates, metric = stacked("coordinates"), stacked("metric")
    vertex_offsets = np.concatenate(([0], np.cumsum(counts["vertex_global_ids"])))
    cell_rank = np.repeat(np.arange(len(ranks)), counts["cell_global_ids"])
    cells = vertex_ids[vertex_offsets[cell_rank, None] + stacked("cells")]
    cell_ids = stacked("cell_global_ids")
    cell_rows, cell_owned = _owner_rows(
        counts["cell_global_ids"],
        stacked("cell_owner_ranks"),
        stacked("cell_owner_indices"),
        cell_ids,
        "cell",
    )
    cell_classes = stacked("cell_class_ids")
    field_values = tuple(
        np.concatenate([item.fields[index] for item in ranks])
        for index in range(len(ranks[0].fields))
    )
    _require(
        np.array_equal(coordinates[vertex_rows], coordinates)
        and np.array_equal(metric[vertex_rows], metric)
        and np.array_equal(cells[cell_rows], cells)
        and np.array_equal(cell_classes[cell_rows], cell_classes),
        "resident copies disagree with their owners",
    )
    facet_rank = np.repeat(np.arange(len(ranks)), counts["facet_global_ids"])
    facet_ids = stacked("facet_global_ids")
    facets = np.sort(
        vertex_ids[vertex_offsets[facet_rank, None] + stacked("facet_vertices")], axis=1
    )
    facet_classes = stacked("facet_class_ids")
    facet_owned = stacked("facet_owner_ranks") == facet_rank
    unique_facets, facet_group = np.unique(facet_ids, return_inverse=True)
    _require(
        np.array_equal(
            np.bincount(facet_group[facet_owned], minlength=unique_facets.size),
            np.ones(unique_facets.size, dtype=np.int64),
        ),
        "every resident facet needs exactly one owner copy",
    )
    owner_of_group = np.empty(unique_facets.size, dtype=np.int64)
    owner_of_group[facet_group[facet_owned]] = np.flatnonzero(facet_owned)
    facet_rows = owner_of_group[facet_group]
    _require(
        np.array_equal(facets[facet_rows], facets)
        and np.array_equal(facet_classes[facet_rows], facet_classes),
        "resident facet copies disagree with their owners",
    )
    return {
        "vertex": (vertex_ids, vertex_rows, vertex_owned),
        "cell": (cell_ids, cell_rows, cell_owned),
        "coordinates": coordinates,
        "metric": metric,
        "cells": cells,
        "cell_classes": cell_classes,
        "cell_owner_ranks": stacked("cell_owner_ranks"),
        "cell_rank": cell_rank,
        "facets": (facet_ids, facets, facet_classes, facet_owned),
        "fields": field_values,
    }


def _verify_fields(merged: Mapping[str, Any], fields: tuple[OmegaHField, ...], /) -> None:
    for field, values in zip(fields, merged["fields"], strict=True):
        rows = merged["vertex"][1] if _on_vertices(field.transfer) else merged["cell"][1]
        _require(
            np.all(np.isfinite(values)) and np.array_equal(values[rows], values),
            f"field {field.name!r} copies disagree with their owners",
        )


def _decode_record(
    record: Mapping[str, Any],
    options: OmegaHOptions,
    fields: tuple[OmegaHField, ...],
    ranks: int,
    /,
) -> None:
    keys = {
        "cell_count",
        "classification",
        "facet_count",
        "fields",
        "iterations",
        "maximum_length",
        "maximum_quality",
        "minimum_length",
        "minimum_quality",
        "options",
        "ranks",
        "vertex_count",
    }
    _require(isinstance(record, Mapping) and set(record) == keys, "result record fields")
    _require(record["ranks"] == ranks, "result rank count")
    counts = ("cell_count", "facet_count", "iterations", "vertex_count")
    _require(
        all(type(record[name]) is int and record[name] >= 0 for name in counts), "counts"
    )
    reals = ("maximum_length", "maximum_quality", "minimum_length", "minimum_quality")
    _require(
        all(
            isinstance(record[name], (int, float)) and math.isfinite(record[name])
            for name in reals
        ),
        "achieved quality and lengths must be finite",
    )
    classification = record["classification"]
    _require(
        isinstance(classification, Mapping)
        and set(classification) == {"generated_facet_classes"}
        and type(classification["generated_facet_classes"]) is int,
        "classification summary",
    )
    # Requested values must come back unchanged; None quality targets report
    # Omega_h's effective dimension defaults.
    expected = {
        **options.worker_record()["adapt"],
        "feature_angle": options.feature_angle,
        "gradation_rate": options.gradation_rate,
        "maximum_iterations": options.maximum_iterations,
    }
    echoed = record["options"]
    _require(
        isinstance(echoed, Mapping)
        and set(echoed) == set(expected)
        and all(
            echoed[name] == value for name, value in expected.items() if value is not None
        )
        and (options.gradation_rate is not None or echoed["gradation_rate"] is None)
        and all(
            isinstance(echoed[name], (int, float))
            for name in ("min_quality_allowed", "min_quality_desired")
        ),
        "the worker did not apply the requested options verbatim",
    )
    reported = record["fields"]
    _require(
        isinstance(reported, list) and len(reported) == len(fields),
        "field evidence count",
    )
    for index, (field, entry) in enumerate(zip(fields, reported, strict=True)):
        conservative = not _on_vertices(field.transfer)
        components = math.prod(field.values.shape[1:])
        _require(
            isinstance(entry, Mapping)
            and entry.get("name") == f"field-{index}"
            and entry.get("method")
            == ("OMEGA_H_CONSERVE" if conservative else "OMEGA_H_LINEAR_INTERP")
            and set(entry)
            == (
                {"name", "method", "integral_before", "integral_after"}
                if conservative
                else {"name", "method"}
            ),
            f"field {field.name!r} evidence",
        )
        if conservative:
            _require(
                all(
                    isinstance(entry[name], list) and len(entry[name]) == components
                    for name in ("integral_before", "integral_after")
                ),
                f"field {field.name!r} integrals",
            )


def _decode_outputs(
    call: NativeWorkerCall,
    dimension: int,
    fields: tuple[OmegaHField, ...],
    limits: MeshingLimits,
    carrier: _Carrier,
    /,
) -> tuple[tuple[_RankArrays, ...], dict[str, Any]]:
    """Verified rank arrays and their owner-resolved concatenation."""
    record = call.result
    ranks = record["ranks"]
    _require(
        record["vertex_count"] <= limits.maximum_vertices
        and record["cell_count"] <= limits.maximum_cells
        and record["cell_count"] * (dimension + 1) <= limits.maximum_connectivity_entries,
        "adapted mesh exceeds entity limits",
    )
    if ranks == 1:
        _require(not call.parts, "a serial worker returned rank parts")
        outputs = (call.arrays,)
    else:
        names = tuple(f"rank-{rank}" for rank in range(ranks))
        _require(
            not call.arrays and set(call.parts) == set(names),
            "a distributed worker must return exactly one part per rank",
        )
        outputs = tuple(call.parts[name] for name in names)
    rank_arrays = tuple(
        _rank_arrays(arrays, dimension, fields, ranks) for arrays in outputs
    )
    merged = _merged(rank_arrays)
    _verify_fields(merged, fields)
    _require(
        np.all(merged["cell_classes"] < len(carrier.classification.cell_blocks)),
        "cells carry unknown region classes",
    )
    return rank_arrays, merged


def _match_facets(reference: np.ndarray, query: np.ndarray, /) -> np.ndarray:
    """Rows of ``reference`` equal to each ``query`` row (sorted vertex IDs)."""
    stacked = np.concatenate((reference, query))
    _, inverse = np.unique(stacked, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    lookup = np.full(inverse.max(initial=-1) + 1, -1, dtype=np.int64)
    lookup[inverse[: reference.shape[0]]] = np.arange(reference.shape[0])
    return lookup[inverse[reference.shape[0] :]]


def _target_organization(
    source: CellMeshingResult,
    mesh: CellMesh,
    classification: OmegaHClassification,
    cell_classes: np.ndarray,
    facet_ids: np.ndarray,
    facet_classes: np.ndarray,
    /,
) -> tuple[tuple[MeshPatch, ...], tuple[MeshZone, ...], tuple[MeshLabel, ...]]:
    """Rebuild named organization on the target from preserved class IDs."""
    dimension = mesh.topological_dimension
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )

    def scope(dim: int, table: tuple, selects) -> MeshingScope:
        ids, classes = (
            (cell_ids, cell_classes) if dim == dimension else (facet_ids, facet_classes)
        )
        selected = np.asarray([selects(entry) for entry in table], dtype=np.bool_)
        chosen = np.zeros(classes.shape, dtype=np.bool_)
        named = classes < selected.size
        chosen[named] = selected[classes[named]]
        if not np.any(chosen):
            raise _conversion("a named region or facet group vanished during adaptation")
        return _entity_scope(mesh, dim, ids[chosen])

    zones, zone_ids = [], {}
    for zone in source.zones:
        dim = zone.scope.entity_dimension
        table = (
            classification.cell_zones if dim == dimension else classification.facet_zones
        )
        rebuilt = MeshZone(
            zone.name,
            zone.role,
            scope(dim, table, lambda entry, name=zone.name: entry == name),
            material_id=zone.material_id,
            region_role=zone.region_role,
        )
        zones.append(rebuilt)
        zone_ids[zone.zone_id] = rebuilt.zone_id
    patches = tuple(
        MeshPatch(
            patch.name,
            scope(
                dimension - 1,
                classification.facet_patches,
                lambda names, name=patch.name: name in names,
            ),
            connected=patch.connected,
            adjacent_zone_ids=tuple(zone_ids[value] for value in patch.adjacent_zone_ids),
        )
        for patch in source.patches
    )
    labels = tuple(
        MeshLabel(
            label.name,
            scope(
                label.scope.entity_dimension,
                classification.cell_labels
                if label.scope.entity_dimension == dimension
                else classification.facet_labels,
                lambda names, name=label.name: name in names,
            ),
        )
        for label in source.labels
    )
    return patches, tuple(zones), labels


class _Assembled(NamedTuple):
    mesh: CellMesh
    patches: tuple[MeshPatch, ...]
    zones: tuple[MeshZone, ...]
    labels: tuple[MeshLabel, ...]
    vertex_rows: np.ndarray
    cell_rows: np.ndarray


def _assemble(
    source: CellMeshingResult,
    merged: Mapping[str, Any],
    classification: OmegaHClassification,
    counts: Mapping[str, int],
    /,
) -> _Assembled:
    """Global carrier and organization from owner copies, in global-ID order."""
    dimension = source.mesh.topological_dimension
    vertex_ids, _, vertex_owned = merged["vertex"]
    cell_ids, _, cell_owned = merged["cell"]
    facet_ids, facets, facet_classes, facet_owned = merged["facets"]
    vertex_rows = np.flatnonzero(vertex_owned)[np.argsort(vertex_ids[vertex_owned])]
    cell_rows = np.flatnonzero(cell_owned)[np.argsort(cell_ids[cell_owned])]
    facet_rows = np.flatnonzero(facet_owned)[np.argsort(facet_ids[facet_owned])]
    _require(
        (vertex_rows.size, cell_rows.size, facet_rows.size)
        == (counts["vertex_count"], counts["cell_count"], counts["facet_count"]),
        "owner copies do not cover the reported global counts",
    )
    target_vertex_ids = vertex_ids[vertex_rows]
    connectivity = np.searchsorted(target_vertex_ids, merged["cells"][cell_rows])
    classes = merged["cell_classes"][cell_rows]
    _require(
        np.all(classes < len(classification.cell_blocks))
        and np.unique(classes).size == len(classification.cell_blocks),
        "cell classes must cover exactly the source regions",
    )
    names = [block.name for block in source.mesh.blocks]
    region_blocks = np.asarray(
        [names.index(name) for name in classification.cell_blocks], dtype=np.int64
    )[classes]
    blocks = tuple(
        CellBlock(
            name,
            _CELL_KINDS[dimension],
            connectivity[region_blocks == index].astype(np.int32),
            global_ids=cell_ids[cell_rows][region_blocks == index],
        )
        for index, name in enumerate(names)
    )
    mesh = canonicalize_cell_mesh(
        CellMesh(
            merged["coordinates"][vertex_rows],
            blocks,
            vertex_global_ids=target_vertex_ids,
            numeric_version=_TARGET_REVISION,
        )
    )
    mesh_cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    reference = np.sort(
        np.asarray(mesh.vertex_global_ids, dtype=np.int64)[_facet_rows(mesh)], axis=1
    )
    matched = _match_facets(reference, facets[facet_rows])
    _require(np.all(matched >= 0), "classified facets are not facets of the target")
    target_facet_ids = np.asarray(
        mesh.entity_set(dimension - 1).entity_ids, dtype=np.int64
    )
    patches, zones, labels = _target_organization(
        source,
        mesh,
        classification,
        classes[np.searchsorted(cell_ids[cell_rows], mesh_cell_ids)],
        target_facet_ids[matched],
        facet_classes[facet_rows],
    )
    return _Assembled(mesh, patches, zones, labels, vertex_rows, cell_rows)


def _gather(
    source: CellMeshingResult,
    metric: MeshMetricField,
    fields: tuple[OmegaHField, ...],
    merged: Mapping[str, Any],
    classification: OmegaHClassification,
    counts: Mapping[str, int],
    provider: MeshingProviderInfo,
    runtime: MeshingRuntimeInfo,
    provenance: SemanticProvenance,
    transfer: BRepAssociationTransfer | None,
    /,
) -> _GatheredTarget:
    """Certify the gathered global carrier and bind metric and fields to it."""
    dimension = source.mesh.topological_dimension
    mesh, patches, zones, labels, vertex_rows, cell_rows = _assemble(
        source, merged, classification, counts
    )
    vertex_ids = merged["vertex"][0][vertex_rows]
    cell_ids = merged["cell"][0][cell_rows]
    certified = certify_cell_mesh(
        mesh, source.coordinate_contract, patches=patches, zones=zones, labels=labels
    )
    associations = ()
    if transfer is not None and source.associations:
        # Unknown lineage: re-derive B-Rep associations by classification transfer
        # through the output facet class IDs, then projection.
        associations = transfer.rederive(source, certified)
        certified = certify_cell_mesh(
            mesh,
            source.coordinate_contract,
            patches=patches,
            zones=zones,
            labels=labels,
            associations=associations,
        )
    result = CellMeshingResult(
        certified.mesh,
        certified.geometry,
        source.coordinate_contract,
        certified.audit,
        certified.quality,
        certified.compliance,
        MeshingTrace(
            (
                MeshingStageReport(
                    MeshingStageKind.OPTIMIZATION,
                    MeshingStageStatus.PASSED,
                    input_ids=(source.mesh.mesh_id, metric.metric_id),
                    output_ids=(certified.mesh.mesh_id,),
                ),
                *certified.trace.stages,
            )
        ),
        provider,
        runtime,
        MeshingDerivativeMode.NONDIFFERENTIABLE,
        provenance,
        patches=patches,
        zones=zones,
        labels=labels,
        associations=associations,
    )
    target_mesh = result.mesh
    vertex_scope = _entity_scope(target_mesh, 0, vertex_ids)
    cell_scope = _entity_scope(target_mesh, dimension, cell_ids)
    target_metric = MeshMetricField(
        vertex_scope,
        merged["metric"][vertex_rows],
        minimum_size=metric.minimum_size,
        maximum_size=metric.maximum_size,
        maximum_anisotropy=metric.maximum_anisotropy,
        maximum_gradation=metric.maximum_gradation,
    )
    transferred = tuple(
        OmegaHTransferredField(
            field.name, field.transfer, vertex_scope, values[vertex_rows]
        )
        if _on_vertices(field.transfer)
        else OmegaHTransferredField(
            field.name, field.transfer, cell_scope, values[cell_rows]
        )
        for field, values in zip(fields, merged["fields"], strict=True)
    )
    return _GatheredTarget(
        result,
        target_metric,
        transferred,
        merged["cell_owner_ranks"][cell_rows].astype(np.int32),
        cell_ids,
    )


def _distribution(
    gathered: _GatheredTarget, merged: Mapping[str, Any], ranks: int, /
) -> MeshDistribution:
    cell_ids, _, cell_owned = merged["cell"]
    ghost_rank = merged["cell_rank"][~cell_owned]
    ghost_ids = cell_ids[~cell_owned]
    order = np.lexsort((ghost_ids, ghost_rank))
    bounds = np.searchsorted(ghost_rank[order], np.arange(ranks + 1))
    halos = tuple(
        ghost_ids[order][bounds[rank] : bounds[rank + 1]] for rank in range(ranks)
    )
    return MeshDistribution(
        MeshPart("omega_h", gathered.result),
        CellPartition(gathered.cell_owner_ranks, ranks),
        cell_global_ids=gathered.cell_global_ids,
        halo_global_ids=halos,
        halo_width=1,
        partition_kind=MeshPartitionKind.PROVIDER,
        partition_provenance="omega_h",
    )


class OmegaHProvider:
    """Adapt affine planar triangles or volume tetrahedra with real Omega_h.

    Build ``native/providers/omega_h`` against an installed Omega_h CMake
    package and set PHYDRAX_OMEGA_H_WORKER (or pass the executable). The worker
    stays alive between calls: one session per rank count, identity probed
    once. Distributed runs launch ``(*mpi_launcher, "-n", ranks)``; launcher
    arguments and environment are deployment configuration passed explicitly.
    Differentiation, CAD projection, and lineage are deliberately not claimed.
    """

    def __init__(
        self,
        executable: str | os.PathLike[str] | None = None,
        /,
        *,
        mpi_launcher: Sequence[str] = ("mpiexec",),
        environment: Mapping[str, str] | None = None,
        policy: NativeWorkerPolicy | None = None,
    ):
        if isinstance(mpi_launcher, str):
            raise TypeError("mpi_launcher must be a sequence of argv tokens.")
        self.executable = None if executable is None else os.fspath(executable)
        self.mpi_launcher = tuple(str(value) for value in mpi_launcher)
        self.environment = dict(environment or {})
        self.policy = policy
        self._workers: dict[int, ProviderWorker] = {}

    def worker(self, ranks: int = 1, /) -> ProviderWorker:
        """The persistent worker session of one rank count (launched lazily)."""
        count = _integer(ranks, "ranks", 1, 1_000_000)
        worker = self._workers.get(count)
        if worker is None:
            if count > 1 and not self.mpi_launcher:
                raise _failure(
                    MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                    "Distributed Omega_h adaptation requires an MPI launcher.",
                )
            worker = ProviderWorker(
                "omega_h",
                executable=self.executable,
                environment_variable="PHYDRAX_OMEGA_H_WORKER",
                default_executable="phydrax-omega-h-worker",
                build_hint=(
                    "Build native/providers/omega_h against an Omega_h CMake package "
                    "and set PHYDRAX_OMEGA_H_WORKER to phydrax-omega-h-worker."
                ),
                launcher=() if count == 1 else (*self.mpi_launcher, "-n", str(count)),
                policy=self.policy,
                environment=self.environment,
            )
            self._workers[count] = worker
        return worker

    def _identity(self, ranks: int, /) -> tuple[ProviderWorker, Mapping[str, Any], str]:
        worker = self.worker(ranks)
        identity = worker.identity
        reported = identity.reported
        if (
            reported.get("library") != "Omega_h"
            or not isinstance(reported.get("version"), str)
            or not isinstance(reported.get("commit"), str)
            or "adapt" not in reported.get("operations", ())
        ):
            worker.close()
            raise _failure(
                MeshingFailureCategory.PROVIDER_UNAVAILABLE,
                "The executable is not a phydrax Omega_h worker.",
            )
        if identity.ranks != ranks or (ranks > 1 and reported.get("mpi") is None):
            worker.close()
            raise _failure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                f"The Omega_h worker did not start as one {ranks}-rank MPI world.",
            )
        return worker, reported, identity.identity_id

    def info(self) -> MeshingProviderInfo:
        return self._provider(self._identity(1)[1])

    @staticmethod
    def _provider(reported: Mapping[str, Any], /) -> MeshingProviderInfo:
        capabilities = (
            MeshingCapability.ANISOTROPIC_METRIC,
            MeshingCapability.DETERMINISTIC,
            MeshingCapability.MULTI_MATERIAL,
        )
        if reported.get("mpi") is not None:
            capabilities += (MeshingCapability.PARALLEL, MeshingCapability.DISTRIBUTED)
        return MeshingProviderInfo(
            "omega_h",
            reported["version"],
            "BSD-2-Clause",
            operations=(MeshingOperation.REMESH_SURFACE, MeshingOperation.ADAPT_VOLUME),
            source_kinds=(MeshingSourceKind.CELL_MESH,),
            capabilities=capabilities,
            cell_kinds=("triangle", "tetrahedron"),
            dimensions=(2, 3),
            execution_modes=(MeshingExecutionMode.SUBPROCESS,),
        )

    def execute(
        self,
        source: CellMeshingResult,
        metric: MeshMetricField,
        /,
        *,
        options: OmegaHOptions | None = None,
        fields: Sequence[OmegaHField] = (),
        ranks: int = 1,
        gather: bool = False,
        limits: MeshingLimits | None = None,
        association_transfer: BRepAssociationTransfer | None = None,
    ) -> OmegaHAdaptationResult:
        """Adapt ``source`` to ``metric`` and transfer ``fields``.

        Everything is validated and resource bounds are refused before the
        worker launches. ``gather`` assembles the global carrier of a
        distributed run; a serial run always returns it. ``association_transfer``
        re-derives the source's B-Rep associations on the gathered target.
        """
        if not isinstance(source, CellMeshingResult) or not isinstance(
            metric, MeshMetricField
        ):
            raise TypeError("Omega_h requires a CellMeshingResult and MeshMetricField.")
        options_ = OmegaHOptions() if options is None else options
        limits_ = MeshingLimits() if limits is None else limits
        if not isinstance(options_, OmegaHOptions):
            raise TypeError("options must be OmegaHOptions or None.")
        if not isinstance(limits_, MeshingLimits):
            raise TypeError("limits must be MeshingLimits or None.")
        fields_ = tuple(fields)
        if not all(isinstance(field, OmegaHField) for field in fields_):
            raise TypeError("fields must contain OmegaHField values.")
        if len({field.name for field in fields_}) != len(fields_):
            raise ValueError("Omega_h field names must be unique.")
        count = _integer(ranks, "ranks", 1, 1_000_000)
        _flag(gather, "gather")
        if association_transfer is not None and not isinstance(
            association_transfer, BRepAssociationTransfer
        ):
            raise TypeError(
                "association_transfer must be BRepAssociationTransfer or None."
            )
        if (
            max(
                limits_.maximum_vertices,
                limits_.maximum_cells,
                limits_.maximum_connectivity_entries,
            )
            > _LOCAL_INDEX_LIMIT
        ):
            raise ValueError("limits exceed Omega_h local indexing.")
        mesh, dimension = _source_mesh(source, association_transfer)
        _refuse_resources(mesh, limits_)
        carrier = _carrier(source, metric, fields_, options_)
        _refuse_bytes(carrier.arrays, limits_)

        worker, reported, identity_id = self._identity(count)
        parameters = {
            "dimension": dimension,
            **options_.worker_record(),
            "fields": [
                {
                    "name": f"field-{index}",
                    "transfer": field.transfer.value,
                    "diffusion_tolerance": field.diffusion_tolerance,
                }
                for index, field in enumerate(fields_)
            ],
            "maximum_vertices": limits_.maximum_vertices,
            "maximum_cells": limits_.maximum_cells,
            "maximum_connectivity_entries": limits_.maximum_connectivity_entries,
        }
        call = worker.call("adapt", parameters, carrier.arrays, limits=limits_)
        return self._result(
            source,
            metric,
            fields_,
            options_,
            limits_,
            count,
            gather,
            carrier,
            call,
            worker,
            reported,
            identity_id,
            association_transfer,
        )

    def _result(
        self,
        source: CellMeshingResult,
        metric: MeshMetricField,
        fields: tuple[OmegaHField, ...],
        options: OmegaHOptions,
        limits: MeshingLimits,
        ranks: int,
        gather: bool,
        carrier: _Carrier,
        call: NativeWorkerCall,
        worker: ProviderWorker,
        reported: Mapping[str, Any],
        identity_id: str,
        association_transfer: BRepAssociationTransfer | None,
        /,
    ) -> OmegaHAdaptationResult:
        record = call.result
        _decode_record(record, options, fields, ranks)
        rank_arrays, merged = _decode_outputs(
            call, source.mesh.topological_dimension, fields, limits, carrier
        )
        field_evidence = tuple(
            OmegaHFieldEvidence(
                field,
                entry["method"],
                entry.get("integral_before"),
                entry.get("integral_after"),
            )
            for field, entry in zip(fields, record["fields"], strict=True)
        )
        evidence = OmegaHAdaptationEvidence(
            options, field_evidence, record, call, identity_id
        )
        provider = self._provider(reported)
        runtime = MeshingRuntimeInfo(
            provider.provider_id,
            provider.version,
            MeshingExecutionMode.SUBPROCESS,
            deterministic=True,
            enforced_limits=(
                "wall_seconds",
                "input_entities",
                "input_connectivity_entries",
                "input_bytes",
                "output_entities",
                "output_bytes",
                "aggregate_output_bytes",
                worker.memory_limit_evidence(),
            ),
            unenforced_limits=("native_adaptation_intermediate_entities",),
        )
        provenance = SemanticProvenance(
            {
                "kind": "omega-h-metric-adaptation",
                "source": source.result_id,
                "source_mesh": source.mesh.mesh_id,
                "input_metric": metric.metric_id,
                "options": options.options_id,
                "fields": [field.field_id for field in fields],
                "classification": carrier.classification.classification_id,
                "omega_h_version": reported["version"],
                "omega_h_commit": reported["commit"],
                "worker_identity": identity_id,
                "worker_session": evidence.session_id,
                "worker_sequence": evidence.sequence,
                "ranks": ranks,
                "evidence": evidence.evidence_id,
                "lineage": "unknown",
                "global_ids": "authoritative-output-revision-only",
            }
        )
        target = (
            _gather(
                source,
                metric,
                fields,
                merged,
                carrier.classification,
                record,
                provider,
                runtime,
                provenance,
                association_transfer,
            )
            if ranks == 1 or gather
            else None
        )
        distribution = (
            None if target is None or ranks == 1 else _distribution(target, merged, ranks)
        )
        return OmegaHAdaptationResult(
            source,
            metric.metric_id,
            tuple(
                OmegaHPartition(rank, arrays) for rank, arrays in enumerate(rank_arrays)
            ),
            carrier.classification,
            evidence,
            target,
            distribution,
            provider,
            runtime,
            provenance,
        )

    def close(self) -> None:
        """End every worker session of this provider."""
        for worker in self._workers.values():
            worker.close()

    def __enter__(self) -> OmegaHProvider:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "OmegaHAdaptationEvidence",
    "OmegaHAdaptationResult",
    "OmegaHClassification",
    "OmegaHField",
    "OmegaHFieldEvidence",
    "OmegaHFieldTransfer",
    "OmegaHOptions",
    "OmegaHPartition",
    "OmegaHProvider",
    "OmegaHTransferredField",
]
