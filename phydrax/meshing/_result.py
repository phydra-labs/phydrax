#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import (
    canonical_fingerprint,
    logical_array_value_collection_digest,
)
from .._identity import SemanticProvenance
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization._adaptive_simplex import AdaptiveSimplexLayout, AdaptiveSimplexState
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..geometry._mesh_certificates import (
    DomainCoverageCertificate,
    GlobalEmbeddingCertificate,
)
from ..geometry._surface_source_support import SurfaceSourceCharts
from ..geometry.surface import SurfaceModel
from ..interchange import AdapterReport
from ..typing import checked
from ._association import GeometryAssociation
from ._audit import (
    _evidence_id,
    _mesh_evidence_issues,
    CellMeshAuditReport,
    CellMeshAuditScope,
)
from ._certification import MeshCertificationReport
from ._contracts import (
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingProviderInfo,
)
from ._measurements import NativeExecutionRecord
from ._organization import (
    MeshAttribute,
    MeshAttributeProjection,
    MeshLabel,
    MeshPatch,
    MeshZone,
    RegionBoundaryEvidence,
    RegionMeshingEvidence,
    validate_local_mesh_organization,
    validate_mesh_labels,
    validate_mesh_zones,
)
from ._quality import CellQualityReport
from ._scope import MeshScopeProjection
from ._trace import MeshingTrace


if TYPE_CHECKING:
    from ._adaptation import PreparedMeshAdaptation
    from ._bisection import BisectionUniformRefinement
    from ._device_adaptation import PartitionedAdaptiveSimplex
    from ._initial_certification import InitialCollectiveMeshEvidence


_BISECTION_CERTIFICATE_CHECKS = (
    "source_roots",
    "subdivision",
    "midpoint_geometry",
    "owned_cell_identity",
    "reciprocal_facets",
    "shared_geometry",
    "collective_status",
)


class AbstractCollectiveMeshTheorem(StrictModule):
    """Accepted all-owner scientific theorem consumed by owner-local publication.

    Concrete theorems are the adaptive-subdivision ``CollectiveMeshEvidence`` and
    the independent initial source theorem; publication dispatches on the
    concrete kind and refuses any other implementation.
    """

    logical_arrays: eqx.AbstractVar[tuple[tuple[str, Array], ...]]
    entity_keys: eqx.AbstractVar[tuple[Array, ...]]
    entity_ids: eqx.AbstractVar[tuple[Array, ...]]
    entity_owners: eqx.AbstractVar[tuple[Array, ...]]
    topology_id: eqx.AbstractVar[str]
    geometry_id: eqx.AbstractVar[str]
    global_entity_counts: eqx.AbstractVar[tuple[int, ...]]
    coordinate_geometry_id: eqx.AbstractVar[str]
    source_evidence_id: eqx.AbstractVar[str]
    global_organization_id: eqx.AbstractVar[str]
    mesh_id: eqx.AbstractVar[str]
    partition_count: eqx.AbstractVar[int]
    evidence_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def require_passed(self) -> None:
        """Refuse unless every global verdict of the theorem passed."""


class CollectiveMeshEvidence(AbstractCollectiveMeshTheorem, NonTrainableState):
    """Consumed partition-indexed affine-subdivision witnesses and verdicts.

    Findings and canonical array witnesses stay sharded. ``compiled_states``
    retains the untouched binary epoch; logical epoch arrays describe the
    independently admitted packed inverse publication. Publication consumes
    bounded global summaries and streamed canonical hashing chunks, never a
    caller-supplied positive flag or declared target digest.

    ``raw_source_blocks`` owns the initial/compiled block declaration namespace.
    ``publication_source_blocks`` owns the independently published epoch's
    namespace, including original coarse roots restored by the host inverse.
    Geometry regrouping changes presentation blocks, not these source axes;
    cold replay retains both scientific declaration identities explicitly.
    """

    partition_checks: Array
    global_checks: Array
    source: CellMeshingResult
    preparation: PreparedMeshAdaptation
    initial_states: AdaptiveSimplexState
    compiled_states: AdaptiveSimplexState
    raw_source_blocks: tuple[CellBlock, ...]
    publication_source_blocks: tuple[CellBlock, ...]
    uniform_refinement: BisectionUniformRefinement | None
    source_exterior: Array
    layout: AdaptiveSimplexLayout
    neighbor_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    logical_arrays: tuple[tuple[str, Array], ...]
    entity_keys: tuple[Array, ...]
    entity_ids: tuple[Array, ...]
    entity_owners: tuple[Array, ...]
    global_entity_counts: tuple[int, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    coordinate_geometry_id: str = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_audit_id: str = eqx.field(static=True)
    source_evidence_id: str = eqx.field(static=True)
    global_organization_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    partition_count: int = eqx.field(static=True)
    checks: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        partitioned: PartitionedAdaptiveSimplex | CollectiveMeshEvidence,
        states: AdaptiveSimplexState,
        /,
    ) -> None:
        from ._bisection import _uniform_execution, _UniformAllowance
        from ._device_adaptation import PartitionedAdaptiveSimplex

        if isinstance(partitioned, CollectiveMeshEvidence):
            policy = partitioned.preparation.policy
        elif isinstance(partitioned, PartitionedAdaptiveSimplex):
            policy = partitioned.prepared.adaptation.policy
        else:
            raise TypeError(
                "Collective evidence requires its actual prepared source owner."
            )
        limits = policy.limits
        allowance = _UniformAllowance(
            limits.maximum_work_units,
            limits.maximum_geometry_queries,
            limits.maximum_cells,
            limits.maximum_vertices,
            limits.maximum_scratch_bytes,
            limits.maximum_wall_seconds,
            limits.maximum_cavity_cells,
        )
        with _uniform_execution(allowance):
            self._initialize(partitioned, states)

    def _initialize(
        self,
        partitioned: PartitionedAdaptiveSimplex | CollectiveMeshEvidence,
        states: AdaptiveSimplexState,
        /,
    ) -> None:
        from ._device_adaptation import (
            _logical_publication,
            _packed_publication_states,
            _require_partitioned_exact_restrictions,
            _source_cache_digest,
            _source_entity_tables,
            _source_solver_cell_owners,
            PartitionedAdaptiveSimplex,
        )
        from ._distribution import (
            _compiled_bisection_checks,
            AffineBisectionSourceWitness,
            replay_logical_bisection_checks,
        )
        from ._distribution_forest import (
            inherit_solver_cell_owners,
            replay_logical_solver_cell_owners,
        )
        from ._initial_certification import InitialCollectiveMeshEvidence

        if not isinstance(states, AdaptiveSimplexState):
            raise TypeError(
                "Collective evidence consumes actual adaptive numerical states."
            )
        cold = isinstance(partitioned, CollectiveMeshEvidence)
        if cold:
            partitioned.require_passed()
            preparation = partitioned.preparation
            initial = partitioned.initial_states
            raw_source_blocks = partitioned.raw_source_blocks
            saved_publication_blocks = partitioned.publication_source_blocks
            if (
                not isinstance(saved_publication_blocks, tuple)
                or not saved_publication_blocks
            ):
                raise ValueError(
                    "Saved publication lacks its actual source block declaration namespace."
                )
            if any(
                not isinstance(block, CellBlock) for block in saved_publication_blocks
            ):
                raise ValueError(
                    "Saved publication source declarations must be actual scientific cell blocks."
                )
            uniform_refinement = partitioned.uniform_refinement
            exterior = partitioned.source_exterior
            layout = partitioned.layout
            neighbor_pairs = partitioned.neighbor_pairs
            axis_name = partitioned.axis_name
            placement_arrays = tuple(
                (name, value)
                for name, value in partitioned.logical_arrays
                if name.startswith("placement/")
            )
            source = preparation.source
            if source.result_id != partitioned.source.result_id:
                raise ValueError(
                    "Saved preparation lost its actual accepted predecessor."
                )
            witness = AffineBisectionSourceWitness(
                source, uniform_refinement=uniform_refinement
            )
            source_entities = _source_entity_tables(preparation)
            initial_owners = dict(placement_arrays).get(
                "placement/initial_solver_cell_owners"
            )
            if initial_owners is None:
                initial_owners = _source_solver_cell_owners(preparation, initial)
        elif isinstance(partitioned, PartitionedAdaptiveSimplex):
            preparation = partitioned.prepared.adaptation
            initial = partitioned.states
            uniform_refinement = partitioned.prepared.anchor.start.uniform_refinement
            declared_blocks = partitioned.prepared.anchor.source_blocks
            raw_source_blocks = (
                partitioned.prepared.anchor.source.mesh.blocks
                if declared_blocks is None
                else declared_blocks
            )
            exterior = partitioned.source_exterior
            layout = partitioned.layout
            neighbor_pairs = partitioned.parts.neighbor_pairs
            axis_name = partitioned.parts.axis_name
            placement_arrays = partitioned.placement_arrays
            source = preparation.source
            witness = partitioned.source_witness
            source_entities = partitioned.source_entities
            initial_owners = partitioned.solver_cell_owners
        else:
            raise TypeError(
                "Collective evidence requires actual preparation or its retained typed receipt."
            )
        if not isinstance(raw_source_blocks, tuple) or not raw_source_blocks:
            raise ValueError(
                "Collective raw states require their actual retained source block declarations."
            )
        kind = "triangle" if layout.dimension == 2 else "tetrahedron"
        raw_blocks = tuple(
            block for block in raw_source_blocks if isinstance(block, CellBlock)
        )
        if len(raw_blocks) != len(raw_source_blocks) or any(
            block.cell_kind != kind for block in raw_blocks
        ):
            raise ValueError(
                "Collective raw source declarations differ from the actual simplex family."
            )
        from ._bisection import _uniform_charge

        lanes = initial.mesh.cell_ids.size + states.mesh.cell_ids.size
        _uniform_charge(8 * lanes + len(raw_blocks), 8 * lanes)
        for raw in (initial, states):
            if raw.blocks.shape != raw.mesh.cell_ids.shape or not jnp.issubdtype(
                raw.blocks.dtype, jnp.integer
            ):
                raise ValueError(
                    "Raw source block indices require the exact allocated cell axis."
                )
            used = (raw.mesh.cell_ids >= 0) & (raw.mesh.cell_active | ~raw.retired)
            invalid = used & ((raw.blocks < 0) | (raw.blocks >= len(raw_blocks)))
            if bool(jax.device_get(jnp.any(invalid))):
                raise ValueError(
                    "An actual nonretired raw cell lacks its declared scientific source block."
                )
        original = require_original_meshing_source(
            source if uniform_refinement is None else uniform_refinement.source,
        )
        if isinstance(original, InitialCollectiveMeshEvidence):
            original.require_passed()
            scientific_id = original.evidence_id
            premise_id = original.evidence_id
            source_audit_id = original.evidence_id
            scientific_binding = {
                "kind": "independent-native-initial-theorem",
                "theorem": original.evidence_id,
                "source": original.source_evidence_id,
                "compiled": original.compiled.compiled_id,
                "specification": original.specification.specification_id,
            }
        else:
            certification = original.certification
            if certification is None or certification.embedding is None:
                raise ValueError(
                    "Collective evidence requires its original certified affine source premise."
                )
            scientific_id = original.result_id
            premise_id = certification.report_id
            source_audit_id = original.audit.report_id
            scientific_binding = {
                "kind": "original-mesh-certification",
                "certification": certification.report_id,
                "request": certification.request.request_id,
                "source_regions": None
                if original.region_evidence is None
                else original.region_evidence.evidence_id,
                "region_boundaries": [
                    value.evidence_id for value in original.region_boundary_evidence
                ],
                "associations": [value.association_id for value in original.associations],
            }
        if witness is None:
            raise ValueError(
                "Collective evidence requires the actual affine source-map witness."
            )
        if (
            witness.source_result_id != source.result_id
            or witness.source_geometry_id != source.mesh.geometry_id
            or witness.source_layout_id != source.geometry.geometry_layout_id
            or witness.source_coordinate_geometry_id != cell_geometry_id(source.geometry)
            or witness.scientific_source_result_id != scientific_id
            or witness.scientific_certification_id != premise_id
            or witness.predecessor_evidence_id
            != (
                None
                if source.collective_evidence is None
                else source.collective_evidence.evidence_id
            )
            or states.cursors.shape != initial.cursors.shape
            or states.cursors.shape != (initial.mesh.cell_ids.shape[0], 4)
            or exterior.shape != initial.mesh.cells.shape
            or exterior.dtype != jnp.bool_
        ):
            raise ValueError(
                "Prepared source/placement witnesses do not bind these states."
            )
        source_cache_id = _source_cache_digest(source_entities)
        initial_state_id = logical_array_value_collection_digest(
            {
                jax.tree_util.keystr(path): value
                for path, value in jax.tree_util.tree_flatten_with_path(initial)[0]
                if isinstance(value, Array)
            }
        )
        if (
            not cold
            and isinstance(partitioned, PartitionedAdaptiveSimplex)
            and (
                source_cache_id != partitioned.source_cache_id
                or initial_state_id != partitioned.initial_state_id
            )
        ):
            raise ValueError(
                "Prepared source caches or initial numerical states lost their owning binding."
            )
        midpoint_valid, midpoint_signs = _require_partitioned_exact_restrictions(
            layout, states, initial
        )
        native_arrays = (
            ("native/midpoint_valid", midpoint_valid),
            ("native/midpoint_signs", midpoint_signs),
        )
        native_receipt_id = logical_array_value_collection_digest(dict(native_arrays))
        if cold:
            partition_checks = replay_logical_bisection_checks(
                states,
                initial,
                exterior,
                neighbor_pairs=neighbor_pairs,
                axis_name=axis_name,
            )
        elif isinstance(partitioned, PartitionedAdaptiveSimplex):
            partition_checks = _compiled_bisection_checks(
                partitioned.parts,
                states,
                initial,
                exterior,
            )
        else:
            raise TypeError("Collective preparation type changed during numerical proof.")
        verdicts = jnp.all(partition_checks, axis=0)
        summary = np.asarray(jax.device_get(verdicts), dtype=np.bool_)
        if not np.all(summary):
            raise ValueError("Collective raw subdivision witnesses did not pass.")
        if cold:
            owners, owner_status = replay_logical_solver_cell_owners(
                states,
                initial,
                initial_owners,
                neighbor_pairs=neighbor_pairs,
                axis_name=axis_name,
            )
        elif isinstance(partitioned, PartitionedAdaptiveSimplex):
            owners, owner_status = inherit_solver_cell_owners(
                partitioned.parts,
                states,
                initial,
                initial_owners,
            )
        else:
            raise TypeError("Collective preparation type changed during owner proof.")
        if bool(jax.device_get(jnp.any(owner_status != 0))):
            raise ValueError(
                "Collective solver ownership lost its accepted predecessor ancestry."
            )
        compiled_states = states
        states, publication_exterior, publication_source_blocks = (
            _packed_publication_states(
                preparation,
                layout,
                initial,
                compiled_states,
                exterior,
                uniform_refinement,
                raw_source_blocks=raw_blocks,
                initial_solver_cell_owners=initial_owners,
            )
        )
        if (
            not isinstance(publication_source_blocks, tuple)
            or not publication_source_blocks
        ):
            raise ValueError(
                "Published raw epoch lacks its actual scientific source block declarations."
            )
        if any(
            not isinstance(block, CellBlock) or block.cell_kind != kind
            for block in publication_source_blocks
        ):
            raise ValueError(
                "Published source declarations differ from the actual simplex family."
            )
        if cold:
            if tuple(block.block_id for block in publication_source_blocks) != tuple(
                block.block_id for block in saved_publication_blocks
            ):
                raise ValueError(
                    "Replayed publication differs from its actual retained source block namespace."
                )
            publication_source_blocks = saved_publication_blocks
        _uniform_charge(
            8 * states.mesh.cell_ids.size + len(publication_source_blocks),
            8 * states.mesh.cell_ids.size,
        )
        used = (states.mesh.cell_ids >= 0) & (states.mesh.cell_active | ~states.retired)
        if states.blocks.shape != states.mesh.cell_ids.shape or not jnp.issubdtype(
            states.blocks.dtype, jnp.integer
        ):
            raise ValueError(
                "Published source block indices require the exact allocated cell axis."
            )
        invalid = used & (
            (states.blocks < 0) | (states.blocks >= len(publication_source_blocks))
        )
        if bool(jax.device_get(jnp.any(invalid))):
            raise ValueError(
                "An actual published nonretired cell lacks its declared scientific source block."
            )
        if uniform_refinement is not None:
            sentinel = jnp.iinfo(jnp.int64).max
            raw_keys = jnp.where(
                compiled_states.mesh.cell_ids >= 0,
                compiled_states.mesh.cell_ids,
                sentinel,
            )
            positions = jax.vmap(jnp.searchsorted)(raw_keys, states.mesh.cell_ids)
            safe = jnp.minimum(positions, raw_keys.shape[1] - 1)
            inherited = jnp.take_along_axis(owners, safe, axis=1)
            present = jnp.take_along_axis(raw_keys, safe, axis=1) == states.mesh.cell_ids
            # A restored uniform root inherits the same canonical minimum of
            # actual source-leaf solver owners as a binary coarsening. Physical
            # placement rank is not scientific solver ownership.
            roots = jnp.asarray(uniform_refinement.parent_ids)
            siblings = jnp.asarray(uniform_refinement.child_ids)
            root_positions = jnp.searchsorted(roots, states.mesh.cell_ids)
            root_safe = jnp.minimum(root_positions, roots.shape[0] - 1)
            root_present = roots[root_safe] == states.mesh.cell_ids
            child_ids = jnp.broadcast_to(siblings, (raw_keys.shape[0], *siblings.shape))
            child_positions = jax.vmap(jnp.searchsorted)(
                raw_keys, child_ids.reshape((raw_keys.shape[0], -1))
            )
            child_positions = child_positions.reshape(child_ids.shape)
            child_safe = jnp.minimum(child_positions, raw_keys.shape[1] - 1)
            child_keys = jax.vmap(lambda keys, positions: keys[positions])(
                raw_keys, child_safe
            )
            child_owners = jax.vmap(lambda values, positions: values[positions])(
                owners, child_safe
            )
            root_complete = jnp.all(
                (child_keys == child_ids) & (child_owners >= 0), axis=-1
            )
            complete = jnp.take_along_axis(root_complete, root_safe, axis=1)
            newly_restored = states.mesh.cell_active & ~present
            if bool(jax.device_get(jnp.any(newly_restored & ~(root_present & complete)))):
                raise ValueError(
                    "A restored uniform root lost its complete actual solver-owner source support."
                )
            restored_owners = jnp.take_along_axis(
                jnp.min(child_owners, axis=-1), root_safe, axis=1
            )
            owners = jnp.where(
                states.mesh.cell_active,
                jnp.where(present, inherited, restored_owners),
                -1,
            )
        logical = _logical_publication(
            source,
            states,
            owners,
            publication_exterior,
            layout,
            placement_arrays=placement_arrays,
            original_source=None
            if uniform_refinement is None
            else uniform_refinement.source,
            uniform_refinement=uniform_refinement,
            # The serial commit owner versions an adapted target by its preparation.
            numeric_version=f"adaptation:{preparation.prepared_id}",
        )
        source_mesh_id = source.mesh.mesh_id
        logical_values = dict(logical.arrays)
        capacity = logical_values["cell_global_ids"].shape[0]
        active = states.mesh.cell_active.reshape(-1)
        order = jnp.argsort(
            jnp.where(active, states.mesh.cell_ids.reshape(-1), jnp.iinfo(jnp.int64).max),
            stable=True,
        )[:capacity]
        classes = states.cell_classes.reshape(-1)[order]
        facets = states.facet_classes.reshape((-1, states.mesh.cells.shape[-1]))[order]
        placement = logical_values["cell_global_ids"].sharding
        organization_content = logical_array_value_collection_digest(
            {
                "cell_classes": jax.device_put(classes, placement),
                "facet_classes": jax.device_put(facets, placement),
            },
            logical_shapes={
                "cell_classes": (logical.counts[-1],),
                "facet_classes": (logical.counts[-1], states.mesh.cells.shape[-1]),
            },
        )
        from ._collective_organization import build_collective_organization_witness

        organization_arrays, global_organization_id = (
            build_collective_organization_witness(
                original,
                states,
                logical.entity_keys,
                logical.entity_ids,
                logical.counts,
                uniform_refinement=uniform_refinement,
            )
        )
        from ._collective_geometry import build_collective_geometry_witness

        geometry_arrays, coordinate_geometry_id = build_collective_geometry_witness(
            original,
            states,
            organization_arrays,
            logical.entity_ids,
            logical.entity_owners,
            logical.counts,
            uniform_refinement=uniform_refinement,
        )
        global_organization_id = canonical_fingerprint(
            {
                "kind": "collective-source-organization",
                "scientific_occurrences": global_organization_id,
                "target_class_content": organization_content,
            }
        )
        source_evidence_id = canonical_fingerprint(
            {
                "kind": "collective-source-evidence",
                "accepted_source": source.result_id,
                "scientific_source": scientific_id,
                "scientific_binding": scientific_binding,
                "predecessor": witness.predecessor_evidence_id,
                "source_premise": premise_id,
                "global_organization": global_organization_id,
            }
        )
        target_mesh_id = canonical_fingerprint(
            {
                "kind": "cell-mesh",
                "topology": logical.topology_id,
                "geometry": logical.geometry_id,
            }
        )
        self.partition_checks = partition_checks
        self.global_checks = verdicts
        self.source = source
        self.preparation = preparation
        self.initial_states = initial
        self.compiled_states = compiled_states
        self.raw_source_blocks = raw_blocks
        self.publication_source_blocks = publication_source_blocks
        self.uniform_refinement = uniform_refinement
        self.source_exterior = exterior
        self.layout = layout
        self.neighbor_pairs = neighbor_pairs
        self.axis_name = axis_name
        self.logical_arrays = tuple(
            sorted(
                (*logical.arrays, *organization_arrays, *geometry_arrays, *native_arrays)
            )
        )
        self.entity_keys = logical.entity_keys
        self.entity_ids = logical.entity_ids
        self.entity_owners = logical.entity_owners
        self.global_entity_counts = logical.counts
        self.topology_id = logical.topology_id
        self.geometry_id = logical.geometry_id
        self.coordinate_geometry_id = coordinate_geometry_id
        self.source_mesh_id = source_mesh_id
        self.source_audit_id = source_audit_id
        self.source_evidence_id = source_evidence_id
        self.global_organization_id = global_organization_id
        self.mesh_id = target_mesh_id
        self.partition_count = partition_checks.shape[0]
        self.checks = _BISECTION_CERTIFICATE_CHECKS
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "collective-affine-bisection-evidence",
                "source_mesh": source_mesh_id,
                "source_audit": source_audit_id,
                "source_routing": source_cache_id,
                "initial_numerical_state": initial_state_id,
                "compiled_numerical_state": logical_array_value_collection_digest(
                    {
                        jax.tree_util.keystr(path): value
                        for path, value in jax.tree_util.tree_flatten_with_path(
                            compiled_states
                        )[0]
                        if isinstance(value, Array)
                    }
                ),
                "raw_source_blocks": [block.block_id for block in raw_source_blocks],
                "publication_source_blocks": [
                    block.block_id for block in publication_source_blocks
                ],
                "uniform_source": None
                if uniform_refinement is None
                else uniform_refinement.lineage_id,
                "target_mesh": target_mesh_id,
                "coordinate_geometry": coordinate_geometry_id,
                "native_midpoint_receipts": native_receipt_id,
                "source_evidence": source_evidence_id,
                "global_organization": global_organization_id,
                "partition_count": self.partition_count,
                "numerical_receipts": logical_array_value_collection_digest(
                    dict(self.logical_arrays)
                ),
                "checks": list(zip(self.checks, summary.tolist(), strict=True)),
            }
        )

    def require_passed(self) -> None:
        """Collective publication barrier, never a per-candidate synchronization."""
        summary = np.asarray(jax.device_get(self.global_checks), dtype=np.bool_)
        if not np.all(summary):
            failed = tuple(
                name
                for name, passed in zip(self.checks, summary, strict=True)
                if not passed
            )
            raise ValueError("Collective mesh certification failed: " + ", ".join(failed))


class MeshingComplianceReport(StrictModule, NonTrainableState):
    specification_id: str = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    issues: tuple[str, ...] = eqx.field(static=True)
    requested: tuple[tuple[str, float], ...] = eqx.field(static=True)
    achieved: tuple[tuple[str, float], ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        specification_id: str,
        /,
        *,
        issues: tuple[str, ...] = (),
        requested: tuple[tuple[str, float], ...] = (),
        achieved: tuple[tuple[str, float], ...] = (),
    ) -> None:
        specification = str(specification_id).strip()
        if not specification:
            raise ValueError("Compliance specification_id must be non-empty.")
        issues_ = tuple(str(value) for value in issues)
        requested_ = tuple((str(name), float(value)) for name, value in requested)
        achieved_ = tuple((str(name), float(value)) for name, value in achieved)
        self.specification_id = specification
        self.passed = not issues_
        self.issues = issues_
        self.requested = requested_
        self.achieved = achieved_
        self.report_id = canonical_fingerprint(
            {
                "kind": "meshing-compliance-report",
                "specification": specification,
                "issues": issues_,
                "requested": requested_,
                "achieved": achieved_,
            }
        )


class MeshingRuntimeInfo(StrictModule, NonTrainableState):
    provider_id: str = eqx.field(static=True)
    actual_version: str = eqx.field(static=True)
    execution_mode: MeshingExecutionMode = eqx.field(static=True)
    deterministic: bool = eqx.field(static=True)
    enforced_limits: tuple[str, ...] = eqx.field(static=True)
    unenforced_limits: tuple[str, ...] = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider_id: str,
        actual_version: str,
        execution_mode: MeshingExecutionMode,
        /,
        *,
        deterministic: bool,
        enforced_limits: tuple[str, ...] = (),
        unenforced_limits: tuple[str, ...] = (),
    ) -> None:
        provider = str(provider_id).strip()
        version = str(actual_version).strip()
        if not provider or not version:
            raise ValueError("Runtime provider and version identities must be non-empty.")
        if not isinstance(execution_mode, MeshingExecutionMode):
            raise TypeError("execution_mode must be MeshingExecutionMode.")
        self.provider_id = provider
        self.actual_version = version
        self.execution_mode = execution_mode
        self.deterministic = bool(deterministic)
        self.enforced_limits = tuple(str(value) for value in enforced_limits)
        self.unenforced_limits = tuple(str(value) for value in unenforced_limits)
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "meshing-runtime-info",
                "provider": provider,
                "actual_version": version,
                "execution_mode": execution_mode.value,
                "deterministic": bool(deterministic),
                "enforced_limits": self.enforced_limits,
                "unenforced_limits": self.unenforced_limits,
            }
        )


def _require_bound_evidence(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    audit: CellMeshAuditReport,
    trace: MeshingTrace,
    runtime: MeshingRuntimeInfo,
    certification: MeshCertificationReport | None,
    /,
) -> None:
    """Refuse evidence of other inputs and mandatory checks left undecided."""

    geometry_id = cell_geometry_id(geometry)
    audit.require_decided()
    if certification is not None:
        if not isinstance(certification, MeshCertificationReport):
            raise TypeError("certification must be MeshCertificationReport or None.")
        if (
            certification.mesh_id != mesh.mesh_id
            or certification.geometry_id != geometry_id
            or certification.audit_report_id != audit.report_id
        ):
            raise ValueError("Certification must be bound to the result mesh and audit.")
        certification.require_passed()
    binding = trace.binding
    if binding is not None and (
        binding.topology_id != mesh.topology_id
        or binding.geometry_id != geometry_id
        or binding.geometry_layout_id != geometry.geometry_layout_id
        or binding.runtime_id != runtime.runtime_id
        or audit.policy_id not in binding.policy_ids
    ):
        raise ValueError(
            "Trace evidence must be bound to the result topology, coordinates, "
            "layout, audit policy and runtime."
        )


class CollectiveMeshStorageBinding(StrictModule, NonTrainableState):
    """Actual theorem-bank byte admission computed before owner-local construction."""

    evidence: AbstractCollectiveMeshTheorem
    arrays: tuple[tuple[str, Array], ...]
    global_entity_counts: tuple[int, ...] = eqx.field(static=True)
    content_id: str = eqx.field(static=True)

    def __init__(
        self,
        evidence: AbstractCollectiveMeshTheorem,
        arrays: tuple[tuple[str, Array], ...],
        global_entity_counts: tuple[int, ...],
        /,
    ) -> None:
        if not isinstance(evidence, AbstractCollectiveMeshTheorem):
            raise TypeError(
                "Scientific storage admission requires an actual owning collective theorem."
            )
        evidence.require_passed()
        if global_entity_counts != evidence.global_entity_counts:
            raise ValueError(
                "Actual storage entity counts differ from the accepted numerical theorem."
            )
        stored = dict(arrays)
        theorem = dict(evidence.logical_arrays)
        if any(name not in stored for name in theorem):
            raise ValueError("Actual storage omits an accepted scientific bank.")
        actual = {name: stored[name] for name in theorem}
        digest = logical_array_value_collection_digest(actual)
        if digest != logical_array_value_collection_digest(theorem):
            raise ValueError(
                "Actual storage scientific bytes differ from their accepted collective theorem."
            )
        self.evidence = evidence
        self.arrays = tuple(sorted(actual.items()))
        self.global_entity_counts = global_entity_counts
        self.content_id = canonical_fingerprint(
            {
                "kind": "accepted-collective-storage-bank",
                "evidence": evidence.evidence_id,
                "counts": global_entity_counts,
                "actual_bytes": digest,
            }
        )

    def require_storage(
        self,
        mesh: CellMesh,
        evidence: AbstractCollectiveMeshTheorem,
        /,
    ) -> None:
        storage = mesh.storage
        if storage is None or (
            evidence.evidence_id != self.evidence.evidence_id
            or evidence.mesh_id != self.evidence.mesh_id
            or evidence.coordinate_geometry_id != self.evidence.coordinate_geometry_id
            or storage.global_entity_counts != self.global_entity_counts
        ):
            raise ValueError(
                "Local storage differs from its actually admitted collective source."
            )
        actual = dict(storage.logical_arrays)
        if any(
            name not in actual or actual[name] is not value for name, value in self.arrays
        ):
            raise ValueError(
                "Local storage lost its immutable accepted scientific bank bindings."
            )


def restore_collective_mesh_storage_binding(
    mesh: CellMesh,
    evidence: AbstractCollectiveMeshTheorem,
    /,
) -> CollectiveMeshStorageBinding:
    """Admit independently restored scientific arrays by actual content, not aliases."""
    if mesh.storage is None:
        raise ValueError(
            "Collective restored storage requires an actual owner-local carrier."
        )
    return CollectiveMeshStorageBinding(
        evidence,
        mesh.storage.logical_arrays,
        mesh.storage.global_entity_counts,
    )


class CellMeshingResult(StrictModule, NonTrainableState):
    """Successful meshing publication with bound, decided evidence.

    Construction refuses a failed audit, a REJECT-governed check left
    unresolved, a failed or unresolved trace stage, and (when supplied) a route
    certification that is not passed or not bound to this mesh and audit.
    """

    mesh: CellMesh
    geometry: CellGeometrySpec
    coordinate_contract: SpatialCoordinateContract
    boundary: SurfaceModel | None
    patches: tuple[MeshPatch, ...]
    zones: tuple[MeshZone, ...]
    labels: tuple[MeshLabel, ...]
    attributes: tuple[MeshAttribute, ...]
    associations: tuple[GeometryAssociation, ...]
    audit: CellMeshAuditReport
    quality: CellQualityReport
    compliance: MeshingComplianceReport
    certification: MeshCertificationReport | None
    region_evidence: RegionMeshingEvidence | None
    region_boundary_evidence: tuple[RegionBoundaryEvidence, ...]
    collective_evidence: AbstractCollectiveMeshTheorem | None
    trace: MeshingTrace
    adapter_reports: tuple[AdapterReport, ...]
    provider: MeshingProviderInfo
    runtime: MeshingRuntimeInfo
    derivative_mode: MeshingDerivativeMode = eqx.field(static=True)
    storage_binding: CollectiveMeshStorageBinding | None
    provenance: SemanticProvenance
    result_id: str = eqx.field(static=True)
    collective_certificates: (
        tuple[GlobalEmbeddingCertificate, DomainCoverageCertificate] | None
    )
    execution_evidence: NativeExecutionRecord | None
    surface_source: SurfaceSourceCharts | None

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        coordinate_contract: SpatialCoordinateContract,
        audit: CellMeshAuditReport,
        quality: CellQualityReport,
        compliance: MeshingComplianceReport,
        trace: MeshingTrace,
        provider: MeshingProviderInfo,
        runtime: MeshingRuntimeInfo,
        derivative_mode: MeshingDerivativeMode,
        provenance: SemanticProvenance,
        /,
        *,
        boundary: SurfaceModel | None = None,
        patches: tuple[MeshPatch, ...] = (),
        zones: tuple[MeshZone, ...] = (),
        labels: tuple[MeshLabel, ...] = (),
        attributes: tuple[MeshAttribute, ...] = (),
        associations: tuple[GeometryAssociation, ...] = (),
        adapter_reports: tuple[AdapterReport, ...] = (),
        certification: MeshCertificationReport | None = None,
        region_evidence: RegionMeshingEvidence | None = None,
        region_boundary_evidence: tuple[RegionBoundaryEvidence, ...] = (),
        collective_evidence: AbstractCollectiveMeshTheorem | None = None,
        scope_projections: tuple[tuple[str, MeshScopeProjection], ...] | None = None,
        attribute_projections: tuple[tuple[str, MeshAttributeProjection], ...]
        | None = None,
        storage_binding: CollectiveMeshStorageBinding | None = None,
        collective_certificates: tuple[
            GlobalEmbeddingCertificate, DomainCoverageCertificate
        ]
        | None = None,
        execution_evidence: NativeExecutionRecord | None = None,
        surface_source: SurfaceSourceCharts | None = None,
    ) -> None:
        from ._initial_certification import InitialCollectiveMeshEvidence

        if execution_evidence is not None:
            if not isinstance(execution_evidence, NativeExecutionRecord):
                raise TypeError(
                    "execution_evidence must be an actual native execution record."
                )
            execution_evidence.require_valid()
        if surface_source is not None:
            if not isinstance(surface_source, SurfaceSourceCharts):
                raise TypeError(
                    "surface_source must be an actual original native surface chart premise."
                )
            if certification is None or certification.fidelity is None:
                raise ValueError(
                    "Retained surface charts require the publication's actual continuous source certificate."
                )
            if (
                surface_source.coordinate_contract.spatial_id
                != coordinate_contract.spatial_id
            ):
                raise ValueError(
                    "Retained surface charts bind another physical coordinate contract."
                )
            surface_source.require_bound(mesh, geometry, certification.request.source)

        if mesh.storage is not None:
            if not isinstance(collective_evidence, AbstractCollectiveMeshTheorem):
                raise ValueError(
                    "Owner-local results require consumed collective mesh evidence."
                )
            if (
                collective_evidence.mesh_id != mesh.mesh_id
                or collective_evidence.partition_count != mesh.storage.partition_count
                or collective_evidence.evidence_id != mesh.storage.evidence_id
                or collective_evidence.coordinate_geometry_id
                != cell_geometry_id(geometry)
            ):
                raise ValueError(
                    "Collective evidence differs from the owner-local mesh publication."
                )
            collective_evidence.require_passed()
            if storage_binding is None:
                if any(
                    not value.is_fully_addressable
                    for _, value in mesh.storage.logical_arrays
                ):
                    raise ValueError(
                        "Nonaddressable scientific storage requires actual all-owner bank admission."
                    )
                storage_binding = restore_collective_mesh_storage_binding(
                    mesh, collective_evidence
                )
            storage_binding.require_storage(mesh, collective_evidence)
        elif collective_evidence is not None:
            raise ValueError(
                "Dense serial results cannot carry owner-local collective evidence."
            )
        if collective_certificates is not None:
            _require_collective_certificates(mesh, geometry, collective_certificates)
        self.collective_certificates = collective_certificates
        self.storage_binding = storage_binding
        geometry.resolve(mesh)
        if boundary is not None and not isinstance(boundary, SurfaceModel):
            raise TypeError("boundary must be SurfaceModel or None.")
        if not isinstance(audit, CellMeshAuditReport) or audit.mesh_id != mesh.mesh_id:
            raise ValueError("Audit must be bound to the result mesh.")
        expected_scope = (
            CellMeshAuditScope.DENSE_SERIAL
            if mesh.storage is None
            else CellMeshAuditScope.OWNER_LOCAL_CLOSURE
        )
        if audit.audit_scope != expected_scope or audit.storage_id != (
            None if mesh.storage is None else mesh.storage.storage_id
        ):
            raise ValueError(
                "Audit storage scope differs from the accepted mesh carrier."
            )
        if (
            audit.geometry_layout_id != geometry.geometry_layout_id
            or audit.geometry_id != cell_geometry_id(geometry)
        ):
            raise ValueError(
                "Audit must be bound to the result geometry values and layout."
            )
        audit.require_passed()
        if (
            not isinstance(quality, CellQualityReport)
            or quality.report_id != audit.quality.report_id
        ):
            raise ValueError("Quality report must match the mesh audit.")
        if not isinstance(compliance, MeshingComplianceReport) or not compliance.passed:
            raise ValueError("Successful meshing results require passed compliance.")
        if not isinstance(trace, MeshingTrace) or not trace.successful:
            raise ValueError("Successful meshing results require a successful trace.")
        if (
            not isinstance(runtime, MeshingRuntimeInfo)
            or runtime.provider_id != provider.provider_id
        ):
            raise ValueError("Meshing runtime must match the provider.")
        _require_bound_evidence(mesh, geometry, audit, trace, runtime, certification)
        if not isinstance(derivative_mode, MeshingDerivativeMode):
            raise TypeError("derivative_mode must be MeshingDerivativeMode.")
        patches_ = tuple(patches)
        zones_ = validate_mesh_zones(tuple(zones))
        labels_ = validate_mesh_labels(tuple(labels))
        if not all(isinstance(value, MeshPatch) for value in patches_):
            raise TypeError("patches must contain MeshPatch values.")
        if region_evidence is not None:
            if not isinstance(region_evidence, RegionMeshingEvidence):
                raise TypeError("region_evidence must be RegionMeshingEvidence or None.")
            region_evidence.require_current(mesh, zones_, patches_, geometry=geometry)
        boundary_regions = tuple(region_boundary_evidence)
        if not all(
            isinstance(value, RegionBoundaryEvidence) for value in boundary_regions
        ):
            raise TypeError(
                "region_boundary_evidence must contain RegionBoundaryEvidence values."
            )
        source_regions = {value.source_region_id for value in boundary_regions}
        if len(source_regions) != len(boundary_regions):
            raise ValueError("Source region boundary evidence identities must be unique.")
        for value in boundary_regions:
            value.require_current(mesh, labels_, patches_)
        for patch in patches_:
            if patch.source_adjacent_region_ids is not None and any(
                region is not None and region not in source_regions
                for region in patch.source_adjacent_region_ids
            ):
                raise ValueError(
                    "Surface patch source-region incidence lacks authoritative boundary evidence."
                )
        if not all(isinstance(value, MeshAttribute) for value in attributes):
            raise TypeError("attributes must contain MeshAttribute values.")
        if not all(isinstance(value, GeometryAssociation) for value in associations):
            raise TypeError("associations must contain GeometryAssociation values.")
        if not all(isinstance(value, AdapterReport) for value in adapter_reports):
            raise TypeError("adapter_reports must contain AdapterReport values.")
        if (
            boundary is not None
            and boundary.metadata.coordinate_contract.spatial_id
            != coordinate_contract.spatial_id
        ):
            raise ValueError("Boundary coordinate contract must match the result.")
        evidence_issues = _mesh_evidence_issues(
            mesh,
            boundary,
            patches_,
            zones_,
            labels_,
            attributes,
            associations,
        )
        if evidence_issues:
            raise ValueError(
                "Result evidence is not bound to the mesh: " + "; ".join(evidence_issues)
            )
        if audit.evidence_id != _evidence_id(
            patches_,
            zones_,
            labels_,
            attributes,
            associations,
        ):
            raise ValueError(
                "Result organization and associations must match the audited evidence."
            )
        self.mesh = mesh
        self.geometry = geometry
        self.coordinate_contract = coordinate_contract
        self.boundary = boundary
        self.patches = patches_
        self.zones = zones_
        self.labels = labels_
        self.attributes = tuple(attributes)
        self.associations = tuple(associations)
        self.audit = audit
        self.quality = quality
        self.compliance = compliance
        self.certification = certification
        self.region_evidence = region_evidence
        self.region_boundary_evidence = boundary_regions
        self.collective_evidence = collective_evidence
        self.trace = trace
        self.adapter_reports = tuple(adapter_reports)
        self.provider = provider
        self.runtime = runtime
        self.derivative_mode = derivative_mode
        self.provenance = provenance
        self.execution_evidence = execution_evidence
        self.surface_source = surface_source
        if mesh.storage is not None:
            if collective_evidence is None:
                raise ValueError(
                    "Owner-local publication lacks its actual collective proof."
                )
            if isinstance(collective_evidence, CollectiveMeshEvidence):
                original = require_original_meshing_source(
                    collective_evidence.source
                    if collective_evidence.uniform_refinement is None
                    else collective_evidence.uniform_refinement.source,
                )
                validate_local_mesh_organization(
                    original,
                    mesh,
                    dict(collective_evidence.logical_arrays),
                    patches_,
                    zones_,
                    labels_,
                    attributes,
                    scope_projections=scope_projections,
                    attribute_projections=attribute_projections,
                )
                source_identity = {
                    "source_mesh": collective_evidence.source_mesh_id,
                    "source_audit": collective_evidence.source_audit_id,
                }
            elif isinstance(collective_evidence, InitialCollectiveMeshEvidence):
                if (
                    compliance.specification_id
                    != collective_evidence.specification.specification_id
                ):
                    raise ValueError(
                        "Initial publication compliance differs from its actual authored request."
                    )
                if scope_projections is None:
                    raise ValueError(
                        "Initial results require their prepared all-owner organization scope receipts."
                    )
                validate_local_mesh_organization(
                    collective_evidence,
                    mesh,
                    dict(collective_evidence.logical_arrays),
                    patches_,
                    zones_,
                    labels_,
                    attributes,
                    scope_projections=scope_projections,
                    attribute_projections=attribute_projections,
                )
                source_identity = {
                    "source_domain": collective_evidence.compiled.domain.domain_id,
                    "compiled_domain": collective_evidence.compiled.compiled_id,
                    "specification": collective_evidence.specification.specification_id,
                }
            else:
                raise TypeError(
                    "Owner-local publication has an unknown collective theorem kind."
                )
            self.result_id = canonical_fingerprint(
                {
                    "kind": "owner-local-cell-meshing-result",
                    "mesh": mesh.mesh_id,
                    "topology": mesh.topology_id,
                    "geometry": cell_geometry_id(geometry),
                    "geometry_layout": geometry.geometry_layout_id,
                    "coordinate_contract": coordinate_contract.spatial_id,
                    **source_identity,
                    "source_evidence": collective_evidence.source_evidence_id,
                    "organization": collective_evidence.global_organization_id,
                    "collective": collective_evidence.evidence_id,
                    "provider": provider.provider_id,
                    "derivative_mode": derivative_mode.value,
                    "collective_certificates": _collective_certificate_ids(
                        collective_certificates
                    ),
                    "surface_source": None
                    if surface_source is None
                    else surface_source.source_chart_id,
                }
            )
            return
        self.result_id = canonical_fingerprint(
            {
                "kind": "cell-meshing-result",
                "mesh": mesh.mesh_id,
                "geometry_layout": geometry.geometry_layout_id,
                "geometry": audit.geometry_id,
                "coordinate_contract": coordinate_contract.spatial_id,
                "boundary": None if boundary is None else boundary.model_id,
                "patches": [value.patch_id for value in patches_],
                "zones": [value.zone_id for value in zones_],
                "labels": [value.label_id for value in labels_],
                "attributes": [value.attribute_id for value in attributes],
                "associations": [value.association_id for value in associations],
                "audit": audit.report_id,
                "quality": quality.report_id,
                "compliance": compliance.report_id,
                "certification": (
                    None if certification is None else certification.report_id
                ),
                "regions": (
                    None if region_evidence is None else region_evidence.evidence_id
                ),
                "region_boundaries": tuple(
                    value.evidence_id for value in boundary_regions
                ),
                "collective": (
                    None
                    if collective_evidence is None
                    else collective_evidence.evidence_id
                ),
                "trace": trace.trace_id,
                "adapter_reports": [value.report_id for value in adapter_reports],
                "provider": provider.provider_id,
                "runtime": runtime.runtime_id,
                "derivative_mode": derivative_mode.value,
                "provenance": provenance.semantic_id,
                "collective_certificates": None,
                "surface_source": None
                if surface_source is None
                else surface_source.source_chart_id,
            }
        )

    @checked
    def with_execution_evidence(
        self, record: NativeExecutionRecord, /
    ) -> CellMeshingResult:
        """Attach an actual ended receipt without reconstructing scientific holders.

        The receipt is excluded from ``result_id`` and every scientific holder is
        already certified, so only this field changes. Rebuilding would
        refingerprint geometry that may be traced inside a derivative route.
        Only this root is unflattened from its direct children, so all shared
        nested authority keeps its object identity (``eqx.tree_at`` deliberately
        copies every composite node and would duplicate each shared subtree).
        """
        record.require_valid()
        target = (jax.tree_util.GetAttrKey("execution_evidence"),)
        children, structure = jax.tree_util.tree_flatten_with_path(
            self, is_leaf=lambda value: value is not self
        )
        if sum(path == target for path, _ in children) != 1:
            raise RuntimeError("A meshing result lost its execution evidence slot.")
        return jax.tree_util.tree_unflatten(
            structure,
            [record if path == target else value for path, value in children],
        )


def _collective_certificate_ids(
    certificates: tuple[GlobalEmbeddingCertificate, DomainCoverageCertificate] | None,
    /,
) -> tuple[str, str] | None:
    return (
        None
        if certificates is None
        else (certificates[0].certificate_id, certificates[1].certificate_id)
    )


def _require_collective_certificates(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    certificates: tuple[GlobalEmbeddingCertificate, DomainCoverageCertificate],
    /,
) -> None:
    """Admit only target-bound, wholly certified collective embedding and coverage."""
    if mesh.storage is None:
        raise ValueError(
            "Collective target certificates belong only to owner-local publications."
        )
    if (
        type(certificates) is not tuple
        or len(certificates) != 2
        or not isinstance(certificates[0], GlobalEmbeddingCertificate)
        or not isinstance(certificates[1], DomainCoverageCertificate)
    ):
        raise TypeError(
            "collective_certificates must be (GlobalEmbeddingCertificate, DomainCoverageCertificate)."
        )
    embedding, coverage = certificates
    embedding.binding.require(mesh, geometry)
    coverage.binding.require(mesh, geometry)
    if (
        embedding.status != "certified"
        or coverage.status != "certified"
        or coverage.embedding_certificate_id != embedding.certificate_id
        or mesh.storage.evidence_id not in coverage.premise_certificate_ids
    ):
        raise ValueError(
            "Collective target certificates must be certified and bound to this owner-local partition."
        )
    if any(
        value is None
        for value in (
            *coverage.achieved_region_measures,
            *coverage.achieved_region_measure_bounds,
            *coverage.integration_error_bounds,
        )
    ):
        raise ValueError(
            "Collective target coverage has absent required region integrals."
        )


def require_original_meshing_source(
    result: CellMeshingResult,
    /,
) -> CellMeshingResult | InitialCollectiveMeshEvidence:
    """Validate the execution chain and resolve its actual scientific owner."""
    if not isinstance(result, CellMeshingResult):
        raise TypeError("Original source resolution requires an accepted meshing result.")
    from ._initial_certification import InitialCollectiveMeshEvidence

    active: set[int] = set()
    resolved: dict[int, CellMeshingResult | InitialCollectiveMeshEvidence] = {}

    def resolve(
        current: CellMeshingResult,
    ) -> CellMeshingResult | InitialCollectiveMeshEvidence:
        identity = id(current)
        if identity in active:
            raise ValueError(
                "Collective source authority contains a cyclic predecessor or scientific owner."
            )
        if identity in resolved:
            return resolved[identity]
        active.add(identity)
        try:
            evidence = current.collective_evidence
            if isinstance(evidence, InitialCollectiveMeshEvidence):
                evidence.require_passed()
                storage = current.mesh.storage
                if (
                    storage is None
                    or storage.evidence_id != evidence.evidence_id
                    or current.mesh.mesh_id != evidence.mesh_id
                    or cell_geometry_id(current.geometry)
                    != evidence.coordinate_geometry_id
                    or current.compliance.specification_id
                    != evidence.specification.specification_id
                ):
                    raise ValueError(
                        "Initial scientific source lost its actual theorem and storage binding."
                    )
                owner: CellMeshingResult | InitialCollectiveMeshEvidence = evidence
            elif isinstance(evidence, CollectiveMeshEvidence):
                evidence.require_passed()
                predecessor = evidence.source
                storage = current.mesh.storage
                if (
                    evidence.mesh_id != current.mesh.mesh_id
                    or evidence.topology_id != current.mesh.topology_id
                    or evidence.geometry_id != current.mesh.geometry_id
                    or evidence.coordinate_geometry_id
                    != cell_geometry_id(current.geometry)
                    or storage is None
                    or evidence.evidence_id != storage.evidence_id
                    or evidence.partition_count != storage.partition_count
                    or evidence.source_mesh_id != predecessor.mesh.mesh_id
                ):
                    raise ValueError(
                        "Collective predecessor authority lost its actual source binding."
                    )
                predecessor_owner = resolve(predecessor)
                uniform = evidence.uniform_refinement
                owner = predecessor_owner if uniform is None else resolve(uniform.source)
                audit_id = (
                    owner.evidence_id
                    if isinstance(owner, InitialCollectiveMeshEvidence)
                    else owner.audit.report_id
                )
                if evidence.source_audit_id != audit_id:
                    raise ValueError(
                        "A collective predecessor lost its actual global scientific audit binding."
                    )
            elif evidence is not None:
                raise TypeError(
                    "Collective source authority has an unknown theorem kind."
                )
            else:
                certification = current.certification
                if certification is None:
                    raise ValueError(
                        "The original scientific source has no full acceptance theorem."
                    )
                certification.require_passed()
                certification.request.validate_source_integrity()
                if (
                    certification.mesh_id != current.mesh.mesh_id
                    or certification.topology_id != current.mesh.topology_id
                    or certification.geometry_id != cell_geometry_id(current.geometry)
                    or certification.geometry_layout_id
                    != current.geometry.geometry_layout_id
                    or certification.audit_report_id != current.audit.report_id
                ):
                    raise ValueError(
                        "The original certification lost its actual scientific mesh/map/audit binding."
                    )
                owner = current
            resolved[identity] = owner
            return owner
        finally:
            active.remove(identity)

    return resolve(result)


__all__ = [
    "CellMeshingResult",
    "CollectiveMeshEvidence",
    "MeshingComplianceReport",
    "MeshingRuntimeInfo",
]
