#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""PIC restart through lifecycle distributed checkpoints.

`PICRestartPlan` composes the per-component restart leaves of one PIC run —
`ElectromagneticPICPlan.checkpoint` components (species with their persistent
identities and lineage, the field state with its ADE/plasma/CPML/PML memory,
the particle-boundary ledger, recorders, process states such as radiation
accumulators and QED photon/pair banks, the staggered field history), the
moving-window epoch, and auxiliary `PICRestartState` owners (for example
boosted-frame diagnostic buffers) — into one topology-neutral
`PICRestartManifest`, and publishes and restores them with
`phydrax.lifecycle` addressable-shard checkpoints.

Every component is admitted only by an owner with the same identity. A
restart on the topology that wrote the checkpoint is ``"bitwise"``; a restart
on a different mesh repartitions particles (with their slot-aligned process
state) into the new slot blocks and process banks by their own positions (slot
permutations that leave the state exact) and is a ``"tolerance"``
restart, because the continued run sums deposits in a different order. The
lifecycle `TopologyRestartPolicy` admits or refuses the relation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec, Sharding, SingleDeviceSharding

from .._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from .._fingerprint import canonical_fingerprint, canonical_json
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..lifecycle._chunk_repository import ArtifactRepository, ChunkEncoding
from ..lifecycle._distributed_checkpoint import (
    assemble_distributed_checkpoint_from_repository,
    ProcessCheckpointPublication,
    publish_process_checkpoint,
    restore_global_array_from_checkpoint,
)
from ..lifecycle._models import CheckpointManifest
from ..lifecycle._restart_topology import (
    admit_topology_restart,
    RestartClass,
    TopologyRestartPolicy,
    TopologyRestartRelation,
)
from ._distributed_pic import DistributedElectromagneticPICPlan
from ._electromagnetic_pic import (
    ElectromagneticPICPlan,
    ElectromagneticPICState,
    PICRestartCheckpoint,
)
from ._moving_window_pic import PICMovingWindowPlan, PICMovingWindowState
from ._pic_field_solver import (
    PICRestartComponent,
    PICRestartState,
    restart_component,
    restore_component,
)


_WINDOW_COMPONENT = "moving-window"
_SINGLE_DEVICE_TOPOLOGY = canonical_fingerprint({"kind": "single-device-pic-topology"})
# One shard per component leaf and device: decomposed runs with stateful
# processes exceed the generic archive member cap.
_PIC_CHECKPOINT_LIMITS = replace(DEFAULT_ARRAY_ARCHIVE_LIMITS, max_members=1 << 16)


class PICRestartManifest(StrictModule, NonTrainableState):
    """Topology-neutral inventory of one PIC run's restart components.

    ``names``/``owners`` list every component with the identity that admits
    it; ``shapes``/``dtypes`` are its leaf signatures. ``topology_id`` and
    ``part_count`` record the execution that wrote it; ``window_component`` and
    ``auxiliary_names`` identify the moving-window epoch and auxiliary owners'
    components.
    """

    plan_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    owners: tuple[str, ...] = eqx.field(static=True)
    shapes: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    dtypes: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    window_component: bool = eqx.field(static=True)
    auxiliary_names: tuple[str, ...] = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan_id: str,
        topology_id: str,
        part_count: int,
        names: Sequence[str],
        owners: Sequence[str],
        shapes: Sequence[Sequence[Sequence[int]]],
        dtypes: Sequence[Sequence[str]],
        /,
        *,
        window_component: bool,
        auxiliary_names: Sequence[str],
    ) -> None:
        names_ = tuple(str(value) for value in names)
        owners_ = tuple(str(value) for value in owners)
        shapes_ = tuple(
            tuple(tuple(int(size) for size in shape) for shape in leaves)
            for leaves in shapes
        )
        dtypes_ = tuple(
            tuple(np.dtype(value).str for value in leaves) for leaves in dtypes
        )
        auxiliary = tuple(str(value) for value in auxiliary_names)
        if not (len(names_) == len(owners_) == len(shapes_) == len(dtypes_)):
            raise ValueError("PIC restart inventory entries must align.")
        if len(set(names_)) != len(names_):
            raise ValueError("PIC restart component names must be distinct.")
        if any(
            len(shape) != len(dtype)
            for shape, dtype in zip(shapes_, dtypes_, strict=True)
        ):
            raise ValueError("PIC restart leaf shapes and dtypes must align.")
        if not set(auxiliary) <= set(names_) or (
            window_component and _WINDOW_COMPONENT not in names_
        ):
            raise ValueError("PIC restart auxiliary components are missing.")
        if int(part_count) <= 0:
            raise ValueError("part_count must be positive.")
        self.plan_id = str(plan_id)
        self.topology_id = str(topology_id)
        self.part_count = int(part_count)
        self.names = names_
        self.owners = owners_
        self.shapes = shapes_
        self.dtypes = dtypes_
        self.window_component = bool(window_component)
        self.auxiliary_names = auxiliary
        self.manifest_id = canonical_fingerprint(
            {"kind": "pic-restart-manifest", **self.record()}
        )

    def record(self) -> dict[str, Any]:
        """Canonical JSON-compatible record (excluding the manifest identity)."""
        return {
            "plan_id": self.plan_id,
            "topology_id": self.topology_id,
            "part_count": self.part_count,
            "names": list(self.names),
            "owners": list(self.owners),
            "shapes": [[list(shape) for shape in leaves] for leaves in self.shapes],
            "dtypes": [list(leaves) for leaves in self.dtypes],
            "window_component": self.window_component,
            "auxiliary_names": list(self.auxiliary_names),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any], /) -> PICRestartManifest:
        return cls(
            record["plan_id"],
            record["topology_id"],
            record["part_count"],
            record["names"],
            record["owners"],
            record["shapes"],
            record["dtypes"],
            window_component=record["window_component"],
            auxiliary_names=record["auxiliary_names"],
        )

    def same_inventory(self, other: PICRestartManifest, /) -> bool:
        """Whether two manifests hold the same components with the same leaves."""
        return (
            self.plan_id == other.plan_id
            and self.names == other.names
            and self.owners == other.owners
            and self.shapes == other.shapes
            and self.dtypes == other.dtypes
            and self.window_component == other.window_component
            and self.auxiliary_names == other.auxiliary_names
        )


class PICRestartResult(StrictModule):
    """A restored run state and its restart-equivalence evidence."""

    state: Any
    auxiliary_states: tuple[Any, ...]
    source: PICRestartManifest
    restart_class: RestartClass = eqx.field(static=True)
    relation_id: str = eqx.field(static=True)
    admission_id: str = eqx.field(static=True)


type PICRestartRun = ElectromagneticPICPlan | DistributedElectromagneticPICPlan


def _paths(names: Sequence[str], counts: Sequence[int], /) -> dict[str, tuple[str, ...]]:
    """Lifecycle array paths of every component leaf and the manifest record."""
    tree = {
        "components": {
            name: tuple(np.zeros((), dtype=np.int8) for _ in range(count))
            for name, count in zip(names, counts, strict=True)
        },
        "restart_manifest": np.zeros((), dtype=np.int8),
    }
    flattened = jax.tree_util.tree_flatten_with_path(tree)[0]
    keyed = [jax.tree_util.keystr(path) for path, _ in flattened]
    paths: dict[str, tuple[str, ...]] = {}
    for name, count in zip(names, counts, strict=True):
        prefix = jax.tree_util.keystr(
            (jax.tree_util.DictKey("components"), jax.tree_util.DictKey(name))
        )
        paths[name] = tuple(value for value in keyed if value.startswith(prefix + "["))
        if len(paths[name]) != count:
            raise ValueError(f"Restart component {name!r} paths are ambiguous.")
    paths["<manifest>"] = (
        jax.tree_util.keystr((jax.tree_util.DictKey("restart_manifest"),)),
    )
    return paths


class PICRestartPlan(StrictModule, NonTrainableState):
    """Publishes and restores one PIC run through lifecycle checkpoints.

    ``run`` is an `ElectromagneticPICPlan` or a
    `DistributedElectromagneticPICPlan`; ``window`` the moving window driving
    it (its epoch is a component); ``auxiliaries`` further restart owners whose
    states are passed alongside. The default policy admits topology-changing
    restarts as tolerance restarts within ``repartition_tolerance``. ``limits``
    bound the published and restored shards (one per component leaf and
    device).
    """

    run: ElectromagneticPICPlan | DistributedElectromagneticPICPlan
    window: PICMovingWindowPlan | None
    auxiliaries: tuple[Any, ...]
    policy: TopologyRestartPolicy
    repartition_tolerance: float = eqx.field(static=True)
    limits: ArrayArchiveLimits = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    analysis_plan_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)

    def __init__(
        self,
        run: PICRestartRun,
        /,
        *,
        window: PICMovingWindowPlan | None = None,
        auxiliaries: Sequence[Any] = (),
        policy: TopologyRestartPolicy | None = None,
        repartition_tolerance: float = 1.0e-10,
        limits: ArrayArchiveLimits = _PIC_CHECKPOINT_LIMITS,
    ) -> None:
        if isinstance(run, DistributedElectromagneticPICPlan):
            pic = run.pic
            topology, parts = run.topology_id, run.solver.decomposition.part_count
        elif isinstance(run, ElectromagneticPICPlan):
            pic = run
            topology, parts = _SINGLE_DEVICE_TOPOLOGY, 1
        else:
            raise TypeError(
                "run must be an ElectromagneticPICPlan or its distributed plan."
            )
        if window is not None and (
            not isinstance(window, PICMovingWindowPlan)
            or window.pic.plan_id != pic.plan_id
        ):
            raise ValueError("The moving window must drive the restarted PIC plan.")
        auxiliary = tuple(auxiliaries)
        if any(not isinstance(value, PICRestartState) for value in auxiliary):
            raise TypeError("Restart auxiliaries must implement PICRestartState.")
        tolerance = float(repartition_tolerance)
        if not (np.isfinite(tolerance) and tolerance > 0.0):
            raise ValueError("repartition_tolerance must be finite and positive.")
        policy_ = (
            TopologyRestartPolicy(
                allow_topology_change=True,
                allow_tolerance_restart=True,
                maximum_relative_tolerance=tolerance,
            )
            if policy is None
            else policy
        )
        if not isinstance(policy_, TopologyRestartPolicy):
            raise TypeError("policy must be TopologyRestartPolicy or None.")
        if not isinstance(limits, ArrayArchiveLimits):
            raise TypeError("limits must be ArrayArchiveLimits.")
        self.run = run
        self.window = window
        self.auxiliaries = auxiliary
        self.policy = policy_
        self.repartition_tolerance = tolerance
        self.limits = limits
        self.topology_id = topology
        self.part_count = parts
        self.analysis_plan_id = pic.plan_id
        self.numeric_revision_id = canonical_fingerprint(
            {
                "kind": "pic-restart-inventory",
                "pic": pic.plan_id,
                "window": None if window is None else window.plan_id,
                "auxiliaries": len(auxiliary),
            }
        )

    @property
    def pic(self) -> ElectromagneticPICPlan:
        run = self.run
        return run.pic if isinstance(run, DistributedElectromagneticPICPlan) else run

    # -- composition --------------------------------------------------------------

    def _pic_state(self, state: Any, /) -> ElectromagneticPICState:
        if self.window is None:
            if not isinstance(state, ElectromagneticPICState):
                raise TypeError("state must be ElectromagneticPICState.")
            return state
        if not isinstance(state, PICMovingWindowState):
            raise TypeError("A moving-window restart requires PICMovingWindowState.")
        return state.pic

    def components(
        self, state: Any, /, *, auxiliary_states: Sequence[Any] = ()
    ) -> tuple[PICRestartComponent, ...]:
        """Every restart component: PIC, moving-window epoch, then auxiliaries."""
        values = tuple(auxiliary_states)
        if len(values) != len(self.auxiliaries):
            raise ValueError("One auxiliary state is required per restart auxiliary.")
        components = list(self.pic.checkpoint(self._pic_state(state)).components)
        window = self.window
        if window is not None and isinstance(state, PICMovingWindowState):
            components.append(
                restart_component(
                    _WINDOW_COMPONENT,
                    window.plan_id,
                    (state.origin, state.cumulative_cells, state.shift_epoch),
                )
            )
        components.extend(
            owner.restart_component(value)
            for owner, value in zip(self.auxiliaries, values, strict=True)
        )
        return tuple(components)

    def manifest(
        self, components: Sequence[PICRestartComponent], /
    ) -> PICRestartManifest:
        values = tuple(components)
        count = len(self.auxiliaries)
        return PICRestartManifest(
            self.analysis_plan_id,
            self.topology_id,
            self.part_count,
            tuple(value.name for value in values),
            tuple(value.owner_id for value in values),
            tuple(tuple(jnp.shape(leaf) for leaf in value.leaves) for value in values),
            tuple(
                tuple(jnp.result_type(leaf).str for leaf in value.leaves)
                for value in values
            ),
            window_component=self.window is not None,
            auxiliary_names=tuple(value.name for value in values[len(values) - count :])
            if count
            else (),
        )

    # -- lifecycle ------------------------------------------------------------------

    def publish(
        self,
        repository: ArtifactRepository,
        state: Any,
        /,
        *,
        checkpoint_id: str,
        writer_id: str,
        auxiliary_states: Sequence[Any] = (),
        attempt_id: str | None = None,
        encoding: ChunkEncoding = "identity",
        parent_manifest: CheckpointManifest | None = None,
    ) -> ProcessCheckpointPublication:
        """Publish this process's addressable shards of every component leaf."""
        components = self.components(state, auxiliary_states=auxiliary_states)
        manifest = self.manifest(components)
        record = canonical_json(
            {**manifest.record(), "manifest_id": manifest.manifest_id}
        )
        tree = {
            "components": {value.name: value.leaves for value in components},
            "restart_manifest": jnp.asarray(
                np.frombuffer(record.encode("utf-8"), dtype=np.uint8).copy()
            ),
        }
        return publish_process_checkpoint(
            repository,
            checkpoint_id,
            self.topology_id,
            tree,
            analysis_plan_id=self.analysis_plan_id,
            numeric_revision_id=self.numeric_revision_id,
            writer_id=writer_id,
            attempt_id=attempt_id,
            encoding=encoding,
            limits=self.limits,
            parent_manifest=parent_manifest,
        )

    def assemble(
        self,
        repository: ArtifactRepository,
        checkpoint_id: str,
        /,
        *,
        expected_process_count: int,
        parent_manifest: CheckpointManifest | None = None,
    ) -> CheckpointManifest:
        """Commit the global checkpoint manifest once every process published."""
        return assemble_distributed_checkpoint_from_repository(
            repository,
            checkpoint_id,
            self.analysis_plan_id,
            self.numeric_revision_id,
            self.topology_id,
            expected_process_count=expected_process_count,
            limits=self.limits,
            parent_manifest=parent_manifest,
        )

    def _sharding(self) -> Sharding:
        run = self.run
        if isinstance(run, DistributedElectromagneticPICPlan):
            return NamedSharding(run.solver.mesh, PartitionSpec())
        return SingleDeviceSharding(jax.devices()[0])

    def restore(
        self,
        repository: ArtifactRepository,
        checkpoint: CheckpointManifest,
        /,
        *,
        parent_manifest: CheckpointManifest | None = None,
    ) -> PICRestartResult:
        """Restore, admit, and (on a new topology) repartition a published run."""
        if not isinstance(checkpoint, CheckpointManifest):
            raise TypeError("checkpoint must be a CheckpointManifest.")
        if (
            checkpoint.analysis_plan_id != self.analysis_plan_id
            or checkpoint.numeric_revision_id != self.numeric_revision_id
        ):
            raise ValueError("The checkpoint belongs to another PIC run or inventory.")
        sharding = self._sharding()

        def load(path: str) -> Array:
            return restore_global_array_from_checkpoint(
                repository,
                checkpoint,
                path,
                sharding,
                limits=self.limits,
                parent_manifest=parent_manifest,
            )

        manifest_path = _paths((), ())["<manifest>"][0]
        stored = json.loads(np.asarray(load(manifest_path)).astype(np.uint8).tobytes())
        source = PICRestartManifest.from_record(
            {key: value for key, value in stored.items() if key != "manifest_id"}
        )
        if source.manifest_id != stored.get("manifest_id"):
            raise ValueError("The PIC restart manifest record is corrupted.")
        if source.plan_id != self.analysis_plan_id or (
            source.window_component != (self.window is not None)
            or len(source.auxiliary_names) != len(self.auxiliaries)
        ):
            raise ValueError("The PIC restart inventory differs from this run.")
        same = source.topology_id == self.topology_id
        restart_class: RestartClass = "bitwise" if same else "tolerance"
        relation = TopologyRestartRelation(
            source.topology_id,
            self.topology_id,
            restart_class,
            relative_tolerance=0.0 if same else self.repartition_tolerance,
            reason="PIC repartition changes the deposit reduction order.",
        )
        admission = admit_topology_restart(relation, self.policy)
        if not admission.admitted:
            raise ValueError(f"PIC restart is not admitted: {admission.reason}")
        paths = _paths(source.names, tuple(len(value) for value in source.shapes))
        components = {
            name: PICRestartComponent(
                name, owner, tuple(load(path) for path in paths[name])
            )
            for name, owner in zip(source.names, source.owners, strict=True)
        }
        auxiliary_names = set(source.auxiliary_names)
        pic_components = tuple(
            value
            for name, value in components.items()
            if name != _WINDOW_COMPONENT and name not in auxiliary_names
        )
        pic_state = self.pic.restore(PICRestartCheckpoint(pic_components))
        run = self.run
        if isinstance(run, DistributedElectromagneticPICPlan):
            pic_state = run.place(pic_state) if same else run.repartition(pic_state)
        state: Any = pic_state
        window = self.window
        if window is not None:
            template = window.initialize(pic_state)
            origin, cells, epoch = restore_component(
                components[_WINDOW_COMPONENT],
                _WINDOW_COMPONENT,
                window.plan_id,
                (template.origin, template.cumulative_cells, template.shift_epoch),
            )
            state = PICMovingWindowState(pic_state, origin, cells, epoch)
        auxiliary_states = tuple(
            owner.restore_component(components[name])
            for owner, name in zip(self.auxiliaries, source.auxiliary_names, strict=True)
        )
        return PICRestartResult(
            state,
            auxiliary_states,
            source,
            restart_class,
            relation.relation_id,
            admission.admission_id,
        )


__all__ = [
    "PICRestartManifest",
    "PICRestartPlan",
    "PICRestartResult",
]
