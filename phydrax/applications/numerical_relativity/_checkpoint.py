#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Topology-bound local and distributed restart records for NR runtimes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import SingleDeviceSharding
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint, canonical_json
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...lifecycle._chunk_repository import (
    ArtifactManifest,
    ArtifactRepository,
    ChunkEncoding,
)
from ...lifecycle._distributed_checkpoint import (
    assemble_distributed_checkpoint_from_repository,
    ProcessCheckpointPublication,
    publish_process_checkpoint,
    restore_global_array_from_checkpoint,
)
from ...lifecycle._models import CheckpointManifest
from ...lifecycle._restart_topology import (
    admit_topology_restart,
    TopologyRestartPolicy,
    TopologyRestartRelation,
)
from ...solver._coupled_field_checkpoint import (
    CoupledFieldCheckpointPlan,
    read_coupled_field_checkpoint,
    write_coupled_field_checkpoint,
)
from ...solver._grmhd_ct import GRMHDConstrainedTransportPlan, GRMHDCTState
from ...solver._grmhd_runtime import GRMHDState
from ...solver._grrmhd_runtime import GRRMHDState
from ...solver._relativistic_finite_volume import GRHDFiniteVolumeState
from ._coupled_runtime import CoupledEvolutionState
from ._distributed import _formulation, NumericalRelativityFormulation
from ._matter_coupling import CoupledBudget
from ._state import Z4cState
from ._temporal import Z4cRuntimeState


RestartRelation: TypeAlias = Literal["exact", "tolerance"]


def _restart_field_names(
    formulation: NumericalRelativityFormulation, /
) -> tuple[str, ...]:
    normalized = _formulation(formulation)
    if normalized == "z4c":
        return ("z4c_runtime_state",)
    if normalized == "grhd":
        return ("grhd_state",)
    if normalized == "grmhd":
        return ("material", "constrained_transport", "step_size", "status")
    if normalized == "grrmhd":
        return (
            "material",
            "constrained_transport",
            "radiation",
            "step_size",
            "status",
        )
    return (
        "z4c",
        "matter",
        "coupled_budget",
        "accepted_steps",
        "rejected_steps",
        "consecutive_failures",
        "terminal",
        "next_step_id",
    )


def _restart_support_id(topology_id: str, /) -> str:
    return canonical_fingerprint(
        {"kind": "numerical-relativity-restart-support", "topology": topology_id}
    )


class NumericalRelativityRestartPolicy(StrictModule, NonTrainableState):
    """NR exact/tolerance relation backed by lifecycle restart admission."""

    relation: RestartRelation = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    topology_policy: TopologyRestartPolicy
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        relation: RestartRelation = "exact",
        /,
        *,
        absolute_tolerance: float = 0.0,
        relative_tolerance: float = 0.0,
    ):
        relation_ = str(relation)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if relation_ not in ("exact", "tolerance"):
            raise ValueError("Restart relation must be 'exact' or 'tolerance'.")
        if (
            not np.isfinite(absolute)
            or absolute < 0.0
            or not np.isfinite(relative)
            or relative < 0.0
            or (relation_ == "exact" and (absolute != 0.0 or relative != 0.0))
            or (relation_ == "tolerance" and absolute == 0.0 and relative == 0.0)
        ):
            raise ValueError("Restart tolerances are invalid for the declared relation.")
        topology_policy = TopologyRestartPolicy(
            allow_topology_change=False,
            allow_tolerance_restart=relation_ == "tolerance",
            maximum_absolute_tolerance=absolute,
            maximum_relative_tolerance=relative,
        )
        self.relation = relation_  # type: ignore[assignment]
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.topology_policy = topology_policy
        self.policy_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-restart-policy",
                "relation": relation_,
                "topology_policy": topology_policy.policy_id,
            }
        )

    def topology_relation(
        self, source_topology_id: str, target_topology_id: str, /
    ) -> TopologyRestartRelation:
        return TopologyRestartRelation(
            _restart_support_id(source_topology_id),
            _restart_support_id(target_topology_id),
            "bitwise" if self.relation == "exact" else "tolerance",
            absolute_tolerance=self.absolute_tolerance,
            relative_tolerance=self.relative_tolerance,
        )


class NumericalRelativityRestartState(StrictModule):
    """Complete authoritative runtime content at one accepted synchronization point."""

    time: Array
    step_index: Array
    fields: tuple[Any, ...]
    formulation: NumericalRelativityFormulation = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    topology_epoch: int = eqx.field(static=True)
    field_names: tuple[str, ...] = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        formulation: NumericalRelativityFormulation,
        runtime_id: str,
        geometry_id: str,
        topology_id: str,
        topology_epoch: int,
        time: ArrayLike,
        step_index: ArrayLike,
        fields: Sequence[Any],
        /,
    ):
        formulation_ = _formulation(formulation)
        runtime, geometry, topology = tuple(
            str(value).strip() for value in (runtime_id, geometry_id, topology_id)
        )
        epoch = int(topology_epoch)
        time_ = jnp.asarray(time)
        step = jnp.asarray(step_index)
        values = tuple(jax.tree.map(jnp.asarray, value) for value in fields)
        names = _restart_field_names(formulation_)
        if not runtime or not geometry or not topology or epoch < 0:
            raise ValueError(
                "Restart runtime, geometry, topology, and epoch identities are invalid."
            )
        if (
            time_.shape != ()
            or not jnp.issubdtype(time_.dtype, jnp.inexact)
            or not bool(jnp.isfinite(time_))
        ):
            raise ValueError("Restart time must be one finite inexact scalar.")
        if step.shape != () or step.dtype.kind not in "iu" or bool(step < 0):
            raise ValueError("Restart step_index must be one nonnegative integer scalar.")
        if len(values) != len(names) or any(
            not jax.tree.leaves(value) for value in values
        ):
            raise ValueError(
                "Restart fields must exactly match the declared formulation."
            )
        self.time = time_
        self.step_index = step
        self.fields = values
        self.formulation = formulation_
        self.runtime_id = runtime
        self.geometry_id = geometry
        self.topology_id = topology
        self.topology_epoch = epoch
        self.field_names = names
        self.state_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-restart-state",
                "formulation": formulation_,
                "runtime": runtime,
                "geometry": geometry,
                "topology": topology,
                "topology_epoch": epoch,
                "field_names": list(names),
                "time": array_tree_fingerprint(time_),
                "step_index": array_tree_fingerprint(step),
                "fields": array_tree_fingerprint(values),
            }
        )

    @classmethod
    def from_z4c(
        cls,
        state: Z4cRuntimeState,
        /,
        *,
        topology_id: str,
        topology_epoch: int,
    ) -> NumericalRelativityRestartState:
        if not isinstance(state, Z4cRuntimeState):
            raise TypeError("state must be Z4cRuntimeState.")
        return cls(
            "z4c",
            state.runtime_id,
            state.state.grid_id,
            topology_id,
            topology_epoch,
            state.time,
            state.step_index,
            (state,),
        )

    @classmethod
    def from_grhd(
        cls,
        state: GRHDFiniteVolumeState,
        /,
        *,
        topology_epoch: int,
    ) -> NumericalRelativityRestartState:
        if not isinstance(state, GRHDFiniteVolumeState):
            raise TypeError("state must be GRHDFiniteVolumeState.")
        return cls(
            "grhd",
            state.runtime_id,
            state.content.geometry_family_id,
            state.content.topology_epoch_id,
            topology_epoch,
            state.time,
            state.accepted_step,
            (state,),
        )

    @classmethod
    def from_grmhd(
        cls,
        state: GRMHDState,
        /,
        *,
        runtime_id: str,
        geometry_id: str,
        topology_id: str,
        topology_epoch: int,
    ) -> NumericalRelativityRestartState:
        if not isinstance(state, GRMHDState):
            raise TypeError("state must be GRMHDState.")
        return cls(
            "grmhd",
            runtime_id,
            geometry_id,
            topology_id,
            topology_epoch,
            state.time,
            state.accepted_step,
            (
                state.material_state,
                state.constrained_transport,
                state.step_size,
                state.status,
            ),
        )

    @classmethod
    def from_grrmhd(
        cls,
        state: GRRMHDState,
        /,
        *,
        runtime_id: str,
        geometry_id: str,
        topology_id: str,
        topology_epoch: int,
    ) -> NumericalRelativityRestartState:
        if not isinstance(state, GRRMHDState):
            raise TypeError("state must be GRRMHDState.")
        return cls(
            "grrmhd",
            runtime_id,
            geometry_id,
            topology_id,
            topology_epoch,
            state.time,
            state.accepted_step,
            (
                state.material_state,
                state.constrained_transport,
                state.radiation_state,
                state.step_size,
                state.status,
            ),
        )

    @classmethod
    def from_coupled(
        cls,
        formulation: Literal["z4c-grhd", "z4c-grmhd", "z4c-grrmhd"],
        state: CoupledEvolutionState,
        /,
        *,
        geometry_id: str,
        topology_epoch: int,
    ) -> NumericalRelativityRestartState:
        formulation_ = _formulation(formulation)
        if formulation_ not in ("z4c-grhd", "z4c-grmhd", "z4c-grrmhd"):
            raise ValueError("Coupled restart formulation must include Z4c and matter.")
        if not isinstance(state, CoupledEvolutionState):
            raise TypeError("state must be CoupledEvolutionState.")
        return cls(
            formulation_,
            state.runtime_id,
            geometry_id,
            state.topology_id,
            topology_epoch,
            state.time,
            state.accepted_steps,
            (
                state.z4c,
                state.matter,
                state.budget,
                state.accepted_steps,
                state.rejected_steps,
                state.consecutive_failures,
                state.terminal,
                state.next_step_id,
            ),
        )

    def field(self, name: str, /) -> Any:
        identifier = str(name)
        if identifier not in self.field_names:
            raise KeyError(f"Restart state has no field {identifier!r}.")
        return self.fields[self.field_names.index(identifier)]


def _restart_state_signature(
    state: NumericalRelativityRestartState, /
) -> tuple[str, tuple[tuple[tuple[int, ...], str], ...]]:
    leaves, structure = jax.tree.flatten((state.time, state.step_index, state.fields))
    structure_record = str(structure)
    signatures = tuple((tuple(leaf.shape), np.dtype(leaf.dtype).str) for leaf in leaves)
    if len(structure_record.encode("utf-8")) > 65_536 or len(signatures) > 4096:
        raise ValueError("Restart PyTree reconstruction metadata exceeds fixed bounds.")
    return structure_record, signatures


class NumericalRelativityCheckpointPlan(StrictModule, NonTrainableState):
    """Canonical local/distributed checkpoint contract bound to one topology ID."""

    formulation: NumericalRelativityFormulation = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    analysis_plan_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    execution_plan_id: str = eqx.field(static=True)
    topology_epoch: int = eqx.field(static=True)
    field_names: tuple[str, ...] = eqx.field(static=True)
    state_structure: str = eqx.field(static=True)
    state_leaf_signatures: tuple[tuple[tuple[int, ...], str], ...] = eqx.field(
        static=True
    )
    restart: NumericalRelativityRestartPolicy
    constrained_transport: GRMHDConstrainedTransportPlan | None
    coupled: CoupledFieldCheckpointPlan
    plan_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        formulation: NumericalRelativityFormulation,
        runtime_id: str,
        geometry_id: str,
        topology_id: str,
        /,
        *,
        analysis_plan_id: str,
        numeric_revision_id: str,
        execution_plan_id: str,
        topology_epoch: int,
        state_template: NumericalRelativityRestartState,
        constrained_transport: GRMHDConstrainedTransportPlan | None = None,
        restart: NumericalRelativityRestartPolicy | None = None,
    ):
        formulation_ = _formulation(formulation)
        identifiers = tuple(
            str(value).strip()
            for value in (
                runtime_id,
                geometry_id,
                topology_id,
                analysis_plan_id,
                numeric_revision_id,
                execution_plan_id,
            )
        )
        epoch = int(topology_epoch)
        policy = NumericalRelativityRestartPolicy() if restart is None else restart
        if any(not value for value in identifiers) or epoch < 0:
            raise ValueError("NR checkpoint identities and topology epoch must be valid.")
        if not isinstance(policy, NumericalRelativityRestartPolicy):
            raise TypeError("restart must be a NumericalRelativityRestartPolicy.")
        if not isinstance(state_template, NumericalRelativityRestartState):
            raise TypeError("state_template must be NumericalRelativityRestartState.")
        runtime, geometry, topology, analysis, revision, execution = identifiers
        uses_grmhd = formulation_ in (
            "grmhd",
            "grrmhd",
            "z4c-grmhd",
            "z4c-grrmhd",
        )
        if uses_grmhd and not isinstance(
            constrained_transport, GRMHDConstrainedTransportPlan
        ):
            raise TypeError(
                "GRMHD checkpoints require their exact constrained-transport plan."
            )
        if not uses_grmhd and constrained_transport is not None:
            raise ValueError(
                "A constrained-transport plan is valid only for a GRMHD checkpoint."
            )
        names = _restart_field_names(formulation_)
        structure, leaf_signatures = _restart_state_signature(state_template)
        coupled = CoupledFieldCheckpointPlan(
            runtime,
            f"numerical-relativity:{formulation_}",
            names,
            geometry_id=geometry,
            topology_id=topology,
        )
        self.formulation = formulation_
        self.runtime_id = runtime
        self.geometry_id = geometry
        self.topology_id = topology
        self.analysis_plan_id = analysis
        self.numeric_revision_id = revision
        self.execution_plan_id = execution
        self.topology_epoch = epoch
        self.field_names = names
        self.state_structure = structure
        self.state_leaf_signatures = leaf_signatures
        self.restart = policy
        self.constrained_transport = constrained_transport
        self.coupled = coupled
        self.plan_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-checkpoint-plan",
                "formulation": formulation_,
                "runtime": runtime,
                "geometry": geometry,
                "topology": topology,
                "analysis": analysis,
                "numeric_revision": revision,
                "execution": execution,
                "topology_epoch": epoch,
                "restart": policy.policy_id,
                "state_structure": structure,
                "state_leaf_signatures": leaf_signatures,
                "constrained_transport": (
                    None
                    if constrained_transport is None
                    else {
                        "plan": constrained_transport.plan_id,
                        "layout": constrained_transport.layout.layout_id,
                        "gauge": constrained_transport.gauge.gauge_id,
                    }
                ),
                "coupled": coupled.plan_id,
            }
        )
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-checkpoint",
                "plan": self.plan_id,
                "topology": topology,
                "topology_epoch": epoch,
            }
        )
        self.validate_state(state_template)

    def validate_state(self, state: NumericalRelativityRestartState, /) -> None:
        if not isinstance(state, NumericalRelativityRestartState):
            raise TypeError("Checkpoint state must be NumericalRelativityRestartState.")
        if (
            state.formulation != self.formulation
            or state.runtime_id != self.runtime_id
            or state.geometry_id != self.geometry_id
            or state.topology_id != self.topology_id
            or state.topology_epoch != self.topology_epoch
            or state.field_names != self.field_names
        ):
            raise ValueError(
                "Checkpoint state runtime/formulation/geometry/topology/epoch does not match its plan."
            )
        if not all(
            bool(jnp.all(jnp.isfinite(leaf)))
            for leaf in jax.tree.leaves((state.time, state.step_index, state.fields))
        ):
            raise ValueError(
                "Checkpoint state must contain only finite accepted content."
            )
        if _restart_state_signature(state) != (
            self.state_structure,
            self.state_leaf_signatures,
        ):
            raise ValueError(
                "Checkpoint state PyTree, shapes, or dtypes do not match its plan."
            )
        if self.formulation == "z4c":
            z4c_runtime = state.field("z4c_runtime_state")
            if not isinstance(z4c_runtime, Z4cRuntimeState):
                raise TypeError("Z4c restart field must be Z4cRuntimeState.")
            if (
                z4c_runtime.runtime_id != self.runtime_id
                or z4c_runtime.state.grid_id != self.geometry_id
                or not bool(z4c_runtime.time == state.time)
                or not bool(z4c_runtime.step_index == state.step_index)
            ):
                raise ValueError(
                    "Z4c restart runtime, grid, time, or step identity is inconsistent."
                )
        if self.formulation in ("z4c-grhd", "z4c-grmhd", "z4c-grrmhd"):
            if not isinstance(state.field("z4c"), Z4cState):
                raise TypeError("Coupled restart z4c field must be Z4cState.")
            if state.field("z4c").grid_id != self.geometry_id:
                raise ValueError(
                    "Coupled Z4c restart grid does not match checkpoint geometry."
                )
            matter = state.field("matter")
            expected_matter_type = (
                GRHDFiniteVolumeState
                if self.formulation == "z4c-grhd"
                else GRMHDState
                if self.formulation == "z4c-grmhd"
                else GRRMHDState
            )
            if not isinstance(matter, expected_matter_type):
                raise TypeError(
                    "Coupled restart matter field does not match its formulation."
                )
            if not bool(jnp.asarray(matter.time) == state.time):
                raise ValueError(
                    "Coupled matter and macro restart times must agree exactly."
                )
            if not isinstance(state.field("coupled_budget"), CoupledBudget):
                raise TypeError("Coupled restart budget must be CoupledBudget.")
            accepted_steps = jnp.asarray(state.field("accepted_steps"))
            rejected_steps = jnp.asarray(state.field("rejected_steps"))
            failures = jnp.asarray(state.field("consecutive_failures"))
            terminal = jnp.asarray(state.field("terminal"))
            next_step = jnp.asarray(state.field("next_step_id"))
            if (
                any(
                    value.shape != () or value.dtype.kind not in "iu"
                    for value in (
                        accepted_steps,
                        rejected_steps,
                        failures,
                        next_step,
                    )
                )
                or terminal.shape != ()
                or terminal.dtype != jnp.dtype(jnp.bool_)
                or bool(accepted_steps < 0)
                or bool(rejected_steps < 0)
                or bool(failures < 0)
                or bool(next_step < 0)
                or bool(accepted_steps != state.step_index)
            ):
                raise ValueError(
                    "Coupled restart counters, terminal flag, or accepted-step identity are invalid."
                )
        if self.formulation == "grhd":
            grhd = state.field("grhd_state")
            if not isinstance(grhd, GRHDFiniteVolumeState):
                raise TypeError("GRHD restart field must be GRHDFiniteVolumeState.")
            if (
                grhd.runtime_id != self.runtime_id
                or grhd.content.geometry_family_id != self.geometry_id
                or grhd.content.topology_epoch_id != self.topology_id
                or not bool(grhd.time == state.time)
                or not bool(grhd.accepted_step == state.step_index)
            ):
                raise ValueError(
                    "GRHD restart runtime, grid, time, or step identity is inconsistent."
                )
        if self.constrained_transport is not None:
            if self.formulation in ("grmhd", "grrmhd"):
                material = state.field("material")
                transport = state.field("constrained_transport")
                if not isinstance(transport, GRMHDCTState):
                    raise TypeError(
                        "GRMHD restart constrained_transport must be GRMHDCTState."
                    )
                expected = self.constrained_transport.cell_shape + (
                    self.constrained_transport.layout.reduced_component_count,
                )
                if jnp.asarray(material).shape != expected:
                    raise ValueError(
                        "GRMHD restart material state does not match the CT layout."
                    )
                if self.formulation == "grrmhd":
                    radiation = jnp.asarray(state.field("radiation"))
                    if radiation.shape != self.constrained_transport.cell_shape + (4,):
                        raise ValueError(
                            "GRRMHD restart radiation state does not match the CT grid."
                        )
            else:
                matter = state.field("matter")
                expected_type = (
                    GRRMHDState if self.formulation == "z4c-grrmhd" else GRMHDState
                )
                if not isinstance(matter, expected_type):
                    raise TypeError(
                        "Coupled Z4c restart matter has the wrong GRMHD type."
                    )
                transport = matter.constrained_transport
            magnetic = self.constrained_transport.validate_magnetic_flux(
                transport.magnetic_flux
            )
            potential = self.constrained_transport.validate_vector_potential(
                transport.vector_potential
            )
            scalar = self.constrained_transport.validate_gauge_scalar(
                transport.gauge_scalar
            )
            finite = bool(
                jnp.all(jnp.isfinite(magnetic))
                & jnp.all(jnp.isfinite(potential))
                & jnp.all(jnp.isfinite(scalar))
            )
            divergence = self.constrained_transport.magnetic_divergence(magnetic)
            divergence_valid = bool(
                jnp.max(jnp.abs(divergence), initial=0.0)
                <= self.constrained_transport.divergence_tolerance
            )
            curl_valid = True
            if self.constrained_transport.gauge.evolves_vector_potential:
                from_potential = (
                    self.constrained_transport.magnetic_from_vector_potential(potential)
                )
                curl_valid = bool(
                    jnp.max(jnp.abs(magnetic - from_potential), initial=0.0)
                    <= self.constrained_transport.compatibility_tolerance
                )
            if not finite or not divergence_valid or not curl_valid:
                raise ValueError(
                    "GRMHD checkpoint is nonfinite, divergence-invalid, or inconsistent with dA."
                )


class NumericalRelativityCheckpoint(StrictModule):
    state: NumericalRelativityRestartState
    runtime_args: Any
    plan_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)
    content_id: str = eqx.field(static=True)

    def __init__(
        self,
        state: NumericalRelativityRestartState,
        runtime_args: Any,
        plan_id: str,
        checkpoint_id: str,
        /,
    ):
        if not isinstance(state, NumericalRelativityRestartState):
            raise TypeError("NR checkpoint requires a restart state.")
        plan = str(plan_id)
        checkpoint = str(checkpoint_id)
        if not plan or not checkpoint:
            raise ValueError("NR checkpoint identities must be non-empty.")
        self.state = state
        self.runtime_args = runtime_args
        self.plan_id = plan
        self.checkpoint_id = checkpoint
        self.content_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-checkpoint-content",
                "plan": plan,
                "checkpoint": checkpoint,
                "state": state.state_id,
                "runtime_args": array_tree_fingerprint(runtime_args),
            }
        )


class NumericalRelativityRestartEvidence(StrictModule):
    maximum_absolute_error: Array
    maximum_relative_error: Array
    finite: Array
    topology_matches: Array
    structure_matches: Array
    exact: Array
    within_tolerance: Array
    qualified: Array
    derivative_valid: Array
    restart_admitted: Array
    relation: RestartRelation = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    topology_relation_id: str = eqx.field(static=True)
    restart_admission_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def _physical_restart_arrays(
    state: NumericalRelativityRestartState, /
) -> tuple[Array, ...]:
    if state.formulation == "z4c":
        return (state.field("z4c_runtime_state").state.values,)
    if state.formulation == "grhd":
        return (state.field("grhd_state").content.conservative_content,)
    if state.formulation == "grmhd":
        transport = state.field("constrained_transport")
        return (
            state.field("material"),
            transport.magnetic_flux,
            transport.vector_potential,
            transport.gauge_scalar,
        )
    z4c = state.field("z4c")
    matter = state.field("matter")
    if state.formulation == "z4c-grhd":
        matter_arrays = (matter.content.conservative_content,)
    else:
        matter_arrays = (
            matter.material_state,
            matter.constrained_transport.magnetic_flux,
            matter.constrained_transport.vector_potential,
            matter.constrained_transport.gauge_scalar,
        )
    return (z4c.values, *matter_arrays)


def _restart_arrays(
    state: NumericalRelativityRestartState, /
) -> tuple[tuple[tuple[str, Array, bool], ...], Any]:
    tree = (state.time, state.step_index, state.fields)
    flattened, structure = jax.tree_util.tree_flatten_with_path(tree)
    physical_ids = {id(value) for value in _physical_restart_arrays(state)}
    records = tuple(
        (
            jax.tree_util.keystr(path) or "<root>",
            jnp.asarray(value),
            id(value) in physical_ids,
        )
        for path, value in flattened
    )
    return records, structure


def evaluate_numerical_relativity_restart(
    reference: NumericalRelativityRestartState,
    restarted: NumericalRelativityRestartState,
    policy: NumericalRelativityRestartPolicy,
    /,
) -> NumericalRelativityRestartEvidence:
    """Evaluate exact and tolerance relations without conflating either result."""

    if not isinstance(reference, NumericalRelativityRestartState) or not isinstance(
        restarted, NumericalRelativityRestartState
    ):
        raise TypeError("Restart comparison requires two NR restart states.")
    if not isinstance(policy, NumericalRelativityRestartPolicy):
        raise TypeError("policy must be NumericalRelativityRestartPolicy.")
    source_support = canonical_fingerprint(
        {
            "geometry": reference.geometry_id,
            "topology": reference.topology_id,
            "epoch": reference.topology_epoch,
        }
    )
    target_support = canonical_fingerprint(
        {
            "geometry": restarted.geometry_id,
            "topology": restarted.topology_id,
            "epoch": restarted.topology_epoch,
        }
    )
    topology_relation = policy.topology_relation(source_support, target_support)
    restart_admission = admit_topology_restart(topology_relation, policy.topology_policy)
    topology_matches = (
        reference.formulation == restarted.formulation
        and reference.runtime_id == restarted.runtime_id
        and reference.geometry_id == restarted.geometry_id
        and reference.topology_id == restarted.topology_id
        and reference.topology_epoch == restarted.topology_epoch
        and reference.field_names == restarted.field_names
    )
    left, left_structure = _restart_arrays(reference)
    right, right_structure = _restart_arrays(restarted)
    structure_matches = (
        left_structure == right_structure
        and len(left) == len(right)
        and all(
            first_path == second_path
            and first.shape == second.shape
            and first.dtype == second.dtype
            and first_physical == second_physical
            for (first_path, first, first_physical), (
                second_path,
                second,
                second_physical,
            ) in zip(left, right, strict=True)
        )
    )
    finite = all(bool(jnp.all(jnp.isfinite(value))) for _, value, _ in (*left, *right))
    payload_exact = structure_matches and all(
        np.array_equal(np.asarray(first), np.asarray(second))
        for (_, first, _), (_, second, _) in zip(left, right, strict=True)
    )
    within = structure_matches
    maximum_absolute = jnp.asarray(0.0)
    maximum_relative = jnp.asarray(0.0)
    if structure_matches:
        for (_, first, physical), (_, second, _) in zip(left, right, strict=True):
            if physical:
                difference = jnp.abs(first - second)
                absolute = jnp.max(difference, initial=0.0)
                denominator = jnp.maximum(
                    jnp.abs(first),
                    jnp.asarray(jnp.finfo(first.dtype).tiny, dtype=first.dtype),
                )
                relative = jnp.max(difference / denominator, initial=0.0)
                maximum_absolute = jnp.maximum(maximum_absolute, absolute)
                maximum_relative = jnp.maximum(maximum_relative, relative)
                within = within and bool(
                    jnp.allclose(
                        first,
                        second,
                        rtol=policy.relative_tolerance,
                        atol=policy.absolute_tolerance,
                    )
                )
            else:
                within = within and np.array_equal(np.asarray(first), np.asarray(second))
    else:
        maximum_absolute = jnp.asarray(jnp.inf)
        maximum_relative = jnp.asarray(jnp.inf)
        within = False
    exact = topology_matches and payload_exact
    within = topology_matches and within
    relation_satisfied = exact if policy.relation == "exact" else within
    qualified = finite and relation_satisfied and restart_admission.admitted
    evidence_id = canonical_fingerprint(
        {
            "kind": "numerical-relativity-restart-evidence",
            "policy": policy.policy_id,
            "reference": reference.state_id,
            "restarted": restarted.state_id,
            "topology_matches": topology_matches,
            "structure_matches": structure_matches,
            "exact": exact,
            "within_tolerance": within,
            "topology_relation": topology_relation.relation_id,
            "restart_admission": restart_admission.admission_id,
        }
    )
    return NumericalRelativityRestartEvidence(
        maximum_absolute,
        maximum_relative,
        jnp.asarray(finite),
        jnp.asarray(topology_matches),
        jnp.asarray(structure_matches),
        jnp.asarray(exact),
        jnp.asarray(within),
        jnp.asarray(qualified),
        jnp.asarray(False),
        jnp.asarray(restart_admission.admitted),
        policy.relation,
        policy.policy_id,
        topology_relation.relation_id,
        restart_admission.admission_id,
        evidence_id,
    )


def write_numerical_relativity_checkpoint(
    path: str | Path,
    plan: NumericalRelativityCheckpointPlan,
    state: NumericalRelativityRestartState,
    /,
    *,
    runtime_args: Any = None,
) -> NumericalRelativityCheckpoint:
    """Atomically write one topology-bound, pickle-free local restart archive."""

    if not isinstance(plan, NumericalRelativityCheckpointPlan):
        raise TypeError("plan must be NumericalRelativityCheckpointPlan.")
    plan.validate_state(state)
    write_coupled_field_checkpoint(
        path,
        plan.coupled,
        state.time,
        state.step_index,
        state.fields,
        runtime_args=runtime_args,
    )
    return NumericalRelativityCheckpoint(
        state, runtime_args, plan.plan_id, plan.checkpoint_id
    )


def read_numerical_relativity_checkpoint(
    path: str | Path,
    plan: NumericalRelativityCheckpointPlan,
    template: NumericalRelativityRestartState,
    /,
    *,
    runtime_args_template: Any = None,
) -> NumericalRelativityCheckpoint:
    """Read only an archive whose runtime, geometry, and topology IDs all agree."""

    if not isinstance(plan, NumericalRelativityCheckpointPlan):
        raise TypeError("plan must be NumericalRelativityCheckpointPlan.")
    plan.validate_state(template)
    restored = read_coupled_field_checkpoint(
        path,
        plan.coupled,
        template.fields,
        runtime_args_template=runtime_args_template,
    )
    state = NumericalRelativityRestartState(
        plan.formulation,
        plan.runtime_id,
        plan.geometry_id,
        plan.topology_id,
        plan.topology_epoch,
        restored.time,
        restored.step_index,
        restored.state,
    )
    plan.validate_state(state)
    return NumericalRelativityCheckpoint(
        state,
        restored.runtime_args,
        plan.plan_id,
        plan.checkpoint_id,
    )


_MAX_RECONSTRUCTION_METADATA_BYTES = 65_536
_MAX_RESTART_LEAVES = 4096
_MAX_RESTART_SHARDS = 262_144


class DistributedNumericalRelativityRestart(StrictModule):
    checkpoint: NumericalRelativityCheckpoint
    evidence: NumericalRelativityRestartEvidence
    manifest_id: str = eqx.field(static=True)
    reconstruction_metadata_id: str = eqx.field(static=True)


def _distributed_base_tree(
    state: NumericalRelativityRestartState,
    runtime_args: Any,
    /,
) -> dict[str, Any]:
    args = None if runtime_args is None else jax.tree.map(jnp.asarray, runtime_args)
    return {
        "fields": state.fields,
        "runtime_args": args,
        "step_index": state.step_index,
        "time": state.time,
    }


def _array_tree_records(tree: Any, /) -> tuple[dict[str, Any], ...]:
    flattened = jax.tree_util.tree_flatten_with_path(tree)[0]
    if len(flattened) > _MAX_RESTART_LEAVES:
        raise ValueError("Distributed restart exceeds the fixed leaf-count bound.")
    records = []
    for path, leaf in flattened:
        if not isinstance(leaf, jax.Array):
            raise TypeError("Distributed restart PyTrees may contain only JAX arrays.")
        records.append(
            {
                "path": jax.tree_util.keystr(path) or "<root>",
                "shape": [size for size in leaf.shape],
                "dtype": np.dtype(leaf.dtype).str,
            }
        )
    return tuple(records)


def _reconstruction_metadata(
    plan: NumericalRelativityCheckpointPlan,
    state: NumericalRelativityRestartState,
    runtime_args: Any,
    /,
) -> bytes:
    base = _distributed_base_tree(state, runtime_args)
    structure = str(jax.tree.structure(base))
    record = {
        "kind": "numerical-relativity-distributed-reconstruction",
        "plan_id": plan.plan_id,
        "checkpoint_id": plan.checkpoint_id,
        "state_id": state.state_id,
        "runtime_args_id": array_tree_fingerprint(base["runtime_args"]),
        "runtime_id": state.runtime_id,
        "geometry_id": state.geometry_id,
        "topology_id": state.topology_id,
        "topology_epoch": state.topology_epoch,
        "formulation": state.formulation,
        "field_names": list(state.field_names),
        "state_structure": plan.state_structure,
        "state_leaf_signatures": [
            [list(shape), dtype] for shape, dtype in plan.state_leaf_signatures
        ],
        "payload_structure": structure,
        "arrays": list(_array_tree_records(base)),
    }
    content = canonical_json(record).encode("utf-8")
    envelope = canonical_json(
        {
            "record": record,
            "record_sha256": hashlib.sha256(content).hexdigest(),
        }
    ).encode("utf-8")
    if len(envelope) > _MAX_RECONSTRUCTION_METADATA_BYTES:
        raise ValueError("Distributed restart reconstruction metadata is too large.")
    return envelope


def _distributed_tree(
    plan: NumericalRelativityCheckpointPlan,
    state: NumericalRelativityRestartState,
    runtime_args: Any,
    /,
) -> tuple[dict[str, Any], bytes]:
    metadata = _reconstruction_metadata(plan, state, runtime_args)
    tree = {
        **_distributed_base_tree(state, runtime_args),
        "reconstruction_metadata": jnp.asarray(
            np.frombuffer(metadata, dtype=np.uint8).copy()
        ),
    }
    return tree, metadata


def _validate_global_manifest(
    manifest: CheckpointManifest,
    plan: NumericalRelativityCheckpointPlan,
    /,
) -> None:
    if not isinstance(manifest, CheckpointManifest) or not manifest.complete:
        raise ValueError(
            "Distributed NR restore requires a complete checkpoint manifest."
        )
    if (
        manifest.checkpoint_id != plan.checkpoint_id
        or manifest.analysis_plan_id != plan.analysis_plan_id
        or manifest.numeric_revision_id != plan.numeric_revision_id
        or manifest.execution_plan_id != plan.execution_plan_id
        or len(manifest.shards) > _MAX_RESTART_SHARDS
    ):
        raise ValueError(
            "Distributed NR checkpoint identities or shard bounds do not match the plan."
        )


def _validate_parent_manifest(
    repository: ArtifactRepository,
    parent_manifest: CheckpointManifest | None,
    plan: NumericalRelativityCheckpointPlan,
    /,
) -> None:
    if parent_manifest is None:
        return
    if (
        not isinstance(parent_manifest, CheckpointManifest)
        or not parent_manifest.complete
    ):
        raise TypeError("parent_manifest must be a complete CheckpointManifest or None.")
    if parent_manifest.checkpoint_id == plan.checkpoint_id:
        raise ValueError("A numerical-relativity checkpoint cannot parent itself.")
    if (
        parent_manifest.analysis_plan_id != plan.analysis_plan_id
        or parent_manifest.numeric_revision_id != plan.numeric_revision_id
        or parent_manifest.execution_plan_id != plan.execution_plan_id
    ):
        raise ValueError("Parent checkpoint has incompatible NR lifecycle ownership.")
    processes: set[int] = set()
    for shard in parent_manifest.shards:
        metadata = dict(shard.metadata)
        process_record = metadata.get("process_index")
        artifact_id = metadata.get("artifact_id")
        if process_record is None or artifact_id is None:
            raise ValueError("Parent checkpoint shard lacks repository ownership.")
        if (
            metadata.get("repository_id") != repository.provider_id
            or metadata.get("checkpoint_id") != parent_manifest.checkpoint_id
            or metadata.get("analysis_plan_id") != parent_manifest.analysis_plan_id
            or metadata.get("numeric_revision_id") != parent_manifest.numeric_revision_id
            or metadata.get("execution_plan_id") != parent_manifest.execution_plan_id
        ):
            raise ValueError("Parent checkpoint shard ownership is invalid.")
        process_index = int(process_record)
        if artifact_id != (f"{parent_manifest.checkpoint_id}.process-{process_index}"):
            raise ValueError("Parent checkpoint shard artifact identity is invalid.")
        durable = repository.get_manifest(artifact_id)
        durable_metadata = dict(durable.metadata)
        if (
            durable.provider_id != repository.provider_id
            or durable.artifact_id != artifact_id
            or durable_metadata.get("repository_id") != repository.provider_id
            or durable_metadata.get("checkpoint_id") != parent_manifest.checkpoint_id
            or durable_metadata.get("execution_plan_id")
            != parent_manifest.execution_plan_id
            or durable_metadata.get("analysis_plan_id")
            != parent_manifest.analysis_plan_id
            or durable_metadata.get("numeric_revision_id")
            != parent_manifest.numeric_revision_id
            or durable_metadata.get("process_index") != process_record
            or not durable.complete
        ):
            raise ValueError("Parent checkpoint repository ownership is invalid.")
        processes.add(process_index)
    if processes != set(range(len(processes))):
        raise ValueError("Parent checkpoint process coverage is not contiguous.")


def _validate_manifest_parent(
    manifest: CheckpointManifest,
    parent_manifest: CheckpointManifest | None,
    /,
) -> None:
    expected = (
        (None, None)
        if parent_manifest is None
        else (parent_manifest.checkpoint_id, parent_manifest.manifest_id)
    )
    if (manifest.parent_checkpoint_id, manifest.parent_manifest_id) != expected:
        raise ValueError("Checkpoint manifest does not bind the exact parent manifest.")


def publish_distributed_numerical_relativity_checkpoint(
    repository: ArtifactRepository,
    plan: NumericalRelativityCheckpointPlan,
    state: NumericalRelativityRestartState,
    /,
    *,
    writer_id: str,
    runtime_args: Any = None,
    attempt_id: str | None = None,
    encoding: ChunkEncoding = "identity",
    parent_manifest: CheckpointManifest | None = None,
) -> ProcessCheckpointPublication:
    """Publish typed state, runtime arguments, and bounded reconstruction metadata."""

    if not isinstance(plan, NumericalRelativityCheckpointPlan):
        raise TypeError("plan must be NumericalRelativityCheckpointPlan.")
    plan.validate_state(state)
    _validate_parent_manifest(repository, parent_manifest, plan)
    tree, _ = _distributed_tree(plan, state, runtime_args)
    publication = publish_process_checkpoint(
        repository,
        plan.checkpoint_id,
        plan.execution_plan_id,
        tree,
        analysis_plan_id=plan.analysis_plan_id,
        numeric_revision_id=plan.numeric_revision_id,
        writer_id=writer_id,
        attempt_id=attempt_id,
        topology_epoch=plan.topology_epoch,
        encoding=encoding,
        parent_manifest=parent_manifest,
    )
    artifact = publication.artifact_manifest
    if artifact is None or (
        artifact.provider_id != repository.provider_id
        or artifact.artifact_id
        != f"{plan.checkpoint_id}.process-{publication.process_index}"
        or not artifact.complete
    ):
        raise RuntimeError("Distributed checkpoint publication lacks its durable commit.")
    durable = repository.get_manifest(artifact.artifact_id)
    if durable.manifest_id != artifact.manifest_id:
        raise RuntimeError(
            "Distributed checkpoint publication is not repository-committed."
        )
    return publication


def _validate_publication(
    repository: ArtifactRepository,
    plan: NumericalRelativityCheckpointPlan,
    publication: ProcessCheckpointPublication,
    process_index: int,
    parent_manifest: CheckpointManifest | None,
    /,
) -> None:
    if (
        not isinstance(publication, ProcessCheckpointPublication)
        or publication.process_index != process_index
        or publication.artifact_manifest is None
    ):
        raise ValueError("Distributed checkpoint publication rank is unauthenticated.")
    artifact_id = f"{plan.checkpoint_id}.process-{process_index}"
    durable = repository.get_manifest(artifact_id)
    supplied = publication.artifact_manifest
    if (
        durable.provider_id != repository.provider_id
        or supplied.provider_id != repository.provider_id
        or durable.artifact_id != artifact_id
        or supplied.artifact_id != artifact_id
        or durable.manifest_id != supplied.manifest_id
        or durable.transaction_id != supplied.transaction_id
        or not durable.complete
    ):
        raise ValueError("Distributed checkpoint artifact/repository binding is invalid.")
    metadata = dict(durable.metadata)
    supplied_metadata = dict(supplied.metadata)
    parent_checkpoint_id = (
        "" if parent_manifest is None else parent_manifest.checkpoint_id
    )
    parent_manifest_id = "" if parent_manifest is None else parent_manifest.manifest_id
    expected = {
        "repository_id": repository.provider_id,
        "checkpoint_id": plan.checkpoint_id,
        "analysis_plan_id": plan.analysis_plan_id,
        "numeric_revision_id": plan.numeric_revision_id,
        "execution_plan_id": plan.execution_plan_id,
        "process_index": str(process_index),
        "topology_epoch": str(plan.topology_epoch),
        "parent_checkpoint_id": parent_checkpoint_id,
        "parent_manifest_id": parent_manifest_id,
        "shard_count": str(len(publication.shards)),
    }
    if any(
        metadata.get(key) != value or supplied_metadata.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError("Distributed checkpoint committed metadata is incompatible.")
    descriptor_payload = metadata["shards"]
    if len(descriptor_payload.encode("utf-8")) > _MAX_RECONSTRUCTION_METADATA_BYTES:
        raise ValueError("Distributed checkpoint shard descriptors exceed bounds.")
    descriptors = json.loads(descriptor_payload)
    if (
        not isinstance(descriptors, list)
        or len(descriptors) != len(publication.shards)
        or len(descriptors) > _MAX_RESTART_SHARDS
    ):
        raise ValueError("Distributed checkpoint shard descriptors are invalid.")
    for descriptor, shard in zip(descriptors, publication.shards, strict=True):
        shard_metadata = dict(shard.metadata)
        if (
            not isinstance(descriptor, dict)
            or descriptor.get("shard_id") != shard.shard_id
            or descriptor.get("payload_digest") != shard.payload_digest
            or int(descriptor.get("byte_count", -1)) != shard.byte_count
            or descriptor.get("layout_id") not in shard.layout_ids
            or any(
                str(descriptor.get(key)) != value for key, value in shard_metadata.items()
            )
            or shard_metadata.get("artifact_id") != artifact_id
            or shard_metadata.get("process_index") != str(process_index)
            or shard_metadata.get("topology_epoch") != str(plan.topology_epoch)
        ):
            raise ValueError("Distributed checkpoint shard bindings were substituted.")
    if any(chunk.transaction_id != durable.transaction_id for chunk in durable.chunks):
        raise ValueError("Distributed checkpoint chunks cross transaction boundaries.")


def assemble_distributed_numerical_relativity_checkpoint(
    repository: ArtifactRepository,
    plan: NumericalRelativityCheckpointPlan,
    publications: Sequence[ProcessCheckpointPublication],
    /,
    *,
    expected_process_count: int,
    parent_manifest: CheckpointManifest | None = None,
    diagnostic_ids: Sequence[str] = (),
) -> CheckpointManifest:
    """Assemble only exact committed repository publications for every rank."""

    if not isinstance(plan, NumericalRelativityCheckpointPlan):
        raise TypeError("plan must be NumericalRelativityCheckpointPlan.")
    _validate_parent_manifest(repository, parent_manifest, plan)
    values = tuple(publications)
    if len(values) != int(expected_process_count):
        raise ValueError("Distributed checkpoint publications do not cover every rank.")
    ordered = tuple(sorted(values, key=lambda value: value.process_index))
    for process_index, publication in enumerate(ordered):
        _validate_publication(
            repository,
            plan,
            publication,
            process_index,
            parent_manifest,
        )
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        plan.checkpoint_id,
        plan.analysis_plan_id,
        plan.numeric_revision_id,
        plan.execution_plan_id,
        expected_process_count=expected_process_count,
        parent_manifest=parent_manifest,
        diagnostic_ids=diagnostic_ids,
    )
    _validate_global_manifest(manifest, plan)
    _validate_manifest_parent(manifest, parent_manifest)
    for process_index, publication in enumerate(ordered):
        committed = tuple(
            shard.shard_id
            for shard in manifest.shards
            if dict(shard.metadata).get("process_index") == str(process_index)
        )
        if committed != tuple(shard.shard_id for shard in publication.shards):
            raise ValueError("Distributed checkpoint committed shards were substituted.")
    return manifest


def _validate_manifest_repository_binding(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    plan: NumericalRelativityCheckpointPlan,
    /,
) -> None:
    parent_checkpoint_id = manifest.parent_checkpoint_id or ""
    parent_manifest_id = manifest.parent_manifest_id or ""
    processes = set()
    durable_by_artifact: dict[str, ArtifactManifest] = {}
    descriptors_by_artifact: dict[str, dict[str, dict[str, Any]]] = {}
    for shard in manifest.shards:
        metadata = dict(shard.metadata)
        process_record = metadata.get("process_index")
        artifact_id = metadata.get("artifact_id")
        if process_record is None or artifact_id is None:
            raise ValueError(
                "Distributed checkpoint shard lacks process/artifact binding."
            )
        process_index = int(process_record)
        expected_artifact = f"{plan.checkpoint_id}.process-{process_index}"
        if process_index < 0 or artifact_id != expected_artifact:
            raise ValueError(
                "Distributed checkpoint shard rank/artifact binding is invalid."
            )
        processes.add(process_index)
        if artifact_id not in durable_by_artifact:
            durable = repository.get_manifest(artifact_id)
            durable_metadata = dict(durable.metadata)
            descriptor_payload = durable_metadata.get("shards", "")
            if (
                durable.provider_id != repository.provider_id
                or durable.artifact_id != artifact_id
                or durable_metadata.get("repository_id") != repository.provider_id
                or durable_metadata.get("checkpoint_id") != plan.checkpoint_id
                or durable_metadata.get("analysis_plan_id") != plan.analysis_plan_id
                or durable_metadata.get("numeric_revision_id") != plan.numeric_revision_id
                or durable_metadata.get("execution_plan_id") != plan.execution_plan_id
                or durable_metadata.get("process_index") != process_record
                or durable_metadata.get("topology_epoch") != str(plan.topology_epoch)
                or durable_metadata.get("parent_checkpoint_id") != parent_checkpoint_id
                or durable_metadata.get("parent_manifest_id") != parent_manifest_id
                or len(descriptor_payload.encode("utf-8"))
                > _MAX_RECONSTRUCTION_METADATA_BYTES
            ):
                raise ValueError(
                    "Distributed checkpoint repository artifact is incompatible."
                )
            descriptors = json.loads(descriptor_payload)
            if (
                not isinstance(descriptors, list)
                or len(descriptors) > _MAX_RESTART_SHARDS
            ):
                raise ValueError(
                    "Distributed checkpoint repository descriptors are invalid."
                )
            durable_by_artifact[artifact_id] = durable
            descriptors_by_artifact[artifact_id] = {
                str(value["shard_id"]): value for value in descriptors
            }
        descriptor = descriptors_by_artifact[artifact_id].get(shard.shard_id)
        if descriptor is None or (
            descriptor.get("payload_digest") != shard.payload_digest
            or int(descriptor.get("byte_count", -1)) != shard.byte_count
            or descriptor.get("layout_id") not in shard.layout_ids
            or metadata.get("logical_name") != descriptor.get("logical_name")
            or metadata.get("repository_id") != repository.provider_id
            or metadata.get("parent_checkpoint_id") != parent_checkpoint_id
            or metadata.get("checkpoint_id") != plan.checkpoint_id
            or metadata.get("analysis_plan_id") != plan.analysis_plan_id
            or metadata.get("numeric_revision_id") != plan.numeric_revision_id
            or metadata.get("execution_plan_id") != plan.execution_plan_id
            or metadata.get("parent_manifest_id") != parent_manifest_id
        ):
            raise ValueError("Distributed checkpoint manifest shard was substituted.")
    if processes != set(range(len(processes))):
        raise ValueError("Distributed checkpoint process coverage is not contiguous.")


def _manifest_inventory(
    manifest: CheckpointManifest, /
) -> dict[str, tuple[tuple[int, ...], str]]:
    inventory: dict[str, tuple[tuple[int, ...], str]] = {}
    for shard in manifest.shards:
        metadata = dict(shard.metadata)
        path = metadata.get("array_path")
        shape_record = metadata.get("global_shape")
        dtype = metadata.get("dtype")
        if (
            path is None
            or shape_record is None
            or dtype is None
            or len(path.encode("utf-8")) > 4096
            or len(shape_record.encode("utf-8")) > 4096
            or len(dtype.encode("utf-8")) > 128
        ):
            raise ValueError("Distributed checkpoint shard metadata exceeds bounds.")
        shape = tuple(json.loads(shape_record))
        record = (shape, dtype)
        if path in inventory and inventory[path] != record:
            raise ValueError("Distributed checkpoint array inventory is inconsistent.")
        inventory[path] = record
    return inventory


def _restore_tree(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    template: Any,
    /,
    *,
    parent_manifest: CheckpointManifest | None = None,
) -> Any:
    flattened, structure = jax.tree_util.tree_flatten_with_path(template)
    inventory = _manifest_inventory(manifest)
    expected_paths = tuple(
        jax.tree_util.keystr(path) or "<root>" for path, _ in flattened
    )
    if set(inventory) != set(expected_paths):
        raise ValueError("Distributed checkpoint array inventory changed.")
    leaves = []
    for path, leaf in flattened:
        array_path = jax.tree_util.keystr(path) or "<root>"
        expected = (tuple(leaf.shape), np.dtype(leaf.dtype).str)
        if inventory[array_path] != expected:
            raise ValueError("Distributed checkpoint array shape or dtype changed.")
        sharding = (
            leaf.sharding
            if isinstance(leaf, jax.Array)
            else SingleDeviceSharding(jax.devices()[0])
        )
        leaves.append(
            restore_global_array_from_checkpoint(
                repository,
                manifest,
                array_path,
                sharding,
                parent_manifest=parent_manifest,
            )
        )
    return jax.tree.unflatten(structure, leaves)


def restore_distributed_numerical_relativity_checkpoint(
    repository: ArtifactRepository,
    manifest: CheckpointManifest,
    plan: NumericalRelativityCheckpointPlan,
    template: NumericalRelativityRestartState,
    /,
    *,
    runtime_args_template: Any = None,
    parent_manifest: CheckpointManifest | None = None,
) -> DistributedNumericalRelativityRestart:
    """Reconstruct and validate the complete typed state directly from shards."""

    if not isinstance(plan, NumericalRelativityCheckpointPlan):
        raise TypeError("plan must be NumericalRelativityCheckpointPlan.")
    plan.validate_state(template)
    _validate_global_manifest(manifest, plan)
    _validate_parent_manifest(repository, parent_manifest, plan)
    _validate_manifest_parent(manifest, parent_manifest)
    _validate_manifest_repository_binding(repository, manifest, plan)
    template_tree, expected_metadata = _distributed_tree(
        plan, template, runtime_args_template
    )
    expected_envelope = json.loads(expected_metadata)
    expected_record = expected_envelope["record"]
    restored_tree = _restore_tree(
        repository,
        manifest,
        template_tree,
        parent_manifest=parent_manifest,
    )
    metadata_array = np.asarray(restored_tree["reconstruction_metadata"])
    observed_metadata = metadata_array.astype(np.uint8, copy=False).tobytes()
    if len(observed_metadata) > _MAX_RECONSTRUCTION_METADATA_BYTES:
        raise ValueError("Distributed checkpoint reconstruction metadata is invalid.")
    envelope = json.loads(observed_metadata)
    record = envelope["record"]
    content = canonical_json(record).encode("utf-8")
    if envelope.get("record_sha256") != hashlib.sha256(content).hexdigest():
        raise ValueError("Distributed checkpoint reconstruction checksum is invalid.")
    structural_keys = (
        "kind",
        "plan_id",
        "checkpoint_id",
        "runtime_id",
        "geometry_id",
        "topology_id",
        "topology_epoch",
        "formulation",
        "field_names",
        "state_structure",
        "state_leaf_signatures",
        "payload_structure",
        "arrays",
    )
    if any(record.get(key) != expected_record.get(key) for key in structural_keys):
        raise ValueError("Distributed checkpoint reconstruction schema is incompatible.")
    state = NumericalRelativityRestartState(
        plan.formulation,
        plan.runtime_id,
        plan.geometry_id,
        plan.topology_id,
        plan.topology_epoch,
        restored_tree["time"],
        restored_tree["step_index"],
        restored_tree["fields"],
    )
    plan.validate_state(state)
    runtime_args = restored_tree["runtime_args"]
    if record.get("state_id") != state.state_id or record.get(
        "runtime_args_id"
    ) != array_tree_fingerprint(runtime_args):
        raise ValueError(
            "Distributed checkpoint reconstructed content identity is invalid."
        )
    checkpoint = NumericalRelativityCheckpoint(
        state,
        runtime_args,
        plan.plan_id,
        plan.checkpoint_id,
    )
    topology_relation = plan.restart.topology_relation(
        canonical_fingerprint(
            {
                "geometry": state.geometry_id,
                "topology": state.topology_id,
                "epoch": state.topology_epoch,
            }
        ),
        canonical_fingerprint(
            {
                "geometry": state.geometry_id,
                "topology": state.topology_id,
                "epoch": state.topology_epoch,
            }
        ),
    )
    admission = admit_topology_restart(topology_relation, plan.restart.topology_policy)
    evidence = NumericalRelativityRestartEvidence(
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(admission.admitted),
        jnp.asarray(False),
        jnp.asarray(admission.admitted),
        plan.restart.relation,
        plan.restart.policy_id,
        topology_relation.relation_id,
        admission.admission_id,
        canonical_fingerprint(
            {
                "kind": "distributed-numerical-relativity-restart-evidence",
                "state": state.state_id,
                "metadata": hashlib.sha256(observed_metadata).hexdigest(),
                "admission": admission.admission_id,
            }
        ),
    )
    if not admission.admitted:
        raise ValueError("Distributed checkpoint failed restart admission.")
    return DistributedNumericalRelativityRestart(
        checkpoint,
        evidence,
        manifest.manifest_id,
        hashlib.sha256(observed_metadata).hexdigest(),
    )


__all__ = [
    "DistributedNumericalRelativityRestart",
    "NumericalRelativityCheckpoint",
    "NumericalRelativityCheckpointPlan",
    "NumericalRelativityRestartEvidence",
    "NumericalRelativityRestartPolicy",
    "NumericalRelativityRestartState",
    "RestartRelation",
    "assemble_distributed_numerical_relativity_checkpoint",
    "evaluate_numerical_relativity_restart",
    "publish_distributed_numerical_relativity_checkpoint",
    "read_numerical_relativity_checkpoint",
    "restore_distributed_numerical_relativity_checkpoint",
    "write_numerical_relativity_checkpoint",
]
