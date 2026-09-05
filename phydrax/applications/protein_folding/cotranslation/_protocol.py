#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ...._fingerprint import canonical_fingerprint
from ....atomistic._checkpoint import (
    AtomisticCheckpoint,
    AtomisticCheckpointPlan,
    read_atomistic_checkpoint,
    write_atomistic_checkpoint,
)
from ....atomistic._dynamics import AtomisticDynamicsState, PreparedAtomisticDynamics
from ....atomistic._topology_epoch import (
    activate_topology_epoch,
    InsertionLedger,
    TopologyEpochTransition,
)
from ....qualification._reference import ReferenceArtifactManifest
from ....series import SampledSeries, SeriesSupport
from .._construct import ProteinConstruct


# Standard nuclear genetic code, RNA alphabet, independently specified ordering.
_CODONS = tuple(a + b + c for a in "UCAG" for b in "UCAG" for c in "UCAG")
_AMINO_ACIDS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
_CODE = dict(zip(_CODONS, _AMINO_ACIDS, strict=True))


@dataclass(frozen=True)
class CotranslationStage:
    """One fixed-topology physical MD dwell, with insertion at its left boundary.

    ``codon`` is the RNA sense codon for the newly active residue; it is None
    only for a same-length protocol switch (for example tether release).
    Insertion rows follow stable capacity order of newly active particle IDs.
    The first stage is an explicitly supplied already-present nascent chain.
    """

    runtime: PreparedAtomisticDynamics
    nascent_residue_count: int
    dwell_steps: int
    codon: str | None
    source_id: str
    inserted_positions: tuple[tuple[float, float, float], ...] = ()
    inserted_momenta: tuple[tuple[float, float, float], ...] = ()
    maximum_absolute_work: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.dwell_steps, bool)
            or not isinstance(self.dwell_steps, int)
            or self.dwell_steps <= 0
        ):
            raise ValueError(
                "A codon/stage dwell requires a positive integer number of MD steps."
            )
        if (
            isinstance(self.nascent_residue_count, bool)
            or not isinstance(self.nascent_residue_count, int)
            or self.nascent_residue_count <= 0
        ):
            raise ValueError("A nascent stage requires a positive integer residue count.")
        if not self.source_id or self.source_id != self.source_id.strip():
            raise ValueError(
                "Every stage requires an explicit canonical source identity."
            )
        if self.codon is not None and (
            self.codon not in _CODE or _CODE[self.codon] == "*"
        ):
            raise ValueError(
                "Activation requires a standard-code RNA sense codon (no U/T conversion)."
            )
        for name, values in (
            ("inserted_positions", self.inserted_positions),
            ("inserted_momenta", self.inserted_momenta),
        ):
            rows = tuple(tuple(float(x) for x in row) for row in values)
            if any(len(row) != 3 or not all(np.isfinite(x) for x in row) for row in rows):
                raise ValueError("Insertion states require finite Cartesian triples.")
            object.__setattr__(self, name, rows)

    @property
    def dwell_time(self) -> float:
        """Dwell in the runtime time unit, not automatically biological time."""
        return self.dwell_steps * self.runtime.integrator.step_size


@dataclass(frozen=True)
class CotranslationCursor:
    protocol_id: str
    stage_index: int
    completed_steps: int
    state: AtomisticDynamicsState


@dataclass(frozen=True)
class CotranslationRun:
    cursor: CotranslationCursor
    segments: tuple[SampledSeries, ...]
    insertions: tuple[InsertionLedger, ...]
    successful: bool
    refusal: str | None


@eqx.filter_jit
def _step(runtime: PreparedAtomisticDynamics, state: AtomisticDynamicsState):
    return runtime.step_detailed(state)


@dataclass(frozen=True)
class CotranslationProtocol:
    """Caller-parameterized, one-bead-per-residue, single-chain insertion protocol.

    Environment particles remain active throughout. Biological timing is an
    explicit acceptance gate: supplied MD dwells alone do not calibrate codon
    kinetics. No molecular parameters, codon rates, coordinates or source
    permissions are guessed. Stage potentials may include the native elastic
    network/bonded/steric terms and ``RibosomeBoundaryPotential``.
    """

    construct: ProteinConstruct
    residue_particle_ids: tuple[int, ...]
    stages: tuple[CotranslationStage, ...]
    parameter_source: ReferenceArtifactManifest
    schedule_source: ReferenceArtifactManifest
    timing_calibration: ReferenceArtifactManifest | None = None
    timing_calibration_scope: str | None = None
    reference_conditioned: bool = True
    non_native_qualification: ReferenceArtifactManifest | None = None
    commercial_use: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "residue_particle_ids", tuple(self.residue_particle_ids))
        if len(self.construct.chain_ids) != 1:
            raise ValueError(
                "The cotranslation profile requires exactly one construct chain."
            )
        if (
            len(self.residue_particle_ids) != self.construct.residue_count
            or len(set(self.residue_particle_ids)) != self.construct.residue_count
        ):
            raise ValueError(
                "Each residue must bind one distinct stable coarse-particle ID."
            )
        if (
            not self.stages
            or self.stages[-1].nascent_residue_count != self.construct.residue_count
        ):
            raise ValueError(
                "A complete protocol must reach the full declared construct."
            )
        for source in (self.parameter_source, self.schedule_source):
            source.require_rights(commercial_use=self.commercial_use)
        if (self.timing_calibration is None) != (self.timing_calibration_scope is None):
            raise ValueError(
                "Biological timing calibration requires both source and stated scope."
            )
        if self.timing_calibration is not None:
            self.timing_calibration.require_rights(commercial_use=self.commercial_use)
            self.timing_calibration.require_uncertainty()
            if not self.timing_calibration_scope:
                raise ValueError("Timing calibration scope must not be empty.")
        if not self.reference_conditioned:
            if self.non_native_qualification is None:
                raise ValueError(
                    "A non-native folding claim requires independent model qualification."
                )
            self.non_native_qualification.require_rights(
                commercial_use=self.commercial_use
            )
            self.non_native_qualification.require_uncertainty()
        first = self.stages[0]
        all_ids = tuple(
            int(x) for x in np.asarray(first.runtime.system.plan.particle_ids)
        )
        if not set(self.residue_particle_ids) <= set(all_ids):
            raise ValueError("Residue binding contains IDs outside prepared capacity.")
        residue_set = set(self.residue_particle_ids)
        environment = (
            set(
                int(x)
                for x in np.asarray(first.runtime.system.plan.particle_ids)[
                    np.asarray(first.runtime.system.active_mask)
                ]
            )
            - residue_set
        )
        previous_count = first.nascent_residue_count
        for index, stage in enumerate(self.stages):
            count = stage.nascent_residue_count
            if not 0 < count <= self.construct.residue_count:
                raise ValueError("Nascent residue count exceeds the construct.")
            ids = tuple(
                int(x) for x in np.asarray(stage.runtime.system.plan.particle_ids)
            )
            if ids != all_ids:
                raise ValueError("Stages must preserve stable capacity ordering.")
            active = set(
                int(x)
                for x in np.asarray(stage.runtime.system.plan.particle_ids)[
                    np.asarray(stage.runtime.system.active_mask)
                ]
            )
            if active != environment | set(self.residue_particle_ids[:count]):
                raise ValueError(
                    "Material support must be exactly the nascent prefix plus environment."
                )
            if stage.runtime.system.cell is not None:
                raise ValueError(
                    "This cotranslation profile requires nonperiodic systems."
                )
            if (
                stage.runtime.system.topology.constraint_count
                and stage.runtime.constraints is None
            ):
                raise ValueError(
                    "Stage constraints require an actual prepared constraint executor."
                )
            if index == 0:
                if stage.inserted_positions or stage.inserted_momenta:
                    raise ValueError(
                        "The initial nascent state is supplied at initialization, not inserted twice."
                    )
            else:
                increment = count - previous_count
                if increment not in (0, 1):
                    raise ValueError(
                        "Each codon epoch activates one residue; protocol switches activate none."
                    )
                if (increment == 0) != (stage.codon is None):
                    raise ValueError(
                        "Same-length switches have no codon; insertion requires a codon."
                    )
                transition = self.transition(index)
                if len(stage.inserted_positions) != len(
                    transition.inserted_particle_ids
                ) or len(stage.inserted_momenta) != len(transition.inserted_particle_ids):
                    raise ValueError(
                        "Every inserted particle requires an explicit position and momentum."
                    )
            if (
                stage.codon is not None
                and _CODE[stage.codon] != self.construct.sequences[0][count - 1]
            ):
                raise ValueError("Codon does not encode the activated construct residue.")
            previous_count = count

    @property
    def protocol_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "cotranslation-protocol",
                "construct": self.construct.fingerprint(),
                "residue_particle_ids": self.residue_particle_ids,
                "parameters": self.parameter_source.manifest_id,
                "schedule": self.schedule_source.manifest_id,
                "timing": None
                if self.timing_calibration is None
                else self.timing_calibration.manifest_id,
                "timing_scope": self.timing_calibration_scope,
                "reference_conditioned": self.reference_conditioned,
                "non_native": None
                if self.non_native_qualification is None
                else self.non_native_qualification.manifest_id,
                "stages": [
                    {
                        "runtime": s.runtime.prepared_id,
                        "count": s.nascent_residue_count,
                        "steps": s.dwell_steps,
                        "codon": s.codon,
                        "source": s.source_id,
                        "positions": s.inserted_positions,
                        "momenta": s.inserted_momenta,
                        "work_bound": s.maximum_absolute_work,
                    }
                    for s in self.stages
                ],
            }
        )

    def transition(self, stage_index: int, /) -> TopologyEpochTransition:
        if not 1 <= stage_index < len(self.stages):
            raise ValueError("A transition requires a noninitial stage index.")
        stage = self.stages[stage_index]
        return TopologyEpochTransition(
            self.stages[stage_index - 1].runtime,
            stage.runtime,
            stage.source_id,
            stage.maximum_absolute_work,
        )

    def initialize(
        self, positions: ArrayLike, momenta: ArrayLike, /, *, key: Key[Array, ""]
    ) -> CotranslationCursor:
        runtime = self.stages[0].runtime
        state = runtime.initialize_state(positions, momentum=momenta, key=key)
        if np.any(
            np.asarray(state.force.forces)[~np.asarray(runtime.system.active_mask)] != 0
        ):
            raise ValueError("Initial potential exerts a force on dormant material.")
        return CotranslationCursor(self.protocol_id, 0, 0, state)

    def _validate_cursor(self, cursor: CotranslationCursor) -> None:
        if cursor.protocol_id != self.protocol_id or not 0 <= cursor.stage_index < len(
            self.stages
        ):
            raise ValueError("Cursor belongs to another cotranslation protocol.")
        stage = self.stages[cursor.stage_index]
        if cursor.state.prepared_dynamics_id != stage.runtime.prepared_id:
            raise ValueError("Cursor state belongs to another topology epoch.")
        if not 0 <= cursor.completed_steps <= stage.dwell_steps:
            raise ValueError("Cursor lies outside its declared dwell.")
        global_step = (
            sum(s.dwell_steps for s in self.stages[: cursor.stage_index])
            + cursor.completed_steps
        )
        if int(cursor.state.step_index) != global_step:
            raise ValueError("Cursor and native event-addressed random step disagree.")
        expected_time = (
            sum(s.dwell_time for s in self.stages[: cursor.stage_index])
            + cursor.completed_steps * stage.runtime.integrator.step_size
        )
        dtype = np.asarray(cursor.state.time).dtype
        if not np.isclose(
            float(cursor.state.time),
            expected_time,
            rtol=32 * np.finfo(dtype).eps,
            atol=32
            * np.finfo(dtype).eps
            * max(1, global_step)
            * stage.runtime.integrator.step_size,
        ):
            raise ValueError(
                "Cursor physical time does not match the declared dwell schedule."
            )

    def run(
        self, cursor: CotranslationCursor, /, *, stop_after_stage: int | None = None
    ) -> CotranslationRun:
        """Execute native MD and atomic insertions, preserving each event boundary.

        A rejected activation returns the exact preactivation cursor, allowing
        rollback/replay without changing randomness. Segments never include a
        lag pair or continuous interpolation across topology switches.
        """
        self._validate_cursor(cursor)
        stop = len(self.stages) - 1 if stop_after_stage is None else stop_after_stage
        if not cursor.stage_index <= stop < len(self.stages):
            raise ValueError(
                "Requested stopping stage is outside the remaining protocol."
            )
        segments, ledgers = [], []
        current = cursor
        for index in range(cursor.stage_index, stop + 1):
            stage = self.stages[index]
            if index > current.stage_index:
                activation = activate_topology_epoch(
                    self.transition(index),
                    current.state,
                    np.asarray(stage.inserted_positions).reshape((-1, 3)),
                    np.asarray(stage.inserted_momenta).reshape((-1, 3)),
                )
                if not activation.successful:
                    return CotranslationRun(
                        current,
                        tuple(segments),
                        tuple(ledgers),
                        False,
                        activation.refusal,
                    )
                ledgers.append(activation.ledger)
                current = CotranslationCursor(
                    self.protocol_id, index, 0, activation.state
                )
            times = [current.state.time]
            positions = [current.state.kinematics.positions]
            failure = None
            for completed in range(current.completed_steps, stage.dwell_steps):
                step = _step(stage.runtime, current.state)
                if not bool(step.successful):
                    failure = f"native-step-rejected:{int(step.rejection_reasons)}"
                    break
                current = CotranslationCursor(
                    self.protocol_id, index, completed + 1, step.accepted_state
                )
                times.append(current.state.time)
                positions.append(current.state.kinematics.positions)
            support = SeriesSupport(
                jnp.stack(times),
                coordinate_name="atomistic-time",
                coordinate_id=f"{self.protocol_id}:stage:{index}",
            )
            values = jnp.stack(positions)
            segments.append(
                SampledSeries(
                    support,
                    values,
                    value_valid=jnp.broadcast_to(
                        stage.runtime.system.active_mask[None, :, None], values.shape
                    ),
                    series_id=f"{self.protocol_id}:stage:{index}:positions",
                )
            )
            if failure is not None:
                return CotranslationRun(
                    current, tuple(segments), tuple(ledgers), False, failure
                )
        return CotranslationRun(current, tuple(segments), tuple(ledgers), True, None)

    def write_checkpoint(
        self, path: str | Path, cursor: CotranslationCursor, /
    ) -> AtomisticCheckpoint:
        """Use native checkpoint storage; caller retains the source-pinned protocol."""
        self._validate_cursor(cursor)
        return write_atomistic_checkpoint(
            path,
            AtomisticCheckpointPlan(
                self.stages[cursor.stage_index].runtime, scope_id=self.protocol_id
            ),
            cursor.state,
        )

    def read_checkpoint(
        self, path: str | Path, cursor_template: CotranslationCursor, /
    ) -> CotranslationCursor:
        """Restore an epoch with native identity/integrity checks and schedule checks."""
        self._validate_cursor(cursor_template)
        index = cursor_template.stage_index
        restored = read_atomistic_checkpoint(
            path,
            AtomisticCheckpointPlan(
                self.stages[index].runtime, scope_id=self.protocol_id
            ),
            cursor_template.state,
        )
        completed = int(restored.state.step_index) - sum(
            s.dwell_steps for s in self.stages[:index]
        )
        cursor = CotranslationCursor(self.protocol_id, index, completed, restored.state)
        self._validate_cursor(cursor)
        return cursor


__all__ = [
    "CotranslationStage",
    "CotranslationProtocol",
    "CotranslationCursor",
    "CotranslationRun",
]
