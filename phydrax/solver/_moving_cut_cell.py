#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative accepted-boundary transactions for moving multivalued cut cells."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.amr._cut_complex import (
    MultivaluedCutCellComplex,
    MultivaluedCutCellPlan,
)
from ..discretization.amr._cut_transition import MultivaluedCutCellTransition


UncoveredStateProvider = Callable[
    [MultivaluedCutCellComplex, Array, Array, Any], ArrayLike
]


class MovingCutCellState(StrictModule):
    """Accepted cut topology and extensive content at one synchronized time."""

    complex: MultivaluedCutCellComplex
    content: Array
    time: Array
    revision: Array
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex_: MultivaluedCutCellComplex,
        content: ArrayLike,
        time: ArrayLike,
        revision: ArrayLike,
        /,
    ):
        if not isinstance(complex_, MultivaluedCutCellComplex):
            raise TypeError("Moving cut-cell state requires MultivaluedCutCellComplex.")
        value = jnp.asarray(content)
        time_ = jnp.asarray(time)
        revision_ = jnp.asarray(revision)
        if (
            value.ndim == 0
            or value.shape[0] != complex_.component_capacity
            or time_.shape != ()
            or revision_.shape != ()
            or revision_.dtype.kind not in "iu"
        ):
            raise ValueError("Moving cut-cell content/time/revision shapes are invalid.")
        trailing = (1,) * (value.ndim - 1)
        inactive = (~complex_.component_active).reshape(
            complex_.component_active.shape + trailing
        )
        value = eqx.error_if(
            value,
            jnp.any(inactive & (value != 0.0)) | jnp.any(~jnp.isfinite(value)),
            "Moving cut-cell content must be finite and zero on inactive slots.",
        )
        self.complex = complex_
        self.content = value
        self.time = time_
        self.revision = revision_.astype(jnp.int32)
        self.state_id = canonical_fingerprint(
            {
                "kind": "moving-cut-cell-state",
                "topology": complex_.topology_id,
                "geometry": complex_.geometry_id,
                "component_shape": value.shape[1:],
            }
        )

    def cell_average(self, /) -> Array:
        trailing = (1,) * (self.content.ndim - 1)
        volumes = jnp.where(
            self.complex.component_active,
            self.complex.component_volumes,
            1.0,
        ).reshape(self.complex.component_volumes.shape + trailing)
        return jnp.where(
            self.complex.component_active.reshape(
                self.complex.component_active.shape + trailing
            ),
            self.content / volumes,
            jnp.zeros((), dtype=self.content.dtype),
        )


class MovingCutCellStepEvidence(StrictModule, NonTrainableState):
    """Space-time measure and content ledger for one moving topology transaction."""

    source_volume: Array
    target_volume: Array
    overlap_volume: Array
    covered_volume: Array
    uncovered_volume: Array
    volume_balance_defect: Array
    wall_content_integral: Array
    topology_changed: bool = eqx.field(static=True)
    topology_event_count: int = eqx.field(static=True)
    valid: Array
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source_volume: ArrayLike,
        target_volume: ArrayLike,
        overlap_volume: ArrayLike,
        covered_volume: ArrayLike,
        uncovered_volume: ArrayLike,
        volume_balance_defect: ArrayLike,
        wall_content_integral: ArrayLike,
        topology_changed: bool,
        topology_event_count: int,
        tolerance: float,
    ):
        source = jnp.asarray(source_volume)
        target = jnp.asarray(target_volume)
        overlap = jnp.asarray(overlap_volume)
        covered = jnp.asarray(covered_volume)
        uncovered = jnp.asarray(uncovered_volume)
        defect = jnp.asarray(volume_balance_defect)
        wall = jnp.asarray(wall_content_integral)
        if any(
            value.shape != ()
            for value in (source, target, overlap, covered, uncovered, defect)
        ):
            raise ValueError("Moving cut-cell volume evidence must be scalar.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Moving cut-cell evidence tolerance must be positive.")
        scale = jnp.maximum(jnp.maximum(jnp.abs(source), jnp.abs(target)), 1.0)
        valid = (
            jnp.all(jnp.isfinite(wall))
            & jnp.all(
                jnp.isfinite(
                    jnp.stack((source, target, overlap, covered, uncovered, defect))
                )
            )
            & (covered >= -tolerance_)
            & (uncovered >= -tolerance_)
            & (defect <= tolerance_ * scale)
        )
        self.source_volume = source
        self.target_volume = target
        self.overlap_volume = overlap
        self.covered_volume = covered
        self.uncovered_volume = uncovered
        self.volume_balance_defect = defect
        self.wall_content_integral = wall
        self.topology_changed = bool(topology_changed)
        self.topology_event_count = int(topology_event_count)
        self.valid = valid
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "moving-cut-cell-step-evidence",
                "topology_changed": bool(topology_changed),
                "topology_event_count": int(topology_event_count),
                "tolerance": tolerance_,
            }
        )


class MovingCutCellStepResult(StrictModule):
    """Candidate successor state with stage geometries and swept ledger."""

    state: MovingCutCellState
    stage_complexes: tuple[
        MultivaluedCutCellComplex,
        MultivaluedCutCellComplex,
        MultivaluedCutCellComplex,
    ]
    transition: MultivaluedCutCellTransition
    evidence: MovingCutCellStepEvidence
    accepted: Array
    result_id: str = eqx.field(static=True)


class MovingTopologyLocalizationEvidence(StrictModule, NonTrainableState):
    """Bounded dyadic probe and bisection evidence for moving topology events."""

    event_times: tuple[float, ...] = eqx.field(static=True)
    probe_count: int = eqx.field(static=True)
    bisection_iterations: int = eqx.field(static=True)
    minimum_event_separation: float = eqx.field(static=True)
    valid: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class LocalizedMovingCutCellResult(StrictModule):
    """Accepted successor assembled from fixed-topology event subintervals."""

    state: MovingCutCellState
    substeps: tuple[MovingCutCellStepResult, ...]
    localization: MovingTopologyLocalizationEvidence
    accepted: Array
    result_id: str = eqx.field(static=True)


class MovingTopologyLocalizationPlan(StrictModule, NonTrainableState):
    """Locate every topology identity crossing visible to a finite probe envelope."""

    probe_count: int = eqx.field(static=True)
    bisection_iterations: int = eqx.field(static=True)
    minimum_event_separation: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        probe_count: int = 16,
        bisection_iterations: int = 32,
        minimum_event_separation: float = 1.0e-10,
    ):
        probes = int(probe_count)
        iterations = int(bisection_iterations)
        separation = float(minimum_event_separation)
        if (
            probes < 2
            or iterations <= 0
            or not np.isfinite(separation)
            or separation <= 0.0
        ):
            raise ValueError("Moving topology localization bounds are invalid.")
        self.probe_count = probes
        self.bisection_iterations = iterations
        self.minimum_event_separation = separation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "moving-topology-localization-plan",
                "probe_count": probes,
                "bisection_iterations": iterations,
                "minimum_event_separation": separation,
            }
        )

    def locate(
        self,
        moving: "MovingMultivaluedCutCellPlan",
        start_time: float,
        end_time: float,
        args: Any = None,
        /,
    ) -> MovingTopologyLocalizationEvidence:
        if not isinstance(moving, MovingMultivaluedCutCellPlan):
            raise TypeError("Topology localization requires moving cut-cell plan.")
        start = float(start_time)
        end = float(end_time)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("Moving topology localization interval is invalid.")
        probe_times = np.linspace(start, end, self.probe_count + 1)
        probe_topologies = tuple(
            moving.cut_plan.sign_topology(float(time), args) for time in probe_times
        )
        brackets = [
            (
                float(probe_times[index]),
                float(probe_times[index + 1]),
                probe_topologies[index].topology_id,
            )
            for index in range(self.probe_count)
            if probe_topologies[index].topology_id
            != probe_topologies[index + 1].topology_id
        ]
        event_times = []
        for left, right, left_id in brackets:
            for _ in range(self.bisection_iterations):
                midpoint = 0.5 * (left + right)
                middle = moving.cut_plan.sign_topology(midpoint, args)
                if middle.topology_id == left_id:
                    left = midpoint
                else:
                    right = midpoint
            event_times.append(min(end, right + self.minimum_event_separation))
        event_times = sorted(set(event_times))
        separated = all(
            right - left >= self.minimum_event_separation
            for left, right in zip(event_times[:-1], event_times[1:], strict=True)
        )
        capacity_ok = (
            len(event_times) <= moving.cut_plan.resources.maximum_topology_events
        )
        valid = bool(separated and capacity_ok)
        return MovingTopologyLocalizationEvidence(
            event_times=tuple(event_times),
            probe_count=self.probe_count,
            bisection_iterations=self.bisection_iterations,
            minimum_event_separation=self.minimum_event_separation,
            valid=valid,
            evidence_id=canonical_fingerprint(
                {
                    "kind": "moving-topology-localization-evidence",
                    "plan": self.plan_id,
                    "moving": moving.plan_id,
                    "interval": (start, end),
                    "event_times": event_times,
                    "valid": valid,
                }
            ),
        )


class MovingMultivaluedCutCellPlan(StrictModule, NonTrainableState):
    """Host-localized moving cut-cell geometry and conservative content remap."""

    cut_plan: MultivaluedCutCellPlan
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cut_plan: MultivaluedCutCellPlan,
        /,
        *,
        tolerance: float = 1.0e-8,
    ):
        tolerance_ = float(tolerance)
        if not isinstance(cut_plan, MultivaluedCutCellPlan):
            raise TypeError("Moving cut-cell plan requires MultivaluedCutCellPlan.")
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Moving cut-cell tolerance must be positive and finite.")
        self.cut_plan = cut_plan
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "moving-multivalued-cut-cell-plan",
                "cut_plan": cut_plan.plan_id,
                "tolerance": tolerance_,
            }
        )

    def initialize(
        self,
        cell_average: ArrayLike,
        time: ArrayLike = 0.0,
        args: Any = None,
        /,
    ) -> MovingCutCellState:
        complex_ = self.cut_plan.prepare(time, args)
        average = jnp.asarray(cell_average)
        if average.ndim == 0:
            average = jnp.broadcast_to(average, (complex_.component_capacity,))
        elif average.shape[0] == complex_.component_count:
            padded = jnp.zeros(
                (complex_.component_capacity,) + average.shape[1:], dtype=average.dtype
            )
            average = padded.at[: complex_.component_count].set(average)
        elif average.shape[0] != complex_.component_capacity:
            average = jnp.broadcast_to(
                average,
                (complex_.component_capacity,) + average.shape,
            )
        trailing = (1,) * (average.ndim - 1)
        content = average * complex_.component_volumes.reshape(
            complex_.component_volumes.shape + trailing
        )
        content = jnp.where(
            complex_.component_active.reshape(complex_.component_active.shape + trailing),
            content,
            jnp.zeros((), dtype=content.dtype),
        )
        return MovingCutCellState(complex_, content, time, 0)

    def _uncovered_average(
        self,
        provider: ArrayLike | UncoveredStateProvider,
        target: MultivaluedCutCellComplex,
        uncovered: Array,
        time: Array,
        args: Any,
        component_shape: tuple[int, ...],
    ) -> Array:
        supplied = (
            provider(target, uncovered, time, args) if callable(provider) else provider
        )
        value = jnp.asarray(supplied)
        target_shape = (target.component_count,) + component_shape
        if value.shape == component_shape:
            value = jnp.broadcast_to(value, target_shape)
        elif value.shape == (target.component_capacity,) + component_shape:
            value = value[: target.component_count]
        if value.shape != target_shape:
            raise ValueError(
                "Uncovered-state provider must return one average per target component."
            )
        return eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value)),
            "Uncovered target averages must be finite.",
        )

    def advance(
        self,
        state: MovingCutCellState,
        step_size: ArrayLike,
        uncovered_state: ArrayLike | UncoveredStateProvider,
        args: Any = None,
        /,
    ) -> MovingCutCellStepResult:
        if not isinstance(state, MovingCutCellState):
            raise TypeError("Moving cut-cell advancement requires MovingCutCellState.")
        step = jnp.asarray(step_size, dtype=state.time.dtype)
        if step.shape != () or not bool(jnp.isfinite(step) & (step > 0.0)):
            raise ValueError("Moving cut-cell step_size must be positive and finite.")
        end_time = state.time + step
        midpoint_time = state.time + 0.5 * step
        midpoint = self.cut_plan.prepare(midpoint_time, args)
        target = self.cut_plan.prepare(end_time, args)
        transition = MultivaluedCutCellTransition(
            state.complex,
            target,
            tolerance=self.tolerance,
            require_complete=False,
        )
        source_count = state.complex.component_count
        target_count = target.component_count
        source_content = state.content[:source_count]
        overlap_content = transition.remap.apply_content(
            source_content,
            target_active_mask=jnp.ones((target_count,), dtype=bool),
        )
        target_volumes = target.component_volumes[:target_count]
        source_volumes = state.complex.component_volumes[:source_count]
        uncovered = jnp.maximum(
            target_volumes - transition.target_coverage,
            jnp.zeros((), dtype=target_volumes.dtype),
        )
        covered = jnp.maximum(
            source_volumes - transition.source_coverage,
            jnp.zeros((), dtype=source_volumes.dtype),
        )
        component_shape = state.content.shape[1:]
        extension = self._uncovered_average(
            uncovered_state,
            target,
            uncovered,
            end_time,
            args,
            component_shape,
        )
        trailing = (1,) * len(component_shape)
        target_content_active = overlap_content + extension * uncovered.reshape(
            uncovered.shape + trailing
        )
        target_content = (
            jnp.zeros(
                (target.component_capacity,) + component_shape,
                dtype=target_content_active.dtype,
            )
            .at[:target_count]
            .set(target_content_active)
        )
        source_total = jnp.sum(source_content, axis=0)
        target_total = jnp.sum(target_content_active, axis=0)
        wall_content = target_total - source_total
        source_volume = jnp.sum(source_volumes)
        target_volume = jnp.sum(target_volumes)
        overlap_volume = jnp.sum(transition.target_coverage)
        covered_volume = jnp.sum(covered)
        uncovered_volume = jnp.sum(uncovered)
        volume_defect = jnp.abs(
            (target_volume - source_volume) - (uncovered_volume - covered_volume)
        )
        topology_changed = state.complex.topology_id != target.topology_id
        event_count = int(topology_changed)
        if event_count > self.cut_plan.resources.maximum_topology_events:
            raise ValueError("Moving cut-cell topology event capacity is exceeded.")
        evidence = MovingCutCellStepEvidence(
            source_volume=source_volume,
            target_volume=target_volume,
            overlap_volume=overlap_volume,
            covered_volume=covered_volume,
            uncovered_volume=uncovered_volume,
            volume_balance_defect=volume_defect,
            wall_content_integral=wall_content,
            topology_changed=topology_changed,
            topology_event_count=event_count,
            tolerance=self.tolerance,
        )
        candidate = MovingCutCellState(
            target,
            target_content,
            end_time,
            state.revision + 1,
        )
        return MovingCutCellStepResult(
            state=candidate,
            stage_complexes=(state.complex, target, midpoint),
            transition=transition,
            evidence=evidence,
            accepted=evidence.valid,
            result_id=canonical_fingerprint(
                {
                    "kind": "moving-cut-cell-step-result",
                    "plan": self.plan_id,
                    "source": state.state_id,
                    "target": candidate.state_id,
                    "transition": transition.transition_id,
                    "evidence": evidence.evidence_id,
                }
            ),
        )

    def advance_localized(
        self,
        state: MovingCutCellState,
        step_size: ArrayLike,
        uncovered_state: ArrayLike | UncoveredStateProvider,
        localization: MovingTopologyLocalizationPlan,
        args: Any = None,
        /,
    ) -> LocalizedMovingCutCellResult:
        """Split one requested step at every bounded localized topology event."""

        if not isinstance(state, MovingCutCellState) or not isinstance(
            localization, MovingTopologyLocalizationPlan
        ):
            raise TypeError("Localized advancement requires state and localization plan.")
        step = float(np.asarray(step_size))
        start = float(np.asarray(state.time))
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("Localized moving step must be positive and finite.")
        evidence = localization.locate(self, start, start + step, args)
        if not evidence.valid:
            raise ValueError("Moving topology localization exceeded event bounds.")
        boundaries = (*evidence.event_times, start + step)
        current = state
        results = []
        for boundary in boundaries:
            local_step = boundary - float(np.asarray(current.time))
            if local_step <= 0.0:
                continue
            result = self.advance(
                current,
                local_step,
                uncovered_state,
                args,
            )
            if not bool(result.accepted):
                raise ValueError("Localized moving cut-cell substep was rejected.")
            results.append(result)
            current = result.state
        accepted = jnp.asarray(
            bool(results)
            and len(results) == len(boundaries)
            and all(bool(result.accepted) for result in results)
        )
        return LocalizedMovingCutCellResult(
            state=current,
            substeps=tuple(results),
            localization=evidence,
            accepted=accepted,
            result_id=canonical_fingerprint(
                {
                    "kind": "localized-moving-cut-cell-result",
                    "plan": self.plan_id,
                    "source": state.state_id,
                    "target": current.state_id,
                    "localization": evidence.evidence_id,
                    "substeps": [result.result_id for result in results],
                }
            ),
        )


__all__ = [
    "LocalizedMovingCutCellResult",
    "MovingTopologyLocalizationEvidence",
    "MovingTopologyLocalizationPlan",
    "MovingCutCellState",
    "MovingCutCellStepEvidence",
    "MovingCutCellStepResult",
    "MovingMultivaluedCutCellPlan",
    "UncoveredStateProvider",
]
