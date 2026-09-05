#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded, identity-addressed telegraph experiments in declared physical time.

Preparation and orchestration are host operations. Each constant-rate interval uses
native SSA. Restarting independent exponential clocks at an exogenous boundary is
exact by memorylessness; it does not preserve a path under schedule refinement.
Neither a scenario fork nor a repeated protocol denotes biological cell division.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.linalg import matrix_exponential_action, MatrixFunctionPolicy
from phydrax.series import SampledSeries, SeriesSupport
from phydrax.solver import JumpSolution, solve_direct_ssa
from phydrax.stochastic import PoissonClockRealization
from phydrax.units import (
    conversion_factor,
    convert_value,
    derived_unit,
    SECOND,
    UnitDefinition,
)

from .._gene_expression import TelegraphGeneExpressionPlan


def _identity(value: int, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{owner} must be an integer.")
    if not 0 <= int(value) < 2**63:
        raise ValueError(f"{owner} must be a nonnegative signed-int64 identity.")
    return int(value)


def _label(value: str, owner: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{owner} must be nonempty and trimmed.")
    return value


def _positive_integer(value: int, owner: str) -> int:
    result = _identity(value, owner)
    if result == 0:
        raise ValueError(f"{owner} must be positive.")
    return result


def _address(key: Array, *identities: int) -> Array:
    for identity in identities:
        key = jax.random.fold_in(key, identity & 0xFFFFFFFF)
        key = jax.random.fold_in(key, identity >> 32)
    return key


@dataclass(frozen=True, slots=True)
class CellIdentity:
    cell_id: int
    label: str

    def __post_init__(self):
        _identity(self.cell_id, "cell_id")
        _label(self.label, "Cell label")


@dataclass(frozen=True, slots=True)
class GeneIdentity:
    gene_id: int
    label: str

    def __post_init__(self):
        _identity(self.gene_id, "gene_id")
        _label(self.label, "Gene label")


@dataclass(frozen=True, slots=True, init=False)
class PiecewiseConstantRates:
    """Positive rates ordered activation, deactivation, transcription, β, γ.

    ``rates`` has shape (interval, gene, 5). Boundaries are expressed in
    ``time_unit``; input rates carry an independently specified inverse-time unit.
    Values at an interior boundary use the interval on its right. No callable or
    smooth-rate approximation is admitted by this exact execution profile.
    """

    boundaries: tuple[float, ...]
    rates: Array
    time_unit: UnitDefinition
    schedule_id: str

    def __init__(
        self,
        boundaries: ArrayLike,
        rates: ArrayLike,
        /,
        *,
        rate_unit: UnitDefinition,
        time_unit: UnitDefinition = SECOND,
    ):
        conversion_factor(time_unit, SECOND)
        raw_boundaries = np.asarray(boundaries, dtype=float)
        if raw_boundaries.ndim != 1 or raw_boundaries.size < 2:
            raise ValueError(
                "Schedule boundaries must be a vector with at least two nodes."
            )
        if not np.all(np.isfinite(raw_boundaries)) or np.any(
            np.diff(raw_boundaries) <= 0
        ):
            raise ValueError(
                "Schedule boundaries must be finite and strictly increasing."
            )
        values = np.asarray(rates)
        if values.dtype.kind not in "ifu" or values.ndim != 3 or values.shape[2] != 5:
            raise ValueError("Rates must be a real (interval, gene, 5) array.")
        if values.shape[0] != raw_boundaries.size - 1 or values.shape[1] == 0:
            raise ValueError(
                "Rate intervals and boundaries must match with nonempty genes."
            )
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Every runtime rate must be finite and strictly positive.")
        inverse_time = derived_unit(f"1/({time_unit.symbol})", ((time_unit, -1),))
        runtime = convert_value(
            jnp.asarray(values, dtype=float), source=rate_unit, target=inverse_time
        )
        if not np.all(np.isfinite(np.asarray(runtime))) or np.any(
            np.asarray(runtime) <= 0
        ):
            raise ValueError(
                "Unit conversion must preserve finite positive runtime rates."
            )
        object.__setattr__(self, "boundaries", tuple(float(t) for t in raw_boundaries))
        object.__setattr__(self, "rates", runtime)
        object.__setattr__(self, "time_unit", time_unit)
        object.__setattr__(
            self,
            "schedule_id",
            canonical_fingerprint(
                {
                    "kind": "single-cell-piecewise-constant",
                    "boundaries": raw_boundaries.tolist(),
                    "rates": array_tree_fingerprint(np.asarray(runtime)),
                    "time_unit": time_unit.unit_id,
                }
            ),
        )

    def repeat(self, cycles: int, /) -> PiecewiseConstantRates:
        """Unroll a finite cyclic external protocol, not a biological lineage."""
        count = _positive_integer(cycles, "cycles")
        duration = self.boundaries[-1] - self.boundaries[0]
        boundaries = tuple(
            t + cycle * duration for cycle in range(count) for t in self.boundaries[:-1]
        ) + (self.boundaries[0] + count * duration,)
        return PiecewiseConstantRates(
            boundaries,
            jnp.tile(self.rates, (count, 1, 1)),
            rate_unit=derived_unit("runtime-rate", ((self.time_unit, -1),)),
            time_unit=self.time_unit,
        )


@dataclass(frozen=True, slots=True)
class ScenarioSegment:
    segment_id: int
    schedule: PiecewiseConstantRates
    save_times: tuple[float, ...]
    parent_id: int | None = None

    def __post_init__(self):
        _identity(self.segment_id, "segment_id")
        if not isinstance(self.schedule, PiecewiseConstantRates):
            raise TypeError("schedule must be PiecewiseConstantRates.")
        if self.parent_id is not None:
            _identity(self.parent_id, "parent_id")
        values = tuple(float(t) for t in self.save_times)
        if (
            not values
            or not np.all(np.isfinite(values))
            or np.any(np.diff(values) <= 0)
            or values[0] < self.schedule.boundaries[0]
            or values[-1] > self.schedule.boundaries[-1]
        ):
            raise ValueError(
                "Saved physical times must increase within the schedule interval."
            )
        object.__setattr__(self, "save_times", values)


@dataclass(frozen=True, slots=True, init=False)
class TranscriptScenario:
    """Finite cells × genes × ordered scenario-tree support.

    A child starts from the parent's terminal state with new addressed randomness.
    Multiple children are counterfactual continuations; molecular material is NOT
    partitioned. Biological division is deliberately not an admitted profile.
    ``initial_states`` is (cell, gene, 3), ordered promoter-on, U, S.
    """

    cells: tuple[CellIdentity, ...]
    genes: tuple[GeneIdentity, ...]
    segments: tuple[ScenarioSegment, ...]
    initial_states: Array
    max_paths: int
    max_events_per_interval: int
    scenario_id: str

    def __init__(
        self,
        cells: tuple[CellIdentity, ...],
        genes: tuple[GeneIdentity, ...],
        segments: tuple[ScenarioSegment, ...],
        initial_states: ArrayLike,
        /,
        *,
        max_paths: int,
        max_events_per_interval: int,
    ):
        cells, genes, segments = tuple(cells), tuple(genes), tuple(segments)
        capacity = _positive_integer(max_paths, "max_paths")
        events = _positive_integer(max_events_per_interval, "max_events_per_interval")
        if not cells or not genes or not segments:
            raise ValueError("Scenario cells, genes, and segments must be nonempty.")
        if not all(isinstance(x, CellIdentity) for x in cells) or not all(
            isinstance(x, GeneIdentity) for x in genes
        ):
            raise TypeError(
                "Scenario supports require explicit cell and gene identities."
            )
        if len({x.cell_id for x in cells}) != len(cells) or len(
            {x.gene_id for x in genes}
        ) != len(genes):
            raise ValueError(
                "Cell and gene identities must be unique within their supports."
            )
        if len(cells) * len(genes) * len(segments) > capacity:
            raise ValueError("Scenario exceeds the declared path capacity.")
        seen: dict[int, ScenarioSegment] = {}
        unit = segments[0].schedule.time_unit
        for segment in segments:
            if not isinstance(segment, ScenarioSegment) or segment.segment_id in seen:
                raise ValueError("Scenario segments must have unique identities.")
            if segment.schedule.rates.shape[1] != len(genes):
                raise ValueError("Every schedule must bind the declared gene order.")
            if segment.schedule.time_unit != unit:
                raise ValueError(
                    "All segments must use exactly the same runtime time unit."
                )
            if segment.parent_id is not None:
                if segment.parent_id not in seen:
                    raise ValueError(
                        "A parent must precede its child; unroll cycles explicitly."
                    )
                if (
                    segment.schedule.boundaries[0]
                    != seen[segment.parent_id].schedule.boundaries[-1]
                ):
                    raise ValueError(
                        "A child must begin at its parent's terminal physical time."
                    )
            seen[segment.segment_id] = segment
        states = np.asarray(initial_states)
        if states.dtype.kind not in "ifu" or states.shape != (len(cells), len(genes), 3):
            raise ValueError("Initial states must have shape (cell, gene, 3).")
        if (
            not np.all(np.isfinite(states))
            or np.any(states < 0)
            or np.any(states != np.floor(states))
            or np.any(states[..., 0] > 1)
        ):
            raise ValueError(
                "Initial promoter must be binary and RNA counts nonnegative integers."
            )
        for name, value in (
            ("cells", cells),
            ("genes", genes),
            ("segments", segments),
            ("initial_states", jnp.asarray(states, dtype=float)),
            ("max_paths", capacity),
            ("max_events_per_interval", events),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "scenario_id",
            canonical_fingerprint(
                {
                    "kind": "telegraph-scenario",
                    "cells": [(x.cell_id, x.label) for x in cells],
                    "genes": [(x.gene_id, x.label) for x in genes],
                    "segments": [
                        (x.segment_id, x.parent_id, x.schedule.schedule_id, x.save_times)
                        for x in segments
                    ],
                    "initial": array_tree_fingerprint(states),
                    "randomness": "cell-gene-segment-interval-event",
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class TranscriptPath:
    cell: CellIdentity
    gene: GeneIdentity
    segment: ScenarioSegment
    latent: SampledSeries
    conditional_drift: SampledSeries
    intervals: tuple[JumpSolution, ...]
    scenario_id: str

    @property
    def path_id(self) -> str:
        return canonical_fingerprint(
            {
                "scenario": self.scenario_id,
                "cell": self.cell.cell_id,
                "gene": self.gene.gene_id,
                "segment": self.segment.segment_id,
                "realizations": [x.realization.realization_id for x in self.intervals],
            }
        )

    @property
    def conditional_drift_unit(self) -> UnitDefinition:
        """Transcript counts per declared physical-time unit (counts are dimensionless)."""
        return derived_unit(
            "transcript-drift-rate", ((self.segment.schedule.time_unit, -1),)
        )


class ScenarioExecutionError(RuntimeError):
    """An incomplete native SSA interval; no descendant is executed from this state."""

    def __init__(self, solution: JumpSolution):
        super().__init__(
            "Transcript SSA interval failed; inspect solution.events.status and increase capacity if exhausted."
        )
        self.solution = solution


@dataclass(frozen=True, slots=True)
class TranscriptExperiment:
    scenario: TranscriptScenario
    paths: tuple[TranscriptPath, ...]

    def joined_series(self, cell_id: int, gene_id: int, /) -> SampledSeries:
        """Concatenate scenarios with disconnected edges at EVERY segment reset.

        Branches can share terminal/initial coordinates, but never form lag pairs.
        Selecting a physical continuation is a separate caller decision.
        """
        paths = tuple(
            p
            for p in self.paths
            if p.cell.cell_id == cell_id and p.gene.gene_id == gene_id
        )
        if not paths:
            raise ValueError("The requested cell/gene path was not executed.")
        coordinates = jnp.concatenate(tuple(p.latent.support.coordinates for p in paths))
        values = jnp.concatenate(tuple(p.latent.values for p in paths))
        edges = np.ones(coordinates.shape[0] - 1, dtype=bool)
        offset = 0
        for path in paths[:-1]:
            offset += path.latent.support.capacity
            edges[offset - 1] = False
        support = SeriesSupport(
            coordinates,
            edge_valid=edges,
            coordinate_name="physical_time",
            coordinate_id=self.scenario.segments[0].schedule.time_unit.unit_id,
        )
        return SampledSeries(
            support,
            values,
            series_id=f"{self.scenario.scenario_id}:{cell_id}:{gene_id}:reset-series",
        )


def generate_transcripts(
    scenario: TranscriptScenario,
    key: Array,
    /,
    *,
    cell_ids: tuple[int, ...] | None = None,
    gene_ids: tuple[int, ...] | None = None,
) -> TranscriptExperiment:
    """Execute native direct SSA with event-addressed, workset-independent keys.

    Cell/gene subsets do not renumber randomness. Interval restarts use independent
    clocks, so replay requires the same explicit schedule and segment identities.
    Host loops keep every path's distinct inherited initial state; compiled native
    interval kernels are reused. Saved boundaries are always retained exactly.
    """
    if not isinstance(scenario, TranscriptScenario):
        raise TypeError("scenario must be TranscriptScenario.")
    selected_cells = (
        tuple(x.cell_id for x in scenario.cells) if cell_ids is None else tuple(cell_ids)
    )
    selected_genes = (
        tuple(x.gene_id for x in scenario.genes) if gene_ids is None else tuple(gene_ids)
    )
    if (
        not selected_cells
        or not selected_genes
        or len(set(selected_cells)) != len(selected_cells)
        or len(set(selected_genes)) != len(selected_genes)
        or not set(selected_cells) <= {x.cell_id for x in scenario.cells}
        or not set(selected_genes) <= {x.gene_id for x in scenario.genes}
    ):
        raise ValueError("Worksets must contain unique declared cell/gene identities.")
    model = TelegraphGeneExpressionPlan(
        *tuple(scenario.segments[0].schedule.rates[0, 0]),
        name="single-cell-runtime-telegraph",
    ).prepare()
    process = model.network.exact_jump_process()
    paths = []
    final_states: dict[tuple[int, int, int], Array] = {}
    for ci, cell in enumerate(scenario.cells):
        if cell.cell_id not in selected_cells:
            continue
        for gi, gene in enumerate(scenario.genes):
            if gene.gene_id not in selected_genes:
                continue
            for segment in scenario.segments:
                latent_initial = scenario.initial_states[ci, gi]
                state = (
                    jnp.concatenate((1.0 - latent_initial[:1], latent_initial))
                    if segment.parent_id is None
                    else final_states[(cell.cell_id, gene.gene_id, segment.parent_id)]
                )
                solutions, times, states = [], [], []
                for interval, (start, end) in enumerate(
                    zip(
                        segment.schedule.boundaries[:-1],
                        segment.schedule.boundaries[1:],
                        strict=True,
                    )
                ):
                    saved = tuple(
                        sorted(
                            {
                                start,
                                end,
                                *(t for t in segment.save_times if start <= t <= end),
                            }
                        )
                    )
                    addressed_key = _address(
                        key, 0, cell.cell_id, gene.gene_id, segment.segment_id, interval
                    )
                    realization = PoissonClockRealization(
                        addressed_key,
                        process.num_channels,
                        support=(start, end),
                        max_events_per_channel=scenario.max_events_per_interval,
                        process_id=process.process_id,
                    )
                    solution = solve_direct_ssa(
                        process,
                        realization,
                        state,
                        t0=start,
                        t1=end,
                        save_times=jnp.asarray(saved),
                        args=model.runtime(segment.schedule.rates[interval, gi]),
                        max_events=scenario.max_events_per_interval,
                    )
                    if not bool(jnp.all(solution.successful)):
                        raise ScenarioExecutionError(solution)
                    solutions.append(solution)
                    times.append(solution.times if interval == 0 else solution.times[1:])
                    states.append(
                        solution.states if interval == 0 else solution.states[1:]
                    )
                    state = solution.states[-1]
                final_states[(cell.cell_id, gene.gene_id, segment.segment_id)] = state
                coordinates, full = (
                    jnp.concatenate(tuple(times)),
                    jnp.concatenate(tuple(states)),
                )
                latent_values = full[:, 1:]
                rate_index = jnp.clip(
                    jnp.searchsorted(
                        jnp.asarray(segment.schedule.boundaries),
                        coordinates,
                        side="right",
                    )
                    - 1,
                    0,
                    segment.schedule.rates.shape[0] - 1,
                )
                rates = segment.schedule.rates[rate_index, gi]
                drift = (
                    rates[:, 3] * latent_values[:, 1] - rates[:, 4] * latent_values[:, 2]
                )
                support = SeriesSupport(
                    coordinates,
                    coordinate_name="physical_time",
                    coordinate_id=segment.schedule.time_unit.unit_id,
                )
                identity = f"{scenario.scenario_id}:{cell.cell_id}:{gene.gene_id}:{segment.segment_id}"
                paths.append(
                    TranscriptPath(
                        cell,
                        gene,
                        segment,
                        SampledSeries(
                            support, latent_values, series_id=identity + ":latent-P-U-S"
                        ),
                        SampledSeries(
                            support, drift, series_id=identity + ":conditional-drift"
                        ),
                        tuple(solutions),
                        scenario.scenario_id,
                    )
                )
    return TranscriptExperiment(scenario, tuple(paths))


def transient_transcript_mean(
    rates: ArrayLike, initial_mean: ArrayLike, duration: ArrayLike, /
) -> Array:
    """Exact affine first-moment law for one constant-rate interval, ordered P,U,S.

    Native matrix-exponential action solves the four-dimensional augmented system;
    this differentiable moment map is not a derivative of an SSA sample path.
    """
    a, b, alpha, beta, gamma = jnp.asarray(rates)
    zero = jnp.zeros_like(a)
    matrix = jnp.stack(
        (
            jnp.stack((-(a + b), zero, zero, a)),
            jnp.stack((alpha, -beta, zero, zero)),
            jnp.stack((zero, beta, -gamma, zero)),
            jnp.stack((zero, zero, zero, zero)),
        )
    )
    initial = jnp.concatenate(
        (jnp.asarray(initial_mean), jnp.ones((1,), dtype=matrix.dtype))
    )
    result = matrix_exponential_action(
        lambda value: matrix @ value,
        initial,
        duration,
        policy=MatrixFunctionPolicy("arnoldi", max_dimension=4),
    )
    return jnp.where(result.converged, result.value[:3], jnp.nan)


def scheduled_transcript_mean(
    schedule: PiecewiseConstantRates, initial_mean: ArrayLike, /, *, gene_index: int = 0
) -> Array:
    """First moments at every exact boundary of a prepared schedule."""
    if not 0 <= gene_index < schedule.rates.shape[1]:
        raise ValueError("gene_index is outside the prepared gene support.")
    state = jnp.asarray(initial_mean, dtype=schedule.rates.dtype)
    if state.shape != (3,):
        raise ValueError("initial_mean must have shape (3,).")
    values = [state]
    for i, (start, end) in enumerate(
        zip(schedule.boundaries[:-1], schedule.boundaries[1:], strict=True)
    ):
        state = transient_transcript_mean(
            schedule.rates[i, gene_index], state, end - start
        )
        values.append(state)
    return jnp.stack(values)
