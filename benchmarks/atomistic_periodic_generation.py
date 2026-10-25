# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Compiled periodic generation admission: preparation, compile, first/warm and yield.

Every case prepares one periodic sampler (fixed-cell or variable-cell reverse
SDE, fixed-cell probability flow, or variable-composition reverse SDE) over
the same analytic wrapped-site score, then measures the lowering and
compilation of its compiled admission entrypoint for ``lanes`` resident lanes
and first/warm admissions with ``1 .. lanes`` admitted lanes.  Warm admission
cost against the admitted-lane count shows that vacant lanes do no work.
Every timed admission keeps its own result evidence: the first all-lane
admission, and for each admitted-lane mask its first (warm-up) and final
repeated admission, each with the selected lanes' terminal, solver-validity,
solver-detail and lane-status counts (pre-emission lane status; the stream's
emission screening is not applied).  Vacant lanes never count toward
success; failed lanes are recorded as measured, never retried.  A bounded
stream then reports attempts, accepted proposals, statuses and shortfall.
Attempts per second alone is not quality: proposals still need chemistry
qualification and physical relaxation.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax import _jax_config
from phydrax.atomistic._diffusion import (
    AtomisticLogCholeskyCellChart,
    AtomisticPeriodicCoordinateDiffusion,
    AtomisticPeriodicSamplerMethod,
    AtomisticSpeciesDiffusion,
    PreparedPeriodicDiffusionSampler,
    PreparedPeriodicFlowSampler,
)
from phydrax.atomistic._generation import (
    AbstractPreparedAtomisticSampler,
    AtomisticCellBounds,
    AtomisticCompositionBounds,
    AtomisticGenerationStreamPlan,
    AtomisticGenerationSupport,
    AtomisticMemoryGenerationSink,
    AtomisticProposalStatus,
    AtomisticSamplerLaneState,
)
from phydrax.domain import DomainFunction, HyperRectangle, TimeInterval
from phydrax.stochastic._categorical_diffusion import CategoricalDiffusionSchedule
from phydrax.stochastic._gaussian_diffusion import VarianceExplodingDiffusion
from phydrax.units import ANGSTROM


INITIAL_SCALE = 0.01
TERMINAL_SCALE = 1.0
LOG_RATIO = float(np.log(TERMINAL_SCALE / INITIAL_SCALE))
CELL = np.asarray([[4.0, 0.0, 0.0], [0.8, 3.8, 0.0], [0.3, 0.4, 4.2]])
SITES = np.asarray([[0.1, 0.2, 0.95], [0.6, 0.5, 0.45], [0.0, 0.0, 0.0]])
CHART = np.asarray([0.05, -0.02, 0.1, 0.1, -0.05, 0.08])
MASSES = "IUPAC standard atomic weights (benchmark fixture)"
KINDS = ("fixed-cell-sde", "variable-cell-sde", "fixed-cell-flow", "composition-sde")


class _WrappedSiteScore(eqx.Module):
    """Exact marginal score of wrapped normals at fixed sites (optional chart Gaussian)."""

    sites: jax.Array
    chart: jax.Array | None

    def __call__(self, state: jax.Array, time: jax.Array) -> jax.Array:
        variance = (INITIAL_SCALE * jnp.exp(LOG_RATIO * time)) ** 2
        size = self.sites.shape[0]
        delta = state[:size] - self.sites
        delta = delta - jnp.round(delta)
        images = jnp.arange(-6, 7, dtype=state.dtype)
        shifted = delta[:, None] + images[None, :]
        weights = jax.nn.softmax(-0.5 * shifted**2 / variance, axis=-1)
        fractional = -jnp.sum(weights * shifted, axis=-1) / variance
        if self.chart is None:
            return fractional
        return jnp.concatenate((fractional, -(state[size:] - self.chart) / variance))


def _score(sites: np.ndarray, chart: np.ndarray | None) -> DomainFunction:
    dimension = sites.size + (0 if chart is None else chart.size)
    domain = HyperRectangle(
        jnp.full((dimension,), -1e3), jnp.full((dimension,), 1e3), label="x"
    ) @ TimeInterval(0.0, 1.0)
    model = _WrappedSiteScore(
        jnp.asarray(sites.reshape(-1)), None if chart is None else jnp.asarray(chart)
    )
    return domain.Function("x", "t")(model)


def _bounds(maximum_volume: float = 200.0) -> AtomisticCellBounds:
    return AtomisticCellBounds(
        minimum_height=2.0,
        maximum_condition=10.0,
        minimum_volume=10.0,
        maximum_volume=maximum_volume,
    )


def _process(dimension: int) -> VarianceExplodingDiffusion:
    return VarianceExplodingDiffusion(
        dimension, initial_scale=INITIAL_SCALE, terminal_scale=TERMINAL_SCALE
    )


def _fixed_support(
    cell_admission: str,
    bounds: AtomisticCellBounds,
) -> AtomisticGenerationSupport:
    return AtomisticGenerationSupport(
        atom_capacity=3,
        species_vocabulary=(8, 14),
        vocabulary_masses=(15.999, 28.085),
        mass_provenance=MASSES,
        chemistry="fixed",
        length_unit=ANGSTROM,
        fixed_species=(14, 8, 0),
        fixed_mask=(True, True, False),
        cell_admission=cell_admission,
        cell=None if cell_admission == "variable-cell" else CELL,
        cell_bounds=bounds,
    )


def _species_logits(
    classes: jax.Array, timestep: jax.Array, *, key: jax.Array
) -> jax.Array:
    del timestep, key
    preferred = jnp.asarray([[1.5, 0.0, 0.0], [0.0, 1.5, 0.0]])
    return jnp.broadcast_to(preferred, classes.shape + (3,))


def _sampler(
    kind: str,
    lane_chunk: int,
) -> PreparedPeriodicDiffusionSampler | PreparedPeriodicFlowSampler:
    if kind == "fixed-cell-sde":
        return PreparedPeriodicDiffusionSampler(
            _fixed_support("fixed-cell", _bounds()),
            AtomisticPeriodicCoordinateDiffusion(3, _process(9), image_extent=4),
            _score(SITES, None),
            score_id="analytic-wrapped-sites",
            dt0=0.01,
            wiener_tolerance=1e-4,
            lane_chunk=lane_chunk,
        )
    if kind == "variable-cell-sde":
        bounds = _bounds()
        chart = AtomisticLogCholeskyCellChart(bounds, reference_length=4.0)
        return PreparedPeriodicDiffusionSampler(
            _fixed_support("variable-cell", bounds),
            AtomisticPeriodicCoordinateDiffusion(
                3, _process(15), image_extent=4, cell_chart=chart
            ),
            _score(SITES, CHART),
            score_id="analytic-wrapped-sites-and-chart",
            dt0=0.01,
            wiener_tolerance=1e-4,
            lane_chunk=lane_chunk,
        )
    if kind == "fixed-cell-flow":
        return PreparedPeriodicFlowSampler(
            _fixed_support("fixed-cell", _bounds()),
            AtomisticPeriodicCoordinateDiffusion(3, _process(9), image_extent=4),
            _score(SITES, None),
            score_id="analytic-wrapped-sites",
            rtol=1e-7,
            atol=1e-9,
            lane_chunk=lane_chunk,
        )
    support = AtomisticGenerationSupport(
        atom_capacity=2,
        species_vocabulary=(11, 17),
        vocabulary_masses=(22.99, 35.45),
        mass_provenance=MASSES,
        chemistry="variable-composition",
        length_unit=ANGSTROM,
        composition=AtomisticCompositionBounds(
            {11: (0, 2), 17: (0, 2)}, minimum_atoms=1, maximum_atoms=2
        ),
        cell_admission="fixed-cell",
        cell=np.diag([5.0, 5.0, 5.0]),
        cell_bounds=_bounds(500.0),
    )
    sites = np.asarray([[0.1, 0.1, 0.1], [0.6, 0.6, 0.6]])
    return PreparedPeriodicDiffusionSampler(
        support,
        AtomisticPeriodicCoordinateDiffusion(2, _process(6), image_extent=4),
        _score(sites, None),
        score_id="analytic-wrapped-rock-salt-sites",
        dt0=0.01,
        wiener_tolerance=1e-4,
        species=AtomisticSpeciesDiffusion(
            CategoricalDiffusionSchedule.uniform(8, 3), (11, 17), vacancy=True
        ),
        species_predictor=_species_logits,
        lane_chunk=lane_chunk,
    )


@eqx.filter_jit
def _admit(
    sampler: AbstractPreparedAtomisticSampler,
    state: AtomisticSamplerLaneState,
    mask: jax.Array,
    keys: jax.Array,
    ordinals: jax.Array,
) -> AtomisticSamplerLaneState:
    return sampler.admit(state, mask, keys, ordinals, None)


def _status_name(status: int, /) -> str:
    if status == 0:
        return "accepted"
    return "|".join(
        str(flag.name) for flag in AtomisticProposalStatus if flag and status & flag
    )


def _admission_evidence(
    state: AtomisticSamplerLaneState, mask: jax.Array, /
) -> dict[str, Any]:
    """Selected-lane outcome of one admission; vacant lanes never count as success."""
    method = state.method
    if not isinstance(method, AtomisticPeriodicSamplerMethod):
        raise TypeError("Periodic samplers carry AtomisticPeriodicSamplerMethod lanes.")
    selected = np.asarray(mask, dtype=np.bool_)
    vacant = ~selected
    active = np.asarray(state.active, dtype=np.bool_)
    terminal = np.asarray(state.terminal, dtype=np.bool_)
    status = np.asarray(method.status, dtype=np.int64)[selected]
    valid = np.asarray(method.solver_valid, dtype=np.bool_)[selected]
    detail = np.asarray(method.solver_detail).astype(np.int64)[selected]
    return {
        "selected_lanes": int(np.sum(selected)),
        "vacant_lanes": int(np.sum(vacant)),
        "vacant_lanes_occupied": int(np.sum((active | terminal)[vacant])),
        "terminal": int(np.sum(terminal[selected])),
        "solver_valid": int(np.sum(valid)),
        "accepted": int(np.sum(terminal[selected] & (status == 0))),
        "statuses": dict(Counter(_status_name(int(value)) for value in status)),
        "solver_detail": detail.tolist(),
    }


def _case(
    kind: str, lanes: int, lane_chunk: int, budget: int, repeats: int
) -> dict[str, Any]:
    sampler, prepare_seconds = measure_synchronized(lambda: _sampler(kind, lane_chunk))
    keys = jr.split(jr.key(0), lanes)
    ordinals = jnp.arange(lanes, dtype=jnp.int64)
    state = sampler.vacant(lanes)
    everything = jnp.ones((lanes,), dtype=jnp.bool_)
    compiled, compile_timing = measure_lower_and_compile(
        lambda: _admit.lower(sampler, state, everything, keys, ordinals),  # ty: ignore[unresolved-attribute]
        lambda lowered: lowered.compile(),
    )
    cost = compiled.compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    compiler = compiler_evidence(cost, compiled.compiled.memory_analysis(), source="xla")
    first_result, first = measure_synchronized(
        lambda: compiled(sampler, state, everything, keys, ordinals)
    )
    admitted = []
    for active in sorted({0, 1, max(1, lanes // 2), lanes}):
        mask = jnp.arange(lanes) < active
        warmup_result, warmup_seconds = measure_synchronized(
            lambda mask=mask: compiled(sampler, state, mask, keys, ordinals)
        )
        final_result, warm = measure_repeated(
            lambda mask=mask: compiled(sampler, state, mask, keys, ordinals),
            warmup=0,
            repeats=repeats,
        )
        admitted.append(
            {
                "admitted_lanes": active,
                "first_seconds": warmup_seconds,
                "warm": warm.to_dict(unit="seconds"),
                "first": _admission_evidence(warmup_result, mask),
                "final": _admission_evidence(final_result, mask),
            }
        )
    runtime = AtomisticGenerationStreamPlan(
        sampler,
        campaign_id=f"benchmark-{kind}",
        lanes_per_member=lanes,
        pending_capacity=max(lanes, budget),
    ).prepare()
    sink = AtomisticMemoryGenerationSink("benchmark")
    start = runtime.start(jr.key(1), target=budget, attempt_budget=budget)
    final, campaign_seconds = measure_synchronized(
        lambda: runtime.run(start, sink, maximum_transitions=4 * budget + 4)
    )
    statuses = Counter(_status_name(int(record.status)) for record in sink.records)
    return {
        "kind": kind,
        "lanes": lanes,
        "lane_chunk": lane_chunk,
        "attempt_budget": budget,
        "signature": sampler.signature_facts(),
        "solver_detail_kind": (
            "backend_successful" if sampler.detail_dtype == "bool" else "transport_status"
        ),
        "prepare_seconds": prepare_seconds,
        "compile": asdict(compile_timing),
        "compiler": asdict(compiler),
        "first_admission_seconds": first,
        "first_admission": _admission_evidence(first_result, everything),
        "warm_admission_by_admitted_lanes": admitted,
        "campaign_seconds": campaign_seconds,
        "attempts": len(sink.records),
        "accepted": statuses.get("accepted", 0),
        "shortfall": runtime.shortfall(final),
        "stop_reason": final.stop_reason.name,
        "statuses": dict(statuses),
        "logical_retained_bytes": logical_array_bytes(final),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--lanes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--lane-chunks", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.budget < 1
        or args.repeats < 1
        or any(lane < 1 for lane in args.lanes)
        or any(chunk < 1 for chunk in args.lane_chunks)
    ):
        raise ValueError("Budgets, repeats, lanes and lane chunks must be positive.")
    phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
    _jax_config.set_compilation_cache(False)
    root = Path(__file__).resolve().parents[1]
    owners = (
        "phydrax/atomistic/_diffusion.py",
        "phydrax/atomistic/_generation.py",
        "phydrax/transport/diffusion/_reverse.py",
        "phydrax/transport/diffusion/_probability_flow.py",
        "phydrax/stochastic/_wiener.py",
    )
    record = {
        "benchmark": "native-periodic-generation",
        "label": args.label,
        "environment": capture_environment().to_dict(),
        "cache_policy": {
            "persistent_compilation_cache": bool(
                _jax_config.configuration_values()["jax_enable_compilation_cache"]
            ),
            "in_process_trace_cache": True,
        },
        "identity": capture_benchmark_identity(
            root, Path(__file__), ("cases", "environment", "source_sha256")
        ).to_dict(),
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in owners
        },
        "cases": [
            _case(kind, lanes, chunk, args.budget, args.repeats)
            for kind in args.kinds
            for lanes in args.lanes
            for chunk in args.lane_chunks
            if chunk <= lanes
        ],
    }
    write_json_atomic(args.output, record)
    print(args.output)


if __name__ == "__main__":
    main()
