#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated scaling campaign for native enhanced sampling (P9).

Every compiled phase reports lowering, compilation, the first execution and
synchronized warm executions separately, with XLA cost/memory analysis.
Host transactions (preparation, history fill, freeze/digest, capacity growth,
profile integration) are timed separately.  Scenarios:

``history``
    Kernel histories alone: the deposition kernel (one simultaneous event of
    ``walkers`` kernels against the pre-event history, all-or-nothing), the
    vectorized history energy and gradient over every walker, the host fill
    loop, freeze/digest and terminal capacity growth.  Swept over history
    capacity and filled fraction, CV dimension, walker/owner count, ownership
    (shared/walker/state), storage (terminal/FIFO), tempering
    (standard/well-tempered) and the fixed-correspondence aligned-MSD kernel.
    A filled terminal history refusing a deposition is reported as an
    explicit ``refused-terminal-exhaustion`` row.
``dynamics``
    One biased walker (harmonic bond chain, velocity Verlet): the unbiased
    native step, the umbrella step and transaction, the hill-history step
    split into propagation, CV evaluation plus deposition and force refresh
    (re-prime after the history changed), the full adaptive transaction and
    the same runtime with frozen histories.  Overheads are warm-median ratios
    against the unbiased step.  A short committed run reports deposits and
    history work; when the run outlasts the history capacity the terminal
    exhaustion, its latched repeat refusal and the host capacity-growth
    transaction are reported explicitly.  Swept over atom count, CV
    support/dimension and history capacity.
``abf``
    Force-only ABF on a BAOAB chain: native propagation with and without the
    ABF force port, physical-force sampling plus accumulation, port force
    evaluation, refresh and the full transaction; a short run reports
    in-range/out-of-range/irregular sample accounting and bin occupancy, and
    the host profile integration (or its explicit refusal).  Swept over bins
    per CV, CV dimension (grid size is the product) and atom count.
``multistate``
    Bias-aware multistate segments (umbrella windows of one temperature):
    the full segment and, separately, one iteration, vectorized propagation,
    the ``K x R`` reduced-potential cross-evaluation, the label rebase,
    deposition plus force refresh and the force refresh alone.  Modes:
    ``umbrella`` (window exchange, no history), ``adaptive-<ownership>``,
    ``frozen-<ownership>`` (authenticated ``frozen:<digest>`` rows) and
    ``exhaustion-<ownership>`` (history capacity for half the iterations; the
    segment ends at the refused iteration).  ``K == R`` cases use replica
    exchange; ``K != R`` cases run independent replicas with full
    cross-evaluation.  Statistics: exchange acceptance per pair, deposited
    events and history revisions, adapting/active draw counts, digest
    authentication and reduced-potential dataset conversion (or refusal).
``refusals``
    Declared host refusals: an unbound history term, mixed history
    declarations, a declaration on a term without history, a force-only ABF
    port in a multi-state table, a mixed-temperature well-tempered history
    and a frozen declaration whose bound content does not authenticate.

Statistical fields are execution evidence for the sampled segments only.
Short runs, and the smoke sweep in particular, establish no free-energy
convergence.  Nothing is retried: refused and failed rows are recorded as
they occur.  The radius-of-gyration CV evaluation micro-benchmark is owned by
``tools/atomistic_sampling_benchmarks.py`` and is not repeated here.

Sweeps vary one axis at a time around the first value of every axis list.

Smoke run::

    python -m benchmarks.atomistic_enhanced_sampling --smoke --output /tmp/enhanced.json
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict
from operator import methodcaller
from pathlib import Path
from typing import Any, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._memory import process_peak_rss_bytes
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax import config as pxconfig
from phydrax.atomistic._force_port import AtomisticForcePortGeometry
from phydrax.atomistic._term_state import AtomisticTermStateBindings


UNITS = phx.atomistic.AtomisticUnitSystem.reduced()
BOLTZMANN = float(UNITS.boltzmann_constant)
TEMPERATURE = 1.0
BOND_STIFFNESS = 100.0
BOND_LENGTH = 1.0
ZIGZAG_ANGLE = 0.35
TIMESTEP = 2.0e-3
FRICTION = 1.0
VELOCITY_SCALE = 0.3
UMBRELLA_STIFFNESS = 10.0
WINDOW_STIFFNESS = 20.0
WINDOW_HALF_SPAN = 0.15
HILL_HEIGHT = 0.05
HILL_WIDTH = 0.1
RMSD_WIDTH = 0.2
BIAS_FACTOR = 10.0
ABF_MIN_SAMPLES = 2
ABF_FULL_SAMPLES = 4
ABF_MAX_FORCE = 50.0
QUALIFICATION_BIAS_BOUND = 1.0
MODE_PATTERN = re.compile(
    r"^(umbrella|(adaptive|frozen|exhaustion)-(shared|walker|state)):(\d+):(\d+)$"
)
CV_PATTERN = re.compile(r"^(distance|rg|distances-(\d+))$")
OWNERS = (
    "phydrax/atomistic/sampling/_abf.py",
    "phydrax/atomistic/sampling/_adaptive_profile.py",
    "phydrax/atomistic/sampling/_bias.py",
    "phydrax/atomistic/sampling/_bias_history.py",
    "phydrax/atomistic/sampling/_biased_dynamics.py",
    "phydrax/atomistic/sampling/_multistate.py",
    "phydrax/atomistic/_dynamics.py",
    "phydrax/atomistic/_thermodynamic.py",
    "phydrax/uq/_atomistic_free_energy.py",
)
EVIDENCE_FIELDS = (
    "abf",
    "all_expected_outcomes",
    "configuration",
    "dynamics",
    "environment",
    "history",
    "multistate",
    "refusals",
    "source_sha256",
    "statistical_scope",
)
SCOPE = (
    "Counts, acceptance rates, deposits and occupancies describe the sampled "
    "segments only; no free-energy convergence is claimed or implied."
)


# ------------------------------------------------------------------ measurement


class _StepRefusal(TypedDict):
    attempt: int
    reasons: list[str]


def _phase(
    function: Callable[..., Any], *arguments: Any, warmup: int, repeats: int
) -> tuple[Any, Any, dict[str, Any]]:
    """Lower, compile, run once, then time synchronized warm executions."""
    jitted = eqx.filter_jit(function)
    executable, compilation = measure_lower_and_compile(
        lambda: jitted.lower(  # ty: ignore[unresolved-attribute]
            *arguments
        ),
        methodcaller("compile"),
    )
    _, first_seconds = measure_synchronized(lambda: executable(*arguments))
    result, warm = measure_repeated(
        lambda: executable(*arguments), warmup=warmup, repeats=repeats
    )
    compiler = compiler_evidence(
        executable.compiled.cost_analysis(),
        executable.compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    return (
        result,
        executable,
        {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
            "first_seconds": first_seconds,
            "warm": warm.to_dict(),
            "compiler": asdict(compiler),
        },
    )


def _median(phase: dict[str, Any] | None) -> float | None:
    return None if phase is None else phase["warm"]["median_seconds"]


def _ratio(
    numerator: dict[str, Any] | None, denominator: dict[str, Any] | None
) -> float | None:
    top, bottom = _median(numerator), _median(denominator)
    if top is None or bottom is None or bottom <= 0.0:
        return None
    return top / bottom


def _float(value: Any) -> float | None:
    number = float(np.asarray(value))
    return number if math.isfinite(number) else None


def _refusal(error: Exception) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _reason_names(value: int) -> list[str]:
    reasons = phx.atomistic.sampling.BiasedStepReason
    return [str(flag.name) for flag in reasons if flag and value & int(flag)]


def _one_at_a_time(axes: dict[str, list[Any]], /) -> list[dict[str, Any]]:
    base = {name: values[0] for name, values in axes.items()}
    configurations = [dict(base)]
    for name, values in axes.items():
        configurations.extend(dict(base, **{name: value}) for value in values[1:])
    return configurations


def _unique(configurations: Iterable[dict[str, Any]], /) -> list[dict[str, Any]]:
    seen: set[tuple[tuple[str, Any], ...]] = set()
    unique = []
    for configuration in configurations:
        key = tuple(sorted(configuration.items()))
        if key not in seen:
            seen.add(key)
            unique.append(configuration)
    return unique


# ---------------------------------------------------------------------- fixtures


def _chain_system(atoms: int, /) -> Any:
    ids = np.arange(atoms)
    topology = phx.atomistic.MolecularTopologyPlan(
        bonds=[[index, index + 1] for index in range(atoms - 1)]
    )
    return phx.atomistic.AtomisticSystemPlan(
        ids,
        np.ones((atoms,), dtype=np.int32),
        np.ones((atoms,)),
        UNITS,
        atom_type_ids=np.zeros((atoms,), dtype=np.int32),
        topology=topology,
    ).prepare()


def _chain_positions(atoms: int, /) -> jax.Array:
    """Planar zigzag chain with every bond at its equilibrium length."""
    index = np.arange(atoms, dtype=np.float64)
    x = index * BOND_LENGTH * math.cos(ZIGZAG_ANGLE)
    y = np.where(index % 2 == 1, BOND_LENGTH * math.sin(ZIGZAG_ANGLE), 0.0)
    return jnp.asarray(np.stack((x, y, np.zeros_like(x)), axis=-1))


def _cv_program(system: Any, specification: str, atoms: int, /) -> Any:
    sampling = phx.atomistic.sampling
    kind = sampling.CollectiveVariableKind
    plan = sampling.CollectiveVariablePlan
    match = CV_PATTERN.match(specification)
    if match is None:
        raise ValueError(f"Unknown CV specification {specification!r}.")
    if specification == "distance":
        variables = (plan(kind.DISTANCE, [0, atoms - 1]).prepare(system),)
    elif specification == "rg":
        variables = (plan(kind.RADIUS_OF_GYRATION, list(range(atoms))).prepare(system),)
    else:
        count = int(match.group(2))
        if not 0 < count < atoms:
            raise ValueError(f"{specification!r} needs fewer CVs than atoms ({atoms}).")
        variables = tuple(
            plan(kind.DISTANCE, [index, index + 1]).prepare(system)
            for index in range(count)
        )
    return sampling.CollectiveVariableProgram(variables)


def _bond() -> Any:
    return phx.atomistic.HarmonicBondPotential(
        np.asarray([BOND_STIFFNESS], dtype=np.float64),
        np.asarray([BOND_LENGTH], dtype=np.float64),
    )


def _potential(system: Any, terms: tuple[Any, ...], /) -> Any:
    # Bonded and explicit-index CV terms declare no neighborhood requirement:
    # the bundle is explicitly empty.
    return phx.atomistic.AtomisticPotentialProgram(terms).prepare(
        system,
        neighborhood_bundle=phx.atomistic.AtomisticNeighborhoodBundlePlan((), (), ()),
    )


def _dynamics(
    system: Any, potential: Any, integrator: Any, /, *, force_ports: tuple[Any, ...] = ()
) -> Any:
    return phx.atomistic.AtomisticDynamicsPlan(
        system, potential, integrator, force_ports=force_ports
    ).prepare()


def _row(
    system: Any,
    ensemble: str,
    /,
    *,
    temperature: float | None = None,
    bias: Any = None,
    state_id: str | None = None,
) -> Any:
    return phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system),
        ensemble=ensemble,  # ty: ignore[invalid-argument-type]
        temperature=temperature,
        bias=bias,
        state_id=state_id,
    )


def _table(dynamics: Any, rows: tuple[Any, ...], /) -> Any:
    return phx.atomistic.PreparedThermodynamicStateTable(dynamics, rows)


def _start(
    dynamics: Any,
    table: Any,
    positions: jax.Array,
    /,
    *,
    seed: int,
    state_index: int = 0,
    term_states: Any = None,
) -> Any:
    velocity = VELOCITY_SCALE * np.random.default_rng(seed).standard_normal(
        positions.shape
    )
    return dynamics.initialize_state(
        positions,
        table,
        state_index=state_index,
        velocity=jnp.asarray(velocity),
        key=jax.random.key(seed),
        term_states=term_states,
    )


def _qualification(dynamics: Any, table: Any, /) -> Any:
    return phx.atomistic.sampling.AtomisticCanonicalSamplingQualification(
        dynamics,
        table,
        "enhanced-sampling-benchmark-demonstration-qualification",
        sampling_exact=False,
        sampling_bias_bound=QUALIFICATION_BIAS_BOUND,
    )


def _history_plan(
    dimension: int,
    capacity: int,
    /,
    *,
    ownership: str = "shared",
    owners: int = 1,
    storage: str = "terminal",
    kernel: str = "nearest-image",
    well_tempered: bool = False,
    temperature: float = TEMPERATURE,
) -> Any:
    sampling = phx.atomistic.sampling
    aligned = kernel == "aligned-msd"
    return sampling.AtomisticBiasHistoryPlan(
        dimension,
        height=HILL_HEIGHT,
        widths=RMSD_WIDTH if aligned else HILL_WIDTH,
        capacity=capacity,
        ownership=sampling.BiasHistoryOwnership(ownership),
        owner_count=owners,
        storage=sampling.BiasHistoryStorage(storage),
        kernel=sampling.HillKernel.ALIGNED_MSD
        if aligned
        else sampling.HillKernel.NEAREST_IMAGE,
        bias_factor=BIAS_FACTOR if well_tempered else None,
        temperature=temperature if well_tempered else None,
    )


# ------------------------------------------------------------ compiled phases


def _native_step(dynamics: Any, table: Any, state: Any) -> Any:
    return dynamics.step_detailed(state, table)


def _transaction(runtime: Any, state: Any) -> Any:
    return runtime.step(state)


def _refresh(runtime: Any, state: Any) -> Any:
    return runtime.dynamics.prime_state(state.dynamics, runtime.thermodynamic)


def _propagation(runtime: Any, state: Any) -> Any:
    return runtime.dynamics.step_detailed(state.dynamics, runtime.thermodynamic)


def _hill_deposition(runtime: Any, state: Any) -> Any:
    """CV evaluation plus the single-walker deposition event of the transaction."""
    (term,) = runtime.bias_terms
    dynamics_state = state.dynamics
    history = dynamics_state.term_states[term.name].history
    evaluation = term.plan.variables.evaluate(runtime.geometry(dynamics_state, term))
    return phx.atomistic.sampling.deposit_bias_history(
        term.plan.form.history,
        history,
        evaluation.values[None, :],
        runtime.owner_key(term, dynamics_state.thermodynamic_state_index)[None],
        evaluation.successful[None],
        term.plan.variables.metrics,
        boltzmann_constant=BOLTZMANN,
    )


def _deposit_history(
    plan: Any,
    state: Any,
    values: jax.Array,
    keys: jax.Array,
    active: jax.Array,
    metrics: Any,
) -> Any:
    return phx.atomistic.sampling.deposit_bias_history(
        plan, state, values, keys, active, metrics, boltzmann_constant=BOLTZMANN
    )


def _history_energy(
    plan: Any, state: Any, points: jax.Array, keys: jax.Array, metrics: Any
) -> tuple[jax.Array, jax.Array]:
    return jax.vmap(
        jax.value_and_grad(
            lambda point, key: plan.evaluate(point, metrics, state, key).energy
        )
    )(points, keys)


def _abf_sampling(runtime: Any, state: Any) -> Any:
    """Physical-force ABF sample and accumulation at a committed configuration."""
    (port,) = runtime.abf_ports
    dynamics_state = state.dynamics
    forces, _ = runtime.physical_forces(dynamics_state)
    sample = port.abf.sample(
        phx.atomistic.sampling.CollectiveVariableGeometry(
            dynamics_state.kinematics.positions
        ),
        forces,
    )
    return port.abf.accumulate(dynamics_state.term_states[port.name], sample)


def _abf_port_forces(runtime: Any, state: Any) -> Any:
    (port,) = runtime.abf_ports
    dynamics_state = state.dynamics
    positions = dynamics_state.kinematics.positions
    return port.forces(
        AtomisticForcePortGeometry(
            positions=positions,
            unwrapped_positions=positions,
            cell_vectors=None,
            image_counts=None,
            species=dynamics_state.species,
        ),
        dynamics_state.term_states[port.name],
    )


def _segment_run(segment: Any, state: Any) -> Any:
    return segment.run(state)


def _iterate(runtime: Any, state: Any) -> Any:
    return runtime.iterate(state)


def _multistate_propagation(runtime: Any, state: Any) -> Any:
    return jax.vmap(
        lambda lane: runtime.dynamics.step_detailed(lane, runtime.thermodynamic)
    )(state.dynamics)


def _cross_evaluation(runtime: Any, state: Any) -> Any:
    return runtime.reduced_potentials(state.dynamics)


def _multistate_rebase(runtime: Any, state: Any) -> Any:
    return jax.vmap(
        lambda lane, label: runtime.dynamics.rebase_thermodynamic_state(
            lane,
            runtime.thermodynamic,
            label,
            term_states=runtime._row_bindings(lane.term_states, label),
        )
    )(state.dynamics, state.state_at_replica)


def _multistate_deposition(runtime: Any, state: Any) -> Any:
    """All-walker deposition event plus the force refresh it triggers."""
    return runtime._deposit(
        state.dynamics,
        state.state_at_replica,
        state.state_at_replica,
        state.iteration_index + 1,
    )


def _multistate_refresh(runtime: Any, state: Any) -> Any:
    return jax.vmap(
        lambda lane: runtime.dynamics.prime_state(lane, runtime.thermodynamic)
    )(state.dynamics)


# ------------------------------------------------------------------- history


def _history_configurations(arguments: argparse.Namespace, /) -> list[dict[str, Any]]:
    axes = {
        "capacity": arguments.history_capacities,
        "fill": arguments.history_fill,
        "dimension": arguments.history_dimensions,
        "walkers": arguments.history_walkers,
        "ownership": arguments.history_ownerships,
        "storage": arguments.history_storages,
        "tempering": arguments.history_temperings,
    }
    configurations = [
        dict(value, kernel="nearest-image", rmsd_atoms=None)
        for value in _one_at_a_time(axes)
    ]
    base = configurations[0]
    configurations.extend(
        dict(base, storage=storage, fill=fill)
        for storage in axes["storage"]
        for fill in axes["fill"]
    )
    configurations.extend(
        dict(
            base,
            kernel="aligned-msd",
            rmsd_atoms=atoms,
            dimension=3 * atoms,
            walkers=1,
            ownership="shared",
            storage="terminal",
            tempering="standard",
            fill=fill,
        )
        for atoms in arguments.rmsd_atoms
        for fill in axes["fill"]
    )
    return _unique(configurations)


def _history_reference(state: Any, point: np.ndarray, key: int, shared: bool, /) -> float:
    """Independent NumPy sum of Gaussian kernels (nonperiodic metric, no ramp)."""
    centers = np.asarray(state.centers)
    heights = np.asarray(state.heights)
    live = np.asarray(state.valid)
    if not shared:
        live = live & (np.asarray(state.owners) == key)
    scaled = (point[None, :] - centers) / HILL_WIDTH
    return float(
        np.sum(np.where(live, heights * np.exp(-0.5 * np.sum(scaled**2, axis=-1)), 0.0))
    )


def _history_case(
    configuration: dict[str, Any], /, *, warmup: int, repeats: int, seed: int
) -> dict[str, Any]:
    sampling = phx.atomistic.sampling
    capacity = int(configuration["capacity"])
    walkers = int(configuration["walkers"])
    dimension = int(configuration["dimension"])
    shared = configuration["ownership"] == "shared"
    aligned = configuration["kernel"] == "aligned-msd"
    fifo = configuration["storage"] == "fifo"
    owners = 1 if shared else walkers
    row: dict[str, Any] = {
        "scenario": "history",
        "configuration": dict(configuration, owner_count=owners),
    }
    expected_refusal = fifo and capacity < walkers
    try:
        plan, prepare_seconds = measure_host(
            lambda: _history_plan(
                dimension,
                capacity,
                ownership=configuration["ownership"],
                owners=owners,
                storage=configuration["storage"],
                kernel=configuration["kernel"],
                well_tempered=configuration["tempering"] == "well-tempered",
            )
        )
        metrics = (
            ()
            if aligned
            else tuple(sampling.CollectiveVariableMetric() for _ in range(dimension))
        )
        keys = (
            jnp.zeros((walkers,), dtype=jnp.int32)
            if shared
            else jnp.arange(walkers, dtype=jnp.int32)
        )
        active = jnp.ones((walkers,), dtype=jnp.bool_)
        rng = np.random.default_rng(seed)
        events = (
            min(capacity, walkers * round(configuration["fill"] * capacity / walkers))
            // walkers
        )
        draws = [
            jnp.asarray(
                rng.standard_normal((walkers, dimension)) * (0.3 if aligned else 1.0)
            )
            for _ in range(events + 2)
        ]
        empty = plan.initialize(jnp.float64)
        _, deposit_executable, deposit_phase = _phase(
            _deposit_history,
            plan,
            empty,
            draws[0],
            keys,
            active,
            metrics,
            warmup=warmup,
            repeats=repeats,
        )
    except ValueError as error:
        return {
            **row,
            "status": "refused",
            "refusal": _refusal(error),
            "gates": {"outcome_matches_policy": expected_refusal},
        }

    def fill() -> Any:
        state = empty
        for values in draws[:events]:
            state = deposit_executable(plan, state, values, keys, active, metrics).state
        return state

    filled, fill_seconds = measure_synchronized(fill)
    probe = deposit_executable(plan, filled, draws[events], keys, active, metrics)
    points = draws[events + 1]
    (energies, gradients), _, energy_phase = _phase(
        _history_energy,
        plan,
        filled,
        points,
        keys,
        metrics,
        warmup=warmup,
        repeats=repeats,
    )
    frozen, freeze_seconds = measure_host(
        lambda: sampling.FrozenBiasHistory(plan, filled)
    )
    growth = None
    if not fifo:
        (grown_plan, grown_state), grow_seconds = measure_synchronized(
            lambda: sampling.grow_bias_history(plan, filled, 2 * capacity)
        )
        growth = {
            "seconds": grow_seconds,
            "capacity": grown_plan.capacity,
            "scientific_id_preserved": grown_plan.scientific_id == plan.scientific_id,
            "written_preserved": int(grown_state.written) == int(filled.written),
        }
    written = int(filled.written)
    inserted = bool(probe.inserted)
    exhausted = bool(probe.exhausted)
    expected_exhaustion = (not fifo) and written + walkers > capacity
    reference_error = None
    if not aligned:
        reference = _history_reference(
            filled, np.asarray(points[0]), int(keys[0]), shared
        )
        reference_error = abs(float(energies[0]) - reference) / max(1.0, abs(reference))
    history_bytes = logical_array_bytes(filled)
    gates = {
        "outcome_matches_policy": (
            (exhausted and not inserted)
            if expected_exhaustion
            else (inserted and not exhausted)
        ),
        "energy_finite": bool(jnp.all(jnp.isfinite(energies))),
        "gradient_finite": bool(jnp.all(jnp.isfinite(gradients))),
        "fill_written": written == events * walkers,
    }
    if reference_error is not None:
        gates["energy_matches_numpy_reference"] = reference_error <= 1.0e-10
    if growth is not None:
        gates["growth_preserves_content"] = (
            growth["scientific_id_preserved"] and growth["written_preserved"]
        )
    return {
        **row,
        "status": "refused-terminal-exhaustion" if exhausted else "ok",
        "prepare_seconds": prepare_seconds,
        "phases": {"deposition": deposit_phase, "energy_and_gradient": energy_phase},
        "host": {
            "fill_events": events,
            "fill_seconds": fill_seconds,
            "freeze_digest_seconds": freeze_seconds,
            "capacity_growth": growth,
        },
        "evidence": {
            "filled_kernels": written,
            "valid_kernels": int(jnp.sum(filled.valid)),
            "deposition_events": int(filled.deposits),
            "probe_inserted": inserted,
            "probe_exhausted": exhausted,
            "probe_height_minimum": _float(jnp.min(probe.heights)),
            "probe_height_maximum": _float(jnp.max(probe.heights)),
            "energy_mean": _float(jnp.mean(energies)),
            "gradient_norm_maximum": _float(
                jnp.max(jnp.linalg.norm(gradients.reshape((walkers, -1)), axis=-1))
            ),
            "energy_reference_relative_error": reference_error,
            "fifo_overwritten_after_probe": bool(
                fifo and int(probe.state.written) > capacity
            ),
            "frozen_digest": frozen.digest,
        },
        "memory": {
            "history_logical_bytes": history_bytes,
            "bytes_per_kernel_slot": history_bytes / capacity,
            "process_peak_rss_bytes": process_peak_rss_bytes(),
        },
        "gates": gates,
    }


# ------------------------------------------------------------------ dynamics


# Keep construction, cold/warm execution and the accepted-state controls in one
# scenario so no timing phase accidentally receives another model or initial state.
# The size is explicit setup/reporting, not a second dynamics implementation.
def _dynamics_case(
    configuration: dict[str, Any], /, *, steps: int, warmup: int, repeats: int, seed: int
) -> dict[str, Any]:
    sampling = phx.atomistic.sampling
    atoms = int(configuration["atoms"])
    capacity = int(configuration["capacity"])

    def build() -> dict[str, Any]:
        system = _chain_system(atoms)
        variables = _cv_program(system, configuration["cv"], atoms)
        positions = _chain_positions(atoms)
        center = np.asarray(
            variables.evaluate(sampling.CollectiveVariableGeometry(positions)).values
        )
        rows = (_row(system, "nve"),)
        integrator = phx.atomistic.VelocityVerletPlan(TIMESTEP)
        unbiased = _dynamics(system, _potential(system, (_bond(),)), integrator)
        umbrella_term = sampling.AtomisticBiasTerm(
            variables,
            sampling.HarmonicBiasForm(center, UMBRELLA_STIFFNESS),
            name="umbrella",
            force_group=1,
        )
        umbrella = _dynamics(
            system, _potential(system, (_bond(), umbrella_term)), integrator
        )
        hills_term = sampling.AtomisticBiasTerm(
            variables,
            sampling.HillHistoryBiasForm(_history_plan(variables.output_size, capacity)),
            name="hills",
            force_group=1,
        )
        hills = _dynamics(system, _potential(system, (_bond(), hills_term)), integrator)
        unbiased_table = _table(unbiased, rows)
        umbrella_table = _table(umbrella, rows)
        hills_table = _table(hills, rows)
        umbrella_runtime = sampling.PreparedBiasedDynamics(
            umbrella, umbrella_table, physical_force_groups=(0,)
        )
        hills_runtime = sampling.PreparedBiasedDynamics(
            hills, hills_table, physical_force_groups=(0,)
        )
        frozen_runtime = sampling.PreparedBiasedDynamics(
            hills, hills_table, physical_force_groups=(0,), adaptive=False
        )
        return {
            "variables": variables,
            "rows": rows,
            "unbiased": unbiased,
            "unbiased_table": unbiased_table,
            "unbiased_state": _start(unbiased, unbiased_table, positions, seed=seed),
            "umbrella_runtime": umbrella_runtime,
            "umbrella_state": umbrella_runtime.initialize(
                _start(umbrella, umbrella_table, positions, seed=seed)
            ),
            "hills_runtime": hills_runtime,
            "hills_state": hills_runtime.initialize(
                _start(hills, hills_table, positions, seed=seed)
            ),
            "frozen_runtime": frozen_runtime,
            "frozen_state": frozen_runtime.initialize(
                _start(hills, hills_table, positions, seed=seed)
            ),
        }

    fixture, prepare_seconds = measure_synchronized(build)
    variables = fixture["variables"]
    hills_runtime = fixture["hills_runtime"]
    hills_state = fixture["hills_state"]
    umbrella_runtime = fixture["umbrella_runtime"]
    phases: dict[str, Any] = {}
    unbiased_result, _, phases["unbiased_step"] = _phase(
        _native_step,
        fixture["unbiased"],
        fixture["unbiased_table"],
        fixture["unbiased_state"],
        warmup=warmup,
        repeats=repeats,
    )
    _, _, phases["umbrella_step"] = _phase(
        _native_step,
        umbrella_runtime.dynamics,
        umbrella_runtime.thermodynamic,
        fixture["umbrella_state"].dynamics,
        warmup=warmup,
        repeats=repeats,
    )
    umbrella_result, _, phases["umbrella_transaction"] = _phase(
        _transaction,
        umbrella_runtime,
        fixture["umbrella_state"],
        warmup=warmup,
        repeats=repeats,
    )
    _, _, phases["hills_propagation"] = _phase(
        _propagation, hills_runtime, hills_state, warmup=warmup, repeats=repeats
    )
    deposition, _, phases["hills_cv_and_deposition"] = _phase(
        _hill_deposition, hills_runtime, hills_state, warmup=warmup, repeats=repeats
    )
    _, _, phases["hills_refresh"] = _phase(
        _refresh, hills_runtime, hills_state, warmup=warmup, repeats=repeats
    )
    _, transaction, phases["hills_transaction"] = _phase(
        _transaction, hills_runtime, hills_state, warmup=warmup, repeats=repeats
    )
    frozen_result, _, phases["frozen_transaction"] = _phase(
        _transaction,
        fixture["frozen_runtime"],
        fixture["frozen_state"],
        warmup=warmup,
        repeats=repeats,
    )
    baseline = phases["unbiased_step"]
    overhead = {
        f"{name}_over_unbiased_step": _ratio(phase, baseline)
        for name, phase in phases.items()
        if name != "unbiased_step"
    }

    # Committed run with the compiled transaction; it stops at the first refusal.
    (term,) = hills_runtime.bias_terms
    state = hills_state
    committed = 0
    refusal: _StepRefusal | None = None
    run_started_history_work = float(state.history_work)
    for attempt in range(1, steps + 1):
        evaluation = transaction(hills_runtime, state)
        state = evaluation.accepted_state
        if not bool(evaluation.successful):
            refusal = {
                "attempt": attempt,
                "reasons": _reason_names(int(evaluation.reasons)),
            }
            break
        committed += 1
    history = state.dynamics.term_states[term.name].history
    statistics = {
        "attempted_steps": committed + (refusal is not None),
        "committed_steps": committed,
        "deposition_events": int(state.deposition_events),
        "history_written": int(history.written),
        "history_work": _float(state.history_work - run_started_history_work),
        "final_cv": [
            float(value)
            for value in np.asarray(
                variables.evaluate(hills_runtime.geometry(state.dynamics, term)).values
            )
        ],
    }
    expected_exhaustion = steps > capacity
    exhausted = refusal is not None and "HISTORY_EXHAUSTED" in refusal["reasons"]
    exhaustion = None
    gates = {
        "unbiased_step_successful": bool(unbiased_result.successful),
        "umbrella_transaction_successful": bool(umbrella_result.successful),
        "frozen_transaction_deposits_nothing": bool(frozen_result.successful)
        and int(frozen_result.accepted_state.deposition_events) == 0,
        "single_walker_deposit_inserted": bool(deposition.inserted),
        "deposits_match_committed_steps": int(state.deposition_events) == committed,
        "exhaustion_matches_capacity": (
            refusal is not None
            and refusal["attempt"] == capacity + 1
            and "HISTORY_EXHAUSTED" in refusal["reasons"]
        )
        if expected_exhaustion
        else refusal is None and committed == steps,
    }
    if exhausted:
        repeat = transaction(hills_runtime, state)
        latched = {
            "refused": not bool(repeat.successful),
            "reasons": _reason_names(int(repeat.reasons)),
            "step_unchanged": int(repeat.accepted_state.dynamics.step_index)
            == int(state.dynamics.step_index),
            "deposits_unchanged": int(repeat.accepted_state.deposition_events)
            == int(state.deposition_events),
            "random_key_unchanged": bool(
                np.array_equal(
                    np.asarray(repeat.accepted_state.dynamics.random_key),
                    np.asarray(state.dynamics.random_key),
                )
            ),
        }
        (grown_runtime, grown_state), growth_seconds = measure_synchronized(
            lambda: phx.atomistic.sampling.grow_bias_history_capacity(
                hills_runtime, state, term.name, 2 * capacity, fixture["rows"]
            )
        )
        resumed, _, growth_phase = _phase(
            _transaction, grown_runtime, grown_state, warmup=warmup, repeats=repeats
        )
        exhaustion = {
            "status": "refused-terminal-exhaustion",
            "first_refusal": refusal,
            "latched_repeat": latched,
            "capacity_growth_seconds": growth_seconds,
            "grown_capacity": 2 * capacity,
            "post_growth_transaction": growth_phase,
            "post_growth_successful": bool(resumed.successful),
        }
        gates["latched_refusal_commits_nothing"] = (
            latched["refused"]
            and latched["step_unchanged"]
            and latched["deposits_unchanged"]
            and latched["random_key_unchanged"]
        )
        gates["post_growth_successful"] = bool(resumed.successful)
    elif expected_exhaustion:
        exhaustion = {"status": "missing-expected-exhaustion", "first_refusal": refusal}
    freeze, freeze_seconds = measure_host(lambda: hills_runtime.freeze(state))
    return {
        "scenario": "dynamics",
        "configuration": {
            **configuration,
            "cv_dimension": variables.output_size,
            "cv_support": len(variables.support),
            "steps": steps,
            "integrator": "velocity-verlet",
            "timestep": TIMESTEP,
        },
        "status": (
            "refused-terminal-exhaustion"
            if exhausted
            else "failed"
            if refusal is not None
            else "ok"
        ),
        "prepare_seconds": prepare_seconds,
        "phases": phases,
        "overhead": overhead,
        "statistics": statistics,
        "exhaustion": exhaustion,
        "host": {
            "freeze_digest_seconds": freeze_seconds,
            "frozen_digests": {key: value.digest for key, value in freeze.items()},
        },
        "memory": {
            "unbiased_state_logical_bytes": logical_array_bytes(
                fixture["unbiased_state"]
            ),
            "hills_state_logical_bytes": logical_array_bytes(hills_state),
            "history_logical_bytes": logical_array_bytes(history),
            "process_peak_rss_bytes": process_peak_rss_bytes(),
        },
        "gates": gates,
    }


# ----------------------------------------------------------------------- ABF


def _abf_variables(
    system: Any, atoms: int, dimension: int, /
) -> tuple[Any, list[float], list[float]]:
    sampling = phx.atomistic.sampling
    kind = sampling.CollectiveVariableKind
    plan = sampling.CollectiveVariablePlan
    end = plan(kind.DISTANCE, [0, atoms - 1]).prepare(system)
    end_range = (0.3, 1.05 * (atoms - 1) * BOND_LENGTH)
    if dimension == 1:
        return sampling.CollectiveVariableProgram((end,)), [end_range[0]], [end_range[1]]
    bond = plan(kind.DISTANCE, [0, 1]).prepare(system)
    return (
        sampling.CollectiveVariableProgram((end, bond)),
        [end_range[0], 0.85 * BOND_LENGTH],
        [end_range[1], 1.15 * BOND_LENGTH],
    )


def _abf_case(
    configuration: dict[str, Any], /, *, steps: int, warmup: int, repeats: int, seed: int
) -> dict[str, Any]:
    sampling = phx.atomistic.sampling
    atoms = int(configuration["atoms"])
    bins = int(configuration["bins"])
    dimension = int(configuration["dimension"])

    def build() -> dict[str, Any]:
        system = _chain_system(atoms)
        variables, minimum, maximum = _abf_variables(system, atoms, dimension)
        abf = sampling.AtomisticABFPlan(
            variables,
            minimum=np.asarray(minimum, dtype=np.float64),
            maximum=np.asarray(maximum, dtype=np.float64),
            bins=bins,
            temperature=TEMPERATURE,
            min_samples=ABF_MIN_SAMPLES,
            full_samples=ABF_FULL_SAMPLES,
            max_force=ABF_MAX_FORCE,
        ).prepare(system)
        port = sampling.AtomisticABFForcePort(abf, name="abf")
        positions = _chain_positions(atoms)
        rows = (_row(system, "nvt", temperature=TEMPERATURE),)
        integrator = phx.atomistic.BAOABLangevinPlan(TIMESTEP, FRICTION)
        potential = _potential(system, (_bond(),))
        unbiased = _dynamics(system, potential, integrator)
        biased = _dynamics(system, potential, integrator, force_ports=(port,))
        unbiased_table = _table(unbiased, rows)
        biased_table = _table(biased, rows)
        runtime = sampling.PreparedBiasedDynamics(
            biased, biased_table, physical_force_groups=(0,)
        )
        return {
            "abf": abf,
            "port": port,
            "unbiased": unbiased,
            "unbiased_table": unbiased_table,
            "unbiased_state": _start(unbiased, unbiased_table, positions, seed=seed),
            "runtime": runtime,
            "state": runtime.initialize(
                _start(biased, biased_table, positions, seed=seed)
            ),
        }

    fixture, prepare_seconds = measure_synchronized(build)
    runtime, initial, abf, port = (
        fixture["runtime"],
        fixture["state"],
        fixture["abf"],
        fixture["port"],
    )
    phases: dict[str, Any] = {}
    _, _, phases["unbiased_step"] = _phase(
        _native_step,
        fixture["unbiased"],
        fixture["unbiased_table"],
        fixture["unbiased_state"],
        warmup=warmup,
        repeats=repeats,
    )
    _, _, phases["abf_propagation"] = _phase(
        _propagation, runtime, initial, warmup=warmup, repeats=repeats
    )
    _, _, phases["abf_sample_and_accumulate"] = _phase(
        _abf_sampling, runtime, initial, warmup=warmup, repeats=repeats
    )
    _, _, phases["abf_port_forces"] = _phase(
        _abf_port_forces, runtime, initial, warmup=warmup, repeats=repeats
    )
    _, _, phases["abf_refresh"] = _phase(
        _refresh, runtime, initial, warmup=warmup, repeats=repeats
    )
    _, transaction, phases["abf_transaction"] = _phase(
        _transaction, runtime, initial, warmup=warmup, repeats=repeats
    )
    overhead = {
        f"{name}_over_unbiased_step": _ratio(phase, phases["unbiased_step"])
        for name, phase in phases.items()
        if name != "unbiased_step"
    }
    state = initial
    committed = 0
    failure = None
    for attempt in range(1, steps + 1):
        evaluation = transaction(runtime, state)
        state = evaluation.accepted_state
        if not bool(evaluation.successful):
            failure = {
                "attempt": attempt,
                "reasons": _reason_names(int(evaluation.reasons)),
            }
            break
        committed += 1
    estimator = state.dynamics.term_states[port.name]
    counts = np.asarray(estimator.counts)
    in_range = int(state.abf_samples)
    out_of_range = int(estimator.out_of_range)
    irregular = int(estimator.irregular)
    try:
        profile, profile_seconds = measure_host(
            lambda: sampling.abf_profile(abf, estimator)
        )
        profile_record = {
            "status": "ok",
            "seconds": profile_seconds,
            "formula_admitted": profile.formula_admitted,
            "connected": profile.connected,
            "refusals": list(profile.refusals),
            "supported_bins": int(np.sum(np.asarray(profile.support))),
            "integrability_residual": _float(profile.integrability_residual),
        }
    except ValueError as error:
        profile_record = {"status": "refused", "refusal": _refusal(error)}
    return {
        "scenario": "abf",
        "configuration": {
            **configuration,
            "grid_size": abf.plan.bin_count,
            "closure_atoms": int(abf.closure.shape[0]),
            "steps": steps,
            "integrator": "baoab-langevin",
            "timestep": TIMESTEP,
            "temperature": TEMPERATURE,
        },
        "status": "failed" if failure is not None else "ok",
        "prepare_seconds": prepare_seconds,
        "phases": phases,
        "overhead": overhead,
        "statistics": {
            "committed_steps": committed,
            "failure": failure,
            "in_range_regular_samples": in_range,
            "out_of_range_samples": out_of_range,
            "irregular_samples": irregular,
            "occupied_bins": int(np.sum(counts > 0)),
            "bins_at_full_ramp": int(np.sum(counts >= ABF_FULL_SAMPLES)),
            "maximum_bin_count": int(np.max(counts)),
            "estimator_revision": int(estimator.revision),
        },
        "profile": profile_record,
        "memory": {
            "estimator_logical_bytes": logical_array_bytes(estimator),
            "biased_state_logical_bytes": logical_array_bytes(initial),
            "unbiased_state_logical_bytes": logical_array_bytes(
                fixture["unbiased_state"]
            ),
            "process_peak_rss_bytes": process_peak_rss_bytes(),
        },
        "gates": {
            "all_steps_committed": failure is None and committed == steps,
            "sample_accounting_closes": in_range + out_of_range + irregular == committed,
            "counts_match_samples": int(np.sum(counts)) == in_range,
        },
    }


# ----------------------------------------------------------------- multistate


def _parse_mode(value: str, /) -> dict[str, Any]:
    match = MODE_PATTERN.match(value)
    if match is None:
        raise ValueError(
            f"Multistate case {value!r} must be 'umbrella:K:R' or "
            "'<adaptive|frozen|exhaustion>-<shared|walker|state>:K:R'."
        )
    states, replicas = int(match.group(4)), int(match.group(5))
    if states < 2 or replicas < 1:
        raise ValueError("Multistate cases need K >= 2 windows and R >= 1 replicas.")
    return {
        "mode": match.group(1),
        "family": match.group(2) or "umbrella",
        "ownership": match.group(3),
        "states": states,
        "replicas": replicas,
    }


def _frozen_bindings(dynamics: Any, slot: str, history: Any, /) -> Any:
    initial = dynamics.initial_term_states(jnp.float64)
    bindings = dict(zip(initial.slots, initial.values, strict=True))
    bindings[slot] = phx.atomistic.sampling.with_bias_history(bindings[slot], history)
    return AtomisticTermStateBindings(bindings)


def _multistate_fixture(
    case: dict[str, Any], /, *, iterations: int, atoms: int, seed: int
) -> dict[str, Any]:
    sampling = phx.atomistic.sampling
    states, replicas, family = case["states"], case["replicas"], case["family"]
    system = _chain_system(atoms)
    positions = _chain_positions(atoms)
    variables = _cv_program(system, "distance", atoms)
    origin = float(
        variables.evaluate(sampling.CollectiveVariableGeometry(positions)).values[0]
    )
    centers = origin + np.linspace(-WINDOW_HALF_SPAN, WINDOW_HALF_SPAN, states)
    umbrella = sampling.AtomisticBiasTerm(
        variables,
        sampling.HarmonicBiasForm(
            np.asarray([origin], dtype=np.float64), WINDOW_STIFFNESS
        ),
        name="umbrella",
        force_group=1,
    )
    terms: list[Any] = [_bond(), umbrella]
    hill = frozen = capacity = declaration = None
    if family != "umbrella":
        owners = {"shared": 1, "walker": replicas, "state": states}[case["ownership"]]
        # Every iteration deposits one kernel per replica (interval 1).
        capacity = replicas * (
            max(1, iterations // 2) if family == "exhaustion" else iterations
        )
        history_plan = _history_plan(
            1, capacity, ownership=case["ownership"], owners=owners
        )
        hill = sampling.AtomisticBiasTerm(
            variables,
            sampling.HillHistoryBiasForm(history_plan),
            name="hills",
            force_group=2,
        )
        terms.append(hill)
        declaration = "adaptive"
        if family == "frozen":
            filled = min(capacity, 2 * states)
            seeded = sampling.deposit_bias_history(
                history_plan,
                history_plan.initialize(jnp.float64),
                jnp.asarray(np.linspace(centers[0], centers[-1], filled))[:, None],
                jnp.zeros((filled,), dtype=jnp.int32),
                jnp.ones((filled,), dtype=jnp.bool_),
                variables.metrics,
                boltzmann_constant=BOLTZMANN,
            )
            frozen = sampling.FrozenBiasHistory(history_plan, seeded.state)
            declaration = frozen.declaration
    histories: dict[str, str] | None = None
    if hill is not None:
        if declaration is None:
            raise RuntimeError("A history-bearing benchmark requires its declaration.")
        histories = {hill.name: declaration}
    potential = _potential(system, tuple(terms))
    dynamics = _dynamics(
        system, potential, phx.atomistic.BAOABLangevinPlan(TIMESTEP, FRICTION)
    )
    rows = tuple(
        _row(
            system,
            "nvt",
            temperature=TEMPERATURE,
            state_id=f"window-{index}",
            bias=sampling.AtomisticBiasRowBinding(
                {umbrella.name: np.asarray([float(centers[index])], dtype=np.float64)},
                owner_keys=(
                    {hill.name: index}
                    if hill is not None and case["ownership"] == "state"
                    else None
                ),
                histories=histories,
            ),
        )
        for index in range(states)
    )
    table = _table(dynamics, rows)
    exchange = sampling.AtomisticReplicaExchangePlan(1) if states == replicas else None
    runtime = sampling.AtomisticMultistatePlan(
        table,
        np.arange(replicas),
        qualification=_qualification(dynamics, table),
        exchange=exchange,
        run_id=f"enhanced-sampling-{case['mode']}-{states}-{replicas}",
    ).prepare(dynamics)
    labels = np.arange(replicas) % states
    term_states = (
        None
        if frozen is None or hill is None
        else _frozen_bindings(dynamics, hill.name, frozen.state)
    )
    lanes = tuple(
        _start(
            dynamics,
            table,
            positions,
            seed=seed + replica,
            state_index=int(labels[replica]),
            term_states=term_states,
        )
        for replica in range(replicas)
    )
    initial = runtime.initialize(lanes, labels, jax.random.key(seed))
    segment = sampling.AtomisticMultistateSegmentPlan(
        runtime, iterations, 0, 0, runtime.initial_continuation_id
    )
    return {
        "runtime": runtime,
        "initial": initial,
        "segment": segment,
        "capacity": capacity,
        "declaration": declaration,
        "exchange": exchange is not None,
    }


def _multistate_statistics(result: Any, /) -> dict[str, Any]:
    count = int(result.count)
    valid = np.asarray(result.iteration_valid, dtype=np.bool_)
    attempted = np.asarray(result.exchange_attempted).sum(axis=0)
    accepted = np.asarray(result.exchange_accepted).sum(axis=0)
    deposited = np.asarray(result.bias_deposited)
    revisions = np.asarray(result.bias_history_revision)
    exhausted = np.flatnonzero(np.asarray(result.bias_exhausted))
    coverage = np.asarray(result.coverage)[valid]
    histories = [
        {
            "term_id": term_id,
            "declaration": declaration,
            "deposition_events": int(np.sum(deposited[:, position])),
            "final_revision": int(revisions[count - 1, position]) if count else None,
        }
        for position, (term_id, declaration) in enumerate(
            zip(
                result.bias_history_term_ids,
                result.bias_history_declarations,
                strict=True,
            )
        )
    ]
    try:
        digests: dict[str, Any] = {
            "status": "ok",
            "digests": list(result.bias_history_digests()),
        }
    except ValueError as error:
        digests = {"status": "refused", "refusal": _refusal(error)}
    try:
        dataset, conversion_seconds = measure_host(
            lambda: phx.uq.reduced_potential_dataset_from_multistate(result)
        )
        conversion: dict[str, Any] = {
            "status": "ok",
            "seconds": conversion_seconds,
            "active_samples": int(np.sum(np.asarray(dataset.sample_active))),
        }
    except ValueError as error:
        conversion = {"status": "refused", "refusal": _refusal(error)}
    return {
        "committed_iterations": count,
        "successful": bool(result.successful),
        "exhausted_at_iteration": int(exhausted[0]) if exhausted.size else None,
        "exchange_attempts_per_pair": attempted.astype(int).tolist(),
        "exchange_acceptance_per_pair": [
            None if tried == 0 else float(taken) / float(tried)
            for tried, taken in zip(attempted, accepted, strict=True)
        ],
        "active_draws": int(np.sum(np.asarray(result.sample_active))),
        "adapting_draws": int(np.sum(np.asarray(result.bias_adapting))),
        "cross_evaluation_coverage": _float(np.mean(coverage)) if coverage.size else None,
        "histories": histories,
        "history_digests": digests,
        "dataset_conversion": conversion,
    }


def _multistate_case(
    value: str, /, *, iterations: int, atoms: int, warmup: int, repeats: int, seed: int
) -> dict[str, Any]:
    case = _parse_mode(value)
    states, replicas, family = case["states"], case["replicas"], case["family"]
    fixture, prepare_seconds = measure_synchronized(
        lambda: _multistate_fixture(case, iterations=iterations, atoms=atoms, seed=seed)
    )
    runtime, initial = fixture["runtime"], fixture["initial"]
    phases: dict[str, Any] = {}
    cross = None
    result, _, phases["segment"] = _phase(
        _segment_run, fixture["segment"], initial, warmup=warmup, repeats=repeats
    )
    if family != "exhaustion":
        _, _, phases["iteration"] = _phase(
            _iterate, runtime, initial, warmup=warmup, repeats=repeats
        )
        _, _, phases["propagation"] = _phase(
            _multistate_propagation, runtime, initial, warmup=warmup, repeats=repeats
        )
        cross, _, phases["cross_evaluation"] = _phase(
            _cross_evaluation, runtime, initial, warmup=warmup, repeats=repeats
        )
        _, _, phases["rebase"] = _phase(
            _multistate_rebase, runtime, initial, warmup=warmup, repeats=repeats
        )
        _, _, phases["refresh"] = _phase(
            _multistate_refresh, runtime, initial, warmup=warmup, repeats=repeats
        )
        if family == "adaptive":
            _, _, phases["deposition_and_refresh"] = _phase(
                _multistate_deposition, runtime, initial, warmup=warmup, repeats=repeats
            )
    statistics = _multistate_statistics(result)
    segment_seconds = _median(phases["segment"])
    evaluations = states * replicas * statistics["committed_iterations"]
    expected_draws = replicas * iterations
    history = statistics["histories"][0] if statistics["histories"] else None
    gates: dict[str, bool] = {}
    if family == "exhaustion":
        stop = max(1, iterations // 2)
        gates["terminal_exhaustion_ends_segment"] = (
            not statistics["successful"]
            and statistics["committed_iterations"] == stop
            and statistics["exhausted_at_iteration"] == stop
        )
        gates["unsuccessful_segment_conversion_refused"] = (
            statistics["dataset_conversion"]["status"] == "refused"
        )
        status = "refused-terminal-exhaustion"
    else:
        gates["segment_complete"] = (
            statistics["successful"] and statistics["committed_iterations"] == iterations
        )
        gates["cross_evaluation_covered"] = cross is not None and bool(
            jnp.all(cross.successful)
        )
        if family == "adaptive":
            gates["adaptive_draws_inactive"] = (
                statistics["active_draws"] == 0
                and statistics["adapting_draws"] == expected_draws
            )
            gates["one_event_per_iteration"] = (
                history is not None
                and history["deposition_events"] == iterations
                and history["final_revision"] == expected_draws
            )
        else:
            gates["fixed_hamiltonian_draws_active"] = (
                statistics["active_draws"] == expected_draws
                and statistics["adapting_draws"] == 0
            )
            gates["dataset_converted"] = (
                statistics["dataset_conversion"]["status"] == "ok"
            )
        if family == "frozen":
            gates["frozen_content_authenticated"] = (
                statistics["history_digests"]["status"] == "ok"
                and fixture["declaration"]
                == f"frozen:{statistics['history_digests']['digests'][0]}"
                and history is not None
                and history["deposition_events"] == 0
            )
        status = "ok" if gates["segment_complete"] else "failed"
    return {
        "scenario": "multistate",
        "configuration": {
            **case,
            "iterations": iterations,
            "atoms": atoms,
            "replica_exchange": fixture["exchange"],
            "history_capacity": fixture["capacity"],
            "cross_evaluation_capacity": states * replicas,
            "temperature": TEMPERATURE,
            "timestep": TIMESTEP,
        },
        "status": status,
        "prepare_seconds": prepare_seconds,
        "phases": phases,
        "performance": {
            "reduced_potentials_per_second": (
                evaluations / segment_seconds if segment_seconds else None
            ),
            "committed_draws_per_second": (
                replicas * statistics["committed_iterations"] / segment_seconds
                if segment_seconds
                else None
            ),
        },
        "statistics": statistics,
        "memory": {
            "initial_logical_bytes": logical_array_bytes(initial),
            "segment_logical_bytes": logical_array_bytes(result),
            "process_peak_rss_bytes": process_peak_rss_bytes(),
        },
        "gates": gates,
    }


# ------------------------------------------------------------------ refusals


def _expect_refusal(
    name: str, stage: str, call: Callable[[], object], /
) -> dict[str, Any]:
    record: dict[str, Any] = {"scenario": "refusals", "probe": name, "stage": stage}
    try:
        _, seconds = measure_host(call)
    except ValueError as error:
        return {
            **record,
            "status": "refused",
            "refusal": _refusal(error),
            "gates": {"refused": True},
        }
    return {
        **record,
        "status": "admitted",
        "seconds": seconds,
        "gates": {"refused": False},
    }


def _refusal_probes(*, atoms: int, seed: int) -> list[dict[str, Any]]:
    sampling = phx.atomistic.sampling
    system = _chain_system(atoms)
    positions = _chain_positions(atoms)
    variables = _cv_program(system, "distance", atoms)
    integrator = phx.atomistic.BAOABLangevinPlan(TIMESTEP, FRICTION)

    def hill_dynamics(history: Any) -> tuple[Any, Any]:
        term = sampling.AtomisticBiasTerm(
            variables, sampling.HillHistoryBiasForm(history), name="hills", force_group=1
        )
        return term, _dynamics(system, _potential(system, (_bond(), term)), integrator)

    def multistate(dynamics: Any, rows: tuple[Any, ...]) -> Callable[[], object]:
        table = _table(dynamics, rows)
        plan = sampling.AtomisticMultistatePlan(
            table,
            np.arange(len(rows)),
            qualification=_qualification(dynamics, table),
            run_id="enhanced-sampling-refusal-probe",
        )
        return lambda: plan.prepare(dynamics)

    def binding(slot: str, declaration: str, **kwargs: Any) -> Any:
        return sampling.AtomisticBiasRowBinding(
            {}, histories={slot: declaration}, **kwargs
        )

    records = []
    history = _history_plan(1, 4)
    hill, dynamics = hill_dynamics(history)
    records.append(
        _expect_refusal(
            "unbound-history-term",
            "multistate-prepare",
            multistate(dynamics, (_row(system, "nvt", temperature=TEMPERATURE),)),
        )
    )
    seeded = sampling.deposit_bias_history(
        history,
        history.initialize(jnp.float64),
        jnp.asarray([[1.0]]),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.ones((1,), dtype=jnp.bool_),
        variables.metrics,
        boltzmann_constant=BOLTZMANN,
    )
    frozen = sampling.FrozenBiasHistory(history, seeded.state)
    records.append(
        _expect_refusal(
            "mixed-history-declarations",
            "multistate-prepare",
            multistate(
                dynamics,
                tuple(
                    _row(
                        system,
                        "nvt",
                        temperature=TEMPERATURE,
                        state_id=f"declared-{index}",
                        bias=binding(hill.name, declaration),
                    )
                    for index, declaration in enumerate(("adaptive", frozen.declaration))
                ),
            ),
        )
    )
    # Frozen declaration of seeded content while every lane binds the empty history.
    frozen_rows = tuple(
        _row(
            system,
            "nvt",
            temperature=TEMPERATURE,
            state_id=f"frozen-{index}",
            bias=sampling.AtomisticBiasRowBinding(
                {}, histories={hill.name: frozen.declaration}
            ),
        )
        for index in range(2)
    )
    frozen_table = _table(dynamics, frozen_rows)
    frozen_runtime = sampling.AtomisticMultistatePlan(
        frozen_table,
        np.arange(2),
        qualification=_qualification(dynamics, frozen_table),
        exchange=sampling.AtomisticReplicaExchangePlan(1),
        run_id="enhanced-sampling-refusal-probe",
    ).prepare(dynamics)
    lanes = tuple(
        _start(dynamics, frozen_table, positions, seed=seed + index, state_index=index)
        for index in range(2)
    )
    records.append(
        _expect_refusal(
            "frozen-declaration-content-mismatch",
            "multistate-initialize",
            lambda: frozen_runtime.initialize(
                lanes, np.asarray([0, 1], dtype=np.int32), jax.random.key(seed)
            ),
        )
    )
    umbrella = sampling.AtomisticBiasTerm(
        variables,
        sampling.HarmonicBiasForm(np.asarray([1.0], dtype=np.float64), WINDOW_STIFFNESS),
        name="umbrella",
        force_group=1,
    )
    umbrella_dynamics = _dynamics(
        system, _potential(system, (_bond(), umbrella)), integrator
    )
    records.append(
        _expect_refusal(
            "declaration-on-term-without-history",
            "multistate-prepare",
            multistate(
                umbrella_dynamics,
                tuple(
                    _row(
                        system,
                        "nvt",
                        temperature=TEMPERATURE,
                        state_id=f"window-{index}",
                        bias=sampling.AtomisticBiasRowBinding(
                            {
                                umbrella.name: np.asarray(
                                    [1.0 + 0.1 * index], dtype=np.float64
                                )
                            },
                            histories={umbrella.name: "adaptive"},
                        ),
                    )
                    for index in range(2)
                ),
            ),
        )
    )
    tempered, tempered_dynamics = hill_dynamics(_history_plan(1, 4, well_tempered=True))
    records.append(
        _expect_refusal(
            "mixed-temperature-well-tempered-history",
            "multistate-prepare",
            multistate(
                tempered_dynamics,
                tuple(
                    _row(
                        system,
                        "nvt",
                        temperature=temperature,
                        bias=binding(tempered.name, "adaptive"),
                    )
                    for temperature in (TEMPERATURE, 2.0 * TEMPERATURE)
                ),
            ),
        )
    )
    abf = sampling.AtomisticABFPlan(
        variables,
        minimum=0.3,
        maximum=1.05 * (atoms - 1) * BOND_LENGTH,
        bins=4,
        temperature=TEMPERATURE,
    ).prepare(system)
    abf_dynamics = _dynamics(
        system,
        _potential(system, (_bond(),)),
        integrator,
        force_ports=(sampling.AtomisticABFForcePort(abf, name="abf"),),
    )
    records.append(
        _expect_refusal(
            "force-only-abf-in-multistate-table",
            "thermodynamic-table",
            lambda: _table(
                abf_dynamics,
                tuple(
                    _row(system, "nvt", temperature=temperature)
                    for temperature in (TEMPERATURE, 2.0 * TEMPERATURE)
                ),
            ),
        )
    )
    return records


# ----------------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        nargs="+",
        choices=("history", "dynamics", "abf", "multistate", "refusals", "all"),
        default=["all"],
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Small sizes for a quick check."
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--label", default="local")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--history-capacities", nargs="+", type=int, default=[64, 512, 4096]
    )
    parser.add_argument("--history-fill", nargs="+", type=float, default=[0.5, 0.0, 1.0])
    parser.add_argument("--history-dimensions", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--history-walkers", nargs="+", type=int, default=[4, 1, 16])
    parser.add_argument(
        "--history-ownerships",
        nargs="+",
        choices=("shared", "walker", "state"),
        default=["shared", "walker", "state"],
    )
    parser.add_argument(
        "--history-storages",
        nargs="+",
        choices=("terminal", "fifo"),
        default=["terminal", "fifo"],
    )
    parser.add_argument(
        "--history-temperings",
        nargs="+",
        choices=("standard", "well-tempered"),
        default=["standard", "well-tempered"],
    )
    parser.add_argument("--rmsd-atoms", nargs="+", type=int, default=[4, 16])
    parser.add_argument("--atoms", nargs="+", type=int, default=[4, 16, 64])
    parser.add_argument("--cvs", nargs="+", default=["distance", "rg", "distances-3"])
    parser.add_argument("--dynamics-capacities", nargs="+", type=int, default=[16, 1024])
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--abf-atoms", nargs="+", type=int, default=[4, 16])
    parser.add_argument("--abf-bins", nargs="+", type=int, default=[16, 64, 256])
    parser.add_argument(
        "--abf-dimensions", nargs="+", type=int, choices=(1, 2), default=[1, 2]
    )
    parser.add_argument(
        "--multistate-cases",
        nargs="+",
        default=[
            "umbrella:2:2",
            "umbrella:4:4",
            "umbrella:8:8",
            "adaptive-shared:4:4",
            "adaptive-walker:4:4",
            "adaptive-state:4:4",
            "adaptive-state:8:8",
            "frozen-shared:4:4",
            "frozen-state:4:4",
            "exhaustion-shared:4:4",
            "umbrella:4:2",
            "umbrella:2:4",
            "umbrella:8:4",
        ],
    )
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--multistate-atoms", type=int, default=4)
    return parser


SMOKE = {
    "repeats": 1,
    "history_capacities": [16],
    "history_fill": [0.5, 1.0],
    "history_dimensions": [1],
    "history_walkers": [2],
    "history_ownerships": ["shared", "walker", "state"],
    "history_storages": ["terminal", "fifo"],
    "history_temperings": ["standard"],
    "rmsd_atoms": [4],
    "atoms": [4],
    "cvs": ["distance"],
    "dynamics_capacities": [4],
    "steps": 6,
    "abf_atoms": [4],
    "abf_bins": [4],
    "abf_dimensions": [1],
    "multistate_cases": [
        "adaptive-state:2:2",
        "frozen-shared:2:2",
        "exhaustion-shared:2:2",
    ],
    "iterations": 4,
}


def _validate(arguments: argparse.Namespace, /) -> None:
    if arguments.warmup < 0 or arguments.repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats positive.")
    if arguments.steps < 1 or arguments.iterations < 2:
        raise ValueError("steps must be positive and iterations at least two.")
    positive = (
        arguments.history_capacities
        + arguments.history_dimensions
        + arguments.history_walkers
        + arguments.dynamics_capacities
        + arguments.abf_bins
    )
    if any(value < 1 for value in positive):
        raise ValueError("Capacities, dimensions, walkers and bins must be positive.")
    if any(not 0.0 <= value <= 1.0 for value in arguments.history_fill):
        raise ValueError("History fill fractions lie in [0, 1].")
    if any(value < 3 for value in arguments.rmsd_atoms + arguments.abf_atoms):
        raise ValueError("Aligned-MSD kernels and ABF chains need at least three atoms.")
    if any(value < 2 for value in arguments.atoms) or arguments.multistate_atoms < 2:
        raise ValueError("Chains need at least two atoms.")
    for value in arguments.cvs:
        for atoms in arguments.atoms:
            match = CV_PATTERN.match(value)
            if match is None or (match.group(2) and not 0 < int(match.group(2)) < atoms):
                raise ValueError(f"CV {value!r} is invalid for {atoms} atoms.")
    for value in arguments.multistate_cases:
        _parse_mode(value)


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.smoke:
        for name, value in SMOKE.items():
            setattr(arguments, name, value)
    _validate(arguments)
    pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
    selected = set(arguments.scenario)
    if "all" in selected:
        selected = {"history", "dynamics", "abf", "multistate", "refusals"}
    timing = {"warmup": arguments.warmup, "repeats": arguments.repeats}
    seed = arguments.seed
    history = (
        [
            _history_case(configuration, seed=seed + index, **timing)
            for index, configuration in enumerate(_history_configurations(arguments))
        ]
        if "history" in selected
        else None
    )
    dynamics = (
        [
            _dynamics_case(configuration, steps=arguments.steps, seed=seed, **timing)
            for configuration in _one_at_a_time(
                {
                    "atoms": arguments.atoms,
                    "cv": arguments.cvs,
                    "capacity": arguments.dynamics_capacities,
                }
            )
        ]
        if "dynamics" in selected
        else None
    )
    abf = (
        [
            _abf_case(configuration, steps=arguments.steps, seed=seed, **timing)
            for configuration in _one_at_a_time(
                {
                    "atoms": arguments.abf_atoms,
                    "bins": arguments.abf_bins,
                    "dimension": arguments.abf_dimensions,
                }
            )
        ]
        if "abf" in selected
        else None
    )
    multistate = (
        [
            _multistate_case(
                value,
                iterations=arguments.iterations,
                atoms=arguments.multistate_atoms,
                seed=seed,
                **timing,
            )
            for value in arguments.multistate_cases
        ]
        if "multistate" in selected
        else None
    )
    refusals = (
        _refusal_probes(atoms=arguments.multistate_atoms, seed=seed)
        if "refusals" in selected
        else None
    )
    rows = [
        row
        for scenario in (history, dynamics, abf, multistate, refusals)
        if scenario is not None
        for row in scenario
    ]
    root = Path(__file__).resolve().parents[1]
    configuration = {
        name: value for name, value in vars(arguments).items() if name not in ("output",)
    }
    configuration["scenario"] = sorted(selected)
    record = {
        "benchmark": "atomistic-enhanced-sampling",
        "label": arguments.label,
        "configuration": configuration,
        "environment": capture_environment().to_dict(),
        "identity": capture_benchmark_identity(
            root, Path(__file__), EVIDENCE_FIELDS
        ).to_dict(),
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in OWNERS
        },
        "statistical_scope": SCOPE,
        "history": history,
        "dynamics": dynamics,
        "abf": abf,
        "multistate": multistate,
        "refusals": refusals,
        "all_expected_outcomes": all(
            value for row in rows for value in row["gates"].values()
        ),
    }
    write_json_atomic(arguments.output, record)
    print(arguments.output)


if __name__ == "__main__":
    main()
