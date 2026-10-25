# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Phase-separated native model-family scaling: MACE, MACELES, AIMNet2, UMA.

Every row is one native family form (``mace`` exact, ``mace-prepared``,
``maceles`` exact, ``aimnet2``, ``aimnet2-nse``, ``uma`` raw routed MoLE,
``uma-merged`` explicitly prepared MoLE realization) on a sheared periodic
water box at liquid density, at one controlling scale: molecules (atoms =
3 x molecules), cases, the local cutoff (MACE radial cutoff, also MACELES's
local MACE; AIMNet2 source ``aev.rc_s``; UMA source ``cutoff``), channels
(MACE channel count, UMA sphere channels), degree (MACE hidden degree, UMA
lmax), the UMA source nearest-neighbor cap (a scientific selection rule) and
the graph neighbor capacity (a resource). Axes a family does not own are
recorded as ``null`` and not swept; a ``null`` cutoff is the family-native
radius. Each row records the realized local radius and every distinct
declared radius (MACELES latent Ewald ``dl``/``sigma``, AIMNet2 pair radii).

Each row separately records:

* ``preparation``: native model construction or source-layout reconstruction,
  the prepared realization (MACE prepared, UMA merge), the graph payload, and
  their logical retained bytes;
* per workload: ``lowering``, ``compilation``, ``first`` and ``warm``
  executions, XLA compiler argument/output/temporary/code bytes, the sampled
  device peak above the warm-phase baseline (an observation, never a bound),
  and output status. Workloads are ``efs`` (energy, conservative forces and
  tensile-positive stress), ``force-loss-gradient`` (mixed parameter-coordinate
  E/F/S training gradient), ``hvp-loss-gradient`` (mixed third-order parameter
  gradient of a Hessian-vector loss, admitted only by a declared derivative
  contract under ``--hvp-policy``), ``fit-step`` (actual `fit_atomistic_potential`
  first update and continued update through the canonical loader; compilation
  is inside the trainer's first call and is not separable there) and
  ``merge-refresh`` (UMA merged staleness refusal after a parameter change and
  explicit re-preparation).

Fit rows declare scale 1 in each objective's canonical label unit, rather than
estimating a variance from the single-case default. The loss definition stays
fixed across row sizes; the native refusal of degenerate fitted scales remains
unchanged. JAX tracing errors are recorded as failures, not support refusals.

Exact and prepared forms of one row are compared numerically on the same
payload. Refusals (undeclared derivative contracts, prepared forms that are
not training forms, capacity overflow, scientific failure) are kept with their
exception type and message; nothing is dropped.

Timing refuses to start when another GPU compute process or an elevated host
load is observed, and every row records the post-row observation. All models
carry random or random source-layout parameters: rows say nothing about
pretrained accuracy, source-provider equivalence or other hardware.
``main`` captures the canonical benchmark identity first and fails closed when
its source-build prerequisites are absent; `family_cases` and `measure_case`
are the reusable scenario entry points of the same measured kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import subprocess
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from itertools import product
from operator import methodcaller
from pathlib import Path
from typing import Any, get_args, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

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
    synchronize,
)
from phydrax._differentiation import (
    ComponentAuthority,
    DerivativeSurface,
    DifferentiationRequest,
    RegularityPolicy,
)
from phydrax._trainable import combine_parameters, partition_parameters
from phydrax.atomistic import (
    atomistic_energy_derivatives,
    atomistic_model_revision,
    atomistic_training_loader,
    AtomisticBatch,
    AtomisticEnergyObjective,
    AtomisticForceObjective,
    AtomisticGraphLayout,
    AtomisticObservable,
    AtomisticObservableRequest,
    AtomisticStressObjective,
    AtomisticTrainingPolicy,
    AtomisticTrainingProblem,
    fit_atomistic_potential,
)
from phydrax.atomistic._hessian import ATOMISTIC_HESSIAN_REQUEST, bind_atomistic_hessian
from phydrax.execution import PhaseMemorySampler
from phydrax.nn.atomistic import prepare_mace_potential, StaleUMAMergeError
from phydrax.sparse import StreamedRelationPlan
from tests._support.aimnet2_source import RC_S
from tests._support.atomistic_model_families import (
    aimnet2_model,
    family_scenario,
    FamilyScenario,
    MACE_CUTOFF,
    mace_model,
    maceles_model,
    NEUTRAL,
    periodic_execution,
    teacher_source,
    training_fields,
    uma_binding,
    uma_model,
    water_batch,
    water_box,
)


Family: TypeAlias = Literal[
    "mace", "mace-prepared", "maceles", "aimnet2", "aimnet2-nse", "uma", "uma-merged"
]
Workload: TypeAlias = Literal[
    "efs", "force-loss-gradient", "hvp-loss-gradient", "fit-step", "merge-refresh"
]
HVPPolicy: TypeAlias = Literal["strict", "almost-everywhere"]
_FAMILIES: tuple[str, ...] = get_args(Family)
_WORKLOADS: tuple[str, ...] = get_args(Workload)
_PREPARED: dict[str, str] = {"mace-prepared": "mace", "uma-merged": "uma"}
_SAMPLE_INTERVAL = 0.005
_MIXED = DifferentiationRequest(
    (DerivativeSurface.INPUT, DerivativeSurface.MODEL_PARAMETER),
    order=3,
    authority=ComponentAuthority.MODEL,
)
_HVP_REQUEST = AtomisticObservableRequest(
    (AtomisticObservable.HESSIAN_ACTION,),
    derivatives=(ATOMISTIC_HESSIAN_REQUEST, _MIXED),
)
_OWNERS = (
    "benchmarks/atomistic_model_families.py",
    "tests/_support/atomistic_model_families.py",
    "tests/_support/aimnet2_source.py",
    "tests/_support/uma_source.py",
    "phydrax/atomistic/_prediction.py",
    "phydrax/atomistic/_hessian.py",
    "phydrax/atomistic/_training.py",
    "phydrax/atomistic/_latent_electrostatics.py",
    "phydrax/nn/atomistic/_mace.py",
    "phydrax/nn/atomistic/_mace_prepare.py",
    "phydrax/nn/atomistic/_mace_les.py",
    "phydrax/nn/atomistic/_aimnet2.py",
    "phydrax/nn/atomistic/_aimnet2_source.py",
    "phydrax/nn/atomistic/_uma.py",
    "phydrax/nn/atomistic/_uma_source.py",
)


@dataclass(frozen=True, slots=True)
class FamilyCase:
    """One family form at one controlling scale.

    ``cutoff`` is the requested local radius in angstrom, built through each
    family's canonical configuration: the MACE architecture cutoff (for
    MACELES its local MACE, distinct from the latent Ewald ``dl``), the AIMNet2
    source ``aev.rc_s`` (its declared short-range Coulomb radius is unchanged
    and must lie within it) or the UMA source ``cutoff``; ``None`` keeps the
    family-native radius. ``channels``/``degree`` are MACE channel count/hidden
    degree or UMA sphere channels/lmax; ``uma_max_neighbors`` is UMA's source
    nearest-neighbor cap. ``None`` marks an axis the family's fixture fixes.
    """

    family: str
    molecules: int
    cases: int
    cutoff: float | None
    channels: int | None
    degree: int | None
    uma_max_neighbors: int | None
    neighbor_capacity: int
    workloads: tuple[str, ...]
    hvp_policy: str
    repeats: int
    seed: int = 3

    @property
    def base(self) -> str:
        return _PREPARED.get(self.family, self.family)

    @property
    def atoms(self) -> int:
        return 3 * self.molecules


def family_cases(
    families: Sequence[str],
    molecules: Sequence[int],
    cases: Sequence[int],
    cutoffs: Sequence[float | None],
    channels: Sequence[int],
    degrees: Sequence[int],
    uma_max_neighbors: Sequence[int],
    neighbor_capacities: Sequence[int],
    workloads: Sequence[str],
    *,
    hvp_policy: str = "strict",
    repeats: int = 3,
) -> tuple[FamilyCase, ...]:
    """Cartesian scale matrix, with family-fixed axes collapsed to ``None``.

    A ``None`` cutoff selects each family's native radius.
    """
    unknown = set(families).difference(_FAMILIES) | set(workloads).difference(_WORKLOADS)
    if unknown:
        raise ValueError(f"Unknown families or workloads: {sorted(unknown)!r}.")
    if hvp_policy not in get_args(HVPPolicy):
        raise ValueError(f"hvp_policy must be one of {get_args(HVPPolicy)!r}.")
    for label, values in (
        ("molecules", molecules),
        ("cases", cases),
        ("channels", channels),
        ("degrees", degrees),
        ("uma_max_neighbors", uma_max_neighbors),
        ("neighbor_capacities", neighbor_capacities),
    ):
        if not values or any(int(value) < 1 for value in values):
            raise ValueError(f"{label} must be nonempty positive integers.")
    if not cutoffs or any(
        value is not None and not (math.isfinite(value) and value > 0.0)
        for value in cutoffs
    ):
        raise ValueError(
            "cutoffs must be nonempty: None (family-native) or positive finite "
            "radii in angstrom."
        )
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    rows: dict[FamilyCase, None] = {}
    for family in families:
        base = _PREPARED.get(family, family)
        scaled = base in ("mace", "maceles", "uma")
        for count, case_count, cutoff, channel, degree, cap, capacity in product(
            molecules,
            cases,
            cutoffs,
            channels if scaled else (None,),
            degrees if scaled else (None,),
            uma_max_neighbors if base == "uma" else (None,),
            neighbor_capacities,
        ):
            rows[
                FamilyCase(
                    family,
                    int(count),
                    int(case_count),
                    None if cutoff is None else float(cutoff),
                    None if channel is None else int(channel),
                    None if degree is None else int(degree),
                    None if cap is None else int(cap),
                    int(capacity),
                    tuple(workloads),
                    hvp_policy,
                    int(repeats),
                )
            ] = None
    return tuple(rows)


# ------------------------------------------------------- competing jobs


def competing_jobs() -> dict[str, Any]:
    """Foreign GPU compute processes and host load observed right now."""
    foreign: list[str] = []
    query: str
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        query = "nvidia-smi" if completed.returncode == 0 else "nvidia-smi-failed"
        for line in completed.stdout.splitlines():
            pid, _, name = line.partition(",")
            if pid.strip().isdigit() and int(pid) != os.getpid():
                foreign.append(f"{pid.strip()}:{name.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        query = "nvidia-smi-unavailable"
    load = os.getloadavg()[0]
    cpus = os.cpu_count() or 1
    return {
        "gpu_query": query,
        "foreign_gpu_processes": foreign,
        "load_average_1m": load,
        "cpu_count": cpus,
        "competing": bool(foreign) or load > 0.25 * cpus,
    }


# ------------------------------------------------------------- scenario


@dataclass(frozen=True)
class PreparedCase:
    """Native model forms, graph payload and conditions of one row."""

    case: FamilyCase
    scenario: FamilyScenario
    exact: Any
    form: Any
    batch: AtomisticBatch
    execution: Any
    payload: Any
    conditioning: Any
    binding: Any
    preparation: dict[str, Any]


def _build(case: FamilyCase, cell: np.ndarray, atom_capacity: int, seed: int) -> Any:
    match case.base:
        case "mace" | "maceles":
            mace = mace_model(
                seed,
                channels=case.channels or 3,
                hidden_degree=case.degree or 1,
                cutoff=MACE_CUTOFF if case.cutoff is None else case.cutoff,
                streaming=StreamedRelationPlan(accumulation="deterministic"),
            )
            if case.base == "mace":
                return mace
            # The latent Ewald definition (and its dl) is the source default,
            # independent of the local MACE cutoff.
            return maceles_model(mace, atom_capacity=atom_capacity, cell=cell)
        case "aimnet2" | "aimnet2-nse":
            return aimnet2_model(
                seed,
                channels=2 if case.base == "aimnet2-nse" else 1,
                aev_cutoff=RC_S if case.cutoff is None else case.cutoff,
            )
        case "uma":
            return uma_model(
                seed,
                ff_type="spectral",
                sphere_channels=case.channels or 8,
                lmax=case.degree or 2,
                mmax=min(case.degree or 2, 1),
                max_neighbors=case.uma_max_neighbors or 30,
                streaming=StreamedRelationPlan(
                    receiver_tile=32, edge_tile=512, channel_capacity=512
                ),
                node_block=32,
                # Absent: the fixture's own source configuration cutoff.
                **({} if case.cutoff is None else {"cutoff": case.cutoff}),
            )
    raise ValueError(f"Unknown family {case.family!r}.")


def _radii(case: FamilyCase, model: Any) -> dict[str, Any]:
    """Realized local radius and each distinct declared radius (angstrom)."""
    configuration = model.configuration
    match case.base:
        case "mace":
            return {"local_cutoff": float(configuration.cutoff)}
        case "maceles":
            definition = model.latent_ewald.definition
            return {
                "local_cutoff": float(configuration.cutoff),
                "latent_ewald_dl": float(definition.dl),
                "latent_ewald_sigma": float(definition.sigma),
            }
        case "aimnet2" | "aimnet2-nse":
            return {
                "aev_cutoff": float(configuration.aev_cutoff),
                "pair_cutoffs": {
                    stage.name: float(stage.pair_cutoff)
                    for stage in configuration.outputs
                    if stage.pair_cutoff is not None
                },
                "graph_cutoff": float(configuration.cutoff),
            }
        case "uma":
            return {
                "local_cutoff": float(configuration.cutoff),
                "max_neighbors": int(configuration.max_neighbors),
            }
    raise ValueError(f"Unknown family {case.family!r}.")


def prepare_case(case: FamilyCase, /) -> PreparedCase:
    """Construct, prepare and bind one row, timing each preparation phase."""
    boxes = [
        water_box(case.molecules, seed=case.seed + index) for index in range(case.cases)
    ]
    cell = boxes[0][0]
    positions = np.stack([box[1] for box in boxes])
    batch = water_batch(
        positions, cells=np.stack([box[0] for box in boxes]), prefix="box"
    )
    execution = periodic_execution(
        degree=case.neighbor_capacity,
        edges=case.neighbor_capacity * case.atoms * case.cases,
        particles_per_cell=32,
        search_cell=cell,
        search_margin=0.1,
    )
    exact, construction = measure_host(
        lambda: synchronize(_build(case, cell, batch.atom_capacity, case.seed))
    )
    scenario = family_scenario(
        case.base,
        periodic=True,
        atom_capacity=batch.atom_capacity,
        cell=cell,
        model=exact,
    )
    conditioning = scenario.conditioning(batch, [NEUTRAL] * case.cases)
    binding = None
    form = exact
    realization: dict[str, Any] = {"form": "exact"}
    if case.family == "mace-prepared":
        form, seconds = measure_host(lambda: synchronize(prepare_mace_potential(exact)))
        realization = {"form": "prepared-mace-exact-radial", "seconds": seconds}
    elif case.family == "uma-merged":
        binding = uma_binding(exact, batch, periodic=True)
        form, seconds = measure_host(lambda: synchronize(exact.merged(binding)))
        realization = {
            "form": "merged-mole",
            "seconds": seconds,
            "binding_id": binding.binding_id,
            "prepared_id": form.prepared_id,
        }
    cutoff = float(exact.configuration.cutoff)
    payload, layout = measure_synchronized(
        lambda: AtomisticGraphLayout.for_batch(batch, execution, cutoff=cutoff).prepare(
            batch
        )
    )
    preparation = {
        "construction_seconds": construction,
        "realization": realization,
        "graph_payload_seconds": layout,
        "retained_bytes": {
            "exact_model": logical_array_bytes(exact),
            "form": logical_array_bytes(form),
            "payload": logical_array_bytes(payload),
        },
        "architecture_id": exact.architecture_id,
        "revision_id": atomistic_model_revision(exact).revision_id,
        "cutoff": cutoff,
        "radii": _radii(case, exact),
        "cell_lengths": np.linalg.norm(cell, axis=1).tolist(),
    }
    return PreparedCase(
        case,
        scenario,
        exact,
        form,
        batch,
        execution,
        payload,
        conditioning,
        binding,
        preparation,
    )


# ------------------------------------------------------------ kernels


def _efs(
    model: Any, payload: Any, positions: Array, cells: Array, conditioning: Any
) -> Any:
    result = atomistic_energy_derivatives(
        model,
        payload,
        positions,
        cell_vectors=cells,
        compute_stress=True,
        conditioning=conditioning,
    )
    return result.energy, result.forces, result.stress, result.successful, result.overflow


def _force_loss_gradient(
    lane: Any,
    state: Any,
    fixed: Any,
    payload: Any,
    positions: Array,
    cells: Array,
    conditioning: Any,
) -> Any:
    def loss(values: Any) -> Array:
        result = atomistic_energy_derivatives(
            combine_parameters(values, state, fixed),
            payload,
            positions,
            cell_vectors=cells,
            compute_stress=True,
            conditioning=conditioning,
        )
        if result.forces is None or result.stress is None:
            raise RuntimeError(
                "The force/stress benchmark requires both requested responses."
            )
        return (
            jnp.sum(result.energy**2)
            + jnp.sum(result.forces**2)
            + jnp.sum(result.stress**2)
        )

    return jax.value_and_grad(loss)(lane)


def _hvp_loss_gradient(
    lane: Any,
    state: Any,
    fixed: Any,
    payload: Any,
    positions: Array,
    cells: Array,
    conditioning: Any,
    probe: Array,
    policy: RegularityPolicy | None,
) -> Any:
    def loss(values: Any) -> Array:
        operator = bind_atomistic_hessian(
            combine_parameters(values, state, fixed),
            payload,
            positions,
            request=_HVP_REQUEST,
            policy=policy,
            cell_vectors=cells,
            conditioning=conditioning,
        )
        return jnp.sum(operator.mv(probe) ** 2)

    return jax.value_and_grad(loss)(lane)


_KERNELS: dict[str, Callable[..., Any]] = {
    "efs": eqx.filter_jit(_efs),
    "force-loss-gradient": eqx.filter_jit(_force_loss_gradient),
    "hvp-loss-gradient": eqx.filter_jit(_hvp_loss_gradient),
}


def failure_record(error: BaseException, /) -> dict[str, Any]:
    tracing_failure = isinstance(
        error,
        (
            jax.errors.JAXTypeError,
            jax.errors.JaxprTypeError,
            jax.errors.InconclusiveDimensionOperation,
        ),
    )
    return {
        "status": (
            "refused"
            if isinstance(error, (ValueError, TypeError)) and not tracing_failure
            else "failed"
        ),
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exception(error),
    }


def _finite(value: Any) -> Any:
    array = np.asarray(value, dtype=np.float64)
    if np.all(np.isfinite(array)):
        return array.tolist()
    return {"kind": "nonfinite", "finite_mask": np.isfinite(array).tolist()}


def _measure(
    name: str, arguments: tuple[Any, ...], repeats: int
) -> tuple[Any, dict[str, Any]]:
    lowering = _KERNELS[name].lower  # ty: ignore[unresolved-attribute]
    executable, phases = measure_lower_and_compile(
        partial(lowering, *arguments), methodcaller("compile")
    )
    invocation = partial(executable, *arguments)
    output, first = measure_synchronized(invocation)
    devices = tuple(device for device in jax.local_devices() if device.platform != "cpu")
    with PhaseMemorySampler(
        name, interval_seconds=_SAMPLE_INTERVAL, devices=devices
    ) as sampler:
        _, warm = measure_repeated(invocation, warmup=1, repeats=repeats)
    compiled = executable.compiled
    return output, {
        "status": "measured",
        "lowering_seconds": phases.lowering_seconds,
        "compilation_seconds": phases.compilation_seconds,
        "first_seconds": first,
        "warm": warm.to_dict(unit="seconds"),
        "compiler": asdict(
            compiler_evidence(
                compiled.cost_analysis(),
                compiled.memory_analysis(),
                source="xla",
                unavailable_reason="The selected backend did not expose compiler estimates.",
            )
        ),
        "sampled_peak_increase_bytes": sampler.evidence.sampled_peak_increase_bytes,
        "memory": sampler.evidence.to_payload(),
        "output_retained_bytes": logical_array_bytes(output),
    }


def _arguments(prepared: PreparedCase) -> tuple[Any, Any]:
    return jnp.asarray(prepared.batch.positions), jnp.asarray(prepared.batch.cells)


def _efs_workload(prepared: PreparedCase) -> dict[str, Any]:
    positions, cells = _arguments(prepared)
    output, record = _measure(
        "efs",
        (prepared.form, prepared.payload, positions, cells, prepared.conditioning),
        prepared.case.repeats,
    )
    energy, forces, stress, successful, overflow = (np.asarray(value) for value in output)
    if np.any(overflow):
        record["status"] = "capacity-overflow"
    elif not np.all(successful):
        record["status"] = "scientific-failure"
    record["outputs"] = {
        "energy": _finite(energy),
        "max_abs_force": _finite(np.max(np.abs(forces), axis=(1, 2))),
        "net_force": _finite(np.sum(forces, axis=1)),
        "stress": _finite(stress),
        "successful": successful.tolist(),
        "overflow": np.asarray(overflow).tolist(),
    }
    if prepared.form is not prepared.exact:
        exact = atomistic_energy_derivatives(
            prepared.exact,
            prepared.payload,
            positions,
            cell_vectors=cells,
            compute_forces=False,
            conditioning=prepared.conditioning,
        ).energy
        record["max_abs_energy_difference_vs_exact"] = float(
            np.max(np.abs(np.asarray(exact) - energy))
        )
    return record


def _not_a_training_form(prepared: PreparedCase) -> dict[str, Any] | None:
    if prepared.form is prepared.exact:
        return None
    return {
        "status": "not-applicable",
        "message": "Prepared realizations are frozen inference forms; the exact "
        "model is the training form.",
    }


def _gradient_workload(prepared: PreparedCase, name: str) -> dict[str, Any]:
    refusal = _not_a_training_form(prepared)
    if refusal is not None:
        return refusal
    positions, cells = _arguments(prepared)
    lane, state, fixed = partition_parameters(prepared.exact)
    arguments: tuple[Any, ...] = (
        lane,
        state,
        fixed,
        prepared.payload,
        positions,
        cells,
        prepared.conditioning,
    )
    if name == "hvp-loss-gradient":
        probe = jnp.asarray(
            np.random.default_rng(prepared.case.seed).choice(
                [-1.0, 1.0], size=positions.shape
            )
        )
        policy = (
            RegularityPolicy(allow_almost_everywhere=True)
            if prepared.case.hvp_policy == "almost-everywhere"
            else None
        )
        arguments = (*arguments, probe, policy)
    output, record = _measure(name, arguments, prepared.case.repeats)
    value, gradient = output
    record["loss"] = _finite(value)
    record["gradient_finite"] = bool(
        all(
            np.all(np.isfinite(np.asarray(leaf)))
            for leaf in jax.tree_util.tree_leaves(gradient)
        )
    )
    record["hvp_policy"] = (
        prepared.case.hvp_policy if name == "hvp-loss-gradient" else None
    )
    return record


def _fit_workload(prepared: PreparedCase) -> dict[str, Any]:
    refusal = _not_a_training_form(prepared)
    if refusal is not None:
        return refusal
    case = prepared.case
    cell = np.asarray(prepared.batch.cells)[0]
    teacher = family_scenario(
        case.base,
        periodic=True,
        atom_capacity=prepared.batch.atom_capacity,
        cell=cell,
        model=_build(case, cell, prepared.batch.atom_capacity, case.seed + 100),
    )

    def prepare_problem() -> AtomisticTrainingProblem:
        states = [NEUTRAL] * case.cases
        source = teacher_source(
            teacher,
            prepared.batch,
            prepared.execution,
            states,
            prefix=f"{case.family}-bench",
        )
        # One-case rows cannot estimate an energy standard deviation. Fixed
        # label-unit scales also keep the loss definition comparable across rows.
        return AtomisticTrainingProblem(
            atomistic_training_loader(source, shuffle=False),
            prepared.execution,
            (
                AtomisticEnergyObjective(scale=1.0),
                AtomisticForceObjective(scale=1.0),
                AtomisticStressObjective(scale=1.0),
            ),
            scale=prepared.batch.scale,
            cutoff=teacher.model.configuration.cutoff,
            fields=training_fields(teacher, teacher.columns(states)),
        )

    problem, labels = measure_host(prepare_problem)
    first, first_seconds = measure_host(
        lambda: synchronize(
            fit_atomistic_potential(
                prepared.exact,
                problem,
                AtomisticTrainingPolicy(maximum_steps=1, learning_rate=1.0e-3),
                key=jax.random.key(case.seed),
            )
        )
    )
    continued, continued_seconds = measure_host(
        lambda: synchronize(
            fit_atomistic_potential(
                prepared.exact,
                problem,
                AtomisticTrainingPolicy(maximum_steps=2, learning_rate=1.0e-3),
                key=jax.random.key(case.seed),
                continuation=first,
            )
        )
    )
    return {
        "status": "measured" if bool(continued.successful) else "training-failure",
        "problem_preparation_seconds": labels,
        "first_update_seconds_including_compilation": first_seconds,
        "continued_update_seconds": continued_seconds,
        "compilation_separable": False,
        "termination": str(continued.termination),
        "training_values": _finite(continued.training_values),
        "objectives": [objective.name for objective in problem.objectives],
        "objective_scales_in_label_units": {
            objective.name: objective.scale for objective in problem.objectives
        },
    }


def _merge_workload(prepared: PreparedCase) -> dict[str, Any]:
    if prepared.case.family != "uma-merged":
        return {"status": "not-applicable", "message": "Only UMA merged rows refresh."}
    lane, state, fixed = partition_parameters(prepared.exact)
    changed = combine_parameters(
        jax.tree_util.tree_map(lambda leaf: leaf * (1.0 + 1.0e-3), lane), state, fixed
    )
    stale, seconds = measure_host(partial(_stale, prepared.form, changed))
    refreshed, refresh = measure_host(
        lambda: synchronize(changed.merged(prepared.binding))
    )
    _, validation = measure_host(refreshed.validate)
    return {
        "status": "measured" if stale else "stale-not-refused",
        "stale_refusal": stale,
        "stale_check_seconds": seconds,
        "refresh_seconds": refresh,
        "validate_seconds": validation,
        "refreshed_prepared_id": refreshed.prepared_id,
        "previous_prepared_id": prepared.form.prepared_id,
    }


def _stale(merged: Any, model: Any) -> bool:
    try:
        merged.require_current(model)
    except StaleUMAMergeError:
        return True
    return False


def measure_case(case: FamilyCase, /) -> dict[str, Any]:
    """Complete row: scales, preparation and every workload, failures kept."""
    row: dict[str, Any] = {
        "family": case.family,
        "scales": {
            "molecules": case.molecules,
            "atoms": case.atoms,
            "cases": case.cases,
            "cutoff": case.cutoff,
            "channels": case.channels,
            "degree": case.degree,
            "uma_max_neighbors": case.uma_max_neighbors,
            "neighbor_capacity": case.neighbor_capacity,
        },
        "workloads": {},
    }
    try:
        prepared = prepare_case(case)
    except Exception as error:  # every refusal or failure is a recorded row
        row["preparation"] = failure_record(error)
        row["post_row_jobs"] = competing_jobs()
        return row
    row["preparation"] = prepared.preparation
    for name in case.workloads:
        try:
            match name:
                case "efs":
                    record = _efs_workload(prepared)
                case "force-loss-gradient" | "hvp-loss-gradient":
                    record = _gradient_workload(prepared, name)
                case "fit-step":
                    with PhaseMemorySampler(
                        name,
                        interval_seconds=_SAMPLE_INTERVAL,
                        devices=tuple(
                            device
                            for device in jax.local_devices()
                            if device.platform != "cpu"
                        ),
                    ) as sampler:
                        record = _fit_workload(prepared)
                    record["memory"] = sampler.evidence.to_payload()
                case "merge-refresh":
                    with PhaseMemorySampler(
                        name,
                        interval_seconds=_SAMPLE_INTERVAL,
                        devices=tuple(
                            device
                            for device in jax.local_devices()
                            if device.platform != "cpu"
                        ),
                    ) as sampler:
                        record = _merge_workload(prepared)
                    record["memory"] = sampler.evidence.to_payload()
                case _:
                    raise ValueError(f"Unknown workload {name!r}.")
        except Exception as error:  # every refusal or failure is a recorded row
            record = failure_record(error)
        row["workloads"][name] = record
    row["process_peak_rss_bytes"] = process_peak_rss_bytes()
    row["post_row_jobs"] = competing_jobs()
    return row


# ------------------------------------------------------------------ CLI


def _cutoff(text: str, /) -> float | None:
    """``native`` (the family's own radius) or a local radius in angstrom."""
    return None if text == "native" else float(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families", nargs="+", choices=_FAMILIES, default=list(_FAMILIES)
    )
    parser.add_argument("--molecules", nargs="+", type=int, default=[2, 8])
    parser.add_argument("--cases", nargs="+", type=int, default=[1])
    parser.add_argument("--cutoffs", nargs="+", type=_cutoff, default=[None])
    parser.add_argument("--channels", nargs="+", type=int, default=[4])
    parser.add_argument("--degrees", nargs="+", type=int, default=[1])
    parser.add_argument("--uma-max-neighbors", nargs="+", type=int, default=[30])
    parser.add_argument("--neighbor-capacities", nargs="+", type=int, default=[128])
    parser.add_argument(
        "--workloads", nargs="+", choices=_WORKLOADS, default=list(_WORKLOADS)
    )
    parser.add_argument("--hvp-policy", choices=get_args(HVPPolicy), default="strict")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
    root = Path(__file__).resolve().parents[1]
    identity = capture_benchmark_identity(
        root,
        Path(__file__),
        ("cases", "competing_jobs", "environment", "scope", "source_sha256"),
    )
    cases = family_cases(
        args.families,
        args.molecules,
        args.cases,
        args.cutoffs,
        args.channels,
        args.degrees,
        args.uma_max_neighbors,
        args.neighbor_capacities,
        args.workloads,
        hvp_policy=args.hvp_policy,
        repeats=args.repeats,
    )
    before = competing_jobs()
    if before["competing"]:
        raise SystemExit(f"Refusing to time under competing jobs: {before}.")
    rows = [measure_case(case) for case in cases]
    record = {
        "benchmark": "native-atomistic-model-families",
        "label": args.label,
        "environment": capture_environment().to_dict(),
        "identity": identity.to_dict(),
        "competing_jobs": before,
        "scope": {
            "parameters": "random native or random source-layout parameters; no "
            "pretrained accuracy or source-provider equivalence",
            "precision": "float64 (x64 enabled)",
            "memory": "compiler bytes are XLA estimates; sampled peaks are "
            "observations above the warm-phase baseline; retained bytes are "
            "logical array payloads",
            "hardware": "only the devices recorded in environment",
        },
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in _OWNERS
        },
        "cases": rows,
    }
    write_json_atomic(args.output, record)
    print(args.output)


__all__ = [
    "FamilyCase",
    "PreparedCase",
    "competing_jobs",
    "failure_record",
    "family_cases",
    "measure_case",
    "prepare_case",
]


if __name__ == "__main__":
    sys.exit(main())
