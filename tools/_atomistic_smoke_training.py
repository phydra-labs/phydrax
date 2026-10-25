#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Rank-side data×domain training scenarios of the distributed atomistic smoke.

`tools.atomistic_distributed_smoke` launches the coupled processes and
dispatches ``dp-domain`` / ``dp-domain-restart`` here. Each rank binds REAL
devices (never lanes):

* the domain ownership group is child group 0 of
  ``ExecutionRuntime.child_groups(2)`` (``P * D / 2`` owners spanning every
  process), so ONE native two-layer MACE domain plan serves every data row;
* the training group is a child of the root group with the devices of child
  groups 0 and 1 and the mesh ``(("data", 2), (owner_axis, P * D / 2))``:
  data row ``d`` executes on child group ``d``'s devices, both axes have size
  greater than one at the default 2 processes × 2 devices.

Two distinct periodic MACE systems from `tests._support.atomistic_domain`
(six atoms with energy and every force label; five atoms with NO energy label
and two masked force rows, case weight 1.5) are ingressed process-locally and
form the two data shards. Shard 1's energy objective support is exactly zero
(an empty-support shard), and the force supports are unequal (18 vs 13.5).

``dp-domain`` runs two shared-kernel SGD attempts through
`run_distributed_training_attempt` and compares objective value, the
parameter gradient implied by the exact SGD update, the updated parameters,
gradient norm and reduced numerator supports with an INDEPENDENT single-device
reference: the brute-force global MACE oracle
(`tests._support.atomistic_domain.global_energy`, forces ``-dE/dx``) with
the batch objective arithmetic and global numerator/support normalization,
differentiated by `jax.value_and_grad` over the PARAMETER lane. Perturbing
every unavailable label must leave the update bitwise unchanged (inert empty
support). After step 1 the kernel state is published durably as the
``"training"`` section of `publish_domain_checkpoint` (bound to the training
plan identity); per-rank digests of steps 1 and 2 are retained in
``directory/dp-domain-reference/rank-<process>.json``.

``dp-domain-restart`` (a fresh coupled launch, same topology) restores that
section with `restore_domain_checkpoint`, continues with step 2 and requires
bitwise equality with the uninterrupted two-step run of the same launch and
with the digests recorded by ``dp-domain``.

Forced-CPU gloo launches are a functional oracle of the collective program
only; they are not accelerator, multi-host or pretrained-model evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, get_args, Literal, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from phydrax import _jax_config
from phydrax._array_archive import ArrayArchiveLimits


if TYPE_CHECKING:
    from tools.atomistic_distributed_smoke import Strategy


TrainingScenario = Literal["dp-domain", "dp-domain-restart"]
TRAINING_SCENARIOS: tuple[TrainingScenario, ...] = get_args(TrainingScenario)
TRAINING_CHECKPOINT = "atomistic-smoke-dp-domain"
TRAINING_REFERENCE_DIRECTORY = "dp-domain-reference"
# The bound covers complete domain and kernel state, not just trained parameters.
TRAINING_CHECKPOINT_LIMITS = ArrayArchiveLimits(
    max_members=512, max_manifest_bytes=1 << 22
)
DATA_AXIS = "data"
DATA_SHARDS = 2
KERNEL_KEY = 3
LEARNING_RATE = 1.0e-3
ENERGY_SCALE = 1.0
FORCE_SCALE = 0.5
PERTURBED_LABEL = 1.0e3
# Declared roundoff contract: ordered owner/data sums of shard totals versus
# one global single-device sum (never bitwise partition invariant).
OBJECTIVE_RELATIVE_TOLERANCE = 1.0e-10
GRADIENT_NORM_RELATIVE_TOLERANCE = 1.0e-9
# Relative to max |reference gradient|; the distributed gradient is recovered
# from the exact SGD update (θ_before - θ_after) / learning_rate.
GRADIENT_RELATIVE_TOLERANCE = 1.0e-8
PARAMETER_RELATIVE_TOLERANCE = 1.0e-9
PARAMETER_ABSOLUTE_TOLERANCE = 1.0e-14
SUPPORT_ABSOLUTE_TOLERANCE = 1.0e-12


# ------------------------------------------------------------------ data rows


@dataclass(frozen=True)
class TrainingRow:
    """One periodic MACE system with logical-order labels and availability."""

    name: str
    positions: np.ndarray
    atomic_numbers: np.ndarray
    energy: float
    energy_available: bool
    forces: np.ndarray
    force_available: np.ndarray
    weight: float

    def perturbed(self) -> TrainingRow:
        """Every unavailable label replaced by a large finite value."""
        return replace(
            self,
            energy=self.energy if self.energy_available else PERTURBED_LABEL,
            forces=np.where(self.force_available[:, None], self.forces, PERTURBED_LABEL),
        )


def training_rows() -> tuple[TrainingRow, TrainingRow]:
    """Two distinct systems with unequal support and one empty energy support."""
    from tests._support.atomistic_domain import structure

    rng = np.random.default_rng(31)
    full_positions, full_numbers = structure()
    sparse_positions, sparse_numbers = structure(5, 9)
    full = TrainingRow(
        name="full-6",
        positions=full_positions,
        atomic_numbers=full_numbers,
        energy=-2.0,
        energy_available=True,
        forces=0.05 * rng.standard_normal(full_positions.shape),
        force_available=np.ones((full_positions.shape[0],), dtype=np.bool_),
        weight=1.0,
    )
    sparse = TrainingRow(
        name="energy-free-5",
        positions=sparse_positions,
        atomic_numbers=sparse_numbers,
        energy=-1.5,
        energy_available=False,
        forces=0.05 * rng.standard_normal(sparse_positions.shape),
        force_available=np.asarray([True, False, True, True, False]),
        weight=1.5,
    )
    return full, sparse


def training_objectives() -> tuple[Any, Any]:
    from phydrax.atomistic._training_objectives import (
        AtomisticEnergyObjective,
        AtomisticForceObjective,
    )

    return (
        AtomisticEnergyObjective(reduction="per-atom-structure"),
        AtomisticForceObjective(reduction="component"),
    )


# ------------------------------------------------------------- device setup


@dataclass(frozen=True)
class _Training:
    domain: Any
    model: Any
    group: Any
    distribution: Any
    reduction: Any
    kernel: Any
    initial: Any
    rows: tuple[TrainingRow, ...]
    states: tuple[Any, ...]


def training_group(execution: Any, children: Sequence[Any], ownership: Any, /) -> Any:
    """Root child group ``(data, owner)``: data row ``d`` is child group ``d``."""
    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax._execution_runtime import bind_execution_group
    from phydrax._fingerprint import canonical_fingerprint

    root = execution.root_group.spec
    keys = tuple(key for child in children for key in child.spec.device_keys)
    axes = ((DATA_AXIS, len(children)), (ownership.axis_name, ownership.owner_count))
    return bind_execution_group(
        ExecutionGroupSpec(
            canonical_fingerprint(
                {
                    "parent_group_id": root.group_id,
                    "kind": "data-domain-training",
                    "device_keys": [list(key) for key in keys],
                    "mesh_axes": [[name, size] for name, size in axes],
                }
            ),
            root.process_indices,
            keys,
            mesh_axes=axes,
            parent_group_id=root.group_id,
            process_host_ids=root.process_host_ids,
        )
    )


def _ingress(domain: Any, model: Any, row: TrainingRow, /) -> Any:
    from tools.atomistic_distributed_smoke import process_local_ingress

    return process_local_ingress(
        domain,
        model,
        row.positions,
        row.atomic_numbers,
        np.zeros_like(row.positions),
        case_identity=f"dp-domain:{row.name}",
    )


def _setup(execution: Any, strategy: Strategy, /) -> _Training:
    import optax

    from phydrax._differentiation import ComponentAuthority
    from phydrax._execution_array import replicate_tree
    from phydrax._training_kernel import (
        OptaxUpdateRule,
        prepare_training_kernel,
        TrainingKernelSpec,
    )
    from phydrax.atomistic._training_distribution import (
        atomistic_domain_training_objectives,
        AtomisticDataParallelReduction,
        AtomisticTrainingDistributionPlan,
    )
    from phydrax.discretization.spatial._distributed_relations import (
        DistributedOwnershipPlan,
    )
    from tests._support.atomistic_domain import LOCAL_CAPACITY
    from tools.atomistic_distributed_smoke import build_domain, owner_address

    children = execution.child_groups(DATA_SHARDS)
    ownership = DistributedOwnershipPlan(owner_address(), children[0], LOCAL_CAPACITY)
    model, domain = build_domain(ownership, strategy)
    group = training_group(execution, children, ownership)
    distribution = AtomisticTrainingDistributionPlan(
        group, data_axis=DATA_AXIS, domain=domain.plan
    )
    reduction = AtomisticDataParallelReduction(distribution)
    kernel = prepare_training_kernel(
        model,
        atomistic_domain_training_objectives(
            domain,
            training_objectives(),
            reduction,
            scales=(ENERGY_SCALE, FORCE_SCALE),
        ),
        TrainingKernelSpec(
            OptaxUpdateRule(
                optax.sgd(LEARNING_RATE), rule_id=f"sgd:{reduction.reduction_id}"
            ),
            context="atomistic distributed smoke data x domain training",
            rejection_budget=0,
        ),
        root_authority=ComponentAuthority.MODEL,
    )
    rows = training_rows()
    return _Training(
        domain=domain,
        model=model,
        group=group,
        distribution=distribution,
        reduction=reduction,
        kernel=kernel,
        initial=replicate_tree(kernel.init(model, jax.random.key(KERNEL_KEY)), group),
        rows=rows,
        states=tuple(_ingress(domain, model, row) for row in rows),
    )


def _shard(
    domain: Any, row: TrainingRow, state: Any, /, *, draw_index: int, case_epoch: int
) -> Any:
    """Labels of one row, owner-blocked from this process's layout rows only."""
    from phydrax._data_plane import make_global_array_from_process_local_data
    from phydrax.atomistic._training_distribution import AtomisticDomainTrainingShard
    from tools.atomistic_distributed_smoke import addressable_rows

    ownership = domain.plan.ownership
    active = addressable_rows(state.layout.active)
    logical = np.where(active, addressable_rows(state.layout.logical_indices), 0)

    def owner_blocked(local: np.ndarray) -> jax.Array:
        return make_global_array_from_process_local_data(
            ownership.sharding(local.ndim),
            local,
            global_shape=(ownership.total_capacity,) + local.shape[1:],
        )

    forces = np.where(active[:, None], row.forces[logical], 0.0).astype(np.float64)
    available = active & row.force_available[logical]
    return AtomisticDomainTrainingShard.from_domain_state(
        domain,
        state,
        labels={
            "energy": (
                jnp.asarray([row.energy], dtype=jnp.float64),
                jnp.asarray([row.energy_available]),
            ),
            "forces": (owner_blocked(forces), owner_blocked(available)),
        },
        label_levels={"energy": "structure", "forces": "atom"},
        case_weight=row.weight,
        draw_index=draw_index,
        case_epoch=case_epoch,
    )


def _batch(
    training: _Training, rows: Sequence[TrainingRow], /, *, case_epoch: int
) -> Any:
    shards = tuple(
        _shard(training.domain, row, state, draw_index=index, case_epoch=case_epoch)
        for index, (row, state) in enumerate(zip(rows, training.states, strict=True))
    )
    return training.reduction.assemble(
        tuple(shards[index] for index in training.distribution.local_shards)
    )


def _attempt(training: _Training, state: Any, batch: Any, /) -> tuple[Any, Any]:
    from phydrax.atomistic._training_distribution import (
        run_distributed_training_attempt,
    )

    return run_distributed_training_attempt(
        training.kernel, training.reduction, state, batch
    )


# ------------------------------------------------------------ host evidence


def _host(leaf: Any, /) -> np.ndarray:
    """This process's replica/rows of one leaf (typed keys as key data)."""
    from tools.atomistic_distributed_smoke import addressable_rows

    if not isinstance(leaf, jax.Array):
        return np.asarray(leaf)
    if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
        leaf = jax.random.key_data(leaf)
    return addressable_rows(leaf)


def kernel_digest(state: Any, /) -> dict[str, str]:
    """SHA-256 of every leaf of this process's replicated kernel state."""
    return {
        jax.tree_util.keystr(path): hashlib.sha256(
            np.ascontiguousarray(_host(leaf)).tobytes()
        ).hexdigest()
        for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]
    }


def _replicas_identical(tree: Any, /) -> bool:
    """Every parameter leaf is replicated and its local replicas agree bitwise."""
    for leaf in jax.tree.leaves(tree):
        if not leaf.is_fully_replicated:
            return False
        replicas = [np.asarray(shard.data) for shard in leaf.addressable_shards]
        if any(not np.array_equal(replicas[0], other) for other in replicas[1:]):
            return False
    return True


# ------------------------------------------------- independent reference


@dataclass(frozen=True)
class ReferenceStep:
    """One SGD step of the single-device global MACE reference."""

    value: float
    gradient: tuple[np.ndarray, ...]
    parameters_after: tuple[np.ndarray, ...]
    energy_support: float
    force_support: float
    row_supports: tuple[tuple[float, float], ...]


def reference_steps(
    model: Any, rows: Sequence[TrainingRow], steps: int, /
) -> tuple[ReferenceStep, ...]:
    """Independent single-device objective/gradient/SGD steps of the same model.

    Energies come from the brute-force global image-route oracle (not the
    domain route), forces are ``-dE/dx``; the objective is
    ``Σ_k (Σ_rows n_k) / stop_gradient(Σ_rows s_k)`` with the batch route's
    own energy/force numerator and support arithmetic.
    """
    from phydrax._trainable import combine_parameters, partition_parameters
    from phydrax.atomistic._training_objectives import (
        energy_objective_term,
        force_objective_term,
    )
    from tests._support.atomistic_domain import CELL, global_energy, image_routes

    energy_objective, force_objective = training_objectives()
    parameters, model_state, fixed = partition_parameters(model)
    routes = tuple(image_routes(row.positions, CELL) for row in rows)
    zero = jnp.zeros((3, 3), dtype=jnp.float64)
    ok = jnp.asarray([True])
    overflow = jnp.asarray([False])

    def row_terms(lane: Any) -> tuple[tuple[jax.Array, ...], ...]:
        current = combine_parameters(lane, model_state, fixed)
        terms = []
        for row, route in zip(rows, routes, strict=True):
            count = row.positions.shape[0]
            coordinates = jnp.asarray(row.positions, dtype=jnp.float64)
            energy = global_energy(
                current, coordinates, row.atomic_numbers, zero, route, CELL
            )
            forces = -jax.grad(global_energy, argnums=1)(
                current, coordinates, row.atomic_numbers, zero, route, CELL
            )
            mask = jnp.ones((1, count), dtype=jnp.bool_)
            species = current.species_indices(jnp.asarray(row.atomic_numbers), mask[0])
            weight = jnp.asarray([row.weight], dtype=jnp.float64)
            energy_term = energy_objective_term(
                energy_objective,
                energy[None],
                ok,
                overflow,
                labels=jnp.asarray([row.energy], dtype=jnp.float64),
                available=jnp.asarray([row.energy_available]),
                case_valid=ok,
                case_weight=weight,
                species=species[None, :],
                atom_mask=mask,
                scale=ENERGY_SCALE,
                dtype="float64",
            )
            force_term = force_objective_term(
                force_objective,
                forces[None],
                ok,
                overflow,
                labels=jnp.asarray(row.forces, dtype=jnp.float64)[None],
                available=jnp.asarray(row.force_available)[None],
                case_valid=ok,
                case_weight=weight,
                atom_mask=mask,
                scale=FORCE_SCALE,
                dtype="float64",
            )
            terms.append(
                (
                    energy_term.numerator,
                    energy_term.support,
                    force_term.numerator,
                    force_term.support,
                )
            )
        return tuple(terms)

    def objective(lane: Any) -> tuple[jax.Array, tuple[tuple[jax.Array, ...], ...]]:
        terms = row_terms(lane)
        energy_numerator = jnp.sum(jnp.stack([term[0] for term in terms]))
        energy_support = jnp.sum(jnp.stack([term[1] for term in terms]))
        force_numerator = jnp.sum(jnp.stack([term[2] for term in terms]))
        force_support = jnp.sum(jnp.stack([term[3] for term in terms]))
        value = energy_numerator / jax.lax.stop_gradient(
            energy_support
        ) + force_numerator / jax.lax.stop_gradient(force_support)
        return value, terms

    evaluate = jax.jit(jax.value_and_grad(objective, has_aux=True))
    results = []
    lane = parameters
    with _jax_config.default_device(jax.local_devices()[0]):
        for _ in range(steps):
            (value, terms), gradient = evaluate(lane)
            lane = jax.tree.map(
                lambda leaf, step: leaf - LEARNING_RATE * step, lane, gradient
            )
            supports = tuple((float(term[1]), float(term[3])) for term in terms)
            results.append(
                ReferenceStep(
                    value=float(value),
                    gradient=tuple(
                        np.asarray(leaf) for leaf in jax.tree.leaves(gradient)
                    ),
                    parameters_after=tuple(
                        np.asarray(leaf) for leaf in jax.tree.leaves(lane)
                    ),
                    energy_support=math.fsum(energy for energy, _ in supports),
                    force_support=math.fsum(force for _, force in supports),
                    row_supports=supports,
                )
            )
    return tuple(results)


def compare_step(
    before: Any, after: Any, evidence: Any, reference: ReferenceStep, /
) -> tuple[dict[str, Any], list[str]]:
    """Objective, gradient, update and support errors of one distributed step."""
    errors: list[str] = []
    start = [_host(leaf) for leaf in jax.tree.leaves(before.parameters)]
    updated = [_host(leaf) for leaf in jax.tree.leaves(after.parameters)]
    if len(updated) != len(reference.parameters_after):
        errors.append("distributed and reference parameter trees differ")
        return {}, errors
    gradient = [(old - new) / LEARNING_RATE for old, new in zip(start, updated)]
    gradient_error = max(
        float(np.max(np.abs(left - right)))
        for left, right in zip(gradient, reference.gradient, strict=True)
    )
    gradient_scale = max(float(np.max(np.abs(leaf))) for leaf in reference.gradient)
    parameter_error = max(
        float(np.max(np.abs(left - right)))
        for left, right in zip(updated, reference.parameters_after, strict=True)
    )
    parameters_close = all(
        np.allclose(
            left,
            right,
            rtol=PARAMETER_RELATIVE_TOLERANCE,
            atol=PARAMETER_ABSOLUTE_TOLERANCE,
        )
        for left, right in zip(updated, reference.parameters_after, strict=True)
    )
    value = float(_host(evidence.value))
    objective_error = abs(value - reference.value)
    gradient_norm = float(_host(evidence.gradient_norm))
    reference_norm = math.sqrt(
        math.fsum(float(np.vdot(leaf, leaf)) for leaf in reference.gradient)
    )
    energy_term, force_term = evidence.diagnostics
    energy_support = float(_host(energy_term.support))
    force_support = float(_host(force_term.support))
    capacity_failures = int(_host(energy_term.capacity_failures)) + int(
        _host(force_term.capacity_failures)
    )
    scientific_failures = int(_host(energy_term.scientific_failures)) + int(
        _host(force_term.scientific_failures)
    )
    supported = bool(_host(evidence.supported))
    finite = bool(_host(evidence.finite))
    accepted = bool(_host(evidence.rule_accepted))
    if not (supported and finite and accepted):
        errors.append(
            f"attempt not accepted (supported={supported}, finite={finite}, "
            f"accepted={accepted})"
        )
    if capacity_failures or scientific_failures:
        errors.append(
            f"failures: capacity={capacity_failures}, scientific={scientific_failures}"
        )
    if not objective_error <= OBJECTIVE_RELATIVE_TOLERANCE * abs(reference.value):
        errors.append(f"objective differs from the reference by {objective_error}")
    if not gradient_error <= GRADIENT_RELATIVE_TOLERANCE * gradient_scale:
        errors.append(
            f"parameter gradient differs from the reference by {gradient_error}"
        )
    if not abs(gradient_norm - reference_norm) <= (
        GRADIENT_NORM_RELATIVE_TOLERANCE * reference_norm
    ):
        errors.append(f"gradient norm {gradient_norm} differs from {reference_norm}")
    if not parameters_close:
        errors.append(
            f"updated parameters differ from the reference by {parameter_error}"
        )
    if abs(energy_support - reference.energy_support) > SUPPORT_ABSOLUTE_TOLERANCE:
        errors.append(
            f"energy support {energy_support} is not Σ shards {reference.energy_support}"
        )
    if abs(force_support - reference.force_support) > SUPPORT_ABSOLUTE_TOLERANCE:
        errors.append(
            f"force support {force_support} is not Σ shards {reference.force_support}"
        )
    if not _replicas_identical(after.parameters):
        errors.append("updated parameters are not identical replicas on this process")
    return {
        "value": value,
        "reference_value": reference.value,
        "objective_error": objective_error,
        "gradient_error": gradient_error,
        "gradient_scale": gradient_scale,
        "parameter_error": parameter_error,
        "gradient_norm": gradient_norm,
        "reference_gradient_norm": reference_norm,
        "energy_support": energy_support,
        "force_support": force_support,
        "reference_energy_support": reference.energy_support,
        "reference_force_support": reference.force_support,
        "row_supports": [list(item) for item in reference.row_supports],
        "capacity_failures": capacity_failures,
        "scientific_failures": scientific_failures,
        "outcome": int(_host(evidence.outcome)),
    }, errors


# ------------------------------------------------------------ rank entry


def training_reference_path(directory: Path, process: int, /) -> Path:
    """Per-rank digests of the uninterrupted ``dp-domain`` run."""
    return Path(directory) / TRAINING_REFERENCE_DIRECTORY / f"rank-{process}.json"


def _topology(training: _Training, /) -> dict[str, Any]:
    distribution = training.distribution
    ownership = training.domain.plan.ownership
    return {
        "data_axis": distribution.data_axis,
        "data_shards": distribution.data_shards,
        "owner_axis": distribution.owner_axis,
        "owner_count": distribution.owner_count,
        "mesh_axes": [
            [name, int(size)] for name, size in training.group.mesh.shape.items()
        ],
        "local_shards": list(distribution.local_shards),
        "training_group_id": training.group.spec.group_id,
        "domain_group_id": ownership.execution_group.spec.group_id,
        "owner_processes": list(ownership.owner_processes),
        "strategy": training.domain.plan.strategy.value,
        "domain_plan_id": training.domain.plan.plan_id,
        "plan_id": distribution.plan_id,
        "reduction_id": training.reduction.reduction_id,
        "reduction_mode": distribution.reduction.mode,
        "collective_schedule": list(distribution.collective_schedule),
        "rows": [
            {
                "name": row.name,
                "atoms": int(row.positions.shape[0]),
                "energy_available": row.energy_available,
                "force_rows_available": int(np.sum(row.force_available)),
                "weight": row.weight,
            }
            for row in training.rows
        ],
    }


def _evidence_bits(evidence: Any, /) -> list[str]:
    """Exact (hex) objective value and gradient norm of one attempt."""
    return [
        float(_host(evidence.value)).hex(),
        float(_host(evidence.gradient_norm)).hex(),
    ]


def run_training_rank(
    scenario: TrainingScenario, directory: Path, strategy: Strategy
) -> dict[str, Any]:
    """Execute one data×domain training scenario as one rank of a coupled launch."""
    from phydrax.atomistic._domain_checkpoint import (
        publish_domain_checkpoint,
        restore_domain_checkpoint,
    )
    from phydrax.execution import ExecutionRuntime
    from tools._accelerator_process import read_json_evidence, write_json_evidence
    from tools.atomistic_distributed_smoke import domain_repository, runtime_record

    if scenario not in TRAINING_SCENARIOS:
        raise ValueError(f"Unknown training scenario {scenario!r}.")
    execution = ExecutionRuntime.current()
    process = jax.process_index()
    record: dict[str, Any] = {
        "scenario": scenario,
        "strategy": strategy,
        "runtime": runtime_record(execution),
    }
    errors: list[str] = []
    training = _setup(execution, strategy)
    topology = _topology(training)
    if training.distribution.data_shards < 2 or training.distribution.owner_count < 2:
        errors.append(
            "data×domain qualification needs data and owner axes both larger than one; "
            f"observed data={training.distribution.data_shards}, "
            f"owners={training.distribution.owner_count}"
        )
    first = _batch(training, training.rows, case_epoch=0)
    second = _batch(training, training.rows, case_epoch=1)
    references = reference_steps(training.model, training.rows, 2)
    summary: dict[str, Any] = {"topology": topology}
    repository = domain_repository(directory)
    if scenario == "dp-domain":
        state1, evidence1 = _attempt(training, training.initial, first)
        state2, evidence2 = _attempt(training, state1, second)
        steps = []
        for before, after, evidence, reference in (
            (training.initial, state1, evidence1, references[0]),
            (state1, state2, evidence2, references[1]),
        ):
            step, step_errors = compare_step(before, after, evidence, reference)
            steps.append(step)
            errors.extend(f"step {len(steps)}: {error}" for error in step_errors)
        perturbed = _batch(
            training, tuple(row.perturbed() for row in training.rows), case_epoch=0
        )
        inert_state, inert_evidence = _attempt(training, training.initial, perturbed)
        inert = kernel_digest(inert_state) == kernel_digest(state1) and (
            _evidence_bits(inert_evidence) == _evidence_bits(evidence1)
        )
        if not inert:
            errors.append("perturbing unavailable labels changed the update")
        publication = publish_domain_checkpoint(
            repository,
            training.domain,
            training.states[0],
            checkpoint_id=TRAINING_CHECKPOINT,
            writer_id=f"rank-{process}",
            sections={"training": state1},
            training_plan_id=training.distribution.plan_id,
            limits=TRAINING_CHECKPOINT_LIMITS,
        )
        digests = {"step1": kernel_digest(state1), "step2": kernel_digest(state2)}
        bits = {"step1": _evidence_bits(evidence1), "step2": _evidence_bits(evidence2)}
        reference_path = training_reference_path(directory, process)
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_evidence(
            reference_path,
            {
                "plan_id": training.distribution.plan_id,
                "manifest_id": publication.manifest.manifest_id,
                "digests": digests,
                "evidence_bits": bits,
            },
        )
        summary.update(
            steps=steps,
            inert_empty_support={"bitwise": inert, "perturbed_label": PERTURBED_LABEL},
            manifest_id=publication.manifest.manifest_id,
            record_id=publication.record.record_id,
            digests=digests,
            evidence_bits=bits,
        )
    else:
        template = _ingress(training.domain, training.model, training.rows[0])
        restored = restore_domain_checkpoint(
            repository,
            training.domain,
            template,
            TRAINING_CHECKPOINT,
            section_templates={"training": training.initial},
            limits=TRAINING_CHECKPOINT_LIMITS,
        )
        restored1 = restored.sections["training"]
        if restored.restart_class != "bitwise":
            errors.append(f"same-topology restart class is {restored.restart_class!r}")
        if restored.record.training_plan_id != training.distribution.plan_id:
            errors.append("the checkpoint is bound to another training plan")
        continued, continued_evidence = _attempt(training, restored1, second)
        state1, evidence1 = _attempt(training, training.initial, first)
        state2, evidence2 = _attempt(training, state1, second)
        step, step_errors = compare_step(
            restored1, continued, continued_evidence, references[1]
        )
        errors.extend(f"continued step 2: {error}" for error in step_errors)
        digests = {
            "restored_step1": kernel_digest(restored1),
            "step2": kernel_digest(continued),
        }
        bits = {"step2": _evidence_bits(continued_evidence)}
        if digests["restored_step1"] != kernel_digest(state1):
            errors.append("the restored kernel state differs from step 1 of this launch")
        if digests["step2"] != kernel_digest(state2) or bits["step2"] != _evidence_bits(
            evidence2
        ):
            errors.append("the continued step 2 differs from the uninterrupted run")
        reference_path = training_reference_path(directory, process)
        recorded: dict[str, Any] | None = None
        if reference_path.exists():
            recorded = read_json_evidence(reference_path)
            if recorded["plan_id"] != training.distribution.plan_id:
                errors.append("dp-domain recorded another training plan")
            if recorded["digests"]["step1"] != digests["restored_step1"]:
                errors.append("the restored kernel state differs from dp-domain step 1")
            if (
                recorded["digests"]["step2"] != digests["step2"]
                or recorded["evidence_bits"]["step2"] != bits["step2"]
            ):
                errors.append("the continued step 2 differs from dp-domain step 2")
        else:
            errors.append(
                f"no dp-domain reference record at {reference_path}; run dp-domain first"
            )
        summary.update(
            steps=[step],
            restart_class=restored.restart_class,
            relation_id=restored.relation_id,
            admission_id=restored.admission_id,
            manifest_id=None if recorded is None else recorded["manifest_id"],
            record_id=restored.record.record_id,
            digests=digests,
            evidence_bits=bits,
        )
    record["training"] = summary
    record["errors"] = errors
    record["passed"] = not errors
    return record


# ------------------------------------------------------------ parent checks


def training_checks(
    scenario: TrainingScenario,
    ranks: list[dict[str, Any] | None],
    processes: int,
    launch: dict[str, Any],
    directory: Path,
) -> list[str]:
    """Cross-rank checks of one coupled data×domain training launch."""
    from tools.atomistic_distributed_smoke import commit_present, domain_repository

    errors: list[str] = []
    if launch["error"] is not None:
        errors.append(f"coupled launch failed: {launch['error']}")
    if len(ranks) != processes or any(rank is None for rank in ranks):
        errors.append("missing terminal rank records")
        return errors
    complete = [rank for rank in ranks if rank is not None]
    for index, rank in enumerate(complete):
        if rank["runtime"]["process_index"] != index:
            errors.append(f"rank {index} reported another process index")
        if rank["runtime"]["process_count"] != processes:
            errors.append(f"rank {index} observed another process count")
        errors.extend(f"rank {index}: {error}" for error in rank["errors"])
    if len({rank["runtime"]["execution_group_id"] for rank in complete}) != 1:
        errors.append("ranks bound different execution groups")
    summaries = [rank["training"] for rank in complete]
    for key in ("plan_id", "reduction_id", "domain_plan_id", "training_group_id"):
        if len({summary["topology"][key] for summary in summaries}) != 1:
            errors.append(f"ranks disagree on {key}")
    if (
        len({json.dumps(summary["digests"], sort_keys=True) for summary in summaries})
        != 1
    ):
        errors.append("ranks hold different kernel states")
    if (
        len(
            {
                json.dumps(summary["evidence_bits"], sort_keys=True)
                for summary in summaries
            }
        )
        != 1
    ):
        errors.append("ranks observed different objective values or gradient norms")
    topology = summaries[0]["topology"]
    if topology["data_shards"] < 2 or topology["owner_count"] < 2:
        errors.append("the launch did not compose data and owner axes larger than one")
    if scenario == "dp-domain":
        if len({summary["manifest_id"] for summary in summaries}) != 1:
            errors.append("ranks committed different manifests")
        if not commit_present(
            domain_repository(directory),
            TRAINING_CHECKPOINT,
            limits=TRAINING_CHECKPOINT_LIMITS,
        ):
            errors.append("the training checkpoint commit is not loadable")
    elif any(summary["restart_class"] != "bitwise" for summary in summaries):
        errors.append("a rank did not restore bitwise")
    return errors


__all__ = (
    "DATA_SHARDS",
    "ENERGY_SCALE",
    "FORCE_SCALE",
    "GRADIENT_NORM_RELATIVE_TOLERANCE",
    "GRADIENT_RELATIVE_TOLERANCE",
    "LEARNING_RATE",
    "OBJECTIVE_RELATIVE_TOLERANCE",
    "PARAMETER_ABSOLUTE_TOLERANCE",
    "PARAMETER_RELATIVE_TOLERANCE",
    "ReferenceStep",
    "run_training_rank",
    "SUPPORT_ABSOLUTE_TOLERANCE",
    "TRAINING_CHECKPOINT",
    "TRAINING_REFERENCE_DIRECTORY",
    "TRAINING_SCENARIOS",
    "TrainingRow",
    "TrainingScenario",
    "compare_step",
    "kernel_digest",
    "reference_steps",
    "training_checks",
    "training_group",
    "training_objectives",
    "training_reference_path",
    "training_rows",
)
