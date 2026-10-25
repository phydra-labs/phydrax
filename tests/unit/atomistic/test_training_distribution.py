#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Data-parallel training reduction, placement and kernel execution.

Lanes are the numerical oracle of the device route; the device test needs two
real or explicitly configured JAX devices. References are single-process
evaluations of the SAME objective on the concatenated shards.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from phydrax._data_plane import (
    DistributedIndexEpochPlan,
    epoch_shard,
    ExecutionGroupRank,
    IndexEpochPlan,
)
from phydrax._differentiation import ComponentAuthority, DerivativeRoute, ObjectiveKind
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import bind_execution_group
from phydrax._strict import StrictModule
from phydrax._trainable import (
    combine_parameters,
    NonTrainableState,
    parameter_field,
    partition_parameters,
)
from phydrax._training_kernel import (
    KernelObjective,
    OptaxUpdateRule,
    prepare_training_kernel,
    TrainingKernelSpec,
)
from phydrax._training_objective import _ObjectiveContribution
from phydrax.atomistic._domain_plan import (
    AtomisticDomainReductionMode,
    AtomisticDomainReductionPolicy,
)
from phydrax.atomistic._training_distribution import (
    atomistic_domain_training_objectives,
    AtomisticDataParallelReduction,
    AtomisticDomainTrainingShard,
    AtomisticTrainingDistributionPlan,
)
from phydrax.atomistic._training_objectives import (
    AbstractAtomisticObjectiveReduction,
    AtomisticEnergyObjective,
    AtomisticForceObjective,
    AtomisticLocalObjectiveReduction,
    AtomisticObjectiveTerm,
    energy_objective_term,
    force_objective_term,
)
from tests._support.atomistic_domain import (
    domain_training_fixture,
    global_energy,
    image_routes,
    structure,
)


_SHARDS = 4
_CASES = 3


class _Linear(StrictModule):
    weight: jax.Array = parameter_field()


class _WeightedLeastSquares(StrictModule, NonTrainableState):
    """Σ_c w_c (x_c·θ - y_c)² with support Σ_c w_c, through a reduction."""

    reduction: AbstractAtomisticObjectiveReduction

    def term(
        self, weight: jax.Array, payload: dict[str, jax.Array]
    ) -> AtomisticObjectiveTerm:
        residual = payload["x"] @ weight - payload["y"]
        return AtomisticObjectiveTerm(
            numerator=jnp.sum(payload["w"] * residual**2),
            support=jnp.sum(payload["w"]),
            capacity_failures=jnp.asarray(0, dtype=jnp.int32),
            scientific_failures=jnp.sum(~jnp.isfinite(residual), dtype=jnp.int32),
        )

    def __call__(
        self, parameters: Any, model_state: Any, fixed: Any, payload: Any, keys: Any
    ) -> Any:
        del fixed, keys
        bound = self.reduction.bind_parameters(parameters)
        term = self.reduction.reduce(self.term(bound.weight, payload))
        return _ObjectiveContribution(term.numerator, term.support), model_state, term


def _shards(
    seed: int = 0, *, empty: int | None = None
) -> tuple[dict[str, jax.Array], ...]:
    rng = np.random.default_rng(seed)
    shards = []
    for shard in range(_SHARDS):
        weights = rng.uniform(0.5, 2.0, _CASES)
        if shard == empty:
            weights = np.zeros(_CASES)
        shards.append(
            {
                "x": jnp.asarray(rng.normal(size=(_CASES, 2)), dtype=jnp.float64),
                "y": jnp.asarray(rng.normal(size=_CASES), dtype=jnp.float64),
                "w": jnp.asarray(weights, dtype=jnp.float64),
            }
        )
    return tuple(shards)


def _concatenated(shards: tuple[dict[str, jax.Array], ...]) -> dict[str, jax.Array]:
    return {key: jnp.concatenate([shard[key] for shard in shards]) for key in shards[0]}


def _lane_group() -> Any:
    device = jax.devices()[0]
    return bind_execution_group(
        ExecutionGroupSpec(
            "training-lanes",
            (device.process_index,),
            ((device.process_index, device.id),),
            mesh_axes=(("device", 1),),
        )
    )


def _lane_reduction(
    mode: AtomisticDomainReductionMode = "deterministic", lanes: int = _SHARDS
) -> Any:
    plan = AtomisticTrainingDistributionPlan(
        _lane_group(), data_lanes=lanes, reduction=AtomisticDomainReductionPolicy(mode)
    )
    return AtomisticDataParallelReduction(plan)


def _kernel(reduction: AbstractAtomisticObjectiveReduction) -> Any:
    tree = _Linear(jnp.asarray([0.3, -0.7], dtype=jnp.float64))
    kernel = prepare_training_kernel(
        tree,
        (
            KernelObjective(
                objective_id="least-squares",
                kind=ObjectiveKind.DATA_FIT,
                route=DerivativeRoute.DIRECT,
                fn=_WeightedLeastSquares(reduction),
            ),
        ),
        TrainingKernelSpec(
            OptaxUpdateRule(optax.sgd(0.1), rule_id=f"sgd:{reduction.reduction_id}"),
            context="training distribution test",
            rejection_budget=0,
        ),
        root_authority=ComponentAuthority.MODEL,
    )
    return kernel, kernel.init(tree, jr.key(7))


@eqx.filter_jit
def _local_attempt(kernel: Any, state: Any, payload: Any) -> Any:
    return kernel.attempt(state, payload)


@eqx.filter_jit
def _mapped_attempt(kernel: Any, reduction: Any, state: Any, payload: Any) -> Any:
    return reduction.map_step(kernel.attempt)(state, payload)


@eqx.filter_jit
def _mapped_accumulate(kernel: Any, reduction: Any, state: Any, payload: Any) -> Any:
    return reduction.map_step(kernel.accumulate)(state, payload)


def _reference(shards: tuple[dict[str, jax.Array], ...]) -> Any:
    kernel, state = _kernel(AtomisticLocalObjectiveReduction())
    return _local_attempt(kernel, state, _concatenated(shards))


def _assert_attempts_close(left: Any, right: Any, *, rtol: float = 1e-12) -> None:
    (left_state, left_evidence), (right_state, right_evidence) = left, right
    np.testing.assert_allclose(left_evidence.value, right_evidence.value, rtol=rtol)
    np.testing.assert_allclose(
        left_evidence.gradient_norm, right_evidence.gradient_norm, rtol=rtol
    )
    np.testing.assert_allclose(
        left_state.parameters.weight, right_state.parameters.weight, rtol=rtol
    )
    assert int(left_evidence.outcome) == int(right_evidence.outcome)


@pytest.mark.parametrize("mode", ["fast", "deterministic", "compensated"])
def test_lane_data_parallel_attempt_matches_single_process_objective_and_gradient(
    mode: AtomisticDomainReductionMode,
) -> None:
    shards = _shards(1)
    reduction = _lane_reduction(mode)
    kernel, state = _kernel(reduction)
    batch = reduction.assemble(shards)
    assert batch["x"].shape == (_SHARDS, _CASES, 2)
    distributed = _mapped_attempt(kernel, reduction, state, batch)
    _assert_attempts_close(distributed, _reference(shards))
    # Reduced diagnostics are global (replicated) terms, not rank 0's share.
    (term,) = distributed[1].diagnostics
    np.testing.assert_allclose(
        term.support, sum(float(jnp.sum(shard["w"])) for shard in shards), rtol=1e-12
    )


def test_accumulated_windows_merge_global_supports_before_normalization() -> None:
    first, second = _shards(2), _shards(3)
    reduction = _lane_reduction()
    kernel, state = _kernel(reduction)
    state = _mapped_accumulate(kernel, reduction, state, reduction.assemble(first))
    distributed = _mapped_attempt(kernel, reduction, state, reduction.assemble(second))
    local_kernel, local_state = _kernel(AtomisticLocalObjectiveReduction())
    local_state = eqx.filter_jit(lambda k, s, p: k.accumulate(s, p))(
        local_kernel, local_state, _concatenated(first)
    )
    reference = _local_attempt(local_kernel, local_state, _concatenated(second))
    _assert_attempts_close(distributed, reference)


def test_zero_support_shard_is_inert() -> None:
    shards = _shards(4, empty=2)
    perturbed = tuple(
        {**shard, "x": shard["x"] * 1e3 + 5.0, "y": shard["y"] - 17.0}
        if index == 2
        else shard
        for index, shard in enumerate(shards)
    )
    reduction = _lane_reduction()
    kernel, state = _kernel(reduction)
    baseline = _mapped_attempt(kernel, reduction, state, reduction.assemble(shards))
    moved = _mapped_attempt(kernel, reduction, state, reduction.assemble(perturbed))
    # Bitwise: an empty shard contributes exact zeros to numerator, support
    # and gradient in the ordered reduction.
    assert np.array_equal(baseline[0].parameters.weight, moved[0].parameters.weight)
    assert np.array_equal(baseline[1].value, moved[1].value)
    without = tuple(shard for index, shard in enumerate(shards) if index != 2)
    _assert_attempts_close(baseline, _reference(without))


def test_every_shard_empty_is_an_unsupported_skip() -> None:
    shards = tuple({**shard, "w": jnp.zeros_like(shard["w"])} for shard in _shards(5))
    reduction = _lane_reduction()
    kernel, state = _kernel(reduction)
    next_state, evidence = _mapped_attempt(
        kernel, reduction, state, reduction.assemble(shards)
    )
    assert not bool(evidence.supported)
    assert np.array_equal(next_state.parameters.weight, state.parameters.weight)


def test_parameter_cotangent_returns_each_rank_contribution_exactly_once() -> None:
    shards = _shards(6)
    reduction = _lane_reduction()
    objective = _WeightedLeastSquares(reduction)
    batch = reduction.assemble(shards)
    weight = jnp.asarray([0.25, 0.5], dtype=jnp.float64)

    def gradient(weight_: jax.Array, shard: dict[str, jax.Array]) -> jax.Array:
        def numerator(value: jax.Array) -> jax.Array:
            bound = reduction.bind_parameters(_Linear(value))
            return reduction.reduce(objective.term(bound.weight, shard)).numerator

        return jax.grad(numerator)(weight_)

    distributed = reduction.map_step(gradient)(weight, batch)
    local = AtomisticLocalObjectiveReduction()
    reference = jax.grad(
        lambda value: (
            _WeightedLeastSquares(local).term(value, _concatenated(shards)).numerator
        )
    )(weight)
    # Σ_r ∂n_r/∂θ once: neither rank 0's share nor the rank count times it.
    np.testing.assert_allclose(distributed, reference, rtol=1e-12)


def test_failure_counts_are_global_and_poison_every_rank() -> None:
    shards = list(_shards(7))
    shards[3] = {**shards[3], "y": shards[3]["y"].at[1].set(jnp.nan)}
    reduction = _lane_reduction()
    kernel, state = _kernel(reduction)
    next_state, evidence = _mapped_attempt(
        kernel, reduction, state, reduction.assemble(tuple(shards))
    )
    (term,) = evidence.diagnostics
    assert int(term.scientific_failures) == 1
    assert not bool(evidence.finite)
    assert np.array_equal(next_state.parameters.weight, state.parameters.weight)


def test_reduction_identity_binds_layout_and_arithmetic_mode() -> None:
    identities = {
        _lane_reduction(mode).reduction_id
        for mode in ("fast", "deterministic", "compensated")
    }
    assert len(identities) == 3
    assert _lane_reduction(lanes=2).reduction_id != _lane_reduction(lanes=4).reduction_id
    assert _lane_reduction().reduction_id == _lane_reduction().reduction_id
    assert AtomisticLocalObjectiveReduction().reduction_id not in identities
    plan = _lane_reduction().plan
    assert plan.collective_schedule == (
        "objective-numerator-support:deterministic:data",
        "objective-failure-counts:psum:int32:data",
        "parameter-cotangent:deterministic:data",
    )


def test_lane_epoch_shards_equal_device_group_shards_and_global_draws() -> None:
    global_plan = IndexEpochPlan(23, 10, True, 11, 3, False)
    plan = _lane_reduction().plan
    device_spec = ExecutionGroupSpec(
        "emulated", (0, 1), ((0, 0), (0, 1), (1, 2), (1, 3)), mesh_axes=(("data", 4),)
    )
    for batch_index in range(global_plan.batch_count):
        lane_shards = plan.epoch_shards(global_plan, batch_index)
        device_shards = tuple(
            DistributedIndexEpochPlan(
                global_plan, ExecutionGroupRank(device_spec, process)
            ).shard(batch_index, shard)
            for process, shard in ((0, 0), (0, 1), (1, 2), (1, 3))
        )
        assert lane_shards == device_shards
        drawn = tuple(
            index
            for shard in lane_shards
            for index, valid in zip(shard.indices, shard.valid, strict=True)
            if valid
        )
        assert drawn == global_plan.batch(batch_index)
        assert lane_shards == tuple(
            epoch_shard(global_plan, 4, batch_index, shard) for shard in range(4)
        )


def test_subgroup_ranks_place_cases_by_group_not_process_world() -> None:
    global_plan = IndexEpochPlan(12, 6, False, 0, 0, False)
    # Processes 2 and 6 of a larger world form a child training group.
    child = ExecutionGroupSpec(
        "child",
        (6, 2),
        ((2, 4), (6, 9)),
        mesh_axes=(("data", 2),),
        parent_group_id="root",
    )
    ranks = {process: ExecutionGroupRank(child, process) for process in (2, 6)}
    assert ranks[2].rank == 0 and ranks[6].rank == 1
    local = {
        process: DistributedIndexEpochPlan(global_plan, rank).batch(1)
        for process, rank in ranks.items()
    }
    assert local[2].indices == (6, 7, 8)
    assert local[6].indices == (9, 10, 11)


def test_assemble_refuses_shards_with_different_static_identity() -> None:
    class _Shard(StrictModule):
        values: jax.Array
        layout_id: str = eqx.field(static=True)

    reduction = _lane_reduction(lanes=2)
    with pytest.raises(ValueError, match="static identity"):
        reduction.assemble(
            (_Shard(jnp.zeros(2), "layout-a"), _Shard(jnp.zeros(2), "layout-b"))
        )
    with pytest.raises(ValueError, match="holds 2 data shards"):
        reduction.assemble((_Shard(jnp.zeros(2), "layout-a"),))


def test_device_data_parallel_matches_lane_reference() -> None:
    devices = tuple(jax.devices()[:2])
    if len(devices) < 2:
        pytest.skip("requires 2 real or explicitly configured JAX devices")
    group = bind_execution_group(
        ExecutionGroupSpec(
            "training-devices",
            tuple(sorted({device.process_index for device in devices})),
            tuple((device.process_index, device.id) for device in devices),
            mesh_axes=(("data", 2),),
        )
    )
    shards = _shards(8)[:2]
    device = AtomisticDataParallelReduction(AtomisticTrainingDistributionPlan(group))
    lanes = _lane_reduction(lanes=2)
    device_kernel, device_state = _kernel(device)
    lane_kernel, lane_state = _kernel(lanes)
    on_devices = _mapped_attempt(
        device_kernel, device, device_state, device.assemble(shards)
    )
    on_lanes = _mapped_attempt(lane_kernel, lanes, lane_state, lanes.assemble(shards))
    _assert_attempts_close(on_devices, on_lanes)
    _assert_attempts_close(on_devices, _reference(shards))


# ------------------------------------------------------------- data × domain


_ENERGY_SCALE = 1.0
_FORCE_SCALE = 0.5
_LEARNING_RATE = 1e-3


def _row_data(
    positions: np.ndarray,
) -> tuple[tuple[np.ndarray, float, np.ndarray, float], ...]:
    """Per row: positions, energy label, logical-order force labels, case weight.

    Row 1 is a small displacement of row 0 (within the reuse skin).
    """

    rng = np.random.default_rng(31)
    rows = []
    for row in range(2):
        moved = positions + 0.01 * row * rng.standard_normal(positions.shape)
        forces = 0.05 * rng.standard_normal(positions.shape)
        rows.append((moved, -2.0 + 0.5 * row, forces, 1.0 + row))
    return tuple(rows)


def _domain_objectives() -> tuple[Any, Any]:
    return (
        AtomisticEnergyObjective(reduction="per-atom-structure"),
        AtomisticForceObjective(reduction="component"),
    )


def _domain_rows(owner_lanes: int, data_lanes: int) -> Any:
    """Two periodic MACE systems with labels, one domain row each.

    Force labels are owner-blocked through each state's layout.
    """

    runtime, ingress, model, _, positions, cell = domain_training_fixture(owner_lanes)
    rows = []
    for row, (moved, energy, forces, weight) in enumerate(_row_data(positions)):
        state = ingress(moved, cell)
        rows.append(
            AtomisticDomainTrainingShard.from_domain_state(
                runtime,
                state,
                labels={
                    "energy": (
                        jnp.asarray([energy], dtype=jnp.float64),
                        jnp.asarray([True]),
                    ),
                    "forces": (
                        state.layout.distribute(jnp.asarray(forces, dtype=jnp.float64)),
                        state.layout.distribute(
                            jnp.ones((positions.shape[0],), dtype=jnp.bool_)
                        ),
                    ),
                },
                label_levels={"energy": "structure", "forces": "atom"},
                case_weight=weight,
                draw_index=row,
            )
        )
    plan = AtomisticTrainingDistributionPlan(
        runtime.plan.ownership.execution_group,
        data_lanes=data_lanes,
        domain=runtime.plan,
    )
    reduction = AtomisticDataParallelReduction(plan)
    objectives = atomistic_domain_training_objectives(
        runtime,
        _domain_objectives(),
        reduction,
        scales=(_ENERGY_SCALE, _FORCE_SCALE),
    )
    kernel = prepare_training_kernel(
        model,
        objectives,
        TrainingKernelSpec(
            OptaxUpdateRule(
                optax.sgd(_LEARNING_RATE), rule_id=f"sgd:{reduction.reduction_id}"
            ),
            context="data-domain training test",
            rejection_budget=0,
        ),
        root_authority=ComponentAuthority.MODEL,
    )
    return kernel, kernel.init(model, jr.key(3)), reduction, tuple(rows)


def _parameter_leaves(state: Any) -> list[np.ndarray]:
    return [np.asarray(leaf) for leaf in jax.tree.leaves(state.parameters)]


def _global_native_reference() -> tuple[float, Any, Any]:
    """Independent single-device reference of the same MACE model and cases.

    Energies come from the non-domain global oracle (brute-force periodic image
    routes, `MACEPotential.node_energies`), forces are ``-dE/dx`` by autodiff,
    and the objective arithmetic is the batch route's own
    `energy_objective_term`/`force_objective_term` with the default per-case
    atom sum. Returns the kernel value, the PARAMETER-lane gradient of the
    weighted normalized objective and the SGD-updated PARAMETER lane.
    """

    _, _, model, species, positions, cell = domain_training_fixture(1)
    _, atomic_numbers = structure()
    parameters, model_state, fixed = partition_parameters(model)
    energy_objective, force_objective = _domain_objectives()
    rows = _row_data(positions)
    count = positions.shape[0]
    routes = tuple(image_routes(moved, cell) for moved, _, _, _ in rows)
    zero = jnp.zeros((3, 3), dtype=jnp.float64)
    mask = jnp.ones((1, count), dtype=jnp.bool_)
    valid = jnp.asarray([True])
    ok = jnp.asarray([True])
    overflow = jnp.asarray([False])

    def totals(lane: Any) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        current = combine_parameters(lane, model_state, fixed)
        energy_numerator = force_numerator = jnp.zeros((), dtype=jnp.float64)
        energy_support = force_support = jnp.zeros((), dtype=jnp.float64)
        for (moved, energy_label, force_label, weight), route in zip(
            rows, routes, strict=True
        ):
            coordinates = jnp.asarray(moved, dtype=jnp.float64)
            energy = global_energy(
                current, coordinates, atomic_numbers, zero, route, cell
            )
            forces = -jax.grad(global_energy, argnums=1)(
                current, coordinates, atomic_numbers, zero, route, cell
            )
            case_weight = jnp.asarray([weight], dtype=jnp.float64)
            energy_term = energy_objective_term(
                energy_objective,
                energy[None],
                ok,
                overflow,
                labels=jnp.asarray([energy_label], dtype=jnp.float64),
                available=jnp.asarray([True]),
                case_valid=valid,
                case_weight=case_weight,
                species=jnp.asarray(species)[None, :],
                atom_mask=mask,
                scale=_ENERGY_SCALE,
                dtype="float64",
            )
            force_term = force_objective_term(
                force_objective,
                forces[None],
                ok,
                overflow,
                labels=jnp.asarray(force_label, dtype=jnp.float64)[None],
                available=mask,
                case_valid=valid,
                case_weight=case_weight,
                atom_mask=mask,
                scale=_FORCE_SCALE,
                dtype="float64",
            )
            energy_numerator = energy_numerator + energy_term.numerator
            energy_support = energy_support + energy_term.support
            force_numerator = force_numerator + force_term.numerator
            force_support = force_support + force_term.support
        return energy_numerator, energy_support, force_numerator, force_support

    def objective(lane: Any) -> jax.Array:
        energy_numerator, energy_support, force_numerator, force_support = totals(lane)
        return energy_numerator / jax.lax.stop_gradient(
            energy_support
        ) + force_numerator / jax.lax.stop_gradient(force_support)

    value, gradient = jax.value_and_grad(objective)(parameters)
    updated = jax.tree.map(
        lambda leaf, step: leaf - _LEARNING_RATE * step, parameters, gradient
    )
    return float(value), gradient, updated


def test_data_domain_attempt_matches_independent_single_device_native_reference() -> None:
    kernel, state, reduction, rows = _domain_rows(owner_lanes=2, data_lanes=2)
    distributed_state, distributed = _mapped_attempt(
        kernel, reduction, state, reduction.assemble(rows)
    )
    value, gradient, updated = _global_native_reference()
    assert bool(distributed.supported) and bool(distributed.finite)
    # Declared roundoff contract: owner/data ordered sums vs one global sum.
    np.testing.assert_allclose(distributed.value, value, rtol=1e-10)
    np.testing.assert_allclose(
        distributed.gradient_norm,
        np.sqrt(sum(float(jnp.vdot(leaf, leaf)) for leaf in jax.tree.leaves(gradient))),
        rtol=1e-9,
    )
    for left, right in zip(
        _parameter_leaves(distributed_state), jax.tree.leaves(updated), strict=True
    ):
        np.testing.assert_allclose(left, np.asarray(right), rtol=1e-9, atol=1e-14)


def test_data_domain_attempt_matches_single_owner_sequential_reference() -> None:
    kernel, state, reduction, rows = _domain_rows(owner_lanes=2, data_lanes=2)
    distributed_state, distributed = _mapped_attempt(
        kernel, reduction, state, reduction.assemble(rows)
    )
    # DP-only reference of the same model: one owner, one data shard, the two
    # systems as two microbatches of one accumulation window.
    ref_kernel, ref_state, ref_reduction, ref_rows = _domain_rows(
        owner_lanes=1, data_lanes=1
    )
    ref_state = _mapped_accumulate(
        ref_kernel, ref_reduction, ref_state, ref_reduction.assemble(ref_rows[:1])
    )
    reference_state, reference = _mapped_attempt(
        ref_kernel, ref_reduction, ref_state, ref_reduction.assemble(ref_rows[1:])
    )
    assert bool(distributed.supported) and bool(distributed.finite)
    np.testing.assert_allclose(distributed.value, reference.value, rtol=1e-10)
    np.testing.assert_allclose(
        distributed.gradient_norm, reference.gradient_norm, rtol=1e-9
    )
    for left, right in zip(
        _parameter_leaves(distributed_state),
        _parameter_leaves(reference_state),
        strict=True,
    ):
        np.testing.assert_allclose(left, right, rtol=1e-9, atol=1e-14)
    energy, forces = distributed.diagnostics
    assert int(energy.capacity_failures) == 0 and int(forces.scientific_failures) == 0


def test_data_domain_owner_count_does_not_scale_parameter_gradient() -> None:
    two_owners = _domain_rows(owner_lanes=2, data_lanes=2)
    one_owner = _domain_rows(owner_lanes=1, data_lanes=2)
    results = [
        _mapped_attempt(kernel, reduction, state, reduction.assemble(rows))
        for kernel, state, reduction, rows in (two_owners, one_owner)
    ]
    # Owner partials are summed once over the owner axis, then the data axis.
    np.testing.assert_allclose(
        results[0][1].gradient_norm, results[1][1].gradient_norm, rtol=1e-9
    )
    np.testing.assert_allclose(results[0][1].value, results[1][1].value, rtol=1e-10)


def test_data_domain_zero_weight_row_is_inert() -> None:
    kernel, state, reduction, rows = _domain_rows(owner_lanes=2, data_lanes=2)
    inert = eqx.tree_at(
        lambda row: row.case_weight, rows[1], jnp.zeros_like(rows[1].case_weight)
    )
    _, with_inert = _mapped_attempt(
        kernel, reduction, state, reduction.assemble((rows[0], inert))
    )
    single_kernel, single_state, single_reduction, single_rows = _domain_rows(
        owner_lanes=2, data_lanes=1
    )
    _, alone = _mapped_attempt(
        single_kernel,
        single_reduction,
        single_state,
        single_reduction.assemble(single_rows[:1]),
    )
    np.testing.assert_allclose(with_inert.value, alone.value, rtol=1e-12)
    np.testing.assert_allclose(with_inert.gradient_norm, alone.gradient_norm, rtol=1e-12)
