#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.applications.incompressible_flow import (
    PeriodicFlowEnsembleResult,
    PeriodicFlowInitialMeasure,
    PeriodicFlowInitialState,
    restrict_periodic_flow_initial_state,
    TaylorGreenFamily,
)
from phydrax.applications.incompressible_flow._operator_data import (
    periodic_flow_operator_dataset,
)
from phydrax.applications.incompressible_flow._statistical_evidence import _member_labels
from phydrax.discretization import (
    AxisDomain,
    DiscretizationBundle,
    DiscretizationRecord,
    FourierBasisPlan,
    TensorSpectralPlan,
)
from phydrax.discretization.spectral import PeriodicLerayProjector
from phydrax.discretization.spectral._space import TensorSpectralDiscretization
from phydrax.execution import MemoryBudgetError
from phydrax.nn.operator.training import (
    split_operator_dataset,
)
from phydrax.solver import (
    execute_fixed_step_ensemble,
    FixedStepEnsemblePlan,
    FixedStepEnsembleResult,
    FixedStepProblem,
    SSPRK33FixedStepMethod,
)


def _space(count: int = 7, dimension: int = 2) -> TensorSpectralDiscretization:
    return TensorSpectralPlan(
        tuple(FourierBasisPlan(count) for _ in range(dimension))
    ).prepare(tuple(AxisDomain.periodic(0.0, 1.0) for _ in range(dimension)))


def _decay(time: Array, state: Array, args: Any, /) -> Array:
    del time, args
    return -state


def _result(initial: PeriodicFlowInitialState) -> FixedStepEnsembleResult:
    problem = FixedStepProblem(
        SSPRK33FixedStepMethod(_decay),
        initial.velocity[0],
        t0=0.0,
        t1=0.02,
        step_size=0.01,
        discretization_bundle=DiscretizationBundle(
            (
                DiscretizationRecord(
                    initial.discretization.key,
                    "TensorSpectralDiscretization",
                    initial.discretization_id,
                ),
            )
        ),
    )
    plan = FixedStepEnsemblePlan(
        problem,
        initial.velocity,
        _member_labels(initial),
        member_chunk_size=2,
    )
    return execute_fixed_step_ensemble(plan.prepare())


@pytest.fixture
def campaign() -> tuple[PeriodicFlowInitialState, FixedStepEnsembleResult]:
    initial = (
        PeriodicFlowInitialMeasure(TaylorGreenFamily(2))
        .prepare(_space())
        .realize(jax.random.key(12), np.array([9, 2, 7, 0, 3, 5], dtype=np.uint32))
    )
    return initial, _result(initial)


def test_member_major_layout_and_lead_time_conditions(
    campaign: tuple[PeriodicFlowInitialState, FixedStepEnsembleResult],
) -> None:
    initial, result = campaign
    dataset = periodic_flow_operator_dataset(
        PeriodicFlowEnsembleResult(result, initial), target_time_indices=(2, 1)
    )
    assert dataset.size == 12
    assert dataset.batch.case_axes == ("case",)
    assert dataset.batch.inputs["initial"].values is not None
    np.testing.assert_allclose(
        dataset.batch.inputs["initial"].values,
        np.repeat(
            jax.vmap(initial.discretization.reconstruct)(initial.velocity), 2, axis=0
        ),
    )
    expected = jax.vmap(jax.vmap(initial.discretization.reconstruct))(
        result.states[:, [2, 1]]
    )
    np.testing.assert_allclose(
        dataset.targets.fields["velocity"].values, expected.reshape(12, 7, 7, 2)
    )
    assert dataset.batch.inputs["conditions"].values is not None
    np.testing.assert_allclose(
        dataset.batch.inputs["conditions"].values,
        np.tile([0.02, 0.01], 6).reshape(12, 1, 1),
    )
    assert dataset.provenance is not None
    assert [p.order["member"] for p in dataset.provenance] == [
        9,
        9,
        2,
        2,
        7,
        7,
        0,
        0,
        3,
        3,
        5,
        5,
    ]
    assert all(tuple(p.identities) == ("realization",) for p in dataset.provenance)
    assert dataset.batch.queries["velocity"].has_physical_quadrature


def test_default_grouped_split_keeps_all_member_times_together(
    campaign: tuple[PeriodicFlowInitialState, FixedStepEnsembleResult],
) -> None:
    initial, result = campaign
    dataset = periodic_flow_operator_dataset(
        PeriodicFlowEnsembleResult(result, initial), target_time_indices=(1, 2)
    )
    split = split_operator_dataset(dataset, train_fraction=0.5, validation_fraction=0.25)
    groups = []
    for part in (split.train, split.validation, split.test):
        assert part.provenance is not None
        groups.append({p.identities["realization"] for p in part.provenance})
        assert part.size % 2 == 0
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert len(set.union(*groups)) == 6


def test_coupled_resolutions_share_only_realization_group_identity(
    campaign: tuple[PeriodicFlowInitialState, FixedStepEnsembleResult],
) -> None:
    fine, result = campaign
    coarse = restrict_periodic_flow_initial_state(fine, _space(5))
    a = periodic_flow_operator_dataset(
        PeriodicFlowEnsembleResult(result, fine), target_time_indices=(1, 2)
    )
    b = periodic_flow_operator_dataset(
        PeriodicFlowEnsembleResult(_result(coarse), coarse), target_time_indices=(1, 2)
    )
    assert a.provenance is not None and b.provenance is not None
    for left, right in zip(a.provenance, b.provenance, strict=True):
        assert left.identities == right.identities
        assert left.case_id != right.case_id
    # The default deterministic grouping preserves coupled partitions as well.
    splits = tuple(
        split_operator_dataset(dataset, train_fraction=0.5, validation_fraction=0.25)
        for dataset in (a, b)
    )
    for left, right in zip(
        (splits[0].train, splits[0].validation, splits[0].test),
        (splits[1].train, splits[1].validation, splits[1].test),
        strict=True,
    ):
        assert left.provenance is not None and right.provenance is not None
        assert {p.identities["realization"] for p in left.provenance} == {
            p.identities["realization"] for p in right.provenance
        }
    assert len({p.identities["realization"] for p in a.provenance}) == 6


@pytest.mark.parametrize("dimension", [2, 3])
def test_vorticity_is_native_spectral_curl(dimension: int) -> None:
    space = _space(7, dimension)
    initial = (
        PeriodicFlowInitialMeasure(TaylorGreenFamily(dimension))
        .prepare(space)
        .realize(jax.random.key(4), np.array([0], dtype=np.uint32))
    )
    result = _result(initial)
    dataset = periodic_flow_operator_dataset(
        PeriodicFlowEnsembleResult(result, initial),
        field="vorticity",
        target_time_indices=(1,),
    )
    angles = np.meshgrid(
        *(2 * np.pi * np.asarray(a.nodes) for a in space.axes), indexing="ij"
    )
    x, y = angles[:2]
    if dimension == 2:
        expected = (4 * np.pi * np.sin(x) * np.sin(y))[..., None]
    else:
        z = angles[2]
        expected = (
            2
            * np.pi
            * np.stack(
                (
                    -np.cos(x) * np.sin(y) * np.sin(z),
                    -np.sin(x) * np.cos(y) * np.sin(z),
                    2 * np.sin(x) * np.sin(y) * np.cos(z),
                ),
                axis=-1,
            )
        )
    assert dataset.batch.inputs["initial"].values is not None
    np.testing.assert_allclose(
        dataset.batch.inputs["initial"].values[0], expected, atol=3e-13
    )
    factor = 1 - 0.01 + 0.01**2 / 2 - 0.01**3 / 6
    np.testing.assert_allclose(
        dataset.targets.fields["vorticity"].values[0], factor * expected, atol=3e-13
    )


def test_retained_band_downsampling_matches_restricted_initial_state(
    campaign: tuple[PeriodicFlowInitialState, FixedStepEnsembleResult],
) -> None:
    initial, result = campaign
    output = _space(4)
    restricted = restrict_periodic_flow_initial_state(initial, output)
    dataset = periodic_flow_operator_dataset(
        PeriodicFlowEnsembleResult(result, initial),
        target_time_indices=(1,),
        output_discretization=output,
    )
    assert dataset.batch.inputs["initial"].values is not None
    np.testing.assert_allclose(
        dataset.batch.inputs["initial"].values,
        jax.vmap(output.reconstruct)(restricted.velocity),
        atol=2e-14,
    )
    assert dataset.targets.fields["velocity"].values.shape == (6, 4, 4, 2)
    for got, axis in zip(
        dataset.batch.queries["velocity"].axes, output.axes, strict=True
    ):
        np.testing.assert_array_equal(got.nodes, axis.nodes)


def test_downsampling_discards_high_modes_instead_of_aliasing(
    campaign: tuple[PeriodicFlowInitialState, FixedStepEnsembleResult],
) -> None:
    initial, result = campaign
    space = initial.discretization
    x, y = jnp.meshgrid(*(axis.nodes for axis in space.axes), indexing="ij")
    high = jnp.stack((jnp.zeros_like(y), jnp.sin(6 * jnp.pi * x)), axis=-1)
    high_modal = space.project(high)
    perturbed = eqx.tree_at(
        lambda r: r.states,
        result,
        result.states + high_modal[None, None],
    )
    output = _space(4)
    baseline = periodic_flow_operator_dataset(
        PeriodicFlowEnsembleResult(result, initial),
        target_time_indices=(1,),
        output_discretization=output,
    )
    downsampled = periodic_flow_operator_dataset(
        PeriodicFlowEnsembleResult(perturbed, initial),
        target_time_indices=(1,),
        output_discretization=output,
    )
    np.testing.assert_allclose(
        downsampled.targets.fields["velocity"].values,
        baseline.targets.fields["velocity"].values,
        atol=2e-14,
    )
    native = periodic_flow_operator_dataset(
        PeriodicFlowEnsembleResult(perturbed, initial),
        target_time_indices=(1,),
    )
    unperturbed = periodic_flow_operator_dataset(
        PeriodicFlowEnsembleResult(result, initial),
        target_time_indices=(1,),
    )
    assert (
        np.max(
            np.abs(
                np.asarray(native.targets.fields["velocity"].values)
                - np.asarray(unperturbed.targets.fields["velocity"].values)
            )
        )
        > 0.9
    )


@pytest.mark.parametrize("targets", [(), (0,), (1, 1), (99,), (-99,)])
def test_invalid_time_selection_is_refused(
    campaign: tuple[PeriodicFlowInitialState, FixedStepEnsembleResult],
    targets: tuple[int, ...],
) -> None:
    initial, result = campaign
    with pytest.raises(ValueError):
        periodic_flow_operator_dataset(
            PeriodicFlowEnsembleResult(result, initial), target_time_indices=targets
        )


@pytest.mark.parametrize(
    ("input_index", "targets"), [(False, (1,)), (0, (True,))], ids=["input", "target"]
)
def test_boolean_time_indices_are_refused(
    campaign: tuple[PeriodicFlowInitialState, FixedStepEnsembleResult],
    input_index: object,
    targets: tuple[object, ...],
) -> None:
    initial, result = campaign
    with pytest.raises(TypeError):
        periodic_flow_operator_dataset(
            PeriodicFlowEnsembleResult(result, initial),
            input_time_index=input_index,  # ty: ignore[invalid-argument-type]
            target_time_indices=targets,  # ty: ignore[invalid-argument-type]
        )


def test_selected_validity_and_finiteness_are_fail_closed(
    campaign: tuple[PeriodicFlowInitialState, FixedStepEnsembleResult],
) -> None:
    initial, result = campaign
    invalid = eqx.tree_at(lambda r: r.valid, result, result.valid.at[0, 2].set(False))
    with pytest.raises(ValueError, match="valid"):
        periodic_flow_operator_dataset(
            PeriodicFlowEnsembleResult(invalid, initial), target_time_indices=(2,)
        )
    assert (
        periodic_flow_operator_dataset(
            PeriodicFlowEnsembleResult(invalid, initial), target_time_indices=(1,)
        ).size
        == 6
    )
    nonfinite = eqx.tree_at(
        lambda r: r.states, result, result.states.at[0, 1, 0, 0, 0].set(jnp.nan)
    )
    with pytest.raises(ValueError, match="finite"):
        periodic_flow_operator_dataset(
            PeriodicFlowEnsembleResult(nonfinite, initial), target_time_indices=(1,)
        )


def test_resource_and_discretization_refusals(
    campaign: tuple[PeriodicFlowInitialState, FixedStepEnsembleResult],
) -> None:
    initial, result = campaign
    with pytest.raises(MemoryBudgetError, match="maximum_dataset_bytes"):
        periodic_flow_operator_dataset(
            PeriodicFlowEnsembleResult(result, initial),
            target_time_indices=(1, 2),
            maximum_dataset_bytes=1,
        )
    with pytest.raises(ValueError, match="refinement"):
        periodic_flow_operator_dataset(
            PeriodicFlowEnsembleResult(result, initial),
            target_time_indices=(1,),
            output_discretization=_space(9),
        )
    wrong_owner = eqx.tree_at(lambda i: i.discretization, initial, _space(5))
    with pytest.raises(ValueError, match="identity"):
        periodic_flow_operator_dataset(
            PeriodicFlowEnsembleResult(result, wrong_owner), target_time_indices=(1,)
        )
    with pytest.raises(ValueError):
        periodic_flow_operator_dataset(
            PeriodicFlowEnsembleResult(result, initial),
            target_time_indices=(1,),
            field="pressure",  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("dimension", [2, 3])
def test_curl_biot_savart_round_trip(dimension: int) -> None:
    space = _space(5, dimension)
    initial = (
        PeriodicFlowInitialMeasure(TaylorGreenFamily(dimension))
        .prepare(space)
        .realize(jax.random.key(4), np.asarray([0, 2], dtype=np.uint32))
    )
    projector = PeriodicLerayProjector(space)
    vorticity = projector.vorticity_from_velocity(initial.velocity)
    velocity = projector.velocity_from_vorticity(vorticity, tolerance=1e-10)
    np.testing.assert_allclose(velocity, initial.velocity, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        projector.vorticity_from_velocity(velocity), vorticity, rtol=1e-12, atol=1e-12
    )


def test_dataset_refuses_relabeling_and_distinguishes_measure_leakage(
    campaign: tuple[PeriodicFlowInitialState, FixedStepEnsembleResult],
) -> None:
    initial, native = campaign
    foreign = (
        PeriodicFlowInitialMeasure(TaylorGreenFamily(2, amplitude=2.0))
        .prepare(initial.discretization)
        .realize(jax.random.key(12), np.asarray(initial.member_indices))
    )
    with pytest.raises(ValueError, match="semantic paths"):
        PeriodicFlowEnsembleResult(native, foreign)
    result = PeriodicFlowEnsembleResult(native, initial)
    with pytest.raises(TypeError):
        periodic_flow_operator_dataset(native, target_time_indices=(1,))  # ty: ignore[invalid-argument-type]
    left = periodic_flow_operator_dataset(result, target_time_indices=(1,))
    right = periodic_flow_operator_dataset(
        PeriodicFlowEnsembleResult(_result(foreign), foreign), target_time_indices=(1,)
    )
    assert left.provenance is not None and right.provenance is not None
    assert not {p.identities["realization"] for p in left.provenance} & {
        p.identities["realization"] for p in right.provenance
    }


def test_dataset_chunk_policy_admits_retained_and_transient_bytes(
    campaign: tuple[PeriodicFlowInitialState, FixedStepEnsembleResult],
) -> None:
    initial, native = campaign
    result = PeriodicFlowEnsembleResult(native, initial)
    with pytest.raises(MemoryBudgetError) as one:
        periodic_flow_operator_dataset(
            result,
            target_time_indices=(1, 2),
            member_chunk_size=1,
            maximum_dataset_bytes=1,
        )
    with pytest.raises(MemoryBudgetError) as six:
        periodic_flow_operator_dataset(
            result,
            target_time_indices=(1, 2),
            member_chunk_size=6,
            maximum_dataset_bytes=1,
        )
    assert one.value.required_bytes < six.value.required_bytes
    dataset = periodic_flow_operator_dataset(
        result,
        target_time_indices=(1, 2),
        member_chunk_size=1,
        maximum_dataset_bytes=one.value.required_bytes,
    )
    assert dataset.size == 12
    with pytest.raises(MemoryBudgetError):
        periodic_flow_operator_dataset(
            result,
            target_time_indices=(1, 2),
            member_chunk_size=6,
            maximum_dataset_bytes=one.value.required_bytes,
        )
