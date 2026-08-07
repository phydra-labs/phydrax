#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.constraints import FunctionalConstraint
from phydrax.domain import (
    FixedEnd,
    SampleLayout,
    TRAJECTORY_CASE_INDEX_KEY,
    TrajectoryDatasetDomain,
    UniformAxisSpec,
)
from phydrax.integration import from_samples, over
from phydrax.operators.differential import partial_t
from phydrax.operators.integral import integral


def test_trajectory_dataset_domain_samples_coupled_data_time_points():
    inputs = jnp.arange(3.0).reshape((3, 1))
    domain = TrajectoryDatasetDomain(inputs, jnp.asarray([2, 4, 3]), dt=0.25)
    component = domain.component()
    structure = SampleLayout((("data", "t"),))

    batch = component.sample(phx.domain.PointSampling(8, layout=structure), key=jr.key(0))
    axis = batch.structure.axis_for("data")
    assert axis is not None
    assert batch.structure.axis_for("t") == axis
    assert batch["data"].dims == (axis, None)
    assert batch["t"].dims == (axis,)

    case_indices = jnp.asarray(batch[TRAJECTORY_CASE_INDEX_KEY].data, dtype=jnp.int32)
    assert jnp.allclose(batch["data"].data[:, 0], inputs[case_indices, 0])
    assert jnp.all(batch["t"].data >= domain.start)
    assert jnp.all(batch["t"].data <= domain.end_times[case_indices])


def test_trajectory_dataset_probability_integral_of_constant_is_one():
    inputs = jnp.zeros((3, 1))
    domain = TrajectoryDatasetDomain(inputs, jnp.asarray([2, 4, 3]), dt=0.25)
    component = domain.component()
    structure = SampleLayout((("data", "t"),))
    batch = component.sample(phx.domain.PointSampling(17, layout=structure), key=jr.key(1))

    realization = from_samples(over(component), batch)
    out = integral(1.0, realization)
    assert jnp.allclose(jnp.asarray(out.data), 1.0)


def test_trajectory_dataset_measure_modes():
    inputs = jnp.zeros((3, 1))
    avg = TrajectoryDatasetDomain(
        inputs,
        jnp.asarray([2, 4, 3]),
        dt=0.25,
        measure="time_integral_average",
    )
    summed = TrajectoryDatasetDomain(
        inputs,
        jnp.asarray([2, 4, 3]),
        dt=0.25,
        measure="time_integral_sum",
    )

    assert jnp.allclose(avg.component().mass.value, jnp.mean(avg.durations))
    assert jnp.allclose(summed.component().mass.value, jnp.sum(summed.durations))


def test_trajectory_dataset_fixed_end_is_row_specific():
    inputs = jnp.arange(3.0).reshape((3, 1))
    domain = TrajectoryDatasetDomain(inputs, jnp.asarray([2, 4, 3]), dt=0.25)
    component = domain.component({"t": FixedEnd()})
    structure = SampleLayout((("data", "t"),))

    batch = component.sample(phx.domain.PointSampling(8, layout=structure), key=jr.key(2))
    case_indices = jnp.asarray(batch[TRAJECTORY_CASE_INDEX_KEY].data, dtype=jnp.int32)
    assert jnp.allclose(batch["t"].data, domain.end_times[case_indices])


def test_trajectory_dataset_rejects_coord_separable_sampling():
    inputs = jnp.arange(3.0).reshape((3, 1))
    domain = TrajectoryDatasetDomain(inputs, jnp.asarray([2, 4, 3]), dt=0.25)

    with pytest.raises(ValueError, match="paired data-time"):
        domain.component().sample(phx.domain.GridSampling({"t": UniformAxisSpec(8)}))


def test_trajectory_dataset_equivalence_includes_ragged_lengths():
    inputs = jnp.arange(3.0).reshape((3, 1))
    first = TrajectoryDatasetDomain(inputs, jnp.asarray([2, 4, 3]), dt=0.25)
    same = TrajectoryDatasetDomain(inputs, jnp.asarray([2, 4, 3]), dt=0.25)
    different = TrajectoryDatasetDomain(inputs, jnp.asarray([4, 2, 3]), dt=0.25)

    assert first.same_support(same)
    assert not first.same_support(different)


def test_trajectory_dataset_participates_in_time_residual_constraints():
    inputs = jnp.arange(3.0).reshape((3, 1))
    domain = TrajectoryDatasetDomain(inputs, jnp.asarray([2, 4, 3]), dt=0.25)
    component = domain.component()
    structure = SampleLayout((("data", "t"),))

    @domain.Function("data", "t")
    def exact(data, t):
        return data[0] + t

    constraint = FunctionalConstraint.from_operator(component=component,
    operator=lambda u: partial_t(u, var="t") - 1.0,
    constraint_vars="u", sampling=phx.domain.PointSampling(16, layout=structure), reduction="mean",)

    loss = constraint.loss({"u": exact}, key=jr.key(3))
    assert jnp.allclose(loss, 0.0, atol=1e-12)
