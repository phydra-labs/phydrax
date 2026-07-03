#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr

from phydrax.constraints import RaggedTimeSeriesDataConstraint
from phydrax.domain import (
    Interval1d,
    IrregularTrajectoryDatasetDomain,
    ProductStructure,
    TimeInterval,
)
from phydrax.nn.models import LatentContractionModel
from phydrax.nn.models.core._base import _AbstractBaseModel
from phydrax.nn.models.wrappers._axis_contraction import (
    AxisContractionPlan,
    AxisFactor,
    AxisGather,
    AxisProductTerm,
    contract_axis_factors,
)


def _as_scalar(x):
    arr = jnp.asarray(x)
    if arr.ndim == 0:
        return arr
    if arr.size != 1:
        raise ValueError("Expected scalar input.")
    return arr.reshape(())


class AffineScalarLatent(_AbstractBaseModel):
    in_size: int | str
    out_size: int | str
    offset: jax.Array

    def __init__(self, offset: float = 0.0) -> None:
        self.in_size = "scalar"
        self.out_size = 2
        self.offset = jnp.asarray(offset)

    def __call__(self, x, /, *, key=jr.key(0)):
        del key
        x = _as_scalar(x)
        return jnp.stack([x + self.offset, jnp.array(1.0)], axis=-1)


class DataPlusOneLatent(_AbstractBaseModel):
    in_size: int | str
    out_size: int | str

    def __init__(self) -> None:
        self.in_size = "scalar"
        self.out_size = 2

    def __call__(self, x, /, *, key=jr.key(0)):
        del key
        data = _as_scalar(x)
        return jnp.stack([data, jnp.array(1.0)], axis=-1)


class OnePlusTimeLatent(_AbstractBaseModel):
    in_size: int | str
    out_size: int | str

    def __init__(self) -> None:
        self.in_size = "scalar"
        self.out_size = 2

    def __call__(self, x, /, *, key=jr.key(0)):
        del key
        t = _as_scalar(x)
        return jnp.stack([jnp.array(1.0), t], axis=-1)


def test_axis_contraction_plan_supports_multi_term_and_gather():
    case = AxisFactor(
        "case",
        jnp.asarray([[[1.0], [2.0]], [[3.0], [4.0]]]),
        ("case",),
    )
    obs_case = AxisFactor(
        "obs_case",
        jnp.asarray([[[10.0], [20.0]], [[30.0], [40.0]]]),
        ("case",),
        gathers=(AxisGather("case", ("obs",), jnp.asarray([0, 1, 0])),),
    )
    time = AxisFactor(
        "time",
        jnp.asarray([[[5.0], [7.0]], [[11.0], [13.0]], [[17.0], [19.0]]]),
        ("obs",),
    )

    plan = AxisContractionPlan(
        (
            AxisProductTerm(("obs_case", "time")),
            AxisProductTerm(("time",), coefficient=2.0),
        )
    )
    out = contract_axis_factors({"case": case, "obs_case": obs_case, "time": time}, plan)

    gathered = jnp.asarray([[10.0, 20.0], [30.0, 40.0], [10.0, 20.0]])
    trunk = jnp.asarray([[5.0, 7.0], [11.0, 13.0], [17.0, 19.0]])
    expected = jnp.sum(gathered * trunk, axis=-1) + 2.0 * jnp.sum(trunk, axis=-1)

    assert out.axes == ("obs",)
    assert jnp.allclose(out.data[:, 0], expected)


def test_latent_contraction_axis_batch_matches_product_grid_and_grad():
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    model = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        factors={"x": AffineScalarLatent(), "t": AffineScalarLatent()},
        factor_inputs={"x": ("x",), "t": ("t",)},
    )
    u = domain.Model("x", "t", input_mode="structured")(model)
    batch = domain.component().sample(
        (4, 5),
        structure=ProductStructure((("x",), ("t",))),
        key=jr.key(1),
    )

    out = u(batch)
    x = jnp.asarray(batch["x"].data)[:, 0]
    t = jnp.asarray(batch["t"].data)
    expected = x[:, None] * t[None, :] + 1.0

    assert out.dims[:2] == (
        batch.structure.axis_for("x"),
        batch.structure.axis_for("t"),
    )
    assert jnp.allclose(out.data, expected)

    def total(scale):
        scaled = LatentContractionModel(
            latent_size=2,
            out_size="scalar",
            factors={"x": AffineScalarLatent(scale), "t": AffineScalarLatent()},
            factor_inputs={"x": ("x",), "t": ("t",)},
        )
        fn = domain.Model("x", "t", input_mode="structured")(scaled)
        return jnp.sum(fn(batch).data)

    grad = jax.grad(total)(jnp.asarray(0.25))
    assert jnp.isfinite(grad)


def test_irregular_ragged_constraint_uses_case_major_axis_batch():
    inputs = jnp.asarray([[0.0], [1.0], [2.0]])
    times = jnp.asarray(
        [
            [0.0, 0.2, 0.7, 0.0],
            [0.1, 0.4, 1.2, 1.8],
            [-0.2, 0.3, 0.0, 0.0],
        ]
    )
    lengths = jnp.asarray([3, 4, 2])
    domain = IrregularTrajectoryDatasetDomain(inputs, times, lengths)
    values = inputs[:, 0, None] + times

    model = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        factors={"data": DataPlusOneLatent(), "t": OnePlusTimeLatent()},
        factor_inputs={"data": ("data",), "t": ("t",)},
    )
    u = domain.Model("data", "t", input_mode="structured")(model)
    constraint = RaggedTimeSeriesDataConstraint(
        "u",
        domain.component(),
        values,
        num_points=(3, 4),
        structure=ProductStructure((("data",), ("t",))),
        sampling="case_time_uniform",
        interpolation="linear",
    )

    batch = constraint.sample(key=jr.key(2))
    pred = u(batch.points)
    loss = constraint.loss({"u": u}, batch=batch)

    assert pred.dims[:2] == (
        batch.points.structure.axis_for("data"),
        batch.points.structure.axis_for("t"),
    )
    assert pred.data.shape == batch.target.shape
    assert jnp.allclose(loss, 0.0, atol=1e-12)
