#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from phydrax.constraints import DiscreteInteriorDataConstraint
from phydrax.constraints._pointset import points_batch_from_points
from phydrax.domain import Interval1d
from phydrax.nn import add_model_loss, MLP, SeparableMLP
from phydrax.nn.models.core._base import _AbstractBaseModel
from phydrax.solver import FunctionalSolver


def _domain_model_solver(model) -> FunctionalSolver:
    domain = Interval1d(0.0, 1.0)
    u = domain.Model("x")(model)
    return FunctionalSolver(functions={"u": u}, constraints=[])


def test_add_model_loss_contributes_to_solver_loss_without_constraints():
    model = MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        key=jr.key(0),
    ).add_model_loss(lambda m: 2.0, weight=3.0, label="constant")

    solver = _domain_model_solver(model)

    assert jnp.allclose(solver.loss(key=jr.key(0)), 6.0)


def test_standalone_add_model_loss_helper_matches_method_api():
    base = MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        key=jr.key(1),
    )
    model = add_model_loss(base, lambda m: 4.0, weight=0.5, label="constant")

    solver = _domain_model_solver(model)

    assert jnp.allclose(solver.loss(key=jr.key(0)), 2.0)


def test_model_with_loss_preserves_wrapped_forward_call():
    base = MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        key=jr.key(6),
    )
    model = base.add_model_loss(lambda m: 0.0)
    x = jnp.asarray([0.25])

    assert jnp.allclose(model(x, key=jr.key(0)), base(x, key=jr.key(0)))


def test_model_loss_is_optimized_by_solve():
    model = MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        key=jr.key(2),
    ).add_model_loss(
        lambda m: (jnp.linalg.norm(m.layers[0].weight) - 1.0) ** 2,
        label="unit_weight_norm",
    )
    solver = _domain_model_solver(model)
    init_loss = solver.loss(key=jr.key(0))

    trained = solver.solve(
        num_iter=80,
        optim=optax.adam(5e-2),
        seed=0,
        log_every=0,
    )
    final_loss = trained.loss(key=jr.key(0))

    assert final_loss < init_loss


class CustomLossModel(_AbstractBaseModel):
    weight: jax.Array
    in_size: Literal["scalar"]
    out_size: Literal["scalar"]

    def __init__(self, init: float = 0.0):
        self.weight = jnp.asarray(init, dtype=float)
        self.in_size = "scalar"
        self.out_size = "scalar"

    def __call__(self, x, /, *, key=None):
        del key
        return self.weight * jnp.asarray(x)

    def __loss__(self, *, key=None, iter_=None):
        del key, iter_
        return (self.weight - 2.0) ** 2


class ScaleModel(_AbstractBaseModel):
    weight: jax.Array
    in_size: Literal["scalar"]
    out_size: Literal["scalar"]

    def __init__(self, init: float = 0.0):
        self.weight = jnp.asarray(init, dtype=float)
        self.in_size = "scalar"
        self.out_size = "scalar"

    def __call__(self, x, /, *, key=None):
        del key
        return self.weight * jnp.asarray(x)


def test_custom_model_dunder_loss_contributes_to_solver_loss():
    solver = _domain_model_solver(CustomLossModel(init=0.5))

    assert jnp.allclose(solver.loss(key=jr.key(0)), 2.25)


def test_model_loss_regularizes_domain_model_forward_weights():
    domain = Interval1d(0.0, 1.0)
    model = ScaleModel(init=1.0).add_model_loss(
        lambda m: m.weight**2,
        label="l2",
    )
    u = domain.Model("x")(model)
    data = DiscreteInteriorDataConstraint(
        "u",
        domain,
        points={"x": jnp.asarray([[1.0]])},
        values=jnp.asarray([1.0]),
        label="data",
    )
    tx = optax.sgd(0.1)
    solver = FunctionalSolver(functions={"u": u}, constraints=[data])

    trained = solver.solve(
        num_iter=1,
        optim=optax.GradientTransformation(tx.init, tx.update),
        keep_best=False,
        log_every=0,
    )

    batch = points_batch_from_points(domain.component(), {"x": jnp.asarray([[1.0]])})
    pred = jnp.asarray(trained.functions["u"](batch).data).reshape(())
    assert jnp.allclose(pred, 0.8, atol=1e-6)


def test_model_loss_is_deduped_for_shared_model_aliases():
    domain = Interval1d(0.0, 1.0)
    model = MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        key=jr.key(3),
    ).add_model_loss(lambda m: 2.0, label="shared")
    u = domain.Model("x")(model)
    v = domain.Model("x")(model)

    solver = FunctionalSolver(functions={"u": u, "v": v}, constraints=[])

    assert jnp.allclose(solver.loss(key=jr.key(0)), 2.0)


def test_loss_wrapper_preserves_domain_model_metadata():
    domain = Interval1d(0.0, 1.0)
    model = SeparableMLP(
        in_size=2,
        out_size="scalar",
        latent_size=2,
        width_size=None,
        depth=None,
        hidden_sizes=(),
        key=jr.key(4),
    ).add_model_loss(lambda m: 0.0)

    u = domain.Model("x")(model)

    assert u.func.input_mode == "flat"
    assert u.func.supports_blockwise_input


def test_solver_logs_model_losses_to_text_and_tensorboard(tmp_path):
    model = MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        key=jr.key(5),
    ).add_model_loss(lambda m: 1.0, label="unit_penalty")
    solver = _domain_model_solver(model)
    log_path = tmp_path / "model_loss.log"
    log_dir = tmp_path / "tb"

    solver.solve(
        num_iter=1,
        optim=optax.adam(1e-2),
        seed=0,
        log_every=1,
        log_path=log_path,
        tensorboard_log_dir=log_dir,
        tensorboard_every=1,
    )

    text = log_path.read_text(encoding="utf-8")
    assert "[model 0] unit_penalty:" in text

    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags()["scalars"])
    assert "train/model_losses/000_unit_penalty/loss" in scalar_tags
