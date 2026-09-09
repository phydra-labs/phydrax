#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import signal
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import optax

import phydrax as phx
import phydrax.solver._functional_gradient as functional_gradient
from phydrax.domain import HyperRectangle
from phydrax.nn.models import MLP
from phydrax.solver import FunctionalSolver


def _make_solver(seed: int = 0) -> FunctionalSolver:
    domain = HyperRectangle(jnp.asarray([0.0]), jnp.asarray([1.0]), label="x")
    points = jnp.linspace(0.0, 1.0, 5).reshape((-1, 1))

    @domain.Function("x")
    def target(x):
        return 1.0 + 2.0 * x[0]

    model = MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        key=jr.key(seed),
    )
    u = domain.Model("x")(model)
    component = domain.component()
    batch = component.points({"x": points})
    condition = phx.conditions.Observation("u", component, target)
    source = phx.integration.fixed(
        phx.integration.from_samples(phx.integration.mean_over(component), batch)
    )
    data = phx.terms.ObservationPenalty(condition, source, label="data")
    return FunctionalSolver(functions={"u": u}, terms=[data])


def test_training_signal_guard_records_sigint_and_restores_handler():
    previous = signal.getsignal(signal.SIGINT)
    with functional_gradient._TrainingSignalGuard() as guard:
        signal.raise_signal(signal.SIGINT)
        assert guard.stop_requested
        assert guard.signal_name == "SIGINT"

    assert signal.getsignal(signal.SIGINT) == previous


def test_optax_solve_returns_after_signal_stop_request(
    monkeypatch, phydrax_events
):
    class StopAfterFirstStep:
        signal_name = "SIGTERM"
        def __init__(self):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        @property
        def stop_requested(self):
            self.calls += 1
            return self.calls >= 2

    guard = StopAfterFirstStep()
    monkeypatch.setattr(functional_gradient, "_TrainingSignalGuard", lambda: guard)

    trained = _make_solver().solve(
        num_iter=5,
        optim=optax.adam(1e-2),
        seed=0,
        log_every=1,
    )

    assert isinstance(trained, FunctionalSolver)
    event = phydrax_events.records("training.stopped")[-1]
    assert event["fields"]["signal_name"] == "SIGTERM"
    assert event["fields"]["completed_steps"] == 1
    assert event["fields"]["total_steps"] == 5


def test_optax_solve_returns_after_keyboard_interrupt_from_step(
    monkeypatch, phydrax_events
):
    def init(_params):
        return ()

    def update(_grads, state, _params=None):
        raise KeyboardInterrupt

    init_fn: Any = init
    update_fn: Any = update

    optim = optax.GradientTransformation(init_fn, update_fn)

    trained = _make_solver().solve(
        num_iter=5,
        optim=optim,
        seed=0,
        jit=False,
        log_every=1,
    )

    assert isinstance(trained, FunctionalSolver)
    event = phydrax_events.records("training.stopped")[-1]
    assert event["fields"]["signal_name"] == "SIGINT"
    assert event["fields"]["completed_steps"] == 0
    assert event["fields"]["total_steps"] == 5
