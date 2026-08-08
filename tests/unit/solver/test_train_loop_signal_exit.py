#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import signal

import jax.numpy as jnp
import jax.random as jr
import optax

import phydrax as phx
import phydrax.solver._functional_train as functional_train
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
    with functional_train._TrainingSignalGuard() as guard:
        signal.raise_signal(signal.SIGINT)
        assert guard.stop_requested
        assert guard.signal_name == "SIGINT"

    assert signal.getsignal(signal.SIGINT) == previous


def test_optax_solve_returns_after_signal_stop_request(monkeypatch, tmp_path):
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
    monkeypatch.setattr(functional_train, "_TrainingSignalGuard", lambda: guard)

    log_path = tmp_path / "train.log"
    trained = _make_solver().solve(
        num_iter=5,
        optim=optax.adam(1e-2),
        seed=0,
        log_every=1,
        log_path=log_path,
    )

    assert isinstance(trained, FunctionalSolver)
    assert "received SIGTERM" in log_path.read_text(encoding="utf-8")
    assert "after 1/5 iteration(s)" in log_path.read_text(encoding="utf-8")


def test_optax_solve_returns_after_keyboard_interrupt_from_step(tmp_path):
    def init(_params):
        return ()

    def update(_grads, state, _params=None):
        raise KeyboardInterrupt

    optim = optax.GradientTransformation(init, update)
    log_path = tmp_path / "keyboard_interrupt.log"

    trained = _make_solver().solve(
        num_iter=5,
        optim=optim,
        seed=0,
        jit=False,
        log_every=1,
        log_path=log_path,
    )

    text = log_path.read_text(encoding="utf-8")
    assert isinstance(trained, FunctionalSolver)
    assert "received SIGINT" in text
    assert "after 0/5 iteration(s)" in text
