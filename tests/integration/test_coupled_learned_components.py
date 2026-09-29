#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Learned physical and accelerator components of one coupled FE-VEM plate.

Two learned roles train through two different permitted objectives:

- A learned conductivity field of the finite-element region changes the
  accepted equations. It is a ``MODEL`` (or ``DISCRETIZATION``) authority
  ``ComponentBinding`` bound as the value of the ``conductivity-left``
  ``ParameterBinding`` and trains through the accepted coupled solve with a
  ``SolverObjective`` (implicit solution-map derivatives) and
  ``train_components``. The data are point temperatures of an independent host
  analytic plate whose left conductivity grows exponentially along ``x``.
- A learned initial guess of the same linear solve is an accelerator
  (``LearnedInitialGuess``, ``ACCELERATOR`` authority). It trains only through
  a fixed-work GMRES ``AlgorithmicWorkObjective``; a production solve takes it
  as an untrusted provider, and the accepted solution is the dense LU one
  whatever the guess.

Every authority that would change the accepted equation, or train through an
objective its authority does not admit, is refused.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest
from jax import Array

import phydrax as phx
from phydrax.solver import coupling as cpl
from tests._support import coupled_plate as cp


M = phx.measurement

LEVEL = 0
KAPPA_RIGHT = 0.7
# Log-linear left conductivity ``kappa(x) = exp(c0 + c1 (x - 1/2) + c2 (y - 1/2))``.
TRUE_FIELD = np.asarray([np.log(1.3), 0.6, 0.0])
INITIAL_FIELD = np.asarray([np.log(0.8), 0.0, 0.0])
FLUXES = np.asarray([0.0, 0.5, 1.0])
SENSORS = np.asarray(
    [
        [0.1, 0.3],
        [0.25, 0.8],
        [0.4, 0.45],
        [0.55, 0.15],
        [0.7, 0.65],
        [0.85, 0.35],
        [0.95, 0.9],
        [1.5, 0.5],
    ]
)
TEMPERATURE_SCALE = 0.01  # Explicit least-squares reference weighting.
# Fixed work of the accelerator objective and its training schedule.
WORK = 8
ACCELERATOR_STEPS = 400
ACCELERATOR_RATE = 5e-3
PRODUCTION_RESTART = 30

CONTRACT = phx.SpatialCoordinateContract(phx.units.METER)
TEMPERATURE = M.QuantitySpec(
    "test", "temperature", "temperature", phx.units.KELVIN, "temperature"
)
POINT = M.SamplingSemantics(M.SpatialSamplingKind.POINT)
POSITION = phx.ValuePort(
    "plate-position",
    event_shape=(2,),
    component_ids=("x", "y"),
    representation="cartesian",
    dimensions=(phx.units.METER.dimension,) * 2,
)


def graded_temperature(
    points: np.ndarray, coefficients: np.ndarray, flux: float
) -> np.ndarray:
    """Host reference ``u(x)`` for ``kappa(x) = exp(a + b x)`` left of the interface.

    The heat flux ``kappa u' = 2 s + g - s x`` is fixed by the source and the
    right-wall flux, so ``u(x) = exp(-a) (A I0(x) - s I1(x))`` with
    ``A = 2 s + g``, ``I0 = (1 - e^{-bx}) / b``, ``I1 = (1 - e^{-bx}(1 + bx)) / b^2``;
    the right region adds the constant-conductivity profile to ``u(1)``.
    """
    x = np.asarray(points, dtype=np.float64)[..., 0]
    slope = float(coefficients[1])
    offset = float(coefficients[0]) - 0.5 * slope
    source = cp.SOURCE
    total = 2.0 * source + flux

    def left(t: np.ndarray) -> np.ndarray:
        decay = np.exp(-slope * t)
        first = (1.0 - decay) / slope
        second = (1.0 - decay * (1.0 + slope * t)) / slope**2
        return np.exp(-offset) * (total * first - source * second)

    right = (
        left(np.asarray(1.0))
        + (-0.5 * source * (x - 1.0) ** 2 + (source + flux) * (x - 1.0)) / KAPPA_RIGHT
    )
    return np.where(x <= cp.INTERFACE_X, left(np.minimum(x, cp.INTERFACE_X)), right)


def field_values(coefficients: np.ndarray, points: np.ndarray) -> np.ndarray:
    exponent = (
        coefficients[0]
        + coefficients[1] * (points[:, 0] - 0.5)
        + coefficients[2] * (points[:, 1] - 0.5)
    )
    return np.exp(exponent)


# --- Learned conductivity (MODEL / DISCRETIZATION authority) ----------------------------


class _ConductivityField(phx.AbstractArrayModel):
    """Positive log-linear conductivity field of the plate position."""

    coefficients: Array
    in_size: int = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)

    def __init__(self, coefficients: Any) -> None:
        self.coefficients = jnp.asarray(coefficients, jnp.float64)
        self.in_size = 2
        self.out_size = "scalar"

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        c = self.coefficients
        return jnp.exp(c[0] + c[1] * (x[..., 0] - 0.5) + c[2] * (x[..., 1] - 0.5))

    def model_execution_contract(self) -> phx.ModelExecutionContract:
        return phx.ModelExecutionContract(
            derivative=phx.DerivativeContract.smooth(
                (phx.DerivativeSurface.INPUT, phx.DerivativeSurface.MODEL_PARAMETER)
            ),
            execution=phx.ExecutionCapabilities("native-jax"),
            precision=phx.ComponentPrecisionContract.native("float64"),
            randomness=phx.RandomnessContract("deterministic"),
        )


def _conductivity(
    coefficients: Any,
    authority: phx.ComponentAuthority = phx.ComponentAuthority.MODEL,
    output: phx.ValuePort | None = None,
) -> phx.ComponentBinding:
    return phx.bind_component(
        _ConductivityField(coefficients),
        authority,
        owner_ports=phx.ModelPorts(
            inputs=(POSITION,),
            outputs=(cp.conductivity_port("left") if output is None else output,),
        ),
    )


def _learned_conductivity(points: Array, context: object) -> Array:
    """FE coefficient: the bound learned model evaluated at the quadrature points."""
    if not isinstance(context, phx.equations.FiniteElementExecutionContext):
        raise TypeError("FE coefficients receive the execution context.")
    arguments = context.user_args
    if not isinstance(arguments, Mapping):
        raise TypeError("Owner user arguments must be a mapping.")
    model = arguments["conductivity"]
    if not isinstance(model, phx.AbstractArrayModel):
        raise TypeError("The conductivity parameter is bound to a learned model.")
    return model(points)


class _FieldSolve(eqx.Module):
    """Fixed prepared solve of the solution-map objective."""

    prepared: cpl.PreparedCoupledProblem
    kappa_right: Array


class _BoundField(eqx.Module):
    solve: _FieldSolve
    conductivity: phx.ComponentBinding


def _field_parameters(
    conductivity: phx.ComponentBinding, kappa_right: Array, flux: Array
) -> dict[str, object]:
    return {
        "conductivity-left": conductivity,
        "conductivity-right": kappa_right,
        "heat-flux": flux,
    }


def _field_measure(
    owner: _BoundField, case: tuple[Array, Array]
) -> phx.solver.SolverCaseResult:
    flux, observed = case
    solution = cpl.solve_coupled_problem(
        owner.solve.prepared,
        parameters=_field_parameters(owner.conductivity, owner.solve.kappa_right, flux),
        policy=cp.dense_policy(),
    )
    predicted = jnp.concatenate(
        [solution.observation(name).values for name in ("left", "right")]
    )
    return phx.solver.SolverCaseResult(
        residual=(predicted - observed) / TEMPERATURE_SCALE,
        accepted=solution.accepted,
    )


def _sensor_observations() -> tuple[cpl.FieldPointObservation, ...]:
    points = np.concatenate([SENSORS, np.zeros((len(SENSORS), 1))], axis=1)
    return (
        cpl.FieldPointObservation(
            "left",
            "triangles",
            "u",
            quantity=TEMPERATURE,
            support=M.PointSampleSupport(points[:-1], tuple("abcdefg"), CONTRACT),
            sampling=POINT,
            field_unit=phx.units.KELVIN,
        ),
        cpl.FieldPointObservation(
            "right",
            "polygons",
            "u",
            quantity=TEMPERATURE,
            support=M.PointSampleSupport(points[-1:], ("h",), CONTRACT),
            sampling=POINT,
            field_unit=phx.units.KELVIN,
        ),
    )


@dataclass(frozen=True)
class _FieldPlate:
    prepared: cpl.PreparedCoupledProblem
    objective: phx.solver.SolverObjective


@pytest.fixture(scope="module")
def field_plate() -> _FieldPlate:
    plate = cp.build_plate(LEVEL, triangle_conductivity=_learned_conductivity)
    plan = cpl.CoupledProblemPlan(
        "learned-conductivity-plate",
        components=(plate.triangles, plate.polygons),
        bindings=(plate.binding,),
        laws=(plate.law,),
        parameters=cp.parameter_bindings(),
        observations=_sensor_observations(),
    )
    prepared = cpl.prepare_coupled_problem(
        plan,
        interface_owners=(plate.cover,),
        parameters=_field_parameters(
            _conductivity(INITIAL_FIELD), jnp.asarray(KAPPA_RIGHT), jnp.asarray(0.5)
        ),
    )
    observed = np.stack([graded_temperature(SENSORS, TRUE_FIELD, g) for g in FLUXES])
    objective = phx.solver.SolverObjective(
        _FieldSolve(prepared, jnp.asarray(KAPPA_RIGHT, jnp.float64)),
        _BoundField,
        _field_measure,
        objective_id="learned-conductivity",
        cases=(jnp.asarray(FLUXES), jnp.asarray(observed)),
    )
    return _FieldPlate(prepared, objective)


def _with_coefficients(template: phx.ComponentBinding, coefficients: Array) -> Any:
    return eqx.tree_at(lambda binding: binding.model.coefficients, template, coefficients)


def _field_error(coefficients: np.ndarray) -> float:
    """Relative RMS error of a log-linear field against the analytic conductivity."""
    axis = np.linspace(0.0, 1.0, 21)
    grid = np.stack(np.meshgrid(axis, axis, indexing="xy"), axis=-1).reshape(-1, 2)
    truth = field_values(TRUE_FIELD, grid)
    error = field_values(coefficients, grid) - truth
    return float(np.sqrt(np.mean(error**2) / np.mean(truth**2)))


@pytest.mark.parametrize(
    "authority",
    [phx.ComponentAuthority.MODEL, phx.ComponentAuthority.DISCRETIZATION],
    ids=("model", "discretization"),
)
def test_learned_conductivity_gradient_is_the_implicit_solution_map_derivative(
    field_plate: _FieldPlate, authority: phx.ComponentAuthority
) -> None:
    objective = field_plate.objective
    point = jnp.asarray(INITIAL_FIELD + np.asarray([0.2, 0.3, -0.2]))
    template = _conductivity(point, authority)
    assert field_plate.prepared.derivative_capability(cp.dense_policy()).admits(
        "conductivity-left"
    )
    evaluation = objective.evaluate(template)
    assert bool(jnp.all(evaluation.accepted))
    assert [owner for _, owner in evaluation.trained] == [authority]

    def value(coefficients: Array) -> Array:
        return objective.evaluate(_with_coefficients(template, coefficients)).value

    gradient = np.asarray(jax.jit(jax.grad(value))(point))
    compiled = jax.jit(value)
    step = 1.0e-6
    central = np.asarray(
        [
            (float(compiled(point + step * unit)) - float(compiled(point - step * unit)))
            / (2.0 * step)
            for unit in jnp.eye(3)
        ]
    )
    # The implicit adjoint reuses the dense LU factors; central differences of
    # the accepted objective carry an O(step^2) truncation and roundoff of about
    # eps |value| / step.
    np.testing.assert_allclose(
        gradient, central, rtol=1e-6, atol=1e-6 * np.abs(central).max()
    )


def test_learned_conductivity_trains_through_accepted_solves_toward_the_analytic_field(
    field_plate: _FieldPlate,
) -> None:
    objective = field_plate.objective
    steps = 15
    result = phx.solver.train_components(
        _conductivity(INITIAL_FIELD),
        (objective,),
        optimizer=optax.lbfgs(),
        steps=steps,
        key=jr.key(0),
    )
    assert result.accepted_updates == steps
    assert result.authorities == ((".model.coefficients", "model"),)
    values = np.asarray(result.values)
    assert values[-1] < 1e-3 * values[0]
    trained = np.asarray(result.tree.model.coefficients)
    evaluation = objective.evaluate(result.tree)
    assert bool(jnp.all(evaluation.accepted))

    # Against the independent analytic conductivity the field error drops by
    # more than an order of magnitude.
    assert _field_error(trained) < 0.1 * _field_error(INITIAL_FIELD)
    # What remains is the O(h^2) P1 bias of the predictions: on this mesh the
    # trained field fits the analytic data better than the true field does, so
    # no optimizer shortfall is left to account for the error.
    at_truth = objective.evaluate(_conductivity(TRUE_FIELD))
    assert float(evaluation.value) < float(at_truth.value)


# --- Learned preconditioner (ACCELERATOR authority) ------------------------------------


class _InverseAction(phx.AbstractArrayModel):
    """Learned approximate-inverse action ``(I + sum_j phi_j W_j) r``.

    The input is the feature vector ``phi = (1, 1 / kappa_left, 1 / kappa_right)``
    followed by a flattened residual; zero weights are the identity, that is,
    unpreconditioned FGMRES.
    """

    weights: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, size: int) -> None:
        self.weights = jnp.zeros((3, size, size), jnp.float64)
        self.in_size = 3 + size
        self.out_size = size

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        features, residual = x[:3], x[3:]
        correction = jnp.sum(features[:, None, None] * self.weights, axis=0)
        return residual + correction @ residual

    def model_execution_contract(self) -> phx.ModelExecutionContract:
        return phx.ModelExecutionContract(
            derivative=phx.DerivativeContract.smooth(
                (phx.DerivativeSurface.INPUT, phx.DerivativeSurface.MODEL_PARAMETER)
            ),
            execution=phx.ExecutionCapabilities("native-jax"),
            precision=phx.ComponentPrecisionContract.native("float64"),
            randomness=phx.RandomnessContract("deterministic"),
        )


class _LearnedInverse(phx.linalg.AbstractPreconditioner):
    """Right preconditioner of the coupled solve: the bound learned inverse action."""

    action: phx.ComponentBinding
    features: Array = phx.fixed_field()

    def __init__(
        self, action: phx.ComponentBinding, theta: Array, space: phx.linalg.BlockSpace
    ) -> None:
        self.action = action
        self.features = jnp.stack(
            (jnp.ones_like(theta[0]), 1.0 / theta[0], 1.0 / theta[1])
        )
        self.space = space
        self.properties = phx.linalg.PreconditionerProperties(
            linear=True,
            stationary=True,
            evidence={"linear": "construction", "stationary": "construction"},
        )
        self.preconditioner_id = "learned-approximate-inverse"

    def apply(self, residual: Any, /, *, iteration: Any = None) -> Any:
        del iteration
        flat = self.space.flatten(residual)
        return self.space.unflatten(
            self.action.model(jnp.concatenate((self.features, flat)))
        )


def _plate_parameters(theta: Array) -> dict[str, Array]:
    return {
        "conductivity-left": theta[0],
        "conductivity-right": theta[1],
        "heat-flux": theta[2],
    }


def _krylov_policy(
    preconditioner: phx.linalg.AbstractPreconditioner | None,
    *,
    restart: int,
    relative: float,
    max_steps: int,
    mode: phx.linalg.DifferentiationMode,
) -> phx.linalg.LinearSolvePolicy:
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.FGMRES(restart=restart),
        preconditioning=None
        if preconditioner is None
        else phx.linalg.PreconditioningPolicy(preconditioner, side="right"),
        tolerance=phx.linalg.TolerancePolicy(
            relative=relative, absolute=0.0, max_steps=max_steps
        ),
        differentiation=phx.linalg.DifferentiationPolicy(mode),
        failure=phx.linalg.FailurePolicy("status"),
    )


class _KrylovPlate(eqx.Module):
    """Fixed prepared solve of the fixed-work objective."""

    prepared: cpl.PreparedCoupledProblem


class _BoundKrylov(eqx.Module):
    solve: _KrylovPlate
    action: phx.ComponentBinding


def _fixed_work_measure(
    owner: _BoundKrylov, theta: Array
) -> phx.solver.AlgorithmicWorkResult:
    """Exactly ``WORK`` preconditioned FGMRES steps from the native zero state."""
    prepared = owner.solve.prepared
    parameters = _plate_parameters(theta)
    preconditioner = _LearnedInverse(owner.action, theta, prepared.state_space)
    solution = cpl.solve_coupled_problem(
        prepared,
        parameters=parameters,
        policy=_krylov_policy(
            preconditioner, restart=WORK, relative=0.0, max_steps=WORK, mode="algorithmic"
        ),
    )
    linear = solution.linear
    if linear is None:
        raise AssertionError("An affine coupled solve reports its linear result.")
    arguments = prepared.bind_arguments(parameters=parameters).arguments
    return phx.solver.AlgorithmicWorkResult(
        initial_residual=prepared.residual(prepared.state_space.zeros(), arguments),
        final_residual=prepared.residual(solution.state, arguments),
        iterations=linear.diagnostics.iterations,
        accepted=jnp.all(jnp.isfinite(prepared.state_space.flatten(solution.state))),
    )


@dataclass(frozen=True)
class _KrylovCase:
    prepared: cpl.PreparedCoupledProblem
    solve: _KrylovPlate
    objective: phx.solver.AlgorithmicWorkObjective
    initial: phx.ComponentBinding
    result: phx.solver.ComponentTrainingResult


@pytest.fixture(scope="module")
def krylov_plate() -> _KrylovCase:
    plate = cp.build_plate(LEVEL)
    plan = cpl.CoupledProblemPlan(
        "accelerated-plate",
        components=(plate.triangles, plate.polygons),
        bindings=(plate.binding,),
        laws=(plate.law,),
        parameters=cp.parameter_bindings(),
    )
    prepared = cpl.prepare_coupled_problem(
        plan,
        interface_owners=(plate.cover,),
        parameters=_plate_parameters(jnp.asarray([1.0, 1.0, 0.5])),
    )
    rng = np.random.default_rng(0)
    cases = np.stack(
        (rng.uniform(0.5, 2.0, 8), rng.uniform(0.5, 2.0, 8), rng.uniform(0.0, 1.0, 8)),
        axis=1,
    )
    solve = _KrylovPlate(prepared)
    objective = phx.solver.AlgorithmicWorkObjective(
        solve,
        _BoundKrylov,
        _fixed_work_measure,
        work=WORK,
        objective_id="learned-inverse-fixed-work",
        cases=jnp.asarray(cases),
    )
    initial = phx.bind_component(
        _InverseAction(prepared.state_space.size), _LearnedInverse
    )
    result = phx.solver.train_components(
        initial,
        (objective,),
        optimizer=optax.adam(ACCELERATOR_RATE),
        steps=ACCELERATOR_STEPS,
        key=jr.key(0),
    )
    return _KrylovCase(prepared, solve, objective, initial, result)


def test_learned_preconditioner_trains_only_through_fixed_krylov_work(
    krylov_plate: _KrylovCase,
) -> None:
    objective, initial, result = (
        krylov_plate.objective,
        krylov_plate.initial,
        krylov_plate.result,
    )
    solution_map = phx.solver.SolverObjective(
        krylov_plate.solve, _BoundKrylov, _never_measured, objective_id="solution-map"
    )
    with pytest.raises(ValueError, match=r"no admissible training signal.*accelerator"):
        solution_map.evaluate(initial)

    assert result.authorities == ((".model.weights", "accelerator"),)
    assert result.accepted_updates == ACCELERATOR_STEPS
    before = objective.evaluate(initial)
    after = objective.evaluate(result.tree)
    for evaluation in (before, after):
        assert evaluation.work is not None
        assert evaluation.work.tolist() == [WORK] * 8
        assert bool(jnp.all(evaluation.accepted))
    # Eight unpreconditioned FGMRES steps barely reduce the saddle-point
    # residual (log(||r_8|| / ||r_0||) is about -0.4); after training the same
    # fixed work reduces it by more than a further one and a half decades.
    assert float(after.value) < float(before.value) - 1.5 * np.log(10.0)


def _never_measured(owner: Any, case: Any) -> phx.solver.SolverCaseResult:
    raise AssertionError("A refused objective never binds or measures.")


def _flat(
    prepared: cpl.PreparedCoupledProblem, solution: cpl.CoupledSolution
) -> np.ndarray:
    return np.asarray(prepared.state_space.flatten(solution.state))


def _iterations(solution: cpl.CoupledSolution) -> int:
    linear = solution.linear
    if linear is None:
        raise AssertionError("An affine coupled solve reports its linear result.")
    return int(linear.diagnostics.iterations)


@pytest.mark.parametrize(
    "theta",
    [(1.3, 0.7, 0.5), (0.6, 1.8, 0.9)],
    ids=("held-out-a", "held-out-b"),
)
def test_learned_preconditioner_accelerates_without_changing_the_accepted_solution(
    krylov_plate: _KrylovCase, theta: tuple[float, float, float]
) -> None:
    prepared = krylov_plate.prepared
    values = jnp.asarray(theta)
    parameters = _plate_parameters(values)
    reference = cpl.solve_coupled_problem(
        prepared, parameters=parameters, policy=cp.dense_policy()
    )
    solves = {
        name: cpl.solve_coupled_problem(
            prepared,
            parameters=parameters,
            policy=_krylov_policy(
                preconditioner,
                restart=PRODUCTION_RESTART,
                relative=1e-12,
                max_steps=4000,
                mode="none",
            ),
        )
        for name, preconditioner in (
            ("plain", None),
            (
                "learned",
                _LearnedInverse(krylov_plate.result.tree, values, prepared.state_space),
            ),
        )
    }
    expected = _flat(prepared, reference)
    assert bool(reference.accepted)
    for solution in solves.values():
        assert bool(solution.accepted)
        np.testing.assert_allclose(
            _flat(prepared, solution),
            expected,
            rtol=0.0,
            atol=1e-9 * np.abs(expected).max(),
        )
    assert 3 * _iterations(solves["learned"]) < _iterations(solves["plain"])


def test_a_singular_learned_preconditioner_cannot_produce_an_accepted_solution(
    krylov_plate: _KrylovCase,
) -> None:
    prepared = krylov_plate.prepared
    values = jnp.asarray([1.3, 0.7, 0.5])
    size = prepared.state_space.size
    # W_0 = -I makes the learned action identically zero.
    singular = eqx.tree_at(
        lambda binding: binding.model.weights,
        krylov_plate.initial,
        jnp.zeros((3, size, size)).at[0].set(-jnp.eye(size)),
    )
    solution = cpl.solve_coupled_problem(
        prepared,
        parameters=_plate_parameters(values),
        policy=_krylov_policy(
            _LearnedInverse(singular, values, prepared.state_space),
            restart=PRODUCTION_RESTART,
            relative=1e-12,
            max_steps=200,
            mode="none",
        ),
    )
    assert not bool(solution.native_successful)
    assert not bool(solution.accepted)


# --- Untrusted initial-guess proposals ----------------------------------------------------


class _StoredState(phx.AbstractArrayModel):
    """An accelerator's stored proposal: one flat solve-coordinate state."""

    state: Array
    in_size: Literal["scalar"] = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, state: Any) -> None:
        self.state = jnp.asarray(state, jnp.float64)
        self.in_size = "scalar"
        self.out_size = self.state.shape[0]

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del x, key
        return self.state

    def model_execution_contract(self) -> phx.ModelExecutionContract:
        return phx.ModelExecutionContract(
            derivative=phx.DerivativeContract.smooth(
                (phx.DerivativeSurface.INPUT, phx.DerivativeSurface.MODEL_PARAMETER)
            ),
            execution=phx.ExecutionCapabilities("native-jax"),
            precision=phx.ComponentPrecisionContract.native("float64"),
            randomness=phx.RandomnessContract("deterministic"),
        )


class _StoredProposal(eqx.Module):
    """Proposal of a ``LearnedInitialGuess``: the stored state in solve coordinates."""

    guess: phx.ComponentBinding
    space: phx.linalg.BlockSpace

    def __call__(self, data: Any, baseline: Any) -> Any:
        del data, baseline
        return self.space.unflatten(self.guess.model(jnp.zeros((), jnp.float64)))


@pytest.mark.parametrize("proposal", ["neighbor-state", "invalid-state"])
def test_initial_guess_proposals_stay_distinct_from_the_accepted_state(
    krylov_plate: _KrylovCase, proposal: str
) -> None:
    prepared = krylov_plate.prepared
    space = prepared.state_space
    parameters = _plate_parameters(jnp.asarray([1.3, 0.7, 0.5]))
    reference = cpl.solve_coupled_problem(
        prepared, parameters=parameters, policy=cp.dense_policy()
    )
    # The accepted state of a neighboring parameter is a warm start; a huge
    # constant state has a larger residual than the native zero state.
    neighbor = cpl.solve_coupled_problem(
        prepared,
        parameters=_plate_parameters(jnp.asarray([1.25, 0.75, 0.5])),
        policy=cp.dense_policy(),
    )
    stored = (
        _flat(prepared, neighbor)
        if proposal == "neighbor-state"
        else np.full((space.size,), 1.0e3)
    )
    guess = phx.bind_component(_StoredState(stored), phx.linalg.LearnedInitialGuess)
    provider = phx.linalg.LearnedInitialGuess(
        _StoredProposal(guess, space), provider_id="stored-state"
    )
    policy = _krylov_policy(
        None, restart=PRODUCTION_RESTART, relative=1e-12, max_steps=4000, mode="none"
    )
    native = cpl.solve_coupled_problem(prepared, parameters=parameters, policy=policy)
    guided = cpl.solve_coupled_problem(
        prepared, parameters=parameters, policy=policy, initial_state=provider
    )

    expected = _flat(prepared, reference)
    assert bool(guided.accepted)
    np.testing.assert_allclose(
        _flat(prepared, guided), expected, rtol=0.0, atol=1e-9 * np.abs(expected).max()
    )
    # The proposal itself is certified against the original equations and is
    # not an accepted solution.
    certificate = cpl.certify_coupled_state(
        prepared,
        space.unflatten(jnp.asarray(stored)),
        prepared.bind_arguments(parameters=parameters),
        policy=policy,
        linear=None,
        nonlinear=None,
        native=jnp.asarray(True),
        derivative=jnp.asarray(False),
        tolerance=guided.tolerance,
    )
    assert not bool(certificate.accepted)
    evidence = guided.initial_guess
    assert evidence is not None and native.initial_guess is None
    assert evidence.provider_id == "stored-state"
    if proposal == "neighbor-state":
        assert bool(evidence.accepted)
        assert float(evidence.proposal_residual_norm) < float(
            evidence.baseline_residual_norm
        )
        assert _iterations(guided) < _iterations(native)
    else:
        # The native zero state is kept: the same solve, iteration for iteration.
        assert not bool(evidence.accepted)
        assert _iterations(guided) == _iterations(native)
        np.testing.assert_array_equal(_flat(prepared, guided), _flat(prepared, native))


# --- Refusals ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("authority", "output", "message"),
    [
        (
            phx.ComponentAuthority.ACCELERATOR,
            None,
            r"component with accelerator authority cannot supply parameter "
            r"'conductivity-left'",
        ),
        (
            phx.ComponentAuthority.SURROGATE,
            None,
            r"component with surrogate authority cannot supply parameter "
            r"'conductivity-left'",
        ),
        (
            phx.ComponentAuthority.DECISION,
            None,
            r"component with decision authority cannot supply parameter "
            r"'conductivity-left'",
        ),
        (
            phx.ComponentAuthority.MODEL,
            cp.conductivity_port("right"),
            r"does not publish its port among its owner output ports",
        ),
    ],
    ids=("accelerator", "surrogate", "decision", "unpublished-port"),
)
def test_parameter_binding_refuses_learned_inputs_that_would_change_the_equation(
    field_plate: _FieldPlate,
    authority: phx.ComponentAuthority,
    output: phx.ValuePort | None,
    message: str,
) -> None:
    learned = _conductivity(TRUE_FIELD, authority, output)
    with pytest.raises(ValueError, match=message):
        field_plate.prepared.bind_arguments(
            parameters=_field_parameters(
                learned, jnp.asarray(KAPPA_RIGHT), jnp.asarray(0.5)
            )
        )


@pytest.mark.parametrize(
    "authority",
    [phx.ComponentAuthority.SURROGATE, phx.ComponentAuthority.ACCELERATOR],
    ids=("surrogate", "accelerator"),
)
def test_solution_map_objective_refuses_authorities_it_does_not_admit(
    field_plate: _FieldPlate, authority: phx.ComponentAuthority
) -> None:
    # A surrogate is not the implicit solution map, and an accelerator changes
    # work, not the solution: neither reaches bind or measure.
    with pytest.raises(
        ValueError, match=rf"no admissible training signal for \['{authority.value}'\]"
    ):
        field_plate.objective.evaluate(_conductivity(INITIAL_FIELD, authority))


def test_each_objective_trains_only_the_authorities_it_admits(
    field_plate: _FieldPlate, krylov_plate: _KrylovCase
) -> None:
    tree = {
        "conductivity": _conductivity(INITIAL_FIELD),
        "preconditioner": krylov_plate.initial,
    }
    for objective, trained, stopped in (
        (field_plate.objective, "model", "accelerator"),
        (krylov_plate.objective, "accelerator", "model"),
    ):
        admission = objective.admit(tree)
        assert {authority for _, authority in admission.trained} == {trained}
        assert {authority for _, authority in admission.stopped} == {stopped}
