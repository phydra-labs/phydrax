#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Inverse problem over the parameter and observation bindings of a coupled plate.

The FE-VEM plate of ``tests._support.coupled_plate`` has an exact solution that
depends on ``x`` only, so observations at the true parameters are generated on
the host (``exact_temperature``, ``exact_left_wall_heat``) independently of both
discretizations, perturbed by seeded Gaussian noise whose standard deviation is
declared through ``IndependentStandardUncertainty``. The right-region
conductivity and the right-wall heat flux are inferred with a
``SolverObjective`` over the whitened ``MeasurementComparisonPlan`` residuals
(conductivity-left known) and ``train_components``; derivatives are checked
against host central differences, and every refusal is exercised.
"""

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
# Triangle side and brick width of the refinement level (``build_plate``).
FE_H = 1.0 / (4 * 2**LEVEL)
VEM_H = 1.0 / (3 * 2**LEVEL)
TRUTH = {"conductivity-left": 1.3, "conductivity-right": 0.7, "heat-flux": 0.5}
SENSORS = np.asarray([[0.3, 0.4, 0.0], [0.7, 0.6, 0.0], [1.4, 0.3, 0.0], [1.8, 0.7, 0.0]])
TEMPERATURE_STD = 0.01
HEAT_STD = 0.02
TRAINING_STEPS = 12
NOISE_SEED = 20260928

CONTRACT = phx.SpatialCoordinateContract(phx.units.METER)
TEMPERATURE = M.QuantitySpec(
    "test", "temperature", "temperature", phx.units.KELVIN, "temperature"
)
# The flux observation reports the conormal flux content through the left wall
# (outward normal), i.e. the heat entering the plate there.
WALL_HEAT = M.QuantitySpec(
    "test",
    "wall-heat-inflow",
    "heat-rate-per-depth",
    cp.LINE_HEAT_UNIT,
    "wall-heat-inflow",
)
POINT = M.SamplingSemantics(M.SpatialSamplingKind.POINT)
AVERAGE = M.SamplingSemantics(M.SpatialSamplingKind.SURFACE_AVERAGE)
PATH = M.SamplingSemantics(M.SpatialSamplingKind.PATH_INTEGRAL)
WALL = M.IndexSampleSupport((1,), ("wall",))
RULE = phx.discretization.FacetTraceRule(points=2)


@dataclass(frozen=True)
class _Record:
    """Host records of one measurement: its identity, noise, and exact value."""

    binding_id: str
    quantity: M.QuantitySpec
    support: M.SampleSupport
    sampling: M.SamplingSemantics
    std: float

    def exact(self, kappa_left: float, kappa_right: float, flux: float) -> np.ndarray:
        match self.binding_id:
            case "left-sensors":
                return cp.exact_temperature(SENSORS[:2], kappa_left, kappa_right, flux)
            case "right-sensors":
                return cp.exact_temperature(SENSORS[2:], kappa_left, kappa_right, flux)
            case "right-wall-mean":
                # u depends on x only: the wall mean is u(2).
                wall = np.asarray([[2.0, 0.5]])
                return cp.exact_temperature(wall, kappa_left, kappa_right, flux)
            case "left-wall-heat":
                return np.asarray([cp.exact_left_wall_heat(flux)])
            case _:
                raise ValueError(f"Unknown observation {self.binding_id!r}.")


LEFT_SENSORS = M.PointSampleSupport(SENSORS[:2], ("a", "b"), CONTRACT)
RIGHT_SENSORS = M.PointSampleSupport(SENSORS[2:], ("c", "d"), CONTRACT)
RECORDS = (
    _Record("left-sensors", TEMPERATURE, LEFT_SENSORS, POINT, TEMPERATURE_STD),
    _Record("right-sensors", TEMPERATURE, RIGHT_SENSORS, POINT, TEMPERATURE_STD),
    _Record("right-wall-mean", TEMPERATURE, WALL, AVERAGE, TEMPERATURE_STD),
    _Record("left-wall-heat", WALL_HEAT, WALL, PATH, HEAT_STD),
)


def _observations(plate: cp.CoupledPlate) -> tuple[cpl.AbstractObservationBinding, ...]:
    left, right, wall, heat = RECORDS
    return (
        cpl.FieldPointObservation(
            left.binding_id,
            "triangles",
            "u",
            quantity=left.quantity,
            support=LEFT_SENSORS,
            sampling=left.sampling,
            field_unit=phx.units.KELVIN,
        ),
        cpl.FieldPointObservation(
            right.binding_id,
            "polygons",
            "u",
            quantity=right.quantity,
            support=RIGHT_SENSORS,
            sampling=right.sampling,
            field_unit=phx.units.KELVIN,
        ),
        cpl.FieldBoundaryObservation(
            wall.binding_id,
            "polygons",
            "u",
            plate.right_wall,
            statistic="average",
            rule=RULE,
            quantity=wall.quantity,
            support=wall.support,
            sampling=wall.sampling,
            field_unit=phx.units.KELVIN,
        ),
        cpl.FieldFluxObservation(
            heat.binding_id,
            "triangles",
            "u",
            plate.left_wall,
            rule=RULE,
            quantity=heat.quantity,
            support=heat.support,
            sampling=heat.sampling,
            reaction_unit=cp.LINE_HEAT_UNIT,
        ),
    )


def _data(record: _Record, values: np.ndarray) -> M.PreparedQuantityField:
    """Observed data prepared from the same records as the prediction."""
    return M.QuantityField(
        f"{record.binding_id}-data",
        record.quantity,
        M.ValueLayout.scalar(),
        record.support,
        record.sampling,
        values,
        uncertainty=M.IndependentStandardUncertainty(
            np.full(values.shape, record.std), record.quantity.unit
        ),
    ).prepare()


def _noisy_values() -> tuple[np.ndarray, ...]:
    """Host samples: the analytic plate at the truth plus seeded declared noise."""
    rng = np.random.default_rng(NOISE_SEED)
    truth = tuple(TRUTH.values())
    return tuple(
        record.exact(*truth)
        + record.std * rng.standard_normal(record.exact(*truth).shape)
        for record in RECORDS
    )


def _noisy_data() -> tuple[M.PreparedQuantityField, ...]:
    return tuple(
        _data(record, values)
        for record, values in zip(RECORDS, _noisy_values(), strict=True)
    )


@dataclass(frozen=True)
class _Plate:
    plate: cp.CoupledPlate
    prepared: cpl.PreparedCoupledProblem
    plans: tuple[phx.observation.MeasurementComparisonPlan, ...]


@pytest.fixture(scope="module")
def plate() -> _Plate:
    geometry = cp.build_plate(LEVEL)
    plan = cpl.CoupledProblemPlan(
        "plate",
        components=(geometry.triangles, geometry.polygons),
        bindings=(geometry.binding,),
        laws=(geometry.law,),
        parameters=cp.parameter_bindings(),
        observations=_observations(geometry),
    )
    prepared = cpl.prepare_coupled_problem(
        plan,
        interface_owners=(geometry.cover,),
        parameters={name: jnp.asarray(value) for name, value in TRUTH.items()},
    )
    plans = tuple(
        phx.observation.MeasurementComparisonPlan(data) for data in _noisy_data()
    )
    return _Plate(geometry, prepared, plans)


def _parameters(theta: Array) -> dict[str, Array]:
    """``theta = (kappa_left, kappa_right, g)`` as bound parameter values."""
    return {
        "conductivity-left": theta[0],
        "conductivity-right": theta[1],
        "heat-flux": theta[2],
    }


def _truth_parameters() -> dict[str, Array]:
    return {name: jnp.asarray(value, jnp.float64) for name, value in TRUTH.items()}


@pytest.fixture(scope="module")
def truth(plate: _Plate) -> cpl.CoupledSolution:
    """The accepted dense-LU solution at the true parameters."""
    return jax.jit(
        lambda theta: cpl.solve_coupled_problem(
            plate.prepared, parameters=_parameters(theta), policy=cp.dense_policy()
        )
    )(jnp.asarray(tuple(TRUTH.values())))


def _gmres_policy(max_steps: int, derivative_steps: int) -> phx.linalg.LinearSolvePolicy:
    """Unpreconditioned GMRES whose implicit derivative solves are Krylov solves too.

    The derivative solves run the callable restarted GMRES of the implicit route
    (restart 30) with their own step budget: at level 0 the mortar saddle-point
    system needs about 2000 steps of it to meet the derivative tolerance.
    """
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.GMRES(restart=60),
        tolerance=phx.linalg.TolerancePolicy(
            relative=1e-12, absolute=1e-14, max_steps=max_steps
        ),
        derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(
            relative_tolerance=1e-10,
            absolute_tolerance=1e-14,
            maximum_steps=derivative_steps,
        ),
    )


# --- Learned-component parameterization ----------------------------------------------


class _PlateParameters(phx.AbstractArrayModel):
    """Unknowns of the plate: ``log kappa_right`` (positivity) and the heat flux ``g``.

    A constant map: its value ``(kappa_right, g)`` does not depend on the input.
    """

    log_conductivity: Array
    heat_flux: Array
    in_size: Literal["scalar"] = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, conductivity: float, heat_flux: float) -> None:
        self.log_conductivity = jnp.log(jnp.asarray(conductivity, jnp.float64))
        self.heat_flux = jnp.asarray(heat_flux, jnp.float64)
        self.in_size = "scalar"
        self.out_size = 2

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del x, key
        return jnp.stack((jnp.exp(self.log_conductivity), self.heat_flux))

    def model_execution_contract(self) -> phx.ModelExecutionContract:
        return phx.ModelExecutionContract(
            derivative=phx.DerivativeContract.smooth(
                (phx.DerivativeSurface.INPUT, phx.DerivativeSurface.MODEL_PARAMETER)
            ),
            execution=phx.ExecutionCapabilities("native-jax"),
            precision=phx.ComponentPrecisionContract.native("float64"),
            randomness=phx.RandomnessContract("deterministic"),
        )


class _PlateSolve(eqx.Module):
    """The fixed prepared solve of the objective: problem, data, and known inputs.

    ``observation_ids`` names the prediction each comparison plan scores; the
    objective's callables read every array through this module.
    """

    prepared: cpl.PreparedCoupledProblem
    plans: tuple[phx.observation.MeasurementComparisonPlan, ...]
    kappa_left: Array
    policy: phx.linalg.LinearSolvePolicy
    observation_ids: tuple[str, ...] = eqx.field(static=True)


class _BoundPlate(eqx.Module):
    solve: _PlateSolve
    theta: Array


def _bind(solve: _PlateSolve, component: phx.ComponentBinding) -> _BoundPlate:
    unknowns = component.model(jnp.zeros((), jnp.float64))
    return _BoundPlate(solve, jnp.concatenate((solve.kappa_left[None], unknowns)))


def _declared(term: Array | None) -> Array:
    """A comparison term the declared independent uncertainty always provides."""
    if term is None:
        raise AssertionError("Independent standard uncertainty yields every term.")
    return term


def _whitened(
    solve: _PlateSolve, theta: Array
) -> tuple[Array, Array, tuple[phx.observation.MeasurementComparisonResult, ...]]:
    """Concatenated whitened residuals, acceptance, and per-observation comparisons."""
    solution = cpl.solve_coupled_problem(
        solve.prepared, parameters=_parameters(theta), policy=solve.policy
    )
    comparisons = tuple(
        comparison_plan.evaluate(solution.observation(binding_id))
        for comparison_plan, binding_id in zip(
            solve.plans, solve.observation_ids, strict=True
        )
    )
    residual = jnp.concatenate(
        [jnp.ravel(_declared(comparison.whitened_residual)) for comparison in comparisons]
    )
    accepted = solution.accepted & jnp.all(
        jnp.stack([comparison.successful for comparison in comparisons])
    )
    return residual, accepted, comparisons


def _measure(owner: _BoundPlate, case: None) -> phx.solver.SolverCaseResult:
    del case
    residual, accepted, _ = _whitened(owner.solve, owner.theta)
    return phx.solver.SolverCaseResult(residual=residual, accepted=accepted)


def _solve(
    plate: _Plate, policy: phx.linalg.LinearSolvePolicy | None = None
) -> _PlateSolve:
    return _PlateSolve(
        plate.prepared,
        plate.plans,
        jnp.asarray(TRUTH["conductivity-left"], jnp.float64),
        cp.dense_policy() if policy is None else policy,
        tuple(record.binding_id for record in RECORDS),
    )


def _objective(solve: _PlateSolve) -> phx.solver.SolverObjective:
    return phx.solver.SolverObjective(
        solve, _bind, _measure, objective_id="plate-inverse"
    )


def _component(conductivity: float, heat_flux: float) -> phx.ComponentBinding:
    return phx.bind_component(
        _PlateParameters(conductivity, heat_flux), phx.ComponentAuthority.MODEL
    )


# --- Forward observations -------------------------------------------------------------


def _curvature(record: _Record) -> tuple[float, float]:
    """Mesh size and ``max |u''| = s / kappa`` of the region a record observes."""
    if record.binding_id == "left-sensors":
        return FE_H, cp.SOURCE / TRUTH["conductivity-left"]
    return VEM_H, cp.SOURCE / TRUTH["conductivity-right"]


@pytest.mark.parametrize(
    ("record", "approximation"),
    [
        pytest.param(RECORDS[0], "exact", id="fe-point-values"),
        pytest.param(RECORDS[1], "h1-projection", id="vem-point-projection"),
        pytest.param(RECORDS[2], "exact", id="vem-wall-average"),
        pytest.param(RECORDS[3], "variational-reaction", id="fe-wall-heat-content"),
    ],
)
def test_observations_at_the_truth_match_the_analytic_plate(
    plate: _Plate, truth: cpl.CoupledSolution, record: _Record, approximation: str
) -> None:
    assert bool(truth.accepted)
    predicted = truth.observation(record.binding_id)
    assert plate.prepared.observation(record.binding_id).approximation == approximation
    assert bool(jnp.all(predicted.valid_mask))
    exact = record.exact(*TRUTH.values())
    if record.sampling is POINT:
        # The exact u is quadratic in x: its piecewise-linear interpolation error
        # is at most h^2 |u''| / 8; the Galerkin (and mortar) error at the nodes is
        # of the same O(h^2) order, so twice the interpolation bound is admitted.
        h, curvature = _curvature(record)
        tolerance = h**2 * curvature / 4.0
        np.testing.assert_allclose(predicted.values, exact, rtol=0.0, atol=tolerance)
    else:
        # Both functionals are exact to roundoff. The wall average is a(u, z) for
        # the dual solution z (kappa dz/dn = 1 on the wall, z = 0 on x = 0), which
        # is linear in x on each region and so lies in the P1, degree-1 VEM, and
        # mortar spaces: Galerkin orthogonality leaves no error. The wall heat is
        # the discrete balance of the total source and the prescribed flux, which
        # both spaces conserve exactly because they contain the constants.
        np.testing.assert_allclose(predicted.values, exact, rtol=1e-10, atol=0.0)


# --- Derivatives of the whitened misfit ------------------------------------------------


OFF_TRUTH = (1.25, 0.8, 0.6)
FD_STEP = 1e-6


def _central_jacobian(residual: Any, theta: np.ndarray) -> np.ndarray:
    """Host central differences of the whitened residual in every parameter."""
    columns = []
    for axis in range(theta.size):
        step = np.zeros_like(theta)
        step[axis] = FD_STEP
        plus = np.asarray(residual(jnp.asarray(theta + step)))
        minus = np.asarray(residual(jnp.asarray(theta - step)))
        columns.append((plus - minus) / (2.0 * FD_STEP))
    return np.stack(columns, axis=1)


# Central differences with step 1e-6 resolve the O(10-100) whitened
# derivatives to ~1e-8 relative (observed 3e-9 dense, 6e-9 GMRES), so 1e-6 leaves
# margin. Transposition is exact to roundoff when the tangent and adjoint reuse
# one LU factorization, and to the 1e-10 derivative-solve tolerance with GMRES.
@pytest.mark.parametrize(
    ("policy", "transpose_rtol"),
    [
        pytest.param(cp.dense_policy(), 1e-12, id="dense-lu-factor-reuse"),
        pytest.param(_gmres_policy(400, 2000), 1e-8, id="gmres-krylov-derivative"),
    ],
)
def test_misfit_derivatives_match_host_central_differences(
    plate: _Plate, policy: phx.linalg.LinearSolvePolicy, transpose_rtol: float
) -> None:
    solve = _solve(plate, policy)

    def residual(theta: Array) -> Array:
        return _whitened(solve, theta)[0]

    def misfit(theta: Array) -> Array:
        return 0.5 * jnp.sum(residual(theta) ** 2)

    theta = np.asarray(OFF_TRUTH)
    direction = jnp.asarray([0.3, -0.2, 0.5])
    cotangent = jnp.asarray(np.random.default_rng(3).standard_normal(len(SENSORS) + 2))
    value, tangent = jax.jit(lambda point: jax.jvp(residual, (point,), (direction,)))(
        jnp.asarray(theta)
    )
    adjoint = jax.jit(lambda point, weight: jax.vjp(residual, point)[1](weight)[0])(
        jnp.asarray(theta), cotangent
    )
    gradient = jax.jit(jax.grad(misfit))(jnp.asarray(theta))
    jacobian = _central_jacobian(jax.jit(residual), theta)

    expected_tangent = jacobian @ np.asarray(direction)
    expected_adjoint = jacobian.T @ np.asarray(cotangent)
    expected_gradient = jacobian.T @ np.asarray(value)
    for label, derivative, expected in (
        ("tangent", tangent, expected_tangent),
        ("adjoint", adjoint, expected_adjoint),
        ("gradient", gradient, expected_gradient),
    ):
        scale = np.max(np.abs(expected))
        np.testing.assert_allclose(
            derivative, expected, rtol=0.0, atol=1e-6 * scale, err_msg=label
        )
    # The implicit tangent and adjoint are transposes of one derivative map.
    np.testing.assert_allclose(
        float(jnp.vdot(cotangent, tangent)),
        float(jnp.vdot(adjoint, direction)),
        rtol=transpose_rtol,
    )


# --- Inference -------------------------------------------------------------------------


def test_training_recovers_conductivity_and_heat_flux(
    plate: _Plate, truth: cpl.CoupledSolution
) -> None:
    solve = _solve(plate)
    objective = _objective(solve)
    result = phx.solver.train_components(
        _component(1.0, 1.0),
        (objective,),
        optimizer=optax.lbfgs(),
        steps=TRAINING_STEPS,
        key=jr.key(0),
    )
    assert result.accepted_updates > 0
    assert result.nonfinite_rejections == 0
    values = np.asarray(result.values)
    assert values[-1] < 1e-3 * values[0]
    model = result.tree.model
    estimate = np.asarray(
        [float(jnp.exp(model.log_conductivity)), float(model.heat_flux)]
    )
    expected = np.asarray([TRUTH["conductivity-right"], TRUTH["heat-flux"]])

    # Fixed-noise prewhitened posterior over the flat parameter lane
    # (log kappa_right, g) of the trained component.
    position = jnp.stack((model.log_conductivity, model.heat_flux))
    space = phx.uq.ParameterSpace(position, priors=phx.uq.Normal(0.0, 10.0))
    posterior = phx.uq.posterior_problem_from_solver_objective(
        objective, result.tree, space
    )
    estimated = jnp.asarray([TRUTH["conductivity-left"], *estimate])
    solution = jax.jit(
        lambda theta: cpl.solve_coupled_problem(
            plate.prepared, parameters=_parameters(theta), policy=cp.dense_policy()
        )
    )(estimated)
    assert bool(solution.accepted)
    comparisons = tuple(
        comparison_plan.evaluate(solution.observation(record.binding_id))
        for comparison_plan, record in zip(plate.plans, RECORDS, strict=True)
    )
    assert all(bool(comparison.successful) for comparison in comparisons)
    # Host Gaussian terms of the declared independent noise: the standardized
    # misfit of the predictions against the host samples, and log det of the
    # diagonal covariance 2 sum log(sigma).
    standardized = np.concatenate(
        [
            (np.asarray(solution.observation(record.binding_id).values) - values)
            / record.std
            for record, values in zip(RECORDS, _noisy_values(), strict=True)
        ]
    )
    quadratic = float(standardized @ standardized)
    logdet = sum(
        2.0 * values.size * np.log(record.std)
        for record, values in zip(RECORDS, _noisy_values(), strict=True)
    )
    log_likelihood = float(
        jax.jit(lambda point: posterior.log_likelihood(point))(position)
    )
    np.testing.assert_allclose(log_likelihood, -0.5 * quadratic, rtol=1e-10)
    # The normalized comparison likelihood adds the fixed-noise constant.
    np.testing.assert_allclose(
        sum(float(_declared(comparison.log_likelihood)) for comparison in comparisons),
        -0.5 * (quadratic + logdet + standardized.size * np.log(2.0 * np.pi)),
        rtol=1e-10,
    )

    # Tolerance from the linearized estimator: the noise spreads the estimate
    # with the Gauss-Newton covariance C = (J^T J)^-1 of the whitened residual,
    # and the O(h^2) discretization bias b of the predictions at the truth
    # (against the host analytic plate) shifts it by C J^T b.
    jacobian = np.asarray(jax.jit(jax.jacfwd(posterior.gauss_newton_residual))(position))
    physical = jacobian * np.asarray([1.0 / estimate[0], 1.0])  # d/d(kappa, g)
    normal = physical.T @ physical
    covariance = np.linalg.solve(normal, np.eye(2))
    standard_deviation = np.sqrt(np.diag(covariance))
    assert np.all(standard_deviation < 0.05 * np.abs(expected))
    bias = np.concatenate(
        [
            (
                np.asarray(truth.observation(record.binding_id).values)
                - record.exact(*TRUTH.values())
            )
            / record.std
            for record in RECORDS
        ]
    )
    shift = np.abs(covariance @ physical.T @ bias)
    np.testing.assert_array_less(
        np.abs(estimate - expected), 4.0 * standard_deviation + shift
    )


# --- Refusals -------------------------------------------------------------------------


def _rejected_training(objective: phx.solver.SolverObjective) -> None:
    """One training attempt from the truth commits no update of the component."""
    parameters = _PlateParameters(TRUTH["conductivity-right"], TRUTH["heat-flux"])
    result = phx.solver.train_components(
        phx.bind_component(parameters, phx.ComponentAuthority.MODEL),
        (objective,),
        optimizer=optax.adam(1e-2),
        steps=1,
        key=jr.key(0),
        rejection_budget=1,
    )
    assert result.accepted_updates == 0
    assert result.nonfinite_rejections == 1
    assert float(result.tree.model.log_conductivity) == float(parameters.log_conductivity)


def test_failed_primal_is_rejected_and_has_no_derivative(plate: _Plate) -> None:
    # Five GMRES steps cannot reach the relative tolerance on 52 unknowns.
    solve = _solve(plate, _gmres_policy(5, 2000))
    theta = jnp.asarray(tuple(TRUTH.values()))
    solution = jax.jit(
        lambda point: cpl.solve_coupled_problem(
            plate.prepared, parameters=_parameters(point), policy=solve.policy
        )
    )(theta)
    assert not bool(solution.native_successful)
    assert not bool(solution.accepted)
    assert not bool(solution.derivative_valid)
    gradient = jax.jit(jax.grad(lambda point: jnp.sum(_whitened(solve, point)[0] ** 2)))(
        theta
    )
    assert bool(jnp.all(jnp.isnan(gradient)))

    objective = _objective(solve)
    evaluation = eqx.filter_jit(objective.evaluate)(
        _component(TRUTH["conductivity-right"], TRUTH["heat-flux"])
    )
    assert not bool(jnp.any(evaluation.accepted))
    assert bool(jnp.isnan(evaluation.value))
    _rejected_training(objective)


def test_failed_derivative_solve_poisons_only_the_derivative(plate: _Plate) -> None:
    # The primal GMRES converges; one derivative GMRES step cannot.
    solve = _solve(plate, _gmres_policy(400, 1))
    theta = jnp.asarray(tuple(TRUTH.values()))
    residual, accepted, _ = jax.jit(lambda point: _whitened(solve, point))(theta)
    reference, _, _ = jax.jit(lambda point: _whitened(_solve(plate), point))(theta)
    assert bool(accepted)
    np.testing.assert_allclose(residual, reference, rtol=1e-8, atol=1e-8)
    gradient = jax.jit(jax.grad(lambda point: jnp.sum(_whitened(solve, point)[0] ** 2)))(
        theta
    )
    assert bool(jnp.all(jnp.isnan(gradient)))
    _rejected_training(_objective(solve))


@pytest.mark.parametrize("name", ["conductivity-left", "conductivity-right"])
def test_rhs_only_policy_refuses_conductivity_derivatives(
    plate: _Plate, name: str
) -> None:
    policy = cp.dense_policy("rhs-only")
    capability = plate.prepared.derivative_capability(policy)
    assert capability.admitted == (("heat-flux", phx.DerivativeSurface.SOLVER_ARGUMENT),)
    assert name in dict(capability.refused)

    def wall_mean(value: Array) -> Array:
        solution = cpl.solve_coupled_problem(
            plate.prepared,
            parameters={**_truth_parameters(), name: value},
            policy=policy,
        )
        return solution.observation("right-wall-mean").values[0]

    with pytest.raises(ValueError, match="derivative-unsupported"):
        jax.grad(wall_mean)(jnp.asarray(TRUTH[name], jnp.float64))


def test_rhs_only_heat_flux_derivative_is_the_analytic_sensitivity(plate: _Plate) -> None:
    policy = cp.dense_policy("rhs-only")

    def observed(flux: Array) -> Array:
        solution = cpl.solve_coupled_problem(
            plate.prepared,
            parameters={**_truth_parameters(), "heat-flux": flux},
            policy=policy,
        )
        return jnp.concatenate(
            [solution.observation(record.binding_id).values for record in RECORDS]
        )

    flux = jnp.asarray(TRUTH["heat-flux"], jnp.float64)
    forward = jax.jit(jax.jacfwd(observed))(flux)
    reverse = jax.jit(jax.grad(lambda value: jnp.sum(observed(value))))(flux)
    # Every observation is affine in g, and its g-sensitivity is the plate with no
    # source and unit wall flux: a piecewise-linear-in-x field that the P1, VEM,
    # and mortar spaces (and the VEM H1 projection) reproduce exactly.
    kappa_left, kappa_right = TRUTH["conductivity-left"], TRUTH["conductivity-right"]
    exact = np.concatenate(
        [
            record.exact(kappa_left, kappa_right, 1.0)
            - record.exact(kappa_left, kappa_right, 0.0)
            for record in RECORDS
        ]
    )
    np.testing.assert_allclose(forward, exact, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(reverse, np.sum(exact), rtol=1e-9)


MILLIKELVIN = phx.units.UnitDefinition(
    "mK", phx.units.KELVIN.dimension, phx.units.KELVIN.reference_system_id, "0.001"
)


@pytest.mark.parametrize(
    ("record", "data", "role"),
    [
        pytest.param(
            RECORDS[2],
            _Record(
                "right-wall-mean",
                M.QuantitySpec(
                    "test",
                    "surface-temperature",
                    "temperature",
                    phx.units.KELVIN,
                    "surface-temperature",
                ),
                WALL,
                AVERAGE,
                TEMPERATURE_STD,
            ),
            "quantity",
            id="other-quantity",
        ),
        pytest.param(
            RECORDS[2],
            _Record(
                "right-wall-mean",
                M.QuantitySpec(
                    "test", "temperature", "temperature", MILLIKELVIN, "temperature"
                ),
                WALL,
                AVERAGE,
                1e3 * TEMPERATURE_STD,
            ),
            "unit",
            id="other-unit",
        ),
        pytest.param(
            RECORDS[2],
            _Record("right-wall-mean", TEMPERATURE, WALL, POINT, TEMPERATURE_STD),
            "sampling",
            id="point-value-for-an-average",
        ),
        pytest.param(
            RECORDS[1],
            _Record(
                "right-sensors",
                TEMPERATURE,
                M.PointSampleSupport(SENSORS[2:], ("e", "f"), CONTRACT),
                POINT,
                TEMPERATURE_STD,
            ),
            "support",
            id="other-sensors",
        ),
    ],
)
def test_incompatible_data_is_refused_by_the_comparison(
    truth: cpl.CoupledSolution, record: _Record, data: _Record, role: str
) -> None:
    predicted = truth.observation(record.binding_id)
    observed = _data(data, record.exact(*TRUTH.values()))
    plan = phx.observation.MeasurementComparisonPlan(observed)
    with pytest.raises(ValueError, match=f"{role} identities differ"):
        plan.evaluate(predicted)


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("average-sampled-as-point", "distinct measurements"),
        ("heat-sampled-as-average", "distinct measurements"),
        ("heat-declared-as-temperature", "never identified"),
    ],
)
def test_misdeclared_observation_is_refused_at_construction(
    plate: _Plate, kind: str, message: str
) -> None:
    geometry = plate.plate
    with pytest.raises(ValueError, match=message):
        match kind:
            case "average-sampled-as-point":
                cpl.FieldBoundaryObservation(
                    "right-wall-mean",
                    "polygons",
                    "u",
                    geometry.right_wall,
                    statistic="average",
                    rule=RULE,
                    quantity=TEMPERATURE,
                    support=WALL,
                    sampling=POINT,
                    field_unit=phx.units.KELVIN,
                )
            case "heat-sampled-as-average" | "heat-declared-as-temperature":
                cpl.FieldFluxObservation(
                    "left-wall-heat",
                    "triangles",
                    "u",
                    geometry.left_wall,
                    rule=RULE,
                    quantity=WALL_HEAT
                    if kind == "heat-sampled-as-average"
                    else TEMPERATURE,
                    support=WALL,
                    sampling=AVERAGE if kind == "heat-sampled-as-average" else PATH,
                    reaction_unit=cp.LINE_HEAT_UNIT,
                )
            case _:
                raise ValueError(f"Unknown case {kind!r}.")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unbound-parameter", "not bound by the plan"),
        ("missing-refresh-parameter", "must be bound at every solve"),
        ("parameter-through-arguments", "supply them through parameters"),
    ],
)
def test_parameter_binding_errors_are_refused_at_solve(
    plate: _Plate, case: str, message: str
) -> None:
    parameters = _truth_parameters()
    arguments: dict[str, object] | None = None
    match case:
        case "unbound-parameter":
            parameters["conductivity-top"] = jnp.asarray(1.0, jnp.float64)
        case "missing-refresh-parameter":
            del parameters["heat-flux"]
        case "parameter-through-arguments":
            arguments = {"polygons": {"heat-flux": jnp.asarray(0.5, jnp.float64)}}
        case _:
            raise ValueError(f"Unknown case {case!r}.")
    with pytest.raises(ValueError, match=message):
        cpl.solve_coupled_problem(
            plate.prepared,
            arguments=arguments,
            parameters=parameters,
            policy=cp.dense_policy(),
        )
