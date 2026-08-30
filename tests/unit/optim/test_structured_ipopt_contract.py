import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.optim._ipopt import (
    _canonical_hessian_structure,
    _canonical_structure,
    _mapped_status,
    _StructuredIpoptCallbacks,
)


def _structured_program(*, exact_hessian=False):
    objective = lambda value, args: 0.5 * jnp.vdot(value, value)
    constraints = lambda value, args: jnp.asarray(
        (value[0] + value[1], value[0] - value[1])
    )
    source = phx.linalg.ArraySpace((2,), dtype=jnp.float64)
    target = phx.linalg.ArraySpace((2,), dtype=jnp.float64)
    point = jnp.asarray((0.75, 0.25))
    jacobian = phx.sparse.compile_sparse_jacobian(
        constraints,
        point,
        source=source,
        target=target,
        compiler="auto",
        plan_id="ipopt-contract-jacobian",
    )
    hessian = None
    if exact_hessian:
        lagrangian = lambda value, packed: (
            packed[1] * objective(value, packed[0])
            + jnp.vdot(packed[2], constraints(value, packed[0]))
        )
        hessian = phx.sparse.compile_sparse_hessian(
            lagrangian,
            point,
            space=source,
            sample_args=(None, jnp.asarray(1.0), jnp.zeros(2)),
            compiler="auto",
            plan_id="ipopt-contract-hessian",
        )
    program = phx.optim.StructuredNonlinearProgram(
        objective,
        constraints,
        jacobian,
        variable_lower=jnp.zeros(2),
        variable_upper=jnp.full(2, jnp.inf),
        constraint_lower=jnp.asarray((1.0, 0.0)),
        constraint_upper=jnp.asarray((1.0, jnp.inf)),
        constraint_sources=("sum", "difference"),
        hessian_plan=hessian,
        program_id="ipopt-contract-program",
        structure_id=f"ipopt-contract:{exact_hessian}",
    )
    return program, point


def test_ipopt_status_mapping_is_explicit():
    assert _mapped_status(0) == phx.optim.OptimizationStatus.SUCCESS
    assert _mapped_status(1) == phx.optim.OptimizationStatus.SUCCESS
    assert _mapped_status(2) == phx.optim.OptimizationStatus.INFEASIBLE
    assert _mapped_status(3) == phx.optim.OptimizationStatus.STAGNATION
    assert _mapped_status(4) == phx.optim.OptimizationStatus.DIVERGENCE
    assert _mapped_status(5) == phx.optim.OptimizationStatus.BACKEND_FAILED
    assert _mapped_status(6) == phx.optim.OptimizationStatus.CERTIFICATION_FAILED
    assert _mapped_status(-1) == phx.optim.OptimizationStatus.MAXIMUM_STEPS_REACHED
    assert _mapped_status(-10) == phx.optim.OptimizationStatus.CONSTRAINT_QUALIFICATION_FAILED
    assert _mapped_status(-13) == phx.optim.OptimizationStatus.NONFINITE_EVALUATION
    assert _mapped_status(-199) == phx.optim.OptimizationStatus.BACKEND_FAILED


def test_ipopt_sparse_structures_are_canonical_and_duplicate_free():
    rows, cols, positions = _canonical_structure(
        jnp.asarray((1, 0, 1)),
        jnp.asarray((1, 1, 0)),
        shape=(2, 2),
        owner="test Jacobian",
    )
    assert rows.tolist() == [0, 1, 1]
    assert cols.tolist() == [1, 0, 1]
    assert positions.tolist() == [1, 2, 0]
    with pytest.raises(ValueError, match="duplicate"):
        _canonical_structure(
            jnp.asarray((0, 0)),
            jnp.asarray((1, 1)),
            shape=(2, 2),
            owner="duplicate Jacobian",
        )

    rows, cols, positions = _canonical_hessian_structure(
        jnp.asarray((0, 0, 1, 1)),
        jnp.asarray((0, 1, 0, 1)),
        size=2,
    )
    assert rows.tolist() == [0, 1, 1]
    assert cols.tolist() == [0, 0, 1]
    assert positions.tolist() == [0, 2, 3]


def test_structured_callbacks_count_exact_sparse_work():
    program, point = _structured_program(exact_hessian=True)
    callbacks = _StructuredIpoptCallbacks(program, None)
    assert callbacks.objective(point) == pytest.approx(0.3125)
    assert jnp.allclose(callbacks.gradient(point), point)
    assert jnp.allclose(callbacks.constraints(point), jnp.asarray((1.0, 0.5)))
    assert callbacks.jacobian(point).shape == (program.jacobian_plan.nnz,)
    hessian = callbacks.hessian(point, jnp.zeros(2), 1.0)
    assert jnp.allclose(hessian, jnp.ones(2))
    assert callbacks.intermediate(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    counts = callbacks.counts.freeze()
    assert counts.objective == 1
    assert counts.gradient == 1
    assert counts.constraints == 1
    assert counts.jacobian == 1
    assert counts.hessian == 1
    assert counts.intermediate == 1
    assert counts.host_to_device == 7
    assert counts.device_to_host == 5


def test_structured_ipopt_options_cannot_override_owned_semantics():
    program, _ = _structured_program()
    termination = phx.optim.OptimizationTermination()
    method = phx.optim.IpoptMinimize(options={"print_level": 0})
    options = method._structured_options(program, termination)
    assert options["max_iter"] == termination.maximum_steps
    assert options["tol"] == termination.absolute_optimality
    assert options["hessian_approximation"] == "limited-memory"

    with pytest.raises(ValueError, match="owned by Phydrax"):
        phx.optim.IpoptMinimize(options={"tol": 1.0e-3})._structured_options(
            program,
            termination,
        )
    exact_program, _ = _structured_program(exact_hessian=True)
    with pytest.raises(ValueError, match="exact structured Hessian"):
        phx.optim.IpoptMinimize(
            options={"hessian_approximation": "limited-memory"}
        )._structured_options(exact_program, termination)


def test_structured_warm_start_tracks_source_and_rejects_invalid_duals():
    program, point = _structured_program()
    warm = program.warm_start(
        point,
        jnp.asarray((1.0, -0.5)),
        jnp.asarray((0.0, 0.25)),
        jnp.asarray((0.0, 0.0)),
        source_result_id="source-result",
        source_backend="ipopt",
    )
    assert warm.structure_id == program.structure_id
    assert warm.source_program_id == program.program_id
    assert warm.source_result_id == "source-result"
    assert warm.source_backend == "ipopt"
    assert warm.warm_start_id

    with pytest.raises(ValueError, match="non-negative"):
        phx.optim.StructuredNonlinearWarmStart(
            point,
            jnp.zeros(2),
            jnp.asarray((-1.0, 0.0)),
            jnp.zeros(2),
            structure_id=program.structure_id,
        )
