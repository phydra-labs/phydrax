#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_geometry_supports_distinct_structures_and_weighted_pairing():
    state_space = phx.linalg.PyTreeSpace(
        {"x": jnp.zeros((2,), dtype=jnp.float64)},
        pairing=phx.linalg.DiagonalPairing(
            {"x": jnp.asarray([2.0, 3.0], dtype=jnp.float64)}
        ),
        space_id="weighted-continuation-state",
    )
    residual_space = phx.linalg.PyTreeSpace(
        {"f": jnp.zeros((2,), dtype=jnp.float64)},
        space_id="continuation-residual",
    )
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, coordinate, _: {
            "f": state["x"] - jnp.asarray([coordinate, 2.0 * coordinate])
        },
        state_space=state_space,
        residual_space=residual_space,
        problem_id="different-continuation-spaces",
    )
    result = phx.continuation.continue_branch(
        problem,
        {"x": jnp.zeros((2,), dtype=jnp.float64)},
        jnp.asarray(0.0, dtype=jnp.float64),
        num_steps=2,
        method=phx.continuation.NaturalParameterContinuation(initial_step=0.1),
    )

    assert result.status == phx.continuation.ContinuationStatus.SUCCESS
    np.testing.assert_allclose(result.points[-1].state["x"], [0.25, 0.5])
    assert result.branch.geometry.public_state_space.space_id == state_space.space_id
    assert (
        result.branch.geometry.public_residual_space.space_id == residual_space.space_id
    )
    np.testing.assert_allclose(
        result.branch.geometry.state_norm({"x": jnp.ones((2,))}),
        np.sqrt(5.0),
    )


def test_native_complex_nonholomorphic_branch_uses_real_execution_coordinates():
    public_space = phx.linalg.ArraySpace((1,), dtype=jnp.complex128)
    coordinates = phx.linalg.ComplexCartesianCoordinates(public_space)
    representation = phx.continuation.ContinuationRepresentationPolicy(
        state_coordinates=coordinates,
        residual_coordinates=coordinates,
    )
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, coordinate, _: (
            state + 0.2 * jnp.conj(state) - (1.0 + 1.0j) * coordinate
        ),
        representation=representation,
        problem_id="nonholomorphic-complex-branch",
    )
    result = phx.continuation.continue_branch(
        problem,
        jnp.asarray([0.0 + 0.0j], dtype=jnp.complex128),
        jnp.asarray(0.0, dtype=jnp.float64),
        num_steps=4,
        method=phx.continuation.NaturalParameterContinuation(initial_step=0.2),
        terminal_coordinate=0.5,
    )

    assert result.status == phx.continuation.ContinuationStatus.SUCCESS
    assert jnp.iscomplexobj(result.points[-1].state)
    np.testing.assert_allclose(
        result.points[-1].state,
        jnp.asarray([5.0 / 12.0 + 5.0j / 8.0]),
        atol=1e-10,
    )
    assert result.branch.geometry.execution_state_space.shape == (2, 1)
    assert np.issubdtype(
        result.branch.geometry.execution_state_space.dtype,
        np.floating,
    )
    assert result.provenance.representation_id == representation.policy_id


def test_finite_algebra_public_axis_round_trips_through_continuation():
    coordinates = phx.linalg.AlgebraCoordinatePlan(
        phx.metrix.algebra.QuaternionAlgebraSpec(),
        public_storage="real_coordinates",
        public_dtype=jnp.float64,
    ).prepare((1,))
    representation = phx.continuation.ContinuationRepresentationPolicy(
        state_coordinates=coordinates,
        residual_coordinates=coordinates,
    )
    initial = jnp.zeros(coordinates.public_shape, dtype=jnp.float64)
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, coordinate, _: state - coordinate * jnp.ones_like(state),
        representation=representation,
        problem_id="quaternion-coordinate-branch",
    )
    result = phx.continuation.continue_branch(
        problem,
        initial,
        jnp.asarray(0.0, dtype=jnp.float64),
        num_steps=2,
        method=phx.continuation.NaturalParameterContinuation(initial_step=0.1),
        terminal_coordinate=0.2,
    )

    assert result.status == phx.continuation.ContinuationStatus.SUCCESS
    assert result.points[-1].state.shape == coordinates.public_shape
    np.testing.assert_allclose(result.points[-1].state, 0.2)
    assert (
        result.branch.geometry.representation.state_coordinates.coordinate_id
        == coordinates.coordinate_id
    )


def test_complex_execution_requires_an_explicit_real_coordinate_map():
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, coordinate, _: state - coordinate,
    )

    with pytest.raises(TypeError, match="real floating-point execution"):
        phx.continuation.prepare_continuation(
            problem,
            jnp.asarray([0.0 + 0.0j]),
            jnp.asarray(0.0),
            phx.continuation.plan_continuation(problem, num_steps=1),
        )


def test_constrained_spectral_coordinates_preserve_the_declared_subspace():
    spectral = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(8),),
        axis_names=("x",),
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))
    coordinates = phx.discretization.HermitianSpectralCoordinates(spectral)
    reference = spectral.project(jnp.sin(2.0 * jnp.pi * spectral.axes[0].nodes))
    representation = phx.continuation.ContinuationRepresentationPolicy(
        state_coordinates=coordinates,
        residual_coordinates=coordinates,
        defect_tolerance=1e-9,
    )
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, coordinate, _: state - (1.0 + coordinate) * reference,
        representation=representation,
        problem_id="constrained-spectral-branch",
    )
    result = phx.continuation.continue_branch(
        problem,
        reference,
        jnp.asarray(0.0),
        num_steps=1,
        method=phx.continuation.NaturalParameterContinuation(initial_step=0.1),
    )

    assert result.status == phx.continuation.ContinuationStatus.SUCCESS
    assert float(coordinates.defect(result.points[-1].state)) <= 1e-9
    assert (
        result.branch.geometry.representation.state_coordinates.evidence.domain_kind
        == "constrained_subspace"
    )


def test_refresh_rejects_changed_representation_geometry():
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, coordinate, _: state - coordinate,
        problem_id="geometry-refresh",
    )
    plan = phx.continuation.plan_continuation(problem, num_steps=1)
    prepared = phx.continuation.prepare_continuation(
        problem,
        jnp.asarray([0.0]),
        jnp.asarray(0.0),
        plan,
    )

    with pytest.raises(ValueError, match="geometry"):
        phx.continuation.refresh_continuation(
            prepared,
            jnp.zeros((2,)),
            jnp.asarray(0.0),
        )


def test_stability_uses_realified_complex_execution_operator():
    public_space = phx.linalg.ArraySpace((1,), dtype=jnp.complex128)
    coordinates = phx.linalg.ComplexCartesianCoordinates(public_space)
    representation = phx.continuation.ContinuationRepresentationPolicy(
        state_coordinates=coordinates,
        residual_coordinates=coordinates,
    )
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, coordinate, _: (coordinate + 1.0j) * state,
        representation=representation,
        problem_id="complex-stability",
    )
    evidence = phx.continuation.DenseSchurStabilityAnalyzer().analyze(
        problem,
        jnp.asarray([0.0 + 0.0j]),
        jnp.asarray(0.3),
    )

    assert bool(evidence.successful)
    values = np.asarray(evidence.eigenvalues)
    np.testing.assert_allclose(np.real(values), 0.3, atol=1e-10)
    np.testing.assert_allclose(np.sort(np.imag(values)), [-1.0, 1.0], atol=1e-10)


def test_stability_rejects_nonendomorphic_execution_spaces():
    state_space = phx.linalg.PyTreeSpace(
        {"x": jnp.zeros((1,), dtype=jnp.float64)},
        space_id="nonendomorphic-state",
    )
    residual_space = phx.linalg.PyTreeSpace(
        {"f": jnp.zeros((1,), dtype=jnp.float64)},
        space_id="nonendomorphic-residual",
    )
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, coordinate, _: {"f": state["x"] - coordinate},
        state_space=state_space,
        residual_space=residual_space,
    )
    evidence = phx.continuation.DenseSchurStabilityAnalyzer().analyze(
        problem,
        {"x": jnp.zeros((1,), dtype=jnp.float64)},
        jnp.asarray(0.0),
    )

    assert int(evidence.status) == int(
        phx.continuation.StabilityAnalysisStatus.CAPABILITY_REJECTED
    )
    assert not bool(evidence.successful)
