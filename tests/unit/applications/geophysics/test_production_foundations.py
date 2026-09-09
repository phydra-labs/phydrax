#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications import geophysics as geo
from phydrax.interchange import (
    BoreholeTrajectory,
    bounded_resource_from_bytes,
    convert_time,
    GeospatialContract,
    LeapSecondTable,
    PlanetaryCoordinateContract,
    ReferenceBodyContract,
    ResourceLimits,
    TimeReferenceContract,
)
from phydrax.observation import (
    CirculantCovarianceAction,
    CoordinateLayout,
    DiagonalCovarianceAction,
    KroneckerCholeskyCovarianceAction,
    LinearNuisancePlan,
    LowRankDiagonalCovarianceAction,
    PrecisionOperatorCovarianceAction,
)
from phydrax.units import DEGREE, METER
from phydrax.uq import (
    CensoredGaussianLikelihood,
    ContaminatedGaussianLikelihood,
    CrossGradientPrior,
    GraphMetricPrior,
    SPDEPrecisionPrior,
    StochasticExperimentDesignPlan,
    TemporalDifferencePrior,
)


def _tetra_mesh():
    return phx.discretization.CellMesh.from_tetrahedra(
        np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=float),
        np.asarray(((0, 1, 2, 3),)),
    )


def test_time_borehole_and_planetary_coordinate_contracts_are_explicit():
    resource = bounded_resource_from_bytes(
        b"pinned leap table", limits=ResourceLimits(1024, 2, 4, 4, 0)
    ).manifest
    leaps = LeapSecondTable([100.0], [11.0], 10.0, resource)
    utc = TimeReferenceContract(
        "utc",
        "nominal-origin",
        10.0,
        epoch_nominal_seconds=0.0,
        leap_seconds=leaps,
    )
    tai = TimeReferenceContract("tai", "same-origin", 0.0)
    converted, transform = convert_time(jnp.asarray((99.0, 100.0)), utc, tai)
    np.testing.assert_allclose(converted, [109.0, 111.0])
    np.testing.assert_allclose(convert_time(converted, tai, utc)[0], [99.0, 100.0])
    assert transform.resources[0].manifest_id == resource.manifest_id

    coordinates = GeospatialContract.local_cartesian(
        phx.SpatialCoordinateContract.si(), vertical_datum="local"
    )
    trajectory = BoreholeTrajectory(
        "well-a",
        [0.0, 10.0, 20.0],
        [[0.0, 0.0, 0.0], [0.0, 0.0, -8.0], [6.0, 0.0, -16.0]],
        coordinates,
    )
    np.testing.assert_allclose(
        trajectory.sample_positions([5.0, 15.0]),
        [[0.0, 0.0, -4.0], [3.0, 0.0, -12.0]],
    )
    sample = trajectory.prepare_sampling([5.0, 15.0])
    station = jnp.asarray((2.0, 4.0, 8.0))
    query = sample.apply(station)
    cotangent = jnp.asarray((0.3, -0.2))
    np.testing.assert_allclose(
        jnp.vdot(query, cotangent),
        jnp.vdot(station, sample.transpose(cotangent, 3)),
    )

    body = ReferenceBodyContract("test", 4.0e14, 6.4e6, 6.3e6, 1e-4, 0.0)
    angular_coordinates = GeospatialContract(
        phx.SpatialCoordinateContract(METER, reference_frame="test-body"),
        horizontal_crs="test-body-geographic",
        horizontal_kind="geographic",
        horizontal_axes=("longitude", "latitude"),
        horizontal_units=(DEGREE, DEGREE),
        horizontal_datum="test-body",
        epoch_required=False,
        vertical_kind="elevation",
        vertical_positive="up",
        vertical_datum="test-body-surface",
        vertical_unit=METER,
        registration="unknown",
        longitude_seam="none",
        longitude_domain=(-180.0, 180.0),
        mask_semantics="none",
    )
    planetary = PlanetaryCoordinateContract(
        body,
        angular_coordinates,
        latitude_kind="planetographic",
    )
    point = planetary.to_body_fixed_cartesian(jnp.asarray((0.0, 0.0, 0.0)))
    np.testing.assert_allclose(point, [6.4e6, 0.0, 0.0])
    rotated = planetary.body_fixed_to_inertial(point, np.pi / (2e-4))
    np.testing.assert_allclose(rotated[:2], [0.0, 6.4e6], atol=1e-8)
    np.testing.assert_allclose(
        planetary.inertial_to_body_fixed(rotated, np.pi / (2e-4)), point, atol=1e-8
    )


def test_tetrahedral_hcurl_exact_sequence_and_positive_actions():
    space = phx.discretization.TetrahedralNedelecSpace(_tetra_mesh())
    vertex = jnp.asarray((0.3, -0.2, 1.1, 0.7))
    edge_gradient = space.gradient(vertex)
    np.testing.assert_allclose(space.discrete_curl(edge_gradient), 0.0, atol=1e-14)
    edge = jnp.arange(space.edge_count, dtype=float) + 1
    np.testing.assert_allclose(
        space.discrete_divergence(space.discrete_curl(edge)), 0.0, atol=1e-14
    )
    mass = space.mass_action(edge, 2.0)
    curl = space.curl_curl_action(edge, 3.0)
    assert float(jnp.vdot(edge, mass)) > 0
    assert float(jnp.vdot(edge, curl)) >= 0


def test_covariance_actions_variable_projection_and_priors_match_dense_references():
    layout = CoordinateLayout(("a", "b", "c", "d"))
    diagonal = DiagonalCovarianceAction([1.0, 2.0, 3.0, 4.0], layout)
    residual = jnp.asarray((0.2, -0.4, 0.5, 0.1))
    np.testing.assert_allclose(
        diagonal.quadratic(residual), np.sum(np.asarray(residual) ** 2 / np.arange(1, 5))
    )
    factors = np.asarray(((0.2,), (0.1,), (-0.3,), (0.4,)))
    low_rank = LowRankDiagonalCovarianceAction(np.arange(1, 5), factors, layout)
    dense = np.diag(np.arange(1, 5)) + factors @ factors.T
    np.testing.assert_allclose(
        low_rank.quadratic(residual),
        residual @ np.linalg.solve(dense, residual),
        rtol=1e-12,
    )
    first = np.linalg.cholesky(np.asarray(((1.2, 0.2), (0.2, 0.8))))
    second = np.linalg.cholesky(np.asarray(((0.7, 0.1), (0.1, 1.1))))
    kronecker = KroneckerCholeskyCovarianceAction((first, second), layout)
    covariance = np.kron(first @ first.T, second @ second.T)
    np.testing.assert_allclose(
        kronecker.quadratic(residual),
        residual @ np.linalg.solve(covariance, residual),
        rtol=1e-12,
    )
    circulant = CirculantCovarianceAction([1.0, 2.0, 1.5], layout)
    assert jnp.isfinite(circulant.quadratic(residual))
    precision_values = jnp.asarray([2.0, 3.0, 4.0, 5.0])
    precision_space = phx.linalg.ArraySpace((4,))
    precision_operator = phx.linalg.DiagonalLinearOperator(
        precision_values,
        space=precision_space,
        properties=phx.linalg.OperatorProperties(
            diagonal=True,
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "diagonal": "construction",
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    precision = PrecisionOperatorCovarianceAction(
        precision_operator, -jnp.sum(jnp.log(precision_values)), layout
    )
    np.testing.assert_allclose(
        precision.quadratic(residual), jnp.sum(precision_values * residual**2)
    )
    spde = SPDEPrecisionPrior(
        precision_operator,
        jnp.zeros(4),
        logdet_precision=jnp.sum(jnp.log(precision_values)),
    )
    assert jnp.isfinite(spde.log_prob(jnp.zeros(4)))

    design = jnp.asarray(((1.0, 0.0), (1.0, 1.0), (1.0, 2.0), (1.0, 3.0)))
    truth = jnp.asarray((2.0, -0.5))
    projection = LinearNuisancePlan(design, diagonal, layout, ("offset", "slope"))
    result = projection.evaluate(design @ truth, jnp.zeros(4))
    np.testing.assert_allclose(result.parameters, truth, atol=1e-12)
    assert result.successful

    graph = GraphMetricPrior([1.0, 1.0], [0], [1], [2.0], gradient_precision=1.0)
    assert graph.log_prob([1.0, 1.0]) == 0
    temporal = TemporalDifferencePrior([0.0, 1.0, 3.0], 2.0)
    np.testing.assert_allclose(
        temporal.standardized_difference([[0.0], [2.0], [6.0]]), 1.0
    )
    cross = CrossGradientPrior(np.asarray([[[1, 0], [0, 1]]], dtype=float), [1.0], 2, 1.0)
    np.testing.assert_allclose(cross.log_prob([1.0, 0.0], [2.0, 0.0]), 0.0)


def test_stochastic_design_never_materializes_identity_and_is_differentiable():
    dimension = 4
    space = phx.linalg.ArraySpace((dimension,))

    def factory(design):
        diagonal = 1.0 + jnp.exp(design) * jnp.arange(1, dimension + 1)
        return phx.linalg.DiagonalLinearOperator(
            diagonal,
            space=space,
            properties=phx.linalg.OperatorProperties(
                diagonal=True,
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "diagonal": "construction",
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                    "positive_semidefinite": "construction",
                },
            ),
        )

    policy = phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())
    design = StochasticExperimentDesignPlan(
        factory,
        jnp.asarray(0.0),
        jax.random.key(4),
        precision_factory_id="diagonal-exp-precision-v1",
        parameter_dimension=dimension,
        probe_count=16,
        criterion="a-optimal",
        linear_policy=policy,
        linear_policy_id="dense-lu-v1",
    )
    result = design.evaluate(jnp.asarray(0.1))
    assert result.successful
    assert jnp.isfinite(jax.grad(design.objective)(jnp.asarray(0.1)))

    def identity_factory(_design):
        return phx.linalg.DiagonalLinearOperator(
            jnp.ones(dimension),
            space=space,
            properties=phx.linalg.OperatorProperties(
                diagonal=True,
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "diagonal": "construction",
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                    "positive_semidefinite": "construction",
                },
            ),
        )

    d_optimal = StochasticExperimentDesignPlan(
        identity_factory,
        jnp.asarray(0.0),
        jax.random.key(5),
        precision_factory_id="identity-precision-v1",
        parameter_dimension=dimension,
        probe_count=8,
        criterion="d-optimal",
    ).evaluate(jnp.asarray(0.0))
    e_optimal = StochasticExperimentDesignPlan(
        identity_factory,
        jnp.asarray(0.0),
        jax.random.key(6),
        precision_factory_id="identity-precision-v1",
        parameter_dimension=dimension,
        probe_count=8,
        criterion="e-optimal",
    ).evaluate(jnp.asarray(0.0))
    assert d_optimal.successful and d_optimal.lanczos_breakdown
    assert e_optimal.successful and e_optimal.lanczos_breakdown
    np.testing.assert_allclose(d_optimal.estimate, 0.0, atol=1e-12)
    np.testing.assert_allclose(e_optimal.estimate, 1.0, atol=1e-12)


def test_robust_and_censored_likelihoods_are_normalized_and_finite():
    contaminated = ContaminatedGaussianLikelihood(
        1.0, outlier_scale_factor=10.0, outlier_probability=0.05
    )
    assert jnp.isfinite(contaminated.log_prob(0.0, 100.0))
    censored = CensoredGaussianLikelihood(1.0, -2.0, 2.0)
    values = censored.log_prob(
        jnp.zeros(3), jnp.asarray((-2.0, 0.0, 2.0)), censoring=jnp.asarray((-1, 0, 1))
    )
    assert jnp.all(jnp.isfinite(values))


def test_resource_evidence_and_checkpoint_round_trip(tmp_path):
    grid = geo.AcousticGrid((7, 7), (1.0, 1.0))
    plan = geo.ConstantDensityAcousticPlan(grid, 0.1, 4, 2.0)
    evidence = plan.capability_evidence
    estimate = plan.resource_estimate(receiver_count=2, checkpoint_count=2)
    assert evidence.model == "constant-density-acoustic"
    assert estimate.total_bytes > 0
    policy = geo.GeophysicalResourcePolicy(
        maximum_device_bytes=estimate.total_bytes,
        maximum_checkpoint_bytes=10_000_000,
        maximum_sources=2,
        maximum_observations=100,
        maximum_steps=10,
    )
    policy.admit(estimate, source_count=1, observation_count=2, step_count=4)
    continuation = geo.GeophysicalContinuationState(
        plan.initial_state(),
        time=0.0,
        accepted_step=0,
        source_position=0,
        random_key=jax.random.key_data(jax.random.key(1)),
        topology_epoch=0,
    )
    checkpoint = geo.GeophysicalCheckpointPlan(
        plan.plan_id, grid.grid_id, "observation", "serial", policy
    )
    path = tmp_path / "restart.phx"
    checkpoint.write(path, continuation)
    restored = checkpoint.read(path, continuation)
    assert restored.physical_state.plan_id == plan.plan_id
