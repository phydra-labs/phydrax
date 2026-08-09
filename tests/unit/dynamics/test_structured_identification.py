#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_implicit_sindy_recovers_rational_dynamics_as_sparse_equation():
    state = jnp.linspace(0.1, 2.0, 300)[:, None]
    derivative = -state / (1.0 + state)
    layout = phx.dynamics.StateLayout((1,), component_names=("x",))
    data = phx.dynamics.TrajectoryData(
        jnp.arange(state.shape[0], dtype=float),
        state,
        state_layout=layout,
        derivatives=derivative,
        source_id="implicit-rational",
    )
    problem = phx.dynamics.identification.ImplicitSINDyProblem(
        data=data,
        library=phx.dynamics.identification.PolynomialImplicitFeatureLibrary(
            layout, degree=2
        ),
    )

    result = phx.dynamics.identification.fit_implicit_sindy(
        problem,
        phx.dynamics.identification.SequentialThresholdedLeastSquares(
            1e-7, threshold_space="physical"
        ),
        targets=("state:d(x)/dcoordinate",),
    )

    assert bool(result.valid)
    names = result.feature_names
    expected = np.zeros((len(names),))
    expected[names.index("state:x")] = 1.0
    expected[names.index("state:d(x)/dcoordinate")] = 1.0
    expected[names.index("state:x * state:d(x)/dcoordinate")] = 1.0
    np.testing.assert_allclose(
        np.asarray(result.selected.coefficients), expected, atol=2e-10
    )
    assert result.selected.equation().endswith(" = 0")


def test_pde_find_recovers_diffusion_from_structured_grid():
    time = jnp.linspace(0.0, 1.0, 61)
    space = jnp.linspace(0.0, 2.0 * jnp.pi, 161)
    diffusivity = 0.1
    values = (
        jnp.exp(-diffusivity * time[:, None]) * jnp.sin(space)[None, :]
        + 0.35
        * jnp.exp(-4.0 * diffusivity * time[:, None])
        * jnp.sin(2.0 * space)[None, :]
    )[..., None]
    layout = phx.dynamics.StateLayout((1,), component_names=("u",))
    data = phx.dynamics.identification.StructuredPDEData(
        (time, space),
        values,
        state_layout=layout,
        coordinate_names=("t", "x"),
        source_id="two-mode-diffusion",
    )
    library = phx.dynamics.identification.PolynomialPDELibrary(
        layout,
        ("t", "x"),
        polynomial_degree=1,
        spatial_derivative_order=2,
        include_interactions=False,
    )
    problem = phx.dynamics.identification.PDEIdentificationProblem(
        data=data,
        library=library,
    )

    result = phx.dynamics.identification.fit_pde_find(
        problem,
        phx.dynamics.identification.SequentialThresholdedLeastSquares(
            0.02, threshold_space="physical"
        ),
    )

    assert bool(result.valid)
    names = result.design.feature_names
    second_derivative = names.index("d^2(u)/dx^2")
    assert int(jnp.sum(result.support)) == 1
    np.testing.assert_allclose(
        np.asarray(result.coefficients[0, second_derivative]),
        diffusivity,
        rtol=5e-3,
    )
    assert result.equations()[0].startswith("d(u)/dt = ")


def test_structured_regression_enforces_groups_forbidden_terms_and_constraint():
    generator = np.random.default_rng(9)
    states = generator.normal(size=(400, 2))
    derivatives = np.stack((2.0 * states[:, 0], -2.0 * states[:, 0]), axis=-1)
    layout = phx.dynamics.StateLayout((2,), component_names=("a", "b"))
    problem = phx.dynamics.identification.SINDyProblem(
        data=phx.dynamics.TrajectoryData(
            jnp.arange(states.shape[0], dtype=float),
            states,
            state_layout=layout,
            derivatives=derivatives,
            source_id="conserved-pair",
        ),
        library=phx.dynamics.identification.PolynomialFeatureLibrary(layout, degree=1),
        formulation=phx.dynamics.identification.StrongSINDyFormulation(),
    )
    design = problem.build_design()
    constraint = phx.dynamics.identification.named_coefficient_constraint(
        design,
        (
            {
                ("a", "state:a"): 1.0,
                ("b", "state:a"): 1.0,
            },
        ),
        (0.0,),
        constraint_id="pairwise-conservation",
    )
    allowed = (
        jnp.ones((design.output_size, design.num_features), dtype=bool)
        .at[:, design.feature_names.index("1")]
        .set(False)
    )
    structure = phx.dynamics.identification.CoefficientStructure(
        groups=phx.dynamics.identification.shared_feature_groups(
            design.output_size, design.num_features
        ),
        allowed=allowed,
        constraint=constraint,
    )

    result = phx.dynamics.identification.StructuredSequentialThresholdedLeastSquares(
        0.1,
        structure=structure,
        tolerance=1e-10,
    ).fit(design)

    assert bool(result.successful)
    expected = np.zeros_like(np.asarray(result.coefficients))
    expected[0, design.feature_names.index("state:a")] = 2.0
    expected[1, design.feature_names.index("state:a")] = -2.0
    np.testing.assert_allclose(np.asarray(result.coefficients), expected, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(constraint.matrix @ result.coefficients.reshape((-1,))),
        np.asarray(constraint.rhs),
        atol=1e-12,
    )


def test_symmetry_average_projects_odd_features_out():
    layout = phx.dynamics.StateLayout((1,), component_names=("x",))
    base = phx.dynamics.identification.PolynomialFeatureLibrary(layout, degree=2)
    symmetric = phx.dynamics.identification.SymmetryAveragedFeatureLibrary(
        base,
        (lambda state: state, lambda state: -state),
        symmetry_id="reflection-x",
    )

    values = symmetric(jnp.asarray([[-2.0], [3.0]]))

    np.testing.assert_allclose(
        np.asarray(values[:, symmetric.feature_names.index("state:x")]), 0.0
    )
    np.testing.assert_allclose(
        np.asarray(values[:, symmetric.feature_names.index("state:x^2")]),
        np.asarray([4.0, 9.0]),
    )
