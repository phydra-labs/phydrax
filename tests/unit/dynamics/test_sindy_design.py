#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_strong_design_uses_sample_aligned_controls_and_derivatives():
    time = jnp.linspace(0.0, 2.0, 21)
    state = (0.2 + time**2)[:, None]
    control = jnp.sin(time)[:, None]
    derivative = 2.0 * state + 3.0 * control
    state_layout = phx.dynamics.StateLayout((1,), component_names=("x",))
    input_layout = phx.dynamics.InputLayout((1,), component_names=("u",), roles="control")
    data = phx.dynamics.TrajectoryData(
        time,
        state,
        state_layout=state_layout,
        inputs=control,
        input_layout=input_layout,
        input_alignment="samples",
        derivatives=derivative,
        source_id="controlled-strong",
    )
    library = phx.dynamics.identification.PolynomialFeatureLibrary(
        state_layout, input_layout=input_layout, degree=1
    )
    problem = phx.dynamics.identification.SINDyProblem(
        data=data,
        library=library,
        formulation=phx.dynamics.identification.StrongSINDyFormulation(),
    )

    design = phx.dynamics.identification.build_sindy_design(problem)

    assert design.num_rows == time.size
    assert design.feature_names == ("1", "state:x", "input:u")
    np.testing.assert_allclose(
        np.asarray(design.target[:, 0]),
        np.asarray(2.0 * design.matrix[:, 1] + 3.0 * design.matrix[:, 2]),
        atol=1e-12,
    )


def test_discrete_design_targets_next_state_without_derivative_reinterpretation():
    state = jnp.asarray([0.1, 0.58, 0.964, 1.2712, 1.51696])[:, None]
    layout = phx.dynamics.StateLayout((1,), component_names=("x",))
    data = phx.dynamics.TrajectoryData(
        jnp.arange(state.shape[0]),
        state,
        state_layout=layout,
        source_id="affine-map",
    )
    library = phx.dynamics.identification.PolynomialFeatureLibrary(layout, degree=1)

    design = phx.dynamics.identification.SINDyProblem(
        data=data,
        library=library,
        formulation=phx.dynamics.identification.DiscreteSINDyFormulation(),
    ).build_design()

    np.testing.assert_allclose(
        np.asarray(design.target[:, 0]),
        np.asarray(0.5 + 0.8 * design.matrix[:, 1]),
        atol=1e-12,
    )


def test_integral_design_matches_constant_vector_field_identity():
    time = jnp.asarray([0.0, 0.1, 0.35, 0.8, 1.4, 2.0])
    layout = phx.dynamics.StateLayout((1,), component_names=("x",))
    data = phx.dynamics.TrajectoryData(
        time,
        (2.0 * time - 0.3)[:, None],
        state_layout=layout,
        source_id="constant-flow",
    )
    constant = phx.dynamics.identification.PolynomialFeatureLibrary(layout, degree=0)

    design = phx.dynamics.identification.SINDyProblem(
        data=data,
        library=constant,
        formulation=phx.dynamics.identification.IntegralSINDyFormulation(
            window_size=3, stride=2, boundary="partial"
        ),
    ).build_design()

    assert bool(jnp.all(design.valid))
    np.testing.assert_allclose(
        np.asarray(design.target[:, 0]),
        np.asarray(2.0 * design.matrix[:, 0]),
        atol=1e-12,
    )


def test_weak_and_integral_windows_crossing_reset_are_invalid():
    time = jnp.linspace(0.0, 1.0, 11)
    layout = phx.dynamics.StateLayout((1,))
    data = phx.dynamics.TrajectoryData(
        time,
        time[:, None],
        state_layout=layout,
        reset_mask=jnp.asarray(
            [False, False, False, False, True, False, False, False, False, False]
        ),
        source_id="reset-flow",
    )
    library = phx.dynamics.identification.PolynomialFeatureLibrary(layout, degree=1)
    integral = phx.dynamics.identification.SINDyProblem(
        data=data,
        library=library,
        formulation=phx.dynamics.identification.IntegralSINDyFormulation(window_size=3),
    ).build_design()
    weak = phx.dynamics.identification.SINDyProblem(
        data=data,
        library=library,
        formulation=phx.dynamics.identification.WeakSINDyFormulation(
            window_size=3, test_orders=(1, 2)
        ),
    ).build_design()

    crossing_integral = (integral.window_start <= 4) & (integral.window_end > 4)
    crossing_weak = (weak.window_start <= 4) & (weak.window_end > 4)
    assert not bool(jnp.any(integral.valid[crossing_integral]))
    assert not bool(jnp.any(weak.valid[crossing_weak]))
