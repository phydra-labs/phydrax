import math

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_periodic_spectral_conservation_and_entropy_diagnostics():
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(24),),
        axis_names=("x",),
        field_name="u",
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="unit-advection",
    )
    entropy = phx.equations.ConvexEntropyPair(
        system,
        lambda state: 0.5 * state[..., 0] ** 2,
        lambda state: state,
        lambda state, axis, args: 0.5 * state[..., 0] ** 2,
        lambda state: jnp.ones(state.shape[:-1], dtype=bool),
        entropy_id="quadratic",
    )
    problem = phx.equations.ConservationProblemIR(
        "advection",
        "u",
        system,
        None,
    )
    method = phx.discretization.SpectralConservationMethodPlan(
        phx.discretization.PseudospectralMethodPlan(),
        flux_polynomial_degree=1,
        entropy_diagnostics=True,
    )
    compiled = phx.equations.compile_conservation_problem(
        problem,
        space,
        method,
        entropy_pair=entropy,
    )
    x = space.axes[0].nodes
    physical = jnp.sin(2.0 * jnp.pi * x)[..., None]
    state = space.project(physical)

    residual, diagnostics = compiled.residual_with_diagnostics(0.0, state)
    physical_residual = space.reconstruct(residual)
    balance_terms = np.asarray(space.quadrature_weights[..., None] * physical_residual)
    expected_rate = np.asarray(
        [
            math.fsum(balance_terms[:, index].tolist())
            for index in range(balance_terms.shape[1])
        ]
    )
    np.testing.assert_allclose(
        diagnostics.semidiscrete_integral_rate,
        expected_rate,
        rtol=8e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        diagnostics.conservation_defect,
        expected_rate,
        rtol=8e-15,
        atol=0.0,
    )
    expected = -2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * x)

    assert jnp.allclose(
        space.reconstruct(residual)[..., 0],
        expected,
        rtol=1e-10,
        atol=1e-10,
    )
    assert jnp.max(jnp.abs(diagnostics.conservation_defect)) < 1e-11
    assert diagnostics.entropy is not None
    assert jnp.abs(diagnostics.entropy.semidiscrete_entropy_rate) < 1e-11
    assert diagnostics.entropy.admissible


def test_spectral_conservation_honors_widened_reduction_precision():
    precision = phx.discretization.SpectralPrecisionPolicy(
        jnp.float32,
        reduction_dtype=jnp.float64,
        certification_dtype=jnp.float64,
    )
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(32),),
        axis_names=("x",),
        field_name="u",
        precision=precision,
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="mixed-precision-unit-advection",
    )
    problem = phx.equations.ConservationProblemIR(
        "mixed-precision-advection",
        "u",
        system,
        None,
    )
    method = phx.discretization.SpectralConservationMethodPlan(
        phx.discretization.PseudospectralMethodPlan(),
        flux_polynomial_degree=1,
    )
    compiled = phx.equations.compile_conservation_problem(
        problem,
        space,
        method,
    )
    x = space.axes[0].nodes
    state = space.project(jnp.sin(2.0 * jnp.pi * x)[..., None])

    residual, diagnostics = compiled.residual_with_diagnostics(0.0, state)
    physical_residual = precision.reduction(space.reconstruct(residual))
    weights = precision.reduction(space.quadrature_weights[..., None])
    balance_terms = np.asarray(weights * physical_residual)
    expected = np.asarray(
        [
            math.fsum(balance_terms[:, index].tolist())
            for index in range(balance_terms.shape[1])
        ]
    )

    assert diagnostics.semidiscrete_integral_rate.dtype == jnp.float64
    assert diagnostics.conservation_defect.dtype == jnp.float64
    np.testing.assert_allclose(
        diagnostics.semidiscrete_integral_rate,
        expected,
        rtol=8e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        diagnostics.conservation_defect,
        expected,
        rtol=8e-15,
        atol=0.0,
    )


def test_periodic_fourier_split_form_is_conservative_entropy_stable_and_chunk_invariant():
    from phydrax.discretization import spectral

    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(12),),
        axis_names=("x",),
        field_name="state",
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 2.0),))
    system = phx.equations.EulerSystem(1)
    pair = phx.equations.ideal_gas_euler_entropy_pair(system)
    problem = phx.equations.ConservationProblemIR("split-euler", "state", system, None)
    x = space.axes[0].nodes
    primitive = jnp.stack(
        (
            1.0 + 0.05 * jnp.sin(jnp.pi * x),
            0.2 * jnp.cos(jnp.pi * x),
            1.0 + 0.03 * jnp.sin(2.0 * jnp.pi * x),
        ),
        axis=-1,
    )
    state = space.project(system.primitive_to_conserved(primitive))

    def compiled(chunk):
        method = spectral.SpectralConservationMethodPlan(
            split_form=spectral.SpectralSplitFormPlan(
                phx.discretization.EntropyConservativeEulerFluxPlan(),
                pair_chunk_size=chunk,
                maximum_pair_workspace_bytes=1_000_000,
            ),
            entropy_diagnostics=True,
        )
        return phx.equations.compile_conservation_problem(
            problem, space, method, entropy_pair=pair
        )

    residual_a, diagnostics = compiled(5).residual_with_diagnostics(0.0, state)
    residual_b = compiled(64)(0.0, state)
    np.testing.assert_allclose(residual_a, residual_b, rtol=2e-12, atol=2e-12)
    assert jnp.max(jnp.abs(diagnostics.conservation_defect)) < 5e-12
    assert diagnostics.entropy.entropy_stable
    assert diagnostics.entropy.convective_entropy_defect < 5e-11
    assert diagnostics.entropy.pair_workspace_bytes > 0
    assert space.periodic_cell.fully_periodic


def test_split_form_workspace_bound_includes_every_transverse_line():
    from phydrax.discretization import spectral

    space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(4),
            phx.discretization.FourierBasisPlan(6),
            phx.discretization.FourierBasisPlan(8),
        ),
        axis_names=("x", "y", "z"),
        field_name="state",
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 1.0),
        )
    )
    component_count = len(space.axes) + 2
    itemsize = np.dtype(space.plan.precision.physical_dtype).itemsize
    one_line_workspace = component_count * 3 * itemsize
    transverse_line_count = max(
        math.prod(space.modal_shape[:axis] + space.modal_shape[axis + 1 :])
        for axis in range(len(space.axes))
    )
    expected_workspace = transverse_line_count * one_line_workspace

    insufficient = spectral.SpectralConservationMethodPlan(
        split_form=spectral.SpectralSplitFormPlan(
            phx.discretization.EntropyConservativeEulerFluxPlan(),
            pair_chunk_size=1,
            maximum_pair_workspace_bytes=one_line_workspace,
        )
    )
    with np.testing.assert_raises_regex(ValueError, "workspace"):
        insufficient.prepare(space)

    prepared = spectral.SpectralConservationMethodPlan(
        split_form=spectral.SpectralSplitFormPlan(
            phx.discretization.EntropyConservativeEulerFluxPlan(),
            pair_chunk_size=1,
            maximum_pair_workspace_bytes=expected_workspace,
        )
    ).prepare(space)
    assert prepared.split_form.report.pair_workspace_bytes == expected_workspace


def test_split_form_rejects_unsupported_flux_source_and_basis():
    from phydrax.discretization import spectral

    with np.testing.assert_raises(TypeError):
        spectral.SpectralSplitFormPlan(phx.discretization.RusanovFluxPlan())
    polynomial = phx.discretization.TensorSpectralPlan(
        (phx.discretization.ChebyshevBasisPlan(8),)
    ).prepare((phx.discretization.AxisDomain.interval(-1.0, 1.0),))
    method = spectral.SpectralConservationMethodPlan(
        split_form=spectral.SpectralSplitFormPlan(
            phx.discretization.EntropyConservativeEulerFluxPlan()
        )
    )
    with np.testing.assert_raises(ValueError):
        method.prepare(polynomial)
