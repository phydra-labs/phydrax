import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications import polymer_field_theory as pft


def _spectral(count=8):
    return phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(count),),
        axis_names=("x",),
        field_name="scft-field",
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 4.0),))


def _diblock_model():
    architecture = pft.PolymerContourArchitecturePlan(
        "AB-linear",
        (
            pft.ContourBlockPlan("A", "left", "junction", 0, 0.5, 4),
            pft.ContourBlockPlan("B", "junction", "right", 1, 0.5, 4),
        ),
        root_node="left",
    )
    component = pft.PolymerComponentPlan("AB", architecture, 1.0, 20.0)
    return pft.IncompressibleGaussianMixturePlan(
        ("A", "B"), [1.0, 1.0], [[0.0, 0.0], [0.0, 0.0]], (component,)
    )


def test_linear_scft_homogeneous_control_fixes_pressure_gauge():
    prepared = pft.SCFTPlan(_diblock_model()).prepare(_spectral())
    fields = jnp.zeros(prepared.field_shape)

    evaluation = prepared.evaluate(fields)
    result = pft.solve_scft(prepared, fields)
    implicit = pft.solve_scft_implicit(prepared, fields)

    assert evaluation.successful & result.successful & implicit.successful
    np.testing.assert_allclose(evaluation.partition_functions, [1.0], atol=1.0e-12)
    np.testing.assert_allclose(evaluation.densities[..., 0], 0.5, atol=1.0e-12)
    np.testing.assert_allclose(evaluation.densities[..., 1], 0.5, atol=1.0e-12)
    np.testing.assert_allclose(evaluation.residual, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(evaluation.gauge_residual, 0.0, atol=0.0)


def test_richardson_strang_propagator_is_exact_for_constant_field():
    prepared = pft.SCFTPlan(
        _diblock_model(),
        contour_integrator=pft.ContourIntegratorPlan("richardson-strang-4"),
    ).prepare(_spectral())
    fields = jnp.full(prepared.field_shape, 0.3)

    evaluation = prepared.evaluate(fields)

    assert evaluation.successful
    np.testing.assert_allclose(
        evaluation.partition_functions, [np.exp(-0.3)], rtol=1.0e-11, atol=1.0e-12
    )


def test_acyclic_branched_scft_messages_preserve_homogeneous_mass():
    architecture = pft.PolymerContourArchitecturePlan(
        "three-arm",
        (
            pft.ContourBlockPlan("arm-a", "root", "a", 0, 0.25, 2),
            pft.ContourBlockPlan("arm-b", "root", "b", 1, 0.25, 2),
            pft.ContourBlockPlan("arm-c", "root", "c", 0, 0.5, 4),
        ),
        root_node="root",
    )
    model = pft.IncompressibleGaussianMixturePlan(
        ("A", "B"),
        [1.0, 1.0],
        [[0.0, 0.0], [0.0, 0.0]],
        (pft.PolymerComponentPlan("star", architecture, 1.0, 12.0),),
    )
    prepared = pft.SCFTPlan(model).prepare(_spectral())

    evaluation = prepared.evaluate(jnp.zeros(prepared.field_shape))

    assert evaluation.successful
    np.testing.assert_allclose(evaluation.partition_functions, [1.0], atol=1.0e-12)
    np.testing.assert_allclose(jnp.sum(evaluation.densities, axis=-1), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(evaluation.densities[..., 0], 0.75, atol=1.0e-12)
    np.testing.assert_allclose(evaluation.densities[..., 1], 0.25, atol=1.0e-12)


def test_scft_continuation_variable_cell_symmetry_and_field_sampling():
    spectral = _spectral(4)
    symmetry = phx.discretization.TensorSpectralSymmetry(spectral, component_count=2)
    prepared = pft.SCFTPlan(_diblock_model()).prepare(spectral, symmetries=(symmetry,))
    fields = jnp.zeros(prepared.field_shape)
    root = pft.solve_scft(prepared, fields)

    symmetry_evidence = pft.scft_symmetry_evidence(prepared, fields)
    assert symmetry_evidence.successful

    continuation = pft.continue_scft_interactions(
        prepared,
        root,
        0.0,
        0.1,
        pft.SCFTInteractionContinuationPlan(
            minimum_scale=0.0,
            maximum_scale=1.0,
            initial_step=0.05,
            maximum_step=0.1,
            maximum_steps=4,
        ),
    )
    assert continuation.successful

    variable_cell = pft.solve_isotropic_cell_scft(
        prepared,
        fields,
        1.1,
        pft.IsotropicSCFTCellPlan(
            minimum_scale=0.5,
            maximum_scale=2.0,
            maximum_iterations=4,
        ),
    )
    assert variable_cell.successful
    np.testing.assert_allclose(variable_cell.cell_scale, 1.1)

    partial_runtime = pft.PartialSaddleFTSPlan(
        num_steps=2,
        step_size=1.0e-10,
        pressure_iterations=2,
        maximum_incompressibility=1.0,
    ).prepare(prepared)
    partial_state = pft.initialize_partial_saddle_fts(
        partial_runtime, fields, key=jax.random.key(4)
    )
    partial = pft.sample_partial_saddle_fts(partial_runtime, partial_state)
    assert partial.successful
    assert partial.samples.shape[0] == 2

    complex_runtime = pft.ComplexFTSPlan(
        phx.applications.sign_problem.ComplexLangevinPlan(
            num_steps=2,
            step_size=1.0e-10,
            tail_window=2,
            maximum_tail_probability=1.0,
            maximum_state_norm=1.0e8,
        )
    ).prepare(prepared)
    complex_result = pft.sample_complex_fts(
        complex_runtime,
        jnp.zeros(prepared.field_shape, dtype=jnp.complex128),
        key=jax.random.key(5),
    )
    assert complex_result.successful
