import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _affine_model(*, basis_matrix=None):
    full = phx.linalg.ArraySpace((3,), dtype=jnp.float64, space_id="affine-full")
    basis_values = (
        jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=jnp.float64)
        if basis_matrix is None
        else jnp.asarray(basis_matrix, dtype=jnp.float64)
    )
    basis = phx.rom.ReducedBasisArtifact(
        phx.linalg.LinearSubspace(
            full,
            basis_values,
            orthonormal=basis_matrix is None,
            subspace_id=f"affine-subspace-{float(jnp.sum(basis_values))}",
        ),
        role="state",
        state_contract_id="three-component-state",
        support_id="fixed-support",
        measure_id="euclidean-measure",
        geometry_id="fixed-geometry",
        source_artifact_ids=("affine-snapshots",),
    )
    reduction = phx.rom.trial_test_reduction_from_bases(basis)
    dual = phx.linalg.DualSpace(full)
    operators = (
        phx.linalg.DenseLinearOperator(
            jnp.asarray(
                [[2.0, 0.0, 1.0], [0.0, 3.0, 2.0], [0.0, 0.0, 4.0]],
                dtype=jnp.float64,
            ),
            source=full,
            target=dual,
            operator_id="affine-A0",
        ),
        phx.linalg.DenseLinearOperator(
            jnp.diag(jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)),
            source=full,
            target=dual,
            operator_id="affine-A1",
        ),
    )
    output = phx.linalg.ArraySpace((1,), dtype=jnp.float64, space_id="affine-qoi")
    observation = phx.linalg.DenseLinearOperator(
        jnp.ones((1, 3), dtype=jnp.float64),
        source=full,
        target=output,
        operator_id="affine-sum-qoi",
    )
    problem = phx.rom.AffineLinearROMProblem(
        reduction,
        operators,
        (
            jnp.asarray([2.0, 6.0, 0.0], dtype=jnp.float64),
            jnp.asarray([2.0, 2.0, 4.0], dtype=jnp.float64),
        ),
        operator_term_ids=("A0", "A1"),
        right_hand_side_term_ids=("b0", "b1"),
        lift_terms=(jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64),),
        lift_term_ids=("boundary",),
        observations=(("sum", observation),),
        source_artifact_ids=("manufactured-affine-family",),
    )
    coefficients = phx.rom.ArrayAffineCoefficientMap(
        jnp.asarray([[0.0], [1.0]], dtype=jnp.float64),
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        jnp.asarray([[0.0], [1.0]], dtype=jnp.float64),
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        operator_term_ids=("A0", "A1"),
        right_hand_side_term_ids=("b0", "b1"),
        lift_matrix=jnp.asarray([[1.0]], dtype=jnp.float64),
        lift_offset=jnp.asarray([0.0], dtype=jnp.float64),
        lift_term_ids=("boundary",),
        lower=jnp.asarray([0.0], dtype=jnp.float64),
        upper=jnp.asarray([1.0], dtype=jnp.float64),
        input_contract_id="mu-vector",
        unit_contract_id="dimensionless-mu",
        support_id="fixed-support",
    )
    return problem, phx.rom.prepare_affine_linear_rom(problem, coefficients)


def test_affine_rom_precomputes_lift_cross_terms_and_audits_physical_state():
    _, model = _affine_model()
    evaluation = model.evaluate(jnp.asarray([0.5], dtype=jnp.float64), reconstruct=True)

    assert bool(evaluation.valid)
    np.testing.assert_allclose(evaluation.reduced_state, np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(
        evaluation.reconstructed_state,
        np.asarray([1.0, 2.0, 0.5]),
    )
    np.testing.assert_allclose(evaluation.observations[0].value, np.asarray([3.5]))
    np.testing.assert_allclose(evaluation.reduced_residual, 0.0, atol=1e-12)

    audit = phx.rom.audit_affine_linear_rom(
        model,
        evaluation,
        jnp.asarray([1.0, 2.0, 0.5], dtype=jnp.float64),
        truth_observations={"sum": jnp.asarray([3.5], dtype=jnp.float64)},
    )
    assert bool(audit.valid)
    np.testing.assert_allclose(audit.state_error_norm, 0.0, atol=1e-12)


def test_affine_rom_is_invariant_to_trial_and_test_coordinates():
    _, canonical = _affine_model()
    _, rescaled = _affine_model(basis_matrix=[[2.0, 0.0], [0.0, 0.5], [0.0, 0.0]])
    inputs = jnp.asarray([0.25], dtype=jnp.float64)
    left = canonical.evaluate(inputs, reconstruct=True)
    right = rescaled.evaluate(inputs, reconstruct=True)

    np.testing.assert_allclose(left.reconstructed_state, right.reconstructed_state)
    np.testing.assert_allclose(left.observations[0].value, right.observations[0].value)


def test_affine_rom_refuses_unsupported_input_before_solve():
    _, model = _affine_model()
    evaluation = model.evaluate(jnp.asarray([2.0], dtype=jnp.float64), reconstruct=True)

    assert not bool(evaluation.valid)
    assert evaluation.solve_result is None
    assert evaluation.reconstructed_state is None


def test_affine_rom_archive_and_certificate_are_content_bound(tmp_path):
    problem, model = _affine_model()
    path = phx.rom.write_affine_linear_rom(
        tmp_path / "affine.phx",
        model,
        analysis_plan_id="affine-test-analysis",
    )
    restored = phx.rom.read_affine_linear_rom(path, model)
    inputs = jnp.asarray([0.75], dtype=jnp.float64)
    np.testing.assert_allclose(
        restored.evaluate(inputs, reconstruct=True).reconstructed_state,
        model.evaluate(inputs, reconstruct=True).reconstructed_state,
    )

    residual = phx.rom.prepare_residual_dual_norm(problem, model)
    stability = phx.rom.ArrayAffineStabilityBound(
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
        family_id=model.family_id,
        error_space_id=model.reduction.test.full_space.space_id,
        support_id=model.reduction.support_id,
        evidence_id="analytic-manufactured-coercivity",
    )
    certification = phx.rom.AffineROMCertification(residual, stability)
    evaluation = model.evaluate(inputs, reconstruct=True)
    certified = phx.rom.certify_affine_rom_evaluation(
        model,
        evaluation,
        inputs,
        certification,
    )

    assert bool(certified.valid)
    np.testing.assert_allclose(certified.residual_dual_norm, 0.0, atol=1e-10)
    np.testing.assert_allclose(certified.absolute_state_error_bound, 0.0, atol=1e-10)
