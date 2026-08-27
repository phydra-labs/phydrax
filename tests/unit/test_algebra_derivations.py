import jax.numpy as jnp
import jax.scipy as jsp
import pytest

import phydrax as phx


def _prepare(algebra, **policy_kwargs):
    policy = phx.linalg.AlgebraDerivationPolicy(**policy_kwargs)
    return phx.linalg.prepare_algebra_derivations(
        phx.linalg.plan_algebra_derivations(algebra, policy=policy)
    )


def test_canonical_composition_algebras_have_expected_derivation_dimensions():
    complex_derivations = _prepare(phx.metrix.algebra.ComplexAlgebraSpec())
    quaternion_derivations = _prepare(phx.metrix.algebra.QuaternionAlgebraSpec())
    octonion_derivations = _prepare(phx.metrix.algebra.OctonionAlgebraSpec())

    assert int(complex_derivations.dimension) == 0
    assert int(quaternion_derivations.dimension) == 3
    assert int(octonion_derivations.dimension) == 14
    assert bool(complex_derivations.converged)
    assert bool(quaternion_derivations.converged)
    assert bool(octonion_derivations.converged)
    assert octonion_derivations.maximum_leibniz_residual < 1e-10
    assert octonion_derivations.maximum_unit_fixing_residual < 1e-10
    assert octonion_derivations.maximum_commutator_closure_residual < 1e-10


def test_octonion_derivations_infinitesimally_preserve_the_g2_structure():
    algebra = phx.metrix.algebra.OctonionAlgebraSpec()
    derivations = _prepare(algebra)
    bridge = phx.metrix.OctonionG2Bridge(
        algebra,
        phx.metrix.CoordinateChart("g2", tuple(f"x{i}" for i in range(7))),
    )
    report = phx.metrix.validate_g2_derivations(bridge, derivations)

    assert bool(report.valid)
    assert int(report.derivation_dimension) == 14
    assert report.maximum_form_invariance_residual < 1e-9
    assert report.maximum_metric_skew_residual < 1e-9
    assert report.maximum_scalar_mixing_residual < 1e-9


def test_exponentiated_small_derivation_preserves_octonion_multiplication():
    algebra = phx.metrix.algebra.OctonionAlgebraSpec()
    product = algebra.prepare_product(backend="sparse")
    derivations = _prepare(algebra)
    generator = derivations.subspace.basis[:, 0].reshape((8, 8))
    automorphism = jsp.linalg.expm(0.05 * generator)
    left = jnp.linspace(-0.4, 0.6, 8)
    right = jnp.linspace(0.7, -0.3, 8)

    transformed_product = automorphism @ product(left, right)
    product_of_transforms = product(automorphism @ left, automorphism @ right)

    assert jnp.allclose(transformed_product, product_of_transforms, atol=1e-10)


def test_derivation_projector_returns_a_leibniz_matrix():
    algebra = phx.metrix.algebra.QuaternionAlgebraSpec()
    derivations = _prepare(algebra)
    candidate = jnp.arange(16, dtype=jnp.float64).reshape((4, 4)) / 7.0
    projected = derivations.project(candidate)
    constraint = derivations.plan.constraint.materialize(jnp.float64)

    assert projected.shape == (4, 4)
    assert jnp.linalg.norm(constraint @ projected.reshape((-1,))) < 1e-10


def test_derivation_rank_ambiguity_and_resource_failures_are_explicit():
    algebra = phx.metrix.algebra.QuaternionAlgebraSpec()
    ambiguous = _prepare(algebra, minimum_singular_gap=1e30)

    assert not bool(ambiguous.converged)
    assert int(ambiguous.status) == int(phx.linalg.AlgebraDerivationStatus.AMBIGUOUS_RANK)
    with pytest.raises(ValueError, match="materialization"):
        phx.metrix.algebra.AlgebraDerivationConstraint(
            algebra,
            budget=phx.metrix.algebra.AlgebraSymmetryBudget(maximum_materialized_bytes=1),
        )


def test_derivation_plan_id_records_exact_constraints_and_numeric_policy():
    algebra = phx.metrix.algebra.QuaternionAlgebraSpec()
    first = phx.linalg.plan_algebra_derivations(algebra)
    second = phx.linalg.plan_algebra_derivations(algebra)
    relaxed = phx.linalg.plan_algebra_derivations(
        algebra,
        policy=phx.linalg.AlgebraDerivationPolicy(relative_cutoff=1e-8),
    )

    assert first.constraint.constraint_id == second.constraint.constraint_id
    assert first.plan_id == second.plan_id
    assert first.plan_id != relaxed.plan_id
