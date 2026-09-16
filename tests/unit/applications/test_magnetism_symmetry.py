import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.magnetism import (
    compile_magnetic_symmetry_constraints,
    MagneticSymmetryRepresentationPlan,
    magnetism_candidate_profiles,
    magnetism_support_tuples,
)
from phydrax.applications.superconductivity import (
    superconductivity_candidate_profiles,
    superconductivity_support_tuples,
)
from phydrax.metrix.clifford import CliffordAlgebraSpec, FiniteMetricIsometryGroup


def test_antiunitary_magnetic_symmetry_compiles_real_linear_invariants():
    algebra = CliffordAlgebraSpec((1, 1, 1))
    group = FiniteMetricIsometryGroup(algebra, np.stack((np.eye(3), -np.eye(3))))
    plan = MagneticSymmetryRepresentationPlan(
        group,
        [False, True],
        [[0, 1], [1, 0]],
        [[0], [0]],
        np.zeros((2, 1, 3), dtype=int),
        np.ones((2, 1, 1), dtype=complex),
    )
    certificate = compile_magnetic_symmetry_constraints(plan)
    assert int(certificate.rank) == 1
    assert int(certificate.nullity) == 1
    projected = certificate.project(jnp.asarray([1.0 + 2.0j]))
    assert jnp.allclose(projected, jnp.asarray([1.0 + 0.0j]))
    assert certificate.invariance_residual(projected) < 1.0e-12
    # Under inversion plus time reversal, spin/B flip, momentum is fixed, and
    # the time-even axial DMI vector remains invariant.
    assert jnp.allclose(plan.operations[1].axial_spin_matrix, -jnp.eye(3))
    assert jnp.allclose(plan.operations[1].magnetic_field_matrix, -jnp.eye(3))
    assert jnp.allclose(plan.operations[1].momentum_matrix, jnp.eye(3))
    assert jnp.allclose(plan.operations[1].dmi_matrix, jnp.eye(3))


def test_antiunitary_flags_must_form_group_homomorphism():
    algebra = CliffordAlgebraSpec((1, 1, 1))
    group = FiniteMetricIsometryGroup(algebra, np.stack((np.eye(3), -np.eye(3))))
    with pytest.raises(ValueError, match="Z2 homomorphism|identity"):
        MagneticSymmetryRepresentationPlan(
            group,
            [True, False],
            [[0], [0]],
            [[0], [0]],
            np.zeros((2, 1, 3), dtype=int),
            np.ones((2, 1, 1), dtype=complex),
        )


def test_candidate_profiles_keep_maturity_out_of_support_coordinates():
    support = (*magnetism_support_tuples(), *superconductivity_support_tuples())
    profiles = (
        *magnetism_candidate_profiles(),
        *superconductivity_candidate_profiles(),
    )
    assert len(support) == len(profiles)
    assert all("candidate" not in item.capability for item in support)
    assert all(not profile.released for profile in profiles)
    assert [profile.support_tuples[0].support_tuple_id for profile in profiles] == [
        item.support_tuple_id for item in support
    ]
