#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.spin_foam import (
    assess_eprl_semantics,
    assess_su2_bf_identities,
    EPRLVertexPlan,
    evaluate_zero_spin_b4_booster,
    SL2CBoosterReferencePlan,
)
from phydrax.applications.spin_network import (
    prepare_spin_network,
    spin_network_state,
    SpinNetworkEdge,
    SpinNetworkGraphPlan,
)
from phydrax.operators.quantum.lattice import SU2SectorResourcePolicy


def _resources():
    return SU2SectorResourcePolicy(
        maximum_product_dimension=10_000,
        maximum_sector_dimension=1_000,
        maximum_matrix_elements=10_000_000,
    )


def test_finite_su2_bf_identity_portfolio_is_exact():
    evidence = assess_su2_bf_identities(3)
    assert bool(evidence.accepted)
    assert float(evidence.maximum_clebsch_orthogonality_residual) < 1e-12
    assert float(evidence.maximum_recoupling_unitarity_residual) < 1e-12
    assert float(evidence.maximum_tetrahedral_symmetry_residual) < 1e-12
    assert float(evidence.maximum_pentagon_residual) < 1e-12


def test_fixed_theta_spin_network_is_gauss_invariant_with_area_spectrum():
    edges = tuple(SpinNetworkEdge(f"e{index}", "left", "right", 2) for index in range(3))
    plan = SpinNetworkGraphPlan(
        ("left", "right"),
        edges,
        {"left": ("e0", "e1", "e2"), "right": ("e0", "e1", "e2")},
        _resources(),
    )
    prepared = prepare_spin_network(
        plan,
        immirzi_parameter=0.2,
        planck_length_squared=1.0,
    )
    assert prepared.basis_dimension == 1
    assert bool(prepared.evidence.accepted)
    expected_area = 8.0 * np.pi * 0.2 * np.sqrt(2.0)
    np.testing.assert_allclose(prepared.edge_area_eigenvalues, expected_area)
    state = spin_network_state(prepared, jnp.asarray((1.0 + 0.0j,)))
    assert bool(state.normalized)
    assert "no-graph-changing" in state.claim


def test_eprl_plan_pins_boundary_cutoff_and_all_conventions():
    plan = EPRLVertexPlan(
        (0,) * 10,
        (0,) * 5,
        immirzi_parameter=0.2,
        delta_l=1,
        face_amplitude="dimension-2j-plus-1",
        edge_amplitude="unit",
        coherent_phase_convention="condon-shortley-outward-normals",
        normal_frame_id="five-outward-time-gauge-normals",
        quadrature_id="external-b4-declared",
        precision_bits=256,
        maximum_support_tuples=2048,
    )
    evidence = assess_eprl_semantics(plan)
    assert bool(evidence.accepted)
    assert int(evidence.support_tuple_count) == 2**10
    assert "not-an-amplitude" in evidence.claim
    with pytest.raises(ValueError, match="inadmissible"):
        EPRLVertexPlan(
            (0,) * 10,
            (2, 0, 0, 0, 0),
            immirzi_parameter=0.2,
            delta_l=0,
            face_amplitude="dimension",
            edge_amplitude="unit",
            coherent_phase_convention="fixed",
            normal_frame_id="fixed",
            quadrature_id="fixed",
            precision_bits=128,
            maximum_support_tuples=1,
        )


def test_native_zero_spin_sl2c_booster_matches_exact_integral():
    evidence = evaluate_zero_spin_b4_booster(
        SL2CBoosterReferencePlan(
            radial_cutoff=20.0,
            quadrature_order=256,
            tolerance=1e-11,
        )
    )
    assert bool(evidence.accepted)
    np.testing.assert_allclose(evidence.exact_value, np.pi**3 / 120.0)
    assert float(evidence.relative_error) < 1e-10
    assert "no-nonzero-spin" in evidence.claim
