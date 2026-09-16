#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from math import pi

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.linalg import (
    DifferentiationPolicy,
    MatrixFunctionPolicy,
    ShiftedSolvePolicy,
    ShiftedSolveResourcePolicy,
)
from phydrax.operators.quantum import FermionModeOrder
from phydrax.operators.quantum._response import (
    FiniteTemperatureResponsePlan,
    QuantumSectorProbe,
    ZeroTemperatureResponsePlan,
)
from phydrax.operators.quantum.lattice import (
    FixedCardinalityFermionBasis,
    LocalOperatorPlan,
    LocalSpacePlan,
    prepare_quantum_lattice,
    QuantumLatticeResourcePolicy,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
    QuantumSectorOperator,
    SectorBasisResourcePolicy,
    SectorChargeMap,
)
from phydrax.solver._quantum_response import (
    finite_temperature_response,
    zero_temperature_response,
)
from phydrax.solver._thermal_pure_quantum import (
    prepare_thermal_pure_quantum,
    thermal_pure_quantum,
    ThermalPureQuantumPlan,
)


def _response_case():
    order = FermionModeOrder(("a", "b"))
    spaces = tuple(LocalSpacePlan.fermion(label, label) for label in order.labels)
    create_matrix = np.asarray(((0.0, 0.0), (1.0, 0.0)))
    annihilate_matrix = create_matrix.T
    number_terms = []
    for energy, space in zip((1.0, 2.0), spaces, strict=True):
        number_terms.append(
            QuantumLatticeTerm(
                (
                    LocalOperatorPlan(space, "create", create_matrix, (1,)),
                    LocalOperatorPlan(space, "annihilate", annihilate_matrix, (-1,)),
                ),
                coefficient=energy,
                label=f"energy:{space.site_id}",
            )
        )
    resources = QuantumLatticeResourcePolicy(
        maximum_terms=8,
        maximum_factors_per_term=4,
        maximum_branches_per_input=64,
        maximum_sector_dimension=16,
        maximum_workspace_bytes=100_000,
    )
    hamiltonian = prepare_quantum_lattice(
        QuantumLatticeSpecification(spaces, number_terms, fermion_mode_order=order),
        resources,
    )
    basis_resources = SectorBasisResourcePolicy(
        maximum_dimension=16, maximum_table_bytes=10_000
    )
    vacuum = FixedCardinalityFermionBasis(order, 0, resources=basis_resources)
    one_particle = FixedCardinalityFermionBasis(order, 1, resources=basis_resources)
    source = QuantumSectorOperator(hamiltonian, SectorChargeMap(vacuum, vacuum, 0))
    target = QuantumSectorOperator(
        hamiltonian, SectorChargeMap(one_particle, one_particle, 0)
    )
    creation = prepare_quantum_lattice(
        QuantumLatticeSpecification(
            spaces,
            (
                QuantumLatticeTerm(
                    (LocalOperatorPlan(spaces[0], "create-a", create_matrix, (1,)),),
                    label="create-a-probe",
                ),
            ),
            fermion_mode_order=order,
        ),
        resources,
    )
    probe = QuantumSectorProbe(
        QuantumSectorOperator(creation, SectorChargeMap(vacuum, one_particle, 1)),
        probe_id="create-a",
    )
    return source, target, probe


def _shifted_policy():
    return ShiftedSolvePolicy(
        "lanczos",
        max_dimension=2,
        relative_tolerance=1e-10,
        differentiation="none",
        resources=ShiftedSolveResourcePolicy(
            max_matvec_count=16,
            max_storage_bytes=100_000,
            max_workspace_bytes=100_000,
        ),
    )


def test_zero_temperature_shifted_response_has_exact_moments_and_positive_line():
    source, target, probe = _response_case()
    frequencies = jnp.asarray((-1.0, 0.0, 1.0, 2.0, 3.0))
    eta = 0.2
    plan = ZeroTemperatureResponsePlan(
        frequencies,
        eta,
        moment_count=3,
        shifted_solve=_shifted_policy(),
        maximum_frequency_points=8,
        maximum_result_bytes=100_000,
        required_window=(-1.0, 3.0),
    )
    result = zero_temperature_response(
        plan,
        source,
        target,
        probe,
        jnp.asarray((1.0 + 0.0j,)),
        0.0,
    )
    expected = eta / (pi * ((np.asarray(frequencies) - 1.0) ** 2 + eta**2))
    np.testing.assert_allclose(result.spectral, expected, atol=1e-9)
    np.testing.assert_allclose(result.evidence.moments, (1.0, 1.0, 1.0))
    assert bool(result.evidence.nonnegative)
    assert bool(result.evidence.valid)


def _tpq(hamiltonian, beta):
    plan = ThermalPureQuantumPlan(
        beta,
        probe_count=2,
        observable_count=0,
        matrix_function=MatrixFunctionPolicy(
            "lanczos",
            max_dimension=2,
            error_tolerance=1e-10,
            differentiation=DifferentiationPolicy("none"),
        ),
        maximum_retained_bytes=100_000,
        maximum_workspace_bytes=100_000,
    )
    return thermal_pure_quantum(
        prepare_thermal_pure_quantum(plan, hamiltonian), (), key=jr.key(4)
    )


def test_finite_temperature_response_records_source_target_kms_and_positivity():
    source, target, probe = _response_case()
    plan = FiniteTemperatureResponsePlan(
        (-pi, -pi / 2.0, 0.0, pi / 2.0, pi),
        (-1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0, 1.0, 1.0),
        moment_count=2,
        matrix_function=MatrixFunctionPolicy(
            "lanczos",
            max_dimension=2,
            error_tolerance=1e-10,
            differentiation=DifferentiationPolicy("none"),
        ),
        maximum_result_bytes=100_000,
        maximum_workspace_bytes=100_000,
        positivity_tolerance=1e-9,
        kms_tolerance=1e-9,
    )
    result = finite_temperature_response(
        plan, source, target, probe, _tpq(source, 0.0), _tpq(target, 0.0)
    )
    np.testing.assert_allclose(result.forward_spectrum, (0.0, 0.0, 1.0), atol=1e-9)
    np.testing.assert_allclose(result.reverse_spectrum, (0.5, 0.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(result.evidence.forward_moments, (1.0, 1.0), atol=1e-9)
    np.testing.assert_allclose(result.evidence.partition_ratio, 0.5, atol=1e-9)
    assert bool(result.evidence.positivity_satisfied)
    assert bool(result.evidence.kms_satisfied)
    assert bool(result.evidence.valid)


def test_response_plans_refuse_missing_resources_uncovered_or_asymmetric_windows():
    with pytest.raises(ValueError, match="explicit shifted-solve"):
        ZeroTemperatureResponsePlan(
            (0.0, 1.0),
            0.1,
            moment_count=1,
            shifted_solve=ShiftedSolvePolicy(
                "lanczos", max_dimension=2, differentiation="none"
            ),
            maximum_frequency_points=4,
            maximum_result_bytes=10_000,
        )
    with pytest.raises(ValueError, match="cover"):
        ZeroTemperatureResponsePlan(
            (0.0, 1.0),
            0.1,
            moment_count=1,
            shifted_solve=_shifted_policy(),
            maximum_frequency_points=4,
            maximum_result_bytes=10_000,
            required_window=(-1.0, 1.0),
        )
    with pytest.raises(ValueError, match="symmetric"):
        FiniteTemperatureResponsePlan(
            (-1.0, 0.0, 2.0),
            (-1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            moment_count=1,
            matrix_function=MatrixFunctionPolicy(
                "lanczos",
                max_dimension=2,
                differentiation=DifferentiationPolicy("none"),
            ),
            maximum_result_bytes=10_000,
            maximum_workspace_bytes=10_000,
        )
