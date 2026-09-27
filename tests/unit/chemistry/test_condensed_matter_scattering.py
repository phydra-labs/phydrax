from typing import Any

import numpy as np

from phydrax import units
from phydrax.chemistry.spectroscopy._scattering import (
    DynamicStructureFactorPlan,
    ElasticNeutronScatteringPlan,
    ElasticXRayScatteringPlan,
    XRayFormFactorRequest,
    XRayFormFactorResult,
)


def _elastic_plan(plan_type: Any, positions: Any) -> Any:
    return plan_type(
        positions,
        np.zeros((2, 3, 3)),
        [1, 0],
        "two-atom-cell",
        q_capacity=2,
        atom_capacity=2,
        symmetry_tolerance=1.0e-12,
    )


def test_condensed_matter_scattering_scenario_1() -> None:
    q = np.asarray([[2.0 * np.pi, 0.0, 0.0], [-2.0 * np.pi, 0.0, 0.0]])
    positions = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    shifted = positions + np.asarray([0.137, -0.23, 0.41])
    request = XRayFormFactorRequest(q, ("A", "B"), "two-atom-cell")
    factors = XRayFormFactorResult(
        # ty: ignore[invalid-argument-type]
        [[1.0, 1.0], [1.0, 1.0]],
        request,
        "xray-factor-provider",
        ("sha256:xray",),
        True,
    )
    xray = _elastic_plan(ElasticXRayScatteringPlan, positions).evaluate(factors)
    shifted_xray = _elastic_plan(ElasticXRayScatteringPlan, shifted).evaluate(factors)
    neutron = _elastic_plan(ElasticNeutronScatteringPlan, positions).evaluate(
        q,
        [1.0, 1.0],
        "coherent-length-table",
        ("sha256:neutron",),
    )

    np.testing.assert_allclose(xray.intensities, shifted_xray.intensities, atol=1.0e-28)
    np.testing.assert_allclose(xray.intensities, xray.intensities[::-1], atol=1.0e-28)
    np.testing.assert_allclose(neutron.intensities, 0.0, atol=1.0e-28)
    assert bool(xray.evidence.successful)
    assert bool(neutron.evidence.successful)
    beta = 1.3
    energies = np.asarray([0.0, 1.0])
    probabilities = np.exp(-beta * energies)
    probabilities /= probabilities.sum()
    operator = np.asarray([[[0.0, 1.0], [1.0, 0.0]]], dtype="complex128")
    result = DynamicStructureFactorPlan(
        beta=beta,
        state_capacity=2,
        q_capacity=1,
        transition_capacity=4,
        residual_tolerance=1.0e-12,
    ).evaluate(
        energies,
        probabilities,
        operator,
        # ty: ignore[invalid-argument-type]
        [0],
        units.ELECTRONVOLT,
        units.ONE,
        "exact-two-level",
    )

    assert bool(result.evidence.successful)
    np.testing.assert_allclose(
        np.sum(result.raw_response.values) / (2.0 * np.pi),
        1.0,
        atol=1.0e-12,
    )
    negative = np.where(np.asarray(result.raw_response.coordinates) == -1.0)[0][0]
    positive = np.where(np.asarray(result.raw_response.coordinates) == 1.0)[0][0]
    np.testing.assert_allclose(
        result.raw_response.values[0, negative],
        np.exp(-beta) * result.raw_response.values[0, positive],
        atol=1.0e-12,
    )
