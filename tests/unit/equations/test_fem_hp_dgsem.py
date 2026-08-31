#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.fem import (
    FiniteElementMortarMetricData,
    serial_finite_element_mortar_plan,
)
from phydrax.equations.fem import (
    certify_dgsem_mortar_compatibility,
    dgsem_mortar_flux_ledger,
    DGSEMNonconformingMortarPlan,
)


def _certified_mortar():
    left_nodes = np.linspace(-1.0, 1.0, 4)
    right_nodes = np.linspace(-1.0, 1.0, 3)
    quadrature, weights = np.polynomial.legendre.leggauss(6)
    coordinates = np.stack((quadrature, 0.1 * quadrature**2), axis=1)
    measure = np.sqrt(1.0 + (0.2 * quadrature) ** 2)
    mortar = serial_finite_element_mortar_plan(
        left_nodes,
        right_nodes,
        left_nodes,
        quadrature,
        weights,
        declared_reproduction_degree=2,
        left_physical_coordinates=coordinates,
        right_physical_coordinates=coordinates,
        coordinate_measure=measure,
        interface_id="entropy-compatible-hp-mortar",
    )
    tangent = jnp.stack(
        (jnp.ones_like(jnp.asarray(quadrature)), 0.2 * quadrature), axis=1
    )
    owner_normal = jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=1)
    metric = FiniteElementMortarMetricData(
        coordinates,
        jnp.asarray(weights * measure),
        owner_normal,
        -owner_normal,
    )
    certificate = certify_dgsem_mortar_compatibility(mortar, metric)
    return mortar, metric, certificate


def test_entropy_compatible_mortar_accumulates_conservative_ledgers():
    mortar, metric, certificate = _certified_mortar()
    assert certificate.passed
    flux = 1.0 + mortar.quadrature_points[:, 0] ** 2
    entropy_flux = 0.5 * flux**2
    ledger = dgsem_mortar_flux_ledger(
        mortar,
        metric,
        certificate,
        flux,
        entropy_flux=entropy_flux,
    )

    np.testing.assert_allclose(
        np.asarray(ledger.conservation_residual), 0.0, atol=2.0e-13
    )
    np.testing.assert_allclose(
        np.asarray(ledger.entropy_flux),
        np.asarray(mortar.integrated_flux(entropy_flux, metric)),
        atol=2.0e-13,
    )
    plan = DGSEMNonconformingMortarPlan((mortar,), (metric,), (certificate,))
    planned = plan.ledgers((flux,), entropy_fluxes=(entropy_flux,))
    np.testing.assert_allclose(
        np.asarray(planned[0].conservation_residual),
        0.0,
        atol=2.0e-13,
    )


def test_uncertified_nonconforming_dgsem_is_rejected():
    mortar, metric, _ = _certified_mortar()
    failed = certify_dgsem_mortar_compatibility(
        mortar,
        metric,
        entropy_error=1.0,
        tolerance=1.0e-10,
    )
    assert not failed.passed
    with pytest.raises(ValueError, match="passing mortar certificate"):
        dgsem_mortar_flux_ledger(mortar, metric, failed, jnp.ones((6,)))
    with pytest.raises(ValueError, match="passing evidence"):
        DGSEMNonconformingMortarPlan((mortar,), (metric,), (failed,))
