#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.applications.reacting_flow._transport_runtime import (
    TransportPropertyReusePlan,
)
from phydrax.equations._gas_transport_properties import (
    KineticTheoryGasTransportPlan,
    LogPolynomialGasTransportPlan,
    ReferencePowerLawGasTransportPlan,
)


def _reference():
    return ReferencePowerLawGasTransportPlan(
        jnp.asarray(((0.0, 1.0e-5), (1.0e-5, 0.0))),
        jnp.asarray((1.0e-5, 2.0e-5)),
        jnp.asarray((0.02, 0.03)),
    )


def test_log_polynomial_properties_enforce_support_and_error_certificate():
    viscosity = np.log(np.asarray((1.0e-5, 2.0e-5)))[:, None]
    conductivity = np.log(np.asarray((0.02, 0.03)))[:, None]
    diffusion = np.zeros((2, 2, 1))
    diffusion[0, 1, 0] = np.log(1.0e-5)
    diffusion[1, 0, 0] = np.log(1.0e-5)
    plan = LogPolynomialGasTransportPlan(
        viscosity,
        conductivity,
        diffusion,
        (300.0, 2000.0),
        reference_pressure=101325.0,
        relative_error_bounds=(0.01, 0.02, 0.03),
        reference_id="synthetic-certified-fit",
    )

    evaluated = plan.evaluate(jnp.asarray(1000.0), jnp.asarray(202650.0))
    np.testing.assert_allclose(evaluated.species_viscosity, (1.0e-5, 2.0e-5))
    np.testing.assert_allclose(evaluated.binary_diffusion_coefficients[0, 1], 0.5e-5)
    assert bool(evaluated.successful)
    assert float(evaluated.diffusion_relative_error_bound) == 0.03
    assert not bool(plan.evaluate(250.0, 101325.0).supported)


def test_kinetic_theory_route_returns_positive_symmetric_properties():
    plan = KineticTheoryGasTransportPlan(
        (2.016, 31.998),
        (2.92, 3.46),
        (38.0, 107.4),
        (2.5, 2.5),
        (200.0, 3000.0),
    )
    evaluated = plan.evaluate(1000.0, 101325.0)

    assert bool(evaluated.successful)
    assert jnp.all(evaluated.species_viscosity > 0.0)
    assert jnp.all(evaluated.species_thermal_conductivity > 0.0)
    np.testing.assert_allclose(
        evaluated.binary_diffusion_coefficients[0, 1],
        evaluated.binary_diffusion_coefficients[1, 0],
    )


def test_reuse_state_commits_atomically_and_refreshes_after_bound():
    plan = TransportPropertyReusePlan(
        _reference(),
        temperature_bounds=(200.0, 3000.0),
        pressure_bounds=(1.0e4, 1.0e7),
        logarithmic_sensitivities=((0.7, 0.0), (0.7, 0.0), (1.75, 1.0)),
        maximum_relative_errors=(0.02, 0.02, 0.04),
        maximum_reuse_count=2,
    )
    accepted = plan.initialize(1000.0, 101325.0)
    candidate = plan.propose(accepted, 1001.0, 101325.0)

    assert bool(candidate.viscosity_reused)
    assert bool(candidate.conductivity_reused)
    assert bool(candidate.diffusion_reused)
    rejected = plan.commit(accepted, candidate, False)
    np.testing.assert_array_equal(rejected.temperature, accepted.temperature)
    np.testing.assert_array_equal(rejected.reuse_count, accepted.reuse_count)

    committed = plan.commit(accepted, candidate, True)
    assert float(committed.temperature) == 1001.0
    assert int(committed.reuse_count) == 1
    refreshed = plan.propose(committed, 1200.0, 101325.0)
    assert not bool(refreshed.viscosity_reused)
    assert not bool(refreshed.diffusion_reused)
