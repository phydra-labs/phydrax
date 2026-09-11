#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_evolution_sensitivity_certificate_checks_jvp_and_transpose_duality():
    matrix = jnp.asarray([[1.2, -0.3], [0.4, 0.8]])
    system = phx.dynamics.DiscreteSystem(
        lambda coordinate, state, args: matrix @ state + args,
        state_layout=phx.dynamics.StateLayout((2,)),
        system_id="sensitivity-certificate-map",
    )
    evolution = phx.dynamics.DiscreteEvolution(system)
    policy = phx.dynamics.EvolutionSensitivityPolicy(
        (1.0e-3, 5.0e-4),
        relative_tolerance=1.0e-9,
        absolute_tolerance=1.0e-11,
    )

    evidence = phx.dynamics.certify_evolution_sensitivity(
        evolution,
        jnp.asarray([0.2, -0.4]),
        jnp.asarray([0.7, 0.1]),
        jnp.asarray([-0.5, 0.9]),
        0,
        1,
        args=jnp.asarray([0.3, -0.2]),
        policy=policy,
        reference_evolution=evolution,
    )

    assert bool(evidence.valid)
    assert int(evidence.status) == int(phx.dynamics.EvolutionSensitivityStatus.SUCCESS)
    assert evidence.reference_evolution_id == evolution.evolution_id
    np.testing.assert_allclose(evidence.jvp_defects, 0.0, atol=1e-12)
    np.testing.assert_allclose(evidence.duality_defect, 0.0, atol=1e-12)
    np.testing.assert_allclose(evidence.reference_tangent_defect, 0.0, atol=1e-12)
