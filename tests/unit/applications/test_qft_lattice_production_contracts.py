#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.applications.lattice_field._production_contracts import (
    ContinuumExtrapolationPlan,
    LatticeEnsemblePlan,
    LatticeRegulator,
    LatticeTheoryPoint,
)
from phydrax.discretization import TensorTopology
from phydrax.discretization._lattice_boundary import LatticeBoundaryPhasePlan


def _theory_point(size, spacing):
    boundary = LatticeBoundaryPhasePlan(
        TensorTopology(("x",), (size,), periodic=(True,)),
        jnp.ones((1,), dtype=complex),
    )
    regulator = LatticeRegulator(
        boundary,
        spacing,
        gauge_topology_id=f"cycle-{size}",
    )
    return LatticeTheoryPoint(
        regulator,
        action_id="su2-wilson",
        bare_parameters={"beta": 2.2 + 0.01 * size},
        trajectory_id="fixed-physical-line",
        scale_setting_id="declared-reference-scale",
    )


def test_ensemble_admission_retains_finite_statistics_and_qualification_ids():
    point = _theory_point(4, 0.25)
    plan = LatticeEnsemblePlan(
        point,
        ("plaquette", "polyakov"),
        sample_count=4,
        qualification_ids=("chain-diagnostics",),
    )
    values = jnp.asarray([[0.4, 0.1], [0.5, -0.1], [0.6, 0.2], [0.5, 0.0]])
    evidence = plan.prepare().admit(
        values,
        evidence_ids=("chain-diagnostics", "run-manifest"),
    )

    assert evidence.successful
    assert evidence.valid_sample_count == 4
    assert jnp.allclose(evidence.means, jnp.mean(values, axis=0))
    assert jnp.allclose(evidence.covariance, jnp.cov(values, rowvar=False))


def test_continuum_fit_is_fixed_volume_and_uses_native_weighted_solve():
    points = (
        _theory_point(4, 0.25),
        _theory_point(8, 0.125),
        _theory_point(16, 0.0625),
    )
    prepared = ContinuumExtrapolationPlan(
        points,
        observable_name="gradient-flow-scale",
        powers=(0, 2),
    ).prepare()
    scaled_spacing = prepared.lattice_spacings / jnp.max(prepared.lattice_spacings)
    values = 2.0 + 3.0 * scaled_spacing**2
    result = prepared.fit(values, jnp.full((3,), 0.1))

    assert result.evidence.successful
    assert jnp.allclose(result.continuum_value, 2.0, atol=1e-5)
    assert jnp.allclose(result.fitted_values, values, atol=1e-5)
