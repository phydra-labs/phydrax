#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import phydrax as phx


def test_advanced_biophysics_uses_canonical_public_owners():
    assert (
        phx.discretization.DynamicPairRelationPlan
        is phx.discretization.particle.DynamicPairRelationPlan
    )
    assert (
        phx.applications.cellular_mechanics.BiomembranePlan.__module__
        == "phydrax.applications.cellular_mechanics._membrane"
    )
    assert (
        phx.applications.cellular_mechanics.VertexTissuePlan.__module__
        == "phydrax.applications.cellular_mechanics._vertex_tissue"
    )
    assert (
        phx.applications.cellular_mechanics.ChromatinDynamicsPlan.__module__
        == "phydrax.applications.cellular_mechanics._active_polymers"
    )
    assert (
        phx.applications.electrophysiology.CableSolverPlan.__module__
        == "phydrax.applications.electrophysiology._cable"
    )
    assert (
        phx.applications.systems_biology.StoichiometricNetworkPlan.__module__
        == "phydrax.applications.systems_biology._network"
    )
    assert (
        phx.stochastic.path_sampling.TPSPlan.__module__
        == "phydrax.stochastic.path_sampling._samplers"
    )
    assert phx.atomistic.AlchemicalTransformationPlan.__module__.endswith("._alchemical")
    assert phx.atomistic.PolarizationSolverPlan.__module__.endswith("._polarization")
    assert phx.observation.FluorescenceCorrelationPlan.__module__ == "phydrax.observation"
    assert phx.qualification.nernst_equilibrium_potential.__module__.endswith(
        "._biophysics"
    )
