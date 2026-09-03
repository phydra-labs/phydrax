import math

from tools.incompressible_ou_qualification import qualification


def test_ou_forced_etdrk_qualification_closes_every_declared_gate():
    evidence = qualification(sample_count=512, stationary_steps=48)

    assert evidence["passed"]
    assert "schema_version" not in evidence
    assert evidence["gates"] == {
        "coefficient_covariance_and_subdivision": True,
        "basis_spectrum_and_isotropy": True,
        "logical_randomness": True,
        "fluid_temporal_refinement_and_restart": True,
        "stationary_energy_budget_and_block_uncertainty": True,
        "fixed_realization_jvp_vjp": True,
    }
    assert evidence["coefficient_covariance_and_subdivision"]["sample_count"] == 512
    assert evidence["logical_randomness"]["maximum_sharding_difference"] == 0.0
    assert (
        evidence["fluid_temporal_refinement_and_restart"]["restart_forcing_error"] == 0.0
    )
    assert evidence["stationary_energy_budget_and_block_uncertainty"]["block_count"] >= 2
    assert math.isfinite(evidence["fixed_realization_jvp_vjp"]["jvp"])
    assert math.isfinite(evidence["fixed_realization_jvp_vjp"]["vjp"])
