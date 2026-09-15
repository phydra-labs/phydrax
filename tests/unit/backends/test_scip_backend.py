#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

import phydrax as phx


def test_scip_backend_declares_optional_host_milp_boundary():
    capabilities = phx.backends.SCIP_CAPABILITIES
    availability = phx.backends.scip_availability()

    assert capabilities.host_only
    assert capabilities.requires_explicit_release
    assert capabilities.supports("optimization.mixed-integer-linear-program")
    assert not capabilities.supports_plan_prepare_solve_refresh
    assert availability.requirement == "install phydrax[scip] (pyscipopt==6.2.1)"


def test_unavailable_scip_fails_at_explicit_prepare_boundary():
    availability = phx.backends.scip_availability()
    if availability.available:
        pytest.skip("PySCIPOpt is installed in this environment.")

    with pytest.raises(phx.backends.BackendUnavailableError):
        phx.backends.prepare_scip()


def test_scip_plan_rejects_invalid_resource_settings():
    with pytest.raises(ValueError, match="positive"):
        phx.backends.SCIPPlan(maximum_nodes=0)
    with pytest.raises(ValueError, match="positive"):
        phx.backends.SCIPPlan(time_limit=0.0)
