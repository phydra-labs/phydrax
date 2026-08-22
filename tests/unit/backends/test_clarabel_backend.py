#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

import phydrax as phx


def test_clarabel_capabilities_are_lazy_and_host_only():
    availability = phx.backends.clarabel_availability()
    capabilities = availability.capabilities

    assert capabilities.backend == "clarabel"
    assert capabilities.host_only
    assert capabilities.supports("optimization.linear-program")
    assert capabilities.supports("optimization.quadratic-program")
    assert capabilities.supports("optimization.conic-program")
    assert availability.requirement == ("install phydrax[clarabel] (clarabel==0.11.1)")


def test_missing_clarabel_raises_selected_backend_error():
    availability = phx.backends.clarabel_availability()
    if availability.available:
        pytest.skip("Clarabel is installed in this environment.")
    with pytest.raises(phx.backends.BackendUnavailableError, match="clarabel"):
        phx.backends.prepare_clarabel()
