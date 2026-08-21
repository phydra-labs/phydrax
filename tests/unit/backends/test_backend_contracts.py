#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _capabilities():
    return phx.backends.BackendCapabilities(
        backend="contract-probe",
        problem_kinds=("linear.system",),
        execution="host",
        host_only=True,
        supports_matrix_free=False,
        supports_assembled=True,
        coordinate_dtypes=("float64",),
    )


def test_backend_probe_is_lazy_and_reports_missing_requirement_exactly():
    availability = phx.backends.probe_backend(
        _capabilities(),
        module="__phydrax_backend_that_does_not_exist__",
        requirement="missing-test-provider>=1",
        distributions=("__phydrax_missing_distribution__",),
    )

    assert not availability.available
    assert availability.backend == "contract-probe"
    assert availability.versions == ()
    assert "not installed" in availability.reason
    with pytest.raises(phx.backends.BackendUnavailableError) as error:
        availability.require("linear.system")
    assert "contract-probe" in str(error.value)
    assert "missing-test-provider>=1" in str(error.value)
    assert "linear.system" in str(error.value)


def test_backend_capability_rejection_is_distinct_from_provider_availability():
    availability = phx.backends.BackendAvailability(
        capabilities=_capabilities(),
        available=True,
        requirement="installed-provider",
        reason="provider module imported successfully",
        versions=(("installed-provider", "1.2.3"),),
    )

    availability.require("linear.system")
    with pytest.raises(phx.backends.BackendUnavailableError, match="does not declare"):
        availability.require("eigen.general")


def test_backend_transfer_evidence_preserves_array_scalars():
    evidence = phx.backends.BackendTransferEvidence(
        host_to_device_bytes=128,
        device_to_host_bytes=64,
        synchronization_count=2,
    )

    assert evidence.host_to_device_bytes.dtype == jnp.int64
    assert int(evidence.host_to_device_bytes) == 128
    assert int(evidence.device_to_host_bytes) == 64
    assert int(evidence.synchronization_count) == 2
