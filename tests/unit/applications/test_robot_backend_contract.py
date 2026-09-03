#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.applications.robotics._backend import (
    ROBOTICS_OPERATIONS,
    RoboticsBackendProfile,
    RoboticsIndexEntry,
    RoboticsOperationCapability,
    RoboticsOperationEvidence,
    RoboticsOperationRequirement,
    RoboticsOperationStatus,
    RoboticsProjection,
    RoboticsProjectionMap,
)
from phydrax.applications.robotics._mjx import (
    mjx_availability,
    MJX_JAX_PROFILE,
    MJX_WARP_PROFILE,
)
from phydrax.backends._types import BackendUnavailableError


def _profile():
    return RoboticsBackendProfile(
        backend="contract-probe",
        implementation="native-probe",
        operations=(
            RoboticsOperationCapability(
                "step",
                supported=True,
                implementation="probe.step",
                devices=("cpu",),
                dtypes=("float32",),
                differentiability="conditional",
                solver_exclusions=("pgs",),
                contact_exclusions=("sdf",),
            ),
            RoboticsOperationCapability(
                "jvp",
                supported=False,
                implementation="probe.jvp",
                reason="the probe has no derivative implementation",
            ),
        ),
    )


def test_capability_negotiation_accepts_only_declared_operation_conditions():
    profile = _profile()
    accepted = profile.negotiate(
        (
            RoboticsOperationRequirement(
                "step",
                device="cpu",
                dtype=jnp.float32,
                minimum_differentiability="conditional",
                solver="newton",
                contact_feature="sphere-capsule",
            ),
        )
    )

    assert accepted.accepted
    assert accepted.status == RoboticsOperationStatus.SUCCESS
    accepted.require()

    rejected = profile.negotiate(
        (
            RoboticsOperationRequirement("step", device="gpu"),
            RoboticsOperationRequirement("step", dtype="float64"),
            RoboticsOperationRequirement("step", solver="PGS"),
            RoboticsOperationRequirement("step", contact_feature="SDF"),
            RoboticsOperationRequirement(
                "step", minimum_differentiability="guaranteed"
            ),
            RoboticsOperationRequirement("jvp"),
            RoboticsOperationRequirement("sensors"),
        )
    )

    assert not rejected.accepted
    assert rejected.status == RoboticsOperationStatus.UNSUPPORTED
    assert len(rejected.rejections) == 7
    with pytest.raises(BackendUnavailableError, match="device 'gpu'"):
        rejected.require()
    with pytest.raises(BackendUnavailableError, match="explicitly excluded"):
        profile.require((RoboticsOperationRequirement("step", solver="pgs"),))


def test_profiles_are_per_operation_and_never_claim_universal_differentiability():
    assert tuple(capability.operation for capability in MJX_JAX_PROFILE.operations) == (
        ROBOTICS_OPERATIONS
    )
    assert tuple(capability.operation for capability in MJX_WARP_PROFILE.operations) == (
        ROBOTICS_OPERATIONS
    )
    assert all(
        capability.differentiability != "guaranteed"
        for capability in MJX_JAX_PROFILE.operations
    )
    assert all(
        capability.differentiability == "none"
        for capability in MJX_WARP_PROFILE.operations
    )
    assert MJX_JAX_PROFILE.capability("step").solver_exclusions == ("pgs", "noslip")
    assert "sdf" in MJX_JAX_PROFILE.capability("contact").contact_exclusions
    assert not MJX_WARP_PROFILE.capability("jvp").supported
    with pytest.raises(BackendUnavailableError, match="automatic differentiation"):
        MJX_WARP_PROFILE.require((RoboticsOperationRequirement("vjp"),))


def test_projection_maps_are_complete_stable_and_immutable():
    index_map = RoboticsProjectionMap(
        "qpos",
        4,
        (
            RoboticsIndexEntry("base", 0, 3),
            RoboticsIndexEntry("hinge", 3, 4),
        ),
    )
    projection = RoboticsProjection(jnp.arange(8).reshape(2, 4), index_map)

    assert index_map.names == ("base", "hinge")
    assert index_map.name_to_range == (("base", (0, 3)), ("hinge", (3, 4)))
    assert index_map.entry("base").indices == (0, 1, 2)
    assert projection.values.shape == (2, 4)
    with pytest.raises(AttributeError):
        index_map.size = 5
    with pytest.raises(ValueError, match="contiguous"):
        RoboticsProjectionMap(
            "qvel",
            3,
            (RoboticsIndexEntry("gap", 1, 3),),
        )
    with pytest.raises(ValueError, match="axis size"):
        RoboticsProjection(jnp.zeros((3,)), index_map)


def test_operation_evidence_has_canonical_status_and_explicit_freshness():
    evidence = RoboticsOperationEvidence(
        status=RoboticsOperationStatus.NONFINITE,
        finite=False,
        backend="probe",
        operation="step",
        implementation="probe.step",
        device="cpu",
        dtype="float32",
        observation_freshness=("pre-step", "post-step-refreshed"),
        detail="candidate rejected and prior opaque state retained",
    )

    assert int(evidence.status) == int(RoboticsOperationStatus.NONFINITE)
    assert not bool(evidence.successful)
    assert evidence.observation_freshness == (
        "pre-step",
        "post-step-refreshed",
    )
    with pytest.raises(AttributeError):
        evidence.detail = "changed"


def test_missing_mjx_provider_uses_shared_unavailable_contract(monkeypatch):
    monkeypatch.setattr(
        "phydrax.backends._availability.importlib.util.find_spec",
        lambda module: None,
    )

    availability = mjx_availability()

    assert not availability.available
    assert availability.backend == "mjx-jax"
    assert "not installed" in availability.reason
    with pytest.raises(BackendUnavailableError, match="mujoco-mjx"):
        availability.require("robotics.step")
