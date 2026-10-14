#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Execution capability profiles bind the exact dynamic bundle and dependencies."""

from __future__ import annotations

import equinox as eqx
import pytest

from phydrax import config as pxconfig
from phydrax.applications.nuclear_fuel import (
    CoolantBoundaryKind,
    FuelModelBundle,
    nuclear_fuel_execution_profile,
)
from phydrax.qualification import CapabilityProfile
from tests._support.nuclear_fuel import synthetic_model_bundle


@pytest.fixture(scope="module")
def bundle() -> FuelModelBundle:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        return synthetic_model_bundle(mode_count=4)


def _profile(
    bundle: FuelModelBundle,
    /,
    *,
    coolant: CoolantBoundaryKind = "prescribed-wall-temperature",
    runtime_id: str = "runtime-contract-fixture",
    channel_profile_id: str | None = None,
) -> CapabilityProfile:
    # Isolated contract-fixture identities, not observed runtime evidence.
    return nuclear_fuel_execution_profile(
        bundle,
        reduction="layered-radial",
        coolant=coolant,
        axial_interaction="frictionless",
        dtype="float64",
        prepared_id="prepared-contract-fixture",
        runtime_id=runtime_id,
        resource_evidence_id="resource-contract-fixture",
        precision_evidence_id="precision-contract-fixture",
        channel_profile_id=channel_profile_id,
    )


def test_execution_support_changes_with_dynamic_parameters(
    bundle: FuelModelBundle,
) -> None:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        changed = eqx.tree_at(
            lambda value: value.bore_emissivity.value,
            bundle,
            bundle.bore_emissivity.value * 1.1,
        )
        original, modified = _profile(bundle), _profile(changed)
    assert not original.supports(modified.support_tuples[0])


def test_execution_profile_declares_applicability_and_is_unreleased(
    bundle: FuelModelBundle,
) -> None:
    profile = _profile(bundle)
    assert not profile.released
    assert (
        dict(profile.support_tuples[0].attributes)["applicability_policy"]
        == "equation-reproduction"
    )


def test_coupled_channel_profile_requires_its_exact_channel_profile(
    bundle: FuelModelBundle,
) -> None:
    with pytest.raises(ValueError, match="exact admitted channel profile"):
        _profile(bundle, coolant="coupled-channel")


def test_coupled_channel_profile_depends_on_its_channel_profile(
    bundle: FuelModelBundle,
) -> None:
    profile = _profile(
        bundle,
        coolant="coupled-channel",
        channel_profile_id="exact-channel-contract-fixture",
    )
    assert "exact-channel-contract-fixture" in profile.dependencies
    assert not profile.released


def test_execution_support_refuses_non_string_runtime_identifier(
    bundle: FuelModelBundle,
) -> None:
    with pytest.raises(TypeError, match="runtime_id must be a string"):
        _profile(bundle, runtime_id=7)  # ty: ignore[invalid-argument-type]


def test_execution_support_refuses_non_string_channel_identifier(
    bundle: FuelModelBundle,
) -> None:
    with pytest.raises(TypeError, match="channel_profile_id must be a string"):
        _profile(
            bundle,
            coolant="coupled-channel",
            channel_profile_id=7,  # ty: ignore[invalid-argument-type]
        )
