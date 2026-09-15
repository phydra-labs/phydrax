#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..artifacts import DerivativeEvidence
from ..qualification import CapabilityProfile
from ._identity import ReproducibilityGrade


class HEPProviderBinding(StrictModule, NonTrainableState):
    """Pinned HEP provider configuration bound to released generic support."""

    profile: CapabilityProfile
    differentiation: DerivativeEvidence
    configuration_checksum: str = eqx.field(static=True)
    data_checksums: tuple[str, ...] = eqx.field(static=True)
    input_profile_ids: tuple[str, ...] = eqx.field(static=True)
    output_profile_ids: tuple[str, ...] = eqx.field(static=True)
    unit_ids: tuple[str, ...] = eqx.field(static=True)
    frame_ids: tuple[str, ...] = eqx.field(static=True)
    devices: tuple[str, ...] = eqx.field(static=True)
    dtypes: tuple[str, ...] = eqx.field(static=True)
    side_effects: tuple[str, ...] = eqx.field(static=True)
    license_ids: tuple[str, ...] = eqx.field(static=True)
    reproducibility: ReproducibilityGrade = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        profile: CapabilityProfile,
        differentiation: DerivativeEvidence,
        /,
        *,
        configuration_checksum: str,
        data_checksums: Sequence[str] = (),
        input_profile_ids: Sequence[str],
        output_profile_ids: Sequence[str],
        unit_ids: Sequence[str],
        frame_ids: Sequence[str] = (),
        devices: Sequence[str],
        dtypes: Sequence[str],
        side_effects: Sequence[str],
        license_ids: Sequence[str],
        reproducibility: ReproducibilityGrade,
    ):
        if not isinstance(profile, CapabilityProfile):
            raise TypeError("profile must be CapabilityProfile.")
        if not isinstance(differentiation, DerivativeEvidence):
            raise TypeError("differentiation must be DerivativeEvidence.")
        if not isinstance(reproducibility, ReproducibilityGrade):
            raise TypeError("reproducibility must be ReproducibilityGrade.")

        def identifiers(
            values: Sequence[str], name: str, *, required: bool = True
        ) -> tuple[str, ...]:
            result = tuple(str(value).strip() for value in values)
            if (
                (required and not result)
                or any(not value for value in result)
                or len(set(result)) != len(result)
            ):
                raise ValueError(f"{name} must contain distinct non-empty values.")
            return tuple(sorted(result))

        configuration = str(configuration_checksum).strip()
        if not configuration:
            raise ValueError("configuration_checksum must be non-empty.")
        allowed_side_effects = frozenset(
            {
                "none",
                "filesystem-read",
                "filesystem-write",
                "network-read",
                "network-write",
                "subprocess",
            }
        )
        data = identifiers(data_checksums, "data_checksums", required=False)
        inputs = identifiers(input_profile_ids, "input_profile_ids")
        outputs = identifiers(output_profile_ids, "output_profile_ids")
        units = identifiers(unit_ids, "unit_ids")
        frames = identifiers(frame_ids, "frame_ids", required=False)
        devices_ = identifiers(devices, "devices")
        dtypes_ = identifiers(dtypes, "dtypes")
        effects = identifiers(side_effects, "side_effects")
        licenses = identifiers(license_ids, "license_ids")
        if not set(effects) <= allowed_side_effects:
            raise ValueError("side_effects contains an unsupported effect.")
        if "none" in effects and len(effects) != 1:
            raise ValueError("'none' cannot be combined with another side effect.")
        self.profile = profile
        self.differentiation = differentiation
        self.configuration_checksum = configuration
        self.data_checksums = data
        self.input_profile_ids = inputs
        self.output_profile_ids = outputs
        self.unit_ids = units
        self.frame_ids = frames
        self.devices = devices_
        self.dtypes = dtypes_
        self.side_effects = effects
        self.license_ids = licenses
        self.reproducibility = reproducibility
        self.binding_id = canonical_fingerprint(
            {
                "kind": "hep-provider-binding",
                "profile": profile.profile_id,
                "differentiation": differentiation.evidence_id,
                "configuration": configuration,
                "data": list(data),
                "inputs": list(inputs),
                "outputs": list(outputs),
                "units": list(units),
                "frames": list(frames),
                "devices": list(devices_),
                "dtypes": list(dtypes_),
                "side_effects": list(effects),
                "licenses": list(licenses),
                "reproducibility": reproducibility.value,
            }
        )

    @property
    def provider_id(self) -> str:
        return self.profile.provider

    @property
    def provider_release(self) -> str:
        return self.profile.version

    def supports(self, capability: str, /) -> bool:
        return str(capability) == self.profile.capability


__all__ = ["HEPProviderBinding"]
