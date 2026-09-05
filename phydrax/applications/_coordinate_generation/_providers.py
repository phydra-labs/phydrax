# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Shared source admission without application identity or provider execution."""

from __future__ import annotations

from dataclasses import dataclass

from ..._fingerprint import canonical_fingerprint
from ...artifacts import ScientificArtifactEnvelope
from ...qualification import ReferenceArtifactManifest
from ._native import require_coordinate_rights


@dataclass(frozen=True)
class CoordinateProviderProvenance:
    """Actual provider lineage and explicit egress admission, not a license proxy.

    Learned outputs require model-weight rights separately from output/data rights.
    Prepared MSA/templates belong in input_artifact_ids with their own input_rights.
    None of these declarations causes execution or grants missing authorization.
    """

    provider_id: str
    output_rights: tuple[ReferenceArtifactManifest, ...]
    weight_rights: tuple[ReferenceArtifactManifest, ...] = ()
    input_rights: tuple[ReferenceArtifactManifest, ...] = ()
    code_rights: tuple[ReferenceArtifactManifest, ...] = ()
    input_artifact_ids: tuple[str, ...] = ()
    learned_model: bool = False
    egress_destination: str | None = None
    authorized_egress_destinations: tuple[str, ...] = ()

    def admit(
        self,
        *,
        commercial_use=False,
        training_use=False,
        redistribution=False,
        export=False,
    ):
        if not self.provider_id or self.provider_id != self.provider_id.strip():
            raise ValueError("Provider identity must be explicit and canonical.")
        if self.learned_model and (
            not self.weight_rights or not self.input_artifact_ids or not self.input_rights
        ):
            raise PermissionError(
                "Learned provider outputs require separate weight and prepared-input rights/provenance."
            )
        if any(not value for value in self.input_artifact_ids):
            raise ValueError("Prepared input artifact identities must be nonempty.")
        if self.egress_destination is not None:
            if self.egress_destination not in self.authorized_egress_destinations:
                raise PermissionError(
                    "Provider data egress destination was not explicitly authorized."
                )
            require_coordinate_rights(
                self.input_rights, commercial_use=commercial_use, export=True
            )
        require_coordinate_rights(
            self.output_rights,
            commercial_use=commercial_use,
            training_use=training_use,
            redistribution=redistribution,
            export=export,
        )
        inherited = (
            *self.output_rights,
            *self.weight_rights,
            *self.input_rights,
            *self.code_rights,
        )
        require_coordinate_rights(
            inherited,
            commercial_use=commercial_use,
            training_use=training_use,
            redistribution=redistribution,
            export=export,
        )
        # Retain every parent restriction even when several refer to the same file.
        return inherited

    def require_sources(self, sources):
        """Bind raw output envelopes to admitted bytes, not unrelated license labels."""
        admitted = {(item.checksum, item.license_id) for item in self.output_rights}
        for source in sources:
            if (
                not isinstance(source, ScientificArtifactEnvelope)
                or source.status != "complete"
            ):
                raise ValueError(
                    "Provider sources must be complete native artifact envelopes."
                )
            if (source.content_digest, source.license_id) not in admitted:
                raise ValueError(
                    "Raw provider output digest/license lacks a matching admitted output manifest."
                )

    def fingerprint(self):
        return canonical_fingerprint(
            {
                "provider": self.provider_id,
                "output_rights": [r.manifest_id for r in self.output_rights],
                "weight_rights": [r.manifest_id for r in self.weight_rights],
                "input_rights": [r.manifest_id for r in self.input_rights],
                "code_rights": [r.manifest_id for r in self.code_rights],
                "inputs": self.input_artifact_ids,
                "learned": self.learned_model,
                "destination": self.egress_destination,
                "authorized_destinations": self.authorized_egress_destinations,
            }
        )
