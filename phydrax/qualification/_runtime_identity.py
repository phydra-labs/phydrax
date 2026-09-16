#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier


class QualificationRuntimeIdentity(StrictModule, NonTrainableState):
    build_id: str = eqx.field(static=True)
    environment_id: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    topology: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    identity_id: str = eqx.field(static=True)

    def __init__(
        self,
        build_id: str,
        environment_id: str,
        backend: str,
        topology: str,
        precision: str,
        /,
    ):
        values = {
            "build_id": canonical_identifier(build_id, "build_id"),
            "environment_id": canonical_identifier(environment_id, "environment_id"),
            "backend": canonical_identifier(backend, "backend"),
            "topology": canonical_identifier(topology, "topology"),
            "precision": canonical_identifier(precision, "precision"),
        }
        self.build_id = values["build_id"]
        self.environment_id = values["environment_id"]
        self.backend = values["backend"]
        self.topology = values["topology"]
        self.precision = values["precision"]
        self.identity_id = canonical_fingerprint(
            {"kind": "qualification-runtime", **values}
        )

    def compatible(
        self,
        other: QualificationRuntimeIdentity,
        /,
        *,
        allowed_differences: tuple[str, ...] = (),
    ) -> bool:
        if not isinstance(other, QualificationRuntimeIdentity):
            return False
        allowed = set(allowed_differences)
        left = self.to_record()
        right = other.to_record()
        fields = ("build_id", "environment_id", "backend", "topology", "precision")
        return all(field in allowed or left[field] == right[field] for field in fields)

    def to_record(self) -> Mapping[str, str]:
        return {
            "build_id": self.build_id,
            "environment_id": self.environment_id,
            "backend": self.backend,
            "topology": self.topology,
            "precision": self.precision,
            "identity_id": self.identity_id,
        }


__all__ = ["QualificationRuntimeIdentity"]
