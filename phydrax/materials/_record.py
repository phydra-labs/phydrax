#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._fingerprint import canonical_fingerprint


@dataclass(frozen=True, slots=True)
class MaterialRecord:
    material_id: str
    revision: str
    composition: tuple[tuple[str, float], ...]
    source_ids: tuple[str, ...] = ()

    @classmethod
    def create(cls, material_id, revision, composition, /, *, source_ids=()):
        return cls(
            str(material_id).strip(),
            str(revision).strip(),
            tuple(sorted((str(k), float(v)) for k, v in composition.items())),
            tuple(sorted(str(v) for v in source_ids)),
        )

    def __post_init__(self):
        component_ids = tuple(name for name, _ in self.composition)
        if (
            not self.material_id
            or not self.revision
            or not self.composition
            or any(not name for name in component_ids)
            or len(set(component_ids)) != len(component_ids)
            or self.composition != tuple(sorted(self.composition))
            or any(not source for source in self.source_ids)
            or len(set(self.source_ids)) != len(self.source_ids)
            or self.source_ids != tuple(sorted(self.source_ids))
        ):
            raise ValueError(
                "Material identity, components, and sources must be unique, nonempty, and canonical."
            )
        if any(
            not np.isfinite(v) or v < 0 for _, v in self.composition
        ) or not np.isclose(sum(v for _, v in self.composition), 1.0, atol=1e-12, rtol=0):
            raise ValueError(
                "Material composition must be finite, non-negative, and normalized."
            )

    @property
    def record_id(self):
        return canonical_fingerprint(
            {
                "kind": "material-record",
                "material_id": self.material_id,
                "revision": self.revision,
                "composition": self.composition,
                "source_ids": self.source_ids,
            }
        )


__all__ = ["MaterialRecord"]
