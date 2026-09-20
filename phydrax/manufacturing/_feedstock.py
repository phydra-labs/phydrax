#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeedstockSpecification:
    feedstock_id: str
    material_id: str
    form: str
    mass_rate_kg_s: float

    def __post_init__(self):
        if (
            not self.feedstock_id
            or not self.material_id
            or self.form not in ("powder", "wire", "droplet", "filament")
            or self.mass_rate_kg_s < 0
        ):
            raise ValueError("Feedstock specification invalid.")


__all__ = ["FeedstockSpecification"]
