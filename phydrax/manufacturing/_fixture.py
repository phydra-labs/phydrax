#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FixtureState:
    fixture_id: str
    engaged: bool
    contact_region_id: str

    def __post_init__(self):
        if not self.fixture_id or not self.contact_region_id:
            raise ValueError("Fixture identity and contact region are required.")


__all__ = ["FixtureState"]
