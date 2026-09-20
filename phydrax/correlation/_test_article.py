#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SensorChannel:
    channel_id: str
    coordinate_id: str
    sample_rate_hz: float
    unit: str

    def __post_init__(self):
        if (
            not self.channel_id
            or not self.coordinate_id
            or self.sample_rate_hz <= 0
            or not self.unit
        ):
            raise ValueError("Sensor channel invalid.")


@dataclass(frozen=True, slots=True)
class TestArticle:
    article_id: str
    channels: tuple[SensorChannel, ...]


__all__ = ["SensorChannel", "TestArticle"]
