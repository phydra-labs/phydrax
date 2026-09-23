#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class SensorChannel:
    channel_id: str
    coordinate_id: str
    sample_rate_hz: float
    unit: str

    def __post_init__(self):
        if (
            not isinstance(self.channel_id, str)
            or not self.channel_id
            or not isinstance(self.coordinate_id, str)
            or not self.coordinate_id
            or not isfinite(self.sample_rate_hz)
            or self.sample_rate_hz <= 0
            or not isinstance(self.unit, str)
            or not self.unit
        ):
            raise ValueError("Sensor channel invalid.")


@dataclass(frozen=True, slots=True)
class TestArticle:
    article_id: str
    channels: tuple[SensorChannel, ...]

    def __post_init__(self):
        if not isinstance(self.article_id, str) or not self.article_id:
            raise ValueError("Test article identifier must be nonempty.")
        if not self.channels or any(
            not isinstance(channel, SensorChannel) for channel in self.channels
        ):
            raise ValueError("Test article requires sensor channels.")
        channel_ids = tuple(channel.channel_id for channel in self.channels)
        coordinate_ids = tuple(channel.coordinate_id for channel in self.channels)
        if (
            len(set(channel_ids)) != len(channel_ids)
            or len(set(coordinate_ids)) != len(coordinate_ids)
            or len({channel.unit for channel in self.channels}) != 1
            or len({channel.sample_rate_hz for channel in self.channels}) != 1
        ):
            raise ValueError(
                "Test article channels must have unique identities/coordinates and common units/sample rate."
            )


__all__ = ["SensorChannel", "TestArticle"]
