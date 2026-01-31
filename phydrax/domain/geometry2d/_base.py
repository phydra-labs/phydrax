#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from abc import abstractmethod
from typing import Literal

from jaxtyping import Array

from .._base import _AbstractGeometry


class _AbstractGeometry2D(_AbstractGeometry):
    """Abstract 2D geometry."""

    @property
    @abstractmethod
    def area(self) -> Array:
        raise NotImplementedError

    @property
    def spatial_dim(self) -> Literal[2]:
        return 2

    @property
    def volume(self) -> Array:
        return self.area
