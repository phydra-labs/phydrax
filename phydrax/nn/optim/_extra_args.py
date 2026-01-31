#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import optax
from optax._src import base


class GradientTransformationExtraArgs(optax.GradientTransformationExtraArgs):
    def __new__(
        cls,
        init: base.TransformInitFn,
        update: base.TransformUpdateExtraArgsFn,
    ) -> "GradientTransformationExtraArgs":
        return super().__new__(cls, init, update)
