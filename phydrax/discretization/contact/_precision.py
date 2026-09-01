#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ContactPrecisionPolicy(StrictModule, NonTrainableState):
    """Precision roles for contact geometry, accumulation, and certification."""

    geometry_dtype: np.dtype = eqx.field(static=True)
    accumulation_dtype: np.dtype = eqx.field(static=True)
    certification_dtype: np.dtype = eqx.field(static=True)
    output_dtype: np.dtype = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        geometry_dtype: Any = np.float64,
        accumulation_dtype: Any = np.float64,
        certification_dtype: Any = np.float64,
        output_dtype: Any = np.float64,
    ):
        geometry = np.dtype(geometry_dtype)
        accumulation = np.dtype(accumulation_dtype)
        certification = np.dtype(certification_dtype)
        output = np.dtype(output_dtype)
        for name, dtype in (
            ("geometry_dtype", geometry),
            ("accumulation_dtype", accumulation),
            ("certification_dtype", certification),
            ("output_dtype", output),
        ):
            if not np.issubdtype(dtype, np.floating):
                raise TypeError(f"{name} must be a real floating dtype.")
        if certification.itemsize < geometry.itemsize:
            raise ValueError(
                "Contact certification precision cannot be lower than geometry precision."
            )
        self.geometry_dtype = geometry
        self.accumulation_dtype = accumulation
        self.certification_dtype = certification
        self.output_dtype = output
        self.policy_id = canonical_fingerprint(
            {
                "kind": "contact-precision-policy",
                "geometry": geometry.str,
                "accumulation": accumulation.str,
                "certification": certification.str,
                "output": output.str,
            }
        )

    @property
    def conservative_ccd_supported(self) -> bool:
        return (
            self.geometry_dtype.itemsize >= 8 and self.certification_dtype.itemsize >= 8
        )

    def geometry(self, value):
        return jnp.asarray(value, dtype=self.geometry_dtype)

    def accumulation(self, value):
        return jnp.asarray(value, dtype=self.accumulation_dtype)

    def certification(self, value):
        return jnp.asarray(value, dtype=self.certification_dtype)

    def output(self, value):
        return jnp.asarray(value, dtype=self.output_dtype)


__all__ = ["ContactPrecisionPolicy"]
