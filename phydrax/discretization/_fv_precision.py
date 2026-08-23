#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._precision import (
    precision_dtype_name,
    precision_itemsize,
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
    PrecisionResourceAssumptions,
    real_precision_dtype_name,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState


PrecisionDType: TypeAlias = Literal["float32", "float64"]


def _finite_precision_dtype(value: Any, /) -> PrecisionDType:
    name = real_precision_dtype_name(value)
    if name not in ("float32", "float64"):
        raise ValueError("Finite-volume precision dtypes must be float32 or float64.")
    return name


class FiniteVolumePrecisionPolicy(StrictModule, NonTrainableState):
    """State, reconstruction, flux, reduction, output, and checkpoint precision."""

    storage_dtype: PrecisionDType = eqx.field(static=True)
    reconstruction_dtype: PrecisionDType = eqx.field(static=True)
    flux_dtype: PrecisionDType = eqx.field(static=True)
    reduction_dtype: PrecisionDType = eqx.field(static=True)
    output_dtype: PrecisionDType = eqx.field(static=True)
    checkpoint_dtype: PrecisionDType = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        storage_dtype: PrecisionDType = "float64",
        /,
        *,
        reconstruction_dtype: PrecisionDType | None = None,
        flux_dtype: PrecisionDType | None = None,
        reduction_dtype: PrecisionDType | None = None,
        output_dtype: PrecisionDType | None = None,
        checkpoint_dtype: PrecisionDType | None = None,
    ):
        storage = _finite_precision_dtype(storage_dtype)
        reconstruction = _finite_precision_dtype(
            storage if reconstruction_dtype is None else reconstruction_dtype
        )
        flux = _finite_precision_dtype(storage if flux_dtype is None else flux_dtype)
        reduction = _finite_precision_dtype(
            storage if reduction_dtype is None else reduction_dtype
        )
        output = _finite_precision_dtype(
            storage if output_dtype is None else output_dtype
        )
        checkpoint = _finite_precision_dtype(
            storage if checkpoint_dtype is None else checkpoint_dtype
        )
        if precision_itemsize(reduction) < max(
            precision_itemsize(reconstruction),
            precision_itemsize(flux),
        ):
            raise ValueError(
                "Finite-volume reduction precision cannot be narrower than "
                "reconstruction or flux precision."
            )
        request = PrecisionRequest(
            "finite-volume",
            {
                "storage": storage,
                "compute": flux,
                "accumulation": reduction,
                "certification": reduction,
                "output": output,
                "checkpoint": checkpoint,
            },
        )
        reconstruction_request = PrecisionRequest(
            "finite-volume-reconstruction",
            {"compute": reconstruction},
        )
        self.storage_dtype = storage
        self.reconstruction_dtype = reconstruction
        self.flux_dtype = flux
        self.reduction_dtype = reduction
        self.output_dtype = output
        self.checkpoint_dtype = checkpoint
        self.policy_id = canonical_fingerprint(
            {
                "kind": "finite-volume-precision",
                "request": request.request_id,
                "reconstruction_request": reconstruction_request.request_id,
            }
        )

    @property
    def request(self) -> PrecisionRequest:
        return PrecisionRequest(
            "finite-volume",
            {
                "storage": self.storage_dtype,
                "compute": self.flux_dtype,
                "accumulation": self.reduction_dtype,
                "certification": self.reduction_dtype,
                "output": self.output_dtype,
                "checkpoint": self.checkpoint_dtype,
            },
        )

    @property
    def reconstruction_request(self) -> PrecisionRequest:
        return PrecisionRequest(
            "finite-volume-reconstruction",
            {"compute": self.reconstruction_dtype},
        )

    def validate_state(self, value: Any, /) -> None:
        observed = precision_dtype_name(jnp.asarray(value).dtype)
        if observed != self.storage_dtype:
            raise TypeError(
                f"Finite-volume state dtype {observed} does not match "
                f"{self.storage_dtype}."
            )

    def storage(self, value: Any, /):
        return jnp.asarray(value, dtype=self.storage_dtype)

    def reconstruction(self, value: Any, /):
        return jnp.asarray(value, dtype=self.reconstruction_dtype)

    def flux(self, value: Any, /):
        return jnp.asarray(value, dtype=self.flux_dtype)

    def reduction(self, value: Any, /):
        array = jnp.asarray(value)
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            return array
        return array.astype(self.reduction_dtype)

    def decision(self, value: Any, /):
        return jnp.asarray(value, dtype=self.reduction_dtype)

    def output(self, value: Any, /):
        return jnp.asarray(value, dtype=self.output_dtype)

    def checkpoint(self, value: Any, /):
        return jnp.asarray(value, dtype=self.checkpoint_dtype)

    def numpy_dtype(self, role: str, /):
        values = {
            "storage": self.storage_dtype,
            "reconstruction": self.reconstruction_dtype,
            "flux": self.flux_dtype,
            "reduction": self.reduction_dtype,
            "output": self.output_dtype,
            "checkpoint": self.checkpoint_dtype,
        }
        if role not in values:
            raise ValueError(f"Unknown finite-volume precision role {role!r}.")
        return np.dtype(values[role])

    def evidence(self) -> PrecisionEvidenceEnvelope:
        reconstruction_resolution = PrecisionResolution(
            self.reconstruction_request,
            "phydrax-finite-volume-reconstruction",
            {"compute": self.reconstruction_dtype},
        )
        reconstruction_evidence = PrecisionEvidenceEnvelope(
            reconstruction_resolution,
            dict(reconstruction_resolution.effective),
        )
        resolution = PrecisionResolution(
            self.request,
            "phydrax-finite-volume",
            {
                "storage": self.storage_dtype,
                "compute": self.flux_dtype,
                "accumulation": self.reduction_dtype,
                "certification": self.reduction_dtype,
                "output": self.output_dtype,
                "checkpoint": self.checkpoint_dtype,
            },
        )
        return PrecisionEvidenceEnvelope(
            resolution,
            dict(resolution.effective),
            children={"reconstruction": reconstruction_evidence},
        )

    def resource_assumptions(self) -> PrecisionResourceAssumptions:
        return PrecisionResourceAssumptions(
            "finite-volume",
            dict(self.evidence().observed),
        )


__all__ = ["FiniteVolumePrecisionPolicy", "PrecisionDType"]
