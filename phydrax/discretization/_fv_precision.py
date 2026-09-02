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


PrecisionDType: TypeAlias = Literal["float16", "bfloat16", "float32", "float64"]


def _finite_precision_dtype(value: Any, /) -> PrecisionDType:
    return real_precision_dtype_name(value)


class FiniteVolumePrecisionPolicy(StrictModule, NonTrainableState):
    """State, reconstruction, flux, reduction, output, and checkpoint precision."""

    storage_dtype: PrecisionDType = eqx.field(static=True)
    reconstruction_dtype: PrecisionDType = eqx.field(static=True)
    flux_dtype: PrecisionDType = eqx.field(static=True)
    reduction_dtype: PrecisionDType = eqx.field(static=True)
    output_dtype: PrecisionDType = eqx.field(static=True)
    checkpoint_dtype: PrecisionDType = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    resolution_id: str | None = eqx.field(static=True)
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
        resolution: PrecisionResolution | None = None,
    ):
        storage = _finite_precision_dtype(storage_dtype)
        reconstruction = _finite_precision_dtype(
            storage if reconstruction_dtype is None else reconstruction_dtype
        )
        default_compute: PrecisionDType = (
            "float32" if precision_itemsize(storage) < 4 else storage
        )
        flux = _finite_precision_dtype(
            default_compute if flux_dtype is None else flux_dtype
        )
        reduction = _finite_precision_dtype(
            default_compute if reduction_dtype is None else reduction_dtype
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
        if precision_itemsize(storage) < 4 and resolution is None:
            raise ValueError(
                "Sub-float32 finite-volume storage requires provider precision "
                "resolution evidence."
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
        if resolution is not None:
            if not isinstance(resolution, PrecisionResolution):
                raise TypeError("resolution must be PrecisionResolution or None.")
            effective = dict(resolution.effective)
            expected = dict(request.requested)
            if (
                resolution.request_id != request.request_id
                or resolution.domain != request.domain
                or effective != expected
            ):
                raise ValueError(
                    "Finite-volume precision resolution does not exactly match "
                    "the requested roles."
                )
        self.storage_dtype = storage
        self.reconstruction_dtype = reconstruction
        self.flux_dtype = flux
        self.reduction_dtype = reduction
        self.output_dtype = output
        self.checkpoint_dtype = checkpoint
        self.provider = (
            "phydrax-finite-volume" if resolution is None else resolution.provider
        )
        self.resolution_id = None if resolution is None else resolution.resolution_id
        self.policy_id = canonical_fingerprint(
            {
                "kind": "finite-volume-precision",
                "request": request.request_id,
                "reconstruction_request": reconstruction_request.request_id,
                "provider": self.provider,
                "resolution": self.resolution_id,
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

    @property
    def subfloat_storage(self) -> bool:
        return precision_itemsize(self.storage_dtype) < 4

    def quantize_and_validate(
        self,
        value: Any,
        admissible: Any,
        /,
        *,
        wet_mask: Any | None = None,
    ):
        """Round-trip through storage and reject changed admissibility/topology."""
        if not callable(admissible):
            raise TypeError("admissible must be callable.")
        candidate = self.storage(value)
        decision = self.decision(candidate)
        valid = jnp.asarray(admissible(decision))
        if valid.dtype != jnp.bool_:
            raise TypeError("admissible must return a Boolean array.")
        failed = jnp.any(~jnp.isfinite(decision)) | jnp.any(~valid)
        if wet_mask is not None:
            expected_wet = jnp.asarray(wet_mask)
            observed_wet = jnp.asarray(admissible(decision))
            if (
                expected_wet.dtype != jnp.bool_
                or expected_wet.shape != observed_wet.shape
            ):
                raise ValueError("wet_mask must match the admissibility mask.")
            failed = failed | jnp.any(observed_wet != expected_wet)
        return eqx.error_if(
            candidate,
            failed,
            "Quantized finite-volume state is inadmissible or changes its wet mask.",
        )

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
            self.provider,
            {"compute": self.reconstruction_dtype},
        )
        reconstruction_evidence = PrecisionEvidenceEnvelope(
            reconstruction_resolution,
            dict(reconstruction_resolution.effective),
        )
        resolution = PrecisionResolution(
            self.request,
            self.provider,
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
