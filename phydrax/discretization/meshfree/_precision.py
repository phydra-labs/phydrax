# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Meshfree precision roles bound to the native precision request/evidence owner.

Roles are geometry (coordinates and charts), coefficient (stencil and operator
weights), fit (local moment/basis factorization), compute (field evaluation),
accumulation, residual, certification (neighbor identity, unisolvence, rank,
and acceptance decisions), communication (halo and migration packets),
checkpoint, and output. Certification is never silently narrower than the
data it certifies, and requested float64 is refused rather than canonicalized
to float32 when the JAX runtime has x64 disabled.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._precision import (
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
    PrecisionResourceAssumptions,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import parse


MeshfreePrecisionDType: TypeAlias = Literal["float32", "float64"]
MeshfreePrecisionRole: TypeAlias = Literal[
    "geometry",
    "coefficient",
    "fit",
    "compute",
    "accumulation",
    "residual",
    "certification",
    "communication",
    "checkpoint",
    "output",
]

_PROVIDER = "phydrax-meshfree"
_ITEMSIZE: dict[MeshfreePrecisionDType, int] = {"float32": 4, "float64": 8}


def _requested(value: DTypeLike, name: str, /) -> MeshfreePrecisionDType:
    # Read the requested name before JAX canonicalization so that float64 is
    # refused, not silently narrowed, when x64 is disabled.
    dtype = parse(np.dtype(value).name, MeshfreePrecisionDType, name)
    if dtype == "float64" and not bool(jax.config.read("jax_enable_x64")):
        raise ValueError(f"{name} requests float64 but JAX x64 is disabled.")
    return dtype


def _widest(*values: MeshfreePrecisionDType) -> MeshfreePrecisionDType:
    return "float64" if "float64" in values else "float32"


def _at_least(
    value: MeshfreePrecisionDType,
    floor: MeshfreePrecisionDType,
    message: str,
    /,
) -> None:
    if _ITEMSIZE[value] < _ITEMSIZE[floor]:
        raise ValueError(message)


@final
class MeshfreePrecisionPolicy(NonTrainableState, StrictModule):
    """Resolved meshfree stage dtypes with fail-closed narrowing rules.

    Defaults follow the data: coefficients, fit, and compute inherit geometry;
    accumulation and residual inherit compute; certification is the widest of
    geometry, fit, and residual; checkpoint is the widest stored state so a
    restart is lossless; communication and output inherit compute.
    """

    geometry_dtype: MeshfreePrecisionDType = eqx.field(static=True)
    coefficient_dtype: MeshfreePrecisionDType = eqx.field(static=True)
    fit_dtype: MeshfreePrecisionDType = eqx.field(static=True)
    compute_dtype: MeshfreePrecisionDType = eqx.field(static=True)
    accumulation_dtype: MeshfreePrecisionDType = eqx.field(static=True)
    residual_dtype: MeshfreePrecisionDType = eqx.field(static=True)
    certification_dtype: MeshfreePrecisionDType = eqx.field(static=True)
    communication_dtype: MeshfreePrecisionDType = eqx.field(static=True)
    checkpoint_dtype: MeshfreePrecisionDType = eqx.field(static=True)
    output_dtype: MeshfreePrecisionDType = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        geometry_dtype: DTypeLike = "float64",
        coefficient_dtype: DTypeLike | None = None,
        fit_dtype: DTypeLike | None = None,
        compute_dtype: DTypeLike | None = None,
        accumulation_dtype: DTypeLike | None = None,
        residual_dtype: DTypeLike | None = None,
        certification_dtype: DTypeLike | None = None,
        communication_dtype: DTypeLike | None = None,
        checkpoint_dtype: DTypeLike | None = None,
        output_dtype: DTypeLike | None = None,
    ) -> None:
        geometry = _requested(geometry_dtype, "geometry_dtype")
        coefficient = (
            geometry
            if coefficient_dtype is None
            else _requested(coefficient_dtype, "coefficient_dtype")
        )
        fit = geometry if fit_dtype is None else _requested(fit_dtype, "fit_dtype")
        compute = (
            coefficient
            if compute_dtype is None
            else _requested(compute_dtype, "compute_dtype")
        )
        accumulation = (
            compute
            if accumulation_dtype is None
            else _requested(accumulation_dtype, "accumulation_dtype")
        )
        residual = (
            accumulation
            if residual_dtype is None
            else _requested(residual_dtype, "residual_dtype")
        )
        certification = (
            _widest(geometry, fit, residual)
            if certification_dtype is None
            else _requested(certification_dtype, "certification_dtype")
        )
        communication = (
            compute
            if communication_dtype is None
            else _requested(communication_dtype, "communication_dtype")
        )
        checkpoint = (
            _widest(geometry, coefficient, compute, accumulation)
            if checkpoint_dtype is None
            else _requested(checkpoint_dtype, "checkpoint_dtype")
        )
        output = (
            compute if output_dtype is None else _requested(output_dtype, "output_dtype")
        )
        _at_least(
            accumulation,
            compute,
            "Accumulation precision cannot be narrower than compute.",
        )
        _at_least(
            residual, compute, "Residual precision cannot be narrower than compute."
        )
        for floor, role in ((geometry, "geometry"), (fit, "fit"), (residual, "residual")):
            _at_least(
                certification,
                floor,
                f"Certification precision cannot be narrower than {role} precision.",
            )
        _at_least(
            communication,
            compute,
            "Communication precision cannot narrow exchanged compute fields.",
        )
        _at_least(
            checkpoint,
            _widest(geometry, coefficient, compute, accumulation),
            "Checkpoint precision cannot narrow restartable state.",
        )
        self.geometry_dtype = geometry
        self.coefficient_dtype = coefficient
        self.fit_dtype = fit
        self.compute_dtype = compute
        self.accumulation_dtype = accumulation
        self.residual_dtype = residual
        self.certification_dtype = certification
        self.communication_dtype = communication
        self.checkpoint_dtype = checkpoint
        self.output_dtype = output
        self.policy_id = canonical_fingerprint(
            {"kind": "meshfree-precision-policy", "roles": dict(self.roles)}
        )

    @property
    def roles(self) -> tuple[tuple[MeshfreePrecisionRole, MeshfreePrecisionDType], ...]:
        return (
            ("geometry", self.geometry_dtype),
            ("coefficient", self.coefficient_dtype),
            ("fit", self.fit_dtype),
            ("compute", self.compute_dtype),
            ("accumulation", self.accumulation_dtype),
            ("residual", self.residual_dtype),
            ("certification", self.certification_dtype),
            ("communication", self.communication_dtype),
            ("checkpoint", self.checkpoint_dtype),
            ("output", self.output_dtype),
        )

    def dtype(self, role: MeshfreePrecisionRole, /) -> MeshfreePrecisionDType:
        role = parse(role, MeshfreePrecisionRole, "role")
        match role:
            case "geometry":
                return self.geometry_dtype
            case "coefficient":
                return self.coefficient_dtype
            case "fit":
                return self.fit_dtype
            case "compute":
                return self.compute_dtype
            case "accumulation":
                return self.accumulation_dtype
            case "residual":
                return self.residual_dtype
            case "certification":
                return self.certification_dtype
            case "communication":
                return self.communication_dtype
            case "checkpoint":
                return self.checkpoint_dtype
            case "output":
                return self.output_dtype
            case _:
                assert_never(role)

    def cast(self, role: MeshfreePrecisionRole, value: ArrayLike, /) -> Array:
        """Convert real data to one role's dtype.

        Certification and residual inputs are never narrowed: a wider input
        would lose exactly the information those roles decide on, so it is
        refused rather than silently rounded.
        """
        target = self.dtype(role)
        array = jnp.asarray(value)
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError(f"Meshfree {role} data must be real floating point.")
        match role:
            case "certification" | "residual":
                if array.dtype.itemsize > _ITEMSIZE[target]:
                    raise ValueError(
                        f"Refusing to downcast {array.dtype.name} {role} data "
                        f"to {target}."
                    )
            case _:
                pass
        return array.astype(target)

    @property
    def request(self) -> PrecisionRequest:
        return PrecisionRequest("meshfree", self._native_roles())

    def _native_roles(self) -> dict[str, str]:
        return {
            "storage": self.geometry_dtype,
            "coefficient": self.coefficient_dtype,
            "basis": self.fit_dtype,
            "factorization": self.fit_dtype,
            "compute": self.compute_dtype,
            "accumulation": self.accumulation_dtype,
            "residual": self.residual_dtype,
            "certification": self.certification_dtype,
            "communication": self.communication_dtype,
            "checkpoint": self.checkpoint_dtype,
            "output": self.output_dtype,
        }

    @property
    def resolution(self) -> PrecisionResolution:
        return PrecisionResolution(self.request, _PROVIDER, self._native_roles())

    @property
    def resource_assumptions(self) -> PrecisionResourceAssumptions:
        return PrecisionResourceAssumptions("meshfree", self._native_roles())

    def evidence(
        self,
        observed: Mapping[MeshfreePrecisionRole, DTypeLike] | None = None,
        /,
    ) -> PrecisionEvidenceEnvelope:
        """Native evidence; observed stage dtypes must equal the resolved ones."""
        if observed is not None:
            for role, dtype in observed.items():
                expected = self.dtype(role)
                if np.dtype(dtype).name != expected:
                    raise ValueError(
                        f"Observed meshfree {role} dtype {np.dtype(dtype).name} "
                        f"differs from resolved {expected}."
                    )
        resolution = self.resolution
        return PrecisionEvidenceEnvelope(resolution, dict(resolution.effective))


__all__ = ["MeshfreePrecisionDType", "MeshfreePrecisionPolicy", "MeshfreePrecisionRole"]
