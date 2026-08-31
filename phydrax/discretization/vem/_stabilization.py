#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._projection import VirtualElementProjectionData


class VirtualElementStabilizationPolicy(StrictModule, NonTrainableState):
    kind: str = eqx.field(static=True)
    minimum_scale: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, kind: str = "dofi_dofi", /, *, minimum_scale: float = 1.0e-14):
        kind_ = str(kind)
        scale = float(minimum_scale)
        if kind_ not in ("dofi_dofi", "projected"):
            raise ValueError("VEM stabilization must be dofi_dofi or projected.")
        if scale <= 0.0:
            raise ValueError("minimum_scale must be positive.")
        self.kind = kind_
        self.minimum_scale = scale
        self.policy_id = canonical_fingerprint(
            {
                "kind": "virtual-element-stabilization",
                "policy": kind_,
                "minimum_scale": scale,
            }
        )


class VirtualElementStabilizationEvidence(StrictModule):
    polynomial_leakage: Array
    symmetry_error: Array
    minimum_kernel_eigenvalue: Array
    maximum_kernel_eigenvalue: Array
    scale: Array


class StabilizedVirtualElementTensor(StrictModule):
    consistent: Array
    stabilization: Array
    combined: Array
    evidence: VirtualElementStabilizationEvidence


def stabilize_virtual_element_tensor(
    projection: VirtualElementProjectionData,
    consistent: Array,
    policy: VirtualElementStabilizationPolicy,
    /,
    *,
    projector: str,
) -> StabilizedVirtualElementTensor:
    if not isinstance(projection, VirtualElementProjectionData):
        raise TypeError("projection must be VirtualElementProjectionData.")
    if not isinstance(policy, VirtualElementStabilizationPolicy):
        raise TypeError("policy must be VirtualElementStabilizationPolicy.")
    matrix = jnp.asarray(consistent)
    dof_projector = (
        projection.h1_dof_projector
        if projector == "h1"
        else projection.l2_dof_projector
        if projector == "l2"
        else None
    )
    if dof_projector is None:
        raise ValueError("projector must be h1 or l2.")
    local = dof_projector.shape[-1]
    residual = jnp.eye(local, dtype=matrix.dtype)[None] - dof_projector
    trace = jnp.trace(matrix, axis1=-2, axis2=-1)
    rank = max(projection.basis.feature_count - (1 if projector == "h1" else 0), 1)
    scale = jnp.maximum(jnp.abs(trace) / rank, policy.minimum_scale)
    if policy.kind == "dofi_dofi":
        stabilization = scale[:, None, None] * oe.contract(
            "cki,ckj->cij", residual, residual
        )
    else:
        diagonal = jnp.abs(jnp.diagonal(matrix, axis1=-2, axis2=-1))
        floor = scale[:, None] / local
        weights = jnp.maximum(diagonal, floor)
        stabilization = oe.contract("cki,ck,ckj->cij", residual, weights, residual)
    combined = matrix + stabilization
    leakage = oe.contract("cij,cja->cia", stabilization, projection.dof_matrix)
    symmetry = stabilization - jnp.swapaxes(stabilization, -1, -2)
    eigenvalues = jnp.linalg.eigvalsh(
        0.5 * (stabilization + jnp.swapaxes(stabilization, -1, -2))
    )
    evidence = VirtualElementStabilizationEvidence(
        polynomial_leakage=jnp.max(jnp.abs(leakage), axis=(-2, -1)),
        symmetry_error=jnp.max(jnp.abs(symmetry), axis=(-2, -1)),
        minimum_kernel_eigenvalue=jnp.min(eigenvalues, axis=-1),
        maximum_kernel_eigenvalue=jnp.max(eigenvalues, axis=-1),
        scale=scale,
    )
    return StabilizedVirtualElementTensor(
        consistent=matrix,
        stabilization=stabilization,
        combined=combined,
        evidence=evidence,
    )


__all__ = [
    "StabilizedVirtualElementTensor",
    "VirtualElementStabilizationEvidence",
    "VirtualElementStabilizationPolicy",
    "stabilize_virtual_element_tensor",
]
