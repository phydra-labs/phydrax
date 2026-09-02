#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._precision import precision_itemsize
from .._strict import StrictModule
from ..metrix import ComplexStiefelManifold, StiefelManifold


class MERAResourcePolicy(StrictModule):
    maximum_sites: int = eqx.field(static=True)
    maximum_state_elements: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    isometry_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_sites: int = 32,
        maximum_state_elements: int = 10_000_000,
        maximum_workspace_bytes: int = 2**31,
        isometry_tolerance: float = 1e-6,
    ):
        sites = int(maximum_sites)
        elements = int(maximum_state_elements)
        workspace = int(maximum_workspace_bytes)
        tolerance = float(isometry_tolerance)
        if sites < 1 or elements < 1 or workspace < 1 or tolerance <= 0.0:
            raise ValueError("MERA resource limits and tolerance must be positive.")
        self.maximum_sites = sites
        self.maximum_state_elements = elements
        self.maximum_workspace_bytes = workspace
        self.isometry_tolerance = tolerance
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mera-resource-policy",
                "maximum_sites": sites,
                "maximum_state_elements": elements,
                "maximum_workspace_bytes": workspace,
                "isometry_tolerance": tolerance,
            }
        )


class BinaryMERA(StrictModule):
    """Homogeneous-per-layer finite binary MERA, ordered fine to coarse."""

    isometries: tuple[Array, ...]
    disentanglers: tuple[Array, ...]
    physical_dimension: int = eqx.field(static=True)
    layer_count: int = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    mera_id: str = eqx.field(static=True)

    def __init__(
        self,
        isometries: Sequence[ArrayLike],
        disentanglers: Sequence[ArrayLike],
        /,
    ):
        isometries_ = tuple(jnp.asarray(value) for value in isometries)
        disentanglers_ = tuple(jnp.asarray(value) for value in disentanglers)
        if not isometries_ or len(disentanglers_) != len(isometries_):
            raise ValueError(
                "Binary MERA requires one isometry and disentangler per layer."
            )
        dtype = str(isometries_[0].dtype)
        for layer, (isometry, disentangler) in enumerate(
            zip(isometries_, disentanglers_, strict=True)
        ):
            if isometry.ndim != 3 or isometry.shape[0] != isometry.shape[1]:
                raise ValueError("MERA isometries require axes (fine, fine, coarse).")
            fine = isometry.shape[0]
            if isometry.shape[2] > fine * fine:
                raise ValueError("MERA coarse dimension exceeds the fine product space.")
            if disentangler.shape != (fine, fine, fine, fine):
                raise ValueError(
                    "MERA disentanglers require square two-site input/output axes."
                )
            if str(isometry.dtype) != dtype or str(disentangler.dtype) != dtype:
                raise TypeError("All MERA tensors must use one dtype.")
            if (
                layer + 1 < len(isometries_)
                and isometry.shape[2] != isometries_[layer + 1].shape[0]
            ):
                raise ValueError("Adjacent MERA coarse and fine dimensions must match.")
        self.isometries = isometries_
        self.disentanglers = disentanglers_
        self.physical_dimension = int(isometries_[0].shape[0])
        self.layer_count = len(isometries_)
        self.dtype = dtype
        self.mera_id = canonical_fingerprint(
            {
                "kind": "finite-binary-mera",
                "isometry_shapes": tuple(value.shape for value in isometries_),
                "disentangler_shapes": tuple(value.shape for value in disentanglers_),
                "dtype": dtype,
            }
        )

    def isometry_residuals(self) -> Array:
        residuals = []
        for isometry in self.isometries:
            matrix = isometry.reshape((-1, isometry.shape[2]))
            residuals.append(
                jnp.max(
                    jnp.abs(
                        jnp.conj(matrix.T) @ matrix
                        - jnp.eye(matrix.shape[1], dtype=matrix.dtype)
                    )
                )
            )
        return jnp.stack(tuple(residuals))

    def disentangler_residuals(self) -> Array:
        residuals = []
        for disentangler in self.disentanglers:
            dimension = disentangler.shape[0] * disentangler.shape[1]
            matrix = disentangler.reshape((dimension, dimension))
            residuals.append(
                jnp.max(
                    jnp.abs(
                        jnp.conj(matrix.T) @ matrix
                        - jnp.eye(dimension, dtype=matrix.dtype)
                    )
                )
            )
        return jnp.stack(tuple(residuals))


class MERAContractionEvidence(StrictModule):
    mera_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)
    isometry_residuals: Array
    disentangler_residuals: Array
    state_norm: Array
    finite: Array
    accepted: Array
    exact: bool = eqx.field(static=True)
    claim: str = eqx.field(static=True)
    admitted_state_elements: int = eqx.field(static=True)
    admitted_workspace_bytes: int = eqx.field(static=True)


class MERAContractionResult(StrictModule):
    value: Array
    state_vector: Array
    evidence: MERAContractionEvidence


class MERAUpdateEvidence(StrictModule):
    source_mera_id: str = eqx.field(static=True)
    result_mera_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)
    layer: int = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    residual_before: Array
    residual_after: Array
    tangent_norm: Array
    finite: Array
    accepted: Array
    exact_isometry: Array
    claim: str = eqx.field(static=True)
    admitted_workspace_bytes: int = eqx.field(static=True)


class MERAUpdateResult(StrictModule):
    mera: BinaryMERA
    evidence: MERAUpdateEvidence


def _admit_state(mera: BinaryMERA, policy: MERAResourcePolicy, /) -> tuple[int, int]:
    sites = 2**mera.layer_count
    elements = mera.physical_dimension**sites
    bytes_ = elements * precision_itemsize(mera.dtype) * 2
    if sites > policy.maximum_sites:
        raise MemoryError("MERA contraction exceeds maximum_sites before allocation.")
    if elements > policy.maximum_state_elements:
        raise MemoryError(
            "MERA contraction exceeds maximum_state_elements before allocation."
        )
    if bytes_ > policy.maximum_workspace_bytes:
        raise MemoryError(
            "MERA contraction exceeds maximum_workspace_bytes before allocation."
        )
    return elements, bytes_


def _expand_axis(state: Array, isometry: Array, axis: int, /) -> Array:
    state_symbols = [oe.get_symbol(index) for index in range(state.ndim)]
    first = oe.get_symbol(state.ndim)
    second = oe.get_symbol(state.ndim + 1)
    coarse = state_symbols[axis]
    output_symbols = state_symbols[:axis] + [first, second] + state_symbols[axis + 1 :]
    equation = (
        "".join(state_symbols)
        + ","
        + first
        + second
        + coarse
        + "->"
        + "".join(output_symbols)
    )
    return oe.contract(equation, state, isometry, optimize="greedy")


def _apply_disentangler(state: Array, disentangler: Array, first_axis: int, /) -> Array:
    state_symbols = [oe.get_symbol(index) for index in range(state.ndim)]
    output_first = oe.get_symbol(state.ndim)
    output_second = oe.get_symbol(state.ndim + 1)
    input_first = state_symbols[first_axis]
    input_second = state_symbols[first_axis + 1]
    output_symbols = list(state_symbols)
    output_symbols[first_axis] = output_first
    output_symbols[first_axis + 1] = output_second
    equation = (
        "".join(state_symbols)
        + ","
        + output_first
        + output_second
        + input_first
        + input_second
        + "->"
        + "".join(output_symbols)
    )
    return oe.contract(equation, state, disentangler, optimize="greedy")


def mera_state_vector(
    mera: BinaryMERA,
    top_state: ArrayLike,
    policy: MERAResourcePolicy,
    /,
) -> MERAContractionResult:
    """Exactly lower and contract the finite MERA into its physical state vector."""

    if not isinstance(mera, BinaryMERA) or not isinstance(policy, MERAResourcePolicy):
        raise TypeError("mera and policy have invalid types.")
    elements, bytes_ = _admit_state(mera, policy)
    top = jnp.asarray(top_state)
    if top.shape != (mera.isometries[-1].shape[2],) or str(top.dtype) != mera.dtype:
        raise ValueError("top_state shape and dtype must match the coarsest MERA leg.")
    state = top
    site_count = 1
    for layer in range(mera.layer_count - 1, -1, -1):
        for axis in range(site_count - 1, -1, -1):
            state = _expand_axis(state, mera.isometries[layer], axis)
        site_count *= 2
        for axis in range(1, site_count - 1, 2):
            state = _apply_disentangler(state, mera.disentanglers[layer], axis)
    vector = state.reshape((elements,))
    norm = jnp.real(jnp.vdot(vector, vector))
    isometry_residuals = mera.isometry_residuals()
    disentangler_residuals = mera.disentangler_residuals()
    finite = (
        jnp.all(jnp.isfinite(vector))
        & jnp.isfinite(norm)
        & jnp.all(jnp.isfinite(isometry_residuals))
        & jnp.all(jnp.isfinite(disentangler_residuals))
    )
    constraints = jnp.all(isometry_residuals <= policy.isometry_tolerance) & jnp.all(
        disentangler_residuals <= policy.isometry_tolerance
    )
    accepted = finite & constraints
    replay_id = canonical_fingerprint(
        {
            "kind": "finite-mera-state-replay",
            "mera": mera.mera_id,
            "policy": policy.policy_id,
        }
    )
    evidence = MERAContractionEvidence(
        mera.mera_id,
        policy.policy_id,
        replay_id,
        isometry_residuals,
        disentangler_residuals,
        norm,
        finite,
        accepted,
        True,
        "exact finite binary MERA contraction for the supplied tensors",
        elements,
        bytes_,
    )
    return MERAContractionResult(vector, vector, evidence)


def contract_mera(
    mera: BinaryMERA,
    top_state: ArrayLike,
    policy: MERAResourcePolicy,
    /,
    *,
    operator: ArrayLike | None = None,
) -> MERAContractionResult:
    if not isinstance(mera, BinaryMERA) or not isinstance(policy, MERAResourcePolicy):
        raise TypeError("mera and policy have invalid types.")
    physical_elements = mera.physical_dimension ** (2**mera.layer_count)
    operator_ = None if operator is None else jnp.asarray(operator)
    if operator_ is not None:
        if (
            operator_.shape != (physical_elements, physical_elements)
            or str(operator_.dtype) != mera.dtype
        ):
            raise ValueError(
                "MERA operator shape and dtype must match the physical state space."
            )
        if operator_.size > policy.maximum_state_elements:
            raise MemoryError("MERA operator exceeds maximum_state_elements.")
        operator_bytes = operator_.size * precision_itemsize(mera.dtype)
        state_bytes = physical_elements * precision_itemsize(mera.dtype) * 2
        if operator_bytes + state_bytes > policy.maximum_workspace_bytes:
            raise MemoryError(
                "MERA operator contraction exceeds maximum_workspace_bytes."
            )
    lowered = mera_state_vector(mera, top_state, policy)
    vector = lowered.state_vector
    if operator_ is None:
        value = jnp.vdot(vector, vector)
        route = "norm"
    else:
        value = oe.contract(
            "i,ij,j->", jnp.conj(vector), operator_, vector, optimize="greedy"
        )
        route = "expectation"
    finite = lowered.evidence.finite & jnp.all(jnp.isfinite(value))
    evidence = MERAContractionEvidence(
        lowered.evidence.mera_id,
        lowered.evidence.policy_id,
        canonical_fingerprint(
            {
                "kind": "finite-mera-contraction",
                "lowering": lowered.evidence.replay_id,
                "route": route,
            }
        ),
        lowered.evidence.isometry_residuals,
        lowered.evidence.disentangler_residuals,
        lowered.evidence.state_norm,
        finite,
        lowered.evidence.accepted & finite,
        True,
        f"exact finite binary MERA {route}",
        lowered.evidence.admitted_state_elements,
        lowered.evidence.admitted_workspace_bytes,
    )
    return MERAContractionResult(value, vector, evidence)


def update_mera_isometry(
    mera: BinaryMERA,
    layer: int,
    gradient: ArrayLike,
    step_size: float,
    policy: MERAResourcePolicy,
    /,
) -> MERAUpdateResult:
    """Take one native Stiefel-manifold gradient step on a MERA isometry."""

    if not isinstance(mera, BinaryMERA) or not isinstance(policy, MERAResourcePolicy):
        raise TypeError("mera and policy have invalid types.")
    layer_ = int(layer)
    step = float(step_size)
    if not 0 <= layer_ < mera.layer_count or not 0.0 < step < float("inf"):
        raise ValueError("MERA layer and step_size are invalid.")
    isometry = mera.isometries[layer_]
    gradient_ = jnp.asarray(gradient)
    if gradient_.shape != isometry.shape or gradient_.dtype != isometry.dtype:
        raise ValueError("MERA gradient shape and dtype must match the isometry.")
    matrix = isometry.reshape((-1, isometry.shape[2]))
    gradient_matrix = gradient_.reshape(matrix.shape)
    workspace_bytes = matrix.size * precision_itemsize(mera.dtype) * 4
    if (
        matrix.size > policy.maximum_state_elements
        or workspace_bytes > policy.maximum_workspace_bytes
    ):
        raise MemoryError(
            "MERA isometry update exceeds resource policy before allocation."
        )
    if jnp.issubdtype(matrix.dtype, jnp.complexfloating):
        manifold = ComplexStiefelManifold(
            matrix.shape[0], matrix.shape[1], tolerance=policy.isometry_tolerance
        )
    else:
        manifold = StiefelManifold(
            matrix.shape[0], matrix.shape[1], tolerance=policy.isometry_tolerance
        )
    tangent = manifold.egrad_to_rgrad(matrix, gradient_matrix)
    updated_matrix = manifold.retract(matrix, -step * tangent)
    residual_before = manifold.constraint_residual(matrix)
    residual_after = manifold.constraint_residual(updated_matrix)
    isometries = list(mera.isometries)
    isometries[layer_] = updated_matrix.reshape(isometry.shape)
    updated = BinaryMERA(tuple(isometries), mera.disentanglers)
    finite = jnp.all(jnp.isfinite(updated_matrix)) & jnp.isfinite(residual_after)
    accepted = finite & (residual_after <= policy.isometry_tolerance)
    replay_id = canonical_fingerprint(
        {
            "kind": "mera-isometry-update",
            "mera": mera.mera_id,
            "layer": layer_,
            "policy": policy.policy_id,
            "manifold": manifold.manifold_id,
        }
    )
    evidence = MERAUpdateEvidence(
        mera.mera_id,
        updated.mera_id,
        replay_id,
        layer_,
        manifold.manifold_id,
        residual_before,
        residual_after,
        jnp.linalg.norm(tangent),
        finite,
        accepted,
        accepted,
        "native Stiefel QR retraction preserves the local isometry constraint",
        workspace_bytes,
    )
    return MERAUpdateResult(updated, evidence)


__all__ = [
    "BinaryMERA",
    "MERAContractionEvidence",
    "MERAContractionResult",
    "MERAResourcePolicy",
    "MERAUpdateEvidence",
    "MERAUpdateResult",
    "contract_mera",
    "mera_state_vector",
    "update_mera_isometry",
]
