#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._numerics import solve_weighted_least_squares
from ..._strict import StrictModule
from ...metrix import EuclideanStateGeometry
from .._layout import InputLayout, StateLayout
from .._system import DiscreteSystem
from .._trajectory import TrajectoryData
from ._features import AbstractFeatureLibrary
from ._status import (
    IDENTIFICATION_INSUFFICIENT_SAMPLES,
    IDENTIFICATION_NONFINITE,
    IDENTIFICATION_RANK_DEFICIENT,
    IDENTIFICATION_SUCCESS,
)


DMDMode: TypeAlias = Literal["exact", "projected"]
DMDRankPolicy: TypeAlias = Literal["numerical", "fixed", "energy"]


def _adjoint(matrix: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(matrix), -1, -2)


def _flatten_event(
    values: Array,
    event_shape: tuple[int, ...],
    /,
) -> tuple[Array, tuple[int, ...]]:
    event_rank = len(event_shape)
    leading = values.shape if event_rank == 0 else values.shape[:-event_rank]
    size = int(np.prod(event_shape)) if event_shape else 1
    return values.reshape((-1, size)), leading


def _rank_mask(
    singular_values: Array,
    /,
    *,
    rank: int | None,
    energy_threshold: float | None,
    rcond: float | None,
    rows: int,
    columns: int,
) -> tuple[Array, Array, DMDRankPolicy]:
    dtype = singular_values.real.dtype
    resolved_rcond = (
        float(rcond)
        if rcond is not None
        else float(np.finfo(np.dtype(dtype)).eps * max(rows, columns))
    )
    if not np.isfinite(resolved_rcond) or resolved_rcond < 0.0:
        raise ValueError("rcond must be finite and nonnegative or None.")
    largest = jnp.max(singular_values, initial=0.0)
    numerical = singular_values > largest * resolved_rcond
    positions = jnp.arange(singular_values.shape[0])
    if rank is not None:
        requested = int(rank)
        if requested < 1 or requested > singular_values.shape[0]:
            raise ValueError(f"rank must lie in [1, {singular_values.shape[0]}].")
        retained = numerical & (positions < requested)
        policy: DMDRankPolicy = "fixed"
    elif energy_threshold is not None:
        threshold = float(energy_threshold)
        if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("energy_threshold must lie in (0, 1].")
        energy = singular_values**2
        total = jnp.sum(energy)
        cumulative_before = jnp.concatenate(
            (jnp.zeros((1,), dtype=energy.dtype), jnp.cumsum(energy)[:-1])
        )
        retained = numerical & (cumulative_before < threshold * total)
        policy = "energy"
    else:
        retained = numerical
        policy = "numerical"
    return retained, jnp.asarray(resolved_rcond, dtype=dtype), policy


def _weighted_snapshots(data: TrajectoryData, /):
    transitions = data.transitions()
    source, _ = _flatten_event(transitions.source_states, data.state_layout.shape)
    target, _ = _flatten_event(transitions.target_states, data.state_layout.shape)
    mask = transitions.valid.reshape((-1,))
    weights = transitions.weights.reshape((-1,))
    source = jnp.where(mask[:, None], source, 0.0)
    target = jnp.where(mask[:, None], target, 0.0)
    denominator = jnp.maximum(jnp.sum(jnp.where(mask, weights, 0.0)), 1.0)
    roots = jnp.sqrt(jnp.where(mask, weights, 0.0) / denominator)
    return transitions, source, target, mask, weights, roots


class DMDDiagnostics(StrictModule):
    """Rank, conditioning, and weighted residual evidence for DMD or DMDc."""

    singular_values: Array
    retained_rank: Array
    numerical_rank: Array
    sample_count: Array
    condition_number: Array
    weighted_residual_norm: Array
    rcond: Array
    rank_policy: DMDRankPolicy = eqx.field(static=True)


class _LinearIdentifiedTransition(StrictModule):
    state_matrix: Array
    input_matrix: Array | None
    state_shape: tuple[int, ...] = eqx.field(static=True)
    input_shape: tuple[int, ...] | None = eqx.field(static=True)

    def __call__(
        self,
        coordinate: Array,
        state: Array,
        inputs_or_args,
        args=None,
    ) -> Array:
        del coordinate, args
        flat_state = jnp.asarray(state).reshape((-1,))
        result = self.state_matrix @ flat_state
        if self.input_matrix is not None:
            if inputs_or_args is None:
                raise ValueError("Identified controlled map requires inputs.")
            result = result + self.input_matrix @ jnp.asarray(inputs_or_args).reshape(
                (-1,)
            )
        return result.reshape(self.state_shape)


class DMDResult(StrictModule):
    """A diagnosed exact/projected DMD or controlled-DMD fit."""

    state_matrix: Array
    input_matrix: Array | None
    eigenvalues: Array
    modes: Array
    diagnostics: DMDDiagnostics
    valid: Array
    status: Array
    state_layout: StateLayout
    input_layout: InputLayout | None
    mode: DMDMode = eqx.field(static=True)
    eigenvalue_kind: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def predict(self, states: ArrayLike, inputs: ArrayLike | None = None, /) -> Array:
        values = jnp.asarray(states)
        state_rank = len(self.state_layout.shape)
        batch = values.shape if state_rank == 0 else values.shape[:-state_rank]
        flat = values.reshape(batch + (self.state_layout.size,))
        predicted = oe.contract("ij,...j->...i", self.state_matrix, flat)
        if self.input_matrix is not None:
            if inputs is None:
                raise ValueError("inputs are required by this controlled DMD fit.")
            input_layout = self.input_layout
            if input_layout is None:
                raise RuntimeError("Controlled DMD result is missing its input layout.")
            predicted = predicted + oe.contract(
                "ij,...j->...i",
                self.input_matrix,
                jnp.asarray(inputs).reshape(batch + (input_layout.size,)),
            )
        elif inputs is not None:
            raise ValueError("This DMD fit has no input model.")
        return predicted.reshape(batch + self.state_layout.shape)

    def to_system(self, /, *, system_id: str | None = None) -> DiscreteSystem:
        if not bool(self.valid):
            raise ValueError("Cannot construct a system from an invalid DMD result.")
        if not isinstance(self.state_layout.geometry, EuclideanStateGeometry):
            raise ValueError(
                "DMD defines an ambient Euclidean map; non-Euclidean state "
                "layouts require a structured manifold identification method."
            )
        identifier = (
            f"identified-dmd:{self.source_id}:{self.method_id}"
            if system_id is None
            else system_id
        )
        transition = _LinearIdentifiedTransition(
            state_matrix=self.state_matrix,
            input_matrix=self.input_matrix,
            state_shape=self.state_layout.shape,
            input_shape=(None if self.input_layout is None else self.input_layout.shape),
        )
        return DiscreteSystem(
            transition,
            state_layout=self.state_layout,
            input_layout=self.input_layout,
            system_id=identifier,
        )


def fit_dmd(
    data: TrajectoryData,
    /,
    *,
    rank: int | None = None,
    energy_threshold: float | None = None,
    mode: DMDMode = "exact",
    continuous_eigenvalues: bool = False,
    rcond: float | None = None,
) -> DMDResult:
    """Fit exact/projected DMD or DMDc from mask-safe trajectory transitions."""
    if not isinstance(data, TrajectoryData):
        raise TypeError("data must be TrajectoryData.")
    if rank is not None and energy_threshold is not None:
        raise ValueError("rank and energy_threshold are mutually exclusive.")
    if mode not in ("exact", "projected"):
        raise ValueError("mode must be 'exact' or 'projected'.")
    transitions, source, target, mask, weights, roots = _weighted_snapshots(data)
    input_values = None
    if transitions.inputs is not None:
        if data.input_layout is None:
            raise RuntimeError("Controlled trajectory is missing its input layout.")
        input_values, _ = _flatten_event(transitions.inputs, data.input_layout.shape)
        input_values = jnp.where(mask[:, None], input_values, 0.0)
    design = (
        source
        if input_values is None
        else jnp.concatenate((source, input_values), axis=-1)
    )
    omega = (roots[:, None] * design).T
    target_columns = (roots[:, None] * target).T
    left, singular_values, right_h = jnp.linalg.svd(omega, full_matrices=False)
    retained, resolved_rcond, rank_policy = _rank_mask(
        singular_values,
        rank=rank,
        energy_threshold=energy_threshold,
        rcond=rcond,
        rows=omega.shape[0],
        columns=omega.shape[1],
    )
    inverse = jnp.where(retained, 1.0 / singular_values, 0.0)
    fitted = target_columns @ _adjoint(right_h) @ (inverse[:, None] * _adjoint(left))
    state_size = data.state_layout.size
    state_matrix = fitted[:, :state_size]
    input_matrix = None if input_values is None else fitted[:, state_size:]
    prediction = source @ state_matrix.T
    if input_matrix is not None:
        prediction = prediction + input_values @ input_matrix.T
    residual = jnp.where(mask[:, None], target - prediction, 0.0)
    weighted_residual = jnp.sqrt(
        jnp.sum(jnp.where(mask, weights, 0.0)[:, None] * jnp.abs(residual) ** 2)
    )
    retained_rank = jnp.sum(retained).astype(jnp.int32)
    numerical_rank = jnp.sum(
        singular_values > singular_values[0] * resolved_rcond
    ).astype(jnp.int32)
    smallest = jnp.min(jnp.where(retained, singular_values, jnp.inf), initial=jnp.inf)
    condition = jnp.where(retained_rank > 0, singular_values[0] / smallest, jnp.inf)

    if input_values is None:
        state_columns = (roots[:, None] * source).T
        state_left, state_singular, state_right_h = jnp.linalg.svd(
            state_columns, full_matrices=False
        )
        state_retained, _, _ = _rank_mask(
            state_singular,
            rank=(None if rank is None else min(int(rank), int(state_singular.shape[0]))),
            energy_threshold=energy_threshold,
            rcond=rcond,
            rows=state_columns.shape[0],
            columns=state_columns.shape[1],
        )
        state_inverse = jnp.where(state_retained, 1.0 / state_singular, 0.0)
        reduced = (
            _adjoint(state_left)
            @ target_columns
            @ _adjoint(state_right_h)
            @ jnp.diag(state_inverse)
        )
        eigenvalues, reduced_modes = jnp.linalg.eig(reduced)
        projected_modes = state_left @ reduced_modes
        exact_modes = (
            target_columns
            @ _adjoint(state_right_h)
            @ (state_inverse[:, None] * reduced_modes)
        )
        modes = exact_modes if mode == "exact" else projected_modes
    else:
        eigenvalues, modes = jnp.linalg.eig(state_matrix)

    eigenvalue_kind = "discrete"
    if continuous_eigenvalues:
        duration = np.asarray(
            transitions.target_coordinates - transitions.source_coordinates
        ).reshape((-1,))
        valid_duration = duration[np.asarray(mask)]
        if valid_duration.size == 0:
            raise ValueError(
                "Continuous eigenvalues require at least one valid transition."
            )
        tolerance = (
            100.0
            * np.finfo(valid_duration.dtype).eps
            * max(1.0, abs(float(valid_duration[0])))
        )
        if np.any(np.abs(valid_duration - valid_duration[0]) > tolerance):
            raise ValueError(
                "Continuous-time DMD eigenvalues require uniform valid transition spacing."
            )
        eigenvalues = jnp.log(eigenvalues) / jnp.asarray(valid_duration[0])
        eigenvalue_kind = "continuous"

    finite = (
        jnp.all(jnp.isfinite(state_matrix))
        & (True if input_matrix is None else jnp.all(jnp.isfinite(input_matrix)))
        & jnp.all(jnp.isfinite(eigenvalues))
        & jnp.all(jnp.isfinite(modes))
        & jnp.isfinite(weighted_residual)
    )
    sample_count = jnp.sum(mask).astype(jnp.int32)
    requested_rank = None if rank is None else int(rank)
    rank_valid = (retained_rank > 0) & (
        True if requested_rank is None else retained_rank == requested_rank
    )
    valid = finite & (sample_count > 0) & rank_valid
    status = jnp.where(
        ~finite,
        IDENTIFICATION_NONFINITE,
        jnp.where(
            sample_count == 0,
            IDENTIFICATION_INSUFFICIENT_SAMPLES,
            jnp.where(
                ~rank_valid,
                IDENTIFICATION_RANK_DEFICIENT,
                IDENTIFICATION_SUCCESS,
            ),
        ),
    ).astype(jnp.int32)
    diagnostics = DMDDiagnostics(
        singular_values=singular_values,
        retained_rank=retained_rank,
        numerical_rank=numerical_rank,
        sample_count=sample_count,
        condition_number=condition,
        weighted_residual_norm=weighted_residual,
        rcond=resolved_rcond,
        rank_policy=rank_policy,
    )
    method_id = f"dmd:{mode}:rank={rank_policy}:eigenvalues={eigenvalue_kind}"
    return DMDResult(
        state_matrix=state_matrix,
        input_matrix=input_matrix,
        eigenvalues=eigenvalues,
        modes=modes,
        diagnostics=diagnostics,
        valid=valid,
        status=status,
        state_layout=data.state_layout,
        input_layout=data.input_layout,
        mode=mode,
        eigenvalue_kind=eigenvalue_kind,
        source_id=data.source_id,
        method_id=method_id,
    )


class EDMDDiagnostics(StrictModule):
    """Separate evolution and decoder regression diagnostics for EDMD."""

    evolution_rank: Array
    decoder_rank: Array
    evolution_condition_number: Array
    decoder_condition_number: Array
    evolution_residual_norm: Array
    decoder_residual_norm: Array
    sample_count: Array


class _EDMDIdentifiedTransition(StrictModule):
    library: AbstractFeatureLibrary
    feature_matrix: Array
    input_matrix: Array | None
    decoder_matrix: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)

    def __call__(
        self,
        coordinate: Array,
        state: Array,
        inputs_or_args,
        args=None,
    ) -> Array:
        del coordinate, args
        features = self.library.evaluate(state).values
        next_features = self.feature_matrix @ features
        if self.input_matrix is not None:
            if inputs_or_args is None:
                raise ValueError("Identified controlled EDMD map requires inputs.")
            next_features = next_features + self.input_matrix @ jnp.asarray(
                inputs_or_args
            ).reshape((-1,))
        return (self.decoder_matrix @ next_features).reshape(self.state_shape)


class EDMDResult(StrictModule):
    """A diagnosed EDMD fit with a separate physical-state decoder."""

    feature_matrix: Array
    input_matrix: Array | None
    decoder_matrix: Array
    eigenvalues: Array
    modes: Array
    diagnostics: EDMDDiagnostics
    valid: Array
    status: Array
    library: AbstractFeatureLibrary
    state_layout: StateLayout
    input_layout: InputLayout | None
    source_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def predict(self, states: ArrayLike, inputs: ArrayLike | None = None, /) -> Array:
        evaluation = self.library.evaluate(states)
        features = oe.contract("ij,...j->...i", self.feature_matrix, evaluation.values)
        if self.input_matrix is not None:
            if inputs is None:
                raise ValueError("inputs are required by this controlled EDMD fit.")
            input_layout = self.input_layout
            if input_layout is None:
                raise RuntimeError("Controlled EDMD result is missing its input layout.")
            batch = evaluation.valid.shape
            features = features + oe.contract(
                "ij,...j->...i",
                self.input_matrix,
                jnp.asarray(inputs).reshape(batch + (input_layout.size,)),
            )
        elif inputs is not None:
            raise ValueError("This EDMD fit has no input model.")
        decoded = oe.contract("ij,...j->...i", self.decoder_matrix, features)
        return decoded.reshape(evaluation.valid.shape + self.state_layout.shape)

    def to_system(self, /, *, system_id: str | None = None) -> DiscreteSystem:
        if not bool(self.valid):
            raise ValueError("Cannot construct a system from an invalid EDMD result.")
        identifier = (
            f"identified-edmd:{self.source_id}:{self.library.library_id}"
            if system_id is None
            else system_id
        )
        if not isinstance(self.state_layout.geometry, EuclideanStateGeometry):
            raise ValueError(
                "EDMD decoding defines an ambient Euclidean map; non-Euclidean "
                "state layouts require a structured manifold identification method."
            )
        return DiscreteSystem(
            _EDMDIdentifiedTransition(
                library=self.library,
                feature_matrix=self.feature_matrix,
                input_matrix=self.input_matrix,
                decoder_matrix=self.decoder_matrix,
                state_shape=self.state_layout.shape,
            ),
            state_layout=self.state_layout,
            input_layout=self.input_layout,
            system_id=identifier,
        )


def fit_edmd(
    data: TrajectoryData,
    library: AbstractFeatureLibrary,
    /,
    *,
    ridge: float = 0.0,
    decoder_ridge: float = 0.0,
    rcond: float | None = None,
) -> EDMDResult:
    """Fit feature-space evolution and a separately diagnosed state decoder."""
    if not isinstance(data, TrajectoryData):
        raise TypeError("data must be TrajectoryData.")
    if not isinstance(library, AbstractFeatureLibrary):
        raise TypeError("library must be an AbstractFeatureLibrary.")
    if library.state_layout.layout_id != data.state_layout.layout_id:
        raise ValueError("library and trajectory must use the same state layout.")
    if library.input_layout is not None:
        raise ValueError(
            "EDMD feature libraries must be state-only; controls enter through the declared input layout."
        )
    transitions = data.transitions()
    source_evaluation = library.evaluate(transitions.source_states)
    target_evaluation = library.evaluate(transitions.target_states)
    source_features = source_evaluation.values.reshape((-1, library.num_features))
    target_features = target_evaluation.values.reshape((-1, library.num_features))
    source_states, _ = _flatten_event(transitions.source_states, data.state_layout.shape)
    mask = (
        transitions.valid & source_evaluation.valid & target_evaluation.valid
    ).reshape((-1,))
    weights = transitions.weights.reshape((-1,))
    input_values = None
    design = source_features
    if transitions.inputs is not None:
        if data.input_layout is None:
            raise RuntimeError("Controlled trajectory is missing its input layout.")
        input_values, _ = _flatten_event(transitions.inputs, data.input_layout.shape)
        design = jnp.concatenate((source_features, input_values), axis=-1)
    evolution = solve_weighted_least_squares(
        design,
        target_features,
        mask=mask,
        weights=weights,
        ridge=ridge,
        rcond=rcond,
    )
    decoder = solve_weighted_least_squares(
        source_features,
        source_states,
        mask=mask,
        weights=weights,
        ridge=decoder_ridge,
        rcond=rcond,
    )
    feature_matrix = evolution.coefficients[: library.num_features].T
    input_matrix = (
        None if input_values is None else evolution.coefficients[library.num_features :].T
    )
    decoder_matrix = decoder.coefficients.T
    eigenvalues, eigenvectors = jnp.linalg.eig(feature_matrix)
    modes = decoder_matrix.astype(eigenvectors.dtype) @ eigenvectors
    finite = (
        jnp.all(jnp.isfinite(eigenvalues))
        & jnp.all(jnp.isfinite(modes))
        & jnp.all(jnp.isfinite(feature_matrix))
        & jnp.all(jnp.isfinite(decoder_matrix))
        & (True if input_matrix is None else jnp.all(jnp.isfinite(input_matrix)))
    )
    valid = evolution.valid & decoder.valid & finite
    insufficient = evolution.sample_count < design.shape[-1] | (
        decoder.sample_count < library.num_features
    )
    rank_deficient = ((evolution.rank < design.shape[-1]) & (ridge == 0.0)) | (
        (decoder.rank < library.num_features) & (decoder_ridge == 0.0)
    )
    status = jnp.where(
        ~finite,
        IDENTIFICATION_NONFINITE,
        jnp.where(
            insufficient,
            IDENTIFICATION_INSUFFICIENT_SAMPLES,
            jnp.where(
                rank_deficient,
                IDENTIFICATION_RANK_DEFICIENT,
                IDENTIFICATION_SUCCESS,
            ),
        ),
    ).astype(jnp.int32)
    diagnostics = EDMDDiagnostics(
        evolution_rank=evolution.rank,
        decoder_rank=decoder.rank,
        evolution_condition_number=evolution.condition_number,
        decoder_condition_number=decoder.condition_number,
        evolution_residual_norm=jnp.linalg.norm(
            jnp.where(mask[:, None], evolution.residual, 0.0)
        ),
        decoder_residual_norm=jnp.linalg.norm(
            jnp.where(mask[:, None], decoder.residual, 0.0)
        ),
        sample_count=evolution.sample_count,
    )
    return EDMDResult(
        feature_matrix=feature_matrix,
        input_matrix=input_matrix,
        decoder_matrix=decoder_matrix,
        eigenvalues=eigenvalues,
        modes=modes,
        diagnostics=diagnostics,
        valid=valid,
        status=status,
        library=library,
        state_layout=data.state_layout,
        input_layout=data.input_layout,
        source_id=data.source_id,
        method_id=f"edmd:ridge={float(ridge):g}:decoder-ridge={float(decoder_ridge):g}",
    )


__all__ = [
    "DMDDiagnostics",
    "DMDMode",
    "DMDRankPolicy",
    "DMDResult",
    "EDMDDiagnostics",
    "EDMDResult",
    "fit_dmd",
    "fit_edmd",
]
