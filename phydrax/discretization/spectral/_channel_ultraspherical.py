#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ArraySpace,
    BandedLinearOperator,
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    prepare as prepare_linear_solve,
    PreparedLinearSolve,
    RHSLayout,
    solve as solve_linear,
    StructuredDirect,
)


class _PreparedBatchedTauSolve(StrictModule, NonTrainableState):
    """Batched fixed-band solve with fixed-rank dense tau-row corrections."""

    base: PreparedLinearSolve
    correction: PreparedLinearSolve
    delta_rows: Array
    inverse_tau_columns: Array
    column_scale: Array
    tau_rhs_scale: Array
    preparation_failed: Array
    batch_size: int = eqx.field(static=True)
    size: int = eqx.field(static=True)
    tau_start: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    lower_bandwidth: int = eqx.field(static=True)
    upper_bandwidth: int = eqx.field(static=True)
    operator_bytes: int = eqx.field(static=True)
    factor_bytes: int = eqx.field(static=True)
    pivot_margin: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)

    def solve(self, right_hand_side: Array, /) -> tuple[Array, Array]:
        rhs = jnp.asarray(right_hand_side)
        expected = (self.batch_size, self.size)
        if rhs.ndim != 3 or rhs.shape[:2] != expected:
            raise ValueError(
                f"Tau right-hand side must have shape {expected} + (columns,)."
            )
        system_rhs = rhs.at[:, self.tau_start : self.tau_start + self.rank, :].multiply(
            self.tau_rhs_scale[..., None]
        )
        layout = RHSLayout((int(rhs.shape[-1]),))

        def solve_once(value: Array) -> tuple[Array, Array]:
            base_result = solve_linear(self.base, value, rhs_layout=layout)
            correction_rhs = oe.contract(
                "brn,bnq->brq",
                self.delta_rows,
                base_result.value,
                backend="jax",
            )
            correction_result = solve_linear(
                self.correction, correction_rhs, rhs_layout=layout
            )
            solution = base_result.value - oe.contract(
                "bnr,brq->bnq",
                self.inverse_tau_columns,
                correction_result.value,
                backend="jax",
            )
            solve_failed = self.preparation_failed | jnp.any(
                ~jnp.isfinite(solution), axis=(-2, -1)
            )
            return solution, solve_failed

        def action(value: Array) -> Array:
            base_value = self.base.problem.operator.mv_block(value)
            tau_update = oe.contract(
                "brn,bnq->brq", self.delta_rows, value, backend="jax"
            )
            return base_value.at[:, self.tau_start : self.tau_start + self.rank, :].add(
                tau_update
            )

        solution, failed = solve_once(system_rhs)
        for _ in range(2):
            correction, correction_failed = solve_once(system_rhs - action(solution))
            solution = solution + correction
            failed = (
                failed
                | correction_failed
                | jnp.any(~jnp.isfinite(solution), axis=(-2, -1))
            )
        residual = system_rhs - action(solution)
        residual_norm = jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2, axis=-2))
        right_hand_side_norm = jnp.sqrt(jnp.sum(jnp.abs(system_rhs) ** 2, axis=-2))
        relative_residual = residual_norm / jnp.maximum(right_hand_side_norm, 1.0)
        failed = failed | jnp.any(relative_residual > self.residual_tolerance, axis=-1)
        return self.column_scale[..., None] * solution, failed


class PreparedUltrasphericalChannel(StrictModule, NonTrainableState):
    """Pressure-eliminated channel solve retaining primitive public fields."""

    helmholtz: _PreparedBatchedTauSolve
    biharmonic: _PreparedBatchedTauSolve
    pressure_recovery: _PreparedBatchedTauSolve
    chebyshev_to_c1: Array
    chebyshev_to_c2: Array
    chebyshev_to_c4: Array
    modal_derivative: Array
    nonzero_mode_indices: Array
    streamwise_wavenumbers: Array
    spanwise_wavenumbers: Array
    synthesis: Array
    quadrature_weights: Array
    wall_length: Array
    horizontal_scale: Array
    shift: Array
    viscosity: Array
    bulk_influence: Array
    bulk_influence_failed: Array
    bulk_schur: PreparedLinearSolve
    zero_mode_index: int = eqx.field(static=True)
    mode_count: int = eqx.field(static=True)
    horizontal_batch_size: int = eqx.field(static=True)
    lower_bandwidth: int = eqx.field(static=True)
    upper_bandwidth: int = eqx.field(static=True)
    correction_rank: int = eqx.field(static=True)
    shared_basis_bytes: int = eqx.field(static=True)
    operator_bytes: int = eqx.field(static=True)
    factor_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    persistent_bytes: int = eqx.field(static=True)
    preparation_bytes: int = eqx.field(static=True)
    pivot_margin: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def solve(
        self,
        modal_rhs: Array,
        lower_wall_velocity: Array,
        upper_wall_velocity: Array,
        mean_values: Array,
        /,
        *,
        mean_kind: str,
    ) -> tuple[Array, Array, Array, Array, Array]:
        count = self.mode_count
        batch_count = self.horizontal_batch_size
        if modal_rhs.shape != (batch_count, count, 3):
            raise ValueError(
                "Ultraspherical channel right-hand side has an incompatible shape."
            )
        dtype = modal_rhs.dtype
        real_dtype = modal_rhs.real.dtype
        kx = self.streamwise_wavenumbers.astype(real_dtype)
        kz = self.spanwise_wavenumbers.astype(real_dtype)
        wave_square = kx * kx + kz * kz
        horizontal_divergence_rhs = (
            1j * kx[:, None] * modal_rhs[..., 0] + 1j * kz[:, None] * modal_rhs[..., 2]
        )
        vorticity_rhs = (
            1j * kz[:, None] * modal_rhs[..., 0] - 1j * kx[:, None] * modal_rhs[..., 2]
        )

        helmholtz_forcing = (
            jnp.zeros((batch_count, count, 2), dtype=dtype).at[..., 0].set(vorticity_rhs)
        )
        helmholtz_forcing = helmholtz_forcing.at[self.zero_mode_index, :, 0].set(
            modal_rhs[self.zero_mode_index, :, 0]
        )
        helmholtz_forcing = helmholtz_forcing.at[self.zero_mode_index, :, 1].set(
            modal_rhs[self.zero_mode_index, :, 2]
        )
        converted_helmholtz = oe.contract(
            "ij,bjq->biq",
            self.chebyshev_to_c2.astype(dtype),
            helmholtz_forcing,
            backend="jax",
        )
        helmholtz_rhs = jnp.zeros_like(converted_helmholtz)
        helmholtz_rhs = helmholtz_rhs.at[:, : count - 2].set(
            converted_helmholtz[:, : count - 2]
        )
        lower_tangential = self.horizontal_scale.astype(dtype) * jnp.asarray(
            (lower_wall_velocity[0], lower_wall_velocity[2]), dtype=dtype
        )
        upper_tangential = self.horizontal_scale.astype(dtype) * jnp.asarray(
            (upper_wall_velocity[0], upper_wall_velocity[2]), dtype=dtype
        )
        helmholtz_rhs = helmholtz_rhs.at[self.zero_mode_index, count - 2].set(
            lower_tangential
        )
        helmholtz_rhs = helmholtz_rhs.at[self.zero_mode_index, count - 1].set(
            upper_tangential
        )
        helmholtz_solution, helmholtz_failed = self.helmholtz.solve(helmholtz_rhs)

        pressure_gradient = jnp.asarray(mean_values, dtype=real_dtype)
        zero_u = helmholtz_solution[self.zero_mode_index, :, 0]
        zero_w = helmholtz_solution[self.zero_mode_index, :, 1]
        schur_failed = jnp.asarray(False)
        if mean_kind == "bulk_flux":
            base_bulk = self._bulk_velocity(zero_u, zero_w)
            schur_result = solve_linear(
                self.bulk_schur,
                mean_values.astype(base_bulk.dtype) - base_bulk,
            )
            pressure_gradient = schur_result.value
            schur_failed = jnp.any(~schur_result.successful)
        zero_u = zero_u + self.bulk_influence * pressure_gradient[0].astype(dtype)
        zero_w = zero_w + self.bulk_influence * pressure_gradient[1].astype(dtype)

        eliminated_rhs = -wave_square[:, None] * modal_rhs[..., 1] - oe.contract(
            "ij,bj->bi",
            self.modal_derivative.astype(dtype),
            horizontal_divergence_rhs,
            backend="jax",
        )
        nonzero_rhs = eliminated_rhs[self.nonzero_mode_indices]
        converted_biharmonic = oe.contract(
            "ij,bj->bi",
            self.chebyshev_to_c4.astype(dtype),
            nonzero_rhs,
            backend="jax",
        )
        biharmonic_rhs = jnp.zeros(
            (self.nonzero_mode_indices.size, count, 1), dtype=dtype
        )
        biharmonic_rhs = biharmonic_rhs.at[:, : count - 4, 0].set(
            converted_biharmonic[:, : count - 4]
        )
        nonzero_v_columns, biharmonic_failed = self.biharmonic.solve(biharmonic_rhs)
        velocity_v = jnp.zeros((batch_count, count), dtype=dtype)
        velocity_v = velocity_v.at[self.nonzero_mode_indices].set(
            nonzero_v_columns[..., 0]
        )
        velocity_v = velocity_v.at[self.zero_mode_index, 0].set(
            self.horizontal_scale.astype(dtype) * lower_wall_velocity[1].astype(dtype)
        )

        derivative_v = oe.contract(
            "ij,bj->bi",
            self.modal_derivative.astype(dtype),
            velocity_v,
            backend="jax",
        )
        safe_wave_square = jnp.where(wave_square == 0.0, 1.0, wave_square)
        vorticity = helmholtz_solution[..., 0]
        velocity_u = (
            1j * kx[:, None] * derivative_v - 1j * kz[:, None] * vorticity
        ) / safe_wave_square[:, None]
        velocity_w = (
            1j * kz[:, None] * derivative_v + 1j * kx[:, None] * vorticity
        ) / safe_wave_square[:, None]
        velocity_u = velocity_u.at[self.zero_mode_index].set(zero_u)
        velocity_w = velocity_w.at[self.zero_mode_index].set(zero_w)
        velocity = jnp.stack((velocity_u, velocity_v, velocity_w), axis=-1)

        second_derivative_v = oe.contract(
            "ij,bj->bi",
            self.modal_derivative.astype(dtype),
            derivative_v,
            backend="jax",
        )
        third_derivative_v = oe.contract(
            "ij,bj->bi",
            self.modal_derivative.astype(dtype),
            second_derivative_v,
            backend="jax",
        )
        helmholtz_derivative_v = (
            self.shift.astype(dtype) + self.viscosity.astype(dtype) * wave_square[:, None]
        ) * derivative_v - self.viscosity.astype(dtype) * third_derivative_v
        pressure = (
            -(horizontal_divergence_rhs + helmholtz_derivative_v)
            / safe_wave_square[:, None]
        )
        zero_pressure_rhs_chebyshev = (
            modal_rhs[self.zero_mode_index, :, 1]
            - self.shift.astype(dtype) * velocity_v[self.zero_mode_index]
            + self.viscosity.astype(dtype) * second_derivative_v[self.zero_mode_index]
        )
        converted_pressure_rhs = oe.contract(
            "ij,j->i",
            self.chebyshev_to_c1.astype(dtype),
            zero_pressure_rhs_chebyshev,
            backend="jax",
        )
        pressure_rhs = jnp.zeros((1, count, 1), dtype=dtype)
        pressure_rhs = pressure_rhs.at[0, : count - 1, 0].set(
            converted_pressure_rhs[: count - 1]
        )
        reordered_pressure, pressure_failed = self.pressure_recovery.solve(pressure_rhs)
        zero_pressure = jnp.concatenate(
            (reordered_pressure[0, -1:, 0], reordered_pressure[0, :-1, 0])
        )
        pressure = pressure.at[self.zero_mode_index].set(zero_pressure)

        derivative_velocity = oe.contract(
            "ij,bjc->bic",
            self.modal_derivative.astype(dtype),
            velocity,
            backend="jax",
        )
        second_derivative_velocity = oe.contract(
            "ij,bjc->bic",
            self.modal_derivative.astype(dtype),
            derivative_velocity,
            backend="jax",
        )
        helmholtz_velocity = (
            self.shift.astype(dtype)
            + self.viscosity.astype(dtype) * wave_square[:, None, None]
        ) * velocity - self.viscosity.astype(dtype) * second_derivative_velocity
        pressure_gradient_modal = jnp.stack(
            (
                1j * kx[:, None] * pressure,
                oe.contract(
                    "ij,bj->bi",
                    self.modal_derivative.astype(dtype),
                    pressure,
                    backend="jax",
                ),
                1j * kz[:, None] * pressure,
            ),
            axis=-1,
        )
        effective_rhs = modal_rhs
        effective_rhs = effective_rhs.at[self.zero_mode_index, 0, 0].add(
            self.horizontal_scale.astype(dtype) * pressure_gradient[0].astype(dtype)
        )
        effective_rhs = effective_rhs.at[self.zero_mode_index, 0, 2].add(
            self.horizontal_scale.astype(dtype) * pressure_gradient[1].astype(dtype)
        )
        momentum_chebyshev = helmholtz_velocity + pressure_gradient_modal - effective_rhs
        momentum_c2 = oe.contract(
            "ij,bjc->bic",
            self.chebyshev_to_c2.astype(dtype),
            momentum_chebyshev,
            backend="jax",
        )
        horizontal_x_residual = (
            jnp.zeros((batch_count, count - 1), dtype=dtype)
            .at[:, : count - 2]
            .set(momentum_c2[:, : count - 2, 0])
        )
        horizontal_z_residual = (
            jnp.zeros((batch_count, count - 1), dtype=dtype)
            .at[:, : count - 2]
            .set(momentum_c2[:, : count - 2, 2])
        )
        vertical_c4 = oe.contract(
            "ij,bj->bi",
            self.chebyshev_to_c4.astype(dtype),
            momentum_chebyshev[..., 1],
            backend="jax",
        )
        vertical_residual = (
            jnp.zeros((batch_count, count - 1), dtype=dtype)
            .at[:, : count - 4]
            .set(vertical_c4[:, : count - 4])
        )
        vertical_c1_zero = oe.contract(
            "ij,j->i",
            self.chebyshev_to_c1.astype(dtype),
            momentum_chebyshev[self.zero_mode_index, :, 1],
            backend="jax",
        )
        vertical_residual = vertical_residual.at[self.zero_mode_index].set(
            vertical_c1_zero[: count - 1]
        )
        residual = jnp.stack(
            (horizontal_x_residual, vertical_residual, horizontal_z_residual), axis=-1
        ).reshape((batch_count, -1))

        failed = helmholtz_failed
        failed = failed.at[self.nonzero_mode_indices].set(
            failed[self.nonzero_mode_indices] | biharmonic_failed
        )
        failed = failed.at[self.zero_mode_index].set(
            failed[self.zero_mode_index]
            | pressure_failed[0]
            | self.bulk_influence_failed
            | schur_failed
        )
        failed = failed | jnp.any(~jnp.isfinite(residual), axis=-1)
        return velocity, pressure, residual, failed, pressure_gradient

    def _bulk_velocity(self, velocity_u: Array, velocity_w: Array, /) -> Array:
        weights = self.quadrature_weights.astype(
            velocity_u.dtype
        ) @ self.synthesis.astype(velocity_u.dtype)
        denominator = self.wall_length.astype(
            velocity_u.dtype
        ) * self.horizontal_scale.astype(velocity_u.dtype)
        return jnp.asarray(
            (weights @ velocity_u / denominator, weights @ velocity_w / denominator)
        ).real


def _conversion_matrix(count: int, order: int, /) -> np.ndarray:
    degrees = np.arange(count, dtype=float)
    if order == 0:
        diagonal = np.where(degrees == 0.0, 1.0, 0.5)
    else:
        diagonal = float(order) / (degrees + float(order))
    conversion = np.diag(diagonal)
    columns = np.arange(2, count)
    conversion[columns - 2, columns] = -diagonal[columns]
    return conversion


def _upper_bands(matrix: np.ndarray, upper_bandwidth: int, /) -> np.ndarray:
    count = int(matrix.shape[-1])
    batch_shape = matrix.shape[:-2]
    bands = np.zeros(batch_shape + (upper_bandwidth + 1, count), dtype=matrix.dtype)
    for offset in range(upper_bandwidth + 1):
        diagonal = np.diagonal(matrix, offset=offset, axis1=-2, axis2=-1)
        bands[..., upper_bandwidth - offset, offset:] = diagonal
    return bands


def _replace_tau_rows(
    bands: np.ndarray,
    upper_bandwidth: int,
    tau_start: int,
    rank: int,
    /,
) -> np.ndarray:
    updated = np.array(bands, copy=True)
    count = int(updated.shape[-1])
    for row in range(tau_start, tau_start + rank):
        for offset in range(min(upper_bandwidth, count - row - 1) + 1):
            updated[..., upper_bandwidth - offset, row + offset] = 0.0
        updated[..., upper_bandwidth, row] = 1.0
    return updated


def _array_bytes(*arrays: Array) -> int:
    return sum(int(array.nbytes) for array in arrays)


def _tau_factor_estimate(
    batch_size: int,
    count: int,
    upper_bandwidth: int,
    rank: int,
    complex_itemsize: int,
    real_itemsize: int,
    /,
) -> int:
    return batch_size * (
        (upper_bandwidth + 1) * count * complex_itemsize
        + (count + rank) * complex_itemsize
        + count * np.dtype(np.int32).itemsize
        + np.dtype(np.bool_).itemsize
        + real_itemsize
        + count * rank * complex_itemsize
        + rank * rank * complex_itemsize
        + rank * np.dtype(np.int32).itemsize
        + np.dtype(np.bool_).itemsize
    )


def _prepare_batched_tau_solve(
    base_bands: np.ndarray,
    desired_rows: np.ndarray,
    tau_start: int,
    /,
    *,
    upper_bandwidth: int,
    residual_tolerance: float,
) -> _PreparedBatchedTauSolve:
    batch_size, _, count = base_bands.shape
    rank = int(desired_rows.shape[1])
    dtype = jnp.asarray(base_bands).dtype
    row_norm = np.max(np.abs(desired_rows), axis=-1)
    if np.any(row_norm <= 0.0):
        raise ValueError("Tau constraint rows must be nonzero.")
    tau_rhs_scale = (1.0 / row_norm).astype(base_bands.dtype)
    column_norm = np.maximum(
        np.max(np.abs(base_bands), axis=1),
        np.max(
            np.abs(desired_rows * tau_rhs_scale[:, :, None]),
            axis=1,
        ),
    )
    column_scale = np.where(column_norm > 0.0, 1.0 / column_norm, 1.0).astype(
        base_bands.dtype
    )
    scaled_bands = base_bands * column_scale[:, None, :]
    scaled_desired_rows = (
        desired_rows * tau_rhs_scale[:, :, None] * column_scale[:, None, :]
    )
    space = ArraySpace((count,), dtype=dtype)
    base_operator = BandedLinearOperator(
        jnp.asarray(scaled_bands),
        lower_bandwidth=0,
        upper_bandwidth=upper_bandwidth,
        space=space,
    )
    policy = LinearSolvePolicy(StructuredDirect(), failure=FailurePolicy("status"))
    base = prepare_linear_solve(LinearSystem(base_operator), policy)
    tau_columns = jnp.zeros((batch_size, count, rank), dtype=dtype)
    rows = jnp.arange(tau_start, tau_start + rank)
    tau_columns = tau_columns.at[:, rows, jnp.arange(rank)].set(1.0)
    inverse_result = solve_linear(base, tau_columns, rhs_layout=RHSLayout((rank,)))
    identity_rows = np.zeros((batch_size, rank, count), dtype=base_bands.dtype)
    tau_indices = np.arange(tau_start, tau_start + rank)
    identity_rows[:, np.arange(rank), tau_indices] = column_scale[:, tau_indices]
    delta_rows = jnp.asarray(scaled_desired_rows - identity_rows, dtype=dtype)
    correction_matrix = jnp.eye(rank, dtype=dtype)[None, ...] + oe.contract(
        "brn,bns->brs", delta_rows, inverse_result.value, backend="jax"
    )
    correction_space = ArraySpace((rank,), dtype=dtype)
    correction_operator = DenseLinearOperator(
        correction_matrix, source=correction_space, target=correction_space
    )
    correction = prepare_linear_solve(
        LinearSystem(correction_operator),
        LinearSolvePolicy(DenseLU(), failure=FailurePolicy("status")),
    )
    base_state = base.state.prepared
    correction_state = correction.state
    correction_scale = jnp.maximum(jnp.max(jnp.abs(correction_matrix)), 1.0)
    correction_margin = jnp.min(
        jnp.abs(jnp.diagonal(correction_state.factor, axis1=-2, axis2=-1))
        / correction_scale
    )
    pivot_margin = float(jnp.minimum(jnp.min(base_state.pivot_margin), correction_margin))
    operator_bytes = _array_bytes(
        base_operator.bands,
        delta_rows,
        correction_matrix,
        jnp.asarray(column_scale),
        jnp.asarray(tau_rhs_scale),
    )
    factor_bytes = _array_bytes(
        base_state.factor,
        base_state.pivots,
        base_state.singular,
        base_state.pivot_margin,
        inverse_result.value,
        correction_state.factor,
        correction_state.pivots,
        correction_state.singular,
    )
    preparation_failed = (
        jnp.asarray(base_state.singular)
        | jnp.asarray(correction_state.singular)
        | jnp.any(~jnp.isfinite(inverse_result.value), axis=(-2, -1))
        | jnp.any(~jnp.isfinite(correction_matrix), axis=(-2, -1))
    )
    return _PreparedBatchedTauSolve(
        base=base,
        correction=correction,
        delta_rows=delta_rows,
        inverse_tau_columns=inverse_result.value,
        column_scale=jnp.asarray(column_scale),
        tau_rhs_scale=jnp.asarray(tau_rhs_scale),
        preparation_failed=preparation_failed,
        batch_size=batch_size,
        size=count,
        tau_start=tau_start,
        rank=rank,
        lower_bandwidth=0,
        upper_bandwidth=upper_bandwidth,
        operator_bytes=operator_bytes,
        factor_bytes=factor_bytes,
        pivot_margin=pivot_margin,
        residual_tolerance=float(residual_tolerance),
    )


def prepare_ultraspherical_channel(
    plan,
    shift: Array,
    synthesis: Array,
    quadrature_weights: Array,
    streamwise_wavenumbers: Array,
    spanwise_wavenumbers: Array,
    zero_mode_index: int,
    /,
) -> PreparedUltrasphericalChannel:
    count = int(synthesis.shape[0])
    dtype = jnp.result_type(synthesis.dtype, 1j)
    complex_dtype = np.dtype(dtype)
    real_dtype = np.dtype(jnp.empty((), dtype=dtype).real.dtype)
    synthesis_values = np.asarray(synthesis)
    if np.any(np.imag(synthesis_values) != 0.0):
        raise ValueError("Chebyshev synthesis data must be real-valued.")
    synthesis_host = np.asarray(np.real(synthesis_values), dtype=float)
    basis_scale = np.asarray(synthesis_host[-1], dtype=float)
    if np.any(np.abs(basis_scale) <= np.finfo(float).eps):
        raise ValueError("Chebyshev synthesis scaling is singular.")

    s01 = _conversion_matrix(count, 0) * basis_scale[None, :]
    s12 = _conversion_matrix(count, 1)
    s23 = _conversion_matrix(count, 2)
    s34 = _conversion_matrix(count, 3)
    s02 = s12 @ s01
    s24 = s34 @ s23
    s04 = s24 @ s02
    wall_scale = 2.0 / float(plan.discretization.axes[1].length)
    derivative_two = np.zeros((count, count), dtype=float)
    degrees_two = np.arange(2, count)
    derivative_two[degrees_two - 2, degrees_two] = (
        2.0 * degrees_two * wall_scale**2 * basis_scale[degrees_two]
    )
    derivative_four = np.zeros((count, count), dtype=float)
    degrees_four = np.arange(4, count)
    derivative_four[degrees_four - 4, degrees_four] = (
        48.0 * degrees_four * wall_scale**4 * basis_scale[degrees_four]
    )
    derivative_values = np.asarray(plan.discretization.axes[1].derivative_matrix)
    if np.any(np.imag(derivative_values) != 0.0):
        raise ValueError("Chebyshev derivative data must be real-valued.")
    modal_derivative = np.asarray(np.real(derivative_values), dtype=float)

    kx_flat = np.asarray(streamwise_wavenumbers).reshape((-1,))
    kz_flat = np.asarray(spanwise_wavenumbers).reshape((-1,))
    wave_square = kx_flat * kx_flat + kz_flat * kz_flat
    nonzero_indices = np.flatnonzero(wave_square != 0.0).astype(np.int32)
    horizontal_batch_size = int(wave_square.size)
    nonzero_batch_size = int(nonzero_indices.size)
    complex_itemsize = complex_dtype.itemsize
    real_itemsize = real_dtype.itemsize
    helmholtz_bandwidth = min(4, count - 1)
    biharmonic_bandwidth = min(8, count - 1)
    estimated_factor_bytes = (
        _tau_factor_estimate(
            horizontal_batch_size,
            count,
            helmholtz_bandwidth,
            2,
            complex_itemsize,
            real_itemsize,
        )
        + _tau_factor_estimate(
            nonzero_batch_size,
            count,
            biharmonic_bandwidth,
            4,
            complex_itemsize,
            real_itemsize,
        )
        + _tau_factor_estimate(1, count, 0, 1, complex_itemsize, real_itemsize)
        + count * complex_itemsize
        + 8 * real_itemsize
        + 2 * np.dtype(np.int32).itemsize
    )
    if estimated_factor_bytes > plan.maximum_factor_bytes:
        raise ValueError(
            f"Ultraspherical channel factors require at least "
            f"{estimated_factor_bytes} bytes, exceeding "
            f"maximum_factor_bytes={plan.maximum_factor_bytes}."
        )

    s02_bands = _upper_bands(s02, helmholtz_bandwidth)
    derivative_two_bands = _upper_bands(derivative_two, helmholtz_bandwidth)
    helmholtz_bands = (
        (float(shift) + float(plan.viscosity) * wave_square[:, None, None])
        * s02_bands[None, ...]
        - float(plan.viscosity) * derivative_two_bands[None, ...]
    ).astype(complex_dtype)
    helmholtz_bands = _replace_tau_rows(
        helmholtz_bands, helmholtz_bandwidth, count - 2, 2
    )
    helmholtz_desired = np.broadcast_to(
        synthesis_host[[0, -1]][None, ...], (horizontal_batch_size, 2, count)
    ).astype(complex_dtype)
    helmholtz = _prepare_batched_tau_solve(
        helmholtz_bands,
        helmholtz_desired,
        count - 2,
        upper_bandwidth=helmholtz_bandwidth,
        residual_tolerance=plan.constraint_tolerance,
    )

    s24_derivative_two = s24 @ derivative_two
    s04_bands = _upper_bands(s04, biharmonic_bandwidth)
    s24_derivative_two_bands = _upper_bands(s24_derivative_two, biharmonic_bandwidth)
    derivative_four_bands = _upper_bands(derivative_four, biharmonic_bandwidth)
    nonzero_wave_square = wave_square[nonzero_indices]
    biharmonic_bands = (
        -float(plan.viscosity) * derivative_four_bands[None, ...]
        + (
            float(shift)
            + 2.0 * float(plan.viscosity) * nonzero_wave_square[:, None, None]
        )
        * s24_derivative_two_bands[None, ...]
        - (
            float(shift) * nonzero_wave_square[:, None, None]
            + float(plan.viscosity) * nonzero_wave_square[:, None, None] ** 2
        )
        * s04_bands[None, ...]
    ).astype(complex_dtype)
    biharmonic_bands = _replace_tau_rows(
        biharmonic_bands, biharmonic_bandwidth, count - 4, 4
    )
    derivative_traces = synthesis_host[[0, -1]] @ modal_derivative
    biharmonic_desired = np.broadcast_to(
        np.concatenate((synthesis_host[[0, -1]], derivative_traces), axis=0)[None, ...],
        (nonzero_batch_size, 4, count),
    ).astype(complex_dtype)
    biharmonic = _prepare_batched_tau_solve(
        biharmonic_bands,
        biharmonic_desired,
        count - 4,
        upper_bandwidth=biharmonic_bandwidth,
        residual_tolerance=plan.constraint_tolerance,
    )

    pressure_diagonal = np.concatenate(
        (wall_scale * np.arange(1, count, dtype=float), np.ones((1,)))
    )
    pressure_bands = pressure_diagonal.reshape((1, 1, count)).astype(complex_dtype)
    pressure_gauge = np.asarray(quadrature_weights) @ synthesis_host
    pressure_permutation = np.concatenate(
        (np.arange(1, count), np.zeros((1,), dtype=int))
    )
    pressure_desired = pressure_gauge[pressure_permutation][None, None, :].astype(
        complex_dtype
    )
    pressure_recovery = _prepare_batched_tau_solve(
        pressure_bands,
        pressure_desired,
        count - 1,
        upper_bandwidth=0,
        residual_tolerance=plan.constraint_tolerance,
    )

    horizontal_scale = jnp.sqrt(
        plan.discretization.axes[0].length * plan.discretization.axes[2].length
    )
    influence_rhs = (
        jnp.zeros((horizontal_batch_size, count, 1), dtype=dtype)
        .at[zero_mode_index, : count - 2, 0]
        .set(
            horizontal_scale.astype(dtype) * jnp.asarray(s02[: count - 2, 0], dtype=dtype)
        )
    )
    influence_solution, influence_failed = helmholtz.solve(influence_rhs)
    bulk_influence = influence_solution[zero_mode_index, :, 0]
    bulk_weights = jnp.asarray(quadrature_weights, dtype=dtype) @ jnp.asarray(
        synthesis, dtype=dtype
    )
    bulk_denominator = plan.discretization.axes[1].length.astype(
        dtype
    ) * horizontal_scale.astype(dtype)
    bulk_response_scalar = jnp.real(bulk_weights @ bulk_influence / bulk_denominator)
    bulk_response = bulk_response_scalar * jnp.eye(2, dtype=bulk_response_scalar.dtype)
    schur_scale = jnp.maximum(jnp.abs(bulk_response_scalar), 1.0)
    if float(jnp.abs(bulk_response_scalar)) <= float(
        jnp.finfo(bulk_response.dtype).eps * schur_scale
    ):
        raise ValueError("Channel bulk-flux Schur complement is singular.")
    schur_space = ArraySpace((2,), dtype=bulk_response.dtype)
    bulk_schur = prepare_linear_solve(
        LinearSystem(
            DenseLinearOperator(bulk_response, source=schur_space, target=schur_space)
        ),
        LinearSolvePolicy(DenseLU(), failure=FailurePolicy("status")),
    )
    schur_factorization = bulk_schur.state

    operator_bytes = (
        helmholtz.operator_bytes
        + biharmonic.operator_bytes
        + pressure_recovery.operator_bytes
        + _array_bytes(bulk_response)
    )
    factor_bytes = (
        helmholtz.factor_bytes
        + biharmonic.factor_bytes
        + pressure_recovery.factor_bytes
        + _array_bytes(
            bulk_influence,
            schur_factorization.factor,
            schur_factorization.pivots,
            schur_factorization.singular,
        )
    )
    if factor_bytes > plan.maximum_factor_bytes:
        raise RuntimeError(
            "Ultraspherical channel factor preflight underestimated retained storage."
        )
    shared_basis_bytes = _array_bytes(
        jnp.asarray(s01, dtype=dtype),
        jnp.asarray(s02, dtype=dtype),
        jnp.asarray(s04, dtype=dtype),
        jnp.asarray(modal_derivative, dtype=dtype),
        jnp.asarray(synthesis, dtype=dtype),
        jnp.asarray(quadrature_weights, dtype=dtype),
        jnp.asarray(kx_flat),
        jnp.asarray(kz_flat),
        jnp.asarray(nonzero_indices),
    )
    persistent_bytes = shared_basis_bytes + operator_bytes + factor_bytes
    workspace_bytes = int(
        complex_itemsize
        * (
            20 * horizontal_batch_size * count
            + 8 * nonzero_batch_size * count
            + 8 * horizontal_batch_size
        )
    )
    preparation_bytes = persistent_bytes + workspace_bytes
    pivot_margin = min(
        helmholtz.pivot_margin,
        biharmonic.pivot_margin,
        pressure_recovery.pivot_margin,
    )
    return PreparedUltrasphericalChannel(
        helmholtz=helmholtz,
        biharmonic=biharmonic,
        pressure_recovery=pressure_recovery,
        chebyshev_to_c1=jnp.asarray(s01, dtype=dtype),
        chebyshev_to_c2=jnp.asarray(s02, dtype=dtype),
        chebyshev_to_c4=jnp.asarray(s04, dtype=dtype),
        modal_derivative=jnp.asarray(modal_derivative, dtype=dtype),
        nonzero_mode_indices=jnp.asarray(nonzero_indices),
        streamwise_wavenumbers=jnp.asarray(kx_flat),
        spanwise_wavenumbers=jnp.asarray(kz_flat),
        synthesis=synthesis,
        quadrature_weights=quadrature_weights,
        wall_length=plan.discretization.axes[1].length,
        horizontal_scale=horizontal_scale,
        shift=shift,
        viscosity=plan.viscosity,
        bulk_influence=bulk_influence,
        bulk_influence_failed=influence_failed[zero_mode_index],
        bulk_schur=bulk_schur,
        zero_mode_index=zero_mode_index,
        mode_count=count,
        horizontal_batch_size=horizontal_batch_size,
        lower_bandwidth=0,
        upper_bandwidth=8,
        correction_rank=4,
        shared_basis_bytes=shared_basis_bytes,
        operator_bytes=operator_bytes,
        factor_bytes=factor_bytes,
        workspace_bytes=workspace_bytes,
        persistent_bytes=persistent_bytes,
        preparation_bytes=preparation_bytes,
        pivot_margin=pivot_margin,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-pressure-eliminated-ultraspherical-channel",
                "plan": plan.plan_id,
                "shift": float(shift),
                "mode_count": count,
                "bandwidth": (0, 8),
                "tau_ranks": (2, 4, 1),
            }
        ),
    )


__all__ = ["PreparedUltrasphericalChannel", "prepare_ultraspherical_channel"]
