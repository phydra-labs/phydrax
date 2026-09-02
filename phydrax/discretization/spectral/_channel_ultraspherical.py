#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

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
    BasePlusLowRankLinearOperator,
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    LowRankResourcePolicy,
    LowRankSolvePolicy,
    prepare as prepare_linear_solve,
    prepare_low_rank_solve,
    PreparedLinearSolve,
    PreparedLowRankSolve,
    RHSLayout,
    solve as solve_linear,
    solve_low_rank,
    StructuredDirect,
)


class PreparedUltrasphericalChannel(StrictModule, NonTrainableState):
    """Fixed-band Stokes blocks with physical tau constraints and a gauge stage."""

    solves: tuple[PreparedLowRankSolve, ...]
    operators: tuple[BasePlusLowRankLinearOperator, ...]
    chebyshev_to_c2: Array
    momentum_rows: Array
    constraint_rows: Array
    streamwise_wavenumbers: Array
    spanwise_wavenumbers: Array
    synthesis: Array
    quadrature_weights: Array
    wall_length: Array
    horizontal_scale: Array
    bulk_influence: Array
    bulk_schur: PreparedLinearSolve
    zero_mode_index: int = eqx.field(static=True)
    mode_count: int = eqx.field(static=True)
    lower_bandwidth: int = eqx.field(static=True)
    upper_bandwidth: int = eqx.field(static=True)
    factor_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
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
    ) -> tuple[Array, Array, Array, Array]:
        batch_count = int(modal_rhs.shape[0])
        count = self.mode_count
        rows = np.asarray(self.momentum_rows)
        constraint_rows = np.asarray(self.constraint_rows)
        solutions = []
        residuals = []
        failures = []
        pressure_gradient = jnp.asarray(mean_values, dtype=modal_rhs.real.dtype)
        walls = jnp.asarray(
            (
                lower_wall_velocity[0],
                upper_wall_velocity[0],
                lower_wall_velocity[1],
                upper_wall_velocity[1],
                lower_wall_velocity[2],
                upper_wall_velocity[2],
                0.0,
            ),
            dtype=modal_rhs.dtype,
        )
        for mode_index in range(batch_count):
            converted = oe.contract(
                "ij,jc->ic",
                self.chebyshev_to_c2,
                modal_rhs[mode_index],
                backend="jax",
            )
            rhs = jnp.zeros((4 * count,), dtype=modal_rhs.dtype)
            for degree in range(count - 2):
                rhs = rhs.at[4 * degree : 4 * degree + 3].set(converted[degree])
            if mode_index == self.zero_mode_index:
                constraint_rhs = walls * self.horizontal_scale.astype(modal_rhs.dtype)
                constraint_rhs = constraint_rhs.at[3].set(
                    self.synthesis[-1].astype(modal_rhs.dtype)
                    @ modal_rhs[mode_index, :, 1]
                )
                rhs = rhs.at[constraint_rows].set(constraint_rhs)
            gx = jnp.zeros_like(rhs).at[0].set(self.horizontal_scale.astype(rhs.dtype))
            gz = jnp.zeros_like(rhs).at[2].set(self.horizontal_scale.astype(rhs.dtype))
            if mode_index == self.zero_mode_index and mean_kind == "bulk_flux":
                solved = solve_low_rank(self.solves[mode_index], rhs)
                base = solved.value
                base_bulk = self._bulk_velocity(base)
                target = mean_values.astype(base_bulk.dtype) - base_bulk
                schur_result = solve_linear(self.bulk_schur, target)
                pressure_gradient = schur_result.value
                solution = base + self.bulk_influence @ pressure_gradient.astype(
                    self.bulk_influence.dtype
                )
                effective_rhs = (
                    rhs
                    + pressure_gradient[0].astype(rhs.dtype) * gx
                    + pressure_gradient[1].astype(rhs.dtype) * gz
                )
            else:
                if (
                    mode_index == self.zero_mode_index
                    and mean_kind == "pressure_gradient"
                ):
                    rhs = (
                        rhs
                        + mean_values[0].astype(rhs.dtype) * gx
                        + mean_values[1].astype(rhs.dtype) * gz
                    )
                solved = solve_low_rank(self.solves[mode_index], rhs)
                solution = solved.value
                effective_rhs = rhs
            residual = self.operators[mode_index].mv(solution) - effective_rhs
            failed = jnp.any(~jnp.isfinite(solution)) | jnp.any(~jnp.isfinite(residual))
            solutions.append(solution)
            residuals.append(residual)
            failures.append(failed)
        return (
            jnp.stack(tuple(solutions)),
            jnp.stack(tuple(residuals)),
            jnp.stack(tuple(failures)),
            pressure_gradient,
        )

    def _bulk_velocity(self, solution: Array, /) -> Array:
        fields = solution.reshape((self.mode_count, 4))
        weights = self.quadrature_weights.astype(solution.dtype) @ self.synthesis
        denominator = self.wall_length.astype(
            solution.dtype
        ) * self.horizontal_scale.astype(solution.dtype)
        return jnp.asarray(
            (
                weights @ fields[:, 0] / denominator,
                weights @ fields[:, 2] / denominator,
            )
        ).real


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
    s01 = np.zeros((count, count), dtype=float)
    for degree in range(count):
        s01[degree, degree] += 1.0 if degree == 0 else 0.5
        if degree >= 2:
            s01[degree - 2, degree] -= 0.5
    s12 = np.zeros((count, count), dtype=float)
    for degree in range(count):
        coefficient = 1.0 / (degree + 1.0)
        s12[degree, degree] += coefficient
        if degree >= 2:
            s12[degree - 2, degree] -= coefficient
    s02 = s12 @ s01
    basis_scale = np.asarray(synthesis[-1], dtype=float)
    if np.any(np.abs(basis_scale) <= np.finfo(float).eps):
        raise ValueError("Chebyshev synthesis scaling is singular.")
    s01 = s01 * basis_scale[None, :]
    s02 = s02 * basis_scale[None, :]
    derivative_one = np.zeros((count, count), dtype=float)
    derivative_two = np.zeros((count, count), dtype=float)
    for degree in range(1, count):
        derivative_one[degree - 1, degree] = degree
    for degree in range(2, count):
        derivative_two[degree - 2, degree] = 2.0 * degree
    derivative_one *= basis_scale[None, :]
    derivative_two *= basis_scale[None, :]
    wall_scale = 2.0 / float(plan.discretization.axes[1].length)
    derivative_one *= wall_scale
    derivative_two *= wall_scale**2
    derivative_one_c2 = s12 @ derivative_one
    modal_derivative = np.asarray(
        plan.discretization.axes[1].derivative_matrix, dtype=float
    )
    constraint_rows = np.asarray(
        (
            4 * (count - 2),
            4 * (count - 2) + 1,
            4 * (count - 2) + 2,
            4 * (count - 1),
            4 * (count - 1) + 1,
            4 * (count - 1) + 2,
            4 * (count - 1) + 3,
        ),
        dtype=np.int32,
    )
    # S02 reaches four degrees upward; interleaved pressure adds three columns.
    lower_bandwidth, upper_bandwidth = 3, 19
    solves = []
    operators = []
    minimum_pivot_margin = math.inf
    factor_bytes = 0
    kx_flat = np.asarray(streamwise_wavenumbers).reshape((-1,))
    kz_flat = np.asarray(spanwise_wavenumbers).reshape((-1,))
    for mode_index, (kx, kz) in enumerate(zip(kx_flat, kz_flat, strict=True)):
        size = 4 * count
        bands = np.zeros(
            (lower_bandwidth + upper_bandwidth + 1, size), dtype=np.complex128
        )

        def add(row: int, column: int, value) -> None:
            offset = row - column
            if value != 0 and not (-upper_bandwidth <= offset <= lower_bandwidth):
                raise RuntimeError("Ultraspherical Stokes bulk exceeded fixed bandwidth.")
            if value != 0:
                bands[upper_bandwidth + offset, column] += value

        wave_square = float(kx * kx + kz * kz)
        helmholtz = (float(shift) + float(plan.viscosity) * wave_square) * s02 - float(
            plan.viscosity
        ) * derivative_two
        for degree in range(count - 2):
            for column_degree in range(count):
                for component in range(3):
                    add(
                        4 * degree + component,
                        4 * column_degree + component,
                        helmholtz[degree, column_degree],
                    )
                add(
                    4 * degree,
                    4 * column_degree + 3,
                    1j * kx * s02[degree, column_degree],
                )
                add(
                    4 * degree + 1,
                    4 * column_degree + 3,
                    derivative_one_c2[degree, column_degree],
                )
                add(
                    4 * degree + 2,
                    4 * column_degree + 3,
                    1j * kz * s02[degree, column_degree],
                )
        for degree in range(count - 1):
            for column_degree in range(count):
                add(
                    4 * degree + 3,
                    4 * column_degree,
                    1j * kx * s01[degree, column_degree],
                )
                add(
                    4 * degree + 3,
                    4 * column_degree + 1,
                    derivative_one[degree, column_degree],
                )
                add(
                    4 * degree + 3,
                    4 * column_degree + 2,
                    1j * kz * s01[degree, column_degree],
                )
        # Scalar-pivot banded LU cannot factor the saddle block's zero pressure
        # diagonal.  Give every retained divergence row a local unit pressure
        # pivot in the Woodbury base, then remove those pivots exactly with
        # staged update columns before imposing the physical tau rows.
        regularization_rows = tuple(4 * degree + 3 for degree in range(count - 1))
        for row in regularization_rows:
            add(row, row, 1.0)
        for row in constraint_rows:
            add(int(row), int(row), 1.0)
        desired = np.zeros((7, size), dtype=np.complex128)
        desired[0, 0::4] = np.asarray(synthesis[0])
        desired[1, 0::4] = np.asarray(synthesis[-1])
        desired[2, 1::4] = np.asarray(synthesis[0])
        desired[3, 1::4] = np.asarray(synthesis[-1])
        desired[4, 2::4] = np.asarray(synthesis[0])
        desired[5, 2::4] = np.asarray(synthesis[-1])
        desired[6, 3::4] = np.asarray(quadrature_weights) @ np.asarray(synthesis)
        correction_rank = 7 + len(regularization_rows)
        # At zero horizontal wave number, divergence and the lower normal wall
        # imply the upper normal value.  Replace that redundant wall row with
        # the upper endpoint normal-momentum equation, as in the dense oracle.
        if mode_index == zero_mode_index:
            modal_helmholtz = (
                float(shift) + float(plan.viscosity) * wave_square
            ) * np.eye(count) - float(plan.viscosity) * (
                modal_derivative @ modal_derivative
            )
            desired[3] = 0.0
            desired[3, 1::4] = np.asarray(synthesis[-1]) @ modal_helmholtz
            desired[3, 3::4] = np.asarray(synthesis[-1]) @ modal_derivative
        left = np.zeros((size, correction_rank), dtype=np.complex128)
        delta = np.zeros((correction_rank, size), dtype=np.complex128)
        delta[:7] = desired
        for rank_index, row in enumerate(constraint_rows):
            left[int(row), rank_index] = 1.0
            delta[rank_index, int(row)] -= 1.0
        for offset, row in enumerate(regularization_rows):
            rank_index = 7 + offset
            left[row, rank_index] = 1.0
            delta[rank_index, row] = -1.0
        space = ArraySpace((size,), dtype=dtype)
        base = BandedLinearOperator(
            jnp.asarray(bands, dtype=dtype),
            lower_bandwidth=lower_bandwidth,
            upper_bandwidth=upper_bandwidth,
            space=space,
        )
        operator = BasePlusLowRankLinearOperator(
            base,
            jnp.asarray(left, dtype=dtype),
            jnp.conj(jnp.asarray(delta.T, dtype=dtype)),
        )
        policy = LowRankSolvePolicy(
            LinearSolvePolicy(
                StructuredDirect(),
                failure=FailurePolicy("status"),
            ),
            base_nonsingularity="asserted",
            failure=FailurePolicy("status"),
            resources=LowRankResourcePolicy(
                max_rank=correction_rank,
                max_storage_bytes=plan.maximum_factor_bytes,
                max_workspace_bytes=plan.maximum_factor_bytes,
            ),
        )
        prepared = prepare_low_rank_solve(operator, policy)
        solves.append(prepared)
        operators.append(operator)
        base_factorization = prepared.base_prepared.state.prepared
        factor = base_factorization.factor
        diagonal = base_factorization.diagonal_index
        pivots = jnp.abs(factor[diagonal])
        scale = jnp.maximum(jnp.max(jnp.abs(base.bands)), 1.0)
        minimum_pivot_margin = min(minimum_pivot_margin, float(jnp.min(pivots) / scale))
        factor_bytes += sum(
            int(array.nbytes)
            for array in (
                factor,
                base_factorization.pivots,
                prepared.inverse_left_factor,
                prepared.correction_matrix,
                prepared.correction_lu,
                prepared.correction_pivots,
            )
        )
    zero_rhs_x = (
        jnp.zeros((4 * count,), dtype=dtype)
        .at[0]
        .set(
            jnp.sqrt(
                plan.discretization.axes[0].length * plan.discretization.axes[2].length
            ).astype(dtype)
        )
    )
    zero_rhs_z = (
        jnp.zeros((4 * count,), dtype=dtype)
        .at[2]
        .set(
            jnp.sqrt(
                plan.discretization.axes[0].length * plan.discretization.axes[2].length
            ).astype(dtype)
        )
    )
    bulk_influence = solve_low_rank(
        solves[zero_mode_index],
        jnp.stack((zero_rhs_x, zero_rhs_z), axis=-1),
        rhs_layout=RHSLayout((2,)),
    ).value
    bulk_weights = jnp.asarray(quadrature_weights, dtype=dtype) @ jnp.asarray(
        synthesis, dtype=dtype
    )
    bulk_denominator = plan.discretization.axes[1].length.astype(dtype) * jnp.sqrt(
        plan.discretization.axes[0].length * plan.discretization.axes[2].length
    ).astype(dtype)
    influence_fields = bulk_influence.reshape((count, 4, 2))
    bulk_response = jnp.stack(
        (
            bulk_weights @ influence_fields[:, 0, :] / bulk_denominator,
            bulk_weights @ influence_fields[:, 2, :] / bulk_denominator,
        ),
        axis=0,
    ).real
    schur_determinant = (
        bulk_response[0, 0] * bulk_response[1, 1]
        - bulk_response[0, 1] * bulk_response[1, 0]
    )
    schur_scale = jnp.maximum(jnp.max(jnp.abs(bulk_response)), 1.0)
    if float(jnp.abs(schur_determinant)) <= float(
        jnp.finfo(bulk_response.dtype).eps * schur_scale**2
    ):
        raise ValueError("Channel bulk-flux Schur complement is singular.")
    schur_space = ArraySpace((2,), dtype=bulk_response.dtype)
    bulk_schur = prepare_linear_solve(
        LinearSystem(
            DenseLinearOperator(
                bulk_response,
                source=schur_space,
                target=schur_space,
            )
        ),
        LinearSolvePolicy(
            DenseLU(),
            failure=FailurePolicy("status"),
        ),
    )
    schur_factorization = bulk_schur.state
    factor_bytes += sum(
        int(array.nbytes)
        for array in (
            bulk_influence,
            bulk_response,
            schur_factorization.factor,
            schur_factorization.pivots,
        )
    )
    if factor_bytes > plan.maximum_factor_bytes:
        raise ValueError("Ultraspherical channel factors exceed maximum_factor_bytes.")
    return PreparedUltrasphericalChannel(
        solves=tuple(solves),
        operators=tuple(operators),
        chebyshev_to_c2=jnp.asarray(s02, dtype=dtype),
        momentum_rows=jnp.arange(count - 2, dtype=jnp.int32),
        constraint_rows=jnp.asarray(constraint_rows),
        streamwise_wavenumbers=streamwise_wavenumbers,
        spanwise_wavenumbers=spanwise_wavenumbers,
        synthesis=synthesis,
        quadrature_weights=quadrature_weights,
        wall_length=plan.discretization.axes[1].length,
        horizontal_scale=jnp.sqrt(
            plan.discretization.axes[0].length * plan.discretization.axes[2].length
        ),
        bulk_influence=bulk_influence,
        bulk_schur=bulk_schur,
        zero_mode_index=zero_mode_index,
        mode_count=count,
        lower_bandwidth=lower_bandwidth,
        upper_bandwidth=upper_bandwidth,
        factor_bytes=factor_bytes,
        workspace_bytes=max(4 * count * 3 * np.dtype(dtype).itemsize, factor_bytes),
        pivot_margin=minimum_pivot_margin,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-ultraspherical-channel",
                "plan": plan.plan_id,
                "shift": float(shift),
                "mode_count": count,
                "bandwidth": (lower_bandwidth, upper_bandwidth),
            }
        ),
    )


__all__ = ["PreparedUltrasphericalChannel", "prepare_ultraspherical_channel"]
