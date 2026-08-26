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

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg._local_blocks import (
    LocalBlockFactorization,
    prepare_local_block_factorization,
    solve_local_blocks,
)
from ._space import TensorSpectralDiscretization


ChannelMeanConstraintKind: TypeAlias = Literal["pressure_gradient", "bulk_flux"]


class ChannelMeanConstraint(StrictModule, NonTrainableState):
    """Streamwise/spanwise channel mean-flow control."""

    values: Array
    kind: ChannelMeanConstraintKind = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: ChannelMeanConstraintKind = "pressure_gradient",
        values: ArrayLike = (0.0, 0.0),
        /,
    ):
        if kind not in ("pressure_gradient", "bulk_flux"):
            raise ValueError("Unknown channel mean constraint kind.")
        values_ = jnp.asarray(values, dtype=float)
        if values_.shape != (2,) or not bool(jnp.all(jnp.isfinite(values_))):
            raise ValueError("Channel mean constraint values must have shape (2,).")
        self.values = values_
        self.kind = kind
        self.constraint_id = canonical_fingerprint(
            {
                "kind": "channel-mean-constraint-v1",
                "mode": kind,
                "values": [float(values_[0]), float(values_[1])],
            }
        )


class ChannelStokesPlan(StrictModule, NonTrainableState):
    """Dense primitive-variable Fourier–Chebyshev–Fourier Stokes plan."""

    discretization: TensorSpectralDiscretization
    viscosity: Array
    lower_wall_velocity: Array
    upper_wall_velocity: Array
    mean_constraint: ChannelMeanConstraint
    maximum_factor_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: TensorSpectralDiscretization,
        viscosity: ArrayLike,
        /,
        *,
        lower_wall_velocity: ArrayLike = (0.0, 0.0, 0.0),
        upper_wall_velocity: ArrayLike = (0.0, 0.0, 0.0),
        mean_constraint: ChannelMeanConstraint | None = None,
        maximum_factor_bytes: int = 512 * 1024**2,
    ):
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        families = tuple(axis.family for axis in discretization.axes)
        if families != ("fourier", "chebyshev", "fourier"):
            raise ValueError(
                "Channel Stokes requires Fourier x Chebyshev x Fourier axes."
            )
        viscosity_ = jnp.asarray(viscosity, dtype=float)
        if viscosity_.shape != () or not bool(
            jnp.isfinite(viscosity_) & (viscosity_ > 0.0)
        ):
            raise ValueError("viscosity must be one finite positive scalar.")
        lower = jnp.asarray(lower_wall_velocity, dtype=float)
        upper = jnp.asarray(upper_wall_velocity, dtype=float)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not bool(jnp.all(jnp.isfinite(lower)) & jnp.all(jnp.isfinite(upper)))
        ):
            raise ValueError("Wall velocities must be finite vectors with shape (3,).")
        constraint = (
            ChannelMeanConstraint() if mean_constraint is None else mean_constraint
        )
        if not isinstance(constraint, ChannelMeanConstraint):
            raise TypeError("mean_constraint must be ChannelMeanConstraint or None.")
        maximum = int(maximum_factor_bytes)
        if maximum <= 0:
            raise ValueError("maximum_factor_bytes must be positive.")
        identifier = canonical_fingerprint(
            {
                "kind": "channel-stokes-plan-v1",
                "discretization": discretization.prepared_id,
                "viscosity": float(viscosity_),
                "lower_wall": [float(value) for value in lower],
                "upper_wall": [float(value) for value in upper],
                "mean_constraint": constraint.constraint_id,
                "maximum_factor_bytes": maximum,
            }
        )
        self.discretization = discretization
        self.viscosity = viscosity_
        self.lower_wall_velocity = lower
        self.upper_wall_velocity = upper
        self.mean_constraint = constraint
        self.maximum_factor_bytes = maximum
        self.plan_id = identifier

    def prepare(self, shift: ArrayLike, /) -> PreparedChannelStokesSolver:
        return PreparedChannelStokesSolver(self, shift)


class ChannelStokesDiagnostics(StrictModule):
    momentum_constraint_residual: Array
    divergence_norm: Array
    wall_residual: Array
    pressure_gauge_residual: Array
    bulk_velocity: Array
    failed: Array


class ChannelStokesSolveResult(StrictModule):
    velocity: Array
    pressure: Array
    pressure_gradient: Array
    diagnostics: ChannelStokesDiagnostics
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return ~self.diagnostics.failed


class PreparedChannelStokesSolver(StrictModule, NonTrainableState):
    """Prepared dense modewise constrained Stokes factorization."""

    plan: ChannelStokesPlan
    shift: Array
    blocks: Array
    factorization: LocalBlockFactorization
    bulk_block: Array | None
    bulk_factorization: LocalBlockFactorization | None
    synthesis: Array
    derivative: Array
    quadrature_weights: Array
    streamwise_wavenumbers: Array
    spanwise_wavenumbers: Array
    horizontal_admissibility: Array
    horizontal_constant_scale: Array
    zero_mode_index: int = eqx.field(static=True)
    wall_normal_count: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ChannelStokesPlan, shift: ArrayLike, /):
        if not isinstance(plan, ChannelStokesPlan):
            raise TypeError("plan must be a ChannelStokesPlan.")
        shift_ = jnp.asarray(shift, dtype=float)
        if shift_.shape != () or not bool(jnp.isfinite(shift_) & (shift_ > 0.0)):
            raise ValueError("shift must be one finite positive scalar.")
        discretization = plan.discretization
        x_axis, wall_axis, z_axis = discretization.axes
        if wall_axis.modal_transform is None or wall_axis.derivative_matrix is None:
            raise ValueError("Prepared Chebyshev axis lacks modal transform calculus.")
        synthesis = jnp.asarray(
            wall_axis.modal_transform.synthesis,
            dtype=jnp.dtype(discretization.plan.precision.coefficient_dtype),
        )
        derivative = jnp.asarray(
            wall_axis.derivative_matrix,
            dtype=synthesis.dtype,
        )
        mode_count = wall_axis.mode_count
        if mode_count < 4:
            raise ValueError("Channel Stokes requires at least four wall-normal modes.")
        real_dtype = jnp.empty((), dtype=synthesis.dtype).real.dtype
        kx = (
            2.0
            * jnp.asarray(jnp.pi, dtype=real_dtype)
            * x_axis.modes.mode_numbers.astype(real_dtype)
            / x_axis.length.astype(real_dtype)
        )
        kz = (
            2.0
            * jnp.asarray(jnp.pi, dtype=real_dtype)
            * z_axis.modes.mode_numbers.astype(real_dtype)
            / z_axis.length.astype(real_dtype)
        )
        kx_grid, kz_grid = jnp.meshgrid(kx, kz, indexing="ij")
        horizontal_scale = jnp.sqrt(x_axis.length * z_axis.length).astype(real_dtype)
        matrices = []
        zero_mode_index = -1
        for ix, kx_value in enumerate(np.asarray(kx)):
            for iz, kz_value in enumerate(np.asarray(kz)):
                matrix = _channel_mode_matrix(
                    synthesis,
                    derivative,
                    shift_,
                    plan.viscosity,
                    jnp.asarray(kx_value, dtype=real_dtype),
                    jnp.asarray(kz_value, dtype=real_dtype),
                    wall_axis.quadrature_weights,
                    zero_mode=(kx_value == 0.0 and kz_value == 0.0),
                )
                if kx_value == 0.0 and kz_value == 0.0:
                    zero_mode_index = ix * z_axis.mode_count + iz
                matrices.append(matrix)
        if zero_mode_index < 0:
            raise RuntimeError("Channel Fourier layout is missing its zero mode.")
        blocks = jnp.stack(tuple(matrices))
        bulk_block = None
        bulk_factorization = None
        if plan.mean_constraint.kind == "bulk_flux":
            bulk_block = _bulk_flux_block(
                blocks[zero_mode_index],
                synthesis,
                wall_axis.quadrature_weights,
                wall_axis.length,
                horizontal_scale,
            )
            bulk_factorization = prepare_local_block_factorization(bulk_block[None, ...])
        factor_entries = blocks.size + (0 if bulk_block is None else bulk_block.size)
        factor_bytes = int(factor_entries * blocks.dtype.itemsize * 2)
        if factor_bytes > plan.maximum_factor_bytes:
            raise ValueError(
                f"Channel Stokes factors require {factor_bytes} bytes, exceeding "
                f"maximum_factor_bytes={plan.maximum_factor_bytes}."
            )
        factorization = prepare_local_block_factorization(blocks)
        admissible = (~x_axis.modes.nyquist_mask)[:, None] & (~z_axis.modes.nyquist_mask)[
            None, :
        ]
        identifier = canonical_fingerprint(
            {
                "kind": "prepared-channel-stokes-v1",
                "plan": plan.plan_id,
                "shift": float(shift_),
                "block_shape": list(blocks.shape),
            }
        )
        self.plan = plan
        self.shift = shift_
        self.blocks = blocks
        self.factorization = factorization
        self.bulk_block = bulk_block
        self.bulk_factorization = bulk_factorization
        self.synthesis = synthesis
        self.derivative = derivative
        self.quadrature_weights = wall_axis.quadrature_weights
        self.streamwise_wavenumbers = kx_grid
        self.spanwise_wavenumbers = kz_grid
        self.horizontal_constant_scale = horizontal_scale
        self.horizontal_admissibility = admissible
        self.zero_mode_index = zero_mode_index
        self.wall_normal_count = mode_count
        self.block_size = 4 * mode_count
        self.prepared_id = identifier

    def solve(self, right_hand_side: ArrayLike, /) -> ChannelStokesSolveResult:
        value = self._validate_velocity(right_hand_side, "Stokes right-hand side")
        value = value * self.horizontal_admissibility[:, None, :, None]
        modal_modes = jnp.transpose(value, (0, 2, 1, 3)).reshape(
            (-1, self.wall_normal_count, 3)
        )
        interior = self.synthesis[1:-1]
        physical_rhs = oe.contract("ij,kjc->kic", interior, modal_modes, backend="jax")
        batch_rhs = jnp.zeros((modal_modes.shape[0], self.block_size), dtype=value.dtype)
        count = self.wall_normal_count
        interior_count = count - 2
        batch_rhs = batch_rhs.at[:, 0:interior_count].set(physical_rhs[..., 0])
        batch_rhs = batch_rhs.at[:, interior_count : 2 * interior_count].set(
            physical_rhs[..., 1]
        )
        batch_rhs = batch_rhs.at[:, 2 * interior_count : 3 * interior_count].set(
            physical_rhs[..., 2]
        )
        boundary_start = 3 * interior_count
        walls = jnp.concatenate(
            (self.plan.lower_wall_velocity, self.plan.upper_wall_velocity)
        )
        wall_rhs = self.horizontal_constant_scale.astype(value.dtype) * jnp.asarray(
            (
                walls[0],
                walls[3],
                walls[1],
                walls[4],
                walls[2],
                walls[5],
            ),
            dtype=value.dtype,
        )
        batch_rhs = batch_rhs.at[
            self.zero_mode_index, boundary_start : boundary_start + 6
        ].set(wall_rhs)
        zero_vertical_upper_rhs = oe.contract(
            "j,j->",
            self.synthesis[-1],
            modal_modes[self.zero_mode_index, :, 1],
            backend="jax",
        )
        batch_rhs = batch_rhs.at[self.zero_mode_index, boundary_start + 3].set(
            zero_vertical_upper_rhs
        )
        if self.plan.mean_constraint.kind == "pressure_gradient":
            pressure_gradient = self.plan.mean_constraint.values
            gradient = self.horizontal_constant_scale.astype(
                value.dtype
            ) * pressure_gradient.astype(value.dtype)
            batch_rhs = batch_rhs.at[self.zero_mode_index, 0:interior_count].add(
                gradient[0]
            )
            batch_rhs = batch_rhs.at[
                self.zero_mode_index,
                2 * interior_count : 3 * interior_count,
            ].add(gradient[1])
            solution, failed = solve_local_blocks(self.factorization, batch_rhs)
            residual = (
                oe.contract("kij,kj->ki", self.blocks, solution, backend="jax")
                - batch_rhs
            )
        else:
            if self.bulk_block is None or self.bulk_factorization is None:
                raise RuntimeError("Prepared bulk-flux factorization is missing.")
            solution, failed = solve_local_blocks(self.factorization, batch_rhs)
            augmented_rhs = jnp.concatenate(
                (
                    batch_rhs[self.zero_mode_index],
                    self.plan.mean_constraint.values.astype(value.dtype),
                )
            )[None, ...]
            augmented_solution, bulk_failed = solve_local_blocks(
                self.bulk_factorization, augmented_rhs
            )
            solution = solution.at[self.zero_mode_index].set(
                augmented_solution[0, : self.block_size]
            )
            pressure_gradient = jnp.real(augmented_solution[0, self.block_size :])
            residual = (
                oe.contract("kij,kj->ki", self.blocks, solution, backend="jax")
                - batch_rhs
            )
            augmented_residual = (
                self.bulk_block @ augmented_solution[0] - augmented_rhs[0]
            )
            residual = residual.at[self.zero_mode_index].set(
                augmented_residual[: self.block_size]
            )
            failed = failed | bulk_failed
        fields = solution.reshape((-1, 4, count)).transpose((0, 2, 1))
        x_count, _, z_count = self.plan.discretization.modal_shape
        fields = fields.reshape((x_count, z_count, count, 4)).transpose((0, 2, 1, 3))
        velocity = fields[..., :3] * self.horizontal_admissibility[:, None, :, None]
        pressure = fields[..., 3] * self.horizontal_admissibility[:, None, :]
        diagnostics = self._diagnostics(velocity, pressure, residual, failed)
        return ChannelStokesSolveResult(
            velocity=velocity,
            pressure=pressure,
            pressure_gradient=pressure_gradient,
            diagnostics=diagnostics,
            prepared_id=self.prepared_id,
        )

    def _validate_velocity(self, value: ArrayLike, owner: str, /) -> Array:
        array = jnp.asarray(value)
        expected = self.plan.discretization.modal_shape + (3,)
        if array.shape != expected:
            raise ValueError(f"{owner} must have shape {expected}; got {array.shape}.")
        if not jnp.issubdtype(array.dtype, jnp.complexfloating):
            raise TypeError(f"{owner} must use complex modal coefficients.")
        return array

    def _diagnostics(
        self,
        velocity: Array,
        pressure: Array,
        residual: Array,
        failed: Array,
        /,
    ) -> ChannelStokesDiagnostics:
        divergence = (
            1j * self.streamwise_wavenumbers[:, None, :] * velocity[..., 0]
            + self.plan.discretization.modal_derivative(velocity[..., 1], axis=1)
            + 1j * self.spanwise_wavenumbers[:, None, :] * velocity[..., 2]
        )
        lower = (
            oe.contract("j,xjzc->xzc", self.synthesis[0], velocity, backend="jax")
            / self.horizontal_constant_scale
        )
        upper = (
            oe.contract("j,xjzc->xzc", self.synthesis[-1], velocity, backend="jax")
            / self.horizontal_constant_scale
        )
        wall_residual = jnp.maximum(
            jnp.linalg.norm(
                lower.at[0, 0].add(-self.plan.lower_wall_velocity).reshape((-1,))
            ),
            jnp.linalg.norm(
                upper.at[0, 0].add(-self.plan.upper_wall_velocity).reshape((-1,))
            ),
        )
        pressure_profile = (
            self.synthesis @ pressure[0, :, 0]
        ) / self.horizontal_constant_scale
        gauge = jnp.abs(jnp.sum(self.quadrature_weights * pressure_profile))
        velocity_profile = (
            self.synthesis @ velocity[0, :, 0]
        ) / self.horizontal_constant_scale
        length = self.plan.discretization.axes[1].length
        bulk = jnp.asarray(
            (
                jnp.sum(self.quadrature_weights * velocity_profile[:, 0]) / length,
                jnp.sum(self.quadrature_weights * velocity_profile[:, 2]) / length,
            )
        )
        finite = (
            jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(pressure))
            & jnp.all(jnp.isfinite(residual))
        )
        return ChannelStokesDiagnostics(
            momentum_constraint_residual=jnp.linalg.norm(residual.reshape((-1,))),
            divergence_norm=jnp.linalg.norm(divergence.reshape((-1,))),
            wall_residual=wall_residual,
            pressure_gauge_residual=gauge,
            bulk_velocity=bulk,
            failed=failed | ~finite,
        )


def _channel_mode_matrix(
    synthesis: Array,
    derivative: Array,
    shift: Array,
    viscosity: Array,
    kx: Array,
    kz: Array,
    quadrature_weights: Array,
    /,
    *,
    zero_mode: bool,
) -> Array:
    count = int(synthesis.shape[0])
    interior = synthesis[1:-1]
    identity = jnp.eye(count, dtype=synthesis.dtype)
    helmholtz = (
        shift.astype(synthesis.dtype) * identity
        - viscosity.astype(synthesis.dtype) * (derivative @ derivative)
        + viscosity.astype(synthesis.dtype) * (kx**2 + kz**2) * identity
    )
    momentum = interior @ helmholtz
    pressure_x = 1j * kx.astype(synthesis.dtype) * interior
    pressure_y = interior @ derivative
    pressure_z = 1j * kz.astype(synthesis.dtype) * interior
    zeros = jnp.zeros_like(momentum)
    rows = [
        jnp.concatenate((momentum, zeros, zeros, pressure_x), axis=1),
        jnp.concatenate((zeros, momentum, zeros, pressure_y), axis=1),
        jnp.concatenate((zeros, zeros, momentum, pressure_z), axis=1),
    ]
    for component in range(3):
        for endpoint in (0, -1):
            blocks = [jnp.zeros((count,), dtype=synthesis.dtype) for _ in range(4)]
            blocks[component] = synthesis[endpoint]
            rows.append(jnp.concatenate(tuple(blocks))[None, :])
    if zero_mode:
        zero_row = jnp.zeros((count,), dtype=synthesis.dtype)
        rows[6] = jnp.concatenate(
            (
                zero_row,
                synthesis[-1] @ helmholtz,
                zero_row,
                synthesis[-1] @ derivative,
            )
        )[None, :]
    divergence = jnp.concatenate(
        (
            1j * kx.astype(synthesis.dtype) * synthesis,
            synthesis @ derivative,
            1j * kz.astype(synthesis.dtype) * synthesis,
            jnp.zeros_like(synthesis),
        ),
        axis=1,
    )
    if zero_mode:
        gauge = jnp.concatenate(
            (
                jnp.zeros((3 * count,), dtype=synthesis.dtype),
                quadrature_weights.astype(synthesis.dtype) @ synthesis,
            )
        )
        divergence = divergence.at[-1].set(gauge)
    rows.append(divergence)
    matrix = jnp.concatenate(tuple(rows), axis=0)
    if matrix.shape != (4 * count, 4 * count):
        raise RuntimeError("Channel Stokes mode matrix is not square.")
    return matrix


def _bulk_flux_block(
    zero_block: Array,
    synthesis: Array,
    quadrature_weights: Array,
    wall_length: Array,
    horizontal_scale: Array,
    /,
) -> Array:
    count = int(synthesis.shape[0])
    block_size = 4 * count
    interior_count = count - 2
    augmented = jnp.zeros((block_size + 2, block_size + 2), dtype=zero_block.dtype)
    augmented = augmented.at[:block_size, :block_size].set(zero_block)
    scale = horizontal_scale.astype(zero_block.dtype)
    augmented = augmented.at[0:interior_count, block_size].set(-scale)
    augmented = augmented.at[2 * interior_count : 3 * interior_count, block_size + 1].set(
        -scale
    )
    flux = (quadrature_weights.astype(zero_block.dtype) @ synthesis) / (
        wall_length.astype(zero_block.dtype) * scale
    )
    augmented = augmented.at[block_size, 0:count].set(flux)
    augmented = augmented.at[block_size + 1, 2 * count : 3 * count].set(flux)
    return augmented


__all__ = [
    "ChannelMeanConstraint",
    "ChannelMeanConstraintKind",
    "ChannelStokesDiagnostics",
    "ChannelStokesPlan",
    "ChannelStokesSolveResult",
    "PreparedChannelStokesSolver",
]
