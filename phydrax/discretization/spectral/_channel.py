#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from operator import index
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._geometry_precision import GeometryPrecisionPolicy
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg._local_blocks import (
    LocalBlockFactorization,
    prepare_local_block_factorization,
    solve_local_blocks,
)
from ._channel_ultraspherical import (
    prepare_ultraspherical_channel,
    PreparedUltrasphericalChannel,
)
from ._space import TensorSpectralDiscretization


ChannelMeanConstraintKind: TypeAlias = Literal["pressure_gradient", "bulk_flux"]
ChannelStokesRoute: TypeAlias = Literal["ultraspherical_banded", "dense_reference"]


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
        raw_values = jnp.asarray(values)
        if jnp.iscomplexobj(raw_values):
            raise TypeError("Channel mean constraint values must be real.")
        values_ = raw_values.astype(float)
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
    """Primitive-variable Fourier–Chebyshev–Fourier Stokes plan."""

    discretization: TensorSpectralDiscretization
    viscosity: Array
    lower_wall_velocity: Array
    upper_wall_velocity: Array
    mean_constraint: ChannelMeanConstraint
    route: ChannelStokesRoute = eqx.field(static=True)
    maximum_factor_bytes: int = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
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
        route: ChannelStokesRoute = "ultraspherical_banded",
        maximum_factor_bytes: int = 512 * 1024**2,
        constraint_tolerance: float = 1e-8,
    ):
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        families = tuple(axis.family for axis in discretization.axes)
        if families != ("fourier", "chebyshev", "fourier"):
            raise ValueError(
                "Channel Stokes requires Fourier x Chebyshev x Fourier axes."
            )
        raw_viscosity = jnp.asarray(viscosity)
        raw_lower = jnp.asarray(lower_wall_velocity)
        raw_upper = jnp.asarray(upper_wall_velocity)
        if any(
            jnp.iscomplexobj(value) for value in (raw_viscosity, raw_lower, raw_upper)
        ):
            raise TypeError("Channel viscosity and wall velocities must be real.")
        viscosity_ = raw_viscosity.astype(float)
        if viscosity_.shape != () or not bool(
            jnp.isfinite(viscosity_) & (viscosity_ > 0.0)
        ):
            raise ValueError("viscosity must be one finite positive scalar.")
        lower = raw_lower.astype(float)
        upper = raw_upper.astype(float)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not bool(jnp.all(jnp.isfinite(lower)) & jnp.all(jnp.isfinite(upper)))
        ):
            raise ValueError("Wall velocities must be finite vectors with shape (3,).")
        if not bool(jnp.isclose(lower[1], upper[1], rtol=1e-10, atol=1e-12)):
            raise ValueError(
                "Incompressible channel walls must have matching normal velocities."
            )
        constraint = (
            ChannelMeanConstraint() if mean_constraint is None else mean_constraint
        )
        if not isinstance(constraint, ChannelMeanConstraint):
            raise TypeError("mean_constraint must be ChannelMeanConstraint or None.")
        if route not in ("ultraspherical_banded", "dense_reference"):
            raise ValueError(
                "route must be 'ultraspherical_banded' or 'dense_reference'."
            )
        if isinstance(maximum_factor_bytes, bool):
            raise TypeError("maximum_factor_bytes must be an integer.")
        maximum = index(maximum_factor_bytes)
        if maximum <= 0:
            raise ValueError("maximum_factor_bytes must be positive.")
        tolerance = float(constraint_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("constraint_tolerance must be finite and positive.")
        identifier = canonical_fingerprint(
            {
                "kind": "channel-stokes-plan-v3",
                "discretization": discretization.prepared_id,
                "viscosity": float(viscosity_),
                "lower_wall": [float(value) for value in lower],
                "upper_wall": [float(value) for value in upper],
                "mean_constraint": constraint.constraint_id,
                "route": route,
                "maximum_factor_bytes": maximum,
                "constraint_tolerance": tolerance,
            }
        )
        self.discretization = discretization
        self.viscosity = viscosity_
        self.lower_wall_velocity = lower
        self.upper_wall_velocity = upper
        self.mean_constraint = constraint
        self.route = route
        self.maximum_factor_bytes = maximum
        self.constraint_tolerance = tolerance
        self.plan_id = identifier

    def prepare(self, shift: ArrayLike, /) -> PreparedChannelStokesSolver:
        return PreparedChannelStokesSolver(self, shift)


class ChannelStokesPreparationReport(StrictModule, NonTrainableState):
    route: ChannelStokesRoute = eqx.field(static=True)
    lower_bandwidth: int = eqx.field(static=True)
    upper_bandwidth: int = eqx.field(static=True)
    horizontal_batch_size: int = eqx.field(static=True)
    correction_rank: int = eqx.field(static=True)
    constraint_rank: int = eqx.field(static=True)
    shared_basis_bytes: int = eqx.field(static=True)
    operator_bytes: int = eqx.field(static=True)
    factor_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    persistent_bytes: int = eqx.field(static=True)
    preparation_bytes: int = eqx.field(static=True)
    pivot_margin: float = eqx.field(static=True)
    requires_unsharded_axis: bool = eqx.field(static=True)
    required_unsharded_axes: tuple[str, ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


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
    """Prepared fixed-band ultraspherical or explicit dense-reference solver."""

    plan: ChannelStokesPlan
    shift: Array
    blocks: Array | None
    factorization: LocalBlockFactorization | None
    bulk_block: Array | None
    bulk_factorization: LocalBlockFactorization | None
    synthesis: Array
    ultraspherical: PreparedUltrasphericalChannel | None
    derivative: Array
    quadrature_weights: Array
    streamwise_wavenumbers: Array
    spanwise_wavenumbers: Array
    horizontal_admissibility: Array
    horizontal_constant_scale: Array
    report: ChannelStokesPreparationReport
    zero_mode_index: int = eqx.field(static=True)
    wall_normal_count: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ChannelStokesPlan, shift: ArrayLike, /):
        if not isinstance(plan, ChannelStokesPlan):
            raise TypeError("plan must be a ChannelStokesPlan.")
        raw_shift = jnp.asarray(shift)
        if jnp.iscomplexobj(raw_shift):
            raise TypeError("shift must be real.")
        shift_ = raw_shift.astype(float)
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
        cell = discretization.periodic_cell
        if cell is None or cell.rank != 2 or cell.ambient_dimension != 3:
            raise ValueError(
                "Channel Stokes requires the root rank-two PeriodicCell geometry."
            )
        reciprocal = cell.reciprocal_vectors.astype(real_dtype)
        kx = x_axis.modes.mode_numbers.astype(real_dtype) * reciprocal[0, 0]
        kz = z_axis.modes.mode_numbers.astype(real_dtype) * reciprocal[1, 2]
        kx_grid, kz_grid = jnp.meshgrid(kx, kz, indexing="ij")
        horizontal_scale = jnp.sqrt(x_axis.length * z_axis.length).astype(real_dtype)
        if plan.route == "ultraspherical_banded":
            flat_kx = np.asarray(kx_grid).reshape((-1,))
            flat_kz = np.asarray(kz_grid).reshape((-1,))
            zero_candidates = np.flatnonzero((flat_kx == 0.0) & (flat_kz == 0.0))
            if zero_candidates.size != 1:
                raise RuntimeError("Channel Fourier layout must contain one zero mode.")
            zero_mode_index = int(zero_candidates[0])
            ultraspherical = prepare_ultraspherical_channel(
                plan,
                shift_,
                synthesis,
                wall_axis.quadrature_weights,
                kx_grid,
                kz_grid,
                zero_mode_index,
            )
            admissible = (~x_axis.modes.nyquist_mask)[:, None] & (
                ~z_axis.modes.nyquist_mask
            )[None, :]
            identifier = canonical_fingerprint(
                {
                    "kind": "prepared-channel-stokes-ultraspherical",
                    "plan": plan.plan_id,
                    "shift": float(shift_),
                    "prepared": ultraspherical.prepared_id,
                }
            )
            self.plan = plan
            self.shift = shift_
            self.blocks = None
            self.factorization = None
            self.bulk_block = None
            self.bulk_factorization = None
            self.ultraspherical = ultraspherical
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
            self.report = ChannelStokesPreparationReport(
                route=plan.route,
                lower_bandwidth=ultraspherical.lower_bandwidth,
                upper_bandwidth=ultraspherical.upper_bandwidth,
                horizontal_batch_size=int(kx_grid.size),
                correction_rank=ultraspherical.correction_rank,
                constraint_rank=7
                + (2 if plan.mean_constraint.kind == "bulk_flux" else 0),
                shared_basis_bytes=ultraspherical.shared_basis_bytes,
                operator_bytes=ultraspherical.operator_bytes,
                factor_bytes=ultraspherical.factor_bytes,
                workspace_bytes=ultraspherical.workspace_bytes,
                persistent_bytes=ultraspherical.persistent_bytes,
                preparation_bytes=ultraspherical.preparation_bytes,
                pivot_margin=ultraspherical.pivot_margin,
                requires_unsharded_axis=True,
                required_unsharded_axes=(discretization.plan.axis_names[1],),
                report_id=canonical_fingerprint(
                    {
                        "kind": "channel-stokes-preparation-report",
                        "route": plan.route,
                        "bandwidth": (
                            ultraspherical.lower_bandwidth,
                            ultraspherical.upper_bandwidth,
                        ),
                        "tau_rank": ultraspherical.correction_rank,
                        "persistent_bytes": ultraspherical.persistent_bytes,
                        "preparation_bytes": ultraspherical.preparation_bytes,
                        "pivot_margin": ultraspherical.pivot_margin,
                    }
                ),
            )
            self.prepared_id = identifier
            return
        dense_batch_size = int(kx.size * kz.size)
        dense_block_size = 4 * mode_count
        dense_dtype = np.dtype(jnp.result_type(synthesis.dtype, 1j))
        dense_real_dtype = np.dtype(real_dtype)
        factor_bytes_preflight = dense_batch_size * (
            dense_block_size * dense_block_size * dense_dtype.itemsize
            + dense_block_size * np.dtype(np.int32).itemsize
            + dense_block_size * dense_real_dtype.itemsize
            + np.dtype(np.bool_).itemsize
        )
        if plan.mean_constraint.kind == "bulk_flux":
            bulk_size = dense_block_size + 2
            factor_bytes_preflight += (
                bulk_size * bulk_size * dense_dtype.itemsize
                + bulk_size * np.dtype(np.int32).itemsize
                + bulk_size * dense_real_dtype.itemsize
                + np.dtype(np.bool_).itemsize
            )
        if factor_bytes_preflight > plan.maximum_factor_bytes:
            raise ValueError(
                f"Channel Stokes factors require at least {factor_bytes_preflight} "
                f"bytes, exceeding maximum_factor_bytes={plan.maximum_factor_bytes}."
            )
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
        if bulk_block is not None:
            bulk_factorization = prepare_local_block_factorization(bulk_block[None, ...])
        factorization = prepare_local_block_factorization(blocks)
        factor_arrays = (
            factorization.factors,
            factorization.pivots,
            factorization.metric_sqrt,
            factorization.failed_blocks,
        )
        if bulk_factorization is not None:
            factor_arrays = factor_arrays + (
                bulk_factorization.factors,
                bulk_factorization.pivots,
                bulk_factorization.metric_sqrt,
                bulk_factorization.failed_blocks,
            )
        factor_bytes = sum(int(array.nbytes) for array in factor_arrays)
        admissible = (~x_axis.modes.nyquist_mask)[:, None] & (~z_axis.modes.nyquist_mask)[
            None, :
        ]
        identifier = canonical_fingerprint(
            {
                "kind": "prepared-channel-stokes-dense-reference-v2",
                "plan": plan.plan_id,
                "shift": float(shift_),
                "block_shape": list(blocks.shape),
            }
        )
        self.plan = plan
        self.ultraspherical = None
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
        factor_diagonal = jnp.abs(jnp.diagonal(factorization.factors, axis1=-2, axis2=-1))
        factor_scale = jnp.maximum(jnp.max(jnp.abs(blocks)), 1.0)
        pivot_margin = float(jnp.min(factor_diagonal) / factor_scale)
        bandwidth = self.block_size - 1
        shared_basis_bytes = sum(
            int(array.nbytes)
            for array in (
                synthesis,
                derivative,
                wall_axis.quadrature_weights,
                kx_grid,
                kz_grid,
                admissible,
            )
        )
        operator_bytes = int(blocks.nbytes) + (
            0 if bulk_block is None else int(bulk_block.nbytes)
        )
        workspace_bytes = int(blocks.shape[0] * self.block_size * blocks.dtype.itemsize)
        persistent_bytes = shared_basis_bytes + operator_bytes + factor_bytes
        preparation_bytes = persistent_bytes + workspace_bytes
        self.report = ChannelStokesPreparationReport(
            route=plan.route,
            lower_bandwidth=bandwidth,
            upper_bandwidth=bandwidth,
            horizontal_batch_size=int(blocks.shape[0]),
            correction_rank=0,
            constraint_rank=7 + (2 if plan.mean_constraint.kind == "bulk_flux" else 0),
            shared_basis_bytes=shared_basis_bytes,
            operator_bytes=operator_bytes,
            factor_bytes=factor_bytes,
            workspace_bytes=workspace_bytes,
            persistent_bytes=persistent_bytes,
            preparation_bytes=preparation_bytes,
            pivot_margin=pivot_margin,
            requires_unsharded_axis=True,
            required_unsharded_axes=(discretization.plan.axis_names[1],),
            report_id=canonical_fingerprint(
                {
                    "kind": "channel-stokes-preparation-report",
                    "plan": plan.plan_id,
                    "route": plan.route,
                    "bandwidth": bandwidth,
                    "persistent_bytes": persistent_bytes,
                    "preparation_bytes": preparation_bytes,
                    "pivot_margin": pivot_margin,
                }
            ),
        )
        self.prepared_id = identifier

    def solve(self, right_hand_side: ArrayLike, /) -> ChannelStokesSolveResult:
        value = self._validate_velocity(right_hand_side, "Stokes right-hand side")
        value = value * self.horizontal_admissibility[:, None, :, None]
        modal_modes = jnp.transpose(value, (0, 2, 1, 3)).reshape(
            (-1, self.wall_normal_count, 3)
        )
        if self.ultraspherical is not None:
            modal_velocity, modal_pressure, residual, failed, pressure_gradient = (
                self.ultraspherical.solve(
                    modal_modes,
                    self.plan.lower_wall_velocity,
                    self.plan.upper_wall_velocity,
                    self.plan.mean_constraint.values,
                    mean_kind=self.plan.mean_constraint.kind,
                )
            )
            x_count, _, z_count = self.plan.discretization.modal_shape
            velocity = modal_velocity.reshape(
                (x_count, z_count, self.wall_normal_count, 3)
            ).transpose((0, 2, 1, 3))
            pressure = modal_pressure.reshape(
                (x_count, z_count, self.wall_normal_count)
            ).transpose((0, 2, 1))
            velocity = jnp.where(
                self.horizontal_admissibility[:, None, :, None], velocity, 0.0
            )
            pressure = jnp.where(self.horizontal_admissibility[:, None, :], pressure, 0.0)
            active_modes = self.horizontal_admissibility.reshape((-1,))
            residual = jnp.where(active_modes[:, None], residual, 0.0)
            failed = failed & active_modes
            diagnostics = self._diagnostics(velocity, pressure, residual, failed)
            return ChannelStokesSolveResult(
                velocity=velocity,
                pressure=pressure,
                pressure_gradient=pressure_gradient,
                diagnostics=diagnostics,
                prepared_id=self.prepared_id,
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
        velocity = jnp.where(
            self.horizontal_admissibility[:, None, :, None], fields[..., :3], 0.0
        )
        pressure = jnp.where(
            self.horizontal_admissibility[:, None, :], fields[..., 3], 0.0
        )
        active_modes = self.horizontal_admissibility.reshape((-1,))
        residual = jnp.where(active_modes[:, None], residual, 0.0)
        failed = failed & active_modes
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
        precision = GeometryPrecisionPolicy()
        wall_residual = jnp.maximum(
            precision.norm(
                lower.at[0, 0].add(-self.plan.lower_wall_velocity).reshape((-1,))
            ),
            precision.norm(
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
        mean_residual = (
            precision.norm((bulk - self.plan.mean_constraint.values).reshape((-1,)))
            if self.plan.mean_constraint.kind == "bulk_flux"
            else jnp.asarray(0.0, dtype=bulk.dtype)
        )
        finite = (
            jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(pressure))
            & jnp.all(jnp.isfinite(residual))
        )
        momentum_residual = precision.norm(residual.reshape((-1,)))
        divergence_norm = precision.norm(divergence.reshape((-1,)))
        tolerance = self.plan.constraint_tolerance
        return ChannelStokesDiagnostics(
            momentum_constraint_residual=momentum_residual,
            divergence_norm=divergence_norm,
            wall_residual=wall_residual,
            pressure_gauge_residual=gauge,
            bulk_velocity=bulk,
            failed=(
                jnp.any(failed)
                | ~finite
                | (momentum_residual > tolerance)
                | (divergence_norm > tolerance)
                | (wall_residual > tolerance)
                | (gauge > tolerance)
                | (mean_residual > tolerance)
            ),
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
    "ChannelStokesPreparationReport",
    "ChannelStokesRoute",
    "ChannelStokesDiagnostics",
    "ChannelStokesPlan",
    "ChannelStokesSolveResult",
    "PreparedChannelStokesSolver",
]
