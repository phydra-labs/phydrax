#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mixed MAC projection with closed gas compartments and an optional atmosphere.

Each closed compartment ``b`` is a homobaric gas region with gas-volume weight
``w_b = chi_b g / (V V_b)`` and a backward-Euler compliance closure
``p_b^{n+1} = p_b^n - (dt / C_b) Q_b``. The projected face velocity, the
compartment volume rates ``Q`` and the atmosphere rate ``Q_a`` solve the
symmetric KKT system of the resolved-bubbly-flow guide (research gate D3)::

    [ L        -W        -w_a ] [ phi ]   [ -D u*                 ]
    [ -W^*   -diag(dt/C)   0  ] [ Q   ] = [ -(p^n - <w, h>_V)     ]
    [ -w_a^*     0         0  ] [ Q_a ]   [ -(p_atm - <w_a, h>_V) ]

with ``L = -D(dt/rho_f G)`` in the pairing ``V (+) R^K (+) R``. The tiny
compartment Schur route writes ``phi = psi + mu`` with mean-free ``psi``, runs
the ``K + n_atm + 1`` gauged variable-density solves as one batch and solves
the bordered ``(K + n_atm + 1)^2`` system with the native dense LU.
"""

from __future__ import annotations

from enum import IntEnum
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_integer
from ..discretization.finite_volume._incompressible import (
    FaceVelocity,
    PreparedMACOperators,
)
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    prepare,
    PreparedLinearSolve,
    refresh,
    solve,
)
from ..typing import (
    as_array,
    Bool,
    checked,
    Dim,
    Float,
    Int32,
    Scalar,
    Scope,
    VariadicDim,
)
from ._mac_variable_density import (
    MACVariableDensityProjectionPlan,
    MACVariableDensityProjectionResult,
)


# The bordered Schur system is dense; the route is reserved for a statically
# tiny compartment count.
_MAXIMUM_COMPARTMENT_CAPACITY = 32

# Block vector ``(phi cells, Q (K), Q_a scalar)`` of the compartment KKT system.
type MACCompartmentBlockVector = tuple[Array, Array, Array]


class MACCompartmentGauge(IntEnum):
    """How the absolute level of the projected pressure is fixed."""

    COMPARTMENT_DETERMINED = 0
    ATMOSPHERE_REFERENCE = 1
    MEAN_ZERO = 2


class MACCompartmentProjectionStatus(IntEnum):
    """Outcome of one mixed compartment projection."""

    CONVERGED = 0
    PRESSURE_SOLVE_FAILED = 1
    SCHUR_SOLVE_FAILED = 2
    INVALID_CONSTRAINT = 3
    NONFINITE = 4


class _CellDims(VariadicDim):
    """MAC cell-pressure layout."""


class _SlotDim(Dim, minimum=1):
    """Closed-compartment slots of one projection plan."""


@final
class MACCompartmentConstraint(StrictModule):
    """Per-call compartment data of the mixed MAC projection.

    ``labels`` holds the closed-compartment slot of every cell (``-1`` outside
    every closed compartment), ``gas_volume`` the gas volume content
    ``g_c = V_c (1 - alpha_c)`` in m^3, ``compliance`` ``C_b = -dV/dp > 0`` per
    slot, ``pressure`` the absolute compartment pressure ``p_b^n``, ``active``
    the slots that carry a compartment, ``atmosphere`` the vented gas cells
    (all ``False`` when unused), ``atmosphere_pressure`` ``p_atm`` and
    ``pressure_offset`` the field ``h`` such that the absolute pressure is
    ``phi + h``. Structural agreement is checked here; admissibility (positive
    gas volume of active slots, positive finite compliance, finite values,
    disjoint supports) is reported by the projection status.
    """

    __strict_contract__ = True

    labels: Int32[_CellDims]
    gas_volume: Float[_CellDims]
    compliance: Float[_SlotDim]
    pressure: Float[_SlotDim]
    active: Bool[_SlotDim]
    atmosphere: Bool[_CellDims]
    atmosphere_pressure: Float[Scalar]
    pressure_offset: Float[_CellDims]

    def __init__(
        self,
        *,
        labels: ArrayLike,
        gas_volume: ArrayLike,
        compliance: ArrayLike,
        pressure: ArrayLike,
        active: ArrayLike,
        atmosphere: ArrayLike,
        atmosphere_pressure: ArrayLike,
        pressure_offset: ArrayLike,
    ) -> None:
        scope = Scope()
        labels_ = as_array(labels, Int32[_CellDims], "labels", scope=scope)
        gas = as_array(gas_volume, Float[_CellDims], "gas_volume", scope=scope)
        compliance_ = as_array(compliance, Float[_SlotDim], "compliance", scope=scope)
        pressure_ = as_array(pressure, Float[_SlotDim], "pressure", scope=scope)
        active_ = as_array(active, Bool[_SlotDim], "active", scope=scope)
        atmosphere_ = as_array(atmosphere, Bool[_CellDims], "atmosphere", scope=scope)
        atmosphere_pressure_ = as_array(
            atmosphere_pressure, Float[Scalar], "atmosphere_pressure"
        )
        offset = as_array(
            pressure_offset, Float[_CellDims], "pressure_offset", scope=scope
        )
        self.labels = labels_
        self.gas_volume = gas
        self.compliance = compliance_
        self.pressure = pressure_
        self.active = active_
        self.atmosphere = atmosphere_
        self.atmosphere_pressure = atmosphere_pressure_
        self.pressure_offset = offset


class _CompartmentSupport(StrictModule):
    """Sanitized gas-volume weights of one constraint.

    Columns are the closed slots followed by the atmosphere when the plan
    declares one. Inactive columns carry zero weight and inadmissible values
    are replaced before use, so every derived quantity stays finite while
    ``valid`` records whether the constraint was admissible.
    """

    weights: Array
    active: Array
    compliance: Array
    reference_pressure: Array
    offset_mean: Array
    offset: Array
    valid: Array


class _ProjectionCandidate(StrictModule):
    """Recovered pressure, rates and momentum impulse before the commit decision."""

    pressure: Array
    rates: Array
    level: Array
    impulse: FaceVelocity
    increment: Array
    schur: LinearSolveResult


@final
class MACCompartmentBlockOperator(StrictModule, NonTrainableState):
    """Symmetric KKT action of the mixed compartment projection.

    The block vector is ``(phi, Q, Q_a)`` with the pairing
    ``<phi, phi'>_V + Q . Q' + Q_a Q_a'``. Inactive closed slots, and the
    atmosphere row when the plan declares none or the atmosphere holds no gas,
    act as identity rows.
    """

    operators: PreparedMACOperators
    face_inverse_density: FaceVelocity
    step_size: Array
    weights: Array
    diagonal: Array
    reference: Array
    compartment_capacity: int = eqx.field(static=True)
    atmosphere: bool = eqx.field(static=True)

    def _validated(
        self, vector: MACCompartmentBlockVector, /
    ) -> MACCompartmentBlockVector:
        pressure, rates, atmosphere_rate = vector
        pressure_ = self.operators.validate_pressure(pressure)
        rates_ = jnp.asarray(rates, dtype=pressure_.dtype)
        atmosphere_rate_ = jnp.asarray(atmosphere_rate, dtype=pressure_.dtype)
        if rates_.shape != (self.compartment_capacity,) or atmosphere_rate_.shape != ():
            raise ValueError(
                "Compartment block vectors hold one rate per slot and a scalar atmosphere rate."
            )
        return pressure_, rates_, atmosphere_rate_

    def _columns(self, rates: Array, atmosphere_rate: Array, /) -> Array:
        if self.atmosphere:
            return jnp.concatenate((rates, atmosphere_rate.reshape(1)))
        return rates

    def _split(self, columns: Array, atmosphere_rate: Array, /) -> tuple[Array, Array]:
        if self.atmosphere:
            capacity = self.compartment_capacity
            return columns[:capacity], columns[capacity]
        # An undeclared atmosphere keeps its component as an identity row.
        return columns, atmosphere_rate

    def _weighted(self, /) -> Array:
        volumes = self.operators.discretization.cell_volumes.astype(self.weights.dtype)
        return (self.weights * volumes).reshape((self.weights.shape[0], -1))

    def _distribute(self, columns: Array, /) -> Array:
        count = self.weights.shape[0]
        return (columns @ self.weights.reshape((count, -1))).reshape(
            self.weights.shape[1:]
        )

    def _coefficient(self, /) -> FaceVelocity:
        return tuple(self.step_size * value for value in self.face_inverse_density)

    def mv(self, vector: MACCompartmentBlockVector, /) -> MACCompartmentBlockVector:
        """Row action ``(L phi - W Q - w_a Q_a, -W^* phi - (dt/C) Q, -<w_a, phi>)``."""
        pressure, rates, atmosphere_rate = self._validated(vector)
        flux = tuple(
            coefficient * derivative
            for coefficient, derivative in zip(
                self._coefficient(), self.operators.gradient(pressure), strict=True
            )
        )
        columns = self._columns(rates, atmosphere_rate)
        pressure_row = -self.operators.divergence(flux) - self._distribute(columns)
        column_rows = -(self._weighted() @ pressure.reshape(-1)) + self.diagonal * columns
        rate_row, atmosphere_row = self._split(column_rows, atmosphere_rate)
        return pressure_row, rate_row, atmosphere_row

    def transpose_mv(
        self, vector: MACCompartmentBlockVector, /
    ) -> MACCompartmentBlockVector:
        """Adjoint action in the block pairing, assembled column by column.

        The pressure block uses ``L^* = V^{-1} L_E^T V`` with the Euclidean
        transposes of the MAC divergence and gradient, independent of the
        discrete adjoint identity that makes ``L`` self-adjoint.
        """
        pressure, rates, atmosphere_rate = self._validated(vector)
        discretization = self.operators.discretization
        volumes = discretization.cell_volumes.astype(pressure.dtype)
        faces = tuple(
            jnp.zeros(layout.shape, dtype=pressure.dtype)
            for layout in discretization.face_layouts
        )
        divergence_transpose = jax.linear_transpose(self.operators.divergence, faces)
        gradient_transpose = jax.linear_transpose(
            self.operators.gradient, jnp.zeros_like(pressure)
        )
        (face_cotangent,) = divergence_transpose(volumes * pressure)
        weighted = tuple(
            coefficient * value
            for coefficient, value in zip(
                self._coefficient(), face_cotangent, strict=True
            )
        )
        (cell_cotangent,) = gradient_transpose(weighted)
        columns = self._columns(rates, atmosphere_rate)
        pressure_column = -cell_cotangent / volumes - self._distribute(columns)
        rate_columns = (
            -(self._weighted() @ pressure.reshape(-1)) + self.diagonal * columns
        )
        rate_column, atmosphere_column = self._split(rate_columns, atmosphere_rate)
        return pressure_column, rate_column, atmosphere_column

    def pairing(
        self, first: MACCompartmentBlockVector, second: MACCompartmentBlockVector, /
    ) -> Array:
        """``<phi, phi'>_V + Q . Q' + Q_a Q_a'``."""
        pressure, rates, atmosphere_rate = self._validated(first)
        other_pressure, other_rates, other_atmosphere_rate = self._validated(second)
        volumes = self.operators.discretization.cell_volumes.astype(pressure.dtype)
        return (
            jnp.sum(volumes * pressure * other_pressure)
            + jnp.sum(rates * other_rates)
            + atmosphere_rate * other_atmosphere_rate
        )

    def right_hand_side(self, momentum: FaceVelocity, /) -> MACCompartmentBlockVector:
        """``(-D u*, -(p^n - <w, h>_V), -(p_atm - <w_a, h>_V))``; identity rows are zero."""
        values = self.operators.validate_velocity(momentum)
        velocity = tuple(
            inverse * value
            for inverse, value in zip(self.face_inverse_density, values, strict=True)
        )
        pressure_row = -self.operators.divergence(velocity)
        rate_row, atmosphere_row = self._split(
            -self.reference, jnp.zeros((), dtype=pressure_row.dtype)
        )
        return pressure_row, rate_row, atmosphere_row


@final
class MACCompartmentProjectionResult(StrictModule):
    """Fail-closed mixed compartment projection with its constraint and work evidence.

    State fields (`momentum`, `velocity`, `pressure`, `absolute_pressure`,
    `compartment_pressure`, `pressure_increment`, `volume_rate`,
    `atmosphere_volume_rate`, `divergence_after`, `divergence_target`) and the
    work terms describe the committed step: on failure they return the input
    momentum and pressure, the previous compartment pressures and zero rates
    and work. Residual fields are evidence of the attempted solve.
    """

    momentum: FaceVelocity
    velocity: FaceVelocity
    pressure: Array
    absolute_pressure: Array
    compartment_pressure: Array
    pressure_increment: Array
    volume_rate: Array
    atmosphere_volume_rate: Array
    divergence_before: Array
    divergence_after: Array
    divergence_target: Array
    continuity_residual: Array
    continuity_scale: Array
    compartment_pressure_residual: Array
    atmosphere_pressure_residual: Array
    schur_residual: Array
    compartment_work: Array
    atmosphere_work: Array
    dynamic_pressure_work: Array
    offset_pressure_work: Array
    work_identity_residual: Array
    projection_dissipation: Array
    kinetic_energy_change: Array
    gauge: Array
    lane_converged: Array
    pressure_solves: LinearSolveResult
    schur: LinearSolveResult
    valid_constraint: Array
    finite: Array
    status: Array
    successful: Array
    projection_id: str = eqx.field(static=True)


def _schur_problem(matrix: Array, operator_id: str, problem_id: str, /) -> LinearSystem:
    return LinearSystem(
        DenseLinearOperator(matrix, operator_id=operator_id), problem_id=problem_id
    )


@final
class MACCompartmentProjectionPlan(StrictModule, NonTrainableState):
    """Prepared tiny-compartment Schur route of the mixed MAC projection.

    The ``K + n_atm + 1`` gauged solves ``z_0 = L_g^{-1} P0 (-D u*)`` and
    ``z_j = L_g^{-1} P0 w_j`` share the composed
    `MACVariableDensityProjectionPlan` and run as one batch; every lane keeps
    that owner's acceptance threshold. These localized mean-free Schur basis
    right-hand sides use unpreconditioned PCG: diagonal scaling can amplify the
    gauged operator's condition number at a sharp density jump, while the
    unscaled operator reaches the declared true-residual tolerance. The bordered
    system in ``(Q, Q_a, mu)`` is solved with the native dense LU after one
    symmetric rescaling that makes its compliance block of unit size. With
    neither an active closed compartment nor an atmosphere the volume row is
    dependent and is replaced by the mean-zero gauge ``mu = 0``, which recovers
    the variable-density projection.
    """

    projection: MACVariableDensityProjectionPlan
    compartment_capacity: int = eqx.field(static=True)
    atmosphere: bool = eqx.field(static=True)
    schur_policy: LinearSolvePolicy
    prepared_schur: PreparedLinearSolve
    schur_operator_id: str = eqx.field(static=True)
    schur_problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        operators: PreparedMACOperators,
        /,
        *,
        compartment_capacity: int,
        atmosphere: bool,
        tolerance: float = 1e-9,
        maximum_iterations: int = 500,
    ) -> None:
        capacity = positive_integer(compartment_capacity, "compartment_capacity")
        if capacity > _MAXIMUM_COMPARTMENT_CAPACITY:
            raise ValueError(
                "The tiny compartment Schur route supports at most "
                f"{_MAXIMUM_COMPARTMENT_CAPACITY} closed compartments."
            )
        if not isinstance(atmosphere, bool):
            raise TypeError("atmosphere must be a bool.")
        projection = MACVariableDensityProjectionPlan(
            operators,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
            jacobi_preconditioning=False,
        )
        size = capacity + (2 if atmosphere else 1)
        schur_operator_id = canonical_fingerprint(
            {
                "kind": "mac-compartment-bordered-schur-operator",
                "operators": operators.prepared_id,
                "size": size,
            }
        )
        schur_problem_id = canonical_fingerprint(
            {
                "kind": "mac-compartment-bordered-schur-system",
                "operator": schur_operator_id,
            }
        )
        policy = LinearSolvePolicy(DenseLU())
        prepared = prepare(
            _schur_problem(
                jnp.eye(size, dtype=operators.pressure_space.dtype),
                schur_operator_id,
                schur_problem_id,
            ),
            policy,
        )
        identifier = canonical_fingerprint(
            {
                "kind": "mac-compartment-projection-plan",
                "projection": projection.plan_id,
                "compartment_capacity": capacity,
                "atmosphere": atmosphere,
                "schur_plan": prepared.plan.plan_id,
                "route": "tiny-compartment-schur",
            }
        )
        self.projection = projection
        self.compartment_capacity = capacity
        self.atmosphere = atmosphere
        self.schur_policy = policy
        self.prepared_schur = prepared
        self.schur_operator_id = schur_operator_id
        self.schur_problem_id = schur_problem_id
        self.plan_id = identifier

    @checked
    def _support(self, constraint: MACCompartmentConstraint, /) -> _CompartmentSupport:
        operators = self.projection.operators
        cell_shape = tuple(operators.discretization.cell_shape)
        capacity = self.compartment_capacity
        if constraint.labels.shape != cell_shape:
            raise ValueError("Compartment cell fields must match the MAC cell shape.")
        if constraint.compliance.shape != (capacity,):
            raise ValueError("Compartment slot fields must match compartment_capacity.")
        dtype = operators.pressure_space.dtype
        volumes = operators.discretization.cell_volumes.astype(dtype)
        gas = constraint.gas_volume.astype(dtype)
        compliance = constraint.compliance.astype(dtype)
        pressure = constraint.pressure.astype(dtype)
        offset = constraint.pressure_offset.astype(dtype)
        atmosphere_pressure = constraint.atmosphere_pressure.astype(dtype)
        labels = constraint.labels
        vented = constraint.atmosphere
        column_shape = (-1,) + (1,) * len(cell_shape)
        supports = labels[None] == jnp.arange(capacity, dtype=jnp.int32).reshape(
            column_shape
        )
        if self.atmosphere:
            supports = jnp.concatenate((supports, vented[None]), axis=0)
        admissible_gas = jnp.isfinite(gas) & (gas >= 0.0)
        content = jnp.where(supports & admissible_gas, gas, 0.0)
        volume = jnp.sum(content, axis=tuple(range(1, len(cell_shape) + 1)))
        occupied = volume > 0.0
        slot_admissible = (
            occupied[:capacity]
            & jnp.isfinite(compliance)
            & (compliance > 0.0)
            & jnp.isfinite(pressure)
        )
        closed_active = constraint.active & slot_admissible
        valid = (
            jnp.all(admissible_gas)
            & jnp.all((labels >= -1) & (labels < capacity))
            & ~jnp.any(vented & (labels >= 0))
            & jnp.all(~constraint.active | slot_admissible)
            & jnp.all(jnp.isfinite(offset))
            & jnp.isfinite(atmosphere_pressure)
        )
        if self.atmosphere:
            # A declared atmosphere without gas has no pressure mean; its row
            # becomes an identity row, like an inactive closed slot.
            active = jnp.concatenate((closed_active, occupied[capacity:]))
            reference = jnp.concatenate((pressure, atmosphere_pressure.reshape(1)))
        else:
            active = closed_active
            reference = pressure
            valid = valid & ~jnp.any(vented)
        safe_volume = jnp.where(active, volume, 1.0)
        weights = jnp.where(
            active.reshape(column_shape),
            content / (volumes * safe_volume.reshape(column_shape)),
            0.0,
        )
        safe_offset = jnp.where(jnp.isfinite(offset), offset, 0.0)
        offset_mean = (weights * volumes).reshape((weights.shape[0], -1)) @ (
            safe_offset.reshape(-1)
        )
        return _CompartmentSupport(
            weights=weights,
            active=active,
            compliance=jnp.where(closed_active, compliance, 1.0),
            reference_pressure=jnp.where(active, reference, 0.0),
            offset_mean=offset_mean,
            offset=safe_offset,
            valid=valid,
        )

    def _compliance_diagonal(self, support: _CompartmentSupport, step: Array, /) -> Array:
        """``dt / C_b`` of active closed slots; zero for the atmosphere and inactive slots."""
        capacity = self.compartment_capacity
        closed = jnp.where(support.active[:capacity], step / support.compliance, 0.0)
        if self.atmosphere:
            return jnp.concatenate((closed, jnp.zeros((1,), dtype=closed.dtype)))
        return closed

    def block_operator(
        self,
        face_inverse_density: FaceVelocity,
        step_size: ArrayLike,
        constraint: MACCompartmentConstraint,
        /,
    ) -> MACCompartmentBlockOperator:
        """Symmetric KKT operator of one step in the pairing ``V (+) R^K (+) R``."""
        operators = self.projection.operators
        inverse = self.projection.validate_face_inverse_density(face_inverse_density)
        step = jnp.asarray(step_size, dtype=operators.pressure_space.dtype).reshape(())
        support = self._support(constraint)
        diagonal = jnp.where(
            support.active, -self._compliance_diagonal(support, step), 1.0
        )
        return MACCompartmentBlockOperator(
            operators=operators,
            face_inverse_density=inverse,
            step_size=step,
            weights=support.weights,
            diagonal=diagonal,
            reference=jnp.where(
                support.active, support.reference_pressure - support.offset_mean, 0.0
            ),
            compartment_capacity=self.compartment_capacity,
            atmosphere=self.atmosphere,
        )

    def _lane_solves(
        self,
        momentum: FaceVelocity,
        face_inverse_density: FaceVelocity,
        step: Array,
        incoming_pressure: Array,
        base_target: Array,
        weights: Array,
        /,
    ) -> MACVariableDensityProjectionResult:
        """Lane 0 projects ``u*``; lane ``j`` solves ``L_g z_j = P0 w_j`` from rest."""
        volumes = self.projection.operators.discretization.cell_volumes.astype(
            weights.dtype
        )
        cell_axes = tuple(range(1, weights.ndim))
        means = jnp.sum(weights * volumes, axis=cell_axes, keepdims=True) / jnp.sum(
            volumes
        )
        count = weights.shape[0]
        targets = jnp.concatenate((base_target[None], weights - means), axis=0)
        guesses = jnp.concatenate(
            (incoming_pressure[None], jnp.zeros_like(weights)), axis=0
        )
        momenta = tuple(
            jnp.concatenate(
                (value[None], jnp.zeros((count,) + value.shape, dtype=value.dtype)),
                axis=0,
            )
            for value in momentum
        )

        return self.projection._project_many(
            momenta,
            face_inverse_density,
            step,
            pressure=guesses,
            target_divergence=targets,
        )

    def _schur_solve(
        self,
        coupling: Array,
        compliance_diagonal: Array,
        active: Array,
        rows: Array,
        volume_flux: Array,
        /,
    ) -> tuple[Array, Array, LinearSolveResult]:
        """Solve the bordered system for ``(Q, Q_a, mu)``.

        With ``s`` the largest active diagonal of ``Z + diag(dt/C)``, the
        symmetric scaling ``Q = q / sqrt(s)``, ``mu = m sqrt(s)`` gives a
        compliance block of unit size and a unit border, so the dense LU
        singularity threshold is measured against an O(1) matrix.
        """
        dtype = coupling.dtype
        count = active.shape[0]
        any_active = jnp.any(active)
        stiffness = coupling + jnp.diag(compliance_diagonal)
        largest = jnp.max(jnp.where(active, jnp.diagonal(stiffness), 0.0))
        scale = jnp.where(largest > 0.0, largest, 1.0)
        root = jnp.sqrt(scale)
        block = jnp.where(
            active[:, None] & active[None, :], stiffness / scale, 0.0
        ) + jnp.diag(jnp.where(active, 0.0, 1.0))
        border = active.astype(dtype)
        corner = jnp.where(any_active, 0.0, 1.0).astype(dtype).reshape(1)
        matrix = jnp.concatenate(
            (
                jnp.concatenate((block, border[:, None]), axis=1),
                jnp.concatenate((border, corner))[None, :],
            ),
            axis=0,
        ).astype(dtype)
        rhs = jnp.concatenate(
            (
                jnp.where(active, rows / root, 0.0),
                jnp.where(any_active, volume_flux * root, 0.0).reshape(1),
            )
        ).astype(dtype)
        prepared = refresh(
            self.prepared_schur,
            _schur_problem(matrix, self.schur_operator_id, self.schur_problem_id),
        )
        result = solve(prepared, rhs)
        rates = jnp.where(active, result.value[:count] / root, 0.0)
        return rates, result.value[count] * root, result

    def _recover(
        self,
        step: Array,
        support: _CompartmentSupport,
        lanes: MACVariableDensityProjectionResult,
        volume_flux: Array,
        /,
    ) -> _ProjectionCandidate:
        """Schur solve and recovery ``phi = z_0 + sum_j Q_j z_j + mu``."""
        capacity = self.compartment_capacity
        count = support.weights.shape[0]
        volumes = self.projection.operators.discretization.cell_volumes.astype(
            support.weights.dtype
        )
        weighted = (support.weights * volumes).reshape((count, -1))
        solutions = lanes.pressure_increment.reshape((count + 1, -1))
        rates, level, schur = self._schur_solve(
            weighted @ solutions[1:].T,
            self._compliance_diagonal(support, step),
            support.active,
            support.reference_pressure - support.offset_mean - weighted @ solutions[0],
            volume_flux,
        )
        pressure = (solutions[0] + rates @ solutions[1:]).reshape(
            support.offset.shape
        ) + level
        impulse = tuple(
            lane[0] + (rates @ lane[1:].reshape((count, -1))).reshape(lane.shape[1:])
            for lane in lanes.pressure_impulse
        )
        increment = jnp.where(
            support.active[:capacity],
            -(step / support.compliance) * rates[:capacity],
            0.0,
        )
        return _ProjectionCandidate(
            pressure=pressure,
            rates=rates,
            level=level,
            impulse=impulse,
            increment=increment,
            schur=schur,
        )

    def _constraint_residuals(
        self,
        support: _CompartmentSupport,
        lanes: MACVariableDensityProjectionResult,
        candidate: _ProjectionCandidate,
        velocity: FaceVelocity,
        next_pressure: Array,
        atmosphere_pressure: Array,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        """Continuity residual and scale, and the compartment and atmosphere pressure rows.

        The continuity defect is ``res_0 + sum_j Q_j res_j`` of the lane
        residuals, so it is bounded by ``tolerance * continuity_scale`` with
        each lane's owner threshold scale weighted by ``|Q_j|``.
        """
        operators = self.projection.operators
        capacity = self.compartment_capacity
        count = support.weights.shape[0]
        volumes = operators.discretization.cell_volumes.astype(next_pressure.dtype)
        target = (candidate.rates @ support.weights.reshape((count, -1))).reshape(
            volumes.shape
        )
        defect = operators.divergence(velocity) - target
        continuity_residual = jnp.sqrt(jnp.sum(volumes * defect**2))
        lane_scale = lanes.divergence_scale + jnp.sqrt(
            jnp.sum(
                volumes * lanes.compatible_rhs**2,
                axis=tuple(range(1, lanes.compatible_rhs.ndim)),
            )
        )
        continuity_scale = lane_scale[0] + jnp.sum(
            jnp.abs(candidate.rates) * lane_scale[1:]
        )
        means = (support.weights * volumes).reshape((count, -1)) @ (
            candidate.pressure + support.offset
        ).reshape(-1)
        compartment_residual = jnp.max(
            jnp.where(
                support.active[:capacity],
                jnp.abs(means[:capacity] - next_pressure),
                0.0,
            )
        )
        atmosphere_residual = (
            jnp.where(
                support.active[capacity],
                jnp.abs(means[capacity] - atmosphere_pressure),
                0.0,
            )
            if self.atmosphere
            else jnp.zeros((), next_pressure.dtype)
        )
        return (
            continuity_residual,
            continuity_scale,
            compartment_residual,
            atmosphere_residual,
        )

    def _status(
        self,
        support: _CompartmentSupport,
        lanes: MACVariableDensityProjectionResult,
        candidate: _ProjectionCandidate,
        momentum: FaceVelocity,
        next_pressure: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        """Status, candidate finiteness and gauge code of one projection."""
        capacity = self.compartment_capacity
        finite = (
            jnp.all(jnp.isfinite(candidate.pressure))
            & jnp.all(jnp.isfinite(candidate.rates))
            & jnp.isfinite(candidate.level)
            & jnp.all(jnp.isfinite(next_pressure))
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in momentum))
            )
        )
        status = jnp.where(
            ~support.valid,
            int(MACCompartmentProjectionStatus.INVALID_CONSTRAINT),
            jnp.where(
                ~jnp.all(lanes.successful),
                int(MACCompartmentProjectionStatus.PRESSURE_SOLVE_FAILED),
                jnp.where(
                    ~candidate.schur.successful,
                    int(MACCompartmentProjectionStatus.SCHUR_SOLVE_FAILED),
                    jnp.where(
                        ~finite,
                        int(MACCompartmentProjectionStatus.NONFINITE),
                        int(MACCompartmentProjectionStatus.CONVERGED),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        vented = support.active[capacity] if self.atmosphere else jnp.asarray(False)
        gauge = jnp.where(
            vented,
            int(MACCompartmentGauge.ATMOSPHERE_REFERENCE),
            jnp.where(
                jnp.any(support.active[:capacity]),
                int(MACCompartmentGauge.COMPARTMENT_DETERMINED),
                int(MACCompartmentGauge.MEAN_ZERO),
            ),
        ).astype(jnp.int32)
        return status, finite, gauge

    def _work_evidence(
        self,
        support: _CompartmentSupport,
        candidate: _ProjectionCandidate,
        step: Array,
        values: FaceVelocity,
        inverse: FaceVelocity,
        momentum_candidate: FaceVelocity,
        velocity_before: FaceVelocity,
        velocity_candidate: FaceVelocity,
        next_pressure: Array,
        atmosphere_pressure: Array,
        /,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        """Evaluate the constraint and pressure-update work identities."""
        operators = self.projection.operators
        capacity = self.compartment_capacity
        work_step = step.astype(jnp.float64)
        work_rates = candidate.rates.astype(jnp.float64)
        work_pressure = candidate.pressure.astype(jnp.float64)
        work_volumes = operators.discretization.cell_volumes.astype(jnp.float64)
        work_divergence = operators.divergence(velocity_candidate).astype(jnp.float64)
        weighted_pressure = (support.weights.astype(jnp.float64) * work_volumes).reshape(
            (support.weights.shape[0], -1)
        ) @ work_pressure.reshape(-1)
        absolute_constraint_pressure = (
            jnp.concatenate(
                (
                    next_pressure.astype(jnp.float64),
                    atmosphere_pressure.astype(jnp.float64).reshape(1),
                )
            )
            if self.atmosphere
            else next_pressure.astype(jnp.float64)
        )
        offset_mean = support.offset_mean.astype(jnp.float64)
        compartment_work = work_step * jnp.sum(
            next_pressure.astype(jnp.float64) * work_rates[:capacity]
        )
        atmosphere_work = (
            work_step * atmosphere_pressure.astype(jnp.float64) * work_rates[capacity]
            if self.atmosphere
            else jnp.zeros((), dtype=jnp.float64)
        )
        offset_work = work_step * jnp.sum(work_rates * offset_mean)
        measures = operators.face_dual_measures
        dynamic_work = jnp.sum(
            jnp.stack(
                tuple(
                    jnp.sum(
                        measure.astype(jnp.float64)
                        * change.astype(jnp.float64)
                        * velocity_after.astype(jnp.float64)
                    )
                    for measure, change, velocity_after in zip(
                        measures, candidate.impulse, velocity_candidate, strict=True
                    )
                )
            )
        )
        cell_dynamic_work = work_step * jnp.sum(
            work_volumes * work_pressure * work_divergence
        )
        # Continuity has its own norm. Factoring the pressure-row differences
        # before reduction prevents that iterative residual from being
        # independently amplified by the absolute pressure level here.
        work_identity_residual = (
            work_step
            * jnp.sum(
                work_rates
                * (absolute_constraint_pressure - offset_mean - weighted_pressure)
            )
            + cell_dynamic_work
            - dynamic_work
        )
        dissipation = 0.5 * jnp.sum(
            jnp.stack(
                tuple(
                    jnp.sum(
                        measure.astype(jnp.float64)
                        * change.astype(jnp.float64)
                        * coefficient.astype(jnp.float64)
                        * change.astype(jnp.float64)
                    )
                    for measure, change, coefficient in zip(
                        measures, candidate.impulse, inverse, strict=True
                    )
                )
            )
        )
        kinetic_change = 0.5 * jnp.sum(
            jnp.stack(
                tuple(
                    jnp.sum(
                        measure.astype(jnp.float64)
                        * (
                            after.astype(jnp.float64) * velocity_after.astype(jnp.float64)
                            - before.astype(jnp.float64)
                            * velocity_start.astype(jnp.float64)
                        )
                    )
                    for measure, after, velocity_after, before, velocity_start in zip(
                        measures,
                        momentum_candidate,
                        velocity_candidate,
                        values,
                        velocity_before,
                        strict=True,
                    )
                )
            )
        )
        return (
            compartment_work,
            atmosphere_work,
            dynamic_work,
            offset_work,
            work_identity_residual,
            dissipation,
            kinetic_change,
            work_divergence,
        )

    def project(
        self,
        momentum: FaceVelocity,
        face_inverse_density: FaceVelocity,
        step_size: ArrayLike,
        constraint: MACCompartmentConstraint,
        /,
        *,
        pressure: ArrayLike | None = None,
    ) -> MACCompartmentProjectionResult:
        """Project face momentum ``rho_f u*`` onto the compartment constraints.

        ``pressure`` is the previous dynamic pressure: the initial guess of
        the ``u*`` lane and the pressure returned on failure.
        """
        operators = self.projection.operators
        dtype = operators.pressure_space.dtype
        capacity = self.compartment_capacity
        values = operators.validate_velocity(momentum)
        inverse = self.projection.validate_face_inverse_density(face_inverse_density)
        step = jnp.asarray(step_size, dtype=dtype).reshape(())
        incoming = (
            jnp.zeros(operators.discretization.cell_shape, dtype=dtype)
            if pressure is None
            else operators.validate_pressure(pressure)
        )
        support = self._support(constraint)
        volumes = operators.discretization.cell_volumes.astype(dtype)
        velocity_before = tuple(
            coefficient * value
            for coefficient, value in zip(inverse, values, strict=True)
        )
        divergence_before = operators.divergence(velocity_before)
        volume_flux = jnp.sum(volumes * divergence_before)
        # With a compartment or the atmosphere the volume row absorbs the net
        # flux, so the u* lane targets its constant mean and passes the owner's
        # compatibility test; without one that test applies unchanged. P0
        # removes the constant from every solve, so it carries no derivative.
        base_target = jax.lax.stop_gradient(
            jnp.full(
                incoming.shape,
                jnp.where(jnp.any(support.active), volume_flux / jnp.sum(volumes), 0.0),
                dtype=dtype,
            )
        )
        lanes = self._lane_solves(
            values,
            inverse,
            step,
            jax.lax.stop_gradient(incoming),
            base_target,
            support.weights,
        )
        candidate = self._recover(step, support, lanes, volume_flux)
        momentum_candidate = tuple(
            value + change
            for value, change in zip(values, candidate.impulse, strict=True)
        )
        velocity_candidate = tuple(
            coefficient * value
            for coefficient, value in zip(inverse, momentum_candidate, strict=True)
        )
        rates = candidate.rates
        atmosphere_rate = rates[capacity] if self.atmosphere else jnp.zeros((), dtype)
        closed_pressure = constraint.pressure.astype(dtype)
        next_pressure = closed_pressure + candidate.increment
        atmosphere_pressure = constraint.atmosphere_pressure.astype(dtype)
        residuals = self._constraint_residuals(
            support,
            lanes,
            candidate,
            velocity_candidate,
            next_pressure,
            atmosphere_pressure,
        )
        status, finite, gauge = self._status(
            support, lanes, candidate, momentum_candidate, next_pressure
        )
        successful = status == int(MACCompartmentProjectionStatus.CONVERGED)

        (
            compartment_work,
            atmosphere_work,
            dynamic_work,
            offset_work,
            work_identity_residual,
            dissipation,
            kinetic_change,
            work_divergence,
        ) = self._work_evidence(
            support,
            candidate,
            step,
            values,
            inverse,
            momentum_candidate,
            velocity_before,
            velocity_candidate,
            next_pressure,
            atmosphere_pressure,
        )

        def committed(value: Array) -> Array:
            return jnp.where(successful, value, jnp.zeros((), dtype=value.dtype))

        momentum_value = tuple(
            jnp.where(successful, candidate_value, original)
            for candidate_value, original in zip(momentum_candidate, values, strict=True)
        )
        velocity_value = tuple(
            coefficient * value
            for coefficient, value in zip(inverse, momentum_value, strict=True)
        )
        pressure_value = jnp.where(successful, candidate.pressure, incoming)
        rates_value = committed(rates)
        count = support.weights.shape[0]
        return MACCompartmentProjectionResult(
            momentum=momentum_value,
            velocity=velocity_value,
            pressure=pressure_value,
            absolute_pressure=pressure_value + support.offset,
            compartment_pressure=jnp.where(successful, next_pressure, closed_pressure),
            pressure_increment=committed(candidate.increment),
            volume_rate=rates_value[:capacity],
            atmosphere_volume_rate=committed(atmosphere_rate),
            divergence_before=divergence_before,
            divergence_after=jnp.where(
                successful, work_divergence.astype(dtype), divergence_before
            ),
            divergence_target=(
                rates_value @ support.weights.reshape((count, -1))
            ).reshape(incoming.shape),
            continuity_residual=residuals[0],
            continuity_scale=residuals[1],
            compartment_pressure_residual=residuals[2],
            atmosphere_pressure_residual=residuals[3],
            schur_residual=candidate.schur.diagnostics.relative_residual,
            compartment_work=committed(compartment_work),
            atmosphere_work=committed(atmosphere_work),
            dynamic_pressure_work=committed(dynamic_work),
            offset_pressure_work=committed(offset_work),
            work_identity_residual=committed(work_identity_residual),
            projection_dissipation=committed(dissipation),
            kinetic_energy_change=committed(kinetic_change),
            gauge=gauge,
            lane_converged=lanes.successful,
            pressure_solves=lanes.linear,
            schur=candidate.schur,
            valid_constraint=support.valid,
            finite=finite,
            status=status,
            successful=successful,
            projection_id=self.plan_id,
        )


__all__ = [
    "MACCompartmentBlockOperator",
    "MACCompartmentBlockVector",
    "MACCompartmentConstraint",
    "MACCompartmentGauge",
    "MACCompartmentProjectionPlan",
    "MACCompartmentProjectionResult",
    "MACCompartmentProjectionStatus",
]
