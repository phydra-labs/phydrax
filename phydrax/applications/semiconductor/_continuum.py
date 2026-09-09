#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative isothermal homojunction Poisson/drift-diffusion.

Coordinates are (electrostatic potential/Vt, electron Fermi energy/kT,
hole Fermi energy/kT). Number fluxes are positive from tail to head;
terminal electric currents are positive from the external circuit INTO the
device. Exterior faces without a contact have zero displacement/number flux.

``residual`` has units 1/s and ``storage`` is dimensionless. Specifically,
carrier storage is the physical carrier count divided by ``volume * ni``;
carrier residual is the outgoing number rate plus integrated SRH divided by
that same count. Algebraic Poisson and contact rows are divided by the single
``time_scale``. Thus ``d(storage)/dt + residual = 0`` uses physical seconds.
The steady nonlinear solve multiplies ALL rows by ``time_scale``. No carrier
storage is placed on a prescribed contact or a dielectric coordinate row.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ...continuation import (
    ContinuationResult,
    continue_branch,
    NaturalParameterContinuation,
    ParameterPathContinuationProblem,
)
from ...linalg import (
    ArraySpace,
    DiagonalPairing,
    GMRES,
    ILUPreconditionerBuilder,
    JacobianLinearOperator,
    LinearSolvePolicy,
    PreconditioningPolicy,
    prepare_linearization,
    TolerancePolicy,
)
from ...nonlinear import (
    AbstractNonlinearMethod,
    ImplicitRootDerivativePolicy,
    JacobianPolicy,
    NewtonKrylov,
    NewtonTrustRegion,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
    prepare_nonlinear,
    PreparedNonlinearSolve,
    refresh_nonlinear,
    root,
    RootLineSearch,
    solve_prepared_nonlinear,
)
from ...sparse import (
    compile_sparse_jacobian,
    SparseColoring,
    SparseCoordinateOperator,
    SparseDerivativePlan,
    SparsePattern,
)
from ...sparse._coloring import native_coloring
from ._coupled import ClassicalPhysics
from ._device import DevicePlan
from ._quantities import ELEMENTARY_CHARGE_SI


class SemiconductorEvidence(StrictModule):
    """Physical closure evidence; all norms are independently recomputed."""

    finite: Array
    positive_densities: Array
    scaled_residual_norm: Array
    poisson_residual_norm: Array
    electron_residual_norm: Array
    hole_residual_norm: Array
    terminal_kcl_error: Array
    terminal_kcl_relative_error: Array
    charge_balance_error: Array
    charge_balance_relative_error: Array
    energy_balance_error: Array
    energy_balance_relative_error: Array
    nonlinear_successful: Array
    valid: Array


class SemiconductorOperatingPoint(StrictModule):
    """One solved bias, retaining native solver diagnostics even on failure."""

    coordinates: Array
    voltages: Array
    successful: Array
    nonlinear_result: NonlinearResult
    terminal_currents: Array
    terminal_charges: Array
    terminal_heat_flows: Array
    total_stored_energy: Array
    evidence: SemiconductorEvidence
    equilibrium_result: NonlinearResult | None = None
    initial_result: NonlinearResult | None = None


class SemiconductorSweepResult(StrictModule):
    """Ordered requested points and native adaptive attempt/rollback evidence.

    The sweep stops at the first unreachable requested voltage. A failed point
    is retained and marked invalid; no later requested point is fabricated.
    Each continuation result retains its committed checkpoint and all rejected
    attempts. ``requested_voltages`` distinguishes partial from complete runs.
    """

    points: tuple[SemiconductorOperatingPoint, ...]
    continuations: tuple[ContinuationResult, ...]
    requested_voltages: Array
    successful: Array

    @property
    def coordinates(self) -> Array:
        return jnp.stack(tuple(point.coordinates for point in self.points))

    @property
    def voltages(self) -> Array:
        return jnp.stack(tuple(point.voltages for point in self.points))

    @property
    def terminal_currents(self) -> Array:
        return jnp.stack(tuple(point.terminal_currents for point in self.points))

    @property
    def terminal_charges(self) -> Array:
        return jnp.stack(tuple(point.terminal_charges for point in self.points))

    @property
    def terminal_heat_flows(self) -> Array:
        return jnp.stack(tuple(point.terminal_heat_flows for point in self.points))

    @property
    def total_stored_energy(self) -> Array:
        return jnp.stack(tuple(point.total_stored_energy for point in self.points))


def _rms_space(size: int, dtype: Any) -> ArraySpace:
    """Mesh-size-independent nonlinear norms, with native Riesz semantics."""
    return ArraySpace(
        (size,),
        dtype=dtype,
        pairing=DiagonalPairing(
            jnp.full((size,), 1 / size, dtype=dtype),
            pairing_id=f"semiconductor-rms-{size}",
        ),
    )


def _harmonic(left: Array, right: Array) -> Array:
    total = left + right
    return 2 * left * right / jnp.where(total > 0, total, 1)


class _SemiconductorJacobian(StrictModule):
    """Rebindable numeric derivative on a fixed semiconductor sparse pattern."""

    derivative: SparseDerivativePlan

    def __call__(self, flat: Array, args: Any):
        return self.derivative.operator(flat)


def _state_pattern(plan: DevicePlan) -> SparsePattern:
    """Exact local/edge/interface dependency envelope for the selected layout."""
    node_count = int(plan.support.volumes.shape[0])
    nodes = np.arange(node_count, dtype=np.int32)
    bulk = np.asarray(plan.layout.node_indices(nodes), dtype=np.int32)
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []

    def couple(rows, columns):
        row_values = np.asarray(rows, dtype=np.int32).reshape(-1)
        column_values = np.asarray(columns, dtype=np.int32).reshape(-1)
        row_parts.append(
            np.broadcast_to(
                row_values[:, None], (row_values.size, column_values.size)
            ).reshape(-1)
        )
        column_parts.append(
            np.broadcast_to(
                column_values[None, :], (row_values.size, column_values.size)
            ).reshape(-1)
        )

    for node in range(node_count):
        couple(bulk[node], bulk[node])
    for tail, head in zip(
        np.asarray(plan.support.tail), np.asarray(plan.support.head), strict=True
    ):
        indices = np.concatenate((bulk[int(tail)], bulk[int(head)]))
        couple(indices, indices)
    for index, interface_nodes in enumerate(plan.interface_nodes):
        if plan.interfaces[index].electron_law is None:
            continue
        traces = np.concatenate(
            tuple(
                np.asarray(
                    plan.layout.indices(f"interface_{index}_{name}"),
                    dtype=np.int32,
                )
                for name in (
                    "electron_left",
                    "electron_right",
                    "hole_left",
                    "hole_right",
                )
            )
        )
        indices = np.concatenate(
            (bulk[interface_nodes[0]], bulk[interface_nodes[1]], traces)
        )
        couple(indices, indices)
    for index, binding in enumerate(plan.traps):
        trap = np.asarray(plan.layout.indices(f"trap_{index}"), dtype=np.int32)
        if binding.trap.population_kind == "bulk":
            adjacent = bulk[binding.location]
        else:
            left, right = plan.interface_nodes[binding.location]
            adjacent = np.concatenate((bulk[left], bulk[right]))
        indices = np.concatenate((adjacent, trap))
        couple(indices, indices)
    for channel in plan.tunneling:
        source = int(channel.path.path_nodes[0])
        destination = int(channel.path.path_nodes[-1])
        indices = np.concatenate((bulk[source], bulk[destination]))
        couple(indices, indices)
    return SparsePattern.from_coo(
        np.concatenate(row_parts),
        np.concatenate(column_parts),
        (plan.layout.size, plan.layout.size),
    )


class PreparedSemiconductorDevice(ClassicalPhysics):
    """Prepared classical topology, named state and native sparse solve route.

    Numerical coefficients and physical storage are derived from the current
    plan. Replacing topology, ownership or layout requires reprepare and the
    corresponding conservative transfer policy.
    """

    plan: DevicePlan
    coloring: SparseColoring
    poisson_coloring: SparseColoring

    def __init__(self, plan: DevicePlan):
        if not isinstance(plan, DevicePlan):
            raise TypeError("plan must be a DevicePlan.")
        count = int(plan.support.volumes.shape[0])
        nodes = np.arange(count, dtype=np.int32)
        tail = np.asarray(plan.support.tail)
        head = np.asarray(plan.support.head)
        rows = np.concatenate((nodes, tail, head))
        columns = np.concatenate((nodes, head, tail))
        poisson_pattern = SparsePattern.from_coo(rows, columns, (count, count))
        self.plan = plan
        self.coloring = native_coloring(
            _state_pattern(plan), derivative_kind="jacobian", mode="fwd"
        )
        self.poisson_coloring = native_coloring(
            poisson_pattern, derivative_kind="jacobian", mode="fwd"
        )

    @property
    def num_nodes(self) -> int:
        return int(self.plan.support.volumes.shape[0])

    @property
    def num_terminals(self) -> int:
        return len(self.plan.terminal_names)

    @property
    def count_scale(self) -> Array:
        """Intrinsic carrier count per dual volume (dimensionless number)."""
        return self.plan.support.volumes * self.plan.intrinsic_density

    @property
    def edge_capacitance(self) -> Array:
        support = self.plan.support
        return support.transmissibility * _harmonic(
            self.plan.permittivity[support.tail], self.plan.permittivity[support.head]
        )

    @property
    def charge_scale(self) -> Array:
        support = self.plan.support
        capacitance = self.edge_capacitance
        diagonal = jnp.zeros_like(support.volumes)
        diagonal = diagonal.at[support.tail].add(capacitance)
        diagonal = diagonal.at[support.head].add(capacitance)
        charge = (
            ELEMENTARY_CHARGE_SI
            * support.volumes
            * (
                self.plan.intrinsic_density
                + self.plan.donor_density
                + self.plan.acceptor_density
            )
        )
        return jnp.maximum(charge, self.plan.thermal_voltage * diagonal)

    @property
    def time_scale(self) -> Array:
        """Domain diffusion time in seconds; 1 s for pure dielectric algebra."""
        extent = jnp.max(self.plan.support.positions, axis=0) - jnp.min(
            self.plan.support.positions, axis=0
        )
        length_squared = jnp.sum(extent * extent)
        diffusion = self.plan.thermal_voltage * jnp.maximum(
            jnp.max(self.plan.electron_mobility), jnp.max(self.plan.hole_mobility)
        )
        return jnp.where(
            diffusion > 0, length_squared / jnp.where(diffusion > 0, diffusion, 1), 1.0
        )

    @property
    def differential_mask(self) -> Array:
        carriers = self.plan.semiconductor_mask & ~self.plan.ohmic_mask
        result = self.plan.layout.pack(
            potential=jnp.zeros_like(carriers),
            electron=carriers,
            hole=carriers,
        ).astype(bool)
        if self.plan.electrothermal:
            result = self.layout.set(result, "lattice_energy", ~self.plan.ohmic_mask)
        if self.plan.carrier_energy:
            result = self.layout.set(result, "electron_energy", carriers)
            result = self.layout.set(result, "hole_energy", carriers)
        for index in range(len(self.plan.traps)):
            result = self.layout.set(result, f"trap_{index}", True)
        return result

    def _coordinates(self, coordinates: Any) -> Array:
        value = jnp.asarray(coordinates)
        if value.shape != self.layout.shape:
            raise ValueError(f"coordinates must have shape {self.layout.shape}.")
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("coordinates must have a real floating dtype.")
        return value

    def _voltages(self, voltages: Any) -> Array:
        value = jnp.asarray(voltages, dtype=self.plan.support.positions.dtype)
        if value.shape != (self.num_terminals,):
            raise ValueError(f"voltages must have shape {(self.num_terminals,)}.")
        return value

    def jacobian(
        self, u: Any, voltages: Any, *, steady: bool = True
    ) -> JacobianLinearOperator:
        """Native flat matrix-free residual Jacobian with both JVP and VJP."""
        u = self._coordinates(u)
        voltages = self._voltages(voltages)
        scale = self.time_scale if steady else jnp.asarray(1, dtype=u.dtype)
        function = lambda flat: (
            scale * self.residual(flat.reshape(u.shape), voltages)
        ).reshape(-1)
        space = _rms_space(u.size, u.dtype)
        return JacobianLinearOperator(
            prepare_linearization(function, u.reshape(-1), source=space, target=space)
        )

    def storage_jacobian(self, u: Any) -> JacobianLinearOperator:
        """Native mass action: differentiates density, not QF coordinates alone."""
        u = self._coordinates(u)
        space = _rms_space(u.size, u.dtype)
        return JacobianLinearOperator(
            prepare_linearization(
                lambda flat: self.storage(flat.reshape(u.shape)).reshape(-1),
                u.reshape(-1),
                source=space,
                target=space,
            )
        )

    def _valid_coordinates(self, u: Array) -> Array:
        return self._state_valid(self._coordinates(u))

    def nonlinear_problem(self) -> NonlinearSystemProblem:
        dtype = self.plan.support.positions.dtype
        space = _rms_space(self.layout.size, dtype)
        return NonlinearSystemProblem(
            self.residual_function,
            state_space=space,
            residual_space=space,
            validity=lambda flat, residual, auxiliary, args: (
                self._state_valid(flat.reshape(self.layout.shape))
                & jnp.all(jnp.isfinite(residual))
            ),
            problem_id="semiconductor-classical-coupled",
        )

    def _derivative(
        self, initial: Array, *, reduced: bool = False
    ) -> SparseDerivativePlan:
        zero = jnp.zeros((self.num_terminals,), dtype=initial.dtype)
        if reduced:
            function = lambda potential, _: self._equilibrium_residual(potential, zero)
            coloring = self.poisson_coloring
        else:
            function = lambda flat, _: self.residual_function(flat, zero)
            coloring = self.coloring
        space = _rms_space(initial.size, initial.dtype)
        return compile_sparse_jacobian(
            function,
            initial,
            source=space,
            target=space,
            structure=coloring,
            compiler="native",
            mode="fwd",
        )

    def _method(self, initial: Array, *, reduced: bool = False) -> NewtonKrylov:
        derivative = self._derivative(initial, reduced=reduced)
        # Bias enters only affine contact rows, so this derivative is exact at
        # every voltage. Native continuation binds residuals with args=None;
        # the explicit policy evaluates the actual residual, not the zero-bias
        # primal stored by the derivative compiler.
        return NewtonKrylov(
            jacobian_policy=JacobianPolicy(
                "explicit", operator=_SemiconductorJacobian(derivative)
            ),
            linear_policy=LinearSolvePolicy(
                GMRES(restart=min(60, initial.size)),
                tolerance=TolerancePolicy(
                    relative=1e-10, absolute=0.0, max_steps=max(200, 4 * initial.size)
                ),
                preconditioning=PreconditioningPolicy(
                    ILUPreconditionerBuilder(), side="right", refresh="numeric"
                ),
            ),
            line_search=RootLineSearch(maximum_steps=30),
        )

    @staticmethod
    def _termination(termination: NonlinearTermination | None) -> NonlinearTermination:
        return (
            NonlinearTermination(
                absolute_residual=1e-8,
                relative_residual=0.0,
                absolute_step=0.0,
                relative_step=0.0,
                maximum_steps=150,
            )
            if termination is None
            else termination
        )

    def prepare(
        self,
        voltages: Any,
        initial: Any,
        *,
        method: AbstractNonlinearMethod | None = None,
        termination: NonlinearTermination | None = None,
        precision: Any = None,
    ) -> PreparedNonlinearSolve:
        """Prepare a native reusable nonlinear/linear symbolic solve route."""
        u = self._coordinates(
            initial.coordinates
            if isinstance(initial, SemiconductorOperatingPoint)
            else initial
        )
        return prepare_nonlinear(
            self.nonlinear_problem(),
            u.reshape(-1),
            method=self._method(u.reshape(-1)) if method is None else method,
            termination=self._termination(termination),
            args=self._voltages(voltages),
            precision=precision,
        )

    def refresh(
        self, prepared: PreparedNonlinearSolve, voltages: Any, initial: Any
    ) -> PreparedNonlinearSolve:
        """Refresh state, bias AND plan coefficients with fixed native topology.

        The built-in sparse derivative is rebound to this device's current SI
        arrays and normalization before native numeric/preconditioner refresh.
        Opaque user-supplied explicit/sparse derivative policies cannot be
        rebound safely: prepare a new solve for those policies.
        """
        if not isinstance(prepared, PreparedNonlinearSolve):
            raise TypeError("prepared must be a PreparedNonlinearSolve.")
        u = self._coordinates(
            initial.coordinates
            if isinstance(initial, SemiconductorOperatingPoint)
            else initial
        )
        policy = prepared.method.jacobian_policy
        if isinstance(policy.operator_function, _SemiconductorJacobian):
            old = policy.operator_function.derivative
            if old.pattern.pattern_id != self.coloring.pattern.pattern_id:
                raise ValueError(
                    "Changed device topology requires reprepare and conservative transfer."
                )
            rebound = JacobianPolicy(
                "explicit",
                operator=_SemiconductorJacobian(self._derivative(u.reshape(-1))),
            )
            prepared = eqx.tree_at(
                lambda value: value.method.jacobian_policy, prepared, rebound
            )
        elif policy.mode in ("explicit", "sparse"):
            raise ValueError(
                "An opaque user derivative requires prepare(), not numeric device refresh."
            )
        return refresh_nonlinear(
            prepared,
            self.nonlinear_problem(),
            u.reshape(-1),
            args=self._voltages(voltages),
        )

    def evidence(
        self,
        u: Any,
        voltages: Any,
        *,
        nonlinear_successful: Any = True,
        tolerance: float = 1e-8,
    ) -> SemiconductorEvidence:
        u = self._coordinates(u)
        voltage = self._voltages(voltages)
        scaled = self.time_scale * self.residual(u, voltage)
        rms = jnp.sqrt(jnp.mean(scaled**2))
        poisson_norm = jnp.max(jnp.abs(self.field(scaled, "potential")))
        electron_norm = jnp.max(jnp.abs(self.field(scaled, "electron")))
        hole_norm = jnp.max(jnp.abs(self.field(scaled, "hole")))
        currents, charges = self.terminal_current(u), self.terminal_charge(u)
        volume_charge = self.plan.support.volumes * self.charge_density(u)
        sheet_charge = jnp.sum(self.sheet_charge(u))
        kcl = jnp.abs(jnp.sum(currents))
        charge_error = jnp.abs(jnp.sum(charges) + jnp.sum(volume_charge) + sheet_charge)
        current_reference = (
            ELEMENTARY_CHARGE_SI * jnp.sum(self.count_scale) / self.time_scale
        )
        charge_reference = jnp.sum(self.charge_scale)
        kcl_relative = kcl / jnp.maximum(jnp.sum(jnp.abs(currents)), current_reference)
        charge_relative = charge_error / jnp.maximum(
            jnp.sum(jnp.abs(charges))
            + jnp.sum(jnp.abs(volume_charge))
            + jnp.abs(sheet_charge),
            charge_reference,
        )
        if self.plan.electrothermal:
            heat_flows = self.thermal_power(u)
            energy_error = jnp.abs(jnp.sum(voltage * currents) + jnp.sum(heat_flows))
            energy_reference = jnp.maximum(
                jnp.sum(jnp.abs(voltage * currents)) + jnp.sum(jnp.abs(heat_flows)),
                jnp.sum(self.plan.energy_scale) / self.time_scale,
            )
            energy_relative = energy_error / energy_reference
        else:
            heat_flows = jnp.empty((0,), dtype=u.dtype)
            energy_error = jnp.asarray(0.0, dtype=u.dtype)
            energy_relative = jnp.asarray(0.0, dtype=u.dtype)
        finite = (
            jnp.all(jnp.isfinite(u))
            & jnp.all(jnp.isfinite(scaled))
            & jnp.all(jnp.isfinite(currents))
            & jnp.all(jnp.isfinite(charges))
            & jnp.all(jnp.isfinite(heat_flows))
            & jnp.isfinite(energy_error)
        )
        positive = self._valid_coordinates(u)
        solver_ok = jnp.asarray(nonlinear_successful, dtype=bool)
        # Conservation is audited in physical SI independently of root status.
        floor = 100 * jnp.finfo(u.dtype).eps
        limit = jnp.maximum(jnp.asarray(tolerance, dtype=u.dtype), floor)
        valid = (
            finite
            & positive
            & solver_ok
            & (rms <= limit)
            & (kcl_relative <= 2 * limit)
            & (charge_relative <= limit)
            & (energy_relative <= 4 * limit)
        )
        return SemiconductorEvidence(
            finite=finite,
            positive_densities=positive,
            scaled_residual_norm=rms,
            poisson_residual_norm=poisson_norm,
            electron_residual_norm=electron_norm,
            hole_residual_norm=hole_norm,
            terminal_kcl_error=kcl,
            terminal_kcl_relative_error=kcl_relative,
            charge_balance_error=charge_error,
            charge_balance_relative_error=charge_relative,
            energy_balance_error=energy_error,
            energy_balance_relative_error=energy_relative,
            nonlinear_successful=solver_ok,
            valid=valid,
        )

    def _point(
        self,
        result: NonlinearResult,
        voltages: Array,
        termination: NonlinearTermination,
        *,
        initial: Array,
        equilibrium_result: NonlinearResult | None = None,
        initial_result: NonlinearResult | None = None,
        accepted: Any = True,
    ) -> SemiconductorOperatingPoint:
        u = result.state.reshape(self.layout.shape)
        initial_norm = jnp.sqrt(
            jnp.mean(self.residual_function(initial.reshape(-1), voltages) ** 2)
        )
        tolerance = termination.residual_threshold(initial_norm)
        evidence = self.evidence(
            u,
            voltages,
            nonlinear_successful=result.successful & jnp.asarray(accepted),
            tolerance=tolerance,
        )
        return SemiconductorOperatingPoint(
            coordinates=u,
            voltages=voltages,
            successful=evidence.valid,
            nonlinear_result=result,
            terminal_currents=self.terminal_current(u),
            terminal_charges=self.terminal_charge(u),
            terminal_heat_flows=self.thermal_power(u)
            if self.plan.electrothermal
            else jnp.empty((0,), dtype=u.dtype),
            total_stored_energy=self.total_energy(u)
            if self.plan.electrothermal
            else jnp.asarray(0.0, dtype=u.dtype),
            evidence=evidence,
            equilibrium_result=equilibrium_result,
            initial_result=initial_result,
        )

    def _equilibrium_residual(self, potential: Array, voltages: Array) -> Array:
        coordinates = self.layout.set(
            self.plan.equilibrium_coordinates(), "potential", potential
        )
        residual = self.time_scale * self.residual(coordinates, voltages)
        return self.field(residual, "potential")

    def equilibrium(
        self,
        *,
        method: AbstractNonlinearMethod | None = None,
        termination: NonlinearTermination | None = None,
        precision: Any = None,
    ) -> SemiconductorOperatingPoint:
        """Solve reduced nonlinear Poisson with one common Fermi energy.

        The common Fermi coordinates are exactly zero, so SG and SRH vanish
        identically rather than acquiring cancellation-sized equilibrium leaks.
        A monolithic residual certificate is still required for the result.
        """
        initial = self.plan.equilibrium_coordinates()
        zero = jnp.zeros((self.num_terminals,), dtype=initial.dtype)
        term = self._termination(termination)
        if self.layout.size != 3 * self.num_nodes:
            full = root(
                self.nonlinear_problem(),
                initial.reshape(-1),
                method=self._method(initial.reshape(-1)) if method is None else method,
                termination=term,
                args=zero,
                precision=precision,
            )
            return self._point(
                full,
                zero,
                term,
                initial=initial,
                equilibrium_result=full,
            )
        space = _rms_space(self.num_nodes, initial.dtype)
        problem = NonlinearSystemProblem(
            self._equilibrium_residual,
            state_space=space,
            residual_space=space,
            problem_id="semiconductor-equilibrium-poisson",
        )
        reduced = root(
            problem,
            self.field(initial, "potential"),
            method=self._method(self.field(initial, "potential"), reduced=True)
            if method is None
            else method,
            termination=term,
            args=zero,
            precision=precision,
        )
        coordinates = self.layout.set(initial, "potential", reduced.state)
        # Retain the actual reduced nonlinear result, not a fabricated full root.
        evidence = self.evidence(
            coordinates,
            zero,
            nonlinear_successful=reduced.successful,
            tolerance=term.residual_threshold(
                jnp.sqrt(
                    jnp.mean(
                        self._equilibrium_residual(self.field(initial, "potential"), zero)
                        ** 2
                    )
                )
            ),
        )
        return SemiconductorOperatingPoint(
            coordinates=coordinates,
            voltages=zero,
            successful=evidence.valid,
            nonlinear_result=reduced,
            terminal_currents=self.terminal_current(coordinates),
            terminal_charges=self.terminal_charge(coordinates),
            terminal_heat_flows=self.thermal_power(coordinates)
            if self.plan.electrothermal
            else jnp.empty((0,), dtype=coordinates.dtype),
            total_stored_energy=self.total_energy(coordinates)
            if self.plan.electrothermal
            else jnp.asarray(0.0, dtype=coordinates.dtype),
            evidence=evidence,
            equilibrium_result=reduced,
        )

    def solve(
        self,
        voltages: Any,
        initial: Any = None,
        *,
        method: AbstractNonlinearMethod | None = None,
        termination: NonlinearTermination | None = None,
        precision: Any = None,
        prepared: PreparedNonlinearSolve | None = None,
    ) -> SemiconductorOperatingPoint:
        """Monolithic biased root, preserving failed operating-point prerequisites.

        A bare coordinate array is an explicitly uncertified numerical seed.
        An operating point instead carries an acceptance prerequisite, retained
        in ``initial_result`` even if this new numerical root converges.
        """
        voltage = self._voltages(voltages)
        prerequisite = (
            self.equilibrium(termination=termination, precision=precision)
            if initial is None
            else (initial if isinstance(initial, SemiconductorOperatingPoint) else None)
        )
        u = self._coordinates(
            prerequisite.coordinates if prerequisite is not None else initial
        )
        term = self._termination(termination)
        if (
            prepared is None
            and method is not None
            and not isinstance(method, (NewtonKrylov, NewtonTrustRegion))
        ):
            result = root(
                self.nonlinear_problem(),
                u.reshape(-1),
                method=method,
                termination=term,
                args=voltage,
                precision=precision,
            )
        else:
            if prepared is None:
                native = self.prepare(
                    voltage, u, method=method, termination=term, precision=precision
                )
            else:
                if method is not None or precision is not None:
                    raise ValueError(
                        "A prepared solve already owns its method and precision."
                    )
                native = self.refresh(prepared, voltage, u)
            result = solve_prepared_nonlinear(native, termination=term)
        return self._point(
            result,
            voltage,
            term,
            initial=u,
            equilibrium_result=None
            if prerequisite is None
            else prerequisite.equilibrium_result,
            initial_result=None
            if prerequisite is None
            else prerequisite.nonlinear_result,
            accepted=True if prerequisite is None else prerequisite.successful,
        )

    def sweep(
        self,
        voltages_path: Any,
        initial: Any = None,
        *,
        method: AbstractNonlinearMethod | None = None,
        termination: NonlinearTermination | None = None,
        initial_step: float = 0.1,
        minimum_step: float = 1e-5,
        maximum_step: float = 1.0,
        maximum_retries: int = 12,
        maximum_steps: int = 1000,
    ) -> SemiconductorSweepResult:
        """Follow every requested voltage in order with native adaptive rollback.

        Each segment uses its own scalar path coordinate s in [0,1], so arbitrary
        multi-terminal paths and reversals preserve user ordering. A segment is
        successful only if native continuation actually reaches its endpoint;
        an intermediate committed state is never relabeled as the target bias.
        """
        path = jnp.asarray(voltages_path, dtype=self.plan.support.positions.dtype)
        if path.ndim != 2 or path.shape[1] != self.num_terminals or path.shape[0] < 1:
            raise ValueError("voltages_path must have shape (K, num_terminals), K >= 1.")
        if not bool(jnp.all(jnp.isfinite(path))):
            raise ValueError("voltages_path must be finite.")
        term = self._termination(termination)
        first = self.solve(path[0], initial=initial, method=method, termination=term)
        points = [first]
        continuations = []
        corrector = (
            self._method(first.coordinates.reshape(-1)) if method is None else method
        )
        problem_id = "semiconductor-voltage-path"
        flat = first.coordinates.reshape(-1)
        derivative = self._derivative(flat).operator(flat)
        coordinates = ArraySpace(
            flat.shape,
            dtype=flat.dtype,
            space_id=f"{problem_id}/state-tangent-coordinates",
        )
        setup = SparseCoordinateOperator(
            derivative.relation,
            derivative.coefficients,
            source=coordinates,
            target=coordinates,
        )
        tangent_policy = LinearSolvePolicy(
            GMRES(restart=min(60, flat.size)),
            tolerance=TolerancePolicy(
                relative=1e-6, absolute=0.0, max_steps=max(200, 4 * flat.size)
            ),
            preconditioning=PreconditioningPolicy(
                ILUPreconditionerBuilder(),
                setup_operator=setup,
                side="right",
                refresh="frozen",
            ),
        )
        policy = NaturalParameterContinuation(
            corrector=corrector,
            termination=term,
            derivative_policy=ImplicitRootDerivativePolicy(
                tangent_linear_policy=tangent_policy
            ),
            initial_step=initial_step,
            minimum_step=minimum_step,
            maximum_step=maximum_step,
            maximum_retries=maximum_retries,
            predictor="tangent",
            target_corrector_steps=min(4, term.maximum_steps),
        )
        for target in path[1:]:
            if not bool(points[-1].successful):
                break
            previous = points[-1]
            origin = previous.voltages
            segment = ParameterPathContinuationProblem(
                lambda flat, voltage, _: self.residual_function(flat, voltage),
                lambda s, _, start=origin, end=target: start + s * (end - start),
                origin,
                coordinate_lower=0.0,
                coordinate_upper=1.0,
                state_space=_rms_space(self.layout.size, path.dtype),
                residual_space=_rms_space(self.layout.size, path.dtype),
                problem_id=problem_id,
            )
            continued = continue_branch(
                segment,
                previous.coordinates.reshape(-1),
                jnp.asarray(0.0, dtype=path.dtype),
                num_steps=maximum_steps,
                method=policy,
                terminal_coordinate=1.0,
                branch_id=f"semiconductor-segment-{len(continuations)}",
            )
            continuations.append(continued)
            reached = bool(continued.points) and bool(
                continued.points[-1].coordinate == 1.0
            )
            if reached:
                last = continued.points[-1]
                point = self.solve(
                    target,
                    initial=last.state.reshape(self.layout.shape),
                    method=corrector,
                    termination=term,
                )
                valid = point.successful & continued.successful
                evidence = eqx.tree_at(
                    lambda value: (value.valid, value.nonlinear_successful),
                    point.evidence,
                    (valid, point.evidence.nonlinear_successful & continued.successful),
                )
                point = eqx.tree_at(
                    lambda value: (value.successful, value.evidence),
                    point,
                    (valid, evidence),
                )
                points.append(point)
            else:
                # Preserve the last committed physical voltage, not the target.
                if continued.points:
                    last = continued.points[-1]
                    point = self.solve(
                        last.parameters,
                        initial=last.state.reshape(self.layout.shape),
                        method=corrector,
                        termination=term,
                    )
                else:
                    point = previous
                evidence = eqx.tree_at(
                    lambda value: (value.valid, value.nonlinear_successful),
                    point.evidence,
                    (jnp.asarray(False), jnp.asarray(False)),
                )
                points.append(
                    eqx.tree_at(
                        lambda value: (value.successful, value.evidence),
                        point,
                        (jnp.asarray(False), evidence),
                    )
                )
                break
        successful = (len(points) == path.shape[0]) & jnp.all(
            jnp.stack(tuple(point.successful for point in points))
        )
        return SemiconductorSweepResult(
            points=tuple(points),
            continuations=tuple(continuations),
            requested_voltages=path,
            successful=jnp.asarray(successful),
        )


__all__ = [
    "PreparedSemiconductorDevice",
    "SemiconductorEvidence",
    "SemiconductorOperatingPoint",
    "SemiconductorSweepResult",
]
