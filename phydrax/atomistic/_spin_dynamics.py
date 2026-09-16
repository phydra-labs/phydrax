#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..linalg import prepare_real_coordinate_tree
from ..metrix import (
    AbstractStateGeometry,
    GeodesicManifoldStateGeometry,
    SphereManifold,
)
from ..solver._differential import DifferentialProblem, WienerTerm
from ..solver._diffrax_backend import solve_diffrax
from ..solver._geometric import RKMK, SRKMK
from ..stochastic import WienerRealization
from ._spin import (
    ClassicalSpinState,
    evaluate_classical_spin_hamiltonian,
    PreparedClassicalSpinHamiltonian,
)


class ProductSphereStateGeometry(AbstractStateGeometry):
    """Product-S² geometry accepting public ``(site,3)`` and flat real storage."""

    base: GeodesicManifoldStateGeometry
    site_count: int = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(self, site_count: int, /):
        sites = int(site_count)
        if sites < 1:
            raise ValueError("Product sphere geometry requires at least one site.")
        base = GeodesicManifoldStateGeometry(SphereManifold(3))
        self.base = base
        self.site_count = sites
        self.geometry_id = f"{base.geometry_id}:product:{sites}"
        self.retraction_method = base.retraction_method
        self.trivial = False
        self.supports_exact_inverse = base.supports_exact_inverse
        self.supports_exact_differential = base.supports_exact_differential
        self.supports_transport = base.supports_transport
        self.supports_isometric_transport = base.supports_isometric_transport
        self.supports_commutator_free = base.supports_commutator_free

    def _matrix(self, value: ArrayLike, name: str, /) -> tuple[Array, bool]:
        array = jnp.asarray(value)
        if array.shape == (self.site_count, 3):
            return array, False
        if array.shape == (3 * self.site_count,):
            return array.reshape((self.site_count, 3)), True
        raise ValueError(
            f"{name} must have shape ({self.site_count}, 3) or "
            f"({3 * self.site_count},); got {array.shape}."
        )

    @staticmethod
    def _restore(value: Array, flattened: bool, /) -> Array:
        return value.reshape((-1,)) if flattened else value

    def contains(self, state: ArrayLike, /) -> Array:
        matrix, _ = self._matrix(state, "Product-sphere state")
        return self.base.contains(matrix)

    def project_tangent(self, state: ArrayLike, vector: ArrayLike, /) -> Array:
        matrix, flattened = self._matrix(state, "Product-sphere state")
        tangent, tangent_flat = self._matrix(vector, "Product-sphere tangent")
        if flattened != tangent_flat:
            raise ValueError("Product-sphere state and tangent storage must match.")
        return self._restore(self.base.project_tangent(matrix, tangent), flattened)

    def retract(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        matrix, flattened = self._matrix(state, "Product-sphere state")
        tangent, tangent_flat = self._matrix(
            local_tangent, "Product-sphere local tangent"
        )
        if flattened != tangent_flat:
            raise ValueError("Product-sphere state and local storage must match.")
        return self._restore(self.base.retract(matrix, tangent), flattened)

    def inverse_retract(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        matrix, flattened = self._matrix(state, "Product-sphere state")
        target, target_flat = self._matrix(point, "Product-sphere point")
        if flattened != target_flat:
            raise ValueError("Product-sphere point storage must match.")
        return self._restore(self.base.inverse_retract(matrix, target), flattened)

    def retraction_jvp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        local_velocity: ArrayLike,
        /,
    ) -> Array:
        matrix, flattened = self._matrix(state, "Product-sphere state")
        tangent, tangent_flat = self._matrix(
            local_tangent, "Product-sphere local tangent"
        )
        velocity, velocity_flat = self._matrix(
            local_velocity, "Product-sphere local velocity"
        )
        if flattened != tangent_flat or flattened != velocity_flat:
            raise ValueError("Product-sphere differential storage must match.")
        return self._restore(
            self.base.retraction_jvp(matrix, tangent, velocity), flattened
        )

    def retraction_inverse_jvp(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        matrix, flattened = self._matrix(state, "Product-sphere state")
        target, target_flat = self._matrix(point, "Product-sphere point")
        velocity, velocity_flat = self._matrix(tangent, "Product-sphere tangent")
        if flattened != target_flat or flattened != velocity_flat:
            raise ValueError("Product-sphere inverse differential storage must match.")
        return self._restore(
            self.base.retraction_inverse_jvp(matrix, target, velocity), flattened
        )

    def retraction_vjp(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        matrix, flattened = self._matrix(state, "Product-sphere state")
        tangent, tangent_flat = self._matrix(
            local_tangent, "Product-sphere local tangent"
        )
        covector, covector_flat = self._matrix(cotangent, "Product-sphere cotangent")
        if flattened != tangent_flat or flattened != covector_flat:
            raise ValueError("Product-sphere adjoint differential storage must match.")
        return self._restore(
            self.base.retraction_vjp(matrix, tangent, covector), flattened
        )

    def transport_tangent(
        self,
        state: ArrayLike,
        point: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        matrix, flattened = self._matrix(state, "Product-sphere state")
        target, target_flat = self._matrix(point, "Product-sphere point")
        vector, vector_flat = self._matrix(tangent, "Product-sphere tangent")
        if flattened != target_flat or flattened != vector_flat:
            raise ValueError("Product-sphere transport storage must match.")
        return self._restore(
            self.base.transport_tangent(matrix, target, vector), flattened
        )

    def transport_cotangent_pullback(
        self,
        state: ArrayLike,
        point: ArrayLike,
        cotangent: ArrayLike,
        /,
    ) -> Array:
        matrix, flattened = self._matrix(state, "Product-sphere state")
        target, target_flat = self._matrix(point, "Product-sphere point")
        covector, covector_flat = self._matrix(cotangent, "Product-sphere cotangent")
        if flattened != target_flat or flattened != covector_flat:
            raise ValueError("Product-sphere cotangent storage must match.")
        return self._restore(
            self.base.transport_cotangent_pullback(matrix, target, covector),
            flattened,
        )

    def cut_locus_margin(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        matrix, flattened = self._matrix(state, "Product-sphere state")
        target, target_flat = self._matrix(point, "Product-sphere point")
        if flattened != target_flat:
            raise ValueError("Product-sphere point storage must match.")
        return self.base.cut_locus_margin(matrix, target)


class LandauLifshitzGilbertPlan(StrictModule):
    """Fixed-step geometric Gilbert dynamics in explicit reduced/SI-compatible units.

    ``gyromagnetic_ratio`` is positive in rad/(time·field), ``damping`` is
    dimensionless, and temperature uses the prepared atomistic unit system. The
    deterministic drift is

    ``-gamma/(1+alpha^2) [m×H + alpha m×(m×H)]``.

    Supplying temperature selects the distinct Stratonovich thermal profile with
    field-noise variance ``2 alpha k_B T/(gamma mu)``.
    """

    hamiltonian: PreparedClassicalSpinHamiltonian
    gyromagnetic_ratio: Array
    damping: Array
    temperature: Array | None
    step_size: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    thermal: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian: PreparedClassicalSpinHamiltonian,
        gyromagnetic_ratio: ArrayLike,
        damping: ArrayLike,
        /,
        *,
        step_size: float,
        maximum_steps: int,
        temperature: ArrayLike | None = None,
    ):
        if not isinstance(hamiltonian, PreparedClassicalSpinHamiltonian):
            raise TypeError("hamiltonian must be PreparedClassicalSpinHamiltonian.")
        sites = hamiltonian.plan.site_count
        gamma = np.asarray(gyromagnetic_ratio)
        alpha = np.asarray(damping)
        if gamma.shape != (sites,) or alpha.shape != (sites,):
            raise ValueError("gamma and damping must have one value per spin site.")
        if np.any(~np.isfinite(gamma)) or np.any(~np.isfinite(alpha)):
            raise ValueError("gamma and damping must be finite.")
        active = np.asarray(hamiltonian.plan.site_mask, dtype=bool)
        if np.any(gamma[active] <= 0.0) or np.any(alpha[active] < 0.0):
            raise ValueError("Active gamma must be positive and damping non-negative.")
        step = float(step_size)
        steps = int(maximum_steps)
        if not isfinite(step) or step <= 0.0 or steps < 1:
            raise ValueError("LLG step_size and maximum_steps must be positive.")
        thermal = temperature is not None
        if thermal:
            temperature_host = np.asarray(temperature)
            if temperature_host.shape == ():
                temperature_host = np.full((sites,), temperature_host)
            if temperature_host.shape != (sites,):
                raise ValueError("Thermal LLG temperature must be scalar or per-site.")
            if np.any(~np.isfinite(temperature_host)) or np.any(
                temperature_host[active] <= 0.0
            ):
                raise ValueError("Active thermal LLG temperatures must be positive.")
            if np.any(alpha[active] <= 0.0):
                raise ValueError(
                    "Thermal LLG requires positive damping on every active site."
                )
            temperature_array: Array | None = jnp.asarray(
                np.where(active, temperature_host, 0.0)
            )
        else:
            temperature_array = None
        gamma = np.where(active, gamma, 0.0)
        alpha = np.where(active, alpha, 0.0)
        self.hamiltonian = hamiltonian
        self.gyromagnetic_ratio = jnp.asarray(gamma)
        self.damping = jnp.asarray(alpha)
        self.temperature = temperature_array
        self.step_size = step
        self.maximum_steps = steps
        self.thermal = thermal
        self.plan_id = canonical_fingerprint(
            {
                "kind": "landau-lifshitz-gilbert-plan",
                "hamiltonian": hamiltonian.prepared_id,
                "step_size": step,
                "maximum_steps": steps,
                "interpretation": "stratonovich" if thermal else "deterministic",
                "arrays": array_tree_fingerprint(
                    {
                        "gamma": gamma,
                        "damping": alpha,
                        **(
                            {}
                            if temperature_array is None
                            else {"temperature": np.asarray(temperature_array)}
                        ),
                    }
                ),
            }
        )


class PreparedLandauLifshitzGilbert(StrictModule):
    plan: LandauLifshitzGilbertPlan
    geometry: ProductSphereStateGeometry
    solver: RKMK | SRKMK
    thermal_field_amplitude: Array
    prepared_id: str = eqx.field(static=True)


class ClassicalSpinDynamicsState(StrictModule):
    """Restartable spin-only state, distinct from particle dynamics state."""

    directions: Array
    time: Array
    step_index: Array
    prepared_dynamics_id: str = eqx.field(static=True)
    wiener_realization_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        directions: ArrayLike,
        time: ArrayLike,
        step_index: ArrayLike,
        prepared_dynamics_id: str,
        /,
        *,
        wiener_realization_id: str | None = None,
    ):
        values = jnp.asarray(directions)
        time_ = jnp.asarray(time)
        step_ = jnp.asarray(step_index, dtype=jnp.int64)
        identifier = str(prepared_dynamics_id)
        if values.ndim != 2 or values.shape[-1] != 3:
            raise ValueError("Spin dynamics directions must have shape (site, 3).")
        if time_.shape != () or step_.shape != ():
            raise ValueError("Spin dynamics time and step_index must be scalar.")
        if not identifier:
            raise ValueError("prepared_dynamics_id must be non-empty.")
        if wiener_realization_id is not None and not str(wiener_realization_id):
            raise ValueError("wiener_realization_id must be non-empty or None.")
        self.directions = values
        self.time = time_
        self.step_index = step_
        self.prepared_dynamics_id = identifier
        self.wiener_realization_id = (
            None if wiener_realization_id is None else str(wiener_realization_id)
        )


class LLGDynamicsEvidence(StrictModule):
    maximum_norm_residual: Array
    maximum_tangent_residual: Array
    energy_drift: Array
    dissipative_energy_violation: Array
    fluctuation_dissipation_residual: Array
    backend_successful: Array
    all_states_finite: Array
    solver_id: str = eqx.field(static=True)
    interpretation: str = eqx.field(static=True)
    wiener_realization_id: str | None = eqx.field(static=True)
    coordinate_map_id: str | None = eqx.field(static=True)


class AtomisticSpinTrajectory(StrictModule):
    """Saved intrinsic spin trajectory and complete deterministic/path identity."""

    times: Array
    directions: Array
    valid: Array
    final_state: ClassicalSpinDynamicsState
    evidence: LLGDynamicsEvidence
    trajectory_id: str = eqx.field(static=True)


def prepare_llg_dynamics(
    plan: LandauLifshitzGilbertPlan,
    /,
) -> PreparedLandauLifshitzGilbert:
    if not isinstance(plan, LandauLifshitzGilbertPlan):
        raise TypeError("plan must be LandauLifshitzGilbertPlan.")
    geometry = ProductSphereStateGeometry(plan.hamiltonian.plan.site_count)
    solver: RKMK | SRKMK = SRKMK(geometry) if plan.thermal else RKMK(geometry)
    if plan.temperature is None:
        amplitude = jnp.zeros_like(plan.gyromagnetic_ratio)
    else:
        units = plan.hamiltonian.plan.system.plan.units
        amplitude = jnp.sqrt(
            2.0
            * plan.damping
            * units.boltzmann_constant
            * plan.temperature
            / (plan.gyromagnetic_ratio * plan.hamiltonian.moments)
        )
        amplitude = jnp.where(plan.hamiltonian.plan.site_mask, amplitude, 0.0)
    identifier = canonical_fingerprint(
        {
            "kind": "prepared-landau-lifshitz-gilbert",
            "plan": plan.plan_id,
            "geometry": geometry.geometry_id,
            "solver": solver.solver_id,
        }
    )
    return PreparedLandauLifshitzGilbert(plan, geometry, solver, amplitude, identifier)


def initial_spin_dynamics_state(
    prepared: PreparedLandauLifshitzGilbert,
    directions: ClassicalSpinState | ArrayLike,
    /,
    *,
    time: float = 0.0,
    step_index: int = 0,
) -> ClassicalSpinDynamicsState:
    if not isinstance(prepared, PreparedLandauLifshitzGilbert):
        raise TypeError("prepared must be PreparedLandauLifshitzGilbert.")
    spin = (
        directions
        if isinstance(directions, ClassicalSpinState)
        else ClassicalSpinState(
            directions,
            prepared.plan.hamiltonian.plan.site_mask,
            prepared.plan.hamiltonian.prepared_id,
            tolerance=prepared.plan.hamiltonian.plan.validation_tolerance * 10.0,
        )
    )
    if spin.hamiltonian_id != prepared.plan.hamiltonian.prepared_id:
        raise ValueError("Initial spin state belongs to another Hamiltonian.")
    time_ = float(time)
    step_ = int(step_index)
    if not isfinite(time_) or step_ < 0:
        raise ValueError("Initial spin time must be finite and step non-negative.")
    return ClassicalSpinDynamicsState(
        spin.directions,
        time_,
        step_,
        prepared.prepared_id,
    )


def _llg_field_action(
    directions: Array,
    field: Array,
    gamma: Array,
    alpha: Array,
    /,
) -> Array:
    first = jnp.cross(directions, field)
    second = jnp.cross(directions, first)
    scale = -gamma / (1.0 + alpha * alpha)
    return scale[:, None] * (first + alpha[:, None] * second)


def _llg_problem(
    prepared: PreparedLandauLifshitzGilbert,
    state: ClassicalSpinDynamicsState,
    final_time: float,
    /,
) -> DifferentialProblem:
    plan = prepared.plan

    def drift(time, directions, args):
        del time, args
        evaluation = evaluate_classical_spin_hamiltonian(plan.hamiltonian, directions)
        return _llg_field_action(
            directions,
            evaluation.effective_field,
            plan.gyromagnetic_ratio,
            plan.damping,
        )

    if plan.thermal:
        sites = plan.hamiltonian.plan.site_count

        def coefficient(time, directions, args):
            del time, args

            def action(noise):
                field = prepared.thermal_field_amplitude[:, None] * noise
                return _llg_field_action(
                    directions,
                    field,
                    plan.gyromagnetic_ratio,
                    plan.damping,
                )

            shape = jax.ShapeDtypeStruct((sites, 3), directions.dtype)
            return lx.FunctionLinearOperator(action, shape, closure_convert=False)

        noise = WienerTerm(
            "thermal-magnetic-field",
            coefficient,
            (sites, 3),
            structure="general",
            basis_id=f"llg-field-noise:{prepared.prepared_id}",
            representation="operator",
        )
        terms = (noise,)
        interpretation = "stratonovich"
    else:
        terms = ()
        interpretation = "ito"
    return DifferentialProblem(
        drift,
        state.directions,
        t0=state.time,
        t1=final_time,
        wiener_terms=terms,
        interpretation=interpretation,
        state_geometry=prepared.geometry,
        problem_id=f"llg:{prepared.prepared_id}",
    )


def solve_llg_dynamics(
    prepared: PreparedLandauLifshitzGilbert,
    state: ClassicalSpinDynamicsState,
    save_times: ArrayLike,
    /,
    *,
    realization: WienerRealization | None = None,
) -> AtomisticSpinTrajectory:
    """Run deterministic RKMK or thermal Stratonovich SRKMK without projection."""

    if not isinstance(prepared, PreparedLandauLifshitzGilbert):
        raise TypeError("prepared must be PreparedLandauLifshitzGilbert.")
    if not isinstance(state, ClassicalSpinDynamicsState):
        raise TypeError("state must be ClassicalSpinDynamicsState.")
    if state.prepared_dynamics_id != prepared.prepared_id:
        raise ValueError("Spin dynamics state belongs to another prepared runtime.")
    times_host = np.asarray(save_times)
    if times_host.ndim != 1 or times_host.size == 0:
        raise ValueError("save_times must be a nonempty vector.")
    if np.any(~np.isfinite(times_host)) or np.any(np.diff(times_host) <= 0.0):
        raise ValueError("save_times must be finite and strictly increasing.")
    if float(times_host[0]) < float(state.time):
        raise ValueError("save_times cannot precede the spin state time.")
    duration_steps = (float(times_host[-1]) - float(state.time)) / prepared.plan.step_size
    rounded_steps = int(round(duration_steps))
    if rounded_steps < 1 or not np.isclose(duration_steps, rounded_steps, atol=1.0e-10):
        raise ValueError("Final save time must lie on the fixed LLG step grid.")
    if rounded_steps > prepared.plan.maximum_steps:
        raise ValueError("Requested LLG interval exceeds maximum_steps.")
    if prepared.plan.thermal:
        if realization is None:
            raise ValueError("Thermal Stratonovich LLG requires a WienerRealization.")
    elif realization is not None:
        raise ValueError("Deterministic LLG does not accept a WienerRealization.")
    problem = _llg_problem(prepared, state, float(times_host[-1]))
    if realization is not None:
        if realization.noise_shape != problem.noise_shape:
            raise ValueError(
                f"LLG Wiener noise shape must be {problem.noise_shape}; "
                f"got {realization.noise_shape}."
            )
        if realization.noise_id != problem.noise_id:
            raise ValueError("LLG Wiener realization has the wrong noise identity.")
        if (
            state.wiener_realization_id is not None
            and state.wiener_realization_id != realization.realization_id
        ):
            raise ValueError("Thermal LLG restart state belongs to another Wiener path.")
        if realization.support[0] > float(state.time) or realization.support[1] < float(
            times_host[-1]
        ):
            raise ValueError("LLG Wiener support does not cover the solve interval.")
    state_coordinates = (
        prepare_real_coordinate_tree(state.directions, None)
        if prepared.plan.thermal
        else None
    )
    solution = solve_diffrax(
        problem,
        save_times=jnp.asarray(times_host),
        realization=realization,
        solver=prepared.solver,
        dt0=prepared.plan.step_size,
        max_steps=prepared.plan.maximum_steps,
        throw=False,
        state_coordinates=state_coordinates,
    )
    directions = jnp.asarray(solution.states)
    mask = prepared.plan.hamiltonian.plan.site_mask
    norms = jnp.sqrt(jnp.sum(directions * directions, axis=-1))
    norm_residual = jnp.max(
        jnp.where(mask[None, :], jnp.abs(norms - 1.0), 0.0), initial=0.0
    )

    def trajectory_values(value):
        evaluation = evaluate_classical_spin_hamiltonian(prepared.plan.hamiltonian, value)
        drift = _llg_field_action(
            value,
            evaluation.effective_field,
            prepared.plan.gyromagnetic_ratio,
            prepared.plan.damping,
        )
        tangent = jnp.max(jnp.abs(jnp.sum(value * drift, axis=-1)), initial=0.0)
        return evaluation.total_energy, tangent

    energies, tangent_residuals = jax.vmap(trajectory_values)(directions)
    energy_drift = jnp.max(jnp.abs(energies - energies[0]), initial=0.0)
    dissipative_violation = jnp.max(jnp.maximum(jnp.diff(energies), 0.0), initial=0.0)
    if prepared.plan.temperature is None:
        fdt_residual = jnp.asarray(0.0, dtype=directions.dtype)
    else:
        units = prepared.plan.hamiltonian.plan.system.plan.units
        reconstructed = (
            prepared.thermal_field_amplitude**2
            * prepared.plan.gyromagnetic_ratio
            * prepared.plan.hamiltonian.moments
            / (
                2.0
                * prepared.plan.damping
                * units.boltzmann_constant
                * prepared.plan.temperature
            )
        )
        fdt_residual = jnp.max(
            jnp.where(mask, jnp.abs(reconstructed - 1.0), 0.0), initial=0.0
        )
    realization_id = None if realization is None else realization.realization_id
    evidence = LLGDynamicsEvidence(
        norm_residual,
        jnp.max(tangent_residuals, initial=0.0),
        energy_drift,
        dissipative_violation,
        fdt_residual,
        jnp.asarray(solution.backend_successful),
        jnp.all(jnp.isfinite(directions)),
        prepared.solver.solver_id,
        "stratonovich" if prepared.plan.thermal else "deterministic",
        realization_id,
        None if state_coordinates is None else state_coordinates.coordinate_id,
    )
    final_state = ClassicalSpinDynamicsState(
        directions[-1],
        solution.times[-1],
        state.step_index + rounded_steps,
        prepared.prepared_id,
        wiener_realization_id=realization_id,
    )
    trajectory_id = canonical_fingerprint(
        {
            "kind": "atomistic-spin-trajectory",
            "prepared": prepared.prepared_id,
            "initial_step": int(state.step_index),
            "final_step": int(state.step_index) + rounded_steps,
            "save_times": times_host.tolist(),
            "wiener": realization_id,
        }
    )
    return AtomisticSpinTrajectory(
        solution.times,
        directions,
        solution.valid,
        final_state,
        evidence,
        trajectory_id,
    )


__all__ = [
    "AtomisticSpinTrajectory",
    "ClassicalSpinDynamicsState",
    "LLGDynamicsEvidence",
    "LandauLifshitzGilbertPlan",
    "PreparedLandauLifshitzGilbert",
    "ProductSphereStateGeometry",
    "initial_spin_dynamics_state",
    "prepare_llg_dynamics",
    "solve_llg_dynamics",
]
