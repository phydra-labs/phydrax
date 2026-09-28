#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Symmetric two-interface surfactant transport on a fixed film manifold.

Both interfaces of a symmetric film carry the same surface concentration
``Gamma = N / A`` where ``N`` is the amount on one interface of a barycentric
cell. Total interfacial amount, exchange and Marangoni force therefore carry
the explicit factor two; the single-interface laws never embed it. One step
applies optional conservative donor-cell advection of liquid volume,
interfacial and dissolved amounts by a tangential material velocity, then an
implicit surface-diffusion/adsorption solve through
``CoupledBulkSurfaceTransport`` with surface area ``2 A``, conductance
``2 D_s w`` and bulk volume equal to the film liquid volume.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import fixed_field, parameter_field, ParameterOwner
from .._validation import positive_integer
from ..linalg import PreparedSparseFactorization
from ._core import AdsorptionKinetics, LangmuirSurfactantLaw
from ._coupled import CoupledBulkSurfaceTransport
from ._film_contracts import PreparedFilmSurface
from ._film_evidence import FilmStepStatus, resolve_film_status
from ._film_transport import outflow_courant, upwind_edge_flux


class SymmetricFilmSurfactantState(StrictModule):
    """Extensive symmetric-film state on one film topology.

    ``surfactant_amount_mol`` is the amount on one interface per cell; the
    film carries twice this amount. ``dissolved_amount_mol`` is ``None`` when
    the film declares no bulk exchange.
    """

    liquid_volume_m3: Array
    surfactant_amount_mol: Array
    dissolved_amount_mol: Array | None
    geometry_revision: Array
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        liquid_volume_m3: ArrayLike,
        surfactant_amount_mol: ArrayLike,
        dissolved_amount_mol: ArrayLike | None,
        /,
        *,
        topology_id: str,
        geometry_revision: ArrayLike = 0,
    ) -> None:
        volume = jnp.asarray(liquid_volume_m3, dtype=jnp.float64)
        amount = jnp.asarray(surfactant_amount_mol, dtype=jnp.float64)
        if volume.ndim != 1 or amount.shape != volume.shape:
            raise ValueError(
                "Liquid volume and surfactant amount must be vertex vectors."
            )
        dissolved = (
            None
            if dissolved_amount_mol is None
            else jnp.asarray(dissolved_amount_mol, dtype=jnp.float64)
        )
        if dissolved is not None and dissolved.shape != volume.shape:
            raise ValueError("dissolved_amount_mol must match the vertex vector.")
        if not isinstance(topology_id, str) or not topology_id:
            raise ValueError("topology_id must be a non-empty string.")
        self.liquid_volume_m3 = volume
        self.surfactant_amount_mol = amount
        self.dissolved_amount_mol = dissolved
        self.geometry_revision = jnp.asarray(geometry_revision, dtype=jnp.int32)
        self.topology_id = topology_id

    def total_surfactant_mol(self) -> Array:
        """Return both interfaces plus dissolved surfactant."""
        total = 2.0 * jnp.sum(self.surfactant_amount_mol)
        if self.dissolved_amount_mol is not None:
            total = total + jnp.sum(self.dissolved_amount_mol)
        return total


class SurfactantTransportEvidence(StrictModule):
    """Conservation, support, Courant, finiteness, and solver evidence."""

    surfactant_residual_mol: Array
    liquid_volume_residual_m3: Array
    transferred_to_interfaces_mol: Array
    minimum_surface_concentration_mol_m2: Array
    maximum_coverage: Array
    minimum_surface_tension_n_m: Array
    surface_state_admissible: Array
    courant_number: Array
    nonlinear_status: Array
    nonlinear_iterations: Array
    nonlinear_residual_norm: Array
    conductance_admissible: Array
    finite: Array

    def __init__(
        self,
        *,
        surfactant_residual_mol: Array,
        liquid_volume_residual_m3: Array,
        transferred_to_interfaces_mol: Array,
        minimum_surface_concentration_mol_m2: Array,
        maximum_coverage: Array,
        minimum_surface_tension_n_m: Array,
        surface_state_admissible: Array,
        courant_number: Array,
        nonlinear_status: Array,
        nonlinear_iterations: Array,
        nonlinear_residual_norm: Array,
        conductance_admissible: Array,
        finite: Array,
    ) -> None:
        self.surfactant_residual_mol = jnp.asarray(surfactant_residual_mol)
        self.liquid_volume_residual_m3 = jnp.asarray(liquid_volume_residual_m3)
        self.transferred_to_interfaces_mol = jnp.asarray(transferred_to_interfaces_mol)
        self.minimum_surface_concentration_mol_m2 = jnp.asarray(
            minimum_surface_concentration_mol_m2
        )
        self.maximum_coverage = jnp.asarray(maximum_coverage)
        self.minimum_surface_tension_n_m = jnp.asarray(minimum_surface_tension_n_m)
        self.surface_state_admissible = jnp.asarray(
            surface_state_admissible, dtype=jnp.bool_
        )
        self.courant_number = jnp.asarray(courant_number)
        self.nonlinear_status = jnp.asarray(nonlinear_status, dtype=jnp.int32)
        self.nonlinear_iterations = jnp.asarray(nonlinear_iterations, dtype=jnp.int32)
        self.nonlinear_residual_norm = jnp.asarray(nonlinear_residual_norm)
        self.conductance_admissible = jnp.asarray(conductance_admissible, dtype=jnp.bool_)
        self.finite = jnp.asarray(finite, dtype=jnp.bool_)


class SymmetricFilmSurfactantStepResult(StrictModule):
    state: SymmetricFilmSurfactantState
    candidate_state: SymmetricFilmSurfactantState
    status: Array
    evidence: SurfactantTransportEvidence

    def __init__(
        self,
        state: SymmetricFilmSurfactantState,
        candidate_state: SymmetricFilmSurfactantState,
        status: Array,
        evidence: SurfactantTransportEvidence,
        /,
    ) -> None:
        self.state = state
        self.candidate_state = candidate_state
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.evidence = evidence

    @property
    def accepted(self) -> Array:
        return self.status == FilmStepStatus.ACCEPTED


class SymmetricFilmSurfactantPlan(StrictModule, ParameterOwner):
    """Symmetric two-interface surfactant model on a prepared film surface.

    ``kinetics=None`` declares insoluble surfactant (no dissolved amount).
    """

    surface: PreparedFilmSurface = fixed_field()
    law: LangmuirSurfactantLaw
    surface_diffusivity_m2_s: Array = parameter_field()
    kinetics: AdsorptionKinetics | None
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface: PreparedFilmSurface,
        law: LangmuirSurfactantLaw,
        /,
        *,
        surface_diffusivity_m2_s: ArrayLike,
        kinetics: AdsorptionKinetics | None = None,
        tolerance: float = 1e-12,
        maximum_iterations: int = 30,
    ) -> None:
        if not isinstance(surface, PreparedFilmSurface):
            raise TypeError("surface must be a PreparedFilmSurface.")
        if not isinstance(law, LangmuirSurfactantLaw):
            raise TypeError("law must be a LangmuirSurfactantLaw.")
        if kinetics is not None and not isinstance(kinetics, AdsorptionKinetics):
            raise TypeError("kinetics must be AdsorptionKinetics or None.")
        diffusivity = np.asarray(surface_diffusivity_m2_s, dtype=np.float64)
        if diffusivity.shape != () or not np.isfinite(diffusivity) or diffusivity < 0.0:
            raise ValueError("surface_diffusivity_m2_s must be finite and nonnegative.")
        if kinetics is not None and not np.isclose(
            float(kinetics.maximum_surface_concentration_mol_m2),
            float(law.maximum_surface_concentration_mol_m2),
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError(
                "Kinetics and equation of state must share Langmuir capacity."
            )
        self.surface = surface
        self.law = law
        self.surface_diffusivity_m2_s = jnp.asarray(diffusivity)
        self.kinetics = kinetics
        self.tolerance = float(tolerance)
        self.maximum_iterations = positive_integer(
            maximum_iterations, "maximum_iterations"
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "symmetric-film-surfactant",
                "operator_id": surface.topology.operator_id,
                "soluble": kinetics is not None,
                "leaflets": "symmetric-shared-concentration",
                "tolerance": self.tolerance,
                "maximum_iterations": self.maximum_iterations,
            }
        )

    def prepare(self) -> PreparedSymmetricFilmSurfactant:
        return PreparedSymmetricFilmSurfactant(self)

    def surface_concentration(self, state: SymmetricFilmSurfactantState, /) -> Array:
        return state.surfactant_amount_mol / self.surface.vertex_area

    def marangoni_force(self, state: SymmetricFilmSurfactantState, /) -> Array:
        """Return the symmetric-film Marangoni force ``2 int phi_i grad sigma``.

        The vertex force (N) is the exact adjoint of the dual-edge area flux:
        one third of every incident face area times the face gradient of
        ``sigma(Gamma)``, doubled for the two interfaces.
        """
        tension = self.law.evaluate(self.surface_concentration(state)).surface_tension_n_m
        return 2.0 * vertex_gradient_force(self.surface, tension)


def vertex_gradient_force(surface: PreparedFilmSurface, values: Array, /) -> Array:
    """Return ``sum_f (A_f / 3) grad_f u`` per vertex for scalar ``u``."""
    operators = surface.operators
    # Gradients own only tension differences; centering makes constant tension
    # exactly force-free instead of leaving a mesh-wide cancellation residual.
    relative = values - values[0]
    face_force = operators.face_area[:, None] * operators.gradient(relative) / 3.0
    force = jnp.zeros((surface.topology.num_vertices, 3), dtype=face_force.dtype)
    return force.at[operators.faces.reshape((-1,))].add(jnp.repeat(face_force, 3, axis=0))


def _langmuir_admissibility(
    law: LangmuirSurfactantLaw, concentration: Array, /
) -> tuple[Array, Array, Array]:
    """Return candidate status, tension and the aggregate law support mask."""
    evaluation = law.evaluate(concentration)
    tension = evaluation.surface_tension_n_m
    finite = jnp.all(jnp.isfinite(concentration)) & jnp.all(jnp.isfinite(tension))
    nonnegative = jnp.all(concentration >= 0.0)
    below_capacity = jnp.all(
        concentration < law.maximum_surface_concentration_mol_m2
    )
    positive_tension = jnp.all(tension > 0.0)
    status = resolve_film_status(
        (FilmStepStatus.NONFINITE, ~finite),
        (FilmStepStatus.POSITIVITY_VIOLATED, ~nonnegative),
        (FilmStepStatus.CAPACITY_EXCEEDED, ~below_capacity),
        (FilmStepStatus.NONPOSITIVE_TENSION, ~positive_tension),
    )
    return status, tension, jnp.all(evaluation.admissible)


class PreparedSymmetricFilmSurfactant(StrictModule):
    """Plan with its prepared implicit two-interface transport."""

    plan: SymmetricFilmSurfactantPlan
    transport: CoupledBulkSurfaceTransport = fixed_field()
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: SymmetricFilmSurfactantPlan, /) -> None:
        if not isinstance(plan, SymmetricFilmSurfactantPlan):
            raise TypeError("plan must be a SymmetricFilmSurfactantPlan.")
        topology = plan.surface.topology
        count = topology.num_vertices
        vertices = np.arange(count)
        soluble = plan.kinetics is not None
        self.plan = plan
        self.transport = CoupledBulkSurfaceTransport(
            np.asarray(topology.edges),
            vertices if soluble else np.zeros((0,), dtype=np.int64),
            vertices if soluble else np.zeros((0,), dtype=np.int64),
            np.ones((count,)) if soluble else np.zeros((0,)),
            plan.kinetics,
            bulk_size=count if soluble else 0,
            surface_size=count,
            tolerance=plan.tolerance,
            maximum_iterations=plan.maximum_iterations,
        )
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-symmetric-film-surfactant", "plan_id": plan.plan_id}
        )

    def initial_state(
        self,
        thickness_m: ArrayLike,
        surface_concentration_mol_m2: ArrayLike,
        dissolved_concentration_mol_m3: ArrayLike | None = None,
        /,
    ) -> SymmetricFilmSurfactantState:
        """Return extensive state from thickness, ``Gamma`` and bulk concentration."""
        surface = self.plan.surface
        count = surface.topology.num_vertices
        area = surface.vertex_area
        volume = area * jnp.broadcast_to(
            jnp.asarray(thickness_m, dtype=jnp.float64), (count,)
        )
        amount = area * jnp.broadcast_to(
            jnp.asarray(surface_concentration_mol_m2, dtype=jnp.float64), (count,)
        )
        if (self.plan.kinetics is None) != (dissolved_concentration_mol_m3 is None):
            raise ValueError(
                "Dissolved concentration is required exactly for soluble films."
            )
        dissolved = (
            None
            if dissolved_concentration_mol_m3 is None
            else volume
            * jnp.broadcast_to(
                jnp.asarray(dissolved_concentration_mol_m3, dtype=jnp.float64), (count,)
            )
        )
        return SymmetricFilmSurfactantState(
            volume,
            amount,
            dissolved,
            topology_id=surface.topology.topology_id,
            geometry_revision=surface.geometry_revision,
        )

    def rates(self, state: SymmetricFilmSurfactantState, /) -> tuple[Array, Array | None]:
        """Return per-interface surfactant and dissolved rates (mol/s).

        These are the diffusion/exchange rates whose implicit step ``step``
        solves; both interfaces share one concentration.
        """
        plan = self.plan
        surface = plan.surface
        _check_state(surface, state, soluble=plan.kinetics is not None)
        dissolved = state.dissolved_amount_mol
        bulk_rate, surface_rate = self.transport.rates(
            jnp.zeros((0,)) if dissolved is None else dissolved,
            2.0 * state.surfactant_amount_mol,
            jnp.zeros((0,)) if dissolved is None else state.liquid_volume_m3,
            2.0 * surface.vertex_area,
            2.0 * plan.surface_diffusivity_m2_s * surface.operators.edge_weights,
        )
        return 0.5 * surface_rate, None if dissolved is None else bulk_rate

    def _factorization(
        self, state: SymmetricFilmSurfactantState, step_size_s: ArrayLike, /
    ) -> PreparedSparseFactorization:
        """Factor the fixed insoluble surface-diffusion step."""
        plan = self.plan
        surface = plan.surface
        _check_state(surface, state, soluble=plan.kinetics is not None)
        if plan.kinetics is not None:
            raise ValueError("Soluble transport does not have a fixed linear operator.")
        step_size = jnp.asarray(step_size_s, dtype=jnp.float64)
        conductance = 2.0 * plan.surface_diffusivity_m2_s * surface.operators.edge_weights
        return self.transport._factorization(
            2.0 * surface.vertex_area, conductance, step_size
        )

    def step(
        self,
        state: SymmetricFilmSurfactantState,
        step_size_s: ArrayLike,
        /,
        *,
        tangential_velocity_m_s: ArrayLike | None = None,
    ) -> SymmetricFilmSurfactantStepResult:
        """Advect (optionally) and diffuse/exchange surfactant over one step."""
        return self._step(
            state,
            step_size_s,
            None,
            tangential_velocity_m_s=tangential_velocity_m_s,
        )

    def _step(
        self,
        state: SymmetricFilmSurfactantState,
        step_size_s: ArrayLike,
        factorization: PreparedSparseFactorization | None,
        /,
        *,
        tangential_velocity_m_s: ArrayLike | None = None,
    ) -> SymmetricFilmSurfactantStepResult:
        """Advance with an optional fixed surface-only diffusion factor."""
        plan = self.plan
        surface = plan.surface
        _check_state(surface, state, soluble=plan.kinetics is not None)
        step_size = jnp.asarray(step_size_s, dtype=jnp.float64)
        area = surface.vertex_area
        volume = state.liquid_volume_m3
        amount = state.surfactant_amount_mol
        dissolved = state.dissolved_amount_mol
        courant = jnp.zeros((), dtype=jnp.float64)
        if tangential_velocity_m_s is not None:
            velocity = jnp.asarray(tangential_velocity_m_s, dtype=jnp.float64)
            area_flux = surface.edge_area_flux(velocity)
            courant = outflow_courant(surface, area_flux, area, step_size)
            volume = volume - step_size * surface.edge_divergence(
                upwind_edge_flux(surface, volume, area_flux, area)
            )
            amount = amount - step_size * surface.edge_divergence(
                upwind_edge_flux(surface, amount, area_flux, area)
            )
            if dissolved is not None:
                dissolved = dissolved - step_size * surface.edge_divergence(
                    upwind_edge_flux(surface, dissolved, area_flux, area)
                )
        conductance = 2.0 * plan.surface_diffusivity_m2_s * surface.operators.edge_weights
        bulk_amount = jnp.zeros((0,)) if dissolved is None else dissolved
        bulk_volume = jnp.zeros((0,)) if dissolved is None else volume
        transport = self.transport.advance(
            bulk_amount,
            2.0 * amount,
            bulk_volume,
            2.0 * area,
            conductance,
            step_size,
            factorization=factorization,
        )
        candidate_amount = 0.5 * transport.candidate_surface_amount_mol
        candidate_dissolved = (
            None if dissolved is None else transport.candidate_bulk_amount_mol
        )
        candidate = SymmetricFilmSurfactantState(
            volume,
            candidate_amount,
            candidate_dissolved,
            topology_id=state.topology_id,
            geometry_revision=state.geometry_revision,
        )
        residual = candidate.total_surfactant_mol() - state.total_surfactant_mol()
        candidate_concentration = candidate_amount / area
        law_status, candidate_tension, law_admissible = _langmuir_admissibility(
            plan.law, candidate_concentration
        )
        conductance_ok = surface.evidence.admissible
        finite = (
            jnp.all(jnp.isfinite(volume))
            & jnp.all(jnp.isfinite(candidate_amount))
            & (
                jnp.asarray(True)
                if candidate_dissolved is None
                else jnp.all(jnp.isfinite(candidate_dissolved))
            )
        )
        status = resolve_film_status(
            (FilmStepStatus.INADMISSIBLE_CONDUCTANCE, ~conductance_ok),
            (FilmStepStatus.COURANT_LIMIT, courant > 1.0),
            (FilmStepStatus.NONFINITE, ~finite),
            (FilmStepStatus.POSITIVITY_VIOLATED, jnp.any(volume <= 0.0)),
        )
        status = jnp.where(status == FilmStepStatus.ACCEPTED, transport.status, status)
        status = jnp.where(status == FilmStepStatus.ACCEPTED, law_status, status)
        accepted = status == FilmStepStatus.ACCEPTED
        evidence = SurfactantTransportEvidence(
            surfactant_residual_mol=residual,
            liquid_volume_residual_m3=jnp.sum(volume) - jnp.sum(state.liquid_volume_m3),
            transferred_to_interfaces_mol=jnp.sum(transport.transferred_to_surface_mol),
            minimum_surface_concentration_mol_m2=jnp.min(candidate_concentration),
            maximum_coverage=(
                jnp.max(
                    candidate_concentration
                    / plan.law.maximum_surface_concentration_mol_m2
                )
                if plan.kinetics is None
                else transport.maximum_coverage
            ),
            minimum_surface_tension_n_m=jnp.min(candidate_tension),
            surface_state_admissible=law_admissible,
            courant_number=courant,
            nonlinear_status=transport.nonlinear_status,
            nonlinear_iterations=transport.nonlinear_iterations,
            nonlinear_residual_norm=transport.nonlinear_residual_norm,
            conductance_admissible=conductance_ok,
            finite=finite,
        )
        accepted_state = _select_state(accepted, candidate, state)
        return SymmetricFilmSurfactantStepResult(
            accepted_state, candidate, status, evidence
        )


def _select_state(
    condition: Array,
    selected: SymmetricFilmSurfactantState,
    fallback: SymmetricFilmSurfactantState,
    /,
) -> SymmetricFilmSurfactantState:
    """Select array leaves of two states with identical static structure."""
    dissolved = (
        None
        if selected.dissolved_amount_mol is None or fallback.dissolved_amount_mol is None
        else jnp.where(
            condition, selected.dissolved_amount_mol, fallback.dissolved_amount_mol
        )
    )
    return SymmetricFilmSurfactantState(
        jnp.where(condition, selected.liquid_volume_m3, fallback.liquid_volume_m3),
        jnp.where(
            condition, selected.surfactant_amount_mol, fallback.surfactant_amount_mol
        ),
        dissolved,
        topology_id=fallback.topology_id,
        geometry_revision=jnp.where(
            condition, selected.geometry_revision, fallback.geometry_revision
        ),
    )


def _check_state(
    surface: PreparedFilmSurface, state: SymmetricFilmSurfactantState, /, *, soluble: bool
) -> None:
    if not isinstance(state, SymmetricFilmSurfactantState):
        raise TypeError("state must be a SymmetricFilmSurfactantState.")
    if state.topology_id != surface.topology.topology_id:
        raise ValueError("State topology does not match the prepared surface.")
    if state.liquid_volume_m3.shape != (surface.topology.num_vertices,):
        raise ValueError("State does not match the surface vertex count.")
    if (state.dissolved_amount_mol is not None) != soluble:
        raise ValueError("State dissolved amount does not match the declared solubility.")


__all__ = [
    "PreparedSymmetricFilmSurfactant",
    "SurfactantTransportEvidence",
    "SymmetricFilmSurfactantPlan",
    "SymmetricFilmSurfactantState",
    "SymmetricFilmSurfactantStepResult",
    "vertex_gradient_force",
]
