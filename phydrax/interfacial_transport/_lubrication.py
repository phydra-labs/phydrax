#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-manifold lubrication drainage of thin liquid films.

State authority is the vertex liquid volume ``V_i``; the thickness
``h_i = V_i / A_i`` is derived from the barycentric dual area. One backward
Euler step solves the mixed content--pressure system

``A h - V^n + dt div F(h, p) = 0`` and ``p = Phi(h)``

with antisymmetric edge fluxes ``F_ij = w_ij m_ij (p_i - p_j)`` on
cotangent conductances ``w_ij`` and the film potential

``Phi = sigma_eff (K h / A - (kappa_1^2 + kappa_2^2) h) - Pi(h)
        - rho g.x [- rho (g.n) h for supported films]``.

The thickness unknown is ``ln h`` (a positive-domain iteration) and the edge
mobility is the entropy mean ``m_ij = (h_j - h_i) / (G'(h_j) - G'(h_i))`` with
``G'' = 1/m`` (Zhornitskaya & Bertozzi, SIAM J. Numer. Anal. 37, 2000;
Grün & Rumpf, Numer. Math. 87, 2000). The committed content is recomputed in
flux form from the converged solution, so totals are conserved to roundoff.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import fixed_field, parameter_field, ParameterOwner
from .._validation import positive_integer
from ..nonlinear import NonlinearStatus
from ..typing import ConvertibleToArray, parse
from ._disjoining import AbstractDisjoiningPressure
from ._film_contracts import PreparedFilmSurface
from ._film_evidence import FilmStepStatus, resolve_film_status, SurfaceFilmEvidence
from ._film_solve import film_termination, PreparedFilmNewton, vertex_block_pattern


FilmMobilityLaw: TypeAlias = Literal[
    "immobile-free-film", "one-sided-substrate", "navier-slip"
]
"""Lubrication mobility: ``h^3/(12 mu)`` for a symmetric free film with
immobile (no-slip) interfaces, ``h^3/(3 mu)`` for a film on a no-slip
substrate with a free surface, ``(h^3 + 3 b h^2)/(3 mu)`` with Navier slip
length ``b`` on the substrate."""

FilmConfiguration: TypeAlias = Literal["symmetric-free", "supported"]
"""Interface geometry: two interfaces displaced by ``+-h/2`` from the
midsurface (capillary coefficient ``sigma/2``) or one interface displaced by
``h`` from a substrate (coefficient ``sigma``)."""

FilmBoundaryPolicy: TypeAlias = Literal["no-flux", "fixed-thickness"]
"""``no-flux`` closes every boundary; ``fixed-thickness`` holds boundary
vertices at a prescribed thickness (a reservoir such as a Plateau border)
and reports the exchanged liquid volume."""


def film_capillary_pressure(
    surface: PreparedFilmSurface,
    thickness_m: ArrayLike,
    surface_tension_n_m: ArrayLike,
    /,
    *,
    configuration: FilmConfiguration,
) -> Array:
    """Return ``-sigma_eff (Delta_s h + (kappa_1^2 + kappa_2^2) h)`` at vertices.

    ``sigma_eff`` is ``sigma/2`` for a symmetric free film and ``sigma`` for a
    supported film. The Laplace--Beltrami operator is the cotangent stiffness
    over the barycentric dual area, and the curvature term is the prepared
    normal-cycle estimate, so the pressure is exact zero for a uniform film
    on a planar mesh.
    """
    if not isinstance(surface, PreparedFilmSurface):
        raise TypeError("surface must be a PreparedFilmSurface.")
    configuration = parse(configuration, FilmConfiguration, "configuration")
    thickness = jnp.asarray(thickness_m)
    if thickness.shape != (surface.topology.num_vertices,):
        raise ValueError("thickness_m must have shape (num_vertices,).")
    coefficient = _capillary_coefficient(configuration) * jnp.asarray(surface_tension_n_m)
    return coefficient * (
        surface.operators.apply_stiffness(thickness) / surface.vertex_area
        - surface.curvature_squared * thickness
    )


def _capillary_coefficient(configuration: FilmConfiguration, /) -> float:
    match configuration:
        case "symmetric-free":
            return 0.5
        case "supported":
            return 1.0
        case _:
            raise ValueError(f"Unknown film configuration {configuration!r}.")


def _configuration(law: FilmMobilityLaw, /) -> FilmConfiguration:
    match law:
        case "immobile-free-film":
            return "symmetric-free"
        case "one-sided-substrate" | "navier-slip":
            return "supported"
        case _:
            raise ValueError(f"Unknown film mobility law {law!r}.")


def _mobility(
    law: FilmMobilityLaw, thickness: Array, viscosity: Array, slip: Array, /
) -> Array:
    match law:
        case "immobile-free-film":
            return thickness**3 / (12.0 * viscosity)
        case "one-sided-substrate":
            return thickness**3 / (3.0 * viscosity)
        case "navier-slip":
            return (thickness**3 + 3.0 * slip * thickness**2) / (3.0 * viscosity)
        case _:
            raise ValueError(f"Unknown film mobility law {law!r}.")


def _entropy_density(
    law: FilmMobilityLaw, thickness: Array, viscosity: Array, slip: Array, /
) -> tuple[Array, Array]:
    """Return ``(G(h), G'(h))`` with ``G'' = 1/m``, ``G > 0`` convex."""
    match law:
        case "immobile-free-film" | "one-sided-substrate":
            scale = 12.0 if law == "immobile-free-film" else 3.0
            factor = scale * viscosity
            return factor / (2.0 * thickness), -factor / (2.0 * thickness**2)
        case "navier-slip":
            factor = 3.0 * viscosity
            shifted = thickness + 3.0 * slip
            value = factor * (
                -jnp.log(thickness) / (3.0 * slip)
                + (shifted * jnp.log(shifted) - thickness * jnp.log(thickness))
                / (9.0 * slip**2)
            )
            derivative = factor * (
                -1.0 / (3.0 * slip * thickness)
                + (jnp.log(shifted) - jnp.log(thickness)) / (9.0 * slip**2)
            )
            return value, derivative
        case _:
            raise ValueError(f"Unknown film mobility law {law!r}.")


def _edge_mobility(
    law: FilmMobilityLaw,
    first: Array,
    second: Array,
    viscosity: Array,
    slip: Array,
    /,
) -> Array:
    """Entropy-consistent edge mobility ``(h_j - h_i) / (G'(h_j) - G'(h_i))``."""
    match law:
        case "immobile-free-film" | "one-sided-substrate":
            scale = 12.0 if law == "immobile-free-film" else 3.0
            # Closed form of the entropy mean for m = h^3 / (scale mu).
            return 2.0 * first**2 * second**2 / (scale * viscosity * (first + second))
        case "navier-slip":
            _, first_derivative = _entropy_density(law, first, viscosity, slip)
            _, second_derivative = _entropy_density(law, second, viscosity, slip)
            difference = second - first
            separated = jnp.abs(difference) > 1e-6 * (first + second)
            safe = jnp.where(separated, second_derivative - first_derivative, 1.0)
            midpoint = _mobility(law, 0.5 * (first + second), viscosity, slip)
            return jnp.where(separated, difference / safe, midpoint)
        case _:
            raise ValueError(f"Unknown film mobility law {law!r}.")


class SurfaceLubricationPlan(StrictModule, ParameterOwner):
    """Physical lubrication model on one prepared manifold film surface.

    ``surface_tension_n_m`` is the single-interface tension; the symmetric
    free film applies its own ``sigma/2`` displacement factor. ``gravity_m_s2``
    is the gravitational acceleration vector. ``boundary_thickness_m`` is the
    prescribed boundary thickness for the ``fixed-thickness`` policy.
    ``rupture_thickness_m`` only flags vertices in the evidence; lubrication
    never mutates topology.
    """

    surface: PreparedFilmSurface = fixed_field()
    surface_tension_n_m: Array = parameter_field()
    viscosity_pa_s: Array = parameter_field()
    slip_length_m: Array = parameter_field()
    density_kg_m3: Array = parameter_field()
    gravity_m_s2: Array = parameter_field()
    boundary_thickness_m: Array = fixed_field()
    disjoining: AbstractDisjoiningPressure | None
    mobility_law: FilmMobilityLaw = eqx.field(static=True)
    boundary_policy: FilmBoundaryPolicy = eqx.field(static=True)
    rupture_thickness_m: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface: PreparedFilmSurface,
        /,
        *,
        mobility_law: FilmMobilityLaw,
        surface_tension_n_m: ArrayLike,
        viscosity_pa_s: ArrayLike,
        density_kg_m3: ArrayLike = 0.0,
        gravity_m_s2: ConvertibleToArray = (0.0, 0.0, 0.0),
        slip_length_m: ArrayLike = 0.0,
        disjoining: AbstractDisjoiningPressure | None = None,
        boundary_policy: FilmBoundaryPolicy = "no-flux",
        boundary_thickness_m: ArrayLike = 0.0,
        rupture_thickness_m: float = 0.0,
        tolerance: float = 1e-10,
        maximum_iterations: int = 40,
    ) -> None:
        if not isinstance(surface, PreparedFilmSurface):
            raise TypeError("surface must be a PreparedFilmSurface.")
        law = parse(mobility_law, FilmMobilityLaw, "mobility_law")
        policy = parse(boundary_policy, FilmBoundaryPolicy, "boundary_policy")
        if disjoining is not None and not isinstance(
            disjoining, AbstractDisjoiningPressure
        ):
            raise TypeError("disjoining must be an AbstractDisjoiningPressure or None.")
        tension = _positive(surface_tension_n_m, "surface_tension_n_m")
        viscosity = _positive(viscosity_pa_s, "viscosity_pa_s")
        density = _nonnegative(density_kg_m3, "density_kg_m3")
        slip = _nonnegative(slip_length_m, "slip_length_m")
        gravity = np.asarray(gravity_m_s2, dtype=np.float64)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("gravity_m_s2 must be a finite 3-vector.")
        if law == "navier-slip" and float(slip) <= 0.0:
            raise ValueError("navier-slip mobility requires a positive slip length.")
        vertex_count = surface.topology.num_vertices
        boundary = np.broadcast_to(
            np.asarray(boundary_thickness_m, dtype=np.float64), (vertex_count,)
        )
        if policy == "fixed-thickness":
            if surface.topology.watertight:
                raise ValueError("fixed-thickness boundaries require a bordered surface.")
            mask = np.asarray(surface.topology.boundary_vertices)
            if not np.all(np.isfinite(boundary[mask]) & (boundary[mask] > 0.0)):
                raise ValueError(
                    "Prescribed boundary thickness must be finite and positive."
                )
        rupture = float(rupture_thickness_m)
        if not np.isfinite(rupture) or rupture < 0.0:
            raise ValueError("rupture_thickness_m must be finite and nonnegative.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        iterations = positive_integer(maximum_iterations, "maximum_iterations")
        self.surface = surface
        self.surface_tension_n_m = tension
        self.viscosity_pa_s = viscosity
        self.slip_length_m = slip
        self.density_kg_m3 = density
        self.gravity_m_s2 = jnp.asarray(gravity)
        self.boundary_thickness_m = jnp.asarray(np.array(boundary))
        self.disjoining = disjoining
        self.mobility_law = law
        self.boundary_policy = policy
        self.rupture_thickness_m = rupture
        self.tolerance = tolerance_
        self.maximum_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "surface-lubrication",
                "operator_id": surface.topology.operator_id,
                "mobility_law": law,
                "boundary_policy": policy,
                "disjoining": None if disjoining is None else disjoining.law_id,
                "time_integration": "backward-euler-mixed-log-thickness",
                "edge_mobility": "entropy-mean",
                "tolerance": tolerance_,
                "maximum_iterations": iterations,
            }
        )

    @property
    def configuration(self) -> FilmConfiguration:
        return _configuration(self.mobility_law)

    def prepare(self) -> PreparedSurfaceLubrication:
        """Compile the sparse Jacobian and symbolic LU once for this topology."""
        return PreparedSurfaceLubrication(self)

    def potential(self, surface: PreparedFilmSurface, thickness: Array, /) -> Array:
        """Return the film potential ``Phi`` driving the edge fluxes."""
        pressure = film_capillary_pressure(
            surface,
            thickness,
            self.surface_tension_n_m,
            configuration=self.configuration,
        )
        if self.disjoining is not None:
            pressure = pressure - self.disjoining.pressure(thickness)
        weight = self.density_kg_m3
        pressure = pressure - weight * (surface.coordinates @ self.gravity_m_s2)
        if self.configuration == "supported":
            pressure = (
                pressure
                - weight * (surface.vertex_normal @ self.gravity_m_s2) * thickness
            )
        return pressure

    def energy(self, surface: PreparedFilmSurface, thickness: Array, /) -> Array:
        """Return the discrete free energy whose ``V``-gradient is ``Phi``."""
        coefficient = (
            _capillary_coefficient(self.configuration) * self.surface_tension_n_m
        )
        area = surface.vertex_area
        energy = (
            0.5
            * coefficient
            * (
                jnp.dot(thickness, surface.operators.apply_stiffness(thickness))
                - jnp.sum(area * surface.curvature_squared * thickness**2)
            )
        )
        if self.disjoining is not None:
            energy = energy + jnp.sum(area * self.disjoining.energy(thickness))
        weight = self.density_kg_m3
        energy = energy - weight * jnp.sum(
            area * thickness * (surface.coordinates @ self.gravity_m_s2)
        )
        if self.configuration == "supported":
            energy = energy - 0.5 * weight * jnp.sum(
                area * thickness**2 * (surface.vertex_normal @ self.gravity_m_s2)
            )
        return energy

    def entropy(self, surface: PreparedFilmSurface, thickness: Array, /) -> Array:
        """Return the discrete mobility entropy ``sum_i A_i G(h_i)``."""
        density, _ = _entropy_density(
            self.mobility_law, thickness, self.viscosity_pa_s, self.slip_length_m
        )
        return jnp.sum(surface.vertex_area * density)

    def convex_energy(self, surface: PreparedFilmSurface, /) -> Array:
        """Return whether the discrete energy is convex in the thickness."""
        convex = jnp.all(surface.curvature_squared == 0.0)
        if self.disjoining is not None:
            convex = convex & self.disjoining.monotone_decreasing()
        if self.configuration == "supported":
            convex = convex & (
                (self.density_kg_m3 == 0.0)
                | jnp.all(surface.vertex_normal @ self.gravity_m_s2 <= 0.0)
            )
        return convex

    def edge_flux(
        self, surface: PreparedFilmSurface, thickness: Array, potential: Array, /
    ) -> Array:
        """Return antisymmetric edge fluxes ``w m (Phi_i - Phi_j)`` in m^3/s."""
        edges = surface.topology.edges
        mobility = _edge_mobility(
            self.mobility_law,
            thickness[edges[:, 0]],
            thickness[edges[:, 1]],
            self.viscosity_pa_s,
            self.slip_length_m,
        )
        return (
            surface.operators.edge_weights * mobility * surface.edge_gradient(potential)
        )


class SurfaceLubricationState(StrictModule):
    """Vertex liquid volume on one film topology."""

    liquid_volume_m3: Array
    topology_id: str = eqx.field(static=True)

    def __init__(self, liquid_volume_m3: ArrayLike, /, *, topology_id: str) -> None:
        volume = jnp.asarray(liquid_volume_m3, dtype=jnp.float64)
        if volume.ndim != 1:
            raise ValueError("liquid_volume_m3 must be a vertex vector.")
        if not isinstance(topology_id, str) or not topology_id:
            raise ValueError("topology_id must be a non-empty string.")
        self.liquid_volume_m3 = volume
        self.topology_id = topology_id


class SurfaceLubricationStepResult(StrictModule):
    """Accepted state, candidate, status, evidence and boundary exchange.

    ``boundary_exchange_m3`` is the per-vertex fixed-thickness reservoir
    exchange over the step, positive when liquid enters the film. It is zero
    for ``no-flux`` boundaries; its sum is the scalar exchange in ``evidence``.
    """

    state: SurfaceLubricationState
    candidate_state: SurfaceLubricationState
    status: Array
    evidence: SurfaceFilmEvidence
    entropy_change: Array
    boundary_exchange_m3: Array

    def __init__(
        self,
        state: SurfaceLubricationState,
        candidate_state: SurfaceLubricationState,
        status: Array,
        evidence: SurfaceFilmEvidence,
        entropy_change: Array,
        boundary_exchange_m3: Array,
        /,
    ) -> None:
        self.state = state
        self.candidate_state = candidate_state
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.evidence = evidence
        self.entropy_change = jnp.asarray(entropy_change)
        self.boundary_exchange_m3 = jnp.asarray(boundary_exchange_m3)

    @property
    def accepted(self) -> Array:
        return self.status == FilmStepStatus.ACCEPTED


class PreparedSurfaceLubrication(StrictModule):
    """Plan plus the prepared native Newton solver for its fixed topology."""

    plan: SurfaceLubricationPlan
    solver: PreparedFilmNewton = fixed_field()
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: SurfaceLubricationPlan, /) -> None:
        if not isinstance(plan, SurfaceLubricationPlan):
            raise TypeError("plan must be a SurfaceLubricationPlan.")
        surface = plan.surface
        vertex_count = surface.topology.num_vertices
        thickness = jnp.ones((vertex_count,), dtype=jnp.float64)
        content = surface.vertex_area * thickness
        sample_args = _residual_arguments(plan, content, jnp.asarray(1.0))
        sample_state = jnp.concatenate(
            (jnp.zeros((vertex_count,)), jnp.zeros((vertex_count,)))
        )
        self.plan = plan
        self.solver = PreparedFilmNewton(
            _lubrication_residual,
            vertex_block_pattern(np.asarray(surface.topology.edges), vertex_count, 2),
            sample_state,
            sample_args,
            termination=film_termination(
                tolerance=plan.tolerance, maximum_iterations=plan.maximum_iterations
            ),
            solver_id=f"surface-lubrication/{plan.plan_id}",
        )
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-surface-lubrication", "plan_id": plan.plan_id}
        )

    def initial_state(self, thickness_m: ArrayLike, /) -> SurfaceLubricationState:
        """Return the extensive state ``V = A h`` for a vertex thickness field."""
        thickness = jnp.broadcast_to(
            jnp.asarray(thickness_m, dtype=jnp.float64),
            (self.plan.surface.topology.num_vertices,),
        )
        return SurfaceLubricationState(
            self.plan.surface.vertex_area * thickness,
            topology_id=self.plan.surface.topology.topology_id,
        )

    def with_surface(self, surface: PreparedFilmSurface, /) -> PreparedSurfaceLubrication:
        """Rebind refreshed geometry on identical topology without recompiling."""
        if not isinstance(surface, PreparedFilmSurface):
            raise TypeError("surface must be a PreparedFilmSurface.")
        if surface.topology.topology_id != self.plan.surface.topology.topology_id:
            raise ValueError("Refreshed surface must keep the prepared topology.")
        return eqx.tree_at(lambda value: value.plan.surface, self, surface)

    def thickness(self, state: SurfaceLubricationState, /) -> Array:
        return state.liquid_volume_m3 / self.plan.surface.vertex_area

    def step(
        self, state: SurfaceLubricationState, step_size_s: ArrayLike, /
    ) -> SurfaceLubricationStepResult:
        """Advance one backward-Euler step; reject the candidate on any failure."""
        if not isinstance(state, SurfaceLubricationState):
            raise TypeError("state must be a SurfaceLubricationState.")
        plan = self.plan
        surface = plan.surface
        if state.topology_id != surface.topology.topology_id:
            raise ValueError("State topology does not match the prepared surface.")
        if state.liquid_volume_m3.shape != (surface.topology.num_vertices,):
            raise ValueError("State liquid volume does not match the surface.")
        step_size = jnp.asarray(step_size_s, dtype=jnp.float64)
        content = state.liquid_volume_m3
        valid_input = (
            jnp.all(jnp.isfinite(content))
            & jnp.all(content > 0.0)
            & jnp.isfinite(step_size)
            & (step_size > 0.0)
        )
        safe_content = jnp.where(valid_input, content, surface.vertex_area)
        args = _residual_arguments(plan, safe_content, step_size)
        thickness = safe_content / surface.vertex_area
        initial = jnp.concatenate(
            (jnp.log(thickness), plan.potential(surface, thickness) / args[-1][1])
        )
        solution = self.solver.solve(initial, args)
        new_thickness = jnp.exp(solution.state[: surface.topology.num_vertices])
        potential = args[-1][1] * solution.state[surface.topology.num_vertices :]
        flux = plan.edge_flux(surface, new_thickness, potential)
        candidate = safe_content - step_size * surface.edge_divergence(flux)
        boundary_exchange = jnp.zeros_like(content)
        if plan.boundary_policy == "fixed-thickness":
            boundary = surface.topology.boundary_vertices
            held = surface.vertex_area * plan.boundary_thickness_m
            boundary_exchange = jnp.where(boundary, held - candidate, 0.0)
            candidate = jnp.where(boundary, held, candidate)
        exchange = jnp.sum(boundary_exchange)
        residual = jnp.sum(candidate) - jnp.sum(safe_content) - exchange
        converged = solution.status == NonlinearStatus.SUCCESS
        finite = jnp.all(jnp.isfinite(candidate)) & jnp.all(jnp.isfinite(potential))
        positive = jnp.all(candidate > 0.0)
        conductance = surface.evidence.admissible
        positivity = conductance & converged & finite & positive
        candidate_thickness = candidate / surface.vertex_area
        status = resolve_film_status(
            (FilmStepStatus.INADMISSIBLE_INPUT, ~valid_input),
            (FilmStepStatus.INADMISSIBLE_CONDUCTANCE, ~conductance),
            (FilmStepStatus.SOLVE_FAILED, ~converged),
            (FilmStepStatus.NONFINITE, ~finite),
            (FilmStepStatus.POSITIVITY_VIOLATED, ~positive),
        )
        accepted = status == FilmStepStatus.ACCEPTED
        energy_change = plan.energy(surface, candidate_thickness) - plan.energy(
            surface, thickness
        )
        entropy_change = plan.entropy(surface, candidate_thickness) - plan.entropy(
            surface, thickness
        )
        evidence = SurfaceFilmEvidence(
            liquid_volume_residual_m3=residual,
            boundary_exchange_m3=exchange,
            minimum_thickness_m=jnp.min(candidate_thickness),
            rupture_mask=candidate_thickness < plan.rupture_thickness_m,
            energy_change_j=energy_change,
            dissipation_guaranteed=positivity
            & plan.convex_energy(surface)
            & (plan.boundary_policy == "no-flux"),
            positivity_guaranteed=positivity,
            conductance_admissible=conductance,
            nonlinear_status=solution.status,
            nonlinear_iterations=solution.diagnostics.iterations,
            nonlinear_residual_norm=solution.diagnostics.final_residual_norm,
            converged=converged,
            finite=finite,
            geometry_revision=surface.geometry_revision,
        )
        topology_id = state.topology_id
        candidate_state = SurfaceLubricationState(candidate, topology_id=topology_id)
        accepted_state = SurfaceLubricationState(
            jnp.where(accepted, candidate, content), topology_id=topology_id
        )
        return SurfaceLubricationStepResult(
            accepted_state,
            candidate_state,
            status,
            evidence,
            entropy_change,
            boundary_exchange,
        )


def _residual_arguments(
    plan: SurfaceLubricationPlan, content: Array, step_size: Array, /
) -> tuple[SurfaceLubricationPlan, Array, Array, tuple[Array, Array]]:
    """Return residual arguments with dimensionless thickness/pressure scales."""
    surface = plan.surface
    thickness_scale = jnp.sum(content) / jnp.sum(surface.vertex_area)
    coefficient = _capillary_coefficient(plan.configuration) * plan.surface_tension_n_m
    pressure_scale = coefficient * thickness_scale / jnp.mean(surface.vertex_area)
    return plan, content, step_size, (thickness_scale, pressure_scale)


def _lubrication_residual(
    unknowns: Array,
    args: tuple[SurfaceLubricationPlan, Array, Array, tuple[Array, Array]],
) -> Array:
    plan, content, step_size, (thickness_scale, pressure_scale) = args
    surface = plan.surface
    vertex_count = surface.topology.num_vertices
    log_thickness = unknowns[:vertex_count]
    potential = pressure_scale * unknowns[vertex_count:]
    thickness = jnp.exp(log_thickness)
    flux = plan.edge_flux(surface, thickness, potential)
    content_residual = (
        surface.vertex_area * thickness
        - content
        + step_size * surface.edge_divergence(flux)
    ) / (surface.vertex_area * thickness_scale)
    if plan.boundary_policy == "fixed-thickness":
        content_residual = jnp.where(
            surface.topology.boundary_vertices,
            log_thickness - jnp.log(plan.boundary_thickness_m),
            content_residual,
        )
    pressure_residual = (potential - plan.potential(surface, thickness)) / pressure_scale
    return jnp.concatenate((content_residual, pressure_residual))


def _positive(value: ArrayLike, name: str, /) -> Array:
    host = np.asarray(value, dtype=np.float64)
    if host.shape != () or not np.isfinite(host) or host <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar.")
    return jnp.asarray(host)


def _nonnegative(value: ArrayLike, name: str, /) -> Array:
    host = np.asarray(value, dtype=np.float64)
    if host.shape != () or not np.isfinite(host) or host < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative scalar.")
    return jnp.asarray(host)


__all__ = [
    "FilmBoundaryPolicy",
    "FilmConfiguration",
    "FilmMobilityLaw",
    "PreparedSurfaceLubrication",
    "SurfaceLubricationPlan",
    "SurfaceLubricationState",
    "SurfaceLubricationStepResult",
    "film_capillary_pressure",
]
