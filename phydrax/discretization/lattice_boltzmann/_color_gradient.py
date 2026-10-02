#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Matched-density N-component color-gradient lattice Boltzmann dynamics.

Every component ``k`` of a declared, ordered set of ``N >= 2`` components carries its
own populations ``f_k[*grid, Q]``; the mixture ``f = sum_k f_k`` collides once with a
forced hydrodynamic method, is recolored and is routed component by component.

Interfaces are pairwise (Spencer, Halliday and Care, Phys. Rev. E 82, 066701, 2010).
For each unordered pair ``(k, l)``, ``k < l``, in component order:

- pair phase field ``phi_kl = (rho_k - rho_l) / (rho_k + rho_l)`` with normal
  ``n_kl = grad phi_kl / |grad phi_kl|`` pointing into ``k``;
- pair presence ``C_kl = min(1, c_k c_l / threshold)`` with ``c_k = rho_k / rho``
  (Leclaire, Reggio and Trepanier, J. Comput. Phys. 246, 318, 2013), which removes
  the ill-defined ``phi_kl`` where neither component is present;
- capillary force in conservative continuum-surface-stress form
  ``F = div S`` with ``S = sum_kl (sigma_kl / 2) C_kl |grad phi_kl| (I - n_kl n_kl)``
  (Lafaurie et al., J. Comput. Phys. 113, 134, 1994). In the continuum
  ``div(|grad phi| (I - n n)) = kappa grad phi`` with ``kappa = -div n``, so this is the
  Lishchuk--Care--Halliday body force ``(sigma / 2) kappa grad phi``; discretely the
  lattice divergence of a stress telescopes, so the capillary force carries no net
  momentum on a periodic lattice;
- recoloring (Latva-Kokko and Rothman, Phys. Rev. E 71, 056702, 2005, generalized
  pairwise by Spencer et al.)
  ``f_k,i = c_k f_i + sum_{l != k} beta_kl (rho_k rho_l / rho) w_i cos(n_kl, e_i)``
  with ``n_lk = -n_kl``; the pairwise terms cancel in ``sum_k f_k,i`` so the mixture
  population and all its moments are unchanged, and ``sum_i w_i cos(n, e_i) = 0``
  preserves every component mass.

Pairwise tensions ``sigma_kl`` come from `phydrax.interfacial_transport.InterfaceTensionMatrix`
whose label ids must equal the declared component ids. The binary model is the
``N = 2`` case of the same API. A near-contact repulsion
(`NearContactRepulsionPlan`) optionally adds a pairwise antisymmetric film force.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_where
from ..._validation import unique_identifiers
from ...qualification._registry import CapabilityProfile, SupportTuple
from ...typing import checked
from ._boundary import PreparedLatticeBoltzmannBoundary
from ._collision import macroscopic_raw_moments, quadratic_equilibrium
from ._discretization import LatticeBoltzmannDiscretization
from ._interfacial import (
    isotropic_divergence,
    normalized_gradient,
    static_contact_angle_normal,
)
from ._lattice import LatticeBoltzmannVelocitySet
from ._method import (
    LatticeBoltzmannMethodPlan,
    PreparedLatticeBoltzmannMethodPlan,
)
from ._near_contact import (
    NearContactForce,
    NearContactRepulsionEvidence,
    NearContactRepulsionPlan,
    PreparedNearContactRepulsion,
)
from ._program import coupled_population_manifest, KineticProgramManifest
from ._scaling import LatticeBoltzmannScaling
from ._thermodynamics import isotropic_tensor_divergence


if TYPE_CHECKING:
    from ...interfacial_transport import InterfaceTensionMatrix


def _component_pairs(count: int, /) -> tuple[tuple[int, int], ...]:
    """Canonical unordered component pairs ``(k, l)``, ``k < l``, in component order."""

    return tuple(itertools.combinations(range(count), 2))


def _pair_incidence(count: int, /) -> np.ndarray:
    """``[N, P]`` signs assigning each pair term to its first (+1) and second (-1) member."""

    pairs = _component_pairs(count)
    incidence = np.zeros((count, len(pairs)), dtype=np.float64)
    for index, (first, second) in enumerate(pairs):
        incidence[first, index] = 1.0
        incidence[second, index] = -1.0
    return incidence


def _pair_members(count: int, /) -> tuple[np.ndarray, np.ndarray]:
    pairs = _component_pairs(count)
    first = np.asarray([pair[0] for pair in pairs], dtype=np.int32)
    second = np.asarray([pair[1] for pair in pairs], dtype=np.int32)
    return first, second


def _pair_recoloring_strengths(
    value: float | Sequence[Sequence[float]], count: int, /
) -> tuple[float, ...]:
    host = np.asarray(value, dtype=np.float64)
    first, second = _pair_members(count)
    match host.ndim:
        case 0:
            strengths = np.full(first.shape, float(host), dtype=np.float64)
        case 2:
            if host.shape != (count, count):
                raise ValueError(
                    "A pairwise recoloring_strength must be one N x N matrix."
                )
            if not np.array_equal(host, host.T) or np.any(np.diagonal(host) != 0.0):
                raise ValueError(
                    "A pairwise recoloring_strength must be symmetric with zero diagonal."
                )
            strengths = host[first, second]
        case _:
            raise ValueError("recoloring_strength must be a scalar or an N x N matrix.")
    if not np.all(np.isfinite(strengths)) or np.any(
        (strengths <= 0.0) | (strengths > 1.0)
    ):
        raise ValueError("Every pairwise recoloring_strength must lie in (0, 1].")
    return tuple(float(item) for item in strengths)


class ColorGradientLBMState(StrictModule):
    """Component populations ``[N, *grid, Q]`` and the near-contact work ledger.

    ``near_contact_work`` is the cumulative physical work done by the near-contact
    repulsion on the fluid since initialization; only accepted steps commit it.
    """

    color_populations: Array
    near_contact_work: Array


class ColorGradientLBMRuntimeParameters(StrictModule):
    """Differentiable physical, wetting and repulsion controls for one rollout.

    ``surface_tension`` holds the pairwise tensions ``sigma_kl``; its label ids must
    equal the method component ids in order. ``contact_angles`` holds one static
    contact angle per canonical component pair ``(k, l)``, ``k < l``, measured
    through component ``k``; a scalar applies to every pair. ``near_contact_strength``
    holds one physical force density per declared repelling pair of the method's
    `NearContactRepulsionPlan` (empty without one).
    """

    kinematic_viscosity: Array
    surface_tension: InterfaceTensionMatrix
    near_contact_strength: Array
    moving_wall_velocities: Array
    wall_normal: Array
    wetting_mask: Array
    contact_angles: Array

    def __init__(
        self,
        kinematic_viscosity: ArrayLike,
        surface_tension: InterfaceTensionMatrix,
        /,
        *,
        near_contact_strength: ArrayLike | None = None,
        moving_wall_velocities: ArrayLike | None = None,
        wall_normal: ArrayLike | None = None,
        wetting_mask: ArrayLike | None = None,
        contact_angles: ArrayLike = 0.5 * jnp.pi,
    ) -> None:
        # The tension contract's owner imports the discretization package.
        from ...interfacial_transport import InterfaceTensionMatrix

        viscosity = jnp.asarray(kinematic_viscosity)
        if viscosity.shape != () or not jnp.issubdtype(viscosity.dtype, jnp.inexact):
            raise ValueError("kinematic_viscosity must be one inexact scalar array.")
        if not isinstance(surface_tension, InterfaceTensionMatrix):
            raise TypeError("surface_tension must be an InterfaceTensionMatrix.")
        pair_count = len(_component_pairs(surface_tension.label_count))
        angles = jnp.asarray(contact_angles, dtype=viscosity.dtype)
        if angles.shape == ():
            angles = jnp.broadcast_to(angles, (pair_count,))
        if angles.shape != (pair_count,):
            raise ValueError("contact_angles must be scalar or hold one angle per pair.")
        strengths = (
            jnp.empty((0,), dtype=viscosity.dtype)
            if near_contact_strength is None
            else jnp.asarray(near_contact_strength, dtype=viscosity.dtype)
        )
        if strengths.ndim != 1:
            raise ValueError("near_contact_strength must be one vector.")
        if (wall_normal is None) != (wetting_mask is None):
            raise ValueError("wall_normal and wetting_mask must be supplied together.")
        walls = (
            jnp.empty((0,), dtype=viscosity.dtype)
            if moving_wall_velocities is None
            else jnp.asarray(moving_wall_velocities, dtype=viscosity.dtype)
        )
        normals = (
            jnp.empty((0,), dtype=viscosity.dtype)
            if wall_normal is None
            else jnp.asarray(wall_normal, dtype=viscosity.dtype)
        )
        mask = (
            jnp.empty((0,), dtype=jnp.bool_)
            if wetting_mask is None
            else jnp.asarray(wetting_mask, dtype=jnp.bool_)
        )
        self.kinematic_viscosity = viscosity
        self.surface_tension = surface_tension
        self.near_contact_strength = strengths
        self.moving_wall_velocities = walls
        self.wall_normal = normals
        self.wetting_mask = mask
        self.contact_angles = angles


class ColorGradientLBMMethod(StrictModule, NonTrainableState):
    """Conservative pairwise recoloring over one forced LBM method.

    ``component_ids`` fixes the component identity and order. ``recoloring_strength``
    is one segregation parameter ``beta`` shared by every pair or a symmetric
    zero-diagonal ``N x N`` matrix ``beta_kl`` in component order.
    ``pair_presence_threshold`` is the Leclaire concentration product below which a
    pair's capillary stress is attenuated.
    """

    hydrodynamic_method: LatticeBoltzmannMethodPlan
    near_contact: NearContactRepulsionPlan | None
    component_ids: tuple[str, ...] = eqx.field(static=True)
    recoloring_strength: tuple[float, ...] = eqx.field(static=True)
    pair_presence_threshold: float = eqx.field(static=True)
    density_floor: float = eqx.field(static=True)
    gradient_floor: float = eqx.field(static=True)
    maximum_mach: float = eqx.field(static=True)
    maximum_capillary_number: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        hydrodynamic_method: LatticeBoltzmannMethodPlan,
        component_ids: Sequence[str],
        /,
        *,
        recoloring_strength: float | Sequence[Sequence[float]] = 0.7,
        near_contact: NearContactRepulsionPlan | None = None,
        pair_presence_threshold: float = 1.0e-6,
        density_floor: float = 1.0e-12,
        gradient_floor: float = 1.0e-14,
        maximum_mach: float = 0.3,
        maximum_capillary_number: float = 1.0,
        conservation_tolerance: float = 1.0e-11,
    ) -> None:
        if hydrodynamic_method.forcing is None:
            raise ValueError("Color-gradient capillary forcing requires a forced method.")
        if isinstance(component_ids, str):
            raise TypeError("component_ids must be a sequence of identifiers.")
        identifiers = unique_identifiers(component_ids, "component_ids")
        if len(identifiers) < 2:
            raise ValueError("A color-gradient method requires at least two components.")
        if near_contact is not None and not isinstance(
            near_contact, NearContactRepulsionPlan
        ):
            raise TypeError("near_contact must be NearContactRepulsionPlan or None.")
        if near_contact is not None:
            unknown = sorted(
                {value for pair in near_contact.repelling_pairs for value in pair}
                - set(identifiers)
            )
            if unknown:
                raise ValueError(f"Repelling pairs name unknown components {unknown}.")
        strengths = _pair_recoloring_strengths(recoloring_strength, len(identifiers))
        values = tuple(
            float(value)
            for value in (
                pair_presence_threshold,
                density_floor,
                gradient_floor,
                maximum_mach,
                maximum_capillary_number,
                conservation_tolerance,
            )
        )
        presence, rho_floor, grad_floor, mach, capillary, tolerance = values
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Color-gradient method limits must be finite and positive.")
        if presence > 0.25:
            raise ValueError("pair_presence_threshold must not exceed 1/4.")
        if mach >= 1.0:
            raise ValueError("maximum_mach must be smaller than one.")
        self.hydrodynamic_method = hydrodynamic_method
        self.near_contact = near_contact
        self.component_ids = identifiers
        self.recoloring_strength = strengths
        self.pair_presence_threshold = presence
        self.density_floor = rho_floor
        self.gradient_floor = grad_floor
        self.maximum_mach = mach
        self.maximum_capillary_number = capillary
        self.conservation_tolerance = tolerance
        self.method_id = canonical_fingerprint(
            {
                "kind": "color-gradient-lattice-boltzmann-method",
                "hydrodynamic_method": hydrodynamic_method.method_id,
                "component_ids": list(identifiers),
                "recoloring_strength": list(strengths),
                "near_contact": None if near_contact is None else near_contact.plan_id,
                "capillary_force": "continuum-surface-stress",
                "pair_presence_threshold": presence,
                "density_floor": rho_floor,
                "gradient_floor": grad_floor,
                "maximum_mach": mach,
                "maximum_capillary_number": capillary,
                "conservation_tolerance": tolerance,
            }
        )

    @property
    def component_count(self) -> int:
        return len(self.component_ids)

    @property
    def component_pairs(self) -> tuple[tuple[str, str], ...]:
        """Canonical unordered component pairs; every pair axis follows this order."""

        return tuple(
            (self.component_ids[first], self.component_ids[second])
            for first, second in _component_pairs(self.component_count)
        )


class ColorGradientInterfacialFields(StrictModule):
    """Pair-resolved diffuse-interface geometry and the conservative capillary force.

    Pair axes follow `ColorGradientLBMMethod.component_pairs`; all values are in
    lattice units. ``normals`` are the (wetting-imposed) unit normals of
    ``phase_fields`` pointing into the first pair member, ``curvatures`` are
    ``-div n``, ``surface_deltas`` are ``|grad phi| / 2`` and ``pair_presence`` is the
    Leclaire attenuation. ``force_density`` is the divergence of the summed stress.
    """

    phase_fields: Array
    normals: Array
    curvatures: Array
    surface_deltas: Array
    pair_presence: Array
    force_density: Array


class ColorGradientMacroscopicState(StrictModule):
    """Physical component densities, velocity and pressure with lattice interfaces."""

    component_densities: Array
    density: Array
    concentrations: Array
    velocity: Array
    pressure: Array
    interfacial: ColorGradientInterfacialFields
    near_contact_force: Array


class RecoloringConservation(StrictModule):
    component_mass_defect: Array
    population_closure_defect: Array
    momentum_closure_defect: Array


class ColorGradientDiagnostics(StrictModule):
    """Lattice-unit masses and momentum with admissibility and conservation evidence.

    ``capillary_net_force_residual`` is the norm of the summed capillary force relative
    to its summed magnitude; the stress form makes it roundoff on periodic lattices.
    """

    component_masses: Array
    total_mass: Array
    component_mass_defects: Array
    total_mass_defect: Array
    total_momentum: Array
    minimum_component_density: Array
    minimum_density: Array
    maximum_mach: Array
    maximum_capillary_number: Array
    force_norm: Array
    capillary_net_force_residual: Array
    recoloring: RecoloringConservation


class ColorGradientStepResult(StrictModule):
    """One atomic step; ``near_contact`` is the evidence of the applied repulsion."""

    candidate_state: ColorGradientLBMState
    accepted_state: ColorGradientLBMState
    successful: Array
    residual: Array
    work: Array
    diagnostics: ColorGradientDiagnostics
    near_contact: NearContactRepulsionEvidence


class _ColorGradientFields(StrictModule):
    component_densities: Array
    density: Array
    concentrations: Array
    raw_momentum: Array
    velocity: Array
    interfacial: ColorGradientInterfacialFields
    near_contact: NearContactForce


class _WettingData(StrictModule):
    wall_normal: Array | None
    wetting_mask: Array | None
    contact_angles: Array
    valid: Array


def recolor_populations(
    total_populations: ArrayLike,
    component_densities: ArrayLike,
    pair_normals: ArrayLike,
    velocity_set: LatticeBoltzmannVelocitySet,
    recoloring_strength: ArrayLike,
    /,
    *,
    density_floor: ArrayLike = 1.0e-14,
) -> Array:
    """Conservatively split a mixture population into ``N`` component populations.

    ``component_densities`` is ``[N, *grid]``, ``pair_normals`` is ``[P, *grid, d]`` and
    ``recoloring_strength`` is ``[P]`` over the canonical pairs ``(k, l)``, ``k < l``.
    The split preserves every component zeroth moment and the complete mixture
    population direction by direction, so every mixture moment is unchanged.
    """

    populations = jnp.asarray(total_populations)
    densities = jnp.asarray(component_densities, dtype=populations.dtype)
    normals = jnp.asarray(pair_normals, dtype=populations.dtype)
    if densities.ndim < 2 or densities.shape[0] < 2:
        raise ValueError("component_densities must be [N, *grid] with N >= 2.")
    count = densities.shape[0]
    grid = densities.shape[1:]
    first, second = _pair_members(count)
    if populations.shape != (*grid, velocity_set.population_count):
        raise ValueError("Population and component-density shapes are incompatible.")
    if normals.shape != (first.size, *grid, velocity_set.dimension):
        raise ValueError("pair_normals must hold one normal field per component pair.")
    beta = jnp.asarray(recoloring_strength, dtype=populations.dtype)
    floor = jnp.asarray(density_floor, dtype=populations.dtype)
    if beta.shape != (first.size,) or floor.shape != ():
        raise ValueError("Recoloring needs one strength per pair and a scalar floor.")
    beta = eqx.error_if(
        beta,
        jnp.any(~jnp.isfinite(beta) | (beta <= 0.0) | (beta > 1.0)),
        "recoloring_strength must lie in (0, 1].",
    )
    floor = eqx.error_if(
        floor,
        ~jnp.isfinite(floor) | (floor <= 0.0),
        "density_floor must be finite and positive.",
    )
    density = jnp.sum(densities, axis=0)
    safe_density = jnp.maximum(density, floor)
    velocities = jnp.asarray(velocity_set.velocities, dtype=populations.dtype)
    speed = jnp.sqrt(ein.contract("qd,qd->q", velocities, velocities))
    direction = velocities / jnp.where(speed > 0.0, speed, 1.0)[:, None]
    cosine = ein.contract("p...d,qd->p...q", normals, direction)
    weights = jnp.asarray(velocity_set.weights, dtype=populations.dtype)
    pair_beta = beta.reshape((first.size,) + (1,) * len(grid))
    segregation = (
        (pair_beta * (densities[first] * densities[second] / safe_density))[..., None]
        * weights
        * cosine
    )
    incidence = jnp.asarray(_pair_incidence(count), dtype=populations.dtype)
    fractions = densities / safe_density
    components = fractions[..., None] * populations + ein.contract(
        "np,p...q->n...q", incidence, segregation
    )
    occupied = density > floor
    return jnp.where(occupied[..., None], components, 0.0)


class PreparedColorGradientLBMDynamics(StrictModule, NonTrainableState):
    """Pure matched-density color-gradient collide, recolor, and route dynamics."""

    discretization: LatticeBoltzmannDiscretization
    scaling: LatticeBoltzmannScaling
    method: ColorGradientLBMMethod
    hydrodynamic_method: PreparedLatticeBoltzmannMethodPlan
    near_contact: PreparedNearContactRepulsion | None
    program_manifest: KineticProgramManifest
    boundary: PreparedLatticeBoltzmannBoundary
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        discretization: LatticeBoltzmannDiscretization,
        scaling: LatticeBoltzmannScaling,
        method: ColorGradientLBMMethod,
        boundary: PreparedLatticeBoltzmannBoundary,
        /,
    ) -> None:
        if boundary.discretization.prepared_id != discretization.prepared_id:
            raise ValueError("Boundary and color-gradient discretizations do not match.")
        if not np.isclose(
            float(scaling.sound_speed_squared),
            float(discretization.velocity_set.sound_speed_squared),
        ):
            raise ValueError("Scaling and velocity-set sound speeds do not match.")
        if not np.isclose(float(scaling.cell_size), float(discretization.cell_size)):
            raise ValueError("Scaling and discretization cell sizes do not match.")
        hydrodynamic_method = method.hydrodynamic_method.prepare(
            discretization.velocity_set,
            discretization.precision,
        )
        near_contact = (
            None
            if method.near_contact is None
            else method.near_contact.prepare(method.component_ids, discretization)
        )
        program_manifest = coupled_population_manifest(
            "color_gradient_lattice_boltzmann",
            discretization.velocity_set.lattice_id,
            discretization.precision.policy_id,
            discretization.velocity_set.population_count,
            discretization.velocity_set.dimension,
            tuple(f"{component}_populations" for component in method.component_ids),
            tuple(
                (f"{component}_mass", "momentum") for component in method.component_ids
            ),
        )
        self.discretization = discretization
        self.scaling = scaling
        self.method = method
        self.hydrodynamic_method = hydrodynamic_method
        self.near_contact = near_contact
        self.program_manifest = program_manifest
        self.boundary = boundary
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-color-gradient-lattice-boltzmann-dynamics",
                "discretization": discretization.prepared_id,
                "scaling": scaling.scaling_id,
                "method": method.method_id,
                "prepared_hydrodynamic_method": hydrodynamic_method.method_id,
                "near_contact": None
                if near_contact is None
                else near_contact.prepared_id,
                "program_manifest": program_manifest.manifest_id,
                "boundary": boundary.boundary_id,
            }
        )

    def _parameters(self, args: Any, /) -> ColorGradientLBMRuntimeParameters:
        if not isinstance(args, ColorGradientLBMRuntimeParameters):
            raise TypeError(
                "Color-gradient fixed-step args must be ColorGradientLBMRuntimeParameters."
            )
        if args.surface_tension.label_ids != self.method.component_ids:
            raise ValueError(
                "surface_tension label ids must equal the method component ids in order."
            )
        expected = 0 if self.near_contact is None else self.near_contact.plan.pair_count
        if args.near_contact_strength.shape != (expected,):
            raise ValueError(
                "near_contact_strength must hold one value per declared repelling pair."
            )
        return args

    @checked
    def _validate_state(self, state: ColorGradientLBMState, /) -> ColorGradientLBMState:
        populations = jnp.asarray(state.color_populations)
        if populations.shape != (
            self.method.component_count,
            *self.discretization.population_shape,
        ):
            raise ValueError("color_populations must have shape [N, *grid, Q].")
        validated = jax.vmap(self.discretization.validate_populations)(populations)
        work = jnp.asarray(state.near_contact_work, dtype=validated.dtype)
        if work.shape != ():
            raise ValueError("near_contact_work must be one scalar.")
        return ColorGradientLBMState(validated, work)

    def _wetting_data(
        self, parameters: ColorGradientLBMRuntimeParameters, dtype: DTypeLike, /
    ) -> _WettingData:
        shape = self.discretization.grid.shape
        dimension = self.discretization.velocity_set.dimension
        angles = jnp.asarray(parameters.contact_angles, dtype=dtype)
        admissible = jnp.isfinite(angles) & (angles > 0.0) & (angles < jnp.pi)
        safe_angles = jnp.where(admissible, angles, 0.5 * jnp.pi)
        angle_valid = jnp.all(admissible)
        if parameters.wetting_mask.size == 0:
            return _WettingData(None, None, safe_angles, angle_valid)
        if parameters.wetting_mask.shape != shape:
            raise ValueError("wetting_mask must match the lattice grid shape.")
        if parameters.wall_normal.shape != (*shape, dimension):
            raise ValueError("wall_normal must contain one vector per lattice cell.")
        mask = parameters.wetting_mask
        wall = jnp.asarray(parameters.wall_normal, dtype=dtype)
        norm = jnp.sqrt(ein.contract("...d,...d->...", wall, wall))
        normal_valid = jnp.all(jnp.isfinite(wall), axis=-1) & (norm > 0.0)
        fallback = jnp.zeros_like(wall).at[..., 0].set(1.0)
        safe_wall = jnp.where((~mask | normal_valid)[..., None], wall, fallback)
        valid = angle_valid & jnp.all(~mask | normal_valid)
        return _WettingData(safe_wall, mask, safe_angles, valid)

    def _pair_tensions(
        self, parameters: ColorGradientLBMRuntimeParameters, dtype: DTypeLike, /
    ) -> tuple[Array, Array, Array]:
        """Physical pair tensions, their lattice values, and representation validity."""

        first, second = _pair_members(self.method.component_count)
        physical = parameters.surface_tension.pair_values(first, second).astype(dtype)
        admissible = jnp.isfinite(physical) & (physical >= 0.0)
        safe = jnp.where(admissible, physical, 0.0)
        dt = self.scaling.time_step.astype(dtype)
        dx = self.scaling.cell_size.astype(dtype)
        rho0 = self.scaling.reference_density.astype(dtype)
        return safe, safe * dt**2 / (rho0 * dx**3), jnp.all(admissible)

    def _interfacial(
        self,
        densities: Array,
        concentrations: Array,
        lattice_tensions: Array,
        wetting: _WettingData,
        /,
    ) -> ColorGradientInterfacialFields:
        velocity_set = self.discretization.velocity_set
        first, second = _pair_members(self.method.component_count)
        floor = jnp.asarray(self.method.gradient_floor, dtype=densities.dtype)
        pair_density = jnp.maximum(
            densities[first] + densities[second], self.method.density_floor
        )
        phase = (densities[first] - densities[second]) / pair_density
        _, magnitude, normals = jax.vmap(
            lambda field: normalized_gradient(field, velocity_set, 1.0, epsilon=floor)
        )(phase)
        if wetting.wall_normal is not None and wetting.wetting_mask is not None:
            wall, mask = wetting.wall_normal, wetting.wetting_mask
            normals = jax.vmap(
                lambda normal, angle: static_contact_angle_normal(
                    normal, wall, angle, mask, epsilon=floor
                )
            )(normals, wetting.contact_angles)
        curvatures = -jax.vmap(lambda normal: isotropic_divergence(normal, velocity_set))(
            normals
        )
        presence = jnp.minimum(
            concentrations[first]
            * concentrations[second]
            / self.method.pair_presence_threshold,
            1.0,
        )
        tension = lattice_tensions.reshape((first.size,) + (1,) * (densities.ndim - 1))
        coefficient = 0.5 * tension * presence * magnitude
        identity = jnp.eye(velocity_set.dimension, dtype=densities.dtype)
        projector = identity - ein.contract("p...i,p...j->p...ij", normals, normals)
        stress = ein.contract("p...,p...ij->...ij", coefficient, projector)
        force = isotropic_tensor_divergence(stress, velocity_set)
        return ColorGradientInterfacialFields(
            phase, normals, curvatures, 0.5 * magnitude, presence, force
        )

    def _lattice_strengths(
        self, parameters: ColorGradientLBMRuntimeParameters, dtype: DTypeLike, /
    ) -> tuple[Array, Array]:
        """Lattice near-contact strengths and their physical admissibility."""

        strengths = jnp.asarray(parameters.near_contact_strength, dtype=dtype)
        admissible = jnp.isfinite(strengths) & (strengths >= 0.0)
        dt = self.scaling.time_step.astype(dtype)
        dx = self.scaling.cell_size.astype(dtype)
        rho0 = self.scaling.reference_density.astype(dtype)
        lattice = jnp.where(admissible, strengths, 0.0) * dt**2 / (rho0 * dx)
        return lattice, jnp.all(admissible)

    def _near_contact_force(
        self, concentrations: Array, lattice_strengths: Array, /
    ) -> NearContactForce:
        dtype = concentrations.dtype
        if self.near_contact is None:
            zero_force = jnp.zeros(
                (
                    *self.discretization.grid.shape,
                    self.discretization.velocity_set.dimension,
                ),
                dtype=dtype,
            )
            return NearContactForce(
                zero_force, jnp.zeros((), dtype=jnp.int32), jnp.zeros((), dtype=dtype)
            )
        return self.near_contact.force(
            concentrations,
            self.boundary.geometry.fluid_mask,
            lattice_strengths,
            jnp.asarray(self.method.gradient_floor, dtype=dtype),
        )

    def _fields(
        self,
        state: ColorGradientLBMState,
        parameters: ColorGradientLBMRuntimeParameters,
        /,
    ) -> tuple[_ColorGradientFields, Array]:
        densities, momenta = macroscopic_raw_moments(
            state.color_populations,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        density = jnp.sum(densities, axis=0)
        safe_density = jnp.maximum(
            density,
            jnp.asarray(self.method.density_floor, dtype=density.dtype),
        )
        concentrations = densities / safe_density
        wetting = self._wetting_data(parameters, state.color_populations.dtype)
        _, lattice_tensions, tension_valid = self._pair_tensions(
            parameters, density.dtype
        )
        interfacial = self._interfacial(
            densities, concentrations, lattice_tensions, wetting
        )
        lattice_strengths, strength_valid = self._lattice_strengths(
            parameters, density.dtype
        )
        near_contact = self._near_contact_force(concentrations, lattice_strengths)
        raw_momentum = jnp.sum(momenta, axis=0)
        force = interfacial.force_density + near_contact.force_density
        velocity = (raw_momentum + 0.5 * force) / safe_density[..., None]
        fields = _ColorGradientFields(
            densities,
            density,
            concentrations,
            raw_momentum,
            velocity,
            interfacial,
            near_contact,
        )
        return fields, tension_valid & wetting.valid & strength_valid

    def _initial_components(
        self, component_densities: ArrayLike, dtype: DTypeLike, /
    ) -> Array:
        shape = self.discretization.grid.shape
        count = self.method.component_count
        physical = jnp.asarray(component_densities, dtype=dtype)
        if physical.shape == (count,):
            physical = jnp.broadcast_to(
                physical.reshape((count,) + (1,) * len(shape)), (count, *shape)
            )
        if physical.shape != (count, *shape):
            raise ValueError(
                "Initial component densities must be [N] or [N, *grid] in component order."
            )
        return self.scaling.lattice_density(physical)

    def _initial_velocity(self, velocity: ArrayLike, dtype: DTypeLike, /) -> Array:
        shape = self.discretization.grid.shape
        dimension = self.discretization.velocity_set.dimension
        physical_velocity = jnp.asarray(velocity, dtype=dtype)
        if physical_velocity.shape == (dimension,):
            physical_velocity = jnp.broadcast_to(physical_velocity, (*shape, dimension))
        if physical_velocity.shape != (*shape, dimension):
            raise ValueError(
                "Initial velocity must be one vector or one vector per cell."
            )
        return self.scaling.lattice_velocity(physical_velocity)

    def initialize_state(
        self,
        component_densities: ArrayLike,
        velocity: ArrayLike,
        parameters: ColorGradientLBMRuntimeParameters,
        /,
    ) -> ColorGradientLBMState:
        """Equilibrium component populations for physical densities ``[N, *grid]``."""

        parameters_ = self._parameters(parameters)
        dtype = jnp.dtype(self.discretization.precision.population_dtype)
        lattice = self._initial_components(component_densities, dtype)
        lattice_velocity = self._initial_velocity(velocity, dtype)
        density = jnp.sum(lattice, axis=0)
        safe_density = jnp.maximum(density, self.method.density_floor)
        concentrations = lattice / safe_density
        wetting = self._wetting_data(parameters_, dtype)
        _, lattice_tensions, tension_valid = self._pair_tensions(parameters_, dtype)
        interfacial = self._interfacial(
            lattice, concentrations, lattice_tensions, wetting
        )
        lattice_strengths, strength_valid = self._lattice_strengths(parameters_, dtype)
        near_contact = self._near_contact_force(concentrations, lattice_strengths)
        force = interfacial.force_density + near_contact.force_density
        raw_velocity = lattice_velocity - 0.5 * force / safe_density[..., None]
        total = quadratic_equilibrium(
            density,
            raw_velocity,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        initial = recolor_populations(
            total,
            lattice,
            interfacial.normals,
            self.discretization.velocity_set,
            jnp.asarray(self.method.recoloring_strength, dtype=dtype),
            density_floor=self.method.density_floor,
        )
        fluid = self.boundary.geometry.fluid_mask
        solid_total = quadratic_equilibrium(
            jnp.ones_like(density),
            jnp.zeros_like(lattice_velocity),
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        populations = self.discretization.precision.population(
            jnp.where(
                fluid[..., None], initial, solid_total / self.method.component_count
            )
        )
        valid = (
            tension_valid
            & wetting.valid
            & strength_valid
            & jnp.all(jnp.isfinite(populations))
            & jnp.all((~fluid) | (lattice >= 0.0))
            & jnp.all((~fluid) | (density > self.method.density_floor))
        )
        checked = eqx.error_if(
            populations,
            ~valid,
            "Initial color-gradient state is not finite and admissible.",
        )
        return ColorGradientLBMState(checked, jnp.zeros((), dtype=dtype))

    def macroscopic_state(
        self,
        state: ColorGradientLBMState,
        parameters: ColorGradientLBMRuntimeParameters,
        /,
    ) -> ColorGradientMacroscopicState:
        values = self._validate_state(state)
        fields, _ = self._fields(values, self._parameters(parameters))
        return ColorGradientMacroscopicState(
            self.scaling.physical_density(fields.component_densities),
            self.scaling.physical_density(fields.density),
            fields.concentrations,
            self.scaling.physical_velocity(fields.velocity),
            self.scaling.physical_pressure(fields.density),
            fields.interfacial,
            fields.near_contact.force_density,
        )

    def _recoloring_conservation(
        self, recolored: Array, total: Array, targets: Array, /
    ) -> RecoloringConservation:
        component_moment = jnp.sum(recolored, axis=-1)
        closure = jnp.sum(recolored, axis=0) - total
        velocities = jnp.asarray(
            self.discretization.velocity_set.velocities, dtype=total.dtype
        )
        momentum_closure = ein.contract("...q,qd->...d", closure, velocities)
        return RecoloringConservation(
            jnp.max(jnp.abs(component_moment - targets)),
            jnp.max(jnp.abs(closure)),
            jnp.max(jnp.abs(momentum_closure)),
        )

    def _diagnostics(
        self,
        fields: _ColorGradientFields,
        component_defects: Array,
        total_defect: Array,
        conservation: RecoloringConservation,
        parameters: ColorGradientLBMRuntimeParameters,
        /,
    ) -> ColorGradientDiagnostics:
        fluid = self.boundary.geometry.fluid_mask
        grid_axes = tuple(range(self.discretization.velocity_set.dimension))
        speed = jnp.sqrt(ein.contract("...d,...d->...", fields.velocity, fields.velocity))
        cs = jnp.sqrt(
            jnp.asarray(
                self.discretization.velocity_set.sound_speed_squared,
                dtype=speed.dtype,
            )
        )
        physical_tensions, _, _ = self._pair_tensions(parameters, speed.dtype)
        weakest = jnp.min(jnp.where(physical_tensions > 0.0, physical_tensions, jnp.inf))
        capillary = jnp.where(
            jnp.isfinite(weakest),
            self.scaling.physical_density(fields.density)
            * jnp.asarray(parameters.kinematic_viscosity, dtype=speed.dtype)
            * self.scaling.physical_velocity(speed)
            / jnp.where(jnp.isfinite(weakest), weakest, 1.0),
            0.0,
        )
        capillary_force = jnp.where(
            fluid[..., None], fields.interfacial.force_density, 0.0
        )
        capillary_net = jnp.sqrt(jnp.sum(jnp.sum(capillary_force, axis=grid_axes) ** 2))
        capillary_total = jnp.sum(
            jnp.sqrt(ein.contract("...d,...d->...", capillary_force, capillary_force))
        )
        return ColorGradientDiagnostics(
            component_masses=jnp.sum(
                jnp.where(fluid, fields.component_densities, 0.0),
                axis=tuple(axis + 1 for axis in grid_axes),
            ),
            total_mass=jnp.sum(jnp.where(fluid, fields.density, 0.0)),
            component_mass_defects=component_defects,
            total_mass_defect=total_defect,
            total_momentum=jnp.sum(
                jnp.where(
                    fluid[..., None], fields.density[..., None] * fields.velocity, 0.0
                ),
                axis=grid_axes,
            ),
            minimum_component_density=jnp.min(
                jnp.where(fluid, jnp.min(fields.component_densities, axis=0), jnp.inf)
            ),
            minimum_density=jnp.min(jnp.where(fluid, fields.density, jnp.inf)),
            maximum_mach=jnp.max(jnp.where(fluid, speed / cs, 0.0)),
            maximum_capillary_number=jnp.max(jnp.where(fluid, capillary, 0.0)),
            force_norm=jnp.sqrt(jnp.sum(capillary_force**2)),
            capillary_net_force_residual=jnp.where(
                capillary_total > 0.0,
                capillary_net / jnp.where(capillary_total > 0.0, capillary_total, 1.0),
                0.0,
            ),
            recoloring=conservation,
        )

    def _near_contact_evidence(
        self, fields: _ColorGradientFields, strength_valid: Array, /
    ) -> NearContactRepulsionEvidence:
        dtype = fields.density.dtype
        if self.near_contact is None:
            zero = jnp.zeros((), dtype=dtype)
            return NearContactRepulsionEvidence(
                net_force=jnp.zeros(
                    (self.discretization.velocity_set.dimension,), dtype=dtype
                ),
                net_force_residual=zero,
                power=zero,
                work=zero,
                active_pair_count=jnp.zeros((), dtype=jnp.int32),
                maximum_pair_force=zero,
                strength_valid=strength_valid,
            )
        return self.near_contact.evidence(
            fields.near_contact,
            fields.velocity,
            self.boundary.geometry.fluid_mask,
            self.scaling,
            strength_valid,
        )

    def scalar_diagnostics(
        self,
        step_index: Array,
        time: Array,
        state: ColorGradientLBMState,
        parameters: ColorGradientLBMRuntimeParameters,
        /,
    ) -> ColorGradientDiagnostics:
        del step_index, time
        values = self._validate_state(state)
        parameters_ = self._parameters(parameters)
        fields, _ = self._fields(values, parameters_)
        zero = jnp.zeros((), dtype=values.color_populations.dtype)
        conservation = RecoloringConservation(zero, zero, zero)
        return self._diagnostics(
            fields,
            jnp.zeros((self.method.component_count,), dtype=zero.dtype),
            zero,
            conservation,
            parameters_,
        )

    def _advance(
        self,
        state: ColorGradientLBMState,
        fields: _ColorGradientFields,
        parameters: ColorGradientLBMRuntimeParameters,
        /,
    ) -> tuple[Array, RecoloringConservation, Array]:
        """Collide the mixture, recolor it pairwise and route every component."""

        dtype = state.color_populations.dtype
        fluid = self.boundary.geometry.fluid_mask
        viscosity = jnp.asarray(parameters.kinematic_viscosity, dtype=dtype)
        viscosity_valid = jnp.isfinite(viscosity) & (viscosity > 0.0)
        even_rate = self.scaling.relaxation_rate(
            jnp.where(viscosity_valid, viscosity, 1.0)
        )
        total = jnp.sum(state.color_populations, axis=0)
        collision_result = self.hydrodynamic_method.collide(
            self.discretization.precision.compute(total),
            fields.density,
            fields.velocity,
            fields.interfacial.force_density + fields.near_contact.force_density,
            even_rate,
            self.discretization.velocity_set,
            self.discretization.precision,
        )
        post_collision = jnp.where(
            fluid[..., None], collision_result.candidate_populations, total
        )
        targets = fields.concentrations * jnp.sum(post_collision, axis=-1)
        recolored = recolor_populations(
            post_collision,
            targets,
            fields.interfacial.normals,
            self.discretization.velocity_set,
            jnp.asarray(self.method.recoloring_strength, dtype=dtype),
            density_floor=self.method.density_floor,
        )
        conservation = self._recoloring_conservation(recolored, post_collision, targets)
        wall_velocity = jnp.asarray(parameters.moving_wall_velocities, dtype=dtype)
        wall_valid = jnp.all(jnp.isfinite(wall_velocity))
        lattice_wall = self.scaling.lattice_velocity(
            jnp.where(jnp.isfinite(wall_velocity), wall_velocity, 0.0)
        )
        routed = jax.vmap(self.boundary.route, in_axes=(0, 0, None))(
            self.discretization.precision.population(recolored),
            targets,
            lattice_wall,
        )
        tolerance = jnp.asarray(self.method.conservation_tolerance, dtype=dtype)
        valid = (
            collision_result.successful
            & viscosity_valid
            & wall_valid
            & (conservation.component_mass_defect <= tolerance)
            & (conservation.population_closure_defect <= tolerance)
            & (conservation.momentum_closure_defect <= tolerance)
        )
        return self.discretization.precision.population(routed), conservation, valid

    def _mass_defects(
        self, fields: _ColorGradientFields, candidate: _ColorGradientFields, /
    ) -> tuple[Array, Array]:
        fluid = self.boundary.geometry.fluid_mask
        axes = tuple(range(1, self.discretization.velocity_set.dimension + 1))
        previous = jnp.sum(jnp.where(fluid, fields.component_densities, 0.0), axis=axes)
        current = jnp.sum(jnp.where(fluid, candidate.component_densities, 0.0), axis=axes)
        component = jnp.abs(current - previous) / jnp.maximum(jnp.abs(previous), 1.0)
        previous_total = jnp.sum(previous)
        total = jnp.abs(jnp.sum(current) - previous_total) / jnp.maximum(
            jnp.abs(previous_total), 1.0
        )
        return component, total

    def step_detailed(
        self,
        step_index: Array,
        time: Array,
        state: ColorGradientLBMState,
        step_size: Array,
        args: Any,
        /,
    ) -> ColorGradientStepResult:
        del step_index, time
        values = self._validate_state(state)
        parameters = self._parameters(args)
        dtype = values.color_populations.dtype
        dt = jnp.asarray(step_size, dtype=dtype)
        expected_dt = jnp.asarray(self.scaling.time_step, dtype=dtype)
        fields, fields_valid = self._fields(values, parameters)
        routed, conservation, advance_valid = self._advance(values, fields, parameters)
        _, strength_valid = self._lattice_strengths(parameters, dtype)
        evidence = self._near_contact_evidence(fields, strength_valid)
        candidate = ColorGradientLBMState(
            routed, values.near_contact_work + evidence.work
        )
        candidate_fields, candidate_valid = self._fields(candidate, parameters)
        component_defects, total_defect = self._mass_defects(fields, candidate_fields)
        provisional = self._diagnostics(
            candidate_fields, component_defects, total_defect, conservation, parameters
        )
        fluid = self.boundary.geometry.fluid_mask
        successful = (
            advance_valid
            & jnp.isclose(dt, expected_dt, rtol=1.0e-12, atol=1.0e-12)
            & fields_valid
            & candidate_valid
            & jnp.all(jnp.isfinite(routed))
            & jnp.isfinite(candidate.near_contact_work)
            & jnp.all((~fluid) | (candidate_fields.component_densities >= 0.0))
            & jnp.all((~fluid) | (candidate_fields.density > self.method.density_floor))
            & (provisional.maximum_mach <= self.method.maximum_mach)
            & (
                provisional.maximum_capillary_number
                <= self.method.maximum_capillary_number
            )
        )
        accepted = tree_where(successful, candidate, values)
        zero = jnp.zeros((), dtype=dtype)
        current = self._diagnostics(
            fields,
            jnp.zeros_like(component_defects),
            zero,
            RecoloringConservation(zero, zero, zero),
            parameters,
        )
        return ColorGradientStepResult(
            candidate,
            accepted,
            successful,
            jnp.maximum(total_defect, jnp.max(component_defects)),
            jnp.asarray(
                self.method.component_count
                * self.boundary.geometry.fluid_count
                * self.discretization.velocity_set.population_count,
                dtype=jnp.int32,
            ),
            tree_where(successful, provisional, current),
            evidence,
        )


def color_gradient_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Unreleased N-color profile; `tools/color_gradient_lbm_qualification.py` gates it."""

    return (
        CapabilityProfile(
            "lattice-boltzmann.n-color-emulsion.profile",
            "phydrax",
            "candidate",
            (
                SupportTuple(
                    "lattice-boltzmann.n-color-emulsion",
                    {
                        "density": "matched",
                        "capillary-force": "continuum-surface-stress",
                        "recoloring": "pairwise-latva-kokko",
                        "near-contact": "pairwise-antisymmetric",
                    },
                ),
            ),
            required_gates=(
                "analytic-control",
                "reference-qualification",
                "public-workflow",
            ),
        ),
    )


__all__ = [
    "ColorGradientDiagnostics",
    "ColorGradientInterfacialFields",
    "ColorGradientLBMMethod",
    "ColorGradientLBMRuntimeParameters",
    "ColorGradientLBMState",
    "ColorGradientMacroscopicState",
    "ColorGradientStepResult",
    "PreparedColorGradientLBMDynamics",
    "RecoloringConservation",
    "color_gradient_candidate_profiles",
    "recolor_populations",
]
