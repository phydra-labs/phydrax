#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Lagrangian GMLS particle flow with an admitted weak pressure projection.

Material points carry constant mass ``m`` and move with the fluid, ``x' = u``.
The measure is named, never inferred: ``"quadrature-volume"`` points carry an
authoritative GMLS quadrature volume ``V`` evolved by the discrete geometric
conservation law ``V^{n+1} = V^n (1 + dt D_h u^{n+1})`` with ``m = rho V``;
``"material-mass"`` points carry the persistent masses of a
``MaterialParticleMeasure`` and the volume ``V = m / rho_SPH`` of the native
SPH summation density on the particle Verlet cache. Masses never change, so
total mass is conserved exactly by every material step.

One step composes owners in this order: the GMLS cloud is refreshed at the
accepted positions on its anchored support (``PreparedPointCloudDiscretization
.refresh``) and re-prepared with the same stable ids only when the refresh
refuses; the explicit predictor ``u* = u + dt (g + nu lap_h u)`` uses the GMLS
Laplacian; the pressure is the native minimum-norm weighted least-squares
solution (LSMR) of ``min_p ||G p - (rho/dt) u*||_M``, whose normal equations
are ``G^T M G p = (rho/dt) G^T M u*`` (boundary-free: a fully periodic
address, or the natural weak free-slip boundary of a bounded cloud); the
correction ``u = u* - (dt/rho) G p`` and the material update ``x += dt u``
follow. ``G`` is the GMLS gradient and ``M`` the cloud quadrature, so the
projected velocity is the scaled least-squares residual, ``D_h u = 0`` for the
weak divergence ``D_h u = -M^{-1} G^T M u`` holds to solver tolerance, and the
projection is ``M``-orthogonal: kinetic energy with masses ``rho M`` cannot
increase.

The kernel of ``G`` is not only the constants: on symmetric lattices the
antisymmetric GMLS weights also annihilate odd-even modes, and nearby clouds
have near-null modes. A gauged square solve of ``G^T M G`` is then singular
or ill-conditioned beyond its gauge. The least-squares form needs no gauge and
is consistent by construction: Krylov iterates never leave the row space of
``G``, so the minimum-norm pressure carries no kernel content and ``G p`` is
exact to solver tolerance whatever the kernel.
The strong GMLS divergence is published separately; it is a consistency
measure, not a projection invariant. Pressure correction changes momentum by
``-(dt/rho) sum m G p``; that impulse is reported, never claimed conserved.

The collocated Neumann route is deliberately not offered: its wall rows
replace the pressure equation, so wall-row divergence is uncontrolled and
grows under material motion.

Acceptance is transactional (``phydrax.lifecycle.commit_candidate``): a
refused step (invalid step, nonfinite candidate, Verlet refusal, pressure
solve status failure or Courant violation) keeps the accepted state and
publishes the candidate, status and evidence; ``fixed_step`` exposes the
native ``FixedStepResult`` record. Cloud refresh/re-preparation is host
preparation per step, so a step is driven eagerly by a host loop and is not a
traced ``FixedStepRolloutPlan`` method.

SPH interoperability exchanges reconstructions only:
``MeshfreeSPHReconstruction`` evaluates the native summation density and the
continuity divergence on the particle Verlet relation and compares them with
the GMLS divergence and measure density of the same particles. IISPH and DFSPH
remain separate SPH owners; their public ``initialize_state(position,
velocity, ...)`` already accepts these particle fields directly, so no second
pressure loop or adapter exists here.
"""

from __future__ import annotations

from enum import IntEnum
from typing import assert_never, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..discretization import (
    PeriodicCell,
    PointCloudPlan,
    PreparedPointCloudDiscretization,
)
from ..discretization.meshfree import (
    EvolutionMeasure,
    LocalStencilPolicy,
    MaterialParticleMeasure,
    MeshfreeFunctional,
    MeshfreeNeighborhoodPlan,
    PointTransferPlan,
    PointTransferRequest,
    PointTransferStatus,
    prepare_local_stencils,
    PreparedPointTransfer,
)
from ..discretization.particle import (
    AbstractSPHSmoothingKernel,
    particle_pair_geometry,
    ParticleBox,
    ParticleDiscretization,
    ParticleExecutionPolicy,
    ParticlePairGeometry,
    ParticlePairRelation,
    ParticlePrecisionPolicy,
    ParticleVerletState,
    PreparedVerletParticleNeighborhood,
    sph_continuity_density_rate,
    sph_summation_density,
)
from ..discretization.spatial import MortonAddressPlan
from ..lifecycle import commit_candidate, TransactionalCandidate
from ..linalg import (
    ArraySpace,
    bind_numeric,
    FailurePolicy,
    FunctionLinearOperator,
    LeastSquaresProblem,
    LinearSolvePolicy,
    LinearSolveTemplate,
    LSMR,
    prepare_template,
    solve,
    TolerancePolicy,
)
from ..typing import parse
from ._fixed_step import FixedStepResult


class MeshfreeLagrangianStatus(IntEnum):
    """Fail-closed Lagrangian step acceptance in reporting precedence."""

    ACCEPTED = 0
    INVALID_STEP = 1
    NONFINITE = 2
    NEIGHBORS_REFUSED = 3
    PRESSURE_REFUSED = 4
    COURANT_REFUSED = 5


def _positive(value: float, name: str, /) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _host_points(points: ArrayLike, name: str, /) -> np.ndarray:
    value = np.asarray(jax.device_get(points), dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 1 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite (points, dimension) coordinates.")
    return value


def _host_measure(values: ArrayLike, count: int, name: str, /) -> np.ndarray:
    value = np.asarray(jax.device_get(values), dtype=np.float64)
    if value.shape != (count,) or not np.all(np.isfinite(value) & (value > 0.0)):
        raise ValueError(f"{name} must be finite, positive and one per point.")
    return value


def _weighted_rms(weights: Array, values: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(weights * values**2) / jnp.sum(weights))


def _center_of_mass(masses: Array, points: Array, /) -> Array:
    return jnp.sum(masses[:, None] * points, axis=0) / jnp.sum(masses)


def _kinetic_energy(masses: Array, velocity: Array, /) -> Array:
    return 0.5 * jnp.sum(masses * jnp.sum(velocity**2, axis=1))


@final
class MeshfreeSPHComparison(StrictModule):
    """SPH and GMLS reconstructions of density and divergence on one particle set.

    ``sph_divergence = -(1/rho_SPH) Drho/Dt`` from the native continuity rate;
    ``measure_density = m / V``. ``density_mismatch`` is
    ``max |rho_SPH - m/V| / max(m/V)``; ``divergence_mismatch`` is the
    volume-weighted RMS of ``sph_divergence - gmls_divergence``.
    """

    sph_density: Array
    measure_density: Array
    sph_divergence: Array
    gmls_divergence: Array
    density_mismatch: Array
    divergence_mismatch: Array
    neighbors_successful: Array


@final
class MeshfreeMeasureConversion(StrictModule):
    """Same-point measure change with its mass and density evidence.

    The material mass is kept (``m = rho V`` of a quadrature representation is
    already the material mass) and ``V = m / rho_SPH`` is adopted. ``measure``
    names which quantity is authoritative afterwards: ``"quadrature-volume"``
    evolves ``V`` by the discrete GCL, ``"material-mass"`` recomputes it from
    the SPH density. ``mass_defect = sum(rho_SPH V) - sum(m)`` is round-off;
    ``density_mismatch = max |rho_SPH - m/V_in| / max(m/V_in)`` compares the
    SPH density with the incoming measure density.
    """

    masses: Array
    volumes: Array
    sph_density: Array
    mass_defect: Array
    density_mismatch: Array
    neighbors_successful: Array
    measure: EvolutionMeasure = eqx.field(static=True)


@final
class MeshfreeSPHReconstruction(StrictModule):
    """Native SPH summation density and continuity divergence on a Verlet cache.

    The Verlet interaction radius must cover the kernel support so the cached
    relation contains every physical pair; the kernel-support mask is applied
    explicitly as the SPH owners do.
    """

    particles: ParticleDiscretization
    neighbors: PreparedVerletParticleNeighborhood
    kernel: AbstractSPHSmoothingKernel
    execution: ParticleExecutionPolicy
    precision: ParticlePrecisionPolicy
    smoothing_length: float = eqx.field(static=True)

    def __init__(
        self,
        particles: ParticleDiscretization,
        neighbors: PreparedVerletParticleNeighborhood,
        kernel: AbstractSPHSmoothingKernel,
        smoothing_length: float,
        /,
        *,
        execution: ParticleExecutionPolicy | None = None,
        precision: ParticlePrecisionPolicy | None = None,
    ) -> None:
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if not isinstance(neighbors, PreparedVerletParticleNeighborhood):
            raise TypeError("neighbors must be a PreparedVerletParticleNeighborhood.")
        if not isinstance(kernel, AbstractSPHSmoothingKernel):
            raise TypeError("kernel must be an AbstractSPHSmoothingKernel.")
        length = _positive(smoothing_length, "smoothing_length")
        if neighbors.particle_discretization_id != particles.prepared_id:
            raise ValueError("Verlet neighbors were prepared for another particle set.")
        if kernel.dimension != particles.ambient_dimension:
            raise ValueError("SPH kernel dimension must match the particle dimension.")
        support = float(np.asarray(kernel.support_radius(length)))
        if neighbors.plan.interaction_radius < support:
            raise ValueError(
                "Verlet interaction radius must cover the SPH kernel support radius."
            )
        execution_ = ParticleExecutionPolicy() if execution is None else execution
        precision_ = ParticlePrecisionPolicy() if precision is None else precision
        if not isinstance(execution_, ParticleExecutionPolicy) or not isinstance(
            precision_, ParticlePrecisionPolicy
        ):
            raise TypeError("execution and precision must be native particle policies.")
        self.particles = particles
        self.neighbors = neighbors
        self.kernel = kernel
        self.execution = execution_
        self.precision = precision_
        self.smoothing_length = length

    def initialize(self, positions: ArrayLike, /) -> ParticleVerletState:
        return self.neighbors.initialize(positions)

    def update(
        self, positions: ArrayLike, previous: ParticleVerletState, /
    ) -> ParticleVerletState:
        return self.neighbors.update(positions, previous)

    def density(
        self, positions: Array, masses: Array, cache: ParticleVerletState, /
    ) -> Array:
        """Self-inclusive SPH summation density of the given masses."""
        pairs, geometry, physical = self._geometry(positions, cache)
        return sph_summation_density(
            masses,
            self.particles.active_mask,
            pairs,
            geometry,
            physical,
            self.kernel,
            self.smoothing_length,
            particle_count=self.particles.capacity,
            execution=self.execution,
            precision=self.precision,
        )

    def divergence(
        self,
        positions: Array,
        velocity: Array,
        masses: Array,
        cache: ParticleVerletState,
        /,
    ) -> Array:
        """Continuity divergence ``-(1/rho_SPH) Drho/Dt`` on active particles."""
        pairs, geometry, physical = self._geometry(positions, cache)
        rate = sph_continuity_density_rate(
            masses,
            velocity,
            pairs,
            geometry,
            physical,
            self.kernel,
            self.smoothing_length,
            particle_count=self.particles.capacity,
            execution=self.execution,
            precision=self.precision,
        )
        density = self.density(positions, masses, cache)
        active = self.particles.active_mask
        return jnp.where(active, -rate / jnp.where(active, density, 1.0), 0.0)

    def compare(
        self,
        cloud: PreparedPointCloudDiscretization,
        positions: ArrayLike,
        velocity: ArrayLike,
        masses: ArrayLike,
        volumes: ArrayLike,
        cache: ParticleVerletState,
        /,
    ) -> MeshfreeSPHComparison:
        """SPH versus GMLS density/divergence on the cloud's own particles."""
        if not isinstance(cloud, PreparedPointCloudDiscretization):
            raise TypeError("cloud must be a PreparedPointCloudDiscretization.")
        x, u, m, v = self._fields(positions, velocity, masses, volumes)
        if cloud.state_shape[0] != x.shape[0]:
            raise ValueError("The GMLS cloud must hold exactly these particles.")
        sph_density = self.density(x, m, cache)
        measure_density = m / v
        sph_divergence = self.divergence(x, u, m, cache)
        gmls_divergence = cloud.divergence(u)
        return MeshfreeSPHComparison(
            sph_density,
            measure_density,
            sph_divergence,
            gmls_divergence,
            jnp.max(jnp.abs(sph_density - measure_density)) / jnp.max(measure_density),
            _weighted_rms(v, sph_divergence - gmls_divergence),
            cache.successful,
        )

    def convert(
        self,
        positions: ArrayLike,
        masses: ArrayLike,
        volumes: ArrayLike,
        cache: ParticleVerletState,
        /,
        *,
        target: EvolutionMeasure,
    ) -> MeshfreeMeasureConversion:
        """Change the authoritative measure of one particle set at fixed points."""
        measure = parse(target, EvolutionMeasure, "target")
        zero = jnp.zeros(jnp.shape(positions), dtype=jnp.float64)
        x, _, m, v = self._fields(positions, zero, masses, volumes)
        rho = self.density(x, m, cache)
        converted = m / rho
        incoming = m / v
        return MeshfreeMeasureConversion(
            m,
            converted,
            rho,
            jnp.sum(rho * converted) - jnp.sum(m),
            jnp.max(jnp.abs(rho - incoming)) / jnp.max(incoming),
            cache.successful,
            measure,
        )

    def _fields(
        self,
        positions: ArrayLike,
        velocity: ArrayLike,
        masses: ArrayLike,
        volumes: ArrayLike,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        count = self.particles.capacity
        dimension = self.particles.ambient_dimension
        x = jnp.asarray(positions, dtype=jnp.float64)
        u = jnp.asarray(velocity, dtype=jnp.float64)
        m = jnp.asarray(masses, dtype=jnp.float64)
        v = jnp.asarray(volumes, dtype=jnp.float64)
        if (
            x.shape != (count, dimension)
            or u.shape != x.shape
            or m.shape != (count,)
            or v.shape != (count,)
        ):
            raise ValueError(
                "Particle fields must match the particle capacity and dimension."
            )
        return x, u, m, v

    def _geometry(
        self, positions: Array, cache: ParticleVerletState, /
    ) -> tuple[ParticlePairRelation, ParticlePairGeometry, Array]:
        if not isinstance(cache, ParticleVerletState):
            raise TypeError("cache must be a ParticleVerletState.")
        if cache.prepared_verlet_id != self.neighbors.prepared_id:
            raise ValueError("Verlet state belongs to another neighbor plan.")
        pairs = cache.neighborhood.pair_relation
        geometry = particle_pair_geometry(positions, pairs, box=self.neighbors.box)
        physical = geometry.valid & (
            geometry.distance < self.kernel.support_radius(self.smoothing_length)
        )
        return pairs, geometry, physical


@final
class MeshfreeLagrangianFlowState(StrictModule):
    """Material positions, velocity, masses, volumes and last projection pressure.

    ``neighbors`` is the SPH Verlet cache when the plan declares an SPH
    reconstruction, otherwise ``None``.
    """

    positions: Array
    velocity: Array
    masses: Array
    volumes: Array
    pressure: Array
    time: Array
    neighbors: ParticleVerletState | None


@final
class MeshfreeLagrangianEvidence(StrictModule):
    """Outcome-independent evidence of one attempted Lagrangian step.

    ``divergence_*`` are volume-weighted RMS norms of the weak divergence
    ``D_h u = -M^{-1} G^T M u`` that the projection annihilates;
    ``strong_divergence_*`` are the same norms of the strong GMLS divergence.
    ``before`` is the predicted and ``after`` the projected velocity, both on
    the step's cloud. Impulses are ``sum m du`` contributions of body force,
    explicit viscosity and pressure correction. ``density_deviation`` is
    ``max |m/(V rho_0) - 1|`` at the candidate. ``sph_density_mismatch``
    (``None`` without SPH) is ``max |rho_SPH - m/V| / rho_0`` at the candidate.
    ``pressure_*`` publish the native least-squares solve: status,
    iterations, the normal residual ``||G^T M (G p - (rho/dt) u*)||`` that
    the weak divergence measures, and LSMR's condition estimate of ``G`` on
    the explored Krylov space. Kernel and near-kernel modes of ``G`` (constants,
    and odd-even modes on symmetric lattices) leave ``G p`` unchanged and never
    enter the minimum-norm pressure.
    """

    status: Array
    successful: Array
    reprepared: Array
    refresh_status: Array
    refresh_displacement: Array
    courant: Array
    pressure_successful: Array
    pressure_status: Array
    pressure_iterations: Array
    pressure_residual: Array
    pressure_condition: Array
    divergence_before: Array
    divergence_after: Array
    strong_divergence_before: Array
    strong_divergence_after: Array
    mass_before: Array
    mass_after: Array
    momentum_before: Array
    momentum_after: Array
    body_impulse: Array
    viscous_impulse: Array
    pressure_impulse: Array
    center_of_mass_before: Array
    center_of_mass_after: Array
    kinetic_energy_before: Array
    kinetic_energy_predicted: Array
    kinetic_energy_after: Array
    volume_before: Array
    volume_after: Array
    density_deviation: Array
    neighbors_successful: Array
    neighbors_rebuilt: Array
    sph_density_mismatch: Array | None


@final
class MeshfreeLagrangianFlowPlan(StrictModule):
    """Declaration of one Lagrangian GMLS flow and its measure representation.

    A fully periodic ``address`` gives a boundary-free periodic projection; a
    bounded cloud (``address`` ``None`` or nonperiodic) uses the natural weak
    free-slip boundary of the dissipative pressure operator, where ``u.n = 0``
    holds only weakly. A domain-bound periodic address comes from
    ``MortonAddressPlan.from_periodic_identifications``. An SPH reconstruction
    shares that cell: its Verlet box must be ``ParticleBox.from_address(address)``
    (bounds and periodic mask exactly), or nonperiodic without an address.
    ``"material-mass"`` requires the
    ``MaterialParticleMeasure`` (masses, identities, Verlet cache) and an SPH
    reconstruction on the same cache. ``linear_policy`` is the native LSMR
    least-squares policy of the pressure projection (minimum-norm iterates
    from zero, no preconditioning, which would change the minimized norm); it
    must publish status so a failed pressure solve refuses the step instead of
    raising.
    """

    material: MaterialParticleMeasure | None
    sph: MeshfreeSPHReconstruction | None
    stencil: LocalStencilPolicy
    address: MortonAddressPlan | None
    body_acceleration: Array | None
    linear_policy: LinearSolvePolicy
    measure: EvolutionMeasure = eqx.field(static=True)
    reference_density: float = eqx.field(static=True)
    viscosity: float = eqx.field(static=True)
    courant_limit: float = eqx.field(static=True)
    neighbors: int | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        measure: EvolutionMeasure,
        /,
        *,
        reference_density: float,
        stencil: LocalStencilPolicy | None = None,
        neighbors: int | None = None,
        address: MortonAddressPlan | None = None,
        material: MaterialParticleMeasure | None = None,
        sph: MeshfreeSPHReconstruction | None = None,
        viscosity: float = 0.0,
        body_acceleration: ArrayLike | None = None,
        courant_limit: float = 0.5,
        linear_policy: LinearSolvePolicy | None = None,
    ) -> None:
        measure_ = parse(measure, EvolutionMeasure, "measure")
        density = _positive(reference_density, "reference_density")
        nu = float(viscosity)
        if not np.isfinite(nu) or nu < 0.0:
            raise ValueError("viscosity must be finite and nonnegative.")
        courant = _positive(courant_limit, "courant_limit")
        if sph is not None and not isinstance(sph, MeshfreeSPHReconstruction):
            raise TypeError("sph must be a MeshfreeSPHReconstruction.")
        match measure_:
            case "quadrature-volume":
                if material is not None:
                    raise ValueError(
                        "Quadrature-volume flow carries V; a material particle measure "
                        "belongs to the material-mass representation."
                    )
            case "material-mass":
                if not isinstance(material, MaterialParticleMeasure) or sph is None:
                    raise ValueError(
                        "Material-mass flow needs a MaterialParticleMeasure and an "
                        "SPH reconstruction for V = m / rho_SPH."
                    )
                if (
                    material.neighbors is None
                    or material.neighbors.prepared_id != sph.neighbors.prepared_id
                ):
                    raise ValueError(
                        "The material measure and SPH reconstruction must share one "
                        "Verlet neighbor cache."
                    )
            case unknown:
                assert_never(unknown)
        policy = (
            LinearSolvePolicy(
                LSMR(),
                tolerance=TolerancePolicy(relative=1e-10, absolute=0.0, max_steps=20_000),
                failure=FailurePolicy("status"),
            )
            if linear_policy is None
            else linear_policy
        )
        if not isinstance(policy, LinearSolvePolicy) or policy.failure.mode != "status":
            raise ValueError(
                "linear_policy must be a status-mode LinearSolvePolicy so pressure "
                "failure refuses the step."
            )
        if not isinstance(policy.method, LSMR) or policy.preconditioning is not None:
            raise ValueError(
                "The pressure projection is a minimum-norm least-squares solve: "
                "linear_policy must use unpreconditioned LSMR."
            )
        stencil_ = LocalStencilPolicy(polynomial_degree=3) if stencil is None else stencil
        if not isinstance(stencil_, LocalStencilPolicy):
            raise TypeError("stencil must be a LocalStencilPolicy.")
        if address is not None and not isinstance(address, MortonAddressPlan):
            raise TypeError("address must be a MortonAddressPlan.")
        if (
            address is not None
            and any(address.periodic_axes)
            and not all(address.periodic_axes)
        ):
            raise ValueError("A periodic projection needs a fully periodic address.")
        if sph is not None:
            _validate_sph_box(sph.neighbors.box, address)
        acceleration = (
            None
            if body_acceleration is None
            else np.asarray(jax.device_get(body_acceleration), dtype=np.float64)
        )
        if acceleration is not None and (
            acceleration.ndim != 1 or not np.all(np.isfinite(acceleration))
        ):
            raise ValueError("body_acceleration must be a finite vector.")
        self.material = material
        self.sph = sph
        self.stencil = stencil_
        self.address = address
        self.body_acceleration = (
            None if acceleration is None else jnp.asarray(acceleration)
        )
        self.linear_policy = policy
        self.measure = measure_
        self.reference_density = density
        self.viscosity = nu
        self.courant_limit = courant
        self.neighbors = neighbors
        self.plan_id = canonical_fingerprint(
            {
                "kind": "meshfree-lagrangian-flow-plan",
                "measure": measure_,
                "reference_density": density,
                "viscosity": nu,
                "courant_limit": courant,
                "neighbors": neighbors,
                "address": None if address is None else address.plan_id,
                "body": None
                if acceleration is None
                else array_tree_fingerprint(acceleration),
                "stencil": [
                    stencil_.approximation,
                    stencil_.polynomial_degree,
                    stencil_.phs_power,
                    stencil_.weight_kernel,
                ],
                "sph": None if sph is None else sph.neighbors.prepared_id,
            }
        )

    @property
    def periodic(self) -> bool:
        return self.address is not None and all(self.address.periodic_axes)

    def prepare(
        self,
        positions: ArrayLike,
        /,
        *,
        volumes: ArrayLike | None = None,
        point_ids: ArrayLike | None = None,
    ) -> PreparedMeshfreeLagrangianFlow:
        """Anchor the GMLS support epoch at ``positions`` (host preparation).

        Quadrature-volume flow takes its quadrature ``volumes``; material-mass
        flow derives ``V = m / rho_SPH`` and takes the material identities.
        """
        points = _host_points(positions, "positions")
        count, dimension = points.shape
        if self.body_acceleration is not None and self.body_acceleration.shape != (
            dimension,
        ):
            raise ValueError("body_acceleration must match the point dimension.")
        match self.measure:
            case "quadrature-volume":
                if volumes is None:
                    raise ValueError("Quadrature-volume flow needs quadrature volumes.")
                measure = _host_measure(volumes, count, "volumes")
                identities = point_ids
            case "material-mass":
                material, sph = self.material, self.sph
                if material is None or sph is None:
                    raise ValueError("Material-mass flow lost its particle measure.")
                if volumes is not None or point_ids is not None:
                    raise ValueError(
                        "Material-mass volumes and identities come from the particle measure."
                    )
                if material.masses.shape != (count,):
                    raise ValueError("Particle measure capacity must match the points.")
                x = jnp.asarray(points)
                density = sph.density(x, material.masses, sph.initialize(x))
                measure = _host_measure(material.masses / density, count, "SPH volumes")
                identities = material.identities
            case unknown:
                assert_never(unknown)
        cloud = _cloud(self, points, measure, identities)
        return PreparedMeshfreeLagrangianFlow(self, cloud)


def _validate_sph_box(
    box: ParticleBox | PeriodicCell | None, address: MortonAddressPlan | None, /
) -> None:
    """Refuse an SPH particle box that disagrees with the GMLS cloud address.

    A periodic address requires exactly ``ParticleBox.from_address(address)``;
    otherwise the SPH geometry must be nonperiodic as well. Affine
    ``PeriodicCell`` lattices are not meshfree Morton cells.
    """
    if isinstance(box, PeriodicCell):
        raise ValueError(
            "Meshfree SPH reconstruction uses a ParticleBox, not a PeriodicCell."
        )
    if address is not None and any(address.periodic_axes):
        if box is None or box.box_id != ParticleBox.from_address(address).box_id:
            raise ValueError(
                "The SPH particle box must be ParticleBox.from_address(address): the "
                "GMLS and SPH reconstructions share one periodic cell."
            )
    elif box is not None and any(box.periodic_axes):
        raise ValueError(
            "A periodic SPH particle box needs the matching periodic cloud address."
        )


def _cloud(
    plan: MeshfreeLagrangianFlowPlan,
    points: np.ndarray,
    volumes: np.ndarray,
    point_ids: ArrayLike | None,
    /,
) -> PreparedPointCloudDiscretization:
    return PointCloudPlan(
        points,
        volumes,
        stencil=plan.stencil,
        neighbors=plan.neighbors,
        point_ids=point_ids,
        address=plan.address,
    ).prepare()


def _weak_divergence(
    cloud: PreparedPointCloudDiscretization, weights: Array, velocity: Array, /
) -> Array:
    """``D_h u = -M^{-1} G^T M u``: the divergence dual to the GMLS gradient."""
    total = jnp.zeros(velocity.shape[:1], dtype=velocity.dtype)
    for axis in range(cloud.spatial_dimension):
        total = total + cloud.transpose_partial_derivative(
            weights * velocity[:, axis], axis=axis
        )
    return -total / weights


def _pressure_problem(
    cloud: PreparedPointCloudDiscretization, volumes: Array, operator_id: str, /
) -> LeastSquaresProblem:
    """``min_p ||M^{1/2} (G p - g)||``: the scaled GMLS gradient ``M^{1/2} G``.

    The current-volume root is part of the operator (native LSMR minimizes the
    Euclidean residual); it weights only the target, so the minimum-norm
    pressure gauge is unchanged. The right-hand side is ``M^{1/2} g``.
    """
    count, dimension = cloud.state_shape[0], cloud.spatial_dimension
    root = jnp.sqrt(volumes)[:, None]
    # Module-level actions with the cloud as dynamic data: every step's operator
    # has one static structure, so the compiled pressure solve is reused.
    operator = FunctionLinearOperator(
        eqx.Partial(_scaled_gradient, root, cloud),
        source=ArraySpace((count,), dtype=jnp.float64),
        target=ArraySpace((count, dimension), dtype=jnp.float64),
        transpose_action=eqx.Partial(_scaled_gradient_transpose, root, cloud),
        operator_id=operator_id,
        closure_convert=False,
    )
    return LeastSquaresProblem(operator, problem_id=operator_id)


def _scaled_gradient(
    root: Array, cloud: PreparedPointCloudDiscretization, values: Array, /
) -> Array:
    return root * cloud.gradient(values)


def _scaled_gradient_transpose(
    root: Array, cloud: PreparedPointCloudDiscretization, values: Array, /
) -> Array:
    scaled = root * values
    total = jnp.zeros((cloud.state_shape[0],), dtype=values.dtype)
    for axis in range(cloud.spatial_dimension):
        total = total + cloud.transpose_partial_derivative(scaled[:, axis], axis=axis)
    return total


@final
class MeshfreeLagrangianStepResult(StrictModule):
    """Committed state, raw candidate, evidence and the step's support epoch.

    ``flow`` is the prepared flow whose anchored cloud served this step (a new
    epoch when the refresh refused); it is valid for the committed state either
    way and should drive the next step.
    """

    flow: PreparedMeshfreeLagrangianFlow
    state: MeshfreeLagrangianFlowState
    candidate: MeshfreeLagrangianFlowState
    evidence: MeshfreeLagrangianEvidence

    @property
    def successful(self) -> Array:
        return self.evidence.successful

    @property
    def fixed_step(self) -> FixedStepResult:
        """Native fixed-step record of this attempt."""
        evidence = self.evidence
        return FixedStepResult(
            self.candidate,
            self.state,
            evidence.successful,
            evidence.pressure_residual,
            evidence.pressure_iterations,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((), dtype=jnp.float64),
            evidence=evidence,
        )


@final
class PreparedMeshfreeLagrangianFlow(StrictModule):
    """One anchored GMLS support epoch of a Lagrangian flow."""

    plan: MeshfreeLagrangianFlowPlan
    cloud: PreparedPointCloudDiscretization
    pressure_template: LinearSolveTemplate

    def __init__(
        self, plan: MeshfreeLagrangianFlowPlan, cloud: PreparedPointCloudDiscretization, /
    ) -> None:
        if not isinstance(plan, MeshfreeLagrangianFlowPlan):
            raise TypeError("plan must be a MeshfreeLagrangianFlowPlan.")
        if not isinstance(cloud, PreparedPointCloudDiscretization):
            raise TypeError("cloud must be a PreparedPointCloudDiscretization.")
        if bool(np.any(np.asarray(cloud.plan.boundary_mask))):
            raise ValueError("The weak projection uses a boundary-free cloud.")
        self.plan = plan
        self.cloud = cloud
        # The gradient's structure is fixed by the epoch; refreshed stencil
        # weights and current state volumes bind numerically to this template.
        self.pressure_template = prepare_template(
            _pressure_problem(cloud, cloud.quadrature_weights, self.pressure_id),
            plan.linear_policy,
        )

    @property
    def pressure_id(self) -> str:
        """Problem identity of the epoch's pressure projection (refresh keeps it)."""
        return f"{self.plan.plan_id}:{self.cloud.prepared_id}:pressure"

    @property
    def point_count(self) -> int:
        return self.cloud.state_shape[0]

    @property
    def dimension(self) -> int:
        return self.cloud.spatial_dimension

    def weak_divergence(self, velocity: ArrayLike, /) -> Array:
        """Divergence ``-M^{-1} G^T M u`` using the support anchor's quadrature."""
        u = jnp.asarray(velocity, dtype=jnp.float64)
        if u.shape != (self.point_count, self.dimension):
            raise ValueError("velocity must be (points, dimension).")
        return _weak_divergence(self.cloud, self.cloud.quadrature_weights, u)

    def initialize(
        self, velocity: ArrayLike, /, *, time: float = 0.0
    ) -> MeshfreeLagrangianFlowState:
        """Material state at the anchored points (masses ``m = rho_0 V`` or particle)."""
        u = jnp.asarray(velocity, dtype=jnp.float64)
        if u.shape != (self.point_count, self.dimension) or not bool(
            jnp.all(jnp.isfinite(u))
        ):
            raise ValueError("velocity must be finite (points, dimension).")
        if not np.isfinite(time):
            raise ValueError("time must be finite.")
        points = self.cloud.points
        volumes = self.cloud.quadrature_weights
        match self.plan.measure:
            case "quadrature-volume":
                masses = self.plan.reference_density * volumes
            case "material-mass":
                material = self.plan.material
                if material is None:
                    raise ValueError("Material-mass flow lost its particle measure.")
                masses = material.masses
            case unknown:
                assert_never(unknown)
        sph = self.plan.sph
        return MeshfreeLagrangianFlowState(
            points,
            u,
            masses,
            volumes,
            jnp.zeros((self.point_count,), dtype=jnp.float64),
            jnp.asarray(time, dtype=jnp.float64),
            None if sph is None else sph.initialize(points),
        )

    def rebase(
        self, state: MeshfreeLagrangianFlowState, /
    ) -> PreparedMeshfreeLagrangianFlow:
        """Re-anchor identical nodes at the state's coordinates (new support epoch).

        Stable ids, order, stencil policy and address are kept; the state's
        volumes become the quadrature measure.
        """
        cloud = _cloud(
            self.plan,
            _host_points(state.positions, "positions"),
            _host_measure(state.volumes, self.point_count, "volumes"),
            self.cloud.stable_ids,
        )
        return PreparedMeshfreeLagrangianFlow(self.plan, cloud)

    def step(
        self, state: MeshfreeLagrangianFlowState, step_size: ArrayLike, /
    ) -> MeshfreeLagrangianStepResult:
        """Attempt one projection step; refusal keeps ``state`` with evidence."""
        if not isinstance(state, MeshfreeLagrangianFlowState):
            raise TypeError("state must be a MeshfreeLagrangianFlowState.")
        if state.positions.shape != (self.point_count, self.dimension):
            raise ValueError("State does not match this flow's support epoch.")
        dt = jnp.asarray(step_size, dtype=jnp.float64)
        if dt.shape != ():
            raise ValueError("step_size must be scalar.")
        refresh = self.cloud.refresh(state.positions)
        # Support discovery is host preparation by the cloud owner: the refresh
        # status is the one host epoch decision of a step.
        reprepared = not bool(np.asarray(refresh.accepted))
        flow = (
            self.rebase(state)
            if reprepared
            else eqx.tree_at(lambda item: item.cloud, self, refresh.discretization)
        )
        cloud = flow.cloud
        valid = jnp.isfinite(dt) & (dt > 0.0)
        rho = self.plan.reference_density
        u = state.velocity
        body = (
            jnp.zeros((self.dimension,), dtype=jnp.float64)
            if self.plan.body_acceleration is None
            else self.plan.body_acceleration
        )
        viscous = self.plan.viscosity * cloud.laplacian(u)
        predicted = u + dt * (body[None, :] + viscous)
        volumes = state.volumes
        divergence_before = _weak_divergence(cloud, volumes, predicted)
        # min_p ||G p - (rho/dt) u*||_M: the normal equations are the weak
        # projection G^T M G p = (rho/dt) G^T M u*, consistent for every kernel.
        pressure = solve(
            bind_numeric(
                flow.pressure_template,
                _pressure_problem(cloud, volumes, flow.pressure_id),
            ),
            jnp.sqrt(volumes)[:, None] * (rho / jnp.where(valid, dt, 1.0)) * predicted,
        )
        correction = -(dt / rho) * cloud.gradient(pressure.value)
        velocity = predicted + correction
        divergence_after = _weak_divergence(cloud, volumes, velocity)
        positions = self._wrapped(state.positions + dt * velocity)
        candidate, cache_ok, rebuilt, mismatch = self._candidate(
            state, positions, velocity, pressure.value, divergence_after, dt
        )
        courant = dt * jnp.max(jnp.linalg.norm(velocity, axis=1)) / _spacing(cloud)
        finite = (
            jnp.all(jnp.isfinite(candidate.positions))
            & jnp.all(jnp.isfinite(candidate.velocity))
            & jnp.all(jnp.isfinite(candidate.volumes))
            & jnp.all(candidate.volumes > 0.0)
        )
        refusals = (
            (~valid, MeshfreeLagrangianStatus.INVALID_STEP),
            (~finite, MeshfreeLagrangianStatus.NONFINITE),
            (~cache_ok, MeshfreeLagrangianStatus.NEIGHBORS_REFUSED),
            (~pressure.successful, MeshfreeLagrangianStatus.PRESSURE_REFUSED),
            (
                ~(courant <= self.plan.courant_limit),
                MeshfreeLagrangianStatus.COURANT_REFUSED,
            ),
        )
        accepted = ~jnp.any(jnp.stack([refused for refused, _ in refusals]))
        status = jnp.select(
            [refused for refused, _ in refusals],
            [jnp.asarray(int(code), dtype=jnp.int32) for _, code in refusals],
            jnp.asarray(int(MeshfreeLagrangianStatus.ACCEPTED), dtype=jnp.int32),
        )
        m = state.masses
        evidence = MeshfreeLagrangianEvidence(
            status=status,
            successful=accepted,
            reprepared=jnp.asarray(reprepared),
            refresh_status=refresh.status.astype(jnp.int32),
            refresh_displacement=refresh.displacement,
            courant=courant,
            pressure_successful=pressure.successful,
            pressure_status=jnp.asarray(pressure.status, dtype=jnp.int32),
            pressure_iterations=jnp.asarray(
                pressure.diagnostics.iterations, dtype=jnp.int32
            ),
            pressure_residual=jnp.asarray(
                pressure.diagnostics.normal_residual_norm, dtype=jnp.float64
            ),
            pressure_condition=jnp.asarray(
                pressure.diagnostics.condition_estimate, dtype=jnp.float64
            ),
            divergence_before=_weighted_rms(volumes, divergence_before),
            divergence_after=_weighted_rms(volumes, divergence_after),
            strong_divergence_before=_weighted_rms(volumes, cloud.divergence(predicted)),
            strong_divergence_after=_weighted_rms(volumes, cloud.divergence(velocity)),
            mass_before=jnp.sum(m),
            mass_after=jnp.sum(candidate.masses),
            momentum_before=jnp.sum(m[:, None] * u, axis=0),
            momentum_after=jnp.sum(candidate.masses[:, None] * velocity, axis=0),
            body_impulse=dt * jnp.sum(m) * body,
            viscous_impulse=dt * jnp.sum(m[:, None] * viscous, axis=0),
            pressure_impulse=jnp.sum(m[:, None] * correction, axis=0),
            center_of_mass_before=_center_of_mass(m, state.positions),
            center_of_mass_after=_center_of_mass(candidate.masses, positions),
            kinetic_energy_before=_kinetic_energy(m, u),
            kinetic_energy_predicted=_kinetic_energy(m, predicted),
            kinetic_energy_after=_kinetic_energy(m, velocity),
            volume_before=jnp.sum(volumes),
            volume_after=jnp.sum(candidate.volumes),
            density_deviation=jnp.max(
                jnp.abs(candidate.masses / candidate.volumes / rho - 1.0)
            ),
            neighbors_successful=cache_ok,
            neighbors_rebuilt=rebuilt,
            sph_density_mismatch=mismatch,
        )
        committed = commit_candidate(
            TransactionalCandidate(
                state, candidate, evidence, accepted, self.plan.plan_id
            )
        )
        return MeshfreeLagrangianStepResult(flow, committed.state, candidate, evidence)

    def _wrapped(self, positions: Array, /) -> Array:
        address = self.plan.address
        if address is None or not self.plan.periodic:
            return positions
        lower = jnp.asarray(address.lower, dtype=positions.dtype)
        length = jnp.asarray(address.upper, dtype=positions.dtype) - lower
        return lower + jnp.mod(positions - lower, length)

    def _candidate(
        self,
        state: MeshfreeLagrangianFlowState,
        positions: Array,
        velocity: Array,
        pressure: Array,
        divergence: Array,
        dt: Array,
        /,
    ) -> tuple[MeshfreeLagrangianFlowState, Array, Array, Array | None]:
        sph = self.plan.sph
        cache = None
        cache_ok = jnp.asarray(True)
        rebuilt = jnp.asarray(False)
        rho_sph = None
        if sph is not None:
            if state.neighbors is None:
                raise ValueError("SPH flow state lost its Verlet cache.")
            cache = sph.update(positions, state.neighbors)
            cache_ok = cache.successful
            rebuilt = cache.rebuilt
            rho_sph = sph.density(positions, state.masses, cache)
        match self.plan.measure:
            case "quadrature-volume":
                volumes = state.volumes * (1.0 + dt * divergence)
            case "material-mass":
                if rho_sph is None:
                    raise ValueError("Material-mass flow needs its SPH density.")
                volumes = state.masses / rho_sph
            case unknown:
                assert_never(unknown)
        mismatch = (
            None
            if rho_sph is None
            else jnp.max(jnp.abs(rho_sph - state.masses / volumes))
            / self.plan.reference_density
        )
        candidate = MeshfreeLagrangianFlowState(
            positions,
            velocity,
            state.masses,
            volumes,
            pressure,
            state.time + dt,
            cache,
        )
        return candidate, cache_ok, rebuilt, mismatch


def _spacing(cloud: PreparedPointCloudDiscretization, /) -> Array:
    """Smallest positive neighbor distance of the cloud's current stencils."""
    distance = jnp.linalg.norm(cloud.stencils.offsets, axis=-1)
    usable = cloud.relation.valid & (distance > 0.0)
    return jnp.min(jnp.where(usable, distance, jnp.inf))


@final
class MeshfreeMeasureTransferResult(StrictModule):
    """Transferred material content and its conservation ledger.

    ``masses``/``velocity`` are target material mass and mass-averaged
    velocity (``NaN`` where a signed transfer leaves nonpositive mass, flagged
    by ``positive``); ``density = masses / target volumes``. Defects are
    target minus source totals; ``center_of_mass_defect`` compares mass
    centroids (first moments are not a declared transfer equation).
    ``density_mismatch`` is ``max |density / rho_s - 1|`` with the source mean
    density ``rho_s = sum m / sum V_s``.
    """

    masses: Array
    velocity: Array
    density: Array
    positive: Array
    mass_defect: Array
    momentum_defect: Array
    center_of_mass_defect: Array
    density_mismatch: Array
    measure: EvolutionMeasure = eqx.field(static=True)
    status: PointTransferStatus = eqx.field(static=True)


@final
class PreparedMeshfreeMeasureTransfer(StrictModule):
    """Audited transfer between two declared point measures.

    ``transfer`` exists with any status; ``apply`` refuses (raises through the
    point-transfer owner) unless the relation was admitted, e.g. when target
    routes do not cover every source (``UNCOVERED_SOURCE``).
    """

    transfer: PreparedPointTransfer
    source_points: Array
    target_points: Array
    source_measure: EvolutionMeasure = eqx.field(static=True)
    target_measure: EvolutionMeasure = eqx.field(static=True)

    @property
    def status(self) -> PointTransferStatus:
        return self.transfer.evidence.status

    @property
    def admitted(self) -> bool:
        return self.transfer.admitted

    def apply(
        self, masses: ArrayLike, velocity: ArrayLike, /
    ) -> MeshfreeMeasureTransferResult:
        """Transfer extensive mass and momentum; conservation is exact by audit."""
        m = jnp.asarray(masses, dtype=jnp.float64)
        u = jnp.asarray(velocity, dtype=jnp.float64)
        count, dimension = self.source_points.shape
        if m.shape != (count,) or u.shape != (count, dimension):
            raise ValueError("Transfer fields must match the source points.")
        target_mass = self.transfer.apply_content(m)
        momentum = jnp.stack(
            [self.transfer.apply_content(m * u[:, axis]) for axis in range(dimension)],
            axis=1,
        )
        positive = target_mass > 0.0
        safe = jnp.where(positive, target_mass, 1.0)
        target_velocity = jnp.where(positive[:, None], momentum / safe[:, None], jnp.nan)
        density = target_mass / self.transfer.target_measures
        source_density = jnp.sum(m) / jnp.sum(self.transfer.source_measures)
        return MeshfreeMeasureTransferResult(
            target_mass,
            target_velocity,
            density,
            jnp.all(positive),
            jnp.sum(target_mass) - jnp.sum(m),
            jnp.sum(momentum, axis=0) - jnp.sum(m[:, None] * u, axis=0),
            _center_of_mass(target_mass, self.target_points)
            - _center_of_mass(m, self.source_points),
            jnp.max(jnp.abs(density / source_density - 1.0)),
            self.target_measure,
            self.status,
        )


@final
class MeshfreeMeasureTransferPlan(StrictModule):
    """Declared transfer of material content between two point measures.

    Both sides declare their ``EvolutionMeasure`` and positive volumes (a
    material-mass side passes its ``V = m / rho``). Routes are cross-target
    local interpolation stencils; ``request`` (default
    ``conservative-positive``) is the ``PointTransfer`` constraint set, so mass
    and momentum are conserved exactly by the audited conservation equations
    and target masses stay nonnegative. ``joint`` requests moments and refuses
    unequal total measures with the owner's obstruction witness.
    """

    source_points: Array
    source_volumes: Array
    target_points: Array
    target_volumes: Array
    source_ids: Array | None
    stencil: LocalStencilPolicy
    address: MortonAddressPlan | None
    request: PointTransferRequest
    source_measure: EvolutionMeasure = eqx.field(static=True)
    target_measure: EvolutionMeasure = eqx.field(static=True)
    neighbors: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        source_points: ArrayLike,
        source_volumes: ArrayLike,
        target_points: ArrayLike,
        target_volumes: ArrayLike,
        /,
        *,
        source_measure: EvolutionMeasure,
        target_measure: EvolutionMeasure,
        source_ids: ArrayLike | None = None,
        neighbors: int = 9,
        stencil: LocalStencilPolicy | None = None,
        address: MortonAddressPlan | None = None,
        request: PointTransferRequest | None = None,
        tolerance: float = 1e-10,
    ) -> None:
        source = _host_points(source_points, "source_points")
        target = _host_points(target_points, "target_points")
        if source.shape[1] != target.shape[1]:
            raise ValueError("Source and target dimensions must match.")
        request_ = (
            PointTransferRequest("conservative-positive") if request is None else request
        )
        if not isinstance(request_, PointTransferRequest):
            raise TypeError("request must be a PointTransferRequest.")
        stencil_ = LocalStencilPolicy(polynomial_degree=1) if stencil is None else stencil
        if not isinstance(stencil_, LocalStencilPolicy):
            raise TypeError("stencil must be a LocalStencilPolicy.")
        if address is not None and not isinstance(address, MortonAddressPlan):
            raise TypeError("address must be a MortonAddressPlan.")
        if isinstance(neighbors, bool) or not isinstance(neighbors, int) or neighbors < 1:
            raise ValueError("neighbors must be a positive integer.")
        self.source_points = jnp.asarray(source)
        self.source_volumes = jnp.asarray(
            _host_measure(source_volumes, source.shape[0], "source_volumes")
        )
        self.target_points = jnp.asarray(target)
        self.target_volumes = jnp.asarray(
            _host_measure(target_volumes, target.shape[0], "target_volumes")
        )
        self.source_ids = None if source_ids is None else jnp.asarray(source_ids)
        self.stencil = stencil_
        self.address = address
        self.request = request_
        self.source_measure = parse(source_measure, EvolutionMeasure, "source_measure")
        self.target_measure = parse(target_measure, EvolutionMeasure, "target_measure")
        self.neighbors = neighbors
        self.tolerance = _positive(tolerance, "tolerance")

    def prepare(self) -> PreparedMeshfreeMeasureTransfer:
        """Host preparation: cross-target stencils, then the audited transfer."""
        neighborhood = MeshfreeNeighborhoodPlan(
            self.source_points,
            self.neighbors,
            targets=self.target_points,
            source_ids=self.source_ids,
            address=self.address,
        ).prepare()
        dimension = self.source_points.shape[1]
        stencils = prepare_local_stencils(
            neighborhood,
            self.source_points,
            self.target_points,
            (MeshfreeFunctional(((0,) * dimension,), (1.0,), name="value"),),
            self.stencil,
        )
        transfer = PointTransferPlan.from_stencils(
            stencils,
            self.source_volumes,
            self.target_volumes,
            request=self.request,
            tolerance=self.tolerance,
        ).prepare()
        return PreparedMeshfreeMeasureTransfer(
            transfer,
            self.source_points,
            self.target_points,
            self.source_measure,
            self.target_measure,
        )


__all__ = [
    "MeshfreeLagrangianEvidence",
    "MeshfreeLagrangianFlowPlan",
    "MeshfreeLagrangianFlowState",
    "MeshfreeLagrangianStatus",
    "MeshfreeLagrangianStepResult",
    "MeshfreeMeasureConversion",
    "MeshfreeMeasureTransferPlan",
    "MeshfreeMeasureTransferResult",
    "MeshfreeSPHComparison",
    "MeshfreeSPHReconstruction",
    "PreparedMeshfreeLagrangianFlow",
    "PreparedMeshfreeMeasureTransfer",
]
