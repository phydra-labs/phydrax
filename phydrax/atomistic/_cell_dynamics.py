#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Extended-system cell dynamics: MTK charts, inertia, operators and measure.

The identified reference is the Martyna-Tobias-Klein isothermal-isobaric
equations (J. Chem. Phys. 101, 4177, 1994) in the Liouville factorization of
Tuckerman, Alejandre, Lopez-Rendon, Jochim and Martyna (J. Phys. A 39, 5629,
2006), column cell convention ``h = H^T``:

* ``dr/dt = p/m + G r`` with ``G = p_g / W`` (``G = (p_eps / W) I`` on the
  isotropic chart);
* ``dp/dt = F - (G + tr(G) / N_f I) p - (particle chain friction) p``;
* ``dh/dt = G h`` (rows: ``dH/dt = H G^T``);
* ``dp_g/dt = Pi_chart[sum_i p_i p_i^T / m_i - sym(dE/d strain)
  - V P_ext I + (2K / N_f) I] - (cell chain friction) p_g``.

``dE/d strain`` is the homogeneous-strain response of the same scalar
Hamiltonian at fixed fractional coordinates and fixed integer images, so
explicitly stored self images and boundary-crossing routes contribute exactly.
The chart projection ``Pi`` is the Frobenius-orthogonal projection onto the
admitted strain-rate subspace: isotropic (scalar ``p_eps`` with
``dp_eps/dt = tr(...)``, ``K_cell = p_eps^2 / 2W``), semi-isotropic
``diag(a, a, c)``, diagonal and symmetric (rotation-free gauge, six cell
degrees of freedom) charts with ``K_cell = tr(p_g^2) / 2W``.  Hydrostatic
``P_ext`` only: a tensor target is a different work/measure contract.

Numerical realization: ``mttk-tuckerman-2006-exponential``; position, cell and
momentum sub-flows are integrated exactly with the native small matrix
exponential and ``phi_1`` actions (not the source's linear cell update).  The
conserved extended energy is ``K + U + K_cell + P_ext V + E_chains``; it is a
diagnostic of the realization, not a claim of stepwise equality with any
external implementation.  Inertia follows the source policy
``W = (N_f + 3) kT tau_p^2``; NPH uses a declared ``inertia_temperature``
only to set ``W`` and never as a physical target.
"""

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    DenseLinearOperator,
    determinant_small_linear,
    inverse_small_linear,
    matrix_exponential_action,
    matrix_phi1_action,
    SmallLinearSolvePlan,
)
from ._neighborhood_bundle import (
    AtomisticNeighborhoodGeometry,
    PreparedAtomisticNeighborhoodBundle,
)
from ._sites import AtomisticSiteDomain
from ._system import PreparedAtomisticSystem
from ._thermal import NoseHooverChainParameters


_CELL_SOLVE = SmallLinearSolvePlan(3)


def bind_atomistic_bundle_geometry(
    bundle: PreparedAtomisticNeighborhoodBundle,
    system: PreparedAtomisticSystem,
    positions: ArrayLike,
    /,
    *,
    cell_vectors: ArrayLike | None = None,
    image_counts: ArrayLike | None = None,
) -> AtomisticNeighborhoodGeometry:
    """Bind one system configuration as the single-case geometry of ``bundle``.

    ``positions`` are the wrapped ``(N, 3)`` coordinates, ``cell_vectors`` the
    ``(r, 3)`` runtime lattice rows (``None``: the prepared cell or finite
    frame) and ``image_counts`` the ``(N, r)`` integer wraps of the unwrapped
    representation.  Interaction sites are realized through the system
    coordinate map, under the same runtime lattice, exactly when a source
    searches interaction sites.
    """
    if not isinstance(bundle, PreparedAtomisticNeighborhoodBundle):
        raise TypeError("bundle must be a PreparedAtomisticNeighborhoodBundle.")
    # Static layout check only: the material identity of the bundle's system is
    # bound at host preparation and refused numerically by the owning program
    # (``bundle_identity`` words), so this binding is traceable.
    if bundle.system.layout_id != system.layout_id:
        raise ValueError("Neighborhood bundle belongs to another atomistic layout.")
    if bundle.plan.case_count != 1:
        raise ValueError("Dynamics binds exactly one neighborhood case.")
    position = jnp.asarray(positions, dtype=system.plan.coordinate_dtype)
    vectors = None
    if cell_vectors is not None:
        vectors = jnp.asarray(cell_vectors, dtype=position.dtype)
        if vectors.size == 0:
            vectors = None
    counts = None if image_counts is None else jnp.asarray(image_counts)
    if counts is not None and counts.size == 0:
        counts = None
    sites = None
    if any(
        source.entity is AtomisticSiteDomain.INTERACTION_SITES
        for source in bundle.plan.sources
    ):
        cell = system.cell
        if cell is None or vectors is None:
            realized = system.coordinate_map.realize(position)
        else:
            realized = system.coordinate_map.realize(
                position,
                cell=cell,
                fractional_positions=cell.fractional_with_vectors(position, vectors),
                cell_vectors=vectors,
            )
        sites = eqx.tree_at(
            lambda value: value.positions, realized, realized.positions[None]
        )
    return bundle.geometry(
        position[None],
        cell_vectors=None if vectors is None else vectors[None],
        image_counts=None if counts is None else counts[None],
        sites=sites,
    )


class AtomisticCellChart(StrEnum):
    """Admitted strain-rate subspace of an extended-system cell method."""

    ISOTROPIC = "isotropic"
    SEMI_ISOTROPIC_XY = "semi-isotropic-xy"
    DIAGONAL = "diagonal"
    TRICLINIC_SYMMETRIC = "triclinic-symmetric"

    @property
    def dimension(self) -> int:
        return {
            AtomisticCellChart.ISOTROPIC: 1,
            AtomisticCellChart.SEMI_ISOTROPIC_XY: 2,
            AtomisticCellChart.DIAGONAL: 3,
            AtomisticCellChart.TRICLINIC_SYMMETRIC: 6,
        }[self]


class MTKBarostatPlan(StrictModule, NonTrainableState):
    """Chart, inertia and admissibility of one MTK cell-momentum controller.

    ``inertia_temperature`` sets ``W = (N_f + 3) k inertia_temperature tau^2``;
    when ``None`` the bound thermodynamic row temperature is used (NPT only).
    ``maximum_condition`` bounds the runtime lattice condition number; a
    candidate cell beyond it is a scientific cell failure.

    The chart, ``time_constant`` and ``maximum_condition`` are static method
    selectors (like every chain relaxation time); whether an
    ``inertia_temperature`` is declared is structural, while its value is a
    dynamic scalar leaf, so inertia-temperature revisions share one executable.
    The canonical plan digest keeps the exact host value and is the dynamic
    ``plan_identity`` (``plan_id`` decodes it on the host).
    """

    inertia_temperature: Array | None
    plan_identity: IdentityWords
    time_constant: float = eqx.field(static=True)
    chart: AtomisticCellChart = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)

    def __init__(
        self,
        time_constant: float,
        /,
        *,
        chart: AtomisticCellChart | str = AtomisticCellChart.ISOTROPIC,
        inertia_temperature: float | None = None,
        maximum_condition: float = 1.0e3,
    ) -> None:
        tau = float(time_constant)
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("Barostat time_constant must be finite and positive.")
        chart_ = AtomisticCellChart(chart)
        reference = None if inertia_temperature is None else float(inertia_temperature)
        if reference is not None and (not np.isfinite(reference) or reference <= 0.0):
            raise ValueError("inertia_temperature must be finite and positive.")
        condition = float(maximum_condition)
        if not np.isfinite(condition) or condition <= 1.0:
            raise ValueError("maximum_condition must exceed one.")
        self.inertia_temperature = (
            None if reference is None else jnp.asarray(reference, dtype=jnp.float64)
        )
        self.plan_identity = IdentityWords.of(
            {
                "kind": "mtk-barostat",
                "mass_policy": "w-nf-plus-3-kt-tau2",
                "time_constant": tau,
                "chart": chart_.value,
                "inertia_temperature": reference,
                "maximum_condition": condition,
            }
        )
        self.time_constant = tau
        self.chart = chart_
        self.maximum_condition = condition

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical barostat digest."""
        return self.plan_identity.fingerprint()


class MTKNPTPlan(StrictModule, NonTrainableState):
    """Integrator-owned MTK isothermal-isobaric method with particle and cell chains.

    Schedule ``Tc(dt/2) Tp(dt/2) B(dt/2) K(dt/2) D(dt) K(dt/2) B(dt/2)
    Tp(dt/2) Tc(dt/2)`` (Tuckerman et al. 2006); ``D`` is the exact affine
    position and cell flow under fixed ``G``.  The canonical plan digest embeds
    the barostat digest (a dynamic inertia temperature), so it is the dynamic
    ``plan_identity``; ``plan_id`` decodes it on the host.
    """

    barostat: MTKBarostatPlan
    particle_chain: NoseHooverChainParameters
    cell_chain: NoseHooverChainParameters
    plan_identity: IdentityWords
    step_size: float = eqx.field(static=True)

    def __init__(
        self,
        step_size: float,
        barostat: MTKBarostatPlan,
        particle_chain: NoseHooverChainParameters,
        cell_chain: NoseHooverChainParameters,
        /,
    ) -> None:
        step = float(step_size)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        if not isinstance(barostat, MTKBarostatPlan):
            raise TypeError("barostat must be an MTKBarostatPlan.")
        if not isinstance(particle_chain, NoseHooverChainParameters) or not isinstance(
            cell_chain, NoseHooverChainParameters
        ):
            raise TypeError("Chains must be NoseHooverChainParameters.")
        self.barostat = barostat
        self.particle_chain = particle_chain
        self.cell_chain = cell_chain
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-mtk-npt",
                "realization": "mttk-tuckerman-2006-exponential",
                "step_size": step,
                "barostat": barostat.plan_id,
                "particle_chain": particle_chain.parameters_id,
                "cell_chain": cell_chain.parameters_id,
            }
        )
        self.step_size = step

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical MTK NPT digest."""
        return self.plan_identity.fingerprint()

    @property
    def canonical(self) -> bool:
        return True

    @property
    def stochastic(self) -> bool:
        return False


class MTKNPHPlan(StrictModule, NonTrainableState):
    """Integrator-owned MTK isenthalpic method: no thermostat of any kind.

    Schedule ``B(dt/2) K(dt/2) D(dt) K(dt/2) B(dt/2)``; ``K + U + K_cell +
    P_ext V`` is the conserved extended energy.  ``barostat.inertia_temperature``
    is required because no thermodynamic temperature exists.  The canonical
    plan digest is the dynamic ``plan_identity`` (``plan_id`` on the host).
    """

    barostat: MTKBarostatPlan
    plan_identity: IdentityWords
    step_size: float = eqx.field(static=True)

    def __init__(self, step_size: float, barostat: MTKBarostatPlan, /) -> None:
        step = float(step_size)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        if not isinstance(barostat, MTKBarostatPlan):
            raise TypeError("barostat must be an MTKBarostatPlan.")
        if barostat.inertia_temperature is None:
            raise ValueError(
                "NPH has no thermodynamic temperature; declare the barostat "
                "inertia_temperature explicitly."
            )
        self.barostat = barostat
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-mtk-nph",
                "realization": "mttk-tuckerman-2006-exponential",
                "step_size": step,
                "barostat": barostat.plan_id,
            }
        )
        self.step_size = step

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical MTK NPH digest."""
        return self.plan_identity.fingerprint()

    @property
    def canonical(self) -> bool:
        return False

    @property
    def stochastic(self) -> bool:
        return False


class AtomisticCellControllerState(StrictModule):
    """Typed continuation of an MTK cell controller.

    ``momentum`` is the chart momentum (scalar ``p_eps`` on the isotropic chart,
    otherwise a symmetric ``(3, 3)`` ``p_g`` inside the chart subspace);
    ``reference_vectors`` are the lattice rows at controller initialization,
    retained for strain diagnostics.
    """

    momentum: Array
    reference_vectors: Array


def initial_cell_controller(
    barostat: MTKBarostatPlan, vectors: ArrayLike, /
) -> AtomisticCellControllerState:
    value = jnp.asarray(vectors)
    shape = () if barostat.chart is AtomisticCellChart.ISOTROPIC else (3, 3)
    return AtomisticCellControllerState(jnp.zeros(shape, dtype=value.dtype), value)


def barostat_mass(
    barostat: MTKBarostatPlan,
    degrees_of_freedom: ArrayLike,
    thermal_energy: ArrayLike,
    /,
) -> Array:
    """``W = (N_f + 3) kT tau^2``; ``N_f`` may be a traced integer scalar."""
    thermal = jnp.asarray(thermal_energy)
    degrees = jnp.asarray(degrees_of_freedom, dtype=thermal.dtype)
    return (degrees + 3.0) * thermal * barostat.time_constant**2


def project_cell_force(chart: AtomisticCellChart, force: ArrayLike, /) -> Array:
    """Frobenius projection of a symmetric tensor force onto the chart momentum."""
    value = jnp.asarray(force)
    if chart is AtomisticCellChart.ISOTROPIC:
        return jnp.trace(value)
    if chart is AtomisticCellChart.SEMI_ISOTROPIC_XY:
        lateral = 0.5 * (value[0, 0] + value[1, 1])
        return jnp.diag(jnp.stack((lateral, lateral, value[2, 2])))
    if chart is AtomisticCellChart.DIAGONAL:
        return jnp.diag(jnp.diagonal(value))
    return 0.5 * (value + value.T)


def cell_velocity_gradient(
    chart: AtomisticCellChart, momentum: ArrayLike, mass: ArrayLike, /
) -> Array:
    value = jnp.asarray(momentum)
    if chart is AtomisticCellChart.ISOTROPIC:
        return (value / mass) * jnp.eye(3, dtype=value.dtype)
    return value / mass


def cell_kinetic_energy(
    chart: AtomisticCellChart, momentum: ArrayLike, mass: ArrayLike, /
) -> Array:
    value = jnp.asarray(momentum)
    if chart is AtomisticCellChart.ISOTROPIC:
        return 0.5 * value * value / mass
    return 0.5 * jnp.sum(value * value) / mass


def cell_volume(vectors: ArrayLike, /) -> Array:
    return jnp.abs(determinant_small_linear(_CELL_SOLVE, jnp.asarray(vectors)))


def cell_admissible(vectors: ArrayLike, maximum_condition: float, /) -> Array:
    """Finite, right-handed and conditioned runtime lattice (native small solve)."""
    value = jnp.asarray(vectors)
    determinant = determinant_small_linear(_CELL_SOLVE, value)
    inverse = inverse_small_linear(_CELL_SOLVE, value)
    return (
        jnp.all(jnp.isfinite(value))
        & jnp.isfinite(determinant)
        & (determinant > 0.0)
        & inverse.successful
        & (inverse.condition_estimate <= maximum_condition)
    )


def kinetic_tensor(
    momenta: ArrayLike,
    inverse_masses: ArrayLike,
    mobile_mask: ArrayLike,
    kinetic_to_energy: float,
    /,
) -> Array:
    """``sum_i p_i p_i^T / m_i`` in energy units (twice the kinetic tensor)."""
    momentum = jnp.asarray(momenta)
    weights = jnp.where(
        jnp.asarray(mobile_mask), jnp.asarray(inverse_masses, dtype=momentum.dtype), 0.0
    )
    return kinetic_to_energy * contract("ni,nj,n->ij", momentum, momentum, weights)


def barostat_force(
    twice_kinetic_tensor: ArrayLike,
    strain_derivative: ArrayLike,
    volume: ArrayLike,
    pressure: ArrayLike,
    degrees_of_freedom: ArrayLike,
    /,
) -> Array:
    """Unprojected MTK cell force ``2K_t - sym(dE/d strain) - (V P - 2K/N_f) I``."""
    kinetic = jnp.asarray(twice_kinetic_tensor)
    strain = jnp.asarray(strain_derivative, dtype=kinetic.dtype)
    degrees = jnp.asarray(degrees_of_freedom, dtype=kinetic.dtype)
    identity = jnp.eye(3, dtype=kinetic.dtype)
    twice = jnp.trace(kinetic)
    return (
        kinetic
        - 0.5 * (strain + strain.T)
        - (volume * pressure - twice / degrees) * identity
    )


class SmallFlow(StrictModule):
    exponential: Array
    phi1: Array
    successful: Array


def small_affine_flow(generator: ArrayLike, duration: ArrayLike, /) -> SmallFlow:
    """``exp(t A)`` and ``phi_1(t A)`` of a 3x3 generator via native actions."""
    matrix = jnp.asarray(generator)
    operator = DenseLinearOperator(matrix, operator_id="atomistic-cell-generator")
    span = jnp.asarray(duration, dtype=matrix.dtype)
    identity = jnp.eye(3, dtype=matrix.dtype)

    def column(vector: Array) -> tuple[Array, Array, Array]:
        exponential = matrix_exponential_action(operator, vector, span)
        phi = matrix_phi1_action(operator, vector, span)
        return (
            jnp.asarray(exponential.value),
            jnp.asarray(phi.value),
            exponential.successful & phi.successful,
        )

    exponential, phi, successful = jax.vmap(column, in_axes=1, out_axes=(1, 1, 0))(
        identity
    )
    return SmallFlow(exponential, phi, jnp.all(successful))


def mtk_momentum_kick(
    generator: ArrayLike,
    degrees_of_freedom: ArrayLike,
    momenta: ArrayLike,
    momentum_rate: ArrayLike,
    duration: ArrayLike,
    mobile_mask: ArrayLike,
    /,
) -> tuple[Array, Array]:
    """Exact ``dp/dt = f - (G + tr G / N_f I) p`` over ``duration`` at fixed ``G``."""
    gradient = jnp.asarray(generator)
    degrees = jnp.asarray(degrees_of_freedom, dtype=gradient.dtype)
    drag = gradient + jnp.trace(gradient) / degrees * jnp.eye(3, dtype=gradient.dtype)
    flow = small_affine_flow(-drag, duration)
    momentum = jnp.asarray(momenta)
    rate = jnp.asarray(momentum_rate, dtype=momentum.dtype)
    span = jnp.asarray(duration, dtype=momentum.dtype)
    updated = contract("ij,nj->ni", flow.exponential, momentum) + span * contract(
        "ij,nj->ni", flow.phi1, rate
    )
    return jnp.where(jnp.asarray(mobile_mask)[:, None], updated, 0.0), flow.successful


class MTKDrift(StrictModule):
    unwrapped_positions: Array
    cell_vectors: Array
    successful: Array


def mtk_drift(
    generator: ArrayLike,
    origin: ArrayLike,
    unwrapped_positions: ArrayLike,
    velocities: ArrayLike,
    cell_vectors: ArrayLike,
    duration: ArrayLike,
    mobile_mask: ArrayLike,
    /,
) -> MTKDrift:
    """Exact affine flow ``dr/dt = v + G (r - o)``, ``dH/dt = H G^T`` at fixed ``G``.

    Immobile particles follow the homogeneous deformation only (their fractional
    coordinates are retained), so the material representation stays affine.
    """
    gradient = jnp.asarray(generator)
    flow = small_affine_flow(gradient, duration)
    position = jnp.asarray(unwrapped_positions)
    center = jnp.asarray(origin, dtype=position.dtype)
    velocity = jnp.asarray(velocities, dtype=position.dtype)
    span = jnp.asarray(duration, dtype=position.dtype)
    relative = position - center
    deformed = center + contract("ij,nj->ni", flow.exponential, relative)
    moved = deformed + span * contract("ij,nj->ni", flow.phi1, velocity)
    updated = jnp.where(jnp.asarray(mobile_mask)[:, None], moved, deformed)
    vectors = contract("ki,ji->kj", jnp.asarray(cell_vectors), flow.exponential)
    return MTKDrift(updated, vectors, flow.successful)


__all__ = [
    "AtomisticCellChart",
    "AtomisticCellControllerState",
    "MTKBarostatPlan",
    "MTKDrift",
    "MTKNPHPlan",
    "MTKNPTPlan",
    "SmallFlow",
    "barostat_force",
    "barostat_mass",
    "bind_atomistic_bundle_geometry",
    "cell_admissible",
    "cell_kinetic_energy",
    "cell_velocity_gradient",
    "cell_volume",
    "initial_cell_controller",
    "kinetic_tensor",
    "mtk_drift",
    "mtk_momentum_kick",
    "project_cell_force",
    "small_affine_flow",
]
