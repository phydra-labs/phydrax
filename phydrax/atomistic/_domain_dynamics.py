#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Distributed atomistic dynamics over one ``AtomisticDomainState``.

Per-atom arrays (positions, momenta, masses, forces, constraint rows) stay
owner-blocked; controllers, ledger, RNG key, step counter, cell and every
scientific scalar are replicated.  Each global scientific quantity of a method
(kinetic energy and tensor, linear momentum, reservoir heat, constraint
residuals, rank evidence, degrees of freedom, collective-variable sums and
optimizer inner products) is a native collective action inside the owner map
of the domain ownership, reduced with the domain reduction policy (integer
counts and maxima are exact collectives).

The executed schedule is P5's ``AtomisticSplittingPlan`` of the declared
integrator; controller updates reuse the P5 kernels, which are pure functions
of the reduced scalars, so every owner computes the identical replicated
controller.  Stochastic draws are addressed exactly as on one device: per
particle by stable atom ID, step counter, operator and realization
(``stable_particle_normals``) and per scalar draw by step and operator; no draw
depends on the owner, slot or lane.  A transition is one transaction: every
force evaluation first certifies the accepted domain topology and, when the
certificate fails, rebuilds (wraps, migrates and re-routes every
continuation leaf, including constraint rows) inside the step; the candidate
is committed atomically only when every owner admits every phase, otherwise
the accepted state is returned bitwise unchanged.

Distance constraints (P5 SHAKE/RATTLE) have their own support topology,
independent of the Verlet skin: each connected constraint component is owned
by the owner of its anchor (smallest stable member ID), member atoms are
resolved by stable ID through a bounded hashed directory and gathered through
a prepared constraint ``DistributedHaloPlan``; the anchor owner runs the P5
Jacobi SHAKE/RATTLE iterations of the complete component and returns each
member's result to its owner exactly once through the halo transpose.
Residuals, rank evidence and success are global collectives.

Per-atom continuations that P5 keeps in its single-device arrays are
owner-blocked atom payload here and migrate with their atoms in every rebuild:
generalized-Langevin memory (``"generalized-langevin-memory"``), the fixed-atom
mask (``"mobile"``) and molecule labels for molecular-center volume scaling
(``"molecule"``).  P5's scheduled isotropic Monte Carlo volume move runs as
``monte_carlo_volume_move`` with its draws, Jacobian and counter rules.
"""

from __future__ import annotations

import math
from typing import Any, final, Literal, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.sharding import PartitionSpec
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._identity import IdentityWords
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_allfinite
from ..backends.distributed import JaxCollectiveProvider
from ..discretization.spatial._distributed_relations import (
    DistributedHaloEvidence,
    DistributedHaloPlan,
    DistributedMigrationEvidence,
    DistributedOwnershipPlan,
    DistributedPointLayout,
)
from ..linalg._dense_svd import economy_svd
from ..typing import Bool, Dim, Float, Int32, parse, Scalar
from ._barostat import IsotropicMonteCarloBarostatPlan, MONTE_CARLO_VOLUME_OPERATOR_ID
from ._cell_dynamics import (
    barostat_force,
    barostat_mass,
    cell_admissible,
    cell_kinetic_energy,
    cell_velocity_gradient,
    cell_volume,
    initial_cell_controller,
    kinetic_tensor,
    mtk_drift,
    mtk_momentum_kick,
    MTKNPHPlan,
    MTKNPTPlan,
    project_cell_force,
)
from ._constraints import DistanceConstraintPlan
from ._domain_plan import AtomisticDomainPlan, AtomisticDomainReductionPolicy
from ._domain_state import (
    AtomisticDomainConstraintState,
    AtomisticDomainForceCache,
    AtomisticDomainState,
    owner_directory_lookup,
)
from ._dynamics import (
    _schedule,
    _strain_reasons,
    advance_atomistic_energy_ledger,
    AtomisticControllerState,
    AtomisticDynamicsStatus,
    AtomisticEnergyLedgerState,
    AtomisticStepRejectionReason,
    initial_atomistic_energy_ledger,
    VelocityVerletPlan,
)
from ._ensemble_advanced import AtomisticSplittingPlan, SplittingOperatorKind
from ._stochastic_runtime import (
    apply_generalized_langevin,
    GeneralizedLangevinMemoryState,
    GeneralizedLangevinPlan,
)
from ._thermal import (
    apply_baoab_ornstein_uhlenbeck,
    BAOABLangevinPlan,
    CSVR_FIRST_OPERATOR_ID,
    CSVR_SECOND_OPERATOR_ID,
    deterministic_velocity_rescaling_scale,
    initial_nose_hoover_chain_state,
    nose_hoover_chain_energy,
    NoseHooverChainParameters,
    NoseHooverChainPlan,
    propagate_nose_hoover_chain,
    stochastic_velocity_rescaling_scale,
    StochasticVelocityRescalingPlan,
    VelocityRescalingPlan,
)
from ._thermodynamic import (
    _admit_method,
    AtomisticEnsembleKind,
    AtomisticMomentumRestriction,
    AtomisticVolumeCoordinateConvention,
)
from ._units import AtomisticUnitSystem


if TYPE_CHECKING:
    from ._domain_runtime import AtomisticDomainEvaluation, PreparedAtomisticDomainRuntime


class _AtomRowDim(Dim, minimum=1):
    """Owner-blocked atom rows: ``owner_count * local_capacity``."""


class _ConstraintRowDim(Dim, minimum=1):
    """Owner-blocked constraint rows: ``owner_count * row_capacity``."""


_INT64_MAX = int(np.iinfo(np.int64).max)
_REPLICATED = PartitionSpec()

AtomisticDomainIntegratorPlan = (
    VelocityVerletPlan
    | BAOABLangevinPlan
    | NoseHooverChainPlan
    | StochasticVelocityRescalingPlan
    | VelocityRescalingPlan
    | GeneralizedLangevinPlan
    | MTKNPTPlan
    | MTKNPHPlan
)
_DOMAIN_INTEGRATORS = (
    VelocityVerletPlan,
    BAOABLangevinPlan,
    NoseHooverChainPlan,
    StochasticVelocityRescalingPlan,
    VelocityRescalingPlan,
    GeneralizedLangevinPlan,
    MTKNPTPlan,
    MTKNPHPlan,
)
_CELL_METHODS = (MTKNPTPlan, MTKNPHPlan)
_CAPACITY = int(AtomisticStepRejectionReason.ROUTE_CAPACITY)
_CONSTRAINT = int(AtomisticStepRejectionReason.CONSTRAINT)


def _spec(axis: str, ndim: int, /) -> PartitionSpec:
    return PartitionSpec(axis, *([None] * (ndim - 1)))


def _positive(name: str, value: int, /) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _flag(condition: Array, reason: int, /) -> Array:
    return jnp.where(condition, reason, 0).astype(jnp.int32)


def _inverse_masses(masses: Array, active: Array, /) -> Array:
    safe = jnp.where(active, masses, 1.0)
    return jnp.where(active, 1.0 / safe, 0.0).astype(masses.dtype)


def _select_tree[T](predicate: Array, candidate: T, previous: T, /) -> T:
    """Leafwise transactional selection of congruent trees (typed keys kept typed)."""

    def select(new: Any, old: Any) -> Any:
        if not eqx.is_array(old):
            return old
        if jax.dtypes.issubdtype(old.dtype, jax.dtypes.prng_key):
            return jr.wrap_key_data(
                jnp.where(predicate, jr.key_data(new), jr.key_data(old)),
                impl=jr.key_impl(old),
            )
        return jnp.where(predicate, new, old)

    return jax.tree.map(select, candidate, previous)


# --------------------------------------------------------------------------
# Owner-region scientific reductions (call inside ``ownership.map``)
# --------------------------------------------------------------------------


def owner_kinetic_energy(
    momenta: Array,
    masses: Array,
    active: Array,
    kinetic_to_energy: float,
    /,
    *,
    reduction: AtomisticDomainReductionPolicy,
    axis_name: str,
) -> Array:
    """Global ``sum_i |p_i|^2 / (2 m_i)`` of active atoms (P5 kinetic energy)."""
    inverse = _inverse_masses(masses, active)
    per_atom = (
        0.5 * kinetic_to_energy * jnp.sum(momenta * momenta * inverse[:, None], axis=-1)
    )
    return reduction.owner_sum(jnp.sum(jnp.where(active, per_atom, 0.0)), axis_name)


def owner_kinetic_tensor(
    momenta: Array,
    masses: Array,
    active: Array,
    kinetic_to_energy: float,
    /,
    *,
    reduction: AtomisticDomainReductionPolicy,
    axis_name: str,
) -> Array:
    """Global ``sum_i p_i p_i^T / m_i`` (twice the kinetic tensor, P5 convention)."""
    local = kinetic_tensor(
        momenta, _inverse_masses(masses, active), active, kinetic_to_energy
    )
    return reduction.owner_sum(local, axis_name)


def owner_linear_momentum(
    momenta: Array,
    active: Array,
    /,
    *,
    reduction: AtomisticDomainReductionPolicy,
    axis_name: str,
) -> Array:
    return reduction.owner_sum(
        jnp.sum(jnp.where(active[:, None], momenta, 0.0), axis=0), axis_name
    )


def owner_masked_sum(
    values: Array,
    active: Array,
    /,
    *,
    reduction: AtomisticDomainReductionPolicy,
    axis_name: str,
) -> Array:
    """Global sum of per-atom contributions (collective-variable sums, masses, ...)."""
    mask = active.reshape(active.shape + (1,) * (values.ndim - 1))
    return reduction.owner_sum(jnp.sum(jnp.where(mask, values, 0.0), axis=0), axis_name)


def owner_inner_product(
    left: Array,
    right: Array,
    active: Array,
    /,
    *,
    reduction: AtomisticDomainReductionPolicy,
    axis_name: str,
) -> Array:
    """Global Euclidean inner product of owner-blocked per-atom fields (optimizers)."""
    product = jnp.sum((left * right).reshape((left.shape[0], -1)), axis=1)
    return reduction.owner_sum(jnp.sum(jnp.where(active, product, 0.0)), axis_name)


def owner_count(active: Array, /, *, axis_name: str) -> Array:
    """Exact global count of true rows (integer collective)."""
    return JaxCollectiveProvider(axis_name).sum(jnp.sum(active, dtype=jnp.int32))


def owner_degrees_of_freedom(
    atom_active: Array,
    constraint_active: Array | None,
    momentum_restriction: AtomisticMomentumRestriction,
    /,
    *,
    axis_name: str,
) -> Array:
    """Thermostatted DOF ``3 N - N_constraints - 3 [center of mass]`` (exact)."""
    atoms = owner_count(atom_active, axis_name=axis_name)
    rows = (
        jnp.zeros((), dtype=jnp.int32)
        if constraint_active is None
        else owner_count(constraint_active, axis_name=axis_name)
    )
    removed = 3 if momentum_restriction == "center-of-mass" else 0
    return (3 * atoms - rows - removed).astype(jnp.int32)


# --------------------------------------------------------------------------
# Ensemble and constraint plans
# --------------------------------------------------------------------------


MOBILE_PAYLOAD = "mobile"
MOLECULE_PAYLOAD = "molecule"
GLE_MEMORY_PAYLOAD = "generalized-langevin-memory"


@final
class AtomisticDomainEnsemblePlan(StrictModule, NonTrainableState):
    """Thermodynamic state and phase-space measure of one distributed system.

    The P5 ``AtomisticPhaseSpaceMeasurePlan`` semantics without a global host
    system object: ``atom_count`` active atoms of which ``mobile_count`` are
    mobile (fixed atoms are flagged by the owner-blocked ``"mobile"`` atom
    payload, which migrates with its atom), a volume convention that scales
    every active atom (``"atomic"``) or ``molecule_count`` molecule centers
    (``"molecular-center"``, molecules named by the ``"molecule"`` atom
    payload) and the momentum restriction.  The executed degrees of freedom
    ``3 mobile - constraints - 3 [center of mass]`` are verified against exact
    collective counts at every transition.  As in P5, a center-of-mass
    restriction requires every active atom to be mobile.

    PyTree layout: the thermodynamic state is dynamic — ``temperature`` and
    ``pressure`` (float64 scalars, ``None`` exactly when the ensemble kind
    declares none), ``atom_count`` and ``mobile_count`` (int32 scalars) — and
    ``plan_identity`` is the exact canonical plan digest (``plan_id`` decodes
    it on the host), so every state point of one layout shares one
    executable.  Static layout: the ensemble kind, the momentum restriction,
    the volume convention, ``fixed_atoms`` (whether the ``"mobile"`` payload
    exists) and ``molecule_count`` (the molecule segment count of
    molecular-center scaling, a shape).
    """

    units: AtomisticUnitSystem
    temperature: Array | None
    pressure: Array | None
    atom_count: Array
    mobile_count: Array
    plan_identity: IdentityWords
    ensemble: AtomisticEnsembleKind = eqx.field(static=True)
    fixed_atoms: bool = eqx.field(static=True)
    molecule_count: int | None = eqx.field(static=True)
    momentum_restriction: AtomisticMomentumRestriction = eqx.field(static=True)
    volume_coordinate_convention: AtomisticVolumeCoordinateConvention = eqx.field(
        static=True
    )

    def __init__(
        self,
        units: AtomisticUnitSystem,
        /,
        *,
        ensemble: AtomisticEnsembleKind,
        atom_count: int,
        temperature: float | None = None,
        pressure: float | None = None,
        mobile_count: int | None = None,
        momentum_restriction: AtomisticMomentumRestriction = "none",
        volume_coordinate_convention: AtomisticVolumeCoordinateConvention = "atomic",
        molecule_count: int | None = None,
    ) -> None:
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be an AtomisticUnitSystem.")
        ensemble = parse(ensemble, AtomisticEnsembleKind, "ensemble")
        momentum_restriction = parse(
            momentum_restriction, AtomisticMomentumRestriction, "momentum_restriction"
        )
        count = _positive("atom_count", atom_count)
        mobile = (
            count if mobile_count is None else _positive("mobile_count", mobile_count)
        )
        if mobile > count:
            raise ValueError("mobile_count cannot exceed atom_count.")
        if momentum_restriction == "center-of-mass" and mobile != count:
            raise ValueError(
                "A center-of-mass momentum restriction requires every active atom "
                "to be mobile; fixed atoms exchange momentum with the system."
            )
        if volume_coordinate_convention == "atomic":
            if molecule_count is not None:
                raise ValueError("Atomic volume scaling declares no molecule_count.")
            molecules = None
        elif volume_coordinate_convention == "molecular-center":
            if molecule_count is None:
                raise ValueError("Molecular-center scaling requires molecule_count.")
            molecules = _positive("molecule_count", molecule_count)
        else:
            raise ValueError(
                "volume_coordinate_convention must be 'molecular-center' or 'atomic'."
            )
        temperature_ = None if temperature is None else float(temperature)
        pressure_ = None if pressure is None else float(pressure)
        positive_temperature = (
            temperature_ is not None
            and math.isfinite(temperature_)
            and temperature_ > 0.0
        )
        finite_pressure = pressure_ is not None and math.isfinite(pressure_)
        admitted = {
            "nve": temperature_ is None and pressure_ is None,
            "nvt": positive_temperature and pressure_ is None,
            "npt": positive_temperature and finite_pressure,
            "nph": temperature_ is None and finite_pressure,
        }[ensemble]
        if not admitted:
            raise ValueError(
                "Thermodynamic intensive variables do not match the ensemble."
            )
        self.units = units
        self.temperature = (
            None if temperature_ is None else jnp.asarray(temperature_, dtype=jnp.float64)
        )
        self.pressure = (
            None if pressure_ is None else jnp.asarray(pressure_, dtype=jnp.float64)
        )
        self.atom_count = jnp.asarray(count, dtype=jnp.int32)
        self.mobile_count = jnp.asarray(mobile, dtype=jnp.int32)
        self.ensemble = ensemble
        self.fixed_atoms = mobile != count
        self.molecule_count = molecules
        self.momentum_restriction = momentum_restriction
        self.volume_coordinate_convention = volume_coordinate_convention
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-domain-ensemble",
                "units": units.unit_system_id,
                "ensemble": ensemble,
                "temperature": temperature_,
                "pressure": pressure_,
                "atom_count": count,
                "mobile_count": mobile,
                "molecule_count": molecules,
                "momentum_restriction": momentum_restriction,
                "volume_coordinate_convention": volume_coordinate_convention,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical ensemble-plan digest."""
        return self.plan_identity.fingerprint()

    @property
    def scaled_entity_count(self) -> Array | int:
        """Scaled volume entities: active atoms (traced) or declared molecules."""
        return self.atom_count if self.molecule_count is None else self.molecule_count


@final
class AtomisticDomainConstraintPlan(StrictModule, NonTrainableState):
    """Distributed P5 SHAKE/RATTLE constraints and their own support capacities.

    ``constraint_count`` global distance rows (each removes one DOF, constant
    rank certified per geometry as in P5).  Capacities, all refused (never
    truncated) when exceeded: ``row_capacity`` rows per owner;
    ``halo_capacity`` gathered member atoms per (requesting, owning) owner
    pair; ``directory_capacity`` stable-ID registrations or queries per
    (sender, directory bucket) owner pair; ``migration_capacity`` rows per
    owner pair of one rebuild; ``component_row_capacity`` rows per connected
    component in the per-component rank Jacobian (members are bounded by
    ``method.maximum_component_atoms``).  None of them is a Verlet skin.

    Every field is declared layout: ``constraint_count`` is the logical row
    count of the constraint-row layout (it fixes the row ownership's static
    ``logical_count``), the capacities size buffers, and ``plan_id`` is the
    configuration fingerprint of the method settings and these counts.
    Distances (``targets``) are dynamic constraint-row data.
    """

    method: DistanceConstraintPlan
    constraint_count: int = eqx.field(static=True)
    row_capacity: int = eqx.field(static=True)
    halo_capacity: int = eqx.field(static=True)
    directory_capacity: int = eqx.field(static=True)
    migration_capacity: int = eqx.field(static=True)
    component_row_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: DistanceConstraintPlan,
        /,
        *,
        constraint_count: int,
        row_capacity: int,
        halo_capacity: int,
        directory_capacity: int,
        migration_capacity: int,
        component_row_capacity: int,
    ) -> None:
        if not isinstance(method, DistanceConstraintPlan):
            raise TypeError("method must be a DistanceConstraintPlan.")
        values = {
            name: _positive(name, value)
            for name, value in (
                ("constraint_count", constraint_count),
                ("row_capacity", row_capacity),
                ("halo_capacity", halo_capacity),
                ("directory_capacity", directory_capacity),
                ("migration_capacity", migration_capacity),
                ("component_row_capacity", component_row_capacity),
            )
        }
        self.method = method
        self.constraint_count = values["constraint_count"]
        self.row_capacity = values["row_capacity"]
        self.halo_capacity = values["halo_capacity"]
        self.directory_capacity = values["directory_capacity"]
        self.migration_capacity = values["migration_capacity"]
        self.component_row_capacity = values["component_row_capacity"]
        self.plan_id = canonical_fingerprint(
            {"kind": "atomistic-domain-constraints", "method": method.plan_id, **values}
        )


# --------------------------------------------------------------------------
# Constraint rows: host rows, directory, binding, migration hook
# --------------------------------------------------------------------------


@final
class AtomisticDomainConstraintRows(StrictModule):
    """Host rows of a constraint state held by this process (no global gather).

    ``R = (owners of this process) * row_capacity`` owner-major rows in the
    process's owner order; rows may sit on any owner of the process, import
    migrates them to their anchor owners on device.  Padding rows are
    inactive with ``atoms``/``components`` ``-1``.
    """

    row_ids: np.ndarray
    atoms: np.ndarray
    components: np.ndarray
    targets: np.ndarray
    lagrange: np.ndarray
    active: np.ndarray
    position_residual: np.ndarray
    velocity_residual: np.ndarray
    topology_epoch: np.ndarray
    row_capacity: int = eqx.field(static=True)
    row_count: int = eqx.field(static=True)
    halo_capacity: int = eqx.field(static=True)
    directory_capacity: int = eqx.field(static=True)
    migration_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        row_ids: ArrayLike,
        atoms: ArrayLike,
        components: ArrayLike,
        targets: ArrayLike,
        lagrange: ArrayLike,
        active: ArrayLike,
        position_residual: ArrayLike,
        velocity_residual: ArrayLike,
        topology_epoch: ArrayLike,
        row_capacity: int,
        row_count: int,
        halo_capacity: int,
        directory_capacity: int,
        migration_capacity: int,
        plan_id: str,
    ) -> None:
        capacity = _positive("row_capacity", row_capacity)
        ids = np.asarray(row_ids, dtype=np.int64)
        members = np.asarray(atoms, dtype=np.int64)
        anchors = np.asarray(components, dtype=np.int64)
        distances = np.asarray(targets)
        multipliers = np.asarray(lagrange)
        mask = np.asarray(active, dtype=np.bool_)
        rows = ids.shape[0] if ids.ndim == 1 else -1
        if (
            rows <= 0
            or rows % capacity
            or members.shape != (rows, 2)
            or anchors.shape != (rows,)
            or distances.shape != (rows,)
            or multipliers.shape != (rows,)
            or mask.shape != (rows,)
        ):
            raise ValueError("Constraint rows must be aligned process-local row blocks.")
        if (
            not np.issubdtype(distances.dtype, np.floating)
            or distances.dtype != multipliers.dtype
        ):
            raise TypeError("targets and lagrange must share one floating dtype.")
        count = _positive("row_count", row_count)
        if np.any(mask & ((ids < 0) | (ids >= count))) or np.unique(
            ids[mask]
        ).size != int(np.count_nonzero(mask)):
            raise ValueError("Active row IDs must be distinct and lie in [0, row_count).")
        if np.any(mask[:, None] & (members < 0)) or np.any(mask & (anchors < 0)):
            raise ValueError("Active constraint rows must name non-negative stable IDs.")
        if np.any(mask & (members[:, 0] == members[:, 1])):
            raise ValueError("A distance constraint must join two distinct atoms.")
        if not np.all(np.isfinite(distances[mask])) or np.any(distances[mask] <= 0.0):
            raise ValueError("Constraint distances must be finite and positive.")
        identity = str(plan_id).strip()
        if not identity:
            raise ValueError("plan_id must be non-empty.")
        self.row_ids = ids
        self.atoms = np.where(mask[:, None], members, -1)
        self.components = np.where(mask, anchors, -1)
        self.targets = np.where(mask, distances, 0.0).astype(distances.dtype)
        self.lagrange = np.where(mask, multipliers, 0.0).astype(distances.dtype)
        self.active = mask
        self.position_residual = np.asarray(
            position_residual, dtype=distances.dtype
        ).reshape(())
        self.velocity_residual = np.asarray(
            velocity_residual, dtype=distances.dtype
        ).reshape(())
        self.topology_epoch = np.asarray(topology_epoch, dtype=np.int32).reshape(())
        self.row_capacity = capacity
        self.row_count = count
        self.halo_capacity = _positive("halo_capacity", halo_capacity)
        self.directory_capacity = _positive("directory_capacity", directory_capacity)
        self.migration_capacity = _positive("migration_capacity", migration_capacity)
        self.plan_id = identity


def domain_constraint_rows(
    plan: AtomisticDomainConstraintPlan,
    ownership: DistributedOwnershipPlan,
    pairs: ArrayLike,
    distances: ArrayLike,
    /,
) -> AtomisticDomainConstraintRows:
    """Host rows of global constraints for single-process global ingress.

    ``pairs`` hold stable atom IDs; row ``m`` keeps global ID ``m`` (the P5
    topology order).  Components are the connected components of the
    constraint graph; a component wider than ``method.maximum_component_atoms``
    or taller than ``component_row_capacity`` is refused before allocation.
    Rows are placed round-robin by row ID; import migrates them to anchors.
    """
    if not isinstance(plan, AtomisticDomainConstraintPlan):
        raise TypeError("plan must be an AtomisticDomainConstraintPlan.")
    if not isinstance(ownership, DistributedOwnershipPlan):
        raise TypeError("ownership must be a DistributedOwnershipPlan.")
    if set(ownership.owner_processes) != {jax.process_index()}:
        raise ValueError(
            "Global constraint rows require one process owning every owner; "
            "multi-process ingress supplies per-process AtomisticDomainConstraintRows."
        )
    members = np.asarray(pairs)
    targets = np.asarray(distances)
    if (
        members.ndim != 2
        or members.shape[1] != 2
        or not np.issubdtype(members.dtype, np.integer)
    ):
        raise ValueError("pairs must be (M, 2) integer stable atom IDs.")
    members = members.astype(np.int64)
    count = members.shape[0]
    if count != plan.constraint_count:
        raise ValueError("pairs must hold exactly plan.constraint_count rows.")
    if targets.shape != (count,) or not np.issubdtype(targets.dtype, np.floating):
        raise ValueError("distances must hold one floating distance per pair.")
    if np.any(members < 0) or np.any(members[:, 0] == members[:, 1]):
        raise ValueError("Constraint pairs must join two distinct non-negative IDs.")
    unordered = np.sort(members, axis=1)
    if np.unique(unordered, axis=0).shape[0] != count:
        raise ValueError("Constraint pairs must be unique.")
    parent: dict[int, int] = {}

    def find(value: int) -> int:
        root = value
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[value] != root:
            parent[value], value = root, parent[value]
        return root

    for left, right in members.tolist():
        first, second = find(int(left)), find(int(right))
        if first != second:
            parent[max(first, second)] = min(first, second)
    groups: dict[int, list[int]] = {}
    for atom in parent:
        groups.setdefault(find(atom), []).append(atom)
    anchor = {root: min(atoms) for root, atoms in groups.items()}
    width = max(len(atoms) for atoms in groups.values())
    if width > plan.method.maximum_component_atoms:
        raise ValueError(
            f"A constraint component spans {width} atoms; the method admits "
            f"{plan.method.maximum_component_atoms}."
        )
    components = np.asarray([anchor[find(int(left))] for left in members[:, 0]], np.int64)
    height = int(np.max(np.unique(components, return_counts=True)[1]))
    if height > plan.component_row_capacity:
        raise ValueError(
            f"A constraint component has {height} rows; component_row_capacity is "
            f"{plan.component_row_capacity}."
        )
    owners = ownership.owner_count
    if -(-count // owners) > plan.row_capacity:
        raise ValueError("Constraint rows exceed owners * row_capacity.")
    total = owners * plan.row_capacity
    row = np.arange(count)
    slots = (row % owners) * plan.row_capacity + row // owners
    ids = np.zeros((total,), np.int64)
    atoms = np.full((total, 2), -1, np.int64)
    anchors = np.full((total,), -1, np.int64)
    lengths = np.zeros((total,), targets.dtype)
    active = np.zeros((total,), np.bool_)
    ids[slots] = row
    atoms[slots] = members
    anchors[slots] = components
    lengths[slots] = targets
    active[slots] = True
    unused = np.setdiff1d(np.arange(total), slots)
    ids[unused] = count + np.arange(unused.size)
    zero = np.zeros((), targets.dtype)
    return AtomisticDomainConstraintRows(
        row_ids=ids,
        atoms=atoms,
        components=anchors,
        targets=lengths,
        lagrange=np.zeros_like(lengths),
        active=active,
        position_residual=zero,
        velocity_residual=zero,
        topology_epoch=np.zeros((), np.int32),
        row_capacity=plan.row_capacity,
        row_count=count,
        halo_capacity=plan.halo_capacity,
        directory_capacity=plan.directory_capacity,
        migration_capacity=plan.migration_capacity,
        plan_id=plan.plan_id,
    )


@final
class AtomisticDomainConstraintEvidence(StrictModule):
    """Directory, migration, halo and component-closure evidence of one binding.

    ``reasons`` separates capacity refusal (``ROUTE_CAPACITY``: directory,
    row packet/receive or constraint-halo overflow) from scientific topology
    failure (``CONSTRAINT``: missing anchor/member, member conflict across
    components, a component split over owners) and stale owners (``DOMAIN``).
    """

    __strict_contract__ = True

    committed: Bool[Scalar]
    maximum_directory_load: Int32[Scalar]
    directory_capacity: Int32[Scalar]
    missing_anchors: Int32[Scalar]
    missing_members: Int32[Scalar]
    component_conflicts: Int32[Scalar]
    split_members: Int32[Scalar]
    migration: DistributedMigrationEvidence
    halo: DistributedHaloEvidence
    reasons: Int32[Scalar]


@final
class AtomisticDomainConstraintTransition(StrictModule):
    """Candidate constraint state of one binding; the caller commits globally."""

    __strict_contract__ = True

    state: AtomisticDomainConstraintState
    committed: Bool[Scalar]
    evidence: AtomisticDomainConstraintEvidence


def _lookup(
    layout: DistributedPointLayout,
    queries: Array,
    query_valid: Array,
    capacity: int,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    plan = layout.plan
    axis = plan.axis_name
    row = _spec(axis, 1)

    def body(
        ids: Array, active: Array, asked: Array, valid: Array
    ) -> tuple[Array, Array, Array, Array, Array]:
        return owner_directory_lookup(
            ids,
            active,
            asked,
            valid,
            owners=plan.owner_count,
            capacity=capacity,
            axis=axis,
        )

    return plan.map(
        body, (row, row, row, row), (row, row, row, _REPLICATED, _REPLICATED)
    )(layout.stable_ids, layout.active, queries, query_valid)


def _owner_closure(
    halo: DistributedHaloPlan,
    route_columns: Array,
    route_valid: Array,
    row_active: Array,
    components: Array,
    send_slots: Array,
    send_valid: Array,
    /,
) -> tuple[Array, Array]:
    """Component closure inside one owner region.

    Rows sharing a member column must belong to one component, and every
    member atom must be touched by rows of exactly one owner (counted
    through the halo transpose), so each component lives on one owner.
    """
    provider = JaxCollectiveProvider(halo.ownership.axis_name)
    columns = halo.column_count
    left = route_columns[0::2]
    right = route_columns[1::2]
    ok = row_active & route_valid[0::2] & route_valid[1::2]
    endpoint_column = jnp.stack((left, right), axis=1).reshape((-1,))
    endpoint_component = jnp.repeat(components, 2)
    endpoint_ok = jnp.repeat(ok, 2)
    key = jnp.where(endpoint_ok, endpoint_column, columns)
    order = jnp.argsort(key, stable=True)
    ordered = key[order]
    ordered_component = endpoint_component[order]
    conflicts = jnp.sum(
        (ordered[1:] == ordered[:-1])
        & (ordered[1:] < columns)
        & (ordered_component[1:] != ordered_component[:-1]),
        dtype=jnp.int32,
    ) + jnp.sum(ok & (left == right), dtype=jnp.int32)
    touched = (
        jnp.zeros((columns,), dtype=jnp.int32)
        .at[jnp.where(endpoint_ok, endpoint_column, columns)]
        .max(1, mode="drop")
    )
    counts = halo.local_transpose(touched, send_slots, send_valid)
    splits = jnp.sum(counts > 1, dtype=jnp.int32)
    return provider.sum(conflicts), provider.sum(splits)


def _row_ownership(
    atoms: DistributedOwnershipPlan, row_capacity: int, /
) -> DistributedOwnershipPlan:
    return DistributedOwnershipPlan(
        atoms.address_plan,
        atoms.execution_group,
        row_capacity,
        owner_lanes=atoms.owner_count if atoms.lane_reference else None,
    )


def _bind_constraints(
    rows: DistributedPointLayout,
    atoms: Array,
    components: Array,
    targets: Array,
    lagrange: Array,
    layout: DistributedPointLayout,
    /,
    *,
    position_residual: Array,
    velocity_residual: Array,
    topology_epoch: Array,
    halo_capacity: int,
    directory_capacity: int,
    migration_capacity: int,
    plan_id: str,
) -> AtomisticDomainConstraintTransition:
    """Anchor-owner migration, member routing and closure of constraint rows."""
    atom_plan = layout.plan
    if not atom_plan.compatible(rows.plan) or rows.plan.address_plan.plan_id != (
        atom_plan.address_plan.plan_id
    ):
        raise ValueError(
            "Constraint rows and atoms must share one owner mesh and address plan."
        )
    anchor_owner, _, anchor_found, anchor_load, anchor_missing = _lookup(
        layout, components, rows.active, directory_capacity
    )
    destinations = jnp.where(rows.active & anchor_found, anchor_owner, -1).astype(
        jnp.int32
    )
    migrated = rows.migrate(
        destinations,
        packet_capacity=migration_capacity,
        payload={
            "atoms": atoms,
            "components": components,
            "targets": targets,
            "lagrange": lagrange,
        },
    )
    new_rows = migrated.layout
    moved = migrated.payload
    endpoints = moved["atoms"].reshape((-1,))
    endpoint_valid = jnp.repeat(new_rows.active, 2)
    member_owner, member_slot, member_found, member_load, member_missing = _lookup(
        layout, endpoints, endpoint_valid, directory_capacity
    )
    routed = endpoint_valid & member_found
    halo = DistributedHaloPlan(
        atom_plan,
        jnp.where(routed, member_owner, 0),
        jnp.where(routed, member_slot, 0),
        routed,
        halo_capacity=halo_capacity,
    )
    axis = atom_plan.axis_name
    row = _spec(axis, 1)
    conflicts, splits = atom_plan.map(
        lambda *values: _owner_closure(halo, *values),
        (row, row, row, row, _spec(axis, 2), _spec(axis, 2)),
        (_REPLICATED, _REPLICATED),
    )(
        halo.route_columns,
        halo.route_valid,
        new_rows.active,
        moved["components"],
        halo.send_slots,
        halo.send_valid,
    )
    load = jnp.maximum(anchor_load, member_load)
    directory = jnp.asarray(directory_capacity, dtype=jnp.int32)
    directory_overflow = load > directory
    migration = migrated.evidence
    migration_overflow = (migration.maximum_packet > migration.packet_capacity) | (
        migration.maximum_received > migration.local_capacity
    )
    capacity_failure = directory_overflow | migration_overflow | ~halo.evidence.successful
    # The directory owner qualifies absence by complete registration. A separate
    # query overflow must not erase an already proven missing-ID failure.
    missing_anchors = anchor_missing
    missing_members = member_missing
    scientific_failure = (
        (missing_anchors > 0) | (missing_members > 0) | (conflicts > 0) | (splits > 0)
    )
    reasons = (
        _flag(capacity_failure, _CAPACITY)
        | _flag(scientific_failure, _CONSTRAINT)
        | _flag(~migration.epochs_consistent, int(AtomisticStepRejectionReason.DOMAIN))
    )
    committed = migration.committed & (reasons == 0)
    state = AtomisticDomainConstraintState(
        rows=new_rows,
        atoms=moved["atoms"],
        components=moved["components"],
        targets=moved["targets"],
        lagrange=moved["lagrange"],
        halo=halo,
        position_residual=position_residual,
        velocity_residual=velocity_residual,
        topology_epoch=(topology_epoch + 1).astype(jnp.int32),
        successful=committed,
        directory_capacity=directory_capacity,
        migration_capacity=migration_capacity,
        plan_id=plan_id,
    )
    evidence = AtomisticDomainConstraintEvidence(
        committed=committed,
        maximum_directory_load=load,
        directory_capacity=directory,
        missing_anchors=missing_anchors.astype(jnp.int32),
        missing_members=missing_members.astype(jnp.int32),
        component_conflicts=conflicts,
        split_members=splits,
        migration=migration,
        halo=halo.evidence,
        reasons=reasons,
    )
    return AtomisticDomainConstraintTransition(state, committed, evidence)


def rebuild_domain_constraints(
    constraints: AtomisticDomainConstraintState,
    layout: DistributedPointLayout,
    /,
) -> AtomisticDomainConstraintTransition:
    """Migrate constraint rows with their anchors and re-route members (traceable).

    Called by the domain rebuild after the atom migration, on the migrated
    layout.  Returns a candidate; the caller ANDs ``committed`` into its global
    commit and selects the accepted constraint state otherwise.
    """
    if not isinstance(constraints, AtomisticDomainConstraintState):
        raise TypeError("constraints must be an AtomisticDomainConstraintState.")
    if not isinstance(layout, DistributedPointLayout):
        raise TypeError("layout must be a DistributedPointLayout.")
    return _bind_constraints(
        constraints.rows,
        constraints.atoms,
        constraints.components,
        constraints.targets,
        constraints.lagrange,
        layout,
        position_residual=constraints.position_residual,
        velocity_residual=constraints.velocity_residual,
        topology_epoch=constraints.topology_epoch,
        halo_capacity=constraints.halo.halo_capacity,
        directory_capacity=constraints.directory_capacity,
        migration_capacity=constraints.migration_capacity,
        plan_id=constraints.plan_id,
    )


def _process_rows(value: Array, /) -> np.ndarray:
    shards = sorted(value.addressable_shards, key=lambda shard: shard.index[0].start or 0)
    starts = []
    blocks = []
    for shard in shards:
        start = shard.index[0].start or 0
        if start in starts:
            continue
        starts.append(start)
        blocks.append(np.asarray(shard.data))
    return np.concatenate(blocks, axis=0)


def _replicated_host(value: Array, /) -> np.ndarray:
    return np.asarray(value.addressable_shards[0].data)


def export_domain_constraints(
    constraints: AtomisticDomainConstraintState, /
) -> AtomisticDomainConstraintRows:
    """This process's addressable constraint row blocks (explicit host egress)."""
    if not isinstance(constraints, AtomisticDomainConstraintState):
        raise TypeError("constraints must be an AtomisticDomainConstraintState.")
    rows = constraints.rows
    return AtomisticDomainConstraintRows(
        row_ids=_process_rows(rows.stable_ids),
        atoms=_process_rows(constraints.atoms),
        components=_process_rows(constraints.components),
        targets=_process_rows(constraints.targets),
        lagrange=_process_rows(constraints.lagrange),
        active=_process_rows(rows.active),
        position_residual=_replicated_host(constraints.position_residual),
        velocity_residual=_replicated_host(constraints.velocity_residual),
        topology_epoch=_replicated_host(constraints.topology_epoch),
        row_capacity=rows.plan.local_capacity,
        row_count=rows.logical_count,
        halo_capacity=constraints.halo.halo_capacity,
        directory_capacity=constraints.directory_capacity,
        migration_capacity=constraints.migration_capacity,
        plan_id=constraints.plan_id,
    )


def import_domain_constraints(
    rows: AtomisticDomainConstraintRows,
    layout: DistributedPointLayout,
    /,
) -> AtomisticDomainConstraintTransition:
    """Assemble process-local rows on device and bind them to ``layout``.

    Eager ingress/restore boundary (the row-ID audit of ``from_blocks`` is
    concrete); rows reach their anchor owners through the same directory,
    migration and constraint-halo binding as a rebuild.
    """
    if not isinstance(rows, AtomisticDomainConstraintRows):
        raise TypeError("rows must be AtomisticDomainConstraintRows.")
    if not isinstance(layout, DistributedPointLayout):
        raise TypeError("layout must be a DistributedPointLayout.")
    plan = _row_ownership(layout.plan, rows.row_capacity)
    local_owners = sum(
        1 for process in plan.owner_processes if process == jax.process_index()
    )
    if rows.row_ids.shape[0] != local_owners * rows.row_capacity:
        raise ValueError("Process-local rows must cover every owner of this process.")
    total = plan.total_capacity
    dtype = np.dtype(layout.points.dtype)

    def assemble(value: np.ndarray) -> Array:
        return jax.make_array_from_process_local_data(
            plan.sharding(value.ndim), value, (total,) + value.shape[1:]
        )

    points = np.zeros((rows.row_ids.shape[0], plan.address_plan.dimension), dtype)
    row_layout = DistributedPointLayout.from_blocks(
        plan,
        assemble(points),
        active=assemble(rows.active),
        stable_ids=assemble(rows.row_ids),
        logical_indices=assemble(
            np.where(rows.active, rows.row_ids, -1).astype(np.int32)
        ),
        logical_count=rows.row_count,
    )
    if not bool(row_layout.stable_ids_unique):
        raise ValueError("Constraint row IDs are not globally unique.")
    return _bind_constraints(
        row_layout,
        assemble(rows.atoms),
        assemble(rows.components),
        assemble(rows.targets.astype(dtype)),
        assemble(rows.lagrange.astype(dtype)),
        layout,
        position_residual=jnp.asarray(rows.position_residual, dtype=dtype),
        velocity_residual=jnp.asarray(rows.velocity_residual, dtype=dtype),
        topology_epoch=jnp.asarray(rows.topology_epoch, dtype=jnp.int32),
        halo_capacity=rows.halo_capacity,
        directory_capacity=rows.directory_capacity,
        migration_capacity=rows.migration_capacity,
        plan_id=rows.plan_id,
    )


# --------------------------------------------------------------------------
# Distributed SHAKE/RATTLE
# --------------------------------------------------------------------------


@final
class AtomisticDomainConstraintSolve(StrictModule):
    """Globally reduced result of one distributed constraint projection.

    ``positions`` are unwrapped Cartesian rows (owner-blocked like atoms);
    ``capacity_successful`` is false when a component exceeds the declared
    rank-Jacobian rows or members (a capacity refusal, not a rank failure).
    """

    __strict_contract__ = True

    positions: Float[_AtomRowDim, Literal[3]]
    momenta: Float[_AtomRowDim, Literal[3]]
    lagrange: Float[_ConstraintRowDim]
    position_residual: Float[Scalar]
    velocity_residual: Float[Scalar]
    regular: Bool[Scalar]
    minimum_relative_singular_value: Float[Scalar]
    capacity_successful: Bool[Scalar]
    finite: Bool[Scalar]
    fixed_members: Int32[Scalar]
    successful: Bool[Scalar]
    reasons: Int32[Scalar]


def _component_rank(
    positions: Array,
    weights: Array,
    left: Array,
    right: Array,
    ok: Array,
    components: Array,
    /,
    *,
    height: int,
    width: int,
    rank_tolerance: float,
) -> tuple[Array, Array, Array]:
    """P5 per-component mass-weighted Jacobian rank of this owner's components.

    Components group rows by anchor ID; members are the distinct gathered
    columns of a component (columns are unique per owner).  Returns
    (regular, smallest relative singular value, capacity admitted).
    """
    rows = ok.shape[0]
    columns = positions.shape[0]
    key = jnp.where(ok, components.astype(jnp.int64), _INT64_MAX)
    order = jnp.argsort(key, stable=True)
    ordered = key[order]
    present = ordered != _INT64_MAX
    head = jnp.concatenate((present[:1], (ordered[1:] != ordered[:-1]) & present[1:]))
    slot_sorted = (jnp.cumsum(head, dtype=jnp.int32) - 1).astype(jnp.int32)
    start = jnp.searchsorted(ordered, ordered, side="left").astype(jnp.int32)
    position_sorted = jnp.arange(rows, dtype=jnp.int32) - start
    component_slot = jnp.zeros((rows,), jnp.int32).at[order].set(slot_sorted)
    row_position = jnp.zeros((rows,), jnp.int32).at[order].set(position_sorted)

    endpoint_slot = jnp.concatenate((component_slot, component_slot)).astype(jnp.int64)
    endpoint_column = jnp.concatenate((left, right)).astype(jnp.int64)
    endpoint_ok = jnp.concatenate((ok, ok))
    endpoint_key = jnp.where(
        endpoint_ok, endpoint_slot * (columns + 1) + endpoint_column, _INT64_MAX
    )
    endpoint_order = jnp.argsort(endpoint_key, stable=True)
    sorted_key = endpoint_key[endpoint_order]
    valid = sorted_key != _INT64_MAX
    unique_head = jnp.concatenate(
        (valid[:1], (sorted_key[1:] != sorted_key[:-1]) & valid[1:])
    )
    unique_index = jnp.cumsum(unique_head, dtype=jnp.int32) - 1
    sorted_component = jnp.where(valid, sorted_key // (columns + 1), _INT64_MAX)
    component_start = jnp.searchsorted(sorted_component, sorted_component, side="left")
    member_sorted = unique_index - unique_index[component_start]
    member = jnp.zeros((2 * rows,), jnp.int32).at[endpoint_order].set(member_sorted)
    member_left = member[:rows]
    member_right = member[rows:]
    capacity = jnp.all(
        ~ok | ((row_position < height) & (member_left < width) & (member_right < width))
    )

    displacement = positions[left] - positions[right]
    length = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
    direction = jnp.where(
        (length > 0.0)[:, None],
        displacement / jnp.where(length > 0.0, length, 1.0)[:, None],
        0.0,
    )
    direction = jnp.where(ok[:, None], direction, 0.0)
    target = jnp.where(
        ok & (row_position < height) & (member_left < width) & (member_right < width),
        component_slot,
        rows,
    )
    row_index = jnp.clip(row_position, 0, height - 1)
    axes = jnp.arange(3, dtype=jnp.int32)
    left_index = 3 * jnp.clip(member_left, 0, width - 1)[:, None] + axes
    right_index = 3 * jnp.clip(member_right, 0, width - 1)[:, None] + axes
    jacobian = jnp.zeros((rows, height, 3 * width), dtype=positions.dtype)
    jacobian = jacobian.at[target[:, None], row_index[:, None], left_index].add(
        weights[left][:, None] * direction, mode="drop"
    )
    jacobian = jacobian.at[target[:, None], row_index[:, None], right_index].add(
        -weights[right][:, None] * direction, mode="drop"
    )
    _, singular, _ = economy_svd(jacobian)
    largest = jnp.max(singular, axis=-1, keepdims=True)
    relative = singular / jnp.where(largest > 0.0, largest, 1.0)
    rank = jnp.sum(relative > rank_tolerance, axis=-1).astype(jnp.int32)
    required = jax.ops.segment_sum(
        ok.astype(jnp.int32), jnp.where(ok, component_slot, 0), rows
    )
    padded = jnp.arange(singular.shape[-1])[None, :] < required[:, None]
    smallest = jnp.min(jnp.where(padded, relative, jnp.inf))
    regular = jnp.all(rank == required) & jnp.all(jnp.isfinite(singular))
    return regular, smallest, capacity


def _owner_constraint_solve(
    method: DistanceConstraintPlan,
    halo: DistributedHaloPlan,
    position: Array,
    momentum: Array,
    inverse: Array,
    active: Array,
    mobile: Array,
    row_active: Array,
    targets: Array,
    components: Array,
    route_columns: Array,
    route_valid: Array,
    send_slots: Array,
    send_valid: Array,
    /,
    *,
    positions_mode: bool,
    component_row_capacity: int,
) -> tuple[Array, ...]:
    """P5 Jacobi SHAKE (+ RATTLE) of every component this owner anchors."""
    provider = JaxCollectiveProvider(halo.ownership.axis_name)
    dtype = position.dtype
    packed = jnp.concatenate(
        (position, momentum, inverse[:, None], mobile[:, None].astype(dtype)), axis=1
    )
    columns = halo.local_gather(packed, send_slots, send_valid)
    initial_position = columns[:, :3]
    initial_momentum = columns[:, 3:6]
    weight = columns[:, 6]
    member_mobile = columns[:, 7] > 0.0
    left = route_columns[0::2]
    right = route_columns[1::2]
    ok = row_active & route_valid[0::2] & route_valid[1::2]
    # P5 admits distance constraints between mobile particles only.
    fixed_members = provider.sum(
        jnp.sum(ok & ~(member_mobile[left] & member_mobile[right]), dtype=jnp.int32)
    )
    target = jnp.where(ok, targets, 0.0).astype(dtype)

    def shake(_: int | Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        value, multipliers = carry
        displacement = value[left] - value[right]
        squared = jnp.sum(displacement * displacement, axis=-1)
        residual = squared - target * target
        denominator = 2.0 * (weight[left] + weight[right]) * squared
        valid = ok & (denominator > 0.0)
        increment = jnp.where(valid, -residual / denominator, 0.0)
        correction = increment[:, None] * displacement
        delta = jnp.zeros_like(value)
        delta = delta.at[left].add(weight[left, None] * correction)
        delta = delta.at[right].add(-weight[right, None] * correction)
        return value + delta, multipliers + increment

    if positions_mode:
        projected, multipliers = jax.lax.fori_loop(
            0,
            method.maximum_iterations,
            shake,
            (initial_position, jnp.zeros(ok.shape, dtype=dtype)),
        )
        displacement = projected[left] - projected[right]
        distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
        position_residual = provider.maximum(
            jnp.max(jnp.where(ok, jnp.abs(distance - target), 0.0), initial=0.0)
        )
    else:
        projected = initial_position
        multipliers = jnp.zeros(ok.shape, dtype=dtype)
        position_residual = jnp.zeros((), dtype=dtype)

    displacement = projected[left] - projected[right]
    squared = jnp.sum(displacement * displacement, axis=-1)
    denominator = (weight[left] + weight[right]) * squared

    def rattle(_: int | Array, value: Array) -> Array:
        relative = value[left] * weight[left, None] - value[right] * weight[right, None]
        valid = ok & (denominator > 0.0)
        increment = jnp.where(
            valid, -jnp.sum(displacement * relative, axis=-1) / denominator, 0.0
        )
        correction = increment[:, None] * displacement
        delta = jnp.zeros_like(value)
        delta = delta.at[left].add(correction)
        delta = delta.at[right].add(-correction)
        return value + delta

    projected_momentum = jax.lax.fori_loop(
        0, method.maximum_iterations, rattle, initial_momentum
    )
    relative = (
        projected_momentum[left] * weight[left, None]
        - projected_momentum[right] * weight[right, None]
    )
    velocity_residual = provider.maximum(
        jnp.max(
            jnp.where(ok, jnp.abs(jnp.sum(displacement * relative, axis=-1)), 0.0),
            initial=0.0,
        )
    )
    column_count = columns.shape[0]
    touched = (
        jnp.zeros((column_count,), jnp.int32)
        .at[left]
        .add(ok.astype(jnp.int32))
        .at[right]
        .add(ok.astype(jnp.int32))
    ) > 0
    contributions = jnp.where(
        touched[:, None],
        jnp.concatenate((projected, projected_momentum), axis=1),
        0.0,
    )
    returned = halo.local_transpose(contributions, send_slots, send_valid)
    holders = halo.local_transpose(touched.astype(jnp.int32), send_slots, send_valid)
    moved = (holders > 0)[:, None]
    new_position = jnp.where(moved, returned[:, :3], position)
    new_momentum = jnp.where(
        active[:, None], jnp.where(moved, returned[:, 3:], momentum), 0.0
    )
    finite = (
        provider.minimum(
            jnp.all(
                ~active[:, None]
                | (jnp.isfinite(new_position) & jnp.isfinite(new_momentum))
            ).astype(jnp.int32)
        )
        == 1
    )
    if positions_mode:
        regular, smallest, capacity = _component_rank(
            projected,
            jnp.sqrt(weight),
            left,
            right,
            ok,
            components,
            height=component_row_capacity,
            width=method.maximum_component_atoms,
            rank_tolerance=method.rank_tolerance,
        )
        regular = provider.minimum(regular.astype(jnp.int32)) == 1
        capacity = provider.minimum(capacity.astype(jnp.int32)) == 1
        smallest = provider.minimum(smallest)
    else:
        regular = jnp.asarray(True)
        capacity = jnp.asarray(True)
        smallest = jnp.asarray(jnp.inf, dtype=dtype)
    return (
        new_position,
        new_momentum,
        jnp.where(row_active, multipliers, 0.0),
        position_residual,
        velocity_residual,
        regular,
        smallest,
        capacity,
        finite,
        fixed_members,
    )


def solve_domain_constraints(
    method: DistanceConstraintPlan,
    constraints: AtomisticDomainConstraintState,
    layout: DistributedPointLayout,
    unwrapped_positions: Array,
    momenta: Array,
    masses: Array,
    /,
    *,
    mobile: Array,
    positions_mode: bool,
    component_row_capacity: int,
) -> AtomisticDomainConstraintSolve:
    """Distributed P5 SHAKE/RATTLE (``positions_mode``) or RATTLE projection.

    Owner-blocked unwrapped positions and momenta are gathered once through
    the constraint halo; each component is iterated on its anchor owner with
    P5's fixed-count Jacobi updates and every member result returns to its
    owner exactly once.  Status is global.
    """
    plan = layout.plan
    axis = plan.axis_name
    halo = constraints.halo
    active = layout.active
    inverse = _inverse_masses(masses, mobile)
    row = _spec(axis, 1)
    point = _spec(axis, 2)

    def body(*values: Array) -> tuple[Array, ...]:
        return _owner_constraint_solve(
            method,
            halo,
            *values,
            positions_mode=positions_mode,
            component_row_capacity=component_row_capacity,
        )

    (
        positions,
        projected,
        lagrange,
        position_residual,
        velocity_residual,
        regular,
        smallest,
        capacity,
        finite,
        fixed_members,
    ) = plan.map(
        body,
        (point, point, row, row, row, row, row, row, row, row, point, point),
        (point, point, row) + (_REPLICATED,) * 7,
    )(
        unwrapped_positions,
        momenta,
        inverse,
        active,
        mobile,
        constraints.rows.active,
        constraints.targets,
        constraints.components,
        halo.route_columns,
        halo.route_valid,
        halo.send_slots,
        halo.send_valid,
    )
    tolerance = method.tolerance
    scientific = (
        finite
        & (position_residual <= tolerance)
        & (velocity_residual <= tolerance)
        & regular
        & (fixed_members == 0)
    )
    topology = constraints.successful
    reasons = _flag(~capacity, _CAPACITY) | _flag(~scientific | ~topology, _CONSTRAINT)
    return AtomisticDomainConstraintSolve(
        positions=positions,
        momenta=projected,
        lagrange=lagrange if positions_mode else constraints.lagrange,
        position_residual=position_residual,
        velocity_residual=velocity_residual,
        regular=regular,
        minimum_relative_singular_value=smallest,
        capacity_successful=capacity,
        finite=finite,
        fixed_members=fixed_members,
        successful=scientific & capacity & topology,
        reasons=reasons,
    )


# --------------------------------------------------------------------------
# Dynamics plan, reductions, step
# --------------------------------------------------------------------------


class _Admission(NamedTuple):
    """The method facts P5 ``_admit_method`` reads from a dynamics plan."""

    integrator: AtomisticDomainIntegratorPlan
    constraints: AtomisticDomainConstraintPlan | None


@final
class AtomisticDomainDynamicsPlan(StrictModule, NonTrainableState):
    """Distributed P5 method over one domain plan.

    Every P5 integrator executes with its P5 schedule and P5 ensemble
    admission (``_admit_method``): velocity Verlet, BAOAB, Nose-Hoover chains,
    CSVR, deterministic rescaling, generalized Langevin (per-atom memory is
    owner-blocked atom payload that migrates with its atom) and MTK NPT/NPH.
    ``barostat`` adds P5's scheduled isotropic Monte Carlo volume move with
    the P5 ``_admit`` rules.  Refusals are exactly the P5 ones plus the domain
    precondition that a changing cell needs a ``dynamic_cell`` domain plan.

    ``degrees_of_freedom`` (int32 scalar, ``3 mobile - constraints - 3
    [center of mass]``) is dynamic: it follows the material's mobile count
    and the P5 kernels take it as an array.  ``plan_identity`` is the exact
    canonical dynamics-plan digest (``plan_id`` decodes it on the host);
    ``strain_response`` is a static method selector.
    """

    domain: AtomisticDomainPlan
    integrator: AtomisticDomainIntegratorPlan
    ensemble: AtomisticDomainEnsemblePlan
    constraints: AtomisticDomainConstraintPlan | None
    barostat: IsotropicMonteCarloBarostatPlan | None
    schedule: AtomisticSplittingPlan
    degrees_of_freedom: Array
    plan_identity: IdentityWords
    strain_response: bool = eqx.field(static=True)

    def __init__(
        self,
        domain: AtomisticDomainPlan,
        integrator: AtomisticDomainIntegratorPlan,
        ensemble: AtomisticDomainEnsemblePlan,
        /,
        *,
        constraints: AtomisticDomainConstraintPlan | None = None,
        barostat: IsotropicMonteCarloBarostatPlan | None = None,
        strain_response: bool | None = None,
    ) -> None:
        if not isinstance(domain, AtomisticDomainPlan):
            raise TypeError("domain must be an AtomisticDomainPlan.")
        if not isinstance(integrator, _DOMAIN_INTEGRATORS):
            raise TypeError(f"Unsupported dynamics method {type(integrator).__name__}.")
        if not isinstance(ensemble, AtomisticDomainEnsemblePlan):
            raise TypeError("ensemble must be an AtomisticDomainEnsemblePlan.")
        if constraints is not None and not isinstance(
            constraints, AtomisticDomainConstraintPlan
        ):
            raise TypeError(
                "constraints must be an AtomisticDomainConstraintPlan or None."
            )
        if barostat is not None and not isinstance(
            barostat, IsotropicMonteCarloBarostatPlan
        ):
            raise TypeError(
                "barostat must be an IsotropicMonteCarloBarostatPlan or None."
            )
        # P5: AtomisticDynamicsPlan refuses constrained MTK and _admit_method
        # admits ensembles per method, refuses non-atomic MTK scaling and a
        # center-of-mass restriction under per-particle stochastic thermostats.
        _admit_method(_Admission(integrator, constraints), (ensemble,), ensemble)
        cell = domain.partition.cell
        full_cell = cell.fully_periodic and cell.rank == 3 and cell.ambient_dimension == 3
        cell_method = isinstance(integrator, _CELL_METHODS)
        if cell_method:
            if not full_cell or not domain.dynamic_cell:
                raise ValueError(
                    "Distributed MTK cell dynamics requires a fully periodic 3x3 cell "
                    "and a domain plan declaring dynamic_cell."
                )
            if strain_response is False:
                raise ValueError("Cell dynamics requires the strain response.")
        if barostat is not None:
            # P5 _barostat._admit, plus the dynamic-cell domain precondition.
            if not cell.fully_periodic:
                raise ValueError("Isotropic barostat requires a fully periodic cell.")
            if cell_method:
                raise ValueError(
                    "The integrator already owns the cell; a thermodynamic run has "
                    "exactly one cell-evolution owner."
                )
            if not integrator.canonical:
                raise ValueError(
                    "A Monte Carlo NPT move requires a canonical NVT particle method."
                )
            if (
                constraints is not None
                and ensemble.volume_coordinate_convention == "atomic"
            ):
                raise ValueError(
                    "Atomic volume scaling would deform holonomic constraints; declare "
                    "molecular-center scaling."
                )
            if ensemble.ensemble != "npt":
                raise ValueError("A Monte Carlo volume move samples an NPT state.")
            if not domain.dynamic_cell:
                raise ValueError(
                    "A Monte Carlo volume move requires a dynamic_cell domain."
                )
        requested = cell_method if strain_response is None else bool(strain_response)
        if requested and not full_cell:
            raise ValueError("The strain response requires a fully periodic 3x3 cell.")
        rows = 0 if constraints is None else constraints.constraint_count
        dof = (
            3 * int(jax.device_get(ensemble.mobile_count))
            - rows
            - (3 if ensemble.momentum_restriction == "center-of-mass" else 0)
        )
        if dof <= 0:
            raise ValueError(
                "The admitted phase space has no kinetic degrees of freedom."
            )
        schedule = _schedule(integrator, constraints is not None)
        self.domain = domain
        self.integrator = integrator
        self.ensemble = ensemble
        self.constraints = constraints
        self.barostat = barostat
        self.schedule = schedule
        self.degrees_of_freedom = jnp.asarray(dof, dtype=jnp.int32)
        self.strain_response = requested
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-domain-dynamics",
                "domain": domain.plan_id,
                "integrator": integrator.plan_id,
                "ensemble": ensemble.plan_id,
                "constraints": None if constraints is None else constraints.plan_id,
                "barostat": None if barostat is None else barostat.plan_id,
                "schedule": schedule.plan_id,
                "strain_response": requested,
            }
        )

    @property
    def plan_id(self) -> str:
        """Host decoding of the canonical dynamics-plan digest."""
        return self.plan_identity.fingerprint()

    def atom_payload(
        self,
        /,
        *,
        mobile: ArrayLike | None = None,
        molecules: ArrayLike | None = None,
        dtype: Any = np.float64,
    ) -> dict[str, np.ndarray]:
        """Logical-order atom payload entries this method requires at ingress.

        ``mobile`` (bool per atom) when the ensemble declares fixed atoms,
        ``molecules`` (int labels in ``[0, molecule_count)``) for
        molecular-center scaling and zero GLE memory ``(N, 3, K)`` (P5 initial
        memory) for generalized Langevin.  Pass the result as
        ``atom_payload=`` to the domain ingress (host).
        """
        ensemble = self.ensemble
        count = int(jax.device_get(ensemble.atom_count))
        mobile_count = int(jax.device_get(ensemble.mobile_count))
        payload: dict[str, np.ndarray] = {}
        if ensemble.fixed_atoms:
            if mobile is None:
                raise ValueError("The ensemble declares fixed atoms; supply mobile.")
            flags = np.asarray(mobile, dtype=np.bool_)
            if flags.shape != (count,) or int(np.count_nonzero(flags)) != mobile_count:
                raise ValueError(
                    "mobile must flag exactly mobile_count of atom_count atoms."
                )
            payload[MOBILE_PAYLOAD] = flags
        elif mobile is not None:
            raise ValueError("The ensemble declares every atom mobile.")
        if ensemble.molecule_count is not None:
            if molecules is None:
                raise ValueError("Molecular-center scaling requires molecule labels.")
            labels = np.asarray(molecules)
            if (
                labels.shape != (count,)
                or not np.issubdtype(labels.dtype, np.integer)
                or np.any(labels < 0)
                or np.any(labels >= ensemble.molecule_count)
            ):
                raise ValueError(
                    "molecules must label every atom in [0, molecule_count)."
                )
            payload[MOLECULE_PAYLOAD] = labels.astype(np.int32)
        elif molecules is not None:
            raise ValueError("Atomic volume scaling uses no molecule labels.")
        if isinstance(self.integrator, GeneralizedLangevinPlan):
            payload[GLE_MEMORY_PAYLOAD] = np.zeros(
                (count, 3, self.integrator.auxiliary_count), dtype=dtype
            )
        return payload

    def prepare(
        self, runtime: PreparedAtomisticDomainRuntime, /
    ) -> PreparedAtomisticDomainDynamics:
        return PreparedAtomisticDomainDynamics(self, runtime)


@final
class AtomisticDomainDynamicsReductions(StrictModule):
    """Collective scientific reductions of one domain state (over mobile atoms)."""

    __strict_contract__ = True

    kinetic_energy: Float[Scalar]
    twice_kinetic_tensor: Float[Literal[3], Literal[3]]
    linear_momentum: Float[Literal[3]]
    total_mass: Float[Scalar]
    atom_count: Int32[Scalar]
    mobile_count: Int32[Scalar]
    constraint_count: Int32[Scalar]
    degrees_of_freedom: Int32[Scalar]


@final
class AtomisticDomainDynamicsDiagnostics(StrictModule):
    __strict_contract__ = True

    kinetic_energy: Float[Scalar]
    potential_energy: Float[Scalar]
    total_energy: Float[Scalar]
    extended_energy: Float[Scalar]
    temperature: Float[Scalar]
    pressure: Float[Scalar]
    pressure_tensor: Float[Literal[3], Literal[3]]
    volume: Float[Scalar]
    total_linear_momentum: Float[Literal[3]]
    degrees_of_freedom: Int32[Scalar]
    constraint_residual: Float[Scalar]
    energy: AtomisticEnergyLedgerState
    successful: Bool[Scalar]
    rejection_reasons: Int32[Scalar]


@final
class AtomisticDomainStepEvaluation(StrictModule):
    """One transaction: the candidate, the accepted state and their evidence."""

    __strict_contract__ = True

    candidate_state: AtomisticDomainState
    accepted_state: AtomisticDomainState
    diagnostics: AtomisticDomainDynamicsDiagnostics
    successful: Bool[Scalar]
    status: Int32[Scalar]
    residual: Float[Scalar]
    rejection_reasons: Int32[Scalar]


@final
class AtomisticDomainPrimeEvaluation(StrictModule):
    """Force/strain priming without consuming a step, time or random draw."""

    __strict_contract__ = True

    candidate_state: AtomisticDomainState
    accepted_state: AtomisticDomainState
    successful: Bool[Scalar]
    rejection_reasons: Int32[Scalar]


@final
class AtomisticDomainBarostatEvaluation(StrictModule):
    """One Monte Carlo volume move: candidate, successor and acceptance evidence."""

    __strict_contract__ = True

    candidate_state: AtomisticDomainState
    accepted_state: AtomisticDomainState
    accepted: Bool[Scalar]
    log_acceptance_probability: Float[Scalar]
    volume_before: Float[Scalar]
    volume_after: Float[Scalar]
    energy_before: Float[Scalar]
    energy_after: Float[Scalar]
    successful: Bool[Scalar]
    rejection_reasons: Int32[Scalar]


class _Evaluated(NamedTuple):
    domain: AtomisticDomainState
    evaluation: AtomisticDomainEvaluation
    forces: Array
    energy: Array
    strain: Array
    strain_available: Array
    successful: Array
    reasons: Array


@final
class PreparedAtomisticDomainDynamics(StrictModule):
    """A dynamics plan bound to its prepared domain runtime.

    ``prepared_identity`` is the exact canonical preparation digest as dynamic
    `IdentityWords` (``prepared_id`` decodes it on the host). Transitions
    admit states by static layout; a state of another domain plan or model
    revision is refused numerically by the runtime's evaluation
    (``status.identity_bound``), so one executable serves every revision.

    ``scientific_identity`` (host ``scientific_id``) is the layout-independent
    digest of the physics, computed once here from the runtime's domain
    physics (`PreparedAtomisticDomainRuntime.scientific_identity`: model,
    terms, roles, reciprocal definition, cell) and its Hamiltonian numeric
    revision (``hamiltonian_revision_identity``), and the method: integrator,
    ensemble (units, thermodynamic state, counts, restriction, volume
    convention), barostat and schedule, the strain response, and the
    constraint method with its global row count. Constraint capacities, like
    every domain execution choice, only size or place the computation and
    stay bound by ``prepared_identity``.
    """

    plan: AtomisticDomainDynamicsPlan
    runtime: PreparedAtomisticDomainRuntime
    prepared_identity: IdentityWords
    scientific_identity: IdentityWords

    def __init__(
        self,
        plan: AtomisticDomainDynamicsPlan,
        runtime: PreparedAtomisticDomainRuntime,
        /,
    ) -> None:
        from ._domain_runtime import PreparedAtomisticDomainRuntime

        if not isinstance(plan, AtomisticDomainDynamicsPlan):
            raise TypeError("plan must be an AtomisticDomainDynamicsPlan.")
        if not isinstance(runtime, PreparedAtomisticDomainRuntime):
            raise TypeError("runtime must be a PreparedAtomisticDomainRuntime.")
        if runtime.plan.plan_id != plan.domain.plan_id:
            raise ValueError("The domain runtime belongs to another domain plan.")
        self.plan = plan
        self.runtime = runtime
        self.prepared_identity = IdentityWords.of(
            {"kind": "prepared-atomistic-domain-dynamics", "plan": plan.plan_id}
        )
        constraints = plan.constraints
        self.scientific_identity = IdentityWords.of(
            {
                "kind": "atomistic-domain-dynamics-science",
                "domain": runtime.scientific_id,
                "hamiltonian_revision": runtime.hamiltonian_revision_id,
                "integrator": plan.integrator.plan_id,
                "ensemble": plan.ensemble.plan_id,
                "constraints": None
                if constraints is None
                else {
                    "method": constraints.method.plan_id,
                    "constraint_count": constraints.constraint_count,
                },
                "barostat": None if plan.barostat is None else plan.barostat.plan_id,
                "schedule": plan.schedule.plan_id,
                "strain_response": plan.strain_response,
            }
        )

    @property
    def prepared_id(self) -> str:
        """Host decoding of the canonical preparation digest."""
        return self.prepared_identity.fingerprint()

    @property
    def scientific_id(self) -> str:
        """Host decoding of the layout-independent scientific digest."""
        return self.scientific_identity.fingerprint()

    # ------------------------------------------------------------ identity

    @property
    def units(self) -> AtomisticUnitSystem:
        return self.plan.ensemble.units

    def _require_state(
        self, state: AtomisticDomainState, /, *, initialized: bool
    ) -> None:
        if not isinstance(state, AtomisticDomainState):
            raise TypeError("state must be an AtomisticDomainState.")
        constraints = self.plan.constraints
        if (constraints is None) != (state.constraints is None):
            raise ValueError("State constraint rows do not match the dynamics plan.")
        if constraints is not None and state.constraints is not None:
            if state.constraints.plan_id != constraints.plan_id:
                raise ValueError(
                    "State constraint rows belong to another constraint plan."
                )
            if state.constraints.rows.logical_count != constraints.constraint_count:
                raise ValueError(
                    "State constraint rows do not hold constraint_count rows."
                )
        if initialized and (state.controllers is None or state.ledger is None):
            raise ValueError("Initialize the domain state before stepping it.")
        rows = state.positions.shape[0]
        required: dict[str, tuple[tuple[int, ...], str]] = {}
        if self.plan.ensemble.fixed_atoms:
            required[MOBILE_PAYLOAD] = ((rows,), "b")
        if self.plan.ensemble.molecule_count is not None:
            required[MOLECULE_PAYLOAD] = ((rows,), "i")
        if isinstance(self.plan.integrator, GeneralizedLangevinPlan):
            required[GLE_MEMORY_PAYLOAD] = (
                (rows, 3, self.plan.integrator.auxiliary_count),
                "f",
            )
        for name, (shape, kind) in required.items():
            value = state.atom_payload.get(name)
            if value is None or value.shape != shape or value.dtype.kind != kind:
                raise ValueError(
                    f"The method requires owner-blocked atom_payload[{name!r}] of shape "
                    f"{shape}; ingress it with AtomisticDomainDynamicsPlan.atom_payload()."
                )

    # ----------------------------------------------------------- geometry

    def _unwrapped(self, positions: Array, images: Array, vectors: Array, /) -> Array:
        return positions + contract(
            "ni,ij->nj", images.astype(positions.dtype), vectors.astype(positions.dtype)
        )

    def _stored(
        self, unwrapped: Array, images: Array, vectors: Array, active: Array, /
    ) -> Array:
        stored = unwrapped - contract(
            "ni,ij->nj", images.astype(unwrapped.dtype), vectors.astype(unwrapped.dtype)
        )
        return jnp.where(active[:, None], stored, 0.0)

    def _with(
        self,
        domain: AtomisticDomainState,
        positions: Array,
        momenta: Array,
        /,
        **changes: Any,
    ) -> AtomisticDomainState:
        changes.setdefault("step_index", domain.step_index)
        changes.setdefault("time", domain.time)
        return self.runtime.with_dynamics(domain, positions, momenta, **changes)

    def _mobile(self, domain: AtomisticDomainState, /) -> Array:
        """Owner-blocked mobile rows (active, and flagged mobile when declared)."""
        if not self.plan.ensemble.fixed_atoms:
            return domain.layout.active
        return domain.layout.active & domain.atom_payload[MOBILE_PAYLOAD].astype(
            jnp.bool_
        )

    # --------------------------------------------------------- reductions

    def reductions(
        self, state: AtomisticDomainState, /
    ) -> AtomisticDomainDynamicsReductions:
        """Kinetic energy/tensor, momentum, mass and exact DOF counts (collective)."""
        domain = self.plan.domain
        ownership = state.layout.plan
        axis = ownership.axis_name
        reduction = domain.reduction
        factor = self.units.kinetic_to_energy
        restriction = self.plan.ensemble.momentum_restriction
        row = _spec(axis, 1)
        point = _spec(axis, 2)
        constrained = state.constraints is not None

        def body(
            momenta: Array, masses: Array, active: Array, mobile: Array, rows: Array
        ) -> tuple[Array, ...]:
            constraint_rows = rows if constrained else None
            return (
                owner_kinetic_energy(
                    momenta, masses, mobile, factor, reduction=reduction, axis_name=axis
                ),
                owner_kinetic_tensor(
                    momenta, masses, mobile, factor, reduction=reduction, axis_name=axis
                ),
                owner_linear_momentum(
                    momenta, mobile, reduction=reduction, axis_name=axis
                ),
                owner_masked_sum(masses, mobile, reduction=reduction, axis_name=axis),
                owner_count(active, axis_name=axis),
                owner_count(mobile, axis_name=axis),
                jnp.zeros((), jnp.int32)
                if constraint_rows is None
                else owner_count(constraint_rows, axis_name=axis),
                owner_degrees_of_freedom(
                    mobile, constraint_rows, restriction, axis_name=axis
                ),
            )

        rows = (
            state.constraints.rows.active
            if state.constraints is not None
            else jnp.zeros((ownership.owner_count,), jnp.bool_)
        )
        values = ownership.map(body, (point, row, row, row, row), (_REPLICATED,) * 8)(
            state.momenta, state.masses, state.layout.active, self._mobile(state), rows
        )
        return AtomisticDomainDynamicsReductions(*values)

    # ------------------------------------------------------------ evaluation

    def _evaluate(self, domain: AtomisticDomainState, /) -> _Evaluated:
        """Certify (rebuild when needed) and evaluate E/F(/strain) on the domain."""
        runtime = self.runtime
        certificate = runtime.certificate(domain)

        def keep(value: AtomisticDomainState) -> tuple[AtomisticDomainState, Array]:
            return value, jnp.zeros((), jnp.int32)

        def rebuild(value: AtomisticDomainState) -> tuple[AtomisticDomainState, Array]:
            transition = runtime.rebuild(value)
            return transition.state, jnp.asarray(transition.reasons, dtype=jnp.int32)

        domain, transition_reasons = jax.lax.cond(
            certificate.certified, keep, rebuild, domain
        )
        evaluation: AtomisticDomainEvaluation = runtime.evaluate(
            domain, forces=True, stress=self.plan.strain_response
        )
        if self.plan.strain_response and not evaluation.strain_available:
            raise ValueError(
                "The dynamics plan requires the strain response, but the domain "
                "model route does not provide dE/d strain; refusing before stepping."
            )
        if evaluation.strain_available:
            strain = evaluation.strain_derivative
            available = evaluation.successful & jnp.all(jnp.isfinite(strain))
        else:
            strain = jnp.full((3, 3), jnp.nan, dtype=domain.positions.dtype)
            available = jnp.asarray(False)
        reasons = (
            transition_reasons
            | jnp.asarray(evaluation.reasons, dtype=jnp.int32)
            | _flag(~evaluation.successful, int(AtomisticStepRejectionReason.POTENTIAL))
        )
        reasons = reasons | _strain_reasons(self.plan.strain_response, available, reasons)
        return _Evaluated(
            domain=domain,
            evaluation=evaluation,
            forces=evaluation.forces,
            energy=evaluation.energy,
            strain=strain,
            strain_available=available,
            successful=evaluation.successful,
            reasons=reasons,
        )

    def _force_cache(
        self, evaluated: _Evaluated, step: Array, /
    ) -> AtomisticDomainForceCache:
        return evaluated.evaluation.force_cache(step)

    def _force_finite(self, force: AtomisticDomainForceCache, /) -> Array:
        """Finite required cache fields; strain/stress only when requested."""
        finite = tree_allfinite((force.forces, force.energy, force.owner_energies))
        if self.plan.strain_response:
            finite = finite & tree_allfinite((force.strain_derivative, force.stress))
        return finite

    def force_cache_valid(self, state: AtomisticDomainState, /) -> Array:
        force = state.force
        valid = force.successful & (force.step == state.step_index)
        if self.plan.strain_response:
            valid = valid & force.strain_available
        return valid

    # ----------------------------------------------------------- controllers

    def _temperature(self, dtype: Any, /) -> Array:
        value = self.plan.ensemble.temperature
        return jnp.asarray(0.0 if value is None else value, dtype=dtype)

    def _thermal_energy(self, dtype: Any, /) -> Array:
        return self.units.boltzmann_constant * self._temperature(dtype)

    def _barostat_mass(self, dtype: Any, /) -> Array:
        integrator = self.plan.integrator
        if not isinstance(integrator, _CELL_METHODS):
            raise RuntimeError("Only MTK methods own a barostat mass.")
        barostat = integrator.barostat
        temperature = (
            self._temperature(dtype)
            if barostat.inertia_temperature is None
            else jnp.asarray(barostat.inertia_temperature, dtype=dtype)
        )
        return barostat_mass(
            barostat,
            self.plan.degrees_of_freedom,
            self.units.boltzmann_constant * temperature,
        )

    def initial_controllers(
        self, cell_vectors: Array, dtype: Any, /
    ) -> AtomisticControllerState:
        integrator = self.plan.integrator
        particle_chain = None
        cell_chain = None
        cell = None
        if isinstance(integrator, NoseHooverChainPlan):
            particle_chain = initial_nose_hoover_chain_state(integrator.chain, dtype)
        if isinstance(integrator, MTKNPTPlan):
            particle_chain = initial_nose_hoover_chain_state(
                integrator.particle_chain, dtype
            )
            cell_chain = initial_nose_hoover_chain_state(integrator.cell_chain, dtype)
        if isinstance(integrator, _CELL_METHODS):
            cell = initial_cell_controller(integrator.barostat, cell_vectors)
        return AtomisticControllerState(
            particle_chain=particle_chain,
            cell_chain=cell_chain,
            gle=None,
            cell=cell,
            move_counter=jnp.zeros((), dtype=jnp.uint32),
            accepted_moves=jnp.zeros((), dtype=jnp.int32),
            cell_epoch=jnp.zeros((), dtype=jnp.int32),
            rescale_events=jnp.zeros((), dtype=jnp.int32),
        )

    def extended_terms(
        self, controllers: AtomisticControllerState, cell_vectors: Array, dtype: Any, /
    ) -> tuple[Array, Array, Array]:
        """``(K_cell, P_ext V, E_chains)`` exactly as P5 ``extended_terms``."""
        integrator = self.plan.integrator
        dof = self.plan.degrees_of_freedom
        zero = jnp.zeros((), dtype=dtype)
        chains = zero
        cell_kinetic = zero
        pressure_volume = zero
        thermal = self._thermal_energy(dtype)
        if controllers.particle_chain is not None:
            if isinstance(integrator, NoseHooverChainPlan):
                parameters = integrator.chain
            elif isinstance(integrator, MTKNPTPlan):
                parameters = integrator.particle_chain
            else:
                raise RuntimeError(
                    "A particle chain belongs to a Nose-Hoover chain or MTK NPT integrator."
                )
            chains = chains + nose_hoover_chain_energy(
                parameters, controllers.particle_chain, dof, thermal
            )
        if isinstance(integrator, _CELL_METHODS) and controllers.cell is not None:
            mass = self._barostat_mass(dtype)
            cell_kinetic = cell_kinetic_energy(
                integrator.barostat.chart, controllers.cell.momentum, mass
            )
            pressure = jnp.asarray(self.plan.ensemble.pressure, dtype=dtype)
            pressure_volume = pressure * cell_volume(cell_vectors)
            if isinstance(integrator, MTKNPTPlan) and controllers.cell_chain is not None:
                chains = chains + nose_hoover_chain_energy(
                    integrator.cell_chain,
                    controllers.cell_chain,
                    integrator.barostat.chart.dimension,
                    thermal,
                )
        return cell_kinetic, pressure_volume, chains

    def _dof_reasons(self, reductions: AtomisticDomainDynamicsReductions, /) -> Array:
        ensemble = self.plan.ensemble
        mismatch = (
            (reductions.atom_count != ensemble.atom_count)
            | (reductions.mobile_count != ensemble.mobile_count)
            | (reductions.degrees_of_freedom != self.plan.degrees_of_freedom)
        )
        return _flag(mismatch, int(AtomisticStepRejectionReason.THERMODYNAMIC_STATE))

    # -------------------------------------------------------------- thermostats

    @staticmethod
    def _particle_ids(ids: Array, active: Array, /) -> Array:
        # Inactive padding rows take distinct IDs above every local active ID;
        # their draws are masked and never touch an atom's lineage.
        identifiers = ids.astype(jnp.int64)
        top = jnp.max(jnp.where(active, identifiers, -1))
        padding = top + 1 + jnp.arange(identifiers.shape[0], dtype=jnp.int64)
        return jnp.where(active, identifiers, padding)

    def _baoab(
        self, domain: AtomisticDomainState, key_data: Array, /
    ) -> tuple[Array, Array, Array]:
        integrator = self.plan.integrator
        if not isinstance(integrator, BAOABLangevinPlan):
            raise RuntimeError("BAOAB thermostat requested for another method.")
        ownership = domain.layout.plan
        axis = ownership.axis_name
        reduction = self.plan.domain.reduction
        units = self.units
        row = _spec(axis, 1)
        point = _spec(axis, 2)

        def body(
            ids: Array,
            momenta: Array,
            masses: Array,
            active: Array,
            mobile: Array,
            key: Array,
            step: Array,
            temperature: Array,
        ) -> tuple[Array, Array, Array]:
            ornstein = apply_baoab_ornstein_uhlenbeck(
                integrator,
                key,
                self._particle_ids(ids, active),
                momenta,
                jnp.where(active, masses, 1.0),
                mobile,
                step,
                temperature,
                boltzmann_constant=units.boltzmann_constant,
                kinetic_to_energy=units.kinetic_to_energy,
            )
            provider = JaxCollectiveProvider(axis)
            return (
                ornstein.momenta,
                reduction.owner_sum(ornstein.heat, axis),
                provider.minimum(ornstein.successful.astype(jnp.int32)) == 1,
            )

        dtype = domain.momenta.dtype
        return ownership.map(
            body,
            (row, point, row, row, row, _REPLICATED, _REPLICATED, _REPLICATED),
            (point, _REPLICATED, _REPLICATED),
        )(
            domain.layout.stable_ids,
            domain.momenta,
            domain.masses,
            domain.layout.active,
            self._mobile(domain),
            key_data,
            domain.step_index,
            self._temperature(dtype),
        )

    def _generalized_langevin(
        self, domain: AtomisticDomainState, key_data: Array, phase: int, /
    ) -> tuple[Array, Array, Array, Array]:
        """P5 addressed GLE half step on owner-blocked momenta and memory."""
        integrator = self.plan.integrator
        if not isinstance(integrator, GeneralizedLangevinPlan):
            raise RuntimeError("GLE thermostat requested for another method.")
        ownership = domain.layout.plan
        axis = ownership.axis_name
        reduction = self.plan.domain.reduction
        units = self.units
        row = _spec(axis, 1)
        point = _spec(axis, 2)
        memory_spec = _spec(axis, 3)

        def body(
            ids: Array,
            momenta: Array,
            memory: Array,
            masses: Array,
            active: Array,
            mobile: Array,
            key: Array,
            step: Array,
            temperature: Array,
        ) -> tuple[Array, Array, Array, Array]:
            langevin = apply_generalized_langevin(
                integrator,
                key,
                self._particle_ids(ids, active),
                momenta,
                GeneralizedLangevinMemoryState(memory),
                jnp.where(active, masses, 1.0),
                mobile,
                step,
                temperature,
                phase=phase,
                boltzmann_constant=units.boltzmann_constant,
                kinetic_to_energy=units.kinetic_to_energy,
            )
            provider = JaxCollectiveProvider(axis)
            return (
                langevin.momenta,
                langevin.memory.memory,
                reduction.owner_sum(langevin.heat, axis),
                provider.minimum(langevin.successful.astype(jnp.int32)) == 1,
            )

        dtype = domain.momenta.dtype
        return ownership.map(
            body,
            (
                row,
                point,
                memory_spec,
                row,
                row,
                row,
                _REPLICATED,
                _REPLICATED,
                _REPLICATED,
            ),
            (point, memory_spec, _REPLICATED, _REPLICATED),
        )(
            domain.layout.stable_ids,
            domain.momenta,
            domain.atom_payload[GLE_MEMORY_PAYLOAD],
            domain.masses,
            domain.layout.active,
            self._mobile(domain),
            key_data,
            domain.step_index,
            self._temperature(dtype),
        )

    def _kinetic(self, domain: AtomisticDomainState, momenta: Array, /) -> Array:
        ownership = domain.layout.plan
        axis = ownership.axis_name
        reduction = self.plan.domain.reduction
        factor = self.units.kinetic_to_energy
        row = _spec(axis, 1)
        point = _spec(axis, 2)
        return ownership.map(
            lambda p, m, a: owner_kinetic_energy(
                p, m, a, factor, reduction=reduction, axis_name=axis
            ),
            (point, row, row),
            _REPLICATED,
        )(momenta, domain.masses, self._mobile(domain))

    def _twice_kinetic_tensor(
        self, domain: AtomisticDomainState, momenta: Array, /
    ) -> Array:
        ownership = domain.layout.plan
        axis = ownership.axis_name
        reduction = self.plan.domain.reduction
        factor = self.units.kinetic_to_energy
        row = _spec(axis, 1)
        point = _spec(axis, 2)
        return ownership.map(
            lambda p, m, a: owner_kinetic_tensor(
                p, m, a, factor, reduction=reduction, axis_name=axis
            ),
            (point, row, row),
            _REPLICATED,
        )(momenta, domain.masses, self._mobile(domain))

    def _constraint_crossings(self, domain: AtomisticDomainState, /) -> Array:
        """Global count of constraint rows joining two scaled molecules."""
        constraints = domain.constraints
        if constraints is None or self.plan.ensemble.molecule_count is None:
            return jnp.zeros((), jnp.int32)
        halo = constraints.halo
        ownership = domain.layout.plan
        axis = ownership.axis_name
        row = _spec(axis, 1)
        point = _spec(axis, 2)

        def body(
            labels: Array,
            row_active: Array,
            route_columns: Array,
            route_valid: Array,
            send_slots: Array,
            send_valid: Array,
        ) -> Array:
            columns = halo.local_gather(labels, send_slots, send_valid)
            ok = row_active & route_valid[0::2] & route_valid[1::2]
            crossing = ok & (columns[route_columns[0::2]] != columns[route_columns[1::2]])
            return JaxCollectiveProvider(axis).sum(jnp.sum(crossing, dtype=jnp.int32))

        return ownership.map(body, (row, row, row, row, point, point), _REPLICATED)(
            domain.atom_payload[MOLECULE_PAYLOAD],
            constraints.rows.active,
            halo.route_columns,
            halo.route_valid,
            halo.send_slots,
            halo.send_valid,
        )

    # ---------------------------------------------------------- initialization

    def initialize(self, state: AtomisticDomainState, /) -> AtomisticDomainState:
        """Remove declared drift, project constraints, prime and admit (raises).

        As P5 ``initialize_state``: fixed atoms carry zero momentum, a declared
        center-of-mass restriction removes the (collective) drift, constraints
        are projected and GLE memory starts at zero.  Static refusals (missing
        payload or strain response) raise before any trajectory; numerical
        inadmissibility raises through ``eqx.error_if``.
        """
        self._require_state(state, initialized=False)
        if state.controllers is not None or state.ledger is not None:
            raise ValueError("initialize admits a fresh ingress state; use prime().")
        dtype = state.positions.dtype
        active = state.layout.active
        mobile = self._mobile(state)
        masses_ok = jnp.all(~active | (jnp.isfinite(state.masses) & (state.masses > 0.0)))
        momenta = jnp.where(mobile[:, None], state.momenta, 0.0)
        payload = None
        if isinstance(self.plan.integrator, GeneralizedLangevinPlan):
            payload = {
                **state.atom_payload,
                GLE_MEMORY_PAYLOAD: jnp.zeros_like(
                    state.atom_payload[GLE_MEMORY_PAYLOAD]
                ),
            }
        domain = self._with(state, state.positions, momenta, atom_payload=payload)
        if self.plan.ensemble.momentum_restriction == "center-of-mass":
            reductions = self.reductions(domain)
            drift = reductions.linear_momentum / reductions.total_mass
            momenta = jnp.where(
                mobile[:, None], momenta - state.masses[:, None] * drift, 0.0
            )
            domain = self._with(domain, domain.positions, momenta)
        constraint_successful = jnp.asarray(True)
        constraint_reasons = jnp.zeros((), jnp.int32)
        if self.plan.constraints is not None and domain.constraints is not None:
            unwrapped = self._unwrapped(
                domain.positions, domain.image_counts, domain.cell_vectors
            )
            solve = solve_domain_constraints(
                self.plan.constraints.method,
                domain.constraints,
                domain.layout,
                unwrapped,
                domain.momenta,
                domain.masses,
                mobile=mobile,
                positions_mode=True,
                component_row_capacity=self.plan.constraints.component_row_capacity,
            )
            constraints = eqx.tree_at(
                lambda value: (
                    value.lagrange,
                    value.position_residual,
                    value.velocity_residual,
                ),
                domain.constraints,
                (solve.lagrange, solve.position_residual, solve.velocity_residual),
            )
            domain = self._with(
                domain,
                self._stored(
                    solve.positions, domain.image_counts, domain.cell_vectors, active
                ),
                solve.momenta,
                constraints=constraints,
            )
            constraint_successful = solve.successful
            constraint_reasons = solve.reasons
            if self.plan.barostat is not None:
                # P5 _barostat._admit: constraints stay within one scaled molecule.
                constraint_reasons = constraint_reasons | _flag(
                    self._constraint_crossings(domain) > 0, _CONSTRAINT
                )
        evaluated = self._evaluate(domain)
        domain = evaluated.domain
        controllers = self.initial_controllers(domain.cell_vectors, dtype)
        reductions = self.reductions(domain)
        terms = self.extended_terms(controllers, domain.cell_vectors, dtype)
        ledger = initial_atomistic_energy_ledger(
            reductions.kinetic_energy, evaluated.energy, terms
        )
        force = self._force_cache(evaluated, domain.step_index)
        reasons = (
            evaluated.reasons
            | constraint_reasons
            | self._dof_reasons(reductions)
            | _flag(~masses_ok, int(AtomisticStepRejectionReason.NONFINITE))
            | _flag(
                ~tree_allfinite((domain.positions, domain.momenta, ledger, domain.time)),
                int(AtomisticStepRejectionReason.NONFINITE),
            )
        )
        successful = (reasons == 0) & constraint_successful
        positions = eqx.error_if(
            domain.positions,
            ~successful,
            "Initial distributed atomistic dynamics state is not admissible.",
        )
        primed = self._with(domain, positions, domain.momenta, force=force)
        # Initialization is the one transition that changes the controller and
        # ledger structure (absent -> typed P5 state); every later update keeps
        # it and goes through ``with_dynamics``.
        return eqx.tree_at(
            lambda value: (value.controllers, value.ledger),
            primed,
            (controllers, ledger),
            is_leaf=lambda value: value is None,
        )

    def prime(self, state: AtomisticDomainState, /) -> AtomisticDomainPrimeEvaluation:
        """Re-evaluate the force cache of an accepted state (P5 ``prime_state``).

        Step, time, RNG key and controllers are untouched; the energy jump is
        booked as external work.
        """
        self._require_state(state, initialized=True)
        if state.controllers is None or state.ledger is None:
            raise ValueError("prime requires an initialized state.")
        dtype = state.positions.dtype
        evaluated = self._evaluate(state)
        domain = evaluated.domain
        reductions = self.reductions(domain)
        terms = self.extended_terms(state.controllers, domain.cell_vectors, dtype)
        kinetic = reductions.kinetic_energy
        extended = kinetic + evaluated.energy + terms[0] + terms[1] + terms[2]
        zero = jnp.zeros_like(kinetic)
        ledger = advance_atomistic_energy_ledger(
            state.ledger,
            kinetic,
            evaluated.energy,
            terms,
            thermostat_heat=zero,
            reservoir_heat=zero,
            constraint_work=zero,
            barostat_work=zero,
            external_work=extended - state.ledger.extended_energy,
            balanced=False,
            accepted_increment=0,
        )
        force = self._force_cache(evaluated, domain.step_index)
        reasons = (
            evaluated.reasons
            | self._dof_reasons(reductions)
            | _flag(
                ~(self._force_finite(force) & tree_allfinite(ledger)),
                int(AtomisticStepRejectionReason.NONFINITE),
            )
        )
        successful = reasons == 0
        candidate = self._with(
            domain, domain.positions, domain.momenta, force=force, ledger=ledger
        )
        return AtomisticDomainPrimeEvaluation(
            candidate_state=candidate,
            accepted_state=_select_tree(successful, candidate, state),
            successful=successful,
            rejection_reasons=reasons,
        )

    # ----------------------------------------------------------- volume move

    def _molecule_centers(
        self, domain: AtomisticDomainState, unwrapped: Array, /
    ) -> Array:
        """Replicated mass-weighted molecule centers (collective segment sums)."""
        count = self.plan.ensemble.molecule_count
        if count is None:
            raise RuntimeError("Atomic volume scaling has no molecule centers.")
        ownership = domain.layout.plan
        axis = ownership.axis_name
        reduction = self.plan.domain.reduction
        row = _spec(axis, 1)
        point = _spec(axis, 2)

        def body(
            positions: Array, masses: Array, labels: Array, active: Array
        ) -> tuple[Array, Array]:
            segment = jnp.where(active, labels, 0)
            weight = jnp.where(active, masses, 0.0)
            total = jax.ops.segment_sum(weight, segment, count)
            moment = jax.ops.segment_sum(
                jnp.where(active[:, None], masses[:, None] * positions, 0.0),
                segment,
                count,
            )
            return reduction.owner_sum(total, axis), reduction.owner_sum(moment, axis)

        total, moment = ownership.map(
            body, (point, row, row, row), (_REPLICATED, _REPLICATED)
        )(
            unwrapped,
            domain.masses,
            domain.atom_payload[MOLECULE_PAYLOAD],
            domain.layout.active,
        )
        return moment / total[:, None]

    def monte_carlo_volume_move(
        self, state: AtomisticDomainState, /
    ) -> AtomisticDomainBarostatEvaluation:
        """P5 ``apply_isotropic_monte_carlo_barostat`` on the distributed state.

        Same draws (run key, realization, move operator, typed move counter),
        same proposal at fixed fractional coordinates of the declared scaled
        entities, the complete energy at the proposed cell (certificate and,
        when needed, a rebuild that migrates every continuation leaf), the
        ``(N + 1) ln(V'/V)`` Jacobian and the P5 counter rules: a completed
        Metropolis decision advances the counter, a failure advances nothing.
        """
        barostat = self.plan.barostat
        if barostat is None:
            raise ValueError("The dynamics plan declares no Monte Carlo barostat.")
        self._require_state(state, initialized=True)
        controllers = state.controllers
        previous_ledger = state.ledger
        if controllers is None or previous_ledger is None:
            raise ValueError("The volume move requires an initialized state.")
        ensemble = self.plan.ensemble
        cell = self.plan.domain.partition.cell
        dtype = state.positions.dtype
        force_valid = self.force_cache_valid(state)
        counter = controllers.move_counter
        key = jr.wrap_key_data(jr.key_data(state.rng_key).astype(jnp.uint32))
        key = jr.fold_in(key, jnp.asarray(barostat.realization_id, dtype=jnp.uint32))
        key = jr.fold_in(key, jnp.uint32(MONTE_CARLO_VOLUME_OPERATOR_ID))
        key = jr.fold_in(key, counter)
        log_volume_change = jr.uniform(
            jr.fold_in(key, jnp.uint32(0)),
            (),
            minval=-barostat.maximum_log_volume_change,
            maxval=barostat.maximum_log_volume_change,
            dtype=dtype,
        )
        log_uniform = jnp.log(jr.uniform(jr.fold_in(key, jnp.uint32(1)), (), dtype=dtype))
        linear_scale = jnp.exp(log_volume_change / 3.0)
        old_vectors = state.cell_vectors
        new_vectors = old_vectors * linear_scale
        old_unwrapped = self._unwrapped(state.positions, state.image_counts, old_vectors)
        active = state.layout.active
        if ensemble.volume_coordinate_convention == "atomic":
            fractional = cell.fractional_with_vectors(old_unwrapped, old_vectors)
            scaled = cell.cartesian_with_vectors(fractional, new_vectors)
            proposed = jnp.where(active[:, None], scaled, old_unwrapped)
        else:
            centers = self._molecule_centers(state, old_unwrapped)
            proposed_centers = cell.cartesian_with_vectors(
                cell.fractional_with_vectors(centers, old_vectors), new_vectors
            )
            labels = jnp.where(active, state.atom_payload[MOLECULE_PAYLOAD], 0)
            displacement = proposed_centers[labels] - centers[labels]
            proposed = jnp.where(
                active[:, None], old_unwrapped + displacement, old_unwrapped
            )
        moved = self._with(
            state,
            self._stored(proposed, state.image_counts, new_vectors, active),
            state.momenta,
            cell_vectors=new_vectors,
        )
        evaluated = self._evaluate(moved)
        domain = evaluated.domain
        volume_before = cell_volume(old_vectors)
        volume_after = cell_volume(new_vectors)
        energy_change = evaluated.energy - state.force.energy
        pressure_work = jnp.asarray(ensemble.pressure, dtype=dtype) * (
            volume_after - volume_before
        )
        beta = 1.0 / (self.units.boltzmann_constant * self._temperature(dtype))
        log_acceptance = -beta * (energy_change + pressure_work) + (
            ensemble.scaled_entity_count + 1
        ) * jnp.log(volume_after / volume_before)
        force = self._force_cache(evaluated, state.step_index)
        accepted_controllers = eqx.tree_at(
            lambda value: (value.move_counter, value.accepted_moves, value.cell_epoch),
            controllers,
            (counter + 1, controllers.accepted_moves + 1, controllers.cell_epoch + 1),
        )
        reductions = self.reductions(domain)
        kinetic = reductions.kinetic_energy
        terms = self.extended_terms(accepted_controllers, new_vectors, dtype)
        zero = jnp.zeros_like(kinetic)
        ledger = advance_atomistic_energy_ledger(
            previous_ledger,
            kinetic,
            evaluated.energy,
            terms,
            thermostat_heat=zero,
            reservoir_heat=energy_change,
            constraint_work=zero,
            barostat_work=pressure_work,
            external_work=zero,
            balanced=False,
            accepted_increment=0,
        )
        reasons = (
            evaluated.reasons
            | self._dof_reasons(reductions)
            | _flag(~force_valid, int(AtomisticStepRejectionReason.STALE_FORCE))
            | _flag(
                ~(
                    jnp.isfinite(log_acceptance)
                    & self._force_finite(force)
                    & tree_allfinite((ledger, new_vectors))
                ),
                int(AtomisticStepRejectionReason.NONFINITE),
            )
            | _flag(
                counter == jnp.iinfo(jnp.uint32).max,
                int(AtomisticStepRejectionReason.BAROSTAT),
            )
        )
        successful = reasons == 0
        accepted = successful & (log_uniform < jnp.minimum(log_acceptance, 0.0))
        candidate = self._with(
            domain,
            domain.positions,
            domain.momenta,
            force=force,
            controllers=accepted_controllers,
            ledger=ledger,
        )
        rejected = self._with(
            state,
            state.positions,
            state.momenta,
            controllers=eqx.tree_at(
                lambda value: value.move_counter, controllers, counter + 1
            ),
        )
        successor = _select_tree(
            accepted, candidate, _select_tree(successful, rejected, state)
        )
        return AtomisticDomainBarostatEvaluation(
            candidate_state=candidate,
            accepted_state=successor,
            accepted=accepted,
            log_acceptance_probability=log_acceptance,
            volume_before=volume_before,
            volume_after=volume_after,
            energy_before=state.force.energy,
            energy_after=evaluated.energy,
            successful=successful,
            rejection_reasons=reasons,
        )

    # -------------------------------------------------------------------- step

    def step(self, state: AtomisticDomainState, /) -> AtomisticDomainState:
        return self.step_detailed(state).accepted_state

    # This is the ordered P5 numerical schedule with distributed owner reductions.
    # Preserve force/constraint/controller and commit/rollback order together;
    # splitting source-faithful stages for line count risks a different trajectory.
    def step_detailed(
        self, state: AtomisticDomainState, /
    ) -> AtomisticDomainStepEvaluation:
        """Execute the declared P5 schedule as one distributed transaction."""
        self._require_state(state, initialized=True)
        controllers = state.controllers
        previous_ledger = state.ledger
        if controllers is None or previous_ledger is None:
            raise ValueError("step requires an initialized state.")
        plan = self.plan
        integrator = plan.integrator
        units = self.units
        dtype = state.positions.dtype
        dt = jnp.asarray(integrator.step_size, dtype=dtype)
        force_rate = units.force_to_momentum_rate
        dof = plan.degrees_of_freedom
        thermal = self._thermal_energy(dtype)
        key_data = jr.key_data(state.rng_key).astype(jnp.uint32)
        cell_method = isinstance(integrator, _CELL_METHODS)
        mass = self._barostat_mass(dtype) if cell_method else None
        origin = plan.domain.partition.cell.origin.astype(dtype)
        force_valid = self.force_cache_valid(state)

        domain = state
        forces = state.force.forces
        strain = state.force.strain_derivative
        evaluated: _Evaluated | None = None
        position_dirty = False
        zero = jnp.zeros((), dtype=dtype)
        thermostat_heat = zero
        reservoir_heat = zero
        constraint_work = zero
        constraint_successful = jnp.asarray(True)
        thermostat_successful = jnp.asarray(True)
        barostat_successful = jnp.asarray(True)
        cell_successful = jnp.asarray(True)
        reasons = jnp.zeros((), jnp.int32)
        thermostat_phase = 0
        if domain.constraints is not None:
            domain = self._with(
                domain,
                domain.positions,
                domain.momenta,
                constraints=eqx.tree_at(
                    lambda value: (value.position_residual, value.velocity_residual),
                    domain.constraints,
                    (zero, zero),
                ),
            )

        def gradient(cell: Any) -> Array:
            if not isinstance(integrator, _CELL_METHODS) or cell is None or mass is None:
                raise RuntimeError("Cell schedule lacks an MTK cell controller.")
            return cell_velocity_gradient(integrator.barostat.chart, cell.momentum, mass)

        for operator, coefficient in zip(
            plan.schedule.operators, plan.schedule.coefficients, strict=True
        ):
            duration = jnp.asarray(coefficient, dtype=dtype) * dt
            active = domain.layout.active
            mobile = self._mobile(domain)
            if operator is SplittingOperatorKind.FORCE_KICK:
                if position_dirty:
                    evaluated = self._evaluate(domain)
                    domain = evaluated.domain
                    active = domain.layout.active
                    mobile = self._mobile(domain)
                    forces = evaluated.forces
                    if self.plan.strain_response:
                        strain = evaluated.strain
                    reasons = reasons | evaluated.reasons
                    position_dirty = False
                if cell_method:
                    momenta, flow_ok = mtk_momentum_kick(
                        gradient(controllers.cell),
                        dof,
                        domain.momenta,
                        force_rate * forces,
                        duration,
                        mobile,
                    )
                    cell_successful = cell_successful & flow_ok
                else:
                    momenta = jnp.where(
                        mobile[:, None],
                        domain.momenta + duration * force_rate * forces,
                        0.0,
                    )
                domain = self._with(domain, domain.positions, momenta)
            elif operator is SplittingOperatorKind.DRIFT:
                inverse = _inverse_masses(domain.masses, mobile)
                unwrapped = self._unwrapped(
                    domain.positions, domain.image_counts, domain.cell_vectors
                )
                unwrapped = jnp.where(
                    mobile[:, None],
                    unwrapped + duration * domain.momenta * inverse[:, None],
                    unwrapped,
                )
                domain = self._with(
                    domain,
                    self._stored(
                        unwrapped, domain.image_counts, domain.cell_vectors, active
                    ),
                    domain.momenta,
                )
                position_dirty = True
            elif operator is SplittingOperatorKind.CELL_UPDATE:
                if not isinstance(integrator, _CELL_METHODS):
                    raise RuntimeError("Cell update requires an MTK method.")
                inverse = _inverse_masses(domain.masses, mobile)
                unwrapped = self._unwrapped(
                    domain.positions, domain.image_counts, domain.cell_vectors
                )
                drift = mtk_drift(
                    gradient(controllers.cell),
                    origin,
                    unwrapped,
                    domain.momenta * inverse[:, None],
                    domain.cell_vectors,
                    duration,
                    mobile,
                )
                vectors = drift.cell_vectors
                cell_successful = (
                    cell_successful
                    & drift.successful
                    & cell_admissible(vectors, integrator.barostat.maximum_condition)
                )
                domain = self._with(
                    domain,
                    self._stored(
                        drift.unwrapped_positions, domain.image_counts, vectors, active
                    ),
                    domain.momenta,
                    cell_vectors=vectors,
                )
                position_dirty = True
            elif operator is SplittingOperatorKind.BAROSTAT:
                if not isinstance(integrator, _CELL_METHODS) or controllers.cell is None:
                    raise RuntimeError("Barostat schedule lacks an MTK cell controller.")
                cell_force = barostat_force(
                    self._twice_kinetic_tensor(domain, domain.momenta),
                    strain,
                    cell_volume(domain.cell_vectors),
                    jnp.asarray(plan.ensemble.pressure, dtype=dtype),
                    dof,
                )
                updated = controllers.cell.momentum + duration * project_cell_force(
                    integrator.barostat.chart, cell_force
                )
                barostat_successful = barostat_successful & jnp.all(jnp.isfinite(updated))
                controllers = eqx.tree_at(
                    lambda value: value.cell.momentum, controllers, updated
                )
            elif operator is SplittingOperatorKind.CELL_THERMOSTAT:
                if (
                    not isinstance(integrator, MTKNPTPlan)
                    or controllers.cell is None
                    or controllers.cell_chain is None
                    or mass is None
                ):
                    raise RuntimeError("Cell thermostat requires an MTK NPT cell chain.")
                chart = integrator.barostat.chart
                cell_kinetic = cell_kinetic_energy(chart, controllers.cell.momentum, mass)
                chain = propagate_nose_hoover_chain(
                    integrator.cell_chain,
                    controllers.cell_chain,
                    2.0 * cell_kinetic,
                    chart.dimension,
                    thermal,
                    duration,
                )
                controllers = eqx.tree_at(
                    lambda value: (value.cell_chain, value.cell.momentum),
                    controllers,
                    (chain.state, controllers.cell.momentum * chain.scale),
                )
                thermostat_successful = thermostat_successful & chain.successful
            elif operator is SplittingOperatorKind.THERMOSTAT:
                kinetic_before = self._kinetic(domain, domain.momenta)
                if isinstance(integrator, BAOABLangevinPlan):
                    momenta, heat, ornstein_ok = self._baoab(domain, key_data)
                    thermostat_successful = thermostat_successful & ornstein_ok
                    reservoir_heat = reservoir_heat + heat
                elif isinstance(integrator, (NoseHooverChainPlan, MTKNPTPlan)):
                    parameters: NoseHooverChainParameters = (
                        integrator.chain
                        if isinstance(integrator, NoseHooverChainPlan)
                        else integrator.particle_chain
                    )
                    if controllers.particle_chain is None:
                        raise RuntimeError(
                            "Thermostat schedule lacks its particle chain."
                        )
                    chain = propagate_nose_hoover_chain(
                        parameters,
                        controllers.particle_chain,
                        2.0 * kinetic_before,
                        dof,
                        thermal,
                        duration,
                    )
                    momenta = jnp.where(
                        mobile[:, None], domain.momenta * chain.scale, 0.0
                    )
                    controllers = eqx.tree_at(
                        lambda value: value.particle_chain, controllers, chain.state
                    )
                    thermostat_successful = thermostat_successful & chain.successful
                elif isinstance(integrator, StochasticVelocityRescalingPlan):
                    scale = stochastic_velocity_rescaling_scale(
                        integrator,
                        key_data,
                        domain.step_index,
                        kinetic_before,
                        dof,
                        thermal,
                        duration,
                        operator_id=(
                            CSVR_FIRST_OPERATOR_ID
                            if thermostat_phase == 0
                            else CSVR_SECOND_OPERATOR_ID
                        ),
                    )
                    momenta = jnp.where(
                        mobile[:, None], domain.momenta * scale.scale, 0.0
                    )
                    thermostat_successful = thermostat_successful & scale.successful
                    reservoir_heat = (
                        reservoir_heat + self._kinetic(domain, momenta) - kinetic_before
                    )
                elif isinstance(integrator, GeneralizedLangevinPlan):
                    momenta, memory, heat, langevin_ok = self._generalized_langevin(
                        domain, key_data, thermostat_phase
                    )
                    domain = self._with(
                        domain,
                        domain.positions,
                        domain.momenta,
                        atom_payload={**domain.atom_payload, GLE_MEMORY_PAYLOAD: memory},
                    )
                    thermostat_successful = thermostat_successful & langevin_ok
                    reservoir_heat = reservoir_heat + heat
                elif isinstance(integrator, VelocityRescalingPlan):
                    due = (domain.step_index + 1) % integrator.interval == 0
                    scale = deterministic_velocity_rescaling_scale(
                        kinetic_before, dof, thermal
                    )
                    factor = jnp.where(due, scale.scale, 1.0)
                    momenta = jnp.where(mobile[:, None], domain.momenta * factor, 0.0)
                    thermostat_successful = thermostat_successful & (
                        ~due | scale.successful
                    )
                    controllers = eqx.tree_at(
                        lambda value: value.rescale_events,
                        controllers,
                        controllers.rescale_events + due.astype(jnp.int32),
                    )
                    reservoir_heat = (
                        reservoir_heat + self._kinetic(domain, momenta) - kinetic_before
                    )
                else:
                    raise RuntimeError(
                        f"{type(integrator).__name__} declares no particle thermostat."
                    )
                thermostat_heat = (
                    thermostat_heat + self._kinetic(domain, momenta) - kinetic_before
                )
                thermostat_phase += 1
                domain = self._with(domain, domain.positions, momenta)
            elif operator in (
                SplittingOperatorKind.POSITION_CONSTRAINT,
                SplittingOperatorKind.MOMENTUM_CONSTRAINT,
            ):
                constraint_plan = plan.constraints
                constraints = domain.constraints
                if constraint_plan is None or constraints is None:
                    raise RuntimeError("Constraint schedule lacks constraint rows.")
                positions_mode = operator is SplittingOperatorKind.POSITION_CONSTRAINT
                kinetic_before = self._kinetic(domain, domain.momenta)
                unwrapped = self._unwrapped(
                    domain.positions, domain.image_counts, domain.cell_vectors
                )
                solve = solve_domain_constraints(
                    constraint_plan.method,
                    constraints,
                    domain.layout,
                    unwrapped,
                    domain.momenta,
                    domain.masses,
                    mobile=mobile,
                    positions_mode=positions_mode,
                    component_row_capacity=constraint_plan.component_row_capacity,
                )
                constraint_work = (
                    constraint_work
                    + self._kinetic(domain, solve.momenta)
                    - kinetic_before
                )
                if positions_mode:
                    updated_constraints = eqx.tree_at(
                        lambda value: (
                            value.lagrange,
                            value.position_residual,
                            value.velocity_residual,
                        ),
                        constraints,
                        (
                            solve.lagrange,
                            solve.position_residual,
                            solve.velocity_residual,
                        ),
                    )
                    constraint_successful = constraint_successful & solve.successful
                    positions = self._stored(
                        solve.positions,
                        domain.image_counts,
                        domain.cell_vectors,
                        active,
                    )
                    position_dirty = True
                else:
                    updated_constraints = eqx.tree_at(
                        lambda value: value.velocity_residual,
                        constraints,
                        jnp.maximum(
                            constraints.velocity_residual, solve.velocity_residual
                        ),
                    )
                    constraint_successful = (
                        constraint_successful
                        & (solve.velocity_residual <= constraint_plan.method.tolerance)
                        & solve.finite
                        & constraints.successful
                    )
                    positions = domain.positions
                reasons = reasons | (solve.reasons & _CAPACITY)
                domain = self._with(
                    domain, positions, solve.momenta, constraints=updated_constraints
                )
            else:
                raise RuntimeError(
                    f"Dynamics schedule cannot execute {operator.value!r}."
                )
        if position_dirty or evaluated is None:
            evaluated = self._evaluate(domain)
            domain = evaluated.domain
            reasons = reasons | evaluated.reasons
        if cell_method:
            controllers = eqx.tree_at(
                lambda value: value.cell_epoch, controllers, controllers.cell_epoch + 1
            )
        step = state.step_index + 1
        reductions = self.reductions(domain)
        kinetic = reductions.kinetic_energy
        terms = self.extended_terms(controllers, domain.cell_vectors, dtype)
        previous_barostat = (
            previous_ledger.cell_kinetic_energy + previous_ledger.pressure_volume
        )
        ledger = advance_atomistic_energy_ledger(
            previous_ledger,
            kinetic,
            evaluated.energy,
            terms,
            thermostat_heat=thermostat_heat,
            reservoir_heat=reservoir_heat,
            constraint_work=constraint_work,
            barostat_work=terms[0] + terms[1] - previous_barostat,
            external_work=jnp.zeros_like(kinetic),
            balanced=True,
            accepted_increment=1,
        )
        force = self._force_cache(evaluated, step)
        candidate_finite = tree_allfinite(
            (
                domain.positions,
                domain.momenta,
                domain.cell_vectors,
                ledger,
                controllers,
                evaluated.forces,
            )
        )
        reasons = (
            reasons
            | self._dof_reasons(reductions)
            | _flag(~constraint_successful, _CONSTRAINT)
            | _flag(~thermostat_successful, int(AtomisticStepRejectionReason.THERMOSTAT))
            | _flag(~barostat_successful, int(AtomisticStepRejectionReason.BAROSTAT))
            | _flag(~cell_successful, int(AtomisticStepRejectionReason.CELL))
            | _flag(~candidate_finite, int(AtomisticStepRejectionReason.NONFINITE))
            | _flag(~force_valid, int(AtomisticStepRejectionReason.STALE_FORCE))
        )
        successful = reasons == 0
        status = jnp.where(
            successful,
            int(AtomisticDynamicsStatus.SUCCESS),
            jnp.where(
                candidate_finite,
                int(AtomisticDynamicsStatus.REJECTED),
                int(AtomisticDynamicsStatus.NONFINITE),
            ),
        ).astype(jnp.int32)
        candidate = self._with(
            domain,
            domain.positions,
            domain.momenta,
            step_index=step,
            time=state.time + jnp.asarray(integrator.step_size, dtype=state.time.dtype),
            force=force,
            controllers=controllers,
            ledger=ledger,
        )
        accepted = _select_tree(successful, candidate, state)
        diagnostics = self._diagnostics(candidate, reductions, successful, reasons)
        return AtomisticDomainStepEvaluation(
            candidate_state=candidate,
            accepted_state=accepted,
            diagnostics=diagnostics,
            successful=successful,
            status=status,
            residual=ledger.last_relative_energy_change,
            rejection_reasons=reasons,
        )

    # -------------------------------------------------------------- diagnostics

    def pressure_tensor(
        self,
        state: AtomisticDomainState,
        reductions: AtomisticDomainDynamicsReductions | None = None,
        /,
    ) -> tuple[Array, Array]:
        """``(sum p p^T / m - sym(dE/d strain)) / V`` (NaN without strain response)."""
        dtype = state.momenta.dtype
        if not self.plan.strain_response:
            return jnp.full((3, 3), jnp.nan, dtype=dtype), jnp.asarray(
                jnp.nan, dtype=dtype
            )
        values = self.reductions(state) if reductions is None else reductions
        volume = cell_volume(state.cell_vectors)
        strain = state.force.strain_derivative
        tensor = (values.twice_kinetic_tensor - 0.5 * (strain + strain.T)) / volume
        return jnp.where(state.force.strain_available, tensor, jnp.nan), volume

    def _diagnostics(
        self,
        state: AtomisticDomainState,
        reductions: AtomisticDomainDynamicsReductions,
        successful: Array,
        reasons: Array,
        /,
    ) -> AtomisticDomainDynamicsDiagnostics:
        ledger = state.ledger
        if ledger is None:
            raise ValueError("Diagnostics require an initialized state.")
        tensor, volume = self.pressure_tensor(state, reductions)
        temperature = (
            2.0
            * reductions.kinetic_energy
            / (self.plan.degrees_of_freedom * self.units.boltzmann_constant)
        )
        constraints = state.constraints
        residual = (
            jnp.zeros((), dtype=state.positions.dtype)
            if constraints is None
            else jnp.maximum(constraints.position_residual, constraints.velocity_residual)
        )
        return AtomisticDomainDynamicsDiagnostics(
            kinetic_energy=reductions.kinetic_energy,
            potential_energy=ledger.potential_energy,
            total_energy=ledger.total_energy,
            extended_energy=ledger.extended_energy,
            temperature=temperature,
            pressure=jnp.trace(tensor) / 3.0,
            pressure_tensor=tensor,
            volume=volume,
            total_linear_momentum=reductions.linear_momentum,
            degrees_of_freedom=reductions.degrees_of_freedom,
            constraint_residual=residual,
            energy=ledger,
            successful=jnp.asarray(successful, dtype=jnp.bool_),
            rejection_reasons=jnp.asarray(reasons, dtype=jnp.int32),
        )

    def diagnostics(
        self, state: AtomisticDomainState, /
    ) -> AtomisticDomainDynamicsDiagnostics:
        self._require_state(state, initialized=True)
        return self._diagnostics(
            state,
            self.reductions(state),
            jnp.asarray(True),
            jnp.zeros((), dtype=jnp.int32),
        )


__all__ = [
    "GLE_MEMORY_PAYLOAD",
    "MOBILE_PAYLOAD",
    "MOLECULE_PAYLOAD",
    "AtomisticDomainBarostatEvaluation",
    "AtomisticDomainConstraintEvidence",
    "AtomisticDomainConstraintPlan",
    "AtomisticDomainConstraintRows",
    "AtomisticDomainConstraintSolve",
    "AtomisticDomainConstraintTransition",
    "AtomisticDomainDynamicsDiagnostics",
    "AtomisticDomainDynamicsPlan",
    "AtomisticDomainDynamicsReductions",
    "AtomisticDomainEnsemblePlan",
    "AtomisticDomainIntegratorPlan",
    "AtomisticDomainPrimeEvaluation",
    "AtomisticDomainStepEvaluation",
    "PreparedAtomisticDomainDynamics",
    "domain_constraint_rows",
    "export_domain_constraints",
    "import_domain_constraints",
    "owner_count",
    "owner_degrees_of_freedom",
    "owner_inner_product",
    "owner_kinetic_energy",
    "owner_kinetic_tensor",
    "owner_linear_momentum",
    "owner_masked_sum",
    "rebuild_domain_constraints",
    "solve_domain_constraints",
]
