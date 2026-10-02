#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Distributed explicit electromagnetic PIC over a static block decomposition.

`distribute_pic_field_solver` executes a prepared cochain (periodic 3-D),
reduced 1-D/2-D, or Cartesian PSATD field solver on a device mesh of one to
three axes, which split the leading grid axes into slabs, pencils, or blocks.
`pic_distribution_support` states whether and how a base solver distributes
(quasi-cylindrical PSATD, unstructured Whitney, and bounded cochain grids are
refused). The result is an `AbstractDistributedPICFieldSolver`, so
`ElectromagneticPICPlan` runs over it unchanged:

- particle transfers run per device on the device's slot block: each device
  deposits its own particles on its block window, and one plane-level
  `DistributedHaloPlan` per decomposed axis accumulates guard contributions
  into their owners; gathers read the owned cells plus guards exchanged from
  every neighbor, diagonal neighbors included;
- the cochain, reduced, and global-FFT PSATD field updates are the base
  solver's own global update, SPMD-partitioned over the same mesh (global
  FFTs run through the solver's `DistributedSpectralExecutionPlan` slab or
  pencil schedule on this mesh);
- local-guarded PSATD (Kirchen et al., Phys. Rev. E 102, 063215 (2020)) runs
  per device: the owned fields, current, and charge receive ``guard_cells``
  cells exchanged through the same halo substrate on every decomposed axis, and
  each of the device's local-guarded subdomains is transformed, advanced, and
  inverted locally with the finite-order stencil; only the evidence (Gauss
  residual maxima, energy) is reduced over the mesh.

The distributed solver shares its base solver's identity: decomposition
changes reduction order, not the discretization, so restart components are
admitted across topologies and `PICDistributedEvidence` records execution.
Each route-specific solver publishes exactly the optional protocols whose
distributed route it executes and withholds the others with a stated reason
(`pic_capabilities`): moving windows are withheld because the window shift
does not migrate particles to their new owners, and relativistic self-fields
because they need a grounded (bounded) cochain boundary.

`DistributedElectromagneticPICPlan` places every species' slot block on its
owner device and installs a `DistributedPICExecutor` in the PIC plan:
creation-stage, population-stage, stateful, and charge-redistributing
processes (`PICDistributedProcess`) run per device on the device's slot
blocks, allocating created particles inside the block with
decomposition-independent identities (`PICIdentityAllocator`); before the
population stage and after every step, particles migrate with their
slot-aligned process state (and process-owned banks by their own positions)
through fixed-capacity ``ppermute`` packets (`PICMigrationPlan`). A migration
failure on any device rejects the whole step atomically
(`PICRejectionReason.MIGRATION`).
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from math import prod
from typing import Any, assert_never, final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding
from jax.typing import ArrayLike

from .._dtype_names import RealPrecisionDType
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.particle import (
    ChargedParticlePlan,
    ParticlePopulationPlan,
    ParticleSetPlan,
)
from ..discretization.pic import (
    AbstractPICParticleExecutor,
    AbstractPICProcess,
    ChargeConservingCurrentPlan,
    PICDistributedProcess,
    PICFieldProbeSample,
    PICParticleExchangeResult,
    PICProcessBank,
    PICProcessContext,
    PICProcessResult,
    PICProcessStatePartition,
    PICRejectionReason,
    PICRunStatus,
    PICSpeciesPlan,
    PICSpeciesState,
    PreparedPICParticleCochainTransfer,
)
from ..discretization.pic._distributed import (
    assemble_population,
    assemble_species,
    permute_slots,
    PICDomainDecomposition,
    PICGuardWindow,
    PICIdentityAllocator,
    PICMigrationEvidence,
    PICMigrationPlan,
    PICSlotGroup,
    population_slot_leaves,
    repartition_destinations,
    species_slot_leaves,
    with_population_slot_leaves,
    with_species_slot_leaves,
)
from ..typing import checked, parse
from ._cochain_pic_field import CochainMaxwellPICFieldSolver
from ._electromagnetic_pic import (
    ElectromagneticPICDiagnostics,
    ElectromagneticPICPlan,
    ElectromagneticPICState,
)
from ._maxwell import MaxwellCapabilities
from ._maxwell_far_field import HuygensSurfacePhasors
from ._maxwell_reduced import CompatibleMaxwell1DState, CompatibleMaxwell2DState
from ._pic_field_solver import (
    AbstractPreparedPICFieldSolver,
    add_deposits,
    PICCapabilityRecord,
    PICFieldAdvance,
    PICFieldDeposit,
    PICFieldEnergy,
    PICFieldSolverCapability,
    PICGaussProjection,
    PICGaussProjectionResult,
    PICMultiDeposit,
    PICRestartComponent,
    PICRestartState,
    PICSpectralSymbol,
    PICTensorKind,
    PICTensorMap,
)
from ._reduced_pic import ReducedMaxwellPICFieldSolver
from ._unstructured_em_pic import UnstructuredMaxwellPICFieldSolver
from .maxwell.spectral import (
    PreparedQuasiCylindricalMaxwell,
    PreparedSpectralMaxwell,
    SpectralMaxwellSource,
    SpectralMaxwellState,
)


PICDistributedRoute: TypeAlias = Literal[
    "cochain", "reduced", "spectral-global-fft", "spectral-local-guarded"
]
type _SharedCapability = Literal[
    "multi-deposit", "spectral-symbol", "restart-state", "gauss-projection"
]
type _Planes = tuple[Array, ...]
type _LeafMap = Callable[[Array], Any]


class _AbstractPlaneLayout(StrictModule):
    """Grid-leading views of one solver's charge, current, and gather fields.

    Every plane array leads with the solver grid's cell axes (``counts``).
    """

    counts: eqx.AbstractVar[tuple[int, ...]]
    lower: eqx.AbstractVar[tuple[float, ...]]
    spacing: eqx.AbstractVar[tuple[float, ...]]
    periodic: eqx.AbstractVar[tuple[bool, ...]]

    @abc.abstractmethod
    def charge_planes(self, value: Array, /) -> _Planes:
        raise NotImplementedError

    @abc.abstractmethod
    def charge_from(self, planes: _Planes, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def current_planes(self, value: Any, /) -> _Planes:
        raise NotImplementedError

    @abc.abstractmethod
    def current_from(self, planes: _Planes, /) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def field_planes(self, field: Any, /) -> _Planes:
        """The field arrays the solver's gather reads."""
        raise NotImplementedError

    @abc.abstractmethod
    def field_from(self, template: Any, planes: _Planes, /) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def map_field(self, field: Any, planes: _LeafMap, other: _LeafMap, /) -> Any:
        """Apply ``planes`` to grid-leading field leaves and ``other`` elsewhere."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def field_sharded(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def local_solver(self, parts: int, /) -> AbstractPreparedPICFieldSolver:
        """The solver's transfers bound to one device's ``C/P`` slot block."""
        raise NotImplementedError


def _block_particles(plan: ParticleSetPlan, parts: int, /) -> Any:
    """Prepared particle support of one ``capacity/parts`` slot block."""
    capacity = np.asarray(plan.particle_ids).shape[0]
    if capacity % parts:
        raise ValueError(
            "Distributed PIC capacities must divide into equal device slot blocks."
        )
    block = capacity // parts
    return ParticleSetPlan(
        np.asarray(plan.particle_ids)[:block],
        np.ones((block,)),
        ambient_dimension=plan.ambient_dimension,
        coordinate_dtype=plan.coordinate_dtype,
    ).prepare()


def _local_transfers(
    transfers: Sequence[PreparedPICParticleCochainTransfer],
    currents: Sequence[ChargeConservingCurrentPlan],
    parts: int,
    /,
) -> tuple[
    tuple[PreparedPICParticleCochainTransfer, ...],
    tuple[ChargeConservingCurrentPlan, ...],
]:
    """Per-species transfers of one slot block.

    Runtime deposits and gathers pass positions, activity, and macrocharge
    explicitly, so the block template only fixes the slot count.
    """
    local_transfers, local_currents = [], []
    for transfer, current in zip(transfers, currents, strict=True):
        species = transfer.species
        support = _block_particles(species.particles.plan, parts)
        block = np.asarray(species.particles.plan.particle_ids).shape[0] // parts
        charged = ChargedParticlePlan(
            np.ones((block,)),
            species.plan.species_id,
            require_constant_specific_charge=False,
        ).prepare(support)
        local = transfer.plan.prepare(charged)
        local_transfers.append(local)
        local_currents.append(
            ChargeConservingCurrentPlan(
                local,
                maximum_segments_per_particle=current.maximum_segments_per_particle,
                tolerance=current.tolerance,
            )
        )
    return tuple(local_transfers), tuple(local_currents)


def _local_species(plan: PICSpeciesPlan, parts: int, /) -> PICSpeciesPlan:
    """The species plan of one device's ``capacity/parts`` slot block."""
    population = plan.population
    if not bool(np.all(np.asarray(population.particles.active_mask))):
        raise ValueError("Distributed PIC species must have every slot structural.")
    support = _block_particles(population.particles.plan, parts)
    block = plan.capacity // parts
    return PICSpeciesPlan(
        ParticlePopulationPlan(
            support,
            reuse_policy=population.reuse_policy,
            allocation_capacity=min(population.allocation_capacity, block),
            incarnation_maximum=population.incarnation_maximum,
        ),
        plan.charge_model,
    )


def _require_planes(values: _Planes, counts: tuple[int, ...], /) -> _Planes:
    if any(value.shape[: len(counts)] != counts for value in values):
        raise ValueError("Solver layout arrays do not lead with the grid cells.")
    return values


@final
class _CochainLayout(_AbstractPlaneLayout):
    solver: CochainMaxwellPICFieldSolver
    counts: tuple[int, ...] = eqx.field(static=True)
    lower: tuple[float, ...] = eqx.field(static=True)
    spacing: tuple[float, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)

    def __init__(self, solver: CochainMaxwellPICFieldSolver, /) -> None:
        counts, lower, spacing, periodic = [], [], [], []
        for axis in solver.bridge.grid.structured_axes:
            widths = np.asarray(axis.interval_widths)
            if not np.allclose(widths, widths[0], rtol=1.0e-12, atol=0.0):
                raise ValueError("Distributed PIC requires uniform grid axes.")
            counts.append(widths.size)
            lower.append(float(np.asarray(axis.bounds[0])))
            spacing.append(float(widths[0]))
            periodic.append(bool(axis.periodic))
        self.solver = solver
        self.counts = tuple(counts)
        self.lower = tuple(lower)
        self.spacing = tuple(spacing)
        self.periodic = tuple(periodic)

    def _unpack(self, degree: int, value: Array, /) -> _Planes:
        kernel = self.solver.transfers[0].kernel
        shapes = kernel.component_shapes[degree]
        offsets = kernel.dof_offsets[degree]
        expected = sum(prod(shape) for shape in shapes)
        if value.shape != (expected,):
            raise ValueError("Distributed cochain values do not match kernel DOFs.")
        return _require_planes(
            tuple(
                value[offset : offset + prod(shape)].reshape(shape)
                for shape, offset in zip(shapes, offsets, strict=True)
            ),
            self.counts,
        )

    def _pack(self, degree: int, planes: _Planes, /) -> Array:
        shapes = self.solver.transfers[0].kernel.component_shapes[degree]
        if len(planes) != len(shapes) or any(
            plane.shape != shape for plane, shape in zip(planes, shapes, strict=True)
        ):
            raise ValueError("Distributed cochain planes do not match kernel DOFs.")
        return jnp.concatenate(tuple(plane.reshape(-1) for plane in planes))

    def charge_planes(self, value: Array, /) -> _Planes:
        return self._unpack(0, value)

    def charge_from(self, planes: _Planes, /) -> Array:
        return self._pack(0, planes)

    def current_planes(self, value: Any, /) -> _Planes:
        return self._unpack(1, value)

    def current_from(self, planes: _Planes, /) -> Any:
        return self._pack(1, planes)

    def field_planes(self, field: Any, /) -> _Planes:
        primary = field.primary
        return self._unpack(1, primary.electric_displacement) + self._unpack(
            2, primary.magnetic_flux
        )

    def field_from(self, template: Any, planes: _Planes, /) -> Any:
        electric = self._pack(1, planes[:3])
        magnetic = self._pack(2, planes[3:])
        return eqx.tree_at(
            lambda state: (
                state.primary.electric_displacement,
                state.primary.magnetic_flux,
            ),
            template,
            (electric, magnetic),
        )

    def map_field(self, field: Any, planes: _LeafMap, other: _LeafMap, /) -> Any:
        # Cochains are packed component-major; no leaf leads with the grid.
        del planes
        return jax.tree.map(other, field)

    @property
    def field_sharded(self) -> bool:
        return False

    def local_solver(self, parts: int, /) -> AbstractPreparedPICFieldSolver:
        solver = self.solver
        transfers, currents = _local_transfers(solver.transfers, solver.currents, parts)
        return CochainMaxwellPICFieldSolver(
            solver.maxwell, solver.electrostatic, transfers, currents
        )


class _ReducedLayout(_AbstractPlaneLayout):
    solver: ReducedMaxwellPICFieldSolver
    counts: tuple[int, ...] = eqx.field(static=True)
    lower: tuple[float, ...] = eqx.field(static=True)
    spacing: tuple[float, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)

    def __init__(self, solver: ReducedMaxwellPICFieldSolver, /) -> None:
        transfer = solver.transfer
        self.solver = solver
        self.counts = tuple(int(value) for value in transfer.shape)
        self.lower = tuple(float(value) for value in transfer.lower)
        self.spacing = tuple(float(value) for value in transfer.spacing)
        self.periodic = tuple(bool(value) for value in transfer.periodic)

    def charge_planes(self, value: Array, /) -> _Planes:
        return _require_planes((value,), self.counts)

    def charge_from(self, planes: _Planes, /) -> Array:
        return planes[0]

    def current_planes(self, value: Any, /) -> _Planes:
        return _require_planes(tuple(value), self.counts)

    def current_from(self, planes: _Planes, /) -> Any:
        return planes[0], planes[1], planes[2]

    def field_planes(self, field: Any, /) -> _Planes:
        return _require_planes(tuple(field.electric) + tuple(field.magnetic), self.counts)

    def field_from(self, template: Any, planes: _Planes, /) -> Any:
        return type(template)(
            (planes[0], planes[1], planes[2]),
            (planes[3], planes[4], planes[5]),
            template.charge,
            template.pml_memory,
        )

    def map_field(self, field: Any, planes: _LeafMap, other: _LeafMap, /) -> Any:
        if not isinstance(field, (CompatibleMaxwell1DState, CompatibleMaxwell2DState)):
            raise TypeError("Reduced field state has an unexpected type.")
        return type(field)(
            tuple(planes(value) for value in field.electric),
            tuple(planes(value) for value in field.magnetic),
            planes(field.charge),
            jax.tree.map(other, field.pml_memory),
        )

    @property
    def field_sharded(self) -> bool:
        return True

    def local_solver(self, parts: int, /) -> AbstractPreparedPICFieldSolver:
        del parts
        return self.solver


class _SpectralLayout(_AbstractPlaneLayout):
    solver: PreparedSpectralMaxwell
    counts: tuple[int, ...] = eqx.field(static=True)
    lower: tuple[float, ...] = eqx.field(static=True)
    spacing: tuple[float, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)

    def __init__(self, solver: PreparedSpectralMaxwell, /) -> None:
        self.solver = solver
        self.counts = solver.plan.counts
        self.lower = solver.plan.origin
        self.spacing = solver.plan.spacing
        self.periodic = (True, True, True)

    def charge_planes(self, value: Array, /) -> _Planes:
        return _require_planes((value,), self.counts)

    def charge_from(self, planes: _Planes, /) -> Array:
        return planes[0]

    def current_planes(self, value: Any, /) -> _Planes:
        if not isinstance(value, SpectralMaxwellSource):
            raise TypeError("Spectral current must be SpectralMaxwellSource.")
        # Sub-interval axis first in the solver layout; grid axes lead here.
        return _require_planes(
            (
                jnp.moveaxis(value.current, 0, 3),
                jnp.moveaxis(value.charge_change, 0, 3),
            ),
            self.counts,
        )

    def current_from(self, planes: _Planes, /) -> Any:
        return SpectralMaxwellSource(
            jnp.moveaxis(planes[0], 3, 0), jnp.moveaxis(planes[1], 3, 0)
        )

    def field_planes(self, field: Any, /) -> _Planes:
        if field.averaged_electric is not None and field.averaged_magnetic is not None:
            return _require_planes(
                (field.averaged_electric, field.averaged_magnetic), self.counts
            )
        return _require_planes((field.electric, field.magnetic), self.counts)

    def field_from(self, template: Any, planes: _Planes, /) -> Any:
        averaged = template.averaged_electric is not None
        return SpectralMaxwellState(
            electric=template.electric if averaged else planes[0],
            magnetic=template.magnetic if averaged else planes[1],
            charge=template.charge,
            averaged_electric=planes[0] if averaged else None,
            averaged_magnetic=planes[1] if averaged else None,
            electric_split=template.electric_split,
            magnetic_split=template.magnetic_split,
            absorber_charge=template.absorber_charge,
            absorber_magnetic_charge=template.absorber_magnetic_charge,
            antenna_charge=template.antenna_charge,
            antenna_magnetic_charge=template.antenna_magnetic_charge,
            observations=template.observations,
        )

    def map_field(self, field: Any, planes: _LeafMap, other: _LeafMap, /) -> Any:
        def optional(value: Array | None) -> Any:
            return None if value is None else planes(value)

        return SpectralMaxwellState(
            electric=planes(field.electric),
            magnetic=planes(field.magnetic),
            charge=planes(field.charge),
            averaged_electric=optional(field.averaged_electric),
            averaged_magnetic=optional(field.averaged_magnetic),
            electric_split=optional(field.electric_split),
            magnetic_split=optional(field.magnetic_split),
            absorber_charge=optional(field.absorber_charge),
            absorber_magnetic_charge=optional(field.absorber_magnetic_charge),
            antenna_charge=optional(field.antenna_charge),
            antenna_magnetic_charge=optional(field.antenna_magnetic_charge),
            observations=jax.tree.map(other, field.observations),
        )

    @property
    def field_sharded(self) -> bool:
        return True

    def local_solver(self, parts: int, /) -> AbstractPreparedPICFieldSolver:
        solver = self.solver
        transfers, currents = _local_transfers(solver.transfers, solver.currents, parts)
        return solver.plan.prepare(transfers, currents)


class PICDistributionSupport(StrictModule, NonTrainableState):
    """Whether `distribute_pic_field_solver` admits one prepared base solver.

    ``route`` is the distributed execution route, or ``None`` when distribution
    is refused; ``basis`` states the route's numerical scheme or the refusal.
    """

    configuration: str = eqx.field(static=True)
    route: PICDistributedRoute | None = eqx.field(static=True)
    basis: str = eqx.field(static=True)

    def __init__(
        self,
        configuration: str,
        route: PICDistributedRoute | None,
        basis: str,
        /,
    ) -> None:
        if not isinstance(configuration, str) or not configuration:
            raise ValueError("configuration must be a nonempty label.")
        parsed = None if route is None else parse(route, PICDistributedRoute, "route")
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("A distribution support record needs a nonempty basis.")
        self.configuration = configuration
        self.route = parsed
        self.basis = basis


def _spectral_support(solver: PreparedSpectralMaxwell, /) -> PICDistributionSupport:
    configuration = solver.pic_configuration
    match solver.plan.decomposition:
        case "global-fft":
            return PICDistributionSupport(
                configuration,
                "spectral-global-fft",
                "The base PSATD update SPMD-partitioned over the PIC mesh; its global "
                "FFTs run on a slab (one-axis) or pencil (two-axis) schedule of that "
                "mesh.",
            )
        case "local-guarded":
            return PICDistributionSupport(
                configuration,
                "spectral-local-guarded",
                "Per-device local-guarded subdomains with guard cells exchanged by "
                "halo; subdomains must tile the device blocks and the Coulomb and "
                "Gauss-projection FFTs run on the PIC mesh.",
            )
        case _:
            assert_never(solver.plan.decomposition)


def pic_distribution_support(
    solver: AbstractPreparedPICFieldSolver, /
) -> PICDistributionSupport:
    """The distributed route of ``solver``, or the reason distribution is refused."""
    if not isinstance(solver, AbstractPreparedPICFieldSolver):
        raise TypeError("solver must be an AbstractPreparedPICFieldSolver.")
    configuration = solver.pic_configuration
    if isinstance(solver, CochainMaxwellPICFieldSolver):
        if not all(solver.periodic):
            return PICDistributionSupport(
                configuration,
                None,
                "Distributed cochain PIC requires periodic axes: a bounded axis adds "
                "a wall vertex plane that does not tile the cell blocks.",
            )
        return PICDistributionSupport(
            configuration,
            "cochain",
            "The base cochain update SPMD-partitioned over the mesh with replicated "
            "packed cochains; block-window deposits and gathers exchange guards.",
        )
    if isinstance(solver, ReducedMaxwellPICFieldSolver):
        return PICDistributionSupport(
            configuration,
            "reduced",
            "The base reduced update on block-sharded fields; a reduced 2-D "
            "continuity projection is global, so it decomposes only when the guard "
            "windows cover the grid.",
        )
    if isinstance(solver, PreparedSpectralMaxwell):
        return _spectral_support(solver)
    if isinstance(solver, PreparedQuasiCylindricalMaxwell):
        return PICDistributionSupport(
            configuration,
            None,
            "Quasi-cylindrical PSATD is not distributed: its radial Hankel "
            "transforms couple every radius of every azimuthal mode.",
        )
    if isinstance(solver, UnstructuredMaxwellPICFieldSolver):
        return PICDistributionSupport(
            configuration,
            None,
            "Unstructured Whitney PIC is not distributed: tetrahedral cochains have "
            "no block decomposition or halo plan.",
        )
    return PICDistributionSupport(
        configuration,
        None,
        f"No distributed route is declared for {type(solver).__name__}.",
    )


def _layout(
    solver: AbstractPreparedPICFieldSolver, route: PICDistributedRoute, /
) -> _AbstractPlaneLayout:
    match route:
        case "cochain" if isinstance(solver, CochainMaxwellPICFieldSolver):
            return _CochainLayout(solver)
        case "reduced" if isinstance(solver, ReducedMaxwellPICFieldSolver):
            return _ReducedLayout(solver)
        case "spectral-global-fft" | "spectral-local-guarded" if isinstance(
            solver, PreparedSpectralMaxwell
        ):
            return _SpectralLayout(solver)
        case _:
            raise RuntimeError("The distributed route does not match its base solver.")


def _mesh_devices(mesh: Mesh, /) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(device.process_index), int(device.id)) for device in mesh.devices.flat
    )


def _validate_alignment(
    solver: AbstractPreparedPICFieldSolver,
    route: PICDistributedRoute,
    mesh: Mesh,
    /,
) -> None:
    """Refuse spectral solvers whose global transforms are off the PIC mesh.

    Global FFTs (the global-FFT update; the Coulomb initialization and Gauss
    projection of both routes) must run on the PIC mesh: a slab schedule on a
    one-axis mesh or a pencil schedule on a two-axis mesh. Local-guarded
    subdomains must tile every device block.
    """
    if not isinstance(solver, PreparedSpectralMaxwell) or mesh.size == 1:
        return
    topology = solver.plan.topology
    shape = tuple(int(value) for value in mesh.devices.shape)
    if len(shape) > 2:
        raise ValueError(
            "Distributed PSATD supports one- and two-axis meshes (slab and pencil "
            "global transforms)."
        )
    if (
        topology.mesh_shape != shape
        or tuple(topology.mesh_axis_names) != tuple(mesh.axis_names)
        or _mesh_devices(topology.mesh) != _mesh_devices(mesh)
    ):
        raise ValueError(
            "The PSATD FFT topology must be the distributed PIC mesh (slab or pencil "
            "schedule)."
        )
    subdomains = solver.plan.subdomains
    if route == "spectral-local-guarded" and (
        subdomains is None
        or any(block % parts for block, parts in zip(subdomains, shape, strict=False))
    ):
        raise ValueError(
            "Local-guarded PSATD subdomains must tile every device block: the mesh "
            "part counts must divide the subdomain counts."
        )


class PICDistributedEvidence(StrictModule, NonTrainableState):
    """Host evidence of one distributed PIC execution.

    ``mesh_shape`` splits the leading grid axes (``len(mesh_shape)`` of them:
    slabs, pencils, or blocks). ``maxwell_capabilities`` is the cochain Maxwell
    capability set with ``distributed`` and ``spatial_distribution`` set
    (``None`` for reduced and spectral solvers, which carry no Maxwell
    capability set). ``field_sharded`` states whether field arrays are sharded
    over the blocks (packed cochains are replicated and updated redundantly).
    ``spectral_guard_cells`` are the local-guarded PSATD guards exchanged per
    step (``None`` on other routes); `PreparedSpectralMaxwell.guard_truncation`
    bounds their stencil truncation.
    """

    distributed: bool = eqx.field(static=True)
    route: PICDistributedRoute = eqx.field(static=True)
    mesh_shape: tuple[int, ...] = eqx.field(static=True)
    axis_names: tuple[str, ...] = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    guard_cells: int = eqx.field(static=True)
    particle_margin: float = eqx.field(static=True)
    identity_tiles: tuple[int, ...] = eqx.field(static=True)
    field_sharded: bool = eqx.field(static=True)
    halo_plan_ids: tuple[str, ...] = eqx.field(static=True)
    halo_phases: int = eqx.field(static=True)
    halo_message_capacity: int = eqx.field(static=True)
    spectral_guard_cells: tuple[int, ...] | None = eqx.field(static=True)
    maxwell_capabilities: MaxwellCapabilities | None
    execution_id: str = eqx.field(static=True)


def _distributed_capabilities(
    solver: AbstractPreparedPICFieldSolver, /
) -> MaxwellCapabilities | None:
    if not isinstance(solver, CochainMaxwellPICFieldSolver):
        return None
    base = solver.maxwell.capabilities
    return MaxwellCapabilities(
        lossless=base.lossless,
        passive=base.passive,
        active=base.active,
        dispersive=base.dispersive,
        nonlinear=base.nonlinear,
        reversible=base.reversible,
        complex_required=base.complex_required,
        structured_only=base.structured_only,
        pml=base.pml,
        observers=base.observers,
        frequency_domain=base.frequency_domain,
        distributed=True,
        magnetic_closedness_preserving=base.magnetic_closedness_preserving,
        linear_time_invariant=base.linear_time_invariant,
        local_tensors=base.local_tensors,
        spatial_distribution=True,
        ffi=base.ffi,
    )


class AbstractDistributedPICFieldSolver(
    AbstractPreparedPICFieldSolver, NonTrainableState
):
    """A prepared PIC field solver executed over a one- to three-axis device mesh.

    Mesh axis ``a`` splits grid axis ``a`` into ``mesh.devices.shape[a]``
    blocks; ``identity_tiles`` (default one per cell) fixes the identity
    partition of created particles (`PICDomainDecomposition`). Construct it
    with `distribute_pic_field_solver`, which selects the route-specific
    solver: each publishes exactly the optional protocols whose distributed
    route it executes (`pic_capabilities`).
    """

    base: AbstractPreparedPICFieldSolver
    local: AbstractPreparedPICFieldSolver
    layout: _AbstractPlaneLayout
    decomposition: PICDomainDecomposition
    guard_window: PICGuardWindow | None
    evidence: PICDistributedEvidence
    mesh: Mesh = eqx.field(static=True)
    axis_names: tuple[str, ...] = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    field_dtype: RealPrecisionDType = eqx.field(static=True)

    @checked
    def __init__(
        self,
        base: AbstractPreparedPICFieldSolver,
        mesh: Mesh,
        /,
        *,
        guard_cells: int = 3,
        particle_margin: float = 1.0,
        identity_tiles: Sequence[int] | None = None,
    ) -> None:
        support = pic_distribution_support(base)
        route = support.route
        if route is None:
            raise ValueError(support.basis)
        if not self._executes(route):
            raise TypeError(f"{type(self).__name__} does not execute the {route} route.")
        layout = _layout(base, route)
        shape = tuple(int(value) for value in mesh.devices.shape)
        axes = len(shape)
        if axes > min(3, len(layout.counts), base.spatial_dimension):
            raise ValueError(
                "The device mesh has more axes than the solver grid can decompose."
            )
        if axes > 1 and min(shape) == 1:
            raise ValueError(
                "Distributed PIC mesh axes need at least two devices; drop size-one axes."
            )
        _validate_alignment(base, route, mesh)
        decomposition = PICDomainDecomposition(
            shape,
            layout.counts[:axes],
            layout.lower[:axes],
            layout.spacing[:axes],
            periodic=layout.periodic[:axes],
            guard_cells=guard_cells,
            particle_margin=particle_margin,
            identity_tiles=identity_tiles,
        )
        guard_window = None
        spectral_guards = None
        if route == "spectral-local-guarded" and isinstance(
            base, PreparedSpectralMaxwell
        ):
            guards = base.plan.guard_cells
            if guards is None:
                raise ValueError("Local-guarded PSATD lost its guard cells.")
            spectral_guards = guards[:axes]
            guard_window = decomposition.guard_window(spectral_guards)
        axis_names = tuple(str(name) for name in mesh.axis_names)
        execution_id = canonical_fingerprint(
            {
                "kind": "distributed-pic-execution",
                "solver": base.solver_id,
                "decomposition": decomposition.decomposition_id,
                "axis_names": list(axis_names),
                "devices": [list(key) for key in _mesh_devices(mesh)],
            }
        )
        halos = decomposition.halos
        self.base = base
        self.local = layout.local_solver(decomposition.part_count)
        self.layout = layout
        self.decomposition = decomposition
        self.guard_window = guard_window
        self.evidence = PICDistributedEvidence(
            distributed=True,
            route=route,
            mesh_shape=shape,
            axis_names=axis_names,
            part_count=decomposition.part_count,
            guard_cells=decomposition.guard_cells,
            particle_margin=decomposition.particle_margin,
            identity_tiles=decomposition.identity_tiles,
            field_sharded=layout.field_sharded,
            halo_plan_ids=tuple(value.plan_id for value in halos),
            halo_phases=sum(len(value.permutations) for value in halos),
            halo_message_capacity=max(value.message_capacity for value in halos),
            spectral_guard_cells=spectral_guards,
            maxwell_capabilities=_distributed_capabilities(base),
            execution_id=execution_id,
        )
        self.mesh = mesh
        self.axis_names = axis_names
        # Decomposition changes reduction order, not the discretization.
        self.solver_id = base.solver_id
        self.spatial_dimension = base.spatial_dimension
        self.field_dtype = base.field_dtype

    # -- execution helpers --------------------------------------------------------

    def slot_spec(self) -> PartitionSpec:
        """Partition of slot-leading arrays: one contiguous block per device."""
        return PartitionSpec(self.axis_names)

    def plane_spec(self) -> PartitionSpec:
        """Partition of grid-leading arrays over the decomposed axes."""
        return PartitionSpec(*self.axis_names)

    def coordinates(self) -> tuple[Array, ...]:
        """This device's mesh coordinates; call inside ``shard_map``."""
        return tuple(jax.lax.axis_index(name) for name in self.axis_names)

    def map(
        self,
        function: Callable[..., Any],
        in_specs: tuple[Any, ...],
        out_specs: Any,
        /,
    ) -> Callable[..., Any]:
        """Return ``function`` mapped over the prepared device mesh.

        Public distributed operations are stable module-level
        ``eqx.filter_jit`` entry points; this helper only builds the inner
        ``shard_map`` while those entry points are traced.
        """
        return jax.shard_map(
            function,
            mesh=self.mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )

    def _accumulate(
        self, planes: _Planes, coordinates: tuple[Array, ...], /
    ) -> tuple[_Planes, Array]:
        decomposition = self.decomposition
        outside = jnp.asarray(False)
        for value in planes:
            outside = outside | decomposition.outside_window(value, coordinates)
        owned = tuple(
            decomposition.accumulate(value, coordinates, axis_names=self.axis_names)
            for value in planes
        )
        return owned, outside

    def _global_ok(self, value: Array, /) -> Array:
        return jax.lax.pmin(value.astype(jnp.int32), self.axis_names) == 1

    def _distributed_deposit(
        self, deposit: Callable[[], PICFieldDeposit], /
    ) -> tuple[tuple[_Planes, _Planes, _Planes], Array, Array]:
        coordinates = self.coordinates()
        value = deposit()
        current, current_outside = self._accumulate(
            self.layout.current_planes(value.current), coordinates
        )
        start, start_outside = self._accumulate(
            self.layout.charge_planes(value.start_charge), coordinates
        )
        end, end_outside = self._accumulate(
            self.layout.charge_planes(value.end_charge), coordinates
        )
        local = value.successful & ~(current_outside | start_outside | end_outside)
        # Continuity defect and its certifying scale reduce together as one pair.
        return (
            (current, start, end),
            jax.lax.pmax(
                jnp.stack((value.continuity_defect, value.continuity_scale)),
                self.axis_names,
            ),
            self._global_ok(local),
        )

    def _deposit_result(
        self, planes: tuple[_Planes, _Planes, _Planes], defect: Array, ok: Array, /
    ) -> PICFieldDeposit:
        current, start, end = planes
        return PICFieldDeposit(
            self.layout.current_from(current),
            self.layout.charge_from(start),
            self.layout.charge_from(end),
            defect[0],
            ok,
            defect[1],
        )

    # -- core protocol ------------------------------------------------------------

    @property
    def stable_step(self) -> Array:
        return self.base.stable_step

    @property
    def displacement_widths(self) -> Array:
        return self.base.displacement_widths

    def validate_species(self, species: tuple[PICSpeciesPlan, ...], /) -> None:
        self.base.validate_species(species)
        parts = self.decomposition.part_count
        if any(value.capacity % parts for value in species):
            raise ValueError(
                "Distributed PIC species capacities must divide into equal device "
                "slot blocks."
            )
        for index, value in enumerate(species):
            self._validate_locality(index, value)

    def _probe_rows(
        self, species: int, capacity: int, /
    ) -> tuple[np.ndarray, np.ndarray]:
        start, end = self.base.pairing_probe(species, capacity)
        return np.array(start), np.array(end)

    def _faces(self, part: int, /) -> list[tuple[int, float, float]]:
        """Probe paths of one block: start inside a face, end ``margin`` beyond it."""
        decomposition = self.decomposition
        margin = decomposition.particle_margin
        faces = []
        for axis, coordinate in enumerate(decomposition.coordinates(part)):
            width = decomposition.block_cells[axis]
            periodic = decomposition.periodic[axis]
            if periodic or coordinate > 0:
                faces.append(
                    (axis, coordinate * width + 1.0e-3, coordinate * width - margin)
                )
            if periodic or coordinate < decomposition.parts[axis] - 1:
                faces.append(
                    (
                        axis,
                        (coordinate + 1) * width - 1.0e-3,
                        (coordinate + 1) * width + margin - 1.0e-9,
                    )
                )
        return faces

    def _validate_locality(self, species: int, plan: PICSpeciesPlan, /) -> None:
        """Refuse guards narrower than the transfer footprint within the margin.

        Paths start inside a block and end ``particle_margin`` cells beyond each
        interior block face; their current and charge must vanish outside the
        block window, and a field supported only outside the window must gather
        to exactly zero at their endpoints.
        """
        decomposition = self.decomposition
        if decomposition.part_count == 1:
            return
        capacity = plan.capacity
        start, _ = self._probe_rows(species, capacity)
        margin = decomposition.particle_margin
        active = plan.population.particles.active_mask
        charge = jnp.where(active, 1.0, 0.0).astype(start.dtype)
        step = 0.5 * jnp.asarray(self.base.stable_step)
        empty, _ = self.base.deposit_charge(
            species, jnp.asarray(start), charge, jnp.zeros_like(active)
        )
        template = self.base.field_with_charge(jnp.zeros_like(empty))
        axes = decomposition.axis_count
        for part in range(decomposition.part_count):
            faces = self._faces(part)
            if not faces:
                continue
            coordinates = decomposition.coordinates(part)
            begin = start.copy()
            finish = start.copy()
            for axis, coordinate in enumerate(coordinates):
                center = (
                    decomposition.lower[axis]
                    + ((coordinate + 0.5) * decomposition.block_cells[axis])
                    * decomposition.spacing[axis]
                )
                begin[:, axis] = finish[:, axis] = center
            for row in range(capacity):
                axis, first, last = faces[row % len(faces)]
                origin, width = decomposition.lower[axis], decomposition.spacing[axis]
                begin[row, axis] = origin + first * width
                finish[row, axis] = origin + last * width
            begin_, finish_ = jnp.asarray(begin), jnp.asarray(finish)
            velocity = jnp.pad(
                (finish_ - begin_) / step, ((0, 0), (0, 3 - begin.shape[1]))
            )
            deposited = self.base.deposit(
                species, begin_, finish_, velocity, charge, active, step
            )
            density, _ = self.base.deposit_charge(species, finish_, charge, active)
            planes = (
                *self.layout.current_planes(deposited.current),
                *self.layout.charge_planes(deposited.start_charge),
                *self.layout.charge_planes(deposited.end_charge),
                *self.layout.charge_planes(density),
            )
            index = tuple(jnp.asarray(value, dtype=jnp.int32) for value in coordinates)
            outside = any(
                bool(decomposition.outside_window(value, index)) for value in planes
            )
            indicator = (~decomposition.window_indicator(part)).astype(np.float64)
            field = self.layout.field_from(
                template,
                tuple(
                    jnp.broadcast_to(
                        jnp.asarray(indicator).reshape(
                            indicator.shape + (1,) * (value.ndim - axes)
                        ),
                        value.shape,
                    ).astype(value.dtype)
                    for value in self.layout.field_planes(template)
                ),
            )
            leaked = 0.0
            for position in (begin_, finish_):
                electric, magnetic, _ = self.base.gather_fields(
                    species, position, active, field
                )
                leaked = max(
                    leaked,
                    float(jnp.max(jnp.abs(electric), initial=0.0)),
                    float(jnp.max(jnp.abs(magnetic), initial=0.0)),
                )
            if outside or leaked != 0.0:
                raise ValueError(
                    f"guard_cells={decomposition.guard_cells} does not contain the "
                    f"transfer footprint of species {species} within particle_margin="
                    f"{margin} cells of block {coordinates}, or the solver's deposit "
                    "is not window-local."
                )

    def pairing_probe(self, species: int, capacity: int, /) -> tuple[Array, Array]:
        """The base probe paths regrouped so each slot block starts in its block.

        Even blocks stack their slots on one path, so the probe charge change
        cannot vanish by the translation symmetry of a uniform lattice.
        """
        decomposition = self.decomposition
        parts = decomposition.part_count
        block = capacity // parts
        start, end = self._probe_rows(species, capacity)
        owner = np.asarray(decomposition.owner(jnp.asarray(start)))
        rows = []
        shifts = []
        cells = np.floor(
            (start[0, : decomposition.axis_count] - np.asarray(decomposition.lower))
            / np.asarray(decomposition.spacing)
        )
        widths = np.asarray(decomposition.block_cells)
        for part in range(parts):
            owned = np.flatnonzero(owner == part)
            spread = np.zeros((block,), np.int64) if part % 2 == 0 else np.arange(block)
            shift = np.zeros((block, start.shape[1]))
            if owned.size:
                rows.append(owned[spread % owned.size])
                shifts.append(shift)
                continue
            target = np.asarray(decomposition.coordinates(part)) * widths + cells % widths
            shift[:, : decomposition.axis_count] = (target - cells) * np.asarray(
                decomposition.spacing
            )
            rows.append(np.zeros((block,), dtype=np.int64))
            shifts.append(shift)
        selected = np.concatenate(rows)
        offset = np.concatenate(shifts)
        return (
            jnp.asarray(start[selected] + offset),
            jnp.asarray(end[selected] + offset),
        )

    def field_with_charge(self, charge: Array, /) -> Any:
        return self.base.field_with_charge(charge)

    def initialize_field(
        self, charge: Array, /, *, magnetic: Any = None
    ) -> tuple[Any, Array]:
        return self.base.initialize_field(charge, magnetic=magnetic)

    def field_charge(self, field: Any, /) -> Array:
        return self.base.field_charge(field)

    def field_energy(self, field: Any, /) -> Array:
        return self.base.field_energy(field)

    @eqx.filter_jit
    def advance(
        self, time: Array, field: Any, current: Any, step_size: Array, /
    ) -> PICFieldAdvance:
        base = self.base
        window = self.guard_window
        if window is None or not isinstance(base, PreparedSpectralMaxwell):
            return base.advance(time, field, current, step_size)
        return self._guarded_advance(base, window, time, field, current, step_size)

    def _guarded_advance(
        self,
        base: PreparedSpectralMaxwell,
        window: PICGuardWindow,
        time: Array,
        field: SpectralMaxwellState,
        current: SpectralMaxwellSource,
        step_size: Array,
        /,
    ) -> PICFieldAdvance:
        """Local-guarded PSATD step: per-device guard exchange and local FFTs."""
        axes = self.decomposition.axis_count
        padded = (axes > 0, axes > 1, axes > 2)
        conductivity = (
            None
            if base.electric_conductivity is None or base.magnetic_conductivity is None
            else (base.electric_conductivity, base.magnetic_conductivity)
        )
        owned = SpectralMaxwellState(
            electric=field.electric,
            magnetic=field.magnetic,
            charge=field.charge,
            averaged_electric=field.averaged_electric,
            averaged_magnetic=field.averaged_magnetic,
            electric_split=field.electric_split,
            magnetic_split=field.magnetic_split,
            absorber_charge=field.absorber_charge,
            absorber_magnetic_charge=field.absorber_magnetic_charge,
            antenna_charge=field.antenna_charge,
            antenna_magnetic_charge=field.antenna_magnetic_charge,
            observations=(),
        )

        def local(
            field: SpectralMaxwellState,
            current: SpectralMaxwellSource,
            conductivity: tuple[Array, Array] | None,
            step_size: Array,
        ) -> Any:
            coordinates = self.coordinates()

            def extend(values: Array) -> Array:
                return window.extend(values, coordinates, self.axis_names)

            return base.guarded_update(
                field, current, step_size, conductivity, extend, padded
            )

        planes = self.plane_spec()
        update = self.map(
            local,
            (planes, PartitionSpec(None, *self.axis_names), planes, PartitionSpec()),
            planes,
        )(owned, current, conductivity, jnp.asarray(step_size))
        return base.complete_advance(time, field, current, step_size, update)

    @eqx.filter_jit
    def deposit_charge(
        self,
        species: int,
        position: Array,
        macrocharge: Array,
        active: Array,
        /,
    ) -> tuple[Array, Array]:
        def local(
            position: Array, macrocharge: Array, active: Array
        ) -> tuple[_Planes, Array]:
            value, successful = self.local.deposit_charge(
                species, position, macrocharge, active
            )
            owned, outside = self._accumulate(
                self.layout.charge_planes(value), self.coordinates()
            )
            return owned, self._global_ok(successful & ~outside)

        slot = self.slot_spec()
        owned, ok = self.map(
            local, (slot, slot, slot), (self.plane_spec(), PartitionSpec())
        )(position, macrocharge, active)
        return self.layout.charge_from(owned), ok

    @eqx.filter_jit
    def deposit(
        self,
        species: int,
        start: Array,
        end: Array,
        velocity: Array,
        macrocharge: Array,
        active: Array,
        step_size: Array,
        /,
    ) -> PICFieldDeposit:
        def local(
            start: Array,
            end: Array,
            velocity: Array,
            macrocharge: Array,
            active: Array,
            step_size: Array,
        ) -> tuple[tuple[_Planes, _Planes, _Planes], Array, Array]:
            return self._distributed_deposit(
                lambda: self.local.deposit(
                    species, start, end, velocity, macrocharge, active, step_size
                )
            )

        slot = self.slot_spec()
        planes, defect, ok = self.map(
            local,
            (slot, slot, slot, slot, slot, PartitionSpec()),
            (self.plane_spec(), PartitionSpec(), PartitionSpec()),
        )(start, end, velocity, macrocharge, active, step_size)
        return self._deposit_result(planes, defect, ok)

    @eqx.filter_jit
    def deposit_all(
        self,
        starts: tuple[Array, ...],
        ends: tuple[Array, ...],
        velocities: tuple[Array, ...],
        macrocharges: tuple[Array, ...],
        actives: tuple[Array, ...],
        step_size: Array,
        /,
    ) -> PICFieldDeposit:
        """Every species' local deposits fused before one halo accumulation."""
        base = self.local

        def local(
            starts: tuple[Array, ...],
            ends: tuple[Array, ...],
            velocities: tuple[Array, ...],
            macrocharges: tuple[Array, ...],
            actives: tuple[Array, ...],
            step_size: Array,
        ) -> tuple[tuple[_Planes, _Planes, _Planes], Array, Array]:
            def deposit() -> PICFieldDeposit:
                if isinstance(base, PICMultiDeposit):
                    return base.deposit_all(
                        starts, ends, velocities, macrocharges, actives, step_size
                    )
                return add_deposits(
                    tuple(
                        base.deposit(index, *arrays, step_size)
                        for index, arrays in enumerate(
                            zip(
                                starts,
                                ends,
                                velocities,
                                macrocharges,
                                actives,
                                strict=True,
                            )
                        )
                    )
                )

            return self._distributed_deposit(deposit)

        slot = self.slot_spec()
        planes, defect, ok = self.map(
            local,
            (slot, slot, slot, slot, slot, PartitionSpec()),
            (self.plane_spec(), PartitionSpec(), PartitionSpec()),
        )(starts, ends, velocities, macrocharges, actives, step_size)
        return self._deposit_result(planes, defect, ok)

    @eqx.filter_jit
    def gather_fields(
        self,
        species: int,
        position: Array,
        active: Array,
        field: Any,
        /,
    ) -> tuple[Array, Array, Array]:
        """Fields at owned particles from the owned cells plus exchanged guards.

        Particles farther than ``particle_margin`` cells from their device's
        block are unsupported.
        """
        charge = self.base.field_charge(field)
        shape, dtype = charge.shape, charge.dtype
        decomposition = self.decomposition

        def local(
            position: Array, active: Array, planes: _Planes
        ) -> tuple[Array, Array, Array]:
            coordinates = self.coordinates()
            template = self.base.field_with_charge(jnp.zeros(shape, dtype=dtype))
            window = tuple(
                decomposition.fill_window(value, coordinates, axis_names=self.axis_names)
                for value in planes
            )
            electric, magnetic, support = self.local.gather_fields(
                species, position, active, self.layout.field_from(template, window)
            )
            region = decomposition.in_region(position, coordinates)
            return electric, magnetic, support & (region | ~active)

        slot = self.slot_spec()
        return self.map(
            local,
            (slot, slot, self.plane_spec()),
            (slot, slot, slot),
        )(position, active, self.layout.field_planes(field))

    # -- optional capabilities ------------------------------------------------------

    @abc.abstractmethod
    def _executes(self, route: PICDistributedRoute, /) -> bool:
        """Whether this route-specific solver executes ``route``."""
        raise NotImplementedError

    @property
    def pic_configuration(self) -> str:
        return f"distributed-{self.base.pic_configuration}"

    def _forwarded(
        self, capability: PICFieldSolverCapability, route: str, /
    ) -> PICCapabilityRecord:
        """A base protocol whose distributed route this solver executes."""
        record = self.base.pic_capability(capability)
        if not record.published:
            raise RuntimeError(
                f"Distribution forwards {capability!r}, which its base does not publish."
            )
        return PICCapabilityRecord(
            capability,
            published=True,
            admitted=record.admitted,
            basis=f"{record.basis} Distributed: {route}",
        )

    def _withheld(
        self, capability: PICFieldSolverCapability, reason: str, /
    ) -> PICCapabilityRecord:
        """A base protocol distribution refuses, with the refusal reason."""
        if not self.base.pic_capability(capability).published:
            raise RuntimeError(
                f"Distribution withholds {capability!r}, which its base does not publish."
            )
        return PICCapabilityRecord.refusal(
            capability, f"Withheld by distribution: {reason}"
        )

    def _unpublished(
        self, capability: PICFieldSolverCapability, /
    ) -> PICCapabilityRecord:
        """A protocol the base does not publish; the base refusal carries over."""
        record = self.base.pic_capability(capability)
        if record.published:
            raise RuntimeError(
                f"Distribution leaves the base-published {capability!r} undeclared."
            )
        return record

    def _shared(self, capability: _SharedCapability, /) -> PICCapabilityRecord:
        """Protocols every distributed route executes."""
        match capability:
            case "multi-deposit":
                return PICCapabilityRecord.route(
                    capability,
                    "Every species' block-window deposit is summed on its device "
                    "before one halo accumulation per decomposed axis.",
                )
            case "spectral-symbol":
                return self._forwarded(
                    capability, "the distributed update is the base update."
                )
            case "restart-state":
                return self._forwarded(
                    capability,
                    "components keep the base identity and restart across topologies.",
                )
            case "gauss-projection":
                return self._forwarded(
                    capability,
                    "the projection runs on the distributed field with the "
                    "halo-accumulated charge.",
                )
            case _:
                assert_never(capability)

    def project_gauss(self, field: Any, charge: Array, /) -> PICGaussProjectionResult:
        """The base solver's Gauss projection on the distributed field.

        The charge is the halo-accumulated distributed deposit; the projection
        runs on the same mesh (partitioned cochain/reduced Poisson solves, PSATD
        global transform on the PIC mesh).
        """
        base = self.base
        if not isinstance(base, PICGaussProjection):
            raise RuntimeError("A distributed base solver lost PICGaussProjection.")
        return base.project_gauss(field, charge)

    def dispersion_frequency(
        self, wavevector: ArrayLike, step_size: ArrayLike, /
    ) -> Array:
        base = self.base
        if not isinstance(base, PICSpectralSymbol):
            raise RuntimeError("A distributed base solver lost PICSpectralSymbol.")
        return base.dispersion_frequency(wavevector, step_size)

    def restart_component(self, field: Any, /) -> PICRestartComponent:
        base = self.base
        if not isinstance(base, PICRestartState):
            raise RuntimeError("A distributed base solver lost PICRestartState.")
        return base.restart_component(field)

    def restore_component(self, component: PICRestartComponent, /) -> Any:
        base = self.base
        if not isinstance(base, PICRestartState):
            raise RuntimeError("A distributed base solver lost PICRestartState.")
        return base.restore_component(component)


_WINDOW_REFUSAL = (
    "PICMovingWindowPlan translates particles and fills the leading cells without "
    "migrating them to their new owner blocks."
)


class _DistributedCochainPICFieldSolver(AbstractDistributedPICFieldSolver):
    """Distributed periodic 3-D cochain PIC and its cochain-local protocols."""

    if TYPE_CHECKING:
        __init__ = AbstractDistributedPICFieldSolver.__init__

    def _executes(self, route: PICDistributedRoute, /) -> bool:
        return route == "cochain"

    def _cochain(self) -> CochainMaxwellPICFieldSolver:
        base = self.base
        if not isinstance(base, CochainMaxwellPICFieldSolver):
            raise RuntimeError("Distributed cochain PIC lost its cochain base solver.")
        return base

    def pic_capability(
        self, capability: PICFieldSolverCapability, /
    ) -> PICCapabilityRecord:
        match capability:
            case (
                "multi-deposit"
                | "spectral-symbol"
                | "restart-state"
                | ("gauss-projection")
            ):
                return self._shared(capability)
            case "tensor-layout":
                return self._forwarded(
                    capability,
                    "filters act on the global charge, current, and field of the "
                    "partitioned update.",
                )
            case "huygens-sampling":
                return self._withheld(
                    capability,
                    "cochain PIC never carries Huygens boxes (Maxwell refuses them "
                    "with dynamic PIC currents), so no distributed route exists.",
                )
            case "energy-accounting":
                return self._forwarded(
                    capability, "energy split and loss power of the replicated cochains."
                )
            case "open-domain":
                return self._forwarded(
                    capability, "every distributed cochain axis is periodic."
                )
            case "window-shift":
                return self._withheld(capability, _WINDOW_REFUSAL)
            case "relativistic-self-fields":
                return self._withheld(
                    capability,
                    "superposed Coulomb fields need a zero-valued grounded boundary "
                    "on bounded axes, and distributed cochain axes are periodic.",
                )
            case "galilean-grid":
                return self._unpublished(capability)
            case _:
                assert_never(capability)

    @property
    def tensor_periodic(self) -> tuple[bool, ...]:
        return self._cochain().tensor_periodic

    def tensor_template(self, kind: PICTensorKind, /) -> Any:
        return self._cochain().tensor_template(kind)

    def map_tensors(
        self, kind: PICTensorKind, value: Any, function: PICTensorMap, /
    ) -> Any:
        return self._cochain().map_tensors(kind, value, function)

    def energy_components(self, field: Any, step_size: Array, /) -> PICFieldEnergy:
        return self._cochain().energy_components(field, step_size)

    def loss_power(self, field: Any, /) -> Array:
        return self._cochain().loss_power(field)

    @property
    def domain_periodic(self) -> tuple[bool, ...]:
        return self._cochain().domain_periodic

    @property
    def domain_bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return self._cochain().domain_bounds

    def boundary_inset(self, species: int, /) -> tuple[float, ...]:
        return self._cochain().boundary_inset(species)


class _DistributedReducedPICFieldSolver(AbstractDistributedPICFieldSolver):
    """Distributed reduced 1-D/2-D PIC and its filter layout."""

    if TYPE_CHECKING:
        __init__ = AbstractDistributedPICFieldSolver.__init__

    def _executes(self, route: PICDistributedRoute, /) -> bool:
        return route == "reduced"

    def _reduced(self) -> ReducedMaxwellPICFieldSolver:
        base = self.base
        if not isinstance(base, ReducedMaxwellPICFieldSolver):
            raise RuntimeError("Distributed reduced PIC lost its reduced base solver.")
        return base

    def pic_capability(
        self, capability: PICFieldSolverCapability, /
    ) -> PICCapabilityRecord:
        match capability:
            case (
                "multi-deposit"
                | "spectral-symbol"
                | "restart-state"
                | ("gauss-projection")
            ):
                return self._shared(capability)
            case "tensor-layout":
                return self._forwarded(
                    capability,
                    "filters act on the global charge, current, and field triples of "
                    "the sharded update.",
                )
            case "window-shift":
                return self._withheld(capability, _WINDOW_REFUSAL)
            case (
                "huygens-sampling"
                | "galilean-grid"
                | "energy-accounting"
                | "open-domain"
                | "relativistic-self-fields"
            ):
                return self._unpublished(capability)
            case _:
                assert_never(capability)

    @property
    def tensor_periodic(self) -> tuple[bool, ...]:
        return self._reduced().tensor_periodic

    def tensor_template(self, kind: PICTensorKind, /) -> Any:
        return self._reduced().tensor_template(kind)

    def map_tensors(
        self, kind: PICTensorKind, value: Any, function: PICTensorMap, /
    ) -> Any:
        return self._reduced().map_tensors(kind, value, function)


class _DistributedSpectralPICFieldSolver(AbstractDistributedPICFieldSolver):
    """Distributed Cartesian PSATD (global-FFT or local-guarded)."""

    if TYPE_CHECKING:
        __init__ = AbstractDistributedPICFieldSolver.__init__

    def _executes(self, route: PICDistributedRoute, /) -> bool:
        return route in ("spectral-global-fft", "spectral-local-guarded")

    def _spectral(self) -> PreparedSpectralMaxwell:
        base = self.base
        if not isinstance(base, PreparedSpectralMaxwell):
            raise RuntimeError("Distributed PSATD lost its spectral base solver.")
        return base

    def pic_capability(
        self, capability: PICFieldSolverCapability, /
    ) -> PICCapabilityRecord:
        match capability:
            case (
                "multi-deposit"
                | "spectral-symbol"
                | "restart-state"
                | ("gauss-projection")
            ):
                return self._shared(capability)
            case "huygens-sampling":
                return self._forwarded(
                    capability,
                    "observers accumulate from the global fields when each step "
                    "completes.",
                )
            case "galilean-grid":
                return self._forwarded(
                    capability,
                    "particles drift and migrate in grid coordinates; the base forms "
                    "the lab current.",
                )
            case (
                "tensor-layout"
                | "window-shift"
                | "energy-accounting"
                | "open-domain"
                | "relativistic-self-fields"
            ):
                return self._unpublished(capability)
            case _:
                assert_never(capability)

    @property
    def grid_velocity(self) -> tuple[float, ...]:
        return self._spectral().grid_velocity

    def huygens_phasors(self, field: Any, /) -> tuple[HuygensSurfacePhasors, ...]:
        return self._spectral().huygens_phasors(field)


def distribute_pic_field_solver(
    base: AbstractPreparedPICFieldSolver,
    mesh: Mesh,
    /,
    *,
    guard_cells: int = 3,
    particle_margin: float = 1.0,
    identity_tiles: Sequence[int] | None = None,
) -> AbstractDistributedPICFieldSolver:
    """Execute ``base`` over ``mesh`` through its route-specific distributed solver.

    Distribution is admitted exactly as `pic_distribution_support` states; a
    refused base raises its stated reason.
    """
    support = pic_distribution_support(base)
    options: dict[str, Any] = {
        "guard_cells": guard_cells,
        "particle_margin": particle_margin,
        "identity_tiles": identity_tiles,
    }
    match support.route:
        case None:
            raise ValueError(support.basis)
        case "cochain":
            return _DistributedCochainPICFieldSolver(base, mesh, **options)
        case "reduced":
            return _DistributedReducedPICFieldSolver(base, mesh, **options)
        case "spectral-global-fft" | "spectral-local-guarded":
            return _DistributedSpectralPICFieldSolver(base, mesh, **options)
        case _:
            assert_never(support.route)


# -- particle execution ---------------------------------------------------------------


def _stack(value: Any, /) -> Any:
    """One device's values with a leading device row (shard_map stacks rows)."""
    return jax.tree.map(lambda leaf: jnp.asarray(leaf)[None], value)


def _bank_slot_leaves(bank: PICProcessBank, /) -> tuple[Array, ...]:
    return (*population_slot_leaves(bank.population), bank.position, *bank.leaves)


def _bank_from(
    leaves: Sequence[Array], counters: tuple[Array, Array], /
) -> PICProcessBank:
    population = assemble_population(leaves[:9], counters)
    return PICProcessBank(population, leaves[9], tuple(leaves[10:]))


def _empty_partition() -> PICProcessStatePartition:
    return PICProcessStatePartition((), (), (), (), companion_species=())


class DistributedPICExecutor(AbstractPICParticleExecutor, NonTrainableState):
    """Particle execution of one distributed PIC run.

    `PICDistributedProcess` processes run per device inside ``shard_map``:
    each device applies the process localized to its slot blocks (with a
    `PICIdentityAllocator` in its context) to its species blocks, slot-aligned
    process state, and bank blocks; additive process totals are accumulated per
    device and summed, and ledgers and evidence are combined by the process.
    Other (momentum-stage, stateless) processes run on the global arrays.
    ``exchange`` migrates species with their slot-aligned process state and
    banks by their own positions (`PICMigrationPlan`).
    """

    solver: AbstractDistributedPICFieldSolver
    migration: PICMigrationPlan
    processes: tuple[AbstractPICProcess, ...]
    local_processes: tuple[AbstractPICProcess | None, ...]
    species: tuple[PICSpeciesPlan, ...]
    local_species: tuple[PICSpeciesPlan, ...]

    def _partition(self, index: int, state: Any, /) -> PICProcessStatePartition:
        process = self.processes[index]
        if isinstance(process, PICDistributedProcess):
            return process.partition_state(state)
        return _empty_partition()

    def _assemble(
        self, index: int, partition: PICProcessStatePartition, state: Any, /
    ) -> Any:
        process = self.processes[index]
        if isinstance(process, PICDistributedProcess):
            return process.assemble_state(partition)
        return state

    def _bank_plans(self, index: int, /) -> tuple[ParticlePopulationPlan, ...]:
        process = self.processes[index]
        if isinstance(process, PICDistributedProcess):
            return process.bank_plans()
        return ()

    @eqx.filter_jit
    def apply_process(
        self,
        index: int,
        process: AbstractPICProcess,
        species: tuple[PICSpeciesPlan, ...],
        context: PICProcessContext,
        /,
    ) -> PICProcessResult:
        local = self.local_processes[index]
        if local is None or not isinstance(process, PICDistributedProcess):
            return process.apply(species, context)
        if not isinstance(local, PICDistributedProcess):
            raise TypeError("A localized distributed process lost its protocol.")
        solver = self.solver
        parts = solver.decomposition.part_count
        partition = process.partition_state(context.state)
        probe = context.probe
        slot_in = (
            tuple(species_slot_leaves(value) for value in context.species),
            context.electric,
            context.magnetic,
            context.step_start_proper_velocity,
            None if probe is None else (probe.electric, probe.magnetic),
            partition.companions,
            tuple(_bank_slot_leaves(bank) for bank in partition.banks),
        )
        replicated_in = (
            tuple(
                (value.population.next_id_hi, value.population.next_id_lo)
                for value in context.species
            ),
            tuple(
                (bank.population.next_id_hi, bank.population.next_id_lo)
                for bank in partition.banks
            ),
            partition.shared,
            None if probe is None else probe.successful,
            (
                context.time,
                context.step_size,
                context.step_index,
                context.key,
                context.grid_cutoff_frequency,
                context.grid_velocity,
            ),
        )
        # Device 0 carries the running totals; every device adds its increments.
        stacked_in = tuple(
            jnp.concatenate(
                (value[None], jnp.zeros((parts - 1, *value.shape), dtype=value.dtype))
            )
            for value in partition.totals
        )
        pusher = context.pusher
        companion_species = partition.companion_species

        def device(slot: Any, replicated: Any, stacked: Any) -> tuple[Any, Any]:
            leaves, electric, magnetic, start, probed, companions, banks = slot
            counters, bank_counters, shared, probe_ok, scalars = replicated
            time, step_size, step_index, key, cutoff, grid_velocity = scalars
            states = tuple(
                assemble_species(value, counter)
                for value, counter in zip(leaves, counters, strict=True)
            )
            state = process.assemble_state(
                PICProcessStatePartition(
                    companions,
                    tuple(
                        _bank_from(value, counter)
                        for value, counter in zip(banks, bank_counters, strict=True)
                    ),
                    tuple(value[0] for value in stacked),
                    shared,
                    companion_species=companion_species,
                )
            )
            result = local.apply(
                self.local_species,
                PICProcessContext(
                    states,
                    electric,
                    magnetic,
                    time,
                    step_size,
                    step_index,
                    key,
                    None,
                    None,
                    None,
                    None,
                    cutoff,
                    state,
                    None
                    if probed is None or probe_ok is None
                    else PICFieldProbeSample(probed[0], probed[1], probe_ok),
                    pusher,
                    start,
                    grid_velocity,
                    PICIdentityAllocator(solver.decomposition, solver.coordinates()),
                ),
            )
            after = local.partition_state(result.state)
            return (
                (
                    tuple(species_slot_leaves(value) for value in result.species),
                    after.companions,
                    tuple(_bank_slot_leaves(bank) for bank in after.banks),
                ),
                _stack(
                    (
                        tuple(
                            (value.population.next_id_hi, value.population.next_id_lo)
                            for value in result.species
                        ),
                        tuple(
                            (bank.population.next_id_hi, bank.population.next_id_lo)
                            for bank in after.banks
                        ),
                        after.totals,
                        after.shared,
                        result.ledger,
                        result.evidence,
                    )
                ),
            )

        slot, rows = solver.slot_spec(), PartitionSpec(solver.axis_names)
        slot_out, stacked_out = solver.map(
            device, (slot, PartitionSpec(), rows), (slot, rows)
        )(slot_in, replicated_in, stacked_in)
        leaves, companions, banks = slot_out
        counters, bank_counters, totals, shared, ledger, evidence = stacked_out
        # Every device advances the replicated counters identically.
        species_out = tuple(
            assemble_species(value, (hi[0], lo[0]))
            for value, (hi, lo) in zip(leaves, counters, strict=True)
        )
        state = process.assemble_state(
            PICProcessStatePartition(
                companions,
                tuple(
                    _bank_from(value, (hi[0], lo[0]))
                    for value, (hi, lo) in zip(banks, bank_counters, strict=True)
                ),
                tuple(jnp.sum(value, axis=0) for value in totals),
                tuple(value[0] for value in shared),
                companion_species=companion_species,
            )
        )
        ledger, evidence = process.combine(ledger, evidence)
        return PICProcessResult(species_out, ledger, evidence, state)

    def _groups(
        self, species: tuple[PICSpeciesState, ...], states: tuple[Any, ...], /
    ) -> tuple[
        tuple[PICSlotGroup, ...],
        tuple[ParticlePopulationPlan, ...],
        tuple[PICProcessStatePartition, ...],
        tuple[tuple[tuple[int, int], ...], ...],
    ]:
        """Slot groups: species with their companions, then process banks."""
        partitions = tuple(
            self._partition(index, state) for index, state in enumerate(states)
        )
        owners: list[list[tuple[int, int]]] = [[] for _ in species]
        for index, partition in enumerate(partitions):
            for position, target in enumerate(partition.companion_species):
                owners[target].append((index, position))
        groups, plans = [], []
        for plan, value, owned in zip(self.species, species, owners, strict=True):
            charge = value.charge
            groups.append(
                PICSlotGroup(
                    value.population,
                    value.particles.position,
                    (value.particles.proper_velocity,),
                    (
                        charge.charge_number,
                        charge.transition_count,
                        charge.last_transition_step,
                        *(
                            leaf
                            for index, position in owned
                            for leaf in partitions[index].companions[position]
                        ),
                    ),
                    plan.population.particles.active_mask,
                )
            )
            plans.append(plan.population)
        for index, partition in enumerate(partitions):
            for bank, plan in zip(partition.banks, self._bank_plans(index), strict=True):
                groups.append(
                    PICSlotGroup(
                        bank.population,
                        bank.position,
                        (),
                        bank.leaves,
                        plan.particles.active_mask,
                    )
                )
                plans.append(plan)
        return (
            tuple(groups),
            tuple(plans),
            partitions,
            tuple(tuple(value) for value in owners),
        )

    def _ungroup(
        self,
        groups: Sequence[PICSlotGroup],
        species: tuple[PICSpeciesState, ...],
        states: tuple[Any, ...],
        partitions: tuple[PICProcessStatePartition, ...],
        owners: tuple[tuple[tuple[int, int], ...], ...],
        /,
    ) -> tuple[tuple[PICSpeciesState, ...], tuple[Any, ...]]:
        companions = [list(value.companions) for value in partitions]
        species_out = []
        for group, value, owned in zip(
            groups[: len(species)], species, owners, strict=True
        ):
            kept = group.kept
            species_out.append(
                PICSpeciesState(
                    type(value.particles)(group.position, group.cleared[0]),
                    group.population,
                    type(value.charge)(kept[0], kept[1], kept[2]),
                )
            )
            offset = 3
            for index, position in owned:
                count = len(partitions[index].companions[position])
                companions[index][position] = tuple(kept[offset : offset + count])
                offset += count
        rest = list(groups[len(species) :])
        states_out = []
        for index, (partition, state) in enumerate(zip(partitions, states, strict=True)):
            banks = tuple(
                PICProcessBank(group.population, group.position, group.kept)
                for group in rest[: len(partition.banks)]
            )
            rest = rest[len(partition.banks) :]
            states_out.append(
                self._assemble(
                    index,
                    PICProcessStatePartition(
                        tuple(companions[index]),
                        banks,
                        partition.totals,
                        partition.shared,
                        companion_species=partition.companion_species,
                    ),
                    state,
                )
            )
        return tuple(species_out), tuple(states_out)

    @eqx.filter_jit
    def exchange(
        self, species: tuple[PICSpeciesState, ...], processes: tuple[Any, ...], /
    ) -> PICParticleExchangeResult:
        """Route particles to their owners; unchanged unless every device succeeds."""
        solver = self.solver
        migration = self.migration
        groups, plans, partitions, owners = self._groups(species, processes)
        slot_in = tuple(
            (
                population_slot_leaves(group.population),
                group.position,
                group.cleared,
                group.kept,
                group.structural,
            )
            for group in groups
        )
        counters = tuple(
            (group.population.next_id_hi, group.population.next_id_lo) for group in groups
        )

        def device(
            slot: Any, counters: Any
        ) -> tuple[tuple[Any, ...], PICMigrationEvidence]:
            local_groups = tuple(
                PICSlotGroup(
                    assemble_population(population, (hi, lo)),
                    position,
                    cleared,
                    kept,
                    structural,
                )
                for (population, position, cleared, kept, structural), (hi, lo) in zip(
                    slot, counters, strict=True
                )
            )
            migrated, evidence = migration.migrate_local(
                plans,
                local_groups,
                solver.coordinates(),
                axis_names=solver.axis_names,
            )
            return (
                tuple(
                    (
                        population_slot_leaves(group.population),
                        group.position,
                        group.cleared,
                        group.kept,
                        group.structural,
                    )
                    for group in migrated
                ),
                evidence,
            )

        slot = solver.slot_spec()
        slot_out, evidence = solver.map(
            device, (slot, PartitionSpec()), (slot, PartitionSpec())
        )(slot_in, counters)
        moved = tuple(
            PICSlotGroup(
                with_population_slot_leaves(group.population, population),
                position,
                cleared,
                kept,
                structural,
            )
            for group, (population, position, cleared, kept, structural) in zip(
                groups, slot_out, strict=True
            )
        )
        species_out, states_out = self._ungroup(
            moved, species, processes, partitions, owners
        )
        return PICParticleExchangeResult(
            species_out, states_out, evidence, evidence.successful
        )


class DistributedPICStepResult(StrictModule):
    """One distributed step: the core step, then migration, committed atomically."""

    candidate_state: ElectromagneticPICState
    accepted_state: ElectromagneticPICState
    diagnostics: ElectromagneticPICDiagnostics
    current: Any
    migration: PICMigrationEvidence
    rejection_reason: Array
    successful: Array


def _select(predicate: Array, candidate: Any, current: Any, /) -> Any:
    return jax.tree.map(
        lambda proposed, old: jnp.where(predicate, proposed, old), candidate, current
    )


def _needs_devices(process: AbstractPICProcess, /) -> bool:
    return process.stage != "momentum" or process.stateful or process.redistributes_charge


class DistributedElectromagneticPICPlan(StrictModule, NonTrainableState):
    """Explicit EM PIC with owner-computes particles and atomic ppermute migration.

    Creation-stage, population-stage, stateful, and charge-redistributing
    processes must implement `PICDistributedProcess`; they run per device
    through the installed `DistributedPICExecutor`. ``pic`` is the given plan
    with that executor installed (same plan identity).
    """

    pic: ElectromagneticPICPlan
    migration: PICMigrationPlan
    plan_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        pic: ElectromagneticPICPlan,
        /,
        *,
        packet_capacity: int,
        reach: int = 1,
    ) -> None:
        solver = pic.solver
        if not isinstance(solver, AbstractDistributedPICFieldSolver):
            raise TypeError(
                "Distributed PIC requires a solver from distribute_pic_field_solver."
            )
        for process in pic.processes:
            if _needs_devices(process) and not isinstance(process, PICDistributedProcess):
                raise ValueError(
                    f"Process {process.process_id!r} creates, redistributes, or "
                    "carries per-particle state but does not implement "
                    "PICDistributedProcess."
                )
            if process.redistributes_charge and not isinstance(
                solver.base, PICGaussProjection
            ):
                raise TypeError(
                    "Charge-redistributing processes require a base field solver "
                    "implementing PICGaussProjection."
                )
        decomposition = solver.decomposition
        if pic.maximum_displacement_fraction > decomposition.particle_margin:
            raise ValueError(
                "The PIC displacement limit exceeds the distributed particle margin."
            )
        parts = decomposition.part_count
        local_species = tuple(_local_species(plan, parts) for plan in pic.species)
        local_processes = tuple(
            process.localize(local_species, parts)
            if isinstance(process, PICDistributedProcess) and _needs_devices(process)
            else None
            for process in pic.processes
        )
        migration = PICMigrationPlan(
            decomposition, packet_capacity=packet_capacity, reach=reach
        )
        executor = DistributedPICExecutor(
            solver,
            migration,
            pic.processes,
            local_processes,
            pic.species,
            local_species,
        )
        self.pic = eqx.tree_at(
            lambda value: value.executor,
            pic,
            executor,
            is_leaf=lambda value: value is None,
        )
        self.migration = migration
        self.topology_id = canonical_fingerprint(
            {
                "kind": "distributed-pic-topology",
                "execution": solver.evidence.execution_id,
                "migration": migration.plan_id,
            }
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-electromagnetic-pic",
                "pic": pic.plan_id,
                "topology": self.topology_id,
            }
        )

    @property
    def solver(self) -> AbstractDistributedPICFieldSolver:
        solver = self.pic.solver
        if not isinstance(solver, AbstractDistributedPICFieldSolver):
            raise TypeError("Distributed PIC lost its distributed field solver.")
        return solver

    @property
    def executor(self) -> DistributedPICExecutor:
        executor = self.pic.executor
        if not isinstance(executor, DistributedPICExecutor):
            raise TypeError("Distributed PIC lost its DistributedPICExecutor.")
        return executor

    @property
    def evidence(self) -> PICDistributedEvidence:
        return self.solver.evidence

    # -- placement ----------------------------------------------------------------

    def _map_state(
        self,
        state: ElectromagneticPICState,
        slots: Callable[[Array, PartitionSpec], Array],
        /,
    ) -> ElectromagneticPICState:
        """Apply ``slots(leaf, spec)`` with each leaf's canonical partition spec."""
        solver = self.solver
        executor = self.executor
        slot, planes = solver.slot_spec(), solver.plane_spec()

        def replicated(value: Array) -> Array:
            return slots(value, PartitionSpec())

        def blocks(value: Array) -> Array:
            return slots(value, slot)

        def grid(value: Array) -> Array:
            return slots(value, planes)

        mapped = jax.tree.map(replicated, state)
        species = tuple(
            with_species_slot_leaves(
                other, tuple(blocks(value) for value in species_slot_leaves(original))
            )
            for original, other in zip(state.species, mapped.species, strict=True)
        )
        processes = []
        for index, value in enumerate(state.processes):
            partition = executor._partition(index, value)
            processes.append(
                executor._assemble(
                    index,
                    PICProcessStatePartition(
                        jax.tree.map(blocks, partition.companions),
                        tuple(
                            _bank_from(
                                tuple(blocks(leaf) for leaf in _bank_slot_leaves(bank)),
                                (
                                    replicated(bank.population.next_id_hi),
                                    replicated(bank.population.next_id_lo),
                                ),
                            )
                            for bank in partition.banks
                        ),
                        jax.tree.map(replicated, partition.totals),
                        jax.tree.map(replicated, partition.shared),
                        companion_species=partition.companion_species,
                    ),
                    jax.tree.map(replicated, value),
                )
            )
        mapped = eqx.tree_at(lambda value: value.species, mapped, species)
        mapped = eqx.tree_at(
            lambda value: value.processes,
            mapped,
            tuple(processes),
            is_leaf=lambda value: value is None,
        )
        mapped = eqx.tree_at(
            lambda value: value.field,
            mapped,
            solver.layout.map_field(state.field, grid, replicated),
        )
        history = state.field_history
        if history is not None:
            mapped = eqx.tree_at(
                lambda value: value.field_history.field,
                mapped,
                solver.layout.map_field(history.field, grid, replicated),
            )
        return mapped

    def place(self, state: ElectromagneticPICState, /) -> ElectromagneticPICState:
        """Device-put a state with its canonical slot and grid shardings."""
        mesh = self.solver.mesh

        def put(value: Array, spec: PartitionSpec) -> Array:
            return jax.device_put(value, NamedSharding(mesh, spec))

        return self._map_state(state, put)

    def _constrain(self, state: ElectromagneticPICState, /) -> ElectromagneticPICState:
        mesh = self.solver.mesh

        def constrain(value: Array, spec: PartitionSpec) -> Array:
            sharding: Sharding = NamedSharding(mesh, spec)
            return jax.lax.with_sharding_constraint(value, sharding)

        return self._map_state(state, constrain)

    def repartition(self, state: ElectromagneticPICState, /) -> ElectromagneticPICState:
        """Move every particle into its owner's slot block (a slot permutation).

        Species move with their slot-aligned process state; process banks move
        by their own positions. Used when a run restarts on a different mesh;
        refuses a state whose particles do not fit the per-device slot blocks.
        """
        executor = self.executor
        decomposition = self.solver.decomposition
        groups, _, partitions, owners = executor._groups(state.species, state.processes)

        def permuted(group: PICSlotGroup, name: str) -> PICSlotGroup:
            destination, fits = repartition_destinations(
                decomposition, group.position, group.population.active
            )
            if not bool(fits):
                raise ValueError(f"{name} does not fit its per-device slot blocks.")
            return PICSlotGroup(
                with_population_slot_leaves(
                    group.population,
                    tuple(
                        permute_slots(leaf, destination)
                        for leaf in population_slot_leaves(group.population)
                    ),
                ),
                permute_slots(group.position, destination),
                tuple(permute_slots(leaf, destination) for leaf in group.cleared),
                tuple(permute_slots(leaf, destination) for leaf in group.kept),
                group.structural,
            )

        moved = tuple(
            permuted(
                group,
                f"Species {index}" if index < len(state.species) else "A process bank",
            )
            for index, group in enumerate(groups)
        )
        species, processes = executor._ungroup(
            moved, state.species, state.processes, partitions, owners
        )
        state = eqx.tree_at(lambda item: item.species, state, species)
        state = eqx.tree_at(
            lambda item: item.processes,
            state,
            processes,
            is_leaf=lambda value: value is None,
        )
        return self.place(state)

    # -- lifecycle ------------------------------------------------------------------

    def initialize(
        self,
        positions: Sequence[ArrayLike],
        velocities: Sequence[ArrayLike],
        step_size: ArrayLike,
        /,
        *,
        active_masks: Sequence[ArrayLike | None] | None = None,
        masses: Sequence[ArrayLike | None] | None = None,
        magnetic: Any = None,
        time: ArrayLike = 0.0,
    ) -> ElectromagneticPICState:
        """Initial state with each particle in its owner's slot block.

        Persistent identities follow the caller's slot order, as in a
        single-device run, so identity-addressed randomness and diagnostics are
        independent of the decomposition. Traceable: an owner whose particles
        exceed its slot block fails through ``eqx.error_if``.
        """
        pic = self.pic
        count = len(pic.species)
        masks = (None,) * count if active_masks is None else tuple(active_masks)
        mass_values = (None,) * count if masses is None else tuple(masses)
        position_values, velocity_values = tuple(positions), tuple(velocities)
        if (
            not (
                len(position_values)
                == len(velocity_values)
                == len(masks)
                == len(mass_values)
            )
            or len(position_values) != count
        ):
            raise ValueError("One position and velocity array is required per species.")
        dtype = jnp.dtype(pic.precision.particle_dtype)
        destinations = []
        placed_positions, placed_velocities, placed_masks, placed_masses = [], [], [], []
        for index, (plan, position, velocity, mask, mass) in enumerate(
            zip(
                pic.species,
                position_values,
                velocity_values,
                masks,
                mass_values,
                strict=True,
            )
        ):
            position_ = jnp.asarray(position, dtype=dtype)
            structural = plan.population.particles.active_mask
            active = (
                structural
                if mask is None
                else structural & jnp.asarray(mask, dtype=jnp.bool_)
            )
            if position_.ndim != 2 or position_.shape[0] != plan.capacity:
                raise ValueError("PIC positions must have capacity-by-dimension shape.")
            destination, fits = repartition_destinations(
                self.solver.decomposition, position_, active
            )
            position_ = eqx.error_if(
                position_,
                ~fits,
                f"Species {index} does not fit its per-device slot blocks.",
            )
            destinations.append(destination)
            placed_positions.append(permute_slots(position_, destination))
            placed_velocities.append(
                permute_slots(jnp.asarray(velocity, dtype=dtype), destination)
            )
            placed_masks.append(permute_slots(active, destination))
            placed_masses.append(
                None if mass is None else permute_slots(jnp.asarray(mass), destination)
            )
        state = pic.initialize(
            placed_positions,
            placed_velocities,
            step_size,
            active_masks=placed_masks,
            masses=placed_masses,
            magnetic=magnetic,
            time=time,
        )
        species = tuple(
            eqx.tree_at(
                lambda value: value.population,
                value,
                assemble_population(
                    tuple(
                        permute_slots(leaf, destination)
                        for leaf in population_slot_leaves(
                            plan.population.initialize(active_mask=mask, masses=mass)
                        )
                    ),
                    (value.population.next_id_hi, value.population.next_id_lo),
                ),
            )
            for plan, value, mask, mass, destination in zip(
                pic.species, state.species, masks, mass_values, destinations, strict=True
            )
        )
        state = eqx.tree_at(lambda value: value.species, state, species)
        state = eqx.tree_at(
            lambda value: value.recorders,
            state,
            tuple(recorder.initialize(species, state.time) for recorder in pic.recorders),
        )
        state = eqx.tree_at(
            lambda value: value.processes,
            state,
            pic.initialize_process_states(species),
            is_leaf=lambda value: value is None,
        )
        return self.place(state)

    def migrate(
        self, state: ElectromagneticPICState, /
    ) -> tuple[ElectromagneticPICState, PICMigrationEvidence]:
        """Route particles to their owners; unchanged unless every device succeeds."""
        exchange = self.executor.exchange(state.species, state.processes)
        migrated = eqx.tree_at(lambda value: value.species, state, exchange.species)
        migrated = eqx.tree_at(
            lambda value: value.processes,
            migrated,
            exchange.processes,
            is_leaf=lambda value: value is None,
        )
        return migrated, exchange.evidence

    def step_detailed(
        self, state: ElectromagneticPICState, step_size: ArrayLike, /
    ) -> DistributedPICStepResult:
        """One PIC step and migration; either both commit or the state is kept."""
        result = self.pic.step_detailed(state, step_size)
        migrated_state, migration = self.migrate(result.accepted_state)
        migrated = migration.successful
        successful = result.successful & migrated
        accepted = _select(migrated, migrated_state, state)
        candidate = eqx.tree_at(
            lambda value: value.status,
            result.candidate_state,
            jnp.where(
                successful,
                int(PICRunStatus.SUCCESS),
                int(PICRunStatus.INVALID_STATE),
            ).astype(jnp.int32),
        )
        reason = result.diagnostics.rejection_reason
        reason = jnp.where(migrated, reason, reason | int(PICRejectionReason.MIGRATION))
        return DistributedPICStepResult(
            candidate,
            self._constrain(accepted),
            result.diagnostics,
            result.current,
            migration,
            reason.astype(jnp.int32),
            successful,
        )


__all__ = [
    "AbstractDistributedPICFieldSolver",
    "DistributedElectromagneticPICPlan",
    "DistributedPICExecutor",
    "DistributedPICStepResult",
    "distribute_pic_field_solver",
    "pic_distribution_support",
    "PICDistributedEvidence",
    "PICDistributedRoute",
    "PICDistributionSupport",
]
