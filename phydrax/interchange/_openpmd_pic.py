#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""openPMD 1.1.0 export, import, and bounded streaming output of PIC states.

`OpenPMDPICLayout` binds one `ElectromagneticPICPlan` over a structured
Cartesian field solver to openPMD records. One PIC state becomes one file of a
``fileBased`` series: the ``E``, ``B``, and ``rho`` (plus the step's ``J``)
mesh records on the solver grid with their exact staggering, and one particle
species per PIC species. The bound scale's units are the run's code units;
records are stored in them with ``unitSI``/``unitDimension`` from the scale.

Leapfrog timing is explicit: positions, fields, and charge sit at the
iteration time; proper velocities and the current are half a step earlier
(``timeOffset = -dt/2``). Particles are written in ascending ``uint64``
identity order with a ``parentId`` lineage record; ``weighting`` is the
macroparticle mass over the declared single-particle rest mass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from numbers import Real
from pathlib import Path
from typing import Any, TYPE_CHECKING

import h5py
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._external_resource import (
    account_bounded_resource,
    bounded_resource_from_bytes,
    BoundedResource,
    ResourceLimits,
    ResourceReadError,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import ElectromagneticScaleContract
from .._publication import publish_bytes
from ..typing import checked
from ._openpmd_base import (
    BoundedHDF5Buffer,
    component_shape,
    component_values,
    identity_values,
    OPENPMD_MESH_REVISION,
    OpenPMDDimension,
    OpenPMDIterationTime,
    OpenPMDUnit,
    OpenPMDUnsupportedError,
    preflight_hdf5,
    read_record_unit,
    scalar_attribute,
    weighting_metadata,
    write_component,
    write_iteration,
    write_particle_metadata,
    write_record_unit,
    write_series_root,
)
from ._openpmd_mesh import (
    CARTESIAN_AXES,
    convert_openpmd_hdf5_to_adios2,
    decode_mesh_record,
    MESHES_PATH,
    open_iteration,
    OpenPMDADIOS2Provider,
    OpenPMDFieldRecordName,
    OpenPMDMeshImportPolicy,
    OpenPMDMeshIteration,
    OpenPMDMeshRecord,
    record_dimension,
    records_group,
    refusal,
    require_encoding_budget,
    scale_unit,
    scan_mesh_record,
    series_name,
    write_mesh_record,
)
from ._report import AdapterLoss, AdapterReport, AdapterStatus


if TYPE_CHECKING:
    from ..discretization.pic import PICSpeciesPlan, PICSpeciesState
    from ..solver import (
        CochainMaxwellPICFieldSolver,
        ElectromagneticPICPlan,
        ElectromagneticPICState,
        ElectromagneticPICStepResult,
        ReducedMaxwellPICFieldSolver,
    )


_REVISION = OPENPMD_MESH_REVISION
_FORMAT = "openPMD-1.1.0-PIC-HDF5"
_PARTICLES_PATH = "particles/"
_DIMENSIONLESS: OpenPMDDimension = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_NO_IDENTITY = np.uint64(2**64 - 1)
_REQUIRED_RECORDS: tuple[OpenPMDFieldRecordName, ...] = ("E", "B", "rho")
_ASSUMPTIONS = (
    "openPMD 1.1.0 fileBased PIC iteration: E, B, rho, optional J, one species "
    "per PIC species",
    "positions, fields, and charge at the iteration time; momenta and J at "
    "timeOffset -dt/2",
    "the bound scale's units are the PIC run's code units",
)
# Scale quantity of each particle record; ``None`` marks a dimensionless record.
_PARTICLE_RECORDS: dict[str, str | None] = {
    "position": "length",
    "positionOffset": "length",
    "momentum": "momentum",
    "weighting": None,
    "charge": "charge",
    "mass": "mass",
    "id": None,
    "parentId": None,
}
# Host words per stored particle: position and offset (2d), momentum (3),
# weighting, charge, mass, identity, parent; and per capacity slot of the
# rebuilt state: position (d), proper velocity (3), mass, charge number,
# identity and parent words (4).
_PARTICLE_WORDS = 8 + 2 * 3
_SLOT_WORDS = 3 + 3 + 2 + 4


class _StateMismatchError(ValueError):
    """Well-formed records that contradict the bound PIC plan."""


@dataclass(frozen=True, slots=True)
class _MeshGrid:
    """Cartesian openPMD grid of one structured PIC field solver.

    Exactly one of ``cochain`` and ``reduced`` holds the bound solver.
    """

    cochain: CochainMaxwellPICFieldSolver | None
    reduced: ReducedMaxwellPICFieldSolver | None
    labels: tuple[str, ...]
    spacing: tuple[float, ...]
    offset: tuple[float, ...]
    shape: tuple[int, ...]
    staggering: dict[OpenPMDFieldRecordName, tuple[tuple[float, ...], ...]]


def _cochain_grid(solver: CochainMaxwellPICFieldSolver, /) -> _MeshGrid:
    spacing, offset, shape = [], [], []
    for axis in solver.bridge.grid.structured_axes:
        points = np.asarray(axis.point_coordinates, dtype=np.float64)
        widths = np.asarray(axis.interval_widths, dtype=np.float64)
        centers = np.asarray(axis.interval_centers, dtype=np.float64)
        if (
            points.size != widths.size
            or not np.allclose(widths, widths[0], rtol=1.0e-12, atol=0.0)
            or not np.allclose(centers, points + 0.5 * widths, rtol=1.0e-12, atol=0.0)
        ):
            raise OpenPMDUnsupportedError(
                "openPMD meshes require uniform periodic cochain axes."
            )
        spacing.append(float(widths[0]))
        offset.append(float(points[0]))
        shape.append(points.size)
    kernel = solver.transfers[0].kernel
    edge = kernel.entity_offsets(1, proxy="circulation")
    face = kernel.entity_offsets(2, proxy="flux")
    return _MeshGrid(
        solver,
        None,
        CARTESIAN_AXES,
        tuple(spacing),
        tuple(offset),
        tuple(shape),
        {"E": edge, "B": face, "J": edge, "rho": kernel.entity_offsets(0)},
    )


def _reduced_grid(solver: ReducedMaxwellPICFieldSolver, /) -> _MeshGrid:
    transfer = solver.transfer
    rank = transfer.dimension
    # The reduced CIC transfer deposits and gathers every component at cells.
    center = (0.5,) * rank
    return _MeshGrid(
        None,
        solver,
        CARTESIAN_AXES[:rank],
        tuple(float(value) for value in transfer.spacing),
        tuple(float(value) for value in transfer.lower),
        tuple(transfer.shape),
        {
            "E": (center,) * 3,
            "B": (center,) * 3,
            "J": (center,) * 3,
            "rho": (center,),
        },
    )


def _mesh_grid(plan: ElectromagneticPICPlan, /) -> _MeshGrid:
    from ..solver import CochainMaxwellPICFieldSolver, ReducedMaxwellPICFieldSolver

    solver = plan.solver
    if isinstance(solver, CochainMaxwellPICFieldSolver):
        return _cochain_grid(solver)
    if isinstance(solver, ReducedMaxwellPICFieldSolver):
        return _reduced_grid(solver)
    raise OpenPMDUnsupportedError(
        "openPMD PIC meshes require the cochain or reduced Cartesian field solver."
    )


@dataclass(frozen=True, slots=True, eq=False)
class OpenPMDPICLayout:
    """Binding of one `ElectromagneticPICPlan` to openPMD records.

    ``scale`` declares the run's code units (its speed of light must equal the
    pusher's) and supplies every ``unitSI``. ``particle_masses[s]`` is the rest
    mass of one real particle of species ``s`` in the scale mass unit; the
    openPMD ``weighting`` is the macroparticle mass divided by it and the
    per-particle charge is ``particle_mass * base_specific_charge *
    charge_number``. Species are named by their ``species_id``.
    """

    plan: ElectromagneticPICPlan
    scale: ElectromagneticScaleContract
    particle_masses: Sequence[float]
    layout_id: str = field(init=False)
    grid: _MeshGrid = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from ..solver import ElectromagneticPICPlan

        if not isinstance(self.plan, ElectromagneticPICPlan):
            raise TypeError("plan must be an ElectromagneticPICPlan.")
        if not isinstance(self.scale, ElectromagneticScaleContract):
            raise TypeError("scale must be an ElectromagneticScaleContract.")
        if isinstance(self.particle_masses, (str, bytes)) or not isinstance(
            self.particle_masses, Sequence
        ):
            raise TypeError("particle_masses must be a sequence of rest masses.")
        if any(not isinstance(value, Real) for value in self.particle_masses):
            raise TypeError("particle_masses must contain real rest masses.")
        masses = tuple(float(value) for value in self.particle_masses)
        if len(masses) != len(self.plan.species):
            raise ValueError("particle_masses must give one rest mass per PIC species.")
        if any(not np.isfinite(value) or value <= 0.0 for value in masses):
            raise ValueError("particle_masses must be finite and positive.")
        if not np.isclose(
            float(self.scale.speed_of_light),
            self.plan.pusher.speed_of_light,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(
                "The scale's speed of light differs from the PIC pusher's; the scale "
                "must declare the run's code units."
            )
        for species in self.plan.species:
            series_name(species.species_id)
        # unit_si_map refuses scales that are not referenced to SI.
        self.scale.unit_si_map()
        grid = _mesh_grid(self.plan)
        object.__setattr__(self, "particle_masses", masses)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(
            self,
            "layout_id",
            canonical_fingerprint(
                {
                    "kind": "openpmd-pic-layout",
                    "standard": _REVISION.version,
                    "plan": self.plan.plan_id,
                    "scale": self.scale.scale_id,
                    "particle_masses": list(masses),
                }
            ),
        )


@dataclass(frozen=True, slots=True, eq=False)
class OpenPMDPICExportResult:
    path: Path
    iteration: int
    resource: BoundedResource
    report: AdapterReport


@dataclass(frozen=True, slots=True, eq=False)
class OpenPMDPICImportResult:
    """Imported PIC state, its step size, and the decoded mesh records.

    ``step_size`` is the iteration ``dt`` in the scale time unit, the step the
    half-step proper velocities and the current were staggered with.
    """

    state: ElectromagneticPICState
    step_size: float
    meshes: OpenPMDMeshIteration
    resource: BoundedResource
    report: AdapterReport


# Export -----------------------------------------------------------------------------


def _identity(hi: np.ndarray, lo: np.ndarray, /) -> np.ndarray:
    return (hi.astype(np.uint64) << np.uint64(32)) | lo.astype(np.uint64)


def _field_components(
    layout: OpenPMDPICLayout, value: Any, /
) -> dict[OpenPMDFieldRecordName, tuple[Any, ...]]:
    cochain = layout.grid.cochain
    if cochain is not None:
        bridge = cochain.bridge
        return {
            "E": bridge.unpack_edge_circulation(cochain.maxwell.electric_field(value)),
            "B": bridge.unpack_face_flux(cochain.maxwell.magnetic_flux(value)),
            "rho": bridge.unpack(0, value.primary.charge),
        }
    return {
        "E": tuple(value.electric),
        "B": tuple(value.magnetic),
        "rho": (value.charge,),
    }


def _current_components(layout: OpenPMDPICLayout, current: Any, /) -> tuple[Any, ...]:
    cochain = layout.grid.cochain
    if cochain is not None:
        return cochain.bridge.unpack_edge_circulation(current)
    return tuple(current)


def _mesh_records(
    layout: OpenPMDPICLayout, state: Any, current: Any, step_size: float, /
) -> tuple[OpenPMDMeshRecord, ...]:
    grid = layout.grid
    components = _field_components(layout, state.field)
    if current is not None:
        components["J"] = _current_components(layout, current)
    return tuple(
        OpenPMDMeshRecord(
            name,
            "cartesian",
            grid.labels,
            grid.spacing,
            grid.offset,
            tuple(np.asarray(value, dtype=np.float64) for value in values),
            grid.staggering[name],
            -0.5 * step_size if name == "J" else 0.0,
        )
        for name, values in components.items()
    )


@dataclass(frozen=True, slots=True)
class _SpeciesArrays:
    identities: np.ndarray
    parents: np.ndarray
    positions: np.ndarray
    proper_velocities: np.ndarray
    macro_masses: np.ndarray
    charge_numbers: np.ndarray


def _species_arrays(state: PICSpeciesState, /) -> _SpeciesArrays:
    """Active particles of one species in ascending identity order."""
    population = state.population
    active = np.asarray(population.active, dtype=np.bool_)
    identities = _identity(
        np.asarray(population.id_hi)[active], np.asarray(population.id_lo)[active]
    )
    order = np.argsort(identities, kind="stable")
    return _SpeciesArrays(
        identities[order],
        _identity(
            np.asarray(population.parent_hi)[active],
            np.asarray(population.parent_lo)[active],
        )[order],
        np.asarray(state.particles.position, dtype=np.float64)[active][order],
        np.asarray(state.particles.proper_velocity, dtype=np.float64)[active][order],
        np.asarray(population.mass, dtype=np.float64)[active][order],
        np.asarray(state.charge.charge_number, dtype=np.float64)[active][order],
    )


def _write_species(
    group: h5py.Group,
    arrays: _SpeciesArrays,
    particle_mass: float,
    specific_charge: float,
    labels: tuple[str, ...],
    units: dict[str, tuple[float, tuple[float, ...]]],
    step_size: float,
    /,
) -> None:
    length = scale_unit(units, "length")
    count = arrays.identities.shape[0]
    for name, values, unit, power, offset, axes in (
        ("position", arrays.positions, length, 0.0, 0.0, labels),
        ("positionOffset", np.zeros_like(arrays.positions), length, 0.0, 0.0, labels),
        (
            "momentum",
            particle_mass * arrays.proper_velocities,
            scale_unit(units, "momentum"),
            1.0,
            -0.5 * step_size,
            CARTESIAN_AXES,
        ),
    ):
        record = group.create_group(name)
        for axis, axis_name in enumerate(axes):
            component = write_component(record, axis_name, values[:, axis])
            write_record_unit(record.attrs, component.attrs, unit)
        write_particle_metadata(record.attrs, 0, power, offset)
    dimensionless = OpenPMDUnit(1.0, _DIMENSIONLESS)
    for name, values, unit, macro, power in (
        ("weighting", arrays.macro_masses / particle_mass, dimensionless, 1, 1.0),
        (
            "charge",
            particle_mass * specific_charge * arrays.charge_numbers,
            scale_unit(units, "charge"),
            0,
            1.0,
        ),
        ("mass", np.full(count, particle_mass), scale_unit(units, "mass"), 0, 1.0),
    ):
        # A scalar record is its own component.
        component = write_component(group, name, values)
        write_record_unit(component.attrs, component.attrs, unit)
        write_particle_metadata(component.attrs, macro, power, 0.0)
    for name, values in (("id", arrays.identities), ("parentId", arrays.parents)):
        # Identities stay uint64 datasets: constant components would carry them
        # through floating-point attribute values in some openPMD providers.
        dataset = group.create_dataset(name, data=values.astype(np.uint64))
        write_record_unit(dataset.attrs, dataset.attrs, dimensionless)
        write_particle_metadata(dataset.attrs, 0, 0.0, 0.0)


def _export_losses(layout: OpenPMDPICLayout, state: Any, /) -> tuple[AdapterLoss, ...]:
    plan = layout.plan
    losses = [
        AdapterLoss(
            "species/charge",
            "export",
            "dropped",
            "Charge-transition counters and last-transition steps have no openPMD "
            "record.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "species/population",
            "export",
            "dropped",
            "Slot incarnations, retired and inactive slots, and the identity counter "
            "are not openPMD records; import resumes the counter past the largest id.",
            changes_interpretation=False,
        ),
    ]
    if plan.boundaries is not None:
        losses.append(
            AdapterLoss(
                "boundaries",
                "export",
                "dropped",
                "Particle-boundary surface ledgers are not exported; wall charge "
                "remains inside rho.",
                changes_interpretation=False,
            )
        )
    if plan.recorders:
        losses.append(
            AdapterLoss(
                "recorders",
                "export",
                "dropped",
                "Recorder states are diagnostics outside the openPMD records.",
                changes_interpretation=False,
            )
        )
    if state.field_history is not None:
        losses.append(
            AdapterLoss(
                "field_history",
                "export",
                "dropped",
                "The previous accepted field kept for field time derivatives is not "
                "exported.",
                changes_interpretation=False,
            )
        )
    grid = layout.grid
    auxiliary = (grid.cochain is not None and bool(grid.cochain.maxwell.observers)) or (
        grid.reduced is not None and grid.reduced.field.pml is not None
    )
    if auxiliary:
        losses.append(
            AdapterLoss(
                "field/auxiliary",
                "export",
                "dropped",
                "Maxwell observer accumulations and CPML memory are not exported.",
                changes_interpretation=False,
            )
        )
    return tuple(losses)


@dataclass(frozen=True, slots=True)
class _EncodedIteration:
    data: bytes
    iteration: int
    file_format: str
    source_id: str
    losses: tuple[AdapterLoss, ...]


def _encode_state(
    layout: OpenPMDPICLayout,
    state: ElectromagneticPICState,
    step_size: float,
    current: Any,
    limits: ResourceLimits,
    series: str,
    /,
) -> _EncodedIteration:
    from ..discretization.pic import PICRunStatus
    from ..solver import ElectromagneticPICState

    if not isinstance(state, ElectromagneticPICState):
        raise TypeError("state must be an ElectromagneticPICState.")
    if not isinstance(limits, ResourceLimits):
        raise TypeError("limits must be ResourceLimits.")
    step = float(step_size)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step_size must be finite and positive.")
    if int(state.status) != int(PICRunStatus.SUCCESS):
        raise ValueError("Only successful PIC states are exported.")
    iteration = int(state.accepted_step)
    time = float(state.time)
    units = layout.scale.unit_si_map()
    plan = layout.plan
    records = _mesh_records(layout, state, current, step)
    arrays = tuple(_species_arrays(value) for value in state.species)
    rank = len(layout.grid.labels)
    require_encoding_budget(
        sum(value.size for record in records for value in record.components)
        + sum(value.identities.shape[0] * (2 * rank + 8) for value in arrays),
        limits,
    )
    file_format = f"{series_name(series)}_%T.h5"
    buffer = BoundedHDF5Buffer(limits.max_bytes)
    with h5py.File(buffer, "w") as handle:
        write_series_root(
            handle,
            _REVISION,
            meshes_path=MESHES_PATH,
            particles_path=_PARTICLES_PATH,
            file_format=file_format,
        )
        group = write_iteration(
            handle, iteration, OpenPMDIterationTime(time, step, units["time"][0])
        )
        meshes = group.create_group(MESHES_PATH.rstrip("/"))
        for record in records:
            write_mesh_record(meshes, record, units)
        particles = group.create_group(_PARTICLES_PATH.rstrip("/"))
        for species, value, mass in zip(
            plan.species, arrays, layout.particle_masses, strict=True
        ):
            _write_species(
                particles.create_group(species.species_id),
                value,
                mass,
                species.charge_model.base_specific_charge,
                layout.grid.labels,
                units,
                step,
            )
    source_id = canonical_fingerprint(
        {
            "kind": "pic-state-openpmd-export",
            "layout": layout.layout_id,
            "iteration": iteration,
            "time": time,
            "step_size": step,
            "records": [value.record_id for value in records],
            "species": array_tree_fingerprint(
                tuple(
                    (
                        value.identities,
                        value.parents,
                        value.positions,
                        value.proper_velocities,
                        value.macro_masses,
                        value.charge_numbers,
                    )
                    for value in arrays
                )
            ),
        }
    )
    return _EncodedIteration(
        buffer.getvalue(),
        iteration,
        file_format,
        source_id,
        _export_losses(layout, state),
    )


def _export_report(
    encoded: _EncodedIteration, target_id: str, target_format: str, /
) -> AdapterReport:
    return AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "ElectromagneticPICState",
        target_format,
        source_id=encoded.source_id,
        target_id=target_id,
        coordinate_mapping=(
            "solver field layout -> E, B, rho (and step J) Cartesian mesh "
            "components with their staggering positions",
            "accepted_step -> iteration; state time and step size -> time, dt",
            "active slots -> particles in ascending (id_hi << 32 | id_lo) order",
            "particle_mass * proper velocity -> momentum at timeOffset -dt/2",
            "macroparticle mass / particle_mass -> weighting",
        ),
        preserved_fields=(
            "E, B, rho, J",
            "particle positions, proper velocities, macro masses, charge numbers",
            "particle identity and parent lineage",
            "time and step size",
        ),
        assumptions=_ASSUMPTIONS,
        losses=encoded.losses,
    )


def write_openpmd_pic_state(
    directory: str | Path,
    layout: OpenPMDPICLayout,
    state: ElectromagneticPICState,
    /,
    *,
    step_size: float,
    limits: ResourceLimits,
    series: str = "pic",
    current: Any = None,
) -> OpenPMDPICExportResult:
    """Exclusively publish one PIC state as ``<series>_<accepted_step>.h5``.

    ``current`` is the solver-layout current of the step that produced
    ``state`` (``ElectromagneticPICStepResult.current``) and becomes the ``J``
    record; omit it for an initial state. Slot bookkeeping, charge-transition
    history, boundary ledgers, recorders, and field auxiliary state are
    declared losses of the report.
    """
    if not isinstance(layout, OpenPMDPICLayout):
        raise TypeError("layout must be an OpenPMDPICLayout.")
    encoded = _encode_state(layout, state, step_size, current, limits, series)
    destination = Path(directory) / encoded.file_format.replace(
        "%T", str(encoded.iteration)
    )
    resource = bounded_resource_from_bytes(
        encoded.data, limits=limits, source_path=str(destination)
    )
    publish_bytes(
        destination, encoded.data, maximum_bytes=limits.max_bytes, mode="exclusive"
    )
    return OpenPMDPICExportResult(
        destination,
        encoded.iteration,
        resource,
        _export_report(encoded, resource.manifest.manifest_id, _FORMAT),
    )


# Import -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ParticleRecord:
    components: tuple[tuple[h5py.Dataset | h5py.Group, OpenPMDUnit], ...]
    macro_weighted: bool
    weighting_power: float
    time_offset: float


@dataclass(frozen=True, slots=True)
class _ScannedSpecies:
    count: int
    records: dict[str, _ParticleRecord]


def _record_axes(name: str, labels: tuple[str, ...], /) -> tuple[str, ...]:
    if name == "momentum":
        return CARTESIAN_AXES
    if name in ("position", "positionOffset"):
        return labels
    return ()


def _scan_particle_record(
    species: h5py.Group,
    name: str,
    labels: tuple[str, ...],
    dimension: OpenPMDDimension,
    tolerance: float,
    counts: set[int],
    /,
) -> _ParticleRecord:
    record = species[name]
    path = f"{species.name}/{name}"
    axes = _record_axes(name, labels)
    if axes:
        if not isinstance(record, h5py.Group) or "value" in record.attrs:
            raise ValueError(f"Particle record {path} must be a vector record group.")
        missing = [axis for axis in axes if axis not in record]
        if missing:
            raise OpenPMDUnsupportedError(
                f"Particle record {path} lacks components {missing}."
            )
        extra = [axis for axis in CARTESIAN_AXES if axis in record and axis not in axes]
        if extra:
            raise _StateMismatchError(
                f"Particle record {path} has components {extra} outside the PIC grid."
            )
        items = tuple((record[axis], f"{path}/{axis}") for axis in axes)
    else:
        items = ((record, path),)
    components = []
    for item, item_path in items:
        shape = component_shape(item, item_path)
        if len(shape) != 1:
            raise ValueError(f"Particle record component {item_path} must be 1-D.")
        counts.add(shape[0])
        components.append(
            (item, read_record_unit(record.attrs, item.attrs, dimension, tolerance))
        )
    macro, power = weighting_metadata(record.attrs, path)
    return _ParticleRecord(
        tuple(components), macro, power, scalar_attribute(record.attrs, "timeOffset")
    )


def _scan_species(
    particles: h5py.Group,
    name: str,
    labels: tuple[str, ...],
    units: dict[str, tuple[float, tuple[float, ...]]],
    time: OpenPMDIterationTime,
    tolerance: float,
    /,
) -> _ScannedSpecies:
    if name not in particles or not isinstance(particles[name], h5py.Group):
        raise ValueError(f"Species {name} is absent from the iteration.")
    species = particles[name]
    counts: set[int] = set()
    records = {}
    for record, quantity in _PARTICLE_RECORDS.items():
        if record not in species:
            if record == "parentId":
                continue
            raise ValueError(f"Missing required particle record {record} of {name}.")
        dimension = (
            _DIMENSIONLESS
            if quantity is None
            else scale_unit(units, quantity).unit_dimension
        )
        records[record] = _scan_particle_record(
            species, record, labels, dimension, tolerance, counts
        )
    if len(counts) != 1:
        raise ValueError(
            f"Particle records of {name} are truncated: extents {sorted(counts)} disagree."
        )
    for record in ("id", "parentId"):
        if record in records and records[record].components[0][1].unit_si != 1.0:
            raise ValueError(f"Particle {record} records must carry unitSI = 1.")
    scale = tolerance * max(abs(time.dt), np.finfo(np.float64).tiny)
    for record in ("position", "positionOffset"):
        if abs(records[record].time_offset) > scale:
            raise OpenPMDUnsupportedError(
                "PIC positions must be defined at the iteration time."
            )
    if abs(records["momentum"].time_offset + 0.5 * time.dt) > scale:
        raise OpenPMDUnsupportedError(
            "PIC momenta must be leapfrog-staggered at timeOffset -dt/2."
        )
    return _ScannedSpecies(counts.pop(), records)


@dataclass(frozen=True, slots=True)
class _DecodedSpecies:
    identities: np.ndarray
    parents: np.ndarray
    positions: np.ndarray
    momenta: np.ndarray
    weights: np.ndarray
    charges: np.ndarray
    masses: np.ndarray


def _record_values(
    record: _ParticleRecord, count: int, weights: np.ndarray, /
) -> np.ndarray:
    """Per-particle SI values ``[N, components]``."""
    values = np.stack(
        [
            unit.to_si(component_values(item, (count,)))
            for item, unit in record.components
        ],
        axis=-1,
    )
    if record.macro_weighted:
        values = values / (weights**record.weighting_power)[:, None]
    return values


def _decode_species(
    scanned: _ScannedSpecies,
    units: dict[str, tuple[float, tuple[float, ...]]],
    /,
) -> _DecodedSpecies:
    count, records = scanned.count, scanned.records
    identities = identity_values(records["id"].components[0][0], (count,))
    parents = (
        identity_values(records["parentId"].components[0][0], (count,))
        if "parentId" in records
        else np.full(count, _NO_IDENTITY, dtype=np.uint64)
    )
    weights = _record_values(records["weighting"], count, np.ones(count))[:, 0]
    positions = _record_values(records["position"], count, weights) + _record_values(
        records["positionOffset"], count, weights
    )
    decoded = _DecodedSpecies(
        identities,
        parents,
        positions / np.float64(units["length"][0]),
        _record_values(records["momentum"], count, weights)
        / np.float64(units["momentum"][0]),
        weights,
        _record_values(records["charge"], count, weights)[:, 0]
        / np.float64(units["charge"][0]),
        _record_values(records["mass"], count, weights)[:, 0]
        / np.float64(units["mass"][0]),
    )
    if not all(
        np.all(np.isfinite(value))
        for value in (
            decoded.positions,
            decoded.momenta,
            decoded.weights,
            decoded.charges,
            decoded.masses,
        )
    ):
        raise ValueError("Particle records must be finite.")
    if np.any(weights <= 0.0) or np.any(decoded.masses <= 0.0):
        raise ValueError("Particle weighting and mass must be positive.")
    return decoded


def _words(identities: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    return (
        (identities >> np.uint64(32)).astype(np.uint32),
        (identities & np.uint64(0xFFFFFFFF)).astype(np.uint32),
    )


def _species_state(
    plan: PICSpeciesPlan,
    decoded: _DecodedSpecies,
    particle_mass: float,
    tolerance: float,
    dtype: np.dtype,
    /,
) -> PICSpeciesState:
    from ..discretization.particle import ParticlePopulationState
    from ..discretization.pic import PICChargeState, PICParticleState, PICSpeciesState

    name = plan.species_id
    count, capacity = decoded.identities.shape[0], plan.capacity
    if count > capacity:
        raise _StateMismatchError(
            f"Species {name} holds {count} particles beyond its capacity {capacity}."
        )
    order = np.argsort(decoded.identities, kind="stable")
    identities = decoded.identities[order]
    if np.any(identities[1:] == identities[:-1]) or np.any(identities == _NO_IDENTITY):
        raise _StateMismatchError(
            f"Species {name} ids must be unique and below the reserved all-ones id."
        )
    if np.any(np.abs(decoded.masses - particle_mass) > tolerance * particle_mass):
        raise _StateMismatchError(
            f"Species {name} rest mass differs from the layout's particle mass."
        )
    model = plan.charge_model
    ratio = decoded.charges[order] / (particle_mass * model.base_specific_charge)
    numbers = np.rint(ratio)
    if (
        np.any(np.abs(ratio - numbers) > tolerance * np.maximum(1.0, np.abs(numbers)))
        or np.any(numbers < model.minimum_charge_number)
        or np.any(numbers > model.maximum_charge_number)
    ):
        raise _StateMismatchError(
            f"Species {name} charges are not admissible charge numbers of its model."
        )
    rank = decoded.positions.shape[1]
    active = np.zeros(capacity, dtype=np.bool_)
    active[:count] = True
    position = np.zeros((capacity, rank), dtype=np.float64)
    position[:count] = decoded.positions[order]
    velocity = np.zeros((capacity, 3), dtype=np.float64)
    velocity[:count] = decoded.momenta[order] / particle_mass
    mass = np.zeros(capacity, dtype=np.float64)
    mass[:count] = decoded.weights[order] * particle_mass
    state = plan.initialize(
        jnp.asarray(position, dtype=dtype),
        jnp.asarray(velocity, dtype=dtype),
        active_mask=jnp.asarray(active),
        masses=jnp.asarray(mass, dtype=dtype),
    )
    population = state.population
    if not np.array_equal(np.asarray(population.active), active):
        raise _StateMismatchError(
            f"Species {name} has structurally inactive slots below its particle count."
        )
    id_hi, id_lo = (
        np.asarray(population.id_hi).copy(),
        np.asarray(population.id_lo).copy(),
    )
    parent_hi = np.asarray(population.parent_hi).copy()
    parent_lo = np.asarray(population.parent_lo).copy()
    id_hi[:count], id_lo[:count] = _words(identities)
    parent_hi[:count], parent_lo[:count] = _words(decoded.parents[order])
    next_hi, next_lo = (
        _words(np.asarray(identities[-1] + np.uint64(1)))
        if count
        else (np.asarray(population.next_id_hi), np.asarray(population.next_id_lo))
    )
    charge_number = np.zeros(capacity, dtype=np.int16)
    charge_number[:count] = numbers.astype(np.int16)
    return PICSpeciesState(
        PICParticleState(state.particles.position, state.particles.proper_velocity),
        ParticlePopulationState(
            population.active,
            population.mass,
            population.incarnation,
            population.ever_occupied,
            population.retired,
            jnp.asarray(id_hi, dtype=jnp.uint32),
            jnp.asarray(id_lo, dtype=jnp.uint32),
            jnp.asarray(parent_hi, dtype=jnp.uint32),
            jnp.asarray(parent_lo, dtype=jnp.uint32),
            jnp.asarray(next_hi, dtype=jnp.uint32),
            jnp.asarray(next_lo, dtype=jnp.uint32),
        ),
        PICChargeState(
            jnp.asarray(charge_number),
            jnp.zeros(capacity, dtype=jnp.int32),
            jnp.full(capacity, -1, dtype=jnp.int32),
        ),
    )


def _require_grid(
    layout: OpenPMDPICLayout, record: OpenPMDMeshRecord, tolerance: float, /
) -> None:
    grid = layout.grid
    if record.geometry != "cartesian" or record.axis_labels != grid.labels:
        raise _StateMismatchError(
            f"Mesh record {record.name} axes differ from the PIC grid {grid.labels}."
        )
    if record.components[0].shape != grid.shape:
        raise _StateMismatchError(
            f"Mesh record {record.name} extents differ from the PIC grid {grid.shape}."
        )
    if not (
        np.allclose(record.grid_spacing, grid.spacing, rtol=tolerance, atol=0.0)
        and np.allclose(
            record.grid_global_offset,
            grid.offset,
            rtol=0.0,
            atol=tolerance * max(grid.spacing),
        )
        and np.allclose(record.positions, grid.staggering[record.name], atol=tolerance)
    ):
        raise _StateMismatchError(
            f"Mesh record {record.name} grid or staggering differs from the PIC solver."
        )


def _triple(values: tuple[Array, ...], /) -> tuple[Array, Array, Array]:
    first, second, third = values
    return first, second, third


def _field_state(layout: OpenPMDPICLayout, meshes: OpenPMDMeshIteration, /) -> Any:
    electric, magnetic, charge = (
        tuple(
            jnp.asarray(value, dtype=jnp.float64)
            for value in meshes.record(name).components
        )
        for name in _REQUIRED_RECORDS
    )
    grid = layout.grid
    if grid.cochain is not None:
        maxwell = grid.cochain.maxwell
        bridge = grid.cochain.bridge
        material = maxwell.constitutive.initialize_state()
        return maxwell.pack(
            maxwell.constitutive.electric_displacement(
                bridge.pack_edge_circulation(_triple(electric)), material
            ),
            bridge.pack_face_flux(_triple(magnetic)),
            bridge.pack(0, charge),
            material_state=material,
        )
    if grid.reduced is None:
        raise RuntimeError("A PIC layout grid binds no field solver.")
    return grid.reduced.field.initialize(
        electric=_triple(electric), magnetic=_triple(magnetic), charge=charge[0]
    )


def _wall_charge(
    layout: OpenPMDPICLayout,
    species: tuple[Any, ...],
    field_charge: Any,
    tolerance: float,
    /,
) -> Any:
    """Grid charge of absorbed particles: the field charge minus the particles'."""
    plan = layout.plan
    deposited, successful = plan.species_charge(species)
    if not bool(successful):
        raise _StateMismatchError("Imported particles cannot be deposited on the grid.")
    if not plan.filters:
        return field_charge - deposited
    # A filtered run cannot separate wall charge from rho; it must be absent.
    filtered = plan._filtered_charge(deposited)
    scale = max(1.0, float(jnp.max(jnp.abs(field_charge))))
    if float(jnp.max(jnp.abs(field_charge - filtered))) > tolerance * scale:
        raise _StateMismatchError(
            "rho differs from the filtered particle charge; wall charge of a "
            "filtered run is not recoverable."
        )
    return jnp.zeros_like(field_charge)


def _decode_pic(
    resource: BoundedResource,
    layout: OpenPMDPICLayout,
    policy: OpenPMDMeshImportPolicy,
    units: dict[str, tuple[float, tuple[float, ...]]],
    /,
) -> tuple[OpenPMDMeshIteration, tuple[_DecodedSpecies, ...], int, int, int]:
    tolerance = policy.metadata_tolerance
    with h5py.File(BytesIO(resource.data), "r") as handle:
        inventory = preflight_hdf5(handle, resource.manifest.limits)
        root, group, time = open_iteration(handle, policy.iteration)
        meshes = records_group(group, root.meshes_path, "meshesPath")
        particles = records_group(group, root.particles_path, "particlesPath")
        scanned_meshes = tuple(
            scan_mesh_record(meshes, name, record_dimension(units, name), tolerance)
            for name in policy.records
        )
        scanned_species = tuple(
            _scan_species(
                particles, value.species_id, layout.grid.labels, units, time, tolerance
            )
            for value in layout.plan.species
        )
        elements = (
            sum(value.element_count for value in scanned_meshes)
            + sum(value.count for value in scanned_species) * _PARTICLE_WORDS
            + sum(value.capacity for value in layout.plan.species) * _SLOT_WORDS
        )
        inventory.require_budget(
            resource.manifest.limits, policy.maximum_decoded_bytes, elements * 16
        )
        records = tuple(
            decode_mesh_record(value, units, time.time_unit_si)
            for value in scanned_meshes
        )
        species = tuple(_decode_species(value, units) for value in scanned_species)
    time_unit = time.time_unit_si / units["time"][0]
    return (
        OpenPMDMeshIteration(
            policy.iteration, time.time * time_unit, time.dt * time_unit, records
        ),
        species,
        inventory.maximum_depth,
        inventory.object_count + elements,
        inventory.attribute_count,
    )


def _build_state(
    layout: OpenPMDPICLayout,
    meshes: OpenPMDMeshIteration,
    species: tuple[_DecodedSpecies, ...],
    tolerance: float,
    /,
) -> ElectromagneticPICState:
    from ..discretization.pic import PICRunStatus
    from ..solver import ElectromagneticPICState, PICFieldHistory

    plan = layout.plan
    for record in meshes.records:
        _require_grid(layout, record, tolerance)
    if meshes.dt <= 0.0:
        raise _StateMismatchError("The iteration dt must be a positive PIC step.")
    if meshes.iteration >= 2**31:
        raise _StateMismatchError("The iteration exceeds the PIC int32 step counter.")
    dtype = np.dtype(plan.precision.particle_dtype)
    states = tuple(
        _species_state(value, decoded, mass, tolerance, dtype)
        for value, decoded, mass in zip(
            plan.species, species, layout.particle_masses, strict=True
        )
    )
    field_value = _field_state(layout, meshes)
    field_charge = plan.solver.field_charge(field_value)
    time = jnp.asarray(meshes.time, dtype=dtype)
    return ElectromagneticPICState(
        states,
        field_value,
        ()
        if plan.boundaries is None
        else tuple(plan.boundaries.initialize_surface(dtype) for _ in plan.species),
        _wall_charge(layout, states, field_charge, tolerance),
        tuple(recorder.initialize(states, time) for recorder in plan.recorders),
        time,
        jnp.asarray(meshes.iteration, dtype=jnp.int32),
        jnp.asarray(int(PICRunStatus.SUCCESS), dtype=jnp.int32),
        # As at initialization, the imported field is taken as static over the
        # preceding step.
        PICFieldHistory(field_value, time - meshes.dt)
        if plan.field_derivatives
        else None,
        plan.initialize_process_states(states),
    )


def _import_losses(layout: OpenPMDPICLayout, /) -> tuple[AdapterLoss, ...]:
    plan = layout.plan
    losses = [
        AdapterLoss(
            "species/population",
            "import",
            "synthesized",
            "Particles occupy the leading slots in ascending id order; the identity "
            "counter resumes past the largest id and charge-transition history "
            "restarts.",
            changes_interpretation=False,
        )
    ]
    if plan.boundaries is not None or plan.recorders:
        losses.append(
            AdapterLoss(
                "boundaries+recorders",
                "import",
                "synthesized",
                "Particle-boundary ledgers and recorder states restart; wall charge "
                "is rho minus the deposited particle charge.",
                changes_interpretation=False,
            )
        )
    if plan.field_derivatives:
        losses.append(
            AdapterLoss(
                "field_history",
                "import",
                "synthesized",
                "The field history treats the imported field as static over the "
                "preceding step.",
                changes_interpretation=True,
            )
        )
    if any(process.stateful for process in plan.processes):
        losses.append(
            AdapterLoss(
                "processes",
                "import",
                "synthesized",
                "Process states (strong-field QED optical depths, trajectory "
                "histories and photons) restart from their initial state.",
                changes_interpretation=True,
            )
        )
    return tuple(losses)


def read_openpmd_pic_state(
    resource: BoundedResource,
    layout: OpenPMDPICLayout,
    policy: OpenPMDMeshImportPolicy,
    /,
) -> OpenPMDPICImportResult:
    """Rebuild a PIC state of ``layout.plan`` from one bounded openPMD iteration.

    ``policy.records`` must include ``E``, ``B``, and ``rho`` (``J`` is read
    into ``result.meshes`` when requested). Every mesh record must lie on the
    plan's grid with its solver staggering and every species must be present,
    fit its capacity, carry the layout's particle rest mass, and have integral
    admissible charge numbers. Refusals raise ``OpenPMDMeshError``: records
    that contradict the plan, repeated ids, and resource overflow are
    ``INCONSISTENT_SOURCE``; unsupported semantics (other geometries, missing
    components, unstaggered momenta) are ``UNSUPPORTED_REQUIRED_SEMANTIC``;
    malformed or truncated records are ``MALFORMED_SOURCE``.
    """
    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be a BoundedResource.")
    if not isinstance(layout, OpenPMDPICLayout):
        raise TypeError("layout must be an OpenPMDPICLayout.")
    if not isinstance(policy, OpenPMDMeshImportPolicy):
        raise TypeError("policy must be an OpenPMDMeshImportPolicy.")
    if any(name not in tuple(policy.records) for name in _REQUIRED_RECORDS):
        raise ValueError("PIC import requires the E, B, and rho mesh records.")
    units = layout.scale.unit_si_map()
    source_id = resource.manifest.manifest_id
    tolerance = policy.metadata_tolerance
    try:
        meshes, species, depth, nodes, attributes = _decode_pic(
            resource, layout, policy, units
        )
        state = _build_state(layout, meshes, species, tolerance)
    except (ResourceReadError, _StateMismatchError) as error:
        raise refusal(
            AdapterStatus.INCONSISTENT_SOURCE,
            source_id,
            str(error),
            "ElectromagneticPICState",
            _ASSUMPTIONS,
        ) from error
    except OpenPMDUnsupportedError as error:
        raise refusal(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            source_id,
            str(error),
            "ElectromagneticPICState",
            _ASSUMPTIONS,
        ) from error
    except (OSError, KeyError, UnicodeError, TypeError, ValueError) as error:
        raise refusal(
            AdapterStatus.MALFORMED_SOURCE,
            source_id,
            str(error),
            "ElectromagneticPICState",
            _ASSUMPTIONS,
        ) from error
    accounted = account_bounded_resource(
        resource, depth=depth, nodes=nodes, attributes=attributes, losses=0
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        _FORMAT,
        "ElectromagneticPICState",
        source_id=accounted.manifest.manifest_id,
        target_id=canonical_fingerprint(
            {
                "kind": "openpmd-pic-import",
                "layout": layout.layout_id,
                "iteration": meshes.iteration,
                "records": [value.record_id for value in meshes.records],
                "species": array_tree_fingerprint(
                    tuple(
                        (value.identities, value.parents, value.positions, value.momenta)
                        for value in species
                    )
                ),
            }
        ),
        coordinate_mapping=(
            "E, B, rho mesh components -> solver field layout with rho as Gauss charge",
            "particles in ascending id order -> leading capacity slots",
            "momentum / mass -> proper velocity; weighting * mass -> macro mass",
            "charge / (mass * base_specific_charge) -> integral charge number",
            "iteration, time, dt -> accepted_step, state time, step size",
        ),
        preserved_fields=(
            "E, B, rho",
            "particle positions, proper velocities, macro masses, charge numbers",
            "particle identity and parent lineage",
            "time and step size",
        ),
        assumptions=_ASSUMPTIONS,
        losses=_import_losses(layout),
    )
    return OpenPMDPICImportResult(state, meshes.dt, meshes, accounted, report)


# Streaming output --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpenPMDPICStreamReceipt:
    """One published iteration of a streamed PIC series."""

    iteration: int
    path: Path
    size_bytes: int
    report: AdapterReport


class OpenPMDPICStreamWriter:
    """Bounded per-step openPMD output of one PIC run as a ``fileBased`` series.

    `write` publishes one complete iteration file ``<series>_<step>.h5`` (or,
    with an ADIOS2 ``provider``, the BP4 directory ``<series>_<step>.bp``) for
    every accepted step that is a multiple of ``interval``. Each iteration is
    encoded within ``limits`` in memory and published exclusively and
    atomically, so a reader of the series never sees a partial iteration. The
    iteration count and the total published bytes are bounded; a write that
    would exceed either is refused before anything is published.
    """

    __slots__ = (
        "_directory",
        "_interval",
        "_last",
        "_layout",
        "_limits",
        "_maximum_iterations",
        "_maximum_total_bytes",
        "_provider",
        "_receipts",
        "_series",
        "_total_bytes",
    )

    @checked
    def __init__(
        self,
        directory: str | Path,
        layout: OpenPMDPICLayout,
        /,
        *,
        limits: ResourceLimits,
        maximum_iterations: int,
        maximum_total_bytes: int,
        interval: int = 1,
        series: str = "pic",
        provider: OpenPMDADIOS2Provider | None = None,
    ) -> None:
        if provider is not None and not isinstance(provider, OpenPMDADIOS2Provider):
            raise TypeError("provider must be an OpenPMDADIOS2Provider or None.")
        bounds = (maximum_iterations, maximum_total_bytes, interval)
        if any(type(value) is not int or value <= 0 for value in bounds):
            raise ValueError(
                "maximum_iterations, maximum_total_bytes, and interval must be "
                "positive integers."
            )
        destination = Path(directory)
        if not destination.is_dir():
            raise ValueError("directory must be an existing directory.")
        self._directory = destination
        self._layout = layout
        self._limits = limits
        self._maximum_iterations = maximum_iterations
        self._maximum_total_bytes = maximum_total_bytes
        self._interval = interval
        self._series = series_name(series)
        self._provider = provider
        self._receipts: list[OpenPMDPICStreamReceipt] = []
        self._total_bytes = 0
        self._last = -1

    @property
    def receipts(self) -> tuple[OpenPMDPICStreamReceipt, ...]:
        return tuple(self._receipts)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def write(
        self,
        value: ElectromagneticPICState | ElectromagneticPICStepResult,
        /,
        *,
        step_size: float,
    ) -> OpenPMDPICStreamReceipt | None:
        """Publish ``value`` if its accepted step is due; return its receipt.

        A step result contributes its accepted state and, when successful, the
        step's current as ``J``; a rejected step publishes nothing. A plain
        state (such as the initial state) has no ``J``. Steps off the cadence
        or already published return ``None``; an earlier step is refused.
        """
        from ..solver import ElectromagneticPICState, ElectromagneticPICStepResult

        if isinstance(value, ElectromagneticPICStepResult):
            if not bool(value.successful):
                return None
            state, current = value.accepted_state, value.current
        elif isinstance(value, ElectromagneticPICState):
            state, current = value, None
        else:
            raise TypeError(
                "value must be an ElectromagneticPICState or ElectromagneticPICStepResult."
            )
        step = int(state.accepted_step)
        if step < self._last:
            raise ValueError("Streamed PIC iterations must not go back in time.")
        if step == self._last or step % self._interval:
            return None
        if len(self._receipts) >= self._maximum_iterations:
            raise ResourceReadError("limit", "The PIC stream reached maximum_iterations.")
        encoded = _encode_state(
            self._layout, state, step_size, current, self._limits, self._series
        )
        if self._total_bytes + len(encoded.data) > self._maximum_total_bytes:
            raise ResourceReadError(
                "limit", "The PIC stream would exceed maximum_total_bytes."
            )
        member = encoded.file_format.replace("%T", str(encoded.iteration))
        resource = bounded_resource_from_bytes(
            encoded.data,
            limits=self._limits,
            source_path=str(self._directory / member),
        )
        if self._provider is None:
            path = self._directory / member
            publish_bytes(
                path, encoded.data, maximum_bytes=self._limits.max_bytes, mode="exclusive"
            )
            size = len(encoded.data)
            report = _export_report(encoded, resource.manifest.manifest_id, _FORMAT)
        else:
            artifact = convert_openpmd_hdf5_to_adios2(
                self._provider, resource, self._directory, limits=self._limits
            )
            path, size = artifact.path, artifact.size_bytes
            report = AdapterReport(
                AdapterStatus.DECLARED_LOSS,
                "ElectromagneticPICState",
                artifact.report.target_format,
                source_id=encoded.source_id,
                target_id=artifact.report.target_id,
                assumptions=_ASSUMPTIONS,
                losses=encoded.losses,
                stages=(
                    _export_report(encoded, resource.manifest.manifest_id, _FORMAT),
                    artifact.report,
                ),
            )
        receipt = OpenPMDPICStreamReceipt(encoded.iteration, path, size, report)
        self._receipts.append(receipt)
        self._total_bytes += size
        self._last = step
        return receipt


__all__ = [
    "OpenPMDPICExportResult",
    "OpenPMDPICImportResult",
    "OpenPMDPICLayout",
    "OpenPMDPICStreamReceipt",
    "OpenPMDPICStreamWriter",
    "read_openpmd_pic_state",
    "write_openpmd_pic_state",
]
