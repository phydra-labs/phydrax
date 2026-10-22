# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import (
    array_tree_fingerprint,
    BranchDifferentiationPolicy,
    canonical_fingerprint,
    StrictModule,
)
from phydrax._array_archive import (
    array_collection_digest,
    read_array_archive,
    write_array_archive,
)
from phydrax._frozendict import frozendict
from phydrax._meshcore import NativeBatchTicket, NativeExecutionScope, NativeScopeTicket
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import coordinate_lagrange_element, FieldTracePolicy
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    swept_coordinate_element,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry import (
    ImplicitBoundarySource,
    MappedDomainBoundarySource,
    ParametricCurveBoundarySource,
)
from phydrax.geometry._surface_source_support import (
    SurfaceNativeRestrictionBoundarySource,
)
from phydrax.geometry.brep import CircleCurve, LineCurve, PlanePatch
from phydrax.lifecycle._meshing_field_records import (
    MeshingFieldDeclaration,
    MeshingFieldStateRole,
    MeshingFiniteElementFieldDeclaration,
    MeshingFiniteVolumeFieldDeclaration,
    prepare_meshing_field_index_binding,
)
from phydrax.meshing import SweepControl, SweepMapKind
from phydrax.meshing._association import (
    PlcAssociationTransfer,
    SurfaceAssociationTransfer,
)
from phydrax.meshing._hex_generation import exact_mapped_volume_quality
from phydrax.meshing._volume_generation import recover
from phydrax.meshing.providers._native_planar_domain import source_arrays
from phydrax.meshing.providers._native_publication import (
    coordinate_budget,
    coordinate_phase,
    derived_certificate_limits,
)
from phydrax.meshing.providers._native_structured import mapped_edge_evidence
from phydrax.meshing.providers._native_surface import quad_edge_evidence
from phydrax.typing import checked
from phydrax.units import METER

from ._layer_continuation import (
    FiniteVolumeFieldTemplate,
    LAYER_CORE_FIELD_KINDS,
    LAYER_CORE_FIELD_ROLES,
    layer_core_field_templates,
)


def family_finite_volume_field_templates(
    dimension: int, /, *, scalar_quadrature: bool = False
) -> tuple[FiniteVolumeFieldTemplate, ...]:
    """Return the Euler cell-average or embedded scalar-quadrature declaration."""
    if scalar_quadrature:
        components = ("u",)
        role = "intensive"
    else:
        components = phx.equations.EulerSystem(dimension).component_names
        role = "conservative-density"
    return (
        {
            "name": "state",
            "space_family": "finite-volume",
            "degree": 0,
            "continuity": "cell-local",
            "component_shape": (len(components),),
            "component_names": components,
            "role": role,
        },
    )


def family_solver_field_templates(
    cell_kind: str, /
) -> tuple[tuple[str, phx.discretization.FiniteElementFieldSpec], ...]:
    """Return the scalar linear diffusion declaration for one cell family."""
    return (
        (
            "solve:intensive",
            phx.discretization.FiniteElementFieldSpec(
                "u", phx.discretization.lagrange_element(cell_kind, 1)
            ),
        ),
    )


def _family_transport_field_roles(kind: str, /) -> tuple[tuple[str, str], ...]:
    if kind not in LAYER_CORE_FIELD_ROLES:
        raise ValueError(
            f"continuation.field_kind must be one of {LAYER_CORE_FIELD_KINDS!r}; got {kind!r}"
        )
    return (
        (f"{kind}:{LAYER_CORE_FIELD_ROLES[kind]}", "current"),
        (f"{kind}:material-history", "history"),
    )


def family_transport_field_templates(
    cell_kind: str, /, *, kind: str
) -> tuple[tuple[str, phx.discretization.FiniteElementFieldSpec], ...]:
    """Return the current and accepted-history declarations for one field kind."""
    field = layer_core_field_templates(cell_kind, kinds=(kind,))[0][1]
    return tuple(
        (
            role,
            phx.discretization.FiniteElementFieldSpec(
                name, field.elements[0], component_shape=field.component_shape
            ),
        )
        for role, name in _family_transport_field_roles(kind)
    )


type FamilySource = (
    phx.meshing.NativePlanarSource
    | phx.meshing.NativeSurfaceSource
    | phx.meshing.NativePlcSource
    | phx.meshing.NativeSweepSource
    | phx.meshing.NativeStructuredSource
    | phx.meshing.NativeMappedHexSource
)
type FamilyAssociationTransfer = (
    phx.meshing.PlcAssociationTransfer
    | phx.meshing.SurfaceAssociationTransfer
    | phx.meshing.MappedReferenceAssociationTransfer
    | None
)


@final
class FamilyFieldBindings(StrictModule):
    declarations: tuple[phx.lifecycle.MeshingFieldDeclaration, ...]
    arrays: dict[str, jax.Array | np.ndarray]


@checked
def prepare_family_finite_element_bindings(
    result: phx.meshing.CellMeshingResult,
    spaces: Mapping[str, phx.discretization.FiniteElementDiscretization],
    problem: phx.equations.CompiledFiniteElementProblem,
    values: jax.Array,
    fields: dict[str, jax.Array],
    epoch: int,
) -> FamilyFieldBindings:
    """Bind finite-element coefficients, history, and essential-boundary state."""
    declarations: list[MeshingFieldDeclaration] = []
    arrays: dict[str, jax.Array | np.ndarray] = {}
    inverse_meter = phx.units.derived_unit("m^-1", ((phx.units.METER, -1),))
    for kind, owner in spaces.items():
        template_roles = (
            {}
            if kind == "Q1"
            else {name: role for role, name in _family_transport_field_roles(kind)}
        )
        roles = []
        physical_units = (
            (inverse_meter,) if kind in ("Hcurl", "Hdiv") else (phx.units.ONE,)
        )
        for index, field in enumerate(owner.field_spaces):
            state = values if kind == "Q1" else fields[kind][:, index]
            binding = prepare_meshing_field_index_binding(
                result, "field/global_coefficient_ids", field_space=field
            )
            entry = f"{kind}/{field.name}/coefficients"
            arrays[entry] = state
            roles.append(
                MeshingFieldStateRole(
                    entry,
                    field.name,
                    "coefficients",
                    tuple(state.shape),
                    np.dtype(state.dtype),
                    epoch,
                    index_binding=binding,
                )
            )
            if kind != "Q1":
                history = (
                    state
                    if template_roles[field.name] == f"{kind}:material-history"
                    else None
                )
                if history is not None:
                    entry = f"{kind}/{field.name}/stage-history"
                    arrays[entry] = history
                    roles.append(
                        MeshingFieldStateRole(
                            entry,
                            field.name,
                            "stage-history",
                            tuple(history.shape),
                            np.dtype(history.dtype),
                            epoch,
                            value_units=physical_units,
                            index_binding=binding,
                        )
                    )
            else:
                mask = jnp.zeros(values.shape, dtype=jnp.bool_)
                if problem.dirichlet_rows[0] is None:
                    raise FamilyPreparationError(
                        "_family_accepted_epoch",
                        "family.problem.dirichlet_rows must retain essential-boundary rows; got None",
                        {"problem": problem},
                    )
                mask = mask.at[problem.dirichlet_rows[0]].set(True)
                for role_name, bank in (
                    ("boundary-mask", mask),
                    ("boundary-values", problem.lift),
                ):
                    match bank:
                        case jax.Array() | np.ndarray():
                            pass
                        case _:
                            raise FamilyPreparationError(
                                "_family_accepted_epoch.boundary_values",
                                f"family.boundary_values must retain a single coefficient array; got {type(bank).__name__}",
                                {"bank": bank},
                            )
                    entry = f"Q1/{field.name}/{role_name}"
                    arrays[entry] = bank
                    roles.append(
                        MeshingFieldStateRole(
                            entry,
                            field.name,
                            role_name,
                            tuple(bank.shape),
                            np.dtype(bank.dtype),
                            epoch,
                            index_binding=binding,
                        )
                    )
        epoch_entry = f"{kind}/accepted-epoch"
        arrays[epoch_entry] = np.asarray(epoch, dtype=np.int64)
        roles.append(
            MeshingFieldStateRole(
                epoch_entry,
                owner.field_spaces[0].name,
                "state-epoch",
                (),
                np.dtype(np.int64),
                epoch,
            )
        )
        declarations.append(
            MeshingFiniteElementFieldDeclaration(
                part_name="family",
                topology_id=result.mesh.topology_id,
                geometry_layout_id=result.geometry.geometry_layout_id,
                field_space_ids=frozendict(
                    {field.name: field.field_space_id for field in owner.field_spaces}
                ),
                value_units=frozendict(
                    {field.name: physical_units for field in owner.field_spaces}
                ),
                maximum_derivative_orders=frozendict(
                    {field.name: 1 for field in owner.field_spaces}
                ),
                branch_policy=BranchDifferentiationPolicy.FROZEN_DECISION,
                trace_policy=FieldTracePolicy("cell-sided"),
                state_roles=tuple(roles),
                history_policy="accepted-state",
                numeric_version=owner.numeric_version,
                finite_element_fields=owner.construction_plan.fields,
                precision_policy=owner.precision_policy,
            )
        )
    return FamilyFieldBindings(tuple(declarations), arrays)


@checked
def prepare_family_finite_volume_bindings(
    result: phx.meshing.CellMeshingResult,
    fv: phx.discretization.UnstructuredFiniteVolumeDiscretization,
    content: jax.Array,
    epoch: int,
    *,
    scalar_quadrature: bool = False,
) -> FamilyFieldBindings:
    """Bind accepted Euler or intrinsic scalar FV content and extensive history."""
    declarations: list[MeshingFieldDeclaration] = []
    arrays: dict[str, jax.Array | np.ndarray] = {}
    system = None if scalar_quadrature else phx.equations.EulerSystem(
        result.mesh.ambient_dimension
    )
    template = family_finite_volume_field_templates(
        result.mesh.topological_dimension, scalar_quadrature=scalar_quadrature
    )[0]
    field = fv.field_spaces[0]
    binding = prepare_meshing_field_index_binding(
        result, "field/global_coefficient_ids", field_space=field
    )
    state = content / fv.cell_volumes[:, None]
    momentum_density = phx.units.derived_unit(
        "kg/(m2*s)",
        ((phx.units.KILOGRAM, 1), (phx.units.METER, -2), (phx.units.SECOND, -1)),
    )
    momentum = phx.units.derived_unit(
        "kg*m/s",
        ((phx.units.KILOGRAM, 1), (phx.units.METER, 1), (phx.units.SECOND, -1)),
    )
    if system is None:
        intensive_units = (phx.units.ONE,)
        content_units = (
            phx.units.derived_unit("m2", ((phx.units.METER, 2),)),
        )
    else:
        intensive_units = (
            phx.units.KILOGRAM_PER_CUBIC_METER,
            *((momentum_density,) * system.dimension),
            phx.units.PASCAL,
        )
        content_units = (
            phx.units.KILOGRAM,
            *((momentum,) * system.dimension),
            phx.units.JOULE,
        )
        if result.mesh.topological_dimension == 2:
            # A planar Euler carrier integrates over area: these extensive
            # quantities are per unit out-of-plane length, not 3-D cell totals.
            content_units = (
                phx.units.derived_unit(
                    "kg/m", ((phx.units.KILOGRAM, 1), (phx.units.METER, -1))
                ),
                *((phx.units.derived_unit(
                    "kg/s", ((phx.units.KILOGRAM, 1), (phx.units.SECOND, -1))
                ),) * system.dimension),
                phx.units.derived_unit(
                    "J/m", ((phx.units.JOULE, 1), (phx.units.METER, -1))
                ),
            )
    arrays.update(
        {
            "FV/state/coefficients": state,
            "FV/state/content": content,
            "FV/accepted-epoch": np.asarray(epoch, dtype=np.int64),
        }
    )
    roles = (
        MeshingFieldStateRole(
            "FV/state/coefficients",
            "state",
            "coefficients",
            tuple(state.shape),
            np.dtype(state.dtype),
            epoch,
            index_binding=binding,
        ),
        MeshingFieldStateRole(
            "FV/state/content",
            "state",
            "stage-history",
            tuple(content.shape),
            np.dtype(content.dtype),
            epoch,
            value_units=content_units,
            index_binding=binding,
        ),
        MeshingFieldStateRole(
            "FV/accepted-epoch", "state", "state-epoch", (), np.dtype(np.int64), epoch
        ),
    )
    reconstruction = phx.discretization.PiecewiseConstantReconstruction()
    declarations.append(
        MeshingFiniteVolumeFieldDeclaration(
            part_name="family",
            topology_id=result.mesh.topology_id,
            geometry_layout_id=result.geometry.geometry_layout_id,
            field_space_ids=frozendict({"state": field.field_space_id}),
            value_units=frozendict({"state": intensive_units}),
            maximum_derivative_orders=frozendict({"state": 0}),
            branch_policy=BranchDifferentiationPolicy.SMOOTH,
            trace_policy=FieldTracePolicy("cell-sided"),
            state_roles=roles,
            history_policy="accepted-state",
            numeric_version=fv.numeric_version,
            finite_volume_field_name=template["name"],
            finite_volume_component_names=template["component_names"],
            finite_volume_coordinate_policy="accepted-map",
            reconstruction_policy=reconstruction,
            reconstruction_policy_id=reconstruction.plan_id,
            finite_volume_system=system,
            finite_volume_flux_policy=None
            if system is None else phx.discretization.RusanovFluxPlan(),
            finite_volume_boundaries=frozendict(
                {}
                if system is None
                else {
                    name: phx.discretization.SlipWallBoundary()
                    for name in fv.boundary_patch_names
                }
            ),
        )
    )
    return FamilyFieldBindings(tuple(declarations), arrays)


def family_oracle_content_id(
    metadata: dict[str, Any], arrays: Mapping[str, jax.Array | np.ndarray]
) -> str:

    return canonical_fingerprint(
        {"metadata": metadata, "array_payloads": array_collection_digest(arrays)}
    )


@checked
def write_family_oracle(
    path: str | Path,
    *,
    metadata: dict[str, Any],
    arrays: Mapping[str, jax.Array | np.ndarray],
    limits: phx.ArrayArchiveLimits,
) -> Path:
    """Persist the independently authored family oracle under explicit archive limits."""
    return write_array_archive(path, manifest=metadata, arrays=arrays, limits=limits)


@checked
def read_family_oracle(
    path: str | Path,
    content_id: str,
    source: FamilySource | None,
    *,
    limits: phx.ArrayArchiveLimits,
) -> dict[str, Any]:
    """Restore the independent authored oracle and reject changed content or domain identity."""

    manifest, arrays = read_array_archive(Path(path), limits=limits)
    metadata = {name: value for name, value in manifest.items() if name != "arrays"}
    if family_oracle_content_id(metadata, arrays) != content_id:
        raise FamilyPreparationError(
            "_family_read_oracle",
            (
                f"family.oracle.content_id must equal {content_id!r}; got differing content with metadata={metadata!r}"
            ),
            {"arrays": arrays, "content_id": content_id, "metadata": metadata},
        )
    case = dict(metadata["case"])
    if metadata["domain_role"] == "piecewise_linear":
        domain = phx.geometry.PiecewiseLinearDomain(
            arrays["vertices"],
            arrays["facets"],
            arrays["facet_regions"],
            tuple(metadata["region_ids"]),
            source_id=metadata["source_id"],
        )
    elif metadata["domain_role"] == "generation_source_domain":
        match source:
            case (
                phx.meshing.NativeSurfaceSource()
                | phx.meshing.NativeStructuredSource()
                | phx.meshing.NativeSweepSource()
                | phx.meshing.NativeMappedHexSource()
            ):
                domain = source.domain
            case _:
                raise AttributeError(
                    f"family.source must expose domain; got {type(source).__name__}"
                )
    else:
        raise ValueError(
            f"family.domain_role must name a supported domain role; got {metadata['domain_role']!r}"
        )
    if domain is None:
        raise AttributeError("family.domain must expose domain_id; got None")
    if domain.domain_id != metadata["domain_id"]:
        raise FamilyPreparationError(
            "_family_read_oracle",
            f"family.oracle.domain_id must equal {metadata['domain_id']!r}; got {domain.domain_id!r}",
            {"domain": domain, "metadata": metadata},
        )
    case["domain"] = domain
    return case


@final
class FamilyPreparationError(RuntimeError):
    """An impossible publication state prevents preparation of family evidence."""

    def __init__(self, field: str, expectation: str, values: dict[str, object]) -> None:
        super().__init__(expectation)
        self.field = field
        self.expectation = expectation
        self.values = values


@final
class FamilyQualityMeasurements(StrictModule):
    """Independent quality and physical-edge measurements from one bounded preparation."""

    values: phx.meshing.CellQualityEvaluation
    minimum_scaled: float = eqx.field(static=True)
    minimum_mean: float = eqx.field(static=True)
    maximum_aspect: float = eqx.field(static=True)
    lengths: np.ndarray
    edge_lower: np.ndarray | None
    edge_upper: np.ndarray | None


@final
class _FrameSweepProfile(StrictModule):
    axis: np.ndarray
    uv: np.ndarray
    profile_cells: np.ndarray
    profile: phx.discretization.CellMesh
    element: phx.discretization.FiniteElementSpec
    bank: np.ndarray
    routes: np.ndarray
    profile_geometry: phx.discretization.CellGeometrySpec
    control: phx.meshing.SweepControl
    frames: np.ndarray
    translations: np.ndarray


@final
class _FrameSweepAtlas(StrictModule):
    revision: str = eqx.field(static=True)
    domain: phx.geometry.MappedReferenceDomain
    query: phx.geometry.MappedDomainBoundarySource
    correspondence: phx.discretization.CellGeometryRestrictionSource


def _frame_sweep_profile(n: int) -> _FrameSweepProfile:
    """Author curved profile coefficients and absolute station frames."""
    axis = np.asarray(tuple(index / n for index in range(n + 1)), dtype=np.float64)
    uv = np.asarray(tuple((u, v) for u in axis for v in axis), dtype=np.float64)
    profile_cells = np.asarray(
        tuple(
            (
                i * (n + 1) + j,
                (i + 1) * (n + 1) + j,
                (i + 1) * (n + 1) + j + 1,
                i * (n + 1) + j + 1,
            )
            for i in range(n)
            for j in range(n)
        ),
        dtype=np.int32,
    )
    corners = np.column_stack(
        (uv[:, 0] + 0.25 * uv[:, 1], uv[:, 1] + 0.125 * uv[:, 0] * (1.0 - uv[:, 0]))
    )
    profile = phx.discretization.CellMesh(
        corners,
        (
            phx.discretization.CellBlock(
                "profile",
                "quadrilateral",
                profile_cells,
                global_ids=np.arange(n * n, dtype=np.int64) + 500,
            ),
        ),
        vertex_global_ids=np.arange(uv.shape[0], dtype=np.int64) + 40,
    )
    element = coordinate_lagrange_element("quadrilateral", 2)
    local = np.asarray(element.reference_nodes)
    controls = []
    for i in range(n):
        for j in range(n):
            u, v = (i + local[:, 0]) / n, (j + local[:, 1]) / n
            controls.append(np.column_stack((u + 0.25 * v, v + 0.125 * u * (1.0 - u))))
    bank = np.concatenate(controls)
    routes = np.arange(bank.shape[0], dtype=np.int32).reshape(
        n * n, element.local_dof_count
    )
    profile_geometry = phx.discretization.CellGeometrySpec(
        {"profile": element}, {"profile": routes}, bank
    )
    angles = 0.15 * axis
    frames = np.asarray(
        tuple(
            (
                (np.cos(angle), -np.sin(angle), 0.0),
                (np.sin(angle), np.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            )
            for angle in angles
        )
    )
    translations = axis[:, None] * np.asarray((0.1, -0.05, 1.0))
    control = SweepControl(
        SweepMapKind.FRAMES,
        phx.meshing.LayerSchedule(np.full(n, 1.0 / n, dtype=np.float64)),
        "authored-curved-profile-absolute-frames",
        frames=frames,
        translations=translations,
    )
    return _FrameSweepProfile(
        axis,
        uv,
        profile_cells,
        profile,
        element,
        bank,
        routes,
        profile_geometry,
        control,
        frames,
        translations,
    )


def _frame_sweep_atlas(
    n: int, profile: _FrameSweepProfile, limits: phx.meshing.MeshingLimits
) -> _FrameSweepAtlas:
    """Bind the station atlas to its mapped reference source."""
    axis, uv, profile_cells = profile.axis, profile.uv, profile.profile_cells
    element, bank, routes = profile.element, profile.bank, profile.routes
    frames, translations = profile.frames, profile.translations
    reference_points = np.asarray(tuple((u, v, w) for w in axis for u, v in uv))
    root_cells = np.asarray(
        tuple(
            tuple(profile_cell + layer * uv.shape[0])
            + tuple(profile_cell + (layer + 1) * uv.shape[0])
            for layer in range(n)
            for profile_cell in profile_cells
        ),
        dtype=np.int32,
    )
    root_ids = np.arange(root_cells.shape[0], dtype=np.int64) + 900
    reference_mesh = phx.discretization.CellMesh(
        reference_points,
        (
            phx.discretization.CellBlock(
                "root", "hexahedron", root_cells, global_ids=root_ids
            ),
        ),
        vertex_global_ids=np.arange(reference_points.shape[0], dtype=np.int64) + 100,
    )
    cube_points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        )
    )
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    facets = np.asarray(
        tuple(triangle for a, b, c, d in faces for triangle in ((a, b, c), (a, c, d))),
        dtype=np.int64,
    )
    reference_domain = phx.geometry.PiecewiseLinearDomain(
        cube_points,
        facets,
        np.tile(np.asarray((0, -1), dtype=np.int64), (12, 1)),
        ("solid",),
        source_id="curved-frame-reference",
    )
    padded = np.pad(bank, ((0, 0), (0, 1)))
    transformed = (
        padded[None, :, :] @ np.swapaxes(frames, -1, -2) + translations[:, None, :]
    )
    root_controls = np.concatenate(
        tuple(
            np.concatenate((transformed[layer, route], transformed[layer + 1, route]))
            for layer in range(n)
            for route in routes
        )
    )

    allowance = limits.native_limits()
    ticket = HostBatchTicket(
        allowance.maximum_work_units,
        queries_upper=allowance.maximum_geometry_queries,
        bytes_upper=allowance.maximum_managed_bytes,
        cavity_upper=allowance.maximum_cavity_cells,
    )
    budget = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=limits.maximum_work_units,
        maximum_memory_bytes=limits.maximum_scratch_bytes,
    )
    with ticket, budget.activate():
        root_element = swept_coordinate_element(element, budget=budget)
    root_geometry = phx.discretization.CellGeometrySpec(
        {"root": root_element},
        {
            "root": np.arange(root_controls.shape[0], dtype=np.int32).reshape(
                root_cells.shape[0], root_element.local_dof_count
            )
        },
        root_controls,
    )
    revision = canonical_fingerprint(
        {
            "kind": "authored-curved-frame-source",
            "profile_basis": element.element_id,
            "profile_controls": bank,
            "frames": frames,
            "translations": translations,
        }
    )
    domain = phx.geometry.MappedReferenceDomain(
        reference_domain,
        reference_mesh,
        root_geometry,
        np.zeros(root_cells.shape[0], dtype=np.int64),
        source_id="curved-frame-source",
        source_revision=revision,
        certificate_limits=derived_certificate_limits(limits),
    )
    query = MappedDomainBoundarySource(
        domain, np.zeros(root_cells.shape[0], dtype=np.int64)
    )
    correspondence = CellGeometryRestrictionSource(
        cell_geometry_id(root_geometry),
        reference_mesh.topology_id,
        {"sweep:profile": root_ids},
        {"sweep:profile": np.asarray(reference_mesh.vertex_global_ids)[root_cells]},
    )
    return _FrameSweepAtlas(revision, domain, query, correspondence)


@checked
def prepare_family_frame_sweep_source(
    n: int, *, limits: phx.meshing.MeshingLimits
) -> phx.meshing.NativeSweepSource:
    """Author the curved profile and station atlas before constructing a target."""
    prepared = _frame_sweep_profile(n)
    atlas = _frame_sweep_atlas(n, prepared, limits)
    profile, control, profile_geometry = (
        prepared.profile,
        prepared.control,
        prepared.profile_geometry,
    )
    revision, domain, query, correspondence = (
        atlas.revision,
        atlas.domain,
        atlas.query,
        atlas.correspondence,
    )
    return phx.meshing.NativeSweepSource(
        profile,
        control,
        domain.source_id,
        revision,
        fidelity_source=query,
        maximum_deviation=1e-10,
        domain=domain,
        profile_geometry=profile_geometry,
        root_correspondence=correspondence,
    )


def prepare_family_reversed_multiblock_source(
    n: int, materials: bool
) -> tuple[phx.meshing.NativeStructuredSource, np.ndarray]:
    """Return independent box boundary/material sources and reversed logical charts."""

    width = 2.0 if materials else 1.0
    split = width / 2.0
    geometry = phx.geometry.Box(
        (width / 2.0, 0.5, 0.5), (width, 1.0, 1.0), feature_id="reversed-block-source"
    ).compile(coordinate_unit=METER)
    query = ImplicitBoundarySource(
        geometry, source_id="reversed-block-source", spacing=0.025
    )
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (split, 0.0, 0.0),
            (split, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (split, 0.0, 1.0),
            (split, 1.0, 1.0),
            (0.0, 1.0, 1.0),
            (width, 0.0, 0.0),
            (width, 1.0, 0.0),
            (width, 0.0, 1.0),
            (width, 1.0, 1.0),
        )
    )
    source_cells = ((0, 1, 2, 3, 4, 5, 6, 7), (1, 8, 9, 2, 5, 10, 11, 6))
    face_template = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    face_rows: dict[tuple[int, ...], tuple[tuple[int, ...], list[int]]] = {}
    for material, cell in enumerate(source_cells):
        for face in face_template:
            row = tuple(cell[index] for index in face)
            key = tuple(sorted(row))
            if key in face_rows:
                face_rows[key][1][1] = material
            else:
                face_rows[key] = (row, [material, -1])
    facets, adjacency, interface_ids = [], [], []
    for row, labels in face_rows.values():
        if not materials and labels[1] >= 0:
            continue
        a, b, c, d = row
        if labels[1] >= 0:
            interface_ids.extend((len(facets), len(facets) + 1))
        facets.extend(((a, b, c), (a, c, d)))
        adjacency.extend(
            (tuple(labels), tuple(labels)) if materials else ((0, -1), (0, -1))
        )
    domain = phx.geometry.PiecewiseLinearDomain(
        vertices,
        np.asarray(facets, dtype=np.int64),
        np.asarray(adjacency, dtype=np.int64),
        ("left", "right") if materials else ("solid",),
        source_id=query.source_id,
    )
    blocks = []
    for name, origin, diagonal in (
        ("left", (0.0, 0.0, 0.0), (split, 1.0, 1.0)),
        ("right", (split, 1.0, 1.0), (split, -1.0, -1.0)),
    ):
        matrix = np.diag(np.asarray(diagonal))
        controls = []
        for axis in range(3):
            other = tuple(index for index in range(3) if index != axis)
            for side in range(2):
                patch = PlanePatch(
                    np.asarray(origin) + side * matrix[:, axis],
                    matrix[:, other[0]],
                    matrix[:, other[1]],
                )
                controls.append(
                    phx.meshing.TransfiniteSurfaceControl(
                        patch,
                        f"{name}:{2 * axis + side}",
                        np.asarray(((0.0, 0.0), (1.0, 1.0))),
                        (n, n),
                    )
                )
        blocks.append(phx.meshing.TransfiniteBlock(name, tuple(controls)))
    interface = phx.meshing.BlockInterfaceControl(
        "left", 1, "right", 0, (0, 1), (True, True)
    )
    source = phx.meshing.NativeStructuredSource(
        tuple(blocks),
        (interface,),
        query.source_id,
        query.source_revision,
        fidelity_source=query,
        maximum_deviation=0.2,
        domain=domain,
        block_regions=(("left", "left"), ("right", "right")) if materials else (),
    )
    return source, np.asarray(interface_ids, dtype=np.int64)


def prepare_family_annular_block_source(
    n: int,
) -> tuple[phx.meshing.NativeStructuredSource, phx.geometry.PiecewiseLinearDomain, float]:
    """Return circle sources and the declared source polygon domain."""

    inner = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0)
    outer = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 2.0)
    # A radius-two quarter arc has length pi. Four angular intervals per
    # physical resolution unit bound each source arc by pi/(4*n) < 1/n.
    angular = 4 * n
    angles = np.concatenate(
        tuple(
            np.linspace(index * np.pi / 2.0, (index + 1) * np.pi / 2.0, angular + 1)[:-1]
            for index in range(4)
        )
    )
    points = np.concatenate(
        (
            np.asarray(inner.evaluate(jnp.asarray(angles))),
            np.asarray(outer.evaluate(jnp.asarray(angles))),
        )
    )
    count = angles.size
    facets = tuple(((index + 1) % count, index) for index in range(count)) + tuple(
        (count + index, count + (index + 1) % count) for index in range(count)
    )
    revision = canonical_fingerprint(
        {
            "kind": "authored-circle-annular-blocks",
            "curves": array_tree_fingerprint((inner, outer)),
            "polygon_parameters": angles,
            "polygon_points": points,
        }
    )
    query = ParametricCurveBoundarySource(
        (inner, outer),
        ((0.0, 2.0 * np.pi),) * 2,
        source_id="annular-block-source",
        source_revision=revision,
        covering_radius=0.01,
    )
    domain = phx.geometry.PiecewiseLinearDomain(
        points,
        np.asarray(facets, dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (len(facets), 1)),
        ("annulus",),
        source_id=query.source_id,
    )
    blocks = []
    for index in range(4):
        first, last = index * np.pi / 2.0, (index + 1) * np.pi / 2.0
        first_ray, last_ray = (
            np.asarray((np.cos(first), np.sin(first))),
            np.asarray((np.cos(last), np.sin(last))),
        )
        blocks.append(
            phx.meshing.TransfiniteBlock(
                f"sector:{index}",
                (
                    phx.meshing.TransfiniteCurveControl(
                        inner, "inner", (first, last), angular
                    ),
                    phx.meshing.TransfiniteCurveControl(
                        outer, "outer", (first, last), angular
                    ),
                    phx.meshing.TransfiniteCurveControl(
                        LineCurve(first_ray, first_ray), f"ray:{index}", (0.0, 1.0), n
                    ),
                    phx.meshing.TransfiniteCurveControl(
                        LineCurve(last_ray, last_ray),
                        f"ray:{(index + 1) % 4}",
                        (0.0, 1.0),
                        n,
                    ),
                ),
            )
        )
    interfaces = tuple(
        phx.meshing.BlockInterfaceControl(
            f"sector:{index}", 3, f"sector:{(index + 1) % 4}", 2, (0,), (False,)
        )
        for index in range(4)
    )
    source = phx.meshing.NativeStructuredSource(
        tuple(blocks),
        interfaces,
        query.source_id,
        revision,
        fidelity_source=query,
        maximum_deviation=0.08,
    )
    area = (
        sum(
            Fraction(float(points[a, 0])) * Fraction(float(points[b, 1]))
            - Fraction(float(points[a, 1])) * Fraction(float(points[b, 0]))
            for a, b in facets
        )
        / 2
    )
    return source, domain, float(area)


@checked
def prepare_family_association_transfer(
    source: FamilySource,
    result: phx.meshing.CellMeshingResult,
    request: phx.meshing.SurfaceMeshingSpec | phx.meshing.VolumeMeshingSpec,
    *,
    domain: phx.geometry.PiecewiseLinearDomain
    | phx.geometry.MappedReferenceDomain
    | phx.geometry.MeshingDomain
    | None,
    limits: phx.meshing.MeshingLimits,
    record_phase: Callable[[phx.meshing.NativeMeshingPhaseMeasurement], None],
    budget: CoordinateEnclosureBudget | None = None,
    scope: NativeExecutionScope | None = None,
) -> FamilyAssociationTransfer:
    """Prepare source-table association transfer without inferring IDs from generated coordinates."""
    if (budget is None) != (scope is None):
        raise ValueError("Family request-owned association preparation requires its explicit scope and budget together")
    if limits.limits_id != request.limits.limits_id:
        raise ValueError("Family association limits must match the authored request")
    if budget is None:
        with NativeExecutionScope(limits.native_limits()) as owned_scope:
            with owned_scope.ticket_view() as ticket:
                owned_budget = CoordinateEnclosureBudget(
                    ticket,
                    maximum_work_units=limits.maximum_work_units,
                    maximum_memory_bytes=limits.maximum_scratch_bytes,
                )
                return prepare_family_association_transfer(
                    source, result, request, domain=domain, limits=limits,
                    record_phase=record_phase, budget=owned_budget, scope=owned_scope,
                )
    if (
        scope is None
        or not eqx.tree_equal(scope.limits, limits.native_limits())
        or budget.maximum_memory_bytes != limits.maximum_scratch_bytes
        or budget.maximum_work_units != limits.maximum_work_units
        or not isinstance(budget.ticket, (NativeBatchTicket, NativeScopeTicket))
        or budget.ticket.scope is not scope
        or budget.ticket.settled
    ):
        raise ValueError("Family association budget must use the authored request's live scope and limits")
    if not result.associations:
        return None
    match source:
        case phx.meshing.NativePlcSource():
            # Recovery owns source edge and triangle identity ordering.
            match request:
                case phx.meshing.VolumeMeshingSpec():
                    recovered = recover(
                        source.complex,
                        request,
                        scope=scope,
                        record_phase=record_phase,
                    )
                case _:
                    raise ValueError(
                        "qualification.family.request must be a volume specification for PLC association recovery; "
                        f"got {type(request).__name__}"
                    )
            match domain:
                case phx.geometry.PiecewiseLinearDomain():
                    transfer = PlcAssociationTransfer(
                        domain,
                        result.coordinate_contract,
                        source.source_revision,
                        edge_vertices=recovered.plc_edges,
                        triangle_vertices=recovered.input_triangles,
                        triangle_facets=source.complex.polygon_facets[
                            recovered.input_polygons
                        ],
                        facet_regions=source.complex.facet_regions,
                        maximum_support_queries=limits.maximum_geometry_queries,
                    )
                case _:
                    raise TypeError(
                        "qualification.family.domain must be PiecewiseLinearDomain for PLC association recovery; "
                        f"got {type(domain).__name__}"
                    )
        case phx.meshing.NativePlanarSource():
            _, edges, _ = source_arrays(source)
            certification = result.certification
            if certification is None:
                raise AttributeError(
                    "family.result.certification must expose request; got None"
                )
            planar_domain = certification.request.domain
            match planar_domain:
                case phx.geometry.PiecewiseLinearDomain():
                    transfer = PlcAssociationTransfer(
                        planar_domain,
                        result.coordinate_contract,
                        source.source_revision,
                        edge_vertices=edges,
                        triangle_vertices=np.empty((0, 3), dtype=np.int64),
                        triangle_facets=np.empty((0,), dtype=np.int64),
                        facet_regions=planar_domain.facet_regions,
                        maximum_support_queries=limits.maximum_geometry_queries,
                    )
                case _:
                    raise TypeError(
                        "result.certification.request.domain must be PiecewiseLinearDomain; "
                        f"got {type(planar_domain).__name__}"
                    )
        case phx.meshing.NativeSurfaceSource():
            certification = result.certification
            if certification is None:
                raise AttributeError(
                    "family.result.certification must expose request; got None"
                )
            retained = certification.request.source
            match retained:
                case SurfaceNativeRestrictionBoundarySource():
                    transfer = SurfaceAssociationTransfer(retained.support)
                case _:
                    raise TypeError(
                        f"family.result.certification.request.source must be SurfaceNativeRestrictionBoundarySource; got {type(retained).__name__}"
                    )
        case phx.meshing.NativeMappedHexSource():
            transfer = phx.meshing.MappedReferenceAssociationTransfer(
                source.domain, maximum_support_queries=limits.maximum_geometry_queries
            )
        case phx.meshing.NativeSweepSource(
            domain=phx.geometry.MappedReferenceDomain() as domain
        ):
            transfer = phx.meshing.MappedReferenceAssociationTransfer(
                domain, maximum_support_queries=limits.maximum_geometry_queries
            )
        case _:
            raise ValueError(
                f"family.source must support source-table association transfer; got {type(source).__name__}"
            )
    transfer.source_associations(result, budget=budget)
    return transfer


def _family_mesh_edges(mesh: phx.discretization.CellMesh) -> jax.Array:

    match mesh.connectivity:
        case (
            phx.discretization.PolygonalConnectivity()
            | phx.discretization.TetrahedralConnectivity()
            | phx.discretization.HexahedralConnectivity()
            | phx.discretization.PolyhedralConnectivity()
        ) as connectivity:
            return connectivity.edges
        case _:
            raise FamilyPreparationError(
                "family.connectivity",
                f"family.connectivity must publish explicit cell edges; got {type(mesh.connectivity).__name__}",
                {"connectivity": mesh.connectivity},
            )


@checked
def prepare_family_quality_measurements(
    result: phx.meshing.CellMeshingResult,
    request: phx.meshing.SurfaceMeshingSpec | phx.meshing.VolumeMeshingSpec,
    *,
    profile: str,
    limits: phx.meshing.MeshingLimits,
    budget: CoordinateEnclosureBudget | None = None,
) -> FamilyQualityMeasurements:
    """Prepare independent cell-quality and physical-edge evidence under declared domain limits."""
    mesh = result.mesh
    receiving_budget = budget
    with (
        NativeExecutionScope(limits.native_limits()) as scope,
        scope.host_workspace(),
    ):
        budget = coordinate_budget(limits) if budget is None else budget
        with (
            budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size))
            if receiving_budget is None
            else nullcontext()
        ):
            values = phx.meshing.evaluate_cell_quality(
                mesh,
                geometry=result.geometry
                if profile in ("curved-quarter-extrusion-quad", "curved-frame-sweep-hex")
                else None,
            )
            minimum_scaled = float(np.min(np.asarray(values.scaled_jacobian)))
            minimum_mean = float(np.min(np.asarray(values.mean_ratios)))
            maximum_aspect = float(np.max(np.asarray(values.aspect_ratios)))
            if profile == "hex-dominant":
                scaled, mean, aspect = exact_mapped_volume_quality(
                    mesh, result.geometry, scope=scope, budget=budget
                )
                minimum_scaled, minimum_mean, maximum_aspect = (
                    float(np.min(scaled)),
                    float(np.min(mean)),
                    float(np.max(aspect)),
                )
            edge_lower, edge_upper = None, None
            if profile == "curved-quarter-extrusion-quad":
                match request:
                    case phx.meshing.SurfaceMeshingSpec():
                        pass
                    case _:
                        raise FamilyPreparationError(
                            "family.quad_request",
                            f"family.quad_request must retain a surface specification; got {type(request).__name__}",
                            {"request": request},
                        )
                with (
                    coordinate_phase(scope, limits, budget)
                    if receiving_budget is None
                    else nullcontext(budget)
                ):
                    _, lengths, edge_lower, edge_upper = quad_edge_evidence(
                        mesh, result.geometry, request, budget=budget
                    )
                if not np.all(np.isfinite(edge_lower)) or not np.all(
                    np.isfinite(edge_upper)
                ):
                    raise FamilyPreparationError(
                        "_family_quality_acceptance",
                        f"family.edge_bounds must be finite physical edge integration bounds; got lower={edge_lower!r}, upper={edge_upper!r}",
                        {"edge_lower": edge_lower, "edge_upper": edge_upper},
                    )
            elif profile == "curved-frame-sweep-hex":
                _, lengths, edge_lower, edge_upper = mapped_edge_evidence(
                    mesh,
                    result.geometry,
                    request,
                    scope=scope,
                    budget=receiving_budget,
                )
            else:
                edges = _family_mesh_edges(mesh)
                lengths = np.linalg.norm(
                    np.asarray(mesh.coordinates)[np.asarray(edges)[:, 1]]
                    - np.asarray(mesh.coordinates)[np.asarray(edges)[:, 0]],
                    axis=1,
                )
            return FamilyQualityMeasurements(
                values,
                minimum_scaled,
                minimum_mean,
                maximum_aspect,
                lengths,
                edge_lower,
                edge_upper,
            )
