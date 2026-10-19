#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._array_archive import ArrayArchiveLimits
from phydrax._frozendict import frozendict
from phydrax.discretization import FiniteElementDiscretization
from phydrax.discretization._conservation_boundary import ExtrapolationBoundary
from phydrax.discretization._views import FieldTracePolicy
from phydrax.discretization.fem import form_element
from phydrax.discretization.fem._generic import FiniteElementFieldSpec
from phydrax.discretization.fem._precision import FiniteElementPrecisionPolicy
from phydrax.discretization.finite_volume import (
    PiecewiseConstantReconstruction,
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from phydrax.discretization.finite_volume._riemann import RusanovFluxPlan
from phydrax.lifecycle import (
    MeshingFieldDeclaration,
    MeshingFieldStateRole,
    MeshingFiniteElementFieldDeclaration,
    MeshingFiniteVolumeFieldDeclaration,
    prepare_meshing_field_index_binding,
)
from phydrax.linalg import ArraySpace
from phydrax.meshing import (
    MeshPart,
    OversetPartSpec,
    OversetPolicy,
    OversetRegistration,
    prepare_overset_connectivity,
)
from phydrax.meshing._result import CellMeshingResult
from tests._support.native_family_generation import native_family_generation


@dataclass(frozen=True, slots=True)
class FieldCarrier:
    part: MeshPart
    source: Any
    specification: Any
    options: Any
    carrier: CellMeshingResult
    wall: OversetPartSpec
    companion: MeshPart
    companion_wall: OversetPartSpec
    registration: OversetRegistration


@dataclass(frozen=True, slots=True)
class FieldDiscretization:
    original: FiniteElementDiscretization | UnstructuredFiniteVolumeDiscretization
    coefficients: np.ndarray
    maximum_order: int
    owner: Literal["finite_element", "finite_volume"]
    constructor_options: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FieldBanks:
    fields: dict[str, np.ndarray]
    roles: tuple[MeshingFieldStateRole, ...]
    history_policy: Literal["none", "material-history"]


@dataclass(frozen=True, slots=True)
class NativeFieldScenario:
    part: MeshPart
    carrier: CellMeshingResult
    original: FiniteElementDiscretization | UnstructuredFiniteVolumeDiscretization
    coefficients: np.ndarray
    fields: dict[str, np.ndarray]
    name: str
    declaration: MeshingFieldDeclaration
    closure: dict[str, Any]
    limits: ArrayArchiveLimits
    report: Any


def prepare_field_carrier(field_kind: str, /) -> FieldCarrier:
    """Publish the native carrier and its two-part registration."""
    family = (
        "mapped"
        if field_kind.startswith("mapped")
        else "polyhedral"
        if field_kind.startswith("fv")
        else "plc"
    )
    part, source, specification, options = native_family_generation(family)
    carrier = part.carrier
    assert isinstance(carrier, CellMeshingResult)
    wall = OversetPartSpec(part.name)
    companion = phx.meshing.MeshPart("companion", carrier)
    companion_wall = OversetPartSpec(companion.name)
    registration = prepare_overset_connectivity(
        phx.meshing.MeshAssembly((part, companion)),
        (wall, companion_wall),
        policy=OversetPolicy(fringe_layers=1),
    ).registration()
    return FieldCarrier(
        part,
        source,
        specification,
        options,
        carrier,
        wall,
        companion,
        companion_wall,
        registration,
    )


def prepare_field_discretization(
    field_kind: str, carrier: CellMeshingResult, /
) -> FieldDiscretization:
    """Prepare the declared basis or reconstruction and its coefficient state."""
    name = "physical"
    owner: Literal["finite_element", "finite_volume"]
    flux_policy: RusanovFluxPlan | None = None
    boundary_policies: frozendict[str, ExtrapolationBoundary] = frozendict()
    if field_kind.startswith("fv"):
        system = (
            phx.equations.EulerSystem(
                3, material=phx.equations.IdealGasMaterial(1.67, 287.0)
            )
            if field_kind.endswith("history")
            else None
        )
        components = system.component_names if system is not None else ("temperature",)
        original = UnstructuredFiniteVolumePlan.from_cell_mesh(
            carrier.mesh,
            field_name=name,
            component_names=components,
        ).prepare(numeric_version="accepted-7")
        assert isinstance(original, UnstructuredFiniteVolumeDiscretization)
        fv_original = original
        owner = "finite_volume"
        finite_volume_field_name = name
        finite_volume_component_names = components
        coordinate_policy = "represented-polyhedral"
        reconstruction_policy = PiecewiseConstantReconstruction()
        reconstruction_policy_id = reconstruction_policy.plan_id
        coefficient_space = original.field_spaces[0].vector_space
        assert isinstance(coefficient_space, ArraySpace)
        shape = coefficient_space.shape
        coefficients = np.arange(prod(shape), dtype=np.float64).reshape(shape) + 2.0
        if system is not None:
            primitive = np.tile(
                np.asarray((1.0, 0.0, 0.0, 0.0, 1.0)), (original.cell_count, 1)
            )
            primitive[:, 0] += 0.1 * np.asarray(fv_original.cell_centers)[:, 0]
            coefficients = np.asarray(
                system.primitive_to_conserved(jnp.asarray(primitive))
            ).copy()
            flux_policy = RusanovFluxPlan()
            boundary_policies = frozendict(
                {
                    patch: ExtrapolationBoundary()
                    for patch in original.boundary_patch_names
                }
            )
        maximum_order = 0
        constructor_options: dict[str, Any] = {
            "finite_volume_field_name": finite_volume_field_name,
            "finite_volume_component_names": finite_volume_component_names,
            "finite_volume_coordinate_policy": coordinate_policy,
            "reconstruction_policy": reconstruction_policy,
            "reconstruction_policy_id": reconstruction_policy_id,
            "finite_volume_system": system,
            "finite_volume_flux_policy": flux_policy,
            "finite_volume_boundaries": boundary_policies,
        }
    else:
        if field_kind == "rt":
            element = form_element("tetrahedron", 2, 1, twist="twisted", proxy="flux")
        elif field_kind == "bdm2":
            element = form_element(
                "tetrahedron", 2, 2, family="full", twist="twisted", proxy="flux"
            )
        elif field_kind.startswith("nedelec"):
            element = form_element(
                "tetrahedron",
                1,
                2 if field_kind == "nedelec1" else 1,
                proxy="circulation",
            )
        else:
            element = phx.discretization.lagrange_element("hexahedron", 2)
        field = phx.discretization.FiniteElementFieldSpec(
            name,
            {block.name: element for block in carrier.mesh.blocks},
            component_shape=(2,) if field_kind == "mapped-vector" else (),
        )
        precision = FiniteElementPrecisionPolicy()
        original = phx.discretization.FiniteElementPlan(
            carrier.mesh,
            field,
            coordinate_spec=carrier.geometry,
            precision_policy=precision,
        ).prepare(numeric_version="accepted-7")
        owner = "finite_element"
        finite_element_fields = (field,)
        precision_policy = precision
        coefficient_space = original.field_spaces[0].vector_space
        assert isinstance(coefficient_space, ArraySpace)
        shape = coefficient_space.shape
        coefficients = (
            np.arange(prod(shape), dtype=np.float64).reshape(shape) * 0.125 + 1.0
        )
        maximum_order = 1
        constructor_options = {
            "finite_element_fields": finite_element_fields,
            "precision_policy": precision_policy,
        }
    return FieldDiscretization(
        original, coefficients, maximum_order, owner, constructor_options
    )


def prepare_field_banks(
    field_kind: str, carrier: CellMeshingResult, field: FieldDiscretization, /
) -> FieldBanks:
    """Bind coefficients, boundary values, and history to scientific rows."""
    original, coefficients = field.original, field.coefficients
    name, shape = "physical", coefficients.shape
    fields: dict[str, np.ndarray] = {}
    history_policy: Literal["none", "material-history"]
    coefficients.flags.writeable = False
    fields["physical-coefficients"] = coefficients
    coefficient_binding = prepare_meshing_field_index_binding(
        carrier,
        "mesh/cell_ids"
        if field_kind.startswith("fv")
        else "field/global_coefficient_ids",
        field_space=original.field_spaces[0],
    )
    roles = [
        MeshingFieldStateRole(
            "physical-coefficients",
            name,
            "coefficients",
            shape,
            coefficients.dtype,
            7,
            index_binding=coefficient_binding,
        )
    ]
    history_policy = "none"
    if field_kind == "mapped-scalar":
        assert isinstance(original, FiniteElementDiscretization)
        boundary_values = np.asarray(original.dof_maps[0].dof_coordinates) @ np.asarray(
            (1.0, 2.0, 3.0)
        )
        boundary_mask = np.asarray(original.dof_maps[0].boundary_dof_mask).copy()
        boundary_values.flags.writeable = boundary_mask.flags.writeable = False
        fields["essential-values"] = boundary_values
        fields["essential-mask"] = boundary_mask
        roles.extend(
            (
                MeshingFieldStateRole(
                    "essential-values",
                    name,
                    "boundary-values",
                    boundary_values.shape,
                    boundary_values.dtype,
                    7,
                    index_binding=coefficient_binding,
                ),
                MeshingFieldStateRole(
                    "essential-mask",
                    name,
                    "boundary-mask",
                    boundary_mask.shape,
                    boundary_mask.dtype,
                    7,
                    index_binding=coefficient_binding,
                ),
            )
        )
    if field_kind.endswith("history"):
        if not isinstance(original, UnstructuredFiniteVolumeDiscretization):
            raise TypeError(
                "FieldDiscretization.original must be finite-volume for material history; got finite-element discretization"
            )
        fraction = 0.2 + 0.1 * np.asarray(original.cell_centers)[:, 0]
        history = np.column_stack((fraction, 1.0 - fraction))
        history.flags.writeable = False
        fields["material-history"] = history
        fields["state-epoch"] = np.asarray(7, dtype=np.int64)
        roles.extend(
            (
                MeshingFieldStateRole(
                    "material-history",
                    name,
                    "material-history",
                    history.shape,
                    history.dtype,
                    7,
                    (phx.units.ONE,),
                    index_binding=prepare_meshing_field_index_binding(
                        carrier, "mesh/cell_ids"
                    ),
                ),
                MeshingFieldStateRole(
                    "state-epoch", name, "state-epoch", (), np.dtype("int64"), 7
                ),
            )
        )
        history_policy = "material-history"
    return FieldBanks(fields, tuple(roles), history_policy)


def prepare_native_field_scenario(field_kind: str, /) -> NativeFieldScenario:
    """Assemble native field declarations and the accepted source closure."""
    generation = prepare_field_carrier(field_kind)
    field = prepare_field_discretization(field_kind, generation.carrier)
    banks = prepare_field_banks(field_kind, generation.carrier, field)
    part, source = generation.part, generation.source
    specification, options = generation.specification, generation.options
    carrier, wall = generation.carrier, generation.wall
    companion, companion_wall = generation.companion, generation.companion_wall
    registration = generation.registration
    original, coefficients = field.original, field.coefficients
    name, maximum_order = "physical", field.maximum_order
    fields, roles, history_policy = banks.fields, banks.roles, banks.history_policy
    units = (phx.units.KELVIN,)
    if field_kind == "fv-euler-history":
        density_unit = phx.units.derived_unit(
            "kg/m^3", ((phx.units.KILOGRAM, 1), (phx.units.METER, -3))
        )
        momentum_unit = phx.units.derived_unit(
            "kg/(m^2*s)",
            ((phx.units.KILOGRAM, 1), (phx.units.METER, -2), (phx.units.SECOND, -1)),
        )
        units = (
            density_unit,
            momentum_unit,
            momentum_unit,
            momentum_unit,
            phx.units.PASCAL,
        )
    common_declaration: dict[str, Any] = dict(
        part_name=part.name,
        topology_id=carrier.mesh.topology_id,
        geometry_layout_id=carrier.geometry.geometry_layout_id,
        field_space_ids=frozendict({name: original.field_spaces[0].field_space_id}),
        value_units=frozendict({name: units}),
        maximum_derivative_orders=frozendict({name: maximum_order}),
        branch_policy=phx.BranchDifferentiationPolicy.SMOOTH,
        trace_policy=FieldTracePolicy("cell-sided"),
        state_roles=tuple(roles),
        history_policy=history_policy,
        numeric_version="accepted-7",
        source_wall_policy=wall,
    )
    match field.owner:
        case "finite_element":
            declaration = MeshingFiniteElementFieldDeclaration(
                **common_declaration, **field.constructor_options
            )
        case "finite_volume":
            declaration = MeshingFiniteVolumeFieldDeclaration(
                **common_declaration, **field.constructor_options
            )
        case _:
            raise ValueError(
                f"FieldDiscretization.owner must select FE or FV; got {field.owner!r}"
            )
    companion_roles = tuple(
        replace(role, entry_name=f"companion/{role.entry_name}") for role in roles
    )
    companion_declaration = replace(
        declaration,
        part_name=companion.name,
        source_wall_policy=companion_wall,
        state_roles=companion_roles,
    )
    fields.update(
        {f"companion/{key}": value.copy() for key, value in tuple(fields.items())}
    )
    report = carrier.certification
    assert report is not None
    closure = {
        "certification_inputs": report.request,
        "report": report,
        "associations": carrier.associations,
        "registration": registration,
        "generation_sources": {part.name: source, companion.name: source},
        "primary_generation_part": part.name,
        "generation_specifications": {
            part.name: specification,
            companion.name: specification,
        },
        "generation_options": {part.name: options, companion.name: options},
        "field_declarations": {
            part.name: declaration,
            companion.name: companion_declaration,
        },
        "accepted_data": {"fields": fields},
    }
    limits = ArrayArchiveLimits(
        max_members=8192, max_manifest_nesting=128, max_manifest_bytes=1 << 26
    )
    return NativeFieldScenario(
        part,
        carrier,
        original,
        coefficients,
        fields,
        name,
        declaration,
        closure,
        limits,
        report,
    )


def prepare_single_native_field_scenario() -> NativeFieldScenario:
    """Assemble standalone FE fields and material history without a registration."""
    part, source, specification, options = native_family_generation("plc")
    carrier = part.carrier
    assert isinstance(carrier, phx.meshing.CellMeshingResult)
    field = FiniteElementFieldSpec(
        "temperature", phx.discretization.lagrange_element("tetrahedron", 1)
    )
    precision = FiniteElementPrecisionPolicy()
    original = phx.discretization.FiniteElementPlan(
        carrier.mesh,
        (field,),
        coordinate_spec=carrier.geometry,
        precision_policy=precision,
    ).prepare(numeric_version="accepted-7")
    coefficient_space = original.field_spaces[0].vector_space
    assert isinstance(coefficient_space, phx.linalg.ArraySpace)
    shape = coefficient_space.shape
    coefficients = np.full(shape, 42.0, dtype=np.float64)
    history = np.full(shape, 37.0, dtype=np.float64)
    coefficient_binding = prepare_meshing_field_index_binding(
        carrier, "mesh/vertex_ids", field_space=original.field_spaces[0]
    )
    history_binding = prepare_meshing_field_index_binding(carrier, "mesh/vertex_ids")
    roles = (
        MeshingFieldStateRole(
            "temperature",
            "temperature",
            "coefficients",
            shape,
            np.dtype(np.float64),
            7,
            index_binding=coefficient_binding,
        ),
        MeshingFieldStateRole(
            "material-temperature",
            "temperature",
            "material-history",
            shape,
            np.dtype(np.float64),
            7,
            value_units=(phx.units.KELVIN,),
            index_binding=history_binding,
        ),
        MeshingFieldStateRole(
            "accepted-epoch", "temperature", "state-epoch", (), np.dtype(np.int64), 7
        ),
    )
    declaration = MeshingFiniteElementFieldDeclaration(
        part_name=part.name,
        topology_id=carrier.mesh.topology_id,
        geometry_layout_id=carrier.geometry.geometry_layout_id,
        field_space_ids=frozendict(
            {"temperature": original.field_spaces[0].field_space_id}
        ),
        value_units=frozendict({"temperature": (phx.units.KELVIN,)}),
        maximum_derivative_orders=frozendict({"temperature": 1}),
        branch_policy=phx.BranchDifferentiationPolicy.SMOOTH,
        trace_policy=FieldTracePolicy("cell-sided"),
        state_roles=roles,
        history_policy="material-history",
        numeric_version="accepted-7",
        finite_element_fields=(field,),
        precision_policy=precision,
    )
    report = carrier.certification
    assert report is not None
    closure = {
        "certification_inputs": report.request,
        "report": report,
        "associations": carrier.associations,
        "generation_part": part,
        "generation_source": source,
        "generation_specification": specification,
        "generation_options": options,
        "field_declarations": {part.name: declaration},
        "accepted_data": {
            "fields": {
                "temperature": coefficients,
                "material-temperature": history,
                "accepted-epoch": np.asarray(7, dtype=np.int64),
            }
        },
    }
    phx.lifecycle.validate_meshing_field_declarations(closure)
    limits = ArrayArchiveLimits(
        max_members=8192, max_manifest_nesting=128, max_manifest_bytes=1 << 26
    )
    return NativeFieldScenario(
        part,
        carrier,
        original,
        coefficients,
        closure["accepted_data"]["fields"],
        "temperature",
        declaration,
        closure,
        limits,
        report,
    )
