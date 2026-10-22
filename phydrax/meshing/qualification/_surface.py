# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Accepted surface fields and their checkpoint declarations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, final

import numpy as np
from jax import Array

import phydrax as phx
from phydrax import BranchDifferentiationPolicy
from phydrax.discretization import FieldTracePolicy

from ..._frozendict import frozendict
from ...lifecycle._meshing_field_records import (
    MeshingFieldStateRole,
    MeshingFiniteElementFieldDeclaration,
    prepare_meshing_field_index_binding,
)
from ...typing import checked


@final
class SurfaceAcceptedEpoch(phx.StrictModule):
    result: phx.meshing.CellMeshingResult
    fields: tuple[Array, ...]
    material: phx.equations.MaterialTransaction | None
    error: float
    density_content: float


@final
class SurfaceCheckpointState(phx.StrictModule):
    fields: dict[str, Array | np.ndarray]
    declaration: MeshingFiniteElementFieldDeclaration
    owner: phx.discretization.FiniteElementDiscretization
    values: dict[str, Array]
    material_identity: dict[str, str | int] | None


def surface_field_templates() -> tuple[
    tuple[
        phx.discretization.SourceRealizationFieldSemantics,
        phx.discretization.FiniteElementFieldSpec,
    ],
    ...,
]:
    """Declare intensive H1, conservative DG, circulation, and flux fields.

    Density retains scalar P1 order with cell-local coefficients so physical
    inventory uses the conservative-density projection contract.
    """
    return (
        (
            "intensive",
            phx.discretization.FiniteElementFieldSpec(
                "u", phx.discretization.lagrange_element("triangle", 1)
            ),
        ),
        (
            "conservative-density",
            phx.discretization.FiniteElementFieldSpec(
                "density", phx.discretization.discontinuous_element("triangle", 1)
            ),
        ),
        (
            "intensive",
            phx.discretization.FiniteElementFieldSpec(
                "history", phx.discretization.lagrange_element("triangle", 1)
            ),
        ),
        (
            "material-compatible",
            phx.discretization.FiniteElementFieldSpec(
                "curl",
                phx.discretization.form_element(
                    "triangle",
                    1,
                    1,
                    family="trimmed",
                    proxy="circulation",
                    twist="untwisted",
                ),
            ),
        ),
        (
            "material-compatible",
            phx.discretization.FiniteElementFieldSpec(
                "flux",
                phx.discretization.form_element(
                    "triangle", 1, 1, family="trimmed", proxy="flux", twist="twisted"
                ),
            ),
        ),
    )


@checked
def prepare_surface_fields(
    result: phx.meshing.CellMeshingResult,
    /,
    *,
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
    reuse_prepared: phx.discretization.FiniteElementDiscretization | None = None,
) -> phx.discretization.FiniteElementDiscretization:
    """Prepare declared fields on the caller's accepted-carrier coordinate ledger."""
    fields = tuple(field for _, field in surface_field_templates())
    return phx.discretization.FiniteElementPlan(
        result.mesh, fields, coordinate_spec=result.geometry, budget=budget
    ).prepare(reuse_prepared=reuse_prepared, budget=budget)


@checked
def prepare_surface_checkpoint(
    accepted: SurfaceAcceptedEpoch,
    epoch: int,
    /,
    *,
    failure_factory: Callable[[str, str, dict[str, Any]], RuntimeError],
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
    reuse_prepared: phx.discretization.FiniteElementDiscretization | None = None,
) -> SurfaceCheckpointState:
    """Bind accepted coefficients and material history to their carrier indices."""
    names = ("u", "density", "history", "curl", "flux")
    carrier = accepted.result
    owner = prepare_surface_fields(
        carrier, budget=budget, reuse_prepared=reuse_prepared
    )
    spaces = {space.name: space for space in owner.field_spaces}
    values = dict(zip(names, accepted.fields, strict=True))
    arrays: dict[str, Array | np.ndarray] = {
        f"surface:{name}": values[name] for name in names
    }
    roles = [
        MeshingFieldStateRole(
            f"surface:{name}",
            name,
            "coefficients",
            tuple(values[name].shape),
            np.dtype(values[name].dtype),
            epoch,
            index_binding=prepare_meshing_field_index_binding(
                carrier, "field/global_coefficient_ids", field_space=spaces[name]
            ),
        )
        for name in names
    ]
    arrays["surface:accepted-epoch"] = np.asarray(epoch, dtype=np.int64)
    roles.append(
        MeshingFieldStateRole(
            "surface:accepted-epoch",
            "u",
            "state-epoch",
            (),
            np.dtype(np.int64),
            epoch,
        )
    )
    material = accepted.material
    material_identity = None
    if material is not None:
        state = material.state("surface-history")
        if len(material.states) != 1 or state.model_id != "surface-history":
            raise failure_factory(
                "_surface_qualification_checkpoint",
                ("Surface material checkpoint requires its authored site declaration."),
                {"material": material, "state": state},
            )
        arrays["surface:material"] = state.committed
        arrays["surface:material-trial"] = state.trial
        for entry in ("surface:material", "surface:material-trial"):
            roles.append(
                MeshingFieldStateRole(
                    entry,
                    "history",
                    "material-history",
                    tuple(arrays[entry].shape),
                    np.dtype(arrays[entry].dtype),
                    epoch,
                    value_units=(phx.units.KELVIN,),
                    index_binding=prepare_meshing_field_index_binding(
                        carrier, "mesh/cell_ids"
                    ),
                )
            )
        material_identity = {
            "state_version": state.state_version,
            "transaction_id": material.transaction_id,
        }
    declaration = MeshingFiniteElementFieldDeclaration(
        part_name="generation",
        topology_id=carrier.mesh.topology_id,
        geometry_layout_id=carrier.geometry.geometry_layout_id,
        field_space_ids=frozendict({name: spaces[name].field_space_id for name in names}),
        value_units=frozendict({name: (phx.units.ONE,) for name in names}),
        maximum_derivative_orders=frozendict({name: 1 for name in names}),
        branch_policy=BranchDifferentiationPolicy.FROZEN_DECISION,
        trace_policy=FieldTracePolicy("cell-sided"),
        state_roles=tuple(roles),
        history_policy="accepted-state",
        numeric_version=owner.numeric_version,
        finite_element_fields=owner.construction_plan.fields,
        precision_policy=owner.construction_plan.precision_policy,
    )
    return SurfaceCheckpointState(
        fields=arrays,
        declaration=declaration,
        owner=owner,
        values=values,
        material_identity=material_identity,
    )
