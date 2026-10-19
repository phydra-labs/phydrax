# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bind accepted periodic scalar and compatible fields to their coefficient layouts."""

from collections.abc import Callable

import numpy as np
from jax import Array

import phydrax as phx
from phydrax import BranchDifferentiationPolicy
from phydrax._frozendict import frozendict
from phydrax.discretization import (
    FieldTracePolicy,
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
)
from phydrax.lifecycle import (
    MeshingAcceptedEpoch,
    MeshingFieldStateRole,
    MeshingFiniteElementFieldDeclaration,
    prepare_meshing_field_index_binding,
)
from phydrax.meshing import CellMeshingResult, MeshAdaptationResult
from phydrax.typing import checked


def periodic_field_templates() -> tuple[tuple[str, FiniteElementFieldSpec], ...]:
    """Return the scalar and compatible periodic field transfer declarations."""
    D = phx.discretization
    return (
        ("intensive", D.FiniteElementFieldSpec("u", D.lagrange_element("triangle", 1))),
        (
            "compatible-cochain",
            D.FiniteElementFieldSpec(
                "vector",
                D.form_element(
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
            "compatible-cochain",
            D.FiniteElementFieldSpec(
                "flux",
                D.form_element(
                    "triangle", 1, 1, family="trimmed", proxy="flux", twist="twisted"
                ),
            ),
        ),
    )


@checked
def prepare_periodic_accepted_epoch(
    result: CellMeshingResult,
    spaces: dict[str, FiniteElementDiscretization],
    state: dict[str, Array],
    epoch: int,
    *,
    transition: MeshAdaptationResult | None = None,
    failure_factory: Callable[[str, str, dict[str, object]], RuntimeError],
) -> MeshingAcceptedEpoch:
    """Bind every periodic primitive field to its accepted coefficient layout.

    The declared finite-element owner retains the solver's degree-of-freedom
    identity and records the accepted epoch alongside each coefficient bank.
    """
    D = phx.discretization
    templates = periodic_field_templates()
    names = tuple(field.name for _, field in templates)
    plan = D.FiniteElementPlan(
        result.mesh,
        tuple(field for _, field in templates),
        coordinate_spec=result.geometry,
    )
    owner = plan.prepare_transfer()
    owned = {space.name: space for space in owner.field_spaces}
    # The declared owner must carry exactly the solver spaces' coefficient layout.
    for index, name in enumerate(names):
        declared, solver = owner.dof_maps[index], spaces[name].dof_maps[0]
        if declared.dof_map_id != solver.dof_map_id:
            raise failure_factory(
                "_periodic_qualification_accepted_epoch",
                (
                    f"The declared {name} coefficient layout differs from its solver space."
                ),
                {"declared": declared, "solver": solver},
            )
    declaration = MeshingFiniteElementFieldDeclaration(
        part_name="periodic",
        topology_id=result.mesh.topology_id,
        geometry_layout_id=result.geometry.geometry_layout_id,
        field_space_ids=frozendict({name: owned[name].field_space_id for name in names}),
        value_units=frozendict({name: (phx.units.ONE,) for name in names}),
        maximum_derivative_orders=frozendict({name: 1 for name in names}),
        branch_policy=BranchDifferentiationPolicy.FROZEN_DECISION,
        trace_policy=FieldTracePolicy("cell-sided"),
        state_roles=tuple(
            MeshingFieldStateRole(
                f"periodic:{name}",
                name,
                "coefficients",
                tuple(state[name].shape),
                np.dtype(state[name].dtype),
                epoch,
                index_binding=prepare_meshing_field_index_binding(
                    result, "field/global_coefficient_ids", field_space=owned[name]
                ),
            )
            for name in names
        )
        + (
            MeshingFieldStateRole(
                "periodic:accepted-epoch",
                "u",
                "state-epoch",
                (),
                np.dtype(np.int64),
                epoch,
            ),
        ),
        history_policy="accepted-state",
        numeric_version=owner.numeric_version,
        finite_element_fields=plan.fields,
        precision_policy=plan.precision_policy,
    )
    fields: dict[str, Array | np.ndarray] = {
        f"periodic:{name}": state[name] for name in names
    }
    fields["periodic:accepted-epoch"] = np.asarray(epoch, dtype=np.int64)
    return MeshingAcceptedEpoch(result, (declaration,), fields, transition=transition)
