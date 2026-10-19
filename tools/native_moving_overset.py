# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native wall cutting, moving donor refresh, Poisson update and durable restart.

Run: PYTHONPATH=. python -m tools.native_moving_overset
Interpolation uses each P1 FE layout and is explicitly nonconservative.
The moving square annulus has a protected solid wall and an artificial exterior.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import ArrayArchiveLimits, config as pxconfig
from phydrax.discretization import PolygonalConnectivity
from phydrax.meshing import (
    CellMeshingResult,
    OversetPartSpec,
    OversetPolicy,
    OversetVertexStatus,
    prepare_overset_connectivity,
    prepare_overset_field_transfer,
    prepare_overset_motion_rebind,
)
from phydrax.meshing.providers import NativePlanarSource
from phydrax.meshing.qualification import (
    field_declarations,
    field_entries,
    field_plans,
    native_planar_part,
    poisson_update,
    restart_native,
    rigid_planar_successor,
)


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))


def main() -> None:
    background, background_plan = native_planar_part("background", 3.0)
    body, moving_plan = native_planar_part("moving", 2.25, cavity=0.25)
    generation_plans = {"background": background_plan, "moving": moving_plan}
    carrier = body.carrier
    if not isinstance(carrier, CellMeshingResult):
        raise TypeError(
            f"main.carrier must be an accepted CellMeshingResult; got {type(carrier).__name__!r}."
        )
    mesh = carrier.mesh
    incidence = mesh.connectivity
    if not isinstance(incidence, PolygonalConnectivity):
        raise TypeError(
            f"main.incidence must be PolygonalConnectivity; got {type(incidence).__name__!r}."
        )
    boundary = np.asarray(incidence.boundary_edges)
    corners = np.asarray(mesh.coordinates)[np.asarray(incidence.edges)]
    wall = boundary & np.all(np.abs(corners) <= 0.25, axis=(1, 2))
    identifiers = np.asarray(mesh.entity_set(1).entity_ids)
    connectivity = prepare_overset_connectivity(
        phx.meshing.MeshAssembly((background, body)),
        (
            OversetPartSpec(background.name),
            OversetPartSpec(
                body.name,
                wall=body.scope(1, identifiers[wall]),
                boundary=body.scope(1, identifiers[boundary & ~wall]),
            ),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    connectivity.require_complete()
    physical_plans = field_plans(connectivity)
    discretizations = {name: plan.prepare() for name, plan in physical_plans.items()}
    coefficients = {
        name: 5 + jnp.sum(field.dof_maps[0].dof_coordinates, axis=1)
        for name, field in discretizations.items()
    }
    prepared, states = field_entries(
        connectivity, discretizations, coefficients, "initial"
    )
    composition = phx.lifecycle.Composition(
        (*connectivity.composition_entries(), *prepared, *states),
        boundary_id="accepted-poisson-step",
    )
    residuals = []
    for step in range(2):
        moving = connectivity.assembly.part("moving")
        offset = 0.55 if step == 0 else 0.05
        successor, moving_plan = rigid_planar_successor(
            moving,
            np.asarray((offset, 0.0)),
            motion_id=f"native-motion:{step}",
            plan=generation_plans["moving"],
        )
        generation_plans["moving"] = moving_plan
        candidate = connectivity.reregister({moving.name: successor})
        candidate.require_complete()
        physical_plans = field_plans(candidate)
        next_fields = {name: plan.prepare() for name, plan in physical_plans.items()}
        route = prepare_overset_field_transfer(
            candidate, next_fields, "u", previous=connectivity
        )
        receptors = route.apply(coefficients)
        next_values = dict(
            coefficients
        )  # Material-attached active DOFs retain their values.
        for evidence in route.receptors:
            name = evidence.part_name
            next_values[name] = (
                next_values[name].at[evidence.receptor_rows].set(receptors[name])
            )
        prepared, states = field_entries(
            candidate, next_fields, next_values, f"motion-{step}"
        )
        old_states = tuple(
            composition.entry(f"temperature:{name}") for name in sorted(coefficients)
        )
        transport = phx.lifecycle.CompositionTransport(
            "physical-remap",
            tuple(entry.entry_id for entry in old_states),
            states,
            source_structure_ids=tuple(entry.structure_id for entry in old_states),
            route_id=route.transfer_id,
            successful=True,
        )
        motion = prepare_overset_motion_rebind(
            composition,
            connectivity,
            candidate,
            transports=(transport,),
            reprepare=prepared,
        )
        refused = motion.commit(accepted_boundary=False)
        if refused.published or refused.composition is not composition:
            raise RuntimeError(
                f"main.refused must remain unpublished and retain the accepted composition; got {(refused.published, refused.composition)!r}."
            )
        receipt = motion.commit(accepted_boundary=True)
        if not receipt.published:
            raise RuntimeError(
                f"main.published must be True for the requested moving registration; got {receipt.published!r}."
            )
        connectivity, composition, discretizations = (
            candidate,
            receipt.composition,
            next_fields,
        )
        coefficients = next_values
        coefficients["moving"], residual = poisson_update(
            discretizations["moving"], coefficients["moving"]
        )
        residuals.append(residual)
        prepared, states = field_entries(
            connectivity, discretizations, coefficients, f"poisson-{step}"
        )
        composition = phx.lifecycle.Composition(
            (*connectivity.composition_entries(), *prepared, *states),
            boundary_id="accepted-poisson-step",
        )

    archive_limits = ArrayArchiveLimits(
        max_members=4096,
        max_manifest_bytes=8 * 1024 * 1024,
        max_central_directory_bytes=4 * 1024 * 1024,
        max_manifest_nesting=64,
    )
    declarations = field_declarations(
        connectivity, physical_plans, discretizations, coefficients
    )
    restored, restored_values, restored_generation, restored_fields = restart_native(
        connectivity,
        coefficients,
        plans=generation_plans,
        declarations=declarations,
        archive_limits=archive_limits,
    )
    replay, residual = poisson_update(
        restored_fields["moving"], restored_values["moving"]
    )
    uninterrupted, _ = poisson_update(discretizations["moving"], coefficients["moving"])
    np.testing.assert_array_equal(replay, uninterrupted)
    authority_preserved = True
    for name, generation_plan in generation_plans.items():
        original_source = generation_plan.source
        restored_source = restored_generation[name].source
        if not isinstance(original_source, NativePlanarSource) or not isinstance(
            restored_source, NativePlanarSource
        ):
            raise TypeError(
                f"main.sources must be NativePlanarSource objects after restart; got {(type(original_source).__name__, type(restored_source).__name__)!r}."
            )
        authority_preserved &= restored_source.binding_id == original_source.binding_id
    print(
        {
            "epochs": connectivity.epoch,
            "orphans": connectivity.orphan_count,
            "protected_wall_conflicts": connectivity.conflict_count,
            "background_holes": int(
                connectivity.blanking_of("background")
                .vertex_ids_with(OversetVertexStatus.HOLE)
                .size
            ),
            "poisson_residuals": residuals + [residual],
            "rollback_preserved": True,
            "restart_reproduced": restored.connectivity_id
            == connectivity.connectivity_id,
            "source_authority_preserved": authority_preserved,
            "interpolative_conservative": False,
        }
    )


if __name__ == "__main__":
    main()
