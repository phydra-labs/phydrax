"""Write the owner-local physical cone restart as a canonical FE composition."""

from __future__ import annotations

from typing import Any

import numpy as np

from tools.meshing_qualification.scenarios.cold_owner_local import (
    _periodic_cone_triangles,
    _source_composition,
)


def write_physical_cone_restart(
    archive: Any,
    source_view: Any,
    geometry_manifest: Any,
    source_discretization: Any,
    source_compiled: Any,
    source_identities: tuple[Any, ...],
    field_layouts: tuple[Any, ...],
    owned_fields: tuple[Any, ...],
    specification: Any,
    adaptation_specification: Any,
    source_graph_plan: Any,
    target_graph_plan: Any,
    *,
    group: Any,
    rank: int,
    step_size: float = 0.05,
) -> tuple[Any, Any, Any]:
    """Archive a real one-step Newmark/J2 state, retaining only owned payloads.

    ``owned_fields`` contains the already-solved P1 and P0 owner rows in the
    ordering of their respective identity plans. The returned material plan is
    local, while its ownership table describes the entire three-rank group.
    Each caller writes its own archive path.
    """
    import jax.numpy as jnp

    import phydrax as phx
    from phydrax.meshing._distribution import (
        _owner_group_allgather,
        _owner_group_fail_if_any,
    )
    from phydrax.meshing._model_codec import write_model_artifact

    from .cold_fe_cone_physics import ConePhysicalWorkset

    equations = phx.equations
    solver = phx.solver
    mechanics = phx.applications.solid_mechanics
    axis = "physical_cone_writer"
    error = None
    try:
        if rank not in (0, 1, 2) or len(group.spec.process_indices) != 3:
            raise ValueError("Physical cone source requires a three-rank group.")
        if not np.isfinite(step_size) or step_size <= 0:
            raise ValueError("Cone time step must be finite and positive.")
        if (
            len(source_identities) != 2
            or len(field_layouts) != 2
            or len(owned_fields) != 2
        ):
            raise ValueError("Physical cone source requires mixed P1/P0 fields.")
        ownerships = tuple(identity.ownership() for identity in source_identities)
        if any(ownership.local_rank != rank for ownership in ownerships):
            raise ValueError("Field identities do not belong to the source rank.")
        keys = np.asarray(ownerships[1].dof_keys, dtype=np.int64)
        if (
            keys.ndim != 2
            or keys.shape[1] != 3
            or (keys.size and (np.any(keys[:, 0] != 2) or np.any(keys[:, 2] != 0)))
        ):
            raise ValueError("Cone fibers require P0 cell ownership keys.")
        cell_ids = keys[:, 1]
        if len(set(map(int, cell_ids))) != cell_ids.size or not np.array_equal(
            np.sort(cell_ids),
            np.sort(np.asarray(source_view.owned_cell_ids, dtype=np.int64)),
        ):
            raise ValueError("P0 owner rows must match the owned source cells exactly.")
        if rank == 2 and cell_ids.size:
            raise ValueError("The physical cone source must leave rank 2 empty.")
        triangles = _periodic_cone_triangles(source_view)
        if any(int(cell) not in triangles for cell in cell_ids):
            raise ValueError("Owned cone cell has no exact lifted geometry.")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _owner_group_fail_if_any(error, group, axis + "_preflight", "physical cone source")

    counts = _owner_group_allgather(
        np.asarray([cell_ids.size], dtype=np.int64), group, axis + "_counts"
    ).reshape(-1)
    if counts[0] == 0 or counts[1] == 0 or counts[2] != 0:
        raise ValueError(
            "Physical cone source requires owned cells on ranks 0 and 1 only."
        )
    padded = np.full((max(int(np.max(counts)), 1),), -1, dtype=np.int64)
    padded[: cell_ids.size] = cell_ids
    gathered = _owner_group_allgather(padded, group, axis + "_cells")

    def site(cell: int) -> Any:
        return equations.MaterialSiteId.for_cell_quadrature(
            cell,
            "centroid",
            domain_id="periodic-cone",
            quadrature_rule_id="tetra-centroid",
        )

    authority = equations.MaterialSiteOwnership(
        tuple(
            (site(int(cell)), owner)
            for owner, count in enumerate(counts)
            for cell in gathered[owner, : int(count)]
        ),
        part_count=3,
    )
    law = mechanics.FiniteStrainJ2MPMConstitutivePlan()
    model = mechanics.FiniteStrainConstitutiveModel(law, 1.0)
    plan = equations.MaterialIntegrationPlan(
        tuple((site(int(cell)), model) for cell in cell_ids),
        ownership=authority,
        owner_rank=rank,
    )
    rows = [[], [], []]
    states = []
    error = None
    try:
        workset = ConePhysicalWorkset()
        vertices_bank = []
        velocities = []
        initial_states = []
        for cell in cell_ids:
            _, triangle = triangles[int(cell)]
            vertices = np.asarray(
                [(float(x), float(y), 0.0) for x, y in triangle] + [(0.0, 0.0, 1.0)],
                dtype=np.float64,
            )
            vertices_bank.append(vertices)
            velocities.append(
                vertices @ np.diag((4.0, -2.0, -2.0)) * (1.0 + 0.01 * (int(cell) % 17))
            )
            initial_states.append(
                equations.MaterialState(
                    site(int(cell)), model.model_id, law.initialize_state((), jnp.float64)
                )
            )
        if initial_states:
            vertices = np.asarray(vertices_bank)
            velocity = np.asarray(velocities)
            zero = np.zeros_like(vertices)
            accepted, trials = workset.advance(
                vertices,
                zero,
                velocity,
                zero,
                initial_states,
                0.0,
                0,
                0,
                step_size,
                pending_increment=0.01 * velocity,
            )
            if accepted.materials is None or len(accepted.materials.states) != len(
                initial_states
            ):
                raise RuntimeError("Native cone step lost its material state.")
            states.extend(
                accepted.materials.state(item.site_id).with_trial(
                    trials.state(item.site_id).trial
                )
                for item in initial_states
            )
            for target, value in zip(
                rows,
                (accepted.displacement, accepted.velocity, accepted.acceleration),
                strict=True,
            ):
                target.extend(np.asarray(value))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _owner_group_fail_if_any(error, group, axis + "_advance", "physical cone advance")

    materials = equations.MaterialTransaction(
        states, ownership=authority, owner_rank=rank
    )
    plan.validate(materials)
    state = solver.FiniteElementAcceptedState(
        owned_fields,
        np.asarray(step_size, dtype=np.float64),
        1,
        source_view.source_topology_id,
        source_discretization.prepared_id,
        source_compiled.compilation_id,
        field_layouts=field_layouts,
        materials=materials,
        state_version=1,
        field_ownership=ownerships,
    )
    names = ("cone/displacement", "cone/velocity", "cone/acceleration")
    fibers = tuple(
        (name, np.asarray(values, dtype=np.float64).reshape((-1, 4, 3)))
        for name, values in zip(names, rows, strict=True)
    )
    role = solver.FiniteElementRestartStateRole
    association = solver.FiniteElementRestartStateAssociation
    layout = solver.FiniteElementRestartStateLayout
    manifest = solver.FiniteElementRestartManifest(
        state,
        auxiliary_state=fibers,
        auxiliary_layouts=tuple(
            layout(name, role.HISTORY, association.OWNED_ROWS, ownership=ownerships[1])
            for name in names
        ),
        integrator_state=(("cone/timestep", np.asarray(step_size, dtype=np.float64)),),
        integrator_layouts=(
            layout("cone/timestep", role.HISTORY, association.REPLICATED),
        ),
        material_layouts=tuple(
            layout(item.site_id.key, role.MODEL_STATE, association.OWNED_SITES)
            for item in materials.states
        ),
    )
    composition = _source_composition(
        source_view,
        geometry_manifest,
        manifest,
        field_layouts,
        specification,
        adaptation_specification,
        source_graph_plan,
        target_graph_plan,
        archive_safe=True,
    )
    write_model_artifact(archive, composition, limits=None)
    return manifest, plan, composition
