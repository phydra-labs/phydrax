"""Cold owner-local continuation of the physical periodic-cone restart."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .cold_owner_local import (
    _periodic_cone_triangles,
    _refine_periodic_cone_materials,
    _transfer_periodic_cone_kinematics,
)


def continue_cone_restart(
    composition: Any,
    manifest: Any,
    source_view: Any,
    adaptation_view: Any,
    target_view: Any,
    source_compiled: Any,
    adaptation_compiled: Any,
    target_compiled: Any,
    source_identities: Any,
    adaptation_identities: Any,
    target_identities: Any,
    adaptive_result: Any,
    migration: Any,
    group: Any,
    adaptation_reprepared_entries: Any,
    target_reprepared_entries: Any,
    retained_mesh_entry_ids: Any,
    *,
    rank: int,
) -> dict[str, Any]:
    """Adapt a restored cone bank, migrate it, and independently continue it.

    Only site ownership tables and per-cell digests are exchanged for the
    oracle; the material and kinematic banks remain on their owners.
    """
    import jax

    import phydrax as phx
    from phydrax.applications.solid_mechanics import (
        FiniteStrainConstitutiveModel,
        FiniteStrainJ2MPMConstitutivePlan,
    )
    from phydrax.meshing._distribution import (
        _owner_group_allgather,
        _owner_group_fail_if_any,
    )

    from .cold_fe_cone_physics import ConePhysicalWorkset

    parts = len(group.spec.process_indices)

    def guarded(phase: str, action: Any) -> Any:
        error = None
        value = None
        try:
            value = action()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _owner_group_fail_if_any(
            error, group, f"cold_cone_{phase}", "cold cone continuation"
        )
        return value

    def preflight() -> None:
        if parts != 3 or group.spec.process_indices.index(jax.process_index()) != rank:
            raise ValueError(
                "Cone restart requires the declared three-owner execution group."
            )
        if (
            composition.entry("solver/state").value.accepted_id
            != manifest.state.accepted_id
        ):
            raise ValueError("Restored composition and cone restart manifest differ.")

    guarded("preflight", preflight)
    names = ("cone/displacement", "cone/velocity", "cone/acceleration")
    clock_name = "cone/timestep"

    def gather(rows: Any, width: int) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64).reshape(-1, width)
        counts = np.asarray(
            _owner_group_allgather(
                np.asarray([len(rows)], dtype=np.int64), group, "cold_cone_rows_count"
            )
        ).reshape(-1)
        padded = np.full((int(np.max(counts)), width), -1, dtype=np.int64)
        padded[: len(rows)] = rows
        all_rows = np.asarray(
            _owner_group_allgather(padded, group, "cold_cone_rows")
        ).reshape(parts, -1, width)
        return np.concatenate(tuple(all_rows[i, : int(counts[i])] for i in range(parts)))

    def site(cell: int) -> Any:
        return phx.equations.MaterialSiteId.for_cell_quadrature(
            cell,
            "centroid",
            domain_id="periodic-cone",
            quadrature_rule_id="tetra-centroid",
        )

    law = FiniteStrainConstitutiveModel(FiniteStrainJ2MPMConstitutivePlan(), 1.0)

    def plan(view: Any) -> Any:
        cells = guarded(
            "plan_cells", lambda: np.asarray(view.owned_cell_ids, dtype=np.int64)
        )
        rows = np.column_stack((cells, np.full(cells.shape, rank, dtype=np.int64)))
        gathered = gather(rows, 2)

        def build() -> Any:
            ownership = phx.equations.MaterialSiteOwnership(
                tuple((site(int(cell)), int(owner)) for cell, owner in gathered),
                part_count=parts,
            )
            return phx.equations.MaterialIntegrationPlan(
                tuple((s, law) for s in ownership.for_owner(rank)),
                ownership=ownership,
                owner_rank=rank,
            )

        return guarded("plan_build", build)

    source_plan, adapted_plan, target_plan = (
        plan(v) for v in (source_view, adaptation_view, target_view)
    )
    guarded("source_material", lambda: source_plan.validate(manifest.state.materials))

    def auxiliary_layouts(identities: Any) -> tuple[Any, ...]:
        return tuple(
            phx.solver.FiniteElementRestartStateLayout(
                name,
                phx.solver.FiniteElementRestartStateRole.HISTORY,
                phx.solver.FiniteElementRestartStateAssociation.OWNED_ROWS,
                ownership=identities[1].ownership(),
            )
            for name in names
        )

    def material_layouts(material_plan: Any) -> tuple[Any, ...]:
        return tuple(
            phx.solver.FiniteElementRestartStateLayout(
                s.key,
                phx.solver.FiniteElementRestartStateRole.MODEL_STATE,
                phx.solver.FiniteElementRestartStateAssociation.OWNED_SITES,
            )
            for s in material_plan.site_ids
        )

    adapted_aux, target_aux, adapted_material, target_material = guarded(
        "layouts",
        lambda: (
            auxiliary_layouts(adaptation_identities),
            auxiliary_layouts(target_identities),
            material_layouts(adapted_plan),
            material_layouts(target_plan),
        ),
    )

    common = dict(
        state_entry_id="solver/state",
        geometry_entry_id="mesh/geometry",
        field_entry_ids=(
            ("u", "solver/state/field/u"),
            ("content", "solver/state/field/content"),
        ),
        material_entry_id="solver/state/materials",
        auxiliary_entry_ids=tuple((n, f"solver/restart/auxiliary/{n}") for n in names),
        integrator_entry_ids=((clock_name, f"solver/restart/integrator/{clock_name}"),),
        target_integrator_layouts=manifest.integrator_layouts,
        retain_entry_ids=retained_mesh_entry_ids,
    )

    def transaction(certify: Any, ident: str) -> Any:
        return phx.solver.FiniteElementTopologyTransaction(
            certify,
            material_transfer=_refine_periodic_cone_materials,
            payload_transfers=tuple(
                (("auxiliary", name), _transfer_periodic_cone_kinematics)
                for name in names
            ),
            transaction_id=ident,
        )

    def adapt(tx: Any, axis: str) -> Any:
        return tx.adapt_owner_local(
            manifest,
            source_compiled,
            adaptation_compiled,
            source_identities,
            adaptation_identities,
            adaptive_result,
            composition,
            group,
            source_view,
            adaptation_view,
            target_auxiliary_layouts=adapted_aux,
            target_material_layouts=adapted_material,
            source_material_plan=source_plan,
            target_material_plan=adapted_plan,
            reprepare_entries=adaptation_reprepared_entries,
            axis_name=axis,
            **common,
        )

    def snapshot(candidate: Any) -> tuple[Any, ...]:
        state = candidate.state

        def array(value: Any) -> tuple[str, tuple[int, ...], bytes]:
            item = np.ascontiguousarray(np.asarray(value))
            return str(item.dtype), item.shape, item.tobytes()

        return (
            tuple(array(value) for value in state.fields),
            tuple((name, array(value)) for name, value in candidate.auxiliary_state),
            tuple((name, array(value)) for name, value in candidate.integrator_state),
            tuple(
                (
                    item.site_id.key,
                    item.model_id,
                    item.state_version,
                    array(item.committed),
                    array(item.trial),
                )
                for item in state.materials.states
            ),
            float(state.time),
            state.step,
            state.state_version,
        )

    before = guarded("snapshot", lambda: snapshot(manifest))
    certified = [False]

    def refuse_candidate(*args: Any) -> bool:
        certified[0] = True
        return False

    rejected = adapt(
        transaction(refuse_candidate, "cold-cone-reject"), "cold_cone_reject"
    )

    def check_rollback() -> None:
        if (
            not certified[0]
            or bool(np.asarray(rejected.committed))
            or rejected.state.accepted_id != manifest.state.accepted_id
            or rejected.manifest.manifest_id != manifest.manifest_id
            or (rejected.rebind_receipt is not None and rejected.rebind_receipt.published)
            or snapshot(rejected.manifest) != before
            or snapshot(manifest) != before
        ):
            raise RuntimeError(
                f"Rejected cone adaptation did not roll back atomically: {rejected.diagnostics}"
            )

    guarded("rollback", check_rollback)
    tx = transaction(lambda *args: True, "cold-cone-adapt")
    adapted = adapt(tx, "cold_cone_adapt")

    def check_adapted() -> None:
        if (
            not bool(np.asarray(adapted.committed))
            or adapted.rebind_receipt is None
            or not adapted.rebind_receipt.published
        ):
            raise RuntimeError(f"Cone adaptation refused: {adapted.diagnostics}")

    guarded("adapted", check_adapted)

    def transfer_momenta(restart: Any, view: Any, identities: Any) -> np.ndarray:
        bank = dict(restart.auxiliary_state)
        keys = np.asarray(identities[1].ownership().dof_keys, dtype=np.int64)
        triangles = _periodic_cone_triangles(view)
        totals = np.zeros((2, 3), dtype=np.float64)
        for index, key in enumerate(keys):
            lift = np.asarray(triangles[int(key[1])][1], dtype=np.float64)
            edge1, edge2 = lift[1] - lift[0], lift[2] - lift[0]
            volume = abs(edge1[0] * edge2[1] - edge1[1] * edge2[0]) / 6.0
            for row, name in enumerate(names[1:]):
                totals[row] += (volume / 4.0) * np.sum(bank[name][index], axis=0)
        return totals

    source_momenta = guarded(
        "source_momenta",
        lambda: transfer_momenta(manifest, source_view, source_identities),
    )
    adapted_momenta = guarded(
        "adapted_momenta",
        lambda: transfer_momenta(
            adapted.manifest, adaptation_view, adapted.target_identity_plans
        ),
    )
    source_total = np.sum(
        np.asarray(
            _owner_group_allgather(source_momenta, group, "cold_cone_source_momenta")
        ).reshape(parts, 2, 3),
        axis=0,
    )
    adapted_total = np.sum(
        np.asarray(
            _owner_group_allgather(adapted_momenta, group, "cold_cone_adapted_momenta")
        ).reshape(parts, 2, 3),
        axis=0,
    )
    np.testing.assert_allclose(adapted_total, source_total, rtol=1e-8, atol=1e-10)

    moved = tx.repartition(
        adapted.manifest,
        adaptation_compiled,
        target_compiled,
        adapted.target_identity_plans,
        target_identities,
        migration,
        adapted.rebind_receipt.composition,
        group,
        adaptation_view,
        target_view,
        target_auxiliary_layouts=target_aux,
        target_material_layouts=target_material,
        source_material_plan=adapted_plan,
        target_material_plan=target_plan,
        reprepare_entries=target_reprepared_entries,
        axis_name="cold_cone_repartition",
        **common,
    )

    def check_moved() -> None:
        if (
            not bool(np.asarray(moved.committed))
            or moved.rebind_receipt is None
            or not moved.rebind_receipt.published
        ):
            raise RuntimeError(f"Cone repartition refused: {moved.diagnostics}")
        if adapted_plan.ownership.ownership_id == target_plan.ownership.ownership_id:
            raise RuntimeError("Cone continuation did not change material placement.")
        if (
            len(
                {
                    manifest.manifest_id,
                    adapted.manifest.manifest_id,
                    moved.manifest.manifest_id,
                }
            )
            != 3
        ):
            raise RuntimeError(
                "Source, adapted and migrated cone restart identities are not distinct."
            )

    guarded("moved", check_moved)

    def digest(*values: Any) -> bytes:
        h = hashlib.sha256()
        for value in values:
            arr = np.ascontiguousarray(np.asarray(value))
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        return h.digest()

    def changed(a: Any, b: Any) -> bool:
        return not np.allclose(np.asarray(a), np.asarray(b), rtol=1e-8, atol=1e-10)

    workset = ConePhysicalWorkset()
    momentum_evidence: list[dict[str, Any]] = []

    def input_rows(result: Any, identities: Any) -> np.ndarray:
        state = result.state
        bank = dict(result.manifest.auxiliary_state)
        keys = np.asarray(identities[1].ownership().dof_keys, dtype=np.int64)
        materials = {s.site_id.cell_id: s for s in state.materials.states}
        rows = []
        for index, key in enumerate(keys):
            cell = int(key[1])
            material = materials[cell]
            signature = digest(
                *(bank[name][index] for name in names),
                material.site_id.key,
                material.model_id,
                material.layout_id,
                material.committed,
                material.trial,
                material.state_version,
                state.time,
                state.step,
                state.state_version,
                dict(result.manifest.integrator_state)[clock_name],
            )
            rows.append([cell, *signature])
        return np.asarray(rows, dtype=np.int64).reshape(-1, 33)

    def advance_bank(
        result: Any, view: Any, identities: Any, *, sensitivity: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        state = result.state
        bank = dict(result.manifest.auxiliary_state)
        dt = float(np.asarray(dict(result.manifest.integrator_state)[clock_name]))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid restored cone timestep.")
        keys = np.asarray(identities[1].ownership().dof_keys, dtype=np.int64)
        if (
            keys.ndim != 2
            or keys.shape[1] != 3
            or np.any(keys[:, 0] != 2)
            or np.any(keys[:, 2] != 0)
        ):
            raise ValueError("Cone fibers require P0 cell ownership keys.")
        triangles = _periodic_cone_triangles(view)
        materials = {s.site_id.cell_id: s for s in state.materials.states}
        if set(map(int, keys[:, 1])) != set(materials) or set(materials) != set(
            map(int, view.owned_cell_ids)
        ):
            raise ValueError("Cone kinematics, materials, and owned cells disagree.")
        rows = []
        flags = np.zeros(4, dtype=np.int64)
        ordered = [materials[int(key[1])] for key in keys]
        vertices = np.asarray(
            [
                [(float(x), float(y), 0.0) for x, y in triangles[int(key[1])][1]]
                + [(0.0, 0.0, 1.0)]
                for key in keys
            ],
            dtype=np.float64,
        ).reshape(-1, 4, 3)
        u, v, a = (np.asarray(bank[n]).reshape(-1, 4, 3) for n in names)
        for material in ordered:
            rolled = material.rollback()
            if (
                rolled.state_version != material.state_version
                or not np.array_equal(
                    np.asarray(rolled.committed), np.asarray(material.committed)
                )
                or not np.array_equal(
                    np.asarray(rolled.trial), np.asarray(material.committed)
                )
            ):
                raise RuntimeError(
                    "Material rollback changed committed history or version."
                )
        if not ordered:
            return np.empty((0, 33), dtype=np.int64), flags
        out = workset.advance(
            vertices,
            u,
            v,
            a,
            ordered,
            float(state.time),
            int(state.step),
            int(state.state_version),
            dt,
        )
        if (
            out.step != state.step + 1
            or out.state_version != state.state_version + 1
            or not np.isclose(float(out.time), float(state.time) + dt)
        ):
            raise RuntimeError("Cone acceptance did not advance clock versions.")
        for index, key in enumerate(keys):
            cell = int(key[1])
            material = ordered[index]
            m = out.materials.state(material.site_id)
            if m.state_version != material.state_version + 1 or not np.array_equal(
                np.asarray(m.committed), np.asarray(m.trial)
            ):
                raise RuntimeError("Cone acceptance did not commit material versions.")
            np.testing.assert_allclose(
                out.displacement[index],
                u[index]
                + dt * v[index]
                + dt * dt * 0.25 * (a[index] + np.asarray(out.acceleration[index])),
                rtol=1e-8,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                out.velocity[index],
                v[index] + 0.5 * dt * (a[index] + np.asarray(out.acceleration[index])),
                rtol=1e-8,
                atol=1e-10,
            )
            # Refinement interpolates the parent's acceleration to child corners.
            # That inherited field need not have zero resultant on each child;
            # Newmark's trapezoidal impulse includes it on the first child step.
            points = vertices[index]
            mass_per_vertex = abs(np.linalg.det((points[1:] - points[0]).T)) / 24.0
            np.testing.assert_allclose(
                mass_per_vertex * np.sum(out.acceleration[index], axis=0),
                0.0,
                rtol=1e-8,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                mass_per_vertex * np.sum(out.velocity[index], axis=0),
                mass_per_vertex * np.sum(v[index] + 0.5 * dt * a[index], axis=0),
                rtol=1e-8,
                atol=1e-10,
            )
            if sensitivity and not momentum_evidence:
                momentum_evidence.append(
                    {
                        "cell": cell,
                        "previous_acceleration_sum": np.sum(a[index], axis=0).tolist(),
                        "velocity_sum_change": (
                            np.sum(out.velocity[index], axis=0) - np.sum(v[index], axis=0)
                        ).tolist(),
                        "previous_acceleration_impulse": (
                            0.5 * dt * np.sum(a[index], axis=0)
                        ).tolist(),
                        "mass_per_vertex": float(mass_per_vertex),
                        "new_acceleration_resultant": (
                            mass_per_vertex * np.sum(out.acceleration[index], axis=0)
                        ).tolist(),
                    }
                )
            signature = digest(
                u[index],
                v[index],
                a[index],
                material.committed,
                material.trial,
                material.state_version,
                out.displacement[index],
                out.velocity[index],
                out.acceleration[index],
                m.committed,
                m.trial,
                m.state_version,
            )
            rows.append([cell, *signature])
        if sensitivity:
            for index, material in enumerate(ordered):
                if np.all(flags):
                    break
                fresh = FiniteStrainJ2MPMConstitutivePlan().initialize_state(
                    (), np.asarray(material.committed).dtype
                )
                reset = phx.equations.MaterialState(
                    material.site_id,
                    material.model_id,
                    fresh,
                    state_version=material.state_version,
                )
                perturb = vertices[index] * 0.1
                for j in range(4):
                    if flags[j]:
                        continue
                    uu, vv, aa = u.copy(), v.copy(), a.copy()
                    mm = list(ordered)
                    dd = dt
                    if j == 0:
                        mm[index] = reset
                    elif j == 1:
                        vv[index] += perturb
                    elif j == 2:
                        aa[index] += perturb
                    else:
                        dd *= 1.2
                    other = workset.advance(
                        vertices,
                        uu,
                        vv,
                        aa,
                        mm,
                        float(state.time),
                        int(state.step),
                        int(state.state_version),
                        dd,
                    )
                    flags[j] = int(
                        changed(out.displacement[index], other.displacement[index])
                        or changed(
                            out.materials.state(material.site_id).committed,
                            other.materials.state(material.site_id).committed,
                        )
                    )
        return np.asarray(rows, dtype=np.int64).reshape(-1, 33), flags

    error = None
    reference_rows = target_rows = None
    flags = np.zeros(4, dtype=np.int64)
    try:
        reference_rows = input_rows(adapted, adapted.target_identity_plans)
        target_rows = input_rows(moved, target_identities)
        continued_rows, flags = advance_bank(
            moved, target_view, target_identities, sensitivity=True
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _owner_group_fail_if_any(error, group, "cold_cone_continue", "cold cone continuation")
    reference = gather(reference_rows, 33)
    actual = gather(target_rows, 33)
    reference = reference[np.argsort(reference[:, 0])]
    actual = actual[np.argsort(actual[:, 0])]
    if not np.array_equal(reference, actual):
        raise RuntimeError(
            "Migrated cone input differs from the exact adapted reference."
        )
    all_flags = np.asarray(
        _owner_group_allgather(flags, group, "cold_cone_sensitivity")
    ).reshape(-1, 4)
    if not np.all(np.any(all_flags, axis=0)):
        raise RuntimeError(
            "Cone continuation did not respond to history, velocity, acceleration and timestep."
        )
    return {
        "cells": int(len(actual)),
        "changed_placement": True,
        "momentum_evidence": momentum_evidence,
        "transfer_momentum_error": float(np.max(np.abs(adapted_total - source_total))),
        "independent_reference": True,
        "newmark": True,
        "linear_momentum": True,
        "material_commit_trial_version": True,
        "rollback": True,
        "history_velocity_acceleration_timestep_sensitive": True,
        "owned_cell_digests": {
            str(int(row[0])): bytes(row[1:].astype(np.uint8)).hex()
            for row in sorted(continued_rows, key=lambda row: int(row[0]))
        },
        "source_manifest_id": manifest.manifest_id,
        "adapted_manifest_id": adapted.manifest.manifest_id,
        "target_manifest_id": moved.manifest.manifest_id,
    }
