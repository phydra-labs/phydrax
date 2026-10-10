from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.discretization.vem._polyhedral import (
    prepare_polyhedral_h1_virtual_element_3d,
    PreparedPolyhedralH1VirtualElement3D,
)
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing import PeriodicConstraint, PiecewiseLinearComplex
from phydrax.meshing._polyhedral_generation import (
    generate_polyhedral_volume,
    NativePolyhedralSchedule,
    PolyhedralConstruction,
)


type _Recipe = tuple[
    PiecewiseLinearComplex,
    np.ndarray,
    np.ndarray,
    tuple[PeriodicConstraint, ...],
    dict[str, Any],
]
type _ArchiveRoot = tuple[
    PiecewiseLinearComplex,
    np.ndarray,
    np.ndarray,
    tuple[PeriodicConstraint, ...],
    dict[str, Any],
    phx.discretization.CellMesh,
    phx.discretization.CellGeometrySpec,
    ArrayLike,
    ArrayLike,
    int,
    tuple[ArrayLike, ArrayLike],
]


def _original_periodic_source() -> tuple[PolyhedralConstruction, _Recipe]:
    points = np.asarray(
        [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)],
        dtype=np.float64,
    )
    loops = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )
    domain = PiecewiseLinearComplex(
        points, loops, np.arange(6), [(-1, 0)] * 6, ("material",)
    )
    constraints: list[PeriodicConstraint] = []
    for axis, (left, right) in enumerate(((4, 5), (2, 3), (0, 1))):
        source = phx.meshing.MeshingScope(
            "vem-periodic-cube",
            "r1",
            phx.meshing.MeshingEntityKind.GEOMETRY,
            2,
            "source",
            np.asarray((left,), dtype=np.int64),
        )
        target = phx.meshing.MeshingScope(
            "vem-periodic-cube",
            "r1",
            phx.meshing.MeshingEntityKind.GEOMETRY,
            2,
            "target",
            np.asarray((right,), dtype=np.int64),
        )
        action = np.eye(4)
        action[axis, 3] = 1
        constraints.append(
            PeriodicConstraint(
                source,
                target,
                action,
                tolerance=0,
                source_entity_ids=np.asarray((left,), dtype=np.int64),
                orientations=np.asarray((-1,), dtype=np.int32),
            )
        )
    sites = np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    weights = np.asarray([0.0, 0.0])
    schedule = NativePolyhedralSchedule()
    construction = generate_polyhedral_volume(
        domain,
        sites=sites,
        weights=weights,
        schedule=schedule,
        periodic_constraints=constraints,
    )
    return construction, (domain, sites, weights, tuple(constraints), asdict(schedule))


def _solve_and_continue(
    construction: PolyhedralConstruction,
    state: ArrayLike | None = None,
    cell_coefficients: ArrayLike | None = None,
    pinned_dof: int = 0,
) -> tuple[PreparedPolyhedralH1VirtualElement3D, Array, np.ndarray]:
    assert pinned_dof == 0
    prepared = prepare_polyhedral_h1_virtual_element_3d(construction.mesh)
    periodic = construction.mesh.periodic_topology
    assert periodic is not None
    assert prepared.dof_count == periodic.quotient.entities(0).count
    assert prepared.dof_count < construction.mesh.connectivity.vertex_count
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id
    from phydrax.discretization._transfer import TransferGeometryBinding
    from phydrax.discretization.fem._topology_transfer import (
        vertex_interpolation_transfer,
    )

    vertex_count = construction.mesh.coordinates.shape[0]
    identity = vertex_interpolation_transfer(
        np.arange(vertex_count)[:, None],
        np.ones((vertex_count, 1)),
        np.ones((vertex_count, 1), dtype=np.bool_),
        source_size=vertex_count,
        source_topology_id=construction.mesh.topology_id,
        target_topology_id=construction.mesh.topology_id,
    )
    binding = TransferGeometryBinding(
        cell_geometry_id(construction.geometry),
        cell_geometry_id(construction.geometry),
        "identity",
        source_topology_id=construction.mesh.topology_id,
        target_topology_id=construction.mesh.topology_id,
    )
    transfer = prepared.vertex_field_transfer(
        prepared, identity, binding, field_name="temperature"
    )
    nodal_probe = jnp.linspace(-0.7, 0.8, prepared.dof_count)
    np.testing.assert_allclose(
        transfer.primal_operator.mv(nodal_probe), nodal_probe, atol=1e-12
    )
    dual = transfer.dual_pullback_operator
    if dual is None:
        raise AssertionError("Periodic VEM transfer requires its dual pullback.")
    np.testing.assert_allclose(dual.mv(nodal_probe), nodal_probe, atol=1e-12)
    layout = transfer.source.layout
    if not isinstance(layout, phx.discretization.EntityDofLayout):
        raise AssertionError("Periodic VEM transfer requires a quotient entity layout.")
    assert layout.entity_set_id == periodic.quotient.entities(0).entity_set_id
    physical_force = (
        jnp.arange(construction.mesh.connectivity.cell_count, dtype=jnp.float64) + 1
    )
    physical_load = prepared.assemble_cellwise_constant_load(physical_force)
    np.testing.assert_allclose(
        eqx.filter_jit(lambda owner, force: owner.assemble_cellwise_constant_load(force))(
            prepared, physical_force
        ),
        physical_load,
        atol=1e-12,
    )
    assert physical_load.shape == (prepared.dof_count,)
    np.testing.assert_allclose(
        jnp.sum(physical_load),
        jnp.vdot(physical_force, prepared.evidence.cell_volumes),
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="zero-total"):
        prepared.compatible_pinned_load(
            physical_load, pinned_dof, compatibility_tolerance=1e-12
        )
    assert prepared.evidence.maximum_reproduction_defect < 1e-10
    assert float(jnp.sum(prepared.evidence.cell_volumes)) == pytest.approx(1.0, abs=1e-12)
    connectivity = construction.mesh.connectivity
    if not isinstance(connectivity, phx.discretization.PolyhedralConnectivity):
        raise AssertionError("Periodic polyhedral VEM requires polyhedral connectivity.")
    offsets = np.asarray(connectivity.cell_vertex_offsets)
    vertices = np.asarray(connectivity.cell_vertex_values)
    source_points = np.asarray(
        construction.geometry.source_coordinates(), dtype=np.float64
    )
    affine_energy = np.zeros(3)
    for arity, tensors in zip(
        sorted(set(np.diff(offsets))), prepared.operator.local_tensors(), strict=True
    ):
        cells = np.flatnonzero(np.diff(offsets) == arity)
        for cell, tensor in zip(cells, np.asarray(tensors), strict=True):
            local_points = source_points[vertices[offsets[cell] : offsets[cell + 1]]]
            affine_energy += np.diag(local_points.T @ tensor @ local_points)
    # Independent original unit-cube integral of each Cartesian |grad(x_i)|².
    np.testing.assert_allclose(affine_energy, np.ones(3), atol=1e-10)
    np.testing.assert_allclose(prepared.mv(jnp.ones(prepared.dof_count)), 0, atol=1e-11)
    target = (
        jnp.linspace(0.0, 1.0, prepared.dof_count)
        if state is None
        else jnp.asarray(state) * 1.25
    )
    target = target - target[0]
    local_energy = sum(
        float(jnp.einsum("ci,cij,cj->", target[gather], tensor, target[gather]))
        for gather, tensor in zip(
            prepared.operator.gathers, prepared.operator.local_tensors(), strict=True
        )
    )
    assert float(jnp.vdot(target, prepared.mv(target))) == pytest.approx(
        local_energy, abs=1e-11
    )
    coefficients = (
        np.linspace(
            1.0,
            1.5,
            construction.mesh.connectivity.cell_count,
            dtype=np.float64,
        )
        if cell_coefficients is None
        else np.asarray(cell_coefficients)
    )
    diffusion = prepared.bind_scalar_diffusion(coefficients)
    for original, rebound in zip(
        prepared.operator.gathers, diffusion.gathers, strict=True
    ):
        assert original is rebound
    load = diffusion.mv(target)
    rhs = prepared.compatible_pinned_load(load, 0, compatibility_tolerance=1e-11)
    operator = prepared.pinned_diffusion_operator(0, cell_coefficients=coefficients)
    result = phx.linalg.solve(
        phx.linalg.LinearSystem(operator),
        rhs,
        policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.ConjugateGradient(),
            tolerance=phx.linalg.TolerancePolicy(
                relative=1e-10, absolute=1e-12, max_steps=100
            ),
        ),
    )
    assert bool(result.successful)
    full = jnp.concatenate((jnp.zeros(1), result.value))
    np.testing.assert_allclose(full, target, atol=1e-9)
    np.testing.assert_allclose(diffusion.mv(full), load, atol=1e-9)
    probe = jnp.linspace(-0.3, 0.9, prepared.dof_count)
    np.testing.assert_allclose(
        jnp.vdot(probe, diffusion.mv(full)),
        jnp.vdot(diffusion.transpose_mv(probe), full),
        atol=1e-11,
    )
    with pytest.raises(ValueError, match="zero-total"):
        prepared.compatible_pinned_load(
            jnp.ones(prepared.dof_count), 0, compatibility_tolerance=0
        )
    with pytest.raises(ValueError, match="strictly positive"):
        prepared.bind_scalar_diffusion(
            np.zeros(construction.mesh.connectivity.cell_count)
        )
    return prepared, full, coefficients


def _restore_and_continue(
    root: _ArchiveRoot,
) -> tuple[PreparedPolyhedralH1VirtualElement3D, Array, np.ndarray]:
    (
        domain,
        sites,
        weights,
        constraints,
        controls,
        mesh,
        geometry,
        state,
        coefficients,
        gauge,
        history,
    ) = root
    np.testing.assert_array_equal(history[-1], state)
    construction = generate_polyhedral_volume(
        domain,
        sites=sites,
        weights=weights,
        schedule=NativePolyhedralSchedule(**controls),
        periodic_constraints=constraints,
    )
    assert construction.mesh.mesh_id == mesh.mesh_id
    assert construction.geometry.source_coordinates() == geometry.source_coordinates()
    np.testing.assert_array_equal(
        np.asarray(construction.mesh.coordinates).view(np.uint64),
        np.asarray(mesh.coordinates).view(np.uint64),
    )
    return _solve_and_continue(construction, history[-1], coefficients * 1.25, gauge)


def test_original_native_periodic_source_vem_solve_archive_continuation(
    tmp_path: Path,
) -> None:
    construction, recipe = _original_periodic_source()
    prepared, state, coefficients = _solve_and_continue(construction)
    from phydrax.discretization.vem import VirtualElementResourceBudget

    with pytest.raises(ValueError, match="capacity"):
        prepare_polyhedral_h1_virtual_element_3d(
            construction.mesh,
            resource_budget=VirtualElementResourceBudget(maximum_cells=1),
        )
    root = recipe + (
        construction.mesh,
        construction.geometry,
        state,
        coefficients,
        0,
        (jnp.zeros_like(state), state),
    )
    receipt = write_meshing_source_closure(tmp_path / "periodic-vem", root)
    restored = read_meshing_source_closure(
        tmp_path / "periodic-vem", expected_content_id=receipt.content_id
    )
    continued, _, _ = _restore_and_continue(restored)
    assert continued.prepared_id == prepared.prepared_id
    cell_geometry = continued.cell_geometry
    if cell_geometry is None:
        raise AssertionError("Prepared periodic VEM must retain its cell geometry.")
    assert (
        cell_geometry.source_coordinates() == construction.geometry.source_coordinates()
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; from phydrax.lifecycle._meshing_sources import read_meshing_source_closure; "
            "helpers=runpy.run_path(sys.argv[1]); restored=read_meshing_source_closure(sys.argv[2],expected_content_id=sys.argv[3]); "
            "helpers['_restore_and_continue'](restored)",
            __file__,
            str(tmp_path / "periodic-vem"),
            receipt.content_id,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
