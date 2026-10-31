from __future__ import annotations

from fractions import Fraction

import jax.numpy as jnp
import numpy as np

from phydrax.discretization.finite_volume._material_chain_action import (
    material_chain_apply,
    material_chain_transpose,
)
from phydrax.discretization.finite_volume._material_schedule import (
    MaterialChainSchedule,
    prepare_material_chain_schedule,
)
from phydrax.discretization.finite_volume._remap_csr_view import material_csr_rows
from phydrax.geometry._volume_material_deformation import (
    MaterialReferenceDeclaration,
    MaterialReferencePolicy,
    VolumeMaterialDeformation,
)
from phydrax.geometry._volume_material_map import (
    VolumeMaterialChain,
    VolumeMaterialMap,
)
from phydrax.linalg import SparseAssemblyPolicy
from tests._support.volume_material_map_fixtures import (
    make_curved_material_refinement,
    make_material_coverage_pair,
    make_material_source,
    reference_map_witness,
    ROOT_REFERENCE_TETRAHEDRON,
)


def _two_edit_schedule() -> MaterialChainSchedule:
    return MaterialChainSchedule(
        np.asarray(((0, 1, 0, 0), (2, 3, 4, 5)), dtype=np.int64),
        np.asarray(((2, 3, 4, 5), (0, 1, 0, 0)), dtype=np.int64),
        np.asarray(((True, True, False, False), (True, True, True, True))),
        np.asarray(((True, True, True, True), (True, True, False, False))),
        np.asarray(((0, 0, 1, 1), (2, 3, 4, 5)), dtype=np.int64),
        np.asarray(((2, 3, 4, 5), (0, 0, 1, 1)), dtype=np.int64),
        np.asarray(((0.5, 0.75, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))),
        np.asarray(((-1, 0, 0, 0), (1, 1, 1, 1)), dtype=np.int64),
        np.ones((2, 4), dtype=np.bool_),
        np.asarray((0, 1), dtype=np.int64),
        np.asarray((0, 1), dtype=np.int64),
        source_count=2,
        target_count=2,
        slot_count=6,
        schedule_id="two-edit-material-content-test",
    )


def test_two_edit_exponent_safe_content_action_and_transpose() -> None:
    schedule = _two_edit_schedule()
    source = jnp.asarray(((2.0**900, 2.0**899), (2.0**899, 2.0**898)), dtype=jnp.float64)
    target = material_chain_apply(schedule, source)
    np.testing.assert_array_equal(target, source)

    cotangents = jnp.asarray(((0.5, -0.25), (0.25, 0.5)), dtype=jnp.float64)
    transposed = material_chain_transpose(schedule, cotangents)
    np.testing.assert_allclose(
        np.vdot(np.asarray(target), np.asarray(cotangents)),
        np.vdot(np.asarray(source), np.asarray(transposed)),
        rtol=1e-15,
        atol=0.0,
    )
    for material in range(source.shape[1]):
        before = sum(
            (
                Fraction.from_float(float(value))
                for value in np.asarray(source)[:, material]
            ),
            Fraction(0),
        )
        after = sum(
            (
                Fraction.from_float(float(value))
                for value in np.asarray(target)[:, material]
            ),
            Fraction(0),
        )
        assert after == before


def test_two_edit_csr_view_matches_the_sparse_action() -> None:
    schedule = _two_edit_schedule()
    offsets, sources, weights = material_csr_rows(schedule, SparseAssemblyPolicy())
    np.testing.assert_array_equal(offsets, (0, 1, 2))
    np.testing.assert_array_equal(sources, (0, 1))
    np.testing.assert_array_equal(weights, (1.0, 1.0))


def _refine_coarsen_refine_chain() -> VolumeMaterialChain:
    (
        root_mesh,
        root_geometry,
        source_geometry,
        refined_mesh,
        refined_geometry,
        refined_map,
        _,
    ) = make_curved_material_refinement(exact_chart_arguments=True)
    domain, root_coverage, refined_coverage = make_material_coverage_pair(
        root_mesh,
        root_geometry,
        refined_mesh,
        refined_geometry,
        region_id="aluminum",
        occurrence_path=("assembly-1", "occurrence-9"),
    )
    reference = MaterialReferenceDeclaration(
        root_mesh,
        root_geometry,
        make_material_source(domain),
        ("solid-cell",),
        ("aluminum",),
        root_coverage,
        frame_id="material-frame-4",
        policy=MaterialReferencePolicy.ACCEPTED_CONFIGURATION,
        occurrence_paths=(("assembly-1", "occurrence-9"),),
    )
    root_map = {"root": reference_map_witness((0,), (7,), (ROOT_REFERENCE_TETRAHEDRON,))}
    child_map = {refined_mesh.blocks[0].name: refined_map}

    def material_map(refine: bool) -> VolumeMaterialMap:
        endpoints = (
            (root_mesh, source_geometry, root_map, root_coverage, 1),
            (refined_mesh, refined_geometry, child_map, refined_coverage, 2),
        )
        source, target = endpoints if refine else endpoints[::-1]
        deformation = VolumeMaterialDeformation(
            reference,
            source[0],
            source[1],
            target[0],
            target[1],
            source_reference_maps=source[2],
            target_reference_maps=target[2],
            source_coverage=source[3],
            target_coverage=target[3],
            source_frame_id=reference.frame_id,
            target_frame_id=reference.frame_id,
        )
        return VolumeMaterialMap(
            source[0],
            source[1],
            target[0],
            target[1],
            deformation,
            ("solid-cell",) * source[4],
            ("solid-cell",) * target[4],
            ("aluminum",) * source[4],
            ("aluminum",) * target[4],
            source_coverage=source[3],
            target_coverage=target[3],
        )

    refined = material_map(True)
    return VolumeMaterialChain((refined, material_map(False), refined))


def test_prepared_multi_edit_schedule_writes_every_target_row() -> None:
    chain = _refine_coarsen_refine_chain()
    source_ids = np.asarray(chain.maps[0].source_mesh.blocks[0].global_ids)
    target_ids = np.asarray(chain.maps[-1].target_mesh.blocks[0].global_ids)
    schedule = prepare_material_chain_schedule(chain, source_ids, target_ids)
    assert schedule.map_count == 3
    for index, material_map in enumerate(chain.maps):
        rows = np.asarray(schedule.target_row_slots[index])[
            np.asarray(schedule.target_row_valid[index])
        ]
        assert rows.size == material_map.target_mesh.blocks[0].cell_count
        assert np.unique(rows).size == rows.size

    source = jnp.asarray(((6.0, 2.0**900),), dtype=jnp.float64)
    target = np.asarray(material_chain_apply(schedule, source))
    np.testing.assert_array_equal(target, ((3.0, 2.0**899), (3.0, 2.0**899)))
    for material in range(source.shape[1]):
        assert sum(
            (Fraction.from_float(float(value)) for value in target[:, material]),
            Fraction(0),
        ) == Fraction.from_float(float(source[0, material]))
