#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from phydrax.applications.cardiovascular import (
    anatomy,
    circulation,
    electrophysiology,
    observations,
)
from phydrax.discretization import (
    CellMesh,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)


def _unit_cube_anatomy():
    coordinates = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ]
    )
    tetrahedra = np.asarray(
        [
            (0, 1, 3, 7),
            (0, 3, 2, 7),
            (0, 2, 6, 7),
            (0, 6, 4, 7),
            (0, 4, 5, 7),
            (0, 5, 1, 7),
        ],
        dtype=np.int32,
    )
    mesh = CellMesh.from_tetrahedra(coordinates, tetrahedra)
    face_points = coordinates[np.asarray(mesh.connectivity.faces)]
    role_faces = {
        "endocardium": np.flatnonzero(np.all(face_points[..., 0] == 0.0, axis=1)),
        "epicardium": np.flatnonzero(np.all(face_points[..., 0] == 1.0, axis=1)),
        "apex": np.flatnonzero(np.all(face_points[..., 2] == 0.0, axis=1)),
        "base": np.flatnonzero(np.all(face_points[..., 2] == 1.0, axis=1)),
        "posterior": np.flatnonzero(np.all(face_points[..., 1] == 0.0, axis=1)),
        "anterior": np.flatnonzero(np.all(face_points[..., 1] == 1.0, axis=1)),
    }
    orthogonal_pairs = tuple(
        (first, second)
        for first in ("endocardium", "epicardium")
        for second in ("apex", "base", "posterior", "anterior")
    ) + tuple(
        (first, second)
        for first in ("apex", "base")
        for second in ("posterior", "anterior")
    )
    profile = anatomy.CardiacBoundaryProfile(
        "public-api-unit-cube",
        required_roles=tuple(role_faces),
        connected_roles=tuple(role_faces),
        disjoint_closure_pairs=(
            ("endocardium", "epicardium"),
            ("apex", "base"),
            ("posterior", "anterior"),
        ),
        shared_closure_pairs=orthogonal_pairs,
        exhaustive=True,
    )
    return mesh, anatomy.CardiacBoundaryRoles(mesh, role_faces, profile=profile)


def test_anatomy_microstructure_drives_phenomenological_ep():
    mesh, roles = _unit_cube_anatomy()
    fields = (
        anatomy.HarmonicCoordinatePlan(
            mesh,
            roles,
            (
                anatomy.HarmonicCoordinateSpec("transmural", "endocardium", "epicardium"),
                anatomy.HarmonicCoordinateSpec("longitudinal", "apex", "base"),
            ),
        )
        .prepare(numeric_version="integration")
        .solve()
        .commit()
    )
    microstructure = (
        anatomy.VentricularMicrostructurePlan("transmural", "longitudinal")
        .prepare(fields)
        .build()
        .commit()
    )
    finite_element = FiniteElementPlan(
        mesh, FiniteElementFieldSpec("activation", lagrange_element("tetrahedron", 1))
    ).prepare()
    diffusivity = electrophysiology.CellwiseDiffusivity.from_fibers(
        microstructure.fiber, 0.2, 0.05
    )
    runtime = electrophysiology.PhenomenologicalMonodomainPlan(
        finite_element,
        diffusivity,
        electrophysiology.AlievPanfilovParameters(0.05, 0.15, 8.0, 0.002, 0.2, 0.3, 12.9),
        pulses=(electrophysiology.CellStimulusPulse((0,), 0.0, 0.02, 1.0),),
    ).prepare(0.01)
    candidate = runtime.evaluate(runtime.initialize(jnp.zeros(8), jnp.zeros(8)))

    assert bool(fields.evidence.all_successful)
    assert bool(microstructure.evidence.all_successful)
    assert bool(candidate.evidence.successful)
    assert runtime.plan.diffusivity.diffusivity_id == diffusivity.diffusivity_id


def test_circulation_work_and_observation_loop_share_sign_and_units():
    pressure_kpa = jnp.asarray([1.0, 3.0, 3.0, 1.0, 1.0])
    volume_mm3 = jnp.asarray([3.0, 3.0, 1.0, 1.0, 3.0])
    observed = observations.PressureVolumeLoopPlan(
        observations.TimeBase.uniform("pv-cycle", 5, 1.0),
        pressure_reference_kpa=0.0,
        reference_configuration="absolute chamber pressure",
        loop_id="public-api-pv-loop",
    ).evaluate(pressure_kpa, volume_mm3)
    circulation_work = circulation.pressure_volume_work(pressure_kpa, volume_mm3)

    np.testing.assert_allclose(observed.external_work_mg_mm2_per_ms2, circulation_work)
    np.testing.assert_allclose(observed.external_work_mj, circulation_work * 1.0e-3)
    assert bool(observed.evidence.closed & observed.evidence.counterclockwise)
