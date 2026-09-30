#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.


import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization import VariablePatchEntityComplex


def _complexes() -> tuple[VariablePatchEntityComplex, ...]:
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    signature = phx.discretization.PatchShapeSignature((4,), halo_width=1)
    hierarchy = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0, (phx.discretization.PatchBucketPlan(signature, 1),)
            ),
            phx.discretization.VariablePatchLevelPlan(
                1, (phx.discretization.PatchBucketPlan(signature, 2),)
            ),
        ),
        (phx.discretization.LogicalPatchBox(0, (0,), (4,)),),
    )
    compiler = phx.discretization.VariablePatchTopologyCompiler(hierarchy)
    initial = compiler.initial_topology()
    tags = ((jnp.asarray([[True, True, False, False]]),),)
    refined = compiler.compile(initial, tags).topology
    complexes = phx.discretization.VariablePatchEntityComplexPlan(
        phx.discretization.BlockHierarchyCapacityPlan(
            ((8, 8), (16, 16)),
            (8, 16),
        )
    ).prepare(refined)
    return complexes


def test_compatible_entity_transfer_preserves_affine_potentials() -> None:
    coarse, fine = _complexes()
    family = phx.discretization.CompatibleEntityTransferFamily(coarse, fine, 2, (16, 16))
    values = np.zeros((coarse.capacity[0],), dtype=np.float64)
    for index, key in enumerate(coarse.entity_keys[0]):
        if key is not None:
            values[index] = 2.0 + key[1][0] / 4.0
    image = family.transfer(0).prolongation.mv(jnp.asarray(values))
    expected = np.zeros((fine.capacity[0],), dtype=np.float64)
    for index, key in enumerate(fine.entity_keys[0]):
        if key is not None:
            expected[index] = 2.0 + key[1][0] / 8.0
    np.testing.assert_allclose(image, expected, atol=1.0e-13)
    assert bool(family.evidence.valid)
    np.testing.assert_allclose(family.evidence.commuting_defects, 0.0, atol=1.0e-13)
    gradient = coarse.complex.incidences[0].exterior_derivative().mv(jnp.asarray(values))
    fine_gradient = fine.complex.incidences[0].exterior_derivative().mv(image)
    np.testing.assert_allclose(
        family.transfer(1).prolongation.mv(gradient), fine_gradient, atol=1.0e-13
    )


def test_compatible_entity_transfer_transpose_duality() -> None:
    coarse, fine = _complexes()
    transfer = phx.discretization.CompatibleEntityTransferFamily(
        coarse, fine, 2, (16, 16)
    ).transfer(0)
    source = jnp.arange(coarse.capacity[0], dtype=jnp.float64)
    target = jnp.linspace(0.0, 1.0, fine.capacity[0])
    np.testing.assert_allclose(
        jnp.vdot(transfer.prolongation.mv(source), target),
        jnp.vdot(source, transfer.dual_pullback.mv(target)),
        atol=1.0e-13,
    )


def test_complex_map_evidence_detects_noncommuting_entity_transfer() -> None:
    coarse, fine = _complexes()
    family = phx.discretization.CompatibleEntityTransferFamily(coarse, fine, 2, (16, 16))
    mapping = family.complex_map
    broken = phx.linalg.ComplexMap(
        mapping.source,
        mapping.target,
        (2.0 * mapping.maps[0], mapping.maps[1]),
        map_id="broken-affine-entity-transfer",
    )
    evidence = phx.linalg.complex_map_evidence(
        broken,
        key=jax.random.key(17),
        probes=3,
        tolerance=1.0e-12,
    )
    assert not bool(evidence.valid)
