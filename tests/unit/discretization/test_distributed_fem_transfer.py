import numpy as np

from phydrax.discretization._distributed_field import DistributedSparseTransfer
from phydrax.sparse import EdgeRelation


def test_two_sided_distributed_sparse_transfer_handles_empty_source_owners() -> None:
    relation = EdgeRelation(
        np.asarray((0, 2, 1), dtype=np.int32),
        np.asarray((0, 0, 1), dtype=np.int32),
        source_size=3,
        target_size=2,
    )
    transfer = DistributedSparseTransfer(
        relation,
        np.asarray((2.0, 1.0, -1.0), dtype=np.float64),
        np.asarray((0, 1, 1), dtype=np.int32),
        np.asarray((2, 0), dtype=np.int32),
        3,
    )
    source = np.asarray(((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)))
    expected = np.asarray(((7.0, 10.0), (-3.0, -4.0)))

    np.testing.assert_allclose(transfer.serial_reference(source), expected)
    np.testing.assert_allclose(
        transfer.serial_transpose_reference(np.asarray(((7.0, 11.0), (13.0, 17.0)))),
        np.asarray(((14.0, 22.0), (-13.0, -17.0), (7.0, 11.0))),
    )
    packed = transfer.pack_source_owned(source)
    assert packed.shape == (3, 2, 2)
    assert not np.any(np.asarray(transfer.source_exchange.owned_valid[2]))


def test_distributed_de_rham_transfer_matches_sparse_complex_actions() -> None:
    from phydrax.discretization._cell_mesh import CellBlock, CellMesh
    from phydrax.discretization.fem._de_rham import FiniteElementDeRhamComplex

    mesh = CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        (
            CellBlock(
                "triangle",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int32),
                global_ids=np.asarray((42,), dtype=np.int64),
            ),
        ),
    )
    source = FiniteElementDeRhamComplex(mesh, family="trimmed", order=1)
    target = FiniteElementDeRhamComplex(mesh, family="trimmed", order=2)
    source_owners = tuple(
        np.zeros(dof_map.global_dof_count, dtype=np.int32)
        for dof_map in source.discretization.dof_maps
    )
    target_owners = tuple(
        np.zeros(dof_map.global_dof_count, dtype=np.int32)
        for dof_map in target.discretization.dof_maps
    )
    transfer = source.distributed_transfer(target, source_owners, target_owners, 1)
    assert transfer.transfer_result.complex_map.map_id == transfer.complex_map.map_id
    assert transfer.transfer_result.entry_evidence
    generator = np.random.default_rng(7)
    for degree in range(source.dimension + 1):
        source_count = source.discretization.dof_maps[degree].global_dof_count
        target_count = target.discretization.dof_maps[degree].global_dof_count
        values = generator.normal(size=source_count)
        dual = generator.normal(size=target_count)
        sparse_map = transfer.complex_map.map(degree)
        np.testing.assert_allclose(
            transfer.map(degree).serial_reference(values),
            sparse_map.mv(values),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            transfer.map(degree).serial_transpose_reference(dual),
            sparse_map.transpose_mv(dual),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
