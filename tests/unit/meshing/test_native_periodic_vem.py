from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization._cell_geometry import cell_geometry_preparation_query_upper
from phydrax.discretization.vem import (
    prepare_polyhedral_h1_virtual_element_3d,
    PreparedPolyhedralH1VirtualElement3D,
    VirtualElementResourceBudget,
)
from phydrax.lifecycle import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing._polyhedral_generation import PolyhedralConstruction
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.distributed_meshing.cold_lifecycle_identities import (
    array_record,
    record_array,
)
from tests._support.distributed_meshing.cold_lifecycle_periodic import (
    original_periodic_source,
    PeriodicArchiveRoot,
    restore_and_continue,
    solve_and_continue,
)
from tests._support.forced_devices import run_on_forced_devices


# Fixed empirical allowances from test_native_periodic_vem.py
# at a6827a9e5; no meshing acceptance policy is relaxed.
# Primal/dual nodal probes and assembled physical-force load integrals.
_VEM_TRANSFER_LOAD_ATOL = 1e-12
# Three affine VEM energy probes, each with analytic unit energy.
_VEM_AFFINE_ENERGY_ATOL = 1e-10
# Constant stiffness nullspace and primal/dual dot-product equality.
_VEM_NULLSPACE_DUALITY_ATOL = 1e-11
# Continued diffusion solution and operator-applied solution against the stored load.
_VEM_DIFFUSION_SOLVE_ATOL = 1e-9


# Fixed empirical residual bounds from test_native_periodic_vem.py
# at a6827a9e5; each comparison retains its strictness.
# Maximum VEM polynomial reproduction defect on the unit periodic continuation fixture.
_VEM_REPRODUCTION_DEFECT = 1e-10


# Fixed scalar-reference allowances from test_native_periodic_vem.py
# at a6827a9e5; values and relative/absolute roles are unchanged.
# Summed physical load against its analytic unit integral.
_VEM_SCALAR_LOAD_ATOL = 1e-12
# Global periodic energy versus the sum of locally assembled energy contributions.
_VEM_SCALAR_ENERGY_ATOL = 1e-11


def _assert_periodic_continuation(record: Mapping[str, Any]) -> None:
    assert record["pinned_dof"] == 0
    assert record["periodic_present"]
    assert record["dof_count"] == record["quotient_vertex_count"]
    assert record["dof_count"] < record["physical_vertex_count"]
    np.testing.assert_allclose(
        record_array(record["primal_probe"]),
        record_array(record["nodal_probe"]),
        atol=_VEM_TRANSFER_LOAD_ATOL,
    )
    np.testing.assert_allclose(
        record_array(record["dual_probe"]),
        record_array(record["nodal_probe"]),
        atol=_VEM_TRANSFER_LOAD_ATOL,
    )
    assert record["layout_entity_set_id"] == record["quotient_entity_set_id"]
    np.testing.assert_allclose(
        record_array(record["jitted_physical_load"]),
        record_array(record["physical_load"]),
        atol=_VEM_TRANSFER_LOAD_ATOL,
    )
    assert record_array(record["physical_load"]).shape == (record["dof_count"],)
    np.testing.assert_allclose(
        record["physical_load_sum"],
        record["physical_force_integral"],
        atol=_VEM_TRANSFER_LOAD_ATOL,
    )
    message = record["incompatible_physical_load"]
    assert message is not None
    assert re.search("zero-total", message)
    assert record["maximum_reproduction_defect"] < _VEM_REPRODUCTION_DEFECT
    assert record["volume"] == pytest.approx(1.0, abs=_VEM_SCALAR_LOAD_ATOL)
    # Independent unit-cube integral of each Cartesian |grad(x_i)|².
    np.testing.assert_allclose(
        record_array(record["affine_energy"]),
        np.ones(3, dtype=np.float64),
        atol=_VEM_AFFINE_ENERGY_ATOL,
    )
    np.testing.assert_allclose(
        record_array(record["constant_image"]), 0, atol=_VEM_NULLSPACE_DUALITY_ATOL
    )
    assert record["global_energy"] == pytest.approx(
        record["local_energy"], abs=_VEM_SCALAR_ENERGY_ATOL
    )
    for alias in record["gather_aliases"]:
        assert alias
    assert record["successful"]
    np.testing.assert_allclose(
        record_array(record["full"]),
        record_array(record["target"]),
        atol=_VEM_DIFFUSION_SOLVE_ATOL,
    )
    np.testing.assert_allclose(
        record_array(record["diffusion_image"]),
        record_array(record["load"]),
        atol=_VEM_DIFFUSION_SOLVE_ATOL,
    )
    np.testing.assert_allclose(
        record["primal_pairing"], record["dual_pairing"], atol=_VEM_NULLSPACE_DUALITY_ATOL
    )
    message = record["incompatible_unit_load"]
    assert message is not None
    assert re.search("zero-total", message)
    message = record["nonpositive_diffusion"]
    assert message is not None
    assert re.search("strictly positive", message)


def _assert_restored_continuation(record: Mapping[str, Any]) -> None:
    restored = record["restore"]
    np.testing.assert_array_equal(
        record_array(restored["history_last"]), record_array(restored["state"])
    )
    assert restored["regenerated_mesh_id"] == restored["archived_mesh_id"]
    assert (
        restored["regenerated_source_coordinates"]
        == restored["archived_source_coordinates"]
    )
    np.testing.assert_array_equal(
        record_array(restored["regenerated_coordinates"]).view(np.uint64),
        record_array(restored["archived_coordinates"]).view(np.uint64),
    )
    _assert_periodic_continuation(record)


@dataclass(frozen=True)
class _PeriodicCase:
    construction: PolyhedralConstruction
    prepared: PreparedPolyhedralH1VirtualElement3D
    root: PeriodicArchiveRoot
    record: dict[str, Any]


@pytest.fixture(scope="module")
def periodic_case() -> _PeriodicCase:
    construction, recipe = original_periodic_source()
    prepared, state, coefficients, record = solve_and_continue(construction)
    root = PeriodicArchiveRoot(
        recipe=recipe,
        mesh=construction.mesh,
        geometry=construction.geometry,
        state=state,
        coefficients=coefficients,
        pinned_dof=0,
        history=(jnp.zeros_like(state), state),
    )
    return _PeriodicCase(construction, prepared, root, record)


def test_original_native_periodic_source_vem_solve(periodic_case: _PeriodicCase) -> None:
    _assert_periodic_continuation(periodic_case.record)


def test_periodic_vem_cell_capacity_refusal(periodic_case: _PeriodicCase) -> None:
    with pytest.raises(ValueError, match="capacity"):
        prepare_polyhedral_h1_virtual_element_3d(
            periodic_case.construction.mesh,
            resource_budget=VirtualElementResourceBudget(maximum_cells=1),
        )


def test_periodic_vem_archive_continuation(
    tmp_path: Path, periodic_case: _PeriodicCase
) -> None:
    construction, prepared = periodic_case.construction, periodic_case.prepared
    root = periodic_case.root
    # Top-level mappings are reserved for the lifecycle's declared role namespace.
    # A single scientific root carries named continuation fields inside the archive.
    receipt = write_meshing_source_closure(tmp_path / "periodic-vem", (root,))
    restored = read_meshing_source_closure(
        tmp_path / "periodic-vem", expected_content_id=receipt.content_id
    )
    continued, _, _, restored_record = restore_and_continue(restored[0])
    _assert_restored_continuation(restored_record)
    assert continued.prepared_id == prepared.prepared_id
    cell_geometry = continued.cell_geometry
    if cell_geometry is None:
        raise AssertionError(
            "continued.cell_geometry must retain periodic VEM cell geometry; got None."
        )
    with coordinate_reference_budget(
        queries_upper=(
            cell_geometry_preparation_query_upper(cell_geometry)
            + cell_geometry_preparation_query_upper(construction.geometry)
        ),
    ) as budget:
        assert cell_geometry.source_coordinates(
            budget=budget
        ) == construction.geometry.source_coordinates(budget=budget)
    cold = run_on_forced_devices(
        "tests._support.distributed_meshing.cold_lifecycle_periodic",
        "periodic_vem",
        (str(tmp_path / "periodic-vem"), receipt.content_id),
        devices=1,
        env={},
    )
    _assert_restored_continuation(cold)


@pytest.mark.parametrize(
    ("field", "fault"),
    [
        ("primal_probe", 99.0),
        ("physical_load_sum", -1.0),
        ("volume", 2.0),
        ("successful", False),
    ],
    ids=["quotient-transfer", "physical-load", "affine-energy", "diffusion-solve"],
)
def test_periodic_continuation_oracles_reject_faulty_phase_records(
    periodic_case: _PeriodicCase, field: str, fault: Any
) -> None:
    record = dict(periodic_case.record)
    if field == "primal_probe":
        record[field] = array_record(
            np.full(record_array(record[field]).shape, fault, dtype=np.float64)
        )
    else:
        record[field] = fault
    with pytest.raises(AssertionError):
        _assert_periodic_continuation(record)
