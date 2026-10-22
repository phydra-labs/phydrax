from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization._cell_geometry import cell_geometry_preparation_query_upper
from phydrax.discretization.vem import (
    prepare_polyhedral_h1_virtual_element_3d,
    PreparedPolyhedralH1VirtualElement3D,
    VirtualElementResourceBudget,
)
from phydrax.discretization.vem._operator import FactorizedVirtualElementOperator
from phydrax.lifecycle import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing._polyhedral_generation import PolyhedralConstruction
from phydrax.sparse import RelationAccumulation
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


def _with_accumulation(
    prepared: PreparedPolyhedralH1VirtualElement3D,
    accumulation: RelationAccumulation,
) -> PreparedPolyhedralH1VirtualElement3D:
    operator = prepared.operator
    replacement = FactorizedVirtualElementOperator(
        operator.coefficient_maps,
        operator.polynomial_matrices,
        operator.stabilization_matrices,
        operator.gathers,
        operator.global_size,
        accumulation=accumulation,
        properties=operator.properties,
    )
    return eqx.tree_at(lambda value: value.operator, prepared, replacement)


@pytest.mark.parametrize("accumulation", ["fast", "deterministic"])
@pytest.mark.parametrize("scale", [1.0, float.fromhex("0x1p-1020")])
def test_periodic_zero_mean_load_all_stage_bound(
    periodic_case: _PeriodicCase,
    accumulation: RelationAccumulation,
    scale: float,
) -> None:
    prepared = _with_accumulation(periodic_case.prepared, accumulation)
    count = prepared.mesh.connectivity.cell_count
    density = np.linspace(0.1, 0.2, count, dtype=np.float64) * scale
    weights = np.asarray(prepared.evidence.cell_volumes)
    # Exercise the actual backend's weighted dot, reduction and quotient.
    mean = float(jnp.dot(jnp.asarray(density), jnp.asarray(weights)) / jnp.sum(weights))
    force = density - np.float64(mean)
    bound = prepared.zero_mean_load_compatibility_bound(density, weights, mean, force)
    # The bound is obtained before assembling or observing the residual.
    load = prepared.assemble_cellwise_constant_load(force)
    residual = abs(math.fsum(map(float, np.asarray(load))))
    assert Fraction(residual) <= Fraction(bound.tolerance)
    if scale == 1.0:
        assert residual > 0  # This genuinely needs the approved nonzero bound.
    assert math.isfinite(bound.tolerance)
    assert bound.tolerance > 0
    assert bound.cell_count == count
    assert bound.maximum_incidence > 0
    assert bound.mean_reduction_bound > 0
    assert bound.assembly_bound > 0
    assert bound.accumulation_bound > 0
    prepared.compatible_pinned_load(load, 0, compatibility_tolerance=bound.tolerance)
    repeated = prepared.zero_mean_load_compatibility_bound(density, weights, mean, force)
    assert repeated.inputs_id == bound.inputs_id
    assert repeated.bound_id == bound.bound_id
    assert repeated.tolerance == bound.tolerance
    changed_density = density[::-1].copy()
    changed_mean = float(
        jnp.dot(jnp.asarray(changed_density), jnp.asarray(weights)) / jnp.sum(weights)
    )
    changed = prepared.zero_mean_load_compatibility_bound(
        changed_density, weights, changed_mean, changed_density - changed_mean
    )
    assert changed.inputs_id != bound.inputs_id
    assert changed.bound_id != bound.bound_id


def test_periodic_zero_mean_load_rejects_physical_incompatibility(
    periodic_case: _PeriodicCase,
) -> None:
    prepared = periodic_case.prepared
    count = prepared.mesh.connectivity.cell_count
    density = np.ones(count, dtype=np.float64)
    weights = np.asarray(prepared.evidence.cell_volumes)
    # A perfectly finite, bitwise-correct subtraction is not enough: this
    # omits the physical mean and must not be absorbed by the rounding bound.
    with pytest.raises(ValueError, match="physically incompatible mean"):
        prepared.zero_mean_load_compatibility_bound(density, weights, 0.0, density)
    with pytest.raises(ValueError, match="another force"):
        prepared.zero_mean_load_compatibility_bound(density, weights, 1.0, density)
    # Preserve the old exact-zero negative admissions, independently of the
    # approved bound for a correctly constructed zero-mean physical force.
    unit_load = prepared.assemble_cellwise_constant_load(density)
    with pytest.raises(ValueError, match="zero-total"):
        prepared.compatible_pinned_load(unit_load, 0, compatibility_tolerance=0.0)
    with pytest.raises(ValueError, match="zero-total"):
        prepared.compatible_pinned_load(
            np.ones(prepared.dof_count, dtype=np.float64),
            0,
            compatibility_tolerance=0.0,
        )


@pytest.mark.parametrize("weight_fault", ["changed", "zero", "negative"])
def test_periodic_zero_mean_load_rejects_weights(
    periodic_case: _PeriodicCase, weight_fault: str
) -> None:
    prepared = periodic_case.prepared
    weights = np.asarray(prepared.evidence.cell_volumes).copy()
    match weight_fault:
        case "changed":
            weights[0] = np.nextafter(weights[0], np.inf)
        case "zero":
            weights[0] = 0.0
            prepared = eqx.tree_at(
                lambda value: value.evidence.cell_volumes,
                prepared,
                jnp.asarray(weights),
            )
        case "negative":
            weights[0] = -weights[0]
            prepared = eqx.tree_at(
                lambda value: value.evidence.cell_volumes,
                prepared,
                jnp.asarray(weights),
            )
        case _:
            raise AssertionError(f"unexpected test weight fault {weight_fault}")
    zero = np.zeros(weights.shape, dtype=np.float64)
    with pytest.raises(ValueError, match="admitted positive cell volumes"):
        prepared.zero_mean_load_compatibility_bound(zero, weights, 0.0, zero)


def test_periodic_zero_mean_load_rejects_unproved_accumulation(
    periodic_case: _PeriodicCase,
) -> None:
    prepared = _with_accumulation(periodic_case.prepared, "compensated")
    weights = np.asarray(prepared.evidence.cell_volumes)
    zero = np.zeros(weights.shape, dtype=np.float64)
    with pytest.raises(ValueError, match="plain fast or deterministic"):
        prepared.zero_mean_load_compatibility_bound(zero, weights, 0.0, zero)


def test_periodic_zero_mean_load_rejects_unproved_denominator_range(
    periodic_case: _PeriodicCase,
) -> None:
    prepared = periodic_case.prepared
    weights = np.full(
        (prepared.mesh.connectivity.cell_count,),
        np.finfo(np.float64).tiny,
        dtype=np.float64,
    )
    prepared = eqx.tree_at(
        lambda value: value.evidence.cell_volumes, prepared, jnp.asarray(weights)
    )
    zero = np.zeros(weights.shape, dtype=np.float64)
    with pytest.raises(ValueError, match="normal positive rounded total"):
        prepared.zero_mean_load_compatibility_bound(zero, weights, 0.0, zero)


@pytest.mark.parametrize("fault", ["float32", "shape"])
def test_periodic_zero_mean_load_rejects_nonbinary64_inputs(
    periodic_case: _PeriodicCase, fault: str
) -> None:
    prepared = periodic_case.prepared
    weights = np.asarray(prepared.evidence.cell_volumes)
    zero = np.zeros(weights.shape, dtype=np.float64)
    match fault:
        case "float32":
            density = zero.astype(np.float32)
        case "shape":
            density = zero[:, None]
        case _:
            raise AssertionError(f"unexpected test input fault {fault}")
    with pytest.raises(ValueError, match="finite float64"):
        prepared.zero_mean_load_compatibility_bound(density, weights, 0.0, zero)


@pytest.mark.parametrize("field", ["density", "weights", "mean", "force"])
@pytest.mark.parametrize("nonfinite", [math.inf, math.nan])
def test_periodic_zero_mean_load_rejects_nonfinite(
    periodic_case: _PeriodicCase, field: str, nonfinite: float
) -> None:
    prepared = periodic_case.prepared
    weights = np.asarray(prepared.evidence.cell_volumes).copy()
    density = np.zeros(weights.shape, dtype=np.float64)
    force = density.copy()
    mean = 0.0
    match field:
        case "density":
            density[0] = nonfinite
        case "weights":
            weights[0] = nonfinite
        case "mean":
            mean = nonfinite
        case "force":
            force[0] = nonfinite
        case _:
            raise AssertionError(f"unexpected test nonfinite field {field}")
    with pytest.raises(ValueError, match="finite float64"):
        prepared.zero_mean_load_compatibility_bound(density, weights, mean, force)


@pytest.mark.parametrize("stage", ["subtraction", "weight reduction", "final fsum"])
def test_periodic_zero_mean_load_rejects_overflow_ranges(
    periodic_case: _PeriodicCase, stage: str
) -> None:
    prepared = periodic_case.prepared
    weights = np.asarray(prepared.evidence.cell_volumes).copy()
    maximum = np.finfo(np.float64).max
    mean = 0.0
    match stage:
        case "subtraction":
            density = np.full(weights.shape, -maximum, dtype=np.float64)
            force = density.copy()
            mean = float(maximum)
        case "weight reduction":
            weights.fill(maximum)
            prepared = eqx.tree_at(
                lambda value: value.evidence.cell_volumes,
                prepared,
                jnp.asarray(weights),
            )
            density = np.zeros(weights.shape, dtype=np.float64)
            force = density.copy()
        case "final fsum":
            density = np.full(weights.shape, maximum / 2, dtype=np.float64)
            density[::2] *= -1
            mean = float(
                jnp.dot(jnp.asarray(density), jnp.asarray(weights)) / jnp.sum(weights)
            )
            force = density - mean
        case _:
            raise AssertionError(f"unexpected test overflow stage {stage}")
    with pytest.raises(ValueError, match="overflow range"):
        prepared.zero_mean_load_compatibility_bound(density, weights, mean, force)


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
