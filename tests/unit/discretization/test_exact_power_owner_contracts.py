# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Nominal owner refusals at public power-source construction boundaries."""

from __future__ import annotations

from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from tests._support.exact_power_geometry import power_source, source_arrays


def test_power_source_refuses_nonpreparation_owner() -> None:
    source = power_source()
    with pytest.raises(TypeError, match="periodic_preparation"):
        ExactPowerCellGeometrySource(
            *source_arrays(source),
            periodic_preparation=object(),  # ty: ignore[invalid-argument-type]
        )


def test_power_source_refuses_nondomain_owner() -> None:
    source = power_source()
    with pytest.raises(TypeError, match="domain_source"):
        ExactPowerCellGeometrySource(
            *source_arrays(source),
            domain_source=object(),  # ty: ignore[invalid-argument-type]
        )


def test_power_source_refuses_nonconstraint_owner() -> None:
    source = power_source()
    with pytest.raises(TypeError, match="periodic_constraints"):
        ExactPowerCellGeometrySource(
            *source_arrays(source),
            periodic_constraints=(object(),),  # ty: ignore[invalid-argument-type]
        )


def test_power_linear_action_refuses_nonparent_owner() -> None:
    with pytest.raises(TypeError, match="parent"):
        ExactPowerCellGeometryLinearActionSource(
            object(),  # ty: ignore[invalid-argument-type]
            np.asarray([[0]], dtype=np.int64),
            (),
            np.empty((1, 0), dtype=np.int64),
        )


def test_power_restriction_refuses_nonparent_owner() -> None:
    with pytest.raises(TypeError, match="parent"):
        ExactPowerCellGeometryRestrictionSource(
            object(),  # ty: ignore[invalid-argument-type]
            np.empty((0, 4), dtype=np.float64),
            np.asarray([[0, -1]], dtype=np.int64),
            np.asarray([-1], dtype=np.int64),
        )


def test_power_source_fixture_rejects_unsorted_site_witnesses() -> None:
    source = power_source()
    sites = np.asarray(source.vertex_sites).copy()
    sites[-2:] = sites[-2:][::-1]
    with pytest.raises(ValueError, match=r"ExactPowerCellGeometrySource\.indices"):
        ExactPowerCellGeometrySource(
            source.site_points,
            source.site_weights,
            source.carrier_points,
            source.carrier_tets,
            source.vertex_site_offsets,
            sites,
            source.vertex_carriers,
        )


def test_power_linear_action_canonicalizes_fraction_sequences() -> None:
    source = power_source()
    action = ExactPowerCellGeometryLinearActionSource(
        source,
        np.asarray([[0]], dtype=np.int64),
        [[Fraction(1)]],
        np.empty((1, 0), dtype=np.int64),
    )
    assert action.vertex_coefficients == ((Fraction(1),),)
    assert action.parent_source_id == source.source_id


def test_power_linear_action_refuses_nonfraction_sequence_entries() -> None:
    source = power_source()
    with pytest.raises(
        TypeError, match=r"ExactPowerCellGeometryLinearActionSource\.vertex_coefficients"
    ):
        ExactPowerCellGeometryLinearActionSource(
            source,
            np.asarray([[0]], dtype=np.int64),
            [[1.0]],  # ty: ignore[invalid-argument-type]
            np.empty((1, 0), dtype=np.int64),
        )


@pytest.fixture(scope="module")
def linear_action() -> ExactPowerCellGeometryLinearActionSource:
    return ExactPowerCellGeometryLinearActionSource(
        power_source(),
        np.asarray([[0]], dtype=np.int64),
        ((Fraction(1),),),
        np.empty((1, 0), dtype=np.int64),
    )


def test_power_source_restoration_retains_source_identity() -> None:
    source = power_source()
    restored = ExactPowerCellGeometrySource.from_stored_fields(source)
    assert restored.source_id == source.source_id
    assert restored.periodic_source_binding == source.periodic_source_binding
    assert restored.authority_binding == source.authority_binding


def test_power_linear_action_restoration_retains_construction(
    linear_action: ExactPowerCellGeometryLinearActionSource,
) -> None:
    restored = ExactPowerCellGeometryLinearActionSource.from_stored_fields(linear_action)
    assert restored.parent_source_id == linear_action.parent_source_id
    assert restored.construction_id == linear_action.construction_id
    assert restored.source_id == linear_action.source_id


def test_power_linear_action_restoration_refuses_changed_parent_binding(
    linear_action: ExactPowerCellGeometryLinearActionSource,
) -> None:
    changed_parent = eqx.tree_at(
        lambda value: value.site_weights,
        power_source(),
        jnp.asarray([0.04, -0.02], dtype=jnp.float64),
    )
    changed = eqx.tree_at(lambda value: value.parent, linear_action, changed_parent)
    with pytest.raises(
        ValueError, match=r"ExactPowerCellGeometryLinearActionSource\.parent_source_id"
    ):
        ExactPowerCellGeometryLinearActionSource.from_stored_fields(changed)


def test_power_linear_action_restoration_refuses_changed_construction(
    linear_action: ExactPowerCellGeometryLinearActionSource,
) -> None:
    changed = eqx.tree_at(
        lambda value: value.vertex_parents,
        linear_action,
        jnp.asarray([[1]], dtype=jnp.int64),
    )
    with pytest.raises(
        ValueError, match=r"ExactPowerCellGeometryLinearActionSource\.construction_id"
    ):
        ExactPowerCellGeometryLinearActionSource.from_stored_fields(changed)


def test_power_restriction_restoration_retains_source_identity() -> None:
    source = ExactPowerCellGeometryRestrictionSource(
        power_source(),
        np.empty((0, 4), dtype=np.float64),
        np.asarray([[0, -1]], dtype=np.int64),
        np.asarray([-1], dtype=np.int64),
    )
    restored = ExactPowerCellGeometryRestrictionSource.from_stored_fields(source)
    assert restored.source_id == source.source_id


def test_power_preparation_reuses_only_request_owned_earned_intersections() -> None:
    source = power_source()
    with HostBatchTicket(100000, queries_upper=1000, bytes_upper=10000000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100000, maximum_memory_bytes=10000000
        )
        with budget.activate():
            first = source.prepare(budget=budget)
            work = budget.work_units
            retained = budget.retained_basis_bytes
            assert source.prepare(budget=budget) is first
            assert budget.work_units == work
            assert budget.retained_basis_bytes == retained
            assert source.prepare(first.rounded_vertices, budget=budget) is first
            assert budget.work_units == work + 6 * len(first.vertices)
            assert budget.retained_basis_bytes == retained
    with HostBatchTicket(100000, queries_upper=1000, bytes_upper=10000000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100000, maximum_memory_bytes=10000000
        )
        with budget.activate():
            independent = source.prepare(budget=budget)
            assert independent is not first
            assert independent == first
            assert budget.work_units == work


def test_power_preparation_cache_rechecks_every_carrier() -> None:
    source = power_source()
    with HostBatchTicket(100000, queries_upper=1000, bytes_upper=10000000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100000, maximum_memory_bytes=10000000
        )
        with budget.activate():
            first = source.prepare(budget=budget)
            carrier = first.rounded_vertices
            carrier[0, 0] = np.nextafter(carrier[0, 0], np.inf)
            with pytest.raises(ValueError, match="different coordinate bits"):
                source.prepare(carrier, budget=budget)
            assert source.prepare(first.rounded_vertices, budget=budget) is first


def test_power_preparation_cache_keys_current_source_content() -> None:
    source = power_source()
    changed = eqx.tree_at(
        lambda value: value.site_weights,
        source,
        jnp.asarray([0.04, -0.02], dtype=jnp.float64),
    )
    with HostBatchTicket(100000, queries_upper=1000, bytes_upper=10000000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100000, maximum_memory_bytes=10000000
        )
        with budget.activate():
            original = source.prepare(budget=budget)
            work = budget.work_units
            renewed = changed.prepare(budget=budget)
            assert renewed is not original
            assert renewed.source_id != original.source_id
            assert renewed.vertices != original.vertices
            assert budget.work_units > work
            assert len(budget.power_witness_cache) == 2
