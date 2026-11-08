#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Padded particle and storage capacity never changes MPM conservation bits."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


Geometry = Literal["cartesian", "axisymmetric", "multifield", "nodal"]

_INACTIVE_MASS = 5.0


def _discriminator(count: int, /) -> np.ndarray:
    """Masses whose compensated sum reveals the padded reduction layout.

    Sequential Sum2 drops every 2**-108 term against the 2**-53 correction, so
    up to 64 terms sum to 1 but to 1 + 2**-52 once padding makes them blocked.
    Above 64 terms the six 2**-108 terms share the first 32-term block with 1
    and 2**-53 and vanish; interleaved padding moves them into the next block,
    whose lumped sum survives.
    """
    if count <= 64:
        return np.asarray([1.0, 2.0**-53] + [2.0**-108] * (count - 2), dtype=np.float64)
    stream = np.full((count,), 2.0**-200, dtype=np.float64)
    stream[:2] = (1.0, 2.0**-53)
    stream[26:32] = 2.0**-108
    return stream


def _lattice(count: int, geometry: Geometry, /) -> np.ndarray:
    index = np.arange(count)
    if geometry == "nodal":
        # Row-major on the 1/8-spaced nodes: every particle mass lands, exactly
        # and alone, on one storage slot in particle order.
        return np.stack((0.125 * (1 + index // 7), 0.125 * (1 + index % 7)), axis=-1)
    if geometry == "axisymmetric":
        # Off the radial nodes, where the linear APIC moment would be singular.
        return np.stack((0.313 + 0.1 * (index % 9), 0.2 + 0.07 * (index // 9)), axis=-1)
    return np.stack((0.3 + 0.05 * (index % 9), 0.3 + 0.05 * (index // 9)), axis=-1)


def _slots(count: int, capacity: int, /) -> np.ndarray:
    """Spread the logical particles over the capacity, interleaving padding."""
    return np.round(np.linspace(0.0, capacity - 1.0, count)).astype(np.int64)


def _acceleration(time: Array, position: Array, velocity: Array, args: object) -> Array:
    del time, velocity, args
    return jnp.broadcast_to(jnp.asarray((0.0, -0.2), dtype=jnp.float64), position.shape)


def _step(
    geometry: Geometry,
    count: int,
    capacity: int,
    *,
    storage_blocks: int | None = None,
) -> phx.discretization.MPMStepResult:
    slots = _slots(count, capacity)
    active = np.zeros((capacity,), dtype=np.bool_)
    active[slots] = True
    masses = np.full((capacity,), _INACTIVE_MASS, dtype=np.float64)
    masses[slots] = _discriminator(count)
    position = np.tile(np.asarray(((0.5, 0.5),), dtype=np.float64), (capacity, 1))
    position[slots] = _lattice(count, geometry)
    velocity = np.tile(np.asarray(((3.0, -2.0),), dtype=np.float64), (capacity, 1))
    velocity[slots] = (1.0, 0.5)
    volume = np.full((capacity,), 0.02, dtype=np.float64)
    # A uniform density keeps the acoustic step limit independent of the
    # discriminator masses, and carries the stream into the volume sums.
    volume[slots] = masses[slots] / 100.0
    axisymmetric = geometry == "axisymmetric"
    if axisymmetric:
        bounds = jnp.asarray(((0.0, -0.25), (2.0, 1.25)), dtype=jnp.float64)
        axes = (
            phx.discretization.UniformAxisSpec(17, periodic=False, endpoint=True),
            phx.discretization.UniformAxisSpec(13, periodic=False, endpoint=True),
        )
        assignment = phx.discretization.TensorBSplineSplatAssignment(1)
        margin = 0.125 * assignment.capabilities.maximum_support_radius_cells
        domain = phx.discretization.MPMParticleDomainPlan(
            jnp.asarray(((0.0, 0.0), (2.0 - margin, 1.0)), dtype=jnp.float64),
            periodic=(False, False),
            support_margin=(margin, margin),
        )
        material = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3)
        method = phx.discretization.ExplicitMPMMethodPlan(
            geometry=phx.discretization.MPMGeometryPlan(
                "axisymmetric", spatial_dimension=2
            )
        )
    elif geometry == "nodal":
        bounds = jnp.asarray(((-0.25, -0.25), (1.25, 1.25)), dtype=jnp.float64)
        axes = tuple(
            phx.discretization.UniformAxisSpec(13, periodic=False, endpoint=True)
            for _ in range(2)
        )
        assignment = phx.discretization.TensorBSplineSplatAssignment(1)
        domain = phx.discretization.MPMParticleDomainPlan(
            jnp.asarray(((-0.125, -0.125), (1.125, 1.125)), dtype=jnp.float64),
            periodic=(False, False),
            support_margin=0.125,
        )
        material = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2)
        # A node-centred linear stencil has a singular APIC moment.
        method = phx.discretization.ExplicitMPMMethodPlan(
            phx.discretization.PICTransferPlan()
        )
    else:
        bounds = jnp.asarray(((-0.25, -0.25), (1.25, 1.25)), dtype=jnp.float64)
        axes = tuple(
            phx.discretization.UniformAxisSpec(12, periodic=False, endpoint=True)
            for _ in range(2)
        )
        assignment = phx.discretization.TensorBSplineSplatAssignment(2)
        domain = phx.discretization.MPMParticleDomainPlan(
            jnp.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=jnp.float64),
            periodic=(False, False),
            support_margin=0.21,
        )
        material = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2)
        method = phx.discretization.ExplicitMPMMethodPlan()
    grid_plan = phx.discretization.TensorGridPlan(axes, axis_names=("x", "y"))
    particles = phx.discretization.ParticleSetPlan(
        np.arange(capacity),
        masses,
        ambient_dimension=2,
        active_mask=active,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid_plan.prepare(bounds), assignment=assignment, boundary="reject"
    ).prepare(particles)
    storage = None
    if storage_blocks is not None:
        index = grid_plan.prepare_index_space(bounds)
        storage = phx.discretization.BlockSparseMPMNodalStoragePlan(
            phx.discretization.SparseBlockTopologyPlan(
                index, (1, 1), storage_blocks, layout=index.vertices()
            )
        )
    field_slots = np.zeros((capacity,), dtype=np.int32)
    field_slots[slots] = np.arange(count) % 2
    fields = (
        phx.discretization.MPMNodalFieldPlan(("left", "right"), jnp.asarray(field_slots))
        if geometry == "multifield"
        else None
    )
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "compensated-capacity",
            material,
            external_acceleration=_acceleration,
            external_acceleration_id="constant-downward",
        ),
        particles,
        splat,
        method,
        domain,
        nodal_storage=storage,
        nodal_fields=fields,
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    state = compiled.initialize_state(
        jnp.asarray(position),
        jnp.asarray(velocity),
        jnp.asarray(volume),
        arguments,
        **(
            {
                "velocity_field_slots": jnp.asarray(field_slots),
                "body_ids": jnp.asarray(field_slots),
            }
            if geometry == "multifield"
            else {}
        ),
    )
    step_size = jnp.asarray(1.0e-3, dtype=jnp.float64)
    return jax.jit(
        lambda value: compiled.dynamics.step_detailed(value, step_size, arguments)
    )(state)


def _conservation_bits(result: phx.discretization.MPMStepResult) -> dict[str, bytes]:
    transfer = result.diagnostics.transfer
    energy = result.diagnostics.energy
    schedule = result.diagnostics.schedule
    values = {
        "particle_mass": transfer.particle_mass,
        "grid_mass": transfer.grid_mass,
        "relative_mass_defect": transfer.relative_mass_defect,
        "particle_momentum": transfer.particle_momentum,
        "grid_momentum": transfer.grid_momentum,
        "relative_momentum_defect": transfer.relative_momentum_defect,
        "particle_angular_momentum": transfer.particle_angular_momentum,
        "grid_angular_momentum": transfer.grid_angular_momentum,
        "relative_angular_momentum_defect": transfer.relative_angular_momentum_defect,
        "net_internal_force": transfer.net_internal_force,
        "transfer_successful": transfer.successful,
        "second_transfer_momentum_defect": schedule.second_transfer_momentum_defect,
    }
    for path, value in jax.tree_util.tree_flatten_with_path(energy)[0]:
        values[f"energy{jax.tree_util.keystr(path)}"] = value
    return {name: np.asarray(value).tobytes() for name, value in values.items()}


@pytest.mark.parametrize("geometry", ("cartesian", "axisymmetric", "multifield"))
@pytest.mark.parametrize(
    ("count", "padded"),
    ((8, 12), (60, 70), (60, 128), (70, 128)),
    ids=("8-in-12", "60-in-70", "60-in-128", "70-in-128"),
)
def test_padded_particle_capacity_keeps_conservation_bits(
    geometry: Geometry, count: int, padded: int
) -> None:
    reference = _step(geometry, count, count)
    unpadded = _conservation_bits(reference)
    padding = _conservation_bits(_step(geometry, count, padded))

    assert bool(reference.successful)
    assert bool(reference.diagnostics.transfer.successful)
    assert padding == unpadded


def test_block_sparse_storage_capacity_keeps_conservation_bits() -> None:
    # 64 storage slots reduce sequentially; 100 slots would block the padding.
    narrow = _step("nodal", 49, 49, storage_blocks=64)
    wide = _step("nodal", 49, 49, storage_blocks=100)

    assert bool(narrow.successful)
    assert bool(narrow.diagnostics.transfer.successful)
    assert _conservation_bits(wide) == _conservation_bits(narrow)
