"""Owner-local reciprocal distribution against single-device canonical terms.

Owner regions run as lane-reference vmap lanes with the same collectives as
device ownership (a functional oracle, not hardware evidence). Every reference
is the SAME canonical single-device term evaluated with its own prepared mesh,
mode set, influence and solver; the distributed result is the owner sum of
shares, differentiated with ordinary JAX through the collectives.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec

import phydrax as phx
from phydrax import config as pxconfig
from phydrax.atomistic._domain_plan import (
    AtomisticDomainPlan,
    AtomisticDomainReductionPolicy,
)
from phydrax.atomistic._domain_runtime import prepare_atomistic_domain
from phydrax.atomistic._electrostatics import (
    _coulomb_influence,
    ChargeNeutralityPolicy,
    ewald_background_energy,
    ewald_self_energy,
    ewald_slab_boundary_energy,
    ewald_slab_height,
    EwaldBoundaryConvention,
    EwaldReferencePotential,
    ParticleMeshEwaldPotential,
)
from phydrax.atomistic._force_field import (
    lennard_jones_pme_influence,
    lennard_jones_pme_self_energy,
    PreparedGeneralForceFieldTerm,
)
from phydrax.atomistic._latent_electrostatics import (
    LatentEwaldDefinition,
    LatentEwaldResources,
)
from phydrax.atomistic._multipole_basis import (
    CartesianMultipoles,
    GaussianMultipoleWidths,
)
from phydrax.atomistic._multipole_electrostatics import (
    FiniteMultipoleRoute,
    MultipoleElectrostaticsDefinition,
    MultipoleElectrostaticsPlan,
    MultipoleSiteSet,
    ParticleMeshMultipoleRoute,
)
from phydrax.atomistic._polarization import (
    evaluate_prepared_polarization,
    PermanentMultipoleSiteData,
    PolarizationCurvatureProbe,
    PolarizationPlan,
    PolarizationScaleData,
    PolarizationSolverKind,
    PolarizationStabilityPolicy,
)
from phydrax.atomistic._reciprocal_distribution import (
    AtomisticReciprocalDistributionPlan,
    AtomisticReciprocalKind,
    owner_reciprocal_actions,
    owner_reciprocal_energy,
    OwnerMultipoleRelation,
    OwnerPolarizationPlan,
    OwnerPolarizationRelations,
    OwnerPolarizationSites,
    prepare_owner_polarization,
    prepare_reciprocal_distribution,
    ReciprocalDistributionCapacityError,
    ReciprocalSitePayload,
    solve_owner_polarization,
)
from phydrax.atomistic._reciprocal_mesh import PreparedReciprocalMesh, ReciprocalMeshPlan
from phydrax.atomistic._types import AtomisticScaleContract
from phydrax.discretization import PeriodicCell
from phydrax.discretization.particle._distributed import FractionalOwnerPartition
from phydrax.discretization.spatial._distributed_relations import (
    DistributedOwnershipPlan,
)
from phydrax.discretization.spectral._distributed import (
    DistributedSpectralExecutionPlan,
    SpectralMeshTopology,
)
from phydrax.discretization.spectral._precision import SpectralPrecisionPolicy
from phydrax.nn.atomistic._mace import MACEArchitecture, MACEPotential
from phydrax.nn.atomistic._mace_les import MACELESPotential
from phydrax.units import ANGSTROM, ELECTRONVOLT
from tests._support import atomistic_domain as domain_support


TRICLINIC = np.asarray([[6.0, 0.0, 0.0], [1.0, 6.5, 0.0], [0.5, 1.0, 7.0]])
LOCAL = 8


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


# --- ownership and owner-blocked layouts ------------------------------------------------


def _ownership(
    owners: int, local: int = LOCAL, *, devices: bool = False
) -> DistributedOwnershipPlan:
    """Lane-reference owners, or device owners (skipped when too few devices)."""
    return domain_support.ownership(owners, devices=devices, local_capacity=local)


class _Blocks:
    """Host owner-blocking of global rows by their declared fractional owner."""

    def __init__(self, owners_of_atom: np.ndarray, owners: int, local: int) -> None:
        self.owners = owners
        self.local = local
        self.owner = np.asarray(owners_of_atom, dtype=np.int64)
        self.slot = np.zeros_like(self.owner)
        counts = np.zeros((owners,), dtype=np.int64)
        for atom, owner in enumerate(self.owner):
            self.slot[atom] = counts[owner]
            counts[owner] += 1
        assert np.all(counts <= local)
        self.row = self.owner * local + self.slot

    def block(self, values: Any, fill: float = 0.0) -> np.ndarray:
        array = np.asarray(values)
        out = np.full(
            (self.owners * self.local,) + array.shape[1:], fill, dtype=array.dtype
        )
        out[self.row] = array
        return out

    def mask(self) -> np.ndarray:
        out = np.zeros((self.owners * self.local,), dtype=np.bool_)
        out[self.row] = True
        return out

    def unblock(self, values: Any) -> np.ndarray:
        return np.asarray(values)[self.row]


def _partition_owners(
    partition: FractionalOwnerPartition, positions: np.ndarray
) -> np.ndarray:
    fractional = positions @ np.linalg.inv(np.asarray(partition.cell.vectors))
    wrapped = fractional - np.floor(fractional)
    return np.asarray(partition.owners(jnp.asarray(wrapped)))


def _configuration(seed: int, count: int, vectors: np.ndarray = TRICLINIC) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, 1.0, (count, 3)) @ vectors
    positions[0] += vectors[1]  # a stored, unwrapped coordinate
    return positions


def _neutral(seed: int, count: int) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=count)
    return values - values.mean()


def _system(charges: Any, cell: PeriodicCell, *, types: Any = None) -> Any:
    count = len(charges)
    return phx.atomistic.AtomisticSystemPlan(
        np.arange(count),
        np.ones((count,), dtype=np.int64),
        np.ones((count,)),
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=np.zeros((count,), dtype=np.int64)
        if types is None
        else np.asarray(types),
        charges=np.asarray(charges),
        cell=cell,
    ).prepare()


def _energy_function(
    prepared: Any,
    ownership: DistributedOwnershipPlan,
    payload_specs: ReciprocalSitePayload,
) -> Callable[..., tuple[jax.Array, ...]]:
    """Owner-mapped ``(shares, successful, capacity, covered)`` of the term."""
    axis = ownership.axis_name
    row = PartitionSpec(axis)

    def region(positions: Any, payload: Any, mask: Any, cell: Any) -> tuple[Any, ...]:
        share, evidence = owner_reciprocal_energy(
            prepared, positions, payload, mask, cell, axis
        )
        return (
            share[None],
            evidence.successful[None],
            evidence.capacity_successful[None],
            evidence.scientific_successful[None],
            evidence.patch_covered[None],
        )

    return ownership.map(
        region,
        (row, payload_specs, row, PartitionSpec()),
        (row, row, row, row, row),
    )


def _specs(axis: str, payload: ReciprocalSitePayload) -> ReciprocalSitePayload:
    return jax.tree.map(lambda _: PartitionSpec(axis), payload)


# --- spectral lane topology -------------------------------------------------------------


def test_lane_reference_spectral_topology_runs_only_inside_owner_regions() -> None:
    topology = SpectralMeshTopology.lane_reference(2, axis_name="owner")
    assert topology.lanes == 2 and topology.mesh_shape == (2,)
    assert topology.topology_id != SpectralMeshTopology.one_device().topology_id
    plan = DistributedSpectralExecutionPlan(
        topology,
        (4, 4, 4),
        owner_id="lane-spectral-test",
        admitted_payload_shapes=(),
        precision=SpectralPrecisionPolicy(np.float64),
    )
    assert plan.physical_layout.local_shape(topology) == (2, 4, 4)
    values = np.random.default_rng(1).normal(size=(4, 4, 4))
    with pytest.raises(ValueError, match="lane-reference"):
        plan.to_modal(values)
    blocks = jnp.asarray(values).reshape((2, 2, 4, 4))
    modal = jax.vmap(plan.forward_local, axis_name="owner")(blocks)
    assembled = jnp.concatenate(tuple(modal), axis=1)
    np.testing.assert_allclose(
        assembled, np.fft.fftn(values, norm="ortho"), rtol=1e-12, atol=1e-12
    )
    back = jax.vmap(plan.inverse_local, axis_name="owner")(modal)
    np.testing.assert_allclose(
        jnp.real(back.reshape((4, 4, 4))), values, rtol=1e-12, atol=1e-12
    )


# --- particle-mesh Ewald -------------------------------------------------------------------


def _pme_case(
    owners: int,
    *,
    count: int = 9,
    neutrality: ChargeNeutralityPolicy = "require-neutral",
    boundary: EwaldBoundaryConvention = "tin-foil",
    positions: np.ndarray | None = None,
    grid: tuple[int, int, int] = (12, 12, 12),
) -> tuple[Any, np.ndarray, np.ndarray, PeriodicCell]:
    cell = PeriodicCell(TRICLINIC)
    position = _configuration(3, count) if positions is None else positions
    charges = _neutral(5, position.shape[0])
    if neutrality == "uniform-background":
        charges = charges + 0.3
    term = ParticleMeshEwaldPotential(
        1.1,
        2.8,
        grid,
        spline_degree=4,
        neutrality=neutrality,
        boundary=boundary,
    ).prepare(_system(charges, cell))
    return term, position, charges, cell


def _pme_reference(term: Any, positions: Any, charges: Any, vectors: Any) -> jax.Array:
    """The canonical single-device mesh, influence and system terms of the same term."""
    settings = term.plan.settings
    constant = term.system.plan.units.coulomb_constant
    geometry = term.mesh.geometry(vectors)
    routes = term.mesh.build(positions, geometry)
    influence = _coulomb_influence(geometry, settings.alpha, constant)
    reciprocal = term.mesh.quadratic_energy(
        routes, charges[:, None], geometry, influence, zero_mode="exclude"
    )
    total = reciprocal.value + ewald_self_energy(settings.alpha, constant, charges)
    if settings.neutrality == "uniform-background":
        total = total + ewald_background_energy(
            settings.alpha, constant, jnp.sum(charges), geometry.volume
        )
    if settings.boundary == "vacuum-sphere":
        dipole = charges @ positions
        total = total + 2.0 * jnp.pi * constant * jnp.sum(dipole**2) / (
            3.0 * geometry.volume
        )
    return total


def _distributed_energy(
    term: Any,
    owners: int,
    positions: np.ndarray,
    charges: np.ndarray,
    cell: PeriodicCell,
    *,
    owner_of_atom: np.ndarray | None = None,
    image_counts: np.ndarray | None = None,
    devices: bool = False,
    **plan_options: Any,
) -> tuple[Callable[..., Any], _Blocks, np.ndarray, Any, Any]:
    ownership = _ownership(owners, devices=devices)
    partition = FractionalOwnerPartition(cell, (owners, 1, 1))
    owner = (
        _partition_owners(partition, positions)
        if owner_of_atom is None
        else owner_of_atom
    )
    blocks = _Blocks(owner, owners, LOCAL)
    plan_options.setdefault("reuse_reach", 0.4)
    prepared = prepare_reciprocal_distribution(
        AtomisticReciprocalDistributionPlan(term, **plan_options), ownership, partition
    )
    payload = ReciprocalSitePayload(
        values=jnp.asarray(blocks.block(charges)),
        image_counts=None
        if image_counts is None
        else jnp.asarray(blocks.block(image_counts)),
    )
    mapped = _energy_function(prepared, ownership, _specs(ownership.axis_name, payload))
    return mapped, blocks, blocks.mask(), payload, prepared


@pytest.mark.parametrize(
    ("strategy", "devices"),
    [
        ("distributed-fft", False),
        ("replicated-mesh", False),
        ("distributed-fft", True),
    ],
    ids=["lanes-fft", "lanes-replicated", "devices-fft"],
)
@pytest.mark.parametrize("owners", [2, 3])
def test_pme_energy_forces_and_cell_gradient_match_single_device(
    owners: int, strategy: str, devices: bool
) -> None:
    term, positions, charges, cell = _pme_case(owners)
    mapped, blocks, mask, payload, prepared = _distributed_energy(
        term, owners, positions, charges, cell, strategy=strategy, devices=devices
    )
    assert prepared.kind is AtomisticReciprocalKind.PME
    vectors = jnp.asarray(TRICLINIC)
    x = jnp.asarray(blocks.block(positions))

    def distributed(position: Any, lattice: Any) -> jax.Array:
        return jnp.sum(mapped(position, payload, jnp.asarray(mask), lattice)[0])

    shares, successful, capacity, scientific, covered = mapped(
        x, payload, jnp.asarray(mask), vectors
    )
    assert bool(jnp.all(successful & capacity & scientific & covered))
    reference = lambda p, h: _pme_reference(term, p, jnp.asarray(charges), h)
    np.testing.assert_allclose(
        float(jnp.sum(shares)),
        float(reference(jnp.asarray(positions), vectors)),
        rtol=1e-11,
        atol=1e-12,
    )
    gradient, cell_gradient = jax.grad(distributed, argnums=(0, 1))(x, vectors)
    expected, expected_cell = jax.grad(reference, argnums=(0, 1))(
        jnp.asarray(positions), vectors
    )
    np.testing.assert_allclose(blocks.unblock(gradient), expected, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(cell_gradient, expected_cell, rtol=1e-9, atol=1e-11)
    padding = np.asarray(gradient)[~mask]
    assert np.all(padding == 0.0)


def test_in_region_gradients_return_each_cotangent_exactly_once() -> None:
    """Each owner seeds only its own share; collectives return the full gradient."""
    owners = 3
    term, positions, charges, cell = _pme_case(owners)
    ownership = _ownership(owners)
    partition = FractionalOwnerPartition(cell, (owners, 1, 1))
    blocks = _Blocks(_partition_owners(partition, positions), owners, LOCAL)
    prepared = prepare_reciprocal_distribution(
        AtomisticReciprocalDistributionPlan(term, reuse_reach=0.4), ownership, partition
    )
    axis = ownership.axis_name
    row = PartitionSpec(axis)

    def region(
        position: Any, values: Any, mask: Any, lattice: Any
    ) -> tuple[Any, Any, Any]:
        def share(p: Any, q: Any, h: Any) -> Any:
            return owner_reciprocal_energy(
                prepared, p, ReciprocalSitePayload(values=q), mask, h, axis
            )[0]

        gradient, charge_gradient, cell_partial = jax.grad(share, argnums=(0, 1, 2))(
            position, values, lattice
        )
        return gradient, charge_gradient, cell_partial[None]

    gradient, charge_gradient, cell_partials = ownership.map(
        region, (row, row, row, PartitionSpec()), (row, row, row)
    )(
        jnp.asarray(blocks.block(positions)),
        jnp.asarray(blocks.block(charges)),
        jnp.asarray(blocks.mask()),
        jnp.asarray(TRICLINIC),
    )
    expected, expected_charges, expected_cell = jax.grad(
        lambda p, q, h: _pme_reference(term, p, q, h), argnums=(0, 1, 2)
    )(jnp.asarray(positions), jnp.asarray(charges), jnp.asarray(TRICLINIC))
    np.testing.assert_allclose(blocks.unblock(gradient), expected, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(
        blocks.unblock(charge_gradient), expected_charges, rtol=1e-9, atol=1e-11
    )
    # Cell partials are per-owner: their owner sum (never times the owner count).
    np.testing.assert_allclose(
        jnp.sum(cell_partials, axis=0), expected_cell, rtol=1e-9, atol=1e-11
    )


def test_pme_actions_are_exact_potentials_and_forces() -> None:
    owners = 2
    term, positions, charges, cell = _pme_case(owners)
    ownership = _ownership(owners)
    partition = FractionalOwnerPartition(cell, (owners, 1, 1))
    blocks = _Blocks(_partition_owners(partition, positions), owners, LOCAL)
    prepared = prepare_reciprocal_distribution(
        AtomisticReciprocalDistributionPlan(term, reuse_reach=0.4), ownership, partition
    )
    axis = ownership.axis_name
    row = PartitionSpec(axis)

    def region(position: Any, values: Any, mask: Any, lattice: Any) -> tuple[Any, Any]:
        actions = owner_reciprocal_actions(
            prepared, position, ReciprocalSitePayload(values=values), mask, lattice, axis
        )
        return actions.payload_gradient.values, actions.forces

    mapped = ownership.map(region, (row, row, row, PartitionSpec()), (row, row))
    x = jnp.asarray(blocks.block(positions))
    q = jnp.asarray(blocks.block(charges))
    mask = jnp.asarray(blocks.mask())
    vectors = jnp.asarray(TRICLINIC)
    potentials, forces = mapped(x, q, mask, vectors)
    expected_forces, expected_potentials = jax.grad(
        lambda p, c: -_pme_reference(term, p, c, vectors), argnums=(0, 1)
    )(jnp.asarray(positions), jnp.asarray(charges))
    np.testing.assert_allclose(
        blocks.unblock(potentials), -expected_potentials, rtol=1e-9, atol=1e-11
    )
    np.testing.assert_allclose(
        blocks.unblock(forces), expected_forces, rtol=1e-9, atol=1e-11
    )

    # The actions are differentiable: their charge Jacobian is the energy's
    # charge Hessian and the force/charge mixed derivative. Reverse-over-reverse
    # first, then forward mode (JVP) through the same collectives.
    rng = np.random.default_rng(29)
    direction = jnp.asarray(blocks.block(rng.normal(size=charges.shape)))
    direction3 = jnp.asarray(blocks.block(rng.normal(size=positions.shape)))
    energy_mapped = _energy_function(
        prepared, ownership, _specs(axis, ReciprocalSitePayload(values=q))
    )

    def energy(p: Any, c: Any) -> jax.Array:
        return jnp.sum(
            energy_mapped(p, ReciprocalSitePayload(values=c), mask, vectors)[0]
        )

    hessian_action = jax.grad(
        lambda c: jnp.vdot(direction, mapped(x, c, mask, vectors)[0])
    )(q)
    expected_action = jax.grad(
        lambda c: jnp.vdot(direction, jax.grad(energy, argnums=1)(x, c))
    )(q)
    assert float(jnp.max(jnp.abs(hessian_action))) > 0.0
    np.testing.assert_allclose(hessian_action, expected_action, rtol=1e-8, atol=1e-11)
    reference_action = jax.grad(
        lambda c: jnp.vdot(
            jnp.asarray(blocks.unblock(direction)),
            jax.grad(lambda v: _pme_reference(term, jnp.asarray(positions), v, vectors))(
                c
            ),
        )
    )(jnp.asarray(charges))
    np.testing.assert_allclose(
        blocks.unblock(hessian_action), reference_action, rtol=1e-8, atol=1e-11
    )
    mixed = jax.grad(lambda c: jnp.vdot(direction3, mapped(x, c, mask, vectors)[1]))(q)
    expected_mixed = jax.grad(
        lambda c: -jnp.vdot(direction3, jax.grad(energy, argnums=0)(x, c))
    )(q)
    np.testing.assert_allclose(mixed, expected_mixed, rtol=1e-8, atol=1e-11)

    # Forward mode: JVPs of the actions against forward-over-reverse second
    # derivatives of the owner energy shares.
    _, potential_tangent = jax.jvp(
        lambda c: mapped(x, c, mask, vectors)[0], (q,), (direction,)
    )
    _, expected_potential_tangent = jax.jvp(
        lambda c: jax.grad(energy, argnums=1)(x, c), (q,), (direction,)
    )
    np.testing.assert_allclose(
        potential_tangent, expected_potential_tangent, rtol=1e-8, atol=1e-11
    )
    # The charge Hessian is symmetric: forward and reverse actions agree.
    np.testing.assert_allclose(potential_tangent, hessian_action, rtol=1e-8, atol=1e-11)
    _, force_charge_tangent = jax.jvp(
        lambda c: mapped(x, c, mask, vectors)[1], (q,), (direction,)
    )
    _, expected_force_charge = jax.jvp(
        lambda c: -jax.grad(energy, argnums=0)(x, c), (q,), (direction,)
    )
    np.testing.assert_allclose(
        force_charge_tangent, expected_force_charge, rtol=1e-8, atol=1e-11
    )
    _, force_position_tangent = jax.jvp(
        lambda p: mapped(p, q, mask, vectors)[1], (x,), (direction3,)
    )
    _, expected_force_position = jax.jvp(
        lambda p: -jax.grad(energy, argnums=0)(p, q), (x,), (direction3,)
    )
    assert float(jnp.max(jnp.abs(force_position_tangent))) > 0.0
    np.testing.assert_allclose(
        force_position_tangent, expected_force_position, rtol=1e-8, atol=1e-11
    )


def test_nonfinite_active_inputs_fail_scientific_status() -> None:
    """Sanitized NaN inputs must never yield a successful finite-looking energy."""
    owners = 2
    term, positions, charges, cell = _pme_case(owners)
    mapped, blocks, mask, payload, _ = _distributed_energy(
        term, owners, positions, charges, cell
    )
    x = jnp.asarray(blocks.block(positions))
    vectors = jnp.asarray(TRICLINIC)
    valid = mapped(x, payload, jnp.asarray(mask), vectors)
    assert bool(jnp.all(valid[1] & valid[3]))
    bad_charge = ReciprocalSitePayload(
        values=payload.values.at[int(blocks.row[3])].set(jnp.nan)
    )
    _, successful, capacity, scientific, _ = mapped(
        x, bad_charge, jnp.asarray(mask), vectors
    )
    assert not bool(jnp.any(successful)) and not bool(jnp.any(scientific))
    assert bool(jnp.all(capacity))
    bad_position = x.at[int(blocks.row[1]), 0].set(jnp.inf)
    _, successful, _, scientific, _ = mapped(
        bad_position, payload, jnp.asarray(mask), vectors
    )
    assert not bool(jnp.any(successful)) and not bool(jnp.any(scientific))
    # Inactive padding rows may hold anything.
    padding = int(np.flatnonzero(~mask)[0])
    _, successful, _, _, _ = mapped(
        x.at[padding].set(jnp.nan),
        ReciprocalSitePayload(values=payload.values.at[padding].set(jnp.nan)),
        jnp.asarray(mask),
        vectors,
    )
    assert bool(jnp.all(successful))


def test_empty_owner_and_repeated_self_images_match_single_device() -> None:
    """Every atom on one owner of three; a short axis makes each spline wrap onto itself."""
    owners = 3
    short = np.asarray([[2.4, 0.0, 0.0], [0.7, 6.5, 0.0], [0.5, 1.0, 7.0]])
    cell = PeriodicCell(short)
    positions = np.asarray([[0.2, 1.0, 2.0], [0.5, 4.0, 5.5], [0.6, 2.2, 0.4]])
    charges = np.asarray([0.8, -0.5, -0.3])
    term = ParticleMeshEwaldPotential(1.5, 1.1, (6, 12, 12), spline_degree=4).prepare(
        _system(charges, cell)
    )
    mapped, blocks, mask, payload, _ = _distributed_energy(
        term,
        owners,
        positions,
        charges,
        cell,
        owner_of_atom=np.zeros((3,), dtype=np.int64),
        reuse_reach=6.0,
    )
    x = jnp.asarray(blocks.block(positions))
    shares, successful, capacity, _, covered = mapped(
        x, payload, jnp.asarray(mask), jnp.asarray(short)
    )
    assert bool(jnp.all(successful & capacity & covered))
    assert (
        float(shares[1]) != 0.0 or float(shares[2]) != 0.0
    )  # empty owners still own modes
    np.testing.assert_allclose(
        float(jnp.sum(shares)),
        float(
            _pme_reference(
                term, jnp.asarray(positions), jnp.asarray(charges), jnp.asarray(short)
            )
        ),
        rtol=1e-11,
        atol=1e-12,
    )


def test_patch_capacity_refusals_are_distinct_from_scientific_status() -> None:
    owners = 2
    # At grid12 the spline halo covers the whole x axis; grid24 has a genuinely
    # local patch, so an incorrectly owned interior point can overflow it.
    term, positions, charges, cell = _pme_case(owners, grid=(24, 24, 24))
    ownership = _ownership(owners)
    partition = FractionalOwnerPartition(cell, (owners, 1, 1))
    with pytest.raises(ValueError, match="packet_planes"):
        prepare_reciprocal_distribution(
            AtomisticReciprocalDistributionPlan(term, reuse_reach=0.4, packet_planes=64),
            ownership,
            partition,
        )
    with pytest.raises(ReciprocalDistributionCapacityError, match="maximum_packet_bytes"):
        prepare_reciprocal_distribution(
            AtomisticReciprocalDistributionPlan(
                term, reuse_reach=0.4, maximum_packet_bytes=64
            ),
            ownership,
            partition,
        )
    with pytest.raises(ValueError, match="not divisible"):
        prepare_reciprocal_distribution(
            AtomisticReciprocalDistributionPlan(term, reuse_reach=0.4),
            _ownership(5),
            FractionalOwnerPartition(cell, (5, 1, 1)),
        )
    # An atom far outside its owner's certified region overflows the patch:
    # capacity failure, scientifically valid inputs.
    fractional = np.linalg.solve(np.asarray(cell.vectors).T, positions.T).T
    fractional[1, 0] = 0.25
    positions = fractional @ np.asarray(cell.vectors)
    owner = _partition_owners(partition, positions).copy()
    owner[1] = 1 - owner[1]
    mapped, blocks, mask, payload, _ = _distributed_energy(
        term,
        owners,
        positions,
        charges,
        cell,
        owner_of_atom=owner,
        reuse_reach=0.0,
        patch_margin_cells=0,
    )
    _, successful, capacity, scientific, covered = mapped(
        jnp.asarray(blocks.block(positions)),
        payload,
        jnp.asarray(mask),
        jnp.asarray(TRICLINIC),
    )
    assert not bool(jnp.any(covered))
    assert not bool(jnp.any(capacity)) and not bool(jnp.any(successful))
    assert bool(jnp.all(scientific))


@pytest.mark.parametrize(
    ("neutrality", "boundary"),
    [("uniform-background", "tin-foil"), ("require-neutral", "vacuum-sphere")],
    ids=["uniform-background", "vacuum-sphere"],
)
def test_background_and_vacuum_boundary_system_reductions_match_single_device(
    neutrality: ChargeNeutralityPolicy, boundary: EwaldBoundaryConvention
) -> None:
    """Each system term separately: the vacuum sphere requires a neutral cell."""
    owners = 2
    term, positions, charges, cell = _pme_case(
        owners, neutrality=neutrality, boundary=boundary
    )
    counts = np.zeros((positions.shape[0], 3), dtype=np.int64)
    stored = positions.copy()
    stored[2] -= TRICLINIC[0]
    counts[2] = (1, 0, 0)  # unwrapped = stored + counts @ H
    mapped, blocks, mask, payload, _ = _distributed_energy(
        term, owners, stored, charges, cell, image_counts=counts
    )

    def distributed(position: Any, lattice: Any) -> jax.Array:
        return jnp.sum(mapped(position, payload, jnp.asarray(mask), lattice)[0])

    x = jnp.asarray(blocks.block(stored))
    vectors = jnp.asarray(TRICLINIC)
    reference = lambda p, h: _pme_reference(term, p, jnp.asarray(charges), h)
    np.testing.assert_allclose(
        float(distributed(x, vectors)),
        float(reference(jnp.asarray(positions), vectors)),
        rtol=1e-11,
        atol=1e-12,
    )
    gradient = jax.grad(distributed)(x, vectors)
    expected = jax.grad(reference)(jnp.asarray(positions), vectors)
    np.testing.assert_allclose(blocks.unblock(gradient), expected, rtol=1e-9, atol=1e-11)


# --- explicit modes ---------------------------------------------------------------------


@pytest.mark.parametrize("devices", [False, True], ids=["lanes", "devices"])
def test_explicit_mode_ewald_matches_single_device(devices: bool) -> None:
    owners = 3
    cell = PeriodicCell(TRICLINIC)
    positions = _configuration(7, 7)
    charges = _neutral(9, 7)
    term = EwaldReferencePotential(1.0, 2.6, 5, mode_chunk=17).prepare(
        _system(charges, cell)
    )
    mapped, blocks, mask, payload, prepared = _distributed_energy(
        term, owners, positions, charges, cell, devices=devices
    )
    assert prepared.report.strategy == "explicit-modes"
    assert prepared.report.mode_chunk % owners == 0

    def reference(p: Any, q: Any, h: Any) -> jax.Array:
        settings = term.plan.settings
        constant = term.system.plan.units.coulomb_constant
        geometry = term.modes.geometry(h)
        influence = _coulomb_influence(geometry, settings.alpha, constant)
        value = term.modes.quadratic_energy(
            p, q[:, None], geometry, influence, zero_mode="exclude"
        ).value
        return value + ewald_self_energy(settings.alpha, constant, q)

    def distributed(position: Any, lattice: Any) -> jax.Array:
        return jnp.sum(mapped(position, payload, jnp.asarray(mask), lattice)[0])

    x = jnp.asarray(blocks.block(positions))
    vectors = jnp.asarray(TRICLINIC)
    np.testing.assert_allclose(
        float(distributed(x, vectors)),
        float(reference(jnp.asarray(positions), jnp.asarray(charges), vectors)),
        rtol=1e-11,
    )
    gradient, cell_gradient = jax.grad(distributed, argnums=(0, 1))(x, vectors)
    expected, expected_cell = jax.grad(reference, argnums=(0, 2))(
        jnp.asarray(positions), jnp.asarray(charges), vectors
    )
    np.testing.assert_allclose(blocks.unblock(gradient), expected, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(cell_gradient, expected_cell, rtol=1e-9, atol=1e-11)


SLAB = np.asarray([[6.0, 0.0, 0.0], [1.5, 6.0, 0.0], [0.0, 0.0, 20.0]])


def test_slab_yeh_berkowitz_ewald_matches_single_device() -> None:
    """Two periodic axes: the slab moments are owner-summed system reductions."""
    owners = 2
    cell = PeriodicCell(SLAB, periodic_axes=(True, True, False))
    rng = np.random.default_rng(17)
    planar = rng.uniform(0.0, 1.0, (5, 2)) @ SLAB[:2, :2]
    positions = np.column_stack([planar, np.asarray([1.0, 1.8, 2.7, 3.5, 4.1])])
    charges = _neutral(19, 5)
    axial = int(np.ceil(13.2 * 20.0 / (2.0 * np.pi))) + 1
    term = EwaldReferencePotential(
        1.2, 2.8, (6, 6, axial), boundary="slab-yeh-berkowitz"
    ).prepare(_system(charges, cell))
    mapped, blocks, mask, payload, _ = _distributed_energy(
        term, owners, positions, charges, cell
    )
    vectors = jnp.asarray(SLAB)

    def reference(p: Any) -> jax.Array:
        settings = term.plan.settings
        constant = term.system.plan.units.coulomb_constant
        q = jnp.asarray(charges)
        geometry = term.modes.geometry(vectors)
        influence = _coulomb_influence(geometry, settings.alpha, constant)
        value = term.modes.quadratic_energy(
            p, q[:, None], geometry, influence, zero_mode="exclude"
        ).value
        height, normal = ewald_slab_height(cell, vectors, geometry.volume)
        coordinate = (p - cell.origin) @ geometry.inverse[:, normal] * height
        boundary = ewald_slab_boundary_energy(
            constant,
            geometry.volume,
            height,
            jnp.sum(q),
            jnp.sum(q * coordinate),
            jnp.sum(q * coordinate**2),
        )
        return value + ewald_self_energy(settings.alpha, constant, q) + boundary

    def distributed(position: Any) -> jax.Array:
        return jnp.sum(mapped(position, payload, jnp.asarray(mask), vectors)[0])

    x = jnp.asarray(blocks.block(positions))
    _, successful, capacity, scientific, _ = mapped(
        x, payload, jnp.asarray(mask), vectors
    )
    assert bool(jnp.all(successful & capacity & scientific))
    np.testing.assert_allclose(
        float(distributed(x)), float(reference(jnp.asarray(positions))), rtol=1e-11
    )
    np.testing.assert_allclose(
        blocks.unblock(jax.grad(distributed)(x)),
        jax.grad(reference)(jnp.asarray(positions)),
        rtol=1e-9,
        atol=1e-11,
    )


def test_latent_ewald_matches_single_device_and_refuses_finite_branch() -> None:
    owners = 2
    scale = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
    definition = LatentEwaldDefinition()
    count = 6
    binding = definition.prepare_graph(
        scale,
        count,
        resources=LatentEwaldResources(mode_extents=definition.covering_extents(9.0)),
    )
    cell = PeriodicCell(TRICLINIC)
    positions = _configuration(11, count)
    charges = np.random.default_rng(13).normal(size=count)
    mapped, blocks, mask, payload, _ = _distributed_energy(
        binding, owners, positions, charges, cell
    )

    def reference(p: Any) -> jax.Array:
        result = binding.energy(
            jnp.asarray(charges),
            p,
            jnp.ones((count,), dtype=bool),
            jnp.asarray(TRICLINIC)[None],
            jnp.ones((1, 3), dtype=bool),
        )
        return result.reciprocal_energy[0] + result.self_energy[0]

    def distributed(position: Any) -> jax.Array:
        return jnp.sum(
            mapped(position, payload, jnp.asarray(mask), jnp.asarray(TRICLINIC))[0]
        )

    x = jnp.asarray(blocks.block(positions))
    np.testing.assert_allclose(
        float(distributed(x)), float(reference(jnp.asarray(positions))), rtol=1e-11
    )
    np.testing.assert_allclose(
        blocks.unblock(jax.grad(distributed)(x)),
        jax.grad(reference)(jnp.asarray(positions)),
        rtol=1e-9,
        atol=1e-11,
    )
    finite = LatentEwaldDefinition(branch="finite").prepare_graph(
        scale, count, resources=LatentEwaldResources()
    )
    with pytest.raises(ValueError, match="pair"):
        AtomisticReciprocalDistributionPlan(finite, reuse_reach=0.4)


# --- dispersion meshes ------------------------------------------------------------------


def _reciprocal_mesh(term: PreparedGeneralForceFieldTerm) -> PreparedReciprocalMesh:
    mesh = term.reciprocal_mesh
    if mesh is None:
        pytest.fail("A prepared LJ-PME term binds its term-owned reciprocal mesh.")
    return mesh


# Every particle-mesh kind runs unchanged through both canonical FFT layouts:
# x-slabs and a non-square (2, 3) pencil grid (owner r at (r % 2, r // 2)).
MESH_LAYOUTS = pytest.mark.parametrize(
    ("owners", "layout"),
    [(2, {}), (6, {"schedule": "pencil", "pencil_grid": (2, 3)})],
    ids=["slab-2", "pencil-2x3"],
)


@MESH_LAYOUTS
def test_lj_pme_anchor_frame_matches_single_device(
    owners: int, layout: dict[str, Any]
) -> None:
    cell = PeriodicCell(TRICLINIC)
    positions = _configuration(17, 6)
    types = np.asarray([0, 1, 0, 1, 1, 0])
    c6 = np.asarray([[0.1, 0.2], [0.2, 0.5]])
    term = phx.atomistic.LennardJonesPMEPotential(c6, 0.9, 2.8, (12, 12, 12)).prepare(
        _system(np.zeros(6), cell, types=types)
    )
    mesh = _reciprocal_mesh(term)
    assert mesh.anchor_index is not None
    ownership = _ownership(owners)
    partition = FractionalOwnerPartition(cell, (owners, 1, 1))
    blocks = _Blocks(_partition_owners(partition, positions), owners, LOCAL)
    anchor = np.zeros((6,), dtype=np.bool_)
    anchor[term.site_anchor] = True
    prepared = prepare_reciprocal_distribution(
        AtomisticReciprocalDistributionPlan(term, reuse_reach=0.4, **layout),
        ownership,
        partition,
    )
    payload = ReciprocalSitePayload(
        rows=jnp.asarray(blocks.block(types).astype(np.int32)),
        anchor=jnp.asarray(blocks.block(anchor)),
    )
    mapped = _energy_function(prepared, ownership, _specs(ownership.axis_name, payload))
    c6_, alpha, factors, _ = term.plan.arrays

    def reference(p: Any) -> jax.Array:
        vectors = jnp.asarray(TRICLINIC)
        geometry = mesh.geometry(vectors)
        routes = mesh.build(p, geometry)
        site_factor = factors[jnp.asarray(types)]
        influence = lennard_jones_pme_influence(alpha.reshape(()), geometry.volume)
        value = mesh.quadratic_energy(
            routes, site_factor, geometry, influence, zero_mode="include"
        ).value
        return value + lennard_jones_pme_self_energy(
            alpha.reshape(()), c6_, jnp.asarray(types), jnp.ones((6,), dtype=bool)
        )

    def distributed(position: Any) -> jax.Array:
        return jnp.sum(
            mapped(position, payload, jnp.asarray(blocks.mask()), jnp.asarray(TRICLINIC))[
                0
            ]
        )

    x = jnp.asarray(blocks.block(positions))
    np.testing.assert_allclose(
        float(distributed(x)), float(reference(jnp.asarray(positions))), rtol=1e-10
    )
    np.testing.assert_allclose(
        blocks.unblock(jax.grad(distributed)(x)),
        jax.grad(reference)(jnp.asarray(positions)),
        rtol=1e-8,
        atol=1e-11,
    )
    missing = ReciprocalSitePayload(rows=payload.rows)
    with pytest.raises(ValueError, match="anchor"):
        _energy_function(prepared, ownership, _specs(ownership.axis_name, missing))(
            x, missing, jnp.asarray(blocks.mask()), jnp.asarray(TRICLINIC)
        )


@MESH_LAYOUTS
def test_fourier_d3_matches_single_device(owners: int, layout: dict[str, Any]) -> None:
    from tests.unit.atomistic.test_dispersion_fourier_d3 import (
        _system as d3_system,
        _term as d3_term,
        TRICLINIC as D3_CELL,
    )

    term = d3_term((12, 12, 12), degree=4).prepare(d3_system(vectors=D3_CELL))
    count = term.system.capacity
    cell = term.mesh.cell
    positions = np.asarray([[0.3, 0.4, 0.2], [1.9, 1.1, 1.6], [3.4, 3.1, 3.6]])
    coordination = jnp.asarray([0.7, 1.3, 0.4])
    coefficients = term.coefficients(coordination)
    rows = np.asarray(term.channel_rows)
    ownership = _ownership(owners)
    partition = FractionalOwnerPartition(cell, (owners, 1, 1))
    blocks = _Blocks(_partition_owners(partition, positions), owners, LOCAL)
    prepared = prepare_reciprocal_distribution(
        AtomisticReciprocalDistributionPlan(term, reuse_reach=0.4, **layout),
        ownership,
        partition,
    )
    payload = ReciprocalSitePayload(
        values=jnp.asarray(blocks.block(np.asarray(coefficients))),
        rows=jnp.asarray(blocks.block(rows).astype(np.int32)),
    )
    mapped = _energy_function(prepared, ownership, _specs(ownership.axis_name, payload))
    vectors = jnp.asarray(D3_CELL)

    def reference(p: Any) -> jax.Array:
        geometry = term.mesh.geometry(vectors)
        value, _ = term.reciprocal_energy(p, geometry, coefficients)
        return (
            value + term.self_energy(coefficients, term.channel_rows, term.active)
        ) * term.energy_scale

    def distributed(position: Any) -> jax.Array:
        return jnp.sum(mapped(position, payload, jnp.asarray(blocks.mask()), vectors)[0])

    x = jnp.asarray(blocks.block(positions))
    assert count == positions.shape[0]
    np.testing.assert_allclose(
        float(distributed(x)), float(reference(jnp.asarray(positions))), rtol=1e-10
    )
    np.testing.assert_allclose(
        blocks.unblock(jax.grad(distributed)(x)),
        jax.grad(reference)(jnp.asarray(positions)),
        rtol=1e-8,
        atol=1e-12,
    )


# --- multipoles -------------------------------------------------------------------------


def _multipole_moments(count: int) -> CartesianMultipoles:
    rng = np.random.default_rng(21)
    charges = rng.normal(size=count)
    charges = charges - charges.mean()
    dipoles = 0.2 * rng.normal(size=(count, 3))
    raw = 0.1 * rng.normal(size=(count, 3, 3))
    second = 0.5 * (raw + np.swapaxes(raw, -1, -2))
    return CartesianMultipoles(charges, dipoles, second, maximum_degree=2)


@MESH_LAYOUTS
def test_multipole_mesh_energy_matches_single_device_parts(
    owners: int, layout: dict[str, Any]
) -> None:
    cell = PeriodicCell(TRICLINIC)
    count = 6
    positions = _configuration(23, count)
    system = _system(np.zeros(count), cell)
    electrostatics = MultipoleElectrostaticsPlan(
        MultipoleElectrostaticsDefinition("point", 2, coulomb_constant=1.0),
        ParticleMeshMultipoleRoute(
            1.1, 2.8, ReciprocalMeshPlan((12, 12, 12), spline_degree=4)
        ),
    ).prepare(count, cell=cell, particles=system.particle_discretization())
    moments = _multipole_moments(count)
    ownership = _ownership(owners)
    partition = FractionalOwnerPartition(cell, (owners, 1, 1))
    blocks = _Blocks(_partition_owners(partition, positions), owners, LOCAL)
    prepared = prepare_reciprocal_distribution(
        AtomisticReciprocalDistributionPlan(electrostatics, reuse_reach=0.4, **layout),
        ownership,
        partition,
    )
    blocked = jax.tree.map(
        lambda leaf: jnp.asarray(blocks.block(np.asarray(leaf))), moments
    )
    payload = ReciprocalSitePayload(moments=blocked)
    mapped = _energy_function(prepared, ownership, _specs(ownership.axis_name, payload))
    vectors = jnp.asarray(TRICLINIC)

    def reference(p: Any) -> jax.Array:
        sites = MultipoleSiteSet(moments, active=jnp.ones((count,), dtype=bool))
        parts = electrostatics.bilinear(
            p, sites, sites, cell_vectors=vectors, symmetric=True
        )
        return 0.5 * (parts.reciprocal + parts.intrasite + parts.background)

    def distributed(position: Any) -> jax.Array:
        return jnp.sum(mapped(position, payload, jnp.asarray(blocks.mask()), vectors)[0])

    x = jnp.asarray(blocks.block(positions))
    np.testing.assert_allclose(
        float(distributed(x)), float(reference(jnp.asarray(positions))), rtol=1e-10
    )
    np.testing.assert_allclose(
        blocks.unblock(jax.grad(distributed)(x)),
        jax.grad(reference)(jnp.asarray(positions)),
        rtol=1e-8,
        atol=1e-11,
    )


# --- owner-local polarization -------------------------------------------------------------


def _image_routes(
    positions: np.ndarray, vectors: np.ndarray, cutoff: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Brute-force directed routes ``(receiver, source, n)``: ``|x_r - x_s + n H| < rc``."""
    reach = 3
    grid = np.stack(
        np.meshgrid(*(np.arange(-reach, reach + 1),) * 3, indexing="ij"), axis=-1
    ).reshape((-1, 3))
    receivers, sources, shifts = [], [], []
    count = positions.shape[0]
    for receiver in range(count):
        for source in range(count):
            displacement = positions[receiver] - positions[source] + grid @ vectors
            near = np.linalg.norm(displacement, axis=1) < cutoff
            if receiver == source:
                near &= np.any(grid != 0, axis=1)
            for shift in grid[near]:
                receivers.append(receiver)
                sources.append(source)
                shifts.append(shift)
    return (
        np.asarray(receivers),
        np.asarray(sources),
        np.asarray(shifts, dtype=np.float64),
    )


def _relation(
    ownership: DistributedOwnershipPlan,
    blocks: _Blocks,
    routes: tuple[np.ndarray, np.ndarray, np.ndarray],
    capacity: int,
    *,
    weights: np.ndarray | None = None,
) -> OwnerMultipoleRelation:
    receivers, sources, shifts = routes
    owners = ownership.owner_count
    receiver = np.zeros((owners, capacity), dtype=np.int32)
    source_owner = np.zeros((owners, capacity), dtype=np.int32)
    source_slot = np.zeros((owners, capacity), dtype=np.int32)
    shift = np.zeros((owners, capacity, 3))
    weight = np.ones((owners, capacity, 3))
    valid = np.zeros((owners, capacity), dtype=np.bool_)
    load = np.zeros((owners,), dtype=np.int64)
    for index, (r, s) in enumerate(zip(receivers, sources, strict=True)):
        owner = blocks.owner[r]
        position = load[owner]
        load[owner] += 1
        receiver[owner, position] = blocks.slot[r]
        source_owner[owner, position] = blocks.owner[s]
        source_slot[owner, position] = blocks.slot[s]
        shift[owner, position] = shifts[index]
        if weights is not None:
            weight[owner, position] = weights[index]
        valid[owner, position] = True
    return OwnerMultipoleRelation.from_routes(
        ownership,
        receiver.reshape((-1,)),
        source_owner.reshape((-1,)),
        source_slot.reshape((-1,)),
        shift.reshape((-1, 3)),
        valid.reshape((-1,)),
        halo_capacity=LOCAL,
        weights=weight.reshape((-1, 3)),
    )


def _polarization_sites(
    blocks: _Blocks,
    multipoles: PermanentMultipoleSiteData,
    width_classes: np.ndarray | None = None,
) -> OwnerPolarizationSites:
    moments = multipoles.moments(jnp.dtype(jnp.float64))
    return OwnerPolarizationSites(
        jax.tree.map(lambda leaf: jnp.asarray(blocks.block(np.asarray(leaf))), moments),
        jnp.asarray(blocks.block(np.asarray(multipoles.polarizabilities))),
        jnp.asarray(blocks.block(np.asarray(multipoles.damping), fill=1.0)),
        None
        if width_classes is None
        else jnp.asarray(blocks.block(np.asarray(width_classes, dtype=np.int32))),
    )


def _solve_distributed(
    prepared: Any,
    ownership: DistributedOwnershipPlan,
    blocks: _Blocks,
    sites: OwnerPolarizationSites,
    relation: OwnerMultipoleRelation,
    periodic: bool,
    *,
    corrections: OwnerMultipoleRelation | None = None,
) -> Callable[..., Any]:
    """Owner-mapped ``(shares, induced, converged, stationary, successful, ritz)``."""
    axis = ownership.axis_name
    row = PartitionSpec(axis)
    site_specs = jax.tree.map(lambda _: row, sites)
    correction_specs = None if corrections is None else corrections.route_specs()

    def region(
        position: Any, site: Any, mask: Any, routes: Any, fixes: Any, lattice: Any
    ) -> tuple[Any, ...]:
        result = solve_owner_polarization(
            prepared,
            position,
            site,
            mask,
            lattice if periodic else None,
            axis,
            OwnerPolarizationRelations(relation, routes, corrections, fixes),
        )
        state = result.state
        return (
            result.energy_share[None],
            result.induced_dipoles,
            state.converged[None],
            state.stationary[None],
            result.successful[None],
            result.stability.minimum_ritz_value[None],
        )

    mapped = ownership.map(
        region,
        (row, site_specs, row, relation.route_specs(), correction_specs, PartitionSpec()),
        (row, row, row, row, row, row),
    )

    def run(position: Any, lattice: Any) -> tuple[Any, ...]:
        return mapped(
            position,
            sites,
            jnp.asarray(blocks.mask()),
            relation.routes,
            None if corrections is None else corrections.routes,
            lattice,
        )

    return run


PERIODIC = np.asarray([[7.0, 0.0, 0.0], [0.8, 7.2, 0.0], [0.4, 0.9, 7.5]])


def _periodic_multipoles(count: int) -> PermanentMultipoleSiteData:
    rng = np.random.default_rng(37)
    charges = rng.normal(size=count)
    charges -= charges.mean()
    return PermanentMultipoleSiteData(
        charges,
        np.zeros((count, 3)),
        np.zeros((count, 3, 3)),
        rng.uniform(0.1, 0.3, size=count),
        np.full((count,), 4.0),
    )


def _assert_periodic_polarization(
    solver: Any,
    positions: np.ndarray,
    multipoles: PermanentMultipoleSiteData,
    *,
    width_classes: np.ndarray | None = None,
    scaling: PolarizationScaleData | None = None,
    owners: int = 2,
) -> None:
    """Distributed PCG equals the canonical single-device solve: u, E, forces, status."""
    cell = PeriodicCell(PERIODIC)
    reference = evaluate_prepared_polarization(
        solver, jnp.asarray(positions), cell_vectors=jnp.asarray(PERIODIC)
    )
    assert bool(reference.successful)
    ownership = _ownership(owners)
    partition = FractionalOwnerPartition(cell, (owners, 1, 1))
    blocks = _Blocks(_partition_owners(partition, positions), owners, LOCAL)
    prepared = prepare_owner_polarization(
        OwnerPolarizationPlan(solver, reuse_reach=0.4), ownership, partition
    )
    relation = _relation(ownership, blocks, _image_routes(positions, PERIODIC, 3.0), 256)
    corrections = (
        None
        if scaling is None
        else OwnerMultipoleRelation.scaled_pairs(
            ownership,
            scaling.scales,
            blocks.owner,
            blocks.slot,
            cell,
            route_capacity=8,
            halo_capacity=LOCAL,
        )
    )
    run = _solve_distributed(
        prepared,
        ownership,
        blocks,
        _polarization_sites(blocks, multipoles, width_classes),
        relation,
        True,
        corrections=corrections,
    )
    x = jnp.asarray(blocks.block(positions))
    vectors = jnp.asarray(PERIODIC)
    shares, induced, converged, stationary, successful, _ = run(x, vectors)
    assert bool(jnp.all(converged & stationary & successful))
    np.testing.assert_allclose(
        blocks.unblock(induced), reference.state.induced_dipoles, rtol=1e-8, atol=1e-11
    )
    np.testing.assert_allclose(
        float(jnp.sum(shares)), float(reference.energy), rtol=1e-10
    )
    gradient, cell_gradient = jax.grad(
        lambda p, h: jnp.sum(run(p, h)[0]), argnums=(0, 1)
    )(x, vectors)
    np.testing.assert_allclose(
        -blocks.unblock(gradient), reference.forces, rtol=1e-7, atol=1e-10
    )
    reference_cell_gradient = reference.cell_gradient
    if reference_cell_gradient is None:
        pytest.fail("A periodic reference evaluation reports its cell gradient.")
    np.testing.assert_allclose(
        cell_gradient, reference_cell_gradient, rtol=1e-7, atol=1e-10
    )


def _mesh_route() -> ParticleMeshMultipoleRoute:
    return ParticleMeshMultipoleRoute(
        1.0, 3.0, ReciprocalMeshPlan((12, 12, 12), spline_degree=4)
    )


def test_owner_polarization_pcg_matches_single_device_with_global_status() -> None:
    count = 5
    positions = _configuration(31, count, PERIODIC)
    multipoles = _periodic_multipoles(count)
    system = _system(np.zeros(count), PeriodicCell(PERIODIC))
    solver = PolarizationPlan(
        coulomb_constant=1.0,
        route=_mesh_route(),
        tolerance=1e-12,
        maximum_iterations=60,
        stability=PolarizationStabilityPolicy("assumed-positive-definite"),
    ).prepare(multipoles, cell=system.cell, particles=system.particle_discretization())
    _assert_periodic_polarization(solver, positions, multipoles)


def test_owner_polarization_gaussian_sites_match_single_device() -> None:
    """Host width-class table, owner-row class ids: smeared real space and mesh."""
    count = 5
    positions = _configuration(41, count, PERIODIC)
    multipoles = _periodic_multipoles(count)
    classes = np.asarray([0, 1, 0, 1, 1])
    widths = GaussianMultipoleWidths(jnp.asarray([0.35, 0.6]), classes)
    system = _system(np.zeros(count), PeriodicCell(PERIODIC))
    solver = PolarizationPlan(
        coulomb_constant=1.0,
        route=_mesh_route(),
        site_kind="gaussian",
        damping="none",
        tolerance=1e-12,
        maximum_iterations=60,
        stability=PolarizationStabilityPolicy("assumed-positive-definite"),
    ).prepare(
        multipoles,
        widths=widths,
        cell=system.cell,
        particles=system.particle_discretization(),
    )
    _assert_periodic_polarization(solver, positions, multipoles, width_classes=classes)


def test_owner_polarization_scaled_pairs_match_single_device() -> None:
    """Listed pairs act through the owner-blocked minimum-image correction topology."""
    count = 5
    positions = _configuration(43, count, PERIODIC)
    multipoles = _periodic_multipoles(count)
    # The two pairs cross the owner boundary in at least one direction.
    scaling = PolarizationScaleData(
        np.asarray([[0, 3], [1, 2]]),
        np.asarray([0.0, 0.5]),
        np.asarray([0.3, 0.5]),
        np.asarray([0.6, 0.8]),
        site_capacity=count,
    )
    system = _system(np.zeros(count), PeriodicCell(PERIODIC))
    solver = PolarizationPlan(
        coulomb_constant=1.0,
        route=_mesh_route(),
        tolerance=1e-12,
        maximum_iterations=60,
        stability=PolarizationStabilityPolicy("assumed-positive-definite"),
    ).prepare(
        multipoles,
        scaling=scaling,
        cell=system.cell,
        particles=system.particle_discretization(),
    )
    _assert_periodic_polarization(solver, positions, multipoles, scaling=scaling)
    with pytest.raises(ReciprocalDistributionCapacityError, match="route_capacity"):
        OwnerMultipoleRelation.scaled_pairs(
            _ownership(2),
            scaling.scales,
            np.zeros((count,), dtype=np.int64),
            np.arange(count),
            PeriodicCell(PERIODIC),
            route_capacity=3,
            halo_capacity=LOCAL,
        )


def _finite_case(
    kind: PolarizationSolverKind,
) -> tuple[Any, np.ndarray, PermanentMultipoleSiteData]:
    count = 4
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [2.1, 0.2, 0.0], [0.4, 2.3, 0.5], [1.8, 1.9, 2.2]]
    )
    multipoles = PermanentMultipoleSiteData(
        np.asarray([0.6, -0.4, -0.3, 0.1]),
        np.zeros((count, 3)),
        np.zeros((count, 3, 3)),
        np.asarray([0.2, 0.15, 0.25, 0.1]),
        np.full((count,), 2.0),
    )
    solver = PolarizationPlan(
        coulomb_constant=1.0,
        route=FiniteMultipoleRoute(),
        tolerance=1e-12,
        maximum_iterations=40,
        solver_kind=kind,
        tcg_order=3,
    ).prepare(multipoles)
    return solver, positions, multipoles


@pytest.mark.parametrize("kind", [PolarizationSolverKind.PCG, PolarizationSolverKind.TCG])
def test_owner_polarization_finite_route_with_gershgorin_matches_single_device(
    kind: PolarizationSolverKind,
) -> None:
    owners = 2
    solver, positions, multipoles = _finite_case(kind)
    reference = evaluate_prepared_polarization(solver, jnp.asarray(positions))
    assert bool(reference.successful)
    ownership = _ownership(owners)
    partition = FractionalOwnerPartition(PeriodicCell(10.0 * np.eye(3)), (owners, 1, 1))
    blocks = _Blocks(np.asarray([0, 1, 1, 0]), owners, LOCAL)
    prepared = prepare_owner_polarization(
        OwnerPolarizationPlan(solver, reuse_reach=0.0), ownership, partition
    )
    count = positions.shape[0]
    pairs = [(r, s) for r in range(count) for s in range(count) if r != s]
    routes = (
        np.asarray([pair[0] for pair in pairs]),
        np.asarray([pair[1] for pair in pairs]),
        np.zeros((len(pairs), 3)),
    )
    relation = _relation(ownership, blocks, routes, 16)
    run = _solve_distributed(
        prepared,
        ownership,
        blocks,
        _polarization_sites(blocks, multipoles),
        relation,
        False,
    )
    x = jnp.asarray(blocks.block(positions))
    shares, induced, converged, _, successful, _ = run(x, jnp.eye(3))
    assert bool(jnp.all(successful))
    if kind is PolarizationSolverKind.TCG:
        assert not bool(jnp.any(converged))  # fixed-order TCG never reports convergence
    else:
        assert bool(jnp.all(converged))
    np.testing.assert_allclose(
        blocks.unblock(induced), reference.state.induced_dipoles, rtol=1e-8, atol=1e-11
    )
    np.testing.assert_allclose(
        float(jnp.sum(shares)), float(reference.energy), rtol=1e-10
    )
    gradient = jax.grad(lambda p: jnp.sum(run(p, jnp.eye(3))[0]))(x)
    np.testing.assert_allclose(
        -blocks.unblock(gradient), reference.forces, rtol=1e-7, atol=1e-10
    )


def test_distributed_lanczos_probe_matches_the_dense_single_device_spectrum() -> None:
    """Full Krylov depth on a small finite system: the Ritz value is lambda_min(S)."""
    owners = 2
    _, positions, multipoles = _finite_case(PolarizationSolverKind.PCG)
    count = positions.shape[0]
    dense = PolarizationPlan(
        coulomb_constant=1.0,
        route=FiniteMultipoleRoute(),
        tolerance=1e-12,
        maximum_iterations=40,
        stability=PolarizationStabilityPolicy("dense-spectrum"),
    ).prepare(multipoles)
    expected = float(dense.solve(jnp.asarray(positions)).state.stability_lower_bound)
    probed = PolarizationPlan(
        coulomb_constant=1.0,
        route=FiniteMultipoleRoute(),
        tolerance=1e-12,
        maximum_iterations=40,
        curvature_probe=PolarizationCurvatureProbe(
            subspace_dimension=3 * count, maximum_steps=3 * count
        ),
        stability=PolarizationStabilityPolicy("assumed-positive-definite"),
    ).prepare(multipoles)
    ownership = _ownership(owners)
    partition = FractionalOwnerPartition(PeriodicCell(10.0 * np.eye(3)), (owners, 1, 1))
    blocks = _Blocks(np.asarray([0, 1, 1, 0]), owners, LOCAL)
    prepared = prepare_owner_polarization(
        OwnerPolarizationPlan(probed, reuse_reach=0.0), ownership, partition
    )
    pairs = [(r, s) for r in range(count) for s in range(count) if r != s]
    routes = (
        np.asarray([pair[0] for pair in pairs]),
        np.asarray([pair[1] for pair in pairs]),
        np.zeros((len(pairs), 3)),
    )
    run = _solve_distributed(
        prepared,
        ownership,
        blocks,
        _polarization_sites(blocks, multipoles),
        _relation(ownership, blocks, routes, 16),
        False,
    )
    *_, successful, ritz = run(jnp.asarray(blocks.block(positions)), jnp.eye(3))
    assert bool(jnp.all(successful))
    # Only the active rows span the Krylov space (3N = 12 = steps): exact.
    np.testing.assert_allclose(np.asarray(ritz), expected, rtol=1e-8, atol=1e-10)


def test_distributed_polarization_refuses_the_dense_spectrum_certificate() -> None:
    _, _, multipoles = _finite_case(PolarizationSolverKind.PCG)
    dense = PolarizationPlan(
        coulomb_constant=1.0,
        route=FiniteMultipoleRoute(),
        stability=PolarizationStabilityPolicy("dense-spectrum"),
    ).prepare(multipoles)
    with pytest.raises(ValueError, match="global"):
        OwnerPolarizationPlan(dense, reuse_reach=0.0)


# --- end to end: MACELES through the domain runtime --------------------------------------


def _maceles(count: int) -> MACELESPotential:
    architecture = MACEArchitecture(
        species=domain_support.SPECIES,
        cutoff=domain_support.CUTOFF,
        radial_basis_count=4,
        cutoff_power=5,
        channel_count=4,
        hidden_degree=1,
        edge_degree=2,
        interactions=("real-agnostic", "real-agnostic-residual"),
        correlations=(2, 2),
        radial_widths=(8,),
        readout_width=4,
        average_neighbor_count=8.0,
        energy_scaling="scale-shift",
    )
    mace = MACEPotential(
        domain_support.SCALE,
        architecture,
        atomic_energies=np.asarray([[-0.3, -1.1]]),
        energy_scale=np.asarray([1.3]),
        energy_shift=np.asarray([0.2]),
        key=jax.random.key(3),
    )
    definition = LatentEwaldDefinition()
    longest = float(np.max(np.linalg.norm(domain_support.CELL, axis=1)))
    return MACELESPotential.initialize(
        mace,
        definition,
        atom_capacity=count,
        resources=LatentEwaldResources(mode_extents=definition.covering_extents(longest)),
        key=jax.random.key(5),
    )


def _maceles_reference(
    model: MACELESPotential,
    positions: Any,
    species_ids: np.ndarray,
    strain: Any,
    routes: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> jax.Array:
    """Single-device MACELES: global MACE node states plus the prepared latent Ewald."""
    from phydrax.sparse import EdgeRelation

    senders, receivers, shifts = routes
    count = positions.shape[0]
    deformation = jnp.eye(3) + strain
    deformed = positions @ deformation.T
    lattice = jnp.asarray(domain_support.CELL) @ deformation.T
    relation = domain_support.streaming(model.mace).prepare(
        EdgeRelation(senders, receivers, source_size=count, target_size=count),
        owner_id="maceles-oracle",
    )
    vectors = (
        deformed[receivers] - deformed[senders] + shifts.astype(np.float64) @ lattice
    )
    mask = jnp.ones((count,), dtype=jnp.bool_)
    species = model.species_indices(jnp.asarray(species_ids), mask)
    states = model.mace.node_states(species, mask, relation, vectors)
    charges = model.latent_charges(species, states)
    latent = model.latent_ewald.energy(
        charges, deformed, mask, lattice[None], jnp.ones((1, 3), dtype=jnp.bool_)
    )
    return jnp.sum(states.atom_energy) + latent.energy[0]


@pytest.mark.parametrize("owners", [2, 3])
def test_maceles_domain_runtime_matches_single_device_energy_forces_and_strain(
    owners: int,
) -> None:
    """AtomisticDomainPlan(reciprocal=LATENT_EWALD) + the layered MACELES epilogue."""
    positions, species_ids = domain_support.structure()
    count = positions.shape[0]
    model = _maceles(count)
    cell = PeriodicCell(domain_support.CELL)
    plan = AtomisticDomainPlan(
        domain_support.ownership(owners),
        FractionalOwnerPartition(cell, (owners, 1, 1)),
        (domain_support.role_plan(model),),
        migration_capacity=8,
        message_capacity_bytes=1 << 20,
        reduction=AtomisticDomainReductionPolicy("deterministic"),
        reciprocal=AtomisticReciprocalDistributionPlan(
            model.latent_ewald, reuse_reach=domain_support.SKIN
        ),
    )
    runtime = prepare_atomistic_domain(plan, model)
    assert runtime.reciprocal is not None
    assert runtime.reciprocal.kind is AtomisticReciprocalKind.LATENT_EWALD
    # Ingress needs only species indices, which MACELES delegates to its MACE model.
    state = domain_support.ingress(runtime, model.mace, positions, species_ids)
    evaluation = runtime.evaluate(state)
    assert bool(evaluation.successful)
    routes = domain_support.image_routes(positions, domain_support.CELL)
    zero = jnp.zeros((3, 3))
    reference = jnp.asarray(positions)
    energy = _maceles_reference(model, reference, species_ids, zero, routes)
    forces = -jax.grad(_maceles_reference, argnums=1)(
        model, reference, species_ids, zero, routes
    )
    strain = jax.grad(_maceles_reference, argnums=3)(
        model, reference, species_ids, zero, routes
    )
    np.testing.assert_allclose(float(evaluation.energy), float(energy), rtol=1e-10)
    np.testing.assert_allclose(
        np.asarray(state.layout.collect(evaluation.forces)), forces, rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(evaluation.strain_derivative), strain, rtol=1e-8, atol=1e-10
    )
