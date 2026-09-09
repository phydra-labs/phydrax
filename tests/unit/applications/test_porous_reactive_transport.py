#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.porous_media._fracture_exchange import FractureMatrixExchange
from phydrax.applications.porous_media._reactions import (
    MineralKinetics,
    reactive_transport_step,
)
from phydrax.applications.porous_media._speciation import MassActionSystem
from phydrax.applications.porous_media._transport import (
    ComponentTransport,
    TransportBoundary,
)
from phydrax.discretization import UnstructuredFiniteVolumePlan
from phydrax.discretization.finite_volume._hybrid_diffusion import HybridMimeticDiffusion
from phydrax.ein import contract
from phydrax.linalg import DenseLU, LinearSolvePolicy
from phydrax.nonlinear import NewtonKrylov, NonlinearTermination


def _geometry():
    return UnstructuredFiniteVolumePlan(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            )
        ),
        tetrahedra=np.asarray(((0, 1, 2, 3), (0, 2, 1, 4))),
    ).prepare()


def _method():
    return NewtonKrylov(linear_policy=LinearSolvePolicy(DenseLU()))


def _termination():
    return NonlinearTermination(
        absolute_residual=1e-10,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=80,
    )


def _dimer():
    return MassActionSystem(
        ("A",), ("A2",), [[2.0]], [np.log(2.0)], [0.0, 0.0], reference_concentration=1.0
    )


def test_water_volume_change_and_upstream_transport_preserve_component_moles():
    d = _geometry()
    face = int(np.flatnonzero(np.asarray(d.neighbour_cells) >= 0)[0])
    owner, neighbour = int(d.owner_cells[face]), int(d.neighbour_cells[face])
    q = jnp.zeros(d.owner_cells.size).at[face].set(0.01)
    previous = jnp.full((2, 1), 0.1).at[owner, 0].set(0.3)
    volumes = jnp.full((2,), 0.1).at[owner].add(-0.02).at[neighbour].add(0.02)
    result = ComponentTransport(d, ("A",)).step(
        previous,
        volumes,
        q,
        2.0,
        TransportBoundary(d, 1),
        method=_method(),
        termination=_termination(),
    )
    np.testing.assert_allclose(result.concentrations[owner], [3.0], atol=1e-10)
    np.testing.assert_allclose(result.concentrations[neighbour], [4.0 / 3.0], atol=1e-10)
    np.testing.assert_allclose(result.face_component_rates[face], [0.03], atol=1e-11)
    np.testing.assert_allclose(result.component_balance, 0.0, atol=1e-11)


def test_boundary_injection_zero_initial_inventory_and_outflow_use_correct_states():
    d = _geometry()
    faces = np.flatnonzero(
        (np.asarray(d.neighbour_cells) < 0) & (np.asarray(d.owner_cells) == 0)
    )
    inflow, outflow = int(faces[0]), int(faces[1])
    q = jnp.zeros(d.owner_cells.size).at[inflow].set(-0.01).at[outflow].set(0.01)
    incoming = jnp.full((d.owner_cells.size, 1), 1000.0).at[inflow, 0].set(2.0)
    source = jnp.zeros((2, 1)).at[0, 0].set(0.003)
    result = ComponentTransport(d, ("A",)).step(
        jnp.zeros((2, 1)),
        jnp.full((2,), 0.1),
        q,
        1.0,
        TransportBoundary(d, 1, inflow_concentration=incoming),
        source=source,
        method=_method(),
        termination=_termination(),
    )
    expected = 0.023 / 0.11
    np.testing.assert_allclose(result.concentrations[0, 0], expected, atol=1e-10)
    np.testing.assert_allclose(result.face_component_rates[inflow, 0], -0.02, atol=1e-12)
    np.testing.assert_allclose(
        result.face_component_rates[outflow, 0], 0.01 * expected, atol=1e-12
    )
    np.testing.assert_allclose(result.component_balance, 0.0, atol=1e-11)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="upstream"):
        ComponentTransport(d, ("A",)).step(
            jnp.zeros((2, 1)), jnp.full((2,), 0.1), q, 1.0, TransportBoundary(d, 1)
        )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="water volumes"):
        ComponentTransport(d, ("A",)).step(
            jnp.zeros((2, 1)),
            jnp.zeros(2),
            jnp.zeros_like(q),
            1.0,
            TransportBoundary(d, 1),
        )


def test_rotated_spd_hybrid_dispersion_preserves_affine_field_and_physical_flux():
    d = _geometry()
    tensor = jnp.asarray(((2.0, 0.4, 0.2), (0.4, 1.5, -0.1), (0.2, -0.1, 1.0)))
    gradient = jnp.asarray((0.3, -0.2, 0.4))
    c = 2.0 + contract("ci,i->c", d.cell_centers, gradient)
    f = 2.0 + contract("fi,i->f", d.face_centers, gradient)
    plan = ComponentTransport(
        d, ("A",), dispersion=HybridMimeticDiffusion(d), dispersion_tensor=tensor
    )
    boundary = TransportBoundary(
        d, 1, dirichlet_mask=d.neighbour_cells < 0, dispersion_concentration=f[:, None]
    )
    volumes = 0.3 * d.cell_volumes
    result = plan.step(
        volumes[:, None] * c[:, None],
        volumes,
        jnp.zeros(d.owner_cells.size),
        0.1,
        boundary,
        method=_method(),
        termination=_termination(),
    )
    np.testing.assert_allclose(result.concentrations[:, 0], c, atol=1e-9)
    expected = -contract("fi,ij,j->f", d.area_vectors, tensor, gradient)
    np.testing.assert_allclose(result.face_component_rates[:, 0], expected, atol=1e-9)
    np.testing.assert_allclose(result.component_balance, 0.0, atol=1e-10)


def test_mass_action_component_balance_and_native_forward_reverse_derivatives():
    chemistry = _dimer()

    def primary(total):
        return chemistry.solve(
            jnp.asarray([total]), initial_concentrations=jnp.asarray([0.5, 0.5])
        ).concentrations[0]

    value, derivative = jax.jvp(primary, (jnp.asarray(3.0),), (jnp.asarray(1.0),))
    reverse = jax.grad(primary)(jnp.asarray(3.0))
    expected = (np.sqrt(49.0) - 1.0) / 8.0
    np.testing.assert_allclose(value, expected, atol=1e-10)
    np.testing.assert_allclose(derivative, 1.0 / (1.0 + 8.0 * expected), atol=1e-9)
    np.testing.assert_allclose(reverse, derivative, atol=1e-10)
    result = chemistry.solve([3.0], initial_concentrations=[0.5, 0.5])
    np.testing.assert_allclose(result.concentrations[1], 2.0 * expected**2, atol=1e-10)
    np.testing.assert_allclose(result.component_residual, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.mass_action_residual, 0.0, atol=1e-10)


def test_declared_charge_replacement_exposes_open_component_and_davies_activities():
    acid = MassActionSystem(
        ("H+", "A-"),
        ("HA",),
        [[1.0, 1.0]],
        [0.0],
        [1.0, -1.0, 0.0],
        reference_concentration=1.0,
        charge_balance_component=0,
    )
    result = acid.solve([0.1, 1.0], initial_concentrations=[0.5, 0.5, 0.5])
    np.testing.assert_allclose(
        result.concentrations[:2], [(np.sqrt(5.0) - 1.0) / 2.0] * 2, atol=1e-10
    )
    np.testing.assert_allclose(result.charge_residual, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.component_residual, [0.9, 0.0], atol=1e-10)
    salt = MassActionSystem(
        ("Na+", "Cl-"), (), np.empty((0, 2)), [], [1.0, -1.0], activity_model="davies"
    )
    ions = salt.solve([10.0, 10.0], initial_concentrations=[9.0, 9.0])
    gamma = 10.0 ** (-0.509 * (0.1 / 1.1 - 0.003))
    np.testing.assert_allclose(ions.activities, [0.01 * gamma] * 2, atol=1e-10)
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="successful physical root"
    ):
        salt.solve([1000.0, 1000.0], initial_concentrations=[1000.0, 1000.0])


def test_mineral_exhaustion_and_precipitation_move_identical_stoichiometric_moles():
    aqueous = MassActionSystem(
        ("A",), (), np.empty((0, 1)), [], [0.0], reference_concentration=1.0
    )
    mineral = MineralKinetics(aqueous, ("A(s)",), [[1.0]], [np.log(100.0)], [10.0])
    result = mineral.step([0.2], [0.05], 1.0, 1.0, [1.0], initial_concentrations=[0.2])
    np.testing.assert_allclose(result.mineral_inventory, [0.0], atol=1e-12)
    np.testing.assert_allclose(result.extents, [0.05], atol=1e-12)
    np.testing.assert_allclose(result.component_inventory, [0.25], atol=1e-10)
    np.testing.assert_allclose(result.component_balance, 0.0, atol=1e-10)
    assert result.rates[0] > result.extents[0]
    precipitation = MineralKinetics(
        aqueous, ("A(s)",), [[1.0]], [0.0], [1.0], allow_nucleation=(True,)
    )
    deposited = precipitation.step(
        [2.0], [0.0], 1.0, 1.0, [1.0], initial_concentrations=[2.0]
    )
    np.testing.assert_allclose(deposited.component_inventory, [1.5], atol=1e-10)
    np.testing.assert_allclose(deposited.mineral_inventory, [0.5], atol=1e-10)
    np.testing.assert_allclose(deposited.component_balance, 0.0, atol=1e-10)
    unseeded = MineralKinetics(aqueous, ("A(s)",), [[1.0]], [0.0], [1.0])
    absent = unseeded.step([2.0], [0.0], 1.0, 1.0, [1.0], initial_concentrations=[2.0])
    np.testing.assert_allclose(absent.component_inventory, [2.0], atol=1e-10)
    np.testing.assert_allclose(absent.mineral_inventory, [0.0], atol=1e-12)


def _exchange(d):
    surface = UnstructuredFiniteVolumePlan(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        triangles=np.asarray(((0, 1, 2),)),
    ).prepare()
    return FractureMatrixExchange(
        d,
        surface,
        [0, 1],
        [0, 0],
        parent_global_ids=d.cell_global_ids,
        aperture=0.02,
        porosity=1.0,
        contact_areas=[0.5, 0.5],
        distances=[0.25, 0.25],
        origin=[0.0, 0.0, 0.0],
        tangent_axes=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        fracture_id="resolved-plane-z0",
    )


def test_fracture_storage_parent_exchange_and_water_balance_are_mixed_dimensional():
    d = _geometry()
    exchange = _exchange(d)
    np.testing.assert_allclose(exchange.water_volumes(), [0.01], atol=1e-12)
    np.testing.assert_allclose(exchange.water_volumes(0.5), [0.005], atol=1e-12)
    vm, vf = jnp.asarray([0.1, 0.1]), exchange.water_volumes(0.5)
    oldm, oldf = (
        vm[:, None] * jnp.asarray([[2.0], [1.0]]),
        vf[:, None] * jnp.asarray([[0.2]]),
    )
    q = jnp.asarray([0.001, -0.0002])
    result = exchange.step(
        oldm,
        oldf,
        vm,
        vf,
        q,
        1.0,
        conductance=exchange.diffusive_conductance(1e-4),
        method=_method(),
        termination=_termination(),
    )
    np.testing.assert_allclose(result.matrix_water_volumes, [0.099, 0.1002], atol=1e-12)
    np.testing.assert_allclose(result.fracture_water_volumes, [0.0058], atol=1e-12)
    np.testing.assert_allclose(result.component_balance, 0.0, atol=1e-11)
    np.testing.assert_allclose(result.water_balance, 0.0, atol=1e-12)
    assert result.fracture_concentrations[0, 0] > 0.2
    matrix_source, fracture_source = exchange.exchange_sources(
        result.link_component_rates
    )
    np.testing.assert_allclose(
        jnp.sum(matrix_source, axis=0) + jnp.sum(fracture_source, axis=0), 0.0, atol=1e-14
    )
    # A hydrostatic field must not leak between the 3D and embedded 2D domains.
    rho, g = 1000.0, 9.80665
    pm = 1e5 - rho * g * d.cell_centers[:, 2]
    pf = 1e5 - rho * g * exchange.fracture_centers()[:, 2]
    np.testing.assert_allclose(
        exchange.hydraulic_rates(pm, pf, permeability=1e-12, viscosity=1e-3),
        0.0,
        atol=1e-13,
    )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="overfill"):
        exchange.step(oldm, oldf, vm, vf, jnp.asarray([0.01, 0.01]), 1.0)


def test_reactive_split_transports_totals_then_solves_real_chemistry_and_solids():
    d, chemistry = _geometry(), _dimer()
    kinetics = MineralKinetics(chemistry, ("A(s)",), [[1.0]], [np.log(100.0)], [10.0])
    transport = ComponentTransport(d, chemistry.primary_names)
    volumes = jnp.full((2,), 0.1)
    initial_species = jnp.asarray([[0.5, 0.5], [0.5, 0.5]])
    old = volumes[:, None] * chemistry.component_totals(initial_species)
    result = reactive_transport_step(
        transport,
        kinetics,
        old,
        volumes,
        jnp.zeros(d.owner_cells.size),
        1.0,
        TransportBoundary(d, 1),
        initial_species_concentrations=initial_species,
        previous_mineral_inventory=jnp.full((2, 1), 0.01),
        reactive_area=jnp.ones((2, 1)),
        transport_method=_method(),
        transport_termination=_termination(),
    )
    np.testing.assert_allclose(result.mineral_inventory, 0.0, atol=1e-11)
    np.testing.assert_allclose(result.component_inventory, old + 0.01, atol=1e-10)
    np.testing.assert_allclose(
        result.concentrations[:, 1], 2.0 * result.concentrations[:, 0] ** 2, atol=1e-10
    )
    np.testing.assert_allclose(result.component_balance, 0.0, atol=1e-10)
