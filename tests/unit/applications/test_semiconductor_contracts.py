import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.applications.semiconductor._device import (
    DevicePlan,
    GateContact,
    MaterialBinding,
    OhmicContact,
)
from phydrax.applications.semiconductor._materials import (
    DielectricMaterial,
    DopingDependentMobility,
    SemiconductorMaterial,
    srh_recombination,
)
from phydrax.applications.semiconductor._quantities import (
    PER_CUBIC_METER,
    SemiconductorQuantitySpec,
)
from phydrax.applications.semiconductor._support import TransportSupport
from phydrax.discretization import CellBlock, CellMesh
from phydrax.meshing import (
    certify_cell_mesh,
    MeshAttribute,
    MeshAttributeRole,
    MeshingEntityKind,
    MeshingScope,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
)
from phydrax.units import (
    CENTIMETER,
    derived_unit,
    METER,
    MICROMETER,
    MOLE_PER_CUBIC_METER,
    SECOND,
    VOLT,
)


def _contacts(support):
    return (
        OhmicContact("left", support.boundary_patch("left")),
        OhmicContact("right", support.boundary_patch("right", side="upper")),
    )


def _zone(support, name, ids=None):
    return MeshZone(name, MeshZoneRole.MATERIAL, support.node_scope(ids))


def test_nonuniform_interval_integrates_physical_volume_and_diffusive_flux():
    area_unit = derived_unit("um2", ((MICROMETER, 2),))
    support = TransportSupport.interval(
        [0.0, 1.0, 3.0], area=2.0, length_unit=MICROMETER, area_unit=area_unit
    )
    np.testing.assert_allclose(support.volumes, np.asarray([1.0, 3.0, 2.0]) * 1e-18)
    np.testing.assert_allclose(support.transmissibility, [2e-6, 1e-6])
    concentration = 4e20 + 2e25 * support.positions[:, 0]
    flux = (
        support.transmissibility
        * 1e-3
        * (concentration[support.tail] - concentration[support.head])
    )
    np.testing.assert_allclose(flux, [-4e10, -4e10], rtol=1e-12)
    incoming = jnp.zeros(3).at[support.tail].add(-flux).at[support.head].add(flux)
    np.testing.assert_allclose(incoming, [4e10, 0.0, -4e10], atol=1e-3)
    np.testing.assert_allclose(jnp.sum(incoming), 0.0, atol=1e-3)


@pytest.mark.parametrize(
    "axes, transverse, expected_volume",
    [
        (([0.0, 1.0, 3.0], [0.0, 2.0]), 0.2, 1.2),
        (([0.0, 1.0, 3.0], [0.0, 2.0], [0.0, 0.5, 1.0]), 1.0, 6.0),
    ],
)
def test_tensor_metrics_preserve_volume_and_affine_energy(
    axes, transverse, expected_volume
):
    support = TransportSupport.tensor_grid(axes, transverse_measure=transverse)
    gradient = np.arange(1.0, len(axes) + 1.0)
    potential = np.sum(np.asarray(support.positions) * gradient, axis=1)
    energy = np.sum(
        np.asarray(support.transmissibility)
        * (potential[support.head] - potential[support.tail]) ** 2
    )
    np.testing.assert_allclose(np.sum(support.volumes), expected_volume)
    np.testing.assert_allclose(energy, expected_volume * np.sum(gradient**2))


@pytest.mark.parametrize(
    "kind, points, volume",
    [
        ("interval", [[0.0], [2.0]], 2.0),
        ("triangle", [[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]], np.sqrt(3) / 4),
        (
            "tetrahedron",
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, np.sqrt(3) / 2, 0.0],
                [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)],
            ],
            np.sqrt(2) / 12,
        ),
    ],
)
def test_native_affine_simplex_preserves_linear_energy_and_persistent_identity(
    kind, points, volume
):
    count = len(points)
    mesh = CellMesh(
        points,
        (CellBlock("bulk", kind, [list(range(count))]),),
        vertex_global_ids=np.arange(count) * 7 + 13,
        numeric_version="revision-7",
    )
    result = certify_cell_mesh(mesh, SpatialCoordinateContract.si())
    support = TransportSupport.from_meshing(result, transverse_measure=0.25)
    gradient = np.arange(1.0, len(points[0]) + 1)
    values = np.sum(np.asarray(support.positions) * gradient, axis=1)
    energy = np.sum(
        np.asarray(support.transmissibility)
        * (values[support.head] - values[support.tail]) ** 2
    )
    np.testing.assert_allclose(np.sum(support.volumes), volume * 0.25)
    np.testing.assert_allclose(energy, volume * 0.25 * np.sum(gradient**2))
    np.testing.assert_array_equal(support.node_ids, result.mesh.vertex_global_ids)
    material_scope = MeshingScope(
        result.mesh.mesh_id,
        result.mesh.numeric_version,
        MeshingEntityKind.MESH,
        result.mesh.topological_dimension,
        result.mesh.entity_set(result.mesh.topological_dimension).entity_set_id,
        result.mesh.entity_set(result.mesh.topological_dimension).entity_ids,
    )
    material = MaterialBinding(
        MeshZone("bulk", MeshZoneRole.MATERIAL, material_scope),
        SemiconductorMaterial.silicon(),
    )
    contact = OhmicContact(
        "reservoir", MeshPatch("vertex", support.node_scope(support.node_ids[:1]))
    )
    plan = DevicePlan(support, materials=(material,), contacts=(contact,))
    np.testing.assert_array_equal(plan.semiconductor_mask, np.ones(count, dtype=bool))
    stale_scope = MeshingScope(
        support.source_id,
        "revision-8",
        MeshingEntityKind.MESH,
        0,
        support.entity_set_ids[0],
        support.node_ids[:1],
    )
    with pytest.raises(ValueError):
        support.resolve_scope(stale_scope)


def test_native_obtuse_simplex_rejects_negative_two_point_metric():
    mesh = CellMesh.from_triangles([[0.0, 0.0], [1.0, 0.0], [0.2, 0.1]], [[0, 1, 2]])
    result = certify_cell_mesh(mesh, SpatialCoordinateContract.si())
    with pytest.raises(ValueError):
        TransportSupport.from_meshing(result)


@pytest.mark.parametrize(
    "coordinates, unit", [([0.0, 0.0, 1.0], METER), ([0.0, 1.0], SECOND)]
)
def test_support_rejects_collapsed_axis_and_dimensionally_wrong_coordinates(
    coordinates, unit
):
    with pytest.raises(ValueError):
        TransportSupport.interval(coordinates, length_unit=unit)


def test_native_dopant_units_and_neutrality_use_number_not_molar_density():
    support = TransportSupport.interval(np.linspace(0.0, 1e-6, 5))
    per_cm3 = derived_unit("1/cm3", ((CENTIMETER, -3),))
    donors = MeshAttribute(
        "donors",
        MeshAttributeRole.USER,
        support.node_scope(),
        np.full(5, 1e15),
        unit=per_cm3,
    )
    plan = DevicePlan(
        support,
        materials=(
            MaterialBinding(_zone(support, "Si"), SemiconductorMaterial.silicon()),
        ),
        donor_attributes=(donors,),
        contacts=_contacts(support),
    )
    coordinates = plan.equilibrium_coordinates()
    n = plan.intrinsic_density * jnp.exp(coordinates[:, 0] + coordinates[:, 1])
    p = plan.intrinsic_density * jnp.exp(-coordinates[:, 0] - coordinates[:, 2])
    np.testing.assert_allclose(n - p, 1e21, rtol=1e-12)
    np.testing.assert_allclose(n * p, plan.intrinsic_density**2, rtol=1e-12)
    molar = MeshAttribute(
        "wrong",
        MeshAttributeRole.USER,
        support.node_scope(),
        np.ones(5),
        unit=MOLE_PER_CUBIC_METER,
    )
    with pytest.raises(ValueError):
        DevicePlan(
            support,
            SemiconductorMaterial.silicon(),
            donor_attributes=(molar,),
            contacts=_contacts(support),
        )
    with pytest.raises(ValueError):
        SemiconductorQuantitySpec("donors", "number_density", MOLE_PER_CUBIC_METER)


def test_dopant_binding_rejects_stale_revision_and_overlapping_attributes():
    support = TransportSupport.interval([0.0, 1.0, 2.0])
    stale = MeshingScope(
        support.source_id,
        "other",
        MeshingEntityKind.MESH,
        0,
        support.entity_set_ids[0],
        support.node_ids,
    )
    stale_attribute = MeshAttribute(
        "donor", MeshAttributeRole.USER, stale, np.ones(3), unit=PER_CUBIC_METER
    )
    with pytest.raises(ValueError):
        DevicePlan(
            support,
            SemiconductorMaterial.silicon(),
            donor_attributes=(stale_attribute,),
            contacts=_contacts(support),
        )
    attribute = MeshAttribute(
        "donor",
        MeshAttributeRole.USER,
        support.node_scope(),
        np.ones(3),
        unit=PER_CUBIC_METER,
    )
    with pytest.raises(ValueError):
        DevicePlan(
            support,
            SemiconductorMaterial.silicon(),
            donor_attributes=(attribute, attribute),
            contacts=_contacts(support),
        )


def test_material_and_contact_admission_is_exclusive_complete_and_boundary_only():
    support = TransportSupport.interval([0.0, 1.0, 2.0])
    silicon = SemiconductorMaterial.silicon()
    first = MaterialBinding(_zone(support, "first", [0, 1]), silicon)
    overlap = MaterialBinding(_zone(support, "second", [1, 2]), silicon)
    with pytest.raises(ValueError):
        DevicePlan(support, materials=(first,), contacts=_contacts(support))
    with pytest.raises(ValueError):
        DevicePlan(support, materials=(first, overlap), contacts=_contacts(support))
    interior = OhmicContact("interior", MeshPatch("interior", support.node_scope([1])))
    with pytest.raises(ValueError):
        DevicePlan(support, silicon, contacts=(interior,))
    same_patch = support.boundary_patch("left")
    with pytest.raises(ValueError):
        DevicePlan(
            support,
            silicon,
            contacts=(OhmicContact("one", same_patch), OhmicContact("two", same_patch)),
        )
    with pytest.raises(ValueError):
        DevicePlan(support, silicon, contacts=())


def test_oxide_owns_no_carriers_and_gate_keeps_explicit_reference():
    support = TransportSupport.interval(np.linspace(0.0, 1e-6, 5))
    silicon, oxide = (
        SemiconductorMaterial.silicon(),
        DielectricMaterial("oxide", permittivity=3.45e-11, provenance="test dielectric"),
    )
    contacts = (
        OhmicContact("bulk", support.boundary_patch("bulk")),
        GateContact(
            "gate",
            support.boundary_patch("gate", side="upper"),
            potential_offset=0.2,
            voltage_unit=VOLT,
        ),
    )
    plan = DevicePlan(
        support,
        materials=(
            MaterialBinding(_zone(support, "Si", [0, 1, 2]), silicon),
            MaterialBinding(_zone(support, "oxide", [3, 4]), oxide),
        ),
        donor_density=[1e21, 1e21, 1e21, 0.0, 0.0],
        contacts=contacts,
    )
    np.testing.assert_array_equal(
        plan.semiconductor_mask, [True, True, True, False, False]
    )
    np.testing.assert_array_equal(plan.electron_mobility[3:], [0.0, 0.0])
    np.testing.assert_array_equal(
        plan.equilibrium_coordinates()[3:, 1:], np.zeros((2, 2))
    )
    np.testing.assert_allclose(plan.contact_potential_offset[-1], 0.2)
    with pytest.raises(ValueError):
        DevicePlan(support, silicon, contacts=contacts)


def test_unlike_semiconductor_band_reference_is_not_silently_treated_as_homojunction():
    support = TransportSupport.interval([0.0, 1.0, 2.0])
    silicon = SemiconductorMaterial.silicon()
    different_gap = eqx.tree_at(
        lambda material: material.band_gap, silicon, silicon.band_gap * 1.1
    )
    with pytest.raises(ValueError):
        DevicePlan(
            support,
            materials=(
                MaterialBinding(_zone(support, "left", [0, 1]), silicon),
                MaterialBinding(_zone(support, "right", [2]), different_gap),
            ),
            contacts=_contacts(support),
        )
    with pytest.raises(ValueError):
        DevicePlan(support, silicon, contacts=_contacts(support), temperature=400.0)


def test_doping_reclosure_updates_mobility_and_contact_neutrality_under_jit():
    mobility = DopingDependentMobility(
        0.01, 0.15, 1e22, 1.0, provenance="test parameters"
    )
    silicon = eqx.tree_at(
        lambda material: material.electron_mobility,
        SemiconductorMaterial.silicon(),
        mobility,
    )
    support = TransportSupport.interval([0.0, 1e-6, 2e-6])
    plan = DevicePlan(support, silicon, donor_density=1e20, contacts=_contacts(support))

    @eqx.filter_jit
    def response(density):
        changed = plan.with_doping(jnp.full(3, density), jnp.zeros(3))
        seed = changed.equilibrium_coordinates()
        n = changed.intrinsic_density * jnp.exp(seed[:, 0])
        p = changed.intrinsic_density * jnp.exp(-seed[:, 0])
        return changed.electron_mobility, n - p, changed.contact_potential_offset

    mu, neutrality, offset = response(1e22)
    np.testing.assert_allclose(mu, 0.08)
    np.testing.assert_allclose(neutrality, 1e22, rtol=1e-12)
    assert float(offset[0]) > float(plan.contact_potential_offset[0])
    derivative = jax.grad(lambda density: response(density)[0][0])(1e22)
    np.testing.assert_allclose(derivative, -0.14 / (4 * 1e22), rtol=1e-12)


def test_srh_mass_action_and_small_quasi_fermi_departure_remain_resolved():
    n = jnp.asarray([1e16, 1e21])
    ni = jnp.asarray(1e16)
    p = ni**2 / n
    equilibrium_rate = srh_recombination(
        n, p, ni, 1e-6, 2e-6, log_mass_action=jnp.zeros_like(n)
    )
    np.testing.assert_array_equal(equilibrium_rate, np.zeros(2))
    rate = srh_recombination(2 * ni, 3 * ni, ni, 1e-6, 2e-6)
    np.testing.assert_allclose(rate, 5 * ni / 1e-5)
    departure = srh_recombination(ni, ni, ni, 1e-6, 2e-6, log_mass_action=1e-20)
    np.testing.assert_allclose(departure, 1e-4 / 6e-6, rtol=1e-12)
    generation = srh_recombination(0.0, 0.0, ni, 1e-6, 2e-6)
    np.testing.assert_allclose(generation, -ni / 3e-6)
