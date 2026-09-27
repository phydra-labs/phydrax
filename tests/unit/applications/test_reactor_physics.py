import hashlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _data() -> Any:
    payload = b"reactor-data"
    reference = phx.qualification.ReferenceArtifactManifest(
        "synthetic-reactor-data",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="fixture",
        nondimensionalization={"identity": 1.0},
        uncertainty=None,
        lineage_ids=("synthetic",),
    )
    return phx.nuclear.NuclearDataProvenance(
        reference,
        "https://example.invalid/reactor",
        "synthetic",
        "release",
        "one-group",
    )


def _one_group() -> Any:
    return phx.nuclear.EnergyGroupStructure(
        # ty: ignore[invalid-argument-type]
        [0.0, 20.0],
        phx.units.MEGAELECTRONVOLT,
        source_id="one-group",
    )


def test_reactor_physics_scenario_1() -> None:
    geometry = phx.discretization.finite_volume.MetricLinePlan(
        # ty: ignore[invalid-argument-type]
        [0.0, 1.0, 2.0],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0, 1.0],
        "slab",
    )
    material = phx.applications.reactor_physics.MultigroupMaterialData(
        _one_group(),
        np.ones((2, 1)),
        np.ones((2, 1)),
        np.zeros((2, 1, 1)),
        np.zeros((2, 1)),
        np.zeros((2, 1)),
        _data(),
        "homogeneous",
    )
    prepared = phx.applications.reactor_physics.MultigroupDiffusionPlan(
        geometry, material
    ).prepare()
    result = prepared.solve_fixed_source(np.ones((2, 1)))

    assert bool(result.successful)
    assert result.residual_norm < 1.0e-9
    assert np.all(result.scalar_flux_m2_s >= 0.0)
    geometry = phx.discretization.finite_volume.MetricLinePlan(
        # ty: ignore[invalid-argument-type]
        [0.0, 1.0],
        # ty: ignore[invalid-argument-type]
        [1.0],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        "critical-slab",
    )
    material = phx.applications.reactor_physics.MultigroupMaterialData(
        _one_group(),
        np.ones((1, 1)),
        np.ones((1, 1)),
        np.zeros((1, 1, 1)),
        10.0 * np.ones((1, 1)),
        np.ones((1, 1)),
        _data(),
        "critical-material",
    )
    prepared = phx.applications.reactor_physics.MultigroupDiffusionPlan(
        geometry, material
    ).prepare()
    result = prepared.solve_criticality(
        np.ones((1, 1)), maximum_iterations=8, residual_tolerance=1.0e-10
    )

    assert bool(result.successful)
    np.testing.assert_allclose(result.k_effective, 2.0, rtol=1.0e-10)
    plan = phx.applications.reactor_physics.DelayedNeutronKineticsPlan(
        np.asarray([0.004, 0.002]),
        np.asarray([0.1, 1.0]),
        1.0e-4,
        "synthetic-delayed-data",
    )
    state = plan.equilibrium_state(1.0)
    result = plan.prepare().step(state, 0.0, 0.0, 0.1)

    assert bool(result.successful)
    np.testing.assert_allclose(result.accepted_state.neutron_population, 1.0)
    np.testing.assert_allclose(
        result.accepted_state.precursor_populations,
        state.precursor_populations,
        rtol=1.0e-12,
    )


def test_delayed_neutron_kinetics_is_differentiable_in_reactivity() -> None:
    plan = phx.applications.reactor_physics.DelayedNeutronKineticsPlan(
        np.asarray([0.006]), np.asarray([0.2]), 1.0e-4, "synthetic-delayed-data"
    )
    prepared = plan.prepare()
    state = plan.equilibrium_state(1.0)

    def population(reactivity: Any) -> Any:
        return prepared.step(
            state, reactivity, 0.0, 1.0e-3
        ).accepted_state.neutron_population

    derivative = jax.grad(population)(jnp.asarray(0.0))
    assert jnp.isfinite(derivative)
    assert derivative > 0.0
