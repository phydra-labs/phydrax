import hashlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


@pytest.mark.parametrize("cells,groups", [(1, 1), (1, 3), (2, 2), (7, 3)])
def test_multigroup_sparse_routes_match_dense_reference(cells: int, groups: int) -> None:
    faces = np.cumsum(np.r_[0.0, np.linspace(0.3, 1.2, cells)])
    volumes = np.diff(faces) * np.linspace(0.8, 1.3, cells)
    areas = np.linspace(0.7, 1.4, cells + 1)
    geometry = phx.discretization.finite_volume.MetricLinePlan(
        faces, volumes, areas, "nonuniform-reference"
    )
    energy = phx.nuclear.EnergyGroupStructure(
        np.arange(groups + 1, dtype=np.float64),
        phx.units.MEGAELECTRONVOLT,
        source_id="reference-groups",
    )
    rng = np.random.default_rng(83)
    diffusion = rng.uniform(0.2, 2.0, (cells, groups))
    removal = rng.uniform(0.1, 0.3, (cells, groups))
    scattering = rng.uniform(0.01, 0.08, (cells, groups, groups))
    scattering[:, np.arange(groups), np.arange(groups)] = 0.0
    scattering[::2, 0, :] = 0.0
    nu_fission = rng.uniform(0.1, 0.4, (cells, groups))
    nu_fission[::2, :] = 0.0
    spectrum = np.full((cells, groups), 1.0 / groups, dtype=np.float64)
    spectrum[::2, :] = 0.0
    if groups > 1:
        spectrum[1::2, 0] = 0.0
        spectrum[1::2, 1:] = 1.0 / (groups - 1)
    material = phx.applications.reactor_physics.MultigroupMaterialData(
        energy,
        diffusion,
        removal,
        scattering,
        nu_fission,
        spectrum,
        _data(),
        "reference-material",
    )
    plan = phx.applications.reactor_physics.MultigroupDiffusionPlan(geometry, material)
    prepared = plan.prepare()
    centers = np.asarray(prepared.geometry.coordinate_cells)
    loss = np.zeros((cells * groups, cells * groups), dtype=np.float64)
    fission = np.zeros_like(loss)
    for c in range(cells):
        for g in range(groups):
            row = c * groups + g
            diagonal = 0.0
            for neighbor, face in ((c - 1, c), (c + 1, c + 1)):
                if 0 <= neighbor < cells:
                    coefficient = (
                        2.0
                        * diffusion[min(c, neighbor), g]
                        * diffusion[max(c, neighbor), g]
                        / (
                            diffusion[min(c, neighbor), g]
                            + diffusion[max(c, neighbor), g]
                        )
                    )
                    distance = abs(centers[c] - centers[neighbor])
                else:
                    coefficient = diffusion[c, g]
                    distance = abs(centers[c] - faces[face])
                conductance = areas[face] * coefficient / distance
                diagonal += conductance
                if 0 <= neighbor < cells:
                    loss[row, neighbor * groups + g] = -conductance
            loss[row, row] = diagonal + removal[c, g] * volumes[c]
            for s in range(groups):
                if s != g:
                    loss[row, c * groups + s] = -scattering[c, s, g] * volumes[c]
                fission[row, c * groups + s] = (
                    spectrum[c, g] * nu_fission[c, s] * volumes[c]
                )
    from phydrax._fingerprint import canonical_fingerprint

    expected_plan_id = canonical_fingerprint(
        {
            "kind": "multigroup-diffusion-plan",
            "geometry": geometry.plan_id,
            "material": material.data_id,
            "boundary": "zero-dirichlet-vacuum",
            "relative_tolerance": plan.relative_tolerance,
            "maximum_linear_steps": plan.maximum_linear_steps,
        }
    )
    assert prepared.plan_id == expected_plan_id
    assert prepared.loss_operator.operator_id == expected_plan_id
    assert prepared.fission_operator.operator_id == canonical_fingerprint(
        {"kind": "reactor-fission-operator", "plan": expected_plan_id}
    )
    vector = rng.uniform(0.1, 1.0, cells * groups)
    for operator, dense in (
        (prepared.loss_operator, loss),
        (prepared.fission_operator, fission),
    ):
        rows, columns = np.nonzero(dense)
        assert isinstance(operator.relation, phx.sparse.EdgeRelation)
        np.testing.assert_array_equal(operator.relation.target_indices, rows)
        np.testing.assert_array_equal(operator.relation.source_indices, columns)
        np.testing.assert_array_equal(operator.coefficients, dense[rows, columns])
        np.testing.assert_allclose(
            operator.mv(jnp.asarray(vector)), dense @ vector, rtol=2e-15, atol=2e-15
        )
        np.testing.assert_allclose(
            operator.transpose_mv(jnp.asarray(vector)),
            dense.T @ vector,
            rtol=2e-15,
            atol=2e-15,
        )
    source = rng.uniform(0.1, 1.0, (cells, groups))
    result = prepared.solve_fixed_source(source)
    assert bool(result.successful)
    np.testing.assert_allclose(
        result.scalar_flux_m2_s.reshape(-1),
        np.linalg.solve(loss, (source * volumes[:, None]).reshape(-1)),
        rtol=2e-8,
        atol=2e-10,
    )
