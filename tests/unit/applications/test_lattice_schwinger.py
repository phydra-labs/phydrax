#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _model(**kwargs):
    parameters = {
        "lattice_spacing": 0.4,
        "mass": 0.3,
        "gauge_coupling": 0.8,
        "left_boundary_flux": 0.2,
        "external_flux": jnp.zeros((3,)),
    }
    parameters.update(kwargs)
    return phx.applications.lattice_field.SchwingerChainModel(4, **parameters)


def test_schwinger_charge_and_flux_share_one_encoding():
    model = _model(external_flux=jnp.asarray([0.1, -0.2, 0.05]))
    staggered_vacuum = jnp.asarray([1, 0, 1, 0], dtype=jnp.int32)
    observables = phx.applications.lattice_field.schwinger_observables(
        model, staggered_vacuum
    )

    assert jnp.array_equal(observables.charges, jnp.zeros((4,)))
    assert jnp.allclose(
        observables.electric_flux,
        model.left_boundary_flux + model.external_flux,
    )
    assert (
        phx.applications.lattice_field.schwinger_gauss_residual(
            model,
            staggered_vacuum,
            observables.electric_flux,
        )
        == 0.0
    )


def test_schwinger_local_and_mpo_hamiltonians_agree():
    model = _model(external_flux=jnp.asarray([0.1, -0.2, 0.05]))
    local = phx.applications.lattice_field.schwinger_local_hamiltonian(model)
    mpo = phx.applications.lattice_field.schwinger_mpo(model)
    local_dense = phx.solver.materialize_local_hamiltonian(local)
    mpo_dense = mpo.operator.to_dense()

    assert local.valid
    assert mpo.local_evidence.hermitian
    assert mpo.electric_evidence.hermitian
    assert mpo.electric_evidence.maximum_bond_dimension <= 3
    assert jnp.allclose(local_dense, mpo_dense, atol=1e-10)


def test_schwinger_zero_coupling_and_background_flux_are_explicit():
    model = _model(
        gauge_coupling=0.0,
        left_boundary_flux=-0.3,
        external_flux=jnp.asarray([0.1, 0.2, -0.1]),
    )
    occupations = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
    flux = phx.applications.lattice_field.reconstruct_schwinger_flux(model, occupations)
    dense = phx.solver.materialize_local_hamiltonian(
        phx.applications.lattice_field.schwinger_local_hamiltonian(model)
    )

    assert jnp.all(jnp.isfinite(flux))
    assert jnp.all(jnp.isfinite(dense))


def test_schwinger_model_is_jittable_and_resource_guarded():
    model = _model()
    occupations = jnp.asarray([1, 0, 1, 0], dtype=jnp.int32)
    flux = jax.jit(
        lambda value: phx.applications.lattice_field.reconstruct_schwinger_flux(
            model, value
        )
    )(occupations)

    assert flux.shape == (3,)
    with pytest.raises(ValueError, match="maximum_terms"):
        phx.applications.lattice_field.schwinger_local_hamiltonian(model, maximum_terms=1)
    with pytest.raises(ValueError, match="at least two"):
        phx.applications.lattice_field.SchwingerChainModel(
            1,
            lattice_spacing=1.0,
            mass=0.0,
            gauge_coupling=0.0,
        )


def test_schwinger_background_schedules_match_direct_hamiltonians():
    model = _model(external_flux=jnp.asarray([0.05, -0.1, 0.02]))
    time_grid = jnp.asarray([0.0, 0.4, 1.0])
    backgrounds = jnp.asarray(
        [
            [0.05, -0.1, 0.02],
            [0.1, 0.0, -0.03],
            [-0.04, 0.08, 0.06],
        ]
    )
    mpo_schedule = phx.applications.lattice_field.schwinger_background_schedule(
        model,
        time_grid,
        backgrounds,
    )
    for index in range(time_grid.size):
        direct_model = _model(external_flux=backgrounds[index])
        direct = phx.applications.lattice_field.schwinger_mpo(
            direct_model
        ).operator.to_dense()
        scheduled = mpo_schedule.coefficients.operator_at(index).to_dense()
        assert jnp.allclose(scheduled, direct, atol=1e-10)

    interval_backgrounds = backgrounds[:-1]
    local_schedule = phx.applications.lattice_field.schwinger_local_background_schedule(
        model,
        time_grid,
        interval_backgrounds,
    )
    for index in range(local_schedule.interval_count):
        scheduled = phx.solver.materialize_local_hamiltonian(
            local_schedule.hamiltonian,
            local_schedule.coefficients[index],
        )
        direct_model = _model(external_flux=interval_backgrounds[index])
        direct = phx.solver.materialize_local_hamiltonian(
            phx.applications.lattice_field.schwinger_local_hamiltonian(direct_model)
        )
        assert jnp.allclose(scheduled, direct, atol=1e-10)
