import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax.equations._materials import (
    IdealGasMaterial,
    StiffenedGasMaterial,
    TwoMaterialEOSClosure,
)
from phydrax.equations._multiphase import (
    TwoMaterialVOFStateLayout,
    TwoMaterialVOFSystem,
)


@pytest.fixture(scope="module")
def eos():
    return TwoMaterialEOSClosure(
        IdealGasMaterial(1.4),
        StiffenedGasMaterial(4.4, 2.0, 1.0),
    )


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_layout_is_static_and_explicit(dimension):
    layout = TwoMaterialVOFStateLayout(dimension)
    assert layout.component_count == dimension + 4
    assert layout.component_names == (
        "partial_mass_0",
        "partial_mass_1",
        *(f"momentum_{axis}" for axis in range(dimension)),
        "total_energy",
        "alpha_0",
    )
    assert layout.alpha_index == dimension + 3
    assert layout.energy_index == dimension + 2
    assert tuple(range(layout.momentum_start, layout.momentum_stop)) == tuple(
        range(2, dimension + 2)
    )


def _state(system, *, alpha=0.35, velocity=(0.8, -0.2, 0.15)):
    d = system.dimension
    primitive = jnp.asarray([1.2, 0.7, *velocity[:d], 2.5, alpha], dtype=jnp.float64)
    return system.primitive_to_conserved(primitive)


def test_conservative_and_advective_fluxes(eos):
    system = TwoMaterialVOFSystem(2, eos=eos)
    state = _state(system)
    primitive = system.conserved_to_primitive(state)
    flux = system.physical_flux(state, 0)
    assert jnp.allclose(flux[0], state[0] * primitive[2])
    assert jnp.allclose(flux[1], state[1] * primitive[2])
    assert jnp.allclose(flux[2], state[2] * primitive[2] + system.pressure(state))
    assert jnp.allclose(flux[3], state[3] * primitive[2])
    assert jnp.allclose(flux[-2], (state[-2] + system.pressure(state)) * primitive[2])
    assert flux[-1] == 0.0
    assert jnp.allclose(
        system.phase_transport_flux(0.25, primitive[2]), 0.25 * primitive[2]
    )


def test_normal_flux_covariance_and_signal_bounds(eos):
    system = TwoMaterialVOFSystem(2, eos=eos)
    state = _state(system)
    normal = jnp.asarray([0.6, -0.8])
    normal_flux = system.physical_normal_flux(state, normal)
    expected = normal[0] * system.physical_flux(state, 0) + normal[
        1
    ] * system.physical_flux(state, 1)
    assert normal_flux[-1] == 0.0
    assert jnp.allclose(normal_flux, expected)
    lower, upper = system.normal_signal_bounds(state, state, normal)
    assert lower <= upper
    with pytest.raises(ValueError):
        system.physical_normal_flux(state, jnp.asarray(1.0))


def test_admissibility_pure_phase_and_fail_closed(eos):
    system = TwoMaterialVOFSystem(1, eos=eos)
    mixed = _state(system, alpha=0.5)
    pure0 = _state(system, alpha=1.0)
    pure1 = _state(system, alpha=0.0)
    assert bool(system.admissible(mixed))
    assert bool(system.admissible(pure0))
    assert bool(system.admissible(pure1))
    assert not bool(system.admissible(mixed.at[system.alpha_index].set(jnp.nan)))
    assert not bool(system.admissible(mixed.at[0].set(-1.0)))
    assert not bool(system.admissible(pure1.at[0].set(eos.mass_floor)))
    assert not bool(system.admissible(pure0.at[1].set(eos.mass_floor)))


@pytest.mark.parametrize("alpha", (1.0e-4, 1.0 - 1.0e-4))
def test_system_round_trips_active_alpha_floor_boundaries(alpha):
    floor_eos = TwoMaterialEOSClosure(
        IdealGasMaterial(1.4),
        StiffenedGasMaterial(4.4, 2.0, 1.0),
        alpha_floor=1.0e-4,
        mass_floor=1.0e-8,
    )
    system = TwoMaterialVOFSystem(1, eos=floor_eos)
    primitive = jnp.asarray([1.2, 0.7, 0.8, 2.5, alpha], dtype=jnp.float64)

    conserved = system.primitive_to_conserved(primitive)
    compiled_conserved = jax.jit(system.primitive_to_conserved)(primitive)

    assert bool(system.admissible(conserved))
    assert bool(jax.jit(system.admissible)(compiled_conserved))
    np.testing.assert_allclose(
        system.conserved_to_primitive(conserved), primitive, rtol=2.0e-12
    )
    np.testing.assert_allclose(
        jax.jit(system.conserved_to_primitive)(compiled_conserved),
        primitive,
        rtol=2.0e-12,
    )


def test_jit_grad_and_dtype(eos):
    system = TwoMaterialVOFSystem(2, eos=eos)
    state = _state(system)
    normal = jnp.asarray([0.8, 0.6], dtype=state.dtype)
    flux = jax.jit(system.physical_normal_flux)(state, normal)
    jacobian = jax.jacfwd(lambda q: system.physical_normal_flux(q, normal))(state)
    assert flux.dtype == state.dtype
    assert jacobian.shape == (system.component_count, system.component_count)
    assert np.isfinite(np.asarray(jacobian)).all()


def test_material_identity_is_part_of_system_identity(eos):
    first = TwoMaterialVOFSystem(2, eos=eos)
    second = TwoMaterialVOFSystem(2, eos=eos)
    altered_eos = TwoMaterialEOSClosure(
        IdealGasMaterial(1.5),
        StiffenedGasMaterial(4.4, 2.0, 1.0),
    )
    altered = TwoMaterialVOFSystem(2, eos=altered_eos)
    assert first.system_id == second.system_id
    assert first.diagnostics.diagnostics_id
    assert altered.system_id != first.system_id
    floor_altered_eos = TwoMaterialEOSClosure(
        IdealGasMaterial(1.4),
        StiffenedGasMaterial(4.4, 2.0, 1.0),
        alpha_floor=1.0e-8,
    )
    floor_altered = TwoMaterialVOFSystem(2, eos=floor_altered_eos)
    assert floor_altered.system_id != first.system_id


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_kapila_coefficient_is_exactly_zero_in_pure_phases(eos, alpha):
    system = TwoMaterialVOFSystem(2, eos=eos)
    state = _state(system, alpha=alpha)
    density_0, density_1 = system.phase_densities(state)
    sound_0, sound_1 = system.phase_sound_speeds(state)
    assert system.dilatation_coefficient(state) == 0.0
    assert jnp.isfinite(density_0)
    assert jnp.isfinite(density_1)
    assert jnp.isfinite(sound_0)
    assert jnp.isfinite(sound_1)


def test_kapila_coefficient_is_exactly_zero_for_equal_materials():
    equal_eos = TwoMaterialEOSClosure(
        IdealGasMaterial(1.4),
        IdealGasMaterial(1.4),
    )
    system = TwoMaterialVOFSystem(1, eos=equal_eos)
    state = _state(system, alpha=0.4)
    assert system.dilatation_coefficient(state) == 0.0


def test_mixed_stiffened_ideal_kapila_coefficient_has_expected_sign(eos):
    system = TwoMaterialVOFSystem(2, eos=eos)
    state = _state(system, alpha=0.35)
    density_0, density_1 = system.phase_densities(state)
    sound_0, sound_1 = system.phase_sound_speeds(state)
    pressure = system.pressure(state)
    stiffness_0 = density_0 * sound_0**2
    stiffness_1 = density_1 * sound_1**2
    expected = (
        0.35
        * 0.65
        * (stiffness_1 - stiffness_0)
        / (0.35 * stiffness_1 + 0.65 * stiffness_0)
    )
    coefficient = system.dilatation_coefficient(state)
    assert jnp.allclose(
        sound_0,
        jnp.sqrt(eos.material_0.gamma * pressure / density_0),
    )
    assert jnp.allclose(
        sound_1,
        jnp.sqrt(
            eos.material_1.gamma * (pressure + eos.material_1.pressure_offset) / density_1
        ),
    )
    assert jnp.isfinite(system.sound_speed(state))
    assert jnp.isfinite(coefficient)
    assert coefficient > 0.0
    assert jnp.allclose(coefficient, expected)
    invalid = state.at[0].set(-1.0)
    assert jnp.isnan(system.dilatation_coefficient(invalid))


def test_uniform_divergence_uses_conservative_volume_fraction_source(eos):
    system = TwoMaterialVOFSystem(2, eos=eos)
    states = jnp.stack(
        (
            _state(system, alpha=0.2),
            _state(system, alpha=0.5),
            _state(system, alpha=0.8),
        )
    )
    alpha = states[..., system.alpha_index]
    divergence = jnp.asarray(0.125, dtype=states.dtype)
    coefficient = system.dilatation_coefficient(states)
    source = system.volume_fraction_source(alpha, divergence, states)
    assert jnp.allclose(source, (alpha + coefficient) * divergence)
    shifted_alpha = alpha + 0.01
    assert jnp.allclose(
        system.volume_fraction_source(shifted_alpha, divergence, states),
        (shifted_alpha + coefficient) * divergence,
    )
    assert jnp.isnan(system.volume_fraction_source(alpha + 2.0, divergence, states)).all()


def test_incompressible_divergence_has_zero_volume_fraction_source(eos):
    system = TwoMaterialVOFSystem(2, eos=eos)
    state = _state(system, alpha=0.35)
    source = system.volume_fraction_source(
        state[system.alpha_index],
        jnp.asarray(0.0, dtype=state.dtype),
        state,
    )
    assert source == 0.0


def test_kapila_dilatation_is_jittable_and_differentiable(eos):
    system = TwoMaterialVOFSystem(2, eos=eos)
    primitive = system.conserved_to_primitive(_state(system, alpha=0.35))

    def coefficient(alpha):
        state = system.primitive_to_conserved(primitive.at[-1].set(alpha))
        return system.dilatation_coefficient(state)

    def source(alpha):
        state = system.primitive_to_conserved(primitive.at[-1].set(alpha))
        return system.volume_fraction_source(alpha, 0.2, state)

    state = system.primitive_to_conserved(primitive)
    phase_values = jax.jit(
        lambda value: (
            system.phase_densities(value),
            system.phase_sound_speeds(value),
        )
    )(state)
    assert all(
        np.isfinite(np.asarray(value)).all() for pair in phase_values for value in pair
    )
    assert jnp.isfinite(jax.jit(coefficient)(jnp.asarray(0.35)))
    assert jnp.isfinite(jax.grad(coefficient)(jnp.asarray(0.35)))
    assert jnp.isfinite(jax.jit(source)(jnp.asarray(0.35)))
    assert jnp.isfinite(jax.grad(source)(jnp.asarray(0.35)))


def test_kapila_model_variant_changes_all_two_material_identities(eos):
    system = TwoMaterialVOFSystem(2, eos=eos)
    closure_payload = {
        "kind": "two-material-eos-closure",
        "model_variant": "kapila-five-equation-v1",
        "material_0": eos.material_0.material_id,
        "material_1": eos.material_1.material_id,
        "alpha_floor": eos.alpha_floor,
        "density_floor": eos.density_floor,
        "mass_floor": eos.mass_floor,
        "energy_floor": eos.energy_floor,
        "identity": None,
    }
    legacy_payload = dict(closure_payload)
    legacy_payload.pop("model_variant")
    assert eos.model_variant == "kapila-five-equation-v1"
    assert eos.closure_id == canonical_fingerprint(closure_payload)
    assert eos.closure_id != canonical_fingerprint(legacy_payload)
    assert system.diagnostics.diagnostics_id == canonical_fingerprint(
        {
            "kind": "two-material-vof-diagnostics",
            "dimension": system.dimension,
            "model_variant": eos.model_variant,
            "eos": eos.closure_id,
        }
    )
    assert system.system_id == canonical_fingerprint(
        {
            "kind": "two-material-vof-system",
            "dimension": system.dimension,
            "model_variant": eos.model_variant,
            "eos": eos.closure_id,
            "components": system.component_names,
        }
    )
