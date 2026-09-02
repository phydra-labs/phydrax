#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def peng_robinson_model():
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("methane", "ethane"),
        (phx.equations.ChemicalPhaseKind.GAS,) * 2,
        jnp.asarray((0.016043, 0.03007)),
        ("C", "H"),
        jnp.asarray(((1, 2), (4, 6)), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
        provenance="NIST critical constants transcribed for qualification",
    )
    species_thermodynamics = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray(((2.5 * phx.equations.UNIVERSAL_GAS_CONSTANT,),) * 2),
        jnp.zeros((2,)),
        reference_temperature=300.0,
        minimum_temperature=100.0,
        maximum_temperature=1000.0,
    )
    parameters = phx.equations.PengRobinsonParameters(
        schema.catalog,
        jnp.asarray((190.564, 305.322)),
        jnp.asarray((4.5992e6, 4.8722e6)),
        jnp.asarray((0.01142, 0.0995)),
        jnp.zeros((2, 2)),
        provenance="public critical constants; zero binary interaction assumption",
    )
    ideal = phx.equations.IdealGasReferenceHelmholtzTerm(schema, species_thermodynamics)
    residual = phx.equations.PengRobinsonResidualHelmholtzTerm(schema, parameters)
    return phx.equations.HomogeneousHelmholtzPlan(
        ideal,
        residual,
        maximum_molar_density=5.0e4,
    )


def test_peng_robinson_pressure_has_ideal_low_density_limit():
    model = peng_robinson_model()
    temperature = jnp.asarray(350.0)
    density = jnp.asarray(1.0e-3)
    composition = jnp.asarray((0.4, 0.6))

    state = model.evaluate(temperature, density, composition)

    np.testing.assert_allclose(
        state.pressure,
        density * phx.equations.UNIVERSAL_GAS_CONSTANT * temperature,
        rtol=1.0e-5,
    )
    assert bool(state.evidence.successful)


def test_peng_robinson_enumerates_fixed_three_root_slots():
    model = peng_robinson_model()
    roots = phx.equations.peng_robinson_roots(
        model,
        jnp.asarray(150.0),
        jnp.asarray(1.0e6),
        jnp.asarray((1.0, 0.0)),
    )

    assert roots.compressibility.shape == (3,)
    assert roots.valid.shape == (3,)
    assert bool(roots.successful)
    assert int(jnp.sum(roots.valid)) >= 1
    assert bool(roots.stable[roots.minimum_gibbs_index])


def test_peng_robinson_parameter_identity_is_content_sensitive():
    model = peng_robinson_model()
    base = model.residual.parameters
    changed = phx.equations.PengRobinsonParameters(
        base.catalog,
        base.critical_temperature,
        base.critical_pressure,
        base.acentric_factor,
        jnp.asarray(((0.0, 0.01), (0.01, 0.0))),
        provenance="qualification perturbation",
    )
    assert changed.parameter_id != base.parameter_id


def test_peng_robinson_drives_single_phase_homogeneous_euler():
    model = peng_robinson_model()
    system = phx.equations.HomogeneousMixtureEulerSystem(model)
    primitive = jnp.asarray((0.2, 0.3, 1.0, 350.0))

    state = system.primitive_to_conserved(primitive)
    recovered = system.conserved_to_primitive(state)
    flux = system.physical_flux(state, 0)

    np.testing.assert_allclose(recovered, primitive, rtol=2.0e-9)
    assert bool(system.admissible(state))
    assert bool(jnp.all(jnp.isfinite(flux)))
