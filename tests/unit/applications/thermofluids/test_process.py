#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _model():
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("air",),
        (phx.equations.ChemicalPhaseKind.GAS,),
        jnp.asarray((0.02897,)),
        ("air",),
        jnp.asarray(((1,),), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    species = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray(((2.5 * phx.equations.UNIVERSAL_GAS_CONSTANT,),)),
        jnp.zeros((1,)),
        reference_temperature=300.0,
        minimum_temperature=100.0,
        maximum_temperature=2000.0,
    )
    ideal = phx.equations.IdealGasReferenceHelmholtzTerm(schema, species)
    return phx.equations.HomogeneousHelmholtzPlan(
        ideal, phx.equations.ZeroResidualHelmholtzTerm(schema)
    )


def test_typed_material_process_lowers_to_acausal_dae():
    model = _model()
    tf = phx.applications.thermofluids
    source = tf.fixed_material_boundary_component(
        "source",
        pressure=2.0e5,
        specific_enthalpy=4.0e5,
        mass_flow=1.0,
        catalog_id=model.schema.catalog.catalog_id,
        thermodynamics_id=model.model_id,
        direction=tf.MaterialFlowDirection.OUTLET,
    )
    valve = tf.isenthalpic_valve_component(
        "valve",
        pressure_ratio=0.5,
        catalog_id=model.schema.catalog.catalog_id,
        thermodynamics_id=model.model_id,
    )
    sink = tf.fixed_material_boundary_component(
        "sink",
        pressure=1.0e5,
        specific_enthalpy=4.0e5,
        mass_flow=-1.0,
        catalog_id=model.schema.catalog.catalog_id,
        thermodynamics_id=model.model_id,
        direction=tf.MaterialFlowDirection.INLET,
    )
    process = tf.ThermofluidProcessPlan(
        (source, valve, sink),
        (
            tf.ThermofluidConnection("source", "material", "valve", "inlet"),
            tf.ThermofluidConnection("valve", "outlet", "sink", "material"),
        ),
    )

    assert isinstance(process.source, phx.dynamics.AcausalDAESource)
    assert len(process.source.connections) == 2
    assert process.process_model_id != process.source.source_id


def test_material_connection_rejects_mismatched_thermodynamics():
    model = _model()
    tf = phx.applications.thermofluids
    source = tf.fixed_material_boundary_component(
        "source",
        pressure=1.0e5,
        specific_enthalpy=1.0,
        mass_flow=1.0,
        catalog_id=model.schema.catalog.catalog_id,
        thermodynamics_id=model.model_id,
        direction=tf.MaterialFlowDirection.OUTLET,
    )
    sink = tf.fixed_material_boundary_component(
        "sink",
        pressure=1.0e5,
        specific_enthalpy=1.0,
        mass_flow=-1.0,
        catalog_id=model.schema.catalog.catalog_id,
        thermodynamics_id="different",
        direction=tf.MaterialFlowDirection.INLET,
    )
    with np.testing.assert_raises(ValueError):
        tf.ThermofluidProcessPlan(
            (source, sink),
            (tf.ThermofluidConnection("source", "material", "sink", "material"),),
        )


def test_compressor_map_design_and_ideal_station_balance():
    model = _model()
    tf = phx.applications.thermofluids
    performance_map = tf.CompressorMapPlan(
        jnp.asarray((0.8, 1.0)),
        jnp.asarray((0.0, 1.0)),
        jnp.asarray(((8.0, 9.0), (10.0, 11.0))),
        jnp.asarray(((1.5, 1.6), (2.0, 2.1))),
        jnp.asarray(((0.75, 0.76), (0.8, 0.81))),
        reference_temperature=288.15,
        reference_pressure=101325.0,
        provenance="synthetic qualification map",
    )
    compressor = tf.CompressorPlan(model, performance_map)
    design = compressor.design(
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        corrected_flow=10.0,
        pressure_ratio=2.0,
        isentropic_efficiency=0.8,
    )
    inlet = compressor.station(
        jnp.asarray(300.0),
        jnp.asarray(1.0e5),
        jnp.asarray(1.0),
        jnp.asarray((1.0,)),
    )
    result = compressor.evaluate(inlet, jnp.asarray(1.0), jnp.asarray(0.0), design)

    assert bool(result.successful)
    np.testing.assert_allclose(result.map_evaluation.pressure_ratio, 2.0)
    np.testing.assert_allclose(result.map_evaluation.isentropic_efficiency, 0.8)
    assert result.outlet.total_pressure > inlet.total_pressure
    assert result.outlet.mass_specific_enthalpy > inlet.mass_specific_enthalpy
    assert result.shaft_power > 0.0
