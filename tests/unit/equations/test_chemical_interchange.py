#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_yaml_import_normalizes_units_and_calibration_changes_rate(tmp_path):
    source = tmp_path / "mechanism.yaml"
    source.write_text(
        """
name: conversion
units:
  length: cm
  time: s
  amount: mol
  mass: g
  energy: cal
  pressure: atm
  temperature: K
phases:
  - name: gas
    kind: gas
    measure-dimension: 3
    standard-pressure: 1.0
species:
  - name: A
    phase: gas
    molar-mass: 1.0
    charge: 0
    composition: {X: 1}
  - name: B
    phase: gas
    molar-mass: 1.0
    charge: 0
    composition: {X: 1}
thermodynamics:
  model: polynomial-internal-energy
  reference-temperature: 300.0
  minimum-temperature: 200.0
  maximum-temperature: 1000.0
  species:
    A:
      heat-capacity-volume: [10.0]
      reference-internal-energy: 0.0
    B:
      heat-capacity-volume: [10.0]
      reference-internal-energy: 0.0
reactions:
  - name: A-to-B
    reactants: {A: 1}
    products: {B: 1}
    rate: {type: arrhenius, A: 2.0, b: 0.0, Ea: 0.0}
""",
        encoding="utf-8",
    )
    report = phx.equations.load_chemical_mechanism_yaml(source)
    mechanism = report.mechanism.prepare()
    baseline = mechanism.evaluate(
        jnp.asarray((1.0, 0.0)),
        jnp.asarray(500.0),
        jnp.asarray(101325.0),
    )
    parameter = phx.equations.ChemicalCalibrationParameter(
        "forward-A",
        0,
        "pre_exponential",
        phx.equations.ChemicalParameterCoordinate.MULTIPLICATIVE,
        jnp.asarray(2.0),
    )
    calibrated = phx.equations.ChemicalCalibrationPlan(mechanism, (parameter,)).apply(
        jnp.asarray((1.0,))
    )
    updated = calibrated.evaluate(
        jnp.asarray((1.0, 0.0)),
        jnp.asarray(500.0),
        jnp.asarray(101325.0),
    )

    assert "length" in report.converted_fields
    np.testing.assert_allclose(
        updated.forward_progress_rates,
        2.0 * baseline.forward_progress_rates,
    )
