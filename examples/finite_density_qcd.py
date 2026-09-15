#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Synthetic B/Q/S coefficients through a qualified finite-density EoS table."""

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    qcd = phx.applications.lattice_field
    convention = qcd.ChemicalChargeConvention(energy_unit=phx.units.GIGAELECTRONVOLT)
    domain = qcd.FiniteDensityDomain(
        (0.15, 0.25),
        ((-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5)),
        maximum_total_order=2,
    )
    indices = (
        qcd.GeneralizedSusceptibilityIndex(0, 0, 0),
        qcd.GeneralizedSusceptibilityIndex(2, 0, 0),
        qcd.GeneralizedSusceptibilityIndex(0, 2, 0),
        qcd.GeneralizedSusceptibilityIndex(0, 0, 2),
    )
    values = jnp.asarray(
        [[1.0, 0.2, 0.15, 0.1], [1.1, 0.22, 0.16, 0.11], [1.2, 0.24, 0.17, 0.12]]
    )
    estimate = qcd.SusceptibilityEstimate(
        jnp.asarray([0.15, 0.20, 0.25]),
        values,
        1.0e-6 * jnp.eye(values.size),
        indices=indices,
        convention=convention,
        domain=domain,
        source_kind=qcd.FiniteDensitySourceKind.CONTINUUM_EXTRAPOLATED,
        provenance_ids=("synthetic-continuum-example",),
    )
    prepared = qcd.prepare_taylor_eos(estimate)
    point = qcd.evaluate_taylor_eos(prepared, 0.20, jnp.asarray([0.02, 0.0, 0.0]))
    table = qcd.build_taylor_eos_table(
        prepared,
        qcd.EOSGridPlan(
            jnp.asarray([0.16, 0.20, 0.24]),
            jnp.asarray([-0.04, 0.0, 0.04]),
            convention=convention,
            domain=domain,
        ),
    )
    qualification = qcd.qualify_eos_table(table)
    if not bool(point.successful) or not bool(qualification.passed):
        raise RuntimeError("Finite-density QCD example failed its declared checks.")
    print(
        {
            "pressure_over_T4": float(point.pressure_over_temperature4),
            "maximum_identity_residual": float(qualification.maximum_identity_residual),
            "qualified_cells": int(jnp.sum(qualification.qualified_cells)),
        }
    )


if __name__ == "__main__":
    main()
