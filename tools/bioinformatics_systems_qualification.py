#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Flux-balance and kinetic-rate qualification for systems biology."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.bioinformatics import systems
from tools.bioinformatics_common_qualification import (
    emit_report,
    fingerprint,
    method_contract_evidence,
    qualification_report,
)


def _flux_network() -> systems.StoichiometricNetwork:
    compartment = systems.Compartment("cell", volume=1.0)
    species = systems.Species("A", "cell", initial_amount=1.0)
    uptake = systems.Reaction(
        "uptake",
        ("A",),
        (1.0,),
        lower_bound=0.0,
        upper_bound=10.0,
        exchange=True,
    )
    biomass = systems.Reaction(
        "biomass",
        ("A",),
        (-1.0,),
        lower_bound=0.0,
        upper_bound=100.0,
        objective_coefficient=1.0,
        exchange=True,
    )
    return systems.StoichiometricNetwork((compartment,), (species,), (uptake, biomass))


def _flux_balance_case() -> dict[str, object]:
    network = _flux_network()
    result = systems.flux_balance_analysis(network, detect_alternate_optima=False)
    fluxes = np.asarray(result.fluxes)
    objective = float(np.asarray(result.objective_value))
    mass_residual = float(
        np.max(np.abs(np.asarray(result.evidence.mass_balance_residual)))
    )
    oracle_fluxes = np.asarray((10.0, 10.0))
    flux_error = float(np.max(np.abs(fluxes - oracle_fluxes)))

    capacity_rejected = False
    capacity_error = ""
    try:
        systems.flux_balance_analysis(
            network,
            detect_alternate_optima=True,
            max_auxiliary_solves=3,
        )
    except systems.FluxCapacityError as error:
        capacity_rejected = True
        capacity_error = str(error)

    contract = method_contract_evidence(result.method_contract)
    inputs = {
        "network_id": network.network_id,
        "stoichiometric_matrix": network.stoichiometric_matrix,
        "lower_bounds": network.lower_bounds,
        "upper_bounds": network.upper_bounds,
        "objective_coefficients": network.objective_coefficients,
    }
    return {
        "scope": "unit_qualification",
        "oracle": "analytic optimum of v_uptake = v_biomass with uptake bound 10",
        "input_fingerprint": fingerprint(inputs),
        "method_fingerprint": contract["fingerprint"],
        "method": contract,
        "network_id": network.network_id,
        "observed_fluxes": fluxes.tolist(),
        "oracle_fluxes": oracle_fluxes.tolist(),
        "maximum_flux_error": flux_error,
        "objective_value": objective,
        "maximum_mass_balance_residual": mass_residual,
        "status": int(np.asarray(result.status)),
        "valid": bool(np.asarray(result.valid)),
        "capacity_check": {
            "requested_auxiliary_solve_capacity": 3,
            "required_auxiliary_solve_capacity": 4,
            "rejected_before_materialization": capacity_rejected,
            "error": capacity_error,
        },
        "passed": bool(
            np.asarray(result.valid)
            and flux_error <= 3.0e-4
            and abs(objective - 10.0) <= 3.0e-4
            and mass_residual <= 3.0e-5
            and capacity_rejected
        ),
    }


def _kinetic_gradient_case() -> dict[str, object]:
    reaction = systems.KineticReaction(
        0,
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((1.0,)),
        jnp.asarray((2.0, 3.0)),
        rate_law=systems.RateLawKind.MICHAELIS_MENTEN,
        rate_unit=systems.SUBSTANCE_FLUX,
        kinetic_id="michaelis-menten-unit",
    )
    concentrations = jnp.asarray((4.0,))
    rate = float(np.asarray(reaction.evaluate(concentrations)))
    oracle_rate = 2.0 * 4.0 / (3.0 + 4.0)
    rate_error = abs(rate - oracle_rate)

    def objective(values):
        return reaction.evaluate(values)

    automatic_gradient = float(np.asarray(jax.grad(objective)(concentrations)[0]))
    analytic_gradient = 2.0 * 3.0 / (3.0 + 4.0) ** 2
    gradient_error = abs(automatic_gradient - analytic_gradient)
    method = {
        "method_name": "michaelis-menten-rate-law",
        "method_kind": "approximate_model",
        "execution_kind": "floating_point_direct",
        "differentiation_kind": "exact_ad",
        "conditioning_statement": (
            "The quasi-steady-state Michaelis-Menten approximation is conditioned "
            "on declared Vmax and half-saturation parameters."
        ),
        "concentration_unit": "declared_network_concentration",
        "rate_unit": "amount_per_time",
    }
    method_fingerprint = fingerprint(method)
    inputs = {
        "parameters": reaction.parameters,
        "reactant_orders": reaction.reactant_orders,
        "concentrations": concentrations,
        "rate_scale": reaction.rate_scale,
    }
    return {
        "scope": "unit_qualification",
        "oracle": "Vmax * substrate / (Km + substrate)",
        "gradient_oracle": "Vmax * Km / (Km + substrate)^2",
        "input_fingerprint": fingerprint(inputs),
        "method_fingerprint": method_fingerprint,
        "method": {"fingerprint": method_fingerprint, **method},
        "observed_rate": rate,
        "oracle_rate": oracle_rate,
        "absolute_rate_error": rate_error,
        "automatic_gradient": automatic_gradient,
        "analytic_gradient": analytic_gradient,
        "absolute_gradient_error": gradient_error,
        "passed": rate_error <= 2.0e-6 and gradient_error <= 2.0e-6,
    }


def qualification() -> dict[str, object]:
    return qualification_report(
        "systems",
        {
            "flux_balance": _flux_balance_case(),
            "michaelis_menten_gradient": _kinetic_gradient_case(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qualify public systems-biology flux and kinetic APIs."
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    return emit_report(qualification(), arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
