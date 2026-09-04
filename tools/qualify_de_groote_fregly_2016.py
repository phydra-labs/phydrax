"""Independently qualify De Groote--Fregly 2016 equations against the primary source."""

from __future__ import annotations

import argparse
import json

import jax.numpy as jnp
import numpy as np

from phydrax.applications.skeletal_muscle.musculotendon import (
    de_groote_fregly_2016_active_force_length,
    de_groote_fregly_2016_force_velocity,
    de_groote_fregly_2016_inverse_force_velocity,
    de_groote_fregly_2016_inverse_tendon_force_length,
    de_groote_fregly_2016_passive_force_length,
    de_groote_fregly_2016_tendon_force_length,
    DeGrooteFregly2016ImplicitTendonForcePlan,
    DeGrooteFregly2016Parameters,
    DeGrooteFregly2016Plan,
    DeGrooteFregly2016State,
)


SOURCE = {
    "article": "De Groote, Kinney, Rao & Fregly, Ann Biomed Eng 44 (2016) 2922-2936",
    "doi": "10.1007/s10439-016-1591-9",
    "claims": "manuscript Eqs. 1-7; online supplement Eqs. S1-S19 and Table 1",
}


def _parameters() -> DeGrooteFregly2016Parameters:
    return DeGrooteFregly2016Parameters(
        jnp.asarray([1500.0]),
        jnp.asarray([0.1]),
        jnp.asarray([0.2]),
        jnp.asarray([0.15]),
        jnp.asarray([1.0]),
    )


def qualify(sample_count: int) -> dict[str, object]:
    parameters = _parameters()
    tendon_length = np.linspace(0.98, 1.06, sample_count)
    fiber_length = np.linspace(0.4, 1.6, sample_count)
    fiber_velocity = np.linspace(-1.0, 1.0, sample_count)

    reference_tendon = 0.2 * np.exp(35.0 * (tendon_length - 0.995)) - 0.25
    b1 = np.asarray([0.815, 0.433, 0.100])
    b2 = np.asarray([1.055, 0.717, 1.000])
    b3 = np.asarray([0.162, -0.030, 0.354])
    b4 = np.asarray([0.063, 0.200, 0.000])
    width = b3 + fiber_length[:, None] * b4
    reference_active = np.sum(
        b1 * np.exp(-0.5 * ((fiber_length[:, None] - b2) / width) ** 2), axis=-1
    )
    reference_passive = (
        np.exp(4.0 * (fiber_length - 1.0) / 0.6) - 1.0
    ) / np.expm1(4.0)
    reference_velocity = -0.318 * np.arcsinh(-8.149 * fiber_velocity - 0.374) + 0.886

    observed_tendon = np.asarray(
        de_groote_fregly_2016_tendon_force_length(parameters, tendon_length)
    )
    observed_active = np.asarray(
        de_groote_fregly_2016_active_force_length(parameters, fiber_length)
    )
    observed_passive = np.asarray(
        de_groote_fregly_2016_passive_force_length(parameters, fiber_length)
    )
    observed_velocity = np.asarray(
        de_groote_fregly_2016_force_velocity(parameters, fiber_velocity)
    )
    tendon_roundtrip = np.asarray(
        de_groote_fregly_2016_inverse_tendon_force_length(
            parameters, observed_tendon
        )
    )
    velocity_roundtrip = np.asarray(
        de_groote_fregly_2016_inverse_force_velocity(parameters, observed_velocity)
    )

    activation = np.asarray([0.45])
    active_at_one = np.asarray(
        de_groote_fregly_2016_active_force_length(parameters, np.asarray([1.0]))
    )
    velocity_at_zero = np.asarray(
        de_groote_fregly_2016_force_velocity(parameters, np.asarray([0.0]))
    )
    tendon_force = activation * active_at_one * velocity_at_zero * np.cos(0.15)
    tendon_length_m = np.asarray(
        de_groote_fregly_2016_inverse_tendon_force_length(parameters, tendon_force)
    ) * 0.2
    length_mt = tendon_length_m + 0.1 * np.cos(0.15)
    prepared = DeGrooteFregly2016Plan(parameters, ("qualification",)).prepare()
    equilibrium = prepared.evaluate(
        DeGrooteFregly2016State(activation, tendon_force),
        activation,
        length_mt,
        np.zeros((1,)),
    )
    state = DeGrooteFregly2016State(activation, tendon_force)
    implicit = DeGrooteFregly2016ImplicitTendonForcePlan(
        parameters, ("qualification",)
    ).prepare(state)
    implicit_candidate = implicit.candidate(
        state,
        activation,
        length_mt,
        np.zeros((1,)),
        np.asarray(1.0e-5),
    )

    errors = {
        "tendon_curve_max_abs": float(np.max(np.abs(observed_tendon - reference_tendon))),
        "active_curve_max_abs": float(np.max(np.abs(observed_active - reference_active))),
        "passive_curve_max_abs": float(np.max(np.abs(observed_passive - reference_passive))),
        "force_velocity_max_abs": float(np.max(np.abs(observed_velocity - reference_velocity))),
        "tendon_inverse_max_abs": float(np.max(np.abs(tendon_roundtrip - tendon_length))),
        "force_velocity_inverse_max_abs": float(np.max(np.abs(velocity_roundtrip - fiber_velocity))),
        "equilibrium_residual_max_abs": float(
            np.max(np.abs(equilibrium.evidence.force_equilibrium_residual_normalized))
        ),
        "tendon_constitutive_residual_max_abs": float(
            np.max(
                np.abs(
                    equilibrium.evidence.tendon_constitutive_residual_normalized
                )
            )
        ),
        "force_velocity_inverse_residual_max_abs": float(
            np.max(
                np.abs(
                    equilibrium.evidence.force_velocity_inverse_residual_normalized
                )
            )
        ),
        "tendon_rate_residual_per_s_max_abs": float(
            np.max(np.abs(equilibrium.evidence.tendon_rate_residual_per_s))
        ),
        "power_balance_residual_W_max_abs": float(
            np.max(np.abs(equilibrium.evidence.power_balance_residual_W))
        ),
        "implicit_S25_residual_max_abs": float(
            np.max(np.abs(implicit_candidate.evidence.algebraic_residual))
        ),
    }
    tolerance = 2.0e-5
    return {
        "source": SOURCE,
        "sample_count": sample_count,
        "errors": errors,
        "tolerance": tolerance,
        "passed": max(errors.values()) <= tolerance
        and bool(np.all(equilibrium.successful))
        and bool(implicit_candidate.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1001)
    arguments = parser.parse_args()
    if arguments.samples < 3:
        raise ValueError("--samples must be at least 3.")
    report = qualify(arguments.samples)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
