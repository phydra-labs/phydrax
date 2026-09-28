#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native no-table matter-radiation references and charged-step emission."""

from __future__ import annotations

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    energy = jnp.asarray(1.0e9)
    nuclear, electron = phx.equations.bethe_heitler_pair_cross_section_m2(
        energy, jnp.asarray(13.0)
    )
    transition = phx.equations.FoilStackTransitionRadiationPlan(
        jnp.asarray((0.0, 20.0e-6, 220.0e-6, 240.0e-6)),
        jnp.asarray((225.0, -225.0, 225.0, -225.0)),
        jnp.asarray((200.0, 5.0, 200.0, 5.0)),
    )
    formation = transition.formation_length_m(5000.0, 1000.0, 1.0e-3)
    count, optical_energy = phx.equations.cherenkov_yield_in_band(
        0.01, 0.99, 1.5, 300.0e-9, 600.0e-9
    )
    print(
        {
            "pair_nuclear_m2": float(nuclear),
            "pair_electron_m2": float(electron),
            "transition_formation_length_m": float(formation),
            "cherenkov_expected_count": float(count),
            "cherenkov_energy_ev": float(optical_energy),
        }
    )


if __name__ == "__main__":
    main()
