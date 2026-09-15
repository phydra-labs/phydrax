#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity accelerator bunch through a reference focusing channel."""

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    accelerator = phx.applications.accelerator
    convention = accelerator.AcceleratorConvention()
    bunch = accelerator.AcceleratorBunch(
        jnp.asarray(
            [
                [1.0e-3, 0.0, 0.5e-3, 0.0, 0.0, 0.0],
                [-1.0e-3, 0.0, -0.5e-3, 0.0, 0.0, 0.0],
            ]
        ),
        jnp.ones(2),
        jnp.asarray([10, 11]),
        reference_rest_energy=0.938,
        reference_momentum=1.0,
        reference_charge=1.0,
        convention=convention,
        bunch_id="example-bunch",
    )
    beamline = accelerator.BeamlinePlan(
        jnp.asarray([0, 1, 0]),
        jnp.asarray([0.5, 0.2, 0.5]),
        jnp.asarray([0.0, 1.2, 0.0]),
        jnp.zeros(3),
        element_ids=("D1", "Q1", "D2"),
        convention=convention,
    )
    result = accelerator.track_beamline(beamline, bunch)
    diagnostics = accelerator.beam_diagnostics(result.bunch)
    if not bool(result.accepted):
        raise RuntimeError("Accelerator beamline rejected a finite reference bunch.")
    print(
        {
            "transmission": float(diagnostics.transmission),
            "loss_indices": result.loss_element_indices.tolist(),
        }
    )


if __name__ == "__main__":
    main()
