#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import math

import jax.numpy as jnp

from phydrax.applications import magnetic_resonance as mr


def run() -> dict[str, object]:
    isotope = mr.ResonanceIsotope(
        "smoke",
        "nucleus",
        0.5,
        2.0 * math.pi * 100.0,
    )
    prepared = mr.ExactSingleCrystalNMRProfile(
        mr.MagneticResonanceSpinSystem(
            (mr.SpinSite("spin", isotope),),
            (0.0, 0.0, 1.0),
            system_id="magnetic-resonance-smoke",
        )
    ).prepare()
    plus_x = jnp.asarray([1.0, 1.0], dtype="complex128") / jnp.sqrt(2.0)
    density = jnp.outer(plus_x, jnp.conj(plus_x))
    fid = mr.acquire_fid(
        prepared,
        density,
        mr.AcquisitionPlan(1.0 / 1024.0, 1024),
    )
    instrument = mr.apply_instrument(fid, mr.InstrumentPlan())
    peak_index = jnp.argmax(jnp.abs(instrument.spectrum.amplitude))
    return {
        "profile_id": "magnetic-resonance:nmr:exact-small-single-crystal",
        "hilbert_dimension": prepared.layout.dimension,
        "density_elements": prepared.layout.dimension**2,
        "peak_frequency_hz": float(instrument.spectrum.frequency_hz[peak_index]),
        "hamiltonian_hermiticity_residual": float(prepared.evidence.hermiticity_residual),
        "maximum_trace_residual": float(jnp.max(fid.trace_residuals)),
        "step_unitarity_residual": float(fid.step_unitarity_residual),
        "valid": bool(prepared.evidence.valid & instrument.valid),
    }


def main() -> int:
    report = run()
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
