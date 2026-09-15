#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Small public-boundary spectroscopy smoke workflow."""

from __future__ import annotations

import json

import numpy as np

from phydrax import units
from phydrax.chemistry.spectroscopy._instrument import (
    apply_spectral_instrument,
    prepare_spectral_instrument,
    SpectralInstrumentKind,
    SpectralInstrumentPlan,
)
from phydrax.chemistry.spectroscopy._profile import SpectralLineShape, SpectralProfilePlan
from phydrax.chemistry.spectroscopy._response import (
    SpectralResponseEvidence,
    SpectralResponseProduct,
    SpectralResponseRepresentation,
)
from phydrax.observation import CoordinateLayout, LinearObservationPlan


def main() -> None:
    raw = SpectralResponseProduct(
        [1.0, 2.0],
        [[1.0, 2.0]],
        [True, True],
        units.ELECTRONVOLT,
        units.ONE,
        ("parallel",),
        SpectralResponseRepresentation.LINES,
        "smoke-lines",
        "analytic-source",
        SpectralResponseEvidence(0.0, 0.0, 0.0, 0.0, True),
    )
    profile = SpectralProfilePlan(
        SpectralLineShape.GAUSSIAN,
        0.0,
        3.0,
        grid_size=601,
        fwhm=0.1,
        area_tolerance=1.0e-8,
    )
    instrument = prepare_spectral_instrument(
        SpectralInstrumentPlan(
            SpectralInstrumentKind.LINE_PROFILE,
            "smoke-lines",
            1,
            601,
            profile=profile,
            area_tolerance=1.0e-8,
        )
    )
    convolved = apply_spectral_instrument(instrument, raw)
    selected_layout = CoordinateLayout(("integrated-channel",))
    weights = np.ones((1, convolved.theory.layout.size)) * (
        float(convolved.coordinates[1] - convolved.coordinates[0])
    )
    observed = LinearObservationPlan(
        weights,
        convolved.theory.layout,
        selected_layout,
    ).apply(convolved.theory)
    print(
        json.dumps(
            {
                "raw_product_id": raw.product_id,
                "instrument_result_id": convolved.result_id,
                "theory_product_id": convolved.theory.product_id,
                "integrated_response": float(observed.values[0]),
                "expected_response": 3.0,
                "area_residual": float(convolved.evidence.area_residual),
                "successful": bool(convolved.evidence.successful),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
