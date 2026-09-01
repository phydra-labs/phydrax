from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output = Path(sys.argv[2])
    scale_factors = np.asarray([0.5, 1.0])
    wavenumbers = np.asarray([1.0, 2.0, 4.0])
    field_count = len(request["transfer_fields"])
    transfer_values = np.stack(
        tuple(
            (index + 1.0) * scale_factors[:, None] * wavenumbers[None, :]
            for index in range(field_count)
        )
    )
    power_values = scale_factors[:, None] ** 2 * wavenumbers[None, :]
    np.savez(
        output,
        scale_factors=scale_factors,
        wavenumbers=wavenumbers,
        transfer_values=transfer_values,
        power_values=power_values,
        ionization_fraction=np.asarray([1.0, 1.0e-4]),
        baryon_temperature=np.asarray([3000.0, 2.7255]),
        opacity_derivative=np.asarray([-10.0, -1.0e-5]),
        visibility=np.asarray([0.1, 0.0]),
    )


if __name__ == "__main__":
    main()
