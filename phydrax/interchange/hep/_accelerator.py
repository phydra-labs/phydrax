#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from ...applications.accelerator._beam import (
        AcceleratorBunch,
        AcceleratorConvention,
        BeamlinePlan,
    )



def accelerator_bunch_from_openpmd_columns(
    columns: Mapping[str, object],
    /,
    *,
    reference_rest_energy: float,
    reference_momentum: float,
    reference_charge: float,
    convention: AcceleratorConvention,
    bunch_id: str,
) -> AcceleratorBunch:
    """Import one explicit normalized six-coordinate openPMD column profile."""
    from ...applications.accelerator._beam import AcceleratorBunch

    required = (
        "x",
        "px_over_p0",
        "y",
        "py_over_p0",
        "zeta",
        "delta",
        "weight",
        "id",
        "active",
    )
    if set(columns) != set(required):
        raise ValueError(
            "openPMD column profile must contain exactly the normalized bunch fields."
        )
    coordinates = np.stack(
        tuple(np.asarray(columns[name]) for name in required[:6]), axis=1
    )
    return AcceleratorBunch(
        coordinates,
        columns["weight"],
        columns["id"],
        active=columns["active"],
        reference_rest_energy=reference_rest_energy,
        reference_momentum=reference_momentum,
        reference_charge=reference_charge,
        convention=convention,
        bunch_id=bunch_id,
    )


def write_madx_sequence(plan: BeamlinePlan, /, *, sequence_name: str = "PHYDRAX") -> str:
    """Write the supported drift/quadrupole/steerer subset as a MAD-X sequence."""
    from ...applications.accelerator._beam import (
        BeamlineElementKind,
        BeamlinePlan,
    )

    if not isinstance(plan, BeamlinePlan):
        raise TypeError("plan must be BeamlinePlan.")
    name = str(sequence_name).strip()
    if not name:
        raise ValueError("sequence_name must be non-empty.")
    lines = [f"{name}: SEQUENCE, L={float(np.sum(plan.lengths)):.17g};"]
    for index, element_id in enumerate(plan.element_ids):
        if not bool(plan.active[index]):
            continue
        kind = BeamlineElementKind(int(plan.kinds[index]))
        length = float(plan.lengths[index])
        strength = float(plan.strengths[index])
        if kind is BeamlineElementKind.DRIFT:
            declaration = f"DRIFT, L={length:.17g}"
        elif kind is BeamlineElementKind.QUADRUPOLE:
            declaration = f"QUADRUPOLE, L={length:.17g}, K1={strength:.17g}"
        elif kind is BeamlineElementKind.STEERER:
            declaration = f"HKICKER, L={length:.17g}, KICK={strength:.17g}"
        else:
            raise ValueError(
                "MAD-X export supports only drift, quadrupole, and horizontal steerer elements."
            )
        lines.append(f"  {element_id}: {declaration};")
    lines.extend(("ENDSEQUENCE;", ""))
    return "\n".join(lines)


__all__ = ["accelerator_bunch_from_openpmd_columns", "write_madx_sequence"]
