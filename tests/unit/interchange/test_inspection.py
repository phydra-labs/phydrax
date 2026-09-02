#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.interchange import (
    AdapterReport,
    AdapterStatus,
    HostInspectionConversion,
    HostInspectionField,
    HostInspectionFrame,
)


def test_host_inspection_is_an_immutable_non_pytree_host_record():
    source = np.arange(6.0).reshape((3, 2))
    field = HostInspectionField(
        "velocity",
        source,
        jnp.asarray([True, False, True]),
        "particle",
        "particle-support",
        "particle-vector-layout",
        "particle_value",
        component_labels=("x", "y"),
        provenance_id="simulation:step-4:candidate",
    )
    source[0, 0] = -1.0

    assert isinstance(field.values, np.ndarray)
    assert isinstance(field.valid, np.ndarray)
    assert not field.values.flags.writeable
    assert not field.valid.flags.writeable
    assert field.values[0, 0] == 0.0
    with pytest.raises(ValueError):
        field.values[0, 0] = 7.0

    frame = HostInspectionFrame(
        0.25,
        4,
        "candidate",
        False,
        3,
        (field,),
        "simulation",
        "simulation:step-4:candidate",
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "native-diagnostic",
        "phydrax-host-inspection-frame",
        source_id="simulation:step-4",
        target_id=frame.result_id,
        preserved_fields=("velocity",),
    )
    conversion = HostInspectionConversion(frame, report)

    assert conversion.frame is frame
    leaves = jax.tree_util.tree_leaves(frame)
    assert len(leaves) == 1 and leaves[0] is frame


def test_host_inspection_rejects_ambiguous_field_and_frame_layouts():
    with pytest.raises(ValueError, match="validity"):
        HostInspectionField(
            "alpha",
            np.ones((2, 3)),
            np.ones((2,), dtype=bool),
            "cell",
            "cells",
            "cell-layout",
            "cell_average",
            provenance_id="source",
        )

    field = HostInspectionField(
        "alpha",
        np.ones((2, 3)),
        True,
        "cell",
        "cells",
        "cell-layout",
        "cell_average",
        provenance_id="source",
    )
    with pytest.raises(ValueError, match="unique"):
        HostInspectionFrame(
            0.0,
            0,
            "accepted",
            True,
            0,
            (field, field),
            "producer",
            "result",
        )
