#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

from phydrax.geometry.analytic import Circle, SharpCSG
from phydrax.geometry.design import (
    CSGContinuationPolicy,
    ParameterId,
    ParameterTarget,
    prepare_csg_continuation,
)


def test_recursive_csg_continuation_has_fixed_width_schema_and_exact_transfer():
    left = Circle((0.0, 0.0), 1.0, feature_id="left")
    middle = Circle((0.5, 0.0), 1.0, feature_id="middle")
    right = Circle((1.0, 0.0), 1.0, feature_id="right")
    nested = SharpCSG((left, middle), "intersection")
    source = SharpCSG((nested, right), "difference")
    policy = CSGContinuationPolicy((0.4, 0.2, 0.05))
    prepared = prepare_csg_continuation(
        source,
        (ParameterTarget(ParameterId("left", "center"), (0.0, 0.0)),),
        policy,
    )
    assert len(prepared.width_parameter_ids) == 2
    assert all(
        not prepared.smooth_geometry.schema.spec(parameter_id).trainable
        for parameter_id in prepared.width_parameter_ids
    )
    assert set(prepared.shared_parameter_ids) == set(
        prepared.sharp_geometry.schema.parameter_ids
    )


def test_csg_continuation_schedule_must_be_positive_and_nonincreasing():
    with pytest.raises(ValueError, match="nonincreasing"):
        CSGContinuationPolicy((0.1, 0.2))
    with pytest.raises(ValueError, match="positive"):
        CSGContinuationPolicy((0.1, 0.0))
