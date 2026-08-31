import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.nn.operator import (
    AxisDependencyReach,
    operator_dependency_support,
    OperatorDependencySupport,
)


def test_axis_dependency_reach_composes_sequentially_and_in_parallel():
    left = AxisDependencyReach(1, 3)
    right = AxisDependencyReach(4, 2)

    assert left.sequential(right) == AxisDependencyReach(5, 5)
    assert left.parallel(right) == AxisDependencyReach(4, 3)
    with pytest.raises(ValueError, match="non-negative"):
        AxisDependencyReach(-1, 0)


def test_dependency_support_algebra_propagates_evidence_global_and_unknown():
    first = OperatorDependencySupport.finite(
        (AxisDependencyReach(1, 2), AxisDependencyReach(3, 4)),
        evidence="exact",
    )
    second = OperatorDependencySupport.finite(
        (AxisDependencyReach(2, 1), AxisDependencyReach(1, 5)),
        evidence="conservative",
    )

    sequential = first.sequential(second)
    parallel = first.parallel(second)
    assert sequential.kind == "finite"
    assert sequential.reach == (
        AxisDependencyReach(3, 3),
        AxisDependencyReach(4, 9),
    )
    assert sequential.evidence == "conservative"
    assert parallel.reach == (
        AxisDependencyReach(2, 2),
        AxisDependencyReach(3, 5),
    )
    assert first.sequential(OperatorDependencySupport.global_(2)).kind == "global"
    assert first.parallel(OperatorDependencySupport.unknown(2)).kind == "unknown"


def test_dependency_scale_is_explicit_and_periodic_reach_saturates():
    support = OperatorDependencySupport.finite(
        (AxisDependencyReach(2, 2), AxisDependencyReach(1, 1))
    ).rescaled((0.25, 2.0))
    assert support.dimension == 2
    assert support.scale == (0.25, 2.0)

    partly_saturated = support.saturated_periodic((5, 9))
    assert partly_saturated.kind == "finite"
    assert partly_saturated.reach[0] == AxisDependencyReach(2, 2)
    assert support.saturated_periodic((5, 3)).kind == "global"

    with pytest.raises(ValueError, match="explicitly rescaled"):
        support.sequential(OperatorDependencySupport.pointwise(2, scale=(1.0, 1.0)))


def test_measure_convolution_authors_dilated_directional_reach():
    layer = phx.nn.layers.MeasureNormalizedConvND(
        spatial_ndim=2,
        in_channels=1,
        out_channels=1,
        kernel_size=(3, 5),
        dilation=(2, 1),
        circular=True,
        key=jr.key(10),
    )

    support = operator_dependency_support(layer)
    assert support.kind == "finite"
    assert support.dimension == 2
    assert support.reach == (
        AxisDependencyReach(2, 2),
        AxisDependencyReach(2, 2),
    )
    assert support.evidence == "conservative"
    periodic_axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.arange(5, dtype=float) / 5,
        basis="fourier",
        periodic=True,
    )
    bound = operator_dependency_support(layer, (periodic_axis, periodic_axis))
    assert bound.kind == "global"
    assert bound.evidence == "conservative"


def test_operator_dependency_support_prefers_instance_provider_and_defaults_unknown():
    expected = OperatorDependencySupport.pointwise(1)

    class AuthoredProvider:
        def dependency_support(self, axes=None, /):
            assert axes is None
            return expected

    class UnauthoredModel:
        pass

    assert operator_dependency_support(AuthoredProvider()) is expected
    unknown = operator_dependency_support(UnauthoredModel(), (object(), object()))
    assert unknown.kind == "unknown"
    assert unknown.dimension == 2
    assert unknown.evidence == "conservative"


def test_periodic_operator_instances_classify_their_authored_dependency():
    finite_cno = phx.nn.operator.architectures.CNO(
        spatial_ndim=1,
        width=3,
        depth=2,
        kernel_size=3,
        oversample_factor=1,
        key=jr.key(0),
    )
    global_cno = phx.nn.operator.architectures.CNO(
        spatial_ndim=1,
        width=3,
        depth=1,
        oversample_factor=2,
        key=jr.key(1),
    )
    uno = phx.nn.operator.architectures.UNO(
        spatial_ndim=1,
        widths=(3, 4),
        oversample_factor=1,
        key=jr.key(2),
    )
    fno = phx.nn.operator.architectures.FNO(
        n_modes=(3,),
        width=3,
        depth=1,
        key=jr.key(3),
    )

    finite = operator_dependency_support(finite_cno)
    assert finite.kind == "finite"
    assert finite.dimension == 1
    assert finite.reach == (AxisDependencyReach(4, 4),)
    assert operator_dependency_support(global_cno).kind == "global"
    uno_support = operator_dependency_support(uno)
    assert uno_support.kind == "global"
    assert uno_support.evidence == "conservative"
    assert operator_dependency_support(fno).kind == "global"

    axis = phx.nn.operator.OperatorAxis(
        "x",
        2.0 * jnp.arange(16, dtype=float) / 16,
        basis="fourier",
        periodic=True,
    )
    scaled = operator_dependency_support(finite_cno, (axis,))
    assert scaled.kind == "finite"
    assert scaled.scale == (0.125,)

    saturated_axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.arange(9, dtype=float) / 9,
        basis="fourier",
        periodic=True,
    )
    assert operator_dependency_support(finite_cno, (saturated_axis,)).kind == "global"
