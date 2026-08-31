import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_hard_initial_condition_precedes_all_coordinate_spectral_residual():
    space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.ChebyshevBasisPlan(6),
            phx.discretization.FourierBasisPlan(8),
        ),
        axis_names=("t", "x"),
        field_name="u",
    ).prepare(
        (
            phx.discretization.AxisDomain.interval(0.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 1.0),
        )
    )
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    x = phx.equations.PDECoordinate(
        "x",
        "space",
        bounds=(0.0, 1.0),
        periodic=True,
    )
    field = phx.equations.PDEField("u", coordinates=("t", "x"))
    initial = phx.equations.PDEParameter("initial", functional=True)
    u = phx.equations.PDEExpression.field("u")
    h = phx.equations.PDEExpression.parameter("initial")
    problem = phx.equations.PDEProblemIR(
        (t, x),
        (field,),
        parameters=(initial,),
        equations=(
            phx.equations.PDEEquation(
                "unit-rate",
                u.derivative("t"),
                phx.equations.PDEExpression.constant(1.0),
            ),
        ),
        conditions=(
            phx.equations.PDECondition(
                "initial",
                "initial",
                u,
                h,
                region="initial",
                coordinate="t",
            ),
        ),
        regions=(phx.equations.PDERegion("initial", "initial", ("x",)),),
    )
    compiled = phx.equations.compile_spectral_residual(
        problem,
        space,
        phx.discretization.PseudospectralMethodPlan(),
        condition_handling="external",
    )
    time_axis = phx.nn.operator.OperatorAxis(
        "t",
        space.axes[0].nodes,
        quadrature_weights=space.axes[0].quadrature_weights,
        basis="legendre",
    )
    space_axis = phx.nn.operator.OperatorAxis(
        "x",
        space.axes[1].nodes,
        quadrature_weights=space.axes[1].quadrature_weights,
        basis="fourier",
        periodic=True,
    )
    query = phx.nn.operator.FunctionSamples(
        values=None,
        axes=(time_axis, space_axis),
    )
    initial_values = jnp.sin(2.0 * jnp.pi * space.axes[1].nodes)
    initial_values = jnp.broadcast_to(
        initial_values,
        (1,) + space.physical_shape,
    )
    initial_samples = phx.nn.operator.FunctionSamples(
        values=initial_values,
        axes=(time_axis, space_axis),
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={"initial": initial_samples},
        queries={"query": query},
        case_axes=("case",),
        case_shape=(1,),
    )
    raw = phx.nn.operator.OperatorPrediction.from_field(
        "output",
        jnp.ones((1,) + space.physical_shape),
        "query",
        query,
        spec=phx.nn.operator.OperatorOutputSpec("scalar"),
        case_axes=("case",),
        case_shape=(1,),
    )
    pipeline = phx.nn.operator.training.OperatorOutputPipeline(
        phx.nn.operator.training.HardConstraintTransform(
            "output",
            lambda coordinates, batch, **kwargs: coordinates[..., 0],
            "tests.spectral_initial_condition",
            lift_fn=lambda coordinates, batch, **kwargs: batch.input("initial").values,
        )
    )
    constrained = pipeline(raw, batch, key=jr.key(0))
    targets = phx.nn.operator.OperatorTargetBatch.from_arrays({}, batch)
    context = phx.nn.operator.training.OperatorLossContext(
        constrained,
        batch,
        targets,
        constrained,
        batch,
        targets,
    )
    term = phx.nn.operator.training.SpectralPDEResidualLoss(
        "unit_rate",
        compiled,
        {"u": "output"},
        {},
    )
    loss = term(
        None,
        constrained,
        batch,
        targets,
        key=jr.key(1),
        step=jnp.asarray(0),
        training=False,
        context=context,
    )
    values = constrained.field("output").values

    assert jnp.allclose(values[:, 0], initial_values[:, 0], atol=1e-12)
    assert loss < 1e-20
