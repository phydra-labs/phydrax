import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray

import phydrax as phx


def test_periodic_spectral_conservation_and_entropy_diagnostics() -> None:
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(24),),
        axis_names=("x",),
        field_name="u",
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="unit-advection",
    )
    entropy = phx.equations.ConvexEntropyPair(
        system,
        lambda state: 0.5 * state[..., 0] ** 2,
        lambda state: state,
        lambda state, axis, args: 0.5 * state[..., 0] ** 2,
        lambda state: jnp.ones(state.shape[:-1], dtype="bool"),
        entropy_id="quadratic",
    )
    problem = phx.equations.ConservationProblemIR(
        "advection",
        "u",
        system,
        None,
    )
    method = phx.discretization.SpectralConservationMethodPlan(
        phx.discretization.PseudospectralMethodPlan(),
        flux_polynomial_degree=1,
        entropy_diagnostics=True,
    )
    compiled = phx.equations.compile_conservation_problem(
        problem,
        space,
        method,
        entropy_pair=entropy,
    )
    x = space.axes[0].nodes
    physical = jnp.sin(2.0 * jnp.pi * x)[..., None]
    state = space.project(physical)

    # ty: ignore[invalid-argument-type]
    residual, diagnostics = compiled.residual_with_diagnostics(0.0, state)
    physical_residual = space.reconstruct(residual)
    balance_terms = np.asarray(space.quadrature_weights[..., None] * physical_residual)
    expected_rate = np.asarray(
        [
            math.fsum(balance_terms[:, index].tolist())
            for index in range(balance_terms.shape[1])
        ]
    )
    np.testing.assert_allclose(
        # ty: ignore[unresolved-attribute]
        diagnostics.semidiscrete_integral_rate,
        expected_rate,
        rtol=8e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        # ty: ignore[unresolved-attribute]
        diagnostics.conservation_defect,
        expected_rate,
        rtol=8e-15,
        atol=0.0,
    )
    expected = -2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * x)

    assert jnp.allclose(
        space.reconstruct(residual)[..., 0],
        expected,
        rtol=1e-10,
        atol=1e-10,
    )
    # ty: ignore[unresolved-attribute]
    assert jnp.max(jnp.abs(diagnostics.conservation_defect)) < 1e-11
    # ty: ignore[unresolved-attribute]
    assert diagnostics.entropy is not None
    # ty: ignore[unresolved-attribute]
    assert jnp.abs(diagnostics.entropy.semidiscrete_entropy_rate) < 1e-11
    # ty: ignore[unresolved-attribute]
    assert diagnostics.entropy.admissible


def test_spectral_conservation_honors_widened_reduction_precision() -> None:
    precision = phx.discretization.SpectralPrecisionPolicy(
        jnp.float32,
        reduction_dtype=jnp.float64,
        certification_dtype=jnp.float64,
    )
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(32),),
        axis_names=("x",),
        field_name="u",
        precision=precision,
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="mixed-precision-unit-advection",
    )
    problem = phx.equations.ConservationProblemIR(
        "mixed-precision-advection",
        "u",
        system,
        None,
    )
    method = phx.discretization.SpectralConservationMethodPlan(
        phx.discretization.PseudospectralMethodPlan(),
        flux_polynomial_degree=1,
    )
    compiled = phx.equations.compile_conservation_problem(
        problem,
        space,
        method,
    )
    x = space.axes[0].nodes
    state = space.project(jnp.sin(2.0 * jnp.pi * x)[..., None])

    # ty: ignore[invalid-argument-type]
    residual, diagnostics = compiled.residual_with_diagnostics(0.0, state)
    physical_residual = precision.reduction(space.reconstruct(residual))
    weights = precision.reduction(space.quadrature_weights[..., None])
    balance_terms = np.asarray(weights * physical_residual)
    expected = np.asarray(
        [
            math.fsum(balance_terms[:, index].tolist())
            for index in range(balance_terms.shape[1])
        ]
    )

    # ty: ignore[unresolved-attribute]
    assert diagnostics.semidiscrete_integral_rate.dtype == jnp.float64
    # ty: ignore[unresolved-attribute]
    assert diagnostics.conservation_defect.dtype == jnp.float64
    np.testing.assert_allclose(
        # ty: ignore[unresolved-attribute]
        diagnostics.semidiscrete_integral_rate,
        expected,
        rtol=8e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        # ty: ignore[unresolved-attribute]
        diagnostics.conservation_defect,
        expected,
        rtol=8e-15,
        atol=0.0,
    )


def test_periodic_fourier_split_form_is_conservative_entropy_stable_and_chunk_invariant() -> (
    None
):
    from phydrax.discretization import spectral

    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(12),),
        axis_names=("x",),
        field_name="state",
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 2.0),))
    system = phx.equations.EulerSystem(1)
    pair = phx.equations.ideal_gas_euler_entropy_pair(system)
    problem = phx.equations.ConservationProblemIR("split-euler", "state", system, None)
    x = space.axes[0].nodes
    primitive = jnp.stack(
        (
            1.0 + 0.05 * jnp.sin(jnp.pi * x),
            0.2 * jnp.cos(jnp.pi * x),
            1.0 + 0.03 * jnp.sin(2.0 * jnp.pi * x),
        ),
        axis=-1,
    )
    state = space.project(system.primitive_to_conserved(primitive))

    def compiled(chunk: Any) -> Any:
        method = spectral.SpectralConservationMethodPlan(
            split_form=spectral.SpectralSplitFormPlan(
                phx.discretization.EntropyConservativeEulerFluxPlan(),
                pair_chunk_size=chunk,
                maximum_pair_workspace_bytes=1_000_000,
            ),
            entropy_diagnostics=True,
        )
        return phx.equations.compile_conservation_problem(
            problem, space, method, entropy_pair=pair
        )

    residual_a, diagnostics = compiled(5).residual_with_diagnostics(0.0, state)
    residual_b = compiled(64)(0.0, state)
    np.testing.assert_allclose(residual_a, residual_b, rtol=2e-12, atol=2e-12)
    assert jnp.max(jnp.abs(diagnostics.conservation_defect)) < 5e-12
    assert diagnostics.entropy.entropy_stable
    assert diagnostics.entropy.convective_entropy_defect < 5e-11
    assert diagnostics.entropy.pair_workspace_bytes > 0
    # ty: ignore[unresolved-attribute]
    assert space.periodic_cell.fully_periodic


def test_split_form_workspace_bound_includes_every_transverse_line() -> None:
    from phydrax.discretization import spectral

    space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(4),
            phx.discretization.FourierBasisPlan(6),
            phx.discretization.FourierBasisPlan(8),
        ),
        axis_names=("x", "y", "z"),
        field_name="state",
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 1.0),
        )
    )
    component_count = len(space.axes) + 2
    itemsize = np.dtype(space.plan.precision.physical_dtype).itemsize
    one_line_workspace = component_count * 3 * itemsize
    transverse_line_count = max(
        math.prod(space.modal_shape[:axis] + space.modal_shape[axis + 1 :])
        for axis in range(len(space.axes))
    )
    expected_workspace = transverse_line_count * one_line_workspace

    insufficient = spectral.SpectralConservationMethodPlan(
        split_form=spectral.SpectralSplitFormPlan(
            phx.discretization.EntropyConservativeEulerFluxPlan(),
            pair_chunk_size=1,
            maximum_pair_workspace_bytes=one_line_workspace,
        )
    )
    with np.testing.assert_raises_regex(ValueError, "workspace"):
        insufficient.prepare(space)

    prepared = spectral.SpectralConservationMethodPlan(
        split_form=spectral.SpectralSplitFormPlan(
            phx.discretization.EntropyConservativeEulerFluxPlan(),
            pair_chunk_size=1,
            maximum_pair_workspace_bytes=expected_workspace,
        )
    ).prepare(space)
    # ty: ignore[unresolved-attribute]
    assert prepared.split_form.report.pair_workspace_bytes == expected_workspace


def test_split_form_rejects_unsupported_flux_source_and_basis() -> None:
    from phydrax.discretization import spectral

    with np.testing.assert_raises(TypeError):
        # ty: ignore[invalid-argument-type]
        spectral.SpectralSplitFormPlan(phx.discretization.RusanovFluxPlan())
    polynomial = phx.discretization.TensorSpectralPlan(
        (phx.discretization.ChebyshevBasisPlan(8),)
    ).prepare((phx.discretization.AxisDomain.interval(-1.0, 1.0),))
    method = spectral.SpectralConservationMethodPlan(
        split_form=spectral.SpectralSplitFormPlan(
            phx.discretization.EntropyConservativeEulerFluxPlan()
        )
    )
    with np.testing.assert_raises(ValueError):
        method.prepare(polynomial)


def _dense_euler_split_reference(
    physical: NDArray[np.float64], lengths: tuple[float, ...], gamma: float
) -> NDArray[np.float64]:
    """FFT differentiation and quadrature of the logarithmic mean, test only."""
    density = physical[..., 0]
    velocity = physical[..., 1:-1] / density[..., None]
    pressure = (gamma - 1.0) * (
        physical[..., -1] - 0.5 * density * np.sum(velocity**2, axis=-1)
    )
    beta = density / (2.0 * pressure)
    nodes, weights = np.polynomial.legendre.leggauss(16)

    def log_mean(left: np.float64, right: np.float64) -> float:
        # L(a,b) = integral_0^1 a^t b^(1-t) dt avoids the implementation's
        # series/rational branch, including equal arguments.
        t = 0.5 * (nodes + 1.0)
        return float(0.5 * np.sum(weights * left**t * right ** (1.0 - t)))

    residual = np.zeros_like(physical)
    for axis, length in enumerate(lengths):
        count = physical.shape[axis]
        frequencies = np.fft.fftfreq(count, d=length / count)
        if count % 2 == 0:
            frequencies[count // 2] = 0.0
        derivative = np.fft.ifft(
            2j * np.pi * frequencies[:, None] * np.fft.fft(np.eye(count), axis=0),
            axis=0,
        ).real
        for target in np.ndindex(physical.shape[:-1]):
            for column in range(count):
                source = target[:axis] + (column,) + target[axis + 1 :]
                rho_mean = 0.5 * (density[target] + density[source])
                beta_mean = 0.5 * (beta[target] + beta[source])
                v_mean = 0.5 * (velocity[target] + velocity[source])
                mass = log_mean(density[target], density[source]) * v_mean[axis]
                momentum = mass * v_mean
                momentum[axis] += rho_mean / (2.0 * beta_mean)
                internal = 1.0 / (
                    2.0 * (gamma - 1.0) * log_mean(beta[target], beta[source])
                )
                v_square = 0.5 * (velocity[target] ** 2 + velocity[source] ** 2)
                energy = mass * (internal - 0.5 * np.sum(v_square)) + v_mean @ momentum
                flux = np.concatenate(
                    (np.asarray([mass]), momentum, np.asarray([energy]))
                )
                residual[target] -= 2.0 * derivative[target[axis], column] * flux
    return residual


@pytest.mark.parametrize("shape", [(7,), (8,), (5, 6)], ids=["odd", "even", "tensor"])
@pytest.mark.parametrize("chunk", [1, 5, 128], ids=["single", "ragged", "oversized"])
def test_split_form_matches_independent_dense_reference(
    shape: tuple[int, ...], chunk: int
) -> None:
    from phydrax.discretization import spectral

    lengths = tuple(2.0 + axis for axis in range(len(shape)))
    space = phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for count in shape),
        axis_names=tuple(f"x{axis}" for axis in range(len(shape))),
        field_name="state",
    ).prepare(
        tuple(phx.discretization.AxisDomain.periodic(0.0, length) for length in lengths)
    )
    system = phx.equations.EulerSystem(len(shape))
    coordinates = np.meshgrid(
        *(np.arange(count, dtype=np.float64) / count for count in shape), indexing="ij"
    )
    phase = sum((axis + 1) * coordinate for axis, coordinate in enumerate(coordinates))
    density = 1.0 + 0.17 * np.sin(2.0 * np.pi * phase)
    velocity = np.stack(
        tuple(
            0.23 * np.cos(2.0 * np.pi * coordinate + axis)
            for axis, coordinate in enumerate(coordinates)
        ),
        axis=-1,
    )
    pressure = 1.0 + 0.11 * np.cos(4.0 * np.pi * phase)
    physical = np.concatenate(
        (
            density[..., None],
            density[..., None] * velocity,
            (
                pressure / (system.gamma - 1.0)
                + 0.5 * density * np.sum(velocity**2, axis=-1)
            )[..., None],
        ),
        axis=-1,
    )
    dynamics = spectral.PreparedSpectralConservationDynamics(
        system,
        space,
        spectral.SpectralConservationMethodPlan(
            split_form=spectral.SpectralSplitFormPlan(
                phx.discretization.EntropyConservativeEulerFluxPlan(),
                pair_chunk_size=chunk,
            ),
            entropy_diagnostics=True,
        ),
        entropy_pair=phx.equations.ideal_gas_euler_entropy_pair(system),
    )
    state = space.project(jnp.asarray(physical))

    @eqx.filter_jit
    def evaluate(
        prepared: spectral.PreparedSpectralConservationDynamics, coefficients: Array
    ) -> Array:
        return prepared(jnp.asarray(0.0, dtype=jnp.float64), coefficients)

    result = evaluate(dynamics, state)
    np.testing.assert_allclose(
        space.reconstruct(result),
        _dense_euler_split_reference(physical, lengths, system.gamma),
        rtol=2e-11,
        atol=2e-11,
    )
    split = dynamics.method.split_form
    assert split is not None
    expected_pairs = sum(
        count * (count - 1) // 2 * math.prod(shape[:axis] + shape[axis + 1 :])
        for axis, count in enumerate(shape)
    )
    assert split.report.pair_count == expected_pairs
    assert split.report.entropy_stable
    assert split.report.skew_sbp_defect < split.plan.certification_tolerance


@pytest.mark.parametrize("count", [32, 64, 128], ids=["32", "64", "128"])
def test_split_form_retained_storage_is_linear(count: int) -> None:
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(count),),
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 2.0),))
    prepared = phx.discretization.SpectralConservationMethodPlan(
        split_form=phx.discretization.SpectralSplitFormPlan(
            phx.discretization.EntropyConservativeEulerFluxPlan(), pair_chunk_size=5
        )
    ).prepare(space)
    split = prepared.split_form
    assert split is not None
    retained_bytes = sum(
        leaf.nbytes
        for leaf in jax.tree_util.tree_leaves(split)
        if isinstance(leaf, Array)
    )
    assert retained_bytes <= 32 * count
    assert split.report.pair_workspace_bytes == 5 * 3 * 3 * 8


@pytest.mark.parametrize("chunk", [3, 5, 32], ids=["exact", "ragged", "oversized"])
def test_split_form_flux_observes_lexicographic_pairs(
    chunk: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from phydrax.discretization import spectral

    count = 7
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(count),),
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 2.0),))
    system = phx.equations.EulerSystem(1)
    physical = system.primitive_to_conserved(
        jnp.stack(
            (
                1.0 + jnp.arange(count, dtype=jnp.float64) / 10.0,
                jnp.full(count, 0.2, dtype=jnp.float64),
                jnp.ones(count, dtype=jnp.float64),
            ),
            axis=-1,
        )
    )
    observed: list[tuple[int, int]] = []

    def record(left: NDArray[np.float64], right: NDArray[np.float64]) -> None:
        rows = np.rint((left[..., 0] - 1.0) * 10.0).astype(np.int64)
        columns = np.rint((right[..., 0] - 1.0) * 10.0).astype(np.int64)
        observed.extend(zip(rows.tolist(), columns.tolist(), strict=True))

    original = phx.discretization.EntropyConservativeEulerFluxPlan.two_point_flux

    def recording_flux(
        self: phx.discretization.EntropyConservativeEulerFluxPlan,
        euler: phx.equations.EulerSystem,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
    ) -> Array:
        jax.debug.callback(record, left, right, ordered=True)
        return original(self, euler, left, right, axis, args)

    monkeypatch.setattr(
        phx.discretization.EntropyConservativeEulerFluxPlan,
        "two_point_flux",
        recording_flux,
    )
    dynamics = spectral.PreparedSpectralConservationDynamics(
        system,
        space,
        spectral.SpectralConservationMethodPlan(
            split_form=spectral.SpectralSplitFormPlan(
                phx.discretization.EntropyConservativeEulerFluxPlan(),
                pair_chunk_size=chunk,
            )
        ),
        entropy_pair=phx.equations.ideal_gas_euler_entropy_pair(system),
    )
    result = dynamics(jnp.asarray(0.0, dtype=jnp.float64), space.project(physical))
    result.block_until_ready()
    assert observed == [
        (row, column) for row in range(count) for column in range(row + 1, count)
    ]
