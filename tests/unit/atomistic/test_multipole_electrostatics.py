"""Native point and Gaussian multipole electrostatics through degree two.

Every reference is built from different mathematics than the implementation:
jax autodiff of the bare radial kernels contracted with the primitive
``O_i(-grad) O_j(grad)`` operators, explicit point-charge clouds, NumPy
quadrature of Gaussian densities, closed forms, the host Ewald sums of
``tests/_support`` and central finite differences.
"""

import itertools
import math
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import pytest

import phydrax as phx
from phydrax.atomistic._multipole_basis import (
    CartesianMultipoles,
    gaussian_coefficients,
    gaussian_density_moments,
    gaussian_self_energy,
    GaussianMultipoleWidths,
    pack_multipoles,
    shift_multipole_origin,
    unpack_multipoles,
)
from phydrax.atomistic._multipole_electrostatics import (
    EwaldMultipoleRoute,
    ExplicitImagePairs,
    FiniteMultipoleRoute,
    MultipoleElectrostaticPotential,
    MultipoleElectrostaticsDefinition,
    MultipoleElectrostaticsEvidence,
    MultipoleElectrostaticsPlan,
    MultipoleSiteKind,
    MultipoleSiteSet,
    ParticleMeshMultipoleRoute,
)
from phydrax.atomistic._reciprocal_mesh import (
    ReciprocalBudget,
    ReciprocalMeshPlan,
    ReciprocalModeSetPlan,
)
from phydrax.nn.operator.representations import o3_irrep_action
from tests._support.atomistic_bundles import bundle_state, prepare_program
from tests._support.atomistic_long_range_reference import (
    central_difference,
    converged_ewald_energy,
)


TRICLINIC = np.asarray([[6.0, 0.0, 0.0], [1.0, 6.5, 0.0], [0.5, 1.0, 7.0]])
FINITE_POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [2.1, 0.3, -0.4], [-0.5, 1.9, 0.8], [0.7, -0.6, 2.3]]
)
# Includes a close pair (0, 4) so the Gaussian kernel is probed at beta R < 1.
GAUSSIAN_POSITIONS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [2.1, 0.3, -0.4],
        [-0.5, 1.9, 0.8],
        [0.7, -0.6, 2.3],
        [0.3, 0.2, -0.36],
    ]
)
GAUSSIAN_SIGMAS = np.asarray([0.4, 0.3, 0.5, 0.35, 0.45])
_SQRT_PI = math.sqrt(math.pi)
_UNIQUE_ROWS = [0, 0, 0, 1, 1, 2]
_UNIQUE_COLUMNS = [0, 1, 2, 1, 2, 2]


# --- independent helpers -------------------------------------------------------


def _configuration(
    seed: int, vectors: np.ndarray, count: int, minimum: float
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shifts = np.asarray(list(itertools.product((-1, 0, 1), repeat=3))) @ vectors
    points: list[np.ndarray] = []
    while len(points) < count:
        candidate = rng.uniform(0.0, 1.0, 3) @ vectors
        if all(
            np.min(np.linalg.norm(candidate - point + shifts, axis=1)) > minimum
            for point in points
        ):
            points.append(candidate)
    return np.asarray(points)


def _neutral(seed: int, count: int) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=count)
    return values - values.mean()


def _random_moments(
    seed: int,
    count: int,
    *,
    neutral: bool = True,
    charge: float = 1.0,
    dipole: float = 0.3,
    second: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q = charge * rng.normal(size=count)
    if neutral:
        q = q - q.mean()
    mu = dipole * rng.normal(size=(count, 3))
    raw = rng.normal(size=(count, 3, 3))
    theta = second * 0.5 * (raw + np.swapaxes(raw, -1, -2))
    return q, mu, theta


def _traceless(theta: np.ndarray) -> np.ndarray:
    trace = np.trace(theta, axis1=-2, axis2=-1)
    return theta - trace[..., None, None] * np.eye(3) / 3.0


def _rotation(seed: int, *, proper: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix, upper = np.linalg.qr(rng.normal(size=(3, 3)))
    matrix = matrix * np.sign(np.diag(upper))[None, :]
    if np.linalg.det(matrix) < 0.0:
        matrix[:, 0] = -matrix[:, 0]
    return matrix if proper else -matrix


def _multipoles(
    q: Any, mu: Any = None, theta: Any = None, *, degree: int
) -> CartesianMultipoles:
    return CartesianMultipoles(
        jnp.asarray(q, dtype=jnp.float64),
        None if mu is None else jnp.asarray(mu, dtype=jnp.float64),
        None if theta is None else jnp.asarray(theta, dtype=jnp.float64),
        maximum_degree=degree,
    )


def _site_widths(values: Any) -> GaussianMultipoleWidths:
    values = np.asarray(values, dtype=np.float64)
    return GaussianMultipoleWidths(values, np.arange(values.shape[0]))


def _class_widths(count: int) -> GaussianMultipoleWidths:
    return GaussianMultipoleWidths(
        np.asarray([0.2, 0.3]), np.asarray([index % 2 for index in range(count)])
    )


def _sites(
    moments: CartesianMultipoles, widths: GaussianMultipoleWidths | None = None
) -> MultipoleSiteSet:
    return MultipoleSiteSet(
        moments,
        active=jnp.ones((moments.site_capacity,), dtype=jnp.bool_),
        widths=widths,
    )


def _system(count: int, *, cell: Any = None) -> Any:
    return phx.atomistic.AtomisticSystemPlan(
        np.asarray(list(range(count)), dtype=np.int64),
        np.asarray([1] * count, dtype=np.int64),
        np.asarray([1.0] * count, dtype=np.float64),
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=np.asarray([0] * count, dtype=np.int64),
        cell=cell,
    ).prepare()


def _definition(
    kind: MultipoleSiteKind, degree: int, **kwargs: Any
) -> MultipoleElectrostaticsDefinition:
    kwargs.setdefault("coulomb_constant", 1.0)
    return MultipoleElectrostaticsDefinition(kind, degree, **kwargs)


def _finite_engine(
    kind: MultipoleSiteKind, degree: int, count: int, **kwargs: Any
) -> Any:
    return MultipoleElectrostaticsPlan(
        _definition(kind, degree, **kwargs), FiniteMultipoleRoute()
    ).prepare(count)


def _ewald_engine(
    kind: MultipoleSiteKind,
    degree: int,
    cell: Any,
    count: int,
    *,
    alpha: float,
    cutoff: float,
    extents: int,
    route_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    route = EwaldMultipoleRoute(
        alpha,
        cutoff,
        ReciprocalModeSetPlan((extents, extents, extents)),
        **(route_kwargs or {}),
    )
    return MultipoleElectrostaticsPlan(
        _definition(kind, degree, **kwargs), route
    ).prepare(count, cell=cell)


def _pme_engine(
    kind: MultipoleSiteKind,
    degree: int,
    system: Any,
    *,
    alpha: float,
    cutoff: float,
    grid: int,
    mesh_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    route = ParticleMeshMultipoleRoute(
        alpha, cutoff, ReciprocalMeshPlan((grid, grid, grid), **(mesh_kwargs or {}))
    )
    return MultipoleElectrostaticsPlan(
        _definition(kind, degree, **kwargs), route
    ).prepare(
        system.capacity, cell=system.cell, particles=system.particle_discretization()
    )


# --- autodiff radial references -------------------------------------------------


def _radial_derivatives(function: Any) -> Any:
    """Jitted ``(f, grad f, grad^2 f, grad^3 f, grad^4 f)`` at ``R``."""
    first = jax.grad(function)
    second = jax.jacfwd(first)
    third = jax.jacfwd(second)
    fourth = jax.jacfwd(third)
    return jax.jit(
        lambda r, *parameters: (
            function(r, *parameters),
            first(r, *parameters),
            second(r, *parameters),
            third(r, *parameters),
            fourth(r, *parameters),
        )
    )


def _inverse_distance(r: jax.Array) -> jax.Array:
    return 1.0 / jnp.sqrt(jnp.sum(r * r))


def _erf_over_distance(r: jax.Array, beta: jax.Array) -> jax.Array:
    distance = jnp.sqrt(jnp.sum(r * r))
    return jsp.erf(beta * distance) / distance


def _erf_over_distance_origin_series(r: jax.Array, beta: jax.Array) -> jax.Array:
    """Analytic ``erf(beta r)/r`` as a polynomial in ``s = r^2`` (exact through s^3)."""
    s = jnp.sum(r * r)
    x2 = beta * beta * s
    return 2.0 * beta / _SQRT_PI * (1.0 - x2 / 3.0 + x2**2 / 10.0 - x2**3 / 42.0)


_COULOMB_TENSORS = _radial_derivatives(_inverse_distance)
_ERF_TENSORS = _radial_derivatives(_erf_over_distance)
_ERF_ORIGIN_TENSORS = _radial_derivatives(_erf_over_distance_origin_series)


def _operator_pair(tensors: tuple, left: tuple, right: tuple) -> float:
    """``O_L(-grad) O_R(grad) f`` from explicit Cartesian derivative tensors.

    ``O(grad) = q - mu . grad + 1/2 Theta : grad grad``.
    """
    d0, d1, d2, d3, d4 = (np.asarray(item, dtype=np.float64) for item in tensors)
    qi, mi, ti = left
    qj, mj, tj = right
    return float(
        qi * qj * d0
        + (qj * mi - qi * mj) @ d1
        + 0.5 * np.sum((qi * tj + qj * ti) * d2)
        - mi @ d2 @ mj
        + 0.5 * np.einsum("a,bc,abc->", mi, tj, d3)
        - 0.5 * np.einsum("ab,c,abc->", ti, mj, d3)
        + 0.25 * np.einsum("ab,cd,abcd->", ti, tj, d4)
    )


def _reference_pair_energy(
    positions: np.ndarray,
    q: np.ndarray,
    mu: np.ndarray,
    theta: np.ndarray,
    tensors: Any,
    constant: float,
) -> float:
    total = 0.0
    for i, j in itertools.combinations(range(len(q)), 2):
        displacement = jnp.asarray(positions[i] - positions[j], dtype=jnp.float64)
        total += _operator_pair(
            tensors(i, j, displacement), (q[i], mu[i], theta[i]), (q[j], mu[j], theta[j])
        )
    return constant * total


def _gaussian_pair_tensors(sigma: np.ndarray) -> Any:
    def tensors(i: int, j: int, displacement: jax.Array) -> tuple:
        beta = 1.0 / math.sqrt(2.0 * (sigma[i] ** 2 + sigma[j] ** 2))
        return _ERF_TENSORS(displacement, jnp.asarray(beta))

    return tensors


def _gaussian_self_reference(
    q: np.ndarray, mu: np.ndarray, theta: np.ndarray, sigma: np.ndarray, constant: float
) -> float:
    total = 0.0
    origin = jnp.zeros((3,), dtype=jnp.float64)
    for i in range(len(q)):
        beta = 1.0 / (2.0 * sigma[i])
        site = (q[i], mu[i], theta[i])
        total += 0.5 * _operator_pair(
            _ERF_ORIGIN_TENSORS(origin, jnp.asarray(beta)), site, site
        )
    return constant * total


# --- 1. packed bases -------------------------------------------------------------


def test_cartesian_primitive_pack_unpack_round_trip() -> None:
    q, mu, theta = _random_moments(1, 4)
    moments = _multipoles(q, mu, theta, degree=2)
    packed = pack_multipoles(moments, "cartesian-primitive", trace="retain")
    assert packed.removed_trace is None
    expected = np.column_stack((q, mu, theta[:, _UNIQUE_ROWS, _UNIQUE_COLUMNS]))
    np.testing.assert_array_equal(np.asarray(packed.values), expected)
    restored = unpack_multipoles(packed.values, "cartesian-primitive", 2)
    np.testing.assert_array_equal(np.asarray(restored.charges), q)
    np.testing.assert_array_equal(np.asarray(restored.dipoles), mu)
    np.testing.assert_array_equal(np.asarray(restored.second_moments), theta)
    dipolar = pack_multipoles(
        _multipoles(q, mu, degree=1), "cartesian-primitive", trace="retain"
    )
    np.testing.assert_array_equal(np.asarray(dipolar.values), np.column_stack((q, mu)))


def test_native_real_orthonormal_layout_is_o3_equivariant() -> None:
    q, mu, theta = _random_moments(2, 5)
    moments = _multipoles(q, mu, theta, degree=2)
    packed = np.asarray(
        pack_multipoles(moments, "native-real-orthonormal", trace="project").values
    )
    np.testing.assert_array_equal(packed[:, 0], q)
    np.testing.assert_array_equal(packed[:, 1:4], mu[:, [1, 2, 0]])
    # Orthonormal coordinates of the traceless tensor preserve its Frobenius norm.
    np.testing.assert_allclose(
        np.sum(packed[:, 4:9] ** 2, axis=-1),
        np.sum(_traceless(theta) ** 2, axis=(-2, -1)),
        rtol=1.0e-12,
    )
    for orthogonal in (_rotation(3), _rotation(4, proper=False)):
        rotated = np.asarray(
            pack_multipoles(
                moments.rotated(orthogonal), "native-real-orthonormal", trace="project"
            ).values
        )
        vector_action = np.asarray(o3_irrep_action(jnp.asarray(orthogonal), 1, -1))
        tensor_action = np.asarray(o3_irrep_action(jnp.asarray(orthogonal), 2, 1))
        np.testing.assert_allclose(rotated[:, 0], packed[:, 0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            rotated[:, 1:4], packed[:, 1:4] @ vector_action.T, atol=1.0e-12
        )
        np.testing.assert_allclose(
            rotated[:, 4:9], packed[:, 4:9] @ tensor_action.T, atol=1.0e-12
        )


def test_source_e3nn_component_layout_matches_explicit_y_polar_harmonics() -> None:
    q, mu, theta = _random_moments(5, 6)
    traceless = _traceless(theta)
    moments = _multipoles(q, mu, traceless, degree=2)
    packed = pack_multipoles(moments, "nvalchemi-ops-e3nn-component", trace="project")
    values = np.asarray(packed.values)
    np.testing.assert_allclose(np.asarray(packed.removed_trace), 0.0, atol=1.0e-14)
    np.testing.assert_array_equal(values[:, 0], q)
    np.testing.assert_array_equal(values[:, 1:4], mu[:, [1, 2, 0]])
    s5, s15 = math.sqrt(5.0), math.sqrt(15.0)
    rng = np.random.default_rng(6)
    for _ in range(4):
        direction = rng.normal(size=(6, 3))
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        x, y, z = direction.T
        harmonics = np.stack(
            (
                s15 * x * z,
                s15 * x * y,
                0.5 * s5 * (3.0 * y * y - 1.0),
                s15 * y * z,
                0.5 * s15 * (z * z - x * x),
            ),
            axis=-1,
        )
        np.testing.assert_allclose(
            np.sum(values[:, 4:9] * harmonics, axis=-1),
            np.einsum("na,nab,nb->n", direction, traceless, direction),
            rtol=1.0e-12,
            atol=1.0e-14,
        )
    restored = unpack_multipoles(
        packed.values, "nvalchemi-ops-e3nn-component", 2, trace="zero"
    )
    np.testing.assert_allclose(
        np.asarray(restored.second_moments), traceless, atol=1.0e-13
    )
    np.testing.assert_allclose(np.asarray(restored.dipoles), mu, atol=0.0)


# --- 2. explicit trace handling ------------------------------------------------------


def test_spherical_packing_requires_explicit_trace_handling() -> None:
    q, mu, theta = _random_moments(11, 3)
    moments = _multipoles(q, mu, theta, degree=2)
    for basis in ("native-real-orthonormal", "nvalchemi-ops-e3nn-component"):
        with pytest.raises(ValueError, match="trace='project'"):
            pack_multipoles(moments, basis, trace="retain")
        packed = pack_multipoles(moments, basis, trace="project")
        np.testing.assert_allclose(
            np.asarray(packed.removed_trace),
            np.trace(theta, axis1=-2, axis2=-1),
            rtol=1.0e-14,
            atol=1.0e-15,
        )
        with pytest.raises(ValueError, match="explicit primitive trace"):
            unpack_multipoles(packed.values, basis, 2)
        restored = unpack_multipoles(packed.values, basis, 2, trace=packed.removed_trace)
        np.testing.assert_allclose(np.asarray(restored.charges), q, atol=0.0)
        np.testing.assert_allclose(np.asarray(restored.dipoles), mu, atol=1.0e-15)
        np.testing.assert_allclose(
            np.asarray(restored.second_moments), theta, rtol=1.0e-13, atol=1.0e-14
        )
        zero = unpack_multipoles(packed.values, basis, 2, trace="zero")
        np.testing.assert_allclose(
            np.asarray(zero.second_moments), _traceless(theta), atol=1.0e-14
        )
    with pytest.raises(ValueError, match="real-spherical"):
        pack_multipoles(moments, "cartesian-primitive", trace="project")
    with pytest.raises(ValueError, match="real-spherical"):
        pack_multipoles(
            _multipoles(q, mu, degree=1), "native-real-orthonormal", trace="project"
        )
    cartesian = pack_multipoles(moments, "cartesian-primitive", trace="retain")
    with pytest.raises(ValueError, match="only applies"):
        unpack_multipoles(cartesian.values, "cartesian-primitive", 2, trace=np.zeros(3))


# --- 3. origin shift and point-charge clouds ------------------------------------------


def _cloud_moments(
    cloud: np.ndarray, weights: np.ndarray, origin: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    relative = cloud - origin[:, None, :]
    return (
        np.sum(weights, axis=1),
        np.einsum("np,npa->na", weights, relative),
        np.einsum("np,npa,npb->nab", weights, relative, relative),
    )


def test_origin_shift_matches_point_charge_cloud_moments() -> None:
    rng = np.random.default_rng(13)
    centres = rng.normal(size=(3, 3))
    cloud = centres[:, None, :] + 0.4 * rng.normal(size=(3, 5, 3))
    weights = rng.normal(size=(3, 5))
    displacement = rng.normal(size=(3, 3))
    first = _cloud_moments(cloud, weights, centres)
    second = _cloud_moments(cloud, weights, centres + displacement)
    shifted = shift_multipole_origin(
        _multipoles(*first, degree=2), jnp.asarray(displacement)
    )
    np.testing.assert_allclose(np.asarray(shifted.charges), second[0], atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(shifted.dipoles), second[1], atol=1.0e-13)
    np.testing.assert_allclose(
        np.asarray(shifted.second_moments), second[2], atol=1.0e-12
    )
    with pytest.raises(ValueError, match="degree-2 primitive"):
        shift_multipole_origin(_multipoles(first[0], first[1], degree=1), displacement)


def test_point_charge_clouds_reproduce_degree_two_multipole_energy() -> None:
    rng = np.random.default_rng(17)
    centres = 2.0 * FINITE_POSITIONS
    extent = 0.02
    cloud = centres[:, None, :] + extent * rng.normal(size=(4, 4, 3))
    weights = rng.normal(size=(4, 4))
    exact = 0.0
    for i, j in itertools.combinations(range(4), 2):
        separation = cloud[i][:, None, :] - cloud[j][None, :, :]
        exact += float(
            np.sum(
                weights[i][:, None]
                * weights[j][None, :]
                / np.linalg.norm(separation, axis=-1)
            )
        )
    q, mu, theta = _cloud_moments(cloud, weights, centres)
    position = jnp.asarray(centres)
    quadrupolar = _finite_engine("point", 2, 4).energy(
        position, _sites(_multipoles(q, mu, theta, degree=2))
    )
    dipolar = _finite_engine("point", 1, 4).energy(
        position, _sites(_multipoles(q, mu, degree=1))
    )
    assert bool(quadrupolar.successful) and bool(dipolar.successful)
    second_error = abs(float(quadrupolar.energy) - exact)
    first_error = abs(float(dipolar.energy) - exact)
    assert second_error < 0.05 * first_error
    assert second_error < 1.0e-6 * max(1.0, abs(exact))


# --- 4. Gaussian basis ------------------------------------------------------------------


def test_gaussian_density_moments_match_quadrature_and_invert() -> None:
    q, mu, theta = _random_moments(19, 2, neutral=False)
    sigma = np.asarray([0.7, 0.45])
    widths = _site_widths(sigma)
    coefficients = _multipoles(q, mu, theta, degree=2)
    density = gaussian_density_moments(coefficients, widths)
    for site in range(2):
        s = sigma[site]
        axis = np.arange(-45, 46) * (s / 5.0)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        r = np.stack((x, y, z), axis=-1)
        r2 = np.sum(r * r, axis=-1)
        g = np.exp(-r2 / (2.0 * s * s)) / (2.0 * np.pi * s * s) ** 1.5
        # rho = (q - mu.grad + 1/2 Theta:grad grad) g with analytic derivatives.
        rho = g * (
            q[site]
            + np.einsum("...a,a->...", r, mu[site]) / s**2
            + 0.5
            * (
                np.einsum("...a,ab,...b->...", r, theta[site], r) / s**4
                - np.trace(theta[site]) / s**2
            )
        )
        volume = (s / 5.0) ** 3
        np.testing.assert_allclose(np.sum(rho) * volume, q[site], atol=1.0e-10)
        weighted = rho.reshape(-1) * volume
        points = r.reshape(-1, 3)
        np.testing.assert_allclose(weighted @ points, mu[site], atol=1.0e-10)
        second = (points * weighted[:, None]).T @ points
        np.testing.assert_allclose(
            np.asarray(density.second_moments[site]), second, atol=1.0e-10
        )
        np.testing.assert_allclose(
            second, theta[site] + q[site] * s * s * np.eye(3), atol=1.0e-10
        )
    restored = gaussian_coefficients(density, widths)
    np.testing.assert_allclose(np.asarray(restored.second_moments), theta, atol=1.0e-14)
    np.testing.assert_array_equal(np.asarray(restored.charges), q)
    exponent = GaussianMultipoleWidths(
        np.asarray([2.0]), np.asarray([0]), convention="gaussian-exponent"
    )
    np.testing.assert_allclose(np.asarray(exponent.site_widths()), [0.5], rtol=1.0e-15)


def test_gaussian_self_energy_matches_closed_forms() -> None:
    rng = np.random.default_rng(23)
    sigma = np.asarray([0.3, 0.45, 0.6, 0.5])
    constant = 1.7
    q = np.asarray([0.8, 0.0, 0.0, -0.6])
    mu = np.zeros((4, 3))
    mu[1] = [0.3, -0.2, 0.5]
    mu[3] = [0.1, 0.4, -0.3]
    theta = np.zeros((4, 3, 3))
    raw = rng.normal(size=(2, 3, 3))
    theta[2:] = _traceless(0.5 * (raw + np.swapaxes(raw, -1, -2)))
    energy = np.asarray(
        gaussian_self_energy(
            _multipoles(q, mu, theta, degree=2), _site_widths(sigma), constant
        )
    )
    dipole2 = np.sum(mu * mu, axis=-1)
    frobenius2 = np.sum(theta * theta, axis=(-2, -1))
    expected = constant * (
        q * q / (2.0 * _SQRT_PI * sigma)
        + dipole2 / (12.0 * _SQRT_PI * sigma**3)
        + frobenius2 / (80.0 * _SQRT_PI * sigma**5)
    )
    np.testing.assert_allclose(energy, expected, rtol=1.0e-13)
    coupling = 4.0 * math.pi * constant
    pinned = (
        coupling * q * q / (8.0 * math.pi**1.5 * sigma)
        + coupling * dipole2 / (48.0 * math.pi**1.5 * sigma**3)
        + coupling * frobenius2 / (320.0 * math.pi**1.5 * sigma**5)
    )
    np.testing.assert_allclose(energy, pinned, rtol=1.0e-13)


# --- 5. finite point and Gaussian multipoles ---------------------------------------------


def test_finite_point_multipole_energy_matches_autodiff_reference() -> None:
    q, mu, theta = _random_moments(29, 4, neutral=False)
    constant = 1.3
    engine = _finite_engine("point", 2, 4, coulomb_constant=constant)
    result = engine.energy(
        jnp.asarray(FINITE_POSITIONS), _sites(_multipoles(q, mu, theta, degree=2))
    )
    reference = _reference_pair_energy(
        FINITE_POSITIONS,
        q,
        mu,
        theta,
        lambda i, j, displacement: _COULOMB_TENSORS(displacement),
        constant,
    )
    assert bool(result.successful)
    assert result.evidence.method == "point-multipole-finite-all-pairs"
    np.testing.assert_allclose(float(result.energy), reference, rtol=1.0e-11)
    np.testing.assert_allclose(float(result.evidence.intrasite_energy), 0.0, atol=0.0)


@pytest.mark.parametrize("intrasite", ["exclude", "include"])
def test_finite_gaussian_multipole_energy_matches_autodiff_reference(
    intrasite: str,
) -> None:
    q, mu, theta = _random_moments(31, 5, neutral=False)
    constant = 0.9
    engine = _finite_engine(
        "gaussian", 2, 5, coulomb_constant=constant, intrasite=intrasite
    )
    result = engine.energy(
        jnp.asarray(GAUSSIAN_POSITIONS),
        _sites(_multipoles(q, mu, theta, degree=2), _site_widths(GAUSSIAN_SIGMAS)),
    )
    pair = _reference_pair_energy(
        GAUSSIAN_POSITIONS,
        q,
        mu,
        theta,
        _gaussian_pair_tensors(GAUSSIAN_SIGMAS),
        constant,
    )
    own = _gaussian_self_reference(q, mu, theta, GAUSSIAN_SIGMAS, constant)
    assert bool(result.successful)
    if intrasite == "include":
        np.testing.assert_allclose(float(result.energy), pair + own, rtol=1.0e-10)
        np.testing.assert_allclose(
            float(result.evidence.intrasite_energy), own, rtol=1.0e-11
        )
        # Traceless second moments: the self energy collapses to the closed forms.
        traceless = _multipoles(q, mu, _traceless(theta), degree=2)
        closed = gaussian_self_energy(traceless, _site_widths(GAUSSIAN_SIGMAS), constant)
        np.testing.assert_allclose(
            float(jnp.sum(closed)),
            _gaussian_self_reference(q, mu, _traceless(theta), GAUSSIAN_SIGMAS, constant),
            rtol=1.0e-12,
        )
    else:
        np.testing.assert_allclose(float(result.energy), pair, rtol=1.0e-10)
        np.testing.assert_allclose(float(result.evidence.intrasite_energy), 0.0, atol=0.0)


# --- 6. Gaussian-to-point limit -------------------------------------------------------------


def test_gaussian_energy_reduces_to_point_energy_for_vanishing_widths() -> None:
    q, mu, theta = _random_moments(37, 4)
    moments = _multipoles(q, mu, theta, degree=2)
    tiny = _site_widths(np.full((4,), 1.0e-4))
    position = jnp.asarray(FINITE_POSITIONS)
    point = _finite_engine("point", 2, 4).energy(position, _sites(moments))
    gaussian = _finite_engine("gaussian", 2, 4).energy(position, _sites(moments, tiny))
    assert bool(point.successful) and bool(gaussian.successful)
    np.testing.assert_allclose(float(gaussian.energy), float(point.energy), rtol=1.0e-10)

    cell = phx.discretization.PeriodicCell(TRICLINIC)
    periodic = jnp.asarray(_configuration(41, TRICLINIC, 4, 1.0))
    vectors = jnp.asarray(TRICLINIC)
    point = _ewald_engine("point", 2, cell, 4, alpha=1.4, cutoff=4.5, extents=10).energy(
        periodic, _sites(moments), cell_vectors=vectors
    )
    gaussian = _ewald_engine(
        "gaussian", 2, cell, 4, alpha=1.4, cutoff=4.5, extents=10
    ).energy(periodic, _sites(moments, tiny), cell_vectors=vectors)
    assert bool(point.successful) and bool(gaussian.successful)
    np.testing.assert_allclose(
        float(gaussian.energy), float(point.energy), rtol=1.0e-6, atol=1.0e-9
    )


# --- 7. periodic point charges -------------------------------------------------------------


def _charge_engine(route: str, system: Any, **kwargs: Any) -> Any:
    if route == "ewald":
        return _ewald_engine(
            "point",
            0,
            system.cell,
            system.capacity,
            alpha=1.6,
            cutoff=3.2,
            extents=20,
            **kwargs,
        )
    return _pme_engine(
        "point",
        0,
        system,
        alpha=1.0,
        cutoff=5.0,
        grid=48,
        mesh_kwargs={"spline_degree": 4},
        **kwargs,
    )


_CHARGE_TOLERANCE = {"ewald": 1.0e-8, "pme": 2.0e-5}


@pytest.mark.parametrize("route", ["ewald", "pme"])
def test_periodic_point_charges_match_converged_ewald_reference(route: str) -> None:
    positions = _configuration(3, TRICLINIC, 6, 0.9)
    charges = _neutral(5, 6)
    system = _system(6, cell=phx.discretization.PeriodicCell(TRICLINIC))
    reference, history = converged_ewald_energy(
        positions, charges, TRICLINIC, 1.2, maximum_shells=16
    )
    assert len(history) >= 2
    result = _charge_engine(route, system).energy(
        jnp.asarray(positions),
        _sites(_multipoles(charges, degree=0)),
        cell_vectors=jnp.asarray(TRICLINIC),
    )
    assert bool(result.successful) and bool(result.evidence.neutral)
    assert abs(float(result.energy) - reference) < _CHARGE_TOLERANCE[route] * max(
        1.0, abs(reference)
    )


@pytest.mark.parametrize("route", ["ewald", "pme"])
def test_charged_cell_uniform_background_and_neutrality_refusal(route: str) -> None:
    positions = _configuration(7, TRICLINIC, 4, 1.0)
    charges = np.asarray([1.0, -0.5, 0.25, 0.75])
    system = _system(4, cell=phx.discretization.PeriodicCell(TRICLINIC))
    sites = _sites(_multipoles(charges, degree=0))
    position = jnp.asarray(positions)
    vectors = jnp.asarray(TRICLINIC)
    reference, _ = converged_ewald_energy(
        positions, charges, TRICLINIC, 1.2, background=True, maximum_shells=16
    )
    background = _charge_engine(route, system, neutrality="uniform-background").energy(
        position, sites, cell_vectors=vectors
    )
    assert bool(background.successful)
    assert not bool(background.evidence.neutral)
    assert float(background.evidence.background_energy) < 0.0
    assert abs(float(background.energy) - reference) < _CHARGE_TOLERANCE[route] * max(
        1.0, abs(reference)
    )
    refused = _charge_engine(route, system).energy(position, sites, cell_vectors=vectors)
    assert not bool(refused.successful)
    assert bool(jnp.isnan(refused.energy))
    assert not bool(refused.evidence.neutral)
    assert bool(refused.evidence.cell_successful)
    np.testing.assert_allclose(float(refused.evidence.total_charge), 1.5, atol=1.0e-12)


# --- 8. periodic degree two: alpha independence and mesh refinement ---------------------------


@pytest.mark.parametrize(
    ("kind", "alphas", "extents"),
    [("point", (0.9, 1.1), 14), ("gaussian", (1.3, 1.5), 17)],
)
def test_periodic_degree_two_ewald_is_alpha_independent(
    kind: MultipoleSiteKind, alphas: tuple[float, float], extents: int
) -> None:
    count = 5
    positions = jnp.asarray(_configuration(43, TRICLINIC, count, 1.0))
    q, mu, theta = _random_moments(47, count, charge=0.8, dipole=0.3, second=0.15)
    widths = _class_widths(count) if kind == "gaussian" else None
    sites = _sites(_multipoles(q, mu, theta, degree=2), widths)
    cell = phx.discretization.PeriodicCell(TRICLINIC)
    results = [
        _ewald_engine(
            kind, 2, cell, count, alpha=alpha, cutoff=6.0, extents=extents
        ).energy(positions, sites, cell_vectors=jnp.asarray(TRICLINIC))
        for alpha in alphas
    ]
    assert all(bool(result.successful) for result in results)
    first, second = (float(result.energy) for result in results)
    np.testing.assert_allclose(first, second, rtol=1.0e-8, atol=1.0e-10)
    # The split itself moves: the agreement is not a trivially shared component.
    assert (
        abs(
            float(results[0].evidence.real_energy)
            - float(results[1].evidence.real_energy)
        )
        > 1.0e-6
    )


@pytest.mark.parametrize("kind", ["point", "gaussian"])
def test_particle_mesh_degree_two_converges_to_explicit_modes(
    kind: MultipoleSiteKind,
) -> None:
    count = 5
    positions = jnp.asarray(_configuration(53, TRICLINIC, count, 1.0))
    q, mu, theta = _random_moments(59, count, charge=0.8, dipole=0.3, second=0.15)
    widths = _class_widths(count) if kind == "gaussian" else None
    sites = _sites(_multipoles(q, mu, theta, degree=2), widths)
    system = _system(count, cell=phx.discretization.PeriodicCell(TRICLINIC))
    vectors = jnp.asarray(TRICLINIC)
    reference = _ewald_engine(
        kind, 2, system.cell, count, alpha=0.9, cutoff=5.0, extents=11
    ).energy(positions, sites, cell_vectors=vectors)
    assert bool(reference.successful)
    target = float(reference.energy)
    scale = max(1.0, abs(target))

    def mesh(grid: int, nyquist: str = "symmetrized") -> Any:
        result = _pme_engine(
            kind,
            2,
            system,
            alpha=0.9,
            cutoff=5.0,
            grid=grid,
            mesh_kwargs={"spline_degree": 4, "nyquist": nyquist},
        ).energy(positions, sites, cell_vectors=vectors)
        assert bool(result.successful)
        return result

    errors = [abs(float(mesh(grid).energy) - target) / scale for grid in (12, 20, 32)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1.0e-5
    methods = set()
    for nyquist in ("symmetrized", "excluded"):
        result = mesh(32, nyquist)
        assert abs(float(result.energy) - target) / scale < 1.0e-5
        assert f"nyquist-{nyquist}" in result.evidence.method
        methods.add(result.evidence.method)
    assert len(methods) == 2


# --- 9. derivatives on a triclinic cell ------------------------------------------------------


def test_periodic_gaussian_multipole_derivatives_match_finite_differences() -> None:
    count = 4
    positions = _configuration(61, TRICLINIC, count, 1.1)
    q, mu, theta = _random_moments(67, count, charge=0.8, dipole=0.3, second=0.15)
    unique = theta[:, _UNIQUE_ROWS, _UNIQUE_COLUMNS]
    widths = GaussianMultipoleWidths(np.asarray([0.25, 0.35]), np.asarray([0, 1, 0, 1]))
    engine = _ewald_engine(
        "gaussian",
        2,
        phx.discretization.PeriodicCell(TRICLINIC),
        count,
        alpha=1.5,
        cutoff=4.5,
        extents=8,
    )
    charges = jnp.asarray(q)
    active = np.ones((count,), dtype=bool)

    @jax.jit
    def energy(
        position: jax.Array, vectors: jax.Array, dipoles: jax.Array, second: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        moments = CartesianMultipoles(charges, dipoles, second, maximum_degree=2)
        sites = MultipoleSiteSet(moments, active=active, widths=widths)
        result = engine.energy(position, sites, cell_vectors=vectors)
        return result.energy, result.successful

    x = jnp.asarray(positions)
    h = jnp.asarray(TRICLINIC)
    m = jnp.asarray(mu)
    u = jnp.asarray(unique)
    value, successful = energy(x, h, m, u)
    assert bool(successful) and np.isfinite(float(value))

    def scalar(*arguments: jax.Array) -> jax.Array:
        return energy(*arguments)[0]

    g_position, g_cell, g_dipole, g_second = (
        np.asarray(item) for item in jax.grad(scalar, argnums=(0, 1, 2, 3))(x, h, m, u)
    )
    forces = -g_position
    numeric_forces = -central_difference(
        lambda p: float(scalar(jnp.asarray(p), h, m, u)), positions, 1.0e-5
    )
    np.testing.assert_allclose(forces, numeric_forces, rtol=1.0e-6, atol=1.0e-8)
    assert np.max(np.abs(np.sum(forces, axis=0))) < 1.0e-9 * max(
        1.0, float(np.max(np.abs(forces)))
    )

    def strained(strain: np.ndarray) -> float:
        deformation = np.eye(3) + strain
        return float(
            scalar(
                jnp.asarray(positions @ deformation.T),
                jnp.asarray(TRICLINIC @ deformation.T),
                m,
                u,
            )
        )

    strain_derivative = np.einsum("ia,ib->ab", g_position, positions) + np.einsum(
        "ka,kb->ab", g_cell, TRICLINIC
    )
    np.testing.assert_allclose(
        strain_derivative,
        central_difference(strained, np.zeros((3, 3)), 1.0e-6),
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        g_dipole,
        central_difference(lambda d: float(scalar(x, h, jnp.asarray(d), u)), mu, 1.0e-5),
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        g_second,
        central_difference(
            lambda s: float(scalar(x, h, m, jnp.asarray(s))), unique, 1.0e-5
        ),
        rtol=1.0e-6,
        atol=1.0e-8,
    )


# --- 10. field features -------------------------------------------------------------------


def test_point_charge_field_features_match_coulomb_sums() -> None:
    q = np.random.default_rng(71).normal(size=4)
    constant = 1.3
    engine = _finite_engine("point", 2, 4, coulomb_constant=constant)
    features = engine.features(
        jnp.asarray(FINITE_POSITIONS), _sites(_multipoles(q, degree=0))
    )
    assert bool(features.successful) and features.own_site == "exclude"
    potential = np.zeros(4)
    field = np.zeros((4, 3))
    gradient = np.zeros((4, 3, 3))
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            r = FINITE_POSITIONS[i] - FINITE_POSITIONS[j]
            d = np.linalg.norm(r)
            potential[i] += constant * q[j] / d
            field[i] += constant * q[j] * r / d**3
            gradient[i] += (
                constant * q[j] * (3.0 * np.outer(r, r) - d * d * np.eye(3)) / d**5
            )
    assert features.potential.shape == (4, 1)
    np.testing.assert_allclose(
        np.asarray(features.potential[:, 0]), potential, rtol=1.0e-12
    )
    np.testing.assert_allclose(
        np.asarray(features.field[:, 0]), field, rtol=1.0e-11, atol=1.0e-13
    )
    np.testing.assert_allclose(
        np.asarray(features.field_gradient[:, 0]), gradient, rtol=1.0e-11, atol=1.0e-13
    )


@pytest.mark.parametrize("own_site", ["exclude", "include"])
def test_gaussian_field_features_match_erf_sums(own_site: str) -> None:
    count = 4
    q = np.random.default_rng(73).normal(size=count)
    sigma = np.asarray([0.3, 0.45, 0.6, 0.35])
    receivers = np.asarray([0.25, 0.5])
    constant = 1.3
    engine = _finite_engine("gaussian", 2, count, coulomb_constant=constant)
    features = engine.features(
        jnp.asarray(FINITE_POSITIONS),
        _sites(_multipoles(q, degree=0), _site_widths(sigma)),
        receiver_widths=jnp.asarray(receivers),
        own_site=own_site,
    )
    assert bool(features.successful)
    potential = np.zeros((count, 2))
    field = np.zeros((count, 2, 3))
    for i in range(count):
        for k, s_r in enumerate(receivers):
            for j in range(count):
                beta = 1.0 / math.sqrt(2.0 * (sigma[j] ** 2 + s_r**2))
                if i == j:
                    if own_site == "include":
                        potential[i, k] += constant * q[i] * 2.0 * beta / _SQRT_PI
                    continue
                r = FINITE_POSITIONS[i] - FINITE_POSITIONS[j]
                d = float(np.linalg.norm(r))
                potential[i, k] += constant * q[j] * math.erf(beta * d) / d
                radial = (
                    2.0 * beta / _SQRT_PI * math.exp(-((beta * d) ** 2)) / d
                    - math.erf(beta * d) / d**2
                )
                field[i, k] -= constant * q[j] * radial * r / d
    assert features.potential.shape == (count, 2)
    np.testing.assert_allclose(np.asarray(features.potential), potential, rtol=1.0e-11)
    np.testing.assert_allclose(
        np.asarray(features.field), field, rtol=1.0e-10, atol=1.0e-13
    )


def test_invalid_receiver_widths_are_refused_eagerly_and_fail_when_traced() -> None:
    count = 4
    q = np.random.default_rng(79).normal(size=count)
    sigma = np.asarray([0.3, 0.45, 0.6, 0.35])
    engine = _finite_engine("gaussian", 2, count)
    sites = _sites(_multipoles(q, degree=0), _site_widths(sigma))
    positions = jnp.asarray(FINITE_POSITIONS)
    for invalid in ([-0.3], [0.0], [np.nan]):
        with pytest.raises(ValueError, match="finite and positive"):
            engine.features(positions, sites, receiver_widths=jnp.asarray(invalid))

    def traced(width: Any) -> Any:
        return engine.features(positions, sites, receiver_widths=width[None]).successful

    compiled = jax.jit(traced)
    assert bool(compiled(jnp.asarray(0.3)))
    assert not bool(compiled(jnp.asarray(-0.3)))
    assert not bool(compiled(jnp.asarray(jnp.nan)))


# --- 11. invariance and covariance ---------------------------------------------------------


@pytest.mark.parametrize("kind", ["point", "gaussian"])
def test_finite_energy_is_rotation_and_translation_invariant(
    kind: MultipoleSiteKind,
) -> None:
    q, mu, theta = _random_moments(79, 5, neutral=False)
    moments = _multipoles(q, mu, theta, degree=2)
    widths = _site_widths(GAUSSIAN_SIGMAS) if kind == "gaussian" else None
    positions = GAUSSIAN_POSITIONS
    engine = _finite_engine(kind, 2, 5)
    base = float(engine.energy(jnp.asarray(positions), _sites(moments, widths)).energy)
    rotation = _rotation(83)
    rotated = float(
        engine.energy(
            jnp.asarray(positions @ rotation.T), _sites(moments.rotated(rotation), widths)
        ).energy
    )
    translated = float(
        engine.energy(
            jnp.asarray(positions + np.asarray([3.1, -0.7, 12.4])),
            _sites(moments, widths),
        ).energy
    )
    np.testing.assert_allclose(rotated, base, rtol=1.0e-11)
    np.testing.assert_allclose(translated, base, rtol=1.0e-11)


def test_periodic_energy_is_invariant_under_rigid_cell_rotation_and_translation() -> None:
    count = 4
    positions = _configuration(89, TRICLINIC, count, 1.0)
    q, mu, theta = _random_moments(97, count, charge=0.8, dipole=0.3, second=0.15)
    moments = _multipoles(q, mu, theta, degree=2)
    widths = _class_widths(count)
    engine = _ewald_engine(
        "gaussian",
        2,
        phx.discretization.PeriodicCell(TRICLINIC),
        count,
        alpha=1.4,
        cutoff=4.5,
        extents=10,
    )
    base = engine.energy(
        jnp.asarray(positions),
        _sites(moments, widths),
        cell_vectors=jnp.asarray(TRICLINIC),
    )
    shifted = engine.energy(
        jnp.asarray(positions + np.asarray([0.37, -1.2, 2.5])),
        _sites(moments, widths),
        cell_vectors=jnp.asarray(TRICLINIC),
    )
    rotation = _rotation(101)
    rotated_cell = TRICLINIC @ rotation.T
    rotated_engine = _ewald_engine(
        "gaussian",
        2,
        phx.discretization.PeriodicCell(rotated_cell),
        count,
        alpha=1.4,
        cutoff=4.5,
        extents=10,
    )
    rotated = rotated_engine.energy(
        jnp.asarray(positions @ rotation.T),
        _sites(moments.rotated(rotation), widths),
        cell_vectors=jnp.asarray(rotated_cell),
    )
    assert bool(base.successful) and bool(shifted.successful) and bool(rotated.successful)
    np.testing.assert_allclose(float(shifted.energy), float(base.energy), rtol=1.0e-10)
    np.testing.assert_allclose(float(rotated.energy), float(base.energy), rtol=1.0e-10)


def test_field_features_rotate_covariantly() -> None:
    q, mu, theta = _random_moments(103, 5, neutral=False)
    moments = _multipoles(q, mu, theta, degree=2)
    widths = _site_widths(GAUSSIAN_SIGMAS)
    receivers = jnp.asarray([0.3, 0.5])
    engine = _finite_engine("gaussian", 2, 5)
    base = engine.features(
        jnp.asarray(GAUSSIAN_POSITIONS),
        _sites(moments, widths),
        receiver_widths=receivers,
    )
    rotation = _rotation(107)
    rotated = engine.features(
        jnp.asarray(GAUSSIAN_POSITIONS @ rotation.T),
        _sites(moments.rotated(rotation), widths),
        receiver_widths=receivers,
    )
    assert bool(base.successful) and bool(rotated.successful)
    np.testing.assert_allclose(
        np.asarray(rotated.potential), np.asarray(base.potential), rtol=1.0e-11
    )
    np.testing.assert_allclose(
        np.asarray(rotated.field),
        np.einsum("ab,nrb->nra", rotation, np.asarray(base.field)),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(rotated.field_gradient),
        np.einsum(
            "ab,nrbc,dc->nrad", rotation, np.asarray(base.field_gradient), rotation
        ),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


# --- 12. refusals ---------------------------------------------------------------------------


def test_octupole_degree_is_refused() -> None:
    with pytest.raises(ValueError, match="octupole"):
        CartesianMultipoles(
            jnp.ones((2,)), jnp.zeros((2, 3)), jnp.zeros((2, 3, 3)), maximum_degree=3
        )
    with pytest.raises(ValueError, match="octupole"):
        MultipoleElectrostaticsDefinition("point", 3, coulomb_constant=1.0)


def test_point_sites_refuse_intrasite_inclusion() -> None:
    with pytest.raises(ValueError, match="intrasite"):
        MultipoleElectrostaticsDefinition(
            "point", 2, coulomb_constant=1.0, intrasite="include"
        )
    engine = _finite_engine("point", 2, 4)
    q, mu, theta = _random_moments(109, 4)
    sites = _sites(_multipoles(q, mu, theta, degree=2))
    with pytest.raises(ValueError, match="intrasite"):
        engine.bilinear(jnp.asarray(FINITE_POSITIONS), sites, sites, intrasite="include")
    with pytest.raises(ValueError, match="own-site"):
        engine.features(jnp.asarray(FINITE_POSITIONS), sites, own_site="include")


def test_prepared_engine_refuses_mismatched_sites() -> None:
    position = jnp.asarray(FINITE_POSITIONS)
    q, mu, theta = _random_moments(113, 4)
    point = _finite_engine("point", 2, 4)
    gaussian = _finite_engine("gaussian", 2, 4)
    with pytest.raises(ValueError, match="site capacity"):
        point.energy(position, _sites(_multipoles(q[:3], mu[:3], theta[:3], degree=2)))
    with pytest.raises(ValueError, match="widths do not match"):
        gaussian.energy(position, _sites(_multipoles(q, mu, theta, degree=2)))
    with pytest.raises(ValueError, match="widths do not match"):
        point.energy(
            position,
            _sites(_multipoles(q, mu, theta, degree=2), _site_widths(np.ones(4))),
        )
    with pytest.raises(ValueError, match="share their site capacity"):
        _sites(_multipoles(q, mu, theta, degree=2), _site_widths(np.ones(3)))
    with pytest.raises(ValueError):
        _finite_engine("point", 1, 4).energy(
            position, _sites(_multipoles(q, mu, theta, degree=2))
        )


def test_neighborhood_real_space_requires_neighborhood_pairs() -> None:
    cell = phx.discretization.PeriodicCell(TRICLINIC)
    engine = _ewald_engine(
        "point",
        0,
        cell,
        4,
        alpha=1.0,
        cutoff=3.0,
        extents=4,
        route_kwargs={"real_space": "neighborhood"},
    )
    assert engine.reference_pairs is None
    with pytest.raises(ValueError, match="NeighborhoodPairs"):
        engine.energy(
            jnp.asarray(_configuration(127, TRICLINIC, 4, 1.0)),
            _sites(_multipoles(_neutral(131, 4), degree=0)),
            cell_vectors=jnp.asarray(TRICLINIC),
        )


def test_explicit_image_workspace_is_admitted_at_preparation() -> None:
    with pytest.raises(ValueError, match="maximum_workspace_bytes"):
        ExplicitImagePairs(4, maximum_workspace_bytes=100)
    with pytest.raises(ValueError, match="maximum_workspace_bytes"):
        MultipoleElectrostaticsPlan(
            _definition("point", 2), FiniteMultipoleRoute(maximum_workspace_bytes=1000)
        ).prepare(4)
    with pytest.raises(ValueError, match="maximum_workspace_bytes"):
        _ewald_engine(
            "point",
            2,
            phx.discretization.PeriodicCell(TRICLINIC),
            4,
            alpha=1.0,
            cutoff=4.0,
            extents=4,
            route_kwargs={"maximum_workspace_bytes": 10_000},
        )


def test_declared_reciprocal_cutoff_beyond_mode_support_fails_closed() -> None:
    cell = phx.discretization.PeriodicCell(TRICLINIC)
    position = jnp.asarray(_configuration(137, TRICLINIC, 4, 1.0))
    sites = _sites(_multipoles(_neutral(139, 4), degree=0))
    support = min(2.0 * math.pi * 5.0 / np.linalg.norm(row) for row in TRICLINIC)

    def evaluate(cutoff: float) -> Any:
        return _ewald_engine(
            "point",
            0,
            cell,
            4,
            alpha=1.0,
            cutoff=3.0,
            extents=4,
            route_kwargs={"reciprocal_cutoff": cutoff},
        ).energy(position, sites, cell_vectors=jnp.asarray(TRICLINIC))

    accepted = evaluate(0.5 * support)
    assert bool(accepted.successful)
    assert bool(accepted.evidence.mode_coverage_successful)
    np.testing.assert_allclose(
        float(accepted.evidence.support_radius), support, rtol=1.0e-12
    )
    refused = evaluate(50.0)
    assert not bool(refused.successful)
    assert not bool(refused.evidence.mode_coverage_successful)
    assert bool(refused.evidence.cell_successful)
    assert refused.evidence.declared_reciprocal_cutoff == 50.0
    assert bool(jnp.isnan(refused.energy))


def test_reciprocal_budget_refuses_oversized_mesh_and_modes() -> None:
    system = _system(4, cell=phx.discretization.PeriodicCell(TRICLINIC))
    with pytest.raises(ValueError, match="grid_points"):
        _pme_engine(
            "point",
            0,
            system,
            alpha=1.0,
            cutoff=3.0,
            grid=16,
            mesh_kwargs={"budget": ReciprocalBudget(maximum_grid_points=1000)},
        )
    with pytest.raises(ValueError, match="modes"):
        MultipoleElectrostaticsPlan(
            _definition("point", 0),
            EwaldMultipoleRoute(
                1.0,
                3.0,
                ReciprocalModeSetPlan(
                    (6, 6, 6), budget=ReciprocalBudget(maximum_modes=50)
                ),
            ),
        ).prepare(4, cell=system.cell)


# --- 13. program term ------------------------------------------------------------------------


@pytest.mark.parametrize("periodic", [False, True])
def test_program_term_energy_equals_prepared_engine(periodic: bool) -> None:
    count = 4
    q, mu, theta = _random_moments(149, count, charge=0.8, dipole=0.3, second=0.15)
    moments = _multipoles(q, mu, theta, degree=2)
    if periodic:
        cell = phx.discretization.PeriodicCell(TRICLINIC)
        positions = _configuration(151, TRICLINIC, count, 1.0)
        widths = _class_widths(count)
        plan = MultipoleElectrostaticsPlan(
            _definition("gaussian", 2),
            EwaldMultipoleRoute(1.4, 4.5, ReciprocalModeSetPlan((8, 8, 8))),
        )
    else:
        cell = None
        positions = FINITE_POSITIONS
        widths = None
        plan = MultipoleElectrostaticsPlan(
            _definition("point", 2), FiniteMultipoleRoute()
        )
    term = MultipoleElectrostaticPotential(plan, moments=moments, widths=widths)
    system = _system(count, cell=cell)
    program = prepare_program(phx.atomistic.AtomisticPotentialProgram([term]), system)
    position = jnp.asarray(positions, dtype=jnp.float64)
    state = bundle_state(program, position)
    evaluation = program.evaluate(position, state)
    engine = plan.prepare(count, cell=system.cell) if periodic else plan.prepare(count)
    expected = engine.energy(
        position,
        _sites(moments, widths),
        cell_vectors=jnp.asarray(TRICLINIC) if periodic else None,
    )
    assert bool(evaluation.successful) and bool(expected.successful)
    np.testing.assert_allclose(
        float(evaluation.energy), float(expected.energy), rtol=1.0e-12
    )
    term_evaluation = program.terms[0].energy(program.context(position, state))
    assert bool(term_evaluation.successful)
    assert not term_evaluation.atom_energy_complete
    matches = [
        value
        for value in term_evaluation.evidence
        if isinstance(value, MultipoleElectrostaticsEvidence)
    ]
    assert len(matches) == 1
    (evidence,) = matches
    assert evidence.moment_source == "explicit-site-data"
    assert evidence.method == engine.method
    np.testing.assert_allclose(
        float(term_evaluation.energy), float(expected.energy), rtol=1.0e-12
    )
