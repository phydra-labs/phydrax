from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from fractions import Fraction
from itertools import product
from typing import final

import equinox as eqx
import jax
import numpy as np
from jax.typing import ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg._exact import determinant_exact, solve_exact
from ..typing import checked, Dim, Float64
from ._coordinate_budget import CoordinateEnclosureBudget
from ._periodic_cell import PeriodicCell


MAXIMUM_ROTATION_ORDER = 1024
MAXIMUM_GROUP_IMAGES = 100_000


class GeneratorDim(Dim, minimum=1):
    """Number of commuting isometry generators."""


class HomogeneousDim(Dim, minimum=3):
    """Homogeneous coordinate extent: ambient dimension plus one."""


def isometry_power_work_upper(dimension: int, exponent_bits: int, /) -> int:
    homogeneous = dimension + 1
    return homogeneous**2 + (2 * exponent_bits + 1) * homogeneous**3


def isometry_element_work_upper(rank: int, dimension: int, exponent_bits: int, /) -> int:
    homogeneous = dimension + 1
    return homogeneous**2 + rank * (
        isometry_power_work_upper(dimension, exponent_bits) + homogeneous**3
    )


def isometry_group_work_upper(rank: int, dimension: int, /) -> int:
    """Bound classification, commutation and every supported exponent relation."""
    homogeneous = dimension + 1
    exponent_bits = MAXIMUM_ROTATION_ORDER.bit_length()
    solve = rank * rank * (rank + 1)
    return (
        rank * homogeneous**2
        + homogeneous**2
        + rank
        * (
            2 * dimension**3
            + MAXIMUM_ROTATION_ORDER * homogeneous**3
            + isometry_power_work_upper(dimension, exponent_bits)
        )
        + rank * (rank - 1) * homogeneous**3
        + rank * rank * dimension
        + solve
        + MAXIMUM_GROUP_IMAGES
        * (
            isometry_element_work_upper(rank, dimension, exponent_bits)
            + 2 * rank * dimension
            + solve
        )
    )


@contextmanager
def owned_periodic_preparation(
    work_upper: int,
    bytes_upper: int,
    /,
    *,
    queries_upper: int = 0,
    budget: CoordinateEnclosureBudget | None = None,
) -> Iterator[CoordinateEnclosureBudget]:
    """Use the caller budget or reserve a host-only preparation scope."""
    if budget is not None:
        yield budget
        return
    with HostBatchTicket(
        work_upper, queries_upper=queries_upper, bytes_upper=bytes_upper
    ) as ticket:
        owner = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=work_upper,
            maximum_memory_bytes=bytes_upper,
        )
        with owner.activate():
            yield owner


def isometry_inverse(matrix: np.ndarray, /) -> np.ndarray:
    """Return the inverse homogeneous isometry ``x -> Q x + b``."""

    dimension = matrix.shape[0] - 1
    inverse = np.eye(dimension + 1, dtype=np.float64)
    inverse[:dimension, :dimension] = matrix[:dimension, :dimension].T
    inverse[:dimension, dimension] = (
        -matrix[:dimension, :dimension].T @ matrix[:dimension, dimension]
    )
    return inverse


def isometry_power(matrix: np.ndarray, exponent: int, /) -> np.ndarray:
    """Integer power of a homogeneous isometry by repeated squaring."""

    base = matrix if exponent >= 0 else isometry_inverse(matrix)
    remaining = abs(exponent)
    result = np.eye(matrix.shape[0], dtype=np.float64)
    while remaining:
        if remaining & 1:
            result = base @ result
        base = base @ base
        remaining >>= 1
    return result


def require_proper_isometries(matrices: np.ndarray, /) -> None:
    dimension = matrices.shape[1] - 1
    identity = np.eye(dimension, dtype=np.float64)
    bottom = np.eye(dimension + 1, dtype=np.float64)[-1]
    # Orthonormality within the rounding of trigonometric entries.
    bound = 128.0 * float(np.finfo(np.float64).eps) * dimension
    for index, matrix in enumerate(matrices):
        linear = matrix[:dimension, :dimension]
        if not np.array_equal(matrix[dimension], bottom):
            raise ValueError(
                f"PeriodicIsometryGroup.generators must have homogeneous bottom rows; got generator {index}"
            )
        if np.max(np.abs(linear.T @ linear - identity)) > bound:
            raise ValueError(
                f"PeriodicIsometryGroup.generators must preserve the Euclidean inner product; got generator {index}"
            )
        if np.linalg.det(linear) < 0.0:
            raise ValueError(
                f"PeriodicIsometryGroup.generators must preserve orientation; got reversing generator {index}"
            )


def _isometry_group_content_id(
    generators: np.ndarray,
    orders: tuple[int, ...],
    linear_orders: tuple[int, ...],
    translation_periods: tuple[int, ...],
    tolerance: float,
) -> str:
    """Authenticate retained group content before reusing its admission."""
    return canonical_fingerprint({
        "kind": "periodic-isometry-group",
        "generators": array_tree_fingerprint(generators),
        "orders": list(orders),
        "linear_orders": list(linear_orders),
        "translation_periods": list(translation_periods),
        "tolerance": tolerance,
    })


@final
class PeriodicIsometryGroup(StrictModule, NonTrainableState):
    """Abelian group of proper Euclidean isometries identifying a periodic mesh.

    ``generators`` (r, d + 1, d + 1) are homogeneous maps ``x -> Q x + b`` with
    orthogonal ``Q`` of determinant ``+1``. ``orders`` records the full
    homogeneous order proved using exact arithmetic (zero means infinite);
    ``linear_orders`` records the finite order of Q. Infinite screws retain
    their input generator; ``translation_periods`` gives its translation exponent.
    Generators must commute, so every cycle ``G_a G_b G_a^-1 G_b^-1`` is the
    identity and integer exponents ``s`` denote ``prod_c G_c^(s_c)`` in any order.
    Reflections and other improper maps reverse orientation and are refused.
    ``tolerance`` bounds the coordinate residual of every declared image.
    """

    __strict_contract__ = True

    generators: Float64[GeneratorDim, HomogeneousDim, HomogeneousDim]
    orders: tuple[int, ...] = eqx.field(static=True)
    linear_orders: tuple[int, ...] = eqx.field(static=True)
    translation_periods: tuple[int, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    group_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        generators: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-10,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        matrices = np.asarray(generators, dtype=np.float64)
        if (
            matrices.ndim != 3
            or matrices.shape[0] == 0
            or matrices.shape[1] != matrices.shape[2]
            or matrices.shape[1] < 3
        ):
            raise ValueError(
                "PeriodicIsometryGroup.generators must have shape (r > 0, d + 1, d + 1) with d >= 2; got another bank shape"
            )
        threshold = float(tolerance)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "PeriodicIsometryGroup.tolerance must be finite and nonnegative; got an invalid residual bound"
            )
        if not np.all(np.isfinite(matrices)):
            raise ValueError(
                "PeriodicIsometryGroup.generators must be finite; got a nonfinite matrix entry"
            )
        require_proper_isometries(matrices)
        work_upper = isometry_group_work_upper(matrices.shape[0], matrices.shape[1] - 1)
        with owned_periodic_preparation(work_upper, 0, budget=budget) as owner:
            admission = admit_exact_isometry_group(matrices, budget=owner)
        orders = admission.orders
        linear_orders = admission.linear_orders
        translation_periods = admission.translation_periods
        self.generators = jax.device_put(matrices)
        self.orders = orders
        self.linear_orders = linear_orders
        self.translation_periods = translation_periods
        self.tolerance = threshold
        self.ambient_dimension = matrices.shape[1] - 1
        self.group_id = _isometry_group_content_id(
            matrices, orders, linear_orders, translation_periods, threshold
        )

    def validate_restored(self) -> None:
        """Validate retained source controls and constructor-derived metadata."""
        replay = PeriodicIsometryGroup(self.generators, tolerance=self.tolerance)
        for name, value, expected in (
            ("orders", self.orders, replay.orders),
            ("linear_orders", self.linear_orders, replay.linear_orders),
            ("translation_periods", self.translation_periods, replay.translation_periods),
            ("ambient_dimension", self.ambient_dimension, replay.ambient_dimension),
            ("group_id", self.group_id, replay.group_id),
        ):
            if value != expected:
                raise ValueError(
                    f"PeriodicIsometryGroup.{name} must match constructor-derived metadata; got a changed stored value"
                )

    @property
    def rank(self) -> int:
        return len(self.orders)

    def element(self, exponents: ArrayLike, /) -> np.ndarray:
        """Homogeneous matrix of the group element with integer ``exponents``."""

        values = np.asarray(exponents)
        if not np.issubdtype(values.dtype, np.integer) or values.shape != (self.rank,):
            raise ValueError(
                "PeriodicIsometryGroup.exponents must contain one integer per generator; got another shape or dtype"
            )
        matrices = np.asarray(self.generators, dtype=np.float64)
        result = np.eye(self.ambient_dimension + 1, dtype=np.float64)
        for matrix, exponent, order in zip(
            matrices, values.tolist(), self.orders, strict=True
        ):
            reduced = exponent % order if order else exponent
            result = isometry_power(np.asarray(matrix), reduced) @ result
        return result

    def apply(self, points: ArrayLike, exponents: ArrayLike, /) -> np.ndarray:
        """Map each point by the group element of its exponent row."""

        array = np.asarray(points, dtype=np.float64)
        values = np.asarray(exponents)
        if (
            array.ndim != 2
            or array.shape[1] != self.ambient_dimension
            or values.shape != (array.shape[0], self.rank)
            or not np.issubdtype(values.dtype, np.integer)
        ):
            raise ValueError(
                "PeriodicIsometryGroup.points must be (n, ambient dimension) with one integer exponent row per point; got another shape or dtype"
            )
        reduced = reduce_periodic_exponents(
            values.astype(np.int64), np.asarray(self.orders)
        )
        unique, inverse = np.unique(reduced, axis=0, return_inverse=True)
        mapped = np.empty_like(array)
        dimension = self.ambient_dimension
        for row, exponent in enumerate(unique):
            matrix = self.element(exponent)
            members = inverse.reshape(-1) == row
            mapped[members] = (
                array[members] @ matrix[:dimension, :dimension].T
                + matrix[:dimension, dimension]
            )
        return mapped

    def fixed(self, points: ArrayLike, /) -> np.ndarray:
        """Whether each generator fixes each point within the group tolerance."""

        array = np.asarray(points, dtype=np.float64)
        dimension = self.ambient_dimension
        matrices = np.asarray(self.generators, dtype=np.float64)
        images = (
            np.asarray(
                ein.contract("gij,nj->ngi", matrices[:, :dimension, :dimension], array)
            )
            + matrices[None, :, :dimension, dimension]
        )
        return np.linalg.norm(images - array[:, None, :], axis=2) <= self.tolerance


def reduce_periodic_exponents(values: np.ndarray, orders: np.ndarray, /) -> np.ndarray:
    """Reduce exponents modulo the finite generator orders (last axis)."""

    return np.where(orders > 0, np.mod(values, np.maximum(orders, 1)), values)


@checked
def identification_id(cell: PeriodicMeshIdentification, /) -> str:
    match cell:
        case PeriodicCell():
            return cell.cell_id
        case PeriodicIsometryGroup():
            return cell.group_id
        case _:
            raise TypeError(
                "periodic_identification.cell must be a PeriodicCell or PeriodicIsometryGroup; got another identification kind"
            )


@checked
def identification_orders(cell: PeriodicMeshIdentification, /) -> np.ndarray:
    match cell:
        case PeriodicCell():
            return np.zeros((cell.rank,), dtype=np.int64)
        case PeriodicIsometryGroup():
            return np.asarray(cell.orders, dtype=np.int64)
        case _:
            raise TypeError(
                "periodic_identification.cell must be a PeriodicCell or PeriodicIsometryGroup; got another identification kind"
            )


class PeriodicIsometryIdentityError(ValueError):
    """Report generators that do not establish the required rational group identities."""


def reserve_exact_isometry_work(
    work: int, /, *, budget: CoordinateEnclosureBudget
) -> None:
    budget.admit_work_bound(work)
    budget.reserve(work)


def exact_isometry_identity(
    dimension: int, *, budget: CoordinateEnclosureBudget
) -> tuple[tuple[Fraction, ...], ...]:
    reserve_exact_isometry_work((dimension + 1) ** 2, budget=budget)
    return tuple(
        tuple(Fraction(i == j) for j in range(dimension + 1))
        for i in range(dimension + 1)
    )


def exact_isometry_multiply(
    first: tuple[tuple[Fraction, ...], ...],
    second: tuple[tuple[Fraction, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Fraction, ...], ...]:
    reserve_exact_isometry_work(len(first) * len(second) ** 2, budget=budget)
    return tuple(
        tuple(
            sum((a * second[k][j] for k, a in enumerate(row)), Fraction(0))
            for j in range(len(second))
        )
        for row in first
    )


def exact_isometry_power(
    matrix: tuple[tuple[Fraction, ...], ...],
    exponent: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Fraction, ...], ...]:
    if exponent < 0:
        dimension = len(matrix) - 1
        reserve_exact_isometry_work((dimension + 1) ** 3, budget=budget)
        inverse = tuple(
            tuple(matrix[j][i] for j in range(dimension))
            + (
                -sum(
                    (matrix[j][i] * matrix[j][-1] for j in range(dimension)), Fraction(0)
                ),
            )
            for i in range(dimension)
        ) + (matrix[-1],)
        matrix, exponent = inverse, -exponent
    result = exact_isometry_identity(len(matrix) - 1, budget=budget)
    while exponent:
        if exponent & 1:
            result = exact_isometry_multiply(result, matrix, budget=budget)
        exponent >>= 1
        if exponent:
            matrix = exact_isometry_multiply(matrix, matrix, budget=budget)
    return result


def exact_group_solve(
    matrix: tuple[tuple[Fraction, ...], ...],
    rhs: tuple[Fraction, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Fraction, ...]:
    """Solve a rational system after reserving its unchanged elimination work."""
    size = len(rhs)
    reserve_exact_isometry_work(size * size * (size + 1), budget=budget)
    result = solve_exact(
        [list(row) for row in matrix],
        [[value] for value in rhs],
        singular_error=PeriodicIsometryIdentityError,
        singular_message="exact_group_solve.matrix must be nonsingular; got a singular translation-period Gram matrix",
    )
    return tuple(row[0] for row in result)


@final
class _ExactIsometryGenerator(StrictModule):
    """Classify one homogeneous generator and its derived translation."""

    order: int = eqx.field(static=True)
    linear_order: int = eqx.field(static=True)
    translation_period: int = eqx.field(static=True)
    translation: tuple[Fraction, ...] | None = eqx.field(static=True)


@final
class ExactIsometryAdmission(StrictModule):
    """Return rational homogeneous generators and admitted exponent structure."""

    matrices: tuple[tuple[tuple[Fraction, ...], ...], ...] = eqx.field(static=True)
    orders: tuple[int, ...] = eqx.field(static=True)
    linear_orders: tuple[int, ...] = eqx.field(static=True)
    translation_periods: tuple[int, ...] = eqx.field(static=True)


def _classify_exact_isometry_generator(
    index: int,
    matrix: tuple[tuple[Fraction, ...], ...],
    identity: tuple[tuple[Fraction, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> _ExactIsometryGenerator:
    """Prove properness and finite linear order, retaining translation periods."""
    dimension = len(matrix) - 1
    if matrix[-1] != identity[-1]:
        raise PeriodicIsometryIdentityError(
            "PeriodicIsometryGroup.generators must be homogeneous affine maps; got an invalid bottom row"
        )
    if matrix == identity:
        raise PeriodicIsometryIdentityError(
            f"PeriodicIsometryGroup.generators must contain nonidentity actions; got identity generator {index}"
        )
    linear = tuple(
        tuple(row[j] for j in range(dimension)) + (Fraction(0),) for row in matrix[:-1]
    ) + (identity[-1],)
    reserve_exact_isometry_work(dimension**3, budget=budget)
    if any(
        sum((matrix[k][i] * matrix[k][j] for k in range(dimension)), Fraction(0))
        != Fraction(i == j)
        for i in range(dimension)
        for j in range(dimension)
    ):
        raise PeriodicIsometryIdentityError(
            "PeriodicIsometryGroup.generators must have orthogonal linear parts in rational arithmetic; got a failed isometry identity"
        )
    reserve_exact_isometry_work(dimension**3, budget=budget)
    determinant = determinant_exact(tuple(row[:dimension] for row in matrix[:dimension]))
    if determinant != 1:
        raise PeriodicIsometryIdentityError(
            "PeriodicIsometryGroup.generators must have determinant-one linear parts; got an improper map"
        )
    power = identity
    for order in range(1, MAXIMUM_ROTATION_ORDER + 1):
        power = exact_isometry_multiply(power, linear, budget=budget)
        if power == identity:
            break
    else:
        raise PeriodicIsometryIdentityError(
            f"PeriodicIsometryGroup.linear_orders must be proven finite and at most {MAXIMUM_ROTATION_ORDER}; got an unsupported or unproven order for generator {index}"
        )
    cycle = exact_isometry_power(matrix, order, budget=budget)
    full_order = order if cycle == identity else 0
    return _ExactIsometryGenerator(
        order=full_order,
        linear_order=order,
        translation_period=0 if full_order else order,
        translation=None if full_order else tuple(row[-1] for row in cycle[:-1]),
    )


def _require_faithful_isometry_exponents(
    matrices: tuple[tuple[tuple[Fraction, ...], ...], ...],
    orders: tuple[int, ...],
    linear_orders: tuple[int, ...],
    vectors: list[tuple[Fraction, ...]],
    identity: tuple[tuple[Fraction, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Prove discreteness and refuse nontrivial relations among retained exponents."""
    dimension = len(identity) - 1
    reserve_exact_isometry_work(len(vectors) ** 2 * dimension, budget=budget)
    gram = tuple(
        tuple(
            sum((a * b for a, b in zip(v, w, strict=True)), Fraction(0)) for w in vectors
        )
        for v in vectors
    )
    exact_group_solve(gram, (Fraction(0),) * len(vectors), budget=budget)
    count = 1
    for order in linear_orders:
        count *= order
    if count > MAXIMUM_GROUP_IMAGES:
        raise PeriodicIsometryIdentityError(
            f"PeriodicIsometryGroup.exponents must fit the bounded relation validation; got more than {MAXIMUM_GROUP_IMAGES} images"
        )
    for exponents in product(*(range(order) for order in linear_orders)):
        if not any(exponents):
            continue
        action = exact_periodic_element(matrices, tuple(orders), exponents, budget=budget)
        if any(
            action[i][j] != identity[i][j]
            for i in range(dimension)
            for j in range(dimension)
        ):
            continue
        translation = tuple(row[-1] for row in action[:-1])
        reserve_exact_isometry_work(len(vectors) * dimension * 2, budget=budget)
        rhs = tuple(
            sum((a * b for a, b in zip(v, translation, strict=True)), Fraction(0))
            for v in vectors
        )
        coefficients = exact_group_solve(gram, rhs, budget=budget)
        if all(value.denominator == 1 for value in coefficients) and all(
            sum(
                (c * v[i] for c, v in zip(coefficients, vectors, strict=True)),
                Fraction(0),
            )
            == translation[i]
            for i in range(dimension)
        ):
            raise PeriodicIsometryIdentityError(
                "PeriodicIsometryGroup.exponents must have no nontrivial rational relation; got dependent generator exponents"
            )


@checked
def admit_exact_isometry_group(
    source: np.ndarray, /, *, budget: CoordinateEnclosureBudget
) -> ExactIsometryAdmission:
    """Prove group closure and exponent structure from the binary64 source."""
    dimension = source.shape[1] - 1
    reserve_exact_isometry_work(source.size, budget=budget)
    matrices = tuple(
        tuple(tuple(Fraction(float(v)) for v in row) for row in matrix)
        for matrix in source
    )
    identity = exact_isometry_identity(dimension, budget=budget)
    orders: list[int] = []
    linear_orders: list[int] = []
    periods: list[int] = []
    vectors: list[tuple[Fraction, ...]] = []
    for index, matrix in enumerate(matrices):
        generator = _classify_exact_isometry_generator(
            index, matrix, identity, budget=budget
        )
        orders.append(generator.order)
        linear_orders.append(generator.linear_order)
        periods.append(generator.translation_period)
        if generator.translation is not None:
            vectors.append(generator.translation)
    if any(
        exact_isometry_multiply(a, b, budget=budget)
        != exact_isometry_multiply(b, a, budget=budget)
        for i, a in enumerate(matrices)
        for b in matrices[i + 1 :]
    ):
        raise PeriodicIsometryIdentityError(
            "PeriodicIsometryGroup.generators must commute in rational arithmetic; got noncommuting maps"
        )
    _require_faithful_isometry_exponents(
        matrices, tuple(orders), tuple(linear_orders), vectors, identity, budget=budget
    )
    return ExactIsometryAdmission(
        matrices, tuple(orders), tuple(linear_orders), tuple(periods)
    )


@checked
def exact_periodic_generators(
    cell: PeriodicCell | PeriodicIsometryGroup, *, budget: CoordinateEnclosureBudget
) -> tuple[tuple[tuple[tuple[Fraction, ...], ...], ...], tuple[int, ...]]:
    dimension = cell.ambient_dimension
    match cell:
        case PeriodicCell():
            generators = []
            for vector in np.asarray(cell.vectors):
                matrix = [
                    list(row) for row in exact_isometry_identity(dimension, budget=budget)
                ]
                for i, value in enumerate(vector):
                    reserve_exact_isometry_work(1, budget=budget)
                    matrix[i][-1] = Fraction(float(value))
                generators.append(tuple(tuple(row) for row in matrix))
            return tuple(generators), tuple(
                0 if axis else 1 for axis in cell.periodic_axes
            )
        case PeriodicIsometryGroup():
            matrices = np.asarray(cell.generators)
            if (
                matrices.shape != (cell.rank, cell.ambient_dimension + 1, cell.ambient_dimension + 1)
                or _isometry_group_content_id(
                    matrices, cell.orders, cell.linear_orders,
                    cell.translation_periods, cell.tolerance,
                ) != cell.group_id
            ):
                raise PeriodicIsometryIdentityError(
                    "PeriodicIsometryGroup.group_id must bind current generators and metadata before admission reuse; got changed content"
                )
            # ``group_id`` binds the full binary64 generators and metadata; an
            # admission earned under this request's budget is reused verbatim.
            earned = budget.periodic_generator_cache.get(cell.group_id)
            if earned is not None:
                return earned
            admission = admit_exact_isometry_group(matrices, budget=budget)
            if (
                admission.orders,
                admission.linear_orders,
                admission.translation_periods,
            ) != (cell.orders, cell.linear_orders, cell.translation_periods):
                raise PeriodicIsometryIdentityError(
                    "PeriodicIsometryGroup.orders must follow retained generator actions with linear_orders and translation_periods; got mismatched metadata"
                )
            earned = (admission.matrices, admission.orders)
            budget.retain_owner(earned)
            budget.periodic_generator_cache[cell.group_id] = earned
            return earned
        case _:
            raise TypeError(
                "periodic_identification.cell must be a PeriodicCell or PeriodicIsometryGroup; got another identification kind"
            )


@checked
def exact_periodic_element(
    matrices: tuple[tuple[tuple[Fraction, ...], ...], ...],
    orders: tuple[int, ...],
    exponents: tuple[int, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Fraction, ...], ...]:
    result = exact_isometry_identity(len(matrices[0]) - 1, budget=budget)
    for matrix, order, exponent in zip(matrices, orders, exponents, strict=True):
        result = exact_isometry_multiply(
            result,
            exact_isometry_power(
                matrix, exponent % order if order else exponent, budget=budget
            ),
            budget=budget,
        )
    return result


type ExactIsometry = tuple[tuple[Fraction, ...], ...]


class ExactPeriodicActionLattice:
    """Evaluate ``exact_periodic_element`` for a bank of exponents once each.

    The element of exponents ``e`` is the ordered word
    ``G_0^{e_0} ... G_{r-1}^{e_{r-1}}`` with finite exponents reduced modulo
    their orders and negative infinite-order exponents using the exact inverse.
    When every exponent after axis ``g`` vanishes, that word equals the word one
    step nearer zero on axis ``g`` right-multiplied by ``G_g`` or its inverse:
    rational associativity alone gives the identity, so generator commutation
    is never assumed. Requested reduced exponents are visited in the
    deterministic order (absolute sum, exponents); an element whose neighbour
    is retained costs one multiplication, and any other is evaluated directly,
    so the charged work never exceeds direct evaluation of each distinct
    element. Instances live for one caller request on its ledger.
    """

    def __init__(
        self, matrices: tuple[ExactIsometry, ...], orders: tuple[int, ...], /
    ) -> None:
        if not matrices or len(matrices) != len(orders):
            raise ValueError(
                "ExactPeriodicActionLattice.orders must align with nonempty generators; "
                f"got {len(matrices)} generators and {len(orders)} orders"
            )
        self.matrices, self.orders = matrices, tuple(int(order) for order in orders)
        self.size = len(matrices[0])
        self.elements: dict[tuple[int, ...], ExactIsometry] = {}
        self.inverses: dict[int, ExactIsometry] = {}

    def key(self, exponents: Sequence[int], /) -> tuple[int, ...]:
        """Return the reduced exponent key naming one group element."""
        values = tuple(int(value) for value in exponents)
        if len(values) != len(self.orders):
            raise ValueError(
                "ExactPeriodicActionLattice.exponents must have one entry per generator; "
                f"got {len(values)}"
            )
        return tuple(
            value % order if order else value
            for value, order in zip(values, self.orders, strict=True)
        )

    def prepare(
        self,
        exponents: Sequence[Sequence[int]],
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[ExactIsometry, ...]:
        """Admit the planned exact work, then return elements in caller order."""
        keys = tuple(self.key(row) for row in exponents)
        pending = sorted(
            set(keys) - set(self.elements),
            key=lambda value: (sum(map(abs, value)), value),
        )
        budget.admit_work_bound(self._planned_work(pending))
        for key in pending:
            self.elements[key] = self._element(key, budget=budget)
        return tuple(self.elements[key] for key in keys)

    def _nearer(self, key: tuple[int, ...], /) -> tuple[int, tuple[int, ...]] | None:
        axis = next((index for index in reversed(range(len(key))) if key[index]), None)
        if axis is None:
            return None
        step = 1 if key[axis] > 0 else -1
        return axis, tuple(
            value - step if index == axis else value for index, value in enumerate(key)
        )

    def _planned_work(self, pending: Sequence[tuple[int, ...]], /) -> int:
        square, cube = self.size**2, self.size**3
        known, inverses, work = set(self.elements), set(self.inverses), 0
        for key in pending:
            nearer = self._nearer(key)
            if nearer is None:
                work += square
            elif nearer[1] in known or not any(nearer[1]):
                if nearer[1] not in known:
                    work += square
                    known.add(nearer[1])
                if key[nearer[0]] < 0 and nearer[0] not in inverses:
                    work += 2 * cube + square
                    inverses.add(nearer[0])
                work += cube
            else:
                work += square
                for exponent in key:
                    magnitude = abs(exponent)
                    work += square + cube * (exponent < 0) + cube
                    if magnitude:
                        work += cube * (magnitude.bit_count() + magnitude.bit_length() - 1)
            known.add(key)
        return work

    def _element(
        self, key: tuple[int, ...], /, *, budget: CoordinateEnclosureBudget
    ) -> ExactIsometry:
        nearer = self._nearer(key)
        if nearer is None:
            return exact_isometry_identity(self.size - 1, budget=budget)
        axis, previous = nearer
        neighbour = self.elements.get(previous)
        if neighbour is None and not any(previous):
            neighbour = self.elements[previous] = exact_isometry_identity(
                self.size - 1, budget=budget
            )
        if neighbour is None:
            return exact_periodic_element(self.matrices, self.orders, key, budget=budget)
        if key[axis] > 0:
            factor = self.matrices[axis]
        else:
            inverse = self.inverses.get(axis)
            if inverse is None:
                inverse = self.inverses[axis] = exact_isometry_power(
                    self.matrices[axis], -1, budget=budget
                )
            factor = inverse
        return exact_isometry_multiply(neighbour, factor, budget=budget)


type PeriodicMeshIdentification = PeriodicCell | PeriodicIsometryGroup
