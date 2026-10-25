#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sparse/operator symmetry tangents, retractions and residuals for relaxation.

Two symmetry-constrained charts share one operator form.  The *rigid* chart
moves each ASU molecule of an `AsymmetricUnitPlan` by its site parameters
(fractional ``W`` directions of its `SitePosition`), by a rotation restricted
to the axial vectors fixed by ``det(R_h) R_h`` over its stabilizer, and moves
the cell through the invariant metric chart.  The *atomic* chart moves every
symmetry-distinct atom (an atom-level asymmetric unit derived from the
admitted site permutations) along its exact atom-site directions and moves
the cell with fractional coordinates held fixed.

Tangent vectors map to the flat Cartesian vector ``(positions (N, 3), dH (3,
3))`` through a `SymmetryTangentOperator`: a native `SparseCoordinateOperator`
over per-atom routes (at most ``3 x 6`` coefficients per expanded atom) plus a
matrix-free strain action prepared from ``K <= 6`` small ``3 x 3`` generators.
No dense ``(3N + 9) x K`` basis is formed by any action; dense materialization
exists only through the native materialization policy.

For chart coordinates ``c`` with metric ``G = sum c_k B_k`` and lower Cholesky
row cell ``L`` (``L L^T = G``), ``dL = L Phi(L^-1 B_k L^-T)`` with ``Phi`` the
lower triangle with halved diagonal; the strain generator is
``X_k = Phi(L^-1 B_k L^-T)^T`` so a lattice-fixed point moves by ``X_k r`` and
the cell by ``L X_k^T``.  Rigid molecule offsets follow the polar rotation of
the frame change, ``Omega_k = (X_k - X_k^T) / 2``, which keeps special-site
orientations compatible exactly.  Retractions are genuine: affine for site
parameters and the linear metric chart, quaternion exponential for
orientations and the polar frame rotation for rigid molecules under strain.
"""

from __future__ import annotations

from typing import final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    OperatorCapabilities,
    OperatorProperties,
    polar_rotation3,
    SmallLinearSolvePlan,
    solve_small_linear,
)
from ...metrix import ScalarFirstQuaternionStateGeometry, SpecialOrthogonalGroup
from ...sparse import EdgeRelation, SparseCoordinateOperator
from ...typing import Dim, HostFloat64, HostInt64
from ._asu import (
    AsymmetricSiteDim,
    AsymmetricUnitLayout,
    AsymmetricUnitPlan,
    AsymmetricUnitState,
    expand_asymmetric_unit,
    ExpandedAtomDim,
    ExpandedCrystal,
)
from ._hall import (
    IntegerMatrix,
    multiply_rotations,
    negate_rotation,
    rotate_vector,
    rotation_determinant,
)
from ._lattice import SettingCellResult, SettingMetricChart
from ._wyckoff import _integer_kernel, _minus_identity, LatticeShift


_SMALL3 = SmallLinearSolvePlan(3)
_ORIENTATION = ScalarFirstQuaternionStateGeometry(convention="spatial")
_SO3 = SpecialOrthogonalGroup(3)
# A frame change is refused beyond a rotation angle of 0.9 pi (trace bound).
_MINIMUM_FRAME_TRACE = 1.0 + 2.0 * float(np.cos(0.9 * np.pi))


class SymmetryRouteDim(Dim, minimum=0):
    """Scalar sparse routes of a symmetry tangent."""


class RepresentativeDim(Dim, minimum=1):
    """Symmetry-distinct representative atoms."""


def _axial_directions(
    rotations: tuple[IntegerMatrix, ...], /
) -> tuple[LatticeShift, ...]:
    """Return the exact fractional axial vectors fixed by ``det(R) R`` for every ``R``."""
    rows = [
        row
        for rotation in rotations
        for row in _minus_identity(
            rotation if rotation_determinant(rotation) > 0 else negate_rotation(rotation)
        )
    ]
    return _integer_kernel(rows)


def _padded(directions: tuple[LatticeShift, ...], /) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for column, direction in enumerate(directions):
        matrix[:, column] = direction
    return matrix


def _hat(vectors: Array, /) -> Array:
    """Return the cross-product matrices ``[v]_x`` of vectors ``(..., 3)``."""
    zero = jnp.zeros_like(vectors[..., 0])
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    return jnp.stack(
        (
            jnp.stack((zero, -z, y), axis=-1),
            jnp.stack((z, zero, -x), axis=-1),
            jnp.stack((-y, x, zero), axis=-1),
        ),
        axis=-2,
    )


def strain_generators(chart: SettingMetricChart, vectors: ArrayLike, /) -> Array:
    """Return ``X_k = Phi(L^-1 B_k L^-T)^T`` for every chart basis metric ``B_k``.

    ``vectors`` is the lower-triangular Cholesky row cell of the chart; the
    two triangular solves use the native small solver.
    """
    cell = jnp.asarray(vectors)
    basis = chart.basis_metrics(cell.dtype)
    count = basis.shape[0]
    lower = jnp.broadcast_to(cell, (count, 3, 3))
    left = solve_small_linear(_SMALL3, lower, basis).value
    middle = solve_small_linear(_SMALL3, lower, jnp.swapaxes(left, -1, -2)).value
    middle = jnp.swapaxes(middle, -1, -2)
    phi = jnp.tril(middle) - 0.5 * jnp.eye(3, dtype=cell.dtype) * middle
    return jnp.swapaxes(phi, -1, -2)


@final
class SymmetryTangentOperator(AbstractLinearOperator):
    """Tangent map of a symmetry chart onto flat ``(positions, dH)`` Cartesian vectors.

    ``site_operator`` carries every site/orientation route; the strain part
    applies ``r_anchor @ X^T + r_offset @ Omega^T`` per atom and ``H X^T`` to
    the cell.  The source is ``ArraySpace((parameters,))`` and the target
    ``ArraySpace((3 N + 9,))`` with Euclidean pairings, positions first and
    the cell row-major.
    """

    site_operator: SparseCoordinateOperator
    generators: Array
    anchors: Array
    offsets: Array
    vectors: Array
    strain_start: int = eqx.field(static=True)
    atom_count: int = eqx.field(static=True)
    route_count: int = eqx.field(static=True)

    def __init__(
        self,
        site_operator: SparseCoordinateOperator,
        generators: Array,
        anchors: Array,
        offsets: Array,
        vectors: Array,
        /,
        *,
        strain_start: int,
        layout_id: str,
    ) -> None:
        count = anchors.shape[0]
        parameters = site_operator.source.size
        if generators.shape != (parameters - strain_start, 3, 3):
            raise ValueError("Strain generators do not match the parameter layout.")
        if anchors.shape != (count, 3) or offsets.shape != (count, 3):
            raise ValueError("Strain anchors and offsets must have shape (atoms, 3).")
        if site_operator.target.size != 3 * count + 9:
            raise ValueError("The site operator target must be (3 N + 9) coordinates.")
        self.site_operator = site_operator
        self.generators = generators
        self.anchors = anchors
        self.offsets = offsets
        self.vectors = vectors
        self.strain_start = strain_start
        self.atom_count = count
        self.route_count = int(site_operator.relation.source_indices.shape[0])
        self.source = site_operator.source
        self.target = site_operator.target
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=True
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {"kind": "crystal-symmetry-tangent", "layout_id": layout_id}
        )

    def _strain(self, coordinates: Array, /) -> Array:
        combined = contract("k,kij->ij", coordinates, self.generators, backend="jax")
        rotation = 0.5 * (combined - combined.T)
        moved = contract("ai,ji->aj", self.anchors, combined, backend="jax") + contract(
            "ai,ji->aj", self.offsets, rotation, backend="jax"
        )
        cell = contract("ij,kj->ik", self.vectors, combined, backend="jax")
        return jnp.concatenate((moved.reshape((-1,)), cell.reshape((-1,))))

    def _strain_transpose(self, vector: Array, /) -> Array:
        forces = vector[: 3 * self.atom_count].reshape((self.atom_count, 3))
        cell = vector[3 * self.atom_count :].reshape((3, 3))
        anchor = contract("ai,aj->ij", forces, self.anchors, backend="jax")
        offset = contract("ai,aj->ij", forces, self.offsets, backend="jax")
        total = (
            anchor
            + 0.5 * (offset - offset.T)
            + contract("ki,kj->ij", cell, self.vectors, backend="jax")
        )
        return contract("kij,ij->k", self.generators, total, backend="jax")

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(vector)
        return self.site_operator.mv(value) + self._strain(value[self.strain_start :])

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        strain = self._strain_transpose(value)
        return self.site_operator.transpose_mv(value).at[self.strain_start :].add(strain)

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        covector = self.target.riesz(self.target.validate(vector))
        return self.source.inverse_riesz(jnp.conj(self.transpose_mv(jnp.conj(covector))))

    def _materialize(self, /) -> Array:
        dense = self.site_operator.as_dense()
        strain = jax.vmap(self._strain, out_axes=1)(
            jnp.eye(self.generators.shape[0], dtype=self.generators.dtype)
        )
        return dense.at[:, self.strain_start :].add(strain)

    def split(self, vector: ArrayLike, /) -> tuple[Array, Array]:
        """Split a flat target vector into positions ``(N, 3)`` and ``dH (3, 3)``."""
        value = self.target.validate(vector)
        return (
            value[: 3 * self.atom_count].reshape((self.atom_count, 3)),
            value[3 * self.atom_count :].reshape((3, 3)),
        )

    def join(self, positions: ArrayLike, cell: ArrayLike, /) -> Array:
        """Join Cartesian position and cell (co)vectors into one flat target vector."""
        moved = jnp.asarray(positions)
        cell_ = jnp.asarray(cell)
        if moved.shape != (self.atom_count, 3) or cell_.shape != (3, 3):
            raise ValueError("positions must be (atoms, 3) and cell (3, 3).")
        return jnp.concatenate((moved.reshape((-1,)), cell_.reshape((-1,))))


def _sparse_operator(
    route_source: np.ndarray,
    route_atom: np.ndarray,
    route_alpha: np.ndarray,
    coefficients: Array,
    parameters: int,
    atoms: int,
    layout_id: str,
    /,
) -> SparseCoordinateOperator:
    relation = EdgeRelation(
        jnp.asarray(route_source, dtype=jnp.int32),
        jnp.asarray(3 * route_atom + route_alpha, dtype=jnp.int32),
        source_size=parameters,
        target_size=3 * atoms + 9,
    )
    return SparseCoordinateOperator(
        relation,
        coefficients,
        source=ArraySpace((parameters,), dtype=coefficients.dtype),
        target=ArraySpace((3 * atoms + 9,), dtype=coefficients.dtype),
        operator_id=canonical_fingerprint(
            {"kind": "crystal-symmetry-routes", "layout_id": layout_id}
        ),
    )


@final
class RigidSymmetryLayout(StrictModule, NonTrainableState):
    """Static parameter layout and sparse routes of the rigid ASU chart.

    Per ASU site the parameters are its ``center_dimension`` site parameters
    followed by its ``orientation_dimension`` axial rotation parameters (none
    for a one-atom component); the ``chart.dimension`` metric coordinates
    come last, starting at ``strain_start``.  Built from a host
    `AsymmetricUnitPlan`, it keeps only the plan's atom-level symmetry
    (``atomic``), whose prepared `AsymmetricUnitLayout` is ``unit``, never the
    plan's molecular witnesses.
    """

    __strict_contract__ = True

    atomic: AtomicSymmetryPlan
    site_directions: HostFloat64[AsymmetricSiteDim, Literal[3], Literal[3]]
    site_axes: HostFloat64[AsymmetricSiteDim, Literal[3], Literal[3]]
    site_axes_cartesian: tuple[bool, ...] = eqx.field(static=True)
    center_dimension: tuple[int, ...] = eqx.field(static=True)
    orientation_dimension: tuple[int, ...] = eqx.field(static=True)
    site_start: tuple[int, ...] = eqx.field(static=True)
    strain_start: int = eqx.field(static=True)
    parameter_count: int = eqx.field(static=True)
    route_source: HostInt64[SymmetryRouteDim]
    route_atom: HostInt64[SymmetryRouteDim]
    route_alpha: HostInt64[SymmetryRouteDim]
    route_column: HostInt64[SymmetryRouteDim]
    layout_id: str = eqx.field(static=True)

    def __init__(self, plan: AsymmetricUnitPlan, /) -> None:
        if not isinstance(plan, AsymmetricUnitPlan):
            raise TypeError("plan must be an AsymmetricUnitPlan.")
        operations = plan.setting.exact_operations()
        directions = np.zeros((plan.site_count, 3, 3), dtype=np.float64)
        axes = np.zeros((plan.site_count, 3, 3), dtype=np.float64)
        cartesian, centers, orientations, starts = [], [], [], []
        start = 0
        for index, site in enumerate(plan.sites):
            directions[index] = _padded(site.site.directions)
            rotations = tuple(operations[k][0] for k, _ in site.site.stabilizer)
            axial = (
                _axial_directions(rotations) if site.pool.component.atom_count > 1 else ()
            )
            full = len(axial) == 3
            if not full:
                axes[index] = _padded(axial)
            else:
                axes[index] = np.eye(3)
            cartesian.append(full)
            centers.append(site.site.dimension)
            orientations.append(len(axial))
            starts.append(start)
            start += site.site.dimension + len(axial)
        source, atom, alpha, column = [], [], [], []
        for a in range(plan.atom_count):
            m = int(plan.atom_site[a])
            for j in range(centers[m] + orientations[m]):
                for row in range(3):
                    source.append(starts[m] + j)
                    atom.append(a)
                    alpha.append(row)
                    column.append(j if j < centers[m] else 3 + j - centers[m])
        self.atomic = AtomicSymmetryPlan(plan)
        self.site_directions = directions
        self.site_axes = axes
        self.site_axes_cartesian = tuple(cartesian)
        self.center_dimension = tuple(centers)
        self.orientation_dimension = tuple(orientations)
        self.site_start = tuple(starts)
        self.strain_start = start
        self.parameter_count = start + plan.chart.dimension
        self.route_source = np.asarray(source, dtype=np.int64)
        self.route_atom = np.asarray(atom, dtype=np.int64)
        self.route_alpha = np.asarray(alpha, dtype=np.int64)
        self.route_column = np.asarray(column, dtype=np.int64)
        self.layout_id = canonical_fingerprint(
            {"kind": "crystal-rigid-symmetry-layout", "plan_id": plan.plan_id}
        )

    @property
    def unit(self) -> AsymmetricUnitLayout:
        """The prepared ASU layout (shared with ``atomic``)."""
        return self.atomic.unit

    def _axes(self, vectors: Array, /) -> Array:
        """Return Cartesian axial bases ``(sites, 3, 3)`` at the cell ``vectors``."""
        fractional = jnp.asarray(self.site_axes, dtype=vectors.dtype)
        mapped = contract("ji,mjk->mik", vectors, fractional, backend="jax")
        mask = jnp.asarray(self.site_axes_cartesian)[:, None, None]
        return jnp.where(mask, fractional, mapped)

    def _site_steps(self, step: Array, /) -> tuple[Array, Array]:
        centers = np.zeros((self.unit.site_count, 3), dtype=np.int64)
        orientations = np.zeros((self.unit.site_count, 3), dtype=np.int64)
        center_mask = np.zeros((self.unit.site_count, 3), dtype=np.bool_)
        orientation_mask = np.zeros((self.unit.site_count, 3), dtype=np.bool_)
        for m, start in enumerate(self.site_start):
            d, e = self.center_dimension[m], self.orientation_dimension[m]
            centers[m, :d] = np.arange(start, start + d)
            center_mask[m, :d] = True
            orientations[m, :e] = np.arange(start + d, start + d + e)
            orientation_mask[m, :e] = True
        zero = jnp.zeros((), dtype=step.dtype)
        return (
            jnp.where(jnp.asarray(center_mask), step[jnp.asarray(centers)], zero),
            jnp.where(
                jnp.asarray(orientation_mask), step[jnp.asarray(orientations)], zero
            ),
        )

    def tangent(
        self, state: AsymmetricUnitState, /
    ) -> tuple[SymmetryTangentOperator, ExpandedCrystal]:
        """Prepare the tangent operator at ``state`` and return it with its expansion."""
        plan = self.unit
        expansion = expand_asymmetric_unit(plan, state)
        dtype = expansion.positions.dtype
        vectors = expansion.cell.vectors
        rotations = jnp.asarray(plan.rotations, dtype=dtype)
        atom_site = jnp.asarray(plan.atom_site, dtype=jnp.int32)
        atom_operation = jnp.asarray(plan.atom_operation, dtype=jnp.int32)
        center_block = contract(
            "ji,ajk,akl->ail",
            vectors,
            rotations[atom_operation],
            jnp.asarray(self.site_directions, dtype=dtype)[atom_site],
            backend="jax",
        )
        offsets = expansion.site_offsets[
            atom_site, jnp.asarray(plan.atom_local, dtype=jnp.int32)
        ]
        orientation_block = -contract(
            "aij,ajk,akl->ail",
            expansion.cell.cartesian_rotations[atom_operation],
            _hat(offsets),
            self._axes(vectors)[atom_site],
            backend="jax",
        )
        block = jnp.concatenate((center_block, orientation_block), axis=-1)
        coefficients = block[
            jnp.asarray(self.route_atom, dtype=jnp.int32),
            jnp.asarray(self.route_alpha, dtype=jnp.int32),
            jnp.asarray(self.route_column, dtype=jnp.int32),
        ]
        site_operator = _sparse_operator(
            self.route_source,
            self.route_atom,
            self.route_alpha,
            coefficients,
            self.parameter_count,
            plan.atom_count,
            self.layout_id,
        )
        anchors = expansion.copy_center_positions[
            jnp.asarray(plan.atom_copy, dtype=jnp.int32)
        ]
        operator = SymmetryTangentOperator(
            site_operator,
            strain_generators(plan.chart, vectors),
            anchors,
            expansion.positions - anchors,
            vectors,
            strain_start=self.strain_start,
            layout_id=self.layout_id,
        )
        return operator, expansion

    def retract(
        self, state: AsymmetricUnitState, step: ArrayLike, /
    ) -> AsymmetricUnitState:
        """Return the state moved along ``step`` by the genuine rigid retraction.

        Centers move affinely along their site directions, orientations by the
        spatial quaternion exponential of their admitted axial rotation, and
        the metric chart linearly; the polar rotation of the resulting change of
        Cartesian frame is then applied to every orientation.  A refused frame
        change (failed solve or polar admission, or a rotation beyond
        ``0.9 pi``) leaves NaN orientations, so the expansion reports failed
        offset solves rather than a repaired state.
        """
        values = jnp.asarray(step, dtype=state.centers.dtype)
        if values.shape != (self.parameter_count,):
            raise ValueError(f"step must have shape ({self.parameter_count},).")
        cell = self.unit.chart.cell(state.cell_coefficients)
        center_step, orientation_step = self._site_steps(values)
        centers = state.centers + contract(
            "mij,mj->mi",
            jnp.asarray(self.site_directions, dtype=values.dtype),
            center_step,
            backend="jax",
        )
        rotation_vectors = contract(
            "mij,mj->mi", self._axes(cell.vectors), orientation_step, backend="jax"
        )
        orientations = jax.vmap(_ORIENTATION.retract)(
            state.orientations, rotation_vectors
        )
        coefficients = state.cell_coefficients + values[self.strain_start :]
        moved = self.unit.chart.cell(coefficients)
        frame, admitted = _frame_rotation_vector(cell, moved)
        rotated = jax.vmap(_ORIENTATION.retract, in_axes=(0, None))(
            orientations, jnp.where(admitted, frame, 0.0)
        )
        orientations = jnp.where(admitted, rotated, jnp.nan)
        return AsymmetricUnitState(
            centers, orientations, coefficients, state.conformer_indices
        )


def _frame_rotation_vector(
    old: SettingCellResult, new: SettingCellResult, /
) -> tuple[Array, Array]:
    """Return the polar rotation vector of ``H_new^T H_old^-T`` and its admission.

    The frame change is admitted when the native solve succeeds, its polar
    rotation is admitted (finite, nonsingular, proper) and the rotation angle
    is at most ``0.9 pi``; a refused vector is NaN.
    """
    relative = solve_small_linear(_SMALL3, old.vectors, new.vectors)
    polar = polar_rotation3(relative.value.T)
    rotation = polar.value
    admitted = (
        relative.successful
        & polar.successful
        & (jnp.trace(rotation) >= _MINIMUM_FRAME_TRACE)
    )
    principal = jnp.where(admitted, rotation, jnp.eye(3, dtype=rotation.dtype))
    vector = _SO3.vee(_SO3.log(principal))
    return jnp.where(admitted, vector, jnp.nan), admitted


@final
class AtomicSymmetryPlan(StrictModule, NonTrainableState):
    """Atom-level asymmetric unit of an ASU plan with exact atom-site directions.

    Atoms of a special-site molecule related by the admitted site
    permutations form one orbit; its least atom is the representative and
    moves along the directions fixed by the stabilizer elements that fix it.
    Expanded atom ``a`` is ``x_a = R_a x_rep(a) + tau_a + n_copy(a)``.  Built
    from a host `AsymmetricUnitPlan`, it keeps only the prepared
    `AsymmetricUnitLayout` (``unit``).
    """

    __strict_contract__ = True

    unit: AsymmetricUnitLayout
    representative_site: HostInt64[RepresentativeDim]
    representative_local: HostInt64[RepresentativeDim]
    representative_directions: HostFloat64[RepresentativeDim, Literal[3], Literal[3]]
    representative_atom: HostInt64[RepresentativeDim]
    atom_representative: HostInt64[ExpandedAtomDim]
    atom_rotation: HostInt64[ExpandedAtomDim, Literal[3], Literal[3]]
    atom_translation: HostFloat64[ExpandedAtomDim, Literal[3]]
    representative_dimension: tuple[int, ...] = eqx.field(static=True)
    representative_start: tuple[int, ...] = eqx.field(static=True)
    strain_start: int = eqx.field(static=True)
    parameter_count: int = eqx.field(static=True)
    route_source: HostInt64[SymmetryRouteDim]
    route_atom: HostInt64[SymmetryRouteDim]
    route_alpha: HostInt64[SymmetryRouteDim]
    route_column: HostInt64[SymmetryRouteDim]
    layout_id: str = eqx.field(static=True)

    def __init__(self, plan: AsymmetricUnitPlan, /) -> None:
        if not isinstance(plan, AsymmetricUnitPlan):
            raise TypeError("plan must be an AsymmetricUnitPlan.")
        operations = plan.setting.exact_operations()
        rep_site, rep_local, rep_directions, rep_dimension = [], [], [], []
        # (site, local atom) -> (representative index, stabilizer element index)
        assignment: dict[tuple[int, int], tuple[int, int]] = {}
        for m, site in enumerate(plan.sites):
            count = site.pool.component.atom_count
            for i in range(count):
                if (m, i) in assignment:
                    continue
                representative = len(rep_site)
                fixing = []
                for e, permutation in enumerate(site.site_permutations):
                    j = permutation[i]
                    if (m, j) not in assignment:
                        assignment[(m, j)] = (representative, e)
                    if j == i:
                        fixing.append(operations[site.site.stabilizer[e][0]][0])
                rows = [row for rotation in fixing for row in _minus_identity(rotation)]
                directions = _integer_kernel(rows)
                rep_site.append(m)
                rep_local.append(i)
                rep_directions.append(_padded(directions).astype(np.float64))
                rep_dimension.append(len(directions))
        starts, start = [], 0
        for dimension in rep_dimension:
            starts.append(start)
            start += dimension
        count = plan.atom_count
        atom_rep = np.zeros((count,), dtype=np.int64)
        atom_rotation = np.zeros((count, 3, 3), dtype=np.int64)
        atom_translation = np.zeros((count, 3), dtype=np.float64)
        representative_atom = np.full((len(rep_site),), -1, dtype=np.int64)
        for a in range(count):
            m, j = int(plan.atom_site[a]), int(plan.atom_local[a])
            representative, e = assignment[(m, j)]
            k, shift = plan.sites[m].site.stabilizer[e]
            h_rotation, h_translation = operations[k]
            g_rotation, g_translation = operations[int(plan.atom_operation[a])]
            shifted = (
                h_translation[0] + shift[0],
                h_translation[1] + shift[1],
                h_translation[2] + shift[2],
            )
            moved = rotate_vector(g_rotation, shifted)
            atom_rep[a] = representative
            atom_rotation[a] = np.asarray(
                multiply_rotations(g_rotation, h_rotation)
            ).reshape((3, 3))
            atom_translation[a] = [float(moved[i] + g_translation[i]) for i in range(3)]
            if (
                representative_atom[representative] < 0
                and int(plan.atom_operation[a]) == 0
                and e == 0
            ):
                representative_atom[representative] = a
        if np.any(representative_atom < 0):
            raise RuntimeError("A representative atom has no identity-copy image.")
        source, atom, alpha, column = [], [], [], []
        for a in range(count):
            representative = int(atom_rep[a])
            for j in range(rep_dimension[representative]):
                for row in range(3):
                    source.append(starts[representative] + j)
                    atom.append(a)
                    alpha.append(row)
                    column.append(j)
        self.unit = AsymmetricUnitLayout(plan)
        self.representative_site = np.asarray(rep_site, dtype=np.int64)
        self.representative_local = np.asarray(rep_local, dtype=np.int64)
        self.representative_directions = np.stack(rep_directions)
        self.representative_atom = representative_atom
        self.atom_representative = atom_rep
        self.atom_rotation = atom_rotation
        self.atom_translation = atom_translation
        self.representative_dimension = tuple(rep_dimension)
        self.representative_start = tuple(starts)
        self.strain_start = start
        self.parameter_count = start + plan.chart.dimension
        self.route_source = np.asarray(source, dtype=np.int64)
        self.route_atom = np.asarray(atom, dtype=np.int64)
        self.route_alpha = np.asarray(alpha, dtype=np.int64)
        self.route_column = np.asarray(column, dtype=np.int64)
        self.layout_id = canonical_fingerprint(
            {"kind": "crystal-atomic-symmetry-layout", "plan_id": plan.plan_id}
        )

    @property
    def representative_count(self) -> int:
        return int(self.representative_site.shape[0])


@final
class AtomicSymmetricState(StrictModule):
    """Fractional representative atoms, metric chart and frozen copy branches."""

    representatives: Array
    cell_coefficients: Array
    copy_shifts: Array

    def __init__(
        self,
        representatives: ArrayLike,
        cell_coefficients: ArrayLike,
        copy_shifts: ArrayLike,
    ) -> None:
        values = jnp.asarray(representatives)
        coefficients = jnp.asarray(cell_coefficients)
        shifts = jnp.asarray(copy_shifts)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("representatives must have shape (representatives, 3).")
        if coefficients.ndim != 1 or coefficients.dtype != values.dtype:
            raise ValueError(
                "cell_coefficients must be a vector of the coordinate dtype."
            )
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise TypeError("representatives must be real floating values.")
        if (
            shifts.ndim != 2
            or shifts.shape[1] != 3
            or not jnp.issubdtype(shifts.dtype, jnp.integer)
        ):
            raise ValueError("copy_shifts must be integers of shape (copies, 3).")
        self.representatives = values
        self.cell_coefficients = coefficients
        self.copy_shifts = shifts.astype(jnp.int32)


@final
class AtomicExpansion(StrictModule):
    """Expanded positions of an atomic symmetric state with cell evidence."""

    positions: Array
    fractional: Array
    cell: SettingCellResult
    layout_id: str = eqx.field(static=True)


def atomic_state_from_expansion(
    layout: AtomicSymmetryPlan, state: AsymmetricUnitState, expansion: ExpandedCrystal, /
) -> AtomicSymmetricState:
    """Return the atomic chart state realizing a rigid ASU expansion exactly."""
    if expansion.plan_id != layout.unit.plan_id:
        raise ValueError("expansion belongs to another plan.")
    atoms = jnp.asarray(layout.representative_atom, dtype=jnp.int32)
    copies = layout.unit.atom_copy[atoms]
    representatives = expansion.fractional[atoms] - expansion.copy_center_shifts[
        copies
    ].astype(expansion.fractional.dtype)
    return AtomicSymmetricState(
        representatives, state.cell_coefficients, expansion.copy_center_shifts
    )


def expand_symmetric_atoms(
    layout: AtomicSymmetryPlan, state: AtomicSymmetricState, /
) -> AtomicExpansion:
    """Expand representative atoms into the full cell on frozen copy branches."""
    if state.representatives.shape != (layout.representative_count, 3):
        raise ValueError(
            f"state must hold {layout.representative_count} representatives."
        )
    dtype = state.representatives.dtype
    cell = layout.unit.chart.cell(state.cell_coefficients)
    copies = layout.unit.atom_copy
    fractional = (
        contract(
            "aij,aj->ai",
            jnp.asarray(layout.atom_rotation, dtype=dtype),
            state.representatives[
                jnp.asarray(layout.atom_representative, dtype=jnp.int32)
            ],
            backend="jax",
        )
        + jnp.asarray(layout.atom_translation, dtype=dtype)
        + state.copy_shifts[copies].astype(dtype)
    )
    positions = contract("ai,ij->aj", fractional, cell.vectors, backend="jax")
    return AtomicExpansion(positions, fractional, cell, layout.layout_id)


def atomic_tangent(
    layout: AtomicSymmetryPlan, state: AtomicSymmetricState, /
) -> tuple[SymmetryTangentOperator, AtomicExpansion]:
    """Prepare the atomic-chart tangent operator at ``state``."""
    expansion = expand_symmetric_atoms(layout, state)
    dtype = expansion.positions.dtype
    block = contract(
        "ji,ajk,akl->ail",
        expansion.cell.vectors,
        jnp.asarray(layout.atom_rotation, dtype=dtype),
        jnp.asarray(layout.representative_directions, dtype=dtype)[
            jnp.asarray(layout.atom_representative, dtype=jnp.int32)
        ],
        backend="jax",
    )
    coefficients = block[
        jnp.asarray(layout.route_atom, dtype=jnp.int32),
        jnp.asarray(layout.route_alpha, dtype=jnp.int32),
        jnp.asarray(layout.route_column, dtype=jnp.int32),
    ]
    site_operator = _sparse_operator(
        layout.route_source,
        layout.route_atom,
        layout.route_alpha,
        coefficients,
        layout.parameter_count,
        layout.unit.atom_count,
        layout.layout_id,
    )
    operator = SymmetryTangentOperator(
        site_operator,
        strain_generators(layout.unit.chart, expansion.cell.vectors),
        expansion.positions,
        jnp.zeros_like(expansion.positions),
        expansion.cell.vectors,
        strain_start=layout.strain_start,
        layout_id=layout.layout_id,
    )
    return operator, expansion


def retract_atomic(
    layout: AtomicSymmetryPlan, state: AtomicSymmetricState, step: ArrayLike, /
) -> AtomicSymmetricState:
    """Move representatives along their exact site directions and the chart linearly."""
    values = jnp.asarray(step, dtype=state.representatives.dtype)
    if values.shape != (layout.parameter_count,):
        raise ValueError(f"step must have shape ({layout.parameter_count},).")
    indices = np.zeros((layout.representative_count, 3), dtype=np.int64)
    mask = np.zeros((layout.representative_count, 3), dtype=np.bool_)
    for r, (start, dimension) in enumerate(
        zip(layout.representative_start, layout.representative_dimension, strict=True)
    ):
        indices[r, :dimension] = np.arange(start, start + dimension)
        mask[r, :dimension] = True
    local = jnp.where(
        jnp.asarray(mask), values[jnp.asarray(indices)], jnp.zeros((), values.dtype)
    )
    moved = state.representatives + contract(
        "rij,rj->ri",
        jnp.asarray(layout.representative_directions, dtype=values.dtype),
        local,
        backend="jax",
    )
    return AtomicSymmetricState(
        moved, state.cell_coefficients + values[layout.strain_start :], state.copy_shifts
    )


def symmetry_residual(
    layout: AtomicSymmetryPlan, positions: ArrayLike, vectors: ArrayLike, /
) -> Array:
    """Return per-atom Cartesian deviation from the exact symmetric images.

    Each atom is compared with ``R_a s_rep + tau_a`` of its representative's
    own position, modulo lattice translations (componentwise nearest
    fractional image, exact while the residual is below half the shortest
    cell height).  Use this after a relaxation to verify that symmetry held
    or to measure how far an unconstrained run broke it.
    """
    cartesian = jnp.asarray(positions)
    cell = jnp.asarray(vectors, dtype=cartesian.dtype)
    if cartesian.shape != (layout.unit.atom_count, 3) or cell.shape != (3, 3):
        raise ValueError("positions must be (atoms, 3) and vectors (3, 3).")
    solved = solve_small_linear(_SMALL3, cell.T, cartesian.T)
    fractional = solved.value.T
    reference = fractional[
        jnp.asarray(layout.representative_atom, dtype=jnp.int32)[
            jnp.asarray(layout.atom_representative, dtype=jnp.int32)
        ]
    ]
    predicted = contract(
        "aij,aj->ai",
        jnp.asarray(layout.atom_rotation, dtype=cartesian.dtype),
        reference,
        backend="jax",
    ) + jnp.asarray(layout.atom_translation, dtype=cartesian.dtype)
    difference = fractional - predicted
    difference = difference - jnp.round(difference)
    return jnp.linalg.norm(
        contract("ai,ij->aj", difference, cell, backend="jax"), axis=-1
    )


@final
class SymmetryResidual(StrictModule):
    """Projected (chart) and full (Cartesian) residuals of one force evaluation.

    ``projected`` is ``T^T (f, F_H)``, the generalized force on the chart
    parameters; ``full_force`` and ``full_cell_force`` are the largest atomic
    force norm and the Frobenius norm of the cell covector.  A projected
    minimum is not a full minimum unless ``full_force`` is also small.
    """

    projected: Array
    projected_norm: Array
    full_force: Array
    full_cell_force: Array


def symmetry_residuals(
    operator: SymmetryTangentOperator, forces: ArrayLike, cell_covector: ArrayLike, /
) -> SymmetryResidual:
    """Return projected and full residuals of Cartesian forces and a cell covector."""
    force = jnp.asarray(forces)
    cell = jnp.asarray(cell_covector, dtype=force.dtype)
    projected = operator.transpose_mv(operator.join(force, cell))
    return SymmetryResidual(
        projected,
        jnp.linalg.norm(projected),
        jnp.max(jnp.linalg.norm(force, axis=-1)),
        jnp.linalg.norm(cell),
    )


__all__ = [
    "atomic_state_from_expansion",
    "atomic_tangent",
    "AtomicExpansion",
    "AtomicSymmetricState",
    "AtomicSymmetryPlan",
    "expand_symmetric_atoms",
    "retract_atomic",
    "RigidSymmetryLayout",
    "strain_generators",
    "symmetry_residual",
    "symmetry_residuals",
    "SymmetryResidual",
    "SymmetryTangentOperator",
]
