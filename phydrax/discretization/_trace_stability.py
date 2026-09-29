#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified discrete trace-inverse constants of pointwise conormal fluxes.

Nitsche and penalty impositions are coercive only when their penalty dominates
the discrete trace-inverse inequality

    ||q(v)||^2_{L2(F)} <= C_F a_K(v, v)   for every v in V_h(K),

between an owner's pointwise conormal flux ``q`` on a facet ``F`` and the
energy ``a_K`` of its physical operator on the side cell ``K``. The sharp
constant is the largest eigenvalue of the local pencil ``(B_F, A_K)`` on the
complement of the energy kernel, where ``B_F`` is the facet flux Gram matrix
and ``A_K`` the cell energy matrix. The constants here are those eigenvalues,
solved by the native batched dense Hermitian eigensolver; no hand-written
inverse-inequality formula is used.

The energy of a scalar diffusion operator vanishes on constants, and so does
its flux. With ``r = int_K phi`` (the cell moments of the local basis) the
deflated metric ``A_K + s r r^T`` is positive definite whenever the energy
kernel is one-dimensional, spanned by ``c`` with ``r^T c != 0``: every
``v = alpha c + w`` with a mean-free ``w`` gives ``v^T B v = w^T B w`` and
``v^T (A + s r r^T) v = w^T A w + s (alpha r^T c)^2``, so the largest
generalized eigenvalue equals ``C_F`` exactly for every ``s > 0``. Both
premises are certified rather than assumed: the energy kernel is solved first,
and a pencil whose energy has no kernel, a kernel of dimension above one, a
kernel invisible to the moments, or a kernel carrying flux is refused. A flux
``T c != 0`` has zero energy, so no finite trace-inverse constant exists; the
deflated pencil would silently report a finite one.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, nonnegative_integer
from ..linalg import DenseLinearOperator, OperatorProperties
from ..linalg.eigen import (
    DenseEigh,
    Eigenproblem,
    eigensolve,
    EigenSolvePolicy,
    GeneralizedEigenproblem,
)


@final
class TraceInverseEvidence(StrictModule, NonTrainableState):
    """Certified discrete trace-inverse constants of one pointwise conormal flux.

    For every selected facet (in the flux action's facet order) with side cell
    ``K``, ``constants`` holds ``C_F = max_{v in V_h(K)} ||q(v)||^2_F / a_K(v, v)``
    for the owner's pointwise flux ``q`` and cell energy ``a_K``.
    ``cell_multiplicity`` counts the selected facets that share each facet's
    side cell (a cell whose energy bounds several facet fluxes is shared among
    them). ``relative_residuals`` are the eigen-residuals of the certified
    modes. The facet norms were integrated by the facet rule ``facet_rule_id``,
    exact to ``facet_exact_degree`` for the flux degree the owner published.
    """

    owner_id: str = eqx.field(static=True)
    flux_action_id: str = eqx.field(static=True)
    facets: Array
    side_cells: Array
    constants: Array
    cell_multiplicity: Array
    relative_residuals: Array
    facet_rule_id: str = eqx.field(static=True)
    facet_exact_degree: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        owner_id: str,
        flux_action_id: str,
        facets: ArrayLike,
        side_cells: ArrayLike,
        constants: ArrayLike,
        relative_residuals: ArrayLike,
        facet_rule_id: str,
        facet_exact_degree: int,
    ) -> None:
        facets_ = np.asarray(facets)
        cells = np.asarray(side_cells)
        values = np.asarray(constants, dtype=np.float64)
        residuals = np.asarray(relative_residuals, dtype=np.float64)
        count = facets_.shape[0] if facets_.ndim == 1 else -1
        if (
            count < 1
            or not np.issubdtype(facets_.dtype, np.integer)
            or not np.issubdtype(cells.dtype, np.integer)
            or cells.shape != facets_.shape
            or values.shape != facets_.shape
            or residuals.shape != facets_.shape
        ):
            raise ValueError(
                "Trace-inverse evidence needs one side cell, constant, and residual "
                "per selected facet."
            )
        if not (np.all(np.isfinite(values)) and np.all(values >= 0.0)):
            raise ValueError("Trace-inverse constants must be finite and nonnegative.")
        _, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
        self.owner_id = canonical_identifier(owner_id, "owner_id")
        self.flux_action_id = canonical_identifier(flux_action_id, "flux_action_id")
        self.facets = jnp.asarray(facets_.astype(np.int32))
        self.side_cells = jnp.asarray(cells.astype(np.int32))
        self.constants = jnp.asarray(values)
        self.cell_multiplicity = jnp.asarray(counts[inverse].astype(np.int32))
        self.relative_residuals = jnp.asarray(residuals)
        self.facet_rule_id = canonical_identifier(facet_rule_id, "facet_rule_id")
        self.facet_exact_degree = nonnegative_integer(
            facet_exact_degree, "facet_exact_degree"
        )
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "trace-inverse-evidence",
                "owner": self.owner_id,
                "flux": self.flux_action_id,
                "facets": array_tree_fingerprint(facets_.astype(np.int32)),
                "constants": array_tree_fingerprint(values),
                "rule": self.facet_rule_id,
                "exact_degree": self.facet_exact_degree,
            }
        )


def _certify_energy_kernel(
    rows: Array,
    energy: Array,
    moments: Array,
    energy_scale: Array,
    self_adjoint: OperatorProperties,
    flux_action_id: str,
    /,
) -> None:
    """Certify the deflation premises: a one-dimensional energy kernel ``c``.

    ``c`` must be visible to the moments (``r^T c != 0``) and annihilated by
    the flux (``T c = 0``). The thresholds are backward-stable roundoff bounds:
    a dense Hermitian eigensolve perturbs eigenvalues by at most
    ``rho = 16 n eps ||A||_2 <= 16 n eps tr(A)`` for a positive semidefinite
    ``A``, and turns the kernel vector by at most ``rho / lambda_1``, so an
    exact kernel carries at most ``16 n eps ||T||_F (1 + tr(A) / lambda_1)``
    of flux. Anything larger is a physical flux of zero energy.
    """
    width = moments.shape[-1]
    modes = min(2, width)
    kernel = eigensolve(
        Eigenproblem(
            DenseLinearOperator(energy, properties=self_adjoint),
            problem_id=canonical_fingerprint(
                {"kind": "trace-inverse-energy-kernel", "flux": flux_action_id}
            ),
        ),
        policy=EigenSolvePolicy(DenseEigh(), count=modes, which="smallest-algebraic"),
    )
    if not bool(jnp.all(kernel.successful)):
        raise ValueError("The local energy-kernel eigensolve did not converge.")
    roundoff = 16.0 * width * jnp.finfo(energy.dtype).eps
    bound = roundoff * energy_scale
    values = kernel.eigenvalues
    if not bool(jnp.all(values[:, 0] <= bound)):
        raise ValueError(
            "The cell energy has no kernel; the constant-deflated trace-inverse "
            "pencil applies only to a diffusion energy that vanishes on constants."
        )
    gap = values[:, 1] if modes == 2 else energy_scale
    if not bool(jnp.all(gap > bound)):
        raise ValueError(
            "The cell energy kernel is larger than the constants; the deflated "
            "trace-inverse metric would be singular."
        )
    constant = kernel.eigenvectors[:, :, 0]
    visible = ein.contract("fl,fl->f", moments, constant) ** 2
    if not bool(jnp.all(visible > roundoff * jnp.sum(moments**2, axis=-1))):
        raise ValueError(
            "The cell moments do not see the energy kernel; the deflated "
            "trace-inverse metric would be singular."
        )
    flux = jnp.sqrt(jnp.sum(ein.contract("fql,fl->fq", rows, constant) ** 2, axis=-1))
    flux_scale = jnp.sqrt(jnp.sum(rows**2, axis=(-2, -1)))
    if not bool(jnp.all(flux <= roundoff * flux_scale * (1.0 + energy_scale / gap))):
        raise ValueError(
            "The flux does not vanish on the cell energy kernel: a zero-energy "
            "state carries flux, so no finite trace-inverse constant exists."
        )


def certify_trace_inverse(
    flux_rows: ArrayLike,
    energy: ArrayLike,
    moments: ArrayLike,
    valid: ArrayLike,
    /,
    *,
    owner_id: str,
    flux_action_id: str,
    facets: ArrayLike,
    side_cells: ArrayLike,
    facet_rule_id: str,
    facet_exact_degree: int,
) -> TraceInverseEvidence:
    """Solve the local trace-inverse pencils of one flux and certify their constants.

    ``flux_rows[f, q, l] = sqrt(w_q) q(phi_l)(x_q)`` samples the flux of every
    local basis function of facet ``f``'s side cell on an exact facet rule, so
    ``B_F = T^T T``. ``energy[f]`` is the side cell's energy matrix ``A_K`` and
    ``moments[f]`` its basis moments ``int_K phi_l``; ``valid[f, l]`` marks the
    real local slots of padded cells. Refuses nonfinite data, an energy whose
    kernel is not exactly one-dimensional and visible to the moments, a flux
    that does not annihilate that kernel (no finite constant exists), or a
    failed eigensolve.
    """
    rows = jnp.asarray(flux_rows)
    matrices = jnp.asarray(energy)
    cell_moments = jnp.asarray(moments)
    mask = jnp.asarray(valid, dtype=jnp.bool_)
    count, _, width = rows.shape if rows.ndim == 3 else (0, 0, 0)
    if (
        count < 1
        or matrices.shape != (count, width, width)
        or cell_moments.shape != (count, width)
        or mask.shape != (count, width)
    ):
        raise ValueError(
            "Trace-inverse data must be (facets, sites, local) flux rows with "
            "(facets, local, local) energies, (facets, local) moments and masks."
        )
    if not bool(
        jnp.all(jnp.isfinite(rows))
        & jnp.all(jnp.isfinite(matrices))
        & jnp.all(jnp.isfinite(cell_moments))
    ):
        raise ValueError("Trace-inverse data must be finite.")
    pair = mask[:, :, None] & mask[:, None, :]
    matrices = jnp.where(pair, matrices, 0.0)
    gram = jnp.where(pair, ein.contract("fql,fqm->flm", rows, rows), 0.0)
    rows = jnp.where(mask[:, None, :], rows, 0.0)
    moments_ = jnp.where(mask, cell_moments, 0.0)
    energy_scale = jnp.trace(matrices, axis1=-2, axis2=-1)
    moment_norm = jnp.sum(moments_**2, axis=-1)
    if not bool(jnp.all(energy_scale > 0.0) & jnp.all(moment_norm > 0.0)):
        raise ValueError(
            "Every trace-inverse cell needs a nonzero energy and nonzero basis "
            "moments; a vanishing energy bounds no flux."
        )
    self_adjoint = OperatorProperties(
        self_adjoint=True, evidence={"self_adjoint": "construction"}
    )
    identity = jnp.eye(width, dtype=matrices.dtype)
    # Padded slots get the energy scale on the diagonal: they stay out of the
    # kernel and never become the smallest nonzero energy mode.
    _certify_energy_kernel(
        rows,
        jnp.where(mask[:, :, None], matrices, energy_scale[:, None, None] * identity),
        moments_,
        energy_scale,
        self_adjoint,
        flux_action_id,
    )
    scale = energy_scale / moment_norm
    deflated = matrices + scale[:, None, None] * ein.contract(
        "fl,fm->flm", moments_, moments_
    )
    # Padded slots carry no flux; an identity metric there keeps them inert.
    metric = jnp.where(pair, deflated, jnp.where(mask[:, :, None], 0.0, identity))
    positive = OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
        },
    )
    solved = eigensolve(
        GeneralizedEigenproblem(
            DenseLinearOperator(gram, properties=self_adjoint),
            DenseLinearOperator(metric, properties=positive),
            problem_id=canonical_fingerprint(
                {"kind": "trace-inverse-pencils", "flux": flux_action_id}
            ),
        ),
        policy=EigenSolvePolicy(DenseEigh(), count=1, which="largest-algebraic"),
    )
    if not bool(jnp.all(solved.successful)):
        raise ValueError(
            "The local trace-inverse eigensolve did not certify every facet; the "
            "flux and energy pencils are not a supported diffusion pair."
        )
    return TraceInverseEvidence(
        owner_id=owner_id,
        flux_action_id=flux_action_id,
        facets=facets,
        side_cells=side_cells,
        constants=np.asarray(solved.eigenvalues[:, 0]),
        relative_residuals=np.asarray(solved.diagnostics.relative_residuals[:, 0]),
        facet_rule_id=facet_rule_id,
        facet_exact_degree=facet_exact_degree,
    )


__all__ = ["certify_trace_inverse", "TraceInverseEvidence"]
