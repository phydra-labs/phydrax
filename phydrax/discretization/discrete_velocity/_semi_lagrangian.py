#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace
from .._transfer import FieldTransfer
from ._quadrature import CertifiedDiscreteVelocityQuadrature


class SemiLagrangianTransferRequirements(StrictModule, NonTrainableState):
    """Capabilities required from every prepared departure-point transfer."""

    constant_preserving: bool = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    positivity_preserving: bool = eqx.field(static=True)
    differentiable_geometry: bool = eqx.field(static=True)
    exact_on: tuple[str, ...] = eqx.field(static=True)
    requirement_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        constant_preserving: bool = True,
        conservative: bool = True,
        positivity_preserving: bool = True,
        differentiable_geometry: bool = False,
        exact_on: Sequence[str] = (),
    ):
        exact = tuple(str(value) for value in exact_on)
        if any(not value for value in exact) or len(set(exact)) != len(exact):
            raise ValueError("exact_on requirements must be unique non-empty strings.")
        self.constant_preserving = bool(constant_preserving)
        self.conservative = bool(conservative)
        self.positivity_preserving = bool(positivity_preserving)
        self.differentiable_geometry = bool(differentiable_geometry)
        self.exact_on = exact
        self.requirement_id = canonical_fingerprint(
            {
                "kind": "semi-lagrangian-transfer-requirements-v1",
                "constant_preserving": bool(constant_preserving),
                "conservative": bool(conservative),
                "positivity_preserving": bool(positivity_preserving),
                "differentiable_geometry": bool(differentiable_geometry),
                "exact_on": list(exact),
            }
        )

    def validate(self, transfer: FieldTransfer, /) -> None:
        if not isinstance(transfer, FieldTransfer):
            raise TypeError(
                "Semi-Lagrangian population transfers must be FieldTransfer values."
            )
        properties = transfer.properties
        missing = []
        for name in (
            "constant_preserving",
            "conservative",
            "positivity_preserving",
            "differentiable_geometry",
        ):
            if getattr(self, name) and not getattr(properties, name):
                missing.append(name)
        unavailable_exactness = tuple(
            name for name in self.exact_on if name not in properties.exact_on
        )
        if missing or unavailable_exactness:
            details = []
            if missing:
                details.append("properties=" + ",".join(missing))
            if unavailable_exactness:
                details.append("exact_on=" + ",".join(unavailable_exactness))
            raise ValueError(
                "FieldTransfer does not satisfy semi-Lagrangian requirements: "
                + "; ".join(details)
            )


class SemiLagrangianTransportEvidence(StrictModule):
    """Conserved-moment balance across one prepared off-lattice transfer."""

    source_population_integrals: Array
    target_population_integrals: Array
    source_moments: Array
    target_moments: Array
    conservation_residual: Array
    maximum_absolute_residual: Array


class PreparedOffLatticeSemiLagrangianDVM(StrictModule, NonTrainableState):
    """Fixed-step, off-lattice departure transfer for trailing-Q populations.

    Each velocity owns a prepared ``FieldTransfer``. No interpolation is built at
    execution time, and capability claims are checked before the plan can exist.
    """

    quadrature: CertifiedDiscreteVelocityQuadrature
    population_transfers: tuple[FieldTransfer, ...]
    requirements: SemiLagrangianTransferRequirements
    time_step: float = eqx.field(static=True)
    source_shape: tuple[int, ...] = eqx.field(static=True)
    target_shape: tuple[int, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        population_transfers: Sequence[FieldTransfer],
        time_step: float,
        /,
        *,
        requirements: SemiLagrangianTransferRequirements | None = None,
    ):
        if not isinstance(quadrature, CertifiedDiscreteVelocityQuadrature):
            raise TypeError("quadrature must be a CertifiedDiscreteVelocityQuadrature.")
        if quadrature.transport_kind != "off_lattice":
            raise ValueError(
                "PreparedOffLatticeSemiLagrangianDVM requires an explicitly off-lattice quadrature."
            )
        transfers = tuple(population_transfers)
        if len(transfers) != quadrature.population_count:
            raise ValueError(
                "One prepared FieldTransfer is required per discrete velocity."
            )
        requirements_ = (
            SemiLagrangianTransferRequirements() if requirements is None else requirements
        )
        if not isinstance(requirements_, SemiLagrangianTransferRequirements):
            raise TypeError("requirements must be SemiLagrangianTransferRequirements.")
        step = float(time_step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("Semi-Lagrangian time_step must be finite and positive.")
        for transfer in transfers:
            requirements_.validate(transfer)
        source = transfers[0].source
        target = transfers[0].target
        if not isinstance(source.vector_space, ArraySpace) or not isinstance(
            target.vector_space, ArraySpace
        ):
            raise TypeError(
                "Semi-Lagrangian transfers require array-valued field spaces."
            )
        for transfer in transfers[1:]:
            if not transfer.source.vector_space.compatible(
                source.vector_space
            ) or not transfer.target.vector_space.compatible(target.vector_space):
                raise ValueError(
                    "Every population transfer must share exact source and target spaces."
                )
        self.quadrature = quadrature
        self.population_transfers = transfers
        self.requirements = requirements_
        self.time_step = step
        self.source_shape = source.vector_space.shape
        self.target_shape = target.vector_space.shape
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-off-lattice-semi-lagrangian-dvm-v1",
                "quadrature": quadrature.quadrature_id,
                "transfers": [transfer.transfer_id for transfer in transfers],
                "requirements": requirements_.requirement_id,
                "time_step": step,
            }
        )

    def transport(self, populations: ArrayLike, /) -> Array:
        values = self.quadrature.validate_populations(populations)
        if values.shape != self.source_shape + (self.quadrature.population_count,):
            raise ValueError(
                "Semi-Lagrangian source populations must have shape "
                f"{self.source_shape + (self.quadrature.population_count,)}."
            )
        return jnp.stack(
            tuple(
                transfer.operator.mv(values[..., population])
                for population, transfer in enumerate(self.population_transfers)
            ),
            axis=-1,
        )

    @staticmethod
    def _population_integrals(space: ArraySpace, populations: Array, /) -> Array:
        ones = jnp.ones(space.shape, dtype=space.dtype)
        return jnp.stack(
            tuple(
                space.inner(ones, populations[..., index])
                for index in range(populations.shape[-1])
            )
        )

    def transport_with_evidence(
        self, populations: ArrayLike, /
    ) -> tuple[Array, SemiLagrangianTransportEvidence]:
        values = self.quadrature.validate_populations(populations)
        transported = self.transport(values)
        source_space = self.population_transfers[0].source.vector_space
        target_space = self.population_transfers[0].target.vector_space
        if not isinstance(source_space, ArraySpace) or not isinstance(
            target_space, ArraySpace
        ):
            raise TypeError("Semi-Lagrangian evidence requires array field spaces.")
        source_integrals = self._population_integrals(source_space, values)
        target_integrals = self._population_integrals(target_space, transported)
        moment_matrix = self.quadrature.hydrodynamic_moment_matrix()
        source_moments = oe.contract("mq,q->m", moment_matrix, source_integrals)
        target_moments = oe.contract("mq,q->m", moment_matrix, target_integrals)
        residual = target_moments - source_moments
        return transported, SemiLagrangianTransportEvidence(
            source_population_integrals=source_integrals,
            target_population_integrals=target_integrals,
            source_moments=source_moments,
            target_moments=target_moments,
            conservation_residual=residual,
            maximum_absolute_residual=jnp.max(jnp.abs(residual)),
        )


__all__ = [
    "PreparedOffLatticeSemiLagrangianDVM",
    "SemiLagrangianTransferRequirements",
    "SemiLagrangianTransportEvidence",
]
