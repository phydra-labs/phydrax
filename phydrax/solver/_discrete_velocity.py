#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.discrete_velocity._quadrature import (
    CertifiedDiscreteVelocityQuadrature,
)
from ..discretization.finite_volume._boundary import FiniteVolumeBoundarySet
from ..discretization.finite_volume._dynamics import (
    FiniteVolumeMethodPlan,
    FiniteVolumeResidualDiagnostics,
    PreparedFiniteVolumeDynamics,
)
from ..discretization.finite_volume._mapped import MappedFiniteVolumeDiscretization
from ..discretization.finite_volume._precision import FiniteVolumePrecisionPolicy
from ..discretization.finite_volume._structured import FiniteVolumeDiscretization
from ..discretization.lattice_boltzmann._program import (
    finite_volume_dvm_manifest,
    KineticProgramManifest,
)
from ..equations._discrete_velocity import (
    AbstractConservativeDVMSource,
    DiscreteVelocityAdvectionSystem,
)


class FiniteVolumeDVMResidualEvidence(StrictModule):
    """Population and declared-moment conservation evidence for one residual."""

    finite_volume: FiniteVolumeResidualDiagnostics
    population_conservation_defect: Array
    declared_moment_conservation_defect: Array
    maximum_absolute_declared_moment_defect: Array


class PreparedConservativeFiniteVolumeDVM(StrictModule, NonTrainableState):
    """Prepared finite-volume DVM that delegates transport to the FV substrate."""

    quadrature: CertifiedDiscreteVelocityQuadrature
    system: DiscreteVelocityAdvectionSystem
    dynamics: PreparedFiniteVolumeDynamics
    source: AbstractConservativeDVMSource | None
    declared_moment_matrix: Array
    program_manifest: KineticProgramManifest
    declared_moment_names: tuple[str, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        system: DiscreteVelocityAdvectionSystem,
        dynamics: PreparedFiniteVolumeDynamics,
        source: AbstractConservativeDVMSource | None,
        declared_moment_matrix: Array,
        program_manifest: KineticProgramManifest,
        declared_moment_names: tuple[str, ...],
        prepared_id: str,
        /,
    ):
        self.quadrature = quadrature
        self.system = system
        self.dynamics = dynamics
        self.source = source
        self.program_manifest = program_manifest
        self.declared_moment_matrix = declared_moment_matrix
        self.declared_moment_names = declared_moment_names
        self.prepared_id = prepared_id

    def __call__(self, time: Array, state: Array, args: Any = None, /) -> Array:
        return self.dynamics(time, state, args)

    def stable_step(
        self,
        state: Array,
        args: Any = None,
        /,
        *,
        cfl: float = 0.45,
    ) -> Array:
        return self.dynamics.stable_step(state, args, cfl=cfl)

    def residual_with_evidence(
        self,
        time: Array,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> tuple[Array, FiniteVolumeDVMResidualEvidence]:
        values = self.quadrature.validate_populations(state)
        residual, diagnostics = self.dynamics.residual_with_diagnostics(
            time, values, args
        )
        population_defect = diagnostics.conservation_defect
        moment_defect = ein.contract(
            "mq,q->m", self.declared_moment_matrix, population_defect
        )
        return residual, FiniteVolumeDVMResidualEvidence(
            finite_volume=diagnostics,
            population_conservation_defect=population_defect,
            declared_moment_conservation_defect=moment_defect,
            maximum_absolute_declared_moment_defect=jnp.max(jnp.abs(moment_defect)),
        )


class ConservativeFiniteVolumeDVMPlan(StrictModule, NonTrainableState):
    """Conservative FV-DVM composition over prepared finite-volume abstractions."""

    quadrature: CertifiedDiscreteVelocityQuadrature
    discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization
    method: FiniteVolumeMethodPlan
    boundaries: FiniteVolumeBoundarySet
    source: AbstractConservativeDVMSource | None
    precision: FiniteVolumePrecisionPolicy | None
    population_floor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        method: FiniteVolumeMethodPlan,
        boundaries: FiniteVolumeBoundarySet,
        /,
        *,
        source: AbstractConservativeDVMSource | None = None,
        precision: FiniteVolumePrecisionPolicy | None = None,
        population_floor: float = 0.0,
    ):
        if not isinstance(quadrature, CertifiedDiscreteVelocityQuadrature):
            raise TypeError("quadrature must be a CertifiedDiscreteVelocityQuadrature.")
        if not isinstance(
            discretization,
            (FiniteVolumeDiscretization, MappedFiniteVolumeDiscretization),
        ):
            raise TypeError(
                "discretization must be prepared structured finite-volume geometry."
            )
        if not isinstance(method, FiniteVolumeMethodPlan):
            raise TypeError("method must be FiniteVolumeMethodPlan.")
        if not isinstance(boundaries, FiniteVolumeBoundarySet):
            raise TypeError("boundaries must be FiniteVolumeBoundarySet.")
        if source is not None and not isinstance(source, AbstractConservativeDVMSource):
            raise TypeError("source must be a conservative DVM source or None.")
        if (
            source is not None
            and source.quadrature.quadrature_id != quadrature.quadrature_id
        ):
            raise ValueError(
                "FV-DVM source and transport quadratures must match exactly."
            )
        if precision is not None and not isinstance(
            precision, FiniteVolumePrecisionPolicy
        ):
            raise TypeError("precision must be FiniteVolumePrecisionPolicy or None.")
        floor = float(population_floor)
        if not np.isfinite(floor) or floor < 0.0:
            raise ValueError("population_floor must be finite and non-negative.")
        if discretization.component_count != quadrature.population_count:
            raise ValueError("FV-DVM discretization must have exactly Q components.")
        if len(discretization.cell_shape) != quadrature.dimension:
            raise ValueError("FV-DVM spatial dimension must equal quadrature dimension.")
        if method.viscous is not None:
            raise ValueError(
                "DVM viscosity belongs in collision/source composition, not FV viscous fluxes."
            )
        self.quadrature = quadrature
        self.discretization = discretization
        self.method = method
        self.boundaries = boundaries
        self.source = source
        self.precision = precision
        self.population_floor = floor
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conservative-finite-volume-dvm-plan-v1",
                "quadrature": quadrature.quadrature_id,
                "discretization": discretization.prepared_id,
                "method": method.method_id,
                "boundaries": boundaries.boundary_set_id,
                "source": None if source is None else source.source_id,
                "precision": None if precision is None else precision.policy_id,
                "population_floor": floor,
            }
        )

    def prepare(self, /) -> PreparedConservativeFiniteVolumeDVM:
        system = DiscreteVelocityAdvectionSystem(
            self.quadrature, population_floor=self.population_floor
        )
        if tuple(self.discretization.component_names) != system.component_names:
            raise ValueError(
                "FV-DVM discretization component names must equal the system population names."
            )
        if self.source is None:
            moment_matrix = self.quadrature.hydrodynamic_moment_matrix()
            names = (
                "mass",
                *(f"momentum_{axis}" for axis in range(self.quadrature.dimension)),
                "kinetic_energy",
            )
        else:
            moment_matrix = self.source.moment_matrix
            names = self.source.moment_names
        if self.source is None:
            source_function = None
        else:
            source_plan = self.source

            def evaluate_source(time, state, coordinates, args):
                return source_plan(time, state, coordinates, args)

            source_function = evaluate_source

        dynamics = PreparedFiniteVolumeDynamics(
            system,
            self.discretization,
            self.method,
            self.boundaries,
            source=source_function,
            source_id=None if self.source is None else self.source.source_id,
            precision=self.precision,
        )
        program_manifest = finite_volume_dvm_manifest(
            self.quadrature.quadrature_id,
            dynamics.precision.policy_id,
            self.quadrature.population_count,
            tuple(names),
            has_source=self.source is not None,
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-conservative-finite-volume-dvm-v1",
                "plan": self.plan_id,
                "system": system.system_id,
                "dynamics": dynamics.dynamics_id,
                "declared_moments": list(names),
                "program_manifest": program_manifest.manifest_id,
            }
        )
        return PreparedConservativeFiniteVolumeDVM(
            self.quadrature,
            system,
            dynamics,
            self.source,
            moment_matrix,
            program_manifest,
            tuple(names),
            prepared_id,
        )


__all__ = [
    "ConservativeFiniteVolumeDVMPlan",
    "FiniteVolumeDVMResidualEvidence",
    "PreparedConservativeFiniteVolumeDVM",
]
