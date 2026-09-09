#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative backward-Euler Richards flow with global hybrid face pressures."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ...discretization.finite_volume._hybrid_diffusion import (
    _tensor,
    HybridMimeticDiffusion,
)
from ...discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
)
from ...ein import contract
from ...nonlinear import (
    AbstractNonlinearMethod,
    implicit_root_result,
    ImplicitRootDerivativePolicy,
    NewtonKrylov,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._boundaries import PorousBoundaryConditions
from ._materials import _finite, PorousMaterial
from ._retention import VanGenuchtenMualem
from ._state import _successful_root_value, PorousFluxes, PorousState, PorousStepResult


def _cell_array(value, count, name):
    value = jnp.asarray(value)
    value = value.astype(jnp.result_type(value, 1.0))
    if value.shape not in ((), (count,)):
        raise ValueError(f"{name} must be scalar or have shape ({count},).")
    return jnp.broadcast_to(value, (count,))


def _finish_root(root, successful):
    status = jnp.where(
        root.successful & ~successful,
        int(NonlinearStatus.RECOVERABLE_DOMAIN_FAILURE),
        root.status,
    )
    return eqx.tree_at(
        lambda result: (result.state, result.residual, result.status),
        root,
        (
            _successful_root_value(root.state, successful),
            _successful_root_value(root.residual, successful),
            status,
        ),
    )


class RichardsPlan(StrictModule):
    """Single liquid, stationary porous skeleton, fixed three-dimensional SI mesh.

    The conserved inventory is V*phi(p)*S(p)*rho(p,T). Darcy's volume rate uses
    the full intrinsic tensor and pressure drive grad(p)-rho*g. Face density is
    selected by liquid flow direction, and *mass* continuity is enforced across
    the single shared face pressure. No saturated storage or mobility floors are
    added. Gas pressure is a fixed zero gauge reference; vapor/freezing are absent.

    Public residual rows are [cell kg/s, face kg/s except Dirichlet rows in Pa].
    ``step`` nondimensionalizes these rows only for the native nonlinear solver.
    """

    discretization: UnstructuredFiniteVolumeDiscretization
    diffusion: HybridMimeticDiffusion
    material: PorousMaterial
    retention: VanGenuchtenMualem
    boundaries: PorousBoundaryConditions
    gravity_m_s2: Array
    temperature_K: Array
    intrinsic_tensor: Array
    intrinsic_matrices: Array
    method: AbstractNonlinearMethod
    termination: NonlinearTermination
    derivative_policy: ImplicitRootDerivativePolicy | None
    pressure_scale_Pa: float = eqx.field(static=True)
    mass_rate_scale_kg_s: float = eqx.field(static=True)

    def __init__(
        self,
        discretization,
        material,
        retention,
        boundaries,
        /,
        *,
        gravity_m_s2=(0.0, 0.0, -9.80665),
        temperature_K=293.15,
        stabilization=1.0,
        method=None,
        termination=None,
        derivative_policy=None,
        pressure_scale_Pa=1.0e4,
        mass_rate_scale_kg_s=1.0,
    ):
        if not isinstance(material, PorousMaterial) or not isinstance(
            retention, VanGenuchtenMualem
        ):
            raise TypeError("Richards requires PorousMaterial and VanGenuchtenMualem.")
        if not isinstance(boundaries, PorousBoundaryConditions):
            raise TypeError("Richards requires unit-qualified PorousBoundaryConditions.")
        if boundaries.geometry_id != discretization.geometry_id:
            raise ValueError("Richards boundary geometry must match the discretization.")
        diffusion = HybridMimeticDiffusion(discretization, stabilization=stabilization)
        gravity = _finite(gravity_m_s2, "gravity")
        if gravity.shape != (3,):
            raise ValueError(
                "gravity_m_s2 must be a three-vector in numerical Cartesian axes."
            )
        self.discretization, self.diffusion = discretization, diffusion
        self.material, self.retention, self.boundaries = material, retention, boundaries
        self.gravity_m_s2 = gravity
        self.temperature_K = _cell_array(
            _finite(temperature_K, "temperature", positive=True),
            diffusion.cell_count,
            "temperature_K",
        )
        self.intrinsic_tensor = _tensor(material.permeability_m2, diffusion.cell_count)
        self.intrinsic_matrices = diffusion.local_matrices(self.intrinsic_tensor)
        self.method = NewtonKrylov() if method is None else method
        self.termination = NonlinearTermination() if termination is None else termination
        self.derivative_policy = derivative_policy
        for value in (pressure_scale_Pa, mass_rate_scale_kg_s):
            if not float(value) > 0 or not jnp.isfinite(value):
                raise ValueError(
                    "Nonlinear physical reference scales must be finite and positive."
                )
        self.pressure_scale_Pa = float(pressure_scale_Pa)
        self.mass_rate_scale_kg_s = float(mass_rate_scale_kg_s)
        material_fields = (
            (material.porosity, "porosity"),
            (material.density_kg_m3, "density_kg_m3"),
            (material.viscosity_Pa_s, "viscosity_Pa_s"),
            (
                material.fluid_compressibility_Pa_inverse,
                "fluid_compressibility_Pa_inverse",
            ),
            (material.pore_compressibility_Pa_inverse, "pore_compressibility_Pa_inverse"),
            (material.thermal_expansion_K_inverse, "thermal_expansion_K_inverse"),
            (material.viscosity_temperature_K_inverse, "viscosity_temperature_K_inverse"),
            (material.reference_pressure_Pa, "reference_pressure_Pa"),
            (material.reference_temperature_K, "reference_temperature_K"),
        )
        retention_fields = (
            (retention.alpha_Pa_inverse, "alpha_Pa_inverse"),
            (retention.n, "n"),
            (retention.residual_saturation, "residual_saturation"),
            (retention.pore_connectivity, "pore_connectivity"),
        )
        for value, name in (*material_fields, *retention_fields):
            _cell_array(value, diffusion.cell_count, name)

    def _temperature(self, temperature_K=None):
        return (
            self.temperature_K
            if temperature_K is None
            else _cell_array(temperature_K, self.diffusion.cell_count, "temperature_K")
        )

    def _face_temperature(self, temperature, face_temperature_K=None):
        if face_temperature_K is not None:
            return _cell_array(
                face_temperature_K, self.diffusion.face_count, "face_temperature_K"
            )
        owner, neighbour = (
            self.discretization.owner_cells,
            self.discretization.neighbour_cells,
        )
        return jnp.where(
            neighbour >= 0,
            (temperature[owner] + temperature[jnp.maximum(neighbour, 0)]) / 2,
            temperature[owner],
        )

    def _face_density(self, face_pressure, face_temperature):
        owner = self.discretization.owner_cells
        reference_density = _cell_array(
            self.material.density_kg_m3, self.diffusion.cell_count, "density_kg_m3"
        )[owner]
        fluid_compressibility = _cell_array(
            self.material.fluid_compressibility_Pa_inverse,
            self.diffusion.cell_count,
            "fluid_compressibility_Pa_inverse",
        )[owner]
        reference_pressure = _cell_array(
            self.material.reference_pressure_Pa,
            self.diffusion.cell_count,
            "reference_pressure_Pa",
        )[owner]
        thermal_expansion = _cell_array(
            self.material.thermal_expansion_K_inverse,
            self.diffusion.cell_count,
            "thermal_expansion_K_inverse",
        )[owner]
        reference_temperature = _cell_array(
            self.material.reference_temperature_K,
            self.diffusion.cell_count,
            "reference_temperature_K",
        )[owner]
        return reference_density * jnp.exp(
            fluid_compressibility * (face_pressure - reference_pressure)
            - thermal_expansion * (face_temperature - reference_temperature)
        )

    def water_volume(self, pressure_Pa):
        return (
            self.discretization.cell_volumes
            * self.material.pore_fraction(pressure_Pa)
            * self.retention.saturation(pressure_Pa)
        )

    def water_mass(self, pressure_Pa, temperature_K=None):
        return self.water_volume(pressure_Pa) * self.material.density(
            pressure_Pa, self._temperature(temperature_K)
        )

    def fluxes(
        self, pressure_Pa, face_pressure_Pa, temperature_K=None, face_temperature_K=None
    ):
        pressure = _cell_array(pressure_Pa, self.diffusion.cell_count, "pressure_Pa")
        face_pressure = _cell_array(
            face_pressure_Pa, self.diffusion.face_count, "face_pressure_Pa"
        )
        temperature = self._temperature(temperature_K)
        face_temperature = self._face_temperature(temperature, face_temperature_K)
        density = self.material.density(pressure, temperature)
        mobility = self.retention.relative_permeability(
            pressure
        ) / self.material.viscosity(temperature)
        difference = face_pressure[self.diffusion.cell_faces] - pressure[:, None]
        local = -contract("cfg,cg->cf", self.intrinsic_matrices, difference)
        local = local + contract(
            "cfi,cij,cj->cf",
            self.diffusion.outward_areas,
            self.intrinsic_tensor,
            density[:, None] * self.gravity_m_s2,
        )
        local = jnp.where(self.diffusion.valid, local * mobility[:, None], 0.0)
        face = self.diffusion.cell_faces
        owner, neighbour = (
            self.discretization.owner_cells,
            self.discretization.neighbour_cells,
        )
        cells = jnp.arange(self.diffusion.cell_count)[:, None]
        opposite = jnp.where(cells == owner[face], neighbour[face], owner[face])
        boundary_density = self._face_density(face_pressure, face_temperature)
        incoming_density = jnp.where(
            opposite >= 0, density[jnp.maximum(opposite, 0)], boundary_density[face]
        )
        local_mass = local * jnp.where(local >= 0, density[:, None], incoming_density)
        return PorousFluxes(
            self.diffusion.owner_rates(local),
            self.diffusion.owner_rates(local_mass),
            local,
            local_mass,
        )

    def well_posed(
        self, pressure_Pa, temperature_K=None, *, additional_anchored_cells=None
    ):
        """Every connected component needs pressure anchoring or positive storage.

        ``additional_anchored_cells`` lets a coupled surface supply its own pressure
        reference when it replaces exterior face equations.
        """
        temperature = self._temperature(temperature_K)
        _, capacity = jax.jvp(
            lambda pressure: self.water_mass(pressure, temperature),
            (jnp.asarray(pressure_Pa),),
            (jnp.ones_like(pressure_Pa),),
        )
        storage = capacity > 0
        if additional_anchored_cells is not None:
            storage = storage | jnp.asarray(additional_anchored_cells, dtype=bool)
        stored = (
            jnp.zeros(self.diffusion.component_count, dtype=jnp.int32)
            .at[self.diffusion.component_ids]
            .max(storage.astype(jnp.int32))
            > 0
        )
        return jnp.all(stored | self.diffusion.anchored_components(self.boundaries))

    def admissible(self, pressure_Pa, temperature_K=None):
        temperature = self._temperature(temperature_K)
        mobility = self.retention.relative_permeability(
            pressure_Pa
        ) / self.material.viscosity(temperature)
        return self.material.admissible(pressure_Pa, temperature) & jnp.all(
            jnp.isfinite(mobility) & (mobility > 0)
        )

    def state_from_unknown(
        self, unknown, *, time_s=0.0, temperature_K=None, face_temperature_K=None
    ):
        """Construct inventories without altering any solved face pressure."""
        count = self.diffusion.cell_count
        unknown = jnp.asarray(unknown)
        if unknown.shape != (count + self.diffusion.face_count,):
            raise ValueError(
                "Richards unknown must concatenate all cell and global face pressures."
            )
        temperature = self._temperature(temperature_K)
        face_temperature = self._face_temperature(temperature, face_temperature_K)
        pressure, face_pressure = unknown[:count], unknown[count:]
        volume = self.water_volume(pressure)
        return PorousState(
            pressure,
            face_pressure,
            temperature,
            face_temperature,
            volume * self.material.density(pressure, temperature),
            volume,
            jnp.asarray(time_s, dtype=pressure.dtype),
        )

    def initialize(
        self,
        pressure_Pa,
        face_pressure_Pa=None,
        *,
        time_s=0.0,
        temperature_K=None,
        face_temperature_K=None,
    ):
        pressure = _cell_array(pressure_Pa, self.diffusion.cell_count, "pressure_Pa")
        if face_pressure_Pa is None:
            owner, neighbour = (
                self.discretization.owner_cells,
                self.discretization.neighbour_cells,
            )
            face_pressure = jnp.where(
                neighbour >= 0,
                (pressure[owner] + pressure[jnp.maximum(neighbour, 0)]) / 2,
                pressure[owner],
            )
        else:
            face_pressure = _cell_array(
                face_pressure_Pa, self.diffusion.face_count, "face_pressure_Pa"
            )
        pressure = eqx.error_if(
            pressure,
            ~self.admissible(pressure, temperature_K),
            "Initial porous state is outside the constitutive domain.",
        )
        return self.state_from_unknown(
            jnp.concatenate((pressure, self.boundaries.impose_dirichlet(face_pressure))),
            time_s=time_s,
            temperature_K=temperature_K,
            face_temperature_K=face_temperature_K,
        )

    def residual(
        self,
        unknown,
        previous: PorousState,
        dt_s,
        source_kg_s=0.0,
        *,
        temperature_K=None,
        face_temperature_K=None,
    ):
        count = self.diffusion.cell_count
        pressure, face_pressure = unknown[:count], unknown[count:]
        temperature = (
            previous.temperature_K
            if temperature_K is None
            else self._temperature(temperature_K)
        )
        face_temperature = (
            previous.face_temperature_K
            if face_temperature_K is None
            else face_temperature_K
        )
        flux = self.fluxes(pressure, face_pressure, temperature, face_temperature)
        mass = self.water_mass(pressure, temperature)
        cells = (
            (mass - previous.water_mass_kg) / dt_s
            + jnp.sum(flux.local_mass_rates, axis=1)
            - source_kg_s
        )
        faces = self.boundaries.face_residual(
            face_pressure, self.diffusion.continuity_residual(flux.local_mass_rates)
        )
        return jnp.concatenate((cells, faces))

    def residual_scales(self):
        faces = jnp.where(
            self.boundaries.kind == 1, self.pressure_scale_Pa, self.mass_rate_scale_kg_s
        )
        return jnp.concatenate(
            (jnp.full(self.diffusion.cell_count, self.mass_rate_scale_kg_s), faces)
        )

    def step(self, previous: PorousState, dt_s, *, source_kg_s=0.0, initial_unknown=None):
        dt = _finite(dt_s, "time step", positive=True)
        if dt.shape != ():
            raise ValueError("dt_s must be scalar.")
        source = _cell_array(
            _finite(source_kg_s, "mass source"), self.diffusion.cell_count, "source_kg_s"
        )
        initial = (
            previous.unknown if initial_unknown is None else jnp.asarray(initial_unknown)
        )
        scale = self.residual_scales()

        def residual(scaled, args):
            return (
                self.residual(scaled * self.pressure_scale_Pa, previous, dt, source)
                / scale
            )

        def valid(scaled, residual_value, auxiliary, args):
            pressure = scaled[: self.diffusion.cell_count] * self.pressure_scale_Pa
            return self.admissible(pressure, previous.temperature_K) & self.well_posed(
                pressure, previous.temperature_K
            )

        root = implicit_root_result(
            NonlinearSystemProblem(
                residual, validity=valid, problem_id="richards-hybrid-backward-euler"
            ),
            initial / self.pressure_scale_Pa,
            method=self.method,
            termination=self.termination,
            derivative_policy=self.derivative_policy,
        )
        successful = root.successful & valid(root.state, root.residual, None, None)
        root = _finish_root(root, successful)
        unknown = root.state * self.pressure_scale_Pa
        candidate = self.state_from_unknown(
            unknown,
            time_s=previous.time_s + dt,
            temperature_K=previous.temperature_K,
            face_temperature_K=previous.face_temperature_K,
        )
        state = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, previous
        )
        flux = self.fluxes(
            candidate.pressure_Pa,
            candidate.face_pressure_Pa,
            candidate.temperature_K,
            candidate.face_temperature_K,
        )
        return PorousStepResult(
            state,
            candidate,
            flux,
            root,
            self.residual(unknown, previous, dt, source),
            successful,
        )


__all__ = ["RichardsPlan"]
