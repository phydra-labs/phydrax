#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Resolved lower-dimensional fracture storage and conservative parent exchange."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
)
from ...nonlinear import (
    AbstractNonlinearMethod,
    implicit_root_result,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)


def _link_values(value, count, name, *, positive=False):
    values = jnp.asarray(value, dtype=float)
    if values.shape not in ((), (count,)):
        raise ValueError(f"{name} must be scalar or one value per exchange link.")
    values = jnp.broadcast_to(values, (count,))
    invalid = ~jnp.isfinite(values) | (values <= 0 if positive else values < 0)
    return eqx.error_if(
        values,
        jnp.any(invalid),
        f"{name} must be finite and {'positive' if positive else 'nonnegative'}.",
    )


class ExchangeStep(StrictModule):
    matrix_concentrations: Array
    fracture_concentrations: Array
    matrix_inventory: Array
    fracture_inventory: Array
    matrix_water_volumes: Array
    fracture_water_volumes: Array
    link_component_rates: Array
    component_balance: Array
    water_balance: Array
    root: NonlinearResult


class FractureMatrixExchange(StrictModule):
    """Mixed-dimensional storage on a declared 2D FV fracture and 3D parent mesh.

    The 2D mesh is in metre-valued LOCAL fracture coordinates; its cell measures
    are true surface areas, not projected areas. ``origin`` and two orthonormal
    ``tangent_axes`` embed this planar fracture into the matrix Cartesian frame.
    Apertures and link distances are metres, contact areas m², porosity and
    saturation dimensionless. A fracture cell stores area*aperture*porosity*S
    cubic metres of water. Matrix water storage must EXCLUDE separately resolved
    fracture void volume: this API does not double-count it or infer it from an
    intersection heuristic.

    Every exchange link explicitly names a matrix cell and a fracture cell.
    Stable parent global IDs are checked against the native parent mesh. Contact
    areas and center-to-interface distances are supplied by the geometric model;
    this class does not claim to construct or intersect an arbitrary DFN.
    Multiple matrix parents may contact one fracture, but its storage is counted
    exactly once. Tangential fracture transport may independently use
    ComponentTransport on ``fracture_discretization`` with these water volumes
    and aperture-weighted, integrated tangential water rates.
    """

    matrix_discretization: UnstructuredFiniteVolumeDiscretization
    fracture_discretization: UnstructuredFiniteVolumeDiscretization
    matrix_cells: Array
    fracture_cells: Array
    parent_global_ids: Array
    aperture: Array
    porosity: Array
    contact_areas: Array
    distances: Array
    origin: Array
    tangent_axes: Array
    fracture_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix_discretization: UnstructuredFiniteVolumeDiscretization,
        fracture_discretization: UnstructuredFiniteVolumeDiscretization,
        matrix_cells: ArrayLike,
        fracture_cells: ArrayLike,
        *,
        parent_global_ids: ArrayLike,
        aperture: ArrayLike,
        porosity: ArrayLike,
        contact_areas: ArrayLike,
        distances: ArrayLike,
        origin: ArrayLike,
        tangent_axes: ArrayLike,
        fracture_id: str,
    ):
        if (
            not isinstance(matrix_discretization, UnstructuredFiniteVolumeDiscretization)
            or matrix_discretization.cell_dimension != 3
            or not isinstance(
                fracture_discretization, UnstructuredFiniteVolumeDiscretization
            )
            or fracture_discretization.cell_dimension != 2
        ):
            raise ValueError(
                "Exchange requires native prepared 3D matrix and 2D local fracture geometry."
            )
        matrix, fracture = np.asarray(matrix_cells), np.asarray(fracture_cells)
        if (
            matrix.ndim != 1
            or matrix.size == 0
            or fracture.shape != matrix.shape
            or not np.issubdtype(matrix.dtype, np.integer)
            or not np.issubdtype(fracture.dtype, np.integer)
        ):
            raise ValueError(
                "Exchange links require matching nonempty integer cell-index vectors."
            )
        if (
            np.any(matrix < 0)
            or np.any(matrix >= matrix_discretization.cell_count)
            or np.any(fracture < 0)
            or np.any(fracture >= fracture_discretization.cell_count)
        ):
            raise ValueError(
                "Exchange link references a cell outside its parent geometry."
            )
        if len(set(zip(matrix.tolist(), fracture.tolist(), strict=True))) != matrix.size:
            raise ValueError(
                "Duplicate parent-fracture links would double-count exchange area."
            )
        ids = jnp.asarray(parent_global_ids)
        if ids.shape != matrix.shape or not jnp.issubdtype(ids.dtype, jnp.integer):
            raise ValueError(
                "parent_global_ids must explicitly identify every matrix parent."
            )
        ids = eqx.error_if(
            ids,
            jnp.any(ids != matrix_discretization.cell_global_ids[matrix]),
            "Exchange parent global IDs disagree with the prepared matrix mesh.",
        )
        if not isinstance(fracture_id, str) or not fracture_id.strip():
            raise ValueError("A stable fracture_id is required.")
        nc, links = fracture_discretization.cell_count, matrix.size
        opening = _link_values(aperture, nc, "aperture", positive=True)
        pores = _link_values(porosity, nc, "fracture porosity", positive=True)
        pores = eqx.error_if(
            pores, jnp.any(pores > 1), "Fracture porosity cannot exceed one."
        )
        origin_, axes = (
            jnp.asarray(origin, dtype=float),
            jnp.asarray(tangent_axes, dtype=float),
        )
        if origin_.shape != (3,) or axes.shape != (2, 3):
            raise ValueError(
                "Planar fracture embedding requires origin (3,) and tangent_axes (2,3)."
            )
        origin_ = eqx.error_if(
            origin_,
            jnp.any(~jnp.isfinite(origin_)),
            "Fracture origin must be finite metres.",
        )
        from ...ein import contract

        axes = eqx.error_if(
            axes,
            jnp.any(~jnp.isfinite(axes))
            | jnp.any(jnp.abs(contract("id,jd->ij", axes, axes) - jnp.eye(2)) > 1e-10),
            "Fracture tangent axes must be orthonormal in the parent Cartesian frame.",
        )
        self.matrix_discretization, self.fracture_discretization = (
            matrix_discretization,
            fracture_discretization,
        )
        self.matrix_cells, self.fracture_cells, self.parent_global_ids = (
            jnp.asarray(matrix, dtype=jnp.int32),
            jnp.asarray(fracture, dtype=jnp.int32),
            ids,
        )
        self.aperture, self.porosity = opening, pores
        self.contact_areas = _link_values(
            contact_areas, links, "contact_areas", positive=True
        )
        self.distances = _link_values(distances, links, "distances", positive=True)
        self.origin, self.tangent_axes, self.fracture_id = origin_, axes, fracture_id

    def fracture_centers(self) -> Array:
        from ...ein import contract

        return self.origin + contract(
            "ci,id->cd", self.fracture_discretization.cell_centers, self.tangent_axes
        )

    def water_volumes(self, saturation: ArrayLike = 1.0) -> Array:
        """Fracture storage in m³; zero saturation is real zero storage."""
        values = _link_values(
            saturation, self.fracture_discretization.cell_count, "fracture saturation"
        )
        values = eqx.error_if(
            values, jnp.any(values > 1), "Fracture saturation cannot exceed one."
        )
        return (
            self.fracture_discretization.cell_volumes
            * self.aperture
            * self.porosity
            * values
        )

    def hydraulic_rates(
        self,
        matrix_pressure: ArrayLike,
        fracture_pressure: ArrayLike,
        *,
        permeability: ArrayLike,
        viscosity: ArrayLike,
        density: ArrayLike = 1000.0,
        gravity: ArrayLike | tuple[float, float, float] = (0.0, 0.0, -9.80665),
    ) -> Array:
        """Integrated m³/s, positive matrix→fracture, with hydrostatic correction.

        ``permeability`` is the effective normal link permeability in m², viscosity
        Pa s and density kg/m³. Pressure work is p_matrix-p_fracture + rho*g·dx
        where dx points matrix→fracture. No cubic-law aperture guess is made for
        normal matrix exchange; callers may supply their qualified interface law.
        """
        pm, pf = jnp.asarray(matrix_pressure), jnp.asarray(fracture_pressure)
        if pm.shape != (self.matrix_discretization.cell_count,) or pf.shape != (
            self.fracture_discretization.cell_count,
        ):
            raise ValueError(
                "Pressures must cover their respective matrix/fracture cells."
            )
        pm = eqx.error_if(
            pm,
            jnp.any(~jnp.isfinite(pm)) | jnp.any(~jnp.isfinite(pf)),
            "Exchange pressures must be finite Pa.",
        )
        count = self.matrix_cells.size
        k = _link_values(permeability, count, "normal permeability")
        mu = _link_values(viscosity, count, "viscosity", positive=True)
        rho = _link_values(density, count, "density", positive=True)
        g = jnp.asarray(gravity, dtype=pm.dtype)
        if g.shape != (3,):
            raise ValueError("gravity must have three Cartesian components in m/s².")
        g = eqx.error_if(g, jnp.any(~jnp.isfinite(g)), "Gravity must be finite.")
        from ...ein import contract

        delta = (
            self.fracture_centers()[self.fracture_cells]
            - self.matrix_discretization.cell_centers[self.matrix_cells]
        )
        driving = (
            pm[self.matrix_cells]
            - pf[self.fracture_cells]
            + rho * contract("ld,d->l", delta, g)
        )
        return k * self.contact_areas / (mu * self.distances) * driving

    def diffusive_conductance(self, diffusivity: ArrayLike) -> Array:
        """Effective normal D (m²/s) × contact area / distance → m³/s."""
        return (
            _link_values(diffusivity, self.matrix_cells.size, "normal diffusivity")
            * self.contact_areas
            / self.distances
        )

    def component_rates(
        self,
        matrix_concentrations: ArrayLike,
        fracture_concentrations: ArrayLike,
        water_rates: ArrayLike,
        conductance: ArrayLike = 0.0,
    ) -> Array:
        """Owner-like matrix→fracture mol/s, one equal/opposite value per link."""
        cm, cf = jnp.asarray(matrix_concentrations), jnp.asarray(fracture_concentrations)
        if (
            cm.ndim != 2
            or cf.shape != (self.fracture_discretization.cell_count, cm.shape[1])
            or cm.shape[0] != self.matrix_discretization.cell_count
        ):
            raise ValueError(
                "Exchange concentrations must be cell-by-component arrays on both domains."
            )
        q = jnp.asarray(water_rates)
        if q.shape != self.matrix_cells.shape:
            raise ValueError(
                "water_rates must contain one integrated m³/s rate per link."
            )
        q = eqx.error_if(
            q, jnp.any(~jnp.isfinite(q)), "Exchange water rates must be finite."
        )
        diffusion = _link_values(
            conductance, self.matrix_cells.size, "diffusive conductance"
        )
        left, right = cm[self.matrix_cells], cf[self.fracture_cells]
        return q[:, None] * jnp.where((q >= 0)[:, None], left, right) + diffusion[
            :, None
        ] * (left - right)

    def exchange_sources(self, link_rates: ArrayLike) -> tuple[Array, Array]:
        """Equal/opposite integrated source arrays; valid for water or components."""
        rates = jnp.asarray(link_rates)
        if rates.shape[0] != self.matrix_cells.size:
            raise ValueError("Link rates must cover each declared exchange link.")
        shape = rates.shape[1:]
        matrix = (
            jnp.zeros((self.matrix_discretization.cell_count,) + shape, dtype=rates.dtype)
            .at[self.matrix_cells]
            .add(-rates)
        )
        fracture = (
            jnp.zeros(
                (self.fracture_discretization.cell_count,) + shape, dtype=rates.dtype
            )
            .at[self.fracture_cells]
            .add(rates)
        )
        return matrix, fracture

    def residual(
        self,
        state,
        previous_matrix_inventory: ArrayLike,
        previous_fracture_inventory: ArrayLike,
        matrix_water_volumes: ArrayLike,
        fracture_water_volumes: ArrayLike,
        water_rates: ArrayLike,
        dt: ArrayLike,
        *,
        conductance: ArrayLike = 0.0,
    ):
        """Coupled integrated-mole equations, directly appendable to global roots."""
        cm, cf = state
        sources = self.exchange_sources(
            self.component_rates(cm, cf, water_rates, conductance)
        )
        return (
            jnp.asarray(matrix_water_volumes)[:, None] * cm
            - jnp.asarray(previous_matrix_inventory)
            - jnp.asarray(dt) * sources[0],
            jnp.asarray(fracture_water_volumes)[:, None] * cf
            - jnp.asarray(previous_fracture_inventory)
            - jnp.asarray(dt) * sources[1],
        )

    def step(
        self,
        previous_matrix_inventory: ArrayLike,
        previous_fracture_inventory: ArrayLike,
        previous_matrix_water_volumes: ArrayLike,
        previous_fracture_water_volumes: ArrayLike,
        water_rates: ArrayLike,
        dt: ArrayLike,
        *,
        conductance: ArrayLike = 0.0,
        nonnegative_components: tuple[bool, ...] | None = None,
        method: AbstractNonlinearMethod | None = None,
        termination: NonlinearTermination | None = None,
    ) -> ExchangeStep:
        """Closed exchange-only BE step, updating water and solute conservatively.

        Water rates are accepted hydraulic inputs over this step. Water volumes
        advance with those SAME equal/opposite rates. A drying/overfilled fracture
        step is rejected, not repaired by clipping. For exchange embedded in a
        full hydraulic solve use ``residual`` with its accepted endpoint volumes.
        """
        oldm, oldf = (
            jnp.asarray(previous_matrix_inventory, dtype=float),
            jnp.asarray(previous_fracture_inventory, dtype=float),
        )
        vm, vf = (
            jnp.asarray(previous_matrix_water_volumes, dtype=float),
            jnp.asarray(previous_fracture_water_volumes, dtype=float),
        )
        q, time = jnp.asarray(water_rates, dtype=float), jnp.asarray(dt, dtype=float)
        if (
            oldm.ndim != 2
            or oldm.shape[0] != self.matrix_discretization.cell_count
            or oldf.shape != (self.fracture_discretization.cell_count, oldm.shape[1])
            or vm.shape != (oldm.shape[0],)
            or vf.shape != (oldf.shape[0],)
            or q.shape != self.matrix_cells.shape
            or time.shape != ()
        ):
            raise ValueError(
                "Exchange inventories, water volumes, rates and timestep have incompatible shapes."
            )
        signs = (
            (True,) * oldm.shape[1]
            if nonnegative_components is None
            else tuple(nonnegative_components)
        )
        if len(signs) != oldm.shape[1] or any(not isinstance(v, bool) for v in signs):
            raise ValueError("nonnegative_components must declare every component.")
        positive = jnp.asarray(signs)
        oldm = eqx.error_if(
            oldm,
            jnp.any(~jnp.isfinite(oldm))
            | jnp.any(~jnp.isfinite(oldf))
            | jnp.any((oldm < 0) & positive)
            | jnp.any((oldf < 0) & positive),
            "Exchange inventories violate their component domain.",
        )
        time = eqx.error_if(
            time,
            ~jnp.isfinite(time) | (time <= 0),
            "Exchange dt must be positive finite seconds.",
        )
        vm = eqx.error_if(
            vm,
            jnp.any(~jnp.isfinite(vm) | (vm <= 0)),
            "Matrix water volumes must be positive finite m³.",
        )
        vf = eqx.error_if(
            vf,
            jnp.any(~jnp.isfinite(vf) | (vf <= 0) | (vf > self.water_volumes())),
            "Fracture water volumes must be positive and no larger than aperture-resolved pore capacity.",
        )
        q = eqx.error_if(
            q,
            jnp.any(~jnp.isfinite(q)),
            "Exchange water rates must be finite integrated m³/s.",
        )
        water_sources = self.exchange_sources(q)
        newm, newf = vm + time * water_sources[0], vf + time * water_sources[1]
        newm = eqx.error_if(
            newm,
            jnp.any(newm <= 0) | jnp.any(~jnp.isfinite(newm)),
            "Accepted exchange rates would dry matrix water storage.",
        )
        newf = eqx.error_if(
            newf,
            jnp.any(newf <= 0)
            | jnp.any(newf > self.water_volumes())
            | jnp.any(~jnp.isfinite(newf)),
            "Accepted exchange rates would dry or overfill aperture-resolved fracture storage.",
        )
        initial = (oldm / vm[:, None], oldf / vf[:, None])
        scale = jnp.maximum(
            jnp.maximum(jnp.max(jnp.abs(oldm), axis=0), jnp.max(jnp.abs(oldf), axis=0)),
            jnp.maximum(jnp.max(vm), jnp.max(vf)),
        )
        arguments = (oldm, oldf, newm, newf, q, time)

        def scaled(state, args):
            rm, rf = self.residual(state, *args, conductance=conductance)
            return rm / scale, rf / scale

        def valid(state, residual, auxiliary, args):
            return jnp.all(~positive | (state[0] >= 0)) & jnp.all(
                ~positive | (state[1] >= 0)
            )

        root = implicit_root_result(
            NonlinearSystemProblem(
                scaled,
                validity=valid,
                problem_id="mixed-dimensional-conservative-fracture-exchange",
            ),
            initial,
            args=arguments,
            method=method,
            termination=termination,
        )
        cm, cf = root.state
        cm = eqx.error_if(
            cm,
            ~root.successful | ~valid(root.state, None, None, None),
            "Fracture exchange requires a successful physical root; no inventory clipping is admitted.",
        )
        im, iff = newm[:, None] * cm, newf[:, None] * cf
        balance = jnp.sum(im - oldm, axis=0) + jnp.sum(iff - oldf, axis=0)
        water_balance = jnp.sum(newm - vm) + jnp.sum(newf - vf)
        return ExchangeStep(
            cm,
            cf,
            im,
            iff,
            newm,
            newf,
            self.component_rates(cm, cf, q, conductance),
            balance,
            water_balance,
            root,
        )


__all__ = ["ExchangeStep", "FractureMatrixExchange"]
