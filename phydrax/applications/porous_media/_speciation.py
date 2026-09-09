#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Local aqueous mass action in an explicitly declared component basis."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...ein import contract
from ...linalg import DenseLU, LinearSolvePolicy
from ...nonlinear import (
    AbstractNonlinearMethod,
    implicit_root_result,
    NewtonKrylov,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)


def _chemical_method(method):
    return (
        NewtonKrylov(linear_policy=LinearSolvePolicy(DenseLU()))
        if method is None
        else method
    )


def _chemical_termination(termination):
    return (
        NonlinearTermination(
            absolute_residual=1e-10,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=100,
        )
        if termination is None
        else termination
    )


class SpeciationResult(StrictModule):
    """Concentrations in mol/m³, dimensionless activities and balance evidence.

    ``component_residual`` includes the replaced component when charge balance is
    used; that component is open, NOT a silently conserved analytical total.
    Accessing ``concentrations`` requires a successful native root. Derivatives
    are native implicit derivatives and require a nonsingular root Jacobian.
    """

    concentrations: Array
    activities: Array
    component_totals: Array
    component_residual: Array
    charge_residual: Array
    mass_action_residual: Array
    root: NonlinearResult


class MassActionSystem(StrictModule):
    """Primary/secondary formation convention ``a_secondary = K ∏a_primary**ν``.

    Primary species form the component basis (identity primary composition).
    Secondary composition is ``stoichiometry[secondary, primary]``; signed
    coefficients admit proton/alkalinity bases. All species concentrations are
    strictly positive. Zero analytical totals requiring absent species must use
    a reduced chemical basis, not a numerical concentration floor.

    Concentrations and fixed charge are mol/m³. Activities use concentration /
    ``reference_concentration`` (default 1000 mol/m³ = 1 mol/L), so ``log_k`` is
    the NATURAL logarithm of dimensionless formation constants on that standard
    state. Charges are signed elementary-charge equivalents per mole.

    ``charge_balance_component`` explicitly replaces one component conservation
    equation by sum(z*c)+fixed_charge=0. Its supplied total is only an initial
    scale; it is not conserved. With no replacement, every component is conserved
    and charge residual is reported, not silently forced. Charge-balanced closed
    inputs stay charge balanced because every formation reaction conserves charge.

    Davies uses log10(gamma)=-A*z²*(sqrt(I)/(1+sqrt(I))-0.3*I), with
    I=0.5*sum(z²*c/reference_concentration), qualified only to I<=maximum_ionic_strength.
    It is a concentration-standard-state approximation, not a molality/Pitzer model.
    """

    primary_names: tuple[str, ...] = eqx.field(static=True)
    secondary_names: tuple[str, ...] = eqx.field(static=True)
    stoichiometry: Array
    log_k: Array
    charges: Array
    activity_model: str = eqx.field(static=True)
    reference_concentration: float = eqx.field(static=True)
    davies_a: float = eqx.field(static=True)
    maximum_ionic_strength: float = eqx.field(static=True)
    charge_balance_component: int | None = eqx.field(static=True)

    def __init__(
        self,
        primary_names: tuple[str, ...],
        secondary_names: tuple[str, ...],
        stoichiometry: ArrayLike,
        log_k: ArrayLike,
        charges: ArrayLike,
        *,
        activity_model: str = "ideal",
        reference_concentration: float = 1000.0,
        davies_a: float = 0.509,
        maximum_ionic_strength: float = 0.5,
        charge_balance_component: int | None = None,
    ):
        primary = tuple(primary_names)
        secondary = tuple(secondary_names)
        names = primary + secondary
        if (
            not primary
            or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n.strip() for n in names)
        ):
            raise ValueError(
                "Chemical species require nonempty unique names and a primary basis."
            )
        b, s = len(primary), len(secondary)
        nu = jnp.asarray(stoichiometry, dtype=float)
        constants = jnp.asarray(log_k, dtype=float)
        z = jnp.asarray(charges, dtype=float)
        if nu.shape != (s, b) or constants.shape != (s,) or z.shape != (b + s,):
            raise ValueError(
                "Stoichiometry, log_k and charges must match the declared primary/secondary basis."
            )
        nu = eqx.error_if(nu, jnp.any(~jnp.isfinite(nu)), "Stoichiometry must be finite.")
        constants = eqx.error_if(
            constants,
            jnp.any(~jnp.isfinite(constants)),
            "Formation constants must be finite.",
        )
        z = eqx.error_if(z, jnp.any(~jnp.isfinite(z)), "Species charges must be finite.")
        z = eqx.error_if(
            z,
            jnp.any(jnp.abs(contract("sb,b->s", nu, z[:b]) - z[b:]) > 1e-10),
            "Formation stoichiometry must conserve charge.",
        )
        if activity_model not in ("ideal", "davies"):
            raise ValueError("activity_model must be ideal or davies.")
        if not np.isfinite(reference_concentration) or reference_concentration <= 0:
            raise ValueError("reference_concentration must be positive and finite.")
        if (
            not np.isfinite(davies_a)
            or davies_a <= 0
            or not 0 < maximum_ionic_strength <= 0.5
        ):
            raise ValueError(
                "Davies requires positive A and an ionic-strength limit in (0, 0.5]."
            )
        if charge_balance_component is not None and (
            not isinstance(charge_balance_component, int)
            or isinstance(charge_balance_component, bool)
            or not 0 <= charge_balance_component < b
        ):
            raise ValueError("charge_balance_component must index one primary component.")
        if charge_balance_component is not None:
            z = eqx.error_if(
                z,
                z[charge_balance_component] == 0,
                "The replaced component must carry charge.",
            )
        self.primary_names, self.secondary_names = primary, secondary
        self.stoichiometry, self.log_k, self.charges = nu, constants, z
        self.activity_model = activity_model
        self.reference_concentration = float(reference_concentration)
        self.davies_a, self.maximum_ionic_strength = (
            float(davies_a),
            float(maximum_ionic_strength),
        )
        self.charge_balance_component = charge_balance_component

    @property
    def component_count(self) -> int:
        return len(self.primary_names)

    @property
    def species_count(self) -> int:
        return self.component_count + len(self.secondary_names)

    def component_totals(self, concentrations: ArrayLike) -> Array:
        c = jnp.asarray(concentrations)
        if c.shape[-1] != self.species_count:
            raise ValueError(
                "Concentration species axis does not match the chemical basis."
            )
        return c[..., : self.component_count] + contract(
            "...s,sb->...b", c[..., self.component_count :], self.stoichiometry
        )

    def ionic_strength(self, concentrations: ArrayLike) -> Array:
        return (
            0.5
            * contract("...s,s->...", jnp.asarray(concentrations), self.charges**2)
            / self.reference_concentration
        )

    def log_activities(self, log_concentrations: ArrayLike) -> Array:
        """Input is log(c/reference_concentration), not log of a dimensional value."""
        x = jnp.asarray(log_concentrations)
        if x.shape[-1] != self.species_count:
            raise ValueError(
                "Log-concentration species axis does not match the chemical basis."
            )
        if self.activity_model == "ideal":
            return x
        strength = self.ionic_strength(self.reference_concentration * jnp.exp(x))
        # For an entirely neutral system gamma=1; avoid sqrt'(0) in AD.
        charged = jnp.any(self.charges != 0)
        root = jnp.sqrt(jnp.where(charged, strength, 1.0))
        correction = (
            -jnp.log(10.0) * self.davies_a * (root / (1.0 + root) - 0.3 * strength)
        )
        return x + correction[..., None] * self.charges**2

    def valid(self, log_concentrations: ArrayLike) -> Array:
        x = jnp.asarray(log_concentrations)
        c = self.reference_concentration * jnp.exp(x)
        valid = jnp.all(jnp.isfinite(c) & (c > 0))
        if self.activity_model == "davies":
            valid = valid & jnp.all(self.ionic_strength(c) <= self.maximum_ionic_strength)
        return valid

    def residual(
        self,
        log_concentrations: ArrayLike,
        totals: ArrayLike,
        *,
        fixed_charge: ArrayLike = 0.0,
    ) -> Array:
        """Scaled component equations followed by logarithmic mass-action equations.

        Exposed for monolithic transport/reaction roots. Last axis holds species;
        arbitrary batch axes are supported without coupling local equilibria.
        """
        x, target = jnp.asarray(log_concentrations), jnp.asarray(totals)
        if x.shape[-1] != self.species_count or target.shape != x.shape[:-1] + (
            self.component_count,
        ):
            raise ValueError(
                "Speciation residual shapes do not match the declared basis."
            )
        c = self.reference_concentration * jnp.exp(x)
        scale = jnp.maximum(jnp.abs(target), self.reference_concentration * 1e-12)
        balance = (self.component_totals(c) - target) / scale
        if self.charge_balance_component is not None:
            charge = contract("...s,s->...", c, self.charges) + jnp.asarray(fixed_charge)
            charge_scale = jnp.maximum(
                jnp.max(scale, axis=-1), jnp.abs(jnp.asarray(fixed_charge))
            )
            balance = balance.at[..., self.charge_balance_component].set(
                charge / charge_scale
            )
        log_a = self.log_activities(x)
        action = (
            log_a[..., self.component_count :]
            - contract(
                "...b,sb->...s", log_a[..., : self.component_count], self.stoichiometry
            )
            - self.log_k
        )
        return jnp.concatenate((balance, action), axis=-1)

    def solve(
        self,
        totals: ArrayLike,
        *,
        initial_concentrations: ArrayLike,
        fixed_charge: ArrayLike = 0.0,
        method: AbstractNonlinearMethod | None = None,
        termination: NonlinearTermination | None = None,
    ) -> SpeciationResult:
        """Solve one local cell; use jax.vmap for independent cell chemistry."""
        target, initial = jnp.asarray(totals), jnp.asarray(initial_concentrations)
        if target.shape != (self.component_count,) or initial.shape != (
            self.species_count,
        ):
            raise ValueError("Local chemistry requires one component/species vector.")
        target = eqx.error_if(
            target, jnp.any(~jnp.isfinite(target)), "Component totals must be finite."
        )
        initial = eqx.error_if(
            initial,
            jnp.any(~jnp.isfinite(initial) | (initial <= 0)),
            "Initial species concentrations must be positive and finite.",
        )
        charge = jnp.asarray(fixed_charge)
        if charge.shape != ():
            raise ValueError("fixed_charge must be a local scalar in mol charge/m³.")
        charge = eqx.error_if(
            charge, ~jnp.isfinite(charge), "Fixed charge must be finite."
        )
        problem = NonlinearSystemProblem(
            lambda x, args: self.residual(x, args[0], fixed_charge=args[1]),
            validity=lambda x, residual, auxiliary, args: self.valid(x),
            problem_id="aqueous-mass-action-component-basis",
        )
        root = implicit_root_result(
            problem,
            jnp.log(initial / self.reference_concentration),
            args=(target, charge),
            method=_chemical_method(method),
            termination=_chemical_termination(termination),
        )
        x = eqx.error_if(
            root.state,
            ~root.successful | ~self.valid(root.state),
            "Speciation requires a successful physical root; no failed-root derivatives are admitted.",
        )
        c = self.reference_concentration * jnp.exp(x)
        log_a = self.log_activities(x)
        computed = self.component_totals(c)
        return SpeciationResult(
            c,
            jnp.exp(log_a),
            computed,
            computed - target,
            contract("s,s->", c, self.charges) + charge,
            log_a[self.component_count :]
            - contract("b,sb->s", log_a[: self.component_count], self.stoichiometry)
            - self.log_k,
            root,
        )


__all__ = ["MassActionSystem", "SpeciationResult"]
