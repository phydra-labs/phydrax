#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Volume-based gas, liquid, interface and drive contracts for radial bubbles.

Every law is a `StrictModule` whose static fields are structural selectors and
capacities and whose dynamic leaves are the physical coefficients. Stacking
homogeneous laws along a leading lane axis (for example with
`jax.tree.map(jnp.stack, ...)`) therefore yields a batched species group that
`eqx.filter_vmap` can evaluate without per-bubble dispatch.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState, parameter_field


MOLAR_GAS_CONSTANT = 8.31446261815324
"""Exact SI molar gas constant `N_A k_B` in J mol⁻¹ K⁻¹."""


def scalar_parameter(
    value: ArrayLike,
    name: str,
    /,
    *,
    lower: float | None = None,
    inclusive: bool = False,
    upper: float | None = None,
) -> Array:
    """Validate one finite scalar physical coefficient and return a float64 leaf.

    Validation happens on the host before assignment; laws are constructed
    outside traced code and transformed afterwards as PyTrees.
    """
    host = np.asarray(value, dtype=np.float64)
    if host.shape != ():
        raise ValueError(f"{name} must be a scalar.")
    number = float(host)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    if lower is not None and (number < lower if inclusive else number <= lower):
        relation = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be {relation} {lower}.")
    if upper is not None and number >= upper:
        raise ValueError(f"{name} must be < {upper}.")
    return jnp.asarray(host, dtype=jnp.float64)


class BubbleEnvironment(StrictModule):
    """Far-field liquid state shared by every law of one bubble.

    `ambient_pressure` is the static far-field liquid pressure `p0`; drives add
    an excess pressure to it. `vapor_pressure` is added to the gas pressure at
    the wall. All three coefficients are trainable parameter leaves.
    """

    ambient_pressure: Array = parameter_field()
    ambient_temperature: Array = parameter_field()
    vapor_pressure: Array = parameter_field()

    def __init__(
        self,
        ambient_pressure: ArrayLike,
        ambient_temperature: ArrayLike,
        /,
        *,
        vapor_pressure: ArrayLike = 0.0,
    ) -> None:
        pressure = scalar_parameter(
            ambient_pressure, "ambient_pressure", lower=0.0, inclusive=True
        )
        temperature = scalar_parameter(ambient_temperature, "ambient_temperature", lower=0.0)
        vapor = scalar_parameter(vapor_pressure, "vapor_pressure", lower=0.0, inclusive=True)
        self.ambient_pressure = pressure
        self.ambient_temperature = temperature
        self.vapor_pressure = vapor


class BubbleScales(StrictModule, NonTrainableState):
    """Characteristic scales used to nondimensionalize one radial solve."""

    radius: Array
    velocity: Array
    time: Array
    pressure: Array
    temperature: Array
    energy: Array
    amount: Array


class BubbleGasState(StrictModule):
    """Extensive gas content of one bubble.

    `amount` is the gas amount of substance in mol. `internal_energy` is the
    thermodynamic internal energy in J for caloric laws and `None` for closures
    such as polytropic laws that do not define one. `internal` is the fixed-
    shape law-specific state (possibly empty).
    """

    amount: Array
    internal_energy: Array | None
    internal: Array

    @property
    def finite(self) -> Array:
        """Whether every stored value is finite."""
        finite = jnp.isfinite(self.amount) & jnp.all(jnp.isfinite(self.internal))
        if self.internal_energy is None:
            return finite
        return finite & jnp.isfinite(self.internal_energy)


class BubbleGasEvaluation(StrictModule):
    """Gas pressure, temperature, state rates and admissibility at one volume."""

    pressure: Array
    temperature: Array
    internal_rate: Array
    amount_rate: Array
    energy_rate: Array | None
    heat_rate: Array | None
    hard_core_margin: Array
    admissible: Array


class BubbleGasCapabilities(StrictModule, NonTrainableState):
    """Static thermodynamic support declared by one gas law."""

    caloric: bool = eqx.field(static=True)
    heat_transfer: bool = eqx.field(static=True)
    mass_transfer: bool = eqx.field(static=True)
    mixing: bool = eqx.field(static=True)
    stiff: bool = eqx.field(static=True)


class AbstractBubbleGasLaw(StrictModule):
    """Volume-based gas closure shared by radial, resolved and foam bubbles."""

    law_id: eqx.AbstractVar[str]

    @property
    @abc.abstractmethod
    def capabilities(self) -> BubbleGasCapabilities:
        """Static energy, heat, mass-transfer, mixing and stiffness support."""
        raise NotImplementedError

    @abc.abstractmethod
    def initialize(
        self,
        reference_volume: Array,
        reference_pressure: Array,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasState:
        """Gas at rest at `reference_volume`, `reference_pressure` and the ambient temperature."""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self,
        volume: Array,
        volume_rate: Array,
        state: BubbleGasState,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasEvaluation:
        """Pressure, temperature and rates for the instantaneous bubble volume."""
        raise NotImplementedError

    @abc.abstractmethod
    def internal_scale(self, scales: BubbleScales, /) -> Array:
        """Positive characteristic magnitude of every internal-state entry."""
        raise NotImplementedError


class BubbleGasMergeResult(StrictModule):
    """Conservative merge of several gas compartments into one."""

    state: BubbleGasState
    pressure: Array
    temperature: Array
    entropy_production: Array
    amount_residual: Array
    energy_residual: Array
    volume_change: Array
    admissible: Array


class BubbleGasSplitResult(StrictModule):
    """Partition of one gas compartment under a declared equilibrium policy."""

    states: BubbleGasState
    entropy_production: Array
    amount_residual: Array
    energy_residual: Array
    volume_residual: Array
    admissible: Array
    policy: str = eqx.field(static=True)


class AbstractBubbleCompartmentGasLaw(AbstractBubbleGasLaw):
    """Caloric gas law whose extensive state can be merged and split.

    States carry gas amount and internal energy. `merge` conserves both and
    reports the irreversible entropy production of mixing at the merged volume;
    `split` partitions both under the uniform-intensive policy (every child has
    the parent pressure and temperature), which produces no entropy.
    """

    @abc.abstractmethod
    def entropy(
        self,
        volume: Array,
        state: BubbleGasState,
        environment: BubbleEnvironment,
        /,
    ) -> Array:
        """Gas entropy in J K⁻¹ up to a law-constant reference."""
        raise NotImplementedError

    @abc.abstractmethod
    def merge(
        self,
        states: BubbleGasState,
        volumes: Array,
        merged_volume: Array,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasMergeResult:
        """Merge compartments stacked along the leading axis of `states`."""
        raise NotImplementedError

    @abc.abstractmethod
    def split(
        self,
        state: BubbleGasState,
        volume: Array,
        child_volumes: Array,
        environment: BubbleEnvironment,
        /,
    ) -> BubbleGasSplitResult:
        """Split one compartment into children with the given volumes."""
        raise NotImplementedError


class BubbleLiquidEvaluation(StrictModule):
    """Liquid stress integral at the wall, internal rates and dissipation.

    `stress` is `S = 2 ∫_R^∞ (τ_rr − τ_θθ)/r dr`, added to the wall pressure.
    """

    stress: Array
    internal_rate: Array
    dissipation_rate: Array
    admissible: Array


class AbstractBubbleLiquidLaw(StrictModule):
    """Rheology of the liquid surrounding a spherical bubble."""

    law_id: eqx.AbstractVar[str]

    @property
    @abc.abstractmethod
    def stiff(self) -> bool:
        """Whether the internal relaxation requires a stiff integrator."""
        raise NotImplementedError

    @abc.abstractmethod
    def initialize(
        self, reference_radius: Array, environment: BubbleEnvironment, /
    ) -> Array:
        """Stress-free internal state at the reference radius."""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self, radius: Array, wall_velocity: Array, internal: Array, /
    ) -> BubbleLiquidEvaluation:
        """Wall stress integral for the liquid boundary kinematics."""
        raise NotImplementedError

    @abc.abstractmethod
    def internal_scale(self, scales: BubbleScales, /) -> Array:
        """Positive characteristic magnitude of every internal-state entry."""
        raise NotImplementedError


class BubbleInterfaceEvaluation(StrictModule):
    """Interface stresses, liquid-side kinematics and regime-local rates.

    The wall pressure contribution is
    `−capillary_pressure − elastic_pressure − viscous_pressure`. The liquid law is
    evaluated at `outer_radius`/`outer_wall_velocity`, which differ from the
    gas-side kinematics only for shells of finite thickness.
    """

    surface_tension: Array
    capillary_pressure: Array
    elastic_pressure: Array
    viscous_pressure: Array
    outer_radius: Array
    outer_wall_velocity: Array
    internal_rate: Array
    dissipation_rate: Array
    admissible: Array


class AbstractBubbleInterfaceLaw(StrictModule):
    """Gas–liquid interface or encapsulating shell.

    Piecewise laws expose a closed set of regimes. Inside one regime the law is
    smooth. `regime_guards` returns one signed value per guard that is positive
    while the state remains in `regime`; a downcrossing through zero ends the
    regime and `regime_after_crossing` selects the next one.
    """

    law_id: eqx.AbstractVar[str]

    @property
    @abc.abstractmethod
    def regime_names(self) -> tuple[str, ...]:
        """Names of the closed regime set; one name for smooth laws."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def guard_count(self) -> int:
        """Number of signed regime guards."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def stiff(self) -> bool:
        """Whether the internal relaxation requires a stiff integrator."""
        raise NotImplementedError

    @abc.abstractmethod
    def initialize(
        self, reference_radius: Array, environment: BubbleEnvironment, /
    ) -> Array:
        """Internal state at the reference (equilibrium) radius."""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self,
        radius: Array,
        wall_velocity: Array,
        internal: Array,
        regime: Array,
        /,
    ) -> BubbleInterfaceEvaluation:
        """Interface stresses inside one fixed regime."""
        raise NotImplementedError

    @abc.abstractmethod
    def internal_scale(self, scales: BubbleScales, /) -> Array:
        """Positive characteristic magnitude of every internal-state entry."""
        raise NotImplementedError

    @abc.abstractmethod
    def initial_regime(self, radius: Array, internal: Array, /) -> Array:
        """Regime occupied by a bubble at rest at `radius`."""
        raise NotImplementedError

    @abc.abstractmethod
    def regime_guards(self, radius: Array, internal: Array, regime: Array, /) -> Array:
        """Dimensionless guards, positive while `regime` remains valid."""
        raise NotImplementedError

    @abc.abstractmethod
    def regime_after_crossing(
        self,
        radius: Array,
        wall_velocity: Array,
        internal: Array,
        regime: Array,
        guard: Array,
        /,
    ) -> Array:
        """Regime entered after `guard` crosses zero from inside `regime`."""
        raise NotImplementedError


class PressureDriveEvaluation(StrictModule):
    """Excess far-field pressure, its time derivative and support evidence."""

    pressure: Array
    pressure_rate: Array
    in_support: Array


class AbstractBubblePressureDrive(StrictModule):
    """Excess far-field liquid pressure `p_d(t)`; `p_∞(t) = p0 + p_d(t)`."""

    drive_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def evaluate(self, time: Array, /) -> PressureDriveEvaluation:
        """Excess pressure and rate at `time`."""
        raise NotImplementedError

    @abc.abstractmethod
    def characteristic_pressure(self) -> Array:
        """Nonnegative pressure magnitude used for nondimensionalization."""
        raise NotImplementedError


__all__ = [
    "MOLAR_GAS_CONSTANT",
    "AbstractBubbleCompartmentGasLaw",
    "AbstractBubbleGasLaw",
    "AbstractBubbleInterfaceLaw",
    "AbstractBubbleLiquidLaw",
    "AbstractBubblePressureDrive",
    "BubbleEnvironment",
    "BubbleGasCapabilities",
    "BubbleGasEvaluation",
    "BubbleGasMergeResult",
    "BubbleGasSplitResult",
    "BubbleGasState",
    "BubbleInterfaceEvaluation",
    "BubbleLiquidEvaluation",
    "BubbleScales",
    "PressureDriveEvaluation",
    "scalar_parameter",
]
