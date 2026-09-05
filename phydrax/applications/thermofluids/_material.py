#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ...dynamics import (
    DAEComponent,
    DAEDerivativeIncidence,
    DAEEquationBlock,
    DAEPort,
    DAEVariableBlock,
)
from ...equations import HomogeneousHelmholtzPlan
from ._heat import _heat_port
from ._process import (
    HeatFlowOrientation,
    MaterialFlowDirection,
    ThermofluidComponent,
    ThermofluidPortKind,
    ThermofluidPortSpec,
)


def _species_count(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("species_count must be a nonnegative integer.")
    return value


def _material_port(
    name,
    prefix,
    direction,
    catalog_id,
    thermodynamics_id,
    species_count,
    *,
    orientation=1,
):
    potentials = (f"{prefix}pressure", f"{prefix}specific_enthalpy") + tuple(
        f"{prefix}mass_fraction_{index}" for index in range(species_count)
    )
    return (
        DAEPort(name, potentials, (f"{prefix}mass_flow",)),
        ThermofluidPortSpec(
            name,
            ThermofluidPortKind.MATERIAL,
            direction,
            catalog_id=catalog_id,
            thermodynamics_id=thermodynamics_id,
            mass_flow_orientation=orientation,
            state_pair=(
                "pressure-enthalpy"
                if species_count == 0
                else f"pressure-enthalpy-mass-fractions:{species_count}"
            ),
        ),
    )


def material_boundary_component(
    name: str,
    /,
    *,
    catalog_id: str,
    thermodynamics_id: str,
    direction: MaterialFlowDirection,
    pressure: float | None = None,
    specific_enthalpy: float | None = None,
    mass_flow: float | None = None,
    species_count: int = 0,
    mass_fractions: tuple[float, ...] | None = None,
) -> ThermofluidComponent:
    """Prescribe only known boundary data; unspecified values are solved.

    A flow/enthalpy source leaves pressure free; a pressure sink leaves enthalpy
    and flow free. This closes a valve or mixer without prescribing its answer.
    Mass-flow signs follow the existing directed process convention: a source
    outlet is positive and a sink inlet negative. Within a through component,
    its inlet is positive and outlet negative. Explicit orientation metadata
    conserves mass at links, including between successive through components.
    """
    count = _species_count(species_count)
    if direction not in (MaterialFlowDirection.INLET, MaterialFlowDirection.OUTLET):
        raise ValueError("A material boundary must have a fixed inlet/outlet direction.")
    prescribed = {}
    for variable, value in (
        ("pressure", pressure),
        ("specific_enthalpy", specific_enthalpy),
        ("mass_flow", mass_flow),
    ):
        if value is not None:
            target = float(value)
            if not np.isfinite(target) or (variable == "pressure" and target <= 0):
                raise ValueError(
                    "Boundary data must be finite; pressure must be positive."
                )
            prescribed[variable] = target
    if mass_flow is not None:
        if (direction is MaterialFlowDirection.OUTLET and mass_flow < 0) or (
            direction is MaterialFlowDirection.INLET and mass_flow > 0
        ):
            raise ValueError(
                "Boundary mass_flow disagrees with its fixed stream direction."
            )
    if mass_fractions is not None:
        fractions = tuple(float(value) for value in mass_fractions)
        if len(fractions) != count or count == 0:
            raise ValueError("mass_fractions must match a positive species_count.")
        if any(
            not np.isfinite(value) or value < 0 for value in fractions
        ) or not np.isclose(sum(fractions), 1.0, rtol=0.0, atol=1.0e-10):
            raise ValueError("Mass fractions must be nonnegative, finite and normalized.")
        prescribed.update(
            {f"mass_fraction_{index}": value for index, value in enumerate(fractions)}
        )
    port, typed = _material_port(
        "material",
        "",
        direction,
        catalog_id,
        thermodynamics_id,
        count,
        orientation=-1,
    )
    variable_names = ("pressure", "specific_enthalpy", "mass_flow") + tuple(
        f"mass_fraction_{index}" for index in range(count)
    )

    def prescribe(variable, target):
        def residual(time, jet, args):
            del time, args
            return jet.value(variable) - target

        return residual

    return ThermofluidComponent(
        DAEComponent(
            name,
            tuple(
                DAEVariableBlock(
                    variable, (), 0, max(abs(prescribed.get(variable, 1.0)), 1.0)
                )
                for variable in variable_names
            ),
            tuple(
                DAEEquationBlock(
                    f"prescribe_{variable}",
                    prescribe(variable, target),
                    (DAEDerivativeIncidence(variable),),
                )
                for variable, target in prescribed.items()
            ),
            (port,),
        ),
        (typed,),
        model_parameters=tuple(prescribed.items()),
    )


def material_mixer_component(
    name: str,
    /,
    *,
    inlet_count: int,
    species_count: int,
    catalog_id: str,
    thermodynamics_id: str,
) -> ThermofluidComponent:
    """Steady adiabatic, zero-volume mixer with explicit advective balances.

    Ports ``inlet_0``, ... and ``outlet`` share pressure, NOT enthalpy or
    composition. Sum(m)=0, sum(m*h)=0, sum(m*Y_s)=0. Inlet flows are positive;
    outlet flow is negative. Positive total inflow is required for a unique
    outlet state; zero flow is singular, not an invented mixture or fallback.
    """
    if (
        not isinstance(inlet_count, int)
        or isinstance(inlet_count, bool)
        or inlet_count < 2
    ):
        raise ValueError("inlet_count must be an integer >= 2.")
    count = _species_count(species_count)
    names = tuple(f"inlet_{index}" for index in range(inlet_count)) + ("outlet",)
    fields = ("pressure", "specific_enthalpy", "mass_flow") + tuple(
        f"mass_fraction_{index}" for index in range(count)
    )
    ports = tuple(
        _material_port(
            port,
            f"{port}_",
            MaterialFlowDirection.OUTLET
            if port == "outlet"
            else MaterialFlowDirection.INLET,
            catalog_id,
            thermodynamics_id,
            count,
        )
        for port in names
    )

    def equal_pressure(inlet):
        def residual(time, jet, args):
            del time, args
            return jet.value(f"{inlet}_pressure") - jet.value("outlet_pressure")

        return residual

    def mass_balance(time, jet, args):
        del time, args
        return sum(jet.value(f"{port}_mass_flow") for port in names)

    def advective_balance(field):
        def residual(time, jet, args):
            del time, args
            return sum(
                jet.value(f"{port}_mass_flow") * jet.value(f"{port}_{field}")
                for port in names
            )

        return residual

    equations = (
        tuple(
            DAEEquationBlock(
                f"pressure_{inlet}",
                equal_pressure(inlet),
                (
                    DAEDerivativeIncidence(f"{inlet}_pressure"),
                    DAEDerivativeIncidence("outlet_pressure"),
                ),
            )
            for inlet in names[:-1]
        )
        + (
            DAEEquationBlock(
                "mass_balance",
                mass_balance,
                tuple(DAEDerivativeIncidence(f"{port}_mass_flow") for port in names),
            ),
        )
        + tuple(
            DAEEquationBlock(
                f"advective_{field}",
                advective_balance(field),
                tuple(
                    DAEDerivativeIncidence(f"{port}_{variable}")
                    for port in names
                    for variable in ("mass_flow", field)
                ),
            )
            for field in ("specific_enthalpy",)
            + tuple(f"mass_fraction_{index}" for index in range(count))
        )
    )
    return ThermofluidComponent(
        DAEComponent(
            name,
            tuple(
                DAEVariableBlock(f"{port}_{field}", (), 0, 1.0)
                for port in names
                for field in fields
            ),
            equations,
            tuple(port[0] for port in ports),
        ),
        tuple(port[1] for port in ports),
    )


def homogeneous_fluid_heat_exchanger_component(
    name: str,
    /,
    *,
    thermodynamics: HomogeneousHelmholtzPlan,
    mole_fraction: tuple[float, ...],
    conductance: float,
) -> ThermofluidComponent:
    """A steady, isobaric, perfectly mixed single-phase fluid heat exchanger.

    Fixed composition is evaluated by the native homogeneous Helmholtz provider;
    temperature and molar density are simultaneous DAE unknowns, not a second
    property inversion solver. Q_into = G (T_heat - T_outlet), and advected
    enthalpy closes the energy balance. No phase transition or reverse flow is
    implied. Inspect provider ``evidence.successful`` at the solved outlet.
    """
    if not isinstance(thermodynamics, HomogeneousHelmholtzPlan):
        raise TypeError("thermodynamics must be HomogeneousHelmholtzPlan.")
    fraction = tuple(float(value) for value in mole_fraction)
    if (
        len(fraction) != thermodynamics.schema.species_count
        or any(not np.isfinite(value) or value < 0 for value in fraction)
        or not np.isclose(
            sum(fraction), 1.0, rtol=0.0, atol=thermodynamics.composition_tolerance
        )
    ):
        raise ValueError(
            "mole_fraction must match the provider's normalized species catalog."
        )
    conductance_value = float(conductance)
    if not np.isfinite(conductance_value) or conductance_value <= 0:
        raise ValueError("conductance must be finite and positive.")
    catalog = thermodynamics.schema.catalog.catalog_id
    model = thermodynamics.model_id
    inlet = _material_port(
        "inlet", "inlet_", MaterialFlowDirection.INLET, catalog, model, 0
    )
    outlet = _material_port(
        "outlet", "outlet_", MaterialFlowDirection.OUTLET, catalog, model, 0
    )
    heat = _heat_port(
        "heat", "heat_temperature", "heat_flow", HeatFlowOrientation.INTO_COMPONENT
    )

    def pressure_balance(time, jet, args):
        del time, args
        return jet.value("inlet_pressure") - jet.value("outlet_pressure")

    def mass_balance(time, jet, args):
        del time, args
        return jet.value("inlet_mass_flow") + jet.value("outlet_mass_flow")

    def state(jet):
        return thermodynamics.evaluate(
            jet.value("temperature"), jet.value("molar_density"), jnp.asarray(fraction)
        )

    def pressure_closure(time, jet, args):
        del time, args
        return jet.value("outlet_pressure") - state(jet).pressure

    def enthalpy_closure(time, jet, args):
        del time, args
        evaluation = state(jet)
        return (
            jet.value("outlet_specific_enthalpy")
            - evaluation.molar_enthalpy / evaluation.molar_mass
        )

    def energy_balance(time, jet, args):
        del time, args
        return (
            jet.value("inlet_mass_flow") * jet.value("inlet_specific_enthalpy")
            + jet.value("outlet_mass_flow") * jet.value("outlet_specific_enthalpy")
            + jet.value("heat_flow")
        )

    def heat_transfer(time, jet, args):
        del time, args
        return jet.value("heat_flow") - conductance_value * (
            jet.value("heat_temperature") - jet.value("temperature")
        )

    specifications = (
        ("pressure_balance", pressure_balance, ("inlet_pressure", "outlet_pressure")),
        ("mass_balance", mass_balance, ("inlet_mass_flow", "outlet_mass_flow")),
        (
            "pressure_closure",
            pressure_closure,
            ("outlet_pressure", "temperature", "molar_density"),
        ),
        (
            "enthalpy_closure",
            enthalpy_closure,
            ("outlet_specific_enthalpy", "temperature", "molar_density"),
        ),
        (
            "energy_balance",
            energy_balance,
            (
                "inlet_mass_flow",
                "inlet_specific_enthalpy",
                "outlet_mass_flow",
                "outlet_specific_enthalpy",
                "heat_flow",
            ),
        ),
        (
            "heat_transfer",
            heat_transfer,
            ("heat_flow", "heat_temperature", "temperature"),
        ),
    )
    return ThermofluidComponent(
        DAEComponent(
            name,
            tuple(
                DAEVariableBlock(variable, (), 0, 1.0)
                for variable in (
                    "inlet_pressure",
                    "inlet_specific_enthalpy",
                    "inlet_mass_flow",
                    "outlet_pressure",
                    "outlet_specific_enthalpy",
                    "outlet_mass_flow",
                    "temperature",
                    "molar_density",
                    "heat_temperature",
                    "heat_flow",
                )
            ),
            tuple(
                DAEEquationBlock(
                    equation,
                    residual,
                    tuple(DAEDerivativeIncidence(variable) for variable in variables),
                )
                for equation, residual, variables in specifications
            ),
            (inlet[0], outlet[0], heat[0]),
        ),
        (inlet[1], outlet[1], heat[1]),
        model_parameters=(("conductance", conductance_value),)
        + tuple(
            (f"mole_fraction_{index}", value) for index, value in enumerate(fraction)
        ),
    )


__all__ = [
    "homogeneous_fluid_heat_exchanger_component",
    "material_boundary_component",
    "material_mixer_component",
]
