#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....dynamics import DAEComponent, DAEEquationBlock, DAEVariableBlock
from ._components import (
    _finite_scalar,
    _incidence,
    _positive_scalar,
    _PressureEqualityResidual,
    _two_port_variables,
    _two_ports,
    _VolumeBalanceResidual,
    PressureFlowComponent,
    PressureSource,
    Resistance,
    StorageOwner,
)
from ._network import CirculationNetwork, PressureFlowConnection


ExtravascularPressureWaveform = Callable[[Array], Array]


class PhasicExtravascularPressure(StrictModule):
    """Smooth systolic intramyocardial pressure in kPa."""

    baseline: Array
    amplitude: Array
    cycle_length: Array
    systolic_duration: Array
    phase_offset: Array
    waveform_id: str = eqx.field(static=True)

    def __init__(
        self,
        baseline: ArrayLike,
        amplitude: ArrayLike,
        cycle_length: ArrayLike,
        systolic_duration: ArrayLike,
        /,
        *,
        phase_offset: ArrayLike = 0.0,
    ) -> None:
        arrays = tuple(
            jnp.asarray(value)
            for value in (
                baseline,
                amplitude,
                cycle_length,
                systolic_duration,
                phase_offset,
            )
        )
        if any(value.shape != () for value in arrays):
            raise ValueError("Phasic pressure parameters must be scalars.")
        values = tuple(float(value) for value in arrays)
        if (
            any(not np.isfinite(value) for value in values)
            or values[0] < 0.0
            or values[1] < 0.0
            or values[2] <= 0.0
            or values[3] <= 0.0
            or values[3] > values[2]
        ):
            raise ValueError("Phasic pressure parameters must be finite and physical.")
        (
            self.baseline,
            self.amplitude,
            self.cycle_length,
            self.systolic_duration,
            self.phase_offset,
        ) = arrays
        self.waveform_id = canonical_fingerprint(
            {
                "kind": "phasic-extravascular-pressure",
                "baseline": values[0].hex(),
                "amplitude": values[1].hex(),
                "cycle_length": values[2].hex(),
                "systolic_duration": values[3].hex(),
                "phase_offset": values[4].hex(),
            }
        )

    def __call__(self, time: Array, /) -> Array:
        phase = jnp.mod(jnp.asarray(time) - self.phase_offset, self.cycle_length)
        normalized = jnp.clip(phase / self.systolic_duration, 0.0, 1.0)
        activation = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * normalized))
        activation = jnp.where(phase <= self.systolic_duration, activation, 0.0)
        return self.baseline + self.amplitude * activation


class _CoronaryConstitutiveResidual(StrictModule):
    compliance: Array
    unstressed_volume: Array
    extravascular_pressure: ExtravascularPressureWaveform

    def __call__(self, time: Array, jet, args: Any, /) -> Array:
        del args
        transmural_pressure = jet.value("pressure_in") - self.extravascular_pressure(time)
        return (
            jet.value("volume")
            - self.unstressed_volume
            - self.compliance * transmural_pressure
        )


class CoronaryCompliance(PressureFlowComponent):
    """Intramyocardial compliance referenced to phasic extravascular pressure."""

    compliance: Array
    unstressed_volume: Array
    initial_volume: Array
    extravascular_pressure: ExtravascularPressureWaveform
    waveform_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        compliance: ArrayLike,
        extravascular_pressure: ExtravascularPressureWaveform,
        /,
        *,
        waveform_id: str | None = None,
        unstressed_volume: ArrayLike = 0.0,
        initial_volume: ArrayLike,
        pressure_scale: float = 15.0,
        flow_scale: float = 10.0,
        volume_scale: float = 20_000.0,
    ) -> None:
        compliance_, compliance_host = _positive_scalar(compliance, "compliance")
        if not callable(extravascular_pressure):
            raise TypeError("extravascular_pressure must be callable.")
        if isinstance(extravascular_pressure, PhasicExtravascularPressure):
            identifier = extravascular_pressure.waveform_id
        elif waveform_id is not None and str(waveform_id).strip():
            identifier = str(waveform_id).strip()
        else:
            raise ValueError(
                "Custom extravascular pressure requires a stable waveform_id."
            )
        unstressed, unstressed_host = _finite_scalar(
            unstressed_volume, "unstressed_volume"
        )
        initial, initial_host = _finite_scalar(initial_volume, "initial_volume")
        if initial_host < 0.0:
            raise ValueError("initial_volume must be nonnegative.")
        external_initial = jnp.asarray(extravascular_pressure(jnp.asarray(0.0)))
        if external_initial.shape != () or not bool(jnp.isfinite(external_initial)):
            raise ValueError("extravascular_pressure must return a finite scalar.")
        initial_pressure = external_initial + (initial - unstressed) / compliance_
        p_scale = _positive_scalar(pressure_scale, "pressure_scale")[1]
        q_scale = _positive_scalar(flow_scale, "flow_scale")[1]
        v_scale = _positive_scalar(volume_scale, "volume_scale")[1]
        variables = _two_port_variables(p_scale, q_scale) + (
            DAEVariableBlock("volume", (), 1, v_scale),
        )
        component = DAEComponent(
            name,
            variables,
            (
                DAEEquationBlock(
                    "equal_pressure",
                    _PressureEqualityResidual(),
                    _incidence(("pressure_in", 0), ("pressure_out", 0)),
                ),
                DAEEquationBlock(
                    "transmural_compliance",
                    _CoronaryConstitutiveResidual(
                        compliance_, unstressed, extravascular_pressure
                    ),
                    _incidence(("volume", 0), ("pressure_in", 0)),
                ),
                DAEEquationBlock(
                    "volume_balance",
                    _VolumeBalanceResidual(),
                    _incidence(("volume", 1), ("flow_in", 0), ("flow_out", 0)),
                ),
            ),
            _two_ports(),
        )
        PressureFlowComponent.__init__(
            self,
            component,
            component_kind="coronary-transmural-compliance",
            parameters=(
                ("compliance_mm3_per_kPa", compliance_host),
                ("unstressed_volume_mm3", unstressed_host),
                ("waveform_id", identifier),
            ),
            storage_owner=StorageOwner.CIRCULATION,
            storage_variable_names=("volume",),
            initial_values=(
                ("pressure_in", initial_pressure),
                ("pressure_out", initial_pressure),
                ("volume", initial),
            ),
        )
        self.compliance = compliance_
        self.unstressed_volume = unstressed
        self.initial_volume = initial
        self.extravascular_pressure = extravascular_pressure
        self.waveform_id = identifier

    def transmural_pressure(
        self, time: ArrayLike, intravascular_pressure: ArrayLike, /
    ) -> Array:
        return jnp.asarray(intravascular_pressure) - self.extravascular_pressure(
            jnp.asarray(time)
        )

    def stored_energy(self, volume: ArrayLike, /) -> Array:
        displacement = jnp.asarray(volume) - self.unstressed_volume
        return 0.5 * displacement * displacement / self.compliance


class CoronaryCirculation(StrictModule):
    network: CirculationNetwork
    bed_name: str = eqx.field(static=True)
    reference_total_volume: Array
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        network: CirculationNetwork,
        bed_name: str,
        reference_total_volume: ArrayLike,
        /,
    ) -> None:
        if not isinstance(network, CirculationNetwork) or not network.closed:
            raise ValueError("Coronary circulation requires a closed network.")
        bed = network.component(bed_name)
        if not isinstance(bed, CoronaryCompliance):
            raise TypeError("bed_name must identify a CoronaryCompliance.")
        total = jnp.asarray(reference_total_volume)
        if total.shape != () or not bool(jnp.isfinite(total) & (total > 0.0)):
            raise ValueError("reference_total_volume must be finite and positive.")
        self.network = network
        self.bed_name = str(bed_name)
        self.reference_total_volume = total
        self.model_id = canonical_fingerprint(
            {
                "kind": "coronary-circulation",
                "network": network.network_id,
                "bed": self.bed_name,
                "reference_total_volume": float(total).hex(),
            }
        )


def coronary_closed_loop(
    *,
    perfusion_pressure: ArrayLike = 12.0,
    cycle_length: ArrayLike = 800.0,
    epicardial_resistance: ArrayLike = 0.12,
    microvascular_resistance: ArrayLike = 0.4,
    venous_resistance: ArrayLike = 0.08,
    compliance: ArrayLike = 2_500.0,
    unstressed_volume: ArrayLike = 15_000.0,
    initial_volume: ArrayLike = 25_000.0,
) -> CoronaryCirculation:
    """Build a closed coronary bed with phasic intramyocardial compression."""

    cycle, _ = _positive_scalar(cycle_length, "cycle_length")
    components: tuple[PressureFlowComponent, ...] = (
        PressureSource(
            "coronary_perfusion_source",
            perfusion_pressure,
            pressure_scale=15.0,
            flow_scale=10.0,
        ),
        Resistance(
            "epicardial_resistance",
            epicardial_resistance,
            pressure_scale=15.0,
            flow_scale=10.0,
        ),
        Resistance(
            "microvascular_resistance",
            microvascular_resistance,
            pressure_scale=15.0,
            flow_scale=10.0,
        ),
        CoronaryCompliance(
            "intramyocardial_bed",
            compliance,
            PhasicExtravascularPressure(0.5, 10.0, cycle, 0.38 * cycle),
            unstressed_volume=unstressed_volume,
            initial_volume=initial_volume,
            pressure_scale=15.0,
            flow_scale=10.0,
            volume_scale=30_000.0,
        ),
        Resistance(
            "coronary_venous_resistance",
            venous_resistance,
            pressure_scale=15.0,
            flow_scale=10.0,
        ),
    )
    names = tuple(value.name for value in components)
    connections = tuple(
        PressureFlowConnection(left, "outlet", right, "inlet")
        for left, right in zip(names, names[1:] + names[:1], strict=True)
    )
    network = CirculationNetwork(components, connections)
    return CoronaryCirculation(network, "intramyocardial_bed", initial_volume)


__all__ = [
    "CoronaryCirculation",
    "CoronaryCompliance",
    "ExtravascularPressureWaveform",
    "PhasicExtravascularPressure",
    "coronary_closed_loop",
]
