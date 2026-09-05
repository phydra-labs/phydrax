#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Real external-format fixtures exercising electrical and fail-closed contracts."""

from math import pi, sqrt

import numpy as np
import pytest

from phydrax.applications.power._interchange import (
    parse_cgmes,
    parse_matpower,
    parse_psse,
    PowerImportError,
    PowerParserLimits,
)
from phydrax.applications.power._network import compile_network
from phydrax.applications.power._power_flow import solve_power_flow
from phydrax.interchange import AdapterStatus


MATPOWER = """function mpc = analytic_two_bus
% Total three-phase powers are MW/Mvar; no executable code is needed.
mpc.version = '2';
mpc.baseMVA = 50;
mpc.bus = [
1 3 0 0 0 0 1 1 0 110 1 1.1 0.9;
2 1 10 0 0 0 1 1 0 110 1 1.1 0.9;
];
mpc.gen = [
1 10 0 50 -50 1 50 1 50 0;
2 30 5 50 -50 1 50 0 50 0;
];
mpc.branch = [
1 2 0 0.1 0 100 0 0 0 0 1 -360 360;
1 2 0.01 0.2 0.04 100 0 0 1 0 0 -360 360;
];
mpc.gencost = [2 0 0 3 0.01 2 3; 2 0 0 1 0;];
end
"""


RAW = """0, 50, 33, 0, 0, 60
Paired revision-33 fixture
Two-winding ratios on both sides, with an inactive parallel line
1, 'SOURCE', 110, 3, 1, 1, 1, 1, 0, 1.1, 0.9, 1.2, 0.8
2, 'LOAD', 110, 1, 1, 1, 1, 1, 0, 1.1, 0.9, 1.2, 0.8
0 / END OF BUS DATA, BEGIN LOAD DATA
2, '1', 1, 1, 1, 10, 0, 0, 0, 0, 0, 1, 1, 0
0 / END OF LOAD DATA, BEGIN FIXED SHUNT DATA
2, '1', 1, 1, 2
2, '2', 0, 100, 100
0 / END OF FIXED SHUNT DATA, BEGIN GENERATOR DATA
1, '1', 10, 0, 50, -50, 1, 0, 25, 0.01, 0.2, 0, 0, 1, 1, 100, 50, 0, 1, 1, 0, 1
2, '2', 30, 5, 50, -50, 1, 0, 50, 0, 0.2, 0, 0, 1, 0, 100, 50, 0, 1, 1, 0, 1
0 / END OF GENERATOR DATA, BEGIN BRANCH DATA
1, 2, '1', 0, 0.1, 0.02, 100, 0, 0, 0, 0, 0, 0, 0, 1, 0
0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA
1, 2, 0, '1', 1, 1, 1, 0, 0, 2, 'TWO WINDING', 1
0.01, 0.1, 50
1.1, 110, 15, 100, 0, 0, 0, 0, 1.2, 0.8, 1.1, 0.9, 33, 0, 0, 0, 0
0.9, 110
0 / END OF TRANSFORMER DATA
Q
"""
DYR = """1 'GENCLS' '1'
4.0 0.2 /
2 'NOT_AN_ACTIVE_MODEL' '2' 123 /
"""


CGMES = """<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
 xmlns:cim="http://iec.ch/TC57/2013/CIM-schema-cim16#"
 xmlns:md="http://iec.ch/TC57/61970-552/ModelDescription/1#">
 <md:FullModel rdf:about="urn:uuid:merged-case">
  <md:Model.profile>http://entsoe.eu/CIM/EquipmentCore/3/1</md:Model.profile>
  <md:Model.profile>http://entsoe.eu/CIM/Topology/4/1</md:Model.profile>
  <md:Model.profile>http://entsoe.eu/CIM/SteadyStateHypothesis/1/1</md:Model.profile>
 </md:FullModel>
 <cim:BaseVoltage rdf:ID="base"><cim:BaseVoltage.nominalVoltage>110</cim:BaseVoltage.nominalVoltage></cim:BaseVoltage>
 <cim:TopologicalNode rdf:ID="n1"><cim:TopologicalNode.BaseVoltage rdf:resource="#base"/></cim:TopologicalNode>
 <cim:TopologicalNode rdf:ID="n2"><cim:TopologicalNode.BaseVoltage rdf:resource="#base"/></cim:TopologicalNode>
 <rdf:Description rdf:about="#line">
  <rdf:type rdf:resource="http://iec.ch/TC57/2013/CIM-schema-cim16#ACLineSegment"/>
  <cim:ACLineSegment.r>1.21</cim:ACLineSegment.r><cim:ACLineSegment.x>12.1</cim:ACLineSegment.x>
  <cim:ACLineSegment.bch>0.0002</cim:ACLineSegment.bch>
 </rdf:Description>
 <cim:Terminal rdf:ID="line1">
  <cim:Terminal.ConductingEquipment rdf:resource="#line"/>
  <cim:Terminal.TopologicalNode rdf:resource="#n1"/>
  <cim:ACDCTerminal.sequenceNumber>1</cim:ACDCTerminal.sequenceNumber>
  <cim:ACDCTerminal.connected>true</cim:ACDCTerminal.connected>
 </cim:Terminal>
 <cim:Terminal rdf:ID="line2">
  <cim:Terminal.ConductingEquipment rdf:resource="#line"/>
  <cim:Terminal.TopologicalNode rdf:resource="#n2"/>
  <cim:ACDCTerminal.sequenceNumber>2</cim:ACDCTerminal.sequenceNumber>
  <cim:ACDCTerminal.connected>true</cim:ACDCTerminal.connected>
 </cim:Terminal>
 <cim:EnergyConsumer rdf:ID="load">
  <cim:EnergyConsumer.p>20</cim:EnergyConsumer.p>
  <cim:EnergyConsumer.q>5</cim:EnergyConsumer.q>
 </cim:EnergyConsumer>
 <cim:Terminal rdf:ID="load1">
  <cim:Terminal.ConductingEquipment rdf:resource="#load"/>
  <cim:Terminal.TopologicalNode rdf:resource="#n2"/>
  <cim:ACDCTerminal.sequenceNumber>1</cim:ACDCTerminal.sequenceNumber>
  <cim:ACDCTerminal.connected>true</cim:ACDCTerminal.connected>
 </cim:Terminal>
 <cim:GeneratingUnit rdf:ID="unit">
  <cim:GeneratingUnit.minOperatingP>0</cim:GeneratingUnit.minOperatingP>
  <cim:GeneratingUnit.maxOperatingP>100</cim:GeneratingUnit.maxOperatingP>
 </cim:GeneratingUnit>
 <cim:SynchronousMachine rdf:ID="generator">
  <cim:RotatingMachine.GeneratingUnit rdf:resource="#unit"/>
  <cim:RotatingMachine.p>-20</cim:RotatingMachine.p><cim:RotatingMachine.q>-5</cim:RotatingMachine.q>
  <cim:SynchronousMachine.minQ>-50</cim:SynchronousMachine.minQ><cim:SynchronousMachine.maxQ>50</cim:SynchronousMachine.maxQ>
  <cim:SynchronousMachine.referencePriority>1</cim:SynchronousMachine.referencePriority>
  <cim:SynchronousMachine.operatingMode rdf:resource="http://iec.ch/TC57/2013/CIM-schema-cim16#SynchronousMachineOperatingMode.generator"/>
  <cim:RegulatingCondEq.RegulatingControl rdf:resource="#control"/>
  <cim:RegulatingCondEq.controlEnabled>true</cim:RegulatingCondEq.controlEnabled>
 </cim:SynchronousMachine>
 <cim:Terminal rdf:ID="gen1">
  <cim:Terminal.ConductingEquipment rdf:resource="#generator"/>
  <cim:Terminal.TopologicalNode rdf:resource="#n1"/>
  <cim:ACDCTerminal.sequenceNumber>1</cim:ACDCTerminal.sequenceNumber>
  <cim:ACDCTerminal.connected>true</cim:ACDCTerminal.connected>
 </cim:Terminal>
 <cim:RegulatingControl rdf:ID="control">
  <cim:RegulatingControl.Terminal rdf:resource="#gen1"/>
  <cim:RegulatingControl.mode rdf:resource="http://iec.ch/TC57/2013/CIM-schema-cim16#RegulatingControlModeKind.voltage"/>
  <cim:RegulatingControl.targetValue>110</cim:RegulatingControl.targetValue><cim:RegulatingControl.enabled>true</cim:RegulatingControl.enabled>
 </cim:RegulatingControl>
 <cim:LinearShuntCompensator rdf:ID="shunt">
  <cim:LinearShuntCompensator.bPerSection>0.0001</cim:LinearShuntCompensator.bPerSection>
  <cim:ShuntCompensator.nomU>110</cim:ShuntCompensator.nomU>
  <cim:ShuntCompensator.sections>2</cim:ShuntCompensator.sections><cim:ShuntCompensator.maximumSections>4</cim:ShuntCompensator.maximumSections>
 </cim:LinearShuntCompensator>
 <cim:Terminal rdf:ID="shunt1">
  <cim:Terminal.ConductingEquipment rdf:resource="#shunt"/>
  <cim:Terminal.TopologicalNode rdf:resource="#n2"/>
  <cim:ACDCTerminal.sequenceNumber>1</cim:ACDCTerminal.sequenceNumber>
  <cim:ACDCTerminal.connected>true</cim:ACDCTerminal.connected>
 </cim:Terminal>
</rdf:RDF>
"""


def test_matpower_analytic_power_flow_and_mw_cost():
    adapted = parse_matpower(MATPOWER)
    result = solve_power_flow(adapted.network, study=adapted.study)
    assert bool(result.converged)
    # Lossless receiving-bus solution: Im(V2)=-PX; |V2|²=Re(V2).
    px = (10 / 50) * 0.1
    expected = (1 + sqrt(1 - 4 * px**2)) / 2 - 1j * px
    np.testing.assert_allclose(result.voltage, [1, expected], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(result.generator_power[1], 0, atol=1e-7)
    c2, c1, c0 = adapted.network.generators[0].cost
    assert c2 * 0.3**2 + c1 * 0.3 + c0 == pytest.approx(0.01 * 15**2 + 2 * 15 + 3)
    assert adapted.report.valid
    assert adapted.report.status == AdapterStatus.DECLARED_LOSS


def test_generator_voltage_changes_study_without_changing_passive_network():
    original = parse_matpower(MATPOWER)
    changed = parse_matpower(
        MATPOWER.replace(
            "1 10 0 50 -50 1 50 1 50 0",
            "1 10 0 50 -50 1.04 50 1 50 0",
        )
    )
    original_compiled = compile_network(original.network, original.study)
    changed_compiled = compile_network(changed.network, changed.study)
    np.testing.assert_allclose(original_compiled.ybus, changed_compiled.ybus)
    result = solve_power_flow(changed.network, study=changed.study)
    assert bool(result.converged)
    assert abs(complex(result.voltage[0])) == pytest.approx(1.04, abs=2e-6)


def test_same_bus_generator_voltage_disagreement_is_not_lost_in_study_mapping():
    source = MATPOWER.replace(
        "2 30 5 50 -50 1 50 0 50 0",
        "1 30 5 50 -50 1.02 50 1 50 0",
    )
    with pytest.raises(PowerImportError) as failure:
        parse_matpower(source)
    assert not failure.value.report.valid


def test_matpower_tap_phase_shunt_and_offline_branch_stamp():
    source = MATPOWER.replace("2 1 10 0 0 0", "2 1 10 0 1 2").replace(
        "1 2 0 0.1 0 100 0 0 0 0 1", "1 2 0 0.1 0.02 100 0 0 1.05 10 1"
    )
    adapted = parse_matpower(source)
    network = compile_network(adapted.network, adapted.study)
    series = 1 / (0.1j)
    tap = 1.05 * np.exp(10j * pi / 180)
    expected = np.array(
        [
            [(series + 0.01j) / abs(tap) ** 2, -series / np.conj(tap)],
            [-series / tap, series + 0.01j + (1 + 2j) / 50],
        ]
    )
    np.testing.assert_allclose(network.ybus, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(network.branch_admittance[1], 0)


def test_matpower_executable_rejected_without_side_effect(tmp_path):
    marker = tmp_path / "must-not-exist"
    executable = MATPOWER.replace("end\n", f"system('touch {marker}');\nend\n")
    with pytest.raises(PowerImportError) as failure:
        parse_matpower(executable)
    assert failure.value.status == AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    assert not failure.value.report.valid
    assert not marker.exists()


@pytest.mark.parametrize(
    "source",
    [
        MATPOWER.replace("mpc.baseMVA = 50;", "mpc.baseMVA = 25 * 2;"),
        MATPOWER.replace("2 0 0 3 0.01 2 3", "1 0 0 3 0.01 2 3"),
        MATPOWER.replace("1 -360 360", "1 -30 30"),
    ],
)
def test_matpower_rejects_expressions_and_unsupported_active_semantics(source):
    with pytest.raises(PowerImportError) as failure:
        parse_matpower(source)
    assert not failure.value.report.valid


def test_raw_dyr_machine_base_status_and_two_sided_transformer():
    adapted = parse_psse(RAW, DYR)
    network = compile_network(adapted.network, adapted.study)
    # Original two-sided ideal winding ratios, not just a normalized field copy.
    z = 0.01 + 0.1j
    a = 1.1 * np.exp(15j * pi / 180)
    b = 0.9
    expected = np.array(
        [
            [1 / (z * abs(a) ** 2), -1 / (z * np.conj(a) * b)],
            [-1 / (z * a * b), 1 / (z * b**2) + (1 + 2j) / 50],
        ]
    )
    np.testing.assert_allclose(network.ybus, expected, rtol=3e-6, atol=3e-6)
    assert adapted.network.generators[0].p == pytest.approx(0.2)
    assert not adapted.network.generators[1].in_service
    (machine,) = adapted.dynamics
    # H * machine MVA is the kinetic-energy coefficient, not H * system MVA.
    assert machine.inertia * machine.base_mva == pytest.approx(100)
    assert (
        machine.xd_prime * adapted.network.base_mva / machine.base_mva
        == pytest.approx(0.4)
    )
    assert any(
        issue.path == "/DYR/1" and issue.category == "dropped"
        for issue in adapted.report.losses
    )


@pytest.mark.parametrize(
    "raw,dyr",
    [
        (RAW, DYR.replace("GENCLS", "GENROU")),
        (RAW, ""),
        (RAW.replace("0, 50, 33", "0, 50, 35"), DYR),
        (RAW.replace("10, 0, 0, 0, 0, 0, 1, 1, 0", "10, 0, 1, 0, 0, 0, 1, 1, 0"), DYR),
        (RAW, DYR.replace("4.0 0.2", "0.0 0.2")),
    ],
)
def test_raw_dyr_rejects_unknown_missing_and_unrepresentable_models(raw, dyr):
    with pytest.raises(PowerImportError) as failure:
        parse_psse(raw, dyr)
    assert not failure.value.report.valid


@pytest.mark.parametrize(
    "record",
    [
        "system('untrusted')",
        "1, 1, 50, 1, 'ACTIVE AREA CONTROL'",
    ],
)
def test_raw_tail_cannot_hide_executable_or_active_control_semantics(record):
    source = RAW.replace("Q\n", f"{record}\n0 / END OF AREA DATA\nQ\n")
    with pytest.raises(PowerImportError) as failure:
        parse_psse(source, DYR)
    assert not failure.value.report.valid


def test_cgmes_real_rdf_resource_links_units_and_inward_signs():
    adapted = parse_cgmes(CGMES)
    network = compile_network(adapted.network, adapted.study)
    series = 1 / (0.01 + 0.1j)
    expected = np.array(
        [
            [series + 0.0121j, -series],
            [-series, series + 0.0121j + 0.0242j],
        ]
    )
    np.testing.assert_allclose(network.ybus, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        network.specified_power, [0.2 + 0.05j, -0.2 - 0.05j], atol=2e-7
    )
    assert adapted.study.controls[0].kind == "reference"
    assert adapted.network.frequency == 50
    assert adapted.report.valid


def test_cgmes_disconnected_load_does_not_inject():
    terminal = """<cim:Terminal rdf:ID="load1">
  <cim:Terminal.ConductingEquipment rdf:resource="#load"/>
  <cim:Terminal.TopologicalNode rdf:resource="#n2"/>
  <cim:ACDCTerminal.sequenceNumber>1</cim:ACDCTerminal.sequenceNumber>
  <cim:ACDCTerminal.connected>true</cim:ACDCTerminal.connected>
 </cim:Terminal>"""
    source = CGMES.replace(terminal, terminal.replace(">true<", ">false<"))
    adapted = parse_cgmes(source)
    compiled = compile_network(adapted.network, adapted.study)
    np.testing.assert_allclose(compiled.specified_power[1], 0)


@pytest.mark.parametrize(
    "source",
    [
        CGMES.replace(
            "<rdf:RDF",
            '<!DOCTYPE rdf:RDF [<!ENTITY x SYSTEM "file:///etc/passwd">]><rdf:RDF',
            1,
        ),
        CGMES.replace("CIM-schema-cim16", "CIM-schema-cim17"),
        CGMES.replace("</rdf:RDF>", '<cim:Breaker rdf:ID="unknown"/></rdf:RDF>'),
        CGMES.replace('rdf:resource="#n2"', 'rdf:resource="#missing"'),
        CGMES.replace("EquipmentCore/3/1", "EquipmentCore/9/9"),
    ],
)
def test_cgmes_rejects_entities_unknown_semantics_and_dangling_links(source):
    with pytest.raises(PowerImportError) as failure:
        parse_cgmes(source)
    assert not failure.value.report.valid


def test_parser_resource_budgets_fail_closed():
    for limits in (
        PowerParserLimits(max_bytes=32),
        PowerParserLimits(max_rows=1),
        PowerParserLimits(max_tokens=5),
        PowerParserLimits(max_token_chars=3),
    ):
        with pytest.raises(PowerImportError) as failure:
            parse_matpower(MATPOWER, limits=limits)
        assert failure.value.status == AdapterStatus.MALFORMED_SOURCE
    with pytest.raises(PowerImportError):
        parse_cgmes(CGMES, limits=PowerParserLimits(max_xml_depth=2))
