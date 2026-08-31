import jax.numpy as jnp
import pytest

from phydrax.circuit import (
    ElectricalWaveReference,
    full_scattering_matrix,
    InstancePort,
    MatrixScatteringComponent,
    prepare_scattering_network,
    scattering_submatrix,
    ScatteringInstance,
    ScatteringNetwork,
    ScatteringNetworkPolicy,
    ScatteringNetworkStatus,
    solve_scattering_network,
    WaveConnection,
    WavePort,
    WaveProbe,
)


def _component(matrix, name):
    reference = ElectricalWaveReference(50.0)
    count = int(jnp.asarray(matrix).shape[-1])
    return MatrixScatteringComponent(
        jnp.asarray(matrix, dtype=jnp.complex128),
        tuple(WavePort(f"p{index + 1}", reference) for index in range(count)),
        component_id=name,
    )


def test_selected_full_parity_and_noninvasive_hierarchical_probe():
    through = _component([[0.0, 1.0], [1.0, 0.0]], "through")
    inner = ScatteringNetwork(
        (ScatteringInstance("a", through), ScatteringInstance("b", through)),
        (WaveConnection(InstancePort("a", "p2"), InstancePort("b", "p1")),),
        (InstancePort("a", "p1"), InstancePort("b", "p2")),
        external_port_ids=("left", "right"),
        probes=(WaveProbe("middle", InstancePort("a", "p2")),),
        network_id="inner",
    )
    outer = ScatteringNetwork(
        (ScatteringInstance("nested", inner),),
        (),
        (InstancePort("nested", "left"), InstancePort("nested", "right")),
        external_port_ids=("in", "out"),
        network_id="outer",
    )
    prepared = prepare_scattering_network(outer, jnp.asarray(2.0))
    full = full_scattering_matrix(prepared)
    selected = scattering_submatrix(prepared, ("in",), ("out",))
    assert jnp.allclose(selected, full[1:2, 0:1])
    result = solve_scattering_network(prepared, jnp.asarray([[1.0], [0.0]]))
    assert bool(result.diagnostics.successful)
    assert result.probe_ids == ("nested/middle",)
    assert jnp.allclose(result.external_outgoing[:, 0], jnp.asarray([0.0, 1.0]))


def test_unit_gain_closed_loop_reports_singular_without_regularization():
    reflector = _component([[1.0]], "reflector")
    matched = _component([[0.0]], "matched")
    network = ScatteringNetwork(
        (
            ScatteringInstance("r1", reflector),
            ScatteringInstance("r2", reflector),
            ScatteringInstance("external", matched),
        ),
        (WaveConnection(InstancePort("r1", "p1"), InstancePort("r2", "p1")),),
        (InstancePort("external", "p1"),),
        external_port_ids=("port",),
        network_id="singular-loop",
    )
    prepared = prepare_scattering_network(network, jnp.asarray(1.0))
    result = solve_scattering_network(prepared, jnp.asarray([[1.0]]))
    assert int(result.diagnostics.status) == int(ScatteringNetworkStatus.SINGULAR)


def test_scattering_rhs_resource_envelope_is_enforced_before_allocation():
    through = _component([[0.0, 1.0], [1.0, 0.0]], "through")
    network = ScatteringNetwork(
        (ScatteringInstance("device", through),),
        (),
        (InstancePort("device", "p1"), InstancePort("device", "p2")),
        external_port_ids=("left", "right"),
        network_id="rhs-budget",
    )
    prepared = prepare_scattering_network(
        network,
        jnp.asarray(1.0),
        ScatteringNetworkPolicy(maximum_rhs_bytes=192),
    )
    with pytest.raises(MemoryError, match="maximum_rhs_bytes"):
        solve_scattering_network(prepared, jnp.ones((2, 3)))
