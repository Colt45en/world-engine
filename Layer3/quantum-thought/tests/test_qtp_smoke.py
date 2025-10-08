import pytest

qtp = pytest.importorskip("qtp")


def test_qtp_metadata_and_accessors():
    assert qtp.__version__ == "0.2.0"

    pipeline = qtp.QuantumThoughtPipeline()
    zones = pipeline.get_zones()
    assert isinstance(zones, list)
    assert zones, "Expected at least one ThoughtZone"

    zone0 = zones[0]
    assert isinstance(zone0.position, tuple)
    assert len(zone0.position) == 3
    pos_np = zone0.position_np
    assert pos_np.shape == (3,)
    assert pos_np.dtype.kind == "f"

    agent = pipeline.get_agent()
    assert isinstance(agent.position, tuple)
    assert len(agent.position) == 3
    agent_np = agent.position_np
    assert agent_np.shape == (3,)
    assert agent_np.dtype.kind == "f"
