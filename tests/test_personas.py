from cryptoquant.config import load_config
from cryptoquant.personas import run_persona


def test_protocol_persona_report() -> None:
    config = load_config("configs/default.yaml")
    report = run_persona(config, "protocol")
    assert "microstructure" in report
    assert "agent_simulation" in report
