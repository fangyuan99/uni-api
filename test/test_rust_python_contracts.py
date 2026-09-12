import json
from pathlib import Path


def test_rust_python_runtime_contract_fixture_is_shared():
    fixture = json.loads((Path(__file__).parents[1] / "test" / "runtime_contracts.fixture").read_text())
    assert "/v1/responses" in fixture["routes"]
    assert "/v1/video/tasks" in fixture["routes"]
    assert {"tools", "image", "project_id"}.issubset(fixture["provider_fields"])
    assert fixture["nested_key"]["rule"] == "child-key/*"
