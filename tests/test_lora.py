from fastapi.testclient import TestClient

from lora.api import app
from lora.pipeline import LoRAPipeline

client = TestClient(app)


def test_lora_dataset() -> None:
    p = LoRAPipeline()
    res = p.validate_dataset([{"instruction": "Sum", "output": "42"}])
    assert res["valid_samples"] == 1
    assert res["ready"] is True


def test_lora_empty_and_invalid_samples() -> None:
    p = LoRAPipeline()
    empty = p.validate_dataset([])
    assert empty["ready"] is False
    assert empty["readiness"] == "Empty dataset"

    mixed = p.validate_dataset([
        {"instruction": "Valid", "output": "OK"},
        {"instruction": "   ", "output": "Missing instruction"},
        {"instruction": "Missing output", "output": ""},
    ])
    assert mixed["total_samples"] == 3
    assert mixed["valid_samples"] == 1
    assert mixed["invalid_indices"] == [1, 2]
    assert mixed["ready"] is False


def test_lora_api_endpoints() -> None:
    assert client.get("/health").json() == {"status": "ok"}

    ok = client.post(
        "/api/v1/validate-dataset",
        json={"samples": [{"instruction": "Summarize", "output": "Done"}]},
    )
    assert ok.status_code == 200
    assert ok.json()["ready"] is True

    empty = client.post("/api/v1/validate-dataset", json={"samples": []})
    assert empty.status_code == 200
    assert empty.json()["ready"] is False
