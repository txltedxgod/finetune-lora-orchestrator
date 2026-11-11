from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from lora.pipeline import LoRAPipeline

app = FastAPI(title="LoRA Fine-Tune Orchestrator", version="0.1.0")
lora = LoRAPipeline()


class DatasetReq(BaseModel):
    samples: list[dict[str, str]] = Field(default_factory=list)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/validate-dataset")
def validate(req: DatasetReq) -> dict:
    return lora.validate_dataset(req.samples)
