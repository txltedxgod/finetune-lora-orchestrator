from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from lora.pipeline import LoRAPipeline

app = FastAPI(title="LoRA Fine-Tune Orchestrator", version="0.1.0")
lora = LoRAPipeline()

class DatasetReq(BaseModel):
    samples: List[Dict[str, str]]

@app.post("/api/v1/validate-dataset")
def validate(req: DatasetReq):
    return lora.validate_dataset(req.samples)
