from typing import Dict, Any, List

class LoRAPipeline:
    def validate_dataset(self, samples: List[Dict[str, str]]) -> Dict[str, Any]:
        valid_count = 0
        for s in samples:
            if "instruction" in s and "output" in s and len(s["output"]) > 0:
                valid_count += 1
        return {
            "total_samples": len(samples),
            "valid_samples": valid_count,
            "readiness": "100% Ready for LoRA fine-tuning" if valid_count == len(samples) else "Warning: missing fields"
        }
