from __future__ import annotations

from typing import Any


class LoRAPipeline:
    """Dataset readiness validator for supervised LoRA fine-tuning."""

    def validate_dataset(self, samples: list[dict[str, str]]) -> dict[str, Any]:
        if not samples:
            return {
                "total_samples": 0,
                "valid_samples": 0,
                "invalid_indices": [],
                "ready": False,
                "readiness": "Empty dataset",
            }

        valid_count = 0
        invalid_indices: list[int] = []
        for idx, sample in enumerate(samples):
            instruction = sample.get("instruction", "")
            output = sample.get("output", "")
            if (
                isinstance(instruction, str)
                and isinstance(output, str)
                and instruction.strip()
                and output.strip()
            ):
                valid_count += 1
            else:
                invalid_indices.append(idx)

        ready = valid_count == len(samples)
        return {
            "total_samples": len(samples),
            "valid_samples": valid_count,
            "invalid_indices": invalid_indices,
            "ready": ready,
            "readiness": (
                "100% Ready for LoRA fine-tuning" if ready else "Warning: missing fields"
            ),
        }
