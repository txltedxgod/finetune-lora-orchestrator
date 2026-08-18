from lora.pipeline import LoRAPipeline

def test_lora_dataset():
    p = LoRAPipeline()
    res = p.validate_dataset([{"instruction": "Sum", "output": "42"}])
    assert res["valid_samples"] == 1
