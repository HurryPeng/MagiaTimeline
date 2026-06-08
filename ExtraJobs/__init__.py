import IR
from ExtraJobs.TextExtraction import IIRTextExtractionPass
from ExtraJobs.StyleClassification import IIRStyleClassificationPass

_REGISTRY = {
    "ocr": IIRTextExtractionPass,
    "sty": IIRStyleClassificationPass,
}

def createExtraJob(name: str, config: dict, frameKey, dest: str = "") -> IR.IIRPass:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown extra job '{name}'. Available: {list(_REGISTRY.keys())}")
    if name == "ocr":
        return cls(config, frameKey, dest)
    return cls(config, frameKey)
