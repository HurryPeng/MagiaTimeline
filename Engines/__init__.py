from Engines.AbstractEngine import AbstractEngine
from Engines.FramewiseEngine import FramewiseEngine
from Engines.SpeculativeEngine import SpeculativeEngine

_REGISTRY = {
    "speculative": SpeculativeEngine,
    "framewise": FramewiseEngine,
}

def createEngine(name: str, scaleDown: int, config: dict) -> AbstractEngine:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown engine '{name}'. Available: {list(_REGISTRY.keys())}")
    return cls(scaleDown, config)
