from Strategies.AbstractStrategy import (
    AbstractStrategy,
    AbstractFramewiseStrategy,
    AbstractSpeculativeStrategy,
    AbstractExtraJobStrategy
)
from Strategies.MagirecoStrategy import MagirecoStrategy
from Strategies.MagirecoScene0Strategy import MagirecoScene0Strategy
from Strategies.MadodoraStrategy import MadodoraStrategy
from Strategies.LimbusCompanyStrategy import LimbusCompanyStrategy
from Strategies.LimbusCompanyMechanicsStrategy import LimbusCompanyMechanicsStrategy
from Strategies.PokemonEmeraldStrategy import PokemonEmeraldStrategy
from Strategies.ParakoStrategy import ParakoStrategy
from Strategies.BanGDreamStrategy import BanGDreamStrategy
from Strategies.OutlineStrategy import OutlineStrategy
from Strategies.BoxColourStatStrategy import BoxColourStatStrategy
from Strategies.DiffTextDetectionStrategy import DiffTextDetectionStrategy

_REGISTRY = {
    "mr": MagirecoStrategy,
    "mr-s0": MagirecoScene0Strategy,
    "md": MadodoraStrategy,
    "lcb": LimbusCompanyStrategy,
    "lcb-mech": LimbusCompanyMechanicsStrategy,
    "pkm": PokemonEmeraldStrategy,
    "prk": ParakoStrategy,
    "bdr": BanGDreamStrategy,
    "otl": OutlineStrategy,
    "bcs": BoxColourStatStrategy,
    "dtd": DiffTextDetectionStrategy,
}

def createStrategy(name: str, config: dict, contentRect) -> AbstractStrategy:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(_REGISTRY.keys())}")
    return cls(config, contentRect)
