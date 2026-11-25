from ._base import OBJECTIVE_HOOKS
from .byol.byol import BYOLObjective
from .contrastive.contrastive import ContrastiveObjective
from .dino.dino import DINOObjective
from .hubert.hubert import HuBERTObjective
from .jepa.jepa import JEPAObjective
from .mae.mae import MAEObjective
from .objective import Objective, build_objective
from .ufo.ufo import UFOObjective
from .wav2vec2.wav2vec2 import Wav2Vec2Objective

__all__ = [
    "BYOLObjective",
    "ContrastiveObjective",
    "DINOObjective",
    "HuBERTObjective",
    "JEPAObjective",
    "MAEObjective",
    "OBJECTIVE_HOOKS",
    "Objective",
    "UFOObjective",
    "Wav2Vec2Objective",
    "build_objective",
]
