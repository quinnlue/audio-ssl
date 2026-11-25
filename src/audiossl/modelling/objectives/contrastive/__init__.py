from .loss import InfoNCELossOutput, contrastive_info_nce_loss
from .projection import ProjectionHead
from .views import create_contrastive_views

__all__ = [
    "ContrastiveObjective",
    "InfoNCELossOutput",
    "ProjectionHead",
    "contrastive_info_nce_loss",
    "create_contrastive_views",
]


def __getattr__(name: str):
    if name == "ContrastiveObjective":
        from .contrastive import ContrastiveObjective

        return ContrastiveObjective
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
