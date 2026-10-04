"""İki öğrenme yöntemi: aynı ağ, aynı görev, farklı öğrenme kuralı."""

from .rstdp_trainer import train_rstdp
from .surrogate_trainer import train_surrogate

TRAINERS = {"ml": train_surrogate, "bio": train_rstdp}

__all__ = ["TRAINERS", "train_rstdp", "train_surrogate"]
