import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monkeybrain.config import Config  # noqa: E402


@pytest.fixture
def small_cfg() -> Config:
    """Testler için küçük ve hızlı bir ağ/görev."""
    cfg = Config()
    cfg.task.t_fix = 50
    cfg.task.t_stim = 150
    cfg.task.decision_window = 100
    cfg.task.n_mt = 16
    cfg.net.n_exc = 40
    cfg.net.n_inh = 10
    cfg.net.pool_size = 10
    cfg.train.batch_size = 4
    return cfg
