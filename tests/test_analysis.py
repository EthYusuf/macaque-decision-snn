import numpy as np

from monkeybrain import analysis
from monkeybrain.config import Config


def test_weibull_is_chance_at_zero_and_perfect_at_high_coherence():
    assert analysis.weibull(0.0, 0.1, 1.5) == 0.5
    assert analysis.weibull(1.0, 0.1, 1.5) > 0.999


def test_fit_weibull_recovers_known_parameters():
    levels = np.array([0.0, 0.032, 0.064, 0.128, 0.256, 0.512])
    acc = analysis.weibull(levels, 0.12, 1.4)
    fit = analysis.fit_weibull(levels, acc)
    assert abs(fit["alpha"] - 0.12) < 1e-3
    assert abs(fit["beta"] - 1.4) < 1e-2


def _fake_result(cfg, rng):
    """Kolay koşulda hızlı, zor koşulda yavaş tırmanan sentetik havuz izleri."""
    T = cfg.task.n_steps
    onset = cfg.task.stim_onset_step
    cohs = np.repeat([0.0, 0.512], 50)
    t = np.clip(np.arange(T) - onset, 0, None)
    traces = np.zeros((len(cohs), T, 2))
    for i, c in enumerate(cohs):
        slope = (0.02 + 0.2 * c) * rng.uniform(0.5, 1.5)
        traces[i, :, 0] = 2 + slope * t
        traces[i, :, 1] = 2.0
    return {"coherence": cohs, "traces": traces}


def test_decision_threshold_is_reached_in_most_hard_trials():
    cfg = Config()
    res = _fake_result(cfg, np.random.default_rng(0))
    thr = analysis.decision_threshold(res, cfg)
    rts = analysis.decision_times(res, cfg, thr)
    hard = res["coherence"] == 0.0
    assert np.mean(~np.isnan(rts[hard])) >= analysis.MIN_CROSSING
    assert np.nanmean(rts[~hard]) < np.nanmean(rts[hard])   # kolay karar daha hızlı
