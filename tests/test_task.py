import torch

from monkeybrain.task import RandomDotTask


def test_shapes_and_labels(small_cfg):
    task = RandomDotTask(small_cfg.task)
    b = task.sample(8, torch.Generator().manual_seed(0))
    assert b.x.shape == (small_cfg.task.n_steps, 8, small_cfg.task.n_mt)
    assert set(b.x.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(b.label, (b.direction > 0).long())


def test_rightward_motion_drives_right_preferring_mt(small_cfg):
    small_cfg.task.stim_noise = 0.0
    task = RandomDotTask(small_cfg.task)
    n = 16
    rates = task.rates(torch.full((n,), 0.5), torch.ones(n))     # sağa hareket
    stim = rates[small_cfg.task.n_steps - 1]                      # son adım (uyaran açık)
    right_pref = task.cos_pref > 0.5
    left_pref = task.cos_pref < -0.5
    assert stim[:, right_pref].mean() > stim[:, left_pref].mean() + 10


def test_zero_coherence_has_no_mean_bias(small_cfg):
    task = RandomDotTask(small_cfg.task)
    g = torch.Generator().manual_seed(1)
    n = 2000
    rates = task.rates(torch.zeros(n), torch.ones(n), g)
    right_pref = task.cos_pref > 0.5
    left_pref = task.cos_pref < -0.5
    diff = rates[-1][:, right_pref].mean() - rates[-1][:, left_pref].mean()
    assert abs(diff) < 1.0  # Hz


def test_fixation_period_has_no_motion_signal(small_cfg):
    task = RandomDotTask(small_cfg.task)
    rates = task.rates(torch.full((4,), 0.512), torch.ones(4), torch.Generator().manual_seed(2))
    fix = rates[: small_cfg.task.stim_onset_step]
    assert torch.allclose(fix, torch.full_like(fix, small_cfg.task.mt_spontaneous))


def test_motion_onset_raises_mt_rate_even_at_zero_coherence(small_cfg):
    small_cfg.task.stim_noise = 0.0
    task = RandomDotTask(small_cfg.task)
    rates = task.rates(torch.zeros(4), torch.ones(4))
    assert torch.allclose(rates[-1], torch.full_like(rates[-1], small_cfg.task.mt_baseline))
    assert small_cfg.task.mt_baseline > small_cfg.task.mt_spontaneous
