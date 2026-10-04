import torch
import torch.nn.functional as F

from monkeybrain.network import SpikingBrain
from monkeybrain.task import RandomDotTask


def test_dale_law_signs(small_cfg):
    m = SpikingBrain(small_cfg.net, small_cfg.task)
    w_in, w_rec = m.effective_weights()
    ne = small_cfg.net.n_exc
    assert torch.all(w_in >= 0)
    assert torch.all(w_rec[:ne] >= 0)      # E nöronları yalnızca uyarır
    assert torch.all(w_rec[ne:] <= 0)      # I nöronları yalnızca baskılar
    assert torch.all(torch.diagonal(w_rec) == 0)


def test_dale_law_survives_training_step(small_cfg):
    task = RandomDotTask(small_cfg.task)
    m = SpikingBrain(small_cfg.net, small_cfg.task)
    opt = torch.optim.SGD(m.parameters(), lr=100.0)  # bilerek büyük adım
    b = task.sample(4, torch.Generator().manual_seed(0))
    spikes = m(b.x)
    loss = F.cross_entropy(m.pool_rates(spikes, 50) / 5.0, b.label)
    loss.backward()
    assert m.w_in.grad is not None and m.w_in.grad.abs().sum() > 0
    opt.step()
    m.apply_constraints()
    w_in, w_rec = m.effective_weights()
    ne = small_cfg.net.n_exc
    assert torch.all(w_in >= 0)
    assert torch.all(w_rec[:ne] >= 0) and torch.all(w_rec[ne:] <= 0)


def test_seed_reproducibility(small_cfg):
    def run():
        torch.manual_seed(7)
        task = RandomDotTask(small_cfg.task)
        m = SpikingBrain(small_cfg.net, small_cfg.task, seed=3)
        b = task.sample(2, torch.Generator().manual_seed(5))
        return m(b.x)

    assert torch.equal(run(), run())


def test_firing_rates_are_biological(small_cfg):
    torch.manual_seed(0)
    cfg = small_cfg
    task = RandomDotTask(cfg.task)
    m = SpikingBrain(cfg.net, cfg.task)
    with torch.no_grad():
        spikes = m(task.sample(16, torch.Generator().manual_seed(0)).x)
    hz = spikes.mean(dim=(0, 1)) * 1000 / cfg.task.dt
    assert 0.5 < hz[: cfg.net.n_exc].mean() < 30
    assert hz.max() < 200  # refrakter dönem sayesinde
