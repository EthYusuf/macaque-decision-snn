import numpy as np
import torch

from monkeybrain.neurons import IZHIKEVICH_TYPES, simulate_alif, simulate_izhikevich, spike_fn


def count_spikes(v):
    return int(np.sum((v[1:] >= 29.9) & (v[:-1] < 29.9)))


def test_izhikevich_fs_fires_faster_than_rs():
    rs = simulate_izhikevich(*IZHIKEVICH_TYPES["RS"][0])[1]
    fs = simulate_izhikevich(*IZHIKEVICH_TYPES["FS"][0])[1]
    assert count_spikes(fs) > 2 * count_spikes(rs) > 0


def test_izhikevich_silent_without_current():
    _, v, _ = simulate_izhikevich(*IZHIKEVICH_TYPES["RS"][0], current=0.0)
    assert count_spikes(v) == 0


def test_alif_adapts_and_respects_refractory():
    t, v, _ = simulate_alif(current=3.0, refractory_ms=2.0)
    spike_times = t[v >= 29.9]
    isi = np.diff(spike_times)
    assert len(isi) > 3
    assert isi.min() > 2.0                 # refrakter dönem: iki spike arası > 2 ms
    assert isi[-1] > isi[0]                # adaptasyon: aralıklar uzar


def test_surrogate_gradient_is_nonzero_near_threshold():
    x = torch.tensor([-0.5, -0.05, 0.05, 0.5], requires_grad=True)
    s = spike_fn(x, slope=10.0)
    assert s.tolist() == [0.0, 0.0, 1.0, 1.0]
    s.sum().backward()
    assert torch.all(x.grad > 0)
    assert x.grad[1] > x.grad[0]           # eşiğe yakın gradyan daha büyük
