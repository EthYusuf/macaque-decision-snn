import torch

from monkeybrain.config import TrainParams
from monkeybrain.learning.rstdp_trainer import eligibility_traces


def _pair(pre_times, post_times, T=200):
    pre = torch.zeros(T, 1, 1)
    post = torch.zeros(T, 1, 1)
    pre[pre_times, 0, 0] = 1
    post[post_times, 0, 0] = 1
    return pre, post


def test_causal_pairing_gives_positive_eligibility():
    p = TrainParams()
    pre, post = _pair([100], [105])          # önce -> sonra (nedensel)
    assert eligibility_traces(pre, post, p, 1.0).item() > 0


def test_anticausal_pairing_gives_negative_eligibility():
    p = TrainParams()
    pre, post = _pair([105], [100])          # sonra -> önce
    assert eligibility_traces(pre, post, p, 1.0).item() < 0


def test_eligibility_decays_with_time_to_reward():
    p = TrainParams()
    early = eligibility_traces(*_pair([10], [15]), p, 1.0).item()
    late = eligibility_traces(*_pair([180], [185]), p, 1.0).item()
    assert 0 < early < late                   # ödüle yakın olay daha güçlü iz bırakır


def test_three_factor_rule_sign():
    """Pozitif dopamin nedensel sinapsı güçlendirir, negatif zayıflatır."""
    p = TrainParams()
    e = eligibility_traces(*_pair([100], [105]), p, 1.0)
    for dopamine, sign in ((1.0, 1), (-1.0, -1)):
        dw = p.eta * dopamine * e
        assert torch.sign(dw).item() == sign
