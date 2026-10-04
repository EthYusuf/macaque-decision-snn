"""Biyolojik yöntem: ödülle modüle edilen STDP (R-STDP, "üç faktörlü kural").

Gerçek beyinde öğrenme yerel ve gecikmelidir:
  1. faktör: önceki nöronun spike'ı
  2. faktör: sonraki nöronun spike'ı
     STDP: önce -> sonra sırası (nedensel) sinapsı güçlendirir (LTP),
     sonra -> önce sırası zayıflatır (LTD). Bu hemen ağırlığı DEĞİŞTİRMEZ,
     sinapsta bir "uygunluk izi" (eligibility trace) bırakır.
  3. faktör: dopamin. Deneme sonunda maymun meyve suyu alır ya da almaz.
     Dopamin nöronları ödül tahmin hatasını kodlar (Schultz 1997):
         D = alınan ödül - beklenen ödül
     Ağırlık değişimi:  dW = eta * D * e

Beklenenden iyi sonuç (D > 0) o denemede aktif olan sinapsları güçlendirir,
kötü sonuç (D < 0) zayıflatır. Plastisite yalnızca MT -> LIP(E) sinapslarında.
Ek olarak sinaptik ölçekleme her nöronun toplam girdi ağırlığını sabit tutar;
böylece ağırlıklar sınırsız büyümez ve girdiler birbiriyle yarışır.

Kredi atama sorunu ve çözümü:
Dopamin tüm sinapslara aynı anda yayılır. Kaybeden havuz da düşük hızda
ateşlemeye devam ettiği için, ham spike'larla kazanan havuzun ödülünden o da pay
alır; sonuçta iki havuz da aynı yönü öğrenir. Çözüm, sonraki nöronun ham
spike'ı yerine ortalama aktivitesinden SAPMASINI kullanmaktır:
    post_merkezli = spike - ortalama_hız
Bu kural "EH kuralı" (Legenstein et al. 2010) olarak bilinir ve maymun motor
korteksi beyin-bilgisayar arayüzü deneyini açıklamak için önerilmiştir.
Kayan eşikli BCM kuralına benzer: nöron her zamankinden fazla ateşlediyse kredi
alır, her zamankinden az ateşlediyse suçu paylaşır.
"""

import math
import time

import torch

from ..config import Config, TrainParams
from ..network import SpikingBrain
from ..task import RandomDotTask


@torch.no_grad()
def eligibility_traces(
    pre: torch.Tensor, post: torch.Tensor, p: TrainParams, dt: float,
) -> torch.Tensor:
    """Deneme sonundaki uygunluk izleri, şekil (B, n_pre, n_post).

    pre: (T, B, n_pre), post: (T, B, n_post) spike'ları.
    LTP: sonraki nöron ateşlediğinde, önceki nöronun yakın geçmişteki spike izi.
    LTD: önceki nöron ateşlediğinde, sonraki nöronun yakın geçmişteki spike izi.
    Her katkı, ödül anına (deneme sonu) kadar tau_elig ile söner.
    """
    T, B, n_pre = pre.shape
    n_post = post.shape[2]
    a_pre = math.exp(-dt / p.tau_plus)
    a_post = math.exp(-dt / p.tau_minus)

    x_tr = torch.zeros(B, n_pre, device=pre.device)
    y_tr = torch.zeros(B, n_post, device=pre.device)
    pre_traces = torch.empty_like(pre)
    post_traces_prev = torch.empty_like(post)
    for t in range(T):
        post_traces_prev[t] = y_tr               # yalnızca kesin önceki sonraki-spike'lar
        x_tr = a_pre * x_tr + pre[t]
        y_tr = a_post * y_tr + post[t]
        pre_traces[t] = x_tr

    steps_to_reward = torch.arange(T - 1, -1, -1, device=pre.device, dtype=pre.dtype)
    decay = torch.exp(-steps_to_reward * dt / p.tau_elig)          # (T,)
    ltp = torch.einsum("t,tbi,tbj->bij", decay, pre_traces, post)
    ltd = torch.einsum("t,tbi,tbj->bij", decay, pre, post_traces_prev)
    return p.a_plus * ltp - p.a_minus * ltd


def train_rstdp(
    model: SpikingBrain, task: RandomDotTask, cfg: Config,
    generator: torch.Generator, log=print, on_step=None,
) -> list[dict]:
    p = cfg.train
    dt = cfg.task.dt
    window = int(cfg.task.decision_window / dt)
    to_hz = 1000.0 / dt
    exc = model.exc
    levels = torch.tensor(cfg.task.coherences, device=model.w_in.device)

    # "Eleştirmen": her zorluk seviyesi için beklenen ödül (başta yazı-tura = 0.5)
    expected_reward = torch.full((len(levels),), 0.5, device=model.w_in.device)
    mask = model.mask_in[:, exc]
    # Sinaptik ölçekleme hedefi: her E nöronunun başlangıçtaki toplam girdi ağırlığı
    target_sum = (model.w_in.data[:, exc] * mask).sum(dim=0, keepdim=True)
    # Her E nöronunun uzun vadeli ortalama spike olasılığı (adım başına); kayan eşik
    mean_activity = None

    history = []
    t0 = time.time()
    with torch.no_grad():
        for it in range(1, p.iters + 1):
            batch = task.sample(p.batch_size, generator)
            spikes = model(batch.x)
            choice = model.choice(spikes, window)
            reward = (choice == batch.label).float()                  # meyve suyu: 1 ya da 0

            # Dopamin = ödül tahmin hatası
            level_idx = (batch.coherence[:, None] - levels[None, :]).abs().argmin(dim=1)
            dopamine = reward - expected_reward[level_idx]
            for k in level_idx.unique():
                sel = level_idx == k
                expected_reward[k] += p.reward_avg_rate * sel.sum() * (reward[sel].mean() - expected_reward[k])
            expected_reward.clamp_(0.0, 1.0)

            # Sonraki nöron aktivitesi: ortalamadan sapma (EH kuralı)
            post = spikes[:, :, exc]
            batch_activity = post.mean(dim=(0, 1))
            if mean_activity is None:
                mean_activity = batch_activity.clone()
            mean_activity += p.activity_avg_rate * (batch_activity - mean_activity)
            post_centered = post - mean_activity

            # Üç faktörlü güncelleme: dW = eta * D * e  (denemeler üzerinden ortalama)
            elig = eligibility_traces(batch.x, post_centered, p, dt)       # (B, n_in, n_exc)
            dw = p.eta * (dopamine[:, None, None] * elig).mean(dim=0)
            w = model.w_in.data[:, exc]
            w += dw * mask
            w.clamp_(min=0.0)
            w *= target_sum / (w * mask).sum(dim=0, keepdim=True).clamp(min=1e-8)
            model.w_in.data[:, exc] = w

            neuron_hz = spikes.mean(dim=(0, 1)) * to_hz
            rate_e = neuron_hz[model.exc].mean().item()
            rate_i = neuron_hz[model.inh].mean().item()
            acc = reward.mean().item()
            history.append({
                "iter": it, "trials": it * p.batch_size, "accuracy": acc,
                "dopamine": dopamine.abs().mean().item(), "rate_e": rate_e, "rate_i": rate_i,
            })
            if on_step is not None:
                on_step(it, model)
            if it % p.log_every == 0 or it == p.iters:
                recent = history[-p.log_every:]
                mean_acc = sum(h["accuracy"] for h in recent) / len(recent)
                elapsed = time.time() - t0
                eta = elapsed / it * (p.iters - it)
                log(f"  [BIO] adım {it:4d}/{p.iters}  |dopamin| {history[-1]['dopamine']:.2f}  "
                    f"doğruluk {mean_acc:5.1%}  E {rate_e:4.1f} Hz  I {rate_i:4.1f} Hz  "
                    f"(kalan ~{eta:4.0f} sn)")
    return history
