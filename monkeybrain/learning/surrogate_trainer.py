"""ML yöntemi: surrogate gradient + zamanda geri yayılım (BPTT).

Fikir: Ağı bir deneme boyunca (700 adım) simüle et, kararı bir kayıp fonksiyonuyla
ölç, sonra hatayı zamanda geriye doğru yay ve TÜM ağırlıkları (MT -> LIP ve LIP
içi) güncelle. Spike türevlenemediği için geri yönde SurrogateSpike kullanılır.

Beyin bu yöntemi büyük olasılıkla birebir kullanmaz: bir sinapsın, yüzlerce ms
önceki başka nöronların hatasını "bilmesi" gerekir. Ama ne öğrenilebileceğinin
üst sınırını gösterir.
"""

import time

import torch
import torch.nn.functional as F

from ..config import Config
from ..network import SpikingBrain
from ..task import RandomDotTask


def train_surrogate(
    model: SpikingBrain, task: RandomDotTask, cfg: Config,
    generator: torch.Generator, log=print, on_step=None,
) -> list[dict]:
    p = cfg.train
    dt = cfg.task.dt
    window = int(cfg.task.decision_window / dt)
    to_hz = 1000.0 / dt
    opt = torch.optim.Adam(model.parameters(), lr=p.lr)
    history = []
    t0 = time.time()

    for it in range(1, p.iters + 1):
        batch = task.sample(p.batch_size, generator)
        spikes = model(batch.x)                                   # (T, B, n)

        # 1) Görev kaybı: havuz hızları -> logit -> çapraz entropi
        rates = model.pool_rates(spikes, window)                  # (B, 2) Hz
        task_loss = F.cross_entropy(rates / p.logit_temperature, batch.label)

        # 2) Biyolojik kısıt: her nöronun ortalama hızı [rate_min, rate_max] Hz içinde kalsın
        neuron_hz = spikes.mean(dim=(0, 1)) * to_hz
        reg = (F.relu(neuron_hz - p.rate_max) ** 2).mean() + (F.relu(p.rate_min - neuron_hz) ** 2).mean()
        loss = task_loss + p.rate_reg * reg

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p.grad_clip)
        opt.step()
        model.apply_constraints()                                 # Dale yasası

        with torch.no_grad():
            choice = model.choice(spikes, window)
            acc = (choice == batch.label).float().mean().item()
            rate_e = neuron_hz[model.exc].mean().item()
            rate_i = neuron_hz[model.inh].mean().item()
        history.append({
            "iter": it, "trials": it * p.batch_size, "accuracy": acc,
            "loss": loss.item(), "rate_e": rate_e, "rate_i": rate_i,
        })
        if on_step is not None:
            on_step(it, model)
        if it % p.log_every == 0 or it == p.iters:
            recent = history[-p.log_every:]
            mean_acc = sum(h["accuracy"] for h in recent) / len(recent)
            elapsed = time.time() - t0
            eta = elapsed / it * (p.iters - it)
            log(f"  [ML]  adım {it:4d}/{p.iters}  kayıp {loss.item():.3f}  "
                f"doğruluk {mean_acc:5.1%}  E {rate_e:4.1f} Hz  I {rate_i:4.1f} Hz  "
                f"(kalan ~{eta:4.0f} sn)")
    return history
