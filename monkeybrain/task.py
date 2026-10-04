"""Rastgele nokta hareket görevi ve MT bölgesinin nüfus kodlaması.

Deney: Maymun ekrana bakar. Noktaların bir kısmı (tutarlılık = c) aynı yöne
(sağ veya sol) hareket eder, geri kalanı rastgele. Maymun hareket yönünü
gözüyle işaret eder. c = 0 ise doğru cevap yoktur ve ödül rastgele verilir.

MT modeli (Britten et al. 1993'ten esinlenen):
    r_i(t) = r0 + g * c(t) * cos(theta_uyaran - theta_tercih_i)
Tercih ettiği yönde hareket varsa MT nöronu daha çok, tersinde daha az ateşler.
c(t) kare kare dalgalanır, çünkü rastgele noktaların anlık hareket enerjisi
gürültülüdür. Görevi zorlaştıran şey de budur. V1 aşaması soyutlanmıştır.
"""

from dataclasses import dataclass

import torch

from .config import TaskParams

LEFT, RIGHT = 0, 1  # etiketler (sınıf indeksleri)


@dataclass
class TrialBatch:
    x: torch.Tensor           # (T, B, n_mt) MT spike'ları (0/1)
    coherence: torch.Tensor   # (B,) tutarlılık [0, 1]
    direction: torch.Tensor   # (B,) -1 = sol, +1 = sağ
    label: torch.Tensor       # (B,) 0 = sol, 1 = sağ

    @property
    def signed_coherence(self) -> torch.Tensor:
        return self.coherence * self.direction


class RandomDotTask:
    def __init__(self, p: TaskParams, device: torch.device | str = "cpu"):
        self.p = p
        self.device = torch.device(device)
        prefs = torch.arange(p.n_mt, dtype=torch.float32) * (2 * torch.pi / p.n_mt)
        self.pref_dirs = prefs.to(self.device)            # radyan; 0 = sağ, pi = sol
        self.cos_pref = torch.cos(prefs).to(self.device)  # sağa hareket için ayar eğrisi

    def sample(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
        coherence: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
    ) -> TrialBatch:
        """Bir grup deneme üretir. Tutarlılık/yön verilmezse rastgele seçilir."""
        p = self.p
        if coherence is None:
            levels = torch.tensor(p.coherences, dtype=torch.float32)
            coherence = levels[torch.randint(len(levels), (batch_size,), generator=generator)]
        if direction is None:
            direction = torch.randint(2, (batch_size,), generator=generator).float() * 2 - 1
        coherence = coherence.float().cpu()
        direction = direction.float().cpu()

        rates = self.rates(coherence, direction, generator)          # (T, B, n_mt) Hz
        prob = (rates * p.dt / 1000.0).clamp(0, 1)
        x = torch.bernoulli(prob, generator=generator)               # Poisson spike'lar
        label = (direction > 0).long()
        return TrialBatch(
            x=x.to(self.device),
            coherence=coherence.to(self.device),
            direction=direction.to(self.device),
            label=label.to(self.device),
        )

    def rates(
        self, coherence: torch.Tensor, direction: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """MT ateşleme hızları (Hz), şekil (T, B, n_mt). CPU üzerinde hesaplanır."""
        p = self.p
        B = coherence.shape[0]
        T = p.n_steps
        onset = p.stim_onset_step + int(p.mt_latency / p.dt)
        n_sig = T - onset

        # Kare kare dalgalanan anlık hareket sinyali: c(t) = yön * c + gürültü
        frame_steps = max(1, int(p.frame_ms / p.dt))
        n_frames = -(-n_sig // frame_steps)
        noise = torch.randn(n_frames, B, generator=generator) * p.stim_noise
        c_frames = (direction * coherence)[None, :] + noise
        c_t = c_frames.repeat_interleave(frame_steps, dim=0)[:n_sig]  # (n_sig, B)

        signal = torch.zeros(T, B)
        signal[onset:] = c_t
        base = torch.full((T, 1, 1), p.mt_spontaneous)   # fiksasyon: hareket yok
        base[onset:] = p.mt_baseline                     # hareket başladı (yönden bağımsız artış)
        cos_pref = self.cos_pref.cpu()
        rates = base + p.mt_gain * signal[:, :, None] * cos_pref[None, None, :]
        return rates.clamp(min=0.0)
