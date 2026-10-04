"""SpikingBrain: MT -> LIP spiking ağı. İki eğiticinin ortak simülasyon motoru.

LIP nöron düzeni (indeksler):
    [0, pool)              Sol seçim havuzu (E)
    [pool, 2*pool)         Sağ seçim havuzu (E)
    [2*pool, n_exc)        Seçici olmayan E nöronları
    [n_exc, n_exc+n_inh)   Baskılayıcı (I) internöronlar

Ağırlık kuralı: W[önce, sonra]. Yani `spikes @ W` sonraki nöronların girdisidir.
Dale yasası: ağırlıklar >= 0 büyüklük olarak saklanır. İşareti önceki nöronun
tipi belirler (E: +, I: -). Her eğitim adımından sonra apply_constraints()
negatif büyüklükleri sıfıra çeker; böylece bir nöron tipini asla değiştirmez.
"""

import math

import torch
from torch import nn

from .config import NetworkParams, TaskParams
from .neurons import spike_fn


def _fixed_indegree(
    n_pre: int, n_post: int, p: float, g: torch.Generator, self_offset: int | None = None,
) -> torch.Tensor:
    """(n_pre, n_post) maske: her sonraki nöron tam k = round(p * n_pre) girdi alır.

    self_offset verilirse önceki grup sonraki nöronlarla örtüşür ve kendine
    bağlantı engellenir (önceki i, sonraki self_offset + i ile aynı nöron).
    """
    scores = torch.rand(n_pre, n_post, generator=g)
    k = round(p * n_pre)
    if self_offset is not None:
        idx = torch.arange(n_pre)
        post_idx = self_offset + idx
        valid = post_idx < n_post
        scores[idx[valid], post_idx[valid]] = 2.0   # en sona it, seçilmesin
        k = min(k, n_pre - 1)
    chosen = scores.argsort(dim=0)[:k]
    mask = torch.zeros(n_pre, n_post)
    mask.scatter_(0, chosen, 1.0)
    return mask


class SpikingBrain(nn.Module):
    def __init__(self, net: NetworkParams, task: TaskParams, seed: int = 0):
        super().__init__()
        self.net = net
        self.dt = task.dt
        self.n_in = task.n_mt
        n, ne, pool = net.n_total, net.n_exc, net.pool_size
        g = torch.Generator().manual_seed(seed)

        # --- Popülasyonlar ---
        self.pool_left = slice(0, pool)
        self.pool_right = slice(pool, 2 * pool)
        self.exc = slice(0, ne)
        self.inh = slice(ne, n)

        sign = torch.ones(n)
        sign[ne:] = -1.0
        tau_mem = torch.full((n,), net.tau_mem_exc)
        tau_mem[ne:] = net.tau_mem_inh
        adapt = torch.zeros(n)
        adapt[:ne] = 1.0  # adaptasyon yalnızca piramidal (E) hücrelerde
        self.register_buffer("sign", sign)
        self.register_buffer("alpha_mem", torch.exp(-self.dt / tau_mem))
        self.register_buffer("adapt_mask", adapt)
        self.alpha_syn = math.exp(-self.dt / net.tau_syn)
        self.alpha_nmda = math.exp(-self.dt / net.tau_nmda)
        # NMDA ölçeği: düşük hızlarda bir spike'ın toplam yükü, aynı ağırlıklı AMPA ile eşit olur.
        # Yüksek hızlarda g -> 1 doyar ve akım artık büyümez (Wang 2002).
        self.nmda_scale = net.tau_syn / (net.nmda_kappa * net.tau_nmda)
        self.alpha_adapt = math.exp(-self.dt / net.tau_adapt)
        self.n_ref = int(round(net.refractory_ms / self.dt))

        # --- Seyrek bağlantı maskeleri: sabit girdi derecesi (Brunel 2000) ---
        # Her nöron her gruptan TAM OLARAK aynı sayıda girdi alır. Tamamen rastgele
        # bağlantıda havuzlar arasında ~%5 fark olur; güçlü özyineleme bu farkı
        # büyütür ve %0 tutarlılıkta bile hep aynı havuz kazanır.
        mask_in = _fixed_indegree(self.n_in, n, net.p_in, g)
        mask_rec = torch.zeros(n, n)
        groups = [self.pool_left, self.pool_right, slice(2 * pool, ne), self.inh]
        for grp in groups:
            size = grp.stop - grp.start
            mask_rec[grp] = _fixed_indegree(size, n, net.p_rec, g, self_offset=grp.start)
        self.register_buffer("mask_in", mask_in)
        self.register_buffer("mask_rec", mask_rec)
        nmda = torch.zeros(n, n)
        nmda[:ne, :ne] = 1.0          # E -> E sinapsları yavaş (NMDA)
        self.register_buffer("nmda_mask", nmda)

        # --- Başlangıç ağırlıkları (büyüklükler) ---
        # MT -> LIP: hafif rastgele, yön seçiciliği YOK; öğrenilecek olan bu
        w_in = net.w_in * (0.8 + 0.4 * torch.rand(self.n_in, n, generator=g))

        w_rec = torch.zeros(n, n)
        w_rec[:ne, :ne] = net.w_ee
        for p in (self.pool_left, self.pool_right):  # havuz içi öz-uyarım
            w_rec[p, p] = net.w_ee * net.w_plus
        w_rec[:ne, ne:] = net.w_ei
        w_rec[ne:, :ne] = net.w_ie
        w_rec[ne:, ne:] = net.w_ii

        self.w_in = nn.Parameter(w_in)
        self.w_rec = nn.Parameter(w_rec)

    # ------------------------------------------------------------------
    def effective_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Maskelenmiş ve işaretli (Dale yasası) ağırlıklar."""
        w_in = self.w_in * self.mask_in
        w_rec = self.w_rec * self.mask_rec * self.sign[:, None]
        return w_in, w_rec

    @torch.no_grad()
    def apply_constraints(self) -> None:
        """Dale yasası: büyüklükler negatif olamaz."""
        self.w_in.clamp_(min=0.0)
        self.w_rec.clamp_(min=0.0)

    def snapshot(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ağı T adım simüle eder.

        x: (T, B, n_in) MT spike'ları. Döndürür: (T, B, n) LIP spike'ları.
        Her adımda (dt = 1 ms):
            i_fast <- a_s * i_fast + x W_in + s W_hızlı        AMPA / GABA_A akımı
            i_nmda  = g W_E->E                                 yavaş NMDA akımı
            v      <- a_m * v + (1 - a_m) * (i_fast + i_nmda + arka plan)   membran
            s       = spike(v - (v_th + beta * a))             adaptif eşik
            spike sonrası: v <- 0, refrakter dönem, a <- a + 1
            g      <- a_n * g + kappa * (1 - g) * s            NMDA açık oranı (0..1, doyar)
        """
        net = self.net
        T, B, _ = x.shape
        n = net.n_total
        dev = x.device
        w_in, w_rec = self.effective_weights()
        w_fast = w_rec * (1 - self.nmda_mask)
        w_nmda = w_rec * self.nmda_mask * self.nmda_scale
        alpha_mem = self.alpha_mem
        beta = net.beta_adapt * self.adapt_mask

        v = torch.zeros(B, n, device=dev)
        i_fast = torch.zeros(B, n, device=dev)
        g = torch.zeros(B, n, device=dev)       # önceki nöron başına NMDA açık oranı
        a = torch.zeros(B, n, device=dev)
        ref = torch.zeros(B, n, device=dev)
        s = torch.zeros(B, n, device=dev)
        # Dış girdiler zamandan bağımsız hesaplanabilir: tek büyük çarpım döngüden hızlıdır
        # unbind: geri yayılımda her adım için tam boyutlu tensör oluşturmaz (indekslemeden çok hızlı)
        mt_input = (x @ w_in).unbind(0)
        background = net.bg_mean + net.bg_std * torch.randn(T, B, n, device=dev)
        out = []
        for t in range(T):
            i_fast = self.alpha_syn * i_fast + mt_input[t] + s @ w_fast
            i_nmda = g @ w_nmda
            can_fire = (ref <= 0).float()
            v = can_fire * (alpha_mem * v + (1 - alpha_mem) * (i_fast + i_nmda + background[t]))
            s = spike_fn(v - (net.v_th + beta * a), net.surrogate_slope) * can_fire
            s_d = s.detach()
            v = v * (1 - s_d)                              # sıfıra sıfırlama
            ref = (ref - 1).clamp(min=0) + s_d * self.n_ref
            a = self.alpha_adapt * a + s
            g = self.alpha_nmda * g + net.nmda_kappa * (1 - g) * s
            out.append(s)
        return torch.stack(out)

    # ------------------------------------------------------------------
    def pool_rates(self, spikes: torch.Tensor, window_steps: int) -> torch.Tensor:
        """Son `window_steps` adımdaki havuz ateşleme hızları (Hz). Şekil (B, 2): [sol, sağ]."""
        win = spikes[-window_steps:]
        to_hz = 1000.0 / self.dt
        left = win[:, :, self.pool_left].mean(dim=(0, 2)) * to_hz
        right = win[:, :, self.pool_right].mean(dim=(0, 2)) * to_hz
        return torch.stack([left, right], dim=1)

    def choice(self, spikes: torch.Tensor, window_steps: int) -> torch.Tensor:
        """Karar: karar penceresinde daha çok ateşleyen havuz (0 = sol, 1 = sağ).

        Eşitlikte yazı-tura atılır; aksi halde sessiz bir ağ hep "sol" derdi.
        """
        rates = self.pool_rates(spikes, window_steps)
        diff = rates[:, 1] - rates[:, 0]
        coin = torch.rand(diff.shape, device=diff.device) < 0.5
        return torch.where(diff == 0, coin, diff > 0).long()
