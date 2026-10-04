"""Nöron modelleri.

1. Izhikevich modeli (numpy): neokortekste (primatlar dahil memelilerde) gözlenen
   ateşleme desenlerini 4 parametreyle üretir. Yalnızca tek nöron demosunda kullanılır.
2. Adaptif LIF (ALIF): ağda kullanılan model. Basit, hızlı ve PyTorch ile
   türevlenebilir (surrogate gradient sayesinde).
"""

import math

import numpy as np
import torch

# Izhikevich (2003) parametreleri: (a, b, c, d)
IZHIKEVICH_TYPES = {
    "RS": ((0.02, 0.20, -65.0, 8.0), "Düzenli ateşleyen (piramidal)"),
    "IB": ((0.02, 0.20, -55.0, 4.0), "İçsel patlamalı (katman 5 piramidal)"),
    "CH": ((0.02, 0.20, -50.0, 2.0), "Chattering: hızlı ritmik patlamalar"),
    "FS": ((0.10, 0.20, -65.0, 2.0), "Hızlı ateşleyen (baskılayıcı internöron)"),
    "LTS": ((0.02, 0.25, -65.0, 2.0), "Düşük eşikli ateşleyen (internöron)"),
}
IZHIKEVICH_DESC_EN = {
    "RS": "Regular spiking (pyramidal)",
    "IB": "Intrinsically bursting (layer-5 pyramidal)",
    "CH": "Chattering: fast rhythmic bursts",
    "FS": "Fast spiking (inhibitory interneuron)",
    "LTS": "Low-threshold spiking (interneuron)",
    "ALIF": "The model used in the network (adaptive LIF)",
}


def simulate_izhikevich(
    a: float, b: float, c: float, d: float,
    current: float = 10.0, t_total: float = 500.0, t_on: float = 50.0,
    t_off: float = 450.0, dt: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Basamak akımla uyarılan tek bir Izhikevich nöronu.

    dv/dt = 0.04 v^2 + 5 v + 140 - u + I
    du/dt = a (b v - u)
    v >= 30 mV ise: v <- c, u <- u + d

    Döndürür: zaman (ms), membran potansiyeli (mV), giriş akımı.
    """
    n = int(t_total / dt)
    t = np.arange(n) * dt
    I = np.where((t >= t_on) & (t < t_off), current, 0.0)
    v = np.empty(n)
    v_now = -65.0
    u_now = b * v_now
    for k in range(n):
        v_now += dt * (0.04 * v_now**2 + 5 * v_now + 140 - u_now + I[k])
        u_now += dt * a * (b * v_now - u_now)
        if v_now >= 30.0:
            v[k] = 30.0          # spike tepesini çiz
            v_now, u_now = c, u_now + d
        else:
            v[k] = v_now
    return t, v, I


def simulate_alif(
    current: float = 1.5, t_total: float = 500.0, t_on: float = 50.0,
    t_off: float = 450.0, dt: float = 1.0, tau_mem: float = 20.0,
    tau_adapt: float = 200.0, beta: float = 0.2, refractory_ms: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ağdaki adaptif LIF nöronunun tek başına davranışı (görselleştirme için).

    Normalize v birimini mV'a çevirir: mV = -70 + 20 v.
    """
    n = int(t_total / dt)
    t = np.arange(n) * dt
    I = np.where((t >= t_on) & (t < t_off), current, 0.0)
    alpha = math.exp(-dt / tau_mem)
    rho = math.exp(-dt / tau_adapt)
    n_ref = int(round(refractory_ms / dt))
    v, a, ref = 0.0, 0.0, 0
    v_mv = np.empty(n)
    for k in range(n):
        if ref > 0:
            ref -= 1
            v = 0.0
        else:
            v = alpha * v + (1 - alpha) * I[k]
        spike = v >= 1.0 + beta * a
        a = rho * a + float(spike)
        if spike:
            v_mv[k] = 30.0
            v, ref = 0.0, n_ref
        else:
            v_mv[k] = -70.0 + 20.0 * v
    return t, v_mv, I


class SurrogateSpike(torch.autograd.Function):
    """Spike fonksiyonu: ileri yönde basamak, geri yönde yumuşak türev.

    Gerçek spike'ın türevi her yerde 0 (eşikte sonsuz). Bu yüzden gradyan
    akmaz ve geri yayılım çalışmaz. Çözüm (Neftci, Mostafa & Zenke 2019):
    geri yayılımda türevi "hızlı sigmoid" türeviyle değiştirmek:
        d spike / dx ~ 1 / (slope * |x| + 1)^2
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        surrogate = 1.0 / (ctx.slope * x.abs() + 1.0) ** 2
        return grad_out * surrogate, None


def spike_fn(x: torch.Tensor, slope: float = 10.0) -> torch.Tensor:
    return SurrogateSpike.apply(x, slope)
