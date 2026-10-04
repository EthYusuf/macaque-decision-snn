"""Değerlendirme ve analiz: modeli gerçek maymun bulgularıyla aynı ölçütlerle ölçer.

- Psikometrik eğri: tutarlılığa göre doğruluk; Weibull fit (eşik alpha, eğim beta)
- Kronometrik eğri: tutarlılığa göre karar süresi
- LIP rampa (PSTH): seçilen ve seçilmeyen havuzun zaman içindeki hızı
- Ağırlık seçiciliği: MT -> havuz ağırlıkları yön tercihine uygun mu?
"""

import numpy as np
import torch
from scipy.optimize import curve_fit

from .config import Config
from .network import SpikingBrain
from .task import RandomDotTask

# Gerçek makaklar için YAKLAŞIK psikometrik referans (Weibull). Rastgele nokta
# deneylerinde (ör. Roitman & Shadlen 2002) tipik olarak %0'da şans düzeyi,
# %51.2'de ~%100 doğruluk ve ~%10 civarı eşik görülür. Bu kesin veri DEĞİL,
# tipik davranışı temsil eden bir eğridir.
MONKEY_REFERENCE = {"alpha": 0.10, "beta": 1.3}

SMOOTH_MS = 50  # PSTH ve karar süresi için kayan pencere


def weibull(c, alpha, beta):
    """İki seçenekli görev için Weibull: c = 0'da 0.5, büyük c'de 1."""
    return 1.0 - 0.5 * np.exp(-(np.asarray(c) / alpha) ** beta)


def smooth_rate(spikes_pop: torch.Tensor, win: int, to_hz: float) -> torch.Tensor:
    """(T, B) ortalama spike -> nedensel kayan pencereli hız (Hz)."""
    c = torch.cumsum(spikes_pop, dim=0)
    out = c.clone()
    out[win:] = c[win:] - c[:-win]
    counts = torch.arange(1, spikes_pop.shape[0] + 1, device=c.device).clamp(max=win)
    return out / counts[:, None] * to_hz


@torch.no_grad()
def run_evaluation(
    model: SpikingBrain, task: RandomDotTask, cfg: Config,
    n_per_cond: int, generator: torch.Generator, chunk: int = 200,
) -> dict:
    """Her (tutarlılık, yön) koşulunda n_per_cond deneme çalıştırır."""
    p = cfg.task
    dev = model.w_in.device
    window = int(p.decision_window / p.dt)
    to_hz = 1000.0 / p.dt
    win = int(SMOOTH_MS / p.dt)

    cohs, dirs = [], []
    for c in p.coherences:
        for d in (-1.0, 1.0):
            cohs += [c] * n_per_cond
            dirs += [d] * n_per_cond
    coh_all = torch.tensor(cohs)
    dir_all = torch.tensor(dirs)

    choices, traces = [], []
    spike_sum = torch.zeros(cfg.net.n_total, device=dev)
    example = None
    for start in range(0, len(cohs), chunk):
        sl = slice(start, start + chunk)
        batch = task.sample(len(coh_all[sl]), generator, coh_all[sl], dir_all[sl])
        spikes = model(batch.x)
        choices.append(model.choice(spikes, window).cpu())
        left = smooth_rate(spikes[:, :, model.pool_left].mean(-1), win, to_hz)
        right = smooth_rate(spikes[:, :, model.pool_right].mean(-1), win, to_hz)
        traces.append(torch.stack([left, right], dim=-1).transpose(0, 1).cpu())  # (B, T, 2)
        spike_sum += spikes.sum(dim=(0, 1))
        # Örnek deneme: en kolay koşul, sağa hareket
        if example is None:
            idx = ((batch.coherence == max(p.coherences)) & (batch.direction > 0)).nonzero()
            if len(idx):
                i = idx[0].item()
                example = {"mt": batch.x[:, i].cpu().numpy().astype(bool),
                           "lip": spikes[:, i].cpu().numpy().astype(bool)}

    n_trials = len(cohs)
    return {
        "coherence": coh_all.numpy(),
        "direction": dir_all.numpy(),
        "label": (dir_all > 0).long().numpy(),
        "choice": torch.cat(choices).numpy(),
        "traces": torch.cat(traces).numpy(),           # (N, T, 2) Hz
        "neuron_hz": (spike_sum / (n_trials * p.n_steps) * to_hz).cpu().numpy(),
        "example": example,
    }


def accuracy_by_coherence(res: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    levels = np.unique(res["coherence"])
    acc, sem = [], []
    for c in levels:
        sel = res["coherence"] == c
        correct = (res["choice"][sel] == res["label"][sel]).astype(float)
        acc.append(correct.mean())
        sem.append(correct.std(ddof=1) / np.sqrt(sel.sum()))
    return levels, np.array(acc), np.array(sem)


def p_right_by_signed_coherence(res: dict) -> tuple[np.ndarray, np.ndarray]:
    signed = res["coherence"] * res["direction"]
    levels = np.unique(signed)
    p_right = np.array([(res["choice"][signed == s] == 1).mean() for s in levels])
    return levels, p_right


def fit_weibull(levels: np.ndarray, acc: np.ndarray) -> dict:
    try:
        (alpha, beta), _ = curve_fit(
            weibull, levels, acc, p0=[0.1, 1.5],
            bounds=([1e-3, 0.3], [5.0, 10.0]), maxfev=10000,
        )
        return {"alpha": float(alpha), "beta": float(beta)}
    except (RuntimeError, ValueError):
        return {"alpha": float("nan"), "beta": float("nan")}


MIN_CROSSING = 0.8  # her koşulda denemelerin en az bu kadarı eşiğe ulaşmalı


def decision_threshold(res: dict, cfg: Config) -> float:
    """Karar eşiği (Hz). Maymun LIP'inde karar, aktivite belli bir seviyeye
    ulaşınca verilir (Roitman & Shadlen 2002).

    Eşik, en zor koşulda bile denemelerin %80'inin ulaştığı EN YÜKSEK hızdır.
    Eşik çok yüksek seçilirse zor denemelerin çoğu sınıra hiç ulaşmaz ve ortalama
    yalnızca "şanslı" denemelerden hesaplanır (hayatta kalan yanlılığı).
    """
    onset = cfg.task.stim_onset_step
    winner_peak = res["traces"][:, onset:].max(axis=(1, 2))        # (N,) deneme başına tepe
    baseline = float(res["traces"][:, :onset].mean())
    limit = min(np.quantile(winner_peak[res["coherence"] == c], 1 - MIN_CROSSING)
                for c in np.unique(res["coherence"]))
    return float(max(limit, baseline + 1.0))


def decision_times(res: dict, cfg: Config, threshold: float) -> np.ndarray:
    """Hareket başlangıcından itibaren herhangi bir havuzun eşiği ilk geçtiği an (ms).
    Eşik hiç geçilmezse NaN."""
    onset = cfg.task.stim_onset_step
    winner = res["traces"][:, onset:].max(axis=2)          # (N, T_stim)
    crossed = winner >= threshold
    first = crossed.argmax(axis=1).astype(float)
    first[~crossed.any(axis=1)] = np.nan
    return first * cfg.task.dt


def chronometric(res: dict, rts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    levels = np.unique(res["coherence"])
    mean, sem = [], []
    for c in levels:
        r = rts[res["coherence"] == c]
        r = r[~np.isnan(r)]
        mean.append(r.mean() if len(r) else np.nan)
        sem.append(r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else np.nan)
    return levels, np.array(mean), np.array(sem)


def psth_by_coherence(res: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seçilen ve seçilmeyen havuzun ortalama hızı, tutarlılığa göre.

    Doğru denemeler kullanılır (%0'da tüm denemeler). Şekil (n_coh, T).
    """
    levels = np.unique(res["coherence"])
    tr = res["traces"]
    chosen_idx = res["choice"]
    n = np.arange(len(chosen_idx))
    chosen = tr[n, :, chosen_idx]          # (N, T)
    unchosen = tr[n, :, 1 - chosen_idx]
    correct = res["choice"] == res["label"]
    out_c, out_u = [], []
    for c in levels:
        sel = (res["coherence"] == c) & (correct | (c == 0))
        out_c.append(chosen[sel].mean(axis=0))
        out_u.append(unchosen[sel].mean(axis=0))
    return levels, np.array(out_c), np.array(out_u)


def pool_weight_profiles(w_in: torch.Tensor, mask_in: torch.Tensor, model_like) -> dict:
    """Her MT nöronundan Sol/Sağ havuzlara ortalama ağırlık (tercih yönüne göre)."""
    w = (w_in * mask_in).cpu()
    m = mask_in.cpu()
    left = w[:, model_like.pool_left].sum(1) / m[:, model_like.pool_left].sum(1).clamp(min=1)
    right = w[:, model_like.pool_right].sum(1) / m[:, model_like.pool_right].sum(1).clamp(min=1)
    return {"left": left.numpy(), "right": right.numpy()}


def direction_selectivity(profiles: dict, cos_pref: np.ndarray) -> float:
    """Ağırlıkların doğru yöne ne kadar ayarlandığı (0 = hiç, > 0 = doğru yön).

    Sağ havuz sağı tercih eden MT'den, sol havuz solu tercih eden MT'den daha
    güçlü girdi almalı. Ortalama ağırlığa bölünerek normalize edilir.
    """
    right_pref, left_pref = cos_pref > 0.1, cos_pref < -0.1
    r = profiles["right"][right_pref].mean() - profiles["right"][left_pref].mean()
    l = profiles["left"][left_pref].mean() - profiles["left"][right_pref].mean()
    scale = 0.5 * (profiles["right"].mean() + profiles["left"].mean())
    return float((r + l) / 2 / max(scale, 1e-8))


def peak_rates(res: dict, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Seçilen havuzun uyaran sırasındaki tepe hızı (Hz), tutarlılığa göre ortalama.

    Gerçek LIP nöronları karar anında ~60-70 Hz civarına çıkar (Roitman & Shadlen 2002).
    """
    onset = cfg.task.stim_onset_step
    n = np.arange(len(res["choice"]))
    chosen = res["traces"][n, onset:, res["choice"]]               # (N, T_stim)
    peak = chosen.max(axis=1)
    levels = np.unique(res["coherence"])
    return levels, np.array([peak[res["coherence"] == c].mean() for c in levels])


def summarize(res: dict, cfg: Config, model: SpikingBrain) -> dict:
    """metrics.json için JSON uyumlu özet."""
    levels, acc, sem = accuracy_by_coherence(res)
    fit = fit_weibull(levels, acc)
    thr = decision_threshold(res, cfg)
    rts = decision_times(res, cfg, thr)
    _, rt_mean, rt_sem = chronometric(res, rts)
    _, peaks = peak_rates(res, cfg)
    hz = res["neuron_hz"]
    ne = cfg.net.n_exc
    return {
        "peak_rate_hz": peaks.tolist(),
        "peak_rate_max_hz": float(peaks.max()),
        "coherences": levels.tolist(),
        "accuracy": acc.tolist(),
        "accuracy_sem": sem.tolist(),
        "overall_accuracy": float((res["choice"] == res["label"]).mean()),
        "weibull": fit,
        "decision_threshold_hz": thr,
        "decision_time_ms": [None if np.isnan(x) else float(x) for x in rt_mean],
        "decision_time_sem": [None if np.isnan(x) else float(x) for x in rt_sem],
        "no_decision_fraction": float(np.isnan(rts).mean()),
        "rate_exc_hz": float(hz[:ne].mean()),
        "rate_inh_hz": float(hz[ne:].mean()),
        "rate_pools_hz": float(hz[: 2 * cfg.net.pool_size].mean()),
    }
