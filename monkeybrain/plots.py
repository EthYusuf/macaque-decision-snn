"""Tüm şekiller. Ortak stil ve renk rolleri tek yerde.

Renk rolleri (tüm şekillerde aynı):
    Sol havuz = mavi, Sağ havuz = turuncu, seçici olmayan E = gri, I = mor
    Tutarlılık seviyeleri = açıktan koyuya tek renk (mavi) skalası
    Maymun referansı = kesikli koyu gri
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"     # kategorik 1
ORANGE = "#eb6834"   # kategorik 2
VIOLET = "#4a3aa7"
COH_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

POOL_COLORS = {"left": BLUE, "right": ORANGE}
METHOD_COLORS = {"ml": BLUE, "bio": ORANGE}
METHOD_NAMES = {"ml": "ML (surrogate gradient)", "bio": "Biyolojik (R-STDP)"}
MARKER = dict(marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)

# Şekil dili: "tr" veya "en". CLI'de `--lang en` ile değişir.
LANG = "tr"
_TEXT = {  # anahtar: (Türkçe, English)
    "coh_axis": ("Hareket tutarlılığı (%)", "Motion coherence (%)"),
    "neuron_time": ("Zaman (ms)   (gri alan = akım verildi)", "Time (ms)   (grey band = current on)"),
    "neuron_title": ("Kortikal nöron tipleri: aynı akıma farklı tepkiler", "Cortical cell types: same input, different responses"),
    "spikes": ("{} spike", "{} spikes"),
    "mt_pref": ("MT tercih yönü (°)", "MT preferred direction (°)"),
    "mt_title": ("MT girdisi: sağı (0°) tercih edenler daha çok ateşliyor", "MT input: right-preferring (0°) neurons fire more"),
    "pool_left_e": ("Sol havuz (E)", "Left pool (E)"),
    "pool_right_e": ("Sağ havuz (E)", "Right pool (E)"),
    "nonsel_e": ("Seçici olmayan (E)", "Non-selective (E)"),
    "inh_i": ("Baskılayıcı (I)", "Inhibitory (I)"),
    "lip_index": ("LIP nöron no", "LIP neuron #"),
    "lip_title": ("LIP: seçim havuzları yarışıyor", "LIP: choice pools compete"),
    "rate_hz": ("Hız (Hz)", "Rate (Hz)"),
    "time_from_onset": ("Hareket başlangıcından itibaren zaman (ms)", "Time from motion onset (ms)"),
    "exc_mean": ("Uyarıcı E (ort. {:.1f} Hz)", "Excitatory E (mean {:.1f} Hz)"),
    "inh_mean": ("Baskılayıcı I (ort. {:.1f} Hz)", "Inhibitory I (mean {:.1f} Hz)"),
    "mean_rate": ("Ortalama ateşleme hızı (Hz)", "Mean firing rate (Hz)"),
    "n_neurons": ("Nöron sayısı", "Number of neurons"),
    "trials_seen": ("Görülen deneme sayısı", "Trials seen"),
    "accuracy": ("Doğruluk (%)", "Accuracy (%)"),
    "learning_title": ("Öğrenme eğrisi (tüm tutarlılıklar karışık)", "Learning curve (all coherences mixed)"),
    "monkey_ref": ("Maymun (yaklaşık referans)", "Monkey (approximate reference)"),
    "weibull_fit": ("Model Weibull fit (eşik %{:.1f})", "Model Weibull fit (threshold {:.1f}%)"),
    "psycho": ("Psikometrik eğri", "Psychometric curve"),
    "signed_coh": ("İşaretli tutarlılık (%)   (− sol, + sağ)", "Signed coherence (%)   (− left, + right)"),
    "p_right": ("'Sağ' seçimi (%)", "'Right' choices (%)"),
    "choice_prob": ("Seçim olasılığı", "Choice probability"),
    "decision_time": ("Karar süresi (ms)", "Decision time (ms)"),
    "pool_rate": ("Havuz hızı (Hz)", "Pool rate (Hz)"),
    "coherence": ("Tutarlılık", "Coherence"),
    "pct": ("%{:g}", "{:g}%"),
    "psth_note": ("düz çizgi: seçilen havuz\nkesikli: seçilmeyen havuz", "solid: chosen pool\ndashed: unchosen pool"),
    "pool_left": ("Sol havuz", "Left pool"),
    "pool_right": ("Sağ havuz", "Right pool"),
    "after_training": ("{} (eğitim sonrası)", "{} (after training)"),
    "before_thin": ("ince çizgi: eğitim öncesi", "thin line: before training"),
    "mt_pref_long": ("MT nöronunun tercih ettiği yön (°)   0° = sağ, 180° = sol",
                     "Preferred direction of the MT neuron (°)   0° = right, 180° = left"),
    "mean_weight": ("Ortalama MT → havuz ağırlığı", "Mean MT → pool weight"),
    "psycho_cmp": ("Psikometrik eğri: model ve maymun", "Psychometric curve: model vs monkey"),
    "chrono_cmp": ("Kronometrik eğri: zor karar daha uzun sürer mi?", "Chronometric curve: do hard decisions take longer?"),
    "mt_pref_short": ("MT tercih yönü (°)   0° = sağ, 180° = sol", "MT preferred direction (°)   0° = right, 180° = left"),
    "weight_diff": ("Sağ havuz − Sol havuz ağırlığı", "Right-pool − left-pool weight"),
    "learned_link": ("Öğrenilen bağlantı: hangi MT nöronu hangi havuza?", "Learned wiring: which MT neurons drive which pool?"),
    "cmp_title": ("Makine öğrenmesi ve beyin tarzı öğrenme: karşılaştırma", "Machine learning vs brain-style learning"),
    # __main__'deki şekil başlıkları
    "t_untrained_raster": ("Eğitilmemiş ağ, %51.2 sağa hareket", "Untrained network, 51.2% rightward motion"),
    "t_rates_untrained": ("Ateşleme hızı dağılımı (eğitim öncesi)", "Firing-rate distribution (before training)"),
    "t_rates_trained": ("Ateşleme hızı dağılımı (eğitim sonrası)", "Firing-rate distribution (after training)"),
    "t_psycho": ("Psikometrik eğri: {}", "Psychometric curve: {}"),
    "t_chrono": ("Karar süresi: {}", "Decision time: {}"),
    "t_ramp": ("LIP rampa aktivitesi: {}", "LIP ramping activity: {}"),
    "t_weights": ("Öğrenilen MT → LIP bağlantıları: {}", "Learned MT → LIP connections: {}"),
    "t_raster_trained": ("Eğitilmiş ağ, %51.2 sağa hareket: {}", "Trained network, 51.2% rightward motion: {}"),
}
_METHOD_NAMES_EN = {"ml": "ML (surrogate gradient)", "bio": "Biological (R-STDP)"}


def L(key: str, *args) -> str:
    """Şekil metni, seçili dilde."""
    tr, en = _TEXT[key]
    text = en if LANG == "en" else tr
    return text.format(*args) if args else text


def method_name(method: str) -> str:
    names = _METHOD_NAMES_EN if LANG == "en" else METHOD_NAMES
    return names.get(method, method)


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9, "text.color": INK,
    })


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _coh_x(levels) -> np.ndarray:
    """Tutarlılık eksenini log ölçekte çizebilmek için %0'ı en küçük seviyenin yarısına koyar."""
    levels = np.asarray(levels, dtype=float) * 100
    nonzero = levels[levels > 0]
    zero_pos = nonzero.min() / 2 if len(nonzero) else 1.0
    return np.where(levels == 0, zero_pos, levels)


def _fit_grid(levels) -> np.ndarray:
    """Fit eğrisi için x noktaları (%). En küçük sıfır olmayan seviyeden başlar; %0
    log eksende yapay bir konumda durduğu için eğri oradan geçiyormuş gibi çizilmez."""
    levels = np.asarray(levels, dtype=float) * 100
    nonzero = levels[levels > 0]
    return np.geomspace(nonzero.min(), nonzero.max(), 200)


def _coh_axis(ax, levels) -> None:
    ax.set_xscale("log")
    ax.set_xticks(_coh_x(levels))
    ax.set_xticklabels([f"{c * 100:g}" for c in levels])
    ax.minorticks_off()
    ax.set_xlabel(L("coh_axis"))


def _chance_line(ax, y=50) -> None:
    ax.axhline(y, color=MUTED, linewidth=1, zorder=0)


# ----------------------------------------------------------------------------
def plot_neuron_types(traces: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]], path: Path) -> Path:
    """traces: (kısa ad, açıklama, t, v_mV, I) listesi."""
    _style()
    fig, axes = plt.subplots(len(traces), 1, figsize=(9, 1.55 * len(traces)), sharex=True)
    for ax, (name, desc, t, v, I) in zip(axes, traces):
        on = I > 0
        ax.fill_between(t, -90, 40, where=on, color=GRID, alpha=0.5, linewidth=0, zorder=0)
        ax.plot(t, v, color=BLUE, linewidth=1.2)
        n_spikes = int(np.sum((v[1:] >= 29.9) & (v[:-1] < 29.9)))
        ax.set_title(f"{name}: {desc}   ({L('spikes', n_spikes)})", fontsize=10)
        ax.set_ylim(-90, 40)
        ax.set_ylabel("mV")
        ax.grid(axis="x", visible=False)
    axes[-1].set_xlabel(L("neuron_time"))
    fig.suptitle(L("neuron_title"), x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_raster(example: dict, model, cfg, path: Path, title: str) -> Path:
    _style()
    net, p = cfg.net, cfg.task
    mt, lip = example["mt"], example["lip"]
    t_ms = np.arange(mt.shape[0]) * p.dt - p.t_fix  # hareket başlangıcı = 0
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.6, 0.9]})

    # MT raster: tercih yönüne göre sıralı
    ax = axes[0]
    tt, nn = np.nonzero(mt)
    pref_deg = np.arange(p.n_mt) * 360 / p.n_mt
    ax.scatter(t_ms[tt], pref_deg[nn], s=1.5, color=INK2, linewidths=0)
    ax.set_ylabel(L("mt_pref"))
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_title(L("mt_title"))

    # LIP raster: gruplara göre renkli
    ax = axes[1]
    groups = [
        (model.pool_left, BLUE, L("pool_left_e")),
        (model.pool_right, ORANGE, L("pool_right_e")),
        (slice(2 * net.pool_size, net.n_exc), MUTED, L("nonsel_e")),
        (model.inh, VIOLET, L("inh_i")),
    ]
    for sl, color, label in groups:
        tt, nn = np.nonzero(lip[:, sl])
        ax.scatter(t_ms[tt], nn + sl.start, s=2, color=color, linewidths=0, label=label)
    ax.set_ylabel(L("lip_index"))
    ax.set_ylim(net.n_total, 0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), markerscale=5)
    ax.set_title(L("lip_title"))

    # Havuz hızları (bu deneme)
    ax = axes[2]
    win = int(50 / p.dt)
    kernel = np.ones(win) / win * 1000 / p.dt
    for sl, color, label in groups[:2]:
        r = np.convolve(lip[:, sl].mean(1), kernel, mode="full")[: len(t_ms)]
        ax.plot(t_ms, r, color=color, label=label)
    ax.set_ylabel(L("rate_hz"))
    ax.set_xlabel(L("time_from_onset"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

    for ax in axes:
        ax.axvline(0, color=INK2, linewidth=1)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_rate_hist(neuron_hz: np.ndarray, cfg, path: Path, title: str) -> Path:
    _style()
    ne = cfg.net.n_exc
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(0, max(60, neuron_hz.max() + 2), 31)
    ax.hist(neuron_hz[:ne], bins=bins, color=BLUE, rwidth=0.85, label=L("exc_mean", neuron_hz[:ne].mean()))
    ax.hist(neuron_hz[ne:], bins=bins, color=VIOLET, rwidth=0.85, alpha=0.85,
            label=L("inh_mean", neuron_hz[ne:].mean()))
    ax.set_xlabel(L("mean_rate"))
    ax.set_ylabel(L("n_neurons"))
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, len(y)))
    c = np.cumsum(np.insert(y, 0, 0.0))
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def _learning_curve(ax, histories: dict) -> None:
    for method, hist in histories.items():
        trials = np.array([h["trials"] for h in hist])
        acc = np.array([h["accuracy"] for h in hist]) * 100
        color = METHOD_COLORS.get(method, BLUE)
        ax.plot(trials, acc, color=color, linewidth=0.8, alpha=0.25)
        ax.plot(trials, _smooth(acc, 20), color=color, label=method_name(method))
    _chance_line(ax)
    ax.set_ylim(30, 100)
    ax.set_xlabel(L("trials_seen"))
    ax.set_ylabel(L("accuracy"))
    ax.set_title(L("learning_title"))


def plot_learning_curve(histories: dict, path: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    _learning_curve(ax, histories)
    if len(histories) > 1:
        ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_psychometric(levels, acc, sem, fit, signed, p_right, path: Path, title: str) -> Path:
    from .analysis import MONKEY_REFERENCE, weibull

    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = _coh_x(levels)
    grid = _fit_grid(levels)
    ref = MONKEY_REFERENCE
    ax1.plot(grid, weibull(grid / 100, ref["alpha"], ref["beta"]) * 100, color=INK2,
             linestyle="--", linewidth=1.5, label=L("monkey_ref"))
    if np.isfinite(fit["alpha"]):
        ax1.plot(grid, weibull(grid / 100, fit["alpha"], fit["beta"]) * 100, color=BLUE,
                 label=L("weibull_fit", fit["alpha"] * 100))
    ax1.errorbar(x, acc * 100, yerr=sem * 100, color=BLUE, linestyle="none", capsize=0, **MARKER)
    _chance_line(ax1)
    _coh_axis(ax1, levels)
    ax1.set_ylim(40, 102)
    ax1.set_ylabel(L("accuracy"))
    ax1.set_title(L("psycho"))
    ax1.legend(loc="lower right")

    ax2.plot(np.asarray(signed) * 100, np.asarray(p_right) * 100, color=BLUE, **MARKER)
    _chance_line(ax2)
    ax2.axvline(0, color=MUTED, linewidth=1, zorder=0)
    ax2.set_xlabel(L("signed_coh"))
    ax2.set_ylabel(L("p_right"))
    ax2.set_ylim(-2, 102)
    ax2.set_title(L("choice_prob"))
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_chronometric(levels, mean, sem, path: Path, title: str) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = _coh_x(levels)
    ok = np.isfinite(mean)
    ax.errorbar(x[ok], mean[ok], yerr=np.nan_to_num(sem[ok]), color=BLUE, capsize=0, **MARKER)
    _coh_axis(ax, levels)
    ax.set_ylabel(L("decision_time"))
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, path)


def plot_psth(levels, chosen, unchosen, cfg, path: Path, title: str) -> Path:
    _style()
    p = cfg.task
    t_ms = np.arange(chosen.shape[1]) * p.dt - p.t_fix
    fig, ax = plt.subplots(figsize=(8, 4.4))
    colors = COH_RAMP[-len(levels):] if len(levels) <= len(COH_RAMP) else COH_RAMP
    for c, col, yc, yu in zip(levels, colors, chosen, unchosen):
        ax.plot(t_ms, yc, color=col, label=L("pct", c * 100))
        ax.plot(t_ms, yu, color=col, linestyle="--", linewidth=1.5)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlabel(L("time_from_onset"))
    ax.set_ylabel(L("pool_rate"))
    ax.set_title(title)
    leg = ax.legend(title=L("coherence"), loc="upper left", bbox_to_anchor=(1.0, 1.0))
    leg.get_title().set_color(INK2)
    ax.text(1.02, 0.08, L("psth_note"),
            transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    fig.tight_layout()
    return _save(fig, path)


def plot_weights(pref_deg, init_prof: dict, final_prof: dict, path: Path, title: str) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    names = {"left": L("pool_left"), "right": L("pool_right")}
    for key in ("left", "right"):
        col = POOL_COLORS[key]
        ax.plot(pref_deg, init_prof[key], color=col, linewidth=1, alpha=0.4)
        ax.plot(pref_deg, final_prof[key], color=col, label=L("after_training", names[key]))
    ax.plot([], [], color=MUTED, linewidth=1, alpha=0.6, label=L("before_thin"))
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel(L("mt_pref_long"))
    ax.set_ylabel(L("mean_weight"))
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return _save(fig, path)


def plot_comparison(runs: dict, levels_ref, pref_deg, path: Path) -> Path:
    """runs: method -> {"history", "eval", "init_prof", "final_prof"}"""
    from .analysis import MONKEY_REFERENCE, weibull

    _style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    _learning_curve(axes[0, 0], {m: r["history"] for m, r in runs.items()})
    axes[0, 0].legend(loc="lower right")

    ax = axes[0, 1]
    x_ref = _coh_x(levels_ref)
    grid = _fit_grid(levels_ref)
    ref = MONKEY_REFERENCE
    ax.plot(grid, weibull(grid / 100, ref["alpha"], ref["beta"]) * 100, color=INK2,
            linestyle="--", linewidth=1.5, label=L("monkey_ref"))
    for m, r in runs.items():
        ev = r["eval"]
        ax.plot(_coh_x(ev["coherences"]), np.array(ev["accuracy"]) * 100,
                color=METHOD_COLORS[m], label=method_name(m), **MARKER)
    _chance_line(ax)
    _coh_axis(ax, levels_ref)
    ax.set_ylim(40, 102)
    ax.set_ylabel(L("accuracy"))
    ax.set_title(L("psycho_cmp"))
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    for m, r in runs.items():
        ev = r["eval"]
        rt = np.array([np.nan if v is None else v for v in ev["decision_time_ms"]], dtype=float)
        ok = np.isfinite(rt)
        ax.plot(_coh_x(ev["coherences"])[ok], rt[ok], color=METHOD_COLORS[m],
                label=method_name(m), **MARKER)
    _coh_axis(ax, levels_ref)
    ax.set_ylabel(L("decision_time"))
    ax.set_title(L("chrono_cmp"))
    ax.legend()

    ax = axes[1, 1]
    for m, r in runs.items():
        diff = r["final_prof"]["right"] - r["final_prof"]["left"]
        ax.plot(pref_deg, diff, color=METHOD_COLORS[m], label=method_name(m))
    ax.axhline(0, color=MUTED, linewidth=1, zorder=0)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel(L("mt_pref_short"))
    ax.set_ylabel(L("weight_diff"))
    ax.set_title(L("learned_link"))
    ax.legend()

    fig.suptitle(L("cmp_title"), x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)
