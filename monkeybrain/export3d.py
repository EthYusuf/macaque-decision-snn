"""3D beyin görselleştirmesi için veri üretimi.

Eğitilmiş modelleri yükler. Öğrenmenin her aşamasında (anlık görüntü) AYNI uyaranı
(aynı MT spike'ları, aynı arka plan gürültüsü) ağa verir ve LIP spike'larını kaydeder.
Böylece aşamalar arasındaki tek fark öğrenilen ağırlıklardır. Sonuç, şablondaki
etkileşimli Three.js sayfasına JSON olarak gömülür.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .analysis import smooth_rate
from .plots import METHOD_NAMES
from .task import RandomDotTask
from .utils import load_checkpoint, load_json, load_snapshots

TEMPLATE = Path(__file__).parent / "templates" / "brain3d.html"
VIEW_COHERENCES = (0.064, 0.512)   # bir zor, bir kolay deneme
TRIAL_SEED = 7
RATE_STEP_MS = 5                   # havuz hız izlerinin örnekleme aralığı
SMOOTH_MS = 50


def _spike_lists(spikes: np.ndarray) -> list[list[int]]:
    """(T, n) bool -> her nöron için spike zamanları (adım)."""
    return [np.flatnonzero(spikes[:, i]).tolist() for i in range(spikes.shape[1])]


def _make_stimuli(cfg) -> dict[str, torch.Tensor]:
    """Her tutarlılık için tek bir sağa hareket denemesinin MT spike'ları."""
    task = RandomDotTask(cfg.task, "cpu")
    out = {}
    for k, c in enumerate(VIEW_COHERENCES):
        g = torch.Generator().manual_seed(TRIAL_SEED + k)
        batch = task.sample(1, g, coherence=torch.tensor([c]), direction=torch.tensor([1.0]))
        out[f"{c:.3f}"] = batch.x
    return out


@torch.no_grad()
def _simulate(model, x: torch.Tensor, cfg) -> dict:
    torch.manual_seed(TRIAL_SEED)  # aynı arka plan gürültüsü -> aşamalar karşılaştırılabilir
    spikes = model(x)                                              # (T, 1, n)
    window = int(cfg.task.decision_window / cfg.task.dt)
    rates = model.pool_rates(spikes, window)[0]
    choice = int(rates[1] > rates[0])                              # 1 = sağ
    to_hz = 1000.0 / cfg.task.dt
    win = int(SMOOTH_MS / cfg.task.dt)
    step = int(RATE_STEP_MS / cfg.task.dt)
    left = smooth_rate(spikes[:, :, model.pool_left].mean(-1), win, to_hz)[:, 0]
    right = smooth_rate(spikes[:, :, model.pool_right].mean(-1), win, to_hz)[:, 0]
    return {
        "lip": _spike_lists(spikes[:, 0].numpy().astype(bool)),
        "choice": choice,
        "correct": choice == 1,
        "rates": {"left": np.round(left[::step].numpy(), 1).tolist(),
                  "right": np.round(right[::step].numpy(), 1).tolist()},
    }


def _method_data(run_dir: Path, stimuli: dict) -> dict:
    model, cfg, init_state = load_checkpoint(run_dir, torch.device("cpu"))
    snaps = load_snapshots(run_dir)
    if not snaps:  # eski checkpoint: yalnızca başlangıç ve son
        snaps = [{"iter": 0, "w_in": init_state["w_in"], "w_rec": init_state["w_rec"]},
                 {"iter": cfg.train.iters, "w_in": model.w_in.detach().clone(),
                  "w_rec": model.w_rec.detach().clone()}]
    history = load_json(run_dir / "metrics.json")["history"]
    acc = np.array([h["accuracy"] for h in history])
    kernel = np.ones(20) / 20
    smooth = np.convolve(np.pad(acc, (19, 0), mode="edge"), kernel, mode="valid")

    pools = 2 * cfg.net.pool_size
    src, dst = np.nonzero(model.mask_in[:, :pools].numpy())       # MT -> seçim havuzu bağlantıları
    stages = []
    for snap in snaps:
        model.w_in.data.copy_(snap["w_in"])
        model.w_rec.data.copy_(snap["w_rec"])
        w = (model.w_in * model.mask_in)[:, :pools].detach().numpy()[src, dst]
        it = snap["iter"]
        stages.append({
            "iter": it,
            "trials": it * cfg.train.batch_size,
            "accuracy": None if it == 0 else float(smooth[it - 1]),
            "weights": np.round(w, 4).tolist(),
            "sims": {k: _simulate(model, x, cfg) for k, x in stimuli.items()},
        })
    idx = np.unique(np.linspace(0, len(acc) - 1, min(160, len(acc))).astype(int))
    return {
        "name": METHOD_NAMES[cfg.train.method],
        "connections": {"src": src.tolist(), "dst": dst.tolist()},
        "history": {"trials": [history[i]["trials"] for i in idx],
                    "accuracy": np.round(smooth[idx], 3).tolist()},
        "stages": stages,
    }


def _social_meta(site_url: str) -> str:
    """Bağlantı paylaşılınca (LinkedIn, X, ...) görünen önizleme kartı."""
    url = site_url.rstrip("/")
    title = "Anatomy of a decision · macaque brain in 3D"
    desc = ("264 spiking neurons learn the classic random-dot motion task. "
            "Watch real simulated spikes flow from MT to LIP and see learning rewire the brain.")
    return (
        f'<meta name="description" content="{desc}">\n'
        f'<meta property="og:type" content="website">\n'
        f'<meta property="og:title" content="{title}">\n'
        f'<meta property="og:description" content="{desc}">\n'
        f'<meta property="og:url" content="{url}/">\n'
        f'<meta property="og:image" content="{url}/images/hero.jpg">\n'
        f'<meta name="twitter:card" content="summary_large_image">\n'
    )


def export_brain3d(runs: dict[str, Path], out_path: Path, site_url: str | None = None) -> Path:
    """runs: yöntem -> çalışma klasörü. Etkileşimli HTML sayfasını yazar.

    site_url verilirse sayfaya sosyal medya önizleme etiketleri eklenir (GitHub Pages için).
    """
    _, cfg, _ = load_checkpoint(next(iter(runs.values())), torch.device("cpu"))
    stimuli = _make_stimuli(cfg)
    p, net = cfg.task, cfg.net
    data = {
        "meta": {
            "T": p.n_steps, "dt": p.dt, "t_fix": p.t_fix, "mt_latency": p.mt_latency,
            "frame_ms": p.frame_ms, "n_mt": p.n_mt, "n_exc": net.n_exc, "n_inh": net.n_inh,
            "pool_size": net.pool_size, "rate_step_ms": RATE_STEP_MS,
            "pref_deg": [round(i * 360 / p.n_mt, 2) for i in range(p.n_mt)],
            "coherences": [f"{c:.3f}" for c in VIEW_COHERENCES],
            "mt": {k: _spike_lists(x[:, 0].numpy().astype(bool)) for k, x in stimuli.items()},
        },
        "methods": {m: _method_data(run, stimuli) for m, run in runs.items()},
    }
    html = TEMPLATE.read_text(encoding="utf-8")
    if site_url:
        html = html.replace("<title>", _social_meta(site_url) + "<title>", 1)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html.replace("/*__DATA__*/null", payload), encoding="utf-8")
    return out_path
