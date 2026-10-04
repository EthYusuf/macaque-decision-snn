"""Komut satırı arayüzü.

    python -m monkeybrain neuron                   # 1. tek nöron tipleri
    python -m monkeybrain simulate                 # 2. eğitilmemiş ağ
    python -m monkeybrain train --method ml        # 3. ML ile öğret
    python -m monkeybrain train --method bio       # 4. beyin tarzı öğret
    python -m monkeybrain evaluate --method ml     # 5. maymunla aynı testleri uygula
    python -m monkeybrain compare                  # 6. karşılaştır
    python -m monkeybrain brain3d                  # 7. 3D beyin görselleştirmesi
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import analysis, plots
from .config import Config
from .network import SpikingBrain
from .task import RandomDotTask
from .utils import (
    get_device, latest_run, load_checkpoint, load_json, make_run_dir, save_checkpoint,
    save_json, set_seed,
)

QUICK = {"ml": 60, "bio": 100}   # --quick modunda adım sayısı (~1-2 dk)
QUICK_EVAL_TRIALS = 40
SNAPSHOT_FRACTIONS = (0.02, 0.05, 0.15, 0.4, 1.0)  # eğitimin hangi anlarında ağırlıklar saklanır


def _header(text: str) -> None:
    print("\n" + "=" * 70 + f"\n  {text}\n" + "=" * 70)


def _done(paths) -> None:
    print("\n  Kaydedilen dosyalar:")
    for p in paths:
        print(f"    - {p}")


# ----------------------------------------------------------------------------
def cmd_neuron(args) -> None:
    from .neurons import IZHIKEVICH_DESC_EN, IZHIKEVICH_TYPES, simulate_alif, simulate_izhikevich

    _header("1. Adım: Tek nöron. Kortikal hücre tipleri")
    traces = []
    for name, (params, desc) in IZHIKEVICH_TYPES.items():
        t, v, I = simulate_izhikevich(*params)
        traces.append((name, IZHIKEVICH_DESC_EN[name] if plots.LANG == "en" else desc, t, v, I))
    cfg = Config()
    t, v, I = simulate_alif(tau_mem=cfg.net.tau_mem_exc, tau_adapt=cfg.net.tau_adapt,
                            beta=cfg.net.beta_adapt, refractory_ms=cfg.net.refractory_ms)
    traces.append(("ALIF", IZHIKEVICH_DESC_EN["ALIF"] if plots.LANG == "en"
                   else "Ağda kullandığımız model (adaptif LIF)", t, v, I))
    for name, desc, t, v, I in traces:
        n = int(np.sum((v[1:] >= 29.9) & (v[:-1] < 29.9)))
        print(f"  {name:5s} {desc:42s} {n:3d} spike / 400 ms")
    run = make_run_dir("neuron")
    _done([plots.plot_neuron_types(traces, run / "neuron_types.png")])


def cmd_simulate(args) -> None:
    _header("2. Adım: Eğitilmemiş ağ. Nöronlar ateşliyor ama henüz bir şey bilmiyor")
    cfg = Config()
    gen = set_seed(args.seed)
    device = get_device(args.device)
    task = RandomDotTask(cfg.task, device)
    model = SpikingBrain(cfg.net, cfg.task, seed=args.seed).to(device)
    res = analysis.run_evaluation(model, task, cfg, n_per_cond=20, generator=gen)
    s = analysis.summarize(res, cfg, model)
    print(f"  Ağ: {cfg.task.n_mt} MT girdi, {cfg.net.n_exc} E + {cfg.net.n_inh} I LIP nöronu")
    print(f"  Ortalama hız: E {s['rate_exc_hz']:.1f} Hz, I {s['rate_inh_hz']:.1f} Hz "
          "(gerçek korteks: E ~1-10 Hz, FS internöron daha yüksek)")
    print(f"  Doğruluk (eğitimsiz): %{s['overall_accuracy'] * 100:.1f}  (beklenen ~%50 = yazı-tura)")
    run = make_run_dir("simulate")
    _done([
        plots.plot_raster(res["example"], model, cfg, run / "raster_untrained.png",
                          plots.L("t_untrained_raster")),
        plots.plot_rate_hist(res["neuron_hz"], cfg, run / "rates_untrained.png",
                             plots.L("t_rates_untrained")),
    ])


def cmd_train(args) -> None:
    from .learning import TRAINERS

    method = args.method
    cfg = Config()
    cfg.train.method = method
    cfg.train.seed = args.seed
    if args.quick:
        cfg.train.iters = QUICK[method]
        cfg.train.eval_trials = QUICK_EVAL_TRIALS
    elif method == "bio":
        cfg.train.iters = 400   # biyolojik öğrenme daha çok deneme ister ama adımı ucuzdur
    if args.iters:
        cfg.train.iters = args.iters
    if args.batch_size:
        cfg.train.batch_size = args.batch_size

    name = plots.METHOD_NAMES[method]
    _header(f"Eğitim: {name}  ({cfg.train.iters} adım x {cfg.train.batch_size} deneme)")
    gen = set_seed(cfg.train.seed)
    device = get_device(args.device)
    task = RandomDotTask(cfg.task, device)
    model = SpikingBrain(cfg.net, cfg.task, seed=cfg.train.seed).to(device)
    init_state = model.snapshot()
    run = make_run_dir(method)

    # Öğrenmenin seyrini (3D görselleştirme için) birkaç anda kaydet
    snap_iters = {max(1, round(f * cfg.train.iters)) for f in SNAPSHOT_FRACTIONS}
    snapshots = [{"iter": 0, "w_in": init_state["w_in"], "w_rec": init_state["w_rec"]}]

    def on_step(it, m):
        if it in snap_iters:
            snapshots.append({"iter": it, "w_in": m.w_in.detach().cpu().clone(),
                              "w_rec": m.w_rec.detach().cpu().clone()})

    t0 = time.time()
    history = TRAINERS[method](model, task, cfg, gen, on_step=on_step)
    print(f"\n  Eğitim süresi: {time.time() - t0:.0f} sn")

    save_checkpoint(run / "model.pt", model, cfg, init_state, snapshots)
    save_json(run / "config.json", cfg.to_dict())
    save_json(run / "metrics.json", {"method": method, "history": history})
    paths = [run / "model.pt", plots.plot_learning_curve({method: history}, run / "learning_curve.png")]
    _done(paths)
    if not args.no_eval:
        evaluate_run(run, device, cfg.train.eval_trials)


def evaluate_run(run: Path, device, n_trials: int | None = None) -> dict:
    model, cfg, init_state = load_checkpoint(run, device)
    method = cfg.train.method
    n = n_trials or cfg.train.eval_trials
    _header(f"Değerlendirme: {plots.METHOD_NAMES[method]}  ({n} deneme x {2 * len(cfg.task.coherences)} koşul)")
    gen = set_seed(cfg.train.seed + 1000)  # eğitimde görülmemiş denemeler
    task = RandomDotTask(cfg.task, device)
    res = analysis.run_evaluation(model, task, cfg, n, gen)
    summary = analysis.summarize(res, cfg, model)

    pref_deg = (task.pref_dirs * 180 / torch.pi).cpu().numpy()
    init_prof = analysis.pool_weight_profiles(init_state["w_in"], init_state["mask_in"], model)
    final_prof = analysis.pool_weight_profiles(model.w_in.detach(), model.mask_in, model)
    cos_pref = task.cos_pref.cpu().numpy()
    summary["direction_selectivity_init"] = analysis.direction_selectivity(init_prof, cos_pref)
    summary["direction_selectivity"] = analysis.direction_selectivity(final_prof, cos_pref)

    # Tablo
    print(f"\n  {'Tutarlılık':>11s} {'Doğruluk':>9s} {'Karar süresi':>13s}")
    for c, a, rt in zip(summary["coherences"], summary["accuracy"], summary["decision_time_ms"]):
        rt_s = f"{rt:7.0f} ms" if rt is not None else "      -"
        print(f"  {c * 100:10.1f}% {a * 100:8.1f}% {rt_s:>13s}")
    w = summary["weibull"]
    print(f"\n  Psikometrik eşik (Weibull alpha): %{w['alpha'] * 100:.1f}   "
          f"(maymun ~%{analysis.MONKEY_REFERENCE['alpha'] * 100:.0f})")
    print(f"  Yön seçiciliği (ağırlıklar): {summary['direction_selectivity_init']:+.2f} -> "
          f"{summary['direction_selectivity']:+.2f}")
    print(f"  Ortalama hız: E {summary['rate_exc_hz']:.1f} Hz, I {summary['rate_inh_hz']:.1f} Hz")
    print(f"  Seçilen havuzun tepe hızı: {summary['peak_rate_max_hz']:.0f} Hz  (gerçek LIP ~60-70 Hz)")

    levels, acc, sem = analysis.accuracy_by_coherence(res)
    signed, p_right = analysis.p_right_by_signed_coherence(res)
    rts = analysis.decision_times(res, cfg, summary["decision_threshold_hz"])
    _, rt_mean, rt_sem = analysis.chronometric(res, rts)
    psth_levels, chosen, unchosen = analysis.psth_by_coherence(res)
    name = plots.method_name(method)
    paths = [
        plots.plot_psychometric(levels, acc, sem, w, signed, p_right, run / "psychometric.png",
                                plots.L("t_psycho", name)),
        plots.plot_chronometric(levels, rt_mean, rt_sem, run / "chronometric.png",
                                plots.L("t_chrono", name)),
        plots.plot_psth(psth_levels, chosen, unchosen, cfg, run / "lip_ramp.png",
                        plots.L("t_ramp", name)),
        plots.plot_weights(pref_deg, init_prof, final_prof, run / "weights.png",
                           plots.L("t_weights", name)),
        plots.plot_raster(res["example"], model, cfg, run / "raster_trained.png",
                          plots.L("t_raster_trained", name)),
        plots.plot_rate_hist(res["neuron_hz"], cfg, run / "rates_trained.png",
                             plots.L("t_rates_trained")),
    ]
    metrics = load_json(run / "metrics.json")
    metrics["eval"] = summary
    metrics["weight_profiles"] = {
        "pref_deg": pref_deg.tolist(),
        "init": {k: v.tolist() for k, v in init_prof.items()},
        "final": {k: v.tolist() for k, v in final_prof.items()},
    }
    save_json(run / "metrics.json", metrics)
    _done(paths + [run / "metrics.json"])
    return metrics


def cmd_evaluate(args) -> None:
    run = Path(args.run) if args.run else latest_run(args.method)
    if run is None or not (run / "model.pt").exists():
        sys.exit(f"Eğitilmiş model bulunamadı. Önce: python -m monkeybrain train --method {args.method}")
    evaluate_run(run, get_device(args.device), args.trials)


def cmd_compare(args) -> None:
    _header("Karşılaştırma: ML, biyolojik öğrenme ve maymun")
    runs = {}
    for method, given in (("ml", args.ml), ("bio", args.bio)):
        run = Path(given) if given else latest_run(method)
        if run is None:
            sys.exit(f"'{method}' çalışması bulunamadı. Önce: python -m monkeybrain train --method {method}")
        metrics = load_json(run / "metrics.json")
        if "eval" not in metrics:
            metrics = evaluate_run(run, get_device(args.device))
        prof = metrics["weight_profiles"]
        runs[method] = {
            "history": metrics["history"], "eval": metrics["eval"],
            "init_prof": {k: np.array(v) for k, v in prof["init"].items()},
            "final_prof": {k: np.array(v) for k, v in prof["final"].items()},
            "pref_deg": np.array(prof["pref_deg"]), "dir": run,
        }
        print(f"  {plots.METHOD_NAMES[method]:26s} <- {run}")

    ml, bio = runs["ml"]["eval"], runs["bio"]["eval"]
    from .analysis import MONKEY_REFERENCE, weibull
    ref = MONKEY_REFERENCE
    print(f"\n  {'Tutarlılık':>11s} {'ML':>8s} {'Biyolojik':>10s} {'Maymun*':>9s}")
    for c, a_ml, a_bio in zip(ml["coherences"], ml["accuracy"], bio["accuracy"]):
        print(f"  {c * 100:10.1f}% {a_ml * 100:7.1f}% {a_bio * 100:9.1f}% "
              f"{weibull(c, ref['alpha'], ref['beta']) * 100:8.1f}%")
    print("  * yaklaşık referans eğrisi, gerçek veri değil")
    print(f"\n  {'':26s} {'ML':>10s} {'Biyolojik':>10s}")
    rows = [
        ("Psikometrik eşik", f"%{ml['weibull']['alpha'] * 100:.1f}", f"%{bio['weibull']['alpha'] * 100:.1f}"),
        ("Yön seçiciliği", f"{ml['direction_selectivity']:+.2f}", f"{bio['direction_selectivity']:+.2f}"),
        ("Görülen deneme", f"{runs['ml']['history'][-1]['trials']}", f"{runs['bio']['history'][-1]['trials']}"),
        ("E hızı (Hz)", f"{ml['rate_exc_hz']:.1f}", f"{bio['rate_exc_hz']:.1f}"),
        ("Tepe havuz hızı (Hz)", f"{ml.get('peak_rate_max_hz', float('nan')):.0f}",
         f"{bio.get('peak_rate_max_hz', float('nan')):.0f}"),
    ]
    for name, a, b in rows:
        print(f"  {name:26s} {a:>10s} {b:>10s}")

    run = make_run_dir("compare")
    path = plots.plot_comparison(runs, ml["coherences"], runs["ml"]["pref_deg"], run / "comparison.png")
    save_json(run / "summary.json", {m: {"run": str(r["dir"]), "eval": r["eval"]} for m, r in runs.items()})
    _done([path, run / "summary.json"])


def cmd_brain3d(args) -> None:
    from .export3d import export_brain3d

    _header("3D beyin: eğitilmiş modellerin gerçek spike verisiyle")
    runs = {}
    for method, given in (("ml", args.ml), ("bio", args.bio)):
        run = Path(given) if given else latest_run(method)
        if run is not None:
            runs[method] = run
            print(f"  {plots.METHOD_NAMES[method]:26s} <- {run}")
    if not runs:
        sys.exit("Eğitilmiş model bulunamadı. Önce: python -m monkeybrain train --method bio")
    out_path = Path(args.out) if args.out else make_run_dir("brain3d") / "beyin_3d.html"
    path = export_brain3d(runs, out_path, site_url=args.site_url)
    _done([path])
    print()
    print("  Dosyaya çift tıklayarak tarayıcıda aç. (three.js ve yazı tipleri internetten yüklenir.)")


# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m monkeybrain",
        description="Makak korteksinden esinlenen spiking ağ: hareket yönü kararı öğrenimi.",
    )
    ap.add_argument("--lang", choices=["tr", "en"], default=os.environ.get("MONKEYBRAIN_LANG", "tr"),
                    help="şekillerin dili (tr/en); ör. python -m monkeybrain --lang en compare")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--seed", type=int, default=42, help="rastgelelik tohumu (tekrarlanabilirlik)")
        p.add_argument("--device", default="auto", help="auto | cpu | cuda")

    sub.add_parser("neuron", help="tek nöron tiplerini çiz")
    p = sub.add_parser("simulate", help="eğitilmemiş ağı simüle et")
    common(p)
    p = sub.add_parser("train", help="ağı eğit")
    common(p)
    p.add_argument("--method", choices=["ml", "bio"], required=True)
    p.add_argument("--iters", type=int, help="eğitim adımı sayısı")
    p.add_argument("--batch-size", type=int, help="adım başına deneme")
    p.add_argument("--quick", action="store_true", help="hızlı demo (~1-2 dk)")
    p.add_argument("--no-eval", action="store_true", help="eğitim sonrası değerlendirmeyi atla")
    p = sub.add_parser("evaluate", help="eğitilmiş modeli test et")
    common(p)
    p.add_argument("--run", help="çalışma klasörü (varsayılan: en yenisi)")
    p.add_argument("--method", choices=["ml", "bio"], default="ml")
    p.add_argument("--trials", type=int, help="koşul başına deneme")
    p = sub.add_parser("compare", help="ML ve biyolojik öğrenmeyi karşılaştır")
    common(p)
    p.add_argument("--ml", help="ML çalışma klasörü (varsayılan: en yenisi)")
    p.add_argument("--bio", help="biyolojik çalışma klasörü (varsayılan: en yenisi)")
    p = sub.add_parser("brain3d", help="etkileşimli 3D beyin sayfası oluştur")
    p.add_argument("--ml", help="ML çalışma klasörü (varsayılan: en yenisi)")
    p.add_argument("--bio", help="biyolojik çalışma klasörü (varsayılan: en yenisi)")
    p.add_argument("--out", help="HTML dosyasının yolu (varsayılan: outputs/brain3d_<tarih>/beyin_3d.html)")
    p.add_argument("--site-url", help="yayın adresi; verilirse sosyal medya önizleme etiketleri eklenir")
    return ap


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    plots.LANG = args.lang
    {"neuron": cmd_neuron, "simulate": cmd_simulate, "train": cmd_train,
     "evaluate": cmd_evaluate, "compare": cmd_compare, "brain3d": cmd_brain3d}[args.command](args)


if __name__ == "__main__":
    main()
