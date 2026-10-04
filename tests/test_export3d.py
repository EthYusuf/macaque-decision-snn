import json
import re

import torch

from monkeybrain.export3d import export_brain3d
from monkeybrain.network import SpikingBrain
from monkeybrain.utils import save_checkpoint, save_json


def _fake_run(tmp_path, cfg, method):
    run = tmp_path / method
    run.mkdir()
    cfg.train.method = method
    cfg.train.iters = 4
    model = SpikingBrain(cfg.net, cfg.task, seed=0)
    init = model.snapshot()
    snaps = [{"iter": 0, "w_in": init["w_in"], "w_rec": init["w_rec"]},
             {"iter": 4, "w_in": init["w_in"] * 1.1, "w_rec": init["w_rec"]}]
    save_checkpoint(run / "model.pt", model, cfg, init, snaps)
    history = [{"iter": i, "trials": i * 4, "accuracy": 0.5} for i in range(1, 5)]
    save_json(run / "metrics.json", {"method": method, "history": history})
    return run


def test_export_embeds_valid_data(tmp_path, small_cfg):
    runs = {"bio": _fake_run(tmp_path, small_cfg, "bio")}
    out = export_brain3d(runs, tmp_path / "beyin_3d.html")
    html = out.read_text(encoding="utf-8")
    m = re.search(r"window\.BRAIN_DATA = (\{.*?\});</script>", html, re.S)
    assert m, "veri sayfaya gömülmedi"
    data = json.loads(m.group(1))
    meta, bio = data["meta"], data["methods"]["bio"]
    assert meta["T"] == small_cfg.task.n_steps
    assert len(meta["mt"][meta["coherences"][0]]) == small_cfg.task.n_mt
    assert [s["iter"] for s in bio["stages"]] == [0, 4]
    n_conn = len(bio["connections"]["src"])
    assert all(len(s["weights"]) == n_conn for s in bio["stages"])
    sim = bio["stages"][0]["sims"][meta["coherences"][0]]
    assert len(sim["lip"]) == small_cfg.net.n_total
    assert sim["choice"] in (0, 1)


def test_same_stimulus_across_stages(tmp_path, small_cfg):
    """Aşamalar arasında tek fark ağırlıklar olmalı: aynı ağırlık = aynı spike'lar."""
    run = _fake_run(tmp_path, small_cfg, "ml")
    ckpt = torch.load(run / "model.pt", weights_only=False)
    ckpt["snapshots"][1]["w_in"] = ckpt["snapshots"][0]["w_in"].clone()
    torch.save(ckpt, run / "model.pt")
    out = export_brain3d({"ml": run}, tmp_path / "b.html")
    data = json.loads(re.search(r"window\.BRAIN_DATA = (\{.*?\});</script>", out.read_text(encoding="utf-8"), re.S).group(1))
    stages = data["methods"]["ml"]["stages"]
    assert stages[0]["sims"] == stages[1]["sims"]
