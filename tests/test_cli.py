"""Uçtan uca entegrasyon testi: CLI'nin tüm hattı, kısaltılmış eğitimle.

neuron -> simulate -> train (ml, bio) -> evaluate -> compare -> brain3d
Çıktılar geçici bir klasöre yazılır; depodaki outputs/ ve pretrained/ kullanılmaz.
"""

import json
import re

import pytest

from monkeybrain import i18n, utils
from monkeybrain.__main__ import main


@pytest.fixture
def isolated_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(utils, "PRETRAINED", tmp_path / "no-pretrained")
    yield tmp_path / "outputs"
    i18n.set_lang("tr")


def _one(outputs, prefix):
    runs = sorted(outputs.glob(f"{prefix}_*"))
    assert len(runs) == 1, f"{prefix}: {runs}"
    return runs[0]


def test_full_pipeline_end_to_end(isolated_outputs, capsys):
    out = isolated_outputs
    main(["--lang", "en", "neuron"])
    main(["--lang", "en", "simulate"])
    for method in ("ml", "bio"):
        main(["--lang", "en", "train", "--method", method, "--quick", "--iters", "2", "--batch-size", "4"])
    main(["--lang", "en", "compare"])
    main(["--lang", "en", "brain3d", "--out", str(out / "brain3d.html"), "--site-url", "https://example.org/demo"])

    assert (_one(out, "neuron") / "neuron_types.png").exists()
    assert (_one(out, "simulate") / "raster_untrained.png").exists()
    for method in ("ml", "bio"):
        run = _one(out, method)
        for name in ("model.pt", "learning_curve.png", "psychometric.png", "chronometric.png",
                     "lip_ramp.png", "weights.png", "raster_trained.png"):
            assert (run / name).exists(), f"{method}: {name} yok"
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        assert len(metrics["history"]) == 2
        assert 0.0 <= metrics["eval"]["overall_accuracy"] <= 1.0
        assert len(metrics["eval"]["accuracy"]) == 6
    assert (_one(out, "compare") / "comparison.png").exists()

    html = (out / "brain3d.html").read_text(encoding="utf-8")
    assert 'og:image" content="https://example.org/demo/images/hero.jpg"' in html
    data = json.loads(re.search(r"window\.BRAIN_DATA = (\{.*?\});</script>", html, re.S).group(1))
    assert set(data["methods"]) == {"ml", "bio"}
    assert [s["iter"] for s in data["methods"]["ml"]["stages"]] == [0, 1, 2]

    text = capsys.readouterr().out
    assert "Training: ML (surrogate gradient)" in text       # --lang en konsol çıktısı
    assert "Comparison: ML, biological learning and the monkey" in text


def test_missing_model_gives_helpful_error(isolated_outputs):
    with pytest.raises(SystemExit) as exc:
        main(["--lang", "en", "evaluate", "--method", "bio"])
    assert "python -m monkeybrain train --method bio" in str(exc.value)
