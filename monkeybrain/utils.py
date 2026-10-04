"""Yardımcılar: tekrarlanabilirlik, çalışma klasörleri, kayıt/yükleme."""

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"
PRETRAINED = PROJECT_ROOT / "pretrained"


def set_seed(seed: int) -> torch.Generator:
    """Tüm rastgelelik kaynaklarını sabitler; aynı seed = aynı sonuç."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_run_dir(prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = OUTPUTS / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_run(prefix: str) -> Path | None:
    """outputs/ altında verilen önekli en yeni (checkpoint içeren) klasör.

    Henüz hiç eğitim yapılmadıysa depodaki hazır modele (pretrained/<önek>) düşer.
    """
    runs = sorted(p for p in OUTPUTS.glob(f"{prefix}_*") if (p / "model.pt").exists()) if OUTPUTS.exists() else []
    if runs:
        return runs[-1]
    pretrained = PRETRAINED / prefix
    return pretrained if (pretrained / "model.pt").exists() else None


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, model, cfg, init_state: dict, snapshots: list | None = None) -> None:
    torch.save(
        {"state": model.state_dict(), "init_state": init_state, "config": cfg.to_dict(),
         "snapshots": snapshots or []},
        path,
    )


def load_snapshots(run_dir: Path) -> list[dict]:
    """Eğitim sırasında saklanan ağırlık anlık görüntüleri (yoksa boş liste)."""
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    return ckpt.get("snapshots", [])


def load_checkpoint(run_dir: Path, device: torch.device):
    """Bir çalışma klasöründen modeli ve konfigürasyonu geri yükler."""
    from .config import Config
    from .network import SpikingBrain

    ckpt = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = SpikingBrain(cfg.net, cfg.task).to(device)
    model.load_state_dict(ckpt["state"])
    return model, cfg, ckpt["init_state"]
