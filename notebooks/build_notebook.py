"""Kaggle notebook'unu üretir: python notebooks/build_notebook.py

Notebook'u elle düzenlemek yerine buradan üretmek, hücre içeriklerini sürüm kontrolünde
okunabilir tutar.
"""

import json
from pathlib import Path

REPO = "https://github.com/EthYusuf/macaque-decision-snn"
DEMO = "https://ethyusuf.github.io/macaque-decision-snn/"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md(f"""
# Anatomy of a Decision: training a spiking model of macaque MT and LIP

**264 spiking neurons modeled on the macaque brain learn the classic random-dot motion task, first with machine learning and then with a brain-style, dopamine-gated learning rule. Both are put through the same tests used on real monkeys.**

[GitHub repository]({REPO}) · [Live interactive 3D demo]({DEMO})

![3D view of the model]({REPO}/raw/main/docs/images/demo.gif)

### What this notebook does, from scratch
1. Clones the repository and runs its **24 tests** (unit tests plus an end-to-end run of the pipeline).
2. Shows the building block: **spiking cortical neurons**.
3. Trains the network with **surrogate-gradient backpropagation through time** (machine learning).
4. Trains the same network with **reward-modulated STDP** (dopamine as a reward-prediction error, closer to how brains learn).
5. Runs both through the **monkey's tests**: psychometric and chronometric curves and LIP ramping activity.
6. Exports the **interactive 3D brain** built from the spikes of the models trained here.

Runtime: about 20 minutes on a CPU session. Set `QUICK = True` for a ~5 minute demo with shorter training.

> **On Kaggle:** turn on *Settings → Internet* so the notebook can clone the repository. No GPU is needed. The same notebook also runs on Google Colab or locally.
"""),
    code("""
QUICK = False   # True: shorter training (~5 min total) for a fast demo
SEED = 42
REPO = "https://github.com/EthYusuf/macaque-decision-snn.git"
"""),
    md("## 1 · Setup"),
    code("""
import importlib.util, os, platform, shutil, subprocess, sys
from pathlib import Path

ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ          # set by Kaggle in every notebook session
WORK = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()  # notebook output folder
SRC = Path("/tmp/macaque-decision-snn")
if not SRC.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(SRC)], check=True)
os.chdir(SRC)
sys.path.insert(0, str(SRC))   # import straight from the clone; Kaggle already has torch, numpy, scipy
for pkg in ("torch", "scipy", "matplotlib", "pandas", "pytest"):
    if importlib.util.find_spec(pkg) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

import torch
from IPython.display import HTML, Image, Markdown, display
from monkeybrain.__main__ import main
from monkeybrain.utils import latest_run, load_json

def run(*args):
    \"\"\"Run a CLI command in English, e.g. run("train", "--method", "ml").\"\"\"
    main(["--lang", "en", *args])

def show(*paths, width=None):
    for p in paths:
        display(Image(filename=str(p), width=width))

def newest(prefix):
    \"\"\"Newest output folder of a command, e.g. newest("compare").\"\"\"
    return sorted(Path("outputs").glob(f"{prefix}_*"))[-1]

print(f"Python {platform.python_version()} · PyTorch {torch.__version__} · "
      f"device: {'cuda' if torch.cuda.is_available() else 'cpu'} · CPUs: {os.cpu_count()}")
"""),
    md("""
## 2 · Run the test suite

24 tests cover the neuron models, the task, Dale's law, the sign of the STDP updates, the analysis code, the 3D export and an end-to-end run of the whole command-line pipeline (99 % line coverage).
"""),
    code("""
result = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
print(result.stdout[-1200:])
assert result.returncode == 0, "tests failed"
"""),
    md("""
## 3 · The building block: a spiking neuron

A neuron integrates input current until its membrane potential crosses a threshold, fires a spike and resets. Below, five classic cortical cell types (Izhikevich 2003) receive the same current step and respond in very different ways. The network uses the simpler **adaptive leaky integrate-and-fire** neuron at the bottom: fast, differentiable with a surrogate gradient, and adapting like a pyramidal cell.
"""),
    code("""
run("neuron")
show(newest("neuron") / "neuron_types.png", width=900)
"""),
    md("""
## 4 · Train with machine learning (surrogate gradients)

A spike is a step function, so its true derivative is zero almost everywhere and backpropagation stalls. The **surrogate-gradient** trick keeps the hard spike in the forward pass and uses a smooth fast-sigmoid derivative in the backward pass (Neftci, Mostafa & Zenke 2019). The error is then propagated **back through all 700 time steps** of a trial (BPTT), and every synapse is updated with Adam. A firing-rate penalty keeps neurons in a biological range, and Dale's law is re-imposed after each step.
"""),
    code("""
quick = ["--quick"] if QUICK else []
run("train", "--method", "ml", "--seed", str(SEED), *quick)
ml_run = latest_run("ml")
show(ml_run / "learning_curve.png", width=640)
show(ml_run / "psychometric.png", width=950)
"""),
    md("""
## 5 · Train like a brain (dopamine-modulated STDP)

Brains cannot backpropagate errors through time. They seem to use a **three-factor rule** instead:

1. **Pre- and post-synaptic spike timing** (STDP) leaves an *eligibility trace* at each synapse; the weight does not change yet.
2. At the end of the trial the monkey gets juice, or not. **Dopamine** neurons signal the *reward-prediction error*: reward minus expected reward (Schultz et al. 1997).
3. The weight change is `Δw = η · dopamine · eligibility`.

The first version of this rule learned nothing: the losing pool kept firing a little and stole credit for the winner's rewards. Using each neuron's *deviation from its own mean activity* (Legenstein et al. 2010), together with a strong winner-take-all circuit, fixed it.
"""),
    code("""
run("train", "--method", "bio", "--seed", str(SEED), *quick)
bio_run = latest_run("bio")
show(bio_run / "learning_curve.png", width=640)
show(bio_run / "psychometric.png", width=950)
"""),
    md("""
## 6 · Put both models through the monkey's tests

The same measurements used in macaque experiments:

- **Psychometric curve:** accuracy against motion coherence, with a Weibull fit. Its threshold α is the coherence giving about 82 % correct.
- **Chronometric curve:** do harder decisions take longer?
- **Learned wiring:** which MT neurons does each LIP pool listen to?

The dashed curve is an *approximate* reference for typical macaque performance, not a dataset.
"""),
    code("""
run("compare", "--ml", str(ml_run), "--bio", str(bio_run))
cmp_dir = newest("compare")
show(cmp_dir / "comparison.png", width=1000)
"""),
    code("""
import pandas as pd
from monkeybrain.analysis import MONKEY_REFERENCE, weibull

ml_eval, bio_eval = load_json(ml_run / "metrics.json")["eval"], load_json(bio_run / "metrics.json")["eval"]
table = pd.DataFrame({
    "coherence (%)": [c * 100 for c in ml_eval["coherences"]],
    "ML accuracy (%)": [a * 100 for a in ml_eval["accuracy"]],
    "R-STDP accuracy (%)": [a * 100 for a in bio_eval["accuracy"]],
    "monkey, approx. (%)": [weibull(c, MONKEY_REFERENCE["alpha"], MONKEY_REFERENCE["beta"]) * 100 for c in ml_eval["coherences"]],
}).round(1)
display(table)
display(Markdown(
    f"**Psychometric threshold:** ML {ml_eval['weibull']['alpha'] * 100:.1f} % · "
    f"R-STDP {bio_eval['weibull']['alpha'] * 100:.1f} % · monkey ~{MONKEY_REFERENCE['alpha'] * 100:.0f} %  \\n"
    f"**Peak firing rate of the chosen LIP pool:** ML {ml_eval['peak_rate_max_hz']:.0f} Hz · "
    f"R-STDP {bio_eval['peak_rate_max_hz']:.0f} Hz · real LIP ~60–70 Hz"
))
"""),
    md("""
## 7 · Did LIP ramping emerge?

Roitman & Shadlen (2002) found that LIP neurons **ramp up** while the monkey watches the dots, faster for stronger motion, and that neurons for the other choice are suppressed. Nothing in either training rule asks for a ramp: the network is only rewarded for the right answer. Solid lines are the chosen pool, dashed lines the other pool.
"""),
    code("""
show(ml_run / "lip_ramp.png", bio_run / "lip_ramp.png", width=800)
show(bio_run / "weights.png", width=800)
show(bio_run / "raster_trained.png", width=850)
"""),
    md(f"""
## 8 · Explore it in 3D

The 3D page replays real spikes from the models trained above at six snapshots during training. Spikes travel from MT to LIP, the choice pools race, and at the end of a trial a dopamine wave (biological rule) or a backpropagated error (ML) shows how learning assigns credit.

The cell below writes the page to the notebook output (`results/brain3d.html`). Download it and open it in a browser, or use the hosted version: **[{DEMO}]({DEMO})**.
"""),
    code("""
RESULTS = WORK / "results"
RESULTS.mkdir(exist_ok=True)
run("brain3d", "--ml", str(ml_run), "--bio", str(bio_run), "--out", str(RESULTS / "brain3d.html"))
for d in (ml_run, bio_run, cmp_dir):
    for png in d.glob("*.png"):
        shutil.copy(png, RESULTS / f"{d.name.split('_')[0]}_{png.name}")
print(sorted(p.name for p in RESULTS.iterdir()))
display(HTML('<a href="https://ethyusuf.github.io/macaque-decision-snn/" target="_blank">▶ Open the live 3D demo</a>'))
"""),
    md(f"""
## Takeaways

- A biologically grounded spiking circuit learns the task with **both** learning rules, and both independently discover the same wiring: each LIP pool listens to the MT neurons tuned to its direction.
- The ML-trained network reaches monkey-like sensitivity. The dopamine rule is less sensitive but uses only local, biologically plausible signals.
- **LIP-like ramping emerges** without being trained for.

**Limitations.** These are 264 simulated neurons, not recorded data. V1 is abstracted, the task has a fixed duration, and the monkey curve is an approximate reference. More detail is in the [repository README]({REPO}#limitations).

*Code, tests, pretrained models and a step-by-step tutorial: [{REPO}]({REPO})*
"""),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    out = Path(__file__).with_name("macaque-decision-snn.ipynb")
    out.write_text(json.dumps(NOTEBOOK, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{out} ({len(CELLS)} cells)")
