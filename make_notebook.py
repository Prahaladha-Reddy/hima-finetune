"""Builds finetune_preset_voices.ipynb: the guide as one notebook that runs unchanged on Kaggle and Colab (free T4).
Everything a reader changes sits in the first code cell (language, voice sources, which voices, names)."""
import json

REPO = "https://github.com/Prahaladha-Reddy/hima-finetune"
COLAB = "https://colab.research.google.com/github/Prahaladha-Reddy/hima-finetune/blob/main/finetune_preset_voices.ipynb"


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


cells = [
md(f"""
# Train your own voices for Hima TTS, on a free T4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB}) &nbsp; [GitHub]({REPO}) &nbsp;·&nbsp; [Hima TTS Telugu](https://huggingface.co/Prahaladha/hima-tts-nano-te) &nbsp;·&nbsp; [Hima TTS Hindi](https://huggingface.co/Prahaladha/hima-tts-nano-hi)

**Hima TTS nano** is a 62M-parameter text-to-speech model for Telugu and Hindi that runs on a laptop CPU and on phones. This notebook
gives it **new built-in voices**: a small adapter (about 5M numbers) trained on the frozen base model, the same way its released
voices were made. Everything runs on **one free T4** (Kaggle or Colab).

**One voice, 40 minutes of recordings: 5.4 minutes on a free Colab T4, from a blank machine to a finished voice.**

| Step | Time on a T4 (1 voice x 40 min) |
|---|---|
| Install + download the model | 0.2 min |
| Prepare | 0.6 min |
| Train | 4.4 min |
| Package | 0.2 min |

Run it as it is to train two ready-made expressive demo voices, or change the first cell to train your own.
"""),
md("""
## 1. Choose your voices
The only cell you need to edit.

**`SOURCES`**: where the voices come from. Any mix of:

| Source | Example |
|---|---|
| Ready-made demo voices | `"hf:Prahaladha/hima-finetune-demo-voices/te"` (Telugu: Anjali, Kiran) · `"hf:Prahaladha/hima-finetune-demo-voices/hi"` (Hindi: Priya, Arjun) |
| Your Hugging Face dataset | `"hf:your-name/your-voices"` |
| A folder (Kaggle input, Drive, upload) | `"/kaggle/input/my-voices"` · `"/content/drive/MyDrive/my-voices"` |
| A zip on the web | `"https://example.com/my-voices.zip"` |

Each source is **clips + text**: a `metadata.csv` (`path|text|voice`, one line per clip) next to the clips. The demo voices come
with their text, so nothing extra runs.

- **Recordings but no text?** Put them in one folder per voice (any length, any format) and add that folder to `SOURCES`: they are
  cut at the pauses and written down by [Hima-ASR](https://huggingface.co/Prahaladha/hima-asr). Install with
  `hima-finetune[transcribe]` in step 2 for this.
- **No recordings at all?** Generate a synthetic voice with [Maya 2 Native](https://huggingface.co/maya-research/maya2-native)
  (see the last section) and add the result to `SOURCES`.

**Good voices:** 30-45 minutes each, one speaker, clean audio (no music, no other voices), delivered the way you want it to sound:
the model copies the style of its data, so lively recordings give a lively voice. Only use voices you have the right to use.
"""),
code("""
LANG    = "te"                                          # "te" Telugu or "hi" Hindi
SOURCES = ["hf:Prahaladha/hima-finetune-demo-voices/te"]       # one or more sources (see the table above)
VOICES  = []                                            # [] = every voice found, or e.g. ["Anjali"]
RENAME  = {}                                            # optional new names, e.g. {"Anjali": "Meera"}

WORK, OUT = "work", "my-model"                          # Colab: put WORK on Google Drive to survive a disconnect
"""),
md("""
## 2. Setup
**Kaggle:** Settings → Accelerator → **GPU T4**, Internet **on**. &nbsp; **Colab:** Runtime → Change runtime type → **T4 GPU**.
"""),
code(f"""
import os, sys, json, subprocess
!pip install -q "hima-finetune @ git+{REPO}"          # recordings without text? use "hima-finetune[transcribe] @ git+..." instead
from huggingface_hub import snapshot_download
BASE = os.path.abspath(snapshot_download(f"Prahaladha/hima-tts-nano-{{LANG}}", local_dir=f"hima-tts-nano-{{LANG}}"))
sys.path.insert(0, BASE)

def run(cmd):
    \"\"\"Run a step, stream its output, and stop here if it fails (a plain `!command` would carry on silently).\"\"\"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="")
    if p.wait():
        raise RuntimeError(f"step failed (exit {{p.returncode}}): {{cmd}}")

!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"""),
md("""
## 3. Gather the voices
Downloads every source, transcribes any recordings that have no text, keeps the voices you chose, and prints the minutes per voice.
"""),
code("""
DATA = f"{WORK}/data"
args = " ".join(f"--source '{s}'" for s in SOURCES)
args += f" --voices {','.join(VOICES)}" if VOICES else ""
args += f" --rename {','.join(f'{k}={v}' for k, v in RENAME.items())}" if RENAME else ""
run(f"hima-finetune get {args} --lang {LANG} --out {DATA}")
"""),
md("""
## 4. Prepare
Encodes every clip with the model's own codec, takes a voice print of each clip, and holds out about 3% of every voice to check
training on clips it never saw.
"""),
code("""
run(f"hima-finetune prepare --base {BASE} --data {DATA} --work {WORK}")
"""),
md("""
## 5. Train
Steps are set from your data (about 60 passes over it). The adapter is saved every 500 steps with the full training state.
**Disconnected?** Run this cell again with `--resume` added. **Out of memory?** Add `--batch_frames 3000`.
"""),
code("""
run(f"hima-finetune train --base {BASE} --work {WORK}")

import matplotlib.pyplot as plt                       # held-out loss: falls from step 0 (the base model) and levels off
v = [json.loads(l) for l in open(f"{WORK}/run/progress.jsonl") if '"valid"' in l]
plt.figure(figsize=(7, 2.6)); plt.plot([r["step"] for r in v], [r["loss"] for r in v], "o-", color="#16a34a")
plt.xlabel("step"); plt.ylabel("held-out loss"); plt.grid(alpha=.3); plt.show()
"""),
md("""
## 6. Listen: the real voice, then yours
For every voice: a real recording that training never saw, then the trained voice on new sentences, including some with feeling.
"""),
code("""
run(f"hima-finetune export --base {BASE} --work {WORK} --out {OUT}")
import soundfile as sf
from IPython.display import Audio, Markdown, display
from hima_tts import TTS
tts = TTS(OUT)
LINES = {"te": ["ఈ రోజు సాయంత్రం మనం కలిసి సినిమాకి వెళ్దామా?", "అరె! నువ్వు నిజంగా గెలిచావా? ఇది అద్భుతం!",
                "ఎన్నిసార్లు చెప్పాలి నీకు? ఇదే చివరిసారి!", "ఆయన ఇక రారని తెలిసి నా మనసు విరిగిపోయింది."],
         "hi": ["आज शाम हम साथ में फ़िल्म देखने चलें?", "अरे वाह! तुमने सच में कर दिखाया!",
                "कितनी बार कहूँ तुमसे? ये आख़िरी बार है!", "मुझे यकीन नहीं हो रहा कि वो अब कभी नहीं आएगा।"]}[LANG]
held = [json.loads(l) for l in open(f"{WORK}/enc/index.jsonl", encoding="utf-8")]
for name in tts.voices:
    display(Markdown(f"### {name}"))
    real = next((r for r in held if r["voice"] == name and r["split"] == "holdout"), None)
    if real:
        x, sr = sf.read(real["path"] if os.path.isabs(real["path"]) else os.path.join(DATA, real["path"]))
        display(Markdown(f"**Real recording** (never seen in training): *{real['text_raw']}*")); display(Audio(x, rate=sr))
    for i, line in enumerate(LINES):
        display(Markdown(f"**Trained voice:** *{line}*")); display(Audio(tts.say(line, voice=name, seed=i), rate=24000))
"""),
md("""
## 7. Use and share
`my-model` is a normal Hima TTS folder:
```python
from hima_tts import TTS
tts = TTS("my-model")
tts.save(tts.say("మీ వాక్యం ఇక్కడ", voice="Anjali"), "out.wav")
```
Your voices are two small files, `my-model/voices/adapter.safetensors` and `voices.safetensors`. Download them, or push the folder
to the Hugging Face Hub:
"""),
code("""
# !zip -qr my-model.zip my-model          # then download it from the file browser
# from huggingface_hub import HfApi        # needs a write token: from huggingface_hub import login; login()
# HfApi().upload_folder(folder_path=OUT, repo_id="your-name/my-hima-voices", repo_type="model", create_repo=True, private=True)
"""),
md("""
## No recordings? Distill from Maya 2 Native
[Maya 2 Native](https://huggingface.co/maya-research/maya2-native) is an open 3B text-to-speech model (Apache-2.0) with 20
built-in voices in 12 languages. Have it read a few thousand sentences in the voices you like and train Hima on them: the same
voices from a model ~50x smaller.

The teacher needs an **Ampere-or-newer GPU** (L4, A10G, A100, H100; 12 GB+) and its own environment, so this part does not run
on the T4. On that machine:
```bash
pip install "maya3 @ git+https://github.com/MayaResearch/maya2-native" --extra-index-url https://download.pytorch.org/whl/cu128
pip install --no-deps "hima-finetune @ git+https://github.com/Prahaladha-Reddy/hima-finetune" pyarrow
hima-finetune distill --voices Vaidehi,Sreekar --lang te --minutes 45 --out distilled
```
Then add the `distilled` folder to `SOURCES` above. Licences: the teacher's weights are Apache-2.0, its codec (Kyutai Mimi) is
CC-BY-4.0, and the default sentences come from Wikipedia (CC BY-SA 4.0).

**Demo voices:** a subset of [Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) by AI4Bharat (CC-BY-4.0), conversational and
emotional speech only.
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("finetune_preset_voices.ipynb", "w"), ensure_ascii=False, indent=1)
print("ok finetune_preset_voices.ipynb", len(cells), "cells")
