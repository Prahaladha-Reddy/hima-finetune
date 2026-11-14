"""Builds finetune_preset_voices.ipynb: the guide, as one notebook that runs unchanged on Kaggle and Colab (free T4).
    python make_notebook.py                      # the public notebook
    python make_notebook.py --test DATA PKG      # a test copy: data from DATA, the package from PKG, a short training that is
                                                 # interrupted and resumed
"""
import json
import sys

REPO = "https://github.com/Prahaladha-Reddy/hima-finetune"
COLAB = "https://colab.research.google.com/github/Prahaladha-Reddy/hima-finetune/blob/main/finetune_preset_voices.ipynb"
TEST = "--test" in sys.argv
if TEST:
    T_DATA, T_PKG = sys.argv[sys.argv.index("--test") + 1: sys.argv.index("--test") + 3]


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


install = f'pip install -q {T_PKG}' if TEST else f'pip install -q "hima-finetune @ git+{REPO}"'
data_line = f'INPUT = {T_DATA}' if TEST else 'INPUT = "/kaggle/input/my-voices"     # Colab: e.g. "/content/my-voices" (upload a zip and unzip it, or use Google Drive)'

train_cell = code("""
run(f"hima-finetune train --base {BASE} --work {WORK}")
""")
if TEST:  # train 400 steps, kill it at step 250 (after the save at 200), then resume: it must continue from 200
    train_cell = code("""
import subprocess as sp
p = sp.Popen(["hima-finetune", "train", "--base", BASE, "--work", WORK, "--steps", "400", "--save_every", "200", "--valid_every", "200"],
             stdout=sp.PIPE, stderr=sp.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="")
    if line.startswith("step 250/"):
        p.kill(); break
p.wait(); print("TEST: killed at step 250")
run(f"hima-finetune train --base {BASE} --work {WORK} --steps 400 --save_every 200 --valid_every 200 --resume")
ev = [json.loads(l) for l in open(f"{WORK}/run/progress.jsonl")]
assert any(e["type"] == "resume" and e["step"] == 200 for e in ev), "resume did not continue from step 200"
print("TEST_RESUME_OK")
""")

cells = [
md(f"""
# Train your own voices for Hima TTS, on a free T4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB}) &nbsp; [GitHub]({REPO}) &nbsp;·&nbsp; [Hima TTS Telugu](https://huggingface.co/Prahaladha/hima-tts-nano-te) &nbsp;·&nbsp; [Hima TTS Hindi](https://huggingface.co/Prahaladha/hima-tts-nano-hi)

**Hima TTS nano** is a 62M-parameter text-to-speech model for Telugu and Hindi that runs on a laptop CPU and on phones. This notebook
gives it **new built-in voices**: a small adapter (about 5M numbers) trained on the frozen base model, the same way its released
voices were made. Everything runs on **one free T4** (Kaggle or Colab).

**Bring your voices in any of three ways:**

| You have | What happens |
|---|---|
| **Recordings + their text** (`metadata.csv`) | straight to training |
| **Recordings only** (any length, any format) | cut into sentences at the pauses and written down by [Hima-ASR](https://huggingface.co/Prahaladha/hima-asr), on this T4 |
| **No recordings** | distill voices from [Maya 2 Native](https://huggingface.co/maya-research/maya2-native) (3B, Apache-2.0): see the last section. Generating the clips needs an L4/A100-class GPU; training then runs here |

**How long, on a T4**

| | Prepare | Train |
|---|---|---|
| 2 voices x 40 min | 2 min | 9 min |
| 6 voices x 2 h | 7 min | 90 min |

We retrained the 6 released Telugu voices this way on one free Kaggle T4: they match the released voices on speaker similarity
(0.622 vs 0.609; the base model alone scores 0.572).
"""),
md("""
## 1. Setup
**Kaggle:** Settings → Accelerator → **GPU T4**, Internet **on**. &nbsp; **Colab:** Runtime → Change runtime type → **T4 GPU**.
"""),
code(f"""
LANG = "te"                     # "te" Telugu or "hi" Hindi: the language your voices speak

import os, sys, json, subprocess
!{install}
from huggingface_hub import snapshot_download
BASE = os.path.abspath(snapshot_download(f"Prahaladha/hima-tts-nano-{{LANG}}", local_dir=f"hima-tts-nano-{{LANG}}"))
sys.path.insert(0, BASE)

def run(cmd):
    \"\"\"Run a step, stream its output, and stop the notebook if it fails (a plain `!command` would carry on silently).\"\"\"
    if not globals().get("READY", True):
        print("skipped: add your voices in step 2 first"); return
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="")
    if p.wait():
        raise RuntimeError(f"step failed (exit {{p.returncode}}): {{cmd}}")

!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"""),
md("""
## 2. Your voices
Point `INPUT` at a folder in **one** of these two layouts:

**A. Recordings + text.** A `metadata.csv` with one line per clip: `file | exact words spoken | voice name`
```
path|text|voice
clips/meera_001.wav|నమస్కారం, ఈ రోజు ఎలా ఉన్నారు?|Meera
clips/ravi_001.wav|ఆ సినిమా చాలా బాగుంది.|Ravi
```
**B. Recordings only.** One folder per voice; the folder name becomes the voice name. Any length, any format.
```
my-voices/Meera/interview.mp3
my-voices/Ravi/podcast_part1.wav
```

**What makes a good voice**
- **30-45 minutes per voice.** Under 20 minutes works, but the voice comes out weaker.
- One speaker per voice, clean audio: no music, no other voices, little echo.
- Layout A: clips of 2-20 s, and the text must match the audio word for word (English words in English letters are fine).
- Only use voices you have the right to use, with the speaker's consent.

**Kaggle:** Add Input → Upload → your folder. **Colab:** upload a zip and `!unzip` it, or mount Google Drive.
"""),
code(f"""
{data_line}
WORK = "work"          # Colab: put this on Google Drive (e.g. "/content/drive/MyDrive/hima-work") to survive a disconnect
OUT = "my-model"

AUDIO = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".webm", ".mp4")
HAS_TEXT = os.path.exists(f"{{INPUT}}/metadata.csv")
HAS_AUDIO = os.path.isdir(INPUT) and any(f.lower().endswith(AUDIO) for _, _, fs in os.walk(INPUT) for f in fs)
READY = HAS_TEXT or HAS_AUDIO
DATA = INPUT if HAS_TEXT else f"{{WORK}}/transcribed"
if HAS_TEXT:
    print("layout A: recordings + text"); run(f"head -4 '{{INPUT}}/metadata.csv'")
elif HAS_AUDIO:
    print("layout B: recordings only. Voices:", sorted(d for d in os.listdir(INPUT) if os.path.isdir(f"{{INPUT}}/{{d}}")))
else:
    print(f"No voices found in {{INPUT}}. Add your folder (step 2), set INPUT, and run again.")
"""),
md("""
## 3. Transcribe (recordings only)
Skipped if you have a `metadata.csv`. Cuts every recording into 3-15 s clips at the pauses, writes down each clip with
**Hima-ASR** (a 0.46B speech recognizer for 11 Indian languages and English), and drops clips that are silent, too short, or
whose text does not fit the audio. Read a few lines afterwards: training learns whatever the text says.
"""),
code("""
if READY and not HAS_TEXT:
    !pip install -q "transformers>=5.13"
    run(f"hima-finetune transcribe --audio '{INPUT}' --out '{DATA}' --lang {LANG}")
    print("\\n".join(open(f"{DATA}/metadata.csv", encoding="utf-8").read().splitlines()[:6]))
"""),
md("""
## 4. Prepare
Encodes every clip with the model's own codec, takes a voice print of each clip, and holds out about 3% of every voice to
check training on clips it never saw. It prints the minutes per voice and anything it skipped, with the reason.
"""),
code("""
run(f"hima-finetune prepare --base {BASE} --data '{DATA}' --work {WORK}")
"""),
md("""
## 5. Train
The number of steps is set from your data: about 60 passes over it, what the released voices got. The adapter is saved every
500 steps. **Disconnected?** Run this cell again with `--resume` added: it continues from the last save (keep `WORK` on Google
Drive in Colab). Out of memory? Add `--batch_frames 3000`.
"""),
train_cell,
code("""
# held-out loss: it should fall from step 0 (the base model on your voices) and level off
if READY:
    import matplotlib.pyplot as plt
    v = [json.loads(l) for l in open(f"{WORK}/run/progress.jsonl") if '"valid"' in l]
    plt.figure(figsize=(7, 3)); plt.plot([r["step"] for r in v], [r["loss"] for r in v], "o-", color="#16a34a")
    plt.xlabel("step"); plt.ylabel("held-out loss"); plt.grid(alpha=.3); plt.show()
"""),
md("""
## 6. Package and listen
Builds `my-model`: the base model with your voices in `voices/`, exactly like the released ones. Optional: `--genders Meera=female,Ravi=male`.
"""),
code("""
run(f"hima-finetune export --base {BASE} --work {WORK} --out {OUT}")
if READY:
    from IPython.display import Audio, display
    from hima_tts import TTS
    tts = TTS(OUT)
    lines = {"te": ["ఈ రోజు సాయంత్రం మనం కలిసి సినిమాకి వెళ్దామా?", "మీ ఆర్డర్ ఇంకా పది నిమిషాల్లో వస్తుంది."],
             "hi": ["आज शाम हम साथ में फ़िल्म देखने चलें?", "आपका ऑर्डर दस मिनट में पहुँच जाएगा।"]}[LANG]
    for name in tts.voices:
        print(name)
        for i, line in enumerate(lines):
            display(Audio(tts.say(line, voice=name, seed=i), rate=24000))
"""),
md("""
## 7. Use and share
`my-model` is a normal Hima TTS folder:
```python
from hima_tts import TTS
tts = TTS("my-model")
tts.save(tts.say("మీ వాక్యం ఇక్కడ", voice="Meera"), "out.wav")
```
Your voices are two small files: `my-model/voices/adapter.safetensors` and `voices.safetensors`. Download them, or push the folder
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
built-in voices in 12 languages. Have it read a few thousand sentences in the voices you like, and train Hima on them: the same
voices from a model ~50x smaller.

The teacher needs an **Ampere-or-newer GPU** (L4, A10G, A100, H100; 12 GB+) and its own environment, so this part does not run
on the T4. On that machine:
```bash
pip install "maya3 @ git+https://github.com/MayaResearch/maya2-native" --extra-index-url https://download.pytorch.org/whl/cu128
pip install --no-deps "hima-finetune @ git+https://github.com/Prahaladha-Reddy/hima-finetune" pyarrow
hima-finetune distill --voices Vaidehi,Sreekar --lang te --minutes 45 --out distilled
```
It reads sentences from that language's Wikipedia (or your own: `--texts sentences.txt`), gives each voice different sentences,
and keeps a clip only if the teacher finished it cleanly. Then bring `distilled/` here as layout A and run steps 4-7.
Licences: the teacher's weights are Apache-2.0, its codec (Kyutai Mimi) is CC-BY-4.0, Wikipedia text is CC BY-SA 4.0.
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}},
      "nbformat": 4, "nbformat_minor": 5}
out = "test_notebook.ipynb" if TEST else "finetune_preset_voices.ipynb"
json.dump(nb, open(out, "w"), ensure_ascii=False, indent=1)
print("ok", out, len(cells), "cells")
