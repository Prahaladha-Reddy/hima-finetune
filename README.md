# hima-finetune

Train your own built-in voices for **[Hima TTS nano](https://huggingface.co/Prahaladha/hima-tts-nano-te)** (62M parameters,
Telugu and Hindi, runs on a laptop CPU and on phones) on **one free T4 GPU**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prahaladha-Reddy/hima-finetune/blob/main/finetune_preset_voices.ipynb)

You train a small adapter (about 5M numbers) on the frozen base model, the same way the released voices were made. The output
is a normal model folder: `TTS("my-model").say("...", voice="Meera")`.

## Start from what you have

| You have | Command | GPU |
|---|---|---|
| Recordings + their text | straight to `prepare` | T4 |
| Recordings only, any length or format | `hima-finetune transcribe`: cuts them into sentences at the pauses, writes them down with [Hima-ASR](https://huggingface.co/Prahaladha/hima-asr) | T4 |
| No recordings | `hima-finetune distill`: voices read by [Maya 2 Native](https://huggingface.co/maya-research/maya2-native) (3B, Apache-2.0) | L4 / A100 class |

Then:

```bash
pip install "hima-finetune @ git+https://github.com/Prahaladha-Reddy/hima-finetune"
hf download Prahaladha/hima-tts-nano-te --local-dir hima-tts-nano-te

hima-finetune prepare --base hima-tts-nano-te --data my-voices --work work    # clips -> training data
hima-finetune train   --base hima-tts-nano-te --work work                     # the adapter; add --resume after a disconnect
hima-finetune export  --base hima-tts-nano-te --work work --out my-model      # a complete model folder + samples
```

The easiest path is the notebook, [`finetune_preset_voices.ipynb`](finetune_preset_voices.ipynb), which runs unchanged on Kaggle and
Colab.

## How long, on a free T4

| Voices | Prepare | Train | GPU memory |
|---|---|---|---|
| 2 voices x 40 min | 2 min | 9 min | 10.8 GB |
| 6 voices x 2 h | 7 min | 90 min | 10.6 GB |

**Does it match the released voices?** We retrained the 6 released Telugu voices from their own recordings with this repo, on one free
Kaggle T4, and scored both against each speaker's held-out real recordings (speaker similarity, same voice settings, 8 new sentences
per voice):

| | Speaker similarity |
|---|---|
| Base model, no adapter | 0.572 |
| Released voices | 0.609 |
| **Retrained on a free T4** | **0.622** |
| Real recordings vs each other (ceiling) | 0.808 |

## The steps

| Step | Input | Output |
|---|---|---|
| `transcribe` | `recordings/<Voice>/*` (any format, any length) | 3-15 s clips cut at pauses, `metadata.csv`, `transcribe_report.json`. Drops silent clips and clips whose text does not fit the audio (real Telugu speech is 8-23 letters per second; outside 4-30 the recognizer has invented or missed words). |
| `distill` | teacher voice names, a language, minutes per voice | clips + `metadata.csv`, read from Wikipedia sentences (or `--texts your.txt`). Keeps a clip only if the teacher stopped on its own and its length fits the text. |
| `prepare` | `metadata.csv` (`path\|text\|voice`) + clips | codec latents, a voice print per clip, ~3% of each voice held out. Prints minutes per voice and every skipped clip with the reason. |
| `train` | the prepared data | LoRA rank 32 on every linear layer of the backbone, flow head and stop head (99 layers); base frozen. bf16 on Ampere+, fp16 with gradient scaling on T4. Steps default to ~60 passes over your data. Saves every 500 steps with the full optimizer state, so `--resume` continues exactly. |
| `export` | the adapter | `voices/adapter.safetensors`, `voices/voices.safetensors` (each voice's mean voice print and ~5 s of its own audio), `voices/voices.json`, next to the base model files, plus a sample per voice. |

Every step has `-h`.

## Good data

- 30-45 minutes per voice. Under 20 minutes works, but the voice comes out weaker.
- One speaker per voice, clean audio: no music, no other voices, little echo.
- The text must match the audio word for word. English words in English letters are fine.

## Please

Only train on voices you have the right to use, with the speaker's consent.

## Licence

Code: MIT (`LICENSE`; adapted parts: `LICENSE-THIRD-PARTY`). Hima TTS, Hima-ASR and Maya 2 Native have their own licences: see their
model cards. Distilled data carries the licences of its sources: Maya 2 Native weights Apache-2.0, its codec (Kyutai Mimi)
CC-BY-4.0, Wikipedia text CC BY-SA 4.0.
