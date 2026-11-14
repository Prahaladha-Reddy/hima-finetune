"""No recordings at all? Distill voices from a big open model instead.

Maya 2 Native (https://huggingface.co/maya-research/maya2-native, 3B, Apache-2.0) speaks 12 Indian languages in 20 built-in
voices. This step makes it read a few thousand sentences in the voices you pick and writes a training set (clips +
metadata.csv) for `prepare`. The 62M Hima model then learns those voices: a model ~50x smaller that runs on a phone.

**Needs a bigger GPU than a T4.** The teacher runs a bf16 backbone with fused kernels: an Ampere-or-newer NVIDIA GPU
(L4, A10G, A100, H100; 12 GB+), Linux, CUDA 12.8, and its own environment (it pins its own torch / transformers):

    pip install "maya3 @ git+https://github.com/MayaResearch/maya2-native" --extra-index-url https://download.pytorch.org/whl/cu128
    pip install --no-deps "hima-finetune @ git+https://github.com/Prahaladha-Reddy/hima-finetune"
    hima-finetune distill --voices Vaidehi,Sreekar --lang te --minutes 45 --out distilled

Then copy `distilled/` to the T4 and continue with `prepare` and `train` as usual. Texts: `--texts my_sentences.txt` (one
sentence per line, written as it should be spoken), or the default `wikipedia`, which takes sentences from that language's
Wikipedia (CC BY-SA 4.0). Each voice reads different sentences. Every clip is kept only if the teacher stopped on its own
(no cut-off, no loop) and its length fits the text.
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SCRIPT = {"te": (0x0C00, 0x0C7F), "hi": (0x0900, 0x097F)}
SR = 24000


def wiki_sentences(lang, need, seed):
    """Sentences from <lang> Wikipedia: 25-160 characters, a full stop at the end, in the language's own script, no digits
    (digits are read differently by different text normalisers, so they make the text and the audio disagree)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    lo, hi = SCRIPT[lang]
    out, seen, ends = [], set(), {}
    for shard in range(2):
        p = hf_hub_download("wikimedia/wikipedia", f"20231101.{lang}/train-{shard:05d}-of-00002.parquet", repo_type="dataset")
        for text in pq.read_table(p, columns=["text"]).column("text").to_pylist():
            for s in re.split(r"(?<=[.?!।])\s+", text):
                s = " ".join(s.split())
                letters = [c for c in s if c.isalpha()]
                if not (25 <= len(s) <= 160) or s[-1] not in ".?!।" or re.search(r"[0-9౦-౯०-९()\[\]{}=|/<>:;@#%&*_]", s):
                    continue
                if not letters or sum(lo <= ord(c) <= hi for c in letters) / len(letters) < 0.95 or s in seen or " - " in s or '"' in s:
                    continue
                if any(0x0900 <= ord(c) <= 0x0DFF and not lo <= ord(c) <= hi for c in s):   # another Indian script (or its digits)
                    continue
                tail = " ".join(s.split()[-3:])                         # Wikipedia repeats templates ("..., X mandal village.")
                if ends.get(tail, 0) >= 2:
                    continue
                ends[tail] = ends.get(tail, 0) + 1
                seen.add(s); out.append(s)
        if len(out) >= need * 3:
            break
    random.Random(seed).shuffle(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hima-finetune distill", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voices", required=True, help="teacher voices, comma separated (e.g. Vaidehi,Sreekar); see the teacher's model card")
    ap.add_argument("--lang", required=True, choices=["te", "hi"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=45, help="audio to keep per voice (30-45 is the sweet spot)")
    ap.add_argument("--texts", default="wikipedia", help="'wikipedia' or a .txt file, one sentence per line")
    ap.add_argument("--teacher", default="maya-research/maya2-native")
    ap.add_argument("--seats", type=int, default=16, help="sentences generated at once (16 fits 24 GB; 8 for 12-16 GB)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    import soundfile as sf
    import torch
    assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8, \
        "distill needs an Ampere-or-newer GPU (L4, A10G, A100, H100): the teacher runs in bf16. Prepare and train still run on a T4."
    from maya3 import Maya3TTS

    voices = [v.strip() for v in a.voices.split(",") if v.strip()]
    per_voice = int(a.minutes * 60 / 5.0 * 1.4) + 50                    # ~5 s per sentence, 40% spare for dropped clips
    if a.texts == "wikipedia":
        texts = wiki_sentences(a.lang, per_voice * len(voices), a.seed)
        source = f"{a.lang}.wikipedia.org (CC BY-SA 4.0), via wikimedia/wikipedia 20231101.{a.lang}"
    else:
        texts = [l.strip() for l in open(a.texts, encoding="utf-8") if l.strip()]
        random.Random(a.seed).shuffle(texts); source = os.path.basename(a.texts)
    if len(texts) < per_voice * len(voices):
        print(f"note: {len(texts)} sentences for {len(voices)} voices; some voices will get less than {a.minutes} min", flush=True)
    print(f"loading {a.teacher} ({a.seats} at once) on {torch.cuda.get_device_name()}", flush=True)
    tts = Maya3TTS.from_pretrained(a.teacher, max_seats=a.seats)
    known = {v.lower() for v in tts.voices}
    bad = [v for v in voices if v.lower() not in known]
    assert not bad, f"unknown teacher voices {bad}; available: {sorted(tts.voices)}"

    t0, lock, rows, report = time.time(), threading.Lock(), [], {}
    for vi, v in enumerate(voices):
        os.makedirs(os.path.join(a.out, "clips", v), exist_ok=True)
        mine = texts[vi::len(voices)]                                     # each voice reads different sentences
        rep = dict(tried=0, kept=0, kept_min=0.0, dropped={"stopped early or looped": 0, "length does not fit the text": 0, "error": 0})
        state = {"sec": 0.0, "i": 0}

        def one(k):
            text = mine[k]
            with lock:
                if state["sec"] >= a.minutes * 60:
                    return
            try:
                wav, info = tts.generate(text, lang=a.lang, voice=v, seed=a.seed * 100003 + vi * 10007 + k, return_info=True)
            except Exception as e:
                with lock: rep["dropped"]["error"] += 1
                return
            stops = info.get("stops") or [{"reason": info.get("reason", "")}]
            dur = len(wav) / SR
            with lock:
                rep["tried"] += 1
                if not all(str(s.get("reason", "")).startswith("aeos") for s in stops):
                    rep["dropped"]["stopped early or looped"] += 1; return
                if not (4 <= len(text) / max(dur, 1e-3) <= 30):            # the same bounds as transcribe: real speech is 8-23 letters/s
                    rep["dropped"]["length does not fit the text"] += 1; return
                if state["sec"] >= a.minutes * 60:
                    return
                p = os.path.join("clips", v, f"{v}_{k:05d}.wav")
                sf.write(os.path.join(a.out, p), wav, SR, subtype="PCM_16")
                rows.append((p, text, v)); state["sec"] += dur; rep["kept"] += 1
                if rep["kept"] % 100 == 0:
                    print(f"  {v}: {state['sec'] / 60:.1f}/{a.minutes:.0f} min", flush=True)

        with ThreadPoolExecutor(a.seats) as ex:                            # concurrent requests are batched by the engine
            list(ex.map(one, range(len(mine))))
        rep["kept_min"] = round(state["sec"] / 60, 1)
        report[v] = rep
        print(f"{v}: kept {rep['kept']} clips, {rep['kept_min']} min (dropped: {rep['dropped']})", flush=True)
    tts.close()

    with open(os.path.join(a.out, "metadata.csv"), "w", encoding="utf-8") as f:
        f.write("path|text|voice\n")
        for p, t, v in rows:
            f.write(f"{p}|{t.replace('|', ' ')}|{v}\n")
    json.dump(dict(teacher=a.teacher, teacher_license="Apache-2.0 (codec: Kyutai Mimi, CC-BY-4.0)", texts=source, lang=a.lang,
                   minutes=round((time.time() - t0) / 60, 1), voices=report),
              open(os.path.join(a.out, "distill_report.json"), "w"), ensure_ascii=False, indent=1)
    print(f"DISTILL_DONE {len(rows)} clips -> {a.out}/metadata.csv in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
