"""Recordings only, no text? This turns raw recordings into a training set: it cuts them into sentence-sized clips at the
pauses, writes down what is said with Hima-ASR (https://huggingface.co/Prahaladha/hima-asr), drops what is not usable and
writes metadata.csv for `prepare`.

Input: one folder per voice (the folder name becomes the voice name), any audio format, any length:
    recordings/Meera/interview.mp3
    recordings/Meera/day2/*.wav
    recordings/Ravi/podcast.m4a

    hima-finetune transcribe --audio recordings --out my_voices --lang te
Writes my_voices/clips/*.wav, my_voices/metadata.csv (path|text|voice) and my_voices/transcribe_report.json.
Already have one voice in a flat folder? Add --voice Meera.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

AUDIO_EXT = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".webm", ".mp4")
SR_OUT, SR_ASR, HOP = 24000, 16000, 0.02     # clips are written at the model's 24 kHz; the recognizer listens at 16 kHz


def find_audio(root, voice=None):
    """{voice: [files]}: one sub-folder per voice, or the whole folder as one voice when --voice is given."""
    out = {}
    if voice:
        dirs = {voice: root}
    else:
        dirs = {d: os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")}
        if not dirs:
            sys.exit(f"no voice folders in {root}: put each voice's recordings in its own folder (recordings/Meera/...), or pass --voice NAME")
    for v, d in dirs.items():
        files = sorted(os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.lower().endswith(AUDIO_EXT))
        if files:
            out[v] = files
    return out


def load(path):
    import librosa
    x, _ = librosa.load(path, sr=SR_OUT, mono=True)
    return x.astype(np.float32)


def frame_db(x):
    n = int(SR_OUT * HOP)
    m = len(x) // n
    if m == 0:
        return np.full(1, -100.0)
    e = np.sqrt(np.mean(x[: m * n].reshape(m, n) ** 2, 1) + 1e-12)
    return 20 * np.log10(e)


def split(x, min_s=3.0, max_s=15.0, min_pause=0.25):
    """Cut at pauses into clips of min_s..max_s seconds. A pause = 35 dB below the recording's loud speech (its 95th
    percentile frame), so it adapts to each recording's level and noise. Returns [(start, end)] in samples."""
    db = frame_db(x)
    quiet = db < np.percentile(db, 95) - 35
    pauses, i = [], 0                                         # (middle frame, length in frames)
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) * HOP >= min_pause or i == 0 or j == len(quiet):
                pauses.append(((i + j) // 2, j - i))
            i = j
        else:
            i += 1
    voiced = np.flatnonzero(~quiet)
    if len(voiced) == 0:
        return []
    start, last, segs = voiced[0], voiced[-1], []
    f = lambda s: int(s / HOP)
    while last - start > f(1.0):
        if last - start <= f(max_s):
            segs.append((start, last + 1)); break
        best = [p for p in pauses if start + f(min_s) <= p[0] <= start + f(max_s)] or \
               [p for p in pauses if start + f(1.5) <= p[0] <= start + f(max_s)]
        cut = max(best, key=lambda p: p[1])[0] if best else start + f(max_s)    # the longest pause in range
        segs.append((start, cut))
        nxt = voiced[voiced > cut]
        if len(nxt) == 0:
            break
        start = nxt[0]
    out = []
    pad = int(0.15 / HOP)
    for a, b in segs:                                          # trim the silence at both ends, keep 150 ms of air
        v = np.flatnonzero(~quiet[a:b])
        if len(v) == 0:
            continue
        a2, b2 = max(0, a + v[0] - pad), min(len(quiet), a + v[-1] + 1 + pad)
        out.append((a2 * int(SR_OUT * HOP), min(len(x), b2 * int(SR_OUT * HOP))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hima-finetune transcribe", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="folder with one sub-folder of recordings per voice")
    ap.add_argument("--out", required=True, help="where clips/, metadata.csv and the report go")
    ap.add_argument("--lang", required=True, choices=["te", "hi"], help="language spoken in the recordings")
    ap.add_argument("--voice", default=None, help="treat the whole --audio folder as this one voice")
    ap.add_argument("--min_sec", type=float, default=3.0); ap.add_argument("--max_sec", type=float, default=15.0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--asr", default="Prahaladha/hima-asr", help="speech recognizer (Hugging Face repo)")
    a = ap.parse_args(argv)
    import soundfile as sf
    import torch
    from transformers import AutoModel

    t0 = time.time()
    voices = find_audio(a.audio, a.voice)
    if not voices:
        sys.exit(f"no audio files ({', '.join(AUDIO_EXT)}) under {a.audio}")
    os.makedirs(os.path.join(a.out, "clips"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {a.asr} on {device}", flush=True)
    asr = AutoModel.from_pretrained(a.asr, trust_remote_code=True, dtype=torch.float32).to(device).eval()

    rows, report = [], {}
    for v, files in voices.items():
        rep = dict(files=len(files), unreadable=0, input_min=0.0, clips=0, kept_min=0.0, dropped={"no speech": 0, "too short": 0, "no text": 0, "text too long for the audio": 0, "text too short for the audio": 0})
        clips = []                                             # (name, 24 kHz audio)
        for fp in files:
            try:
                x = load(fp)
            except Exception as e:                             # one bad file must not stop the run
                print(f"  skip {fp}: {e}", flush=True); rep["unreadable"] += 1; continue
            rep["input_min"] += len(x) / SR_OUT / 60
            peak = np.abs(x).max()
            if peak < 1e-4:
                rep["dropped"]["no speech"] += 1; continue
            x = x * min(1.0, 0.95 / peak)                         # leave level alone unless it clips
            segs = split(x, a.min_sec, a.max_sec)
            if not segs:
                rep["dropped"]["no speech"] += 1
            stem = os.path.splitext(os.path.relpath(fp, a.audio))[0].replace(os.sep, "_").replace(" ", "_")
            for k, (s, e) in enumerate(segs):
                if (e - s) / SR_OUT < 1.0:
                    rep["dropped"]["too short"] += 1; continue
                clips.append((f"{stem}_{k:04d}", x[s:e]))
        import librosa
        for i in range(0, len(clips), a.batch_size):
            part = clips[i:i + a.batch_size]
            texts = asr.transcribe([librosa.resample(c, orig_sr=SR_OUT, target_sr=SR_ASR) for _, c in part],
                                   language=a.lang, style="mixed", batch_size=a.batch_size, sampling_rate=SR_ASR)
            for (name, c), t in zip(part, texts):
                t, dur = " ".join(t.split()), len(c) / SR_OUT
                if len(t) < 2:
                    rep["dropped"]["no text"] += 1; continue
                # real Telugu speech: 8.4-22.7 letters per second (1st-100th percentile of 1,500 clips), median 13.8.
                # Outside 4-30 the recognizer has invented words or missed most of them.
                if len(t) / dur > 30:
                    rep["dropped"]["text too long for the audio"] += 1; continue
                if len(t) / dur < 4:
                    rep["dropped"]["text too short for the audio"] += 1; continue
                p = os.path.join("clips", v, name + ".wav")
                os.makedirs(os.path.join(a.out, "clips", v), exist_ok=True)
                sf.write(os.path.join(a.out, p), c, SR_OUT, subtype="PCM_16")
                rows.append((p, t, v)); rep["clips"] += 1; rep["kept_min"] += dur / 60
            print(f"  {v}: {min(i + a.batch_size, len(clips))}/{len(clips)} clips transcribed", flush=True)
        rep = {k: (round(x, 1) if isinstance(x, float) else x) for k, x in rep.items()}
        report[v] = rep
        warn = "  <- under 20 min: the voice will come out weaker" if rep["kept_min"] < 20 else ""
        print(f"{v}: {rep['input_min']:.1f} min of recordings -> {rep['clips']} clips, {rep['kept_min']:.1f} min kept{warn}", flush=True)

    with open(os.path.join(a.out, "metadata.csv"), "w", encoding="utf-8") as f:
        f.write("path|text|voice\n")
        for p, t, v in rows:
            f.write(f"{p}|{t.replace('|', ' ')}|{v}\n")
    json.dump(dict(asr=a.asr, lang=a.lang, minutes=round((time.time() - t0) / 60, 1), voices=report),
              open(os.path.join(a.out, "transcribe_report.json"), "w"), ensure_ascii=False, indent=1)
    print(f"TRANSCRIBE_DONE {len(rows)} clips -> {os.path.join(a.out, 'metadata.csv')} in {(time.time() - t0) / 60:.1f} min. "
          f"Read a few lines of it: the text should match the audio word for word.", flush=True)


if __name__ == "__main__":
    main()
