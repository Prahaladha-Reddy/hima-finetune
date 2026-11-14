"""Gather voices from any mix of sources into one training set (metadata.csv), ready for `prepare`.

A source is any of:
    hf:OWNER/NAME           a Hugging Face dataset in the guide's layout, e.g. hf:Prahaladha/hima-finetune-demo-voices/te
    hf:OWNER/NAME/SUBDIR    one folder inside it
    https://.../voices.zip  a zip or tar.gz to download and unpack
    /path/to/folder         a local folder (or a local .zip)

and each source is either
    A. clips + text:     metadata.csv (path|text|voice) next to the clips, or
    B. recordings only:  one sub-folder of recordings per voice; these are cut at the pauses and written down by Hima-ASR
                         (needs: pip install "hima-finetune[transcribe]")

    hima-finetune get --source hf:Prahaladha/hima-finetune-demo-voices/te --out data
    hima-finetune get --source hf:Prahaladha/hima-finetune-demo-voices/te --source /kaggle/input/my-voices \\
                      --voices Anjali,Meera --rename Anjali=Asha --lang te --out data
Writes OUT/metadata.csv (absolute clip paths) and OUT/voices.json (clips and minutes per voice, and where each came from).
"""
import argparse
import csv
import json
import os
import sys
import tarfile
import urllib.request
import zipfile

AUDIO_EXT = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".webm", ".mp4")


def fetch(src, cache):
    """A source string -> a local folder."""
    if src.startswith("hf:"):
        from huggingface_hub import snapshot_download
        parts = src[3:].strip("/").split("/")
        repo, sub = "/".join(parts[:2]), "/".join(parts[2:])
        root = snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{sub}/**"] if sub else None)
        return os.path.join(root, sub) if sub else root
    if src.startswith(("http://", "https://")):
        name = os.path.basename(src.split("?")[0]) or "download"
        dst = os.path.join(cache, name)
        if not os.path.exists(dst):
            print(f"downloading {src}", flush=True)
            urllib.request.urlretrieve(src, dst)
        src = dst
    if os.path.isfile(src) and (src.endswith(".zip") or ".tar" in src or src.endswith(".tgz")):
        out = os.path.join(cache, os.path.basename(src).split(".")[0])
        if not os.path.isdir(out):
            (zipfile.ZipFile(src).extractall if src.endswith(".zip") else tarfile.open(src).extractall)(out)
        src = out
    if not os.path.isdir(src):
        sys.exit(f"source not found: {src}")
    return src


def find_metadata(d):
    """metadata.csv in d, or in its only sub-folder (zips often add one level)."""
    for base in [d] + [os.path.join(d, s) for s in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, s))]:
        p = os.path.join(base, "metadata.csv")
        if os.path.exists(p):
            return p
    return None


def read_metadata(p):
    base, rows = os.path.dirname(p), []
    with open(p, encoding="utf-8") as f:
        for n, parts in enumerate(csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)):
            if len(parts) < 3 or (n == 0 and parts[0].strip().lower() == "path"):
                continue
            path, text, voice = parts[0].strip(), "|".join(parts[1:-1]).strip(), parts[-1].strip()
            rows.append((path if os.path.isabs(path) else os.path.join(base, path), text, voice))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hima-finetune get", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True, help="repeat for several sources (see above)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voices", default="", help="keep only these voices (comma separated; default: all)")
    ap.add_argument("--rename", default="", help="Old=New,Old2=New2: give voices new names")
    ap.add_argument("--lang", default=None, choices=["te", "hi"], help="needed only for recordings without text")
    a = ap.parse_args(argv)
    import soundfile as sf

    keep = {v.strip() for v in a.voices.split(",") if v.strip()}
    rename = dict(kv.split("=", 1) for kv in a.rename.split(",") if "=" in kv)
    cache = os.path.join(a.out, "_sources"); os.makedirs(cache, exist_ok=True)
    rows, origin = [], {}
    for i, src in enumerate(a.source):
        d = fetch(src, cache)
        meta = find_metadata(d)
        if meta is None:                                                   # layout B: recordings only
            voices = [v for v in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, v)) and not v.startswith((".", "_"))]
            voices = [v for v in voices if not keep or v in keep or rename.get(v) in keep]
            if not voices:
                sys.exit(f"{src}: no metadata.csv and no voice folders with recordings")
            if a.lang is None:
                sys.exit(f"{src} has recordings without text: pass --lang so they can be transcribed")
            try:
                import transformers
                assert tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (5, 13)
            except (ImportError, AssertionError, ValueError):
                sys.exit(f"{src} has recordings without text. Writing them down needs Hima-ASR's packages: "
                         f"pip install \"hima-finetune[transcribe]\" (or give the clips' text in a metadata.csv)")
            from . import transcribe
            out = os.path.join(a.out, f"_transcribed_{i}")
            for v in voices:
                transcribe.main(["--audio", os.path.join(d, v), "--voice", v, "--out", os.path.join(out, v), "--lang", a.lang])
                rows += [(p, t, vv, src) for p, t, vv in read_metadata(os.path.join(out, v, "metadata.csv"))]
        else:
            got = read_metadata(meta)
            print(f"{src}: {len(got)} clips with text", flush=True)
            rows += [(p, t, v, src) for p, t, v in got]

    out_rows, stats = [], {}
    for p, t, v, src in rows:
        if keep and v not in keep and rename.get(v) not in keep:
            continue
        v = rename.get(v, v)
        if not os.path.exists(p):
            print(f"  missing file, skipped: {p}", flush=True); continue
        s = stats.setdefault(v, {"clips": 0, "minutes": 0.0, "sources": []})
        if src not in s["sources"]:
            s["sources"].append(src)
        s["clips"] += 1; s["minutes"] += sf.info(p).duration / 60
        out_rows.append((p, t, v))
    if not out_rows:
        sys.exit("no clips left: check --voices (names are case-sensitive)")
    with open(os.path.join(a.out, "metadata.csv"), "w", encoding="utf-8") as f:
        f.write("path|text|voice\n")
        for p, t, v in out_rows:
            f.write(f"{os.path.abspath(p)}|{t.replace('|', ' ')}|{v}\n")
    for v, s in stats.items():
        s["minutes"] = round(s["minutes"], 1)
    json.dump(stats, open(os.path.join(a.out, "voices.json"), "w"), ensure_ascii=False, indent=1)
    print("\nvoice            clips   minutes  from")
    for v, s in sorted(stats.items()):
        warn = "   <- under 20 min: will come out weaker" if s["minutes"] < 20 else ""
        print(f"{v:16s} {s['clips']:5d}   {s['minutes']:7.1f}  {', '.join(s['sources'])}{warn}")
    print(f"GET_DONE {len(out_rows)} clips, {len(stats)} voices -> {os.path.join(a.out, 'metadata.csv')}", flush=True)


if __name__ == "__main__":
    main()
